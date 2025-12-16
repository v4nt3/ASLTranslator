"""
Pipeline NUEVO: Sin creación de clips
Trabaja directamente con secuencias completas de longitud variable

FLUJO:
1. Video → Frames (TODOS o subsampling inteligente)
2. Frames → Keypoints (TODOS)
3. Detectar región de la seña (ActionSegmenter)
4. Extraer features visuales y pose de la región completa
5. Entrenar LSTM con secuencias de longitud variable
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import pandas as pd
import json

from config import config
from pipelines.data_preparation1 import (
    FrameExtractor, 
    KeypointExtractor, 
    ActionSegmenter
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceProcessor:
    """
    Procesa videos completos sin crear clips
    Mantiene la secuencia temporal completa
    """
    
    def __init__(self, config, max_frames: int = 120):
        self.config = config
        self.max_frames = max_frames  # Límite para evitar OOM
        
        self.frame_extractor = FrameExtractor(config)
        self.keypoint_extractor = KeypointExtractor(config)
        self.action_segmenter = ActionSegmenter(config, debug=True)
        
        logger.info("="*60)
        logger.info("SequenceProcessor: Sin clips, secuencias completas")
        logger.info(f"  Max frames por secuencia: {max_frames}")
        logger.info("="*60)
    
    def process_single_video(
        self, 
        video_path: Path,
        class_id: int
    ) -> Optional[Dict]:
        """
        Procesa UN video completo
        
        Returns:
            {
                'video_name': str,
                'class_id': int,
                'frames': np.ndarray,  # (T, H, W, C)
                'keypoints': np.ndarray,  # (T, 75, 4)
                'sequence_length': int,
                'segmented': bool,
                'segment_bounds': (start, end)
            }
        """
        video_name = video_path.stem
        
        try:
            # 1. Extraer frames
            frames = self.frame_extractor.extract_frames(
                video_path, 
                self.config.data_paths.extracted_frames
            )
            
            if len(frames) == 0:
                logger.warning(f"[{video_name}] No frames extraídos")
                return None
            
            frames = np.array(frames)
            
            # 2. Extraer keypoints
            keypoints_sequence = []
            for frame in frames:
                pose_kpts = self.keypoint_extractor.extract_pose_keypoints(frame)
                
                if pose_kpts is not None:
                    pose_kpts = self.keypoint_extractor.normalize_keypoints(pose_kpts)
                    hand_kpts = self.keypoint_extractor.extract_hand_keypoints(frame)
                    
                    if hand_kpts is not None:
                        combined_kpts = np.concatenate([pose_kpts, hand_kpts], axis=0)
                    else:
                        hand_placeholder = np.zeros((42, 4))
                        combined_kpts = np.concatenate([pose_kpts, hand_placeholder], axis=0)
                    
                    keypoints_sequence.append(combined_kpts)
            
            if len(keypoints_sequence) == 0:
                logger.warning(f"[{video_name}] No keypoints extraídos")
                return None
            
            keypoints = np.array(keypoints_sequence)
            
            # 3. Segmentar acción (encontrar región relevante)
            try:
                seg_start, seg_end = self.action_segmenter.segment_action(keypoints)
                
                # Validar segmento
                segment_length = seg_end - seg_start
                total_length = len(keypoints)
                
                # Si segmento muy corto, expandir
                if segment_length < total_length * 0.3:
                    expand = int((total_length * 0.5 - segment_length) / 2)
                    seg_start = max(0, seg_start - expand)
                    seg_end = min(total_length, seg_end + expand)
                    logger.debug(f"[{video_name}] Segmento expandido: [{seg_start}, {seg_end}]")
                
                # Recortar a región segmentada
                frames = frames[seg_start:seg_end]
                keypoints = keypoints[seg_start:seg_end]
                segmented = True
                
            except Exception as e:
                logger.warning(f"[{video_name}] Error en segmentación: {e}, usando completo")
                seg_start, seg_end = 0, len(keypoints)
                segmented = False
            
            # 4. Submuestrear si es muy largo
            sequence_length = len(frames)
            
            if sequence_length > self.max_frames:
                # Submuestrear uniformemente
                indices = np.linspace(0, sequence_length - 1, self.max_frames, dtype=int)
                frames = frames[indices]
                keypoints = keypoints[indices]
                sequence_length = self.max_frames
                logger.debug(f"[{video_name}] Submuestreado: {sequence_length} → {self.max_frames}")
            
            # 5. Retornar datos
            return {
                'video_name': video_name,
                'class_id': class_id,
                'frames': frames.astype(np.uint8),
                'keypoints': keypoints.astype(np.float32),
                'sequence_length': sequence_length,
                'segmented': segmented,
                'segment_bounds': (seg_start, seg_end)
            }
            
        except Exception as e:
            logger.error(f"[{video_name}] Error procesando: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def process_all_videos(self) -> List[Dict]:
        """
        Procesa todos los videos del dataset
        
        Returns:
            Lista de diccionarios con información de cada secuencia
        """
        video_dir = self.config.data_paths.raw_videos
        metadata_path = video_dir / "dataset_meta.json"
        
        # Cargar metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extraer mapeo video → class_id
        video_to_class = {}
        
        if isinstance(metadata, dict):
            if 'videos' in metadata:
                for entry in metadata['videos']:
                    video_file = entry.get('video_file', '')
                    class_id = entry.get('class_id')
                    if video_file and class_id is not None:
                        video_to_class[video_file] = int(class_id)
        
        logger.info(f"Metadata cargada: {len(video_to_class)} videos")
        
        # Buscar videos
        video_files = list(video_dir.glob("**/*.mp4")) + list(video_dir.glob("**/*.avi"))
        logger.info(f"Videos encontrados: {len(video_files)}")
        
        # Procesar cada video
        sequences = []
        processed = 0
        skipped = 0
        
        for video_path in tqdm(video_files, desc="Procesando videos"):
            # Buscar class_id
            video_file = video_path.name
            class_id = video_to_class.get(video_file)
            
            if class_id is None:
                # Intentar match parcial
                for key, cid in video_to_class.items():
                    if video_path.stem in key or key in video_path.stem:
                        class_id = cid
                        break
            
            if class_id is None:
                logger.warning(f"No class_id para: {video_file}")
                skipped += 1
                continue
            
            # Procesar
            result = self.process_single_video(video_path, class_id)
            
            if result is not None:
                sequences.append(result)
                processed += 1
        
        logger.info("="*60)
        logger.info(f"Videos procesados: {processed}")
        logger.info(f"⊘ Videos saltados: {skipped}")
        logger.info("="*60)
        
        return sequences
    
    def save_sequences(self, sequences: List[Dict], output_dir: Path):
        """
        Guarda secuencias procesadas
        
        Formato:
        - sequences/
          - {video_name}_frames.npy  (T, H, W, C)
          - {video_name}_keypoints.npy  (T, 75, 4)
        - sequences_metadata.json
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_list = []
        
        for seq in tqdm(sequences, desc="Guardando secuencias"):
            video_name = seq['video_name']
            
            # Guardar frames
            frames_path = output_dir / f"{video_name}_frames.npy"
            np.save(frames_path, seq['frames'])
            
            # Guardar keypoints
            keypoints_path = output_dir / f"{video_name}_keypoints.npy"
            np.save(keypoints_path, seq['keypoints'])
            
            # Metadata
            metadata_list.append({
                'video_name': video_name,
                'class_id': seq['class_id'],
                'sequence_length': seq['sequence_length'],
                'frames_path': str(frames_path.name),
                'keypoints_path': str(keypoints_path.name),
                'segmented': seq['segmented'],
                'segment_bounds': seq['segment_bounds']
            })
        
        # Guardar metadata
        metadata_path = output_dir / "sequences_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        logger.info(f"Secuencias guardadas en: {output_dir}")
        logger.info(f"   Metadata: {metadata_path}")
        
        # Estadísticas
        lengths = [s['sequence_length'] for s in sequences]
        logger.info(f"\nEstadísticas de longitud:")
        logger.info(f"  Min: {min(lengths)}")
        logger.info(f"  Max: {max(lengths)}")
        logger.info(f"  Mean: {np.mean(lengths):.1f}")
        logger.info(f"  Median: {np.median(lengths):.1f}")


class FeatureExtractorVariableLength:
    """
    Extrae features de secuencias de longitud variable
    """
    
    def __init__(
        self, 
        sequences_dir: Path,
        visual_extractor_path: Path,
        pose_extractor_path: Path,
        device: str = "cuda"
    ):
        self.sequences_dir = sequences_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Cargar extractores
        self.visual_extractor = torch.load(
            visual_extractor_path, 
            map_location=self.device,
            weights_only=False
        )
        self.visual_extractor.eval()
        
        self.pose_extractor = torch.load(
            pose_extractor_path,
            map_location=self.device,
            weights_only=False
        )
        self.pose_extractor.eval()
        
        # Transform
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Feature extractors cargados")
    
    @torch.no_grad()
    def extract_features_from_sequence(
        self,
        frames: np.ndarray,
        keypoints: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extrae features de una secuencia completa
        
        Args:
            frames: (T, H, W, C)
            keypoints: (T, 75, 4)
            batch_size: Batch size para procesamiento
        
        Returns:
            fused_features: (T, 1152) = visual(1024) + pose(128)
        """
        T = len(frames)
        
        # Extraer visual features
        all_visual = []
        for i in range(0, T, batch_size):
            batch_frames = frames[i:i+batch_size]
            
            # Convertir a tensor
            batch_tensors = []
            for frame in batch_frames:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                frame_tensor = self.transform(frame)
                batch_tensors.append(frame_tensor)
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Extraer
            visual_feat = self.visual_extractor(batch_tensor)
            all_visual.append(visual_feat.cpu().numpy())
        
        visual_features = np.concatenate(all_visual, axis=0)  # (T, 1024)
        
        # Extraer pose features
        all_pose = []
        for i in range(0, T, batch_size):
            batch_keypoints = keypoints[i:i+batch_size]
            
            # Flatten
            batch_flat = batch_keypoints.reshape(len(batch_keypoints), -1)
            batch_tensor = torch.from_numpy(batch_flat).float().to(self.device)
            
            # Extraer
            pose_feat = self.pose_extractor(batch_tensor)
            all_pose.append(pose_feat.cpu().numpy())
        
        pose_features = np.concatenate(all_pose, axis=0)  # (T, 128)
        
        # Fusionar
        fused_features = np.concatenate([visual_features, pose_features], axis=1)  # (T, 1152)
        
        return fused_features.astype(np.float16)
    
    def process_all_sequences(self, output_dir: Path):
        """
        Procesa todas las secuencias y guarda features
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar metadata
        metadata_path = self.sequences_dir / "sequences_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Procesando {len(metadata)} secuencias...")
        
        processed = 0
        for entry in tqdm(metadata, desc="Extrayendo features"):
            video_name = entry['video_name']
            
            # Paths
            frames_path = self.sequences_dir / entry['frames_path']
            keypoints_path = self.sequences_dir / entry['keypoints_path']
            output_path = output_dir / f"{video_name}_fused.npy"
            
            # Skip si ya existe
            if output_path.exists():
                continue
            
            try:
                # Cargar
                frames = np.load(frames_path)
                keypoints = np.load(keypoints_path)
                
                # Extraer features
                fused_features = self.extract_features_from_sequence(frames, keypoints)
                
                # Guardar
                np.save(output_path, fused_features)
                
                processed += 1
                
            except Exception as e:
                logger.error(f"Error en {video_name}: {e}")
        
        logger.info(f"Features extraídas: {processed}")


def main():
    """Pipeline completo sin clips"""
    
    # 1. Procesar videos a secuencias
    logger.info("\n" + "="*60)
    logger.info("PASO 1: Procesar videos a secuencias")
    logger.info("="*60)
    
    processor = SequenceProcessor(config, max_frames=120)
    sequences = processor.process_all_videos()
    
    sequences_dir = Path("data/sequences")
    processor.save_sequences(sequences, sequences_dir)
    
    # 2. Extraer features
    logger.info("\n" + "="*60)
    logger.info("PASO 2: Extraer features de secuencias")
    logger.info("="*60)
    
    feature_extractor = FeatureExtractorVariableLength(
        sequences_dir=sequences_dir,
        visual_extractor_path=Path("models/extractors/visual_extractor_full.pt"),
        pose_extractor_path=Path("models/extractors/pose_extractor_full.pt"),
        device="cuda"
    )
    
    features_dir = Path("data/sequence_features")
    feature_extractor.process_all_sequences(features_dir)
    
    logger.info("\n" + "="*60)
    logger.info("Pipeline completado!")
    logger.info(f"   Secuencias: {sequences_dir}")
    logger.info(f"   Features: {features_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()