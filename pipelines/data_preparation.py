"""
Pipeline HÍBRIDO: Reutiliza clases originales + Optimizaciones paralelas

MANTIENE:
- FrameExtractor (para compatibilidad)
- KeypointExtractor (para compatibilidad)
- ActionSegmenter (para segmentación automática)

AGREGA:
- Multiprocessing para velocidad
- GPU batching
- I/O paralelo
"""

import numpy as np #type: ignore
import torch #type: ignore
import torch.multiprocessing as mp #type: ignore
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm #type: ignore
import json
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from config import config
from pipelines.data_preparation1 import (
    FrameExtractor,
    KeypointExtractor, 
    ActionSegmenter
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# WRAPPER FUNCTIONS para Multiprocessing
# ============================================================

def process_video_frames_wrapper(args: Tuple[Path, dict]) -> Tuple[str, Optional[np.ndarray]]:
    """
    Wrapper que usa FrameExtractor original pero es serializable
    """
    video_path, config_dict = args
    video_name = video_path.stem
    
    try:
        # Recrear config en cada worker (evita problemas de serialización)
        from types import SimpleNamespace
        temp_config = SimpleNamespace(**config_dict)
        
        # Usar FrameExtractor ORIGINAL
        extractor = FrameExtractor(temp_config)
        frames = extractor.extract_frames(video_path, temp_config.data_paths.extracted_frames)
        
        if len(frames) == 0:
            return video_name, None
        
        return video_name, np.array(frames, dtype=np.uint8)
        
    except Exception as e:
        logger.error(f"Error en {video_name}: {e}")
        return video_name, None


def process_video_keypoints_wrapper(args: Tuple[str, np.ndarray, dict]) -> Tuple[str, Optional[np.ndarray]]:
    """
    Wrapper que usa KeypointExtractor original pero es serializable
    """
    video_name, frames, config_dict = args
    
    try:
        from types import SimpleNamespace
        temp_config = SimpleNamespace(**config_dict)
        
        # Usar KeypointExtractor ORIGINAL
        extractor = KeypointExtractor(temp_config)
        
        keypoints_sequence = []
        for frame in frames:
            pose_kpts = extractor.extract_pose_keypoints(frame)
            
            if pose_kpts is not None:
                pose_kpts = extractor.normalize_keypoints(pose_kpts)
                hand_kpts = extractor.extract_hand_keypoints(frame)
                
                if hand_kpts is not None:
                    combined_kpts = np.concatenate([pose_kpts, hand_kpts], axis=0)
                else:
                    hand_placeholder = np.zeros((42, 4))
                    combined_kpts = np.concatenate([pose_kpts, hand_placeholder], axis=0)
                
                keypoints_sequence.append(combined_kpts)
        
        if len(keypoints_sequence) == 0:
            return video_name, None
        
        return video_name, np.array(keypoints_sequence, dtype=np.float32)
        
    except Exception as e:
        logger.error(f"Error en {video_name}: {e}")
        return video_name, None


# ============================================================
# PROCESADOR HÍBRIDO OPTIMIZADO
# ============================================================

class HybridSequenceProcessor:
    """
    Procesador que REUTILIZA clases originales pero AGREGA paralelización
    """
    
    def __init__(
        self, 
        config, 
        use_action_segmenter: bool = True,
        num_workers: int = 14
    ):
        self.config = config
        self.use_action_segmenter = use_action_segmenter
        self.num_workers = num_workers
        
        # Inicializar ActionSegmenter (si se usa)
        if use_action_segmenter:
            self.action_segmenter = ActionSegmenter(config, debug=False)
        
        logger.info("="*60)
        logger.info("HybridSequenceProcessor")
        logger.info(f"  Workers: {num_workers}")
        logger.info(f"  ActionSegmenter: {use_action_segmenter}")
        logger.info("="*60)
    
    def _config_to_dict(self) -> dict:
        """
        Convierte config a dict serializable
        """
        return {
            'data': {
                'fps': self.config.data.fps,
                'frame_height': self.config.data.frame_height,
                'frame_width': self.config.data.frame_width,
                'num_frames_per_clip': self.config.data.num_frames_per_clip
            },
            'data_paths': {
                'extracted_frames': self.config.data_paths.extracted_frames,
                'keypoints': self.config.data_paths.keypoints
            }
        }
    
    def process_all_videos_parallel(self) -> List[Dict]:
        """
        Procesa videos usando clases originales pero en paralelo
        """
        video_dir = self.config.data_paths.raw_videos
        metadata_path = video_dir / "dataset_meta.json"
        
        # Cargar metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        video_to_class = {}
        if isinstance(metadata, dict):
            if 'videos' in metadata:
                for entry in metadata['videos']:
                    video_file = entry.get('video_file', '')
                    class_id = entry.get('class_id')
                    if video_file and class_id is not None:
                        video_to_class[video_file] = int(class_id)
        
        # Buscar videos
        video_files = list(video_dir.glob("**/*.mp4")) + list(video_dir.glob("**/*.avi"))
        logger.info(f"Videos encontrados: {len(video_files)}")
        
        # ========================================
        # PASO 1: Frames (paralelo)
        # ========================================
        logger.info("\n[1/3] Extrayendo frames (FrameExtractor + Multiprocessing)...")
        start_time = time.time()
        
        config_dict = self._config_to_dict()
        args_list = [(video_path, config_dict) for video_path in video_files]
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(tqdm(
                executor.map(process_video_frames_wrapper, args_list),
                total=len(args_list),
                desc="Frames"
            ))
        
        frames_data = {}
        for video_name, frames in results:
            if frames is not None:
                frames_data[video_name] = frames
        
        logger.info(f" Frames: {len(frames_data)} videos en {time.time()-start_time:.1f}s")
        
        # ========================================
        # PASO 2: Keypoints (paralelo)
        # ========================================
        logger.info("\n[2/3] Extrayendo keypoints (KeypointExtractor + Multiprocessing)...")
        start_time = time.time()
        
        keypoints_args = [
            (name, frames, config_dict)
            for name, frames in frames_data.items()
        ]
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(tqdm(
                executor.map(process_video_keypoints_wrapper, keypoints_args),
                total=len(keypoints_args),
                desc="Keypoints"
            ))
        
        keypoints_data = {}
        for video_name, keypoints in results:
            if keypoints is not None:
                keypoints_data[video_name] = keypoints
        
        logger.info(f" Keypoints: {len(keypoints_data)} videos en {time.time()-start_time:.1f}s")
        
        # ========================================
        # PASO 3: Segmentación (opcional)
        # ========================================
        if self.use_action_segmenter:
            logger.info("\n[3/3] Segmentando acciones (ActionSegmenter)...")
            start_time = time.time()
        
        sequences = []
        for video_name in tqdm(frames_data.keys(), desc="Procesando"):
            if video_name not in keypoints_data:
                continue
            
            # Buscar class_id
            class_id = None
            for key, cid in video_to_class.items():
                if video_name in key or key.replace('.mp4', '').replace('.avi', '') in video_name:
                    class_id = cid
                    break
            
            if class_id is None:
                continue
            
            frames = frames_data[video_name]
            keypoints = keypoints_data[video_name]
            
            # Aplicar ActionSegmenter si está habilitado
            if self.use_action_segmenter:
                try:
                    seg_start, seg_end = self.action_segmenter.segment_action(keypoints)
                    
                    # Validar segmento
                    segment_length = seg_end - seg_start
                    total_length = len(keypoints)
                    
                    if segment_length < total_length * 0.3:
                        expand = int((total_length * 0.5 - segment_length) / 2)
                        seg_start = max(0, seg_start - expand)
                        seg_end = min(total_length, seg_end + expand)
                    
                    frames = frames[seg_start:seg_end]
                    keypoints = keypoints[seg_start:seg_end]
                    
                except Exception as e:
                    logger.warning(f"[{video_name}] Segmentación falló: {e}, usando completo")
            
            sequences.append({
                'video_name': video_name,
                'class_id': class_id,
                'frames': frames,
                'keypoints': keypoints,
                'sequence_length': len(frames)
            })
        
        if self.use_action_segmenter:
            logger.info(f" Segmentación: {len(sequences)} videos en {time.time()-start_time:.1f}s")
        
        return sequences
    
    def save_sequences_fast(self, sequences: List[Dict], output_dir: Path):
        """
        Guarda secuencias con I/O paralelo
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        def save_worker(seq):
            video_name = seq['video_name']
            
            frames_path = output_dir / f"{video_name}_frames.npy"
            keypoints_path = output_dir / f"{video_name}_keypoints.npy"
            
            np.save(frames_path, seq['frames'])
            np.save(keypoints_path, seq['keypoints'])
            
            return {
                'video_name': video_name,
                'class_id': seq['class_id'],
                'sequence_length': seq['sequence_length'],
                'frames_path': str(frames_path.name),
                'keypoints_path': str(keypoints_path.name)
            }
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            metadata_list = list(tqdm(
                executor.map(save_worker, sequences),
                total=len(sequences),
                desc="Guardando"
            ))
        
        metadata_path = output_dir / "sequences_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        logger.info(f" Guardado: {output_dir}")


# ============================================================
# GPU FEATURE EXTRACTOR (igual que antes)
# ============================================================

class GPUFeatureExtractor:
    """Extractor GPU (sin cambios)"""
    
    def __init__(
        self,
        sequences_dir: Path,
        visual_extractor_path: Path,
        pose_extractor_path: Path,
        device: str = "cuda",
        batch_size: int = 64
    ):
        self.sequences_dir = sequences_dir
        self.device = torch.device(device)
        self.batch_size = batch_size
        
        self.visual_extractor = torch.load(visual_extractor_path, map_location=self.device, weights_only=False)
        self.visual_extractor.eval()
        
        self.pose_extractor = torch.load(pose_extractor_path, map_location=self.device, weights_only=False)
        self.pose_extractor.eval()
        
        from torchvision import transforms #type: ignore
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(" GPU Extractor cargado")
    
    @torch.no_grad()
    def extract_batch_features(self, frames_batch: List[np.ndarray], keypoints_batch: List[np.ndarray]) -> List[np.ndarray]:
        all_fused = []
        
        for frames, keypoints in zip(frames_batch, keypoints_batch):
            T = len(frames)
            
            # Visual
            all_visual = []
            for i in range(0, T, self.batch_size):
                batch_frames = frames[i:i+self.batch_size]
                batch_tensors = [self.transform(f if f.max() > 1.0 else (f * 255).astype(np.uint8)) for f in batch_frames]
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                visual_feat = self.visual_extractor(batch_tensor)
                all_visual.append(visual_feat.cpu().numpy())
            
            visual_features = np.concatenate(all_visual, axis=0)
            
            # Pose
            all_pose = []
            for i in range(0, T, self.batch_size):
                batch_keypoints = keypoints[i:i+self.batch_size]
                batch_flat = batch_keypoints.reshape(len(batch_keypoints), -1)
                batch_tensor = torch.from_numpy(batch_flat).float().to(self.device)
                pose_feat = self.pose_extractor(batch_tensor)
                all_pose.append(pose_feat.cpu().numpy())
            
            pose_features = np.concatenate(all_pose, axis=0)
            
            fused = np.concatenate([visual_features, pose_features], axis=1)
            all_fused.append(fused.astype(np.float16))
        
        return all_fused
    
    def process_all_sequences_gpu(self, output_dir: Path, mega_batch_size: int = 8):
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_path = self.sequences_dir / "sequences_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Extrayendo features de {len(metadata)} secuencias...")
        
        total_batches = (len(metadata) + mega_batch_size - 1) // mega_batch_size
        processed = 0
        
        for batch_idx in tqdm(range(total_batches), desc="GPU batches"):
            start_idx = batch_idx * mega_batch_size
            end_idx = min(start_idx + mega_batch_size, len(metadata))
            batch_metadata = metadata[start_idx:end_idx]
            
            frames_batch = []
            keypoints_batch = []
            names_batch = []
            
            for entry in batch_metadata:
                video_name = entry['video_name']
                output_path = output_dir / f"{video_name}_fused.npy"
                
                if output_path.exists():
                    continue
                
                try:
                    frames = np.load(self.sequences_dir / entry['frames_path'])
                    keypoints = np.load(self.sequences_dir / entry['keypoints_path'])
                    
                    frames_batch.append(frames)
                    keypoints_batch.append(keypoints)
                    names_batch.append(video_name)
                except Exception as e:
                    logger.error(f"Error {video_name}: {e}")
            
            if len(frames_batch) == 0:
                continue
            
            try:
                fused_batch = self.extract_batch_features(frames_batch, keypoints_batch)
                
                for name, fused in zip(names_batch, fused_batch):
                    np.save(output_dir / f"{name}_fused.npy", fused)
                    processed += 1
            except Exception as e:
                logger.error(f"Error batch {batch_idx}: {e}")
        
        logger.info(f" Features: {processed} secuencias")


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():
    """Pipeline híbrido completo"""
    
    total_start = time.time()
    
    # ========================================
    # PASO 1: Procesar videos
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("PASO 1: Procesamiento (Clases Originales + Paralelización)")
    logger.info("="*60)
    
    processor = HybridSequenceProcessor(
        config,
        use_action_segmenter=True,  #  USA ActionSegmenter
        num_workers=14
    )
    
    sequences = processor.process_all_videos_parallel()
    
    sequences_dir = Path("data/sequences")
    processor.save_sequences_fast(sequences, sequences_dir)
    
    # ========================================
    # PASO 2: Features
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("PASO 2: Extracción de Features (GPU)")
    logger.info("="*60)
    
    feature_extractor = GPUFeatureExtractor(
        sequences_dir=sequences_dir,
        visual_extractor_path=Path("models/extractors/visual_extractor_full.pt"),
        pose_extractor_path=Path("models/extractors/pose_extractor_full.pt"),
        device="cuda",
        batch_size=64
    )
    
    features_dir = Path("data/sequence_features")
    feature_extractor.process_all_sequences_gpu(features_dir, mega_batch_size=8)
    
    # ========================================
    # RESUMEN
    # ========================================
    total_time = time.time() - total_start
    
    logger.info("\n" + "="*60)
    logger.info(" PIPELINE COMPLETADO")
    logger.info("="*60)
    logger.info(f"  Tiempo: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  Videos: {len(sequences)}")
    logger.info(f"  Velocidad: {len(sequences)/(total_time/60):.1f} videos/min")
    logger.info("="*60)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()