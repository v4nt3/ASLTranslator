"""
Pipeline SIMPLE Y RÁPIDO
Sin problemas de serialización, mantiene lógica original

ESTRATEGIA:
- Paralelización a nivel de LOTES (no de workers individuales)
- Usa ThreadPoolExecutor en vez de ProcessPoolExecutor (evita serialización)
- Procesa múltiples videos simultáneamente con threading
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import json
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import mediapipe as mp

from config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# PROCESADOR SIMPLE CON THREADING
# ============================================================

class SimpleSequenceProcessor:
    """
    Procesador que usa threading (evita problemas de serialización)
    y agrupa procesamiento en batches
    """
    
    def __init__(self, config, max_frames: int = 120, num_threads: int = 8):
        self.config = config
        self.max_frames = max_frames
        self.num_threads = num_threads
        
        logger.info("="*60)
        logger.info("SimpleSequenceProcessor")
        logger.info(f"  Max frames: {max_frames}")
        logger.info(f"  Threads: {num_threads}")
        logger.info("="*60)
    
    def extract_frames_from_video(self, video_path: Path) -> Optional[np.ndarray]:
        """Extrae frames de UN video"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                cap.release()
                return None
            
            # Submuestrear si es necesario
            if total_frames > self.max_frames:
                indices = np.linspace(0, total_frames - 1, self.max_frames, dtype=int)
            else:
                indices = np.arange(total_frames)
            
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, (224, 224))
                    frames.append(frame)
            
            cap.release()
            
            if len(frames) == 0:
                return None
            
            return np.array(frames, dtype=np.uint8)
            
        except Exception as e:
            logger.error(f"Error extrayendo frames de {video_path.name}: {e}")
            return None
    
    def extract_keypoints_from_frames(self, frames: np.ndarray) -> Optional[np.ndarray]:
        """Extrae keypoints de frames usando MediaPipe"""
        try:
            # Inicializar MediaPipe
            mp_pose = mp.solutions.pose
            mp_hands = mp.solutions.hands
            
            pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5
            )
            
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5
            )
            
            keypoints_sequence = []
            
            for frame in frames:
                # Extraer pose
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = pose.process(frame_rgb)
                
                pose_kpts = np.zeros((33, 4), dtype=np.float32)
                if pose_results.pose_landmarks:
                    for i, lm in enumerate(pose_results.pose_landmarks.landmark):
                        pose_kpts[i] = [lm.x, lm.y, lm.z, lm.visibility]
                
                # Normalizar pose (IGUAL que data_preparation.py)
                left_hip = pose_kpts[23][:3]
                right_hip = pose_kpts[24][:3]
                center = (left_hip + right_hip) / 2
                
                left_shoulder = pose_kpts[11][:3]
                right_shoulder = pose_kpts[12][:3]
                shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
                scale = max(shoulder_dist, 0.1)
                
                pose_kpts[:, :3] = (pose_kpts[:, :3] - center) / scale
                pose_kpts[:, :3] = np.clip(pose_kpts[:, :3], -1, 1)
                
                # Extraer manos
                hands_results = hands.process(frame_rgb)
                hand_kpts = np.zeros((42, 4), dtype=np.float32)
                
                if hands_results.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks[:2]):
                        start_idx = hand_idx * 21
                        for i, lm in enumerate(hand_landmarks.landmark):
                            hand_kpts[start_idx + i] = [lm.x, lm.y, lm.z, 1.0]
                
                # Combinar
                combined = np.concatenate([pose_kpts, hand_kpts], axis=0)
                keypoints_sequence.append(combined)
            
            pose.close()
            hands.close()
            
            if len(keypoints_sequence) == 0:
                return None
            
            return np.array(keypoints_sequence, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extrayendo keypoints: {e}")
            return None
    
    def process_single_video(self, video_path: Path, class_id: int) -> Optional[Dict]:
        """Procesa UN video completo (frames + keypoints)"""
        video_name = video_path.stem
        
        # Extraer frames
        frames = self.extract_frames_from_video(video_path)
        if frames is None:
            return None
        
        # Extraer keypoints
        keypoints = self.extract_keypoints_from_frames(frames)
        if keypoints is None:
            return None
        
        return {
            'video_name': video_name,
            'class_id': class_id,
            'frames': frames,
            'keypoints': keypoints,
            'sequence_length': len(frames)
        }
    
    def process_all_videos(self) -> List[Dict]:
        """
        Procesa todos los videos usando ThreadPoolExecutor
        (evita problemas de serialización de ProcessPoolExecutor)
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
        
        # Mapear videos a class_id
        video_class_pairs = []
        for video_path in video_files:
            class_id = None
            for key, cid in video_to_class.items():
                if video_path.name in key or video_path.stem in key.replace('.mp4', '').replace('.avi', ''):
                    class_id = cid
                    break
            
            if class_id is not None:
                video_class_pairs.append((video_path, class_id))
        
        logger.info(f"Videos con class_id válido: {len(video_class_pairs)}")
        
        # Procesar con ThreadPoolExecutor
        sequences = []
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [
                executor.submit(self.process_single_video, video_path, class_id)
                for video_path, class_id in video_class_pairs
            ]
            
            for future in tqdm(futures, desc="Procesando videos"):
                result = future.result()
                if result is not None:
                    sequences.append(result)
        
        logger.info(f"✅ Videos procesados exitosamente: {len(sequences)}")
        
        return sequences
    
    def save_sequences(self, sequences: List[Dict], output_dir: Path):
        """Guarda secuencias"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_list = []
        
        for seq in tqdm(sequences, desc="Guardando"):
            video_name = seq['video_name']
            
            frames_path = output_dir / f"{video_name}_frames.npy"
            keypoints_path = output_dir / f"{video_name}_keypoints.npy"
            
            np.save(frames_path, seq['frames'])
            np.save(keypoints_path, seq['keypoints'])
            
            metadata_list.append({
                'video_name': video_name,
                'class_id': seq['class_id'],
                'sequence_length': seq['sequence_length'],
                'frames_path': str(frames_path.name),
                'keypoints_path': str(keypoints_path.name)
            })
        
        metadata_path = output_dir / "sequences_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        logger.info(f"✅ Guardado: {output_dir}")


# ============================================================
# GPU FEATURE EXTRACTOR
# ============================================================

class GPUFeatureExtractor:
    """Extractor GPU optimizado"""
    
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
        
        logger.info(f"Cargando extractors en {device}...")
        
        self.visual_extractor = torch.load(visual_extractor_path, map_location=self.device, weights_only=False)
        self.visual_extractor.eval()
        
        self.pose_extractor = torch.load(pose_extractor_path, map_location=self.device, weights_only=False)
        self.pose_extractor.eval()
        
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("✅ Extractors cargados")
    
    @torch.no_grad()
    def extract_features(self, frames: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Extrae features de una secuencia"""
        T = len(frames)
        
        # Visual features
        all_visual = []
        for i in range(0, T, self.batch_size):
            batch_frames = frames[i:i+self.batch_size]
            batch_tensors = []
            
            for frame in batch_frames:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                batch_tensors.append(self.transform(frame))
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            visual_feat = self.visual_extractor(batch_tensor)
            all_visual.append(visual_feat.cpu().numpy())
        
        visual_features = np.concatenate(all_visual, axis=0)
        
        # Pose features
        all_pose = []
        for i in range(0, T, self.batch_size):
            batch_keypoints = keypoints[i:i+self.batch_size]
            batch_flat = batch_keypoints.reshape(len(batch_keypoints), -1)
            batch_tensor = torch.from_numpy(batch_flat).float().to(self.device)
            pose_feat = self.pose_extractor(batch_tensor)
            all_pose.append(pose_feat.cpu().numpy())
        
        pose_features = np.concatenate(all_pose, axis=0)
        
        # Fusionar
        fused = np.concatenate([visual_features, pose_features], axis=1)
        return fused.astype(np.float16)
    
    def process_all_sequences(self, output_dir: Path):
        """Procesa todas las secuencias"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_path = self.sequences_dir / "sequences_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Extrayendo features de {len(metadata)} secuencias...")
        
        processed = 0
        for entry in tqdm(metadata, desc="GPU features"):
            video_name = entry['video_name']
            output_path = output_dir / f"{video_name}_fused.npy"
            
            if output_path.exists():
                continue
            
            try:
                frames = np.load(self.sequences_dir / entry['frames_path'])
                keypoints = np.load(self.sequences_dir / entry['keypoints_path'])
                
                fused = self.extract_features(frames, keypoints)
                np.save(output_path, fused)
                
                processed += 1
            except Exception as e:
                logger.error(f"Error {video_name}: {e}")
        
        logger.info(f"✅ Features: {processed} secuencias")


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():
    """Pipeline simple y rápido"""
    
    total_start = time.time()
    
    # ========================================
    # PASO 1: Procesar videos
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("PASO 1: Procesamiento (Threading)")
    logger.info("="*60)
    
    processor = SimpleSequenceProcessor(
        config,
        max_frames=120,
        num_threads=8  # Threads para I/O bound (MediaPipe)
    )
    
    sequences = processor.process_all_videos()
    
    sequences_dir = Path("data/sequences")
    processor.save_sequences(sequences, sequences_dir)
    
    # ========================================
    # PASO 2: Features
    # ========================================
    logger.info("\n" + "="*60)
    logger.info("PASO 2: Features (GPU)")
    logger.info("="*60)
    
    feature_extractor = GPUFeatureExtractor(
        sequences_dir=sequences_dir,
        visual_extractor_path=Path("models/extractors/visual_extractor_full.pt"),
        pose_extractor_path=Path("models/extractors/pose_extractor_full.pt"),
        device="cuda",
        batch_size=64
    )
    
    features_dir = Path("data/sequence_features")
    feature_extractor.process_all_sequences(features_dir)
    
    # ========================================
    # RESUMEN
    # ========================================
    total_time = time.time() - total_start
    
    logger.info("\n" + "="*60)
    logger.info("✅ COMPLETADO")
    logger.info("="*60)
    logger.info(f"  Tiempo: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  Videos: {len(sequences)}")
    logger.info(f"  Velocidad: {len(sequences)/(total_time/60):.1f} videos/min")
    logger.info("="*60)


if __name__ == "__main__":
    main()