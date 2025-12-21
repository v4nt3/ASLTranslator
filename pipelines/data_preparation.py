"""
Pipeline modificado SIN creación de clips
Usa el segmentador para extraer solo la parte útil del video (~30 frames)
Procesa con ResNet y MLP de la misma forma
"""

import cv2 # type: ignore
import numpy as np # type: ignore
import mediapipe as mp # type: ignore
from pathlib import Path
from typing import Tuple, Optional
import logging
from tqdm import tqdm # type: ignore
from config import config
import json
import re

logger = logging.getLogger(__name__)


class ActionSegmenter:
    """
    Detecta automáticamente la región donde ocurre la seña usando keypoints.
    Calcula un action score por frame y encuentra la ventana más relevante.
    """
    
    def __init__(self, config, debug=False):
        self.config = config
        self.debug = debug
    
    def calculate_action_score(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Calcula un score de acción por frame (0 a 1).
        Basado en: visibilidad de manos, altura de manos, movimiento.
        
        Args:
            keypoints: shape (T, 75, 4) - [pose(33) + hands(42)]
        
        Returns:
            scores: shape (T,) - score de acción por frame
        """
        T = len(keypoints)
        scores = np.zeros(T)
        
        for t in range(T):
            frame_kpts = keypoints[t]
            
            pose_kpts = frame_kpts[:33]
            left_hand_kpts = frame_kpts[33:54]
            right_hand_kpts = frame_kpts[54:75]
            
            score = 0.0
            
            # 1. Visibilidad de manos (weight: 0.3)
            left_visibility = np.mean(left_hand_kpts[:, 3])
            right_visibility = np.mean(right_hand_kpts[:, 3])
            hand_visibility = (left_visibility + right_visibility) / 2.0
            score += 0.3 * hand_visibility
            
            # 2. Manos levantadas (weight: 0.3)
            hip_y = np.mean([pose_kpts[23, 1], pose_kpts[24, 1]])
            left_hand_y = np.mean(left_hand_kpts[:, 1])
            right_hand_y = np.mean(right_hand_kpts[:, 1])
            
            hands_raised = 0.0
            if left_visibility > 0.3:
                hands_raised += 1.0 if left_hand_y < hip_y else 0.0
            if right_visibility > 0.3:
                hands_raised += 1.0 if right_hand_y < hip_y else 0.0
            hands_raised = hands_raised / 2.0
            score += 0.3 * hands_raised
            
            # 3. Proximidad a la cara (weight: 0.2)
            nose = pose_kpts[0]
            left_hand_pos = np.mean(left_hand_kpts[:3, :3], axis=0)
            right_hand_pos = np.mean(right_hand_kpts[:3, :3], axis=0)
            
            left_dist = np.linalg.norm(left_hand_pos - nose[:3])
            right_dist = np.linalg.norm(right_hand_pos - nose[:3])
            hand_proximity = 1.0 - np.clip(np.mean([left_dist, right_dist]), 0, 0.5) / 0.5
            score += 0.2 * max(0, hand_proximity)
            
            # 4. Movimiento (weight: 0.2)
            if t > 0:
                prev_kpts = keypoints[t - 1]
                left_hand_prev = np.mean(prev_kpts[33:54, :3], axis=0)
                left_hand_curr = np.mean(left_hand_kpts[:, :3], axis=0)
                left_motion = np.linalg.norm(left_hand_curr - left_hand_prev)
                
                right_hand_prev = np.mean(prev_kpts[54:75, :3], axis=0)
                right_hand_curr = np.mean(right_hand_kpts[:, :3], axis=0)
                right_motion = np.linalg.norm(right_hand_curr - right_hand_prev)
                
                motion_score = np.clip(np.mean([left_motion, right_motion]) / 0.05, 0, 1)
                score += 0.2 * motion_score
            
            scores[t] = np.clip(score, 0, 1)
        
        return scores
    
    def find_action_window(self, scores: np.ndarray, target_length: int = 30) -> Tuple[int, int]:
        """
        Encuentra la ventana de máxima actividad con longitud objetivo
        
        Args:
            scores: shape (T,)
            target_length: longitud deseada de la ventana
        
        Returns:
            (start_idx, end_idx) - índices de la ventana
        """
        T = len(scores)
        
        if T <= target_length:
            return 0, T
        
        # Suavizar scores
        window_size = min(5, T // 4)
        smoothed_scores = np.convolve(scores, np.ones(window_size) / window_size, mode='same')
        
        # Encontrar ventana de target_length con mayor suma
        max_sum = -np.inf
        best_start = 0
        
        for start in range(T - target_length + 1):
            window_sum = np.sum(smoothed_scores[start:start + target_length])
            if window_sum > max_sum:
                max_sum = window_sum
                best_start = start
        
        return best_start, best_start + target_length
    
    def segment_action(self, keypoints: np.ndarray, target_length: int = 30) -> Tuple[int, int]:
        """
        Segmenta la región útil de la seña
        
        Args:
            keypoints: shape (T, 75, 4)
            target_length: número de frames deseado
        
        Returns:
            (start_idx, end_idx) - región de la seña
        """
        try:
            scores = self.calculate_action_score(keypoints)
            
            if self.debug:
                logger.debug(f"Scores: min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}")
            
            # Si scores muy bajos, usar ventana central
            if scores.max() < 0.2:
                logger.warning("Scores bajos, usando ventana central")
                T = len(keypoints)
                mid = T // 2
                start = max(0, mid - target_length // 2)
                end = min(T, start + target_length)
                return start, end
            
            start_idx, end_idx = self.find_action_window(scores, target_length)
            
            if self.debug:
                logger.debug(f"Segmento: [{start_idx}, {end_idx}], longitud={end_idx - start_idx}")
            
            return start_idx, end_idx
            
        except Exception as e:
            logger.warning(f"Error en segmentación: {e}. Usando fallback.")
            T = len(keypoints)
            mid = T // 2
            start = max(0, mid - target_length // 2)
            end = min(T, start + target_length)
            return start, end


class KeypointExtractor:
    """Extrae keypoints usando MediaPipe"""
    
    def __init__(self, config):
        self.config = config
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
    
    def extract_pose_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extrae keypoints de pose de un frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = np.array([
                [lm.x, lm.y, lm.z, lm.visibility] 
                for lm in results.pose_landmarks.landmark
            ])
            return landmarks
        return None
    
    def extract_hand_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extrae keypoints de manos"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        hand_keypoints = np.zeros((42, 4))
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                start_idx = hand_idx * 21
                hand_keypoints[start_idx:start_idx+21] = np.array([
                    [lm.x, lm.y, lm.z, 1.0]
                    for lm in hand_landmarks.landmark
                ])
        
        return hand_keypoints if np.any(hand_keypoints) else None
    
    def normalize_keypoints(self, pose_kpts: np.ndarray) -> np.ndarray:
        """Normaliza keypoints"""
        left_hip = pose_kpts[23][:3]
        right_hip = pose_kpts[24][:3]
        center = (left_hip + right_hip) / 2
        
        shoulder_dist = np.linalg.norm(pose_kpts[11][:3] - pose_kpts[12][:3])
        scale = max(shoulder_dist, 0.1)
        
        normalized = pose_kpts.copy()
        normalized[:, :3] = (pose_kpts[:, :3] - center) / scale
        normalized[:, :3] = np.clip(normalized[:, :3], -1, 1)
        
        return normalized


class VideoProcessor:
    """
    Procesa videos completos:
    1. Extrae todos los frames del video
    2. Extrae keypoints de todos los frames
    3. Usa segmentador para encontrar región útil
    4. Extrae ~30 frames de esa región
    5. Guarda frames y keypoints correspondientes
    """
    
    def __init__(self, config, target_frames: int = 30, debug: bool = False):
        self.config = config
        self.target_frames = target_frames
        self.debug = debug
        
        self.keypoint_extractor = KeypointExtractor(config)
        self.segmenter = ActionSegmenter(config, debug=debug)
        
        self.frame_height = config.data.frame_height
        self.frame_width = config.data.frame_width
    
    def process_video(self, video_path: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Procesa un video completo y retorna frames y keypoints de la región útil
        
        Args:
            video_path: ruta al video
        
        Returns:
            (frames, keypoints) - ambos de shape (T, ...) donde T ~= target_frames
            None, None si hay error
        """
        try:
            # 1. Leer todos los frames del video
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                logger.warning(f"Video vacío: {video_path}")
                return None, None
            
            all_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                all_frames.append(frame)
            
            cap.release()
            all_frames = np.array(all_frames)
            
            if len(all_frames) == 0:
                logger.warning(f"No se pudieron leer frames: {video_path}")
                return None, None
            
            # 2. Extraer keypoints de todos los frames
            all_keypoints = []
            for frame in all_frames:
                pose_kpts = self.keypoint_extractor.extract_pose_keypoints(frame)
                
                if pose_kpts is not None:
                    pose_kpts = self.keypoint_extractor.normalize_keypoints(pose_kpts)
                    hand_kpts = self.keypoint_extractor.extract_hand_keypoints(frame)
                    
                    if hand_kpts is not None:
                        combined_kpts = np.concatenate([pose_kpts, hand_kpts], axis=0)
                    else:
                        hand_placeholder = np.zeros((42, 4))
                        combined_kpts = np.concatenate([pose_kpts, hand_placeholder], axis=0)
                    
                    all_keypoints.append(combined_kpts)
                else:
                    # Si falla extracción, usar placeholder
                    all_keypoints.append(np.zeros((75, 4)))
            
            all_keypoints = np.array(all_keypoints)  # (T, 75, 4)
            
            # 3. Usar segmentador para encontrar región útil
            start_idx, end_idx = self.segmenter.segment_action(all_keypoints, self.target_frames)
            
            # 4. Extraer frames de la región útil
            useful_frames = all_frames[start_idx:end_idx]
            useful_keypoints = all_keypoints[start_idx:end_idx]
            
            # 5. Si la región es más larga que target_frames, samplear uniformemente
            actual_length = len(useful_frames)
            if actual_length > self.target_frames:
                # Samplear uniformemente
                indices = np.linspace(0, actual_length - 1, self.target_frames, dtype=int)
                useful_frames = useful_frames[indices]
                useful_keypoints = useful_keypoints[indices]
            
            if self.debug:
                logger.debug(f"Video procesado: {video_path.name}")
                logger.debug(f"  Total frames: {len(all_frames)}")
                logger.debug(f"  Región útil: [{start_idx}, {end_idx}]")
                logger.debug(f"  Frames finales: {len(useful_frames)}")
            
            return useful_frames, useful_keypoints
            
        except Exception as e:
            logger.error(f"Error procesando video {video_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None, None
    
    def process_dataset(
        self,
        videos_dir: Path,
        output_frames_dir: Path,
        output_keypoints_dir: Path,
        metadata_path: Path
    ):
        """
        Procesa todo el dataset
        
        Args:
            videos_dir: directorio con videos raw
            output_frames_dir: donde guardar frames procesados
            output_keypoints_dir: donde guardar keypoints
            metadata_path: ruta a dataset_meta.json
        """
        output_frames_dir.mkdir(parents=True, exist_ok=True)
        output_keypoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Cargar metadata
        logger.info(f"Cargando metadata desde: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Construir mapeo video_file -> info
        video_info = {}
        if isinstance(metadata, dict) and "videos" in metadata:
            for entry in metadata["videos"]:
                video_file = entry.get("video_file", "")
                if video_file:
                    video_info[video_file] = entry
        
        # Buscar videos
        video_files = list(videos_dir.glob("**/*.mp4")) + list(videos_dir.glob("**/*.avi"))
        logger.info(f"Encontrados {len(video_files)} videos para procesar")
        
        processed = 0
        skipped = 0
        errors = 0
        
        for video_path in tqdm(video_files, desc="Procesando videos"):
            video_filename = video_path.name
            
            # Extraer video_id del nombre
            match = re.match(r'^(\d+)', video_filename)
            if not match:
                logger.warning(f"No se pudo extraer ID de: {video_filename}")
                errors += 1
                continue
            
            video_id = match.group(1)
            
            # Nombres de salida
            frames_output = output_frames_dir / f"{video_id}_frames.npy"
            keypoints_output = output_keypoints_dir / f"{video_id}_keypoints.npy"
            
            # Skip si ya existe
            if frames_output.exists() and keypoints_output.exists():
                skipped += 1
                continue
            
            # Procesar video
            frames, keypoints = self.process_video(video_path)
            
            if frames is not None and keypoints is not None:
                # Guardar
                np.save(frames_output, frames)
                np.save(keypoints_output, keypoints)
                processed += 1
            else:
                errors += 1
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Procesamiento completado!")
        logger.info(f"   ✓ Procesados: {processed}")
        logger.info(f"   ⊘ Omitidos: {skipped}")
        logger.info(f"   ✗ Errores: {errors}")
        logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description="Procesa videos SIN clips - usa segmentador para extraer región útil"
    )
    parser.add_argument(
        "--videos_dir",
        type=Path,
        default=config.data_paths.raw_videos,
        help="Directorio con videos raw"
    )
    parser.add_argument(
        "--output_frames_dir",
        type=Path,
        default=Path("data/processed_frames"),
        help="Directorio de salida para frames"
    )
    parser.add_argument(
        "--output_keypoints_dir",
        type=Path,
        default=Path("data/processed_keypoints"),
        help="Directorio de salida para keypoints"
    )
    parser.add_argument(
        "--metadata_path",
        type=Path,
        default=config.data_paths.dataset_meta,
        help="Ruta a dataset_meta.json"
    )
    parser.add_argument(
        "--target_frames",
        type=int,
        default=30,
        help="Número objetivo de frames a extraer (~30)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activar modo debug"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("PROCESAMIENTO DE VIDEOS SIN CLIPS")
    logger.info("="*60)
    logger.info(f"Videos dir: {args.videos_dir}")
    logger.info(f"Output frames: {args.output_frames_dir}")
    logger.info(f"Output keypoints: {args.output_keypoints_dir}")
    logger.info(f"Target frames: {args.target_frames}")
    logger.info(f"Debug: {args.debug}")
    logger.info("="*60 + "\n")
    
    processor = VideoProcessor(
        config=config,
        target_frames=args.target_frames,
        debug=args.debug
    )
    
    processor.process_dataset(
        videos_dir=args.videos_dir,
        output_frames_dir=args.output_frames_dir,
        output_keypoints_dir=args.output_keypoints_dir,
        metadata_path=args.metadata_path
    )
