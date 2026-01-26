"""
Extractor de features para videos completos (sin clips).
Procesa el video ignorando el 10% inicial y limitando a max_frames.
"""

import cv2  # type: ignore
import torch  # type: ignore
import numpy as np  # type: ignore
import mediapipe as mp  # type: ignore
from pathlib import Path
from typing import Tuple, Optional, List
from tqdm import tqdm  # type: ignore
import logging
from torchvision import transforms  # type: ignore
import json
import re

from pipelines_video.save_extractors import (
    ResNet101FeatureExtractor, PoseFeatureExtractor
)
from pipelines_video.config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoFrameExtractor:
    """
    Extrae frames de videos completos con:
    - Skip del 10% inicial (contenido sin valor)
    - Sampling uniforme hasta max_frames
    """
    
    def __init__(
        self,
        skip_start_percent: float = None,
        max_frames: int = None,
        frame_height: int = None,
        frame_width: int = None
    ):
        self.skip_start_percent = skip_start_percent or config.video.skip_start_percent
        self.max_frames = max_frames or config.video.max_frames
        self.frame_height = frame_height or config.video.frame_height
        self.frame_width = frame_width or config.video.frame_width
        
        logger.info(f"VideoFrameExtractor inicializado:")
        logger.info(f"  Skip inicio: {self.skip_start_percent * 100:.0f}%")
        logger.info(f"  Max frames: {self.max_frames}")
        logger.info(f"  Frame size: {self.frame_width}x{self.frame_height}")
    
    def extract_frames(self, video_path: Path) -> Tuple[np.ndarray, int]:
        """
        Extrae frames del video aplicando skip inicial y sampling.
        
        Args:
            video_path: Ruta al archivo de video
            
        Returns:
            frames: Array (T, H, W, 3) con T <= max_frames
            original_frame_count: Numero total de frames en el video
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            logger.warning(f"Video vacio o corrupto: {video_path}")
            cap.release()
            return np.array([]), 0
        
        # Calcular rango util (ignorar inicio)
        start_frame = int(total_frames * self.skip_start_percent)
        end_frame = total_frames
        useful_frames = end_frame - start_frame
        
        if useful_frames <= 0:
            logger.warning(f"Video muy corto despues de skip: {video_path}")
            cap.release()
            return np.array([]), total_frames
        
        # Calcular indices de frames a extraer
        num_frames_to_extract = min(useful_frames, self.max_frames)
        
        if useful_frames <= self.max_frames:
            # Tomar todos los frames disponibles
            frame_indices = list(range(start_frame, end_frame))
        else:
            # Sampling uniforme
            frame_indices = np.linspace(
                start_frame, 
                end_frame - 1, 
                num_frames_to_extract, 
                dtype=int
            ).tolist()
        
        # Extraer frames
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize y convertir BGR -> RGB
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            logger.warning(f"No se pudieron extraer frames de: {video_path}")
            return np.array([]), total_frames
        
        return np.array(frames, dtype=np.uint8), total_frames


class KeypointExtractor:
    """Extrae keypoints de pose y manos usando MediaPipe"""
    
    def __init__(self):
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
    
    def extract_keypoints(self, frames: np.ndarray) -> np.ndarray:
        """
        Extrae keypoints de todos los frames.
        
        Args:
            frames: Array (T, H, W, 3) RGB
            
        Returns:
            keypoints: Array (T, 75, 4) - pose(33) + hands(42)
        """
        T = len(frames)
        all_keypoints = []
        
        for frame in frames:
            # Pose keypoints
            pose_results = self.pose.process(frame)
            
            if pose_results.pose_landmarks:
                pose_kpts = np.array([
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in pose_results.pose_landmarks.landmark
                ])  # (33, 4)
                
                # Normalizar pose
                pose_kpts = self._normalize_pose(pose_kpts)
            else:
                pose_kpts = np.zeros((33, 4), dtype=np.float32)
            
            # Hand keypoints
            hand_results = self.hands.process(frame)
            hand_kpts = np.zeros((42, 4), dtype=np.float32)
            
            if hand_results.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks[:2]):
                    start_idx = hand_idx * 21
                    hand_kpts[start_idx:start_idx + 21] = np.array([
                        [lm.x, lm.y, lm.z, 1.0]
                        for lm in hand_landmarks.landmark
                    ])
            
            # Combinar pose + hands
            combined = np.concatenate([pose_kpts, hand_kpts], axis=0)  # (75, 4)
            all_keypoints.append(combined)
        
        return np.array(all_keypoints, dtype=np.float32)  # (T, 75, 4)
    
    def _normalize_pose(self, pose_kpts: np.ndarray) -> np.ndarray:
        """Normaliza keypoints usando caderas como referencia"""
        # Centro: promedio de caderas
        left_hip = pose_kpts[23][:3]
        right_hip = pose_kpts[24][:3]
        center = (left_hip + right_hip) / 2
        
        # Escala: distancia entre hombros
        shoulder_dist = np.linalg.norm(pose_kpts[11][:3] - pose_kpts[12][:3])
        scale = max(shoulder_dist, 0.1)
        
        # Normalizar
        normalized = pose_kpts.copy()
        normalized[:, :3] = (pose_kpts[:, :3] - center) / scale
        normalized[:, :3] = np.clip(normalized[:, :3], -1, 1)
        
        return normalized
    
    def close(self):
        """Libera recursos de MediaPipe"""
        self.pose.close()
        self.hands.close()


def load_visual_extractor(extractor_path: Path, device: torch.device):
    """Carga el extractor visual (ResNet101)"""
    if not extractor_path.exists():
        raise FileNotFoundError(f"Extractor visual no encontrado: {extractor_path}")
    
    extractor = torch.load(extractor_path, map_location=device, weights_only=False)
    extractor.eval()
    return extractor


def load_pose_extractor(extractor_path: Path, device: torch.device):
    """Carga el extractor de pose (MLP)"""
    if not extractor_path.exists():
        raise FileNotFoundError(f"Extractor de pose no encontrado: {extractor_path}")
    
    extractor = torch.load(extractor_path, map_location=device, weights_only=False)
    extractor.eval()
    return extractor


def process_single_video(
    video_path: Path,
    frame_extractor: VideoFrameExtractor,
    keypoint_extractor: KeypointExtractor,
    visual_model: torch.nn.Module,
    pose_model: torch.nn.Module,
    device: torch.device,
    visual_transform: transforms.Compose,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Procesa un video completo y extrae features.
    
    Returns:
        visual_features: (T, 1024)
        pose_features: (T, 128)
        fused_features: (T, 1152)
        num_frames: T
    """
    # 1. Extraer frames
    frames, total_frames = frame_extractor.extract_frames(video_path)
    
    if len(frames) == 0:
        return None, None, None, 0
    
    T = len(frames)
    
    # 2. Extraer keypoints
    keypoints = keypoint_extractor.extract_keypoints(frames)  # (T, 75, 4)
    keypoints_flat = keypoints.reshape(T, -1)  # (T, 300)
    
    # 3. Extraer features visuales
    visual_features = []
    with torch.no_grad():
        for i in range(0, T, batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_tensors = torch.stack([
                visual_transform(frame) for frame in batch_frames
            ]).to(device)
            
            features = visual_model(batch_tensors)  # (B, 1024)
            visual_features.append(features.cpu().numpy())
    
    visual_features = np.concatenate(visual_features, axis=0)  # (T, 1024)
    
    # 4. Extraer features de pose
    pose_features = []
    with torch.no_grad():
        for i in range(0, T, batch_size):
            batch_kpts = torch.from_numpy(
                keypoints_flat[i:i + batch_size]
            ).float().to(device)
            
            features = pose_model(batch_kpts)  # (B, 128)
            pose_features.append(features.cpu().numpy())
    
    pose_features = np.concatenate(pose_features, axis=0)  # (T, 128)
    
    # 5. Fusionar features
    fused_features = np.concatenate([visual_features, pose_features], axis=1)  # (T, 1152)
    
    return visual_features, pose_features, fused_features, T


def extract_all_video_features(
    videos_dir: Path = None,
    metadata_path: Path = None,
    output_dir: Path = None,
    visual_extractor_path: Path = None,
    pose_extractor_path: Path = None,
    device: str = None,
    skip_start_percent: float = None,
    max_frames: int = None,
    batch_size: int = 32,
    save_intermediate: bool = False,
    checkpoint_every: int = 50,
    cooldown_every: int = 100,
    cooldown_seconds: int = 10
):
    """
    Procesa todos los videos y extrae features fusionadas.
    
    Args:
        videos_dir: Directorio con videos
        metadata_path: Ruta a dataset_meta.json
        output_dir: Directorio de salida para features
        visual_extractor_path: Ruta al extractor visual
        pose_extractor_path: Ruta al extractor de pose
        device: 'cuda' o 'cpu'
        skip_start_percent: Porcentaje inicial a ignorar
        max_frames: Maximo de frames por video
        batch_size: Batch size para extraccion
        save_intermediate: Guardar visual y pose por separado
        checkpoint_every: Guardar checkpoint cada N videos
        cooldown_every: Pausar cada N videos para enfriar GPU
        cooldown_seconds: Segundos de pausa para enfriamiento
    """
    import time
    import gc
    # Defaults
    if videos_dir is None:
        videos_dir = config.data_paths.raw_videos
    if metadata_path is None:
        metadata_path = config.data_paths.dataset_meta
    if output_dir is None:
        output_dir = config.data_paths.features_fused
    if visual_extractor_path is None:
        visual_extractor_path = config.data_paths.visual_extractor
    if pose_extractor_path is None:
        pose_extractor_path = config.data_paths.pose_extractor
    if device is None:
        device = config.training.device
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Crear directorios
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Archivo de checkpoint para resumir
    checkpoint_file = output_dir / "_checkpoint.json"
    processed_ids = set()
    
    # Cargar checkpoint si existe
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            processed_ids = set(checkpoint_data.get('processed_ids', []))
            logger.info(f"Checkpoint encontrado: {len(processed_ids)} videos ya procesados")
    
    if save_intermediate:
        visual_dir = config.data_paths.features_visual
        pose_dir = config.data_paths.features_pose
        visual_dir.mkdir(parents=True, exist_ok=True)
        pose_dir.mkdir(parents=True, exist_ok=True)
    
    # Cargar metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Parsear metadata
    if isinstance(metadata, dict) and "videos" in metadata:
        videos_list = metadata["videos"]
    elif isinstance(metadata, list):
        videos_list = metadata
    else:
        raise ValueError(f"Formato de metadata no soportado")
    
    logger.info(f"Videos en metadata: {len(videos_list)}")
    
    # Inicializar extractores
    frame_extractor = VideoFrameExtractor(
        skip_start_percent=skip_start_percent,
        max_frames=max_frames
    )
    keypoint_extractor = KeypointExtractor()
    
    # Cargar modelos
    logger.info("Cargando modelos...")
    visual_model = load_visual_extractor(visual_extractor_path, device)
    pose_model = load_pose_extractor(pose_extractor_path, device)
    
    # Transform para visual
    visual_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Procesar videos
    processed = 0
    skipped = 0
    errors = 0
    
    for entry in tqdm(videos_list, desc="Procesando videos"):
        video_file = entry.get("video_file", "")
        class_id = entry.get("class_id")
        
        if not video_file or class_id is None:
            skipped += 1
            continue
        
        # Construir ruta - priorizar video_path absoluto si existe
        video_path_str = entry.get("video_path", "")
        
        # Intentar ruta absoluta primero
        if video_path_str:
            video_path = Path(video_path_str)
        
        # Si no existe, intentar relativo a videos_dir
        if not video_path.exists():
            video_path = videos_dir / video_file
        
        # Si aun no existe, buscar por class_name/video_file
        if not video_path.exists():
            class_name = entry.get("class_name", "")
            if class_name:
                video_path = videos_dir / class_name / video_file
        
        if not video_path.exists():
            logger.warning(f"Video no encontrado: {video_file} (intentado: {video_path_str})")
            skipped += 1
            continue
        
        # Nombre de salida (basado en video_id)
        video_id = re.match(r'^(\d+)', video_file)
        if video_id:
            video_id = video_id.group(1)
        else:
            video_id = video_path.stem
        
        output_path = output_dir / f"{video_id}_fused.npy"
        
        # Skip si ya fue procesado (checkpoint o archivo existente)
        if video_id in processed_ids or output_path.exists():
            if video_id not in processed_ids:
                processed_ids.add(video_id)
            skipped += 1
            continue
        
        try:
            visual_feat, pose_feat, fused_feat, T = process_single_video(
                video_path=video_path,
                frame_extractor=frame_extractor,
                keypoint_extractor=keypoint_extractor,
                visual_model=visual_model,
                pose_model=pose_model,
                device=device,
                visual_transform=visual_transform,
                batch_size=batch_size
            )
            
            if fused_feat is None:
                errors += 1
                continue
            
            # Guardar features fusionadas
            np.save(output_path, fused_feat.astype(np.float16))
            
            # Guardar intermedios si se solicita
            if save_intermediate:
                np.save(visual_dir / f"{video_id}_visual.npy", visual_feat.astype(np.float16))
                np.save(pose_dir / f"{video_id}_pose.npy", pose_feat.astype(np.float16))
            
            processed += 1
            processed_ids.add(video_id)
            
            # Liberar memoria GPU explicitamente
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Checkpoint periodico
            if processed % checkpoint_every == 0:
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'processed_ids': list(processed_ids),
                        'processed_count': processed,
                        'skipped_count': skipped,
                        'error_count': errors
                    }, f)
                logger.info(f"Checkpoint guardado: {processed} videos procesados")
            
            # Cooldown periodico para evitar sobrecalentamiento
            if processed % cooldown_every == 0 and cooldown_seconds > 0:
                logger.info(f"Enfriamiento de {cooldown_seconds}s para GPU...")
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                time.sleep(cooldown_seconds)
            
        except Exception as e:
            logger.error(f"Error procesando {video_file}: {e}")
            errors += 1
            
            # Liberar memoria en caso de error
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            continue
    
    # Cleanup
    keypoint_extractor.close()
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Guardar checkpoint final
    with open(checkpoint_file, 'w') as f:
        json.dump({
            'processed_ids': list(processed_ids),
            'processed_count': processed,
            'skipped_count': skipped,
            'error_count': errors,
            'completed': True
        }, f)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Extraccion completada!")
    logger.info(f"  Procesados: {processed}")
    logger.info(f"  Omitidos: {skipped}")
    logger.info(f"  Errores: {errors}")
    logger.info(f"  Checkpoint guardado en: {checkpoint_file}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extrae features de videos completos")
    
    parser.add_argument("--config", type=Path, default=None,
                        help="Archivo JSON de configuracion")
    parser.add_argument("--videos_dir", type=Path, default=None,
                        help="Directorio con videos")
    parser.add_argument("--metadata_path", type=Path, default=None,
                        help="Ruta a dataset_meta.json")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Directorio de salida")
    parser.add_argument("--visual_extractor", type=Path, default=None,
                        help="Ruta al extractor visual")
    parser.add_argument("--pose_extractor", type=Path, default=None,
                        help="Ruta al extractor de pose")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu"])
    parser.add_argument("--skip_start", type=float, default=None,
                        help=f"Porcentaje inicial a ignorar (default: {config.video.skip_start_percent})")
    parser.add_argument("--max_frames", type=int, default=None,
                        help=f"Max frames por video (default: {config.video.max_frames})")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_intermediate", action="store_true",
                        help="Guardar features visual y pose por separado")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                        help="Guardar checkpoint cada N videos (default: 50)")
    parser.add_argument("--cooldown_every", type=int, default=100,
                        help="Pausar para enfriar GPU cada N videos (default: 100)")
    parser.add_argument("--cooldown_seconds", type=int, default=10,
                        help="Segundos de pausa para enfriamiento (default: 10)")
    
    args = parser.parse_args()
    
    extract_all_video_features(
        videos_dir=args.videos_dir,
        metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        visual_extractor_path=args.visual_extractor,
        pose_extractor_path=args.pose_extractor,
        device=args.device,
        skip_start_percent=args.skip_start,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
        save_intermediate=args.save_intermediate,
        checkpoint_every=args.checkpoint_every,
        cooldown_every=args.cooldown_every,
        cooldown_seconds=args.cooldown_seconds
    )
