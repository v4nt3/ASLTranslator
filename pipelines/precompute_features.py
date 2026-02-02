"""
Script para precomputar features visuales y de pose desde videos procesados
(sin clips, directamente desde la salida del segmentador)
"""

import torch #type: ignore
import numpy as np #type: ignore
from pathlib import Path
from tqdm import tqdm #type: ignore
import logging
from torchvision import transforms #type: ignore
from pipelines_video.save_extractors import ResNet101FeatureExtractor, PoseFeatureExtractor

from pipelines.config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_visual_extractor(extractor_path: Path, device: str):
    """Carga el extractor visual guardado"""
    
    if not extractor_path.exists():
        raise FileNotFoundError(
            f"Extractor no encontrado: {extractor_path}\n"
            f"   Ejecuta primero: python pipelines/save_extractors.py"
        )
    
    logger.info(f"Cargando extractor visual desde: {extractor_path}")
    
    try:
        extractor = torch.load(extractor_path, map_location=device, weights_only=False)
        logger.info("✓ Extractor visual cargado")
    except:
        extractor = ResNet101FeatureExtractor(output_dim=1024)
        extractor.load_state_dict(torch.load(extractor_path, map_location=device, weights_only=False))
        extractor.to(device)
        logger.info("✓ Extractor visual cargado (state_dict)")
    
    extractor.eval()
    return extractor


def load_pose_extractor(extractor_path: Path, device: str):
    """Carga el extractor de pose guardado"""
    
    if not extractor_path.exists():
        raise FileNotFoundError(
            f"Extractor no encontrado: {extractor_path}\n"
            f"   Ejecuta primero: python pipelines/save_extractors.py"
        )
    
    logger.info(f"Cargando extractor de pose desde: {extractor_path}")
    
    try:
        extractor = torch.load(extractor_path, map_location=device, weights_only=False)
        logger.info("✓ Extractor de pose cargado")
    except:
        extractor = PoseFeatureExtractor()
        extractor.load_state_dict(torch.load(extractor_path, map_location=device, weights_only=False))
        extractor.to(device)
        logger.info("✓ Extractor de pose cargado (state_dict)")
    
    extractor.eval()
    return extractor


def precompute_all_features(
    frames_dir: Path,
    keypoints_dir: Path,
    output_fused_dir: Path,
    visual_extractor_path: Path,
    pose_extractor_path: Path,
    device: str = "cuda",
    batch_size: int = 32
):
    """
    Precomputa features visuales, de pose, y las fusiona directamente
    
    Args:
        frames_dir: Directorio con *_frames.npy
        keypoints_dir: Directorio con *_keypoints.npy
        output_fused_dir: Directorio para guardar *_fused.npy
        visual_extractor_path: Ruta al extractor visual
        pose_extractor_path: Ruta al extractor de pose
        device: cuda o cpu
        batch_size: batch size para procesamiento
    """
    output_fused_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # Cargar extractores
    visual_model = load_visual_extractor(visual_extractor_path, device)
    pose_model = load_pose_extractor(pose_extractor_path, device)
    
    # Transform para normalizar frames
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Buscar archivos
    frame_files = sorted(frames_dir.glob("*_frames.npy"))
    logger.info(f"Encontrados {len(frame_files)} archivos para procesar")
    
    if len(frame_files) == 0:
        logger.error(f"No se encontraron archivos *_frames.npy en {frames_dir}")
        return
    
    processed = 0
    skipped = 0
    errors = 0
    
    for frame_file in tqdm(frame_files, desc="Precomputando features"):
        video_id = frame_file.stem.replace("_frames", "")
        keypoints_file = keypoints_dir / f"{video_id}_keypoints.npy"
        output_file = output_fused_dir / f"{video_id}_fused.npy"
        
        # Skip si ya existe
        if output_file.exists():
            skipped += 1
            continue
        
        # Verificar que existan ambos archivos
        if not keypoints_file.exists():
            logger.warning(f"No se encontró keypoints para {video_id}")
            errors += 1
            continue
        
        try:
            # Cargar frames y keypoints
            frames = np.load(frame_file)  # (T, H, W, 3)
            keypoints = np.load(keypoints_file)  # (T, 75, 4)
            
            T = len(frames)
            
            # Validar shapes
            if frames.ndim != 4 or keypoints.ndim != 3:
                logger.warning(f"Shapes inválidos para {video_id}")
                errors += 1
                continue
            
            # Sincronizar longitud
            if len(frames) != len(keypoints):
                logger.warning(f"Longitudes diferentes para {video_id}: frames={len(frames)}, keypoints={len(keypoints)}")
                T_min = min(len(frames), len(keypoints))
                frames = frames[:T_min]
                keypoints = keypoints[:T_min]
                T = T_min
            
            # ========== PROCESAR FRAMES (VISUAL) ==========
            all_visual_features = []
            for i in range(0, T, batch_size):
                batch_frames = frames[i:i+batch_size]
                
                batch_tensors = []
                for frame in batch_frames:
                    if frame.dtype == np.float32 or frame.dtype == np.float64:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                    
                    frame_tensor = transform(frame)
                    batch_tensors.append(frame_tensor)
                
                batch_tensor = torch.stack(batch_tensors).to(device)
                
                with torch.no_grad():
                    features = visual_model(batch_tensor)  # (B, 1024)
                all_visual_features.append(features.cpu().numpy())
            
            visual_features = np.concatenate(all_visual_features, axis=0)  # (T, 1024)
            
            # ========== PROCESAR KEYPOINTS (POSE) ==========
            keypoints_flat = keypoints.reshape(T, -1).astype(np.float32)  # (T, 300)
            
            all_pose_features = []
            for i in range(0, T, batch_size):
                batch_keypoints = keypoints_flat[i:i+batch_size]
                batch_tensor = torch.from_numpy(batch_keypoints).float().to(device)
                
                with torch.no_grad():
                    features = pose_model(batch_tensor)  # (B, 128)
                all_pose_features.append(features.cpu().numpy())
            
            pose_features = np.concatenate(all_pose_features, axis=0)  # (T, 128)
            
            # ========== FUSIONAR ==========
            fused_features = np.concatenate([visual_features, pose_features], axis=1)  # (T, 1152)
            
            # Validar shape final
            assert fused_features.shape == (T, 1152), f"Shape inesperado: {fused_features.shape}"
            
            # Guardar en float16
            fused_features = fused_features.astype(np.float16)
            np.save(output_file, fused_features)
            
            processed += 1
            
        except Exception as e:
            logger.error(f"Error procesando {video_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            errors += 1
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Procesamiento completado!")
    logger.info(f"   ✓ Procesados: {processed}")
    logger.info(f"   ⊘ Omitidos: {skipped}")
    logger.info(f"   ✗ Errores: {errors}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Precomputa features (visual + pose) y las fusiona directamente"
    )
    parser.add_argument(
        "--frames_dir",
        type=Path,
        default=Path("data/processed_frames"),
        help="Directorio con *_frames.npy"
    )
    parser.add_argument(
        "--keypoints_dir",
        type=Path,
        default=Path("data/processed_keypoints"),
        help="Directorio con *_keypoints.npy"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/features_fused"),
        help="Directorio de salida para *_fused.npy"
    )
    parser.add_argument(
        "--visual_extractor",
        type=Path,
        default=Path("models/extractors/visual_extractor_full.pt"),
        help="Ruta al extractor visual"
    )
    parser.add_argument(
        "--pose_extractor",
        type=Path,
        default=Path("models/extractors/pose_extractor_full.pt"),
        help="Ruta al extractor de pose"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device para procesamiento"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size para procesamiento"
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("PRECOMPUTE FEATURES (SIN CLIPS)")
    logger.info("="*60)
    logger.info(f"Frames dir: {args.frames_dir}")
    logger.info(f"Keypoints dir: {args.keypoints_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("="*60 + "\n")
    
    precompute_all_features(
        frames_dir=args.frames_dir,
        keypoints_dir=args.keypoints_dir,
        output_fused_dir=args.output_dir,
        visual_extractor_path=args.visual_extractor,
        pose_extractor_path=args.pose_extractor,
        device=args.device,
        batch_size=args.batch_size
    )
