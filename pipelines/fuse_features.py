"""
Script CORREGIDO para fusionar features visuales y de pose
Crea (T, 1152) = concatenate((T, 1024), (T, 128))
SIN proyección adicional para mantener toda la información
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def safe_delete_file(file_path: Path, max_retries: int = 3) -> bool:
    """
    Intenta borrar un archivo de forma segura con reintentos
    """
    if not file_path.exists():
        return True
    
    for attempt in range(max_retries):
        try:
            file_path.unlink()
            return True
        except PermissionError:
            logger.warning(f" Archivo en uso, reintentando ({attempt+1}/{max_retries}): {file_path.name}")
            import time
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"✗ Error borrando {file_path.name}: {str(e)}")
            return False
    
    return False


def fuse_features(
    visual_dir: Path,
    pose_dir: Path,
    output_dir: Path,
    clips_dir: Path = None,
    delete_originals: bool = False
):
    """
    Fusiona features visuales y de pose mediante concatenación simple
    
    Args:
        visual_dir: Directorio con archivos *_visual.npy
        pose_dir: Directorio con archivos *_pose.npy
        output_dir: Directorio donde guardar *_fused.npy
        clips_dir: Directorio original con frames/keypoints (para borrado)
        delete_originals: Si True, borra archivos originales tras fusionar
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Buscar todos los archivos visuales
    visual_files = sorted(visual_dir.glob("*_visual.npy"))
    logger.info(f"Encontrados {len(visual_files)} archivos visuales")
    
    if len(visual_files) == 0:
        logger.error(f"✗ No se encontraron archivos *_visual.npy en {visual_dir}")
        return
    
    # Contadores
    processed = 0
    skipped = 0
    errors = 0
    deleted_frames = 0
    deleted_keypoints = 0
    
    for visual_file in tqdm(visual_files, desc="Fusionando features"):
        clip_name = visual_file.stem.replace("_visual", "")
        pose_file = pose_dir / f"{clip_name}_pose.npy"
        output_file = output_dir / f"{clip_name}_fused.npy"
        
        # Skip si ya existe
        if output_file.exists():
            skipped += 1
            continue
        
        # Verificar que exista el archivo de pose
        if not pose_file.exists():
            logger.warning(f" No se encontró archivo de pose para {clip_name}")
            errors += 1
            continue
        
        try:
            # Cargar ambos features
            visual_features = np.load(visual_file)  # (T, 1024)
            pose_features = np.load(pose_file)  # (T, 128)
            
            # Validar shapes
            if visual_features.ndim != 2 or pose_features.ndim != 2:
                logger.warning(f" Dimensiones inválidas para {clip_name}")
                logger.warning(f"   Visual: {visual_features.shape}, Pose: {pose_features.shape}")
                errors += 1
                continue
            
            T_visual, D_visual = visual_features.shape
            T_pose, D_pose = pose_features.shape
            
            # Validar dimensiones de features
            if D_visual != 1024:
                logger.warning(f" Dimensión visual inesperada para {clip_name}: {D_visual} (esperado 1024)")
                errors += 1
                continue
            
            if D_pose != 128:
                logger.warning(f" Dimensión pose inesperada para {clip_name}: {D_pose} (esperado 128)")
                errors += 1
                continue
            
            # Sincronizar longitud (usar el mínimo)
            if T_visual != T_pose:
                logger.warning(f" Longitudes diferentes para {clip_name}: visual={T_visual}, pose={T_pose}")
                T_min = min(T_visual, T_pose)
                visual_features = visual_features[:T_min]
                pose_features = pose_features[:T_min]
                logger.warning(f"   → Truncando a T={T_min}")
            
            # Mantiene toda la información: (T, 1024) + (T, 128) = (T, 1152)
            fused_features = np.concatenate([visual_features, pose_features], axis=1)  # (T, 1152)
            
            # Validar shape final
            assert fused_features.shape[1] == 1152, f"Shape inesperado: {fused_features.shape}"
            
            # Guardar en float16 para ahorrar espacio
            fused_features = fused_features.astype(np.float16)
            np.save(output_file, fused_features)
            
            processed += 1
            
            # Borrar archivos originales si se solicitó
            if delete_originals and clips_dir is not None:
                frames_file = clips_dir / f"{clip_name}_frames.npy"
                keypoints_file = clips_dir / f"{clip_name}_keypoints.npy"
                
                if safe_delete_file(frames_file):
                    deleted_frames += 1
                
                if safe_delete_file(keypoints_file):
                    deleted_keypoints += 1
            
        except Exception as e:
            logger.error(f"✗ Error fusionando {clip_name}: {str(e)}")
            errors += 1
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Fusión completada!")
    logger.info(f"   ✓ Procesados: {processed}")
    logger.info(f"   ⊘ Omitidos (ya existían): {skipped}")
    logger.info(f"   ✗ Errores: {errors}")
    
    if delete_originals:
        logger.info(f"\n  Archivos originales borrados:")
        logger.info(f"   Frames: {deleted_frames}")
        logger.info(f"   Keypoints: {deleted_keypoints}")
    
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fusiona features visuales y de pose")
    parser.add_argument(
        "--visual_dir",
        type=Path,
        default=Path("data/features_visual"),
        help="Directorio con archivos *_visual.npy"
    )
    parser.add_argument(
        "--pose_dir",
        type=Path,
        default=Path("data/features_pose"),
        help="Directorio con archivos *_pose.npy"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/features_fused"),
        help="Directorio de salida para *_fused.npy"
    )
    parser.add_argument(
        "--clips_dir",
        type=Path,
        default=Path("data/clips"),
        help="Directorio con clips originales (para borrado)"
    )
    parser.add_argument(
        "--delete_originals",
        action="store_true",
        help="Si se especifica, borra los archivos originales de frames y keypoints"
    )
    
    args = parser.parse_args()
    
    logger.info("Iniciando fusión de features")
    logger.info(f"   Visual dir: {args.visual_dir}")
    logger.info(f"   Pose dir: {args.pose_dir}")
    logger.info(f"   Output dir: {args.output_dir}")
    logger.info(f"   Delete originals: {args.delete_originals}\n")
    
    if args.delete_originals:
        logger.warning(" ATENCIÓN: Se borrarán los archivos originales después de fusionar")
        logger.warning("    Asegúrate de tener backups si es necesario\n")
    
    fuse_features(
        visual_dir=args.visual_dir,
        pose_dir=args.pose_dir,
        output_dir=args.output_dir,
        clips_dir=args.clips_dir,
        delete_originals=args.delete_originals
    )
