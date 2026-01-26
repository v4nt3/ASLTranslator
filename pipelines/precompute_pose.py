"""
Script para precomputar features de pose desde keypoints FILTRADOS
Convierte keypoints filtrados (T, 300) → pose features (T, 128) usando MLP
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import argparse
from pipelines_video.save_extractors import PoseFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_pose_extractor(extractor_path: Path, device: torch.device):
    """Carga el extractor de pose MLP"""
    
    if not extractor_path.exists():
        raise FileNotFoundError(
            f"Extractor no encontrado: {extractor_path}\n"
            f"Ejecuta primero: python scripts/save_extractors.py"
        )
    
    logger.info(f"Cargando MLP pose extractor desde: {extractor_path}")
    
    # Cargar el modelo completo
    extractor = torch.load(extractor_path, map_location=device, weights_only=False)
    extractor.eval()
    
    logger.info("✓ Extractor cargado correctamente")
    return extractor


def precompute_pose_features(
    keypoints_dir: Path,
    output_dir: Path,
    extractor_path: Path,
    device: str = 'cuda',
    batch_size: int = 256
):
    """
    Precomputa pose features desde keypoints filtrados
    
    Args:
        keypoints_dir: Directorio con keypoints filtrados (*_keypoints_filtered.npy)
        output_dir: Directorio donde guardar pose features (*_pose.npy)
        extractor_path: Path al MLP pose extractor
        device: 'cuda' o 'cpu'
        batch_size: Batch size para procesamiento
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Usando device: {device}")
    
    # Cargar MLP extractor
    model = load_pose_extractor(extractor_path, device)
    
    # Buscar todos los archivos de keypoints filtrados
    keypoints_files = sorted(keypoints_dir.glob("**/*_keypoints.npy"))
    
    if len(keypoints_files) == 0:
        logger.error(f"No se encontraron archivos *_keypoints.npy en {keypoints_dir}")
        return
    
    logger.info(f"Encontrados {len(keypoints_files)} archivos para procesar")
    
    # Procesar cada archivo
    processed = 0
    skipped = 0
    errors = 0
    
    for keypoints_file in tqdm(keypoints_files, desc="Extrayendo pose features"):
        # Mantener estructura de directorios
        relative_path = keypoints_file.relative_to(keypoints_dir)
        output_file = output_dir / relative_path.parent / f"{keypoints_file.stem.replace('_keypoints', '_pose')}.npy"
        
        # Crear directorio si no existe
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip si ya existe
        if output_file.exists():
            skipped += 1
            continue
        
        try:
            # Cargar keypoints filtrados
            keypoints = np.load(keypoints_file).astype(np.float32)  # (T, 300)
            
            # Validar shape
            if keypoints.ndim != 2:
                logger.warning(f"Shape inesperado para {keypoints_file.name}: {keypoints.shape}")
                errors += 1
                continue
            
            T, features_dim = keypoints.shape
            
            if features_dim != 300:
                logger.warning(f"Dimensión inesperada para {keypoints_file.name}: {features_dim} (esperado 300)")
                errors += 1
                continue
            
            # Procesar en batches
            all_features = []
            
            with torch.no_grad():
                for i in range(0, T, batch_size):
                    batch_keypoints = keypoints[i:i+batch_size]
                    batch_tensor = torch.from_numpy(batch_keypoints).float().to(device)
                    
                    # Extraer features con MLP (300 → 128)
                    features = model(batch_tensor)  # (B, 128)
                    all_features.append(features.cpu().numpy())
            
            # Concatenar
            pose_features = np.concatenate(all_features, axis=0)  # (T, 128)
            
            # Validar output shape
            assert pose_features.shape == (T, 128), f"Output shape inesperado: {pose_features.shape}"
            
            # Guardar en float16 (como en entrenamiento)
            pose_features = pose_features.astype(np.float16)
            np.save(output_file, pose_features)
            
            processed += 1
            
        except Exception as e:
            logger.error(f"Error procesando {keypoints_file.name}: {str(e)}")
            errors += 1
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Procesamiento completado!")
    logger.info(f"  ✓ Procesados: {processed}")
    logger.info(f"  ⊘ Omitidos (ya existían): {skipped}")
    logger.info(f"  ✗ Errores: {errors}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precomputa pose features desde keypoints filtrados usando MLP"
    )
    parser.add_argument(
        '--keypoints',
        type=str,
        required=True,
        help='Directorio con keypoints filtrados (*_keypoints.npy)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Directorio de salida para pose features (*_pose.npy)'
    )
    parser.add_argument(
        '--pose_extractor',
        type=str,
        required=True,
        help='Path al MLP pose extractor (pose_extractor_full.pt)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device para procesamiento'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size para procesamiento'
    )
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("PRECOMPUTE POSE FEATURES DESDE KEYPOINTS FILTRADOS")
    logger.info("="*60)
    logger.info(f"Keypoints filtrados: {args.keypoints}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Pose extractor: {args.pose_extractor}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("="*60 + "\n")
    
    precompute_pose_features(
        keypoints_dir=Path(args.keypoints),
        output_dir=Path(args.output),
        extractor_path=Path(args.pose_extractor),
        device=args.device,
        batch_size=args.batch_size
    )
