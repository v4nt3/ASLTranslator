"""
Script para precomputar features CNN desde keypoints
Version mejorada que usa CNN en lugar de MLP
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import argparse
from pipelines.cnn.save_extractors import  HybridCNNPoseExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_cnn_extractor(extractor_path: Path, device: torch.device):
    """Carga el extractor CNN"""
    
    if not extractor_path.exists():
        raise FileNotFoundError(
            f"Extractor no encontrado: {extractor_path}\n"
            f"Ejecuta primero: python save_extractors_cnn.py"
        )
    
    logger.info(f"Cargando CNN extractor desde: {extractor_path}")
    
    extractor = torch.load(extractor_path, map_location=device, weights_only=False)
    extractor.eval()
    
    logger.info("✓ Extractor CNN cargado correctamente")
    return extractor


def precompute_cnn_features(
    keypoints_dir: Path,
    output_dir: Path,
    extractor_path: Path,
    device: str = 'cuda',
    batch_size: int = 256,
    use_temporal: bool = False,
    temporal_window: int = 16
):
    """
    Precomputa features CNN desde keypoints filtrados
    
    Args:
        keypoints_dir: Directorio con keypoints (*_keypoints.npy)
        output_dir: Directorio de salida (*_pose.npy)
        extractor_path: Path al extractor CNN
        device: 'cuda' o 'cpu'
        batch_size: Batch size
        use_temporal: Si True, usa ventanas temporales (para Temporal/Hybrid)
        temporal_window: Tamaño de ventana temporal
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Usando device: {device}")
    
    # Cargar CNN extractor
    model = load_cnn_extractor(extractor_path, device)
    
    # Buscar archivos
    keypoints_files = sorted(keypoints_dir.glob("**/*_keypoints.npy"))
    
    if len(keypoints_files) == 0:
        logger.error(f"No se encontraron archivos en {keypoints_dir}")
        return
    
    logger.info(f"Encontrados {len(keypoints_files)} archivos")
    logger.info(f"Modo: {'Temporal' if use_temporal else 'Frame-by-frame'}")
    
    processed = 0
    skipped = 0
    errors = 0
    
    for keypoints_file in tqdm(keypoints_files, desc="Extrayendo CNN features"):
        relative_path = keypoints_file.relative_to(keypoints_dir)
        output_file = output_dir / relative_path.parent / f"{keypoints_file.stem.replace('_keypoints', '_pose')}.npy"
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if output_file.exists():
            skipped += 1
            continue
        
        try:
            # Cargar keypoints
            keypoints = np.load(keypoints_file).astype(np.float32)  # (T, 300)
            
            if keypoints.ndim != 2 or keypoints.shape[1] != 300:
                logger.warning(f"Shape inválido {keypoints_file.name}: {keypoints.shape}")
                errors += 1
                continue
            
            T = keypoints.shape[0]
            
            if use_temporal and T >= temporal_window:
                # Modo temporal: procesa ventanas deslizantes
                all_features = []
                
                with torch.no_grad():
                    for i in range(0, T - temporal_window + 1, temporal_window // 2):
                        window = keypoints[i:i + temporal_window]
                        
                        if window.shape[0] < temporal_window:
                            # Pad si es necesario
                            pad_size = temporal_window - window.shape[0]
                            window = np.pad(window, ((0, pad_size), (0, 0)), mode='edge')
                        
                        window_tensor = torch.from_numpy(window).float().unsqueeze(0).to(device)  # (1, T, 300)
                        features = model(window_tensor)  # (1, 128)
                        all_features.append(features.cpu().numpy())
                
                # Promediar features de ventanas
                pose_features = np.mean(all_features, axis=0)  # (1, 128)
                pose_features = np.tile(pose_features, (T, 1))  # (T, 128)
                
            else:
                # Modo frame-by-frame (para Spatial CNN)
                all_features = []
                
                with torch.no_grad():
                    for i in range(0, T, batch_size):
                        batch = keypoints[i:i+batch_size]
                        batch_tensor = torch.from_numpy(batch).float().to(device)
                        features = model(batch_tensor)
                        all_features.append(features.cpu().numpy())
                
                pose_features = np.concatenate(all_features, axis=0)  # (T, 128)
            
            assert pose_features.shape == (T, 128), f"Shape inesperado: {pose_features.shape}"
            
            # Guardar
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
    logger.info(f"  ⊘ Omitidos: {skipped}")
    logger.info(f"  ✗ Errores: {errors}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precomputa features CNN desde keypoints"
    )
    parser.add_argument('--keypoints', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--extractor', type=str, required=True,
                       help='Path al extractor CNN (pose_extractor_cnn_*.pt)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--temporal', action='store_true',
                       help='Usar ventanas temporales (para Temporal/Hybrid CNN)')
    parser.add_argument('--temporal_window', type=int, default=16)
    
    args = parser.parse_args()
    
    precompute_cnn_features(
        keypoints_dir=Path(args.keypoints),
        output_dir=Path(args.output),
        extractor_path=Path(args.extractor),
        device=args.device,
        batch_size=args.batch_size,
        use_temporal=args.temporal,
        temporal_window=args.temporal_window
    )
