"""
Script para precomputar features de pose usando MLP frozen
Convierte keypoints (T, 300) → (T, 128) en float16
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

from config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PoseFeatureExtractor(nn.Module):
    """Extractor de features de pose con MLP frozen"""
    
    def __init__(self, input_dim=None, hidden_dim=256, output_dim=None):
        super().__init__()
        
        if input_dim is None:
            input_dim = config.model.keypoints_dim
        if output_dim is None:
            output_dim = config.data.pose_feature_dim
        
        # MLP: 300 → 256 → 128
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
        # Freeze todo
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
        
        logger.info(f"✓ Pose MLP feature extractor inicializado (frozen)")
    
    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: (B, 300) tensor de keypoints
        Returns:
            features: (B, 128)
        """
        return self.mlp(x)


def precompute_pose_features(
    clips_dir: Path = None,
    output_dir: Path = None,
    device: str = None,
    batch_size: int = 256
):
    """
    Precomputa features de pose para todos los clips
    
    Args:
        clips_dir: Directorio con archivos *_keypoints.npy (usa config si es None)
        output_dir: Directorio donde guardar *_pose.npy (usa config si es None)
        device: 'cuda' o 'cpu' (usa config si es None)
        batch_size: Batch size para procesamiento
    """
    
    if clips_dir is None:
        clips_dir = config.data_paths.clips
    if output_dir is None:
        output_dir = config.data_paths.features_pose
    if device is None:
        device = config.training.device
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Inicializar modelo
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = PoseFeatureExtractor().to(device)
    
    # Buscar todos los archivos _keypoints.npy
    keypoints_files = sorted(clips_dir.glob("*_keypoints.npy"))
    logger.info(f" Encontrados {len(keypoints_files)} clips para procesar")
    
    if len(keypoints_files) == 0:
        logger.error(f" No se encontraron archivos *_keypoints.npy en {clips_dir}")
        return
    
    # Procesar cada clip
    processed = 0
    skipped = 0
    errors = 0
    
    for keypoints_file in tqdm(keypoints_files, desc=" Extrayendo features de pose"):
        clip_name = keypoints_file.stem.replace("_keypoints", "")
        output_file = output_dir / f"{clip_name}_pose.npy"
        
        # Skip si ya existe
        if output_file.exists():
            skipped += 1
            continue
        
        try:
            # Cargar keypoints
            keypoints = np.load(keypoints_file).astype(np.float32)  # (T, N, D) o (T, N*D)
            
            # Reshape si es necesario
            if keypoints.ndim == 3:
                T, N, D = keypoints.shape
                keypoints = keypoints.reshape(T, N * D)
            
            # Validar shape
            T, features_dim = keypoints.shape
            if features_dim != config.model.keypoints_dim:
                logger.warning(f" Dimensión de keypoints inesperada para {clip_name}: {features_dim} (esperado {config.model.keypoints_dim})")
                errors += 1
                continue
            
            all_features = []
            
            # Procesar en batches
            for i in range(0, T, batch_size):
                batch_keypoints = keypoints[i:i+batch_size]
                batch_tensor = torch.from_numpy(batch_keypoints).float().to(device)
                
                # Extraer features
                features = model(batch_tensor)  # (B, 128)
                all_features.append(features.cpu().numpy())
            
            # Concatenar todos los features
            pose_features = np.concatenate(all_features, axis=0)  # (T, 128)
            
            # Validar shape
            assert pose_features.shape == (T, config.data.pose_feature_dim), f"Shape inesperado: {pose_features.shape}"
            
            # Guardar en float16 para ahorrar espacio
            pose_features = pose_features.astype(np.float16)
            np.save(output_file, pose_features)
            
            processed += 1
            
        except Exception as e:
            logger.error(f" Error procesando {clip_name}: {str(e)}")
            errors += 1
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Procesamiento completado!")
    logger.info(f"   ✓ Procesados: {processed}")
    logger.info(f"   ⊘ Omitidos (ya existían): {skipped}")
    logger.info(f"   ✗ Errores: {errors}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Precomputa features de pose con MLP")
    parser.add_argument(
        "--clips_dir",
        type=Path,
        default=None,
        help=f"Directorio con archivos *_keypoints.npy (default: {config.data_paths.clips})"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=f"Directorio de salida para *_pose.npy (default: {config.data_paths.features_pose})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help=f"Device para procesamiento (default: {config.training.device})"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size para procesamiento"
    )
    
    args = parser.parse_args()
    
    logger.info("Iniciando precompute de features de pose")
    logger.info(f"   Clips dir: {args.clips_dir or config.data_paths.clips}")
    logger.info(f"   Output dir: {args.output_dir or config.data_paths.features_pose}")
    logger.info(f"   Device: {args.device or config.training.device}")
    logger.info(f"   Batch size: {args.batch_size}\n")
    
    precompute_pose_features(
        clips_dir=args.clips_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size
    )
