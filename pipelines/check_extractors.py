"""
Script para verificar si los extractors actuales coinciden con los usados en entrenamiento
"""

import torch
import numpy as np
from pathlib import Path
import logging
import argparse
import json
from pipelines.save_extractors import ResNet101FeatureExtractor, PoseFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def compute_feature_hash(features: np.ndarray) -> str:
    """Computa un hash simple de las features para comparación"""
    return f"mean={features.mean():.6f}_std={features.std():.6f}_min={features.min():.6f}_max={features.max():.6f}"


def check_extractor_consistency(
    visual_extractor_path: Path,
    pose_extractor_path: Path,
    sample_features_dir: Path
):
    """
    Verifica si los extractors actuales producen features consistentes
    con las precomputadas
    """
    
    logger.info("="*80)
    logger.info("VERIFICANDO CONSISTENCIA DE EXTRACTORS")
    logger.info("="*80)
    
    # Cargar extractors
    logger.info(f"\nCargando extractors...")
    logger.info(f"  Visual: {visual_extractor_path}")
    logger.info(f"  Pose: {pose_extractor_path}")
    
    if not visual_extractor_path.exists():
        logger.error(f"❌ No se encontró extractor visual: {visual_extractor_path}")
        return False
    
    if not pose_extractor_path.exists():
        logger.error(f"❌ No se encontró extractor de pose: {pose_extractor_path}")
        return False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    visual_extractor = torch.load(visual_extractor_path, map_location=device, weights_only=False)
    pose_extractor = torch.load(pose_extractor_path, map_location=device, weights_only=False)
    
    visual_extractor.eval()
    pose_extractor.eval()
    
    logger.info("✓ Extractors cargados")
    
    # Analizar pesos del extractor visual
    logger.info("\n" + "="*80)
    logger.info("ANÁLISIS DE PESOS - VISUAL EXTRACTOR")
    logger.info("="*80)
    
    projection_weight = visual_extractor.projection.weight.data.cpu().numpy()
    projection_bias = visual_extractor.projection.bias.data.cpu().numpy()
    
    logger.info(f"\nProjection layer (Linear 2048 -> 1024):")
    logger.info(f"  Weight shape: {projection_weight.shape}")
    logger.info(f"  Weight stats: {compute_feature_hash(projection_weight)}")
    logger.info(f"  Bias shape: {projection_bias.shape}")
    logger.info(f"  Bias stats: {compute_feature_hash(projection_bias)}")
    
    # Analizar pesos del extractor de pose
    logger.info("\n" + "="*80)
    logger.info("ANÁLISIS DE PESOS - POSE EXTRACTOR")
    logger.info("="*80)
    
    for name, param in pose_extractor.named_parameters():
        param_data = param.data.cpu().numpy()
        logger.info(f"\n{name}:")
        logger.info(f"  Shape: {param_data.shape}")
        logger.info(f"  Stats: {compute_feature_hash(param_data)}")
    
    # Buscar archivos de features precomputadas
    logger.info("\n" + "="*80)
    logger.info("BUSCANDO FEATURES PRECOMPUTADAS")
    logger.info("="*80)
    
    if not sample_features_dir.exists():
        logger.warning(f"⚠️  Directorio no encontrado: {sample_features_dir}")
        logger.info("\nNo se pueden comparar con features de entrenamiento")
        return True
    
    # Buscar algunos archivos .npy
    npy_files = list(sample_features_dir.rglob("*.npy"))[:5]
    
    if not npy_files:
        logger.warning(f"⚠️  No se encontraron archivos .npy en: {sample_features_dir}")
        logger.info("\nNo se pueden comparar con features de entrenamiento")
        return True
    
    logger.info(f"\nEncontrados {len(npy_files)} archivos de muestra")
    
    # Analizar estadísticas de features precomputadas
    logger.info("\n" + "="*80)
    logger.info("ESTADÍSTICAS DE FEATURES PRECOMPUTADAS")
    logger.info("="*80)
    
    all_means = []
    all_stds = []
    
    for npy_file in npy_files:
        features = np.load(npy_file).astype(np.float32)
        all_means.append(features.mean())
        all_stds.append(features.std())
        
        logger.info(f"\n{npy_file.name}:")
        logger.info(f"  Shape: {features.shape}")
        logger.info(f"  Stats: {compute_feature_hash(features)}")
        
        # Separar visual y pose
        if features.shape[1] == 1152:
            visual_features = features[:, :1024]
            pose_features = features[:, 1024:]
            
            logger.info(f"  Visual: {compute_feature_hash(visual_features)}")
            logger.info(f"  Pose: {compute_feature_hash(pose_features)}")
    
    # Resumen
    logger.info("\n" + "="*80)
    logger.info("RESUMEN")
    logger.info("="*80)
    
    avg_mean = np.mean(all_means)
    avg_std = np.mean(all_stds)
    
    logger.info(f"\nEstadísticas promedio de {len(npy_files)} muestras:")
    logger.info(f"  Mean promedio: {avg_mean:.6f}")
    logger.info(f"  Std promedio: {avg_std:.6f}")
    
    logger.info("\n" + "="*80)
    logger.info("RECOMENDACIONES")
    logger.info("="*80)
    
    logger.info("\n1. GUARDAR HASH DE EXTRACTORS ACTUALES:")
    logger.info("   Anota estos valores para referencia futura:")
    logger.info(f"   Visual projection: {compute_feature_hash(projection_weight)}")
    logger.info(f"   Features promedio: mean={avg_mean:.6f}, std={avg_std:.6f}")
    
    logger.info("\n2. SI LAS PREDICTIONS SON MALAS:")
    logger.info("   - Opción A: Re-precomputar TODAS las features con estos extractors")
    logger.info("   - Opción B: Usar los extractors ORIGINALES del entrenamiento")
    
    logger.info("\n3. CÓMO IDENTIFICAR SI HAY PROBLEMA:")
    logger.info("   Ejecuta: python scripts/validate_preprocessing.py")
    logger.info("   Si la diferencia absoluta > 0.05, hay inconsistencia")
    
    logger.info("\n" + "="*80)
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Verifica consistencia de extractors vs features precomputadas"
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
        "--sample_features",
        type=Path,
        default=Path("data/features_fused"),
        help="Directorio con features precomputadas"
    )
    
    args = parser.parse_args()
    
    check_extractor_consistency(
        args.visual_extractor,
        args.pose_extractor,
        args.sample_features
    )


if __name__ == "__main__":
    main()
