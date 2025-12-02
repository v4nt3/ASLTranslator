"""
Script maestro para ejecutar todo el pipeline optimizado
Todas las configuraciones se manejan desde config.py
1. Precompute visual features
2. Precompute pose features
3. Fuse features
4. Train temporal model
"""

import subprocess
from pathlib import Path

from config import config
from utils.logging_utils import setup_logging

logger = setup_logging(script_name="run_pipeline")


def run_script(script_name: str, description: str):
    """Ejecuta un script de Python y maneja errores"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🔧 {description}")
    logger.info(f"{'='*60}\n")
    
    try:
        result = subprocess.run(["python", script_name], check=True, capture_output=False, text=True)
        logger.info(f"✅ {description} completado!\n")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Error en {description}")
        logger.error(f"   Código de salida: {e.returncode}")
        return False


def main():
    logger.info("\n" + "="*60)
    logger.info("🚀 PIPELINE OPTIMIZADO DE CLASIFICACIÓN DE SEÑAS")
    logger.info("="*60)
    logger.info("\nConfiguración cargada desde config.py:")
    logger.info(f"  Clips dir: {config.data_paths.clips}")
    logger.info(f"  Features visual: {config.data_paths.features_visual}")
    logger.info(f"  Features pose: {config.data_paths.features_pose}")
    logger.info(f"  Features fused: {config.data_paths.features_fused}")
    logger.info(f"  Model type: {config.training.model_type}")
    logger.info(f"  Use attention: {config.training.use_attention}")
    logger.info(f"  Use augmentation: {config.training.use_augmentation}")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Num epochs: {config.training.num_epochs}")
    logger.info(f"  Device: {config.training.device}")
    logger.info("="*60 + "\n")
    
    success = True
    
    # Step 1: Precompute visual features
    success = run_script("precompute_visual_features.py", "Paso 1: Extracción de features visuales")
    if not success:
        logger.error("Pipeline interrumpido")
        return
    
    # Step 2: Precompute pose features
    success = run_script("precompute_pose_features.py", "Paso 2: Extracción de features de pose")
    if not success:
        logger.error("Pipeline interrumpido")
        return
    
    # Step 3: Fuse features
    success = run_script("fuse_features.py", "Paso 3: Fusión de features")
    if not success:
        logger.error("Pipeline interrumpido")
        return
    
    # Step 4: Train temporal model
    success = run_script("train_temporal.py", "Paso 4: Entrenamiento del modelo temporal")
    if not success:
        logger.error("Pipeline interrumpido")
        return
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    main()
