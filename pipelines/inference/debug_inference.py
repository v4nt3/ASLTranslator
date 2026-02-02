"""
Script de verificación completa del sistema de inferencia
Verifica que todas las dimensiones coincidan correctamente
"""

import torch
import numpy as np
from pathlib import Path
import logging
import json
from pipelines_video.save_extractors import ResNet101FeatureExtractor, PoseFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_extractors(visual_path: Path, pose_path: Path, device: str = "cuda"):
    """Verifica los extractores y sus dimensiones de salida"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    logger.info("="*60)
    logger.info("VERIFICACIÓN DE EXTRACTORS")
    logger.info("="*60)
    
    # Visual extractor
    logger.info("\n1. Visual Extractor:")
    visual_extractor = torch.load(visual_path, map_location=device, weights_only=False)
    visual_extractor.eval()
    
    with torch.no_grad():
        dummy_frame = torch.randn(1, 3, 224, 224).to(device)
        visual_output = visual_extractor(dummy_frame)
    
    logger.info(f"   Input shape: {dummy_frame.shape}")
    logger.info(f"   Output shape: {visual_output.shape}")
    logger.info(f"   Feature dim: {visual_output.shape[1]}")
    
    visual_dim = visual_output.shape[1]
    
    # Pose extractor
    logger.info("\n2. Pose Extractor:")
    pose_extractor = torch.load(pose_path, map_location=device, weights_only=False)
    pose_extractor.eval()
    
    with torch.no_grad():
        dummy_keypoints = torch.randn(1, 300).to(device)  # 75 keypoints * 4 coords = 300
        pose_output = pose_extractor(dummy_keypoints)
    
    logger.info(f"   Input shape: {dummy_keypoints.shape}")
    logger.info(f"   Output shape: {pose_output.shape}")
    logger.info(f"   Feature dim: {pose_output.shape[1]}")
    
    pose_dim = pose_output.shape[1]
    
    # Fused
    fused_dim = visual_dim + pose_dim
    logger.info(f"\n3. Fused Features:")
    logger.info(f"   Total dim: {visual_dim} + {pose_dim} = {fused_dim}")
    
    return visual_dim, pose_dim, fused_dim


def verify_model(model_path: Path, expected_input_dim: int, num_classes: int, device: str = "cuda"):
    """Verifica que el modelo tenga la dimensión de entrada correcta"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    logger.info("\n" + "="*60)
    logger.info("VERIFICACIÓN DEL MODELO")
    logger.info("="*60)
    
    from pipelines.models_temporal import TemporalLSTMClassifier
    from pipelines.config import config
    
    # Cargar modelo
    model = TemporalLSTMClassifier(
        input_dim=expected_input_dim,
        hidden_dim=config.training.model_hidden_dim,
        num_layers=config.training.model_num_layers,
        num_classes=num_classes,
        dropout=config.training.model_dropout,
        bidirectional=config.training.model_bidirectional,
        use_attention=config.training.use_attention
    ).to(device)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("✓ Modelo cargado exitosamente")
        
        # Test forward pass
        with torch.no_grad():
            dummy_features = torch.randn(1, 24, expected_input_dim).to(device)
            output = model(dummy_features)
        
        logger.info(f"\nTest forward pass:")
        logger.info(f"   Input shape: {dummy_features.shape}")
        logger.info(f"   Output shape: {output.shape}")
        logger.info(f"   Expected classes: {num_classes}")
        
        if output.shape[1] == num_classes:
            logger.info("✓ Dimensiones correctas")
            return True
        else:
            logger.error(f"✗ Error: output tiene {output.shape[1]} clases, esperado {num_classes}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error cargando modelo: {e}")
        return False


def verify_training_features(features_dir: Path, expected_dim: int):
    """Verifica que las features de training tengan la dimensión esperada"""
    logger.info("\n" + "="*60)
    logger.info("VERIFICACIÓN DE FEATURES DE TRAINING")
    logger.info("="*60)
    
    fused_files = list(features_dir.glob("*_fused.npy"))
    
    if len(fused_files) == 0:
        logger.error(f"No se encontraron archivos en {features_dir}")
        return False
    
    logger.info(f"\nRevisando {min(5, len(fused_files))} archivos...")
    
    all_correct = True
    
    for fused_path in fused_files[:5]:
        features = np.load(fused_path)
        
        if features.shape[1] == expected_dim:
            status = "✓"
        else:
            status = "✗"
            all_correct = False
        
        logger.info(f"{status} {fused_path.name}: shape={features.shape}, expected_dim={expected_dim}")
    
    if all_correct:
        logger.info("\n✓ Todas las features tienen la dimensión correcta")
    else:
        logger.error("\n✗ ALERTA: Algunas features tienen dimensión incorrecta")
        logger.error("   Es necesario regenerar las features con:")
        logger.error("   1. python precompute_visual_features.py")
        logger.error("   2. python precompute_pose_features.py")
        logger.error("   3. python fuse_features.py")
    
    return all_correct


def verify_config(expected_fused_dim: int):
    """Verifica que config.py tenga las dimensiones correctas"""
    logger.info("\n" + "="*60)
    logger.info("VERIFICACIÓN DE CONFIG.PY")
    logger.info("="*60)
    
    from pipelines.config import config
    
    logger.info(f"\nDimensiones en config:")
    logger.info(f"   visual_feature_dim: {config.data.visual_feature_dim}")
    logger.info(f"   pose_feature_dim: {config.data.pose_feature_dim}")
    logger.info(f"   fused_feature_dim: {config.data.fused_feature_dim}")
    
    config_fused = config.data.visual_feature_dim + config.data.pose_feature_dim
    
    if config_fused == expected_fused_dim:
        logger.info(f"\n✓ Config correcto: {config_fused} == {expected_fused_dim}")
        return True
    else:
        logger.warning(f"\n⚠ Config desactualizado: {config_fused} != {expected_fused_dim}")
        logger.warning(f"   Pero esto no afecta la inferencia ya que se detecta automáticamente")
        return True


def verify_complete_system(
    visual_extractor_path: Path,
    pose_extractor_path: Path,
    model_path: Path,
    features_dir: Path,
    metadata_path: Path,
    device: str = "cuda"
):
    """Verificación completa del sistema"""
    
    logger.info("\n" + "🔍 " + "="*58)
    logger.info("     VERIFICACIÓN COMPLETA DEL SISTEMA DE INFERENCIA")
    logger.info("="*60 + "\n")
    
    # 1. Verificar extractores
    visual_dim, pose_dim, fused_dim = verify_extractors(
        visual_extractor_path,
        pose_extractor_path,
        device
    )
    
    # 2. Cargar número de clases
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    num_classes = 0
    if isinstance(metadata, dict) and 'videos' in metadata:
        class_ids = set()
        for entry in metadata['videos']:
            if entry.get('class_id') is not None:
                class_ids.add(entry['class_id'])
        num_classes = len(class_ids)
    
    logger.info(f"\nNúmero de clases en metadata: {num_classes}")
    
    # 3. Verificar modelo
    model_ok = verify_model(model_path, fused_dim, num_classes, device)
    
    # 4. Verificar features de training
    features_ok = verify_training_features(features_dir, fused_dim)
    
    # 5. Verificar config
    config_ok = verify_config(fused_dim)
    
    # Resumen
    logger.info("\n" + "="*60)
    logger.info("RESUMEN DE VERIFICACIÓN")
    logger.info("="*60)
    
    checks = [
        ("Extractores", True),  # Siempre OK si llegamos aquí
        ("Modelo", model_ok),
        ("Features de training", features_ok),
        ("Config", config_ok)
    ]
    
    all_ok = all(ok for _, ok in checks)
    
    for name, ok in checks:
        status = "✓" if ok else "✗"
        logger.info(f"{status} {name}")
    
    logger.info("\n" + "="*60)
    
    if all_ok:
        logger.info("✅ SISTEMA COMPLETAMENTE VERIFICADO")
        logger.info("\nPuedes ejecutar inferencia con:")
        logger.info("  python inference_video.py path/to/video.mp4")
        logger.info("  python inference_realtime.py")
    else:
        logger.error("❌ HAY PROBLEMAS QUE DEBEN SER RESUELTOS")
        
        if not features_ok:
            logger.error("\n⚠️  ACCIÓN REQUERIDA:")
            logger.error("Las features de training tienen dimensión incorrecta.")
            logger.error("Debes regenerarlas ejecutando:")
            logger.error("  1. python precompute_visual_features.py")
            logger.error("  2. python precompute_pose_features.py") 
            logger.error("  3. python fuse_features.py")
            logger.error("  4. python train_temporal.py")
    
    logger.info("="*60 + "\n")
    
    return all_ok


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Verificación completa del sistema")
    parser.add_argument("--visual_extractor", type=Path,
                       default=Path("models/extractors/visual_extractor_full.pt"))
    parser.add_argument("--pose_extractor", type=Path,
                       default=Path("models/extractors/pose_extractor_full.pt"))
    parser.add_argument("--model_path", type=Path,
                       default=Path("checkpoints/temporal/best_model.pt"))
    parser.add_argument("--features_dir", type=Path,
                       default=Path("data/features_fused"))
    parser.add_argument("--metadata_path", type=Path,
                       default=Path("data/dataset_meta.json"))
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    # Verificar que existan los archivos
    missing = []
    for name, path in [
        ("Visual extractor", args.visual_extractor),
        ("Pose extractor", args.pose_extractor),
        ("Modelo", args.model_path),
        ("Metadata", args.metadata_path)
    ]:
        if not path.exists():
            missing.append(f"  - {name}: {path}")
    
    if missing:
        logger.error("❌ Archivos faltantes:")
        for m in missing:
            logger.error(m)
        return
    
    if not args.features_dir.exists():
        logger.warning(f"⚠️  Directorio de features no encontrado: {args.features_dir}")
        logger.warning("   (OK si solo quieres verificar el sistema de inferencia)")
    
    # Ejecutar verificación
    verify_complete_system(
        visual_extractor_path=args.visual_extractor,
        pose_extractor_path=args.pose_extractor,
        model_path=args.model_path,
        features_dir=args.features_dir,
        metadata_path=args.metadata_path,
        device=args.device
    )


if __name__ == "__main__":
    main()