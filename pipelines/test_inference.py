"""
Script de test para un solo frame y verificar la extracción de features
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import logging
import sys

sys.path.append(str(Path(__file__).parent.parent))

from pipelines.save_extractors import ResNet101FeatureExtractor, PoseFeatureExtractor
from torchvision import transforms

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def test_frame_extraction(video_path: Path, training_features_path: Path = None):
    """
    Test de extracción de features de un solo frame
    Compara con features de entrenamiento si se proporciona el path
    """
    
    # Cargar extractores
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    visual_extractor = torch.load(
        "models/extractors/visual_extractor_full.pt",
        map_location=device,
        weights_only=False
    )
    visual_extractor.eval()
    
    pose_extractor = torch.load(
        "models/extractors/pose_extractor_full.pt", 
        map_location=device,
        weights_only=False
    )
    pose_extractor.eval()
    
    # Transform idéntico al de entrenamiento
    visual_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Leer un frame del video
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        logger.error("No se pudo leer el frame")
        return
    
    logger.info("="*80)
    logger.info("TEST DE EXTRACCIÓN DE FEATURES")
    logger.info("="*80)
    
    # PASO 1: Análisis del frame original
    logger.info("\n[1] Frame original:")
    logger.info(f"    Shape: {frame.shape}")
    logger.info(f"    Dtype: {frame.dtype}")
    logger.info(f"    Min: {frame.min()}, Max: {frame.max()}, Mean: {frame.mean():.2f}")
    
    # PASO 2: Resize
    frame_resized = cv2.resize(frame, (224, 224))
    logger.info("\n[2] Frame resized (224x224):")
    logger.info(f"    Shape: {frame_resized.shape}")
    logger.info(f"    Dtype: {frame_resized.dtype}")
    logger.info(f"    Min: {frame_resized.min()}, Max: {frame_resized.max()}, Mean: {frame_resized.mean():.2f}")
    
    # PASO 3: BGR -> RGB
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    logger.info("\n[3] Frame RGB:")
    logger.info(f"    Shape: {frame_rgb.shape}")
    logger.info(f"    Dtype: {frame_rgb.dtype}")
    logger.info(f"    Min: {frame_rgb.min()}, Max: {frame_rgb.max()}, Mean: {frame_rgb.mean():.2f}")
    
    # PASO 4: Asegurar uint8 [0, 255]
    if frame_rgb.dtype != np.uint8:
        if frame_rgb.max() <= 1.0:
            logger.warning("    Frame en rango [0, 1], convirtiendo a [0, 255]")
            frame_rgb = (frame_rgb * 255).astype(np.uint8)
        else:
            frame_rgb = frame_rgb.astype(np.uint8)
    
    logger.info("\n[4] Frame uint8:")
    logger.info(f"    Dtype: {frame_rgb.dtype}")
    logger.info(f"    Min: {frame_rgb.min()}, Max: {frame_rgb.max()}, Mean: {frame_rgb.mean():.2f}")
    
    # PASO 5: Aplicar transform (ToTensor + Normalize)
    frame_tensor = visual_transform(frame_rgb).unsqueeze(0).to(device)
    logger.info("\n[5] Frame tensor (después de ToTensor + Normalize):")
    logger.info(f"    Shape: {frame_tensor.shape}")
    logger.info(f"    Dtype: {frame_tensor.dtype}")
    logger.info(f"    Min: {frame_tensor.min().item():.4f}")
    logger.info(f"    Max: {frame_tensor.max().item():.4f}")
    logger.info(f"    Mean: {frame_tensor.mean().item():.4f}")
    logger.info(f"    Std: {frame_tensor.std().item():.4f}")
    
    # PASO 6: Extracción de features visuales
    with torch.no_grad():
        visual_features = visual_extractor(frame_tensor).cpu().numpy().squeeze()
    
    logger.info("\n[6] Visual features (1024 dims):")
    logger.info(f"    Shape: {visual_features.shape}")
    logger.info(f"    Dtype: {visual_features.dtype}")
    logger.info(f"    Min: {visual_features.min():.4f}")
    logger.info(f"    Max: {visual_features.max():.4f}")
    logger.info(f"    Mean: {visual_features.mean():.4f}")
    logger.info(f"    Std: {visual_features.std():.4f}")
    
    # PASO 7: Features de pose (usando keypoints dummy)
    dummy_keypoints = np.random.randn(300).astype(np.float32)
    keypoints_tensor = torch.from_numpy(dummy_keypoints).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        pose_features = pose_extractor(keypoints_tensor).cpu().numpy().squeeze()
    
    logger.info("\n[7] Pose features (128 dims) - DUMMY:")
    logger.info(f"    Shape: {pose_features.shape}")
    logger.info(f"    Dtype: {pose_features.dtype}")
    logger.info(f"    Min: {pose_features.min():.4f}")
    logger.info(f"    Max: {pose_features.max():.4f}")
    logger.info(f"    Mean: {pose_features.mean():.4f}")
    logger.info(f"    Std: {pose_features.std():.4f}")
    
    # PASO 8: Features fusionadas
    fused_features = np.concatenate([visual_features, pose_features]).astype(np.float32)
    logger.info("\n[8] Fused features (1152 dims):")
    logger.info(f"    Shape: {fused_features.shape}")
    logger.info(f"    Dtype: {fused_features.dtype}")
    logger.info(f"    Min: {fused_features.min():.4f}")
    logger.info(f"    Max: {fused_features.max():.4f}")
    logger.info(f"    Mean: {fused_features.mean():.4f}")
    logger.info(f"    Std: {fused_features.std():.4f}")
    
    # COMPARACIÓN con features de entrenamiento
    if training_features_path and Path(training_features_path).exists():
        logger.info("\n" + "="*80)
        logger.info("COMPARACIÓN CON FEATURES DE ENTRENAMIENTO")
        logger.info("="*80)
        
        training_features = np.load(training_features_path).astype(np.float32)
        
        # Tomar el primer frame
        if training_features.ndim == 2:
            training_frame = training_features[0]
        else:
            training_frame = training_features
        
        logger.info(f"\nFeatures de entrenamiento (primer frame):")
        logger.info(f"    Shape: {training_frame.shape}")
        logger.info(f"    Dtype: {training_frame.dtype}")
        logger.info(f"    Min: {training_frame.min():.4f}")
        logger.info(f"    Max: {training_frame.max():.4f}")
        logger.info(f"    Mean: {training_frame.mean():.4f}")
        logger.info(f"    Std: {training_frame.std():.4f}")
        
        logger.info(f"\nDIFERENCIAS:")
        logger.info(f"    Mean diff: {abs(fused_features.mean() - training_frame.mean()):.6f}")
        logger.info(f"    Std diff: {abs(fused_features.std() - training_frame.std()):.6f}")
        logger.info(f"    L2 distance: {np.linalg.norm(fused_features - training_frame):.6f}")
        
        # Cosine similarity
        cos_sim = np.dot(fused_features, training_frame) / (np.linalg.norm(fused_features) * np.linalg.norm(training_frame))
        logger.info(f"    Cosine similarity: {cos_sim:.6f}")
    
    logger.info("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True, help="Video de prueba")
    parser.add_argument("--training_features", type=Path, default=None, help="Features de entrenamiento para comparar (.npy)")
    
    args = parser.parse_args()
    
    test_frame_extraction(args.video, args.training_features)
