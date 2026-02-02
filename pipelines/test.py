"""
Script para verificar que el pipeline de inferencia coincide con entrenamiento
Compara features extraídos de un video con features precomputados
"""

import numpy as np #type: ignore
import cv2 #type: ignore
from pathlib import Path
import logging
from pipelines.inference import ASLInferenceFixed
from pipelines.data_preparation import KeypointExtractor, FrameExtractor
from pipelines.config import config
from pipelines_video.save_extractors import ResNet101FeatureExtractor, PoseFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_training_style_features(video_path: Path, config):
    """
    Extrae features usando EXACTAMENTE el mismo pipeline de entrenamiento
    """
    logger.info(f"Extrayendo features estilo entrenamiento de: {video_path}")
    
    # 1. Extraer frames (igual que FrameExtractor)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_indices = np.linspace(0, total_frames - 1, config.data.num_frames_per_clip, dtype=int)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (config.data.frame_width, config.data.frame_height))
            frames.append(frame)
    
    cap.release()
    frames = np.array(frames)
    logger.info(f"  Frames extraídos: {frames.shape}")
    
    # 2. Extraer keypoints (igual que KeypointExtractor)
    keypoint_extractor = KeypointExtractor(config)
    keypoints_sequence = []
    
    for frame in frames:
        pose_kpts = keypoint_extractor.extract_pose_keypoints(frame)
        
        if pose_kpts is not None:
            pose_kpts = keypoint_extractor.normalize_keypoints(pose_kpts)
            hand_kpts = keypoint_extractor.extract_hand_keypoints(frame)
            
            if hand_kpts is not None:
                combined_kpts = np.concatenate([pose_kpts, hand_kpts], axis=0)
            else:
                hand_placeholder = np.zeros((42, 4))
                combined_kpts = np.concatenate([pose_kpts, hand_placeholder], axis=0)
            
            keypoints_sequence.append(combined_kpts)
    
    keypoints = np.array(keypoints_sequence)
    logger.info(f"  Keypoints extraídos: {keypoints.shape}")
    
    return frames, keypoints


def extract_inference_style_features(video_path: Path, inference_system: ASLInferenceFixed):
    """
    Extrae features usando el pipeline de inferencia
    """
    logger.info(f"Extrayendo features estilo inferencia de: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    keypoints = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_resized = cv2.resize(frame, (224, 224))
        kpts = inference_system.extract_keypoints(frame)
        
        frames.append(frame_resized)
        keypoints.append(kpts)
    
    cap.release()
    
    frames = np.array(frames)
    keypoints = np.array(keypoints)
    
    logger.info(f"  Frames inferencia: {frames.shape}")
    logger.info(f"  Keypoints inferencia: {keypoints.shape}")
    
    return frames, keypoints


def compare_features(train_frames, train_kpts, infer_frames, infer_kpts):
    """
    Compara features extraídos con ambos métodos
    """
    logger.info("\n" + "="*60)
    logger.info("COMPARACIÓN DE FEATURES")
    logger.info("="*60)
    
    # Shapes
    logger.info(f"\nShapes:")
    logger.info(f"  Training frames: {train_frames.shape}")
    logger.info(f"  Inference frames: {infer_frames.shape}")
    logger.info(f"  Training keypoints: {train_kpts.shape}")
    logger.info(f"  Inference keypoints: {infer_kpts.shape}")
    
    # Tomar subset común (primeros N frames)
    N = min(len(train_frames), len(infer_frames))
    train_frames_sub = train_frames[:N]
    infer_frames_sub = infer_frames[:N]
    train_kpts_sub = train_kpts[:N]
    infer_kpts_sub = infer_kpts[:N]
    
    # Comparar frames (MSE)
    # Resize inference frames para comparar
    infer_frames_resized = np.array([
        cv2.resize(f, (train_frames_sub.shape[2], train_frames_sub.shape[1]))
        for f in infer_frames_sub
    ])
    
    frame_mse = np.mean((train_frames_sub.astype(float) - infer_frames_resized.astype(float)) ** 2)
    logger.info(f"\nFrames MSE: {frame_mse:.4f}")
    
    if frame_mse > 100:
        logger.warning(" Frame MSE alto - posible problema en extracción de frames")
    
    # Comparar keypoints
    # Normalizar ambos conjuntos a misma forma
    if train_kpts_sub.shape != infer_kpts_sub.shape:
        logger.warning(f" Shapes de keypoints no coinciden!")
        logger.warning(f"    Training: {train_kpts_sub.shape}")
        logger.warning(f"    Inference: {infer_kpts_sub.shape}")
        
        # Intentar comparar solo los primeros 75 keypoints
        min_kpts = min(train_kpts_sub.shape[1], infer_kpts_sub.shape[1])
        train_kpts_sub = train_kpts_sub[:, :min_kpts, :]
        infer_kpts_sub = infer_kpts_sub[:, :min_kpts, :]
    
    kpts_mse = np.mean((train_kpts_sub - infer_kpts_sub) ** 2)
    logger.info(f"Keypoints MSE: {kpts_mse:.6f}")
    
    if kpts_mse > 0.01:
        logger.warning(" Keypoints MSE alto - posible problema en normalización")
    
    # Comparar estadísticas de keypoints
    logger.info(f"\nEstadísticas de Keypoints:")
    logger.info(f"  Training - mean: {train_kpts_sub.mean():.4f}, std: {train_kpts_sub.std():.4f}")
    logger.info(f"  Inference - mean: {infer_kpts_sub.mean():.4f}, std: {infer_kpts_sub.std():.4f}")
    
    # Verificar visibilidad de manos
    train_hand_vis = train_kpts_sub[:, 33:75, 3].mean()  # Dimensión 3 es visibility
    infer_hand_vis = infer_kpts_sub[:, 33:75, 3].mean()
    
    logger.info(f"\nVisibilidad promedio de manos:")
    logger.info(f"  Training: {train_hand_vis:.4f}")
    logger.info(f"  Inference: {infer_hand_vis:.4f}")
    
    if abs(train_hand_vis - infer_hand_vis) > 0.1:
        logger.warning(" Gran diferencia en visibilidad de manos")
    
    # Comparar normalización (verificar que pose esté centrada)
    train_pose_center = train_kpts_sub[:, :33, :3].mean(axis=(0, 1))
    infer_pose_center = infer_kpts_sub[:, :33, :3].mean(axis=(0, 1))
    
    logger.info(f"\nCentro de pose (debería estar cerca de 0):")
    logger.info(f"  Training: {train_pose_center}")
    logger.info(f"  Inference: {infer_pose_center}")
    
    logger.info("\n" + "="*60 + "\n")
    
    return {
        'frame_mse': frame_mse,
        'kpts_mse': kpts_mse,
        'train_hand_vis': train_hand_vis,
        'infer_hand_vis': infer_hand_vis
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Verifica pipeline de inferencia")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--metadata_path", type=str, required=True)
    parser.add_argument("--visual_extractor", type=str, required=True)
    parser.add_argument("--pose_extractor", type=str, required=True)
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    
    # 1. Extraer features estilo entrenamiento
    train_frames, train_kpts = extract_training_style_features(video_path, config)
    
    # 2. Crear sistema de inferencia
    inference = ASLInference(
        model_path=Path(args.model_path),
        metadata_path=Path(args.metadata_path),
        visual_extractor_path=Path(args.visual_extractor),
        pose_extractor_path=Path(args.pose_extractor),
        device="cuda",
        buffer_size=24,
        confidence_threshold=0.05,
        accumulation_frames=60
    )
    
    # 3. Extraer features estilo inferencia
    infer_frames, infer_kpts = extract_inference_style_features(video_path, inference)
    
    # 4. Comparar
    results = compare_features(train_frames, train_kpts, infer_frames, infer_kpts)
    
    # 5. Diagnóstico
    logger.info("DIAGNÓSTICO:")
    
    issues = []
    if results['frame_mse'] > 100:
        issues.append(" Extracción de frames inconsistente")
    
    if results['kpts_mse'] > 0.01:
        issues.append(" Normalización de keypoints inconsistente")
    
    if abs(results['train_hand_vis'] - results['infer_hand_vis']) > 0.1:
        issues.append(" Detección de manos inconsistente")
    
    if issues:
        logger.error("\nProblemas encontrados:")
        for issue in issues:
            logger.error(f"  {issue}")
    else:
        logger.info("\nPipeline de inferencia parece correcto!")
        logger.info("   El problema podría estar en:")
        logger.info("   - Activity scoring / window selection")
        logger.info("   - Merge de predicciones consecutivas")
        logger.info("   - Umbral de confianza muy bajo")


if __name__ == "__main__":
    main()