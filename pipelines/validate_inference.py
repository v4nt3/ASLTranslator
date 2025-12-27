"""
Script de validación detallada del preprocesamiento
Compara paso a paso el pipeline de entrenamiento vs inferencia
"""

import torch
import numpy as np
import cv2
from pathlib import Path
import logging
from torchvision import transforms
import json
import mediapipe as mp
import sys

sys.path.append(str(Path(__file__).parent.parent))

from pipelines.save_extractors import ResNet101FeatureExtractor, PoseFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreprocessingValidator:
    """Valida que el preprocesamiento sea idéntico entre entrenamiento e inferencia"""
    
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Cargar extractores
        self.visual_extractor = torch.load(
            "models/extractors/visual_extractor_full.pt",
            map_location=self.device,
            weights_only=False
        )
        self.visual_extractor.eval()
        
        self.pose_extractor = torch.load(
            "models/extractors/pose_extractor_full.pt",
            map_location=self.device,
            weights_only=False
        )
        self.pose_extractor.eval()
        
        # Transform EXACTAMENTE como en precompute_visual_features.py
        self.visual_transform = transforms.Compose([
            transforms.ToTensor(),  # Convierte [0, 255] uint8 → [0, 1] float32
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
    
    def validate_training_features(self, training_features_path: Path):
        """Carga y valida features de entrenamiento"""
        logger.info("\n" + "="*60)
        logger.info("VALIDANDO FEATURES DE ENTRENAMIENTO")
        logger.info("="*60)
        
        # Cargar features (guardadas como float16, cargadas como float32)
        features = np.load(training_features_path)  # Shape: (T, 1152)
        
        logger.info(f"Shape: {features.shape}")
        logger.info(f"Dtype (en disco): float16")
        logger.info(f"Dtype (cargado): {features.dtype}")
        logger.info(f"Min: {features.min():.4f}")
        logger.info(f"Max: {features.max():.4f}")
        logger.info(f"Mean: {features.mean():.4f}")
        logger.info(f"Std: {features.std():.4f}")
        
        # Separar visual y pose
        visual_features = features[:, :1024]
        pose_features = features[:, 1024:]
        
        logger.info(f"\nVisual features (primeros 1024 dims):")
        logger.info(f"  Shape: {visual_features.shape}")
        logger.info(f"  Mean: {visual_features.mean():.4f}")
        logger.info(f"  Std: {visual_features.std():.4f}")
        
        logger.info(f"\nPose features (últimos 128 dims):")
        logger.info(f"  Shape: {pose_features.shape}")
        logger.info(f"  Mean: {pose_features.mean():.4f}")
        logger.info(f"  Std: {pose_features.std():.4f}")
        
        return features
    
    def extract_keypoints_from_frame(self, frame: np.ndarray):
        """Extrae keypoints usando MediaPipe"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose
        pose_results = self.pose.process(frame_rgb)
        if not pose_results.pose_landmarks:
            return None
        
        pose_kpts = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in pose_results.pose_landmarks.landmark
        ])
        
        pose_kpts = self._normalize_keypoints(pose_kpts)
        
        # Hands
        hand_results = self.hands.process(frame_rgb)
        hand_kpts = np.zeros((42, 4))
        
        if hand_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks[:2]):
                start_idx = hand_idx * 21
                hand_kpts[start_idx:start_idx+21] = np.array([
                    [lm.x, lm.y, lm.z, 1.0]
                    for lm in hand_landmarks.landmark
                ])
        
        combined_kpts = np.concatenate([pose_kpts, hand_kpts], axis=0)
        return combined_kpts
    
    def _normalize_keypoints(self, pose_kpts: np.ndarray) -> np.ndarray:
        """Normaliza keypoints de pose"""
        left_hip = pose_kpts[23][:3]
        right_hip = pose_kpts[24][:3]
        center = (left_hip + right_hip) / 2
        
        shoulder_dist = np.linalg.norm(pose_kpts[11][:3] - pose_kpts[12][:3])
        scale = max(shoulder_dist, 0.1)
        
        normalized = pose_kpts.copy()
        normalized[:, :3] = (pose_kpts[:, :3] - center) / scale
        normalized[:, :3] = np.clip(normalized[:, :3], -1, 1)
        
        return normalized
    
    def extract_features_from_video(self, video_path: Path):
        """Extrae features de un video usando el pipeline de inferencia"""
        logger.info("\n" + "="*60)
        logger.info("EXTRAYENDO FEATURES CON PIPELINE DE INFERENCIA")
        logger.info("="*60)
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"No se pudo abrir el video: {video_path}")
            return None
        
        all_features = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extraer keypoints
            keypoints = self.extract_keypoints_from_frame(frame)
            if keypoints is None:
                continue
            
            # === PROCESAMIENTO VISUAL ===
            logger.info(f"\n--- Frame {frame_count} ---")
            logger.info(f"Original frame shape: {frame.shape}, dtype: {frame.dtype}")
            
            # Resize a 224x224 (EXACTAMENTE como en training)
            frame_resized = cv2.resize(frame, (224, 224))
            logger.info(f"After resize: {frame_resized.shape}")
            
            # Convertir BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            logger.info(f"After BGR->RGB: {frame_rgb.shape}, dtype: {frame_rgb.dtype}")
            logger.info(f"  Value range: [{frame_rgb.min()}, {frame_rgb.max()}], mean: {frame_rgb.mean():.2f}")
            
            # Asegurar uint8 en rango [0, 255]
            if frame_rgb.dtype != np.uint8:
                if frame_rgb.max() <= 1.0:
                    frame_rgb = (frame_rgb * 255).astype(np.uint8)
                else:
                    frame_rgb = frame_rgb.astype(np.uint8)
            
            logger.info(f"After uint8 conversion: dtype: {frame_rgb.dtype}")
            logger.info(f"  Value range: [{frame_rgb.min()}, {frame_rgb.max()}], mean: {frame_rgb.mean():.2f}")
            
            # Aplicar transform (ToTensor + Normalize)
            frame_tensor = self.visual_transform(frame_rgb).unsqueeze(0).to(self.device)
            logger.info(f"After transform: shape: {frame_tensor.shape}, dtype: {frame_tensor.dtype}")
            logger.info(f"  Value range: [{frame_tensor.min().item():.4f}, {frame_tensor.max().item():.4f}]")
            logger.info(f"  Mean: {frame_tensor.mean().item():.4f}, Std: {frame_tensor.std().item():.4f}")
            
            # Extraer visual features
            with torch.no_grad():
                visual_features = self.visual_extractor(frame_tensor).cpu().numpy().squeeze()
            
            logger.info(f"Visual features: shape: {visual_features.shape}, dtype: {visual_features.dtype}")
            logger.info(f"  Mean: {visual_features.mean():.4f}, Std: {visual_features.std():.4f}")
            
            # === PROCESAMIENTO POSE ===
            keypoints_flat = keypoints.flatten()[:300]
            logger.info(f"\nKeypoints: shape: {keypoints_flat.shape}, dtype: {keypoints_flat.dtype}")
            logger.info(f"  Mean: {keypoints_flat.mean():.4f}, Std: {keypoints_flat.std():.4f}")
            
            keypoints_tensor = torch.from_numpy(keypoints_flat).unsqueeze(0).float().to(self.device)
            
            with torch.no_grad():
                pose_features = self.pose_extractor(keypoints_tensor).cpu().numpy().squeeze()
            
            logger.info(f"Pose features: shape: {pose_features.shape}, dtype: {pose_features.dtype}")
            logger.info(f"  Mean: {pose_features.mean():.4f}, Std: {pose_features.std():.4f}")
            
            # === FUSIÓN ===
            fused_features = np.concatenate([visual_features, pose_features]).astype(np.float32)
            logger.info(f"\nFused features: shape: {fused_features.shape}, dtype: {fused_features.dtype}")
            logger.info(f"  Mean: {fused_features.mean():.4f}, Std: {fused_features.std():.4f}")
            
            all_features.append(fused_features)
            frame_count += 1
            
            # Solo procesar primeros 3 frames para debug
            if frame_count >= 3:
                break
        
        cap.release()
        
        if len(all_features) == 0:
            return None
        
        features_array = np.array(all_features)
        
        logger.info("\n" + "="*60)
        logger.info("RESUMEN DE FEATURES EXTRAÍDAS")
        logger.info("="*60)
        logger.info(f"Total frames procesados: {frame_count}")
        logger.info(f"Features shape: {features_array.shape}")
        logger.info(f"Features dtype: {features_array.dtype}")
        logger.info(f"Mean: {features_array.mean():.4f}")
        logger.info(f"Std: {features_array.std():.4f}")
        
        return features_array
    
    def compare_features(self, training_features: np.ndarray, inference_features: np.ndarray):
        """Compara features de entrenamiento vs inferencia"""
        logger.info("\n" + "="*60)
        logger.info("COMPARACIÓN DE FEATURES")
        logger.info("="*60)
        
        # Tomar solo los primeros N frames para comparar
        n_frames = min(len(training_features), len(inference_features))
        train_subset = training_features[:n_frames]
        infer_subset = inference_features[:n_frames]
        
        logger.info(f"\nComparando primeros {n_frames} frames:")
        logger.info(f"Training shape: {train_subset.shape}")
        logger.info(f"Inference shape: {infer_subset.shape}")
        
        # Estadísticas globales
        logger.info(f"\nTRAINING features:")
        logger.info(f"  Mean: {train_subset.mean():.4f}")
        logger.info(f"  Std: {train_subset.std():.4f}")
        logger.info(f"  Min: {train_subset.min():.4f}")
        logger.info(f"  Max: {train_subset.max():.4f}")
        
        logger.info(f"\nINFERENCE features:")
        logger.info(f"  Mean: {infer_subset.mean():.4f}")
        logger.info(f"  Std: {infer_subset.std():.4f}")
        logger.info(f"  Min: {infer_subset.min():.4f}")
        logger.info(f"  Max: {infer_subset.max():.4f}")
        
        # Diferencia absoluta promedio
        abs_diff = np.abs(train_subset - infer_subset).mean()
        logger.info(f"\nDiferencia absoluta promedio: {abs_diff:.6f}")
        
        # Correlación
        correlation = np.corrcoef(train_subset.flatten(), infer_subset.flatten())[0, 1]
        logger.info(f"Correlación: {correlation:.6f}")
        
        # Comparar por componente
        train_visual = train_subset[:, :1024]
        train_pose = train_subset[:, 1024:]
        infer_visual = infer_subset[:, :1024]
        infer_pose = infer_subset[:, 1024:]
        
        logger.info(f"\nVisual features:")
        logger.info(f"  Training mean: {train_visual.mean():.4f}, std: {train_visual.std():.4f}")
        logger.info(f"  Inference mean: {infer_visual.mean():.4f}, std: {infer_visual.std():.4f}")
        logger.info(f"  Diff: {np.abs(train_visual - infer_visual).mean():.6f}")
        
        logger.info(f"\nPose features:")
        logger.info(f"  Training mean: {train_pose.mean():.4f}, std: {train_pose.std():.4f}")
        logger.info(f"  Inference mean: {infer_pose.mean():.4f}, std: {infer_pose.std():.4f}")
        logger.info(f"  Diff: {np.abs(train_pose - infer_pose).mean():.6f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validar preprocesamiento")
    parser.add_argument("--training_features", type=Path, required=True,
                       help="Archivo .npy con features de entrenamiento")
    parser.add_argument("--video", type=Path, required=True,
                       help="Video original correspondiente")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    validator = PreprocessingValidator(device=args.device)
    
    # Validar features de entrenamiento
    training_features = validator.validate_training_features(args.training_features)
    
    # Extraer features usando pipeline de inferencia
    inference_features = validator.extract_features_from_video(args.video)
    
    if inference_features is not None:
        # Comparar
        validator.compare_features(training_features, inference_features)
    else:
        logger.error("No se pudieron extraer features del video")


if __name__ == "__main__":
    main()
