"""
Script de diagnóstico para debugging de inferencia
Compara features de entrenamiento vs inferencia
"""

import torch
import numpy as np
from pathlib import Path
import logging
import json
import sys
import cv2

sys.path.append(str(Path(__file__).parent.parent))

from pipelines.save_extractors import ResNet101FeatureExtractor, PoseFeatureExtractor
from pipelines.models_temporal import get_temporal_model
from config import config
import mediapipe as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceDebugger:
    """Herramienta de diagnóstico para problemas de inferencia"""
    
    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        training_features_dir: Path,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        logger.info("Cargando metadata...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        class_names = {}
        
        # Si el metadata tiene la clave "videos", usar esa lista
        if isinstance(metadata, dict) and "videos" in metadata:
            video_list = metadata["videos"]
        elif isinstance(metadata, list):
            video_list = metadata
        else:
            raise ValueError(f"Formato de metadata no reconocido: {type(metadata)}")
        
        logger.info(f"Videos encontrados en metadata: {len(video_list)}")
        
        # Extraer class_names únicos
        for entry in video_list:
            class_id = entry.get("class_id")
            class_name = entry.get("class_name")
            if class_id is not None and class_name:
                class_names[class_id] = class_name
        
        self.num_classes = len(class_names)
        self.class_names = class_names
        
        logger.info(f"Clases únicas detectadas: {self.num_classes}")
        if self.num_classes == 0:
            raise ValueError("No se encontraron clases en el metadata")
        
        logger.info("Cargando extractores...")
        self.visual_extractor = torch.load(
            Path("models/extractors/visual_extractor_full.pt"),
            map_location=self.device,
            weights_only=False
        )
        self.visual_extractor.eval()
        
        self.pose_extractor = torch.load(
            Path("models/extractors/pose_extractor_full.pt"),
            map_location=self.device,
            weights_only=False
        )
        self.pose_extractor.eval()
        
        logger.info("Inicializando modelo...")
        self.model = get_temporal_model(
            model_type=config.training.model_type,
            num_classes=self.num_classes,  # Ahora tiene el valor correcto (2286)
            hidden_dim=config.training.model_hidden_dim,
            num_layers=config.training.model_num_layers,
            dropout=0.0,
            bidirectional=config.training.model_bidirectional,
            use_attention=config.training.use_attention
        ).to(self.device)
        
        logger.info("Cargando checkpoint...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Checkpoint cargado desde 'model_state_dict'")
        else:
            self.model.load_state_dict(checkpoint)
            logger.info("Checkpoint cargado directamente")
        
        self.model.eval()
        logger.info("Modelo listo")
        
        self.training_features_dir = training_features_dir
        
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
    
    def compare_features(self, video_path: Path, sample_name: str = None):
        """Compara features entre entrenamiento e inferencia"""
        
        logger.info("="*80)
        logger.info("DIAGNÓSTICO DE FEATURES")
        logger.info("="*80)
        
        # 1. Cargar features de entrenamiento
        if sample_name:
            training_feature_path = self.training_features_dir / f"{sample_name}_fused.npy"
        else:
            training_files = list(self.training_features_dir.glob("*_fused.npy"))
            if not training_files:
                logger.error("No se encontraron features de entrenamiento")
                return
            training_feature_path = training_files[0]
        
        if not training_feature_path.exists():
            logger.error(f"No se encontró: {training_feature_path}")
            return
        
        training_features = np.load(training_feature_path).astype(np.float32)
        
        logger.info(f"\n1. Features de ENTRENAMIENTO ({training_feature_path.name}):")
        logger.info(f"   Shape: {training_features.shape}")
        logger.info(f"   Dtype: {training_features.dtype}")
        logger.info(f"   Mean: {training_features.mean():.6f}")
        logger.info(f"   Std: {training_features.std():.6f}")
        logger.info(f"   Min: {training_features.min():.6f}")
        logger.info(f"   Max: {training_features.max():.6f}")
        
        # 2. Extraer features de inferencia del video
        cap = cv2.VideoCapture(str(video_path))
        
        inference_features = []
        frame_count = 0
        
        logger.info(f"\n2. Extrayendo features de INFERENCIA ({video_path.name})...")
        
        while frame_count < min(len(training_features), 30):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extraer keypoints
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            pose_results = self.pose.process(frame_rgb)
            if not pose_results.pose_landmarks:
                continue
            
            pose_kpts = np.array([
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in pose_results.pose_landmarks.landmark
            ])
            
            # Normalizar
            left_hip = pose_kpts[23][:3]
            right_hip = pose_kpts[24][:3]
            center = (left_hip + right_hip) / 2
            shoulder_dist = np.linalg.norm(pose_kpts[11][:3] - pose_kpts[12][:3])
            scale = max(shoulder_dist, 0.1)
            pose_kpts[:, :3] = (pose_kpts[:, :3] - center) / scale
            pose_kpts[:, :3] = np.clip(pose_kpts[:, :3], -1, 1)
            
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
            
            # Extraer features
            # Visual
            frame_resized = cv2.resize(frame, (224, 224))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_normalized = (frame_rgb.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            
            with torch.no_grad():
                visual_features = self.visual_extractor(frame_tensor).cpu().numpy().squeeze()
            
            # Pose
            keypoints_flat = combined_kpts.flatten()[:300]
            keypoints_tensor = torch.from_numpy(keypoints_flat).unsqueeze(0).float().to(self.device)
            
            with torch.no_grad():
                pose_features = self.pose_extractor(keypoints_tensor).cpu().numpy().squeeze()
            
            # Fusionar
            fused = np.concatenate([visual_features, pose_features]).astype(np.float32)
            inference_features.append(fused)
            
            frame_count += 1
        
        cap.release()
        
        if not inference_features:
            logger.error("No se pudieron extraer features de inferencia")
            return
        
        inference_features = np.array(inference_features)
        
        logger.info(f"\n   Features de INFERENCIA:")
        logger.info(f"   Shape: {inference_features.shape}")
        logger.info(f"   Dtype: {inference_features.dtype}")
        logger.info(f"   Mean: {inference_features.mean():.6f}")
        logger.info(f"   Std: {inference_features.std():.6f}")
        logger.info(f"   Min: {inference_features.min():.6f}")
        logger.info(f"   Max: {inference_features.max():.6f}")
        
        # 3. Comparar
        logger.info(f"\n3. COMPARACIÓN:")
        
        mean_diff = abs(training_features.mean() - inference_features.mean())
        std_diff = abs(training_features.std() - inference_features.std())
        
        logger.info(f"   Diferencia de Mean: {mean_diff:.6f}")
        logger.info(f"   Diferencia de Std: {std_diff:.6f}")
        
        if mean_diff > 0.1 or std_diff > 0.1:
            logger.warning("   ⚠ DIFERENCIAS SIGNIFICATIVAS DETECTADAS")
            logger.warning("   Posibles causas:")
            logger.warning("   - Normalización diferente")
            logger.warning("   - Extractores diferentes")
            logger.warning("   - Preprocessing diferente")
        else:
            logger.info("   ✓ Features similares")
        
        # 4. Test de predicción
        logger.info(f"\n4. TEST DE PREDICCIÓN:")
        
        # Predicción con features de entrenamiento
        with torch.no_grad():
            train_tensor = torch.from_numpy(training_features[:len(inference_features)]).unsqueeze(0).float().to(self.device)
            train_lengths = torch.tensor([len(train_tensor[0])], dtype=torch.long).to(self.device)
            train_logits = self.model(train_tensor, train_lengths)
            train_probs = torch.softmax(train_logits, dim=-1).cpu().numpy().squeeze()
            train_top_class = int(np.argmax(train_probs))
            train_confidence = float(train_probs[train_top_class])
        
        logger.info(f"\n   Predicción con features de ENTRENAMIENTO:")
        logger.info(f"   Clase: {train_top_class} ({self.class_names.get(train_top_class, 'Unknown')})")
        logger.info(f"   Confianza: {train_confidence:.4f}")
        
        # Predicción con features de inferencia
        with torch.no_grad():
            inf_tensor = torch.from_numpy(inference_features).unsqueeze(0).float().to(self.device)
            inf_lengths = torch.tensor([len(inference_features)], dtype=torch.long).to(self.device)
            inf_logits = self.model(inf_tensor, inf_lengths)
            inf_probs = torch.softmax(inf_logits, dim=-1).cpu().numpy().squeeze()
            inf_top_class = int(np.argmax(inf_probs))
            inf_confidence = float(inf_probs[inf_top_class])
        
        logger.info(f"\n   Predicción con features de INFERENCIA:")
        logger.info(f"   Clase: {inf_top_class} ({self.class_names.get(inf_top_class, 'Unknown')})")
        logger.info(f"   Confianza: {inf_confidence:.4f}")
        
        if train_top_class == inf_top_class:
            logger.info("\n   ✓ MISMA CLASE PREDICHA")
        else:
            logger.warning("\n   ⚠ CLASES DIFERENTES")
        
        confidence_diff = abs(train_confidence - inf_confidence)
        logger.info(f"   Diferencia de confianza: {confidence_diff:.4f}")
        
        logger.info("\n" + "="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnóstico de inferencia")
    parser.add_argument("--model", type=Path, required=True,
                       help="Ruta al modelo entrenado")
    parser.add_argument("--metadata", type=Path, default=Path("data/dataset_meta.json"),
                       help="Ruta al metadata")
    parser.add_argument("--training_features", type=Path, default=Path("data/features_fused"),
                       help="Directorio con features de entrenamiento")
    parser.add_argument("--video", type=Path, required=True,
                       help="Video para comparar")
    parser.add_argument("--sample_name", type=str, default=None,
                       help="Nombre del sample de entrenamiento para comparar")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device")
    
    args = parser.parse_args()
    
    debugger = InferenceDebugger(
        model_path=args.model,
        metadata_path=args.metadata,
        training_features_dir=args.training_features,
        device=args.device
    )
    
    debugger.compare_features(
        video_path=args.video,
        sample_name=args.sample_name
    )


if __name__ == "__main__":
    main()
