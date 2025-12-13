"""
Script de inferencia para videos pregrabados
Útil para pruebas sin necesidad de cámara en tiempo real
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
import json
import logging
from typing import Optional, Tuple, List
import argparse
from tqdm import tqdm

from pipelines.models_temporal import TemporalLSTMClassifier
from torchvision import models, transforms

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResNet101FeatureExtractor(nn.Module):
    """Extractor de features visuales usando ResNet101 frozen"""
    
    def __init__(self, output_dim=1024):
        super().__init__()
        resnet = models.resnet101(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(2048, output_dim)
        
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    @torch.no_grad()
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.squeeze(-1).squeeze(-1)
        features = self.projection(features)
        return features


class PoseFeatureExtractor(nn.Module):
    """Extractor de features de pose con MLP frozen"""
    
    def __init__(self, input_dim=300, hidden_dim=256, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    @torch.no_grad()
    def forward(self, x):
        return self.mlp(x)


class ASLVideoInference:
    """Sistema de inferencia para videos pregrabados"""
    
    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Cargar metadata
        self.class_to_gloss = self._load_class_mapping(metadata_path)
        self.num_classes = len(self.class_to_gloss)
        
        logger.info(f"Cargadas {self.num_classes} clases")
        
        # Inicializar extractores
        logger.info("Inicializando extractores de features...")
        self.visual_extractor = ResNet101FeatureExtractor(output_dim=1024).to(self.device)
        self.pose_extractor = PoseFeatureExtractor().to(self.device)
        
        # Cargar modelo
        logger.info(f"Cargando modelo desde: {model_path}")
        self.model = TemporalLSTMClassifier(
            input_dim=1152,
            hidden_dim=512,
            num_layers=2,
            num_classes=self.num_classes,
            dropout=0.3,
            bidirectional=True,
            use_attention=True
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # MediaPipe
        logger.info("Inicializando MediaPipe...")
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info("✓ Sistema inicializado")
    
    def _load_class_mapping(self, metadata_path: Path) -> dict:
        """Carga mapeo de class_id a gloss"""
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        class_to_gloss = {}
        
        if isinstance(metadata, dict):
            if 'videos' in metadata:
                entries = metadata['videos']
            else:
                entries = []
                for video_file, info in metadata.items():
                    if isinstance(info, dict):
                        entry = {'video_file': video_file, **info}
                        entries.append(entry)
        elif isinstance(metadata, list):
            entries = metadata
        else:
            raise ValueError("Formato de metadata no reconocido")
        
        for entry in entries:
            if isinstance(entry, dict):
                class_id = entry.get('class_id')
                gloss = entry.get('gloss', 'UNKNOWN')
                if class_id is not None:
                    class_to_gloss[int(class_id)] = gloss
        
        return class_to_gloss
    
    def extract_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extrae keypoints usando MediaPipe"""
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.holistic.process(image_rgb)
        
        pose_kp = np.zeros((33, 4))
        left_hand_kp = np.zeros((21, 4))
        right_hand_kp = np.zeros((21, 4))
        
        if results.pose_landmarks:
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                pose_kp[i] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
        
        if results.left_hand_landmarks:
            for i, landmark in enumerate(results.left_hand_landmarks.landmark):
                left_hand_kp[i] = [landmark.x, landmark.y, landmark.z, 1.0]
        
        if results.right_hand_landmarks:
            for i, landmark in enumerate(results.right_hand_landmarks.landmark):
                right_hand_kp[i] = [landmark.x, landmark.y, landmark.z, 1.0]
        
        keypoints = np.concatenate([
            pose_kp.flatten(),
            left_hand_kp.flatten(),
            right_hand_kp.flatten()
        ])
        
        return keypoints
    
    @torch.no_grad()
    def predict_video(
        self, 
        video_path: Path,
        max_frames: int = None,
        stride: int = 1
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predice la seña de un video completo
        
        Args:
            video_path: Ruta al video
            max_frames: Máximo de frames a procesar (None = todos)
            stride: Tomar 1 de cada N frames (para acelerar)
        
        Returns:
            predicted_gloss, confidence, top5_results
        """
        logger.info(f"Procesando video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"No se pudo abrir el video: {video_path}")
            return None, 0.0, []
        
        frames = []
        keypoints = []
        frame_count = 0
        
        with tqdm(desc="Extrayendo features", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % stride != 0:
                    frame_count += 1
                    continue
                
                if max_frames and len(frames) >= max_frames:
                    break
                
                # Resize
                frame_resized = cv2.resize(frame, (224, 224))
                
                # Extraer keypoints
                kp = self.extract_keypoints(frame)
                
                if kp is not None:
                    frames.append(frame_resized)
                    keypoints.append(kp)
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        
        if len(frames) == 0:
            logger.error("No se pudieron extraer frames del video")
            return None, 0.0, []
        
        logger.info(f"Frames procesados: {len(frames)}")
        
        # Convertir a tensores
        frames_batch = torch.stack([self.transform(f) for f in frames]).to(self.device)
        keypoints_batch = torch.from_numpy(np.stack(keypoints)).float().to(self.device)
        
        # Extraer features
        logger.info("Extrayendo features visuales...")
        visual_features = self.visual_extractor(frames_batch)
        
        logger.info("Extrayendo features de pose...")
        pose_features = self.pose_extractor(keypoints_batch)
        
        # Fusionar
        fused_features = torch.cat([visual_features, pose_features], dim=1)
        fused_features = fused_features.unsqueeze(0)  # (1, T, 1152)
        
        # Predecir
        logger.info("Realizando predicción...")
        logits = self.model(fused_features)
        probs = torch.softmax(logits, dim=1)[0]
        
        # Top-5
        top5_probs, top5_indices = torch.topk(probs, k=min(5, self.num_classes))
        
        top5_results = []
        for prob, idx in zip(top5_probs.cpu().numpy(), top5_indices.cpu().numpy()):
            gloss = self.class_to_gloss.get(int(idx), f"CLASS_{idx}")
            top5_results.append((gloss, float(prob)))
        
        predicted_class = top5_indices[0].item()
        predicted_gloss = self.class_to_gloss.get(predicted_class, f"CLASS_{predicted_class}")
        confidence = top5_probs[0].item()
        
        return predicted_gloss, confidence, top5_results
    
    def process_video_with_output(
        self,
        video_path: Path,
        output_path: Path,
        max_frames: int = None,
        stride: int = 1
    ):
        """
        Procesa un video y guarda el resultado con anotaciones
        
        Args:
            video_path: Video de entrada
            output_path: Video de salida
            max_frames: Máximo de frames
            stride: Stride para frames
        """
        predicted_gloss, confidence, top5 = self.predict_video(
            video_path, max_frames, stride
        )
        
        if predicted_gloss is None:
            logger.error("No se pudo procesar el video")
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PREDICCIÓN: {predicted_gloss}")
        logger.info(f"CONFIANZA: {confidence:.2%}")
        logger.info(f"\nTop-5 predicciones:")
        for i, (gloss, prob) in enumerate(top5, 1):
            logger.info(f"  {i}. {gloss}: {prob:.2%}")
        logger.info(f"{'='*60}\n")
        
        # Crear video con anotaciones
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height + 150))
        
        logger.info(f"Generando video anotado: {output_path}")
        
        with tqdm(desc="Escribiendo video", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Panel de información
                panel = np.zeros((150, width, 3), dtype=np.uint8)
                panel[:] = (30, 30, 30)
                
                # Predicción
                color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
                cv2.putText(panel, f"Seña: {predicted_gloss}", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)
                cv2.putText(panel, f"Confianza: {confidence:.1%}", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                
                # Top-3
                y = 110
                for i, (gloss, prob) in enumerate(top5[:3], 1):
                    text = f"{i}. {gloss}: {prob:.1%}"
                    cv2.putText(panel, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                               0.5, (150, 150, 150), 1, cv2.LINE_AA)
                    y += 20
                
                # Combinar
                result = np.vstack([frame, panel])
                out.write(result)
                pbar.update(1)
        
        cap.release()
        out.release()
        logger.info(f"✓ Video guardado: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Inferencia para videos pregrabados")
    
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Ruta al modelo entrenado"
    )
    parser.add_argument(
        "--metadata_path",
        type=Path,
        required=True,
        help="Ruta a dataset_metadata.json"
    )
    parser.add_argument(
        "--video_path",
        type=Path,
        required=True,
        help="Ruta al video de entrada"
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Ruta al video de salida (opcional)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"]
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Máximo de frames a procesar"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Tomar 1 de cada N frames"
    )
    
    args = parser.parse_args()
    
    # Validar
    if not args.model_path.exists():
        logger.error(f"Modelo no encontrado: {args.model_path}")
        return
    
    if not args.metadata_path.exists():
        logger.error(f"Metadata no encontrado: {args.metadata_path}")
        return
    
    if not args.video_path.exists():
        logger.error(f"Video no encontrado: {args.video_path}")
        return
    
    # Crear sistema
    system = ASLVideoInference(
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        device=args.device
    )
    
    # Procesar
    if args.output_path:
        system.process_video_with_output(
            video_path=args.video_path,
            output_path=args.output_path,
            max_frames=args.max_frames,
            stride=args.stride
        )
    else:
        predicted_gloss, confidence, top5 = system.predict_video(
            video_path=args.video_path,
            max_frames=args.max_frames,
            stride=args.stride
        )
        
        if predicted_gloss:
            logger.info(f"\n{'='*60}")
            logger.info(f"PREDICCIÓN: {predicted_gloss}")
            logger.info(f"CONFIANZA: {confidence:.2%}")
            logger.info(f"\nTop-5:")
            for i, (gloss, prob) in enumerate(top5, 1):
                logger.info(f"  {i}. {gloss}: {prob:.2%}")
            logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()