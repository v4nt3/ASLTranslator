"""
Script de inferencia para procesar señas en tiempo real desde cámara
Detecta señas continuas y construye oraciones
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from collections import deque
from typing import List, Tuple, Optional, Dict
import logging
import json
import sys

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.save_extractors import ResNet101FeatureExtractor, PoseFeatureExtractor
from pipelines.models_temporal import get_temporal_model
from config import config
import mediapipe as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignDetector:
    """Detecta inicio y fin de señas usando varianza de movimiento"""
    
    def __init__(self, window_size: int = 10, threshold: float = 0.01):
        self.window_size = window_size
        self.threshold = threshold
        self.motion_history = deque(maxlen=window_size)
        self.is_signing = False
        self.sign_start_frame = 0
    
    def calculate_motion(self, keypoints_curr: np.ndarray, keypoints_prev: np.ndarray) -> float:
        """Calcula la magnitud de movimiento entre frames consecutivos"""
        if keypoints_curr is None or keypoints_prev is None:
            return 0.0
        
        # Calcular movimiento de manos principalmente
        hand_indices = list(range(33, 75))  # Índices de manos en keypoints
        motion = np.mean(np.linalg.norm(
            keypoints_curr[hand_indices, :3] - keypoints_prev[hand_indices, :3],
            axis=1
        ))
        return motion
    
    def update(self, motion: float) -> Tuple[bool, bool]:
        """
        Actualiza el detector con nuevo movimiento
        
        Returns:
            (sign_detected, sign_ended)
        """
        self.motion_history.append(motion)
        
        if len(self.motion_history) < self.window_size:
            return False, False
        
        avg_motion = np.mean(self.motion_history)
        sign_detected = False
        sign_ended = False
        
        # Detectar inicio de seña (movimiento aumenta)
        if not self.is_signing and avg_motion > self.threshold:
            self.is_signing = True
            self.sign_start_frame = 0
            sign_detected = True
            logger.info(f"[Detector] Seña iniciada (motion: {avg_motion:.4f})")
        
        # Detectar fin de seña (movimiento disminuye)
        elif self.is_signing and avg_motion < self.threshold * 0.5:
            self.is_signing = False
            sign_ended = True
            logger.info(f"[Detector] Seña finalizada (motion: {avg_motion:.4f})")
        
        return sign_detected, sign_ended


class CameraInference:
    """Sistema de inferencia en tiempo real desde cámara"""
    
    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        device: str = "cuda",
        buffer_size: int = 60,
        min_sign_length: int = 12,
        confidence_threshold: float = 0.3
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.buffer_size = buffer_size
        self.min_sign_length = min_sign_length
        self.confidence_threshold = confidence_threshold
        
        logger.info(f"Inicializando CameraInference en {self.device}")
        
        # Cargar mapeo de clases
        self.class_names = self._load_class_names(metadata_path)
        self.num_classes = len(self.class_names)
        logger.info(f"Clases cargadas: {self.num_classes}")
        
        # Cargar extractores
        self.visual_extractor = self._load_visual_extractor()
        self.pose_extractor = self._load_pose_extractor()
        
        # Cargar modelo temporal
        self.model = self._load_temporal_model(model_path)
        
        # MediaPipe para keypoints
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
        
        # Buffers para almacenar frames y features
        self.frame_buffer = deque(maxlen=buffer_size)
        self.keypoints_buffer = deque(maxlen=buffer_size)
        self.features_buffer = deque(maxlen=buffer_size)
        
        # Detector de señas
        self.sign_detector = SignDetector(window_size=10, threshold=0.015)
        
        # Historia de predicciones (para construir oraciones)
        self.sentence = []
        self.last_prediction = None
        self.prev_keypoints = None
        
        logger.info("Sistema de inferencia inicializado correctamente")
    
    def _load_class_names(self, metadata_path: Path) -> Dict[int, str]:
        """Carga el mapeo class_id -> class_name desde metadata"""
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        class_names = {}
        
        if isinstance(metadata, dict) and "videos" in metadata:
            for entry in metadata["videos"]:
                class_id = entry.get("class_id")
                class_name = entry.get("class_name")
                if class_id is not None and class_name:
                    class_names[class_id] = class_name
        elif isinstance(metadata, list):
            for entry in metadata:
                class_id = entry.get("class_id")
                class_name = entry.get("class_name")
                if class_id is not None and class_name:
                    class_names[class_id] = class_name
        
        return class_names
    
    def _load_visual_extractor(self) -> ResNet101FeatureExtractor:
        """Carga el extractor visual"""
        extractor_path = Path("models/extractors/visual_extractor_full.pt")
        logger.info(f"Cargando extractor visual desde {extractor_path}")
        
        extractor = torch.load(extractor_path, map_location=self.device, weights_only=False)
        extractor.eval()
        return extractor
    
    def _load_pose_extractor(self) -> PoseFeatureExtractor:
        """Carga el extractor de pose"""
        extractor_path = Path("models/extractors/pose_extractor_full.pt")
        logger.info(f"Cargando extractor de pose desde {extractor_path}")
        
        extractor = torch.load(extractor_path, map_location=self.device, weights_only=False)
        extractor.eval()
        return extractor
    
    def _load_temporal_model(self, model_path: Path):
        """Carga el modelo temporal"""
        logger.info(f"Cargando modelo temporal desde {model_path}")
        
        model = get_temporal_model(
            model_type=config.training.model_type,
            num_classes=self.num_classes,
            hidden_dim=config.training.model_hidden_dim,
            num_layers=config.training.model_num_layers,
            dropout=0.0,  # Sin dropout en inferencia
            bidirectional=config.training.model_bidirectional,
            use_attention=config.training.use_attention
        ).to(self.device)
        
        # Cargar pesos
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        logger.info("Modelo temporal cargado correctamente")
        return model
    
    def extract_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extrae keypoints de un frame (pose + hands)"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose keypoints
        pose_results = self.pose.process(frame_rgb)
        if not pose_results.pose_landmarks:
            return None
        
        pose_kpts = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in pose_results.pose_landmarks.landmark
        ])  # (33, 4)
        
        # Normalizar pose
        pose_kpts = self._normalize_keypoints(pose_kpts)
        
        # Hand keypoints
        hand_results = self.hands.process(frame_rgb)
        hand_kpts = np.zeros((42, 4))
        
        if hand_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks[:2]):
                start_idx = hand_idx * 21
                hand_kpts[start_idx:start_idx+21] = np.array([
                    [lm.x, lm.y, lm.z, 1.0]
                    for lm in hand_landmarks.landmark
                ])
        
        # Combinar: (33, 4) + (42, 4) = (75, 4)
        combined_kpts = np.concatenate([pose_kpts, hand_kpts], axis=0)
        return combined_kpts
    
    def _normalize_keypoints(self, pose_kpts: np.ndarray) -> np.ndarray:
        """Normaliza keypoints usando centro de cadera"""
        left_hip = pose_kpts[23][:3]
        right_hip = pose_kpts[24][:3]
        center = (left_hip + right_hip) / 2
        
        shoulder_dist = np.linalg.norm(pose_kpts[11][:3] - pose_kpts[12][:3])
        scale = max(shoulder_dist, 0.1)
        
        normalized = pose_kpts.copy()
        normalized[:, :3] = (pose_kpts[:, :3] - center) / scale
        normalized[:, :3] = np.clip(normalized[:, :3], -1, 1)
        
        return normalized
    
    def extract_features(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Extrae features fusionadas (visual + pose) de un frame"""
        # Visual features
        frame_resized = cv2.resize(frame, (224, 224))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = (frame_rgb.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            visual_features = self.visual_extractor(frame_tensor).cpu().numpy().squeeze()  # (1024,)
        
        # Pose features
        keypoints_flat = keypoints.flatten()[:300]  # (75, 4) -> (300,)
        keypoints_tensor = torch.from_numpy(keypoints_flat).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            pose_features = self.pose_extractor(keypoints_tensor).cpu().numpy().squeeze()  # (128,)
        
        # Fusionar: (1024,) + (128,) = (1152,) en float32
        fused_features = np.concatenate([visual_features, pose_features]).astype(np.float32)
        return fused_features
    
    @torch.no_grad()
    def predict_sign(self, features_sequence: np.ndarray) -> Tuple[int, float, List[Tuple[int, float]]]:
        """
        Predice la seña dado una secuencia de features
        
        Returns:
            (class_id, confidence, top5_predictions)
        """
        # Convertir a tensor float32
        features_tensor = torch.from_numpy(features_sequence).unsqueeze(0).float().to(self.device)  # (1, T, 1152)
        lengths = torch.tensor([len(features_sequence)], dtype=torch.long).to(self.device)
        
        # Predicción
        logits = self.model(features_tensor, lengths)  # (1, num_classes)
        
        # Aplicar softmax correctamente
        probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
        
        # Top-5 predicciones
        top5_indices = np.argsort(probs)[-5:][::-1]
        top5_predictions = [(int(idx), float(probs[idx])) for idx in top5_indices]
        
        predicted_class = int(top5_indices[0])
        confidence = float(probs[predicted_class])
        
        return predicted_class, confidence, top5_predictions
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
        """
        Procesa un frame de la cámara
        
        Returns:
            (frame_with_overlay, detected_sign_name)
        """
        detected_sign = None
        
        # Extraer keypoints
        keypoints = self.extract_keypoints(frame)
        
        if keypoints is None:
            cv2.putText(frame, "No se detecta persona", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame, None
        
        # Calcular movimiento
        motion = 0.0
        if self.prev_keypoints is not None:
            motion = self.sign_detector.calculate_motion(keypoints, self.prev_keypoints)
        
        self.prev_keypoints = keypoints.copy()
        
        # Actualizar detector
        sign_started, sign_ended = self.sign_detector.update(motion)
        
        # Agregar frame al buffer
        self.frame_buffer.append(frame.copy())
        self.keypoints_buffer.append(keypoints)
        
        # Extraer features y agregar al buffer
        if len(self.frame_buffer) > 0:
            features = self.extract_features(frame, keypoints)
            self.features_buffer.append(features)
        
        # Si la seña terminó, procesar el buffer
        if sign_ended and len(self.features_buffer) >= self.min_sign_length:
            logger.info(f"Procesando seña con {len(self.features_buffer)} frames")
            
            # Convertir buffer a array
            features_sequence = np.array(list(self.features_buffer))
            
            # Predecir
            class_id, confidence, top5 = self.predict_sign(features_sequence)
            
            if confidence >= self.confidence_threshold:
                class_name = self.class_names.get(class_id, f"Unknown_{class_id}")
                detected_sign = class_name
                
                # Evitar duplicados consecutivos
                if self.last_prediction != class_name:
                    self.sentence.append((class_name, confidence))
                    self.last_prediction = class_name
                    logger.info(f"Seña detectada: {class_name} (confianza: {confidence:.2%})")
                    logger.info(f"Top-5: {[(self.class_names.get(idx, idx), conf) for idx, conf in top5[:5]]}")
            
            # Limpiar buffers para próxima seña
            self.features_buffer.clear()
        
        # Overlay de información
        self._draw_overlay(frame, motion, detected_sign)
        
        return frame, detected_sign
    
    def _draw_overlay(self, frame: np.ndarray, motion: float, detected_sign: Optional[str]):
        """Dibuja información sobre el frame"""
        h, w = frame.shape[:2]
        
        # Estado del detector
        status_color = (0, 255, 0) if self.sign_detector.is_signing else (0, 165, 255)
        status_text = "DETECTANDO" if self.sign_detector.is_signing else "EN ESPERA"
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Movimiento
        cv2.putText(frame, f"Motion: {motion:.4f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Buffer size
        cv2.putText(frame, f"Buffer: {len(self.features_buffer)}/{self.buffer_size}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Última seña detectada
        if detected_sign:
            cv2.rectangle(frame, (10, h-100), (w-10, h-10), (0, 255, 0), -1)
            cv2.putText(frame, f"Detectada: {detected_sign}", (20, h-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Oración construida
        if self.sentence:
            sentence_text = " ".join([word for word, _ in self.sentence[-5:]])  # Últimas 5 palabras
            cv2.putText(frame, f"Oracion: {sentence_text}", (10, h-120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def run(self, camera_id: int = 0):
        """Ejecuta el loop principal de inferencia"""
        logger.info(f"Iniciando captura desde cámara {camera_id}")
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            logger.error("No se pudo abrir la cámara")
            return
        
        logger.info("Cámara inicializada. Presiona 'q' para salir, 'c' para limpiar oración")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("No se pudo leer frame")
                    break
                
                # Procesar frame
                frame_with_overlay, detected_sign = self.process_frame(frame)
                
                # Mostrar
                cv2.imshow("ASL Sign Detection", frame_with_overlay)
                
                # Control
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.sentence.clear()
                    self.last_prediction = None
                    logger.info("Oración limpiada")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Mostrar oración final
            if self.sentence:
                logger.info("\n" + "="*60)
                logger.info("ORACIÓN FINAL:")
                logger.info(" ".join([word for word, _ in self.sentence]))
                logger.info("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Inferencia en tiempo real desde cámara")
    parser.add_argument("--model", type=Path, required=True,
                       help="Ruta al modelo entrenado (.pt)")
    parser.add_argument("--metadata", type=Path, default=Path("data/dataset_meta.json"),
                       help="Ruta a dataset_meta.json con mapeo de clases")
    parser.add_argument("--camera", type=int, default=0,
                       help="ID de la cámara (default: 0)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device: cuda o cpu (default: cuda)")
    parser.add_argument("--confidence", type=float, default=0.3,
                       help="Umbral mínimo de confianza (default: 0.3)")
    parser.add_argument("--buffer_size", type=int, default=60,
                       help="Tamaño del buffer de frames (default: 60)")
    parser.add_argument("--min_length", type=int, default=12,
                       help="Mínimo de frames para considerar una seña (default: 12)")
    
    args = parser.parse_args()
    
    if not args.model.exists():
        logger.error(f"Modelo no encontrado: {args.model}")
        return
    
    if not args.metadata.exists():
        logger.error(f"Metadata no encontrado: {args.metadata}")
        return
    
    # Crear sistema de inferencia
    inference = CameraInference(
        model_path=args.model,
        metadata_path=args.metadata,
        device=args.device,
        buffer_size=args.buffer_size,
        min_sign_length=args.min_length,
        confidence_threshold=args.confidence
    )
    
    # Ejecutar
    inference.run(camera_id=args.camera)


if __name__ == "__main__":
    main()
