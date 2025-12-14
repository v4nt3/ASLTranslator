"""
Script de inferencia CORREGIDO que carga extractores guardados
Ahora funciona correctamente en tiempo real
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from collections import deque
import json
import logging
from typing import Optional, Tuple, List
import argparse
from torchvision import transforms

from pipelines.models_temporal import TemporalLSTMClassifier
from pipelines.save_extractors import ResNet101FeatureExtractor
from pipelines.save_extractors import PoseFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ASLRealtimeInference:
    """Sistema de inferencia en tiempo real CORREGIDO"""
    
    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        visual_extractor_path: Path,
        pose_extractor_path: Path,
        device: str = "cuda",
        buffer_size: int = 24,
        confidence_threshold: float = 0.3,
        smoothing_window: int = 3
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        
        # Cargar metadata
        self.class_to_gloss = self._load_class_mapping(metadata_path)
        self.num_classes = len(self.class_to_gloss)
        logger.info(f"Cargadas {self.num_classes} clases")
        
        # CARGAR EXTRACTORES GUARDADOS
        logger.info("Cargando extractores de features...")
        
        try:
            # Intentar cargar modelos completos primero
            self.visual_extractor = torch.load(visual_extractor_path, map_location=self.device, weights_only=False)
            logger.info(f"✓ Visual extractor cargado: {visual_extractor_path}")
        except:
            # Si falla, cargar solo state_dict
            
            self.visual_extractor = ResNet101FeatureExtractor(output_dim=1024).to(self.device)
            self.visual_extractor.load_state_dict(torch.load(visual_extractor_path, map_location=self.device, weights_only=False))
            logger.info(f"✓ Visual extractor (state_dict) cargado: {visual_extractor_path}")
        
        try:
            self.pose_extractor = torch.load(pose_extractor_path, map_location=self.device, weights_only=False)
            logger.info(f"✓ Pose extractor cargado: {pose_extractor_path}")
        except:
            
            self.pose_extractor = PoseFeatureExtractor().to(self.device)
            self.pose_extractor.load_state_dict(torch.load(pose_extractor_path, map_location=self.device, weights_only=False))
            logger.info(f"✓ Pose extractor (state_dict) cargado: {pose_extractor_path}")
        
        self.visual_extractor.eval()
        self.pose_extractor.eval()
        
        # Transform para frames (IGUAL que en precompute_visual_features.py)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Cargar modelo temporal
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
        
        # MediaPipe
        logger.info("Inicializando MediaPipe...")
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Buffers
        self.frame_buffer = deque(maxlen=buffer_size)
        self.keypoints_buffer = deque(maxlen=buffer_size)
        self.prediction_buffer = deque(maxlen=smoothing_window)
        
        logger.info("✓ Sistema de inferencia inicializado correctamente")
    
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
    
    def extract_keypoints(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], any]:
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
        
        return keypoints, results
    
    def add_frame(self, frame: np.ndarray) -> any:
        """Agrega frame y extrae keypoints"""
        frame_resized = cv2.resize(frame, (224, 224))
        keypoints, results = self.extract_keypoints(frame)
        
        if keypoints is not None:
            self.frame_buffer.append(frame_resized)
            self.keypoints_buffer.append(keypoints)
        
        return results
    
    def is_ready(self) -> bool:
        """Verifica si hay suficientes frames"""
        return len(self.frame_buffer) >= self.buffer_size
    
    @torch.no_grad()
    def predict(self) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Realiza predicción sobre el buffer actual"""
        if not self.is_ready():
            return None, 0.0, []
        
        # Convertir frames a features visuales
        frames_batch = torch.stack([
            self.transform(frame) for frame in self.frame_buffer
        ]).to(self.device)
        
        visual_features = self.visual_extractor(frames_batch)  # (T, 1024)
        
        # Convertir keypoints a features de pose
        keypoints_batch = torch.from_numpy(
            np.stack(list(self.keypoints_buffer))
        ).float().to(self.device)
        
        pose_features = self.pose_extractor(keypoints_batch)  # (T, 128)
        
        # Fusionar features
        fused_features = torch.cat([visual_features, pose_features], dim=1)  # (T, 1152)
        fused_features = fused_features.unsqueeze(0)  # (1, T, 1152)
        
        # Predecir
        logits = self.model(fused_features)
        probs = torch.softmax(logits, dim=1)[0]
        
        # Top-5
        top5_probs, top5_indices = torch.topk(probs, k=min(5, self.num_classes))
        
        top5_results = []
        for prob, idx in zip(top5_probs.cpu().numpy(), top5_indices.cpu().numpy()):
            gloss = self.class_to_gloss.get(int(idx), f"CLASS_{idx}")
            top5_results.append((gloss, float(prob)))
        
        # Mejor predicción
        predicted_class = top5_indices[0].item()
        predicted_gloss = self.class_to_gloss.get(predicted_class, f"CLASS_{predicted_class}")
        confidence = top5_probs[0].item()
        
        # Suavizar
        self.prediction_buffer.append((predicted_gloss, confidence))
        
        if len(self.prediction_buffer) >= self.smoothing_window:
            glosses = [p[0] for p in self.prediction_buffer]
            most_common = max(set(glosses), key=glosses.count)
            avg_confidence = np.mean([p[1] for p in self.prediction_buffer if p[0] == most_common])
            return most_common, avg_confidence, top5_results
        
        return predicted_gloss, confidence, top5_results
    
    def draw_results(
        self,
        frame: np.ndarray,
        predicted_gloss: str,
        confidence: float,
        top5: List[Tuple[str, float]],
        results
    ) -> np.ndarray:
        """Dibuja resultados"""
        frame_annotated = frame.copy()
        h, w = frame.shape[:2]
        
        # Landmarks
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame_annotated, results.pose_landmarks,
                self.mp_holistic.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2)
            )
        
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame_annotated, results.left_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2)
            )
        
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                frame_annotated, results.right_hand_landmarks,
                self.mp_holistic.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2)
            )
        
        # Panel
        panel_height = 200
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (30, 30, 30)
        
        # Predicción
        if confidence >= self.confidence_threshold:
            color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
            text = f"Seña: {predicted_gloss}"
            conf_text = f"Confianza: {confidence:.2%}"
            
            cv2.putText(panel, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                       1.0, color, 2, cv2.LINE_AA)
            cv2.putText(panel, conf_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, color, 2, cv2.LINE_AA)
        else:
            cv2.putText(panel, "No hay detección confiable", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2, cv2.LINE_AA)
        
        # Top-3
        y_offset = 120
        cv2.putText(panel, "Alternativas:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        
        for i, (gloss, prob) in enumerate(top5[:3]):
            y_offset += 25
            text = f"{i+1}. {gloss}: {prob:.1%}"
            cv2.putText(panel, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Buffer status
        buffer_status = f"Buffer: {len(self.frame_buffer)}/{self.buffer_size}"
        cv2.putText(frame_annotated, buffer_status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        result = np.vstack([frame_annotated, panel])
        return result
    
    def run(self, camera_id: int = 0, show_fps: bool = True):
        """Ejecuta inferencia en tiempo real"""
        logger.info(f"Iniciando captura desde cámara {camera_id}...")
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            logger.error("No se pudo abrir la cámara")
            return
        
        logger.info("✓ Cámara inicializada")
        logger.info("\nControles:")
        logger.info("  ESPACIO - Limpiar buffer")
        logger.info("  Q - Salir\n")
        
        import time
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = self.add_frame(frame)
                
                predicted_gloss, confidence, top5 = None, 0.0, []
                if self.is_ready():
                    predicted_gloss, confidence, top5 = self.predict()
                
                frame_display = self.draw_results(
                    frame, predicted_gloss or "Acumulando...",
                    confidence, top5, results
                )
                
                if show_fps:
                    frame_count += 1
                    elapsed = time.time() - start_time
                    if elapsed > 1.0:
                        fps = frame_count / elapsed
                        frame_count = 0
                        start_time = time.time()
                    
                    cv2.putText(frame_display, f"FPS: {fps:.1f}",
                               (frame.shape[1] - 120, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow('ASL Real-time Detection', frame_display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    logger.info("Buffer limpiado")
                    self.frame_buffer.clear()
                    self.keypoints_buffer.clear()
                    self.prediction_buffer.clear()
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Inferencia finalizada")


def main():
    parser = argparse.ArgumentParser(description="Inferencia en tiempo real ASL (CORREGIDO)")
    
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--metadata_path", type=Path, required=True)
    parser.add_argument("--visual_extractor", type=Path, required=True,
                       help="Ruta a visual_extractor_full.pt")
    parser.add_argument("--pose_extractor", type=Path, required=True,
                       help="Ruta a pose_extractor_full.pt")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--buffer_size", type=int, default=24)
    parser.add_argument("--confidence_threshold", type=float, default=0.3)
    parser.add_argument("--smoothing_window", type=int, default=3)
    
    args = parser.parse_args()
    
    # Validar
    for path, name in [
        (args.model_path, "Modelo"),
        (args.metadata_path, "Metadata"),
        (args.visual_extractor, "Visual extractor"),
        (args.pose_extractor, "Pose extractor")
    ]:
        if not path.exists():
            logger.error(f"{name} no encontrado: {path}")
            return
    
    # Crear sistema
    system = ASLRealtimeInference(
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        visual_extractor_path=args.visual_extractor,
        pose_extractor_path=args.pose_extractor,
        device=args.device,
        buffer_size=args.buffer_size,
        confidence_threshold=args.confidence_threshold,
        smoothing_window=args.smoothing_window
    )
    
    # Ejecutar
    system.run(camera_id=args.camera_id)


if __name__ == "__main__":
    main()