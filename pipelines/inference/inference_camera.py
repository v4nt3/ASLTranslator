"""
Sistema de inferencia en tiempo real para lenguaje de señas
Captura desde webcam y traduce señas en vivo
"""

import torch
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from collections import deque
from typing import List, Tuple, Optional
import logging
import json
import time
from dataclasses import dataclass

from config import config
from pipelines.models_temporal import TemporalLSTMClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RealtimeConfig:
    """Configuración para inferencia en tiempo real"""
    # Buffer de frames
    buffer_size: int = 48  # Tamaño del buffer circular
    min_frames_to_predict: int = 24  # Mínimo de frames para hacer predicción
    
    # Detección de seña
    activity_threshold: float = 0.35  # Umbral para detectar seña activa
    min_active_frames: int = 12  # Mínimo de frames activos consecutivos
    cooldown_frames: int = 15  # Frames de espera entre predicciones
    
    # Display
    show_keypoints: bool = True
    show_activity_meter: bool = True
    font_scale: float = 0.7
    font_thickness: int = 2


class RealtimeSignDetector:
    """
    Detector de señas en tiempo real con buffer circular
    """
    
    def __init__(self, config: RealtimeConfig):
        self.config = config
        
        # Buffer circular para frames y keypoints
        self.frame_buffer = deque(maxlen=config.buffer_size)
        self.keypoint_buffer = deque(maxlen=config.buffer_size)
        self.activity_buffer = deque(maxlen=config.buffer_size)
        
        # Estado
        self.is_signing = False
        self.frames_since_prediction = 0
        self.active_frames_count = 0
        
        # MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extrae keypoints de un frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose
        pose_results = self.pose.process(frame_rgb)
        if not pose_results.pose_landmarks:
            return None
        
        pose_kpts = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in pose_results.pose_landmarks.landmark
        ])
        
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
        
        # Combinar
        combined = np.concatenate([pose_kpts, hand_kpts], axis=0)
        
        # Retornar también los resultados para visualización
        return combined, pose_results, hand_results
    
    def calculate_activity_score(self, keypoints: np.ndarray) -> float:
        """Calcula score de actividad para un frame"""
        pose_kpts = keypoints[:33]
        left_hand = keypoints[33:54]
        right_hand = keypoints[54:75]
        
        score = 0.0
        
        # Visibilidad de manos
        left_vis = np.mean(left_hand[:, 3])
        right_vis = np.mean(right_hand[:, 3])
        hand_visibility = (left_vis + right_vis) / 2.0
        score += 0.5 * hand_visibility
        
        # Manos levantadas
        hip_y = np.mean([pose_kpts[23, 1], pose_kpts[24, 1]])
        left_hand_y = np.mean(left_hand[:, 1])
        right_hand_y = np.mean(right_hand[:, 1])
        
        hands_raised = 0.0
        if left_vis > 0.3 and left_hand_y < hip_y:
            hands_raised += 1.0
        if right_vis > 0.3 and right_hand_y < hip_y:
            hands_raised += 1.0
        hands_raised /= 2.0
        score += 0.5 * hands_raised
        
        return np.clip(score, 0, 1)
    
    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Actualiza el buffer con un nuevo frame
        
        Returns:
            ready_to_predict: bool - Si hay suficientes frames para predecir
            frames: np.ndarray o None - Frames para predicción
            keypoints: np.ndarray o None - Keypoints para predicción
        """
        # Extraer keypoints
        result = self.extract_keypoints(frame)
        if result is None:
            return False, None, None
        
        keypoints, pose_results, hand_results = result
        
        # Calcular activity score
        activity_score = self.calculate_activity_score(keypoints)
        
        # Agregar al buffer
        self.frame_buffer.append(frame.copy())
        self.keypoint_buffer.append(keypoints)
        self.activity_buffer.append(activity_score)
        
        # Incrementar cooldown
        self.frames_since_prediction += 1
        
        # Detectar si está haciendo una seña
        is_active = activity_score > self.config.activity_threshold
        
        if is_active:
            self.active_frames_count += 1
        else:
            self.active_frames_count = 0
        
        # Verificar si debe predecir
        should_predict = (
            len(self.frame_buffer) >= self.config.min_frames_to_predict and
            self.active_frames_count >= self.config.min_active_frames and
            self.frames_since_prediction >= self.config.cooldown_frames
        )
        
        if should_predict:
            # Tomar últimos N frames del buffer
            n = self.config.min_frames_to_predict
            frames = np.array(list(self.frame_buffer)[-n:])
            keypoints = np.array(list(self.keypoint_buffer)[-n:])
            
            # Reset cooldown
            self.frames_since_prediction = 0
            self.active_frames_count = 0
            
            return True, frames, keypoints
        
        return False, None, None
    
    def get_current_activity(self) -> float:
        """Retorna el activity score promedio reciente"""
        if len(self.activity_buffer) == 0:
            return 0.0
        
        # Promedio de los últimos 10 frames
        recent = list(self.activity_buffer)[-10:]
        return np.mean(recent)
    
    def reset(self):
        """Resetea el estado del detector"""
        self.frame_buffer.clear()
        self.keypoint_buffer.clear()
        self.activity_buffer.clear()
        self.frames_since_prediction = 0
        self.active_frames_count = 0


class RealtimeSignTranslator:
    """
    Sistema completo de traducción en tiempo real
    """
    
    def __init__(
        self,
        model_path: Path,
        visual_extractor_path: Path,
        pose_extractor_path: Path,
        class_mapping_path: Path,
        device: str = "cuda",
        realtime_config: Optional[RealtimeConfig] = None
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando device: {self.device}")
        
        if realtime_config is None:
            realtime_config = RealtimeConfig()
        
        self.rt_config = realtime_config
        
        # Cargar mapping de clases
        with open(class_mapping_path, 'r') as f:
            metadata = json.load(f)
        
        self.id_to_class = {}
        if isinstance(metadata, dict) and 'videos' in metadata:
            for entry in metadata['videos']:
                class_id = entry.get('class_id')
                class_name = entry.get('class_name')
                if class_id is not None and class_name:
                    self.id_to_class[class_id] = class_name
        
        logger.info(f"Clases cargadas: {len(self.id_to_class)}")
        
        # Cargar extractores
        logger.info("Cargando extractores...")
        self.visual_extractor = torch.load(visual_extractor_path, map_location=self.device)
        self.visual_extractor.eval()
        
        self.pose_extractor = torch.load(pose_extractor_path, map_location=self.device)
        self.pose_extractor.eval()
        
        # Cargar modelo
        logger.info("Cargando modelo temporal...")
        self.model = TemporalLSTMClassifier(
            input_dim=config.data.fused_feature_dim,
            hidden_dim=config.training.model_hidden_dim,
            num_layers=config.training.model_num_layers,
            num_classes=len(self.id_to_class),
            dropout=config.training.model_dropout,
            bidirectional=config.training.model_bidirectional,
            use_attention=config.training.use_attention
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        logger.info("✓ Modelo cargado")
        
        # Detector
        self.detector = RealtimeSignDetector(realtime_config)
        
        # Historial de predicciones
        self.translation_history = []
        self.max_history = 10
    
    def extract_features(self, frames: np.ndarray, keypoints: np.ndarray) -> torch.Tensor:
        """Extrae features fusionadas"""
        T = len(frames)
        
        # Visual features
        batch_tensors = []
        for frame in frames:
            # Resize si es necesario
            if frame.shape[:2] != (config.data.frame_height, config.data.frame_width):
                frame = cv2.resize(frame, (config.data.frame_width, config.data.frame_height))
            
            frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            frame_tensor = (frame_tensor - mean) / std
            batch_tensors.append(frame_tensor)
        
        batch = torch.stack(batch_tensors).to(self.device)
        
        with torch.no_grad():
            visual_features = self.visual_extractor(batch).cpu().numpy()
        
        # Pose features
        keypoints_flat = keypoints.reshape(T, -1).astype(np.float32)
        keypoints_tensor = torch.from_numpy(keypoints_flat).to(self.device)
        
        with torch.no_grad():
            pose_features = self.pose_extractor(keypoints_tensor).cpu().numpy()
        
        # Fusionar
        fused = np.concatenate([visual_features, pose_features], axis=1)
        return torch.from_numpy(fused).float()
    
    def predict(self, features: torch.Tensor) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predice la seña
        
        Returns:
            class_name, confidence, top_3_predictions
        """
        features = features.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(features)
            probs = torch.softmax(logits, dim=1)[0]
        
        top_3_probs, top_3_indices = torch.topk(probs, k=3)
        
        predictions = []
        for prob, idx in zip(top_3_probs, top_3_indices):
            class_id = idx.item()
            class_name = self.id_to_class.get(class_id, f"Unknown_{class_id}")
            predictions.append((class_name, prob.item()))
        
        top_class, top_conf = predictions[0]
        
        return top_class, top_conf, predictions
    
    def draw_ui(self, frame: np.ndarray, prediction: Optional[Tuple[str, float]] = None):
        """Dibuja UI en el frame"""
        h, w = frame.shape[:2]
        
        # Fondo semi-transparente para el texto
        overlay = frame.copy()
        
        # Activity meter
        if self.rt_config.show_activity_meter:
            activity = self.detector.get_current_activity()
            meter_width = 300
            meter_height = 30
            meter_x = w - meter_width - 20
            meter_y = 20
            
            # Background
            cv2.rectangle(overlay, (meter_x, meter_y), 
                         (meter_x + meter_width, meter_y + meter_height),
                         (0, 0, 0), -1)
            
            # Fill
            fill_width = int(meter_width * activity)
            color = (0, int(255 * activity), int(255 * (1 - activity)))
            cv2.rectangle(overlay, (meter_x, meter_y),
                         (meter_x + fill_width, meter_y + meter_height),
                         color, -1)
            
            # Border
            cv2.rectangle(overlay, (meter_x, meter_y),
                         (meter_x + meter_width, meter_y + meter_height),
                         (255, 255, 255), 2)
            
            # Text
            cv2.putText(overlay, f"Activity: {activity:.0%}",
                       (meter_x, meter_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Predicción actual
        if prediction is not None:
            sign, confidence = prediction
            
            # Background
            text_y = 80
            cv2.rectangle(overlay, (10, text_y - 40), (w - 10, text_y + 20),
                         (0, 0, 0), -1)
            
            # Text
            text = f"Seña: {sign}"
            conf_text = f"Confianza: {confidence:.1%}"
            
            cv2.putText(overlay, text, (20, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(overlay, conf_text, (20, text_y + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Historial de traducción
        if len(self.translation_history) > 0:
            history_y = h - 150
            cv2.rectangle(overlay, (10, history_y), (w - 10, h - 10),
                         (0, 0, 0), -1)
            
            cv2.putText(overlay, "Historial:", (20, history_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            for i, (sign, conf) in enumerate(self.translation_history[-5:]):
                y = history_y + 55 + i * 25
                text = f"{i+1}. {sign} ({conf:.0%})"
                cv2.putText(overlay, text, (30, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Instrucciones
        instructions = [
            "Presiona 'q' para salir",
            "Presiona 'r' para resetear",
            "Presiona 's' para screenshot"
        ]
        
        inst_y = h - 100
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (20, inst_y + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Blend overlay
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
    
    def run(self, camera_id: int = 0, save_predictions: bool = True):
        """
        Ejecuta el loop de inferencia en tiempo real
        
        Args:
            camera_id: ID de la cámara (0 para webcam default)
            save_predictions: Si guardar predicciones en archivo
        """
        logger.info(f"\n{'='*60}")
        logger.info("INICIANDO INFERENCIA EN TIEMPO REAL")
        logger.info(f"{'='*60}\n")
        logger.info("Controles:")
        logger.info("  'q' - Salir")
        logger.info("  'r' - Resetear buffer")
        logger.info("  's' - Capturar screenshot")
        logger.info(f"\n{'='*60}\n")
        
        # Abrir cámara
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"No se pudo abrir la cámara {camera_id}")
            return
        
        # Configurar resolución
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Cámara inicializada - FPS: {fps:.1f}")
        
        current_prediction = None
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("No se pudo leer frame de la cámara")
                    break
                
                frame_count += 1
                
                # Actualizar detector
                ready, frames, keypoints = self.detector.update(frame)
                
                # Predecir si está listo
                if ready:
                    logger.info(f"\nDetectada actividad - Prediciendo...")
                    
                    try:
                        # Extraer features
                        features = self.extract_features(frames, keypoints)
                        
                        # Predecir
                        sign, confidence, top_3 = self.predict(features)
                        
                        logger.info(f"  Predicción: {sign} ({confidence:.1%})")
                        logger.info(f"  Top-3: {top_3}")
                        
                        current_prediction = (sign, confidence)
                        
                        # Agregar al historial
                        self.translation_history.append((sign, confidence))
                        if len(self.translation_history) > self.max_history:
                            self.translation_history.pop(0)
                        
                    except Exception as e:
                        logger.error(f"Error en predicción: {e}")
                
                # Dibujar UI
                display_frame = self.draw_ui(frame, current_prediction)
                
                # Mostrar FPS
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(display_frame, f"FPS: {current_fps:.1f}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Mostrar frame
                cv2.imshow("Sign Language Translator", display_frame)
                
                # Procesar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("Saliendo...")
                    break
                elif key == ord('r'):
                    logger.info("Reseteando buffer...")
                    self.detector.reset()
                    current_prediction = None
                elif key == ord('s'):
                    screenshot_path = Path(f"screenshot_{int(time.time())}.jpg")
                    cv2.imwrite(str(screenshot_path), display_frame)
                    logger.info(f"Screenshot guardado: {screenshot_path}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Guardar historial
            if save_predictions and len(self.translation_history) > 0:
                output_file = Path(f"translation_history_{int(time.time())}.json")
                history_data = {
                    'timestamp': time.time(),
                    'total_predictions': len(self.translation_history),
                    'predictions': [
                        {'sign': sign, 'confidence': float(conf)}
                        for sign, conf in self.translation_history
                    ]
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(history_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"\nHistorial guardado: {output_file}")
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Sesión finalizada")
            logger.info(f"  Frames procesados: {frame_count}")
            logger.info(f"  Predicciones: {len(self.translation_history)}")
            logger.info(f"  Tiempo total: {elapsed:.1f}s")
            logger.info(f"{'='*60}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Traducción de señas en tiempo real")
    parser.add_argument("--model_path", type=Path,
                       default=config.model_paths.temporal_checkpoints / "best_model.pt")
    parser.add_argument("--visual_extractor", type=Path,
                       default=Path("models/extractors/visual_extractor_full.pt"))
    parser.add_argument("--pose_extractor", type=Path,
                       default=Path("models/extractors/pose_extractor_full.pt"))
    parser.add_argument("--class_mapping", type=Path,
                       default=config.data_paths.dataset_meta)
    parser.add_argument("--camera_id", type=int, default=0,
                       help="ID de la cámara")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"])
    
    # Configuración de detección
    parser.add_argument("--buffer_size", type=int, default=48)
    parser.add_argument("--min_frames", type=int, default=24)
    parser.add_argument("--activity_threshold", type=float, default=0.35)
    parser.add_argument("--cooldown_frames", type=int, default=15)
    
    args = parser.parse_args()
    
    # Crear configuración
    rt_config = RealtimeConfig(
        buffer_size=args.buffer_size,
        min_frames_to_predict=args.min_frames,
        activity_threshold=args.activity_threshold,
        cooldown_frames=args.cooldown_frames
    )
    
    # Crear traductor
    translator = RealtimeSignTranslator(
        model_path=args.model_path,
        visual_extractor_path=args.visual_extractor,
        pose_extractor_path=args.pose_extractor,
        class_mapping_path=args.class_mapping,
        device=args.device,
        realtime_config=rt_config
    )
    
    # Ejecutar
    translator.run(camera_id=args.camera_id)


if __name__ == "__main__":
    main()