"""
Sistema de inferencia en tiempo real para lenguaje de señas
Soporta:
1. Cámara web en tiempo real (detección continua)
2. Upload de video (procesamiento completo)
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
from collections import deque
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignLanguageDetector:
    """
    Detector de lenguaje de señas continuo
    Procesa videos/streams y detecta señas formando oraciones
    """
    
    def __init__(
        self,
        model_path: str,
        class_names_path: str,
        feature_dim: int = 1152,  # 1024 visual + 128 pose
        num_frames: int = 24,
        device: str = "cuda",
        confidence_threshold: float = 0.3,
        window_stride: int = 12  # Frames entre ventanas (permite overlap)
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.feature_dim = feature_dim
        self.num_frames = num_frames
        self.confidence_threshold = confidence_threshold
        self.window_stride = window_stride
        
        # Cargar modelo
        logger.info(f"Cargando modelo desde {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Importar modelo
        from models_temporal import TemporalLSTMClassifier
        self.model = TemporalLSTMClassifier(
            input_dim=feature_dim,
            hidden_dim=checkpoint.get('hidden_dim', 768),
            num_layers=checkpoint.get('num_layers', 2),
            num_classes=checkpoint.get('num_classes', 2286),
            dropout=0.3,
            bidirectional=True,
            use_attention=checkpoint.get('use_attention', False)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Modelo cargado en {self.device}")
        
        # Cargar nombres de clases
        self.class_names = self._load_class_names(class_names_path)
        logger.info(f"Cargadas {len(self.class_names)} clases")
        
        # Inicializar extractores de features
        self._init_feature_extractors()
        
        # Buffer para frames (ventana deslizante)
        self.frame_buffer = deque(maxlen=num_frames * 2)  # Buffer grande para overlap
        self.keypoint_buffer = deque(maxlen=num_frames * 2)
        
        # Historial de predicciones para suavizado
        self.prediction_history = deque(maxlen=5)
        
    def _load_class_names(self, class_names_path: str) -> List[str]:
        """Carga mapeo de IDs a nombres de señas"""
        import json
        with open(class_names_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            # Si es un diccionario {id: name}
            max_id = max(int(k) for k in data.keys())
            class_names = ["UNKNOWN"] * (max_id + 1)
            for idx, name in data.items():
                class_names[int(idx)] = name
        else:
            # Si es una lista
            class_names = data
        
        return class_names
    
    def _init_feature_extractors(self):
        """Inicializa extractores de features visuales y de pose"""
        # ResNet101 para features visuales
        import torchvision.models as models
        resnet = models.resnet101(pretrained=True)
        self.visual_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.visual_extractor.to(self.device)
        self.visual_extractor.eval()
        
        # Proyección visual de 2048 -> 1024
        self.visual_projection = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU()
        ).to(self.device)
        self.visual_projection.eval()
        
        # MLP para keypoints
        from precompute_pose_features import PoseFeatureExtractor
        self.pose_extractor = PoseFeatureExtractor(
            input_dim=300,  # 75 keypoints * 4 coords (x,y,z,visibility)
            hidden_dim=256,
            output_dim=128
        ).to(self.device)
        
        # MediaPipe para keypoints
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info("Feature extractors inicializados")
    
    def extract_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae keypoints de un frame usando MediaPipe
        
        Returns:
            keypoints: (300,) array [pose(33*4) + hands(42*4) = 75*4 = 300]
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)
        
        keypoints = np.zeros(300)
        
        # Pose landmarks (33 puntos)
        if results.pose_landmarks:
            pose_kpts = np.array([
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in results.pose_landmarks.landmark
            ]).flatten()  # 132 valores
            keypoints[:132] = pose_kpts
        
        # Hand landmarks (21 puntos por mano)
        # Mano izquierda
        if results.left_hand_landmarks:
            left_hand = np.array([
                [lm.x, lm.y, lm.z, 1.0]
                for lm in results.left_hand_landmarks.landmark
            ]).flatten()  # 84 valores
            keypoints[132:216] = left_hand
        
        # Mano derecha
        if results.right_hand_landmarks:
            right_hand = np.array([
                [lm.x, lm.y, lm.z, 1.0]
                for lm in results.right_hand_landmarks.landmark
            ]).flatten()  # 84 valores
            keypoints[216:300] = right_hand
        
        return keypoints if np.any(keypoints) else None
    
    @torch.no_grad()
    def extract_visual_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extrae features visuales de un frame usando ResNet101
        
        Returns:
            features: (1024,) array
        """
        # Preprocesar frame
        frame_resized = cv2.resize(frame, (224, 224))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = (frame_rgb / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        frame_tensor = torch.from_numpy(frame_normalized).float().permute(2, 0, 1).unsqueeze(0)
        frame_tensor = frame_tensor.to(self.device)
        
        # Extraer features
        features_2048 = self.visual_extractor(frame_tensor).squeeze()  # (2048,)
        features_1024 = self.visual_projection(features_2048)  # (1024,)
        
        return features_1024.cpu().numpy()
    
    @torch.no_grad()
    def extract_pose_features(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Extrae features de pose usando MLP
        
        Args:
            keypoints: (300,) array
        
        Returns:
            features: (128,) array
        """
        keypoints_tensor = torch.from_numpy(keypoints).float().unsqueeze(0).to(self.device)
        features = self.pose_extractor(keypoints_tensor).squeeze()  # (128,)
        return features.cpu().numpy()
    
    def process_frame(self, frame: np.ndarray) -> bool:
        """
        Procesa un frame y lo añade al buffer
        
        Returns:
            True si se procesó correctamente, False si no hay keypoints
        """
        # Extraer keypoints
        keypoints = self.extract_keypoints(frame)
        
        if keypoints is None:
            return False
        
        # Extraer features visuales
        visual_features = self.extract_visual_features(frame)
        
        # Extraer features de pose
        pose_features = self.extract_pose_features(keypoints)
        
        # Concatenar features
        fused_features = np.concatenate([visual_features, pose_features])  # (1152,)
        
        # Añadir al buffer
        self.frame_buffer.append(frame)
        self.keypoint_buffer.append(fused_features)
        
        return True
    
    @torch.no_grad()
    def predict_window(self) -> Optional[Tuple[str, float]]:
        """
        Realiza predicción sobre la ventana actual
        
        Returns:
            (sign_name, confidence) o None si no hay suficientes frames
        """
        if len(self.keypoint_buffer) < self.num_frames:
            return None
        
        # Tomar últimos num_frames
        features_window = np.array(list(self.keypoint_buffer)[-self.num_frames:])  # (T, 1152)
        
        # Convertir a tensor
        features_tensor = torch.from_numpy(features_window).float().unsqueeze(0).to(self.device)  # (1, T, 1152)
        
        # Predicción
        logits = self.model(features_tensor)  # (1, num_classes)
        probs = torch.softmax(logits, dim=1)
        
        # Top predicción
        confidence, pred_idx = torch.max(probs, dim=1)
        confidence = confidence.item()
        pred_idx = pred_idx.item()
        
        if confidence < self.confidence_threshold:
            return None
        
        sign_name = self.class_names[pred_idx] if pred_idx < len(self.class_names) else "UNKNOWN"
        
        return sign_name, confidence
    
    def process_video_stream(
        self,
        video_source: int = 0,
        display: bool = True,
        save_output: Optional[str] = None
    ) -> List[Tuple[str, float, float]]:
        """
        Procesa stream de video en tiempo real (cámara o archivo)
        
        Args:
            video_source: 0 para webcam, o ruta a archivo de video
            display: Si True, muestra video con anotaciones
            save_output: Ruta para guardar video anotado (opcional)
        
        Returns:
            Lista de detecciones: [(sign_name, confidence, timestamp), ...]
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            logger.error(f"No se pudo abrir video source: {video_source}")
            return []
        
        # Obtener propiedades del video
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"Procesando video: {width}x{height} @ {fps} FPS")
        
        # Configurar video writer si es necesario
        video_writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
        
        detections = []
        sentence = []  # Oraciones detectadas
        last_sign = None
        last_sign_time = 0
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                current_time = time.time() - start_time
                
                # Procesar frame
                success = self.process_frame(frame)
                
                # Intentar predicción cada window_stride frames
                if frame_count % self.window_stride == 0 and success:
                    prediction = self.predict_window()
                    
                    if prediction:
                        sign_name, confidence = prediction
                        
                        # Evitar duplicados consecutivos (misma seña en ventana temporal)
                        if sign_name != last_sign or (current_time - last_sign_time) > 2.0:
                            detections.append((sign_name, confidence, current_time))
                            sentence.append(sign_name)
                            last_sign = sign_name
                            last_sign_time = current_time
                            
                            logger.info(f"[{current_time:.2f}s] Detectado: {sign_name} ({confidence:.2f})")
                
                # Dibujar en frame
                if display or save_output:
                    # Mostrar última detección
                    if last_sign and (current_time - last_sign_time) < 3.0:
                        cv2.putText(
                            frame,
                            f"{last_sign} ({confidence:.2f})",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                            (0, 255, 0),
                            2
                        )
                    
                    # Mostrar oración acumulada
                    sentence_text = " ".join(sentence[-10:])  # Últimas 10 señas
                    cv2.putText(
                        frame,
                        f"Oracion: {sentence_text}",
                        (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
                    
                    # Indicador de buffer
                    buffer_status = f"Buffer: {len(self.keypoint_buffer)}/{self.num_frames}"
                    cv2.putText(
                        frame,
                        buffer_status,
                        (width - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1
                    )
                
                if display:
                    cv2.imshow("Sign Language Detection", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('c'):  # Clear sentence
                        sentence = []
                
                if video_writer:
                    video_writer.write(frame)
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Mostrar resumen
        logger.info(f"\n{'='*60}")
        logger.info(f"Procesamiento completado")
        logger.info(f"Frames procesados: {frame_count}")
        logger.info(f"Señas detectadas: {len(detections)}")
        logger.info(f"Oración final: {' '.join(sentence)}")
        logger.info(f"{'='*60}\n")
        
        return detections, sentence


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Sistema de inferencia de lenguaje de señas")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ruta al checkpoint del modelo (.pth)"
    )
    parser.add_argument(
        "--classes",
        type=str,
        required=True,
        help="Ruta al archivo de nombres de clases (.json)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Fuente de video: 0 para webcam, ruta para archivo de video"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta para guardar video anotado (opcional)"
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="No mostrar ventana de visualización"
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=1152,
        help="Dimensión de features (1152 para 1024+128, 640 para 512+128)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Umbral de confianza mínimo"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device para inferencia"
    )
    
    args = parser.parse_args()
    
    # Convertir source a int si es número
    try:
        video_source = int(args.source)
    except ValueError:
        video_source = args.source
    
    # Inicializar detector
    detector = SignLanguageDetector(
        model_path=args.model,
        class_names_path=args.classes,
        feature_dim=args.feature_dim,
        device=args.device,
        confidence_threshold=args.confidence
    )
    
    # Procesar video
    detections, sentence = detector.process_video_stream(
        video_source=video_source,
        display=not args.no_display,
        save_output=args.output
    )
    
    # Guardar resultados
    if detections:
        import json
        results = {
            "sentence": " ".join(sentence),
            "detections": [
                {"sign": sign, "confidence": float(conf), "timestamp": float(ts)}
                for sign, conf, ts in detections
            ]
        }
        
        output_json = Path(args.output).with_suffix('.json') if args.output else Path("detections.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Resultados guardados en {output_json}")


if __name__ == "__main__":
    main()
