"""
Modulo de inferencia para videos y camara en tiempo real.
Coherente con el pipeline de entrenamiento (misma normalizacion, misma arquitectura).
"""

import cv2  # type: ignore
import torch  # type: ignore
import numpy as np  # type: ignore
import mediapipe as mp  # type: ignore
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from torchvision import transforms  # type: ignore
import logging
from collections import deque
import time

from pipelines_video.config import config
from pipelines_video.models import get_video_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoInferenceEngine:
    """
    Motor de inferencia para videos y camara en tiempo real.
    
    Caracteristicas:
    - Misma normalizacion que entrenamiento
    - Buffer de frames para secuencias
    - Soporte para inferencia continua (camara)
    - Sin asunciones de longitud fija
    """
    
    def __init__(
        self,
        checkpoint_path: Path,
        visual_extractor_path: Path = None,
        pose_extractor_path: Path = None,
        class_names_path: Path = None,
        device: str = None,
        skip_start_percent: float = None,
        max_frames: int = None,
        model_type: str = "lstm"
    ):
        """
        Args:
            checkpoint_path: Ruta al modelo temporal entrenado
            visual_extractor_path: Ruta al extractor visual
            pose_extractor_path: Ruta al extractor de pose
            class_names_path: Ruta a JSON con {class_id: class_name}
            device: 'cuda' o 'cpu'
            skip_start_percent: Porcentaje inicial a ignorar
            max_frames: Maximo de frames a procesar
            model_type: Tipo de modelo ('lstm' o 'transformer')
        """
        self.device = torch.device(
            device or (config.training.device if torch.cuda.is_available() else "cpu")
        )
        self.skip_start_percent = skip_start_percent or config.video.skip_start_percent
        self.max_frames = max_frames or config.video.max_frames
        self.model_type = model_type
        
        # Cargar modelos
        self._load_models(
            checkpoint_path,
            visual_extractor_path or config.data_paths.visual_extractor,
            pose_extractor_path or config.data_paths.pose_extractor
        )
        
        # Cargar nombres de clases
        self.class_names = self._load_class_names(class_names_path)
        
        # MediaPipe
        self._init_mediapipe()
        
        # Transform para visual
        self.visual_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        logger.info("VideoInferenceEngine inicializado")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Max frames: {self.max_frames}")
        logger.info(f"  Skip start: {self.skip_start_percent * 100:.0f}%")
    
    def _load_models(
        self,
        checkpoint_path: Path,
        visual_extractor_path: Path,
        pose_extractor_path: Path
    ):
        """Carga todos los modelos necesarios"""
        logger.info("Cargando modelos...")
        
        # Temporal classifier
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Obtener config del checkpoint
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            model_config = checkpoint['config']
            num_classes = model_config.get('num_classes', config.model.num_classes)
        else:
            num_classes = config.model.num_classes
        
        self.temporal_model = get_video_model(
            model_type=self.model_type,
            num_classes=num_classes
        ).to(self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.temporal_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.temporal_model.load_state_dict(checkpoint)
        
        self.temporal_model.eval()
        logger.info("  Modelo temporal cargado")
        
        # Visual extractor
        self.visual_extractor = torch.load(
            visual_extractor_path, map_location=self.device
        )
        self.visual_extractor.eval()
        logger.info("  Extractor visual cargado")
        
        # Pose extractor
        self.pose_extractor = torch.load(
            pose_extractor_path, map_location=self.device
        )
        self.pose_extractor.eval()
        logger.info("  Extractor de pose cargado")
    
    def _init_mediapipe(self):
        """Inicializa MediaPipe"""
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True
        )
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
    
    def _load_class_names(self, path: Optional[Path]) -> Dict[int, str]:
        """Carga mapeo de class_id a class_name"""
        if path is None or not Path(path).exists():
            return {}
        
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convertir keys a int
        return {int(k): v for k, v in data.items()}
    
    def _extract_keypoints(self, frame: np.ndarray) -> np.ndarray:
        """Extrae keypoints de un frame"""
        # Pose
        pose_results = self.pose.process(frame)
        
        if pose_results.pose_landmarks:
            pose_kpts = np.array([
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in pose_results.pose_landmarks.landmark
            ])
            pose_kpts = self._normalize_pose(pose_kpts)
        else:
            pose_kpts = np.zeros((33, 4), dtype=np.float32)
        
        # Hands
        hand_results = self.hands.process(frame)
        hand_kpts = np.zeros((42, 4), dtype=np.float32)
        
        if hand_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks[:2]):
                start_idx = hand_idx * 21
                hand_kpts[start_idx:start_idx + 21] = np.array([
                    [lm.x, lm.y, lm.z, 1.0]
                    for lm in hand_landmarks.landmark
                ])
        
        return np.concatenate([pose_kpts, hand_kpts], axis=0)  # (75, 4)
    
    def _normalize_pose(self, pose_kpts: np.ndarray) -> np.ndarray:
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
    
    @torch.no_grad()
    def _extract_features(
        self,
        frames: np.ndarray,
        keypoints: np.ndarray
    ) -> np.ndarray:
        """
        Extrae features fusionadas de frames y keypoints.
        
        Args:
            frames: (T, H, W, 3) RGB
            keypoints: (T, 75, 4)
            
        Returns:
            features: (T, 1152)
        """
        T = len(frames)
        
        # Visual features
        visual_features = []
        for frame in frames:
            tensor = self.visual_transform(frame).unsqueeze(0).to(self.device)
            feat = self.visual_extractor(tensor)  # (1, 1024)
            visual_features.append(feat.cpu().numpy())
        
        visual_features = np.concatenate(visual_features, axis=0)  # (T, 1024)
        
        # Pose features
        keypoints_flat = keypoints.reshape(T, -1)  # (T, 300)
        kpts_tensor = torch.from_numpy(keypoints_flat).float().to(self.device)
        pose_features = self.pose_extractor(kpts_tensor).cpu().numpy()  # (T, 128)
        
        # Fuse
        fused = np.concatenate([visual_features, pose_features], axis=1)  # (T, 1152)
        
        return fused
    
    @torch.no_grad()
    def predict_video(
        self,
        video_path: Path,
        top_k: int = 5
    ) -> Tuple[List[Tuple[int, str, float]], np.ndarray]:
        """
        Predice clase de un video completo.
        
        Args:
            video_path: Ruta al video
            top_k: Numero de predicciones top a retornar
            
        Returns:
            predictions: Lista de (class_id, class_name, probability)
            features: (T, 1152) features extraidas
        """
        # Abrir video
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            logger.error(f"Video vacio: {video_path}")
            return [], np.array([])
        
        # Calcular rango
        start_frame = int(total_frames * self.skip_start_percent)
        end_frame = total_frames
        useful_frames = end_frame - start_frame
        
        # Indices a extraer
        num_frames = min(useful_frames, self.max_frames)
        
        if useful_frames <= self.max_frames:
            frame_indices = list(range(start_frame, end_frame))
        else:
            frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int).tolist()
        
        # Extraer frames
        frames = []
        keypoints = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (config.video.frame_width, config.video.frame_height))
                
                frames.append(frame_resized)
                keypoints.append(self._extract_keypoints(frame_rgb))
        
        cap.release()
        
        if len(frames) == 0:
            return [], np.array([])
        
        frames = np.array(frames)
        keypoints = np.array(keypoints)
        
        # Extraer features
        features = self._extract_features(frames, keypoints)
        
        # Inferencia
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        lengths = torch.tensor([len(features)], dtype=torch.long).to(self.device)
        
        logits = self.temporal_model(features_tensor, lengths)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        
        # Top-K
        top_indices = np.argsort(probs)[-top_k:][::-1]
        predictions = []
        
        for idx in top_indices:
            class_name = self.class_names.get(idx, f"Class_{idx}")
            predictions.append((int(idx), class_name, float(probs[idx])))
        
        return predictions, features
    
    def close(self):
        """Libera recursos"""
        self.pose.close()
        self.hands.close()


class RealtimeInferenceEngine:
    """
    Motor de inferencia en tiempo real para camara.
    Mantiene un buffer de frames y hace predicciones continuas.
    """
    
    def __init__(
        self,
        checkpoint_path: Path,
        visual_extractor_path: Path = None,
        pose_extractor_path: Path = None,
        class_names_path: Path = None,
        device: str = None,
        buffer_size: int = 45,
        prediction_interval: float = 0.5,
        model_type: str = "lstm"
    ):
        """
        Args:
            checkpoint_path: Ruta al modelo entrenado
            buffer_size: Tamano del buffer de frames
            prediction_interval: Segundos entre predicciones
        """
        self.buffer_size = buffer_size
        self.prediction_interval = prediction_interval
        
        # Engine base
        self.engine = VideoInferenceEngine(
            checkpoint_path=checkpoint_path,
            visual_extractor_path=visual_extractor_path,
            pose_extractor_path=pose_extractor_path,
            class_names_path=class_names_path,
            device=device,
            max_frames=buffer_size,
            model_type=model_type
        )
        
        # Buffers
        self.frame_buffer = deque(maxlen=buffer_size)
        self.keypoint_buffer = deque(maxlen=buffer_size)
        
        self.last_prediction_time = 0
        self.current_prediction = None
        
        logger.info(f"RealtimeInferenceEngine: buffer={buffer_size}, interval={prediction_interval}s")
    
    def process_frame(
        self,
        frame: np.ndarray,
        force_predict: bool = False
    ) -> Optional[List[Tuple[int, str, float]]]:
        """
        Procesa un frame y retorna prediccion si es tiempo.
        
        Args:
            frame: Frame BGR de OpenCV
            force_predict: Forzar prediccion inmediata
            
        Returns:
            predictions: Lista de (class_id, class_name, prob) o None
        """
        # Convertir a RGB y resize
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (config.video.frame_width, config.video.frame_height))
        
        # Extraer keypoints
        keypoints = self.engine._extract_keypoints(frame_rgb)
        
        # Agregar a buffers
        self.frame_buffer.append(frame_resized)
        self.keypoint_buffer.append(keypoints)
        
        # Verificar si es tiempo de predecir
        current_time = time.time()
        should_predict = (
            force_predict or
            (current_time - self.last_prediction_time) >= self.prediction_interval
        )
        
        if should_predict and len(self.frame_buffer) >= 10:  # Minimo 10 frames
            self.last_prediction_time = current_time
            return self._predict_from_buffer()
        
        return None
    
    @torch.no_grad()
    def _predict_from_buffer(self) -> List[Tuple[int, str, float]]:
        """Hace prediccion con el contenido actual del buffer"""
        frames = np.array(list(self.frame_buffer))
        keypoints = np.array(list(self.keypoint_buffer))
        
        # Extraer features
        features = self.engine._extract_features(frames, keypoints)
        
        # Inferencia
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.engine.device)
        lengths = torch.tensor([len(features)], dtype=torch.long).to(self.engine.device)
        
        logits = self.engine.temporal_model(features_tensor, lengths)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        
        # Top-5
        top_indices = np.argsort(probs)[-5:][::-1]
        predictions = []
        
        for idx in top_indices:
            class_name = self.engine.class_names.get(idx, f"Class_{idx}")
            predictions.append((int(idx), class_name, float(probs[idx])))
        
        self.current_prediction = predictions
        return predictions
    
    def clear_buffer(self):
        """Limpia los buffers"""
        self.frame_buffer.clear()
        self.keypoint_buffer.clear()
        self.current_prediction = None
    
    def close(self):
        """Libera recursos"""
        self.engine.close()


def demo_video_inference(
    video_path: Path,
    checkpoint_path: Path,
    visual_extractor_path: Path = None,
    pose_extractor_path: Path = None
):
    """Demo de inferencia en video"""
    
    engine = VideoInferenceEngine(
        checkpoint_path=checkpoint_path,
        visual_extractor_path=visual_extractor_path,
        pose_extractor_path=pose_extractor_path
    )
    
    predictions, features = engine.predict_video(video_path)
    
    print(f"\nResultados para: {video_path}")
    print("-" * 40)
    for class_id, class_name, prob in predictions:
        print(f"  {class_name} (ID: {class_id}): {prob:.2%}")
    print(f"\nFeatures shape: {features.shape}")
    
    engine.close()


def demo_realtime_inference(
    checkpoint_path: Path,
    visual_extractor_path: Path = None,
    pose_extractor_path: Path = None,
    camera_id: int = 0
):
    """Demo de inferencia en tiempo real"""
    
    engine = RealtimeInferenceEngine(
        checkpoint_path=checkpoint_path,
        visual_extractor_path=visual_extractor_path,
        pose_extractor_path=pose_extractor_path,
        buffer_size=45,
        prediction_interval=1.0
    )
    
    cap = cv2.VideoCapture(camera_id)
    
    print("Presiona 'q' para salir, 'c' para limpiar buffer")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Procesar frame
        predictions = engine.process_frame(frame)
        
        # Mostrar prediccion actual
        if engine.current_prediction:
            y_offset = 30
            for i, (class_id, class_name, prob) in enumerate(engine.current_prediction[:3]):
                text = f"{class_name}: {prob:.1%}"
                cv2.putText(frame, text, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Info del buffer
        buffer_info = f"Buffer: {len(engine.frame_buffer)}/{engine.buffer_size}"
        cv2.putText(frame, buffer_info, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('ASL Recognition', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            engine.clear_buffer()
    
    cap.release()
    cv2.destroyAllWindows()
    engine.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inferencia de video")
    parser.add_argument("--mode", type=str, default="video",
                        choices=["video", "realtime"])
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--visual_extractor", type=Path, default=None)
    parser.add_argument("--pose_extractor", type=Path, default=None)
    parser.add_argument("--camera", type=int, default=0)
    
    args = parser.parse_args()
    
    if args.mode == "video":
        if args.video is None:
            print("Error: --video requerido para modo video")
        else:
            demo_video_inference(
                video_path=args.video,
                checkpoint_path=args.checkpoint,
                visual_extractor_path=args.visual_extractor,
                pose_extractor_path=args.pose_extractor
            )
    else:
        demo_realtime_inference(
            checkpoint_path=args.checkpoint,
            visual_extractor_path=args.visual_extractor,
            pose_extractor_path=args.pose_extractor,
            camera_id=args.camera
        )
