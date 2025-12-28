"""
Script de inferencia para procesar videos pregrabados
Detecta múltiples señas (oraciones) en videos continuos
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
import json
import sys
from tqdm import tqdm

# Agregar el directorio raíz al path
sys.path.append(str(Path(__file__).parent.parent))

from pipelines.save_extractors import ResNet101FeatureExtractor, PoseFeatureExtractor
from pipelines.models_temporal import get_temporal_model
from config import config
import mediapipe as mp
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoSignSegmenter:
    """Segmenta un video continuo en clips de señas individuales"""
    
    def __init__(self, min_sign_length: int = 12, max_sign_length: int = 60):
        self.min_sign_length = min_sign_length
        self.max_sign_length = max_sign_length
    
    def calculate_motion_scores(self, keypoints_sequence: np.ndarray) -> np.ndarray:
        """
        Calcula score de actividad por frame
        
        Args:
            keypoints_sequence: (T, 75, 4)
        
        Returns:
            scores: (T,) - score de actividad por frame
        """
        T = len(keypoints_sequence)
        scores = np.zeros(T)
        
        for t in range(T):
            kpts = keypoints_sequence[t]
            
            # Pose y manos
            pose_kpts = kpts[:33]
            left_hand_kpts = kpts[33:54]
            right_hand_kpts = kpts[54:75]
            
            score = 0.0
            
            # 1. Visibilidad de manos
            left_vis = np.mean(left_hand_kpts[:, 3])
            right_vis = np.mean(right_hand_kpts[:, 3])
            hand_visibility = (left_vis + right_vis) / 2.0
            score += 0.3 * hand_visibility
            
            # 2. Manos levantadas (arriba de caderas)
            hip_y = np.mean([pose_kpts[23, 1], pose_kpts[24, 1]])
            left_hand_y = np.mean(left_hand_kpts[:, 1])
            right_hand_y = np.mean(right_hand_kpts[:, 1])
            
            hands_raised = 0.0
            if left_vis > 0.3 and left_hand_y < hip_y:
                hands_raised += 0.5
            if right_vis > 0.3 and right_hand_y < hip_y:
                hands_raised += 0.5
            score += 0.4 * hands_raised
            
            # 3. Movimiento (si hay frame anterior)
            if t > 0:
                prev_left = np.mean(keypoints_sequence[t-1][33:54, :3], axis=0)
                curr_left = np.mean(left_hand_kpts[:, :3], axis=0)
                left_motion = np.linalg.norm(curr_left - prev_left)
                
                prev_right = np.mean(keypoints_sequence[t-1][54:75, :3], axis=0)
                curr_right = np.mean(right_hand_kpts[:, :3], axis=0)
                right_motion = np.linalg.norm(curr_right - prev_right)
                
                motion = np.clip((left_motion + right_motion) / 0.1, 0, 1)
                score += 0.3 * motion
            
            scores[t] = np.clip(score, 0, 1)
        
        return scores
    
    def segment_signs(self, keypoints_sequence: np.ndarray) -> List[Tuple[int, int]]:
        """
        Segmenta el video en intervalos de señas
        
        Returns:
            Lista de (start_idx, end_idx) para cada seña detectada
        """
        scores = self.calculate_motion_scores(keypoints_sequence)
        T = len(scores)
        
        # Suavizar scores
        window_size = 5
        smoothed = np.convolve(scores, np.ones(window_size) / window_size, mode='same')
        
        # Umbral adaptativo
        threshold = np.percentile(smoothed, 40)
        
        # Encontrar regiones activas
        active = smoothed > threshold
        
        segments = []
        in_segment = False
        segment_start = 0
        
        for t in range(T):
            if active[t] and not in_segment:
                # Inicio de segmento
                segment_start = t
                in_segment = True
            elif not active[t] and in_segment:
                # Fin de segmento
                segment_end = t
                segment_length = segment_end - segment_start
                
                # Validar longitud
                if segment_length >= self.min_sign_length:
                    # Truncar si es muy largo
                    if segment_length > self.max_sign_length:
                        segment_end = segment_start + self.max_sign_length
                    
                    segments.append((segment_start, segment_end))
                
                in_segment = False
        
        # Cerrar último segmento si está abierto
        if in_segment:
            segment_end = T
            segment_length = segment_end - segment_start
            if segment_length >= self.min_sign_length:
                if segment_length > self.max_sign_length:
                    segment_end = segment_start + self.max_sign_length
                segments.append((segment_start, segment_end))
        
        logger.info(f"Detectados {len(segments)} segmentos de señas en el video")
        return segments


class VideoInference:
    """Sistema de inferencia para videos pregrabados"""
    
    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        device: str = "cuda",
        confidence_threshold: float = 0.05,  # Reducido threshold por defecto de 0.3 a 0.05
        use_segmentation: bool = True  # Agregado parámetro para controlar segmentación
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.use_segmentation = use_segmentation  # Guardar configuración
        
        logger.info(f"Inicializando VideoInference en {self.device}")
        logger.info(f"Segmentación automática: {'ACTIVADA' if use_segmentation else 'DESACTIVADA'}")  # Log de configuración
        logger.info(f"Confidence threshold: {confidence_threshold:.2%}")  # Log threshold
        
        # Cargar mapeo de clases
        self.class_names = self._load_class_names(metadata_path)
        self.num_classes = len(self.class_names)
        logger.info(f"Clases cargadas: {self.num_classes}")
        
        self.visual_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Cargar extractores
        self.visual_extractor = self._load_visual_extractor()
        self.pose_extractor = self._load_pose_extractor()
        
        # Cargar modelo temporal
        self.model = self._load_temporal_model(model_path)
        
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
        
        # Segmentador
        self.segmenter = VideoSignSegmenter(
            min_sign_length=12,
            max_sign_length=60
        )
        
        logger.info("Sistema de inferencia inicializado correctamente")
    
    def _load_class_names(self, metadata_path: Path) -> Dict[int, str]:
        """Carga el mapeo class_id -> class_name"""
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
        extractor = torch.load(extractor_path, map_location=self.device, weights_only=False)
        extractor.eval()
        return extractor
    
    def _load_pose_extractor(self) -> PoseFeatureExtractor:
        """Carga el extractor de pose"""
        extractor_path = Path("models/extractors/pose_extractor_full.pt")
        extractor = torch.load(extractor_path, map_location=self.device, weights_only=False)
        extractor.eval()
        return extractor
    
    def _load_temporal_model(self, model_path: Path):
        """Carga el modelo temporal"""
        model = get_temporal_model(
            model_type=config.training.model_type,
            num_classes=self.num_classes,
            hidden_dim=config.training.model_hidden_dim,
            num_layers=config.training.model_num_layers,
            dropout=0.0,
            bidirectional=config.training.model_bidirectional,
            use_attention=config.training.use_attention
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    
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
        """Normaliza keypoints"""
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
        """Extrae features fusionadas (1152 dims)"""
        # Visual features (1024)
        frame_resized = cv2.resize(frame, (224, 224))
        
        logger.debug(f"[v0] Frame shape: {frame_resized.shape}, dtype: {frame_resized.dtype}")
        logger.debug(f"[v0] Frame value range: min={frame_resized.min()}, max={frame_resized.max()}, mean={frame_resized.mean():.2f}")
        
        # Ensure frame is uint8 in range [0, 255]
        if frame_resized.dtype != np.uint8:
            if frame_resized.max() <= 1.0:
                frame_resized = (frame_resized * 255).astype(np.uint8)
            else:
                frame_resized = frame_resized.astype(np.uint8)
        
        logger.debug(f"[v0] Frame after uint8: dtype={frame_resized.dtype}, min={frame_resized.min()}, max={frame_resized.max()}")
        
        # Apply the same transform as in training (ToTensor + Normalize)
        # Note: ToTensor does NOT convert BGR to RGB, it just changes HWC to CHW
        frame_tensor = self.visual_transform(frame_resized).unsqueeze(0).to(self.device)
        
        logger.debug(f"[v0] Frame tensor after transform: shape={frame_tensor.shape}, dtype={frame_tensor.dtype}")
        logger.debug(f"[v0] Frame tensor value range: min={frame_tensor.min().item():.4f}, max={frame_tensor.max().item():.4f}, mean={frame_tensor.mean().item():.4f}")
        
        with torch.no_grad():
            visual_features = self.visual_extractor(frame_tensor).cpu().numpy().squeeze()  # (1024,)
        
        logger.debug(f"[v0] Visual features: shape={visual_features.shape}, dtype={visual_features.dtype}")

        # Pose features (128)
        keypoints_flat = keypoints.flatten()[:300]
        keypoints_tensor = torch.from_numpy(keypoints_flat).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            pose_features = self.pose_extractor(keypoints_tensor).cpu().numpy().squeeze()  # (128,)
        
        logger.debug(f"[v0] Pose features: shape={pose_features.shape}, dtype={pose_features.dtype}")
        logger.debug(f"[v0] Pose features stats: min={pose_features.min():.4f}, max={pose_features.max():.4f}, mean={pose_features.mean():.4f}, std={pose_features.std():.4f}")
        
        # Fusionar: (1024,) + (128,) = (1152,)
        fused_features = np.concatenate([visual_features, pose_features]).astype(np.float32)
        
        logger.debug(f"[v0] Fused features: shape={fused_features.shape}, dtype={fused_features.dtype}")
        logger.debug(f"[v0] Fused features stats: min={fused_features.min():.4f}, max={fused_features.max():.4f}, mean={fused_features.mean():.4f}, std={fused_features.std():.4f}")
        
        return fused_features
    
    @torch.no_grad()
    def predict_sign(self, features_sequence: np.ndarray) -> Tuple[int, float, List[Tuple[int, float]]]:
        """Predice una seña"""
        # Convertir a tensor float32
        features_tensor = torch.from_numpy(features_sequence).unsqueeze(0).float().to(self.device)
        lengths = torch.tensor([len(features_sequence)], dtype=torch.long).to(self.device)
        
        logger.debug(f"[v0] Input shape: {features_tensor.shape}, dtype: {features_tensor.dtype}")
        
        # Forward pass
        logits = self.model(features_tensor, lengths)  # (1, num_classes)
        
        # Aplicar softmax correctamente
        probs = torch.softmax(logits, dim=-1).cpu().numpy().squeeze()
        
        # Top-5 predictions
        top5_indices = np.argsort(probs)[-5:][::-1]
        top5_predictions = [(int(idx), float(probs[idx])) for idx in top5_indices]
        
        predicted_class = int(top5_indices[0])
        confidence = float(probs[predicted_class])
        
        logger.debug(f"[v0] Top prediction: class={predicted_class}, confidence={confidence:.4f}")
        logger.debug(f"[v0] Top-5: {[(idx, f'{conf:.4f}') for idx, conf in top5_predictions]}")
        
        return predicted_class, confidence, top5_predictions
    
    def process_video(self, video_path: Path, output_path: Optional[Path] = None) -> List[Tuple[str, float]]:
        """
        Procesa un video completo y retorna la oración detectada
        
        Returns:
            sentence: Lista de (sign_name, confidence)
        """
        logger.info(f"Procesando video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"No se pudo abrir el video: {video_path}")
            return []
        
        # Leer todos los frames
        frames = []
        keypoints_sequence = []
        
        logger.info("Extrayendo frames y keypoints...")
        with tqdm(desc="Extrayendo frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                kpts = self.extract_keypoints(frame)
                
                if kpts is not None:
                    frames.append(frame)
                    keypoints_sequence.append(kpts)
                
                pbar.update(1)
        
        cap.release()
        
        if len(frames) == 0:
            logger.warning("No se detectaron keypoints en el video")
            return []
        
        logger.info(f"Frames procesados: {len(frames)}")
        
        if self.use_segmentation:
            # Segmentar video en señas
            keypoints_array = np.array(keypoints_sequence)
            sign_segments = self.segmenter.segment_signs(keypoints_array)
            
            if len(sign_segments) == 0:
                logger.warning("No se detectaron segmentos. Intentando procesar video completo...")
                sign_segments = [(0, len(frames))]
        else:
            # Procesar video completo como una seña
            logger.info("Procesando video completo (sin segmentación)")
            sign_segments = [(0, len(frames))]
        
        # Procesar cada segmento
        sentence = []
        
        logger.info(f"Procesando {len(sign_segments)} señas detectadas...")
        
        for idx, (start, end) in enumerate(sign_segments, 1):
            segment_length = end - start
            logger.info(f"\n{'='*60}")
            logger.info(f"Seña {idx}/{len(sign_segments)}")
            logger.info(f"  Frames: {start}-{end} (longitud: {segment_length})")
            logger.info(f"{'='*60}")
            
            # Extraer features para este segmento
            segment_features = []
            
            for frame_idx in range(start, end):
                frame = frames[frame_idx]
                kpts = keypoints_sequence[frame_idx]
                features = self.extract_features(frame, kpts)
                segment_features.append(features)
            
            segment_features = np.array(segment_features)
            
            logger.info(f"  Features extraídas: shape={segment_features.shape}")
            logger.info(f"  Features stats: mean={segment_features.mean():.4f}, std={segment_features.std():.4f}")
            
            # Predecir
            class_id, confidence, top5 = self.predict_sign(segment_features)
            
            class_name = self.class_names.get(class_id, f"Unknown_{class_id}")
            
            logger.info(f"\n  Predicción:")
            logger.info(f"    Top-1: {class_name} ({confidence:.2%})")
            logger.info(f"    Top-5:")
            for rank, (idx, conf) in enumerate(top5[:5], 1):
                sign_name = self.class_names.get(idx, f"Unknown_{idx}")
                logger.info(f"      {rank}. {sign_name}: {conf:.2%}")
            
            if confidence >= self.confidence_threshold:
                sentence.append((class_name, confidence))
                logger.info(f"\n  ✓ Seña aceptada (threshold: {self.confidence_threshold:.2%})")
            else:
                logger.warning(f"\n  ✗ Confianza baja ({confidence:.2%} < {self.confidence_threshold:.2%}), seña ignorada")
        
        logger.info(f"\n{'='*60}")
        if sentence:
            logger.info(f"RESULTADO FINAL: {len(sentence)} señas detectadas")
            logger.info(f"Oración: {' '.join([sign for sign, _ in sentence])}")
        else:
            logger.warning("NO SE DETECTARON SEÑAS en el video")
            logger.info("Intenta:")
            logger.info("  1. Reducir --confidence (ej: --confidence 0.01)")
            logger.info("  2. Usar --no-segmentation para procesar video completo")
        logger.info(f"{'='*60}\n")
        
        # Guardar resultados si se especifica output
        if output_path:
            self._save_results(video_path, sentence, output_path)
        
        return sentence
    
    def _save_results(self, video_path: Path, sentence: List[Tuple[str, float]], output_path: Path):
        """Guarda los resultados en formato JSON"""
        results = {
            "video_path": str(video_path),
            "detected_signs": [
                {"sign": sign, "confidence": float(conf)}
                for sign, conf in sentence
            ],
            "sentence": " ".join([sign for sign, _ in sentence])
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Resultados guardados en: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Inferencia en videos pregrabados")
    parser.add_argument("--model", type=Path, required=True,
                       help="Ruta al modelo entrenado (.pt)")
    parser.add_argument("--video", type=Path, required=True,
                       help="Ruta al video a procesar")
    parser.add_argument("--metadata", type=Path, default=Path("data/dataset_meta.json"),
                       help="Ruta a dataset_meta.json con mapeo de clases")
    parser.add_argument("--output", type=Path, default=None,
                       help="Ruta para guardar resultados (JSON)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device: cuda o cpu (default: cuda)")
    parser.add_argument("--confidence", type=float, default=0.05,  # Reducido threshold por defecto
                       help="Confidence threshold para aceptar predicciones (default: 0.05)")
    parser.add_argument("--no-segmentation", action="store_true",  # Nuevo parámetro
                       help="Procesar video completo sin segmentación automática")
    
    args = parser.parse_args()
    
    # Inicializar sistema
    inference_system = VideoInference(
        model_path=args.model,
        metadata_path=args.metadata,
        device=args.device,
        confidence_threshold=args.confidence,
        use_segmentation=not args.no_segmentation  # Pasar configuración de segmentación
    )
    
    # Procesar video
    sentence = inference_system.process_video(
        video_path=args.video,
        output_path=args.output
    )
    
    # Mostrar resultado final
    if sentence:
        print(f"\nOración detectada: {' '.join([sign for sign, _ in sentence])}")
    else:
        print("\nNo se detectaron señas en el video")


if __name__ == "__main__":
    main()
