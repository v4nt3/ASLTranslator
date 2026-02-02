"""
Sistema de inferencia CORREGIDO para videos de lenguaje de señas
Incluye normalización adecuada de keypoints y mejoras en segmentación
"""

import torch
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
import json

from pipelines.config import config
from pipelines.models_temporal import TemporalLSTMClassifier
from pipelines_video.save_extractors import ResNet101FeatureExtractor, PoseFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SignSegment:
    """Representa un segmento de seña detectado"""
    start_frame: int
    end_frame: int
    class_id: int
    class_name: str
    confidence: float
    top_3_predictions: List[Tuple[str, float]]


class ImprovedKeypointExtractor:
    """
    Extractor de keypoints mejorado con normalización correcta
    """
    
    def __init__(self):
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
    
    def extract_and_normalize_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae y normaliza keypoints EXACTAMENTE como en training
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose
        pose_results = self.pose.process(frame_rgb)
        if not pose_results.pose_landmarks:
            return None
        
        pose_kpts = np.array([
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in pose_results.pose_landmarks.landmark
        ])  # (33, 4)
        
        # NORMALIZACIÓN CORPORAL (igual que en training)
        left_hip = pose_kpts[23][:3]
        right_hip = pose_kpts[24][:3]
        center = (left_hip + right_hip) / 2
        
        shoulder_dist = np.linalg.norm(pose_kpts[11][:3] - pose_kpts[12][:3])
        scale = max(shoulder_dist, 0.1)
        
        normalized_pose = pose_kpts.copy()
        normalized_pose[:, :3] = (pose_kpts[:, :3] - center) / scale
        normalized_pose[:, :3] = np.clip(normalized_pose[:, :3], -1, 1)
        
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
        
        # Combinar (33 pose + 42 hands = 75 keypoints)
        combined = np.concatenate([normalized_pose, hand_kpts], axis=0)
        return combined


class ImprovedSignSegmenter:
    """
    Segmentador mejorado con mejor detección de boundaries
    """
    
    def __init__(self, min_frames: int = 18, max_frames: int = 36):
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.keypoint_extractor = ImprovedKeypointExtractor()
    
    def calculate_activity_score(self, keypoints_sequence: np.ndarray) -> np.ndarray:
        """Calcula activity score con pesos ajustados"""
        T = len(keypoints_sequence)
        scores = np.zeros(T)
        
        for t in range(T):
            kpts = keypoints_sequence[t]
            pose_kpts = kpts[:33]
            left_hand = kpts[33:54]
            right_hand = kpts[54:75]
            
            score = 0.0
            
            # 1. Visibilidad (35%)
            left_vis = np.mean(left_hand[:, 3])
            right_vis = np.mean(right_hand[:, 3])
            hand_visibility = (left_vis + right_vis) / 2.0
            score += 0.35 * hand_visibility
            
            # 2. Manos levantadas (30%)
            hip_y = np.mean([pose_kpts[23, 1], pose_kpts[24, 1]])
            shoulder_y = np.mean([pose_kpts[11, 1], pose_kpts[12, 1]])
            
            left_hand_y = np.mean(left_hand[:, 1])
            right_hand_y = np.mean(right_hand[:, 1])
            
            hands_raised = 0.0
            # Más estricto: manos entre hombros y cabeza
            if left_vis > 0.4:
                if left_hand_y < shoulder_y and left_hand_y > shoulder_y - 0.3:
                    hands_raised += 1.0
            if right_vis > 0.4:
                if right_hand_y < shoulder_y and right_hand_y > shoulder_y - 0.3:
                    hands_raised += 1.0
            
            hands_raised /= 2.0
            score += 0.30 * hands_raised
            
            # 3. Movimiento (25%)
            if t > 0:
                prev_kpts = keypoints_sequence[t - 1]
                
                left_motion = np.linalg.norm(
                    np.mean(left_hand[:, :3], axis=0) - 
                    np.mean(prev_kpts[33:54, :3], axis=0)
                )
                right_motion = np.linalg.norm(
                    np.mean(right_hand[:, :3], axis=0) - 
                    np.mean(prev_kpts[54:75, :3], axis=0)
                )
                
                motion_score = np.clip((left_motion + right_motion) / 0.08, 0, 1)
                score += 0.25 * motion_score
            
            # 4. Proximidad a la cara (10%)
            nose = pose_kpts[0, :3]
            left_hand_pos = np.mean(left_hand[:5, :3], axis=0)
            right_hand_pos = np.mean(right_hand[:5, :3], axis=0)
            
            left_dist = np.linalg.norm(left_hand_pos - nose)
            right_dist = np.linalg.norm(right_hand_pos - nose)
            
            proximity = 1.0 - np.clip(np.mean([left_dist, right_dist]), 0, 0.6) / 0.6
            score += 0.10 * max(0, proximity)
            
            scores[t] = np.clip(score, 0, 1)
        
        return scores
    
    def find_sign_boundaries_improved(self, scores: np.ndarray, threshold: float = 0.4) -> List[Tuple[int, int]]:
        """
        Encuentra boundaries con algoritmo mejorado
        """
        # Suavizado más agresivo
        kernel_size = 7
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(scores, kernel, mode='same')
        
        # Detectar picos
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(smoothed, height=threshold, distance=self.min_frames)
        
        if len(peaks) == 0:
            # Fallback: usar método de threshold
            active = smoothed > threshold
            boundaries = []
            in_sign = False
            start = 0
            
            for t in range(len(active)):
                if active[t] and not in_sign:
                    start = t
                    in_sign = True
                elif not active[t] and in_sign:
                    if t - start >= self.min_frames:
                        boundaries.append((start, t))
                    in_sign = False
            
            if in_sign and len(scores) - start >= self.min_frames:
                boundaries.append((start, len(scores)))
            
            return boundaries
        
        # Usar picos para definir boundaries
        boundaries = []
        for peak in peaks:
            # Expandir desde el pico
            start = max(0, peak - self.max_frames // 2)
            end = min(len(scores), peak + self.max_frames // 2)
            
            # Ajustar start: buscar donde el score sube
            for i in range(peak, max(0, peak - self.max_frames), -1):
                if smoothed[i] < threshold * 0.5:
                    start = i
                    break
            
            # Ajustar end: buscar donde el score baja
            for i in range(peak, min(len(scores), peak + self.max_frames)):
                if smoothed[i] < threshold * 0.5:
                    end = i
                    break
            
            if end - start >= self.min_frames:
                boundaries.append((start, end))
        
        # Merge overlapping boundaries
        if len(boundaries) > 1:
            merged = [boundaries[0]]
            for start, end in boundaries[1:]:
                last_start, last_end = merged[-1]
                if start <= last_end:
                    # Merge
                    merged[-1] = (last_start, max(end, last_end))
                else:
                    merged.append((start, end))
            boundaries = merged
        
        return boundaries
    
    def segment_video(self, video_path: Path) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Segmenta video mejorado"""
        logger.info(f"Segmentando video: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"  FPS: {fps}, Total frames: {total_frames}")
        
        all_keypoints = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            kpts = self.keypoint_extractor.extract_and_normalize_keypoints(frame)
            if kpts is not None:
                all_keypoints.append(kpts)
            else:
                all_keypoints.append(np.zeros((75, 4)))
            
            frame_idx += 1
            if frame_idx % 30 == 0:
                logger.info(f"  Procesados {frame_idx}/{total_frames} frames")
        
        cap.release()
        
        keypoints_array = np.array(all_keypoints)
        logger.info(f"  Keypoints extraídos: {keypoints_array.shape}")
        
        # Calcular scores
        scores = self.calculate_activity_score(keypoints_array)
        logger.info(f"  Activity scores - min: {scores.min():.3f}, max: {scores.max():.3f}, mean: {scores.mean():.3f}")
        
        # Encontrar boundaries mejorados
        boundaries = self.find_sign_boundaries_improved(scores, threshold=0.35)
        logger.info(f"  Señas detectadas: {len(boundaries)}")
        
        for i, (start, end) in enumerate(boundaries):
            duration = (end - start) / fps
            avg_score = scores[start:end].mean()
            logger.info(f"    Seña {i+1}: frames [{start:4d}, {end:4d}] ({duration:.2f}s) - score: {avg_score:.3f}")
        
        return boundaries, keypoints_array


class SignLanguageInference:
    """Sistema completo CORREGIDO"""
    
    def __init__(
        self,
        model_path: Path,
        visual_extractor_path: Path,
        pose_extractor_path: Path,
        class_mapping_path: Path,
        device: str = "cuda",
        confidence_threshold: float = 0.15  # NUEVO: threshold de confianza
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        logger.info(f"Usando device: {self.device}")
        logger.info(f"Confidence threshold: {confidence_threshold:.2%}")
        
        # Cargar mapping
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
        self.visual_extractor = torch.load(visual_extractor_path, map_location=self.device , weights_only=False)
        self.visual_extractor.eval()
        
        self.pose_extractor = torch.load(pose_extractor_path, map_location=self.device, weights_only=False)
        self.pose_extractor.eval()
        
        # CRÍTICO: Detectar dimensión real del visual extractor
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            dummy_output = self.visual_extractor(dummy_input)
            visual_feature_dim = dummy_output.shape[1]
        
        # Detectar dimensión del pose extractor
        with torch.no_grad():
            dummy_keypoints = torch.randn(1, 300).to(self.device)
            dummy_pose_output = self.pose_extractor(dummy_keypoints)
            pose_feature_dim = dummy_pose_output.shape[1]
        
        fused_feature_dim = visual_feature_dim + pose_feature_dim
        
        logger.info(f"Dimensiones detectadas:")
        logger.info(f"  Visual features: {visual_feature_dim}")
        logger.info(f"  Pose features: {pose_feature_dim}")
        logger.info(f"  Fused features: {fused_feature_dim}")
        
        # Cargar modelo con dimensión correcta
        logger.info("Cargando modelo temporal...")
        self.model = TemporalLSTMClassifier(
            input_dim=fused_feature_dim,  # USAR DIMENSIÓN DETECTADA
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
        
        # Guardar dimensiones para uso posterior
        self.visual_feature_dim = visual_feature_dim
        self.pose_feature_dim = pose_feature_dim
        
        # Segmentador
        self.segmenter = ImprovedSignSegmenter()
    
    def extract_features(self, frames: np.ndarray, keypoints: np.ndarray) -> torch.Tensor:
        """Extrae features con normalización CORRECTA"""
        T = len(frames)
        
        # Visual features
        batch_tensors = []
        for frame in frames:
            if frame.shape[:2] != (config.data.frame_height, config.data.frame_width):
                frame = cv2.resize(frame, (config.data.frame_width, config.data.frame_height))
            
            # Normalización ImageNet (IGUAL QUE EN TRAINING)
            frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            frame_tensor = (frame_tensor - mean) / std
            batch_tensors.append(frame_tensor)
        
        batch = torch.stack(batch_tensors).to(self.device)
        
        with torch.no_grad():
            visual_features = self.visual_extractor(batch).cpu().numpy()
        
        # Pose features (keypoints ya están normalizados)
        keypoints_flat = keypoints.reshape(T, -1).astype(np.float32)  # (T, 300)
        keypoints_tensor = torch.from_numpy(keypoints_flat).to(self.device)
        
        with torch.no_grad():
            pose_features = self.pose_extractor(keypoints_tensor).cpu().numpy()
        
        # Fusionar
        fused = np.concatenate([visual_features, pose_features], axis=1)
        
        # Log estadísticas para debug
        logger.debug(f"Visual features - mean: {visual_features.mean():.4f}, std: {visual_features.std():.4f}")
        logger.debug(f"Pose features - mean: {pose_features.mean():.4f}, std: {pose_features.std():.4f}")
        logger.debug(f"Fused features - mean: {fused.mean():.4f}, std: {fused.std():.4f}")
        
        return torch.from_numpy(fused).float()
    
    def predict(self, features: torch.Tensor, top_k: int = 5) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Predice con verificación de confianza"""
        features = features.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(features)
            probs = torch.softmax(logits, dim=1)[0]
        
        top_k_probs, top_k_indices = torch.topk(probs, k=top_k)
        
        predictions = []
        for prob, idx in zip(top_k_probs, top_k_indices):
            class_id = idx.item()
            class_name = self.id_to_class.get(class_id, f"Unknown_{class_id}")
            predictions.append((class_name, prob.item()))
        
        top_class, top_conf = predictions[0]
        
        # Verificar confianza
        if top_conf < self.confidence_threshold:
            logger.warning(f"Baja confianza detectada: {top_conf:.2%} (threshold: {self.confidence_threshold:.2%})")
        
        return top_class, top_conf, predictions
    
    def process_video(
        self,
        video_path: Path,
        output_dir: Path = None,
        save_visualization: bool = False
    ) -> List[SignSegment]:
        """Procesa video completo"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Procesando video: {video_path.name}")
        logger.info(f"{'='*60}")
        
        # Segmentar
        boundaries, all_keypoints = self.segmenter.segment_video(video_path)
        
        if len(boundaries) == 0:
            logger.warning("No se detectaron señas")
            return []
        
        # Cargar frames
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (config.data.frame_width, config.data.frame_height))
            all_frames.append(frame)
        
        cap.release()
        all_frames = np.array(all_frames)
        
        # Procesar cada seña
        segments = []
        valid_predictions = 0
        
        for i, (start, end) in enumerate(boundaries):
            logger.info(f"\nProcesando seña {i+1}/{len(boundaries)}")
            
            sign_frames = all_frames[start:end]
            sign_keypoints = all_keypoints[start:end]
            
            # Normalizar longitud
            target_length = config.data.num_frames_per_clip
            current_length = len(sign_frames)
            
            if current_length >= target_length:
                mid = current_length // 2
                half = target_length // 2
                sign_frames = sign_frames[mid-half:mid-half+target_length]
                sign_keypoints = sign_keypoints[mid-half:mid-half+target_length]
            else:
                num_pad = target_length - current_length
                last_frame = sign_frames[-1]
                last_keypoint = sign_keypoints[-1]
                
                pad_frames = np.repeat(last_frame[np.newaxis, :, :, :], num_pad, axis=0)
                pad_keypoints = np.repeat(last_keypoint[np.newaxis, :, :], num_pad, axis=0)
                
                sign_frames = np.concatenate([sign_frames, pad_frames], axis=0)
                sign_keypoints = np.concatenate([sign_keypoints, pad_keypoints], axis=0)
            
            # Extraer features
            features = self.extract_features(sign_frames, sign_keypoints)
            
            # Predecir
            sign, confidence, top_3 = self.predict(features, top_k=3)
            
            logger.info(f"  Predicción: {sign} ({confidence:.2%})")
            logger.info(f"  Top-3: {[(name, f'{conf:.2%}') for name, conf in top_3]}")
            
            # Validar confianza
            if confidence >= self.confidence_threshold:
                valid_predictions += 1
                status = "✓"
            else:
                status = "⚠"
            
            logger.info(f"  {status} Confianza: {confidence:.2%}")
            
            segment = SignSegment(
                start_frame=start,
                end_frame=end,
                class_id=-1,
                class_name=sign,
                confidence=confidence,
                top_3_predictions=top_3
            )
            
            segments.append(segment)
        
        # Mostrar traducción
        logger.info(f"\n{'='*60}")
        logger.info("TRADUCCIÓN COMPLETA:")
        logger.info(f"{'='*60}")
        
        # Filtrar predicciones con baja confianza
        valid_segments = [s for s in segments if s.confidence >= self.confidence_threshold]
        
        if len(valid_segments) > 0:
            sentence = " ".join([seg.class_name for seg in valid_segments])
            logger.info(f"\n{sentence}\n")
            
            confidences = [seg.confidence for seg in valid_segments]
            avg_confidence = np.mean(confidences)
            logger.info(f"Confianza promedio: {avg_confidence:.2%}")
            logger.info(f"Señas válidas: {len(valid_segments)}/{len(segments)}")
        else:
            logger.warning("No hay predicciones con confianza suficiente")
        
        # Guardar resultados
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results = {
                'video': video_path.name,
                'translation': " ".join([s.class_name for s in valid_segments]),
                'num_signs_detected': len(segments),
                'num_valid_predictions': len(valid_segments),
                'average_confidence': float(np.mean([s.confidence for s in valid_segments])) if valid_segments else 0.0,
                'confidence_threshold': self.confidence_threshold,
                'signs': [
                    {
                        'index': i,
                        'start_frame': seg.start_frame,
                        'end_frame': seg.end_frame,
                        'start_time': seg.start_frame / fps,
                        'end_time': seg.end_frame / fps,
                        'sign': seg.class_name,
                        'confidence': float(seg.confidence),
                        'is_valid': seg.confidence >= self.confidence_threshold,
                        'top_3': [(name, float(conf)) for name, conf in seg.top_3_predictions]
                    }
                    for i, seg in enumerate(segments)
                ]
            }
            
            output_file = output_dir / f"{video_path.stem}_translation.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"\nResultados guardados: {output_file}")
        
        return segments


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Inferencia mejorada para videos")
    parser.add_argument("--video_path", type=Path)
    parser.add_argument("--model_path", type=Path,
                       default=config.model_paths.temporal_checkpoints / "best_model.pt")
    parser.add_argument("--visual_extractor", type=Path,
                       default=Path("models/extractors/visual_extractor_full.pt"))
    parser.add_argument("--pose_extractor", type=Path,
                       default=Path("models/extractors/pose_extractor_full.pt"))
    parser.add_argument("--class_mapping", type=Path,
                       default=config.data_paths.dataset_meta)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--confidence_threshold", type=float, default=0.15,
                       help="Threshold mínimo de confianza para aceptar predicción")
    
    args = parser.parse_args()
    
    if not args.video_path.exists():
        logger.error(f"Video no encontrado: {args.video_path}")
        return
    
    # Crear sistema
    inference = SignLanguageInference(
        model_path=args.model_path,
        visual_extractor_path=args.visual_extractor,
        pose_extractor_path=args.pose_extractor,
        class_mapping_path=args.class_mapping,
        device=args.device,
        confidence_threshold=args.confidence_threshold
    )
    
    # Procesar
    segments = inference.process_video(
        video_path=args.video_path,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()