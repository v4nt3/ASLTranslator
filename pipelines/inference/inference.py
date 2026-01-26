"""
Inference CORREGIDO - RESUELVE TODOS LOS PROBLEMAS DETECTADOS

CAMBIOS CRÍTICOS:
1. Extracción de frames usando np.linspace() (igual que entrenamiento)
2. Detección mejorada de boundaries entre señas
3. Acumulación inteligente que espera pausas
4. Normalización consistente
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
from typing import Optional, Tuple, List, Dict
import argparse
from torchvision import transforms
import sys

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from pipelines_video.save_extractors import ResNet101FeatureExtractor, PoseFeatureExtractor


class KeypointNormalizer:
    """Normalización EXACTA de data_preparation.py"""
    
    def normalize_pose_keypoints(self, pose_kpts: np.ndarray) -> np.ndarray:
        left_hip = pose_kpts[23][:3]
        right_hip = pose_kpts[24][:3]
        center = (left_hip + right_hip) / 2
        
        left_shoulder = pose_kpts[11][:3]
        right_shoulder = pose_kpts[12][:3]
        shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)
        scale = max(shoulder_dist, 0.1)
        
        normalized = pose_kpts.copy()
        normalized[:, :3] = (pose_kpts[:, :3] - center) / scale
        normalized[:, :3] = np.clip(normalized[:, :3], -1, 1)
        
        return normalized
    
    def process_frame(self, pose_kpts: np.ndarray, hand_kpts: np.ndarray) -> np.ndarray:
        pose_normalized = self.normalize_pose_keypoints(pose_kpts)
        combined = np.concatenate([pose_normalized, hand_kpts], axis=0)
        return combined


class SignBoundaryDetector:
    """
    Detecta boundaries entre señas usando motion energy
    Esto es CRÍTICO para videos con múltiples señas
    """
    
    def __init__(self, min_pause_frames: int = 8, motion_threshold: float = 0.02):
        self.min_pause_frames = min_pause_frames
        self.motion_threshold = motion_threshold
    
    def calculate_motion_energy(self, keypoints_buffer: List[np.ndarray]) -> np.ndarray:
        """
        Calcula energía de movimiento frame a frame
        
        Returns:
            motion: array (T-1,) con energía de movimiento
        """
        if len(keypoints_buffer) < 2:
            return np.array([0.0])
        
        motion = []
        for i in range(1, len(keypoints_buffer)):
            curr = keypoints_buffer[i]
            prev = keypoints_buffer[i-1]
            
            # Motion en manos (más importante)
            hands_curr = curr[33:75, :3]  # 42 keypoints de manos
            hands_prev = prev[33:75, :3]
            
            hand_motion = np.mean(np.linalg.norm(hands_curr - hands_prev, axis=1))
            
            # Motion en pose (menos peso)
            pose_curr = curr[:33, :3]
            pose_prev = prev[:33, :3]
            pose_motion = np.mean(np.linalg.norm(pose_curr - pose_prev, axis=1))
            
            # Combinar: 70% manos, 30% pose
            total_motion = 0.7 * hand_motion + 0.3 * pose_motion
            motion.append(total_motion)
        
        return np.array(motion)
    
    def detect_pause(self, motion: np.ndarray, window_size: int = 5) -> bool:
        """
        Detecta si hay una pausa (movimiento bajo sostenido)
        
        Args:
            motion: Energía de movimiento
            window_size: Ventana para promediar
        
        Returns:
            True si hay pausa
        """
        if len(motion) < window_size:
            return False
        
        recent_motion = motion[-window_size:]
        avg_motion = np.mean(recent_motion)
        
        return avg_motion < self.motion_threshold
    
    def find_sign_boundaries(self, keypoints_buffer: List[np.ndarray]) -> List[Tuple[int, int]]:
        """
        Encuentra segmentos de señas individuales en el buffer
        
        Returns:
            Lista de (start, end) para cada seña detectada
        """
        motion = self.calculate_motion_energy(keypoints_buffer)
        
        if len(motion) == 0:
            return [(0, len(keypoints_buffer))]
        
        # Suavizar motion
        from scipy.ndimage import gaussian_filter1d
        motion_smooth = gaussian_filter1d(motion, sigma=2)
        
        # Encontrar valleys (pausas)
        boundaries = []
        in_pause = False
        pause_start = 0
        
        for i, m in enumerate(motion_smooth):
            if m < self.motion_threshold:
                if not in_pause:
                    pause_start = i
                    in_pause = True
            else:
                if in_pause and (i - pause_start) >= self.min_pause_frames:
                    boundaries.append(pause_start + (i - pause_start) // 2)
                in_pause = False
        
        # Convertir boundaries a segmentos
        if len(boundaries) == 0:
            return [(0, len(keypoints_buffer))]
        
        segments = []
        prev = 0
        for b in boundaries:
            if b - prev >= 12:  # Mínimo 12 frames por seña
                segments.append((prev, b))
            prev = b
        
        # Último segmento
        if len(keypoints_buffer) - prev >= 12:
            segments.append((prev, len(keypoints_buffer)))
        
        return segments if segments else [(0, len(keypoints_buffer))]


class ASLInferenceFixed:
    """Sistema de inferencia CORREGIDO"""
    
    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        visual_extractor_path: Path,
        pose_extractor_path: Path,
        device: str = "cuda",
        frames_per_sign: int = 24,  # MISMO que entrenamiento
        confidence_threshold: float = 0.15,  # Subir umbral
        max_buffer_seconds: float = 3.0  # Máximo 3 segundos de buffer
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.frames_per_sign = frames_per_sign
        self.confidence_threshold = confidence_threshold
        self.max_buffer_seconds = max_buffer_seconds
        
        logger.info("="*60)
        logger.info("ASL Inference FIXED - Resuelve problemas de extracción")
        logger.info("="*60)
        logger.info(f"Device: {self.device}")
        logger.info(f"Frames per sign: {frames_per_sign}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        
        # Cargar modelos
        self.class_to_gloss = self._load_class_mapping(metadata_path)
        self.num_classes = len(self.class_to_gloss)
        logger.info(f"Classes: {self.num_classes}")
        
        self._load_models(model_path, visual_extractor_path, pose_extractor_path)
        self._init_mediapipe()
        
        self.normalizer = KeypointNormalizer()
        self.boundary_detector = SignBoundaryDetector(
            min_pause_frames=8,
            motion_threshold=0.02
        )
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Buffers
        self.frame_buffer = []
        self.keypoints_buffer = []
        self.fps = None
        
        logger.info(" Sistema inicializado\n")
    
    def _load_class_mapping(self, metadata_path: Path) -> dict:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        class_to_gloss = {}
        
        if isinstance(metadata, dict):
            if 'videos' in metadata:
                for entry in metadata['videos']:
                    cid = entry.get('class_id')
                    gloss = entry.get('class_name') or entry.get('gloss', 'UNKNOWN')
                    if cid is not None:
                        class_to_gloss[int(cid)] = gloss
            else:
                for k, v in metadata.items():
                    if isinstance(v, str):
                        try:
                            class_to_gloss[int(k)] = v
                        except:
                            pass
                    elif isinstance(v, dict):
                        cid = v.get('class_id')
                        gloss = v.get('class_name') or v.get('gloss', 'UNKNOWN')
                        if cid is not None and cid not in class_to_gloss:
                            class_to_gloss[int(cid)] = gloss
        
        return class_to_gloss
    
    def _load_models(self, model_path, visual_path, pose_path):
        self.visual_extractor = torch.load(visual_path, map_location=self.device, weights_only=False)
        self.visual_extractor.eval()
        
        self.pose_extractor = torch.load(pose_path, map_location=self.device, weights_only=False)
        self.pose_extractor.eval()
        
        from pipelines.models_temporal import TemporalLSTMClassifier
        
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
        
        logger.info(" Modelos cargados")
    
    def _init_mediapipe(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        logger.info(" MediaPipe inicializado")
    
    def extract_keypoints(self, frame: np.ndarray) -> np.ndarray:
        """Extrae keypoints (75, 4)"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pose
        pose_results = self.pose.process(frame_rgb)
        pose_kpts = np.zeros((33, 4), dtype=np.float32)
        
        if pose_results.pose_landmarks:
            for i, lm in enumerate(pose_results.pose_landmarks.landmark):
                pose_kpts[i] = [lm.x, lm.y, lm.z, lm.visibility]
        
        # Hands
        hands_results = self.hands.process(frame_rgb)
        hand_kpts = np.zeros((42, 4), dtype=np.float32)
        
        if hands_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks[:2]):
                start_idx = hand_idx * 21
                for i, lm in enumerate(hand_landmarks.landmark):
                    hand_kpts[start_idx + i] = [lm.x, lm.y, lm.z, 1.0]
        
        combined = self.normalizer.process_frame(pose_kpts, hand_kpts)
        return combined
    
    def subsample_frames_uniform(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        CRÍTICO: Submuestrea frames usando np.linspace()
        Esto replica EXACTAMENTE FrameExtractor de entrenamiento
        """
        total_frames = len(frames)
        
        if total_frames <= self.frames_per_sign:
            return frames
        
        # np.linspace da índices uniformemente espaciados (IGUAL que entrenamiento)
        indices = np.linspace(0, total_frames - 1, self.frames_per_sign, dtype=int)
        
        return [frames[i] for i in indices]
    
    @torch.no_grad()
    def predict_sign(self, frames: List[np.ndarray], keypoints: List[np.ndarray]) -> Tuple[str, float, List]:
        """
        Predice UNA seña a partir de frames y keypoints
        
        CRÍTICO: Ahora submuestrea frames uniformemente
        """
        # 1. Submuestrear a frames_per_sign usando np.linspace()
        if len(frames) > self.frames_per_sign:
            frames = self.subsample_frames_uniform(frames)
            keypoints = self.subsample_frames_uniform(keypoints)
        
        # 2. Padding si es necesario
        while len(frames) < self.frames_per_sign:
            frames.append(frames[-1].copy())
            keypoints.append(keypoints[-1].copy())
        
        # 3. Truncar a frames_per_sign exacto
        frames = frames[:self.frames_per_sign]
        keypoints = keypoints[:self.frames_per_sign]
        
        # 4. Extraer features visuales
        frames_tensor = torch.stack([
            self.transform(f) for f in frames
        ]).to(self.device)
        
        visual_features = self.visual_extractor(frames_tensor)
        
        # 5. Extraer features de pose
        keypoints_flat = np.stack([kp.flatten() for kp in keypoints])
        keypoints_tensor = torch.from_numpy(keypoints_flat).float().to(self.device)
        
        pose_features = self.pose_extractor(keypoints_tensor)
        
        # 6. Fusionar y predecir
        fused = torch.cat([visual_features, pose_features], dim=1)
        fused = fused.unsqueeze(0)
        
        logits = self.model(fused)
        probs = torch.softmax(logits, dim=1)[0]
        
        # Top-5
        top5_probs, top5_idx = torch.topk(probs, k=min(5, self.num_classes))
        top5 = [(self.class_to_gloss.get(int(i), f"CLASS_{i}"), float(p))
                for p, i in zip(top5_probs.cpu(), top5_idx.cpu())]
        
        best_idx = int(top5_idx[0])
        best_prob = float(top5_probs[0])
        best_gloss = self.class_to_gloss.get(best_idx, f"CLASS_{best_idx}")
        
        return best_gloss, best_prob, top5
    
    def process_video_with_boundaries(
        self,
        video_path: Path,
        output_path: Optional[Path] = None
    ) -> List[Dict]:
        """
        Procesa video detectando boundaries entre señas
        
        ESTO RESUELVE EL PROBLEMA DE SEÑAS CONTINUAS
        """
        logger.info(f"Procesando video con detección de boundaries: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"  FPS: {self.fps:.1f}, Size: {width}x{height}")
        
        # Leer y procesar TODOS los frames
        all_frames = []
        all_keypoints = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_resized = cv2.resize(frame, (224, 224))
            keypoints = self.extract_keypoints(frame)
            
            all_frames.append(frame_resized)
            all_keypoints.append(keypoints)
            
            frame_idx += 1
        
        cap.release()
        
        logger.info(f"  Total frames leídos: {len(all_frames)}")
        
        # Detectar boundaries entre señas
        logger.info("  Detectando boundaries entre señas...")
        segments = self.boundary_detector.find_sign_boundaries(all_keypoints)
        
        logger.info(f"  Señas detectadas: {len(segments)}")
        
        # Predecir cada segmento
        predictions = []
        
        for seg_idx, (start, end) in enumerate(segments):
            seg_frames = all_frames[start:end]
            seg_keypoints = all_keypoints[start:end]
            
            duration = (end - start) / self.fps
            
            logger.info(f"\n  Segmento {seg_idx+1}: frames [{start}, {end}], duración {duration:.2f}s")
            
            # Skip segmentos muy cortos
            if len(seg_frames) < 8:
                logger.info(f"    ⊘ Segmento muy corto, saltando")
                continue
            
            # Predecir
            gloss, conf, top5 = self.predict_sign(seg_frames, seg_keypoints)
            
            logger.info(f"    → {gloss} ({conf:.1%})")
            logger.info(f"    Top-3: {[(g, f'{p:.1%}') for g, p in top5[:3]]}")
            
            if conf >= self.confidence_threshold:
                predictions.append({
                    'start_frame': start,
                    'end_frame': end,
                    'start_time': start / self.fps,
                    'end_time': end / self.fps,
                    'gloss': gloss,
                    'confidence': conf,
                    'top5': top5
                })
        
        # Log resultado final
        logger.info(f"\n{'='*60}")
        logger.info("TRADUCCIÓN DEL VIDEO:")
        logger.info(f"{'='*60}")
        
        sentence = []
        for p in predictions:
            sentence.append(p['gloss'])
            logger.info(f"  [{p['start_time']:.2f}s - {p['end_time']:.2f}s] "
                       f"{p['gloss']} ({p['confidence']:.1%})")
        
        logger.info(f"\nOración: {' '.join(sentence)}")
        logger.info(f"{'='*60}\n")
        
        # Guardar video anotado
        if output_path:
            self._save_annotated_video(
                all_frames, predictions, output_path, 
                self.fps, width, height
            )
        
        return predictions
    
    def _save_annotated_video(self, frames, predictions, output_path, fps, width, height):
        """Guarda video con anotaciones"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Resize frames back to original size
        frames_resized = [cv2.resize(f, (width, height)) for f in frames]
        
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height + 150))
        
        logger.info(f"Generando video anotado: {output_path}")
        
        for i, frame in enumerate(frames_resized):
            # Panel inferior
            panel = np.zeros((150, width, 3), dtype=np.uint8)
            panel[:] = (30, 30, 30)
            
            # Buscar predicción activa
            current_pred = None
            for p in predictions:
                if p['start_frame'] <= i < p['end_frame']:
                    current_pred = p
                    break
            
            if current_pred:
                color = (0, 255, 0) if current_pred['confidence'] > 0.2 else (0, 165, 255)
                text = f"{current_pred['gloss']} ({current_pred['confidence']:.1%})"
                cv2.putText(panel, text, (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            else:
                cv2.putText(panel, "...", (20, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            
            # Oración hasta ahora
            sentence = ' '.join([p['gloss'] for p in predictions if p['end_frame'] <= i])
            if sentence:
                cv2.putText(panel, sentence, (20, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Combinar
            result = np.vstack([frame, panel])
            out.write(result)
        
        out.release()
        logger.info(f" Video guardado: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='ASL Inference FIXED')
    
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--metadata_path', type=str, required=True)
    parser.add_argument('--visual_extractor', type=str, required=True)
    parser.add_argument('--pose_extractor', type=str, required=True)
    parser.add_argument('--output_video', type=str, help='Ruta para video anotado')
    parser.add_argument('--confidence_threshold', type=float, default=0.15)
    parser.add_argument('--frames_per_sign', type=int, default=24)
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        logger.error(f"Video no encontrado: {video_path}")
        return
    
    # Crear sistema de inferencia
    inference = ASLInferenceFixed(
        model_path=Path(args.model_path),
        metadata_path=Path(args.metadata_path),
        visual_extractor_path=Path(args.visual_extractor),
        pose_extractor_path=Path(args.pose_extractor),
        device=args.device,
        frames_per_sign=args.frames_per_sign,
        confidence_threshold=args.confidence_threshold
    )
    
    # Procesar video
    output_path = Path(args.output_video) if args.output_video else None
    
    inference.process_video_with_boundaries(
        video_path=video_path,
        output_path=output_path
    )


if __name__ == "__main__":
    main()