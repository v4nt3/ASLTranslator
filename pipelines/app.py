"""
Script de inferencia para traducir videos completos de señas ASL
Soporta:
1. Análisis de video completo (una seña)
2. Ventana deslizante (múltiples señas en secuencia)
3. Generación de video anotado con subtítulos
4. Exportación de transcripción a texto
"""

import torch #type: ignore
import numpy as np #type: ignore
import cv2 #type: ignore
import mediapipe as mp #type: ignore
from pathlib import Path
import json
import logging
from typing import Optional, Tuple, List, Dict
import argparse
from torchvision import transforms #type: ignore
from tqdm import tqdm #type: ignore

from pipelines.models_temporal import TemporalLSTMClassifier
from pipelines_video.save_extractors import ResNet101FeatureExtractor
from pipelines_video.save_extractors import PoseFeatureExtractor  

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ASLVideoTranslator:
    """Traductor de videos ASL completo"""
    
    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        visual_extractor_path: Path,
        pose_extractor_path: Path,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Cargar metadata
        self.class_to_gloss = self._load_class_mapping(metadata_path)
        self.num_classes = len(self.class_to_gloss)
        logger.info(f"✓ Cargadas {self.num_classes} clases")
        
        # Cargar extractores
        logger.info("Cargando extractores...")
        self.visual_extractor = torch.load(visual_extractor_path, map_location=self.device, weights_only=False)
        self.visual_extractor.eval()
        logger.info(f"✓ Visual extractor cargado")
        
        self.pose_extractor = torch.load(pose_extractor_path, map_location=self.device, weights_only=False)
        self.pose_extractor.eval()
        logger.info(f"✓ Pose extractor cargado")
        
        # Transform
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
        
        logger.info("✓ Traductor ASL inicializado")
    
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
    
    def extract_keypoints(self, frame: np.ndarray) -> np.ndarray:
        """Extrae keypoints de un frame"""
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
    def extract_features(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Extrae features fusionadas de una secuencia de frames
        
        Args:
            frames: Lista de frames (H, W, 3)
        
        Returns:
            features: (T, 1152) tensor de features fusionadas
        """
        # Procesar frames
        frames_resized = [cv2.resize(f, (224, 224)) for f in frames]
        frames_tensor = torch.stack([
            self.transform(f) for f in frames_resized
        ]).to(self.device)
        
        # Features visuales
        visual_features = self.visual_extractor(frames_tensor)  # (T, 1024)
        
        # Keypoints y features de pose
        keypoints_list = [self.extract_keypoints(f) for f in frames]
        keypoints_tensor = torch.from_numpy(
            np.stack(keypoints_list)
        ).float().to(self.device)
        
        pose_features = self.pose_extractor(keypoints_tensor)  # (T, 128)
        
        # Fusionar
        fused_features = torch.cat([visual_features, pose_features], dim=1)  # (T, 1152)
        
        return fused_features
    
    @torch.no_grad()
    def predict_segment(
        self, 
        features: torch.Tensor
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predice la seña de un segmento de features
        
        Args:
            features: (T, 1152) tensor
        
        Returns:
            gloss, confidence, top5
        """
        if features.dim() == 2:
            features = features.unsqueeze(0)  # (1, T, 1152)
        
        logits = self.model(features)
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
    
    def translate_video_full(
        self,
        video_path: Path,
        max_frames: int = None,
        stride: int = 1
    ) -> Dict:
        """
        Traduce video completo como una sola seña
        
        Args:
            video_path: Ruta al video
            max_frames: Máximo de frames a procesar
            stride: Tomar 1 de cada N frames
        
        Returns:
            dict con predicción y metadata
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"TRADUCCIÓN DE VIDEO COMPLETO")
        logger.info(f"{'='*80}")
        logger.info(f"Video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"FPS: {fps:.2f}")
        logger.info(f"Total frames: {total_frames}")
        
        frames = []
        frame_count = 0
        
        with tqdm(desc="Cargando frames", total=total_frames) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % stride == 0:
                    if max_frames and len(frames) >= max_frames:
                        break
                    frames.append(frame)
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        
        logger.info(f"Frames procesados: {len(frames)}")
        
        # Extraer features
        logger.info("Extrayendo features...")
        features = self.extract_features(frames)
        
        # Predecir
        logger.info("Realizando predicción...")
        gloss, confidence, top5 = self.predict_segment(features)
        
        result = {
            'video_path': str(video_path),
            'total_frames': len(frames),
            'fps': fps,
            'duration_seconds': len(frames) / fps,
            'prediction': gloss,
            'confidence': confidence,
            'top5': top5
        }
        
        logger.info(f"\n{'='*80}")
        logger.info(f"RESULTADO:")
        logger.info(f"{'='*80}")
        logger.info(f"Seña detectada: {gloss}")
        logger.info(f"Confianza: {confidence:.2%}")
        logger.info(f"\nTop-5 predicciones:")
        for i, (g, prob) in enumerate(top5, 1):
            logger.info(f"  {i}. {g}: {prob:.2%}")
        logger.info(f"{'='*80}\n")
        
        return result
    
    def translate_video_sliding_window(
        self,
        video_path: Path,
        window_size: int = 24,
        stride: int = 12,
        confidence_threshold: float = 0.3,
        max_frames: int = None
    ) -> List[Dict]:
        """
        Traduce video usando ventana deslizante para detectar múltiples señas
        
        Args:
            video_path: Ruta al video
            window_size: Tamaño de ventana (frames)
            stride: Desplazamiento de ventana (frames)
            confidence_threshold: Umbral mínimo de confianza
            max_frames: Máximo de frames a procesar
        
        Returns:
            Lista de detecciones con timestamps
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"TRADUCCIÓN CON VENTANA DESLIZANTE")
        logger.info(f"{'='*80}")
        logger.info(f"Video: {video_path}")
        logger.info(f"Window size: {window_size} frames")
        logger.info(f"Stride: {stride} frames")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Cargar todos los frames
        frames = []
        with tqdm(desc="Cargando frames", total=total_frames) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and len(frames) >= max_frames:
                    break
                
                frames.append(frame)
                pbar.update(1)
        
        cap.release()
        
        logger.info(f"Frames cargados: {len(frames)}")
        
        # Procesar con ventana deslizante
        detections = []
        num_windows = (len(frames) - window_size) // stride + 1
        
        logger.info(f"Procesando {num_windows} ventanas...")
        
        for i in tqdm(range(0, len(frames) - window_size + 1, stride), desc="Analizando ventanas"):
            window_frames = frames[i:i+window_size]
            
            # Extraer features
            features = self.extract_features(window_frames)
            
            # Predecir
            gloss, confidence, top5 = self.predict_segment(features)
            
            # Solo guardar si supera el umbral
            if confidence >= confidence_threshold:
                start_time = i / fps
                end_time = (i + window_size) / fps
                
                detection = {
                    'start_frame': i,
                    'end_frame': i + window_size,
                    'start_time': start_time,
                    'end_time': end_time,
                    'gloss': gloss,
                    'confidence': confidence,
                    'top5': top5
                }
                
                detections.append(detection)
        
        # Post-procesamiento: fusionar detecciones consecutivas de la misma seña
        merged_detections = self._merge_consecutive_detections(detections)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"DETECCIONES: {len(merged_detections)} señas encontradas")
        logger.info(f"{'='*80}")
        
        for i, det in enumerate(merged_detections, 1):
            logger.info(f"{i}. [{det['start_time']:.2f}s - {det['end_time']:.2f}s] "
                       f"{det['gloss']} ({det['confidence']:.2%})")
        
        logger.info(f"{'='*80}\n")
        
        return merged_detections
    
    def _merge_consecutive_detections(
        self, 
        detections: List[Dict],
        time_threshold: float = 1.0
    ) -> List[Dict]:
        """
        Fusiona detecciones consecutivas de la misma seña
        
        Args:
            detections: Lista de detecciones
            time_threshold: Máxima separación temporal para fusionar (segundos)
        
        Returns:
            Lista de detecciones fusionadas
        """
        if not detections:
            return []
        
        merged = []
        current = detections[0].copy()
        
        for det in detections[1:]:
            # Si es la misma seña y está cerca temporalmente
            if (det['gloss'] == current['gloss'] and 
                det['start_time'] - current['end_time'] < time_threshold):
                # Fusionar: extender tiempo y promediar confianza
                current['end_time'] = det['end_time']
                current['end_frame'] = det['end_frame']
                current['confidence'] = (current['confidence'] + det['confidence']) / 2
            else:
                # Nueva seña diferente
                merged.append(current)
                current = det.copy()
        
        merged.append(current)
        
        return merged
    
    def create_annotated_video(
        self,
        video_path: Path,
        detections: List[Dict],
        output_path: Path,
        mode: str = 'sliding'
    ):
        """
        Crea video anotado con subtítulos de las señas detectadas
        
        Args:
            video_path: Video original
            detections: Lista de detecciones (o dict para modo 'full')
            output_path: Ruta de salida
            mode: 'sliding' o 'full'
        """
        logger.info(f"\nCreando video anotado: {output_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height + 150))
        
        frame_idx = 0
        
        with tqdm(desc="Generando video", total=total_frames) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Panel inferior
                panel = np.zeros((150, width, 3), dtype=np.uint8)
                panel[:] = (30, 30, 30)
                
                if mode == 'full':
                    # Modo full: mostrar predicción única
                    gloss = detections['prediction']
                    confidence = detections['confidence']
                    
                    color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
                    cv2.putText(panel, f"Seña: {gloss}", (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
                    cv2.putText(panel, f"Confianza: {confidence:.1%}", (20, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
                
                else:
                    # Modo sliding: mostrar seña actual
                    current_time = frame_idx / fps
                    current_gloss = None
                    current_conf = 0
                    
                    for det in detections:
                        if det['start_time'] <= current_time <= det['end_time']:
                            current_gloss = det['gloss']
                            current_conf = det['confidence']
                            break
                    
                    if current_gloss:
                        color = (0, 255, 0) if current_conf > 0.5 else (0, 165, 255)
                        cv2.putText(panel, f"Seña: {current_gloss}", (20, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
                        cv2.putText(panel, f"Confianza: {current_conf:.1%}", (20, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
                    else:
                        cv2.putText(panel, "Sin detección", (20, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2, cv2.LINE_AA)
                    
                    # Timeline
                    cv2.putText(panel, f"Tiempo: {current_time:.2f}s", (20, 135),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                
                # Combinar
                result = np.vstack([frame, panel])
                out.write(result)
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        out.release()
        
        logger.info(f"✓ Video guardado: {output_path}")
    
    def export_transcript(
        self,
        detections: List[Dict],
        output_path: Path,
        format: str = 'txt'
    ):
        """
        Exporta transcripción a archivo
        
        Args:
            detections: Lista de detecciones
            output_path: Ruta de salida
            format: 'txt', 'json', o 'srt' (subtítulos)
        """
        logger.info(f"Exportando transcripción: {output_path}")
        
        if format == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("TRANSCRIPCIÓN ASL\n")
                f.write("="*80 + "\n\n")
                
                for i, det in enumerate(detections, 1):
                    f.write(f"{i}. [{det['start_time']:.2f}s - {det['end_time']:.2f}s] "
                           f"{det['gloss']} ({det['confidence']:.1%})\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("Texto completo:\n")
                f.write(" ".join([det['gloss'] for det in detections]))
        
        elif format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(detections, f, indent=2, ensure_ascii=False)
        
        elif format == 'srt':
            with open(output_path, 'w', encoding='utf-8') as f:
                for i, det in enumerate(detections, 1):
                    start = self._format_srt_time(det['start_time'])
                    end = self._format_srt_time(det['end_time'])
                    
                    f.write(f"{i}\n")
                    f.write(f"{start} --> {end}\n")
                    f.write(f"{det['gloss']}\n")
                    f.write("\n")
        
        logger.info(f"✓ Transcripción guardada: {output_path}")
    
    def _format_srt_time(self, seconds: float) -> str:
        """Formatea tiempo para subtítulos SRT"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def main():
    parser = argparse.ArgumentParser(description="Traductor de videos ASL")
    
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--metadata_path", type=Path, required=True)
    parser.add_argument("--visual_extractor", type=Path, required=True)
    parser.add_argument("--pose_extractor", type=Path, required=True)
    parser.add_argument("--video_path", type=Path, required=True)
    
    parser.add_argument("--mode", type=str, default="sliding",
                       choices=["full", "sliding"],
                       help="full: video completo como una seña | sliding: detectar múltiples señas")
    
    parser.add_argument("--output_video", type=Path, default=None,
                       help="Guardar video anotado")
    parser.add_argument("--output_transcript", type=Path, default=None,
                       help="Guardar transcripción")
    parser.add_argument("--transcript_format", type=str, default="txt",
                       choices=["txt", "json", "srt"])
    
    # Parámetros para modo sliding
    parser.add_argument("--window_size", type=int, default=24)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--confidence_threshold", type=float, default=0.3)
    
    # Parámetros generales
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    
    args = parser.parse_args()
    
    # Validar archivos
    for path, name in [
        (args.model_path, "Modelo"),
        (args.metadata_path, "Metadata"),
        (args.visual_extractor, "Visual extractor"),
        (args.pose_extractor, "Pose extractor"),
        (args.video_path, "Video")
    ]:
        if not path.exists():
            logger.error(f"{name} no encontrado: {path}")
            return
    
    # Crear traductor
    translator = ASLVideoTranslator(
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        visual_extractor_path=args.visual_extractor,
        pose_extractor_path=args.pose_extractor,
        device=args.device
    )
    
    # Traducir
    if args.mode == "full":
        result = translator.translate_video_full(
            video_path=args.video_path,
            max_frames=args.max_frames
        )
        
        if args.output_video:
            translator.create_annotated_video(
                video_path=args.video_path,
                detections=result,
                output_path=args.output_video,
                mode='full'
            )
    
    else:  # sliding
        detections = translator.translate_video_sliding_window(
            video_path=args.video_path,
            window_size=args.window_size,
            stride=args.stride,
            confidence_threshold=args.confidence_threshold,
            max_frames=args.max_frames
        )
        
        if args.output_video:
            translator.create_annotated_video(
                video_path=args.video_path,
                detections=detections,
                output_path=args.output_video,
                mode='sliding'
            )
        
        if args.output_transcript:
            translator.export_transcript(
                detections=detections,
                output_path=args.output_transcript,
                format=args.transcript_format
            )


if __name__ == "__main__":
    main()