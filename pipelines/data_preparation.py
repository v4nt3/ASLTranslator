"""
Módulo de preprocesamiento y preparación de datos
Incluye: extracción de frames, keypoints, augmentation y generación de dataset.csv
"""

import cv2 # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import mediapipe as mp # type: ignore
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm # type: ignore
from config import config
import json
import os
import math

logger = logging.getLogger(__name__)


class FrameExtractor:
    """Extrae frames uniformemente distribuidos de los videos"""
    
    def __init__(self, config):
        self.config = config
        self.fps = config.data.fps
        self.frame_height = config.data.frame_height
        self.frame_width = config.data.frame_width
        self.num_frames = config.data.num_frames_per_clip
    
    def extract_frames(self, video_path: Path, output_dir: Path) -> List[np.ndarray]:
        """Extrae frames uniformemente distribuidos del video"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            logger.warning(f"Video vacío: {video_path}")
            return []
        
        # Calcular índices uniformes
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                frames.append(frame)
        
        cap.release()
        return frames
    
    def extract_all_videos(self):
        """Extrae frames de todos los videos en el directorio raw_videos"""
        video_dir = self.config.data_paths.raw_videos
        output_dir = self.config.data_paths.extracted_frames
        
        video_files = list(video_dir.glob("**/*.mp4")) + list(video_dir.glob("**/*.avi"))
        
        logger.info(f"Encontrados {len(video_files)} videos para procesar")
        
        for video_path in tqdm(video_files, desc="Extrayendo frames"):
            video_name = video_path.stem
            frames = self.extract_frames(video_path, output_dir)
            
            if frames:
                # Guardar frames como numpy array
                frames_array = np.array(frames)
                output_file = output_dir / f"{video_name}_frames.npy"
                np.save(output_file, frames_array)
                logger.debug(f"Guardados {len(frames)} frames de {video_name}")


class KeypointExtractor:
    """Extrae keypoints usando MediaPipe"""
    
    def __init__(self, config):
        self.config = config
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face = mp.solutions.face_detection
        
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
    
    def extract_pose_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extrae keypoints de pose de un frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks = np.array([
                [lm.x, lm.y, lm.z, lm.visibility] 
                for lm in results.pose_landmarks.landmark
            ])
            return landmarks  # shape: (33, 4)
        return None
    
    def extract_hand_keypoints(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extrae keypoints de manos"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        hand_keypoints = np.zeros((42, 4))  # 21 * 2 manos, with visibility
        
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
                start_idx = hand_idx * 21
                hand_keypoints[start_idx:start_idx+21] = np.array([
                    [lm.x, lm.y, lm.z, 1.0]  # Added visibility score (1.0 for detected)
                    for lm in hand_landmarks.landmark
                ])
        
        return hand_keypoints if np.any(hand_keypoints) else None
    
    def normalize_keypoints(self, pose_kpts: np.ndarray) -> np.ndarray:
        """Normaliza keypoints (centrado corporal y escalado)"""
        # Usar cadera como punto de referencia
        left_hip = pose_kpts[23][:3]  # índice 23
        right_hip = pose_kpts[24][:3]  # índice 24
        center = (left_hip + right_hip) / 2
        
        # Calcular escala (longitud del torso)
        shoulder_dist = np.linalg.norm(pose_kpts[11][:3] - pose_kpts[12][:3])
        scale = max(shoulder_dist, 0.1)
        
        # Normalizar
        normalized = pose_kpts.copy()
        normalized[:, :3] = (pose_kpts[:, :3] - center) / scale
        normalized[:, :3] = np.clip(normalized[:, :3], -1, 1)
        
        return normalized
    
    def extract_all_keypoints(self):
        """Extrae keypoints de todos los videos"""
        frames_dir = self.config.data_paths.extracted_frames
        output_dir = self.config.data_paths.keypoints
        
        frame_files = list(frames_dir.glob("*_frames.npy"))
        logger.info(f"Extrayendo keypoints de {len(frame_files)} videos")
        
        for frame_file in tqdm(frame_files, desc="Extrayendo keypoints"):
            video_name = frame_file.stem.replace("_frames", "")
            frames = np.load(frame_file)
            
            keypoints_sequence = []
            
            for frame in frames:
                pose_kpts = self.extract_pose_keypoints(frame)
                
                if pose_kpts is not None:
                    pose_kpts = self.normalize_keypoints(pose_kpts)
                    hand_kpts = self.extract_hand_keypoints(frame)
                    
                    if hand_kpts is not None:
                        combined_kpts = np.concatenate([pose_kpts, hand_kpts], axis=0)
                    else:
                        hand_placeholder = np.zeros((42, 4))
                        combined_kpts = np.concatenate([pose_kpts, hand_placeholder], axis=0)
                    
                    keypoints_sequence.append(combined_kpts)
            
            if keypoints_sequence:
                keypoints_array = np.array(keypoints_sequence)
                output_file = output_dir / f"{video_name}_keypoints.npy"
                np.save(output_file, keypoints_array)
                logger.debug(f"Keypoints guardados para {video_name}: shape {keypoints_array.shape}")


class ActionSegmenter:
    """
    Detecta automáticamente la región donde ocurre la seña usando keypoints.
    Calcula un action score por frame y encuentra la ventana más relevante.
    """
    
    def __init__(self, config, debug=False):
        self.config = config
        self.debug = debug
        # Índices de MediaPipe para puntos relevantes
        self.left_hand = list(range(468 - 42, 468 - 21))  # Mano izquierda (21 puntos)
        self.right_hand = list(range(468 - 21, 468))      # Mano derecha (21 puntos)
        self.face_region = [10, 152, 234, 454]             # Puntos de cara para referencia
    
    def calculate_action_score(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Calcula un score de acción por frame (0 a 1).
        Basado en: visibilidad de manos, altura de manos, movimiento.
        
        Args:
            keypoints: shape (T, 75, 4) - [pose(33) + hands(42)]
        
        Returns:
            scores: shape (T,) - score de acción por frame
        """
        T = len(keypoints)
        scores = np.zeros(T)
        
        for t in range(T):
            frame_kpts = keypoints[t]  # (75, 4)
            
            # Pose keypoints (primeros 33)
            pose_kpts = frame_kpts[:33]
            
            # Hand keypoints (42 siguientes: 21 izquierda + 21 derecha)
            left_hand_kpts = frame_kpts[33:54]    # 21 puntos mano izquierda
            right_hand_kpts = frame_kpts[54:75]   # 21 puntos mano derecha
            
            score = 0.0
            
            # 1. Visibilidad de manos (weight: 0.3)
            left_visibility = np.mean(left_hand_kpts[:, 3])      # dimensión 3 es visibility
            right_visibility = np.mean(right_hand_kpts[:, 3])
            hand_visibility = (left_visibility + right_visibility) / 2.0
            score += 0.3 * hand_visibility
            
            # 2. Manos levantadas (arriba de las caderas) (weight: 0.3)
            hip_y = np.mean([pose_kpts[23, 1], pose_kpts[24, 1]])  # altura de caderas
            
            left_hand_y = np.mean(left_hand_kpts[:, 1])
            right_hand_y = np.mean(right_hand_kpts[:, 1])
            
            hands_raised = 0.0
            if left_visibility > 0.3:
                hands_raised += 1.0 if left_hand_y < hip_y else 0.0
            if right_visibility > 0.3:
                hands_raised += 1.0 if right_hand_y < hip_y else 0.0
            hands_raised = hands_raised / 2.0
            score += 0.3 * hands_raised
            
            # 3. Distancia entre manos y cara (weight: 0.2)
            # Puntos de la cara (nariz como referencia)
            nose = pose_kpts[0]
            
            left_hand_pos = np.mean(left_hand_kpts[:3, :3], axis=0)  # 3 primeros puntos
            right_hand_pos = np.mean(right_hand_kpts[:3, :3], axis=0)
            
            left_dist = np.linalg.norm(left_hand_pos - nose[:3])
            right_dist = np.linalg.norm(right_hand_pos - nose[:3])
            
            # Score más alto si las manos están cerca de la cara (0.3 a 1.0 es distancia relevante)
            hand_proximity = 1.0 - np.clip(np.mean([left_dist, right_dist]), 0, 0.5) / 0.5
            score += 0.2 * max(0, hand_proximity)
            
            # 4. Magnitud de movimiento entre frames (weight: 0.2)
            if t > 0:
                prev_kpts = keypoints[t - 1]
                # Calcular diferencia euclidiana en coordenadas de manos
                left_hand_prev = np.mean(prev_kpts[33:54, :3], axis=0)
                left_hand_curr = np.mean(left_hand_kpts[:, :3], axis=0)
                left_motion = np.linalg.norm(left_hand_curr - left_hand_prev)
                
                right_hand_prev = np.mean(prev_kpts[54:75, :3], axis=0)
                right_hand_curr = np.mean(right_hand_kpts[:, :3], axis=0)
                right_motion = np.linalg.norm(right_hand_curr - right_hand_prev)
                
                # Normalizar movimiento (considerar que el rango es 0-0.2 en coordenadas normalizadas)
                motion_score = np.clip(np.mean([left_motion, right_motion]) / 0.05, 0, 1)
                score += 0.2 * motion_score
            
            scores[t] = np.clip(score, 0, 1)
        
        return scores
    
    def find_action_window(self, scores: np.ndarray, min_length: int = 8) -> Tuple[int, int]:
        """
        Encuentra la ventana continua de máxima actividad usando suma móvil.
        
        Args:
            scores: shape (T,)
            min_length: longitud mínima de ventana
        
        Returns:
            (start_idx, end_idx) - índices de la ventana de máxima actividad
        """
        T = len(scores)
        
        if T < min_length:
            return 0, T
        
        # Suavizar scores con media móvil
        window_size = min(5, T // 4)
        smoothed_scores = np.convolve(scores, np.ones(window_size) / window_size, mode='same')
        
        # Encontrar suma móvil para ventanas de min_length
        max_sum = -np.inf
        best_start = 0
        
        for start in range(T - min_length + 1):
            window_sum = np.sum(smoothed_scores[start:start + min_length])
            if window_sum > max_sum:
                max_sum = window_sum
                best_start = start
        
        best_end = min(best_start + min_length, T)
        
        # Expandir ventana si hay frames adyacentes con score alto
        threshold = np.mean(scores) * 0.5
        
        while best_start > 0 and scores[best_start - 1] > threshold:
            best_start -= 1
        
        while best_end < T and scores[best_end] > threshold:
            best_end += 1
        
        return best_start, best_end
    
    def segment_action(self, keypoints: np.ndarray, 
                      frames: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """
        Segmenta la región útil de la seña en los keypoints.
        
        Args:
            keypoints: shape (T, 75, 4)
            frames: shape (T, H, W, 3) - opcional para debug
        
        Returns:
            (start_idx, end_idx) - región de la seña
        """
        try:
            # Calcular action score por frame
            scores = self.calculate_action_score(keypoints)
            
            if self.debug:
                logger.debug(f"[ActionSegmenter] Scores: min={scores.min():.3f}, "
                            f"max={scores.max():.3f}, mean={scores.mean():.3f}")
            
            # Si todos los scores son muy bajos, usar fallback
            if scores.max() < 0.2:
                logger.warning("[ActionSegmenter] Action scores muy bajos, usando fallback (middle window)")
                T = len(keypoints)
                mid = T // 2
                pad = max(self.config.data.num_frames_per_clip // 2, 4)
                return max(0, mid - pad), min(T, mid + pad)
            
            # Encontrar ventana de máxima actividad
            start_idx, end_idx = self.find_action_window(scores, 
                                                         self.config.data.num_frames_per_clip // 2)
            
            # Asegurar que la ventana sea lo suficientemente larga
            if end_idx - start_idx < self.config.data.num_frames_per_clip:
                pad = self.config.data.num_frames_per_clip - (end_idx - start_idx)
                start_idx = max(0, start_idx - pad // 2)
                end_idx = min(len(keypoints), end_idx + (pad - pad // 2))
            
            if self.debug:
                logger.debug(f"[ActionSegmenter] Segmento detectado: [{start_idx}, {end_idx}] "
                            f"longitud={end_idx - start_idx}")
            
            return start_idx, end_idx
        
        except Exception as e:
            logger.warning(f"[ActionSegmenter] Error en segmentación: {e}. Usando fallback.")
            T = len(keypoints)
            return 0, T


class ClipCreator:
    """Crea clips de videos con control de solapamiento inteligente"""
    
    def __init__(self, config, use_augmentation: bool = False, debug: bool = False):
        self.config = config
        self.num_frames = config.data.num_frames_per_clip
        self.use_augmentation = use_augmentation
        self.debug = debug
        self.action_segmenter = ActionSegmenter(config, debug=debug)
        # self.augmentor = MultimodalAugmentor(config) if use_augmentation else None # Placeholder for augmentor
    
    def calculate_max_clips_possible(self, total_frames: int) -> int:
        """
        Calcula el máximo número de clips que puede generarse de un video.
        
        Fórmula: max_clips = ceil(T / (L * 0.6))
        Donde: T = total frames, L = frames por clip
        
        Esto permite solapamiento: si T=100 y L=24:
        - sin solapamiento: max = 100/24 = 4 clips
        - con solapamiento 60%: max = 100/(24*0.6) = 100/14.4 = 7 clips
        
        Args:
            total_frames: Número total de frames en el video
        
        Returns:
            Número máximo de clips posibles (mínimo 1)
        """
        if total_frames < self.num_frames:
            return 1
        
        # stride_factor = 0.6 permite 40% de solapamiento entre clips
        stride_factor = 0.6
        max_clips = math.ceil(total_frames / (self.num_frames * stride_factor))
        return max(1, max_clips)
    
    def generate_clip_offsets(self, total_frames: int, clips_to_create: int) -> List[int]:
        """
        Genera offsets para clips distribuidos uniformemente con solapamiento controlado.
        
        Fórmula: offset_k = (T - L) * (k / (clips_to_create - 1))
        
        Esto asegura que los clips estén distribuidos uniformemente a lo largo del video,
        permitiendo solapamiento cuando sea necesario.
        
        Args:
            total_frames: Número total de frames
            clips_to_create: Número de clips a generar
        
        Returns:
            Lista de offsets (índices de inicio para cada clip)
        """
        if clips_to_create == 1:
            # Un solo clip: centrado
            offset = max(0, (total_frames - self.num_frames) // 2)
            return [offset]
        
        # Múltiples clips: distribuir uniformemente
        offsets = []
        for k in range(clips_to_create):
            offset_k = (total_frames - self.num_frames) * (k / (clips_to_create - 1))
            offset = int(offset_k)
            offsets.append(offset)
        
        return offsets
    
    def load_dataset_metadata(self) -> Dict[str, Dict]:
        """
        Carga el archivo dataset_meta.json que mapea videos a clases.
        
        Soporta el formato: {"root_dir": "...", "videos": [...]}
        """
        meta_path = self.config.data_paths.raw_videos / "dataset_meta.json"
        
        if not meta_path.exists():
            logger.warning(f"dataset_meta.json no encontrado en {meta_path}. "
                          "Usando nombres de archivos para inferir clases.")
            return {}
        
        try:
            with open(meta_path, 'r') as f:
                raw_metadata = json.load(f)
            
            metadata = {}
            
            # Si tiene estructura {"root_dir": "...", "videos": [...]}
            if isinstance(raw_metadata, dict) and "videos" in raw_metadata:
                videos_list = raw_metadata["videos"]
                for entry in videos_list:
                    video_file = entry.get("video_file", "")
                    if video_file:
                        metadata[video_file] = {
                            "class_id": entry.get("class_id"),
                            "class_name": entry.get("class_name"),
                            "video_path": entry.get("video_path")
                        }
                logger.info(f"Metadata cargado: {len(metadata)} videos desde '{meta_path}'")
                
            # Si es una lista directa (legacy format)
            elif isinstance(raw_metadata, list):
                for entry in raw_metadata:
                    video_file = entry.get("video_file", "")
                    if video_file:
                        metadata[video_file] = {
                            "class_id": entry.get("class_id"),
                            "class_name": entry.get("class_name"),
                            "video_path": entry.get("video_path")
                        }
                logger.info(f"Metadata cargado: {len(metadata)} videos desde lista")
                
            # Si es un diccionario (legacy format)
            else:
                metadata = raw_metadata
                logger.info(f"Metadata cargado: {len(metadata)} videos desde diccionario")
            
            logger.debug(f"Sample metadata keys: {list(metadata.keys())[:3]}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error cargando dataset_meta.json: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}
    
    def extract_class_id(self, video_name: str, metadata: Dict[str, Dict]) -> Optional[int]:
        """
        Extrae class_id de los metadatos.
        El metadata tiene la estructura:
        {
            "023931338852502426-1 DOLLAR.mp4": {"class_id": 0, "class_name": "Hello", ...},
            ...
        }
        
        Estrategia de matching (en orden):
        1. Match exacto: "video_name.mp4" en metadata
        2. Match sin extensión (comparar stems)
        3. Búsqueda parcial si la key contiene el video_name
        4. Búsqueda by class_name si es disponible
        """
        if not metadata:
            logger.debug(f"No hay metadata disponible. Retornando None para: {video_name}")
            return None
        
        # Asegurar que video_name tiene extensión
        if not video_name.endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
            video_name_with_ext = f"{video_name}.mp4"
        else:
            video_name_with_ext = video_name
            video_name = video_name.rsplit('.', 1)[0]
        
        # 1. Match exacto
        if video_name_with_ext in metadata:
            class_id = metadata[video_name_with_ext].get('class_id')
            if class_id is not None:
                logger.info(f"[MATCH] Found class_id={class_id} for '{video_name}' (exact match)")
                return class_id
        
        # 2. Match sin extensión (comparar stems)
        for key, value in metadata.items():
            key_stem = key.rsplit('.', 1)[0] if '.' in key else key
            if key_stem == video_name:
                class_id = value.get('class_id')
                if class_id is not None:
                    logger.info(f"[MATCH] Found class_id={class_id} for '{video_name}' (stem match)")
                    return class_id
        
        # 3. Búsqueda parcial
        video_name_lower = video_name.lower()
        for key, value in metadata.items():
            if video_name_lower in key.lower():
                class_id = value.get('class_id')
                if class_id is not None:
                    logger.info(f"[MATCH] Found class_id={class_id} for '{video_name}' (partial match with '{key}')")
                    return class_id
        
        # 4. Si el video_name contiene guiones/espacios, intentar sin ellos
        video_name_clean = video_name.replace('-', '').replace('_', '').replace(' ', '').lower()
        for key, value in metadata.items():
            key_clean = key.rsplit('.', 1)[0].replace('-', '').replace('_', '').replace(' ', '').lower()
            if video_name_clean in key_clean or key_clean in video_name_clean:
                class_id = value.get('class_id')
                if class_id is not None:
                    logger.info(f"[MATCH] Found class_id={class_id} for '{video_name}' (fuzzy match)")
                    return class_id
        
        # Si no encuentra, loguear información útil (ASCII-safe)
        logger.error(f"[ERROR] Could not extract class_id for '{video_name}'")
        logger.debug(f"[DEBUG] Metadata contains {len(metadata)} entries")
        logger.debug(f"[DEBUG] First 3 metadata keys: {list(metadata.keys())[:3]}")
        
        return None
    
    def get_clips_per_class(self, num_samples_per_class: Dict[int, int]) -> Dict[int, int]:
        """Determina cuántos clips crear por clase según el tamaño"""
        clips_per_class = {}
        
        low_threshold = 27
        medium_threshold = 40
        
        for class_id, num_samples in num_samples_per_class.items():
            if num_samples <= low_threshold:
                clips_per_class[class_id] = self.config.data.clips_distribution["low"]
            elif num_samples <= medium_threshold:
                clips_per_class[class_id] = self.config.data.clips_distribution["medium"]
            else:
                clips_per_class[class_id] = self.config.data.clips_distribution["high"]
        
        logger.debug(f"Clips por clase: {clips_per_class}")
        return clips_per_class
    
    def validate_and_sync_shapes(self, frames: np.ndarray, keypoints: np.ndarray, 
                                  video_name: str) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Verifica y sincroniza shapes entre frames y keypoints.
        
        Si hay mismatch en T (número de frames):
        - Toma T = min(T_frames, T_keypoints)
        - Recorta ambos al tamaño T
        - Esto evita inconsistencias sin descartar videos
        
        Expected:
        - frames: (T, H, W, C)
        - keypoints: (T, N, D) donde N=75, D=4
        """
        if frames.ndim != 4:
            logger.warning(f"[{video_name}] Frames shape inválido: {frames.shape}, esperado (T, H, W, C)")
            return frames, keypoints, False
        
        if keypoints.ndim != 3:
            logger.warning(f"[{video_name}] Keypoints ndim inválido: {keypoints.ndim}, esperado 3")
            return frames, keypoints, False
        
        T_f, H, W, C = frames.shape
        T_k, N, D = keypoints.shape
        
        if T_f != T_k:
            T_min = min(T_f, T_k)
            logger.warning(f"[{video_name}] Mismatch en T: frames={T_f}, keypoints={T_k} -> Sincronizando a T={T_min}")
            frames = frames[:T_min]
            keypoints = keypoints[:T_min]
            T_f = T_min
        
        if N != 75:
            logger.warning(f"[{video_name}] N keypoints incorrecto: {N}, esperado 75")
            return frames, keypoints, False
        
        if D != 4:
            logger.warning(f"[{video_name}] D keypoints incorrecto: {D}, esperado 4")
            return frames, keypoints, False
        
        if T_f < self.num_frames:
            logger.warning(f"[{video_name}] Video muy corto: {T_f} frames, se requerirá padding a {self.num_frames}")
        
        logger.debug(f"[{video_name}] Shapes validados y sincronizados: frames={frames.shape}, keypoints={keypoints.shape}")
        return frames, keypoints, True

    def validate_shapes(self, frames: np.ndarray, keypoints: np.ndarray, 
                       video_name: str) -> bool:
        """
        Wrapper legacy para mantener compatibilidad.
        """
        _, _, is_valid = self.validate_and_sync_shapes(frames, keypoints, video_name)
        return is_valid
    
    def clean_intermediate_files(self, video_name: str) -> bool:
        """
        Borra los archivos intermedios (frames.npy y keypoints.npy) después de crear clips.
        
        Returns: True si se borraron exitosamente, False si hubo error
        """
        frames_file = self.config.data_paths.extracted_frames / f"{video_name}_frames.npy"
        keypoints_file = self.config.data_paths.keypoints / f"{video_name}_keypoints.npy"
        
        try:
            if frames_file.exists():
                os.remove(frames_file)
                logger.debug(f"Borrado: {frames_file.name}")
            
            if keypoints_file.exists():
                os.remove(keypoints_file)
                logger.debug(f"Borrado: {keypoints_file.name}")
            
            logger.info(f"[{video_name}] Archivos intermedios eliminados correctamente")
            return True
        
        except Exception as e:
            logger.error(f"[{video_name}] Error borrando archivos intermedios: {e}")
            return False
    
    def create_clips_and_dataset(self, class_mapping: Optional[Dict[str, int]] = None):
        """
        Crea clips de videos y genera dataset.csv con clips solapados e inteligentes.
        
        Garantías:
        - Nunca descarta un video por tener pocos frames
        - Todo video genera mínimo 1 clip
        - Offsets distribuidos uniformemente
        - Solapamiento automático controlado
        
        Args:
            class_mapping: Dict mapeo opcional de nombre_clase -> class_id
        """
        frames_dir = self.config.data_paths.extracted_frames
        keypoints_dir = self.config.data_paths.keypoints
        clips_dir = self.config.data_paths.clips
        
        logger.info("=" * 80)
        logger.info("INICIANDO CREACIÓN DE CLIPS CON SOLAPAMIENTO INTELIGENTE")
        logger.info("=" * 80)
        logger.info(f"Frames por clip: {self.num_frames}")
        
        # Cargar metadata
        metadata = self.load_dataset_metadata()
        
        # 1. Contar videos por clase
        frame_files = sorted(list(frames_dir.glob("*_frames.npy")))
        logger.info(f"Encontrados {len(frame_files)} videos con frames extraídos")
        
        num_samples_per_class = {}
        video_to_class = {}
        
        for frame_file in frame_files:
            video_name = frame_file.stem.replace("_frames", "")
            class_id = self.extract_class_id(video_name, metadata)
            
            if class_id is not None:
                video_to_class[video_name] = class_id
                num_samples_per_class[class_id] = num_samples_per_class.get(class_id, 0) + 1
        
        logger.info(f"Clases encontradas: {dict(sorted(num_samples_per_class.items()))}")
        logger.info(f"Distribución de videos por clase: {dict(sorted(num_samples_per_class.items()))}")
        
        # 2. Determinar clips por clase
        clips_per_class = self.get_clips_per_class(num_samples_per_class)
        logger.info(f"Clips a generar por clase: {dict(sorted(clips_per_class.items()))}")
        
        # 3. Crear clips con solapamiento inteligente
        dataset_rows = []
        successful_clips = 0
        failed_videos = []
        skipped_videos = []
        
        clip_statistics = {}  # Para debug: rastrear clipes por video
        
        for frame_file in tqdm(frame_files, desc="Creando clips"):
            video_name = frame_file.stem.replace("_frames", "")
            keypoint_file = keypoints_dir / f"{video_name}_keypoints.npy"
            
            # Validar que existan los archivos de keypoints
            if not keypoint_file.exists():
                logger.warning(f"[{video_name}] No encontrado keypoints. Saltando...")
                failed_videos.append(video_name)
                continue
            
            # Obtener class_id
            class_id = video_to_class.get(video_name)
            if class_id is None:
                logger.warning(f"[{video_name}] class_id no encontrado. Saltando...")
                failed_videos.append(video_name)
                continue
            
            # Cargar datos
            try:
                frames = np.load(frame_file, mmap_mode='r')
                keypoints = np.load(keypoint_file, mmap_mode='r')
            except Exception as e:
                logger.error(f"[{video_name}] Error cargando archivos: {e}")
                failed_videos.append(video_name)
                continue
            
            # Validar shapes
            frames, keypoints, is_valid = self.validate_and_sync_shapes(frames, keypoints, video_name)
            if not is_valid:
                logger.warning(f"[{video_name}] Shapes inválidos. Saltando...")
                failed_videos.append(video_name)
                continue
            
            # Segmentar acción para encontrar región útil
            try:
                seg_start, seg_end = self.action_segmenter.segment_action(keypoints[:])
                logger.debug(f"[{video_name}] Region util: [{seg_start}, {seg_end}]")
            except Exception as e:
                logger.warning(f"[{video_name}] Error en ActionSegmenter: {e}. Usando todo el video.")
                seg_start, seg_end = 0, len(keypoints)
            
            # Recortar a la región segmentada
            segmented_frames = frames[seg_start:seg_end]
            segmented_keypoints = keypoints[seg_start:seg_end]
            available_frames = len(segmented_frames)
            
            normalized_frames, normalized_keypoints = self.normalize_frames_and_keypoints(
                segmented_frames, segmented_keypoints, self.num_frames
            )
            
            clips_requested = clips_per_class.get(class_id, 1)
            max_clips_possible = self.calculate_max_clips_possible(len(normalized_frames))
            clips_to_create = min(clips_requested, max_clips_possible)
            
            logger.info(f"[{video_name}] class_id={class_id}, {available_frames} frames -> {len(normalized_frames)} "
                       f"normalized | clips_requested={clips_requested}, max_possible={max_clips_possible}, "
                       f"creating={clips_to_create}")
            
            offsets = self.generate_clip_offsets(len(normalized_frames), clips_to_create)
            logger.debug(f"[{video_name}] Offsets generados: {offsets}")
            
            # Crear los clips
            clips_created_for_video = 0
            
            for clip_idx, offset in enumerate(offsets):
                try:
                    # Extraer clip usando offset
                    clip_start = offset
                    clip_end = min(offset + self.num_frames, len(normalized_frames))
                    
                    clip_frames = normalized_frames[clip_start:clip_end]
                    clip_keypoints = normalized_keypoints[clip_start:clip_end]
                    
                    # Garantizar que el clip tiene exactamente num_frames
                    # (por si acaso hay edge cases)
                    if clip_frames.shape[0] < self.num_frames:
                        num_pad = self.num_frames - clip_frames.shape[0]
                        last_frame = clip_frames[-1]
                        last_keypoint = clip_keypoints[-1]
                        pad_frames = np.repeat(last_frame[np.newaxis, :, :, :], num_pad, axis=0)
                        pad_keypoints = np.repeat(last_keypoint[np.newaxis, :, :], num_pad, axis=0)
                        clip_frames = np.concatenate([clip_frames, pad_frames], axis=0)
                        clip_keypoints = np.concatenate([clip_keypoints, pad_keypoints], axis=0)
                    
                    # Validar clip
                    if clip_frames.shape[0] != self.num_frames or clip_keypoints.shape[0] != self.num_frames:
                        logger.warning(f"[{video_name}] Clip {clip_idx} shape invalido: "
                                     f"frames={clip_frames.shape[0]}, keypoints={clip_keypoints.shape[0]}")
                        continue
                    
                    # Guardar clip
                    clip_name = f"{video_name}_clip{clip_idx}"
                    clip_frames_path = clips_dir / f"{clip_name}_frames.npy"
                    clip_keypoints_path = clips_dir / f"{clip_name}_keypoints.npy"
                    
                    np.save(clip_frames_path, clip_frames)
                    np.save(clip_keypoints_path, clip_keypoints)
                    
                    # Agregar a dataset
                    dataset_rows.append({
                        "clip_path": str(clip_frames_path.relative_to(self.config.data_paths.raw_videos.parent)),
                        "keypoints_path": str(clip_keypoints_path.relative_to(self.config.data_paths.raw_videos.parent)),
                        "clip_name": f"{video_name}_clip{clip_idx}",
                        "class_id": class_id,
                        "video_source": video_name,
                        "clip_index": clip_idx,
                        "offset": offset,
                        "total_frames_normalized": len(normalized_frames)
                    })
                    
                    clips_created_for_video += 1
                    successful_clips += 1
                    logger.debug(f"[{video_name}] Clip {clip_idx} creado (offset={offset})")
                
                except Exception as e:
                    logger.error(f"[{video_name}] Error creando clip {clip_idx}: {e}")
            
            if clips_created_for_video > 0:
                clip_statistics[video_name] = clips_created_for_video
                self.clean_intermediate_files(video_name)
            else:
                logger.warning(f"[{video_name}] No se crearon clips. Archivos intermedios NO borrados.")
                failed_videos.append(video_name)
        
        # Guardar CSV
        df = pd.DataFrame(dataset_rows)
        
        logger.info("=" * 80)
        logger.info("CREACION DE CLIPS COMPLETADA")
        logger.info("=" * 80)
        logger.info(f"Clips creados exitosamente: {successful_clips}")
        logger.info(f"Videos procesados: {len(video_to_class)}")
        logger.info(f"Videos exitosos: {len(clip_statistics)}")
        logger.info(f"Videos con errores: {len(failed_videos)}")
        logger.info(f"Videos saltados: {len(skipped_videos)}")
        
        if failed_videos:
            logger.warning(f"Videos fallidos: {failed_videos}")
        
        logger.info(f"Distribucion de clips por clase: {df['class_id'].value_counts().to_dict()}")
        logger.info(f"Clips por video: min={df.groupby('video_source').size().min()}, "
                   f"max={df.groupby('video_source').size().max()}, "
                   f"avg={df.groupby('video_source').size().mean():.2f}")
        
        df.to_csv(self.config.data_paths.dataset_csv, index=False)
        logger.info(f"Dataset CSV guardado: {self.config.data_paths.dataset_csv}")
        logger.info(f"Total clips en dataset: {len(df)}")
        
        return df
    
    def normalize_frames_and_keypoints(self, frames: np.ndarray, keypoints: np.ndarray, 
                                       target_length: int = None) -> tuple:
        """
        Normaliza frames y keypoints al target_length.
        
        Si frames >= target_length: toma target_length frames centrados
        Si frames < target_length: hace padding repitiendo el último frame
        
        Args:
            frames: np.ndarray shape (T, H, W, C)
            keypoints: np.ndarray shape (T, 75, 4)
            target_length: número de frames destino (default: self.num_frames)
        
        Returns:
            (normalized_frames, normalized_keypoints) ambos con shape (target_length, ...)
        """
        if target_length is None:
            target_length = self.num_frames
        
        current_length = len(frames)
        
        if current_length >= target_length:
            start_idx = (current_length - target_length) // 2
            end_idx = start_idx + target_length
            normalized_frames = frames[start_idx:end_idx]
            normalized_keypoints = keypoints[start_idx:end_idx]
        
        else:
            num_pad = target_length - current_length
            last_frame = frames[-1]  # último frame
            last_keypoint = keypoints[-1]  # último keypoint
            
            # Crear arrays de padding
            pad_frames = np.repeat(last_frame[np.newaxis, :, :, :], num_pad, axis=0)
            pad_keypoints = np.repeat(last_keypoint[np.newaxis, :, :], num_pad, axis=0)
            
            # Concatenar: video_original + padding
            normalized_frames = np.concatenate([frames, pad_frames], axis=0)
            normalized_keypoints = np.concatenate([keypoints, pad_keypoints], axis=0)
        
        return normalized_frames, normalized_keypoints


class DataPreprocessor:
    """Orquestador principal del preprocesamiento"""
    
    def __init__(self, config):
        self.config = config
        self.frame_extractor = FrameExtractor(config)
        self.keypoint_extractor = KeypointExtractor(config)
        self.clip_creator = ClipCreator(config)
    
    def extract_frames(self):
        """Extrae frames de videos"""
        self.frame_extractor.extract_all_videos()
    
    def extract_keypoints(self):
        """Extrae keypoints de frames"""
        self.keypoint_extractor.extract_all_keypoints()
    
    def create_clips_and_csv(self, class_mapping: Optional[Dict[str, int]] = None):
        """Crea clips y genera dataset.csv"""
        if class_mapping is None:
            class_mapping = {}
        self.clip_creator.create_clips_and_dataset(class_mapping)
