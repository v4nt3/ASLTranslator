import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KeypointActionSegmenter:
    """
    Detecta automáticamente segmentos de acción útiles en keypoints.
    Basado en el ActionSegmenter original de data_preparation.py
    """
    
    def __init__(self, debug=False):
        self.debug = debug
    
    def calculate_action_score(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Calcula un score de acción por frame (0 a 1).
        
        Args:
            keypoints: shape (T, 75, 4) - [pose(33) + hands(42)]
                      o (T, 300) - formato flat de MediaPipe
        
        Returns:
            scores: shape (T,) - score de acción por frame
        """
        # Si keypoints están en formato flat (T, 300), reshape a (T, 75, 4)
        if keypoints.ndim == 2 and keypoints.shape[1] == 300:
            T = keypoints.shape[0]
            keypoints = keypoints.reshape(T, 75, 4)
        
        T = len(keypoints)
        scores = np.zeros(T)
        
        for t in range(T):
            frame_kpts = keypoints[t]  # (75, 4)
            
            # Pose keypoints (primeros 33)
            pose_kpts = frame_kpts[:33]
            
            # Hand keypoints (42 siguientes: 21 izquierda + 21 derecha)
            left_hand_kpts = frame_kpts[33:54]
            right_hand_kpts = frame_kpts[54:75]
            
            score = 0.0
            
            # 1. Visibilidad de manos (weight: 0.3)
            left_visibility = np.mean(left_hand_kpts[:, 3])
            right_visibility = np.mean(right_hand_kpts[:, 3])
            hand_visibility = (left_visibility + right_visibility) / 2.0
            score += 0.3 * hand_visibility
            
            # 2. Manos levantadas (arriba de las caderas) (weight: 0.3)
            hip_y = np.mean([pose_kpts[23, 1], pose_kpts[24, 1]])
            
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
            nose = pose_kpts[0]
            
            left_hand_pos = np.mean(left_hand_kpts[:3, :3], axis=0)
            right_hand_pos = np.mean(right_hand_kpts[:3, :3], axis=0)
            
            left_dist = np.linalg.norm(left_hand_pos - nose[:3])
            right_dist = np.linalg.norm(right_hand_pos - nose[:3])
            
            hand_proximity = 1.0 - np.clip(np.mean([left_dist, right_dist]), 0, 0.5) / 0.5
            score += 0.2 * max(0, hand_proximity)
            
            # 4. Magnitud de movimiento entre frames (weight: 0.2)
            if t > 0:
                prev_kpts = keypoints[t - 1]
                
                left_hand_prev = np.mean(prev_kpts[33:54, :3], axis=0)
                left_hand_curr = np.mean(left_hand_kpts[:, :3], axis=0)
                left_motion = np.linalg.norm(left_hand_curr - left_hand_prev)
                
                right_hand_prev = np.mean(prev_kpts[54:75, :3], axis=0)
                right_hand_curr = np.mean(right_hand_kpts[:, :3], axis=0)
                right_motion = np.linalg.norm(right_hand_curr - right_hand_prev)
                
                motion_score = np.clip(np.mean([left_motion, right_motion]) / 0.05, 0, 1)
                score += 0.2 * motion_score
            
            scores[t] = np.clip(score, 0, 1)
        
        return scores
    
    def find_action_window(self, scores: np.ndarray, min_length: int = 8) -> Tuple[int, int]:
        """
        Encuentra la ventana continua de máxima actividad.
        
        Args:
            scores: shape (T,)
            min_length: longitud mínima de ventana
        
        Returns:
            (start_idx, end_idx) - índices de la ventana
        """
        T = len(scores)
        
        if T < min_length:
            return 0, T
        
        # Suavizar scores con media móvil
        window_size = min(5, T // 4)
        smoothed_scores = np.convolve(scores, np.ones(window_size) / window_size, mode='same')
        
        # Encontrar ventana de min_length con máxima suma
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
    
    def segment_keypoints(self, keypoints: np.ndarray, 
                          min_length: int = 8) -> Tuple[np.ndarray, Dict]:
        """
        Segmenta keypoints para quedarse solo con frames útiles.
        
        Args:
            keypoints: (T, 300) o (T, 75, 4)
            min_length: longitud mínima de segmento
        
        Returns:
            filtered_keypoints: (T_filtered, 300)
            info: Dict con información de segmentación
        """
        original_shape = keypoints.shape
        scores = self.calculate_action_score(keypoints)
        start_idx, end_idx = self.find_action_window(scores, min_length)
        
        # Extraer segmento
        if keypoints.ndim == 2 and keypoints.shape[1] == 300:
            # Ya está en formato flat
            filtered = keypoints[start_idx:end_idx]
        else:
            # Reshape a flat
            keypoints_flat = keypoints.reshape(keypoints.shape[0], -1)
            filtered = keypoints_flat[start_idx:end_idx]
        
        info = {
            'original_frames': original_shape[0],
            'filtered_frames': filtered.shape[0],
            'start_idx': start_idx,
            'end_idx': end_idx,
            'mean_score': float(np.mean(scores)),
            'max_score': float(np.max(scores)),
            'segment_mean_score': float(np.mean(scores[start_idx:end_idx]))
        }
        
        return filtered, info


def process_keypoints_directory(input_dir: str, output_dir: str, 
                               visualize: bool = False,
                               min_length: int = 8):
    """
    Procesa todos los archivos de keypoints en un directorio.
    
    Args:
        input_dir: Directorio con keypoints crudos (.npy)
        output_dir: Directorio para keypoints filtrados
        visualize: Si True, guarda gráficos de scores
        min_length: Longitud mínima de segmento
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    segmenter = KeypointActionSegmenter(debug=visualize)
    
    # Encontrar todos los archivos .npy
    keypoint_files = list(input_path.rglob("*.npy"))
    
    if not keypoint_files:
        logger.error(f"No se encontraron archivos .npy en {input_dir}")
        return
    
    logger.info(f"Encontrados {len(keypoint_files)} archivos de keypoints")
    
    stats = {
        'total': 0,
        'filtered': 0,
        'total_original_frames': 0,
        'total_filtered_frames': 0
    }
    
    metadata = []
    
    for kpt_file in tqdm(keypoint_files, desc="Filtrando keypoints"):
        try:
            # Cargar keypoints
            keypoints = np.load(kpt_file)  # (T, 300) o (T, 75, 4)
            
            # Segmentar
            filtered_kpts, info = segmenter.segment_keypoints(keypoints, min_length)
            
            # Crear estructura de directorios en output
            relative_path = kpt_file.relative_to(input_path)
            output_file = output_path / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Guardar keypoints filtrados
            np.save(output_file, filtered_kpts.astype(np.float32))
            
            # Actualizar stats
            stats['total'] += 1
            stats['filtered'] += 1
            stats['total_original_frames'] += info['original_frames']
            stats['total_filtered_frames'] += info['filtered_frames']
            
            # Guardar metadata
            metadata.append({
                'file': str(relative_path),
                'output_file': str(output_file),
                **info
            })
            
            # Visualización opcional
            if visualize and stats['total'] <= 10:  # Solo primeros 10
                visualize_segmentation(keypoints, filtered_kpts, info, 
                                     output_path / 'visualizations' / f"{relative_path.stem}.png")
        
        except Exception as e:
            logger.error(f"Error procesando {kpt_file}: {e}")
            continue
    
    # Guardar metadata
    metadata_file = output_path / 'segmentation_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump({
            'statistics': stats,
            'files': metadata
        }, f, indent=2)
    
    # Mostrar resumen
    logger.info("=" * 60)
    logger.info("RESUMEN DE FILTRADO")
    logger.info("=" * 60)
    logger.info(f"Archivos procesados: {stats['total']}")
    logger.info(f"Frames originales totales: {stats['total_original_frames']}")
    logger.info(f"Frames filtrados totales: {stats['total_filtered_frames']}")
    logger.info(f"Reducción: {(1 - stats['total_filtered_frames']/stats['total_original_frames'])*100:.1f}%")
    logger.info(f"Metadata guardada en: {metadata_file}")


def visualize_segmentation(original_kpts, filtered_kpts, info, output_file):
    """Visualiza el proceso de segmentación"""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    segmenter = KeypointActionSegmenter()
    scores = segmenter.calculate_action_score(original_kpts)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Action scores
    axes[0].plot(scores, label='Action Score', linewidth=2)
    axes[0].axvline(info['start_idx'], color='g', linestyle='--', label='Start')
    axes[0].axvline(info['end_idx'], color='r', linestyle='--', label='End')
    axes[0].axhspan(0, 1, xmin=info['start_idx']/len(scores), 
                    xmax=info['end_idx']/len(scores), alpha=0.2, color='green')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Action Score')
    axes[0].set_title(f'Segmentación de Acción (Mean: {info["mean_score"]:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Comparación de longitud
    categories = ['Original', 'Filtrado']
    frames = [info['original_frames'], info['filtered_frames']]
    colors = ['blue', 'green']
    axes[1].bar(categories, frames, color=colors, alpha=0.7)
    axes[1].set_ylabel('Número de Frames')
    axes[1].set_title(f'Reducción: {(1 - info["filtered_frames"]/info["original_frames"])*100:.1f}%')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(frames):
        axes[1].text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Filtra keypoints para quedarse solo con segmentos de acción útiles"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directorio con keypoints crudos (.npy files)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directorio para keypoints filtrados'
    )
    parser.add_argument(
        '--min_length',
        type=int,
        default=24,
        help='Longitud mínima de segmento (default: 24 frames)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generar visualizaciones de los primeros 10 archivos'
    )
    
    args = parser.parse_args()
    
    process_keypoints_directory(
        args.input_dir,
        args.output_dir,
        visualize=args.visualize,
        min_length=args.min_length
    )


if __name__ == '__main__':
    main()
