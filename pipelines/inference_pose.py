import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PoseOnlyTemporalModel(nn.Module):
    """Modelo temporal que usa solo pose features (128 dims)"""
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_classes: int = 2286,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Clasificador
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        from torch.nn.utils.rnn import pack_padded_sequence
        
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed)
        
        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        hidden_cat = self.dropout(hidden_cat)
        logits = self.fc(hidden_cat)
        
        return logits


class PoseOnlyInference:
    """Sistema de inferencia usando solo pose features"""
    
    def __init__(
        self,
        model_checkpoint: str,
        metadata_path: str,
        device: Optional[str] = None
    ):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        logger.info(f"Usando device: {self.device}")
        
        # Cargar metadata
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            metadata = data.get('videos', data) if isinstance(data, dict) else data
        
        # Extraer class_names
        self.class_names = sorted(list(set(item['class_name'] for item in metadata)))
        self.num_classes = len(self.class_names)
        logger.info(f"Cargadas {self.num_classes} clases")
        
        # Cargar modelo
        logger.info(f"Cargando modelo desde {model_checkpoint}")
        checkpoint = torch.load(model_checkpoint, map_location=self.device)
        
        # Crear modelo
        self.model = PoseOnlyTemporalModel(
            input_dim=128,
            hidden_dim=checkpoint.get('hidden_dim', 512),
            num_layers=checkpoint.get('num_layers', 2),
            num_classes=self.num_classes,
            dropout=checkpoint.get('dropout', 0.5)
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        logger.info(f"Modelo cargado (epoch {checkpoint.get('epoch', 'N/A')}, "
                   f"acc {checkpoint.get('train_acc', 'N/A'):.2f}%)")
        
        # Inicializar MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False
        )
    
    def extract_pose_features(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extrae pose features de un frame usando MediaPipe
        
        Returns:
            np.ndarray de shape (128,) o None si no se detecta pose
        """
        # MediaPipe espera RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)
        
        features = []
        
        # Pose landmarks (33 landmarks * 4 coords = 132 valores)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            features.extend([0.0] * 132)
        
        # Left hand (21 landmarks * 3 coords = 63 valores)
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * 63)
        
        # Right hand (21 landmarks * 3 coords = 63 valores)
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * 63)
        
        # Total: 132 + 63 + 63 = 258 valores
        # Reducir a 128 dims usando solo las partes más importantes
        pose_array = np.array(features[:132], dtype=np.float32)  # Solo pose
        hands_array = np.array(features[132:], dtype=np.float32)  # Manos
        
        # Tomar submuestreo para llegar a 128
        # Pose: 33 landmarks × 4 = 132 → tomar 64
        pose_reduced = pose_array[::2][:64]  # Submuestreo
        
        # Hands: 126 valores → tomar 64
        hands_reduced = hands_array[::2][:64]
        
        combined = np.concatenate([pose_reduced, hands_reduced])[:128]
        
        # Pad si es necesario
        if len(combined) < 128:
            combined = np.pad(combined, (0, 128 - len(combined)), mode='constant')
        
        return combined
    
    def process_video(
        self,
        video_path: str,
        confidence_threshold: float = 0.05,
        no_segmentation: bool = False
    ) -> List[Dict]:
        """Procesa un video y devuelve predicciones"""
        
        logger.info(f"Procesando video: {video_path}")
        
        # Extraer frames y pose features
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        pose_features_list = []
        frame_idx = 0
        
        pbar = tqdm(total=total_frames, desc="Extrayendo pose features")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            pose_feat = self.extract_pose_features(frame)
            if pose_feat is not None:
                pose_features_list.append(pose_feat)
            
            frame_idx += 1
            pbar.update(1)
        
        cap.release()
        pbar.close()
        
        logger.info(f"Frames procesados: {len(pose_features_list)}")
        
        if len(pose_features_list) == 0:
            logger.warning("No se pudieron extraer pose features del video")
            return []
        
        # Convertir a tensor
        pose_features = np.array(pose_features_list, dtype=np.float32)  # (T, 128)
        features_tensor = torch.from_numpy(pose_features).unsqueeze(0).to(self.device)  # (1, T, 128)
        lengths = torch.tensor([len(pose_features_list)], dtype=torch.long).to(self.device)
        
        # Predicción
        with torch.no_grad():
            logits = self.model(features_tensor, lengths)  # (1, num_classes)
            probs = torch.softmax(logits, dim=1)
            top5_probs, top5_indices = torch.topk(probs, k=5, dim=1)
        
        # Procesar resultados
        results = []
        for i in range(5):
            class_idx = top5_indices[0, i].item()
            confidence = top5_probs[0, i].item()
            class_name = self.class_names[class_idx]
            
            results.append({
                'rank': i + 1,
                'class_name': class_name,
                'confidence': confidence,
                'percentage': confidence * 100
            })
            
            logger.info(f"  Top-{i+1}: {class_name} ({confidence*100:.2f}%)")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Inferencia usando solo pose features')
    
    parser.add_argument('--video', type=str, required=True,
                       help='Ruta al video de entrada')
    parser.add_argument('--model', type=str, required=True,
                       help='Ruta al checkpoint del modelo')
    parser.add_argument('--metadata', type=str, required=True,
                       help='Ruta al archivo metadata JSON')
    parser.add_argument('--confidence', type=float, default=0.05,
                       help='Threshold de confianza mínima')
    parser.add_argument('--output', type=str, default=None,
                       help='Ruta para guardar resultados JSON')
    
    args = parser.parse_args()
    
    # Crear sistema de inferencia
    inference = PoseOnlyInference(
        model_checkpoint=args.model,
        metadata_path=args.metadata
    )
    
    # Procesar video
    results = inference.process_video(
        video_path=args.video,
        confidence_threshold=args.confidence
    )
    
    # Guardar resultados
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Resultados guardados en {args.output}")
    
    logger.info("Procesamiento completado!")


if __name__ == '__main__':
    main()
