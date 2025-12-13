"""
Script de inferencia en tiempo real CORREGIDO
CRÍTICO: Usa los mismos extractores que en entrenamiento
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
from typing import Optional, Tuple, List
import argparse

from pipelines.models_temporal import TemporalLSTMClassifier
from config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ASLRealtimeInference:
    """Sistema de inferencia en tiempo real para ASL"""
    
    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        visual_extractor_path: Path,  # ← IMPORTANTE
        pose_extractor_path: Path, 
        device: str = "cuda",
        buffer_size: int = 24,
        confidence_threshold: float = 0.3,
        smoothing_window: int = 3
    ):
        """
        IMPORTANTE: Este script NO usa extractores separados
        Carga features PRE-FUSIONADAS directamente desde la cámara/video
        procesando con el MISMO pipeline que en entrenamiento
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.buffer_size = buffer_size
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        
        # Cargar metadata
        self.class_to_gloss = self._load_class_mapping(metadata_path)
        self.num_classes = len(self.class_to_gloss)
        
        logger.info(f"Cargadas {self.num_classes} clases")
        
        self.visual_extractor = torch.load(
            visual_extractor_path, 
            map_location=device
        )
        
        self.pose_extractor = torch.load(
            pose_extractor_path,
            map_location=device
        )
        
        logger.info(f"Cargando modelo desde: {model_path}")
        self.model = TemporalLSTMClassifier(
            input_dim=1152,  # Features PRE-FUSIONADAS
            hidden_dim=512,
            num_layers=2,
            num_classes=self.num_classes,
            dropout=0.3,
            bidirectional=True,
            use_attention=True
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        logger.info("✓ Modelo cargado (SOLO LSTM + Classifier)")
        logger.info("⚠ NOTA: Este modelo espera features PRE-FUSIONADAS (1152D)")
        logger.info("         Necesitas PRIMERO extraer features con los extractores ENTRENADOS")
        
        # Inicializar MediaPipe para keypoints
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
        
        # Buffers
        self.feature_buffer = deque(maxlen=buffer_size)  # Buffer de FEATURES (1152D)
        self.frame_buffer = deque(maxlen=buffer_size)    # Para visualización
        self.prediction_buffer = deque(maxlen=smoothing_window)
        
        logger.info("✓ Sistema de inferencia inicializado")
        logger.info("\n" + "="*80)
        logger.info("⚠ ADVERTENCIA IMPORTANTE:")
        logger.info("="*80)
        logger.info("Este script NO puede funcionar directamente con video/cámara")
        logger.info("porque NO tiene acceso a los extractores ENTRENADOS.")
        logger.info("\nPara usar inferencia en tiempo real, necesitas:")
        logger.info("1. Guardar los extractores (ResNet101 + MLP) durante el entrenamiento")
        logger.info("2. Cargarlos aquí para extraer features de la misma manera")
        logger.info("\nAlternativamente, puedes:")
        logger.info("- Usar un video del dataset de prueba para verificar el modelo")
        logger.info("- Pre-procesar videos y guardar features antes de predecir")
        logger.info("="*80 + "\n")
    
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
    
    def predict_from_fused_features(
        self, 
        fused_features: torch.Tensor
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """
        Predice desde features PRE-FUSIONADAS
        
        Args:
            fused_features: (T, 1152) tensor de features fusionadas
        
        Returns:
            predicted_gloss, confidence, top5
        """
        if fused_features.dim() == 2:
            fused_features = fused_features.unsqueeze(0)  # (1, T, 1152)
        
        with torch.no_grad():
            logits = self.model(fused_features)
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


def test_with_precomputed_features(
    model_path: Path,
    metadata_path: Path,
    features_dir: Path,
    device: str = "cuda",
    num_samples: int = 10
):
    """
    Prueba el modelo con features pre-computadas del dataset
    Esto verifica que el modelo funciona correctamente
    """
    
    logger.info("="*80)
    logger.info("PRUEBA CON FEATURES PRE-COMPUTADAS")
    logger.info("="*80)
    
    system = ASLRealtimeInference(
        model_path=model_path,
        metadata_path=metadata_path,
        device=device
    )
    
    # Buscar archivos de features
    feature_files = list(features_dir.glob("*_fused.npy"))
    
    if len(feature_files) == 0:
        logger.error(f"No se encontraron features en {features_dir}")
        return
    
    logger.info(f"\nEncontrados {len(feature_files)} archivos de features")
    logger.info(f"Probando con {num_samples} muestras aleatorias...\n")
    
    # Tomar muestra aleatoria
    np.random.seed(42)
    sample_files = np.random.choice(feature_files, min(num_samples, len(feature_files)), replace=False)
    
    results = []
    
    for i, feature_file in enumerate(sample_files, 1):
        logger.info(f"[{i}/{num_samples}] Procesando: {feature_file.name}")
        
        # Cargar features
        features = np.load(feature_file).astype(np.float32)
        features_tensor = torch.from_numpy(features).to(system.device)
        
        logger.info(f"  Features shape: {features.shape}")
        logger.info(f"  Features range: [{features.min():.4f}, {features.max():.4f}]")
        
        # Predecir
        predicted_gloss, confidence, top5 = system.predict_from_fused_features(features_tensor)
        
        logger.info(f"  ✓ Predicción: {predicted_gloss} ({confidence:.2%})")
        logger.info(f"    Top-3:")
        for j, (gloss, prob) in enumerate(top5[:3], 1):
            logger.info(f"      {j}. {gloss}: {prob:.2%}")
        
        results.append({
            'file': feature_file.name,
            'predicted': predicted_gloss,
            'confidence': confidence,
            'top5': top5
        })
        logger.info("")
    
    # Estadísticas
    confidences = [r['confidence'] for r in results]
    
    logger.info("="*80)
    logger.info("ESTADÍSTICAS:")
    logger.info("="*80)
    logger.info(f"Confianza promedio: {np.mean(confidences):.2%}")
    logger.info(f"Confianza mediana: {np.median(confidences):.2%}")
    logger.info(f"Confianza mínima: {np.min(confidences):.2%}")
    logger.info(f"Confianza máxima: {np.max(confidences):.2%}")
    
    high_conf = sum(1 for c in confidences if c > 0.5)
    logger.info(f"\nPredicciones con >50% confianza: {high_conf}/{len(results)} ({high_conf/len(results)*100:.1f}%)")
    
    if np.mean(confidences) < 0.1:
        logger.error("\n✗ PROBLEMA: Confianza promedio muy baja")
        logger.error("  El modelo NO está funcionando correctamente")
        logger.error("  Posibles causas:")
        logger.error("  1. Modelo no convergió en entrenamiento")
        logger.error("  2. Features corruptas o mal procesadas")
        logger.error("  3. Mismatch entre arquitectura y pesos cargados")
    elif np.mean(confidences) < 0.3:
        logger.warning("\n⚠ Confianza promedio baja pero funcional")
        logger.warning("  El modelo puede necesitar más entrenamiento")
    else:
        logger.info("\n✓ Modelo funciona correctamente con features pre-computadas")
    
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="Inferencia ASL (REQUIERE FEATURES PRE-FUSIONADAS)")
    
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Ruta al modelo entrenado"
    )
    parser.add_argument(
        "--metadata_path",
        type=Path,
        required=True,
        help="Ruta a dataset_metadata.json"
    )
    parser.add_argument(
        "--features_dir",
        type=Path,
        default=None,
        help="Directorio con features pre-computadas para testing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"]
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Número de muestras para testing"
    )
    
    args = parser.parse_args()
    
    if not args.model_path.exists():
        logger.error(f"Modelo no encontrado: {args.model_path}")
        return
    
    if not args.metadata_path.exists():
        logger.error(f"Metadata no encontrado: {args.metadata_path}")
        return
    
    if args.features_dir and args.features_dir.exists():
        # Modo testing con features pre-computadas
        test_with_precomputed_features(
            model_path=args.model_path,
            metadata_path=args.metadata_path,
            features_dir=args.features_dir,
            device=args.device,
            num_samples=args.num_samples
        )
    else:
        logger.error("\n" + "="*80)
        logger.error("ERROR: Este script NO puede funcionar sin features pre-computadas")
        logger.error("="*80)
        logger.error("\nProporciona --features_dir para probar con el dataset")
        logger.error("\nEjemplo:")
        logger.error("  python inference_realtime.py \\")
        logger.error("    --model_path checkpoints/temporal/best_model.pt \\")
        logger.error("    --metadata_path data/dataset_metadata.json \\")
        logger.error("    --features_dir data/features_fused \\")
        logger.error("    --num_samples 20")


if __name__ == "__main__":
    main()