"""
Pipeline de videos completos para reconocimiento de ASL.

Este pipeline procesa videos completos sin crear clips artificiales.
Caracteristicas:
- Skip del inicio del video (contenido sin valor)
- Sampling uniforme hasta max_frames
- Secuencias de longitud variable con padding dinamico
- Class balancing con weighted loss o focal loss
- Inferencia coherente con entrenamiento

Modulos:
- config_video: Configuracion del pipeline
- extract_video_features: Extraccion de features de videos
- dataset_video: Dataset y DataLoaders
- models_video: Modelos temporales (LSTM, Transformer)
- train_video: Entrenamiento con class balancing
- evaluate_video: Evaluacion y metricas
- inference_video: Inferencia en videos y tiempo real
"""

from pipelines_video.config import config, VideoConfig
from pipelines_video.models import (
    VideoLSTMClassifier,
    VideoTransformerClassifier,
    FocalLoss,
    get_video_model,
    get_loss_function
)
from pipelines_video.dataset import (
    VideoFeaturesDataset,
    video_collate_fn,
    create_video_dataloaders,
    compute_class_weights
)
from pipelines_video.inference import (
    VideoInferenceEngine,
    RealtimeInferenceEngine
)

__all__ = [
    # Config
    'config',
    'VideoConfig',
    
    # Models
    'VideoLSTMClassifier',
    'VideoTransformerClassifier',
    'FocalLoss',
    'get_video_model',
    'get_loss_function',
    
    # Dataset
    'VideoFeaturesDataset',
    'video_collate_fn',
    'create_video_dataloaders',
    'compute_class_weights',
    
    # Inference
    'VideoInferenceEngine',
    'RealtimeInferenceEngine'
]
