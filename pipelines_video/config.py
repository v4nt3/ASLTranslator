"""
Configuracion para el pipeline de videos completos (sin clips)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class VideoProcessingConfig:
    """Configuracion para procesamiento de videos"""
    # Porcentaje del inicio del video a ignorar (contenido sin valor)
    skip_start_percent: float = 0.10  # 10%
    
    # Numero maximo de frames a extraer por video
    max_frames: int = 45  # Limite de frames (30-45 recomendado)
    
    # Dimensiones de frame
    frame_height: int = 224
    frame_width: int = 224
    
    # Stride para sampling de frames (1 = todos, 2 = cada 2, etc.)
    frame_stride: int = 1


@dataclass
class FeatureConfig:
    """Configuracion de features"""
    visual_dim: int = 1024      # ResNet101 output
    pose_dim: int = 128         # MLP pose output  
    keypoints_dim: int = 300    # MediaPipe keypoints (75 * 4)
    fused_dim: int = 1152       # visual_dim + pose_dim


@dataclass
class ModelConfig:
    """Configuracion del modelo temporal"""
    num_classes: int = 2286
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    use_attention: bool = True


@dataclass
class TrainingConfig:
    """Configuracion de entrenamiento"""
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    num_workers: int = 0
    device: str = "cuda"
    use_amp: bool = True
    
    # Class balancing
    use_class_weights: bool = True
    focal_loss_gamma: float = 2.0  # 0 = CrossEntropy, >0 = FocalLoss
    
    # Scheduler
    scheduler_type: str = "plateau"  # "plateau", "onecycle", "cosine"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    
    # Early stopping
    early_stopping_patience: int = 15
    
    # Label smoothing
    label_smoothing: float = 0.0


@dataclass
class DataPathsConfig:
    """Rutas de datos"""
    raw_videos: Path = field(default_factory=lambda: Path("data/dataset"))
    dataset_meta: Path = field(default_factory=lambda: Path("data/dataset/dataset_meta.json"))
    
    # Features por video completo (no clips)
    features_visual: Path = field(default_factory=lambda: Path("data/features_video/visual"))
    features_pose: Path = field(default_factory=lambda: Path("data/features_video/pose"))
    features_fused: Path = field(default_factory=lambda: Path("data/features_video/fused"))
    
    # Extractores
    visual_extractor: Path = field(default_factory=lambda: Path("models/extractors/visual_extractor_full.pt"))
    pose_extractor: Path = field(default_factory=lambda: Path("models/extractors/pose_extractor_full.pt"))


@dataclass
class ModelPathsConfig:
    """Rutas de modelos"""
    checkpoints: Path = field(default_factory=lambda: Path("models/checkpoints_video"))
    best_model: Path = field(default_factory=lambda: Path("models/checkpoints_video/best_model.pt"))


@dataclass
class VideoConfig:
    """Configuracion completa del pipeline de videos"""
    video: VideoProcessingConfig = field(default_factory=VideoProcessingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data_paths: DataPathsConfig = field(default_factory=DataPathsConfig)
    model_paths: ModelPathsConfig = field(default_factory=ModelPathsConfig)


# Instancia global de configuracion
config = VideoConfig()


def load_config_from_json(json_path: Path) -> VideoConfig:
    """
    Carga configuracion desde archivo JSON.
    Solo sobrescribe valores presentes en el JSON.
    """
    import json
    
    with open(json_path, 'r') as f:
        user_config = json.load(f)
    
    cfg = VideoConfig()
    
    # Video processing
    if 'video' in user_config:
        for key, value in user_config['video'].items():
            if hasattr(cfg.video, key):
                setattr(cfg.video, key, value)
    
    # Features
    if 'features' in user_config:
        for key, value in user_config['features'].items():
            if hasattr(cfg.features, key):
                setattr(cfg.features, key, value)
    
    # Model
    if 'model' in user_config:
        for key, value in user_config['model'].items():
            if hasattr(cfg.model, key):
                setattr(cfg.model, key, value)
    
    # Training
    if 'training' in user_config:
        for key, value in user_config['training'].items():
            if hasattr(cfg.training, key):
                setattr(cfg.training, key, value)
    
    # Data paths
    if 'data_paths' in user_config:
        for key, value in user_config['data_paths'].items():
            if hasattr(cfg.data_paths, key):
                setattr(cfg.data_paths, key, Path(value))
    
    # Model paths  
    if 'model_paths' in user_config:
        for key, value in user_config['model_paths'].items():
            if hasattr(cfg.model_paths, key):
                setattr(cfg.model_paths, key, Path(value))
    
    return cfg


def update_global_config(json_path: Path):
    """Actualiza la configuracion global desde archivo JSON"""
    global config
    config = load_config_from_json(json_path)
