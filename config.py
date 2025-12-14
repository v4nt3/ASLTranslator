"""
Configuración centralizada para el pipeline ASL Multimodal
Todas las rutas, hiperparámetros y configuraciones se definen aquí
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

# ==================== RUTAS BASE ====================
PROJECT_ROOT = Path(__file__).parent.absolute()

DATA_ROOT = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Crear directorios si no existen
for dir_path in [DATA_ROOT, MODELS_DIR, LOGS_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==================== RUTAS DE DATOS ====================
@dataclass
class DataPaths:
    """Rutas específicas para los datos"""
    raw_videos: Path = DATA_ROOT / "dataset"
    extracted_frames: Path = DATA_ROOT / "extracted_frames"
    keypoints: Path = DATA_ROOT / "keypoints"
    clips: Path = DATA_ROOT / "clips"
    features_visual: Path = DATA_ROOT / "features_visual"
    features_pose: Path = DATA_ROOT / "features_pose"
    features_fused: Path = DATA_ROOT / "features_fused"
    dataset_csv: Path = DATA_ROOT / "dataset.csv"
    dataset_split_csv: Path = DATA_ROOT / "dataset_split.csv"
    dataset_meta: Path = DATA_ROOT / "dataset_metadata.json"  # Agregando ruta a dataset_meta.json para carga directa sin CSV
    
    def __post_init__(self):
        for path in [
            self.raw_videos, self.extracted_frames, self.keypoints, self.clips,
            self.features_visual, self.features_pose, self.features_fused
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelPaths:
    """Rutas para modelos y checkpoints"""
    best_macro: Path = CHECKPOINTS_DIR / "best_macro.pt"
    last_checkpoint: Path = CHECKPOINTS_DIR / "last.pt"
    final_model: Path = MODELS_DIR / "final_model.pt"
    onnx_model: Path = MODELS_DIR / "model.onnx"
    temporal_checkpoints: Path = CHECKPOINTS_DIR / "temporal"
    
    def __post_init__(self):
        self.temporal_checkpoints.mkdir(parents=True, exist_ok=True)



# ==================== CONFIGURACIÓN DE DATOS ====================
@dataclass
class DataConfig:
    """Configuración del preprocessing de datos"""
    # Video
    fps: int = 60
    frame_height: int = 224
    frame_width: int = 224
    num_frames_per_clip: int = 24
    
    # Clips por video según clase
    clips_distribution: Dict[str, int] = None
    
    # Keypoints
    num_pose_points: int = 33
    num_hand_points: int = 42  # 21 + 21
    num_face_points: int = 468  # reducible a 64 con PCA
    use_face_keypoints: bool = False  # Cambiar a True si se desea usar
    face_pca_dim: int = 64
    
    # Augmentation
    color_jitter_brightness: float = 0.1
    color_jitter_contrast: float = 0.1
    color_jitter_saturation: float = 0.1
    gaussian_blur_kernel: int = 3
    gaussian_blur_sigma: tuple = (0.1, 2.0)
    random_erase_p: float = 0.0
    random_erase_scale: tuple = (0.02, 0.33)
    
    # Pose Augmentation
    pose_jitter_std: float = 0.01  # 1-2 pixels normalizados
    pose_scale_range: tuple = (0.95, 1.05)
    pose_rotation_degrees: float = 3.0
    
    # Temporal Augmentation
    speed_jitter_range: tuple = (0.9, 1.1)
    frame_drop_p: float = 0.1
    temporal_shift_range: int = 2
    
    temporal_augment_prob: float = 0.7  # Probabilidad de aplicar augmentation
    feature_noise_std: float = 0.005  # Ruido en features
    feature_scale_range: tuple = (0.95, 1.05)  # Escalado de features
    
    # Split
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    
    visual_feature_dim: int = 512
    pose_feature_dim: int = 128
    fused_feature_dim: int = 640  # 512 + 128
    
    def __post_init__(self):
        if self.clips_distribution is None:
            self.clips_distribution = {
                "low": 3,      # <=27 videos
                "medium": 2,   # 28-40 videos
                "high": 1      # >40 videos
            }


# ==================== CONFIGURACIÓN DEL MODELO ====================
@dataclass
class ModelConfig:
    """Configuración del modelo multimodal"""
    # Arquitectura
    backbone_name: str = "efficientnet_b3"  # o "resnet34"
    embedding_dim: int = 512
    fusion_type: str = "temporal_transformer"  # concat, sum, temporal_transformer
    
    # Backbone visual
    pretrained: bool = True
    freeze_backbone: bool = False
    
    # Backbone de pose/keypoints
    pose_hidden_dims: List[int] = None
    pose_dropout: float = 0.3

    # Keypoints
    num_pose_points: int = 33
    num_hand_points: int = 42  # 21 + 21
    num_face_points: int = 468  # reducible a 64 con PCA
    use_face_keypoints: bool = False  # Cambiar a True si se desea usar
    face_pca_dim: int = 64

    keypoints_dim: int = 300  # (num_pose_points + num_hand_points) * 4
    
    # Transformer de fusión (si es necesario)
    num_transformer_layers: int = 2
    num_attention_heads: int = 8
    transformer_dropout: float = 0.1
    
    num_attention_heads_lstm: int = 8  # Heads para multihead attention en LSTM
    
    # Clasificador final
    classifier_hidden_dim: int = 1024
    classifier_dropout: float = 0.2
    num_classes: int = 2286
    
    def __post_init__(self):
        if self.pose_hidden_dims is None:
            self.pose_hidden_dims = [256, 512]


# ==================== CONFIGURACIÓN DE ENTRENAMIENTO ====================
@dataclass
class TrainingConfig:
    """Configuración del entrenamiento CORREGIDA"""
    # Básico
    num_epochs: int = 100  # Reducido de 200
    batch_size: int = 64   # Reducido de 128 para mejor convergencia
    num_workers: int = 10
    pin_memory: bool = False
    
    # Optimización
    optimizer: str = "adam"
    learning_rate: float = 3e-4  # CAMBIADO: Más conservador (antes 1e-3)
    weight_decay: float = 1e-5   # CAMBIADO: Menos regularización (antes 1e-4)
    momentum: float = 0.9
    
    # Loss
    loss_fn: str = "cross_entropy"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0  # CAMBIADO: Desactivado (antes 0.1)
    
    # Scheduler - CAMBIADO a ReduceLROnPlateau
    scheduler_type: str = "plateau"  # CAMBIADO: De "onecycle" a "plateau"
    scheduler_patience: int = 5      # NUEVO
    scheduler_factor: float = 0.5    # NUEVO
    scheduler_min_lr: float = 1e-6   # NUEVO
    
    # Los siguientes solo se usan si scheduler_type == "onecycle"
    lr_div_factor: float = 25.0
    pct_start: float = 0.3
    
    # AMP (Automatic Mixed Precision)
    use_amp: bool = True
    scaler_init_scale: float = 65536.0
    
    # Checkpointing
    save_interval: int = 5
    save_best_only: bool = True
    
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 15  # CAMBIADO: Más paciencia (antes 10)
    early_stopping_metric: str = "val_accuracy_macro"
    
    # Device
    device: str = "cuda"
    seed: int = 42
    
    # Configuraciones de modelo temporal
    model_type: str = "lstm"
    use_attention: bool = True
    use_augmentation: bool = False
    run_final_evaluation: bool = True
    delete_original_clips: bool = False
    
    # NUEVO: Parámetros de modelo
    model_hidden_dim: int = 512
    model_num_layers: int = 2
    model_dropout: float = 0.1  # CAMBIADO: Menos dropout (antes 0.3)
    model_bidirectional: bool = True

# ==================== CONFIGURACIÓN DE EVALUACIÓN ====================
@dataclass
class EvalConfig:
    """Configuración de evaluación"""
    batch_size: int = 64
    num_workers: int = 10
    compute_confusion_matrix: bool = True
    compute_per_class_metrics: bool = True
    top_k_accuracy: List[int] = None
    
    
    def __post_init__(self):
        if self.top_k_accuracy is None:
            self.top_k_accuracy = [1, 3, 5]

@dataclass
class OutputPaths:
    """Rutas para outputs y reportes"""
    evaluation_report: Path = OUTPUTS_DIR
    confusion_matrix: Path = OUTPUTS_DIR / "confusion_matrix.png"
    training_history: Path = OUTPUTS_DIR / "training_history.csv"
    tensorboard_logs: Path = LOGS_DIR / "tensorboard"


# ==================== CLASE MAESTRA DE CONFIGURACIÓN ====================
@dataclass
class config:
    """Clase que contiene todas las configuraciones"""
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    evaluation: EvalConfig = None
    data_paths: DataPaths = None
    model_paths: ModelPaths = None
    output_paths: OutputPaths = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.evaluation is None:
            self.evaluation = EvalConfig()
        if self.data_paths is None:
            self.data_paths = DataPaths()
        if self.model_paths is None:
            self.model_paths = ModelPaths()
        if self.output_paths is None:
            self.output_paths = OutputPaths()


# ==================== INSTANCIA GLOBAL ====================
# Esta es la configuración que se usa en todo el proyecto
config = config()
