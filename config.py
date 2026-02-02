"""
Configuracion optimizada para REGULARIZACION y generalizacion
Cambios clave para evitar overfitting despues de corregir data leakage
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ==================== RUTAS BASE ====================
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_ROOT = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for dir_path in [DATA_ROOT, MODELS_DIR, LOGS_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataPaths:
    """Rutas especificas para los datos"""
    raw_videos: Path = DATA_ROOT / "dataset"
    extracted_frames: Path = DATA_ROOT / "extracted_frames"
    keypoints: Path = DATA_ROOT / "keypoints"
    features_visual: Path = DATA_ROOT / "features_visual"
    features_pose: Path = DATA_ROOT / "features_pose"
    features_fused: Path = DATA_ROOT / "features_fused"
    dataset_meta: Path = DATA_ROOT / "dataset_meta.json"
    
    def __post_init__(self):
        for path in [
            self.raw_videos, self.extracted_frames, self.keypoints,
            self.features_visual, self.features_pose, self.features_fused
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelPaths:
    """Rutas para modelos y checkpoints"""
    final_model: Path = MODELS_DIR / "final_model.pt"
    temporal_checkpoints: Path = CHECKPOINTS_DIR / "temporal"
    
    def __post_init__(self):
        self.temporal_checkpoints.mkdir(parents=True, exist_ok=True)


# ==================== CONFIGURACION DE DATOS ====================
@dataclass
class DataConfig:
    """Configuracion del preprocessing de datos"""
    # Video
    fps: int = 60
    frame_height: int = 224
    frame_width: int = 224
    num_frames_per_clip: int = 24
    
    # Keypoints
    num_pose_points: int = 33
    num_hand_points: int = 42
    num_face_points: int = 468
    use_face_keypoints: bool = False
    face_pca_dim: int = 64
    
    # ==================== AUGMENTATION MEJORADO ====================
    # Visual Augmentation (mas agresivo)
    color_jitter_brightness: float = 0.2  # AUMENTADO de 0.1
    color_jitter_contrast: float = 0.2    # AUMENTADO de 0.1
    color_jitter_saturation: float = 0.15 # AUMENTADO de 0.1
    gaussian_blur_kernel: int = 3
    gaussian_blur_sigma: tuple = (0.1, 2.0)
    random_erase_p: float = 0.15          # NUEVO: Random erasing
    random_erase_scale: tuple = (0.02, 0.2)
    
    # Pose Augmentation (mas agresivo)
    pose_jitter_std: float = 0.02         # AUMENTADO de 0.01
    pose_scale_range: tuple = (0.9, 1.1)  # AUMENTADO de (0.95, 1.05)
    pose_rotation_degrees: float = 5.0    # AUMENTADO de 3.0
    
    # Temporal Augmentation
    speed_jitter_range: tuple = (0.85, 1.15)  # AUMENTADO
    frame_drop_p: float = 0.15                 # AUMENTADO de 0.1
    temporal_shift_range: int = 3              # AUMENTADO de 2
    
    # Feature-level Augmentation (CLAVE para regularizacion)
    temporal_augment_prob: float = 0.8         # AUMENTADO de 0.7
    feature_noise_std: float = 0.01            # AUMENTADO de 0.005
    feature_scale_range: tuple = (0.9, 1.1)    # AUMENTADO
    feature_dropout_prob: float = 0.1          # NUEVO: Dropout a nivel de feature
    
    # Mixup y CutMix (NUEVO)
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_cutmix: bool = False  # No recomendado para secuencias temporales
    
    # Split (por VIDEO, no por clip)
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    
    # Dimensiones
    visual_feature_dim: int = 1024
    pose_feature_dim: int = 128
    fused_feature_dim: int = 1152


# ==================== CONFIGURACION DEL MODELO ====================
@dataclass
class ModelConfig:
    """Configuracion del modelo con REGULARIZACION mejorada"""
    
    # Backbone visual
    pretrained: bool = True
    freeze_backbone: bool = False
    
    # Backbone de pose/keypoints - MAS REGULARIZADO
    pose_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])  # Reducido de [256, 512]
    pose_dropout: float = 0.4  # AUMENTADO de 0.3
    
    # Keypoints
    num_pose_points: int = 33
    num_hand_points: int = 42
    num_face_points: int = 468
    use_face_keypoints: bool = False
    face_pca_dim: int = 64
    keypoints_dim: int = 300
    
    # Clasificador final - MAS REGULARIZADO
    classifier_hidden_dim: int = 512      # REDUCIDO de 1024
    classifier_dropout: float = 0.4       # AUMENTADO de 0.2
    num_classes: int = 2286


# ==================== CONFIGURACION DE ENTRENAMIENTO ====================
@dataclass
class TrainingConfig:
    """Configuracion de entrenamiento OPTIMIZADA para regularizacion"""
    
    # Basico
    num_epochs: int = 150                  # Mas epochs con early stopping
    batch_size: int = 32                   # REDUCIDO de 64 (mejor regularizacion)
    num_workers: int = 4
    pin_memory: bool = True
    gradient_accumulation_steps: int = 2   # NUEVO: Simula batch_size=64
    
    # ==================== OPTIMIZACION ====================
    optimizer: str = "adam"               # CAMBIADO de "adam" a "adamw"
    learning_rate: float = 1e-4            # REDUCIDO de 3e-4
    weight_decay: float = 0.05             # AUMENTADO SIGNIFICATIVAMENTE de 1e-5
    momentum: float = 0.9
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    
    # Gradient Clipping (NUEVO)
    max_grad_norm: float = 1.0
    
    # ==================== LOSS ====================
    loss_fn: str = "cross_entropy"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.15          # AUMENTADO de 0.1
    
    # ==================== SCHEDULER ====================
    # Opciones: "cosine_warmup" o "plateau"
    scheduler_type: str = "plateau"  # "cosine_warmup" | "plateau"
    
    # Parametros para Cosine Annealing con Warmup
    warmup_epochs: int = 5
    warmup_lr_init: float = 1e-6
    min_lr: float = 1e-6
    
    # Parametros para ReduceLROnPlateau
    scheduler_patience: int = 8            # Epochs sin mejora antes de reducir LR
    scheduler_factor: float = 0.5          # Factor de reduccion (new_lr = lr * factor)
    scheduler_min_lr: float = 1e-6         # LR minimo
    scheduler_threshold: float = 1e-4      # Threshold para considerar mejora
    scheduler_cooldown: int = 2            # Epochs de cooldown despues de reducir
    
    # AMP
    use_amp: bool = True
    scaler_init_scale: float = 65536.0
    
    # ==================== CHECKPOINTING ====================
    save_interval: int = 10
    save_best_only: bool = True
    save_last_k: int = 3                   # NUEVO: Guardar ultimos K checkpoints
    
    # ==================== EARLY STOPPING ====================
    use_early_stopping: bool = True
    early_stopping_patience: int = 20      # AUMENTADO de 15
    early_stopping_metric: str = "val_loss"  # CAMBIADO: loss es mas estable
    early_stopping_mode: str = "min"
    
    # Device
    device: str = "cuda"
    seed: int = 42
    
    # ==================== MODELO TEMPORAL ====================
    model_type: str = "lstm"
    use_attention: bool = True
    use_augmentation: bool = True          # ACTIVADO
    run_final_evaluation: bool = True
    
    # Arquitectura LSTM - MAS REGULARIZADA
    model_hidden_dim: int = 384            # REDUCIDO de 512
    model_num_layers: int = 2
    model_dropout: float = 0.4             # AUMENTADO de 0.2
    model_bidirectional: bool = True
    
    # ==================== REGULARIZACION ADICIONAL ====================
    # Dropout variacional para LSTM (NUEVO)
    use_variational_dropout: bool = True
    variational_dropout_rate: float = 0.3
    
    # Layer Dropout (NUEVO) - Dropea capas enteras del LSTM
    use_layer_dropout: bool = False
    layer_dropout_rate: float = 0.1
    
    # Stochastic Depth (NUEVO)
    use_stochastic_depth: bool = False
    stochastic_depth_rate: float = 0.1
    
    # Weight Noise (NUEVO)
    use_weight_noise: bool = False
    weight_noise_std: float = 0.01
    
    # ==================== TECNICAS ANTI-OVERFITTING ====================
    # Exponential Moving Average de pesos (NUEVO)
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # R-Drop: Regularizacion por consistencia (NUEVO)
    use_rdrop: bool = False
    rdrop_alpha: float = 0.5


# ==================== CONFIGURACION DE EVALUACION ====================
@dataclass
class EvalConfig:
    """Configuracion de evaluacion"""
    batch_size: int = 64
    num_workers: int = 10
    compute_confusion_matrix: bool = True
    compute_per_class_metrics: bool = True
    top_k_accuracy: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    
    # Test Time Augmentation (NUEVO)
    use_tta: bool = True
    tta_num_augments: int = 5


@dataclass
class OutputPaths:
    """Rutas para outputs y reportes"""
    evaluation_report: Path = OUTPUTS_DIR
    confusion_matrix: Path = OUTPUTS_DIR / "confusion_matrix.png"
    training_history: Path = OUTPUTS_DIR / "training_history.csv"
    tensorboard_logs: Path = LOGS_DIR / "tensorboard"


# ==================== CLASE MAESTRA ====================
@dataclass
class Config:
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
    
    def get_effective_batch_size(self) -> int:
        """Calcula batch size efectivo con gradient accumulation"""
        return self.training.batch_size * self.training.gradient_accumulation_steps
    
    def print_regularization_summary(self):
        """Imprime resumen de tecnicas de regularizacion activas"""
        print("\n" + "="*60)
        print("RESUMEN DE REGULARIZACION")
        print("="*60)
        
        t = self.training
        print(f"\n[Optimizador]")
        print(f"  - Optimizer: {t.optimizer}")
        print(f"  - Weight Decay: {t.weight_decay}")
        print(f"  - Gradient Clipping: {t.max_grad_norm}")
        
        print(f"\n[Scheduler]")
        print(f"  - Tipo: {t.scheduler_type}")
        if t.scheduler_type == "cosine_warmup":
            print(f"  - Warmup epochs: {t.warmup_epochs}")
            print(f"  - Min LR: {t.min_lr}")
        else:
            print(f"  - Patience: {t.scheduler_patience}")
            print(f"  - Factor: {t.scheduler_factor}")
            print(f"  - Min LR: {t.scheduler_min_lr}")
        
        print(f"\n[Dropout]")
        print(f"  - Model Dropout: {t.model_dropout}")
        print(f"  - Variational Dropout: {t.use_variational_dropout} ({t.variational_dropout_rate})")
        print(f"  - Classifier Dropout: {self.model.classifier_dropout}")
        
        print(f"\n[Loss]")
        print(f"  - Label Smoothing: {t.label_smoothing}")
        print(f"  - Loss Function: {t.loss_fn}")
        
        print(f"\n[Data Augmentation]")
        print(f"  - Temporal Augment Prob: {self.data.temporal_augment_prob}")
        print(f"  - Feature Noise: {self.data.feature_noise_std}")
        print(f"  - Mixup: {self.data.use_mixup} (alpha={self.data.mixup_alpha})")
        
        print(f"\n[Tecnicas Avanzadas]")
        print(f"  - EMA: {t.use_ema} (decay={t.ema_decay})")
        print(f"  - R-Drop: {t.use_rdrop}")
        print(f"  - TTA en evaluacion: {self.evaluation.use_tta}")
        
        print(f"\n[Arquitectura]")
        print(f"  - Hidden Dim: {t.model_hidden_dim} (reducido)")
        print(f"  - Batch Size Efectivo: {self.get_effective_batch_size()}")
        print("="*60 + "\n")


# ==================== INSTANCIA GLOBAL ====================
config = Config()


# ==================== PRESETS DE REGULARIZACION ====================
def get_light_regularization() -> Config:
    """Regularizacion ligera - para cuando tienes muchos datos"""
    cfg = Config()
    cfg.training.weight_decay = 0.01
    cfg.training.model_dropout = 0.2
    cfg.training.label_smoothing = 0.1
    cfg.data.temporal_augment_prob = 0.5
    cfg.training.use_ema = False
    return cfg


def get_heavy_regularization() -> Config:
    """Regularizacion pesada - para datasets pequenos o con pocas muestras por clase"""
    cfg = Config()
    cfg.training.weight_decay = 0.1
    cfg.training.model_dropout = 0.5
    cfg.training.label_smoothing = 0.2
    cfg.training.model_hidden_dim = 256
    cfg.data.temporal_augment_prob = 0.9
    cfg.data.mixup_alpha = 0.4
    cfg.training.use_ema = True
    cfg.training.use_rdrop = True
    return cfg


def get_balanced_regularization() -> Config:
    """Regularizacion balanceada - punto de partida recomendado"""
    return Config()  # Los valores por defecto ya estan balanceados
