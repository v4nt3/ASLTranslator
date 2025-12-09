"""
Configuración optimizada para mejorar resultados manteniendo arquitectura simple
Basado en los mejores resultados: input_dim=1152, clasificador simple
"""

class Config:
    # ============================================================
    # ARQUITECTURA (mantener simple - ya validado)
    # ============================================================
    
    # Dimensiones de entrada
    VISUAL_DIM = 1024  # ResNet101
    POSE_DIM = 128     # MLP pose
    INPUT_DIM = 1152   # Total (mejor que 640)
    
    # LSTM
    HIDDEN_DIM = 1024  # Mantener
    NUM_LAYERS = 2     # Mantener
    DROPOUT = 0.3      # Mantener
    BIDIRECTIONAL = True
    USE_ATTENTION = False  # Simple funciona mejor
    
    # Clasificador SIMPLE (el que funciona)
    USE_SIMPLE_CLASSIFIER = True
    CLASSIFIER_DROPOUT = 0.3
    
    # ============================================================
    # OPTIMIZACIONES AVANZADAS PARA MEJORAR RESULTADOS
    # ============================================================
    
    # 1. LABEL SMOOTHING - Reduce overfitting, mejora generalización
    LABEL_SMOOTHING = 0.1  # Suaviza las etiquetas duras
    
    # 2. MIXUP TEMPORAL - Data augmentation para secuencias
    USE_MIXUP = True
    MIXUP_ALPHA = 0.2  # Parámetro de la distribución Beta
    MIXUP_PROB = 0.5   # Probabilidad de aplicar mixup
    
    # 3. STOCHASTIC DEPTH - Dropout en capas LSTM
    USE_STOCHASTIC_DEPTH = True
    STOCHASTIC_DEPTH_PROB = 0.1  # Probabilidad de drop layer
    
    # 4. GRADIENT CLIPPING optimizado
    GRAD_CLIP_NORM = 1.0  # Clip por norma (más estable que por valor)
    
    # 5. LEARNING RATE SCHEDULE mejorado
    # OneCycleLR - mejor que ReduceLROnPlateau para este tipo de modelos
    USE_ONE_CYCLE = True
    INITIAL_LR = 0.001
    MAX_LR = 0.003  # Pico del ciclo
    PCT_START = 0.3  # % del entrenamiento para llegar al max_lr
    
    # Si prefieres ReduceLROnPlateau (backup)
    USE_REDUCE_LR = False
    REDUCE_LR_FACTOR = 0.5
    REDUCE_LR_PATIENCE = 5
    
    # 6. WARMUP - Estabiliza entrenamiento inicial
    USE_WARMUP = True
    WARMUP_EPOCHS = 3
    WARMUP_START_LR = 0.0001
    
    # 7. WEIGHT DECAY optimizado
    WEIGHT_DECAY = 1e-4  # Tu mejor resultado
    
    # 8. OPTIMIZER mejorado
    OPTIMIZER = "adamw"  # AdamW mejor que Adam para regularización
    BETAS = (0.9, 0.999)
    EPS = 1e-8
    
    # 9. GRADIENT ACCUMULATION - Simula batch size más grande
    GRADIENT_ACCUMULATION_STEPS = 2  # Batch efectivo = batch_size * 2
    
    # 10. EMA (Exponential Moving Average) de pesos
    USE_EMA = True
    EMA_DECAY = 0.999  # Promedia pesos del modelo para mejor generalización
    
    # 11. TEST TIME AUGMENTATION (TTA)
    USE_TTA = True
    TTA_AUGMENTATIONS = 5  # Número de augmentations en test
    
    # 12. TEMPORAL AUGMENTATION en entrenamiento
    USE_TEMPORAL_AUG = True
    TEMPORAL_SHIFT_RANGE = 2  # Frames de desplazamiento
    TEMPORAL_SPEED_RANGE = (0.9, 1.1)  # Variación de velocidad
    TEMPORAL_REVERSE_PROB = 0.1  # Probabilidad de reversa temporal
    
    # ============================================================
    # PARÁMETROS DE ENTRENAMIENTO
    # ============================================================
    
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15  # Más paciencia con mejores técnicas
    
    # ============================================================
    # ENSEMBLE (opcional - para máximo rendimiento)
    # ============================================================
    
    # Entrenar múltiples modelos con diferentes seeds
    USE_ENSEMBLE = False
    ENSEMBLE_MODELS = 5
    ENSEMBLE_SEEDS = [42, 123, 456, 789, 1011]
    
    # ============================================================
    # FINE-TUNING PROGRESIVO (opcional)
    # ============================================================
    
    # Fase 1: Solo LSTM y clasificador
    UNFREEZE_POSE_MLP_EPOCH = 20  # Descongelar MLP pose después
    UNFREEZE_RESNET_EPOCH = 40     # Descongelar últimas capas ResNet
    RESNET_LAYERS_TO_UNFREEZE = 1  # Número de bloques finales
    
    FINE_TUNE_LR_MULTIPLIER = 0.1  # LR más bajo para capas descongeladas
    
    # ============================================================
    # PATHS
    # ============================================================
    
    DATA_DIR = "data/processed"
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    
    # ============================================================
    # OTROS
    # ============================================================
    
    NUM_WORKERS = 4
    PIN_MEMORY = True
    SEED = 42


# Función para obtener config según estrategia
def get_config(strategy="aggressive"):
    """
    strategy: 
        - "conservative": Mejoras suaves, más estable
        - "aggressive": Todas las técnicas, máximo rendimiento
        - "simple": Solo las mejoras más probadas
    """
    config = Config()
    
    if strategy == "conservative":
        config.USE_MIXUP = False
        config.USE_STOCHASTIC_DEPTH = False
        config.USE_TTA = False
        config.GRADIENT_ACCUMULATION_STEPS = 1
        config.LABEL_SMOOTHING = 0.05
        
    elif strategy == "simple":
        config.USE_MIXUP = False
        config.USE_STOCHASTIC_DEPTH = False
        config.USE_TTA = False
        config.USE_EMA = True
        config.LABEL_SMOOTHING = 0.1
        config.USE_ONE_CYCLE = True
        config.GRADIENT_ACCUMULATION_STEPS = 2
        
    # aggressive usa todos los defaults (todas las técnicas)
    
    return config
