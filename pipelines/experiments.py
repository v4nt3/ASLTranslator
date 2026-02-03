"""
experiment_configs.py - Configuraciones predefinidas para experimentacion

Objetivo: Superar 70% de accuracy sin data leakage

Estrategia de experimentacion:
1. Empezar con configuracion conservadora (baseline)
2. Ajustar regularizacion segun overfitting observado
3. Probar diferentes schedulers
4. Ajustar capacidad del modelo

USO:
    python pipelines/run_experiment.py --config baseline
    python pipelines/run_experiment.py --config moderate
    python pipelines/run_experiment.py --config aggressive
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
import json
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuracion de un experimento."""
    name: str
    description: str
    
    # Modelo
    model_type: str = "lstm"
    hidden_dim: int = 384
    num_layers: int = 2
    bidirectional: bool = True
    use_attention: bool = True
    
    # Optimizador
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    
    # Scheduler
    scheduler_type: str = "cosine_warmup"
    warmup_epochs: int = 5
    scheduler_patience: int = 8
    scheduler_factor: float = 0.5
    
    # Regularizacion
    dropout: float = 0.4
    label_smoothing: float = 0.15
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_ema: bool = True
    ema_decay: float = 0.999
    max_grad_norm: float = 1.0
    
    # Training
    batch_size: int = 32
    num_epochs: int = 100
    early_stopping_patience: int = 15
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "model": {
                "type": self.model_type,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "bidirectional": self.bidirectional,
                "use_attention": self.use_attention,
            },
            "optimizer": {
                "type": self.optimizer,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
            },
            "scheduler": {
                "type": self.scheduler_type,
                "warmup_epochs": self.warmup_epochs,
                "patience": self.scheduler_patience,
                "factor": self.scheduler_factor,
            },
            "regularization": {
                "dropout": self.dropout,
                "label_smoothing": self.label_smoothing,
                "use_mixup": self.use_mixup,
                "mixup_alpha": self.mixup_alpha,
                "use_ema": self.use_ema,
                "ema_decay": self.ema_decay,
                "max_grad_norm": self.max_grad_norm,
            },
            "training": {
                "batch_size": self.batch_size,
                "num_epochs": self.num_epochs,
                "early_stopping_patience": self.early_stopping_patience,
            }
        }
    
    def to_args(self) -> List[str]:
        """Convierte a argumentos de linea de comandos."""
        args = [
            f"--model_type={self.model_type}",
            f"--hidden_dim={self.hidden_dim}",
            f"--num_layers={self.num_layers}",
            f"--learning_rate={self.learning_rate}",
            f"--weight_decay={self.weight_decay}",
            f"--scheduler={self.scheduler_type}",
            f"--dropout={self.dropout}",
            f"--label_smoothing={self.label_smoothing}",
            f"--batch_size={self.batch_size}",
            f"--num_epochs={self.num_epochs}",
            f"--early_stopping={self.early_stopping_patience}",
            f"--max_grad_norm={self.max_grad_norm}",
        ]
        
        if self.bidirectional:
            args.append("--bidirectional")
        if self.use_attention:
            args.append("--use_attention")
        if self.use_mixup:
            args.append("--use_mixup")
        if self.use_ema:
            args.append("--use_ema")
        
        if self.scheduler_type == "plateau":
            args.extend([
                f"--scheduler_patience={self.scheduler_patience}",
                f"--scheduler_factor={self.scheduler_factor}",
            ])
        
        return args


# =============================================================================
# CONFIGURACIONES PREDEFINIDAS
# =============================================================================

EXPERIMENTS: Dict[str, ExperimentConfig] = {}

# -----------------------------------------------------------------------------
# 1. BASELINE - Configuracion conservadora para establecer linea base
# -----------------------------------------------------------------------------
EXPERIMENTS["baseline"] = ExperimentConfig(
    name="baseline",
    description="Configuracion conservadora. Regularizacion moderada, modelo mediano.",
    
    # Modelo mediano
    hidden_dim=256,
    num_layers=2,
    bidirectional=True,
    use_attention=True,
    
    # LR conservador
    learning_rate=1e-4,
    weight_decay=0.01,
    
    # Scheduler plateau (reactivo)
    scheduler_type="plateau",
    scheduler_patience=10,
    scheduler_factor=0.5,
    
    # Regularizacion moderada
    dropout=0.3,
    label_smoothing=0.1,
    use_mixup=False,  # Sin mixup para baseline
    use_ema=True,
    ema_decay=0.999,
    
    # Training
    batch_size=64,
    num_epochs=100,
    early_stopping_patience=20,
)

# -----------------------------------------------------------------------------
# 2. MODERATE - Regularizacion balanceada
# -----------------------------------------------------------------------------
EXPERIMENTS["moderate"] = ExperimentConfig(
    name="moderate",
    description="Regularizacion balanceada. Buen punto de partida.",
    
    # Modelo
    hidden_dim=384,
    num_layers=2,
    bidirectional=True,
    use_attention=True,
    
    # Optimizador
    learning_rate=1e-4,
    weight_decay=0.05,
    
    # Scheduler cosine
    scheduler_type="cosine_warmup",
    warmup_epochs=5,
    
    # Regularizacion
    dropout=0.4,
    label_smoothing=0.15,
    use_mixup=True,
    mixup_alpha=0.2,
    use_ema=True,
    ema_decay=0.999,
    
    # Training
    batch_size=32,
    num_epochs=100,
    early_stopping_patience=15,
)

# -----------------------------------------------------------------------------
# 3. AGGRESSIVE - Regularizacion fuerte para combatir overfitting severo
# -----------------------------------------------------------------------------
EXPERIMENTS["aggressive"] = ExperimentConfig(
    name="aggressive",
    description="Regularizacion agresiva. Usar si hay mucho overfitting.",
    
    # Modelo mas pequeno
    hidden_dim=256,
    num_layers=2,
    bidirectional=True,
    use_attention=True,
    
    # LR bajo con weight decay alto
    learning_rate=5e-5,
    weight_decay=0.1,
    
    # Scheduler cosine con warmup largo
    scheduler_type="cosine_warmup",
    warmup_epochs=10,
    
    # Regularizacion fuerte
    dropout=0.5,
    label_smoothing=0.2,
    use_mixup=True,
    mixup_alpha=0.4,
    use_ema=True,
    ema_decay=0.9995,
    max_grad_norm=0.5,
    
    # Training
    batch_size=16,  # Batch pequeno = mas ruido
    num_epochs=150,
    early_stopping_patience=25,
)

# -----------------------------------------------------------------------------
# 4. LARGE_MODEL - Modelo grande con regularizacion compensatoria
# -----------------------------------------------------------------------------
EXPERIMENTS["large_model"] = ExperimentConfig(
    name="large_model",
    description="Modelo grande (512 hidden). Mas capacidad con regularizacion fuerte.",
    
    # Modelo grande
    hidden_dim=512,
    num_layers=3,
    bidirectional=True,
    use_attention=True,
    
    # Optimizador
    learning_rate=5e-5,  # LR mas bajo para modelo grande
    weight_decay=0.1,
    
    # Scheduler
    scheduler_type="cosine_warmup",
    warmup_epochs=8,
    
    # Regularizacion fuerte (compensar capacidad)
    dropout=0.5,
    label_smoothing=0.15,
    use_mixup=True,
    mixup_alpha=0.3,
    use_ema=True,
    ema_decay=0.9995,
    
    # Training
    batch_size=24,
    num_epochs=120,
    early_stopping_patience=20,
)

# -----------------------------------------------------------------------------
# 5. SMALL_MODEL - Modelo pequeno, menos propenso a overfitting
# -----------------------------------------------------------------------------
EXPERIMENTS["small_model"] = ExperimentConfig(
    name="small_model",
    description="Modelo pequeno (128 hidden). Menos capacidad = menos overfitting.",
    
    # Modelo pequeno
    hidden_dim=128,
    num_layers=2,
    bidirectional=True,
    use_attention=True,
    
    # Optimizador
    learning_rate=3e-4,  # LR mas alto porque modelo pequeno
    weight_decay=0.01,
    
    # Scheduler plateau
    scheduler_type="plateau",
    scheduler_patience=8,
    scheduler_factor=0.5,
    
    # Regularizacion ligera
    dropout=0.3,
    label_smoothing=0.1,
    use_mixup=False,
    use_ema=True,
    ema_decay=0.999,
    
    # Training
    batch_size=64,
    num_epochs=150,
    early_stopping_patience=25,
)

# -----------------------------------------------------------------------------
# 6. HIGH_LR - Learning rate alto con scheduler agresivo
# -----------------------------------------------------------------------------
EXPERIMENTS["high_lr"] = ExperimentConfig(
    name="high_lr",
    description="LR alto (3e-4) con plateau agresivo. Convergencia rapida.",
    
    # Modelo
    hidden_dim=384,
    num_layers=2,
    bidirectional=True,
    use_attention=True,
    
    # LR alto
    learning_rate=3e-4,
    weight_decay=0.05,
    
    # Plateau agresivo
    scheduler_type="plateau",
    scheduler_patience=5,
    scheduler_factor=0.3,
    
    # Regularizacion
    dropout=0.4,
    label_smoothing=0.15,
    use_mixup=True,
    mixup_alpha=0.2,
    use_ema=True,
    
    # Training
    batch_size=32,
    num_epochs=80,
    early_stopping_patience=15,
)

# -----------------------------------------------------------------------------
# 7. NO_MIXUP - Sin mixup para comparar impacto
# -----------------------------------------------------------------------------
EXPERIMENTS["no_mixup"] = ExperimentConfig(
    name="no_mixup",
    description="Sin mixup. Para comparar impacto de data augmentation.",
    
    # Modelo
    hidden_dim=384,
    num_layers=2,
    bidirectional=True,
    use_attention=True,
    
    # Optimizador
    learning_rate=1e-4,
    weight_decay=0.05,
    
    # Scheduler
    scheduler_type="cosine_warmup",
    warmup_epochs=5,
    
    # Regularizacion sin mixup
    dropout=0.45,  # Compensar falta de mixup
    label_smoothing=0.15,
    use_mixup=False,
    use_ema=True,
    
    # Training
    batch_size=32,
    num_epochs=100,
    early_stopping_patience=15,
)

# -----------------------------------------------------------------------------
# 8. TRANSFORMER_STYLE - Configuracion inspirada en transformers
# -----------------------------------------------------------------------------
EXPERIMENTS["transformer_style"] = ExperimentConfig(
    name="transformer_style",
    description="Hiperparametros estilo transformer. Warmup largo, LR bajo.",
    
    # Modelo
    hidden_dim=384,
    num_layers=2,
    bidirectional=True,
    use_attention=True,
    
    # Optimizador estilo transformer
    learning_rate=1e-4,
    weight_decay=0.1,  # Weight decay alto
    
    # Warmup largo
    scheduler_type="cosine_warmup",
    warmup_epochs=10,
    
    # Regularizacion
    dropout=0.1,  # Dropout bajo (como transformers)
    label_smoothing=0.1,
    use_mixup=True,
    mixup_alpha=0.2,
    use_ema=True,
    ema_decay=0.9999,
    
    # Training
    batch_size=32,
    num_epochs=100,
    early_stopping_patience=20,
)

# -----------------------------------------------------------------------------
# 9. GRADUAL_UNFREEZE - Para fine-tuning (si usas features preentrenadas)
# -----------------------------------------------------------------------------
EXPERIMENTS["gradual_warmup"] = ExperimentConfig(
    name="gradual_warmup",
    description="Warmup muy largo. Para evitar destruir features preentrenadas.",
    
    # Modelo
    hidden_dim=384,
    num_layers=2,
    bidirectional=True,
    use_attention=True,
    
    # LR bajo inicial
    learning_rate=5e-5,
    weight_decay=0.05,
    
    # Warmup MUY largo
    scheduler_type="cosine_warmup",
    warmup_epochs=15,
    
    # Regularizacion
    dropout=0.4,
    label_smoothing=0.15,
    use_mixup=True,
    mixup_alpha=0.2,
    use_ema=True,
    
    # Training largo
    batch_size=32,
    num_epochs=150,
    early_stopping_patience=30,
)

# -----------------------------------------------------------------------------
# 10. PLATEAU_CONSERVATIVE - Plateau muy conservador
# -----------------------------------------------------------------------------
EXPERIMENTS["plateau_conservative"] = ExperimentConfig(
    name="plateau_conservative",
    description="Plateau conservador. Reduce LR solo cuando realmente se estanca.",
    
    # Modelo
    hidden_dim=384,
    num_layers=2,
    bidirectional=True,
    use_attention=True,
    
    # Optimizador
    learning_rate=1e-4,
    weight_decay=0.05,
    
    # Plateau conservador
    scheduler_type="plateau",
    scheduler_patience=15,  # Mucha paciencia
    scheduler_factor=0.5,
    
    # Regularizacion
    dropout=0.4,
    label_smoothing=0.15,
    use_mixup=True,
    mixup_alpha=0.2,
    use_ema=True,
    
    # Training
    batch_size=32,
    num_epochs=150,
    early_stopping_patience=30,
)


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def get_config(name: str) -> ExperimentConfig:
    """Obtiene una configuracion por nombre."""
    if name not in EXPERIMENTS:
        available = ", ".join(EXPERIMENTS.keys())
        raise ValueError(f"Configuracion '{name}' no encontrada. Disponibles: {available}")
    return EXPERIMENTS[name]


def list_configs() -> List[str]:
    """Lista todas las configuraciones disponibles."""
    return list(EXPERIMENTS.keys())


def print_config_summary():
    """Imprime resumen de todas las configuraciones."""
    print("\n" + "=" * 80)
    print("CONFIGURACIONES DE EXPERIMENTOS DISPONIBLES")
    print("=" * 80)
    
    for name, config in EXPERIMENTS.items():
        print(f"\n[{name}]")
        print(f"  Descripcion: {config.description}")
        print(f"  Modelo: {config.model_type} | hidden={config.hidden_dim} | layers={config.num_layers}")
        print(f"  Optimizer: {config.optimizer} | lr={config.learning_rate} | wd={config.weight_decay}")
        print(f"  Scheduler: {config.scheduler_type}", end="")
        if config.scheduler_type == "cosine_warmup":
            print(f" | warmup={config.warmup_epochs}")
        else:
            print(f" | patience={config.scheduler_patience} | factor={config.scheduler_factor}")
        print(f"  Regularizacion: dropout={config.dropout} | ls={config.label_smoothing} | mixup={config.use_mixup}")
        print(f"  Training: batch={config.batch_size} | epochs={config.num_epochs}")
    
    print("\n" + "=" * 80)


def save_config(config: ExperimentConfig, output_path: Path):
    """Guarda configuracion a JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def compare_configs(config_names: List[str]):
    """Compara multiples configuraciones lado a lado."""
    configs = [get_config(name) for name in config_names]
    
    print("\n" + "=" * 100)
    print("COMPARACION DE CONFIGURACIONES")
    print("=" * 100)
    
    # Header
    print(f"{'Parametro':<25}", end="")
    for name in config_names:
        print(f"{name:<15}", end="")
    print()
    print("-" * 100)
    
    # Rows
    params = [
        ("hidden_dim", "Hidden Dim"),
        ("num_layers", "Num Layers"),
        ("learning_rate", "Learning Rate"),
        ("weight_decay", "Weight Decay"),
        ("scheduler_type", "Scheduler"),
        ("dropout", "Dropout"),
        ("label_smoothing", "Label Smooth"),
        ("use_mixup", "Mixup"),
        ("batch_size", "Batch Size"),
        ("num_epochs", "Epochs"),
    ]
    
    for attr, label in params:
        print(f"{label:<25}", end="")
        for config in configs:
            value = getattr(config, attr)
            if isinstance(value, float) and value < 0.01:
                print(f"{value:<15.0e}", end="")
            else:
                print(f"{str(value):<15}", end="")
        print()
    
    print("=" * 100)


# =============================================================================
# RECOMENDACIONES DE USO
# =============================================================================

RECOMMENDATIONS = """
GUIA DE EXPERIMENTACION
=======================

ORDEN RECOMENDADO DE EXPERIMENTOS:

1. BASELINE primero
   - Establece linea base sin tecnicas avanzadas
   - Si accuracy < 50%: problema con datos o features
   - Si accuracy > 70%: quizas no necesitas mas regularizacion

2. Si hay OVERFITTING (train_acc >> val_acc):
   - Probar: aggressive, small_model
   - Aumentar: dropout, weight_decay
   - Reducir: hidden_dim, learning_rate

3. Si hay UNDERFITTING (train_acc ~ val_acc pero ambos bajos):
   - Probar: large_model, high_lr
   - Reducir: dropout, weight_decay
   - Aumentar: hidden_dim, epochs

4. Si el entrenamiento es INESTABLE:
   - Probar: gradual_warmup, transformer_style
   - Reducir: learning_rate, batch_size
   - Aumentar: warmup_epochs, max_grad_norm

5. Para COMPARAR impacto de mixup:
   - Correr: moderate vs no_mixup

6. Para COMPARAR schedulers:
   - Correr: moderate (cosine) vs plateau_conservative

METRICAS A MONITOREAR:
- Gap train_acc - val_acc (overfitting si > 15%)
- Val loss trend (debe decrecer)
- Top-5 accuracy (mas informativo para muchas clases)
- EMA vs normal accuracy

TIPS:
- Cada experimento toma ~2-4 horas con GPU
- Guarda checkpoints frecuentemente
- Usa TensorBoard para visualizar curvas
"""


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gestion de configuraciones de experimentos")
    parser.add_argument("--list", action="store_true", help="Lista todas las configuraciones")
    parser.add_argument("--show", type=str, help="Muestra detalles de una configuracion")
    parser.add_argument("--compare", nargs="+", help="Compara configuraciones")
    parser.add_argument("--recommendations", action="store_true", help="Muestra recomendaciones")
    parser.add_argument("--export", type=str, help="Exporta configuracion a JSON")
    parser.add_argument("--export_all", type=str, help="Exporta todas las configuraciones a directorio")
    
    args = parser.parse_args()
    
    if args.list:
        print_config_summary()
    
    elif args.show:
        config = get_config(args.show)
        print(f"\nConfiguracion: {config.name}")
        print(f"Descripcion: {config.description}")
        print("\nParametros:")
        for key, value in config.to_dict().items():
            if isinstance(value, dict):
                print(f"\n  [{key}]")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")
        
        print("\n\nArgumentos CLI:")
        print("  " + " \\\n    ".join(config.to_args()))
    
    elif args.compare:
        compare_configs(args.compare)
    
    elif args.recommendations:
        print(RECOMMENDATIONS)
    
    elif args.export:
        name, path = args.export.split(":")
        config = get_config(name)
        save_config(config, Path(path))
        print(f"Configuracion '{name}' exportada a {path}")
    
    elif args.export_all:
        output_dir = Path(args.export_all)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, config in EXPERIMENTS.items():
            save_config(config, output_dir / f"{name}.json")
        print(f"Todas las configuraciones exportadas a {output_dir}/")
    
    else:
        print_config_summary()
        print("\nUsa --help para ver opciones disponibles")
