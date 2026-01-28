"""
Script de entrenamiento para modelo de videos completos.
CORREGIDO: Scheduler sincronizado con accuracy, gradient clipping ajustado
"""

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
from torch.cuda.amp import GradScaler, autocast  # type: ignore
import numpy as np  # type: ignore
from pathlib import Path
from tqdm import tqdm  # type: ignore
import logging
import json
from typing import Tuple, Optional
import argparse

from pipelines_video.config import config
from pipelines_video.models import get_video_model, get_loss_function
from pipelines_video.dataset import create_video_dataloaders


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """
    Mixup: mezcla pares de ejemplos y sus labels.
    Muy efectivo contra overfitting.
    
    Args:
        x: (B, T, D) features
        y: (B,) labels
        alpha: parametro de distribucion Beta (0.2-0.4 recomendado)
    
    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss para mixup: combinacion ponderada de losses"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def setup_logging(checkpoint_dir: Path) -> logging.Logger:
    """Configura logging para consola y archivo"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file = checkpoint_dir / "training.log"
    
    logger = logging.getLogger("video_trainer")
    
    # Evitar duplicacion: no propagar al root logger
    logger.propagate = False
    
    # Limpiar handlers existentes para evitar duplicados si se llama multiples veces
    logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    return logger


class VideoTrainer:
    """Entrenador para modelos de video completo - CON MIXUP"""
    
    def __init__(
        self,
        model_type: str = "lstm",
        num_classes: int = None,
        learning_rate: float = None,
        weight_decay: float = None,
        num_epochs: int = None,
        device: str = None,
        use_amp: bool = None,
        checkpoint_dir: Path = None,
        use_attention: bool = True,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = None,
        use_mixup: bool = True,      # NUEVO: Mixup habilitado por defecto
        mixup_alpha: float = 0.2     # NUEVO: Alpha para mixup
    ):
        # Defaults
        if num_classes is None:
            num_classes = config.model.num_classes
        if learning_rate is None:
            learning_rate = config.training.learning_rate
        if weight_decay is None:
            weight_decay = config.training.weight_decay
        if num_epochs is None:
            num_epochs = config.training.num_epochs
        if device is None:
            device = config.training.device
        if use_amp is None:
            use_amp = config.training.use_amp
        if checkpoint_dir is None:
            checkpoint_dir = config.model_paths.checkpoints
        if focal_gamma is None:
            focal_gamma = config.training.focal_loss_gamma
        if max_grad_norm is None:
            max_grad_norm = config.training.max_grad_norm  # CORREGIDO: Usa 2.0
        
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp
        self.checkpoint_dir = Path(checkpoint_dir)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_mixup = use_mixup        # NUEVO
        self.mixup_alpha = mixup_alpha    # NUEVO
        
        # Logger
        self.logger = setup_logging(self.checkpoint_dir)
        
        # Modelo
        self.model = get_video_model(
            model_type=model_type,
            num_classes=num_classes,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            bidirectional=config.model.bidirectional,
            use_attention=use_attention
            
        ).to(self.device)
        
        # CORREGIDO: AdamW en lugar de Adam para mejor regularizacion
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Loss con class weights
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        
        self.criterion = get_loss_function(
            class_weights=class_weights,
            focal_gamma=focal_gamma,
            label_smoothing=config.training.label_smoothing
        )
        
        # Scheduler - CORREGIDO: Configuracion basada en config
        self._setup_scheduler()
        
        # AMP
        self.scaler = GradScaler() if use_amp else None
        
        # History
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'train_top5_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_top5_accuracy': [],
            'learning_rate': []
        }
        
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        
        # Logging
        self.logger.info("="*60)
        self.logger.info("VideoTrainer inicializado")
        self.logger.info("="*60)
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Model type: {model_type}")
        self.logger.info(f"  Num classes: {num_classes}")
        self.logger.info(f"  Learning rate: {learning_rate}")
        self.logger.info(f"  Weight decay: {weight_decay}")
        self.logger.info(f"  Focal gamma: {focal_gamma}")
        self.logger.info(f"  Use attention: {use_attention}")
        self.logger.info(f"  Use AMP: {use_amp}")
        self.logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
        self.logger.info(f"  Max grad norm: {max_grad_norm}")
        self.logger.info(f"  Class weights: {'enabled' if class_weights is not None else 'disabled'}")
        self.logger.info(f"  Scheduler monitor: {config.training.scheduler_monitor}")
        self.logger.info(f"  Mixup: {'enabled (alpha=' + str(mixup_alpha) + ')' if use_mixup else 'disabled'}")
        self.logger.info("="*60)
    
    def _setup_scheduler(self):
        """Configura el scheduler segun config - CORREGIDO"""
        scheduler_type = config.training.scheduler_type
        scheduler_monitor = config.training.scheduler_monitor
        
        if scheduler_type == "plateau":
            # CORREGIDO: Puede monitorear accuracy o loss
            mode = 'max' if scheduler_monitor == 'accuracy' else 'min'
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,  # CORREGIDO: 'max' para accuracy
                factor=config.training.scheduler_factor,
                patience=config.training.scheduler_patience,
                min_lr=config.training.scheduler_min_lr,
                verbose=True
            )
            self.scheduler_type = "plateau"
            self.scheduler_monitor = scheduler_monitor
        elif scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,  # Restart cada 10 epochs
                T_mult=2,
                eta_min=config.training.scheduler_min_lr
            )
            self.scheduler_type = "cosine"
            self.scheduler_monitor = None
        elif scheduler_type == "onecycle":
            # OneCycleLR requiere conocer steps totales
            self.scheduler = None
            self.scheduler_type = "onecycle"
            self.scheduler_monitor = None
        else:
            self.scheduler = None
            self.scheduler_type = None
            self.scheduler_monitor = None
        
        self.logger.info(f"Scheduler: {scheduler_type}, monitor: {scheduler_monitor}")
    
    def _setup_onecycle_scheduler(self, steps_per_epoch: int):
        """Configura OneCycleLR despues de conocer los steps"""
        if self.scheduler_type == "onecycle" and self.scheduler is None:
            total_steps = steps_per_epoch * self.num_epochs
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=config.training.learning_rate * 10,
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos'
            )
            self.logger.info(f"OneCycleLR configurado con {total_steps} steps")
    
    def train(self, train_loader, val_loader):
        """Loop de entrenamiento principal"""
        self.logger.info(f"\nIniciando entrenamiento por {self.num_epochs} epochs")
        self.logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")
        
        # Setup OneCycle si es necesario
        self._setup_onecycle_scheduler(len(train_loader))
        
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            self.logger.info(f"\n{'='*40}")
            self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            self.logger.info(f"{'='*40}")
            
            # Train
            train_loss, train_acc, train_top5 = self._train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_top5 = self._validate(val_loader)
            
            # CORREGIDO: Scheduler step basado en metrica configurada
            if self.scheduler_type == "plateau":
                metric = val_acc if self.scheduler_monitor == 'accuracy' else val_loss
                self.scheduler.step(metric)
            elif self.scheduler_type == "cosine":
                self.scheduler.step()
            # OneCycle hace step en cada batch, no por epoch
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['train_top5_accuracy'].append(train_top5)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['val_top5_accuracy'].append(val_top5)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Logging
            self.logger.info(f"  Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Top5: {train_top5:.4f}")
            self.logger.info(f"  Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Top5: {val_top5:.4f}")
            self.logger.info(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            self.logger.info(f"  Patience: {patience_counter}/{config.training.early_stopping_patience}")
            
            # Save best model - BASADO EN ACCURACY
            improved = False
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint("best_model.pt", epoch, val_acc)
                self.logger.info(f"  >>> New best model! Accuracy: {val_acc:.4f}")
                improved = True
                patience_counter = 0
            
            # Tambien guardar si mejora loss significativamente (backup)
            if val_loss < self.best_val_loss * 0.99:  # 1% mejora
                self.best_val_loss = val_loss
                if not improved:
                    self._save_checkpoint("best_loss_model.pt", epoch, val_acc)
            
            if not improved:
                patience_counter += 1
            
            # Checkpoint periodico
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt", epoch, val_acc)
            
            # Early stopping
            if patience_counter >= config.training.early_stopping_patience:
                self.logger.info(f"\nEarly stopping! No mejora en {patience_counter} epochs.")
                break
        
        # Save final model
        self._save_checkpoint("final_model.pt", epoch, val_acc)
        
        # Save history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Entrenamiento completado!")
        self.logger.info(f"  Best Val Accuracy: {self.best_val_acc:.4f}")
        self.logger.info(f"  Checkpoints en: {self.checkpoint_dir}")
        self.logger.info(f"{'='*60}\n")
        
        return self.history
    
    def _train_epoch(self, train_loader) -> Tuple[float, float, float]:
        """Entrena una epoca"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_logits = []
        
        self.optimizer.zero_grad()
        
        with tqdm(train_loader, desc="Training", leave=False) as pbar:
            for batch_idx, (features, targets, lengths) in enumerate(pbar):
                features = features.to(self.device)
                targets = targets.to(self.device)
                lengths = lengths.to(self.device)
                
                # Aplicar Mixup si esta habilitado
                if self.use_mixup:
                    features, targets_a, targets_b, lam = mixup_data(
                        features, targets, self.mixup_alpha
                    )
                
                # Forward
                if self.use_amp:
                    with autocast():
                        logits = self.model(features, lengths)
                        
                        # Loss con o sin mixup
                        if self.use_mixup:
                            loss = mixup_criterion(self.criterion, logits, targets_a, targets_b, lam)
                        else:
                            loss = self.criterion(logits, targets)
                        loss = loss / self.gradient_accumulation_steps
                    
                    self.scaler.scale(loss).backward()
                    
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.max_grad_norm
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                        
                        # OneCycle step por batch
                        if self.scheduler_type == "onecycle" and self.scheduler is not None:
                            self.scheduler.step()
                else:
                    logits = self.model(features, lengths)
                    
                    # Loss con o sin mixup
                    if self.use_mixup:
                        loss = mixup_criterion(self.criterion, logits, targets_a, targets_b, lam)
                    else:
                        loss = self.criterion(logits, targets)
                    loss = loss / self.gradient_accumulation_steps
                    loss.backward()
                    
                    if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.max_grad_norm
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        
                        # OneCycle step por batch
                        if self.scheduler_type == "onecycle" and self.scheduler is not None:
                            self.scheduler.step()
                
                total_loss += loss.item() * self.gradient_accumulation_steps
                
                # Collect predictions
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_logits.append(logits.detach().cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}'})
        
        # Metrics
        avg_loss = total_loss / len(train_loader)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_logits = np.concatenate(all_logits)
        
        accuracy = np.mean(all_preds == all_targets)
        top5_accuracy = self._compute_top5_accuracy(all_logits, all_targets)
        
        return avg_loss, accuracy, top5_accuracy
    
    @torch.no_grad()
    def _validate(self, val_loader) -> Tuple[float, float, float]:
        """Valida el modelo"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_logits = []
        
        with tqdm(val_loader, desc="Validating", leave=False) as pbar:
            for features, targets, lengths in pbar:
                features = features.to(self.device)
                targets = targets.to(self.device)
                lengths = lengths.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        logits = self.model(features, lengths)
                        loss = self.criterion(logits, targets)
                else:
                    logits = self.model(features, lengths)
                    loss = self.criterion(logits, targets)
                
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
        
        # Metrics
        avg_loss = total_loss / len(val_loader)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_logits = np.concatenate(all_logits)
        
        accuracy = np.mean(all_preds == all_targets)
        top5_accuracy = self._compute_top5_accuracy(all_logits, all_targets)
        
        return avg_loss, accuracy, top5_accuracy
    
    def _compute_top5_accuracy(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Calcula top-5 accuracy"""
        top5_preds = np.argsort(logits, axis=1)[:, -5:]
        correct = np.any(top5_preds == targets[:, None], axis=1)
        return np.mean(correct)
    
    def _save_checkpoint(self, filename: str, epoch: int, val_acc: float):
        """Guarda checkpoint"""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': val_acc,
            'best_val_acc': self.best_val_acc,
            'config': {
                'num_classes': self.num_classes,
                'hidden_dim': config.model.hidden_dim,
                'num_layers': config.model.num_layers,
                'bidirectional': config.model.bidirectional,
                'use_attention': config.model.use_attention,
                'classifier_hidden_dim': config.model.classifier_hidden_dim
            }
        }, checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelo de video completo")
    
    # Data
    parser.add_argument("--features_dir", type=Path, default=None,
                        help="Directorio con features fusionadas")
    parser.add_argument("--metadata_path", type=Path, default=None,
                        help="Ruta a dataset_meta.json")
    parser.add_argument("--checkpoint_dir", type=Path, default=None,
                        help="Directorio para checkpoints")
    
    # Model
    parser.add_argument("--model_type", type=str, default="lstm",
                        choices=["lstm", "transformer"])
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--use_attention", action="store_true", default=True)
    parser.add_argument("--no_attention", dest="use_attention", action="store_false")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_amp", action="store_true")
    
    # Class balancing - CORREGIDO: Defaults coherentes con config
    parser.add_argument("--use_class_weights", action="store_true", 
                        default=config.training.use_class_weights)  # True por defecto
    parser.add_argument("--no_class_weights", dest="use_class_weights", action="store_false")
    parser.add_argument("--focal_gamma", type=float, default=None,
                        help="Gamma para FocalLoss (0 = CrossEntropy)")
    parser.add_argument("--use_weighted_sampler", action="store_true",
                        help="Usar WeightedRandomSampler en lugar de weighted loss")
    
    # Augmentation
    parser.add_argument("--no_augmentation", action="store_true",
                        help="Deshabilitar data augmentation")
    
    # Mixup - REDUCIDO para balancear train/val gap
    parser.add_argument("--use_mixup", action="store_true", default=True,
                        help="Usar Mixup (habilitado por defecto)")
    parser.add_argument("--no_mixup", dest="use_mixup", action="store_false",
                        help="Deshabilitar Mixup")
    parser.add_argument("--mixup_alpha", type=float, default=0.1,
                        help="Alpha para Mixup (0.1 = suave, 0.4 = fuerte)")
    
    # Gradient accumulation
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    
    args = parser.parse_args()
    
    # Setup parameters
    features_dir = args.features_dir or config.data_paths.features_fused
    metadata_path = args.metadata_path or config.data_paths.dataset_meta
    checkpoint_dir = args.checkpoint_dir or config.model_paths.checkpoints
    batch_size = args.batch_size or config.training.batch_size
    num_workers = args.num_workers or config.training.num_workers
    use_augmentation = not args.no_augmentation
    
    print("="*60)
    print("ENTRENAMIENTO DE MODELO DE VIDEO COMPLETO")
    print("="*60)
    print(f"Features dir: {features_dir}")
    print(f"Metadata path: {metadata_path}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Batch size: {batch_size}")
    print(f"Use class weights: {args.use_class_weights}")
    print(f"Focal gamma: {args.focal_gamma or config.training.focal_loss_gamma}")
    print(f"Use augmentation: {use_augmentation}")
    print(f"Use mixup: {args.use_mixup} (alpha={args.mixup_alpha})")
    print(f"Scheduler monitor: {config.training.scheduler_monitor}")
    print(f"Weight decay: {config.training.weight_decay}")
    print("="*60 + "\n")
    
    # Create dataloaders - CORREGIDO: Con augmentation
    train_loader, val_loader, test_loader, class_weights = create_video_dataloaders(
        features_dir=features_dir,
        metadata_path=metadata_path,
        batch_size=batch_size,
        num_workers=num_workers,
        use_weighted_sampler=args.use_weighted_sampler,
        use_augmentation=use_augmentation  # NUEVO
    )
    
    # Class weights for loss
    loss_weights = class_weights if args.use_class_weights else None
    
    # Create trainer
    trainer = VideoTrainer(
        model_type=args.model_type,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        device=args.device,
        use_amp=not args.no_amp,
        checkpoint_dir=checkpoint_dir,
        use_attention=args.use_attention,
        class_weights=loss_weights,
        focal_gamma=args.focal_gamma,
        gradient_accumulation_steps=args.gradient_accumulation,
        use_mixup=args.use_mixup,           # NUEVO
        mixup_alpha=args.mixup_alpha        # NUEVO
    )
    
    # Train
    trainer.train(train_loader, val_loader)
    
    print("\nEntrenamiento completado!")


if __name__ == "__main__":
    main()
