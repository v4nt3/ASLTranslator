"""
Script de entrenamiento con REGULARIZACION COMPLETA

Incluye todas las tecnicas de regularizacion:
- Split por VIDEO (sin data leakage)
- Variational Dropout
- Weight Decay fuerte (AdamW)
- Label Smoothing
- Gradient Clipping
- EMA (Exponential Moving Average)
- Mixup
- Cosine Annealing con Warmup
- Test Time Augmentation (TTA)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import json
import math
from typing import Tuple, Optional, Dict, Any
import argparse
from collections import defaultdict

from pipelines.models_temporal import (
    get_temporal_model,
    ExponentialMovingAverage,
    MixupCutmix,
    mixup_criterion,
    RDropLoss,
    count_parameters
)
from pipelines.dataset_temporal import (
    create_temporal_dataloaders_fixed,
    analyze_class_distribution
)
from config import config


def setup_logging(checkpoint_dir: Path) -> logging.Logger:
    """Configura logging"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file = checkpoint_dir / "training.log"
    
    logger = logging.getLogger("RegularizedTrainer")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


class CosineAnnealingWarmup:
    """
    Learning rate scheduler con warmup lineal y cosine annealing.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_lr_init: float = 1e-6,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr_init = warmup_lr_init
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.last_epoch = last_epoch
        
    def step(self, epoch: int = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.warmup_lr_init + (self.base_lr - self.warmup_lr_init) * (epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy con label smoothing"""
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = pred.size(-1)
        
        # One hot con smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = -true_dist * torch.log_softmax(pred, dim=-1)
        
        if self.reduction == 'mean':
            return loss.sum(dim=-1).mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.sum(dim=-1)


class RegularizedTrainer:
    """
    Entrenador con TODAS las tecnicas de regularizacion.
    """
    
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
        # Regularizacion
        use_ema: bool = None,
        ema_decay: float = None,
        use_mixup: bool = None,
        mixup_alpha: float = None,
        label_smoothing: float = None,
        use_rdrop: bool = None,
        rdrop_alpha: float = None,
        warmup_epochs: int = None,
        max_grad_norm: float = None,
        # Modelo
        hidden_dim: int = None,
        dropout: float = None,
        classifier_dropout: float = None,
        use_variational_dropout: bool = None
    ):
        # Defaults from config
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
            checkpoint_dir = Path("models/temporal_regularized")
        if use_ema is None:
            use_ema = config.training.use_ema
        if ema_decay is None:
            ema_decay = config.training.ema_decay
        if use_mixup is None:
            use_mixup = config.data.use_mixup
        if mixup_alpha is None:
            mixup_alpha = config.data.mixup_alpha
        if label_smoothing is None:
            label_smoothing = config.training.label_smoothing
        if use_rdrop is None:
            use_rdrop = config.training.use_rdrop
        if rdrop_alpha is None:
            rdrop_alpha = config.training.rdrop_alpha
        if warmup_epochs is None:
            warmup_epochs = config.training.warmup_epochs
        if max_grad_norm is None:
            max_grad_norm = config.training.max_grad_norm
        if hidden_dim is None:
            hidden_dim = config.training.model_hidden_dim
        if dropout is None:
            dropout = config.training.model_dropout
        if classifier_dropout is None:
            classifier_dropout = config.model.classifier_dropout
        if use_variational_dropout is None:
            use_variational_dropout = config.training.use_variational_dropout
        
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_grad_norm = max_grad_norm
        
        # Flags de regularizacion
        self.use_ema = use_ema
        self.use_mixup = use_mixup
        self.use_rdrop = use_rdrop
        
        # Setup logging
        self.logger = setup_logging(self.checkpoint_dir)
        
        self.logger.info("="*60)
        self.logger.info("ENTRENAMIENTO CON REGULARIZACION COMPLETA")
        self.logger.info("="*60)
        
        # Crear modelo REGULARIZADO
        self.model = get_temporal_model(
            model_type=model_type,
            num_classes=num_classes,
            regularized=True,
            hidden_dim=hidden_dim,
            num_layers=config.training.model_num_layers,
            dropout=dropout,
            bidirectional=config.training.model_bidirectional,
            use_attention=use_attention,
            use_variational_dropout=use_variational_dropout,
            use_layer_norm=True,
            classifier_dropout=classifier_dropout,
            input_feature_dropout=config.data.feature_dropout_prob
        ).to(self.device)
        
        # Contar parametros
        total_params, trainable_params = count_parameters(self.model)
        self.logger.info(f"Parametros totales: {total_params:,}")
        self.logger.info(f"Parametros entrenables: {trainable_params:,}")
        
        # Optimizer: AdamW con weight decay fuerte
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            eps=config.training.adam_eps
        )
        
        # Scheduler: Cosine con warmup
        self.scheduler = CosineAnnealingWarmup(
            optimizer=self.optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=num_epochs,
            warmup_lr_init=config.training.warmup_lr_init,
            min_lr=config.training.min_lr
        )
        
        # Loss con label smoothing
        if label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
            self.logger.info(f"Usando Label Smoothing: {label_smoothing}")
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # R-Drop loss (opcional)
        if use_rdrop:
            self.rdrop_loss = RDropLoss(alpha=rdrop_alpha)
            self.logger.info(f"Usando R-Drop con alpha: {rdrop_alpha}")
        else:
            self.rdrop_loss = None
        
        # EMA (Exponential Moving Average)
        if use_ema:
            self.ema = ExponentialMovingAverage(self.model, decay=ema_decay)
            self.logger.info(f"Usando EMA con decay: {ema_decay}")
        else:
            self.ema = None
        
        # Mixup
        if use_mixup:
            self.mixup = MixupCutmix(alpha=mixup_alpha, prob=0.5)
            self.logger.info(f"Usando Mixup con alpha: {mixup_alpha}")
        else:
            self.mixup = None
        
        # AMP
        self.scaler = GradScaler() if use_amp else None
        
        # Historia
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'train_top5_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_top5_accuracy': [],
            'val_ema_accuracy': [],
            'learning_rate': []
        }
        
        self.best_val_acc = 0.0
        self.best_ema_acc = 0.0
        self.split_stats = None
        
        # Log configuracion
        self.logger.info(f"\nConfiguracion:")
        self.logger.info(f"   Device: {self.device}")
        self.logger.info(f"   Model type: {model_type}")
        self.logger.info(f"   Hidden dim: {hidden_dim}")
        self.logger.info(f"   Num classes: {num_classes}")
        self.logger.info(f"   Learning rate: {learning_rate}")
        self.logger.info(f"   Weight decay: {weight_decay}")
        self.logger.info(f"   Dropout: {dropout}")
        self.logger.info(f"   Classifier dropout: {classifier_dropout}")
        self.logger.info(f"   Variational dropout: {use_variational_dropout}")
        self.logger.info(f"   Use attention: {use_attention}")
        self.logger.info(f"   Use AMP: {use_amp}")
        self.logger.info(f"   Max grad norm: {max_grad_norm}")
        self.logger.info(f"   Warmup epochs: {warmup_epochs}")
    
    def train(
        self, 
        train_loader, 
        val_loader, 
        split_stats: dict = None
    ) -> float:
        """Loop de entrenamiento principal"""
        
        if split_stats:
            self.split_stats = split_stats
            stats_path = self.checkpoint_dir / "split_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(split_stats, f, indent=2)
        
        self.logger.info(f"\nIniciando entrenamiento por {self.num_epochs} epochs")
        
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Actualizar LR
            current_lr = self.scheduler.step(epoch)
            
            self.logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs} | LR: {current_lr:.6f}")
            
            # Train
            train_loss, train_acc, train_top5 = self.train_epoch(train_loader)
            
            # Validate (modelo normal)
            val_loss, val_acc, val_top5 = self.validate(val_loader, use_ema=False)
            
            # Validate (modelo EMA si esta activo)
            val_ema_acc = 0.0
            if self.ema:
                _, val_ema_acc, _ = self.validate(val_loader, use_ema=True)
            
            # Guardar historia
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['train_top5_accuracy'].append(train_top5)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            self.history['val_top5_accuracy'].append(val_top5)
            self.history['val_ema_accuracy'].append(val_ema_acc)
            self.history['learning_rate'].append(current_lr)
            
            # Log
            self.logger.info(f"   Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Top-5: {train_top5:.4f}")
            self.logger.info(f"   Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Top-5: {val_top5:.4f}")
            if self.ema:
                self.logger.info(f"   Val EMA Acc: {val_ema_acc:.4f}")
            
            # Detectar overfitting
            overfit_gap = train_acc - val_acc
            self.logger.info(f"   Overfit gap: {overfit_gap:.4f}")
            
            # Guardar mejor modelo
            best_acc_this_epoch = max(val_acc, val_ema_acc) if self.ema else val_acc
            
            if best_acc_this_epoch > self.best_val_acc:
                self.best_val_acc = best_acc_this_epoch
                best_path = self.checkpoint_dir / "best_model.pt"
                
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'val_ema_accuracy': val_ema_acc,
                    'split_stats': self.split_stats
                }
                
                if self.ema:
                    save_dict['ema_state_dict'] = self.ema.state_dict()
                
                torch.save(save_dict, best_path)
                self.logger.info(f"   [BEST] Modelo guardado! Acc: {best_acc_this_epoch:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                self.logger.info(f"   Patience: {patience_counter}/{config.training.early_stopping_patience}")
            
            # Checkpoint periodico
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_accuracy': val_acc,
                    'history': self.history
                }, checkpoint_path)
            
            # Early stopping
            if patience_counter >= config.training.early_stopping_patience:
                self.logger.info(f"\nEarly stopping en epoch {epoch + 1}")
                break
        
        # Guardar modelo final
        final_path = self.checkpoint_dir / "final_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'val_accuracy': self.best_val_acc,
            'history': self.history
        }, final_path)
        
        # Guardar historia
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.logger.info(f"\nEntrenamiento completado. Best Acc: {self.best_val_acc:.4f}")
        
        return self.best_val_acc
    
    def train_epoch(self, train_loader) -> Tuple[float, float, float]:
        """Entrena una epoca con todas las regularizaciones"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_logits = []
        
        with tqdm(train_loader, desc="Training") as pbar:
            for features, targets, lengths in pbar:
                features = features.to(self.device)
                targets = targets.to(self.device)
                lengths = lengths.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Mixup (si esta activo)
                use_mixup_this_batch = self.mixup is not None and self.model.training
                if use_mixup_this_batch:
                    features, targets_a, targets_b, lam = self.mixup(features, targets, lengths)
                
                if self.use_amp:
                    with autocast():
                        if self.use_rdrop and self.rdrop_loss:
                            # R-Drop: dos forward passes
                            logits1 = self.model(features, lengths)
                            logits2 = self.model(features, lengths)
                            
                            if use_mixup_this_batch:
                                loss1 = mixup_criterion(self.criterion, logits1, targets_a, targets_b, lam)
                                loss2 = mixup_criterion(self.criterion, logits2, targets_a, targets_b, lam)
                                loss = 0.5 * (loss1 + loss2)
                            else:
                                loss = self.rdrop_loss(logits1, logits2, targets)
                            
                            logits = logits1  # Para metricas
                        else:
                            logits = self.model(features, lengths)
                            
                            if use_mixup_this_batch:
                                loss = mixup_criterion(self.criterion, logits, targets_a, targets_b, lam)
                            else:
                                loss = self.criterion(logits, targets)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.use_rdrop and self.rdrop_loss:
                        logits1 = self.model(features, lengths)
                        logits2 = self.model(features, lengths)
                        
                        if use_mixup_this_batch:
                            loss1 = mixup_criterion(self.criterion, logits1, targets_a, targets_b, lam)
                            loss2 = mixup_criterion(self.criterion, logits2, targets_a, targets_b, lam)
                            loss = 0.5 * (loss1 + loss2)
                        else:
                            loss = self.rdrop_loss(logits1, logits2, targets)
                        
                        logits = logits1
                    else:
                        logits = self.model(features, lengths)
                        
                        if use_mixup_this_batch:
                            loss = mixup_criterion(self.criterion, logits, targets_a, targets_b, lam)
                        else:
                            loss = self.criterion(logits, targets)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                # Actualizar EMA
                if self.ema:
                    self.ema.update()
                
                total_loss += loss.item()
                
                # Para metricas, usar targets originales (no mixup)
                original_targets = targets if not use_mixup_this_batch else targets_a
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(original_targets.cpu().numpy())
                all_logits.append(logits.detach().cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_logits = np.concatenate(all_logits)
        
        accuracy = np.mean(all_preds == all_targets)
        top5_accuracy = self._compute_top5_accuracy(all_logits, all_targets)
        
        return avg_loss, accuracy, top5_accuracy
    
    @torch.no_grad()
    def validate(
        self, 
        val_loader, 
        use_ema: bool = False
    ) -> Tuple[float, float, float]:
        """Valida el modelo"""
        
        # Aplicar pesos EMA si se solicita
        if use_ema and self.ema:
            self.ema.apply_shadow()
        
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_logits = []
        
        desc = "Validating (EMA)" if use_ema else "Validating"
        
        with tqdm(val_loader, desc=desc, leave=False) as pbar:
            for features, targets, lengths in pbar:
                features = features.to(self.device)
                targets = targets.to(self.device)
                lengths = lengths.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        logits = self.model(features, lengths)
                        loss = nn.CrossEntropyLoss()(logits, targets)  # Sin smoothing en val
                else:
                    logits = self.model(features, lengths)
                    loss = nn.CrossEntropyLoss()(logits, targets)
                
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
        
        # Restaurar pesos originales
        if use_ema and self.ema:
            self.ema.restore()
        
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
    
    @torch.no_grad()
    def evaluate_with_tta(
        self, 
        test_loader, 
        num_augments: int = 5
    ) -> Tuple[float, float]:
        """
        Evaluacion con Test Time Augmentation.
        Promedia predicciones de multiples pasadas con ruido.
        """
        self.logger.info(f"Evaluando con TTA ({num_augments} augmentaciones)")
        
        if self.ema:
            self.ema.apply_shadow()
        
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        noise_std = config.data.feature_noise_std
        
        for features, targets, lengths in tqdm(test_loader, desc="TTA Evaluation"):
            features = features.to(self.device)
            targets = targets.to(self.device)
            lengths = lengths.to(self.device)
            
            # Acumular logits de multiples pasadas
            accumulated_logits = None
            
            for i in range(num_augments):
                if i == 0:
                    # Primera pasada sin ruido
                    augmented_features = features
                else:
                    # Pasadas con ruido gaussiano
                    noise = torch.randn_like(features) * noise_std
                    augmented_features = features + noise
                
                if self.use_amp:
                    with autocast():
                        logits = self.model(augmented_features, lengths)
                else:
                    logits = self.model(augmented_features, lengths)
                
                if accumulated_logits is None:
                    accumulated_logits = torch.softmax(logits, dim=1)
                else:
                    accumulated_logits += torch.softmax(logits, dim=1)
            
            # Promediar
            avg_probs = accumulated_logits / num_augments
            preds = torch.argmax(avg_probs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
        
        if self.ema:
            self.ema.restore()
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        accuracy = np.mean(all_preds == all_targets)
        
        return accuracy


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento REGULARIZADO")
    
    # Paths
    parser.add_argument("--features_dir", type=Path, default=None)
    parser.add_argument("--metadata_path", type=Path, default=None)
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("models/temporal_regularized"))
    
    # Model
    parser.add_argument("--model_type", type=str, default="lstm")
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--use_attention", action="store_true", default=True)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Regularizacion
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--use_mixup", action="store_true", default=True)
    parser.add_argument("--use_rdrop", action="store_true", default=False)
    parser.add_argument("--label_smoothing", type=float, default=None)
    
    # Split
    parser.add_argument("--train_split", type=float, default=0.7)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--random_seed", type=int, default=42)
    
    # Flags
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_tta", action="store_true", default=True)
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--print_config", action="store_true")
    
    args = parser.parse_args()
    
    # Print config
    if args.print_config:
        config.print_regularization_summary()
        return
    
    # Defaults
    features_dir = args.features_dir or config.data_paths.features_fused
    metadata_path = args.metadata_path or config.data_paths.dataset_meta
    batch_size = args.batch_size or config.training.batch_size
    num_epochs = args.num_epochs or config.training.num_epochs
    learning_rate = args.learning_rate or config.training.learning_rate
    weight_decay = args.weight_decay or config.training.weight_decay
    num_classes = args.num_classes or config.model.num_classes
    hidden_dim = args.hidden_dim or config.training.model_hidden_dim
    dropout = args.dropout or config.training.model_dropout
    label_smoothing = args.label_smoothing if args.label_smoothing is not None else config.training.label_smoothing
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO CON REGULARIZACION COMPLETA")
    print("="*60)
    
    config.print_regularization_summary()
    
    # Analisis
    print("\n[1/4] Analizando distribucion de clases...")
    analyze_class_distribution(features_dir, metadata_path)
    
    if args.analyze_only:
        return
    
    # Dataloaders
    print("\n[2/4] Creando dataloaders con split por VIDEO...")
    train_loader, val_loader, test_loader, split_stats = create_temporal_dataloaders_fixed(
        features_dir=features_dir,
        metadata_path=metadata_path,
        batch_size=batch_size,
        num_workers=args.num_workers,
        train_split=args.train_split,
        val_split=args.val_split,
        random_seed=args.random_seed
    )
    
    # Trainer
    print("\n[3/4] Iniciando entrenamiento...")
    trainer = RegularizedTrainer(
        model_type=args.model_type,
        num_classes=num_classes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        device=args.device,
        use_amp=True,
        checkpoint_dir=args.checkpoint_dir,
        use_attention=args.use_attention,
        use_ema=args.use_ema,
        use_mixup=args.use_mixup,
        use_rdrop=args.use_rdrop,
        label_smoothing=label_smoothing,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    
    # Train
    best_acc = trainer.train(train_loader, val_loader, split_stats)
    
    # Test
    print("\n[4/4] Evaluacion final en test set...")
    
    # Sin TTA
    test_loss, test_acc, test_top5 = trainer.validate(test_loader, use_ema=trainer.use_ema)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Top-5 Accuracy: {test_top5:.4f}")
    
    # Con TTA
    if args.use_tta:
        tta_acc = trainer.evaluate_with_tta(test_loader, num_augments=config.evaluation.tta_num_augments)
        print(f"Test Accuracy (TTA): {tta_acc:.4f}")
    
    # Guardar resultados
    results = {
        'best_val_accuracy': float(best_acc),
        'test_accuracy': float(test_acc),
        'test_top5_accuracy': float(test_top5),
        'test_tta_accuracy': float(tta_acc) if args.use_tta else None,
        'regularization': {
            'weight_decay': weight_decay,
            'dropout': dropout,
            'label_smoothing': label_smoothing,
            'use_ema': args.use_ema,
            'use_mixup': args.use_mixup,
            'use_rdrop': args.use_rdrop
        }
    }
    
    results_path = args.checkpoint_dir / "final_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResultados guardados en: {results_path}")
    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60)


if __name__ == "__main__":
    main()
