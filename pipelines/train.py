"""
Script de entrenamiento avanzado con técnicas de optimización
Mantiene arquitectura simple pero agrega mejoras de entrenamiento
"""

import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.nn.functional as F #type: ignore
from torch.optim import AdamW #type: ignore
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau #type: ignore
from torch.cuda.amp import autocast, GradScaler #type: ignore
import numpy as np #type: ignore
from tqdm import tqdm #type: ignore
import json
import os
from pathlib import Path

from config_advanced_optimization import get_config #type: ignore
from models_temporal_optimized import TemporalLSTMClassifier #type: ignore
from dataset_temporal import TemporalFeatureDataset


# ============================================================
# UTILIDADES AVANZADAS
# ============================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy con label smoothing para mejor generalización"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(pred, dim=-1)
        
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        
        smooth_loss = -logprobs.mean(dim=-1)
        
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def mixup_data(x, y, alpha=0.2, device='cuda'):
    """Mixup temporal para secuencias"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss para mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class EMA:
    """Exponential Moving Average de los pesos del modelo"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class TemporalAugmentation:
    """Augmentations temporales para secuencias"""
    def __init__(self, shift_range=2, speed_range=(0.9, 1.1), reverse_prob=0.1):
        self.shift_range = shift_range
        self.speed_range = speed_range
        self.reverse_prob = reverse_prob
    
    def __call__(self, sequence):
        """sequence: (T, D)"""
        T, D = sequence.shape
        
        # Temporal shift
        if self.shift_range > 0:
            shift = np.random.randint(-self.shift_range, self.shift_range + 1)
            if shift > 0:
                sequence = torch.cat([sequence[shift:], sequence[-1].unsqueeze(0).repeat(shift, 1)])
            elif shift < 0:
                sequence = torch.cat([sequence[0].unsqueeze(0).repeat(-shift, 1), sequence[:shift]])
        
        # Temporal reverse (baja probabilidad)
        if np.random.random() < self.reverse_prob:
            sequence = torch.flip(sequence, dims=[0])
        
        return sequence


def get_lr_scheduler(optimizer, config, num_training_steps):
    """Obtiene el scheduler según configuración"""
    if config.USE_ONE_CYCLE:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.MAX_LR,
            total_steps=num_training_steps,
            pct_start=config.PCT_START,
            anneal_strategy='cos',
            cycle_momentum=True,
            base_momentum=0.85,
            max_momentum=0.95,
        )
        return scheduler, 'step'  # Step cada batch
    
    elif config.USE_REDUCE_LR:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            verbose=True
        )
        return scheduler, 'epoch'  # Step cada epoch
    
    return None, None


class WarmupScheduler:
    """Warmup para learning rate"""
    def __init__(self, optimizer, warmup_epochs, start_lr, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.start_lr + (self.target_lr - self.start_lr) * (self.current_epoch / self.warmup_epochs)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch += 1
    
    def is_warmup_done(self):
        return self.current_epoch >= self.warmup_epochs


# ============================================================
# ENTRENAMIENTO PRINCIPAL
# ============================================================

def train_epoch(model, dataloader, criterion, optimizer, device, config, 
                scaler, ema=None, temporal_aug=None, epoch=0):
    """Entrenamiento de una época con técnicas avanzadas"""
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} - Train")
    
    for batch_idx, (features, labels) in enumerate(pbar):
        features = features.to(device)
        labels = labels.to(device)
        
        # Temporal augmentation (opcional)
        if temporal_aug is not None and config.USE_TEMPORAL_AUG:
            batch_size = features.size(0)
            aug_features = []
            for i in range(batch_size):
                aug_seq = temporal_aug(features[i])
                aug_features.append(aug_seq)
            features = torch.stack(aug_features)
        
        # Mixup (opcional)
        use_mixup = config.USE_MIXUP and np.random.random() < config.MIXUP_PROB
        
        if use_mixup:
            features, labels_a, labels_b, lam = mixup_data(features, labels, config.MIXUP_ALPHA, device)
        
        # Forward pass con mixed precision
        with autocast():
            outputs = model(features)
            
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)
            
            # Normalizar por gradient accumulation
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # EMA update
            if ema is not None:
                ema.update()
        
        # Métricas
        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        
        if not use_mixup:
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        if not use_mixup:
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        else:
            pbar.set_postfix({
                'loss': f'{total_loss/(batch_idx+1):.4f}'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device, use_tta=False, tta_aug=None):
    """Validación con opcional TTA (Test Time Augmentation)"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc="Validation"):
            features = features.to(device)
            labels = labels.to(device)
            
            if use_tta and tta_aug is not None:
                # TTA: promedia predicciones de múltiples augmentations
                batch_size = features.size(0)
                tta_outputs = []
                
                # Original
                outputs = model(features)
                tta_outputs.append(F.softmax(outputs, dim=1))
                
                # Augmentations
                for _ in range(4):  # 4 augmentations adicionales
                    aug_features = []
                    for i in range(batch_size):
                        aug_seq = tta_aug(features[i])
                        aug_features.append(aug_seq)
                    aug_features = torch.stack(aug_features)
                    
                    outputs_aug = model(aug_features)
                    tta_outputs.append(F.softmax(outputs_aug, dim=1))
                
                # Promedio
                outputs = torch.stack(tta_outputs).mean(0)
                outputs = torch.log(outputs)  # Volver a log-probs para loss
            else:
                outputs = model(features)
            
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels


def progressive_unfreezing(model, config, epoch):
    """Descongelamiento progresivo de capas"""
    # Descongelar MLP de pose
    if epoch == config.UNFREEZE_POSE_MLP_EPOCH:
        print(f"\n[Epoch {epoch}] Unfreezing Pose MLP")
        if hasattr(model, 'pose_mlp'):
            for param in model.pose_mlp.parameters():
                param.requires_grad = True
        return True
    
    # Descongelar últimas capas de ResNet
    if epoch == config.UNFREEZE_RESNET_EPOCH:
        print(f"\n[Epoch {epoch}] Unfreezing ResNet layers")
        if hasattr(model, 'resnet'):
            # Descongelar últimas capas
            layers = list(model.resnet.children())
            for layer in layers[-config.RESNET_LAYERS_TO_UNFREEZE:]:
                for param in layer.parameters():
                    param.requires_grad = True
        return True
    
    return False


def adjust_lr_for_unfreezing(optimizer, config):
    """Ajusta LR cuando se descongelan capas"""
    base_lr = optimizer.param_groups[0]['lr']
    
    # Nuevos parámetros tienen LR más bajo
    for i, param_group in enumerate(optimizer.param_groups):
        if i > 0:  # Grupos agregados después
            param_group['lr'] = base_lr * config.FINE_TUNE_LR_MULTIPLIER


def main():
    # Configuración
    config = get_config(strategy="aggressive")  # "conservative", "simple", "aggressive"
    
    # Crear directorios
    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Datasets
    train_dataset = TemporalFeatureDataset(
        os.path.join(config.DATA_DIR, "train_clips.pkl"),
        mode='train'
    )
    val_dataset = TemporalFeatureDataset(
        os.path.join(config.DATA_DIR, "val_clips.pkl"),
        mode='val'
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    # Modelo
    num_classes = len(train_dataset.label_encoder.classes_)
    model = TemporalLSTMClassifier(
        input_dim=config.INPUT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        num_classes=num_classes,
        dropout=config.DROPOUT,
        bidirectional=config.BIDIRECTIONAL,
        use_attention=config.USE_ATTENTION,
        use_simple_classifier=config.USE_SIMPLE_CLASSIFIER,
        classifier_dropout=config.CLASSIFIER_DROPOUT
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss con label smoothing
    if config.LABEL_SMOOTHING > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.LABEL_SMOOTHING)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.INITIAL_LR,
        weight_decay=config.WEIGHT_DECAY,
        betas=config.BETAS,
        eps=config.EPS
    )
    
    # Scheduler
    num_training_steps = len(train_loader) * config.NUM_EPOCHS // config.GRADIENT_ACCUMULATION_STEPS
    scheduler, scheduler_type = get_lr_scheduler(optimizer, config, num_training_steps)
    
    # Warmup
    warmup_scheduler = None
    if config.USE_WARMUP:
        warmup_scheduler = WarmupScheduler(
            optimizer,
            config.WARMUP_EPOCHS,
            config.WARMUP_START_LR,
            config.INITIAL_LR
        )
    
    # EMA
    ema = None
    if config.USE_EMA:
        ema = EMA(model, decay=config.EMA_DECAY)
    
    # Mixed precision
    scaler = GradScaler()
    
    # Temporal augmentation
    temporal_aug = None
    if config.USE_TEMPORAL_AUG:
        temporal_aug = TemporalAugmentation(
            shift_range=config.TEMPORAL_SHIFT_RANGE,
            speed_range=config.TEMPORAL_SPEED_RANGE,
            reverse_prob=config.TEMPORAL_REVERSE_PROB
        )
    
    # TTA augmentation
    tta_aug = temporal_aug if config.USE_TTA else None
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # Warmup
        if warmup_scheduler is not None and not warmup_scheduler.is_warmup_done():
            warmup_scheduler.step()
            print(f"Warmup LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Progressive unfreezing
        if hasattr(config, 'UNFREEZE_POSE_MLP_EPOCH'):
            if progressive_unfreezing(model, config, epoch):
                adjust_lr_for_unfreezing(optimizer, config)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            config, scaler, ema, temporal_aug, epoch
        )
        
        # Scheduler step (si es por batch, ya se hizo en train_epoch)
        if scheduler is not None and scheduler_type == 'step':
            # Ya se hace en train_epoch por batch
            pass
        
        # Validación con modelo EMA si está disponible
        if ema is not None:
            ema.apply_shadow()
        
        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, device,
            use_tta=config.USE_TTA and epoch > config.NUM_EPOCHS // 2,  # TTA solo en segunda mitad
            tta_aug=tta_aug
        )
        
        if ema is not None:
            ema.restore()
        
        # Scheduler step (si es por epoch)
        if scheduler is not None and scheduler_type == 'epoch':
            scheduler.step(val_acc)
        
        # Guardar historia
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Guardar con EMA si está disponible
            if ema is not None:
                ema.apply_shadow()
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config.__dict__
            }
            
            torch.save(
                checkpoint,
                os.path.join(config.CHECKPOINT_DIR, 'best_model_advanced.pth')
            )
            
            if ema is not None:
                ema.restore()
            
            print(f"✓ New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Guardar historia
    with open(os.path.join(config.LOG_DIR, 'history_advanced.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
