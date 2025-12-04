"""
Script de entrenamiento optimizado para modelo temporal
Entrena SOLO la parte temporal (LSTM/Transformer + Classifier)
Carga directamente desde carpeta features_fused sin necesidad de CSV
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
from typing import Dict, Tuple
import argparse

from pipelines.models_temporal import get_temporal_model
from pipelines.dataset_temporal import create_temporal_dataloaders
from pipelines.evaluate_temporal import evaluate_model_comprehensive

from config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TemporalTrainer:
    """Entrenador para modelos temporales con features precomputadas"""
    
    def __init__(
        self,
        model_type: str = "lstm",
        num_classes: int = None,
        learning_rate: float = None,
        weight_decay: float = None,
        num_epochs: int = None,
        device: str = None,
        use_amp: bool = None,
        checkpoint_dir: Path = None
    ):
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
            checkpoint_dir = config.model_paths.temporal_checkpoints
        
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Modelo
        self.model = get_temporal_model(
            model_type=model_type,
            num_classes=num_classes
        ).to(self.device)
        
         # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',
            factor=0.3,
            patience=5,
            threshold=1e-4,
            min_lr=1e-6
        )
        
        # Loss
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
        
        # AMP
        self.scaler = GradScaler() if use_amp else None
        
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
        
        logger.info(f"TemporalTrainer inicializado")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Model type: {model_type}")
        logger.info(f"   Num classes: {num_classes}")
        logger.info(f"   Use AMP: {use_amp}")
    
    def train_epoch(self, train_loader) -> Tuple[float, float, float]:
        """Entrena una epoca y retorna loss, accuracy y top5_accuracy"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_logits = []
        num_batches = 0
        
        with tqdm(train_loader, desc="Training") as pbar:
            for features, targets, lengths in pbar:
                features = features.to(self.device)
                targets = targets.to(self.device)
                lengths = lengths.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward
                if self.use_amp:
                    with autocast():
                        logits = self.model(features, lengths)
                        loss = self.criterion(logits, targets)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(features, lengths)
                    loss = self.criterion(logits, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Guardar predicciones para calcular accuracy
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_logits.append(logits.detach().cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calcular métricas de entrenamiento
        avg_loss = total_loss / num_batches
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_logits = np.concatenate(all_logits)
        
        train_accuracy = np.mean(all_preds == all_targets)
        train_top5_accuracy = self._compute_top5_accuracy(all_logits, all_targets)
        
        return avg_loss, train_accuracy, train_top5_accuracy
    
    @torch.no_grad()
    def validate(self, val_loader) -> Tuple[float, float, float]:
        """Valida el modelo y retorna loss, accuracy y top5_accuracy"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_logits = []
        
        with tqdm(val_loader, desc="Validating") as pbar:
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
        
        # Calcular métricas
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_logits = np.concatenate(all_logits)
        
        accuracy = np.mean(all_preds == all_targets)
        top5_accuracy = self._compute_top5_accuracy(all_logits, all_targets)
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy, top5_accuracy
    
    def _compute_top5_accuracy(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Calcula top-5 accuracy"""
        top5_preds = np.argsort(logits, axis=1)[:, -5:]  # Top 5 predicciones
        correct = np.any(top5_preds == targets[:, None], axis=1)
        return np.mean(correct)
    
    def train(self, train_loader, val_loader):
        """Loop de entrenamiento completo"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Iniciando entrenamiento por {self.num_epochs} epochs")
        logger.info(f"{'='*60}\n")
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            train_loss, train_acc, train_top5 = self.train_epoch(train_loader)
            
            val_loss, val_accuracy, val_top5 = self.validate(val_loader)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['train_top5_accuracy'].append(train_top5)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['val_top5_accuracy'].append(val_top5)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            logger.info(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Top-5: {train_top5:.4f}")
            logger.info(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val Top-5: {val_top5:.4f}")
            logger.info(f"   LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_accuracy > self.best_val_acc:
                self.best_val_acc = val_accuracy
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(self.model.state_dict(), best_path)
                logger.info(f"   ★ Best model saved! Accuracy: {val_accuracy:.4f}")
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                }, checkpoint_path)
        
        # Save final model
        final_path = self.checkpoint_dir / "final_model.pt"
        torch.save(self.model.state_dict(), final_path)
        
        # Save history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Entrenamiento completado!")
        logger.info(f"   Best Val Accuracy: {self.best_val_acc:.4f}")
        logger.info(f"   Checkpoints guardados en: {self.checkpoint_dir}")
        logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de modelo temporal optimizado")
    
    parser.add_argument("--features_dir", type=Path, default=None,
                       help=f"Directorio con features fusionadas (default: {config.data_paths.features_fused})")
    parser.add_argument("--metadata_path", type=Path, default=None,
                       help=f"Ruta a dataset_meta.json (default: {config.data_paths.dataset_meta})")
    parser.add_argument("--checkpoint_dir", type=Path, default=None,
                       help=f"Directorio para guardar checkpoints (default: {config.model_paths.temporal_checkpoints})")
    
    # Model
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "transformer"],
                       help="Tipo de modelo temporal")
    parser.add_argument("--num_classes", type=int, default=None,
                       help=f"Numero de clases (default: {config.model.num_classes})")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=None,
                       help=f"Batch size (default: {config.training.batch_size})")
    parser.add_argument("--num_epochs", type=int, default=None,
                       help=f"Numero de epochs (default: {config.training.num_epochs})")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help=f"Learning rate (default: {config.training.learning_rate})")
    parser.add_argument("--weight_decay", type=float, default=None,
                       help=f"Weight decay (default: {config.training.weight_decay})")
    parser.add_argument("--num_workers", type=int, default=None,
                       help=f"Numero de workers para DataLoader (default: {config.training.num_workers})")
    
    # Data split
    parser.add_argument("--train_split", type=float, default=None,
                       help=f"Proporcion de datos para entrenamiento (default: {config.data.train_split})")
    parser.add_argument("--val_split", type=float, default=None,
                       help=f"Proporcion de datos para validacion (default: {config.data.val_split})")
    
    # Device
    parser.add_argument("--device", type=str, default=None,
                       help=f"Device (cuda o cpu) (default: {config.training.device})")
    parser.add_argument("--no_amp", action="store_true",
                       help="Deshabilitar AMP")
    
    parser.add_argument("--no_final_eval", action="store_true",
                       help="No ejecutar evaluación completa al finalizar entrenamiento")
    
    args = parser.parse_args()
    
    features_dir = args.features_dir or config.data_paths.features_fused
    metadata_path = args.metadata_path or config.data_paths.dataset_meta
    checkpoint_dir = args.checkpoint_dir or config.model_paths.temporal_checkpoints
    batch_size = args.batch_size or config.training.batch_size
    num_workers = args.num_workers or config.training.num_workers
    train_split = args.train_split or config.data.train_split
    val_split = args.val_split or config.data.val_split
    
    if not features_dir.exists():
        logger.error(f"Directorio de features no encontrado: {features_dir}")
        return
    
    if not metadata_path.exists():
        logger.error(f"Archivo de metadata no encontrado: {metadata_path}")
        return
    
    logger.info(f"Cargando dataset directamente desde: {features_dir}")
    logger.info(f"Usando metadata: {metadata_path}")
    
    # Crear DataLoaders (carga automática desde carpeta)
    train_loader, val_loader, test_loader = create_temporal_dataloaders(
        features_dir=features_dir,
        metadata_path=metadata_path,
        batch_size=batch_size,
        num_workers=num_workers,
        train_split=train_split,
        val_split=val_split
    )
    
    # Crear trainer
    trainer = TemporalTrainer(
        model_type=args.model_type,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        device=args.device,
        use_amp=not args.no_amp,
        checkpoint_dir=checkpoint_dir
    )
    
    # Entrenar
    trainer.train(train_loader, val_loader)
    
    if not args.no_final_eval:
        logger.info("\n" + "="*60)
        logger.info("Iniciando evaluación completa del modelo...")
        logger.info("="*60 + "\n")
        
        # Cargar el mejor modelo
        best_model_path = checkpoint_dir / "best_model.pt"
        trainer.model.load_state_dict(torch.load(best_model_path))
        
        # Ejecutar evaluación completa
        evaluate_model_comprehensive(
            model=trainer.model,
            test_loader=val_loader,  # Usar val_loader como test
            device=trainer.device,
            num_classes=trainer.num_classes,
            save_dir=checkpoint_dir / "evaluation_results",
            history_path=checkpoint_dir / "training_history.json"
        )


if __name__ == "__main__":
    main()
