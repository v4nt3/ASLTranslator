"""
Script de entrenamiento CORREGIDO - Sin data leakage

CAMBIOS PRINCIPALES:
1. Usa dataset_temporal_fixed con split a nivel de VIDEO
2. Guarda estadisticas del split para verificacion
3. Mejor logging del proceso

NOTA: El accuracy real sera menor que el anterior (~82%) porque
ahora el modelo no puede "hacer trampa" viendo clips del mismo video
en train y val/test.
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
from typing import Tuple
import argparse

from pipelines.models_temporal import get_temporal_model
from pipelines.dataset_temporal import (
    create_temporal_dataloaders_fixed,
    analyze_class_distribution
)

from config import config


def setup_logging(checkpoint_dir: Path):
    """Configura logging para consola y archivo"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file = checkpoint_dir / "training.log"
    
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    logger.info(f"Logging iniciado - guardando en: {log_file}")
    
    return logger


logger = logging.getLogger(__name__)


class TemporalTrainerFixed:
    """
    Entrenador CORREGIDO que usa split a nivel de video.
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
        use_attention: bool = True
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
        
        global logger
        logger = setup_logging(self.checkpoint_dir)
        
        logger.info("="*60)
        logger.info("ENTRENAMIENTO CORREGIDO - SIN DATA LEAKAGE")
        logger.info("="*60)
        
        # Crear modelo
        self.model = get_temporal_model(
            model_type=model_type,
            num_classes=num_classes,
            hidden_dim=config.training.model_hidden_dim,
            num_layers=config.training.model_num_layers,
            dropout=config.training.model_dropout,
            bidirectional=config.training.model_bidirectional,
            use_attention=use_attention
        ).to(self.device)
        
        # Optimizer con AdamW (mejor regularizacion)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
            threshold=1e-4,
            min_lr=config.training.scheduler_min_lr
        )
        
        # Loss sin label smoothing (puede causar problemas con clases desbalanceadas)
        self.criterion = nn.CrossEntropyLoss()
        
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
        self.split_stats = None
        
        logger.info(f"Configuracion:")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Model type: {model_type}")
        logger.info(f"   Num classes: {num_classes}")
        logger.info(f"   Learning rate: {learning_rate}")
        logger.info(f"   Weight decay: {weight_decay}")
        logger.info(f"   Dropout: {config.training.model_dropout}")
        logger.info(f"   Use attention: {use_attention}")
        logger.info(f"   Use AMP: {use_amp}")
    
    def train(self, train_loader, val_loader, split_stats: dict = None):
        """Loop de entrenamiento"""
        
        if split_stats:
            self.split_stats = split_stats
            # Guardar estadisticas del split
            stats_path = self.checkpoint_dir / "split_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(split_stats, f, indent=2)
            logger.info(f"Estadisticas del split guardadas en: {stats_path}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Iniciando entrenamiento por {self.num_epochs} epochs")
        logger.info(f"{'='*60}\n")
        
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            
            train_loss, train_acc, train_top5 = self.train_epoch(train_loader)
            val_loss, val_accuracy, val_top5 = self.validate(val_loader)
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Guardar historia
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['train_top5_accuracy'].append(train_top5)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['val_top5_accuracy'].append(val_top5)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Log
            logger.info(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Top-5: {train_top5:.4f}")
            logger.info(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val Top-5: {val_top5:.4f}")
            logger.info(f"   LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            logger.info(f"   Patience: {patience_counter}/{config.training.early_stopping_patience}")
            
            # Guardar mejor modelo
            if val_accuracy > self.best_val_acc:
                self.best_val_acc = val_accuracy
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                    'val_top5_accuracy': val_top5,
                    'split_stats': self.split_stats
                }, best_path)
                logger.info(f"   [BEST] Modelo guardado! Accuracy: {val_accuracy:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Checkpoint periodico
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                    'history': self.history
                }, checkpoint_path)
            
            # Early stopping
            if patience_counter >= config.training.early_stopping_patience:
                logger.info(f"\nEarly stopping! No hay mejora en {config.training.early_stopping_patience} epochs.")
                break
        
        # Guardar modelo final
        final_path = self.checkpoint_dir / "final_model.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'val_accuracy': self.best_val_acc,
            'history': self.history,
            'split_stats': self.split_stats
        }, final_path)
        
        # Guardar historia
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"ENTRENAMIENTO COMPLETADO")
        logger.info(f"   Best Val Accuracy: {self.best_val_acc:.4f}")
        logger.info(f"   Checkpoints en: {self.checkpoint_dir}")
        logger.info(f"{'='*60}\n")
        
        return self.best_val_acc
    
    def train_epoch(self, train_loader) -> Tuple[float, float, float]:
        """Entrena una epoca"""
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
                
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
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
    def validate(self, val_loader) -> Tuple[float, float, float]:
        """Valida el modelo"""
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


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento CORREGIDO sin data leakage")
    
    parser.add_argument("--features_dir", type=Path, default=None,
                       help=f"Directorio con features fusionadas")
    parser.add_argument("--metadata_path", type=Path, default=None,
                       help=f"Ruta a dataset_meta.json")
    parser.add_argument("--checkpoint_dir", type=Path, default=None,
                       help=f"Directorio para checkpoints")
    
    # Model
    parser.add_argument("--model_type", type=str, default="lstm",
                       choices=["lstm", "transformer"])
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--use_attention", action="store_true", default=True)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    
    # Split
    parser.add_argument("--train_split", type=float, default=0.7,
                       help="Proporcion de VIDEOS para train")
    parser.add_argument("--val_split", type=float, default=0.15,
                       help="Proporcion de VIDEOS para val")
    parser.add_argument("--random_seed", type=int, default=42)
    
    # Flags
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"])
    parser.add_argument("--analyze_only", action="store_true",
                       help="Solo analizar distribucion, no entrenar")
    
    args = parser.parse_args()
    
    # Defaults from config
    features_dir = args.features_dir or config.data_paths.features_fused
    metadata_path = args.metadata_path or config.data_paths.dataset_meta
    checkpoint_dir = args.checkpoint_dir or Path("models/temporal_fixed")
    batch_size = args.batch_size or config.training.batch_size
    num_epochs = args.num_epochs or config.training.num_epochs
    learning_rate = args.learning_rate or config.training.learning_rate
    weight_decay = args.weight_decay or config.training.weight_decay
    num_workers = args.num_workers or config.training.num_workers
    num_classes = args.num_classes or config.model.num_classes
    
    # Analisis de distribucion
    print("\n" + "="*60)
    print("PASO 1: Analisis de distribucion de clases")
    print("="*60)
    
    analysis = analyze_class_distribution(features_dir, metadata_path)
    
    if args.analyze_only:
        print("\nAnalisis completado. Use --analyze_only=False para entrenar.")
        return
    
    # Crear dataloaders con split CORREGIDO
    print("\n" + "="*60)
    print("PASO 2: Creando dataloaders con split a nivel de VIDEO")
    print("="*60)
    
    train_loader, val_loader, test_loader, split_stats = create_temporal_dataloaders_fixed(
        features_dir=features_dir,
        metadata_path=metadata_path,
        batch_size=batch_size,
        num_workers=num_workers,
        train_split=args.train_split,
        val_split=args.val_split,
        random_seed=args.random_seed
    )
    
    # Crear trainer
    print("\n" + "="*60)
    print("PASO 3: Iniciando entrenamiento")
    print("="*60)
    
    trainer = TemporalTrainerFixed(
        model_type=args.model_type,
        num_classes=num_classes,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        device=args.device,
        use_amp=True,
        checkpoint_dir=checkpoint_dir,
        use_attention=args.use_attention
    )
    
    # Entrenar
    best_acc = trainer.train(train_loader, val_loader, split_stats)
    
    # Evaluar en test
    print("\n" + "="*60)
    print("PASO 4: Evaluacion final en test set")
    print("="*60)
    
    test_loss, test_acc, test_top5 = trainer.validate(test_loader)
    
    print(f"\nResultados en TEST SET (videos nunca vistos):")
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Test Top-5 Accuracy: {test_top5:.4f}")
    
    # Guardar resultados finales
    results = {
        'best_val_accuracy': float(best_acc),
        'test_accuracy': float(test_acc),
        'test_top5_accuracy': float(test_top5),
        'test_loss': float(test_loss),
        'split_stats': split_stats,
        'training_config': {
            'model_type': args.model_type,
            'num_classes': num_classes,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'train_split': args.train_split,
            'val_split': args.val_split
        }
    }
    
    results_path = checkpoint_dir / "final_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResultados guardados en: {results_path}")
    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO SIN DATA LEAKAGE")
    print("="*60)


if __name__ == "__main__":
    main()
