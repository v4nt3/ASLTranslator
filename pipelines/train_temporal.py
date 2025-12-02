"""
Script de entrenamiento optimizado para modelo temporal
Entrena SOLO la parte temporal (LSTM/Transformer + Classifier)
Carga directamente desde carpeta features_fused sin necesidad de CSV
Todas las configuraciones se manejan desde config.py
"""

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.cuda.amp import GradScaler, autocast # type: ignore
import numpy as np # type: ignore
from pathlib import Path
from tqdm import tqdm # type: ignore
import logging
import json
from typing import Dict, Tuple

from pipelines.models_temporal import get_temporal_model
from pipelines.dataset_temporal import create_temporal_dataloaders, load_data_from_folder, temporal_collate_fn, TemporalFeaturesDataset
from pipelines.evaluate_temporal import evaluate_model_comprehensive
from pipelines.augmentation_temporal import create_augmented_dataloaders  # Import augmentation

from config import config

def setup_logging(checkpoint_dir: Path):
    """Configura logging para consola y archivo"""
    # Crear directorio de checkpoints si no existe
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Archivo de log
    log_file = checkpoint_dir / "training.log"
    
    # Limpiar handlers anteriores
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    
    # Formato
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Handler para archivo
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Agregar handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    logger.info(f"Logging iniciado - guardando en: {log_file}")
    
    return logger

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
        checkpoint_dir: Path = None,
        use_attention: bool = False,
        # num_attention_heads: int = None
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
        # if num_attention_heads is None:  # Default attention heads
        #     num_attention_heads = config.model.num_attention_heads_lstm
        
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        global logger
        logger = setup_logging(self.checkpoint_dir)
        
        self.model = get_temporal_model(
            model_type=model_type,
            num_classes=num_classes,
            use_attention=use_attention,
            # num_attention_heads=num_attention_heads
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
        
        logger.info(f"TemporalTrainer inicializado")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Model type: {model_type}")
        logger.info(f"   Num classes: {num_classes}")
        logger.info(f"   Use attention: {use_attention}")  # Log attention
        # logger.info(f"   Num attention heads: {num_attention_heads}")  # Log attention heads
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
    
    def train(self, train_loader, val_loader, early_stopping_patience: int = 10):
        """Loop de entrenamiento completo"""
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
            
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['train_top5_accuracy'].append(train_top5)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['val_top5_accuracy'].append(val_top5)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            logger.info(f"   Early Stopping Patience: {patience_counter}/{early_stopping_patience}")
            logger.info(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train Top-5: {train_top5:.4f}")
            logger.info(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val Top-5: {val_top5:.4f}")
            logger.info(f"   LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_accuracy > self.best_val_acc:
                self.best_val_acc = val_accuracy
                best_path = self.checkpoint_dir / "best_model.pt"
                torch.save(self.model.state_dict(), best_path)
                logger.info(f"   Best model saved! Accuracy: {val_accuracy:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint every 15 epochs
            if (epoch + 1) % 15 == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_accuracy': val_accuracy,
                }, checkpoint_path)

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best Val Acc: {self.best_val_acc:.4f}% at epoch {epoch - early_stopping_patience}")
                break
        
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
    logger = setup_logging(config.model_paths.temporal_checkpoints)
    
    logger.info("Configuración cargada desde config.py:")
    logger.info(f"  Model type: {config.training.model_type}")
    logger.info(f"  Use attention: {config.training.use_attention}")
    logger.info(f"  Use augmentation: {config.training.use_augmentation}")
    logger.info(f"  Batch size: {config.training.batch_size}")
    logger.info(f"  Num epochs: {config.training.num_epochs}")
    logger.info(f"  Learning rate: {config.training.learning_rate}")
    logger.info(f"  Device: {config.training.device}")
    logger.info(f"  Train split: {config.data.train_split}")
    logger.info(f"  Val split: {config.data.val_split}")
    logger.info(f"  Test split: {config.data.test_split}")
    
    features_dir = config.data_paths.features_fused
    metadata_path = config.data_paths.dataset_meta
    
    if not features_dir.exists():
        logger.error(f"Directorio de features no encontrado: {features_dir}")
        return
    
    if not metadata_path.exists():
        logger.error(f"Archivo de metadata no encontrado: {metadata_path}")
        return
    
    logger.info(f"Cargando dataset directamente desde: {features_dir}")
    logger.info(f"Usando metadata: {metadata_path}")
    
    if config.training.use_augmentation:
        logger.info(f"Augmentation activada SOLO para train split (probabilidad: {config.data.temporal_augment_prob})")
        
        # Cargar datos
        train_data, val_data, test_data = load_data_from_folder(
            features_dir=features_dir,
            metadata_path=metadata_path,
            train_split=config.data.train_split,
            val_split=config.data.val_split,
            random_seed=config.data.random_seed
        )
        
        # Crear datasets base
        train_paths = [item[0] for item in train_data]
        train_class_ids = [item[1] for item in train_data]
        val_paths = [item[0] for item in val_data]
        val_class_ids = [item[1] for item in val_data]
        test_paths = [item[0] for item in test_data]
        test_class_ids = [item[1] for item in test_data]
        
        train_dataset = TemporalFeaturesDataset(train_paths, train_class_ids)
        val_dataset = TemporalFeaturesDataset(val_paths, val_class_ids)
        test_dataset = TemporalFeaturesDataset(test_paths, test_class_ids)
        
        # Crear loaders - augmentation SOLO en train
        train_loader, val_loader, test_loader = create_augmented_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            augment_train=True,  # Solo train tiene augmentation
            augment_prob=config.data.temporal_augment_prob,
            collate_fn=temporal_collate_fn
        )
        logger.info(" Train loader: CON augmentation")
        logger.info(" Val loader: SIN augmentation")
        logger.info(" Test loader: SIN augmentation")
    else:
        # Sin augmentation
        logger.info("Augmentation desactivada para todos los splits")
        train_loader, val_loader, test_loader = create_temporal_dataloaders(
            features_dir=features_dir,
            metadata_path=metadata_path,
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            train_split=config.data.train_split,
            val_split=config.data.val_split
        )
    
    trainer = TemporalTrainer(
        model_type=config.training.model_type,
        num_classes=config.model.num_classes,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        num_epochs=config.training.num_epochs,
        device=config.training.device,
        use_amp=config.training.use_amp,
        checkpoint_dir=config.model_paths.temporal_checkpoints,
        use_attention=config.training.use_attention
    )
    
    # Entrenar
    trainer.train(train_loader, val_loader, early_stopping_patience=config.training.early_stopping_patience)
    
    if config.training.run_final_evaluation:
        logger.info("\n" + "="*60)
        logger.info("Iniciando evaluación completa del modelo...")
        logger.info("="*60 + "\n")
        
        # Cargar el mejor modelo
        best_model_path = config.model_paths.temporal_checkpoints / "best_model.pt"
        trainer.model.load_state_dict(torch.load(best_model_path))
        
        logger.info("Evaluando en test set...")
        evaluate_model_comprehensive(
            model=trainer.model,
            test_loader=test_loader,  # Ahora usa test_loader
            device=trainer.device,
            num_classes=trainer.num_classes,
            save_dir=config.model_paths.temporal_checkpoints / "evaluation_results",
            history_path=config.model_paths.temporal_checkpoints / "training_history.json"
        )


if __name__ == "__main__":
    main()
