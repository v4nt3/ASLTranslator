"""
Módulo de evaluación con métricas completas
Incluye: accuracy macro/micro, top-k accuracy, matriz de confusión, reporte por clase
"""

import torch # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from tqdm import tqdm # type: ignore
import logging

from config import config
from pipelines.models import get_model
from pipelines.datasets import ASLDataset

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluador del modelo multimodal"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.training.device)
        
        # Cargar modelo entrenado
        self.model = get_model(config).to(self.device)
        
        # Cargar checkpoint
        checkpoint_path = config.model_paths.best_macro
        if not checkpoint_path.exists():
            checkpoint_path = config.model_paths.final_model
        
        if checkpoint_path.exists():
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            logger.info(f"Modelo cargado desde {checkpoint_path}")
        else:
            logger.warning("No se encontró checkpoint. Evaluando con modelo sin entrenar.")
        
        self.model.eval()
    
    def evaluate(self):
        """Evaluación completa del modelo"""
        logger.info("Iniciando evaluación del modelo...")
        
        # Cargar datos
        if not self.config.data_paths.dataset_csv.exists():
            logger.error(f"Dataset CSV no encontrado: {self.config.data_paths.dataset_csv}")
            return
        
        df = pd.read_csv(self.config.data_paths.dataset_csv)
        
        # Usar subset para evaluación (últimas filas = test set)
        n = len(df)
        test_ratio = self.config.data.test_split
        train_val_ratio = self.config.data.train_split + self.config.data.val_split
        test_size = int(n * test_ratio)
        
        df_test = df[-test_size:]
        
        # Dataset
        test_dataset = ASLDataset(df_test, self.config.data_paths.clips, augmentor=None)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.evaluation.batch_size,
            shuffle=False,
            num_workers=self.config.evaluation.num_workers
        )
        
        # Evaluar
        all_preds, all_targets, all_probs = self._predict_all(test_loader)
        
        # Calcular métricas
        metrics = self._compute_metrics(all_preds, all_targets, all_probs)
        
        # Generar reportes
        self._generate_reports(all_preds, all_targets, metrics)
        
        logger.info("Evaluación completada!")
    
    def _predict_all(self, test_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Realiza predicciones en todo el test set"""
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            with tqdm(test_loader, desc="Evaluating") as pbar:
                for frames, keypoints, targets in pbar:
                    frames = frames.to(self.device)
                    keypoints = keypoints.to(self.device)
                    
                    logits = self.model(frames, keypoints)
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(logits, dim=1)
                    
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(targets.numpy())
                    all_probs.append(probs.cpu().numpy())
                    
                    pbar.update(1)
        
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        all_probs = np.concatenate(all_probs)
        
        return all_preds, all_targets, all_probs
    
    def _compute_metrics(self, preds: np.ndarray, targets: np.ndarray, 
                        probs: np.ndarray) -> Dict[str, float]:
        """Calcula todas las métricas de evaluación"""
        metrics = {}
        
        # Accuracy micro (global)
        accuracy_micro = np.mean(preds == targets)
        metrics['accuracy_micro'] = accuracy_micro
        logger.info(f"Micro accuracy: {accuracy_micro:.4f}")
        
        # Accuracy macro (por clase)
        class_accuracies = []
        for class_id in np.unique(targets):
            mask = targets == class_id
            class_acc = np.mean(preds[mask] == targets[mask])
            class_accuracies.append(class_acc)
        
        accuracy_macro = np.mean(class_accuracies)
        metrics['accuracy_macro'] = accuracy_macro
        logger.info(f"Macro accuracy: {accuracy_macro:.4f}")
        
        # Top-K accuracy
        for k in self.config.evaluation.top_k_accuracy:
            top_k_acc = self._compute_top_k_accuracy(probs, targets, k)
            metrics[f'top_{k}_accuracy'] = top_k_acc
            logger.info(f"Top-{k} accuracy: {top_k_acc:.4f}")
        
        return metrics
    
    def _compute_top_k_accuracy(self, probs: np.ndarray, targets: np.ndarray, k: int) -> float:
        """Calcula top-k accuracy"""
        top_k_preds = np.argsort(probs, axis=1)[:, -k:]
        correct = np.any(top_k_preds == targets[:, np.newaxis], axis=1)
        return np.mean(correct)
    
    def _generate_reports(self, preds: np.ndarray, targets: np.ndarray, 
                         metrics: Dict[str, float]):
        """Genera reportes de evaluación"""
        
        # Reporte JSON
        report = {
            'metrics': metrics,
            'num_classes': len(np.unique(targets)),
            'num_samples': len(targets),
            'per_class_metrics': self._compute_per_class_metrics(preds, targets)
        }
        
        report_path = self.config.output_paths.evaluation_report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Reporte guardado en {report_path}")
        
        # Matriz de confusión
        if self.config.evaluation.compute_confusion_matrix:
            self._plot_confusion_matrix(preds, targets)
    
    def _compute_per_class_metrics(self, preds: np.ndarray, 
                                   targets: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Calcula métricas por clase"""
        per_class = {}
        
        for class_id in np.unique(targets):
            mask = targets == class_id
            
            class_preds = preds[mask]
            class_targets = targets[mask]
            
            accuracy = np.mean(class_preds == class_targets)
            
            # Precision, recall, f1
            tp = np.sum((class_preds == class_id) & (class_targets == class_id))
            fp = np.sum((class_preds == class_id) & (class_targets != class_id))
            fn = np.sum((class_preds != class_id) & (class_targets == class_id))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class[int(class_id)] = {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'num_samples': int(np.sum(mask))
            }
        
        return per_class
    
    def _plot_confusion_matrix(self, preds: np.ndarray, targets: np.ndarray):
        """Grafica la matriz de confusión"""
        from sklearn.metrics import confusion_matrix # type: ignore
        
        cm = confusion_matrix(targets, preds)
        
        # Plot
        plt.figure(figsize=(20, 20))
        sns.heatmap(cm, cmap='Blues', cbar=True, square=True)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        cm_path = self.config.output_paths.confusion_matrix
        plt.savefig(cm_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Matriz de confusión guardada en {cm_path}")


class ResultVisualizer:
    """Visualiza resultados de entrenamiento y evaluación"""
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], output_path: Path):
        """Grafica el historial de entrenamiento"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid()
        
        # Accuracy
        axes[1].plot(history['val_accuracy_macro'], label='Macro Accuracy')
        axes[1].plot(history['val_accuracy_micro'], label='Micro Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfica de historia guardada en {output_path}")
