"""
Script de evaluación completa con visualizaciones y métricas avanzadas
Genera matrices de confusión, curvas ROC, análisis de peores clases, etc.
INCLUYE: Métricas detalladas por clase con filtrado por umbral de accuracy
"""

import torch #type: ignore
import numpy as np #type: ignore
from pathlib import Path
import matplotlib.pyplot as plt #type: ignore
import seaborn as sns #type: ignore
from sklearn.metrics import ( #type: ignore
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, top_k_accuracy_score,
    precision_score, recall_score
)
from sklearn.preprocessing import label_binarize #type: ignore
import json
import pandas as pd #type: ignore
from typing import Dict, List, Tuple
from tqdm import tqdm #type: ignore

import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluador completo con visualizaciones y métricas avanzadas"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_classes: int,
        save_dir: Path
    ):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
    
    @torch.no_grad()
    def collect_predictions(self, dataloader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Recolecta todas las predicciones y targets"""
        all_logits = []
        all_preds = []
        all_targets = []
        
        logger.info("Recolectando predicciones...")
        
        with tqdm(dataloader, desc="Evaluating") as pbar:
            for features, targets, lengths in pbar:
                features = features.to(self.device)
                lengths = lengths.to(self.device)
                
                logits = self.model(features, lengths)
                preds = torch.argmax(logits, dim=1)
                
                all_logits.append(logits.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.numpy())
        
        all_logits = np.concatenate(all_logits)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        
        return all_logits, all_preds, all_targets
    
    def compute_metrics(self, logits: np.ndarray, preds: np.ndarray, targets: np.ndarray) -> Dict:
        """Calcula todas las métricas"""
        logger.info("Calculando métricas...")
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = np.mean(preds == targets)
        
        # Top-5 Accuracy
        top5_preds = np.argsort(logits, axis=1)[:, -5:]
        metrics['top5_accuracy'] = np.mean(np.any(top5_preds == targets[:, None], axis=1))
        
        # Top-10 Accuracy
        top10_preds = np.argsort(logits, axis=1)[:, -10:]
        metrics['top10_accuracy'] = np.mean(np.any(top10_preds == targets[:, None], axis=1))
        
        # F1 Score (macro y weighted)
        metrics['f1_macro'] = f1_score(targets, preds, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(targets, preds, average='weighted', zero_division=0)
        
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
        logger.info(f"Top-10 Accuracy: {metrics['top10_accuracy']:.4f}")
        logger.info(f"F1 Macro: {metrics['f1_macro']:.4f}")
        logger.info(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
        
        return metrics
    
    def compute_per_class_metrics(
        self, 
        targets: np.ndarray, 
        preds: np.ndarray,
        min_samples: int = 1
    ) -> List[Dict]:
        """
        Calcula métricas detalladas para cada clase individual
        
        Returns:
            Lista de diccionarios con métricas por clase
        """
        logger.info("Calculando métricas por clase...")
        
        unique_classes = np.unique(targets)
        per_class_metrics = []
        
        for cls in tqdm(unique_classes, desc="Computing per-class metrics"):
            mask = targets == cls
            num_samples = mask.sum()
            
            if num_samples < min_samples:
                continue
            
            # Obtener predicciones y targets para esta clase
            cls_targets = targets[mask]
            cls_preds = preds[mask]
            
            # Calcular métricas
            accuracy = np.mean(cls_preds == cls_targets)
            
            # F1, Precision, Recall para esta clase vs resto
            # Convertir a problema binario: clase actual vs resto
            binary_targets = (targets == cls).astype(int)
            binary_preds = (preds == cls).astype(int)
            
            precision = precision_score(binary_targets, binary_preds, zero_division=0)
            recall = recall_score(binary_targets, binary_preds, zero_division=0)
            f1 = f1_score(binary_targets, binary_preds, zero_division=0)
            
            # True Positives, False Positives, False Negatives
            tp = np.sum((binary_targets == 1) & (binary_preds == 1))
            fp = np.sum((binary_targets == 0) & (binary_preds == 1))
            fn = np.sum((binary_targets == 1) & (binary_preds == 0))
            tn = np.sum((binary_targets == 0) & (binary_preds == 0))
            
            per_class_metrics.append({
                'class_id': int(cls),
                'num_samples': int(num_samples),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn)
            })
        
        return per_class_metrics
    
    def filter_classes_by_threshold(
        self,
        per_class_metrics: List[Dict],
        threshold: float = 0.80,
        save_name: str = "classes_above_threshold"
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filtra clases por umbral de accuracy y guarda resultados
        
        Args:
            per_class_metrics: Lista de métricas por clase
            threshold: Umbral de accuracy (default: 0.80)
            save_name: Nombre base para archivos de salida
            
        Returns:
            Tupla con (clases_sobre_umbral, clases_bajo_umbral)
        """
        logger.info(f"Filtrando clases con accuracy >= {threshold:.1%}...")
        
        # Ordenar por accuracy (descendente)
        sorted_metrics = sorted(per_class_metrics, key=lambda x: x['accuracy'], reverse=True)
        
        # Filtrar por umbral
        above_threshold = [m for m in sorted_metrics if m['accuracy'] >= threshold]
        below_threshold = [m for m in sorted_metrics if m['accuracy'] < threshold]
        
        num_above = len(above_threshold)
        num_below = len(below_threshold)
        total = len(sorted_metrics)
        
        logger.info(f"Clases con accuracy >= {threshold:.1%}: {num_above}/{total} ({num_above/total:.1%})")
        logger.info(f"Clases con accuracy < {threshold:.1%}: {num_below}/{total} ({num_below/total:.1%})")
        
        # Guardar en JSON
        summary = {
            'threshold': threshold,
            'total_classes': total,
            'classes_above_threshold': num_above,
            'classes_below_threshold': num_below,
            'percentage_above': num_above / total if total > 0 else 0,
            'classes_above': above_threshold,
            'classes_below': below_threshold
        }
        
        json_path = self.save_dir / f"{save_name}_{int(threshold*100)}.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Resumen guardado en: {json_path}")
        
        # Guardar en CSV (más fácil de leer)
        csv_above_path = self.save_dir / f"{save_name}_{int(threshold*100)}_above.csv"
        csv_below_path = self.save_dir / f"{save_name}_{int(threshold*100)}_below.csv"
        
        if above_threshold:
            df_above = pd.DataFrame(above_threshold)
            df_above.to_csv(csv_above_path, index=False)
            logger.info(f"Clases sobre umbral guardadas en: {csv_above_path}")
        
        if below_threshold:
            df_below = pd.DataFrame(below_threshold)
            df_below.to_csv(csv_below_path, index=False)
            logger.info(f"Clases bajo umbral guardadas en: {csv_below_path}")
        
        return above_threshold, below_threshold
    
    def save_all_per_class_metrics(
        self,
        per_class_metrics: List[Dict],
        filename: str = "per_class_metrics_complete"
    ):
        """
        Guarda todas las métricas por clase en JSON y CSV
        """
        logger.info("Guardando métricas completas por clase...")
        
        # Ordenar por accuracy (descendente)
        sorted_metrics = sorted(per_class_metrics, key=lambda x: x['accuracy'], reverse=True)
        
        # Guardar en JSON
        json_path = self.save_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(sorted_metrics, f, indent=2)
        
        # Guardar en CSV
        csv_path = self.save_dir / f"{filename}.csv"
        df = pd.DataFrame(sorted_metrics)
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Métricas completas guardadas en:")
        logger.info(f"  - JSON: {json_path}")
        logger.info(f"  - CSV: {csv_path}")
        
        # Estadísticas resumidas
        logger.info("\nEstadísticas de métricas por clase:")
        logger.info(f"  - Accuracy media: {np.mean([m['accuracy'] for m in sorted_metrics]):.4f}")
        logger.info(f"  - Accuracy mediana: {np.median([m['accuracy'] for m in sorted_metrics]):.4f}")
        logger.info(f"  - F1 Score medio: {np.mean([m['f1_score'] for m in sorted_metrics]):.4f}")
        logger.info(f"  - Precision media: {np.mean([m['precision'] for m in sorted_metrics]):.4f}")
        logger.info(f"  - Recall medio: {np.mean([m['recall'] for m in sorted_metrics]):.4f}")
    
    def plot_threshold_analysis(
        self,
        per_class_metrics: List[Dict],
        thresholds: List[float] = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    ):
        """
        Visualiza cuántas clases superan diferentes umbrales de accuracy
        """
        logger.info("Generando análisis de umbrales...")
        
        sorted_metrics = sorted(per_class_metrics, key=lambda x: x['accuracy'], reverse=True)
        accuracies = [m['accuracy'] for m in sorted_metrics]
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # Gráfico 1: Distribución de accuracy con umbrales marcados
        ax1 = axes[0]
        ax1.plot(range(len(accuracies)), accuracies, linewidth=2, color='steelblue')
        ax1.fill_between(range(len(accuracies)), accuracies, alpha=0.3, color='steelblue')
        
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(thresholds)))
        for threshold, color in zip(thresholds, colors):
            num_above = sum(1 for acc in accuracies if acc >= threshold)
            ax1.axhline(y=threshold, color=color, linestyle='--', linewidth=2,
                       label=f'{threshold:.0%}: {num_above}/{len(accuracies)} classes ({num_above/len(accuracies):.1%})')
        
        ax1.set_xlabel('Class Rank (sorted by accuracy)', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Per-Class Accuracy Distribution with Thresholds', fontsize=14)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # Gráfico 2: Barras mostrando cantidad de clases por umbral
        ax2 = axes[1]
        counts = [sum(1 for acc in accuracies if acc >= t) for t in thresholds]
        percentages = [c / len(accuracies) * 100 for c in counts]
        
        bars = ax2.bar(range(len(thresholds)), counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_xticks(range(len(thresholds)))
        ax2.set_xticklabels([f'{t:.0%}' for t in thresholds], fontsize=11)
        ax2.set_xlabel('Accuracy Threshold', fontsize=12)
        ax2.set_ylabel('Number of Classes', fontsize=12)
        ax2.set_title('Classes Above Each Accuracy Threshold', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores en las barras
        for i, (bar, count, pct) in enumerate(zip(bars, counts, percentages)):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({pct:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.save_dir / 'threshold_analysis.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Análisis de umbrales guardado en: {save_path}")

    def plot_confusion_matrix(
        self,
        targets: np.ndarray,
        preds: np.ndarray,
        top_k: int = 50,
        normalize: bool = True
    ):
        """Genera matriz de confusión para las top-k clases más frecuentes"""
        logger.info(f"Generando matriz de confusión (top {top_k} clases)...")
        
        unique, counts = np.unique(targets, return_counts=True)
        top_classes = unique[np.argsort(counts)[-top_k:]]
        
        mask = np.isin(targets, top_classes)
        filtered_targets = targets[mask]
        filtered_preds = preds[mask]
        
        cm = confusion_matrix(filtered_targets, filtered_preds, labels=top_classes)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=False, fmt='.2f' if normalize else 'd', cmap='Blues', 
                   xticklabels=top_classes, yticklabels=top_classes, cbar_kws={'label': 'Proportion' if normalize else 'Count'})
        plt.title(f'Confusion Matrix - Top {top_k} Classes' + (' (Normalized)' if normalize else ''))
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        save_path = self.save_dir / f'confusion_matrix_top{top_k}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Matriz de confusión guardada en: {save_path}")
    
    def plot_training_history(self, history_path: Path):
        """Genera gráficas del historial de entrenamiento"""
        logger.info("Generando gráficas de historial de entrenamiento...")
        
        if not history_path.exists():
            logger.warning(f"No se encontró historial en: {history_path}")
            return
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(epochs, history['train_top5_accuracy'], 'b-', label='Train Top-5', linewidth=2)
        axes[1, 0].plot(epochs, history['val_top5_accuracy'], 'r-', label='Val Top-5', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-5 Accuracy')
        axes[1, 0].set_title('Training and Validation Top-5 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        save_path = self.save_dir / 'training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Historial de entrenamiento guardado en: {save_path}")
    
    def analyze_worst_classes(
        self,
        targets: np.ndarray,
        preds: np.ndarray,
        top_k: int = 20,
        min_samples: int = 5
    ):
        """Analiza las peores clases"""
        logger.info(f"Analizando las {top_k} peores clases...")
        
        unique_classes = np.unique(targets)
        class_accuracies = []
        
        for cls in unique_classes:
            mask = targets == cls
            if mask.sum() < min_samples:
                continue
            
            acc = np.mean(preds[mask] == targets[mask])
            class_accuracies.append({
                'class_name': int(cls),
                'accuracy': float(acc),
                'num_samples': int(mask.sum())
            })
        
        class_accuracies = sorted(class_accuracies, key=lambda x: x['accuracy'])
        worst_classes = class_accuracies[:top_k]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        class_names = [c['class_name'] for c in worst_classes]
        accuracies = [c['accuracy'] for c in worst_classes]
        num_samples = [c['num_samples'] for c in worst_classes]
        
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(class_names)))
        bars = ax.barh(range(len(class_names)), accuracies, color=colors)
        
        ax.set_yticks(range(len(class_names)))
        ax.set_yticklabels([f"Class {cid} (n={n})" for cid, n in zip(class_names, num_samples)])
        ax.set_xlabel('Accuracy')
        ax.set_title(f'Top {top_k} Worst Performing Classes')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(acc + 0.01, i, f'{acc:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        save_path = self.save_dir / 'worst_classes.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        worst_classes_json = self.save_dir / 'worst_classes.json'
        with open(worst_classes_json, 'w') as f:
            json.dump(worst_classes, f, indent=2)
        
        logger.info(f"Peores clases guardadas en: {save_path}")
        logger.info(f"Detalles en: {worst_classes_json}")
    
    def plot_roc_curves(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        num_classes_to_plot: int = 15
    ):
        """Genera curvas ROC para algunas clases"""
        logger.info(f"Generando curvas ROC para {num_classes_to_plot} clases...")
        
        unique, counts = np.unique(targets, return_counts=True)
        top_classes = unique[np.argsort(counts)[-num_classes_to_plot:]]
        
        targets_bin = label_binarize(targets, classes=range(self.num_classes))
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        
        plt.figure(figsize=(12, 10))
        
        for cls in top_classes:
            fpr, tpr, _ = roc_curve(targets_bin[:, cls], probs[:, cls])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, linewidth=2, label=f'Class {cls} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curves - Top {num_classes_to_plot} Classes', fontsize=14)
        plt.legend(loc="lower right", fontsize=9)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.save_dir / 'roc_curves.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Curvas ROC guardadas en: {save_path}")
    
    def plot_per_class_accuracy(self, targets: np.ndarray, preds: np.ndarray):
        """Gráfica de distribución de accuracy por clase"""
        logger.info("Generando distribución de accuracy por clase...")
        
        unique_classes = np.unique(targets)
        accuracies = []
        
        for cls in unique_classes:
            mask = targets == cls
            if mask.sum() > 0:
                acc = np.mean(preds[mask] == targets[mask])
                accuracies.append(acc)
        
        plt.figure(figsize=(12, 6))
        plt.hist(accuracies, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        plt.xlabel('Accuracy', fontsize=12)
        plt.ylabel('Number of Classes', fontsize=12)
        plt.title('Distribution of Per-Class Accuracy', fontsize=14)
        plt.axvline(np.mean(accuracies), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(accuracies):.3f}')
        plt.axvline(np.median(accuracies), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(accuracies):.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / 'per_class_accuracy_distribution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Distribución de accuracy guardada en: {save_path}")
    
    def generate_report(self, metrics: Dict, save_path: Path = None):
        """Genera reporte completo en texto"""
        if save_path is None:
            save_path = self.save_dir / 'evaluation_report.txt'
        
        logger.info(f"Generando reporte de evaluación...")
        
        with open(save_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("Overall Metrics:\n")
            f.write("-"*60 + "\n")
            for key, value in metrics.items():
                f.write(f"{key:20s}: {value:.6f}\n")
            f.write("\n")
        
        logger.info(f"Reporte guardado en: {save_path}")


def evaluate_model_comprehensive(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    num_classes: int,
    save_dir: Path,
    history_path: Path = None,
    accuracy_thresholds: List[float] = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
):
    """
    Función principal de evaluación completa
    
    Args:
        accuracy_thresholds: Lista de umbrales de accuracy para análisis
    """
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = ModelEvaluator(model, device, num_classes, save_dir)
    
    # Recolectar predicciones
    logits, preds, targets = evaluator.collect_predictions(test_loader)
    
    # Calcular métricas globales
    metrics = evaluator.compute_metrics(logits, preds, targets)
    
    per_class_metrics = evaluator.compute_per_class_metrics(targets, preds)
    
    evaluator.save_all_per_class_metrics(per_class_metrics)
    
    for threshold in accuracy_thresholds:
        evaluator.filter_classes_by_threshold(
            per_class_metrics, 
            threshold=threshold,
            save_name="classes_by_accuracy"
        )
    
    evaluator.plot_threshold_analysis(per_class_metrics, thresholds=accuracy_thresholds)
    
    # Generar visualizaciones existentes
    evaluator.plot_confusion_matrix(targets, preds, top_k=50, normalize=True)
    evaluator.plot_confusion_matrix(targets, preds, top_k=30, normalize=False)
    evaluator.analyze_worst_classes(targets, preds, top_k=35)
    evaluator.plot_roc_curves(logits, targets, num_classes_to_plot=15)
    evaluator.plot_per_class_accuracy(targets, preds)
    
    # Plot training history si está disponible
    if history_path and history_path.exists():
        evaluator.plot_training_history(history_path)
    
    # Generar reporte
    evaluator.generate_report(metrics)
    
    logger.info("\n" + "="*60)
    logger.info("Evaluación completa finalizada!")
    logger.info(f"Resultados guardados en: {save_dir}")
    logger.info("="*60 + "\n")
    
    return metrics, per_class_metrics


def main():
    """Ejecuta evaluación de forma independiente"""
    import argparse
    from config import config
    from pipelines.models_temporal import get_temporal_model
    from pipelines.dataset_temporal import create_temporal_dataloaders
    
    parser = argparse.ArgumentParser(description="Evaluación completa de modelo temporal")
    
    parser.add_argument("--checkpoint", type=Path, default=None,
                       help="Ruta al checkpoint del modelo (.pt)")
    parser.add_argument("--features_dir", type=Path, default=None,
                       help=f"Directorio con features fusionadas (default: {config.data_paths.features_fused})")
    parser.add_argument("--metadata_path", type=Path, default=None,
                       help=f"Ruta a dataset_meta.json (default: {config.data_paths.dataset_meta})")
    parser.add_argument("--save_dir", type=Path, default=None,
                       help="Directorio para guardar resultados")
    parser.add_argument("--model_type", type=str, default="lstm", choices=["lstm", "transformer"],
                       help="Tipo de modelo temporal")
    parser.add_argument("--num_classes", type=int, default=None,
                       help=f"Numero de clases (default: {config.model.num_classes})")
    parser.add_argument("--batch_size", type=int, default=None,
                       help=f"Batch size (default: {config.training.batch_size})")
    parser.add_argument("--num_workers", type=int, default=None,
                       help=f"Numero de workers (default: {config.training.num_workers})")
    parser.add_argument("--device", type=str, default=None,
                       help=f"Device (default: {config.training.device})")
    parser.add_argument("--history_path", type=Path, default=None,
                       help="Ruta al training_history.json (opcional)")
    parser.add_argument("--thresholds", type=float, nargs='+', default=[0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
                       help="Umbrales de accuracy para análisis (default: 0.60 0.70 0.75 0.80 0.85 0.90 0.95)")
    
    args = parser.parse_args()
    
    checkpoint = args.checkpoint or config.model_paths.temporal_checkpoints
    features_dir = args.features_dir or config.data_paths.features_fused
    metadata_path = args.metadata_path or config.data_paths.dataset_meta
    num_classes = args.num_classes or config.model.num_classes
    batch_size = args.batch_size or config.training.batch_size
    num_workers = args.num_workers or config.training.num_workers
    device = torch.device(args.device or config.training.device)
    output = config.output_paths.evaluation_report
    
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = output / "evaluation_results"
    
    save_dir = Path(save_dir)
    
    if not checkpoint.exists():
        logger.error(f"Checkpoint no encontrado: {checkpoint}")
        return
    
    if not features_dir.exists():
        logger.error(f"Directorio de features no encontrado: {features_dir}")
        return
    
    if not metadata_path.exists():
        logger.error(f"Archivo de metadata no encontrado: {metadata_path}")
        return
    
    logger.info("="*60)
    logger.info("EVALUACIÓN INDEPENDIENTE DE MODELO TEMPORAL")
    logger.info("="*60)
    logger.info(f"Checkpoint: {checkpoint}")
    logger.info(f"Features dir: {features_dir}")
    logger.info(f"Save dir: {save_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Umbrales de accuracy: {args.thresholds}")
    logger.info("="*60 + "\n")
    
    # Cargar datos
    logger.info("Cargando datos...")
    _, _, test_loader = create_temporal_dataloaders(
        features_dir=features_dir,
        metadata_path=metadata_path,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Crear modelo
    logger.info(f"Creando modelo {args.model_type}...")
    model = get_temporal_model(
        model_type=args.model_type,
        num_classes=num_classes
    ).to(device)
    
    # Cargar checkpoint
    logger.info(f"Cargando checkpoint...")
    checkpoint = torch.load(checkpoint, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"Checkpoint val_accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    
    logger.info("Modelo cargado exitosamente!\n")
    
    # Ejecutar evaluación con umbrales personalizados
    evaluate_model_comprehensive(
        model=model,
        test_loader=test_loader,
        device=device,
        num_classes=num_classes,
        save_dir=save_dir,
        history_path=args.history_path,
        accuracy_thresholds=args.thresholds
    )
    
    logger.info("\n" + "="*60)
    logger.info("EVALUACIÓN COMPLETA FINALIZADA")
    logger.info(f"Resultados guardados en: {save_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
