"""
Script de evaluacion completa para modelo de video.
Genera metricas, visualizaciones y reportes.
"""

import torch  # type: ignore
import numpy as np  # type: ignore
from pathlib import Path
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.metrics import (  # type: ignore
    confusion_matrix, f1_score, precision_recall_fscore_support
)
import json
from typing import Dict, Tuple, Optional
from tqdm import tqdm  # type: ignore
import logging
import argparse

from pipelines_video.config import config
from pipelines_video.models import get_video_model
from pipelines_video.dataset import create_video_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VideoModelEvaluator:
    """Evaluador completo para modelo de video"""
    
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
        """Recolecta predicciones del modelo"""
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
        
        return (
            np.concatenate(all_logits),
            np.concatenate(all_preds),
            np.concatenate(all_targets)
        )
    
    def compute_metrics(
        self, 
        logits: np.ndarray, 
        preds: np.ndarray, 
        targets: np.ndarray
    ) -> Dict:
        """Calcula todas las metricas"""
        logger.info("Calculando metricas...")
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = float(np.mean(preds == targets))
        
        # Top-K Accuracy
        for k in [3, 5, 10]:
            topk_preds = np.argsort(logits, axis=1)[:, -k:]
            metrics[f'top{k}_accuracy'] = float(
                np.mean(np.any(topk_preds == targets[:, None], axis=1))
            )
        
        # F1 Scores
        metrics['f1_macro'] = float(f1_score(targets, preds, average='macro', zero_division=0))
        metrics['f1_weighted'] = float(f1_score(targets, preds, average='weighted', zero_division=0))
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, preds, average=None, zero_division=0
        )
        
        metrics['mean_precision'] = float(np.mean(precision[support > 0]))
        metrics['mean_recall'] = float(np.mean(recall[support > 0]))
        
        # Log metrics
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
        logger.info(f"  Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
        logger.info(f"  Top-10 Accuracy: {metrics['top10_accuracy']:.4f}")
        logger.info(f"  F1 Macro: {metrics['f1_macro']:.4f}")
        logger.info(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        targets: np.ndarray,
        preds: np.ndarray,
        top_k: int = 30,
        normalize: bool = True
    ):
        """Genera matriz de confusion para top-k clases"""
        logger.info(f"Generando matriz de confusion (top {top_k} clases)...")
        
        # Top-k clases mas frecuentes
        unique, counts = np.unique(targets, return_counts=True)
        top_classes = unique[np.argsort(counts)[-top_k:]]
        
        # Filtrar
        mask = np.isin(targets, top_classes)
        filtered_targets = targets[mask]
        filtered_preds = preds[mask]
        
        # Matriz
        cm = confusion_matrix(filtered_targets, filtered_preds, labels=top_classes)
        
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        # Plot
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(
            cm, annot=False, fmt='.2f' if normalize else 'd',
            cmap='Blues', ax=ax,
            xticklabels=top_classes, yticklabels=top_classes
        )
        ax.set_title(f'Confusion Matrix - Top {top_k} Classes' + 
                    (' (Normalized)' if normalize else ''))
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        save_path = self.save_dir / f'confusion_matrix_top{top_k}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Guardada en: {save_path}")
    
    def plot_training_history(self, history_path: Path):
        """Grafica historial de entrenamiento"""
        if not history_path.exists():
            logger.warning(f"Historial no encontrado: {history_path}")
            return
        
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, history['train_accuracy'], 'b-', label='Train', linewidth=2)
        axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Val', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-5 Accuracy
        axes[1, 0].plot(epochs, history['train_top5_accuracy'], 'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs, history['val_top5_accuracy'], 'r-', label='Val', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-5 Accuracy')
        axes[1, 0].set_title('Top-5 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        save_path = self.save_dir / 'training_history.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Historial guardado en: {save_path}")
    
    def analyze_worst_classes(
        self,
        targets: np.ndarray,
        preds: np.ndarray,
        top_k: int = 20,
        min_samples: int = 3
    ):
        """Analiza las peores clases"""
        logger.info(f"Analizando {top_k} peores clases...")
        
        unique_classes = np.unique(targets)
        class_stats = []
        
        for cls in unique_classes:
            mask = targets == cls
            n_samples = mask.sum()
            
            if n_samples < min_samples:
                continue
            
            acc = np.mean(preds[mask] == targets[mask])
            class_stats.append({
                'class_id': int(cls),
                'accuracy': float(acc),
                'num_samples': int(n_samples)
            })
        
        # Ordenar por accuracy (peor primero)
        class_stats = sorted(class_stats, key=lambda x: x['accuracy'])
        worst = class_stats[:top_k]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        class_ids = [c['class_id'] for c in worst]
        accuracies = [c['accuracy'] for c in worst]
        samples = [c['num_samples'] for c in worst]
        
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(worst)))
        bars = ax.barh(range(len(worst)), accuracies, color=colors)
        
        ax.set_yticks(range(len(worst)))
        ax.set_yticklabels([f"Class {cid} (n={n})" for cid, n in zip(class_ids, samples)])
        ax.set_xlabel('Accuracy')
        ax.set_title(f'Top {top_k} Worst Performing Classes')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(acc + 0.01, i, f'{acc:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        save_path = self.save_dir / 'worst_classes.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # JSON
        with open(self.save_dir / 'worst_classes.json', 'w') as f:
            json.dump(worst, f, indent=2)
        
        logger.info(f"  Guardado en: {save_path}")
    
    def plot_accuracy_distribution(self, targets: np.ndarray, preds: np.ndarray):
        """Distribucion de accuracy por clase"""
        unique_classes = np.unique(targets)
        accuracies = []
        
        for cls in unique_classes:
            mask = targets == cls
            if mask.sum() > 0:
                acc = np.mean(preds[mask] == targets[mask])
                accuracies.append(acc)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(accuracies, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(accuracies), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(accuracies):.3f}')
        ax.axvline(np.median(accuracies), color='green', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(accuracies):.3f}')
        
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Number of Classes')
        ax.set_title('Distribution of Per-Class Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.save_dir / 'accuracy_distribution.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Distribucion guardada en: {save_path}")
    
    def generate_report(self, metrics: Dict):
        """Genera reporte de texto"""
        report_path = self.save_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("VIDEO MODEL EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write("METRICS:\n")
            f.write("-"*40 + "\n")
            for key, value in metrics.items():
                f.write(f"  {key:25s}: {value:.6f}\n")
            f.write("\n")
        
        # JSON
        with open(self.save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"  Reporte guardado en: {report_path}")


def evaluate_video_model(
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    num_classes: int,
    save_dir: Path,
    history_path: Optional[Path] = None
) -> Dict:
    """Funcion principal de evaluacion"""
    
    evaluator = VideoModelEvaluator(model, device, num_classes, save_dir)
    
    # Predictions
    logits, preds, targets = evaluator.collect_predictions(test_loader)
    
    # Metrics
    metrics = evaluator.compute_metrics(logits, preds, targets)
    
    # Visualizations
    evaluator.plot_confusion_matrix(targets, preds, top_k=30, normalize=True)
    evaluator.analyze_worst_classes(targets, preds, top_k=20)
    evaluator.plot_accuracy_distribution(targets, preds)
    
    if history_path:
        evaluator.plot_training_history(history_path)
    
    evaluator.generate_report(metrics)
    
    logger.info(f"\nEvaluacion completa. Resultados en: {save_dir}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluacion de modelo de video")
    
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Ruta al checkpoint")
    parser.add_argument("--features_dir", type=Path, default=None)
    parser.add_argument("--metadata_path", type=Path, default=None)
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--model_type", type=str, default="lstm",
                        choices=["lstm", "transformer"])
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--history_path", type=Path, default=None)
    
    args = parser.parse_args()
    
    # Defaults
    features_dir = args.features_dir or config.data_paths.features_fused
    metadata_path = args.metadata_path or config.data_paths.dataset_meta
    num_classes = args.num_classes or config.model.num_classes
    batch_size = args.batch_size or config.training.batch_size
    num_workers = args.num_workers or config.training.num_workers
    device = torch.device(args.device or config.training.device)
    
    save_dir = args.save_dir or (args.checkpoint.parent / "evaluation")
    
    if not args.checkpoint.exists():
        logger.error(f"Checkpoint no encontrado: {args.checkpoint}")
        return
    
    logger.info("="*60)
    logger.info("EVALUACION DE MODELO DE VIDEO")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Save dir: {save_dir}")
    logger.info("="*60)
    
    # Load data
    _, _, test_loader, _ = create_video_dataloaders(
        features_dir=features_dir,
        metadata_path=metadata_path,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Load model
    model = get_video_model(
        model_type=args.model_type,
        num_classes=num_classes
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint (epoch {checkpoint.get('epoch', '?')})")
    else:
        model.load_state_dict(checkpoint)
    
    # Evaluate
    history_path = args.history_path
    if history_path is None:
        default_history = args.checkpoint.parent / "training_history.json"
        if default_history.exists():
            history_path = default_history
    
    evaluate_video_model(
        model=model,
        test_loader=test_loader,
        device=device,
        num_classes=num_classes,
        save_dir=save_dir,
        history_path=history_path
    )


if __name__ == "__main__":
    main()
