"""
Script de evaluación completa con visualizaciones y métricas avanzadas
Genera matrices de confusión, curvas ROC, análisis de peores clases, etc.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, top_k_accuracy_score
)
from sklearn.preprocessing import label_binarize
import json
from typing import Dict, List, Tuple
from tqdm import tqdm

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
    
    def plot_confusion_matrix(
        self,
        targets: np.ndarray,
        preds: np.ndarray,
        top_k: int = 50,
        normalize: bool = True
    ):
        """Genera matriz de confusión para las top-k clases más frecuentes"""
        logger.info(f"Generando matriz de confusión (top {top_k} clases)...")
        
        # Encontrar las clases más frecuentes
        unique, counts = np.unique(targets, return_counts=True)
        top_classes = unique[np.argsort(counts)[-top_k:]]
        
        # Filtrar predicciones solo de esas clases
        mask = np.isin(targets, top_classes)
        filtered_targets = targets[mask]
        filtered_preds = preds[mask]
        
        # Matriz de confusión
        cm = confusion_matrix(filtered_targets, filtered_preds, labels=top_classes)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
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
        
        # Loss
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-5 Accuracy
        axes[1, 0].plot(epochs, history['train_top5_accuracy'], 'b-', label='Train Top-5', linewidth=2)
        axes[1, 0].plot(epochs, history['val_top5_accuracy'], 'r-', label='Val Top-5', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-5 Accuracy')
        axes[1, 0].set_title('Training and Validation Top-5 Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
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
        
        # Calcular accuracy por clase
        unique_classes = np.unique(targets)
        class_accuracies = []
        
        for cls in unique_classes:
            mask = targets == cls
            if mask.sum() < min_samples:
                continue
            
            acc = np.mean(preds[mask] == targets[mask])
            class_accuracies.append({
                'class_name': int(cls),  # Convert to Python int
                'accuracy': float(acc),  # Convert to Python float
                'num_samples': int(mask.sum())  # Convert to Python int
            })
        
        # Ordenar por accuracy (peor primero)
        class_accuracies = sorted(class_accuracies, key=lambda x: x['accuracy'])
        worst_classes = class_accuracies[:top_k]
        
        # Plot
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
        
        # Agregar valores de accuracy en las barras
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(acc + 0.01, i, f'{acc:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        save_path = self.save_dir / 'worst_classes.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Guardar en JSON
        worst_classes_json = self.save_dir / 'worst_classes.json'
        with open(worst_classes_json, 'w') as f:
            json.dump(worst_classes, f, indent=2)
        
        logger.info(f"Peores clases guardadas en: {save_path}")
        logger.info(f"Detalles en: {worst_classes_json}")
    
    def plot_roc_curves(
        self,
        logits: np.ndarray,
        targets: np.ndarray,
        num_classes_to_plot: int = 10
    ):
        """Genera curvas ROC para algunas clases"""
        logger.info(f"Generando curvas ROC para {num_classes_to_plot} clases...")
        
        # Seleccionar clases más frecuentes
        unique, counts = np.unique(targets, return_counts=True)
        top_classes = unique[np.argsort(counts)[-num_classes_to_plot:]]
        
        # Binarizar targets
        targets_bin = label_binarize(targets, classes=range(self.num_classes))
        
        # Aplicar softmax a logits
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        
        plt.figure(figsize=(12, 10))
        
        for cls in top_classes:
            # Calcular ROC curve
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
    history_path: Path = None
):
    """Función principal de evaluación completa"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    evaluator = ModelEvaluator(model, device, num_classes, save_dir)
    
    # Recolectar predicciones
    logits, preds, targets = evaluator.collect_predictions(test_loader)
    
    # Calcular métricas
    metrics = evaluator.compute_metrics(logits, preds, targets)
    
    # Generar visualizaciones
    evaluator.plot_confusion_matrix(targets, preds, top_k=50, normalize=True)
    evaluator.plot_confusion_matrix(targets, preds, top_k=30, normalize=False)
    evaluator.analyze_worst_classes(targets, preds, top_k=30)
    evaluator.plot_roc_curves(logits, targets, num_classes_to_plot=10)
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
    
    return metrics

def main():
    """Ejecuta evaluación de forma independiente"""
    import argparse
    from config import config
    from pipelines.models_temporal import get_temporal_model
    from pipelines.dataset_temporal import create_temporal_dataloaders
    
    parser = argparse.ArgumentParser(description="Evaluación completa de modelo temporal")
    
    parser.add_argument("--checkpoint", type=Path, required=True,
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
    
    args = parser.parse_args()
    
    # Setup parameters
    features_dir = args.features_dir or config.data_paths.features_fused
    metadata_path = args.metadata_path or config.data_paths.dataset_meta
    num_classes = args.num_classes or config.model.num_classes
    batch_size = args.batch_size or config.training.batch_size
    num_workers = args.num_workers or config.training.num_workers
    device = torch.device(args.device or config.training.device)
    
    if args.save_dir:
        save_dir = args.save_dir
    else:
        # Save in the same directory as checkpoint
        save_dir = args.checkpoint.parent / "evaluation_results"
    
    save_dir = Path(save_dir)
    
    if not args.checkpoint.exists():
        logger.error(f"Checkpoint no encontrado: {args.checkpoint}")
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
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Features dir: {features_dir}")
    logger.info(f"Save dir: {save_dir}")
    logger.info(f"Device: {device}")
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
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    
    # Manejar diferentes formatos de checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        logger.info(f"Checkpoint val_accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    
    logger.info("Modelo cargado exitosamente!\n")
    
    # Ejecutar evaluación
    evaluate_model_comprehensive(
        model=model,
        test_loader=test_loader,
        device=device,
        num_classes=num_classes,
        save_dir=save_dir,
        history_path=args.history_path
    )
    
    logger.info("\n" + "="*60)
    logger.info("EVALUACIÓN COMPLETA FINALIZADA")
    logger.info(f"Resultados guardados en: {save_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
