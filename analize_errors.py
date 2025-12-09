"""
Script de análisis de errores compatible con TemporalFeaturesDataset
Analiza predicciones en el test set para identificar patrones de error
"""

import torch
import numpy as np
from pathlib import Path
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict, Counter
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(model_path: Path, num_classes: int, input_dim: int, device: torch.device):
    """Carga el modelo entrenado"""
    # Importar tu modelo actual
    from pipelines.models_temporal import TemporalLSTMClassifier
    
    model = TemporalLSTMClassifier(
        input_dim=input_dim,
        hidden_dim=512,
        num_classes=num_classes,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        use_attention=True
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Modelo cargado desde: {model_path}")
    return model


def get_predictions(model, dataloader, device):
    """Obtiene predicciones del modelo en el dataset"""
    all_preds = []
    all_labels = []
    all_probs = []
    all_features_paths = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (features, labels, lengths) in enumerate(dataloader):
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            
            # Forward pass
            outputs = model(features, lengths)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            if batch_idx % 50 == 0:
                logger.info(f"Procesados {batch_idx}/{len(dataloader)} batches")
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def analyze_confusion_patterns(y_true, y_pred, class_names, top_k=10):
    """Analiza los patrones de confusión más comunes"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Encontrar las confusiones más frecuentes (excluir diagonal)
    confusion_counts = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i, j] > 0:
                confusion_counts.append({
                    'true_class': i,
                    'true_name': class_names.get(i, f'Class_{i}'),
                    'pred_class': j,
                    'pred_name': class_names.get(j, f'Class_{j}'),
                    'count': cm[i, j]
                })
    
    # Ordenar por frecuencia
    confusion_counts.sort(key=lambda x: x['count'], reverse=True)
    
    logger.info(f"\nTop {top_k} confusiones más frecuentes:")
    for idx, conf in enumerate(confusion_counts[:top_k], 1):
        logger.info(
            f"{idx}. {conf['true_name']} -> {conf['pred_name']}: "
            f"{conf['count']} errores"
        )
    
    return confusion_counts[:top_k], cm


def analyze_per_class_performance(y_true, y_pred, class_names):
    """Analiza el rendimiento por clase"""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Extraer métricas por clase
    class_metrics = []
    for class_id in sorted(set(y_true)):
        class_name = class_names.get(class_id, f'Class_{class_id}')
        metrics = report.get(str(class_id), {})
        
        class_metrics.append({
            'class_id': class_id,
            'class_name': class_name,
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1': metrics.get('f1-score', 0),
            'support': metrics.get('support', 0)
        })
    
    # Ordenar por F1 score
    class_metrics.sort(key=lambda x: x['f1'])
    
    logger.info(f"\nClases con peor rendimiento (Bottom 10):")
    for idx, cm in enumerate(class_metrics[:10], 1):
        logger.info(
            f"{idx}. {cm['class_name']}: "
            f"F1={cm['f1']:.3f}, P={cm['precision']:.3f}, "
            f"R={cm['recall']:.3f}, Support={cm['support']}"
        )
    
    logger.info(f"\nClases con mejor rendimiento (Top 10):")
    for idx, cm in enumerate(class_metrics[-10:], 1):
        logger.info(
            f"{idx}. {cm['class_name']}: "
            f"F1={cm['f1']:.3f}, P={cm['precision']:.3f}, "
            f"R={cm['recall']:.3f}, Support={cm['support']}"
        )
    
    return class_metrics


def analyze_confidence(y_true, y_pred, y_probs):
    """Analiza la confianza de las predicciones"""
    correct_mask = y_true == y_pred
    incorrect_mask = ~correct_mask
    
    # Confianza máxima por predicción
    max_probs = np.max(y_probs, axis=1)
    
    correct_confidences = max_probs[correct_mask]
    incorrect_confidences = max_probs[incorrect_mask]
    
    logger.info(f"\nAnálisis de confianza:")
    logger.info(f"Predicciones correctas:")
    logger.info(f"  Media: {correct_confidences.mean():.3f}")
    logger.info(f"  Mediana: {np.median(correct_confidences):.3f}")
    logger.info(f"  Min: {correct_confidences.min():.3f}")
    logger.info(f"  Max: {correct_confidences.max():.3f}")
    
    logger.info(f"\nPredicciones incorrectas:")
    logger.info(f"  Media: {incorrect_confidences.mean():.3f}")
    logger.info(f"  Mediana: {np.median(incorrect_confidences):.3f}")
    logger.info(f"  Min: {incorrect_confidences.min():.3f}")
    logger.info(f"  Max: {incorrect_confidences.max():.3f}")
    
    # Análisis de errores con alta confianza
    high_conf_errors = incorrect_confidences > 0.8
    logger.info(f"\nErrores con alta confianza (>0.8): {high_conf_errors.sum()}")
    
    return {
        'correct_mean': correct_confidences.mean(),
        'incorrect_mean': incorrect_confidences.mean(),
        'high_conf_errors': high_conf_errors.sum()
    }


def plot_confusion_matrix(cm, class_names, output_path, top_classes=50):
    """Genera visualización de matriz de confusión"""
    # Limitar a top N clases más frecuentes para legibilidad
    if len(cm) > top_classes:
        # Sumar errores por clase
        class_errors = cm.sum(axis=1) + cm.sum(axis=0)
        top_indices = np.argsort(class_errors)[-top_classes:]
        cm_subset = cm[np.ix_(top_indices, top_indices)]
        class_labels = [class_names.get(i, f'C{i}') for i in top_indices]
    else:
        cm_subset = cm
        class_labels = [class_names.get(i, f'C{i}') for i in range(len(cm))]
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm_subset,
        annot=False,
        fmt='d',
        cmap='Blues',
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={'label': 'Número de muestras'}
    )
    plt.title(f'Matriz de Confusión (Top {len(cm_subset)} clases con más errores)')
    plt.ylabel('Clase Real')
    plt.xlabel('Clase Predicha')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info(f"Matriz de confusión guardada en: {output_path}")
    plt.close()

def default_numpy(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)

def save_error_analysis_report(
    confusion_patterns,
    class_metrics,
    confidence_stats,
    output_path
):
    """Guarda reporte completo de análisis de errores"""
    report = {
        'summary': {
            'total_classes_analyzed': len(class_metrics),
            'avg_confidence_correct': float(confidence_stats['correct_mean']),
            'avg_confidence_incorrect': float(confidence_stats['incorrect_mean']),
            'high_confidence_errors': int(confidence_stats['high_conf_errors'])
        },
        'top_confusions': confusion_patterns,
        'worst_classes': class_metrics[:20],
        'best_classes': class_metrics[-20:]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=default_numpy)
    
    logger.info(f"Reporte de análisis guardado en: {output_path}")


def main():
    # Configuración
    FEATURES_DIR = Path("data/features_fused")  # Tu directorio de features
    METADATA_PATH = Path("data/dataset_metadata.json")  # Tu archivo de metadata
    MODEL_PATH = Path("checkpoints/temporal82/best_model.pt")  # Tu mejor modelo
    OUTPUT_DIR = Path("checkpoints/temporal82/analysis_results")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Parámetros del modelo
    INPUT_DIM = 1152  # 1024 (ResNet) + 128 (MLP) - ajustar según tu config
    NUM_CLASSES = 2286  # Ajustar según tu dataset
    BATCH_SIZE = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Usando device: {device}")
    
    # Cargar nombres de clases desde metadata
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    # Crear mapeo class_id -> nombre de clase
    class_names = {}
    if isinstance(metadata, dict) and 'videos' in metadata:
        entries = metadata['videos']
    elif isinstance(metadata, list):
        entries = metadata
    else:
        entries = []
        for video_file, info in metadata.items():
            if isinstance(info, dict):
                entry = {'video_file': video_file, **info}
                entries.append(entry)
    
    for entry in entries:
        if isinstance(entry, dict):
            class_id = entry.get('class_id')
            gloss = entry.get('gloss', f'Class_{class_id}')
            if class_id is not None:
                class_names[int(class_id)] = gloss
    
    logger.info(f"Cargadas {len(class_names)} clases")
    
    # Importar tu dataset
    sys.path.append(str(Path(__file__).parent.parent))
    from pipelines.dataset_temporal import create_temporal_dataloaders
    
    # Crear dataloaders (solo necesitamos test_loader)
    train_loader, val_loader, test_loader = create_temporal_dataloaders(
        features_dir=FEATURES_DIR,
        metadata_path=METADATA_PATH,
        batch_size=BATCH_SIZE,
        num_workers=0,
        max_length=None,
        train_split=0.7,
        val_split=0.15,
        random_seed=42
    )
    
    logger.info(f"Test loader: {len(test_loader)} batches")
    
    # Cargar modelo
    model = load_model(MODEL_PATH, NUM_CLASSES, INPUT_DIM, device)
    
    # Obtener predicciones
    logger.info("Obteniendo predicciones en test set...")
    y_pred, y_true, y_probs = get_predictions(model, test_loader, device)
    
    # Calcular accuracy general
    accuracy = (y_pred == y_true).mean()
    logger.info(f"\nAccuracy en test set: {accuracy:.4f}")
    
    # Análisis de patrones de confusión
    confusion_patterns, cm = analyze_confusion_patterns(y_true, y_pred, class_names)
    
    # Análisis de rendimiento por clase
    class_metrics = analyze_per_class_performance(y_true, y_pred, class_names)
    
    # Análisis de confianza
    confidence_stats = analyze_confidence(y_true, y_pred, y_probs)
    
    # Generar visualizaciones
    plot_confusion_matrix(
        cm,
        class_names,
        OUTPUT_DIR / "confusion_matrix.png"
    )
    
    # Guardar reporte completo
    save_error_analysis_report(
        confusion_patterns,
        class_metrics,
        confidence_stats,
        OUTPUT_DIR / "error_analysis_report.json"
    )
    
    logger.info(f"\nAnálisis completo guardado en: {OUTPUT_DIR}")
    
    # Recomendaciones basadas en análisis
    logger.info("\n" + "="*60)
    logger.info("RECOMENDACIONES:")
    logger.info("="*60)
    
    if confidence_stats['high_conf_errors'] > len(y_true) * 0.05:
        logger.info("1. ALTO número de errores con alta confianza")
        logger.info("   -> Considera usar label smoothing o aumentar regularización")
    
    if confidence_stats['incorrect_mean'] > 0.6:
        logger.info("2. Modelo muy confiado en predicciones incorrectas")
        logger.info("   -> Revisa si hay clases muy similares que necesitan más features")
    
    logger.info("\n3. Revisa el archivo 'error_analysis_report.json' para:")
    logger.info("   - Identificar pares de clases confundidas frecuentemente")
    logger.info("   - Clases con peor F1 score (candidatas para data augmentation)")


if __name__ == "__main__":
    main()
