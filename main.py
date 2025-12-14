"""
Script de diagnóstico completo para identificar por qué el modelo no aprende
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging
from collections import Counter
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def diagnose_training_pipeline():
    """Diagnóstico completo del pipeline de entrenamiento"""
    
    print("="*80)
    print("DIAGNÓSTICO COMPLETO DEL PIPELINE")
    print("="*80)
    
    # ==================== 1. VERIFICAR FEATURES ====================
    print("\n[1/6] Verificando features fusionadas...")
    
    features_dir = Path("data/features_fused")
    feature_files = list(features_dir.glob("*_fused.npy"))
    
    if len(feature_files) == 0:
        print("❌ ERROR: No hay features en data/features_fused/")
        return
    
    print(f"✓ Encontrados {len(feature_files)} archivos de features")
    
    # Cargar muestra de features
    sample_features = []
    for i in range(min(10, len(feature_files))):
        feat = np.load(feature_files[i])
        sample_features.append(feat)
    
    # Analizar estadísticas
    all_features = np.concatenate(sample_features, axis=0)
    
    print(f"\nEstadísticas de features:")
    print(f"  Shape por clip: {sample_features[0].shape}")
    print(f"  Expected: (T, 1152)")
    print(f"  Mean: {all_features.mean():.6f}")
    print(f"  Std: {all_features.std():.6f}")
    print(f"  Min: {all_features.min():.6f}")
    print(f"  Max: {all_features.max():.6f}")
    print(f"  Zeros: {(all_features == 0).sum()}/{all_features.size} ({(all_features == 0).sum()/all_features.size*100:.1f}%)")
    print(f"  NaNs: {np.isnan(all_features).sum()}")
    print(f"  Infs: {np.isinf(all_features).sum()}")
    
    # Verificar si hay features degeneradas
    if all_features.std() < 0.01:
        print("❌ PROBLEMA CRÍTICO: Features casi constantes (std muy baja)")
        print("   Las features no tienen variabilidad suficiente")
        return
    
    if (all_features == 0).sum() / all_features.size > 0.5:
        print("❌ PROBLEMA: >50% de features son ceros")
        return
    
    if np.isnan(all_features).sum() > 0 or np.isinf(all_features).sum() > 0:
        print("❌ PROBLEMA: Features contienen NaN o Inf")
        return
    
    print("✓ Features parecen válidas")
    
    # ==================== 2. VERIFICAR LABELS ====================
    print("\n[2/6] Verificando labels (class_ids)...")
    
    import json
    metadata_path = Path("data/dataset_metadata.json")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Extraer class_ids
    class_ids = []
    
    if isinstance(metadata, dict):
        if 'videos' in metadata:
            entries = metadata['videos']
        else:
            entries = [{'video_file': k, **v} for k, v in metadata.items() if isinstance(v, dict)]
    else:
        entries = metadata
    
    for entry in entries:
        if isinstance(entry, dict):
            class_id = entry.get('class_id')
            if class_id is not None:
                class_ids.append(int(class_id))
    
    print(f"  Total class_ids: {len(class_ids)}")
    print(f"  Unique classes: {len(set(class_ids))}")
    print(f"  Min class_id: {min(class_ids)}")
    print(f"  Max class_id: {max(class_ids)}")
    
    # Distribución de clases
    class_counts = Counter(class_ids)
    most_common = class_counts.most_common(5)
    least_common = class_counts.most_common()[-5:]
    
    print(f"\n  Clases más frecuentes:")
    for class_id, count in most_common:
        print(f"    Class {class_id}: {count} samples")
    
    print(f"\n  Clases menos frecuentes:")
    for class_id, count in least_common:
        print(f"    Class {class_id}: {count} samples")
    
    # Verificar desbalanceo
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count
    
    print(f"\n  Ratio de desbalanceo: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 100:
        print("⚠ ADVERTENCIA: Desbalanceo extremo de clases")
    
    # ==================== 3. VERIFICAR MODELO ====================
    print("\n[3/6] Verificando arquitectura del modelo...")
    
    from pipelines.models_temporal import TemporalLSTMClassifier
    
    model = TemporalLSTMClassifier(
        input_dim=1152,
        hidden_dim=512,
        num_layers=2,
        num_classes=2286,
        dropout=0.3,
        bidirectional=True,
        use_attention=True
    )
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parámetros: {total_params:,}")
    print(f"  Parámetros entrenables: {trainable_params:,}")
    
    if trainable_params == 0:
        print("❌ PROBLEMA CRÍTICO: No hay parámetros entrenables")
        return
    
    # Test forward pass
    test_input = torch.randn(2, 24, 1152)  # (batch=2, time=24, features=1152)
    
    try:
        with torch.no_grad():
            output = model(test_input)
        
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: (2, 2286)")
        
        if output.shape != (2, 2286):
            print("❌ PROBLEMA: Output shape incorrecto")
            return
        
        # Verificar que no sea constante
        if output.std() < 0.01:
            print("❌ PROBLEMA: Outputs casi constantes")
            return
        
        print("✓ Arquitectura del modelo correcta")
        
    except Exception as e:
        print(f"❌ ERROR en forward pass: {e}")
        return
    
    # ==================== 4. VERIFICAR LOSS ====================
    print("\n[4/6] Verificando función de pérdida...")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    
    # Test con targets aleatorios
    targets = torch.randint(0, 2286, (2,))
    loss = criterion(output, targets)
    
    print(f"  Loss inicial (random): {loss.item():.4f}")
    print(f"  Expected: ~7.7 (log(2286))")
    
    expected_loss = np.log(2286)
    
    if abs(loss.item() - expected_loss) > 2.0:
        print("⚠ ADVERTENCIA: Loss inicial inusual")
    else:
        print("✓ Loss function correcta")
    
    # ==================== 5. VERIFICAR TRAINING LOOP ====================
    print("\n[5/6] Verificando training loop básico...")
    
    # Crear mini-batch de prueba
    batch_features = torch.randn(4, 24, 1152)
    batch_targets = torch.randint(0, 2286, (4,))
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\n  Ejecutando 10 pasos de entrenamiento...")
    losses = []
    
    for step in range(10):
        optimizer.zero_grad()
        
        outputs = model(batch_features)
        loss = criterion(outputs, batch_targets)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step == 0:
            print(f"    Step 0 loss: {loss.item():.4f}")
    
    print(f"    Step 9 loss: {losses[-1]:.4f}")
    print(f"    Loss change: {losses[0] - losses[-1]:.4f}")
    
    if losses[0] - losses[-1] < 0.01:
        print("❌ PROBLEMA CRÍTICO: Loss no está bajando")
        print("   El modelo no está aprendiendo nada")
        return
    else:
        print("✓ Loss está bajando correctamente")
    
    # ==================== 6. VERIFICAR FEATURES REALES ====================
    print("\n[6/6] Probando con features REALES del dataset...")
    
    # Cargar features y labels reales
    from pipelines.dataset_temporal import load_data_from_folder
    
    train_data, val_data, test_data = load_data_from_folder(
        features_dir=Path("data/features_fused"),
        metadata_path=Path("data/dataset_metadata.json"),
        train_split=0.7,
        val_split=0.15,
        random_seed=42
    )
    
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    print(f"  Test samples: {len(test_data)}")
    
    if len(train_data) < 100:
        print("⚠ ADVERTENCIA: Muy pocos datos de entrenamiento")
    
    # Cargar batch real
    real_paths = [item[0] for item in train_data[:4]]
    real_targets = [item[1] for item in train_data[:4]]
    
    real_features = []
    for path in real_paths:
        feat = np.load(path).astype(np.float32)
        real_features.append(torch.from_numpy(feat))
    
    # Pad a misma longitud
    max_len = max(f.shape[0] for f in real_features)
    padded = []
    for f in real_features:
        if f.shape[0] < max_len:
            padding = torch.zeros(max_len - f.shape[0], 1152)
            f = torch.cat([f, padding], dim=0)
        padded.append(f)
    
    real_batch = torch.stack(padded).float()
    real_targets = torch.tensor(real_targets).long()
    
    print(f"\n  Real batch shape: {real_batch.shape}")
    print(f"  Real targets: {real_targets.tolist()}")
    
    # Entrenar con datos reales
    model = TemporalLSTMClassifier(
        input_dim=1152, hidden_dim=512, num_layers=2,
        num_classes=2286, dropout=0.3, bidirectional=True, use_attention=True
    )
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\n  Entrenando 50 pasos con datos REALES...")
    
    for step in range(50):
        optimizer.zero_grad()
        
        outputs = model(real_batch)
        loss = criterion(outputs, real_targets)
        
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            # Calcular accuracy
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == real_targets).float().mean()
            print(f"    Step {step}: Loss={loss.item():.4f}, Acc={acc.item():.2%}")
    
    # Test final
    model.eval()
    with torch.no_grad():
        outputs = model(real_batch)
        preds = torch.argmax(outputs, dim=1)
        probs = torch.softmax(outputs, dim=1)
        max_probs = probs.max(dim=1)[0]
        
        acc = (preds == real_targets).float().mean()
    
    print(f"\n  Accuracy final: {acc.item():.2%}")
    print(f"  Max probabilities: {max_probs.tolist()}")
    print(f"  Predictions: {preds.tolist()}")
    print(f"  Targets: {real_targets.tolist()}")
    
    # ==================== DIAGNÓSTICO FINAL ====================
    print("\n" + "="*80)
    print("DIAGNÓSTICO FINAL")
    print("="*80)
    
    if acc.item() > 0.5:
        print("✓ El modelo PUEDE aprender con estos datos")
        print("\n  Posibles causas del mal entrenamiento:")
        print("  1. Learning rate muy bajo o muy alto")
        print("  2. Epochs insuficientes")
        print("  3. Batch size incorrecto")
        print("  4. Optimizer incorrecto")
        print("  5. Loss function con label smoothing muy alto")
        print("\n  RECOMENDACIÓN: Ajustar hiperparámetros de entrenamiento")
    
    elif acc.item() > 0.2:
        print("⚠ El modelo aprende LENTAMENTE")
        print("\n  Posibles causas:")
        print("  1. Features con poco poder discriminativo")
        print("  2. Modelo muy complejo para los datos")
        print("  3. Regularización muy fuerte (dropout 0.3 + label smoothing 0.15)")
        print("\n  RECOMENDACIÓN: Reducir dropout y label smoothing")
    
    else:
        print("❌ El modelo NO PUEDE aprender con estos datos")
        print("\n  Posibles causas CRÍTICAS:")
        print("  1. Features completamente incorrectas")
        print("  2. Labels incorrectos o corruptos")
        print("  3. Arquitectura inadecuada")
        print("  4. Bug en el training loop")
        print("\n  RECOMENDACIÓN: Revisar generación de features desde cero")
    
    print("="*80)


if __name__ == "__main__":
    diagnose_training_pipeline()