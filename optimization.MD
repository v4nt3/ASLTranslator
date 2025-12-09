# Técnicas Avanzadas de Optimización

## Resultados Base
- **Con input_dim=1152**: 82.27% accuracy ✓
- **Con input_dim=640**: 81.60% accuracy

## Técnicas Implementadas (Sin Cambiar Arquitectura)

### 1. **Label Smoothing** (0.1)
- **Qué hace**: Suaviza las etiquetas one-hot (0.9 para clase correcta, 0.1 distribuido)
- **Impacto esperado**: +0.5-1.5% accuracy
- **Por qué funciona**: Reduce overconfidence, mejora calibración

### 2. **Mixup Temporal** (α=0.2, prob=0.5)
- **Qué hace**: Mezcla pares de secuencias y sus etiquetas
- **Impacto esperado**: +1-2% accuracy
- **Por qué funciona**: Data augmentation implícito, regularización fuerte

### 3. **OneCycleLR Scheduler**
- **Qué hace**: Ciclo de LR: warmup → max → decay cosine
- **Impacto esperado**: +0.5-1% accuracy, converge más rápido
- **Por qué funciona**: Escapa de mínimos locales, mejor que ReduceLROnPlateau

### 4. **EMA (Exponential Moving Average)** (decay=0.999)
- **Qué hace**: Mantiene promedio exponencial de pesos
- **Impacto esperado**: +0.3-0.8% accuracy
- **Por qué funciona**: Promedio de modelos gratis, más estable

### 5. **Gradient Accumulation** (steps=2)
- **Qué hace**: Simula batch size 64 con memoria de batch 32
- **Impacto esperado**: +0.5-1% accuracy
- **Por qué funciona**: Batch size mayor = gradientes más estables

### 6. **Warmup** (3 epochs)
- **Qué hace**: LR crece gradualmente al inicio
- **Impacto esperado**: +0.2-0.5% accuracy
- **Por qué funciona**: Evita inestabilidad inicial

### 7. **Test Time Augmentation (TTA)**
- **Qué hace**: Promedia predicciones de 5 versiones augmented
- **Impacto esperado**: +0.5-1.5% accuracy
- **Por qué funciona**: Reduce varianza de predicciones

### 8. **Temporal Augmentation**
- **Qué hace**: Shift temporal, reversión, cambios de velocidad
- **Impacto esperado**: +0.5-1% accuracy
- **Por qué funciona**: Aumenta diversidad de datos temporales

### 9. **AdamW** (mejor que Adam)
- **Qué hace**: Weight decay desacoplado
- **Impacto esperado**: +0.2-0.5% accuracy
- **Por qué funciona**: Mejor regularización

### 10. **Progressive Unfreezing** (opcional)
- **Qué hace**: Descongela MLP/ResNet después de epochs
- **Impacto esperado**: +1-2% accuracy (si funciona bien)
- **Por qué funciona**: Fine-tuning progresivo de features

## Impacto Total Esperado

### Configuración Conservadora (strategy="conservative")
- Label Smoothing + EMA + OneCycleLR + Warmup
- **Mejora esperada**: +1-2% → **~83-84% accuracy**
- **Riesgo**: Muy bajo

### Configuración Simple (strategy="simple") ⭐ RECOMENDADA
- Label Smoothing + EMA + OneCycleLR + Gradient Accumulation
- **Mejora esperada**: +1.5-3% → **~83.5-85% accuracy**
- **Riesgo**: Bajo

### Configuración Agresiva (strategy="aggressive")
- TODAS las técnicas activadas
- **Mejora esperada**: +2-4% → **~84-86% accuracy**
- **Riesgo**: Medio (puede overfit con mixup muy fuerte)

## Recomendación de Uso

### Para Empezar
\`\`\`python
config = get_config(strategy="simple")
\`\`\`

Esto activará:
- ✓ Label Smoothing (0.1)
- ✓ EMA (0.999)
- ✓ OneCycleLR
- ✓ Gradient Accumulation (2 steps)
- ✗ Mixup (puede ser muy agresivo)
- ✗ TTA (solo para test final)

### Si Funciona Bien, Probar
\`\`\`python
config = get_config(strategy="aggressive")
\`\`\`

Activa TODO, incluyendo Mixup y TTA.

## Configuración Óptima Probable

Basado en tu arquitectura y resultados:

\`\`\`python
# Mejor balance riesgo/beneficio
config = Config()
config.LABEL_SMOOTHING = 0.1       # Imprescindible
config.USE_EMA = True              # Imprescindible  
config.USE_ONE_CYCLE = True        # Imprescindible
config.GRADIENT_ACCUMULATION_STEPS = 2  # Muy recomendado
config.USE_WARMUP = True           # Muy recomendado
config.USE_MIXUP = True            # Probar (0.2 alpha es suave)
config.MIXUP_ALPHA = 0.2
config.MIXUP_PROB = 0.3            # Solo 30% del tiempo
config.USE_TTA = False             # Solo para evaluación final
config.USE_TEMPORAL_AUG = True     # Recomendado
\`\`\`

## Ensemble (Máximo Rendimiento)

Si quieres exprimir hasta el último %, entrena 5 modelos con diferentes seeds:

\`\`\`python
config.USE_ENSEMBLE = True
config.ENSEMBLE_SEEDS = [42, 123, 456, 789, 1011]
\`\`\`

Promedia las predicciones finales.

**Impacto esperado**: +1-2% adicional → **~86-87% accuracy**

## Resumen Ejecutivo

| Técnica | Implementación | Impacto | Riesgo |
|---------|----------------|---------|---------|
| Label Smoothing | ✓ Fácil | Alto | Muy bajo |
| EMA | ✓ Fácil | Medio | Muy bajo |
| OneCycleLR | ✓ Fácil | Alto | Bajo |
| Gradient Accum | ✓ Fácil | Medio | Muy bajo |
| Warmup | ✓ Fácil | Medio | Muy bajo |
| Mixup | ⚠️ Medio | Alto | Medio |
| TTA | ✓ Fácil | Medio | Muy bajo |
| Temporal Aug | ✓ Medio | Medio | Bajo |
| Progressive Unfreeze | ⚠️ Complejo | Alto | Alto |
| Ensemble | ⚠️ Costoso | Alto | Muy bajo |

**Recomendación**: Empieza con "simple", si funciona prueba "aggressive", si quieres máximo rendimiento añade ensemble.
