"""
Data Augmentation para Features Precomputadas (Temporales)
Aplica augmentation directamente sobre features fusionadas (T, 640)
"""

import numpy as np
import torch
from typing import Tuple
import logging
from config import config

logger = logging.getLogger(__name__)


class TemporalFeatureAugmentor:
    """Augmentation para features temporales precomputadas"""
    
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = config.data
        
        self.cfg = cfg
        logger.info(f"TemporalFeatureAugmentor inicializado")
    
    def temporal_jitter(self, features: np.ndarray) -> np.ndarray:
        """
        Aplica jitter temporal (speed jitter) mediante interpolación
        
        Args:
            features: (T, 640) features precomputadas
        
        Returns:
            features_aug: (T', 640) features con velocidad modificada
        """
        speed = np.random.uniform(*self.cfg.speed_jitter_range)
        
        original_length = len(features)
        new_length = max(2, int(original_length * speed))
        
        # Crear índices interpolados
        indices = np.linspace(0, original_length - 1, new_length)
        augmented = np.zeros((new_length, features.shape[1]))
        
        for i, idx in enumerate(indices):
            idx_low = int(np.floor(idx))
            idx_high = min(int(np.ceil(idx)), original_length - 1)
            
            if idx_low == idx_high:
                augmented[i] = features[idx_low]
            else:
                weight = idx - idx_low
                augmented[i] = (1 - weight) * features[idx_low] + weight * features[idx_high]
        
        return augmented.astype(np.float32)
    
    def temporal_shift(self, features: np.ndarray) -> np.ndarray:
        """
        Aplica desplazamiento temporal circular
        
        Args:
            features: (T, 640)
        
        Returns:
            features_aug: (T, 640) desplazadas circularmente
        """
        shift = np.random.randint(-self.cfg.temporal_shift_range, self.cfg.temporal_shift_range + 1)
        return np.roll(features, shift, axis=0)
    
    def frame_dropout(self, features: np.ndarray) -> np.ndarray:
        """
        Elimina frames aleatorios (simula oclusiones temporales)
        
        Args:
            features: (T, 640)
        
        Returns:
            features_aug: (T', 640) con algunos frames eliminados
        """
        T = len(features)
        
        # Crear máscara de frames a mantener
        keep_mask = np.random.random(T) > self.cfg.frame_drop_p
        
        # Asegurar que al menos se mantenga 1 frame
        if not keep_mask.any():
            keep_mask[0] = True
        
        return features[keep_mask]
    
    def feature_noise(self, features: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
        """
        Agrega ruido gaussiano a las features
        
        Args:
            features: (T, 640)
            noise_std: Desviación estándar del ruido
        
        Returns:
            features_aug: (T, 640) con ruido
        """
        noise = np.random.normal(0, noise_std, features.shape)
        return features + noise.astype(np.float32)
    
    def feature_scaling(self, features: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        Escala las features globalmente
        
        Args:
            features: (T, 640)
            scale_range: Rango de escalado
        
        Returns:
            features_aug: (T, 640) escaladas
        """
        scale = np.random.uniform(*scale_range)
        return features * scale
    
    def augment(self, features: np.ndarray, p: float = 0.5) -> np.ndarray:
        """
        Aplica augmentation compuesto a features temporales
        
        Args:
            features: (T, 640) features precomputadas
            p: Probabilidad de aplicar cada augmentation
        
        Returns:
            features_aug: (T', 640) features augmentadas
        """
        features_aug = features.copy()
        
        # 1. Temporal jitter (cambia velocidad)
        if np.random.random() < p:
            features_aug = self.temporal_jitter(features_aug)
        
        # 2. Temporal shift (desplazamiento circular)
        if np.random.random() < p:
            features_aug = self.temporal_shift(features_aug)
        
        # 3. Frame dropout (eliminar frames aleatorios)
        if np.random.random() < p * 0.5:  # Menor probabilidad
            features_aug = self.frame_dropout(features_aug)
        
        # 4. Feature noise (ruido gaussiano)
        if np.random.random() < p:
            features_aug = self.feature_noise(features_aug, noise_std=0.005)
        
        # 5. Feature scaling (escalado global)
        if np.random.random() < p:
            features_aug = self.feature_scaling(features_aug, scale_range=(0.95, 1.05))
        
        return features_aug


class TemporalAugmentedDataset(torch.utils.data.Dataset):
    """
    Wrapper de Dataset que aplica augmentation on-the-fly
    """
    
    def __init__(self, base_dataset, augmentor: TemporalFeatureAugmentor, augment_prob: float = 0.5):
        """
        Args:
            base_dataset: Dataset base (TemporalFeaturesDataset)
            augmentor: Instancia de TemporalFeatureAugmentor
            augment_prob: Probabilidad de aplicar augmentation a cada sample
        """
        self.base_dataset = base_dataset
        self.augmentor = augmentor
        self.augment_prob = augment_prob
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        features, class_id = self.base_dataset[idx]
        
        # Convertir a numpy para augmentation
        features_np = features.numpy()
        
        # Aplicar augmentation con probabilidad augment_prob
        if np.random.random() < self.augment_prob:
            features_np = self.augmentor.augment(features_np, p=0.5)
        
        # Convertir de vuelta a tensor
        features_aug = torch.from_numpy(features_np).float()
        
        return features_aug, class_id


def create_augmented_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    augment_train: bool = True,
    augment_prob: float = 0.7,
    collate_fn=None
):
    """
    Crea DataLoaders con augmentation
    
    Args:
        train_dataset: Dataset de entrenamiento
        val_dataset: Dataset de validación
        test_dataset: Dataset de test
        batch_size: Batch size
        num_workers: Número de workers
        augment_train: Si aplicar augmentation al train set
        augment_prob: Probabilidad de augmentation
        collate_fn: Función de collate personalizada
    
    Returns:
        train_loader, val_loader, test_loader
    """
    augmentor = TemporalFeatureAugmentor()
    
    # Wrap train dataset con augmentation
    if augment_train:
        train_dataset_aug = TemporalAugmentedDataset(
            train_dataset, 
            augmentor, 
            augment_prob=augment_prob
        )
        logger.info(f"Augmentation activada en train set (p={augment_prob})")
    else:
        train_dataset_aug = train_dataset
    
    # Val y Test sin augmentation
    train_loader = torch.utils.data.DataLoader(
        train_dataset_aug,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0
    )
    
    return train_loader, val_loader, test_loader
