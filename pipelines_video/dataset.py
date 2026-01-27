"""
Dataset para videos completos con longitud variable.
CORREGIDO: Augmentation habilitado en training, smoothing reducido en class weights
"""

import torch  # type: ignore
import numpy as np  # type: ignore
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler  # type: ignore
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from collections import Counter
import logging
import json
import re
from pipelines_video.config import config

logger = logging.getLogger(__name__)


class TemporalAugmentation:
    """
    Data augmentation temporal AGRESIVO para secuencias de features.
    Diseñado para reducir overfitting en datasets desbalanceados.
    """
    
    def __init__(
        self,
        time_warp_prob: float = 0.5,       # AUMENTADO
        time_mask_prob: float = 0.4,        # AUMENTADO
        feature_dropout_prob: float = 0.3,  # AUMENTADO
        noise_std: float = 0.05,            # AUMENTADO
        speed_change_range: Tuple[float, float] = (0.8, 1.2),  # MAS VARIACION
        feature_scale_range: Tuple[float, float] = (0.9, 1.1),  # NUEVO
        temporal_shift_prob: float = 0.3    # NUEVO
    ):
        self.time_warp_prob = time_warp_prob
        self.time_mask_prob = time_mask_prob
        self.feature_dropout_prob = feature_dropout_prob
        self.noise_std = noise_std
        self.speed_change_range = speed_change_range
        self.feature_scale_range = feature_scale_range
        self.temporal_shift_prob = temporal_shift_prob
    
    def __call__(self, features: np.ndarray) -> np.ndarray:
        """
        Aplica augmentations aleatorias a la secuencia.
        
        Args:
            features: (T, D) array de features
        
        Returns:
            features_aug: (T', D) array augmentado
        """
        features = features.copy()
        T, D = features.shape
        
        # 1. Time warping (cambio de velocidad)
        if np.random.random() < self.time_warp_prob and T > 10:
            features = self._time_warp(features)
            T = len(features)  # Actualizar T
        
        # 2. Temporal shift (desplazamiento circular)
        if np.random.random() < self.temporal_shift_prob and T > 5:
            features = self._temporal_shift(features)
        
        # 3. Time masking (ocultar segmentos)
        if np.random.random() < self.time_mask_prob and T > 5:
            features = self._time_mask(features)
        
        # 4. Feature dropout (dropout de dimensiones)
        if np.random.random() < self.feature_dropout_prob:
            features = self._feature_dropout(features)
        
        # 5. Feature scaling (escalar algunas dimensiones)
        if np.random.random() < 0.3:
            features = self._feature_scale(features)
        
        # 6. Gaussian noise (siempre aplicar algo de ruido)
        if self.noise_std > 0:
            noise = np.random.randn(*features.shape).astype(np.float32) * self.noise_std
            features = features + noise
        
        return features
    
    def _time_warp(self, features: np.ndarray) -> np.ndarray:
        """Cambio de velocidad con interpolacion"""
        T, D = features.shape
        
        # Factor de velocidad aleatorio
        speed = np.random.uniform(*self.speed_change_range)
        new_T = int(T / speed)
        new_T = max(5, min(new_T, T * 2))  # Limites
        
        # Interpolacion
        old_indices = np.arange(T)
        new_indices = np.linspace(0, T - 1, new_T)
        
        warped = np.zeros((new_T, D), dtype=np.float32)
        for d in range(D):
            warped[:, d] = np.interp(new_indices, old_indices, features[:, d])
        
        return warped
    
    def _temporal_shift(self, features: np.ndarray) -> np.ndarray:
        """Desplazamiento circular en el tiempo"""
        T, D = features.shape
        shift = np.random.randint(-T // 4, T // 4)
        return np.roll(features, shift, axis=0)
    
    def _time_mask(self, features: np.ndarray) -> np.ndarray:
        """Oculta un segmento temporal - MAS AGRESIVO"""
        T, D = features.shape
        
        # Longitud del mask (15-30% de la secuencia)
        mask_len = np.random.randint(1, max(2, int(T * 0.3)))
        mask_start = np.random.randint(0, max(1, T - mask_len))
        
        # Reemplazar con media de la secuencia
        mean_features = features.mean(axis=0, keepdims=True)
        features[mask_start:mask_start + mask_len] = mean_features
        
        return features
    
    def _feature_dropout(self, features: np.ndarray) -> np.ndarray:
        """Dropout aleatorio de dimensiones de features - MAS AGRESIVO"""
        T, D = features.shape
        
        # Dropout de 10-25% de dimensiones
        dropout_ratio = np.random.uniform(0.10, 0.25)
        num_dropout = int(D * dropout_ratio)
        
        dropout_dims = np.random.choice(D, num_dropout, replace=False)
        features[:, dropout_dims] = 0
        
        return features
    
    def _feature_scale(self, features: np.ndarray) -> np.ndarray:
        """Escala aleatoria de algunas dimensiones"""
        T, D = features.shape
        
        # Escalar 20-40% de las dimensiones
        num_scale = int(D * np.random.uniform(0.2, 0.4))
        scale_dims = np.random.choice(D, num_scale, replace=False)
        
        scale_factors = np.random.uniform(*self.feature_scale_range, size=num_scale)
        features[:, scale_dims] *= scale_factors
        
        return features


class VideoFeaturesDataset(Dataset):
    """
    Dataset que carga features de videos completos (longitud variable).
    No asume T fijo - cada video puede tener diferente numero de frames.
    """
    
    def __init__(
        self,
        features_paths: List[Path],
        class_ids: List[int],
        max_length: int = None,
        augment: bool = False,
        augmentation_config: dict = None
    ):
        """
        Args:
            features_paths: Lista de rutas a archivos *_fused.npy
            class_ids: Lista de class_id correspondientes
            max_length: Truncar secuencias mas largas (None = sin limite)
            augment: Aplicar data augmentation temporal
            augmentation_config: Config para TemporalAugmentation
        """
        assert len(features_paths) == len(class_ids), \
            f"Mismatch: {len(features_paths)} paths vs {len(class_ids)} class_ids"
        
        self.features_paths = features_paths
        self.class_ids = class_ids
        self.max_length = max_length
        self.augment = augment
        
        # Augmentation - CORREGIDO: Usar config por defecto si no se especifica
        if augment:
            aug_config = augmentation_config or config.training.augmentation_config
            self.augmenter = TemporalAugmentation(**aug_config)
        else:
            self.augmenter = None
        
        # Estadisticas de longitudes
        self._compute_length_stats()
        
        logger.info(f"VideoFeaturesDataset: {len(self)} videos")
        logger.info(f"  Longitudes: min={self.min_length}, max={self.max_length_found}, "
                   f"mean={self.mean_length:.1f}")
        logger.info(f"  Augmentation: {'ENABLED' if augment else 'disabled'}")
    
    def _compute_length_stats(self):
        """Calcula estadisticas de longitud de secuencias"""
        lengths = []
        for path in self.features_paths[:min(100, len(self.features_paths))]:
            if path.exists():
                features = np.load(path)
                lengths.append(len(features))
        
        if lengths:
            self.min_length = min(lengths)
            self.max_length_found = max(lengths)
            self.mean_length = np.mean(lengths)
        else:
            self.min_length = 0
            self.max_length_found = 0
            self.mean_length = 0
    
    def __len__(self):
        return len(self.features_paths)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int, int]:
        """
        Returns:
            features: (T, feature_dim) tensor
            class_id: int
            length: int (longitud real de la secuencia)
        """
        features_path = self.features_paths[idx]
        class_id = self.class_ids[idx]
        
        # Cargar features
        if not features_path.exists():
            logger.warning(f"Features no encontradas: {features_path}")
            features = np.zeros((1, config.features.fused_dim), dtype=np.float32)
        else:
            features = np.load(features_path).astype(np.float32)
        
        # Aplicar augmentation (solo si esta habilitado)
        if self.augment and self.augmenter is not None:
            features = self.augmenter(features)
        
        # Longitud despues de augmentation (puede cambiar por time_warp)
        length = len(features)
        
        # Truncar si es necesario
        if self.max_length is not None and length > self.max_length:
            features = features[:self.max_length]
            length = self.max_length
        
        return torch.from_numpy(features), class_id, length


def video_collate_fn(batch: List[Tuple[torch.Tensor, int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function con padding dinamico para secuencias de longitud variable.
    
    Args:
        batch: Lista de (features, class_id, length)
            features: (T_i, feature_dim)
            class_id: int
            length: int
    
    Returns:
        padded_features: (B, max_T, feature_dim)
        targets: (B,)
        lengths: (B,)
    """
    features_list = [item[0] for item in batch]
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)
    
    # Max length en este batch
    max_length = lengths.max().item()
    batch_size = len(features_list)
    feature_dim = features_list[0].shape[1]
    
    # Crear tensor padded
    padded_features = torch.zeros(batch_size, max_length, feature_dim, dtype=torch.float32)
    
    for i, features in enumerate(features_list):
        seq_len = features.shape[0]
        padded_features[i, :seq_len] = features
    
    return padded_features, targets, lengths


def load_video_data_from_folder(
    features_dir: Path,
    metadata_path: Path,
    train_split: float = 0.7,
    val_split: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]], Dict[int, int]]:
    """
    Carga datos desde carpeta de features y hace split estratificado.
    
    Returns:
        train_data, val_data, test_data: Listas de (path, class_id)
        class_counts: Diccionario {class_id: count} para class weights
    """
    logger.info(f"Cargando datos desde: {features_dir}")
    
    # Cargar metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Parsear metadata
    video_id_to_class = {}
    
    if isinstance(metadata, dict) and 'videos' in metadata:
        entries = metadata['videos']
    elif isinstance(metadata, list):
        entries = metadata
    else:
        entries = []
        for video_file, info in metadata.items():
            if isinstance(info, dict):
                entries.append({'video_file': video_file, **info})
    
    for entry in entries:
        video_file = entry.get('video_file', '')
        class_id = entry.get('class_id')
        class_name = entry.get('class_name', '')
        
        if video_file and class_id is not None:
            # Extraer video_id del nombre del archivo
            match = re.match(r'^(\d+)', video_file)
            if match:
                video_id = match.group(1)
                video_id_to_class[video_id] = {
                    'class_id': int(class_id),
                    'class_name': class_name,
                    'video_file': video_file
                }
    
    logger.info(f"Metadata: {len(video_id_to_class)} videos con class_id")
    
    # Buscar features
    features_dir = Path(features_dir)
    fused_files = list(features_dir.glob("*_fused.npy"))
    
    logger.info(f"Features encontradas: {len(fused_files)}")
    
    # Emparejar features con class_id
    data = []
    missing = 0
    
    for fused_path in fused_files:
        filename = fused_path.stem.replace('_fused', '')
        match = re.match(r'^(\d+)', filename)
        
        if match:
            video_id = match.group(1)
            if video_id in video_id_to_class:
                entry_info = video_id_to_class[video_id]
                class_id = entry_info['class_id']
                data.append((fused_path, class_id))
            else:
                missing += 1
        else:
            missing += 1
    
    if missing > 0:
        logger.warning(f"Features sin metadata: {missing}")
    
    logger.info(f"Videos validos: {len(data)}")
    
    if len(data) == 0:
        raise ValueError("No se encontraron videos validos")
    
    # Contar clases
    class_counts = Counter([item[1] for item in data])
    
    # Split estratificado por clase
    import random
    random.seed(random_seed)
    
    # Agrupar por clase
    class_to_samples = {}
    for path, class_id in data:
        if class_id not in class_to_samples:
            class_to_samples[class_id] = []
        class_to_samples[class_id].append((path, class_id))
    
    train_data = []
    val_data = []
    test_data = []
    
    for class_id, samples in class_to_samples.items():
        random.shuffle(samples)
        n = len(samples)
        
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))
        
        # CORREGIDO: Asegurar al menos 1 muestra en train para clases con pocas muestras
        train_end = max(1, train_end)
        
        train_data.extend(samples[:train_end])
        val_data.extend(samples[train_end:val_end])
        test_data.extend(samples[val_end:])
    
    # Shuffle final
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    logger.info(f"Split estratificado:")
    logger.info(f"  Train: {len(train_data)} videos")
    logger.info(f"  Val: {len(val_data)} videos")
    logger.info(f"  Test: {len(test_data)} videos")
    
    return train_data, val_data, test_data, dict(class_counts)


def compute_class_weights(
    class_counts: Dict[int, int],
    num_classes: int,
    smoothing: float = None  # CORREGIDO: Usar config por defecto
) -> torch.Tensor:
    """
    Calcula pesos por clase para CrossEntropyLoss.
    
    Formula: weight[c] = total_samples / (num_classes * count[c])
    Con smoothing para evitar pesos extremos.
    
    Args:
        class_counts: {class_id: count}
        num_classes: Numero total de clases
        smoothing: Factor de suavizado (0-1), por defecto usa config
    
    Returns:
        weights: Tensor (num_classes,)
    """
    if smoothing is None:
        smoothing = config.training.class_weight_smoothing  # CORREGIDO: Usa 0.05 en lugar de 0.1
    
    total_samples = sum(class_counts.values())
    
    weights = torch.ones(num_classes, dtype=torch.float32)
    
    for class_id, count in class_counts.items():
        if class_id < num_classes:
            # Peso inversamente proporcional a frecuencia
            weight = total_samples / (num_classes * count)
            # Aplicar smoothing: weight = (1-s) * weight + s * 1.0
            weights[class_id] = (1 - smoothing) * weight + smoothing
    
    # CORREGIDO: Clip para evitar valores extremos
    weights = torch.clamp(weights, min=0.1, max=10.0)
    
    # Normalizar para que la media sea 1
    weights = weights / weights.mean()
    
    logger.info(f"Class weights: min={weights.min():.2f}, max={weights.max():.2f}, "
               f"mean={weights.mean():.2f}, smoothing={smoothing}")
    
    return weights


def create_video_dataloaders(
    features_dir: Path = None,
    metadata_path: Path = None,
    batch_size: int = None,
    num_workers: int = None,
    max_length: int = None,
    train_split: float = 0.7,
    val_split: float = 0.15,
    random_seed: int = 42,
    use_weighted_sampler: bool = False,
    use_augmentation: bool = None  # NUEVO: Parametro explicito
) -> Tuple[DataLoader, DataLoader, DataLoader, Optional[torch.Tensor]]:
    """
    Crea DataLoaders para videos completos.
    
    CORREGIDO: Augmentation habilitado por defecto en training
    
    Returns:
        train_loader, val_loader, test_loader, class_weights
    """
    if features_dir is None:
        features_dir = config.data_paths.features_fused
    if metadata_path is None:
        metadata_path = config.data_paths.dataset_meta
    if batch_size is None:
        batch_size = config.training.batch_size
    if num_workers is None:
        num_workers = config.training.num_workers
    if use_augmentation is None:
        use_augmentation = config.training.use_augmentation  # CORREGIDO: True por defecto
    
    # Cargar datos
    train_data, val_data, test_data, class_counts = load_video_data_from_folder(
        features_dir=features_dir,
        metadata_path=metadata_path,
        train_split=train_split,
        val_split=val_split,
        random_seed=random_seed
    )
    
    # Separar paths y class_ids
    train_paths = [item[0] for item in train_data]
    train_class_ids = [item[1] for item in train_data]
    
    val_paths = [item[0] for item in val_data]
    val_class_ids = [item[1] for item in val_data]
    
    test_paths = [item[0] for item in test_data]
    test_class_ids = [item[1] for item in test_data]
    
    # CORREGIDO: Crear datasets con augmentation habilitado en training
    train_dataset = VideoFeaturesDataset(
        train_paths, 
        train_class_ids, 
        max_length,
        augment=use_augmentation,  # CORREGIDO: Habilitado!
        augmentation_config=config.training.augmentation_config
    )
    val_dataset = VideoFeaturesDataset(val_paths, val_class_ids, max_length, augment=False)
    test_dataset = VideoFeaturesDataset(test_paths, test_class_ids, max_length, augment=False)
    
    # Calcular class weights
    class_weights = compute_class_weights(
        class_counts, 
        num_classes=config.model.num_classes
    )
    
    # Sampler para balanceo (opcional)
    train_sampler = None
    shuffle = True
    
    if use_weighted_sampler:
        # Crear sample weights basados en clase
        sample_weights = [class_weights[c].item() for c in train_class_ids]
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle = False
        logger.info("Usando WeightedRandomSampler para balanceo")
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=video_collate_fn,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=video_collate_fn,
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=video_collate_fn,
        persistent_workers=num_workers > 0
    )
    
    logger.info(f"DataLoaders creados:")
    logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset)} videos) - Augmentation: {use_augmentation}")
    logger.info(f"  Val: {len(val_loader)} batches ({len(val_dataset)} videos)")
    logger.info(f"  Test: {len(test_loader)} batches ({len(test_dataset)} videos)")
    
    return train_loader, val_loader, test_loader, class_weights
