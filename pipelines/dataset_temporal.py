"""
Dataset optimizado para cargar features precomputadas (.npy fusionados)
Carga directamente desde carpeta sin necesidad de CSV
"""

import torch #type: ignore
import numpy as np #type: ignore
from torch.utils.data import Dataset, DataLoader #type: ignore
from pathlib import Path
from typing import Tuple, List
import logging
import json
import re

logger = logging.getLogger(__name__)


class TemporalFeaturesDataset(Dataset):
    """
    Dataset que carga features precomputadas (T, 640)
    Accede directamente a archivos en features_fused/
    """
    
    def __init__(
        self,
        features_paths: List[Path],
        class_ids: List[int],
        max_length: int = None
    ):
        """
        Args:
            features_paths: Lista de rutas a archivos *_fused.npy
            class_ids: Lista de class_id correspondientes
            max_length: Longitud maxima de secuencia (truncar si es mayor)
        """
        assert len(features_paths) == len(class_ids), "Numero de paths y class_ids debe coincidir"
        
        self.features_paths = features_paths
        self.class_ids = class_ids
        self.max_length = max_length
        
        logger.info(f"TemporalFeaturesDataset inicializado con {len(self)} samples")
    
    def __len__(self):
        return len(self.features_paths)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            features: (T, 640) tensor
            class_id: int
        """
        features_path = self.features_paths[idx]
        class_id = self.class_ids[idx]
        
        # Cargar features fusionadas
        if not features_path.exists():
            logger.warning(f"Features no encontradas: {features_path}")
            features = np.zeros((1, 640), dtype=np.float32)
        else:
            features = np.load(features_path).astype(np.float32)  # (T, 640)
        
        # Truncar si es muy largo
        if self.max_length is not None and len(features) > self.max_length:
            features = features[:self.max_length]
        
        # Convertir a tensor
        features = torch.from_numpy(features).float()
        
        return features, class_id


def load_data_from_folder(
    features_dir: Path,
    metadata_path: Path,
    train_split: float = 0.7,
    val_split: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    """
    Carga datos directamente desde la carpeta features_fused
    
    Args:
        features_dir: Directorio con archivos *_fused.npy
        metadata_path: Ruta a dataset_meta.json
        train_split: Proporcion para entrenamiento
        val_split: Proporcion para validacion
        random_seed: Semilla para reproducibilidad
    
    Returns:
        train_data, val_data, test_data: Listas de (path, class_id)
    """
    logger.info(f"Cargando datos desde: {features_dir}")
    logger.info(f"Cargando metadata desde: {metadata_path}")
    
    # Cargar metadata para obtener class_id
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    video_id_to_class = {}
    
    # Manejar diferentes formatos del JSON
    if isinstance(metadata, dict):
        if 'videos' in metadata:
            entries = metadata['videos']
        else:
            entries = []
            for video_file, info in metadata.items():
                if isinstance(info, dict):
                    entry = {'video_file': video_file, **info}
                    entries.append(entry)
    elif isinstance(metadata, list):
        entries = metadata
    else:
        raise ValueError(f"Formato de metadata no reconocido: {type(metadata)}")
    
    for entry in entries:
        if isinstance(entry, dict):
            video_file = entry.get('video_file', '')
            class_id = entry.get('class_id')
            
            if video_file and class_id is not None:
                # Extraer video_id (primeros numeros antes del guion)
                # Ejemplo: "023931338852502426-1 DOLLAR.mp4" -> "023931338852502426"
                match = re.match(r'^(\d+)', video_file)
                if match:
                    video_id = match.group(1)
                    video_id_to_class[video_id] = int(class_id)
    
    logger.info(f"Metadata cargada: {len(video_id_to_class)} videos con class_id")
    
    features_dir = Path(features_dir)
    fused_files = list(features_dir.glob("*_fused.npy"))
    
    logger.info(f"Archivos fusionados encontrados: {len(fused_files)}")
    
    data = []
    missing_metadata = 0
    
    for fused_path in fused_files:
        # Ejemplo: "023931338852502426_clip_0_fused.npy" -> "023931338852502426"
        filename = fused_path.stem.replace('_fused', '')
        match = re.match(r'^(\d+)', filename)
        
        if match:
            video_id = match.group(1)
            
            # Buscar class_id del video original en metadata
            if video_id in video_id_to_class:
                class_id = video_id_to_class[video_id]
                data.append((fused_path, class_id))
            else:
                missing_metadata += 1
        else:
            missing_metadata += 1
    
    if missing_metadata > 0:
        logger.warning(f"Clips sin metadata: {missing_metadata}")
    
    logger.info(f"Clips validos con class_id: {len(data)}")
    
    if len(data) == 0:
        raise ValueError("No se encontraron clips validos con class_id. Verifica el formato del metadata y los nombres de archivos.")
    
    import random
    random.seed(random_seed)
    random.shuffle(data)
    
    n = len(data)
    train_size = int(n * train_split)
    val_size = int(n * val_split)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    logger.info(f"Split completado:")
    logger.info(f"  Train: {len(train_data)} samples")
    logger.info(f"  Val: {len(val_data)} samples")
    logger.info(f"  Test: {len(test_data)} samples")
    
    return train_data, val_data, test_data


def temporal_collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function para secuencias de features de longitud variable
    
    Args:
        batch: Lista de (features, class_id)
            features: (T_i, 640)
            class_id: int
    
    Returns:
        padded_features: (B, max_T, 640)
        targets: (B,)
        lengths: (B,)
    """
    features_list = [item[0] for item in batch]
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    
    # Obtener longitudes
    lengths = torch.tensor([f.shape[0] for f in features_list], dtype=torch.long)
    max_length = lengths.max().item()
    
    # Pad features
    batch_size = len(features_list)
    feature_dim = features_list[0].shape[1]  # 640
    
    padded_features = torch.zeros(batch_size, max_length, feature_dim, dtype=torch.float32)
    
    for i, features in enumerate(features_list):
        length = features.shape[0]
        padded_features[i, :length] = features
    
    return padded_features, targets, lengths


def create_temporal_dataloaders(
    features_dir: Path,
    metadata_path: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    max_length: int = None,
    train_split: float = 0.7,
    val_split: float = 0.15,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea DataLoaders cargando directamente desde carpeta features_fused
    
    Args:
        features_dir: Directorio con archivos *_fused.npy
        metadata_path: Ruta a dataset_meta.json
        batch_size: Batch size
        num_workers: Numero de workers
        max_length: Longitud maxima de secuencia
        train_split: Proporcion para entrenamiento
        val_split: Proporcion para validacion
        random_seed: Semilla para reproducibilidad
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    train_data, val_data, test_data = load_data_from_folder(
        features_dir=features_dir,
        metadata_path=metadata_path,
        train_split=train_split,
        val_split=val_split,
        random_seed=random_seed
    )
    
    train_paths = [item[0] for item in train_data]
    train_class_ids = [item[1] for item in train_data]
    
    val_paths = [item[0] for item in val_data]
    val_class_ids = [item[1] for item in val_data]
    
    test_paths = [item[0] for item in test_data]
    test_class_ids = [item[1] for item in test_data]
    
    # Crear datasets
    train_dataset = TemporalFeaturesDataset(train_paths, train_class_ids, max_length)
    val_dataset = TemporalFeaturesDataset(val_paths, val_class_ids, max_length)
    test_dataset = TemporalFeaturesDataset(test_paths, test_class_ids, max_length)
    
    # Crear loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=temporal_collate_fn,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=temporal_collate_fn,
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=temporal_collate_fn,
        persistent_workers=num_workers > 0
    )
    
    logger.info(f"DataLoaders creados:")
    logger.info(f"   Train: {len(train_loader)} batches")
    logger.info(f"   Val: {len(val_loader)} batches")
    logger.info(f"   Test: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader
