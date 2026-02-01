"""
Dataset CORREGIDO - Split a nivel de VIDEO para evitar data leakage

PROBLEMA ORIGINAL:
- El split era a nivel de CLIP, no de VIDEO
- Clips del mismo video podían estar en train Y val/test
- Esto causaba ~82% accuracy artificial que no generalizaba

SOLUCION:
- Agrupar clips por video_id
- Split estratificado a nivel de VIDEO
- Todos los clips de un video van al mismo split
- Estratificación por clase para mantener distribución
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
import json
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class TemporalFeaturesDataset(Dataset):
    """
    Dataset que carga features precomputadas (T, 1152)
    Sin cambios respecto al original
    """
    
    def __init__(
        self,
        features_paths: List[Path],
        class_ids: List[int],
        max_length: int = None
    ):
        assert len(features_paths) == len(class_ids), "Numero de paths y class_ids debe coincidir"
        
        self.features_paths = features_paths
        self.class_ids = class_ids
        self.max_length = max_length
        
        logger.info(f"TemporalFeaturesDataset inicializado con {len(self)} samples")
    
    def __len__(self):
        return len(self.features_paths)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        features_path = self.features_paths[idx]
        class_id = self.class_ids[idx]
        
        if not features_path.exists():
            logger.warning(f"Features no encontradas: {features_path}")
            features = np.zeros((1, 1152), dtype=np.float32)
        else:
            features = np.load(features_path).astype(np.float32)
        
        if self.max_length is not None and len(features) > self.max_length:
            features = features[:self.max_length]
        
        features = torch.from_numpy(features).float()
        
        return features, class_id


def extract_video_id(filename: str) -> Optional[str]:
    """
    Extrae el video_id de un nombre de archivo de clip.
    
    Ejemplos:
        "023931338852502426_clip_0_fused.npy" -> "023931338852502426"
        "023931338852502426-1 DOLLAR_clip_1_fused.npy" -> "023931338852502426"
    """
    # Remover sufijo _fused si existe
    name = filename.replace('_fused', '').replace('.npy', '')
    
    # Extraer video_id (digitos al inicio)
    match = re.match(r'^(\d+)', name)
    if match:
        return match.group(1)
    return None


def load_data_with_video_split(
    features_dir: Path,
    metadata_path: Path,
    train_split: float = 0.7,
    val_split: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]], Dict]:
    """
    CORREGIDO: Split a nivel de VIDEO, no de clip.
    
    Proceso:
    1. Agrupar clips por video_id
    2. Agrupar videos por class_id (para estratificacion)
    3. Split estratificado de VIDEOS
    4. Asignar todos los clips de cada video a su split
    
    Args:
        features_dir: Directorio con archivos *_fused.npy
        metadata_path: Ruta a dataset_meta.json
        train_split: Proporcion para entrenamiento (de videos, no clips)
        val_split: Proporcion para validacion
        random_seed: Semilla para reproducibilidad
    
    Returns:
        train_data, val_data, test_data: Listas de (path, class_id)
        stats: Diccionario con estadisticas del split
    """
    logger.info(f"="*60)
    logger.info("CARGANDO DATOS CON SPLIT A NIVEL DE VIDEO (SIN DATA LEAKAGE)")
    logger.info(f"="*60)
    logger.info(f"Features dir: {features_dir}")
    logger.info(f"Metadata: {metadata_path}")
    
    # 1. Cargar metadata para obtener class_id por video
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    video_id_to_class = {}
    
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
                match = re.match(r'^(\d+)', video_file)
                if match:
                    video_id = match.group(1)
                    video_id_to_class[video_id] = int(class_id)
    
    logger.info(f"Metadata cargada: {len(video_id_to_class)} videos con class_id")
    
    # 2. Encontrar todos los clips y agrupar por video_id
    features_dir = Path(features_dir)
    fused_files = list(features_dir.glob("*_fused.npy"))
    
    logger.info(f"Archivos fusionados encontrados: {len(fused_files)}")
    
    # video_id -> lista de (clip_path, class_id)
    video_to_clips: Dict[str, List[Tuple[Path, int]]] = defaultdict(list)
    missing_metadata = 0
    
    for fused_path in fused_files:
        video_id = extract_video_id(fused_path.name)
        
        if video_id and video_id in video_id_to_class:
            class_id = video_id_to_class[video_id]
            video_to_clips[video_id].append((fused_path, class_id))
        else:
            missing_metadata += 1
    
    if missing_metadata > 0:
        logger.warning(f"Clips sin metadata: {missing_metadata}")
    
    logger.info(f"Videos unicos con clips: {len(video_to_clips)}")
    
    total_clips = sum(len(clips) for clips in video_to_clips.values())
    logger.info(f"Total clips validos: {total_clips}")
    
    if len(video_to_clips) == 0:
        raise ValueError("No se encontraron videos validos con clips")
    
    # 3. Agrupar videos por clase para estratificacion
    # class_id -> lista de video_ids
    class_to_videos: Dict[int, List[str]] = defaultdict(list)
    
    for video_id, clips in video_to_clips.items():
        class_id = clips[0][1]  # Todos los clips tienen el mismo class_id
        class_to_videos[class_id].append(video_id)
    
    logger.info(f"Clases unicas: {len(class_to_videos)}")
    
    # 4. Split estratificado de VIDEOS
    np.random.seed(random_seed)
    
    train_videos = []
    val_videos = []
    test_videos = []
    
    for class_id, video_ids in class_to_videos.items():
        np.random.shuffle(video_ids)
        
        n_videos = len(video_ids)
        n_train = max(1, int(n_videos * train_split))
        n_val = max(0, int(n_videos * val_split))
        
        # Asegurar al menos 1 video en train
        if n_videos == 1:
            train_videos.extend(video_ids)
        elif n_videos == 2:
            train_videos.append(video_ids[0])
            val_videos.append(video_ids[1])
        else:
            train_videos.extend(video_ids[:n_train])
            val_videos.extend(video_ids[n_train:n_train + n_val])
            test_videos.extend(video_ids[n_train + n_val:])
    
    logger.info(f"\nSplit de VIDEOS:")
    logger.info(f"  Train videos: {len(train_videos)}")
    logger.info(f"  Val videos: {len(val_videos)}")
    logger.info(f"  Test videos: {len(test_videos)}")
    
    # 5. Convertir videos a clips
    train_data = []
    val_data = []
    test_data = []
    
    for video_id in train_videos:
        train_data.extend(video_to_clips[video_id])
    
    for video_id in val_videos:
        val_data.extend(video_to_clips[video_id])
    
    for video_id in test_videos:
        test_data.extend(video_to_clips[video_id])
    
    # Shuffle clips dentro de cada split (pero no entre splits)
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)
    
    logger.info(f"\nSplit de CLIPS:")
    logger.info(f"  Train clips: {len(train_data)}")
    logger.info(f"  Val clips: {len(val_data)}")
    logger.info(f"  Test clips: {len(test_data)}")
    
    # Estadisticas adicionales
    stats = {
        'total_videos': len(video_to_clips),
        'total_clips': total_clips,
        'total_classes': len(class_to_videos),
        'train_videos': len(train_videos),
        'val_videos': len(val_videos),
        'test_videos': len(test_videos),
        'train_clips': len(train_data),
        'val_clips': len(val_data),
        'test_clips': len(test_data),
        'clips_per_video_avg': total_clips / len(video_to_clips) if video_to_clips else 0
    }
    
    # Verificar que no hay leakage
    train_video_ids = set(train_videos)
    val_video_ids = set(val_videos)
    test_video_ids = set(test_videos)
    
    assert len(train_video_ids & val_video_ids) == 0, "LEAKAGE: Videos en train y val!"
    assert len(train_video_ids & test_video_ids) == 0, "LEAKAGE: Videos en train y test!"
    assert len(val_video_ids & test_video_ids) == 0, "LEAKAGE: Videos en val y test!"
    
    logger.info("\n[OK] Verificacion de data leakage: PASADA")
    logger.info("     No hay videos compartidos entre splits")
    logger.info(f"="*60 + "\n")
    
    return train_data, val_data, test_data, stats


def temporal_collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function para secuencias de longitud variable
    Sin cambios respecto al original
    """
    features_list = [item[0] for item in batch]
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    
    lengths = torch.tensor([f.shape[0] for f in features_list], dtype=torch.long)
    max_length = lengths.max().item()
    
    batch_size = len(features_list)
    feature_dim = features_list[0].shape[1]
    
    padded_features = torch.zeros(batch_size, max_length, feature_dim, dtype=torch.float32)
    
    for i, features in enumerate(features_list):
        length = features.shape[0]
        padded_features[i, :length] = features
    
    return padded_features, targets, lengths


def create_temporal_dataloaders_fixed(
    features_dir: Path,
    metadata_path: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    max_length: int = None,
    train_split: float = 0.7,
    val_split: float = 0.15,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    CORREGIDO: Crea DataLoaders con split a nivel de video.
    
    Returns:
        train_loader, val_loader, test_loader, stats
    """
    
    train_data, val_data, test_data, stats = load_data_with_video_split(
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
    
    logger.info(f"DataLoaders creados (SIN DATA LEAKAGE):")
    logger.info(f"   Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    logger.info(f"   Val: {len(val_loader)} batches ({len(val_dataset)} samples)")
    logger.info(f"   Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader, stats


# ============================================================================
# FUNCION ADICIONAL: Analisis de distribucion de clases
# ============================================================================

def analyze_class_distribution(
    features_dir: Path,
    metadata_path: Path
) -> Dict:
    """
    Analiza la distribucion de clases en el dataset.
    Util para entender el desbalanceo.
    """
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    video_id_to_class = {}
    class_to_name = {}
    
    entries = metadata.get('videos', metadata) if isinstance(metadata, dict) else metadata
    
    for entry in entries:
        if isinstance(entry, dict):
            video_file = entry.get('video_file', '')
            class_id = entry.get('class_id')
            class_name = entry.get('class_name', f'CLASS_{class_id}')
            
            if video_file and class_id is not None:
                match = re.match(r'^(\d+)', video_file)
                if match:
                    video_id = match.group(1)
                    video_id_to_class[video_id] = int(class_id)
                    class_to_name[int(class_id)] = class_name
    
    # Contar clips por clase
    features_dir = Path(features_dir)
    fused_files = list(features_dir.glob("*_fused.npy"))
    
    class_counts = defaultdict(int)
    video_counts = defaultdict(set)
    
    for fused_path in fused_files:
        video_id = extract_video_id(fused_path.name)
        if video_id and video_id in video_id_to_class:
            class_id = video_id_to_class[video_id]
            class_counts[class_id] += 1
            video_counts[class_id].add(video_id)
    
    # Estadisticas
    counts = list(class_counts.values())
    video_per_class = [len(v) for v in video_counts.values()]
    
    analysis = {
        'total_classes': len(class_counts),
        'total_clips': sum(counts),
        'clips_per_class': {
            'min': min(counts) if counts else 0,
            'max': max(counts) if counts else 0,
            'mean': np.mean(counts) if counts else 0,
            'median': np.median(counts) if counts else 0,
            'std': np.std(counts) if counts else 0
        },
        'videos_per_class': {
            'min': min(video_per_class) if video_per_class else 0,
            'max': max(video_per_class) if video_per_class else 0,
            'mean': np.mean(video_per_class) if video_per_class else 0,
        },
        'class_to_name': class_to_name,
        'class_counts': dict(class_counts),
        'imbalance_ratio': max(counts) / min(counts) if counts and min(counts) > 0 else float('inf')
    }
    
    logger.info("\n" + "="*60)
    logger.info("ANALISIS DE DISTRIBUCION DE CLASES")
    logger.info("="*60)
    logger.info(f"Total clases: {analysis['total_classes']}")
    logger.info(f"Total clips: {analysis['total_clips']}")
    logger.info(f"Clips por clase: min={analysis['clips_per_class']['min']}, "
                f"max={analysis['clips_per_class']['max']}, "
                f"mean={analysis['clips_per_class']['mean']:.1f}")
    logger.info(f"Videos por clase: min={analysis['videos_per_class']['min']}, "
                f"max={analysis['videos_per_class']['max']}, "
                f"mean={analysis['videos_per_class']['mean']:.1f}")
    logger.info(f"Ratio de desbalanceo: {analysis['imbalance_ratio']:.1f}x")
    logger.info("="*60 + "\n")
    
    return analysis
