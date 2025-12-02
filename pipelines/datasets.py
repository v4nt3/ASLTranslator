"""
DataLoaders para el pipeline de entrenamiento
"""

import torch # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler # type: ignore
from pathlib import Path
from typing import Tuple, List
from config import config
import logging

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.ERROR)   # solo errores
logger.addHandler(console)



class ASLDataset(Dataset):
    """Dataset para clips de ASL"""
    
    def __init__(self, dataframe: pd.DataFrame, clip_dir: Path, 
                 transform=None, augmentor=None):
        self.df = dataframe
        self.clip_dir = clip_dir
        self.transform = transform
        self.augmentor = augmentor
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row = self.df.iloc[idx]
        
        # Las rutas están guardadas relativas a data/
        if 'clip_path' in self.df.columns and 'keypoints_path' in self.df.columns:
            frames_path = self.clip_dir.parent / row['clip_path']
            keypoints_path = self.clip_dir.parent / row['keypoints_path']
        else:
            # Fallback a clip_name si existe
            frames_path = self.clip_dir / f"{row['clip_name']}_frames.npy"
            keypoints_path = self.clip_dir / f"{row['clip_name']}_keypoints.npy"
        
        frames = np.load(frames_path).astype(np.float32)
        keypoints = np.load(keypoints_path).astype(np.float32)
        
        # Normalizar frames [0, 1]
        frames = frames / 255.0
        
        # Reshape keypoints (T, N, D) -> (T, N*D)
        T, N, D = keypoints.shape
        keypoints = keypoints.reshape(T, N * D)
        logger.debug("Loaded clip:")
        logger.debug("frames shape =", frames.shape)
        logger.debug("keypoints shape =", keypoints.shape)

        
        # Augmentation
        if self.augmentor is not None and np.random.random() < 0.5:
            frames, keypoints = self.augmentor.augment(frames, keypoints)
        
        # Convertir a tensores
        frames = torch.from_numpy(frames).float()  # (T, H, W, C)
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
        
        keypoints = torch.from_numpy(keypoints).float()  # (T, N*D)
        
        class_id = int(row['class_id'])
        
        return frames, keypoints, class_id


def custom_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Tuple:
    """
    Collate function que maneja secuencias de longitud variable usando padding.
    
    Args:
        batch: Lista de (frames, keypoints, class_id)
    
    Returns:
        (padded_frames, padded_keypoints, targets, lengths)
    """
    frames_list = [item[0] for item in batch]  # Cada uno es (T_i, C, H, W)
    keypoints_list = [item[1] for item in batch]  # Cada uno es (T_i, N*D)
    targets = torch.tensor([item[2] for item in batch], dtype=torch.long)
    
    frames_lengths = [f.shape[0] for f in frames_list]
    keypoints_lengths = [k.shape[0] for k in keypoints_list]
    
    # # Validate that frames and keypoints have same length for each sample
    # for i, (f_len, k_len) in enumerate(zip(frames_lengths, keypoints_lengths)):
    #     if f_len != k_len:
    #         logger.warning(f"Sample {i}: frames length {f_len} != keypoints length {k_len}. Using max.")
    
    # Use max of all lengths for consistent padding
    max_length = max(max(frames_lengths), max(keypoints_lengths))
    lengths = torch.tensor(frames_lengths, dtype=torch.long)
    
    # Pad frames: (B, max_T, C, H, W)
    batch_size = len(frames_list)
    C, H, W = frames_list[0].shape[1:]
    padded_frames = torch.zeros(batch_size, max_length, C, H, W, dtype=torch.float32)
    for i, frames in enumerate(frames_list):
        padded_frames[i, :frames.shape[0]] = frames
    
    # Pad keypoints: (B, max_T, N*D)
    num_features = keypoints_list[0].shape[1]
    padded_keypoints = torch.zeros(batch_size, max_length, num_features, dtype=torch.float32)
    for i, keypoints in enumerate(keypoints_list):
        padded_keypoints[i, :keypoints.shape[0]] = keypoints
    
    return padded_frames, padded_keypoints, targets, lengths


def create_data_loaders(config, df_train: pd.DataFrame, df_val: pd.DataFrame, 
                       df_test: pd.DataFrame, augmentor=None):
    """Crea DataLoaders para train, val y test"""
    
    clip_dir = config.data_paths.clips
    
    # Datasets
    train_dataset = ASLDataset(df_train, clip_dir, augmentor=augmentor)
    val_dataset = ASLDataset(df_val, clip_dir, augmentor=None)  # Sin augmentation
    test_dataset = ASLDataset(df_test, clip_dir, augmentor=None)
    
    # Weighted sampler para train (handle class imbalance)
    class_counts = df_train['class_id'].value_counts()
    weights = 1.0 / class_counts[df_train['class_id'].values].values
    sampler = WeightedRandomSampler(weights, len(df_train), replacement=True)
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=sampler,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        collate_fn=custom_collate_fn  # Add custom collate function
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        collate_fn=custom_collate_fn  # Add custom collate function
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        collate_fn=custom_collate_fn  # Add custom collate function
    )
    
    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Val loader: {len(val_loader)} batches")
    logger.info(f"Test loader: {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader
