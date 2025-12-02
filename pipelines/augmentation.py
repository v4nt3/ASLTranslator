"""
Módulo de Data Augmentation
Incluye augmentation de frames y keypoints
"""

import cv2 # type: ignore
import numpy as np # type: ignore
import torchvision.transforms as transforms # type: ignore
from config import DataConfig, config
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class FrameAugmentor:
    """Augmentation para frames RGB"""
    
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = config.data
        
        self.cfg = cfg
        
        # Transforms
        self.augment_transform = transforms.Compose([
            transforms.RandomCrop((cfg.frame_height, cfg.frame_width)),
            transforms.ColorJitter(
                brightness=cfg.color_jitter_brightness,
                contrast=cfg.color_jitter_contrast,
                saturation=cfg.color_jitter_saturation
            ),
            transforms.GaussianBlur(
                kernel_size=cfg.gaussian_blur_kernel,
                sigma=cfg.gaussian_blur_sigma
            ),
        ])
    
    def augment_frames(self, frames: np.ndarray) -> np.ndarray:
        """
        Aplica augmentation a una secuencia de frames
        
        Args:
            frames: shape (T, H, W, 3), puede ser float32 [0,1] o uint8 [0,255]
        
        Returns:
            frames augmentados
        """
        augmented = []
        
        for frame in frames:
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                # Si está en rango [0, 1], escalar a [0, 255]
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Frame a PIL
            from PIL import Image # type: ignore
            frame_pil = Image.fromarray(frame)
            
            # Aplicar augmentation
            # Nota: transforms espera PIL Image
            frame_aug = self.augment_transform(frame_pil)
            
            # Volver a numpy como uint8
            frame_aug_np = np.array(frame_aug, dtype=np.uint8)
            
            # Convertir a float32 [0, 1] para consistencia
            frame_aug_np = frame_aug_np.astype(np.float32) / 255.0
            
            augmented.append(frame_aug_np)
        
        return np.array(augmented)


class PoseAugmentor:
    """Augmentation para keypoints de pose"""
    
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = config.data
        
        self.cfg = cfg
    
    def augment_pose(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Aplica augmentation a keypoints de pose
        
        Args:
            keypoints: shape (T, N*D) donde N=75 (pose+hands), D=3
        
        Returns:
            keypoints augmentados
        """
        T, features = keypoints.shape
        N = 75  # num_pose_points (33) + num_hand_points (42)
        D = features // N  # Should be 3 (x, y, z)
        
        augmented = keypoints.copy().astype(np.float32)
        augmented = augmented.reshape(T, N, D)
        
        # 1. Jitter
        jitter = np.random.normal(0, self.cfg.pose_jitter_std, augmented.shape)
        augmented = augmented + jitter
        
        # 2. Scaling
        scale = np.random.uniform(*self.cfg.pose_scale_range)
        augmented = augmented * scale
        
        # 3. Rotation (pequeña rotación en plano)
        angle = np.random.uniform(-self.cfg.pose_rotation_degrees, self.cfg.pose_rotation_degrees)
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        for i in range(len(augmented)):
            x, y = augmented[i, :, 0], augmented[i, :, 1]
            augmented[i, :, 0] = x * cos_a - y * sin_a
            augmented[i, :, 1] = x * sin_a + y * cos_a
        
        augmented = augmented.reshape(T, N * D)
        
        return augmented
    
    def temporal_jitter(self, keypoints: np.ndarray) -> np.ndarray:
        """Aplica jitter temporal (speed jitter)"""
        speed = np.random.uniform(*self.cfg.speed_jitter_range)
        
        # Crear nueva secuencia interpolada
        original_length = len(keypoints)
        new_length = int(original_length * speed)
        
        if new_length < 2:
            new_length = 2
        
        indices = np.linspace(0, original_length - 1, new_length)
        augmented = np.zeros((new_length, *keypoints.shape[1:]))
        
        for i, idx in enumerate(indices):
            idx_low = int(np.floor(idx))
            idx_high = int(np.ceil(idx))
            
            if idx_low == idx_high:
                augmented[i] = keypoints[idx_low]
            else:
                weight = idx - idx_low
                augmented[i] = (1 - weight) * keypoints[idx_low] + weight * keypoints[idx_high]
        
        return augmented


class MultimodalAugmentor:
    """Orquestador de augmentation multimodal"""
    
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = DataConfig()
        
        self.cfg = cfg
        self.frame_augmentor = FrameAugmentor(cfg)
        self.pose_augmentor = PoseAugmentor(cfg)
    
    def augment(self, frames: np.ndarray, keypoints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Aplica augmentation multimodal"""
        aug_frames = self.frame_augmentor.augment_frames(frames)
        aug_keypoints = self.pose_augmentor.augment_pose(keypoints)
        
        # Temporal jitter
        if np.random.random() < 0.5:
            aug_keypoints = self.pose_augmentor.temporal_jitter(aug_keypoints)
        
        return aug_frames, aug_keypoints
