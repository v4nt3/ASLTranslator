"""
Arquitectura del modelo multimodal para ASL
RGB + Pose Keypoints con múltiples opciones de fusión
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VisualBackbone(nn.Module):
    """Backbone visual: EfficientNet o ResNet"""
    
    def __init__(self, backbone_name: str = "efficientnet_b3", pretrained: bool = True, 
                 embedding_dim: int = 512, freeze_backbone: bool = False):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        
        if backbone_name == "efficientnet_b3":
            backbone = models.efficientnet_b3(pretrained=pretrained)
            feature_dim = backbone.classifier[1].in_features
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        elif backbone_name == "resnet34":
            backbone = models.resnet34(pretrained=pretrained)
            feature_dim = 512  # ResNet34 always outputs 512 features from avgpool
            # Remove the classification layer
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        elif backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048  # ResNet50 outputs 2048 features
            self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        else:
            raise ValueError(f"Backbone desconocido: {backbone_name}")
        
        self.feature_dim = feature_dim
        
        # Projection head para embedding
        self.projection = nn.Linear(feature_dim, embedding_dim)
        
        # Freeze backbone si es necesario
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        logger.info(f"Visual backbone inicializado: {backbone_name}, feature_dim={feature_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) - secuencia de frames
        
        Returns:
            embeddings: (B, T, embedding_dim)
        """
        B, T, C, H, W = x.shape
        
        # Procesar cada frame
        x = x.view(B * T, C, H, W)
        features = self.feature_extractor(x)
        
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, 1).view(features.size(0), -1)
        
        embeddings = self.projection(features)
        embeddings = embeddings.view(B, T, self.embedding_dim)
        
        return embeddings


class PoseKeyPointBackbone(nn.Module):
    """Backbone para keypoints de pose/manos"""
    
    def __init__(self, input_dim: int = 300, hidden_dims: list = None, 
                 embedding_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 512]
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # MLP layers to process keypoints (x, y, z, confidence for 75 points)
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        logger.info(f"Pose backbone inicializado: input_dim={input_dim}, embedding_dim={embedding_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, 300) - secuencia de keypoints (75 points * 4 dims)
        
        Returns:
            embeddings: (B, T, embedding_dim)
        """
        B, T, feature_dim = x.shape
        
        # Procesar cada timestep
        x = x.view(B * T, feature_dim)
        embeddings = self.mlp(x)
        embeddings = embeddings.view(B, T, self.embedding_dim)
        
        return embeddings


class TemporalTransformerFusion(nn.Module):
    """Fusión multimodal con Transformer temporal"""
    
    def __init__(self, embedding_dim: int = 512, num_layers: int = 2, 
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Linear projection para fusión inicial
        self.fusion_proj = nn.Linear(embedding_dim * 2, embedding_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        logger.info(f"Temporal Transformer Fusion inicializado")
    
    def forward(self, rgb_emb: torch.Tensor, pose_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_emb: (B, T, embedding_dim)
            pose_emb: (B, T, embedding_dim)
        
        Returns:
            fused: (B, T, embedding_dim)
        """
        # Concatenar y proyectar
        combined = torch.cat([rgb_emb, pose_emb], dim=-1)
        fused = self.fusion_proj(combined)
        
        # Temporal transformer
        fused = self.transformer(fused)
        
        return fused


class ConcatFusion(nn.Module):
    """Fusión por concatenación simple"""
    
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(embedding_dim * 2, embedding_dim)
    
    def forward(self, rgb_emb: torch.Tensor, pose_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([rgb_emb, pose_emb], dim=-1)
        return self.proj(combined)


class SumFusion(nn.Module):
    """Fusión por suma"""
    
    def forward(self, rgb_emb: torch.Tensor, pose_emb: torch.Tensor) -> torch.Tensor:
        return rgb_emb + pose_emb


class ASLMultimodalClassifier(nn.Module):
    """Modelo completo multimodal para ASL"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        model_cfg = config.model
        
        # Backbones
        self.visual_backbone = VisualBackbone(
            backbone_name=model_cfg.backbone_name,
            pretrained=model_cfg.pretrained,
            embedding_dim=model_cfg.embedding_dim,
            freeze_backbone=model_cfg.freeze_backbone
        )
        
        self.pose_backbone = PoseKeyPointBackbone(
            input_dim=model_cfg.keypoints_dim,  # Changed from hardcoded calculation
            hidden_dims=model_cfg.pose_hidden_dims,
            embedding_dim=model_cfg.embedding_dim,
            dropout=model_cfg.pose_dropout
        )
        
        # Fusión multimodal
        if model_cfg.fusion_type == "concat":
            self.fusion = ConcatFusion(model_cfg.embedding_dim)
        elif model_cfg.fusion_type == "sum":
            self.fusion = SumFusion()
        elif model_cfg.fusion_type == "temporal_transformer":
            self.fusion = TemporalTransformerFusion(
                embedding_dim=model_cfg.embedding_dim,
                num_layers=model_cfg.num_transformer_layers,
                num_heads=model_cfg.num_attention_heads,
                dropout=model_cfg.transformer_dropout
            )
        else:
            raise ValueError(f"Fusion type desconocida: {model_cfg.fusion_type}")
        
        # Clasificador
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(model_cfg.embedding_dim, model_cfg.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(model_cfg.classifier_dropout),
            nn.Linear(model_cfg.classifier_hidden_dim, model_cfg.num_classes)
        )
        
        logger.info(f"ASL Multimodal Classifier inicializado")
        logger.info(f"  Fusion type: {model_cfg.fusion_type}")
        logger.info(f"  Num classes: {model_cfg.num_classes}")
    
    def forward(self, frames: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (B, T, C, H, W) - secuencia de frames RGB
            keypoints: (B, T, 300) - secuencia de keypoints (75 points * 4 dims)
        
        Returns:
            logits: (B, num_classes)
        """
        # Extraer embeddings
        rgb_emb = self.visual_backbone(frames)  # (B, T, embedding_dim)
        pose_emb = self.pose_backbone(keypoints)  # (B, T, embedding_dim)
        
        # Fusión multimodal
        fused_emb = self.fusion(rgb_emb, pose_emb)  # (B, T, embedding_dim)
        
        # Temporal pooling
        fused_emb = fused_emb.transpose(1, 2)  # (B, embedding_dim, T)
        pooled = self.temporal_pool(fused_emb)  # (B, embedding_dim, 1)
        pooled = pooled.squeeze(-1)  # (B, embedding_dim)
        
        # Clasificación
        logits = self.classifier(pooled)  # (B, num_classes)
        
        return logits


def get_model(config):
    """Factory function para crear el modelo"""
    model = ASLMultimodalClassifier(config)
    return model
