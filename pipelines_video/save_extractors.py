"""
Script para crear y guardar extractores de features
EJECUTAR ESTE SCRIPT PRIMERO, ANTES DE TODO
"""

import torch #type: ignore
import torch.nn as nn #type: ignore
from pathlib import Path
from torchvision import models #type: ignore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResNet101FeatureExtractor(nn.Module):
    """Extractor visual usando ResNet101 pre-entrenado en ImageNet"""
    
    def __init__(self, output_dim=1024):
        super().__init__()
        
        # Cargar ResNet101 con pesos de ImageNet
        logger.info("Cargando ResNet101 pre-entrenado...")
        resnet = models.resnet101(pretrained=True)
        
        # Remover capa final (FC) para obtener features
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Proyección: 2048 → 1024
        self.projection = nn.Linear(2048, output_dim)
        
        # CONGELAR todos los pesos
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
        logger.info("✓ ResNet101 inicializado y congelado")
    
    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: (B, 3, 224, 224) - Batch de frames normalizados
        Returns:
            features: (B, 1024) - Features visuales
        """
        features = self.feature_extractor(x)  # (B, 2048, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B, 2048)
        features = self.projection(features)  # (B, 1024)
        return features


class PoseFeatureExtractor(nn.Module):
    """Extractor de pose usando MLP simple"""
    
    def __init__(self, input_dim=300, hidden_dim=256, output_dim=128):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )
        
        # CONGELAR todos los pesos
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
        logger.info("✓ MLP Pose inicializado y congelado")
    
    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: (B, 300) - Batch de keypoints aplanados
        Returns:
            features: (B, 128) - Features de pose
        """
        return self.mlp(x)


def save_extractors(output_dir: Path):
    """
    Crea, inicializa y guarda los extractores
    
    Este script debe ejecutarse UNA VEZ al inicio del proyecto
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("CREANDO Y GUARDANDO EXTRACTORES DE FEATURES")
    logger.info("="*80)
    
    # ========== EXTRACTOR VISUAL ==========
    logger.info("\n[1/2] Inicializando extractor visual...")
    visual_extractor = ResNet101FeatureExtractor(output_dim=1024)
    
    # Guardar modelo completo
    visual_path = output_dir / "visual_extractor_full.pt"
    torch.save(visual_extractor, visual_path)
    logger.info(f"✓ Guardado completo: {visual_path}")
    
    # Guardar solo pesos (más portable)
    visual_state_path = output_dir / "visual_extractor_state.pt"
    torch.save(visual_extractor.state_dict(), visual_state_path)
    logger.info(f"✓ Guardado state_dict: {visual_state_path}")
    
    # ========== EXTRACTOR DE POSE ==========
    logger.info("\n[2/2] Inicializando extractor de pose...")
    pose_extractor = PoseFeatureExtractor(
        input_dim=300,
        hidden_dim=256,
        output_dim=128
    )
    
    # Guardar modelo completo
    pose_path = output_dir / "pose_extractor_full.pt"
    torch.save(pose_extractor, pose_path)
    logger.info(f"✓ Guardado completo: {pose_path}")
    
    # Guardar solo pesos
    pose_state_path = output_dir / "pose_extractor_state.pt"
    torch.save(pose_extractor.state_dict(), pose_state_path)
    logger.info(f"✓ Guardado state_dict: {pose_state_path}")
    
    # ========== GUARDAR METADATA ==========
    import json
    metadata = {
        "visual_extractor": {
            "architecture": "ResNet101",
            "input_shape": [3, 224, 224],
            "output_dim": 1024,
            "pretrained": "ImageNet",
            "frozen": True
        },
        "pose_extractor": {
            "architecture": "MLP",
            "input_dim": 300,
            "hidden_dim": 256,
            "output_dim": 128,
            "frozen": True
        },
        "fused_features_dim": 1152  # 1024 + 128
    }
    
    metadata_path = output_dir / "extractors_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Metadata guardada: {metadata_path}")

    logger.info("EXTRACTORES GUARDADOS EXITOSAMENTE")
  


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Crea y guarda extractores de features para el pipeline ASL"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models/extractors"),
        help="Directorio donde guardar extractores (default: models/extractors)"
    )
    
    args = parser.parse_args()
    
    save_extractors(args.output_dir)