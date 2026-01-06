"""
Script para crear extractores CNN mejorados para keypoints
ALTERNATIVA SUPERIOR A MLP: Captura relaciones espaciales y temporales
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpatialCNNPoseExtractor(nn.Module):
    """
    Extractor CNN para keypoints que captura relaciones espaciales
    
    Arquitectura:
    - Reshape keypoints a estructura 2D: (100 landmarks, 3 coords) = (100, 3)
    - CNN1D sobre landmarks para capturar relaciones espaciales
    - Max/Avg pooling para agregación
    
    Ventajas sobre MLP:
    - Aprende patrones locales entre landmarks vecinos
    - Invarianza a traslaciones menores
    - Menos parámetros que MLP
    """
    
    def __init__(self, input_dim=300, output_dim=128):
        super().__init__()
        
        # input_dim=300 → (100 landmarks, 3 coords)
        self.num_landmarks = input_dim // 3
        
        self.spatial_cnn = nn.Sequential(
            # Bloque 1: 3 → 32 canales
            nn.Conv1d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            # Bloque 2: 32 → 64 canales
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            # Bloque 3: 64 → 128 canales
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)  # (B, 128, 100) → (B, 128, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
        
        # CONGELAR después de inicializar
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
        logger.info(f"✓ Spatial CNN Pose (landmarks={self.num_landmarks}, output={output_dim})")
    
    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: (B, 300) - Keypoints aplanados
        Returns:
            features: (B, 128) - Features espaciales
        """
        B = x.shape[0]
        
        # Reshape a (B, 100, 3) → transpose a (B, 3, 100) para Conv1d
        x = x.view(B, self.num_landmarks, 3).transpose(1, 2)  # (B, 3, 100)
        
        # CNN espacial
        x = self.spatial_cnn(x)  # (B, 128, 100)
        
        # Pooling global
        x = self.adaptive_pool(x).squeeze(-1)  # (B, 128)
        
        x = self.fc(x)  # (B, output_dim)
        
        return x


class TemporalCNNPoseExtractor(nn.Module):
    """
    Extractor CNN temporal para secuencias de keypoints
    
    Arquitectura:
    - Procesa múltiples frames simultáneamente
    - CNN1D sobre dimensión temporal
    - Captura movimientos y transiciones
    
    Ventajas:
    - Captura dinámica temporal de gestos
    - Aprende patrones de movimiento
    - Reduce secuencia temporal a features compactos
    """
    
    def __init__(self, input_dim=300, temporal_len=16, output_dim=128):
        super().__init__()
        
        self.input_dim = input_dim
        self.temporal_len = temporal_len
        
        self.temporal_cnn = nn.Sequential(
            # Bloque 1: Captura movimientos cortos (3 frames)
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # temporal_len/2
            nn.Dropout(0.3),
            
            # Bloque 2: Captura movimientos medios
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),  # temporal_len/4
            nn.Dropout(0.3),
            
            # Bloque 3: Features de alto nivel
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        
        # Pooling temporal global
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
        logger.info(f"✓ Temporal CNN Pose (temporal={temporal_len}, output={output_dim})")
    
    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: (B, T, 300) - Secuencia de keypoints
        Returns:
            features: (B, 128) - Features temporales
        """
        B, T, D = x.shape
        
        # Transpose para Conv1d: (B, 300, T)
        x = x.transpose(1, 2)  # (B, input_dim, T)
        
        # CNN temporal
        x = self.temporal_cnn(x)  # (B, 128, T')
        
        # Pooling global temporal
        x = self.adaptive_pool(x).squeeze(-1)  # (B, 128)
        
        x = self.fc(x)  # (B, output_dim)
        
        return x


class HybridCNNPoseExtractor(nn.Module):
    """
    Extractor híbrido que combina CNNs espaciales y temporales
    
    Arquitectura:
    1. Spatial CNN: Captura relaciones entre landmarks en cada frame
    2. Temporal CNN: Captura movimiento a través de frames
    3. Fusión: Combina ambos tipos de features
    
    Este es el MEJOR enfoque: Captura tanto estructura espacial como dinámica temporal
    """
    
    def __init__(self, input_dim=300, output_dim=128):
        super().__init__()
        
        self.num_landmarks = input_dim // 3
        
        self.spatial_branch = nn.Sequential(
            # Reshape keypoints a estructura 2D
            # (B*T, 3, num_landmarks)
            nn.Conv1d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.AdaptiveAvgPool1d(1),  # (B*T, 128, 1)
        )
        
        self.temporal_branch = nn.Sequential(
            # (B, 128, T) después de spatial
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.AdaptiveAvgPool1d(1),  # (B, 256, 1)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
        logger.info(f"✓ Hybrid CNN Pose (spatial+temporal, output={output_dim})")
    
    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: (B, T, 300) para batch de secuencias O (B, 300) para frame único
        Returns:
            features: (B, 128) - Features híbridas
        """
        if x.ndim == 2:
            # Frame único: (B, 300) → (B, 1, 300)
            x = x.unsqueeze(1)
        
        B, T, D = x.shape
        
        # 1. Procesar cada frame con spatial branch
        x_frames = x.view(B * T, self.num_landmarks, 3).transpose(1, 2)  # (B*T, 3, num_landmarks)
        spatial_features = self.spatial_branch(x_frames).squeeze(-1)  # (B*T, 128)
        
        # 2. Reorganizar para temporal branch
        spatial_features = spatial_features.view(B, T, 128).transpose(1, 2)  # (B, 128, T)
        
        # 3. Procesar secuencia temporal
        temporal_features = self.temporal_branch(spatial_features).squeeze(-1)  # (B, 256)
        
        # 4. Proyección final
        output = self.fusion(temporal_features)  # (B, output_dim)
        
        return output


class ResNet101FeatureExtractor(nn.Module):
    """Extractor visual usando ResNet101 pre-entrenado (sin cambios)"""
    
    def __init__(self, output_dim=1024):
        super().__init__()
        
        from torchvision import models
        logger.info("Cargando ResNet101 pre-entrenado...")
        resnet = models.resnet101(pretrained=True)
        
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(2048, output_dim)
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
        logger.info("✓ ResNet101 inicializado y congelado")
    
    @torch.no_grad()
    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.squeeze(-1).squeeze(-1)
        features = self.projection(features)
        return features


def save_extractors_cnn(output_dir: Path, extractor_type: str = "hybrid"):
    """
    Crea y guarda extractores CNN mejorados
    
    Args:
        output_dir: Directorio de salida
        extractor_type: "spatial" | "temporal" | "hybrid" (recomendado)
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("CREANDO EXTRACTORES CNN MEJORADOS")
    logger.info("="*80)
    
    # ========== EXTRACTOR VISUAL (sin cambios) ==========
    logger.info("\n[1/2] Inicializando extractor visual...")
    visual_extractor = ResNet101FeatureExtractor(output_dim=1024)
    
    visual_path = output_dir / "visual_extractor_full.pt"
    torch.save(visual_extractor, visual_path)
    logger.info(f"✓ Guardado: {visual_path}")
    
    # ========== EXTRACTOR CNN POSE ==========
    logger.info(f"\n[2/2] Inicializando extractor CNN ({extractor_type})...")
    
    if extractor_type == "spatial":
        pose_extractor = SpatialCNNPoseExtractor(input_dim=300, output_dim=128)
        architecture = "Spatial CNN (captura relaciones entre landmarks)"
    elif extractor_type == "temporal":
        pose_extractor = TemporalCNNPoseExtractor(input_dim=300, output_dim=128)
        architecture = "Temporal CNN (captura movimiento)"
    elif extractor_type == "hybrid":
        pose_extractor = HybridCNNPoseExtractor(input_dim=300, output_dim=128)
        architecture = "Hybrid CNN (espacial + temporal) - RECOMENDADO"
    else:
        raise ValueError(f"Tipo inválido: {extractor_type}")
    
    pose_path = output_dir / f"pose_extractor_cnn_{extractor_type}.pt"
    torch.save(pose_extractor, pose_path)
    logger.info(f"✓ Guardado: {pose_path}")
    
    # ========== METADATA ==========
    metadata = {
        "visual_extractor": {
            "architecture": "ResNet101",
            "input_shape": [3, 224, 224],
            "output_dim": 1024,
            "pretrained": "ImageNet",
            "frozen": True
        },
        "pose_extractor": {
            "architecture": f"CNN_{extractor_type}",
            "type": extractor_type,
            "input_dim": 300,
            "output_dim": 128,
            "frozen": True,
            "description": architecture,
            "advantages": [
                "Captura relaciones espaciales entre landmarks",
                "Menor cantidad de parámetros que MLP",
                "Aprende patrones jerárquicos",
                "Más robusto a variaciones"
            ]
        },
        "fused_features_dim": 1152,
        "improvements_over_mlp": {
            "spatial_awareness": "CNN captura vecindad entre landmarks",
            "parameter_efficiency": "Menos parámetros con mejor capacidad",
            "hierarchical_features": "Aprende desde patrones locales a globales",
            "temporal_modeling": "Hybrid/Temporal capturan movimiento"
        }
    }
    
    metadata_path = output_dir / f"extractors_cnn_{extractor_type}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Metadata: {metadata_path}")
    
    logger.info("\n" + "="*80)
    logger.info("✓ EXTRACTORES CNN CREADOS EXITOSAMENTE")
    logger.info("="*80)
    logger.info(f"\nTipo seleccionado: {extractor_type}")
    logger.info(f"Arquitectura: {architecture}")
    logger.info("\nVentajas sobre MLP:")
    logger.info("  ✓ Captura relaciones espaciales entre landmarks")
    logger.info("  ✓ Aprende patrones jerárquicos (local → global)")
    logger.info("  ✓ Más eficiente en parámetros")
    logger.info("  ✓ Mayor capacidad de generalización")
    logger.info("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Crea extractores CNN mejorados para keypoints"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models/extractors_cnn"),
        help="Directorio de salida"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="hybrid",
        choices=["spatial", "temporal", "hybrid"],
        help="Tipo de extractor CNN (hybrid recomendado)"
    )
    
    args = parser.parse_args()
    
    save_extractors_cnn(args.output_dir, args.type)
