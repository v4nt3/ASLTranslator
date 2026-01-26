"""
Script para crear y guardar extractores de features
CORREGIDO: PoseFeatureExtractor ahora tiene opcion de entrenamiento
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
        logger.info("ResNet101 inicializado y congelado")
    
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
    """
    Extractor de pose usando MLP.
    CORREGIDO: Ahora puede ser entrenable o usar pesos preentrenados.
    
    IMPORTANTE: Para mejor rendimiento, considerar:
    1. Usar un modelo preentrenado de pose (ej: de COCO keypoints)
    2. Entrenar este MLP junto con el clasificador (fine-tuning)
    3. Usar features geometricas en lugar de coordenadas crudas
    """
    
    def __init__(
        self, 
        input_dim=300, 
        hidden_dim=256, 
        output_dim=128,
        freeze: bool = False,  # CORREGIDO: False por defecto ahora
        dropout: float = 0.3
    ):
        super().__init__()
        
        # MLP mejorado con mas capacidad
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # NUEVO: LayerNorm para estabilidad
            nn.GELU(),  # CORREGIDO: GELU en lugar de ReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)  # NUEVO: Normalizar salida
        )
        
        # Inicializacion de pesos
        self._init_weights()
        
        # CORREGIDO: Freeze es opcional
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
            logger.info("MLP Pose inicializado y CONGELADO")
        else:
            logger.info("MLP Pose inicializado y ENTRENABLE")
    
    def _init_weights(self):
        """Inicializacion Xavier para mejor convergencia"""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, 300) - Batch de keypoints aplanados
        Returns:
            features: (B, 128) - Features de pose
        """
        return self.mlp(x)


class TrainablePoseFeatureExtractor(nn.Module):
    """
    Extractor de pose entrenable con arquitectura mas robusta.
    Diseñado para ser entrenado end-to-end con el clasificador.
    
    RECOMENDACION: Usar esta version y entrenar junto con el LSTM.
    """
    
    def __init__(
        self,
        input_dim: int = 300,
        hidden_dims: list = [256, 256, 128],
        output_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout if i < len(hidden_dims) - 1 else dropout * 0.5)
            ])
            prev_dim = hidden_dim
        
        # Capa final de proyeccion
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Inicializacion
        self._init_weights()
        
        logger.info(f"TrainablePoseFeatureExtractor inicializado:")
        logger.info(f"  Input: {input_dim} -> Hidden: {hidden_dims} -> Output: {output_dim}")
    
    def _init_weights(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.mlp(x)


def save_extractors(output_dir: Path, freeze_pose: bool = False):
    """
    Crea, inicializa y guarda los extractores
    
    Args:
        output_dir: Directorio de salida
        freeze_pose: Si True, congela el extractor de pose (NO recomendado)
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
    logger.info(f"Guardado completo: {visual_path}")
    
    # Guardar solo pesos (más portable)
    visual_state_path = output_dir / "visual_extractor_state.pt"
    torch.save(visual_extractor.state_dict(), visual_state_path)
    logger.info(f"Guardado state_dict: {visual_state_path}")
    
    # ========== EXTRACTOR DE POSE ==========
    logger.info("\n[2/2] Inicializando extractor de pose...")
    
    # Version basica (compatible con codigo existente)
    pose_extractor = PoseFeatureExtractor(
        input_dim=300,
        hidden_dim=256,
        output_dim=128,
        freeze=freeze_pose  # CORREGIDO: Configurable
    )
    
    # Guardar modelo completo
    pose_path = output_dir / "pose_extractor_full.pt"
    torch.save(pose_extractor, pose_path)
    logger.info(f"Guardado completo: {pose_path}")
    
    # Guardar solo pesos
    pose_state_path = output_dir / "pose_extractor_state.pt"
    torch.save(pose_extractor.state_dict(), pose_state_path)
    logger.info(f"Guardado state_dict: {pose_state_path}")
    
    # ========== VERSION ENTRENABLE (RECOMENDADA) ==========
    logger.info("\n[BONUS] Guardando version entrenable del pose extractor...")
    trainable_pose = TrainablePoseFeatureExtractor(
        input_dim=300,
        hidden_dims=[256, 256, 128],
        output_dim=128,
        dropout=0.3
    )
    
    trainable_pose_path = output_dir / "trainable_pose_extractor.pt"
    torch.save(trainable_pose, trainable_pose_path)
    logger.info(f"Guardado: {trainable_pose_path}")
    
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
            "frozen": freeze_pose,
            "note": "RECOMENDACION: Usar trainable_pose_extractor.pt para mejor rendimiento"
        },
        "trainable_pose_extractor": {
            "architecture": "MLP (deeper)",
            "input_dim": 300,
            "hidden_dims": [256, 256, 128],
            "output_dim": 128,
            "frozen": False,
            "recommended": True
        },
        "fused_features_dim": 1152  # 1024 + 128
    }
    
    metadata_path = output_dir / "extractors_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata guardada: {metadata_path}")

    logger.info("\n" + "="*80)
    logger.info("EXTRACTORES GUARDADOS EXITOSAMENTE")
    logger.info("="*80)
    logger.info("\nRECOMENDACION: Para mejor rendimiento, usar trainable_pose_extractor.pt")
    logger.info("y entrenar el pose extractor junto con el clasificador LSTM.")
    logger.info("="*80)


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
    parser.add_argument(
        "--freeze_pose",
        action="store_true",
        help="Congelar el extractor de pose (NO recomendado)"
    )
    
    args = parser.parse_args()
    
    save_extractors(args.output_dir, freeze_pose=args.freeze_pose)
