"""
Script para precomputar features visuales usando ResNet101 frozen
Convierte clips (T, H, W, 3) → (T, 1024) en float16
"""

import torch #type: ignore
import numpy as np #type: ignore
from pathlib import Path
from tqdm import  tqdm #type: ignore
import logging
from torchvision import transforms #type: ignore

from pipelines.config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_visual_extractor(extractor_path: Path, device: str):
    """Carga el extractor visual guardado"""
    
    if not extractor_path.exists():
        raise FileNotFoundError(
            f"Extractor no encontrado: {extractor_path}\n"
            f"   Ejecuta primero: python scripts/save_extractors.py"
        )
    
    logger.info(f"Cargando extractor visual desde: {extractor_path}")
    
    try:
        # Intentar cargar modelo completo
        extractor = torch.load(extractor_path, map_location=device)
        logger.info("✓ Extractor cargado (modelo completo)")
    except:
        # Si falla, cargar state_dict
        from pipelines_video.save_extractors import ResNet101FeatureExtractor
        extractor = ResNet101FeatureExtractor(output_dim=1024)
        extractor.load_state_dict(torch.load(extractor_path, map_location=device))
        extractor.to(device)
        logger.info("✓ Extractor cargado (state_dict)")
    
    extractor.eval()
    return extractor

def precompute_visual_features(
    clips_dir: Path = None,
    output_dir: Path = None,
    extractor_path: Path = None,
    device: str = None,
    batch_size: int = 32
):
    """
    Precomputa features visuales para todos los clips
    
    Args:
        clips_dir: Directorio con archivos *_frames.npy (usa config si es None)
        output_dir: Directorio donde guardar *_visual.npy (usa config si es None)
        device: 'cuda' o 'cpu' (usa config si es None)
        batch_size: Batch size para procesamiento
    """
    
    if clips_dir is None:
        clips_dir = config.data_paths.clips
    if output_dir is None:
        output_dir = config.data_paths.features_visual
    if device is None:
        device = config.training.device
    if extractor_path is None:
        extractor_path = Path("models/extractors/visual_extractor_full.pt")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Inicializar modelo
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = load_visual_extractor(extractor_path, device) 
    
    # Transform para normalizar frames
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Buscar todos los archivos _frames.npy
    frame_files = sorted(clips_dir.glob("*_frames.npy"))
    logger.info(f" Encontrados {len(frame_files)} clips para procesar")
    
    if len(frame_files) == 0:
        logger.error(f" No se encontraron archivos *_frames.npy en {clips_dir}")
        return
    
    # Procesar cada clip
    processed = 0
    skipped = 0
    errors = 0
    
    for frame_file in tqdm(frame_files, desc="Extrayendo features visuales"):
        clip_name = frame_file.stem.replace("_frames", "")
        output_file = output_dir / f"{clip_name}_visual.npy"
        
        # Skip si ya existe
        if output_file.exists():
            skipped += 1
            continue
        
        try:
            # Cargar frames
            frames = np.load(frame_file)  # (T, H, W, 3)
            
            # Validar shape
            if frames.ndim != 4 or frames.shape[-1] != 3:
                logger.warning(f" Shape inválido para {clip_name}: {frames.shape}")
                errors += 1
                continue
            
            T = len(frames)
            all_features = []
            
            # Procesar en batches
            for i in range(0, T, batch_size):
                batch_frames = frames[i:i+batch_size]
                
                # Convertir a tensor y normalizar
                batch_tensors = []
                for frame in batch_frames:
                    # Asegurar que frame esté en [0, 255] uint8
                    if frame.dtype == np.float32 or frame.dtype == np.float64:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                    
                    # Aplicar transform
                    frame_tensor = transform(frame)
                    batch_tensors.append(frame_tensor)
                
                batch_tensor = torch.stack(batch_tensors).to(device)
                
                # Extraer features
                features = model(batch_tensor)  # (B, 1024)
                all_features.append(features.cpu().numpy())
            
            # Concatenar todos los features
            visual_features = np.concatenate(all_features, axis=0)  # (T, 1024)
            
            # Validar shape
            assert visual_features.shape == (T, 1024), f"Shape inesperado: {visual_features.shape}"
            
            # Guardar en float16 para ahorrar espacio
            visual_features = visual_features.astype(np.float16)
            np.save(output_file, visual_features)
            
            processed += 1
            
        except Exception as e:
            logger.error(f" Error procesando {clip_name}: {str(e)}")
            errors += 1
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f" Procesamiento completado!")
    logger.info(f"   ✓ Procesados: {processed}")
    logger.info(f"   ⊘ Omitidos (ya existían): {skipped}")
    logger.info(f"   ✗ Errores: {errors}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Precomputa features visuales con ResNet101")
    parser.add_argument(
        "--clips_dir",
        type=Path,
        default=None,
        help=f"Directorio con archivos *_frames.npy (default: {config.data_paths.clips})"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=f"Directorio de salida para *_visual.npy (default: {config.data_paths.features_visual})"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help=f"Device para procesamiento (default: {config.training.device})"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size para procesamiento"
    )
    
    args = parser.parse_args()
    
    logger.info("Iniciando precompute de features visuales")
    logger.info(f"   Clips dir: {args.clips_dir or config.data_paths.clips}")
    logger.info(f"   Output dir: {args.output_dir or config.data_paths.features_visual}")
    logger.info(f"   Device: {args.device or config.training.device}")
    logger.info(f"   Batch size: {args.batch_size}\n")
    
    precompute_visual_features(
        clips_dir=args.clips_dir,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size
    )