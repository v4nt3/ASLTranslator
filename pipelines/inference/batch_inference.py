"""
Script para inferencia por lotes sobre múltiples videos
Útil para evaluar en datasets completos
"""

import argparse
from pathlib import Path
import json
from tqdm import tqdm
import logging
from realtime_sign_detector import SignLanguageDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def batch_inference(
    video_dir: Path,
    model_path: str,
    classes_path: str,
    output_dir: Path,
    feature_dim: int = 1152,
    confidence: float = 0.3
):
    """
    Realiza inferencia sobre todos los videos en un directorio
    
    Args:
        video_dir: Directorio con videos
        model_path: Ruta al modelo
        classes_path: Ruta a clases
        output_dir: Directorio para guardar resultados
        feature_dim: Dimensión de features
        confidence: Umbral de confianza
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Inicializar detector
    detector = SignLanguageDetector(
        model_path=model_path,
        class_names_path=classes_path,
        feature_dim=feature_dim,
        confidence_threshold=confidence
    )
    
    # Buscar todos los videos
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(f"**/{ext}"))
    
    logger.info(f"Encontrados {len(video_files)} videos para procesar")
    
    # Procesar cada video
    results_summary = []
    
    for video_file in tqdm(video_files, desc="Procesando videos"):
        video_name = video_file.stem
        
        try:
            # Reiniciar buffers
            detector.frame_buffer.clear()
            detector.keypoint_buffer.clear()
            
            # Procesar video
            detections, sentence = detector.process_video_stream(
                video_source=str(video_file),
                display=False
            )
            
            # Guardar resultados individuales
            result = {
                "video": video_name,
                "sentence": " ".join(sentence),
                "num_signs": len(sentence),
                "detections": [
                    {
                        "sign": sign,
                        "confidence": float(conf),
                        "timestamp": float(ts)
                    }
                    for sign, conf, ts in detections
                ]
            }
            
            output_file = output_dir / f"{video_name}_result.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            results_summary.append({
                "video": video_name,
                "sentence": result["sentence"],
                "num_signs": len(sentence)
            })
            
            logger.info(f"✓ {video_name}: {len(sentence)} señas detectadas")
            
        except Exception as e:
            logger.error(f"✗ Error procesando {video_name}: {e}")
            results_summary.append({
                "video": video_name,
                "error": str(e)
            })
    
    # Guardar resumen
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Procesamiento por lotes completado")
    logger.info(f"Videos procesados: {len(video_files)}")
    logger.info(f"Resultados guardados en: {output_dir}")
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Inferencia por lotes sobre múltiples videos")
    parser.add_argument(
        "--video_dir",
        type=Path,
        required=True,
        help="Directorio con videos a procesar"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Ruta al checkpoint del modelo"
    )
    parser.add_argument(
        "--classes",
        type=str,
        required=True,
        help="Ruta al archivo de clases"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("inference_results"),
        help="Directorio para guardar resultados"
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=1152,
        help="Dimensión de features"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Umbral de confianza"
    )
    
    args = parser.parse_args()
    
    batch_inference(
        video_dir=args.video_dir,
        model_path=args.model,
        classes_path=args.classes,
        output_dir=args.output_dir,
        feature_dim=args.feature_dim,
        confidence=args.confidence
    )


if __name__ == "__main__":
    main()
