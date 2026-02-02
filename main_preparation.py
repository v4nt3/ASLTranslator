"""
Punto de entrada principal del pipeline ASL Multimodal
Ejecuta todo el pipeline: preprocessing → entrenamiento → evaluación
"""

import argparse
import logging
from pipelines.config import config
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging():
    """Configura el sistema de logging"""
    log_file = config.output_paths.tensorboard_logs.parent / "pipeline.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)


def generate_metadata():
    """Genera automáticamente dataset_meta.json a partir de los videos"""
    logger.info("=" * 80)
    logger.info("GENERANDO METADATOS DEL DATASET")
    logger.info("=" * 80)
    
    # from scripts.generate_metadata import scan_folders_and_videos, validate_metadata, save_metadata, print_summary
    
    # metadata_path = config.data_paths.raw_videos / "dataset_meta.json"
    
    # # Si ya existe, preguntar si regenerar
    # if metadata_path.exists():
    #     logger.info(f"Archivo de metadatos ya existe: {metadata_path}")
    #     logger.info("Regenerando para incluir nuevos videos")
    
    # # Escanear, validar y guardar
    # videos_info = scan_folders_and_videos()
    
    # if not videos_info:
    #     logger.error("No se encontraron videos para procesar")
    #     return False
    
    # if not validate_metadata(videos_info):
    #     logger.error("Validación de metadatos fallida")
    #     return False
    
    # save_metadata(videos_info, metadata_path)
    # print_summary(videos_info)
    
    logger.info("✓ Metadatos generados exitosamente")
    return True


def preprocess_data():
    """Ejecuta el preprocesamiento de datos"""
    logger.info("=" * 80)
    logger.info("INICIANDO PREPROCESAMIENTO DE DATOS")
    logger.info("=" * 80)
    
    from pipelines.data_preparation import DataPreprocessor
    
    preprocessor = DataPreprocessor(config)
    
    # 1. Extraer frames
    logger.info("Extrayendo frames de videos")
    # preprocessor.extract_frames()  # Descomenta cuando tengas los videos
    
    # 2. Extraer keypoints
    logger.info("Extrayendo keypoints con MediaPipe")
    # preprocessor.extract_keypoints()  # Descomenta cuando tengas los frames
    
    # 3. Crear clips y dataset CSV
    logger.info("Creando clips múltiples y archivo dataset.csv")
    #preprocessor.create_clips_and_csv()  # Descomenta cuando tengas los keypoints
    
    logger.info("Preprocesamiento completado!")


def train_model():
    """Entrena el modelo multimodal"""
    logger.info("=" * 80)
    logger.info("INICIANDO ENTRENAMIENTO DEL MODELO")
    logger.info("=" * 80)
    
    from pipelines.training import Trainer
    
    trainer = Trainer(config)
    trainer.train()
    
    logger.info("Entrenamiento completado!")


def evaluate_model():
    """Evalúa el modelo entrenado"""
    logger.info("=" * 80)
    logger.info("INICIANDO EVALUACIÓN DEL MODELO")
    logger.info("=" * 80)
    
    from pipelines.evaluation import Evaluator
    
    evaluator = Evaluator(config)
    evaluator.evaluate()
    
    logger.info("Evaluación completada!")


def main():
    """Función principal que orquesta todo el pipeline"""
    parser = argparse.ArgumentParser(
        description="Pipeline ASL Multimodal - Clasificación de Lenguaje de Señas"
    )
    
    parser.add_argument(
        "--mode",
        choices=["preprocess", "train", "evaluate", "full"],
        default="full",
        help="Modo de ejecución: preprocess, train, evaluate o full (todos)"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    logger.info(f"Configuración cargada desde config.py")
    logger.info(f"Directorio raíz: {config.data_paths.raw_videos.parent.parent}")
    logger.info(f"Modo de ejecución: {args.mode}")
    
    try:
        if args.mode in ["preprocess", "full"]:
            if not generate_metadata():
                logger.error("No se puede continuar sin metadatos válidos")
                return
        
        if args.mode in ["preprocess", "full"]:
            preprocess_data()
        
        if args.mode in ["train", "full"]:
            train_model()
        
        if args.mode in ["evaluate", "full"]:
            evaluate_model()
        
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("=" * 80)
    
    except Exception as e:
        logger.error(f"Error en el pipeline: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
