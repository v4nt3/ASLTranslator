"""
Script auxiliar para crear archivo de mapeo de clases
Si tienes un archivo con los nombres de las clases, este script genera el JSON necesario
"""

import json
from pathlib import Path


def create_class_mapping_from_list(class_names_file: Path, output_file: Path):
    """
    Crea archivo JSON de mapeo desde un archivo de texto con nombres de clases
    
    Args:
        class_names_file: Archivo .txt con un nombre de clase por línea
        output_file: Archivo JSON de salida
    """
    with open(class_names_file, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    # Crear mapeo {class_id: nombre}
    mapping = {str(i): name for i, name in enumerate(class_names)}
    
    # Guardar
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    print(f"Mapeo creado exitosamente: {len(mapping)} clases")
    print(f"Guardado en: {output_file}")


def create_class_mapping_from_folders(dataset_path: Path, output_file: Path):
    """
    Crea mapeo desde estructura de carpetas (cada carpeta = una clase)
    
    Args:
        dataset_path: Directorio raíz del dataset
        output_file: Archivo JSON de salida
    """
    # Obtener nombres de carpetas
    class_folders = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])
    
    # Crear mapeo
    mapping = {str(i): name for i, name in enumerate(class_folders)}
    
    # Guardar
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    print(f"Mapeo creado exitosamente: {len(mapping)} clases")
    print(f"Guardado en: {output_file}")


def create_class_mapping_from_videos_json(videos_json_path: Path, output_file: Path):
    """
    Crea mapeo desde el archivo JSON de videos que ya contiene class_id y class_name
    
    Args:
        videos_json_path: Archivo JSON con estructura {"videos": [{"class_id": 0, "class_name": "1 dollar", ...}, ...]}
        output_file: Archivo JSON de salida con formato {class_id: class_name}
    """
    # Cargar el JSON de videos
    with open(videos_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    videos = data.get("videos", [])
    
    if not videos:
        print("Error: No se encontraron videos en el archivo JSON")
        return
    
    # Crear mapeo único {class_id: class_name}
    mapping = {}
    for video in videos:
        class_id = str(video.get("class_id"))
        class_name = video.get("class_name")
        
        if class_id and class_name:
            mapping[class_id] = class_name
    
    # Guardar
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Mapeo creado exitosamente: {len(mapping)} clases únicas")
    print(f"✓ Guardado en: {output_file}")
    print(f"\nPrimeras 5 clases:")
    for class_id in sorted(mapping.keys(), key=int)[:5]:
        print(f"  {class_id}: {mapping[class_id]}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Crear archivo de mapeo de clases")
    parser.add_argument("--mode", choices=["list", "folders", "videos_json"], required=True,
                       help="Modo: 'list' (txt), 'folders' (carpetas) o 'videos_json' (desde JSON existente)")
    parser.add_argument("--input", type=Path, required=True,
                       help="Archivo .txt, directorio con carpetas, o JSON de videos")
    parser.add_argument("--output", type=Path, default=Path("data/class_mapping.json"),
                       help="Archivo JSON de salida")
    
    args = parser.parse_args()
    
    if args.mode == "list":
        create_class_mapping_from_list(args.input, args.output)
    elif args.mode == "folders":
        create_class_mapping_from_folders(args.input, args.output)
    else:  # videos_json
        create_class_mapping_from_videos_json(args.input, args.output)
