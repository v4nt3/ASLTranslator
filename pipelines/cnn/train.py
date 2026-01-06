"""
Script OPTIMIZADO para entrenar modelo temporal usando SOLO pose features
OPTIMIZACIONES:
1. Escaneo único del directorio de features (en lugar de 80k búsquedas individuales)
2. Lookup en memoria O(1) en lugar de múltiples exists()
3. Procesamiento por batches para reducir tiempo de carga
4. Detección automática de dimensiones de features
5. Modelo adaptable a MLP (128 dims) o CNN (256 dims)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import json
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

from config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PoseOnlyTemporalModel(nn.Module):
    """
    Modelo temporal que usa pose features
    Soporta MLP (128 dims) o CNN (256 dims)
    """
    def __init__(
        self,
        input_dim: int = 128,  # Ahora configurable, detectado automáticamente
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2286,
        dropout: float = 0.5,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # LSTM temporal
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Classifier con más regularización
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, max_seq_len, input_dim]
            lengths: [batch_size]
        """
        batch_size, max_seq_len, _ = features.shape
        
        # Reshape para batch norm: [batch*seq, features]
        features_reshaped = features.reshape(-1, self.input_dim)
        features_normalized = self.input_norm(features_reshaped)
        features = features_normalized.reshape(batch_size, max_seq_len, self.input_dim)
        
        # Pack sequence
        packed = pack_padded_sequence(
            features, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Usar hidden state del último layer
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        # Classifier
        logits = self.classifier(hidden)
        
        return logits


class PoseOnlyTemporalDataset(Dataset):
    """
    Dataset OPTIMIZADO que carga pose features usando escaneo único del directorio
    """
    def __init__(
        self,
        pose_features_dir: Path,
        metadata_path: Path,
        split: str = 'train'
    ):
        self.pose_features_dir = Path(pose_features_dir)
        self.split = split
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if isinstance(metadata, dict):
            if 'videos' in metadata:
                video_metadata = metadata['videos']
            elif 'dataset' in metadata:
                video_metadata = metadata['dataset']
            else:
                raise ValueError("Metadata debe tener clave 'videos' o 'dataset'")
        elif isinstance(metadata, list):
            video_metadata = metadata
        else:
            raise ValueError(f"Formato de metadata no válido: {type(metadata)}")
        
        logger.info(f"Cargando dataset desde metadata original con {len(video_metadata)} videos")
        
        self.class_to_id = {}
        self.id_to_class = {}
        
        for item in video_metadata:
            class_name = item['class_name']
            class_id = item['class_id']
            
            if class_name not in self.class_to_id:
                self.class_to_id[class_name] = class_id
                self.id_to_class[class_id] = class_name
        
        self.num_classes = len(self.class_to_id)
        logger.info(f"Dataset con {self.num_classes} clases únicas")
        
        expected_ids = set(range(self.num_classes))
        actual_ids = set(self.id_to_class.keys())
        
        if expected_ids != actual_ids:
            logger.warning(f"⚠️ class_ids no son consecutivos!")
            logger.info("Remapeando class_ids a valores consecutivos...")
            old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted(actual_ids))}
            
            new_class_to_id = {}
            new_id_to_class = {}
            
            for class_name, old_id in self.class_to_id.items():
                new_id = old_to_new[old_id]
                new_class_to_id[class_name] = new_id
                new_id_to_class[new_id] = class_name
            
            self.class_to_id = new_class_to_id
            self.id_to_class = new_id_to_class
            
            logger.info(f"✓ Remapeado a class_ids consecutivos: 0 a {self.num_classes-1}")
        
        logger.info("=" * 60)
        logger.info("ESCANEANDO ARCHIVOS DE FEATURES (esto puede tomar unos minutos)...")
        logger.info("=" * 60)
        self.file_index = self._build_file_index()
        logger.info(f"✓ Índice construido con {len(self.file_index)} archivos")
        
        logger.info("Construyendo lista de samples...")
        self.samples = []
        missing_files = []
        class_counts = Counter()
        
        for item in tqdm(video_metadata, desc="Procesando metadata"):
            class_name = item['class_name']
            class_id = self.class_to_id[class_name]
            video_file = item['video_file']
            
            # Buscar archivo usando índice (O(1) lookup)
            pose_file = self._lookup_pose_file(video_file, class_name)
            
            if pose_file:
                self.samples.append({
                    'pose_path': pose_file,
                    'class_id': class_id,
                    'class_name': class_name,
                    'video_file': video_file
                })
                class_counts[class_id] += 1
            else:
                missing_files.append((class_name, video_file))
        
        if missing_files:
            logger.warning(f"⚠️ No se encontraron {len(missing_files)} archivos de pose features")
            if len(missing_files) <= 10:
                logger.warning(f"Archivos faltantes: {missing_files}")
            else:
                logger.warning(f"Primeros 10 ejemplos: {missing_files[:10]}")
        
        self.class_counts = class_counts
        
        logger.info("=" * 60)
        logger.info(f"✓ Cargados {len(self.samples)} samples para split '{split}'")
        logger.info(f"✓ Clases con samples: {len(class_counts)}/{self.num_classes}")
        logger.info(f"✓ Porcentaje de videos encontrados: {100 * len(self.samples) / len(video_metadata):.1f}%")
        logger.info("=" * 60)
        
        # Validar que todas las clases tienen al menos un sample
        missing_classes = []
        for class_id in range(self.num_classes):
            if class_id not in class_counts:
                class_name = self.id_to_class.get(class_id, f"Unknown_{class_id}")
                missing_classes.append((class_id, class_name))
        
        if missing_classes:
            logger.error(f"❌ {len(missing_classes)} clases sin samples!")
            logger.error(f"Esto causará errores en entrenamiento.")
            logger.error(f"Primeras 10 clases faltantes: {missing_classes[:10]}")
    
    def _build_file_index(self) -> Dict[str, Path]:
        """
        Escanea el directorio de features UNA SOLA VEZ y construye índice
        Retorna dict {filename_key: full_path}
        """
        file_index = {}
        
        # Buscar todos los archivos .npy recursivamente
        all_npy_files = list(self.pose_features_dir.rglob("*.npy"))
        
        logger.info(f"Encontrados {len(all_npy_files)} archivos .npy")
        
        for file_path in tqdm(all_npy_files, desc="Indexando archivos"):
            filename = file_path.name
            
            # Crear múltiples keys para diferentes patrones de búsqueda
            # Pattern 1: nombre completo del archivo
            file_index[filename] = file_path
            
            # Pattern 2: sin sufijo _pose o _keypoints
            if '_pose.npy' in filename:
                base_key = filename.replace('_pose.npy', '.mp4')
                file_index[base_key] = file_path
            elif '_keypoints.npy' in filename:
                base_key = filename.replace('_keypoints.npy', '.mp4')
                file_index[base_key] = file_path
            
            # Pattern 3: solo el ID del video (antes del primer guion)
            if '-' in filename:
                video_id = filename.split('-')[0]
                if video_id not in file_index:  # No sobreescribir si ya existe
                    file_index[video_id] = file_path
        
        return file_index
    
    def _lookup_pose_file(self, video_file: str, class_name: str) -> Optional[Path]:
        """
        Busca archivo usando el índice pre-construido (O(1) lookup)
        """
        # Extraer video_id del nombre del archivo
        video_id = video_file.replace('.mp4', '')
        
        # Intentar múltiples patrones de búsqueda
        search_patterns = [
            # Pattern 1: ID-CLASSNAME_pose.npy (formato observado en la imagen)
            f"{video_id}-{class_name}_pose.npy",
            f"{video_id}-{class_name.upper()}_pose.npy",
            
            # Pattern 2: videofile con sufijo pose
            video_file.replace('.mp4', '_pose.npy'),
            
            # Pattern 3: videofile con sufijo keypoints
            video_file.replace('.mp4', '_keypoints.npy'),
            
            # Pattern 4: solo el video_id
            video_id,
            
            # Pattern 5: video_file original (sin cambios)
            video_file,
        ]
        
        for pattern in search_patterns:
            if pattern in self.file_index:
                return self.file_index[pattern]
        
        return None
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Retorna distribución de clases {class_id: count}"""
        return dict(self.class_counts)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        sample = self.samples[idx]
        
        # Cargar pose features
        pose_features = np.load(sample['pose_path']).astype(np.float32)  # (T, 128)
        
        # Convertir a tensor
        features = torch.from_numpy(pose_features)
        
        # Class ID
        class_id = sample['class_id']
        
        # Longitud de la secuencia
        length = features.shape[0]
        
        return features, class_id, length


def collate_fn(batch):
    """Collate function con padding dinámico"""
    features_list, class_ids, lengths = zip(*batch)
    
    max_length = max(lengths)
    batch_size = len(features_list)
    feature_dim = features_list[0].shape[1]
    
    features_padded = torch.zeros(batch_size, max_length, feature_dim)
    
    for i, (features, length) in enumerate(zip(features_list, lengths)):
        features_padded[i, :length, :] = features
    
    class_ids = torch.tensor(class_ids, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return features_padded, class_ids, lengths


def create_stratified_splits(
    dataset: PoseOnlyTemporalDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Crea splits estratificados por clase para mantener distribución balanceada
    """
    np.random.seed(seed)
    
    class_to_indices = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        class_id = sample['class_id']
        class_to_indices[class_id].append(idx)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for class_id, indices in class_to_indices.items():
        n_samples = len(indices)
        
        # Shuffle indices
        np.random.shuffle(indices)
        
        # Calcular splits
        n_train = max(1, int(n_samples * train_ratio))
        n_val = max(1, int(n_samples * val_ratio))
        n_test = max(1, n_samples - n_train - n_val)
        
        # Ajustar si no hay suficientes samples
        if n_train + n_val + n_test > n_samples:
            if n_samples >= 3:
                n_train = max(1, int(n_samples * 0.6))
                n_val = max(1, int(n_samples * 0.2))
                n_test = n_samples - n_train - n_val
            elif n_samples == 2:
                n_train, n_val, n_test = 1, 1, 0
            else:  # n_samples == 1
                n_train, n_val, n_test = 1, 0, 0
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        if n_test > 0:
            test_indices.extend(indices[n_train + n_val:])
    
    logger.info(f"✓ Split estratificado: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    return train_indices, val_indices, test_indices


def create_weighted_sampler(dataset: Dataset, indices: List[int]) -> WeightedRandomSampler:
    """
    Crea un WeightedRandomSampler SUAVIZADO para balancear clases en entrenamiento
    """
    class_counts = Counter()
    for idx in indices:
        _, class_id, _ = dataset[idx]
        class_counts[class_id] += 1
    
    weights = []
    for idx in indices:
        _, class_id, _ = dataset[idx]
        # Suavizado: sqrt(1/count) en lugar de 1/count
        weight = 1.0 / np.sqrt(class_counts[class_id])
        weights.append(weight)
    
    # En lugar de muestrear len(indices), solo muestreamos 50% cada época
    num_samples = len(indices) // 2
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True
    )
    
    logger.info(f"✓ WeightedRandomSampler SUAVIZADO creado")
    logger.info(f"  Muestras por época: {num_samples} (50% del dataset)")
    logger.info(f"  Clase más frecuente: {max(class_counts.values())} samples")
    logger.info(f"  Clase menos frecuente: {min(class_counts.values())} samples")
    
    return sampler


def visualize_class_distribution(dataset: PoseOnlyTemporalDataset, output_path: Path):
    """Genera visualización de distribución de clases"""
    class_counts = dataset.get_class_distribution()
    
    counts = list(class_counts.values())
    
    plt.figure(figsize=(12, 6))
    plt.hist(counts, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(counts), color='r', linestyle='--', linewidth=2, label=f'Media: {np.mean(counts):.1f}')
    plt.axvline(np.median(counts), color='g', linestyle='--', linewidth=2, label=f'Mediana: {np.median(counts):.1f}')
    plt.xlabel('Número de videos por clase')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Videos por Clase')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✓ Gráfico de distribución guardado en: {output_path}")


def create_balanced_dataloaders(
    pose_features_dir: Path,
    metadata_path: Path,
    batch_size: int,
    num_workers: int,
    train_split: float = 0.7,
    val_split: float = 0.15,
    use_class_balancing: bool = False  # Deshabilitado por defecto
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Crea dataloaders CON split estratificado y balanceo OPCIONAL
    """
    logger.info("=" * 60)
    logger.info("CARGANDO DATASET CON OPTIMIZACIONES")
    logger.info("=" * 60)
    
    # Cargar dataset completo
    full_dataset = PoseOnlyTemporalDataset(
        pose_features_dir=pose_features_dir,
        metadata_path=metadata_path,
        split='full'
    )
    
    num_classes = full_dataset.num_classes
    
    # Visualizar distribución
    vis_path = Path('outputs') / 'class_distribution_analysis.png'
    vis_path.parent.mkdir(exist_ok=True)
    visualize_class_distribution(full_dataset, vis_path)
    
    # Split estratificado
    train_indices, val_indices, test_indices = create_stratified_splits(
        full_dataset,
        train_ratio=train_split,
        val_ratio=val_split,
        seed=config.data.random_seed
    )
    
    from torch.utils.data import Subset
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    if use_class_balancing:
        train_sampler = create_weighted_sampler(full_dataset, train_indices)
        shuffle_train = False
        logger.info("✓ Usando WeightedRandomSampler SUAVIZADO para balanceo")
    else:
        train_sampler = None
        shuffle_train = True
        logger.info("✓ Sin balanceo artificial - usando distribución natural")
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train if train_sampler is None else False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=config.training.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=config.training.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=config.training.pin_memory
    )
    
    return train_loader, val_loader, test_loader, num_classes


def detect_feature_dimensions(pose_features_dir: Path) -> int:
    """
    Detecta automáticamente las dimensiones de las features
    buscando un archivo .npy y leyendo su forma
    
    Returns:
        int: Dimensión de features (128 para MLP, 256 para CNN)
    """
    logger.info("Detectando dimensiones de features...")
    
    # Buscar primer archivo .npy disponible
    npy_files = list(pose_features_dir.glob("*.npy"))
    
    if not npy_files:
        logger.warning("No se encontraron archivos .npy, usando dimensión por defecto 128")
        return 128
    
    # Cargar primer archivo
    sample_file = npy_files[0]
    try:
        sample_features = np.load(sample_file)
        feature_dim = sample_features.shape[1] if len(sample_features.shape) == 2 else sample_features.shape[0]
        logger.info(f"✓ Dimensiones detectadas: {feature_dim} (desde {sample_file.name})")
        
        # Validar dimensiones esperadas
        if feature_dim == 128:
            logger.info("  → Usando features MLP (128 dims)")
        elif feature_dim == 256:
            logger.info("  → Usando features CNN (256 dims)")
        else:
            logger.warning(f"  → Dimensión inesperada: {feature_dim}, continuando de todas formas")
        
        return int(feature_dim)
    except Exception as e:
        logger.error(f"Error al leer archivo de muestra: {e}")
        logger.warning("Usando dimensión por defecto 128")
        return 128


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> Tuple[float, float]:
    """Entrena una época con soporte para AMP"""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for features, class_ids, lengths in pbar:
        features = features.to(device)
        class_ids = class_ids.to(device)
        lengths = lengths.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(features, lengths)
                loss = criterion(logits, class_ids)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(features, lengths)
            loss = criterion(logits, class_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == class_ids).sum().item()
        total += class_ids.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Valida el modelo"""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, class_ids, lengths in tqdm(dataloader, desc="Validating"):
            features = features.to(device)
            class_ids = class_ids.to(device)
            lengths = lengths.to(device)
            
            logits = model(features, lengths)
            loss = criterion(logits, class_ids)
            
            total_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == class_ids).sum().item()
            total += class_ids.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def train_model(
    pose_features_dir: Path,
    metadata_json: Path,
    output_path: Path,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 0.001,
    device: str = 'cuda',
    use_amp: bool = True,
    use_balancing: bool = False  # Por defecto False, el sampler causaba problemas
):
    """Entrena el modelo temporal con pose features"""
    
    logger.info("=" * 60)
    logger.info("CARGANDO DATASET CON CORRECCIONES")
    logger.info("=" * 60)
    
    feature_dim = detect_feature_dimensions(pose_features_dir)
    
    # Crear dataloaders con split estratificado
    train_loader, val_loader, test_loader, num_classes = create_balanced_dataloaders(
        pose_features_dir=pose_features_dir,
        metadata_path=metadata_json,
        batch_size=batch_size,
        num_workers=config.training.num_workers,
        use_class_balancing=use_balancing
    )
    
    model = PoseOnlyTemporalModel(
        input_dim=feature_dim,  # Usar dimensión detectada
        hidden_dim=256,
        num_layers=2,
        num_classes=num_classes,
        dropout=0.5,
        bidirectional=True
    ).to(device)
    
    logger.info(f"Modelo creado con {sum(p.numel() for p in model.parameters()):,} parámetros")
    logger.info(f"Input dim: {feature_dim}, Hidden dim: 256, Num classes: {num_classes}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=0.01  # Weight decay para regularización L2
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,  # Reducir a la mitad
        patience=5,   # Aumentado de 3 a 5
        min_lr=1e-6
    )
    
    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 15  # Parar si no mejora en 15 épocas
    
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    logger.info("=" * 60)
    logger.info("INICIANDO ENTRENAMIENTO CON REGULARIZACIÓN MEJORADA")
    logger.info("=" * 60)
    
    for epoch in range(1, epochs + 1):
        logger.info(f"\nEpoch {epoch}/{epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, scaler
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        overfit_gap = train_acc - val_acc
        logger.info(f"Overfitting Gap: {overfit_gap:.2f}%")
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset contador
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'num_classes': num_classes
            }, output_path)
            logger.info(f"✓ Mejor modelo guardado en {output_path}")
        else:
            patience_counter += 1
            logger.info(f"⚠️ No mejora por {patience_counter} épocas")
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"⛔ Early stopping activado después de {epoch} épocas")
                break
    
    # Test final
    logger.info("=" * 60)
    logger.info("EVALUACIÓN FINAL EN TEST SET")
    logger.info("=" * 60)
    
    # Cargar mejor modelo
    checkpoint = torch.load(output_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    logger.info(f"Best Val Acc: {checkpoint['val_acc']:.2f}%")
    logger.info(f"Best Train Acc: {checkpoint['train_acc']:.2f}%")
    
    logger.info("=" * 60)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_features', type=str, required=True)
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--output', type=str, default='outputs/pose_only_model.pth')
    parser.add_argument('--batch_size', type=int, default=64)  # Reducido de 128 a 32
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_workers', type=int, default=config.training.num_workers)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--use_class_balancing', action='store_true')  # Ahora es opt-in
    
    args = parser.parse_args()
    
    pose_features = Path(args.pose_features)
    metadata_path = Path(args.metadata)
    output_path = Path(args.output)
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    num_workers = args.num_workers
    use_amp = args.use_amp
    use_balancing = args.use_class_balancing
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Usando device: {device}")
    logger.info(f"AMP habilitado: {use_amp}")
    logger.info(f"Balanceo de clases: {use_balancing}")
    
    # Train model
    train_model(
        pose_features_dir=pose_features,
        metadata_json=metadata_path,
        output_path=output_path,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        device=device,
        use_amp=use_amp,
        use_balancing=use_balancing
    )
