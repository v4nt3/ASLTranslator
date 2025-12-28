import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PoseOnlyDataset(Dataset):
    """Dataset que carga solo pose features (128 dims) sin visual features"""
    
    def __init__(
        self,
        features_dir: str,
        metadata_path: str,
        split: str = 'train'
    ):
        self.features_dir = Path(features_dir)
        self.split = split
        
        # Cargar metadata
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            metadata = data.get('videos', data) if isinstance(data, dict) else data
        
        # Extraer información de clases
        self.class_names = sorted(list(set(item['class_name'] for item in metadata)))
        self.num_classes = len(self.class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        logger.info(f"Dataset inicializado con {self.num_classes} clases")
        
        # Preparar lista de samples
        self.samples = []
        for item in metadata:
            class_name = item['class_name']
            video_file = item['video_file']
            
            # Construir ruta a features fusionadas
            features_path = self.features_dir / class_name / video_file.replace('.mp4', '_fused.npy')
            
            if features_path.exists():
                self.samples.append({
                    'features_path': str(features_path),
                    'class_name': class_name,
                    'class_id': self.class_to_idx[class_name]
                })
        
        logger.info(f"Split '{split}': {len(self.samples)} samples cargados")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        sample = self.samples[idx]
        
        # Cargar features fusionadas (T, 1152)
        features_full = np.load(sample['features_path']).astype(np.float32)
        
        # Extraer SOLO pose features (últimas 128 dims)
        pose_features = features_full[:, -128:]  # Shape: (T, 128)
        
        # Convertir a tensor
        features_tensor = torch.from_numpy(pose_features)
        
        return features_tensor, sample['class_id'], features_tensor.shape[0]


def collate_fn_pose(batch: List[Tuple[torch.Tensor, int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function con padding para sequences de diferente longitud"""
    features, labels, lengths = zip(*batch)
    
    # Ordenar por longitud (descendente) para pack_padded_sequence
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    sorted_indices = torch.argsort(lengths_tensor, descending=True)
    
    # Reordenar
    features = [features[i] for i in sorted_indices]
    labels = torch.tensor([labels[i] for i in sorted_indices], dtype=torch.long)
    lengths_tensor = lengths_tensor[sorted_indices]
    
    # Padding
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True, padding_value=0.0)
    
    return features_padded, labels, lengths_tensor


class PoseOnlyTemporalModel(nn.Module):
    """Modelo temporal que usa solo pose features (128 dims)"""
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_classes: int = 2286,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Clasificador
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 por bidireccional
        
        logger.info(f"Modelo inicializado: input_dim={input_dim}, hidden_dim={hidden_dim}, "
                   f"num_layers={num_layers}, num_classes={num_classes}")
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, max_T, 128) - pose features
            lengths: (B,) - longitudes reales de cada secuencia
        
        Returns:
            logits: (B, num_classes)
        """
        # Pack sequence
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        
        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Usar último hidden state de ambas direcciones
        # hidden shape: (num_layers * 2, B, hidden_dim)
        hidden_fwd = hidden[-2]  # Forward del último layer
        hidden_bwd = hidden[-1]  # Backward del último layer
        
        # Concatenar
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)  # (B, hidden_dim * 2)
        
        # Clasificador
        hidden_cat = self.dropout(hidden_cat)
        logits = self.fc(hidden_cat)
        
        return logits


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """Entrena el modelo por una época"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for features, labels, lengths in pbar:
        features = features.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(features, lengths)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Métricas
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
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
    """Evalúa el modelo en el set de validación"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels, lengths in tqdm(dataloader, desc="Validating"):
            features = features.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            
            # Forward
            logits = model(features, lengths)
            loss = criterion(logits, labels)
            
            # Métricas
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Entrenar modelo temporal usando solo pose features')
    
    # Datos
    parser.add_argument('--features_dir', type=str, required=True,
                       help='Directorio con features fusionadas')
    parser.add_argument('--metadata', type=str, required=True,
                       help='Archivo JSON con metadata')
    
    # Modelo
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Dimensión oculta del LSTM')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Número de capas LSTM')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    
    # Entrenamiento
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directorio para guardar modelos')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Guardar checkpoint cada N épocas')
    
    args = parser.parse_args()
    
    # Crear directorio de output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Usando device: {device}")
    
    # Crear datasets
    logger.info("Cargando datasets...")
    train_dataset = PoseOnlyDataset(
        features_dir=args.features_dir,
        metadata_path=args.metadata,
        split='train'
    )
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_pose,
        pin_memory=True
    )
    
    # Crear modelo
    logger.info("Inicializando modelo...")
    model = PoseOnlyTemporalModel(
        input_dim=128,  # Solo pose features
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=train_dataset.num_classes,
        dropout=args.dropout
    ).to(device)
    
    # Criterio y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Entrenamiento
    logger.info("Iniciando entrenamiento...")
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        logger.info(f"Epoch {epoch}/{args.epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Update scheduler
        scheduler.step(train_acc)
        
        # Guardar checkpoint
        if epoch % args.save_every == 0 or train_acc > best_acc:
            checkpoint_path = output_dir / f'pose_only_model_epoch{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'num_classes': train_dataset.num_classes,
                'class_names': train_dataset.class_names,
                'hidden_dim': args.hidden_dim,
                'num_layers': args.num_layers,
                'dropout': args.dropout
            }, checkpoint_path)
            logger.info(f"Checkpoint guardado: {checkpoint_path}")
            
            if train_acc > best_acc:
                best_acc = train_acc
                best_path = output_dir / 'pose_only_model_best.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'num_classes': train_dataset.num_classes,
                    'class_names': train_dataset.class_names,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout
                }, best_path)
                logger.info(f"Mejor modelo guardado: {best_path} (acc: {best_acc:.2f}%)")
    
    logger.info(f"Entrenamiento completado! Mejor accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
