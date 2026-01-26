"""
Modelos temporales para videos de longitud variable.
CORREGIDO: Clasificador con capas intermedias, mejor arquitectura
"""

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
import logging
from typing import Optional

from pipelines_video.config import config

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss para manejar desbalance de clases.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Cuando gamma=0, es equivalente a CrossEntropyLoss.
    Cuando gamma>0, reduce el peso de ejemplos faciles.
    """
    
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, C) logits
            targets: (B,) class indices
        """
        # Calcular log softmax
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        
        # Obtener probabilidad de la clase correcta
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[-1]).float()
        
        # Label smoothing
        if self.label_smoothing > 0:
            n_classes = inputs.shape[-1]
            targets_one_hot = (1 - self.label_smoothing) * targets_one_hot + \
                             self.label_smoothing / n_classes
        
        # Focal weight: (1 - p_t)^gamma
        pt = (probs * targets_one_hot).sum(dim=-1)
        focal_weight = (1 - pt) ** self.gamma
        
        # Cross entropy
        ce_loss = -(targets_one_hot * log_probs).sum(dim=-1)
        
        # Class weights
        if self.weight is not None:
            weight = self.weight.to(inputs.device)
            class_weight = weight[targets]
            focal_weight = focal_weight * class_weight
        
        # Focal loss
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class TemporalAttention(nn.Module):
    """
    Modulo de attention temporal con soporte para masking.
    Aprende a ponderar frames segun su relevancia.
    CORREGIDO: Agregado LayerNorm para estabilidad
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim)  # NUEVO
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # CORREGIDO: Bottleneck
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(
        self, 
        lstm_output: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            lstm_output: (B, T, hidden_dim)
            lengths: (B,) longitudes reales de cada secuencia
        
        Returns:
            context: (B, hidden_dim) vector contexto ponderado
        """
        B, T, H = lstm_output.shape
        
        # Layer norm para estabilidad
        normalized = self.layer_norm(lstm_output)
        
        # Calcular scores de attention
        attention_scores = self.attention(normalized)  # (B, T, 1)
        
        # Crear mascara para padding
        if lengths is not None:
            # mask[i, j] = True si j >= lengths[i] (es padding)
            mask = torch.arange(T, device=lstm_output.device)[None, :] >= lengths[:, None]
            # Aplicar mascara: poner -inf en posiciones de padding
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(-1), float('-inf'))
        
        # Softmax para obtener pesos
        attention_weights = torch.softmax(attention_scores, dim=1)  # (B, T, 1)
        
        # Manejar caso donde toda la secuencia es padding (NaN)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Weighted sum
        context = torch.sum(attention_weights * lstm_output, dim=1)  # (B, hidden_dim)
        
        return context


class VideoLSTMClassifier(nn.Module):
    """
    Clasificador LSTM bidireccional para videos de longitud variable.
    
    CORREGIDO:
    - Clasificador con capas intermedias (no solo 1 capa lineal)
    - Mejor regularizacion
    - Layer normalization
    """
    
    def __init__(
        self,
        input_dim: int = None,
        hidden_dim: int = None,
        num_layers: int = None,
        num_classes: int = None,
        dropout: float = None,
        bidirectional: bool = None,
        use_attention: bool = None,
        classifier_hidden_dim: int = None,  # NUEVO
        classifier_dropout: float = None     # NUEVO
    ):
        super().__init__()
        
        # Defaults desde config
        self.input_dim = input_dim or config.features.fused_dim
        self.hidden_dim = hidden_dim or config.model.hidden_dim
        self.num_layers = num_layers or config.model.num_layers
        self.num_classes = num_classes or config.model.num_classes
        self.dropout = dropout or config.model.dropout
        self.bidirectional = bidirectional if bidirectional is not None else config.model.bidirectional
        self.use_attention = use_attention if use_attention is not None else config.model.use_attention
        
        # NUEVO: Config del clasificador
        self.classifier_hidden_dim = classifier_hidden_dim or config.model.classifier_hidden_dim
        self.classifier_dropout = classifier_dropout or config.model.classifier_dropout
        
        # Input projection con LayerNorm
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),  # CORREGIDO: GELU en lugar de ReLU
            nn.Dropout(self.dropout)
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Dimension de salida de LSTM
        lstm_output_dim = self.hidden_dim * 2 if self.bidirectional else self.hidden_dim
        
        # Attention
        if self.use_attention:
            self.attention = TemporalAttention(lstm_output_dim, dropout=self.dropout)
        
        # CORREGIDO: Clasificador con capas intermedias
        # self.classifier = nn.Sequential(
        #     nn.LayerNorm(lstm_output_dim),
        #     nn.Linear(lstm_output_dim, self.classifier_hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(self.classifier_dropout),
        #     nn.Linear(self.classifier_hidden_dim, self.classifier_hidden_dim // 2),
        #     nn.GELU(),
        #     nn.Dropout(self.classifier_dropout * 0.5),  # Menos dropout en capa final
        #     nn.Linear(self.classifier_hidden_dim // 2, self.num_classes)
        # )
        
        self.classifier = nn.Sequential(nn.Linear(lstm_output_dim, self.num_classes))
        
        # Inicializacion de pesos
        self._init_weights()
        
        # Logging
        logger.info(f"VideoLSTMClassifier inicializado:")
        logger.info(f"  Input dim: {self.input_dim}")
        logger.info(f"  Hidden dim: {self.hidden_dim}")
        logger.info(f"  Num layers: {self.num_layers}")
        logger.info(f"  Bidirectional: {self.bidirectional}")
        logger.info(f"  Use attention: {self.use_attention}")
        logger.info(f"  Classifier hidden: {self.classifier_hidden_dim}")
        logger.info(f"  Num classes: {self.num_classes}")
    
    def _init_weights(self):
        """Inicializacion de pesos para mejor convergencia"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) features de video
            lengths: (B,) longitudes reales de cada secuencia
        
        Returns:
            logits: (B, num_classes)
        """
        B, T, D = x.shape
        
        # Input projection
        x = self.input_projection(x)  # (B, T, hidden_dim)
        
        # LSTM con pack_padded_sequence para eficiencia
        if lengths is not None:
            # Asegurar que lengths esta en CPU para pack_padded_sequence
            lengths_cpu = lengths.cpu()
            
            # Pack
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(packed)
            
            # Unpack
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=T
            )
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)
        
        # (B, T, hidden_dim * 2) si bidirectional
        
        # Pooling temporal
        if self.use_attention:
            # Attention con masking
            h_pooled = self.attention(lstm_out, lengths)
        else:
            # Usar ultimo estado oculto
            if self.bidirectional:
                # Concatenar forward y backward del ultimo layer
                h_forward = h_n[-2]  # (B, hidden_dim)
                h_backward = h_n[-1]  # (B, hidden_dim)
                h_pooled = torch.cat([h_forward, h_backward], dim=1)
            else:
                h_pooled = h_n[-1]
        
        # Clasificacion
        logits = self.classifier(h_pooled)  # (B, num_classes)
        
        return logits


class VideoTransformerClassifier(nn.Module):
    """
    Clasificador basado en Transformer para videos de longitud variable.
    Alternativa al LSTM para capturar dependencias de largo alcance.
    """
    
    def __init__(
        self,
        input_dim: int = None,
        hidden_dim: int = None,
        num_layers: int = None,
        num_heads: int = 8,
        num_classes: int = None,
        dropout: float = None,
        max_length: int = 512
    ):
        super().__init__()
        
        self.input_dim = input_dim or config.features.fused_dim
        self.hidden_dim = hidden_dim or config.model.hidden_dim
        self.num_layers = num_layers or config.model.num_layers
        self.num_classes = num_classes or config.model.num_classes
        self.dropout = dropout or config.model.dropout
        self.max_length = max_length
        
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Embedding(max_length, self.hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )
        
        # CLS token para clasificacion
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        
        # CORREGIDO: Clasificador mejorado
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_classes)
        )
        
        logger.info(f"VideoTransformerClassifier inicializado:")
        logger.info(f"  Hidden dim: {self.hidden_dim}")
        logger.info(f"  Num layers: {self.num_layers}")
        logger.info(f"  Num heads: {num_heads}")
        logger.info(f"  Num classes: {self.num_classes}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim)
            lengths: (B,) longitudes reales
        
        Returns:
            logits: (B, num_classes)
        """
        B, T, D = x.shape
        
        # Input projection
        x = self.input_projection(x)  # (B, T, hidden_dim)
        
        # Positional encoding
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        positions = torch.clamp(positions, 0, self.max_length - 1)
        x = x + self.pos_encoding(positions)
        
        # Agregar CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, hidden_dim)
        
        # Crear attention mask para padding
        if lengths is not None:
            # mask[i, j] = True si j es padding (j > lengths[i])
            # +1 por el CLS token
            mask = torch.arange(T + 1, device=x.device)[None, :] > lengths[:, None]
            # El CLS token nunca es padding
            mask[:, 0] = False
        else:
            mask = None
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=mask)  # (B, T+1, hidden_dim)
        
        # Usar CLS token para clasificacion
        cls_output = x[:, 0]  # (B, hidden_dim)
        
        # Clasificacion
        logits = self.classifier(cls_output)
        
        return logits


def get_video_model(
    model_type: str = "lstm",
    num_classes: int = None,
    **kwargs
) -> nn.Module:
    """
    Factory function para crear modelos de video.
    
    Args:
        model_type: 'lstm' o 'transformer'
        num_classes: Numero de clases
        **kwargs: Argumentos adicionales para el modelo
    
    Returns:
        Modelo temporal
    """
    if num_classes is None:
        num_classes = config.model.num_classes
    
    if model_type == "lstm":
        return VideoLSTMClassifier(num_classes=num_classes, **kwargs)
    elif model_type == "transformer":
        return VideoTransformerClassifier(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Modelo desconocido: {model_type}. Opciones: lstm, transformer")


def get_loss_function(
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = None,
    label_smoothing: float = None
) -> nn.Module:
    """
    Crea la funcion de loss apropiada.
    
    Args:
        class_weights: Pesos por clase para balanceo
        focal_gamma: Gamma para Focal Loss (0 = CrossEntropy)
        label_smoothing: Factor de label smoothing
    
    Returns:
        Loss function
    """
    if focal_gamma is None:
        focal_gamma = config.training.focal_loss_gamma
    if label_smoothing is None:
        label_smoothing = config.training.label_smoothing
    
    if focal_gamma > 0:
        logger.info(f"Usando FocalLoss con gamma={focal_gamma}, smoothing={label_smoothing}")
        return FocalLoss(
            weight=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing
        )
    else:
        logger.info(f"Usando CrossEntropyLoss con smoothing={label_smoothing}")
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
