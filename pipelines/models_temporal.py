"""
Modelo temporal optimizado que trabaja con features precomputadas
Solo entrena LSTM + Classifier (sin backbones visuales)
"""

import torch #type: ignore
import torch.nn as nn  #type: ignore
import logging
import numpy as np

logger = logging.getLogger(__name__)


class AttentionModule(nn.Module):
    """Attention mechanism for LSTM outputs"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, lstm_output: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            lstm_output: (B, T, hidden_dim)
            lengths: (B,) optional sequence lengths
        
        Returns:
            context: (B, hidden_dim) weighted sum of lstm_output
        """
        # Compute attention scores
        attention_scores = self.attention(lstm_output)  # (B, T, 1)
        
        # Create mask for padding if lengths provided
        if lengths is not None:
            B, T = lstm_output.shape[0], lstm_output.shape[1]
            mask = torch.arange(T, device=lstm_output.device)[None, :] >= lengths[:, None]
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(-1), float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)  # (B, T, 1)
        
        # Compute weighted sum
        context = torch.sum(attention_weights * lstm_output, dim=1)  # (B, hidden_dim)
        
        return context


class TemporalLSTMClassifier(nn.Module):
    """
    Clasificador temporal con LSTM bidireccional
    Input: (B, T, 640) features precomputadas
    Output: (B, num_classes) logits
    """
    
    def __init__(
        self,
        input_dim: int = 1152,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_classes: int = 2286,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = False  # Add attention parameter
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention  # Store attention flag
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dimensión después de LSTM
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        if use_attention:
            self.attention = AttentionModule(lstm_output_dim)
        

        # # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # self.classifier = nn.Linear(lstm_output_dim, num_classes)
        
        logger.info(f"✓ TemporalLSTMClassifier inicializado")
        logger.info(f"   Input dim: {input_dim}")
        logger.info(f"   Hidden dim: {hidden_dim}")
        logger.info(f"   Num layers: {num_layers}")
        logger.info(f"   Bidirectional: {bidirectional}")
        logger.info(f"   Use attention: {use_attention}")  # Log attention usage
        logger.info(f"   Num classes: {num_classes}")
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, 1152) features precomputadas
            lengths: (B,) longitudes reales de cada secuencia (opcional)
        
        Returns:
            logits: (B, num_classes)
        """
        B, T, D = x.shape
        
        # LSTM forward
        if lengths is not None:
            # Pack padded sequence para eficiencia
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)  # (B, T, hidden_dim*2)
        
        if self.use_attention:
            # Use attention mechanism to get context vector
            h_last = self.attention(lstm_out, lengths)  # (B, hidden_dim*2)
        else:
            # Temporal pooling: tomar el último estado oculto
            if self.bidirectional:
                # Concatenar forward y backward del último layer
                h_forward = h_n[-2]  # Forward del último layer
                h_backward = h_n[-1]  # Backward del último layer
                h_last = torch.cat([h_forward, h_backward], dim=1)  # (B, hidden_dim*2)
            else:
                h_last = h_n[-1]  # (B, hidden_dim)
        
        # Clasificación
        logits = self.classifier(h_last)  # (B, num_classes)
        
        return logits


# class TemporalTransformerClassifier(nn.Module):
#     """
#     Clasificador temporal con Transformer
#     Input: (B, T, 640) features precomputadas
#     Output: (B, num_classes) logits
#     """
    
#     def __init__(
#         self,
#         input_dim: int = 640,
#         d_model: int = 512,
#         nhead: int = 8,
#         num_layers: int = 4,
#         dim_feedforward: int = 2048,
#         num_classes: int = 2286,
#         dropout: float = 0.3
#     ):
#         super().__init__()
        
#         self.input_dim = input_dim
#         self.d_model = d_model
        
#         # Projection de input a d_model
#         self.input_projection = nn.Linear(input_dim, d_model)
        
#         # Positional encoding
#         self.pos_encoder = PositionalEncoding(d_model, dropout)
        
#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         # Clasificador
#         self.classifier = nn.Sequential(
#             nn.Linear(d_model, 512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, num_classes)
#         )
        
#         logger.info(f"✓ TemporalTransformerClassifier inicializado")
#         logger.info(f"   Input dim: {input_dim}")
#         logger.info(f"   d_model: {d_model}")
#         logger.info(f"   Num layers: {num_layers}")
#         logger.info(f"   Num heads: {nhead}")
#         logger.info(f"   Num classes: {num_classes}")
    
#     def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
#         """
#         Args:
#             x: (B, T, 640) features precomputadas
#             lengths: (B,) longitudes reales de cada secuencia (opcional)
        
#         Returns:
#             logits: (B, num_classes)
#         """
#         # Proyectar input
#         x = self.input_projection(x)  # (B, T, d_model)
        
#         # Positional encoding
#         x = self.pos_encoder(x)
        
#         # Crear mask si tenemos longitudes
#         if lengths is not None:
#             B, T = x.shape[0], x.shape[1]
#             mask = torch.arange(T, device=x.device)[None, :] >= lengths[:, None]
#         else:
#             mask = None
        
#         # Transformer
#         x = self.transformer(x, src_key_padding_mask=mask)  # (B, T, d_model)
        
#         # Global average pooling
#         if lengths is not None:
#             # Maskear padding antes de pooling
#             mask_expanded = mask.unsqueeze(-1).expand_as(x)
#             x_masked = x.masked_fill(mask_expanded, 0)
#             x_pooled = x_masked.sum(dim=1) / lengths.unsqueeze(-1).float()
#         else:
#             x_pooled = x.mean(dim=1)  # (B, d_model)
        
#         # Clasificación
#         logits = self.classifier(x_pooled)  # (B, num_classes)
        
#         return logits


# class PositionalEncoding(nn.Module):
#     """Positional encoding para Transformer"""
    
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
        
#         # Crear positional encoding
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)  # (1, max_len, d_model)
#         self.register_buffer('pe', pe)
    
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x + self.pe[:, :x.size(1), :]
#         return self.dropout(x)


def get_temporal_model(model_type: str = "lstm", num_classes: int = 2286, **kwargs):
    """
    Factory function para crear modelos temporales
    
    Args:
        model_type: 'lstm' o 'transformer'
        num_classes: Número de clases
        **kwargs: Argumentos adicionales para el modelo
    
    Returns:
        Modelo temporal
    """
    if model_type == "lstm":
        return TemporalLSTMClassifier(num_classes=num_classes, **kwargs)
    elif model_type == "transformer":
        return TemporalTransformerClassifier(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Modelo desconocido: {model_type}")
