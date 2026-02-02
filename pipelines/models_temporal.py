"""
Modelo temporal REGULARIZADO para ASL
Incluye: Variational Dropout, Layer Norm, Residual Connections, EMA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
from typing import Optional, Tuple
import copy

logger = logging.getLogger(__name__)


class VariationalDropout(nn.Module):
    """
    Variational Dropout: usa la MISMA mascara de dropout para todos los timesteps.
    Esto es mas efectivo para RNNs que el dropout estandar.
    
    Referencia: Gal & Ghahramani, "A Theoretically Grounded Application of Dropout in RNNs"
    """
    
    def __init__(self, dropout_rate: float = 0.3):
        super().__init__()
        self.dropout_rate = dropout_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) tensor
        Returns:
            x con dropout aplicado (misma mascara para todos los T)
        """
        if not self.training or self.dropout_rate == 0:
            return x
        
        B, T, D = x.shape
        # Crear mascara una sola vez para toda la secuencia
        mask = torch.bernoulli(
            torch.full((B, 1, D), 1 - self.dropout_rate, device=x.device)
        )
        mask = mask / (1 - self.dropout_rate)  # Scale para mantener magnitud
        
        return x * mask


class FeatureDropout(nn.Module):
    """
    Dropout a nivel de feature completa (dropea dimensiones enteras)
    Util para regularizar features precomputadas
    """
    
    def __init__(self, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout_rate == 0:
            return x
        
        if x.dim() == 3:  # (B, T, D)
            B, T, D = x.shape
            mask = torch.bernoulli(
                torch.full((B, 1, D), 1 - self.dropout_rate, device=x.device)
            )
        else:  # (B, D)
            B, D = x.shape
            mask = torch.bernoulli(
                torch.full((B, D), 1 - self.dropout_rate, device=x.device)
            )
        
        mask = mask / (1 - self.dropout_rate)
        return x * mask


class AttentionModuleRegularized(nn.Module):
    """
    Attention mechanism con regularizacion adicional
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        lstm_output: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            context: (B, hidden_dim)
            attention_weights: (B, T, 1) para visualizacion
        """
        attention_scores = self.attention(lstm_output)
        
        if lengths is not None:
            B, T = lstm_output.shape[0], lstm_output.shape[1]
            mask = torch.arange(T, device=lstm_output.device)[None, :] >= lengths[:, None]
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(-1), float('-inf'))
        
        attention_weights = torch.softmax(attention_scores, dim=1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.sum(attention_weights * lstm_output, dim=1)
        
        return context, attention_weights


class TemporalLSTMRegularized(nn.Module):
    """
    LSTM Clasificador con REGULARIZACION COMPLETA:
    - Variational Dropout (misma mascara por timestep)
    - Layer Normalization
    - Residual connections donde es posible
    - Feature dropout en entrada
    - Dropout en clasificador
    """
    
    def __init__(
        self,
        input_dim: int = 1152,
        hidden_dim: int = 384,
        num_layers: int = 2,
        num_classes: int = 2286,
        dropout: float = 0.4,
        bidirectional: bool = True,
        use_attention: bool = True,
        use_variational_dropout: bool = True,
        use_layer_norm: bool = True,
        classifier_dropout: float = 0.4,
        input_feature_dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_variational_dropout = use_variational_dropout
        self.use_layer_norm = use_layer_norm
        
        # 1. Feature dropout en entrada
        self.input_dropout = FeatureDropout(input_feature_dropout)
        
        # 2. Proyeccion de entrada con normalizacion
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # 3. LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # 4. Variational Dropout despues de LSTM
        if use_variational_dropout:
            self.var_dropout = VariationalDropout(dropout)
        else:
            self.var_dropout = nn.Dropout(dropout)
        
        # 5. Layer Norm despues de LSTM
        if use_layer_norm:
            self.lstm_norm = nn.LayerNorm(lstm_output_dim)
        else:
            self.lstm_norm = nn.Identity()
        
        # 6. Attention
        if use_attention:
            self.attention = AttentionModuleRegularized(lstm_output_dim, dropout=dropout)
        
        # 7. Clasificador con regularizacion
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.LayerNorm(lstm_output_dim // 2) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )
        
        # Inicializacion de pesos
        self._init_weights()
        
        logger.info(f"TemporalLSTMRegularized inicializado:")
        logger.info(f"  Input dim: {input_dim} -> {hidden_dim}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  LSTM output dim: {lstm_output_dim}")
        logger.info(f"  Layers: {num_layers}, Bidirectional: {bidirectional}")
        logger.info(f"  Dropout: {dropout}, Classifier dropout: {classifier_dropout}")
        logger.info(f"  Variational dropout: {use_variational_dropout}")
        logger.info(f"  Layer norm: {use_layer_norm}")
        logger.info(f"  Attention: {use_attention}")
        logger.info(f"  Num classes: {num_classes}")
    
    def _init_weights(self):
        """Inicializacion Xavier/Kaiming para mejor convergencia"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # Orthogonal init para LSTM (mejor para gradientes)
                    if param.dim() >= 2:
                        nn.init.orthogonal_(param)
                elif param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, 1152) features precomputadas
            lengths: (B,) longitudes reales
            return_attention: Si True, retorna attention weights
        
        Returns:
            logits: (B, num_classes)
            attention_weights: (B, T, 1) si return_attention=True
        """
        B, T, D = x.shape
        
        # 1. Feature dropout en entrada
        x = self.input_dropout(x)
        
        # 2. Proyeccion de entrada
        x = self.input_proj(x)  # (B, T, hidden_dim)
        
        # 3. LSTM
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 4. Variational dropout + Layer norm
        lstm_out = self.var_dropout(lstm_out)
        lstm_out = self.lstm_norm(lstm_out)
        
        # 5. Pooling (attention o ultimo estado)
        attention_weights = None
        if self.use_attention:
            h_pooled, attention_weights = self.attention(lstm_out, lengths)
        else:
            if self.bidirectional:
                h_forward = h_n[-2]
                h_backward = h_n[-1]
                h_pooled = torch.cat([h_forward, h_backward], dim=1)
            else:
                h_pooled = h_n[-1]
        
        # 6. Clasificacion
        logits = self.classifier(h_pooled)
        
        if return_attention:
            return logits, attention_weights
        return logits


class ExponentialMovingAverage:
    """
    EMA de los pesos del modelo para mejor generalizacion.
    Mantiene un promedio exponencial de los pesos durante el entrenamiento.
    En inferencia, usa los pesos promediados.
    
    Referencia: Polyak averaging
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Inicializar shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Actualizar EMA despues de cada paso de optimizacion"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1.0 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Aplicar pesos EMA al modelo (para evaluacion)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restaurar pesos originales (despues de evaluacion)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return {
            'shadow': self.shadow,
            'decay': self.decay
        }
    
    def load_state_dict(self, state_dict):
        self.shadow = state_dict['shadow']
        self.decay = state_dict['decay']


class MixupCutmix:
    """
    Mixup para secuencias temporales.
    Mezcla dos secuencias y sus labels.
    """
    
    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Args:
            x: (B, T, D) features
            y: (B,) labels
            lengths: (B,) sequence lengths
        
        Returns:
            mixed_x, y_a, y_b, lam
        """
        if torch.rand(1).item() > self.prob:
            return x, y, y, 1.0
        
        B = x.size(0)
        
        # Sample lambda from Beta distribution
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()
        
        # Random permutation for mixing
        index = torch.randperm(B, device=x.device)
        
        # Mix features
        mixed_x = lam * x + (1 - lam) * x[index]
        
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module, 
    pred: torch.Tensor, 
    y_a: torch.Tensor, 
    y_b: torch.Tensor, 
    lam: float
) -> torch.Tensor:
    """Loss function para mixup"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class RDropLoss(nn.Module):
    """
    R-Drop: Regularized Dropout for Neural Networks.
    Fuerza consistencia entre dos forward passes con dropout diferente.
    
    Referencia: Liang et al., "R-Drop: Regularized Dropout for Neural Networks"
    """
    
    def __init__(self, alpha: float = 0.5, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction=reduction)
        self.kl = nn.KLDivLoss(reduction='batchmean')
    
    def forward(
        self, 
        logits1: torch.Tensor, 
        logits2: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits1, logits2: Dos forward passes del mismo input
            target: Ground truth labels
        """
        # CE loss para ambos
        ce_loss = 0.5 * (self.ce(logits1, target) + self.ce(logits2, target))
        
        # KL divergence bidireccional
        p = F.log_softmax(logits1, dim=-1)
        q = F.log_softmax(logits2, dim=-1)
        
        kl_loss = 0.5 * (
            self.kl(p, F.softmax(logits2, dim=-1)) +
            self.kl(q, F.softmax(logits1, dim=-1))
        )
        
        return ce_loss + self.alpha * kl_loss


def get_temporal_model(
    model_type: str = "lstm", 
    num_classes: int = 2286,
    regularized: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function para crear modelos temporales.
    
    Args:
        model_type: 'lstm' o 'transformer'
        num_classes: Numero de clases
        regularized: Si usar version regularizada
        **kwargs: Argumentos adicionales
    """
    if model_type == "lstm":
        if regularized:
            return TemporalLSTMRegularized(num_classes=num_classes, **kwargs)
        else:
            # Version original sin regularizacion extra
            from models_temporal import TemporalLSTMClassifier
            return TemporalLSTMClassifier(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Modelo desconocido: {model_type}")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Cuenta parametros totales y entrenables"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
