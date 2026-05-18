"""
Improved Classifier

Classificador aprimorado com:
- Redução gradual de dimensionalidade
- Batch Normalization
- Dropout adaptativo
- Residual connection opcional
"""

import torch
import torch.nn as nn


class ImprovedClassifier(nn.Module):
    """
    Classificador aprimorado para substituir MLP simples.

    Características:
    - Redução gradual: 256 → 192 → 128 → 64 → num_classes
    - Batch Normalization para estabilidade
    - Dropout crescente em camadas profundas
    - Residual connection opcional

    Args:
        hidden_dim: Dimensão de entrada
        num_classes: Número de classes (2 para binário)
        dropout: Taxa de dropout base
        use_batch_norm: Se True, usa Batch Normalization
        use_residual: Se True, adiciona skip connection
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
        use_residual: bool = True
    ):
        super().__init__()

        self.use_residual = use_residual

        # Camadas com redução gradual
        # Layer 1: 256 → 192
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 3 // 4)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 3 // 4) if use_batch_norm else nn.Identity()

        # Layer 2: 192 → 128
        self.fc2 = nn.Linear(hidden_dim * 3 // 4, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2) if use_batch_norm else nn.Identity()

        # Layer 3: 128 → 64
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4) if use_batch_norm else nn.Identity()

        # Layer 4: 64 → num_classes
        self.fc4 = nn.Linear(hidden_dim // 4, num_classes)

        # Activations
        self.activation = nn.GELU()

        # Dropout crescente
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 1.2)  # Um pouco maior
        self.dropout3 = nn.Dropout(dropout * 1.5)  # Ainda maior

        # Residual projection (para conectar input a layer 2)
        if use_residual:
            self.residual_proj = nn.Linear(hidden_dim, hidden_dim // 2)

    def forward(self, x):
        """
        Args:
            x: [batch, hidden_dim]

        Returns:
            logits: [batch, num_classes]
        """
        # Salvar input para residual
        identity = x

        # Layer 1: 256 → 192
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout1(out)

        # Layer 2: 192 → 128 (com residual opcional)
        out = self.fc2(out)
        out = self.bn2(out)

        # Adicionar residual se habilitado
        if self.use_residual:
            residual = self.residual_proj(identity)
            out = out + residual

        out = self.activation(out)
        out = self.dropout2(out)

        # Layer 3: 128 → 64
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.activation(out)
        out = self.dropout3(out)

        # Layer 4: 64 → num_classes (sem activation, será aplicado na loss)
        logits = self.fc4(out)

        return logits


class SimpleClassifier(nn.Module):
    """
    Classificador simples (baseline) para comparação.
    """

    def __init__(self, hidden_dim: int = 256, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits
