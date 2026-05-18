"""
Attention-Based Pooling Mechanisms

Implementações de pooling avançado para substituir mean/max pooling simples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Attention-based pooling que aprende a importância de cada token.

    Usa uma rede de atenção para computar pesos adaptativos sobre
    a sequência, produzindo um resumo ponderado.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, hidden_states, mask=None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len] (1 para tokens válidos, 0 para padding)

        Returns:
            pooled: [batch, hidden_dim]
            attn_weights: [batch, seq_len] (para interpretabilidade)
        """
        # Computar scores de atenção
        attn_scores = self.attention(hidden_states)  # [batch, seq_len, 1]
        attn_scores = attn_scores.squeeze(-1)  # [batch, seq_len]

        # Aplicar mask se fornecido
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax para obter pesos
        attn_weights = F.softmax(attn_scores, dim=1)  # [batch, seq_len]

        # Soma ponderada
        pooled = torch.bmm(
            attn_weights.unsqueeze(1),  # [batch, 1, seq_len]
            hidden_states  # [batch, seq_len, hidden_dim]
        ).squeeze(1)  # [batch, hidden_dim]

        return pooled, attn_weights


class MeanMaxPooling(nn.Module):
    """
    Combina mean pooling e max pooling.

    Mean captura informação global/distribuída.
    Max captura features mais salientes.
    """

    def forward(self, hidden_states, mask=None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len]

        Returns:
            pooled: [batch, hidden_dim * 2]
        """
        # Mean pooling
        if mask is not None:
            sum_hidden = (hidden_states * mask.unsqueeze(-1)).sum(dim=1)
            mean_pooled = sum_hidden / mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        else:
            mean_pooled = hidden_states.mean(dim=1)

        # Max pooling
        if mask is not None:
            masked_hidden = hidden_states.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            max_pooled = masked_hidden.max(dim=1)[0]
        else:
            max_pooled = hidden_states.max(dim=1)[0]

        # Concatenar
        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)

        return pooled


class MultiHeadAttentionPooling(nn.Module):
    """
    Pooling com múltiplas cabeças de atenção.

    Permite capturar diferentes aspectos do texto em paralelo.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        assert hidden_dim % num_heads == 0, "hidden_dim deve ser divisível por num_heads"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Queries aprendíveis (uma por head)
        self.queries = nn.Parameter(torch.randn(num_heads, self.head_dim))

        # Projeções
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, mask=None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len]

        Returns:
            pooled: [batch, hidden_dim]
            attn_weights: [batch, num_heads, seq_len]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Projetar keys e values
        keys = self.key_proj(hidden_states)  # [batch, seq_len, hidden_dim]
        values = self.value_proj(hidden_states)  # [batch, seq_len, hidden_dim]

        # Reshape para multi-head
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]

        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, head_dim]

        # Queries (broadcast to batch)
        queries = self.queries.unsqueeze(0).unsqueeze(2)  # [1, num_heads, 1, head_dim]
        queries = queries.expand(batch_size, -1, -1, -1)  # [batch, num_heads, 1, head_dim]

        # Atenção: Q * K^T
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))  # [batch, num_heads, 1, seq_len]
        attn_scores = attn_scores / (self.head_dim ** 0.5)  # Scaling
        attn_scores = attn_scores.squeeze(2)  # [batch, num_heads, seq_len]

        # Aplicar mask
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch, 1, seq_len]
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, num_heads, seq_len]
        attn_weights = self.dropout(attn_weights)

        # Aggregate values
        attn_weights_expanded = attn_weights.unsqueeze(-1)  # [batch, num_heads, seq_len, 1]
        pooled = (values * attn_weights_expanded).sum(dim=2)  # [batch, num_heads, head_dim]

        # Concatenar heads
        pooled = pooled.reshape(batch_size, hidden_dim)  # [batch, hidden_dim]

        return pooled, attn_weights
