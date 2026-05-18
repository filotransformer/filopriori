"""
Cross-Attention Fusion Module
Implements cross-attention mechanism to fuse semantic and structural streams
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer"""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.head_dim * num_heads == hidden_dim, \
            "hidden_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            query: [batch_size, seq_len_q, hidden_dim]
            key: [batch_size, seq_len_k, hidden_dim]
            value: [batch_size, seq_len_v, hidden_dim]
            mask: Optional attention mask

        Returns:
            Output tensor [batch_size, seq_len_q, hidden_dim]
        """
        batch_size = query.size(0)

        # Project Q, K, V
        Q = self.q_proj(query)  # [batch, seq_q, hidden]
        K = self.k_proj(key)    # [batch, seq_k, hidden]
        V = self.v_proj(value)  # [batch, seq_v, hidden]

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: [batch, num_heads, seq_len, head_dim]

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # [batch, num_heads, seq_q, seq_k]

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # [batch, num_heads, seq_q, head_dim]

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.hidden_dim)

        # Output projection
        output = self.out_proj(attn_output)
        output = self.dropout(output)

        # Residual connection and layer norm
        output = self.layer_norm(query + output)

        return output


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Module
    Fuses semantic stream and structural stream using bidirectional cross-attention
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Semantic -> Structural cross-attention layers
        self.semantic_to_structural = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Structural -> Semantic cross-attention layers
        self.structural_to_semantic = nn.ModuleList([
            CrossAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Feed-forward networks
        self.ffn_semantic = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(num_layers)
        ])

        self.ffn_structural = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        semantic_features: torch.Tensor,
        structural_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with bidirectional cross-attention

        Args:
            semantic_features: [batch_size, hidden_dim] - from semantic stream
            structural_features: [batch_size, hidden_dim] - from structural stream

        Returns:
            Fused features [batch_size, hidden_dim]
        """
        # Add sequence dimension for attention mechanism
        semantic = semantic_features.unsqueeze(1)  # [batch, 1, hidden]
        structural = structural_features.unsqueeze(1)  # [batch, 1, hidden]

        # Apply cross-attention layers
        for i in range(self.num_layers):
            # Semantic queries structural
            semantic_attended = self.semantic_to_structural[i](
                query=semantic,
                key=structural,
                value=structural
            )
            semantic = semantic + self.ffn_semantic[i](semantic_attended)

            # Structural queries semantic
            structural_attended = self.structural_to_semantic[i](
                query=structural,
                key=semantic,
                value=semantic
            )
            structural = structural + self.ffn_structural[i](structural_attended)

        # Remove sequence dimension
        semantic = semantic.squeeze(1)
        structural = structural.squeeze(1)

        # Combine both streams (concatenate and project, or simply add/mean)
        # Here we use concatenation
        fused = torch.cat([semantic, structural], dim=-1)  # [batch, 2*hidden]

        return fused
