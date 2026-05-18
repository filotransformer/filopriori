"""
Denoising Gate Layer for Graph Neural Networks

Implements a learnable gating mechanism to filter noisy neighbors in k-NN graphs.
The gate learns to score edge relevance based on source and target node features,
effectively denoising the graph structure during training.

References:
- "Adaptive Universal Generalized PageRank Graph Neural Network" (Abu-El-Haija et al., 2019)
- "Graph Structure Learning for Robust Graph Neural Networks" (Jin et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DenoisingGate(nn.Module):
    """
    Learnable gating mechanism to filter noisy edges in k-NN graphs.

    The gate computes a relevance score for each edge based on:
    - Source node features
    - Target node features
    - Optional: Edge features (e.g., cosine similarity)

    Args:
        hidden_dim: Dimension of node features
        gate_type: Type of gating mechanism
            - 'mlp': Multi-layer perceptron (default)
            - 'attention': Attention-based scoring
            - 'bilinear': Bilinear scoring
        dropout: Dropout rate for gating network
        use_edge_features: Whether to incorporate pre-computed edge weights
        hard_threshold: If set, applies hard thresholding (drops edges < threshold)
        temperature: Temperature for soft gating (lower = harder gating)
    """

    def __init__(
        self,
        hidden_dim: int,
        gate_type: str = 'mlp',
        dropout: float = 0.1,
        use_edge_features: bool = True,
        hard_threshold: Optional[float] = None,
        temperature: float = 1.0
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.gate_type = gate_type
        self.use_edge_features = use_edge_features
        self.hard_threshold = hard_threshold
        self.temperature = temperature

        # Build gating network based on type
        if gate_type == 'mlp':
            # MLP-based gating: concat(h_i, h_j, [edge_feat]) -> score
            input_dim = hidden_dim * 2
            if use_edge_features:
                input_dim += 1  # Add edge weight feature

            self.gate_network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()  # Output score in [0, 1]
            )

        elif gate_type == 'attention':
            # Attention-based gating (similar to GAT but simpler)
            self.query_proj = nn.Linear(hidden_dim, hidden_dim)
            self.key_proj = nn.Linear(hidden_dim, hidden_dim)

            if use_edge_features:
                self.edge_proj = nn.Linear(1, hidden_dim)
            else:
                self.edge_proj = None

            self.attn_weight = nn.Parameter(torch.Tensor(hidden_dim, 1))
            self.dropout = nn.Dropout(dropout)
            nn.init.xavier_uniform_(self.attn_weight)

        elif gate_type == 'bilinear':
            # Bilinear scoring: h_i^T W h_j
            self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)

            if use_edge_features:
                self.edge_weight = nn.Parameter(torch.Tensor(1))
                nn.init.constant_(self.edge_weight, 1.0)
            else:
                self.edge_weight = None

            self.dropout = nn.Dropout(dropout)

        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute edge gating scores.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_weights: Optional pre-computed edge weights [num_edges]

        Returns:
            gate_scores: Edge gating scores [num_edges] in [0, 1]
            filtered_edge_index: Edge index after filtering (if hard_threshold is set)
        """
        row, col = edge_index[0], edge_index[1]

        # Get source and target node features
        x_i = x[row]  # [num_edges, hidden_dim]
        x_j = x[col]  # [num_edges, hidden_dim]

        if self.gate_type == 'mlp':
            # Concatenate features
            edge_feat = torch.cat([x_i, x_j], dim=-1)

            if self.use_edge_features and edge_weights is not None:
                edge_feat = torch.cat([edge_feat, edge_weights.unsqueeze(-1)], dim=-1)

            # Compute gate scores
            gate_scores = self.gate_network(edge_feat).squeeze(-1)  # [num_edges]

        elif self.gate_type == 'attention':
            # Project to query and key
            q = self.query_proj(x_i)  # [num_edges, hidden_dim]
            k = self.key_proj(x_j)  # [num_edges, hidden_dim]

            # Add edge features if available
            if self.use_edge_features and edge_weights is not None and self.edge_proj is not None:
                edge_emb = self.edge_proj(edge_weights.unsqueeze(-1))  # [num_edges, hidden_dim]
                qk = q + k + edge_emb
            else:
                qk = q + k

            # Compute attention scores
            attn_logits = torch.matmul(F.leaky_relu(qk), self.attn_weight).squeeze(-1)
            gate_scores = torch.sigmoid(attn_logits / self.temperature)
            gate_scores = self.dropout(gate_scores)

        elif self.gate_type == 'bilinear':
            # Bilinear scoring
            gate_logits = self.bilinear(x_i, x_j).squeeze(-1)

            # Add edge features if available
            if self.use_edge_features and edge_weights is not None and self.edge_weight is not None:
                gate_logits = gate_logits + self.edge_weight * edge_weights

            gate_scores = torch.sigmoid(gate_logits / self.temperature)
            gate_scores = self.dropout(gate_scores)

        # Apply hard threshold if specified
        filtered_edge_index = edge_index
        if self.hard_threshold is not None:
            mask = gate_scores >= self.hard_threshold
            filtered_edge_index = edge_index[:, mask]
            gate_scores = gate_scores[mask]

        return gate_scores, filtered_edge_index

    def get_edge_retention_rate(self, gate_scores: torch.Tensor) -> float:
        """
        Calculate the percentage of edges retained after gating.
        Useful for monitoring during training.
        """
        if self.hard_threshold is not None:
            return (gate_scores >= self.hard_threshold).float().mean().item()
        else:
            # For soft gating, count edges with score > 0.5
            return (gate_scores > 0.5).float().mean().item()


class AdaptiveDenoisingGate(nn.Module):
    """
    Adaptive denoising gate that adjusts its threshold during training.

    Starts with a low threshold (keeps most edges) and gradually increases
    it as the model learns, progressively filtering more aggressively.

    This prevents the model from removing too many edges early in training
    when it hasn't learned good representations yet.
    """

    def __init__(
        self,
        hidden_dim: int,
        gate_type: str = 'mlp',
        dropout: float = 0.1,
        use_edge_features: bool = True,
        initial_threshold: float = 0.1,
        final_threshold: float = 0.5,
        warmup_epochs: int = 10
    ):
        super().__init__()

        self.base_gate = DenoisingGate(
            hidden_dim=hidden_dim,
            gate_type=gate_type,
            dropout=dropout,
            use_edge_features=use_edge_features,
            hard_threshold=None,  # Will be set dynamically
            temperature=1.0
        )

        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Update the current epoch for threshold scheduling."""
        self.current_epoch = epoch

    def get_current_threshold(self) -> float:
        """Compute current threshold based on training progress."""
        if self.current_epoch >= self.warmup_epochs:
            return self.final_threshold

        # Linear interpolation from initial to final
        progress = self.current_epoch / self.warmup_epochs
        threshold = self.initial_threshold + progress * (self.final_threshold - self.initial_threshold)
        return threshold

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with adaptive thresholding."""
        # Get gate scores from base gate
        gate_scores, _ = self.base_gate(x, edge_index, edge_weights)

        # Apply current threshold
        current_threshold = self.get_current_threshold()
        mask = gate_scores >= current_threshold
        filtered_edge_index = edge_index[:, mask]
        filtered_scores = gate_scores[mask]

        return filtered_scores, filtered_edge_index


class DenoisingGateWithNeighborDropout(nn.Module):
    """
    Combines denoising gate with neighbor dropout for additional regularization.

    During training, randomly drops a fraction of neighbors before applying
    the learned gating mechanism. This forces the model to be robust and not
    over-rely on specific neighbors.
    """

    def __init__(
        self,
        hidden_dim: int,
        gate_type: str = 'mlp',
        dropout: float = 0.1,
        use_edge_features: bool = True,
        neighbor_dropout: float = 0.1,
        hard_threshold: Optional[float] = None,
        temperature: float = 1.0
    ):
        super().__init__()

        self.gate = DenoisingGate(
            hidden_dim=hidden_dim,
            gate_type=gate_type,
            dropout=dropout,
            use_edge_features=use_edge_features,
            hard_threshold=hard_threshold,
            temperature=temperature
        )

        self.neighbor_dropout = neighbor_dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with neighbor dropout + gating."""

        # Apply neighbor dropout during training
        if self.training and self.neighbor_dropout > 0:
            num_edges = edge_index.size(1)
            keep_mask = torch.rand(num_edges, device=edge_index.device) > self.neighbor_dropout
            edge_index_dropped = edge_index[:, keep_mask]

            if edge_weights is not None:
                edge_weights_dropped = edge_weights[keep_mask]
            else:
                edge_weights_dropped = None
        else:
            edge_index_dropped = edge_index
            edge_weights_dropped = edge_weights

        # Apply denoising gate
        gate_scores, filtered_edge_index = self.gate(x, edge_index_dropped, edge_weights_dropped)

        return gate_scores, filtered_edge_index
