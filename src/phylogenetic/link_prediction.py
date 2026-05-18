"""
Link Prediction for Graph Rewiring

Implements self-supervised link prediction to learn better graph structure.
The learned edge scores are used to rewire the k-NN graph, replacing
semantic similarity with task-specific structural relevance.

References:
- "Link Prediction Based on Graph Neural Networks" (Zhang & Chen, 2018)
- "Neural Link Prediction with Walk Pooling" (Wang et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, add_self_loops
from typing import Tuple, Optional, Dict
import numpy as np


class LinkPredictionEncoder(nn.Module):
    """
    GNN encoder for link prediction.

    Learns node embeddings that capture structural patterns
    useful for predicting edge existence.

    Args:
        input_dim: Dimension of input node features
        hidden_dim: Dimension of hidden layers
        output_dim: Dimension of output embeddings
        num_layers: Number of GNN layers
        dropout: Dropout rate
        encoder_type: Type of GNN encoder ('gcn', 'gat', 'sage')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        encoder_type: str = 'gcn'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.encoder_type = encoder_type

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GNN layers
        if encoder_type == 'gcn':
            # Simple GCN-style message passing
            self.conv_layers = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ])
        elif encoder_type == 'gat':
            # Use GATv2 if available
            try:
                from ..layers.gatv2 import GATv2Conv
                self.conv_layers = nn.ModuleList([
                    GATv2Conv(
                        hidden_dim,
                        hidden_dim // 4,
                        num_heads=4,
                        concat=True,
                        dropout=dropout
                    )
                    for _ in range(num_layers)
                ])
            except ImportError:
                raise ImportError("GATv2 not available, use encoder_type='gcn'")
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # Normalization and activation
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Encode nodes into embeddings.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        x = self.input_proj(x)

        # GNN layers
        for i in range(self.num_layers):
            if self.encoder_type == 'gcn':
                # Simple message passing
                row, col = edge_index[0], edge_index[1]
                num_nodes = x.size(0)

                # Aggregate messages
                messages = x[col]  # [num_edges, hidden_dim]
                aggregated = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
                aggregated.index_add_(0, row, messages)

                # Normalize by degree
                degree = torch.zeros(num_nodes, device=x.device)
                degree.index_add_(0, row, torch.ones_like(row, dtype=torch.float))
                degree = torch.clamp(degree, min=1.0).unsqueeze(-1)
                aggregated = aggregated / degree

                # Transform
                x_new = self.conv_layers[i](aggregated)

            elif self.encoder_type == 'gat':
                x_new = self.conv_layers[i](x, edge_index)

            # Residual connection + normalization
            x = x + x_new
            x = self.norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)

        # Output projection
        x = self.output_proj(x)

        return x


class LinkPredictionDecoder(nn.Module):
    """
    Decoder for link prediction.

    Takes node embeddings and predicts edge existence scores.

    Args:
        embedding_dim: Dimension of node embeddings
        decoder_type: Type of decoder
            - 'dot': Simple dot product
            - 'mlp': MLP-based decoder
            - 'distmult': DistMult scoring (for heterogeneous graphs)
    """

    def __init__(
        self,
        embedding_dim: int,
        decoder_type: str = 'mlp',
        hidden_dim: Optional[int] = None
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.decoder_type = decoder_type

        if decoder_type == 'dot':
            # Simple dot product decoder (no parameters)
            pass

        elif decoder_type == 'mlp':
            # MLP decoder
            hidden_dim = hidden_dim or embedding_dim
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            )

        elif decoder_type == 'distmult':
            # DistMult-style decoder (element-wise multiplication)
            self.relation_weights = nn.Parameter(torch.ones(embedding_dim))

        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict edge scores.

        Args:
            z: Node embeddings [num_nodes, embedding_dim]
            edge_index: Edge indices to score [2, num_edges]

        Returns:
            Edge scores [num_edges]
        """
        row, col = edge_index[0], edge_index[1]
        z_i = z[row]  # [num_edges, embedding_dim]
        z_j = z[col]  # [num_edges, embedding_dim]

        if self.decoder_type == 'dot':
            # Dot product + sigmoid
            scores = (z_i * z_j).sum(dim=-1)

        elif self.decoder_type == 'mlp':
            # MLP decoder
            edge_feat = torch.cat([z_i, z_j], dim=-1)
            scores = self.mlp(edge_feat).squeeze(-1)

        elif self.decoder_type == 'distmult':
            # DistMult scoring
            scores = (z_i * self.relation_weights * z_j).sum(dim=-1)

        return scores

    def score_all_edges(
        self,
        z: torch.Tensor,
        batch_size: int = 10000
    ) -> torch.Tensor:
        """
        Score all possible edges (for graph rewiring).

        Args:
            z: Node embeddings [num_nodes, embedding_dim]
            batch_size: Batch size for processing (to avoid OOM)

        Returns:
            Edge scores [num_nodes, num_nodes]
        """
        num_nodes = z.size(0)

        if self.decoder_type == 'dot':
            # Efficient matrix multiplication
            scores = torch.mm(z, z.t())  # [num_nodes, num_nodes]

        elif self.decoder_type == 'mlp':
            # Batch processing to avoid OOM
            scores = torch.zeros(num_nodes, num_nodes, device=z.device)

            for i in range(0, num_nodes, batch_size):
                end_i = min(i + batch_size, num_nodes)
                z_i = z[i:end_i].unsqueeze(1)  # [batch, 1, dim]
                z_j = z.unsqueeze(0)  # [1, num_nodes, dim]

                # Broadcast and concatenate
                z_i_expanded = z_i.expand(-1, num_nodes, -1)  # [batch, num_nodes, dim]
                z_j_expanded = z_j.expand(end_i - i, -1, -1)  # [batch, num_nodes, dim]

                edge_feat = torch.cat([z_i_expanded, z_j_expanded], dim=-1)
                edge_feat = edge_feat.reshape(-1, self.embedding_dim * 2)

                batch_scores = self.mlp(edge_feat).squeeze(-1)
                batch_scores = batch_scores.reshape(end_i - i, num_nodes)

                scores[i:end_i] = batch_scores

        elif self.decoder_type == 'distmult':
            # DistMult with batching
            scores = torch.zeros(num_nodes, num_nodes, device=z.device)

            z_weighted = z * self.relation_weights

            for i in range(0, num_nodes, batch_size):
                end_i = min(i + batch_size, num_nodes)
                batch_scores = torch.mm(z_weighted[i:end_i], z.t())
                scores[i:end_i] = batch_scores

        return scores


class LinkPredictor(nn.Module):
    """
    Complete link prediction model.

    Combines encoder and decoder for end-to-end link prediction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        encoder_type: str = 'gcn',
        decoder_type: str = 'mlp'
    ):
        super().__init__()

        self.encoder = LinkPredictionEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=num_layers,
            dropout=dropout,
            encoder_type=encoder_type
        )

        self.decoder = LinkPredictionDecoder(
            embedding_dim=embedding_dim,
            decoder_type=decoder_type,
            hidden_dim=hidden_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_label_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for link prediction.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Training edges [2, num_edges]
            edge_label_index: Edges to predict [2, num_edges_to_predict]

        Returns:
            Edge scores [num_edges_to_predict]
        """
        # Encode
        z = self.encoder(x, edge_index)

        # Decode
        scores = self.decoder(z, edge_label_index)

        return scores

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get node embeddings."""
        return self.encoder(x, edge_index)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Predict edge scores from embeddings."""
        return self.decoder(z, edge_index)

    def score_all_edges(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Score all possible edges."""
        z = self.encode(x, edge_index)
        return self.decoder.score_all_edges(z)


def sample_negative_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_neg_samples: Optional[int] = None,
    method: str = 'uniform'
) -> torch.Tensor:
    """
    Sample negative edges for link prediction training.

    Args:
        edge_index: Positive edge indices [2, num_edges]
        num_nodes: Number of nodes in graph
        num_neg_samples: Number of negative samples (default: same as num_edges)
        method: Sampling method ('uniform' or 'hard')

    Returns:
        Negative edge indices [2, num_neg_samples]
    """
    if num_neg_samples is None:
        num_neg_samples = edge_index.size(1)

    if method == 'uniform':
        # Use PyG's built-in negative sampling
        neg_edge_index = negative_sampling(
            edge_index=edge_index,
            num_nodes=num_nodes,
            num_neg_samples=num_neg_samples
        )

    elif method == 'hard':
        # Hard negative sampling: sample nodes close to positive edges
        # This makes the task harder and more realistic
        raise NotImplementedError("Hard negative sampling not yet implemented")

    return neg_edge_index


def compute_link_prediction_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    loss_type: str = 'bce'
) -> torch.Tensor:
    """
    Compute link prediction loss.

    Args:
        pos_scores: Scores for positive edges [num_pos]
        neg_scores: Scores for negative edges [num_neg]
        loss_type: Type of loss ('bce', 'margin', 'auc')

    Returns:
        Loss value (scalar)
    """
    if loss_type == 'bce':
        # Binary cross-entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores,
            torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores,
            torch.zeros_like(neg_scores)
        )
        loss = pos_loss + neg_loss

    elif loss_type == 'margin':
        # Margin ranking loss
        target = torch.ones(pos_scores.size(0), device=pos_scores.device)
        loss = F.margin_ranking_loss(
            pos_scores,
            neg_scores,
            target,
            margin=1.0
        )

    elif loss_type == 'auc':
        # AUC loss (approximation)
        # Maximize difference between positive and negative scores
        loss = -torch.mean(torch.sigmoid(pos_scores - neg_scores))

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    return loss
