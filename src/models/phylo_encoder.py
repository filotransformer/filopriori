"""
Phylo-Encoder: Gated Graph Neural Network for Phylogenetic Test Case Prioritization

This module implements the PhyloEncoder, a bio-inspired neural network that treats
the Git DAG as a phylogenetic tree and propagates failure signals through evolutionary
history using Gated Graph Neural Networks (GGNN).

Key Concepts:
- Commits are treated as "taxa" in a phylogenetic tree
- The Git DAG represents evolutionary relationships
- Phylogenetic distance weights information propagation
- GGNN uses GRU-like gated updates for message passing

Scientific Foundation:
- Felsenstein (2004): Phylogenetic comparative methods
- Li et al. (2016): Gated Graph Sequence Neural Networks
- German et al. (2009): Change impact graphs

Author: Filo-Priori V9 Team
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging
import numpy as np

logger = logging.getLogger(__name__)


class PhylogeneticDistanceKernel(nn.Module):
    """
    Computes phylogenetic distances between commits in the Git DAG.

    The distance kernel captures evolutionary relationships by considering:
    1. Shortest path length in the DAG
    2. Number of merge commits (synchronization points)
    3. Optional: code churn as branch length

    Distance Formula:
        d_phylo(c_i, c_j) = shortest_path(c_i, c_j) × β^(n_merges)

    Where:
        - shortest_path: Number of edges between commits
        - n_merges: Number of merge commits on the path
        - β: Decay factor (default 0.9), merges "reset" divergence

    The phylogenetic weight for propagation is:
        w_phylo(c_i, c_j) = exp(-d_phylo(c_i, c_j) / τ)

    Where τ is a learnable temperature parameter.
    """

    def __init__(
        self,
        decay_factor: float = 0.9,
        learnable_temperature: bool = True,
        initial_temperature: float = 1.0
    ):
        """
        Initialize the Phylogenetic Distance Kernel.

        Args:
            decay_factor: β parameter for merge decay (default 0.9)
            learnable_temperature: If True, τ is a learnable parameter
            initial_temperature: Initial value for temperature τ
        """
        super().__init__()

        self.decay_factor = decay_factor

        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(initial_temperature))
        else:
            self.register_buffer('temperature', torch.tensor(initial_temperature))

        logger.info(f"Initialized PhylogeneticDistanceKernel:")
        logger.info(f"  - Decay factor (β): {decay_factor}")
        logger.info(f"  - Temperature (τ): {initial_temperature} (learnable={learnable_temperature})")

    def compute_distance(
        self,
        path_lengths: torch.Tensor,
        merge_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute phylogenetic distances.

        Args:
            path_lengths: Shortest path lengths [E] for each edge
            merge_counts: Number of merges on path [E] for each edge

        Returns:
            distances: Phylogenetic distances [E]
        """
        # d_phylo = path_length × β^(n_merges)
        distances = path_lengths * (self.decay_factor ** merge_counts)
        return distances

    def compute_weights(
        self,
        path_lengths: torch.Tensor,
        merge_counts: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute phylogenetic weights for message passing.

        Args:
            path_lengths: Shortest path lengths [E] for each edge
            merge_counts: Number of merges on path [E], defaults to zeros

        Returns:
            weights: Phylogenetic weights [E] in range (0, 1]
        """
        if merge_counts is None:
            merge_counts = torch.zeros_like(path_lengths)

        # Compute distances
        distances = self.compute_distance(path_lengths, merge_counts)

        # Compute weights: w = exp(-d / τ)
        weights = torch.exp(-distances / (self.temperature + 1e-8))

        return weights

    def forward(
        self,
        edge_index: torch.Tensor,
        path_lengths: torch.Tensor,
        merge_counts: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass: compute phylogenetic weights for edges.

        Args:
            edge_index: Graph edges [2, E]
            path_lengths: Path lengths for edges [E]
            merge_counts: Optional merge counts for edges [E]

        Returns:
            weights: Edge weights [E] based on phylogenetic distance
        """
        return self.compute_weights(path_lengths, merge_counts)


class GGNNLayer(nn.Module):
    """
    Single layer of a Gated Graph Neural Network (GGNN).

    GGNN uses GRU-like gated updates for message passing, which is
    particularly suitable for sequential/temporal data like Git history.

    Update equations:
        m_v = Σ_{u ∈ N(v)} w_{uv} · h_u    (weighted message aggregation)
        z_v = σ(W_z · [h_v || m_v] + b_z)   (update gate)
        r_v = σ(W_r · [h_v || m_v] + b_r)   (reset gate)
        h̃_v = tanh(W_h · [r_v ⊙ h_v || m_v] + b_h)  (candidate)
        h'_v = (1 - z_v) ⊙ h_v + z_v ⊙ h̃_v  (new hidden state)

    Reference: Li et al. (2016) "Gated Graph Sequence Neural Networks"
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        use_edge_weights: bool = True
    ):
        """
        Initialize GGNN layer.

        Args:
            hidden_dim: Hidden dimension
            dropout: Dropout probability
            use_edge_weights: Whether to use edge weights in aggregation
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_edge_weights = use_edge_weights

        # GRU-style gates
        # Update gate: z = σ(W_z · [h || m] + b_z)
        self.W_z = nn.Linear(hidden_dim * 2, hidden_dim)

        # Reset gate: r = σ(W_r · [h || m] + b_r)
        self.W_r = nn.Linear(hidden_dim * 2, hidden_dim)

        # Candidate hidden state: h̃ = tanh(W_h · [r ⊙ h || m] + b_h)
        self.W_h = nn.Linear(hidden_dim * 2, hidden_dim)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with GRU-style gated updates.

        Args:
            x: Node features [N, hidden_dim]
            edge_index: Graph edges [2, E]
            edge_weights: Optional edge weights [E]

        Returns:
            x_new: Updated node features [N, hidden_dim]
        """
        num_nodes = x.size(0)

        # Message aggregation: m_v = Σ_{u ∈ N(v)} w_{uv} · h_u
        messages = self._aggregate_messages(x, edge_index, edge_weights)

        # Concatenate hidden state and messages
        h_m = torch.cat([x, messages], dim=-1)  # [N, 2*hidden_dim]

        # Update gate
        z = torch.sigmoid(self.W_z(h_m))  # [N, hidden_dim]

        # Reset gate
        r = torch.sigmoid(self.W_r(h_m))  # [N, hidden_dim]

        # Candidate hidden state
        h_r = torch.cat([r * x, messages], dim=-1)  # [N, 2*hidden_dim]
        h_tilde = torch.tanh(self.W_h(h_r))  # [N, hidden_dim]

        # GRU update
        x_new = (1 - z) * x + z * h_tilde  # [N, hidden_dim]

        # Layer norm and dropout
        x_new = self.layer_norm(x_new)
        x_new = self.dropout(x_new)

        return x_new

    def _aggregate_messages(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate messages from neighbors with optional weighting.

        Args:
            x: Node features [N, hidden_dim]
            edge_index: Graph edges [2, E]
            edge_weights: Optional edge weights [E]

        Returns:
            messages: Aggregated messages [N, hidden_dim]
        """
        num_nodes = x.size(0)
        source, target = edge_index  # source → target

        # Get source node features
        source_features = x[source]  # [E, hidden_dim]

        # Apply edge weights if available
        if self.use_edge_weights and edge_weights is not None:
            # Expand weights for broadcasting
            weights = edge_weights.unsqueeze(-1)  # [E, 1]
            source_features = source_features * weights  # [E, hidden_dim]

        # Aggregate by target node (sum aggregation)
        messages = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
        messages.index_add_(0, target, source_features)

        # Normalize by degree (mean aggregation)
        degree = torch.zeros(num_nodes, device=x.device)
        degree.index_add_(0, target, torch.ones(len(target), device=x.device))
        degree = degree.clamp(min=1).unsqueeze(-1)  # [N, 1]
        messages = messages / degree

        return messages


class PhyloEncoder(nn.Module):
    """
    Phylo-Encoder: GGNN-based encoder for phylogenetic test case prioritization.

    This encoder treats the Git DAG as a phylogenetic tree and propagates
    failure signals through evolutionary history using Gated Graph Neural Networks.

    Architecture:
        1. Input Projection: Project input features to hidden dimension
        2. GGNN Layers: Stack of GGNN layers with phylogenetic distance weighting
        3. Temporal Attention: Attention over ancestor commits
        4. Output Projection: Final representation

    The phylogenetic distance kernel weights message propagation, ensuring
    that information from phylogenetically close commits has more influence.

    Key Innovation:
        - Unlike standard GNNs that treat all edges equally, PhyloEncoder
          weights edges by evolutionary distance (phylogenetic signal)
        - This captures the intuition that related commits (close in the DAG)
          should have similar failure patterns
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_distance_kernel: bool = True,
        decay_factor: float = 0.9,
        learnable_temperature: bool = True
    ):
        """
        Initialize PhyloEncoder.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for GGNN layers
            output_dim: Output dimension
            num_layers: Number of GGNN layers
            dropout: Dropout probability
            use_distance_kernel: Whether to use phylogenetic distance kernel
            decay_factor: Decay factor for distance kernel
            learnable_temperature: Whether temperature is learnable
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.use_distance_kernel = use_distance_kernel

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Phylogenetic Distance Kernel
        if use_distance_kernel:
            self.distance_kernel = PhylogeneticDistanceKernel(
                decay_factor=decay_factor,
                learnable_temperature=learnable_temperature
            )
        else:
            self.distance_kernel = None

        # Stack of GGNN layers
        self.ggnn_layers = nn.ModuleList([
            GGNNLayer(
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_edge_weights=use_distance_kernel
            )
            for _ in range(num_layers)
        ])

        # Temporal attention for aggregating ancestor information
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        logger.info("="*70)
        logger.info("PHYLO-ENCODER INITIALIZED")
        logger.info("="*70)
        logger.info(f"Input dim: {input_dim}")
        logger.info(f"Hidden dim: {hidden_dim}")
        logger.info(f"Output dim: {output_dim}")
        logger.info(f"GGNN layers: {num_layers}")
        logger.info(f"Distance kernel: {use_distance_kernel}")
        if use_distance_kernel:
            logger.info(f"  - Decay factor: {decay_factor}")
            logger.info(f"  - Learnable temperature: {learnable_temperature}")
        logger.info("="*70)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
        path_lengths: Optional[torch.Tensor] = None,
        merge_counts: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through PhyloEncoder.

        Args:
            x: Node features [N, input_dim]
               - For commits: CodeBERT/SBERT embeddings of commit messages
               - For tests: Concatenated test description embeddings
            edge_index: Graph edges [2, E] representing Git DAG
            edge_weights: Optional pre-computed edge weights [E]
            path_lengths: Optional path lengths for distance kernel [E]
            merge_counts: Optional merge counts for distance kernel [E]
            batch: Optional batch assignment for nodes [N]

        Returns:
            output: Phylogenetic representations [N, output_dim]
        """
        # Input projection
        h = self.input_proj(x)  # [N, hidden_dim]

        # Compute phylogenetic weights if using distance kernel
        if self.use_distance_kernel and self.distance_kernel is not None:
            if path_lengths is not None:
                phylo_weights = self.distance_kernel(
                    edge_index, path_lengths, merge_counts
                )
            elif edge_weights is not None:
                # Use provided edge weights directly
                phylo_weights = edge_weights
            else:
                # Default: uniform weights
                phylo_weights = torch.ones(edge_index.size(1), device=x.device)
        else:
            phylo_weights = edge_weights

        # Apply GGNN layers with residual connections
        for layer in self.ggnn_layers:
            h_new = layer(h, edge_index, phylo_weights)
            h = h + h_new  # Residual connection

        # Output projection
        output = self.output_proj(h)  # [N, output_dim]

        return output

    def get_ancestor_attention(
        self,
        x: torch.Tensor,
        ancestor_indices: torch.Tensor,
        ancestor_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention-weighted representation over ancestors.

        This is used for the "macro" level attention in hierarchical attention,
        allowing each commit to selectively attend to its ancestors.

        Args:
            x: Node features [N, hidden_dim]
            ancestor_indices: Indices of ancestors for each node [N, K]
                              where K is max number of ancestors
            ancestor_mask: Optional mask for valid ancestors [N, K]

        Returns:
            attended: Attention-weighted ancestor representations [N, hidden_dim]
        """
        N, K = ancestor_indices.shape

        # Gather ancestor features
        # Handle out-of-bounds indices by clamping
        valid_indices = ancestor_indices.clamp(0, x.size(0) - 1)
        ancestor_features = x[valid_indices.view(-1)].view(N, K, -1)  # [N, K, hidden_dim]

        # Current node features as query
        query = x.unsqueeze(1)  # [N, 1, hidden_dim]

        # Apply attention
        attended, attention_weights = self.temporal_attention(
            query=query,
            key=ancestor_features,
            value=ancestor_features,
            key_padding_mask=ancestor_mask
        )

        attended = attended.squeeze(1)  # [N, hidden_dim]

        return attended


class PhyloEncoderWithCodeBERT(nn.Module):
    """
    PhyloEncoder with CodeBERT integration for commit message encoding.

    This is the full implementation as described in the paper, combining:
    1. CodeBERT for semantic encoding of commit messages
    2. PhyloEncoder (GGNN) for phylogenetic propagation

    Note: For efficiency, this uses pre-computed CodeBERT embeddings
    rather than running CodeBERT during training.
    """

    def __init__(
        self,
        embedding_dim: int = 768,  # CodeBERT/SBERT dimension
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        decay_factor: float = 0.9
    ):
        """
        Initialize PhyloEncoder with CodeBERT.

        Args:
            embedding_dim: Dimension of pre-computed embeddings (768 for CodeBERT)
            hidden_dim: Hidden dimension for GGNN
            output_dim: Output dimension
            num_layers: Number of GGNN layers
            dropout: Dropout probability
            decay_factor: Decay factor for distance kernel
        """
        super().__init__()

        self.phylo_encoder = PhyloEncoder(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_distance_kernel=True,
            decay_factor=decay_factor,
            learnable_temperature=True
        )

        logger.info("Initialized PhyloEncoderWithCodeBERT")
        logger.info(f"  - Embedding dim (CodeBERT): {embedding_dim}")
        logger.info(f"  - Output dim: {output_dim}")

    def forward(
        self,
        commit_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
        path_lengths: Optional[torch.Tensor] = None,
        merge_counts: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            commit_embeddings: Pre-computed CodeBERT embeddings [N, 768]
            edge_index: Git DAG edges [2, E]
            edge_weights: Optional edge weights [E]
            path_lengths: Optional path lengths [E]
            merge_counts: Optional merge counts [E]

        Returns:
            phylo_representations: [N, output_dim]
        """
        return self.phylo_encoder(
            commit_embeddings,
            edge_index,
            edge_weights,
            path_lengths,
            merge_counts
        )


def create_phylo_encoder(config: Dict) -> PhyloEncoder:
    """
    Factory function to create PhyloEncoder from config.

    Args:
        config: Configuration dictionary with keys:
            - input_dim: Input feature dimension
            - hidden_dim: Hidden dimension (default 256)
            - output_dim: Output dimension (default 256)
            - num_layers: Number of GGNN layers (default 3)
            - dropout: Dropout probability (default 0.1)
            - use_distance_kernel: Use phylogenetic distance (default True)
            - decay_factor: Decay factor β (default 0.9)

    Returns:
        PhyloEncoder instance
    """
    return PhyloEncoder(
        input_dim=config.get('input_dim', 768),
        hidden_dim=config.get('hidden_dim', 256),
        output_dim=config.get('output_dim', 256),
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.1),
        use_distance_kernel=config.get('use_distance_kernel', True),
        decay_factor=config.get('decay_factor', 0.9),
        learnable_temperature=config.get('learnable_temperature', True)
    )


__all__ = [
    'PhylogeneticDistanceKernel',
    'GGNNLayer',
    'PhyloEncoder',
    'PhyloEncoderWithCodeBERT',
    'create_phylo_encoder'
]
