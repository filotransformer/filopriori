"""
Phylogenetic Dual-Stream Model for Filo-Priori V9

This module integrates the PhyloEncoder (GGNN) with the existing DualStreamModelV8
architecture, creating a complete phylogenetic approach to TCP.

Architecture Overview:
    1. PhyloEncoder (GGNN): Processes Git DAG with phylogenetic distance weighting
    2. Code-Encoder (GATv2): Processes test relationship graph with semantic features
    3. Hierarchical Attention: Multi-scale attention (micro/meso/macro)
    4. Cross-Attention Fusion: Fuses phylogenetic and structural representations
    5. Ranking Module: Produces failure probabilities for ranking

Key Innovations:
    - Treats Git DAG as phylogenetic tree
    - Phylogenetic distance kernel weights message propagation
    - Hierarchical attention at three scales
    - Combined loss: Focal + Ranking + Phylo-Regularization

Author: Filo-Priori V9 Team
Date: November 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

from .phylo_encoder import PhyloEncoder, PhylogeneticDistanceKernel, create_phylo_encoder
from .dual_stream_v8 import (
    SemanticStream,
    StructuralStreamV8,
    CrossAttentionFusion,
    GatedFusionUnit,
    SimpleClassifier
)

logger = logging.getLogger(__name__)


class HierarchicalAttention(nn.Module):
    """
    Hierarchical Attention mechanism operating at three scales:

    1. Micro Level (Code): Token-level attention within code
    2. Meso Level (Call Graph): Method/test relationship attention
    3. Macro Level (History): Commit history attention

    This multi-scale approach captures dependencies from fine-grained code
    to high-level evolutionary patterns.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize Hierarchical Attention.

        Args:
            hidden_dim: Hidden dimension for all attention layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # Micro-level: Self-attention over code tokens (simulated)
        self.micro_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Meso-level: Attention over method-test relationships
        self.meso_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Macro-level: Temporal attention over commit history
        self.macro_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norms
        self.norm_micro = nn.LayerNorm(hidden_dim)
        self.norm_meso = nn.LayerNorm(hidden_dim)
        self.norm_macro = nn.LayerNorm(hidden_dim)

        # Fusion of three levels
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        logger.info(f"Initialized HierarchicalAttention:")
        logger.info(f"  - Hidden dim: {hidden_dim}")
        logger.info(f"  - Attention heads: {num_heads}")
        logger.info(f"  - Levels: Micro, Meso, Macro")

    def forward(
        self,
        code_features: torch.Tensor,
        graph_features: torch.Tensor,
        history_features: torch.Tensor,
        phylo_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through hierarchical attention.

        Args:
            code_features: Semantic code features [batch, hidden_dim]
            graph_features: Graph-based structural features [batch, hidden_dim]
            history_features: Phylogenetic history features [batch, hidden_dim]
            phylo_weights: Optional phylogenetic weights for macro attention

        Returns:
            fused: Hierarchically fused features [batch, hidden_dim]
        """
        batch_size = code_features.size(0)

        # Add sequence dimension for attention
        code_seq = code_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        graph_seq = graph_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        history_seq = history_features.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Micro-level: Self-attention on code
        micro_out, _ = self.micro_attention(code_seq, code_seq, code_seq)
        micro_out = self.norm_micro(micro_out.squeeze(1) + code_features)

        # Meso-level: Cross-attention (code attends to graph)
        meso_out, _ = self.meso_attention(
            code_seq, graph_seq, graph_seq
        )
        meso_out = self.norm_meso(meso_out.squeeze(1) + graph_features)

        # Macro-level: Cross-attention (code attends to history)
        # Optionally weighted by phylogenetic distance
        macro_out, _ = self.macro_attention(
            code_seq, history_seq, history_seq
        )
        macro_out = self.norm_macro(macro_out.squeeze(1) + history_features)

        # Fuse all three levels
        combined = torch.cat([micro_out, meso_out, macro_out], dim=-1)
        fused = self.fusion(combined)

        return fused


class PhylogeneticRegularization(nn.Module):
    """
    Phylogenetic Regularization Loss.

    Encourages predictions to be consistent with evolutionary structure:
    - Phylogenetically close commits should have similar failure predictions
    - Penalizes large prediction differences for evolutionarily related commits

    Loss Formula:
        L_phylo = Σ_{(c_i, c_j) ∈ E_DAG} w_phylo(c_i, c_j) × |p(c_i) - p(c_j)|

    Where:
        - w_phylo: Phylogenetic weight (higher for closer commits)
        - p(c): Predicted failure probability for commit c
    """

    def __init__(self, weight: float = 0.1):
        """
        Initialize Phylogenetic Regularization.

        Args:
            weight: Weight for this loss component in combined loss
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        predictions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute phylogenetic regularization loss.

        Args:
            predictions: Failure probabilities [N]
            edge_index: Graph edges [2, E]
            edge_weights: Phylogenetic weights [E]

        Returns:
            loss: Regularization loss scalar
        """
        if edge_index.size(1) == 0:
            return torch.tensor(0.0, device=predictions.device)

        source, target = edge_index

        # Get predictions for source and target nodes
        pred_source = predictions[source]
        pred_target = predictions[target]

        # Compute weighted L1 difference
        diff = torch.abs(pred_source - pred_target)
        weighted_diff = edge_weights * diff

        # Mean loss
        loss = weighted_diff.mean() * self.weight

        return loss


class PhylogeneticDualStreamModel(nn.Module):
    """
    Complete Phylogenetic Dual-Stream Model for Filo-Priori V9.

    This model integrates all components of the phylogenetic approach:
    1. PhyloEncoder for Git DAG processing
    2. Code-Encoder (existing StructuralStreamV8 with GATv2)
    3. Hierarchical Attention for multi-scale feature fusion
    4. Cross-Attention Fusion for final combination
    5. Classifier for failure prediction

    The model receives:
    - Semantic embeddings (test descriptions, commit messages)
    - Structural features (historical execution patterns)
    - Test relationship graph (co-failure, semantic edges)
    - Git DAG topology (optional, for PhyloEncoder)
    """

    def __init__(
        self,
        semantic_config: Optional[Dict] = None,
        structural_config: Optional[Dict] = None,
        phylo_config: Optional[Dict] = None,
        fusion_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None,
        use_phylo_encoder: bool = True,
        use_hierarchical_attention: bool = True,
        num_classes: int = 2
    ):
        """
        Initialize Phylogenetic Dual-Stream Model.

        Args:
            semantic_config: Config for semantic stream
            structural_config: Config for structural stream (GATv2)
            phylo_config: Config for PhyloEncoder
            fusion_config: Config for fusion layers
            classifier_config: Config for classifier
            use_phylo_encoder: Whether to use PhyloEncoder
            use_hierarchical_attention: Whether to use hierarchical attention
            num_classes: Number of output classes
        """
        super().__init__()

        self.num_classes = num_classes
        self.use_phylo_encoder = use_phylo_encoder
        self.use_hierarchical_attention = use_hierarchical_attention

        # Default configs
        semantic_config = semantic_config or {}
        structural_config = structural_config or {}
        phylo_config = phylo_config or {}
        fusion_config = fusion_config or {}
        classifier_config = classifier_config or {}

        # Hidden dimension (consistent across all modules)
        hidden_dim = semantic_config.get('hidden_dim', 256)

        # 1. Semantic Stream (processes text embeddings)
        self.semantic_stream = SemanticStream(
            input_dim=semantic_config.get('input_dim', 1536),
            hidden_dim=hidden_dim,
            num_layers=semantic_config.get('num_layers', 2),
            dropout=semantic_config.get('dropout', 0.3)
        )

        # 2. Structural Stream (GATv2 over test graph)
        self.structural_stream = StructuralStreamV8(
            input_dim=structural_config.get('input_dim', 10),
            hidden_dim=hidden_dim,
            num_heads=structural_config.get('num_heads', 2),
            dropout=structural_config.get('dropout', 0.3)
        )

        # 3. PhyloEncoder (GGNN over Git DAG) - OPTIONAL
        # For HYBRID mode, allow smaller dimensions to reduce overhead
        phylo_hidden = phylo_config.get('hidden_dim', hidden_dim)
        phylo_output = phylo_config.get('output_dim', hidden_dim)

        if use_phylo_encoder:
            self.phylo_encoder = PhyloEncoder(
                input_dim=phylo_config.get('input_dim', 768),
                hidden_dim=phylo_hidden,
                output_dim=phylo_output,
                num_layers=phylo_config.get('num_layers', 3),
                dropout=phylo_config.get('dropout', 0.1),
                use_distance_kernel=phylo_config.get('use_distance_kernel', True),
                decay_factor=phylo_config.get('decay_factor', 0.9)
            )
            logger.info(f"  PhyloEncoder: hidden={phylo_hidden}, output={phylo_output}")
        else:
            self.phylo_encoder = None

        # 4. Hierarchical Attention - OPTIONAL
        if use_hierarchical_attention:
            self.hierarchical_attention = HierarchicalAttention(
                hidden_dim=hidden_dim,
                num_heads=fusion_config.get('num_heads', 4),
                dropout=fusion_config.get('dropout', 0.1)
            )
        else:
            self.hierarchical_attention = None

        # 5. Cross-Attention Fusion
        fusion_type = fusion_config.get('type', 'cross_attention')
        if fusion_type == 'gated':
            self.fusion = GatedFusionUnit(
                hidden_dim=hidden_dim,
                dropout=fusion_config.get('dropout', 0.1)
            )
        else:
            self.fusion = CrossAttentionFusion(
                hidden_dim=hidden_dim,
                num_heads=fusion_config.get('num_heads', 4),
                dropout=fusion_config.get('dropout', 0.1)
            )

        # 6. Classifier
        fusion_dim = hidden_dim * 2
        self.classifier = SimpleClassifier(
            input_dim=fusion_dim,
            hidden_dims=classifier_config.get('hidden_dims', [128, 64]),
            num_classes=num_classes,
            dropout=classifier_config.get('dropout', 0.4)
        )

        # 7. Phylogenetic Regularization
        self.phylo_regularization = PhylogeneticRegularization(
            weight=fusion_config.get('phylo_reg_weight', 0.1)
        )

        # 8. Phylo Projection (for hybrid mode: phylo_encoder ON, hierarchical OFF)
        # Projects phylo features to match structural dimension for fusion
        if use_phylo_encoder and not use_hierarchical_attention and phylo_output != hidden_dim:
            self.phylo_projection = nn.Sequential(
                nn.Linear(phylo_output, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
            logger.info(f"  Hybrid mode: PhyloProjection [{phylo_output} -> {hidden_dim}]")
        else:
            self.phylo_projection = None

        self._log_architecture()

    def _log_architecture(self):
        """Log model architecture details."""
        logger.info("="*70)
        logger.info("PHYLOGENETIC DUAL-STREAM MODEL V9")
        logger.info("="*70)

        # Detect mode
        if self.use_phylo_encoder and not self.use_hierarchical_attention:
            mode = "HYBRID (PhyloEncoder + GATv2, no HierarchicalAttention)"
        elif self.use_phylo_encoder and self.use_hierarchical_attention:
            mode = "FULL PHYLOGENETIC"
        else:
            mode = "STANDARD (GATv2 only)"
        logger.info(f"MODE: {mode}")
        logger.info("-"*70)

        logger.info("Modules:")
        logger.info("  1. SemanticStream (text embeddings)")
        logger.info("  2. StructuralStreamV8 (GATv2 over test graph)")
        if self.use_phylo_encoder:
            logger.info("  3. PhyloEncoder (GGNN over Git DAG) ✓")
        else:
            logger.info("  3. PhyloEncoder DISABLED")
        if self.use_hierarchical_attention:
            logger.info("  4. HierarchicalAttention (Micro/Meso/Macro) ✓")
        else:
            logger.info("  4. HierarchicalAttention DISABLED")
        logger.info("  5. CrossAttentionFusion")
        logger.info("  6. Classifier")
        logger.info("  7. PhylogeneticRegularization")
        if self.phylo_projection is not None:
            logger.info("  8. PhyloProjection (hybrid mode dimension alignment)")
        logger.info("="*70)

    def forward(
        self,
        semantic_input: torch.Tensor,
        structural_input: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None,
        phylo_input: Optional[torch.Tensor] = None,
        phylo_edge_index: Optional[torch.Tensor] = None,
        phylo_path_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            semantic_input: Semantic embeddings [batch, 1536]
            structural_input: Structural features [N, 10]
            edge_index: Test relationship graph [2, E]
            edge_weights: Edge weights for test graph [E]
            phylo_input: Commit embeddings for PhyloEncoder [M, 768]
            phylo_edge_index: Git DAG edges [2, E_dag]
            phylo_path_lengths: Path lengths for phylo distance [E_dag]

        Returns:
            logits: Class logits [batch, num_classes]
        """
        # 1. Process semantic stream
        semantic_features = self.semantic_stream(semantic_input)

        # 2. Process structural stream with GATv2
        structural_features = self.structural_stream(
            structural_input, edge_index, edge_weights
        )

        # 3. Process phylogenetic stream (if enabled)
        phylo_from_encoder = False  # Track if phylo_features came from PhyloEncoder
        if self.use_phylo_encoder and self.phylo_encoder is not None:
            if phylo_input is not None and phylo_edge_index is not None:
                phylo_features = self.phylo_encoder(
                    phylo_input,
                    phylo_edge_index,
                    path_lengths=phylo_path_lengths
                )
                # Average pool to match batch size if needed
                if phylo_features.size(0) != semantic_features.size(0):
                    phylo_features = phylo_features.mean(dim=0, keepdim=True)
                    phylo_features = phylo_features.expand(semantic_features.size(0), -1)
                phylo_from_encoder = True
            else:
                # Fallback: use structural features
                phylo_features = structural_features
        else:
            phylo_features = structural_features

        # 4. Apply hierarchical attention (if enabled)
        if self.use_hierarchical_attention and self.hierarchical_attention is not None:
            attended_features = self.hierarchical_attention(
                semantic_features,
                structural_features,
                phylo_features
            )
            # Use attended features for fusion
            fused_features = self.fusion(semantic_features, attended_features)
        elif self.use_phylo_encoder and phylo_from_encoder:
            # HYBRID MODE: PhyloEncoder ON, HierarchicalAttention OFF
            # Combine structural and phylo features before fusion
            if self.phylo_projection is not None:
                phylo_features = self.phylo_projection(phylo_features)
            # Weighted combination: structural + phylo (both contribute)
            combined_features = structural_features + phylo_features
            fused_features = self.fusion(semantic_features, combined_features)
        else:
            # Standard fusion (no phylo)
            fused_features = self.fusion(semantic_features, structural_features)

        # 5. Classify
        logits = self.classifier(fused_features)

        return logits

    def compute_phylo_regularization(
        self,
        predictions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute phylogenetic regularization loss.

        Args:
            predictions: Softmax probabilities [batch, 2]
            edge_index: Graph edges [2, E]
            edge_weights: Edge weights [E]

        Returns:
            loss: Regularization loss
        """
        # Use failure probability (class 1)
        fail_probs = predictions[:, 1] if predictions.dim() > 1 else predictions
        return self.phylo_regularization(fail_probs, edge_index, edge_weights)


def create_phylogenetic_model(config: Dict) -> PhylogeneticDualStreamModel:
    """
    Factory function to create PhylogeneticDualStreamModel from config.

    Args:
        config: Configuration dictionary

    Returns:
        PhylogeneticDualStreamModel instance

    Example config:
        {
            'semantic': {'input_dim': 1536, 'hidden_dim': 256},
            'structural': {'input_dim': 10, 'hidden_dim': 256, 'num_heads': 2},
            'phylo': {'input_dim': 768, 'num_layers': 3, 'decay_factor': 0.9},
            'fusion': {'type': 'cross_attention', 'phylo_reg_weight': 0.1},
            'classifier': {'hidden_dims': [128, 64]},
            'use_phylo_encoder': True,
            'use_hierarchical_attention': True,
            'num_classes': 2
        }
    """
    return PhylogeneticDualStreamModel(
        semantic_config=config.get('semantic', {}),
        structural_config=config.get('structural', {}),
        phylo_config=config.get('phylo', {}),
        fusion_config=config.get('fusion', {}),
        classifier_config=config.get('classifier', {}),
        use_phylo_encoder=config.get('use_phylo_encoder', True),
        use_hierarchical_attention=config.get('use_hierarchical_attention', True),
        num_classes=config.get('num_classes', 2)
    )


__all__ = [
    'HierarchicalAttention',
    'PhylogeneticRegularization',
    'PhylogeneticDualStreamModel',
    'create_phylogenetic_model'
]
