"""
Dual-Stream Model for Filo-Priori V8

This is the V8 implementation that FIXES the "Semantic Echo Chamber" problem.

Key Changes from V7:
- Semantic Stream: Processes BGE embeddings [1024] (text-based) - UNCHANGED
- Structural Stream: Processes historical features [6] (history-based) - NEW!

The structural stream NO LONGER uses k-NN graphs based on semantic similarity.
Instead, it processes true structural/phylogenetic features extracted from
test execution history.

Author: Filo-Priori V8 Team
Date: 2025-11-06
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

# Import GATConv for Step 2.4: Graph Attention Networks
try:
    from torch_geometric.nn import GATConv
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    logging.warning("torch_geometric not available. GAT-based StructuralStreamV8 will not work.")

logger = logging.getLogger(__name__)


class SemanticStream(nn.Module):
    """
    Semantic stream for V8: processes text embeddings.

    UNCHANGED from V7 - still processes BGE embeddings [batch, 1024]
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stack of FFN layers with residual connections
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            )
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input embeddings [batch_size, 1024] from BGE encoder

        Returns:
            Processed semantic features [batch_size, hidden_dim]
        """
        x = self.input_proj(x)

        for layer in self.layers:
            x = x + layer(x)  # Residual connection

        x = self.output_norm(x)
        return x


class StructuralStreamV8(nn.Module):
    """
    GAT-based structural stream for V8: processes phylogenetic features with graph attention.

    STEP 2.4 UPGRADE - GRAPH ATTENTION NETWORKS:
    - V7: Used mean aggregation in MessagePassingLayer (inconsistent with thesis!)
    - V8 (Step 2.2): Used simple FFN without graph structure
    - V8 (Step 2.4): Uses Graph Attention Networks with multi-head attention ✓

    This upgrade unifies the ENTIRE V8 architecture under the attention paradigm:
    - Semantic stream: Uses transformer attention (BGE)
    - Structural stream: Uses graph attention (GAT) ← NEW!
    - Fusion layer: Uses cross-attention

    Strengthens thesis narrative: "Attention is superior to mean aggregation"

    Features processed (from StructuralFeatureExtractor):
    1. test_age: Builds since first appearance
    2. failure_rate: Historical failure rate
    3. recent_failure_rate: Recent failure trend
    4. flakiness_rate: Pass/Fail oscillation
    5. commit_count: Number of associated commits
    6. test_novelty: First appearance flag

    Graph structure (from PhylogeneticGraphBuilder):
    - Co-failure graph: Tests failing together
    - Commit-dependency graph: Tests sharing commits
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3,
        activation: str = 'elu',
        use_edge_weights: bool = True
    ):
        """
        Initialize GAT-based structural stream.

        Args:
            input_dim: Number of input features (default: 6)
            hidden_dim: Hidden dimension for GAT layers (default: 256)
            num_heads: Number of attention heads for first GAT layer (default: 4)
            dropout: Dropout probability (default: 0.3)
            activation: Activation function ('elu' or 'relu', default: 'elu')
            use_edge_weights: Whether to use edge weights from phylogenetic graph
        """
        super().__init__()

        if not HAS_TORCH_GEOMETRIC:
            raise ImportError(
                "torch_geometric is required for GAT-based StructuralStreamV8. "
                "Install with: pip install torch-geometric"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_edge_weights = use_edge_weights

        # First GAT layer: [N, 6] → [N, hidden_dim * num_heads]
        # Multi-head attention with concatenation
        self.conv1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True,  # Concatenate attention heads
            edge_dim=1 if use_edge_weights else None  # Use edge weights if available
        )

        # Second GAT layer: [N, hidden_dim * num_heads] → [N, 256]
        # Single-head attention with averaging
        self.conv2 = GATConv(
            in_channels=hidden_dim * num_heads,
            out_channels=256,
            heads=1,
            dropout=dropout,
            concat=False,  # Average attention heads
            edge_dim=1 if use_edge_weights else None
        )

        self.dropout = dropout

        # Activation function
        if activation == 'elu':
            self.activation = F.elu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = F.gelu

        logger.info(f"Initialized GAT-based StructuralStreamV8:")
        logger.info(f"  - Input: [N, {input_dim}] structural features")
        logger.info(f"  - GAT Layer 1: {num_heads} heads, output [N, {hidden_dim * num_heads}]")
        logger.info(f"  - GAT Layer 2: 1 head, output [N, 256]")
        logger.info(f"  - Edge weights: {use_edge_weights}")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with graph attention.

        Args:
            x: Input structural features [N, 6]
               Features: [test_age, failure_rate, recent_failure_rate,
                         flakiness_rate, commit_count, test_novelty]
            edge_index: Graph connectivity [2, E] from phylogenetic graph
            edge_weights: Optional edge weights [E] from phylogenetic graph

        Returns:
            Processed structural features [N, 256] with graph attention applied
        """
        # Prepare edge attributes if using edge weights
        edge_attr = None
        if self.use_edge_weights and edge_weights is not None:
            # Reshape edge_weights to [E, 1] for GATConv
            edge_attr = edge_weights.unsqueeze(-1) if edge_weights.dim() == 1 else edge_weights

        # First GAT layer with multi-head attention
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.activation(x)

        # Second GAT layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)

        return x


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion for V8.

    Fuses semantic features (from text) with structural features (from history).

    UNCHANGED from V7 architecture - only the inputs are different:
    - V7: Both inputs were derived from BGE embeddings (echo chamber!)
    - V8: Inputs are truly orthogonal (text vs history)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        # Cross-attention: semantic → structural
        self.cross_attn_sem2struct = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention: structural → semantic
        self.cross_attn_struct2sem = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.norm_sem = nn.LayerNorm(hidden_dim)
        self.norm_struct = nn.LayerNorm(hidden_dim)

        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh()
        )

    def forward(
        self,
        semantic_features: torch.Tensor,
        structural_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: fuse semantic and structural features

        Args:
            semantic_features: [batch_size, hidden_dim] from SemanticStream
            structural_features: [batch_size, hidden_dim] from StructuralStreamV8

        Returns:
            fused_features: [batch_size, hidden_dim * 2]
        """
        # Add sequence dimension for attention
        sem_seq = semantic_features.unsqueeze(1)      # [batch, 1, hidden_dim]
        struct_seq = structural_features.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Cross-attention: semantic attends to structural
        sem_attended, _ = self.cross_attn_sem2struct(
            query=sem_seq,
            key=struct_seq,
            value=struct_seq
        )
        sem_attended = sem_attended.squeeze(1)  # [batch, hidden_dim]
        semantic_enhanced = self.norm_sem(semantic_features + sem_attended)

        # Cross-attention: structural attends to semantic
        struct_attended, _ = self.cross_attn_struct2sem(
            query=struct_seq,
            key=sem_seq,
            value=sem_seq
        )
        struct_attended = struct_attended.squeeze(1)  # [batch, hidden_dim]
        structural_enhanced = self.norm_struct(structural_features + struct_attended)

        # Concatenate enhanced features
        fused = torch.cat([semantic_enhanced, structural_enhanced], dim=-1)  # [batch, hidden_dim*2]

        return fused


class GatedFusionUnit(nn.Module):
    """
    Gated Fusion Unit (GFU) for dynamic modality arbitration.

    STEP 2.5: GATED FUSION
    - Replaces CrossAttentionFusion with learned gating mechanism
    - Dynamically decides how much each modality contributes
    - Superior for sparse/noisy structural features (e.g., new tests with no history)

    Scientific Motivation:
    - Cross-attention "combines" modalities
    - Gated fusion "arbitrates" modalities
    - For new test cases with zero history (structural = [0,0,0,0,0,0]),
      the gate learns to suppress structural stream and rely on semantic
    - Handles data sparsity inherent in test execution history

    Mathematical Formulation:
        z = σ(W_z · x_sem + U_z · x_struct + b_z)  # Gate (learned importance)
        y_fused = z ⊙ x_sem + (1-z) ⊙ x_struct      # Gated fusion

    Where:
        - z ∈ [0, 1]^hidden_dim: learned gate (per dimension)
        - z ≈ 1: rely on semantic (suppress structural)
        - z ≈ 0: rely on structural (suppress semantic)
        - z ≈ 0.5: balanced fusion

    References:
        [28] Gated Multimodal Units (GMU)
        [30] Gated Fusion Units (GFU)
        [31] Dynamic modality arbitration
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_projection: bool = True
    ):
        """
        Initialize Gated Fusion Unit.

        Args:
            hidden_dim: Dimension of input features
            dropout: Dropout probability
            use_projection: If True, project inputs before gating
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_projection = use_projection

        # Optional input projections
        if use_projection:
            self.proj_sem = nn.Linear(hidden_dim, hidden_dim)
            self.proj_struct = nn.Linear(hidden_dim, hidden_dim)

        # Gate network: learns importance of each modality
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()  # z ∈ [0, 1]
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout)
        )

        logger.info("Initialized GatedFusionUnit:")
        logger.info(f"  - Hidden dim: {hidden_dim}")
        logger.info(f"  - Gate: learns z ∈ [0,1] for dynamic arbitration")
        logger.info(f"  - Output: [batch, {hidden_dim * 2}]")

    def forward(
        self,
        semantic_features: torch.Tensor,
        structural_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: gated fusion of semantic and structural features.

        Args:
            semantic_features: [batch_size, hidden_dim] from SemanticStream
            structural_features: [batch_size, hidden_dim] from StructuralStreamV8

        Returns:
            fused_features: [batch_size, hidden_dim * 2]

        Algorithm:
            1. Optional: project inputs
            2. Concatenate for gate input
            3. Compute gate z = σ(...)
            4. Gated fusion: y = z ⊙ x_sem + (1-z) ⊙ x_struct
            5. Project output
        """
        # Optional projections
        if self.use_projection:
            x_sem = self.proj_sem(semantic_features)
            x_struct = self.proj_struct(structural_features)
        else:
            x_sem = semantic_features
            x_struct = structural_features

        # Compute gate: z ∈ [0, 1]^hidden_dim
        # Concatenate both modalities to inform gate
        gate_input = torch.cat([x_sem, x_struct], dim=-1)  # [batch, hidden_dim*2]
        z = self.gate(gate_input)  # [batch, hidden_dim]

        # Gated fusion: element-wise weighted combination
        # z ≈ 1: rely on semantic (structural is noise)
        # z ≈ 0: rely on structural (semantic is weak)
        fused = z * x_sem + (1 - z) * x_struct  # [batch, hidden_dim]

        # Project to output dimension
        output = self.output_proj(fused)  # [batch, hidden_dim*2]

        return output


class SimpleClassifier(nn.Module):
    """
    Simple MLP classifier for binary classification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64],
        num_classes: int = 2,
        dropout: float = 0.4
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, num_classes))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            logits: [batch_size, num_classes]
        """
        return self.mlp(x)


class DualStreamModelV8(nn.Module):
    """
    Dual-Stream Model for Filo-Priori V8

    FIXED ARCHITECTURE:
    - Semantic Stream: Processes text embeddings [batch, 1024]
    - Structural Stream: Processes historical features [batch, 6]
    - Fusion: Bidirectional cross-attention
    - Classifier: Binary (Pass vs Not-Pass)

    This architecture properly validates the thesis hypothesis by using
    truly orthogonal information sources.
    """

    def __init__(
        self,
        semantic_config: Optional[Dict] = None,
        structural_config: Optional[Dict] = None,
        fusion_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None,
        num_classes: int = 2
    ):
        super().__init__()

        self.num_classes = num_classes

        # Default configs
        semantic_config = semantic_config or {}
        structural_config = structural_config or {}
        fusion_config = fusion_config or {}
        classifier_config = classifier_config or {}

        # Get hidden dimensions
        semantic_hidden = semantic_config.get('hidden_dim', 256)
        structural_hidden = structural_config.get('hidden_dim', 256)

        # Ensure both streams have same hidden_dim for fusion
        if semantic_hidden != structural_hidden:
            logger.warning(
                f"Semantic and structural hidden dims differ "
                f"({semantic_hidden} vs {structural_hidden}). "
                f"Using semantic_hidden={semantic_hidden} for both."
            )
            structural_hidden = semantic_hidden

        # Build streams
        semantic_input_dim = semantic_config.get('input_dim', 1024)
        self.semantic_stream = SemanticStream(
            input_dim=semantic_input_dim,
            hidden_dim=semantic_hidden,
            num_layers=semantic_config.get('num_layers', 2),
            dropout=semantic_config.get('dropout', 0.3),
            activation=semantic_config.get('activation', 'gelu')
        )

        structural_input_dim = structural_config.get('input_dim', 6)
        self.structural_stream = StructuralStreamV8(
            input_dim=structural_input_dim,
            hidden_dim=structural_hidden,
            num_heads=structural_config.get('num_heads', 4),  # GAT: multi-head attention
            dropout=structural_config.get('dropout', 0.3),
            activation=structural_config.get('activation', 'elu'),  # GAT: ELU activation
            use_edge_weights=structural_config.get('use_edge_weights', True)
        )

        # Build fusion (support both types)
        fusion_type = fusion_config.get('type', 'cross_attention')

        if fusion_type == 'gated':
            logger.info("Using GatedFusionUnit (Step 2.5)")
            self.fusion = GatedFusionUnit(
                hidden_dim=semantic_hidden,
                dropout=fusion_config.get('dropout', 0.1),
                use_projection=fusion_config.get('use_projection', True)
            )
        elif fusion_type == 'cross_attention':
            logger.info("Using CrossAttentionFusion (default)")
            self.fusion = CrossAttentionFusion(
                hidden_dim=semantic_hidden,
                num_heads=fusion_config.get('num_heads', 4),
                dropout=fusion_config.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. Must be 'cross_attention' or 'gated'")

        # Build classifier
        fusion_dim = semantic_hidden * 2  # Concatenated features
        self.classifier = SimpleClassifier(
            input_dim=fusion_dim,
            hidden_dims=classifier_config.get('hidden_dims', [128, 64]),
            num_classes=num_classes,
            dropout=classifier_config.get('dropout', 0.4)
        )

        logger.info("="*70)
        logger.info("DUAL-STREAM MODEL V8 INITIALIZED")
        logger.info("="*70)
        logger.info(f"Semantic Stream: [batch, {semantic_input_dim}] → [batch, {semantic_hidden}]")
        logger.info(f"Structural Stream: [batch, {structural_input_dim}] → [batch, {structural_hidden}]")
        logger.info(f"Fusion: [batch, {fusion_dim}]")
        logger.info(f"Classifier: [batch, {fusion_dim}] → [batch, {num_classes}]")
        logger.info("="*70)

    def forward(
        self,
        semantic_input: torch.Tensor,
        structural_input: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            semantic_input: Text embeddings [batch_size, 1024] from BGE
            structural_input: Historical features [batch_size, 6]
                             [test_age, failure_rate, recent_failure_rate,
                              flakiness_rate, commit_count, test_novelty]
            edge_index: Graph connectivity [2, E] from phylogenetic graph
            edge_weights: Optional edge weights [E] from phylogenetic graph

        Returns:
            logits: [batch_size, num_classes]
        """
        # Process streams
        semantic_features = self.semantic_stream(semantic_input)
        structural_features = self.structural_stream(
            structural_input,
            edge_index,
            edge_weights
        )

        # Fuse features
        fused_features = self.fusion(semantic_features, structural_features)

        # Classify
        logits = self.classifier(fused_features)

        return logits

    def get_feature_representations(
        self,
        semantic_input: torch.Tensor,
        structural_input: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get intermediate feature representations (useful for analysis)

        Args:
            semantic_input: Text embeddings [batch_size, 1024]
            structural_input: Historical features [batch_size, 6]
            edge_index: Graph connectivity [2, E] from phylogenetic graph
            edge_weights: Optional edge weights [E] from phylogenetic graph

        Returns:
            semantic_features: [batch_size, hidden_dim]
            structural_features: [batch_size, hidden_dim]
            fused_features: [batch_size, hidden_dim*2]
        """
        semantic_features = self.semantic_stream(semantic_input)
        structural_features = self.structural_stream(
            structural_input,
            edge_index,
            edge_weights
        )
        fused_features = self.fusion(semantic_features, structural_features)

        return semantic_features, structural_features, fused_features


def create_model_v8(config: Dict) -> DualStreamModelV8:
    """
    Factory function to create DualStreamModelV8 from config.

    Args:
        config: Configuration dictionary

    Returns:
        DualStreamModelV8 instance

    Example config:
        {
            'semantic': {
                'input_dim': 1024,
                'hidden_dim': 256,
                'num_layers': 2,
                'dropout': 0.3
            },
            'structural': {
                'input_dim': 6,
                'hidden_dim': 256,
                'num_layers': 2,
                'dropout': 0.3,
                'use_batch_norm': True
            },
            'fusion': {
                'num_heads': 4,
                'dropout': 0.1
            },
            'classifier': {
                'hidden_dims': [128, 64],
                'dropout': 0.4
            },
            'num_classes': 2
        }
    """
    return DualStreamModelV8(
        semantic_config=config.get('semantic', {}),
        structural_config=config.get('structural', {}),
        fusion_config=config.get('fusion', {}),
        classifier_config=config.get('classifier', {}),
        num_classes=config.get('num_classes', 2)
    )


__all__ = [
    'SemanticStream',
    'StructuralStreamV8',
    'CrossAttentionFusion',
    'GatedFusionUnit',  # Step 2.5
    'SimpleClassifier',
    'DualStreamModelV8',
    'create_model_v8'
]
