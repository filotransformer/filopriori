"""
Ablation-aware Dual-Stream Model for Filo-Priori v9.

This module extends DualStreamModelV8 with ablation support for systematic
evaluation of each component's contribution.

Supported Ablations:
- A1: disable_semantic - Bypasses semantic stream (uses zeros)
- A2: disable_structural - Bypasses structural stream (uses zeros)
- A3: disable_gat - Uses MLP instead of Graph Attention
- A6: disable_cross_attention - Uses simple concatenation instead of cross-attention

Author: Filo-Priori Team
Date: 2025-11-26
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

from .dual_stream_v8 import (
    SemanticStream,
    StructuralStreamV8,
    CrossAttentionFusion,
    GatedFusionUnit,
    SimpleClassifier
)

logger = logging.getLogger(__name__)


class StructuralStreamMLP(nn.Module):
    """
    Simple MLP-based structural stream (no graph attention).

    Used for ablation A3: w/o GATv2.
    Processes structural features without graph structure.
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # MLP layers (no graph structure)
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ))
        self.layers = nn.ModuleList(layers)

        self.output_norm = nn.LayerNorm(hidden_dim)

        logger.info(f"StructuralStreamMLP: No graph attention (ablation mode)")
        logger.info(f"  - Input: [batch, {input_dim}]")
        logger.info(f"  - Output: [batch, {hidden_dim}]")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor = None,  # Ignored
        edge_weights: torch.Tensor = None  # Ignored
    ) -> torch.Tensor:
        """
        Forward pass (ignores graph structure).

        Args:
            x: Structural features [N, input_dim]
            edge_index: Ignored (for API compatibility)
            edge_weights: Ignored (for API compatibility)

        Returns:
            Processed features [N, hidden_dim]
        """
        x = self.input_proj(x)

        for layer in self.layers:
            x = x + layer(x)  # Residual connection

        return self.output_norm(x)


class SimpleConcatFusion(nn.Module):
    """
    Simple concatenation fusion (no cross-attention).

    Used for ablation A6: w/o Cross-Attention.
    Just concatenates semantic and structural features.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Simple projection after concatenation
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout)
        )

        logger.info(f"SimpleConcatFusion: No cross-attention (ablation mode)")

    def forward(
        self,
        semantic_features: torch.Tensor,
        structural_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: simple concatenation.

        Args:
            semantic_features: [batch, hidden_dim]
            structural_features: [batch, hidden_dim]

        Returns:
            Concatenated features: [batch, hidden_dim * 2]
        """
        # Simple concatenation
        fused = torch.cat([semantic_features, structural_features], dim=-1)

        return self.output_proj(fused)


class AblationDualStreamModel(nn.Module):
    """
    Ablation-aware Dual-Stream Model.

    Supports disabling specific components for ablation studies.

    Ablation Flags:
        disable_semantic: Use zeros instead of semantic stream
        disable_structural: Use zeros instead of structural stream
        disable_gat: Use MLP instead of GAT for structural stream
        disable_cross_attention: Use concatenation instead of cross-attention
    """

    def __init__(
        self,
        semantic_config: Optional[Dict] = None,
        structural_config: Optional[Dict] = None,
        fusion_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None,
        ablation_config: Optional[Dict] = None,
        num_classes: int = 2
    ):
        super().__init__()

        self.num_classes = num_classes

        # Default configs
        semantic_config = semantic_config or {}
        structural_config = structural_config or {}
        fusion_config = fusion_config or {}
        classifier_config = classifier_config or {}
        ablation_config = ablation_config or {}

        # Ablation flags
        self.disable_semantic = ablation_config.get('disable_semantic', False)
        self.disable_structural = ablation_config.get('disable_structural', False)
        self.disable_gat = ablation_config.get('disable_gat', False)
        self.disable_cross_attention = ablation_config.get('disable_cross_attention', False)

        # Log ablation mode
        logger.info("=" * 70)
        logger.info("ABLATION DUAL-STREAM MODEL")
        logger.info("=" * 70)
        if self.disable_semantic:
            logger.info("  ABLATION: Semantic stream DISABLED (using zeros)")
        if self.disable_structural:
            logger.info("  ABLATION: Structural stream DISABLED (using zeros)")
        if self.disable_gat:
            logger.info("  ABLATION: GAT DISABLED (using MLP)")
        if self.disable_cross_attention:
            logger.info("  ABLATION: Cross-attention DISABLED (using concatenation)")

        # Get hidden dimensions
        semantic_hidden = semantic_config.get('hidden_dim', 256)
        structural_hidden = structural_config.get('hidden_dim', 256)

        # Ensure both streams have same hidden_dim for fusion
        if semantic_hidden != structural_hidden:
            structural_hidden = semantic_hidden

        # Build semantic stream (or dummy)
        if self.disable_semantic:
            self.semantic_stream = None
            self.semantic_hidden = semantic_hidden
        else:
            self.semantic_stream = SemanticStream(
                input_dim=semantic_config.get('input_dim', 1536),
                hidden_dim=semantic_hidden,
                num_layers=semantic_config.get('num_layers', 2),
                dropout=semantic_config.get('dropout', 0.3),
                activation=semantic_config.get('activation', 'gelu')
            )
            self.semantic_hidden = semantic_hidden

        # Build structural stream (with ablation options)
        if self.disable_structural:
            self.structural_stream = None
            self.structural_hidden = structural_hidden
        elif self.disable_gat:
            # Use MLP instead of GAT
            self.structural_stream = StructuralStreamMLP(
                input_dim=structural_config.get('input_dim', 10),
                hidden_dim=structural_hidden,
                num_layers=structural_config.get('num_layers', 2),
                dropout=structural_config.get('dropout', 0.3),
                activation=structural_config.get('activation', 'gelu')
            )
            self.structural_hidden = structural_hidden
        else:
            # Normal GAT-based stream
            self.structural_stream = StructuralStreamV8(
                input_dim=structural_config.get('input_dim', 10),
                hidden_dim=structural_hidden,
                num_heads=structural_config.get('num_heads', 4),
                dropout=structural_config.get('dropout', 0.3),
                activation=structural_config.get('activation', 'elu'),
                use_edge_weights=structural_config.get('use_edge_weights', True)
            )
            self.structural_hidden = structural_hidden

        # Build fusion (with ablation options)
        if self.disable_cross_attention:
            # Use simple concatenation
            self.fusion = SimpleConcatFusion(
                hidden_dim=semantic_hidden,
                dropout=fusion_config.get('dropout', 0.1)
            )
        else:
            fusion_type = fusion_config.get('type', 'cross_attention')
            if fusion_type == 'gated':
                self.fusion = GatedFusionUnit(
                    hidden_dim=semantic_hidden,
                    dropout=fusion_config.get('dropout', 0.1),
                    use_projection=fusion_config.get('use_projection', True)
                )
            else:
                self.fusion = CrossAttentionFusion(
                    hidden_dim=semantic_hidden,
                    num_heads=fusion_config.get('num_heads', 4),
                    dropout=fusion_config.get('dropout', 0.1)
                )

        # Build classifier
        fusion_dim = semantic_hidden * 2
        self.classifier = SimpleClassifier(
            input_dim=fusion_dim,
            hidden_dims=classifier_config.get('hidden_dims', [128, 64]),
            num_classes=num_classes,
            dropout=classifier_config.get('dropout', 0.4)
        )

        logger.info(f"Semantic Stream: {'DISABLED' if self.disable_semantic else 'enabled'}")
        logger.info(f"Structural Stream: {'DISABLED' if self.disable_structural else ('MLP' if self.disable_gat else 'GAT')}")
        logger.info(f"Fusion: {'concat' if self.disable_cross_attention else 'cross-attention'}")
        logger.info("=" * 70)

    def forward(
        self,
        semantic_input: torch.Tensor,
        structural_input: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with ablation support.
        """
        batch_size = semantic_input.size(0)
        device = semantic_input.device

        # Semantic stream (or zeros if disabled)
        if self.disable_semantic:
            semantic_features = torch.zeros(
                batch_size, self.semantic_hidden, device=device
            )
        else:
            semantic_features = self.semantic_stream(semantic_input)

        # Structural stream (or zeros if disabled)
        if self.disable_structural:
            structural_features = torch.zeros(
                batch_size, self.structural_hidden, device=device
            )
        else:
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


def create_ablation_model(config: Dict) -> AblationDualStreamModel:
    """
    Factory function to create ablation-aware model from config.

    Checks for ablation flags in config and creates appropriate model.

    Args:
        config: Model configuration dictionary

    Returns:
        AblationDualStreamModel instance
    """
    # Check for ablation config
    ablation_config = config.get('ablation', {})

    # Also check for fusion type override (for A6 ablation)
    if config.get('fusion', {}).get('type') == 'concat':
        ablation_config['disable_cross_attention'] = True

    # Check for GNN type override (for A3 ablation)
    if config.get('gnn', {}).get('type') == 'none':
        ablation_config['disable_gat'] = True

    return AblationDualStreamModel(
        semantic_config=config.get('semantic', {}),
        structural_config=config.get('structural', {}),
        fusion_config=config.get('fusion', {}),
        classifier_config=config.get('classifier', {}),
        ablation_config=ablation_config,
        num_classes=config.get('num_classes', 2)
    )
