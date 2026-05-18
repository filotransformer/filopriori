"""
Field Fusion Module
Implements multiple strategies for fusing multi-field embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ProjectionConcatFusion(nn.Module):
    """
    Projects each field embedding to lower dimension, then concatenates

    Example:
        4 fields × 768 dims → 4 × 256 dims → 1024 dims (concat)
    """

    def __init__(self, input_dim: int, num_fields: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_fields = num_fields
        self.output_dim = output_dim

        # Calculate projection dimension per field
        self.proj_dim = output_dim // num_fields

        # Create projection layers for each field
        self.projections = nn.ModuleList([
            nn.Linear(input_dim, self.proj_dim) for _ in range(num_fields)
        ])

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(output_dim)

        logger.info(f"ProjectionConcat: {num_fields} × {input_dim}d → {num_fields} × {self.proj_dim}d → {output_dim}d")

    def forward(self, field_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            field_embeddings: List of [batch_size, input_dim] tensors

        Returns:
            Fused embedding [batch_size, output_dim]
        """
        # Project each field
        projected = []
        for i, emb in enumerate(field_embeddings):
            proj = self.projections[i](emb)
            projected.append(proj)

        # Concatenate
        fused = torch.cat(projected, dim=-1)

        # Layer norm
        fused = self.layer_norm(fused)

        return fused


class WeightedAverageFusion(nn.Module):
    """
    Weighted average with learnable weights

    Example:
        w = softmax([w1, w2, w3, w4])
        output = w1*emb1 + w2*emb2 + w3*emb3 + w4*emb4
    """

    def __init__(self, input_dim: int, num_fields: int, output_dim: int,
                 initial_weights: List[float] = None):
        super().__init__()
        self.input_dim = input_dim
        self.num_fields = num_fields
        self.output_dim = output_dim

        # Initialize weights
        if initial_weights is None:
            initial_weights = [1.0 / num_fields] * num_fields

        # Learnable weights (will be softmaxed)
        self.weight_logits = nn.Parameter(
            torch.tensor(initial_weights, dtype=torch.float32)
        )

        # Optional projection if input_dim != output_dim
        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            self.projection = nn.Identity()

        self.layer_norm = nn.LayerNorm(output_dim)

        logger.info(f"WeightedAverage: {num_fields} × {input_dim}d → {output_dim}d")
        logger.info(f"Initial weights: {initial_weights}")

    def forward(self, field_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            field_embeddings: List of [batch_size, input_dim] tensors

        Returns:
            Fused embedding [batch_size, output_dim]
        """
        # Get normalized weights
        weights = F.softmax(self.weight_logits, dim=0)

        # Weighted sum
        fused = sum(w * emb for w, emb in zip(weights, field_embeddings))

        # Project if needed
        fused = self.projection(fused)

        # Layer norm
        fused = self.layer_norm(fused)

        return fused

    def get_field_weights(self) -> Dict[str, float]:
        """Get current field weights (for logging/analysis)"""
        weights = F.softmax(self.weight_logits, dim=0)
        return weights.detach().cpu().tolist()


class AttentionFusion(nn.Module):
    """
    Multi-head attention based fusion
    Each field embedding is a token, apply self-attention
    """

    def __init__(self, input_dim: int, num_fields: int, output_dim: int,
                 num_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_fields = num_fields
        self.output_dim = output_dim
        self.num_heads = num_heads

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # Projection to output dim
        self.projection = nn.Linear(input_dim, output_dim)

        self.layer_norm = nn.LayerNorm(output_dim)

        logger.info(f"AttentionFusion: {num_fields} × {input_dim}d → {output_dim}d (heads={num_heads})")

    def forward(self, field_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            field_embeddings: List of [batch_size, input_dim] tensors

        Returns:
            Fused embedding [batch_size, output_dim]
        """
        # Stack embeddings: [batch_size, num_fields, input_dim]
        stacked = torch.stack(field_embeddings, dim=1)

        # Self-attention
        attended, _ = self.attention(stacked, stacked, stacked)

        # Pool (mean over fields)
        pooled = attended.mean(dim=1)  # [batch_size, input_dim]

        # Project
        fused = self.projection(pooled)

        # Layer norm
        fused = self.layer_norm(fused)

        return fused


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion: groups fields hierarchically

    Level 1: Test info (summary + steps)
    Level 2: Change info (commits + CR)
    Level 3: Final fusion
    """

    def __init__(self, input_dim: int, num_fields: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_fields = num_fields
        self.output_dim = output_dim

        # Assume 4 fields: summary, steps, commits, CR
        assert num_fields == 4, "Hierarchical fusion requires exactly 4 fields"

        # Level 1: Fuse summary + steps → test_info
        self.test_fusion = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU()
        )

        # Level 2: Fuse commits + CR → change_info
        self.change_fusion = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU()
        )

        # Level 3: Fuse test_info + change_info → final
        self.final_fusion = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

        logger.info(f"HierarchicalFusion: {num_fields} × {input_dim}d → {output_dim}d")
        logger.info("  Level 1: summary + steps → test_info")
        logger.info("  Level 2: commits + CR → change_info")
        logger.info("  Level 3: test_info + change_info → final")

    def forward(self, field_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            field_embeddings: [summary, steps, commits, CR] each [batch_size, input_dim]

        Returns:
            Fused embedding [batch_size, output_dim]
        """
        summary, steps, commits, cr = field_embeddings

        # Level 1: Test info
        test_concat = torch.cat([summary, steps], dim=-1)
        test_info = self.test_fusion(test_concat)

        # Level 2: Change info
        change_concat = torch.cat([commits, cr], dim=-1)
        change_info = self.change_fusion(change_concat)

        # Level 3: Final
        final_concat = torch.cat([test_info, change_info], dim=-1)
        fused = self.final_fusion(final_concat)

        return fused


class FieldFusionFactory:
    """Factory for creating field fusion modules"""

    @staticmethod
    def create(
        strategy: str,
        input_dim: int,
        num_fields: int,
        output_dim: int,
        **kwargs
    ) -> nn.Module:
        """
        Create field fusion module

        Args:
            strategy: One of ["projection_concat", "weighted", "attention", "hierarchical"]
            input_dim: Dimension of each field embedding
            num_fields: Number of fields
            output_dim: Desired output dimension
            **kwargs: Additional arguments for specific strategies

        Returns:
            Field fusion module
        """
        if strategy == "projection_concat":
            return ProjectionConcatFusion(input_dim, num_fields, output_dim)

        elif strategy == "weighted":
            initial_weights = kwargs.get('initial_weights', None)
            return WeightedAverageFusion(input_dim, num_fields, output_dim, initial_weights)

        elif strategy == "attention":
            num_heads = kwargs.get('num_heads', 4)
            return AttentionFusion(input_dim, num_fields, output_dim, num_heads)

        elif strategy == "hierarchical":
            return HierarchicalFusion(input_dim, num_fields, output_dim)

        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")


# Convenience function
def create_field_fusion(config: Dict) -> nn.Module:
    """
    Create field fusion module from config

    Args:
        config: Embedding config dict

    Returns:
        Field fusion module
    """
    strategy = config.get('fusion_strategy', 'projection_concat')
    input_dim = config.get('field_embedding_dim', 768)
    output_dim = config.get('embedding_dim', 1024)

    # Count fields
    fields = config.get('fields', [])
    num_fields = len(fields)

    # Get initial weights if specified
    initial_weights = [f.get('weight', 1.0) for f in fields]

    logger.info(f"Creating field fusion: {strategy}")
    logger.info(f"  Input: {num_fields} fields × {input_dim}d")
    logger.info(f"  Output: {output_dim}d")

    return FieldFusionFactory.create(
        strategy=strategy,
        input_dim=input_dim,
        num_fields=num_fields,
        output_dim=output_dim,
        initial_weights=initial_weights,
        num_heads=config.get('fusion_num_heads', 4)
    )
