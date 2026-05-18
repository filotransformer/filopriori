"""
Dual-Head Model for Filo-Priori V9

Implements the DeepOrder-inspired dual-head architecture:
- Classification Head: Focal Loss for Fail/Pass prediction
- Regression Head: MSE Loss for priority score prediction

The key insight from DeepOrder is that test prioritization is fundamentally
a REGRESSION problem (predicting priority scores), not just classification.

Combined Loss:
    L_total = α × L_focal + β × L_mse

Where:
    - L_focal: Focal Loss for classification (handles class imbalance)
    - L_mse: MSE Loss for priority score regression
    - α, β: Balancing weights (default: α=1.0, β=0.5)

Author: Filo-Priori V9 Team
Date: December 2024
Reference: DeepOrder (ICSME 2021)
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
    GatedFusionUnit
)

logger = logging.getLogger(__name__)


class RegressionHead(nn.Module):
    """
    Regression head for predicting priority scores.

    Outputs a single value in [0, 1] representing the priority score.
    Higher scores indicate higher priority (more likely to fail).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [128, 64],
        dropout: float = 0.3
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

        # Output single value (priority score)
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())  # Constrain to [0, 1]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            priority_scores: [batch_size, 1] in range [0, 1]
        """
        return self.mlp(x)


class ClassificationHead(nn.Module):
    """
    Classification head for Fail/Pass prediction.
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


class DualHeadModel(nn.Module):
    """
    Dual-Head Model for Filo-Priori V9

    Combines classification (Fail/Pass) with regression (priority score)
    to leverage the DeepOrder insight that prioritization is regression.

    Architecture:
        Semantic Stream (SBERT embeddings) ─┐
                                            ├─→ Fusion ─→ Classification Head ─→ logits
        Structural Stream (GAT + history) ──┤           └→ Regression Head ─→ priority_score

    Training:
        L_total = α × L_focal(logits, labels) + β × L_mse(pred_priority, true_priority)

    Inference:
        - Use predicted priority scores for ranking
        - Higher priority score → earlier in test execution order
    """

    def __init__(
        self,
        semantic_config: Optional[Dict] = None,
        structural_config: Optional[Dict] = None,
        fusion_config: Optional[Dict] = None,
        classifier_config: Optional[Dict] = None,
        regressor_config: Optional[Dict] = None,
        num_classes: int = 2
    ):
        super().__init__()

        self.num_classes = num_classes

        # Default configs
        semantic_config = semantic_config or {}
        structural_config = structural_config or {}
        fusion_config = fusion_config or {}
        classifier_config = classifier_config or {}
        regressor_config = regressor_config or {}

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
        semantic_input_dim = semantic_config.get('input_dim', 1536)
        self.semantic_stream = SemanticStream(
            input_dim=semantic_input_dim,
            hidden_dim=semantic_hidden,
            num_layers=semantic_config.get('num_layers', 2),
            dropout=semantic_config.get('dropout', 0.3),
            activation=semantic_config.get('activation', 'gelu')
        )

        structural_input_dim = structural_config.get('input_dim', 19)  # 10 + 9 DeepOrder
        self.structural_stream = StructuralStreamV8(
            input_dim=structural_input_dim,
            hidden_dim=structural_hidden,
            num_heads=structural_config.get('num_heads', 4),
            dropout=structural_config.get('dropout', 0.3),
            activation=structural_config.get('activation', 'elu'),
            use_edge_weights=structural_config.get('use_edge_weights', True)
        )

        # Build fusion
        fusion_type = fusion_config.get('type', 'cross_attention')

        if fusion_type == 'gated':
            logger.info("Using GatedFusionUnit")
            self.fusion = GatedFusionUnit(
                hidden_dim=semantic_hidden,
                dropout=fusion_config.get('dropout', 0.1),
                use_projection=fusion_config.get('use_projection', True)
            )
        else:
            logger.info("Using CrossAttentionFusion")
            self.fusion = CrossAttentionFusion(
                hidden_dim=semantic_hidden,
                num_heads=fusion_config.get('num_heads', 4),
                dropout=fusion_config.get('dropout', 0.1)
            )

        # Fusion output dimension
        fusion_dim = semantic_hidden * 2

        # Classification head (Fail/Pass)
        self.classifier = ClassificationHead(
            input_dim=fusion_dim,
            hidden_dims=classifier_config.get('hidden_dims', [128, 64]),
            num_classes=num_classes,
            dropout=classifier_config.get('dropout', 0.4)
        )

        # Regression head (Priority Score)
        self.regressor = RegressionHead(
            input_dim=fusion_dim,
            hidden_dims=regressor_config.get('hidden_dims', [128, 64]),
            dropout=regressor_config.get('dropout', 0.3)
        )

        logger.info("=" * 70)
        logger.info("DUAL-HEAD MODEL INITIALIZED (DeepOrder-inspired)")
        logger.info("=" * 70)
        logger.info(f"Semantic Stream: [batch, {semantic_input_dim}] → [batch, {semantic_hidden}]")
        logger.info(f"Structural Stream: [batch, {structural_input_dim}] → [batch, {structural_hidden}]")
        logger.info(f"Fusion: [batch, {fusion_dim}]")
        logger.info(f"Classification Head: [batch, {fusion_dim}] → [batch, {num_classes}]")
        logger.info(f"Regression Head: [batch, {fusion_dim}] → [batch, 1] (priority score)")
        logger.info("=" * 70)

    def forward(
        self,
        semantic_input: torch.Tensor,
        structural_input: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            semantic_input: Text embeddings [batch_size, embedding_dim]
            structural_input: Structural features [batch_size, 19]
            edge_index: Graph connectivity [2, E]
            edge_weights: Optional edge weights [E]

        Returns:
            logits: [batch_size, num_classes] for classification
            priority_scores: [batch_size, 1] for regression
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

        # Dual heads
        logits = self.classifier(fused_features)
        priority_scores = self.regressor(fused_features)

        return logits, priority_scores

    def get_feature_representations(
        self,
        semantic_input: torch.Tensor,
        structural_input: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get intermediate feature representations (useful for analysis)

        Returns:
            semantic_features, structural_features, fused_features
        """
        semantic_features = self.semantic_stream(semantic_input)
        structural_features = self.structural_stream(
            structural_input,
            edge_index,
            edge_weights
        )
        fused_features = self.fusion(semantic_features, structural_features)

        return semantic_features, structural_features, fused_features


class DualHeadLoss(nn.Module):
    """
    Combined loss for dual-head model.

    L_total = α × L_focal + β × L_mse_weighted

    Where:
        - L_focal: Focal Loss for classification
        - L_mse_weighted: Weighted MSE Loss for priority score regression
          (samples with priority_score > 0 get higher weight)
        - α: Classification loss weight (default: 1.0)
        - β: Regression loss weight (default: 0.5)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        mse_nonzero_weight: float = 10.0
    ):
        super().__init__()

        self.alpha = alpha  # Classification weight
        self.beta = beta    # Regression weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.mse_nonzero_weight = mse_nonzero_weight

        # Store class weights
        self.register_buffer('class_weights', class_weights)

        logger.info(f"DualHeadLoss initialized:")
        logger.info(f"  α (classification): {alpha}")
        logger.info(f"  β (regression): {beta}")
        logger.info(f"  Focal: alpha={focal_alpha}, gamma={focal_gamma}")
        logger.info(f"  MSE nonzero weight: {mse_nonzero_weight}x")

    def focal_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal Loss.

        FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

        NOTE: We do NOT use class_weights here to avoid double-counting
        class imbalance. Focal Loss has its own alpha balancing mechanism.
        """
        # Standard cross-entropy WITHOUT class_weights (focal_alpha handles imbalance)
        ce_loss = F.cross_entropy(
            logits, targets,
            reduction='none'
        )

        # Get probabilities
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal modulation: down-weight easy examples
        focal_weight = (1 - pt) ** self.focal_gamma

        # Apply alpha balancing for class imbalance
        # focal_alpha=0.75 means: Fail (class 0) gets 0.75, Pass (class 1) gets 0.25
        if self.focal_alpha is not None:
            alpha_t = torch.where(
                targets == 0,  # Fail class (minority)
                torch.tensor(self.focal_alpha, device=logits.device),
                torch.tensor(1 - self.focal_alpha, device=logits.device)  # Pass class (majority)
            )
            focal_weight = alpha_t * focal_weight

        focal_loss = focal_weight * ce_loss
        return focal_loss.mean()

    def weighted_mse_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss.

        Samples with target > 0 get higher weight to prevent the model from
        predicting 0 for everything (since most targets are 0).

        This is critical because:
        - Most priority_score targets are 0 (TCs that never failed)
        - Standard MSE would just learn to predict 0 for everything
        - Weighted MSE forces the model to learn meaningful predictions for
          the non-zero (important) samples

        Args:
            pred: Predicted values [batch]
            target: Target values [batch]

        Returns:
            Weighted mean squared error
        """
        # Compute per-sample MSE
        mse_per_sample = (pred - target) ** 2

        # Create weights: nonzero targets get higher weight
        weights = torch.where(
            target > 0,
            torch.tensor(self.mse_nonzero_weight, device=pred.device),
            torch.tensor(1.0, device=pred.device)
        )

        # Weighted mean
        weighted_mse = (weights * mse_per_sample).sum() / weights.sum()

        return weighted_mse

    def forward(
        self,
        logits: torch.Tensor,
        priority_pred: torch.Tensor,
        labels: torch.Tensor,
        priority_true: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            logits: Classification logits [batch, num_classes]
            priority_pred: Predicted priority scores [batch, 1]
            labels: True labels [batch]
            priority_true: True priority scores [batch]

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual losses
        """
        # Classification loss (Focal)
        loss_focal = self.focal_loss(logits, labels)

        # Regression loss (Weighted MSE)
        priority_pred_flat = priority_pred.squeeze(-1)
        loss_mse = self.weighted_mse_loss(priority_pred_flat, priority_true)

        # Combined loss
        total_loss = self.alpha * loss_focal + self.beta * loss_mse

        loss_dict = {
            'total': total_loss.item(),
            'focal': loss_focal.item(),
            'mse': loss_mse.item()
        }

        return total_loss, loss_dict


def create_dual_head_model(config: Dict) -> DualHeadModel:
    """
    Factory function to create DualHeadModel from config.

    Args:
        config: Configuration dictionary

    Returns:
        DualHeadModel instance
    """
    model_config = config.get('model', {})

    return DualHeadModel(
        semantic_config=model_config.get('semantic', {}),
        structural_config=model_config.get('structural', {}),
        fusion_config=model_config.get('fusion', {}),
        classifier_config=model_config.get('classifier', {}),
        regressor_config=model_config.get('regressor', {
            'hidden_dims': [128, 64],
            'dropout': 0.3
        }),
        num_classes=model_config.get('num_classes', 2)
    )


def create_dual_head_loss(
    config: Dict,
    class_weights: Optional[torch.Tensor] = None
) -> DualHeadLoss:
    """
    Factory function to create DualHeadLoss from config.

    Args:
        config: Configuration dictionary
        class_weights: Optional class weights tensor

    Returns:
        DualHeadLoss instance
    """
    loss_config = config.get('training', {}).get('loss', {})
    dual_head_config = loss_config.get('dual_head', {})

    return DualHeadLoss(
        alpha=dual_head_config.get('alpha', 1.0),
        beta=dual_head_config.get('beta', 0.5),
        focal_alpha=loss_config.get('focal_alpha', 0.75),
        focal_gamma=loss_config.get('focal_gamma', 2.0),
        class_weights=class_weights,
        mse_nonzero_weight=dual_head_config.get('mse_nonzero_weight', 10.0)
    )


__all__ = [
    'RegressionHead',
    'ClassificationHead',
    'DualHeadModel',
    'DualHeadLoss',
    'create_dual_head_model',
    'create_dual_head_loss'
]
