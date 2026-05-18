"""
Loss Functions
Implements Focal Loss and other loss functions for handling class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance

    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002

    Enhanced to support per-class weights (alpha can be a tensor)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Args:
            alpha: Weighting factor. Can be:
                   - None: no weighting
                   - float: single weight for all classes
                   - Tensor: per-class weights [num_classes]
            gamma: Exponent of the modulating factor (1 - p_t)^gamma
            reduction: 'mean', 'sum', or 'none'
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

        # Handle alpha as float or tensor
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            self.alpha = float(alpha)
        elif isinstance(alpha, list):
            self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))
        else:
            self.register_buffer('alpha', alpha)

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Loss value
        """
        # Get probabilities
        p = F.softmax(inputs, dim=-1)

        # Get class probabilities
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction='none',
            label_smoothing=self.label_smoothing
        )

        # Get probability of true class
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, float):
                # Single alpha value for all classes
                alpha_t = self.alpha
            else:
                # Per-class alpha values
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss - Combines class weights with focal loss

    This is the strongest loss for extreme class imbalance:
    1. Applies class weights to CE loss (rebalances classes)
    2. Applies focal modulation (focuses on hard examples)
    3. Applies alpha weighting (additional class-specific weight)

    Recommended for imbalance ratio > 20:1
    """

    def __init__(
        self,
        alpha: float = 0.75,
        gamma: float = 3.0,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Args:
            alpha: Focal loss alpha (higher = more weight to minority)
            gamma: Focal loss gamma (higher = more focus on hard examples)
            class_weights: Per-class weights [num_classes]
            reduction: 'mean', 'sum', or 'none'
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Loss value
        """
        # Step 1: Compute weighted cross-entropy (applies class weights)
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.class_weights,
            reduction='none',
            label_smoothing=self.label_smoothing
        )

        # Step 2: Get probabilities and focal modulation
        p = F.softmax(inputs, dim=-1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma

        # Step 3: Apply focal loss with alpha
        loss = self.alpha * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross-Entropy Loss for class imbalance"""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            inputs: Predicted logits [batch_size, num_classes]
            targets: Ground truth labels [batch_size]

        Returns:
            Loss value
        """
        return F.cross_entropy(
            inputs,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing
        )


def create_loss_function(config: dict, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """
    Create loss function based on configuration

    Args:
        config: Configuration dictionary
        class_weights: Optional class weights tensor (auto-computed)

    Returns:
        Loss function module
    """
    loss_config = config['training']['loss']
    loss_type = loss_config['type']

    if loss_type == 'weighted_focal':
        # STRONGEST LOSS - Combines class weights + focal loss + alpha
        # Recommended for extreme imbalance (>20:1)
        # BUT: if use_class_weights=False, don't apply class weights
        use_class_weights = loss_config.get('use_class_weights', True)

        if use_class_weights and class_weights is not None:
            class_weights = class_weights.to(config['hardware']['device'])
        else:
            class_weights = None  # Disable class weights if use_class_weights=False

        return WeightedFocalLoss(
            alpha=loss_config.get('focal_alpha', 0.75),
            gamma=loss_config.get('focal_gamma', 3.0),
            class_weights=class_weights,
            label_smoothing=loss_config.get('label_smoothing', 0.0)
        )

    elif loss_type == 'focal':
        # Support both single alpha value and per-class weights
        focal_alpha = loss_config.get('focal_alpha', 0.25)
        use_class_weights = loss_config.get('use_class_weights', False)

        if use_class_weights and class_weights is not None:
            # Use per-class weights as alpha
            alpha = class_weights.to(config['hardware']['device'])
        elif isinstance(focal_alpha, list):
            # Alpha specified as list in config
            alpha = torch.tensor(focal_alpha, dtype=torch.float32).to(config['hardware']['device'])
        else:
            # Single alpha value
            alpha = focal_alpha

        return FocalLoss(
            alpha=alpha,
            gamma=loss_config.get('focal_gamma', 2.0),
            label_smoothing=loss_config.get('label_smoothing', 0.0)
        )

    elif loss_type == 'weighted_ce':
        if class_weights is not None:
            class_weights = class_weights.to(config['hardware']['device'])
        return WeightedCrossEntropyLoss(
            class_weights=class_weights,
            label_smoothing=loss_config.get('label_smoothing', 0.0)
        )

    elif loss_type == 'ce':
        return nn.CrossEntropyLoss(
            label_smoothing=loss_config.get('label_smoothing', 0.0)
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
