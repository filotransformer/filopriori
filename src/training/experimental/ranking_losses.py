"""
Learning-to-Rank Loss Functions for Test Case Prioritization.

This module implements several L2R loss functions suitable for TCP:

1. ListNet: Listwise approach using cross-entropy on top-1 probabilities
2. ListMLE: Maximum likelihood estimation for permutation probability
3. LambdaRank: Pairwise approach with NDCG-aware gradients
4. ApproxNDCG: Differentiable approximation of NDCG

Reference papers:
- ListNet: Cao et al., "Learning to Rank: From Pairwise Approach to Listwise Approach", ICML 2007
- ListMLE: Xia et al., "Listwise Approach to Learning to Rank", ICML 2008
- LambdaRank: Burges et al., "Learning to Rank using Gradient Descent", ICML 2005
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ListNetLoss(nn.Module):
    """
    ListNet Loss: Cross-entropy on top-1 probabilities.

    The loss compares the probability distribution induced by predicted
    scores with the distribution induced by true relevance labels.

    For TCP: relevance = 1 for failing tests, 0 for passing tests.

    Loss = -sum(P_y(j) * log(P_s(j)))
    where P_y(j) = exp(y_j) / sum(exp(y_i)) is the "true" distribution
    and P_s(j) = exp(s_j) / sum(exp(s_i)) is the predicted distribution
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-10):
        """
        Args:
            temperature: Temperature for softmax (higher = softer distribution)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute ListNet loss.

        Args:
            scores: Predicted scores [batch_size, list_size] or [list_size]
            relevance: True relevance labels [batch_size, list_size] or [list_size]
            mask: Optional mask for padding [batch_size, list_size]

        Returns:
            Loss value (scalar)
        """
        # Handle 1D input (single list)
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            relevance = relevance.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        # Apply mask if provided
        if mask is not None:
            # Set masked positions to very negative value
            scores = scores.masked_fill(~mask, float('-inf'))
            relevance = relevance.masked_fill(~mask, float('-inf'))

        # Compute softmax distributions
        p_true = F.softmax(relevance / self.temperature, dim=-1)
        p_pred = F.softmax(scores / self.temperature, dim=-1)

        # Cross-entropy loss
        loss = -torch.sum(p_true * torch.log(p_pred + self.eps), dim=-1)

        return loss.mean()


class ListMLELoss(nn.Module):
    """
    ListMLE Loss: Maximum likelihood estimation for ranking.

    Models the probability of observing the true ranking as a product
    of conditional probabilities (Plackett-Luce model).

    Loss = -log P(π* | s) where π* is the optimal permutation
    """

    def __init__(self, eps: float = 1e-10):
        """
        Args:
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps

    def forward(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute ListMLE loss.

        Args:
            scores: Predicted scores [batch_size, list_size]
            relevance: True relevance labels [batch_size, list_size]
            mask: Optional mask for padding [batch_size, list_size]

        Returns:
            Loss value (scalar)
        """
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            relevance = relevance.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        batch_size, list_size = scores.shape
        device = scores.device

        # Sort by relevance to get optimal permutation
        _, perm = relevance.sort(dim=-1, descending=True)

        # Reorder scores according to optimal permutation
        scores_sorted = torch.gather(scores, dim=-1, index=perm)

        # Apply mask
        if mask is not None:
            mask_sorted = torch.gather(mask, dim=-1, index=perm)
            scores_sorted = scores_sorted.masked_fill(~mask_sorted, float('-inf'))

        # Compute log-likelihood using Plackett-Luce
        # log P(π) = sum_i [s_π(i) - log(sum_j>=i exp(s_π(j)))]

        # Compute cumulative logsumexp from the end
        # This gives log(sum_j>=i exp(s_j)) for each position i
        max_score = scores_sorted.max(dim=-1, keepdim=True)[0]
        scores_shifted = scores_sorted - max_score

        # Reverse cumsum of exp
        exp_scores = torch.exp(scores_shifted)
        cumsum_exp = torch.flip(
            torch.cumsum(torch.flip(exp_scores, dims=[-1]), dim=-1),
            dims=[-1]
        )

        log_likelihood = scores_sorted - (torch.log(cumsum_exp + self.eps) + max_score)

        # Sum over list positions
        if mask is not None:
            log_likelihood = log_likelihood.masked_fill(~mask_sorted, 0)

        loss = -log_likelihood.sum(dim=-1)

        return loss.mean()


class LambdaRankLoss(nn.Module):
    """
    LambdaRank Loss: Pairwise loss weighted by NDCG delta.

    For each pair (i, j) where relevance[i] > relevance[j]:
    Loss += |delta_NDCG| * log(1 + exp(s_j - s_i))

    The delta_NDCG term ensures the loss focuses on swaps that
    matter most for the ranking metric.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        k: Optional[int] = None,
        eps: float = 1e-10
    ):
        """
        Args:
            sigma: Scaling factor for score differences
            k: Cutoff for NDCG@k (None = full list)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.sigma = sigma
        self.k = k
        self.eps = eps

    def _dcg_gain(self, relevance: torch.Tensor) -> torch.Tensor:
        """Compute DCG gain: 2^rel - 1"""
        return torch.pow(2.0, relevance) - 1.0

    def _dcg_discount(self, rank: torch.Tensor) -> torch.Tensor:
        """Compute DCG discount: 1 / log2(rank + 1)"""
        return 1.0 / torch.log2(rank.float() + 2.0)

    def forward(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute LambdaRank loss.

        Args:
            scores: Predicted scores [batch_size, list_size]
            relevance: True relevance labels [batch_size, list_size]
            mask: Optional mask for padding [batch_size, list_size]

        Returns:
            Loss value (scalar)
        """
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            relevance = relevance.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        batch_size, list_size = scores.shape
        device = scores.device

        # Get current ranking (by predicted scores)
        _, pred_ranks = scores.sort(dim=-1, descending=True)
        ranks = torch.zeros_like(pred_ranks)
        ranks.scatter_(1, pred_ranks, torch.arange(list_size, device=device).expand(batch_size, -1))
        ranks = ranks + 1  # 1-indexed

        # Compute ideal DCG
        sorted_rel, _ = relevance.sort(dim=-1, descending=True)
        if self.k is not None:
            k = min(self.k, list_size)
            sorted_rel = sorted_rel[:, :k]

        ideal_dcg = (self._dcg_gain(sorted_rel) *
                    self._dcg_discount(torch.arange(sorted_rel.size(1), device=device))).sum(dim=-1)
        ideal_dcg = ideal_dcg.clamp(min=self.eps)

        # Create pairwise comparisons
        # rel_diff[i,j] = relevance[i] - relevance[j]
        rel_diff = relevance.unsqueeze(-1) - relevance.unsqueeze(-2)
        score_diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)

        # Only consider pairs where i should be ranked higher than j
        valid_pairs = (rel_diff > 0).float()

        # Apply mask
        if mask is not None:
            pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            valid_pairs = valid_pairs * pair_mask.float()

        # Compute delta NDCG for each swap
        gain_i = self._dcg_gain(relevance).unsqueeze(-1)
        gain_j = self._dcg_gain(relevance).unsqueeze(-2)

        disc_i = self._dcg_discount(ranks).unsqueeze(-1)
        disc_j = self._dcg_discount(ranks).unsqueeze(-2)

        # Delta if we swap i and j
        delta_ndcg = torch.abs(
            (gain_i - gain_j) * (disc_i - disc_j)
        ) / ideal_dcg.unsqueeze(-1).unsqueeze(-1)

        # Pairwise loss
        pairwise_loss = torch.log1p(torch.exp(-self.sigma * score_diff))

        # Weight by delta NDCG
        weighted_loss = delta_ndcg * pairwise_loss * valid_pairs

        # Sum over pairs
        loss = weighted_loss.sum(dim=(-1, -2)) / (valid_pairs.sum(dim=(-1, -2)) + self.eps)

        return loss.mean()


class ApproxNDCGLoss(nn.Module):
    """
    Approximate NDCG Loss: Differentiable approximation of 1-NDCG.

    Uses a soft ranking function to make NDCG differentiable.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        k: Optional[int] = None,
        eps: float = 1e-10
    ):
        """
        Args:
            temperature: Temperature for soft ranking
            k: Cutoff for NDCG@k (None = full list)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.temperature = temperature
        self.k = k
        self.eps = eps

    def _soft_rank(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Compute soft ranks using sigmoid approximation.

        Rank of item i ≈ 1 + sum_j sigmoid((s_j - s_i) / temperature)
        """
        batch_size, list_size = scores.shape

        # Pairwise differences
        diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)

        # Soft comparison
        comparison = torch.sigmoid(diff / self.temperature)

        # Sum to get soft rank (higher score = lower rank)
        soft_ranks = comparison.sum(dim=-1)

        return soft_ranks

    def forward(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute approximate NDCG loss.

        Args:
            scores: Predicted scores [batch_size, list_size]
            relevance: True relevance labels [batch_size, list_size]
            mask: Optional mask for padding [batch_size, list_size]

        Returns:
            Loss value (scalar): 1 - approx_NDCG
        """
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            relevance = relevance.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        batch_size, list_size = scores.shape
        device = scores.device

        # Compute soft ranks
        soft_ranks = self._soft_rank(scores)

        # DCG computation
        gains = torch.pow(2.0, relevance) - 1.0
        discounts = 1.0 / torch.log2(soft_ranks + 2.0)

        if mask is not None:
            gains = gains * mask.float()
            discounts = discounts * mask.float()

        dcg = (gains * discounts).sum(dim=-1)

        # Ideal DCG (using true relevance order)
        sorted_rel, _ = relevance.sort(dim=-1, descending=True)
        if self.k is not None:
            k = min(self.k, list_size)
            sorted_rel = sorted_rel[:, :k]

        ideal_gains = torch.pow(2.0, sorted_rel) - 1.0
        ideal_discounts = 1.0 / torch.log2(
            torch.arange(sorted_rel.size(1), device=device).float() + 2.0
        )

        ideal_dcg = (ideal_gains * ideal_discounts).sum(dim=-1)

        # NDCG
        ndcg = dcg / (ideal_dcg + self.eps)

        # Loss = 1 - NDCG
        loss = 1.0 - ndcg

        return loss.mean()


class RankingMSELoss(nn.Module):
    """
    Pointwise MSE loss for ranking.

    Simpler approach: directly regress to relevance scores.
    Can work well for TCP where we care about binary relevance (fail/pass).
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute MSE loss between scores and relevance.

        Args:
            scores: Predicted scores [batch_size, list_size]
            relevance: True relevance labels [batch_size, list_size]
            mask: Optional mask for padding

        Returns:
            MSE loss
        """
        if mask is not None:
            scores = scores[mask]
            relevance = relevance[mask]

        return self.mse(scores, relevance.float())


class APFDWeightedPairwiseLoss(nn.Module):
    """
    APFD-Impact Weighted Pairwise Loss (Proposal #3).

    This loss weights each failure-pass pair by its actual impact on the APFD score,
    focusing gradient updates on pairs that matter most for the ranking metric.

    Key insight: In APFD calculation, detecting a failure earlier has more impact
    than detecting it later. This loss uses position-weighted importance to
    prioritize correctly ordering failures at the top of the ranking.

    Loss = sum_i sum_j w_ij * max(0, margin + s_j - s_i)
    where i=failure, j=pass, and w_ij is the APFD-impact weight.
    """

    def __init__(
        self,
        margin: float = 0.5,
        temperature: float = 1.0,
        position_decay: str = 'linear',
        min_pairs_for_loss: int = 1,
        use_soft_margin: bool = True,
        normalize_weights: bool = True,
        eps: float = 1e-10
    ):
        """
        Args:
            margin: Margin for hinge loss (soft or hard)
            temperature: Temperature for soft operations
            position_decay: How to decay weight by position ('linear', 'exponential', 'logarithmic')
            min_pairs_for_loss: Minimum number of pairs to compute loss
            use_soft_margin: Use softplus instead of ReLU for smooth gradients
            normalize_weights: Whether to normalize weights for stability
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.position_decay = position_decay
        self.min_pairs_for_loss = min_pairs_for_loss
        self.use_soft_margin = use_soft_margin
        self.normalize_weights = normalize_weights
        self.eps = eps

    def _compute_position_weight(
        self,
        fail_rank: torch.Tensor,
        total_failures: int,
        n_tests: int
    ) -> torch.Tensor:
        """
        Compute APFD-based position weight.

        The weight reflects how much this failure's position impacts APFD.
        Earlier positions have higher weight.

        Args:
            fail_rank: Rank of the failure (0-indexed)
            total_failures: Total number of failures in the build
            n_tests: Total number of tests

        Returns:
            Position weight
        """
        # Normalize position to [0, 1]
        normalized_pos = fail_rank.float() / max(n_tests - 1, 1)

        if self.position_decay == 'linear':
            # Linear decay: weight = 1 - pos/n
            weight = 1.0 - normalized_pos
        elif self.position_decay == 'exponential':
            # Exponential decay: weight = exp(-pos/n)
            weight = torch.exp(-2.0 * normalized_pos)
        elif self.position_decay == 'logarithmic':
            # Logarithmic decay: weight = 1/log2(pos+2)
            weight = 1.0 / torch.log2(fail_rank.float() + 2.0)
        else:
            weight = torch.ones_like(fail_rank.float())

        return weight

    def forward(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_stats: bool = False
    ) -> torch.Tensor:
        """
        Compute APFD-weighted pairwise loss.

        Args:
            scores: Predicted scores [batch_size, list_size] or [list_size]
            relevance: Binary relevance (1=failure, 0=pass) [batch_size, list_size] or [list_size]
            mask: Optional mask for padding
            return_stats: If True, return loss statistics

        Returns:
            Loss value (scalar)
        """
        # Handle 1D input
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            relevance = relevance.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        batch_size, list_size = scores.shape
        device = scores.device

        total_loss = torch.tensor(0.0, device=device)
        valid_batches = 0
        total_pairs = 0

        for b in range(batch_size):
            batch_scores = scores[b]
            batch_relevance = relevance[b]

            if mask is not None:
                batch_mask = mask[b]
                batch_scores = batch_scores[batch_mask]
                batch_relevance = batch_relevance[batch_mask]

            # Get failure and pass indices
            fail_idx = (batch_relevance > 0.5).nonzero(as_tuple=True)[0]
            pass_idx = (batch_relevance <= 0.5).nonzero(as_tuple=True)[0]

            n_failures = len(fail_idx)
            n_passes = len(pass_idx)

            # Skip if not enough pairs
            if n_failures < 1 or n_passes < 1:
                continue

            n_pairs = n_failures * n_passes
            if n_pairs < self.min_pairs_for_loss:
                continue

            # Get scores for failures and passes
            fail_scores = batch_scores[fail_idx]
            pass_scores = batch_scores[pass_idx]

            # Compute pairwise differences: fail_score - pass_score
            # Shape: [n_failures, n_passes]
            score_diff = fail_scores.unsqueeze(1) - pass_scores.unsqueeze(0)

            # Compute position weights for each failure
            # Use predicted ranking to determine positions
            _, pred_order = batch_scores.sort(descending=True)
            pred_ranks = torch.zeros_like(pred_order)
            pred_ranks[pred_order] = torch.arange(len(batch_scores), device=device)

            fail_ranks = pred_ranks[fail_idx]
            position_weights = self._compute_position_weight(
                fail_ranks, n_failures, len(batch_scores)
            )

            # Expand weights to pair matrix
            pair_weights = position_weights.unsqueeze(1).expand(-1, n_passes)

            if self.normalize_weights:
                pair_weights = pair_weights / (pair_weights.sum() + self.eps)

            # Compute hinge loss: max(0, margin - score_diff)
            # We want failures to have higher scores than passes
            if self.use_soft_margin:
                # Softplus for smooth gradients
                pair_loss = F.softplus(self.margin - score_diff) / self.temperature
            else:
                # Hard margin
                pair_loss = F.relu(self.margin - score_diff)

            # Weighted loss
            weighted_loss = (pair_weights * pair_loss).sum()

            total_loss = total_loss + weighted_loss
            valid_batches += 1
            total_pairs += n_pairs

        if valid_batches == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        final_loss = total_loss / valid_batches

        if return_stats:
            return final_loss, {'pairs': total_pairs, 'batches': valid_batches}

        return final_loss


class SoftAPFDLoss(nn.Module):
    """
    Differentiable approximation of APFD loss.

    APFD = 1 - (sum of failure positions) / (n * m) + 1/(2n)

    This loss uses a soft ranking function to make the position computation
    differentiable.
    """

    def __init__(self, temperature: float = 1.0, eps: float = 1e-10):
        """
        Args:
            temperature: Temperature for soft ranking (lower = sharper)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def _soft_rank(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable soft ranks.

        Rank of item i ≈ 1 + sum_j sigmoid((s_j - s_i) / temperature)
        Higher score = lower (better) rank
        """
        n = scores.shape[-1]
        # Pairwise differences: s_j - s_i
        diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)
        # Soft comparison
        comparison = torch.sigmoid(diff / self.temperature)
        # Sum to get rank (1-indexed)
        soft_ranks = 1.0 + comparison.sum(dim=-1)
        return soft_ranks

    def forward(
        self,
        scores: torch.Tensor,
        relevance: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute soft APFD loss (1 - soft_APFD).

        Args:
            scores: Predicted scores [batch_size, list_size] or [list_size]
            relevance: Binary relevance (1=failure, 0=pass)
            mask: Optional mask for padding

        Returns:
            Loss = 1 - soft_APFD
        """
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            relevance = relevance.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        batch_size, list_size = scores.shape
        device = scores.device

        # Compute soft ranks
        soft_ranks = self._soft_rank(scores)

        # Compute soft APFD for each batch
        losses = []
        for b in range(batch_size):
            batch_ranks = soft_ranks[b]
            batch_relevance = relevance[b]

            if mask is not None:
                batch_mask = mask[b]
                batch_ranks = batch_ranks[batch_mask]
                batch_relevance = batch_relevance[batch_mask]

            n_tests = len(batch_ranks)
            n_failures = batch_relevance.sum()

            if n_failures < 1:
                continue

            # Sum of failure positions (soft)
            failure_positions_sum = (batch_ranks * batch_relevance).sum()

            # Soft APFD
            soft_apfd = 1.0 - failure_positions_sum / (n_tests * n_failures + self.eps) + 1.0 / (2 * n_tests)

            # Loss = 1 - APFD (we want to maximize APFD)
            losses.append(1.0 - soft_apfd)

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(losses).mean()


def create_ranking_loss(config: dict) -> nn.Module:
    """
    Create ranking loss function based on configuration.

    Args:
        config: Configuration dictionary with 'training.ranking_loss' section

    Returns:
        Loss function module
    """
    loss_config = config.get('training', {}).get('ranking_loss', {})
    loss_type = loss_config.get('type', 'listnet')

    if loss_type == 'listnet':
        return ListNetLoss(
            temperature=loss_config.get('temperature', 1.0)
        )
    elif loss_type == 'listmle':
        return ListMLELoss()
    elif loss_type == 'lambdarank':
        return LambdaRankLoss(
            sigma=loss_config.get('sigma', 1.0),
            k=loss_config.get('k', None)
        )
    elif loss_type == 'approx_ndcg':
        return ApproxNDCGLoss(
            temperature=loss_config.get('temperature', 1.0),
            k=loss_config.get('k', None)
        )
    elif loss_type == 'apfd_weighted':
        return APFDWeightedPairwiseLoss(
            margin=loss_config.get('margin', 0.5),
            temperature=loss_config.get('temperature', 1.0),
            position_decay=loss_config.get('position_decay', 'linear'),
            min_pairs_for_loss=loss_config.get('min_pairs_for_loss', 1),
            use_soft_margin=loss_config.get('use_soft_margin', True),
            normalize_weights=loss_config.get('normalize_weights', True)
        )
    elif loss_type == 'soft_apfd':
        return SoftAPFDLoss(
            temperature=loss_config.get('temperature', 1.0)
        )
    elif loss_type == 'mse':
        return RankingMSELoss()
    else:
        raise ValueError(f"Unknown ranking loss type: {loss_type}")
