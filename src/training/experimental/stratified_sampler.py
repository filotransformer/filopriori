"""
Stratified Build Sampler for Test Case Prioritization.

This module implements a custom sampler that ensures a minimum number of
failure samples (positive class) per batch. This is critical for:
1. Ranking loss computation (needs pairs of failures and passes)
2. Stable gradient updates with extreme class imbalance
3. Effective learning of failure patterns

Reference: Batch-Balanced Focal Loss (BBFL) - PMC10289178
"""

import numpy as np
from typing import Iterator, List, Optional
from torch.utils.data import Sampler
import logging

logger = logging.getLogger(__name__)


class StratifiedBuildSampler(Sampler):
    """
    Custom sampler that ensures minimum positive samples per batch.

    With extreme class imbalance (e.g., 37:1 Pass:Fail ratio), standard
    random sampling results in batches with 0-1 failures, making ranking
    loss computation ineffective.

    This sampler guarantees at least `min_positives_per_batch` failure
    samples in each batch, enabling effective pairwise/listwise ranking.

    Args:
        labels: Array of labels (0 = Fail/minority, 1 = Pass/majority)
        build_ids: Optional array of build IDs for build-aware sampling
        batch_size: Number of samples per batch
        min_positives_per_batch: Minimum failure samples per batch (default: 4)
        minority_class: Which class is minority (default: 0 for Fail)
        drop_last: Whether to drop last incomplete batch
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        labels: np.ndarray,
        build_ids: Optional[np.ndarray] = None,
        batch_size: int = 32,
        min_positives_per_batch: int = 4,
        minority_class: int = 0,
        drop_last: bool = False,
        seed: int = 42
    ):
        self.labels = np.array(labels)
        self.build_ids = build_ids
        self.batch_size = batch_size
        self.min_positives = min(min_positives_per_batch, batch_size // 2)
        self.minority_class = minority_class
        self.drop_last = drop_last
        self.seed = seed

        # Find indices for each class
        self.pos_indices = np.where(self.labels == minority_class)[0]
        self.neg_indices = np.where(self.labels != minority_class)[0]

        self.n_pos = len(self.pos_indices)
        self.n_neg = len(self.neg_indices)
        self.n_total = len(self.labels)

        # Calculate number of batches
        # Each batch needs min_positives failures
        # We can create at most n_pos / min_positives complete batches from positives
        self.n_batches = max(1, self.n_pos // self.min_positives)

        # Log sampling statistics
        logger.info("="*70)
        logger.info("STRATIFIED BUILD SAMPLER INITIALIZED")
        logger.info("="*70)
        logger.info(f"  Total samples: {self.n_total}")
        logger.info(f"  Minority class ({minority_class}): {self.n_pos} ({100*self.n_pos/self.n_total:.2f}%)")
        logger.info(f"  Majority class: {self.n_neg} ({100*self.n_neg/self.n_total:.2f}%)")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Min positives per batch: {self.min_positives}")
        logger.info(f"  Expected batches per epoch: {self.n_batches}")
        logger.info(f"  Negatives per batch: {batch_size - self.min_positives}")
        logger.info("="*70)

    def __iter__(self) -> Iterator[int]:
        """Generate indices for one epoch."""
        rng = np.random.RandomState(self.seed + self._epoch if hasattr(self, '_epoch') else self.seed)

        # Shuffle indices
        pos_indices = self.pos_indices.copy()
        neg_indices = self.neg_indices.copy()
        rng.shuffle(pos_indices)
        rng.shuffle(neg_indices)

        # Generate batches
        batches = []
        pos_ptr = 0
        neg_ptr = 0

        while pos_ptr < len(pos_indices):
            batch = []

            # Add minimum positives
            n_pos_to_add = min(self.min_positives, len(pos_indices) - pos_ptr)
            for _ in range(n_pos_to_add):
                batch.append(pos_indices[pos_ptr])
                pos_ptr += 1

            # Fill rest with negatives
            n_neg_to_add = self.batch_size - len(batch)
            if neg_ptr + n_neg_to_add > len(neg_indices):
                # Reshuffle and reset negative pointer if exhausted
                rng.shuffle(neg_indices)
                neg_ptr = 0

            for _ in range(min(n_neg_to_add, len(neg_indices))):
                batch.append(neg_indices[neg_ptr])
                neg_ptr += 1

            # Only add batch if it has minimum positives
            if len([idx for idx in batch if self.labels[idx] == self.minority_class]) >= self.min_positives:
                # Shuffle within batch to avoid positives always being first
                rng.shuffle(batch)
                batches.append(batch)

        # Shuffle batch order
        rng.shuffle(batches)

        # Flatten and yield
        for batch in batches:
            yield from batch

    def __len__(self) -> int:
        """Return total number of samples per epoch."""
        return self.n_batches * self.batch_size

    def set_epoch(self, epoch: int):
        """Set epoch for different shuffling each epoch."""
        self._epoch = epoch


class BuildAwareSampler(Sampler):
    """
    Advanced sampler that groups samples by build for ranking.

    For listwise ranking loss, we need entire builds in a batch.
    This sampler ensures each batch contains complete builds.

    Args:
        labels: Array of labels
        build_ids: Array of build IDs
        batch_size: Target batch size (actual may vary by build size)
        min_failures_per_batch: Minimum failures in selected builds
        seed: Random seed
    """

    def __init__(
        self,
        labels: np.ndarray,
        build_ids: np.ndarray,
        batch_size: int = 32,
        min_failures_per_batch: int = 2,
        minority_class: int = 0,
        seed: int = 42
    ):
        self.labels = np.array(labels)
        self.build_ids = np.array(build_ids)
        self.batch_size = batch_size
        self.min_failures = min_failures_per_batch
        self.minority_class = minority_class
        self.seed = seed

        # Group indices by build
        self.build_to_indices = {}
        self.build_to_n_failures = {}

        for idx, (bid, label) in enumerate(zip(self.build_ids, self.labels)):
            if bid not in self.build_to_indices:
                self.build_to_indices[bid] = []
                self.build_to_n_failures[bid] = 0
            self.build_to_indices[bid].append(idx)
            if label == minority_class:
                self.build_to_n_failures[bid] += 1

        # Filter builds with at least min_failures
        self.valid_builds = [
            bid for bid, n_fail in self.build_to_n_failures.items()
            if n_fail >= self.min_failures
        ]

        logger.info("="*70)
        logger.info("BUILD-AWARE SAMPLER INITIALIZED")
        logger.info("="*70)
        logger.info(f"  Total builds: {len(self.build_to_indices)}")
        logger.info(f"  Builds with >= {min_failures_per_batch} failures: {len(self.valid_builds)}")
        logger.info(f"  Batch size target: {batch_size}")
        logger.info("="*70)

    def __iter__(self) -> Iterator[int]:
        """Generate indices grouped by build."""
        rng = np.random.RandomState(self.seed + self._epoch if hasattr(self, '_epoch') else self.seed)

        # Shuffle builds
        builds = self.valid_builds.copy()
        rng.shuffle(builds)

        # Create batches by combining builds
        current_batch = []

        for bid in builds:
            build_indices = self.build_to_indices[bid]

            if len(current_batch) + len(build_indices) <= self.batch_size:
                # Add entire build to current batch
                current_batch.extend(build_indices)
            else:
                # Yield current batch and start new one
                if current_batch:
                    rng.shuffle(current_batch)
                    yield from current_batch
                current_batch = list(build_indices)

        # Yield last batch
        if current_batch:
            rng.shuffle(current_batch)
            yield from current_batch

    def __len__(self) -> int:
        """Return total samples in valid builds."""
        return sum(len(self.build_to_indices[bid]) for bid in self.valid_builds)

    def set_epoch(self, epoch: int):
        """Set epoch for different shuffling."""
        self._epoch = epoch


def create_stratified_sampler(
    labels: np.ndarray,
    build_ids: Optional[np.ndarray] = None,
    batch_size: int = 32,
    min_positives: int = 4,
    use_build_aware: bool = False,
    seed: int = 42
) -> Sampler:
    """
    Factory function to create the appropriate stratified sampler.

    Args:
        labels: Sample labels
        build_ids: Optional build IDs for build-aware sampling
        batch_size: Batch size
        min_positives: Minimum positive samples per batch
        use_build_aware: If True and build_ids provided, use BuildAwareSampler
        seed: Random seed

    Returns:
        Configured sampler instance
    """
    if use_build_aware and build_ids is not None:
        return BuildAwareSampler(
            labels=labels,
            build_ids=build_ids,
            batch_size=batch_size,
            min_failures_per_batch=min_positives,
            seed=seed
        )
    else:
        return StratifiedBuildSampler(
            labels=labels,
            build_ids=build_ids,
            batch_size=batch_size,
            min_positives_per_batch=min_positives,
            seed=seed
        )
