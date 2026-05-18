"""
Structural Feature Imputation for Test Inference

This module provides advanced imputation strategies for structural/phylogenetic features
during inference when historical data is unavailable.

Problem:
    During inference, new test cases or tests without sufficient history cannot have
    accurate structural features (failure_rate, recent_failure_rate, etc.)

Solution:
    1. Semantic Similarity-Based Imputation (PREFERRED):
       - Find K most similar tests with history using semantic embeddings
       - Use weighted average of their features based on similarity

    2. Statistical Imputation (FALLBACK):
       - Use mean/median from training distribution
       - Add noise to avoid all unknowns having identical features

    3. Cold-Start Strategies:
       - Conservative priors for unknown tests
       - Uncertainty-aware defaults

Author: Filo-Priori V8 Team
Date: 2025-11-07
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class StructuralFeatureImputer:
    """
    Imputes missing structural features using semantic similarity and statistical methods.

    This class addresses the cold-start problem where new tests or tests without
    sufficient history need structural features for inference.
    """

    def __init__(self,
                 k_neighbors: int = 10,
                 similarity_threshold: float = 0.5,
                 use_weighted: bool = True,
                 add_noise: bool = True,
                 noise_std: float = 0.05,
                 verbose: bool = True):
        """
        Initialize the imputer.

        Args:
            k_neighbors: Number of similar tests to use for imputation
            similarity_threshold: Minimum cosine similarity to consider (0-1)
            use_weighted: If True, weight by similarity; else simple average
            add_noise: Add small Gaussian noise to imputed values (avoid identical features)
            noise_std: Standard deviation of noise (as fraction of feature std)
            verbose: Enable verbose logging
        """
        self.k_neighbors = k_neighbors
        self.similarity_threshold = similarity_threshold
        self.use_weighted = use_weighted
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.verbose = verbose

        # Fitted statistics from training data
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.feature_medians: Optional[np.ndarray] = None

        # Training reference data (tests WITH history)
        self.reference_embeddings: Optional[np.ndarray] = None
        self.reference_features: Optional[np.ndarray] = None
        self.reference_tc_keys: Optional[List[str]] = None

        logger.info(f"Initialized StructuralFeatureImputer with k={k_neighbors}, "
                   f"threshold={similarity_threshold:.2f}")

    def fit(self,
            embeddings_train: np.ndarray,
            features_train: np.ndarray,
            tc_keys_train: List[str],
            has_history_mask: Optional[np.ndarray] = None):
        """
        Fit the imputer on training data.

        Args:
            embeddings_train: Semantic embeddings [N, embed_dim]
            features_train: Structural features [N, 6]
            tc_keys_train: List of TC_Key strings [N]
            has_history_mask: Boolean mask [N] indicating which tests have history.
                             If None, assumes all have history.
        """
        logger.info("Fitting StructuralFeatureImputer on training data...")
        logger.info(f"  Training samples: {len(features_train)}")

        # Compute global statistics (from ALL training data)
        self.feature_means = np.mean(features_train, axis=0)
        self.feature_stds = np.std(features_train, axis=0)
        self.feature_medians = np.median(features_train, axis=0)

        logger.info(f"  Feature means: {self.feature_means}")
        logger.info(f"  Feature stds: {self.feature_stds}")

        # Store reference data (only tests WITH history)
        if has_history_mask is None:
            # Assume all have history
            has_history_mask = np.ones(len(features_train), dtype=bool)

        if has_history_mask.all():
            # All samples have history - avoid copying (saves ~7 GB for large datasets)
            self.reference_embeddings = embeddings_train
            self.reference_features = features_train
            self.reference_tc_keys = list(tc_keys_train)
        else:
            self.reference_embeddings = embeddings_train[has_history_mask]
            self.reference_features = features_train[has_history_mask]
            self.reference_tc_keys = [tc_keys_train[i] for i in range(len(tc_keys_train))
                                      if has_history_mask[i]]

        logger.info(f"  Reference tests (with history): {len(self.reference_tc_keys)}")

        # Pre-normalize reference embeddings in-place for dot-product cosine similarity
        # This avoids sklearn's cosine_similarity creating an internal normalized copy (~7 GB)
        norms = np.linalg.norm(self.reference_embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)  # Avoid division by zero
        self.reference_embeddings /= norms
        del norms
        logger.info("  Pre-normalized reference embeddings in-place for dot-product similarity")

        return self

    def impute_features(self,
                       embeddings_query: np.ndarray,
                       features_query: np.ndarray,
                       needs_imputation_mask: np.ndarray,
                       chunk_size: int = 50) -> np.ndarray:
        """
        Impute missing structural features using semantic similarity.

        Computes cosine similarity in chunks to avoid OOM on large reference sets.

        Args:
            embeddings_query: Semantic embeddings for query samples [M, embed_dim]
            features_query: Current structural features [M, 6] (may have zeros/missing)
            needs_imputation_mask: Boolean mask [M] indicating which samples need imputation
            chunk_size: Number of query samples to process per chunk (controls peak memory)

        Returns:
            features_imputed: Imputed structural features [M, 6]
        """
        if self.reference_embeddings is None:
            raise RuntimeError("Imputer not fitted! Call fit() first.")

        features_imputed = features_query.copy()
        num_imputed = needs_imputation_mask.sum()

        if num_imputed == 0:
            logger.info("No samples need imputation.")
            return features_imputed

        num_reference = len(self.reference_embeddings)
        # Estimate peak memory per chunk: chunk_size × num_reference × 8 bytes (float64)
        mem_per_chunk_gb = (chunk_size * num_reference * 8) / (1024**3)
        logger.info(f"Imputing features for {num_imputed}/{len(features_query)} samples "
                   f"(chunked: {chunk_size} samples/chunk, ~{mem_per_chunk_gb:.1f} GB peak per chunk, "
                   f"{num_reference} reference samples)")

        # Get embeddings and original indices for samples needing imputation
        embeddings_to_impute = embeddings_query[needs_imputation_mask]
        original_indices = np.where(needs_imputation_mask)[0]

        imputation_methods_used = {'semantic': 0, 'fallback': 0}

        # Process in chunks to avoid allocating a huge similarity matrix
        num_chunks = (num_imputed + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, num_imputed)
            chunk_embeddings = embeddings_to_impute[chunk_start:chunk_end]

            # Compute cosine similarity via dot product (reference is already L2-normalized)
            # Shape: [chunk_end - chunk_start, num_reference]
            chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
            chunk_norms = np.maximum(chunk_norms, 1e-12)
            chunk_normed = chunk_embeddings / chunk_norms
            chunk_sims = chunk_normed @ self.reference_embeddings.T
            del chunk_normed, chunk_norms

            for i_local in range(chunk_sims.shape[0]):
                i_global = chunk_start + i_local
                sims = chunk_sims[i_local]

                # Find top-k similar tests above threshold
                top_k_indices = np.argsort(sims)[::-1][:self.k_neighbors]
                top_k_sims = sims[top_k_indices]

                # Filter by threshold
                valid_mask = top_k_sims >= self.similarity_threshold

                if valid_mask.sum() > 0:
                    # Semantic similarity-based imputation
                    valid_indices = top_k_indices[valid_mask]
                    valid_sims = top_k_sims[valid_mask]

                    # Get features from similar tests
                    similar_features = self.reference_features[valid_indices]  # [K, 6]

                    if self.use_weighted:
                        # Weighted average by similarity
                        weights = valid_sims / valid_sims.sum()  # Normalize
                        imputed_values = (similar_features.T @ weights).T  # [6]
                    else:
                        # Simple average
                        imputed_values = similar_features.mean(axis=0)  # [6]

                    imputation_methods_used['semantic'] += 1

                    if self.verbose and i_global % 100 == 0:
                        logger.debug(f"  Sample {i_global}: Imputed from {len(valid_indices)} similar tests "
                                   f"(avg sim: {valid_sims.mean():.3f})")
                else:
                    # Fallback: use global statistics
                    imputed_values = self._get_conservative_defaults()
                    imputation_methods_used['fallback'] += 1

                    if self.verbose and i_global % 100 == 0:
                        logger.debug(f"  Sample {i_global}: No similar tests found, using conservative defaults")

                # Add noise to avoid identical features
                if self.add_noise:
                    num_features = len(self.feature_stds)
                    noise = np.random.normal(0, self.noise_std * self.feature_stds, size=num_features)
                    imputed_values = imputed_values + noise
                    imputed_values = self._clip_features(imputed_values)

                # Update features using original index
                features_imputed[original_indices[i_global]] = imputed_values

            # Free chunk similarity matrix immediately
            del chunk_sims

            if (chunk_idx + 1) % 10 == 0 or chunk_idx == num_chunks - 1:
                logger.info(f"  Chunk {chunk_idx + 1}/{num_chunks} done "
                           f"({chunk_end}/{num_imputed} samples processed)")

        logger.info(f"  Imputation complete:")
        logger.info(f"    Semantic-based: {imputation_methods_used['semantic']}")
        logger.info(f"    Fallback (conservative): {imputation_methods_used['fallback']}")

        return features_imputed

    def _get_conservative_defaults(self) -> np.ndarray:
        """
        Get conservative default values for unknown tests.

        Strategy (works for both V1 (6 features) and V2 (29 features)):
            - Use feature_means for most features (population average)
            - Override test_age (index 0): 0 (new test)
            - Override test_novelty (index 1 for V1, index 6 for V2): 1.0 (assume novel)
            - For rates (failure_rate, flakiness_rate), use medians to be conservative

        Returns:
            defaults: [6] or [29] array of conservative values
        """
        # Start with feature means for all features
        defaults = self.feature_means.copy()

        # Override test_age (always index 0): new test
        defaults[0] = 0.0

        # Override test_novelty to 1.0 (assume novel)
        # In V1: test_novelty is feature 5 (last)
        # In V2: test_novelty is feature 6 (after first 6 phylogenetic features)
        num_features = len(defaults)
        if num_features == 6:
            # V1: test_novelty is index 5
            defaults[5] = 1.0
        elif num_features >= 29:
            # V2: test_novelty is index 6 (after phylo features 0-5)
            defaults[6] = 1.0

        # Use medians for rate features to be more conservative
        # flakiness_rate is index 3 in both V1 and V2
        if num_features >= 4:
            defaults[3] = self.feature_medians[3]

        return defaults

    def _clip_features(self, features: np.ndarray) -> np.ndarray:
        """
        Clip features to valid ranges (works for both V1 (6) and V2 (29) features).

        Common ranges across V1 and V2:
            - test_age (index 0): [0, inf)
            - failure_rate (index 1): [0, 1]
            - recent_failure_rate (index 2): [0, 1]
            - flakiness_rate (index 3): [0, 1]

        V1-specific (6 features):
            - commit_count (index 4): [1, inf)
            - test_novelty (index 5): [0, 1]

        V2-specific (29 features):
            - Many more rate features: [0, 1]
            - Counts: >= 0
            - commit_count (index 20): [1, inf)
            - test_novelty (index 21): [0, 1]

        Args:
            features: [6] or [29] array

        Returns:
            clipped_features: array with valid ranges
        """
        clipped = features.copy()
        num_features = len(features)

        # Common clipping for both V1 and V2
        # test_age: >= 0
        clipped[0] = max(0.0, clipped[0])

        # Rates (indices 1, 2, 3): [0, 1]
        for i in [1, 2, 3]:
            clipped[i] = np.clip(clipped[i], 0.0, 1.0)

        if num_features == 6:
            # V1 clipping
            # commit_count: >= 1
            clipped[4] = max(1.0, clipped[4])
            # test_novelty: [0, 1]
            clipped[5] = np.clip(clipped[5], 0.0, 1.0)

        elif num_features >= 29:
            # V2 clipping
            # Additional rate features: execution_frequency(12), very_recent_failure_rate(15),
            # medium_term_failure_rate(16), stability_score(27), pass_fail_ratio(28)
            for i in [12, 15, 16, 27]:
                if i < num_features:
                    clipped[i] = np.clip(clipped[i], 0.0, 1.0)

            # Counts (execution_count(4), failure_count(5), pass_count(6), etc.): >= 0
            for i in [4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 22, 23, 24]:
                if i < num_features:
                    clipped[i] = max(0.0, clipped[i])

            # commit_count (index 20): >= 1
            clipped[20] = max(1.0, clipped[20])

            # test_novelty (index 21): [0, 1]
            clipped[21] = np.clip(clipped[21], 0.0, 1.0)

            # pass_fail_ratio (index 28): >= 0
            if num_features > 28:
                clipped[28] = max(0.0, clipped[28])

        return clipped

    def get_imputation_mask(self,
                           tc_keys: List[str],
                           tc_history: Dict[str, Dict]) -> np.ndarray:
        """
        Determine which samples need imputation.

        Args:
            tc_keys: List of TC_Key strings [N]
            tc_history: Dictionary mapping TC_Key to historical stats

        Returns:
            needs_imputation: Boolean mask [N] indicating samples needing imputation
        """
        needs_imputation = np.array([
            (tc_key not in tc_history) or
            (tc_history[tc_key]['total_executions'] < 2)
            for tc_key in tc_keys
        ])

        return needs_imputation


def impute_structural_features(
    embeddings_train: np.ndarray,
    features_train: np.ndarray,
    tc_keys_train: List[str],
    embeddings_test: np.ndarray,
    features_test: np.ndarray,
    tc_keys_test: List[str],
    tc_history: Dict[str, Dict],
    k_neighbors: int = 10,
    similarity_threshold: float = 0.5,
    verbose: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to impute structural features for test set.

    Args:
        embeddings_train: Training embeddings [N_train, embed_dim]
        features_train: Training structural features [N_train, 6]
        tc_keys_train: Training TC_Keys [N_train]
        embeddings_test: Test embeddings [N_test, embed_dim]
        features_test: Test structural features (may have missing) [N_test, 6]
        tc_keys_test: Test TC_Keys [N_test]
        tc_history: Dictionary with TC_Key historical stats
        k_neighbors: Number of neighbors for imputation
        similarity_threshold: Minimum similarity threshold
        verbose: Enable verbose logging

    Returns:
        features_test_imputed: Imputed test features [N_test, 6]
        imputation_stats: Dictionary with imputation statistics

    Example:
        >>> features_test_imputed, stats = impute_structural_features(
        ...     embeddings_train, features_train, tc_keys_train,
        ...     embeddings_test, features_test, tc_keys_test,
        ...     tc_history, k_neighbors=10
        ... )
        >>> print(f"Imputed {stats['num_imputed']} samples")
    """
    logger.info("="*70)
    logger.info("STRUCTURAL FEATURE IMPUTATION")
    logger.info("="*70)

    # Initialize imputer
    imputer = StructuralFeatureImputer(
        k_neighbors=k_neighbors,
        similarity_threshold=similarity_threshold,
        use_weighted=True,
        add_noise=True,
        noise_std=0.05,
        verbose=verbose
    )

    # Determine which training samples have history (for reference)
    has_history_train = np.array([
        tc_key in tc_history for tc_key in tc_keys_train
    ])

    logger.info(f"Training samples with history: {has_history_train.sum()}/{len(tc_keys_train)}")

    # Fit imputer
    imputer.fit(
        embeddings_train,
        features_train,
        tc_keys_train,
        has_history_mask=has_history_train
    )

    # Determine which test samples need imputation
    needs_imputation_test = imputer.get_imputation_mask(tc_keys_test, tc_history)

    logger.info(f"\nTest samples needing imputation: {needs_imputation_test.sum()}/{len(tc_keys_test)}")

    if needs_imputation_test.sum() > 0:
        # Impute
        features_test_imputed = imputer.impute_features(
            embeddings_test,
            features_test,
            needs_imputation_test
        )

        # Compute statistics
        imputation_stats = {
            'num_imputed': int(needs_imputation_test.sum()),
            'num_total': len(tc_keys_test),
            'imputation_rate': float(needs_imputation_test.sum() / len(tc_keys_test)),
            'feature_means_before': features_test[needs_imputation_test].mean(axis=0).tolist(),
            'feature_means_after': features_test_imputed[needs_imputation_test].mean(axis=0).tolist()
        }

        logger.info(f"\nImputation Statistics:")
        logger.info(f"  Samples imputed: {imputation_stats['num_imputed']}/{imputation_stats['num_total']} "
                   f"({imputation_stats['imputation_rate']*100:.1f}%)")
        logger.info(f"  Feature means before: {imputation_stats['feature_means_before']}")
        logger.info(f"  Feature means after: {imputation_stats['feature_means_after']}")
    else:
        logger.info("No test samples need imputation!")
        features_test_imputed = features_test
        imputation_stats = {
            'num_imputed': 0,
            'num_total': len(tc_keys_test),
            'imputation_rate': 0.0
        }

    # Free imputer reference data to reclaim memory
    del imputer
    logger.info("="*70)

    return features_test_imputed, imputation_stats


# For backwards compatibility and ease of import
__all__ = [
    'StructuralFeatureImputer',
    'impute_structural_features'
]
