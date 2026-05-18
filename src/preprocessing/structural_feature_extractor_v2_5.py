"""
Structural Feature Extractor V2.5 - Feature Selection Version

This is a wrapper around V2 that extracts only the top-10 selected features
identified through expert analysis and domain knowledge.

Selected Features (10):
- Baseline (6): test_age, failure_rate, recent_failure_rate, flakiness_rate,
                commit_count, test_novelty
- New (4): consecutive_failures, max_consecutive_failures, failure_trend, cr_count

This reduces overfitting while maintaining proven signals and adding complementary value.

Author: Filo-Priori V8 Team
Date: 2025-11-14
"""

import numpy as np
import pandas as pd
from typing import List
from .structural_feature_extractor_v2 import StructuralFeatureExtractorV2
import logging

logger = logging.getLogger(__name__)

# Top-10 selected feature indices (from 29 V2 features)
SELECTED_FEATURE_INDICES = [0, 1, 2, 3, 7, 9, 13, 20, 21, 23]

# Corresponding feature names
SELECTED_FEATURE_NAMES = [
    'test_age',                    # 0 - Baseline
    'failure_rate',                # 1 - Baseline
    'recent_failure_rate',         # 2 - Baseline
    'flakiness_rate',              # 3 - Baseline
    'consecutive_failures',        # 7 - NEW
    'max_consecutive_failures',    # 9 - NEW
    'failure_trend',               # 13 - NEW
    'commit_count',                # 20 - Baseline
    'test_novelty',                # 21 - Baseline
    'cr_count',                    # 23 - NEW
]


class StructuralFeatureExtractorV2_5(StructuralFeatureExtractorV2):
    """
    Feature selection wrapper around V2 extractor.

    Extracts all 29 V2 features, then selects only the top-10.
    """

    def __init__(self,
                 recent_window: int = 5,
                 very_recent_window: int = 2,
                 medium_term_window: int = 10,
                 min_history: int = 2,
                 verbose: bool = True,
                 selected_indices: List[int] = None):
        """
        Initialize V2.5 extractor.

        Args:
            recent_window: Window for recent failure rate
            very_recent_window: Window for very recent failure rate
            medium_term_window: Window for medium term failure rate
            min_history: Minimum history required
            verbose: Enable verbose logging
            selected_indices: Feature indices to select (default: top-10)
        """
        # Initialize parent V2
        super().__init__(
            recent_window=recent_window,
            very_recent_window=very_recent_window,
            medium_term_window=medium_term_window,
            min_history=min_history,
            verbose=verbose
        )

        # Use provided indices or default top-10
        self.selected_indices = selected_indices if selected_indices is not None else SELECTED_FEATURE_INDICES
        self.num_selected_features = len(self.selected_indices)

        logger.info(f"Initialized StructuralFeatureExtractorV2.5 with:")
        logger.info(f"  recent_window={recent_window}")
        logger.info(f"  very_recent_window={very_recent_window}")
        logger.info(f"  medium_term_window={medium_term_window}")
        logger.info(f"  → Extracting {self.num_selected_features} selected features from 29")
        logger.info(f"  → Selected indices: {self.selected_indices}")

    def transform(self, df: pd.DataFrame, is_test: bool = False) -> np.ndarray:
        """
        Transform data into selected structural features.

        This method:
        1. Calls parent V2.transform() to get all 29 features
        2. Selects only the chosen 10 features
        3. Returns [N, 10] feature matrix

        Args:
            df: Input DataFrame
            is_test: Whether this is test data

        Returns:
            features: [N, 10] array of selected structural features
        """
        # Get all 29 features from V2
        features_full = super().transform(df, is_test=is_test)  # [N, 29]

        # Select only the chosen features
        features_selected = features_full[:, self.selected_indices]  # [N, 10]

        if self.verbose:
            logger.info(f"✓ Selected {self.num_selected_features} features from {features_full.shape[1]}")
            logger.info(f"  Feature matrix: {features_selected.shape}")

        return features_selected

    def get_feature_names(self) -> List[str]:
        """
        Get names of selected features.

        Returns:
            feature_names: List of 10 feature names
        """
        return SELECTED_FEATURE_NAMES

    def fit_transform_temporal(self, df_train: pd.DataFrame) -> np.ndarray:
        """Fit and transform temporally, then select features."""
        features_selected = super().fit_transform_temporal(df_train)
        if self.verbose:
            logger.info(f"✓ Selected {self.num_selected_features} features (temporal fit)")
        return features_selected
        
    def transform_temporal(self, df: pd.DataFrame) -> np.ndarray:
        """Transform temporally, then select features."""
        features_full = super().transform_temporal(df)
        features_selected = features_full[:, self.selected_indices]
        if self.verbose:
            logger.info(f"✓ Selected {self.num_selected_features} features (temporal transform)")
        return features_selected

    def save_history(self, filepath: str):
        """Save historical state with V2.5 marker."""
        import pickle

        state = {
            'tc_history': self.tc_history,
            'build_chronology': self.build_chronology,
            'tc_first_appearance': self.tc_first_appearance,
            'recent_window': self.recent_window,
            'very_recent_window': self.very_recent_window,
            'medium_term_window': self.medium_term_window,
            'min_history': self.min_history,
            'feature_means': self.feature_means,
            'feature_medians': self.feature_medians,
            'feature_stds': self.feature_stds,
            'selected_indices': self.selected_indices,  # V2.5 specific
            'version': 'v2.5'
        }

        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"✓ Saved V2.5 historical state to {filepath}")

    def load_history(self, filepath: str) -> 'StructuralFeatureExtractorV2_5':
        """Load previously computed statistics."""
        import pickle

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.tc_history = state['tc_history']
        self.build_chronology = state['build_chronology']
        self.tc_first_appearance = state['tc_first_appearance']
        self.recent_window = state.get('recent_window', 5)
        self.very_recent_window = state.get('very_recent_window', 2)
        self.medium_term_window = state.get('medium_term_window', 10)
        self.min_history = state.get('min_history', 2)
        self.feature_means = state.get('feature_means', None)
        self.feature_medians = state.get('feature_medians', None)
        self.feature_stds = state.get('feature_stds', None)
        self.selected_indices = state.get('selected_indices', SELECTED_FEATURE_INDICES)

        version = state.get('version', 'unknown')
        logger.info(f"✓ Loaded V2.5 historical state from {filepath} (version: {version})")
        logger.info(f"  {len(self.tc_history)} test cases")
        logger.info(f"  {len(self.build_chronology)} builds")
        logger.info(f"  {len(self.selected_indices)} selected features")

        return self


# For easy import
__all__ = ['StructuralFeatureExtractorV2_5', 'SELECTED_FEATURE_INDICES', 'SELECTED_FEATURE_NAMES']
