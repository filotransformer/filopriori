"""
Structural Feature Extractor for Filo-Priori V8

This module extracts TRUE structural and phylogenetic features from historical data,
breaking the "semantic echo chamber" of V7 where both streams used the same BGE embeddings.

Features are divided into two categories:

1. PHYLOGENETIC FEATURES (based on test case history):
   - test_age: Number of builds since first appearance
   - failure_rate: Historical failure rate
   - recent_failure_rate: Failure rate in last N builds
   - flakiness_rate: State transition rate (Pass<->Fail oscillation)

2. STRUCTURAL FEATURES (based on code changes):
   - commit_count: Number of unique commits/CRs associated with this execution
   - test_novelty: Boolean flag if this is the first appearance of TC_Key

These features provide genuine structural information orthogonal to semantic embeddings,
validating the thesis hypothesis that semantic + structural fusion improves performance.

Author: Filo-Priori V8 Team
Date: 2025-11-06
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import ast
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class StructuralFeatureExtractor:
    """
    Extracts structural and phylogenetic features from test execution history.

    This class computes features that represent:
    - Test case lifecycle (age, stability)
    - Historical behavior (failure patterns)
    - Code change impact (commits, novelty)

    The features are independent of text content and provide true structural
    information for the Structural Stream of the dual-stream architecture.
    """

    def __init__(self,
                 recent_window: int = 5,
                 min_history: int = 2,
                 verbose: bool = True):
        """
        Initialize the structural feature extractor.

        Args:
            recent_window: Number of recent builds for recent_failure_rate calculation
            min_history: Minimum number of executions needed to compute historical features
            verbose: Enable verbose logging
        """
        self.recent_window = recent_window
        self.min_history = min_history
        self.verbose = verbose

        # Cache for historical statistics (computed from training data)
        self.tc_history: Dict[str, Dict] = {}
        self.build_chronology: List[str] = []
        self._build_to_idx: Dict[str, int] = {}  # O(1) lookup for build index
        self.tc_first_appearance: Dict[str, int] = {}

        # Global statistics for conservative defaults (computed during fit)
        self.feature_means: Optional[np.ndarray] = None
        self.feature_medians: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

        logger.info(f"Initialized StructuralFeatureExtractor with recent_window={recent_window}")

    def fit(self, df_train: pd.DataFrame) -> 'StructuralFeatureExtractor':
        """
        Fit the extractor on training data to learn historical patterns.

        This method:
        1. Establishes build chronology
        2. Computes per-TC_Key historical statistics
        3. Stores first appearance information
        4. Computes global statistics for conservative defaults

        Args:
            df_train: Training DataFrame with columns:
                - TC_Key
                - Build_ID
                - TE_Test_Result
                - Build_Test_Start_Date (optional for chronology)
                - commit (optional)
                - CR (optional)

        Returns:
            self (for method chaining)
        """
        logger.info("Fitting StructuralFeatureExtractor on training data...")
        logger.info(f"Training data shape: {df_train.shape}")

        # 1. Establish build chronology
        self._establish_chronology(df_train)

        # 2. Compute per-TC_Key historical statistics
        self._compute_tc_history(df_train)

        # 3. Store first appearance information
        self._compute_first_appearances(df_train)

        # 4. Compute global statistics for conservative defaults
        self._compute_global_statistics(df_train)

        logger.info(f"Fitted extractor on {len(self.tc_history)} unique test cases")
        logger.info(f"Build chronology spans {len(self.build_chronology)} builds")

        return self

    def transform(self, df: pd.DataFrame, is_test: bool = False) -> np.ndarray:
        """
        Transform a DataFrame into structural feature vectors.

        Args:
            df: DataFrame to transform (train, val, or test)
            is_test: If True, uses historical stats from training only

        Returns:
            feature_matrix: np.ndarray of shape [N, num_features]
                where num_features = 6:
                [test_age, failure_rate, recent_failure_rate,
                 flakiness_rate, commit_count, test_novelty]
        """
        logger.info(f"Transforming {len(df)} samples...")

        features = []

        for idx, row in df.iterrows():
            tc_key = row['TC_Key']
            build_id = row['Build_ID']

            # Extract phylogenetic features
            phylo_features = self._extract_phylogenetic_features(
                tc_key, build_id, is_test
            )

            # Extract structural features
            struct_features = self._extract_structural_features(row)

            # Combine features
            feature_vector = phylo_features + struct_features
            features.append(feature_vector)

            if self.verbose and len(features) % 10000 == 0:
                logger.info(f"  Processed {len(features)}/{len(df)} samples...")

        feature_matrix = np.array(features, dtype=np.float32)

        logger.info(f"Extracted feature matrix: {feature_matrix.shape}")
        logger.info(f"Feature stats: mean={feature_matrix.mean(axis=0)}, std={feature_matrix.std(axis=0)}")

        return feature_matrix

    def fit_transform(self, df_train: pd.DataFrame) -> np.ndarray:
        """
        Fit on training data and transform it in one step.

        Args:
            df_train: Training DataFrame

        Returns:
            feature_matrix: Structural features for training data
        """
        self.fit(df_train)
        return self.transform(df_train, is_test=False)

    # ==================== PRIVATE METHODS ====================

    def _establish_chronology(self, df: pd.DataFrame) -> None:
        """
        Establish chronological order of builds.

        Uses Build_Test_Start_Date if available, otherwise uses Build_ID order.
        """
        if 'Build_Test_Start_Date' in df.columns:
            # Sort by date
            build_dates = df.groupby('Build_ID')['Build_Test_Start_Date'].first().sort_values()
            self.build_chronology = build_dates.index.tolist()
            logger.info("Build chronology established using Build_Test_Start_Date")
        else:
            # Fallback: use order of appearance
            self.build_chronology = df['Build_ID'].unique().tolist()
            logger.warning("Build_Test_Start_Date not found, using order of appearance")

        # Build O(1) index lookup dict
        self._build_to_idx = {build_id: idx for idx, build_id in enumerate(self.build_chronology)}

        logger.info(f"Chronology spans {len(self.build_chronology)} builds")

    def _compute_tc_history(self, df: pd.DataFrame) -> None:
        """
        Compute historical statistics for each TC_Key.

        For each TC_Key, computes:
        - Total executions
        - Total failures
        - Failure rate
        - Recent failure rate
        - Flakiness rate
        - Build-level history
        """
        logger.info("Computing per-TC_Key historical statistics...")

        # Build index mapping for chronology
        build_to_idx = {build_id: idx for idx, build_id in enumerate(self.build_chronology)}

        # Group by TC_Key
        grouped = df.groupby('TC_Key')

        for tc_key, tc_df in grouped:
            # Sort by build chronology
            tc_df = tc_df.copy()
            tc_df['build_idx'] = tc_df['Build_ID'].map(build_to_idx)
            tc_df = tc_df.sort_values('build_idx')

            # Compute statistics
            results = tc_df['TE_Test_Result'].values

            # Total counts
            total_executions = len(results)
            total_failures = (results != 'Pass').sum()

            # Failure rate
            failure_rate = total_failures / total_executions if total_executions > 0 else 0.0

            # Recent failure rate (last N builds)
            recent_results = results[-self.recent_window:] if len(results) >= self.recent_window else results
            recent_failures = (recent_results != 'Pass').sum()
            recent_failure_rate = recent_failures / len(recent_results) if len(recent_results) > 0 else 0.0

            # Flakiness rate (state transitions)
            transitions = 0
            if len(results) > 1:
                for i in range(len(results) - 1):
                    prev_pass = (results[i] == 'Pass')
                    curr_pass = (results[i + 1] == 'Pass')
                    if prev_pass != curr_pass:
                        transitions += 1
                flakiness_rate = transitions / (len(results) - 1)
            else:
                flakiness_rate = 0.0

            # Store history
            self.tc_history[tc_key] = {
                'total_executions': total_executions,
                'total_failures': total_failures,
                'failure_rate': failure_rate,
                'recent_failure_rate': recent_failure_rate,
                'flakiness_rate': flakiness_rate,
                'first_build_idx': tc_df['build_idx'].min(),
                'last_build_idx': tc_df['build_idx'].max(),
                'build_history': tc_df['build_idx'].tolist(),
                'result_history': results.tolist()
            }

        logger.info(f"Computed history for {len(self.tc_history)} test cases")

    def _compute_first_appearances(self, df: pd.DataFrame) -> None:
        """
        Store the first build index where each TC_Key appears.
        """
        build_to_idx = {build_id: idx for idx, build_id in enumerate(self.build_chronology)}

        for tc_key, history in self.tc_history.items():
            self.tc_first_appearance[tc_key] = history['first_build_idx']

        logger.info(f"Stored first appearances for {len(self.tc_first_appearance)} test cases")

    def _compute_global_statistics(self, df_train: pd.DataFrame) -> None:
        """
        Compute global statistics from training features for conservative defaults.

        This is used to provide reasonable values for tests without history,
        rather than using zero which implies "never fails".
        """
        logger.info("Computing global statistics for conservative defaults...")

        # Extract features from training data
        train_features = []
        for idx, row in df_train.iterrows():
            tc_key = row['TC_Key']
            build_id = row['Build_ID']

            # Extract features (using training history)
            phylo_features = self._extract_phylogenetic_features(tc_key, build_id, is_test=False)
            struct_features = self._extract_structural_features(row)
            feature_vector = phylo_features + struct_features
            train_features.append(feature_vector)

        train_features = np.array(train_features, dtype=np.float32)

        # Compute statistics
        self.feature_means = np.mean(train_features, axis=0)
        self.feature_medians = np.median(train_features, axis=0)
        self.feature_stds = np.std(train_features, axis=0)

        logger.info(f"  Feature means: {self.feature_means}")
        logger.info(f"  Feature medians: {self.feature_medians}")
        logger.info(f"  Feature stds: {self.feature_stds}")

    def _extract_phylogenetic_features(self,
                                       tc_key: str,
                                       build_id: str,
                                       is_test: bool) -> List[float]:
        """
        Extract phylogenetic features for a single test case execution.

        Returns:
            [test_age, failure_rate, recent_failure_rate, flakiness_rate]
        """
        # Get current build index (O(1) dict lookup)
        if build_id in self._build_to_idx:
            current_build_idx = self._build_to_idx[build_id]
        else:
            # Build not in training chronology (new build in test set)
            current_build_idx = len(self.build_chronology)

        # Check if we have history for this TC_Key
        if tc_key in self.tc_history:
            history = self.tc_history[tc_key]

            # 1. Test Age: builds since first appearance
            first_build_idx = history['first_build_idx']
            test_age = current_build_idx - first_build_idx

            # 2. Failure Rate: historical failure rate
            failure_rate = history['failure_rate']

            # 3. Recent Failure Rate
            recent_failure_rate = history['recent_failure_rate']

            # 4. Flakiness Rate
            flakiness_rate = history['flakiness_rate']
        else:
            # New test case (not seen in training)
            # Use CONSERVATIVE DEFAULTS (not zero!) to avoid bias
            test_age = 0.0  # New test

            if self.feature_means is not None:
                # Use population statistics (more realistic than zero)
                failure_rate = float(self.feature_means[1])  # Population avg failure rate
                recent_failure_rate = float(self.feature_means[2])  # Population avg recent rate
                flakiness_rate = float(self.feature_medians[3])  # Median flakiness (usually low)
            else:
                # Fallback if statistics not computed (shouldn't happen after fit)
                logger.warning("Global statistics not computed! Using zeros as fallback.")
                failure_rate = 0.0
                recent_failure_rate = 0.0
                flakiness_rate = 0.0

        return [test_age, failure_rate, recent_failure_rate, flakiness_rate]

    def _extract_structural_features(self, row: pd.Series) -> List[float]:
        """
        Extract structural features for a single test case execution.

        Returns:
            [commit_count, test_novelty]
        """
        # 1. Commit Count: number of unique commits + CRs
        commit_count = self._count_commits(row)

        # 2. Test Novelty: is this the first appearance?
        tc_key = row['TC_Key']
        build_id = row['Build_ID']

        if tc_key in self.tc_first_appearance:
            first_build_idx = self.tc_first_appearance[tc_key]
            if build_id in self._build_to_idx:
                current_build_idx = self._build_to_idx[build_id]
            else:
                current_build_idx = len(self.build_chronology)

            test_novelty = 1.0 if current_build_idx == first_build_idx else 0.0
        else:
            # New test case (not in training history)
            test_novelty = 1.0

        return [commit_count, test_novelty]

    def _count_commits(self, row: pd.Series) -> float:
        """
        Count unique commits and CRs for a test execution.

        Reuses logic from src/evaluation/apfd.py:count_total_commits()
        """
        total_commits = set()

        # Count commits from 'commit' column
        if 'commit' in row.index and pd.notna(row['commit']):
            commit_str = row['commit']
            try:
                commits = ast.literal_eval(str(commit_str))
                if isinstance(commits, list):
                    total_commits.update(commits)
                else:
                    total_commits.add(str(commit_str))
            except:
                total_commits.add(str(commit_str))

        # Count CRs
        for cr_col in ['CR', 'CR_y']:
            if cr_col in row.index and pd.notna(row[cr_col]):
                cr_str = row[cr_col]
                try:
                    crs = ast.literal_eval(str(cr_str))
                    if isinstance(crs, list):
                        for cr in crs:
                            total_commits.add(f"CR_{cr}")
                    else:
                        total_commits.add(f"CR_{cr_str}")
                except:
                    total_commits.add(f"CR_{cr_str}")

        return float(max(len(total_commits), 1))

    def get_feature_names(self) -> List[str]:
        """
        Get feature names in the order they appear in the feature vector.

        Returns:
            List of feature names
        """
        return [
            'test_age',
            'failure_rate',
            'recent_failure_rate',
            'flakiness_rate',
            'commit_count',
            'test_novelty'
        ]

    def get_feature_stats(self, feature_matrix: np.ndarray) -> pd.DataFrame:
        """
        Get statistics for extracted features.

        Args:
            feature_matrix: Output of transform()

        Returns:
            DataFrame with feature statistics
        """
        feature_names = self.get_feature_names()

        stats = []
        for i, name in enumerate(feature_names):
            stats.append({
                'feature': name,
                'mean': feature_matrix[:, i].mean(),
                'std': feature_matrix[:, i].std(),
                'min': feature_matrix[:, i].min(),
                'max': feature_matrix[:, i].max(),
                'median': np.median(feature_matrix[:, i])
            })

        return pd.DataFrame(stats)

    def get_imputation_mask(self, tc_keys: List[str]) -> np.ndarray:
        """
        Determine which test cases need imputation (no history or insufficient history).

        Args:
            tc_keys: List of TC_Key strings

        Returns:
            needs_imputation: Boolean array indicating which tests need imputation
        """
        needs_imputation = np.array([
            (tc_key not in self.tc_history) or
            (self.tc_history[tc_key]['total_executions'] < self.min_history)
            for tc_key in tc_keys
        ])

        return needs_imputation

    def save_history(self, filepath: str) -> None:
        """
        Save the computed historical statistics to disk.

        Args:
            filepath: Path to save the history (pickle format)
        """
        import pickle

        state = {
            'tc_history': self.tc_history,
            'build_chronology': self.build_chronology,
            '_build_to_idx': self._build_to_idx,
            'tc_first_appearance': self.tc_first_appearance,
            'recent_window': self.recent_window,
            'min_history': self.min_history,
            'feature_means': self.feature_means,
            'feature_medians': self.feature_medians,
            'feature_stds': self.feature_stds
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Saved historical state to {filepath}")

    def load_history(self, filepath: str) -> 'StructuralFeatureExtractor':
        """
        Load previously computed historical statistics from disk.

        Args:
            filepath: Path to load the history from

        Returns:
            self (for method chaining)
        """
        import pickle

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.tc_history = state['tc_history']
        self.build_chronology = state['build_chronology']
        self._build_to_idx = state.get('_build_to_idx',
            {bid: idx for idx, bid in enumerate(self.build_chronology)})
        self.tc_first_appearance = state['tc_first_appearance']
        self.recent_window = state.get('recent_window', 5)
        self.min_history = state.get('min_history', 2)
        self.feature_means = state.get('feature_means', None)
        self.feature_medians = state.get('feature_medians', None)
        self.feature_stds = state.get('feature_stds', None)

        logger.info(f"Loaded historical state from {filepath}")
        logger.info(f"  {len(self.tc_history)} test cases")
        logger.info(f"  {len(self.build_chronology)} builds in chronology")

        if self.feature_means is not None:
            logger.info(f"  Global statistics loaded: means={self.feature_means}")

        return self


def extract_structural_features(df_train: pd.DataFrame,
                                df_val: Optional[pd.DataFrame] = None,
                                df_test: Optional[pd.DataFrame] = None,
                                recent_window: int = 5,
                                cache_path: Optional[str] = None) -> Tuple[np.ndarray, ...]:
    """
    Convenience function to extract structural features for train/val/test splits.

    Args:
        df_train: Training DataFrame
        df_val: Validation DataFrame (optional)
        df_test: Test DataFrame (optional)
        recent_window: Window size for recent failure rate
        cache_path: Path to cache/load extractor state

    Returns:
        Tuple of feature matrices: (train_features, val_features, test_features)
        If val or test are None, corresponding output is None

    Example:
        >>> train_features, val_features, test_features = extract_structural_features(
        ...     df_train, df_val, df_test,
        ...     cache_path='cache/structural_extractor.pkl'
        ... )
    """
    logger.info("="*70)
    logger.info("EXTRACTING STRUCTURAL FEATURES")
    logger.info("="*70)

    # Initialize extractor
    extractor = StructuralFeatureExtractor(recent_window=recent_window, verbose=True)

    # Load or fit
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached extractor from {cache_path}")
        extractor.load_history(cache_path)
    else:
        logger.info("Fitting extractor on training data...")
        extractor.fit(df_train)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            extractor.save_history(cache_path)

    # Transform splits
    logger.info("\nTransforming training data...")
    train_features = extractor.transform(df_train, is_test=False)

    val_features = None
    if df_val is not None:
        logger.info("\nTransforming validation data...")
        val_features = extractor.transform(df_val, is_test=True)

    test_features = None
    if df_test is not None:
        logger.info("\nTransforming test data...")
        test_features = extractor.transform(df_test, is_test=True)

    # Print summary statistics
    logger.info("\n" + "="*70)
    logger.info("STRUCTURAL FEATURE SUMMARY")
    logger.info("="*70)
    logger.info("\nFeature Statistics (Training Set):")
    stats_df = extractor.get_feature_stats(train_features)
    logger.info("\n" + stats_df.to_string(index=False))
    logger.info("="*70)

    return train_features, val_features, test_features


# For backwards compatibility and ease of import
__all__ = [
    'StructuralFeatureExtractor',
    'extract_structural_features'
]
