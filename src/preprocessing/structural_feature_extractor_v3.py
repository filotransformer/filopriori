"""
Structural Feature Extractor V3 - DeepOrder-Enhanced Version

This version adds critical features that DeepOrder uses for better TCP performance:

NEW Features (vs V2.5):
1. last_verdict (0/1) - Binary result of last execution
2. time_since_failure (normalized) - Builds since last failure
3. weighted_failure_rate - Failure rate with exponential decay
4. execution_frequency - How often the test runs

IMPROVEMENTS:
- Extended recent_window from 5 to 10 (configurable)
- Decay-weighted historical features
- Better handling of new test cases

Selected Features (14 total):
- From V2.5 (10): test_age, failure_rate, recent_failure_rate, flakiness_rate,
                  consecutive_failures, max_consecutive_failures, failure_trend,
                  commit_count, test_novelty, cr_count
- NEW (4): last_verdict, time_since_failure, weighted_failure_rate, execution_frequency

Target: Surpass DeepOrder (0.6500 APFD) while maintaining Filo-Priori contributions.

Author: Filo-Priori V11 Team
Date: 2025-11-29
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import ast
import logging
import math

logger = logging.getLogger(__name__)


class StructuralFeatureExtractorV3:
    """
    Enhanced Structural Feature Extractor with DeepOrder-style features.

    Key improvements:
    1. Adds explicit last_verdict (0/1)
    2. Adds normalized time_since_failure
    3. Adds weighted_failure_rate with exponential decay
    4. Configurable extended history window (default 10)
    """

    # Final selected features (14 total)
    SELECTED_FEATURE_NAMES = [
        # Historical performance (6)
        'test_age',                    # How old is this test
        'failure_rate',                # Overall failure rate
        'recent_failure_rate',         # Recent failure rate (last N builds)
        'weighted_failure_rate',       # NEW: Decay-weighted failure rate
        'flakiness_rate',              # Pass/fail transitions
        'failure_trend',               # Recent - overall (positive = getting worse)

        # Recency features (4)
        'last_verdict',                # NEW: 0=fail, 1=pass
        'time_since_failure',          # NEW: Normalized builds since last failure
        'consecutive_failures',        # Current failure streak
        'max_consecutive_failures',    # Max historical streak

        # Activity features (2)
        'execution_frequency',         # NEW: Executions / builds span
        'test_novelty',                # Is this a new test?

        # Change features (2)
        'commit_count',                # Recent commits
        'cr_count',                    # Change requests
    ]

    def __init__(self,
                 recent_window: int = 10,       # Extended from 5 to 10
                 very_recent_window: int = 3,   # Slightly extended
                 medium_term_window: int = 20,  # Extended
                 min_history: int = 2,
                 decay_alpha: float = 0.1,      # Exponential decay factor
                 max_time_since_failure: float = 50.0,  # For normalization
                 verbose: bool = True):
        """
        Initialize V3 extractor.

        Args:
            recent_window: Window for recent failure rate (default: 10, up from 5)
            very_recent_window: Window for very recent rate (default: 3)
            medium_term_window: Window for medium term rate (default: 20)
            min_history: Minimum executions for reliable features
            decay_alpha: Alpha for exponential decay (higher = faster decay)
            max_time_since_failure: Max value for normalization (default: 50 builds)
            verbose: Enable verbose logging
        """
        self.recent_window = recent_window
        self.very_recent_window = very_recent_window
        self.medium_term_window = medium_term_window
        self.min_history = min_history
        self.decay_alpha = decay_alpha
        self.max_time_since_failure = max_time_since_failure
        self.verbose = verbose

        # Cache for historical statistics
        self.tc_history: Dict[str, Dict] = {}
        self.build_chronology: List[str] = []
        self.tc_first_appearance: Dict[str, int] = {}

        # Global statistics for conservative defaults
        self.feature_means: Optional[np.ndarray] = None
        self.feature_medians: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

        # Number of output features
        self.num_features = len(self.SELECTED_FEATURE_NAMES)

        logger.info(f"Initialized StructuralFeatureExtractorV3 with:")
        logger.info(f"  recent_window={recent_window} (extended from 5)")
        logger.info(f"  very_recent_window={very_recent_window}")
        logger.info(f"  medium_term_window={medium_term_window}")
        logger.info(f"  decay_alpha={decay_alpha}")
        logger.info(f"  → {self.num_features} selected features")

    def fit(self, df_train: pd.DataFrame) -> 'StructuralFeatureExtractorV3':
        """
        Fit the extractor on training data.

        WARNING: This method computes features using the ENTIRE training set at once,
        which causes look-ahead bias for training rows. Use fit_transform_temporal()
        instead to avoid this issue during training.

        This method is kept for backward compatibility and is safe to call before
        transform() on val/test data (since those are temporally after training).
        """
        logger.info("Fitting StructuralFeatureExtractorV3 on training data...")
        logger.info(f"Training data shape: {df_train.shape}")

        # 1. Establish build chronology
        self._establish_chronology(df_train)

        # 2. Compute per-TC history with enhanced features
        self._compute_tc_history_v3(df_train)

        # 3. Store first appearances
        self._compute_first_appearances(df_train)

        # 4. Compute global statistics
        self._compute_global_statistics(df_train)

        logger.info(f"Fitted extractor on {len(self.tc_history)} unique test cases")
        logger.info(f"Build chronology spans {len(self.build_chronology)} builds")
        logger.info(f"Extracting {self.num_features} features per test case")

        return self

    def transform(self, df: pd.DataFrame, is_test: bool = False) -> np.ndarray:
        """
        Transform DataFrame into feature vectors using pre-computed history.

        For val/test data after fit() or fit_transform_temporal(), this is correct
        because the history was accumulated from training data only.

        Returns:
            feature_matrix: np.ndarray of shape [N, 14]
        """
        logger.info(f"Transforming {len(df)} samples into {self.num_features} features...")

        features = []

        for idx, row in df.iterrows():
            tc_key = row['TC_Key']
            build_id = row['Build_ID']

            feature_vector = self._extract_features_v3(tc_key, build_id, row, is_test)
            features.append(feature_vector)

            if self.verbose and len(features) % 10000 == 0:
                logger.info(f"  Processed {len(features)}/{len(df)} samples...")

        feature_matrix = np.array(features, dtype=np.float32)

        logger.info(f"Extracted feature matrix: {feature_matrix.shape}")

        return feature_matrix

    def transform_temporal(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform DataFrame with incremental history updates per build.

        Processes builds in chronological order. For each build:
        1. Compute features using accumulated history (from prior builds only)
        2. Update history with current build's results

        This is the correct method for val/test when you need temporal consistency
        within the split (e.g., early val builds only see train history, later val
        builds see train + earlier val history).

        The tc_history is updated IN-PLACE after this call.

        Returns:
            feature_matrix: np.ndarray of shape [N, 14]
        """
        logger.info(f"Transform temporal: {len(df)} samples, {self.num_features} features...")

        # Establish chronology for this split (append to existing)
        if 'Build_Test_Start_Date' in df.columns:
            build_dates = df.groupby('Build_ID')['Build_Test_Start_Date'].first().sort_values()
            sorted_builds = build_dates.index.tolist()
        else:
            sorted_builds = list(dict.fromkeys(df['Build_ID'].values))

        # Add new builds to chronology
        existing_builds = set(self.build_chronology)
        for b in sorted_builds:
            if b not in existing_builds:
                self.build_chronology.append(b)

        build_to_idx = {build_id: idx for idx, build_id in enumerate(self.build_chronology)}

        # Pre-allocate feature matrix
        feature_matrix = np.zeros((len(df), self.num_features), dtype=np.float32)
        # Map original df index to position
        idx_to_pos = {idx: pos for pos, idx in enumerate(df.index)}

        for build_id in sorted_builds:
            build_mask = df['Build_ID'] == build_id
            build_df = df[build_mask]
            build_idx = build_to_idx[build_id]

            # 1. Extract features using CURRENT accumulated history
            for orig_idx, row in build_df.iterrows():
                tc_key = row['TC_Key']
                feature_vector = self._extract_features_v3(tc_key, build_id, row, is_test=True)
                feature_matrix[idx_to_pos[orig_idx]] = feature_vector

            # 2. Update history with this build's results
            for _, row in build_df.iterrows():
                tc_key = row['TC_Key']
                result = row['TE_Test_Result']
                self._update_running_history(tc_key, result, build_idx)

        logger.info(f"Transform temporal complete: {feature_matrix.shape}")
        return feature_matrix

    def fit_transform_temporal(self, df_train: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform training data WITHOUT look-ahead bias.

        Processes builds in chronological order. For each build:
        1. Compute features using accumulated history from PRIOR builds only
        2. Update running history with current build's results

        After processing all training builds, tc_history contains the full
        training history - correct for use with transform() or transform_temporal()
        on val/test data.

        Returns:
            feature_matrix: np.ndarray of shape [N, 14]
        """
        logger.info("fit_transform_temporal: StructuralFeatureExtractorV3 (NO look-ahead bias)")
        logger.info(f"Training data shape: {df_train.shape}")

        # 1. Establish build chronology
        self._establish_chronology(df_train)
        build_to_idx = {build_id: idx for idx, build_id in enumerate(self.build_chronology)}

        # 2. Reset history - will be built incrementally
        self.tc_history = {}
        self.tc_first_appearance = {}

        # 3. Process builds one by one in chronological order
        sorted_builds = self.build_chronology
        logger.info(f"Processing {len(sorted_builds)} builds chronologically...")

        # Pre-allocate feature matrix
        feature_matrix = np.zeros((len(df_train), self.num_features), dtype=np.float32)
        idx_to_pos = {idx: pos for pos, idx in enumerate(df_train.index)}

        for bi, build_id in enumerate(sorted_builds):
            build_mask = df_train['Build_ID'] == build_id
            build_df = df_train[build_mask]

            if len(build_df) == 0:
                continue

            build_idx = build_to_idx[build_id]

            # STEP A: Extract features using ONLY accumulated history (before this build)
            for orig_idx, row in build_df.iterrows():
                tc_key = row['TC_Key']
                feature_vector = self._extract_features_v3(tc_key, build_id, row, is_test=False)
                feature_matrix[idx_to_pos[orig_idx]] = feature_vector

            # STEP B: Update running history with this build's results
            for _, row in build_df.iterrows():
                tc_key = row['TC_Key']
                result = row['TE_Test_Result']
                self._update_running_history(tc_key, result, build_idx)

            if self.verbose and (bi + 1) % 200 == 0:
                logger.info(f"  Processed {bi+1}/{len(sorted_builds)} builds, "
                           f"{len(self.tc_history)} TCs in history")

        # 4. Store first appearances from accumulated history
        for tc_key, history in self.tc_history.items():
            self.tc_first_appearance[tc_key] = history['first_build_idx']

        # 5. Compute global statistics from the feature matrix
        self.feature_means = np.mean(feature_matrix, axis=0)
        self.feature_medians = np.median(feature_matrix, axis=0)
        self.feature_stds = np.std(feature_matrix, axis=0)

        logger.info(f"fit_transform_temporal complete:")
        logger.info(f"  Feature matrix: {feature_matrix.shape}")
        logger.info(f"  TC history: {len(self.tc_history)} test cases")
        logger.info(f"  Build chronology: {len(self.build_chronology)} builds")
        logger.info(f"  Feature means: {self.feature_means}")

        return feature_matrix

    def fit_transform(self, df_train: pd.DataFrame) -> np.ndarray:
        """Fit and transform using temporal-safe method (no look-ahead bias)."""
        return self.fit_transform_temporal(df_train)

    # ==================== PRIVATE METHODS ====================

    def _establish_chronology(self, df: pd.DataFrame) -> None:
        """Establish chronological order of builds."""
        if 'Build_Test_Start_Date' in df.columns:
            build_dates = df.groupby('Build_ID')['Build_Test_Start_Date'].first().sort_values()
            self.build_chronology = build_dates.index.tolist()
            logger.info("Build chronology established using Build_Test_Start_Date")
        else:
            self.build_chronology = df['Build_ID'].unique().tolist()
            logger.warning("Build_Test_Start_Date not found, using order of appearance")

        logger.info(f"Chronology spans {len(self.build_chronology)} builds")

    def _compute_tc_history_v3(self, df: pd.DataFrame) -> None:
        """
        Compute enhanced historical statistics including DeepOrder features.
        """
        logger.info("Computing enhanced per-TC history (V3)...")

        build_to_idx = {build_id: idx for idx, build_id in enumerate(self.build_chronology)}
        grouped = df.groupby('TC_Key')

        for tc_key, tc_df in grouped:
            # Sort by build chronology
            tc_df = tc_df.copy()
            tc_df['build_idx'] = tc_df['Build_ID'].map(build_to_idx)
            tc_df = tc_df.sort_values('build_idx')

            results = tc_df['TE_Test_Result'].values
            build_indices = tc_df['build_idx'].values

            # === BASIC COUNTS ===
            total_executions = len(results)
            total_failures = (results != 'Pass').sum()
            total_passes = (results == 'Pass').sum()

            # === FAILURE RATES ===
            failure_rate = total_failures / total_executions if total_executions > 0 else 0.0

            # Recent failure rate (extended window)
            recent_results = results[-self.recent_window:] if len(results) >= self.recent_window else results
            recent_failures = (recent_results != 'Pass').sum()
            recent_failure_rate = recent_failures / len(recent_results) if len(recent_results) > 0 else 0.0

            # === NEW: WEIGHTED FAILURE RATE WITH EXPONENTIAL DECAY ===
            weighted_failure_rate = self._compute_weighted_failure_rate(results, build_indices)

            # === FLAKINESS ===
            flakiness_rate = self._compute_flakiness(results)

            # === STREAKS ===
            consecutive_failures, consecutive_passes = self._compute_current_streaks(results)
            max_consecutive_failures, max_consecutive_passes = self._compute_max_streaks(results)

            # === NEW: LAST VERDICT (0=fail, 1=pass) ===
            last_verdict = 1.0 if results[-1] == 'Pass' else 0.0

            # === NEW: TIME SINCE FAILURE (normalized) ===
            time_since_failure = self._compute_time_since_failure(results, build_indices)

            # === EXECUTION FREQUENCY ===
            first_build_idx = build_indices[0] if len(build_indices) > 0 else 0
            current_build_idx = build_indices[-1] if len(build_indices) > 0 else 0
            builds_span = current_build_idx - first_build_idx + 1
            execution_frequency = total_executions / builds_span if builds_span > 0 else 1.0

            # === TRENDS ===
            failure_trend = recent_failure_rate - failure_rate  # positive = getting worse

            # Store ALL statistics
            self.tc_history[tc_key] = {
                # Basic
                'total_executions': total_executions,
                'total_failures': total_failures,
                'total_passes': total_passes,

                # Rates
                'failure_rate': failure_rate,
                'recent_failure_rate': recent_failure_rate,
                'weighted_failure_rate': weighted_failure_rate,
                'flakiness_rate': flakiness_rate,
                'failure_trend': failure_trend,

                # Recency (NEW)
                'last_verdict': last_verdict,
                'time_since_failure': time_since_failure,
                'consecutive_failures': consecutive_failures,
                'consecutive_passes': consecutive_passes,
                'max_consecutive_failures': max_consecutive_failures,

                # Activity
                'execution_frequency': execution_frequency,

                # Build info
                'first_build_idx': first_build_idx,
                'last_build_idx': current_build_idx,
                'result_history': results.tolist(),
                'build_history': build_indices.tolist()
            }

        logger.info(f"✓ Computed enhanced history for {len(self.tc_history)} test cases")

    def _compute_weighted_failure_rate(self, results: np.ndarray, build_indices: np.ndarray) -> float:
        """
        Compute failure rate with exponential decay.

        More recent failures weighted higher than older ones.
        Formula: sum(weight * is_fail) / sum(weight)
        where weight = exp(-alpha * (current - execution))
        """
        if len(results) == 0:
            return 0.0

        current_idx = build_indices[-1]

        weighted_failures = 0.0
        total_weight = 0.0

        for i, (result, build_idx) in enumerate(zip(results, build_indices)):
            age = current_idx - build_idx
            weight = math.exp(-self.decay_alpha * age)

            is_fail = 1.0 if result != 'Pass' else 0.0
            weighted_failures += weight * is_fail
            total_weight += weight

        return weighted_failures / total_weight if total_weight > 0 else 0.0

    def _compute_time_since_failure(self, results: np.ndarray, build_indices: np.ndarray) -> float:
        """
        Compute normalized time since last failure.

        Returns value in [0, 1] where:
        - 0 = just failed
        - 1 = never failed or failed very long ago
        """
        if len(results) == 0:
            return 1.0  # No history = assume stable

        current_idx = build_indices[-1]

        # Find last failure
        last_failure_idx = None
        for i in range(len(results) - 1, -1, -1):
            if results[i] != 'Pass':
                last_failure_idx = build_indices[i]
                break

        if last_failure_idx is None:
            # Never failed
            return 1.0

        time_since = current_idx - last_failure_idx

        # Normalize to [0, 1]
        normalized = min(time_since / self.max_time_since_failure, 1.0)

        return normalized

    def _update_running_history(self, tc_key: str, result: str, build_idx: int) -> None:
        """
        Incrementally update running history for a single test case execution.

        This is used by fit_transform_temporal() and transform_temporal() to
        maintain history without look-ahead bias.
        """
        is_fail = result != 'Pass'

        if tc_key not in self.tc_history:
            # First time seeing this TC
            self.tc_history[tc_key] = {
                'total_executions': 1,
                'total_failures': 1 if is_fail else 0,
                'total_passes': 0 if is_fail else 1,
                'failure_rate': 1.0 if is_fail else 0.0,
                'recent_failure_rate': 1.0 if is_fail else 0.0,
                'weighted_failure_rate': 1.0 if is_fail else 0.0,
                'flakiness_rate': 0.0,
                'failure_trend': 0.0,
                'last_verdict': 0.0 if is_fail else 1.0,
                'time_since_failure': 0.0 if is_fail else 1.0,
                'consecutive_failures': 1 if is_fail else 0,
                'consecutive_passes': 0 if is_fail else 1,
                'max_consecutive_failures': 1 if is_fail else 0,
                'execution_frequency': 1.0,
                'first_build_idx': build_idx,
                'last_build_idx': build_idx,
                'result_history': [result],
                'build_history': [build_idx],
            }
            self.tc_first_appearance[tc_key] = build_idx
        else:
            h = self.tc_history[tc_key]

            # Update basic counts
            h['total_executions'] += 1
            if is_fail:
                h['total_failures'] += 1
            else:
                h['total_passes'] += 1

            # Append to history
            h['result_history'].append(result)
            h['build_history'].append(build_idx)

            results_arr = np.array(h['result_history'])
            build_indices_arr = np.array(h['build_history'])

            # Update rates
            h['failure_rate'] = h['total_failures'] / h['total_executions']

            recent_results = results_arr[-self.recent_window:]
            recent_failures = (recent_results != 'Pass').sum()
            h['recent_failure_rate'] = recent_failures / len(recent_results)

            h['weighted_failure_rate'] = self._compute_weighted_failure_rate(
                results_arr, build_indices_arr
            )

            # Update flakiness
            h['flakiness_rate'] = self._compute_flakiness(results_arr)

            # Update trend
            h['failure_trend'] = h['recent_failure_rate'] - h['failure_rate']

            # Update recency
            h['last_verdict'] = 0.0 if is_fail else 1.0
            h['time_since_failure'] = self._compute_time_since_failure(
                results_arr, build_indices_arr
            )

            # Update streaks
            h['consecutive_failures'], h['consecutive_passes'] = \
                self._compute_current_streaks(results_arr)
            max_f, _ = self._compute_max_streaks(results_arr)
            h['max_consecutive_failures'] = max_f

            # Update execution frequency
            builds_span = build_idx - h['first_build_idx'] + 1
            h['execution_frequency'] = h['total_executions'] / builds_span if builds_span > 0 else 1.0

            h['last_build_idx'] = build_idx

    def _compute_flakiness(self, results: np.ndarray) -> float:
        """Compute flakiness rate (pass/fail transitions)."""
        if len(results) <= 1:
            return 0.0

        transitions = 0
        for i in range(len(results) - 1):
            if (results[i] == 'Pass') != (results[i+1] == 'Pass'):
                transitions += 1

        return transitions / (len(results) - 1)

    def _compute_current_streaks(self, results: np.ndarray) -> Tuple[int, int]:
        """Compute current consecutive failure/pass streaks."""
        if len(results) == 0:
            return 0, 0

        current_streak_failures = 0
        current_streak_passes = 0

        last_result = results[-1]
        if last_result == 'Pass':
            for i in range(len(results) - 1, -1, -1):
                if results[i] == 'Pass':
                    current_streak_passes += 1
                else:
                    break
        else:
            for i in range(len(results) - 1, -1, -1):
                if results[i] != 'Pass':
                    current_streak_failures += 1
                else:
                    break

        return current_streak_failures, current_streak_passes

    def _compute_max_streaks(self, results: np.ndarray) -> Tuple[int, int]:
        """Compute maximum consecutive failure/pass streaks."""
        if len(results) == 0:
            return 0, 0

        max_failures = 0
        max_passes = 0
        current_failures = 0
        current_passes = 0

        for result in results:
            if result == 'Pass':
                current_passes += 1
                max_failures = max(max_failures, current_failures)
                current_failures = 0
            else:
                current_failures += 1
                max_passes = max(max_passes, current_passes)
                current_passes = 0

        max_failures = max(max_failures, current_failures)
        max_passes = max(max_passes, current_passes)

        return max_failures, max_passes

    def _compute_first_appearances(self, df: pd.DataFrame) -> None:
        """Store first build index where each TC_Key appears."""
        for tc_key, history in self.tc_history.items():
            self.tc_first_appearance[tc_key] = history['first_build_idx']

        logger.info(f"✓ Stored first appearances for {len(self.tc_first_appearance)} test cases")

    def _compute_global_statistics(self, df_train: pd.DataFrame) -> None:
        """Compute global statistics for conservative defaults."""
        logger.info("Computing global statistics for new features...")

        train_features = []
        for idx, row in df_train.iterrows():
            tc_key = row['TC_Key']
            build_id = row['Build_ID']
            feature_vector = self._extract_features_v3(tc_key, build_id, row, is_test=False)
            train_features.append(feature_vector)

        train_features = np.array(train_features, dtype=np.float32)

        self.feature_means = np.mean(train_features, axis=0)
        self.feature_medians = np.median(train_features, axis=0)
        self.feature_stds = np.std(train_features, axis=0)

        logger.info(f"  Feature means: {self.feature_means}")
        logger.info(f"  Feature stds: {self.feature_stds}")

    def _extract_features_v3(self, tc_key: str, build_id: str,
                             row: pd.Series, is_test: bool) -> List[float]:
        """
        Extract 14 selected features.

        Order:
        0. test_age
        1. failure_rate
        2. recent_failure_rate
        3. weighted_failure_rate (NEW)
        4. flakiness_rate
        5. failure_trend
        6. last_verdict (NEW)
        7. time_since_failure (NEW)
        8. consecutive_failures
        9. max_consecutive_failures
        10. execution_frequency (NEW)
        11. test_novelty
        12. commit_count
        13. cr_count
        """
        # Get current build index
        if build_id in self.build_chronology:
            current_build_idx = self.build_chronology.index(build_id)
        else:
            current_build_idx = len(self.build_chronology)

        if tc_key in self.tc_history:
            h = self.tc_history[tc_key]

            # Test age
            test_age = float(current_build_idx - h['first_build_idx'])

            # Test novelty
            test_novelty = 1.0 if current_build_idx == h['first_build_idx'] else 0.0

            # Commit/CR counts
            commit_count, cr_count = self._count_commits(row)

            return [
                test_age,                                   # 0
                float(h['failure_rate']),                   # 1
                float(h['recent_failure_rate']),            # 2
                float(h['weighted_failure_rate']),          # 3 (NEW)
                float(h['flakiness_rate']),                 # 4
                float(h['failure_trend']),                  # 5
                float(h['last_verdict']),                   # 6 (NEW)
                float(h['time_since_failure']),             # 7 (NEW)
                float(h['consecutive_failures']),           # 8
                float(h['max_consecutive_failures']),       # 9
                float(h['execution_frequency']),            # 10 (NEW)
                test_novelty,                               # 11
                commit_count,                               # 12
                cr_count                                    # 13
            ]
        else:
            # New test case - use conservative defaults
            commit_count, cr_count = self._count_commits(row)

            if self.feature_means is not None:
                return [
                    0.0,                                    # test_age (new)
                    float(self.feature_means[1]),           # failure_rate
                    float(self.feature_means[2]),           # recent_failure_rate
                    float(self.feature_means[3]),           # weighted_failure_rate
                    float(self.feature_medians[4]),         # flakiness_rate
                    0.0,                                    # failure_trend
                    1.0,                                    # last_verdict (assume pass)
                    1.0,                                    # time_since_failure (no history)
                    0.0,                                    # consecutive_failures
                    0.0,                                    # max_consecutive_failures
                    float(self.feature_medians[10]),        # execution_frequency
                    1.0,                                    # test_novelty (is new)
                    commit_count,                           # commit_count
                    cr_count                                # cr_count
                ]
            else:
                return [
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 1.0, 0.0, 0.0, 0.5, 1.0,
                    commit_count, cr_count
                ]

    def _count_commits(self, row: pd.Series) -> Tuple[float, float]:
        """Count commits and CRs."""
        commits = set()
        crs = set()

        # Count commits
        if 'commit' in row.index and pd.notna(row['commit']):
            commit_str = row['commit']
            try:
                commit_list = ast.literal_eval(str(commit_str))
                if isinstance(commit_list, list):
                    commits.update(commit_list)
                else:
                    commits.add(str(commit_str))
            except:
                commits.add(str(commit_str))

        # Count CRs
        for cr_col in ['CR', 'CR_y']:
            if cr_col in row.index and pd.notna(row[cr_col]):
                cr_str = row[cr_col]
                try:
                    cr_list = ast.literal_eval(str(cr_str))
                    if isinstance(cr_list, list):
                        crs.update(cr_list)
                    else:
                        crs.add(str(cr_str))
                except:
                    crs.add(str(cr_str))

        return float(max(len(commits), 1)), float(len(crs))

    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self.SELECTED_FEATURE_NAMES.copy()

    def get_imputation_mask(self, tc_keys: List[str]) -> np.ndarray:
        """
        Determine which test cases need imputation (insufficient history).

        Args:
            tc_keys: List of test case keys

        Returns:
            Boolean array where True indicates the TC needs imputation
        """
        needs_imputation = np.array([
            (tc_key not in self.tc_history) or
            (self.tc_history[tc_key]['total_executions'] < self.min_history)
            for tc_key in tc_keys
        ])
        return needs_imputation

    def save_history(self, filepath: str) -> None:
        """Save computed statistics to disk."""
        import pickle
        import os

        state = {
            'tc_history': self.tc_history,
            'build_chronology': self.build_chronology,
            'tc_first_appearance': self.tc_first_appearance,
            'recent_window': self.recent_window,
            'very_recent_window': self.very_recent_window,
            'medium_term_window': self.medium_term_window,
            'min_history': self.min_history,
            'decay_alpha': self.decay_alpha,
            'max_time_since_failure': self.max_time_since_failure,
            'feature_means': self.feature_means,
            'feature_medians': self.feature_medians,
            'feature_stds': self.feature_stds,
            'version': 'v3'
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"✓ Saved V3 historical state to {filepath}")

    def load_history(self, filepath: str) -> 'StructuralFeatureExtractorV3':
        """Load previously computed statistics."""
        import pickle

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.tc_history = state['tc_history']
        self.build_chronology = state['build_chronology']
        self.tc_first_appearance = state['tc_first_appearance']
        self.recent_window = state.get('recent_window', 10)
        self.very_recent_window = state.get('very_recent_window', 3)
        self.medium_term_window = state.get('medium_term_window', 20)
        self.min_history = state.get('min_history', 2)
        self.decay_alpha = state.get('decay_alpha', 0.1)
        self.max_time_since_failure = state.get('max_time_since_failure', 50.0)
        self.feature_means = state.get('feature_means', None)
        self.feature_medians = state.get('feature_medians', None)
        self.feature_stds = state.get('feature_stds', None)

        version = state.get('version', 'unknown')
        logger.info(f"✓ Loaded V3 historical state from {filepath} (version: {version})")
        logger.info(f"  {len(self.tc_history)} test cases")
        logger.info(f"  {len(self.build_chronology)} builds")

        return self


# For easy import
__all__ = ['StructuralFeatureExtractorV3']
