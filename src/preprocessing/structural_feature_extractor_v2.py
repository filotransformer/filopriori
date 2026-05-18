"""
Structural Feature Extractor V2 for Filo-Priori V8

EXPANDED VERSION: 29 features (up from 6)

This module extracts RICH structural and phylogenetic features from historical data.

Features are divided into four categories:

1. TEMPORAL/HISTORY FEATURES (16 features):
   - test_age, execution_count, failure/pass counts
   - consecutive streaks, last failure/pass age
   - execution frequency, builds_since_change

2. RECENCY & TREND FEATURES (6 features):
   - failure_trend, recent/very_recent/medium_term rates
   - acceleration, deceleration_factor

3. BUILD/CHANGE FEATURES (4 features):
   - builds_affected, CR count, avg commits, commit surge

4. STABILITY/VOLATILITY FEATURES (3 features):
   - stability_score, pass_fail_ratio, recent_stability

Author: Claude Code - Filo-Priori V8 Enhanced
Date: 2025-11-14
Version: 2.0
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import ast
import logging

logger = logging.getLogger(__name__)


class StructuralFeatureExtractorV2:
    """
    ENHANCED Structural Feature Extractor with 29 features.

    Extracts rich temporal, trend, change, and stability features
    to improve test case prioritization ranking (APFD).
    """

    def __init__(self,
                 recent_window: int = 5,
                 very_recent_window: int = 2,
                 medium_term_window: int = 10,
                 min_history: int = 2,
                 verbose: bool = True):
        """
        Initialize the enhanced structural feature extractor.

        Args:
            recent_window: Window for recent_failure_rate (default: 5 builds)
            very_recent_window: Window for very_recent_failure_rate (default: 2 builds)
            medium_term_window: Window for medium_term_failure_rate (default: 10 builds)
            min_history: Minimum executions for reliable features
            verbose: Enable verbose logging
        """
        self.recent_window = recent_window
        self.very_recent_window = very_recent_window
        self.medium_term_window = medium_term_window
        self.min_history = min_history
        self.verbose = verbose

        # Cache for historical statistics
        self.tc_history: Dict[str, Dict] = {}
        self.build_chronology: List[str] = []
        self._build_id_to_idx: Dict[str, int] = {}  # O(1) lookup cache
        self.tc_first_appearance: Dict[str, int] = {}

        # Global statistics for conservative defaults
        self.feature_means: Optional[np.ndarray] = None
        self.feature_medians: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

        logger.info(f"Initialized StructuralFeatureExtractorV2 with:")
        logger.info(f"  recent_window={recent_window}")
        logger.info(f"  very_recent_window={very_recent_window}")
        logger.info(f"  medium_term_window={medium_term_window}")
        logger.info(f"  → 29 features total")

    def fit(self, df_train: pd.DataFrame) -> 'StructuralFeatureExtractorV2':
        """
        Fit the extractor on training data to learn historical patterns.

        Computes extensive statistics for each TC_Key including:
        - Execution counts and rates
        - Failure patterns and streaks
        - Recency and trends
        - Change activity
        - Stability metrics
        """
        logger.info("Fitting StructuralFeatureExtractorV2 on training data...")
        logger.info(f"Training data shape: {df_train.shape}")

        # 1. Establish build chronology
        self._establish_chronology(df_train)

        # 2. Compute extensive per-TC_Key historical statistics
        self._compute_tc_history_v2(df_train)

        # 3. Store first appearance information
        self._compute_first_appearances(df_train)

        # 4. Compute global statistics for conservative defaults
        self._compute_global_statistics(df_train)

        logger.info(f"✓ Fitted extractor on {len(self.tc_history)} unique test cases")
        logger.info(f"✓ Build chronology spans {len(self.build_chronology)} builds")
        logger.info(f"✓ Extracting 29 features per test case")

        return self

    def transform(self, df: pd.DataFrame, is_test: bool = False) -> np.ndarray:
        """
        Transform DataFrame into 29-dimensional structural feature vectors.

        Returns:
            feature_matrix: np.ndarray of shape [N, 29]
        """
        logger.info(f"Transforming {len(df)} samples into 29 features...")

        features = []

        for idx, row in df.iterrows():
            tc_key = row['TC_Key']
            build_id = row['Build_ID']

            # Extract EXPANDED phylogenetic features (20 features)
            phylo_features = self._extract_phylogenetic_features_v2(
                tc_key, build_id, is_test
            )

            # Extract EXPANDED structural features (9 features)
            struct_features = self._extract_structural_features_v2(row)

            # Combine: 20 + 9 = 29 features
            feature_vector = phylo_features + struct_features
            features.append(feature_vector)

            if self.verbose and len(features) % 10000 == 0:
                logger.info(f"  Processed {len(features)}/{len(df)} samples...")

        feature_matrix = np.array(features, dtype=np.float32)

        logger.info(f"✓ Extracted feature matrix: {feature_matrix.shape}")
        logger.info(f"  Feature means: {feature_matrix.mean(axis=0)[:10]}... (showing first 10)")
        logger.info(f"  Feature stds:  {feature_matrix.std(axis=0)[:10]}... (showing first 10)")

        return feature_matrix

    def fit_transform(self, df_train: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(df_train)
        return self.transform(df_train, is_test=False)


    def fit_transform_temporal(self, df_train: pd.DataFrame) -> 'np.ndarray':
        logger.info("Fitting and transforming sequentially (temporal mode)...")
        self.tc_history = {}
        self._establish_chronology(df_train)
        return self.transform_temporal(df_train)

    def transform_temporal(self, df: pd.DataFrame) -> 'np.ndarray':
        logger.info(f"Transforming {len(df)} samples with incremental updates...")
        
        # Sort builds chronologically
        if 'Build_Test_Start_Date' in df.columns:
            build_dates = df.groupby('Build_ID')['Build_Test_Start_Date'].first().sort_values()
            sorted_builds = build_dates.index.tolist()
        else:
            sorted_builds = df['Build_ID'].unique().tolist()
            
        existing_builds = set(self._build_id_to_idx)
        for b in sorted_builds:
            if b not in existing_builds:
                self._build_id_to_idx[b] = len(self.build_chronology)
                self.build_chronology.append(b)
                existing_builds.add(b)
                
        import numpy as np
        feature_matrix = np.zeros((len(df), 29), dtype=np.float32)
        idx_to_pos = {idx: pos for pos, idx in enumerate(df.index)}
        
        n_builds = len(sorted_builds)
        samples_done = 0
        for build_num, build_id in enumerate(sorted_builds):
            build_mask = df['Build_ID'] == build_id
            build_df = df[build_mask]

            # Extract features BEFORE updating history (no look-ahead)
            for orig_idx, row in build_df.iterrows():
                tc_key = row['TC_Key']
                phylo_features = self._extract_phylogenetic_features_v2(tc_key, build_id, is_test=True)
                struct_features = self._extract_structural_features_v2(row)
                feature_matrix[idx_to_pos[orig_idx]] = phylo_features + struct_features

            # Update history AFTER extraction
            self._update_history_incremental(build_df, build_id)

            samples_done += len(build_df)
            if (build_num + 1) % 5000 == 0 or build_num == n_builds - 1:
                logger.info(f"  Build {build_num+1}/{n_builds} ({samples_done}/{len(df)} samples)")
            
        # Compute global stats from the already-extracted feature matrix (no re-iteration)
        if self.feature_means is None:
            self.feature_means = np.mean(feature_matrix, axis=0)
            self.feature_medians = np.median(feature_matrix, axis=0)
            self.feature_stds = np.std(feature_matrix, axis=0)
            logger.info(f"  Computed global statistics from feature matrix")

        return feature_matrix

    def _update_history_incremental(self, df: pd.DataFrame, build_id: str) -> None:
        build_idx = self._build_id_to_idx[build_id]  # O(1) lookup, cached once per build
        for _, row in df.iterrows():
            tc_key = row['TC_Key']
            result = row['TE_Test_Result']
            is_fail = result == 'Fail'

            if tc_key not in self.tc_history:
                self.tc_history[tc_key] = {
                    'total_executions': 0,
                    'total_failures': 0,
                    'total_passes': 0,
                    'first_build': build_id,
                    'first_build_idx': build_idx,
                    'last_build': build_id,
                    'last_build_idx': build_idx,
                    'last_failure_build': None,
                    'last_pass_build': None,
                    'last_failure_idx': None,
                    'last_pass_idx': None,
                    'current_streak_failures': 0,
                    'current_streak_passes': 0,
                    'max_streak_failures': 0,
                    'max_streak_passes': 0,
                    'commit_count': 0,
                    'cr_count': 0,
                    'recent_results': [],
                    'builds_affected': 0,
                    'stability_score': 1.0,
                    'pass_fail_ratio': 0.0
                }

            history = self.tc_history[tc_key]
            history['total_executions'] += 1
            history['last_build'] = build_id
            history['last_build_idx'] = build_idx
            
            if pd.notna(row.get('commit')):
                commits = str(row['commit']).split('|')
                history['commit_count'] += len(commits)
                
            if pd.notna(row.get('CR')):
                crs = str(row['CR']).split('|')
                history['cr_count'] += len(crs)
                
            history['recent_results'].append(result)
            if len(history['recent_results']) > max(self.recent_window, self.medium_term_window):
                history['recent_results'] = history['recent_results'][-max(self.recent_window, self.medium_term_window):]
                
            if is_fail:
                history['total_failures'] += 1
                history['last_failure_build'] = build_id
                history['last_failure_idx'] = build_idx
                history['current_streak_failures'] += 1
                history['current_streak_passes'] = 0
                if history['current_streak_failures'] > history['max_streak_failures']:
                    history['max_streak_failures'] = history['current_streak_failures']
            elif result == 'Pass':
                history['total_passes'] += 1
                history['last_pass_build'] = build_id
                history['last_pass_idx'] = build_idx
                history['current_streak_passes'] += 1
                history['current_streak_failures'] = 0
                if history['current_streak_passes'] > history['max_streak_passes']:
                    history['max_streak_passes'] = history['current_streak_passes']
                    
            # Update rates
            executions = history['total_failures'] + history['total_passes']
            history['failure_rate'] = history['total_failures'] / executions if executions > 0 else 0.0
            
            recent = history['recent_results'][-self.recent_window:]
            if recent:
                history['recent_failure_rate'] = sum(1 for r in recent if r != 'Pass') / len(recent)
            else:
                history['recent_failure_rate'] = 0.0
                
            very_recent = history['recent_results'][-self.very_recent_window:]
            if very_recent:
                history['very_recent_failure_rate'] = sum(1 for r in very_recent if r != 'Pass') / len(very_recent)
            else:
                history['very_recent_failure_rate'] = 0.0
                
            medium_recent = history['recent_results'][-self.medium_term_window:]
            if medium_recent:
                history['medium_term_failure_rate'] = sum(1 for r in medium_recent if r != 'Pass') / len(medium_recent)
            else:
                history['medium_term_failure_rate'] = 0.0
                
            # Flakiness calculation
            flips = 0
            for i in range(1, len(history['recent_results'])):
                if history['recent_results'][i] != history['recent_results'][i-1]:
                    flips += 1
            history['flakiness_rate'] = flips / len(history['recent_results']) if len(history['recent_results']) > 1 else 0.0
            history['execution_frequency'] = executions / (build_idx - history['first_build_idx'] + 1)
            history['last_failure_age'] = build_idx - history['last_failure_idx'] if history['last_failure_build'] else -1
            history['last_pass_age'] = build_idx - history['last_pass_idx'] if history['last_pass_build'] else -1
            history['failure_trend'] = history['recent_failure_rate'] - history['medium_term_failure_rate']
            
            # Simple trend calculations
            if len(history['recent_results']) >= 2:
                history['acceleration'] = history['recent_failure_rate'] - history['very_recent_failure_rate']
            else:
                history['acceleration'] = 0.0
            
            if history['failure_rate'] > 0:
                history['deceleration_factor'] = (history['failure_rate'] - history['recent_failure_rate']) / history['failure_rate']
            else:
                history['deceleration_factor'] = 0.0
                
            history['recent_execution_count'] = len(history['recent_results'])

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

        # Build O(1) lookup cache
        self._build_id_to_idx = {bid: idx for idx, bid in enumerate(self.build_chronology)}
        logger.info(f"Chronology spans {len(self.build_chronology)} builds")

    def _compute_tc_history_v2(self, df: pd.DataFrame) -> None:
        """
        Compute EXTENSIVE historical statistics for each TC_Key.

        Computes:
        - Execution/failure/pass counts
        - Failure rates (overall, recent, very recent, medium-term)
        - Flakiness and stability
        - Streaks (consecutive failures/passes, max streaks)
        - Last failure/pass ages
        - Execution frequency
        - Build activity
        """
        logger.info("Computing EXTENSIVE per-TC_Key historical statistics (V2)...")

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

            # Recent failure rate (last N builds)
            recent_results = results[-self.recent_window:] if len(results) >= self.recent_window else results
            recent_failures = (recent_results != 'Pass').sum()
            recent_failure_rate = recent_failures / len(recent_results) if len(recent_results) > 0 else 0.0

            # Very recent failure rate (last 2 builds)
            very_recent_results = results[-self.very_recent_window:] if len(results) >= self.very_recent_window else results
            very_recent_failures = (very_recent_results != 'Pass').sum()
            very_recent_failure_rate = very_recent_failures / len(very_recent_results) if len(very_recent_results) > 0 else 0.0

            # Medium-term failure rate (last 10 builds)
            medium_results = results[-self.medium_term_window:] if len(results) >= self.medium_term_window else results
            medium_failures = (medium_results != 'Pass').sum()
            medium_term_failure_rate = medium_failures / len(medium_results) if len(medium_results) > 0 else 0.0

            # === FLAKINESS ===
            transitions = 0
            if len(results) > 1:
                for i in range(len(results) - 1):
                    if (results[i] == 'Pass') != (results[i+1] == 'Pass'):
                        transitions += 1
                flakiness_rate = transitions / (len(results) - 1)
            else:
                flakiness_rate = 0.0

            # === STREAKS ===
            consecutive_failures, consecutive_passes = self._compute_current_streaks(results)
            max_consecutive_failures, max_consecutive_passes = self._compute_max_streaks(results)

            # === LAST FAILURE/PASS AGE ===
            last_failure_idx, last_pass_idx = self._find_last_occurrence_indices(results, build_indices)
            current_build_idx = build_indices[-1] if len(build_indices) > 0 else 0

            last_failure_age = (current_build_idx - last_failure_idx) if last_failure_idx is not None else 999.0
            last_pass_age = (current_build_idx - last_pass_idx) if last_pass_idx is not None else 999.0

            # === EXECUTION FREQUENCY ===
            first_build_idx = build_indices[0] if len(build_indices) > 0 else 0
            builds_span = current_build_idx - first_build_idx + 1
            execution_frequency = total_executions / builds_span if builds_span > 0 else 0.0

            # === BUILD ACTIVITY ===
            builds_affected = len(set(tc_df['Build_ID']))

            # === TRENDS ===
            failure_trend = recent_failure_rate - failure_rate  # positive = increasing failures
            acceleration = very_recent_failure_rate - recent_failure_rate  # positive = accelerating
            deceleration_factor = recent_failure_rate / failure_rate if failure_rate > 0 else 1.0

            # === STABILITY ===
            stability_score = 1.0 - flakiness_rate
            pass_fail_ratio = total_passes / (total_failures + 1)  # +1 to avoid div by zero

            # Recent stability (flakiness in recent window)
            if len(recent_results) > 1:
                recent_transitions = 0
                for i in range(len(recent_results) - 1):
                    if (recent_results[i] == 'Pass') != (recent_results[i+1] == 'Pass'):
                        recent_transitions += 1
                recent_flakiness = recent_transitions / (len(recent_results) - 1)
                recent_stability = 1.0 - recent_flakiness
            else:
                recent_stability = 1.0

            # === RECENT EXECUTION COUNT ===
            recent_execution_count = len(recent_results)

            # Store ALL statistics
            self.tc_history[tc_key] = {
                # Basic counts
                'total_executions': total_executions,
                'total_failures': total_failures,
                'total_passes': total_passes,

                # Rates
                'failure_rate': failure_rate,
                'recent_failure_rate': recent_failure_rate,
                'very_recent_failure_rate': very_recent_failure_rate,
                'medium_term_failure_rate': medium_term_failure_rate,
                'flakiness_rate': flakiness_rate,

                # Streaks (keys match _update_history_incremental / _extract_phylogenetic_features_v2)
                'current_streak_failures': consecutive_failures,
                'current_streak_passes': consecutive_passes,
                'max_streak_failures': max_consecutive_failures,
                'max_streak_passes': max_consecutive_passes,

                # Ages
                'last_failure_age': last_failure_age,
                'last_pass_age': last_pass_age,

                # Frequency & activity
                'execution_frequency': execution_frequency,
                'recent_execution_count': recent_execution_count,
                'builds_affected': builds_affected,

                # Trends
                'failure_trend': failure_trend,
                'acceleration': acceleration,
                'deceleration_factor': deceleration_factor,

                # Stability
                'stability_score': stability_score,
                'pass_fail_ratio': pass_fail_ratio,
                'recent_stability': recent_stability,

                # Build indices (for reference)
                'first_build_idx': build_indices[0] if len(build_indices) > 0 else 0,
                'last_build_idx': build_indices[-1] if len(build_indices) > 0 else 0,
                'build_history': build_indices.tolist(),
                'result_history': results.tolist()
            }

        logger.info(f"✓ Computed EXTENSIVE history for {len(self.tc_history)} test cases")

    def _compute_current_streaks(self, results: np.ndarray) -> Tuple[int, int]:
        """
        Compute current consecutive failure/pass streaks.

        Returns:
            (consecutive_failures, consecutive_passes)
        """
        if len(results) == 0:
            return 0, 0

        current_streak_failures = 0
        current_streak_passes = 0

        # Start from end and count backward
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
        """
        Compute maximum consecutive failure/pass streaks in history.

        Returns:
            (max_consecutive_failures, max_consecutive_passes)
        """
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

        # Check final streak
        max_failures = max(max_failures, current_failures)
        max_passes = max(max_passes, current_passes)

        return max_failures, max_passes

    def _find_last_occurrence_indices(self, results: np.ndarray, build_indices: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
        """
        Find build indices of last failure and last pass.

        Returns:
            (last_failure_idx, last_pass_idx) or (None, None) if not found
        """
        last_failure_idx = None
        last_pass_idx = None

        for i in range(len(results) - 1, -1, -1):
            if results[i] != 'Pass' and last_failure_idx is None:
                last_failure_idx = build_indices[i]
            if results[i] == 'Pass' and last_pass_idx is None:
                last_pass_idx = build_indices[i]

            if last_failure_idx is not None and last_pass_idx is not None:
                break

        return last_failure_idx, last_pass_idx

    def _compute_first_appearances(self, df: pd.DataFrame) -> None:
        """Store first build index where each TC_Key appears."""
        for tc_key, history in self.tc_history.items():
            self.tc_first_appearance[tc_key] = history['first_build_idx']

        logger.info(f"✓ Stored first appearances for {len(self.tc_first_appearance)} test cases")

    def _compute_global_statistics(self, df_train: pd.DataFrame) -> None:
        """Compute global statistics for conservative defaults (29-dim)."""
        logger.info("Computing global statistics for 29 features...")

        train_features = []
        for idx, row in df_train.iterrows():
            tc_key = row['TC_Key']
            build_id = row['Build_ID']

            phylo_features = self._extract_phylogenetic_features_v2(tc_key, build_id, is_test=False)
            struct_features = self._extract_structural_features_v2(row)
            feature_vector = phylo_features + struct_features
            train_features.append(feature_vector)

        train_features = np.array(train_features, dtype=np.float32)

        self.feature_means = np.mean(train_features, axis=0)
        self.feature_medians = np.median(train_features, axis=0)
        self.feature_stds = np.std(train_features, axis=0)

        logger.info(f"  Feature means (first 10): {self.feature_means[:10]}")
        logger.info(f"  Feature medians (first 10): {self.feature_medians[:10]}")
        logger.info(f"  Feature stds (first 10): {self.feature_stds[:10]}")

    def _extract_phylogenetic_features_v2(self,
                                          tc_key: str,
                                          build_id: str,
                                          is_test: bool) -> List[float]:
        """
        Extract EXPANDED phylogenetic features (20 features).

        Returns list of 20 floats:
        [0] test_age
        [1] failure_rate
        [2] recent_failure_rate
        [3] flakiness_rate
        [4] execution_count
        [5] failure_count
        [6] pass_count
        [7] consecutive_failures
        [8] consecutive_passes
        [9] max_consecutive_failures
        [10] last_failure_age
        [11] last_pass_age
        [12] execution_frequency
        [13] failure_trend
        [14] recent_execution_count
        [15] very_recent_failure_rate
        [16] medium_term_failure_rate
        [17] acceleration
        [18] deceleration_factor
        [19] builds_since_change (placeholder, will compute from commits)
        """
        # Get current build index (O(1) dict lookup)
        current_build_idx = self._build_id_to_idx.get(build_id, len(self.build_chronology))

        if tc_key in self.tc_history:
            h = self.tc_history[tc_key]

            # Compute test age
            test_age = current_build_idx - h['first_build_idx']

            # Builds since last appearance (approximation for builds_since_change)
            builds_since_change = current_build_idx - h['last_build_idx']

            return [
                float(test_age),
                float(h['failure_rate']),
                float(h['recent_failure_rate']),
                float(h['flakiness_rate']),
                float(h['total_executions']),
                float(h['total_failures']),
                float(h['total_passes']),
                float(h['current_streak_failures']),
                float(h['current_streak_passes']),
                float(h['max_streak_failures']),
                float(h['last_failure_age']),
                float(h['last_pass_age']),
                float(h['execution_frequency']),
                float(h['failure_trend']),
                float(h['recent_execution_count']),
                float(h['very_recent_failure_rate']),
                float(h['medium_term_failure_rate']),
                float(h['acceleration']),
                float(h['deceleration_factor']),
                float(builds_since_change)
            ]
        else:
            # New test case - use conservative defaults
            if self.feature_means is not None:
                # Use population statistics for unknown tests
                defaults = [
                    0.0,  # test_age (new)
                    float(self.feature_means[1]),   # failure_rate (population avg)
                    float(self.feature_means[2]),   # recent_failure_rate
                    float(self.feature_medians[3]), # flakiness_rate (median)
                    1.0,  # execution_count (at least this one)
                    0.0,  # failure_count (unknown)
                    1.0,  # pass_count (assume passed once)
                    0.0,  # consecutive_failures
                    1.0,  # consecutive_passes
                    0.0,  # max_consecutive_failures
                    999.0,  # last_failure_age (never failed)
                    0.0,  # last_pass_age (just passed)
                    0.1,  # execution_frequency (low)
                    0.0,  # failure_trend
                    1.0,  # recent_execution_count
                    float(self.feature_means[15]) if len(self.feature_means) > 15 else 0.0,  # very_recent
                    float(self.feature_means[16]) if len(self.feature_means) > 16 else 0.0,  # medium_term
                    0.0,  # acceleration
                    1.0,  # deceleration_factor
                    0.0   # builds_since_change
                ]
                return defaults
            else:
                # Fallback zeros
                return [0.0] * 20

    def _extract_structural_features_v2(self, row: pd.Series) -> List[float]:
        """
        Extract EXPANDED structural features (9 features).

        Returns list of 9 floats:
        [0] commit_count (total commits + CRs)
        [1] test_novelty (is first appearance?)
        [2] builds_affected (from history)
        [3] cr_count (CRs only)
        [4] commit_count_actual (commits only, not CRs)
        [5] avg_commits_per_execution
        [6] recent_commit_surge (bool: more than average?)
        [7] stability_score (1 - flakiness)
        [8] pass_fail_ratio
        """
        tc_key = row['TC_Key']
        build_id = row['Build_ID']

        # 1. Commit count (total: commits + CRs)
        commit_count, cr_count_only, commit_count_only = self._count_commits_detailed(row)

        # 2. Test novelty
        if tc_key in self.tc_first_appearance:
            first_build_idx = self.tc_first_appearance[tc_key]
            current_build_idx = self._build_id_to_idx.get(build_id, len(self.build_chronology))
            test_novelty = 1.0 if current_build_idx == first_build_idx else 0.0
        else:
            test_novelty = 1.0  # New test

        # 3-8. Features from history
        if tc_key in self.tc_history:
            h = self.tc_history[tc_key]
            builds_affected = float(h['builds_affected'])
            avg_commits = commit_count / h['total_executions'] if h['total_executions'] > 0 else commit_count
            recent_commit_surge = 1.0 if commit_count > (avg_commits * 1.5) else 0.0
            stability_score = float(h['stability_score'])
            pass_fail_ratio = float(h['pass_fail_ratio'])
        else:
            builds_affected = 1.0
            avg_commits = commit_count
            recent_commit_surge = 0.0
            stability_score = 0.5  # neutral
            pass_fail_ratio = 1.0  # neutral

        return [
            float(commit_count),
            float(test_novelty),
            float(builds_affected),
            float(cr_count_only),
            float(commit_count_only),
            float(avg_commits),
            float(recent_commit_surge),
            float(stability_score),
            float(pass_fail_ratio)
        ]

    def _count_commits_detailed(self, row: pd.Series) -> Tuple[float, float, float]:
        """
        Count commits and CRs separately.

        Returns:
            (total_count, cr_count, commit_count)
        """
        commits = set()
        crs = set()

        # Count commits from 'commit' column
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

        cr_count = len(crs)
        commit_count = len(commits)
        total_count = commit_count + cr_count

        return float(max(total_count, 1)), float(cr_count), float(commit_count)

    def get_feature_names(self) -> List[str]:
        """
        Get all 29 feature names in order.

        Returns:
            List of 29 feature names
        """
        return [
            # PHYLOGENETIC (20 features)
            'test_age',
            'failure_rate',
            'recent_failure_rate',
            'flakiness_rate',
            'execution_count',
            'failure_count',
            'pass_count',
            'consecutive_failures',
            'consecutive_passes',
            'max_consecutive_failures',
            'last_failure_age',
            'last_pass_age',
            'execution_frequency',
            'failure_trend',
            'recent_execution_count',
            'very_recent_failure_rate',
            'medium_term_failure_rate',
            'acceleration',
            'deceleration_factor',
            'builds_since_change',

            # STRUCTURAL (9 features)
            'commit_count',
            'test_novelty',
            'builds_affected',
            'cr_count',
            'commit_count_actual',
            'avg_commits_per_execution',
            'recent_commit_surge',
            'stability_score',
            'pass_fail_ratio'
        ]

    def get_feature_stats(self, feature_matrix: np.ndarray) -> pd.DataFrame:
        """Get statistics for all 29 features."""
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

    def save_history(self, filepath: str) -> None:
        """Save computed statistics to disk."""
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
            'version': 'v2'
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"✓ Saved V2 historical state to {filepath}")

    def load_history(self, filepath: str) -> 'StructuralFeatureExtractorV2':
        """Load previously computed statistics."""
        import pickle

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.tc_history = state['tc_history']
        self.build_chronology = state['build_chronology']
        self._build_id_to_idx = {bid: idx for idx, bid in enumerate(self.build_chronology)}
        self.tc_first_appearance = state['tc_first_appearance']
        self.recent_window = state.get('recent_window', 5)
        self.very_recent_window = state.get('very_recent_window', 2)
        self.medium_term_window = state.get('medium_term_window', 10)
        self.min_history = state.get('min_history', 2)
        self.feature_means = state.get('feature_means', None)
        self.feature_medians = state.get('feature_medians', None)
        self.feature_stds = state.get('feature_stds', None)

        version = state.get('version', 'v1')
        logger.info(f"✓ Loaded V2 historical state from {filepath} (version: {version})")
        logger.info(f"  {len(self.tc_history)} test cases")
        logger.info(f"  {len(self.build_chronology)} builds")

        return self

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


# For easy import
__all__ = ['StructuralFeatureExtractorV2']
