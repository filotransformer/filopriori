"""
Priority Score Generator - DeepOrder-inspired Priority Labels

This module implements the priority score computation from the DeepOrder paper:
"DeepOrder: Deep Learning for Test Case Prioritization in Continuous Integration Testing"
(ICSME 2021)

The key insight is that test case prioritization should be formulated as REGRESSION,
not classification. The priority score p(ti) directly represents how important
it is to run a test early.

Formula:
    p(ti) = Σ(j=1..m) wj × max(ES(i,j), 0)

Where:
    - ES(i,j) ∈ {1 = failed, 0 = passed, -1 = not executed}
    - wj = weight for cycle j (recent cycles have higher weight)
    - Σwj = 1
    - p(ti) ∈ (0, 1)

Author: Filo-Priori V9 Team
Date: December 2024
Reference: https://github.com/AizazSharif/DeepOrder-ICSME21
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class PriorityScoreGenerator:
    """
    Generates DeepOrder-style priority scores from test execution history.

    Priority scores are continuous values in (0, 1) that represent how likely
    a test is to fail and how important it is to run early. Tests with recent
    failures have higher priority scores.

    This creates labels that directly optimize for APFD when used with MSE loss.
    """

    def __init__(
        self,
        num_cycles: int = 10,
        decay_type: str = 'exponential',
        decay_factor: float = 0.8,
        normalize: bool = True
    ):
        """
        Initialize the priority score generator.

        Args:
            num_cycles: Number of historical cycles to consider (default: 10)
            decay_type: Type of weight decay for older cycles
                       'exponential': w_j = decay^(m-j)
                       'linear': w_j linearly increases to recent
                       'inverse': w_j = 1/position
                       'uniform': all cycles equal weight
            decay_factor: Decay factor for exponential decay (default: 0.8)
            normalize: Whether to normalize weights to sum to 1
        """
        self.num_cycles = num_cycles
        self.decay_type = decay_type
        self.decay_factor = decay_factor
        self.normalize = normalize
        self.weights = self._compute_weights()

        logger.info("=" * 70)
        logger.info("PRIORITY SCORE GENERATOR INITIALIZED (DeepOrder-style)")
        logger.info("=" * 70)
        logger.info(f"  Number of cycles: {num_cycles}")
        logger.info(f"  Decay type: {decay_type}")
        logger.info(f"  Decay factor: {decay_factor}")
        logger.info(f"  Weights: {self.weights}")
        logger.info("=" * 70)

    def _compute_weights(self) -> np.ndarray:
        """
        Compute cycle weights where more recent cycles have higher weight.

        Returns:
            Array of weights, one per cycle, ordered oldest to newest
        """
        if self.decay_type == 'exponential':
            # Exponential decay: recent cycles much more important
            # w_j = decay^(num_cycles - 1 - j) for j=0..num_cycles-1
            raw_weights = np.array([
                self.decay_factor ** (self.num_cycles - 1 - j)
                for j in range(self.num_cycles)
            ])
        elif self.decay_type == 'linear':
            # Linear increase: w_j = (j+1) / sum
            raw_weights = np.arange(1, self.num_cycles + 1, dtype=float)
        elif self.decay_type == 'inverse':
            # Inverse: w_j = 1 / (num_cycles - j)
            raw_weights = 1.0 / np.arange(self.num_cycles, 0, -1)
        elif self.decay_type == 'uniform':
            # All cycles equal weight
            raw_weights = np.ones(self.num_cycles)
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")

        if self.normalize:
            return raw_weights / raw_weights.sum()
        return raw_weights

    def compute_priority(
        self,
        execution_history: List[int],
        return_components: bool = False
    ) -> float:
        """
        Compute priority score for a test case based on its execution history.

        Args:
            execution_history: List of execution statuses ordered from oldest to newest
                              [1 = failed, 0 = passed, -1 = not executed]
            return_components: If True, also return the contribution per cycle

        Returns:
            Priority score in (0, 1), or tuple (score, components) if return_components
        """
        # Handle empty history
        if not execution_history:
            if return_components:
                return 0.0, np.zeros(self.num_cycles)
            return 0.0

        # Take the last num_cycles entries
        history = list(execution_history[-self.num_cycles:])

        # Pad with -1 (not executed) for missing old cycles
        if len(history) < self.num_cycles:
            padding = [-1] * (self.num_cycles - len(history))
            history = padding + history

        history = np.array(history)

        # Apply DeepOrder formula: p = Σ wj × max(ES(i,j), 0)
        # max(ES, 0) converts -1 to 0 (ignore not-executed tests)
        contribution = np.maximum(history, 0)
        weighted_contribution = self.weights * contribution
        priority = np.sum(weighted_contribution)

        if return_components:
            return float(priority), weighted_contribution
        return float(priority)

    def compute_priorities_for_dataframe(
        self,
        df: pd.DataFrame,
        build_col: str = 'Build_ID',
        tc_col: str = 'TC_Name',
        result_col: str = 'TE_Test_Result',
        fail_value: str = 'Fail',
        pass_value: str = 'Pass',
        initial_history: Optional[Dict[str, List[int]]] = None,
        extract_features: bool = False
    ) -> Union[Tuple[pd.DataFrame, Dict[str, List[int]]], Tuple[pd.DataFrame, Dict[str, List[int]], np.ndarray]]:
        """
        Compute priority scores for all test cases in a DataFrame.

        IMPORTANT: This processes data chronologically and computes priority
        for each test case BEFORE seeing its result in the current build.
        This prevents data leakage.

        Args:
            df: Input DataFrame with test execution data
            build_col: Column name for build IDs
            tc_col: Column name for test case names
            result_col: Column name for test results
            fail_value: Value indicating failure
            pass_value: Value indicating pass
            initial_history: Optional pre-existing execution history to start from.
                           Use this to carry over history from training to val/test splits.
            extract_features: If True, also extract DeepOrder features chronologically.

        Returns:
            Tuple of (DataFrame with priority_score column, execution history dict)
            If extract_features is True, returns a third element: numpy array of features.
        """
        df = df.copy()
        df['priority_score'] = 0.0

        # Track execution history per test case (start from initial if provided)
        tc_history: Dict[str, List[int]] = {}
        if initial_history is not None:
            # Deep copy to avoid modifying the original
            for tc, hist in initial_history.items():
                tc_history[tc] = list(hist)

        # Get unique builds in order (assuming they're already sorted chronologically)
        builds = df[build_col].unique()

        logger.info(f"Computing priority scores for {len(df)} samples across {len(builds)} builds...")

        priority_stats = {
            'total_samples': len(df),
            'samples_with_history': 0,
            'samples_with_failures': 0,
            'max_priority': 0.0,
            'min_priority': 1.0,
            'mean_priority': 0.0
        }

        priorities = []

        # OPTIMIZED: Vectorized iteration instead of slow df.loc
        import numpy as np
        
        # Pre-extract columns to numpy/lists for extreme speed
        build_ids = df[build_col].values
        tc_names = df[tc_col].values
        results = df[result_col].astype(str).str.strip().values
        indices = df.index.values
        
        # Pre-allocate array for priorities
        priorities_arr = np.zeros(len(df), dtype=np.float32)
        
        if extract_features:
            features_arr = np.zeros((len(df), 9), dtype=np.float32)
        
        for i in range(len(df)):
            if i % 1000000 == 0 and i > 0:
                logger.info(f"    Processed {i}/{len(df)} samples...")
                
            tc_name = tc_names[i]
            history = tc_history.get(tc_name, [])
            
            # Compute priority
            if len(history) > 0:
                priorities_arr[i] = self.compute_priority(history)
                priority_stats['samples_with_history'] += 1
                if any(h == 1 for h in history):
                    priority_stats['samples_with_failures'] += 1
            
            if extract_features:
                features_arr[i] = self.extract_deeporder_features(tc_history, tc_name)
            
            # Update history
            res = results[i]
            if res == fail_value:
                status = 1
            elif res == pass_value:
                status = 0
            else:
                status = -1
                
            if tc_name not in tc_history:
                tc_history[tc_name] = []
            tc_history[tc_name].append(status)
            
        # Assign priorities vectorized
        df['priority_score'] = priorities_arr

        # Compute statistics
        all_priorities = df['priority_score'].values
        priority_stats['max_priority'] = float(np.max(all_priorities))
        priority_stats['min_priority'] = float(np.min(all_priorities[all_priorities > 0])) if np.any(all_priorities > 0) else 0.0
        priority_stats['mean_priority'] = float(np.mean(all_priorities))

        logger.info("Priority score computation complete:")
        logger.info(f"  Samples with history: {priority_stats['samples_with_history']}/{priority_stats['total_samples']}")
        logger.info(f"  Samples with past failures: {priority_stats['samples_with_failures']}")
        logger.info(f"  Priority range: [{priority_stats['min_priority']:.4f}, {priority_stats['max_priority']:.4f}]")
        logger.info(f"  Mean priority: {priority_stats['mean_priority']:.4f}")

        if extract_features:
            return df, tc_history, features_arr
        return df, tc_history

    def extract_deeporder_features(
        self,
        tc_history: Dict[str, List[int]],
        tc_name: str
    ) -> np.ndarray:
        """
        Extract DeepOrder-inspired features for a test case.

        These features capture temporal patterns that the DeepOrder paper
        found important for prioritization.

        Features (9 total):
            - execution_status_last_[1,2,3,5,10]: Status at specific points
            - distance: |oldest_status - newest_status|
            - status_changes: Number of pass↔fail transitions
            - cycles_since_last_fail: How long since last failure
            - fail_rate_last_10: Failure rate in last 10 cycles

        Args:
            tc_history: Dictionary mapping test case names to execution histories
            tc_name: Name of the test case

        Returns:
            Array of 9 DeepOrder features
        """
        history = tc_history.get(tc_name, [])
        features = []

        # Execution statuses at specific points (5 features)
        for offset in [1, 2, 3, 5, 10]:
            if len(history) >= offset:
                # Convert: 1=fail→1, 0=pass→0, -1=not_exec→0.5 (neutral)
                status = history[-offset]
                if status == -1:
                    features.append(0.5)
                else:
                    features.append(float(status))
            else:
                features.append(0.5)  # No data = neutral

        # Distance: |oldest_status - newest_status| (1 feature)
        if len(history) >= 2:
            oldest = history[0] if history[0] != -1 else 0
            newest = history[-1] if history[-1] != -1 else 0
            distance = abs(newest - oldest)
        else:
            distance = 0
        features.append(float(distance))

        # Status changes (pass↔fail transitions) (1 feature)
        changes = 0
        for i in range(1, len(history)):
            prev = history[i - 1]
            curr = history[i]
            # Only count transitions between pass (0) and fail (1)
            if prev != -1 and curr != -1 and prev != curr:
                changes += 1
        features.append(float(changes) / max(len(history), 1))  # Normalize

        # Cycles since last fail (1 feature)
        cycles_since_fail = len(history)  # Default: never failed
        for i, status in enumerate(reversed(history)):
            if status == 1:  # Fail
                cycles_since_fail = i
                break
        # Normalize: 0 = just failed, 1 = never failed or long ago
        features.append(min(cycles_since_fail, 20) / 20.0)

        # Fail rate in last 10 cycles (1 feature)
        recent = [s for s in history[-10:] if s != -1]
        if len(recent) > 0:
            fail_rate = sum(1 for s in recent if s == 1) / len(recent)
        else:
            fail_rate = 0.0
        features.append(fail_rate)

        return np.array(features, dtype=np.float32)

    def extract_deeporder_features_batch(
        self,
        df: pd.DataFrame,
        tc_history: Dict[str, List[int]],
        tc_col: str = 'TC_Name'
    ) -> np.ndarray:
        # OPTIMIZED: Remove slow iterrows loop
        tc_names = df[tc_col].values
        n_samples = len(tc_names)
        
        # Pre-allocate output array [N, 9]
        features_arr = np.zeros((n_samples, 9), dtype=np.float32)
        
        for i in range(n_samples):
            features_arr[i] = self.extract_deeporder_features(tc_history, tc_names[i])
            
        return features_arr


def create_priority_score_generator(config: Dict) -> PriorityScoreGenerator:
    """
    Factory function to create PriorityScoreGenerator from config.

    Args:
        config: Configuration dictionary with optional keys:
                - num_cycles: int (default: 10)
                - decay_type: str (default: 'exponential')
                - decay_factor: float (default: 0.8)

    Returns:
        Configured PriorityScoreGenerator instance
    """
    priority_config = config.get('priority_score', {})

    return PriorityScoreGenerator(
        num_cycles=priority_config.get('num_cycles', 10),
        decay_type=priority_config.get('decay_type', 'exponential'),
        decay_factor=priority_config.get('decay_factor', 0.8),
        normalize=priority_config.get('normalize', True)
    )
