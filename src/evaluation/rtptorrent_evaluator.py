"""
RTPTorrent Evaluation Module.

Provides comprehensive evaluation against the 7 baseline strategies
included in the RTPTorrent dataset:

1. untreated - Original order from build log
2. random - Shuffled schedule
3. recently-failed - Ranked by recent failure history (alpha=0.8)
4. optimal-failure - Optimal ordering (failures first) - upper bound
5. optimal-failure-duration - Optimal (shortest failures first)
6. matrix-naive - File-test-failures matrix
7. matrix-conditional-prob - Conditional probability P(test | changed_files)

Metrics computed:
- APFD (Average Percentage of Faults Detected)
- APFD@k (APFD for first k tests)
- First Failure Position
- Recall@k (percentage of failures found in first k tests)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class RTPTorrentEvaluator:
    """Evaluator for RTPTorrent dataset with baseline comparison."""

    # Baseline strategies provided by RTPTorrent
    BASELINE_STRATEGIES = [
        'untreated',
        'random',
        'recently-failed',
        'optimal-failure',
        'optimal-failure-duration',
        'matrix-naive',
        'matrix-conditional-prob'
    ]

    def __init__(
        self,
        raw_dir: Path,
        project_name: str,
        k_values: Optional[List[int]] = None
    ):
        """
        Initialize evaluator.

        Args:
            raw_dir: Path to RTPTorrent raw data directory
            project_name: Project name (e.g., "square@okhttp")
            k_values: List of k values for APFD@k (default: [5, 10, 20])
        """
        self.raw_dir = Path(raw_dir)
        self.project_name = project_name
        self.k_values = k_values or [5, 10, 20]

        # Load baseline data
        self.baselines = self._load_baselines()

        logger.info(f"Initialized RTPTorrentEvaluator for {project_name}")
        logger.info(f"  Loaded {len(self.baselines)} baselines")

    def _load_baselines(self) -> Dict[str, pd.DataFrame]:
        """Load baseline ranking files for the project."""
        baselines = {}
        project_dir = self.raw_dir / self.project_name
        baseline_dir = project_dir / "baseline"

        if not baseline_dir.exists():
            logger.warning(f"No baseline directory found: {baseline_dir}")
            return baselines

        # Extract short project name (after @)
        short_name = self.project_name.split('@')[1] if '@' in self.project_name else self.project_name

        for strategy in self.BASELINE_STRATEGIES:
            baseline_file = baseline_dir / f"{short_name}@{strategy}.csv"
            if baseline_file.exists():
                baselines[strategy] = pd.read_csv(baseline_file)
                logger.debug(f"  Loaded {strategy}: {len(baselines[strategy])} rows")
            else:
                logger.warning(f"  Baseline not found: {baseline_file}")

        return baselines

    @staticmethod
    def compute_apfd(ranks: np.ndarray, n_tests: int) -> float:
        """
        Compute APFD for a single build.

        APFD = 1 - (sum(TFi) / (n * m)) + 1/(2n)

        Args:
            ranks: Positions of failing tests (1-indexed)
            n_tests: Total number of tests in the build

        Returns:
            APFD value in [0, 1], or None if no failures
        """
        m = len(ranks)
        if m == 0:
            return None

        apfd = 1.0 - (ranks.sum() / (n_tests * m)) + 1.0 / (2 * n_tests)
        return apfd

    @staticmethod
    def compute_apfd_at_k(ranks: np.ndarray, n_tests: int, k: int) -> float:
        """
        Compute APFD considering only first k tests.

        Args:
            ranks: Positions of failing tests (1-indexed)
            n_tests: Total number of tests
            k: Cutoff position

        Returns:
            APFD@k value, or None if no failures in first k
        """
        # Only consider failures in first k positions
        ranks_at_k = ranks[ranks <= k]
        m = len(ranks_at_k)

        if m == 0:
            return 0.0  # No failures found in first k tests

        # Adjusted APFD for truncated list
        apfd = 1.0 - (ranks_at_k.sum() / (k * m)) + 1.0 / (2 * k)
        return apfd

    @staticmethod
    def compute_first_failure_position(ranks: np.ndarray) -> Optional[int]:
        """
        Get position of first failing test.

        Args:
            ranks: Positions of failing tests (1-indexed)

        Returns:
            Position of first failure, or None if no failures
        """
        if len(ranks) == 0:
            return None
        return int(ranks.min())

    @staticmethod
    def compute_recall_at_k(ranks: np.ndarray, total_failures: int, k: int) -> float:
        """
        Compute recall at position k.

        Args:
            ranks: Positions of failing tests (1-indexed)
            total_failures: Total number of failures in build
            k: Cutoff position

        Returns:
            Fraction of failures found in first k tests
        """
        if total_failures == 0:
            return 1.0  # No failures to find

        found_at_k = np.sum(ranks <= k)
        return found_at_k / total_failures

    def evaluate_ranking(
        self,
        build_df: pd.DataFrame,
        score_column: str = 'score'
    ) -> Dict:
        """
        Evaluate a ranking for a single build.

        Args:
            build_df: DataFrame with columns: testName, is_failure, and score_column
            score_column: Name of column containing ranking scores (higher = higher priority)

        Returns:
            Dictionary with evaluation metrics
        """
        # Sort by score (descending) to get ranking
        ranked_df = build_df.sort_values(score_column, ascending=False).copy()
        ranked_df['rank'] = range(1, len(ranked_df) + 1)

        # Get failure positions
        failures = ranked_df[ranked_df['is_failure'] == 1]
        failure_ranks = failures['rank'].values

        n_tests = len(ranked_df)
        n_failures = len(failure_ranks)

        results = {
            'n_tests': n_tests,
            'n_failures': n_failures,
            'apfd': self.compute_apfd(failure_ranks, n_tests),
            'first_failure_position': self.compute_first_failure_position(failure_ranks)
        }

        # APFD@k and Recall@k for various k
        for k in self.k_values:
            if k <= n_tests:
                results[f'apfd_at_{k}'] = self.compute_apfd_at_k(failure_ranks, n_tests, k)
                results[f'recall_at_{k}'] = self.compute_recall_at_k(failure_ranks, n_failures, k)

        return results

    def evaluate_baseline(
        self,
        strategy: str,
        test_build_ids: List
    ) -> Dict:
        """
        Evaluate a baseline strategy on specified test builds.

        Args:
            strategy: Baseline strategy name
            test_build_ids: List of travisJobId to evaluate

        Returns:
            Dictionary with aggregated metrics
        """
        if strategy not in self.baselines:
            logger.warning(f"Baseline {strategy} not loaded")
            return {}

        baseline_df = self.baselines[strategy]
        baseline_df['is_failure'] = ((baseline_df['failures'] > 0) |
                                      (baseline_df['errors'] > 0)).astype(int)

        # Use index as score (lower index = higher priority in baseline)
        baseline_df['score'] = -baseline_df['index']  # Negate to make higher = better

        results = []
        for build_id in test_build_ids:
            build_df = baseline_df[baseline_df['travisJobId'] == build_id]
            if len(build_df) == 0:
                continue

            # Skip builds without failures
            if build_df['is_failure'].sum() == 0:
                continue

            build_results = self.evaluate_ranking(build_df, 'score')
            build_results['build_id'] = build_id
            results.append(build_results)

        if not results:
            return {'mean_apfd': None, 'n_builds': 0}

        # Aggregate results
        results_df = pd.DataFrame(results)

        aggregated = {
            'n_builds': len(results_df),
            'mean_apfd': results_df['apfd'].mean(),
            'median_apfd': results_df['apfd'].median(),
            'std_apfd': results_df['apfd'].std(),
            'mean_first_failure': results_df['first_failure_position'].mean()
        }

        # Add APFD@k and Recall@k
        for k in self.k_values:
            col = f'apfd_at_{k}'
            if col in results_df.columns:
                aggregated[f'mean_{col}'] = results_df[col].mean()

            col = f'recall_at_{k}'
            if col in results_df.columns:
                aggregated[f'mean_{col}'] = results_df[col].mean()

        return aggregated

    def evaluate_model_vs_baselines(
        self,
        model_predictions: pd.DataFrame,
        test_build_ids: List
    ) -> Dict:
        """
        Compare model predictions against all baselines.

        Args:
            model_predictions: DataFrame with columns: travisJobId, testName, score, is_failure
            test_build_ids: List of build IDs to evaluate

        Returns:
            Dictionary with model results and comparison to all baselines
        """
        results = {'model': {}, 'baselines': {}, 'comparison': {}}

        # Evaluate model
        model_results = []
        for build_id in test_build_ids:
            build_df = model_predictions[model_predictions['travisJobId'] == build_id]
            if len(build_df) == 0:
                continue

            if build_df['is_failure'].sum() == 0:
                continue

            build_results = self.evaluate_ranking(build_df, 'score')
            build_results['build_id'] = build_id
            model_results.append(build_results)

        if model_results:
            model_df = pd.DataFrame(model_results)
            results['model'] = {
                'n_builds': len(model_df),
                'mean_apfd': model_df['apfd'].mean(),
                'median_apfd': model_df['apfd'].median(),
                'std_apfd': model_df['apfd'].std(),
                'mean_first_failure': model_df['first_failure_position'].mean()
            }

            for k in self.k_values:
                col = f'apfd_at_{k}'
                if col in model_df.columns:
                    results['model'][f'mean_{col}'] = model_df[col].mean()

        # Evaluate all baselines
        for strategy in self.BASELINE_STRATEGIES:
            if strategy in self.baselines:
                results['baselines'][strategy] = self.evaluate_baseline(strategy, test_build_ids)

        # Compute comparison (improvement over each baseline)
        if results['model'].get('mean_apfd') is not None:
            model_apfd = results['model']['mean_apfd']

            for strategy, baseline_results in results['baselines'].items():
                if baseline_results.get('mean_apfd') is not None:
                    baseline_apfd = baseline_results['mean_apfd']
                    improvement = model_apfd - baseline_apfd
                    improvement_pct = (improvement / baseline_apfd) * 100 if baseline_apfd > 0 else 0

                    results['comparison'][strategy] = {
                        'improvement': improvement,
                        'improvement_pct': improvement_pct,
                        'model_better': model_apfd > baseline_apfd
                    }

        return results

    def statistical_comparison(
        self,
        model_apfds: List[float],
        baseline_apfds: List[float],
        alpha: float = 0.05
    ) -> Dict:
        """
        Perform statistical comparison between model and baseline.

        Uses Wilcoxon signed-rank test (paired, non-parametric).

        Args:
            model_apfds: APFD values from model for each build
            baseline_apfds: APFD values from baseline for each build
            alpha: Significance level

        Returns:
            Dictionary with statistical test results
        """
        if len(model_apfds) != len(baseline_apfds):
            raise ValueError("Lists must have same length")

        if len(model_apfds) < 10:
            logger.warning("Small sample size (<10), statistical tests may be unreliable")

        # Wilcoxon signed-rank test
        try:
            stat, p_value = stats.wilcoxon(model_apfds, baseline_apfds, alternative='greater')
        except ValueError as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            stat, p_value = None, None

        # Effect size (Cliff's Delta)
        n = len(model_apfds)
        greater = sum(1 for m, b in zip(model_apfds, baseline_apfds) if m > b)
        less = sum(1 for m, b in zip(model_apfds, baseline_apfds) if m < b)
        cliffs_delta = (greater - less) / n

        # Interpret effect size
        abs_delta = abs(cliffs_delta)
        if abs_delta < 0.147:
            effect_interpretation = "negligible"
        elif abs_delta < 0.33:
            effect_interpretation = "small"
        elif abs_delta < 0.474:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        return {
            'wilcoxon_statistic': stat,
            'p_value': p_value,
            'significant': p_value < alpha if p_value is not None else None,
            'cliffs_delta': cliffs_delta,
            'effect_size': effect_interpretation,
            'n_samples': n
        }

    def generate_report(
        self,
        evaluation_results: Dict,
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate a formatted evaluation report.

        Args:
            evaluation_results: Results from evaluate_model_vs_baselines()
            output_path: Optional path to save report

        Returns:
            Report as formatted string
        """
        lines = []
        lines.append("=" * 70)
        lines.append("RTPTORRENT EVALUATION REPORT")
        lines.append(f"Project: {self.project_name}")
        lines.append("=" * 70)

        # Model results
        lines.append("\n📊 MODEL RESULTS:")
        model = evaluation_results.get('model', {})
        if model:
            lines.append(f"  Builds evaluated: {model.get('n_builds', 'N/A')}")
            lines.append(f"  Mean APFD: {model.get('mean_apfd', 0):.4f}")
            lines.append(f"  Median APFD: {model.get('median_apfd', 0):.4f}")
            lines.append(f"  Std APFD: {model.get('std_apfd', 0):.4f}")
            lines.append(f"  Mean First Failure Position: {model.get('mean_first_failure', 0):.1f}")

        # Baseline comparison
        lines.append("\n📈 BASELINE COMPARISON:")
        lines.append("-" * 70)
        lines.append(f"{'Strategy':<25} {'Mean APFD':>10} {'Improvement':>12} {'Better?':>8}")
        lines.append("-" * 70)

        model_apfd = model.get('mean_apfd', 0)
        for strategy in self.BASELINE_STRATEGIES:
            baseline = evaluation_results.get('baselines', {}).get(strategy, {})
            comparison = evaluation_results.get('comparison', {}).get(strategy, {})

            baseline_apfd = baseline.get('mean_apfd', 'N/A')
            if isinstance(baseline_apfd, float):
                improvement = comparison.get('improvement_pct', 0)
                better = "✓" if comparison.get('model_better', False) else "✗"
                lines.append(f"{strategy:<25} {baseline_apfd:>10.4f} {improvement:>+11.2f}% {better:>8}")
            else:
                lines.append(f"{strategy:<25} {'N/A':>10} {'N/A':>12} {'N/A':>8}")

        lines.append("-" * 70)
        lines.append(f"{'Filo-Priori (Model)':<25} {model_apfd:>10.4f} {'(baseline)':>12}")

        # APFD@k results
        lines.append("\n📉 APFD@k RESULTS:")
        for k in self.k_values:
            model_k = model.get(f'mean_apfd_at_{k}', 'N/A')
            if isinstance(model_k, float):
                lines.append(f"  APFD@{k}: {model_k:.4f}")

        lines.append("\n" + "=" * 70)

        report = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")

        return report


def evaluate_rtptorrent_experiment(
    predictions_df: pd.DataFrame,
    raw_dir: Path,
    project_name: str,
    output_dir: Path
) -> Dict:
    """
    Convenience function to run full RTPTorrent evaluation.

    Args:
        predictions_df: Model predictions with columns:
            - travisJobId: Build identifier
            - testName: Test case name
            - score: Model's ranking score (higher = higher priority)
            - is_failure: Binary indicator (1 = failed)
        raw_dir: Path to RTPTorrent raw data
        project_name: Name of project being evaluated
        output_dir: Directory for output files

    Returns:
        Dictionary with all evaluation results
    """
    evaluator = RTPTorrentEvaluator(raw_dir, project_name)

    # Get test build IDs
    test_build_ids = predictions_df['travisJobId'].unique().tolist()

    # Run evaluation
    results = evaluator.evaluate_model_vs_baselines(predictions_df, test_build_ids)

    # Generate report
    report = evaluator.generate_report(results, output_dir / "evaluation_report.txt")
    print(report)

    # Save detailed results
    with open(output_dir / "evaluation_results.json", 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(results, f, indent=2, default=convert)

    return results
