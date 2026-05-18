"""
APFD (Average Percentage of Faults Detected) Calculation Module.

This module implements APFD calculation following business rules:
1. APFD is calculated PER BUILD, not globally
2. Only builds with at least 1 failure are considered
3. Builds with only 1 TC have APFD = 1.0 (business rule)
4. Generates report in format: method_name, build_id, test_scenario, count_tc, count_commits, apfd, time

Based on: Filo-Priori V5 implementation
Adapted for: Filo-Priori V7 (Dual-Stream Graph-Semantic Model)
Enhanced with: count_total_commits from master_vini
Date: 2025-11-06

CONSOLIDATION NOTE:
This module consolidates all APFD calculation functionality.
Previous apfd_calculator.py has been archived to eliminate code duplication.
"""

import pandas as pd
import numpy as np
import ast
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def count_total_commits(df_build: pd.DataFrame) -> int:
    """
    Count total commits for a build (including CRs).
    Works with both test.csv and test_filtered.csv structures.

    Migrated from apfd_calculator.py for consolidation.

    Args:
        df_build: DataFrame for a single build

    Returns:
        Total number of unique commits (including CRs)
    """
    total_commits = set()

    # Count commits from 'commit' column
    if 'commit' in df_build.columns:
        for commit_str in df_build['commit'].dropna():
            try:
                commits = ast.literal_eval(commit_str)
                if isinstance(commits, list):
                    total_commits.update(commits)
                else:
                    total_commits.add(str(commit_str))
            except:
                total_commits.add(str(commit_str))

    # Count CRs (works with both CR and CR_y columns)
    cr_column = 'CR_y' if 'CR_y' in df_build.columns else 'CR' if 'CR' in df_build.columns else None
    if cr_column:
        for cr_str in df_build[cr_column].dropna():
            try:
                crs = ast.literal_eval(cr_str)
                if isinstance(crs, list):
                    for cr in crs:
                        total_commits.add(f"CR_{cr}")
            except:
                total_commits.add(f"CR_{cr_str}")

    return max(len(total_commits), 1)


# ============================================================================
# APFD CALCULATION FUNCTIONS
# ============================================================================

def calculate_apfd_single_build(ranks: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """
    Calculate APFD for a single build.

    Args:
        ranks: Array of ranks (1-indexed, lower rank = higher priority)
        labels: Binary array where 1 indicates failure and 0 indicates pass

    Returns:
        APFD score (0 to 1, higher is better), or None if no failures

    Formula:
        APFD = 1 - (sum of failure ranks) / (n_failures * n_tests) + 1 / (2 * n_tests)

    Example:
        If we have 10 tests and 2 failures at ranks [2, 5]:
        APFD = 1 - (2 + 5) / (2 * 10) + 1 / (2 * 10)
             = 1 - 7/20 + 1/20
             = 1 - 0.35 + 0.05
             = 0.70
    """
    labels_arr = np.array(labels)
    ranks_arr = np.array(ranks)

    n_tests = int(len(labels_arr))
    # Treat any non-zero value as failure for robustness (supports {0,1} and {False,True})
    fail_indices = np.where(labels_arr.astype(int) != 0)[0]
    n_failures = len(fail_indices)

    # Business rule: if no failures, APFD is undefined (skip this build)
    if n_failures == 0:
        return None

    # Business rule: if only 1 test case, APFD = 1.0
    if n_tests == 1:
        return 1.0

    # Get ranks of failures
    failure_ranks = ranks_arr[fail_indices]

    # Calculate APFD
    apfd = 1.0 - float(failure_ranks.sum()) / float(n_failures * n_tests) + 1.0 / float(2.0 * n_tests)

    return float(np.clip(apfd, 0.0, 1.0))


def calculate_apfd_per_build(
    df: pd.DataFrame,
    method_name: str = "dual_stream_gnn",
    test_scenario: str = "full_test",
    build_col: str = "Build_ID",
    label_col: str = "label_binary",
    rank_col: str = "rank",
    result_col: str = "TE_Test_Result"
) -> pd.DataFrame:
    """
    Calculate APFD per build for the entire test set.

    BUSINESS RULE: Only builds with at least one test with result "Fail" are included.
    This should result in exactly 277 builds (as per project requirements).

    BUSINESS RULE (count_tc=1): Builds with only 1 unique test case MUST have APFD=1.0
    (Reference: master_vini/src/evaluation/apfd_calculator.py)

    Args:
        df: DataFrame with columns:
            - Build_ID (or build_col): Build identifier
            - label_binary (or label_col): True label (1=failure, 0=pass)
            - rank (or rank_col): Priority rank (1-indexed, lower is better)
            - TE_Test_Result (or result_col): Original test result ("Fail", "Pass", etc.)
        method_name: Name of the prioritization method
        test_scenario: Type of test scenario
        build_col: Name of build ID column
        label_col: Name of label column
        rank_col: Name of rank column
        result_col: Name of test result column (default: "TE_Test_Result")

    Returns:
        DataFrame with columns:
            - method_name: Prioritization method name
            - build_id: Build identifier
            - test_scenario: Test scenario type
            - count_tc: Number of test cases in build
            - count_commits: Number of unique commits (placeholder: 0)
            - apfd: APFD score for the build
            - time: Processing time (placeholder: 0)
    """
    results = []

    # Group by Build_ID
    grouped = df.groupby(build_col)

    logger.info(f"Calculating APFD for {len(grouped)} total builds...")

    builds_with_failures = 0
    builds_skipped = 0

    for build_id, build_df in grouped:
        # Count UNIQUE test cases (drop duplicates by TC_Key if available)
        # This matches master_vini implementation
        if 'TC_Key' in build_df.columns:
            count_tc = build_df['TC_Key'].nunique()
        else:
            count_tc = len(build_df)

        # CRITICAL BUSINESS RULE: count_tc=1 → APFD=1.0
        # When there's only 1 test case, there's no ordering to optimize.
        # The test will be executed anyway, so APFD should be 1.0 (perfect).
        # This check MUST happen early, before the standard calculation.
        if count_tc == 1:
            # Still need to verify this build has at least one failure
            if result_col in build_df.columns:
                fail_mask = (build_df[result_col].astype(str).str.strip() == "Fail")
                if not fail_mask.any():
                    builds_skipped += 1
                    continue
            else:
                # Fallback: use label column
                if label_col in build_df.columns:
                    fail_mask = (build_df[label_col].astype(int) != 0)
                    if not fail_mask.any():
                        builds_skipped += 1
                        continue
                else:
                    builds_skipped += 1
                    continue

            # Build has 1 TC and at least 1 failure: APFD = 1.0
            builds_with_failures += 1

            # Count unique commits
            try:
                count_commits = count_total_commits(build_df)
            except Exception as e:
                logger.debug(f"Could not count commits for build {build_id}: {e}")
                count_commits = 0

            # Add to results with APFD=1.0
            results.append({
                'method_name': method_name,
                'build_id': build_id,
                'test_scenario': test_scenario,
                'count_tc': count_tc,
                'count_commits': count_commits,
                'apfd': 1.0,  # Business rule: count_tc=1 → APFD=1.0
                'time': 0.0
            })
            continue  # Skip standard APFD calculation

        # CRITICAL BUSINESS RULE: Only include builds with at least one "Fail" result
        # Determine failure mask robustly
        if result_col in build_df.columns:
            fail_mask = (build_df[result_col].astype(str).str.strip() == "Fail")
            has_fail = bool(fail_mask.any())
            if not has_fail:
                builds_skipped += 1
                continue
        else:
            # Fallback: use label column. Assume 1 indicates Fail, but be defensive.
            logger.warning(f"Column '{result_col}' not found. Using '{label_col}' to infer failures.")
            if label_col in build_df.columns:
                # If labels are {0,1}, consider non-zero as Fail
                fail_mask = (build_df[label_col].astype(int) != 0)
                if not bool(fail_mask.any()):
                    builds_skipped += 1
                    continue
            else:
                logger.warning("No label column available to infer failures; skipping build.")
                builds_skipped += 1
                continue

        builds_with_failures += 1

        # Count unique commits (including CRs)
        try:
            count_commits = count_total_commits(build_df)
        except Exception as e:
            logger.debug(f"Could not count commits for build {build_id}: {e}")
            count_commits = 0

        # Get ranks and labels for this build
        # Safety: if ranks missing, compute per-build now
        if rank_col in build_df.columns:
            ranks = build_df[rank_col].values
        else:
            prob_col = 'probability' if 'probability' in build_df.columns else None
            if prob_col is None:
                logger.warning("Rank column not found and no 'probability' column to compute ranks; skipping build.")
                builds_skipped += 1
                continue
            ranks = build_df[prob_col].rank(method='first', ascending=False).astype(int).values

        # Build binary labels with Fail=1 using fail_mask derived above
        labels = fail_mask.astype(int).values

        # Calculate APFD for this build
        apfd = calculate_apfd_single_build(ranks, labels)

        # Skip if APFD is None (shouldn't happen due to earlier check, but safe)
        if apfd is None:
            continue

        # Add to results
        results.append({
            'method_name': method_name,
            'build_id': build_id,
            'test_scenario': test_scenario,
            'count_tc': count_tc,
            'count_commits': count_commits,
            'apfd': apfd,
            'time': 0.0  # Placeholder - could be filled with actual time if tracked
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by build_id for consistency
    if len(results_df) > 0:
        results_df = results_df.sort_values('build_id').reset_index(drop=True)

    logger.info(f"APFD calculated for {len(results_df)} builds with 'Fail' results")
    logger.info(f"   Builds included: {builds_with_failures}")
    logger.info(f"   Builds skipped (no failures): {builds_skipped}")
    logger.info(f"   Expected: 277 builds (as per project requirements)")

    if len(results_df) != 277:
        logger.warning(f"⚠️  WARNING: Expected 277 builds but got {len(results_df)}")
        logger.warning(f"   This may indicate incorrect filtering or data issues")

    return results_df


def calculate_ranks_per_build(
    df: pd.DataFrame,
    probability_col: str = "probability",
    build_col: str = "Build_ID"
) -> pd.DataFrame:
    """
    Calculate priority ranks per build based on failure probabilities.

    Ranks are 1-indexed within each build, where:
    - rank=1 is the highest priority (highest probability)
    - rank=n is the lowest priority (lowest probability)

    Args:
        df: DataFrame with probability predictions
        probability_col: Name of column with failure probabilities
        build_col: Name of build ID column

    Returns:
        DataFrame with added 'rank' column
    """
    df = df.copy()

    # Calculate ranks per build (higher probability = lower rank number)
    df['rank'] = df.groupby(build_col)[probability_col] \
                   .rank(method='first', ascending=False) \
                   .astype(int)

    logger.info(f"Ranks calculated per build (rank range: {df['rank'].min()}-{df['rank'].max()})")

    return df


def generate_apfd_report(
    df: pd.DataFrame,
    method_name: str = "dual_stream_gnn",
    test_scenario: str = "full_test",
    output_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate complete APFD report with summary statistics.

    Args:
        df: DataFrame with test results (must have ranks already calculated)
        method_name: Name of the prioritization method
        test_scenario: Type of test scenario
        output_path: Optional path to save CSV report

    Returns:
        Tuple of (results_df, summary_stats)
        - results_df: Per-build APFD results
        - summary_stats: Dictionary with summary statistics
    """
    # Calculate APFD per build
    results_df = calculate_apfd_per_build(df, method_name, test_scenario)

    if len(results_df) == 0:
        logger.warning("No builds with failures found. APFD cannot be calculated.")
        return results_df, {}

    # Calculate summary statistics
    summary_stats = {
        'total_builds': len(results_df),
        'mean_apfd': float(results_df['apfd'].mean()),
        'median_apfd': float(results_df['apfd'].median()),
        'std_apfd': float(results_df['apfd'].std()),
        'min_apfd': float(results_df['apfd'].min()),
        'max_apfd': float(results_df['apfd'].max()),
        'total_test_cases': int(results_df['count_tc'].sum()),
        'mean_tc_per_build': float(results_df['count_tc'].mean()),
        'builds_apfd_1.0': int((results_df['apfd'] == 1.0).sum()),
        'builds_apfd_gte_0.7': int((results_df['apfd'] >= 0.7).sum()),
        'builds_apfd_gte_0.5': int((results_df['apfd'] >= 0.5).sum()),
        'builds_apfd_lt_0.5': int((results_df['apfd'] < 0.5).sum())
    }

    # Save to CSV if path provided
    if output_path:
        results_df.to_csv(output_path, index=False)
        logger.info(f"APFD per-build report saved to: {output_path}")

    return results_df, summary_stats


def print_apfd_summary(summary_stats: Dict):
    """Print formatted APFD summary statistics."""
    if not summary_stats:
        print("\n" + "="*70)
        print("APFD PER BUILD - NO DATA")
        print("="*70)
        print("No builds with failures found.")
        return

    print("\n" + "="*70)
    print("APFD PER BUILD - SUMMARY STATISTICS")
    print("="*70)
    print(f"Total builds analyzed: {summary_stats['total_builds']}")
    print(f"Total test cases: {summary_stats['total_test_cases']}")
    print(f"Mean TCs per build: {summary_stats['mean_tc_per_build']:.1f}")
    print(f"\nAPFD Statistics:")
    print(f"  Mean:   {summary_stats['mean_apfd']:.4f} ⭐ PRIMARY METRIC")
    print(f"  Median: {summary_stats['median_apfd']:.4f}")
    print(f"  Std:    {summary_stats['std_apfd']:.4f}")
    print(f"  Min:    {summary_stats['min_apfd']:.4f}")
    print(f"  Max:    {summary_stats['max_apfd']:.4f}")
    print(f"\nAPFD Distribution:")
    pct_1_0 = summary_stats['builds_apfd_1.0']/summary_stats['total_builds']*100
    pct_gte_0_7 = summary_stats['builds_apfd_gte_0.7']/summary_stats['total_builds']*100
    pct_gte_0_5 = summary_stats['builds_apfd_gte_0.5']/summary_stats['total_builds']*100
    pct_lt_0_5 = summary_stats['builds_apfd_lt_0.5']/summary_stats['total_builds']*100

    print(f"  Builds with APFD = 1.0:  {summary_stats['builds_apfd_1.0']:3d} ({pct_1_0:5.1f}%)")
    print(f"  Builds with APFD ≥ 0.7:  {summary_stats['builds_apfd_gte_0.7']:3d} ({pct_gte_0_7:5.1f}%)")
    print(f"  Builds with APFD ≥ 0.5:  {summary_stats['builds_apfd_gte_0.5']:3d} ({pct_gte_0_5:5.1f}%)")
    print(f"  Builds with APFD < 0.5:  {summary_stats['builds_apfd_lt_0.5']:3d} ({pct_lt_0_5:5.1f}%)")
    print("="*70)


def generate_prioritized_csv(
    df: pd.DataFrame,
    output_path: str,
    probability_col: str = "probability",
    label_col: str = "label_binary",
    build_col: str = "Build_ID"
):
    """
    Generate prioritized test cases CSV with ranks per build.

    Args:
        df: DataFrame with predictions
        output_path: Path to save CSV
        probability_col: Name of probability column
        label_col: Name of label column
        build_col: Name of build ID column
    """
    # Calculate ranks per build
    df_with_ranks = calculate_ranks_per_build(df, probability_col, build_col)

    # Calculate priority score (can include diversity if available)
    # For now, priority_score = probability (diversity_score = 0)
    df_with_ranks['diversity_score'] = 0.0
    df_with_ranks['priority_score'] = df_with_ranks[probability_col]

    # Select and order columns for output
    output_cols = [
        build_col,
        'TC_Key' if 'TC_Key' in df_with_ranks.columns else None,
        'TE_Test_Result' if 'TE_Test_Result' in df_with_ranks.columns else None,
        label_col,
        probability_col,
        'diversity_score',
        'priority_score',
        'rank'
    ]
    output_cols = [col for col in output_cols if col is not None and col in df_with_ranks.columns]

    # Save to CSV
    df_with_ranks[output_cols].to_csv(output_path, index=False)
    logger.info(f"Prioritized test cases saved to: {output_path}")

    return df_with_ranks


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python apfd.py <predictions_csv> [output_apfd_csv]")
        print("Example: python apfd.py results/experiment_012/predictions.csv results/experiment_012/apfd_per_build.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None

    # Load data
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Calculate ranks if not present
    if 'rank' not in df.columns:
        print("Calculating ranks per build...")
        df = calculate_ranks_per_build(df)

    # Generate report
    results_df, summary_stats = generate_apfd_report(
        df,
        method_name="dual_stream_gnn",
        test_scenario="full_test",
        output_path=output_csv
    )

    # Print summary
    print_apfd_summary(summary_stats)

    # Show sample results
    if len(results_df) > 0:
        print(f"\nSample of results (first 10 builds):")
        print(results_df.head(10).to_string(index=False))
