#!/usr/bin/env python3
"""
RTPTorrent Preprocessor for Learning-to-Rank TCP.

This script prepares the RTPTorrent dataset for learning-to-rank based
test case prioritization, preserving the ranking structure needed for APFD evaluation.

Key differences from classification approach:
- Maintains build-level grouping for ranking
- Preserves temporal ordering for train/test split
- Extracts features suitable for ranking models
- Generates comparison data against provided baselines

Usage:
    python scripts/preprocessing/preprocess_rtptorrent_ranking.py [options]

Options:
    --projects: Comma-separated list of projects (default: all)
    --max-builds: Maximum builds per project (default: all)
    --output-dir: Output directory (default: datasets/02_rtptorrent/processed_ranking)
"""

import argparse
import gc
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional psutil for memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
RAW_DIR = BASE_DIR / "datasets" / "02_rtptorrent" / "raw" / "MSR2"
DEFAULT_OUTPUT_DIR = BASE_DIR / "datasets" / "02_rtptorrent" / "processed_ranking"


def get_memory_usage() -> str:
    """Get current memory usage as a formatted string."""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / (1024**3)
        return f"{mem_gb:.2f} GB"
    return "N/A (install psutil)"


def log_memory(context: str = ""):
    """Log current memory usage."""
    if HAS_PSUTIL:
        mem = get_memory_usage()
        logger.info(f"  [Memory] {context}: {mem}")


# Recommended projects for different scenarios
SMALL_PROJECTS = ["dynjs@dynjs", "jsprit@jsprit", "brettwooldridge@HikariCP"]
MEDIUM_PROJECTS = ["square@okhttp", "jOOQ@jOOQ", "doanduyhai@Achilles"]
LARGE_PROJECTS = ["apache@sling", "facebook@buck", "Graylog2@graylog2-server"]


def list_available_projects(raw_dir: Path) -> List[str]:
    """List all available projects in RTPTorrent."""
    projects = []
    for item in raw_dir.iterdir():
        if item.is_dir() and '@' in item.name:
            # Check if main CSV exists
            csv_path = item / f"{item.name}.csv"
            if csv_path.exists():
                projects.append(item.name)
    return sorted(projects)


def parse_test_name(test_name: str) -> Dict[str, str]:
    """
    Parse Java fully qualified test name into components.

    Example: "com.squareup.okhttp.HttpResponseCacheTest"
    Returns: {
        'full_name': original name,
        'package': 'com.squareup.okhttp',
        'class_name': 'HttpResponseCacheTest',
        'simple_name': 'HttpResponseCacheTest'
    }
    """
    parts = test_name.rsplit('.', 1)
    if len(parts) == 2:
        package, class_name = parts
    else:
        package = ""
        class_name = test_name

    return {
        'full_name': test_name,
        'package': package,
        'class_name': class_name,
        'simple_name': class_name
    }


def generate_semantic_text(test_info: Dict[str, str]) -> str:
    """
    Generate semantic text from test class name for embedding.

    Converts camelCase to readable text:
    "HttpResponseCacheTest" -> "http response cache test"
    """
    class_name = test_info['class_name']

    # Remove "Test" suffix
    if class_name.endswith('Test'):
        class_name = class_name[:-4]

    # Convert camelCase to words
    words = re.sub(r'([a-z])([A-Z])', r'\1 \2', class_name)
    words = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', words)

    return words.lower().strip()


def load_project_data(project_dir: Path, project_name: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load test execution data and baseline data for a project.

    Returns: (test_executions_df, baselines_dict)
    """
    # Load main data
    main_csv = project_dir / f"{project_name}.csv"
    if not main_csv.exists():
        raise FileNotFoundError(f"Main CSV not found: {main_csv}")

    logger.info(f"  Loading: {main_csv.name}")
    df = pd.read_csv(main_csv)

    # Load baselines if available
    baseline_dir = project_dir / "baseline"
    baselines = {}

    if baseline_dir.exists():
        short_name = project_name.split('@')[1] if '@' in project_name else project_name
        for baseline_file in baseline_dir.glob(f"{short_name}@*.csv"):
            strategy = baseline_file.stem.split('@')[1]
            baselines[strategy] = pd.read_csv(baseline_file)
            logger.info(f"  Loaded baseline: {strategy}")

    return df, baselines if baselines else None


def extract_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract historical features for each test case at each build.

    OPTIMIZED VERSION: Uses vectorized operations instead of O(n²) loops.
    Speed improvement: ~100x faster on large projects.

    Features extracted:
    - recent_failure_count: Failures in last N builds
    - total_failure_count: Total historical failures
    - failure_rate: Historical failure rate
    - recent_execution_count: Executions in last N builds
    - avg_duration: Average execution duration
    - last_failure_recency: Builds since last failure
    """
    WINDOW_SIZE = 10  # Look back N builds

    log_memory("Before feature extraction")
    logger.info("    Sorting and preparing data...")

    # Sort by build number
    df = df.sort_values('travisJobId').copy()

    # Get unique builds in order
    builds = df['travisJobId'].unique()
    build_to_idx = {b: i for i, b in enumerate(builds)}
    df['build_idx'] = df['travisJobId'].map(build_to_idx)

    # Mark failures
    df['is_failure'] = ((df['failures'] > 0) | (df['errors'] > 0)).astype(int)

    logger.info(f"    Processing {len(builds):,} builds with {df['testName'].nunique():,} unique tests...")

    # =========================================================================
    # OPTIMIZED: Pre-compute cumulative statistics per test using groupby
    # This avoids O(n²) complexity from repeated DataFrame filtering
    # =========================================================================

    # Sort by test and build for efficient cumulative calculations
    df_sorted = df.sort_values(['testName', 'build_idx']).copy()

    # Group by test name
    logger.info("    Computing cumulative statistics per test (vectorized)...")

    # For each test, compute cumulative stats up to (but not including) current row
    # Using shift(1) + cumsum() pattern for "historical" (excluding current)

    grouped = df_sorted.groupby('testName')

    # Cumulative executions (shifted to exclude current)
    df_sorted['total_executions'] = grouped.cumcount()  # 0-indexed count before current

    # Cumulative failures (shifted to exclude current)
    df_sorted['total_failures'] = grouped['is_failure'].transform(
        lambda x: x.shift(1, fill_value=0).cumsum()
    )

    # Cumulative duration sum for average calculation
    df_sorted['cum_duration'] = grouped['duration'].transform(
        lambda x: x.shift(1, fill_value=0).cumsum()
    )

    # Average duration (handle division by zero)
    df_sorted['avg_duration'] = np.where(
        df_sorted['total_executions'] > 0,
        df_sorted['cum_duration'] / df_sorted['total_executions'],
        df_sorted['duration']  # Use current duration for new tests
    )

    # Failure rate
    df_sorted['failure_rate'] = np.where(
        df_sorted['total_executions'] > 0,
        df_sorted['total_failures'] / df_sorted['total_executions'],
        0
    )

    # Is new test (first occurrence)
    df_sorted['is_new_test'] = (df_sorted['total_executions'] == 0).astype(int)

    # =========================================================================
    # Recent window statistics (last WINDOW_SIZE builds per test)
    # =========================================================================
    logger.info(f"    Computing recent window statistics (last {WINDOW_SIZE} builds)...")

    # Recent executions: count of executions in window (excluding current)
    df_sorted['recent_executions'] = grouped['build_idx'].transform(
        lambda x: x.rolling(window=WINDOW_SIZE + 1, min_periods=1).count().shift(1, fill_value=0)
    ).astype(int)

    # Clip to actual window size
    df_sorted['recent_executions'] = df_sorted['recent_executions'].clip(upper=WINDOW_SIZE)

    # Recent failures in window
    df_sorted['recent_failures'] = grouped['is_failure'].transform(
        lambda x: x.rolling(window=WINDOW_SIZE + 1, min_periods=1).sum().shift(1, fill_value=0)
    ).astype(int)

    # =========================================================================
    # Last failure recency
    # =========================================================================
    logger.info("    Computing last failure recency...")

    # Track build_idx where last failure occurred for each test
    df_sorted['_last_failure_build'] = grouped.apply(
        lambda g: g['build_idx'].where(g['is_failure'] == 1).ffill().shift(1),
        include_groups=False
    ).reset_index(level=0, drop=True)

    # Recency = current build_idx - last failure build_idx
    # If never failed, use build_idx + 1
    df_sorted['last_failure_recency'] = np.where(
        df_sorted['_last_failure_build'].notna(),
        df_sorted['build_idx'] - df_sorted['_last_failure_build'],
        df_sorted['build_idx'] + 1  # Never failed
    ).astype(int)

    # Clean up temporary column
    df_sorted.drop('_last_failure_build', axis=1, inplace=True)
    df_sorted.drop('cum_duration', axis=1, inplace=True)

    # =========================================================================
    # Restore original order by travisJobId
    # =========================================================================
    logger.info("    Finalizing features...")
    result = df_sorted.sort_values(['build_idx', 'index']).reset_index(drop=True)

    # Select and order columns
    columns = [
        'travisJobId', 'testName', 'build_idx', 'index', 'duration', 'count',
        'failures', 'errors', 'skipped', 'is_failure',
        'total_executions', 'total_failures', 'failure_rate',
        'recent_failures', 'recent_executions', 'avg_duration',
        'last_failure_recency', 'is_new_test'
    ]

    result = result[columns]

    logger.info(f"    ✓ Feature extraction complete: {len(result):,} rows")
    log_memory("After feature extraction")

    # Clean up intermediate data
    del df_sorted
    gc.collect()

    return result


def split_train_test_temporal(df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally by build.

    Training set: First (1-test_ratio) builds
    Test set: Last test_ratio builds
    """
    builds = df['build_idx'].unique()
    builds = sorted(builds)

    n_test = int(len(builds) * test_ratio)
    n_train = len(builds) - n_test

    train_builds = set(builds[:n_train])
    test_builds = set(builds[n_train:])

    train_df = df[df['build_idx'].isin(train_builds)].copy()
    test_df = df[df['build_idx'].isin(test_builds)].copy()

    return train_df, test_df


def compute_apfd(ranked_df: pd.DataFrame) -> float:
    """
    Compute APFD (Average Percentage of Faults Detected) for a single build.

    APFD = 1 - (sum(TFi) / (n * m)) + 1/(2n)

    Where:
    - TFi = position of test that reveals fault i
    - n = total number of tests
    - m = total number of faults
    """
    n = len(ranked_df)
    failures = ranked_df[ranked_df['is_failure'] == 1]
    m = len(failures)

    if m == 0:
        return None  # No failures in this build

    # Get positions of failures (1-indexed)
    failure_positions = failures['rank'].values

    apfd = 1 - (failure_positions.sum() / (n * m)) + 1 / (2 * n)
    return apfd


def evaluate_baselines(test_df: pd.DataFrame, baselines: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Evaluate APFD for each baseline strategy on test builds.
    """
    results = {}

    for strategy, baseline_df in baselines.items():
        apfd_values = []

        # Get test builds
        test_builds = test_df['travisJobId'].unique()

        for build_id in test_builds:
            # Get baseline ranking for this build
            build_baseline = baseline_df[baseline_df['travisJobId'] == build_id].copy()
            if len(build_baseline) == 0:
                continue

            # Use index as rank (already prioritized)
            build_baseline['rank'] = build_baseline['index'] + 1
            build_baseline['is_failure'] = ((build_baseline['failures'] > 0) |
                                            (build_baseline['errors'] > 0)).astype(int)

            apfd = compute_apfd(build_baseline)
            if apfd is not None:
                apfd_values.append(apfd)

        if apfd_values:
            results[strategy] = {
                'mean_apfd': np.mean(apfd_values),
                'median_apfd': np.median(apfd_values),
                'std_apfd': np.std(apfd_values),
                'n_builds': len(apfd_values)
            }

    return results


def process_project(
    project_name: str,
    raw_dir: Path,
    max_builds: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Process a single project for learning-to-rank.

    Returns: (train_df, test_df, metadata)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {project_name}")
    logger.info("=" * 60)

    project_dir = raw_dir / project_name

    # Load data
    df, baselines = load_project_data(project_dir, project_name)

    logger.info(f"  Raw data: {len(df):,} rows, {df['travisJobId'].nunique():,} builds")

    # Limit builds if requested
    if max_builds:
        builds = df['travisJobId'].unique()[:max_builds]
        df = df[df['travisJobId'].isin(builds)]
        logger.info(f"  Limited to {max_builds} builds: {len(df):,} rows")

    # Extract features
    logger.info("  Extracting historical features...")
    features_df = extract_historical_features(df)

    # Add semantic text
    logger.info("  Generating semantic text...")
    test_info = features_df['testName'].apply(parse_test_name)
    features_df['semantic_text'] = test_info.apply(generate_semantic_text)
    features_df['package'] = test_info.apply(lambda x: x['package'])
    features_df['class_name'] = test_info.apply(lambda x: x['class_name'])

    # Add project identifier
    features_df['project'] = project_name

    # Split train/test
    logger.info("  Splitting train/test...")
    train_df, test_df = split_train_test_temporal(features_df, test_ratio=0.2)

    # Compute metadata
    metadata = {
        'project': project_name,
        'total_rows': len(features_df),
        'total_builds': features_df['travisJobId'].nunique(),
        'total_tests': features_df['testName'].nunique(),
        'train_rows': len(train_df),
        'train_builds': train_df['travisJobId'].nunique(),
        'test_rows': len(test_df),
        'test_builds': test_df['travisJobId'].nunique(),
        'failure_rate': features_df['is_failure'].mean(),
        'builds_with_failures': features_df.groupby('travisJobId')['is_failure'].max().sum()
    }

    # Evaluate baselines if available
    if baselines:
        logger.info("  Evaluating baselines...")
        baseline_results = evaluate_baselines(test_df, baselines)
        metadata['baselines'] = baseline_results

    logger.info(f"  Train: {metadata['train_rows']:,} rows, {metadata['train_builds']:,} builds")
    logger.info(f"  Test: {metadata['test_rows']:,} rows, {metadata['test_builds']:,} builds")
    logger.info(f"  Failure rate: {metadata['failure_rate']*100:.2f}%")

    # Clean up intermediate DataFrames
    del df, features_df
    if baselines:
        del baselines
    gc.collect()
    log_memory("After project cleanup")

    return train_df, test_df, metadata


def main():
    """Main preprocessing routine."""
    parser = argparse.ArgumentParser(description="Preprocess RTPTorrent for Learning-to-Rank")
    parser.add_argument('--projects', type=str, default=None,
                        help='Comma-separated list of projects (default: small projects)')
    parser.add_argument('--preset', type=str, choices=['small', 'medium', 'large', 'all'],
                        default='small', help='Project preset to use')
    parser.add_argument('--max-builds', type=int, default=None,
                        help='Maximum builds per project')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--list-projects', action='store_true',
                        help='List available projects and exit')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RTPTorrent Learning-to-Rank Preprocessor")
    logger.info("=" * 60)

    # Check data directory
    if not RAW_DIR.exists():
        logger.error(f"Data directory not found: {RAW_DIR}")
        logger.error("Please download RTPTorrent first:")
        logger.error("  python scripts/preprocessing/download_rtptorrent.py")
        sys.exit(1)

    # List available projects
    available = list_available_projects(RAW_DIR)
    logger.info(f"Found {len(available)} projects")

    if args.list_projects:
        print("\nAvailable projects:")
        for p in available:
            print(f"  - {p}")
        print(f"\nPresets:")
        print(f"  small:  {SMALL_PROJECTS}")
        print(f"  medium: {MEDIUM_PROJECTS}")
        print(f"  large:  {LARGE_PROJECTS}")
        return

    # Determine projects to process
    if args.projects:
        projects = [p.strip() for p in args.projects.split(',')]
    elif args.preset == 'small':
        projects = [p for p in SMALL_PROJECTS if p in available]
    elif args.preset == 'medium':
        projects = [p for p in MEDIUM_PROJECTS if p in available]
    elif args.preset == 'large':
        projects = [p for p in LARGE_PROJECTS if p in available]
    else:  # all
        projects = available

    logger.info(f"Processing {len(projects)} projects: {projects}")

    # Output directory
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each project
    all_train = []
    all_test = []
    all_metadata = {}

    log_memory("Before processing projects")

    for i, project in enumerate(projects, 1):
        if project not in available:
            logger.warning(f"Project not found: {project}")
            continue

        try:
            logger.info(f"\n[{i}/{len(projects)}] Starting {project}")
            train_df, test_df, metadata = process_project(
                project, RAW_DIR, args.max_builds
            )
            all_train.append(train_df)
            all_test.append(test_df)
            all_metadata[project] = metadata

            # Clean up after each project
            del train_df, test_df
            gc.collect()
            log_memory(f"After processing {project}")

        except Exception as e:
            logger.error(f"Error processing {project}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not all_train:
        logger.error("No data processed!")
        sys.exit(1)

    # Combine data
    logger.info("\n" + "=" * 60)
    logger.info("Combining all projects...")
    log_memory("Before concat")

    train_df = pd.concat(all_train, ignore_index=True)
    test_df = pd.concat(all_test, ignore_index=True)

    # Free memory from individual DataFrames
    del all_train, all_test
    gc.collect()
    log_memory("After concat and cleanup")

    # Create unique Build_ID across projects
    train_df['Build_ID'] = train_df['project'] + '_' + train_df['travisJobId'].astype(str)
    test_df['Build_ID'] = test_df['project'] + '_' + test_df['travisJobId'].astype(str)

    # Create unique TC_Key
    train_df['TC_Key'] = train_df['project'] + '::' + train_df['testName']
    test_df['TC_Key'] = test_df['project'] + '::' + test_df['testName']

    # Save data
    logger.info(f"Saving to {output_dir}...")

    # Define columns to save
    columns = [
        'Build_ID', 'TC_Key', 'project', 'travisJobId', 'testName',
        'build_idx', 'index', 'duration', 'count', 'failures', 'errors',
        'skipped', 'is_failure', 'total_executions', 'total_failures',
        'failure_rate', 'recent_failures', 'recent_executions',
        'avg_duration', 'last_failure_recency', 'is_new_test',
        'semantic_text', 'package', 'class_name'
    ]

    train_df[columns].to_csv(output_dir / "train.csv", index=False)
    test_df[columns].to_csv(output_dir / "test.csv", index=False)

    # Save metadata
    combined_metadata = {
        'projects': all_metadata,
        'combined': {
            'train_rows': len(train_df),
            'train_builds': train_df['Build_ID'].nunique(),
            'test_rows': len(test_df),
            'test_builds': test_df['Build_ID'].nunique(),
            'total_tests': train_df['TC_Key'].nunique(),
            'failure_rate': (train_df['is_failure'].sum() + test_df['is_failure'].sum()) /
                           (len(train_df) + len(test_df))
        },
        'preprocessing_date': datetime.now().isoformat(),
        'parameters': {
            'projects': projects,
            'max_builds': args.max_builds,
            'preset': args.preset
        }
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(combined_metadata, f, indent=2, default=str)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing Complete!")
    logger.info("=" * 60)

    logger.info(f"\nCombined Statistics:")
    logger.info(f"  Train: {len(train_df):,} rows, {train_df['Build_ID'].nunique():,} builds")
    logger.info(f"  Test: {len(test_df):,} rows, {test_df['Build_ID'].nunique():,} builds")
    logger.info(f"  Unique tests: {train_df['TC_Key'].nunique():,}")
    logger.info(f"  Overall failure rate: {combined_metadata['combined']['failure_rate']*100:.2f}%")

    # Print baseline results if available
    for proj, meta in all_metadata.items():
        if 'baselines' in meta:
            logger.info(f"\n{proj} Baseline APFD (test set):")
            for strategy, results in meta['baselines'].items():
                logger.info(f"  {strategy}: {results['mean_apfd']:.4f} (n={results['n_builds']})")

    logger.info(f"\nOutput files:")
    logger.info(f"  {output_dir / 'train.csv'}")
    logger.info(f"  {output_dir / 'test.csv'}")
    logger.info(f"  {output_dir / 'metadata.json'}")

    logger.info("\n" + "=" * 60)
    logger.info("Next: Run learning-to-rank experiment:")
    logger.info("  python main_rtptorrent.py --config configs/experiment_rtptorrent_l2r.yaml")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
