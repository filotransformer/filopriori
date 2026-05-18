#!/usr/bin/env python3
"""
RTPTorrent Preprocessor for Filo-Priori main.py.

Converts all 20 raw MSR2 project CSVs into the format expected by main.py:
  Build_ID, TC_Key, TE_Summary, TC_Steps, TE_Test_Result, commit, Build_Test_Start_Date

Reuses parse_test_name() and generate_semantic_text() from preprocess_rtptorrent_ranking.py.

Usage:
    python scripts/preprocessing/preprocess_rtptorrent_for_main.py [--max-builds-per-project 500]
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
RAW_DIR = BASE_DIR / "datasets" / "02_rtptorrent" / "raw" / "MSR2"
OUTPUT_DIR = BASE_DIR / "datasets" / "02_rtptorrent" / "processed"

MIN_BUILDS = 5  # Skip projects with fewer builds


def parse_test_name(test_name: str) -> Dict[str, str]:
    """
    Parse Java fully qualified test name into components.

    Example: "com.squareup.okhttp.HttpResponseCacheTest"
    Returns: {
        'full_name': original name,
        'package': 'com.squareup.okhttp',
        'class_name': 'HttpResponseCacheTest',
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
    }


def generate_semantic_text(class_name: str) -> str:
    """
    Generate semantic text from test class name for embedding.

    Converts CamelCase to readable text:
    "HttpResponseCacheTest" -> "http response cache test"
    """
    # Remove "Test" suffix for cleaner text, but keep it in output
    name = class_name
    if name.endswith('Test'):
        name = name[:-4]

    # Convert camelCase/PascalCase to words
    words = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    words = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', words)

    return words.lower().strip()


def list_projects(raw_dir: Path) -> List[str]:
    """List all project directories containing a CSV file."""
    projects = []
    for item in sorted(raw_dir.iterdir()):
        if item.is_dir() and item.name not in ('repo',):
            csv_path = item / f"{item.name}.csv"
            if csv_path.exists():
                projects.append(item.name)
    return projects


def process_project(project_name: str, max_builds: int = 0) -> pd.DataFrame:
    """
    Process a single project CSV into main.py format.

    Args:
        project_name: Name of the project directory.
        max_builds: If > 0, keep only the last N builds (most recent).

    Returns a DataFrame with columns:
      Build_ID, TC_Key, TE_Summary, TC_Steps, TE_Test_Result, commit, Build_Test_Start_Date
    """
    csv_path = RAW_DIR / project_name / f"{project_name}.csv"
    df = pd.read_csv(csv_path)

    n_builds = df['travisJobId'].nunique()
    if n_builds < MIN_BUILDS:
        logger.warning(f"  Skipping {project_name}: only {n_builds} builds (min={MIN_BUILDS})")
        return None

    # Cap to the most recent max_builds builds
    if max_builds > 0 and n_builds > max_builds:
        all_builds = sorted(df['travisJobId'].unique())
        keep_builds = set(all_builds[-max_builds:])
        df = df[df['travisJobId'].isin(keep_builds)]
        logger.info(f"  Capped {project_name}: {n_builds} -> {max_builds} builds (kept most recent)")

    # Build output columns
    out = pd.DataFrame()
    out['Build_ID'] = project_name + '_' + df['travisJobId'].astype(str)

    # TC_Key: use class name (last part of fully qualified name)
    class_names = df['testName'].apply(lambda t: parse_test_name(t)['class_name'])
    out['TC_Key'] = project_name + '::' + class_names

    # TE_Summary: semantic text from class name
    out['TE_Summary'] = class_names.apply(generate_semantic_text)

    # TC_Steps: empty (RTPTorrent has no steps)
    out['TC_Steps'] = ''

    # TE_Test_Result: Fail if failures>0 or errors>0, else Pass
    out['TE_Test_Result'] = 'Pass'
    out.loc[(df['failures'] > 0) | (df['errors'] > 0), 'TE_Test_Result'] = 'Fail'

    # commit and Build_Test_Start_Date: not available in raw
    out['commit'] = ''
    out['Build_Test_Start_Date'] = ''

    # Keep travisJobId for temporal splitting
    out['_travisJobId'] = df['travisJobId']

    return out


def main():
    parser = argparse.ArgumentParser(
        description="RTPTorrent Preprocessor for Filo-Priori main.py"
    )
    parser.add_argument(
        '--max-builds-per-project', type=int, default=500,
        help='Max builds to keep per project (most recent). 0 = no limit. Default: 500'
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RTPTorrent Preprocessor for Filo-Priori main.py")
    logger.info("=" * 60)
    if args.max_builds_per_project > 0:
        logger.info(f"Build cap: {args.max_builds_per_project} builds per project")
    else:
        logger.info("Build cap: disabled (no limit)")

    if not RAW_DIR.exists():
        logger.error(f"Raw data directory not found: {RAW_DIR}")
        sys.exit(1)

    projects = list_projects(RAW_DIR)
    logger.info(f"Found {len(projects)} projects in {RAW_DIR}")

    all_train = []
    all_test = []
    stats = {
        'projects_processed': 0,
        'projects_skipped': 0,
        'total_train_rows': 0,
        'total_test_rows': 0,
        'total_train_builds': 0,
        'total_test_builds': 0,
        'total_failures': 0,
        'total_rows': 0,
    }

    for i, project in enumerate(projects, 1):
        logger.info(f"[{i}/{len(projects)}] Processing {project}...")
        result = process_project(project, max_builds=args.max_builds_per_project)

        if result is None:
            stats['projects_skipped'] += 1
            continue

        # Temporal split: 80% train, 20% test by build order
        builds = sorted(result['_travisJobId'].unique())
        n_train = int(len(builds) * 0.8)
        train_builds = set(builds[:n_train])
        test_builds = set(builds[n_train:])

        train_df = result[result['_travisJobId'].isin(train_builds)].drop(columns=['_travisJobId'])
        test_df = result[result['_travisJobId'].isin(test_builds)].drop(columns=['_travisJobId'])

        n_fail_train = (train_df['TE_Test_Result'] == 'Fail').sum()
        n_fail_test = (test_df['TE_Test_Result'] == 'Fail').sum()

        logger.info(f"  Builds: {len(builds)} (train={len(train_builds)}, test={len(test_builds)})")
        logger.info(f"  Rows: train={len(train_df)}, test={len(test_df)}")
        logger.info(f"  Failures: train={n_fail_train}, test={n_fail_test}")

        all_train.append(train_df)
        all_test.append(test_df)

        stats['projects_processed'] += 1
        stats['total_train_rows'] += len(train_df)
        stats['total_test_rows'] += len(test_df)
        stats['total_train_builds'] += len(train_builds)
        stats['total_test_builds'] += len(test_builds)
        stats['total_failures'] += n_fail_train + n_fail_test
        stats['total_rows'] += len(train_df) + len(test_df)

    if not all_train:
        logger.error("No projects processed!")
        sys.exit(1)

    # Concatenate all projects
    train_combined = pd.concat(all_train, ignore_index=True)
    test_combined = pd.concat(all_test, ignore_index=True)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_combined.to_csv(train_path, index=False)
    test_combined.to_csv(test_path, index=False)

    # Print statistics
    failure_rate = stats['total_failures'] / stats['total_rows'] * 100 if stats['total_rows'] > 0 else 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Projects processed: {stats['projects_processed']}")
    logger.info(f"Projects skipped:   {stats['projects_skipped']}")
    logger.info(f"Train rows:         {stats['total_train_rows']:,}")
    logger.info(f"Test rows:          {stats['total_test_rows']:,}")
    logger.info(f"Total rows:         {stats['total_rows']:,}")
    logger.info(f"Train builds:       {stats['total_train_builds']:,}")
    logger.info(f"Test builds:        {stats['total_test_builds']:,}")
    logger.info(f"Failure rate:       {failure_rate:.2f}%")
    logger.info(f"Output:")
    logger.info(f"  {train_path}")
    logger.info(f"  {test_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
