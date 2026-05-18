#!/usr/bin/env python3
"""
RETECS Experiment for RTPTorrent Dataset

Runs RETECS (Network Agent with tcfail reward) on the 02_rtptorrent dataset,
iterating over all projects in MSR2/.

Each project uses temporal split: 80% train, 20% test.
Only evaluates on test builds with at least 1 failure.

Usage:
    python experiments/run_retecs_rtptorrent.py
"""

import gc
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.retecs import RETECSPrioritizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'data_dir': 'datasets/02_rtptorrent/raw/MSR2',
    'seed': 42,
    'output_dir': 'results/retecs_rtptorrent',
    'method_name': 'RETECS',
    'agent_type': 'network',
    'reward_func': 'tcfail',
    'train_ratio': 0.8,
}

# Directories to skip
SKIP_DIRS = {'repo'}


# =============================================================================
# APFD CALCULATION — Identical to Filo-Priori
# =============================================================================

def calculate_apfd_single_build(ranks: np.ndarray, labels: np.ndarray) -> Optional[float]:
    labels_arr = np.array(labels)
    ranks_arr = np.array(ranks)

    n_tests = int(len(labels_arr))
    fail_indices = np.where(labels_arr.astype(int) != 0)[0]
    n_failures = len(fail_indices)

    if n_failures == 0:
        return None
    if n_tests == 1:
        return 1.0

    failure_ranks = ranks_arr[fail_indices]
    apfd = 1.0 - float(failure_ranks.sum()) / float(n_failures * n_tests) + 1.0 / float(2.0 * n_tests)
    return float(np.clip(apfd, 0.0, 1.0))


# =============================================================================
# DATA LOADING
# =============================================================================

def get_project_dirs(data_dir: Path) -> List[Path]:
    """Get all project directories in MSR2/."""
    projects = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and d.name not in SKIP_DIRS:
            csv_file = d / f"{d.name}.csv"
            if csv_file.exists():
                projects.append(d)
    return projects


def load_project_data(project_dir: Path) -> pd.DataFrame:
    """Load project CSV and compute failure labels."""
    csv_path = project_dir / f"{project_dir.name}.csv"
    df = pd.read_csv(csv_path)

    # Failure = failures > 0 OR errors > 0
    df['is_failure'] = ((df['failures'] > 0) | (df['errors'] > 0)).astype(int)

    return df


# =============================================================================
# PER-PROJECT EXPERIMENT
# =============================================================================

def run_project(project_dir: Path, config: Dict) -> Optional[Dict]:
    """Run RETECS on a single RTPTorrent project."""
    project_name = project_dir.name
    logger.info(f"\n{'='*50}")
    logger.info(f"Project: {project_name}")
    logger.info(f"{'='*50}")

    try:
        df = load_project_data(project_dir)
    except Exception as e:
        logger.warning(f"Failed to load {project_name}: {e}")
        return None

    # Get unique builds in order
    builds = df['travisJobId'].unique().tolist()
    n_builds = len(builds)

    if n_builds < 5:
        logger.warning(f"Skipping {project_name}: only {n_builds} builds")
        return None

    # Temporal split
    train_idx = int(n_builds * config['train_ratio'])
    train_builds = builds[:train_idx]
    test_builds = builds[train_idx:]

    logger.info(f"  Total builds: {n_builds}, Train: {len(train_builds)}, Test: {len(test_builds)}")

    # Initialize RETECS
    prioritizer = RETECSPrioritizer(
        agent_type=config['agent_type'],
        reward_func=config['reward_func'],
        seed=config['seed']
    )

    # Pre-group entire DataFrame — O(n) total instead of O(n_builds * n)
    all_grouped = df.groupby('travisJobId')

    # Training phase
    train_start = time.time()
    n_train = len(train_builds)
    train_log_interval = max(1, n_train // 10)

    for i, build_id in enumerate(train_builds):
        if build_id not in all_grouped.groups:
            continue

        build_df = all_grouped.get_group(build_id)

        # Vectorized verdict and duration aggregation
        verdicts = build_df.groupby('testName')['is_failure'].max().to_dict()
        durations_dict = build_df.groupby('testName')['duration'].last().to_dict()
        test_ids = list(verdicts.keys())

        prioritizer.train_on_build(test_ids, verdicts, durations_dict)

        if (i + 1) % train_log_interval == 0 or (i + 1) == n_train:
            elapsed = time.time() - train_start
            logger.info(f"  Train progress: {i+1}/{n_train} builds [{elapsed:.0f}s]")

    train_time = time.time() - train_start
    logger.info(f"  Training completed in {train_time:.1f}s")

    # Free full DataFrame, keep only test data
    del df
    gc.collect()

    # Evaluation phase — only builds with failures
    eval_start = time.time()
    build_results = []
    all_apfd_scores = []
    n_test = len(test_builds)
    eval_log_interval = max(1, n_test // 20)

    for i, build_id in enumerate(test_builds):
        if build_id not in all_grouped.groups:
            continue

        build_df = all_grouped.get_group(build_id)

        # Vectorized verdict and duration aggregation
        verdicts = build_df.groupby('testName')['is_failure'].max().to_dict()
        durations_dict = build_df.groupby('testName')['duration'].last().to_dict()
        test_ids = list(verdicts.keys())

        n_failures = sum(verdicts.values())
        if n_failures == 0:
            prioritizer.update_history(test_ids, verdicts, durations_dict)
            continue

        ranking = prioritizer.prioritize(test_ids, durations_dict)
        labels = np.array([verdicts[tc] for tc in ranking])
        ranks = np.arange(1, len(ranking) + 1)
        apfd = calculate_apfd_single_build(ranks, labels)

        if apfd is not None:
            all_apfd_scores.append(apfd)
            build_results.append({
                'method_name': config['method_name'],
                'project': project_name,
                'build_id': build_id,
                'test_scenario': 'rtptorrent',
                'count_tc': len(test_ids),
                'count_commits': 0,
                'apfd': apfd,
                'time': 0.0
            })

        prioritizer.update_history(test_ids, verdicts, durations_dict)

        # Progress logging
        if (i + 1) % eval_log_interval == 0 or (i + 1) == n_test:
            elapsed = time.time() - eval_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_test - i - 1) / rate if rate > 0 else 0
            logger.info(f"  Eval progress: {i+1}/{n_test} builds "
                       f"({len(all_apfd_scores)} with failures) "
                       f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

    eval_time = time.time() - eval_start

    # Free grouped data
    del all_grouped
    gc.collect()

    if not all_apfd_scores:
        logger.warning(f"No builds with failures in test set for {project_name}")
        return None

    project_result = {
        'project': project_name,
        'n_builds_total': n_builds,
        'n_builds_train': len(train_builds),
        'n_builds_test': len(test_builds),
        'n_builds_with_failures': len(all_apfd_scores),
        'mean_apfd': float(np.mean(all_apfd_scores)),
        'std_apfd': float(np.std(all_apfd_scores)),
        'median_apfd': float(np.median(all_apfd_scores)),
        'min_apfd': float(np.min(all_apfd_scores)),
        'max_apfd': float(np.max(all_apfd_scores)),
        'build_results': build_results,
        'train_time': train_time,
        'eval_time': eval_time,
    }

    logger.info(f"  Builds with failures: {len(all_apfd_scores)}")
    logger.info(f"  Mean APFD: {project_result['mean_apfd']:.4f}")

    return project_result


# =============================================================================
# MAIN
# =============================================================================

def _save_results(output_dir, all_project_results, all_build_results, total_start):
    """Save current results incrementally."""
    total_time = time.time() - total_start

    if all_build_results:
        build_df = pd.DataFrame(all_build_results)
        build_df.to_csv(output_dir / 'apfd_per_build_FULL_testcsv.csv', index=False)

    if all_project_results:
        project_summary = pd.DataFrame([{
            'project': r['project'],
            'n_builds_total': r['n_builds_total'],
            'n_builds_with_failures': r['n_builds_with_failures'],
            'mean_apfd': r['mean_apfd'],
            'std_apfd': r['std_apfd'],
            'median_apfd': r['median_apfd'],
        } for r in all_project_results])
        project_summary.to_csv(output_dir / 'per_project_apfd.csv', index=False)

    all_apfd = [r['mean_apfd'] for r in all_project_results]
    aggregate = {
        'method': CONFIG['method_name'],
        'n_projects': len(all_project_results),
        'n_builds_total': sum(r['n_builds_with_failures'] for r in all_project_results),
        'grand_mean_apfd': float(np.mean(all_apfd)) if all_apfd else 0.0,
        'grand_std_apfd': float(np.std(all_apfd)) if all_apfd else 0.0,
        'grand_median_apfd': float(np.median(all_apfd)) if all_apfd else 0.0,
        'total_time_seconds': total_time,
        'timestamp': datetime.now().isoformat(),
        'per_project': [{k: v for k, v in r.items() if k != 'build_results'}
                        for r in all_project_results]
    }

    with open(output_dir / 'aggregate_results.json', 'w') as f:
        json.dump(aggregate, f, indent=2, default=str)

    summary = {
        'method': CONFIG['method_name'],
        'config': CONFIG,
        'summary': {
            'mean_apfd': aggregate['grand_mean_apfd'],
            'std_apfd': aggregate['grand_std_apfd'],
            'n_projects': aggregate['n_projects'],
            'n_builds': aggregate['n_builds_total'],
        },
        'timing': {'total_time_seconds': total_time},
        'timestamp': datetime.now().isoformat()
    }
    with open(output_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    with open(output_dir / 'comparison_summary.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RETECS - RTPTorrent Dataset Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Projects analyzed: {aggregate['n_projects']}\n")
        f.write(f"Total builds with failures: {aggregate['n_builds_total']}\n\n")
        f.write(f"Grand Mean APFD: {aggregate['grand_mean_apfd']:.4f} (PRIMARY METRIC)\n")
        f.write(f"Grand Std APFD:  {aggregate['grand_std_apfd']:.4f}\n\n")
        f.write(f"Per-project results:\n")
        for r in all_project_results:
            f.write(f"  {r['project']:40s} APFD={r['mean_apfd']:.4f} "
                    f"(n={r['n_builds_with_failures']})\n")
        f.write(f"\nTotal time: {total_time:.2f}s\n")
        f.write("=" * 70 + "\n")


def main():
    print("\n" + "=" * 70)
    print("RETECS Experiment - RTPTorrent Dataset")
    print("=" * 70 + "\n")

    np.random.seed(CONFIG['seed'])
    random.seed(CONFIG['seed'])

    data_dir = PROJECT_ROOT / CONFIG['data_dir']
    output_dir = PROJECT_ROOT / CONFIG['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    projects = get_project_dirs(data_dir)
    logger.info(f"Found {len(projects)} projects")

    total_start = time.time()
    all_project_results = []
    all_build_results = []

    for proj_idx, project_dir in enumerate(projects, 1):
        logger.info(f"\n[{proj_idx}/{len(projects)}] Starting {project_dir.name}...")

        result = run_project(project_dir, CONFIG)
        if result is not None:
            all_project_results.append(result)
            all_build_results.extend(result['build_results'])

            # Save incrementally after each project (in case of crash)
            _save_results(output_dir, all_project_results, all_build_results, total_start)

        # Clean up between projects
        gc.collect()

    total_time = time.time() - total_start

    # Final save
    _save_results(output_dir, all_project_results, all_build_results, total_start)

    # Print results
    print("\n" + "=" * 70)
    print("RETECS - RTPTorrent Results Summary")
    print("=" * 70)

    all_apfd = [r['mean_apfd'] for r in all_project_results]
    if all_apfd:
        print(f"\nProjects: {len(all_project_results)}")
        print(f"Grand Mean APFD: {np.mean(all_apfd):.4f} <<< PRIMARY METRIC")
        print(f"Grand Std APFD:  {np.std(all_apfd):.4f}")
        print(f"\nPer-project:")
        for r in all_project_results:
            print(f"  {r['project']:40s} APFD={r['mean_apfd']:.4f} "
                  f"(n={r['n_builds_with_failures']}, "
                  f"train={r['train_time']:.0f}s, eval={r['eval_time']:.0f}s)")

    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Results saved to: {CONFIG['output_dir']}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
