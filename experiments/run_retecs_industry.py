#!/usr/bin/env python3
"""
RETECS Experiment for Industry Dataset

Runs RETECS (Network Agent with tcfail reward) on the 01_industry dataset.
Results are scientifically comparable with Filo-Priori, DeepOrder, and TCP-Net.

Guarantees:
1. Uses EXACTLY the same split train/test (train.csv and test.csv)
2. Calculates APFD PER BUILD using the same formula as Filo-Priori
3. Only considers builds with at least 1 failure (same criteria)
4. Generates results in the same CSV format
5. Uses the same seed (42) for reproducibility

Usage:
    python experiments/run_retecs_industry.py
"""

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
from src.evaluation.apfd import calculate_apfd_single_build

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
    'train_path': 'datasets/01_industry/train.csv',
    'test_path': 'datasets/01_industry/test.csv',
    'build_col': 'Build_ID',
    'test_col': 'TC_Key',
    'result_col': 'TE_Test_Result',
    'duration_col': None,
    'seed': 42,
    'output_dir': 'results/retecs_industry',
    'method_name': 'RETECS',
    'test_scenario': 'industry_dataset',
    'agent_type': 'network',
    'reward_func': 'tcfail',
}


# =============================================================================
# DATA HELPERS
# =============================================================================

def normalize_results(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    df = df.copy()
    col = config['result_col']
    df[col] = df[col].astype(str).str.strip().apply(lambda x: 'Fail' if x == 'Fail' else 'Pass')
    return df


def load_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading datasets...")
    train_path = PROJECT_ROOT / config['train_path']
    test_path = PROJECT_ROOT / config['test_path']

    raw_train_df = pd.read_csv(train_path)
    raw_test_df = pd.read_csv(test_path)

    train_df = normalize_results(raw_train_df, config)
    test_df = normalize_results(raw_test_df, config)

    logger.info(f"  Train: {len(train_df):,} rows, {train_df[config['build_col']].nunique()} builds")
    logger.info(f"  Test: {len(test_df):,} rows, {test_df[config['build_col']].nunique()} builds")
    return train_df, test_df


def get_builds_with_failures(df: pd.DataFrame, config: Dict) -> List:
    build_col = config['build_col']
    result_col = config['result_col']
    builds_with_failures = []
    for build_id, group in df.groupby(build_col):
        if (group[result_col].astype(str).str.strip() == 'Fail').any():
            builds_with_failures.append(build_id)
    return builds_with_failures


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_experiment(config: Dict) -> Dict:
    start_time = time.time()

    np.random.seed(config['seed'])
    random.seed(config['seed'])

    train_df, test_df = load_data(config)

    build_col = config['build_col']
    test_col = config['test_col']
    result_col = config['result_col']

    # Initialize RETECS
    logger.info(f"Initializing RETECS (agent={config['agent_type']}, reward={config['reward_func']})")
    prioritizer = RETECSPrioritizer(
        agent_type=config['agent_type'],
        reward_func=config['reward_func'],
        seed=config['seed']
    )

    # Training phase: iterate through all training builds
    train_builds = train_df[build_col].unique().tolist()
    logger.info(f"Training on {len(train_builds)} builds...")
    train_start = time.time()

    for build_id in train_builds:
        build_df = train_df[train_df[build_col] == build_id]
        test_ids = build_df[test_col].unique().tolist()

        verdicts = {}
        for _, row in build_df.iterrows():
            tc = row[test_col]
            verdict = 1 if str(row[result_col]).strip() == 'Fail' else 0
            verdicts[tc] = max(verdicts.get(tc, 0), verdict)

        prioritizer.train_on_build(test_ids, verdicts)

    train_time = time.time() - train_start
    logger.info(f"Training completed in {train_time:.2f}s")

    # Evaluation phase
    test_builds_with_failures = get_builds_with_failures(test_df, config)
    logger.info(f"Evaluating on {len(test_builds_with_failures)} builds with failures...")
    eval_start = time.time()

    build_results = []
    all_apfd_scores = []

    for build_id in test_builds_with_failures:
        build_df = test_df[test_df[build_col] == build_id]
        test_ids = build_df[test_col].unique().tolist()

        verdicts = {}
        for _, row in build_df.iterrows():
            tc = row[test_col]
            verdict = 1 if str(row[result_col]).strip() == 'Fail' else 0
            verdicts[tc] = max(verdicts.get(tc, 0), verdict)

        n_failures = sum(verdicts.values())
        if n_failures == 0:
            continue

        ranking = prioritizer.prioritize(test_ids)
        labels = np.array([verdicts[tc] for tc in ranking])
        ranks = np.arange(1, len(ranking) + 1)
        apfd = calculate_apfd_single_build(ranks, labels)

        if apfd is not None:
            all_apfd_scores.append(apfd)
            build_results.append({
                'method_name': config['method_name'],
                'build_id': build_id,
                'test_scenario': config['test_scenario'],
                'count_tc': len(test_ids),
                'count_commits': 0,
                'apfd': apfd,
                'time': 0.0
            })

        # Update history for temporal consistency
        prioritizer.update_history(test_ids, verdicts)

    eval_time = time.time() - eval_start
    total_time = time.time() - start_time

    if all_apfd_scores:
        summary = {
            'mean_apfd': float(np.mean(all_apfd_scores)),
            'std_apfd': float(np.std(all_apfd_scores)),
            'median_apfd': float(np.median(all_apfd_scores)),
            'min_apfd': float(np.min(all_apfd_scores)),
            'max_apfd': float(np.max(all_apfd_scores)),
            'n_builds': len(all_apfd_scores),
            'builds_apfd_1.0': sum(1 for x in all_apfd_scores if x == 1.0),
            'builds_apfd_gte_0.7': sum(1 for x in all_apfd_scores if x >= 0.7),
            'builds_apfd_gte_0.5': sum(1 for x in all_apfd_scores if x >= 0.5),
            'builds_apfd_lt_0.5': sum(1 for x in all_apfd_scores if x < 0.5),
        }
    else:
        summary = {'mean_apfd': 0.0, 'std_apfd': 0.0, 'median_apfd': 0.0,
                    'min_apfd': 0.0, 'max_apfd': 0.0, 'n_builds': 0}

    return {
        'method': config['method_name'],
        'config': config,
        'summary': summary,
        'build_results': build_results,
        'timing': {
            'train_time_seconds': train_time,
            'eval_time_seconds': eval_time,
            'total_time_seconds': total_time
        },
        'timestamp': datetime.now().isoformat()
    }


def save_results(results: Dict, config: Dict):
    output_dir = PROJECT_ROOT / config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-build APFD CSV
    build_results_df = pd.DataFrame(results['build_results'])
    apfd_csv_path = output_dir / 'apfd_per_build_FULL_testcsv.csv'
    build_results_df.to_csv(apfd_csv_path, index=False)
    logger.info(f"Saved APFD per build to: {apfd_csv_path}")

    # Summary JSON
    summary_path = output_dir / 'experiment_summary.json'
    with open(summary_path, 'w') as f:
        save_obj = results.copy()
        save_obj['config'] = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
        json.dump(save_obj, f, indent=2, default=str)
    logger.info(f"Saved experiment summary to: {summary_path}")

    # Comparison summary
    comparison_path = output_dir / 'comparison_summary.txt'
    summary = results['summary']
    with open(comparison_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("RETECS - Industry Dataset Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Agent: {config['agent_type']}, Reward: {config['reward_func']}\n")
        f.write(f"Total builds analyzed: {summary['n_builds']}\n\n")
        f.write(f"APFD Statistics:\n")
        f.write(f"  Mean APFD:   {summary['mean_apfd']:.4f} (PRIMARY METRIC)\n")
        f.write(f"  Std:         {summary['std_apfd']:.4f}\n")
        f.write(f"  Median:      {summary['median_apfd']:.4f}\n")
        f.write(f"  Min:         {summary['min_apfd']:.4f}\n")
        f.write(f"  Max:         {summary['max_apfd']:.4f}\n")
        if 'builds_apfd_1.0' in summary:
            f.write(f"\nAPFD Distribution:\n")
            f.write(f"  APFD = 1.0:  {summary['builds_apfd_1.0']} builds\n")
            f.write(f"  APFD >= 0.7: {summary['builds_apfd_gte_0.7']} builds\n")
            f.write(f"  APFD >= 0.5: {summary['builds_apfd_gte_0.5']} builds\n")
            f.write(f"  APFD < 0.5:  {summary['builds_apfd_lt_0.5']} builds\n")
        f.write(f"\nTiming:\n")
        f.write(f"  Training:    {results['timing']['train_time_seconds']:.2f}s\n")
        f.write(f"  Evaluation:  {results['timing']['eval_time_seconds']:.2f}s\n")
        f.write(f"  Total:       {results['timing']['total_time_seconds']:.2f}s\n")
        f.write("\n" + "=" * 70 + "\n")
    logger.info(f"Saved comparison summary to: {comparison_path}")


def print_results(results: Dict):
    summary = results['summary']
    print("\n" + "=" * 70)
    print("RETECS - Industry Dataset Results")
    print("=" * 70)
    print(f"\nTotal builds analyzed: {summary['n_builds']}")
    print(f"\nAPFD Statistics:")
    print(f"  Mean APFD:   {summary['mean_apfd']:.4f} <<< PRIMARY METRIC")
    print(f"  Std:         {summary['std_apfd']:.4f}")
    print(f"  Median:      {summary['median_apfd']:.4f}")
    print(f"  Min:         {summary['min_apfd']:.4f}")
    print(f"  Max:         {summary['max_apfd']:.4f}")
    if 'builds_apfd_1.0' in summary and summary['n_builds'] > 0:
        n = summary['n_builds']
        print(f"\nAPFD Distribution:")
        print(f"  APFD = 1.0:  {summary['builds_apfd_1.0']:3d} ({100*summary['builds_apfd_1.0']/n:.1f}%)")
        print(f"  APFD >= 0.7: {summary['builds_apfd_gte_0.7']:3d} ({100*summary['builds_apfd_gte_0.7']/n:.1f}%)")
        print(f"  APFD >= 0.5: {summary['builds_apfd_gte_0.5']:3d} ({100*summary['builds_apfd_gte_0.5']/n:.1f}%)")
        print(f"  APFD < 0.5:  {summary['builds_apfd_lt_0.5']:3d} ({100*summary['builds_apfd_lt_0.5']/n:.1f}%)")
    print(f"\nTiming:")
    print(f"  Training:    {results['timing']['train_time_seconds']:.2f}s")
    print(f"  Evaluation:  {results['timing']['eval_time_seconds']:.2f}s")
    print(f"  Total:       {results['timing']['total_time_seconds']:.2f}s")
    print("=" * 70)


def main():
    print("\n" + "=" * 70)
    print("RETECS Experiment - Industry Dataset")
    print("Scientifically Comparable with Filo-Priori")
    print("=" * 70 + "\n")

    results = run_experiment(CONFIG)
    save_results(results, CONFIG)
    print_results(results)
    print(f"\nResults saved to: {CONFIG['output_dir']}/")


if __name__ == '__main__':
    main()
