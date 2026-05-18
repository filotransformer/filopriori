#!/usr/bin/env python3
"""
DeepOrder Experiment for Industry Dataset

Este script executa o DeepOrder no dataset 01_industry de forma que os
resultados sejam CIENTIFICAMENTE COMPARAVEIS com o Filo-Priori.

Garantias de comparabilidade:
1. Usa EXATAMENTE o mesmo split train/test (arquivos train.csv e test.csv)
2. Calcula APFD PER BUILD usando a mesma formula do Filo-Priori
3. Considera apenas builds com pelo menos 1 falha (mesmo criterio)
4. Gera resultados no mesmo formato CSV
5. Usa a mesma seed (42) para reproducibilidade

Uso:
    python experiments/run_deeporder_industry.py

Autor: Experimento comparativo DeepOrder vs Filo-Priori
Data: 2025-12
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
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import DeepOrder
from src.baselines.deeporder import DeepOrderModel
from src.evaluation.apfd import calculate_apfd_single_build

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default device selection
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# CONFIGURATION - Must match Filo-Priori experiment settings
# =============================================================================

CONFIG = {
    # Data paths - same as Filo-Priori
    'train_path': 'datasets/01_industry/train.csv',
    'test_path': 'datasets/01_industry/test.csv',

    # Column names - same as Filo-Priori
    'build_col': 'Build_ID',
    'test_col': 'TC_Key',
    'result_col': 'TE_Test_Result',
    'duration_col': None,  # Not available in this dataset

    # DeepOrder hyperparameters
    'hidden_dims': [64, 32, 16],
    'dropout': 0.2,
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 32,
    'history_window': 10,
    'device': DEFAULT_DEVICE,

    # Reproducibility
    'seed': 42,

    # Output
    'output_dir': 'results/deeporder_industry',
    'method_name': 'DeepOrder',
    'test_scenario': 'industry_dataset'
}


# =============================================================================
# DATA HELPERS
# =============================================================================

def normalize_results(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Normalize verdicts so only exact 'Fail' counts as failure.

    Any status that is NOT exactly 'Fail' becomes 'Pass', matching the
    Filo-Priori rule that only 'Fail' is considerado falha.
    """
    df = df.copy()
    col = config['result_col']
    df[col] = df[col].astype(str).str.strip().apply(lambda x: 'Fail' if x == 'Fail' else 'Pass')
    return df


def load_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets with normalized verdicts."""
    logger.info("Loading datasets...")

    train_path = PROJECT_ROOT / config['train_path']
    test_path = PROJECT_ROOT / config['test_path']

    logger.info(f"  Train: {train_path}")
    logger.info(f"  Test: {test_path}")

    raw_train_df = pd.read_csv(train_path)
    raw_test_df = pd.read_csv(test_path)

    train_df = normalize_results(raw_train_df, config)
    test_df = normalize_results(raw_test_df, config)

    logger.info(f"  Train size: {len(train_df):,} rows")
    logger.info(f"  Test size: {len(test_df):,} rows")
    logger.info(f"  Train builds: {train_df[config['build_col']].nunique()}")
    logger.info(f"  Test builds: {test_df[config['build_col']].nunique()}")

    return train_df, test_df


def get_builds_with_failures(df: pd.DataFrame, config: Dict) -> List[str]:
    """
    Get list of builds that have at least one failure.

    IMPORTANT: Uses same criteria as Filo-Priori:
    - Only 'Fail' counts as failure
    - 'Delete', 'Blocked', 'Conditional Pass', 'Pending' are NOT failures
    """
    build_col = config['build_col']
    result_col = config['result_col']

    builds_with_failures = []
    for build_id, group in df.groupby(build_col):
        has_fail = (group[result_col].astype(str).str.strip() == 'Fail').any()
        if has_fail:
            builds_with_failures.append(build_id)

    return builds_with_failures


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(config: Dict) -> Dict:
    """
    Run DeepOrder experiment with scientific comparability.

    Returns:
        Dictionary with results including per-build APFD
    """
    start_time = time.time()

    # Set random seeds
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load data
    train_df, test_df = load_data(config)

    # Get column names
    build_col = config['build_col']
    test_col = config['test_col']
    result_col = config['result_col']
    duration_col = config['duration_col']

    # Resolve device
    device = config.get('device', DEFAULT_DEVICE)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize DeepOrder model
    logger.info("Initializing DeepOrder model...")
    model = DeepOrderModel(
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
        learning_rate=config['learning_rate'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        history_window=config['history_window'],
        device=device
    )

    # Train on training data
    logger.info(f"Training DeepOrder model on device: {device}")
    train_start = time.time()
    model.train(
        df=train_df,
        build_col=build_col,
        test_col=test_col,
        result_col=result_col,
        duration_col=duration_col
    )
    train_time = time.time() - train_start
    logger.info(f"Training completed in {train_time:.2f}s")

    # Get test builds with failures (same as Filo-Priori)
    test_builds_with_failures = get_builds_with_failures(test_df, config)
    logger.info(f"Test builds with failures: {len(test_builds_with_failures)}")

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    eval_start = time.time()

    build_results = []
    all_apfd_scores = []

    for build_id in test_builds_with_failures:
        build_df = test_df[test_df[build_col] == build_id]
        test_ids = build_df[test_col].unique().tolist()
        count_tc = len(test_ids)

        # Get ground truth verdicts (same as Filo-Priori: only 'Fail' = 1)
        verdicts = {}
        for _, row in build_df.iterrows():
            tc = row[test_col]
            verdict = 1 if str(row[result_col]).strip() == 'Fail' else 0
            verdicts[tc] = max(verdicts.get(tc, 0), verdict)

        # Get number of failures
        n_failures = sum(verdicts.values())

        # Skip if no failures (defensive)
        if n_failures == 0:
            continue

        # Get prioritization ranking from DeepOrder
        ranking = model.prioritize(test_ids)

        # Create labels array in ranking order
        labels = np.array([verdicts[tc] for tc in ranking])

        # Create ranks array (1-indexed)
        ranks = np.arange(1, len(ranking) + 1)

        # Calculate APFD
        apfd = calculate_apfd_single_build(ranks, labels)

        if apfd is not None:
            all_apfd_scores.append(apfd)

            # Count commits (placeholder, match Filo-Priori format)
            count_commits = 0
            if 'commit' in build_df.columns:
                import ast
                for commit_str in build_df['commit'].dropna():
                    try:
                        commits = ast.literal_eval(str(commit_str))
                        if isinstance(commits, list):
                            count_commits = max(count_commits, len(commits))
                    except Exception:
                        pass

            build_results.append({
                'method_name': config['method_name'],
                'build_id': build_id,
                'test_scenario': config['test_scenario'],
                'count_tc': count_tc,
                'count_commits': count_commits,
                'apfd': apfd,
                'time': 0.0
            })

        # Update model history for online learning
        test_results = {}
        for tc in test_ids:
            duration = 1.0
            if duration_col and duration_col in build_df.columns:
                duration = float(build_df[build_df[test_col] == tc][duration_col].values[0])
            test_results[tc] = (verdicts.get(tc, 0), duration)
        model.update_history(build_id, test_results)

    eval_time = time.time() - eval_start
    total_time = time.time() - start_time

    # Calculate summary statistics
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
        summary = {
            'mean_apfd': 0.0,
            'std_apfd': 0.0,
            'median_apfd': 0.0,
            'min_apfd': 0.0,
            'max_apfd': 0.0,
            'n_builds': 0
        }

    results = {
        'method': 'DeepOrder',
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

    return results


def save_results(results: Dict, config: Dict):
    """Save results in format compatible with Filo-Priori."""
    output_dir = PROJECT_ROOT / config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-build APFD CSV (same format as Filo-Priori)
    build_results_df = pd.DataFrame(results['build_results'])
    apfd_csv_path = output_dir / 'apfd_per_build_FULL_testcsv.csv'
    build_results_df.to_csv(apfd_csv_path, index=False)
    logger.info(f"Saved APFD per build to: {apfd_csv_path}")

    # Save summary JSON
    summary_path = output_dir / 'experiment_summary.json'
    with open(summary_path, 'w') as f:
        save_results_obj = results.copy()
        save_results_obj['config'] = {k: str(v) if isinstance(v, Path) else v
                                      for k, v in config.items()}
        json.dump(save_results_obj, f, indent=2, default=str)
    logger.info(f"Saved experiment summary to: {summary_path}")

    # Save comparison-ready summary
    comparison_path = output_dir / 'comparison_summary.txt'
    with open(comparison_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DeepOrder - Industry Dataset Results\n")
        f.write("="*70 + "\n\n")

        summary = results['summary']
        f.write(f"Total builds analyzed: {summary['n_builds']}\n")
        f.write(f"\nAPFD Statistics:\n")
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

        f.write("\n" + "="*70 + "\n")
        f.write("Configuration:\n")
        f.write("="*70 + "\n")
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")

    logger.info(f"Saved comparison summary to: {comparison_path}")


def print_results(results: Dict):
    """Print results to console."""
    summary = results['summary']

    print("\n" + "="*70)
    print("DeepOrder - Industry Dataset Results")
    print("="*70)

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

    print("="*70)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("DeepOrder Experiment - Industry Dataset")
    print("Scientifically Comparable with Filo-Priori")
    print("="*70 + "\n")

    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print()

    # Run experiment
    results = run_experiment(CONFIG)

    # Save results
    save_results(results, CONFIG)

    # Print results
    print_results(results)

    print(f"\nResults saved to: {CONFIG['output_dir']}/")
    print("\nTo compare with Filo-Priori, compare:")
    print(f"  - DeepOrder: {CONFIG['output_dir']}/apfd_per_build_FULL_testcsv.csv")
    print(f"  - Filo-Priori: results/experiment_industry/apfd_per_build_FULL_testcsv.csv")


if __name__ == '__main__':
    main()
