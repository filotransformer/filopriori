#!/usr/bin/env python3
"""
DeepOrder with Filo-Priori Dual-Balancing Strategy -- Industry Dataset

This experiment isolates the effect of Filo-Priori's class-imbalance handling
(balanced sampling + focal loss) on DeepOrder's architecture. The goal is to
determine whether Filo-Priori's advantage comes from its GATv2 graph structure
or merely from superior class-imbalance treatment.

Two configurations are tested:
  1. deeporder_balanced_sampling: DeepOrder + WeightedRandomSampler (29:1 ratio)
     with standard BCELoss (no pos_weight, since sampling handles imbalance).
  2. deeporder_dual_balancing: DeepOrder + WeightedRandomSampler + focal loss
     (alpha=0.75, gamma=2.0), matching Filo-Priori's exact balancing strategy.

Everything else is kept identical to the original DeepOrder experiment:
same architecture (64,32,16), same 8 features, same online history updates,
same APFD calculation, same seed (42).

Usage:
    python experiments/run_deeporder_dualbalancing_industry.py

Author: Ablation experiment -- class-imbalance handling
Date: 2026-04
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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import DeepOrder components
from src.baselines.deeporder import (
    DeepOrderNet,
    DeepOrderFeatureExtractor,
    DeepOrderDataset,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default device selection
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Data paths -- same as Filo-Priori and original DeepOrder
    'train_path': 'datasets/01_industry/train.csv',
    'test_path': 'datasets/01_industry/test.csv',

    # Column names
    'build_col': 'Build_ID',
    'test_col': 'TC_Key',
    'result_col': 'TE_Test_Result',
    'duration_col': None,

    # DeepOrder hyperparameters (identical to original)
    'hidden_dims': [64, 32, 16],
    'dropout': 0.2,
    'learning_rate': 0.001,
    'epochs': 50,
    'batch_size': 32,
    'history_window': 10,
    'device': DEFAULT_DEVICE,

    # Balanced sampling weights (matching Filo-Priori's 29:1 ratio)
    'sample_weight_positive': 1.0,
    'sample_weight_negative': 0.035,

    # Focal loss parameters (matching Filo-Priori)
    'focal_alpha': 0.75,
    'focal_gamma': 2.0,

    # Reproducibility
    'seed': 42,

    # Output
    'output_dir': 'results/deeporder_dualbalancing_industry',
}


# =============================================================================
# FOCAL LOSS
# =============================================================================

def binary_focal_loss(pred_prob, target, alpha=0.75, gamma=2.0):
    """
    Binary focal loss for sigmoid outputs.

    DeepOrderNet already applies sigmoid, so pred_prob is in [0, 1].
    Do NOT use BCEWithLogitsLoss here.

    Args:
        pred_prob: Predicted probabilities (after sigmoid), shape [B]
        target: Ground truth labels (0 or 1), shape [B]
        alpha: Weight for positive class (failures)
        gamma: Focusing parameter
    Returns:
        Scalar mean focal loss
    """
    p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    focal_weight = (1 - p_t) ** gamma
    bce = -torch.log(p_t + 1e-8)
    loss = alpha_t * focal_weight * bce
    return loss.mean()


# =============================================================================
# APFD CALCULATION -- identical to Filo-Priori / original DeepOrder experiment
# =============================================================================

def calculate_apfd_single_build(ranks: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """
    Calculate APFD for a single build.

    Formula:
        APFD = 1 - (sum of failure ranks) / (n_failures * n_tests) + 1 / (2 * n_tests)
    """
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
# DATA HELPERS
# =============================================================================

def normalize_results(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Normalize verdicts so only exact 'Fail' counts as failure."""
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
    """Get list of builds that have at least one failure."""
    build_col = config['build_col']
    result_col = config['result_col']

    builds_with_failures = []
    for build_id, group in df.groupby(build_col):
        has_fail = (group[result_col].astype(str).str.strip() == 'Fail').any()
        if has_fail:
            builds_with_failures.append(build_id)

    return builds_with_failures


# =============================================================================
# TRAINING WITH CUSTOM BALANCING
# =============================================================================

def prepare_training_data(
    df: pd.DataFrame,
    config: Dict,
) -> Tuple[np.ndarray, np.ndarray, DeepOrderFeatureExtractor]:
    """
    Prepare training data from the training DataFrame.

    Returns features, labels, and the feature extractor (with history populated).
    """
    build_col = config['build_col']
    test_col = config['test_col']
    result_col = config['result_col']
    duration_col = config['duration_col']

    feature_extractor = DeepOrderFeatureExtractor(config['history_window'])

    features_list = []
    labels_list = []

    builds = df[build_col].unique().tolist()
    grouped = df.groupby(build_col, sort=False)

    for build_id in builds:
        build_df = grouped.get_group(build_id)
        test_ids = build_df[test_col].values
        result_vals = build_df[result_col].values
        dur_vals = (
            build_df[duration_col].values
            if duration_col and duration_col in build_df.columns
            else np.ones(len(build_df))
        )

        # Extract features BEFORE updating history
        for i in range(len(test_ids)):
            features = feature_extractor.extract_features(test_ids[i])
            features_list.append(features)
            verdict = 1 if str(result_vals[i]).strip() == 'Fail' else 0
            labels_list.append(verdict)

        # Update history with this build's results
        test_results = {}
        for i in range(len(test_ids)):
            verdict = 1 if str(result_vals[i]).strip() == 'Fail' else 0
            test_results[test_ids[i]] = (verdict, float(dur_vals[i]))
        feature_extractor.update_history(build_id, test_results)

    return np.array(features_list), np.array(labels_list), feature_extractor


def train_model_with_balancing(
    X: np.ndarray,
    y: np.ndarray,
    config: Dict,
    use_focal_loss: bool,
) -> DeepOrderNet:
    """
    Train a DeepOrderNet with balanced sampling and optionally focal loss.

    CRITICAL: DeepOrderNet has sigmoid output. We use BCELoss (not BCEWithLogitsLoss)
    for the non-focal variant, and binary_focal_loss for the focal variant.
    """
    device = config.get('device', DEFAULT_DEVICE)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DeepOrderNet(
        input_dim=X.shape[1],
        hidden_dims=config['hidden_dims'],
        dropout=config['dropout'],
    ).to(device)

    dataset = DeepOrderDataset(X, y)

    # --- Balanced sampling via WeightedRandomSampler ---
    sample_weights = np.where(
        y == 1,
        config['sample_weight_positive'],
        config['sample_weight_negative'],
    )
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        sampler=sampler,
    )

    # --- Loss function ---
    if use_focal_loss:
        loss_label = "focal loss (alpha={}, gamma={})".format(
            config['focal_alpha'], config['focal_gamma']
        )
    else:
        loss_label = "BCELoss (no pos_weight; balanced sampling handles imbalance)"
        criterion = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    logger.info(f"  Loss: {loss_label}")
    logger.info(f"  Balanced sampling weights: pos={config['sample_weight_positive']}, "
                f"neg={config['sample_weight_negative']}")

    # --- Training loop ---
    model.train()
    for epoch in range(config['epochs']):
        total_loss = 0.0
        n_batches = 0

        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)

            if use_focal_loss:
                loss = binary_focal_loss(
                    outputs, batch_y,
                    alpha=config['focal_alpha'],
                    gamma=config['focal_gamma'],
                )
            else:
                loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            logger.info(f"  Epoch {epoch + 1}/{config['epochs']}, Loss = {avg_loss:.4f}")

    return model


# =============================================================================
# EVALUATION (shared across configurations)
# =============================================================================

def evaluate_model(
    model: DeepOrderNet,
    feature_extractor: DeepOrderFeatureExtractor,
    test_df: pd.DataFrame,
    test_builds_with_failures: List,
    config: Dict,
    method_name: str,
) -> Tuple[List[Dict], List[float]]:
    """
    Evaluate a trained DeepOrderNet on the test set.

    Returns per-build result dicts and a list of APFD scores.
    """
    device = config.get('device', DEFAULT_DEVICE)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    build_col = config['build_col']
    test_col = config['test_col']
    result_col = config['result_col']
    duration_col = config['duration_col']

    model.eval()

    build_results = []
    all_apfd_scores = []

    for build_id in test_builds_with_failures:
        build_df = test_df[test_df[build_col] == build_id]
        test_ids = build_df[test_col].unique().tolist()
        count_tc = len(test_ids)

        # Ground truth verdicts (only 'Fail' = 1)
        verdicts = {}
        for _, row in build_df.iterrows():
            tc = row[test_col]
            verdict = 1 if str(row[result_col]).strip() == 'Fail' else 0
            verdicts[tc] = max(verdicts.get(tc, 0), verdict)

        n_failures = sum(verdicts.values())
        if n_failures == 0:
            continue

        # Extract features and predict
        features_list = [
            feature_extractor.extract_features(tc) for tc in test_ids
        ]
        X = np.array(features_list)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()

        # Rank by predicted failure probability (descending)
        test_scores = list(zip(test_ids, predictions))
        test_scores.sort(key=lambda x: x[1], reverse=True)
        ranking = [t[0] for t in test_scores]

        # APFD
        labels = np.array([verdicts[tc] for tc in ranking])
        ranks = np.arange(1, len(ranking) + 1)
        apfd = calculate_apfd_single_build(ranks, labels)

        if apfd is not None:
            all_apfd_scores.append(apfd)
            build_results.append({
                'method_name': method_name,
                'build_id': build_id,
                'test_scenario': 'industry_dataset',
                'count_tc': count_tc,
                'count_commits': 0,
                'apfd': apfd,
                'time': 0.0,
            })

        # Update history for online learning
        test_results = {}
        for tc in test_ids:
            duration = 1.0
            if duration_col and duration_col in build_df.columns:
                duration = float(
                    build_df[build_df[test_col] == tc][duration_col].values[0]
                )
            test_results[tc] = (verdicts.get(tc, 0), duration)
        feature_extractor.update_history(build_id, test_results)

    return build_results, all_apfd_scores


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

def compute_summary(apfd_scores: List[float]) -> Dict:
    """Compute summary statistics from a list of APFD scores."""
    if not apfd_scores:
        return {
            'mean_apfd': 0.0, 'std_apfd': 0.0, 'median_apfd': 0.0,
            'min_apfd': 0.0, 'max_apfd': 0.0, 'n_builds': 0,
        }

    return {
        'mean_apfd': float(np.mean(apfd_scores)),
        'std_apfd': float(np.std(apfd_scores)),
        'median_apfd': float(np.median(apfd_scores)),
        'min_apfd': float(np.min(apfd_scores)),
        'max_apfd': float(np.max(apfd_scores)),
        'n_builds': len(apfd_scores),
        'builds_apfd_1.0': sum(1 for x in apfd_scores if x == 1.0),
        'builds_apfd_gte_0.7': sum(1 for x in apfd_scores if x >= 0.7),
        'builds_apfd_gte_0.5': sum(1 for x in apfd_scores if x >= 0.5),
        'builds_apfd_lt_0.5': sum(1 for x in apfd_scores if x < 0.5),
    }


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_single_config(
    config: Dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_builds_with_failures: List,
    method_name: str,
    use_focal_loss: bool,
) -> Dict:
    """
    Run one configuration (balanced sampling only, or dual balancing).
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Running: {method_name}")
    logger.info(f"  Focal loss: {use_focal_loss}")
    logger.info(f"{'='*70}")

    start_time = time.time()

    # Prepare training data (builds history from scratch each time)
    logger.info("Preparing training data...")
    X, y, feature_extractor = prepare_training_data(train_df, config)
    logger.info(f"  Training samples: {len(X):,}")
    logger.info(f"  Failure rate: {y.mean():.4f}")
    logger.info(f"  Failures: {int(y.sum()):,} / {len(y):,}")

    # Train
    logger.info("Training model...")
    train_start = time.time()
    model = train_model_with_balancing(X, y, config, use_focal_loss=use_focal_loss)
    train_time = time.time() - train_start
    logger.info(f"Training completed in {train_time:.2f}s")

    # Evaluate
    logger.info("Evaluating on test set...")
    eval_start = time.time()
    build_results, apfd_scores = evaluate_model(
        model, feature_extractor, test_df,
        test_builds_with_failures, config, method_name,
    )
    eval_time = time.time() - eval_start
    total_time = time.time() - start_time

    summary = compute_summary(apfd_scores)

    return {
        'method': method_name,
        'use_focal_loss': use_focal_loss,
        'summary': summary,
        'build_results': build_results,
        'timing': {
            'train_time_seconds': train_time,
            'eval_time_seconds': eval_time,
            'total_time_seconds': total_time,
        },
        'timestamp': datetime.now().isoformat(),
    }


def run_experiment(config: Dict) -> Dict:
    """
    Run both configurations and return combined results.
    """
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

    # Get test builds with failures
    test_builds_with_failures = get_builds_with_failures(test_df, config)
    logger.info(f"Test builds with failures: {len(test_builds_with_failures)}")

    results = {}

    # --- Configuration 1: Balanced sampling only (BCELoss, no pos_weight) ---
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    results['balanced_sampling'] = run_single_config(
        config=config,
        train_df=train_df,
        test_df=test_df,
        test_builds_with_failures=test_builds_with_failures,
        method_name='DeepOrder + Balanced Sampling',
        use_focal_loss=False,
    )

    # --- Configuration 2: Balanced sampling + focal loss (dual balancing) ---
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    results['dual_balancing'] = run_single_config(
        config=config,
        train_df=train_df,
        test_df=test_df,
        test_builds_with_failures=test_builds_with_failures,
        method_name='DeepOrder + Dual Balancing',
        use_focal_loss=True,
    )

    return results


# =============================================================================
# SAVE AND PRINT RESULTS
# =============================================================================

def save_results(results: Dict, config: Dict):
    """Save results for both configurations."""
    output_dir = PROJECT_ROOT / config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    for key, result in results.items():
        # Per-build APFD CSV
        build_results_df = pd.DataFrame(result['build_results'])
        csv_path = output_dir / f'apfd_per_build_FULL_testcsv_{key}.csv'
        build_results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved: {csv_path}")

    # Also save the dual_balancing CSV with the standard name for easy comparison
    dual_df = pd.DataFrame(results['dual_balancing']['build_results'])
    std_csv = output_dir / 'apfd_per_build_FULL_testcsv.csv'
    dual_df.to_csv(std_csv, index=False)
    logger.info(f"Saved (standard name): {std_csv}")

    # Summary JSON with all results
    summary_path = output_dir / 'experiment_summary.json'
    save_obj = {
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
        'results': {},
    }
    for key, result in results.items():
        save_obj['results'][key] = {
            'method': result['method'],
            'use_focal_loss': result['use_focal_loss'],
            'summary': result['summary'],
            'timing': result['timing'],
            'timestamp': result['timestamp'],
        }
    with open(summary_path, 'w') as f:
        json.dump(save_obj, f, indent=2, default=str)
    logger.info(f"Saved: {summary_path}")


def print_comparison_table(results: Dict):
    """Print a clear comparison table at the end."""
    # Reference values
    filo_priori_apfd = 0.7611
    deeporder_original_apfd = 0.6890

    bs_summary = results['balanced_sampling']['summary']
    db_summary = results['dual_balancing']['summary']

    print("\n" + "=" * 78)
    print("  COMPARISON TABLE: DeepOrder with Filo-Priori Balancing -- Industry Dataset")
    print("=" * 78)

    header = f"{'Method':<40} {'APFD':>8} {'Std':>8} {'Builds':>8} {'vs FP':>8}"
    print(header)
    print("-" * 78)

    rows = [
        ("Filo-Priori (reference)",
         filo_priori_apfd, None, 277, None),
        ("DeepOrder original (reference)",
         deeporder_original_apfd, None, 277, None),
        ("DeepOrder + Balanced Sampling",
         bs_summary['mean_apfd'], bs_summary['std_apfd'],
         bs_summary['n_builds'], None),
        ("DeepOrder + Dual Balancing",
         db_summary['mean_apfd'], db_summary['std_apfd'],
         db_summary['n_builds'], None),
    ]

    for name, apfd, std, n, _ in rows:
        std_str = f"{std:.4f}" if std is not None else "   --"
        delta = ((filo_priori_apfd - apfd) / filo_priori_apfd) * 100
        delta_str = f"{delta:+.1f}%" if apfd != filo_priori_apfd else "  --"
        print(f"  {name:<38} {apfd:>8.4f} {std_str:>8} {n:>8} {delta_str:>8}")

    print("-" * 78)

    # Interpretation
    bs_apfd = bs_summary['mean_apfd']
    db_apfd = db_summary['mean_apfd']
    best_do = max(bs_apfd, db_apfd)
    gap = filo_priori_apfd - best_do

    print(f"\n  Best DeepOrder with balancing: {best_do:.4f}")
    print(f"  Filo-Priori:                  {filo_priori_apfd:.4f}")
    print(f"  Remaining gap (architectural):{gap:+.4f} ({gap/filo_priori_apfd*100:.1f}%)")

    if gap > 0.01:
        print("\n  CONCLUSION: Filo-Priori's GATv2 architecture provides a real advantage")
        print("  beyond class-imbalance handling. The graph structure matters.")
    else:
        print("\n  CONCLUSION: The advantage is largely explained by class-imbalance handling.")
        print("  The GATv2 graph structure provides minimal additional benefit.")

    print("=" * 78)

    # Per-config detail
    for key, label in [
        ('balanced_sampling', 'DeepOrder + Balanced Sampling'),
        ('dual_balancing', 'DeepOrder + Dual Balancing'),
    ]:
        s = results[key]['summary']
        t = results[key]['timing']
        print(f"\n  --- {label} ---")
        print(f"  Mean APFD:   {s['mean_apfd']:.4f}")
        print(f"  Std APFD:    {s['std_apfd']:.4f}")
        print(f"  Median APFD: {s['median_apfd']:.4f}")
        print(f"  Min / Max:   {s['min_apfd']:.4f} / {s['max_apfd']:.4f}")
        if s['n_builds'] > 0:
            n = s['n_builds']
            print(f"  APFD = 1.0:  {s['builds_apfd_1.0']:3d} ({100*s['builds_apfd_1.0']/n:.1f}%)")
            print(f"  APFD >= 0.7: {s['builds_apfd_gte_0.7']:3d} ({100*s['builds_apfd_gte_0.7']/n:.1f}%)")
            print(f"  APFD >= 0.5: {s['builds_apfd_gte_0.5']:3d} ({100*s['builds_apfd_gte_0.5']/n:.1f}%)")
            print(f"  APFD < 0.5:  {s['builds_apfd_lt_0.5']:3d} ({100*s['builds_apfd_lt_0.5']/n:.1f}%)")
        print(f"  Train time:  {t['train_time_seconds']:.2f}s")
        print(f"  Eval time:   {t['eval_time_seconds']:.2f}s")
        print(f"  Total time:  {t['total_time_seconds']:.2f}s")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 78)
    print("  DeepOrder with Filo-Priori Dual-Balancing -- Industry Dataset")
    print("  Isolating class-imbalance handling from GATv2 architecture")
    print("=" * 78 + "\n")

    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print()

    # Run both configurations
    results = run_experiment(CONFIG)

    # Save results
    save_results(results, CONFIG)

    # Print comparison table
    print_comparison_table(results)

    print(f"\nResults saved to: {CONFIG['output_dir']}/")


if __name__ == '__main__':
    main()
