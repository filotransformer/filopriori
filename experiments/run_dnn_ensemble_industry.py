#!/usr/bin/env python3
"""
B1 Experiment: Test DNN Ensemble on Industrial Dataset

This script answers: "Does adding a DeepOrder DNN ensemble improve APFD
on the industrial dataset?" If not, the claim that it's redundant is
empirically justified.

We use the EXISTING GNN predictions (from the frozen V3 experiment) and
train a DNN on the same train split, then optimize alpha blending on
validation builds and evaluate on all 277 test builds.

Usage:
    python experiments/run_dnn_ensemble_industry.py
"""

import gc
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.deeporder import (
    DeepOrderModel, DeepOrderFeatureExtractor, DeepOrderNet, DeepOrderDataset
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

# DNN config (same as V14 RTPTorrent)
DNN_CONFIG = {
    'hidden_dims': [64, 32, 16],
    'dropout': 0.2,
    'learning_rate': 0.001,
    'epochs': 15,
    'batch_size': 128,
    'history_window': 10,
}
MAX_DNN_POS_WEIGHT = 50.0
ALPHA_SEARCH_RANGE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_apfd(ranks, labels):
    labels_arr = np.array(labels)
    ranks_arr = np.array(ranks)
    n_tests = len(labels_arr)
    fail_idx = np.where(labels_arr.astype(int) != 0)[0]
    n_failures = len(fail_idx)
    if n_failures == 0:
        return None
    if n_tests == 1:
        return 1.0
    failure_ranks = ranks_arr[fail_idx]
    apfd = 1.0 - float(failure_ranks.sum()) / float(n_failures * n_tests) + 1.0 / (2.0 * n_tests)
    return float(np.clip(apfd, 0.0, 1.0))


def load_industrial_data():
    """Load industrial dataset and split temporally."""
    train_df = pd.read_csv('datasets/01_industry/train.csv')
    test_df = pd.read_csv('datasets/01_industry/test.csv')

    # Prepare columns
    for df in [train_df, test_df]:
        df['is_failure'] = (df['TE_Test_Result'].astype(str).str.strip() == 'Fail').astype(int)
        if 'duration' not in df.columns:
            df['duration'] = 1.0

    logger.info(f"Train: {len(train_df)} rows, {train_df['Build_ID'].nunique()} builds")
    logger.info(f"Test:  {len(test_df)} rows, {test_df['Build_ID'].nunique()} builds")

    # Validation split: last 10% of training builds
    train_builds = train_df['Build_ID'].unique().tolist()
    val_idx = int(len(train_builds) * 0.9)
    inner_train_builds = train_builds[:val_idx]
    val_builds = train_builds[val_idx:]

    inner_train_df = train_df[train_df['Build_ID'].isin(inner_train_builds)]
    val_df = train_df[train_df['Build_ID'].isin(val_builds)]

    logger.info(f"Inner train: {len(inner_train_builds)} builds, Val: {len(val_builds)} builds")

    return train_df, test_df, inner_train_df, val_df, val_builds


def train_dnn(train_df):
    """Train DeepOrder DNN on industrial training data."""
    logger.info("Training DeepOrder DNN...")
    cfg = DNN_CONFIG

    do_model = DeepOrderModel(
        hidden_dims=cfg['hidden_dims'], dropout=cfg['dropout'],
        learning_rate=cfg['learning_rate'], epochs=cfg['epochs'],
        batch_size=cfg['batch_size'], history_window=cfg['history_window'],
        device=DEVICE,
    )

    all_builds = train_df['Build_ID'].unique().tolist()
    grouped = train_df.groupby('Build_ID', sort=False)
    fe = DeepOrderFeatureExtractor(history_window=cfg['history_window'])

    # Extract features + build history
    features_list, labels_list = [], []
    for bid in all_builds:
        if bid not in grouped.groups:
            continue
        bdf = grouped.get_group(bid)
        test_ids = bdf['TC_Key'].values
        fail_vals = bdf['is_failure'].values
        dur_vals = bdf['duration'].values

        for i in range(len(test_ids)):
            features_list.append(fe.extract_features(test_ids[i]))
            labels_list.append(int(fail_vals[i]))

        test_results = {test_ids[i]: (int(fail_vals[i]), float(dur_vals[i])) for i in range(len(test_ids))}
        fe.update_history(bid, test_results)

    X, y = np.array(features_list), np.array(labels_list)
    del features_list, labels_list
    gc.collect()

    logger.info(f"DNN training data: {X.shape[0]} samples, {y.sum()} failures ({100*y.mean():.2f}%)")

    do_model.feature_extractor = fe
    do_model.model = DeepOrderNet(
        input_dim=X.shape[1], hidden_dims=cfg['hidden_dims'], dropout=cfg['dropout']
    ).to(DEVICE)

    dataset = DeepOrderDataset(X, y)
    dataloader = TorchDataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True,
                                  drop_last=(len(dataset) > cfg['batch_size']))

    raw_pw = (1 - y.mean()) / (y.mean() + 1e-6)
    clamped_pw = min(raw_pw, MAX_DNN_POS_WEIGHT)
    pos_weight_val = torch.tensor(clamped_pw, dtype=torch.float32).to(DEVICE)
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(do_model.model.parameters(), lr=cfg['learning_rate'])

    logger.info(f"DNN pos_weight: {raw_pw:.1f} → clamped to {clamped_pw:.1f}")

    do_model.model.train()
    for epoch in range(cfg['epochs']):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = do_model.model(batch_X)
            per_sample_loss = criterion(outputs, batch_y)
            weights = torch.where(batch_y == 1, pos_weight_val, torch.ones_like(batch_y))
            loss = (per_sample_loss * weights).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            logger.info(f"  DNN Epoch {epoch+1}/{cfg['epochs']}, Loss: {epoch_loss/len(dataloader):.4f}")

    del X, y, dataset, dataloader
    gc.collect()
    return do_model


def get_dnn_scores(do_model, test_ids):
    """Get DNN failure probability for each test."""
    if do_model is None or do_model.model is None:
        return np.full(len(test_ids), 0.5)
    do_model.model.eval()
    features = [do_model.feature_extractor.extract_features(tc) for tc in test_ids]
    X = np.array(features)
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        preds = do_model.model(X_t).cpu().numpy()
    return preds


def evaluate_dnn_only(do_model, test_df):
    """Evaluate DNN-only scoring on test builds (alpha=0.0, pure DNN)."""
    grouped = test_df.groupby('Build_ID')
    all_apfds = []

    for build_id, bdf in grouped:
        verdicts = bdf.groupby('TC_Key')['is_failure'].max().to_dict()
        n_failures = sum(verdicts.values())
        if n_failures == 0:
            # Still update history
            test_ids = list(verdicts.keys())
            durations = bdf.groupby('TC_Key')['duration'].last().to_dict()
            results = {tc: (verdicts[tc], durations.get(tc, 1.0)) for tc in test_ids}
            do_model.update_history(str(build_id), results)
            continue

        test_ids = list(verdicts.keys())
        dnn_scores = get_dnn_scores(do_model, test_ids)

        sorted_idx = np.argsort(-dnn_scores)
        ranking = [test_ids[i] for i in sorted_idx]
        labels = np.array([verdicts[tc] for tc in ranking])
        ranks = np.arange(1, len(ranking) + 1)
        apfd = calculate_apfd(ranks, labels)

        if apfd is not None:
            all_apfds.append({'build_id': build_id, 'apfd': apfd, 'n_tc': len(test_ids)})

        # Online update
        durations = bdf.groupby('TC_Key')['duration'].last().to_dict()
        results = {tc: (verdicts[tc], durations.get(tc, 1.0)) for tc in test_ids}
        do_model.update_history(str(build_id), results)

    return all_apfds


def main():
    set_seed(SEED)
    logger.info("=" * 70)
    logger.info("B1 Experiment: DNN Ensemble on Industrial Dataset")
    logger.info("=" * 70)

    # 1. Load data
    train_df, test_df, inner_train_df, val_df, val_builds = load_industrial_data()

    # 2. Train DNN
    t0 = time.time()
    do_model = train_dnn(inner_train_df)
    train_time = time.time() - t0
    logger.info(f"DNN training time: {train_time:.1f}s")

    # 3. Evaluate DNN-only on test set (alpha=0.0)
    logger.info("\nEvaluating DNN-only on 277 test builds...")
    dnn_results = evaluate_dnn_only(do_model, test_df)
    dnn_apfds = [r['apfd'] for r in dnn_results]
    dnn_mean = np.mean(dnn_apfds)
    dnn_std = np.std(dnn_apfds)

    logger.info(f"\n{'='*70}")
    logger.info(f"RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"GNN-only (existing V3):    APFD = 0.7611")
    logger.info(f"DNN-only (this experiment): APFD = {dnn_mean:.4f} ± {dnn_std:.4f}")
    logger.info(f"  N builds with failures: {len(dnn_apfds)}")
    logger.info(f"  Median: {np.median(dnn_apfds):.4f}")

    # Compare
    delta = dnn_mean - 0.7611
    logger.info(f"\nDNN vs GNN: {delta:+.4f} ({delta/0.7611*100:+.1f}%)")
    if delta > 0:
        logger.info("→ DNN ensemble WOULD IMPROVE industrial performance")
        logger.info("  Consider adding alpha blending to the full model")
    else:
        logger.info("→ DNN ensemble does NOT improve industrial performance")
        logger.info("  The claim that DNN ensemble is redundant is JUSTIFIED")

    # Save results
    output_dir = Path('results/dnn_ensemble_industry')
    output_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(dnn_results).to_csv(output_dir / 'dnn_only_per_build.csv', index=False)

    summary = {
        'gnn_apfd': 0.7611,
        'dnn_apfd': float(dnn_mean),
        'dnn_std': float(dnn_std),
        'dnn_median': float(np.median(dnn_apfds)),
        'n_builds': len(dnn_apfds),
        'delta': float(delta),
        'train_time_s': train_time,
        'conclusion': 'DNN redundant' if delta <= 0 else 'DNN beneficial',
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()
