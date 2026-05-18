#!/usr/bin/env python3
"""
Experiment: GNN+DNN Ensemble with Alpha Blending on Industrial Dataset

Tests whether combining the GNN (frozen V3, APFD=0.761) with a DeepOrder DNN
through validation-optimized alpha blending improves APFD on the industrial
dataset, mirroring the ensemble approach used on RTPTorrent.

  score = alpha * P_GNN(fail) + (1 - alpha) * S_DNN

Alpha is optimized on a held-out validation set (last 10% of training builds).

Usage:
    python experiments/run_gnn_dnn_ensemble_industry.py
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
    DeepOrderFeatureExtractor, DeepOrderNet, DeepOrderDataset
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

DNN_CONFIG = {
    'hidden_dims': [64, 32, 16],
    'dropout': 0.2,
    'learning_rate': 0.001,
    'epochs': 15,
    'batch_size': 128,
    'history_window': 10,
}
MAX_DNN_POS_WEIGHT = 50.0
ALPHA_RANGE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
OUTPUT_DIR = PROJECT_ROOT / 'results/gnn_dnn_ensemble_industry'


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_apfd(ranks, labels):
    labels_arr = np.array(labels)
    ranks_arr = np.array(ranks)
    n = len(labels_arr)
    fail_idx = np.where(labels_arr.astype(int) != 0)[0]
    m = len(fail_idx)
    if m == 0:
        return None
    if n == 1:
        return 1.0
    return float(np.clip(
        1.0 - ranks_arr[fail_idx].sum() / (m * n) + 1.0 / (2.0 * n), 0, 1
    ))


def load_data():
    """Load industrial data + GNN predictions."""
    train_df = pd.read_csv(PROJECT_ROOT / 'datasets/01_industry/train.csv')
    test_df = pd.read_csv(PROJECT_ROOT / 'datasets/01_industry/test.csv')

    for df in [train_df, test_df]:
        df['is_failure'] = (df['TE_Test_Result'].astype(str).str.strip() == 'Fail').astype(int)
        if 'duration' not in df.columns:
            df['duration'] = 1.0

    # GNN predictions from frozen V3
    gnn_preds = pd.read_csv(
        PROJECT_ROOT / 'results/experiment_industry_optimized_v3/prioritized_test_cases_FULL_testcsv.csv'
    )

    # Validation split: last 10% of training builds
    train_builds = train_df['Build_ID'].unique().tolist()
    val_idx = int(len(train_builds) * 0.9)
    inner_train_builds = set(train_builds[:val_idx])
    val_builds_list = train_builds[val_idx:]
    val_builds = set(val_builds_list)

    inner_train_df = train_df[train_df['Build_ID'].isin(inner_train_builds)]

    logger.info(f"Train: {len(inner_train_builds)} builds, Val: {len(val_builds)} builds")
    logger.info(f"Test: {test_df['Build_ID'].nunique()} builds")
    logger.info(f"GNN predictions: {gnn_preds['Build_ID'].nunique()} builds")

    return inner_train_df, train_df, test_df, gnn_preds, val_builds_list


def train_dnn(train_df):
    """Train DeepOrder DNN with history pre-warming."""
    cfg = DNN_CONFIG
    fe = DeepOrderFeatureExtractor(history_window=cfg['history_window'])

    all_builds = train_df['Build_ID'].unique().tolist()
    grouped = train_df.groupby('Build_ID', sort=False)

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

        test_results = {test_ids[i]: (int(fail_vals[i]), float(dur_vals[i]))
                        for i in range(len(test_ids))}
        fe.update_history(bid, test_results)

    X, y = np.array(features_list), np.array(labels_list)
    logger.info(f"DNN data: {X.shape[0]} samples, {y.sum()} failures ({100*y.mean():.2f}%)")

    model = DeepOrderNet(
        input_dim=X.shape[1], hidden_dims=cfg['hidden_dims'], dropout=cfg['dropout']
    ).to(DEVICE)

    dataset = DeepOrderDataset(X, y)
    dataloader = TorchDataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True,
                                 drop_last=(len(dataset) > cfg['batch_size']))

    raw_pw = (1 - y.mean()) / (y.mean() + 1e-6)
    clamped_pw = min(raw_pw, MAX_DNN_POS_WEIGHT)
    criterion = nn.BCELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])

    logger.info(f"pos_weight: {raw_pw:.1f} -> clamped {clamped_pw:.1f}")

    model.train()
    for epoch in range(cfg['epochs']):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_X)
            per_sample_loss = criterion(outputs, batch_y)
            weights = torch.where(
                batch_y == 1,
                torch.tensor(clamped_pw, device=DEVICE),
                torch.ones_like(batch_y)
            )
            loss = (per_sample_loss * weights).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            logger.info(f"  Epoch {epoch+1}/{cfg['epochs']}, Loss: {epoch_loss/len(dataloader):.4f}")

    del X, y, dataset, dataloader
    gc.collect()
    return model, fe


def get_dnn_scores(model, fe, test_ids):
    """Get DNN P(fail) for test cases."""
    model.eval()
    features = [fe.extract_features(tc) for tc in test_ids]
    X = np.array(features)
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        return model(X_t).cpu().numpy()


def evaluate_alpha(alpha, gnn_preds, dnn_model, fe, eval_df, eval_builds):
    """Evaluate a specific alpha on a set of builds."""
    grouped = eval_df.groupby('Build_ID')
    gnn_grouped = gnn_preds.groupby('Build_ID')
    apfds = []

    for build_id in eval_builds:
        if build_id not in grouped.groups:
            continue
        bdf = grouped.get_group(build_id)
        verdicts = bdf.groupby('TC_Key')['is_failure'].max().to_dict()
        n_fail = sum(verdicts.values())
        if n_fail == 0:
            continue

        test_ids = list(verdicts.keys())

        # GNN scores
        if build_id in gnn_grouped.groups:
            gnn_df = gnn_grouped.get_group(build_id)
            gnn_scores_map = gnn_df.groupby('TC_Key')['hybrid_score'].first().to_dict()
            gnn_scores = np.array([gnn_scores_map.get(tc, 0.5) for tc in test_ids])
        else:
            gnn_scores = np.full(len(test_ids), 0.5)

        # DNN scores
        dnn_scores = get_dnn_scores(dnn_model, fe, test_ids)

        # Blend
        blended = alpha * gnn_scores + (1 - alpha) * dnn_scores

        # Rank
        order = np.argsort(-blended)
        ranking = [test_ids[i] for i in order]
        labels = np.array([verdicts[tc] for tc in ranking])
        ranks = np.arange(1, len(ranking) + 1)
        apfd = calculate_apfd(ranks, labels)
        if apfd is not None:
            apfds.append(apfd)

    return float(np.mean(apfds)) if apfds else 0.0, apfds


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("GNN+DNN Ensemble with Alpha Blending (Industrial Dataset)")
    logger.info("=" * 70)

    # 1. Load data
    inner_train_df, full_train_df, test_df, gnn_preds, val_builds = load_data()

    # 2. Train DNN
    t0 = time.time()
    dnn_model, fe = train_dnn(inner_train_df)
    train_time = time.time() - t0
    logger.info(f"DNN training: {train_time:.1f}s")

    # 3. Optimize alpha on validation set
    logger.info("\nOptimizing alpha on validation builds...")
    val_with_fail = full_train_df[
        full_train_df['Build_ID'].isin(val_builds) &
        (full_train_df['is_failure'] == 1)
    ]['Build_ID'].unique().tolist()
    logger.info(f"Validation builds with failures: {len(val_with_fail)}")

    best_alpha, best_val_apfd = 1.0, 0.0
    alpha_results = []
    for alpha in ALPHA_RANGE:
        val_apfd, _ = evaluate_alpha(alpha, gnn_preds, dnn_model, fe, full_train_df, val_with_fail)
        alpha_results.append({'alpha': alpha, 'val_apfd': val_apfd})
        label = ""
        if alpha == 1.0:
            label = " (GNN only)"
        elif alpha == 0.0:
            label = " (DNN only)"
        logger.info(f"  alpha={alpha:.1f}: val APFD = {val_apfd:.4f}{label}")
        if val_apfd > best_val_apfd:
            best_val_apfd = val_apfd
            best_alpha = alpha

    logger.info(f"\nBest alpha: {best_alpha:.1f} (val APFD = {best_val_apfd:.4f})")

    # 4. Evaluate on test set with multiple alpha values
    logger.info("\nEvaluating on 277 test builds...")
    test_builds = test_df['Build_ID'].unique().tolist()

    results = {}
    for alpha in ALPHA_RANGE:
        test_apfd, per_build = evaluate_alpha(alpha, gnn_preds, dnn_model, fe, test_df, test_builds)
        results[alpha] = {
            'apfd': test_apfd,
            'std': float(np.std(per_build)) if per_build else 0,
            'n_builds': len(per_build),
            'per_build': per_build,
        }
        label = ""
        if alpha == 1.0:
            label = " (GNN only)"
        elif alpha == 0.0:
            label = " (DNN only)"
        logger.info(f"  alpha={alpha:.1f}: test APFD = {test_apfd:.4f} (n={len(per_build)}){label}")

    # 5. Summary
    gnn_only = results[1.0]['apfd']
    dnn_only = results[0.0]['apfd']
    best_ensemble = results[best_alpha]['apfd']

    print("\n" + "=" * 70)
    print("RESULTS: GNN+DNN Ensemble on Industrial Dataset")
    print("=" * 70)
    print(f"\nGNN-only (alpha=1.0):     APFD = {gnn_only:.4f}")
    print(f"DNN-only (alpha=0.0):     APFD = {dnn_only:.4f}")
    print(f"Best ensemble (alpha={best_alpha:.1f}): APFD = {best_ensemble:.4f}")
    print(f"Frozen V3 reference:      APFD = 0.7611")
    print(f"\nAlpha sweep on test set:")
    for alpha in ALPHA_RANGE:
        r = results[alpha]
        marker = " <<<" if alpha == best_alpha else ""
        print(f"  alpha={alpha:.1f}: {r['apfd']:.4f} (std={r['std']:.4f}, n={r['n_builds']}){marker}")

    delta_ensemble_vs_gnn = best_ensemble - gnn_only
    print(f"\nEnsemble vs GNN-only: {delta_ensemble_vs_gnn:+.4f} ({delta_ensemble_vs_gnn/gnn_only*100:+.1f}%)")

    if abs(delta_ensemble_vs_gnn) < 0.005:
        print("CONCLUSION: DNN ensemble provides negligible improvement on industrial dataset.")
        print("  The GNN captures sufficient signal from the rich multi-edge graph.")
    elif delta_ensemble_vs_gnn > 0:
        print("CONCLUSION: DNN ensemble IMPROVES industrial performance.")
    else:
        print("CONCLUSION: DNN ensemble DEGRADES industrial performance.")

    # 6. Save
    summary = {
        'gnn_only_apfd': gnn_only,
        'dnn_only_apfd': dnn_only,
        'best_alpha': best_alpha,
        'best_val_apfd': best_val_apfd,
        'best_ensemble_apfd': best_ensemble,
        'frozen_v3_apfd': 0.7611,
        'alpha_sweep': {str(a): results[a]['apfd'] for a in ALPHA_RANGE},
        'alpha_val_sweep': {r['alpha']: r['val_apfd'] for r in alpha_results},
        'train_time_s': train_time,
        'n_builds': results[best_alpha]['n_builds'],
    }
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
