#!/usr/bin/env python3
"""
Control Experiment: FailRank-BB with RANDOM Embeddings on RTPTorrent

This script tests whether FailRank-BB's performance on RTPTorrent is driven by
actual semantic content in test names, or whether the SBERT embeddings merely
serve as unique identifiers (fingerprints) for each test case, allowing the
LogisticRegression to learn per-test failure rates indirectly.

Three conditions:
  1. SBERT (original):  real SBERT embeddings of test names
  2. RANDOM-FIXED:      random but CONSISTENT embeddings per test name
                        (same test always gets the same random vector)
  3. RANDOM-SHUFFLED:   random embeddings re-assigned each build
                        (no identity signal at all)

If RANDOM-FIXED ≈ SBERT >> RANDOM-SHUFFLED, then SBERT is acting as an
identity proxy, not capturing semantic meaning.

If SBERT >> RANDOM-FIXED ≈ RANDOM-SHUFFLED, then semantic content matters.

Usage:
    python experiments/run_failrank_bb_rtptorrent_control.py
"""

import gc
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEED = 42
EMBEDDING_DIM = 768  # Same as bert-base-uncased
DATA_DIR = PROJECT_ROOT / 'datasets/02_rtptorrent/raw/MSR2'
OUTPUT_DIR = PROJECT_ROOT / 'results/failrank_bb_control'
TRAIN_RATIO = 0.8
SKIP_DIRS = {'repo'}


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


def get_projects():
    projects = []
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir() and d.name not in SKIP_DIRS:
            csv_file = d / f"{d.name}.csv"
            if csv_file.exists():
                projects.append(d)
    return projects


def run_condition(project_dir, condition, sbert_model=None):
    """Run one condition on one project. Returns mean APFD or None."""
    project_name = project_dir.name
    df = pd.read_csv(project_dir / f"{project_name}.csv")
    df['is_failure'] = ((df['failures'] > 0) | (df['errors'] > 0)).astype(int)

    builds = df['travisJobId'].unique().tolist()
    n_builds = len(builds)
    if n_builds < 5:
        return None

    train_idx = int(n_builds * TRAIN_RATIO)
    train_builds = set(builds[:train_idx])
    test_builds = builds[train_idx:]

    # Get unique test names
    all_test_names = df['testName'].unique().tolist()

    # Generate embeddings based on condition
    rng = np.random.RandomState(SEED)

    if condition == 'sbert':
        # Real SBERT embeddings
        sbert_model._load_sbert() if hasattr(sbert_model, '_load_sbert') else None
        from sentence_transformers import SentenceTransformer, models as st_models
        if sbert_model is None:
            transformer = st_models.Transformer('google-bert/bert-base-uncased')
            pooling = st_models.Pooling(
                transformer.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True
            )
            sbert_model = SentenceTransformer(modules=[transformer, pooling], device='cuda')
        embeddings = sbert_model.encode(
            all_test_names, batch_size=64, show_progress_bar=False, convert_to_numpy=True
        )
        emb_map = {name: embeddings[i].astype(np.float32) for i, name in enumerate(all_test_names)}
    elif condition == 'random_fixed':
        # Random but consistent per test name
        emb_map = {name: rng.randn(EMBEDDING_DIM).astype(np.float32)
                   for name in all_test_names}
    elif condition == 'random_shuffled':
        # Will generate fresh random vectors each time - no identity signal
        emb_map = None  # handled in the loop
    else:
        raise ValueError(f"Unknown condition: {condition}")

    # Build training data
    train_df = df[df['travisJobId'].isin(train_builds)]

    # Aggregate: for each unique test, count pass/fail
    tc_stats = train_df.groupby('testName')['is_failure'].agg(['sum', 'count']).reset_index()
    tc_stats.columns = ['testName', 'n_fail', 'n_total']
    tc_stats['n_pass'] = tc_stats['n_total'] - tc_stats['n_fail']

    # Build feature matrix and labels with sample weights
    tc_names_train = tc_stats['testName'].tolist()

    if condition == 'random_shuffled':
        X_train = rng.randn(len(tc_names_train), EMBEDDING_DIM * 2).astype(np.float32)
    else:
        X_train = np.array([
            np.concatenate([np.zeros(EMBEDDING_DIM, dtype=np.float32), emb_map[n]])
            for n in tc_names_train
        ])

    # Weighted: each row represents aggregated pass/fail for one test
    # We create two rows per test: one for fail count, one for pass count
    X_expanded = []
    y_expanded = []
    w_expanded = []
    for i, name in enumerate(tc_names_train):
        row = tc_stats.iloc[i]
        if row['n_fail'] > 0:
            X_expanded.append(X_train[i])
            y_expanded.append(1)
            w_expanded.append(row['n_fail'])
        if row['n_pass'] > 0:
            X_expanded.append(X_train[i])
            y_expanded.append(0)
            w_expanded.append(row['n_pass'])

    X_train_exp = np.array(X_expanded)
    y_train_exp = np.array(y_expanded)
    w_train_exp = np.array(w_expanded, dtype=np.float64)

    # Train classifier
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train_exp, y_train_exp, sample_weight=w_train_exp)

    # Evaluate on test builds
    test_df = df[df['travisJobId'].isin(set(test_builds))]
    test_grouped = test_df.groupby('travisJobId')

    apfd_scores = []
    for build_id in test_builds:
        if build_id not in test_grouped.groups:
            continue
        bdf = test_grouped.get_group(build_id)
        verdicts = bdf.groupby('testName')['is_failure'].max().to_dict()
        test_ids = list(verdicts.keys())
        n_fail = sum(verdicts.values())
        if n_fail == 0:
            continue

        # Build feature vectors for this build's tests
        if condition == 'random_shuffled':
            X_test = rng.randn(len(test_ids), EMBEDDING_DIM * 2).astype(np.float32)
        else:
            X_test = np.array([
                np.concatenate([np.zeros(EMBEDDING_DIM, dtype=np.float32), emb_map[t]])
                for t in test_ids
            ])

        # Predict P(fail)
        probs = clf.predict_proba(X_test)
        fail_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else 0
        fail_probs = probs[:, fail_idx]

        # Rank by descending P(fail)
        order = np.argsort(-fail_probs)
        ranked_tests = [test_ids[i] for i in order]
        labels = np.array([verdicts[t] for t in ranked_tests])
        ranks = np.arange(1, len(ranked_tests) + 1)

        apfd = calculate_apfd(ranks, labels)
        if apfd is not None:
            apfd_scores.append(apfd)

    if not apfd_scores:
        return None

    return float(np.mean(apfd_scores)), len(apfd_scores), sbert_model


def main():
    np.random.seed(SEED)
    random.seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    projects = get_projects()
    logger.info(f"Found {len(projects)} projects")

    conditions = ['sbert', 'random_fixed', 'random_shuffled']
    results = {c: [] for c in conditions}

    sbert_model = None

    # Load SBERT model once
    from sentence_transformers import SentenceTransformer, models as st_models
    transformer = st_models.Transformer('google-bert/bert-base-uncased')
    pooling = st_models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )
    sbert_model = SentenceTransformer(modules=[transformer, pooling], device='cuda')

    for pi, proj in enumerate(projects, 1):
        logger.info(f"[{pi}/{len(projects)}] {proj.name}")
        for cond in conditions:
            t0 = time.time()
            result = run_condition(proj, cond, sbert_model if cond == 'sbert' else None)
            elapsed = time.time() - t0
            if result is not None:
                apfd, n_builds, _ = result
                results[cond].append({
                    'project': proj.name,
                    'mean_apfd': apfd,
                    'n_builds': n_builds,
                    'time': elapsed
                })
                logger.info(f"  {cond:20s} APFD={apfd:.4f} ({n_builds} builds, {elapsed:.1f}s)")
            else:
                logger.info(f"  {cond:20s} SKIPPED")

    # Summary
    print("\n" + "=" * 70)
    print("CONTROL EXPERIMENT: FailRank-BB Embedding Source Analysis")
    print("=" * 70)

    summary = {}
    for cond in conditions:
        if results[cond]:
            apfds = [r['mean_apfd'] for r in results[cond]]
            mean = float(np.mean(apfds))
            std = float(np.std(apfds))
            summary[cond] = {'mean': mean, 'std': std, 'n': len(apfds)}
            print(f"\n{cond:20s}: APFD = {mean:.4f} ± {std:.4f} ({len(apfds)} projects)")
            for r in results[cond]:
                print(f"  {r['project']:40s} {r['mean_apfd']:.4f}")

    print("\n" + "-" * 70)
    print("INTERPRETATION:")
    if summary.get('sbert') and summary.get('random_fixed'):
        sbert_apfd = summary['sbert']['mean']
        fixed_apfd = summary['random_fixed']['mean']
        shuffled_apfd = summary.get('random_shuffled', {}).get('mean', 0)

        diff_sf = sbert_apfd - fixed_apfd
        diff_fs = fixed_apfd - shuffled_apfd

        if abs(diff_sf) < 0.02 and diff_fs > 0.05:
            print(">>> SBERT ≈ RANDOM-FIXED >> RANDOM-SHUFFLED")
            print(">>> CONCLUSION: SBERT acts as IDENTITY PROXY, not semantic signal.")
            print(">>> The LogisticRegression learns per-test failure rates via embeddings.")
        elif diff_sf > 0.03:
            print(">>> SBERT >> RANDOM-FIXED")
            print(">>> CONCLUSION: Semantic content in test names IS informative.")
        else:
            print(f">>> SBERT - RANDOM_FIXED = {diff_sf:+.4f}")
            print(f">>> RANDOM_FIXED - RANDOM_SHUFFLED = {diff_fs:+.4f}")
            print(">>> Results are ambiguous; manual analysis required.")

    # Save results
    with open(OUTPUT_DIR / 'control_results.json', 'w') as f:
        json.dump({
            'conditions': {c: results[c] for c in conditions},
            'summary': summary,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
