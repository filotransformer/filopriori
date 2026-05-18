#!/usr/bin/env python3
"""
Heuristic Baselines on Industrial Dataset

Computes 5 heuristic baselines on the industrial QTA dataset:
  1. Random:             Random ordering (expected APFD ~ 0.5)
  2. Recency:            Tests that failed most recently first
  3. RecentFailureRate:  Ranked by failure rate in last 5 builds
  4. FailureRate:        Ranked by overall historical failure rate
  5. GreedyHistorical:   Composite: recency + failure rate + test age

Each heuristic uses ONLY training history accumulated up to
the current build (strict temporal ordering, no future leakage).

Usage:
    python experiments/run_heuristics_industry.py
"""

import json
import logging
import random
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEED = 42
OUTPUT_DIR = PROJECT_ROOT / 'results/heuristics_industry'
RECENT_WINDOW = 5


def calculate_apfd(ranks, labels):
    n = len(labels)
    fail_idx = np.where(np.array(labels).astype(int) != 0)[0]
    m = len(fail_idx)
    if m == 0:
        return None
    if n == 1:
        return 1.0
    return float(np.clip(
        1.0 - np.array(ranks)[fail_idx].sum() / (m * n) + 1.0 / (2.0 * n), 0, 1
    ))


class TestHistory:
    """Maintains per-test execution history for heuristic computation."""

    def __init__(self, recent_window=5):
        self.recent_window = recent_window
        self._exec_count = defaultdict(int)
        self._fail_count = defaultdict(int)
        self._last_fail_build_idx = {}      # test -> build index of last failure
        self._recent_verdicts = defaultdict(lambda: deque(maxlen=recent_window))
        self._first_seen_build_idx = {}     # test -> build index first seen
        self._build_idx = 0

    def update(self, build_id, test_verdicts):
        """Update history with results from one build.
        test_verdicts: dict {test_id: 0 or 1}
        """
        self._build_idx += 1
        for tc, verdict in test_verdicts.items():
            self._exec_count[tc] += 1
            self._recent_verdicts[tc].append(verdict)
            if verdict == 1:
                self._fail_count[tc] += 1
                self._last_fail_build_idx[tc] = self._build_idx
            if tc not in self._first_seen_build_idx:
                self._first_seen_build_idx[tc] = self._build_idx

    def failure_rate(self, tc):
        n = self._exec_count.get(tc, 0)
        if n == 0:
            return 0.0
        return self._fail_count[tc] / n

    def recent_failure_rate(self, tc):
        recent = self._recent_verdicts.get(tc)
        if not recent or len(recent) == 0:
            return 0.0
        return sum(recent) / len(recent)

    def builds_since_last_fail(self, tc):
        last = self._last_fail_build_idx.get(tc)
        if last is None:
            return self._build_idx  # never failed
        return self._build_idx - last

    def test_age(self, tc):
        first = self._first_seen_build_idx.get(tc)
        if first is None:
            return 0
        return self._build_idx - first

    @property
    def current_build_idx(self):
        return self._build_idx


def rank_random(test_ids, history, rng):
    """Random ordering."""
    order = list(test_ids)
    rng.shuffle(order)
    return order


def rank_recency(test_ids, history, rng):
    """Most recently failed tests first."""
    scores = []
    for tc in test_ids:
        bslf = history.builds_since_last_fail(tc)
        scores.append(-bslf)  # lower bslf = more recent = higher priority
    order = np.argsort(scores)  # ascending (most negative = most recent)
    # Break ties randomly
    unique_scores = np.array(scores)
    for val in np.unique(unique_scores):
        mask = unique_scores == val
        if mask.sum() > 1:
            indices = np.where(mask)[0]
            rng.shuffle(indices)
            order_list = list(order)
            positions = sorted([order_list.index(i) for i in indices])
            for pos, idx in zip(positions, indices):
                order_list[pos] = idx
            order = np.array(order_list)
    return [test_ids[i] for i in order]


def rank_recent_failure_rate(test_ids, history, rng):
    """Ranked by failure rate in last N builds."""
    scores = [history.recent_failure_rate(tc) for tc in test_ids]
    # Add small noise for tie-breaking
    noise = np.array([rng.random() * 1e-8 for _ in test_ids])
    combined = np.array(scores) + noise
    order = np.argsort(-combined)
    return [test_ids[i] for i in order]


def rank_failure_rate(test_ids, history, rng):
    """Ranked by overall historical failure rate."""
    scores = [history.failure_rate(tc) for tc in test_ids]
    noise = np.array([rng.random() * 1e-8 for _ in test_ids])
    combined = np.array(scores) + noise
    order = np.argsort(-combined)
    return [test_ids[i] for i in order]


def rank_greedy_historical(test_ids, history, rng):
    """Composite: recency (0.5) + failure_rate (0.3) + novelty (0.2)."""
    scores = []
    max_age = max(history.test_age(tc) for tc in test_ids) if test_ids else 1
    max_age = max(max_age, 1)

    for tc in test_ids:
        bslf = history.builds_since_last_fail(tc)
        recency = 1.0 / (1.0 + bslf)  # higher if failed recently
        fr = history.failure_rate(tc)
        age = history.test_age(tc)
        novelty = 1.0 - (age / max_age)  # newer tests get higher score
        composite = 0.5 * recency + 0.3 * fr + 0.2 * novelty
        scores.append(composite)

    noise = np.array([rng.random() * 1e-8 for _ in test_ids])
    combined = np.array(scores) + noise
    order = np.argsort(-combined)
    return [test_ids[i] for i in order]


HEURISTICS = {
    'Random': rank_random,
    'Recency': rank_recency,
    'RecentFailureRate': rank_recent_failure_rate,
    'FailureRate': rank_failure_rate,
    'GreedyHistorical': rank_greedy_historical,
}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Heuristic Baselines on Industrial Dataset")
    logger.info("=" * 70)

    # Load data
    train_df = pd.read_csv(PROJECT_ROOT / 'datasets/01_industry/train.csv')
    test_df = pd.read_csv(PROJECT_ROOT / 'datasets/01_industry/test.csv')

    for df in [train_df, test_df]:
        df['is_failure'] = (df['TE_Test_Result'].astype(str).str.strip() == 'Fail').astype(int)

    train_builds = train_df['Build_ID'].unique().tolist()
    test_builds = test_df['Build_ID'].unique().tolist()

    logger.info(f"Train: {len(train_builds)} builds")
    logger.info(f"Test: {len(test_builds)} builds")

    # Build history from training data
    history = TestHistory(recent_window=RECENT_WINDOW)
    train_grouped = train_df.groupby('Build_ID', sort=False)

    logger.info("Building history from training builds...")
    for bid in train_builds:
        if bid not in train_grouped.groups:
            continue
        bdf = train_grouped.get_group(bid)
        verdicts = bdf.groupby('TC_Key')['is_failure'].max().to_dict()
        history.update(bid, verdicts)

    logger.info(f"History built: {len(history._exec_count)} unique tests, "
                f"{history.current_build_idx} builds processed")

    # Evaluate each heuristic on test builds
    test_grouped = test_df.groupby('Build_ID', sort=False)

    results = {}
    for hname, hfunc in HEURISTICS.items():
        logger.info(f"\nEvaluating: {hname}")

        # Reset RNG for reproducibility per heuristic
        h_rng = random.Random(SEED)

        # Create a copy of history for this heuristic (online update)
        import copy
        h_history = copy.deepcopy(history)

        apfd_scores = []
        per_build = []

        for bid in test_builds:
            if bid not in test_grouped.groups:
                continue
            bdf = test_grouped.get_group(bid)
            verdicts = bdf.groupby('TC_Key')['is_failure'].max().to_dict()
            test_ids = list(verdicts.keys())
            n_fail = sum(verdicts.values())

            if n_fail > 0:
                ranking = hfunc(test_ids, h_history, h_rng)
                labels = np.array([verdicts[tc] for tc in ranking])
                ranks = np.arange(1, len(ranking) + 1)
                apfd = calculate_apfd(ranks, labels)

                if apfd is not None:
                    apfd_scores.append(apfd)
                    per_build.append({
                        'build_id': bid, 'apfd': apfd,
                        'n_tc': len(test_ids), 'n_fail': n_fail
                    })

            # Online history update
            h_history.update(bid, verdicts)

        mean_apfd = float(np.mean(apfd_scores)) if apfd_scores else 0.0
        std_apfd = float(np.std(apfd_scores)) if apfd_scores else 0.0

        results[hname] = {
            'mean_apfd': mean_apfd,
            'std_apfd': std_apfd,
            'n_builds': len(apfd_scores),
            'median_apfd': float(np.median(apfd_scores)) if apfd_scores else 0.0,
            'per_build': per_build,
        }
        logger.info(f"  APFD = {mean_apfd:.4f} +/- {std_apfd:.4f} (n={len(apfd_scores)})")

    # Summary
    print("\n" + "=" * 70)
    print("HEURISTIC BASELINES - Industrial Dataset (277 builds)")
    print("=" * 70)

    fp_apfd = 0.7611
    print(f"\n{'Method':25s} {'APFD':>8s} {'Std':>8s} {'N':>5s} {'vs FP':>8s}")
    print("-" * 60)
    print(f"{'Filo-Priori (ref)':25s} {fp_apfd:8.4f} {'0.189':>8s} {'277':>5s} {'--':>8s}")
    for hname in ['GreedyHistorical', 'Recency', 'RecentFailureRate',
                   'FailureRate', 'Random']:
        r = results[hname]
        delta = (fp_apfd - r['mean_apfd']) / r['mean_apfd'] * 100
        print(f"{hname:25s} {r['mean_apfd']:8.4f} {r['std_apfd']:8.4f} "
              f"{r['n_builds']:5d} {delta:+7.1f}%")

    # Save
    summary = {
        'dataset': 'industrial',
        'n_test_builds_with_failures': 277,
        'filo_priori_apfd': fp_apfd,
        'results': {h: {k: v for k, v in r.items() if k != 'per_build'}
                    for h, r in results.items()},
        'timestamp': datetime.now().isoformat(),
    }
    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Per-build CSV
    all_rows = []
    for hname, r in results.items():
        for pb in r['per_build']:
            pb['method'] = hname
            all_rows.append(pb)
    pd.DataFrame(all_rows).to_csv(OUTPUT_DIR / 'apfd_per_build.csv', index=False)

    logger.info(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
