#!/usr/bin/env python3
"""
RETECS Rankings Export - Industry Dataset (277 builds)

Re-runs RETECS (Network Agent + tcfail reward) on the 01_industry dataset using
the same configuration as `run_retecs_industry.py`, but instead of computing only
APFD, it dumps the full ranked test list per build to CSV.

Output (long format CSV, one row per (build, ranked test)):
    build_id, rank, test_id, label

Where:
    - rank   : 1-based position in the RETECS-produced ordering (1 = highest priority)
    - test_id: TC_Key
    - label  : 1 if the test failed in this build, 0 otherwise

This file is the input the researcher needs to compute NDCG, MRR and MAP
externally, with the same train/test split, seed, and evaluation universe
(only builds with >=1 failure) as the published APFD result (0.6406).

Usage:
    python experiments/run_retecs_industry_rankings.py
"""

import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.retecs import RETECSPrioritizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


CONFIG = {
    'train_path': 'datasets/01_industry/train.csv',
    'test_path': 'datasets/01_industry/test.csv',
    'build_col': 'Build_ID',
    'test_col': 'TC_Key',
    'result_col': 'TE_Test_Result',
    'seed': 42,
    'output_dir': 'results/retecs_industry',
    'agent_type': 'network',
    'reward_func': 'tcfail',
}


def normalize_results(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    df = df.copy()
    col = config['result_col']
    df[col] = df[col].astype(str).str.strip().apply(lambda x: 'Fail' if x == 'Fail' else 'Pass')
    return df


def load_data(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Loading datasets...")
    train_df = normalize_results(pd.read_csv(PROJECT_ROOT / config['train_path']), config)
    test_df = normalize_results(pd.read_csv(PROJECT_ROOT / config['test_path']), config)
    logger.info(f"  Train: {len(train_df):,} rows, {train_df[config['build_col']].nunique()} builds")
    logger.info(f"  Test:  {len(test_df):,} rows, {test_df[config['build_col']].nunique()} builds")
    return train_df, test_df


def get_builds_with_failures(df: pd.DataFrame, config: Dict) -> List:
    build_col = config['build_col']
    result_col = config['result_col']
    builds = []
    for build_id, group in df.groupby(build_col):
        if (group[result_col].astype(str).str.strip() == 'Fail').any():
            builds.append(build_id)
    return builds


def main():
    print("\n" + "=" * 70)
    print("RETECS Rankings Export - Industry Dataset")
    print("=" * 70 + "\n")

    np.random.seed(CONFIG['seed'])
    random.seed(CONFIG['seed'])

    train_df, test_df = load_data(CONFIG)

    build_col = CONFIG['build_col']
    test_col = CONFIG['test_col']
    result_col = CONFIG['result_col']

    logger.info(f"Initializing RETECS (agent={CONFIG['agent_type']}, reward={CONFIG['reward_func']})")
    prioritizer = RETECSPrioritizer(
        agent_type=CONFIG['agent_type'],
        reward_func=CONFIG['reward_func'],
        seed=CONFIG['seed']
    )

    # Training phase (identical to the APFD experiment so the agent state matches)
    train_builds = train_df[build_col].unique().tolist()
    logger.info(f"Training on {len(train_builds)} builds...")
    train_start = time.time()

    for build_id in train_builds:
        build_df = train_df[train_df[build_col] == build_id]
        test_ids = build_df[test_col].unique().tolist()

        verdicts: Dict[str, int] = {}
        for _, row in build_df.iterrows():
            tc = row[test_col]
            verdict = 1 if str(row[result_col]).strip() == 'Fail' else 0
            verdicts[tc] = max(verdicts.get(tc, 0), verdict)

        prioritizer.train_on_build(test_ids, verdicts)

    logger.info(f"Training done in {time.time()-train_start:.1f}s")

    # Evaluation phase: dump rankings for every test build with >=1 failure
    eval_builds = get_builds_with_failures(test_df, CONFIG)
    logger.info(f"Exporting rankings for {len(eval_builds)} test builds with failures...")

    rows: List[Dict] = []
    eval_start = time.time()

    for build_id in eval_builds:
        build_df = test_df[test_df[build_col] == build_id]
        test_ids = build_df[test_col].unique().tolist()

        verdicts = {}
        for _, row in build_df.iterrows():
            tc = row[test_col]
            verdict = 1 if str(row[result_col]).strip() == 'Fail' else 0
            verdicts[tc] = max(verdicts.get(tc, 0), verdict)

        if sum(verdicts.values()) == 0:
            continue

        ranking = prioritizer.prioritize(test_ids)

        for rank_pos, tc in enumerate(ranking, start=1):
            rows.append({
                'build_id': build_id,
                'rank': rank_pos,
                'test_id': tc,
                'label': int(verdicts[tc]),
            })

        # Update history exactly like the original experiment to keep agent state aligned
        prioritizer.update_history(test_ids, verdicts)

    logger.info(f"Evaluation done in {time.time()-eval_start:.1f}s")

    output_dir = PROJECT_ROOT / CONFIG['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / 'rankings_per_build_industry.csv'

    df_out = pd.DataFrame(rows, columns=['build_id', 'rank', 'test_id', 'label'])
    df_out.to_csv(out_path, index=False)

    n_builds = df_out['build_id'].nunique()
    n_rows = len(df_out)
    n_failures = int(df_out['label'].sum())
    print("\n" + "=" * 70)
    print(f"Saved rankings to: {out_path}")
    print(f"  Builds exported : {n_builds}")
    print(f"  Rows (tests)    : {n_rows}")
    print(f"  Failure rows    : {n_failures}")
    print("=" * 70)


if __name__ == '__main__':
    main()
