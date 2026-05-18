#!/usr/bin/env python3
"""
Filo-Priori V14 — Ablation, Sensitivity, and Temporal Validation for RTPTorrent

This script runs three complementary analyses on the RTPTorrent dataset (20 projects):
  1. Ablation Study: Remove components one at a time
  2. Temporal 5-Fold CV: Train on past, test on future
  3. Sensitivity Analysis: Vary key hyperparameters

All experiments use the FROZEN V14 hyperparameters as the baseline.
DO NOT modify the base CONFIG — only the VariantConfig overrides change behavior.

Usage:
    python experiments/run_rtptorrent_ablation_sensitivity.py [--ablation] [--temporal] [--sensitivity] [--all]
"""

import argparse
import copy
import gc
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.deeporder import (
    DeepOrderModel, DeepOrderFeatureExtractor, DeepOrderNet, DeepOrderDataset
)
from src.embeddings.sbert_encoder import SBERTEncoder
from src.models.model_factory import create_model
from src.phylogenetic.phylogenetic_graph_builder import PhylogeneticGraphBuilder
from src.preprocessing.structural_feature_extractor_v2_5 import StructuralFeatureExtractorV2_5
from src.preprocessing.priority_score_generator import PriorityScoreGenerator
from src.training.losses import FocalLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# =============================================================================
# FROZEN V14 CONFIG (DO NOT MODIFY)
# =============================================================================
BASE_CONFIG = {
    'data_dir': 'datasets/02_rtptorrent/raw/MSR2',
    'seed': 42,
    'output_dir': 'results/rtptorrent_ablation_sensitivity',
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'device': DEFAULT_DEVICE,
    'max_train_rows': 500_000,
    'sbert_model': 'sentence-transformers/all-mpnet-base-v2',
    'sbert_batch_size': 64,
    'embedding_dim': 768,
    'model': {
        'type': 'dual_stream_v8',
        'semantic': {'input_dim': 768, 'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.3, 'activation': 'gelu'},
        'structural': {'input_dim': 19, 'hidden_dim': 128, 'num_heads': 4, 'dropout': 0.3, 'activation': 'elu', 'use_edge_weights': True},
        'fusion': {'type': 'cross_attention', 'hidden_dim': 256, 'num_heads': 4, 'dropout': 0.1},
        'classifier': {'hidden_dims': [128, 64], 'dropout': 0.4},
        'num_classes': 2,
    },
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'max_epochs': 30,
    'patience': 7,
    'focal_gamma': 2.0,
    'max_pos_weight': 10.0,
    'do_dnn': {'hidden_dims': [64, 32, 16], 'dropout': 0.2, 'learning_rate': 0.001, 'epochs': 15, 'batch_size': 128, 'history_window': 10},
    'max_do_train_builds': 5000,
    'max_dnn_pos_weight': 50.0,
    'graph_type': 'co_failure',
    'min_co_occurrences': 2,
    'weight_threshold': 0.1,
    'alpha_search_range': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'min_val_failure_builds': 3,
    'min_tcs_for_model': 30,
}

SKIP_DIRS = {'repo'}

# Maximum rows for feat_extractor.fit() to prevent OOM on large projects like SonarSource
MAX_FIT_ROWS = 2_000_000

# Maximum rows to keep when loading very large projects (SonarSource has 17M rows).
# We keep the LAST N rows to preserve recent history (most relevant for TCP).
# This prevents OOM when running 12 sensitivity variants on the same project.
MAX_LOAD_ROWS = 3_000_000


# =============================================================================
# INCREMENTAL SAVING UTILITIES
# =============================================================================
def _load_completed_keys(csv_path, key_cols):
    """Load existing results CSV and return set of completed (project, variant/fold) tuples."""
    if not Path(csv_path).exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        return set(tuple(row[c] for c in key_cols) for _, row in df.iterrows())
    except Exception:
        return set()


def _append_results_to_csv(csv_path, results, exclude_keys=('build_results',)):
    """Append results to CSV, creating header if file doesn't exist."""
    if not results:
        return
    rows = [{k: v for k, v in r.items() if k not in exclude_keys} for r in results]
    df_new = pd.DataFrame(rows)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        df_new.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(csv_path, index=False)


def _trim_tc_history(feat_extractor):
    """Remove build_history and result_history from tc_history to free memory.
    These lists grow with each execution but are NOT needed by transform()."""
    base = feat_extractor.extractor if hasattr(feat_extractor, 'extractor') else feat_extractor
    if hasattr(base, 'tc_history'):
        for tc_data in base.tc_history.values():
            tc_data.pop('build_history', None)
            tc_data.pop('result_history', None)
            tc_data.pop('recent_results', None)


# =============================================================================
# VARIANT CONFIGURATION
# =============================================================================
@dataclass
class VariantConfig:
    name: str
    use_gnn: bool = True
    use_dnn: bool = True
    use_semantic: bool = True
    use_graph: bool = True
    use_deeporder_features: bool = True
    fixed_alpha: Optional[float] = None
    dnn_epochs: Optional[int] = None
    max_dnn_pos_weight: Optional[float] = None
    # Temporal CV overrides
    train_builds: Optional[List] = None
    test_builds: Optional[List] = None


# =============================================================================
# UTILITY FUNCTIONS (from V14)
# =============================================================================
def calculate_apfd_single_build(ranks, labels):
    labels_arr = np.array(labels)
    ranks_arr = np.array(ranks)
    n_tests = len(labels_arr)
    fail_indices = np.where(labels_arr.astype(int) != 0)[0]
    n_failures = len(fail_indices)
    if n_failures == 0:
        return None
    if n_tests == 1:
        return 1.0
    failure_ranks = ranks_arr[fail_indices]
    apfd = 1.0 - float(failure_ranks.sum()) / float(n_failures * n_tests) + 1.0 / float(2.0 * n_tests)
    return float(np.clip(apfd, 0.0, 1.0))


def get_project_dirs(data_dir):
    projects = []
    for d in sorted(Path(data_dir).iterdir()):
        if d.is_dir() and d.name not in SKIP_DIRS:
            csv_file = d / f"{d.name}.csv"
            if csv_file.exists():
                projects.append(d)
    return projects


def load_project_data(project_dir):
    csv_path = project_dir / f"{project_dir.name}.csv"
    df = pd.read_csv(csv_path)
    df['is_failure'] = ((df['failures'] > 0) | (df['errors'] > 0)).astype(int)
    return df


def generate_semantic_text(fqn):
    parts = fqn.split('.')
    tokens = []
    for part in parts:
        split = re.sub(r'([A-Z])', r' \1', part).strip().lower().split()
        tokens.extend(split)
    return ' '.join(tokens)


def _get_dnn_scores(do_model, test_ids, device):
    if do_model is None or do_model.model is None:
        return np.full(len(test_ids), 0.5)
    do_model.model.eval()
    features_list = [do_model.feature_extractor.extract_features(tc) for tc in test_ids]
    X = np.array(features_list)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = do_model.model(X_tensor).cpu().numpy()
    return predictions


def _set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# CORE: run_project_variant — parameterized V14 pipeline
# =============================================================================
def run_project_variant(
    project_name: str,
    df: pd.DataFrame,
    all_builds: List,
    variant: VariantConfig,
    config: Dict,
    sbert_encoder: SBERTEncoder,
    tc_to_embedding_cache: Optional[Dict] = None,
) -> Optional[Dict]:
    """Run V14 pipeline with variant-specific modifications."""

    _set_seed(config['seed'])
    device = config['device']
    embedding_dim = config['embedding_dim']

    logger.info(f"  [{variant.name}] Starting...")

    # ----- 1. Train/Test Split -----
    if variant.train_builds is not None and variant.test_builds is not None:
        train_builds = variant.train_builds
        test_builds = variant.test_builds
    else:
        train_idx = int(len(all_builds) * config['train_ratio'])
        train_builds = all_builds[:train_idx]
        test_builds = all_builds[train_idx:]

    # Add columns in-place (idempotent — safe for repeated calls on same df)
    if 'Build_ID' not in df.columns:
        df['Build_ID'] = df['travisJobId'].astype(str)
    if 'TC_Key' not in df.columns:
        df['TC_Key'] = df['testName']
    if 'TE_Test_Result' not in df.columns:
        df['TE_Test_Result'] = df['is_failure'].apply(lambda x: 'Fail' if x == 1 else 'Pass')

    train_df = df[df['travisJobId'].isin(train_builds)].copy()
    test_df = df[df['travisJobId'].isin(test_builds)].copy()

    n_train_failures = (train_df['is_failure'] == 1).sum()
    if n_train_failures < 2:
        logger.warning(f"  [{variant.name}] Only {n_train_failures} failures in training, skipping")
        return None

    # Train/Val split
    inner_train_builds = train_builds[:int(len(train_builds) * (1 - config['val_ratio']))]
    val_builds_list = train_builds[int(len(train_builds) * (1 - config['val_ratio'])):]
    train_inner_df = train_df[train_df['travisJobId'].isin(inner_train_builds)].copy()
    val_df_split = train_df[train_df['travisJobId'].isin(val_builds_list)].copy()

    if len(train_inner_df) > config['max_train_rows']:
        train_inner_df = train_inner_df.sample(n=config['max_train_rows'], random_state=config['seed'])

    # ----- 2. DNN Training -----
    do_model = None
    if variant.use_dnn:
        do_cfg = config['do_dnn'].copy()
        if variant.dnn_epochs is not None:
            do_cfg['epochs'] = variant.dnn_epochs
        dnn_max_pw = variant.max_dnn_pos_weight if variant.max_dnn_pos_weight is not None else config['max_dnn_pos_weight']

        do_model = DeepOrderModel(
            hidden_dims=do_cfg['hidden_dims'], dropout=do_cfg['dropout'],
            learning_rate=do_cfg['learning_rate'], epochs=do_cfg['epochs'],
            batch_size=do_cfg['batch_size'], history_window=do_cfg['history_window'],
            device=device,
        )

        max_do_builds = config.get('max_do_train_builds', len(train_builds))
        all_train_builds_str = train_df['Build_ID'].unique().tolist()
        train_grouped = train_df.groupby('Build_ID', sort=False)
        fe = DeepOrderFeatureExtractor(history_window=do_cfg['history_window'])

        # Pre-warm + extract
        n_warmup = max(0, len(all_train_builds_str) - max_do_builds)
        for bid in all_train_builds_str[:n_warmup]:
            if bid not in train_grouped.groups:
                continue
            bdf = train_grouped.get_group(bid)
            tc_arr, fail_arr = bdf['TC_Key'].values, bdf['is_failure'].values
            dur_arr = bdf['duration'].values if 'duration' in bdf.columns else np.ones(len(bdf))
            test_results = {tc_arr[i]: (int(fail_arr[i]), float(dur_arr[i])) for i in range(len(tc_arr))}
            fe.update_history(bid, test_results)

        features_list, labels_list = [], []
        for bid in all_train_builds_str[n_warmup:]:
            if bid not in train_grouped.groups:
                continue
            bdf = train_grouped.get_group(bid)
            test_ids = bdf['TC_Key'].values
            result_vals, dur_vals = bdf['TE_Test_Result'].values, bdf['duration'].values if 'duration' in bdf.columns else np.ones(len(bdf))
            for i in range(len(test_ids)):
                features_list.append(fe.extract_features(test_ids[i]))
                labels_list.append(1 if str(result_vals[i]).upper() != 'PASS' else 0)
            test_results = {test_ids[i]: (1 if str(result_vals[i]).upper() != 'PASS' else 0, float(dur_vals[i])) for i in range(len(test_ids))}
            fe.update_history(bid, test_results)

        X, y = np.array(features_list), np.array(labels_list)
        del features_list, labels_list, train_grouped
        gc.collect()

        do_model.feature_extractor = fe
        do_model.model = DeepOrderNet(input_dim=X.shape[1], hidden_dims=do_cfg['hidden_dims'], dropout=do_cfg['dropout']).to(device)

        dataset = DeepOrderDataset(X, y)
        dataloader = TorchDataLoader(dataset, batch_size=do_cfg['batch_size'], shuffle=True, drop_last=(len(dataset) > do_cfg['batch_size']))

        raw_pw = (1 - y.mean()) / (y.mean() + 1e-6)
        clamped_pw = min(raw_pw, dnn_max_pw)
        pos_weight_val = torch.tensor(clamped_pw, dtype=torch.float32).to(device)
        criterion_dnn = nn.BCELoss(reduction='none')  # MUST use BCELoss (not BCEWithLogitsLoss)
        optimizer_dnn = optim.Adam(do_model.model.parameters(), lr=do_cfg['learning_rate'])

        do_model.model.train()
        for epoch in range(do_cfg['epochs']):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer_dnn.zero_grad()
                outputs = do_model.model(batch_X)
                per_sample_loss = criterion_dnn(outputs, batch_y)
                weights = torch.where(batch_y == 1, pos_weight_val, torch.ones_like(batch_y))
                (per_sample_loss * weights).mean().backward()
                optimizer_dnn.step()

        del X, y, dataset, dataloader
        gc.collect()

    # ----- 3. SBERT Embeddings -----
    all_tc_keys = list(set(
        train_inner_df['TC_Key'].unique().tolist() +
        val_df_split['TC_Key'].unique().tolist() +
        test_df['TC_Key'].unique().tolist()
    ))
    n_unique_tcs = len(all_tc_keys)

    if tc_to_embedding_cache is not None:
        tc_to_embedding = tc_to_embedding_cache.copy()
        # Add any new TCs not in cache
        missing = [tc for tc in all_tc_keys if tc not in tc_to_embedding]
        if missing:
            texts = [generate_semantic_text(tc) for tc in missing]
            embs = sbert_encoder.encode_texts_batch(texts)
            for i, tc in enumerate(missing):
                tc_to_embedding[tc] = embs[i]
    else:
        texts = [generate_semantic_text(tc) for tc in all_tc_keys]
        embs = sbert_encoder.encode_texts_batch(texts)
        tc_to_embedding = {tc: embs[i] for i, tc in enumerate(all_tc_keys)}

    if not variant.use_semantic:
        tc_to_embedding = {tc: np.zeros(embedding_dim, dtype=np.float32) for tc in tc_to_embedding}

    # ----- 4. Structural Features -----
    feat_extractor = StructuralFeatureExtractorV2_5(
        recent_window=5, very_recent_window=2, medium_term_window=10, min_history=2, verbose=False
    )
    # For very large datasets (SonarSource: 17M rows), subsample for fit() to prevent OOM
    if len(train_df) > MAX_FIT_ROWS:
        # Keep last MAX_FIT_ROWS rows (preserves recent history which matters most)
        fit_df = train_df.tail(MAX_FIT_ROWS)
        feat_extractor.fit(fit_df)
        del fit_df
    else:
        feat_extractor.fit(train_df)
    _trim_tc_history(feat_extractor)
    gc.collect()

    tc_exec_history = {}
    if variant.use_deeporder_features:
        priority_gen = PriorityScoreGenerator(num_cycles=10, decay_type='exponential', decay_factor=0.8)
        # Cap input for very large projects to prevent OOM
        prio_df = train_df.tail(MAX_FIT_ROWS) if len(train_df) > MAX_FIT_ROWS else train_df
        _, tc_exec_history = priority_gen.compute_priorities_for_dataframe(
            prio_df, build_col='Build_ID', tc_col='TC_Key',
            result_col='TE_Test_Result', fail_value='Fail', pass_value='Pass'
        )
        if prio_df is not train_df:
            del prio_df
        structural_input_dim = 19
    else:
        priority_gen = None
        structural_input_dim = 10

    # ----- 5. Graph -----
    graph_builder = PhylogeneticGraphBuilder(
        graph_type=config['graph_type'], min_co_occurrences=config['min_co_occurrences'],
        weight_threshold=config['weight_threshold'], verbose=False,
    )
    # Cap input for very large projects
    graph_df = train_df.tail(MAX_FIT_ROWS) if len(train_df) > MAX_FIT_ROWS else train_df
    graph_builder.fit(graph_df)
    if graph_df is not train_df:
        del graph_df

    # train_df no longer needed — free memory (saves ~2-3GB for SonarSource)
    del train_df
    gc.collect()

    def _get_edges(tc_list):
        """Get edges, with graph ablation support."""
        if not variant.use_graph:
            n = len(tc_list)
            sl = torch.arange(n, dtype=torch.long)
            return torch.stack([sl, sl], dim=0), torch.ones(n, dtype=torch.float32)
        edge_idx, edge_w = graph_builder.get_edge_index_and_weights(tc_list, return_torch=True)
        if edge_idx.shape[1] == 0:
            n = len(tc_list)
            sl = torch.arange(n, dtype=torch.long)
            return torch.stack([sl, sl], dim=0), torch.ones(n, dtype=torch.float32)
        return edge_idx, edge_w

    # ----- 6. GNN Training -----
    use_model = variant.use_gnn and n_unique_tcs >= config['min_tcs_for_model']
    gnn_model = None
    train_time = 0.0

    if use_model:
        train_tc_keys = train_inner_df['TC_Key'].unique().tolist()
        val_tc_keys = val_df_split['TC_Key'].unique().tolist()
        zero_sem = np.zeros(embedding_dim, dtype=np.float32)

        train_node_semantic = np.array([tc_to_embedding.get(tc, zero_sem) for tc in train_tc_keys])
        val_node_semantic = np.array([tc_to_embedding.get(tc, zero_sem) for tc in val_tc_keys])

        def _aggregate_struct(df_subset, tc_list, is_test):
            last_rows = df_subset.drop_duplicates(subset='TC_Key', keep='last').set_index('TC_Key')
            ordered_rows = [last_rows.loc[tc] if tc in last_rows.index else last_rows.iloc[0] for tc in tc_list]
            ordered_df = pd.DataFrame(ordered_rows)
            ordered_df['TC_Key'] = tc_list
            ordered_df['Build_ID'] = ordered_df['Build_ID'].astype(str)
            v25 = feat_extractor.transform(ordered_df, is_test=is_test)
            if variant.use_deeporder_features:
                do_feats = np.array([priority_gen.extract_deeporder_features(tc_exec_history, tc) for tc in tc_list], dtype=np.float32)
                return np.concatenate([v25, do_feats], axis=1)
            return v25

        train_node_struct = _aggregate_struct(train_inner_df, train_tc_keys, is_test=False)
        val_node_struct = _aggregate_struct(val_df_split, val_tc_keys, is_test=True)

        train_failure_by_tc = train_inner_df.groupby('TC_Key')['is_failure'].max()
        train_node_labels = np.array([int(train_failure_by_tc.get(tc, 0)) for tc in train_tc_keys], dtype=np.int64)
        val_failure_by_tc = val_df_split.groupby('TC_Key')['is_failure'].max()
        val_node_labels = np.array([int(val_failure_by_tc.get(tc, 0)) for tc in val_tc_keys], dtype=np.int64)

        edge_index, edge_weights = _get_edges(train_tc_keys)
        val_edge_index, val_edge_weights = _get_edges(val_tc_keys)

        train_sem_t = torch.from_numpy(train_node_semantic).float().to(device)
        train_struct_t = torch.from_numpy(train_node_struct).float().to(device)
        train_labels_t = torch.from_numpy(train_node_labels).long().to(device)
        edge_index, edge_weights = edge_index.to(device), edge_weights.to(device)
        val_sem_t = torch.from_numpy(val_node_semantic).float().to(device)
        val_struct_t = torch.from_numpy(val_node_struct).float().to(device)
        val_labels_t = torch.from_numpy(val_node_labels).long().to(device)
        val_edge_index, val_edge_weights = val_edge_index.to(device), val_edge_weights.to(device)

        class_counts = np.bincount(train_node_labels, minlength=2)

        # Create model with correct structural input dim
        model_config = copy.deepcopy(config['model'])
        model_config['structural']['input_dim'] = structural_input_dim
        gnn_model = create_model(model_config).to(device)

        alpha_w = None
        if class_counts[1] > 0:
            pw = min(class_counts[0] / class_counts[1], config['max_pos_weight'])
            alpha_w = torch.tensor([1.0, pw], dtype=torch.float32).to(device)
        criterion_gnn = FocalLoss(alpha=alpha_w, gamma=config['focal_gamma'])
        optimizer_gnn = torch.optim.AdamW(gnn_model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

        train_start = time.time()
        best_val_loss, patience_counter, best_state = float('inf'), 0, None

        for epoch in range(config['max_epochs']):
            gnn_model.train()
            optimizer_gnn.zero_grad()
            logits = gnn_model(train_sem_t, train_struct_t, edge_index, edge_weights)
            loss = criterion_gnn(logits, train_labels_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), 1.0)
            optimizer_gnn.step()

            gnn_model.eval()
            with torch.no_grad():
                val_loss = criterion_gnn(gnn_model(val_sem_t, val_struct_t, val_edge_index, val_edge_weights), val_labels_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in gnn_model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    break

        train_time = time.time() - train_start
        if best_state:
            gnn_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        del best_state, train_sem_t, train_struct_t, train_labels_t
        del val_sem_t, val_struct_t, val_labels_t, val_edge_index, val_edge_weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----- 7. Alpha Optimization -----
    if variant.fixed_alpha is not None:
        best_alpha = variant.fixed_alpha
    elif not variant.use_gnn or gnn_model is None:
        best_alpha = 0.0
    elif not variant.use_dnn or do_model is None:
        best_alpha = 1.0
    else:
        # Optimize alpha on validation set
        best_alpha = _optimize_alpha_simple(
            gnn_model, do_model, val_builds_list, val_df_split, tc_to_embedding,
            graph_builder, priority_gen, tc_exec_history, feat_extractor,
            _get_edges, variant, device, config
        )

    # ----- 8. Evaluation -----
    eval_start = time.time()
    if gnn_model is not None:
        gnn_model.eval()

    test_grouped = test_df.groupby('travisJobId')
    zero_sem = np.zeros(embedding_dim, dtype=np.float32)
    all_apfd_scores = []
    build_results = []

    for build_id in test_builds:
        if build_id not in test_grouped.groups:
            continue
        build_df = test_grouped.get_group(build_id)
        verdicts = build_df.groupby('testName')['is_failure'].max().to_dict()
        durations = build_df.groupby('testName')['duration'].last().to_dict() if 'duration' in build_df.columns else {}
        test_ids = list(verdicts.keys())
        n_failures = sum(verdicts.values())

        if n_failures == 0:
            # Update histories
            if do_model is not None:
                test_results = {tc: (verdicts[tc], durations.get(tc, 1.0)) for tc in test_ids}
                do_model.update_history(str(build_id), test_results)
            for tc in test_ids:
                if tc not in tc_exec_history:
                    tc_exec_history[tc] = []
                tc_exec_history[tc].append(1 if verdicts[tc] == 1 else 0)
            continue

        dnn_scores = _get_dnn_scores(do_model, test_ids, device)

        if gnn_model is not None and best_alpha > 0:
            sem = np.array([tc_to_embedding.get(tc, zero_sem) for tc in test_ids])
            rows = [build_df[build_df['testName'] == tc].iloc[0] for tc in test_ids]
            bdf = pd.DataFrame(rows)
            bdf['Build_ID'] = str(build_id)
            bdf['TC_Key'] = bdf['testName']
            bdf['TE_Test_Result'] = bdf['is_failure'].apply(lambda x: 'Fail' if x == 1 else 'Pass')
            v25 = feat_extractor.transform(bdf, is_test=True)
            if variant.use_deeporder_features:
                do_feats = np.array([priority_gen.extract_deeporder_features(tc_exec_history, tc) for tc in test_ids], dtype=np.float32)
                struct = np.concatenate([v25, do_feats], axis=1)
            else:
                struct = v25

            build_edge_idx, build_edge_w = _get_edges(test_ids)

            with torch.no_grad():
                logits = gnn_model(
                    torch.from_numpy(sem).float().to(device),
                    torch.from_numpy(struct).float().to(device),
                    build_edge_idx.to(device), build_edge_w.to(device)
                )
                gnn_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            final_scores = best_alpha * gnn_probs + (1 - best_alpha) * dnn_scores
        else:
            final_scores = dnn_scores

        sorted_idx = np.argsort(-final_scores)
        ranking = [test_ids[i] for i in sorted_idx]
        labels = np.array([verdicts[tc] for tc in ranking])
        ranks = np.arange(1, len(ranking) + 1)
        apfd = calculate_apfd_single_build(ranks, labels)

        if apfd is not None:
            all_apfd_scores.append(apfd)
            build_results.append({
                'variant': variant.name, 'project': project_name,
                'build_id': build_id, 'apfd': apfd, 'n_tc': len(test_ids),
            })

        # Online updates
        if do_model is not None:
            test_results = {tc: (verdicts[tc], durations.get(tc, 1.0)) for tc in test_ids}
            do_model.update_history(str(build_id), test_results)
        for tc in test_ids:
            if tc not in tc_exec_history:
                tc_exec_history[tc] = []
            tc_exec_history[tc].append(1 if verdicts[tc] == 1 else 0)

    eval_time = time.time() - eval_start

    # Cleanup
    if gnn_model is not None:
        del gnn_model
    if do_model is not None:
        del do_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not all_apfd_scores:
        return None

    result = {
        'variant': variant.name, 'project': project_name,
        'mean_apfd': float(np.mean(all_apfd_scores)),
        'std_apfd': float(np.std(all_apfd_scores)),
        'median_apfd': float(np.median(all_apfd_scores)),
        'n_builds_with_failures': len(all_apfd_scores),
        'alpha': best_alpha, 'train_time': train_time, 'eval_time': eval_time,
        'build_results': build_results,
    }
    logger.info(f"  [{variant.name}] APFD={result['mean_apfd']:.4f} (n={len(all_apfd_scores)}, alpha={best_alpha:.1f})")
    return result


def _optimize_alpha_simple(gnn_model, do_model, val_builds, val_df, tc_to_embedding,
                           graph_builder, priority_gen, tc_exec_history, feat_extractor,
                           get_edges_fn, variant, device, config):
    """Simplified alpha optimization on validation set."""
    embedding_dim = config['embedding_dim']
    zero_sem = np.zeros(embedding_dim, dtype=np.float32)
    gnn_model.eval()

    build_preds = []
    for build_id in val_builds:
        build_rows = val_df[val_df['travisJobId'] == build_id]
        verdicts = build_rows.groupby('testName')['is_failure'].max().to_dict()
        if not any(verdicts.values()):
            continue
        tcs = list(verdicts.keys())
        dnn_scores = _get_dnn_scores(do_model, tcs, device)

        sem = np.array([tc_to_embedding.get(tc, zero_sem) for tc in tcs])
        rows = [build_rows[build_rows['testName'] == tc].iloc[0] for tc in tcs]
        bdf = pd.DataFrame(rows)
        bdf['Build_ID'] = str(build_id)
        bdf['TC_Key'] = bdf['testName']
        bdf['TE_Test_Result'] = bdf['is_failure'].apply(lambda x: 'Fail' if x == 1 else 'Pass')
        v25 = feat_extractor.transform(bdf, is_test=True)
        if variant.use_deeporder_features:
            do_feats = np.array([priority_gen.extract_deeporder_features(tc_exec_history, tc) for tc in tcs], dtype=np.float32)
            struct = np.concatenate([v25, do_feats], axis=1)
        else:
            struct = v25
        edge_idx, edge_w = get_edges_fn(tcs)

        with torch.no_grad():
            logits = gnn_model(
                torch.from_numpy(sem).float().to(device),
                torch.from_numpy(struct).float().to(device),
                edge_idx.to(device), edge_w.to(device)
            )
            gnn_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        build_preds.append((tcs, gnn_probs, dnn_scores, verdicts))

    if len(build_preds) < config.get('min_val_failure_builds', 3):
        return 0.5

    best_alpha, best_apfd = 0.0, 0.0
    for alpha in config['alpha_search_range']:
        apfds = []
        for tcs, gnn_p, dnn_s, verdicts in build_preds:
            blended = alpha * gnn_p + (1 - alpha) * dnn_s
            ranking = [tcs[i] for i in np.argsort(-blended)]
            labels = np.array([verdicts[tc] for tc in ranking])
            a = calculate_apfd_single_build(np.arange(1, len(ranking) + 1), labels)
            if a is not None:
                apfds.append(a)
        mean_a = np.mean(apfds) if apfds else 0.0
        if mean_a > best_apfd:
            best_apfd = mean_a
            best_alpha = alpha
    return best_alpha


# =============================================================================
# EXPERIMENT 1: ABLATION STUDY
# =============================================================================
def run_ablation_study(config, sbert_encoder):
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 1: ABLATION STUDY (RTPTorrent)")
    logger.info("=" * 70)

    variants = [
        VariantConfig(name='full_v14'),
        VariantConfig(name='wo_gatv2', use_gnn=False, fixed_alpha=0.0),
        VariantConfig(name='wo_dnn_ensemble', use_dnn=False, fixed_alpha=1.0),
        VariantConfig(name='wo_semantic', use_semantic=False),
        VariantConfig(name='wo_graph', use_graph=False),
        VariantConfig(name='wo_deeporder_feats', use_deeporder_features=False),
    ]

    all_results = []
    all_build_results = []

    for project_dir in get_project_dirs(config['data_dir']):
        project_name = project_dir.name
        logger.info(f"\n{'='*60}")
        logger.info(f"Project: {project_name}")
        logger.info(f"{'='*60}")

        df = load_project_data(project_dir)
        builds = df['travisJobId'].unique().tolist()
        if len(builds) < 5:
            continue

        # Cache SBERT embeddings once per project
        all_tc_keys = df['testName'].unique().tolist()
        texts = [generate_semantic_text(tc) for tc in all_tc_keys]
        embs = sbert_encoder.encode_texts_batch(texts)
        tc_to_embedding = {tc: embs[i] for i, tc in enumerate(all_tc_keys)}

        for variant in variants:
            result = run_project_variant(
                project_name, df, builds, variant, config,
                sbert_encoder, tc_to_embedding
            )
            if result:
                all_results.append(result)
                all_build_results.extend(result.get('build_results', []))

        # Free project data between projects
        del df, tc_to_embedding
        gc.collect()

    return all_results, all_build_results


# =============================================================================
# EXPERIMENT 2: TEMPORAL 5-FOLD CV
# =============================================================================
def run_temporal_cv(config, sbert_encoder, n_folds=5):
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 2: TEMPORAL 5-FOLD CV (RTPTorrent)")
    logger.info("=" * 70)

    output_dir = Path(config['output_dir']) / 'temporal_cv'
    csv_path = output_dir / 'per_project_per_fold.csv'

    # Resume: load already-completed (project, variant) pairs
    completed = _load_completed_keys(csv_path, ['project', 'variant'])
    if completed:
        logger.info(f"  Resuming: {len(completed)} project/fold combos already done")

    all_results = []

    for project_dir in get_project_dirs(config['data_dir']):
        project_name = project_dir.name
        logger.info(f"\n{'='*60}")
        logger.info(f"Project: {project_name}")
        logger.info(f"{'='*60}")

        df = load_project_data(project_dir)
        builds = df['travisJobId'].unique().tolist()
        n_builds = len(builds)

        if n_builds < 10:
            logger.warning(f"  Skipping {project_name}: only {n_builds} builds")
            continue

        # Check if all folds already done for this project
        folds_needed = [f for f in range(1, n_folds) if (project_name, f'fold_{f}') not in completed]
        if not folds_needed:
            logger.info(f"  All folds already completed, skipping")
            continue

        # Cache SBERT
        all_tc_keys = df['testName'].unique().tolist()
        texts = [generate_semantic_text(tc) for tc in all_tc_keys]
        embs = sbert_encoder.encode_texts_batch(texts)
        tc_to_embedding = {tc: embs[i] for i, tc in enumerate(all_tc_keys)}

        fold_size = n_builds // n_folds
        project_results = []

        for fold_idx in folds_needed:
            train_end = fold_idx * fold_size
            test_end = min((fold_idx + 1) * fold_size, n_builds)
            train_builds_fold = builds[:train_end]
            test_builds_fold = builds[train_end:test_end]

            if len(train_builds_fold) < 5 or len(test_builds_fold) < 1:
                continue

            variant = VariantConfig(
                name=f'fold_{fold_idx}',
                train_builds=train_builds_fold,
                test_builds=test_builds_fold,
            )

            result = run_project_variant(
                project_name, df, builds, variant, config,
                sbert_encoder, tc_to_embedding
            )
            if result:
                result['fold'] = fold_idx
                result['n_train_builds'] = len(train_builds_fold)
                result['n_test_builds'] = len(test_builds_fold)
                project_results.append(result)
                all_results.append(result)

        # Incremental save after each project
        if project_results:
            _append_results_to_csv(csv_path, project_results)
            logger.info(f"  Saved {len(project_results)} fold results for {project_name}")

        # Free project data
        del df, tc_to_embedding, project_results
        gc.collect()

    return all_results


# =============================================================================
# EXPERIMENT 3: SENSITIVITY ANALYSIS
# =============================================================================
def run_sensitivity_analysis(config, sbert_encoder):
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT 3: SENSITIVITY ANALYSIS (RTPTorrent)")
    logger.info("=" * 70)

    variants = [
        # Alpha sensitivity
        VariantConfig(name='alpha_0.0', fixed_alpha=0.0),
        VariantConfig(name='alpha_0.3', fixed_alpha=0.3),
        VariantConfig(name='alpha_0.5', fixed_alpha=0.5),
        VariantConfig(name='alpha_0.7', fixed_alpha=0.7),
        VariantConfig(name='alpha_1.0', fixed_alpha=1.0),
        VariantConfig(name='alpha_optimized'),  # baseline
        # DNN epochs sensitivity
        VariantConfig(name='dnn_epochs_5', dnn_epochs=5),
        VariantConfig(name='dnn_epochs_10', dnn_epochs=10),
        VariantConfig(name='dnn_epochs_20', dnn_epochs=20),
        # Max pos_weight sensitivity
        VariantConfig(name='pos_weight_10', max_dnn_pos_weight=10.0),
        VariantConfig(name='pos_weight_25', max_dnn_pos_weight=25.0),
        VariantConfig(name='pos_weight_100', max_dnn_pos_weight=100.0),
    ]

    output_dir = Path(config['output_dir']) / 'sensitivity'
    csv_path = output_dir / 'per_variant_per_project.csv'

    # Resume: load already-completed (project, variant) pairs
    completed = _load_completed_keys(csv_path, ['project', 'variant'])
    if completed:
        logger.info(f"  Resuming: {len(completed)} project/variant combos already done")

    all_results = []

    for project_dir in get_project_dirs(config['data_dir']):
        project_name = project_dir.name
        logger.info(f"\n{'='*60}")
        logger.info(f"Project: {project_name}")
        logger.info(f"{'='*60}")

        df = load_project_data(project_dir)
        n_rows_orig = len(df)

        # Memory guard: for very large projects (SonarSource: 17M rows),
        # keep only the last MAX_LOAD_ROWS rows to prevent OOM when running
        # 12 sensitivity variants. This preserves recent history (most relevant).
        if len(df) > MAX_LOAD_ROWS:
            logger.info(f"  Large project: {len(df)} rows → keeping last {MAX_LOAD_ROWS} rows")
            df = df.tail(MAX_LOAD_ROWS).reset_index(drop=True)

        builds = df['travisJobId'].unique().tolist()
        if len(builds) < 5:
            continue

        # Check if all variants already done for this project
        variants_needed = [v for v in variants if (project_name, v.name) not in completed]
        if not variants_needed:
            logger.info(f"  All variants already completed, skipping")
            del df
            gc.collect()
            continue

        # Cache SBERT
        all_tc_keys = df['testName'].unique().tolist()
        texts = [generate_semantic_text(tc) for tc in all_tc_keys]
        embs = sbert_encoder.encode_texts_batch(texts)
        tc_to_embedding = {tc: embs[i] for i, tc in enumerate(all_tc_keys)}
        del texts, embs
        gc.collect()

        project_results = []
        for vi, variant in enumerate(variants_needed):
            logger.info(f"  Variant {vi+1}/{len(variants_needed)}: {variant.name}")
            result = run_project_variant(
                project_name, df, builds, variant, config,
                sbert_encoder, tc_to_embedding
            )
            if result:
                project_results.append(result)
                all_results.append(result)

            # Aggressive cleanup between variants to prevent memory accumulation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Incremental save after each project
        if project_results:
            _append_results_to_csv(csv_path, project_results)
            logger.info(f"  Saved {len(project_results)} variant results for {project_name}")

        # Free project data
        del df, tc_to_embedding, project_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results


# =============================================================================
# RESULTS SAVING
# =============================================================================
def save_results(output_dir, ablation_results, ablation_builds, temporal_results, sensitivity_results):
    """Generate aggregate summary files from incrementally-saved CSVs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ablation (still batch-saved since it completed successfully)
    if ablation_results:
        abl_dir = output_dir / 'ablation'
        abl_dir.mkdir(exist_ok=True)

        df = pd.DataFrame([{k: v for k, v in r.items() if k != 'build_results'} for r in ablation_results])
        df.to_csv(abl_dir / 'per_variant_per_project.csv', index=False)

        agg = df.groupby('variant').agg(
            grand_mean_apfd=('mean_apfd', 'mean'),
            grand_std_apfd=('mean_apfd', 'std'),
            n_projects=('project', 'count'),
        ).reset_index()
        agg.to_csv(abl_dir / 'aggregate_per_variant.csv', index=False)

        if ablation_builds:
            pd.DataFrame(ablation_builds).to_csv(abl_dir / 'per_build_apfd.csv', index=False)

        logger.info(f"\nAblation Summary:")
        for _, row in agg.iterrows():
            logger.info(f"  {row['variant']:25s} APFD={row['grand_mean_apfd']:.4f} ± {row['grand_std_apfd']:.4f}")

    # Temporal CV — aggregate from incrementally-saved CSV
    tcv_csv = output_dir / 'temporal_cv' / 'per_project_per_fold.csv'
    if tcv_csv.exists():
        tcv_dir = output_dir / 'temporal_cv'
        df = pd.read_csv(tcv_csv)
        agg = df.groupby('project').agg(
            mean_apfd_across_folds=('mean_apfd', 'mean'),
            std_apfd_across_folds=('mean_apfd', 'std'),
            n_folds=('fold', 'count'),
        ).reset_index()
        agg.to_csv(tcv_dir / 'aggregate_per_project.csv', index=False)
        grand = agg['mean_apfd_across_folds'].mean()
        logger.info(f"\nTemporal CV Grand Mean APFD: {grand:.4f}")
        logger.info(f"  {len(agg)} projects, {len(df)} total fold results")

    # Sensitivity — aggregate from incrementally-saved CSV
    sens_csv = output_dir / 'sensitivity' / 'per_variant_per_project.csv'
    if sens_csv.exists():
        sens_dir = output_dir / 'sensitivity'
        df = pd.read_csv(sens_csv)
        agg = df.groupby('variant').agg(
            grand_mean_apfd=('mean_apfd', 'mean'),
            n_projects=('project', 'count'),
        ).reset_index()
        agg.to_csv(sens_dir / 'aggregate_per_variant.csv', index=False)
        logger.info(f"\nSensitivity Summary:")
        for _, row in agg.iterrows():
            logger.info(f"  {row['variant']:25s} APFD={row['grand_mean_apfd']:.4f}")

    # Meta
    with open(output_dir / 'experiment_meta.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config_seed': BASE_CONFIG['seed'],
            'n_ablation': len(ablation_results) if ablation_results else 0,
            'n_temporal': len(temporal_results) if temporal_results else 0,
            'n_sensitivity': len(sensitivity_results) if sensitivity_results else 0,
        }, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', action='store_true')
    parser.add_argument('--temporal', action='store_true')
    parser.add_argument('--sensitivity', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if not any([args.ablation, args.temporal, args.sensitivity, args.all]):
        args.all = True

    print("\n" + "=" * 70)
    print("Filo-Priori V14 — Ablation, Sensitivity & Temporal Validation")
    print("RTPTorrent (20 projects)")
    print("=" * 70 + "\n")

    config = copy.deepcopy(BASE_CONFIG)
    _set_seed(config['seed'])

    logger.info(f"Device: {config['device']}")
    logger.info(f"Loading SBERT encoder: {config['sbert_model']}")
    sbert_config = {
        'embedding': {
            'model_name': config['sbert_model'],
            'batch_size': config['sbert_batch_size'],
        }
    }
    sbert_encoder = SBERTEncoder(sbert_config, device=config['device'])

    total_start = time.time()
    ablation_results, ablation_builds = [], []
    temporal_results = []
    sensitivity_results = []

    if args.all or args.ablation:
        ablation_results, ablation_builds = run_ablation_study(config, sbert_encoder)

    if args.all or args.temporal:
        temporal_results = run_temporal_cv(config, sbert_encoder)

    if args.all or args.sensitivity:
        sensitivity_results = run_sensitivity_analysis(config, sbert_encoder)

    save_results(config['output_dir'], ablation_results, ablation_builds, temporal_results, sensitivity_results)

    total_time = time.time() - total_start
    logger.info(f"\nTotal time: {total_time / 3600:.1f} hours")


if __name__ == '__main__':
    main()
