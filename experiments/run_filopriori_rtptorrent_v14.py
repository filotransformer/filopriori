#!/usr/bin/env python3
"""
Filo-Priori V14 (GATv2 + DeepOrder DNN Ensemble) for RTPTorrent Dataset

KEY IMPROVEMENTS over V13:
1. FIX double-sigmoid bug: DeepOrderNet has sigmoid output but V13 used
   BCEWithLogitsLoss (which applies sigmoid again). Now uses BCELoss.
2. Clamp DNN pos_weight to max_dnn_pos_weight (default 50) to prevent
   extreme class weighting for rare-failure projects.
3. Reduce max_do_train_builds to 5000 (empirically better for SonarSource).
4. DNN epochs increased to 15 for better convergence with corrected loss.

Architecture:
1. Train DeepOrder DNN per project (8 features, trained network, fixed loss)
2. Train GATv2 (full-graph, "ever-failed" labels, 19 structural features)
3. Alpha blending: final = alpha * GNN_P(Fail) + (1-alpha) * DNN_P(Fail)
4. Alpha optimized on validation set APFD

Usage:
    python experiments/run_filopriori_rtptorrent_v14.py
"""

import gc
import json
import logging
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.deeporder import DeepOrderModel
from src.embeddings.sbert_encoder import SBERTEncoder
from src.models.model_factory import create_model
from src.phylogenetic.phylogenetic_graph_builder import PhylogeneticGraphBuilder
from src.preprocessing.structural_feature_extractor_v2_5 import StructuralFeatureExtractorV2_5
from src.preprocessing.priority_score_generator import PriorityScoreGenerator
from src.training.losses import FocalLoss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'data_dir': 'datasets/02_rtptorrent/raw/MSR2',
    'seed': 42,
    'output_dir': 'results/filopriori_rtptorrent_v14',
    'method_name': 'Filo-Priori',
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'device': DEFAULT_DEVICE,
    'max_train_rows': 500_000,

    # SBERT
    'sbert_model': 'sentence-transformers/all-mpnet-base-v2',
    'sbert_batch_size': 64,
    'embedding_dim': 768,

    # Model architecture — original hyperparameters
    'model': {
        'type': 'dual_stream_v8',
        'semantic': {
            'input_dim': 768,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.3,
            'activation': 'gelu',
        },
        'structural': {
            'input_dim': 19,      # 10 V2.5 + 9 DeepOrder formula features
            'hidden_dim': 128,
            'num_heads': 4,
            'dropout': 0.3,
            'activation': 'elu',
            'use_edge_weights': True,
        },
        'fusion': {
            'type': 'cross_attention',
            'hidden_dim': 256,
            'num_heads': 4,
            'dropout': 0.1,
        },
        'classifier': {
            'hidden_dims': [128, 64],
            'dropout': 0.4,
        },
        'num_classes': 2,
    },

    # GATv2 Training
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'max_epochs': 30,
    'patience': 7,
    'focal_gamma': 2.0,
    'max_pos_weight': 10.0,

    # DeepOrder DNN configuration (fixed loss function)
    'do_dnn': {
        'hidden_dims': [64, 32, 16],
        'dropout': 0.2,
        'learning_rate': 0.001,
        'epochs': 15,
        'batch_size': 128,
        'history_window': 10,
    },

    # Cap DNN training builds for large projects (SonarSource has 42K builds)
    # History is built for all builds, but DNN only trains on the last N builds
    'max_do_train_builds': 5000,

    # Max pos_weight for DNN loss — prevents extreme weighting for rare failures
    'max_dnn_pos_weight': 50.0,

    # Graph — co_failure only (simpler, proven)
    'graph_type': 'co_failure',
    'min_co_occurrences': 2,
    'weight_threshold': 0.1,

    # Alpha blending (max 0.9 — never fully ignore DNN to prevent catastrophic failures)
    'alpha_search_range': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'min_val_failure_builds': 3,  # Default to alpha=0.0 if fewer validation failure builds

    # Small project fallback
    'min_tcs_for_model': 30,
}

SKIP_DIRS = {'repo'}


# =============================================================================
# APFD CALCULATION
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
    projects = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and d.name not in SKIP_DIRS:
            csv_file = d / f"{d.name}.csv"
            if csv_file.exists():
                projects.append(d)
    return projects


def load_project_data(project_dir: Path) -> pd.DataFrame:
    csv_path = project_dir / f"{project_dir.name}.csv"
    df = pd.read_csv(csv_path)
    df['is_failure'] = ((df['failures'] > 0) | (df['errors'] > 0)).astype(int)
    return df


# =============================================================================
# SEMANTIC TEXT GENERATION
# =============================================================================

def generate_semantic_text(fqn: str) -> str:
    """Convert FQN to simple semantic text (CamelCase split)."""
    parts = fqn.split('.')
    tokens = []
    for part in parts:
        split = re.sub(r'([A-Z])', r' \1', part).strip().lower().split()
        tokens.extend(split)
    return ' '.join(tokens)


# =============================================================================
# DNN SCORE EXTRACTION
# =============================================================================

def _get_dnn_scores(do_model: DeepOrderModel, test_ids: List[str], device: str) -> np.ndarray:
    """Get DeepOrder DNN P(Fail) predictions for test cases."""
    if do_model.model is None:
        # Fallback: return 0.5 for all
        return np.full(len(test_ids), 0.5)

    do_model.model.eval()
    features_list = []
    for tc in test_ids:
        features = do_model.feature_extractor.extract_features(tc)
        features_list.append(features)

    X = np.array(features_list)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = do_model.model(X_tensor).cpu().numpy()

    return predictions


# =============================================================================
# VALIDATION-OPTIMIZED ALPHA (blends GNN with DNN instead of formula)
# =============================================================================

def _optimize_alpha(
    gnn_model, do_model, val_builds, val_df, tc_to_embedding,
    graph_builder, priority_gen, tc_exec_history, feat_extractor,
    device, config
):
    """Find optimal blending alpha on validation set APFD.
    Blends GNN P(Fail) with DeepOrder DNN P(Fail).
    """
    embedding_dim = config['embedding_dim']
    zero_sem = np.zeros(embedding_dim, dtype=np.float32)

    if gnn_model is not None:
        gnn_model.eval()

    build_predictions = []

    for build_id in val_builds:
        build_rows = val_df[val_df['travisJobId'] == build_id]
        verdicts = build_rows.groupby('testName')['is_failure'].max().to_dict()
        if not any(verdicts.values()):
            continue

        tcs = list(verdicts.keys())

        # DeepOrder DNN scores (instead of formula)
        dnn_scores = _get_dnn_scores(do_model, tcs, device)

        # GNN model scores
        if gnn_model is not None:
            sem = np.array([tc_to_embedding.get(tc, zero_sem) for tc in tcs])

            rows = []
            for tc in tcs:
                r = build_rows[build_rows['testName'] == tc].iloc[0]
                rows.append(r)
            bdf = pd.DataFrame(rows)
            bdf['Build_ID'] = str(build_id)
            bdf['TC_Key'] = bdf['testName']
            bdf['TE_Test_Result'] = bdf['is_failure'].apply(
                lambda x: 'Fail' if x == 1 else 'Pass'
            )
            v25 = feat_extractor.transform(bdf, is_test=True)
            do_feats = np.array([
                priority_gen.extract_deeporder_features(tc_exec_history, tc)
                for tc in tcs
            ], dtype=np.float32)
            struct = np.concatenate([v25, do_feats], axis=1)

            edge_idx, edge_w = graph_builder.get_edge_index_and_weights(tcs, return_torch=True)
            if edge_idx.shape[1] == 0:
                n = len(tcs)
                sl = torch.arange(n, dtype=torch.long)
                edge_idx = torch.stack([sl, sl], dim=0)
                edge_w = torch.ones(n, dtype=torch.float32)

            with torch.no_grad():
                sem_t = torch.from_numpy(sem).float().to(device)
                struct_t = torch.from_numpy(struct).float().to(device)
                logits = gnn_model(sem_t, struct_t, edge_idx.to(device), edge_w.to(device))
                gnn_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        else:
            gnn_probs = dnn_scores

        build_predictions.append((tcs, gnn_probs, dnn_scores, verdicts))

    if not build_predictions:
        return 0.0, 0.0  # Default to pure DNN

    # If too few validation failure builds, alpha optimization is unreliable
    min_val = config.get('min_val_failure_builds', 3)
    if len(build_predictions) < min_val:
        logger.info(f"    Only {len(build_predictions)} val failure builds (< {min_val}), "
                    f"defaulting to alpha=0.0 (pure DNN)")
        return 0.0, 0.0

    best_alpha = 0.0
    best_apfd = 0.0

    for alpha_test in config['alpha_search_range']:
        apfd_list = []
        for tcs, gnn_probs, dnn_scores, verdicts in build_predictions:
            blended = alpha_test * gnn_probs + (1 - alpha_test) * dnn_scores
            sorted_indices = np.argsort(-blended)
            ranking = [tcs[i] for i in sorted_indices]
            labels = np.array([verdicts[tc] for tc in ranking])
            ranks = np.arange(1, len(ranking) + 1)
            apfd = calculate_apfd_single_build(ranks, labels)
            if apfd is not None:
                apfd_list.append(apfd)

        mean_apfd = float(np.mean(apfd_list)) if apfd_list else 0.0
        if mean_apfd > best_apfd:
            best_apfd = mean_apfd
            best_alpha = alpha_test

    return best_alpha, best_apfd


# =============================================================================
# PER-PROJECT EXPERIMENT
# =============================================================================

def run_project(project_dir: Path, config: Dict, sbert_encoder: SBERTEncoder) -> Optional[Dict]:
    project_name = project_dir.name
    logger.info(f"\n{'='*60}")
    logger.info(f"Project: {project_name}")
    logger.info(f"{'='*60}")

    device = config['device']
    embedding_dim = config['embedding_dim']

    # -------------------------------------------------------------------------
    # 1. Load data and temporal split
    # -------------------------------------------------------------------------
    try:
        df = load_project_data(project_dir)
    except Exception as e:
        logger.warning(f"Failed to load {project_name}: {e}")
        return None

    builds = df['travisJobId'].unique().tolist()
    n_builds = len(builds)

    if n_builds < 5:
        logger.warning(f"Skipping {project_name}: only {n_builds} builds")
        return None

    train_idx = int(n_builds * config['train_ratio'])
    train_builds = builds[:train_idx]
    test_builds = builds[train_idx:]

    logger.info(f"  Total builds: {n_builds}, Train: {len(train_builds)}, Test: {len(test_builds)}")

    df['Build_ID'] = df['travisJobId'].astype(str)
    df['TC_Key'] = df['testName']
    df['TE_Test_Result'] = df['is_failure'].apply(lambda x: 'Fail' if x == 1 else 'Pass')

    train_df = df[df['travisJobId'].isin(train_builds)].copy()
    test_df = df[df['travisJobId'].isin(test_builds)].copy()
    del df
    gc.collect()

    n_train_failures = (train_df['is_failure'] == 1).sum()
    if n_train_failures < 2:
        logger.warning(f"Skipping {project_name}: only {n_train_failures} failures in training")
        return None

    # -------------------------------------------------------------------------
    # 2. Train/Val split
    # -------------------------------------------------------------------------
    inner_train_builds = train_builds[:int(len(train_builds) * (1 - config['val_ratio']))]
    val_builds_list = train_builds[int(len(train_builds) * (1 - config['val_ratio'])):]

    train_inner_df = train_df[train_df['travisJobId'].isin(inner_train_builds)].copy()
    val_df_split = train_df[train_df['travisJobId'].isin(val_builds_list)].copy()

    logger.info(f"  Train inner: {len(train_inner_df)} rows, Val: {len(val_df_split)} rows")

    if len(train_inner_df) > config['max_train_rows']:
        logger.info(f"  Capping training from {len(train_inner_df)} to {config['max_train_rows']} rows")
        train_inner_df = train_inner_df.sample(n=config['max_train_rows'], random_state=config['seed'])

    # -------------------------------------------------------------------------
    # 3. Train DeepOrder DNN (same approach as baseline)
    # -------------------------------------------------------------------------
    logger.info("  Training DeepOrder DNN...")
    do_cfg = config['do_dnn']
    do_model = DeepOrderModel(
        hidden_dims=do_cfg['hidden_dims'],
        dropout=do_cfg['dropout'],
        learning_rate=do_cfg['learning_rate'],
        epochs=do_cfg['epochs'],
        batch_size=do_cfg['batch_size'],
        history_window=do_cfg['history_window'],
        device=device,
    )

    # Unified DNN training: pre-warm history (if needed), then train with corrected BCELoss
    # FIX: DeepOrderNet has sigmoid output, so we must use BCELoss (not BCEWithLogitsLoss)
    max_do_builds = config.get('max_do_train_builds', len(train_builds))

    from src.baselines.deeporder import (
        DeepOrderFeatureExtractor, DeepOrderNet, DeepOrderDataset
    )
    from torch.utils.data import DataLoader
    import torch.optim as optim

    if len(train_builds) > max_do_builds:

        logger.info(f"  Pre-warming DNN history with all {len(train_builds)} builds, "
                    f"then training on last {max_do_builds}...")

        # Phase 1: Build full history by iterating through ALL training builds
        # Only update_history — no feature extraction (fast)
        all_train_builds_str = train_df['Build_ID'].unique().tolist()
        train_grouped = train_df.groupby('Build_ID', sort=False)
        fe = DeepOrderFeatureExtractor(history_window=do_cfg['history_window'])

        n_warmup = len(all_train_builds_str) - max_do_builds
        for bid in all_train_builds_str[:n_warmup]:
            if bid not in train_grouped.groups:
                continue
            bdf = train_grouped.get_group(bid)
            tc_arr = bdf['TC_Key'].values
            fail_arr = bdf['is_failure'].values
            dur_arr = bdf['duration'].values if 'duration' in bdf.columns else np.ones(len(bdf))
            test_results = {}
            for i in range(len(tc_arr)):
                test_results[tc_arr[i]] = (int(fail_arr[i]), float(dur_arr[i]))
            fe.update_history(bid, test_results)

        logger.info(f"  Pre-warmed {n_warmup} builds ({fe.n_builds} history entries)")

        # Phase 2: Extract features + labels from last N builds (with full history)
        features_list = []
        labels_list = []
        for bid in all_train_builds_str[n_warmup:]:
            if bid not in train_grouped.groups:
                continue
            bdf = train_grouped.get_group(bid)
            test_ids = bdf['TC_Key'].values
            result_vals = bdf['TE_Test_Result'].values
            dur_vals = bdf['duration'].values if 'duration' in bdf.columns else np.ones(len(bdf))

            for i in range(len(test_ids)):
                features = fe.extract_features(test_ids[i])
                features_list.append(features)
                verdict = 1 if str(result_vals[i]).upper() != 'PASS' else 0
                labels_list.append(verdict)

            test_results = {}
            for i in range(len(test_ids)):
                verdict = 1 if str(result_vals[i]).upper() != 'PASS' else 0
                test_results[test_ids[i]] = (verdict, float(dur_vals[i]))
            fe.update_history(bid, test_results)

        X = np.array(features_list)
        y = np.array(labels_list)
        del features_list, labels_list, train_grouped
        gc.collect()

        logger.info(f"  DNN training data: {len(X)} samples, failure_rate={y.mean():.4f}")

        # Phase 3: Train DNN on extracted features
        do_model.feature_extractor = fe  # Use the pre-warmed extractor
        do_model.model = DeepOrderNet(
            input_dim=X.shape[1],
            hidden_dims=do_cfg['hidden_dims'],
            dropout=do_cfg['dropout'],
        ).to(device)

        dataset = DeepOrderDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=do_cfg['batch_size'], shuffle=True)

        # FIX: DeepOrderNet already has sigmoid output → use BCELoss, NOT BCEWithLogitsLoss
        # BCEWithLogitsLoss applies sigmoid internally, causing double-sigmoid: sigmoid(sigmoid(x))
        raw_pw = (1 - y.mean()) / (y.mean() + 1e-6)
        max_pw = config.get('max_dnn_pos_weight', 50.0)
        clamped_pw = min(raw_pw, max_pw)
        logger.info(f"  DNN pos_weight: raw={raw_pw:.1f}, clamped={clamped_pw:.1f}")
        pos_weight_val = torch.tensor(clamped_pw, dtype=torch.float32).to(device)
        criterion = nn.BCELoss(reduction='none')
        optimizer = optim.Adam(do_model.model.parameters(), lr=do_cfg['learning_rate'])

        do_model.model.train()
        for epoch in range(do_cfg['epochs']):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                outputs = do_model.model(batch_X)
                per_sample_loss = criterion(outputs, batch_y)
                # Apply pos_weight manually: weight positive samples higher
                weights = torch.where(batch_y == 1, pos_weight_val, torch.ones_like(batch_y))
                loss = (per_sample_loss * weights).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                logger.info(f"    DNN Epoch {epoch+1}/{do_cfg['epochs']}, "
                            f"Loss={total_loss/len(dataloader):.4f}")

        del X, y, dataset, dataloader
        gc.collect()
    else:
        # Small project: no pre-warming needed, iterate all builds
        logger.info(f"  Training DNN on all {len(train_builds)} builds (no pre-warm needed)...")

        all_train_builds_str = train_df['Build_ID'].unique().tolist()
        train_grouped = train_df.groupby('Build_ID', sort=False)
        fe = DeepOrderFeatureExtractor(history_window=do_cfg['history_window'])

        features_list = []
        labels_list = []
        for bid in all_train_builds_str:
            if bid not in train_grouped.groups:
                continue
            bdf = train_grouped.get_group(bid)
            test_ids = bdf['TC_Key'].values
            result_vals = bdf['TE_Test_Result'].values
            dur_vals = bdf['duration'].values if 'duration' in bdf.columns else np.ones(len(bdf))

            for i in range(len(test_ids)):
                features = fe.extract_features(test_ids[i])
                features_list.append(features)
                verdict = 1 if str(result_vals[i]).upper() != 'PASS' else 0
                labels_list.append(verdict)

            test_results = {}
            for i in range(len(test_ids)):
                verdict = 1 if str(result_vals[i]).upper() != 'PASS' else 0
                test_results[test_ids[i]] = (verdict, float(dur_vals[i]))
            fe.update_history(bid, test_results)

        X = np.array(features_list)
        y = np.array(labels_list)
        del features_list, labels_list, train_grouped
        gc.collect()

        logger.info(f"  DNN training data: {len(X)} samples, failure_rate={y.mean():.4f}")

        do_model.feature_extractor = fe
        do_model.model = DeepOrderNet(
            input_dim=X.shape[1],
            hidden_dims=do_cfg['hidden_dims'],
            dropout=do_cfg['dropout'],
        ).to(device)

        dataset = DeepOrderDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=do_cfg['batch_size'], shuffle=True)

        raw_pw = (1 - y.mean()) / (y.mean() + 1e-6)
        max_pw = config.get('max_dnn_pos_weight', 50.0)
        clamped_pw = min(raw_pw, max_pw)
        logger.info(f"  DNN pos_weight: raw={raw_pw:.1f}, clamped={clamped_pw:.1f}")
        pos_weight_val = torch.tensor(clamped_pw, dtype=torch.float32).to(device)
        criterion = nn.BCELoss(reduction='none')
        optimizer = optim.Adam(do_model.model.parameters(), lr=do_cfg['learning_rate'])

        do_model.model.train()
        for epoch in range(do_cfg['epochs']):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                outputs = do_model.model(batch_X)
                per_sample_loss = criterion(outputs, batch_y)
                weights = torch.where(batch_y == 1, pos_weight_val, torch.ones_like(batch_y))
                loss = (per_sample_loss * weights).mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 5 == 0:
                logger.info(f"    DNN Epoch {epoch+1}/{do_cfg['epochs']}, "
                            f"Loss={total_loss/len(dataloader):.4f}")

        del X, y, dataset, dataloader
        gc.collect()

    logger.info("  DeepOrder DNN trained successfully")

    # -------------------------------------------------------------------------
    # 4. SBERT embeddings
    # -------------------------------------------------------------------------
    all_tc_keys = list(set(
        train_inner_df['TC_Key'].unique().tolist() +
        val_df_split['TC_Key'].unique().tolist() +
        test_df['TC_Key'].unique().tolist()
    ))
    n_unique_tcs = len(all_tc_keys)

    logger.info(f"  Generating SBERT embeddings for {n_unique_tcs} unique test cases...")
    semantic_texts = [generate_semantic_text(tc) for tc in all_tc_keys]
    tc_embeddings = sbert_encoder.encode_texts_batch(semantic_texts)
    tc_to_embedding = {tc: tc_embeddings[i] for i, tc in enumerate(all_tc_keys)}

    # -------------------------------------------------------------------------
    # 5. Structural features (V2.5 + DeepOrder formula features for GNN input)
    # -------------------------------------------------------------------------
    logger.info("  Extracting V2.5 structural features (10 features)...")
    feat_extractor = StructuralFeatureExtractorV2_5(
        recent_window=5, very_recent_window=2, medium_term_window=10,
        min_history=2, verbose=False
    )
    feat_extractor.fit(train_df)

    logger.info("  Computing DeepOrder priority features (9 features for GNN input)...")
    priority_gen = PriorityScoreGenerator(
        num_cycles=10, decay_type='exponential', decay_factor=0.8
    )
    _, tc_exec_history = priority_gen.compute_priorities_for_dataframe(
        train_df, build_col='Build_ID', tc_col='TC_Key',
        result_col='TE_Test_Result', fail_value='Fail', pass_value='Pass'
    )

    # -------------------------------------------------------------------------
    # 6. Build graph (co_failure)
    # -------------------------------------------------------------------------
    logger.info("  Building co-failure graph...")
    graph_builder = PhylogeneticGraphBuilder(
        graph_type=config['graph_type'],
        min_co_occurrences=config['min_co_occurrences'],
        weight_threshold=config['weight_threshold'],
        verbose=False,
    )
    graph_builder.fit(train_df)

    stats = graph_builder.get_graph_statistics()
    logger.info(f"  Graph: {stats['num_nodes']} nodes, {stats['num_edges']} edges, "
                f"avg_degree={stats['avg_degree']:.1f}")

    del tc_embeddings, semantic_texts
    gc.collect()

    # -------------------------------------------------------------------------
    # 7. FULL-GRAPH GATv2 TRAINING
    # -------------------------------------------------------------------------
    use_model = n_unique_tcs >= config['min_tcs_for_model']
    if not use_model:
        logger.info(f"  SMALL PROJECT ({n_unique_tcs} TCs < {config['min_tcs_for_model']}): "
                    f"using pure DeepOrder DNN")

    train_time = 0.0
    gnn_model = None

    if use_model:
        train_tc_keys = train_inner_df['TC_Key'].unique().tolist()
        val_tc_keys = val_df_split['TC_Key'].unique().tolist()
        n_train_nodes = len(train_tc_keys)
        n_val_nodes = len(val_tc_keys)

        logger.info(f"  Node-level: {n_train_nodes} train nodes, {n_val_nodes} val nodes")

        # Per-node semantic embeddings
        zero_sem = np.zeros(embedding_dim, dtype=np.float32)
        train_node_semantic = np.array([tc_to_embedding.get(tc, zero_sem) for tc in train_tc_keys])
        val_node_semantic = np.array([tc_to_embedding.get(tc, zero_sem) for tc in val_tc_keys])

        # Per-node structural features: V2.5 (10) + DeepOrder (9) = 19
        def _aggregate_struct_per_node(df_subset, tc_list, is_test):
            last_rows = df_subset.drop_duplicates(subset='TC_Key', keep='last')
            last_rows = last_rows.set_index('TC_Key')
            ordered_rows = []
            for tc in tc_list:
                if tc in last_rows.index:
                    ordered_rows.append(last_rows.loc[tc])
                else:
                    ordered_rows.append(last_rows.iloc[0])
            ordered_df = pd.DataFrame(ordered_rows)
            ordered_df['TC_Key'] = tc_list
            ordered_df['Build_ID'] = ordered_df['Build_ID'].astype(str)
            v25_features = feat_extractor.transform(ordered_df, is_test=is_test)
            deeporder_features = np.array([
                priority_gen.extract_deeporder_features(tc_exec_history, tc)
                for tc in tc_list
            ], dtype=np.float32)
            return np.concatenate([v25_features, deeporder_features], axis=1)

        train_node_struct = _aggregate_struct_per_node(train_inner_df, train_tc_keys, is_test=False)
        val_node_struct = _aggregate_struct_per_node(val_df_split, val_tc_keys, is_test=True)

        logger.info(f"  Structural features: train={train_node_struct.shape}, val={val_node_struct.shape}")

        # Per-node labels: 1 if TC ever failed, 0 otherwise
        train_failure_by_tc = train_inner_df.groupby('TC_Key')['is_failure'].max()
        train_node_labels = np.array([
            int(train_failure_by_tc.get(tc, 0)) for tc in train_tc_keys
        ], dtype=np.int64)

        val_failure_by_tc = val_df_split.groupby('TC_Key')['is_failure'].max()
        val_node_labels = np.array([
            int(val_failure_by_tc.get(tc, 0)) for tc in val_tc_keys
        ], dtype=np.int64)

        # Graph edges
        edge_index, edge_weights = graph_builder.get_edge_index_and_weights(
            train_tc_keys, return_torch=True
        )
        if edge_index.shape[1] == 0:
            self_loops = torch.arange(n_train_nodes, dtype=torch.long)
            edge_index = torch.stack([self_loops, self_loops], dim=0)
            edge_weights = torch.ones(n_train_nodes, dtype=torch.float32)

        val_edge_index, val_edge_weights = graph_builder.get_edge_index_and_weights(
            val_tc_keys, return_torch=True
        )
        if val_edge_index.shape[1] == 0:
            self_loops = torch.arange(n_val_nodes, dtype=torch.long)
            val_edge_index = torch.stack([self_loops, self_loops], dim=0)
            val_edge_weights = torch.ones(n_val_nodes, dtype=torch.float32)

        # Convert to tensors
        train_sem_t = torch.from_numpy(train_node_semantic).float().to(device)
        train_struct_t = torch.from_numpy(train_node_struct).float().to(device)
        train_labels_t = torch.from_numpy(train_node_labels).long().to(device)
        edge_index = edge_index.to(device)
        edge_weights = edge_weights.to(device)

        val_sem_t = torch.from_numpy(val_node_semantic).float().to(device)
        val_struct_t = torch.from_numpy(val_node_struct).float().to(device)
        val_labels_t = torch.from_numpy(val_node_labels).long().to(device)
        val_edge_index = val_edge_index.to(device)
        val_edge_weights = val_edge_weights.to(device)

        class_counts = np.bincount(train_node_labels, minlength=2)

        del train_node_semantic, val_node_semantic
        del train_node_struct, val_node_struct
        del train_node_labels, val_node_labels
        gc.collect()

        # Initialize GNN model
        gnn_model = create_model(config['model'])
        gnn_model = gnn_model.to(device)

        if class_counts[1] > 0:
            pos_weight = class_counts[0] / class_counts[1]
            alpha_weights = torch.tensor(
                [1.0, min(pos_weight, config['max_pos_weight'])],
                dtype=torch.float32
            ).to(device)
        else:
            alpha_weights = None

        criterion = FocalLoss(alpha=alpha_weights, gamma=config['focal_gamma'])
        optimizer = torch.optim.AdamW(
            gnn_model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # --- FULL-GRAPH TRAINING LOOP ---
        logger.info(f"  GNN Training: max {config['max_epochs']} epochs, patience={config['patience']}, "
                    f"{n_train_nodes} nodes ({class_counts[1]} failures)")
        train_start = time.time()

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(config['max_epochs']):
            gnn_model.train()
            optimizer.zero_grad()
            logits = gnn_model(train_sem_t, train_struct_t, edge_index, edge_weights)
            loss = criterion(logits, train_labels_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn_model.parameters(), 1.0)
            optimizer.step()
            train_loss = loss.item()

            gnn_model.eval()
            with torch.no_grad():
                val_logits = gnn_model(val_sem_t, val_struct_t, val_edge_index, val_edge_weights)
                val_loss = criterion(val_logits, val_labels_t).item()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"    Epoch {epoch+1}/{config['max_epochs']}: "
                            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in gnn_model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    logger.info(f"    Early stopping at epoch {epoch+1}")
                    break

        train_time = time.time() - train_start

        if best_state is not None:
            gnn_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        del best_state
        del train_sem_t, train_struct_t, train_labels_t
        del val_sem_t, val_struct_t, val_labels_t
        del val_edge_index, val_edge_weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"  GNN Training completed in {train_time:.1f}s")

        # --- Optimize alpha on validation (GNN vs DNN) ---
        logger.info("  Optimizing blending alpha (GNN vs DeepOrder DNN)...")
        best_alpha, best_val_apfd = _optimize_alpha(
            gnn_model, do_model, val_builds_list, val_df_split, tc_to_embedding,
            graph_builder, priority_gen, tc_exec_history, feat_extractor,
            device, config
        )
        logger.info(f"  Optimal alpha={best_alpha:.1f} (val APFD={best_val_apfd:.4f})")
    else:
        best_alpha = 0.0  # Pure DNN

    # -------------------------------------------------------------------------
    # 8. Evaluation: per-build APFD
    # -------------------------------------------------------------------------
    eval_start = time.time()
    if gnn_model is not None:
        gnn_model.eval()

    test_grouped = test_df.groupby('travisJobId')
    zero_sem = np.zeros(embedding_dim, dtype=np.float32)

    build_results = []
    all_apfd_scores = []
    n_test = len(test_builds)
    log_interval = max(1, n_test // 20)

    alpha_model = best_alpha
    logger.info(f"  Evaluating with alpha_model={alpha_model:.1f}")

    for i, build_id in enumerate(test_builds):
        if build_id not in test_grouped.groups:
            continue

        build_df = test_grouped.get_group(build_id)
        verdicts = build_df.groupby('testName')['is_failure'].max().to_dict()
        durations = build_df.groupby('testName')['duration'].last().to_dict()
        test_ids = list(verdicts.keys())
        n_failures = sum(verdicts.values())

        if n_failures == 0:
            # Still update DNN history for builds without failures
            test_results = {tc: (verdicts[tc], durations.get(tc, 1.0)) for tc in test_ids}
            do_model.update_history(str(build_id), test_results)
            # Also update formula history for GNN features
            for tc in test_ids:
                status = 1 if verdicts[tc] == 1 else 0
                if tc not in tc_exec_history:
                    tc_exec_history[tc] = []
                tc_exec_history[tc].append(status)
            continue

        # DeepOrder DNN scores (TRAINED model, not formula)
        dnn_scores = _get_dnn_scores(do_model, test_ids, device)

        # GNN model scores (if using model and alpha > 0)
        if gnn_model is not None and alpha_model > 0:
            build_semantic = np.array([
                tc_to_embedding.get(tc, zero_sem) for tc in test_ids
            ])

            build_rows = []
            for tc in test_ids:
                row_data = build_df[build_df['testName'] == tc].iloc[0]
                build_rows.append(row_data)
            build_test_df = pd.DataFrame(build_rows)
            build_test_df['Build_ID'] = str(build_id)
            build_test_df['TC_Key'] = build_test_df['testName']
            build_test_df['TE_Test_Result'] = build_test_df['is_failure'].apply(
                lambda x: 'Fail' if x == 1 else 'Pass'
            )
            v25_features = feat_extractor.transform(build_test_df, is_test=True)
            deeporder_features = np.array([
                priority_gen.extract_deeporder_features(tc_exec_history, tc)
                for tc in test_ids
            ], dtype=np.float32)
            build_struct = np.concatenate([v25_features, deeporder_features], axis=1)

            build_edge_index, build_edge_weights = graph_builder.get_edge_index_and_weights(
                test_ids, return_torch=True
            )
            if build_edge_index.shape[1] == 0:
                n_nodes = len(test_ids)
                self_loops = torch.arange(n_nodes, dtype=torch.long)
                build_edge_index = torch.stack([self_loops, self_loops], dim=0)
                build_edge_weights = torch.ones(n_nodes, dtype=torch.float32)

            with torch.no_grad():
                sem_t = torch.from_numpy(build_semantic).float().to(device)
                struct_t = torch.from_numpy(build_struct).float().to(device)
                logits = gnn_model(sem_t, struct_t,
                                 build_edge_index.to(device),
                                 build_edge_weights.to(device))
                gnn_probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

            final_scores = alpha_model * gnn_probs + (1 - alpha_model) * dnn_scores
        else:
            final_scores = dnn_scores

        # Rank by final score descending
        sorted_indices = np.argsort(-final_scores)
        ranking = [test_ids[idx] for idx in sorted_indices]
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
                'time': 0.0,
            })

        # Online history updates
        # 1. DNN feature extractor
        test_results = {tc: (verdicts[tc], durations.get(tc, 1.0)) for tc in test_ids}
        do_model.update_history(str(build_id), test_results)
        # 2. Formula-based history (for GNN structural features)
        for tc in test_ids:
            status = 1 if verdicts[tc] == 1 else 0
            if tc not in tc_exec_history:
                tc_exec_history[tc] = []
            tc_exec_history[tc].append(status)

        if (i + 1) % log_interval == 0 or (i + 1) == n_test:
            elapsed = time.time() - eval_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n_test - i - 1) / rate if rate > 0 else 0
            current_mean = np.mean(all_apfd_scores) if all_apfd_scores else 0
            logger.info(f"  Eval progress: {i+1}/{n_test} builds "
                        f"({len(all_apfd_scores)} with failures, mean={current_mean:.4f}) "
                        f"[{elapsed:.0f}s elapsed, ETA {eta:.0f}s]")

    eval_time = time.time() - eval_start

    # Cleanup
    if gnn_model is not None:
        del gnn_model
    del do_model
    del test_df, test_grouped
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
        'alpha_model': best_alpha,
    }

    logger.info(f"  Builds with failures: {len(all_apfd_scores)}")
    logger.info(f"  Mean APFD: {project_result['mean_apfd']:.4f} (alpha={best_alpha:.1f})")

    return project_result


# =============================================================================
# RESULTS SAVING
# =============================================================================

def _save_results(output_dir, all_project_results, all_build_results, total_start):
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
            'alpha_model': r.get('alpha_model', 0.0),
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
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in CONFIG.items()},
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
        f.write("Filo-Priori V14 (GATv2 + DeepOrder DNN) - RTPTorrent Results\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Projects analyzed: {aggregate['n_projects']}\n")
        f.write(f"Total builds with failures: {aggregate['n_builds_total']}\n\n")
        f.write(f"Grand Mean APFD: {aggregate['grand_mean_apfd']:.4f} (PRIMARY METRIC)\n")
        f.write(f"Grand Std APFD:  {aggregate['grand_std_apfd']:.4f}\n\n")
        f.write(f"Per-project results:\n")
        for r in all_project_results:
            f.write(f"  {r['project']:40s} APFD={r['mean_apfd']:.4f} "
                    f"(n={r['n_builds_with_failures']}, alpha={r.get('alpha_model', 0):.1f})\n")
        f.write(f"\nTotal time: {total_time:.2f}s\n")
        f.write("=" * 70 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("Filo-Priori V14 (GATv2 + DeepOrder DNN Ensemble)")
    print("Full-graph GNN + Trained DeepOrder DNN blending")
    print("=" * 70 + "\n")

    np.random.seed(CONFIG['seed'])
    random.seed(CONFIG['seed'])
    torch.manual_seed(CONFIG['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG['seed'])

    data_dir = PROJECT_ROOT / CONFIG['data_dir']
    output_dir = PROJECT_ROOT / CONFIG['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading SBERT encoder...")
    sbert_config = {
        'semantic': {
            'model_name': CONFIG['sbert_model'],
            'batch_size': CONFIG['sbert_batch_size'],
        }
    }
    sbert_encoder = SBERTEncoder(sbert_config, device=CONFIG['device'])
    logger.info(f"SBERT loaded: dim={sbert_encoder.get_embedding_dim()}")

    projects = get_project_dirs(data_dir)
    logger.info(f"Found {len(projects)} projects")

    total_start = time.time()
    all_project_results = []
    all_build_results = []

    for proj_idx, project_dir in enumerate(projects, 1):
        logger.info(f"\n[{proj_idx}/{len(projects)}] Starting {project_dir.name}...")

        try:
            result = run_project(project_dir, CONFIG, sbert_encoder)
        except Exception as e:
            logger.error(f"FAILED {project_dir.name}: {e}")
            import traceback
            traceback.print_exc()
            result = None

        if result is not None:
            all_project_results.append(result)
            all_build_results.extend(result['build_results'])
            _save_results(output_dir, all_project_results, all_build_results, total_start)

            running_apfd = [r['mean_apfd'] for r in all_project_results]
            logger.info(f"  Running grand mean APFD: {np.mean(running_apfd):.4f} "
                        f"({len(all_project_results)} projects)")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - total_start
    _save_results(output_dir, all_project_results, all_build_results, total_start)

    print("\n" + "=" * 70)
    print("Filo-Priori V14 - RTPTorrent Results Summary")
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
                  f"alpha={r.get('alpha_model', 0):.1f}, "
                  f"train={r['train_time']:.0f}s, eval={r['eval_time']:.0f}s)")

    print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Results saved to: {CONFIG['output_dir']}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
