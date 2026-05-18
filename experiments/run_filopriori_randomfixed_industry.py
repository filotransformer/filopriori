#!/usr/bin/env python3
"""
Filo-Priori with Random-Fixed Embeddings -- Industrial Dataset

This experiment tests the "identity proxy" hypothesis on Filo-Priori's semantic
stream. Instead of SBERT embeddings, we replace them with random but consistent
vectors (hash-based, so each TC_Key and commit text always maps to the same vector).

If APFD remains similar to the SBERT baseline (0.7611), it confirms that SBERT
embeddings do not provide meaningful semantic information to Filo-Priori (consistent
with the ablation finding that the semantic stream is not significant, p=0.309).

If APFD drops significantly, it would indicate that SBERT content matters for
Filo-Priori's performance on the industrial dataset (unlikely given ablation).

Comparison target:
    SBERT baseline (experiment_industry_optimized_v3): APFD = 0.7611

Usage:
    python experiments/run_filopriori_randomfixed_industry.py

Results saved to: results/filopriori_randomfixed_industry/

Author: Random-Fixed Embedding Experiment
Date: April 2026
"""

# CRITICAL: Set environment variables BEFORE importing torch/CUDA libraries
import os
os.environ["PYTORCH_NO_NVML"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import hashlib
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch_geometric.utils import subgraph

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT))

# Import Filo-Priori modules
from preprocessing.data_loader import DataLoader
from preprocessing.structural_feature_extractor_v2_5 import StructuralFeatureExtractorV2_5
from preprocessing.structural_feature_imputation import impute_structural_features
from preprocessing.structural_coldstart_regressor import create_coldstart_regressor
from preprocessing.priority_score_generator import create_priority_score_generator
from phylogenetic.phylogenetic_graph_builder import build_phylogenetic_graph
from models.model_factory import create_model
from models.dual_head_model import DualHeadLoss, create_dual_head_loss
from training.losses import create_loss_function
from evaluation.metrics import compute_metrics
from evaluation.apfd import (
    generate_apfd_report,
    print_apfd_summary,
    generate_prioritized_csv,
)
from evaluation.orphan_ranker import compute_orphan_scores

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'experiment_industry_optimized_v3.yaml'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'filopriori_randomfixed_industry'
SEED = 42
SBERT_BASELINE_APFD = 0.7611


# =============================================================================
# Random-Fixed Embedding Generation
# =============================================================================

def generate_random_fixed_embedding(text: str, dim: int = 768) -> np.ndarray:
    """Generate a deterministic random vector for an arbitrary string.

    Uses the MD5 hash of the text as a seed for a numpy RandomState, ensuring
    the same text always produces the same vector across runs.

    The resulting vector is L2-normalised, mimicking the unit-norm property of
    SBERT embeddings.

    Args:
        text: Any string (TC description, commit message, etc.)
        dim: Dimensionality of the output vector.

    Returns:
        A float32 numpy array of shape (dim,).
    """
    seed = int(hashlib.md5(text.encode('utf-8', errors='replace')).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec


def generate_random_fixed_embeddings_batch(texts: list, dim: int = 768) -> np.ndarray:
    """Generate random-fixed embeddings for a list of texts.

    Args:
        texts: List of N strings.
        dim: Embedding dimensionality.

    Returns:
        numpy array of shape (N, dim), dtype float32.
    """
    embeddings = np.empty((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        embeddings[i] = generate_random_fixed_embedding(text, dim)
    return embeddings


# =============================================================================
# Text preparation helpers (mirrors EmbeddingManager logic)
# =============================================================================

def prepare_tc_texts(df: pd.DataFrame) -> list:
    """Prepare test case texts from dataframe (same as EmbeddingManager)."""
    texts = []
    for _, row in df.iterrows():
        summary = row.get('TE_Summary', row.get('tc_summary', row.get('summary', '')))
        steps = row.get('TC_Steps', row.get('tc_steps', row.get('steps', '')))
        if summary and steps:
            text = f"Summary: {summary}\nSteps: {steps}"
        elif summary:
            text = f"Summary: {summary}"
        elif steps:
            text = f"Steps: {steps}"
        else:
            text = "No test case information"
        texts.append(text)
    return texts


def prepare_commit_texts(df: pd.DataFrame) -> list:
    """Prepare commit texts from dataframe (same as EmbeddingManager)."""
    texts = []
    for _, row in df.iterrows():
        msg = row.get('commit_processed', row.get('commit_msg', row.get('message', '')))
        diff = row.get('commit_diff', row.get('diff', ''))
        if msg and diff:
            diff_truncated = diff[:2000] if len(diff) > 2000 else diff
            text = f"Commit Message: {msg}\n\nDiff:\n{diff_truncated}"
        elif msg:
            text = f"Commit Message: {msg}"
        elif diff:
            diff_truncated = diff[:2000] if len(diff) > 2000 else diff
            text = f"Diff:\n{diff_truncated}"
        else:
            text = "No commit information"
        texts.append(text)
    return texts


# =============================================================================
# Seed helper
# =============================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Training and Evaluation (mirrors main.py)
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, device, edge_index, edge_weights,
                num_nodes_global, is_dual_head=False):
    """Train for one epoch using subgraph extraction (same logic as main.py)."""
    model.train()
    total_loss = 0.0
    loss_details = {'focal': 0.0, 'mse': 0.0} if is_dual_head else None

    edge_index = edge_index.to(device)
    if edge_weights is not None:
        edge_weights = edge_weights.to(device)

    for batch_data in loader:
        if is_dual_head and len(batch_data) == 5:
            embeddings, structural_features, labels, global_indices, priority_scores = batch_data
            priority_scores = priority_scores.to(device)
        else:
            embeddings, structural_features, labels, global_indices = batch_data[:4]
            priority_scores = None

        embeddings = embeddings.to(device)
        structural_features = structural_features.to(device)
        labels = labels.to(device)
        global_indices = global_indices.to(device)

        valid_mask = (global_indices != -1)
        if not valid_mask.any():
            continue

        embeddings_valid = embeddings[valid_mask]
        labels_valid = labels[valid_mask]
        global_indices_valid = global_indices[valid_mask]
        priority_scores_valid = priority_scores[valid_mask] if priority_scores is not None else None

        sub_edge_index, sub_edge_weights = subgraph(
            subset=global_indices_valid,
            edge_index=edge_index,
            edge_attr=edge_weights,
            relabel_nodes=True,
            num_nodes=num_nodes_global,
        )

        batch_structural_features = structural_features[valid_mask]

        if is_dual_head:
            structural_embeddings = model.structural_stream(
                batch_structural_features, sub_edge_index, sub_edge_weights
            )
            semantic_features = model.semantic_stream(embeddings_valid)
            fused_features = model.fusion(semantic_features, structural_embeddings)
            logits = model.classifier(fused_features)
            priority_pred = model.regressor(fused_features)
        else:
            structural_embeddings = model.structural_stream(
                batch_structural_features, sub_edge_index, sub_edge_weights
            )
            semantic_features = model.semantic_stream(embeddings_valid)
            fused_features = model.fusion(semantic_features, structural_embeddings)
            logits = model.classifier(fused_features)
            priority_pred = None

        # Compute loss
        if is_dual_head and priority_pred is not None and priority_scores_valid is not None:
            loss, loss_dict = criterion(logits, priority_pred, labels_valid, priority_scores_valid)
            loss_details['focal'] += loss_dict['focal']
            loss_details['mse'] += loss_dict['mse']
        else:
            if isinstance(criterion, DualHeadLoss):
                loss = criterion.focal_loss(logits, labels_valid)
            else:
                loss = criterion(logits, labels_valid)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / max(len(loader), 1)
    if is_dual_head and loss_details:
        loss_details['focal'] /= max(len(loader), 1)
        loss_details['mse'] /= max(len(loader), 1)
        return avg_loss, loss_details
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, criterion, device, edge_index, edge_weights,
             num_nodes_global, return_full_probs=False, dataset_size=None,
             is_dual_head=False, lightweight_metrics=False):
    """Evaluate model using subgraph extraction (same logic as main.py)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_priority_preds = []
    all_batch_indices = []

    edge_index = edge_index.to(device)
    if edge_weights is not None:
        edge_weights = edge_weights.to(device)
    batch_start_idx = 0

    for batch_data in loader:
        if is_dual_head and len(batch_data) == 5:
            embeddings, structural_features, labels, global_indices, priority_scores = batch_data
            priority_scores = priority_scores.to(device)
        else:
            embeddings, structural_features, labels, global_indices = batch_data[:4]
            priority_scores = None

        batch_size = embeddings.size(0)
        embeddings = embeddings.to(device)
        structural_features = structural_features.to(device)
        labels = labels.to(device)
        global_indices = global_indices.to(device)

        valid_mask = (global_indices != -1)
        if not valid_mask.any():
            batch_start_idx += batch_size
            continue

        valid_batch_indices = torch.arange(
            batch_start_idx, batch_start_idx + batch_size, device=device
        )[valid_mask]

        embeddings_valid = embeddings[valid_mask]
        labels_valid = labels[valid_mask]
        global_indices_valid = global_indices[valid_mask]
        priority_scores_valid = priority_scores[valid_mask] if priority_scores is not None else None

        sub_edge_index, sub_edge_weights = subgraph(
            subset=global_indices_valid,
            edge_index=edge_index,
            edge_attr=edge_weights,
            relabel_nodes=True,
            num_nodes=num_nodes_global,
        )

        batch_structural_features = structural_features[valid_mask]

        if is_dual_head:
            structural_embeddings = model.structural_stream(
                batch_structural_features, sub_edge_index, sub_edge_weights
            )
            semantic_features = model.semantic_stream(embeddings_valid)
            fused_features = model.fusion(semantic_features, structural_embeddings)
            logits = model.classifier(fused_features)
            priority_pred = model.regressor(fused_features)
        else:
            structural_embeddings = model.structural_stream(
                batch_structural_features, sub_edge_index, sub_edge_weights
            )
            semantic_features = model.semantic_stream(embeddings_valid)
            fused_features = model.fusion(semantic_features, structural_embeddings)
            logits = model.classifier(fused_features)
            priority_pred = None

        # Compute loss
        if is_dual_head and priority_pred is not None and priority_scores_valid is not None:
            loss, _ = criterion(logits, priority_pred, labels_valid, priority_scores_valid)
        else:
            if isinstance(criterion, DualHeadLoss):
                loss = criterion.focal_loss(logits, labels_valid)
            else:
                loss = criterion(logits, labels_valid)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_valid.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_batch_indices.extend(valid_batch_indices.cpu().numpy())

        if priority_pred is not None:
            all_priority_preds.extend(priority_pred.squeeze(-1).cpu().numpy())

        batch_start_idx += batch_size

    avg_loss = total_loss / max(len(loader), 1)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs) if len(all_probs) > 0 else np.empty((0, 2))
    all_batch_indices = np.array(all_batch_indices, dtype=np.int64)
    all_priority_preds = np.array(all_priority_preds) if all_priority_preds else None

    if len(all_preds) > 0:
        metrics = compute_metrics(
            predictions=all_preds,
            labels=all_labels,
            num_classes=2,
            label_names=['Not-Pass', 'Pass'] if not lightweight_metrics else None,
            probabilities=all_probs if not lightweight_metrics else None,
            lightweight=lightweight_metrics,
        )
    else:
        metrics = {'accuracy': 0.0, 'f1_macro': 0.0, 'f1_weighted': 0.0, 'auprc_macro': 0.0}

    if return_full_probs and dataset_size is not None:
        full_probs = np.full((dataset_size, 2), 0.5)
        if len(all_batch_indices) > 0 and len(all_probs) > 0:
            full_probs[all_batch_indices] = all_probs
        full_priority_preds = None
        if all_priority_preds is not None and len(all_priority_preds) > 0:
            full_priority_preds = np.full(dataset_size, 0.5)
            full_priority_preds[all_batch_indices] = all_priority_preds
        return avg_loss, metrics, full_probs, full_priority_preds
    else:
        return avg_loss, metrics, all_probs, all_priority_preds


# =============================================================================
# Main Experiment
# =============================================================================

def main():
    start_time = time.time()

    print("\n" + "=" * 70)
    print("EXPERIMENT: Filo-Priori with Random-Fixed Embeddings")
    print("Dataset: Industrial QTA (277 builds with failures)")
    print("Comparison: SBERT baseline APFD = {:.4f}".format(SBERT_BASELINE_APFD))
    print("=" * 70 + "\n")

    # -------------------------------------------------------------------------
    # 0. Setup
    # -------------------------------------------------------------------------
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load original config
    logger.info(f"Loading config from {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)

    # Override results directory so we don't overwrite the original
    config['output'] = config.get('output', {})
    config['output']['results_dir'] = str(RESULTS_DIR)

    # -------------------------------------------------------------------------
    # 1. Data Loading (same as main.py)
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: DATA PREPARATION")
    logger.info("=" * 70)

    data_loader = DataLoader(config)
    data_dict = data_loader.prepare_dataset()

    df_train = data_dict['train']
    df_val = data_dict['val']
    df_test = data_dict['test']
    class_weights = data_dict['class_weights']

    logger.info(f"  Train: {len(df_train)} samples, {df_train['Build_ID'].nunique()} builds")
    logger.info(f"  Val:   {len(df_val)} samples, {df_val['Build_ID'].nunique()} builds")
    logger.info(f"  Test:  {len(df_test)} samples, {df_test['Build_ID'].nunique()} builds")

    # -------------------------------------------------------------------------
    # 1.1 Priority Scores (DeepOrder features)
    # -------------------------------------------------------------------------
    logger.info("\n1.1: Computing DeepOrder-style Priority Scores...")
    priority_generator = create_priority_score_generator(config)

    df_train, tc_history_train, train_do_features = priority_generator.compute_priorities_for_dataframe(
        df_train, build_col='Build_ID', tc_col='TC_Key',
        result_col='TE_Test_Result', fail_value='Fail', pass_value='Pass',
        initial_history=None, extract_features=True,
    )
    df_val, tc_history_val, val_do_features = priority_generator.compute_priorities_for_dataframe(
        df_val, build_col='Build_ID', tc_col='TC_Key',
        result_col='TE_Test_Result', fail_value='Fail', pass_value='Pass',
        initial_history=tc_history_train, extract_features=True,
    )
    df_test, tc_history_test, test_do_features = priority_generator.compute_priorities_for_dataframe(
        df_test, build_col='Build_ID', tc_col='TC_Key',
        result_col='TE_Test_Result', fail_value='Fail', pass_value='Pass',
        initial_history=tc_history_val, extract_features=True,
    )
    logger.info(f"  DeepOrder features shape: {train_do_features.shape}")

    # -------------------------------------------------------------------------
    # 1.2 Random-Fixed Embeddings (REPLACES SBERT)
    # -------------------------------------------------------------------------
    logger.info("\n1.2: Generating RANDOM-FIXED embeddings (REPLACES SBERT)...")
    embedding_dim = config['embedding']['embedding_dim']  # 768
    combined_dim = embedding_dim * 2  # 1536

    n_train = len(df_train)
    n_val = len(df_val)
    n_test = len(df_test)

    # Combine train+val for embedding generation (same as main.py)
    train_val_df = pd.concat([df_train, df_val], ignore_index=True)

    logger.info(f"  Generating random-fixed TC embeddings for {len(train_val_df)} train+val samples...")
    train_val_tc_texts = prepare_tc_texts(train_val_df)
    train_val_tc_emb = generate_random_fixed_embeddings_batch(train_val_tc_texts, embedding_dim)

    logger.info(f"  Generating random-fixed commit embeddings for {len(train_val_df)} train+val samples...")
    train_val_commit_texts = prepare_commit_texts(train_val_df)
    train_val_commit_emb = generate_random_fixed_embeddings_batch(train_val_commit_texts, embedding_dim)

    logger.info(f"  Generating random-fixed TC embeddings for {n_test} test samples...")
    test_tc_texts = prepare_tc_texts(df_test)
    test_tc_emb = generate_random_fixed_embeddings_batch(test_tc_texts, embedding_dim)

    logger.info(f"  Generating random-fixed commit embeddings for {n_test} test samples...")
    test_commit_texts = prepare_commit_texts(df_test)
    test_commit_emb = generate_random_fixed_embeddings_batch(test_commit_texts, embedding_dim)

    # Concatenate TC + Commit embeddings -> combined 1536-dim
    train_embeddings = np.concatenate([train_val_tc_emb[:n_train], train_val_commit_emb[:n_train]], axis=1)
    val_embeddings = np.concatenate([train_val_tc_emb[n_train:], train_val_commit_emb[n_train:]], axis=1)
    test_embeddings = np.concatenate([test_tc_emb, test_commit_emb], axis=1)

    del train_val_df, train_val_tc_emb, train_val_commit_emb, test_tc_emb, test_commit_emb
    gc.collect()

    logger.info(f"  Train embeddings: {train_embeddings.shape}")
    logger.info(f"  Val embeddings:   {val_embeddings.shape}")
    logger.info(f"  Test embeddings:  {test_embeddings.shape}")
    logger.info(f"  NOTE: These are RANDOM-FIXED vectors, NOT SBERT!")

    # -------------------------------------------------------------------------
    # 1.3 Structural Features (V2.5)
    # -------------------------------------------------------------------------
    logger.info("\n1.3: Extracting structural features (V2.5)...")
    structural_config = config['structural']['extractor']
    cache_path = structural_config.get('cache_path')

    extractor = StructuralFeatureExtractorV2_5(
        recent_window=structural_config.get('recent_window', 5),
        very_recent_window=structural_config.get('very_recent_window', 2),
        medium_term_window=structural_config.get('medium_term_window', 10),
        min_history=structural_config.get('min_history', 2),
        verbose=True,
    )

    if cache_path and os.path.exists(cache_path):
        logger.info(f"  Loading cached extractor from {cache_path}")
        extractor.load_history(cache_path)
        train_struct = extractor.transform_temporal(df_train)
    else:
        logger.info("  Fitting and transforming training data temporally...")
        train_struct = extractor.fit_transform_temporal(df_train)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            extractor.save_history(cache_path)

    val_struct = extractor.transform_temporal(df_val)
    test_struct = extractor.transform_temporal(df_test)

    # Cold-start imputation for missing structural features
    tc_keys_train = df_train['TC_Key'].tolist()
    tc_keys_val = df_val['TC_Key'].tolist()
    tc_keys_test = df_test['TC_Key'].tolist()

    needs_imp_val = extractor.get_imputation_mask(tc_keys_val)
    needs_imp_test = extractor.get_imputation_mask(tc_keys_test)

    if needs_imp_val.sum() > 0 or needs_imp_test.sum() > 0:
        logger.info(f"  Imputing structural features: val={needs_imp_val.sum()}, test={needs_imp_test.sum()}")
        coldstart_cfg = config.get('coldstart', {})
        coldstart_cache = coldstart_cfg.get('cache_path', 'cache/coldstart_regressor.pkl')

        # NOTE: Using random embeddings for cold-start regressor -- the regressor learns
        # to predict structural features FROM embeddings, so with random embeddings it
        # becomes essentially a default-value imputer. This is intentional: we want to
        # see how the model performs when the semantic stream carries no real information.
        coldstart_regressor = create_coldstart_regressor(
            config, device=str(device),
        )

        # Train cold-start regressor on train data (with random embeddings)
        coldstart_regressor.fit(
            embeddings=train_embeddings,
            structural_features=train_struct,
            val_embeddings=val_embeddings if needs_imp_val.sum() == 0 else None,
            val_structural=val_struct if needs_imp_val.sum() == 0 else None,
        )

        if needs_imp_val.sum() > 0:
            val_struct = coldstart_regressor.impute_features(
                embeddings=val_embeddings, real_features=val_struct,
                needs_imputation=needs_imp_val,
            )
        if needs_imp_test.sum() > 0:
            test_struct = coldstart_regressor.impute_features(
                embeddings=test_embeddings, real_features=test_struct,
                needs_imputation=needs_imp_test,
            )

    # Concatenate DeepOrder features
    logger.info("\n1.3b: Concatenating DeepOrder features with structural features...")
    train_struct = np.concatenate([train_struct, train_do_features], axis=1)
    val_struct = np.concatenate([val_struct, val_do_features], axis=1)
    test_struct = np.concatenate([test_struct, test_do_features], axis=1)
    logger.info(f"  Combined structural features shape: {train_struct.shape}")

    # -------------------------------------------------------------------------
    # 1.4 Graph Building
    # -------------------------------------------------------------------------
    logger.info("\n1.4: Building phylogenetic graph...")
    graph_config = config['graph']

    # Build multi-edge graph (same as original, but with random embeddings for
    # the semantic edge type). This means semantic edges will be based on random
    # similarity, which is also part of the experiment.
    #
    # CRITICAL: Reset df_train index so that the graph builder's internal
    # tc_to_first_idx mapping aligns with the train_embeddings array.
    # Without this, df_train retains its original pre-split indices which
    # can exceed the embeddings array bounds.
    df_train_reindexed = df_train.reset_index(drop=True)
    graph_builder = build_phylogenetic_graph(
        df_train_reindexed,
        cache_path=None,  # Do NOT use cache; embeddings are different
        use_multi_edge=graph_config.get('use_multi_edge', True),
        embeddings=train_embeddings,
        edge_types=graph_config.get('edge_types', ['co_failure', 'co_success', 'semantic', 'temporal', 'component']),
        edge_weights_config=graph_config.get('edge_weights', None),
        min_co_occurrences=graph_config.get('min_co_occurrences', 1),
        weight_threshold=graph_config.get('weight_threshold', 0.05),
        semantic_top_k=graph_config.get('semantic_top_k', 10),
        semantic_threshold=graph_config.get('semantic_threshold', 0.7),
    )

    all_tc_keys = list(graph_builder.tc_to_idx.keys())
    edge_index, edge_weights_tensor = graph_builder.get_edge_index_and_weights(
        tc_keys=all_tc_keys, return_torch=True,
    )
    num_nodes_global = len(all_tc_keys)
    logger.info(f"  Graph: {edge_index.shape[1]} edges among {num_nodes_global} nodes")

    # TC_Key -> global index mapping
    tc_key_to_global_idx = graph_builder.tc_to_idx.copy()

    # -------------------------------------------------------------------------
    # 1.5 Package data
    # -------------------------------------------------------------------------
    train_data = {
        'embeddings': train_embeddings,
        'structural_features': train_struct,
        'labels': df_train['label'].values,
        'df': df_train,
        'priority_score': df_train['priority_score'].values if 'priority_score' in df_train.columns else None,
        'global_indices': np.array([tc_key_to_global_idx[tc] for tc in df_train['TC_Key']]),
        'tc_history_for_step6': tc_history_test,
    }
    val_data = {
        'embeddings': val_embeddings,
        'structural_features': val_struct,
        'labels': df_val['label'].values,
        'df': df_val,
        'priority_score': df_val['priority_score'].values if 'priority_score' in df_val.columns else None,
        'global_indices': np.array([tc_key_to_global_idx.get(tc, -1) for tc in df_val['TC_Key']]),
    }
    test_data = {
        'embeddings': test_embeddings,
        'structural_features': test_struct,
        'labels': df_test['label'].values,
        'df': df_test,
        'priority_score': df_test['priority_score'].values if 'priority_score' in df_test.columns else None,
        'global_indices': np.array([tc_key_to_global_idx.get(tc, -1) for tc in df_test['TC_Key']]),
    }

    # -------------------------------------------------------------------------
    # 2. Create DataLoaders
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: CREATING DATALOADERS")
    logger.info("=" * 70)

    from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader, WeightedRandomSampler

    batch_size = config['training']['batch_size']
    sampling_config = config['training'].get('sampling', {})
    use_balanced_sampling = sampling_config.get('use_balanced_sampling', False)
    minority_weight = sampling_config.get('minority_weight', 1.0)
    majority_weight = sampling_config.get('majority_weight', 0.05)
    model_type = config['model'].get('type', 'dual_stream')
    is_dual_head = (model_type == 'dual_head')

    def _to_float_tensor(arr):
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))

    def _to_long_tensor(arr):
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int64))

    def make_dataset(data_dict):
        tensors = [
            _to_float_tensor(data_dict['embeddings']),
            _to_float_tensor(data_dict['structural_features']),
            _to_long_tensor(data_dict['labels']),
            _to_long_tensor(data_dict['global_indices']),
        ]
        if is_dual_head and data_dict.get('priority_score') is not None:
            tensors.append(_to_float_tensor(data_dict['priority_score']))
        return TensorDataset(*tensors)

    train_dataset = make_dataset(train_data)
    val_dataset = make_dataset(val_data)
    test_dataset = make_dataset(test_data)

    if use_balanced_sampling:
        labels = train_data['labels']
        class_counts = np.bincount(labels)
        minority_class = int(np.argmin(class_counts))
        sample_weights = np.array([
            minority_weight if label == minority_class else majority_weight
            for label in labels
        ])
        max_samples = min(len(train_dataset), 500_000)
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=max_samples, replacement=True)
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
        logger.info(f"  Balanced sampling enabled: minority_weight={minority_weight}, majority_weight={majority_weight}")
    else:
        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------------------------------------------------------
    # 3. Model, Loss, Optimizer
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: MODEL INITIALIZATION")
    logger.info("=" * 70)

    # Harmonize model dimensions (same as main.py)
    model_cfg = config.get('model', {})
    semantic_hidden = model_cfg.get('semantic', {}).get('hidden_dim')
    gnn_hidden = model_cfg.get('gnn', {}).get('hidden_dim', semantic_hidden)
    gnn_heads = model_cfg.get('gnn', {}).get('num_heads', 1)
    expected_fusion_input = gnn_hidden * gnn_heads + semantic_hidden
    fusion_cfg = model_cfg.get('fusion', {})
    if fusion_cfg.get('input_dim') != expected_fusion_input:
        fusion_cfg['input_dim'] = expected_fusion_input
        model_cfg['fusion'] = fusion_cfg
        config['model'] = model_cfg

    model = create_model(config['model']).to(device)
    logger.info(f"  Model type: {model_type}")
    logger.info(f"  Is dual-head: {is_dual_head}")

    # Loss function
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    if is_dual_head:
        criterion = create_dual_head_loss(config, class_weights_tensor).to(device)
    else:
        criterion = create_loss_function(config, class_weights_tensor).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=float(config['training']['scheduler']['eta_min']),
    )

    # Move graph to device
    edge_index = edge_index.to(device)
    edge_weights_tensor = edge_weights_tensor.to(device)

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: TRAINING (with random-fixed embeddings)")
    logger.info("=" * 70)

    best_model_path = str(RESULTS_DIR / 'best_model.pt')
    best_val_f1 = 0.0
    patience_counter = 0
    patience = config['training']['early_stopping']['patience']
    num_epochs = config['training']['num_epochs']

    for epoch in range(num_epochs):
        train_result = train_epoch(
            model, train_loader, criterion, optimizer, device,
            edge_index, edge_weights_tensor, num_nodes_global,
            is_dual_head=is_dual_head,
        )
        if is_dual_head and isinstance(train_result, tuple):
            train_loss, train_loss_details = train_result
        else:
            train_loss = train_result
            train_loss_details = None

        val_loss, val_metrics, _, _ = evaluate(
            model, val_loader, criterion, device,
            edge_index, edge_weights_tensor, num_nodes_global,
            is_dual_head=is_dual_head, lightweight_metrics=True,
        )

        scheduler.step()

        if train_loss_details:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss={train_loss:.4f} (focal={train_loss_details['focal']:.4f}, mse={train_loss_details['mse']:.4f}), "
                f"Val Loss={val_loss:.4f}, Val F1={val_metrics['f1_macro']:.4f}"
            )
        else:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}: "
                f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                f"Val F1={val_metrics['f1_macro']:.4f}"
            )

        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  -> New best model saved (F1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info("Loaded best model checkpoint")

    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.1f}s")

    # -------------------------------------------------------------------------
    # 5. Evaluate on test split
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: TEST EVALUATION (split)")
    logger.info("=" * 70)

    _, test_metrics, test_probs_split, test_priority_split = evaluate(
        model, test_loader, criterion, device,
        edge_index, edge_weights_tensor, num_nodes_global,
        return_full_probs=True, dataset_size=len(df_test),
        is_dual_head=is_dual_head,
    )
    logger.info(f"  Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"  Test Accuracy:   {test_metrics['accuracy']:.4f}")

    # -------------------------------------------------------------------------
    # 6. Evaluate on FULL test.csv (277 builds) -- the PRIMARY metric
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: FULL TEST.CSV EVALUATION (277 builds)")
    logger.info("=" * 70)

    # Load full test set
    logger.info("\n6.1: Loading FULL test.csv...")
    test_df_full = data_loader.load_full_test_dataset()
    logger.info(f"  Total samples: {len(test_df_full)}")
    logger.info(f"  Total builds: {test_df_full['Build_ID'].nunique()}")

    builds_with_fail = test_df_full[test_df_full['TE_Test_Result'] == 'Fail']['Build_ID'].nunique()
    logger.info(f"  Builds with 'Fail': {builds_with_fail}")

    test_df_full['label_binary'] = (test_df_full['TE_Test_Result'].astype(str).str.strip() == 'Fail').astype(int)
    test_df_full['label'] = 1 - test_df_full['label_binary']

    # Generate random-fixed embeddings for full test set
    logger.info("\n6.2: Generating random-fixed embeddings for full test set...")
    full_tc_texts = prepare_tc_texts(test_df_full)
    full_tc_emb = generate_random_fixed_embeddings_batch(full_tc_texts, embedding_dim)
    full_commit_texts = prepare_commit_texts(test_df_full)
    full_commit_emb = generate_random_fixed_embeddings_batch(full_commit_texts, embedding_dim)
    test_embeddings_full = np.concatenate([full_tc_emb, full_commit_emb], axis=1)
    del full_tc_emb, full_commit_emb
    gc.collect()
    logger.info(f"  Full test embeddings: {test_embeddings_full.shape}")

    # Structural features for full test set
    logger.info("\n6.3: Extracting structural features for full test set...")
    test_struct_full = extractor.transform_temporal(test_df_full)

    tc_keys_test_full = test_df_full['TC_Key'].tolist()
    needs_imp_full = extractor.get_imputation_mask(tc_keys_test_full)

    if needs_imp_full.sum() > 0:
        logger.info(f"  Imputing {needs_imp_full.sum()} samples via KNN...")
        n_base = len(extractor.get_feature_names())
        train_struct_base = train_data['structural_features'][:, :n_base]
        test_struct_full, _ = impute_structural_features(
            train_data['embeddings'], train_struct_base, tc_keys_train,
            test_embeddings_full, test_struct_full, tc_keys_test_full,
            extractor.tc_history, k_neighbors=10, similarity_threshold=0.5,
            verbose=False,
        )

    # DeepOrder features for full test
    logger.info("\n6.3a: Computing DeepOrder features for full test set...")
    priority_gen_full = create_priority_score_generator(config)
    test_df_full, _, test_do_features_full = priority_gen_full.compute_priorities_for_dataframe(
        test_df_full, build_col='Build_ID', tc_col='TC_Key',
        result_col='TE_Test_Result', fail_value='Fail', pass_value='Pass',
        initial_history=train_data.get('tc_history_for_step6'),
        extract_features=True,
    )
    test_struct_full = np.concatenate([test_struct_full, test_do_features_full], axis=1)
    logger.info(f"  Combined features: {test_struct_full.shape}")

    # Map TC_Keys to global indices
    tc_key_to_global_idx_full = {tc: idx for idx, tc in enumerate(train_data['df']['TC_Key'].unique())}
    global_indices_full = np.array([tc_key_to_global_idx_full.get(tc, -1) for tc in tc_keys_test_full])
    in_graph = (global_indices_full != -1).sum()
    logger.info(f"  Samples in graph: {in_graph}/{len(global_indices_full)}")

    # Predict on full test set
    logger.info("\n6.4: Generating predictions...")
    test_dataset_full = TensorDataset(
        _to_float_tensor(test_embeddings_full),
        _to_float_tensor(test_struct_full),
        _to_long_tensor(test_df_full['label'].values),
        _to_long_tensor(global_indices_full),
    )
    test_loader_full = TorchDataLoader(test_dataset_full, batch_size=batch_size, shuffle=False)

    _, _, all_probs_full, all_priority_full = evaluate(
        model, test_loader_full, criterion, device,
        edge_index, edge_weights_tensor, num_nodes_global,
        return_full_probs=True, dataset_size=len(test_df_full),
        is_dual_head=is_dual_head,
    )
    logger.info(f"  Predictions shape: {all_probs_full.shape}")

    # Build hybrid ranking score (same as main.py FASE 4)
    logger.info("\n6.5: Building hybrid ranking score...")
    failure_probs_full = all_probs_full[:, 0]  # P(Fail)
    test_df_full['probability'] = failure_probs_full

    if is_dual_head and all_priority_full is not None:
        ranking_config = config.get('ranking', {})
        historical_boost_weight = ranking_config.get('historical_boost_weight', 0.3)

        orphan_mask = np.abs(all_priority_full - 0.5) < 0.001
        orphan_indices = np.where(orphan_mask)[0]
        in_graph_indices = np.where(~orphan_mask)[0]
        logger.info(f"  In-graph: {len(in_graph_indices)}, Orphans: {len(orphan_indices)}")

        hybrid_score = np.copy(failure_probs_full)

        # Blend P(Fail) with priority_pred for in-graph samples
        if len(in_graph_indices) > 0:
            in_graph_pfail = failure_probs_full[in_graph_indices]
            in_graph_priority = all_priority_full[in_graph_indices]
            priority_std = in_graph_priority.std()

            if priority_std > 0.01:
                priority_min = in_graph_priority.min()
                priority_max = in_graph_priority.max()
                if priority_max > priority_min:
                    normalized_priority = (in_graph_priority - priority_min) / (priority_max - priority_min)
                else:
                    normalized_priority = in_graph_priority
                lambda_pfail = ranking_config.get('lambda_pfail', 0.5)
                hybrid_score[in_graph_indices] = lambda_pfail * in_graph_pfail + (1 - lambda_pfail) * normalized_priority

        # Historical boost
        historical_priority = test_df_full['priority_score'].values if 'priority_score' in test_df_full.columns else None
        if historical_priority is not None:
            hist_max = historical_priority.max()
            if hist_max > 0:
                normalized_hist = historical_priority / hist_max
                has_history = historical_priority > 0
                if has_history.sum() > 0:
                    actual_boost = min(historical_boost_weight * 1.5, 0.8)
                    hybrid_score[has_history] = (
                        (1 - actual_boost) * hybrid_score[has_history]
                        + actual_boost * normalized_hist[has_history]
                    )

        # KNN for orphans
        orphan_strategy = ranking_config.get('orphan_strategy', {})
        if len(orphan_indices) > 0 and orphan_strategy.get('enabled', True) and len(in_graph_indices) > 0:
            priority_fallback = test_df_full['priority_score'].values if 'priority_score' in test_df_full.columns else None
            orphan_scores, orphan_stats = compute_orphan_scores(
                orphan_embeddings=test_embeddings_full[orphan_indices],
                in_graph_embeddings=test_embeddings_full[in_graph_indices],
                in_graph_scores=hybrid_score[in_graph_indices],
                orphan_base_scores=failure_probs_full[orphan_indices],
                strategy_config=orphan_strategy,
                orphan_structural_features=test_struct_full[orphan_indices],
                in_graph_structural_features=test_struct_full[in_graph_indices],
                orphan_priority_fallback=priority_fallback[orphan_indices] if priority_fallback is not None else None,
            )
            hybrid_score[orphan_indices] = orphan_scores
            logger.info(f"  Orphan KNN stats: mean={orphan_stats['mean']:.4f}")

        test_df_full['hybrid_score'] = hybrid_score
    else:
        test_df_full['hybrid_score'] = failure_probs_full

    # Generate prioritized CSV and APFD
    logger.info("\n6.6: Calculating APFD on FULL test.csv (277 builds)...")
    prioritized_path = str(RESULTS_DIR / 'prioritized_test_cases_FULL_testcsv.csv')
    test_df_ranked = generate_prioritized_csv(
        test_df_full, output_path=prioritized_path,
        probability_col='hybrid_score', label_col='label_binary', build_col='Build_ID',
    )

    apfd_path = str(RESULTS_DIR / 'apfd_per_build_FULL_testcsv.csv')
    apfd_results_df, apfd_summary = generate_apfd_report(
        test_df_ranked,
        method_name='Filo-Priori_RandomFixed',
        test_scenario='industry_randomfixed_277builds',
        output_path=apfd_path,
    )

    total_time = time.time() - start_time

    # -------------------------------------------------------------------------
    # 7. Print Results and Comparison
    # -------------------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)
    print_apfd_summary(apfd_summary)

    random_apfd = apfd_summary['mean_apfd'] if apfd_summary else 0.0
    delta = random_apfd - SBERT_BASELINE_APFD
    delta_pct = (delta / SBERT_BASELINE_APFD) * 100 if SBERT_BASELINE_APFD > 0 else 0.0

    print("\n" + "=" * 70)
    print("COMPARISON: Random-Fixed vs SBERT Baseline")
    print("=" * 70)
    print(f"  SBERT Baseline APFD:       {SBERT_BASELINE_APFD:.4f}")
    print(f"  Random-Fixed APFD:         {random_apfd:.4f}")
    print(f"  Delta:                     {delta:+.4f} ({delta_pct:+.1f}%)")
    print(f"  Builds evaluated:          {apfd_summary.get('total_builds', 0)}")
    print(f"  Best Val F1:               {best_val_f1:.4f}")
    print(f"  Total time:                {total_time:.1f}s")
    print()

    if abs(delta_pct) < 3.0:
        print("  CONCLUSION: Random-Fixed ~ SBERT -> Semantic stream is an identity proxy.")
        print("  SBERT embeddings do NOT provide meaningful information to Filo-Priori.")
    elif delta_pct < -3.0:
        print("  CONCLUSION: Random-Fixed << SBERT -> Semantic content matters for Filo-Priori.")
    else:
        print("  CONCLUSION: Random-Fixed > SBERT -> SBERT may be hurting performance (noise).")
    print("=" * 70)

    # Save experiment summary
    summary = {
        'experiment': 'Filo-Priori Random-Fixed Embeddings (Industrial)',
        'date': datetime.now().isoformat(),
        'seed': SEED,
        'device': str(device),
        'sbert_baseline_apfd': SBERT_BASELINE_APFD,
        'random_fixed_apfd': random_apfd,
        'delta': delta,
        'delta_pct': delta_pct,
        'total_builds': apfd_summary.get('total_builds', 0),
        'best_val_f1': best_val_f1,
        'total_time_seconds': total_time,
        'train_time_seconds': train_time,
        'embedding_dim': combined_dim,
        'embedding_type': 'random_fixed_hash_based',
        'apfd_summary': apfd_summary,
    }

    summary_path = RESULTS_DIR / 'experiment_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {RESULTS_DIR}/")
    logger.info(f"  - experiment_summary.json")
    logger.info(f"  - apfd_per_build_FULL_testcsv.csv")
    logger.info(f"  - prioritized_test_cases_FULL_testcsv.csv")
    logger.info(f"  - best_model.pt")

    # Save comparison text file
    comparison_path = RESULTS_DIR / 'comparison_summary.txt'
    with open(comparison_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Filo-Priori: Random-Fixed vs SBERT Embeddings\n")
        f.write("Dataset: Industrial QTA (277 builds)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"SBERT Baseline APFD:   {SBERT_BASELINE_APFD:.4f}\n")
        f.write(f"Random-Fixed APFD:     {random_apfd:.4f}\n")
        f.write(f"Delta:                 {delta:+.4f} ({delta_pct:+.1f}%)\n\n")
        f.write(f"Total builds:          {apfd_summary.get('total_builds', 0)}\n")
        f.write(f"Best Val F1:           {best_val_f1:.4f}\n")
        f.write(f"Training time:         {train_time:.1f}s\n")
        f.write(f"Total time:            {total_time:.1f}s\n\n")
        if apfd_summary:
            f.write(f"APFD Statistics:\n")
            f.write(f"  Mean:   {apfd_summary.get('mean_apfd', 0):.4f}\n")
            f.write(f"  Std:    {apfd_summary.get('std_apfd', 0):.4f}\n")
            f.write(f"  Median: {apfd_summary.get('median_apfd', 0):.4f}\n")
            f.write(f"  Min:    {apfd_summary.get('min_apfd', 0):.4f}\n")
            f.write(f"  Max:    {apfd_summary.get('max_apfd', 0):.4f}\n")
    logger.info(f"  - comparison_summary.txt")


if __name__ == '__main__':
    main()
