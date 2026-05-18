"""
Main Training Script for Filo-Priori

This script implements the production pipeline with:
- SBERT (all-mpnet-base-v2) for embeddings with intelligent caching
- Separate encoding for TCs and Commits
- Combined embedding dimension: 1536 (768 + 768)
- Dual-stream architecture with GAT
- Phylogenetic graph-based test prioritization

Usage:
    python main.py --config configs/experiment.yaml
    python main.py --config configs/experiment.yaml --force-regen-embeddings

Author: Filo-Priori Team
Date: 2024-11-14
"""

# CRITICAL: Set environment variables BEFORE importing torch/CUDA libraries
import os
os.environ["PYTORCH_NO_NVML"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import argparse
import logging
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from torch_geometric.utils import subgraph

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import V9 modules
from preprocessing.data_loader import DataLoader
from preprocessing.commit_extractor import CommitExtractor
from preprocessing.structural_feature_extractor import extract_structural_features, StructuralFeatureExtractor
from preprocessing.structural_feature_extractor_v2 import StructuralFeatureExtractorV2
from preprocessing.structural_feature_extractor_v2_5 import StructuralFeatureExtractorV2_5
from preprocessing.structural_feature_extractor_v3 import StructuralFeatureExtractorV3
from preprocessing.structural_feature_imputation import impute_structural_features
from preprocessing.structural_coldstart_regressor import ColdStartRegressor, create_coldstart_regressor
from preprocessing.priority_score_generator import PriorityScoreGenerator, create_priority_score_generator
from embeddings import EmbeddingManager
from phylogenetic.phylogenetic_graph_builder import build_phylogenetic_graph
from phylogenetic.git_dag_builder import GitDAGBuilder, build_git_dag
from models.model_factory import create_model
from training.losses import FocalLoss, create_loss_function
from evaluation.metrics import compute_metrics
from evaluation.apfd import generate_apfd_report, print_apfd_summary, generate_prioritized_csv
from evaluation.orphan_ranker import compute_orphan_scores

# Import validation utilities
try:
    from utils.config_validator import validate_config, ConfigValidationError
    HAS_CONFIG_VALIDATOR = True
except ImportError:
    HAS_CONFIG_VALIDATOR = False
    logging.warning("Config validator not available. Skipping validation.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """
    Load and validate configuration from YAML file

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML syntax is invalid
        ConfigValidationError: If config validation fails
    """
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    logger.info(f"Loading configuration from {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML syntax in {config_path}: {e}")
        raise

    # Validate configuration if validator available
    if HAS_CONFIG_VALIDATOR:
        logger.info("Validating configuration...")
        try:
            validate_config(config, strict=True)
            logger.info("Configuration validation passed!")
        except ConfigValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    else:
        logger.warning("Configuration validation skipped (validator not available)")

    return config


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def harmonize_model_dimensions(config: Dict) -> None:
    """
    Align model dimensions to avoid hidden_dim / fusion mismatches.
    Adjusts config in-place.
    """
    model_cfg = config.get('model', {})
    semantic_hidden = model_cfg.get('semantic', {}).get('hidden_dim')
    structural_hidden = model_cfg.get('structural', {}).get('hidden_dim')
    gnn_cfg = model_cfg.get('gnn', {})
    gnn_hidden = gnn_cfg.get('hidden_dim', structural_hidden or semantic_hidden)
    gnn_heads = gnn_cfg.get('num_heads', 1)

    expected_fusion_input = None
    if semantic_hidden is not None and gnn_hidden is not None:
        expected_fusion_input = gnn_hidden * gnn_heads + semantic_hidden

    fusion_cfg = model_cfg.get('fusion', {})
    fusion_input = fusion_cfg.get('input_dim')

    if expected_fusion_input is not None and fusion_input != expected_fusion_input:
        logger.warning(
            f"Aligning fusion.input_dim (was {fusion_input}) to "
            f"{expected_fusion_input} = gnn_hidden_dim * num_heads + semantic_hidden_dim"
        )
        fusion_cfg['input_dim'] = expected_fusion_input
        model_cfg['fusion'] = fusion_cfg

    # Keep structural hidden_dim aligned with semantic for fusion stability
    if semantic_hidden is not None and structural_hidden is not None and semantic_hidden != structural_hidden:
        logger.warning(
            f"Aligning structural hidden_dim ({structural_hidden}) to semantic hidden_dim ({semantic_hidden}) "
            f"to avoid fusion mismatch"
        )
        model_cfg.setdefault('structural', {})['hidden_dim'] = semantic_hidden

    config['model'] = model_cfg


def prepare_data(config: Dict, sample_size: int = None) -> Tuple:
    """
    Prepare data for training

    Args:
        config: Configuration dictionary
        sample_size: Optional sample size for testing

    Returns:
        Tuple of (train_data, val_data, test_data, graph_builder, edge_index, edge_weights,
                  class_weights, data_loader, encoder, text_processor, extractor)
    """
    logger.info("="*70)
    logger.info("STEP 1: DATA PREPARATION")
    logger.info("="*70)

    # Load data
    logger.info("\n1.1: Loading datasets...")
    data_loader = DataLoader(config)
    data_dict = data_loader.prepare_dataset(sample_size=sample_size)

    df_train = data_dict['train']
    df_val = data_dict['val']
    df_test = data_dict['test']

    logger.info(f"  Train: {len(df_train)} samples")
    logger.info(f"  Val: {len(df_val)} samples")
    logger.info(f"  Test: {len(df_test)} samples")

    # Compute class weights for weighted cross-entropy loss
    logger.info("\n1.1.1: Computing class weights...")
    class_weights = data_loader.compute_class_weights(df_train)
    logger.info(f"  Class weights: {class_weights}")
    logger.info(f"  Weight ratio (minority/majority): {class_weights.max() / class_weights.min():.2f}:1")

    # Compute DeepOrder-style Priority Scores (NEW in V9!)
    logger.info("\n1.1.2: Computing DeepOrder-style Priority Scores...")
    priority_config = config.get('priority_score', {})
    use_priority_scores = priority_config.get('enabled', True)

    if use_priority_scores:
        # Create priority score generator
        priority_generator = create_priority_score_generator(config)

        # Compute priority scores for all splits chronologically
        # IMPORTANT: Process train first, then val, then test to maintain temporal ordering
        # and carry over execution history between splits
        logger.info("  Processing training data...")
        df_train, tc_history_train, train_deeporder_features = priority_generator.compute_priorities_for_dataframe(
            df_train,
            build_col='Build_ID',
            tc_col='TC_Key',  # Use TC_Key (not TC_Name)
            result_col='TE_Test_Result',
            fail_value='Fail',
            pass_value='Pass',
            initial_history=None,  # Start fresh for training
            extract_features=True
        )

        logger.info("  Processing validation data (carrying over train history)...")
        df_val, tc_history_val, val_deeporder_features = priority_generator.compute_priorities_for_dataframe(
            df_val,
            build_col='Build_ID',
            tc_col='TC_Key',  # Use TC_Key (not TC_Name)
            result_col='TE_Test_Result',
            fail_value='Fail',
            pass_value='Pass',
            initial_history=tc_history_train,  # Carry over history from train
            extract_features=True
        )

        logger.info("  Processing test data (carrying over train+val history)...")
        df_test, tc_history, test_deeporder_features = priority_generator.compute_priorities_for_dataframe(
            df_test,
            build_col='Build_ID',
            tc_col='TC_Key',  # Use TC_Key (not TC_Name)
            result_col='TE_Test_Result',
            fail_value='Fail',
            pass_value='Pass',
            initial_history=tc_history_val,  # Carry over history from train+val
            extract_features=True
        )

        logger.info(f"  Priority scores computed:")
        logger.info(f"    Train: mean={df_train['priority_score'].mean():.4f}, max={df_train['priority_score'].max():.4f}")
        logger.info(f"    Val: mean={df_val['priority_score'].mean():.4f}, max={df_val['priority_score'].max():.4f}")
        logger.info(f"    Test: mean={df_test['priority_score'].mean():.4f}, max={df_test['priority_score'].max():.4f}")
        logger.info(f"  DeepOrder features shape: {train_deeporder_features.shape}")
        # Save tc_history for later use in STEP 6 (full test.csv processing)
        tc_history_for_step6 = tc_history
    else:
        logger.info("  Priority scores disabled in config")
        train_deeporder_features = None
        val_deeporder_features = None
        test_deeporder_features = None
        priority_generator = None
        tc_history_for_step6 = None

    # Extract semantic embeddings with SBERT (INTELLIGENT CACHING)
    logger.info("\n1.2: Extracting semantic embeddings with SBERT...")
    logger.info("  Using EmbeddingManager with intelligent caching")

    # Get embedding config (backward compatible with 'semantic' or 'embedding' key)
    embedding_config = config.get('embedding', config.get('semantic', {}))

    # Check if force regeneration is requested
    force_regen = config.get('_force_regen_embeddings', False)

    # Disable cache if using sample_size (to avoid size mismatches)
    cache_dir = embedding_config.get('cache_dir', 'cache') if sample_size is None else None
    use_cache = embedding_config.get('use_cache', True) and sample_size is None

    if not use_cache:
        logger.info("  Cache disabled (sample_size mode)")
    elif force_regen:
        logger.info("  Force regeneration enabled - ignoring existing cache")

    # Initialize EmbeddingManager
    embedding_manager = EmbeddingManager(
        config,
        force_regenerate=force_regen,
        cache_dir=cache_dir if use_cache else None
    )

    # Prepare combined dataframes for embedding generation
    # Note: EmbeddingManager needs train + val + test together to maintain cache consistency
    # But we'll only use train for initial generation, then reuse cache
    # Encode train + val together to avoid re-loading the SBERT model multiple times
    logger.info(f"  Generating/loading embeddings for {len(df_train) + len(df_val)} train+val + {len(df_test)} test samples...")
    train_val_df = pd.concat([df_train, df_val], ignore_index=True)

    # Get all embeddings at once (uses cache if available)
    all_embeddings = embedding_manager.get_embeddings(train_val_df, df_test)

    # Free the concatenated DataFrame - no longer needed after embedding generation
    del train_val_df
    import gc; gc.collect()

    embedding_dim = all_embeddings['embedding_dim']
    model_name = all_embeddings['model_name']

    logger.info(f"  Embedding dimension: {embedding_dim}")
    logger.info(f"  Combined dimension: {embedding_dim * 2}")
    logger.info(f"  Model: {model_name}")

    # Concatenate TC and Commit embeddings (memory-efficient: free source arrays progressively)
    # Numpy slices are views that keep the parent array alive, so we .copy() slices
    # and delete dict entries to actually free memory before allocating concatenated arrays.
    import gc
    n_train = len(df_train)

    logger.info("\n  Concatenating TC and Commit embeddings (memory-efficient)...")

    # Step 1: Copy TC slices, then free the large train_tc source array (~3.9 GB)
    logger.info("    Splitting and freeing TC embeddings...")
    train_tc = all_embeddings['train_tc'][:n_train]
    val_tc = all_embeddings['train_tc'][n_train:]
    test_tc = all_embeddings.pop('test_tc')
    del all_embeddings['train_tc']
    gc.collect()

    # Step 2: Copy commit slices, then free the large train_commit source array (~3.9 GB)
    logger.info("    Splitting and freeing commit embeddings...")
    train_commit = all_embeddings['train_commit'][:n_train]
    val_commit = all_embeddings['train_commit'][n_train:]
    test_commit = all_embeddings.pop('test_commit')
    del all_embeddings
    gc.collect()

    # Step 3: Concatenate train (largest allocation ~6.8 GB, but source arrays are now ~4.5 GB total)
    logger.info("    Building train embeddings (using memmap for large datasets)...")
    if n_train > 200000:
        logger.info("      Dataset > 200K, creating memmap on disk...")
        cache_dir = config['embedding'].get('cache_dir', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        mm_path = os.path.join(cache_dir, 'train_embeddings_combined.dat')
        train_embeddings = np.memmap(mm_path, dtype=np.float32, mode='w+', shape=(n_train, embedding_dim * 2))
        chunk_size = 100000
        for i in range(0, n_train, chunk_size):
            end = min(i + chunk_size, n_train)
            train_embeddings[i:end, :embedding_dim] = train_tc[i:end]
            train_embeddings[i:end, embedding_dim:] = train_commit[i:end]
        train_embeddings.flush()
    else:
        train_embeddings = np.concatenate([train_tc, train_commit], axis=1)
    del train_tc, train_commit
    gc.collect()

    # Step 4: Concatenate val
    logger.info("    Building val embeddings...")
    if len(val_tc) > 200000:
        mm_path_val = os.path.join(cache_dir, 'val_embeddings_combined.dat')
        val_embeddings = np.memmap(mm_path_val, dtype=np.float32, mode='w+', shape=(len(val_tc), embedding_dim * 2))
        chunk_size = 100000
        for i in range(0, len(val_tc), chunk_size):
            end = min(i + chunk_size, len(val_tc))
            val_embeddings[i:end, :embedding_dim] = val_tc[i:end]
            val_embeddings[i:end, embedding_dim:] = val_commit[i:end]
        val_embeddings.flush()
    else:
        val_embeddings = np.concatenate([val_tc, val_commit], axis=1)
    del val_tc, val_commit
    gc.collect()

    # Step 5: Concatenate test
    logger.info("    Building test embeddings...")
    if len(test_tc) > 200000:
        mm_path_test = os.path.join(cache_dir, 'test_embeddings_combined.dat')
        test_embeddings = np.memmap(mm_path_test, dtype=np.float32, mode='w+', shape=(len(test_tc), embedding_dim * 2))
        chunk_size = 100000
        for i in range(0, len(test_tc), chunk_size):
            end = min(i + chunk_size, len(test_tc))
            test_embeddings[i:end, :embedding_dim] = test_tc[i:end]
            test_embeddings[i:end, embedding_dim:] = test_commit[i:end]
        test_embeddings.flush()
    else:
        test_embeddings = np.concatenate([test_tc, test_commit], axis=1)
    del test_tc, test_commit
    gc.collect()

    logger.info(f"  Train embeddings: {train_embeddings.shape}")
    logger.info(f"  Val embeddings: {val_embeddings.shape}")
    logger.info(f"  Test embeddings: {test_embeddings.shape}")

    # Create CommitExtractor for later use in STEP 6 (full test.csv processing)
    commit_config = config.get('commit', {})
    commit_extractor = CommitExtractor(commit_config)

    # Extract structural features (NEW in V8!)
    logger.info("\n1.4: Extracting structural features...")
    structural_config = config['structural']['extractor']

    # Disable cache if using sample_size to avoid size mismatches
    cache_path = structural_config.get('cache_path') if sample_size is None else None
    if sample_size is not None and structural_config.get('cache_path'):
        logger.info(f"  Note: Cache disabled for sample_size={sample_size} to ensure correct shapes")

    # Create extractor manually to get access to tc_history for imputation
    use_v3 = structural_config.get('use_v3', False)
    use_v2 = structural_config.get('use_v2', False)
    use_v2_5 = structural_config.get('use_v2_5', False)

    if use_v3:
        logger.info("  Initializing StructuralFeatureExtractorV3 (14 DeepOrder-enhanced features)...")
        extractor = StructuralFeatureExtractorV3(
            recent_window=structural_config.get('recent_window', 10),
            very_recent_window=structural_config.get('very_recent_window', 3),
            medium_term_window=structural_config.get('medium_term_window', 20),
            min_history=structural_config.get('min_history', 2),
            decay_alpha=structural_config.get('decay_alpha', 0.1),
            max_time_since_failure=structural_config.get('max_time_since_failure', 50.0),
            verbose=True
        )
        logger.info("  ✓ Using V3 extractor with 14 features (last_verdict, time_since_failure, weighted_failure_rate)")
    elif use_v2_5:
        logger.info("  Initializing StructuralFeatureExtractorV2.5 (10 selected features)...")
        extractor = StructuralFeatureExtractorV2_5(
            recent_window=structural_config.get('recent_window', 5),
            very_recent_window=structural_config.get('very_recent_window', 2),
            medium_term_window=structural_config.get('medium_term_window', 10),
            min_history=structural_config.get('min_history', 2),
            verbose=True
        )
        logger.info("  ✓ Using V2.5 extractor with 10 selected features")
    elif use_v2:
        logger.info("  Initializing StructuralFeatureExtractorV2 (29 features)...")
        extractor = StructuralFeatureExtractorV2(
            recent_window=structural_config.get('recent_window', 5),
            very_recent_window=structural_config.get('very_recent_window', 2),
            medium_term_window=structural_config.get('medium_term_window', 10),
            min_history=structural_config.get('min_history', 2),
            verbose=True
        )
        logger.info("  ✓ Using V2 extractor with 29 features")
    else:
        logger.info("  Initializing StructuralFeatureExtractor (6 features)...")
        extractor = StructuralFeatureExtractor(
            recent_window=structural_config['recent_window'],
            min_history=structural_config.get('min_history', 2),
            verbose=True
        )
        logger.info("  ✓ Using V1 extractor with 6 features")

    # Load or fit (moto_v2 style: fit on all training data, then transform)
    # NOTE: fit_transform_temporal() computes features incrementally per build,
    # preventing look-ahead bias and simulating real-world execution.
    if cache_path and os.path.exists(cache_path):
        logger.info(f"  Loading cached extractor from {cache_path}")
        extractor.load_history(cache_path)
        logger.info("  Transforming training data...")
        train_struct = extractor.transform_temporal(df_train)
    else:
        logger.info("  Fitting and transforming training data temporally...")
        train_struct = extractor.fit_transform_temporal(df_train)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            extractor.save_history(cache_path)

    logger.info("  Transforming validation data...")
    val_struct = extractor.transform_temporal(df_val)

    logger.info("  Transforming test data...")
    test_struct = extractor.transform_temporal(df_test)

    # Impute missing features using Cold-Start Regressor
    logger.info("\n1.4b: Imputing missing structural features with Cold-Start Regressor...")
    logger.info("  (Uses MLP to predict structural features from semantic embeddings)")

    # Get TC_Keys for each split
    tc_keys_train = df_train['TC_Key'].tolist()
    tc_keys_val = df_val['TC_Key'].tolist()
    tc_keys_test = df_test['TC_Key'].tolist()

    # Check how many samples need imputation
    needs_imputation_val = extractor.get_imputation_mask(tc_keys_val)
    needs_imputation_test = extractor.get_imputation_mask(tc_keys_test)

    logger.info(f"  Validation samples needing imputation: {needs_imputation_val.sum()}/{len(tc_keys_val)}")
    logger.info(f"  Test samples needing imputation: {needs_imputation_test.sum()}/{len(tc_keys_test)}")

    # Check if cold-start imputation is enabled
    use_coldstart = config.get('coldstart', {}).get('enabled', True)
    coldstart_cache_path = config.get('coldstart', {}).get('cache_path', 'cache/coldstart_regressor.pkl')

    if use_coldstart and (needs_imputation_val.sum() > 0 or needs_imputation_test.sum() > 0):
        logger.info("\n  Training Cold-Start Regressor on train data...")

        # Create cold-start regressor
        coldstart_regressor = create_coldstart_regressor(
            config,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Check for cached regressor
        if sample_size is None and os.path.exists(coldstart_cache_path):
            logger.info(f"  Loading cached Cold-Start Regressor from {coldstart_cache_path}")
            coldstart_regressor.load(coldstart_cache_path)
        else:
            # Train on training data
            coldstart_regressor.fit(
                embeddings=train_embeddings,
                structural_features=train_struct,
                val_embeddings=val_embeddings if needs_imputation_val.sum() == 0 else None,
                val_structural=val_struct if needs_imputation_val.sum() == 0 else None
            )

            # Save for future use
            if sample_size is None:
                os.makedirs(os.path.dirname(coldstart_cache_path), exist_ok=True)
                coldstart_regressor.save(coldstart_cache_path)

        # Impute validation features
        if needs_imputation_val.sum() > 0:
            logger.info("  Imputing validation features with Cold-Start Regressor...")
            val_struct = coldstart_regressor.impute_features(
                embeddings=val_embeddings,
                real_features=val_struct,
                needs_imputation=needs_imputation_val
            )

        # Impute test features
        if needs_imputation_test.sum() > 0:
            logger.info("  Imputing test features with Cold-Start Regressor...")
            test_struct = coldstart_regressor.impute_features(
                embeddings=test_embeddings,
                real_features=test_struct,
                needs_imputation=needs_imputation_test
            )

        logger.info("  ✅ Cold-Start imputation complete!")

    elif needs_imputation_val.sum() > 0 or needs_imputation_test.sum() > 0:
        # Fallback to k-NN imputation if cold-start is disabled
        logger.info("  Using k-NN imputation (cold-start disabled)...")

        if needs_imputation_val.sum() > 0:
            val_struct, _ = impute_structural_features(
                train_embeddings, train_struct, tc_keys_train,
                val_embeddings, val_struct, tc_keys_val,
                extractor.tc_history,
                k_neighbors=10,
                similarity_threshold=0.5,
                verbose=False
            )

        if needs_imputation_test.sum() > 0:
            test_struct, _ = impute_structural_features(
                train_embeddings, train_struct, tc_keys_train,
                test_embeddings, test_struct, tc_keys_test,
                extractor.tc_history,
                k_neighbors=10,
                similarity_threshold=0.5,
                verbose=False
            )

    # Concatenate DeepOrder features with structural features (if enabled)
    if use_priority_scores and train_deeporder_features is not None:
        logger.info("\n1.4c: Concatenating DeepOrder features with structural features...")
        logger.info(f"  Original structural features shape: {train_struct.shape}")
        logger.info(f"  DeepOrder features shape: {train_deeporder_features.shape}")

        # Concatenate: [structural_features, deeporder_features]
        train_struct = np.concatenate([train_struct, train_deeporder_features], axis=1)
        val_struct = np.concatenate([val_struct, val_deeporder_features], axis=1)
        test_struct = np.concatenate([test_struct, test_deeporder_features], axis=1)

        logger.info(f"  Combined structural features shape: {train_struct.shape}")
        logger.info(f"  ✅ DeepOrder features concatenated!")

    # Apply SMOTE if enabled
    if config['data'].get('smote', {}).get('enabled', False):
        logger.info("\n1.5: Applying SMOTE to balance training data...")
        try:
            from imblearn.over_sampling import SMOTE

            smote_config = config['data']['smote']
            sampling_strategy = smote_config.get('sampling_strategy', 'auto')
            k_neighbors = smote_config.get('k_neighbors', 5)

            logger.info(f"  SMOTE configuration:")
            logger.info(f"    Sampling strategy: {sampling_strategy}")
            logger.info(f"    K neighbors: {k_neighbors}")

            # Combine embeddings and structural features
            X_train = np.concatenate([train_embeddings, train_struct], axis=1)
            y_train = df_train['label'].values

            logger.info(f"  Before SMOTE: {len(y_train)} samples")
            logger.info(f"    Class distribution: {np.bincount(y_train)}")

            # Apply SMOTE
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=config['experiment']['seed']
            )
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            logger.info(f"  After SMOTE: {len(y_train_resampled)} samples")
            logger.info(f"    Class distribution: {np.bincount(y_train_resampled)}")

            # Split back into embeddings and structural features
            train_embeddings = X_train_resampled[:, :train_embeddings.shape[1]]
            train_struct = X_train_resampled[:, train_embeddings.shape[1]:]

            # Update df_train labels (note: TC_Keys will be duplicated)
            # We'll create a synthetic df by repeating rows
            original_len = len(df_train)
            n_synthetic = len(y_train_resampled) - original_len

            if n_synthetic > 0:
                logger.info(f"  Created {n_synthetic} synthetic samples")

                # Get SMOTE sample indices (original + synthetic)
                # Synthetic samples are appended after originals in SMOTE
                df_train_original = df_train.copy()

                # For synthetic samples, duplicate random original samples for metadata
                # This preserves TC_Key, Build_ID etc. (needed for graph building)
                np.random.seed(config['experiment']['seed'])
                synthetic_indices = np.random.choice(len(df_train_original), n_synthetic, replace=True)
                df_synthetic = df_train_original.iloc[synthetic_indices].copy()

                df_train = pd.concat([df_train_original, df_synthetic], ignore_index=True)
                df_train['label'] = y_train_resampled

            logger.info("✅ SMOTE applied successfully!")

        except ImportError:
            logger.error("❌ ERROR: imblearn not installed. Cannot apply SMOTE.")
            logger.error("   Install with: pip install imbalanced-learn")
            logger.error("   Continuing without SMOTE...")
        except Exception as e:
            logger.error(f"❌ ERROR applying SMOTE: {e}")
            logger.error("   Continuing without SMOTE...")

    # Build phylogenetic graph (optional)
    graph_builder = None
    if config['graph'].get('build_graph', True):
        logger.info("\n1.6: Building phylogenetic graph...")
        graph_config = config['graph']

        # Disable cache if using sample_size
        graph_cache_path = graph_config.get('cache_path') if sample_size is None else None

        # Check if multi-edge mode is enabled
        use_multi_edge = graph_config.get('use_multi_edge', False)

        if use_multi_edge:
            # Multi-edge mode: pass embeddings for semantic edges
            logger.info("  Using multi-edge graph builder")

            # Reset df_train index so the graph builder's internal
            # tc_to_first_idx mapping aligns with the embeddings array.
            # df_train may retain non-contiguous indices from the
            # temporal split, causing IndexError in the builder.
            df_train_for_graph = df_train.reset_index(drop=True)

            graph_builder = build_phylogenetic_graph(
                df_train_for_graph,
                cache_path=graph_cache_path,
                use_multi_edge=True,
                embeddings=train_embeddings,
                edge_types=graph_config.get('edge_types', ['co_failure', 'co_success', 'semantic']),
                edge_weights_config=graph_config.get('edge_weights', None),
                min_co_occurrences=graph_config.get('min_co_occurrences', 1),
                weight_threshold=graph_config.get('weight_threshold', 0.05),
                semantic_top_k=graph_config.get('semantic_top_k', 10),
                semantic_threshold=graph_config.get('semantic_threshold', 0.7)
            )
        else:
            # Traditional single-edge mode
            graph_builder = build_phylogenetic_graph(
                df_train,
                graph_type=graph_config['type'],
                min_co_occurrences=graph_config['min_co_occurrences'],
                weight_threshold=graph_config['weight_threshold'],
                cache_path=graph_cache_path
            )

    # Build Git DAG for phylogenetic model (if enabled)
    git_dag = None
    git_dag_embeddings = None
    model_type = config.get('model', {}).get('type', 'dual_stream_v8')
    phylo_enabled = config.get('model', {}).get('phylo', {}).get('enabled', False)

    if model_type == 'phylogenetic_dual_stream' or phylo_enabled:
        logger.info("\n1.6b: Building Git DAG for PhyloEncoder...")
        git_dag_config = config.get('git_dag', {})

        git_dag_cache = git_dag_config.get('cache_path', 'cache/git_dag.pkl') if sample_size is None else None

        git_dag = build_git_dag(
            df_train,
            max_commits=git_dag_config.get('max_commits', 5000),
            temporal_window=git_dag_config.get('temporal_window', 10),
            cache_path=git_dag_cache
        )

        logger.info(f"  Git DAG built: {git_dag}")

        # Generate commit embeddings
        logger.info("  Generating commit embeddings for Git DAG...")
        git_dag_embeddings = git_dag.get_commit_embeddings(
            embedding_manager=embedding_manager,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info(f"  Commit embeddings: {git_dag_embeddings.shape}")

    logger.info("\n✓ Data preparation complete!")

    # Package data
    train_data = {
        'embeddings': train_embeddings,
        'structural_features': train_struct,
        'labels': df_train['label'].values,
        'df': df_train,
        'priority_score': df_train['priority_score'].values if 'priority_score' in df_train.columns else None,
        'deeporder_features': train_deeporder_features,
        'tc_history_for_step6': tc_history_for_step6,
    }

    val_data = {
        'embeddings': val_embeddings,
        'structural_features': val_struct,
        'labels': df_val['label'].values,
        'df': df_val,
        'priority_score': df_val['priority_score'].values if 'priority_score' in df_val.columns else None,
        'deeporder_features': val_deeporder_features
    }

    test_data = {
        'embeddings': test_embeddings,
        'structural_features': test_struct,
        'labels': df_test['label'].values,
        'df': df_test,
        'priority_score': df_test['priority_score'].values if 'priority_score' in df_test.columns else None,
        'deeporder_features': test_deeporder_features
    }

    # Extract edge_index and edge_weights for the graph
    logger.info("\n1.7: Extracting graph structure (edge_index and edge_weights)...")

    # Use TC_Keys from graph_builder
    all_tc_keys = list(graph_builder.tc_to_idx.keys())
    edge_index, edge_weights = graph_builder.get_edge_index_and_weights(
        tc_keys=all_tc_keys,
        return_torch=True
    )
    logger.info(f"Graph structure: {edge_index.shape[1]} edges among {len(all_tc_keys)} nodes")

    # Create TC_Key to global index mapping (for subgraph extraction)
    # Use the mapping from graph_builder to ensure consistency
    logger.info("\n1.8: Creating TC_Key to global index mapping...")
    tc_key_to_global_idx = graph_builder.tc_to_idx.copy()
    logger.info(f"  Mapped {len(tc_key_to_global_idx)} unique TC_Keys to global indices (0-{len(all_tc_keys)-1})")

    # Add global indices to each split's data
    train_data['global_indices'] = np.array([tc_key_to_global_idx[tc_key] for tc_key in df_train['TC_Key']])
    val_data['global_indices'] = np.array([tc_key_to_global_idx.get(tc_key, -1) for tc_key in df_val['TC_Key']])
    test_data['global_indices'] = np.array([tc_key_to_global_idx.get(tc_key, -1) for tc_key in df_test['TC_Key']])

    logger.info(f"  Train: {(train_data['global_indices'] != -1).sum()}/{len(train_data['global_indices'])} in graph")
    logger.info(f"  Val: {(val_data['global_indices'] != -1).sum()}/{len(val_data['global_indices'])} in graph")
    logger.info(f"  Test: {(test_data['global_indices'] != -1).sum()}/{len(test_data['global_indices'])} in graph")

    return (train_data, val_data, test_data, graph_builder, edge_index, edge_weights,
            class_weights, data_loader, embedding_manager, commit_extractor, extractor,
            len(all_tc_keys), git_dag, git_dag_embeddings)


def create_dataloaders(train_data: Dict, val_data: Dict, test_data: Dict, batch_size: int,
                       use_balanced_sampling: bool = False, minority_weight: float = 1.0,
                       majority_weight: float = 0.05, include_priority_score: bool = False):
    """
    Create PyTorch DataLoaders with global indices for subgraph extraction.

    Args:
        train_data: Training data dictionary with 'embeddings', 'structural_features',
                    'labels', and 'global_indices'
        val_data: Validation data dictionary
        test_data: Test data dictionary
        batch_size: Batch size for training
        use_balanced_sampling: If True, use WeightedRandomSampler for training
        minority_weight: Weight for minority class (default 1.0)
        majority_weight: Weight for majority class (default 0.05, i.e., 20:1 ratio)
        include_priority_score: If True, include priority_score in dataloader (for dual-head training)
    """
    from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

    # Debug: Print shapes
    logger.info(f"Train embeddings shape: {train_data['embeddings'].shape}")
    logger.info(f"Train structural features shape: {train_data['structural_features'].shape}")
    logger.info(f"Train labels shape: {train_data['labels'].shape}")
    logger.info(f"Train global indices shape: {train_data['global_indices'].shape}")

    # Check if priority_score is available
    has_priority = (include_priority_score and
                    train_data.get('priority_score') is not None and
                    val_data.get('priority_score') is not None and
                    test_data.get('priority_score') is not None)

    # Helper: convert numpy to torch tensor (avoid full copy of memmap)
    def _to_float_tensor(arr):
        if hasattr(arr, 'filename'):  # numpy memmap
            return torch.FloatTensor(np.array(arr, dtype=np.float32, copy=False))
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))

    def _to_long_tensor(arr):
        if hasattr(arr, 'filename'):  # numpy memmap
            return torch.LongTensor(np.array(arr, dtype=np.int64, copy=False))
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.int64))

    if has_priority:
        logger.info(f"Including priority_score in dataloaders (for dual-head training)")
        logger.info(f"Train priority_score shape: {train_data['priority_score'].shape}")

        # DIAGNOSTIC: Priority score distribution analysis
        train_ps = train_data['priority_score']
        nonzero_mask = train_ps > 0
        logger.info("\n" + "="*70)
        logger.info("PRIORITY SCORE DISTRIBUTION (CRITICAL FOR REGRESSION HEAD)")
        logger.info("="*70)
        logger.info(f"  Total samples: {len(train_ps)}")
        logger.info(f"  Samples with priority_score > 0: {nonzero_mask.sum()} ({100*nonzero_mask.mean():.2f}%)")
        logger.info(f"  Samples with priority_score = 0: {(~nonzero_mask).sum()} ({100*(~nonzero_mask).mean():.2f}%)")
        if nonzero_mask.sum() > 0:
            logger.info(f"  Non-zero priority scores:")
            logger.info(f"    Mean: {train_ps[nonzero_mask].mean():.4f}")
            logger.info(f"    Std:  {train_ps[nonzero_mask].std():.4f}")
            logger.info(f"    Min:  {train_ps[nonzero_mask].min():.4f}")
            logger.info(f"    Max:  {train_ps[nonzero_mask].max():.4f}")
            logger.info(f"    Median: {np.median(train_ps[nonzero_mask]):.4f}")
        logger.info(f"  Overall mean: {train_ps.mean():.4f}")
        logger.info("="*70)

        # Convert to tensors with priority_score (using from_numpy to share memory)
        train_dataset = TensorDataset(
            _to_float_tensor(train_data['embeddings']),
            _to_float_tensor(train_data['structural_features']),
            _to_long_tensor(train_data['labels']),
            _to_long_tensor(train_data['global_indices']),
            _to_float_tensor(train_data['priority_score'])
        )

        val_dataset = TensorDataset(
            _to_float_tensor(val_data['embeddings']),
            _to_float_tensor(val_data['structural_features']),
            _to_long_tensor(val_data['labels']),
            _to_long_tensor(val_data['global_indices']),
            _to_float_tensor(val_data['priority_score'])
        )

        test_dataset = TensorDataset(
            _to_float_tensor(test_data['embeddings']),
            _to_float_tensor(test_data['structural_features']),
            _to_long_tensor(test_data['labels']),
            _to_long_tensor(test_data['global_indices']),
            _to_float_tensor(test_data['priority_score'])
        )
    else:
        # Convert to tensors (without priority_score, using from_numpy to share memory)
        train_dataset = TensorDataset(
            _to_float_tensor(train_data['embeddings']),
            _to_float_tensor(train_data['structural_features']),
            _to_long_tensor(train_data['labels']),
            _to_long_tensor(train_data['global_indices'])
        )

        val_dataset = TensorDataset(
            _to_float_tensor(val_data['embeddings']),
            _to_float_tensor(val_data['structural_features']),
            _to_long_tensor(val_data['labels']),
            _to_long_tensor(val_data['global_indices'])
        )

        test_dataset = TensorDataset(
            _to_float_tensor(test_data['embeddings']),
            _to_float_tensor(test_data['structural_features']),
            _to_long_tensor(test_data['labels']),
            _to_long_tensor(test_data['global_indices'])
        )

    # Create train loader with optional balanced sampling
    if use_balanced_sampling:
        logger.info("\n" + "="*70)
        logger.info("BALANCED SAMPLING ENABLED")
        logger.info("="*70)

        # Get labels
        labels = train_data['labels']

        # Identify minority class (class with fewer samples)
        class_counts = np.bincount(labels)
        minority_class = int(np.argmin(class_counts))
        majority_class = 1 - minority_class

        logger.info(f"  Class distribution:")
        logger.info(f"    Class 0 (Not-Pass/Fail): {class_counts[0]} samples ({100*class_counts[0]/len(labels):.2f}%)")
        logger.info(f"    Class 1 (Pass): {class_counts[1]} samples ({100*class_counts[1]/len(labels):.2f}%)")
        logger.info(f"  Minority class: {minority_class}")
        logger.info(f"  Majority class: {majority_class}")

        # Create sample weights (higher weight for minority class)
        sample_weights = np.array([
            minority_weight if label == minority_class else majority_weight
            for label in labels
        ])

        # Calculate expected class balance in each batch
        total_weight = sample_weights.sum()
        minority_prob = (class_counts[minority_class] * minority_weight) / total_weight
        majority_prob = (class_counts[majority_class] * majority_weight) / total_weight

        logger.info(f"\n  Sampling weights:")
        logger.info(f"    Minority class weight: {minority_weight}")
        logger.info(f"    Majority class weight: {majority_weight}")
        logger.info(f"    Weight ratio (minority/majority): {minority_weight/majority_weight:.1f}:1")
        logger.info(f"\n  Expected sampling probabilities:")
        logger.info(f"    Minority class: {100*minority_prob:.2f}%")
        logger.info(f"    Majority class: {100*majority_prob:.2f}%")
        logger.info(f"  Expected samples per batch (size={batch_size}):")
        logger.info(f"    Minority class: ~{int(batch_size * minority_prob)} samples")
        logger.info(f"    Majority class: ~{int(batch_size * majority_prob)} samples")

        # Create WeightedRandomSampler (cap num_samples for large datasets)
        max_samples_per_epoch = min(len(train_dataset), 500_000)
        if max_samples_per_epoch < len(train_dataset):
            logger.info(f"\n  Capping epoch size from {len(train_dataset):,} to {max_samples_per_epoch:,} samples/epoch")
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=max_samples_per_epoch,
            replacement=True  # Allow replacement to oversample minority class
        )

        # Create train loader with sampler (shuffle must be False when using sampler)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False  # Must be False when using sampler
        )

        logger.info("="*70 + "\n")

    else:
        # Standard shuffled sampling
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Val and test loaders (no sampling, just sequential)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, device, edge_index, edge_weights,
                all_structural_features, num_nodes_global,
                phylo_embeddings=None, phylo_edge_index=None, phylo_path_lengths=None,
                is_dual_head=False):
    """
    Train for one epoch using subgraph extraction.

    Args:
        edge_index: Full graph edge_index [2, num_edges]
        edge_weights: Edge weights for full graph
        all_structural_features: All structural features [N_total, 6] for full-graph GAT processing
        num_nodes_global: Total number of nodes in the full graph (e.g., 161)
        phylo_embeddings: Optional commit embeddings for PhyloEncoder [M, 768]
        phylo_edge_index: Optional Git DAG edges [2, E_dag]
        phylo_path_lengths: Optional path lengths for phylo distance [E_dag]
        is_dual_head: If True, model returns (logits, priority_scores) and uses DualHeadLoss
    """
    model.train()
    total_loss = 0.0
    loss_details = {'focal': 0.0, 'mse': 0.0} if is_dual_head else None

    # Check if model is phylogenetic
    is_phylogenetic = hasattr(model, 'use_phylo_encoder') and model.use_phylo_encoder

    # Move full graph structure to device
    edge_index = edge_index.to(device)
    if edge_weights is not None:
        edge_weights = edge_weights.to(device)

    for batch_data in loader:
        # Unpack batch (4 or 5 elements depending on dual-head mode)
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

        # Filter out nodes not in the training graph (global_idx == -1)
        valid_mask = (global_indices != -1)

        if not valid_mask.any():
            # Skip batch if no valid nodes
            continue

        # Filter to valid nodes only
        embeddings_valid = embeddings[valid_mask]
        labels_valid = labels[valid_mask]
        global_indices_valid = global_indices[valid_mask]
        priority_scores_valid = priority_scores[valid_mask] if priority_scores is not None else None

        # Extract subgraph for this batch
        sub_edge_index, sub_edge_weights = subgraph(
            subset=global_indices_valid,
            edge_index=edge_index,
            edge_attr=edge_weights,
            relabel_nodes=True,
            num_nodes=num_nodes_global
        )

        # Get structural features for the batch nodes (from full graph features)
        batch_structural_features = structural_features[valid_mask]

        # Use different forward path based on model type
        if is_phylogenetic:
            # PhylogeneticDualStreamModel uses full forward()
            logits = model(
                semantic_input=embeddings_valid,
                structural_input=batch_structural_features,
                edge_index=sub_edge_index,
                edge_weights=sub_edge_weights,
                phylo_input=phylo_embeddings,
                phylo_edge_index=phylo_edge_index,
                phylo_path_lengths=phylo_path_lengths
            )
            priority_pred = None
        elif is_dual_head:
            # DualHeadModel: process and get both outputs
            structural_embeddings = model.structural_stream(
                batch_structural_features,
                sub_edge_index,
                sub_edge_weights
            )
            semantic_features = model.semantic_stream(embeddings_valid)
            fused_features = model.fusion(semantic_features, structural_embeddings)

            # Dual heads
            logits = model.classifier(fused_features)
            priority_pred = model.regressor(fused_features)
        else:
            # DualStreamModelV8 uses component-by-component forward
            # Process structural stream with GAT on SUBGRAPH
            structural_embeddings = model.structural_stream(
                batch_structural_features,
                sub_edge_index,
                sub_edge_weights
            )

            # Process semantic stream
            semantic_features = model.semantic_stream(embeddings_valid)

            # Fuse and classify
            fused_features = model.fusion(semantic_features, structural_embeddings)
            logits = model.classifier(fused_features)
            priority_pred = None

        # Compute loss
        if is_dual_head and priority_pred is not None and priority_scores_valid is not None:
            # DualHeadLoss: combined focal + MSE
            loss, loss_dict = criterion(logits, priority_pred, labels_valid, priority_scores_valid)
            loss_details['focal'] += loss_dict['focal']
            loss_details['mse'] += loss_dict['mse']
        else:
            # Standard loss (or fallback for DualHeadLoss without priority_scores)
            from models.dual_head_model import DualHeadLoss
            if isinstance(criterion, DualHeadLoss):
                # Use focal loss only (skip regression component)
                loss = criterion.focal_loss(logits, labels_valid)
            else:
                loss = criterion(logits, labels_valid)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    if is_dual_head and loss_details:
        loss_details['focal'] /= len(loader)
        loss_details['mse'] /= len(loader)
        return avg_loss, loss_details

    return avg_loss


@torch.no_grad()
def evaluate(model, loader, criterion, device, edge_index, edge_weights, all_structural_features, num_nodes_global,
             return_full_probs=False, dataset_size=None,
             phylo_embeddings=None, phylo_edge_index=None, phylo_path_lengths=None,
             is_dual_head=False, lightweight_metrics=False):
    """
    Evaluate model using subgraph extraction

    Args:
        edge_index: Full graph edge_index [2, num_edges]
        edge_weights: Edge weights for full graph
        all_structural_features: All structural features [N_total, 6] for full-graph GAT processing
        num_nodes_global: Total number of nodes in the full graph (e.g., 161)
        return_full_probs: If True, returns probabilities for ALL samples (filling orphans with [0.5, 0.5])
        dataset_size: Total dataset size (required if return_full_probs=True)
        phylo_embeddings: Optional commit embeddings for PhyloEncoder [M, 768]
        phylo_edge_index: Optional Git DAG edges [2, E_dag]
        phylo_path_lengths: Optional path lengths for phylo distance [E_dag]
        is_dual_head: If True, model is DualHeadModel with regression head
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    all_priority_preds = []  # For dual-head: predicted priority scores
    all_batch_indices = []  # Track original batch indices

    # Check if model is phylogenetic
    is_phylogenetic = hasattr(model, 'use_phylo_encoder') and model.use_phylo_encoder

    # Move full graph structure to device
    edge_index = edge_index.to(device)
    if edge_weights is not None:
        edge_weights = edge_weights.to(device)
    batch_start_idx = 0

    for batch_data in loader:
        # Unpack batch (4 or 5 elements depending on dual-head mode)
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

        # Filter out nodes not in the training graph (global_idx == -1)
        valid_mask = (global_indices != -1)

        if not valid_mask.any():
            # Skip batch if no valid nodes, but track indices if needed
            batch_start_idx += batch_size
            continue

        # Get original batch indices for valid samples
        valid_batch_indices = torch.arange(batch_start_idx, batch_start_idx + batch_size, device=device)[valid_mask]

        # Filter to valid nodes only
        embeddings_valid = embeddings[valid_mask]
        labels_valid = labels[valid_mask]
        global_indices_valid = global_indices[valid_mask]
        priority_scores_valid = priority_scores[valid_mask] if priority_scores is not None else None

        # Extract subgraph for this batch
        sub_edge_index, sub_edge_weights = subgraph(
            subset=global_indices_valid,
            edge_index=edge_index,
            edge_attr=edge_weights,
            relabel_nodes=True,
            num_nodes=num_nodes_global
        )

        # Get structural features for the batch nodes (from full graph features)
        batch_structural_features = structural_features[valid_mask]

        # Use different forward path based on model type
        if is_phylogenetic:
            # PhylogeneticDualStreamModel uses full forward()
            logits = model(
                semantic_input=embeddings_valid,
                structural_input=batch_structural_features,
                edge_index=sub_edge_index,
                edge_weights=sub_edge_weights,
                phylo_input=phylo_embeddings,
                phylo_edge_index=phylo_edge_index,
                phylo_path_lengths=phylo_path_lengths
            )
            priority_pred = None
        elif is_dual_head:
            # DualHeadModel: process and get both outputs
            structural_embeddings = model.structural_stream(
                batch_structural_features,
                sub_edge_index,
                sub_edge_weights
            )
            semantic_features = model.semantic_stream(embeddings_valid)
            fused_features = model.fusion(semantic_features, structural_embeddings)

            # Dual heads
            logits = model.classifier(fused_features)
            priority_pred = model.regressor(fused_features)
        else:
            # DualStreamModelV8 uses component-by-component forward
            # Process structural stream with GAT on SUBGRAPH
            structural_embeddings = model.structural_stream(
                batch_structural_features,
                sub_edge_index,
                sub_edge_weights
            )

            # Process semantic stream
            semantic_features = model.semantic_stream(embeddings_valid)

            # Fuse and classify
            fused_features = model.fusion(semantic_features, structural_embeddings)
            logits = model.classifier(fused_features)
            priority_pred = None

        # Compute loss
        if is_dual_head and priority_pred is not None and priority_scores_valid is not None:
            loss, _ = criterion(logits, priority_pred, labels_valid, priority_scores_valid)
        else:
            # Check if criterion is DualHeadLoss (requires 4 args) but we don't have priority_scores
            # This can happen in STEP 6 inference on full test.csv
            from models.dual_head_model import DualHeadLoss
            if isinstance(criterion, DualHeadLoss):
                # Use focal loss only (skip regression component)
                loss = criterion.focal_loss(logits, labels_valid)
            else:
                loss = criterion(logits, labels_valid)
        total_loss += loss.item()

        # Predictions
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_valid.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_batch_indices.extend(valid_batch_indices.cpu().numpy())

        # Collect priority predictions for dual-head
        if priority_pred is not None:
            all_priority_preds.extend(priority_pred.squeeze(-1).cpu().numpy())

        batch_start_idx += batch_size

    avg_loss = total_loss / max(len(loader), 1)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs) if len(all_probs) > 0 else np.empty((0, 2))
    all_batch_indices = np.array(all_batch_indices, dtype=np.int64)  # Ensure int type for indexing
    all_priority_preds = np.array(all_priority_preds) if all_priority_preds else None

    # Compute metrics on valid samples only
    if len(all_preds) > 0:
        metrics = compute_metrics(
            predictions=all_preds,
            labels=all_labels,
            num_classes=2,
            label_names=['Not-Pass', 'Pass'] if not lightweight_metrics else None,
            probabilities=all_probs if not lightweight_metrics else None,
            lightweight=lightweight_metrics
        )
    else:
        # No valid samples - return dummy metrics
        metrics = {
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'f1_weighted': 0.0,
            'auprc_macro': 0.0
        }

    # If requested, create full probability array with default values for orphans
    if return_full_probs and dataset_size is not None:
        full_probs = np.full((dataset_size, 2), 0.5)  # Default: [0.5, 0.5] (maximum uncertainty)
        if len(all_batch_indices) > 0 and len(all_probs) > 0:
            full_probs[all_batch_indices] = all_probs  # Fill in actual predictions

        # Also create full priority predictions array for dual-head ranking
        full_priority_preds = None
        if all_priority_preds is not None and len(all_priority_preds) > 0:
            # Default priority = 0.5 (neutral) for orphans
            # This ensures orphans don't dominate ranking
            full_priority_preds = np.full(dataset_size, 0.5)
            full_priority_preds[all_batch_indices] = all_priority_preds

        return avg_loss, metrics, full_probs, full_priority_preds
    else:
        return avg_loss, metrics, all_probs, all_priority_preds


def main():
    parser = argparse.ArgumentParser(description='Train Filo-Priori Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--sample-size', type=int, default=None, help='Sample size for quick testing')
    parser.add_argument('--force-regen-embeddings', action='store_true',
                       help='Force regeneration of embeddings (ignore cache)')
    args = parser.parse_args()

    # Load config
    logger.info("Loading configuration...")
    config = load_config(args.config)
    harmonize_model_dimensions(config)

    # Add force_regen flag to config for prepare_data to access
    config['_force_regen_embeddings'] = args.force_regen_embeddings

    if args.force_regen_embeddings:
        logger.info("⚠️  Force regeneration of embeddings enabled")

    # Set seed
    set_seed(config['experiment']['seed'])

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Prepare data
    (train_data, val_data, test_data, graph_builder, edge_index, edge_weights,
     class_weights, data_loader, embedding_manager, commit_extractor, extractor,
     num_nodes_global, git_dag, git_dag_embeddings) = prepare_data(config, args.sample_size)

    # Keep training data references for STEP 6 KNN imputation (same as moto_v2)
    train_embeddings = train_data['embeddings']
    train_struct = train_data['structural_features']
    tc_keys_train = train_data['df']['TC_Key'].tolist()
    logger.info(f"Kept training data for STEP 6 KNN imputation: embeddings {train_embeddings.shape}, struct {train_struct.shape}")

    # Free embedding_manager (SBERT model) - no longer needed after data preparation
    import gc
    del embedding_manager
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Freed embedding_manager (SBERT model) to save memory")

    # Move graph structure to device
    edge_index = edge_index.to(device)
    edge_weights = edge_weights.to(device)

    # Prepare Git DAG data for PhyloEncoder (if available)
    phylo_edge_index = None
    phylo_edge_weights = None
    phylo_path_lengths = None
    phylo_embeddings = None

    if git_dag is not None and git_dag_embeddings is not None:
        logger.info("\nPreparing Git DAG data for PhyloEncoder...")
        dag_data = git_dag.get_graph_data()
        phylo_edge_index = dag_data['edge_index'].to(device)
        phylo_edge_weights = dag_data['edge_weights'].to(device)
        phylo_path_lengths = dag_data['path_lengths'].to(device)
        # Clone embeddings to allow autograd (they were created in inference mode)
        phylo_embeddings = git_dag_embeddings.clone().detach().to(device)
        phylo_embeddings.requires_grad_(False)  # Keep as fixed embeddings
        logger.info(f"  Phylo edge_index: {phylo_edge_index.shape}")
        logger.info(f"  Phylo embeddings: {phylo_embeddings.shape}")

    # Extract structural features for GAT processing
    logger.info("\nPreparing structural features for graph...")
    train_structural_features = torch.FloatTensor(train_data['structural_features'])

    # Create data loaders
    logger.info("\nCreating data loaders...")
    batch_size = config['training']['batch_size']

    # Get sampling config
    sampling_config = config['training'].get('sampling', {})
    use_balanced_sampling = sampling_config.get('use_balanced_sampling', False)
    minority_weight = sampling_config.get('minority_weight', 1.0)
    majority_weight = sampling_config.get('majority_weight', 0.05)

    # Check if using dual-head model (needs priority_score in dataloaders)
    model_type = config['model'].get('type', 'dual_stream')
    is_dual_head = (model_type == 'dual_head')

    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, batch_size,
        use_balanced_sampling=use_balanced_sampling,
        minority_weight=minority_weight,
        majority_weight=majority_weight,
        include_priority_score=is_dual_head  # Include priority_score for dual-head training
    )

    # Create model
    logger.info("\n"+"="*70)
    logger.info("STEP 2: MODEL INITIALIZATION")
    logger.info("="*70)

    model = create_model(config['model']).to(device)

    # Loss function (using new unified create_loss_function)
    logger.info("\nInitializing loss function...")

    # Convert class_weights to torch tensor
    class_weights_tensor = torch.FloatTensor(class_weights).to(device) if class_weights is not None else None

    # Log loss configuration
    loss_type = config['training']['loss']['type']
    logger.info(f"  Loss type: {loss_type}")

    # Handle dual-head loss
    if is_dual_head or loss_type == 'dual_head':
        from models.dual_head_model import create_dual_head_loss
        criterion = create_dual_head_loss(config, class_weights_tensor)
        criterion = criterion.to(device)
        logger.info(f"  Using DualHeadLoss (Classification + Regression)")
        dual_head_cfg = config['training']['loss'].get('dual_head', {})
        logger.info(f"    α (classification): {dual_head_cfg.get('alpha', 1.0)}")
        logger.info(f"    β (regression): {dual_head_cfg.get('beta', 0.5)}")
        logger.info(f"    Focal alpha: {config['training']['loss'].get('focal_alpha', 0.75)}")
        logger.info(f"    Focal gamma: {config['training']['loss'].get('focal_gamma', 2.0)}")
    else:
        # Create loss function using unified factory
        # This supports: 'ce', 'weighted_ce', 'focal', 'weighted_focal'
        criterion = create_loss_function(config, class_weights_tensor)
        criterion = criterion.to(device)

        if loss_type == 'weighted_focal':
            use_cw = config['training']['loss'].get('use_class_weights', True)
            logger.info(f"  Using WeightedFocalLoss (STRONGEST for imbalanced data)")
            logger.info(f"    Focal alpha: {config['training']['loss'].get('focal_alpha', 0.75)}")
            logger.info(f"    Focal gamma: {config['training']['loss'].get('focal_gamma', 3.0)}")
            logger.info(f"    Use class weights: {use_cw}")
            if use_cw:
                logger.info(f"    Class weights: {class_weights}")
            else:
                logger.info(f"    Class weights: DISABLED (balanced sampling handles imbalance)")
            logger.info(f"    Label smoothing: {config['training']['loss'].get('label_smoothing', 0.0)}")
        elif loss_type == 'focal':
            logger.info(f"  Using Focal Loss")
            focal_alpha = config['training']['loss'].get('focal_alpha', 0.25)
            logger.info(f"    Focal alpha: {focal_alpha}")
            logger.info(f"    Focal gamma: {config['training']['loss'].get('focal_gamma', 2.0)}")
            logger.info(f"    Use class weights: {config['training']['loss'].get('use_class_weights', False)}")
        elif loss_type == 'weighted_ce':
            logger.info(f"  Using Weighted Cross-Entropy")
            logger.info(f"    Class weights: {class_weights}")
            logger.info(f"    Weight ratio: {class_weights.max() / class_weights.min():.2f}:1")
        else:
            logger.info(f"  Using standard Cross-Entropy")

    # Optimizer
    logger.info("Initializing optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay'])
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=float(config['training']['scheduler']['eta_min'])
    )

    # Training loop
    logger.info("\n"+"="*70)
    logger.info("STEP 3: TRAINING")
    logger.info("="*70)

    # Resolve checkpoint path inside results dir to avoid stale root-level files
    results_dir = config['output']['results_dir']
    checkpoint_cfg = config.get('checkpoint', {})
    checkpoint_dir = checkpoint_cfg.get('dir', results_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, checkpoint_cfg.get('best_name', 'best_model.pt'))

    best_val_f1 = 0.0
    patience_counter = 0
    patience = config['training']['early_stopping']['patience']

    for epoch in range(config['training']['num_epochs']):
        # Train
        train_result = train_epoch(
            model, train_loader, criterion, optimizer, device,
            edge_index, edge_weights, train_structural_features, num_nodes_global,
            phylo_embeddings=phylo_embeddings, phylo_edge_index=phylo_edge_index,
            phylo_path_lengths=phylo_path_lengths,
            is_dual_head=is_dual_head
        )

        # Handle dual-head return value (train_loss, loss_details) or single loss
        if is_dual_head and isinstance(train_result, tuple):
            train_loss, train_loss_details = train_result
        else:
            train_loss = train_result
            train_loss_details = None

        # Validate (use TRAIN structural features for graph structure)
        val_loss, val_metrics, _, _ = evaluate(
            model, val_loader, criterion, device,
            edge_index, edge_weights, train_structural_features, num_nodes_global,
            phylo_embeddings=phylo_embeddings, phylo_edge_index=phylo_edge_index,
            phylo_path_lengths=phylo_path_lengths,
            is_dual_head=is_dual_head,
            lightweight_metrics=True
        )

        # Update scheduler
        scheduler.step()

        # Log
        if is_dual_head and train_loss_details:
            logger.info(
                f"Epoch {epoch+1}/{config['training']['num_epochs']}: "
                f"Train Loss={train_loss:.4f} (focal={train_loss_details['focal']:.4f}, mse={train_loss_details['mse']:.4f}), "
                f"Val Loss={val_loss:.4f}, "
                f"Val F1={val_metrics['f1_macro']:.4f}, "
                f"Val Acc={val_metrics['accuracy']:.4f}"
            )
        else:
            logger.info(
                f"Epoch {epoch+1}/{config['training']['num_epochs']}: "
                f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                f"Val F1={val_metrics['f1_macro']:.4f}, "
                f"Val Acc={val_metrics['accuracy']:.4f}"
            )

        # Early stopping
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  → New best model saved! (F1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model (if it exists)
    if os.path.exists(best_model_path):
        logger.info("\nLoading best model...")
        try:
            model.load_state_dict(torch.load(best_model_path))
        except RuntimeError as e:
            logger.error(f"❌ Failed to load checkpoint at {best_model_path}: {e}")
            logger.error("   The file may come from an older architecture (e.g., 6 structural features).")
            logger.error("   Delete or rename the stale checkpoint and re-run training.")
    else:
        logger.warning("\nNo best model checkpoint found - using final model state")

    # ==============================================================================
    # STEP 3.5: THRESHOLD OPTIMIZATION ON VALIDATION SET
    # ==============================================================================
    logger.info("\n"+"="*70)
    logger.info("STEP 3.5: THRESHOLD OPTIMIZATION")
    logger.info("="*70)

    # Check if threshold optimization is enabled
    threshold_config = config.get('evaluation', {}).get('threshold_search', {})
    use_threshold_optimization = threshold_config.get('enabled', False)

    optimal_threshold = 0.5  # Default
    threshold_metrics_info = {}

    if use_threshold_optimization:
        logger.info("\nFinding optimal classification threshold on validation set...")

        # Get validation probabilities from best model
        model.eval()
        with torch.no_grad():
            _, _, val_probs, _ = evaluate(
                model, val_loader, criterion, device, edge_index, edge_weights,
                train_structural_features, num_nodes_global,
                return_full_probs=True, dataset_size=len(val_data['df']),
                phylo_embeddings=phylo_embeddings, phylo_edge_index=phylo_edge_index,
                phylo_path_lengths=phylo_path_lengths,
                is_dual_head=is_dual_head
            )

        val_labels = val_data['labels']
        val_probs_positive = val_probs[:, 1]  # Probability of Pass class

        # Import threshold optimizer
        from src.evaluation.threshold_optimizer import find_optimal_threshold

        # Find optimal threshold
        optimize_for = threshold_config.get('optimize_for', 'f1_macro')
        min_threshold = threshold_config.get('range', [0.01, 0.99])[0]
        max_threshold = threshold_config.get('range', [0.01, 0.99])[1]
        step = threshold_config.get('step', 0.01)
        coarse_step = threshold_config.get('coarse_step', step)
        fine_step = threshold_config.get('fine_step', max(step / 4, 0.001))
        fine_window = threshold_config.get('fine_window', 0.05)
        two_phase = threshold_config.get('two_phase', False)
        beta = threshold_config.get('beta', 1.0)

        num_thresholds = threshold_config.get('num_thresholds')
        if num_thresholds is None and not two_phase:
            num_thresholds = int((max_threshold - min_threshold) / step) + 1

        try:
            optimal_threshold, threshold_metrics_info = find_optimal_threshold(
                y_true=val_labels,
                y_prob=val_probs_positive,
                strategy=optimize_for,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
                num_thresholds=num_thresholds,
                two_phase=two_phase,
                coarse_step=coarse_step,
                fine_step=fine_step,
                fine_window=fine_window,
                beta=beta
            )

            logger.info(f"\n✅ Threshold Optimization Results:")
            logger.info(f"   Strategy: {optimize_for}")
            if two_phase:
                logger.info(f"   Two-phase search: coarse_step={coarse_step}, fine_step={fine_step}, window={fine_window}")
            logger.info(f"   Optimal threshold: {optimal_threshold:.4f} (default: 0.5)")
            logger.info(f"   Expected validation F1 Macro: {threshold_metrics_info.get('f1_macro', 0):.4f}")
            logger.info(f"   Expected validation Recall (minority): {threshold_metrics_info.get('recall_per_class', [0, 0])[0]:.4f}")

            # Save optimal threshold
            threshold_info_path = os.path.join(config['output']['results_dir'], 'optimal_threshold.txt')
            os.makedirs(config['output']['results_dir'], exist_ok=True)
            with open(threshold_info_path, 'w') as f:
                f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n")
                f.write(f"Strategy: {optimize_for}\n")
                f.write(f"Validation F1 Macro: {threshold_metrics_info.get('f1_macro', 0):.4f}\n")
                f.write(f"Validation Recall (Not-Pass): {threshold_metrics_info.get('recall_per_class', [0, 0])[0]:.4f}\n")
                f.write(f"Validation Recall (Pass): {threshold_metrics_info.get('recall_per_class', [0, 0])[1]:.4f}\n")

            logger.info(f"   Threshold info saved to: {threshold_info_path}")

        except Exception as e:
            logger.warning(f"⚠️  Threshold optimization failed: {e}")
            logger.warning(f"   Using default threshold: 0.5")
            optimal_threshold = 0.5

    else:
        logger.info("\nThreshold optimization disabled in config - using default threshold 0.5")

    logger.info(f"\n📊 Classification threshold for test evaluation: {optimal_threshold:.4f}")

    # Test evaluation
    logger.info("\n"+"="*70)
    logger.info("STEP 4: TEST EVALUATION")
    logger.info("="*70)

    # Test evaluation (use TRAIN structural features for graph structure)
    # Request full probabilities for ALL test samples (including orphans)
    test_loss, test_metrics, test_probs, test_priority_preds = evaluate(
        model, test_loader, criterion, device, edge_index, edge_weights,
        train_structural_features, num_nodes_global,
        return_full_probs=True, dataset_size=len(test_data['df']),
        phylo_embeddings=phylo_embeddings, phylo_edge_index=phylo_edge_index,
        phylo_path_lengths=phylo_path_lengths,
        is_dual_head=is_dual_head
    )

    logger.info("\nTest Results with default threshold (0.5):")
    logger.info(f"  Loss: {test_loss:.4f}")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1 (Macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"  F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
    logger.info(f"  AUPRC (Macro): {test_metrics.get('auprc_macro', 0.0):.4f}")

    # If threshold optimization was enabled, recompute metrics with optimal threshold
    if use_threshold_optimization and optimal_threshold != 0.5:
        logger.info(f"\n📊 Recomputing test metrics with optimal threshold ({optimal_threshold:.4f})...")

        # Import sklearn for per-class recall (needed for comparison)
        from sklearn.metrics import recall_score

        # Get test labels
        test_labels = test_data['labels']

        # Recompute predictions with optimal threshold
        test_probs_positive = test_probs[:, 1]  # P(Pass)
        test_preds_optimized = (test_probs_positive >= optimal_threshold).astype(int)

        # Compute metrics with optimized threshold
        from src.evaluation.metrics import compute_metrics
        test_metrics_optimized = compute_metrics(
            predictions=test_preds_optimized,
            labels=test_labels,
            num_classes=2,
            label_names=['Not-Pass', 'Pass'],
            probabilities=test_probs
        )

        # Show comparison
        logger.info("\n" + "="*80)
        logger.info(f"THRESHOLD COMPARISON: Default (0.5) vs Optimized ({optimal_threshold:.4f})")
        logger.info("="*80)

        logger.info(f"\n{'Metric':<25} {'Default (0.5)':<20} {'Optimized':<20} {'Change':<15}")
        logger.info("-" * 80)

        # Accuracy
        acc_change = test_metrics_optimized['accuracy'] - test_metrics['accuracy']
        logger.info(f"{'Accuracy':<25} {test_metrics['accuracy']:<20.4f} "
                   f"{test_metrics_optimized['accuracy']:<20.4f} "
                   f"{acc_change:+.4f}")

        # F1 Macro
        f1_change = test_metrics_optimized['f1_macro'] - test_metrics['f1_macro']
        f1_change_pct = (f1_change / test_metrics['f1_macro'] * 100) if test_metrics['f1_macro'] > 0 else 0
        logger.info(f"{'F1 Macro':<25} {test_metrics['f1_macro']:<20.4f} "
                   f"{test_metrics_optimized['f1_macro']:<20.4f} "
                   f"{f1_change:+.4f} ({f1_change_pct:+.1f}%)")

        # Precision Macro
        prec_change = test_metrics_optimized['precision_macro'] - test_metrics['precision_macro']
        logger.info(f"{'Precision Macro':<25} {test_metrics['precision_macro']:<20.4f} "
                   f"{test_metrics_optimized['precision_macro']:<20.4f} "
                   f"{prec_change:+.4f}")

        # Recall Macro
        rec_change = test_metrics_optimized['recall_macro'] - test_metrics['recall_macro']
        logger.info(f"{'Recall Macro':<25} {test_metrics['recall_macro']:<20.4f} "
                   f"{test_metrics_optimized['recall_macro']:<20.4f} "
                   f"{rec_change:+.4f}")

        logger.info("\n" + "="*80)
        logger.info("KEY IMPROVEMENT: Minority Class (Not-Pass) Recall")
        logger.info("="*80)

        # Get per-class recalls
        default_recall_per_class = recall_score(test_labels, (test_probs_positive >= 0.5).astype(int),
                                                average=None, zero_division=0)
        opt_recall_per_class = recall_score(test_labels, test_preds_optimized,
                                           average=None, zero_division=0)

        recall_notpass_change = opt_recall_per_class[0] - default_recall_per_class[0]
        recall_notpass_change_pct = (recall_notpass_change / default_recall_per_class[0] * 100) if default_recall_per_class[0] > 0 else float('inf')

        logger.info(f"\nRecall Not-Pass (Minority):")
        logger.info(f"  Default (0.5):   {default_recall_per_class[0]:.4f}")
        logger.info(f"  Optimized ({optimal_threshold:.2f}): {opt_recall_per_class[0] if len(opt_recall_per_class) > 0 else 0.0:.4f}")
        logger.info(f"  Change:          {recall_notpass_change:+.4f} ({recall_notpass_change_pct:+.1f}%)")

        logger.info(f"\nRecall Pass (Majority):")
        logger.info(f"  Default (0.5):   {default_recall_per_class[1]:.4f}")
        
    try:
        opt_recall_0 = opt_recall_per_class[0] if len(opt_recall_per_class) > 0 else 0.0
        opt_recall_1 = opt_recall_per_class[1] if len(opt_recall_per_class) > 1 else 0.0
        logger.info(f"  Optimized ({optimal_threshold:.2f}): {opt_recall_1:.4f}")
    except IndexError:
        logger.info(f"  Optimized ({optimal_threshold:.2f}): N/A")


        logger.info("\n" + "="*80)

        # Use optimized metrics for final reporting
        test_metrics_final = test_metrics_optimized
        logger.info(f"\n✅ Using optimized threshold ({optimal_threshold:.4f}) for final evaluation and APFD calculation")
    else:
        test_metrics_final = test_metrics
        if not use_threshold_optimization:
            logger.info("\n📝 Using default threshold (0.5) - threshold optimization was disabled in config")

    # APFD calculation
    logger.info("\n"+"="*70)
    logger.info("STEP 5: APFD CALCULATION")
    logger.info("="*70)

    # Add probabilities to test DataFrame
    test_df = test_data['df'].copy()
    logger.info(f"  Test DataFrame size: {len(test_df)}")
    logger.info(f"  Probabilities array size: {test_probs.shape}")

    # Verify sizes match
    if len(test_df) != len(test_probs):
        logger.error(f"❌ Size mismatch: test_df={len(test_df)}, test_probs={len(test_probs)}")
        raise ValueError(f"Size mismatch between test_df ({len(test_df)}) and test_probs ({len(test_probs)})")

    # Add model predictions to test DataFrame
    test_df['probability'] = test_probs[:, 0]  # P(Fail) - class 0 with pass_vs_fail

    # Count how many samples have default probabilities (orphans)
    orphan_count = np.sum(np.abs(test_probs[:, 0] - 0.5) < 0.001)
    logger.info(f"  Samples with predictions: {len(test_df) - orphan_count}/{len(test_df)}")
    logger.info(f"  Orphan samples (not in graph): {orphan_count}/{len(test_df)}")

    # FASE 4: Add priority predictions for hybrid ranking with KNN for orphans
    if is_dual_head and test_priority_preds is not None:
        test_df['priority_pred'] = test_priority_preds

        # Create hybrid score
        hybrid_score = np.copy(test_priority_preds)
        orphan_mask = np.abs(test_priority_preds - 0.5) < 0.001
        orphan_indices = np.where(orphan_mask)[0]
        in_graph_indices = np.where(~orphan_mask)[0]

        logger.info(f"  FASE 4: Priority pred available for ranking")
        logger.info(f"    In-graph samples: {len(in_graph_indices)}, Orphans: {len(orphan_indices)}")

        orphan_strategy = config.get('ranking', {}).get('orphan_strategy', {})
        use_orphan_strategy = orphan_strategy.get('enabled', True)

        if len(orphan_indices) > 0 and use_orphan_strategy:
            # Get embeddings and optional structural features from test_data
            test_embeddings = test_data['embeddings']
            test_struct_features = test_data.get('structural_features')

            orphan_embeddings = test_embeddings[orphan_indices]
            in_graph_embeddings = test_embeddings[in_graph_indices]
            in_graph_priorities = test_priority_preds[in_graph_indices]
            orphan_base_scores = test_probs[orphan_indices, 0]

            priority_fallback = test_df['priority_score'].values if 'priority_score' in test_df.columns else None

            orphan_priorities, orphan_stats = compute_orphan_scores(
                orphan_embeddings=orphan_embeddings,
                in_graph_embeddings=in_graph_embeddings,
                in_graph_scores=in_graph_priorities,
                orphan_base_scores=orphan_base_scores,
                strategy_config=orphan_strategy,
                orphan_structural_features=test_struct_features[orphan_indices] if test_struct_features is not None else None,
                in_graph_structural_features=test_struct_features[in_graph_indices] if test_struct_features is not None else None,
                orphan_priority_fallback=priority_fallback[orphan_indices] if priority_fallback is not None else None
            )

            hybrid_score[orphan_indices] = orphan_priorities
            logger.info(
                "    Enhanced orphan priority (KNN): "
                f"mean={orphan_stats['mean']:.4f}, std={orphan_stats['std']:.4f}, "
                f"min={orphan_stats['min']:.4f}, max={orphan_stats['max']:.4f}, "
                f"k={orphan_stats['k_neighbors']}, "
                f"fallbacks={orphan_stats['fallback_count']}"
            )
        elif len(orphan_indices) > 0:
            hybrid_score[orphan_mask] = test_probs[orphan_mask, 0] * 0.5

        test_df['hybrid_score'] = hybrid_score
        logger.info(f"    Priority pred stats: mean={test_priority_preds.mean():.4f}, max={test_priority_preds.max():.4f}")
    else:
        test_df['hybrid_score'] = test_probs[:, 0]  # Fallback to P(Fail)

    # CRITICAL: Use TE_Test_Result from original CSV for correct APFD
    if 'TE_Test_Result' not in test_df.columns:
        logger.error("❌ CRITICAL: TE_Test_Result column not found in test DataFrame!")
        logger.error("   This column is required for correct APFD calculation.")
        logger.error("   APFD should only count builds with TE_Test_Result == 'Fail'")
        logger.error("   Check if data_loader is preserving this column from test.csv")
        # Fallback: create from pass_vs_fail labels (not ideal but better than nothing)
        logger.warning("   Using fallback: mapping labels to TE_Test_Result")
        test_df['TE_Test_Result'] = test_data['labels'].map({0: 'Fail', 1: 'Pass'})
    else:
        logger.info(f"✅ TE_Test_Result column found with {len(test_df['TE_Test_Result'].unique())} unique values")
        logger.info(f"   Values: {test_df['TE_Test_Result'].value_counts().to_dict()}")

    # Create label_binary from TE_Test_Result (not from processed labels)
    test_df['label_binary'] = (test_df['TE_Test_Result'].astype(str).str.strip() == 'Fail').astype(int)
    logger.info(f"   label_binary distribution: {test_df['label_binary'].value_counts().to_dict()}")

    # Verify Build_ID exists
    if 'Build_ID' not in test_df.columns:
        logger.error("❌ CRITICAL: Build_ID column not found!")
        logger.error("   Cannot calculate APFD per build.")
    else:
        logger.info(f"✅ Build_ID column found: {test_df['Build_ID'].nunique()} unique builds")

    # Get results directory from config
    results_dir = config['output']['results_dir']
    os.makedirs(results_dir, exist_ok=True)

    # Generate prioritized CSV with ranks per build
    # FASE 4: Use hybrid_score for ranking (priority_pred + P(Fail) fallback)
    prioritized_path = os.path.join(results_dir, 'prioritized_test_cases.csv')
    test_df_with_ranks = generate_prioritized_csv(
        test_df,
        output_path=prioritized_path,
        probability_col='hybrid_score',  # FASE 4: Use hybrid score for ranking
        label_col='label_binary',
        build_col='Build_ID'
    )
    logger.info(f"✅ Prioritized test cases saved to: {prioritized_path}")

    # Calculate APFD per build
    apfd_path = os.path.join(results_dir, 'apfd_per_build.csv')
    apfd_results_df, apfd_summary = generate_apfd_report(
        test_df_with_ranks,
        method_name=config['experiment']['name'],
        test_scenario="v8_full_test",
        output_path=apfd_path
    )

    # Print summary
    print_apfd_summary(apfd_summary)

    # Log results
    if apfd_summary:
        logger.info(f"\n✅ APFD per-build report saved to: {apfd_path}")
        logger.info(f"📊 Mean APFD: {apfd_summary['mean_apfd']:.4f} (across {apfd_summary['total_builds']} builds)")

        # Verify expected 277 builds
        if apfd_summary['total_builds'] != 277:
            logger.warning(f"⚠️  WARNING: Expected 277 builds but got {apfd_summary['total_builds']}")
            logger.warning(f"   This may indicate incorrect filtering or data issues")
    else:
        logger.warning("⚠️  No builds with failures found - APFD cannot be calculated")

    # ==============================================================================
    # STEP 6: PROCESS FULL TEST.CSV (277 BUILDS) FOR FINAL APFD CALCULATION
    # ==============================================================================

    # Free val/test data no longer needed, but KEEP train_data for STEP 6 KNN imputation
    import gc as _gc
    del val_data, test_data
    del train_loader, val_loader, test_loader
    del test_df, test_probs, test_priority_preds
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Freed validation/test data before STEP 6 (keeping train_data for KNN)")

    if args.sample_size is not None:
        logger.info("\nSample-size mode detected - skipping full test.csv processing (STEP 6).")
    else:
        logger.info("\n"+"="*70)
        logger.info("STEP 6: PROCESSING FULL TEST.CSV FOR FINAL APFD")
        logger.info("="*70)

        try:
            # Load FULL test dataset (test.csv)
            logger.info("\n6.1: Loading FULL test.csv...")
            test_df_full = data_loader.load_full_test_dataset()

            logger.info(f"✅ Loaded full test.csv:")
            logger.info(f"   Total samples: {len(test_df_full)}")
            logger.info(f"   Total builds: {test_df_full['Build_ID'].nunique()}")

            builds_with_fail = test_df_full[test_df_full['TE_Test_Result'] == 'Fail']['Build_ID'].nunique()
            logger.info(f"   Builds with 'Fail': {builds_with_fail}")

            if builds_with_fail != 277:
                logger.warning(f"⚠️  WARNING: Expected 277 builds but found {builds_with_fail}")

            # Prepare labels early for evaluation and APFD
            test_df_full['label_binary'] = (test_df_full['TE_Test_Result'].astype(str).str.strip() == 'Fail').astype(int)
            # Training-consistent label: Pass=1 (positive_class), Fail=0 (matches data_loader encoding)
            test_df_full['label'] = 1 - test_df_full['label_binary']

            # Generate embeddings for full test set using a dedicated cache to avoid clobbering train/val cache
            logger.info("\n6.2: Generating semantic embeddings for full test set...")
            embedding_cfg = config.get('embedding', config.get('semantic', {}))
            base_cache_dir = embedding_cfg.get('cache_dir', 'cache')
            use_cache = embedding_cfg.get('use_cache', True)
            full_test_cache = os.path.join(base_cache_dir, 'full_test') if (base_cache_dir and use_cache) else None

            full_test_embedding_manager = EmbeddingManager(
                config,
                force_regenerate=False,
                cache_dir=full_test_cache
            )

            full_test_embeddings_dict = full_test_embedding_manager.get_embeddings(test_df_full, test_df_full)

            # Extract embeddings
            test_tc_embeddings_full = full_test_embeddings_dict['train_tc']  # Use 'train' key for the full test set
            test_commit_embeddings_full = full_test_embeddings_dict['train_commit']

            # Concatenate TC and Commit embeddings
            test_embeddings_full = np.concatenate([test_tc_embeddings_full, test_commit_embeddings_full], axis=1)
            logger.info(f"✅ Generated embeddings: {test_embeddings_full.shape}")

            # Free intermediate embedding arrays and SBERT model
            del full_test_embedding_manager, full_test_embeddings_dict
            del test_tc_embeddings_full, test_commit_embeddings_full
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Extract structural features for full test set
            logger.info("\n6.3: Extracting structural features for full test set...")

            # Use the already fitted extractor incrementally
            test_struct_full = extractor.transform_temporal(test_df_full)
            logger.info(f"✅ Extracted structural features: {test_struct_full.shape}")

            # Impute structural features BEFORE adding DeepOrder features
            tc_keys_test_full = test_df_full['TC_Key'].tolist()
            needs_imputation_full = extractor.get_imputation_mask(tc_keys_test_full)

            logger.info(f"   Samples needing imputation: {needs_imputation_full.sum()}/{len(tc_keys_test_full)}")

            if needs_imputation_full.sum() > 0:
                logger.info("   Imputing structural features using KNN...")
                # Get training structural features WITHOUT DeepOrder (first N base features only)
                n_base = len(extractor.get_feature_names())
                train_struct_base = train_struct[:, :n_base] if train_struct.shape[1] > n_base else train_struct
                test_struct_full, _ = impute_structural_features(
                    train_embeddings, train_struct_base, tc_keys_train,
                    test_embeddings_full, test_struct_full, tc_keys_test_full,
                    extractor.tc_history,
                    k_neighbors=10,
                    similarity_threshold=0.5,
                    verbose=False
                )

            gc.collect()

            # Add DeepOrder features for full test set AFTER imputation (if enabled)
            priority_config = config.get('priority_score', {})
            use_priority_scores = priority_config.get('enabled', True)

            if use_priority_scores:
                logger.info("\n6.3a: Computing DeepOrder features for full test set...")
                priority_generator = create_priority_score_generator(config)

                # Compute priority scores (using history from train+val+test splits)
                test_df_full, tc_history_full, test_deeporder_features_full = priority_generator.compute_priorities_for_dataframe(
                    test_df_full,
                    build_col='Build_ID',
                    tc_col='TC_Key',
                    result_col='TE_Test_Result',
                    fail_value='Fail',
                    pass_value='Pass',
                    initial_history=train_data.get('tc_history_for_step6'),  # Use carried over history
                    extract_features=True
                )

                # Concatenate with structural features
                logger.info(f"   Original structural features: {test_struct_full.shape}")
                logger.info(f"   DeepOrder features: {test_deeporder_features_full.shape}")
                test_struct_full = np.concatenate([test_struct_full, test_deeporder_features_full], axis=1)
                logger.info(f"   Combined features: {test_struct_full.shape}")

            # Map TC_Keys to global indices for subgraph extraction
            logger.info("\n6.3b: Mapping TC_Keys to global indices...")
            # Use same mapping as moto_v2: enumerate unique TC_Keys from training data
            tc_key_to_global_idx_full = {tc_key: idx for idx, tc_key in enumerate(train_data['df']['TC_Key'].unique())}
            global_indices_full = np.array([tc_key_to_global_idx_full.get(tc_key, -1) for tc_key in tc_keys_test_full])

            samples_in_graph = (global_indices_full != -1).sum()
            logger.info(f"   Samples in training graph: {samples_in_graph}/{len(global_indices_full)} ({100*samples_in_graph/len(global_indices_full):.1f}%)")
            logger.info(f"   Orphan samples: {len(global_indices_full) - samples_in_graph}/{len(global_indices_full)}")

            # Generate predictions on full test set using subgraph approach
            logger.info("\n6.4: Generating predictions on full test set...")

            # Create DataLoader with global indices
            test_dataset_full = torch.utils.data.TensorDataset(
                torch.FloatTensor(test_embeddings_full),
                torch.FloatTensor(test_struct_full),
                torch.LongTensor(test_df_full['label'].values),  # Training-consistent: Pass=1, Fail=0
                torch.LongTensor(global_indices_full)  # Global indices
            )

            test_loader_full = torch.utils.data.DataLoader(
                test_dataset_full,
                batch_size=config['training']['batch_size'],
                shuffle=False
            )

            # Use evaluate() function with subgraph extraction
            # For dual-head model, we get both probabilities AND priority predictions
            # is_dual_head=True to get priority predictions for ranking
            _, _, all_probs_full, all_priority_preds_full = evaluate(
                model, test_loader_full, criterion, device, edge_index, edge_weights,
                train_structural_features, num_nodes_global,
                return_full_probs=True, dataset_size=len(test_df_full),
                phylo_embeddings=phylo_embeddings, phylo_edge_index=phylo_edge_index,
                phylo_path_lengths=phylo_path_lengths,
                is_dual_head=is_dual_head  # Use dual-head mode if model is dual-head
            )

            logger.info(f"✅ Predictions generated: {all_probs_full.shape}")
            orphan_count_full = np.sum(np.abs(all_probs_full[:, 0] - 0.5) < 0.001)
            logger.info(f"   Samples with actual predictions: {len(all_probs_full) - orphan_count_full}")
            logger.info(f"   Orphan samples (default prob 0.5): {orphan_count_full}")

            # DIAGNOSTIC: Priority prediction distribution
            if all_priority_preds_full is not None:
                logger.info("\n" + "="*70)
                logger.info("PRIORITY PREDICTION DIAGNOSTIC")
                logger.info("="*70)
                in_graph_mask = np.abs(all_priority_preds_full - 0.5) >= 0.001
                logger.info(f"   In-graph samples (priority_pred != 0.5): {in_graph_mask.sum()}")
                logger.info(f"   Orphan samples (priority_pred = 0.5): {(~in_graph_mask).sum()}")
                if in_graph_mask.sum() > 0:
                    in_graph_preds = all_priority_preds_full[in_graph_mask]
                    logger.info(f"   In-graph predictions:")
                    logger.info(f"     Mean: {in_graph_preds.mean():.6f}")
                    logger.info(f"     Std:  {in_graph_preds.std():.6f}")
                    logger.info(f"     Min:  {in_graph_preds.min():.6f}")
                    logger.info(f"     Max:  {in_graph_preds.max():.6f}")
                    # Check if predictions have meaningful variance
                    if in_graph_preds.std() < 0.01:
                        logger.warning("   ⚠️  WARNING: Priority predictions have very low variance!")
                        logger.warning("   ⚠️  This indicates the regression head is not learning properly.")
                logger.info("="*70)

            # Prepare DataFrame for APFD
            logger.info("\n6.5: Preparing data for APFD calculation...")

            # P(Fail) = probabilities[:, 0] (class 0 with pass_vs_fail)
            failure_probs_full = all_probs_full[:, 0]
            test_df_full['probability'] = failure_probs_full

            # FASE 4: Hybrid ranking strategy
            # P(Fail) as PRIMARY signal, priority_pred as SECONDARY boost
            # Historical priority_score (DeepOrder) as additional boost for TCs with history
            if is_dual_head and all_priority_preds_full is not None:
                logger.info("\n📊 FASE 4: Using HYBRID ranking strategy")
                logger.info("   - P(Fail) as PRIMARY signal (ensures good base ranking)")
                logger.info("   - priority_pred as SECONDARY boost (adds regression insight)")
                logger.info("   - Historical priority_score boost for TCs with execution history")
                logger.info("   - KNN for orphan samples")

                # Parameters for hybrid scoring from config
                ranking_config = config.get('ranking', {})
                lambda_pfail = ranking_config.get('lambda_pfail', 0.5)  # Weight for P(Fail)
                use_historical_boost = ranking_config.get('use_historical_boost', True)
                historical_boost_weight = ranking_config.get('historical_boost_weight', 0.3)

                # Identify orphans (priority_pred = 0.5 default)
                orphan_mask = np.abs(all_priority_preds_full - 0.5) < 0.001
                orphan_indices = np.where(orphan_mask)[0]
                in_graph_indices = np.where(~orphan_mask)[0]

                logger.info(f"   In-graph samples: {len(in_graph_indices)}")
                logger.info(f"   Orphan samples: {len(orphan_indices)}")
                logger.info(f"   Lambda (P(Fail) weight): {lambda_pfail}")
                logger.info(f"   Historical boost: {use_historical_boost}, weight={historical_boost_weight}")

                # Initialize hybrid score with P(Fail) as base
                hybrid_score = np.copy(failure_probs_full)

                # Get historical priority_score (DeepOrder) for boost
                historical_priority = test_df_full['priority_score'].values if 'priority_score' in test_df_full.columns else None

                orphan_strategy = config.get('ranking', {}).get('orphan_strategy', {})
                use_orphan_strategy = orphan_strategy.get('enabled', True)

                # For in-graph samples: blend P(Fail) with priority_pred
                if len(in_graph_indices) > 0:
                    in_graph_pfail = failure_probs_full[in_graph_indices]
                    in_graph_priority = all_priority_preds_full[in_graph_indices]

                    # Check if priority_pred has meaningful variance
                    priority_std = in_graph_priority.std()
                    if priority_std > 0.01:
                        # Normalize priority_pred to same scale as P(Fail)
                        priority_min = in_graph_priority.min()
                        priority_max = in_graph_priority.max()
                        if priority_max > priority_min:
                            normalized_priority = (in_graph_priority - priority_min) / (priority_max - priority_min)
                        else:
                            normalized_priority = in_graph_priority

                        # Blend P(Fail) with normalized priority_pred
                        hybrid_score[in_graph_indices] = (
                            lambda_pfail * in_graph_pfail +
                            (1 - lambda_pfail) * normalized_priority
                        )
                        logger.info(f"   ✅ Using blended ranking (priority_pred has variance: std={priority_std:.4f})")
                    else:
                        # Priority_pred has no useful information - use P(Fail) only
                        logger.warning(f"   ⚠️ Priority_pred has low variance (std={priority_std:.6f}) - using P(Fail) only for in-graph")
                        hybrid_score[in_graph_indices] = in_graph_pfail

                # Apply historical priority_score as PRIMARY signal for TCs with execution history
                # Analysis shows historical failure rate alone achieves APFD > 0.75!
                if use_historical_boost and historical_priority is not None:
                    # Normalize historical priority to [0, 1] range
                    hist_max = historical_priority.max()
                    if hist_max > 0:
                        normalized_hist = historical_priority / hist_max

                        # Samples with past failure history - use historical as PRIMARY signal
                        has_failure_history_mask = historical_priority > 0
                        n_with_failure_history = has_failure_history_mask.sum()

                        if n_with_failure_history > 0:
                            # For samples with failure history: historical priority is DOMINANT
                            actual_boost_weight = min(historical_boost_weight * 1.5, 0.8)  # Cap at 80%
                            hybrid_score[has_failure_history_mask] = (
                                (1 - actual_boost_weight) * hybrid_score[has_failure_history_mask] +
                                actual_boost_weight * normalized_hist[has_failure_history_mask]
                            )
                            logger.info(f"   ✅ Historical priority as PRIMARY signal for {n_with_failure_history} samples")
                            logger.info(f"      Actual blend weight: {actual_boost_weight:.2f} historical, {1-actual_boost_weight:.2f} model")
                            logger.info(f"      Historical priority stats: mean={historical_priority[has_failure_history_mask].mean():.4f}, max={hist_max:.4f}")

                # For orphan samples: use KNN with P(Fail) blending
                if len(orphan_indices) > 0 and use_orphan_strategy:
                    logger.info("   Computing enhanced KNN-based score for orphans...")

                    priority_fallback = test_df_full['priority_score'].values if 'priority_score' in test_df_full.columns else None

                    orphan_scores, orphan_stats = compute_orphan_scores(
                        orphan_embeddings=test_embeddings_full[orphan_indices],
                        in_graph_embeddings=test_embeddings_full[in_graph_indices],
                        in_graph_scores=hybrid_score[in_graph_indices],
                        orphan_base_scores=failure_probs_full[orphan_indices],
                        strategy_config=orphan_strategy,
                        orphan_structural_features=test_struct_full[orphan_indices] if test_struct_full is not None else None,
                        in_graph_structural_features=test_struct_full[in_graph_indices] if test_struct_full is not None else None,
                        orphan_priority_fallback=priority_fallback[orphan_indices] if priority_fallback is not None else None
                    )

                    hybrid_score[orphan_indices] = orphan_scores

                    logger.info(
                        f"   Orphan score stats: mean={orphan_stats['mean']:.4f}, "
                        f"max={orphan_stats['max']:.4f}, min={orphan_stats['min']:.4f}, "
                        f"std={orphan_stats['std']:.4f}, k={orphan_stats['k_neighbors']}, "
                        f"fallbacks={orphan_stats['fallback_count']}"
                    )

                elif len(orphan_indices) > 0:
                    # No KNN or strategy disabled - use P(Fail) directly for orphans
                    hybrid_score[orphan_indices] = failure_probs_full[orphan_indices]
                    logger.info("   No in-graph samples available or orphan strategy disabled - using P(Fail) directly for orphans")

                test_df_full['priority_pred'] = all_priority_preds_full
                test_df_full['hybrid_score'] = hybrid_score

                logger.info(f"   Final priority pred stats: mean={all_priority_preds_full.mean():.4f}, "
                           f"max={all_priority_preds_full.max():.4f}")
                logger.info(f"   Final hybrid score stats: mean={hybrid_score.mean():.4f}, "
                           f"max={hybrid_score.max():.4f}")
            else:
                # Non-dual-head model: use P(Fail) with optional KNN for orphans
                ranking_config = config.get('ranking', {})
                orphan_config = ranking_config.get('orphan_strategy', {})
                use_knn_orphans = orphan_config.get('enabled', False)

                if use_knn_orphans:
                    logger.info("\n📊 KNN-ENHANCED RANKING (for orphan samples)")
                    logger.info("   - In-graph samples: use P(Fail) directly")
                    logger.info("   - Orphan samples: use KNN with P(Fail) of neighbors")

                    # Identify orphans (P(Fail) = 0.5 is the default for samples not in graph)
                    orphan_mask = np.abs(failure_probs_full - 0.5) < 0.001
                    orphan_indices = np.where(orphan_mask)[0]
                    in_graph_indices = np.where(~orphan_mask)[0]

                    logger.info(f"   In-graph samples: {len(in_graph_indices)}")
                    logger.info(f"   Orphan samples: {len(orphan_indices)}")

                    # Initialize with P(Fail)
                    hybrid_score = np.copy(failure_probs_full)

                    # Apply KNN for orphans if there are both orphans and in-graph samples
                    if len(orphan_indices) > 0 and len(in_graph_indices) > 0:
                        logger.info("   Computing enhanced KNN-based P(Fail) for orphans...")

                        priority_fallback = test_df_full['priority_score'].values if 'priority_score' in test_df_full.columns else None

                        orphan_scores, orphan_stats = compute_orphan_scores(
                            orphan_embeddings=test_embeddings_full[orphan_indices],
                            in_graph_embeddings=test_embeddings_full[in_graph_indices],
                            in_graph_scores=failure_probs_full[in_graph_indices],
                            orphan_base_scores=failure_probs_full[orphan_indices],
                            strategy_config=orphan_config,
                            orphan_structural_features=test_struct_full[orphan_indices] if test_struct_full is not None else None,
                            in_graph_structural_features=test_struct_full[in_graph_indices] if test_struct_full is not None else None,
                            orphan_priority_fallback=priority_fallback[orphan_indices] if priority_fallback is not None else None
                        )

                        hybrid_score[orphan_indices] = orphan_scores

                        logger.info(
                            f"   KNN orphan P(Fail) stats: mean={orphan_stats['mean']:.4f}, "
                            f"max={orphan_stats['max']:.4f}, min={orphan_stats['min']:.4f}, "
                            f"std={orphan_stats['std']:.4f}, k={orphan_stats['k_neighbors']}, "
                            f"fallbacks={orphan_stats['fallback_count']}"
                        )
                    else:
                        logger.info("   No orphans to process or no in-graph samples available")

                    test_df_full['hybrid_score'] = hybrid_score
                else:
                    # No KNN - just use P(Fail) directly
                    test_df_full['hybrid_score'] = failure_probs_full
                    logger.info("   Using P(Fail) for ranking (KNN orphan strategy disabled)")

            logger.info(f"   Failures (TE_Test_Result=='Fail'): {test_df_full['label_binary'].sum()}")
            logger.info(f"   Passes: {(test_df_full['label_binary'] == 0).sum()}")

            # Generate prioritized CSV with ranks per build
            logger.info("\n6.6: Generating prioritized test cases CSV...")

            # FASE 4: Use hybrid_score for ranking (priority_pred + P(Fail) fallback)
            prioritized_path_full = os.path.join(results_dir, 'prioritized_test_cases_FULL_testcsv.csv')
            test_df_full_with_ranks = generate_prioritized_csv(
                test_df_full,
                output_path=prioritized_path_full,
                probability_col='hybrid_score',  # FASE 4: Use hybrid score for ranking
                label_col='label_binary',
                build_col='Build_ID'
            )
            logger.info(f"✅ Prioritized test cases (FULL) saved to: {prioritized_path_full}")

            # Calculate APFD per build using hybrid ranking
            logger.info("\n6.7: Calculating APFD per build on FULL test.csv (HYBRID ranking)...")

            apfd_path_full = os.path.join(results_dir, 'apfd_per_build_FULL_testcsv.csv')
            method_name_full = f"{config['experiment']['name']}_FULL_testcsv"

            apfd_results_df_full, apfd_summary_full = generate_apfd_report(
                test_df_full_with_ranks,
                method_name=method_name_full,
                test_scenario="full_test_csv_277_builds_HYBRID",
                output_path=apfd_path_full
            )

            # Also calculate APFD using only P(Fail) for comparison
            if is_dual_head and 'priority_pred' in test_df_full.columns:
                logger.info("\n📊 COMPARISON: APFD with P(Fail) only vs Hybrid ranking")

                # Generate ranks using P(Fail) only
                test_df_pfail = test_df_full.copy()
                test_df_pfail_with_ranks = generate_prioritized_csv(
                    test_df_pfail,
                    output_path=None,  # Don't save
                    probability_col='probability',  # P(Fail)
                    label_col='label_binary',
                    build_col='Build_ID'
                )
                _, apfd_summary_pfail = generate_apfd_report(
                    test_df_pfail_with_ranks,
                    method_name=f"{config['experiment']['name']}_PFail",
                    test_scenario="full_test_csv_277_builds_PFail",
                    output_path=None
                )

                # Generate ranks using pure priority_pred (no fallback)
                test_df_priority = test_df_full.copy()
                test_df_priority_with_ranks = generate_prioritized_csv(
                    test_df_priority,
                    output_path=None,
                    probability_col='priority_pred',  # Pure priority_pred
                    label_col='label_binary',
                    build_col='Build_ID'
                )
                _, apfd_summary_priority = generate_apfd_report(
                    test_df_priority_with_ranks,
                    method_name=f"{config['experiment']['name']}_PriorityPred",
                    test_scenario="full_test_csv_277_builds_PriorityPred",
                    output_path=None
                )

                logger.info("\n" + "="*70)
                logger.info("RANKING STRATEGY COMPARISON")
                logger.info("="*70)
                logger.info(f"  P(Fail) only:      APFD = {apfd_summary_pfail['mean_apfd']:.4f}")
                logger.info(f"  Priority Pred:     APFD = {apfd_summary_priority['mean_apfd']:.4f}")
                logger.info(f"  Hybrid (FASE 4):   APFD = {apfd_summary_full['mean_apfd']:.4f}")
                logger.info("="*70)

            # Print APFD summary
            logger.info("\n" + "="*70)
            logger.info("FINAL APFD RESULTS - FULL TEST.CSV (277 BUILDS)")
            logger.info("="*70)
            print_apfd_summary(apfd_summary_full)

            # Validation
            logger.info("\n" + "="*70)
            logger.info("VALIDATION")
            logger.info("="*70)

            if apfd_summary_full and apfd_summary_full['total_builds'] == 277:
                logger.info("✅ SUCCESS: Found exactly 277 builds with failures!")
                logger.info(f"✅ Mean APFD: {apfd_summary_full['mean_apfd']:.4f}")
            else:
                builds_found = apfd_summary_full['total_builds'] if apfd_summary_full else 0
                logger.warning(f"⚠️  WARNING: Expected 277 builds but found {builds_found}")

            logger.info(f"\n✅ All results saved to: {results_dir}/")
            logger.info(f"   - prioritized_test_cases.csv (test split)")
            logger.info(f"   - apfd_per_build.csv (test split)")
            logger.info(f"   - prioritized_test_cases_FULL_testcsv.csv (all 277 builds)")
            logger.info(f"   - apfd_per_build_FULL_testcsv.csv (all 277 builds)")

        except Exception as e:
            logger.error(f"\n❌ ERROR processing full test.csv: {e}")
            logger.error("   Continuing with split test results only...")
            import traceback
            traceback.print_exc()

    # ==============================================================================
    # TRAINING COMPLETE
    # ==============================================================================

    logger.info("\n"+"="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"Best Val F1: {best_val_f1:.4f}")
    logger.info(f"Test F1: {test_metrics['f1_macro']:.4f}")

    if apfd_summary:
        logger.info(f"Mean APFD (test split): {apfd_summary.get('mean_apfd', 0.0):.4f}")

    try:
        if apfd_summary_full:
            logger.info(f"Mean APFD (FULL test.csv, 277 builds): {apfd_summary_full.get('mean_apfd', 0.0):.4f}")
    except:
        pass


if __name__ == '__main__':
    main()
