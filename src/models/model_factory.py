"""
Model Factory for Filo-Priori

Unified factory for creating different model types based on configuration.

Supported model types:
- 'dual_stream_v8': Original DualStreamModelV8 (GATv2 + Semantic)
- 'dual_stream' or 'dual_stream_v8': Alias for dual_stream_v8
- 'dual_head': DualHeadModel (Classification + Regression heads, DeepOrder-inspired)
- 'phylogenetic_dual_stream': PhylogeneticDualStreamModel (GGNN + Hierarchical Attention)

Author: Filo-Priori V9 Team
Date: November 2025
"""

import torch.nn as nn
from typing import Dict, Union
import logging

logger = logging.getLogger(__name__)


def create_model(config: Dict) -> nn.Module:
    """
    Unified factory function to create models from config.

    This function detects the model type from config and creates the appropriate model.

    Args:
        config: Model configuration dictionary. Must contain 'type' key to specify
               which model architecture to use.

    Returns:
        Model instance (DualStreamModelV8 or PhylogeneticDualStreamModel)

    Example configs:

    1. DualStreamModelV8 (default):
        {
            'type': 'dual_stream_v8',
            'semantic': {'input_dim': 1536, 'hidden_dim': 256},
            'structural': {'input_dim': 10, 'hidden_dim': 256},
            ...
        }

    2. PhylogeneticDualStreamModel:
        {
            'type': 'phylogenetic_dual_stream',
            'semantic': {'input_dim': 1536, 'hidden_dim': 256},
            'structural': {'input_dim': 10, 'hidden_dim': 256},
            'phylo': {'input_dim': 768, 'num_layers': 3},
            'hierarchical_attention': {'enabled': True},
            ...
        }
    """
    model_type = config.get('type', 'dual_stream_v8')

    logger.info(f"Creating model of type: {model_type}")

    if model_type == 'phylogenetic_dual_stream':
        return _create_phylogenetic_model(config)
    elif model_type == 'dual_head':
        return _create_dual_head_model(config)
    elif model_type in ['dual_stream_v8', 'dual_stream', 'v8', 'default']:
        return _create_v8_model(config)
    else:
        logger.warning(f"Unknown model type '{model_type}', falling back to dual_stream_v8")
        return _create_v8_model(config)


def _create_v8_model(config: Dict) -> nn.Module:
    """
    Create DualStreamModelV8 from config.

    Args:
        config: Model configuration

    Returns:
        DualStreamModelV8 instance
    """
    from .dual_stream_v8 import create_model_v8

    logger.info("="*70)
    logger.info("CREATING MODEL: DualStreamModelV8")
    logger.info("="*70)

    return create_model_v8(config)


def _create_dual_head_model(config: Dict) -> nn.Module:
    """
    Create DualHeadModel from config.

    DualHeadModel has two heads:
    - Classification head: Focal Loss for Fail/Pass
    - Regression head: MSE Loss for priority score

    Args:
        config: Model configuration

    Returns:
        DualHeadModel instance
    """
    from .dual_head_model import create_dual_head_model

    logger.info("="*70)
    logger.info("CREATING MODEL: DualHeadModel (DeepOrder-inspired)")
    logger.info("="*70)
    logger.info("  - Classification Head: Fail/Pass with Focal Loss")
    logger.info("  - Regression Head: Priority Score with MSE Loss")

    return create_dual_head_model(config)


def _create_phylogenetic_model(config: Dict) -> nn.Module:
    """
    Create PhylogeneticDualStreamModel from config.

    Args:
        config: Model configuration

    Returns:
        PhylogeneticDualStreamModel instance
    """
    from .phylogenetic_dual_stream import create_phylogenetic_model

    logger.info("="*70)
    logger.info("CREATING MODEL: PhylogeneticDualStreamModel")
    logger.info("="*70)

    # Extract phylogenetic-specific config
    phylo_config = config.get('phylo', {})
    hierarchical_config = config.get('hierarchical_attention', {})

    # Build the full config for PhylogeneticDualStreamModel
    phylo_model_config = {
        'semantic': config.get('semantic', {}),
        'structural': config.get('structural', {}),
        'phylo': phylo_config,
        'fusion': config.get('fusion', {}),
        'classifier': config.get('classifier', {}),
        'use_phylo_encoder': phylo_config.get('enabled', True),
        'use_hierarchical_attention': hierarchical_config.get('enabled', True),
        'num_classes': config.get('num_classes', 2)
    }

    # Log configuration
    logger.info("Phylogenetic configuration:")
    logger.info(f"  - PhyloEncoder enabled: {phylo_model_config['use_phylo_encoder']}")
    logger.info(f"  - Hierarchical Attention enabled: {phylo_model_config['use_hierarchical_attention']}")
    if phylo_config:
        logger.info(f"  - GGNN layers: {phylo_config.get('num_layers', 3)}")
        logger.info(f"  - Distance kernel: {phylo_config.get('use_distance_kernel', True)}")
        logger.info(f"  - Decay factor: {phylo_config.get('decay_factor', 0.9)}")

    return create_phylogenetic_model(phylo_model_config)


def get_model_info(config: Dict) -> Dict:
    """
    Get information about what model would be created from config.

    Useful for logging and debugging without actually creating the model.

    Args:
        config: Model configuration

    Returns:
        Dictionary with model information
    """
    model_type = config.get('type', 'dual_stream_v8')

    info = {
        'type': model_type,
        'semantic_input_dim': config.get('semantic', {}).get('input_dim', 1536),
        'structural_input_dim': config.get('structural', {}).get('input_dim', 10),
        'hidden_dim': config.get('semantic', {}).get('hidden_dim', 256),
        'num_classes': config.get('num_classes', 2)
    }

    if model_type == 'phylogenetic_dual_stream':
        phylo_config = config.get('phylo', {})
        hierarchical_config = config.get('hierarchical_attention', {})

        info.update({
            'phylo_enabled': phylo_config.get('enabled', True),
            'phylo_input_dim': phylo_config.get('input_dim', 768),
            'phylo_num_layers': phylo_config.get('num_layers', 3),
            'phylo_decay_factor': phylo_config.get('decay_factor', 0.9),
            'hierarchical_attention_enabled': hierarchical_config.get('enabled', True)
        })

    return info


__all__ = ['create_model', 'get_model_info']
