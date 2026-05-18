# V7 imports (disabled for V8 to avoid torch_geometric dependency)
# from .dual_stream import DualStreamPhylogeneticTransformer
# from .cross_attention import CrossAttentionFusion

# V8 imports
# Import directly: from models.dual_stream_v8 import DualStreamModelV8, create_model_v8

# V9 Phylogenetic Imports
# Import directly to avoid dependency issues:
#   from src.models.phylo_encoder import PhyloEncoder, create_phylo_encoder
#   from src.models.phylogenetic_dual_stream import PhylogeneticDualStreamModel, create_phylogenetic_model

# Unified Model Factory (recommended for V9+)
# This factory auto-selects between V8 and Phylogenetic models based on config:
#   from src.models.model_factory import create_model

__all__ = [
    # Unified Factory (V9+)
    # 'create_model',  # Auto-selects model based on config['type']

    # V8 (direct imports recommended)
    # 'DualStreamModelV8', 'create_model_v8',

    # V9 Phylogenetic (direct imports recommended)
    # 'PhyloEncoder', 'PhylogeneticDistanceKernel', 'GGNNLayer',
    # 'PhylogeneticDualStreamModel', 'HierarchicalAttention', 'PhylogeneticRegularization',
]
