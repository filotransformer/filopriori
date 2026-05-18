"""
Embeddings Module

Provides SBERT-based semantic embeddings with intelligent caching.

Main Components:
- SBERTEncoder: Lightweight encoder using all-mpnet-base-v2
- EmbeddingCache: Automatic cache management
- EmbeddingManager: High-level interface with auto-caching
"""

from .sbert_encoder import SBERTEncoder
from .embedding_cache import EmbeddingCache
from .embedding_manager import EmbeddingManager

__all__ = [
    'SBERTEncoder',
    'EmbeddingCache',
    'EmbeddingManager',
]
