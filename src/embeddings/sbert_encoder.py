"""
SBERT Encoder using all-mpnet-base-v2

Lightweight and stable alternative to Qodo-Embed-1-1.5B
- 13x smaller (110M vs 1.5B parameters)
- 15x less VRAM (~200MB vs 3GB)
- Excellent performance for text embeddings
- No NVML/memory fragmentation issues
"""

import logging
import gc
from typing import Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SBERTEncoder:
    """
    Semantic embeddings encoder using SBERT (all-mpnet-base-v2)

    This is a lightweight, stable alternative to Qodo-Embed-1-1.5B:
    - 110M parameters (vs 1.5B)
    - 768 embedding dimensions (vs 1536)
    - ~200MB VRAM (vs ~3GB)
    - Optimized for general text (perfect for test cases and commit messages)
    """

    def __init__(self, config: Dict, device: str = 'cuda'):
        """
        Initialize SBERTEncoder

        Args:
            config: Configuration dictionary
            device: Device to use ('cuda' or 'cpu')
        """
        self.config = config
        self.embedding_config = config.get('embedding', config.get('semantic', config))

        # Model configuration
        self.model_name = self.embedding_config.get('model_name', 'sentence-transformers/all-mpnet-base-v2')
        self.batch_size = self.embedding_config.get('batch_size', 64)  # Can use larger batches!
        self.max_length = self.embedding_config.get('max_length', 384)  # SBERT supports up to 512
        self.normalize = self.embedding_config.get('normalize_embeddings', True)

        # Device setup
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = 'cpu'

        self.device = device
        logger.info(f"Initializing SBERT encoder on {self.device}")
        logger.info(f"Model: {self.model_name}")

        # Load model
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"✓ Loaded SBERT model on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"✓ Embedding dimension: {self.embedding_dim}")
        logger.info(f"✓ Batch size: {self.batch_size}")
        logger.info(f"✓ Max sequence length: {self.max_length}")

    def encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode a batch of texts to embeddings

        Args:
            texts: List of text strings

        Returns:
            embeddings: [N, embedding_dim] numpy array
        """
        if not texts:
            return np.array([])

        # Encode with SBERT
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            device=self.device,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize
        )

        return embeddings

    def encode_texts_chunked(
        self,
        texts: List[str],
        chunk_size: Optional[int] = None,
        desc: str = "Encoding"
    ) -> np.ndarray:
        """
        Encode texts in chunks with progress bar

        Args:
            texts: List of text strings
            chunk_size: Number of texts per chunk (default: batch_size * 10)
            desc: Description for progress bar

        Returns:
            embeddings: [N, embedding_dim] numpy array
        """
        if not texts:
            return np.array([])

        if chunk_size is None:
            chunk_size = self.batch_size * 10  # Process 10 batches at a time

        num_chunks = (len(texts) + chunk_size - 1) // chunk_size

        logger.info(f"Encoding {len(texts)} texts in {num_chunks} chunks (chunk_size={chunk_size})")

        # Pre-allocate output array to avoid doubling memory with vstack
        embeddings = np.empty((len(texts), self.embedding_dim), dtype=np.float32)
        offset = 0

        for i in tqdm(range(0, len(texts), chunk_size), desc=desc):
            chunk = texts[i:i+chunk_size]

            # Encode chunk
            chunk_embeddings = self.encode_texts_batch(chunk)
            n = chunk_embeddings.shape[0]
            embeddings[offset:offset+n] = chunk_embeddings
            offset += n

            # Clear cache periodically (every 10 chunks)
            if self.device == 'cuda' and (i // chunk_size) % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        logger.info(f"✓ Encoded {len(texts)} texts → shape: {embeddings.shape}")

        return embeddings

    def get_embedding_dim(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim

    def clear_cache(self):
        """Clear GPU cache"""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("Cleared CUDA cache")


def create_sbert_encoder(config: Dict, device: str = 'cuda') -> SBERTEncoder:
    """
    Factory function to create SBERT encoder

    Args:
        config: Configuration dictionary
        device: Device to use

    Returns:
        encoder: SBERTEncoder instance
    """
    return SBERTEncoder(config, device)
