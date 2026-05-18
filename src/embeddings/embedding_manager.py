"""
Embedding Manager with Automatic Caching

High-level interface for embedding generation with:
- Automatic cache detection and reuse
- Intelligent regeneration when data changes
- Force regeneration option
- Progress tracking and logging
"""

import gc
import logging
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from .embedding_cache import EmbeddingCache
from .sbert_encoder import SBERTEncoder

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages embedding generation with intelligent caching

    Usage:
        manager = EmbeddingManager(config, force_regenerate=False)
        embeddings = manager.get_embeddings(train_df, test_df)
    """

    def __init__(self, config: Dict, force_regenerate: bool = False, cache_dir: str = 'cache'):
        """
        Initialize embedding manager

        Args:
            config: Configuration dictionary
            force_regenerate: If True, regenerate embeddings even if cached
            cache_dir: Directory for cache storage
        """
        self.config = config
        self.force_regenerate = force_regenerate
        self.cache = EmbeddingCache(cache_dir=cache_dir) if cache_dir is not None else None

        # Get embedding config
        self.embedding_config = config.get('embedding', config.get('semantic', {}))
        self.model_name = self.embedding_config.get('model_name', 'sentence-transformers/all-mpnet-base-v2')
        self.device = self.embedding_config.get('device', 'cuda')

    def _prepare_tc_texts(self, df: pd.DataFrame) -> list:
        """Prepare test case texts from dataframe"""
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

    def _prepare_commit_texts(self, df: pd.DataFrame) -> list:
        """Prepare commit texts from dataframe"""
        texts = []
        for _, row in df.iterrows():
            msg = row.get('commit_processed', row.get('commit_msg', row.get('message', '')))
            diff = row.get('commit_diff', row.get('diff', ''))

            if msg and diff:
                # Truncate diff to 2000 chars (SBERT max is 512 tokens, ~2000 chars)
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

    def _generate_embeddings(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple:
        logger.info("="*70)
        logger.info("GENERATING EMBEDDINGS (OPTIMIZED DEDUPLICATION)")
        logger.info("="*70)

        logger.info(f"Initializing encoder: {self.model_name}")
        encoder = SBERTEncoder(self.config, device=self.device)

        batch_size = self.embedding_config.get('batch_size', 128)
        chunk_size = batch_size * 10

        def encode_optimized(df, text_func, desc, fname):
            texts = text_func(df)
            unique_texts = list(set(texts))
            logger.info(f"  {desc}: {len(texts)} total rows, {len(unique_texts)} unique texts")
            
            unique_embs = encoder.encode_texts_chunked(unique_texts, chunk_size=chunk_size, desc=f"Unique {desc}")
            
            text_to_idx = {t: i for i, t in enumerate(unique_texts)}
            indices = np.array([text_to_idx[t] for t in texts], dtype=np.int32)
            
            if len(texts) > 200000:
                import os
                cache_dir = self.cache.cache_dir if self.cache else 'cache'
                os.makedirs(cache_dir, exist_ok=True)
                mm_path = os.path.join(cache_dir, fname)
                logger.info(f"  Using np.memmap for {desc} ({len(texts)} samples) -> {mm_path}")
                out_arr = np.memmap(mm_path, dtype=np.float32, mode='w+', shape=(len(texts), unique_embs.shape[1]))
                c_size = 100000
                for i in range(0, len(texts), c_size):
                    end = min(i + c_size, len(texts))
                    out_arr[i:end] = unique_embs[indices[i:end]]
                out_arr.flush()
                return out_arr
            else:
                return unique_embs[indices]

        logger.info("Encoding Train TCs...")
        train_tc_emb = encode_optimized(train_df, self._prepare_tc_texts, "Train TCs", 'tmp_train_tc.dat')

        logger.info("Encoding Test TCs...")
        test_tc_emb = encode_optimized(test_df, self._prepare_tc_texts, "Test TCs", 'tmp_test_tc.dat')

        logger.info("Encoding Train Commits...")
        train_commit_emb = encode_optimized(train_df, self._prepare_commit_texts, "Train Commits", 'tmp_train_commit.dat')

        logger.info("Encoding Test Commits...")
        test_commit_emb = encode_optimized(test_df, self._prepare_commit_texts, "Test Commits", 'tmp_test_commit.dat')

        embedding_dim = encoder.get_embedding_dim()

        del encoder; gc.collect()
        logger.info("="*70)

        return train_tc_emb, test_tc_emb, train_commit_emb, test_commit_emb, embedding_dim, self.model_name

    def get_embeddings(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get embeddings (from cache or generate new)

        Automatically:
        1. Checks if cache exists and is valid
        2. Reuses cache if available and valid
        3. Regenerates if cache invalid or force_regenerate=True
        4. Saves newly generated embeddings to cache

        Args:
            train_df: Training dataframe
            test_df: Test dataframe

        Returns:
            embeddings: Dictionary with keys:
                - 'train_tc': Train TC embeddings
                - 'test_tc': Test TC embeddings
                - 'train_commit': Train commit embeddings
                - 'test_commit': Test commit embeddings
                - 'embedding_dim': Embedding dimension
                - 'model_name': Model name
        """
        # Check cache
        use_cache = False

        if self.cache is None:
            logger.info("Cache disabled - generating embeddings")
        elif self.force_regenerate:
            logger.info("Force regenerate enabled - ignoring cache")
        elif self.cache.exists():
            if self.cache.is_valid(train_df, test_df):
                logger.info("Valid cache found - loading embeddings from cache")
                use_cache = True
            else:
                logger.info("Cache found but invalid (data changed) - regenerating")
        else:
            logger.info("No cache found - generating embeddings")

        # Load or generate
        if use_cache:
            try:
                train_tc_emb, test_tc_emb, train_commit_emb, test_commit_emb, embedding_dim, model_name = self.cache.load()
            except (FileNotFoundError, Exception) as e:
                logger.warning(f"Cache load failed ({e}), regenerating embeddings...")
                use_cache = False
        if not use_cache:
            train_tc_emb, test_tc_emb, train_commit_emb, test_commit_emb, embedding_dim, model_name = self._generate_embeddings(train_df, test_df)

            # Save to cache (if enabled)
            if self.cache is not None:
                self.cache.save(
                    train_tc_emb, test_tc_emb, train_commit_emb, test_commit_emb,
                    embedding_dim, model_name, train_df, test_df
                )

        # Return as dictionary
        return {
            'train_tc': train_tc_emb,
            'test_tc': test_tc_emb,
            'train_commit': train_commit_emb,
            'test_commit': test_commit_emb,
            'embedding_dim': embedding_dim,
            'model_name': model_name
        }

    def clear_cache(self):
        """Clear embedding cache"""
        if self.cache is not None:
            self.cache.clear()

    def cache_info(self) -> str:
        """Get cache information"""
        if self.cache is not None:
            return self.cache.info()
        else:
            return "Cache disabled"
