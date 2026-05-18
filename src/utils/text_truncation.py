"""
Text Truncation Utilities for Safe GPU Encoding

This module provides safe text truncation to prevent GPU OOM errors
when encoding long texts (especially commits).

Author: Filo-Priori V8 Team
Date: 2025-11-13
"""

import logging
from typing import List, Optional
import re

logger = logging.getLogger(__name__)


class SafeTextTruncator:
    """
    Safe text truncator that ensures texts fit within model limits
    """

    def __init__(
        self,
        max_tokens: int = 512,
        safety_margin: float = 0.9,  # Use 90% of max to be safe
        char_to_token_ratio: float = 4.0,  # Conservative estimate: 1 token ≈ 4 chars
        verbose: bool = True
    ):
        """
        Initialize truncator

        Args:
            max_tokens: Maximum tokens the model supports (default: 512)
            safety_margin: Safety factor (0.9 = use 90% of max)
            char_to_token_ratio: Estimated chars per token
            verbose: Enable logging
        """
        self.max_tokens = max_tokens
        self.safety_margin = safety_margin
        self.char_to_token_ratio = char_to_token_ratio
        self.verbose = verbose

        # Effective limits
        self.effective_max_tokens = int(max_tokens * safety_margin)
        self.effective_max_chars = int(self.effective_max_tokens * char_to_token_ratio)

        if verbose:
            logger.info(f"SafeTextTruncator initialized:")
            logger.info(f"  Max tokens: {max_tokens}")
            logger.info(f"  Safety margin: {safety_margin} (use {self.effective_max_tokens} tokens)")
            logger.info(f"  Char/token ratio: {char_to_token_ratio}")
            logger.info(f"  Max chars: {self.effective_max_chars}")

    def truncate_text(
        self,
        text: str,
        strategy: str = "tail"
    ) -> str:
        """
        Truncate text safely

        Args:
            text: Input text
            strategy: Truncation strategy
                - "tail": Keep start, truncate end (default for commits)
                - "head": Keep end, truncate start
                - "middle": Keep start and end, truncate middle

        Returns:
            Truncated text
        """
        if not text or len(text) <= self.effective_max_chars:
            return text

        if strategy == "tail":
            # Keep start (most important info usually at beginning)
            truncated = text[:self.effective_max_chars]
            # Try to cut at word boundary
            last_space = truncated.rfind(' ')
            if last_space > self.effective_max_chars * 0.9:  # If space is near end
                truncated = truncated[:last_space]
            return truncated + " [TRUNCATED]"

        elif strategy == "head":
            # Keep end
            truncated = text[-self.effective_max_chars:]
            first_space = truncated.find(' ')
            if first_space > 0 and first_space < self.effective_max_chars * 0.1:
                truncated = truncated[first_space+1:]
            return "[TRUNCATED] " + truncated

        elif strategy == "middle":
            # Keep start and end, remove middle
            half = self.effective_max_chars // 2
            start = text[:half]
            end = text[-half:]
            # Try word boundaries
            last_space_start = start.rfind(' ')
            first_space_end = end.find(' ')
            if last_space_start > half * 0.9:
                start = start[:last_space_start]
            if first_space_end > 0 and first_space_end < half * 0.1:
                end = end[first_space_end+1:]
            return start + " [...] " + end

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def truncate_batch(
        self,
        texts: List[str],
        strategy: str = "tail"
    ) -> List[str]:
        """
        Truncate batch of texts

        Args:
            texts: List of input texts
            strategy: Truncation strategy

        Returns:
            List of truncated texts
        """
        truncated = []
        num_truncated = 0

        for text in texts:
            original_len = len(text) if text else 0
            truncated_text = self.truncate_text(text, strategy=strategy)
            truncated.append(truncated_text)

            if original_len > self.effective_max_chars:
                num_truncated += 1

        if self.verbose and num_truncated > 0:
            logger.info(f"Truncated {num_truncated}/{len(texts)} texts "
                       f"({num_truncated/len(texts)*100:.1f}%)")

        return truncated

    def analyze_lengths(
        self,
        texts: List[str],
        percentiles: List[int] = [50, 75, 90, 95, 99]
    ) -> dict:
        """
        Analyze text lengths to determine appropriate limits

        Args:
            texts: List of texts to analyze
            percentiles: Percentiles to compute

        Returns:
            Dictionary with statistics
        """
        lengths = [len(str(t)) if t else 0 for t in texts]
        lengths_sorted = sorted(lengths)

        stats = {
            'count': len(texts),
            'min': min(lengths) if lengths else 0,
            'max': max(lengths) if lengths else 0,
            'mean': sum(lengths) / len(lengths) if lengths else 0,
            'median': lengths_sorted[len(lengths)//2] if lengths else 0,
            'percentiles': {},
            'exceeding_limit': sum(1 for l in lengths if l > self.effective_max_chars),
            'exceeding_pct': sum(1 for l in lengths if l > self.effective_max_chars) / len(lengths) * 100 if lengths else 0
        }

        for p in percentiles:
            idx = int(len(lengths_sorted) * p / 100)
            stats['percentiles'][f'p{p}'] = lengths_sorted[min(idx, len(lengths_sorted)-1)]

        return stats


def get_model_max_length(model) -> int:
    """
    Get max sequence length from a SentenceTransformer model

    Args:
        model: SentenceTransformer instance

    Returns:
        Max sequence length in tokens
    """
    try:
        # PRIORITY 1: Try to get from model's actual max_seq_length
        # This is the REAL limit enforced by the model architecture
        if hasattr(model, '_first_module'):
            first_module = model._first_module()
            if hasattr(first_module, 'max_seq_length'):
                max_len = first_module.max_seq_length
                if max_len and max_len < 100_000:  # Sanity check
                    logger.info(f"✓ Detected max_seq_length from model: {max_len} tokens")
                    return max_len

        # PRIORITY 2: Try to get from model config
        if hasattr(model, 'max_seq_length'):
            max_len = model.max_seq_length
            if max_len and max_len < 100_000:
                logger.info(f"✓ Detected max_seq_length from config: {max_len} tokens")
                return max_len

        # PRIORITY 3: Try tokenizer (but be careful of huge defaults)
        # Tokenizer's model_max_length is often a huge default (e.g., 32767)
        # and NOT the actual model limit!
        if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'model_max_length'):
            max_len = model.tokenizer.model_max_length
            # More conservative sanity check: realistic models are 128-8192 tokens
            if max_len and 128 <= max_len <= 8192:
                logger.info(f"✓ Detected max_length from tokenizer: {max_len} tokens")
                return max_len
            elif max_len and max_len > 8192:
                logger.warning(f"⚠ Tokenizer reports suspiciously high max_length={max_len}, "
                             f"likely a default. Using safe fallback 512.")

        # Default fallback
        logger.warning("⚠ Could not detect model max_length, using safe default 512")
        return 512

    except Exception as e:
        logger.warning(f"⚠ Error detecting model max_length: {e}, using default 512")
        return 512


def create_truncator_for_model(
    model,
    safety_margin: float = 0.9
) -> SafeTextTruncator:
    """
    Create truncator with settings appropriate for the model

    Args:
        model: SentenceTransformer instance
        safety_margin: Safety factor (0.9 = use 90% of max)

    Returns:
        SafeTextTruncator instance
    """
    max_length = get_model_max_length(model)

    logger.info(f"Creating truncator for model with max_length={max_length}")

    return SafeTextTruncator(
        max_tokens=max_length,
        safety_margin=safety_margin,
        verbose=True
    )


# Convenience function for quick truncation
def truncate_texts_safe(
    texts: List[str],
    max_tokens: int = 512,
    strategy: str = "tail"
) -> List[str]:
    """
    Quick truncation function

    Args:
        texts: List of texts
        max_tokens: Max tokens
        strategy: Truncation strategy

    Returns:
        List of truncated texts
    """
    truncator = SafeTextTruncator(max_tokens=max_tokens, verbose=False)
    return truncator.truncate_batch(texts, strategy=strategy)
