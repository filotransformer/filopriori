"""
Orphan scoring utilities using KNN with structural blending.

Provides a shared KNN scorer used at multiple points of the pipeline
to avoid low-variance orphan scores.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

import logging

logger = logging.getLogger(__name__)


def _normalize_features(features: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Z-normalize features to make cosine similarity meaningful."""
    if features is None:
        return None

    features = np.asarray(features, dtype=np.float32)
    if features.size == 0:
        return features

    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    return (features - mean) / std


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature scaling."""
    if temperature <= 0:
        temperature = 1.0

    x = x / temperature
    x = x - x.max()
    exp_x = np.exp(x)
    denom = exp_x.sum()
    if denom <= 0:
        return np.zeros_like(x)
    return exp_x / denom


def _compute_similarity(
    orphan_embeddings: np.ndarray,
    in_graph_embeddings: np.ndarray,
    metric: str = "cosine"
) -> np.ndarray:
    """Compute similarity matrix between orphan and in-graph embeddings."""
    if metric == "euclidean":
        distances = cdist(orphan_embeddings, in_graph_embeddings, metric="euclidean")
        # Convert distance to similarity (larger is better)
        similarities = np.exp(-distances)
    else:
        similarities = cosine_similarity(orphan_embeddings, in_graph_embeddings)

    return similarities


def _combine_similarities(
    semantic_sims: np.ndarray,
    structural_sims: Optional[np.ndarray],
    structural_weight: float
) -> np.ndarray:
    """Blend semantic and structural similarities."""
    if structural_sims is None or structural_weight <= 0:
        return semantic_sims

    weight = np.clip(structural_weight, 0.0, 1.0)
    return (1 - weight) * semantic_sims + weight * structural_sims


def compute_orphan_scores(
    orphan_embeddings: np.ndarray,
    in_graph_embeddings: np.ndarray,
    in_graph_scores: np.ndarray,
    orphan_base_scores: np.ndarray,
    strategy_config: Optional[Dict] = None,
    orphan_structural_features: Optional[np.ndarray] = None,
    in_graph_structural_features: Optional[np.ndarray] = None,
    orphan_priority_fallback: Optional[np.ndarray] = None,
    orphan_texts: Optional[List[str]] = None  # Kept for API compatibility
) -> Tuple[np.ndarray, Dict]:
    """
    Compute orphan scores using temperature-scaled KNN with structural blending.

    Pipeline:
    1. KNN Similarity: Find k nearest neighbors using euclidean distance
    2. Structural Blend: Combine semantic (65%) and structural (35%) similarities
    3. Temperature-Scaled Softmax: Weight neighbors by similarity
    4. Alpha Blend: Mix KNN score with base score

    Args:
        orphan_embeddings: Embeddings for orphan test cases [N, D]
        in_graph_embeddings: Embeddings for in-graph test cases [M, D]
        in_graph_scores: P(Fail) scores for in-graph tests [M]
        orphan_base_scores: Base scores for orphans (typically 0.5) [N]
        strategy_config: Configuration dict with k_neighbors, alpha_blend, etc.
        orphan_structural_features: Structural features for orphans [N, F]
        in_graph_structural_features: Structural features for in-graph tests [M, F]
        orphan_priority_fallback: Priority scores to use when no neighbors found [N]
        orphan_texts: Ignored (kept for API compatibility)

    Returns:
        Tuple of (orphan_scores [N], stats dict)
    """
    cfg = strategy_config or {}
    k_neighbors = max(1, cfg.get("k_neighbors", 20))
    alpha_blend = cfg.get("alpha_blend", 0.55)
    similarity_metric = cfg.get("similarity_metric", "euclidean")
    structural_weight = cfg.get("structural_weight", 0.35)
    temperature = cfg.get("temperature", 0.7)
    min_similarity = cfg.get("min_similarity", 0.05)

    if len(in_graph_embeddings) == 0:
        return orphan_base_scores, {
            "mean": float(np.mean(orphan_base_scores)),
            "std": float(np.std(orphan_base_scores)),
            "min": float(np.min(orphan_base_scores)),
            "max": float(np.max(orphan_base_scores)),
            "k_neighbors": 0,
            "fallback_count": len(orphan_base_scores),
        }

    # Normalize structural features for cosine distance
    orphan_struct_norm = _normalize_features(orphan_structural_features)
    in_graph_struct_norm = _normalize_features(in_graph_structural_features)

    # Stage 1: Compute semantic similarity
    semantic_sims = _compute_similarity(
        orphan_embeddings, in_graph_embeddings, metric=similarity_metric
    )

    # Stage 2: Compute structural similarity and blend
    structural_sims = None
    if orphan_struct_norm is not None and in_graph_struct_norm is not None:
        structural_sims = cosine_similarity(orphan_struct_norm, in_graph_struct_norm)

    combined_sims = _combine_similarities(semantic_sims, structural_sims, structural_weight)

    orphan_scores = np.array(orphan_base_scores, copy=True)
    fallback_count = 0
    effective_neighbors: List[int] = []

    for i, sim_row in enumerate(combined_sims):
        # Select top-k neighbors
        top_k_idx = np.argsort(sim_row)[-k_neighbors:]
        top_k_sims = sim_row[top_k_idx]

        # Filter by minimum similarity if requested
        valid_mask = top_k_sims >= min_similarity if min_similarity > 0 else np.ones_like(top_k_sims, dtype=bool)
        top_k_idx = top_k_idx[valid_mask]
        top_k_sims = top_k_sims[valid_mask]

        if len(top_k_sims) == 0 or np.all(top_k_sims == 0):
            # No useful neighbors - fallback to priority or base score
            fallback = (
                orphan_priority_fallback[i]
                if orphan_priority_fallback is not None
                else orphan_base_scores[i]
            )
            orphan_scores[i] = fallback
            fallback_count += 1
            effective_neighbors.append(0)
            continue

        # Stage 3: Temperature-scaled softmax weighting
        weights = _softmax(top_k_sims, temperature=temperature)
        knn_score = float(np.dot(weights, in_graph_scores[top_k_idx]))

        # Stage 4: Alpha blend with base score
        blended = alpha_blend * knn_score + (1 - alpha_blend) * float(orphan_base_scores[i])
        orphan_scores[i] = blended
        effective_neighbors.append(len(top_k_idx))

    stats = {
        "mean": float(np.mean(orphan_scores)),
        "std": float(np.std(orphan_scores)),
        "min": float(np.min(orphan_scores)),
        "max": float(np.max(orphan_scores)),
        "k_neighbors": int(min(k_neighbors, len(in_graph_embeddings))),
        "fallback_count": int(fallback_count),
        "effective_neighbors": float(np.mean(effective_neighbors)) if effective_neighbors else 0.0
    }

    return orphan_scores, stats


__all__ = ["compute_orphan_scores"]
