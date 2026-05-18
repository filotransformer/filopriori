"""
Git DAG Builder for Phylogenetic Test Case Prioritization

This module constructs a Git DAG (Directed Acyclic Graph) from commit history data,
treating the software evolution as a phylogenetic tree.

The Git DAG captures:
1. Temporal relationships: Commits from older builds point to commits in newer builds
2. Co-occurrence relationships: Commits associated with the same test case
3. Build relationships: Commits within the same build are siblings

Key Concepts:
- Nodes: Individual commits (represented by their messages)
- Edges: Temporal and co-occurrence relationships
- Edge weights: Based on temporal distance and co-occurrence frequency

Author: Filo-Priori V9 Team
Date: November 2025
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import ast
import logging
import pickle
import os
from datetime import datetime

logger = logging.getLogger(__name__)


class GitDAGBuilder:
    """
    Builds a Git DAG from commit history for phylogenetic analysis.

    The DAG represents software evolution where:
    - Commits are taxa (nodes)
    - Temporal relationships are evolutionary branches (edges)
    - Build sequences represent evolutionary lineages

    This enables the PhyloEncoder to propagate failure signals
    through the evolutionary history of the codebase.
    """

    def __init__(
        self,
        max_commits: int = 10000,
        max_edges_per_node: int = 50,
        temporal_window: int = 10,
        min_edge_weight: float = 0.1,
        cache_path: Optional[str] = None
    ):
        """
        Initialize Git DAG Builder.

        Args:
            max_commits: Maximum number of commit nodes to include
            max_edges_per_node: Maximum edges per node (to limit graph size)
            temporal_window: Number of previous builds to consider for edges
            min_edge_weight: Minimum edge weight threshold
            cache_path: Path to cache the built DAG
        """
        self.max_commits = max_commits
        self.max_edges_per_node = max_edges_per_node
        self.temporal_window = temporal_window
        self.min_edge_weight = min_edge_weight
        self.cache_path = cache_path

        # Data structures
        self.commit_to_idx: Dict[str, int] = {}
        self.idx_to_commit: Dict[int, str] = {}
        self.commit_messages: List[str] = []
        self.commit_to_builds: Dict[str, Set[str]] = defaultdict(set)
        self.commit_to_tcs: Dict[str, Set[str]] = defaultdict(set)
        self.build_to_commits: Dict[str, Set[str]] = defaultdict(set)
        self.build_dates: Dict[str, datetime] = {}
        self.build_order: List[str] = []

        # Graph structures
        self.edge_index: Optional[torch.Tensor] = None
        self.edge_weights: Optional[torch.Tensor] = None
        self.path_lengths: Optional[torch.Tensor] = None
        self.merge_counts: Optional[torch.Tensor] = None

        logger.info(f"Initialized GitDAGBuilder:")
        logger.info(f"  - Max commits: {max_commits}")
        logger.info(f"  - Temporal window: {temporal_window} builds")
        logger.info(f"  - Cache path: {cache_path}")

    def build_from_dataframe(self, df: pd.DataFrame) -> 'GitDAGBuilder':
        """
        Build Git DAG from a DataFrame with commit information.

        Args:
            df: DataFrame with columns:
                - 'commit': List of commit messages (as string repr of list)
                - 'Build_ID': Build identifier
                - 'Build_Test_Start_Date': Build timestamp
                - 'TC_Key': Test case identifier

        Returns:
            self for chaining
        """
        # Check cache
        if self.cache_path and os.path.exists(self.cache_path):
            logger.info(f"Loading cached Git DAG from {self.cache_path}")
            return self._load_cache()

        logger.info("Building Git DAG from DataFrame...")
        logger.info(f"  DataFrame shape: {df.shape}")

        # Step 1: Extract commits and build temporal ordering
        self._extract_commits(df)

        # Step 2: Build temporal edges
        self._build_temporal_edges()

        # Step 3: Compute path lengths (approximate)
        self._compute_path_lengths()

        # Save cache
        if self.cache_path:
            self._save_cache()

        return self

    def _extract_commits(self, df: pd.DataFrame):
        """Extract commits and their relationships from DataFrame."""
        logger.info("  Step 1: Extracting commits...")

        # Parse build dates
        df['_date'] = pd.to_datetime(df['Build_Test_Start_Date'], errors='coerce')

        # Get unique builds and their dates
        build_dates = df.groupby('Build_ID')['_date'].min().dropna()
        self.build_dates = build_dates.to_dict()
        self.build_order = build_dates.sort_values().index.tolist()

        logger.info(f"    Found {len(self.build_order)} builds")
        logger.info(f"    Date range: {build_dates.min()} to {build_dates.max()}")

        # Extract commits
        commit_frequency = defaultdict(int)

        for idx, row in df.iterrows():
            try:
                commit_str = row.get('commit', '')
                if pd.isna(commit_str) or not commit_str:
                    continue

                commits = ast.literal_eval(commit_str) if isinstance(commit_str, str) else []
                build_id = row['Build_ID']
                tc_key = row.get('TC_Key', '')

                for commit in commits:
                    if not isinstance(commit, str):
                        commit = str(commit)
                    commit = commit.strip()

                    if not commit:
                        continue

                    commit_frequency[commit] += 1
                    self.commit_to_builds[commit].add(build_id)
                    self.commit_to_tcs[commit].add(tc_key)
                    self.build_to_commits[build_id].add(commit)

            except Exception as e:
                continue

        # Select top commits by frequency
        sorted_commits = sorted(commit_frequency.items(), key=lambda x: -x[1])
        top_commits = sorted_commits[:self.max_commits]

        for idx, (commit, freq) in enumerate(top_commits):
            self.commit_to_idx[commit] = idx
            self.idx_to_commit[idx] = commit
            self.commit_messages.append(commit)

        logger.info(f"    Extracted {len(self.commit_to_idx)} commits (top by frequency)")
        logger.info(f"    Total unique commits found: {len(commit_frequency)}")

    def _build_temporal_edges(self):
        """Build temporal edges based on build order."""
        logger.info("  Step 2: Building temporal edges...")

        edges_src = []
        edges_dst = []
        weights = []

        # Build position mapping
        build_position = {b: i for i, b in enumerate(self.build_order)}

        # For each commit, find temporal connections
        for commit, builds in self.commit_to_builds.items():
            if commit not in self.commit_to_idx:
                continue

            commit_idx = self.commit_to_idx[commit]

            # Get all builds for this commit
            commit_builds = [b for b in builds if b in build_position]
            if not commit_builds:
                continue

            # Find the earliest build for this commit
            earliest_pos = min(build_position[b] for b in commit_builds)
            earliest_build = self.build_order[earliest_pos]

            # Connect to commits from previous builds (temporal window)
            edge_count = 0
            for prev_pos in range(max(0, earliest_pos - self.temporal_window), earliest_pos):
                prev_build = self.build_order[prev_pos]
                prev_commits = self.build_to_commits.get(prev_build, set())

                for prev_commit in prev_commits:
                    if prev_commit not in self.commit_to_idx:
                        continue

                    prev_idx = self.commit_to_idx[prev_commit]

                    # Compute edge weight based on temporal distance
                    temporal_dist = earliest_pos - prev_pos
                    weight = 1.0 / (1.0 + temporal_dist)

                    if weight >= self.min_edge_weight:
                        edges_src.append(prev_idx)
                        edges_dst.append(commit_idx)
                        weights.append(weight)
                        edge_count += 1

                        if edge_count >= self.max_edges_per_node:
                            break

                if edge_count >= self.max_edges_per_node:
                    break

        # Add co-occurrence edges (commits in same build)
        logger.info("    Adding co-occurrence edges...")
        co_occurrence_edges = 0

        for build, commits in self.build_to_commits.items():
            commits_in_graph = [c for c in commits if c in self.commit_to_idx]

            # Connect commits within same build (bidirectional)
            for i, c1 in enumerate(commits_in_graph):
                for c2 in commits_in_graph[i+1:]:
                    idx1 = self.commit_to_idx[c1]
                    idx2 = self.commit_to_idx[c2]

                    # Bidirectional edges for siblings
                    edges_src.extend([idx1, idx2])
                    edges_dst.extend([idx2, idx1])
                    weights.extend([0.8, 0.8])  # High weight for siblings
                    co_occurrence_edges += 2

                    if co_occurrence_edges > 50000:  # Limit
                        break

            if co_occurrence_edges > 50000:
                break

        # Convert to tensors
        if edges_src:
            self.edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
            self.edge_weights = torch.tensor(weights, dtype=torch.float32)
        else:
            # Empty graph fallback
            self.edge_index = torch.zeros((2, 0), dtype=torch.long)
            self.edge_weights = torch.zeros(0, dtype=torch.float32)

        logger.info(f"    Built {len(edges_src)} edges")
        logger.info(f"      - Temporal edges: {len(edges_src) - co_occurrence_edges}")
        logger.info(f"      - Co-occurrence edges: {co_occurrence_edges}")

    def _compute_path_lengths(self):
        """Compute approximate path lengths for phylogenetic distance."""
        logger.info("  Step 3: Computing path lengths...")

        if self.edge_index is None or self.edge_index.size(1) == 0:
            self.path_lengths = torch.ones(0, dtype=torch.float32)
            self.merge_counts = torch.zeros(0, dtype=torch.float32)
            return

        # Approximate path length as inverse of edge weight
        # (closer commits have higher weight → shorter path)
        self.path_lengths = 1.0 / (self.edge_weights + 1e-6)

        # Approximate merge counts based on co-occurrence
        # Commits in same build (high weight) have lower merge count
        self.merge_counts = torch.zeros_like(self.edge_weights)
        high_weight_mask = self.edge_weights > 0.7
        self.merge_counts[~high_weight_mask] = 1.0  # Temporal edges cross "merges"

        logger.info(f"    Path lengths range: [{self.path_lengths.min():.2f}, {self.path_lengths.max():.2f}]")

    def _save_cache(self):
        """Save built DAG to cache."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)

        cache_data = {
            'commit_to_idx': self.commit_to_idx,
            'idx_to_commit': self.idx_to_commit,
            'commit_messages': self.commit_messages,
            'edge_index': self.edge_index,
            'edge_weights': self.edge_weights,
            'path_lengths': self.path_lengths,
            'merge_counts': self.merge_counts,
            'build_order': self.build_order,
        }

        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)

        logger.info(f"  Saved Git DAG cache to {self.cache_path}")

    def _load_cache(self) -> 'GitDAGBuilder':
        """Load DAG from cache."""
        with open(self.cache_path, 'rb') as f:
            cache_data = pickle.load(f)

        self.commit_to_idx = cache_data['commit_to_idx']
        self.idx_to_commit = cache_data['idx_to_commit']
        self.commit_messages = cache_data['commit_messages']
        self.edge_index = cache_data['edge_index']
        self.edge_weights = cache_data['edge_weights']
        self.path_lengths = cache_data['path_lengths']
        self.merge_counts = cache_data['merge_counts']
        self.build_order = cache_data.get('build_order', [])

        logger.info(f"  Loaded {len(self.commit_to_idx)} commits, {self.edge_index.size(1)} edges")

        return self

    def get_commit_embeddings(
        self,
        embedding_manager,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Generate embeddings for commit messages using SBERT.

        Args:
            embedding_manager: EmbeddingManager instance with encode method
            device: Device for embeddings

        Returns:
            Commit embeddings tensor [num_commits, embedding_dim]
        """
        logger.info("Generating commit embeddings...")

        if not self.commit_messages:
            logger.warning("No commit messages to embed!")
            return torch.zeros((0, 768), device=device)

        # Create a mini DataFrame for embedding
        commit_df = pd.DataFrame({
            'text': self.commit_messages,
            'TC_Key': [f'commit_{i}' for i in range(len(self.commit_messages))]
        })

        # Use SBERT encoder directly if available
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            embeddings = model.encode(
                self.commit_messages,
                batch_size=128,
                show_progress_bar=True,
                convert_to_tensor=True,
                device=device
            )

            logger.info(f"  Generated embeddings: {embeddings.shape}")
            return embeddings

        except ImportError:
            logger.error("sentence-transformers not available!")
            return torch.zeros((len(self.commit_messages), 768), device=device)

    def get_graph_data(self) -> Dict[str, torch.Tensor]:
        """
        Get graph data for PhyloEncoder.

        Returns:
            Dictionary with:
                - edge_index: [2, E] edge indices
                - edge_weights: [E] edge weights
                - path_lengths: [E] path lengths
                - merge_counts: [E] merge counts
                - num_nodes: number of commit nodes
        """
        return {
            'edge_index': self.edge_index,
            'edge_weights': self.edge_weights,
            'path_lengths': self.path_lengths,
            'merge_counts': self.merge_counts,
            'num_nodes': len(self.commit_to_idx)
        }

    def get_tc_to_commit_mapping(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Create mapping from test cases to commit indices.

        This allows associating each test case with its relevant commits
        in the Git DAG, enabling the PhyloEncoder to compute test-specific
        phylogenetic representations.

        Args:
            df: DataFrame with TC_Key and commit columns

        Returns:
            Dictionary mapping TC_Key to list of commit indices
        """
        tc_to_commits = defaultdict(list)

        for idx, row in df.iterrows():
            try:
                commit_str = row.get('commit', '')
                if pd.isna(commit_str) or not commit_str:
                    continue

                commits = ast.literal_eval(commit_str) if isinstance(commit_str, str) else []
                tc_key = row.get('TC_Key', '')

                for commit in commits:
                    if not isinstance(commit, str):
                        commit = str(commit)
                    commit = commit.strip()

                    if commit in self.commit_to_idx:
                        commit_idx = self.commit_to_idx[commit]
                        if commit_idx not in tc_to_commits[tc_key]:
                            tc_to_commits[tc_key].append(commit_idx)

            except Exception:
                continue

        return dict(tc_to_commits)

    def __repr__(self) -> str:
        return (
            f"GitDAGBuilder("
            f"commits={len(self.commit_to_idx)}, "
            f"edges={self.edge_index.size(1) if self.edge_index is not None else 0}, "
            f"builds={len(self.build_order)})"
        )


def build_git_dag(
    df: pd.DataFrame,
    max_commits: int = 10000,
    temporal_window: int = 10,
    cache_path: Optional[str] = None
) -> GitDAGBuilder:
    """
    Convenience function to build Git DAG from DataFrame.

    Args:
        df: DataFrame with commit information
        max_commits: Maximum commits to include
        temporal_window: Temporal window for edges
        cache_path: Cache path

    Returns:
        Built GitDAGBuilder instance
    """
    builder = GitDAGBuilder(
        max_commits=max_commits,
        temporal_window=temporal_window,
        cache_path=cache_path
    )
    return builder.build_from_dataframe(df)


__all__ = ['GitDAGBuilder', 'build_git_dag']
