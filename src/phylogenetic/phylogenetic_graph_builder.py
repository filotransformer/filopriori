"""
Phylogenetic Graph Builder for Filo-Priori V8

This module constructs TRUE phylogenetic graphs based on real software engineering
relationships, replacing the V7 k-NN graph that was based on semantic similarity.

Two types of graphs are supported:

1. CO-FAILURE GRAPH:
   - Nodes: Test Cases (TC_Key)
   - Edges: Exist if tests failed together in the same Build_ID
   - Weights: P(A fails | B fails) - conditional probability

2. COMMIT DEPENDENCY GRAPH:
   - Nodes: Test Cases (TC_Key)
   - Edges: Exist if tests share same commit/CR
   - Weights: Number of shared commits (normalized)

These graphs represent real software engineering relationships, not semantic similarity.

Author: Filo-Priori V8 Team
Date: 2025-11-06
"""

import os
import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import ast
import logging

logger = logging.getLogger(__name__)


class PhylogeneticGraphBuilder:
    """
    Builds phylogenetic graphs based on real software engineering relationships.

    This class constructs graphs that represent:
    - Test co-failure patterns (tests that fail together)
    - Commit dependencies (tests affected by same code changes)

    These are TRUE structural relationships, not semantic similarity proxies.
    """

    def __init__(self,
                 graph_type: str = 'co_failure',
                 min_co_occurrences: int = 2,
                 weight_threshold: float = 0.1,
                 verbose: bool = True):
        """
        Initialize the phylogenetic graph builder.

        Args:
            graph_type: Type of graph to build ('co_failure', 'commit_dependency', or 'hybrid')
            min_co_occurrences: Minimum co-occurrences to create edge
            weight_threshold: Minimum weight to keep edge (0.0 to 1.0)
            verbose: Enable verbose logging
        """
        self.graph_type = graph_type
        self.min_co_occurrences = min_co_occurrences
        self.weight_threshold = weight_threshold
        self.verbose = verbose

        # Graph data structures
        self.tc_to_idx: Dict[str, int] = {}
        self.idx_to_tc: Dict[int, str] = {}
        self.edges: Dict[Tuple[int, int], float] = {}

        logger.info(f"Initialized PhylogeneticGraphBuilder (type={graph_type})")

    def fit(self, df_train: pd.DataFrame) -> 'PhylogeneticGraphBuilder':
        """
        Fit the graph builder on training data to learn relationships.

        Args:
            df_train: Training DataFrame with columns:
                - TC_Key
                - Build_ID
                - TE_Test_Result
                - commit (optional, for commit_dependency graph)
                - CR (optional)

        Returns:
            self (for method chaining)
        """
        logger.info("Building phylogenetic graph from training data...")
        logger.info(f"Graph type: {self.graph_type}")

        # Build TC_Key index
        self._build_tc_index(df_train)

        # Build graph based on type
        if self.graph_type == 'co_failure':
            self._build_co_failure_graph(df_train)
        elif self.graph_type == 'commit_dependency':
            self._build_commit_dependency_graph(df_train)
        elif self.graph_type == 'hybrid':
            # Combine both graphs
            self._build_co_failure_graph(df_train)
            co_failure_edges = self.edges.copy()
            self._build_commit_dependency_graph(df_train)
            # Merge edges (average weights)
            for edge, weight in co_failure_edges.items():
                if edge in self.edges:
                    self.edges[edge] = (self.edges[edge] + weight) / 2.0
                else:
                    self.edges[edge] = weight / 2.0
        else:
            raise ValueError(f"Unknown graph_type: {self.graph_type}")

        logger.info(f"Graph built: {len(self.tc_to_idx)} nodes, {len(self.edges)} edges")

        return self

    def get_edge_index_and_weights(self,
                                   tc_keys: List[str],
                                   return_torch: bool = True) -> Tuple:
        """
        Get edge_index and edge_weights for a list of TC_Keys.

        This method creates a subgraph for the given TC_Keys, mapping them to
        indices [0, 1, 2, ..., len(tc_keys)-1] and returning edges between them.

        Args:
            tc_keys: List of TC_Keys to include in subgraph
            return_torch: If True, return torch tensors; else numpy arrays

        Returns:
            edge_index: [2, E] array of edge connections
            edge_weights: [E] array of edge weights
        """
        # Create mapping from TC_Key to local index
        local_tc_to_idx = {tc: i for i, tc in enumerate(tc_keys)}

        # Build edge list for this subset
        edge_list = []
        weight_list = []

        for (src_global, dst_global), weight in self.edges.items():
            src_tc = self.idx_to_tc.get(src_global)
            dst_tc = self.idx_to_tc.get(dst_global)

            # Check if both nodes in current subset
            if src_tc in local_tc_to_idx and dst_tc in local_tc_to_idx:
                src_local = local_tc_to_idx[src_tc]
                dst_local = local_tc_to_idx[dst_tc]

                # Add edge (bidirectional)
                edge_list.append([src_local, dst_local])
                weight_list.append(weight)

                # Add reverse edge
                edge_list.append([dst_local, src_local])
                weight_list.append(weight)

        if len(edge_list) == 0:
            # No edges in this subgraph, return empty
            if return_torch:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_weights = torch.zeros((0,), dtype=torch.float32)
            else:
                edge_index = np.zeros((2, 0), dtype=np.int64)
                edge_weights = np.zeros((0,), dtype=np.float32)
        else:
            edge_index = np.array(edge_list).T  # Shape: [2, E]
            edge_weights = np.array(weight_list)

            if return_torch:
                edge_index = torch.from_numpy(edge_index).long()
                edge_weights = torch.from_numpy(edge_weights).float()

        return edge_index, edge_weights

    # ==================== PRIVATE METHODS ====================

    def _build_tc_index(self, df: pd.DataFrame) -> None:
        """Build mapping between TC_Key and integer indices."""
        unique_tc_keys = df['TC_Key'].unique()

        self.tc_to_idx = {tc: i for i, tc in enumerate(unique_tc_keys)}
        self.idx_to_tc = {i: tc for tc, i in self.tc_to_idx.items()}

        logger.info(f"Built TC index: {len(self.tc_to_idx)} unique test cases")

    def _build_co_failure_graph(self, df: pd.DataFrame) -> None:
        """
        Build co-failure graph from training data.

        Edge weight = P(A fails | B fails) = co_failures / failures_of_B

        Algorithm:
        1. For each build, find all tests that failed
        2. Create edges between all pairs of failing tests
        3. Compute conditional probability as edge weight
        """
        logger.info("Building co-failure graph...")

        # Count failures per test
        failure_counts = defaultdict(int)
        for _, row in df.iterrows():
            tc_key = row['TC_Key']
            result = row['TE_Test_Result']
            if result != 'Pass':
                failure_counts[tc_key] += 1

        # Count co-failures
        co_failure_counts = defaultdict(int)

        # Group by Build_ID
        grouped = df.groupby('Build_ID')

        for build_id, build_df in grouped:
            # Find all tests that failed in this build
            failing_tests = build_df[build_df['TE_Test_Result'] != 'Pass']['TC_Key'].tolist()

            # Create edges between all pairs of failing tests
            for i, tc_a in enumerate(failing_tests):
                for tc_b in failing_tests[i+1:]:
                    # Get global indices
                    idx_a = self.tc_to_idx.get(tc_a)
                    idx_b = self.tc_to_idx.get(tc_b)

                    if idx_a is not None and idx_b is not None:
                        # Count this co-failure (symmetric)
                        edge = tuple(sorted([idx_a, idx_b]))
                        co_failure_counts[edge] += 1

        # Compute edge weights as conditional probabilities
        self.edges = {}

        for (idx_a, idx_b), co_fail_count in co_failure_counts.items():
            if co_fail_count < self.min_co_occurrences:
                continue

            tc_a = self.idx_to_tc[idx_a]
            tc_b = self.idx_to_tc[idx_b]

            # P(A fails | B fails) = co_failures / failures_of_B
            prob_a_given_b = co_fail_count / failure_counts[tc_b] if failure_counts[tc_b] > 0 else 0.0

            # P(B fails | A fails) = co_failures / failures_of_A
            prob_b_given_a = co_fail_count / failure_counts[tc_a] if failure_counts[tc_a] > 0 else 0.0

            # Use average of both conditional probabilities
            weight = (prob_a_given_b + prob_b_given_a) / 2.0

            if weight >= self.weight_threshold:
                self.edges[(idx_a, idx_b)] = weight

        logger.info(f"Co-failure graph: {len(self.edges)} edges created")

    def _build_commit_dependency_graph(self, df: pd.DataFrame) -> None:
        """
        Build commit dependency graph from training data.

        Edge weight = Normalized count of shared commits

        Algorithm:
        1. For each commit, find all tests associated with it
        2. Create edges between all pairs of tests sharing commits
        3. Weight = shared_commits / max_commits
        """
        logger.info("Building commit dependency graph...")

        # Build commit -> test mapping
        commit_to_tests = defaultdict(set)

        for _, row in df.iterrows():
            tc_key = row['TC_Key']

            # Process commits
            if 'commit' in row.index and pd.notna(row['commit']):
                commit_str = row['commit']
                try:
                    commits = ast.literal_eval(str(commit_str))
                    if isinstance(commits, list):
                        for commit in commits:
                            commit_to_tests[commit].add(tc_key)
                    else:
                        commit_to_tests[str(commit_str)].add(tc_key)
                except:
                    commit_to_tests[str(commit_str)].add(tc_key)

            # Process CRs
            for cr_col in ['CR', 'CR_y']:
                if cr_col in row.index and pd.notna(row[cr_col]):
                    cr_str = row[cr_col]
                    try:
                        crs = ast.literal_eval(str(cr_str))
                        if isinstance(crs, list):
                            for cr in crs:
                                commit_to_tests[f"CR_{cr}"].add(tc_key)
                        else:
                            commit_to_tests[f"CR_{cr_str}"].add(tc_key)
                    except:
                        commit_to_tests[f"CR_{cr_str}"].add(tc_key)

        # Count shared commits between test pairs
        shared_commit_counts = defaultdict(int)

        for commit, test_set in commit_to_tests.items():
            if len(test_set) < 2:
                continue

            tests = list(test_set)
            for i, tc_a in enumerate(tests):
                for tc_b in tests[i+1:]:
                    idx_a = self.tc_to_idx.get(tc_a)
                    idx_b = self.tc_to_idx.get(tc_b)

                    if idx_a is not None and idx_b is not None:
                        edge = tuple(sorted([idx_a, idx_b]))
                        shared_commit_counts[edge] += 1

        # Normalize weights
        self.edges = {}

        if len(shared_commit_counts) > 0:
            max_shared = max(shared_commit_counts.values())

            for (idx_a, idx_b), count in shared_commit_counts.items():
                if count < self.min_co_occurrences:
                    continue

                # Normalize by max shared commits
                weight = count / max_shared

                if weight >= self.weight_threshold:
                    self.edges[(idx_a, idx_b)] = weight

        logger.info(f"Commit dependency graph: {len(self.edges)} edges created")

    def save_graph(self, filepath: str) -> None:
        """
        Save graph structure to disk.

        Args:
            filepath: Path to save the graph (pickle format)
        """
        import pickle

        state = {
            'graph_type': self.graph_type,
            'tc_to_idx': self.tc_to_idx,
            'idx_to_tc': self.idx_to_tc,
            'edges': self.edges,
            'min_co_occurrences': self.min_co_occurrences,
            'weight_threshold': self.weight_threshold
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Graph saved to {filepath}")

    def load_graph(self, filepath: str) -> 'PhylogeneticGraphBuilder':
        """
        Load previously built graph from disk.

        Args:
            filepath: Path to load the graph from

        Returns:
            self (for method chaining)
        """
        import pickle

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.graph_type = state['graph_type']
        self.tc_to_idx = state['tc_to_idx']
        self.idx_to_tc = state['idx_to_tc']
        self.edges = state['edges']
        self.min_co_occurrences = state.get('min_co_occurrences', 2)
        self.weight_threshold = state.get('weight_threshold', 0.1)

        logger.info(f"Graph loaded from {filepath}")
        logger.info(f"  Type: {self.graph_type}")
        logger.info(f"  Nodes: {len(self.tc_to_idx)}")
        logger.info(f"  Edges: {len(self.edges)}")

        return self

    def get_graph_statistics(self) -> Dict:
        """
        Get statistics about the built graph.

        Returns:
            Dictionary with graph statistics
        """
        if len(self.edges) == 0:
            avg_weight = 0.0
            min_weight = 0.0
            max_weight = 0.0
            avg_degree = 0.0
        else:
            weights = list(self.edges.values())
            avg_weight = np.mean(weights)
            min_weight = np.min(weights)
            max_weight = np.max(weights)

            # Compute degree distribution
            degree_count = defaultdict(int)
            for (src, dst) in self.edges.keys():
                degree_count[src] += 1
                degree_count[dst] += 1

            avg_degree = np.mean(list(degree_count.values())) if degree_count else 0.0

        return {
            'graph_type': self.graph_type,
            'num_nodes': len(self.tc_to_idx),
            'num_edges': len(self.edges),
            'avg_edge_weight': avg_weight,
            'min_edge_weight': min_weight,
            'max_edge_weight': max_weight,
            'avg_degree': avg_degree
        }


def build_phylogenetic_graph(df_train: pd.DataFrame,
                             graph_type: str = 'co_failure',
                             min_co_occurrences: int = 2,
                             weight_threshold: float = 0.1,
                             cache_path: Optional[str] = None,
                             use_multi_edge: bool = False,
                             embeddings: Optional[np.ndarray] = None,
                             edge_types: Optional[List[str]] = None,
                             edge_weights_config: Optional[Dict[str, float]] = None,
                             semantic_top_k: int = 10,
                             semantic_threshold: float = 0.7) -> PhylogeneticGraphBuilder:
    """
    Convenience function to build phylogenetic graph.

    Args:
        df_train: Training DataFrame
        graph_type: Type of graph ('co_failure', 'commit_dependency', or 'hybrid')
                   Only used if use_multi_edge=False
        min_co_occurrences: Minimum co-occurrences to create edge
        weight_threshold: Minimum weight to keep edge
        cache_path: Path to cache/load graph
        use_multi_edge: If True, use MultiEdgeGraphBuilder with multiple edge types
        embeddings: Embeddings for semantic edges (required if 'semantic' in edge_types)
        edge_types: List of edge types for multi-edge graph
                   Options: ['co_failure', 'co_success', 'semantic', 'temporal', 'component']
        edge_weights_config: Weights for each edge type (for weighted combination)
        semantic_top_k: Number of semantic neighbors per node
        semantic_threshold: Minimum cosine similarity for semantic edges

    Returns:
        PhylogeneticGraphBuilder or MultiEdgeGraphBuilder instance

    Example:
        >>> # Traditional single-edge graph
        >>> graph_builder = build_phylogenetic_graph(
        ...     df_train,
        ...     graph_type='co_failure',
        ...     cache_path='cache/phylogenetic_graph.pkl'
        ... )

        >>> # Multi-edge graph with semantic edges
        >>> graph_builder = build_phylogenetic_graph(
        ...     df_train,
        ...     use_multi_edge=True,
        ...     embeddings=train_embeddings,
        ...     edge_types=['co_failure', 'co_success', 'semantic'],
        ...     cache_path='cache/multi_edge_graph.pkl'
        ... )
        >>> edge_index, edge_weights = graph_builder.get_edge_index_and_weights(tc_keys)
    """
    logger.info("="*70)
    logger.info("BUILDING PHYLOGENETIC GRAPH")
    logger.info("="*70)

    if use_multi_edge:
        # Use new multi-edge graph builder
        from .multi_edge_graph_builder import MultiEdgeGraphBuilder

        # Default edge types if not specified
        if edge_types is None:
            edge_types = ['co_failure', 'co_success', 'semantic']

        logger.info(f"Using MultiEdgeGraphBuilder with edge types: {edge_types}")

        builder = MultiEdgeGraphBuilder(
            edge_types=edge_types,
            edge_weights=edge_weights_config,
            min_co_occurrences=min_co_occurrences,
            weight_threshold=weight_threshold,
            semantic_top_k=semantic_top_k,
            semantic_threshold=semantic_threshold,
            verbose=True
        )

        # Load or build
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading cached multi-edge graph from {cache_path}")
            builder.load_graph(cache_path)
        else:
            logger.info("Building multi-edge graph from training data...")
            builder.fit(df_train, embeddings=embeddings)

            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                builder.save_graph(cache_path)

        # Print statistics
        stats = builder.get_statistics()
        logger.info("\n" + "="*70)
        logger.info("MULTI-EDGE PHYLOGENETIC GRAPH STATISTICS")
        logger.info("="*70)
        logger.info(f"Nodes: {stats['num_nodes']}")
        logger.info(f"Edges (combined): {stats['num_edges']}")
        logger.info(f"Edge type counts: {stats.get('edge_type_counts', {})}")
        logger.info(f"Density: {stats['density']:.6f}")
        logger.info(f"Avg Degree: {stats['avg_degree']:.2f}")
        logger.info("="*70)

    else:
        # Use traditional single-edge graph builder
        builder = PhylogeneticGraphBuilder(
            graph_type=graph_type,
            min_co_occurrences=min_co_occurrences,
            weight_threshold=weight_threshold,
            verbose=True
        )

        # Load or build
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading cached graph from {cache_path}")
            builder.load_graph(cache_path)
        else:
            logger.info("Building graph from training data...")
            builder.fit(df_train)

            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                builder.save_graph(cache_path)

        # Print statistics
        stats = builder.get_graph_statistics()
        logger.info("\n" + "="*70)
        logger.info("PHYLOGENETIC GRAPH STATISTICS")
        logger.info("="*70)
        logger.info(f"Graph Type: {stats['graph_type']}")
        logger.info(f"Nodes: {stats['num_nodes']}")
        logger.info(f"Edges: {stats['num_edges']}")
        logger.info(f"Avg Edge Weight: {stats['avg_edge_weight']:.4f}")
        logger.info(f"Min Edge Weight: {stats['min_edge_weight']:.4f}")
        logger.info(f"Max Edge Weight: {stats['max_edge_weight']:.4f}")
        logger.info(f"Avg Degree: {stats['avg_degree']:.2f}")
        logger.info("="*70)

    return builder


# For backwards compatibility and ease of import
__all__ = [
    'PhylogeneticGraphBuilder',
    'build_phylogenetic_graph'
]
