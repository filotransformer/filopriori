"""
Multi-Edge Phylogenetic Graph Builder

Extends phylogenetic graphs with multiple edge types for richer connectivity.

Edge Types:
1. CO-FAILURE: Tests that fail together (existing)
2. CO-SUCCESS: Tests that pass together (NEW - negative correlation info)
3. SEMANTIC: Top-k semantically similar tests (NEW - from embeddings)
4. TEMPORAL: Tests executed in sequence (NEW - temporal patterns)
5. COMPONENT: Tests in same component/module (NEW - structural info)

This dramatically increases graph density from ~0.02% to 0.5-1.0%,
giving GAT much more information to propagate.
"""

import pandas as pd
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class MultiEdgeGraphBuilder:
    """
    Builds phylogenetic graphs with multiple edge types.

    Combines traditional co-failure edges with new edge types:
    - Co-success (inverse relationship)
    - Semantic similarity (from embeddings)
    - Temporal adjacency
    - Component relationships
    """

    def __init__(
        self,
        edge_types: List[str] = ['co_failure', 'co_success', 'semantic'],
        edge_weights: Dict[str, float] = None,
        min_co_occurrences: int = 1,
        weight_threshold: float = 0.05,
        semantic_top_k: int = 10,
        semantic_threshold: float = 0.7,
        verbose: bool = True
    ):
        """
        Initialize multi-edge graph builder

        Args:
            edge_types: List of edge types to include
            edge_weights: Weight for each edge type (default: co_failure=1.0, others=0.5)
            min_co_occurrences: Min co-occurrences for co-failure/success edges
            weight_threshold: Min edge weight to keep
            semantic_top_k: Number of semantic neighbors per node
            semantic_threshold: Min cosine similarity for semantic edges
            verbose: Enable verbose logging
        """
        self.edge_types = edge_types
        self.edge_weights = edge_weights or {
            'co_failure': 1.0,
            'co_success': 0.5,
            'semantic': 0.3,
            'temporal': 0.2,
            'component': 0.4
        }
        self.min_co_occurrences = min_co_occurrences
        self.weight_threshold = weight_threshold
        self.semantic_top_k = semantic_top_k
        self.semantic_threshold = semantic_threshold
        self.verbose = verbose

        # Graph data
        self.tc_to_idx: Dict[str, int] = {}
        self.idx_to_tc: Dict[int, str] = {}
        self.edges: Dict[Tuple[int, int], Dict[str, float]] = {}  # Multi-edge: {edge_type: weight}

        logger.info(f"Initialized MultiEdgeGraphBuilder")
        logger.info(f"  Edge types: {edge_types}")
        logger.info(f"  Edge weights: {self.edge_weights}")

    def fit(
        self,
        df_train: pd.DataFrame,
        embeddings: Optional[np.ndarray] = None
    ) -> 'MultiEdgeGraphBuilder':
        """
        Build multi-edge graph from training data

        Args:
            df_train: Training DataFrame
            embeddings: Optional embeddings for semantic edges [N, D]

        Returns:
            self
        """
        logger.info("="*70)
        logger.info("BUILDING MULTI-EDGE PHYLOGENETIC GRAPH")
        logger.info("="*70)

        # Build TC index
        self._build_tc_index(df_train)

        # Build each edge type
        if 'co_failure' in self.edge_types:
            self._build_co_failure_edges(df_train)

        if 'co_success' in self.edge_types:
            self._build_co_success_edges(df_train)

        if 'semantic' in self.edge_types and embeddings is not None:
            # Handle case where embeddings are sample-aligned instead of node-aligned
            if len(embeddings) == len(df_train):
                logger.info("  Extracting node-aligned embeddings from sample-aligned embeddings...")
                node_embeddings = np.zeros((len(self.tc_to_idx), embeddings.shape[1]), dtype=embeddings.dtype)
                # Group by TC_Key and take first embedding for each
                # This ensures we get exactly one embedding per unique test case
                # We can use pandas to find the first index of each TC_Key
                tc_to_first_idx = df_train.reset_index().groupby('TC_Key')['index'].first().to_dict()
                for tc, node_idx in self.tc_to_idx.items():
                    sample_idx = tc_to_first_idx[tc]
                    node_embeddings[node_idx] = embeddings[sample_idx]
                self._build_semantic_edges(node_embeddings)
            elif len(embeddings) == len(self.tc_to_idx):
                self._build_semantic_edges(embeddings)
            else:
                logger.warning(f"  Embeddings shape mismatch! Expected {len(self.tc_to_idx)} or {len(df_train)}, got {len(embeddings)}. Semantic edges might fail.")
                self._build_semantic_edges(embeddings)

        if 'temporal' in self.edge_types:
            self._build_temporal_edges(df_train)

        if 'component' in self.edge_types and 'CR_Component_Name' in df_train.columns:
            self._build_component_edges(df_train)

        # Combine and filter edges
        self._combine_edges()

        logger.info("="*70)

        return self

    def _build_tc_index(self, df: pd.DataFrame):
        """Build TC_Key to index mapping"""
        unique_tcs = sorted(df['TC_Key'].unique())
        self.tc_to_idx = {tc: idx for idx, tc in enumerate(unique_tcs)}
        self.idx_to_tc = {idx: tc for tc, idx in self.tc_to_idx.items()}
        logger.info(f"Built TC index: {len(unique_tcs)} unique test cases")

    def _build_co_failure_edges(self, df: pd.DataFrame):
        """Build co-failure edges (existing logic, improved thresholds)"""
        logger.info("\nBuilding co-failure edges...")

        # Get failures only
        df_fail = df[df['TE_Test_Result'] == 'Fail'].copy()

        # Count co-occurrences
        build_to_tcs = df_fail.groupby('Build_ID')['TC_Key'].apply(list).to_dict()

        co_failure_counts = defaultdict(int)
        tc_failure_counts = defaultdict(int)

        for build_id, tcs in build_to_tcs.items():
            # Count individual failures
            for tc in tcs:
                tc_failure_counts[tc] += 1

            # Count pairwise co-failures
            for i, tc1 in enumerate(tcs):
                for tc2 in tcs[i+1:]:
                    if tc1 != tc2:
                        pair = tuple(sorted([tc1, tc2]))
                        co_failure_counts[pair] += 1

        # Create edges
        num_edges = 0
        for (tc1, tc2), count in co_failure_counts.items():
            if count >= self.min_co_occurrences:
                # Calculate conditional probability: P(tc2 fails | tc1 fails)
                # Using min to make it symmetric
                weight = min(
                    count / tc_failure_counts[tc1],
                    count / tc_failure_counts[tc2]
                ) if tc_failure_counts[tc1] > 0 and tc_failure_counts[tc2] > 0 else 0

                if weight >= self.weight_threshold:
                    idx1, idx2 = self.tc_to_idx[tc1], self.tc_to_idx[tc2]
                    edge_key = (min(idx1, idx2), max(idx1, idx2))

                    if edge_key not in self.edges:
                        self.edges[edge_key] = {}
                    self.edges[edge_key]['co_failure'] = weight
                    num_edges += 1

        logger.info(f"  Created {num_edges} co-failure edges")
        logger.info(f"  Min co-occurrences: {self.min_co_occurrences}")

    def _build_co_success_edges(self, df: pd.DataFrame):
        """
        Build co-success edges (NEW!)

        Tests that consistently PASS together have inverse relationship.
        This captures negative correlation information missing from co-failure.
        """
        logger.info("\nBuilding co-success edges...")

        # Get passes only
        df_pass = df[df['TE_Test_Result'] == 'Pass'].copy()

        # Count co-occurrences
        build_to_tcs = df_pass.groupby('Build_ID')['TC_Key'].apply(list).to_dict()

        co_success_counts = defaultdict(int)
        tc_success_counts = defaultdict(int)

        for build_id, tcs in build_to_tcs.items():
            # Count individual successes
            for tc in tcs:
                tc_success_counts[tc] += 1

            # Count pairwise co-successes
            for i, tc1 in enumerate(tcs):
                for tc2 in tcs[i+1:]:
                    if tc1 != tc2:
                        pair = tuple(sorted([tc1, tc2]))
                        co_success_counts[pair] += 1

        # Create edges
        num_edges = 0
        for (tc1, tc2), count in co_success_counts.items():
            if count >= self.min_co_occurrences:
                weight = min(
                    count / tc_success_counts[tc1],
                    count / tc_success_counts[tc2]
                ) if tc_success_counts[tc1] > 0 and tc_success_counts[tc2] > 0 else 0

                if weight >= self.weight_threshold:
                    idx1, idx2 = self.tc_to_idx[tc1], self.tc_to_idx[tc2]
                    edge_key = (min(idx1, idx2), max(idx1, idx2))

                    if edge_key not in self.edges:
                        self.edges[edge_key] = {}
                    self.edges[edge_key]['co_success'] = weight
                    num_edges += 1

        logger.info(f"  Created {num_edges} co-success edges")

    def _build_semantic_edges(self, embeddings: np.ndarray):
        """
        Build semantic similarity edges (NEW!)

        Connect top-k most similar tests based on embeddings.
        Provides dense connectivity even for tests with no execution history.
        """
        logger.info("\nBuilding semantic similarity edges...")

        # Compute pairwise cosine similarity
        similarities = cosine_similarity(embeddings)

        # For each test, find top-k neighbors
        num_edges = 0
        for idx in range(len(embeddings)):
            # Get similarities for this test
            sims = similarities[idx]

            # Get top-k (excluding self)
            top_k_idx = np.argsort(sims)[::-1][1:self.semantic_top_k+1]

            # Create edges
            for neighbor_idx in top_k_idx:
                sim = sims[neighbor_idx]

                if sim >= self.semantic_threshold:
                    edge_key = (min(idx, neighbor_idx), max(idx, neighbor_idx))

                    if edge_key not in self.edges:
                        self.edges[edge_key] = {}
                    # Use max similarity if edge already exists
                    self.edges[edge_key]['semantic'] = max(
                        self.edges[edge_key].get('semantic', 0),
                        sim
                    )
                    num_edges += 1

        logger.info(f"  Created {num_edges} semantic edges (top-{self.semantic_top_k})")
        logger.info(f"  Similarity threshold: {self.semantic_threshold}")

    def _build_temporal_edges(self, df: pd.DataFrame):
        """
        Build temporal adjacency edges (NEW!)

        Tests executed sequentially in same build may have temporal dependencies.
        """
        logger.info("\nBuilding temporal adjacency edges...")

        # Group by build and sort by execution time
        if 'TE_Date' not in df.columns:
            logger.warning("  Skipping temporal edges: TE_Date column not found")
            return

        num_edges = 0
        for build_id, group in df.groupby('Build_ID'):
            # Sort by execution time
            group_sorted = group.sort_values('TE_Date')
            tcs = group_sorted['TC_Key'].tolist()

            # Connect adjacent tests
            for i in range(len(tcs) - 1):
                tc1, tc2 = tcs[i], tcs[i+1]
                if tc1 in self.tc_to_idx and tc2 in self.tc_to_idx:
                    idx1, idx2 = self.tc_to_idx[tc1], self.tc_to_idx[tc2]
                    edge_key = (min(idx1, idx2), max(idx1, idx2))

                    if edge_key not in self.edges:
                        self.edges[edge_key] = {}

                    # Increment temporal weight (normalized later)
                    self.edges[edge_key]['temporal'] = self.edges[edge_key].get('temporal', 0) + 1
                    num_edges += 1

        # Normalize temporal weights
        if num_edges > 0:
            max_temporal = max(e.get('temporal', 0) for e in self.edges.values())
            for edge in self.edges.values():
                if 'temporal' in edge:
                    edge['temporal'] /= max_temporal

        logger.info(f"  Created {num_edges} temporal edges")

    def _build_component_edges(self, df: pd.DataFrame):
        """
        Build component-based edges (NEW!)

        Tests in same component/module are related.
        """
        logger.info("\nBuilding component edges...")

        # Group by component
        tc_to_components = df.groupby('TC_Key')['CR_Component_Name'].apply(
            lambda x: set(x.dropna())
        ).to_dict()

        num_edges = 0
        tcs = list(self.tc_to_idx.keys())

        for i, tc1 in enumerate(tcs):
            for tc2 in tcs[i+1:]:
                # Get shared components
                comps1 = tc_to_components.get(tc1, set())
                comps2 = tc_to_components.get(tc2, set())

                if comps1 and comps2:
                    shared = comps1 & comps2
                    if shared:
                        # Jaccard similarity
                        weight = len(shared) / len(comps1 | comps2)

                        if weight >= self.weight_threshold:
                            idx1, idx2 = self.tc_to_idx[tc1], self.tc_to_idx[tc2]
                            edge_key = (min(idx1, idx2), max(idx1, idx2))

                            if edge_key not in self.edges:
                                self.edges[edge_key] = {}
                            self.edges[edge_key]['component'] = weight
                            num_edges += 1

        logger.info(f"  Created {num_edges} component edges")

    def _combine_edges(self):
        """Combine multi-edges into single weighted edges"""
        logger.info("\nCombining multi-edge weights...")

        combined_edges = {}

        for edge_key, edge_types in self.edges.items():
            # Weighted sum of edge types
            total_weight = sum(
                edge_types.get(etype, 0) * self.edge_weights.get(etype, 0)
                for etype in self.edge_types
            )

            # Normalize by total weight
            normalizer = sum(self.edge_weights.get(etype, 0) for etype in self.edge_types)
            combined_weight = total_weight / normalizer if normalizer > 0 else 0

            if combined_weight >= self.weight_threshold:
                combined_edges[edge_key] = combined_weight

        # Replace with combined
        num_before = len(self.edges)
        self.edges_multi = self.edges  # Keep multi-edge info
        self.edges = combined_edges

        logger.info(f"  Multi-edges: {num_before}")
        logger.info(f"  Combined edges (after threshold): {len(self.edges)}")

    def get_edge_index_and_weights(
        self,
        tc_keys: List[str],
        return_torch: bool = True
    ) -> Tuple:
        """
        Get edge_index and edge_weights for PyTorch Geometric

        Args:
            tc_keys: List of TC_Keys to include
            return_torch: Return torch tensors (else numpy)

        Returns:
            edge_index: [2, num_edges]
            edge_weights: [num_edges]
        """
        # Map TC_Keys to global indices
        global_tc_to_idx = {tc: idx for idx, tc in enumerate(tc_keys)}

        edges_list = []
        weights_list = []

        for (idx1, idx2), weight in self.edges.items():
            tc1 = self.idx_to_tc.get(idx1)
            tc2 = self.idx_to_tc.get(idx2)

            if tc1 in global_tc_to_idx and tc2 in global_tc_to_idx:
                global_idx1 = global_tc_to_idx[tc1]
                global_idx2 = global_tc_to_idx[tc2]

                # Add both directions (undirected)
                edges_list.append([global_idx1, global_idx2])
                edges_list.append([global_idx2, global_idx1])
                weights_list.extend([weight, weight])

        if len(edges_list) == 0:
            # Empty graph
            edge_index = np.array([[],[]], dtype=np.int64)
            edge_weights = np.array([], dtype=np.float32)
        else:
            edge_index = np.array(edges_list, dtype=np.int64).T
            edge_weights = np.array(weights_list, dtype=np.float32)

        if return_torch:
            edge_index = torch.from_numpy(edge_index).long()
            edge_weights = torch.from_numpy(edge_weights).float()

        return edge_index, edge_weights

    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        num_nodes = len(self.tc_to_idx)
        num_edges = len(self.edges)

        # Count by edge type
        edge_type_counts = defaultdict(int)
        if hasattr(self, 'edges_multi'):
            for edge_types in self.edges_multi.values():
                for etype in edge_types.keys():
                    edge_type_counts[etype] += 1

        # Calculate density
        max_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_edges if max_edges > 0 else 0

        # Calculate avg degree
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0

        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': density,
            'avg_degree': avg_degree,
            'edge_type_counts': dict(edge_type_counts)
        }

    def save_graph(self, filepath: str) -> None:
        """
        Save graph structure to disk

        Args:
            filepath: Path to save the graph (pickle format)
        """
        import pickle

        state = {
            'edge_types': self.edge_types,
            'edge_weights': self.edge_weights,
            'tc_to_idx': self.tc_to_idx,
            'idx_to_tc': self.idx_to_tc,
            'edges': self.edges,
            'edges_multi': getattr(self, 'edges_multi', {}),
            'min_co_occurrences': self.min_co_occurrences,
            'weight_threshold': self.weight_threshold,
            'semantic_top_k': self.semantic_top_k,
            'semantic_threshold': self.semantic_threshold
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Multi-edge graph saved to {filepath}")

    def load_graph(self, filepath: str) -> 'MultiEdgeGraphBuilder':
        """
        Load previously built graph from disk

        Args:
            filepath: Path to load the graph from

        Returns:
            self (for method chaining)
        """
        import pickle

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.edge_types = state['edge_types']
        self.edge_weights = state['edge_weights']
        self.tc_to_idx = state['tc_to_idx']
        self.idx_to_tc = state['idx_to_tc']
        self.edges = state['edges']
        self.edges_multi = state.get('edges_multi', {})
        self.min_co_occurrences = state.get('min_co_occurrences', 1)
        self.weight_threshold = state.get('weight_threshold', 0.05)
        self.semantic_top_k = state.get('semantic_top_k', 10)
        self.semantic_threshold = state.get('semantic_threshold', 0.7)

        logger.info(f"Multi-edge graph loaded from {filepath}")
        logger.info(f"  Edge types: {self.edge_types}")
        logger.info(f"  Nodes: {len(self.tc_to_idx)}")
        logger.info(f"  Combined edges: {len(self.edges)}")

        return self
