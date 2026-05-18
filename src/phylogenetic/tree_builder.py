"""
Phylogenetic Tree Construction Module
Builds tree/graph structure from semantic embeddings using hierarchical clustering
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import torch
from typing import Dict, Tuple, Optional, List
import logging
import pickle
import os
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PhylogeneticTreeBuilder:
    """Constructs phylogenetic tree structure from semantic embeddings"""

    def __init__(self, config: Dict):
        self.config = config
        self.phylo_config = config['phylogenetic']
        self.distance_metric = self.phylo_config['distance_metric']
        self.tree_method = self.phylo_config.get('tree_method', 'hierarchical')
        self.linkage_method = self.phylo_config.get('linkage_method', 'average')
        self.tree_type = self.phylo_config.get('tree_type', 'hierarchical')
        self.use_approx = self.phylo_config.get('use_approx', True)
        self.subsample_size = self.phylo_config.get('subsample_for_tree', 50000)

        self.linkage_matrix = None
        self.tree_structure = None
        self.nj_tree = None

    def compute_distance_matrix(
        self,
        embeddings: np.ndarray,
        metric: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute pairwise distance matrix

        Args:
            embeddings: Array of shape [n_samples, embedding_dim]
            metric: Distance metric (defaults to config)

        Returns:
            Distance matrix [n_samples, n_samples]
        """
        if metric is None:
            metric = self.distance_metric

        logger.info(f"Computing {metric} distance matrix for {len(embeddings)} samples...")

        # For large datasets, use condensed form
        distances = pdist(embeddings, metric=metric)
        distance_matrix = squareform(distances)

        logger.info(f"Distance matrix computed: {distance_matrix.shape}")
        return distance_matrix

    def fast_neighbor_joining(
        self,
        distance_matrix: np.ndarray,
        node_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Fast Neighbor Joining algorithm - O(n²) complexity

        Based on: "A Fast Neighbor Joining Method"
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2843384/

        Args:
            distance_matrix: Pairwise distance matrix [n, n]
            node_names: Optional list of node names

        Returns:
            Dictionary containing tree structure information
        """
        n = len(distance_matrix)

        if node_names is None:
            node_names = [f"node_{i}" for i in range(n)]

        logger.info(f"Building Fast Neighbor Joining tree for {n} nodes...")

        # Make a copy to avoid modifying original
        D = distance_matrix.copy()

        # Initialize active nodes
        active_nodes = list(range(n))
        next_node_id = n

        # Store tree structure
        tree_edges = []
        node_depths = {i: 0 for i in range(n)}
        parent_map = {}

        # Main FNJ loop
        while len(active_nodes) > 2:
            num_active = len(active_nodes)

            # Compute row sums (S_i in the paper)
            row_sums = np.sum(D[np.ix_(active_nodes, active_nodes)], axis=1)

            # Find the pair to join using FNJ criterion
            # Q_ij = (n-2)*d_ij - S_i - S_j
            best_i, best_j = None, None
            best_q = float('inf')

            for idx_i in range(num_active):
                for idx_j in range(idx_i + 1, num_active):
                    i, j = active_nodes[idx_i], active_nodes[idx_j]

                    # FNJ criterion (minimize Q)
                    q = (num_active - 2) * D[i, j] - row_sums[idx_i] - row_sums[idx_j]

                    if q < best_q:
                        best_q = q
                        best_i, best_j = idx_i, idx_j

            # Get actual node IDs
            node_i = active_nodes[best_i]
            node_j = active_nodes[best_j]

            # Compute branch lengths
            dist_i = 0.5 * D[node_i, node_j] + (row_sums[best_i] - row_sums[best_j]) / (2 * (num_active - 2))
            dist_j = D[node_i, node_j] - dist_i

            # Ensure non-negative branch lengths
            dist_i = max(0, dist_i)
            dist_j = max(0, dist_j)

            # Create new internal node
            new_node = next_node_id
            next_node_id += 1

            # Record tree structure
            tree_edges.append((new_node, node_i, dist_i))
            tree_edges.append((new_node, node_j, dist_j))
            parent_map[node_i] = new_node
            parent_map[node_j] = new_node
            node_depths[new_node] = max(node_depths[node_i], node_depths[node_j]) + 1

            # Update distance matrix for new node
            # Expand matrix if needed
            if new_node >= D.shape[0]:
                new_size = new_node + 1
                D_new = np.zeros((new_size, new_size))
                D_new[:D.shape[0], :D.shape[1]] = D
                D = D_new

            # Compute distances to new node
            for idx_k in range(num_active):
                if idx_k == best_i or idx_k == best_j:
                    continue
                k = active_nodes[idx_k]
                # Average distance formula for NJ
                d_new_k = 0.5 * (D[node_i, k] + D[node_j, k] - D[node_i, node_j])
                D[new_node, k] = d_new_k
                D[k, new_node] = d_new_k

            # Remove joined nodes and add new node
            active_nodes = [node for idx, node in enumerate(active_nodes)
                          if idx not in (best_i, best_j)]
            active_nodes.append(new_node)

        # Handle final two nodes
        if len(active_nodes) == 2:
            node_i, node_j = active_nodes
            dist = D[node_i, node_j]
            root_node = next_node_id
            tree_edges.append((root_node, node_i, dist / 2))
            tree_edges.append((root_node, node_j, dist / 2))
            parent_map[node_i] = root_node
            parent_map[node_j] = root_node
            node_depths[root_node] = max(node_depths[node_i], node_depths[node_j]) + 1

        logger.info(f"FNJ tree built: {len(tree_edges)} edges, root at node {root_node}")

        return {
            'edges': tree_edges,
            'parent_map': parent_map,
            'node_depths': node_depths,
            'root': root_node,
            'num_leaves': n
        }

    def build_hierarchical_tree(
        self,
        embeddings: np.ndarray,
        subsample: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Build hierarchical clustering tree

        Args:
            embeddings: Embeddings array [n_samples, embedding_dim]
            subsample: Whether to subsample for large datasets

        Returns:
            Linkage matrix and optional subsample indices
        """
        subsample_indices = None

        # Subsample for very large datasets
        if subsample and len(embeddings) > self.subsample_size:
            logger.info(f"Subsampling {self.subsample_size} samples for tree construction")
            subsample_indices = np.random.choice(
                len(embeddings),
                size=self.subsample_size,
                replace=False
            )
            embeddings_for_tree = embeddings[subsample_indices]
        else:
            embeddings_for_tree = embeddings

        logger.info(f"Building hierarchical tree with {len(embeddings_for_tree)} samples...")
        logger.info(f"Linkage method: {self.linkage_method}")

        # Compute linkage matrix
        # This uses scipy's hierarchical clustering
        self.linkage_matrix = linkage(
            embeddings_for_tree,
            method=self.linkage_method,
            metric=self.distance_metric
        )

        logger.info(f"Linkage matrix computed: {self.linkage_matrix.shape}")

        return self.linkage_matrix, subsample_indices

    def build_knn_graph(
        self,
        embeddings: np.ndarray,
        k: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build k-nearest neighbors graph structure

        Args:
            embeddings: Embeddings array [n_samples, embedding_dim]
            k: Number of neighbors (defaults to config)

        Returns:
            Edge indices [2, num_edges] and edge weights [num_edges]
        """
        if k is None:
            k = self.config['model']['structural_stream']['num_neighbors']

        logger.info(f"Building {k}-NN graph for {len(embeddings)} samples...")

        # Build KNN index
        nbrs = NearestNeighbors(
            n_neighbors=k + 1,  # +1 because point is its own neighbor
            metric=self.distance_metric,
            algorithm='auto',
            n_jobs=-1
        ).fit(embeddings)

        # Find neighbors
        distances, indices = nbrs.kneighbors(embeddings)

        # Build edge list (exclude self-loops)
        edge_index_list = []
        edge_weight_list = []

        for i in range(len(embeddings)):
            for j, neighbor_idx in enumerate(indices[i][1:]):  # Skip first (self)
                edge_index_list.append([i, neighbor_idx])
                # Convert distance to similarity weight
                edge_weight_list.append(1.0 / (1.0 + distances[i][j + 1]))

        edge_index = np.array(edge_index_list).T  # [2, num_edges]
        edge_weights = np.array(edge_weight_list)

        logger.info(f"KNN graph built: {edge_index.shape[1]} edges")

        return edge_index, edge_weights

    def extract_tree_features(
        self,
        linkage_matrix: np.ndarray,
        embeddings: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from hierarchical tree structure

        Args:
            linkage_matrix: Scipy linkage matrix
            embeddings: Original embeddings

        Returns:
            Dictionary with tree-based features
        """
        n_samples = len(embeddings)

        # Convert to tree structure
        root, node_list = to_tree(linkage_matrix, rd=True)

        # Extract features: depth, cluster assignment at different levels
        depths = np.zeros(n_samples)
        cluster_levels = []

        # Get cluster assignments at different cut heights
        from scipy.cluster.hierarchy import fcluster

        # Try different numbers of clusters
        for n_clusters in [5, 10, 20, 50]:
            if n_clusters < n_samples:
                clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
                cluster_levels.append(clusters)

        features = {
            'cluster_assignments': np.array(cluster_levels).T if cluster_levels else None,
            'n_samples': n_samples
        }

        return features

    def save_tree(self, path: str):
        """Save tree structure to disk"""
        logger.info(f"Saving tree structure to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        tree_data = {
            'linkage_matrix': self.linkage_matrix,
            'config': self.phylo_config
        }

        with open(path, 'wb') as f:
            pickle.dump(tree_data, f)

    def load_tree(self, path: str):
        """Load tree structure from disk"""
        logger.info(f"Loading tree structure from {path}")

        with open(path, 'rb') as f:
            tree_data = pickle.load(f)

        self.linkage_matrix = tree_data['linkage_matrix']
        logger.info("Tree structure loaded successfully")

    def build_graph_from_embeddings(
        self,
        embeddings: np.ndarray,
        cache_path: Optional[str] = None
    ) -> Dict:
        """
        Complete pipeline: build graph structure from embeddings

        Args:
            embeddings: Semantic embeddings [n_samples, embedding_dim]
            cache_path: Optional path to cache results

        Returns:
            Dictionary with graph structure (edge_index, edge_weights, etc.)
        """
        # Check cache
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading cached graph from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # Build KNN graph for Graph Neural Network (ALWAYS NEEDED)
        logger.info("Building k-NN graph for GNN (this is the primary structure)")
        edge_index, edge_weights = self.build_knn_graph(embeddings)

        # Initialize tree-related variables
        linkage_matrix = None
        nj_tree = None
        subsample_indices = None
        tree_features = None

        # Check if tree construction is requested
        skip_tree = (self.tree_type == 'none' or
                    self.tree_method == 'knn_only' or
                    self.tree_type == 'knn_only')

        if skip_tree:
            logger.info("Tree construction DISABLED (using k-NN graph only)")
            logger.info("This is the RECOMMENDED configuration for efficiency")
            # No tree construction - k-NN graph is sufficient for GNN
            pass

        elif self.tree_type == 'neighbor_joining' or self.tree_method == 'fast_nj':
            # Use Fast Neighbor Joining (WARNING: This is slow and memory-intensive!)
            logger.warning("=" * 80)
            logger.warning("WARNING: Fast Neighbor Joining tree construction is ENABLED")
            logger.warning("This will be SLOW and MEMORY-INTENSIVE!")
            logger.warning("For datasets > 10K samples, this may take hours and crash.")
            logger.warning("Consider setting tree_type: 'none' in config to disable.")
            logger.warning("=" * 80)

            logger.info("Using Fast Neighbor Joining for tree construction")

            # Subsample if needed
            if self.use_approx and len(embeddings) > self.subsample_size:
                logger.info(f"Subsampling {self.subsample_size} samples for FNJ tree construction")
                subsample_indices = np.random.choice(
                    len(embeddings),
                    size=self.subsample_size,
                    replace=False
                )
                embeddings_for_tree = embeddings[subsample_indices]
            else:
                embeddings_for_tree = embeddings

            # Compute distance matrix with cosine metric
            distance_matrix = self.compute_distance_matrix(
                embeddings_for_tree,
                metric=self.distance_metric
            )

            # Build FNJ tree
            nj_tree = self.fast_neighbor_joining(distance_matrix)
            self.nj_tree = nj_tree

            # Extract tree features from FNJ tree
            tree_features = {
                'nj_tree': nj_tree,
                'node_depths': nj_tree['node_depths'],
                'n_samples': len(embeddings_for_tree)
            }

        elif self.tree_type == 'hierarchical':
            # Use hierarchical clustering (fallback)
            logger.info("Using hierarchical clustering for tree construction")
            linkage_matrix, subsample_indices = self.build_hierarchical_tree(
                embeddings,
                subsample=self.use_approx
            )

            # Extract tree features if we built a tree
            if linkage_matrix is not None:
                if subsample_indices is not None:
                    tree_features = self.extract_tree_features(
                        linkage_matrix,
                        embeddings[subsample_indices]
                    )
                else:
                    tree_features = self.extract_tree_features(
                        linkage_matrix,
                        embeddings
                    )

        else:
            logger.warning(f"Unknown tree_type: {self.tree_type}. Skipping tree construction.")

        graph_data = {
            'edge_index': edge_index,
            'edge_weights': edge_weights,
            'linkage_matrix': linkage_matrix,
            'nj_tree': nj_tree,
            'subsample_indices': subsample_indices,
            'tree_features': tree_features,
            'num_nodes': len(embeddings)
        }

        # Save cache
        if cache_path:
            logger.info(f"Saving graph to {cache_path}")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(graph_data, f)

        logger.info(f"Graph structure built: {graph_data['num_nodes']} nodes, "
                   f"{edge_index.shape[1]} edges")

        return graph_data
