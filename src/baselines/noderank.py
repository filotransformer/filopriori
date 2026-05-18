"""
NodeRank: Test Input Prioritization for Graph Neural Networks

Reference:
    Li, Y., et al. (2024). Test Input Prioritization for Graph Neural Networks.
    IEEE Transactions on Software Engineering, 50(5), 1178-1195.
    DOI: 10.1109/TSE.2024.3385538

Implementation adapted for Test Case Prioritization in CI/CD pipelines.
The original NodeRank is designed for GNN node classification, here we adapt
the mutation-based ensemble learning approach for test case prioritization.

Key concepts from the paper:
    1. Graph Structure Mutation (GSM): Add/modify edges in the graph
    2. Node Feature Mutation (NFM): Perturb node features
    3. GNN Model Mutation (GMM): Alter model parameters
    4. Ensemble Learning: Combine LR, RF, XGBoost, LightGBM for ranking
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, deque
import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


class GraphStructureMutation:
    """
    Graph Structure Mutation (GSM) from NodeRank paper.

    Mutates the test case similarity graph by adding random edges,
    which alters the interdependence between test cases.

    Formula: G' = G + sum_i addEdge(G, t, s_i)
    Where t is the target node and s_i are randomly selected neighbors.
    """

    def __init__(self, n_edges_to_add: int = 5, seed: Optional[int] = None):
        """
        Initialize GSM.

        Args:
            n_edges_to_add: Number of random edges to add per mutation
            seed: Random seed for reproducibility
        """
        self.n_edges = n_edges_to_add
        self.rng = np.random.RandomState(seed)

    def mutate(self, adjacency: np.ndarray) -> np.ndarray:
        """
        Apply graph structure mutation.

        Args:
            adjacency: Adjacency matrix (n_nodes x n_nodes)

        Returns:
            Mutated adjacency matrix
        """
        n = adjacency.shape[0]
        mutated = adjacency.copy()

        for _ in range(self.n_edges):
            # Select random source and target nodes
            i, j = self.rng.randint(0, n, size=2)
            if i != j:
                mutated[i, j] = 1
                mutated[j, i] = 1  # Undirected graph

        return mutated


class NodeFeatureMutation:
    """
    Node Feature Mutation (NFM) from NodeRank paper.

    Perturbs the feature vectors of test cases to evaluate sensitivity.

    Formula: F(T') = F(T) + alpha * F(T)
    Where alpha is a perturbation factor.
    """

    def __init__(self, alpha: float = 0.1, seed: Optional[int] = None):
        """
        Initialize NFM.

        Args:
            alpha: Perturbation factor (magnitude of noise)
            seed: Random seed for reproducibility
        """
        self.alpha = alpha
        self.rng = np.random.RandomState(seed)

    def mutate(self, features: np.ndarray) -> np.ndarray:
        """
        Apply node feature mutation.

        Args:
            features: Feature matrix (n_nodes x n_features)

        Returns:
            Mutated feature matrix
        """
        # F(T') = F(T) + alpha * F(T) * noise
        noise = self.rng.uniform(-1, 1, size=features.shape)
        mutated = features + self.alpha * np.abs(features) * noise
        return mutated


class ModelMutation:
    """
    GNN Model Mutation (GMM) from NodeRank paper.

    Creates model variants by altering hyperparameters.
    In the original paper, this includes:
    - LAB: Label shuffling
    - NS: Neurons per layer
    - CMA: Activation functions
    - CON: Layer configuration
    - ASL: Add silence layer
    - ALC: Add layer with copy weights
    - HC: Change hidden layer count

    Adapted for sklearn classifiers in this implementation.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def create_model_variants(self, base_params: Dict) -> List[Dict]:
        """
        Create multiple model parameter variants.

        Args:
            base_params: Base model parameters

        Returns:
            List of mutated parameter dictionaries
        """
        variants = []

        # Variant 1: Smaller model (reduced complexity)
        variant1 = base_params.copy()
        if 'n_estimators' in variant1:
            variant1['n_estimators'] = max(10, variant1.get('n_estimators', 100) // 2)
        if 'max_depth' in variant1:
            variant1['max_depth'] = max(2, variant1.get('max_depth', 5) - 2)
        variants.append(variant1)

        # Variant 2: Larger model (increased complexity)
        variant2 = base_params.copy()
        if 'n_estimators' in variant2:
            variant2['n_estimators'] = variant2.get('n_estimators', 100) * 2
        if 'max_depth' in variant2:
            variant2['max_depth'] = variant2.get('max_depth', 5) + 2
        variants.append(variant2)

        # Variant 3: Different learning rate
        variant3 = base_params.copy()
        if 'learning_rate' in variant3:
            variant3['learning_rate'] = variant3.get('learning_rate', 0.1) * 0.5
        variants.append(variant3)

        # Variant 4: Different regularization
        variant4 = base_params.copy()
        if 'C' in variant4:  # Logistic Regression
            variant4['C'] = variant4.get('C', 1.0) * 0.1
        if 'min_samples_split' in variant4:  # RF/GB
            variant4['min_samples_split'] = max(2, variant4.get('min_samples_split', 2) * 2)
        variants.append(variant4)

        return variants


class TestCaseGraphBuilder:
    """
    Builds a similarity graph from test case features.

    Uses k-NN to construct edges between similar test cases,
    simulating the graph structure needed for NodeRank.
    """

    def __init__(self, k_neighbors: int = 5, similarity_threshold: float = 0.5):
        """
        Initialize graph builder.

        Args:
            k_neighbors: Number of nearest neighbors for edges
            similarity_threshold: Minimum similarity for edge creation
        """
        self.k = k_neighbors
        self.threshold = similarity_threshold

    def build_adjacency(self, features: np.ndarray) -> np.ndarray:
        """
        Build adjacency matrix from features using k-NN.

        Args:
            features: Feature matrix (n_nodes x n_features)

        Returns:
            Adjacency matrix (n_nodes x n_nodes) - uses float32 to save memory
        """
        n = features.shape[0]

        if n <= self.k:
            # Small graph: connect all nodes
            adjacency = np.ones((n, n), dtype=np.float32) - np.eye(n, dtype=np.float32)
            return adjacency

        # Fit k-NN (use n_jobs=-1 for parallel)
        k_actual = min(self.k + 1, n)  # +1 because it includes self
        nn = NearestNeighbors(n_neighbors=k_actual, metric='cosine', n_jobs=-1)
        nn.fit(features)

        # Get neighbors and distances
        distances, indices = nn.kneighbors(features)

        # Build adjacency - use float32 to save memory (half of float64)
        adjacency = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j_idx, dist in zip(indices[i], distances[i]):
                if j_idx != i:  # Exclude self-loops
                    # Convert cosine distance to similarity
                    similarity = 1 - dist
                    if similarity >= self.threshold:
                        adjacency[i, j_idx] = 1
                        adjacency[j_idx, i] = 1  # Symmetric

        return adjacency


class NodeRankFeatureExtractor:
    """
    Extract features for NodeRank from historical test execution data.
    OPTIMIZED: O(1) per feature extraction using running statistics.

    Combines historical features with mutation-based features to create
    a rich representation for ranking.
    """

    def __init__(self, history_window: int = 10):
        self.history_window = history_window
        # Running statistics — all O(1) per update and extract
        self._recent_verdicts = defaultdict(lambda: deque(maxlen=history_window))
        self._old_half_failures = defaultdict(int)  # failures in older half
        self._old_half_count = defaultdict(int)      # count of older half
        self._exec_count = defaultdict(int)
        self._failure_count = defaultdict(int)
        self._duration_sum = defaultdict(float)
        self._max_duration = defaultdict(float)
        self._last_verdict = {}
        self._last_duration = {}
        self._consecutive_same = defaultdict(lambda: 1)
        self._time_since_failure = {}
        self.n_builds = 0

    def update_history(
        self,
        build_id: str,
        test_results: Dict[str, Tuple[int, float]]
    ):
        """Update test history with results from a build — O(n_tests) total."""
        self.n_builds += 1

        for test_id, (verdict, duration) in test_results.items():
            n_exec = self._exec_count[test_id]

            self._recent_verdicts[test_id].append(verdict)
            self._exec_count[test_id] += 1
            if verdict == 1:
                self._failure_count[test_id] += 1

            self._duration_sum[test_id] += duration
            if duration > self._max_duration[test_id]:
                self._max_duration[test_id] = duration

            # Track old-half failures for velocity computation
            # When n_exec crosses the midpoint, the previous "new half" value
            # becomes the "old half" — approximate by tracking running old half
            half = (n_exec + 1) // 2
            if n_exec > 0 and n_exec % 2 == 0:
                # Approximate: shift the midpoint by including one more in old half
                # Use the overall failure rate to estimate
                self._old_half_count[test_id] = half
                self._old_half_failures[test_id] = int(
                    self._failure_count[test_id] * half / (n_exec + 1)
                )

            # Consecutive same
            if test_id in self._last_verdict:
                if self._last_verdict[test_id] == verdict:
                    self._consecutive_same[test_id] += 1
                else:
                    self._consecutive_same[test_id] = 1

            # Time since failure
            if verdict == 1:
                self._time_since_failure[test_id] = 0
            elif test_id in self._time_since_failure:
                self._time_since_failure[test_id] += 1

            self._last_verdict[test_id] = verdict
            self._last_duration[test_id] = duration

    def extract_base_features(self, test_id: str) -> np.ndarray:
        """Extract base historical features for a test case — O(1)."""
        n_exec = self._exec_count.get(test_id, 0)
        if n_exec == 0:
            return np.array([
                0.5, 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0, 0.0
            ], dtype=np.float32)

        last_verdict = float(self._last_verdict[test_id])

        last_dur = self._last_duration[test_id]
        max_dur = self._max_duration[test_id]
        last_duration_norm = last_dur / (max_dur + 1e-6)

        failure_rate = self._failure_count[test_id] / n_exec

        recent = self._recent_verdicts.get(test_id)
        recent_failure_rate = sum(recent) / len(recent) if recent else 0.0

        exec_count_norm = n_exec / (self.n_builds + 1)

        avg_duration = self._duration_sum[test_id] / n_exec
        avg_duration_norm = avg_duration / (max_dur + 1e-6)

        tsf = self._time_since_failure.get(test_id, n_exec)
        time_since_failure_norm = tsf / (n_exec + 1)

        consecutive_same = self._consecutive_same.get(test_id, 1)
        consecutive_same_norm = consecutive_same / (n_exec + 1)

        # Failure velocity (trend) — approximate O(1)
        if n_exec >= 3:
            old_half_n = self._old_half_count.get(test_id, n_exec // 2)
            old_half_f = self._old_half_failures.get(test_id, 0)
            old_rate = old_half_f / max(old_half_n, 1)
            new_half_n = n_exec - old_half_n
            new_half_f = self._failure_count[test_id] - old_half_f
            new_rate = new_half_f / max(new_half_n, 1)
            failure_velocity = new_rate - old_rate
        else:
            failure_velocity = 0.0

        # Execution regularity
        exec_regularity = 1.0 - (self.n_builds - n_exec) / (self.n_builds + 1)

        return np.array([
            last_verdict,
            last_duration_norm,
            failure_rate,
            recent_failure_rate,
            exec_count_norm,
            avg_duration_norm,
            time_since_failure_norm,
            consecutive_same_norm,
            failure_velocity,
            exec_regularity
        ], dtype=np.float32)


class NodeRankModel:
    """
    NodeRank model for test case prioritization.

    Implements the mutation-based ensemble learning approach from the paper:
    1. Build a test case similarity graph
    2. Apply three types of mutations (GSM, NFM, GMM)
    3. Generate mutation-based features (kill vectors)
    4. Train ensemble of classifiers (LR, RF, XGBoost, LightGBM)
    5. Rank test cases by predicted failure probability
    """

    def __init__(
        self,
        n_gsm_mutations: int = 5,
        n_nfm_mutations: int = 5,
        n_gmm_variants: int = 4,
        gsm_edges: int = 3,
        nfm_alpha: float = 0.1,
        k_neighbors: int = 5,
        history_window: int = 10,
        use_ensemble: bool = True,
        seed: Optional[int] = 42
    ):
        """
        Initialize NodeRank model.

        Args:
            n_gsm_mutations: Number of graph structure mutations
            n_nfm_mutations: Number of node feature mutations
            n_gmm_variants: Number of model variants
            gsm_edges: Edges to add per GSM mutation
            nfm_alpha: Perturbation factor for NFM
            k_neighbors: Neighbors for graph construction
            history_window: Window for historical features
            use_ensemble: Whether to use ensemble ranking
            seed: Random seed
        """
        self.n_gsm = n_gsm_mutations
        self.n_nfm = n_nfm_mutations
        self.n_gmm = n_gmm_variants
        self.gsm_edges = gsm_edges
        self.nfm_alpha = nfm_alpha
        self.k_neighbors = k_neighbors
        self.use_ensemble = use_ensemble
        self.seed = seed

        self.feature_extractor = NodeRankFeatureExtractor(history_window)
        self.graph_builder = TestCaseGraphBuilder(k_neighbors)
        self.scaler = StandardScaler()

        # Mutation operators
        self.gsm = GraphStructureMutation(gsm_edges, seed)
        self.nfm = NodeFeatureMutation(nfm_alpha, seed)
        self.gmm = ModelMutation(seed)

        # Ensemble models
        self.models = {}
        self.trained = False

    def _create_base_classifier(self, clf_type: str, params: Optional[Dict] = None) -> Any:
        """Create a base classifier of the specified type."""
        params = params or {}

        if clf_type == 'lr':
            return LogisticRegression(
                C=params.get('C', 1.0),
                max_iter=1000,
                random_state=self.seed,
                class_weight='balanced'
            )
        elif clf_type == 'rf':
            return RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 5),
                min_samples_split=params.get('min_samples_split', 2),
                random_state=self.seed,
                class_weight='balanced',
                n_jobs=-1
            )
        elif clf_type == 'gb':
            return GradientBoostingClassifier(
                n_estimators=params.get('n_estimators', 50),
                max_depth=params.get('max_depth', 3),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=self.seed
            )
        elif clf_type == 'xgb' and HAS_XGBOOST:
            return xgb.XGBClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 5),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=self.seed,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=params.get('scale_pos_weight', 1.0)
            )
        elif clf_type == 'lgb' and HAS_LIGHTGBM:
            return lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 5),
                learning_rate=params.get('learning_rate', 0.1),
                random_state=self.seed,
                class_weight='balanced',
                verbose=-1
            )
        else:
            # Fallback to RF
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.seed,
                class_weight='balanced'
            )

    def _generate_mutation_features(
        self,
        base_features: np.ndarray,
        adjacency: np.ndarray,
        labels: np.ndarray,
        lightweight: bool = False
    ) -> np.ndarray:
        """
        Generate mutation-based features (kill vectors).

        For each test case, generates a binary vector indicating
        whether mutations caused prediction changes.

        Args:
            base_features: Base feature matrix
            adjacency: Adjacency matrix
            labels: True labels
            lightweight: If True, use faster/simpler models

        Returns:
            Mutation feature matrix
        """
        n_samples = base_features.shape[0]
        n_mutations = self.n_gsm + self.n_nfm + self.n_gmm
        mutation_features = np.zeros((n_samples, n_mutations))

        if n_samples < 5:
            return mutation_features

        # Use lightweight model params for batch processing
        if lightweight:
            light_params = {'n_estimators': 20, 'max_depth': 3, 'n_jobs': -1}
        else:
            light_params = {'n_estimators': 50, 'max_depth': 4, 'n_jobs': -1}

        # Train a base model on original data
        try:
            base_model = self._create_base_classifier('rf', light_params)
            base_model.fit(base_features, labels)
            base_predictions = base_model.predict(base_features)
        except Exception:
            return mutation_features

        feat_idx = 0

        # GSM mutations
        for _ in range(self.n_gsm):
            try:
                mutated_adj = self.gsm.mutate(adjacency)
                # Propagate adjacency influence to features
                adj_normalized = mutated_adj / (mutated_adj.sum(axis=1, keepdims=True) + 1e-6)
                mutated_features = np.dot(adj_normalized, base_features)

                # Train on mutated features
                mutant_model = self._create_base_classifier('rf', light_params)
                mutant_model.fit(mutated_features, labels)
                mutant_predictions = mutant_model.predict(mutated_features)

                # Kill vector: 1 if prediction changed
                mutation_features[:, feat_idx] = (base_predictions != mutant_predictions).astype(float)
            except Exception:
                pass
            feat_idx += 1

        # NFM mutations
        for _ in range(self.n_nfm):
            try:
                mutated_features = self.nfm.mutate(base_features)

                mutant_model = self._create_base_classifier('rf', light_params)
                mutant_model.fit(mutated_features, labels)
                mutant_predictions = mutant_model.predict(mutated_features)

                mutation_features[:, feat_idx] = (base_predictions != mutant_predictions).astype(float)
            except Exception:
                pass
            feat_idx += 1

        # GMM mutations (model variants)
        base_params = {'n_estimators': 50, 'max_depth': 4}
        variants = self.gmm.create_model_variants(base_params)

        for variant_params in variants[:self.n_gmm]:
            try:
                variant_params['n_jobs'] = -1
                mutant_model = self._create_base_classifier('rf', variant_params)
                mutant_model.fit(base_features, labels)
                mutant_predictions = mutant_model.predict(base_features)

                mutation_features[:, feat_idx] = (base_predictions != mutant_predictions).astype(float)
            except Exception:
                pass
            feat_idx += 1

        return mutation_features

    def _prepare_training_data(
        self,
        df: pd.DataFrame,
        build_col: str,
        test_col: str,
        result_col: str,
        duration_col: Optional[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from DataFrame.

        OPTIMIZED VERSION:
        - First pass: collect all base features and update history
        - Second pass: generate mutation features in batch (once!)
        """
        builds = df[build_col].unique().tolist()
        n_builds = len(builds)
        grouped = df.groupby(build_col, sort=False)

        print(f"NodeRank: Processing {n_builds} builds...")

        # ========== PHASE 1: Collect base features and update history ==========
        all_base_features = []
        all_labels = []

        for i, build_id in enumerate(builds):
            if (i + 1) % 500 == 0 or i == 0:
                print(f"  Phase 1: Collecting features... {i+1}/{n_builds} builds")

            build_df = grouped.get_group(build_id)
            test_ids = build_df[test_col].values
            result_vals = build_df[result_col].values
            dur_vals = build_df[duration_col].values if duration_col and duration_col in build_df.columns else np.ones(len(build_df))

            # Update history FIRST (so subsequent builds can use it)
            test_results = {}
            for j in range(len(test_ids)):
                test_id = test_ids[j]
                verdict_raw = str(result_vals[j]).strip().lower()
                verdict = 1 if verdict_raw == 'fail' else 0
                duration = float(dur_vals[j])
                prev_verdict, _ = test_results.get(test_id, (0, duration))
                test_results[test_id] = (max(prev_verdict, verdict), duration)
            self.feature_extractor.update_history(build_id, test_results)

            # Extract base features for each test
            for j in range(len(test_ids)):
                test_id = test_ids[j]
                features = self.feature_extractor.extract_base_features(test_id)
                all_base_features.append(features)

                verdict_raw = str(result_vals[j]).strip().lower()
                verdict = 1 if verdict_raw == 'fail' else 0
                verdict = max(verdict, test_results.get(test_id, (0, 1.0))[0])
                all_labels.append(verdict)

        all_base_features = np.array(all_base_features)
        all_labels = np.array(all_labels)

        print(f"  Phase 1 complete: {len(all_base_features)} samples collected")

        # ========== PHASE 2: Generate mutation features in ONE batch ==========
        print("  Phase 2: Generating mutation features (batch)...")

        # Sample if too large (for speed AND memory)
        # 10K samples = 10K x 10K matrix = 800MB (manageable)
        max_samples_for_mutations = 10000
        if len(all_base_features) > max_samples_for_mutations:
            print(f"    Sampling {max_samples_for_mutations} from {len(all_base_features)} for mutations")
            sample_idx = np.random.choice(
                len(all_base_features),
                max_samples_for_mutations,
                replace=False
            )
            sample_features = all_base_features[sample_idx]
            sample_labels = all_labels[sample_idx]
        else:
            sample_features = all_base_features
            sample_labels = all_labels
            sample_idx = None

        # Build graph on sampled data
        print("    Building k-NN graph...")
        adjacency = self.graph_builder.build_adjacency(sample_features)

        # Generate mutation features (only ONCE, not per-build!)
        print("    Training mutation models...")
        sample_mutation_feats = self._generate_mutation_features(
            sample_features, adjacency, sample_labels, lightweight=True
        )

        # If we sampled, we need to generate mutation features for ALL data
        # using a simpler approach (nearest neighbor interpolation)
        n_mutations = self.n_gsm + self.n_nfm + self.n_gmm

        if sample_idx is not None:
            print("    Interpolating mutation features for full dataset...")
            # Use simple approach: assign based on nearest sampled neighbor
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1)
            nn.fit(sample_features)
            _, nearest_idx = nn.kneighbors(all_base_features)
            nearest_idx = nearest_idx.flatten()
            all_mutation_feats = sample_mutation_feats[nearest_idx]
        else:
            all_mutation_feats = sample_mutation_feats

        # Combine base and mutation features
        print("    Combining features...")
        all_features = np.hstack([all_base_features, all_mutation_feats])

        print(f"  Phase 2 complete: Final feature matrix shape = {all_features.shape}")

        return all_features, all_labels

    def train(
        self,
        df: pd.DataFrame,
        build_col: str = 'Build_ID',
        test_col: str = 'TC_Key',
        result_col: str = 'TE_Test_Result',
        duration_col: Optional[str] = None
    ):
        """
        Train the NodeRank model.

        Args:
            df: Training DataFrame
            build_col: Column name for build ID
            test_col: Column name for test case ID
            result_col: Column name for test result
            duration_col: Optional column for test duration
        """
        # Reset feature extractor
        self.feature_extractor = NodeRankFeatureExtractor()

        print("NodeRank: Preparing training data...")
        X, y = self._prepare_training_data(df, build_col, test_col, result_col, duration_col)

        if len(X) == 0 or len(np.unique(y)) < 2:
            print("NodeRank: Insufficient training data")
            self.trained = False
            return

        print(f"NodeRank: Training on {len(X)} samples")
        print(f"NodeRank: Failure rate = {y.mean():.4f}")

        # Scale features
        X = self.scaler.fit_transform(X)

        # Calculate scale_pos_weight for imbalanced data
        pos_weight = (1 - y.mean()) / (y.mean() + 1e-6)

        # Train ensemble models
        print("NodeRank: Training ensemble models...")

        if self.use_ensemble:
            # Logistic Regression
            try:
                self.models['lr'] = self._create_base_classifier('lr')
                self.models['lr'].fit(X, y)
            except Exception as e:
                print(f"NodeRank: LR training failed - {e}")

            # Random Forest
            try:
                self.models['rf'] = self._create_base_classifier('rf')
                self.models['rf'].fit(X, y)
            except Exception as e:
                print(f"NodeRank: RF training failed - {e}")

            # XGBoost
            if HAS_XGBOOST:
                try:
                    self.models['xgb'] = self._create_base_classifier(
                        'xgb', {'scale_pos_weight': pos_weight}
                    )
                    self.models['xgb'].fit(X, y)
                except Exception as e:
                    print(f"NodeRank: XGBoost training failed - {e}")

            # LightGBM
            if HAS_LIGHTGBM:
                try:
                    self.models['lgb'] = self._create_base_classifier('lgb')
                    self.models['lgb'].fit(X, y)
                except Exception as e:
                    print(f"NodeRank: LightGBM training failed - {e}")

            # Gradient Boosting (fallback)
            try:
                self.models['gb'] = self._create_base_classifier('gb')
                self.models['gb'].fit(X, y)
            except Exception as e:
                print(f"NodeRank: GB training failed - {e}")
        else:
            # Single model mode
            self.models['rf'] = self._create_base_classifier('rf')
            self.models['rf'].fit(X, y)

        self.trained = True
        print(f"NodeRank: Trained {len(self.models)} models")

    def prioritize(
        self,
        test_ids: List[str],
        build_features: Optional[np.ndarray] = None
    ) -> List[str]:
        """
        Prioritize test cases based on ensemble prediction.

        Args:
            test_ids: List of test IDs to prioritize
            build_features: Optional pre-computed features

        Returns:
            Ordered list of test IDs (highest probability first)
        """
        if not self.trained or len(self.models) == 0:
            # Fallback to historical failure rate
            return self._fallback_prioritize(test_ids)

        # Extract features for each test
        if build_features is None:
            features = []
            for test_id in test_ids:
                feat = self.feature_extractor.extract_base_features(test_id)
                features.append(feat)
            features = np.array(features)

            # Add placeholder mutation features (zeros for new data)
            n_mutations = self.n_gsm + self.n_nfm + self.n_gmm
            mutation_feats = np.zeros((len(test_ids), n_mutations))
            features = np.hstack([features, mutation_feats])
        else:
            features = build_features

        # Scale features
        try:
            features = self.scaler.transform(features)
        except Exception:
            pass

        # Get ensemble predictions
        predictions = np.zeros(len(test_ids))
        n_models = 0

        for name, model in self.models.items():
            try:
                proba = model.predict_proba(features)[:, 1]
                predictions += proba
                n_models += 1
            except Exception:
                pass

        if n_models > 0:
            predictions /= n_models
        else:
            return self._fallback_prioritize(test_ids)

        # Sort by prediction (descending)
        test_scores = list(zip(test_ids, predictions))
        test_scores.sort(key=lambda x: x[1], reverse=True)

        return [t[0] for t in test_scores]

    def _fallback_prioritize(self, test_ids: List[str]) -> List[str]:
        """Fallback prioritization based on historical failure rate."""
        test_scores = []
        fe = self.feature_extractor
        for test_id in test_ids:
            n_exec = fe._exec_count.get(test_id, 0)
            if n_exec > 0:
                fail_rate = fe._failure_count[test_id] / n_exec
            else:
                fail_rate = 0.5  # Unknown tests get medium priority
            test_scores.append((test_id, fail_rate))

        test_scores.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in test_scores]

    def update_history(
        self,
        build_id: str,
        test_results: Dict[str, Tuple[int, float]]
    ):
        """Update history after evaluation."""
        self.feature_extractor.update_history(build_id, test_results)


def run_noderank_experiment(
    df: pd.DataFrame,
    build_col: str = 'Build_ID',
    test_col: str = 'TC_Key',
    result_col: str = 'TE_Test_Result',
    duration_col: Optional[str] = None,
    train_ratio: float = 0.8,
    n_gsm: int = 5,
    n_nfm: int = 5,
    n_gmm: int = 4,
    use_ensemble: bool = True,
    seed: int = 42
) -> Dict:
    """
    Run NodeRank experiment on a dataset.

    Args:
        df: DataFrame with test execution data
        build_col: Column name for build ID
        test_col: Column name for test case ID
        result_col: Column name for test result
        duration_col: Optional column for test duration
        train_ratio: Ratio of builds for training
        n_gsm: Number of graph structure mutations
        n_nfm: Number of node feature mutations
        n_gmm: Number of model variants
        use_ensemble: Whether to use ensemble ranking
        seed: Random seed

    Returns:
        Dict with APFD scores and statistics
    """
    # Get unique builds in order
    builds = df[build_col].unique().tolist()

    # Split: first X% for training, rest for evaluation
    train_idx = int(len(builds) * train_ratio)
    train_builds = builds[:train_idx]
    test_builds = builds[train_idx:]

    # Filter test builds to those with failures
    test_builds_with_failures = []
    for build_id in test_builds:
        build_df = df[df[build_col] == build_id]
        if build_df[result_col].apply(lambda x: str(x).upper() != 'PASS').sum() > 0:
            test_builds_with_failures.append(build_id)

    print(f"NodeRank: Training on {len(train_builds)} builds")
    print(f"NodeRank: Evaluating on {len(test_builds_with_failures)} builds with failures")

    # Create train DataFrame
    train_df = df[df[build_col].isin(train_builds)]

    # Initialize and train model
    model = NodeRankModel(
        n_gsm_mutations=n_gsm,
        n_nfm_mutations=n_nfm,
        n_gmm_variants=n_gmm,
        use_ensemble=use_ensemble,
        seed=seed
    )
    model.train(train_df, build_col, test_col, result_col, duration_col)

    # Evaluation phase
    apfd_scores = []
    build_results = []

    for build_id in test_builds_with_failures:
        build_df = df[df[build_col] == build_id]
        test_ids = build_df[test_col].unique().tolist()

        # Get verdicts
        verdicts = {}
        for _, row in build_df.iterrows():
            test_id = row[test_col]
            verdict = 1 if str(row[result_col]).upper() != 'PASS' else 0
            verdicts[test_id] = verdict

        # Prioritize
        ranking = model.prioritize(test_ids)

        # Compute APFD
        n_tests = len(ranking)
        n_faults = sum(verdicts.values())

        if n_faults > 0 and n_tests > 1:
            fault_positions = []
            for i, test_id in enumerate(ranking):
                if verdicts.get(test_id, 0) == 1:
                    fault_positions.append(i + 1)

            apfd = 1 - (sum(fault_positions) / (n_tests * n_faults)) + 1 / (2 * n_tests)
            apfd_scores.append(apfd)

            build_results.append({
                'build_id': build_id,
                'n_tests': n_tests,
                'n_faults': n_faults,
                'apfd': apfd
            })
        elif n_tests == 1 and n_faults == 1:
            # Single test case that fails
            apfd_scores.append(1.0)
            build_results.append({
                'build_id': build_id,
                'n_tests': n_tests,
                'n_faults': n_faults,
                'apfd': 1.0
            })

        # Update history for online learning
        test_results = {}
        for test_id, verdict in verdicts.items():
            duration = 1.0
            if duration_col:
                test_row = build_df[build_df[test_col] == test_id]
                if len(test_row) > 0 and duration_col in test_row.columns:
                    duration = test_row[duration_col].values[0]
            test_results[test_id] = (verdict, duration)
        model.update_history(build_id, test_results)

    results = {
        'method': 'NodeRank',
        'apfd_scores': apfd_scores,
        'mean_apfd': np.mean(apfd_scores) if apfd_scores else 0,
        'std_apfd': np.std(apfd_scores) if apfd_scores else 0,
        'median_apfd': np.median(apfd_scores) if apfd_scores else 0,
        'min_apfd': np.min(apfd_scores) if apfd_scores else 0,
        'max_apfd': np.max(apfd_scores) if apfd_scores else 0,
        'n_builds': len(apfd_scores),
        'build_results': build_results,
        'config': {
            'n_gsm': n_gsm,
            'n_nfm': n_nfm,
            'n_gmm': n_gmm,
            'use_ensemble': use_ensemble,
            'train_ratio': train_ratio
        }
    }

    print(f"\nNodeRank Results:")
    print(f"  Mean APFD:   {results['mean_apfd']:.4f} (+/- {results['std_apfd']:.4f})")
    print(f"  Median APFD: {results['median_apfd']:.4f}")
    print(f"  Range:       [{results['min_apfd']:.4f}, {results['max_apfd']:.4f}]")
    print(f"  Builds:      {results['n_builds']}")

    return results


def compare_with_baselines(
    df: pd.DataFrame,
    build_col: str = 'Build_ID',
    test_col: str = 'TC_Key',
    result_col: str = 'TE_Test_Result',
    duration_col: Optional[str] = None,
    train_ratio: float = 0.8
) -> pd.DataFrame:
    """
    Compare NodeRank with baseline methods.

    Args:
        df: DataFrame with test execution data
        build_col: Column name for build ID
        test_col: Column name for test case ID
        result_col: Column name for test result
        duration_col: Optional column for test duration
        train_ratio: Ratio of builds for training

    Returns:
        DataFrame with comparison results
    """
    results = []

    # NodeRank (full)
    print("\n" + "="*60)
    print("Running NodeRank (Full Ensemble)")
    print("="*60)
    noderank_full = run_noderank_experiment(
        df, build_col, test_col, result_col, duration_col,
        train_ratio=train_ratio, use_ensemble=True
    )
    results.append({
        'method': 'NodeRank (Ensemble)',
        'mean_apfd': noderank_full['mean_apfd'],
        'std_apfd': noderank_full['std_apfd'],
        'median_apfd': noderank_full['median_apfd'],
        'n_builds': noderank_full['n_builds']
    })

    # NodeRank (single model)
    print("\n" + "="*60)
    print("Running NodeRank (Single RF)")
    print("="*60)
    noderank_single = run_noderank_experiment(
        df, build_col, test_col, result_col, duration_col,
        train_ratio=train_ratio, use_ensemble=False
    )
    results.append({
        'method': 'NodeRank (Single)',
        'mean_apfd': noderank_single['mean_apfd'],
        'std_apfd': noderank_single['std_apfd'],
        'median_apfd': noderank_single['median_apfd'],
        'n_builds': noderank_single['n_builds']
    })

    # Random baseline
    print("\n" + "="*60)
    print("Running Random Baseline")
    print("="*60)
    random_results = run_random_baseline(df, build_col, test_col, result_col, train_ratio)
    results.append({
        'method': 'Random',
        'mean_apfd': random_results['mean_apfd'],
        'std_apfd': random_results['std_apfd'],
        'median_apfd': random_results['median_apfd'],
        'n_builds': random_results['n_builds']
    })

    # Failure Rate baseline
    print("\n" + "="*60)
    print("Running Failure Rate Baseline")
    print("="*60)
    failure_rate_results = run_failure_rate_baseline(
        df, build_col, test_col, result_col, train_ratio
    )
    results.append({
        'method': 'Failure Rate',
        'mean_apfd': failure_rate_results['mean_apfd'],
        'std_apfd': failure_rate_results['std_apfd'],
        'median_apfd': failure_rate_results['median_apfd'],
        'n_builds': failure_rate_results['n_builds']
    })

    return pd.DataFrame(results)


def run_random_baseline(
    df: pd.DataFrame,
    build_col: str,
    test_col: str,
    result_col: str,
    train_ratio: float
) -> Dict:
    """Run random prioritization baseline."""
    builds = df[build_col].unique().tolist()
    train_idx = int(len(builds) * train_ratio)
    test_builds = builds[train_idx:]

    apfd_scores = []
    rng = np.random.RandomState(42)

    for build_id in test_builds:
        build_df = df[df[build_col] == build_id]
        test_ids = build_df[test_col].unique().tolist()

        verdicts = {}
        for _, row in build_df.iterrows():
            test_id = row[test_col]
            verdict = 1 if str(row[result_col]).upper() != 'PASS' else 0
            verdicts[test_id] = verdict

        n_faults = sum(verdicts.values())
        if n_faults == 0:
            continue

        # Random ranking
        ranking = test_ids.copy()
        rng.shuffle(ranking)

        n_tests = len(ranking)
        if n_tests > 1:
            fault_positions = []
            for i, test_id in enumerate(ranking):
                if verdicts.get(test_id, 0) == 1:
                    fault_positions.append(i + 1)

            apfd = 1 - (sum(fault_positions) / (n_tests * n_faults)) + 1 / (2 * n_tests)
            apfd_scores.append(apfd)
        elif n_tests == 1:
            apfd_scores.append(1.0)

    return {
        'method': 'Random',
        'mean_apfd': np.mean(apfd_scores) if apfd_scores else 0,
        'std_apfd': np.std(apfd_scores) if apfd_scores else 0,
        'median_apfd': np.median(apfd_scores) if apfd_scores else 0,
        'n_builds': len(apfd_scores)
    }


def run_failure_rate_baseline(
    df: pd.DataFrame,
    build_col: str,
    test_col: str,
    result_col: str,
    train_ratio: float
) -> Dict:
    """Run failure rate baseline (historical failure frequency)."""
    builds = df[build_col].unique().tolist()
    train_idx = int(len(builds) * train_ratio)
    train_builds = builds[:train_idx]
    test_builds = builds[train_idx:]

    # Compute historical failure rates from training data
    train_df = df[df[build_col].isin(train_builds)]
    failure_counts = defaultdict(int)
    exec_counts = defaultdict(int)

    for _, row in train_df.iterrows():
        test_id = row[test_col]
        verdict = 1 if str(row[result_col]).upper() != 'PASS' else 0
        exec_counts[test_id] += 1
        failure_counts[test_id] += verdict

    failure_rates = {
        t: failure_counts[t] / exec_counts[t] if exec_counts[t] > 0 else 0.5
        for t in exec_counts
    }

    # Evaluate
    apfd_scores = []

    for build_id in test_builds:
        build_df = df[df[build_col] == build_id]
        test_ids = build_df[test_col].unique().tolist()

        verdicts = {}
        for _, row in build_df.iterrows():
            test_id = row[test_col]
            verdict = 1 if str(row[result_col]).upper() != 'PASS' else 0
            verdicts[test_id] = verdict

        n_faults = sum(verdicts.values())
        if n_faults == 0:
            continue

        # Rank by failure rate
        test_scores = [(t, failure_rates.get(t, 0.5)) for t in test_ids]
        test_scores.sort(key=lambda x: x[1], reverse=True)
        ranking = [t[0] for t in test_scores]

        n_tests = len(ranking)
        if n_tests > 1:
            fault_positions = []
            for i, test_id in enumerate(ranking):
                if verdicts.get(test_id, 0) == 1:
                    fault_positions.append(i + 1)

            apfd = 1 - (sum(fault_positions) / (n_tests * n_faults)) + 1 / (2 * n_tests)
            apfd_scores.append(apfd)
        elif n_tests == 1:
            apfd_scores.append(1.0)

        # Update rates for online learning
        for test_id, verdict in verdicts.items():
            exec_counts[test_id] += 1
            failure_counts[test_id] += verdict
            failure_rates[test_id] = failure_counts[test_id] / exec_counts[test_id]

    return {
        'method': 'Failure Rate',
        'mean_apfd': np.mean(apfd_scores) if apfd_scores else 0,
        'std_apfd': np.std(apfd_scores) if apfd_scores else 0,
        'median_apfd': np.median(apfd_scores) if apfd_scores else 0,
        'n_builds': len(apfd_scores)
    }


if __name__ == '__main__':
    print("NodeRank Implementation for Test Case Prioritization")
    print("Based on: Li et al. (2024) - Test Input Prioritization for GNNs")
    print("IEEE TSE, DOI: 10.1109/TSE.2024.3385538")
    print()
    print("Usage:")
    print("  from src.baselines.noderank import run_noderank_experiment")
    print("  results = run_noderank_experiment(df)")
