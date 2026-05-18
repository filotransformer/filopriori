"""
FailRank-BB: Test Case Prioritization using SBERT Embeddings + Logistic Regression

Reference:
    Hernandes, V. et al. (2024). FailRank-BB: Test Case Prioritization
    using BERT Embeddings and Binary Classification.

The paper proposes encoding test case steps and commit messages with SBERT,
concatenating the embeddings, then training a LogisticRegression classifier
to predict failure probability. Tests are ordered by P(fail) descending.

Re-implemented from the original paper's methodology.
"""

import ast

import numpy as np
import pandas as pd
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional


class FailRankBBFeatureExtractor:
    """
    O(1) running statistics for historical test execution features.

    Only used when use_historical_features=True.
    Same pattern as DeepOrder's feature extractor.

    Features (8 float32):
    1. Failure rate (historical)
    2. Recent failure rate (last N builds)
    3. Average duration (normalized)
    4. Max duration (normalized)
    5. Execution count (normalized)
    6. Last verdict (0/1)
    7. Time since last failure (normalized)
    8. Consecutive same verdict count (normalized)
    """

    def __init__(self, history_window: int = 10):
        self.history_window = history_window
        self._recent_verdicts = defaultdict(lambda: deque(maxlen=history_window))
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
            self._recent_verdicts[test_id].append(verdict)
            self._exec_count[test_id] += 1
            if verdict == 1:
                self._failure_count[test_id] += 1

            self._duration_sum[test_id] += duration
            if duration > self._max_duration[test_id]:
                self._max_duration[test_id] = duration

            if test_id in self._last_verdict:
                if self._last_verdict[test_id] == verdict:
                    self._consecutive_same[test_id] += 1
                else:
                    self._consecutive_same[test_id] = 1

            if verdict == 1:
                self._time_since_failure[test_id] = 0
            elif test_id in self._time_since_failure:
                self._time_since_failure[test_id] += 1

            self._last_verdict[test_id] = verdict
            self._last_duration[test_id] = duration

    def extract_features(self, test_id: str) -> np.ndarray:
        """Extract 8 historical features for a test case — O(1)."""
        n_exec = self._exec_count.get(test_id, 0)
        if n_exec == 0:
            return np.zeros(8, dtype=np.float32)

        failure_rate = self._failure_count[test_id] / n_exec

        recent = self._recent_verdicts.get(test_id)
        recent_failure_rate = sum(recent) / len(recent) if recent else 0.0

        max_dur = self._max_duration[test_id]
        avg_duration = self._duration_sum[test_id] / n_exec
        avg_duration_norm = avg_duration / (max_dur + 1e-6)
        max_duration_norm = max_dur / (max_dur + 1e-6)  # always ~1 if any duration

        exec_count_norm = n_exec / (self.n_builds + 1)

        last_verdict = float(self._last_verdict[test_id])

        tsf = self._time_since_failure.get(test_id, n_exec)
        time_since_failure_norm = tsf / (n_exec + 1)

        consecutive_same = self._consecutive_same.get(test_id, 1)
        consecutive_same_norm = consecutive_same / (n_exec + 1)

        return np.array([
            failure_rate,
            recent_failure_rate,
            avg_duration_norm,
            max_duration_norm,
            exec_count_norm,
            last_verdict,
            time_since_failure_norm,
            consecutive_same_norm
        ], dtype=np.float32)


class FailRankBBModel:
    """
    FailRank-BB model for test case prioritization.

    Uses SBERT embeddings of test case text and commit messages,
    concatenated and fed to a LogisticRegression classifier to
    predict P(Fail). Tests are ranked by descending P(Fail).
    """

    def __init__(
        self,
        sbert_model_name: str = 'all-MiniLM-L6-v2',
        history_window: int = 10,
        max_iter: int = 1000,
        use_historical_features: bool = False,
        seed: int = 42,
        batch_size: int = 64,
        device: str = 'auto'
    ):
        """
        Initialize FailRank-BB model.

        Args:
            sbert_model_name: SentenceTransformer model name or path.
                Default 'all-MiniLM-L6-v2' (384-dim, fast).
                For paper-faithful: 'bert-base-uncased' (768-dim).
            history_window: Window for recent history features.
            max_iter: Max iterations for LogisticRegression.
            use_historical_features: If True, append 8 history features
                to the embedding vector. Paper uses only embeddings.
            seed: Random seed for reproducibility.
            batch_size: Batch size for SBERT encoding.
            device: Device for SBERT ('auto', 'cpu', 'cuda', 'cuda:0', etc.).
        """
        self.sbert_model_name = sbert_model_name
        self.history_window = history_window
        self.max_iter = max_iter
        self.use_historical_features = use_historical_features
        self.seed = seed
        self.batch_size = batch_size
        self.device = device

        self.feature_extractor = FailRankBBFeatureExtractor(history_window)
        self._sbert_model = None
        self._tc_embedding_cache = {}  # test_id/text -> np.ndarray
        self._commit_embedding_cache = {}  # commit_text -> np.ndarray
        self._embedding_dim = None
        self._classifier = None  # LogisticRegression

    def _load_sbert(self):
        """Lazy-load SentenceTransformer model."""
        if self._sbert_model is not None:
            return

        from sentence_transformers import SentenceTransformer

        device = self.device
        if device == 'auto':
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # For bert-base-uncased (not a sentence-transformer), wrap with
        # Transformer + Pooling modules to match reference implementation
        if 'bert-base' in self.sbert_model_name and 'sentence' not in self.sbert_model_name.lower():
            from sentence_transformers import models as st_models
            transformer = st_models.Transformer(self.sbert_model_name)
            pooling = st_models.Pooling(
                transformer.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True
            )
            self._sbert_model = SentenceTransformer(
                modules=[transformer, pooling], device=device
            )
        else:
            self._sbert_model = SentenceTransformer(
                self.sbert_model_name, device=device
            )

        self._embedding_dim = self._sbert_model.get_sentence_embedding_dimension()
        print(f"FailRank-BB: Loaded SBERT model '{self.sbert_model_name}' "
              f"(dim={self._embedding_dim}, device={device})")

    def _unload_sbert(self):
        """Free SBERT model from memory (keep caches)."""
        if self._sbert_model is not None:
            del self._sbert_model
            self._sbert_model = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _encode_texts(self, texts: List[str], cache: dict) -> np.ndarray:
        """
        Batch-encode texts with caching.

        Only encodes texts not already in cache. Returns array of
        embeddings in the same order as input texts.
        """
        self._load_sbert()

        # Find texts that need encoding
        texts_to_encode = []
        indices_to_encode = []
        for i, text in enumerate(texts):
            if text not in cache:
                texts_to_encode.append(text)
                indices_to_encode.append(i)

        # Batch encode new texts
        if texts_to_encode:
            embeddings = self._sbert_model.encode(
                texts_to_encode,
                batch_size=self.batch_size,
                show_progress_bar=len(texts_to_encode) > 100,
                convert_to_numpy=True
            )
            for j, idx in enumerate(indices_to_encode):
                cache[texts_to_encode[j]] = embeddings[j].astype(np.float32)

        # Build result array from cache
        result = np.array([cache[text] for text in texts], dtype=np.float32)
        return result

    def _parse_commit_column(self, commit_val) -> List[str]:
        """Parse a commit column value into a list of commit strings."""
        if pd.isna(commit_val):
            return []
        commit_str = str(commit_val).strip()
        if not commit_str:
            return []
        try:
            parsed = ast.literal_eval(commit_str)
            if isinstance(parsed, list):
                return [str(c) for c in parsed if c]
            return [commit_str]
        except (ValueError, SyntaxError):
            return [commit_str]

    def train(
        self,
        df: pd.DataFrame,
        build_col: str = 'Build_ID',
        test_col: str = 'TC_Key',
        result_col: str = 'TE_Test_Result',
        duration_col: Optional[str] = None,
        tc_text_col: Optional[str] = None,
        commit_text_col: Optional[str] = None
    ):
        """
        Train the FailRank-BB model.

        Args:
            df: Training DataFrame
            build_col: Column name for build ID
            test_col: Column name for test case ID
            result_col: Column name for test result ('Fail'/'Pass')
            duration_col: Optional column for test duration
            tc_text_col: Column with test case text for SBERT encoding.
                If None, uses test_col values as text.
            commit_text_col: Column with commit messages (may be stringified lists).
                If None, commit component is a zero vector.
        """
        from sklearn.linear_model import LogisticRegression

        # Reset state
        self.feature_extractor = FailRankBBFeatureExtractor(self.history_window)
        self._tc_embedding_cache = {}
        self._commit_embedding_cache = {}

        # Determine text column for TC embeddings
        text_col = tc_text_col if tc_text_col else test_col

        # 1. Encode unique TC texts with SBERT
        self._load_sbert()
        unique_tc_texts = df[text_col].fillna('').astype(str).unique().tolist()
        print(f"FailRank-BB: Encoding {len(unique_tc_texts)} unique TC texts...")
        self._encode_texts(unique_tc_texts, self._tc_embedding_cache)

        # 2. Encode unique commit texts (if available)
        commit_embeddings_per_build = {}
        if commit_text_col and commit_text_col in df.columns:
            # Collect all unique commit strings across all builds
            all_commits = set()
            builds = df[build_col].unique().tolist()
            build_commits = {}  # build_id -> list of commit strings
            grouped = df.groupby(build_col, sort=False)

            for build_id in builds:
                build_df = grouped.get_group(build_id)
                commits = set()
                for val in build_df[commit_text_col].dropna():
                    parsed = self._parse_commit_column(val)
                    commits.update(parsed)
                build_commits[build_id] = list(commits)
                all_commits.update(commits)

            if all_commits:
                all_commits_list = list(all_commits)
                print(f"FailRank-BB: Encoding {len(all_commits_list)} unique commits...")
                self._encode_texts(all_commits_list, self._commit_embedding_cache)

                # Compute mean commit vector per build
                for build_id, commits in build_commits.items():
                    if commits:
                        vecs = np.array([self._commit_embedding_cache[c] for c in commits],
                                        dtype=np.float32)
                        commit_embeddings_per_build[build_id] = vecs.mean(axis=0)
                    # else: will use zero vector
        else:
            print("FailRank-BB: No commit column provided, using zero vectors for commit component.")

        # 3. Build feature matrix and train classifier
        print("FailRank-BB: Building feature matrix...")
        emb_dim = self._embedding_dim
        zero_commit = np.zeros(emb_dim, dtype=np.float32)

        no_commit_data = not commit_embeddings_per_build
        no_history = not self.use_historical_features

        if no_commit_data and no_history:
            # Memory-efficient dedup path: when features depend only on TC text,
            # aggregate pass/fail counts per unique TC and use sample_weight.
            # Mathematically equivalent to the full expansion but uses O(n_unique)
            # memory instead of O(n_total).
            result_series = df[result_col].astype(str).str.strip()
            tc_series = df[text_col].fillna('').astype(str)

            counts = pd.crosstab(tc_series, result_series)
            unique_texts = counts.index.tolist()
            n_unique = len(unique_texts)

            fail_counts = (counts['Fail'].values.astype(np.float64)
                           if 'Fail' in counts.columns
                           else np.zeros(n_unique))
            pass_counts = (counts['Pass'].values.astype(np.float64)
                           if 'Pass' in counts.columns
                           else np.zeros(n_unique))

            # Build compact feature matrix (one row per unique TC)
            X_unique = np.empty((n_unique, emb_dim * 2), dtype=np.float32)
            X_unique[:, :emb_dim] = 0.0
            for i, text in enumerate(unique_texts):
                X_unique[i, emb_dim:] = self._tc_embedding_cache[text]

            # Two rows per TC (Fail + Pass) with sample weights
            X = np.vstack([X_unique, X_unique])
            y = np.array(['Fail'] * n_unique + ['Pass'] * n_unique)
            weights = np.concatenate([fail_counts, pass_counts])

            # Remove zero-weight entries
            nonzero = weights > 0
            X = X[nonzero]
            y = y[nonzero]
            weights = weights[nonzero]

            n_fail = int(fail_counts.sum())
            n_pass = int(pass_counts.sum())
            total = n_fail + n_pass
            print(f"FailRank-BB: Dedup {len(df)} rows -> {len(X)} weighted rows "
                  f"({n_unique} unique TCs)")
            print(f"FailRank-BB: Training on {total} effective samples "
                  f"(Fail={n_fail}, Pass={n_pass}, rate={n_fail/total:.4f})")
            print(f"FailRank-BB: Feature dim = {X.shape[1]} "
                  f"(commit={emb_dim} + tc={emb_dim})")

            self._classifier = LogisticRegression(
                max_iter=self.max_iter,
                random_state=self.seed
            )
            self._classifier.fit(X, y, sample_weight=weights)
            print(f"FailRank-BB: Classifier classes = {self._classifier.classes_.tolist()}")

        else:
            # General path: build full feature matrix
            features_list = []
            labels_list = []
            builds = df[build_col].unique().tolist()
            grouped = df.groupby(build_col, sort=False)

            for build_id in builds:
                build_df = grouped.get_group(build_id)
                test_ids = build_df[test_col].values
                tc_texts = build_df[text_col].fillna('').astype(str).values
                result_vals = build_df[result_col].astype(str).str.strip().values
                dur_vals = (build_df[duration_col].values
                            if duration_col and duration_col in build_df.columns
                            else np.ones(len(build_df)))

                commit_emb = commit_embeddings_per_build.get(build_id, zero_commit)

                for i in range(len(test_ids)):
                    tc_emb = self._tc_embedding_cache[tc_texts[i]]
                    feature_vec = np.concatenate([commit_emb, tc_emb])

                    if self.use_historical_features:
                        hist_features = self.feature_extractor.extract_features(test_ids[i])
                        feature_vec = np.concatenate([feature_vec, hist_features])

                    features_list.append(feature_vec)
                    result = result_vals[i]
                    labels_list.append('Fail' if result == 'Fail' else 'Pass')

                if self.use_historical_features:
                    test_results = {}
                    for i in range(len(test_ids)):
                        verdict = 1 if result_vals[i] == 'Fail' else 0
                        test_results[test_ids[i]] = (verdict, float(dur_vals[i]))
                    self.feature_extractor.update_history(str(build_id), test_results)

            X = np.array(features_list, dtype=np.float32)
            y = np.array(labels_list)

            mask = np.isin(y, ['Fail', 'Pass'])
            X = X[mask]
            y = y[mask]

            n_fail = (y == 'Fail').sum()
            n_pass = (y == 'Pass').sum()
            print(f"FailRank-BB: Training on {len(X)} samples "
                  f"(Fail={n_fail}, Pass={n_pass}, rate={n_fail/len(X):.4f})")
            print(f"FailRank-BB: Feature dim = {X.shape[1]} "
                  f"(commit={emb_dim} + tc={emb_dim}"
                  f"{' + hist=8' if self.use_historical_features else ''})")

            self._classifier = LogisticRegression(
                max_iter=self.max_iter,
                random_state=self.seed
            )
            self._classifier.fit(X, y)
            print(f"FailRank-BB: Classifier classes = {self._classifier.classes_.tolist()}")

        # Free SBERT model from GPU after training (keep caches for inference)
        self._unload_sbert()

    def prioritize(
        self,
        test_ids: List[str],
        tc_texts: Optional[List[str]] = None,
        build_commit_embedding: Optional[np.ndarray] = None
    ) -> List[str]:
        """
        Prioritize test cases by predicted P(Fail) descending.

        Args:
            test_ids: List of test IDs to prioritize.
            tc_texts: Optional list of TC texts (parallel to test_ids).
                If None, looks up from _tc_embedding_cache using test_ids.
            build_commit_embedding: Pre-computed mean commit embedding for
                the current build. If None, uses zero vector.

        Returns:
            Ordered list of test IDs (highest P(Fail) first).
        """
        if self._classifier is None:
            raise ValueError("Model not trained. Call train() first.")

        emb_dim = self._embedding_dim
        zero_commit = np.zeros(emb_dim, dtype=np.float32)
        commit_emb = build_commit_embedding if build_commit_embedding is not None else zero_commit

        # Build feature matrix
        features_list = []
        for i, test_id in enumerate(test_ids):
            # TC embedding
            if tc_texts is not None:
                text = tc_texts[i]
                if text in self._tc_embedding_cache:
                    tc_emb = self._tc_embedding_cache[text]
                else:
                    # Encode on-the-fly (rare case for unseen test texts)
                    self._load_sbert()
                    self._encode_texts([text], self._tc_embedding_cache)
                    tc_emb = self._tc_embedding_cache[text]
            else:
                if test_id in self._tc_embedding_cache:
                    tc_emb = self._tc_embedding_cache[test_id]
                else:
                    # Unseen test case — encode on-the-fly
                    self._load_sbert()
                    self._encode_texts([test_id], self._tc_embedding_cache)
                    tc_emb = self._tc_embedding_cache[test_id]

            # Concatenate: [commit | tc_steps]
            feature_vec = np.concatenate([commit_emb, tc_emb])

            # Optional historical features
            if self.use_historical_features:
                hist_features = self.feature_extractor.extract_features(test_id)
                feature_vec = np.concatenate([feature_vec, hist_features])

            features_list.append(feature_vec)

        X = np.array(features_list, dtype=np.float32)

        # predict_proba[:, 0] = P(Fail) since classes_ = ['Fail', 'Pass']
        proba_fail = self._classifier.predict_proba(X)[:, 0]

        # Sort by P(Fail) descending
        test_scores = list(zip(test_ids, proba_fail))
        test_scores.sort(key=lambda x: x[1], reverse=True)

        return [t[0] for t in test_scores]

    def update_history(
        self,
        build_id: str,
        test_results: Dict[str, Tuple[int, float]]
    ):
        """Update history after evaluation (no-op if use_historical_features=False)."""
        if self.use_historical_features:
            self.feature_extractor.update_history(build_id, test_results)

    def compute_commit_embedding(self, commit_texts: List[str]) -> np.ndarray:
        """
        Compute the mean commit embedding for a list of commit messages.

        Useful for pre-computing build commit embeddings before the eval loop.

        Args:
            commit_texts: List of commit message strings.

        Returns:
            Mean embedding vector (np.ndarray of shape (embedding_dim,)).
        """
        if not commit_texts:
            return np.zeros(self._embedding_dim, dtype=np.float32)

        self._load_sbert()
        self._encode_texts(commit_texts, self._commit_embedding_cache)
        vecs = np.array([self._commit_embedding_cache[c] for c in commit_texts],
                        dtype=np.float32)
        return vecs.mean(axis=0)


if __name__ == '__main__':
    print("FailRank-BB Baseline Implementation")
    print("Usage: from src.baselines.failrank_bb import FailRankBBModel")
