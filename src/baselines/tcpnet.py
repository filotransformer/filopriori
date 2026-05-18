"""
TCP-Net: Test Case Prioritization using End-to-End Deep Neural Networks

Faithful implementation following:
    Abdelkarim, A., Sabor, K. K., Bonnet, G., & Le Ber, F. (2022).
    TCP-Net: Test Case Prioritization using End-to-End Deep Neural Networks.
    ICSOFT 2022.

Key differences from DeepOrder:
    - 12 features (history + temporal multi-scale + metadata)
    - Mish activation instead of ReLU + BatchNorm
    - MSE loss (regression of priority value) instead of BCE
    - Linear output instead of Sigmoid
    - Grid search over architectures
    - Incremental training (retrain every 50 builds)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# MISH ACTIVATION
# =============================================================================

class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))"""
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))


# =============================================================================
# TCP-NET MODEL
# =============================================================================

class TCPNetModel(nn.Module):
    """
    TCP-Net DNN model for test case prioritization.

    Architecture:
    - Input: 12 features
    - Hidden layers with Mish activation and Dropout
    - Output: Linear (regression of priority score)
    """

    def __init__(self, input_dim: int, hidden_dims: List[int],
                 dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(Mish())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, 1))
        # Linear output (no activation) — regression

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

class TCPNetFeatureExtractor:
    """
    Extracts TCP-Net 12-dimensional features for each test case per build.

    Uses O(1) running statistics instead of recomputing from full history,
    enabling efficient processing of large projects (50K+ builds).

    Features:
        0: duration — last execution duration (normalized)
        1: last_result — last verdict (0/1)
        2: exec_status_last_2 — failure proportion in last 2 executions
        3: exec_status_last_3 — failure proportion in last 3 executions
        4: exec_status_last_5 — failure proportion in last 5 executions
        5: exec_status_last_10 — failure proportion in last 10 executions
        6: historical_failure_rate — total failure rate
        7: time_since_last_fail — builds since last failure (normalized)
        8: execution_count — total executions (normalized)
        9: status_changes — transitions pass<->fail (normalized)
       10: failure_streak — consecutive failures at current point
       11: test_age — builds since first appearance (normalized)
    """

    def __init__(self):
        # Running statistics — O(1) access per feature
        self._recent_verdicts = defaultdict(lambda: deque(maxlen=10))
        self._exec_count = defaultdict(int)
        self._failure_count = defaultdict(int)
        self._status_changes = defaultdict(int)
        self._failure_streak = defaultdict(int)
        self._last_verdict = {}
        self.test_first_build = {}
        self.test_last_fail_build = {}
        self.current_build_idx = 0
        self.scaler = MinMaxScaler()
        self.is_scaler_fitted = False

    def extract_features(self, test_id: str, duration: float = 0.0) -> np.ndarray:
        """Extract 12 features for a test case at the current build. O(1) complexity."""
        recent = self._recent_verdicts.get(test_id)
        n_exec = self._exec_count[test_id]

        # Feature 0: duration
        feat_duration = duration

        # Feature 1: last_result
        feat_last_result = float(recent[-1]) if recent else 0.0

        # Features 2-5: failure proportion in last N executions (from deque, max 10 items)
        if recent:
            recent_list = list(recent)
            n_recent = len(recent_list)
            def _fail_prop(n):
                subset = recent_list[-n:] if n_recent >= n else recent_list
                return sum(subset) / len(subset)
            feat_last_2 = _fail_prop(2)
            feat_last_3 = _fail_prop(3)
            feat_last_5 = _fail_prop(5)
            feat_last_10 = _fail_prop(10)
        else:
            feat_last_2 = feat_last_3 = feat_last_5 = feat_last_10 = 0.0

        # Feature 6: historical_failure_rate (running counters)
        feat_fail_rate = self._failure_count[test_id] / n_exec if n_exec > 0 else 0.0

        # Feature 7: time_since_last_fail
        if test_id in self.test_last_fail_build:
            builds_since_fail = self.current_build_idx - self.test_last_fail_build[test_id]
        else:
            builds_since_fail = self.current_build_idx

        # Feature 8: execution_count (running counter)
        feat_exec_count = float(n_exec)

        # Feature 9: status_changes (running counter)
        feat_changes = float(self._status_changes[test_id])

        # Feature 10: failure_streak (running counter)
        feat_streak = float(self._failure_streak[test_id])

        # Feature 11: test_age
        if test_id in self.test_first_build:
            feat_age = float(self.current_build_idx - self.test_first_build[test_id])
        else:
            feat_age = 0.0

        return np.array([
            feat_duration, feat_last_result,
            feat_last_2, feat_last_3, feat_last_5, feat_last_10,
            feat_fail_rate, float(builds_since_fail),
            feat_exec_count, feat_changes, feat_streak, feat_age
        ], dtype=np.float32)

    def update(self, test_id: str, verdict: int, duration: float = 0.0):
        """Update history after observing a test result. O(1) complexity."""
        # Recent verdicts (deque auto-discards oldest beyond maxlen=10)
        self._recent_verdicts[test_id].append(verdict)

        # Execution and failure counts
        self._exec_count[test_id] += 1
        if verdict == 1:
            self._failure_count[test_id] += 1

        # Status changes (increment when verdict differs from previous)
        if test_id in self._last_verdict:
            if self._last_verdict[test_id] != verdict:
                self._status_changes[test_id] += 1
        self._last_verdict[test_id] = verdict

        # Failure streak (reset on pass, increment on fail)
        if verdict == 1:
            self._failure_streak[test_id] += 1
        else:
            self._failure_streak[test_id] = 0

        # First build tracking
        if test_id not in self.test_first_build:
            self.test_first_build[test_id] = self.current_build_idx

        # Last fail build tracking
        if verdict == 1:
            self.test_last_fail_build[test_id] = self.current_build_idx

    def advance_build(self):
        """Move to the next build."""
        self.current_build_idx += 1

    def fit_scaler(self, features: np.ndarray):
        """Fit MinMaxScaler on training features."""
        self.scaler.fit(features)
        self.is_scaler_fitted = True

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply MinMaxScaler transformation."""
        if self.is_scaler_fitted:
            return self.scaler.transform(features)
        return features


# =============================================================================
# TCP-NET PRIORITIZER
# =============================================================================

class TCPNetPrioritizer:
    """
    High-level TCP-Net prioritizer with incremental training.

    Usage:
        prioritizer = TCPNetPrioritizer()

        # Training phase
        prioritizer.train(train_df, build_col, test_col, result_col, duration_col)

        # Evaluation phase (with incremental retraining every 50 builds)
        for build in test_builds:
            ranking = prioritizer.prioritize(test_ids, durations)
            prioritizer.update_and_maybe_retrain(test_ids, verdicts, durations)
    """

    def __init__(
        self,
        architectures: List[List[int]] = None,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 30,
        batch_size: int = 64,
        early_stopping_patience: int = 10,
        retrain_interval: int = 50,
        max_retrain_samples: int = 500_000,
        device: str = 'cpu',
        seed: int = 42
    ):
        if architectures is None:
            architectures = [[64, 32], [128, 64, 32], [256, 128, 64]]

        self.architectures = architectures
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.retrain_interval = retrain_interval
        self.max_retrain_samples = max_retrain_samples
        self.device = device
        self.seed = seed

        self.feature_extractor = TCPNetFeatureExtractor()
        self.model = None
        self.best_architecture = None

        # Store training data for incremental retraining
        self.train_features = []
        self.train_labels = []
        self.eval_count = 0

        torch.manual_seed(seed)
        np.random.seed(seed)

    def train(
        self,
        df: pd.DataFrame,
        build_col: str,
        test_col: str,
        result_col: str,
        duration_col: Optional[str] = None
    ):
        """
        Train TCP-Net on training data with grid search over architectures.

        Uses groupby for O(n) total processing instead of O(n_builds * n_rows).
        """
        has_duration = duration_col is not None and duration_col in df.columns

        all_features = []
        all_labels = []

        n_builds_total = df[build_col].nunique()

        # groupby with sort=False preserves first-occurrence order (temporal order)
        for build_count, (build_id, build_df) in enumerate(df.groupby(build_col, sort=False)):
            test_ids_arr = build_df[test_col].values
            results_arr = build_df[result_col].values
            durations_arr = build_df[duration_col].values if has_duration else None

            for i in range(len(test_ids_arr)):
                test_id = str(test_ids_arr[i])
                verdict = 1 if str(results_arr[i]).strip() == 'Fail' else 0
                duration = float(durations_arr[i]) if durations_arr is not None else 0.0

                features = self.feature_extractor.extract_features(test_id, duration)
                all_features.append(features)
                all_labels.append(float(verdict))

                self.feature_extractor.update(test_id, verdict, duration)

            self.feature_extractor.advance_build()

            if (build_count + 1) % 5000 == 0:
                logger.info(f"    Training feature extraction: {build_count + 1}/{n_builds_total} builds")

        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.float32)

        logger.info(f"    Extracted {len(X):,} feature vectors from {n_builds_total:,} builds")

        # Fit scaler on training data
        self.feature_extractor.fit_scaler(X)
        X = self.feature_extractor.transform(X)

        # Store for incremental retraining (cap to max_retrain_samples)
        if len(X) > self.max_retrain_samples:
            self.train_features = list(X[-self.max_retrain_samples:])
            self.train_labels = list(y[-self.max_retrain_samples:])
            logger.info(f"    Stored last {self.max_retrain_samples:,}/{len(X):,} samples for retraining")
        else:
            self.train_features = list(X)
            self.train_labels = list(y)

        # Grid search over architectures
        self._grid_search_and_train(X, y)

    def _grid_search_and_train(self, X: np.ndarray, y: np.ndarray):
        """Grid search over architectures, then train the best one."""
        n_features = X.shape[1]

        # Simple validation split (last 20% of data)
        val_split = int(len(X) * 0.8)
        X_train, X_val = X[:val_split], X[val_split:]
        y_train, y_val = y[:val_split], y[val_split:]

        best_val_loss = float('inf')
        best_arch = self.architectures[0]

        for arch in self.architectures:
            model = TCPNetModel(n_features, arch, self.dropout).to(self.device)
            val_loss = self._train_model(model, X_train, y_train, X_val, y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_arch = arch

        # Train final model on all data with best architecture
        self.best_architecture = best_arch
        self.model = TCPNetModel(n_features, best_arch, self.dropout).to(self.device)
        self._train_model(self.model, X, y)

    def _train_model(
        self,
        model: TCPNetModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> float:
        """
        Train a model with MSE loss and optional early stopping.

        Returns validation loss (or training loss if no validation set).
        """
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        best_val_loss = float('inf')
        patience_counter = 0

        model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            n_batches = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            # Early stopping on validation set
            if X_val is not None and len(X_val) > 0:
                model.eval()
                with torch.no_grad():
                    val_pred = model(torch.FloatTensor(X_val).to(self.device))
                    val_loss = criterion(val_pred, torch.FloatTensor(y_val).to(self.device)).item()
                model.train()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        break
            else:
                best_val_loss = total_loss / max(n_batches, 1)

        return best_val_loss

    def prioritize(
        self,
        test_ids: List[str],
        durations: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """
        Prioritize test cases using the trained model.

        Args:
            test_ids: List of test IDs to prioritize
            durations: test_id -> execution duration

        Returns:
            Ordered list of test IDs (highest priority first)
        """
        if self.model is None:
            return test_ids

        if durations is None:
            durations = {t: 0.0 for t in test_ids}

        # Extract features
        features = np.array([
            self.feature_extractor.extract_features(tid, durations.get(tid, 0.0))
            for tid in test_ids
        ], dtype=np.float32)

        # Apply scaler
        features = self.feature_extractor.transform(features)

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            scores = self.model(torch.FloatTensor(features).to(self.device)).cpu().numpy()

        # Rank by predicted failure probability (descending)
        ranked_indices = np.argsort(-scores)
        return [test_ids[i] for i in ranked_indices]

    def update_and_maybe_retrain(
        self,
        test_ids: List[str],
        verdicts: Dict[str, int],
        durations: Optional[Dict[str, float]] = None
    ):
        """
        Update feature history and retrain incrementally every retrain_interval builds.

        Args:
            test_ids: List of test IDs
            verdicts: test_id -> 0 (pass) or 1 (fail)
            durations: test_id -> duration
        """
        if durations is None:
            durations = {t: 0.0 for t in test_ids}

        # Batch feature extraction (O(1) per test thanks to running stats)
        features_list = []
        labels_list = []
        for tid in test_ids:
            verdict = verdicts.get(tid, 0)
            dur = durations.get(tid, 0.0)

            features = self.feature_extractor.extract_features(tid, dur)
            features_list.append(features)
            labels_list.append(float(verdict))

            self.feature_extractor.update(tid, verdict, dur)

        self.feature_extractor.advance_build()
        self.eval_count += 1

        # Batch scale and extend (single scaler call instead of N)
        if features_list:
            features_batch = np.array(features_list, dtype=np.float32)
            scaled_batch = self.feature_extractor.transform(features_batch)
            self.train_features.extend(scaled_batch)
            self.train_labels.extend(labels_list)

        # Periodically trim stored data to prevent unbounded memory growth
        if len(self.train_features) > self.max_retrain_samples * 2:
            self.train_features = self.train_features[-self.max_retrain_samples:]
            self.train_labels = self.train_labels[-self.max_retrain_samples:]

        # Incremental retrain every retrain_interval builds
        if self.eval_count % self.retrain_interval == 0:
            n = min(len(self.train_features), self.max_retrain_samples)
            X = np.array(self.train_features[-n:], dtype=np.float32)
            y = np.array(self.train_labels[-n:], dtype=np.float32)

            # Retrain with best architecture found during grid search
            if self.best_architecture is not None:
                self.model = TCPNetModel(
                    X.shape[1], self.best_architecture, self.dropout
                ).to(self.device)
                self._train_model(self.model, X, y)
                logger.info(f"    Retrained at build {self.eval_count} on {n:,} samples")
