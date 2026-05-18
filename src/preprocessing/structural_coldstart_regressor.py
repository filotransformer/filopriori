"""
Cold-Start Regressor for Structural Features

This module provides an MLP-based regressor that learns to predict structural features
from semantic embeddings. This is used to handle the "cold-start" problem where new
test cases (TCs) have no execution history and thus no structural features.

Approach:
1. Train an MLP on TCs with known structural features (from training data)
2. For TCs with execution history: use real structural features
3. For TCs without history (cold-start): use predicted features from the MLP

This is superior to using global mean because:
- It leverages semantic similarity: TCs with similar semantics likely have similar behavior
- The MLP learns the relationship between test case semantics and execution patterns
- Predictions are personalized per TC, not a single global default

Author: Filo-Priori Team
Date: 2025-12-01
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Dict, Tuple, Optional
import logging
import os
import pickle

logger = logging.getLogger(__name__)


class NormalizedNumpyDataset(Dataset):
    """Memory-efficient dataset that normalizes on-the-fly without full copies."""

    def __init__(self, X: np.ndarray, y: np.ndarray,
                 x_mean: np.ndarray, x_std: np.ndarray,
                 y_mean: np.ndarray, y_std: np.ndarray):
        self.X = X
        self.y = y
        self.x_mean = x_mean.astype(np.float32)
        self.x_std = x_std.astype(np.float32)
        self.y_mean = y_mean.astype(np.float32)
        self.y_std = y_std.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = (self.X[idx].astype(np.float32) - self.x_mean) / self.x_std
        y = (self.y[idx].astype(np.float32) - self.y_mean) / self.y_std
        return torch.from_numpy(x), torch.from_numpy(y)


class StructuralFeaturePredictor(nn.Module):
    """
    MLP that predicts structural features from semantic embeddings.

    Architecture:
        Input: Semantic embedding (e.g., 1536-dim from SBERT TC+Commit)
        Hidden: 2-3 layers with ReLU and dropout
        Output: Structural features (e.g., 6-dim or 10-dim depending on extractor version)
    """

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        output_dim: int = 6,
        dropout: float = 0.2
    ):
        """
        Initialize the structural feature predictor.

        Args:
            input_dim: Dimension of semantic embeddings
            hidden_dims: Tuple of hidden layer dimensions
            output_dim: Dimension of structural features to predict
            dropout: Dropout probability
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        # Output layer (no activation - regression)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Semantic embeddings [batch_size, input_dim]

        Returns:
            Predicted structural features [batch_size, output_dim]
        """
        return self.mlp(x)


class ColdStartRegressor:
    """
    Cold-Start Regressor that trains and uses an MLP to predict structural features
    for test cases without execution history.

    Usage:
        1. Fit on training data (embeddings + structural features)
        2. Predict for new TCs without history
        3. Blend predictions with real features for TCs with partial history
    """

    def __init__(
        self,
        embedding_dim: int = 1536,
        structural_dim: int = 6,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        num_epochs: int = 100,
        patience: int = 10,
        device: str = 'cuda',
        verbose: bool = True
    ):
        """
        Initialize the Cold-Start Regressor.

        Args:
            embedding_dim: Dimension of semantic embeddings
            structural_dim: Dimension of structural features
            hidden_dims: Hidden layer dimensions for MLP
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            batch_size: Training batch size
            num_epochs: Maximum training epochs
            patience: Early stopping patience
            device: Device to train on ('cuda' or 'cpu')
            verbose: Enable verbose logging
        """
        self.embedding_dim = embedding_dim
        self.structural_dim = structural_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose

        # Initialize model
        self.model = StructuralFeaturePredictor(
            input_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=structural_dim,
            dropout=dropout
        ).to(self.device)

        # Training state
        self.is_fitted = False
        self.train_mean = None
        self.train_std = None
        self.target_mean = None
        self.target_std = None

        logger.info(f"Initialized ColdStartRegressor:")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Structural dim: {structural_dim}")
        logger.info(f"  Hidden dims: {hidden_dims}")
        logger.info(f"  Device: {self.device}")

    def fit(
        self,
        embeddings: np.ndarray,
        structural_features: np.ndarray,
        val_embeddings: Optional[np.ndarray] = None,
        val_structural: Optional[np.ndarray] = None
    ) -> 'ColdStartRegressor':
        """
        Train the MLP to predict structural features from embeddings.

        Args:
            embeddings: Training semantic embeddings [N, embedding_dim]
            structural_features: Training structural features [N, structural_dim]
            val_embeddings: Optional validation embeddings
            val_structural: Optional validation structural features

        Returns:
            self (for method chaining)
        """
        logger.info("="*70)
        logger.info("TRAINING COLD-START REGRESSOR")
        logger.info("="*70)

        # Validate inputs
        assert embeddings.shape[0] == structural_features.shape[0], \
            f"Mismatch: embeddings {embeddings.shape[0]} vs structural {structural_features.shape[0]}"
        assert embeddings.shape[1] == self.embedding_dim, \
            f"Mismatch: embeddings dim {embeddings.shape[1]} vs expected {self.embedding_dim}"

        # Update structural_dim if different (auto-detect)
        if structural_features.shape[1] != self.structural_dim:
            logger.warning(f"Updating structural_dim from {self.structural_dim} to {structural_features.shape[1]}")
            self.structural_dim = structural_features.shape[1]
            # Rebuild model with correct output dimension
            self.model = StructuralFeaturePredictor(
                input_dim=self.embedding_dim,
                hidden_dims=self.hidden_dims,
                output_dim=self.structural_dim,
                dropout=self.dropout
            ).to(self.device)

        # Compute normalization stats in chunks to avoid memory spikes
        chunk_size = 50000
        n = len(embeddings)
        d = embeddings.shape[1]

        emb_sum = np.zeros(d, dtype=np.float64)
        emb_sq_sum = np.zeros(d, dtype=np.float64)
        for start in range(0, n, chunk_size):
            chunk = embeddings[start:start + chunk_size].astype(np.float64)
            emb_sum += chunk.sum(axis=0)
            emb_sq_sum += (chunk ** 2).sum(axis=0)

        self.train_mean = (emb_sum / n).astype(np.float32)
        variance = emb_sq_sum / n - self.train_mean.astype(np.float64) ** 2
        np.clip(variance, 0, None, out=variance)
        self.train_std = (np.sqrt(variance) + 1e-8).astype(np.float32)

        self.target_mean = structural_features.mean(axis=0).astype(np.float32)
        self.target_std = (structural_features.std(axis=0) + 1e-8).astype(np.float32)

        # Memory-efficient: normalize on-the-fly per batch, no full copies
        train_dataset = NormalizedNumpyDataset(
            embeddings, structural_features,
            self.train_mean, self.train_std,
            self.target_mean, self.target_std
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True
        )

        # Validation data
        val_loader = None
        if val_embeddings is not None and val_structural is not None:
            val_dataset = NormalizedNumpyDataset(
                val_embeddings, val_structural,
                self.train_mean, self.train_std,
                self.target_mean, self.target_std
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True
            )

        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Training loop
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        logger.info(f"\nTraining on {len(train_dataset)} samples...")

        for epoch in range(self.num_epochs):
            # Train
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            val_loss = train_loss  # Default if no validation set
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        predictions = self.model(X_batch)
                        val_loss += criterion(predictions, y_batch).item()
                val_loss /= len(val_loader)

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Logging
            if self.verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}: "
                           f"Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True

        logger.info(f"\n✅ Cold-Start Regressor trained!")
        logger.info(f"   Best validation loss: {best_val_loss:.6f}")

        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict structural features from semantic embeddings.

        Args:
            embeddings: Semantic embeddings [N, embedding_dim]

        Returns:
            Predicted structural features [N, structural_dim]
        """
        if not self.is_fitted:
            raise RuntimeError("ColdStartRegressor not fitted. Call fit() first.")

        # Normalize
        X = (embeddings - self.train_mean) / self.train_std

        # Predict
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()

        # Denormalize
        predictions = predictions * self.target_std + self.target_mean

        return predictions

    def impute_features(
        self,
        embeddings: np.ndarray,
        real_features: np.ndarray,
        needs_imputation: np.ndarray,
        blend_alpha: float = 0.0
    ) -> np.ndarray:
        """
        Impute structural features for TCs that need it.

        Args:
            embeddings: All semantic embeddings [N, embedding_dim]
            real_features: Real structural features [N, structural_dim]
                           (may contain zeros/defaults for TCs without history)
            needs_imputation: Boolean mask [N] indicating which TCs need imputation
            blend_alpha: If > 0, blend predicted with real features (0 = full prediction)

        Returns:
            Imputed structural features [N, structural_dim]
        """
        if not self.is_fitted:
            raise RuntimeError("ColdStartRegressor not fitted. Call fit() first.")

        result = real_features.copy()

        n_impute = needs_imputation.sum()
        if n_impute == 0:
            logger.info("  No samples need imputation")
            return result

        # Predict for samples that need imputation
        embeddings_to_impute = embeddings[needs_imputation]
        predictions = self.predict(embeddings_to_impute)

        # Apply predictions (with optional blending)
        if blend_alpha > 0:
            result[needs_imputation] = (
                blend_alpha * real_features[needs_imputation] +
                (1 - blend_alpha) * predictions
            )
        else:
            result[needs_imputation] = predictions

        logger.info(f"  Imputed {n_impute} samples using Cold-Start Regressor")

        return result

    def save(self, filepath: str) -> None:
        """Save the regressor to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        state = {
            'model_state': self.model.state_dict(),
            'embedding_dim': self.embedding_dim,
            'structural_dim': self.structural_dim,
            'hidden_dims': self.hidden_dims,
            'dropout': self.dropout,
            'train_mean': self.train_mean,
            'train_std': self.train_std,
            'target_mean': self.target_mean,
            'target_std': self.target_std,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        logger.info(f"Cold-Start Regressor saved to {filepath}")

    def load(self, filepath: str) -> 'ColdStartRegressor':
        """Load the regressor from disk."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        self.embedding_dim = state['embedding_dim']
        self.structural_dim = state['structural_dim']
        self.hidden_dims = state['hidden_dims']
        self.dropout = state['dropout']
        self.train_mean = state['train_mean']
        self.train_std = state['train_std']
        self.target_mean = state['target_mean']
        self.target_std = state['target_std']
        self.is_fitted = state['is_fitted']

        # Rebuild and load model
        self.model = StructuralFeaturePredictor(
            input_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.structural_dim,
            dropout=self.dropout
        ).to(self.device)
        self.model.load_state_dict(state['model_state'])

        logger.info(f"Cold-Start Regressor loaded from {filepath}")

        return self


def create_coldstart_regressor(config: Dict, device: str = 'cuda') -> ColdStartRegressor:
    """
    Factory function to create ColdStartRegressor from config.

    Args:
        config: Configuration dictionary with optional 'coldstart' section
        device: Device to use

    Returns:
        Configured ColdStartRegressor instance
    """
    coldstart_config = config.get('coldstart', {})

    # Get embedding and structural dimensions from config
    embedding_dim = config.get('embedding', config.get('semantic', {})).get('combined_embedding_dim', 1536)
    structural_dim = config.get('structural', {}).get('input_dim', 6)

    return ColdStartRegressor(
        embedding_dim=embedding_dim,
        structural_dim=structural_dim,
        hidden_dims=tuple(coldstart_config.get('hidden_dims', [512, 256, 128])),
        dropout=coldstart_config.get('dropout', 0.2),
        learning_rate=coldstart_config.get('learning_rate', 1e-3),
        weight_decay=coldstart_config.get('weight_decay', 1e-4),
        batch_size=coldstart_config.get('batch_size', 64),
        num_epochs=coldstart_config.get('num_epochs', 100),
        patience=coldstart_config.get('patience', 10),
        device=device,
        verbose=True
    )


# For backwards compatibility
__all__ = [
    'StructuralFeaturePredictor',
    'ColdStartRegressor',
    'create_coldstart_regressor'
]
