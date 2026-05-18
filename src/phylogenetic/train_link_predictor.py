"""
Training script for self-supervised link prediction.

This script trains a link predictor on the k-NN graph structure,
learning to predict edge existence based on node features.
The trained model is then used for graph rewiring.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import logging

from .link_prediction import (
    LinkPredictor,
    sample_negative_edges,
    compute_link_prediction_loss
)

logger = logging.getLogger(__name__)


class LinkPredictionTrainer:
    """
    Trainer for link prediction models.

    Handles training loop, validation, and model checkpointing.
    """

    def __init__(
        self,
        model: LinkPredictor,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        neg_sampling_ratio: float = 1.0,
        loss_type: str = 'bce'
    ):
        self.model = model.to(device)
        self.device = device
        self.neg_sampling_ratio = neg_sampling_ratio
        self.loss_type = loss_type

        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.best_val_auc = 0.0
        self.best_model_state = None

    def train_epoch(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        train_pos_edges: torch.Tensor
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Full edge index for message passing [2, num_edges]
            train_pos_edges: Training positive edges [2, num_train_pos]

        Returns:
            Dictionary with training metrics
        """
        self.model.train()

        num_nodes = x.size(0)
        num_pos = train_pos_edges.size(1)
        num_neg = int(num_pos * self.neg_sampling_ratio)

        # Sample negative edges
        neg_edge_index = sample_negative_edges(
            edge_index=train_pos_edges,
            num_nodes=num_nodes,
            num_neg_samples=num_neg,
            method='uniform'
        )

        # Move to device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        train_pos_edges = train_pos_edges.to(self.device)
        neg_edge_index = neg_edge_index.to(self.device)

        # Forward pass
        pos_scores = self.model(x, edge_index, train_pos_edges)
        neg_scores = self.model(x, edge_index, neg_edge_index)

        # Compute loss
        loss = compute_link_prediction_loss(pos_scores, neg_scores, self.loss_type)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute metrics
        with torch.no_grad():
            pos_pred = torch.sigmoid(pos_scores) > 0.5
            neg_pred = torch.sigmoid(neg_scores) <= 0.5
            acc = (pos_pred.sum() + neg_pred.sum()).float() / (num_pos + num_neg)

        return {
            'loss': loss.item(),
            'accuracy': acc.item()
        }

    @torch.no_grad()
    def validate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        val_pos_edges: torch.Tensor,
        val_neg_edges: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            x: Node features
            edge_index: Full edge index for message passing
            val_pos_edges: Validation positive edges
            val_neg_edges: Validation negative edges (optional, will be sampled if not provided)

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()

        num_nodes = x.size(0)
        num_pos = val_pos_edges.size(1)

        # Sample negative edges if not provided
        if val_neg_edges is None:
            val_neg_edges = sample_negative_edges(
                edge_index=val_pos_edges,
                num_nodes=num_nodes,
                num_neg_samples=num_pos,
                method='uniform'
            )

        # Move to device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        val_pos_edges = val_pos_edges.to(self.device)
        val_neg_edges = val_neg_edges.to(self.device)

        # Forward pass
        pos_scores = self.model(x, edge_index, val_pos_edges)
        neg_scores = self.model(x, edge_index, val_neg_edges)

        # Compute loss
        loss = compute_link_prediction_loss(pos_scores, neg_scores, self.loss_type)

        # Compute metrics
        pos_pred = torch.sigmoid(pos_scores) > 0.5
        neg_pred = torch.sigmoid(neg_scores) <= 0.5
        acc = (pos_pred.sum() + neg_pred.sum()).float() / (num_pos + val_neg_edges.size(1))

        # Compute AUC
        all_scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
        all_labels = np.concatenate([
            np.ones(num_pos),
            np.zeros(val_neg_edges.size(1))
        ])

        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(all_labels, all_scores)
        except ValueError:
            auc = 0.5  # If only one class present

        return {
            'loss': loss.item(),
            'accuracy': acc.item(),
            'auc': auc
        }

    def fit(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        train_pos_edges: torch.Tensor,
        val_pos_edges: torch.Tensor,
        num_epochs: int = 50,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train the link prediction model.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Full edge index [2, num_edges]
            train_pos_edges: Training positive edges [2, num_train_pos]
            val_pos_edges: Validation positive edges [2, num_val_pos]
            num_epochs: Number of training epochs
            patience: Early stopping patience
            verbose: Whether to print progress

        Returns:
            Dictionary with training history
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': []
        }

        best_val_auc = 0.0
        patience_counter = 0

        iterator = tqdm(range(num_epochs), desc="Training Link Predictor") if verbose else range(num_epochs)

        for epoch in iterator:
            # Train
            train_metrics = self.train_epoch(x, edge_index, train_pos_edges)

            # Validate
            val_metrics = self.validate(x, edge_index, val_pos_edges)

            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_auc'].append(val_metrics['auc'])

            # Logging
            if verbose:
                iterator.set_postfix({
                    'train_loss': f"{train_metrics['loss']:.4f}",
                    'val_auc': f"{val_metrics['auc']:.4f}"
                })

            # Early stopping
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.best_val_auc = best_val_auc

        return history


def split_edges(
    edge_index: torch.Tensor,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split edges into train/val/test sets.

    Args:
        edge_index: Edge indices [2, num_edges]
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        Tuple of (train_edges, val_edges, test_edges)
    """
    num_edges = edge_index.size(1)
    num_val = int(num_edges * val_ratio)
    num_test = int(num_edges * test_ratio)
    num_train = num_edges - num_val - num_test

    # Shuffle edges
    perm = torch.randperm(num_edges)
    edge_index = edge_index[:, perm]

    # Split
    train_edges = edge_index[:, :num_train]
    val_edges = edge_index[:, num_train:num_train+num_val]
    test_edges = edge_index[:, num_train+num_val:]

    return train_edges, val_edges, test_edges


def train_link_predictor_from_graph(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    config: Dict,
    device: torch.device,
    save_path: Optional[str] = None
) -> LinkPredictor:
    """
    Convenience function to train link predictor from graph data.

    Args:
        x: Node features [num_nodes, input_dim]
        edge_index: Edge indices [2, num_edges]
        config: Configuration dictionary with model and training params
        device: Device to train on
        save_path: Optional path to save trained model

    Returns:
        Trained LinkPredictor model
    """
    # Split edges
    train_edges, val_edges, test_edges = split_edges(
        edge_index,
        val_ratio=config.get('val_ratio', 0.1),
        test_ratio=config.get('test_ratio', 0.1)
    )

    # Create model
    model = LinkPredictor(
        input_dim=x.size(1),
        hidden_dim=config.get('hidden_dim', 256),
        embedding_dim=config.get('embedding_dim', 128),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.1),
        encoder_type=config.get('encoder_type', 'gcn'),
        decoder_type=config.get('decoder_type', 'mlp')
    )

    # Create trainer
    trainer = LinkPredictionTrainer(
        model=model,
        device=device,
        learning_rate=config.get('learning_rate', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5),
        neg_sampling_ratio=config.get('neg_sampling_ratio', 1.0),
        loss_type=config.get('loss_type', 'bce')
    )

    # Train
    history = trainer.fit(
        x=x,
        edge_index=train_edges,  # Use only train edges for message passing
        train_pos_edges=train_edges,
        val_pos_edges=val_edges,
        num_epochs=config.get('num_epochs', 50),
        patience=config.get('patience', 10),
        verbose=config.get('verbose', True)
    )

    # Save model if path provided
    if save_path is not None:
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'history': history,
            'best_val_auc': trainer.best_val_auc
        }, save_path)

        logger.info(f"Saved trained link predictor to {save_path}")

    return model
