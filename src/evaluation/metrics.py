"""
Evaluation Metrics Module
Computes various metrics for model evaluation
"""

import numpy as np
import warnings
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
    precision_recall_curve,
    auc
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def compute_auprc(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_classes: int
) -> Dict[str, float]:
    """
    Compute Area Under Precision-Recall Curve (AUPRC) metrics

    Args:
        probabilities: Predicted probabilities [n_samples, num_classes]
        labels: Ground truth labels [n_samples]
        num_classes: Number of classes

    Returns:
        Dictionary with AUPRC metrics
    """
    # Special handling for binary classification
    if num_classes == 2:
        # For binary classification, label_binarize returns (n_samples, 1)
        # We need to create (n_samples, 2) manually
        labels_binarized = np.zeros((len(labels), 2))
        labels_binarized[:, 1] = (labels == 1).astype(int)  # Positive class
        labels_binarized[:, 0] = (labels == 0).astype(int)  # Negative class
    else:
        # Binarize labels for multi-class
        labels_binarized = label_binarize(labels, classes=range(num_classes))

        # Handle edge case where not all classes are present
        if labels_binarized.shape[1] != num_classes:
            # Pad with zeros for missing classes
            labels_bin_padded = np.zeros((len(labels), num_classes))
            present_classes = np.unique(labels)
            for i, cls in enumerate(present_classes):
                if i < labels_binarized.shape[1]:
                    labels_bin_padded[:, cls] = labels_binarized[:, i]
            labels_binarized = labels_bin_padded

    # Compute per-class AUPRC
    # Suppress warnings for classes with no positive samples (expected for rare classes)
    auprc_per_class = []
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='No positive class found in y_true')
        for i in range(num_classes):
            try:
                # Use average_precision_score which computes AUPRC
                ap = average_precision_score(labels_binarized[:, i], probabilities[:, i])
                auprc_per_class.append(ap)
            except:
                # If class not present or error, set to 0
                auprc_per_class.append(0.0)

    auprc_per_class = np.array(auprc_per_class)

    # Compute macro average (equal weight for all classes)
    auprc_macro = np.mean(auprc_per_class)

    # Compute weighted average (weighted by class support)
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = class_counts / len(labels)
    auprc_weighted = np.sum(auprc_per_class * class_weights)

    return {
        'auprc_macro': auprc_macro,
        'auprc_weighted': auprc_weighted,
        'auprc_per_class': auprc_per_class
    }


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    label_names: Optional[List[str]] = None,
    probabilities: Optional[np.ndarray] = None,
    lightweight: bool = False
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics

    Args:
        predictions: Predicted class labels
        labels: Ground truth labels
        num_classes: Number of classes
        label_names: Optional list of label names
        probabilities: Optional predicted probabilities [n_samples, num_classes]
        lightweight: If True, only compute f1_macro and accuracy (fast, for training epochs)

    Returns:
        Dictionary of metrics
    """
    # Accuracy
    accuracy = accuracy_score(labels, predictions)

    # F1 scores
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)

    if lightweight:
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_macro,  # placeholder
            'auprc_macro': 0.0,
        }

    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    f1_per_class = f1_score(labels, predictions, average=None, zero_division=0)

    # Precision and Recall
    precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
    precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)

    recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
    recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)

    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'confusion_matrix': cm,
        'f1_per_class': f1_per_class
    }

    # Compute AUPRC if probabilities are provided
    if probabilities is not None:
        auprc_metrics = compute_auprc(probabilities, labels, num_classes)
        metrics.update(auprc_metrics)
        logger.info(f"\nAUPRC (Macro): {auprc_metrics['auprc_macro']:.4f}")
        logger.info(f"AUPRC (Weighted): {auprc_metrics['auprc_weighted']:.4f}")

    # Classification report
    if label_names is not None:
        # Only use label names for classes that are actually present
        unique_classes = np.unique(np.concatenate([labels, predictions]))
        present_label_names = [label_names[i] for i in unique_classes if i < len(label_names)]

        report = classification_report(
            labels,
            predictions,
            labels=unique_classes,
            target_names=present_label_names,
            zero_division=0
        )
        logger.info(f"\nClassification Report:\n{report}")

    return metrics


def _dcg_at_k(relevances: np.ndarray, k: int) -> float:
    k = min(k, len(relevances))
    if k <= 0:
        return 0.0
    gains = relevances[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    return float(np.sum(gains * discounts))


def _ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    dcg = _dcg_at_k(relevances, k)
    ideal = _dcg_at_k(np.sort(relevances)[::-1], k)
    if ideal == 0.0:
        return 0.0
    return dcg / ideal


def compute_ranking_metrics_by_build(
    probabilities: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    ks: List[int] = (5, 10),
    percents: List[float] = (0.1,)
) -> Dict[str, float]:
    """
    Compute ranking metrics (NDCG@K, Recall@K%) per build and average over builds
    with at least one Fail.

    For binary classification, relevance = 1 if label == Fail (0), else 0.
    Ranking is by P(Fail) descending.
    """
    # Use probability of Fail (class 0) for ranking
    p_fail = probabilities[:, 0]
    metrics_sum = {f"ndcg@{k}": 0.0 for k in ks}
    for p in percents:
        metrics_sum[f"recall@{int(p*100)}pct"] = 0.0

    counts = 0
    for g in np.unique(groups):
        mask = groups == g
        if not np.any(mask):
            continue
        y = labels[mask]
        if y.ndim > 1:
            y = y.squeeze()
        # relevance: Fail==0 → 1, Pass==1 → 0
        rel = (y == 0).astype(int)
        total_fail = int(rel.sum())
        if total_fail == 0:
            continue  # skip builds with no fails
        scores = p_fail[mask]
        order = np.argsort(-scores)
        rel_sorted = rel[order]

        # NDCG@K
        for k in ks:
            metrics_sum[f"ndcg@{k}"] += _ndcg_at_k(rel_sorted, k)

        # Recall@K% (K = ceil(percent * n))
        n = len(rel_sorted)
        for p in percents:
            K = max(1, int(np.ceil(p * n)))
            top_rel = rel_sorted[:K]
            rec = float(top_rel.sum()) / float(total_fail)
            metrics_sum[f"recall@{int(p*100)}pct"] += rec

        counts += 1

    if counts == 0:
        return {k: 0.0 for k in metrics_sum}
    return {k: (v / counts) for k, v in metrics_sum.items()}


def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    normalize: bool = False
):
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix
        label_names: List of class names
        save_path: Path to save figure
        normalize: Whether to normalize the confusion matrix
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=label_names if label_names else range(len(cm)),
        yticklabels=label_names if label_names else range(len(cm)),
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")

    plt.close()


def analyze_per_class_performance(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_names: List[str]
) -> Dict[str, Dict]:
    """
    Analyze performance for each class

    Args:
        predictions: Predicted labels
        labels: True labels
        label_names: List of class names

    Returns:
        Dictionary with per-class metrics
    """
    per_class_metrics = {}

    for i, class_name in enumerate(label_names):
        # Get samples for this class
        class_mask = labels == i
        class_predictions = predictions[class_mask]
        class_labels = labels[class_mask]

        if len(class_labels) > 0:
            class_accuracy = accuracy_score(class_labels, class_predictions)
            class_f1 = f1_score(class_labels, class_predictions, average='binary', pos_label=i, zero_division=0)

            per_class_metrics[class_name] = {
                'accuracy': class_accuracy,
                'f1': class_f1,
                'support': len(class_labels)
            }

    return per_class_metrics


def print_metrics_summary(metrics: Dict, label_names: Optional[List[str]] = None):
    """
    Print a formatted summary of metrics

    Args:
        metrics: Metrics dictionary
        label_names: Optional list of label names
    """
    print("\n" + "="*80)
    print("EVALUATION METRICS SUMMARY")
    print("="*80)

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")

    # AUPRC metrics (primary for imbalanced datasets)
    if 'auprc_macro' in metrics:
        print(f"\n  AUPRC (Macro):      {metrics['auprc_macro']:.4f}  *** PRIMARY METRIC ***")
        print(f"  AUPRC (Weighted):   {metrics['auprc_weighted']:.4f}")

    print(f"\n  F1 Score (Macro):   {metrics['f1_macro']:.4f}")
    print(f"  F1 Score (Weighted):{metrics['f1_weighted']:.4f}")
    print(f"  Precision (Macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro):     {metrics['recall_macro']:.4f}")

    # Per-class metrics
    if 'f1_per_class' in metrics and label_names:
        print(f"\nPer-Class F1 Scores:")
        for i, (name, f1) in enumerate(zip(label_names, metrics['f1_per_class'])):
            print(f"  {name:20s}: {f1:.4f}")

    if 'auprc_per_class' in metrics and label_names:
        print(f"\nPer-Class AUPRC:")
        for i, (name, auprc) in enumerate(zip(label_names, metrics['auprc_per_class'])):
            print(f"  {name:20s}: {auprc:.4f}")

    print("\n" + "="*80)


def plot_precision_recall_curves(
    probabilities: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    label_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot precision-recall curves for each class

    Args:
        probabilities: Predicted probabilities [n_samples, num_classes]
        labels: Ground truth labels [n_samples]
        num_classes: Number of classes
        label_names: Optional list of class names
        save_path: Path to save figure
    """
    # Special handling for binary classification
    if num_classes == 2:
        # For binary classification, manually create (n_samples, 2)
        labels_binarized = np.zeros((len(labels), 2))
        labels_binarized[:, 1] = (labels == 1).astype(int)  # Positive class
        labels_binarized[:, 0] = (labels == 0).astype(int)  # Negative class
    else:
        # Binarize labels for multi-class
        labels_binarized = label_binarize(labels, classes=range(num_classes))

        # Handle edge case
        if labels_binarized.shape[1] != num_classes:
            labels_bin_padded = np.zeros((len(labels), num_classes))
            present_classes = np.unique(labels)
            for i, cls in enumerate(present_classes):
                if i < labels_binarized.shape[1]:
                    labels_bin_padded[:, cls] = labels_binarized[:, i]
            labels_binarized = labels_bin_padded

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    # Suppress warnings for classes with no positive samples (expected for rare classes)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='No positive class found in y_true')
        for i in range(num_classes):
            ax = axes[i]

            try:
                precision, recall, _ = precision_recall_curve(
                    labels_binarized[:, i],
                    probabilities[:, i]
                )
                ap = average_precision_score(labels_binarized[:, i], probabilities[:, i])

                ax.plot(recall, precision, linewidth=2, label=f'AUPRC = {ap:.3f}')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')

                class_name = label_names[i] if label_names else f'Class {i}'
                ax.set_title(f'{class_name}')
            except:
                ax.text(0.5, 0.5, 'No samples', ha='center', va='center')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.0])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-Recall curves saved to {save_path}")

    plt.close()
