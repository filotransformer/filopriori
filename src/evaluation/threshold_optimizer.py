"""
Threshold Optimization for Imbalanced Classification

Finds optimal classification threshold for imbalanced datasets.
Default threshold of 0.5 is often inappropriate for classes with low prevalence.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, precision_recall_curve
import logging

logger = logging.getLogger(__name__)


def optimize_threshold_for_minority(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1_macro',
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
    num_thresholds: int = 100
) -> Tuple[float, float, Dict]:
    """
    Find optimal threshold that maximizes chosen metric, with focus on minority class

    For highly imbalanced datasets (e.g., 3% positive), the optimal threshold
    is often much lower than 0.5 (typically 0.03-0.15).

    Args:
        y_true: True labels [N]
        y_prob: Predicted probabilities for POSITIVE class [N]
        metric: Metric to optimize
            - 'f1_macro': F1 macro (balanced)
            - 'f1_minority': F1 of minority class only
            - 'recall_minority': Recall of minority class (max sensitivity)
            - 'precision_minority': Precision of minority class
            - 'balanced_accuracy': (TPR + TNR) / 2
        min_threshold: Minimum threshold to try
        max_threshold: Maximum threshold to try
        num_thresholds: Number of thresholds to test

    Returns:
        best_threshold: Optimal threshold
        best_score: Score at optimal threshold
        metrics_dict: Dictionary with all metrics at optimal threshold
    """
    # Determine minority class
    class_counts = np.bincount(y_true)
    minority_class = int(np.argmin(class_counts))

    logger.info(f"Optimizing threshold for metric: {metric}")
    logger.info(f"Class distribution: {dict(enumerate(class_counts))}")
    logger.info(f"Minority class: {minority_class} ({class_counts[minority_class]} samples, {100*class_counts[minority_class]/len(y_true):.2f}%)")

    # Generate threshold candidates
    thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)

    best_threshold = 0.5
    best_score = -1.0
    best_metrics = {}

    for threshold in thresholds:
        # Make predictions
        y_pred = (y_prob >= threshold).astype(int)

        # Skip if all same prediction
        if len(np.unique(y_pred)) == 1:
            continue

        # Calculate metrics
        try:
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            f1_minority = f1_score(y_true == minority_class, y_pred == minority_class, zero_division=0)
            recall_minority = recall_score(y_true == minority_class, y_pred == minority_class, zero_division=0)
            precision_minority = precision_score(y_true == minority_class, y_pred == minority_class, zero_division=0)

            # Compute TPR and TNR for balanced accuracy
            tp = np.sum((y_pred == 1) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            balanced_acc = (tpr + tnr) / 2

            # Select score based on metric
            if metric == 'f1_macro':
                score = f1_macro
            elif metric == 'f1_minority':
                score = f1_minority
            elif metric == 'recall_minority':
                score = recall_minority
            elif metric == 'precision_minority':
                score = precision_minority
            elif metric == 'balanced_accuracy':
                score = balanced_acc
            else:
                raise ValueError(f"Unknown metric: {metric}")

            # Update best
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'f1_macro': f1_macro,
                    'f1_minority': f1_minority,
                    'recall_minority': recall_minority,
                    'precision_minority': precision_minority,
                    'balanced_accuracy': balanced_acc,
                    'tp': int(tp),
                    'tn': int(tn),
                    'fp': int(fp),
                    'fn': int(fn)
                }

        except Exception as e:
            continue

    logger.info(f"\n{'='*70}")
    logger.info("THRESHOLD OPTIMIZATION RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"Best threshold: {best_threshold:.4f} (vs default 0.5)")
    logger.info(f"Best {metric}: {best_score:.4f}")
    logger.info(f"\nMetrics at optimal threshold:")
    logger.info(f"  F1 Macro:           {best_metrics.get('f1_macro', 0):.4f}")
    logger.info(f"  F1 Minority:        {best_metrics.get('f1_minority', 0):.4f}")
    logger.info(f"  Recall Minority:    {best_metrics.get('recall_minority', 0):.4f}")
    logger.info(f"  Precision Minority: {best_metrics.get('precision_minority', 0):.4f}")
    logger.info(f"  Balanced Accuracy:  {best_metrics.get('balanced_accuracy', 0):.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TP: {best_metrics.get('tp', 0):5d}  |  FN: {best_metrics.get('fn', 0):5d}")
    logger.info(f"  FP: {best_metrics.get('fp', 0):5d}  |  TN: {best_metrics.get('tn', 0):5d}")
    logger.info(f"{'='*70}\n")

    return best_threshold, best_score, best_metrics


def optimize_threshold_youden(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Tuple[float, float]:
    """
    Find optimal threshold using Youden's J statistic (maximizes TPR - FPR)

    This is the threshold that maximizes the distance from the ROC curve to
    the random classifier line.

    Args:
        y_true: True labels [N]
        y_prob: Predicted probabilities for positive class [N]

    Returns:
        best_threshold: Optimal threshold
        youden_j: Youden's J statistic at optimal threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    # Youden's J = TPR - FPR
    youden_j = tpr - fpr

    # Find threshold that maximizes J
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]
    best_youden = youden_j[best_idx]

    logger.info(f"Youden's J optimization:")
    logger.info(f"  Optimal threshold: {best_threshold:.4f}")
    logger.info(f"  Youden's J: {best_youden:.4f}")
    logger.info(f"  TPR: {tpr[best_idx]:.4f}, FPR: {fpr[best_idx]:.4f}")

    return best_threshold, best_youden


def optimize_threshold_f_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = 1.0
) -> Tuple[float, float]:
    """
    Find optimal threshold that maximizes F-beta score

    Args:
        y_true: True labels [N]
        y_prob: Predicted probabilities for positive class [N]
        beta: F-beta parameter (1.0 = F1, 2.0 = F2, etc.)

    Returns:
        best_threshold: Optimal threshold
        best_f_score: F-beta score at optimal threshold
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # Compute F-beta score
    # F_beta = (1 + beta^2) * (precision * recall) / (beta^2 * precision + recall)
    beta_squared = beta ** 2
    f_scores = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall + 1e-10)

    # Find threshold that maximizes F-beta
    best_idx = np.argmax(f_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    best_f_score = f_scores[best_idx]

    logger.info(f"F-{beta} optimization:")
    logger.info(f"  Optimal threshold: {best_threshold:.4f}")
    logger.info(f"  F-{beta} score: {best_f_score:.4f}")
    logger.info(f"  Precision: {precision[best_idx]:.4f}, Recall: {recall[best_idx]:.4f}")

    return best_threshold, best_f_score


def evaluate_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict:
    """
    Evaluate metrics at a given threshold

    Args:
        y_true: True labels [N]
        y_prob: Predicted probabilities for positive class [N]
        threshold: Classification threshold

    Returns:
        Dictionary with all metrics
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate all metrics
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (recall + specificity) / 2

    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'balanced_accuracy': balanced_acc,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = 'recall_minority',
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
    num_thresholds: Optional[int] = None,
    two_phase: bool = False,
    coarse_step: float = 0.02,
    fine_step: float = 0.005,
    fine_window: float = 0.05,
    **kwargs
) -> Tuple[float, Dict]:
    """
    Find optimal threshold using specified strategy

    Args:
        y_true: True labels [N]
        y_prob: Predicted probabilities for positive class [N]
        strategy: Optimization strategy
            - 'recall_minority': Maximize recall of minority class (default for imbalanced)
            - 'f1_macro': Maximize F1 macro (balanced classes)
            - 'f1_minority': Maximize F1 of minority class
            - 'youden': Maximize Youden's J (TPR - FPR)
            - 'f_beta': Maximize F-beta score
            - 'balanced_accuracy': Maximize balanced accuracy
        min_threshold: Minimum threshold to consider
        max_threshold: Maximum threshold to consider
        num_thresholds: Number of thresholds (optional if steps provided)
        two_phase: If True, run coarse then fine search (addresses precision/recall trade-offs)
        coarse_step: Step size for coarse search
        fine_step: Step size for fine search
        fine_window: Window around coarse optimum for fine search
        **kwargs: Additional arguments for specific strategies

    Returns:
        optimal_threshold: Best threshold
        metrics: Dictionary with metrics at optimal threshold
    """
    # Determine number of thresholds if not explicitly provided
    if num_thresholds is None:
        step = coarse_step if two_phase else kwargs.get('step', coarse_step)
        num_thresholds = int((max_threshold - min_threshold) / step) + 1

    minority_strategies = {'recall_minority', 'f1_macro', 'f1_minority', 'balanced_accuracy', 'precision_minority'}

    if two_phase and strategy in minority_strategies:
        # Coarse search
        coarse_num = int((max_threshold - min_threshold) / coarse_step) + 1
        coarse_threshold, _, _ = optimize_threshold_for_minority(
            y_true,
            y_prob,
            metric=strategy,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            num_thresholds=coarse_num
        )

        # Fine search around coarse optimum
        fine_min = max(min_threshold, coarse_threshold - fine_window)
        fine_max = min(max_threshold, coarse_threshold + fine_window)
        fine_num = int((fine_max - fine_min) / fine_step) + 1

        fine_threshold, _, fine_metrics = optimize_threshold_for_minority(
            y_true,
            y_prob,
            metric=strategy,
            min_threshold=fine_min,
            max_threshold=fine_max,
            num_thresholds=fine_num
        )
        fine_metrics['coarse_threshold'] = coarse_threshold
        return fine_threshold, fine_metrics

    if strategy in ['recall_minority', 'f1_macro', 'f1_minority', 'balanced_accuracy', 'precision_minority']:
        threshold, score, metrics = optimize_threshold_for_minority(
            y_true,
            y_prob,
            metric=strategy,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            num_thresholds=num_thresholds
        )
    elif strategy == 'youden':
        threshold, score = optimize_threshold_youden(y_true, y_prob)
        metrics = evaluate_threshold(y_true, y_prob, threshold)
        metrics['youden_j'] = score
    elif strategy == 'f_beta':
        beta = kwargs.get('beta', 1.0)
        threshold, score = optimize_threshold_f_score(y_true, y_prob, beta)
        metrics = evaluate_threshold(y_true, y_prob, threshold)
        metrics['f_beta'] = score
    else:
        raise ValueError(f"Unknown optimization strategy: {strategy}")

    return threshold, metrics
