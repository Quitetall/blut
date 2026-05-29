#!/usr/bin/env python3
"""
Clinical seizure detection metrics adapted from NEDC's evaluation toolkit.

This module provides numpy/PyTorch-compatible implementations of medical-grade
metrics for seizure detection, including:
- Confusion matrix metrics (TP, TN, FP, FN)
- Standard performance metrics (sensitivity, specificity, precision, F1)
- Clinical metrics (Matthews Correlation Coefficient, false alarm rate)
- Per-label confusion matrices for multi-class evaluation

Adapted from NEDC EEG Evaluation Toolkit v6.0.0
Reference: nedc_eeg_eval_epoch.py (lines ~400-960)
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional


class SeizureMetrics:
    """
    Computes clinical metrics for seizure detection.

    Converts binary classification results into confusion matrices and derives
    sensitivity, specificity, precision, recall, F1, MCC, and false alarm rates.

    Usage:
        metrics = SeizureMetrics(
            total_duration_secs=86400,
            epoch_duration_secs=1.0,
            null_class='bckg'  # background/normal class
        )

        # For binary seizure detection
        metrics.compute_binary(
            tp=50, tn=5000, fp=10, fn=5
        )
        print(f"Sensitivity: {metrics.sensitivity:.4f}")
        print(f"FAR (per 24h): {metrics.false_alarm_rate_24h:.2f}")
    """

    def __init__(self, total_duration_secs: float = 86400.0,
                 epoch_duration_secs: float = 1.0,
                 null_class: str = 'bckg'):
        """
        Initialize metrics calculator.

        Args:
            total_duration_secs: Total duration of recording in seconds (default 24h)
            epoch_duration_secs: Duration of each epoch for false alarm rate calc
            null_class: Background/normal class label
        """
        self.total_duration_secs = total_duration_secs
        self.epoch_duration_secs = epoch_duration_secs
        self.null_class = null_class

        # Confusion matrix values
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0

        # Derived metrics
        self.sensitivity = 0.0  # TPR = tp / (tp + fn)
        self.specificity = 0.0  # TNR = tn / (tn + fp)
        self.precision = 0.0    # PPV = tp / (tp + fp)
        self.npv = 0.0          # tn / (tn + fn)
        self.fnr = 0.0          # 1 - sensitivity
        self.fpr = 0.0          # 1 - specificity
        self.f1_score = 0.0
        self.mcc = 0.0          # Matthews Correlation Coefficient
        self.false_alarm_rate_24h = 0.0
        self.accuracy = 0.0
        self.prevalence = 0.0

    def compute_binary(self, tp: int, tn: int, fp: int, fn: int) -> None:
        """
        Compute metrics from binary confusion matrix.

        Args:
            tp: True positives
            tn: True negatives
            fp: False positives
            fn: False negatives
        """
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn
        self._compute_metrics()

    def compute_from_arrays(self, y_true: np.ndarray, y_pred: np.ndarray,
                           threshold: float = 0.5) -> None:
        """
        Compute metrics from prediction arrays.

        Args:
            y_true: Ground truth binary labels (0/1)
            y_pred: Predicted probabilities or binary labels
            threshold: Probability threshold for binary classification
        """
        # Ensure binary predictions
        if y_pred.max() > 1.0:
            y_pred = (y_pred > threshold).astype(int)
        else:
            y_pred = (y_pred > threshold).astype(int)

        self.tp = np.sum((y_true == 1) & (y_pred == 1))
        self.tn = np.sum((y_true == 0) & (y_pred == 0))
        self.fp = np.sum((y_true == 0) & (y_pred == 1))
        self.fn = np.sum((y_true == 1) & (y_pred == 0))

        self._compute_metrics()

    def _compute_metrics(self) -> None:
        """Compute all derived metrics from confusion matrix."""
        tp, tn, fp, fn = self.tp, self.tn, self.fp, self.fn

        # Sensitivity (True Positive Rate) = tp / (tp + fn)
        if (tp + fn) > 0:
            self.sensitivity = float(tp) / float(tp + fn)
        else:
            self.sensitivity = 0.0

        # Specificity (True Negative Rate) = tn / (tn + fp)
        if (tn + fp) > 0:
            self.specificity = float(tn) / float(tn + fp)
        else:
            self.specificity = 0.0

        # Precision (Positive Predictive Value) = tp / (tp + fp)
        if (tp + fp) > 0:
            self.precision = float(tp) / float(tp + fp)
        else:
            self.precision = 0.0

        # NPV (Negative Predictive Value) = tn / (tn + fn)
        if (tn + fn) > 0:
            self.npv = float(tn) / float(tn + fn)
        else:
            self.npv = 0.0

        # False Negative Rate & False Positive Rate
        self.fnr = 1.0 - self.sensitivity
        self.fpr = 1.0 - self.specificity

        # F1 Score = 2 * precision * sensitivity / (precision + sensitivity)
        f1_denom = self.precision + self.sensitivity
        if f1_denom > 0:
            self.f1_score = 2.0 * self.precision * self.sensitivity / f1_denom
        else:
            self.f1_score = 0.0

        # Matthews Correlation Coefficient
        # MCC = (tp*tn - fp*fn) / sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        mcc_denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        if mcc_denom > 0:
            mcc_num = (tp * tn) - (fp * fn)
            self.mcc = float(mcc_num) / math.sqrt(float(mcc_denom))
        else:
            self.mcc = 0.0

        # Accuracy = (tp + tn) / (tp + tn + fp + fn)
        total = tp + tn + fp + fn
        if total > 0:
            self.accuracy = float(tp + tn) / float(total)
        else:
            self.accuracy = 0.0

        # Prevalence = (tp + fn) / (tp + tn + fp + fn)
        if total > 0:
            self.prevalence = float(tp + fn) / float(total)
        else:
            self.prevalence = 0.0

        # False Alarm Rate per 24 hours
        # FAR_24h = false_positives * epoch_duration / total_duration * (60*60*24)
        if self.total_duration_secs > 0:
            self.false_alarm_rate_24h = (
                float(fp) * self.epoch_duration_secs / self.total_duration_secs
                * (60 * 60 * 24)
            )
        else:
            self.false_alarm_rate_24h = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Return all metrics as a dictionary."""
        return {
            'tp': int(self.tp),
            'tn': int(self.tn),
            'fp': int(self.fp),
            'fn': int(self.fn),
            'sensitivity': float(self.sensitivity),
            'specificity': float(self.specificity),
            'precision': float(self.precision),
            'npv': float(self.npv),
            'fnr': float(self.fnr),
            'fpr': float(self.fpr),
            'f1_score': float(self.f1_score),
            'mcc': float(self.mcc),
            'accuracy': float(self.accuracy),
            'prevalence': float(self.prevalence),
            'false_alarm_rate_24h': float(self.false_alarm_rate_24h),
        }

    def __repr__(self) -> str:
        """String representation of metrics."""
        return (
            f"SeizureMetrics("
            f"TP={self.tp}, TN={self.tn}, FP={self.fp}, FN={self.fn}, "
            f"Sens={self.sensitivity:.4f}, Spec={self.specificity:.4f}, "
            f"Prec={self.precision:.4f}, F1={self.f1_score:.4f}, "
            f"MCC={self.mcc:.4f}, FAR_24h={self.false_alarm_rate_24h:.2f})"
        )


def sensitivity(tp: int, fn: int) -> float:
    """Compute sensitivity (TPR) = tp / (tp + fn)."""
    if (tp + fn) > 0:
        return float(tp) / float(tp + fn)
    return 0.0


def specificity(tn: int, fp: int) -> float:
    """Compute specificity (TNR) = tn / (tn + fp)."""
    if (tn + fp) > 0:
        return float(tn) / float(tn + fp)
    return 0.0


def precision(tp: int, fp: int) -> float:
    """Compute precision (PPV) = tp / (tp + fp)."""
    if (tp + fp) > 0:
        return float(tp) / float(tp + fp)
    return 0.0


def recall(tp: int, fn: int) -> float:
    """Compute recall (same as sensitivity)."""
    return sensitivity(tp, fn)


def f1_score(tp: int, fp: int, fn: int) -> float:
    """Compute F1 score = 2*precision*recall / (precision + recall)."""
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    if (prec + rec) > 0:
        return 2.0 * prec * rec / (prec + rec)
    return 0.0


def matthews_cc(tp: int, tn: int, fp: int, fn: int) -> float:
    """
    Compute Matthews Correlation Coefficient.

    MCC = (tp*tn - fp*fn) / sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))

    MCC ranges from -1 (perfect disagreement) to +1 (perfect agreement).
    MCC=0 indicates random classification.

    Better than accuracy for imbalanced datasets.

    Args:
        tp, tn, fp, fn: Confusion matrix values

    Returns:
        MCC value in range [-1, 1]
    """
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom > 0:
        num = (tp * tn) - (fp * fn)
        return float(num) / math.sqrt(float(denom))
    return 0.0


def false_alarm_rate_24h(fp: int, epoch_duration_secs: float = 1.0,
                         total_duration_secs: float = 86400.0) -> float:
    """
    Compute false alarm rate per 24 hours.

    FAR_24h = false_positives * epoch_duration / total_duration * 86400

    This gives the expected number of false alarms in 24 hours of recording.

    Args:
        fp: Number of false positives
        epoch_duration_secs: Duration of each epoch (detection interval)
        total_duration_secs: Total recording duration in seconds

    Returns:
        Expected false alarms per 24 hours
    """
    if total_duration_secs > 0:
        return (float(fp) * epoch_duration_secs / total_duration_secs
                * (60 * 60 * 24))
    return 0.0


def per_label_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                               labels: Optional[List[int]] = None) -> Dict[int, Dict]:
    """
    Compute per-label confusion matrices for multi-class classification.

    For each label, converts the N×N confusion matrix to a 2×2 matrix
    (label vs. all others) and computes metrics.

    Args:
        y_true: Ground truth class labels
        y_pred: Predicted class labels
        labels: List of unique labels (if None, auto-detected)

    Returns:
        Dictionary mapping each label to its metrics dict
    """
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    result = {}

    for label in labels:
        # Convert to binary: label vs. not label
        y_true_bin = (y_true == label).astype(int)
        y_pred_bin = (y_pred == label).astype(int)

        # Compute confusion matrix
        tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
        tn = np.sum((y_true_bin == 0) & (y_pred_bin == 0))
        fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
        fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))

        # Compute metrics
        result[label] = {
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'sensitivity': sensitivity(int(tp), int(fn)),
            'specificity': specificity(int(tn), int(fp)),
            'precision': precision(int(tp), int(fp)),
            'f1_score': f1_score(int(tp), int(fp), int(fn)),
            'mcc': matthews_cc(int(tp), int(tn), int(fp), int(fn)),
        }

    return result


def accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    """Compute accuracy = (tp + tn) / (tp + tn + fp + fn)."""
    total = tp + tn + fp + fn
    if total > 0:
        return float(tp + tn) / float(total)
    return 0.0
