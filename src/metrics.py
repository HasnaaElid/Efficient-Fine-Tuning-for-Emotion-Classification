"""
============================================
Title: GoEmotions PEFT vs Full Fine-Tuning - metrics
Author: Hasnaa Elidirissi
Date: 08 November 2025

Description: Metric utilities for multi-label emotion classification:
             micro/macro F1, per-label PR-AUC, ECE placeholder.
Attribution:
- Uses scikit-learn metrics APIs.
- Hugging Face Trainer will call `compute_metrics` here.
============================================
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score

def multilabel_f1(y_true: np.ndarray, y_pred_bin: np.ndarray) -> Tuple[float, float]:
    """
    Compute micro- and macro-averaged F1 for multi-label targets.
    Args:
        y_true: (N, L) ground truth in {0,1}
        y_pred_bin: (N, L) predicted labels in {0,1}
    Returns:(micro_f1, macro_f1)
    """
    micro = f1_score(y_true, y_pred_bin, average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    return micro, macro

def per_label_pr_auc(y_true: np.ndarray, y_score: np.ndarray, labels: Optional[list[str]] = None) -> Dict[str, float]:
    """
    Compute per-label PR-AUC (Average Precision). Returns a dict mapping label->AP.
    Args:
        y_true: (N, L) ground truth
        y_score: (N, L) probabilities/scores in [0,1]
    """
    L = y_true.shape[1]
    aps = {}
    for j in range(L):
        ap = average_precision_score(y_true[:, j], y_score[:, j]) if y_true[:, j].sum() > 0 else 0.0
        key = labels[j] if labels else f"label_{j}"
        aps[key] = float(ap)
    return aps

def binarize_probs(y_score: np.ndarray, thresholds: np.ndarray | float = 0.5) -> np.ndarray:
    """
    Convert probabilities to {0,1}using scalar or per-label thresholds.
    """
    if isinstance(thresholds, float) or np.isscalar(thresholds):
        thr = float(thresholds)
        return (y_score >= thr).astype(int)
    thr = np.asarray(thresholds).reshape(1, -1)
    return (y_score >= thr).astype(int)
