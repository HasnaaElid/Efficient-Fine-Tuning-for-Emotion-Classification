"""
============================================
Title: GoEmotions PEFT vs Full Fine-Tuning - thresholds
Author: Hasnaa Elidrssi
Date:  November 2025

Description:
    Compute one decision threshold per label using validation-set scores.
    For each label, we sweep candidate thresholds derived from the
    precision–recall curve and select the value that maximizes per-label F1.

    This is useful for multi-label classification tasks where a single
    global threshold (e.g., 0.5) is suboptimal due to imbalance or
    skewed label prevalence.

Strategy:
    - 'f1': choose threshold that maximizes F1 for each label.
    - Labels with zero positives in the validation set fall back to 0.5.

Attribution:
    - Uses scikit-learn precision_recall_curve to generate candidate
      thresholds per label.
============================================
"""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve

def tune_per_label_thresholds(
    y_true: np.ndarray, 
    y_val_scores: np.ndarray,
    strategy: str = "f1"
) -> np.ndarray:
    """
    Select a threshold per label using the validation set.
    Strategies: 'f1' (maximize F1 each label), 'youden'(tpr-fpr for bin case; approximate via PR).
    """
    L = y_true.shape[1]
    thresholds = np.zeros(L, dtype=float)

    for j in range(L):
        yj_true = y_true[:, j]
        yj_score = y_val_scores[:, j]
        # If label absent in val, fallback to 0.5
        if yj_true.sum() == 0:
            thresholds[j] = 0.5
            continue
        # PR-based candidate thresholds
        precision, recall, thr = precision_recall_curve(yj_true, yj_score)
        # scikit returns len(thr)=len(precision)-1; build candidates
        cands = np.unique(np.clip(thr, 1e-6, 1-1e-6))
        if cands.size == 0:
            thresholds[j] = 0.5
            continue
        if strategy == "f1":
            best_f1, best_t = -1.0, 0.5
            for t in cands:
                yb = (yj_score >= t).astype(int)
                f1 = f1_score(yj_true, yb, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)
            thresholds[j] = best_t
        else:
            # Default fallback
            thresholds[j] = 0.5
    return thresholds
