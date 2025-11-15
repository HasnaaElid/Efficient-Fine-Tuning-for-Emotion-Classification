"""
============================================
Title: GoEmotions PEFT vs Full FT - Calibration utilities
Author: Hasnaa elidrissi
Date: 01 November 2025

Description: Per-label temperature scaling, Expected Calibration Error (ECE),
             and reliability curve plotting for multi-label classification.

Attribution:
    - Calibration approach adapted from:
      Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger,
      "On Calibration of Modern Neural Networks" (ICML 2017).
    - Designed to work with prediction arrays and metrics produced by
      the training scripts (train.py and train_peft.py).
============================================
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt

# ---------- ECE (per-label) ----------

def ece_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error for a single binary label.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true labels with shape (N,)and values in {0, 1}.
    y_prob : np.ndarray
        Array of predicted probabilities with shape (N,) and values in [0, 1].
    n_bins : int
        Number of probability bins.

    Returns
    -------
    float
        Scalar ECE value for this label.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_prob)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc  = (y_true[mask] == (y_prob[mask] >= 0.5)).mean()
        ece += (mask.sum() / N) * abs(acc - conf)
    return float(ece)

def ece_multilabel(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> Tuple[float, List[float]]:
    """
    Compute macro ECE across labels and the per-label ECE vector.

    Parameters
    ----------
    y_true : np.ndarray
        True labels of shape (N, L).
    y_prob : np.ndarray
        Predicted probabilities of shape (N, L).
    n_bins : int
        Number of bins for calibration.
    Returns
    -------
    macro_ece : float
        Mean ECE across labels.
    per_label_ece : list[float]
        ECE value for each label
    """
    L = y_true.shape[1]
    per = [ece_binary(y_true[:, j], y_prob[:, j], n_bins) for j in range(L)]
    return float(np.mean(per)), per

# ---------- Temperature scaling (per-label) ----------

def fit_temperature_per_label(y_true: np.ndarray, y_prob: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    """
    Fit a scalar temperature T_j > 0 for each label j to minimize NLL on validation.
    We optimize in logit space: p' = sigmoid( logit(p) / T ).
    """
    # clamp to avoid inf logits
    eps = 1e-6
    p = np.clip(y_prob, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    L = y_true.shape[1]
    T = np.ones(L, dtype=float)

    # simple 1D line-search per label (robust & dependency-free)
    for j in range(L):
        y = y_true[:, j]
        z = logit[:, j]
        best_nll, best_t = np.inf, 1.0
        # coarse to fine search
        for t in np.concatenate([np.linspace(0.5, 3.0, 26), np.linspace(0.8, 1.5, 15)]):
            p_adj = 1 / (1 + np.exp(-z / t))
            # NLL
            nll = - (y * np.log(p_adj + eps) + (1 - y) * np.log(1 - p_adj + eps)).mean()
            if nll < best_nll:
                best_nll, best_t = nll, float(t)
        T[j] = best_t
    return T

def apply_temperature(y_prob: np.ndarray, T: np.ndarray) -> np.ndarray:
    eps = 1e-6
    p = np.clip(y_prob, eps, 1 - eps)
    logit = np.log(p / (1 - p))
    z = logit / T.reshape(1, -1)
    return 1 / (1 + np.exp(-z))

# ---------- Reliability curve ----------

def plot_reliability(y_true: np.ndarray, y_prob: np.ndarray, title: str, out_path: Path, n_bins: int = 15):
    """
    macro reliability: average the per-bin accuracy and confidence across labels.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    acc_pts, conf_pts = [], []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        # mask across ALL labels
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins -1 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc  = (y_true[mask] == (y_prob[mask] >= 0.5)).mean()
        acc_pts.append(acc)
        conf_pts.append(conf)

    plt.figure(figsize=(5,5))
    plt.plot([0,1], [0,1], linestyle="--" )
    if len(conf_pts) > 0:
        plt.plot(conf_pts, acc_pts, marker="o")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
