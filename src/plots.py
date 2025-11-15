"""
============================================
Title: GoEmotions - Plotting Utilities
Author: Hasnaa elidrissi
Date: 10 November 2025

Description:
    Plot helpers for comparing full fine-tuning and LoRA runs on the
    GoEmotions dataset. Includes micro/macro F1 vs trainable parameters,
    top-k per-label PR-AUC comparison, and PR-AUC difference distribution
    across emotions.

Attribution:
    - Uses metrics and summary CSVs produced by the training and
      calibration scripts in this project.
    - Built on pandas and matplotlib for figure generation.
============================================
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import json
import pandas as pd
import matplotlib.pyplot as plt

# ---------- plot_f1_vs_params is fine to keep ----------

def plot_f1_vs_params(summary_csv: str = "reports/full_vs_peft_summary.csv",
                      out_path: str = "reports/figures/f1_vs_params.png") -> None:
    """
    Bar chart showing micro-F1 for each run, annotated with
    percent trainable parameters. Uses the summary CSV produced
    by the calibration/comparison script.
    """
    csv_path = Path(summary_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run summarize_runs.py first.")
    df = pd.read_csv(csv_path)

    runs = df["run"].tolist()
    micro = df["micro_f1"].tolist()
    pct = df["pct_trainable"].fillna(100.0).tolist()

    fig_dir = Path(out_path).parent
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    x = range(len(runs))
    plt.bar(x, micro)
    plt.xticks(x, runs)
    plt.ylabel("Micro F1")
    plt.title("Micro F1 vs % Trainable Parameters")

    for i, (m, p) in enumerate(zip(micro, pct)):
        plt.text(i, m + 0.005, f"{p:.1f}% params", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved plot : {Path(out_path).resolve()}")

# ---------- FIGURE 2: Macro F1 vs params -------------------------

def plot_macro_f1( summary_csv: str = "reports/full_vs_peft_summary.csv",
                  out_path: str = "reports/figures/macro_f1_vs_params.png") -> None:
    """
    Bar chart for macro-F1 per run. Complements micro-F1 figure.
    """
    csv_path = Path(summary_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run summarize_runs.py first.")
    df = pd.read_csv(csv_path)

    runs = df["run"].tolist()
    macro = df["macro_f1"].tolist()

    fig_dir = Path(out_path).parent
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    x = range(len(runs))
    plt.bar(x, macro)
    plt.xticks(x, runs)
    plt.ylabel("Macro F1")
    plt.title("Macro F1 by Run")

    for i, m in enumerate(macro):
        plt.text(i, m + 0.005, f"{m:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved plot : {Path(out_path).resolve()}")


def _load_per_label_pr_auc(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        m: Dict[str, Any] = json.load(f)
    return m.get("per_label_pr_auc", {})

# ---------- FIGURE 3: Top-k per-label PR-AUC comparison----------

def plot_pr_auc_topk(full_metrics_path: str = "runs/full_ft/test_metrics.json",
                     peft_metrics_path: str = "runs/peft_lora/test_metrics.json",
                     top_k: int = 10,
                     out_path: str = "reports/figures/pr_auc_topk_full_vs_peft.png") -> None:
    """
    Grouped bar chart: top-k labels (by Full FT PR-AUC) with
    PR-AUC for both Full FT and LoRA.
    """
    full_dict = _load_per_label_pr_auc(full_metrics_path)
    peft_dict = _load_per_label_pr_auc(peft_metrics_path)
    if not full_dict:
        raise ValueError("per_label_pr_auc not found in full_ft metrics.")
    if not peft_dict:
        raise ValueError("per_label_pr_auc not found in peft_lora metrics.")

    df = pd.DataFrame({
        "label": list(full_dict.keys()),
        "full_ft": list(full_dict.values()),
    })
    df["peft_lora"] = df["label"].map(peft_dict)
    # sort by full_ft descending, keep top_k
    df = df.sort_values("full_ft", ascending=False).head(top_k)

    fig_dir = Path(out_path).parent
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    x = range(len(df))
    width = 0.4

    plt.bar([i - width/2 for i in x], df["full_ft"], width=width, label="full_ft")
    plt.bar([i + width/2 for i in x], df["peft_lora"], width=width, label="peft_lora")

    plt.xticks(list(x), df["label"], rotation=45, ha="right")
    plt.ylabel("PR-AUC")
    plt.title(f"Top {top_k} Emotions by PR-AUC (Full FT vs LoRA)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved plot to {Path(out_path).resolve()}")

# ---------- FIGURE 4: PR-AUC difference distribution -------------

def plot_pr_auc_diff_hist(full_metrics_path: str = "runs/full_ft/test_metrics.json",
                          peft_metrics_path: str = "runs/peft_lora/test_metrics.json",
                          out_path: str = "reports/figures/pr_auc_diff_hist.png") -> None:
    """
    Histogram of PR-AUC differences (LoRA - Full FT) across labels.
    Shows where LoRA is better/worse.
    """
    full_dict = _load_per_label_pr_auc(full_metrics_path)
    peft_dict = _load_per_label_pr_auc(peft_metrics_path)
    if not full_dict or not peft_dict:
        raise ValueError("Missing per_label_pr_auc in metrics.")

    labels = sorted(full_dict.keys())
    diffs = []
    for lab in labels:
        f = full_dict.get(lab, 0.0)
        p = peft_dict.get(lab, 0.0)
        diffs.append(p - f)

    fig_dir = Path(out_path).parent
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.hist(diffs, bins=10)
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("PR-AUC (LoRA - Full FT)")
    plt.ylabel("Count of Labels")
    plt.title("Distribution of PR-AUC Differences Across Emotions")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"Saved plot to {Path(out_path).resolve()}")

if __name__ == "__main__":
    # Generate all figures in one go 
    plot_f1_vs_params()
    plot_macro_f1()
    plot_pr_auc_topk()
    plot_pr_auc_diff_hist()
