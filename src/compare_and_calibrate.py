"""
============================================
Title: GoEmotions - Compare & Calibrate (Full FT vs LoRA)
Author: Hasnaa Elidrissi
Date: 01 November 2025

Description: Compare runs (metrics JSON), compute ECE, fit per-label temperature
             on validation, apply to test, and save reliability plots +summary CSV.
Attribution:
- Calibration metrics and temperature scaling rely on utilities
      implemented in calibrate.py (adapted from Guo et al., 2017)
============================================
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .calibrate import (
    ece_multilabel, fit_temperature_per_label, apply_temperature, plot_reliability
)

def load_arrays(run_dir: Path):
    probs_t = np.load(run_dir / "test_probs.npy")
    y_t     = np.load(run_dir / "test_labels.npy")
    probs_v = np.load(run_dir / "val_probs.npy")
    y_v     = np.load(run_dir / "val_labels.npy")
    return probs_v, y_v, probs_t, y_t

def main():
    # Directories (adjust if different)
    full_dir = Path("runs/full_ft")
    lora_dir = Path("runs/peft_lora")
    figs_dir = Path("reports/figures")
    figs_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for name, rdir in [("full_ft", full_dir), ("peft_lora", lora_dir)]:
        # Load metrics.json (for micro/macro F1)
        mpath = rdir / "test_metrics.json"
        with open(mpath, "r", encoding="utf-8") as f:
            m = json.load(f)
        micro = m.get("micro_f1", None)
        macro = m.get("macro_f1", None)

        # Load arrays for calibration/ECE
        v_probs, v_y, t_probs, t_y = load_arrays(rdir)

        # ECE before
        ece_macro_before, _ = ece_multilabel(t_y, t_probs, n_bins=15)

        # Fit per-label temperature on val, apply to test
        T = fit_temperature_per_label(v_y, v_probs)
        t_probs_cal = apply_temperature(t_probs, T)

        # ECE after
        ece_macro_after, _ = ece_multilabel(t_y, t_probs_cal, n_bins=15)

        # Save reliability curves
        plot_reliability(t_y, t_probs,     f"{name} Reliability (before)", figs_dir / f"{name}_reliability_before.png")
        plot_reliability(t_y, t_probs_cal, f"{name} Reliability (after)",  figs_dir / f"{name}_reliability_after.png")

        # Row
        rows.append({
            "run": name,
            "micro_f1": micro,
            "macro_f1": macro,
            "ece_macro_before": ece_macro_before,
            "ece_macro_after": ece_macro_after
        })

        # Save temperatures for transparency
        np.save(rdir / "temperature_per_label.npy", T)

    # Comparison table
    df = pd.DataFrame(rows)
    df.to_csv("reports/full_vs_peft_summary.csv", index=False, encoding="utf-8")
    print(df)

if __name__ == "__main__":
    main()
