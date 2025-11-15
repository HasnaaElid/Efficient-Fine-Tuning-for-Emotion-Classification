"""
============================================
Title: GoEmotions - Run Summary (Full FT vs LoRA)
Author: Hasnaa Elidrissi
Date: 08 November 2025

Description:
    Load evaluation artifacts from the full fine-tuning run and the
    parameter-efficient LoRA run, extract key metrics, and compile them
    into a single comparison table. The output CSV is used by plotting
    utilities and presentation reports.

    Metrics pulled per run:
        - micro-F1
        - macro-F1
        - trainable parameter count
        - total parameter count
        - percent of parameters updated

Attribution:
    - Consumes JSON files written by src.train_full.py and
      src.train_peft.py (test_metrics.json in each run directory).
============================================
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

def _load_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    mpath = run_dir / "test_metrics.json"
    if not mpath.exists():
        return None
    with open(mpath, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    full_dir = Path("runs/full_ft")
    lora_dir = Path("runs/peft_lora")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    # Full FT row
    full_m = _load_metrics(full_dir)
    if full_m is not None:
        rows.append({
            "run": "full_ft",
            "backbone": full_m.get("backbone", "n/a"),
            "micro_f1": full_m.get("micro_f1", None),
            "macro_f1": full_m.get("macro_f1", None),
            "trainable_params": full_m.get("trainable_params", None),
            "total_params": full_m.get("total_params", None),
            "pct_trainable": full_m.get("pct_trainable", 100.0),
        })
    else:
        print("Warning: runs/full_ft/test_metrics.json not found; skipping full FT.")

    # LoRA row
    lora_m = _load_metrics(lora_dir)
    if lora_m is not None:
        rows.append({
            "run": "peft_lora",
            "backbone": lora_m.get("backbone", "n/a"),
            "micro_f1": lora_m.get("micro_f1", None),
            "macro_f1": lora_m.get("macro_f1", None),
            "trainable_params": lora_m.get("trainable_params", None),
            "total_params": lora_m.get("total_params", None),
            "pct_trainable": lora_m.get("pct_trainable", None),
        })
    else:
        print("Warning: runs/peft_lora/test_metrics.json not found; skipping LoRA.")

    if not rows:
        print("No runs found. Make sure you've trained at least one model.")
        return

    df = pd.DataFrame(rows)
    out_csv = reports_dir / "full_vs_peft_summary.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print("\n=== Full FT vs LoRA summary ===")
    print(df.to_string(index=False))
    print(f"\nSaved summary to {out_csv.resolve()}")

if __name__ == "__main__":
    main()
