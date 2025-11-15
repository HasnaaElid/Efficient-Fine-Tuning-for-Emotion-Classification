"""
============================================
Title: GoEmotions - SHAP Explainability for Emotion Models
Author: hasnaa elidrissi
Date: 12 November 2025

Description:
    Load a trained emotion classification model ( full fine-tuning
    or LoRA), sample a small set of test texts, and compute SHAP
    values to visualize which tokens drive predictions for selected
    emotion labels. Saves token-level explanation figures for use
    in reports and presentations.

Attribution:
    - Uses SHAP text explainers with a text masker to attribute
      token-level contributions.
    - Uses Hugging Face Transformers models and optional LoRA
      adapters via peft.PeftModel.
==============================================
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from .config import load_config
from .utils_logging import get_logger

LOGGER = get_logger("goemo.shap")


def load_label_names(label_map_path: Path) -> List[str]:
    """Load label index: name mapping and return an ordered list of label names."""
    import json
    with open(label_map_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    # keys may be strings from JSON
    if isinstance(list(m.keys())[0], str):
        return [m[str(i)] for i in range(len(m))]
    return [m[i] for i in range(len(m))]


def load_model_and_tokenizer(run_dir: Path, backbone: str, num_labels: int):
    """
    Load a trained model from run_dir if possible.
    If that fails, fall back to a fresh backbone (and attach LoRA adapters if found).
    """
    # 1) Try direct load
    try:
        LOGGER.info(f"Trying to load model directly from {run_dir} ...")
        tok = AutoTokenizer.from_pretrained(run_dir)
        model = AutoModelForSequenceClassification.from_pretrained(run_dir)
        LOGGER.info("Loaded model directly from run directory.")
        model.eval()
        if torch.cuda.is_available():
            model.to("cuda")
        return tok, model
    except Exception as e:
        LOGGER.warning(f"Direct model load failed: {e}")

    # 2) Fallback to backbone
    LOGGER.info(f"Falling back to backbone '{backbone}' ...")
    tok = AutoTokenizer.from_pretrained(backbone)
    base = AutoModelForSequenceClassification.from_pretrained(
        backbone,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    # 3) Try attaching LoRA adapters if present
    try:
        adapter_files = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
        if any((run_dir / f).exists() for f in adapter_files):
            LOGGER.info("Found LoRA adapter files; attaching to backbone...")
            model = PeftModel.from_pretrained(base, run_dir)
        else:
            LOGGER.info("No adapter files found;using plain backbone.")
            model = base
    except Exception as e:
        LOGGER.warning(f"failed to load LoRA adapters from {run_dir}: {e}")
        model = base

    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
        LOGGER.info("Using CUDA for SHAP computations.")
    else:
        LOGGER.info("Using CPU for SHAP computations.")
    return tok, model


def make_predict_fn(tokenizer, model, max_length: int = 128):
    """
    Wrap model into a prediction function that SHAP can call:
      f(texts: List[str]) -> np.ndarray (N, num_labels)
    """
    def predict(texts: List[str]) -> np.ndarray:
        # Ensure we always get a list of plain strings
        texts_clean = [str(t) for t in texts]
        enc = tokenizer(
            texts_clean,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if torch.cuda.is_available():
            enc = {k: v.to("cuda") for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    return predict


def main(
    config_path: str = "configs/default.yaml",
    run_dir: str = "runs/full_ft",
    num_explain: int = 5,
    target_labels: str = "joy,anger",
):
    """
    Run SHAP explainability on a few test examples for selected labels.
    """
    cfg = load_config(config_path)
    processed_dir = Path(cfg.get("paths.processed_dir", "data/processed"))
    label_map_path = processed_dir / "label_map.json"
    test_path = processed_dir / "test.parquet"

    if not test_path.exists():
        raise FileNotFoundError("test.parquet not found. Run src.load_data.py first.")

    label_names = load_label_names(label_map_path)
    num_labels = len(label_names)
    backbone = cfg.get("model.backbone", "distilbert-base-uncased")
    LOGGER.info(f"Loaded {num_labels} labels.")

    run_dir_path = Path(run_dir)
    tokenizer, model = load_model_and_tokenizer(run_dir_path, backbone, num_labels)
    predict_fn = make_predict_fn(tokenizer, model, max_length=int(cfg.get("data.max_length", 128)))

    # Load test texts
    df_test = pd.read_parquet(test_path)
    texts = df_test["text"].astype(str).tolist()

    if num_explain > len(texts):
        num_explain = len(texts)

    rng = np.random.default_rng(seed=int(cfg.get("seed", 42)))
    ex_idx = rng.choice(len(texts), size=num_explain, replace=False)

    # Explicitly build a list of strings to match SHAP's expected input format
    explain_texts = [str(texts[i]) for i in ex_idx]
    explain_texts = list(explain_texts)

    LOGGER.info(f"Explaining {len(explain_texts)} test examples with SHAP.")

    # Text masker + generic Explainer (no KernelExplainer)
    masker = shap.maskers.Text()  # let SHAP handle tokenization boundaries
    explainer = shap.Explainer(predict_fn, masker)

    LOGGER.info("Computing SHAP values (this can take a bit)...")
    shap_values = explainer(explain_texts)  # shape: (samples, tokens, labels)

    figs_dir = Path("reports/figures")
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Convert target_labels string : label indices
    target_list = [s.strip() for s in target_labels.split(",") if s.strip()]
    for lab in target_list:
        if lab not in label_names:
            LOGGER.warning(f"Label '{lab}' not found in label list; skipping.")
            continue
        j = label_names.index(lab)
        LOGGER.info(f"Creating SHAP text plots for label '{lab}' (index {j}) ...")

        # Take up to 3 example explanations for that label
        n_examples = min(len(explain_texts), 3)
        for i in range(n_examples):
            out_path = figs_dir / f"shap_text_{lab}_example{i}.png"
            # shap_values[i] is an Explanation over tokens & labels; slice label j
            sv_ij = shap_values[i, :, j]

            # force matplotlib rendering even if shap returns None
            plt.figure(figsize=(6, 3))
            shap.plots.text(sv_ij)
            plt.title(f"SHAP for '{lab}' - example {i} ({run_dir_path.name})", fontsize=9)
            plt.tight_layout()
            plt.savefig(out_path, dpi=160, bbox_inches="tight")
            plt.close()
            LOGGER.info(f"Saved to {out_path.resolve()}")

    LOGGER.info("SHAP text explanations complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SHAP explainability on emotion models.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="runs/full_ft",
        help="Model directory (e.g. runs/full_ft or runs/peft_lora)."
    )
    parser.add_argument(
        "--num_explain",
        type=int,
        default=5,
        help="Number of test samples to explain."
    )
    parser.add_argument(
        "--target_labels",
        type=str,
        default="joy,anger",
        help="Comma-separated emotion labels to analyze."
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        run_dir=args.run_dir,
        num_explain=args.num_explain,
        target_labels=args.target_labels,
    )
