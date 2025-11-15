"""
============================================
Title: GoEmotions - Token-Level Importance via Masking
Author: Hasnaa elidrissi
Date: 13 November 2025

Description:
    Loads a trained emotion classification model (full fine-tuning
    baseline or LoRA), selects a small set of test examples, and
    estimates token-level importance for target emotions by masking
    each token and measuring the change in the predicted probability.

    Outputs:
      - reports/figures/token_importance_<label>_example<i>.png

Interpretation:
    - Positive bar values indicate tokens that increase the probability
      of the target emotion.
    - Larger values indicate more influential tokens.

Attribution:
    - Uses Hugging Face AutoTokenizer and AutoModelForSequenceClassification
      (with optional PEFT adapters via PeftModel).
    - Uses NLTK stopwords as part of a simple preprocessing pipeline
      to filter out common function words from visualizations.
============================================
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple
import re

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

import nltk
from nltk.corpus import stopwords

from .config import load_config
from .utils_logging import get_logger

LOGGER = get_logger("goemo.token_importance")


# Ensure NLTK stopwords are available
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

STOP_WORDS = set(stopwords.words("english"))


def load_label_names(label_map_path: Path) -> List[str]:
    """Load label index: name mapping and return an ordered list of label names."""
    import json
    with open(label_map_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if isinstance(list(m.keys())[0], str):
        return [m[str(i)] for i in range(len(m))]
    return [m[i] for i in range(len(m))]


def load_model_and_tokenizer(run_dir: Path, backbone: str, num_labels: int):
    """
    Load a sequence classification model and tokenizer.

    If run_dir is a valid Hugging Face model directory, it is used directly.
    Otherwise the backbone is loaded and LoRA adapters are attached if found.
    """
    try:
        LOGGER.info(f"Trying direct load from {run_dir} ...")
        tok = AutoTokenizer.from_pretrained(run_dir)
        model = AutoModelForSequenceClassification.from_pretrained(run_dir)
        LOGGER.info("Loaded model directly from run directory.")
    except Exception as e:
        LOGGER.warning(f"Direct load failed: {e}")
        LOGGER.info(f"Falling back to backbone '{backbone}' ...")
        tok = AutoTokenizer.from_pretrained(backbone)
        base = AutoModelForSequenceClassification.from_pretrained(
            backbone,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )

        adapter_files = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"]
        try:
            if any((run_dir / f).exists() for f in adapter_files):
                LOGGER.info("LoRA adapters found; attaching to backbone.")
                model = PeftModel.from_pretrained(base, run_dir)
            else:
                LOGGER.info("No adapters found; using plain backbone.")
                model = base
        except Exception as e2:
            LOGGER.warning(f"Failed to load adapters: {e2}")
            model = base

    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
        LOGGER.info("Using CUDA for token-importance analysis.")
    else:
        LOGGER.info("Using CPU for token-importance analysis.")
    return tok, model


def get_token_importance(
    text: str,
    tokenizer,
    model,
    label_idx: int,
    max_length: int = 128,
) -> Tuple[List[str], List[float], float]:
    """
    Compute token-level importance for one text and one emotion label.

    The method:
      1. Compute baseline probability p(label | text).
      2. For each token position, replace that token with [MASK] (or pad/unk),
         recompute p(label |masked_text).
      3. Importance = baseline_prob - masked_prob for that position.

    Returns:
      tokens: list of tokenizer-level tokens
      importances: list of importance scores per token
      baseline_prob: original model probability for the label
    """
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

    if torch.cuda.is_available():
        enc = {k: v.to("cuda") for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        probs = torch.sigmoid(out.logits)[0]
    baseline = float(probs[label_idx].cpu().item())

    input_ids = enc["input_ids"].clone()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    importances: List[float] = []

    mask_id = tokenizer.mask_token_id or tokenizer.pad_token_id or tokenizer.unk_token_id

    for pos in range(input_ids.size(1)):
        masked_ids = input_ids.clone()
        masked_ids[0, pos] = mask_id
        masked_enc = dict(enc)
        masked_enc["input_ids"] = masked_ids

        with torch.no_grad():
            mout = model(**masked_enc)
            mprobs = torch.sigmoid(mout.logits)[0]
        masked_prob = float(mprobs[label_idx].cpu().item())
        importances.append(baseline - masked_prob)

    return tokens, importances, baseline


def _clean_token(tok: str) -> str:
    """
    Clean a single token for visualization.

    Steps:
      - Drop bracketed special tokens like [CLS], [SEP].
      - Remove common subword markers (e.g., '##', '_').
      - Keep only alphabetic characters and apostrophes.
      - Lowercase.
    """
    if tok.startswith("[") and tok.endswith("]"):
        return ""
    t = tok.replace("▁", "").replace("##", "").strip().lower()
    t = re.sub(r"[^a-zA-Z']+", "", t)
    return t


def clean_tokens_for_plot(tokens: List[str], importances: List[float]) -> Tuple[List[str], List[float]]:
    """
    Clean raw tokens and filter out stopwords before plotting.

    This improves readability by keeping only meaningful content words.
    """
    clean_tokens: List[str] = []
    clean_imps: List[float] = []

    for tok, imp in zip(tokens, importances):
        t = _clean_token(tok)
        if not t:
            continue
        if t in STOP_WORDS:
            continue
        clean_tokens.append(t)
        clean_imps.append(imp)

    return clean_tokens, clean_imps


def plot_token_importance(
    tokens: List[str],
    importances: List[float],
    label: str,
    example_idx: int,
    baseline_prob: float,
    run_name: str,
    out_path: Path,
) -> None:
    """
    Create a bar chart showing token-level importance for a single example.

    Tokens and importances are cleaned and filtered prior to plotting.
    """
    tokens, importances = clean_tokens_for_plot(tokens, importances)
    if not tokens:
        LOGGER.warning("No tokens left after cleaning; skipping plot.")
        return

    max_tokens = 40
    tokens = tokens[:max_tokens]
    importances = importances[:max_tokens]

    positions = np.arange(len(tokens))

    plt.figure(figsize=(min(14, 0.4 * len(tokens) + 4), 4))
    plt.bar(positions, importances)
    plt.xticks(positions, tokens, rotation=45, ha="right")
    plt.ylabel("Importance (diff probability)")
    plt.title(
        f"Token importance for '{label}' – example {example_idx} "
        f"({run_name}, baseline p={baseline_prob:.2f})"
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()
    LOGGER.info(f"Saved token-importance plot to {out_path.resolve()}")


def main(
    config_path: str = "configs/default.yaml",
    run_dir: str = "runs/full_ft",
    num_examples: int = 3,
    target_labels: str = "joy,anger",
):
    """
    Entry point for token-importance analysis.

    Loads the trained model and a subset of test texts, then generates one
    importance plot per target emotion per selected example.
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

    run_dir_path = Path(run_dir)
    tokenizer, model = load_model_and_tokenizer(run_dir_path, backbone, num_labels)

    df_test = pd.read_parquet(test_path)
    texts = df_test["text"].astype(str).tolist()
    num_examples = min(num_examples, len(texts))

    rng = np.random.default_rng(seed=int(cfg.get("seed", 42)))
    chosen_idx = rng.choice(len(texts), size=num_examples, replace=False)

    labels_to_use = [s.strip() for s in target_labels.split(",") if s.strip()]
    figs_dir = Path("reports/figures")

    for lab in labels_to_use:
        if lab not in label_names:
            LOGGER.warning(f"Label '{lab}' not in label map; skipping.")
            continue
        label_idx = label_names.index(lab)
        LOGGER.info(f"Analyzing label '{lab}' (index {label_idx}) ...")

        for i, idx in enumerate(chosen_idx):
            text = texts[idx]
            tokens, imps, base_p = get_token_importance(
                text,
                tokenizer,
                model,
                label_idx,
                max_length=int(cfg.get("data.max_length", 128)),
            )
            out_path = figs_dir / f"token_importance_{lab}_example{i}.png"
            plot_token_importance(tokens, imps, lab, i, base_p, run_dir_path.name, out_path)

    LOGGER.info("Token-importance analysis complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token-level importance for emotion labels.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--run_dir", type=str, default="runs/full_ft", help="Model directory.")
    parser.add_argument("--num_examples", type=int, default=3, help="Number of test examples to analyze.")
    parser.add_argument(
        "--target_labels",
        type=str,
        default="joy,anger",
        help="Comma-separated emotion labels to analyze.",
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        run_dir=args.run_dir,
        num_examples=args.num_examples,
        target_labels=args.target_labels,
    )
