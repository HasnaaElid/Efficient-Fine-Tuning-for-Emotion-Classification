"""
Title: Train LoRA Model on GoEmotions
Author: OpenAI Assist (template)
Date: 02 Nov 2025

Description:
    Download the GoEmotions dataset via Hugging Face datasets
    and produce tidy multi-label Parquet splits plus label metadata
    for downstream training and analysis.

Attribution:
    - Uses the publicly available go_emotions dataset from
      Hugging Face Datasets.
    - Train/validation/test split handling follows the official
      dataset configuration.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
from datasets import load_dataset

from .config import load_config
from .utils_logging import get_logger
from datasets.features import Features 

LOGGER = get_logger("goemo.data")

def _preprocess_text(t: str, strip_urls: bool, strip_usernames: bool, normalize_ws: bool) -> str:
    """
    Light text cleanup; keep punctuation/case for stylometrics later.
    """
    if strip_urls:
        t = re.sub(r"http\S+|www\.\S+", " ", t)
    if strip_usernames:
        t = re.sub(r"@[A-Za-z0-9_]+", " ", t)
    if normalize_ws:
        t = re.sub(r"\s+", " ", t).strip()
    return t
 

def _extract_label_names(features: Features) -> List[str]:
    """
    Extract ordered label names from a Hugging Face `Features` object.
    For GoEmotions: features["labels"] is Sequence(ClassLabel(...))
    """
    return list(features["labels"].feature.names)

def _to_tidy_df(split, label_map: Dict[int, str], cfg) -> pd.DataFrame:
    """
    Convert HF split to a DataFrame with columns:
      id, text, label_<name> for each emotion.
    """
    strip_urls = cfg.get("preprocessing.strip_urls", True)
    strip_user = cfg.get("preprocessing.strip_usernames", True)
    norm_ws    = cfg.get("preprocessing.normalize_whitespace", True)

    ids = split["id"] if "id" in split.column_names else list(range(len(split)))
    texts = split["text"]
    labels = split["labels"]  # list of lists of label indices

    rows = []
    for i, (sid, t, lab) in enumerate(zip(ids, texts, labels)):
        t = _preprocess_text(t, strip_urls, strip_user, norm_ws)
        rec = {"id": str(sid), "text": t}
        # initialize all zeros
        for li, name in label_map.items():
            rec[f"label_{name}"] = 0
        # set present labels to 1
        for li in lab:
            rec[f"label_{label_map[li]}"] = 1
        rows.append(rec)

    return pd.DataFrame(rows)

def _save(df: pd.DataFrame, path: Path, save_csv: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    if save_csv:
        df_csv = path.with_suffix(".csv")
        df.to_csv(df_csv, index=False, encoding="utf-8")

def main(out_dir: str, config_path: str = "configs/default.yaml") -> None:
    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading GoEmotions dataset (official splits)...")
    ds = load_dataset("go_emotions")  # official 'train', 'validation', 'test'
    label_names = _extract_label_names(ds["train"].features)  # robust access
    label_map = {i: name for i, name in enumerate(label_names)}

    LOGGER.info(f"{len(label_map)} labels detected: first five -> {label_names[:5]} ...")
    # Tidy dataframes
    train_df = _to_tidy_df(ds["train"], label_map, cfg)
    val_df   = _to_tidy_df(ds["validation"], label_map, cfg)
    test_df  = _to_tidy_df(ds["test"], label_map, cfg)

    # Save
    save_csv = bool(cfg.get("data.save_csv", False))
    _save(train_df, out / "train.parquet", save_csv)
    _save(val_df,   out / "val.parquet",   save_csv)
    _save(test_df,  out / "test.parquet",  save_csv)

    # Label map
    with open(out / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # Simple label frequency report
    freq_rows = []
    for split_name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        for li, name in label_map.items():
            col = f"label_{name}"
            freq = int(df[col].sum())
            freq_rows.append({"split": split_name, "label": name, "count": freq})
    pd.DataFrame(freq_rows).to_csv(out / "label_frequencies.csv", index=False, encoding="utf-8")

    LOGGER.info(f"Saved processed splits and label map to: {out.resolve()}")
    LOGGER.info("Artifacts: train/val/test.parquet, label_map.json, label_frequencies.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process GoEmotions into tidy Parquet splits.")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Output directory.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    main(args.out_dir, args.config)
