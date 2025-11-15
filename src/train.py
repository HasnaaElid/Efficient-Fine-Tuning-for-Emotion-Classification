"""
============================================
Title: GoEmotions PEFT vs Full Fine-Tuning - training script (Full FT baseline)
Author: Hasnaa elidrissi
Date: 08 November 2025

Description:
    Command-line entry point to train DistilBERT (or another
    Transformers backbone) with full fine-tuning for multi-label
    emotion classification on GoEmotions. Loads processed Parquet
    splits, tunes per-label thresholds on validation, evaluates on
    the test split, and saves metrics, thresholds, and predictions.

Attribution:
    - Uses Hugging Face Trainer and TrainingArguments for training
      and evaluation.
    - Uses EarlyStoppingCallback from the public Transformers API.
============================================
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

from .config import load_config
from .utils_logging import get_logger
try:
    # preferred (package execution: python -m src.train)
    from .models_full import (
        build_tokenizer_and_model,
        build_datasets,
        load_label_names,
        ParquetMultiLabelDataset,
    )
except ImportError:
    # fallback if someone runs the file directly (python src/train.py)
    from models_full import (
        build_tokenizer_and_model,
        build_datasets,
        load_label_names,
        ParquetMultiLabelDataset,
    )
from .metrics import multilabel_f1, per_label_pr_auc
from .thresholds import tune_per_label_thresholds

LOGGER = get_logger("goemo.train")


from inspect import signature

def build_training_args(**kwargs):
    """
    Backward-compatible TrainingArguments builder.
    Filters unknown or renamed parameters (evaluation_strategy, save_strategy, etc.)
    """
    sig = signature(TrainingArguments.__init__).parameters
    # map possible aliases
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in sig and "eval_strategy" in sig:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    if "save_strategy" in kwargs and "save_strategy" not in sig and "save_steps" in sig:
        # convert epoch strategy to default step behaviour
        kwargs.pop("save_strategy")
    if "metric_for_best_model" in kwargs and "metric_for_best_model" not in sig:
        kwargs.pop("metric_for_best_model")
    if "greater_is_better" in kwargs and "greater_is_better" not in sig:
        kwargs.pop("greater_is_better")

    # keep only args this version supports
    filtered = {k: v for k, v in kwargs.items() if k in sig}
    return TrainingArguments(**filtered)


def _load_label_names(processed_dir: Path) -> List[str]:
    label_map_path = processed_dir / "label_map.json"
    with open(label_map_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    # normalize ordering
    if isinstance(list(m.keys())[0], str):
        ordered = [m[str(i)] for i in range(len(m))]
    else:
        ordered = [m[i] for i in range(len(m))]
    return ordered

def main(config_path: str = "configs/default.yaml", output_dir: str = "runs/full_ft"):
    cfg = load_config(config_path)
    processed = Path(cfg.get("paths.processed_dir", "data/processed")).resolve()
    figures   = Path(cfg.get("paths.figures_dir", "reports/figures")).resolve()
    figures.mkdir(parents=True, exist_ok=True)

    label_names = _load_label_names(processed)
    num_labels = len(label_names)
    backbone = cfg.get("model.backbone", "distilbert-base-uncased")
    max_len  = int(cfg.get("data.max_length", 196))

    # Build tokenizer/model and datasets
    tokenizer, model = build_tokenizer_and_model(backbone, num_labels)
    dtrain, dval, collator = build_datasets(
        str(processed / "train.parquet"),
        str(processed / "val.parquet"),
        label_names, tokenizer, max_length=max_len
    )

    # Trainer args (keep reasonable defaults for quick baseline)
    outdir = Path(output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    args = build_training_args(
    output_dir=str(outdir),
    learning_rate=float(cfg.get("train.learning_rate", 5e-5)),
    per_device_train_batch_size=int(cfg.get("train.per_device_train_batch_size", 32)),
    per_device_eval_batch_size=int(cfg.get("train.per_device_eval_batch_size", 64)),
    num_train_epochs=int(cfg.get("train.num_train_epochs", 3)),
    weight_decay=float(cfg.get("train.weight_decay", 0.01)),
    warmup_ratio=float(cfg.get("train.warmup_ratio", 0.1)),
    gradient_accumulation_steps=int(cfg.get("train.gradient_accumulation_steps", 1)),
    fp16=bool(cfg.get("train.fp16", True)),
    logging_steps=50,
    evaluation_strategy="epoch",   # ignored if not supported
    save_strategy="epoch",         # ignored if not supported
    load_best_model_at_end=True,
    metric_for_best_model="eval_micro_f1",
    greater_is_better=True,
    seed=int(cfg.get("seed", 42)),
    report_to=[],
    )


    # compute_metrics callback used by Trainer
    def compute_metrics_fn(eval_pred):
        logits, labels = eval_pred
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        # Default temporary threshold 0.5; proper thresholds tuned after training
        pred_bin = (probs >= 0.5).astype(int)
        micro, macro = multilabel_f1(labels, pred_bin)
        return {"micro_f1": micro, "macro_f1": macro}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dtrain,
        eval_dataset=dval,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    LOGGER.info("Starting training (full fine-tuning baseline)...")
    trainer.train()

    # ===== Validation threshold tuning =====
    LOGGER.info("Running validation predictions for threshold tuning...")
    val_preds = trainer.predict(dval)
    val_probs = torch.sigmoid(torch.tensor(val_preds.predictions)).numpy()
    y_val = val_preds.label_ids

    
    np.save(outdir / "val_probs.npy", val_probs)
    np.save(outdir / "val_labels.npy", y_val)
    tuned_thr = tune_per_label_thresholds(y_val, val_probs, strategy="f1")

    # Save tuned thresholds
    thr_path = outdir / "thresholds_val_per_label.json"
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump({label_names[i]: float(t) for i, t in enumerate(tuned_thr)}, f, indent=2)
    LOGGER.info(f"Saved per-label thresholds to {thr_path}")

    # ===== Evaluate on test split =====
    test_path = processed / "test.parquet"
    
    dtest = ParquetMultiLabelDataset(str(test_path), label_names, tokenizer, max_length=max_len)

    LOGGER.info("Evaluating on test split...")
    test_preds = trainer.predict(dtest)
    test_probs = torch.sigmoid(torch.tensor(test_preds.predictions)).numpy()
    y_test = test_preds.label_ids

    # Metrics at tuned thresholds
    y_test_bin = (test_probs >= tuned_thr.reshape(1, -1)).astype(int)
    micro_f1, macro_f1 = multilabel_f1(y_test, y_test_bin)
    pr_auc_per_label = per_label_pr_auc(y_test, test_probs, labels=label_names)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total > 0 else 0.0

    # Save metrics
    metrics = {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "per_label_pr_auc": pr_auc_per_label,
        "num_labels": num_labels,
        "backbone": backbone,
        "trainable_params": int(trainable),
        "total_params": int(total),
        "pct_trainable": float(pct),
    }
    with open(outdir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    LOGGER.info(f"Test micro-F1={micro_f1:.4f} macro-F1={macro_f1:.4f} (metrics saved).")

    # Save predictions (for later calibration/explainability)
    np.save(outdir / "test_probs.npy", test_probs)
    np.save(outdir / "test_labels.npy", y_test)
    pd.DataFrame(y_test_bin, columns=[f"pred_{n}" for n in label_names]).to_csv(outdir / "test_pred_bin.csv", index=False)

    # if cross-lingual subset exists, evaluate it too
    cfg_tgt = cfg.get("cross_lingual.target_lang", "fr")
    xling_path = processed / f"test_{cfg_tgt}.parquet"
    if xling_path.exists():
        LOGGER.info(f"Evaluating cross-lingual subset ({cfg_tgt})...")
        dtest_x = ParquetMultiLabelDataset(str(xling_path), label_names, tokenizer, max_length=max_len)
        x_preds = trainer.predict(dtest_x)
        x_probs = torch.sigmoid(torch.tensor(x_preds.predictions)).numpy()
        yx = x_preds.label_ids
        yx_bin = (x_probs >= tuned_thr.reshape(1, -1)).astype(int)
        x_micro, x_macro = multilabel_f1(yx, yx_bin)
        out = {
            "lang": cfg_tgt,
            "micro_f1": float(x_micro),
            "macro_f1": float(x_macro),
        }
        with open(outdir / f"test_metrics_xling_{cfg_tgt}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        LOGGER.info(f"Cross-lingual ({cfg_tgt}) micro-F1={x_micro:.4f} macro-F1={x_macro:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train full fine-tuning baseline (multi-label).")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--output_dir", type=str, default="runs/full_ft", help="Directory to save checkpoints/metrics.")
    args = parser.parse_args()
    main(args.config, args.output_dir)
