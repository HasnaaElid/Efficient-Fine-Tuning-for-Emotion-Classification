"""
============================================
Title: GoEmotions PEFT vs Full Fine-Tuning - full FT model
Author: Hasnaa Elidrissi
Date: 08 November 2025

Description:
    DistilBERT full fine-tuning setup for multi-label emotion
    classification. Provides Parquet-backed dataset wrappers,
    a multi-label collator, and a model factory configured with
    problem_type="multi_label_classification" for use with
    Hugging Face training workflows.

Attribution:
    - Built on Hugging Face transformers classification models
      and tokenizers.
============================================
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    #DataCollatorWithPadding,
)

# --- (MultiLabelCollator) ---
class MultiLabelCollator:
    """
    Pads only tokenized inputs and stack multi-label targets separately.
      """
    def __init__(self, tokenizer, max_length: int = 196):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # remove labels before padding
        features = [{k: v for k, v in b.items() if k != "labels"} for b in batch]
        try:
            # newer versions support truncation & max_length in pad
            enc = self.tokenizer.pad(
                features,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        except TypeError:
            # older versions only accept padding & return_tensors
            enc = self.tokenizer.pad(
                features,
                padding=True,
                return_tensors="pt",
            )
        # stack labels as float tensor
        labels = torch.stack([
            b["labels"] if isinstance(b["labels"], torch.Tensor)
            else torch.tensor(b["labels"], dtype=torch.float)
            for b in batch
        ])
        enc["labels"] = labels
        return enc


@dataclass
class TextExample:
    text: str
    labels: np.ndarray  # shape (L,)

class ParquetMultiLabelDataset(Dataset):
    """
    Torch Dataset over tidy Parquet with columns: text, label_<name>...
    """
    def __init__(self, parquet_path: str, label_names: List[str], tokenizer, max_length: int = 196):
        df = pd.read_parquet(parquet_path)
        self.texts = df["text"].astype(str).tolist()
        # stack label columns as multi-hot
        label_cols = [f"label_{n}" for n in label_names]
        self.labels = df[label_cols].astype(int).values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        t = self.texts[idx]
        enc = self.tokenizer(
            t,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        enc["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return enc

def load_label_names(label_map_path: str) -> List[str]:
    """
    Read index:name map and return ordered list of label names.
    """
    with open(label_map_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    # keys are string indices when saved via json; sort by int key
    ordered = [m[str(i)] if isinstance(list(m.keys())[0], str) else m[i] for i in range(len(m))]
    return ordered

def build_tokenizer_and_model(backbone: str, num_labels: int):
    """
    Create tokenizer and a HF classification head configured for multi-label (problem_type).
    """
    tok = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    return tok, model

def build_datasets(
    train_path: str,
    val_path: str,
    label_names: List[str],
    tokenizer,
    max_length: int = 196
):
    dtrain = ParquetMultiLabelDataset(train_path, label_names, tokenizer, max_length)
    dval   = ParquetMultiLabelDataset(val_path,   label_names, tokenizer, max_length)
    collator = MultiLabelCollator(tokenizer=tokenizer, max_length=max_length)
    return dtrain, dval, collator
