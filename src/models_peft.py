"""
============================================
Title: GoEmotions PEFT vs Full Fine-Tuning - PEFT (LoRA) model
Author: Hasnaa elidrissi
Date: 08 November 2025

Description:
    Build a tokenizer and backbone classifier wrapped with LoRA
    adapters for parameter-efficient fine-tuning on multi-label
    emotion tasks, and provide dataset/collator utilities for
    Parquet-based GoEmotions splits.

Attribution:
    - Built on Hugging Face `transformers` and `peft` libraries.
    - LoRA configuration and target modules inspired by PEFT
      documentation and adapted for DistilBERT/BERT-style
      attention and feed-forward layers.
============================================
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification #, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model

import pandas as pd
from torch.utils.data import Dataset

# ---- Dataset wrapper (same shape as full FT)----------------------------
class MultiLabelCollator:
    """
    Pads only tokenized inputs and stacks multi-label targets separately.
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

        
class ParquetMultiLabelDataset(Dataset):
    """
    Torch Dataset ove tidy Parquet with columns: text, label_<name>...
    """
    def __init__(self, parquet_path: str, label_names: List[str], tokenizer, max_length: int = 196):
        df = pd.read_parquet(parquet_path)
        self.texts = df["text"].astype(str).tolist()
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

# ---- Utilities --------------------------------------------------------------

def load_label_names(label_map_path: str) -> List[str]:
    with open(label_map_path, "r", encoding="utf-8") as f:
        m = json.load(f)
    if isinstance(list(m.keys())[0], str):
        return [m[str(i)] for i in range(len(m))]
    return [m[i] for i in range(len(m))]

def count_trainable_params(model: torch.nn.Module) -> Tuple[int, int, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total > 0 else 0.0
    return trainable, total, pct

# ---- Target module helpers --------------------------------------------------

def guess_lora_targets(backbone_name: str) -> List[str]:
    """
    Return a list of module name substrings to target with LoRA.
    """
    name = backbone_name.lower()
    if "distilbert" in name:
        # DistilBERT uses 'transformer.layer.<n>.attention' (q_lin, k_lin, v_lin, out_lin) and ff 'lin1', 'lin2'
        return ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"]
    if "roberta" in name or "bert" in name:
        # BERT/Roberta use 'attention.self.query/key/value', 'attention.output.dense', and 'intermediate/output.dense'
        return ["query", "key", "value", "attention.output.dense", "intermediate.dense", "output.dense"]
    # Fallback: common attention names
    return ["q_proj", "k_proj", "v_proj", "o_proj"]

# ---- Builders ---------------------------------------------------------------

def build_tokenizer_and_lora_model(
    backbone: str,
    num_labels: int,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: List[str] | None = None
):
    """
    Create tokenizer +classification model and inject LoRA adapters.
    """
    tok = AutoTokenizer.from_pretrained(backbone)
    base = AutoModelForSequenceClassification.from_pretrained(
        backbone,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    if target_modules is None:
        target_modules = guess_lora_targets(backbone)

    lcfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=target_modules
    )
    peft_model = get_peft_model(base, lcfg)
    return tok, peft_model

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
