# Efficient Emotion Classification with Full Fine-Tuning vs LoRA

Multi-label emotion classification on the GoEmotions dataset, comparing:
- **Full fine-tuning** of a DistilBERT backbone  
- **Parameter-efficient fine-tuning (LoRA)** where only a small fraction of weights are updated  

The project focuses on three things:
1. Building an **end-to-end NLP pipeline** (data prep, modeling, evaluation).
2. Comparing **accuracy vs efficiency** between full fine-tuning and LoRA.
3. Providing **interpretable outputs**: per-emotion performance, calibration, and token-level importance plots.


## 1. Project Goals

- Turn raw GoEmotions text and labels into **clean, tidy Parquet splits**.
- Train a **baseline model** with full fine-tuning on DistilBERT.
- Train a **LoRA model** on the same task and data.
- Compare both setups on:
  - Micro / macro F1 scores
  - Per-label PR–AUC
  - Trainable vs total parameters
  - Calibration (ECE, reliability curves)
- Generate **visuals** for reports:
  - F1 vs trainable parameters
  - Per-label PR–AUC comparison
  - Calibration curves
  - Token-level importance plots for selected emotions


## 2. Dataset

- **Source**: [GoEmotions](https://huggingface.co/datasets/go_emotions) via `datasets.load_dataset("go_emotions")`.
- **Task**: Multi-label emotion classification (28 emotions + neutral).
- **Splits**:
  - Official train / validation / test splits from the dataset.

### Processed artifacts

`src/load_data.py` creates:

- `data/processed/train.parquet`
- `data/processed/val.parquet`
- `data/processed/test.parquet`
- `data/processed/label_map.json`
- `data/processed/label_frequencies.csv`

Each Parquet file has:
- `text`: the cleaned comment text
- `label_<emotion>`: one column per emotion, with 0/1 indicators


## 3. Methods Overview

### 3.1 Models

- **Backbone**: DistilBERT (`distilbert-base-uncased` by default, configurable).
- **Full fine-tuning**:
  - All backbone weights + classification head are trainable.
  - Loss: multi-label setup with `BCEWithLogitsLoss` via Hugging Face.
- **LoRA (PEFT)**:
  - LoRA adapters injected into attention and MLP layers.
  - Only adapter weights are trained; the backbone stays frozen.
  - Same loss and evaluation protocol as full fine-tuning.

Both models are trained and evaluated on the same processed splits and label set.


### 3.2 Threshold Tuning

Because this is a multi-label task, a single 0.5 threshold is often not optimal.

- Validation scores are passed through **per-label threshold tuning** (`src/thresholds.py`).
- For each emotion:
  - Sweep over candidate thresholds from the precision–recall curve.
  - Select the threshold that maximizes **per-label F1**.
- Tuned thresholds are saved to:
  - `runs/full_ft/thresholds_val_per_label.json`
  - `runs/peft_lora/thresholds_val_per_label.json`


### 3.3 Calibration

Optional calibration analysis uses:

- **Expected Calibration Error (ECE)** per label and macro-averaged.
- Reliability curves before and after per-label temperature scaling.

The utilities in `src/calibrate.py` and the calibration comparison script (if used) operate on:

- `val_probs.npy`, `val_labels.npy`
- `test_probs.npy`, `test_labels.npy`  

These arrays are written by the training scripts.


### 3.4 Interpretability

Instead of relying only on global scores, the project includes **token-level importance**:

- `src/token_importance.py` loads a trained model and:
  - Picks a small sample of test texts.
  - Computes token importance for a target emotion by masking each token and measuring the drop in predicted probability.
- Tokens are:
  - Cleaned (subword markers removed, special tokens dropped).
  - Filtered using NLTK English stopwords to keep content-heavy words.
- The script writes bar-chart figures to:
  - `reports/figures/token_importance_<label>_example<i>.png`


