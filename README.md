# CLARITY

[SemEval 2026 Challenge](https://konstantinosftw.github.io/CLARITY-SemEval-2026/#BibTeX) — Unmasking Political Question Evasions

Dataset: [QEvasion](https://huggingface.co/datasets/ailsntua/QEvasion)

---

## Table of Contents

- [Overview](#overview)
- [Tasks](#tasks)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
- [Source Code](#source-code)
- [Results](#results)
- [Setup](#setup)
- [Group Members](#group-members)

---

## Overview

This repository contains the code and experiments developed for the **CLARITY SemEval 2026 shared task**, which focuses on automatically detecting whether a politician's answer to a journalist's question is evasive or not.

The project explores two complementary families of approaches:

1. **Encoder-based fine-tuning** — transformer models (BERT, RoBERTa, Longformer, etc.) fine-tuned to classify political answers directly.
2. **LLM prompting and fine-tuning** — generative models (Llama 3.1 8B-Instruct) evaluated through zero-shot, chain-of-thought, few-shot, and QLoRA fine-tuning strategies.

All experiments share a reproducible data pipeline built on the [QEvasion](https://huggingface.co/datasets/ailsntua/QEvasion) dataset and are evaluated using macro-F1 as the primary metric.

---

## Tasks

The challenge defines two classification tasks evaluated across three sub-tasks (A / B / C):

| Sub-task | Description |
| :------- | :---------- |
| **A — Direct Clarity** | 3-class classification: `Clear Reply`, `Ambivalent`, `Clear Non-Reply` |
| **B — Direct Evasion** | 9-class fine-grained evasion strategy classification |
| **C — Evasion → Clarity** | Predict evasion (Task B) then map to the 3 clarity classes |

The mapping from fine-grained evasion categories to clarity macro-classes is:

| Evasion category | Clarity class |
| :--------------- | :------------ |
| Explicit | Clear Reply |
| Implicit, General, Partial/half-answer, Dodging, Deflection | Ambivalent |
| Declining to answer, Claims ignorance, Clarification | Clear Non-Reply |

---

## Project Structure

```
CLARITY/
├── config/                     # Centralised configuration
│   ├── config.yml              # Training hyper-parameters, dataset paths, output dirs
│   ├── model_configs.py        # Registry of all encoder models with their training settings
│   └── __init__.py
│
├── notebooks/                  # Experiment notebooks (run in order)
│   ├── 01_data_analysis.ipynb
│   ├── 02_train_encoders.ipynb
│   ├── 03_cot_llm.ipynb
│   ├── 04_few_shots_llm.ipynb
│   ├── 05_clarity_tuned_llm.ipynb
│   ├── 06_evasion_tuned_llm.ipynb
│   └── 07_prompt_chain_llm.ipynb
│
├── src/                        # Reusable Python library
│   ├── data/
│   │   ├── dataset_loader.py   # Reproducible train/validation/test splitting + majority voting
│   │   ├── augmentation.py     # Data augmentation strategies (random deletion, paraphrase, etc.)
│   │   ├── resampling.py       # Class resampling and balancing utilities
│   │   └── label_utils.py      # Label encoding helpers
│   ├── training/
│   │   ├── trainers.py         # HuggingFace Trainer wrappers
│   │   ├── metrics.py          # Macro-F1, accuracy, per-class F1 computation
│   │   └── losses.py           # Custom loss functions (focal loss, label smoothing)
│   └── utils/
│       ├── env_utils.py        # Colab/local environment detection and path setup
│       └── results_utils.py    # Saving and loading JSON/CSV result files
│
├── results/                    # Auto-generated experiment outputs (not committed)
│   ├── cot/                    # Chain-of-Thought and zero-shot LLM results
│   ├── few_shots/              # Few-shot prompting results
│   ├── encoder/                # Fine-tuned encoder model results
│   ├── task1/                  # QLoRA fine-tuning results for Task A
│   └── task2/                  # QLoRA fine-tuning results for Task B/C
│
├── report/
│   └── Paper__LLM_Project_2026___CLARITY.pdf   # Final project report
│
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Notebooks

### 01 — Data Exploratory Analysis

**File**: `notebooks/01_data_analysis.ipynb`

Performs an in-depth exploratory analysis of the QEvasion dataset. The notebook computes label distributions for both clarity and evasion classes, calculates text length statistics (token and word counts), visualises the correlation between fine-grained evasion categories and coarser clarity classes, and identifies class imbalance issues that motivate the augmentation strategies used in later experiments.

---

### 02 — Train Encoders

**File**: `notebooks/02_train_encoders.ipynb`

Orchestrates a multi-model, multi-strategy benchmark of encoder-based classifiers on Task A (direct clarity classification). The notebook iterates over:

- **Models**: Longformer-base, RoBERTa-base, XLNet-base, BERT-base-uncased, DistilBERT-base-uncased, Longformer-large
- **Augmentation strategies**: none, random deletion, length category, tone analysis, semantic downsampling, paraphrase upsampling
- **Loss functions**: cross-entropy, focal loss, label-smoothed cross-entropy

All training hyper-parameters are read from `config/config.yml` and model specifications from `config/model_configs.py`. Results (metrics, predictions) are saved to `results/encoder/`.

---

### 03 — Chain-of-Thought with LLMs

**File**: `notebooks/03_cot_llm.ipynb`

Benchmarks **generative prompting strategies** using `meta-llama/Meta-Llama-3.1-8B-Instruct` (4-bit quantised via bitsandbytes) on all three sub-tasks (A, B, C). Five prompting strategies are compared in increasing complexity:

1. **Zero-Shot** — direct classification with no reasoning
2. **CoT** — structured Chain-of-Thought with 3 explicit reasoning steps
3. **CoT + Table** — CoT with the full evasion taxonomy table embedded in the system prompt
4. **CoT + Table + RAG** — adds Wikipedia context retrieval and question-type heuristics
5. **CoT + Table + Tone** — adds tone-of-answer analysis as an additional reasoning signal

Results (JSON metrics, CSV predictions, PNG confusion matrices) are saved to `results/cot/`.

---

### 04 — Few-Shot Prompting with LLMs

**File**: `notebooks/04_few_shots_llm.ipynb`

Benchmarks **few-shot prompting** using the same Llama 3.1 8B-Instruct model. Three shot configurations are tested:

1. **One-Shot** — a single example from the training set
2. **Three-Shot** — one example per clarity class
3. **Nine-Shot** — one example per evasion sub-category (all 9 fine-grained labels)

A fourth strategy explores **dynamic few-shot selection**, retrieving the most similar training examples at inference time using embedding similarity. Each strategy is evaluated on sub-tasks A, B, and C. Results are saved to `results/few_shots/`.

---

### 05 — Clarity Fine-Tuned LLM (Task 1)

**File**: `notebooks/05_clarity_tuned_llm.ipynb`

Fine-tunes **Llama 3.1 8B-Instruct** with **4-bit QLoRA + DoRA** directly on Task A (3-class clarity classification: `Ambivalent`, `Clear Reply`, `Clear Non-Reply`). The model is trained end-to-end on the QEvasion training split and evaluated on the held-out test set. Results and the confusion matrix are saved to `results/task1/`.

---

### 06 — Evasion Fine-Tuned LLM (Task 2)

**File**: `notebooks/06_evasion_tuned_llm.ipynb`

Fine-tunes **Llama 3.1 8B-Instruct** with **4-bit QLoRA + DoRA** on Task B (9-class evasion strategy classification). After inference, evasion predictions are mapped back to the 3 clarity macro-categories to also evaluate Task C performance. Results are saved to `results/task2/`.

---

### 07 — Prompt Chaining and Question Decomposition

**File**: `notebooks/07_prompt_chain_llm.ipynb`

Implements an advanced **Prompt Chaining** pipeline to handle *multi-barrelled questions* — a key limitation in political discourse analysis where a journalist asks several questions in a single turn. The pipeline first decomposes compound questions into atomic sub-questions using a splitter prompt, then evaluates the politician's answer against each sub-question independently using the fine-tuned Task 2 adapter. Final clarity predictions are obtained by aggregating per-sub-question evasion labels. This addresses cases where overall question complexity would mislead a single-pass model.

---

## Source Code

### `src/data/`

| File | Description |
| :--- | :---------- |
| `dataset_loader.py` | Loads QEvasion from HuggingFace and produces a reproducible 3-way train/validation/test split. Enriches the test set with an `evasion_label` column derived from majority voting across three annotators, with reliability-weighted tie-breaking. |
| `augmentation.py` | Implements multiple text augmentation strategies to address class imbalance: random word deletion, length-based categorisation, tone analysis, semantic downsampling, and paraphrase-based upsampling. |
| `resampling.py` | Utilities for oversampling minority classes and undersampling majority classes at the dataset level. |
| `label_utils.py` | Helpers for encoding string labels to integer IDs and back, shared across notebooks and training scripts. |

### `src/training/`

| File | Description |
| :--- | :---------- |
| `trainers.py` | Thin wrappers around the HuggingFace `Trainer` API with project-specific defaults. |
| `metrics.py` | Computes macro-F1, weighted-F1, accuracy, and per-class F1. Also supports simultaneous evaluation of both the evasion task and the mapped clarity task within a single training run. |
| `losses.py` | Custom loss implementations: focal loss (to down-weight easy examples) and label-smoothed cross-entropy. |

### `src/utils/`

| File | Description |
| :--- | :---------- |
| `env_utils.py` | Detects whether the notebook is running locally or on Google Colab, mounts Google Drive if needed, and resolves project-relative paths. |
| `results_utils.py` | Functions to save and reload experiment results (metrics as JSON, predictions as CSV, plots as PNG). |

---

## Results

All experiment outputs are written to the `results/` directory and organised by approach:

| Directory | Contents |
| :-------- | :------- |
| `results/cot/` | Per-experiment JSON metrics, CSV predictions, and PNG confusion matrices for all CoT and zero-shot strategies |
| `results/few_shots/` | Metrics and predictions for all few-shot configurations (1-shot, 3-shot, 9-shot, dynamic) |
| `results/encoder/` | Fine-tuned encoder model checkpoints and evaluation results |
| `results/task1/` | QLoRA Task A fine-tuning predictions and confusion matrix |
| `results/task2/` | QLoRA Task B/C fine-tuning predictions |

---

## Setup

### Requirements

Python 3.10+ is recommended. Install all dependencies with:

```bash
pip install -r requirements.txt
```

### Dataset

The QEvasion dataset is loaded automatically from HuggingFace at runtime:

```python
from src.data.dataset_loader import load_and_split_dataset
ds = load_and_split_dataset()  # returns train / validation / test splits
```

### GPU

Notebooks `03` through `07` require a GPU with at least **16 GB VRAM** for the Llama 3.1 8B model under 4-bit quantisation. They are designed to run on **Google Colab** (A100 or equivalent) and include automatic environment detection to configure paths accordingly.

---

## Group Members

| Name             | Student ID |
| :--------------- | :--------: |
| Stefano Zizzi    |  s346595   |
| Alessio Perrotti |  s346737   |
| Riccardo Vaccari |  s348856   |
| Davide Candela   |  s347245   |
| Luca Lodesani    |  s346978   |
