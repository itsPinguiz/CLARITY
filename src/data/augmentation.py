"""
Data augmentation strategies for the QEvasion task.

Design principles
-----------------
1. Every public augmentation function has the same signature:
       augment_dataset(ds, label2id, **kwargs) -> Dataset
   so they are fully interchangeable in the training loop.

2. Each function augments ONLY the training split, leaves validation/test
   untouched, and returns a new DatasetDict (original + augmented examples).

3. Augmentation is applied BEFORE tokenization so that the tokenizer always
   sees clean text.

Available strategies
--------------------
- no_augmentation        : identity / baseline (no extra data)
- random_deletion        : EDA-style random word removal
- length_category        : adds a categorical feature based on answer length (Short/Medium/Long)
- tone_analysis          : adds a 'tone' feature using zero-shot classification
- semantic_downsampling  : KMeans-based representative selection via sentence
                           embeddings (requires sentence-transformers, scikit-learn)
- paraphrase_upsampling  : T5-based paraphrase generation for minority classes
                           (requires transformers, sentencepiece)

Usage example
-------------
    from src.data.augmentation import get_augmentation_fn

    augment_fn = get_augmentation_fn("tone_analysis")
    ds_aug = augment_fn(ds, label2id, aug_prob=0.2, seed=42)

    # Smart resampling (model-based, recommended for class imbalance)
    resample_fn = get_augmentation_fn("smart_resampling")
    ds_balanced = resample_fn(ds, label2id, strategy="mean", seed=42)
"""

import random
import math
from collections import Counter
from typing import Callable
import torch
from transformers import pipeline
import numpy as np

from datasets import DatasetDict, Dataset, concatenate_datasets

# Model-based resampling strategies (see src/data/resampling.py)
from src.data.resampling import (
    semantic_downsampling,
    paraphrase_upsampling,
    smart_resampling,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

LABEL_COLUMN = "evasion_label"
TEXT_COLUMNS = ["question", "interview_answer"]


def _set_seed(seed: int):
    random.seed(seed)


def _get_words(text: str) -> list[str]:
    return text.split()


def _join_words(words: list[str]) -> str:
    return " ".join(words)


# ─────────────────────────────────────────────────────────────────────────────
# 1. No augmentation (baseline)
# ─────────────────────────────────────────────────────────────────────────────

def no_augmentation(
    ds: DatasetDict,
    label2id: dict,
    **kwargs,
) -> DatasetDict:
    """Return the dataset unchanged (useful as the control condition)."""
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# 2. EDA — Random Deletion
# ─────────────────────────────────────────────────────────────────────────────

def random_deletion(
    ds: DatasetDict,
    label2id: dict,
    aug_prob: float = 0.10,
    seed: int = 42,
    augment_minority_only: bool = True,
    minority_threshold: float = 0.5,
    **kwargs,
) -> DatasetDict:
    """
    Randomly delete words from the text with probability *aug_prob*.
    Preserves at least one word per text.
    """
    _set_seed(seed)

    counts = Counter(ds["train"][LABEL_COLUMN])
    max_count = max(counts.values())
    minority_labels = (
        {lbl for lbl, cnt in counts.items() if cnt < minority_threshold * max_count}
        if augment_minority_only else None
    )

    def _delete_words(text: str) -> str:
        words = _get_words(text)
        if len(words) == 1:
            return text
        new_words = [w for w in words if random.random() > aug_prob]
        return _join_words(new_words) if new_words else words[0]

    def _augment_example(example: dict) -> dict:
        aug = dict(example)
        for col in TEXT_COLUMNS:
            if col in aug and aug[col]:
                aug[col] = _delete_words(aug[col])
        return aug

    train_ds = ds["train"]
    to_augment = (
        train_ds.filter(lambda ex: ex[LABEL_COLUMN] in minority_labels)
        if minority_labels is not None else train_ds
    )
    new_train = concatenate_datasets([train_ds, to_augment.map(_augment_example)])

    return DatasetDict(
        {"train": new_train, "validation": ds["validation"], "test": ds["test"]}
    )

# ─────────────────────────────────────────────────────────────────────────────
# 3. Answer Length Categorization
# ─────────────────────────────────────────────────────────────────────────────


def add_length_category(
    ds: DatasetDict, 
    text_column: str = "interview_answer",
    seed: int = 42,
    **kwargs
) -> DatasetDict:
    """
    Calculates word counts and categorizes them into Short, Medium, Large.
    This feature is added ONLY to the 'train' split.
    """

    _set_seed(seed)

    print(f"\n[Augmentation] Adding length categories to train split...")

    # Isola il dataset di training
    train_ds = ds["train"]

    # 1. Calcolo del numero di parole
    def _compute_length(example):
        text = str(example[text_column]) if example[text_column] else ""
        return {"word_count": len(text.split())}

    train_ds = train_ds.map(_compute_length, desc="Counting words")

    # 2. Calcolo delle soglie (tertili)
    train_lengths = train_ds["word_count"]
    p33 = np.percentile(train_lengths, 33.33)
    p67 = np.percentile(train_lengths, 66.67)

    # 3. Categorizzazione
    def _categorize_length(example):
        count = example["word_count"]
        if count <= p33:
            return {"length_category": "Short"}
        elif count <= p67:
            return {"length_category": "Medium"}
        else:
            return {"length_category": "Large"}

    # Mappa la colonna e rimuove la colonna temporanea "word_count"
    train_ds = train_ds.map(_categorize_length, desc="Applying categories")
    train_ds = train_ds.remove_columns(["word_count"])

    # Ricostruisce il DatasetDict modificando solo il train
    return DatasetDict({
        "train": train_ds,
        "validation": ds["validation"],
        "test": ds["test"]
    })


# ─────────────────────────────────────────────────────────────────────────────
# 4. Zero-Shot Tone Analysis 
# ─────────────────────────────────────────────────────────────────────────────

def add_tone_feature(
    ds: DatasetDict, 
    label2id: dict, 
    seed: int =42,
    batch_size: int = 16,
    max_words: int = 200,
    **kwargs
) -> DatasetDict:
    """
    Adds a 'tone' column using Zero-Shot Classification.
    This feature is added ONLY to the 'train' split.
    """

    _set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_column = "interview_answer" 

    print(f"\n[Augmentation] Analyzing rhetorical tone on train split...")
    
    candidate_labels = ["Assertive", "Guarded", "Dismissive"]
    
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )

    # Funzione per analizzare un batch di righe
    def _analyze_batch(batch):
        texts = [
            " ".join(str(t).split()[:max_words]) if t else "Unknown" 
            for t in batch[text_column]
        ]
        results = classifier(texts, candidate_labels=candidate_labels, truncation=True)
        tones = [res["labels"][0] for res in results]
        return {"tone": tones}

    # Applica l'analisi SOLO al set di train
    train_ds = ds["train"].map(
        _analyze_batch, 
        batched=True, 
        batch_size=batch_size,
        desc="Analyzing tone (train)"
    )

    # Ricostruisce e restituisce il DatasetDict
    return DatasetDict({
        "train": train_ds,
        "validation": ds["validation"],
        "test": ds["test"]
    })


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

_AUGMENTATION_REGISTRY: dict[str, Callable] = {
    "none":                  no_augmentation,
    "random_deletion":       random_deletion,
    "length_category":       add_length_category,
    "tone_analysis":         add_tone_feature,
    # ── Model-based resampling (src.data.resampling) ──────────────────────────
    "semantic_downsampling": semantic_downsampling,
    "paraphrase_upsampling": paraphrase_upsampling,
}


def get_augmentation_fn(name: str) -> Callable:
    """
    Return the augmentation function registered under *name*.

    Args:
        name: one of the keys in _AUGMENTATION_REGISTRY.

    Raises:
        KeyError with a helpful message if *name* is not registered.
    """
    if name not in _AUGMENTATION_REGISTRY:
        available = ", ".join(_AUGMENTATION_REGISTRY.keys())
        raise KeyError(
            f"Augmentation '{name}' not found.\n"
            f"Available: {available}"
        )
    return _AUGMENTATION_REGISTRY[name]


def list_augmentations() -> list[str]:
    """Return all registered augmentation strategy names."""
    return list(_AUGMENTATION_REGISTRY.keys())