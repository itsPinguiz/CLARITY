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
- synonym_replacement    : EDA-style synonym swap via NLTK WordNet
- random_deletion        : EDA-style random word removal
- random_swap            : EDA-style random word position swap
- oversampling           : duplicate minority-class examples to balance the dataset
- back_translation       : translate EN→DE→EN with Helsinki-NLP MarianMT models
                           (requires 'transformers' and internet access)
- semantic_downsampling  : KMeans-based representative selection via sentence
                           embeddings (requires sentence-transformers, scikit-learn)
- paraphrase_upsampling  : T5-based paraphrase generation for minority classes
                           (requires transformers, sentencepiece)
- smart_resampling       : combined one-shot pipeline (downsampling + upsampling)

Usage example
-------------
    from src.data.augmentation import get_augmentation_fn

    augment_fn = get_augmentation_fn("synonym_replacement")
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
# 2. EDA — Synonym Replacement
# ─────────────────────────────────────────────────────────────────────────────

def synonym_replacement(
    ds: DatasetDict,
    label2id: dict,
    aug_prob: float = 0.15,
    seed: int = 42,
    augment_minority_only: bool = True,
    minority_threshold: float = 0.5,
    **kwargs,
) -> DatasetDict:
    """
    Replace random words with a WordNet synonym (EDA-style).

    Args:
        aug_prob             : probability of replacing each word.
        seed                 : reproducibility seed.
        augment_minority_only: if True, only augment classes whose relative
                               frequency is below *minority_threshold*.
        minority_threshold   : fraction of the majority class size that
                               defines a 'minority' class.

    Requires: nltk  (pip install nltk)
    On first run, downloads WordNet automatically.
    """
    try:
        import nltk
        from nltk.corpus import wordnet
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except ImportError:
        raise ImportError(
            "NLTK is required for synonym_replacement. "
            "Install it with: pip install nltk"
        )

    _set_seed(seed)

    # Identify minority classes if requested
    if augment_minority_only:
        counts = Counter(ds["train"][LABEL_COLUMN])
        max_count = max(counts.values())
        minority_labels = {
            lbl for lbl, cnt in counts.items()
            if cnt < minority_threshold * max_count
        }
    else:
        minority_labels = None

    def _replace_synonyms(text: str) -> str:
        words = _get_words(text)
        new_words = []
        for word in words:
            if random.random() < aug_prob:
                synsets = wordnet.synsets(word)
                synonyms = [
                    lemma.name().replace("_", " ")
                    for syn in synsets
                    for lemma in syn.lemmas()
                    if lemma.name().lower() != word.lower()
                ]
                new_words.append(random.choice(synonyms) if synonyms else word)
            else:
                new_words.append(word)
        return _join_words(new_words)

    def _augment_example(example: dict) -> dict:
        aug = dict(example)
        for col in TEXT_COLUMNS:
            if col in aug and aug[col]:
                aug[col] = _replace_synonyms(aug[col])
        return aug

    train_ds = ds["train"]

    if minority_labels is not None:
        to_augment = train_ds.filter(
            lambda ex: ex[LABEL_COLUMN] in minority_labels
        )
    else:
        to_augment = train_ds

    augmented = to_augment.map(_augment_example)
    new_train = concatenate_datasets([train_ds, augmented])

    return DatasetDict(
        {
            "train":      new_train,
            "validation": ds["validation"],
            "test":       ds["test"],
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. EDA — Random Deletion
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
# 4. EDA — Random Swap
# ─────────────────────────────────────────────────────────────────────────────

def random_swap(
    ds: DatasetDict,
    label2id: dict,
    n_swaps: int = 2,
    seed: int = 42,
    augment_minority_only: bool = True,
    minority_threshold: float = 0.5,
    **kwargs,
) -> DatasetDict:
    """
    Randomly swap the positions of *n_swaps* pairs of words.
    """
    _set_seed(seed)

    counts = Counter(ds["train"][LABEL_COLUMN])
    max_count = max(counts.values())
    minority_labels = (
        {lbl for lbl, cnt in counts.items() if cnt < minority_threshold * max_count}
        if augment_minority_only else None
    )

    def _swap_words(text: str, n: int) -> str:
        words = _get_words(text)
        if len(words) < 2:
            return text
        for _ in range(n):
            i, j = random.sample(range(len(words)), 2)
            words[i], words[j] = words[j], words[i]
        return _join_words(words)

    def _augment_example(example: dict) -> dict:
        aug = dict(example)
        for col in TEXT_COLUMNS:
            if col in aug and aug[col]:
                aug[col] = _swap_words(aug[col], n_swaps)
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
# 5. Oversampling???
# ─────────────────────────────────────────────────────────────────────────────

def oversampling(
    ds: DatasetDict,
    label2id: dict,
    strategy: str = "majority",   # "majority" | "mean" | float (target count)
    seed: int = 42,
    **kwargs,
) -> DatasetDict:
    """
    Duplicate examples from minority classes to reach a target count.

    Args:
        strategy: how to compute the target count per class.
            "majority" → match the majority class count (full balance).
            "mean"     → match the mean class count (softer balance).
            int/float  → use this absolute count as the target.
    """
    _set_seed(seed)
    train_ds = ds["train"]

    counts = Counter(train_ds[LABEL_COLUMN])

    if strategy == "majority":
        target = max(counts.values())
    elif strategy == "mean":
        target = int(sum(counts.values()) / len(counts))
    else:
        target = int(strategy)

    augmented_splits = []
    for label_str, count in counts.items():
        subset = train_ds.filter(lambda ex, l=label_str: ex[LABEL_COLUMN] == l)
        if count >= target:
            augmented_splits.append(subset)
            continue

        needed = target - count
        # Repeat full copies + one partial copy
        full_reps = needed // count
        remainder = needed % count

        copies = [subset] * (full_reps + 1)
        if remainder:
            indices = random.sample(range(len(subset)), remainder)
            copies.append(subset.select(indices))

        augmented_splits.append(concatenate_datasets(copies))

    new_train = concatenate_datasets(augmented_splits).shuffle(seed=seed)

    return DatasetDict(
        {"train": new_train, "validation": ds["validation"], "test": ds["test"]}
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Back-translation  (EN → DE → EN via MarianMT)
# ─────────────────────────────────────────────────────────────────────────────

def back_translation(
    ds: DatasetDict,
    label2id: dict,
    src_lang: str = "en",
    pivot_lang: str = "de",
    batch_size: int = 16,
    augment_minority_only: bool = True,
    minority_threshold: float = 0.5,
    seed: int = 42,
    **kwargs,
) -> DatasetDict:
    """
    Paraphrase via back-translation: EN → pivot_lang → EN.

    Uses Helsinki-NLP MarianMT models from HuggingFace Hub.
    Requires transformers + sentencepiece:
        pip install transformers sentencepiece

    Note: this is slow (runs inference twice per example). Use GPU if available.
    """
    try:
        from transformers import MarianMTModel, MarianTokenizer
        import torch
    except ImportError:
        raise ImportError(
            "transformers and sentencepiece are required for back_translation. "
            "Install with: pip install transformers sentencepiece"
        )

    _set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(src: str, tgt: str):
        model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        tok = MarianTokenizer.from_pretrained(model_name)
        mdl = MarianMTModel.from_pretrained(model_name).to(device)
        mdl.eval()
        return tok, mdl

    def _translate(texts: list[str], tok, mdl) -> list[str]:
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            encoded = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                translated = mdl.generate(**encoded)
            decoded = tok.batch_decode(translated, skip_special_tokens=True)
            results.extend(decoded)
        return results

    print(f"[back_translation] Loading {src_lang}→{pivot_lang} model...")
    tok_fwd, mdl_fwd = _load_model(src_lang, pivot_lang)
    print(f"[back_translation] Loading {pivot_lang}→{src_lang} model...")
    tok_bwd, mdl_bwd = _load_model(pivot_lang, src_lang)

    train_ds = ds["train"]
    counts = Counter(train_ds[LABEL_COLUMN])
    max_count = max(counts.values())

    minority_labels = (
        {lbl for lbl, cnt in counts.items() if cnt < minority_threshold * max_count}
        if augment_minority_only else None
    )

    to_augment = (
        train_ds.filter(lambda ex: ex[LABEL_COLUMN] in minority_labels)
        if minority_labels is not None else train_ds
    )

    # Translate each text column
    augmented_rows = {col: [] for col in TEXT_COLUMNS}
    for col in TEXT_COLUMNS:
        texts = to_augment[col]
        print(f"[back_translation] Translating column '{col}' ({len(texts)} examples)...")
        pivoted = _translate(texts, tok_fwd, mdl_fwd)
        back = _translate(pivoted, tok_bwd, mdl_bwd)
        augmented_rows[col] = back

    # Rebuild augmented dataset preserving all other columns
    aug_dict = {k: list(to_augment[k]) for k in to_augment.column_names}
    for col in TEXT_COLUMNS:
        aug_dict[col] = augmented_rows[col]

    augmented_ds = Dataset.from_dict(aug_dict)
    new_train = concatenate_datasets([train_ds, augmented_ds])

    return DatasetDict(
        {"train": new_train, "validation": ds["validation"], "test": ds["test"]}
    )

# ─────────────────────────────────────────────────────────────────────────────
# 7. Answer Length Categorization
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
# 8. Zero-Shot Tone Analysis 
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
    "synonym_replacement":   synonym_replacement,
    "random_deletion":       random_deletion,
    "random_swap":           random_swap,
    "oversampling":          oversampling,
    "back_translation":      back_translation,
    "length_category":       add_length_category,
    "tone_analysis":         add_tone_feature,
    # ── Model-based resampling (src.data.resampling) ──────────────────────────
    "semantic_downsampling": semantic_downsampling,
    "paraphrase_upsampling": paraphrase_upsampling,
    "smart_resampling":      smart_resampling,
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