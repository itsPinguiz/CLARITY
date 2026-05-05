"""
Model-based resampling strategies for the QEvasion task.

This module provides smart class-balancing techniques that go beyond
simple duplication (``oversampling`` in augmentation.py):

Strategies
----------
- semantic_downsampling : KMeans-based representative selection using
                          sentence embeddings — reduces an over-represented
                          class while preserving its semantic diversity.
- paraphrase_upsampling : T5-based paraphrase generation to grow an
                          under-represented class with varied synthetic
                          examples.
- smart_resampling      : combined one-shot pipeline that applies
                          semantic_downsampling to classes above the target
                          and paraphrase_upsampling to classes below it.

Interface
---------
All functions follow the same interface as src.data.augmentation:
    fn(ds: DatasetDict, label2id: dict, **kwargs) -> DatasetDict

Only the training split is modified. Validation and test splits are
always returned unchanged.

Design notes
------------
- ``LABEL_COLUMN`` is intentionally redefined here (rather than imported
  from augmentation.py) to avoid a circular import when augmentation.py
  registers these functions in its registry.  Both files must always use
  the same string value — currently ``"evasion_label"``.
- Synthetic rows are marked with ``is_augmented = True`` so downstream
  analysis can distinguish real from generated examples.

Dependencies (not in base requirements.txt)
-------------------------------------------
    pip install sentence-transformers scikit-learn
    # transformers is already a project dependency
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Union

import numpy as np
import torch
from datasets import DatasetDict, Dataset, concatenate_datasets


# ── Shared constants ──────────────────────────────────────────────────────────
# NOTE: Keep in sync with the definition in src.data.augmentation.
LABEL_COLUMN = "evasion_label"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_target(
    counts: Counter,
    strategy: Union[str, int, float],
) -> int:
    """
    Derive the per-class target count from a strategy descriptor.

    Args:
        counts   : Counter mapping label string → current example count.
        strategy : one of:
                   "mean"     → arithmetic mean of class counts.
                   "median"   → median of class counts.
                   "majority" → size of the largest class (no downsampling).
                   int/float  → absolute target count (cast to int).

    Returns:
        Target count as a plain Python int.
    """
    if strategy == "mean":
        return int(sum(counts.values()) / len(counts))
    elif strategy == "median":
        return int(np.median(list(counts.values())))
    elif strategy == "majority":
        return max(counts.values())
    else:
        return int(strategy)


def _backfill_augmented_flag(splits: list[Dataset]) -> list[Dataset]:
    """
    Ensure every split in *splits* has an ``is_augmented`` boolean column.

    Splits that were created before the flag was needed receive
    ``is_augmented = False`` for every row.

    Args:
        splits : list of Dataset objects, some of which may already
                 carry the ``is_augmented`` column.

    Returns:
        New list where every Dataset has the ``is_augmented`` column.
    """
    if not any("is_augmented" in s.column_names for s in splits):
        return splits  # flag was never added — nothing to backfill

    backfilled = []
    for split in splits:
        if "is_augmented" not in split.column_names:
            split = split.add_column("is_augmented", [False] * len(split))
        backfilled.append(split)
    return backfilled


# ─────────────────────────────────────────────────────────────────────────────
# 1. Semantic Downsampling
# ─────────────────────────────────────────────────────────────────────────────

def semantic_downsampling(
    ds: DatasetDict,
    label2id: dict,
    target_size: int | None = None,
    strategy: Union[str, int] = "mean",
    embed_column: str = "interview_answer",
    embed_model_name: str = "all-MiniLM-L6-v2",
    seed: int = 42,
    **kwargs,
) -> DatasetDict:
    """
    Reduce over-represented classes while preserving semantic diversity.

    For each class whose training-set count exceeds *target_size*, we:
      1. Encode every example with a SentenceTransformer.
      2. Cluster the embeddings into *target_size* KMeans groups.
      3. Keep only the real example closest to each cluster centroid.

    This guarantees the reduced subset covers the full semantic spread
    of the original class, unlike random undersampling which may
    accidentally collapse rare sub-topics.

    Classes at or below the target size are kept entirely intact.

    Args:
        ds              : DatasetDict with train / validation / test splits.
        label2id        : class → integer id mapping (from label_utils).
        target_size     : desired number of examples per over-represented
                          class.  If None, derived from *strategy*.
        strategy        : "mean" | "median" | "majority" | int.
                          Ignored when *target_size* is set explicitly.
        embed_column    : text column used to compute sentence embeddings.
                          Defaults to ``"interview_answer"``.
        embed_model_name: SentenceTransformer model identifier.
        seed            : random seed forwarded to KMeans.

    Returns:
        DatasetDict with a downsampled training split.

    Requires:
        pip install sentence-transformers scikit-learn
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances_argmin_min
    except ImportError:
        raise ImportError(
            "sentence-transformers and scikit-learn are required for "
            "semantic_downsampling.\n"
            "Install with: pip install sentence-transformers scikit-learn"
        )

    _set_seed(seed)
    train_ds = ds["train"]
    counts = Counter(train_ds[LABEL_COLUMN])
    target = target_size if target_size is not None else _compute_target(counts, strategy)

    print(f"[semantic_downsampling] Target size per class : {target}")

    embed_model = SentenceTransformer(embed_model_name)
    kept_splits: list[Dataset] = []

    for label_str, count in counts.items():
        subset = train_ds.filter(lambda ex, l=label_str: ex[LABEL_COLUMN] == l)

        if count <= target:
            kept_splits.append(subset)
            continue

        print(
            f"[semantic_downsampling]   '{label_str}': {count} → {target} "
            f"(-{count - target})"
        )

        texts = [str(t) if t else "" for t in subset[embed_column]]
        embeddings = embed_model.encode(
            texts, show_progress_bar=True, convert_to_numpy=True
        )

        kmeans = KMeans(n_clusters=target, random_state=seed, n_init=10)
        kmeans.fit(embeddings)
        closest_indices, _ = pairwise_distances_argmin_min(
            kmeans.cluster_centers_, embeddings
        )
        kept_splits.append(subset.select(closest_indices.tolist()))

    new_train = concatenate_datasets(kept_splits).shuffle(seed=seed)
    print(
        f"[semantic_downsampling] Train size : {len(train_ds)} → {len(new_train)}"
    )

    return DatasetDict(
        {"train": new_train, "validation": ds["validation"], "test": ds["test"]}
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Paraphrase Upsampling
# ─────────────────────────────────────────────────────────────────────────────

def paraphrase_upsampling(
    ds: DatasetDict,
    label2id: dict,
    target_size: int | None = None,
    strategy: Union[str, int] = "mean",
    paraphrase_column: str = "interview_answer",
    model_name: str = "humarin/chatgpt_paraphraser_on_T5_base",
    batch_size: int = 12,
    max_input_length: int = 512,
    max_output_length: int = 128,
    num_beams: int = 5,
    seed: int = 42,
    **kwargs,
) -> DatasetDict:
    """
    Grow under-represented classes using T5-based paraphrase generation.

    For each class whose training-set count falls below *target_size*, we:
      1. Sample existing examples (with replacement when needed).
      2. Generate a paraphrase of *paraphrase_column* for each sampled row.
      3. Mark synthetic rows with ``is_augmented = True``.
      4. Concatenate the synthetic rows with the original class examples.

    Only *paraphrase_column* (the interview answer) is rewritten.
    All other columns — the question, metadata, label — are copied
    verbatim from the source row so no information is fabricated.

    Classes at or above the target size are left entirely untouched.

    Args:
        ds                : DatasetDict with train / validation / test splits.
        label2id          : class → integer id mapping (from label_utils).
        target_size       : desired examples per under-represented class.
                            If None, derived from *strategy*.
        strategy          : "mean" | "median" | "majority" | int.
                            Ignored when *target_size* is set explicitly.
        paraphrase_column : text column whose content is paraphrased.
        model_name        : HuggingFace Seq2Seq model for paraphrasing.
        batch_size        : texts processed per GPU forward pass.
        max_input_length  : tokenizer truncation limit for inputs.
        max_output_length : maximum generated token length.
        num_beams         : beam search width (higher → better quality,
                            lower → faster generation).
        seed              : random seed for source-example sampling.

    Returns:
        DatasetDict with an upsampled training split.
        Synthetic rows carry ``is_augmented = True``; original rows carry
        ``is_augmented = False``.

    Requires:
        pip install transformers sentencepiece
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from tqdm.auto import tqdm
    except ImportError:
        raise ImportError(
            "transformers and sentencepiece are required for "
            "paraphrase_upsampling.\n"
            "Install with: pip install transformers sentencepiece"
        )

    _set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = ds["train"]
    counts = Counter(train_ds[LABEL_COLUMN])
    target = target_size if target_size is not None else _compute_target(counts, strategy)

    print(f"[paraphrase_upsampling] Target size per class : {target}")
    print(f"[paraphrase_upsampling] Loading '{model_name}'...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    def _paraphrase_batch(texts: list[str]) -> list[str]:
        """Run one GPU batch and return decoded paraphrases."""
        prefixed = ["paraphrase: " + t for t in texts]
        inputs = tokenizer(
            prefixed,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_output_length,
                num_beams=num_beams,
                early_stopping=True,
            )
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    all_splits: list[Dataset] = []

    for label_str, count in counts.items():
        subset = train_ds.filter(lambda ex, l=label_str: ex[LABEL_COLUMN] == l)
        all_splits.append(subset)  # original examples always preserved

        if count >= target:
            continue

        num_needed = target - count
        print(
            f"[paraphrase_upsampling]   '{label_str}': {count} → {target} "
            f"(+{num_needed} paraphrases)"
        )

        rng = random.Random(seed)
        source_indices = [rng.randrange(count) for _ in range(num_needed)]
        source_rows = subset.select(source_indices)

        texts = [str(t) if t else "" for t in source_rows[paraphrase_column]]
        paraphrases: list[str] = []

        for i in tqdm(
            range(0, len(texts), batch_size),
            desc=f"  Paraphrasing '{label_str}'",
            leave=False,
        ):
            paraphrases.extend(_paraphrase_batch(texts[i: i + batch_size]))

        # Build synthetic rows: copy all columns, overwrite paraphrase_column
        aug_dict = {col: list(source_rows[col]) for col in source_rows.column_names}
        aug_dict[paraphrase_column] = paraphrases
        aug_dict["is_augmented"] = [True] * num_needed
        all_splits.append(Dataset.from_dict(aug_dict))

    # Backfill is_augmented=False on original splits that predate the flag
    all_splits = _backfill_augmented_flag(all_splits)

    new_train = concatenate_datasets(all_splits).shuffle(seed=seed)
    print(
        f"[paraphrase_upsampling] Train size : {len(train_ds)} → {len(new_train)}"
    )

    return DatasetDict(
        {"train": new_train, "validation": ds["validation"], "test": ds["test"]}
    )
