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
All public functions follow the same interface as src.data.augmentation:
    fn(ds: DatasetDict, label2id: dict, **kwargs) -> DatasetDict

Only the training split is modified; validation and test splits are
always returned unchanged.

Design notes
------------
- ``LABEL_COLUMN`` is intentionally redefined here (rather than imported
  from augmentation.py) to avoid a circular import when augmentation.py
  registers these functions in its registry.  Both files must always use
  the same string value — currently ``"evasion_label"``.
- Synthetic rows are marked with ``is_augmented = True`` so downstream
  analysis can distinguish real from generated examples.
- Per-class target counts (returned by ``_compute_targets``) let each
  strategy balance classes asymmetrically, which is especially useful
  with the "soft" strategy.

Dependencies (not in base requirements.txt)
-------------------------------------------
    pip install sentence-transformers scikit-learn
    # transformers + sentencepiece are already project dependencies
"""

from __future__ import annotations

import random
from collections import Counter
from difflib import SequenceMatcher
from typing import Union

import numpy as np
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets


# ── Shared constants ──────────────────────────────────────────────────────────
# NOTE: Keep in sync with the definition in src.data.augmentation.
LABEL_COLUMN = "evasion_label"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_targets(
    counts: Counter,
    strategy: Union[str, int, float] = "soft",
    down_ratio: float = 0.75,
) -> dict[str, int]:
    """
    Derive a per-class target count from a strategy descriptor.

    Args:
        counts     : Counter mapping label string → current example count.
        strategy   : Balancing strategy. Options:
                     "soft"     → (recommended) downsample majority to
                                  ``down_ratio * max``, upsample minority
                                  to at most 2× its original size.
                     "mean"     → arithmetic mean of class counts.
                     "median"   → median of class counts.
                     "majority" → size of the largest class (no downsampling).
                     int/float  → fixed absolute target for every class.
        down_ratio : Fraction of the majority class to keep when
                     strategy="soft". Ignored for other strategies.

    Returns:
        Dict mapping label string → target int count.
    """
    if strategy == "soft":
        max_count = max(counts.values())
        ceiling = int(max_count * down_ratio)
        targets: dict[str, int] = {}
        for lbl, count in counts.items():
            if count >= ceiling:
                targets[lbl] = ceiling
            else:
                # Upsample by at most 1 paraphrase per original example.
                targets[lbl] = min(count * 2, ceiling)
        return targets

    if strategy == "mean":
        t_val = int(sum(counts.values()) / len(counts))
    elif strategy == "median":
        t_val = int(np.median(list(counts.values())))
    elif strategy == "majority":
        t_val = max(counts.values())
    else:
        t_val = int(strategy)

    return {lbl: t_val for lbl in counts}


def _backfill_augmented_flag(splits: list[Dataset]) -> list[Dataset]:
    """
    Ensure every split in *splits* has an ``is_augmented`` boolean column.

    Splits created before augmentation began receive ``is_augmented = False``
    for every row.

    Args:
        splits : list of Dataset objects, some of which may already
                 carry the ``is_augmented`` column.

    Returns:
        New list where every Dataset has the ``is_augmented`` column.
    """
    if not any("is_augmented" in s.column_names for s in splits):
        return splits  # flag was never added — nothing to backfill

    backfilled: list[Dataset] = []
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
    targets: dict[str, int] | None = None,
    strategy: Union[str, int] = "mean",
    down_ratio: float = 0.75,
    embed_column: str = "interview_answer",
    embed_model_name: str = "all-MiniLM-L6-v2",
    seed: int = 42,
    **kwargs,
) -> DatasetDict:
    """
    Reduce over-represented classes while preserving semantic diversity.

    For each class whose training-set count exceeds its target, we:
      1. Encode every example with a SentenceTransformer.
      2. Cluster the embeddings into ``target`` KMeans groups.
      3. Keep the real example closest to each cluster centroid.

    This guarantees the reduced subset covers the full semantic spread of
    the original class, unlike random undersampling, which may accidentally
    collapse rare sub-topics.  Classes at or below the target are kept intact.

    Args:
        ds               : DatasetDict with train / validation / test splits.
        label2id         : class → integer id mapping (from label_utils).
        targets          : per-class target counts produced by
                           ``_compute_targets``. When provided, *strategy*
                           and *down_ratio* are ignored.
        strategy         : "soft" | "mean" | "median" | "majority" | int.
                           Used only when *targets* is None.
        down_ratio       : fraction of majority class to keep (soft strategy).
        embed_column     : text column used to compute sentence embeddings.
        embed_model_name : SentenceTransformer model identifier.
        seed             : random seed forwarded to KMeans.

    Returns:
        DatasetDict with a downsampled training split; val/test unchanged.

    Requires:
        pip install sentence-transformers scikit-learn
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances_argmin_min
    except ImportError:
        raise ImportError(
            "sentence-transformers and scikit-learn are required.\n"
            "Install with: pip install sentence-transformers scikit-learn"
        )

    _set_seed(seed)
    train_ds = ds["train"]
    counts = Counter(train_ds[LABEL_COLUMN])

    if targets is None:
        targets = _compute_targets(counts, strategy=strategy, down_ratio=down_ratio)

    print(f"\n[semantic_downsampling] Initializing '{embed_model_name}' ...")
    embed_model = SentenceTransformer(embed_model_name)
    kept_splits: list[Dataset] = []

    for label_str, count in counts.items():
        subset = train_ds.filter(lambda ex, l=label_str: ex[LABEL_COLUMN] == l)
        target = targets.get(label_str, count)

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
        f"[semantic_downsampling] Train size: {len(train_ds)} → {len(new_train)}"
    )

    # Free GPU memory before returning.
    del embed_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return DatasetDict(
        {"train": new_train, "validation": ds["validation"], "test": ds["test"]}
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Paraphrase Upsampling
# ─────────────────────────────────────────────────────────────────────────────

def paraphrase_upsampling(
    ds: DatasetDict,
    label2id: dict,
    targets: dict[str, int] | None = None,
    strategy: Union[str, int] = "mean",
    down_ratio: float = 0.75,
    paraphrase_column: str = "interview_answer",
    model_name: str = "humarin/chatgpt_paraphraser_on_T5_base",
    batch_size: int = 12,
    max_input_length: int = 512,
    max_output_length: int = 256,
    num_beams: int = 5,
    seed: int = 42,
    # ── quality-control parameters ────────────────────────────────────────────
    max_multiplier: float = 2.0,
    min_diversity_ratio: float = 0.15,
    min_semantic_similarity: float = 0.60,
    max_semantic_similarity: float = 0.92,
    use_semantic_filter: bool = True,
    semantic_model_name: str = "all-MiniLM-L6-v2",
    max_retries_per_sample: int = 3,
    **kwargs,
) -> DatasetDict:
    """
    Grow under-represented classes using high-quality T5-based paraphrases.

    Quality-control pipeline
    ─────────────────────────
    1. **Capped upsampling** — each class grows by at most
       ``count * max_multiplier`` examples, preventing extreme oversampling
       of very rare classes.

    2. **Cyclic source sampling** — source examples are cycled evenly so no
       single example is paraphrased more than ``ceil(needed / count)`` times,
       minimising representation collapse.

    3. **Surface-level diversity filter** — paraphrases whose character-level
       similarity to the original exceeds ``1 - min_diversity_ratio`` are
       rejected as near-duplicates.

    4. **Semantic similarity filter** — optionally uses a small sentence-
       transformer to keep only paraphrases that remain semantically faithful
       to the original (within ``[min_semantic_similarity, max_semantic_similarity]``),
       guarding against label drift.

    5. **Retry logic** — up to *max_retries_per_sample* alternative sources
       are tried before the best available candidate is accepted.

    Args:
        ds                      : DatasetDict with train / validation / test splits.
        label2id                : class → integer id mapping (from label_utils).
        targets                 : per-class target counts produced by
                                  ``_compute_targets``. When provided, *strategy*
                                  and *down_ratio* are ignored.
        strategy                : "soft" | "mean" | "median" | "majority" | int.
                                  Used only when *targets* is None.
        down_ratio              : fraction of majority class to keep (soft strategy).
        paraphrase_column       : text column whose content is paraphrased.
        model_name              : HuggingFace Seq2Seq paraphrase model.
        batch_size              : texts per GPU forward pass.
        max_input_length        : tokenizer truncation limit for inputs.
        max_output_length       : maximum generated token length.
        num_beams               : beam search width.
        seed                    : random seed for source-example sampling.
        max_multiplier          : hard cap on class growth (never exceed
                                  ``count * max_multiplier``).
        min_diversity_ratio     : minimum required surface dissimilarity.
        min_semantic_similarity : lower cosine-similarity bound.
        max_semantic_similarity : upper cosine-similarity bound.
        use_semantic_filter     : whether to apply the sentence-transformer filter.
        semantic_model_name     : sentence-transformers model for scoring.
        max_retries_per_sample  : retries before accepting best available.

    Returns:
        DatasetDict with an upsampled training split.
        Synthetic rows carry ``is_augmented = True``; originals carry ``False``.

    Requires:
        pip install transformers sentencepiece sentence-transformers
    """
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from tqdm.auto import tqdm
    except ImportError:
        raise ImportError(
            "transformers and sentencepiece are required.\n"
            "Install with: pip install transformers sentencepiece"
        )

    if use_semantic_filter:
        try:
            from sentence_transformers import SentenceTransformer
            from sentence_transformers import util as st_util
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for the semantic filter.\n"
                "Install with: pip install sentence-transformers\n"
                "Or pass use_semantic_filter=False to skip it."
            )

    _set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = ds["train"]
    counts = Counter(train_ds[LABEL_COLUMN])

    if targets is None:
        targets = _compute_targets(counts, strategy=strategy, down_ratio=down_ratio)

    print(f"\n[paraphrase_upsampling] Loading paraphrase model '{model_name}' ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    para_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    para_model.eval()

    sem_model = None
    if use_semantic_filter:
        print(f"[paraphrase_upsampling] Loading semantic model  '{semantic_model_name}' ...")
        sem_model = SentenceTransformer(semantic_model_name)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _paraphrase_batch(texts: list[str]) -> list[str]:
        prefixed = ["paraphrase: " + t for t in texts]
        inputs = tokenizer(
            prefixed,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(device)
        with torch.no_grad():
            outputs = para_model.generate(
                **inputs,
                max_length=max_output_length,
                num_beams=num_beams,
                early_stopping=True,
            )
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def _surface_similarity(a: str, b: str) -> float:
        """Character-level similarity in [0, 1]."""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def _semantic_similarity(a: str, b: str) -> float:
        """Cosine similarity between sentence embeddings in [0, 1]."""
        embs = sem_model.encode([a, b], convert_to_tensor=True)
        return float(st_util.cos_sim(embs[0], embs[1]))

    def _is_acceptable(original: str, paraphrase: str) -> bool:
        """Return True if the paraphrase is diverse yet semantically faithful."""
        if _surface_similarity(original, paraphrase) > (1.0 - min_diversity_ratio):
            return False  # near-duplicate on the surface
        if sem_model is not None:
            sem_sim = _semantic_similarity(original, paraphrase)
            if not (min_semantic_similarity <= sem_sim <= max_semantic_similarity):
                return False  # semantically too distant or too close
        return True

    def _cyclic_indices(count: int, num_needed: int, rng: random.Random) -> list[int]:
        """
        Return *num_needed* source indices by cycling through [0, count) evenly.
        Each index appears at most ``ceil(num_needed / count)`` times.
        """
        full_cycles = num_needed // count
        remainder = num_needed % count
        indices = list(range(count)) * full_cycles
        indices += rng.sample(range(count), remainder)
        rng.shuffle(indices)
        return indices

    # ── main loop ─────────────────────────────────────────────────────────────

    all_splits: list[Dataset] = []

    for label_str, count in counts.items():
        subset = train_ds.filter(lambda ex, l=label_str: ex[LABEL_COLUMN] == l)
        all_splits.append(subset)  # original examples always preserved

        raw_target = targets.get(label_str, count)
        # Hard cap: never exceed count * max_multiplier regardless of target.
        effective_target = min(raw_target, int(count * max_multiplier))

        if count >= effective_target:
            continue

        num_needed = effective_target - count
        print(
            f"[paraphrase_upsampling]   '{label_str}': {count} → {effective_target} "
            f"(+{num_needed} paraphrases, cap={int(count * max_multiplier)})"
        )

        rng = random.Random(seed)
        source_indices = _cyclic_indices(count, num_needed, rng)
        source_rows = subset.select(source_indices)
        originals = [str(t) if t else "" for t in source_rows[paraphrase_column]]

        # ── first-pass generation ─────────────────────────────────────────────
        generated: list[str] = []
        for i in tqdm(
            range(0, len(originals), batch_size),
            desc=f"  Paraphrasing '{label_str}'",
            leave=False,
        ):
            generated.extend(_paraphrase_batch(originals[i : i + batch_size]))

        # ── quality filtering with retry ──────────────────────────────────────
        accepted_paraphrases: list[str] = []
        accepted_row_indices: list[int] = []

        for idx, (orig, para) in enumerate(zip(originals, generated)):
            if _is_acceptable(orig, para):
                accepted_paraphrases.append(para)
                accepted_row_indices.append(source_indices[idx])
                continue

            # Retry with different source examples.
            best_para = para
            best_source = source_indices[idx]
            for _ in range(max_retries_per_sample):
                retry_idx = rng.randrange(count)
                retry_orig = str(subset[retry_idx][paraphrase_column] or "")
                retry_para = _paraphrase_batch([retry_orig])[0]
                if _is_acceptable(retry_orig, retry_para):
                    best_para = retry_para
                    best_source = retry_idx
                    break
            # Accept best available even when filters cannot be fully satisfied.
            accepted_paraphrases.append(best_para)
            accepted_row_indices.append(best_source)

        final_source_rows = subset.select(accepted_row_indices)
        aug_dict = {col: list(final_source_rows[col]) for col in final_source_rows.column_names}
        aug_dict[paraphrase_column] = accepted_paraphrases
        aug_dict["is_augmented"] = [True] * len(accepted_paraphrases)
        all_splits.append(Dataset.from_dict(aug_dict))

    all_splits = _backfill_augmented_flag(all_splits)
    new_train = concatenate_datasets(all_splits).shuffle(seed=seed)

    # ── summary ───────────────────────────────────────────────────────────────
    new_counts = Counter(new_train[LABEL_COLUMN])
    print(f"\n[paraphrase_upsampling] Train size: {len(train_ds)} → {len(new_train)}")
    print("[paraphrase_upsampling] Final distribution:")
    for lbl, cnt in sorted(new_counts.items()):
        orig = counts[lbl]
        aug = cnt - orig
        print(f"  {lbl:<30}: {orig:>4} + {aug:>4} synthetic = {cnt:>4}")

    # Free GPU memory before returning.
    del para_model, tokenizer
    if sem_model is not None:
        del sem_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return DatasetDict(
        {"train": new_train, "validation": ds["validation"], "test": ds["test"]}
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Smart Resampling (Combined Pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def smart_resampling(
    ds: DatasetDict,
    label2id: dict,
    strategy: Union[str, int] = "soft",
    down_ratio: float = 0.75,
    embed_column: str = "interview_answer",
    paraphrase_column: str = "interview_answer",
    seed: int = 42,
    **kwargs,
) -> DatasetDict:
    """
    One-shot rebalancing: semantic downsampling + paraphrase upsampling.

    Computes per-class targets once and passes them to both sub-routines so
    the two steps are consistent.  Using ``strategy="soft"`` (default) is
    recommended — it caps upsampling at 2× the original class size, keeping
    data quality high and avoiding overfitting.

    Args:
        ds                : DatasetDict with train / validation / test splits.
        label2id          : class → integer id mapping (from label_utils).
        strategy          : "soft" | "mean" | "median" | "majority" | int.
        down_ratio        : fraction of majority class to keep (soft strategy).
        embed_column      : text column for sentence embeddings (downsampling).
        paraphrase_column : text column to paraphrase (upsampling).
        seed              : master random seed forwarded to both sub-routines.
        **kwargs          : forwarded to both ``semantic_downsampling`` and
                            ``paraphrase_upsampling`` (e.g. ``max_multiplier``,
                            ``use_semantic_filter``).

    Returns:
        DatasetDict with a rebalanced training split; val/test unchanged.
    """
    counts = Counter(ds["train"][LABEL_COLUMN])
    targets = _compute_targets(counts, strategy=strategy, down_ratio=down_ratio)

    print(f"\n[smart_resampling] Strategy: '{strategy}'")
    print("[smart_resampling] Target distribution:")
    for lbl, cnt in counts.items():
        print(f"  {lbl}: {cnt} → {targets[lbl]}")

    result_ds = ds

    if any(counts[lbl] > targets[lbl] for lbl in counts):
        result_ds = semantic_downsampling(
            result_ds,
            label2id,
            targets=targets,
            embed_column=embed_column,
            seed=seed,
            **kwargs,
        )

    if any(counts[lbl] < targets[lbl] for lbl in counts):
        result_ds = paraphrase_upsampling(
            result_ds,
            label2id,
            targets=targets,
            paraphrase_column=paraphrase_column,
            seed=seed,
            **kwargs,
        )

    return result_ds
