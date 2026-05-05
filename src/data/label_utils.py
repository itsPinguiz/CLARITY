"""
Label utilities for the QEvasion evasion-label classification task.

Responsibilities:
- Build label2id / id2label mappings from the training split.
- Compute per-class inverse-frequency weights (alpha) for FocalLoss.
- Provide the map function that adds integer 'label' columns to a dataset.

All functions are stateless (pure) and operate on HuggingFace Dataset objects,
making them safe to call in any order or environment.
"""

from collections import Counter
from datasets import DatasetDict, Dataset
import torch


# ── Label column used throughout the project ──────────────────────────────────
LABEL_COLUMN = "evasion_label"


def build_label_maps(ds: DatasetDict) -> tuple[dict, dict]:
    """
    Build label2id and id2label from the *training* split of *ds*.

    Labels are sorted alphabetically so the mapping is deterministic
    regardless of iteration order.

    Args:
        ds: DatasetDict that must contain a 'train' key.

    Returns:
        (label2id, id2label) — both are plain Python dicts.
    """
    raw_labels = ds["train"][LABEL_COLUMN]
    unique_labels = sorted(
        {lbl for lbl in raw_labels if lbl is not None and lbl.strip() != ""}
    )

    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    return label2id, id2label


def compute_alpha_weights(
    ds: DatasetDict,
    label2id: dict,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for FocalLoss alpha.

    Formula: weight_c = N / (C * count_c)
    where N = total training samples, C = number of classes.

    Classes absent from the training set get weight 0.0 (they cannot
    contribute to the loss anyway).

    Args:
        ds       : DatasetDict with a 'train' key.
        label2id : mapping from string label → integer id.

    Returns:
        Float32 tensor of shape (num_classes,).
    """
    num_classes = len(label2id)
    train_labels = [
        label2id[lbl]
        for lbl in ds["train"][LABEL_COLUMN]
        if lbl in label2id
    ]

    label_counts = Counter(train_labels)
    total = len(train_labels)

    weights = []
    for class_id in range(num_classes):
        count = label_counts.get(class_id, 0)
        w = total / (num_classes * count) if count > 0 else 0.0
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float32)


def make_add_label_id(label2id: dict):
    """
    Return a HuggingFace map-compatible function that adds an integer
    'label' column to each example.

    Unknown / missing labels are assigned -100 (PyTorch ignore index),
    so they are safely skipped by the loss and metrics.

    Args:
        label2id: mapping from string label → integer id.

    Returns:
        A function suitable for ds.map(fn).
    """
    def add_label_id(example: dict) -> dict:
        label_str = example[LABEL_COLUMN]
        example["label"] = label2id.get(label_str, -100)
        return example

    return add_label_id


def apply_labels(ds: DatasetDict, label2id: dict, verbose: bool = True) -> DatasetDict:
    """
    Apply integer label mapping to all splits of *ds*.

    Args:
        ds       : DatasetDict to transform.
        label2id : mapping produced by build_label_maps().
        verbose  : print per-split label distribution when True.

    Returns:
        New DatasetDict with an added 'label' column.
    """
    add_fn = make_add_label_id(label2id)
    ds = ds.map(add_fn)

    if verbose:
        id2label = {v: k for k, v in label2id.items()}
        for split_name, split_ds in ds.items():
            counts = Counter(split_ds["label"])
            print(f"\n[{split_name}] label distribution:")
            for class_id in sorted(id2label):
                label_name = id2label[class_id]
                print(f"  {label_name:<25} : {counts.get(class_id, 0):>4}")
            if counts.get(-100, 0):
                print(f"  {'<unmapped>':25} : {counts[-100]:>4}  ← check label column!")

    return ds
