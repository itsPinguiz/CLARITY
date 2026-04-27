"""
Dataset loading and splitting for the QEvasion task.

The critical guarantee of this module: every call to load_and_split_dataset()
with the same seed returns the IDENTICAL train / validation / test split,
so all models are always evaluated on the same data.
"""

from datasets import load_dataset, DatasetDict


def load_and_split_dataset(
    hf_dataset_name: str = "ailsntua/QEvasion",
    test_size: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> DatasetDict:
    """
    Load the QEvasion dataset and produce a reproducible 3-way split.

    The original dataset ships only with 'train' and 'test' splits.
    We carve 10% of 'train' out as the validation set (same role as in
    the original notebook), keeping the original 'test' untouched for
    final evaluation.

    Args:
        hf_dataset_name : HuggingFace dataset identifier.
        test_size       : fraction of 'train' to use as validation.
        seed            : random seed — DO NOT change between runs.
        verbose         : print split sizes when True.

    Returns:
        DatasetDict with keys 'train', 'validation', 'test'.
    """
    raw_ds = load_dataset(hf_dataset_name)

    split = raw_ds["train"].train_test_split(test_size=test_size, seed=seed)

    ds = DatasetDict(
        {
            "train":      split["train"],
            "validation": split["test"],
            "test":       raw_ds["test"],
        }
    )

    if verbose:
        for split_name, split_ds in ds.items():
            print(f"  {split_name:<12}: {len(split_ds):>5} samples")

    return ds
