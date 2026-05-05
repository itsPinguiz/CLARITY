"""
Dataset loading and splitting for the QEvasion task.

The critical guarantee of this module: every call to load_and_split_dataset()
with the same seed returns the IDENTICAL train / validation / test split,
so all models are always evaluated on the same data.
"""

from collections import Counter

from datasets import Dataset, DatasetDict, load_dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANNOTATORS = ["annotator1", "annotator2", "annotator3"]

# Maps each fine-grained evasion category to the coarser clarity class used
# as the official label in the dataset.
EVASION_TO_CLARITY: dict[str, str] = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "General": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Dodging": "Ambivalent",
    "Deflection": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_votes(row: dict) -> dict[str, str]:
    """Return a ``{annotator: vote}`` dict, skipping empty / None values."""
    return {
        ann: row[ann]
        for ann in ANNOTATORS
        if row.get(ann) and str(row[ann]).strip() not in ("", "None")
    }


def _compute_majorities_and_reliability(data: list[dict]) -> dict[str, dict[str, int]]:
    """
    Phase 1 - Assign ``evasion_label`` to rows that have a clear majority
    among the three annotators.

    Rows without a majority are flagged with ``_needs_tie_break = True`` and
    their votes are stored under ``_tied_votes`` for Phase 2.

    Additionally, a per-annotator reliability score is computed by counting
    how many times each annotator voted for the majority label (grouped by the
    mapped clarity class).  This score is returned for use in Phase 2.

    The function mutates *data* in-place.

    Args:
        data: List of row dicts (mutable).

    Returns:
        Nested dict  ``{annotator: {clarity_class: count}}``.
    """
    reliability: dict[str, dict[str, int]] = {
        ann: {"Clear Reply": 0, "Ambivalent": 0, "Clear Non-Reply": 0}
        for ann in ANNOTATORS
    }

    for row in data:
        votes = _get_votes(row)
        vote_counts = Counter(votes.values())
        max_votes = max(vote_counts.values())
        top_candidates = [v for v, c in vote_counts.items() if c == max_votes]

        if len(top_candidates) == 1:
            # Clear majority
            majority_evasion = top_candidates[0]
            row["evasion_label"] = majority_evasion
            row["_needs_tie_break"] = False

            mapped_clarity = EVASION_TO_CLARITY.get(majority_evasion, majority_evasion)
            for ann, ann_vote in votes.items():
                if ann_vote == majority_evasion and mapped_clarity in reliability[ann]:
                    reliability[ann][mapped_clarity] += 1
        else:
            # Tie – defer to Phase 2
            row["_needs_tie_break"] = True
            row["_tied_votes"] = votes

    return reliability


def _resolve_ties(data: list[dict], reliability: dict[str, dict[str, int]]) -> None:
    """
    Phase 2 - Break ties for rows flagged by Phase 1.

    Strategy (in priority order):
    1. Keep only votes whose mapped clarity matches the official
       ``clarity_label``; if exactly one such vote exists, use it.
    2. If several consistent votes remain, pick the one cast by the
       annotator with the highest reliability score for that clarity class.
    3. Fallback (no vote is consistent with ``clarity_label``): use the
       first available vote.

    Temporary keys ``_needs_tie_break`` and ``_tied_votes`` are removed from
    every row after processing.

    The function mutates *data* in-place.
    """
    for row in data:
        if row.get("_needs_tie_break"):
            official_clarity = row.get("clarity_label")
            tied_votes = row["_tied_votes"]

            # Filter to votes that are coherent with the official clarity label
            valid_candidates = {
                ann: vote
                for ann, vote in tied_votes.items()
                if EVASION_TO_CLARITY.get(vote, vote) == official_clarity
            }

            if not valid_candidates:
                # Fallback: no coherent vote – take the first available one
                row["evasion_label"] = next(iter(tied_votes.values()))

            elif len(valid_candidates) == 1:
                row["evasion_label"] = next(iter(valid_candidates.values()))

            else:
                # Still tied among coherent votes – use annotator reliability
                best_score = -1
                winning_vote = None
                for ann, vote in valid_candidates.items():
                    score = reliability[ann].get(official_clarity, 0)
                    if score > best_score:
                        best_score = score
                        winning_vote = vote
                row["evasion_label"] = winning_vote

        # Clean up temporary keys regardless of which branch was taken
        row.pop("_needs_tie_break", None)
        row.pop("_tied_votes", None)


def _add_evasion_labels(dataset: Dataset) -> Dataset:
    """
    Run the full two-phase pipeline on *dataset* and return a new
    ``Dataset`` that includes the ``evasion_label`` column.

    This function is intentionally side-effect-free with respect to the
    original ``Dataset`` object (it works on a plain Python list copy).

    Args:
        dataset: A HuggingFace ``Dataset`` (typically the test split).

    Returns:
        A new ``Dataset`` with an additional ``evasion_label`` column.
    """
    data = list(dataset)  # shallow copy – dicts will be mutated
    reliability = _compute_majorities_and_reliability(data)
    _resolve_ties(data, reliability)
    return Dataset.from_list(data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_and_split_dataset(
    hf_dataset_name: str = "ailsntua/QEvasion",
    test_size: float = 0.1,
    seed: int = 42,
    verbose: bool = True,
) -> DatasetDict:
    """
    Load the QEvasion dataset and produce a reproducible 3-way split.

    The original dataset ships only with 'train' and 'test' splits.
    We carve ``test_size`` fraction of 'train' out as the validation set,
    keeping the original 'test' untouched for final evaluation.

    The returned test split is enriched with an ``evasion_label`` column
    derived from majority voting across the three annotators (with
    reliability-weighted tie-breaking).

    Args:
        hf_dataset_name : HuggingFace dataset identifier.
        test_size       : fraction of 'train' to use as validation.
        seed            : random seed — DO NOT change between runs.
        verbose         : print split sizes when True.

    Returns:
        ``DatasetDict`` with keys ``'train'``, ``'validation'``, ``'test'``.
        Only the ``'test'`` split carries the ``evasion_label`` column.
    """
    raw_ds = load_dataset(hf_dataset_name)

    split = raw_ds["train"].train_test_split(test_size=test_size, seed=seed)

    ds = DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
            "test": _add_evasion_labels(raw_ds["test"]),
        }
    )

    if verbose:
        for split_name, split_ds in ds.items():
            print(f"  {split_name:<12}: {len(split_ds):>5} samples")

    return ds
