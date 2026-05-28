"""
Evaluation metrics for the QEvasion classification task.

The primary metric is Macro F1 (average F1 across all classes,
regardless of class frequency) to match both the paper baseline
and the convention used in the original notebook.

Additional metrics (accuracy, weighted F1, per-class F1) are also
computed and included in the returned dict so that results_utils
can store the full picture.
"""

import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
)


def build_compute_metrics_fn(id2label: dict | None = None, evasion_to_clarity: dict | None = None):
    """
    HuggingFace Trainer-compatible metric factory.

    Filters out -100 labels (unmapped examples) before computing
    any metric, so missing labels never pollute the scores.

    If id2label and evasion_to_clarity mappings are provided, it also evaluates
    the macro-level 'clarity' task simultaneously.

    Args:
        id2label: mapping from fine-grained class id → string label name.
        evasion_to_clarity: mapping from fine-grained label → macro clarity label.

    Returns:
        A compute_metrics function suitable for the Trainer.
    """
    label2id_macro = {}
    if evasion_to_clarity is not None:
        unique_macros = sorted(list(set(evasion_to_clarity.values())))
        label2id_macro = {lbl: i for i, lbl in enumerate(unique_macros)}

    def compute_metrics(eval_pred) -> dict:
        logits, labels = eval_pred

        predictions = np.argmax(logits, axis=-1)

        # Ignore padding / unmapped labels
        valid_mask = labels != -100
        valid_labels = labels[valid_mask]
        valid_preds = predictions[valid_mask]

        metrics = {
            "macro_f1":    f1_score(valid_labels, valid_preds, average="macro", zero_division=0),
            "weighted_f1": f1_score(valid_labels, valid_preds, average="weighted", zero_division=0),
            "accuracy":    accuracy_score(valid_labels, valid_preds),
        }

        # Simultaneous CLARITY task evaluation
        if id2label and evasion_to_clarity:
            macro_preds = [label2id_macro[evasion_to_clarity[id2label[int(p)]]] for p in valid_preds]
            macro_labels = [label2id_macro[evasion_to_clarity[id2label[int(l)]]] for l in valid_labels]

            metrics["clarity_macro_f1"] = f1_score(macro_labels, macro_preds, average="macro", zero_division=0)
            metrics["clarity_weighted_f1"] = f1_score(macro_labels, macro_preds, average="weighted", zero_division=0)
            metrics["clarity_accuracy"] = accuracy_score(macro_labels, macro_preds)

        return metrics

    return compute_metrics


# Default fallback without simultaneous evaluation
compute_metrics = build_compute_metrics_fn()


def compute_detailed_report(
    logits: np.ndarray,
    labels: np.ndarray,
    id2label: dict,
    evasion_to_clarity: dict | None = None,
) -> str:
    """
    Return a full sklearn classification_report string.

    Useful for printing at the end of a training run or when
    evaluating on the test set.

    Args:
        logits   : raw model output, shape (N, num_classes).
        labels   : integer ground-truth labels, shape (N,).
        id2label : mapping from class id → human-readable label name.
        evasion_to_clarity: optional mapping to simultaneously print macro-level report.

    Returns:
        Multi-line string ready for print().
    """
    predictions = np.argmax(logits, axis=-1)

    valid_mask = labels != -100
    valid_labels = labels[valid_mask]
    valid_preds = predictions[valid_mask]

    target_names = [id2label[i] for i in sorted(id2label.keys())]

    report = classification_report(
        valid_labels, valid_preds, target_names=target_names, zero_division=0
    )

    if evasion_to_clarity:
        unique_macros = sorted(list(set(evasion_to_clarity.values())))
        label2id_macro = {lbl: i for i, lbl in enumerate(unique_macros)}

        macro_preds = [label2id_macro[evasion_to_clarity[id2label[int(p)]]] for p in valid_preds]
        macro_labels = [label2id_macro[evasion_to_clarity[id2label[int(l)]]] for l in valid_labels]

        report += "\n\n" + "═" * 60 + "\n"
        report += " CLARITY TASK (Mapped Macro-categories)\n"
        report += "═" * 60 + "\n\n"
        report += classification_report(
            macro_labels, macro_preds, target_names=unique_macros, zero_division=0
        )

    return report