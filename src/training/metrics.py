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


def compute_metrics(eval_pred) -> dict:
    """
    HuggingFace Trainer-compatible metric function.

    Filters out -100 labels (unmapped examples) before computing
    any metric, so missing labels never pollute the scores.

    Args:
        eval_pred: EvalPrediction namedtuple with fields
                   (predictions: ndarray, label_ids: ndarray).

    Returns:
        dict with keys:
            macro_f1      – primary metric (used for best-model selection)
            weighted_f1   – weighted average F1
            accuracy      – overall accuracy
    """
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    # Ignore padding / unmapped labels
    valid_mask = labels != -100
    valid_labels = labels[valid_mask]
    valid_preds = predictions[valid_mask]

    macro_f1 = f1_score(valid_labels, valid_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(valid_labels, valid_preds, average="weighted", zero_division=0)
    accuracy = accuracy_score(valid_labels, valid_preds)

    return {
        "macro_f1":    macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy":    accuracy,
    }


def compute_detailed_report(
    logits: np.ndarray,
    labels: np.ndarray,
    id2label: dict,
) -> str:
    """
    Return a full sklearn classification_report string.

    Useful for printing at the end of a training run or when
    evaluating on the test set.

    Args:
        logits   : raw model output, shape (N, num_classes).
        labels   : integer ground-truth labels, shape (N,).
        id2label : mapping from class id → human-readable label name.

    Returns:
        Multi-line string ready for print().
    """
    predictions = np.argmax(logits, axis=-1)

    valid_mask = labels != -100
    valid_labels = labels[valid_mask]
    valid_preds = predictions[valid_mask]

    target_names = [id2label[i] for i in sorted(id2label.keys())]

    return classification_report(
        valid_labels,
        valid_preds,
        target_names=target_names,
        zero_division=0,
    )
