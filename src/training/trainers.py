"""
Custom HuggingFace Trainer variants for the QEvasion task.

Classes
-------
- CustomLossTrainer : generic Trainer subclass that accepts any loss function
                      produced by src.training.losses.get_loss_fn().

Factory
-------
- get_trainer()     : instantiate the right trainer given a loss name,
                      keeping the calling notebook free of boilerplate.
"""

import torch
from transformers import Trainer

from src.training.losses import get_loss_fn, FocalLoss, WeightedCELoss, LabelSmoothingLoss


class CustomLossTrainer(Trainer):
    """
    Trainer that delegates compute_loss to an arbitrary loss function.

    The loss function is expected to have the signature:
        loss_fn(logits: Tensor, labels: Tensor) -> scalar Tensor

    Args:
        *args / **kwargs : forwarded to Trainer.__init__().
        loss_fn          : nn.Module instance from src.training.losses.
    """

    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        if loss_fn is None:
            raise ValueError(
                "CustomLossTrainer requires a 'loss_fn' keyword argument. "
                "Pass an instance from src.training.losses.get_loss_fn()."
            )
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_trainer(
    loss_name: str,
    model,
    training_args,
    train_dataset,
    eval_dataset,
    compute_metrics,
    alpha: torch.Tensor | None = None,
    gamma: float = 2.0,
    num_classes: int | None = None,
    epsilon: float = 0.1,
) -> Trainer:
    """
    Build and return the appropriate Trainer for a given loss name.

    If loss_name == "ce" (standard cross-entropy with no weighting), the
    vanilla HuggingFace Trainer is returned so there is zero overhead.

    For all other loss names, a CustomLossTrainer is returned with the
    loss function pre-configured.

    Args:
        loss_name       : "focal" | "weighted_ce" | "label_smoothing" | "ce"
        model           : HuggingFace model instance.
        training_args   : TrainingArguments instance.
        train_dataset   : tokenized training Dataset.
        eval_dataset    : tokenized validation Dataset.
        compute_metrics : callable produced by src.training.metrics.
        alpha           : per-class weight tensor (ignored for "ce").
        gamma           : focal loss gamma (ignored unless loss_name=="focal").
        num_classes     : required for "label_smoothing".
        epsilon         : label smoothing factor.

    Returns:
        A configured Trainer (or CustomLossTrainer) ready to call .train().
    """
    common_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    if loss_name.lower() == "ce":
        # Standard HuggingFace CE loss — use the vanilla Trainer
        return Trainer(**common_kwargs)

    loss_fn = get_loss_fn(
        name=loss_name,
        alpha=alpha,
        gamma=gamma,
        num_classes=num_classes,
        epsilon=epsilon,
    )

    return CustomLossTrainer(loss_fn=loss_fn, **common_kwargs)
