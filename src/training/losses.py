"""
Loss functions for the QEvasion classification task.

Current implementations
-----------------------
- FocalLoss          : down-weights easy examples; good for class imbalance.
- WeightedCELoss     : standard cross-entropy with per-class weights.
- LabelSmoothingLoss : applies label smoothing; reduces overconfidence.

All classes follow the same interface:
    loss_fn(logits: Tensor, targets: Tensor) -> scalar Tensor

The ignore_index convention (-100) is respected by every implementation so
that un-mapped labels are silently skipped.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for dense classification (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha        : per-class weight tensor of shape (num_classes,).
                       Pass None to disable class weighting.
        gamma        : focusing parameter. 0 → standard weighted CE.
        ignore_index : label value to skip (default -100, PyTorch convention).
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Filter out ignored labels
        valid_mask = targets != self.ignore_index
        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]

        if valid_targets.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # 2. Standard per-example cross-entropy
        ce_loss = F.cross_entropy(valid_logits, valid_targets, reduction="none")

        # 3. Probability of the true class: p_t = exp(-CE)
        pt = torch.exp(-ce_loss)

        # 4. Focal modulation
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # 5. Per-class alpha weighting
        if self.alpha is not None:
            alpha = self.alpha.to(valid_targets.device)
            alpha_t = alpha[valid_targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


class WeightedCELoss(nn.Module):
    """
    Weighted cross-entropy loss — useful as an ablation against FocalLoss.

    Args:
        weight       : per-class weight tensor of shape (num_classes,).
        ignore_index : label value to skip.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(logits.device) if self.weight is not None else None
        return F.cross_entropy(
            logits,
            targets,
            weight=weight,
            ignore_index=self.ignore_index,
        )


class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with label smoothing.

    Soft targets: y_smooth = (1 - epsilon) * y_hard + epsilon / num_classes

    Args:
        num_classes  : number of output classes.
        epsilon      : smoothing factor (typically 0.05–0.15).
        ignore_index : label value to skip.
    """

    def __init__(
        self,
        num_classes: int,
        epsilon: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid_mask = targets != self.ignore_index
        valid_logits = logits[valid_mask]
        valid_targets = targets[valid_mask]

        if valid_targets.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        log_probs = F.log_softmax(valid_logits, dim=-1)

        # Hard-target NLL loss
        nll_loss = F.nll_loss(log_probs, valid_targets, reduction="none")

        # Uniform distribution KL term
        smooth_loss = -log_probs.mean(dim=-1)

        loss = (1 - self.epsilon) * nll_loss + self.epsilon * smooth_loss
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_loss_fn(
    name: str,
    alpha: torch.Tensor | None = None,
    gamma: float = 2.0,
    num_classes: int | None = None,
    epsilon: float = 0.1,
) -> nn.Module:
    """
    Instantiate and return a loss function by name.

    Args:
        name        : "focal" | "weighted_ce" | "label_smoothing"
        alpha       : class weight tensor (used by focal and weighted_ce).
        gamma       : focal loss focusing parameter.
        num_classes : required for label_smoothing.
        epsilon     : label smoothing factor.
    """
    name = name.lower()
    if name == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif name == "weighted_ce":
        return WeightedCELoss(weight=alpha)
    elif name == "label_smoothing":
        if num_classes is None:
            raise ValueError("num_classes is required for label_smoothing loss.")
        return LabelSmoothingLoss(num_classes=num_classes, epsilon=epsilon)
    else:
        raise KeyError(
            f"Loss '{name}' not recognised. "
            f"Choose from: focal, weighted_ce, label_smoothing"
        )
