"""
Environment detection and adaptive training configuration.

Detects whether the code is running on Google Colab or locally,
then returns TrainingArguments tuned for the available hardware.

The caller never has to worry about fp16, output paths, or batch sizes:
everything is derived automatically from the detected environment and
the per-model config in config/model_configs.py.
"""

import os
from pathlib import Path


def is_colab() -> bool:
    """Return True if running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def get_device_info() -> dict:
    """
    Return a dict with hardware info:
        has_cuda    : bool
        gpu_name    : str or None
        gpu_mem_gb  : float or None   (total VRAM)
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return {"has_cuda": False, "gpu_name": None, "gpu_mem_gb": None}
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return {"has_cuda": True, "gpu_name": gpu_name, "gpu_mem_gb": round(gpu_mem, 1)}
    except Exception:
        return {"has_cuda": False, "gpu_name": None, "gpu_mem_gb": None}


def get_output_dir(model_key: str, bal_name: str, aug_name: str, loss_name: str, task: str = "evasion") -> str:
    """
    Build the output directory path for a specific run.

    On Colab we write to /content/results/ (ephemeral, but can be
    mounted to Google Drive by the user).
    Locally we write to ./results/ relative to the project root.

    Args:
        model_key  : short model identifier (e.g. "deberta-v3-base").
        bal_name   : balancing strategy name.
        aug_name   : augmentation strategy name.
        loss_name  : loss function name.

    Returns:
        Absolute path string.
    """
    run_name = f"{model_key}__{bal_name}__{aug_name}__{loss_name}"

    if task and task == "clarity":
        run_name = f"c__{run_name}"
    else:
        run_name = f"e__{run_name}"

    if is_colab():
        base = Path("/content/drive/MyDrive/progettoLLM/CLARITY/results/encoder/models") / run_name
    else:
        # Resolve relative to wherever the script is executed from
        base = Path("results") / run_name

    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def get_training_args(
    model_key: str,
    model_cfg: dict,
    bal_name: str = "none",
    aug_name: str = "none",
    loss_name: str = "focal",
    num_epochs: int = 5,
    weight_decay: float = 0.01,
    task: str = "evasion",
):
    """
    Build and return a TrainingArguments object adapted to the current
    environment and model configuration.

    fp16 is enabled automatically when a CUDA GPU is present AND the
    model is marked as fp16_compatible.

    gradient_checkpointing is enabled automatically when VRAM < 16 GB
    (conservative threshold) to avoid OOM errors.

    Args:
        model_key    : short model identifier used in directory names.
        model_cfg    : dict from MODEL_CONFIGS[model_key].
        aug_name     : augmentation strategy used in this run.
        loss_name    : loss function used in this run.
        num_epochs   : number of training epochs.
        weight_decay : L2 regularisation weight.

    Returns:
        transformers.TrainingArguments instance.
    """
    from transformers import TrainingArguments

    output_dir = get_output_dir(model_key, bal_name, aug_name, loss_name, task)
    device_info = get_device_info()

    use_fp16 = (
        device_info["has_cuda"]
        and model_cfg.get("fp16_compatible", True)
    )
    lr = model_cfg.get("learning_rate", 2e-5)

    # Enable gradient checkpointing to save VRAM on small GPUs
    mem_gb = device_info.get("gpu_mem_gb") or 99.0
    use_grad_checkpointing = device_info["has_cuda"] and mem_gb < 20.0

    env_label = "Colab" if is_colab() else "local"
    print(
        f"[env_utils] Environment : {env_label}\n"
        f"            GPU         : {device_info.get('gpu_name', 'none')} "
        f"({mem_gb} GB)\n"
        f"            fp16        : {use_fp16}\n"
        f"            grad_ckpt   : {use_grad_checkpointing}\n"
        f"            output_dir  : {output_dir}"
    )

    return TrainingArguments(
        output_dir=output_dir,

        # ── Evaluation & saving ───────────────────────────────────────────────
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        load_best_model_at_end=True,

        # ── Batch sizes (from model config) ───────────────────────────────────
        per_device_train_batch_size=model_cfg["batch_size"],
        per_device_eval_batch_size=model_cfg["eval_batch_size"],
        gradient_accumulation_steps=model_cfg["grad_accum"],

        # ── Hardware optimisations ────────────────────────────────────────────
        fp16=use_fp16,
        gradient_checkpointing=use_grad_checkpointing,

        # ── Optimiser ────────────────────────────────────────────────────────
        learning_rate=lr,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_ratio=0.1,       

        # ── Logging ───────────────────────────────────────────────────────────
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        report_to="none",   # set to "wandb" if you want experiment tracking
    )
