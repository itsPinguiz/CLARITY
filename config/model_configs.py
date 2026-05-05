"""
Central registry of encoder models to benchmark on the QEvasion task.

Each entry fully specifies how to instantiate and train that model.
To add a new model, add a single dict here — nothing else needs to change.

Fields:
    model_id        : HuggingFace model identifier
    max_length      : max tokenizer length (in tokens)
    batch_size      : per-device train batch size
    eval_batch_size : per-device eval batch size (can be larger)
    grad_accum      : gradient accumulation steps  (effective_bs = batch_size * grad_accum)
    token_type_ids  : whether the model uses token_type_ids (keep in tokenized dataset)
    fp16_compatible : False only for models that break under fp16
    notes           : free-text description / gotchas
"""

MODEL_CONFIGS: dict = {

    # ── Longformer ────────────────────────────────────────────────────────────
    # Used in the original notebook. Supports up to 4096 tokens — ideal for
    # long QA pairs that would be truncated by 512-token models.
    "longformer-base": {
        "model_id":        "allenai/longformer-base-4096",
        "max_length":      1024,
        "batch_size":      4,
        "eval_batch_size": 4,
        "grad_accum":      4,   # effective batch = 16
        "token_type_ids":  False,
        "fp16_compatible": True,
        "learning_rate": 3e-5,
        "notes":           "Baseline from original notebook. Best for long inputs.",
    },

    # ── DeBERTa-v3 base ───────────────────────────────────────────────────────
    "deberta-v3-base": {
        "model_id":        "microsoft/deberta-v3-base",
        "max_length":      512,
        "batch_size":      16,
        "eval_batch_size": 32,
        "grad_accum":      1,
        "token_type_ids":  False,
        "fp16_compatible": False,
        "learning_rate": 1e-5,
        "notes":           "Paper baseline. Strong on short/medium inputs (<=512 tok).",
    },

    # ── RoBERTa base ──────────────────────────────────────────────────────────
    "roberta-base": {
        "model_id":        "roberta-base",
        "max_length":      512,
        "batch_size":      16,
        "eval_batch_size": 32,
        "grad_accum":      1,
        "token_type_ids":  False,
        "fp16_compatible": True,
        "learning_rate": 2e-5,
        "notes":           "Paper baseline. Robustly pre-trained BERT variant.",
    },

    # ── XLNet base ────────────────────────────────────────────────────────────
    "xlnet-base": {
        "model_id":        "xlnet-base-cased",
        "max_length":      512,
        "batch_size":      16,
        "eval_batch_size": 32,
        "grad_accum":      1,
        "token_type_ids":  True,   # XLNet uses token_type_ids
        "fp16_compatible": True,
        "learning_rate": 2e-5,
        "notes":           "Paper baseline. Autoregressive; needs token_type_ids.",
    },

    # ── BERT base (reference) ─────────────────────────────────────────────────
    "bert-base-uncased": {
        "model_id":        "bert-base-uncased",
        "max_length":      512,
        "batch_size":      16,
        "eval_batch_size": 32,
        "grad_accum":      1,
        "token_type_ids":  True,
        "fp16_compatible": True,
        "learning_rate": 3e-5,
        "notes":           "Classic reference model.",
    },

    # ── Longformer large (optional — needs more VRAM) ─────────────────────────
    "longformer-large": {
        "model_id":        "allenai/longformer-large-4096",
        "max_length":      1024,
        "batch_size":      2,
        "eval_batch_size": 2,
        "grad_accum":      8,   # effective batch = 16
        "token_type_ids":  False,
        "fp16_compatible": True,
        "learning_rate": 1e-5,
        "notes":           "Larger Longformer. Needs >=24 GB VRAM or grad_checkpointing.",
    },
    # ── DistilBERT base (fast) ────────────────────────────────────────────────
    "distilbert-base-uncased": {
        "model_id":        "distilbert-base-uncased",
        "max_length":      512,
        "batch_size":      32,  # Model is smaller, can handle larger batches
        "eval_batch_size": 64,
        "grad_accum":      1,
        "token_type_ids":  False, # DistilBERT doesn't use token_type_ids
        "fp16_compatible": True,
        "learning_rate": 5e-5,
        "notes":           "Distilled BERT. Very fast training and inference, slight drop in accuracy.",
    },

    # ── ALBERT base v2 (memory efficient) ─────────────────────────────────────
    "albert-base-v2": {
        "model_id":        "albert-base-v2",
        "max_length":      512,
        "batch_size":      32,  # Parameter sharing allows for larger batches
        "eval_batch_size": 64,
        "grad_accum":      1,
        "token_type_ids":  True,  # ALBERT uses token_type_ids
        "fp16_compatible": True,
        "learning_rate": 3e-5,
        "notes":           "A Lite BERT. Shares parameters across layers. Extremely memory efficient.",
    }
}


def get_model_config(name: str) -> dict:
    """Return the config dict for *name*, raising a clear error if not found."""
    if name not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise KeyError(
            f"Model '{name}' not found in MODEL_CONFIGS.\n"
            f"Available models: {available}"
        )
    return MODEL_CONFIGS[name]
