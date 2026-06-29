"""
Utilities for saving, loading and comparing experiment results.

Each training run produces a JSON file in the results/ directory named:
    {model_key}__{aug_name}__{loss_name}.json

The compare_results() function reads all JSON files and returns a sorted
Pandas DataFrame for easy inspection inside a notebook.
"""

import json
from datetime import datetime
from pathlib import Path


RESULTS_DIR = Path("/content/drive/MyDrive/progettoLLM/CLARITY/results/encoder/summary")

def save_results(
    model_key: str,
    bal_name: str,
    aug_name: str,
    loss_name: str,
    metrics: dict,
    extra: dict | None = None,
) -> str:
    """
    Persist the metrics from one training run as a JSON file.

    Args:
        model_key : short model identifier (e.g. "deberta-v3-base").
        bal_name  : balancing strategy name.
        aug_name  : augmentation strategy name.
        loss_name : loss function name.
        metrics   : dict returned by compute_metrics or from trainer.evaluate().
        extra     : optional dict of additional metadata (e.g. num_params,
                    training_time, hyperparameters).

    Returns:
        Path to the written JSON file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_name = f"{model_key}__{bal_name}__{aug_name}__{loss_name}"
   
    if extra and extra.get("task") == "clarity":
        run_name = f"c__{run_name}"
    else:
        run_name = f"e__{run_name}"

    filepath = RESULTS_DIR / f"{run_name}.json"

    record = {
        "model_key":  model_key,
        "bal_name":   bal_name,
        "aug_name":   aug_name,
        "loss_name":  loss_name,
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "metrics":    {k: float(v) for k, v in metrics.items()},
    }
    if extra:
        record["extra"] = extra

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    print(f"[results] Saved → {filepath}")
    return str(filepath)


def load_all_results(results_dir: str | Path = RESULTS_DIR) -> list[dict]:
    """
    Load every JSON result file from *results_dir*.

    Returns:
        List of record dicts (one per run), sorted by macro_f1 descending.
    """
    results_dir = Path(results_dir)
    records = []
    for path in sorted(results_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            records.append(json.load(f))

    # Sort by primary metric descending (best first)
    records.sort(
        key=lambda r: r.get("metrics", {}).get("macro_f1", 0.0),
        reverse=True,
    )
    return records


def compare_results(
    results_dir: str | Path = RESULTS_DIR,
    metrics_to_show: list[str] | None = None,
) -> "pd.DataFrame":  # type: ignore[name-defined]
    """
    Build a Pandas DataFrame comparing all saved runs.

    Args:
        results_dir     : directory containing JSON result files.
        metrics_to_show : list of metric keys to include as columns.
                          Defaults to ["macro_f1", "weighted_f1", "accuracy"].

    Returns:
        DataFrame with one row per run, sorted by macro_f1 descending.
        Returns an empty DataFrame if no results are found.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required for compare_results(). pip install pandas")

    if metrics_to_show is None:
        metrics_to_show = ["macro_f1", "weighted_f1", "accuracy"]

    records = load_all_results(results_dir)
    if not records:
        print(f"[results] No result files found in '{results_dir}'.")
        return pd.DataFrame()

    rows = []
    for r in records:
        row = {
            "model":       r["model_key"],
            "task":        r.get("extra", {}).get("task", ""),
            "balancing":   r["bal_name"],
            "augmentation": r["aug_name"],
            "loss":        r["loss_name"],
            "timestamp":   r.get("timestamp", ""),
        }
        for metric in metrics_to_show:
            row[metric] = r.get("metrics", {}).get(metric, float("nan"))
        rows.append(row)

    df = pd.DataFrame(rows)

    # Round metric columns for readability
    metric_cols = [c for c in metrics_to_show if c in df.columns]
    df[metric_cols] = df[metric_cols].round(4)

    return df

def print_comparison_table(results_dir: str | Path = RESULTS_DIR) -> None:
    df = compare_results(
        results_dir,
        metrics_to_show=[
            'test_macro_f1',
            'test_weighted_f1',
            'test_accuracy',
            'val_macro_f1'
        ]
    )

    if df.empty:
        return

    df = df.sort_values(by='test_macro_f1',ascending=False,na_position='last')

    columns_to_show = [
        'model',
        'task',
        'balancing',
        'augmentation',
        'loss',
        'test_macro_f1',
        'test_weighted_f1',
        'test_accuracy',
        'val_macro_f1'
    ]

    df_filtered = df[[c for c in columns_to_show if c in df.columns]]

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table(
            title="Experiment Results (sorted by Test Macro F1 ↓)",
            expand=False,              
            show_lines=False,
            row_styles=["", "dim"]
        )

        for col in df_filtered.columns:
            is_numeric = df_filtered[col].dtype != object

            table.add_column(
                col,
                justify="right" if is_numeric else "left",
                no_wrap=False,
                overflow="fold",
                max_width=30
            )

        for _, row in df_filtered.iterrows():
            values = [
                f"{v:.4f}" if isinstance(v, float) else str(v)
                for v in row
            ]

            task = str(row.get("task", "")).lower()

            if task == "clarity":
                row_style = "green"
            elif task == "evasion":
                row_style = "cyan"
            else:
                row_style = None

            table.add_row(*values, style=row_style)
            
        console.print(table)

    except ImportError:
        print(df_filtered.to_string(index=False, max_colwidth=None))
