"""
Derive edited checkpoint paths from the same hyperparameters used by editors when saving metrics.

Layouts (see ``editors/*.py`` + ``run_edit.sh``):

- ``checkpoints/<stem>.pt`` where stem is ``metrics_<basename>.json`` with the ``metrics_`` prefix stripped.
- Default output folder: ``<root>/<method>/<dataset>/<model>``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import argparse


def default_output_folder_dir(method: str, dataset: str, model: str, root: Optional[str] = None) -> str:
    base = root or "/home/model_editing/data/editing_pipelines"
    return os.path.abspath(os.path.join(base, method, dataset, model))


def metrics_filename_stem_to_checkpoint_basename(stem_without_json: str) -> str:
    """``metrics_edit_foo_bar`` -> ``edit_foo_bar.pt`` (parity with ``BaseEditor._edited_checkpoint_filename``)."""
    if stem_without_json.startswith("metrics_"):
        body = stem_without_json[len("metrics_") :]
    else:
        body = stem_without_json
    return f"{body}.pt"


def _lam_slug(lambda_reg: float) -> str:
    x = float(lambda_reg)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    text = ("%g" % x)
    return text


def build_leastsquares_metrics_stem(config: dict) -> str:
    """Matches ``LeastSquaresEditor`` metrics filename stem (without ``.json``)."""
    ls = config["pipeline_params"].get("leastsquares", {})
    strategy_name = str(ls.get("strategy_mode", "unknown"))
    base_name = ls.get("metrics_save_name", "metrics_edit")
    if isinstance(base_name, str) and base_name.endswith(".json"):
        base_name = base_name[:-5]
    lambda_reg = float(ls.get("lambda_reg", 0.01))

    deg_mode = ls.get("degree_filter", None)
    deg_frac = ls.get("degree_fraction", None)
    uses_pr_or_divrank = "_pr" in strategy_name or "_divrank" in strategy_name

    suffixes: list[Any] = [strategy_name, f"lam{_lam_slug(lambda_reg)}"]
    if deg_mode is not None:
        df = deg_frac if deg_frac is not None else 0.5
        suffixes.append(f"deg{deg_mode}{int(float(df) * 100)}")
    top_frac = ls.get("top_fraction")
    if top_frac is not None:
        suffixes.append(f"top{top_frac}")
    if uses_pr_or_divrank:
        suffixes.append(f"pra{ls.get('pr_alpha', 0.85)}")
        suffixes.append(f"rmx{ls.get('rank_mix_tau', 0.5)}")
    if "retention" in strategy_name:
        gamma_val = ls.get("gamma_retain", 0.0)
        suffixes.append(f"gamma{gamma_val}")

    num_layers = config.get("pipeline_params", {}).get("architecture", {}).get("num_layers")
    if num_layers is not None:
        suffixes.append(f"layers{num_layers}")

    seed_src = ls.get("seed")
    if seed_src is None:
        seed_src = config.get("management", {}).get("seed", 0)
    suffixes.append(f"seed{seed_src}")

    joint = "_".join(str(s) for s in suffixes)
    return f"{base_name}_{joint}"


def build_finetune_metrics_stem(config: dict) -> str:
    """Matches ``FinetuneEditor`` metrics stem."""
    ft = config["pipeline_params"].get("finetune", {})
    lr = ft.get("lr", 1e-3)
    num_epochs = ft.get("num_epochs", 50)
    ls = config["pipeline_params"].get("leastsquares", {})
    top_frac = float(ft.get("top_fraction", ls.get("top_fraction", 0.25)))
    num_layers = config.get("pipeline_params", {}).get("architecture", {}).get("num_layers")
    seed = ft.get("seed")
    if seed is None:
        seed = config.get("management", {}).get("seed", 0)
    layers_seg = f"_layers{num_layers}" if num_layers is not None else ""
    return f"metrics_finetune_lr{lr}_ep{num_epochs}_top{top_frac}{layers_seg}_seed{seed}"


def build_egnn_metrics_stem(config: dict, num_edit_targets: int) -> str:
    seed = config.get("management", {}).get("seed", 0)
    return f"metrics_egnn_targets{int(num_edit_targets)}_seed{seed}"


def build_seed_gnn_metrics_stem(config: dict, num_edit_targets: int) -> str:
    pp = config["pipeline_params"]
    alpha = float(pp.get("alpha", 0.5))
    beta = int(pp.get("beta", 10))
    seed = config.get("management", {}).get("seed", 0)
    return f"metrics_seed_gnn_alpha{alpha}_beta{beta}_targets{int(num_edit_targets)}_seed{seed}"


def resolve_derived_checkpoint_path(
    *,
    config: dict,
    args: "argparse.Namespace",
    output_dir: Optional[str] = None,
    output_root: Optional[str] = None,
) -> Path:
    """
    Resolve ``…/checkpoints/<derived>.pt`` from ``args.method`` and ``config``.
    EGNN / SEED-GNN filenames use the number of selected edit targets; override with
    ``args.edited_targets`` when it differs from ``args.num_targets``.
    """
    od = output_dir or default_output_folder_dir(
        getattr(args, "method", "leastsquares"),
        getattr(args, "dataset", "pokec"),
        getattr(args, "model", "GCN_MLP"),
        root=output_root,
    )
    method = getattr(args, "method", "leastsquares")

    if method == "seedgnn":
        method = "seed_gnn"

    stem: str
    if method == "leastsquares":
        stem = build_leastsquares_metrics_stem(config)
    elif method == "finetune":
        stem = build_finetune_metrics_stem(config)
    elif method == "egnn":
        n = getattr(args, "edited_targets", None)
        if n is None:
            n = getattr(args, "num_targets", 1000)
        stem = build_egnn_metrics_stem(config, int(n))
    elif method == "seed_gnn":
        n = getattr(args, "edited_targets", None)
        if n is None:
            n = getattr(args, "num_targets", 1000)
        stem = build_seed_gnn_metrics_stem(config, int(n))
    else:
        raise ValueError(
            f"Cannot derive checkpoint name for method {method!r}. "
            "Use --metrics-json or --edited-checkpoint, or extend edit_checkpoint_resolve."
        )

    basename = metrics_filename_stem_to_checkpoint_basename(stem)
    ckpt = Path(od) / "checkpoints" / basename
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Derived checkpoint not found: {ckpt}\n"
            f"(expected layout from editors; try --metrics-json or --edited-checkpoint if the run differs, "
            "e.g. EGNN/SEED-GNN actual target count vs --num-targets.)"
        )
    return ckpt

