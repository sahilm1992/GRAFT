#!/usr/bin/env python3
"""
Feature-focused evaluation on pretrained vs edited checkpoints:

  (1) **Mask focused feature** — discrete / low-cardinality columns (≤30 unique values): set column
      to **0**. Continuous (>30 unique, same heuristic as perturb / binning plots): replace with the
      **global column mean**. Run `seed_test`, `compute_full_auc_pr_by_split`, plus mask-shift deltas

  (2) **Perturb focused feature** — ``perturb_feature_and_measure_probs`` for sensitivity, plus
      **mean (and std) accuracy / AUC-PR** over the same input perturbation replicates (pipeline-style
      feature resampling).

Checkpoint pairing matches `BaseEditor.attach_edit_checkpoint_artifacts`:
  metrics_<slug>.json  ->  checkpoints/<slug>.pt

Model loading follows `load_model`.

Sensitive feature names and optional perturbation anchors default to
``editing_pipelines.suite_feature_maps``, which mirrors ``run_editing_suite.sh``. Use
``--legacy-feature-map`` to rely only on ``run_edit.build_config`` (e.g. for ``credit``).

With the same CLI as ``run_edit`` (dataset, model, seed, LS strategy, λ, layers, …) you may omit
both ``--metrics-json`` and ``--edited-checkpoint``; the edited ``.pt`` is resolved under
``--output-root/<method>/<dataset>/<model>/checkpoints/`` via ``edit_checkpoint_resolve`` (same naming
as the editors).

By default, results are written under ``${PATH_TO_DATA}/ablations`` (see ``paths.py`` / ``--ablations-dir``)
with a timestamp in the filename. Use ``--deterministic-ablation-filename`` or environment variable
``FEATURE_ABLATION_DETERMINISTIC=1`` for a stable path that re-runs overwrite.
Pass ``--dataset-dir`` if your graphs live elsewhere (default matches ``run_editing_suite.sh``).
Environment variable ``DATASET_DIR`` overrides the default path.

Example::

  python ablations/feature_ablation_forward.py \\
    --dataset pokec --model GCN_MLP --seed 0 --num-layers 2 \\
    --leastsquares-strategy sens_subspace_retention_pr_graphaware \\
    --lambda-reg 0.1 --top-fraction 0.25 \\
    --metrics-json .../metrics_edit_....json \\
    --pretrain-dir .../edit_ckpts
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import numbers
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple, Union

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from editing_pipelines._ensure_repo_paths import bootstrap

bootstrap()

import paths as _paths

# Mirrors editing_pipelines/run_editing_suite.sh DATASET_DIR; `get_data` expects CSVs here
# (e.g. ``<dataset_dir>/pokec/region_job_2.csv``), not the repo ``seed-gnn`` root.
DEFAULT_DATASET_DIR = os.getenv("DATASET_DIR", str(_paths.dataset_dir_default()))
DEFAULT_PRETRAIN_DIR = os.getenv("PRETRAIN_DIR", str(_paths.pretrain_edit_ckpts_dir_default()))
DEFAULT_OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", str(_paths.editing_pipelines_root_default()))
DEFAULT_ABLATIONS_DIR = Path(os.environ.get("ABLATIONS_DIR", str(_paths.ablations_dir_default())))

from editing_pipelines.run_edit import build_config  # noqa: E402
from editing_pipelines.suite_feature_maps import apply_suite_sensitive_feature_config  # noqa: E402
from editing_pipelines.edit_checkpoint_resolve import (  # noqa: E402
    default_output_folder_dir,
    resolve_derived_checkpoint_path,
)
from editing_pipelines.utils.metrics import compute_full_auc_pr_by_split  # noqa: E402
from editing_pipelines.utils.model_io import load_model  # noqa: E402
from editing_pipelines.utils.results import perturb_feature_and_measure_probs  # noqa: E402


# ---------------------------------------------------------------------------
# Checkpoint resolution (same naming as editors.base.BaseEditor)
# ---------------------------------------------------------------------------


def resolve_edited_checkpoint_from_metrics_json(metrics_json_path: str) -> Path:
    p = Path(metrics_json_path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Metrics JSON not found: {p}")
    stem = p.stem
    prefix = "metrics_"
    if stem.startswith(prefix):
        ckpt_stem = stem[len(prefix) :]
    else:
        ckpt_stem = stem
        logger.warning(
            "Metrics filename %s does not start with %r; using stem %r for .pt.",
            p.name,
            prefix,
            ckpt_stem,
        )
    ckpt = p.parent / "checkpoints" / f"{ckpt_stem}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Edited checkpoint missing (pipeline layout): {ckpt}")
    return ckpt


# ---------------------------------------------------------------------------
# Feature typing (align with mean_prob_by_feature / perturb_feature heuristics)
# ---------------------------------------------------------------------------


def focused_column_is_continuous(whole_data, feat_idx: int) -> bool:
    vals = whole_data.x[:, feat_idx].detach().cpu().numpy().ravel()
    unique = np.unique(vals)
    return len(unique) > 30


def build_masked_x(whole_data, feat_idx: int) -> torch.Tensor:
    """
    Discrete / low-cardinality: set column to 0.
    Continuous (>30 unique): set entire column to global mean.
    """
    x_new = whole_data.x.clone()
    if focused_column_is_continuous(whole_data, feat_idx):
        fill = whole_data.x[:, feat_idx].mean()
        x_new[:, feat_idx] = fill
    else:
        x_new[:, feat_idx] = 0
    return x_new


def clone_data_with_x(whole_data, new_x: torch.Tensor):
    d = whole_data.clone()
    d.x = new_x.clone()
    return d


def resolve_focus_feature_index(whole_data, config: Dict, feature_name: str | None) -> Tuple[int, str]:
    name = feature_name or config.get("pipeline_params", {}).get("sensitive_feature")
    if name is None:
        raise ValueError(
            "Set --feature-name or ensure run_edit fills pipeline_params['sensitive_feature'] for this dataset."
        )
    if not hasattr(whole_data, "feature_names"):
        raise ValueError("whole_data has no feature_names; cannot locate focused feature.")
    names = list(whole_data.feature_names)
    if name not in names:
        raise ValueError(f"Feature {name!r} not found in feature_names (first entries: {names[:15]}).")
    return names.index(str(name)), str(name)


# ---------------------------------------------------------------------------
# Metrics helpers (reuse pipeline-style summaries)
# ---------------------------------------------------------------------------


def _json_safe(o: Any) -> Any:
    if o is None or isinstance(o, (str, bool)):
        return o
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, numbers.Real):
        x = float(o)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if isinstance(o, np.floating):
        return _json_safe(float(o))
    if isinstance(o, np.integer):
        return int(o)
    return str(o)


def _empty_sensitivity_split() -> Dict[str, float]:
    return {
        "mean_var": float("nan"),
        "mean_rel_var": float("nan"),
        "mean_flip_fraction": float("nan"),
    }


def summarize_perturb_sensitivity(val_df, test_df, *, compute_flips: bool) -> Dict[str, Any]:
    """Mirror LeastSquaresEditor-style aggregation over perturb_feature DataFrames."""

    def _one(df) -> Dict[str, float]:
        if df is None or len(df) == 0:
            return _empty_sensitivity_split()
        out = {
            "mean_var": float(df["VarProb"].mean()),
            "mean_rel_var": float(df["RelVarProb"].mean()) if "RelVarProb" in df.columns else float("nan"),
            "mean_flip_fraction": (
                float(df["FlipFraction"].mean())
                if compute_flips and "FlipFraction" in df.columns
                else float("nan")
            ),
        }
        return out

    return {"val": _one(val_df), "test": _one(test_df)}


def _is_regression_data(whole_data) -> bool:
    return bool(
        getattr(whole_data, "task_type", "") == "regression"
        or whole_data.y.dtype.is_floating_point
    )


@torch.no_grad()
def mask_impact_classification(
    model: torch.nn.Module, data_orig, data_masked
) -> Dict[str, float]:
    """Mean |Δ p(true class)| on val / test comparing original vs masked inputs."""
    model.eval()
    lo = prediction(model, data_orig)
    lm = prediction(model, data_masked)
    probs_o = torch.softmax(lo, dim=-1).cpu()
    probs_m = torch.softmax(lm, dim=-1).cpu()
    y = data_orig.y.cpu().long()
    idx_y = torch.arange(probs_o.size(0))
    p_o = probs_o[idx_y, y]
    p_m = probs_m[idx_y, y]
    delta = (p_o - p_m).abs()

    def _mean_on(mask) -> float:
        if mask is None:
            return float("nan")
        m = mask.cpu().view(-1)
        if m.sum() == 0:
            return float("nan")
        return float(delta[m].mean().item())

    return {
        "mean_abs_prob_true_delta_val": _mean_on(data_orig.val_mask),
        "mean_abs_prob_true_delta_test": _mean_on(data_orig.test_mask),
    }


@torch.no_grad()
def mask_impact_regression(model: torch.nn.Module, data_orig, data_masked) -> Dict[str, float]:
    model.eval()
    po = prediction(model, data_orig)
    pm = prediction(model, data_masked)
    if po.dim() == 2 and po.size(-1) == 1:
        po = po.squeeze(-1)
        pm = pm.squeeze(-1)
    d = (po - pm).abs()

    def _mean_on(mask):
        if mask is None:
            return float("nan")
        m = mask.to(d.device).view(-1)
        if m.sum() == 0:
            return float("nan")
        return float(d[m].mean().item())

    return {
        "mean_abs_prediction_delta_val": _mean_on(data_orig.val_mask),
        "mean_abs_prediction_delta_test": _mean_on(data_orig.test_mask),
    }


def eval_on_data(model, whole_variant) -> MutableMapping[str, Any]:
    is_reg = _is_regression_data(whole_variant)
    ev = seed_test(model, whole_variant)
    auc = compute_full_auc_pr_by_split(model, whole_variant)
    payload: Dict[str, Any] = {
        "overall": list(ev["overall"]),
        "metrics_splits": {},
        "auc_pr": auc,
        "task": "regression" if is_reg else "classification",
    }
    for split in ("train", "val", "test"):
        metrics_split = dict(ev["metrics"][split])
        if not is_reg and isinstance(auc.get(split), numbers.Real):
            metrics_split["auc_pr"] = float(auc[split])
        payload["metrics_splits"][split] = metrics_split
    return payload


def build_pipeline_style_perturbed_x_copies(
    whole_data,
    *,
    feat_idx: int,
    sensitive_feature_values: Union[List, None],
    perturb_k: int,
    rng_seed: int,
) -> List[torch.Tensor]:
    """
    Match ``perturb_feature_and_measure_probs`` feature resampling on ``x``.
    Caller should set RNG if global reproducibility is required; ``rng_seed`` is applied here via
    ``torch.manual_seed`` / ``np.random.seed`` before draws.
    """
    torch.manual_seed(int(rng_seed))
    np.random.seed(int(rng_seed) % (2**32 - 1))

    original_x = whole_data.x.clone()

    feat_vals = original_x[:, feat_idx].detach().cpu().numpy()
    val_mask_np = whole_data.val_mask.detach().cpu().numpy()
    val_feat_vals = feat_vals[val_mask_np]
    unique_feature_vals = np.unique(feat_vals)
    is_binary_feature = len(unique_feature_vals) == 2

    if sensitive_feature_values is not None:
        candidate_values = np.array(sensitive_feature_values, dtype=np.float64)
    else:
        val_min, val_max = float(val_feat_vals.min()), float(val_feat_vals.max())
        candidate_values = None

    num_nodes = int(original_x.size(0))
    X_copies: List[torch.Tensor] = []

    if is_binary_feature:
        bin_vals = np.sort(unique_feature_vals)
        low_val, high_val = bin_vals[0], bin_vals[1]
        low_tensor = torch.tensor(low_val, dtype=original_x.dtype, device=original_x.device)
        high_tensor = torch.tensor(high_val, dtype=original_x.dtype, device=original_x.device)
        x_mod = original_x.clone()
        feat_tensor = x_mod[:, feat_idx]
        mask_low = torch.isclose(feat_tensor, low_tensor)
        feat_tensor[mask_low] = high_tensor
        feat_tensor[~mask_low] = low_tensor
        X_copies.append(x_mod)
    else:
        for _ in range(int(perturb_k)):
            x_mod = original_x.clone()
            if candidate_values is not None:
                sampled_vals = torch.tensor(
                    np.random.choice(candidate_values, size=num_nodes),
                    dtype=x_mod.dtype,
                    device=x_mod.device,
                )
            else:
                sampled_vals = torch.empty(num_nodes, dtype=x_mod.dtype, device=x_mod.device).uniform_(val_min, val_max)
            x_mod[:, feat_idx] = sampled_vals
            X_copies.append(x_mod)

    return X_copies


def aggregate_eval_metrics_across_perturbed_copies(
    model,
    whole_data_template,
    x_copies: List[torch.Tensor],
) -> Dict[str, Any]:
    """Mean / std (per split metric) across forward passes with each perturbed ``x`` copy."""
    if not x_copies:
        return {"num_replicates": 0}

    payloads: List[Dict[str, Any]] = []
    for x_new in x_copies:
        d = clone_data_with_x(whole_data_template, x_new)
        payloads.append(dict(eval_on_data(model, d)))

    n = len(payloads)
    is_reg = payloads[0].get("task") == "regression"

    ov_means_list = zip(*[p["overall"] for p in payloads])
    ov_means = [float(statistics.fmean(vals)) if n else float("nan") for vals in ov_means_list]
    if n > 1:
        ov_stds = [
            float(statistics.pstdev(vals)) if all(isinstance(x, numbers.Real) and math.isfinite(x) for x in vals)
            else float("nan")
            for vals in zip(*[p["overall"] for p in payloads])
        ]
    else:
        ov_stds = [0.0 for _ in ov_means]

    splits_mean: Dict[str, Dict[str, float]] = {}
    splits_std: Dict[str, Dict[str, float]] = {}
    for split in ("train", "val", "test"):
        keys = payloads[0]["metrics_splits"][split].keys()
        splits_mean[split] = {}
        splits_std[split] = {}
        for k in sorted(keys):
            seq = []
            for p in payloads:
                v = p["metrics_splits"][split].get(k)
                if isinstance(v, numbers.Real) and math.isfinite(v):
                    seq.append(float(v))
            if not seq:
                splits_mean[split][k] = float("nan")
                splits_std[split][k] = float("nan")
            else:
                splits_mean[split][k] = float(statistics.fmean(seq))
                splits_std[split][k] = float(statistics.pstdev(seq)) if len(seq) > 1 else 0.0

    auc_pr_mean: Dict[str, float] = {}
    auc_pr_std: Dict[str, float] = {}
    if not is_reg:
        for sp in ("train", "val", "test"):
            seq = []
            for p in payloads:
                a = (p.get("auc_pr") or {}).get(sp)
                if isinstance(a, numbers.Real) and math.isfinite(a):
                    seq.append(float(a))
            if not seq:
                auc_pr_mean[sp] = float("nan")
                auc_pr_std[sp] = float("nan")
            else:
                auc_pr_mean[sp] = float(statistics.fmean(seq))
                auc_pr_std[sp] = float(statistics.pstdev(seq)) if len(seq) > 1 else 0.0

    out: Dict[str, Any] = {
        "num_replicates": n,
        "task": payloads[0].get("task"),
        "overall_mean_across_replicates": ov_means,
        "overall_std_across_replicates": ov_stds,
        "metrics_splits_mean_across_replicates": splits_mean,
        "metrics_splits_std_across_replicates": splits_std,
    }
    if not is_reg:
        out["auc_pr_mean_across_replicates"] = auc_pr_mean
        out["auc_pr_std_across_replicates"] = auc_pr_std
    return out


def perturb_branch_metrics(
    model,
    whole_data,
    *,
    feature_name: str,
    sensitive_feature_values: Union[List, None],
    perturb_k: int,
    compute_flips: bool,
    perturb_metrics_rng_seed: int,
) -> Mapping[str, Any]:
    regress = _is_regression_data(whole_data)
    val_df, test_df = perturb_feature_and_measure_probs(
        model,
        whole_data,
        feature_name=feature_name,
        sensitive_feature_values=sensitive_feature_values,
        K=perturb_k,
        relative=True,
        prob_mode="true_class",
        compute_flips=compute_flips and not regress,
    )
    fc_idx = whole_data.feature_names.index(feature_name)
    x_copies = build_pipeline_style_perturbed_x_copies(
        whole_data,
        feat_idx=fc_idx,
        sensitive_feature_values=sensitive_feature_values,
        perturb_k=perturb_k,
        rng_seed=perturb_metrics_rng_seed,
    )
    pred_agg = aggregate_eval_metrics_across_perturbed_copies(model, whole_data, x_copies)

    return {
        "sensitivity": summarize_perturb_sensitivity(
            val_df, test_df, compute_flips=compute_flips and not regress,
        ),
        "prediction_metrics_across_input_perturbation_replicates": pred_agg,
        "prediction_replicate_sampling_rng_seed": perturb_metrics_rng_seed,
    }


def _normalized_method_name(args) -> str:
    return "seed_gnn" if getattr(args, "method", None) == "seedgnn" else str(args.method)


def load_config_ns(args) -> Dict:
    if getattr(args, "method", None) == "seedgnn":
        args.method = "seed_gnn"
    return build_config(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focused feature: mask + perturb evaluation (pretrain vs edited)."
    )
    parser.add_argument(
        "--method",
        default="leastsquares",
        choices=["leastsquares", "finetune", "egnn", "seed_gnn", "seedgnn"],
    )
    parser.add_argument("--dataset", default="pokec")
    parser.add_argument("--model", default="GCN_MLP")
    parser.add_argument("--num-targets", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--leastsquares-strategy",
        type=str,
        default="sens_subspace_retention_pr_graphaware",
        choices=[
            "confidence",
            "sensitivity_mean",
            "sensitivity_wtd_mean",
            "sens_pr",
            "sens_pr_graphaware",
            "sens_divrank",
            "sens_divrank_graphaware",
            "sens_subspace_retention",
            "sens_subspace_retention_pr",
            "sens_subspace_retention_pr_graphaware",
            "sens_subspace_retention_divrank",
            "sens_subspace_retention_divrank_graphaware",
            "sens_subspace",
            "sens_subspace_pr",
            "sens_subspace_pr_graphaware",
            "sens_subspace_divrank",
            "sens_subspace_divrank_graphaware",
        ],
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help=(
            "Root folder containing dataset subdirs (pokec/, cora/, …). "
            "Default: DATASET_DIR env or …/seed_gnn_data/dataset (see run_edit.sh / run_editing_suite.sh)."
        ),
    )
    parser.add_argument(
        "--pretrain-dir",
        default=DEFAULT_PRETRAIN_DIR,
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "With no --metrics-json / --edited-checkpoint, checkpoints are resolved under "
            "<output-root>/<method>/<dataset>/<model>/checkpoints/ (same as run_edit.sh)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Experiment output folder (parent of checkpoints/). Overrides --output-root layout when set.",
    )
    parser.add_argument(
        "--metrics-json",
        default="",
        help="Preferred: resolves edited .pt beside this file. Omit with full hyperparams + default layout.",
    )
    parser.add_argument(
        "--edited-checkpoint",
        default="",
        help="Explicit path to edited .pt. If under .../<model>/checkpoints/, output folder is inferred.",
    )
    parser.add_argument(
        "--edited-targets",
        type=int,
        default=None,
        help=(
            "EGNN / SEED-GNN: override N in …_targets{N}_… checkpoint names. "
            "When deriving the path from CLI (same layout as editors), defaults to "
            "max(1, round(top_fraction × number of validation nodes)); --num-targets is not used for that."
        ),
    )
    parser.add_argument(
        "--feature-name",
        default="",
        help="Override focused feature; otherwise uses suite map (see suite_feature_maps) or run_edit.",
    )
    parser.add_argument(
        "--legacy-feature-map",
        action="store_true",
        help="Do not overlay suite_feature_maps; keep run_edit.build_config fields only.",
    )
    parser.add_argument("--skip-mask", action="store_true", help="Skip masked-input evaluation.")
    parser.add_argument("--skip-perturb", action="store_true", help="Skip perturb_feature sensitivity branch.")
    parser.add_argument(
        "--perturb-k",
        type=int,
        default=6,
        help="K augmented samples inside perturb_feature_and_measure_probs.",
    )
    parser.add_argument(
        "--no-compute-flips",
        action="store_true",
        help="Pass compute_flips=False to perturb_feature (faster; omits flip fraction).",
    )
    parser.add_argument(
        "--perturb-metrics-seed",
        type=int,
        default=None,
        help="RNG seed for replicated perturbed-X copies used in accuracy/AUC-PR aggregates (default: --seed).",
    )
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--top-fraction", type=float, default=0.25)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--pr-alpha", type=float, default=0.85)
    parser.add_argument("--rank-mix-tau", type=float, default=0.75)
    parser.add_argument("--gamma-retain", type=float, default=1.0)
    parser.add_argument("--ft-epochs", type=int, default=None)
    parser.add_argument("--ft-lr", type=float, default=None)
    parser.add_argument(
        "--ablations-dir",
        default=str(DEFAULT_ABLATIONS_DIR),
        help="Default directory for JSON results when --save-json is not set.",
    )
    parser.add_argument(
        "--save-json",
        default="",
        help="Explicit output JSON path (overrides default file under --ablations-dir).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write a JSON file (print to stdout only).",
    )
    parser.add_argument(
        "--deterministic-ablation-filename",
        action="store_true",
        help=(
            "Under --ablations-dir, write feature_ablation_{method}_{dataset}_{model}_seed{seed}_{ck_slug}.json "
            "with no timestamp (same checkpoint → same path; re-runs overwrite)."
        ),
    )
    args = parser.parse_args()
    if os.getenv("FEATURE_ABLATION_DETERMINISTIC", "").strip().lower() in ("1", "true", "yes"):
        args.deterministic_ablation_filename = True

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    perturb_metrics_seed = (
        int(args.perturb_metrics_seed) if args.perturb_metrics_seed is not None else int(args.seed)
    )

    config = load_config_ns(args)
    feature_map_source = "run_edit.build_config only"
    if not args.legacy_feature_map:
        apply_suite_sensitive_feature_config(config)
        feature_map_source = "run_editing_suite.sh via editing_pipelines.suite_feature_maps"

    edited_path: Optional[Path] = None
    out_dir: str
    derivation: Optional[str] = None
    defer_derived_ckpt_for_graph_editors = False

    if args.metrics_json:
        metrics_abs = os.path.abspath(args.metrics_json)
        out_dir = os.path.abspath(args.output_dir or str(Path(metrics_abs).parent))
        edited_path = resolve_edited_checkpoint_from_metrics_json(metrics_abs)
        derivation = "metrics_json"
    elif args.edited_checkpoint:
        edited_path = Path(os.path.abspath(args.edited_checkpoint))
        if args.output_dir:
            out_dir = os.path.abspath(args.output_dir)
        elif "checkpoints" in edited_path.parts:
            out_dir = os.path.abspath(str(edited_path.parent.parent))
        else:
            raise SystemExit(
                "With --edited-checkpoint outside …/checkpoints/, pass --output-dir for config bookkeeping."
            )
        derivation = "explicit_pt"
        if not edited_path.is_file():
            raise FileNotFoundError(f"--edited-checkpoint not found: {edited_path}")
    else:
        out_dir = os.path.abspath(
            args.output_dir
            or default_output_folder_dir(
                args.method,
                args.dataset,
                args.model,
                root=args.output_root,
            )
        )
        nm = _normalized_method_name(args)
        if nm in ("egnn", "seed_gnn"):
            defer_derived_ckpt_for_graph_editors = True
        else:
            edited_path = resolve_derived_checkpoint_path(
                config=config,
                args=args,
                output_dir=out_dir,
                output_root=args.output_root,
            )
            derivation = "derived_from_args"

    config["management"]["output_folder_dir"] = os.path.abspath(out_dir)
    config["management"]["pretrain_output_dir"] = os.path.abspath(args.pretrain_dir)

    model_pre, _train, whole_data, _, _ = load_model(config)

    if defer_derived_ckpt_for_graph_editors:
        val_count = int(whole_data.val_mask.sum().item())
        inferred_n = max(1, int(round(float(args.top_fraction) * float(val_count))))
        if args.edited_targets is not None:
            inferred_n = int(args.edited_targets)
        args.edited_targets = inferred_n
        edited_path = resolve_derived_checkpoint_path(
            config=config,
            args=args,
            output_dir=out_dir,
            output_root=args.output_root,
        )
        derivation = "derived_from_args"

    # Use the graph tensor device as canonical (matches `prepare_dataset` + `model.to(get_device())`).
    device = whole_data.x.device
    pre_dev = next(model_pre.parameters()).device
    if pre_dev != device:
        logger.warning(
            "model_pre device %s != whole_data.x.device %s; loading edited weights onto data device",
            pre_dev,
            device,
        )
    # Avoid CUDA `deepcopy` quirks (mixed CPU/CUDA params after load). Deepcopy, move only the copy to CPU
    # for checkpoint I/O, then move the full module tree to the graph device in one step.
    model_ed = copy.deepcopy(model_pre)
    model_ed.cpu()
    state = torch.load(str(edited_path), map_location="cpu")
    if isinstance(state, dict):
        state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
    try:
        model_ed.load_state_dict(state, strict=True)
    except RuntimeError as err:
        logger.warning("Strict load failed (%s); retrying strict=False.", err)
        model_ed.load_state_dict(state, strict=False)
    model_ed.to(device)
    for t in list(model_ed.parameters()) + list(model_ed.buffers()):
        if t is not None and t.device != device:
            t.data = t.data.to(device)
    if hasattr(model_ed, "gnn_output"):
        model_ed.gnn_output = None
    model_ed.eval()

    pp = config.get("pipeline_params", {})
    feat_idx, focused_name = resolve_focus_feature_index(whole_data, config, args.feature_name or None)
    continuous = focused_column_is_continuous(whole_data, feat_idx)

    sensitive_vals = pp.get("fixed_sensitive_values")

    is_reg = _is_regression_data(whole_data)
    compute_flips = not args.no_compute_flips and not is_reg

    payload: Dict[str, Any] = {
        "editing_method": args.method,
        "checkpoint_resolution": derivation,
        "feature_map_source": feature_map_source,
        "focused_feature": focused_name,
        "focused_feature_index": feat_idx,
        "mask_rule": {"continuous_heuristic_gt30_unique_values": continuous, "continuous_fill": "global_mean"},
        "edited_checkpoint": str(edited_path),
        "perturb_branch": {
            "K": args.perturb_k,
            "compute_flips": compute_flips,
            "anchors": sensitive_vals,
            "replicate_aggregate_rng_seed": perturb_metrics_seed,
        },
    }

    variants = {}

    variants["baseline_original_inputs"] = {
        "pretrain": eval_on_data(model_pre, whole_data),
        "edited": eval_on_data(model_ed, whole_data),
    }

    data_masked = clone_data_with_x(whole_data, build_masked_x(whole_data, feat_idx))

    if not args.skip_mask:
        if is_reg:
            mask_shift_pre = mask_impact_regression(model_pre, whole_data, data_masked)
            mask_shift_ed = mask_impact_regression(model_ed, whole_data, data_masked)
        else:
            mask_shift_pre = mask_impact_classification(model_pre, whole_data, data_masked)
            mask_shift_ed = mask_impact_classification(model_ed, whole_data, data_masked)

        variants["masked_inputs"] = {
            "pretrain": dict(
                eval_on_data(model_pre, data_masked),
                mask_shift_vs_original_inputs=mask_shift_pre,
            ),
            "edited": dict(
                eval_on_data(model_ed, data_masked),
                mask_shift_vs_original_inputs=mask_shift_ed,
            ),
        }

    if not args.skip_perturb:
        variants["input_perturbation_sensitivity"] = {
            "pretrain": perturb_branch_metrics(
                model_pre,
                whole_data,
                feature_name=focused_name,
                sensitive_feature_values=sensitive_vals,
                perturb_k=args.perturb_k,
                compute_flips=compute_flips,
                perturb_metrics_rng_seed=perturb_metrics_seed,
            ),
            "edited": perturb_branch_metrics(
                model_ed,
                whole_data,
                feature_name=focused_name,
                sensitive_feature_values=sensitive_vals,
                perturb_k=args.perturb_k,
                compute_flips=compute_flips,
                perturb_metrics_rng_seed=perturb_metrics_seed,
            ),
        }

    payload["variants"] = _json_safe(variants)

    if args.no_save:
        json_out: Optional[Path] = None
    elif args.save_json:
        json_out = Path(os.path.abspath(args.save_json))
    else:
        ck_slug = Path(str(edited_path)).stem[:120].replace(" ", "_")
        base_name = (
            f"feature_ablation_{args.method}_{args.dataset}_{args.model}_seed{args.seed}_{ck_slug}"
        )
        if args.deterministic_ablation_filename:
            json_out = Path(args.ablations_dir).resolve() / f"{base_name}.json"
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            json_out = Path(args.ablations_dir).resolve() / f"{base_name}_{stamp}.json"

    if json_out is not None:
        payload["output_json"] = str(json_out)

    print(json.dumps(payload, indent=2))
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Wrote %s", json_out)


if __name__ == "__main__":
    main()
