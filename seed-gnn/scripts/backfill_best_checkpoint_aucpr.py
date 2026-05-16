#!/usr/bin/env python3
"""
Backfill best-checkpoint split AUC-PR metrics into pretrain JSON outputs.

This script updates existing `metrics_pretrain_*.json` files in-place by:
1) Loading each JSON's best checkpoint.
2) Re-loading the matching dataset split/features configuration.
3) Computing split AUC-PR with the exact editing-pipeline helper:
      editing_pipelines.utils.metrics.compute_full_auc_pr_by_split
4) Writing results back into `training.metrics`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch


def _add_repo_paths() -> None:
    script_path = Path(__file__).resolve()
    seed_gnn_root = script_path.parents[1]
    repo_root = script_path.parents[2]
    for p in (seed_gnn_root, repo_root):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.insert(0, p_str)


_add_repo_paths()

from editing_pipelines.utils.metrics import compute_full_auc_pr_by_split  # noqa: E402
from editing_pipelines.utils.model_io import load_model  # noqa: E402


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_nan(value: Any) -> bool:
    return _is_number(value) and math.isnan(float(value))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill best-checkpoint split AUC-PR into metrics_pretrain JSON files."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="/home/model_editing/data/seed_gnn_data/results/seed_gnn",
        help="Root directory containing metrics_pretrain_*.json files.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/model_editing/data/seed_gnn_data/dataset",
        help="Dataset root passed to seed-gnn get_data().",
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default="**/metrics_pretrain_*.json",
        help="Glob pattern under --results-root for metric JSON files.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="*",
        default=[],
        help="Optional dataset filter(s), e.g. --dataset pokec bail",
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="*",
        default=[],
        help="Optional model filter(s), e.g. --model GCN GAT_MLP",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of files to process (0 means all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print updates without writing files.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing best-checkpoint AUC-PR fields if already present.",
    )
    return parser.parse_args()


def _normalize_filters(values: List[str]) -> set[str]:
    return {v.strip().lower() for v in values if v and v.strip()}


def _read_payload(metrics_path: Path) -> Dict[str, Any]:
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _matches_filters(
    payload: Dict[str, Any],
    dataset_filters: set[str],
    model_filters: set[str],
) -> bool:
    dataset_name = str(payload.get("data", {}).get("dataset", "")).lower()
    model_name = str(payload.get("model", {}).get("name", "")).lower()
    dataset_ok = not dataset_filters or dataset_name in dataset_filters
    model_ok = not model_filters or model_name in model_filters
    return dataset_ok and model_ok


def _infer_feature_variant(metrics_path: Path, dataset_name: str, payload: Dict[str, Any]) -> str:
    parts = list(metrics_path.parts)
    if dataset_name in parts:
        idx = parts.index(dataset_name)
        if idx + 1 < len(parts):
            maybe_variant = parts[idx + 1]
            if maybe_variant != metrics_path.name:
                return maybe_variant

    ckpt_path = (
        payload.get("artifacts", {})
        .get("best_checkpoint", {})
        .get("path", "")
    )
    if ckpt_path:
        ckpt_parts = list(Path(ckpt_path).parts)
        if dataset_name in ckpt_parts:
            idx = ckpt_parts.index(dataset_name)
            if idx + 1 < len(ckpt_parts):
                return ckpt_parts[idx + 1]

    return "full_features"


def _drop_features_from_variant(feature_variant: str) -> List[str]:
    if feature_variant == "full_features":
        return []
    if feature_variant.startswith("no_") and len(feature_variant) > 3:
        return [feature_variant[3:]]
    return []


def _should_skip_existing(metrics: Dict[str, Any], overwrite: bool) -> bool:
    if overwrite:
        return False
    required_keys = [
        "train_auc_pr_best_val",
        "val_auc_pr_best",
        "test_auc_pr_best",
    ]
    return all(k in metrics for k in required_keys)


def _to_json_float_dict(d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        out[k] = float(v) if _is_number(v) else float("nan")
        if _is_nan(out[k]):
            out[k] = float("nan")
    return out


def _model_family_from_name(model_name: str) -> str:
    return model_name.replace("_MLP", "").lower()


def _load_pipeline_defaults(dataset_name: str, model_name: str) -> Dict[str, Any]:
    script_path = Path(__file__).resolve()
    cfg_path = (
        script_path.parents[1]
        / "config"
        / "pipeline_config"
        / "seed_gnn"
        / _model_family_from_name(model_name)
        / f"{dataset_name}.json"
    )
    if not cfg_path.exists():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("pipeline_params", {}) or {}
    except Exception:
        return {}


def _resolve_loader_model_name(json_model_name: str, pipeline_default_model_name: str | None) -> str:
    """
    Choose model name for editing loader.

    For the Polynormer architecture family only, force backbone model loading
    (`Polynormer`) instead of `Polynormer_MLP` to match the requested behavior.
    """
    if _model_family_from_name(json_model_name) == "polynormer":
        return "Polynormer"
    if pipeline_default_model_name:
        return pipeline_default_model_name
    return json_model_name


def _load_model_and_data(
    payload: Dict[str, Any],
    dataset_dir: str,
    metrics_path: Path,
) -> tuple[Any, Any]:
    dataset_name = payload["data"]["dataset"]
    model_name = payload["model"]["name"]
    arch = payload["model"].get("architecture", {}) or {}
    best_ckpt = payload.get("artifacts", {}).get("best_checkpoint", {}).get("path", "")
    if best_ckpt and not os.path.exists(best_ckpt):
        raise FileNotFoundError(f"Missing best checkpoint: {best_ckpt}")

    feature_variant = _infer_feature_variant(metrics_path, dataset_name, payload)
    drop_features = _drop_features_from_variant(feature_variant)
    seed = int(payload.get("experiment", {}).get("seed", 42))
    ckpt_dir = payload.get("artifacts", {}).get("checkpoint_dir")
    if not ckpt_dir:
        raise KeyError(f"Missing artifacts.checkpoint_dir in {metrics_path}")

    pipeline_defaults = _load_pipeline_defaults(dataset_name, model_name)
    merged_arch = dict(pipeline_defaults.get("architecture", {}) or {})
    merged_arch.update(arch)
    loader_model_name = _resolve_loader_model_name(
        json_model_name=model_name,
        pipeline_default_model_name=pipeline_defaults.get("model_name"),
    )

    # Reconstruct an editing-pipeline-compatible config and use the shared loader
    # so loading behavior (model mode, checkpoint discovery, data prep) matches
    # what editing pipelines use before evaluation.
    load_cfg = {
        "management": {
            "output_folder_dir": str(metrics_path.parent),
            "dataset_dir": dataset_dir,
            "pretrain_output_dir": ckpt_dir,
            "seed": seed,
        },
        "pipeline_params": {
            "method": pipeline_defaults.get("method", "seed_gnn"),
            "drop_features": drop_features,
            "model_name": loader_model_name,
            "architecture": merged_arch,
            "feature_variant": feature_variant,
            "load_pretrained_backbone": pipeline_defaults.get("load_pretrained_backbone", True),
            "optim": pipeline_defaults.get("optim", "adam"),
            "pretrain_lr": pipeline_defaults.get("pretrain_lr", 0.01),
            "edit_lr": pipeline_defaults.get("edit_lr", 0.001),
            "loop": pipeline_defaults.get("loop", True),
            "normalize": pipeline_defaults.get("normalize", True),
        },
        "eval_params": {"dataset": dataset_name},
    }
    model, _train_data, whole_data, _num_features, _num_classes = load_model(load_cfg)
    return model, whole_data


def _backfill_file(
    metrics_path: Path,
    dataset_dir: str,
    dry_run: bool,
    overwrite_existing: bool,
    payload: Dict[str, Any] | None = None,
) -> str:
    if payload is None:
        payload = _read_payload(metrics_path)

    training = payload.setdefault("training", {})
    metrics = training.setdefault("metrics", {})
    if _should_skip_existing(metrics, overwrite_existing):
        return "skipped_existing"

    model, whole_data = _load_model_and_data(payload, dataset_dir, metrics_path)
    auc_by_split = _to_json_float_dict(compute_full_auc_pr_by_split(model, whole_data))

    metrics["train_auc_pr_best_val"] = auc_by_split["train"]
    metrics["val_auc_pr_best"] = auc_by_split["val"]
    metrics["test_auc_pr_best"] = auc_by_split["test"]
    metrics["best_checkpoint_auc_pr_by_split"] = auc_by_split

    if not dry_run:
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    return "updated"


def main() -> None:
    args = _parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    files = sorted(results_root.glob(args.glob_pattern))
    dataset_filters = _normalize_filters(args.dataset)
    model_filters = _normalize_filters(args.model)
    selected: List[tuple[Path, Dict[str, Any]]] = []

    for path in files:
        try:
            payload = _read_payload(path)
            if _matches_filters(payload, dataset_filters, model_filters):
                selected.append((path, payload))
        except Exception as exc:
            print(f"[failed] {path} :: {exc}")

    if args.limit > 0:
        selected = selected[: args.limit]

    updated = 0
    skipped = 0
    failed = 0

    for path, payload in selected:
        try:
            status = _backfill_file(
                metrics_path=path,
                dataset_dir=args.dataset_dir,
                dry_run=args.dry_run,
                overwrite_existing=args.overwrite_existing,
                payload=payload,
            )
            if status == "updated":
                updated += 1
            else:
                skipped += 1
            print(f"[{status}] {path}")
        except Exception as exc:
            failed += 1
            print(f"[failed] {path} :: {exc}")

    print(
        f"Done. files={len(selected)}, updated={updated}, skipped={skipped}, failed={failed}, "
        f"dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
