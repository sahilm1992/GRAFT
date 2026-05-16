#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List
from copy import deepcopy

import numpy as np
import torch
import matplotlib

# Use non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# -------------------------------
# Path setup
# -------------------------------
ROOT = str(Path(__file__).resolve().parents[2])
SEED_GNN = os.path.join(ROOT, "seed-gnn")
sys.path.insert(0, ROOT)
sys.path.insert(0, SEED_GNN)

from editing_pipelines.run_edit import build_config  # noqa: E402
from editing_pipelines import create_editor  # noqa: E402
from data import get_data, prepare_dataset  # noqa: E402
from edit_gnn.utils import grab_input  # noqa: E402


def get_device():
    force_cpu = os.getenv("FORCE_CPU", "0") == "1"
    has_visible_cuda = os.getenv("CUDA_VISIBLE_DEVICES", "") != ""
    if torch.cuda.is_available() and has_visible_cuda and not force_cpu:
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_feature(
    data_obj, feature_name: Optional[str], feature_index: Optional[int]
) -> Tuple[int, str]:
    if feature_name is not None and feature_name != "":
        if not hasattr(data_obj, "feature_names") or feature_name not in data_obj.feature_names:  # type: ignore[attr-defined]
            raise ValueError(
                f"Feature name '{feature_name}' not found in data.feature_names"
            )
        idx = int(data_obj.feature_names.index(feature_name))  # type: ignore[attr-defined]
        return idx, feature_name

    if feature_index is not None:
        d = int(data_obj.x.size(1))
        if feature_index < 0 or feature_index >= d:
            raise ValueError(f"feature_index {feature_index} out of range [0, {d-1}]")
        idx = int(feature_index)
        if hasattr(data_obj, "feature_names") and data_obj.feature_names and 0 <= idx < len(data_obj.feature_names):  # type: ignore[attr-defined]
            label = str(data_obj.feature_names[idx])  # type: ignore[index]
        else:
            label = f"feat_{idx}"
        return idx, label

    raise ValueError("Provide either --feature-name or --feature-index.")


def get_mask(data_obj, split: str) -> torch.Tensor:
    device = data_obj.x.device
    if split == "whole":
        return torch.ones(data_obj.num_nodes, dtype=torch.bool, device=device)
    if split == "train":
        if not hasattr(data_obj, "train_mask"):
            raise ValueError("Dataset does not have train_mask")
        return data_obj.train_mask
    if split == "val":
        if not hasattr(data_obj, "val_mask"):
            raise ValueError("Dataset does not have val_mask")
        return data_obj.val_mask
    if split == "test":
        if not hasattr(data_obj, "test_mask"):
            raise ValueError("Dataset does not have test_mask")
        return data_obj.test_mask
    raise ValueError(f"Unknown split '{split}'.")


def load_dataset_and_editor(args):
    """
    Build config, load dataset, and initialize the requested editor.
    """
    cfg = build_config(
        argparse.Namespace(
            method=args.method,
            dataset=args.dataset,
            model=args.model,
            num_targets=args.num_targets,
            max_steps=args.max_steps,
            strategy=args.strategy or "",
            dataset_dir=args.dataset_dir,
            pretrain_dir=args.pretrain_dir,
            output_dir=args.output_dir,
            load_pretrained_backbone=False if args.load_pretrained_backbone is None else args.load_pretrained_backbone,
        )
    )
    raw_data, _nf, _nc = get_data(cfg["management"]["dataset_dir"], cfg["eval_params"]["dataset"], cfg)
    _train_data, whole_data = prepare_dataset(cfg["pipeline_params"], raw_data, remove_edge_index=False)
    del raw_data
    editor = create_editor(args.method, cfg)
    editor.load_model_and_data()
    return cfg, editor, whole_data


@torch.no_grad()
def probs_true_class(model, data_obj) -> torch.Tensor:
    model = model.to(data_obj.x.device).eval()
    logits = model(**grab_input(data_obj))
    probs = torch.softmax(logits, dim=-1)
    y = data_obj.y
    # true-class probability per node
    return probs.gather(1, y.view(-1, 1)).squeeze(1)


def binned_means(x: np.ndarray, y: np.ndarray, num_bins: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(np.nanmin(x), np.nanmax(x), num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(num_bins, np.nan, dtype=float)
    for i in range(num_bins):
        mask = (x >= edges[i]) & (x < edges[i + 1]) if i < num_bins - 1 else (x >= edges[i]) & (x <= edges[i + 1])
        if np.any(mask):
            means[i] = float(np.nanmean(y[mask]))
    return centers, means


def plot_scatter_and_binned(
    feat_vals: np.ndarray,
    p_before: np.ndarray,
    p_after: np.ndarray,
    dataset: str,
    model: str,
    method: str,
    feat_label: str,
    split: str,
    out_dir: Path,
    bins: int,
    sample: Optional[int] = None,
) -> None:
    """
    Scatter plot of feature vs true-class probability before/after,
    with optional binned mean trendlines.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Optional subsampling for very large datasets
    if sample is not None and sample > 0 and feat_vals.shape[0] > sample:
        idx = np.random.RandomState(0).choice(feat_vals.shape[0], size=sample, replace=False)
        fx = feat_vals[idx]
        pb = p_before[idx]
        pa = p_after[idx]
    else:
        fx, pb, pa = feat_vals, p_before, p_after

    plt.figure(figsize=(7, 4))
    plt.scatter(fx, pb, s=6, alpha=0.35, color="tab:blue", label="Before")
    plt.scatter(fx, pa, s=6, alpha=0.35, color="tab:red", label="After")

    # Binned means
    centers, mb = binned_means(fx, pb, num_bins=bins)
    _, ma = binned_means(fx, pa, num_bins=bins)
    plt.plot(centers, mb, color="tab:blue", linewidth=2, label="Before (mean)")
    plt.plot(centers, ma, color="tab:red", linewidth=2, label="After (mean)")

    plt.xlabel(feat_label)
    plt.ylabel("True-class probability")
    plt.ylim(0.0, 1.0)
    plt.title(f"{dataset} | {model} | {method} | {split.upper()}")
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / f"{dataset}_{model}_{method}_{feat_label}_{split}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize feature vs true-class probability before/after editing."
    )
    # Standard args
    parser.add_argument("--dataset", default="pokec")
    parser.add_argument("--model", default="GCN_MLP")
    parser.add_argument("--method", default="leastsquares")
    parser.add_argument("--strategy", default="", help="Optional target selection strategy")
    parser.add_argument("--dataset-dir", default=os.path.join(SEED_GNN))
    parser.add_argument("--pretrain-dir", default=os.path.join(SEED_GNN, "pretrained_models"))
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "editing_pipelines", "output"))
    parser.add_argument("--num-targets", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--load-pretrained-backbone", type=int, default=1, help="1/0 flag")

    # Feature selection
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--feature-name", type=str, default=None)
    g.add_argument("--feature-index", type=int, default=None)

    # Split and plotting controls
    parser.add_argument("--split", choices=["whole", "train", "val", "test", "all"], default="whole")
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--sample", type=int, default=0, help="Optional subsample size for scatter; 0=all")

    args = parser.parse_args()
    args.load_pretrained_backbone = bool(int(args.load_pretrained_backbone)) if args.load_pretrained_backbone is not None else None

    cfg, editor, data = load_dataset_and_editor(args)
    device = get_device()
    data = data.to(device)
    feat_idx, feat_label = resolve_feature(data, args.feature_name, args.feature_index)

    # Capture model before editing
    model_before = deepcopy(editor.model).to(device).eval()

    # Run minimal edit pipeline
    editor.evaluate_before_edit()
    editor.fine_tune_if_needed()
    node_idx_2flip, flipped_label = editor.select_edit_targets()
    _ = editor.edit_model(node_idx_2flip=node_idx_2flip, flipped_label=flipped_label)
    model_after = editor.model.to(device).eval()

    # Compute true-class probabilities
    p_before = probs_true_class(model_before, data).detach().cpu()
    p_after = probs_true_class(model_after, data).detach().cpu()
    x_feat = data.x[:, feat_idx].detach().cpu()

    out_root = Path(args.output_dir) / "dataset_performance_plots" / f"{args.dataset}_{args.model}_{args.method}"
    if args.split == "all":
        for split in ["train", "val", "test"]:
            try:
                mask = get_mask(data, split)
            except ValueError:
                continue
            fx = x_feat[mask].numpy()
            pb = p_before[mask].numpy()
            pa = p_after[mask].numpy()
            plot_scatter_and_binned(
                fx, pb, pa,
                args.dataset, args.model, args.method, feat_label, split,
                out_root, bins=args.bins, sample=args.sample if args.sample > 0 else None
            )
    else:
        mask = get_mask(data, args.split)
        fx = x_feat[mask].numpy()
        pb = p_before[mask].numpy()
        pa = p_after[mask].numpy()
        plot_scatter_and_binned(
            fx, pb, pa,
            args.dataset, args.model, args.method, feat_label, args.split,
            out_root, bins=args.bins, sample=args.sample if args.sample > 0 else None
        )

    print(f"Saved plots under: {out_root}")


if __name__ == "__main__":
    main()




