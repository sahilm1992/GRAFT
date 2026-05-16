#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

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
from data import get_data, prepare_dataset  # noqa: E402


def load_dataset(args) -> "torch_geometric.data.data.Data":
    """
    Load dataset using the same config/utilities as the main pipelines.
    We build a minimal config via build_config for consistency.
    """
    cfg = build_config(
        argparse.Namespace(
            method="none",
            dataset=args.dataset,
            model=args.model,
            num_targets=0,
            max_steps=0,
            strategy="",
            dataset_dir=args.dataset_dir,
            pretrain_dir=args.pretrain_dir,
            output_dir=args.output_dir,
            load_pretrained_backbone=False,
        )
    )
    raw_data, _num_features, _num_classes = get_data(
        cfg["management"]["dataset_dir"], cfg["eval_params"]["dataset"], cfg
    )
    _train_data, whole_data = prepare_dataset(
        cfg["pipeline_params"], raw_data, remove_edge_index=False
    )
    del raw_data
    return whole_data


def resolve_feature(
    data_obj, feature_name: Optional[str], feature_index: Optional[int]
) -> Tuple[int, str]:
    """
    Determine feature index and label from either a provided name or index.
    """
    if feature_name is not None and feature_name != "":
        if not hasattr(data_obj, "feature_names") or feature_name not in data_obj.feature_names:  # type: ignore[attr-defined]
            raise ValueError(
                f"Feature name '{feature_name}' not found in data.feature_names"
            )
        idx = int(data_obj.feature_names.index(feature_name))  # type: ignore[attr-defined]
        label = feature_name
        return idx, label

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
    """
    Return a boolean mask tensor for the requested split.
    """
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


def plot_hist_or_bars(
    values: np.ndarray,
    title: str,
    out_path: Path,
    bins: int = 30,
    color: Optional[str] = None,
) -> None:
    """
    Plot either a histogram (continuous) or bar chart (discrete) depending on value distribution.
    """
    plt.figure(figsize=(6, 4))
    # Decide discrete vs continuous
    unique_vals = np.unique(values)
    is_integer = np.issubdtype(values.dtype, np.integer) or np.allclose(
        unique_vals, np.round(unique_vals)
    )

    if is_integer and unique_vals.size <= 30:
        # Discrete bar plot
        unique_vals_sorted = np.sort(unique_vals)
        counts = np.array([(values == v).sum() for v in unique_vals_sorted], dtype=int)
        plt.bar(unique_vals_sorted, counts, color=color if color else "steelblue")
        plt.xlabel("Value")
        plt.ylabel("Count")
    else:
        # Continuous histogram
        plt.hist(
            values,
            bins=bins,
            color=color if color else "steelblue",
            edgecolor="black",
            alpha=0.8,
        )
        plt.xlabel("Value")
        plt.ylabel("Count")

    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def overlay_histograms(
    split_to_values: List[Tuple[str, np.ndarray]],
    feat_label: str,
    dataset: str,
    model: str,
    out_dir: Path,
    bins: int,
) -> None:
    """
    Overlay histograms for multiple splits on one figure.
    """
    plt.figure(figsize=(7, 4))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for i, (split, vals) in enumerate(split_to_values):
        if vals.size == 0:
            continue
        plt.hist(vals, bins=bins, alpha=0.4, label=split.upper(), color=colors[i % len(colors)])
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.title(f"{dataset} | {model} | {feat_label} distribution by split")
    plt.legend()
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}_{model}_{feat_label}_all.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize input feature distribution by name or index."
    )
    parser.add_argument("--dataset", default="pokec")
    parser.add_argument("--model", default="GCN_MLP")
    parser.add_argument("--dataset-dir", default=os.path.join(SEED_GNN))
    parser.add_argument(
        "--pretrain-dir", default=os.path.join(SEED_GNN, "pretrained_models")
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT, "editing_pipelines", "output"),
        help="Root directory to save plots under dataset_feature_distributions/",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--feature-name", type=str, default=None, help="Feature name")
    group.add_argument(
        "--feature-index", type=int, default=None, help="Feature index (0-based)"
    )

    parser.add_argument(
        "--split",
        choices=["whole", "train", "val", "test", "all"],
        default="whole",
        help="Which subset to visualize",
    )
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--show", action="store_true", help="Also show the figure(s)")
    args = parser.parse_args()

    data = load_dataset(args)
    feat_idx, feat_label = resolve_feature(data, args.feature_name, args.feature_index)
    x = data.x[:, feat_idx].detach().cpu().numpy()

    out_root = Path(args.output_dir) / "dataset_feature_distributions" / f"{args.dataset}_{args.model}"
    out_root.mkdir(parents=True, exist_ok=True)

    if args.split == "all":
        split_vals: List[Tuple[str, np.ndarray]] = []
        for sp in ["train", "val", "test"]:
            try:
                mask = get_mask(data, sp)
            except ValueError:
                # Skip missing masks
                continue
            vals = data.x[mask, feat_idx].detach().cpu().numpy()
            split_vals.append((sp, vals))
            # Save per-split as well
            plot_hist_or_bars(
                vals,
                title=f"{args.dataset} | {args.model} | {feat_label} | {sp.upper()}",
                out_path=out_root / f"{args.dataset}_{args.model}_{feat_label}_{sp}.png",
                bins=args.bins,
            )
        if split_vals:
            overlay_histograms(split_vals, feat_label, args.dataset, args.model, out_root, args.bins)
    else:
        mask = get_mask(data, args.split)
        vals = data.x[mask, feat_idx].detach().cpu().numpy()
        plot_hist_or_bars(
            vals,
            title=f"{args.dataset} | {args.model} | {feat_label} | {args.split.upper()}",
            out_path=out_root / f"{args.dataset}_{args.model}_{feat_label}_{args.split}.png",
            bins=args.bins,
        )

    if args.show:
        # If requested, show the most recently generated plot(s) is not straightforward with Agg,
        # so we skip interactive display in headless mode.
        print("Show requested, but running with non-interactive backend; plots were saved to disk.")


if __name__ == "__main__":
    main()


