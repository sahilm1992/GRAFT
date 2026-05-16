#!/usr/bin/env python3
import argparse
import copy
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# -------------------------------
# Path setup
# -------------------------------
CURRENT_PATH = Path(__file__).resolve()
REPO_ROOT = CURRENT_PATH.parents[1]  # gnn-editing-exploration
SEED_GNN_DIR = REPO_ROOT / "seed-gnn"

sys.path.append(str(SEED_GNN_DIR))
sys.path.append(str(REPO_ROOT))

import main_utils as main_utils  # noqa: E402
from editing_pipelines.utils.model_io import load_model, get_device  # noqa: E402
from edit_gnn.utils import prediction  # noqa: E402


def _parse_model_seed(model_name: str) -> Tuple[str, Optional[int]]:
    match = re.search(r"(?:^|[_-])seed(\d+)", model_name, flags=re.IGNORECASE)
    seed = int(match.group(1)) if match else None
    if match:
        cleaned = re.sub(r"(?:^|[_-])seed\d+", "", model_name, flags=re.IGNORECASE)
        cleaned = cleaned.strip("_-")
    else:
        cleaned = model_name
    return cleaned, seed


def _resolve_feature_index(data, feature_name: Optional[str], feature_index: Optional[int]) -> Tuple[int, str]:
    if feature_index is not None:
        d = int(data.x.size(1))
        if feature_index < 0 or feature_index >= d:
            raise ValueError(f"feature_index {feature_index} out of range [0, {d-1}]")
        label = f"feature_{feature_index}"
        if hasattr(data, "feature_names") and data.feature_names:
            try:
                label = str(data.feature_names[feature_index])
            except Exception:
                pass
        return int(feature_index), label

    if feature_name:
        if hasattr(data, "feature_names") and data.feature_names:
            names = list(data.feature_names)
            if feature_name in names:
                return int(names.index(feature_name)), feature_name
            lower_map = {str(n).lower(): i for i, n in enumerate(names)}
            idx = lower_map.get(feature_name.lower())
            if idx is not None:
                return int(idx), str(names[idx])
        raise ValueError(f"Feature name '{feature_name}' not found in data.feature_names")

    raise ValueError("Provide either --feature-name or --feature-index.")


def _resolve_ablated_root(pretrain_root: Path, dataset: str, feature_name: str) -> Path:
    root_str = pretrain_root.as_posix()
    if "edit_ckpts_feature_ablated" in root_str:
        base = pretrain_root
    elif "edit_ckpts" in root_str:
        base = Path(root_str.replace("edit_ckpts", "edit_ckpts_feature_ablated"))
    else:
        base = pretrain_root.parent / "edit_ckpts_feature_ablated"
    return base / dataset / f"no_{feature_name}" / dataset


def _compute_binned_means(
    feature_vals: np.ndarray, kl_vals: np.ndarray, bins: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    mask = ~np.isnan(feature_vals)
    feat = feature_vals[mask]
    kl = kl_vals[mask]
    if feat.size == 0:
        return np.array([]), np.array([]), np.array([]), []

    unique_vals = np.unique(feat)
    if unique_vals.size <= bins and np.allclose(unique_vals, np.round(unique_vals)):
        centers = unique_vals.astype(float)
        means = np.zeros_like(centers, dtype=float)
        counts = np.zeros_like(centers, dtype=int)
        labels = [str(int(v)) for v in unique_vals]
        for i, v in enumerate(unique_vals):
            m = feat == v
            counts[i] = int(m.sum())
            means[i] = float(np.mean(kl[m])) if counts[i] > 0 else np.nan
        return centers, means, counts, labels

    edges = np.quantile(feat, np.linspace(0.0, 1.0, bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        edges = np.array([np.min(feat), np.max(feat)])
    bin_ids = np.digitize(feat, edges, right=False) - 1
    max_bin = edges.size - 2
    bin_ids = np.clip(bin_ids, 0, max_bin)

    centers = []
    means = []
    counts = []
    labels = []
    for i in range(edges.size - 1):
        m = bin_ids == i
        counts.append(int(m.sum()))
        means.append(float(np.mean(kl[m])) if m.any() else np.nan)
        centers.append(float(0.5 * (edges[i] + edges[i + 1])))
        labels.append(f"{edges[i]:.3g}-{edges[i + 1]:.3g}")
    return np.array(centers), np.array(means), np.array(counts), labels


def _kl_divergence(probs_p: torch.Tensor, probs_q: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    log_p = torch.log(probs_p + eps)
    log_q = torch.log(probs_q + eps)
    return torch.sum(probs_p * (log_p - log_q), dim=1)


def _plot_histogram(kl_vals: np.ndarray, title: str, path: Path, bins: int) -> None:
    plt.figure(figsize=(7, 4))
    plt.hist(kl_vals, bins=bins, edgecolor="black", alpha=0.75)
    plt.xlabel("Per-node KL divergence between model outputs")
    plt.ylabel("Number of nodes")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_scatter_with_avg(
    feat_vals: np.ndarray,
    kl_vals: np.ndarray,
    feat_label: str,
    title: str,
    path: Path,
    bins: int,
) -> None:
    plt.figure(figsize=(7, 4))
    plt.scatter(feat_vals, kl_vals, s=8, alpha=0.35, color="tab:blue")
    centers, means, _counts, _labels = _compute_binned_means(feat_vals, kl_vals, bins)
    if len(centers) > 0:
        plt.plot(centers, means, color="tab:red", linewidth=2, label="Avg (binned)")
        plt.legend(loc="upper left")
    plt.xlabel(f"Feature value ({feat_label})")
    plt.ylabel("Per-node KL divergence between model outputs")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_confidence_by_feature(
    feat_vals: np.ndarray,
    conf_vals: np.ndarray,
    feat_label: str,
    title: str,
    path: Path,
    bins: int,
    correct_mask: Optional[np.ndarray] = None,
) -> None:
    plt.figure(figsize=(7, 4))
    if correct_mask is not None:
        colors = np.where(correct_mask, "green", "red")
        plt.scatter(feat_vals, conf_vals, s=8, alpha=0.35, c=colors)
    else:
        plt.scatter(feat_vals, conf_vals, s=8, alpha=0.35, color="tab:green")
    centers, means, _counts, _labels = _compute_binned_means(feat_vals, conf_vals, bins)
    if len(centers) > 0:
        plt.plot(centers, means, color="tab:red", linewidth=2, label="Avg (binned)")
        plt.legend()
    plt.xlabel(f"Feature value ({feat_label})")
    plt.ylabel("Node confidence (max softmax probability)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_confidence_vs_sensitivity(
    conf_vals: np.ndarray,
    sensitivity_vals: np.ndarray,
    title: str,
    path: Path,
    correct_mask: Optional[np.ndarray] = None,
    color_vals: Optional[np.ndarray] = None,
    color_label: Optional[str] = None,
    color_vmin: Optional[float] = None,
    color_vmax: Optional[float] = None,
) -> None:
    plt.figure(figsize=(6.5, 4.5))
    if color_vals is not None:
        sc = plt.scatter(
            conf_vals,
            sensitivity_vals,
            s=10,
            alpha=0.5,
            c=color_vals,
            cmap="viridis",
            vmin=color_vmin,
            vmax=color_vmax,
        )
        cbar = plt.colorbar(sc)
        if color_label:
            cbar.set_label(color_label)
    elif correct_mask is not None:
        colors = np.where(correct_mask, "green", "red")
        plt.scatter(conf_vals, sensitivity_vals, s=10, alpha=0.4, c=colors)
    else:
        plt.scatter(conf_vals, sensitivity_vals, s=10, alpha=0.4, color="tab:blue")
    plt.xlabel("Node confidence (max softmax probability)")
    plt.ylabel("Sensitivity: KL(perturbed vs unperturbed) full model")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_kl_comparison_scatter(
    kl_a: np.ndarray,
    kl_b: np.ndarray,
    title: str,
    path: Path,
    y_label: Optional[str] = None,
    color_vals: Optional[np.ndarray] = None,
    color_label: Optional[str] = None,
    color_vmin: Optional[float] = None,
    color_vmax: Optional[float] = None,
) -> None:
    plt.figure(figsize=(6, 5))
    if color_vals is not None:
        sc = plt.scatter(
            kl_a,
            kl_b,
            s=10,
            alpha=0.5,
            c=color_vals,
            cmap="viridis",
            vmin=color_vmin,
            vmax=color_vmax,
        )
        cbar = plt.colorbar(sc)
        if color_label:
            cbar.set_label(color_label)
    else:
        plt.scatter(kl_a, kl_b, s=10, alpha=0.4, color="tab:purple")
    min_val = float(np.nanmin([kl_a.min(), kl_b.min()]))
    max_val = float(np.nanmax([kl_a.max(), kl_b.max()]))
    if np.isfinite(min_val) and np.isfinite(max_val):
        plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray")
    plt.xlabel("KL(perturbed vs unperturbed) on full model outputs")
    plt.ylabel(y_label or "KL(perturbed full vs ablated model outputs)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _clone_with_feature(data, feat_idx: int, new_vals: torch.Tensor):
    data_clone = data.clone()
    data_clone.x = data.x.clone()
    data_clone.x[:, feat_idx] = new_vals.to(data_clone.x.device)
    return data_clone


def _build_perturbations(
    feat_vals: np.ndarray,
    is_binary: bool,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    perturbations: Dict[str, np.ndarray] = {}
    if is_binary:
        perturbations["flip"] = 1.0 - feat_vals
    else:
        shuffled = feat_vals.copy()
        rng.shuffle(shuffled)
        perturbations["shuffle"] = shuffled
        std = float(np.nanstd(feat_vals))
        noise_std = std * 0.1 if std > 0 else 1e-3
        perturbations["noise"] = feat_vals + rng.normal(0.0, noise_std, size=feat_vals.shape)
    return perturbations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare full vs feature-ablated checkpoints using KL divergence."
    )
    parser.add_argument("--exp_desc", type=str, default="kl_divergence_analysis")
    parser.add_argument("--pipeline_config_dir", type=str, required=True)
    parser.add_argument("--eval_config_dir", type=str, required=True)
    parser.add_argument("--output_folder_dir", default="results/", type=str)
    parser.add_argument("--job_post_via", default="terminal", type=str)
    parser.add_argument("--pretrain_output_dir", default="ckpts/", type=str)
    parser.add_argument("--dataset_dir", default="datalake/", type=str)
    parser.add_argument("--task", type=str, default="pretrain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--feature-name", type=str, default="AGE")
    parser.add_argument("--feature-index", type=int, default=None)
    parser.add_argument("--bins", type=int, default=50)
    args = parser.parse_args()

    if args.output_folder_dir != "" and args.output_folder_dir[-1] != "/":
        args.output_folder_dir += "/"

    config = main_utils.register_args_and_configs(args)
    _ = main_utils.set_logger(args.output_folder_dir, args)

    model_name = args.model_name or config["pipeline_params"]["model_name"]
    cleaned_model, seed_in_name = _parse_model_seed(model_name)
    config["pipeline_params"]["model_name"] = cleaned_model
    config["management"]["seed"] = seed_in_name if seed_in_name is not None else args.seed

    feature_name = args.feature_name
    pretrain_root = Path(config["management"]["pretrain_output_dir"]).resolve()
    dataset = config["eval_params"]["dataset"]
    ablated_root = _resolve_ablated_root(pretrain_root, dataset, feature_name)

    config_full = copy.deepcopy(config)
    config_full["management"]["pretrain_output_dir"] = str(pretrain_root)
    config_full["pipeline_params"].setdefault("feature_variant", "full_features")

    config_ablated = copy.deepcopy(config)
    config_ablated["management"]["pretrain_output_dir"] = str(ablated_root)
    config_ablated["pipeline_params"]["feature_variant"] = f"no_{feature_name}"
    config_ablated["pipeline_params"]["drop_features"] = [feature_name]

    device = get_device()
    print(f"Using device: {device}")
    print(f"Model: {cleaned_model} | Seed: {config['management']['seed']}")
    print(f"Full ckpts: {pretrain_root}")
    print(f"Ablated ckpts: {ablated_root}")

    model_full, _train_full, data_full, _nf, _nc = load_model(config_full)
    model_ablated, _train_ab, data_ablated, _nf2, _nc2 = load_model(config_ablated)

    if data_full.num_nodes != data_ablated.num_nodes:
        raise ValueError(
            f"Node count mismatch: full={data_full.num_nodes}, ablated={data_ablated.num_nodes}"
        )

    feat_idx, feat_label = _resolve_feature_index(
        data_full, feature_name, args.feature_index
    )

    with torch.no_grad():
        logits_full = prediction(model_full, data_full)
        logits_ablated = prediction(model_ablated, data_ablated)
        probs_full = torch.softmax(logits_full, dim=-1)
        probs_ablated = torch.softmax(logits_ablated, dim=-1)

    kl_vals = _kl_divergence(probs_full, probs_ablated)
    conf_full = probs_full.max(dim=1).values.detach().cpu().numpy()
    conf_ablated = probs_ablated.max(dim=1).values.detach().cpu().numpy()
    pred_full = probs_full.argmax(dim=1)
    pred_ablated = probs_ablated.argmax(dim=1)
    y_true = data_full.y
    correct_full = pred_full.eq(y_true).detach().cpu().numpy()
    correct_ablated = pred_ablated.eq(y_true).detach().cpu().numpy()

    node_ids = torch.arange(data_full.num_nodes, device=kl_vals.device)
    feat_vals = data_full.x[:, feat_idx].detach()

    kl_np = kl_vals.detach().cpu().numpy()
    node_np = node_ids.detach().cpu().numpy()
    feat_np = feat_vals.detach().cpu().numpy()

    out_dir = Path(config["management"]["output_folder_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "Node": node_np,
            "FeatureValue": feat_np,
            "KLDivergence": kl_np,
        }
    )
    df_path = out_dir / f"kl_divergence_nodes_{feat_label}.csv"
    df.to_csv(df_path, index=False)

    hist_path = plot_dir / f"kl_hist_{feat_label}.png"
    _plot_histogram(
        kl_np,
        f"{config['eval_params']['dataset']} | {cleaned_model} | {feat_label}",
        hist_path,
        args.bins,
    )

    centers, means, counts, labels = _compute_binned_means(feat_np, kl_np, args.bins)
    bin_df = pd.DataFrame(
        {
            "Bin": labels,
            "BinCenter": centers,
            "MeanKL": means,
            "Count": counts,
        }
    )
    bin_path = out_dir / f"kl_by_feature_bins_{feat_label}.csv"
    bin_df.to_csv(bin_path, index=False)

    scatter_path = plot_dir / f"kl_scatter_{feat_label}.png"
    _plot_scatter_with_avg(
        feat_np,
        kl_np,
        feat_label,
        f"{config['eval_params']['dataset']} | {cleaned_model} | {feat_label}",
        scatter_path,
        args.bins,
    )

    conf_full_path = plot_dir / f"confidence_scatter_{feat_label}_full.png"
    _plot_confidence_by_feature(
        feat_np,
        conf_full,
        feat_label,
        f"{config['eval_params']['dataset']} | {cleaned_model} | {feat_label} | full model confidence",
        conf_full_path,
        args.bins,
        correct_mask=correct_full,
    )
    conf_abl_path = plot_dir / f"confidence_scatter_{feat_label}_ablated.png"
    _plot_confidence_by_feature(
        feat_np,
        conf_ablated,
        feat_label,
        f"{config['eval_params']['dataset']} | {cleaned_model} | {feat_label} | ablated model confidence",
        conf_abl_path,
        args.bins,
        correct_mask=correct_ablated,
    )

    print(f"Saved per-node KL to: {df_path}")
    print(f"Saved histogram to: {hist_path}")
    print(f"Saved binned KL to: {bin_path}")
    print(f"Saved scatter plot to: {scatter_path}")

    # -----------------------------
    # Perturbation analyses
    # -----------------------------
    feat_vals_full = feat_np
    unique_vals = np.unique(feat_vals_full[~np.isnan(feat_vals_full)])
    is_binary = unique_vals.size <= 2 and np.all(np.isin(unique_vals, [0.0, 1.0]))

    perturbations = _build_perturbations(
        feat_vals_full, is_binary=is_binary, seed=int(config["management"]["seed"])
    )

    feat_idx_ablated = None
    if hasattr(data_ablated, "feature_names"):
        try:
            feat_idx_ablated = list(data_ablated.feature_names).index(feat_label)
        except Exception:
            feat_idx_ablated = None

    for pert_name, pert_vals_np in perturbations.items():
        pert_vals_tensor = torch.tensor(pert_vals_np, dtype=data_full.x.dtype, device=data_full.x.device)
        data_full_pert = _clone_with_feature(data_full, feat_idx, pert_vals_tensor)

        # Ablated model does not include the perturbed feature; use unperturbed ablated data.
        data_ablated_pert = data_ablated

        with torch.no_grad():
            logits_full_pert = prediction(model_full, data_full_pert)
            logits_abl_pert = prediction(model_ablated, data_ablated_pert)
            probs_full_pert = torch.softmax(logits_full_pert, dim=-1)
            probs_abl_pert = torch.softmax(logits_abl_pert, dim=-1)

        kl_pert = _kl_divergence(probs_full_pert, probs_abl_pert)
        kl_pert_np = kl_pert.detach().cpu().numpy()

        out_prefix = f"{feat_label}_{pert_name}"
        df_pert = pd.DataFrame(
            {
                "Node": node_np,
                "FeatureValue": pert_vals_np,
                "KLDivergence": kl_pert_np,
            }
        )
        df_pert_path = out_dir / f"kl_divergence_nodes_{out_prefix}.csv"
        df_pert.to_csv(df_pert_path, index=False)

        hist_path = plot_dir / f"kl_hist_{out_prefix}.png"
        _plot_histogram(
            kl_pert_np,
            f"{config['eval_params']['dataset']} | {cleaned_model} | {feat_label} | {pert_name}",
            hist_path,
            args.bins,
        )

        scatter_path = plot_dir / f"kl_scatter_{out_prefix}.png"
        _plot_scatter_with_avg(
            pert_vals_np,
            kl_pert_np,
            feat_label,
            f"{config['eval_params']['dataset']} | {cleaned_model} | {feat_label} | {pert_name}",
            scatter_path,
            args.bins,
        )

        centers, means, counts, labels = _compute_binned_means(pert_vals_np, kl_pert_np, args.bins)
        bin_df = pd.DataFrame(
            {
                "Bin": labels,
                "BinCenter": centers,
                "MeanKL": means,
                "Count": counts,
            }
        )
        bin_path = out_dir / f"kl_by_feature_bins_{out_prefix}.csv"
        bin_df.to_csv(bin_path, index=False)

        kl_pert_vs_unpert = _kl_divergence(probs_full_pert, probs_full).detach().cpu().numpy()
        kl_pert_vs_abl = _kl_divergence(probs_full_pert, probs_abl_pert).detach().cpu().numpy()
        compare_path = plot_dir / f"kl_compare_{out_prefix}.png"
        age_vmin = None
        age_vmax = None
        if feat_label.upper() == "AGE":
            age_vmin = float(np.nanmin(feat_vals_full))
            age_vmax = 50.0

        _plot_kl_comparison_scatter(
            kl_pert_vs_unpert,
            kl_pert_vs_abl,
            f"{config['eval_params']['dataset']} | {cleaned_model} | {feat_label} | {pert_name}",
            compare_path,
            color_vals=feat_vals_full,
            color_label=f"{feat_label} value",
            color_vmin=age_vmin,
            color_vmax=age_vmax,
        )

        kl_unpert_vs_abl = _kl_divergence(probs_full, probs_ablated).detach().cpu().numpy()
        compare_unpert_path = plot_dir / f"kl_compare_unpert_full_vs_abl_{out_prefix}.png"
        _plot_kl_comparison_scatter(
            kl_pert_vs_unpert,
            kl_unpert_vs_abl,
            f"{config['eval_params']['dataset']} | {cleaned_model} | {feat_label} | {pert_name} | unpert vs ablated",
            compare_unpert_path,
            y_label="KL(unperturbed full vs ablated model outputs)",
            color_vals=feat_vals_full,
            color_label=f"{feat_label} value",
            color_vmin=age_vmin,
            color_vmax=age_vmax,
        )

        if pert_name in {"shuffle", "noise"}:
            pert_scatter_path = plot_dir / f"kl_scatter_{out_prefix}_pert_vs_unpert.png"
            _plot_scatter_with_avg(
                pert_vals_np,
                kl_pert_vs_unpert,
                feat_label,
                f"{config['eval_params']['dataset']} | {cleaned_model} | {feat_label} | {pert_name} | pert vs unpert",
                pert_scatter_path,
                args.bins,
            )

        conf_sens_path = plot_dir / f"confidence_vs_sensitivity_{out_prefix}.png"
        age_vmin = None
        age_vmax = None
        if feat_label.upper() == "AGE":
            age_vmin = float(np.nanmin(feat_vals_full))
            age_vmax = 50.0
        _plot_confidence_vs_sensitivity(
            conf_full,
            kl_pert_vs_unpert,
            f"{config['eval_params']['dataset']} | {cleaned_model} | {feat_label} | {pert_name} | conf vs sensitivity",
            conf_sens_path,
            color_vals=feat_vals_full,
            color_label=f"{feat_label} value",
            color_vmin=age_vmin,
            color_vmax=age_vmax,
        )


if __name__ == "__main__":
    main()
