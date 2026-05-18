def visualize_validation(config, model, whole_data, node_idx_2flip, suffix: str = "", file_suffix: str = ""):
    from pipelines.seed_gnn.utils import visualize_validation as seed_visualize_validation
    return seed_visualize_validation(config, model, whole_data, node_idx_2flip, suffix, file_suffix)


# --- Dependencies ---
import os
from typing import Optional
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from edit_gnn.utils import prediction


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _misclass_rate_by_attr_per_class(logits, y, x, mask, num_classes):
    """Compute misclassification rates per attribute value, overall and per class."""
    idx = mask.nonzero(as_tuple=True)[0]
    logits = logits[idx]
    y = y[idx]
    x = x[idx]
    y_pred = logits.argmax(dim=-1)
    mis = (y_pred != y).to(torch.long)

    result = {}

    for feat in range(x.shape[1]):
        feat_vals = x[:, feat]
        unique_vals = torch.unique(feat_vals)

        # --- Handle continuous vs categorical ---
        if len(unique_vals) > 20:
            feat_float = feat_vals.float()
            bins = torch.quantile(
                feat_float, torch.linspace(0, 1, 6).to(feat_float.device)
            ).unique()
            bins = bins.contiguous()

            # bucketize → indices in [0, len(bins)]
            bin_idx = torch.bucketize(feat_float.contiguous(), bins, right=False)
            # clamp indices to be valid for label count
            bin_idx = bin_idx.clamp(1, len(bins) - 1)
            group_vals = bin_idx
            label_vals = [f"{float(bins[i-1]):.2f}-{float(bins[i]):.2f}" for i in range(1, len(bins))]
        else:
            group_vals = feat_vals
            label_vals = [str(v.item()) for v in unique_vals]

        n_groups = len(label_vals)
        overall_rates = np.full(n_groups, np.nan)
        per_class_rates = np.full((num_classes, n_groups), np.nan)

        # Compute for each group label index
        for j in range(n_groups):
            mask_val = group_vals == (j + 1 if len(unique_vals) > 20 else unique_vals[j])
            if mask_val.sum() == 0:
                continue

            overall_rates[j] = mis[mask_val].float().mean().item()

            for c in range(num_classes):
                mask_cls = y[mask_val] == c
                if mask_cls.sum() == 0:
                    continue
                mis_cls = mis[mask_val][mask_cls]
                per_class_rates[c, j] = mis_cls.float().mean().item()

        result[feat] = {
            "labels": label_vals,
            "overall": overall_rates.tolist(),
            "per_class": per_class_rates,
        }

    return result

def plot_misclassification_by_attributes_before_after(config,
                                                      model_before,
                                                      model_after,
                                                      whole_data,
                                                      method_name,
                                                      model_name,
                                                      file_suffix=''):
    """
    Generates:
      (1) Overall misclassification rate per attribute-value (Before vs After)
      (2) Per-class misclassification rate per attribute-value
    Adds faint histogram bars showing the proportion of nodes per attribute value.
    """
    out_dir = Path(config['management']['output_folder_dir']) / 'visualization_plots' / f"{config['eval_params']['dataset']}_{model_name}" / 'before-after-edit'
    _ensure_dir(out_dir)

    with torch.no_grad():
        logits_b = prediction(model_before, whole_data)
        logits_a = prediction(model_after, whole_data)

    for split, mask in [('val', whole_data.val_mask), ('test', whole_data.test_mask)]:
        num_classes = logits_b.shape[1]
        rates_b = _misclass_rate_by_attr_per_class(logits_b, whole_data.y, whole_data.x, mask, num_classes)
        rates_a = _misclass_rate_by_attr_per_class(logits_a, whole_data.y, whole_data.x, mask, num_classes)

        n_feats = len(rates_b)
        feats_per_fig = 25
        feature_names = getattr(whole_data, 'feature_names', [f"f{i}" for i in range(n_feats)])

        # --- (1) OVERALL BEFORE/AFTER ---
        for start in range(0, n_feats, feats_per_fig):
            end = min(start + feats_per_fig, n_feats)
            subset = range(start, end)
            nrows = int(np.ceil(len(subset) / 5))
            fig, axs = plt.subplots(nrows, 5, figsize=(20, 3 * nrows))
            axs = axs.flatten()

            for i, feat in enumerate(subset):
                d_b = rates_b[feat]
                d_a = rates_a[feat]
                val_labels = d_b["labels"]
                ax = axs[i]

                # ---- (NEW) Histogram proportions ----
                feat_vals = whole_data.x[:, feat][mask].detach().cpu()   # <-- moved to CPU
                if len(val_labels) > 1:
                    if len(torch.unique(feat_vals)) > 20:
                        # same binning as before
                        bins = torch.quantile(
                            feat_vals.float(), torch.linspace(0, 1, len(val_labels) + 1)
                        ).unique()
                        hist_counts = torch.histc(
                            feat_vals.float(), bins=len(val_labels), min=bins[0].item(), max=bins[-1].item()
                        )
                    else:
                        # categorical histogram
                        unique_vals, counts = torch.unique(feat_vals, return_counts=True)
                        hist_counts = torch.tensor([
                            counts[unique_vals == v].item() if v in unique_vals else 0
                            for v in torch.unique(feat_vals)
                        ])
                    hist_prop = (hist_counts / hist_counts.sum()).cpu().numpy()
                else:
                    hist_prop = np.array([1.0])

                # secondary y-axis for histogram
                ax_hist = ax.twinx()
                ax_hist.bar(
                    np.arange(len(val_labels)),
                    hist_prop,
                    color='gray',
                    alpha=0.15,
                    width=0.8,
                    zorder=0
                )
                ax_hist.set_ylim(0, max(hist_prop) * 1.1)
                ax_hist.set_yticks([])

                # ---- (Existing) Misclassification lines ----
                ax.plot(d_b["overall"], label='Before', marker='o', zorder=2)
                ax.plot(d_a["overall"], label='After', marker='x', zorder=3)
                ax.set_title(feature_names[feat], fontsize=9)
                ax.set_ylim(0, 1.0)
                ax.set_xticks(np.arange(len(val_labels)))
                ax.set_xticklabels(val_labels, rotation=45, fontsize=6)
                if i % 5 == 0:
                    ax.set_ylabel("Misclass Rate")
                ax.legend(fontsize=6)

            for j in range(i + 1, len(axs)):
                axs[j].axis('off')

            fig.suptitle(f"{split.upper()} Overall Misclassification (Before vs After) [{start}-{end})", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(out_dir / f"{split}_miscls_overall_{method_name}_{start}-{end}{file_suffix}.png", bbox_inches='tight')
            plt.close(fig)

        # --- (2) PER-CLASS MISCLASSIFICATION ---
        for start in range(0, n_feats, feats_per_fig):
            end = min(start + feats_per_fig, n_feats)
            subset = range(start, end)
            nrows = int(np.ceil(len(subset) / 5))
            fig, axs = plt.subplots(nrows, 5, figsize=(20, 3 * nrows))
            axs = axs.flatten()

            for i, feat in enumerate(subset):
                d_b = rates_b[feat]
                d_a = rates_a[feat]
                val_labels = d_b["labels"]
                ax = axs[i]

                # ---- (NEW) Histogram proportions ----
                feat_vals = whole_data.x[:, feat][mask].detach().cpu()   # <-- moved to CPU
                if len(val_labels) > 1:
                    if len(torch.unique(feat_vals)) > 20:
                        bins = torch.quantile(
                            feat_vals.float(), torch.linspace(0, 1, len(val_labels) + 1)
                        ).unique()
                        hist_counts = torch.histc(
                            feat_vals.float(), bins=len(val_labels), min=bins[0].item(), max=bins[-1].item()
                        )
                    else:
                        unique_vals, counts = torch.unique(feat_vals, return_counts=True)
                        hist_counts = torch.tensor([
                            counts[unique_vals == v].item() if v in unique_vals else 0
                            for v in torch.unique(feat_vals)
                        ])
                    hist_prop = (hist_counts / hist_counts.sum()).cpu().numpy()
                else:
                    hist_prop = np.array([1.0])

                ax_hist = ax.twinx()
                ax_hist.bar(
                    np.arange(len(val_labels)),
                    hist_prop,
                    color='gray',
                    alpha=0.15,
                    width=0.8,
                    zorder=0
                )
                ax_hist.set_ylim(0, max(hist_prop) * 1.1)
                ax_hist.set_yticks([])

                # ---- (Existing) Per-class lines ----
                for c in range(num_classes):
                    cls_b = np.nan_to_num(d_b["per_class"][c], nan=0.0)
                    cls_a = np.nan_to_num(d_a["per_class"][c], nan=0.0)
                    ax.plot(cls_b, linestyle='--', label=f"Class {c} (B)", zorder=2)
                    ax.plot(cls_a, linestyle='-', label=f"Class {c} (A)", zorder=3)

                ax.set_title(feature_names[feat], fontsize=9)
                ax.set_ylim(0, 1.0)
                ax.set_xticks(np.arange(len(val_labels)))
                ax.set_xticklabels(val_labels, rotation=45, fontsize=6)
                if i % 5 == 0:
                    ax.set_ylabel("Misclass Rate per Class")
                ax.legend(fontsize=6, ncol=2)

            for j in range(i + 1, len(axs)):
                axs[j].axis('off')

            fig.suptitle(f"{split.upper()} Per-Class Misclassification (Before vs After) [{start}-{end})", fontsize=11)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(out_dir / f"{split}_miscls_perclass_{method_name}_{start}-{end}{file_suffix}.png", bbox_inches='tight')
            plt.close(fig)


def plot_targeted_edits_distribution(config,
                                     edited_node_idx: torch.Tensor,
                                     whole_data,
                                     method_name,
                                     model_name,
                                     file_suffix=''):
    """
    Plot the distribution of edited samples across attribute values (overall and per-class).
    Saves figures into visualization_plots/<dataset>_<model>/targeted_edits.
    """
    out_dir = Path(config['management']['output_folder_dir']) / 'visualization_plots' / f"{config['eval_params']['dataset']}_{model_name}" / 'targeted_edits'
    _ensure_dir(out_dir)

    idx = edited_node_idx.squeeze()
    if idx.ndim == 0:
        idx = idx.unsqueeze(0)
    idx = idx.detach().cpu()

    x_sel = whole_data.x[idx]
    y_sel = whole_data.y[idx]

    all_classes = torch.unique(whole_data.y).cpu().tolist()
    n_feats = x_sel.shape[1]
    feats_per_fig = 25
    feature_names = getattr(whole_data, 'feature_names', [f"f{i}" for i in range(n_feats)])

    def grouped_props(feat_vals: torch.Tensor):
        unique_vals = torch.unique(feat_vals)
        if len(unique_vals) > 20:
            feat_float = feat_vals.float()
            bins = torch.quantile(
                feat_float, torch.linspace(0, 1, 6).to(feat_float.device)
            ).unique()
            bins = bins.contiguous()
            bin_idx = torch.bucketize(feat_float.contiguous(), bins, right=False)
            bin_idx = bin_idx.clamp(1, len(bins) - 1)
            labels = [f"{float(bins[i-1]):.2f}-{float(bins[i]):.2f}" for i in range(1, len(bins))]
            groups = bin_idx
            label_keys = list(range(1, len(labels) + 1))
        else:
            labels = [str(v.item()) for v in unique_vals]
            groups = feat_vals
            label_keys = unique_vals.tolist()
        return groups, labels, label_keys

    # Overall
    for start in range(0, n_feats, feats_per_fig):
        end = min(start + feats_per_fig, n_feats)
        subset = range(start, end)
        nrows = int(np.ceil(len(subset) / 5))
        fig, axs = plt.subplots(nrows, 5, figsize=(20, 3 * nrows))
        axs = axs.flatten()

        for i, feat in enumerate(subset):
            ax = axs[i]
            feat_vals = x_sel[:, feat].detach().cpu()
            groups, labels, label_keys = grouped_props(feat_vals)
            props = []
            for key in label_keys:
                mask = groups == key
                props.append((mask.sum().item() / max(1, groups.numel())))
            ax.bar(np.arange(len(labels)), props, color='steelblue', edgecolor='black')
            ax.set_title(feature_names[feat], fontsize=9)
            ax.set_ylim(0, 1.0)
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, fontsize=6)
            if i % 5 == 0:
                ax.set_ylabel("Proportion (overall)")

        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        fig.suptitle(f"Targeted Edits - Overall Distribution [{start}-{end})", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_dir / f"overall_targeted_distribution_{method_name}_{start}-{end}{file_suffix}.png", bbox_inches='tight')
        plt.close(fig)

    # Per-class
    for start in range(0, n_feats, feats_per_fig):
        end = min(start + feats_per_fig, n_feats)
        subset = range(start, end)
        nrows = int(np.ceil(len(subset) / 5))
        fig, axs = plt.subplots(nrows, 5, figsize=(20, 3 * nrows))
        axs = axs.flatten()

        for i, feat in enumerate(subset):
            ax = axs[i]
            feat_vals = x_sel[:, feat].detach().cpu()
            groups, labels, label_keys = grouped_props(feat_vals)

            bottoms = np.zeros(len(labels), dtype=float)
            for c in all_classes:
                class_mask = (y_sel.cpu() == c)
                if class_mask.sum().item() == 0:
                    continue
                props = []
                for key in label_keys:
                    mask = (groups == key) & class_mask
                    denom = max(1, class_mask.sum().item())
                    props.append(mask.sum().item() / denom)
                ax.bar(np.arange(len(labels)), props, bottom=bottoms, label=f"Class {int(c)}", alpha=0.7)
                bottoms += np.array(props)

            ax.set_title(feature_names[feat], fontsize=9)
            ax.set_ylim(0, 1.0)
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, fontsize=6)
            if i % 5 == 0:
                ax.set_ylabel("Proportion (per class)")
            ax.legend(fontsize=6, ncol=2)

        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        fig.suptitle(f"Targeted Edits - Per-Class Distribution [{start}-{end})", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_dir / f"perclass_targeted_distribution_{method_name}_{start}-{end}{file_suffix}.png", bbox_inches='tight')
        plt.close(fig)


def plot_validation_correct_confidence_histogram(config,
                                                 model_before,
                                                 model_after,
                                                 whole_data,
                                                 method_name,
                                                 model_name,
                                                 file_suffix=''):
    """
    Plot histogram of softmax probability margin on validation set for ALL nodes (correct and incorrect):
        prob_margin = softmax(logits)[true_class] - max_other_softmax
    Overlays Before vs After distributions.
    """
    out_dir = Path(config['management']['output_folder_dir']) / 'visualization_plots' / f"{config['eval_params']['dataset']}_{model_name}" / 'before-after-edit'
    _ensure_dir(out_dir)

    with torch.no_grad():
        logits_b = prediction(model_before, whole_data)
        logits_a = prediction(model_after, whole_data)

    mask = whole_data.val_mask
    idx = mask.nonzero(as_tuple=True)[0]
    val_logits_b = logits_b[idx]
    val_logits_a = logits_a[idx]
    y_true = whole_data.y[idx]
    row = torch.arange(y_true.size(0), device=val_logits_b.device)

    def prob_margins(val_logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(val_logits, dim=1)
        true_prob = probs[row, y_true]
        masked = probs.clone()
        masked[row, y_true] = float('-inf')
        max_other, _ = masked.max(dim=1)
        return (true_prob - max_other).detach().cpu()

    m_b = prob_margins(val_logits_b).numpy()
    m_a = prob_margins(val_logits_a).numpy()

    # Common binning across both distributions
    combined = np.concatenate([m_b, m_a])
    if combined.size == 0:
        return
    bins = min(50, max(10, int(np.sqrt(combined.size))))

    plt.figure(figsize=(8, 5))
    plt.hist(m_b, bins=bins, alpha=0.5, label='Before', color='tab:blue', edgecolor='black')
    plt.hist(m_a, bins=bins, alpha=0.5, label='After', color='tab:orange', edgecolor='black')
    plt.xlabel('Probability margin (p_true - max_other_p)')
    plt.ylabel('Count')
    plt.title('Validation Probability-Margin Histogram (Before vs After)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"val_prob_margin_hist_{method_name}{file_suffix}.png", bbox_inches='tight')
    plt.close()


# ============================
# Plotting helpers (general)
# ============================
def plot_auc_by_feature_with_counts(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    feature_name: str,
    save_dir: str,
    suffix: str = "",
):
    """
    Plot AUC vs feature bucket with sample counts (Before vs After).

    Assumptions:
      - df_before, df_after each have columns:
            [BinCol, "AUC", "Count"]
        where BinCol is "AgeBin" (old naming) or "FeatureBin" (new naming).

    We:
      - merge before/after on the bin column,
      - keep both AUC_before/AUC_after and Count_before/Count_after (if present),
      - plot Count_before as bars + AUC lines.
    """

    # Determine the name of the bin column (case-insensitive support)
    possible_bin_cols = ["AgeBin", "AGEBin", "FeatureBin", f"{feature_name}Bin"]

    bin_col = None
    for col in df_before.columns:
        if col in possible_bin_cols:
            bin_col = col
            break

    if bin_col is None:
        raise ValueError(
            f"Could not find bin column in df_before; columns={df_before.columns}"
        )


    out_path = Path(save_dir)
    _ensure_dir(out_path)

    # Try to merge with Count from both before & after
    if "Count" in df_after.columns:
        df_after_sub = df_after[[bin_col, "AUC", "Count"]]
    else:
        df_after_sub = df_after[[bin_col, "AUC"]]

    merged = df_before.merge(
        df_after_sub,
        on=bin_col,
        how="inner",
        suffixes=("_before", "_after"),
    )

    # Figure out which count column to use
    if "Count_before" in merged.columns:
        count_col = "Count_before"
    elif "Count" in merged.columns:
        count_col = "Count"
    else:
        count_col = None

    # Filter out bins where both AUCs are NaN or zero
    auc_b = merged["AUC_before"].fillna(0.0)
    auc_a = merged["AUC_after"].fillna(0.0)
    mask = (auc_b > 0) | (auc_a > 0)
    filtered = merged[mask].reset_index(drop=True)

    if len(filtered) == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No nonzero AUC bins to plot", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(
            out_path / f"plot_auc_by_{feature_name.lower()}{suffix}.png",
            dpi=150,
        )
        plt.close(fig)
        return

    fig, ax1 = plt.subplots(figsize=(7, 4))
    x = np.arange(len(filtered))

    # Count bars (if available)
    if count_col is not None:
        ax1.bar(
            x,
            filtered[count_col],
            color="gray",
            alpha=0.25,
            label="Count (samples)",
        )
        ax1.set_ylabel("Count", color="gray")
        ax1.tick_params(axis="y", labelcolor="gray")

    # AUC lines
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        filtered["AUC_before"],
        marker="o",
        linewidth=2,
        label="Before Edit",
    )
    ax2.plot(
        x,
        filtered["AUC_after"],
        marker="o",
        linewidth=2,
        label="After Edit",
    )
    ax2.set_ylabel("AUC-ROC")
    ymin = min(
        float(np.nanmin(filtered["AUC_before"])),
        float(np.nanmin(filtered["AUC_after"])),
    )
    ax2.set_ylim(ymin, 1.0)

    ax1.set_xticks(x)
    ax1.set_xticklabels(filtered[bin_col], rotation=20, ha="right")
    ax2.legend(loc="upper left")

    ax1.set_title(
        f"AUC vs {feature_name} with Sample Counts (nonzero AUC bins)"
    )
    fig.tight_layout()
    fig.savefig(
        out_path / f"plot_auc_by_{feature_name.lower()}{suffix}.png",
        dpi=150,
    )
    plt.close(fig)

def plot_mean_prob_by_feature(
    mm_before: pd.DataFrame,
    mm_after: pd.DataFrame,
    feature_name: str,
    stat: str,
    save_dir: str,
    suffix: str = "",
    prob_label: str = "p(y=1)",
):
    """
    Plot mean/median probability vs binned sensitive feature.

    Expects mm_before / mm_after to have columns:
        f"{feature_name}Bin", "MeanProb", "MedianProb", "Count"
    """
    bin_col = f"{feature_name}Bin"
    stat_col = "MeanProb" if stat == "mean" else "MedianProb"

    merged = mm_before.merge(
        mm_after[[bin_col, stat_col, "Count"]],
        on=bin_col,
        how="inner",
        suffixes=("_before", "_after"),
    )

    out_path = Path(save_dir)
    _ensure_dir(out_path)

    if len(merged) == 0:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No bins to plot", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(
            out_path / f"plot_{feature_name.lower()}_{stat}_prob_by_feature{suffix}.png",
            dpi=150,
        )
        plt.close(fig)
        return

    fig, ax1 = plt.subplots(figsize=(7, 4))
    x = np.arange(len(merged))

    # Counts as bars
    ax1.bar(x, merged["Count_before"], color="gray", alpha=0.25, label="Count (samples)")
    ax1.set_ylabel("Count", color="gray")
    ax1.tick_params(axis="y", labelcolor="gray")

    # Mean/median prob as lines
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        merged[f"{stat_col}_before"],
        marker="o",
        linewidth=2,
        label="Before",
        color="tab:blue",
    )
    ax2.plot(
        x,
        merged[f"{stat_col}_after"],
        marker="o",
        linewidth=2,
        label="After",
        color="tab:orange",
    )
    ax2.set_ylabel(("Median " if stat == "median" else "Mean ") + prob_label)
    ax2.set_ylim(0.0, 1.0)

    ax1.set_xticks(x)
    ax1.set_xticklabels(merged[bin_col], rotation=20, ha="right")
    ax2.legend(loc="upper left")
    ax1.set_title(
        f"{'Median' if stat == 'median' else 'Mean'} {prob_label} by {feature_name}"
    )

    fig.tight_layout()
    fig.savefig(
        out_path / f"plot_{feature_name.lower()}_{stat}_prob_by_feature{suffix}.png",
        dpi=150,
    )
    plt.close(fig)

def plot_rep_and_aug_distributions(
    model,
    data,
    representative_indices,
    X_aug_all,
    feature_name: str,
    save_dir: str,
):
    """
    Plot distributions of:
      - sensitive feature among representatives
      - class labels among representatives
      - true-class probabilities among representatives
      - (optionally) sensitive feature under augmentations for reps
    """
    out_path = Path(save_dir)
    _ensure_dir(out_path)
    device = data.x.device

    if not torch.is_tensor(representative_indices):
        rep_idx = torch.as_tensor(representative_indices, dtype=torch.long, device=device)
    else:
        rep_idx = representative_indices.to(device=device, dtype=torch.long)

    feat_idx = data.feature_names.index(feature_name)
    feat_rep = data.x[rep_idx, feat_idx].detach().cpu().float().numpy()
    y_rep = data.y[rep_idx].detach().cpu().numpy()

    logits = prediction(model, data)
    probs = torch.softmax(logits, dim=-1)
    true_probs = probs[rep_idx, data.y[rep_idx]].detach().cpu().numpy()

    # 1) Sensitive feature distribution among reps
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(feat_rep, bins=20, color="tab:blue", alpha=0.7, edgecolor="black")
    ax.set_xlabel(feature_name)
    ax.set_ylabel("Count")
    ax.set_title(f"Representative {feature_name} distribution")
    fig.tight_layout()
    fig.savefig(out_path / f"rep_{feature_name.lower()}_hist.png", dpi=150)
    plt.close(fig)

    # 2) True label distribution among reps
    fig, ax = plt.subplots(figsize=(6, 4))
    classes, counts = np.unique(y_rep, return_counts=True)
    ax.bar(classes.astype(str), counts, color="tab:green", edgecolor="black")
    ax.set_xlabel("True Class Label")
    ax.set_ylabel("Count")
    ax.set_title("Representative True Label Distribution")
    fig.tight_layout()
    fig.savefig(out_path / "rep_true_label_counts.png", dpi=150)
    plt.close(fig)

    # 3) True-class probability distribution among reps
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(true_probs, bins=20, color="tab:orange", alpha=0.8, edgecolor="black")
    ax.set_xlabel("True-class probability")
    ax.set_ylabel("Count")
    ax.set_title("Representative Confidence (True-class Prob)")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path / "rep_true_confidence_hist.png", dpi=150)
    plt.close(fig)

    # 4) Feature distribution under augmentations (for reps only)
    if X_aug_all and isinstance(X_aug_all, list) and len(X_aug_all) > 0:
        aug_vals_concat = []
        rep_np = rep_idx.detach().cpu().numpy()
        for X_aug in X_aug_all:
            feat_aug = X_aug[:, feat_idx].detach().cpu().float().numpy()
            aug_vals_concat.append(feat_aug[rep_np])
        if len(aug_vals_concat) > 0:
            aug_vals_concat = np.concatenate(aug_vals_concat, axis=0)
            if aug_vals_concat.size > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(
                    aug_vals_concat,
                    bins=20,
                    color="tab:purple",
                    alpha=0.7,
                    edgecolor="black",
                )
                ax.set_xlabel(feature_name)
                ax.set_ylabel("Count")
                ax.set_title(
                    f"Augmented {feature_name} distribution (representatives × K)"
                )
                fig.tight_layout()
                fig.savefig(out_path / f"aug_{feature_name.lower()}_hist.png", dpi=150)
                plt.close(fig)


def plot_feature_sensitivity(
    val_df_before: pd.DataFrame,
    val_df_after: pd.DataFrame,
    test_df_before: pd.DataFrame,
    test_df_after: pd.DataFrame,
    feature_name: str,
    save_dir: str,
):
    """
    Plot histograms of variance of true-class probabilities under
    sensitive-feature perturbations, for VAL and TEST splits.
    """
    out_path = Path(save_dir)
    _ensure_dir(out_path)

    def plot_split(df_b, df_a, split):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(df_b["VarProb"], bins=40, alpha=0.5, label="Before")
        ax.hist(df_a["VarProb"], bins=40, alpha=0.5, label="After")
        ax.set_xlabel(
            f"Variance of true-class probabilities across {feature_name} perturbations"
        )
        ax.set_ylabel("# Nodes")
        ax.set_title(f"{split} – {feature_name} Sensitivity Distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            out_path / f"plot_{feature_name.lower()}_sensitivity_{split.lower()}.png",
            dpi=150,
        )
        plt.close(fig)

    plot_split(val_df_before, val_df_after, "VAL")
    plot_split(test_df_before, test_df_after, "TEST")


def plot_sensitivity_reduction(
    val_before: pd.DataFrame,
    val_after: pd.DataFrame,
    test_before: pd.DataFrame,
    test_after: pd.DataFrame,
    feature_name: str,
    save_dir: str,
):
    """
    Plot and save per-node change in variance (After - Before) under
    sensitive-feature perturbations, for VAL and TEST splits.
    """
    out_path = Path(save_dir)
    _ensure_dir(out_path)

    def process(df_b, df_a, split):
        merged = pd.merge(df_b, df_a, on="Node", suffixes=("_before", "_after"))
        merged["DeltaVar"] = merged["VarProb_after"] - merged["VarProb_before"]

        # Summary statistics
        if len(merged) > 0:
            mean_before = float(merged["VarProb_before"].mean())
            mean_after = float(merged["VarProb_after"].mean())
            mean_delta = mean_after - mean_before
            reduced_pct = float((merged["DeltaVar"] < 0).mean() * 100.0)
        else:
            mean_before = mean_after = mean_delta = float("nan")
            reduced_pct = float("nan")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(merged["DeltaVar"], bins=50, alpha=0.7)
        ax.axvline(0, color="k", linestyle="--")
        ax.set_xlabel("Δ Variance (After - Before)")
        ax.set_ylabel("# Nodes")
        ax.set_title(f"{split} – {feature_name} Sensitivity Change per Node")
        if mean_before == 0:
            percent_reduction = 0.0 if mean_delta == 0 else float("inf")
        else:
            percent_reduction = (mean_delta / mean_before) * 100
        summary_text = (
            f"mean_before={mean_before:.6f}\n"
            f"mean_after={mean_after:.6f}\n"
            f"mean_Δ={mean_delta:+.6f}\n"
            f"reduced var nodes={reduced_pct:.1f}%\n"
            f"percent_var_reduction={percent_reduction:.6f}%"
        )
        ax.text(
            0.98,
            0.98,
            summary_text,
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            ha="right",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.7,
                edgecolor="gray",
            ),
        )
        fig.tight_layout()
        fig.savefig(
            out_path / f"sensitivity_reduction_{feature_name.lower()}_{split.lower()}.png",
            dpi=150,
        )
        plt.close(fig)

        merged.to_csv(
            out_path / f"sensitivity_reduction_{feature_name.lower()}_{split.lower()}.csv",
            index=False,
        )
        print(
            f"[{split}] {feature_name} | "
            f"mean_before={mean_before:.6f}, mean_after={mean_after:.6f}, "
            f"mean_Δ={mean_delta:+.6f}, pct_Δ_var={percent_reduction:+.2f}%, "
            f"reduced_node_pct={reduced_pct:.1f}%"
        )

    process(val_before, val_after, "VAL")
    process(test_before, test_after, "TEST")


def plot_degree_vs_sensitivity(
    deg: torch.Tensor,
    delta_probs: torch.Tensor,
    conf: torch.Tensor,
    save_dir: str,
):
    """
    Plots:
      1) Degree vs sensitivity (Δprob)
      2) Degree vs confidence
    """

    os.makedirs(save_dir, exist_ok=True)

    # -------- Plot 1: Degree vs Sensitivity --------
    plt.figure(figsize=(6, 5))
    plt.scatter(deg, delta_probs, s=10, alpha=0.4)
    plt.xlabel("Node Degree")
    plt.ylabel("Sensitivity (Δ Probability)")
    plt.title("Degree vs Sensitivity")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "degree_vs_sensitivity.png"), dpi=150)
    plt.close()

    # -------- Plot 2: Degree vs Confidence --------
    plt.figure(figsize=(6, 5))
    plt.scatter(deg, conf, s=10, alpha=0.4)
    plt.xlabel("Node Degree")
    plt.ylabel("Confidence (max softmax)")
    plt.title("Degree vs Confidence")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "degree_vs_confidence.png"), dpi=150)
    plt.close()

def plot_transition_subset_distributions(
    split_name: str,
    subset_nodes: dict,
    feature_values: np.ndarray,
    degrees: np.ndarray,
    logits_before: np.ndarray,
    logits_after: np.ndarray,
    prob_before: np.ndarray,
    prob_after: np.ndarray,
    var_before_df: Optional[pd.DataFrame],
    var_after_df: Optional[pd.DataFrame],
    feature_name: str,
    save_dir: str,
    representative_examples: Optional[torch.Tensor] = None,
) -> None:
    if not subset_nodes:
        return
    subset_colors = {
        "0->0": "#781C6D",
        "0->1": "#1a9850",
        "1->0": "#EA7317",
    }
    subset_labels = {
        "0->0": "Wrong → Wrong",
        "0->1": "Wrong → Correct",
        "1->0": "Correct → Wrong",
    }
    subset_order = ["0->0", "0->1", "1->0"]
    subset_idx = {
        key: tensor.detach().cpu().numpy().astype(int)
        for key, tensor in subset_nodes.items()
        if tensor.numel() > 0
    }
    if not subset_idx:
        return

    output_dir = os.path.join(save_dir, "transition_analysis")
    os.makedirs(output_dir, exist_ok=True)

    def _plot_hist(values: np.ndarray, title: str, filename: str, bins: int = 40):
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted = False
        for subset in subset_order:
            idx = subset_idx.get(subset)
            if idx is None or idx.size == 0:
                continue
            if subset == "0->0":
                continue
            ax.hist(
                values[idx],
                bins=bins,
                alpha=0.6,
                label=subset_labels.get(subset, subset),
                color=subset_colors.get(subset, "#555555"),
            )
            plotted = True
        if not plotted:
            plt.close(fig)
            return
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close(fig)

    def _plot_scatter(x_vals, y_vals, xlabel, ylabel, title, filename):
        fig, ax = plt.subplots(figsize=(6, 4))
        plotted = False
        for subset in subset_order:
            idx = subset_idx.get(subset)
            if idx is None or idx.size == 0:
                continue
            if subset == "0->0":
                continue
            ax.scatter(
                x_vals[idx],
                y_vals[idx],
                alpha=0.5,
                s=12,
                label=subset_labels.get(subset, subset),
                color=subset_colors.get(subset, "#555555"),
            )
            plotted = True
        if not plotted:
            plt.close(fig)
            return
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close(fig)

    _plot_hist(
        feature_values,
        title=f"{split_name} | {feature_name} distribution (transition subsets)",
        filename=f"{split_name.lower()}_{feature_name.lower()}_hist.png",
    )
    _plot_hist(
        degrees,
        title=f"{split_name} | Degree distribution (transition subsets)",
        filename=f"{split_name.lower()}_degree_hist.png",
    )
    _plot_hist(
        prob_before,
        title=f"{split_name} | True-class probability BEFORE",
        filename=f"{split_name.lower()}_prob_before_hist.png",
    )
    _plot_hist(
        prob_after,
        title=f"{split_name} | True-class probability AFTER",
        filename=f"{split_name.lower()}_prob_after_hist.png",
    )

    _plot_scatter(
        feature_values,
        prob_before,
        xlabel=feature_name,
        ylabel="p(true class) BEFORE",
        title=f"{split_name} | {feature_name} vs p(true class) BEFORE",
        filename=f"{split_name.lower()}_{feature_name.lower()}_prob_before_scatter.png",
    )
    _plot_scatter(
        feature_values,
        prob_after,
        xlabel=feature_name,
        ylabel="p(true class) AFTER",
        title=f"{split_name} | {feature_name} vs p(true class) AFTER",
        filename=f"{split_name.lower()}_{feature_name.lower()}_prob_after_scatter.png",
    )

    if var_before_df is not None and var_after_df is not None:
        merged = var_before_df.merge(var_after_df, on="Node", suffixes=("_before", "_after"))
        var_before_map = dict(zip(merged["Node"].values, merged["VarProb_before"].values))
        var_after_map = dict(zip(merged["Node"].values, merged["VarProb_after"].values))
        fig, ax = plt.subplots(figsize=(6, 4))
        plotted = False
        for subset in subset_order:
            idx = subset_idx.get(subset)
            if idx is None or idx.size == 0:
                continue
            vb = np.array([var_before_map.get(int(node), np.nan) for node in idx])
            va = np.array([var_after_map.get(int(node), np.nan) for node in idx])
            mask = ~np.isnan(vb) & ~np.isnan(va)
            if mask.sum() == 0:
                continue
            ax.scatter(
                vb[mask],
                va[mask],
                color=subset_colors[subset],
                alpha=0.6,
                s=20,
                label=subset_labels[subset],
            )
            plotted = True
        if plotted:
            ax.set_xlabel("Variance Before")
            ax.set_ylabel("Variance After")
            min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
            max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", linewidth=1)
            ax.set_title(f"{split_name} | Variance shift")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{split_name.lower()}_variance_scatter.png"), dpi=150)
        plt.close(fig)

    if logits_before.shape[1] >= 2:
        pca = PCA(n_components=2)
        coords_before = pca.fit_transform(logits_before)
        coords_after = pca.transform(logits_after)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for subset in subset_order:
            idx = subset_idx.get(subset)
            if idx is None or idx.size == 0:
                continue
            axes[0].scatter(
                coords_before[idx, 0],
                coords_before[idx, 1],
                color=subset_colors[subset],
                alpha=0.6,
                s=15,
                label=subset_labels[subset],
            )
            axes[1].scatter(
                coords_after[idx, 0],
                coords_after[idx, 1],
                color=subset_colors[subset],
                alpha=0.6,
                s=15,
                label=subset_labels[subset],
            )
        if representative_examples is not None:
            rep_idx = representative_examples.detach().cpu().numpy()
            if rep_idx.size > 0:
                axes[0].scatter(
                    coords_before[rep_idx, 0],
                    coords_before[rep_idx, 1],
                    color="black",
                    alpha=0.9,
                    s=25,
                    marker="x",
                    label="Rep Nodes",
                )
                axes[1].scatter(
                    coords_after[rep_idx, 0],
                    coords_after[rep_idx, 1],
                    color="black",
                    alpha=0.9,
                    s=25,
                    marker="x",
                    label="Rep Nodes",
                )
        axes[0].set_title(f"{split_name} | Embeddings Before")
        axes[1].set_title(f"{split_name} | Embeddings After")
        for ax in axes:
            ax.set_xlabel("PCA-1")
            ax.set_ylabel("PCA-2")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=4)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(os.path.join(output_dir, f"{split_name.lower()}_embedding_scatter.png"), dpi=150)
        plt.close(fig)