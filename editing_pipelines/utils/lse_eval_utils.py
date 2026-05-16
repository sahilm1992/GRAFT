from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from edit_gnn.utils import prediction
from editing_pipelines.utils.results import (
    compute_auc_per_feature_bucket,
    mean_median_prob_by_feature,
    perturb_feature_and_measure_probs
)
from editing_pipelines.utils.visualization import (
    plot_auc_by_feature_with_counts,
    plot_mean_prob_by_feature,
    plot_rep_and_aug_distributions,
    plot_feature_sensitivity,
    plot_sensitivity_reduction,
    plot_transition_subset_distributions,
)

logger = logging.getLogger("main")


@torch.no_grad()
def compute_aug_confidences(model, data_obj, X_aug_all, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
    if not X_aug_all:
        return None
    if device is None:
        device = next(model.parameters()).device
    # Work on a device-aligned copy to avoid mixed CPU/CUDA tensors
    # (e.g. x on CUDA while edge_index/adj_t stays on CPU).
    data_obj = data_obj.to(device)
    original_x = data_obj.x
    confidences = []
    try:
        is_regression = bool(getattr(data_obj, "task_type", "") == "regression" or data_obj.y.dtype.is_floating_point)
        for X_aug in X_aug_all:
            data_obj.x = X_aug.to(device)
            logits = prediction(model, data_obj)
            if is_regression:
                pred = logits.squeeze(-1) if logits.dim() == 2 and logits.size(-1) == 1 else logits
                conf = torch.exp(-torch.abs(pred - data_obj.y.to(pred.dtype)))
            else:
                probs = torch.softmax(logits, dim=-1)
                conf, _ = probs.max(dim=-1)
            confidences.append(conf)
    finally:
        data_obj.x = original_x
    return torch.stack(confidences, dim=0)


@torch.no_grad()
def compute_orig_confidences(model, data_obj, device: Optional[torch.device] = None) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device
    data_obj = data_obj.to(device)
    logits = prediction(model, data_obj)
    is_regression = bool(getattr(data_obj, "task_type", "") == "regression" or data_obj.y.dtype.is_floating_point)
    if is_regression:
        pred = logits.squeeze(-1) if logits.dim() == 2 and logits.size(-1) == 1 else logits
        conf = torch.exp(-torch.abs(pred - data_obj.y.to(pred.dtype)))
    else:
        probs = torch.softmax(logits, dim=-1)
        conf, _ = probs.max(dim=-1)
    return conf


@torch.no_grad()
def compute_counterfactual_fraction(model, data, feat_idx: int, mask: torch.Tensor, base_preds: torch.Tensor) -> float:
    idx = mask.nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        return float("nan")
    original_x = data.x
    x_cf = original_x.clone()
    unique_vals = torch.unique(x_cf[:, feat_idx])
    if unique_vals.numel() < 2:
        return float("nan")
    mapping = {
        float(unique_vals[i].item()): float(unique_vals[(i + 1) % unique_vals.numel()].item())
        for i in range(unique_vals.numel())
    }
    idx_cpu = idx.cpu()
    for node in idx_cpu:
        orig_val = float(x_cf[node, feat_idx].item())
        x_cf[node, feat_idx] = mapping.get(orig_val, unique_vals[0])
    data.x = x_cf
    preds_cf = prediction(model, data).argmax(dim=-1)
    data.x = original_x
    diff = (preds_cf[idx] != base_preds[idx]).float().mean().item()
    return diff


@torch.no_grad()
def compute_instability_fraction(model, data, base_preds: torch.Tensor, mask: torch.Tensor) -> float:
    idx = mask.nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        return float("nan")
    original_x = data.x
    noise_scale = 0.01 * (original_x.std(dim=0, unbiased=False) + 1e-6)
    noise = torch.randn_like(original_x) * noise_scale
    x_noisy = original_x + noise
    data.x = x_noisy
    preds_noisy = prediction(model, data).argmax(dim=-1)
    data.x = original_x
    diff = (preds_noisy[idx] != base_preds[idx]).float().mean().item()
    return diff


@torch.no_grad()
def compute_feature_instability_fraction(
    model,
    data,
    base_preds: torch.Tensor,
    mask: torch.Tensor,
    feat_idx: int,
) -> float:
    idx = mask.nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        return float("nan")
    original_x = data.x
    x_noisy = original_x.clone()
    sensitive_vals = x_noisy[:, feat_idx]
    col_std = sensitive_vals.std(unbiased=False)
    noise_scale = 0.01 * (col_std + 1e-6)
    noise = torch.randn_like(sensitive_vals) * noise_scale
    x_noisy[:, feat_idx] = sensitive_vals + noise
    data.x = x_noisy
    preds_noisy = prediction(model, data).argmax(dim=-1)
    data.x = original_x
    diff = (preds_noisy[idx] != base_preds[idx]).float().mean().item()
    return diff


@torch.no_grad()
def compute_fairness_metrics(
    model,
    preds: torch.Tensor,
    data,
    feat_idx: int,
    mask: torch.Tensor,
    num_bins: int = 4,
    cat_threshold: int = 10
):
    def safe_mean(vec: torch.Tensor) -> float:
        if vec.numel() == 0:
            return float("nan")
        vec = vec[~torch.isnan(vec)]
        return float(vec.float().mean().item()) if vec.numel() > 0 else float("nan")

    def safe_max(vec: torch.Tensor) -> float:
        if vec.numel() == 0:
            return float("nan")
        vec = vec[~torch.isnan(vec)]
        return float(vec.max().item()) if vec.numel() > 0 else float("nan")

    result = {
        "sp_max": float("nan"),
        "sp_mean": float("nan"),
        "eo_max": float("nan"),
        "eo_mean": float("nan"),
        "counterfactual": float("nan"),
        "instability": float("nan"),
        "feature_instability": float("nan"),
    }

    idx = mask.nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        return result

    preds_mask = preds[idx]
    y_mask = data.y[idx]
    s_vals = data.x[idx, feat_idx]

    # ---------- categorical vs continuous ----------
    unique_vals = torch.unique(s_vals)
    if unique_vals.numel() <= cat_threshold:
        group_ids = s_vals
        groups = torch.unique(group_ids)
    else:
        quantiles = torch.linspace(0, 1, num_bins + 1, device=s_vals.device)
        bin_edges = torch.quantile(s_vals, quantiles)
        bin_edges = torch.unique(bin_edges)
        group_ids = torch.bucketize(s_vals, bin_edges, right=True) - 1
        groups = torch.unique(group_ids)

    # ---------- group-wise rates ----------
    sp_rates = []
    eo_rates = []

    for g in groups:
        g_mask = group_ids == g
        sp_rates.append(safe_mean(preds_mask[g_mask]))

        g_y1_mask = g_mask & (y_mask == 1)
        eo_rates.append(safe_mean(preds_mask[g_y1_mask]))

    sp_rates = torch.tensor(sp_rates)
    eo_rates = torch.tensor(eo_rates)

    # ---------- pairwise gaps ----------
    def pairwise_gaps(vec):
        diffs = torch.abs(vec.unsqueeze(0) - vec.unsqueeze(1))
        triu_mask = torch.triu(torch.ones_like(diffs), diagonal=1).bool()
        return diffs[triu_mask]

    if len(sp_rates) >= 2:
        sp_gaps = pairwise_gaps(sp_rates)
        result["sp_max"] = safe_max(sp_gaps)
        result["sp_mean"] = safe_mean(sp_gaps)

    if len(eo_rates) >= 2:
        eo_gaps = pairwise_gaps(eo_rates)
        result["eo_max"] = safe_max(eo_gaps)
        result["eo_mean"] = safe_mean(eo_gaps)

    # ---------- your other metrics ----------
    result["counterfactual"] = compute_counterfactual_fraction(
        model, data, feat_idx, mask, preds
    )
    result["instability"] = compute_instability_fraction(
        model, data, preds, mask
    )
    result["feature_instability"] = compute_feature_instability_fraction(
        model, data, preds, mask, feat_idx
    )

    return result



@torch.no_grad()
def evaluate_edit_effects(
    editor,
    save_dir: Optional[str] = None,
    feature_name: Optional[str] = None,
    num_bins: int = 3,
    per_age: bool = False,
):
    device = editor.get_device()
    data = editor.whole_data.to(device)

    save_dir = save_dir or editor.save_dir
    os.makedirs(save_dir, exist_ok=True)
    if not hasattr(editor, "model_before"):
        raise RuntimeError("`model_before` not found. Ensure edit_model saved a pre-edit snapshot.")

    model_before = editor.model_before.to(device).eval()
    model_after = editor.model.to(device).eval()

    logits_before_full = prediction(model_before, data)
    logits_after_full = prediction(model_after, data)

    if feature_name is None:
        feature_name = getattr(
            editor,
            "feature_name",
            editor.config.get("pipeline_params", {}).get("sensitive_feature", "AGE"),
        )

    assert hasattr(data, "feature_names") and feature_name in data.feature_names, (
        f"Feature '{feature_name}' not found in data.feature_names"
    )

    def get_preds(model, data_obj, override_x: Optional[torch.Tensor] = None) -> torch.Tensor:
        if override_x is None and data_obj is data:
            if model is model_before:
                return preds_b
            if model is model_after:
                return preds_a
        original_x = data_obj.x
        if override_x is not None:
            data_obj.x = override_x
        try:
            logits = prediction(model, data_obj)
            return logits.argmax(dim=-1)
        finally:
            if override_x is not None:
                data_obj.x = original_x

    y_true = data.y
    preds_b = logits_before_full.argmax(dim=-1)
    preds_a = logits_after_full.argmax(dim=-1)
    correct_before = preds_b.eq(y_true)
    correct_after = preds_a.eq(y_true)
    split_subsets = {
        "VAL": editor._build_transition_subsets(data.val_mask, correct_before, correct_after),
        "TEST": editor._build_transition_subsets(data.test_mask, correct_before, correct_after),
    }
    feat_idx = data.feature_names.index(feature_name)

    fairness_before = compute_fairness_metrics(
        model=model_before,
        preds=preds_b,
        data=data,
        feat_idx=feat_idx,
        mask=data.test_mask,
    )
    fairness_after = compute_fairness_metrics(
        model=model_after,
        preds=preds_a,
        data=data,
        feat_idx=feat_idx,
        mask=data.test_mask,
    )
    logger.info(
        f"[Fairness BEFORE] "
        f"SP_max={fairness_before['sp_max']:.4f} "
        f"SP_mean={fairness_before['sp_mean']:.4f} "
        f"EO_max={fairness_before['eo_max']:.4f} "
        f"EO_mean={fairness_before['eo_mean']:.4f} "
        f"Counterfactual={fairness_before['counterfactual']:.4f} "
        f"Instability={fairness_before['instability']:.4f} "
        f"FeatureInstability={fairness_before['feature_instability']:.4f}"
    )

    logger.info(
        f"[Fairness AFTER]  "
        f"SP_max={fairness_after['sp_max']:.4f} "
        f"SP_mean={fairness_after['sp_mean']:.4f} "
        f"EO_max={fairness_after['eo_max']:.4f} "
        f"EO_mean={fairness_after['eo_mean']:.4f} "
        f"Counterfactual={fairness_after['counterfactual']:.4f} "
        f"Instability={fairness_after['instability']:.4f} "
        f"FeatureInstability={fairness_after['feature_instability']:.4f}"
    )


    def acc_on_mask(preds: torch.Tensor, mask: torch.Tensor, name: str) -> Tuple[float, torch.Tensor]:
        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            logger.warning(f"[{name}] mask empty.")
            return float("nan"), idx
        acc = (preds[idx] == y_true[idx]).float().mean().item()
        return acc, idx

    val_acc_b, val_idx = acc_on_mask(preds_b, data.val_mask, "VAL")
    val_acc_a, _ = acc_on_mask(preds_a, data.val_mask, "VAL")
    test_acc_b, test_idx = acc_on_mask(preds_b, data.test_mask, "TEST")
    test_acc_a, _ = acc_on_mask(preds_a, data.test_mask, "TEST")

    logger.info(f"[VAL]  accuracy before={val_acc_b:.4f}  after={val_acc_a:.4f}  Δ={val_acc_a - val_acc_b:+.4f}")
    logger.info(f"[TEST] accuracy before={test_acc_b:.4f}  after={test_acc_a:.4f}  Δ={test_acc_a - test_acc_b:+.4f}")

    pd.DataFrame(
        [
            {"Split": "VAL", "Acc_Before": val_acc_b, "Acc_After": val_acc_a, "Delta": val_acc_a - val_acc_b},
            {"Split": "TEST", "Acc_Before": test_acc_b, "Acc_After": test_acc_a, "Delta": test_acc_a - test_acc_b},
        ]
    ).to_csv(os.path.join(save_dir, "split_accuracy_summary.csv"), index=False)

    feature_values_np = data.x[:, feat_idx].detach().cpu().numpy()
    degrees_np = editor._get_node_degrees(data)
    logits_before_np = logits_before_full.detach().cpu().numpy()
    logits_after_np = logits_after_full.detach().cpu().numpy()
    prob_all_before = torch.softmax(logits_before_full, dim=-1)
    prob_all_after = torch.softmax(logits_after_full, dim=-1)
    node_indices = torch.arange(prob_all_before.size(0), device=prob_all_before.device)
    prob_selected_before = prob_all_before[node_indices, y_true].detach().cpu().numpy()
    prob_selected_after = prob_all_after[node_indices, y_true].detach().cpu().numpy()

    var_frames: Dict[str, Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]] = {}
    try:
        fixed_vals = getattr(editor, "fixed_sensitive_values", None)
        val_sens_before_plot, test_sens_before_plot = perturb_feature_and_measure_probs(
            model_before, data, feature_name=feature_name, sensitive_feature_values=fixed_vals, compute_flips=True
        )
        val_sens_after_plot, test_sens_after_plot = perturb_feature_and_measure_probs(
            model_after, data, feature_name=feature_name, sensitive_feature_values=fixed_vals, compute_flips=True
        )
        var_frames = {
            "VAL": (val_sens_before_plot, val_sens_after_plot),
            "TEST": (test_sens_before_plot, test_sens_after_plot),
        }
    except Exception as err:
        logger.warning(f"[TransitionViz] Skipped variance-based plots due to error: {err}")
        var_frames = {}

    aug_b = aug_a = None
    if hasattr(editor, "X_aug_all") and isinstance(editor.X_aug_all, list) and len(editor.X_aug_all) > 0:
        if not hasattr(editor, "representative_examples"):
            logger.warning("`representative_examples` missing; cannot compute augmented-set accuracy.")
        else:
            rep_idx = editor.representative_examples.to(device=device, dtype=torch.long)

            acc_b_list, acc_a_list = [], []
            for i, X_aug in enumerate(editor.X_aug_all):
                X_aug = X_aug.to(device)
                preds_b_aug = get_preds(model_before, data, override_x=X_aug)
                preds_a_aug = get_preds(model_after, data, override_x=X_aug)
                acc_b_i = (preds_b_aug[rep_idx] == y_true[rep_idx]).float().mean().item()
                acc_a_i = (preds_a_aug[rep_idx] == y_true[rep_idx]).float().mean().item()
                acc_b_list.append(acc_b_i)
                acc_a_list.append(acc_a_i)
                logger.info(f"[AUG rep {i}] before={acc_b_i:.4f} after={acc_a_i:.4f} Δ={acc_a_i-acc_b_i:+.4f}")

            aug_b = float(np.mean(acc_b_list))
            aug_a = float(np.mean(acc_a_list))
            logger.info(
                f"[AUG (representative mean over K={len(editor.X_aug_all)})] "
                f"before={aug_b:.4f} after={aug_a:.4f} Δ={aug_a-aug_b:+.4f}"
            )

            pd.DataFrame(
                {
                    "AugIndex": list(range(len(editor.X_aug_all))),
                    "Acc_Before": acc_b_list,
                    "Acc_After": acc_a_list,
                    "Delta": [a - b for a, b in zip(acc_a_list, acc_b_list)],
                }
            ).to_csv(os.path.join(save_dir, "augmented_representatives_accuracy.csv"), index=False)
    else:
        logger.info("No `X_aug_all` found — skipping augmented representative accuracy.")

    def transition_counts(preds_before: torch.Tensor, preds_after: torch.Tensor, idx: Optional[torch.Tensor] = None) -> pd.DataFrame:
        if idx is not None and idx.numel() > 0:
            b = preds_before[idx] == y_true[idx]
            a = preds_after[idx] == y_true[idx]
        else:
            b = preds_before == y_true
            a = preds_after == y_true
        labels = ["BeforeCorrect", "AfterCorrect", "Count"]
        arr = torch.stack([b, a], dim=1).cpu().numpy()
        bb_ff = np.logical_and(arr[:, 0] == False, arr[:, 1] == False).sum()
        bb_ft = np.logical_and(arr[:, 0] == False, arr[:, 1] == True).sum()
        bb_tf = np.logical_and(arr[:, 0] == True, arr[:, 1] == False).sum()
        bb_tt = np.logical_and(arr[:, 0] == True, arr[:, 1] == True).sum()
        df = pd.DataFrame(
            [
                [False, False, bb_ff],
                [False, True, bb_ft],
                [True, False, bb_tf],
                [True, True, bb_tt],
            ],
            columns=labels,
        )
        return df

    def save_transition(df: pd.DataFrame, tag: str):
        p = os.path.join(save_dir, f"transition_{tag}.csv")
        df.to_csv(p, index=False)
        mat = np.zeros((2, 2), dtype=int)
        for _, r in df.iterrows():
            i = 1 if r["BeforeCorrect"] else 0
            j = 1 if r["AfterCorrect"] else 0
            mat[i, j] = int(r["Count"])
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(mat)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Wrong", "Correct"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Wrong", "Correct"])
        ax.set_xlabel("After")
        ax.set_ylabel("Before")
        for (i, j), val in np.ndenumerate(mat):
            ax.text(j, i, str(val), ha="center", va="center")
        ax.set_title(f"Transition ({tag})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"transition_{tag}.png"), dpi=150)
        plt.close(fig)

    trans_all = transition_counts(preds_b, preds_a, idx=None)
    trans_val = transition_counts(preds_b, preds_a, idx=val_idx)
    trans_test = transition_counts(preds_b, preds_a, idx=test_idx)

    save_transition(trans_all, "overall")
    save_transition(trans_val, "val")
    save_transition(trans_test, "test")

    for split_name, idx_subset in zip(["VAL", "TEST"], [val_idx, test_idx]):
        subsets = split_subsets.get(split_name)
        if not subsets:
            continue
        plot_transition_subset_distributions(
            split_name=split_name,
            subset_nodes=subsets,
            feature_values=feature_values_np,
            degrees=degrees_np,
            logits_before=logits_before_np,
            logits_after=logits_after_np,
            prob_before=prob_selected_before,
            prob_after=prob_selected_after,
            var_before_df=var_frames.get(split_name, (None, None))[0],
            var_after_df=var_frames.get(split_name, (None, None))[1],
            feature_name=feature_name,
            save_dir=save_dir,
            representative_examples=getattr(editor, "representative_examples", None),
        )

    feat_col = data.feature_names.index(feature_name)
    feat_vals = data.x[:, feat_col].detach().cpu().float().numpy()

    def feature_bins_quantiles(values: np.ndarray, idx: np.ndarray, n_bins: int):
        if idx.size == 0:
            return np.array([0.0, 1.0]), np.zeros(0, dtype=int)
        split_vals = values[idx]
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(split_vals, qs)
        edges = np.unique(edges)
        if edges.size == 1:
            edges = np.array([edges[0], edges[0] + 1e-6])
        return edges, np.digitize(split_vals, edges[1:-1], right=False)

    def bucket_acc_report(split_name: str, idx_tensor: torch.Tensor, tag: str) -> pd.DataFrame:
        idx = idx_tensor.detach().cpu().numpy()
        edges, bin_ids = feature_bins_quantiles(feat_vals, idx, num_bins)

        labels = []
        for i in range(len(edges) - 1):
            if i < len(edges) - 2:
                labels.append(f"[{edges[i]:.1f},{edges[i+1]:.1f})")
            else:
                labels.append(f"[{edges[i]:.1f},{edges[i+1]:.1f}]")

        rows = []
        for b in range(len(labels)):
            sel = (bin_ids == b)
            if sel.sum() == 0:
                continue
            idx_b = idx[sel]
            acc_b = (preds_b[idx_b] == y_true[idx_b]).float().mean().item()
            acc_a = (preds_a[idx_b] == y_true[idx_b]).float().mean().item()
            rows.append([labels[b], acc_b, acc_a, int(len(idx_b))])

        bin_col = f"{feature_name}Bin"
        df = pd.DataFrame(rows, columns=[bin_col, "Acc_Before", "Acc_After", "Count"])
        df.to_csv(os.path.join(save_dir, f"{feature_name.lower()}_buckets_{tag}.csv"), index=False)

        if len(df) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            x = np.arange(len(df))
            ax.bar(x - 0.18, df["Acc_Before"].values, width=0.36, label="Before")
            ax.bar(x + 0.18, df["Acc_After"].values, width=0.36, label="After")
            ax.set_xticks(x)
            ax.set_xticklabels(df[bin_col].tolist(), rotation=10, ha="right")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"{split_name} {feature_name}-wise Accuracy")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f"{feature_name.lower()}_buckets_{tag}.png"), dpi=150)
            plt.close(fig)
        return df

    val_feat_df = bucket_acc_report("Validation", val_idx, "val")
    test_feat_df = bucket_acc_report("Test", test_idx, "test")

    if per_age:
        def per_feature_dump(idx_tensor: torch.Tensor, tag: str):
            idx = idx_tensor.detach().cpu().numpy()
            df = pd.DataFrame(
                {
                    "Node": idx,
                    feature_name: feat_vals[idx],
                    "Correct_Before": (preds_b[idx] == y_true[idx]).cpu().numpy(),
                    "Correct_After": (preds_a[idx] == y_true[idx]).cpu().numpy(),
                }
            )
            df.to_csv(os.path.join(save_dir, f"per_{feature_name.lower()}_{tag}.csv"), index=False)
        per_feature_dump(val_idx, "val")
        per_feature_dump(test_idx, "test")

    logger.info("\n[=== EDIT EVALUATION SUMMARY ===]")
    logger.info(f"VAL   acc  before={val_acc_b:.4f}  after={val_acc_a:.4f}  Δ={val_acc_a - val_acc_b:+.4f}")
    logger.info(f"TEST  acc  before={test_acc_b:.4f} after={test_acc_a:.4f} Δ={test_acc_a - test_acc_b:+.4f}")
    if aug_b is not None:
        logger.info(
            f"AUG (rep mean over K={len(editor.X_aug_all)})  "
            f"before={aug_b:.4f} after={aug_a:.4f} Δ={aug_a - aug_b:+.4f}"
        )
    logger.info(f"Saved detailed CSVs/plots to: {save_dir}")

    auc_before_eq = compute_auc_per_feature_bucket(model_before, data, feature_name=feature_name, mode="equal_width")
    auc_after_eq = compute_auc_per_feature_bucket(model_after, data, feature_name=feature_name, mode="equal_width")

    auc_before_eq.to_csv(os.path.join(save_dir, f"auc_{feature_name.lower()}_equal_before.csv"), index=False)
    auc_after_eq.to_csv(os.path.join(save_dir, f"auc_{feature_name.lower()}_equal_after.csv"), index=False)

    logger.info(f"[AUC by {feature_name} | equal] before:\n{auc_before_eq}")
    logger.info(f"[AUC by {feature_name} | equal] after:\n{auc_after_eq}")

    auc_before_q = compute_auc_per_feature_bucket(model_before, data, feature_name=feature_name, mode="quantile")
    auc_after_q = compute_auc_per_feature_bucket(model_after, data, feature_name=feature_name, mode="quantile")

    auc_before_q.to_csv(os.path.join(save_dir, f"auc_{feature_name.lower()}_quantile_before.csv"), index=False)
    auc_after_q.to_csv(os.path.join(save_dir, f"auc_{feature_name.lower()}_quantile_after.csv"), index=False)

    logger.info(f"[AUC by {feature_name} | quantile] before:\n{auc_before_q}")
    logger.info(f"[AUC by {feature_name} | quantile] after:\n{auc_after_q}")

    mm_subset = editor.config.get("pipeline_params", {}).get("leastsquares", {}).get("mean_median_subset", "val_test")
    mm_prob_mode = editor.config.get("pipeline_params", {}).get("leastsquares", {}).get("mean_median_prob_mode", "positive")
    prob_label = "p(true class)" if mm_prob_mode == "true_class" else "p(y=1)"

    mm_before_eq = mean_median_prob_by_feature(
        model_before,
        data,
        feature_name=feature_name,
        mode="equal_width",
        subset=mm_subset,
        prob_mode=mm_prob_mode,
    )
    mm_after_eq = mean_median_prob_by_feature(
        model_after,
        data,
        feature_name=feature_name,
        mode="equal_width",
        subset=mm_subset,
        prob_mode=mm_prob_mode,
    )

    mm_before_eq.to_csv(os.path.join(save_dir, f"mean_median_prob_{feature_name.lower()}_equal_before.csv"), index=False)
    mm_after_eq.to_csv(os.path.join(save_dir, f"mean_median_prob_{feature_name.lower()}_equal_after.csv"), index=False)

    mm_before_q = mean_median_prob_by_feature(
        model_before,
        data,
        feature_name=feature_name,
        mode="quantile",
        subset=mm_subset,
        prob_mode=mm_prob_mode,
    )
    mm_after_q = mean_median_prob_by_feature(
        model_after,
        data,
        feature_name=feature_name,
        mode="quantile",
        subset=mm_subset,
        prob_mode=mm_prob_mode,
    )

    mm_before_q.to_csv(os.path.join(save_dir, f"mean_median_prob_{feature_name.lower()}_quantile_before.csv"), index=False)
    mm_after_q.to_csv(os.path.join(save_dir, f"mean_median_prob_{feature_name.lower()}_quantile_after.csv"), index=False)

    fixed_vals = getattr(editor, "fixed_sensitive_values", None)

    val_before, test_before = perturb_feature_and_measure_probs(
        model_before,
        data,
        feature_name=feature_name,
        sensitive_feature_values=fixed_vals,
    )
    val_after, test_after = perturb_feature_and_measure_probs(
        model_after,
        data,
        feature_name=feature_name,
        sensitive_feature_values=fixed_vals,
    )

    val_before.to_csv(os.path.join(save_dir, f"{feature_name.lower()}_sensitivity_val_before.csv"), index=False)
    test_before.to_csv(os.path.join(save_dir, f"{feature_name.lower()}_sensitivity_test_before.csv"), index=False)
    val_after.to_csv(os.path.join(save_dir, f"{feature_name.lower()}_sensitivity_val_after.csv"), index=False)
    test_after.to_csv(os.path.join(save_dir, f"{feature_name.lower()}_sensitivity_test_after.csv"), index=False)

    def compute_and_plot_var_drop(df_before: pd.DataFrame, df_after: pd.DataFrame, split: str) -> pd.DataFrame:
        merged = df_before.merge(df_after, on="Node", suffixes=("_before", "_after"))
        merged["VarDropAbs"] = merged["VarProb_before"] - merged["VarProb_after"]

        eps_thresh = 1e-10
        denom = merged["VarProb_before"].copy()
        small_mask = denom.abs() < eps_thresh
        denom[small_mask] = np.nan
        merged["VarDropPct"] = 100.0 * merged["VarDropAbs"] / denom
        merged["VarDropPct_clipped"] = merged["VarDropPct"].clip(-500, 500)

        merged.to_csv(os.path.join(save_dir, f"{feature_name.lower()}_var_drop_per_node_{split}.csv"), index=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        vals = merged["VarDropPct_clipped"].dropna()
        ax.hist(vals, bins=50)
        ax.axvline(0.0, linestyle="--")
        ax.set_xlabel(
            "Variance drop (%)  (positive = variance decreased)\n"
            "(clipped to [-500, 500])"
        )
        ax.set_ylabel("Number of nodes")
        ax.set_title(f"Per-node variance drop under {feature_name} perturbations ({split.upper()})")
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{feature_name.lower()}_var_drop_hist_{split}.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(merged["VarProb_before"], merged["VarProb_after"], alpha=0.3, s=10)
        min_val = min(merged["VarProb_before"].min(), merged["VarProb_after"].min())
        max_val = max(merged["VarProb_before"].max(), merged["VarProb_after"].max())
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
        ax.set_xlabel("VarProb before")
        ax.set_ylabel("VarProb after")
        ax.set_title(f"Per-node VarProb (before vs after) [{split.upper()}]")
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{feature_name.lower()}_VarProb_before_vs_after_{split}.png"), dpi=150)
        plt.close(fig)

        return merged

    compute_and_plot_var_drop(val_before, val_after, split="val")
    compute_and_plot_var_drop(test_before, test_after, split="test")

    plot_auc_by_feature_with_counts(
        auc_before_eq,
        auc_after_eq,
        feature_name=feature_name,
        save_dir=save_dir,
        suffix="_equal",
    )
    plot_auc_by_feature_with_counts(
        auc_before_q,
        auc_after_q,
        feature_name=feature_name,
        save_dir=save_dir,
        suffix="_quantile",
    )

    plot_mean_prob_by_feature(
        mm_before_eq,
        mm_after_eq,
        feature_name=feature_name,
        stat="mean",
        save_dir=save_dir,
        suffix="_equal",
        prob_label=prob_label,
    )
    plot_mean_prob_by_feature(
        mm_before_q,
        mm_after_q,
        feature_name=feature_name,
        stat="mean",
        save_dir=save_dir,
        suffix="_quantile",
        prob_label=prob_label,
    )
    plot_mean_prob_by_feature(
        mm_before_eq,
        mm_after_eq,
        feature_name=feature_name,
        stat="median",
        save_dir=save_dir,
        suffix="_equal",
        prob_label=prob_label,
    )
    plot_mean_prob_by_feature(
        mm_before_q,
        mm_after_q,
        feature_name=feature_name,
        stat="median",
        save_dir=save_dir,
        suffix="_quantile",
        prob_label=prob_label,
    )

    plot_rep_and_aug_distributions(
        model_before,
        data,
        getattr(editor, "representative_examples", []),
        getattr(editor, "X_aug_all", []),
        feature_name,
        save_dir,
    )

    plot_feature_sensitivity(
        val_before,
        val_after,
        test_before,
        test_after,
        feature_name=feature_name,
        save_dir=save_dir,
    )
    plot_sensitivity_reduction(
        val_before,
        val_after,
        test_before,
        test_after,
        feature_name=feature_name,
        save_dir=save_dir,
    )

    return {"before": fairness_before, "after": fairness_after}
