def process_raw_exp_results(raw_results):
    from pipelines.seed_gnn.utils import process_raw_exp_results as seed_process_raw
    return seed_process_raw(raw_results)


def process_edit_results(bef_edit_results, raw_results):
    from editing_pipelines.utils.train_eval import test  # noqa: F401
    from edit_gnn.utils import process_edit_results as seed_process_edit_results
    return seed_process_edit_results(bef_edit_results, raw_results)


# -----------------------------
# Misclassification TXT saving
# -----------------------------

import os
from pathlib import Path
import logging
import torch
import numpy as np
from typing import Dict, Optional

from edit_gnn.utils import prediction, grab_input  # noqa: E402
from edit_gnn.utils import compute_micro_f1  # noqa: E402
from torch_geometric.loader import NeighborLoader
import torch_geometric.typing as pyg_typing

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, average_precision_score
import pandas as pd

from editing_pipelines.utils.gat_neighbor_eval import (
    gat_backbone_num_layers,
    should_use_gat_neighbor_loader,
)

logger = logging.getLogger(__name__)


def _misclassified_indices(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor):
    y_pred = logits.argmax(dim=-1)
    idx = mask.nonzero(as_tuple=True)[0]
    wrong_mask = y_pred[idx].ne(y[idx])
    wrong_idx = idx[wrong_mask]
    return wrong_idx, y[wrong_idx], y_pred[wrong_idx]


def _write_txt(path: Path, idx: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        f.write('node_id\ttrue\tpred\n')
        for i, t, p in zip(idx.tolist(), y_true.tolist(), y_pred.tolist()):
            f.write(f"{i}\t{int(t)}\t{int(p)}\n")


def save_misclassifications_txt(config: dict,
                                model_before,
                                model_after,
                                whole_data,
                                method_name: str,
                                model_name: str,
                                file_suffix: str = ''):
    """
    Save validation and test misclassifications before and after editing as TXT files.
    Each file contains lines: node_id\ttrue\tpred
    """
    out_root = Path(config['management']['output_folder_dir']) / 'results_txt' / f"{config['eval_params']['dataset']}_{model_name}"

    with torch.no_grad():
        logits_before = prediction(model_before, whole_data)
        logits_after = prediction(model_after, whole_data)

    val_idx_b, val_true_b, val_pred_b = _misclassified_indices(logits_before, whole_data.y, whole_data.val_mask)
    val_idx_a, val_true_a, val_pred_a = _misclassified_indices(logits_after, whole_data.y, whole_data.val_mask)
    tst_idx_b, tst_true_b, tst_pred_b = _misclassified_indices(logits_before, whole_data.y, whole_data.test_mask)
    tst_idx_a, tst_true_a, tst_pred_a = _misclassified_indices(logits_after, whole_data.y, whole_data.test_mask)

    prefix = f"{method_name}{file_suffix}" if method_name else "results"
    _write_txt(out_root / f"val_miscls_before_{prefix}.txt", val_idx_b, val_true_b, val_pred_b)
    _write_txt(out_root / f"val_miscls_after_{prefix}.txt", val_idx_a, val_true_a, val_pred_a)
    _write_txt(out_root / f"test_miscls_before_{prefix}.txt", tst_idx_b, tst_true_b, tst_pred_b)
    _write_txt(out_root / f"test_miscls_after_{prefix}.txt", tst_idx_a, tst_true_a, tst_pred_a)


def _compute_metrics(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    idx = mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return {
            "count": 0,
            "acc": float('nan'),
            "f1_micro": float('nan'),
            "auc": float('nan'),
            "auc_pr": float('nan'),
        }
    logits_s = logits[idx]
    y_s = y[idx]
    y_pred = logits_s.argmax(dim=-1)
    count = int((y_pred != y_s).sum().item())
    acc = accuracy_score(y_s.cpu().numpy(), y_pred.cpu().numpy())
    # F1 micro for multiclass
    try:
        f1_micro = f1_score(y_s.cpu().numpy(), y_pred.cpu().numpy(), average='micro')
    except Exception:
        # fallback to repo's compute_micro_f1
        f1_micro = compute_micro_f1(logits, y, mask)
    # AUC: binary -> use prob of class 1; multiclass -> ovr macro
    auc = float('nan')
    auc_pr = float('nan')
    try:
        if logits_s.shape[1] == 2:
            prob = torch.softmax(logits_s, dim=1)[:, 1].detach().cpu().numpy()
            auc = roc_auc_score(y_s.cpu().numpy(), prob)
            auc_pr = average_precision_score(y_s.cpu().numpy(), prob)
        else:
            prob = torch.softmax(logits_s, dim=1).detach().cpu().numpy()
            auc = roc_auc_score(y_s.cpu().numpy(), prob, multi_class='ovr', average='macro')
            labels = np.unique(y_s.cpu().numpy())
            ap_scores = []
            for cls in labels:
                cls_mask = (y_s.cpu().numpy() == cls)
                if cls_mask.sum() == 0:
                    continue
                ap_scores.append(average_precision_score(cls_mask.astype(int), prob[:, cls]))
            if ap_scores:
                auc_pr = float(np.mean(ap_scores))
    except Exception:
        pass
    return {"count": count, "acc": acc, "f1_micro": f1_micro, "auc": auc, "auc_pr": auc_pr}


def save_misclassification_summary_txt(config: dict,
                                        model_before,
                                        model_after,
                                        whole_data,
                                        method_name: str,
                                        model_name: str,
                                        file_suffix: str = '',
                                        edit_indices: Optional[torch.Tensor] = None):
    out_root = Path(config['management']['output_folder_dir']) / 'results_txt' / f"{config['eval_params']['dataset']}_{model_name}"
    out_root.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        logits_before = prediction(model_before, whole_data)
        logits_after = prediction(model_after, whole_data)
    val_b = _compute_metrics(logits_before, whole_data.y, whole_data.val_mask)
    val_a = _compute_metrics(logits_after, whole_data.y, whole_data.val_mask)
    tst_b = _compute_metrics(logits_before, whole_data.y, whole_data.test_mask)
    tst_a = _compute_metrics(logits_after, whole_data.y, whole_data.test_mask)
    # Targets subset metrics (optional)
    tgt_b = tgt_a = None
    if edit_indices is not None:
        if edit_indices.dim() > 1:
            edit_indices = edit_indices.squeeze(dim=1)
        mask_tgt = torch.zeros_like(whole_data.y, dtype=torch.bool, device=whole_data.y.device)
        mask_tgt[edit_indices] = True
        tgt_b = _compute_metrics(logits_before, whole_data.y, mask_tgt)
        tgt_a = _compute_metrics(logits_after, whole_data.y, mask_tgt)
    prefix = f"{method_name}{file_suffix}" if method_name else "results"
    path = out_root / f"summary_metrics_{prefix}.txt"
    def fmt(metric_before, metric_after):
        def _fmt(val):
            return f"{val:.4f}" if not np.isnan(val) else "nan"
        return f"{_fmt(metric_before)}->{_fmt(metric_after)}"

    with path.open('w') as f:
        f.write("Summary metrics (before -> after)\n")
        f.write("Section\tCount\tAcc\tF1_micro\tAUC_ROC\tAUC_PR\n")
        f.write(
            f"val\t{val_b['count']}->{val_a['count']}\t"
            f"{fmt(val_b['acc'], val_a['acc'])}\t"
            f"{fmt(val_b['f1_micro'], val_a['f1_micro'])}\t"
            f"{fmt(val_b['auc'], val_a['auc'])}\t"
            f"{fmt(val_b['auc_pr'], val_a['auc_pr'])}\n"
        )
        f.write(
            f"test\t{tst_b['count']}->{tst_a['count']}\t"
            f"{fmt(tst_b['acc'], tst_a['acc'])}\t"
            f"{fmt(tst_b['f1_micro'], tst_a['f1_micro'])}\t"
            f"{fmt(tst_b['auc'], tst_a['auc'])}\t"
            f"{fmt(tst_b['auc_pr'], tst_a['auc_pr'])}\n"
        )
        if tgt_b is not None and tgt_a is not None:
            f.write(
                f"targets\t{tgt_b['count']}->{tgt_a['count']}\t"
                f"{fmt(tgt_b['acc'], tgt_a['acc'])}\t"
                f"{fmt(tgt_b['f1_micro'], tgt_a['f1_micro'])}\t"
                f"{fmt(tgt_b['auc'], tgt_a['auc'])}\t"
                f"{fmt(tgt_b['auc_pr'], tgt_a['auc_pr'])}\n"
            )



# ============================
# Evaluation helpers (moved)
# ============================
def compute_auc_per_feature_bucket(
    model,
    data,
    feature_name: str,
    num_bins: int = 10,
    mode: str = "equal_width"
) -> pd.DataFrame:
    """
    Generalized version of compute_auc_per_age_bucket.
    Works for any numeric sensitive feature (continuous or discrete).
    """

    with torch.no_grad():
        logits = prediction(model, data)
    is_regression = bool(getattr(data, "task_type", "") == "regression" or data.y.dtype.is_floating_point)
    if is_regression or (logits.dim() == 2 and logits.size(-1) < 2):
        return pd.DataFrame(columns=[f"{feature_name}Bin", "AUC", "Count"])
    probs_full = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
    y_true_full = data.y.detach().cpu().numpy()

    feat_idx = data.feature_names.index(feature_name)
    feat_vals_full = data.x[:, feat_idx].detach().cpu().float().numpy()

    mask = (data.val_mask | data.test_mask).detach().cpu().numpy()
    probs = probs_full[mask]
    y_true = y_true_full[mask]
    feat_vals = feat_vals_full[mask]

    if feat_vals.size == 0:
        return pd.DataFrame(columns=[f"{feature_name}Bin", "AUC", "Count"])

    unique_vals = np.unique(feat_vals)
    is_continuous = len(unique_vals) > 30

    if is_continuous:
        if mode == "quantile":
            edges = np.quantile(feat_vals, np.linspace(0, 1, num_bins + 1))
        else:
            edges = np.linspace(feat_vals.min(), feat_vals.max(), num_bins + 1)
        bins = np.digitize(feat_vals, edges[1:-1], right=False)
        labels = [f"{edges[b]:.1f}-{edges[b+1]:.1f}" for b in range(num_bins)]
    else:
        vals_sorted = sorted(unique_vals)
        idmap = {v: i for i, v in enumerate(vals_sorted)}
        bins = np.array([idmap[v] for v in feat_vals])
        labels = [str(v) for v in vals_sorted]

    rows = []
    for b in range(len(labels)):
        mask_b = bins == b
        count = int(mask_b.sum())
        if count < 5:
            rows.append([labels[b], float("nan"), count])
            continue
        yb = y_true[mask_b]
        pb = probs[mask_b]
        try:
            auc = roc_auc_score(yb, pb)
        except:
            auc = float("nan")
        rows.append([labels[b], auc, count])

    return pd.DataFrame(rows, columns=[f"{feature_name}Bin", "AUC", "Count"])


def mean_median_prob_by_feature(
    model,
    data,
    feature_name: str,
    num_bins: int = 6,
    mode: str = "quantile",
    subset: str = "val_test",
    prob_mode: str = "positive"
):
    with torch.no_grad():
        logits = prediction(model, data)
    is_regression = bool(getattr(data, "task_type", "") == "regression" or data.y.dtype.is_floating_point)
    if is_regression:
        preds = logits.squeeze(-1) if logits.dim() == 2 and logits.size(-1) == 1 else logits
        probs_full = preds.detach().cpu().numpy()
    else:
        probs_matrix = torch.softmax(logits, dim=-1)
        if prob_mode == "true_class":
            row = torch.arange(probs_matrix.size(0), device=probs_matrix.device)
            probs_full = probs_matrix[row, data.y].detach().cpu().numpy()
        else:
            probs_full = probs_matrix[:, 1].detach().cpu().numpy()

    feat_idx = data.feature_names.index(feature_name)
    feat_vals_full = data.x[:, feat_idx].detach().cpu().numpy()

    if subset == "val_test":
        mask = (data.val_mask | data.test_mask).detach().cpu().numpy()
        probs = probs_full[mask]
        feats = feat_vals_full[mask]
    else:
        probs = probs_full
        feats = feat_vals_full

    if feats.size == 0:
        return pd.DataFrame(columns=[f"{feature_name}Bin", "MeanProb", "MedianProb", "Count"])

    unique_vals = np.unique(feats)
    is_continuous = len(unique_vals) > 30

    if is_continuous:
        if mode == "equal_width":
            edges = np.linspace(feats.min(), feats.max(), num_bins + 1)
        else:
            edges = np.quantile(feats, np.linspace(0, 1, num_bins + 1))
        bins = np.digitize(feats, edges[1:-1], right=False)
        labels = [f"{edges[b]:.1f}-{edges[b+1]:.1f}" for b in range(num_bins)]
    else:
        vals_sorted = sorted(unique_vals)
        idmap = {v: i for i, v in enumerate(vals_sorted)}
        bins = np.array([idmap[v] for v in feats])
        labels = [str(v) for v in vals_sorted]

    rows = []
    for b in range(len(labels)):
        mask_b = bins == b
        if not mask_b.any(): continue
        vals = probs[mask_b]
        rows.append([labels[b], float(vals.mean()), float(np.median(vals)), int(mask_b.sum())])

    return pd.DataFrame(rows, columns=[f"{feature_name}Bin", "MeanProb", "MedianProb", "Count"])

def perturb_feature_and_measure_probs(
    model,
    data,
    feature_name: str,
    sensitive_feature_values=None,
    K: int = 6,
    relative=True,
    eps=1e-8,
    prob_mode="true_class",
    compute_flips=False,
):
    """
    Generalized sensitivity measurement for any numeric sensitive feature.
    If sensitive_feature_values is provided, perturb using those discrete values.
    Else sample random values between min and max of the feature within VAL nodes.

    Args:
        prob_mode: "true_class" or "positive" (class 1)
        compute_flips: if True, also returns the fraction of times the prediction changed
    """

    # Ensure model and graph tensors are on the same device for prediction calls.
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = data.x.device
    data = data.to(model_device)
    feat_idx = data.feature_names.index(feature_name)

    # Save original feature tensor
    original_x = data.x.clone()
    is_regression = bool(getattr(data, "task_type", "") == "regression" or data.y.dtype.is_floating_point)

    prev_with_pyg_lib = getattr(pyg_typing, "WITH_PYG_LIB", False)
    try:
        use_neighbor_eval = should_use_gat_neighbor_loader(model, data)
        data_cpu = None
        target_mask_cpu = None
        if use_neighbor_eval:
            if prev_with_pyg_lib and getattr(pyg_typing, "WITH_TORCH_SPARSE", False):
                pyg_typing.WITH_PYG_LIB = False
            data_cpu = data.to("cpu")
            if data_cpu.val_mask is not None and data_cpu.test_mask is not None:
                target_mask_cpu = (data_cpu.val_mask | data_cpu.test_mask).cpu()
            else:
                target_mask_cpu = torch.ones(int(data_cpu.num_nodes), dtype=torch.bool)
            logger.warning("Using NeighborLoader fallback for perturbation sensitivity evaluation.")

        def _predict_signal(current_x: torch.Tensor, need_preds: bool):
            if not use_neighbor_eval:
                data.x = current_x.to(model_device)
                logits = prediction(model, data)
                if is_regression:
                    sig = (logits.squeeze(-1) if logits.dim() == 2 and logits.size(-1) == 1 else logits).detach().cpu()
                    pred = None
                else:
                    probs = torch.softmax(logits, dim=-1).detach().cpu()
                    if prob_mode == "true_class":
                        y_cpu = data.y.detach().cpu().long()
                        sig = probs[torch.arange(probs.size(0)), y_cpu]
                    else:
                        sig = probs[:, 1]
                    pred = logits.argmax(dim=-1).detach().cpu() if need_preds else None
                return sig, pred

            assert data_cpu is not None and target_mask_cpu is not None
            data_cpu.x = current_x.detach().cpu()
            nl = gat_backbone_num_layers(model)
            hop_layers = nl if nl > 0 else int(getattr(model, "num_layers", 2))
            loader = NeighborLoader(
                data_cpu,
                input_nodes=target_mask_cpu,
                num_neighbors=[10] * hop_layers,
                batch_size=1024,
                shuffle=False,
            )
            num_nodes_local = int(data_cpu.num_nodes)
            signal_all = torch.full((num_nodes_local,), float("nan"), dtype=torch.float32)
            pred_all = torch.full((num_nodes_local,), -1, dtype=torch.long) if need_preds else None

            for batch in loader:
                batch = batch.to(model_device)
                logits = model(**grab_input(batch))
                seed_bs = int(batch.batch_size)
                seed_nodes = batch.n_id[:seed_bs].detach().cpu().long()
                logits_seed = logits[:seed_bs]
                if is_regression:
                    sig_seed = (logits_seed.squeeze(-1) if logits_seed.dim() == 2 and logits_seed.size(-1) == 1 else logits_seed).detach().cpu().to(torch.float32)
                else:
                    probs_seed = torch.softmax(logits_seed, dim=-1).detach().cpu()
                    if prob_mode == "true_class":
                        y_seed = data_cpu.y[seed_nodes].long()
                        sig_seed = probs_seed[torch.arange(seed_bs), y_seed].to(torch.float32)
                    else:
                        sig_seed = probs_seed[:, 1].to(torch.float32)
                    if pred_all is not None:
                        pred_all[seed_nodes] = logits_seed.argmax(dim=-1).detach().cpu().long()
                signal_all[seed_nodes] = sig_seed

            return signal_all, pred_all

        # Baseline predictions (always collected for binary variance + flip stats)
        with torch.no_grad():
            base_signal, preds_orig = _predict_signal(original_x, need_preds=(compute_flips and not is_regression))

        # Extract feature values
        feat_vals = original_x[:, feat_idx].detach().cpu().numpy()
        val_feat_vals = feat_vals[data.val_mask.cpu().numpy()]  # restrict to val set
        unique_feature_vals = np.unique(feat_vals)
        is_binary_feature = len(unique_feature_vals) == 2

        # Decide sampling values
        if sensitive_feature_values is not None:
            candidate_values = np.array(sensitive_feature_values)
        else:
            val_min, val_max = val_feat_vals.min(), val_feat_vals.max()
            candidate_values = None

        num_nodes = original_x.size(0)
        X_copies = []

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
            for _ in range(K):
                x_mod = original_x.clone()
                if candidate_values is not None:
                    sampled_vals = torch.tensor(
                        np.random.choice(candidate_values, size=num_nodes),
                        dtype=x_mod.dtype, device=x_mod.device
                    )
                else:
                    sampled_vals = torch.empty(num_nodes, dtype=x_mod.dtype, device=x_mod.device).uniform_(val_min, val_max)
                x_mod[:, feat_idx] = sampled_vals
                X_copies.append(x_mod)

        probs_all = []
        preds_all = []
        with torch.no_grad():
            for x_mod in X_copies:
                sig, pred = _predict_signal(x_mod, need_preds=(compute_flips and not is_regression))
                probs_all.append(sig)
                if compute_flips and not is_regression and pred is not None:
                    preds_all.append(pred)

        data.x = original_x  # restore

        probs_all = torch.stack(probs_all, dim=0)
        if is_binary_feature:
            probs_all = torch.cat([base_signal.unsqueeze(0), probs_all], dim=0)

        selected_probs = probs_all
        mean_prob = selected_probs.mean(0)
        var_prob = selected_probs.var(0)
        rel_var = var_prob / (mean_prob.pow(2) + eps) if relative else None

        flip_fraction = None
        correct_fraction = None
        if compute_flips and not is_regression and len(preds_all) > 0 and preds_orig is not None:
            preds_all = torch.stack(preds_all, dim=0)  # [K, N]
            preds_orig_expanded = preds_orig.view(1, -1).expand(preds_all.size(0), -1)
            flip_fraction = (preds_all != preds_orig_expanded).float().mean(dim=0)

            y_true = data.y.cpu()
            y_true_expanded = y_true.view(1, -1).expand(preds_all.size(0), -1)
            correct_fraction = (preds_all == y_true_expanded).float().mean(dim=0)

        def df(split, mask):
            idx = mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
            df_dict = {"Node": idx, "VarProb": var_prob[idx].numpy(), "Split": split}
            if rel_var is not None:
                df_dict["RelVarProb"] = rel_var[idx].numpy()
            if flip_fraction is not None:
                df_dict["FlipFraction"] = flip_fraction[idx].numpy()
            if correct_fraction is not None:
                df_dict["CorrectFraction"] = correct_fraction[idx].numpy()
            return pd.DataFrame(df_dict)

        if data.val_mask is None or data.test_mask is None:
            return df("ALL", torch.ones(num_nodes, dtype=torch.bool))

        return df("VAL", data.val_mask), df("TEST", data.test_mask)
    finally:
        pyg_typing.WITH_PYG_LIB = prev_with_pyg_lib
