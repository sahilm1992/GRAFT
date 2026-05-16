import os
import time
import logging
import re, json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import itertools
from tqdm import tqdm
logger = logging.getLogger("main")

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data.data import Data
import matplotlib.pyplot as plt

from torch_geometric.utils import degree
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.metrics import roc_curve, auc

from models.base import BaseModel
from edit_gnn.utils import grab_input, test, prediction


def _is_regression_data(data: Data) -> bool:
    return bool(
        getattr(data, "task_type", "") == "regression"
        or data.y.dtype.is_floating_point
    )


def _to_regression_vector(output: torch.Tensor) -> torch.Tensor:
    if output.dim() == 2 and output.size(-1) == 1:
        return output.squeeze(-1)
    return output


def _regression_success(pred: torch.Tensor, label: torch.Tensor) -> float:
    label_f = label.to(pred.dtype)
    scale = torch.std(label_f) if label_f.numel() > 1 else torch.abs(label_f).mean()
    tol = torch.clamp(scale * 0.1, min=1e-3)
    return float((torch.abs(pred - label_f) <= tol).float().mean().item())


def get_optimizer(model_config, model, pretrain=False):
    lr = model_config['pretrain_lr'] if pretrain else model_config['edit_lr']
    # foreach=True can trigger large temporary allocations in optimizer.step()
    # (notably for large GAT parameter sets). Default to False for memory safety.
    foreach = model_config.get('optimizer_foreach', False)
    if model_config['optim'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=foreach)
    elif model_config['optim'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, foreach=foreach)
    else:
        raise NotImplementedError

    return optimizer
    
def sorted_checkpoints(
        checkpoint_prefix, best_model_checkpoint, output_dir=None
    ) -> List[str]:
        ordering_and_checkpoint_path = []
        glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}_*")]

        for path in glob_checkpoints:
            regex_match = re.match(f".*{checkpoint_prefix}_([0-9]+)", path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        # Make sure we don't delete the best model.
        if best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(str(Path(best_model_checkpoint)))
            checkpoints_sorted[best_model_index], checkpoints_sorted[-1] = (
                checkpoints_sorted[-1],
                checkpoints_sorted[best_model_index],
            )
        return checkpoints_sorted


def save_model(model, save_path, checkpoint_prefix: str, epoch: int):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    best_model_checkpoint = os.path.join(save_path, f'{checkpoint_prefix}_{epoch}.pt')
    torch.save(model.state_dict(), best_model_checkpoint)
    checkpoints_sorted = sorted_checkpoints(checkpoint_prefix, best_model_checkpoint, save_path)
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - 1)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        os.remove(checkpoint)
    return best_model_checkpoint


def finetune_mlp(config, model, whole_data, train_data, batch_size, iters):
        input = grab_input(train_data)
        model.eval()

        # get the original GNN output embedding
        model.mlp_freezed = True
        with torch.no_grad():
            gnn_output = model(**input)
            model.gnn_output = model(**grab_input(whole_data)).cpu()
            log_gnn_output = F.log_softmax(gnn_output, dim=-1)
    
        # here we enable the MLP to be trained
        model.freeze_module(train=False)
        opt = get_optimizer(config['pipeline_params'], model)
        logger.info('start finetuning MLP')
        s = time.time()
        torch.cuda.synchronize()
        for i in tqdm(range(iters)):
            opt.zero_grad()
            idx = np.random.choice(train_data.num_nodes, batch_size)
            idx = torch.from_numpy(idx).to(gnn_output.device)
            MLP_output = model.MLP(train_data.x[idx])
            cur_batch_gnn_output = gnn_output[idx]
            log_prob = F.log_softmax(MLP_output + cur_batch_gnn_output, dim=-1)
            main_loss = F.cross_entropy(MLP_output + gnn_output[idx], train_data.y[idx])
            kl_loss = F.kl_div(log_prob, log_gnn_output[idx], log_target=True, reduction='batchmean')
            (kl_loss + main_loss).backward()
            opt.step()
        torch.cuda.synchronize()
        e = time.time()
        logger.info(f'fine tune MLP used: {e - s} sec.')


def finetune_gnn_mlp(config, model, whole_data, train_data):
    model.freeze_module(train=False)
    dataset = config['eval_params']['dataset']
    if dataset == 'flickr' or (dataset == 'reddit2' and config['pipeline_params']['model_name']) or \
        (dataset in ['amazoncomputers', 'amazonphoto', 'coauthorcs', 'coauthorphysics']):
        finetune_mlp(config=config, model=model, whole_data=whole_data, train_data=train_data, batch_size=512, iters=100)
    else:
        finetune_mlp(config=config, model=model, whole_data=whole_data, train_data=train_data, batch_size=32, iters=100)
    bef_edit_ft_results = test(model, whole_data)
    ft_train_acc, ft_valid_acc, ft_test_acc = bef_edit_ft_results
    logger.info(f'before edit, after fine tune, train acc {ft_train_acc}, valid acc {ft_valid_acc}, test acc {ft_test_acc}')

    return bef_edit_ft_results


def bef_edit_check(model, whole_data, idx, label, curr_edit_target):
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    input = grab_input(whole_data)
    is_regression = _is_regression_data(whole_data)
    if model.__class__.__name__ in ['GCN_MLP', 'SAGE_MLP', 'GAT_MLP', 'GIN_MLP', 'Polynormer_MLP']:
        out = model.fast_forward(input['x'][idx], idx)
        if is_regression:
            y_pred = _to_regression_vector(out)
        else:
            y_pred = out.argmax(dim=-1)
    else:
        out = model(**input)
        if is_regression:
            y_pred = _to_regression_vector(out)[idx]
        else:
            y_pred = out.argmax(dim=-1)[idx]

    if label.shape[0] == 1:
        if is_regression:
            success = (_regression_success(y_pred.view(-1), label.view(-1)) == 1.0)
        else:
            success = (y_pred == label)
    else:
        if is_regression:
            success = _regression_success(y_pred.view(-1), label.view(-1))
        else:
            success = 1.0 if y_pred.eq(label)[curr_edit_target] else 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return success


def single_edit(
    model,
    whole_data,
    idx,
    label,
    optimizer,
    max_num_step,
    num_edit_targets=1,
    debug=False,
):
    s = time.time()
    is_regression = _is_regression_data(whole_data)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    success = False

    for step in range(1, max_num_step + 1):
        optimizer.zero_grad()
        input = grab_input(whole_data)
        if model.__class__.__name__ in ['GCN_MLP', 'SAGE_MLP', 'GAT_MLP', 'GIN_MLP', 'Polynormer_MLP']:
            out = model.fast_forward(input['x'][idx], idx)
            if is_regression:
                pred = _to_regression_vector(out)
                loss = F.mse_loss(pred, label.to(pred.dtype))
                y_pred = pred
            else:
                loss = F.cross_entropy(out, label)
                y_pred = out.argmax(dim=-1)
        else:
            out = model(**input)
            if is_regression:
                pred = _to_regression_vector(out)[idx]
                loss = F.mse_loss(pred, label.to(pred.dtype))
                y_pred = pred
            else:
                loss = F.cross_entropy(out[idx], label)
                y_pred = out.argmax(dim=-1)[idx]
        loss.backward()
        optimizer.step()
        if label.shape[0] == 1:
            if is_regression:
                success = (_regression_success(y_pred.view(-1), label.view(-1)) == 1.0)
            else:
                success = y_pred == label
        else:
            if is_regression:
                success = _regression_success(
                    y_pred[:num_edit_targets].view(-1),
                    label[:num_edit_targets].view(-1),
                )
            else:
                success = int(y_pred[:num_edit_targets].eq(label[:num_edit_targets])[:num_edit_targets].sum()) / num_edit_targets
        if success == 1.:
            break
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    e = time.time()
    mem_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    if debug:
        logger.info(f'max allocated mem: {mem_mb} MB')
        logger.info(f'edit time: {e - s}')
    return model, success, loss, step, mem_mb, e - s



def edit(
    model,
    whole_data,
    idx,
    f_label,
    optimizer,
    max_num_step,
    num_edit_targets=1,
    curr_edit_target=0,
    debug=False,
):
    bef_edit_success = bef_edit_check(model, whole_data, idx, f_label,curr_edit_target=curr_edit_target)
    if bef_edit_success == 1.:
        return model, bef_edit_success, 0, 0, 0, 0

    return single_edit(
        model,
        whole_data,
        idx,
        f_label,
        optimizer,
        max_num_step,
        num_edit_targets=num_edit_targets,
        debug=debug,
    )

def process_raw_exp_results(raw_results):
    processed_results = {}

    processed_results['bef_edit_tst_acc'] = raw_results['bef_edit_tst_acc']
    processed_results['selected_result'] = raw_results['selected_result']
    processed_results['highest_dd'] = raw_results['highest_dd']
    processed_results['average_dd'] = raw_results['average_dd']
    processed_results['average_sucess_rate'] = raw_results['average_success_rate']
    if 'editable_param_count' in raw_results:
        processed_results['editable_param_count'] = raw_results['editable_param_count']
    if 'total_param_count' in raw_results:
        processed_results['total_param_count'] = raw_results['total_param_count']
    
    return processed_results

def plot_roc_auc(ax, y_true, y_score, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=9, wrap=True)
    ax.legend(loc="lower right")

def plot_filtered_logit_change_vs_degree(
    ax, baseline_logits, corrupted_logits, degrees, 
    title="Avg Logit Change (high-change nodes)"
):
    eps = 1e-6
    num = (corrupted_logits - baseline_logits).abs()
    den = (corrupted_logits.abs() + baseline_logits.abs()) / 2.0 + eps
    logit_diff_pct = 100 * num / den   # symmetrized relative change (%)

    max_change_per_node = logit_diff_pct.max(dim=1).values  # max class change per node

    # Filter nodes with "large" changes: keep top 20%
    threshold = np.percentile(max_change_per_node.cpu().numpy(), 80)
    mask = max_change_per_node > threshold

    deg_int = degrees[mask].to(torch.long).cpu().numpy()
    changes = max_change_per_node[mask].cpu().numpy()

    if deg_int.size == 0:
        ax.text(0.5, 0.5, "No high-change nodes", ha="center", va="center")
        return

    # Aggregate averages per degree
    max_deg = deg_int.max() if deg_int.size > 0 else 0
    xs = np.arange(max_deg + 1)
    sums = np.bincount(deg_int, weights=changes, minlength=max_deg + 1)
    counts = np.bincount(deg_int, minlength=max_deg + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.where(counts > 0, sums / counts, np.nan)

    ax.scatter(deg_int, changes, c=deg_int, cmap="coolwarm", alpha=0.5, s=8)
    ax.plot(xs, means, color="black", linewidth=2.0, label="avg (filtered)")
    ax.set_xlabel("Node Degree")
    ax.set_ylabel("Logit Change % (max across classes)")
    ax.set_title(title, fontsize=9, wrap=True)
    ax.legend()

def plot_auc_high_change_nodes(ax, baseline_logits, corrupted_logits, y_true, 
                               top_pct=20, title="ROC AUC (High-Change Nodes)"):
    """
    Plot ROC/AUC for the top-X% high-change nodes.
    Computes high-change mask based on *relative logit change*, 
    and evaluates both baseline and corrupted AUCs on the same subset.
    """

    eps = 1e-6
    num = (corrupted_logits - baseline_logits).abs()
    den = (corrupted_logits.abs() + baseline_logits.abs()) / 2.0 + eps
    logit_diff_pct = 100 * num / den

    max_change_per_node = logit_diff_pct.max(dim=1).values.cpu().numpy()

    # Compute mask using *absolute change*, independent of either output
    threshold = np.percentile(max_change_per_node, 100 - top_pct)
    mask = max_change_per_node > threshold

    if mask.sum() == 0:
        ax.text(0.5, 0.5, "No high-change nodes", ha="center", va="center")
        return

    y_true_sel = y_true[mask].cpu().numpy()
    baseline_sel = baseline_logits[mask].detach().cpu()
    corrupted_sel = corrupted_logits[mask].detach().cpu()

    if baseline_sel.shape[1] == 2:
        # Convert to softmax probabilities
        y_score_base = F.softmax(baseline_sel, dim=1)[:, 1].numpy()
        y_score_corr = F.softmax(corrupted_sel, dim=1)[:, 1].numpy()

        # Compute AUCs
        fpr_b, tpr_b, _ = roc_curve(y_true_sel, y_score_base)
        auc_b = auc(fpr_b, tpr_b)

        fpr_c, tpr_c, _ = roc_curve(y_true_sel, y_score_corr)
        auc_c = auc(fpr_c, tpr_c)

        # Plot both
        ax.plot(fpr_b, tpr_b, lw=2, linestyle="--", label=f"Before AUC={auc_b:.5f}")
        ax.plot(fpr_c, tpr_c, lw=2, label=f"After AUC={auc_c:.5f}")

        ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title, fontsize=9, wrap=True)
        ax.legend(loc="lower right")

        # Debug print
        print(f"[AUC Δ] Before={auc_b:.4f}, After={auc_c:.4f}, Δ={auc_c - auc_b:.4f}, nodes={mask.sum()}")
    else:
        ax.text(0.5, 0.5, "ROC only for binary", ha="center", va="center")
        ax.set_title(title, fontsize=9, wrap=True)


def plot_logit_change_histogram(
    ax, baseline_logits, corrupted_logits, bins=50, 
    title="Histogram of Logit Changes"
):
    eps = 1e-6
    num = (corrupted_logits - baseline_logits).abs()
    den = (corrupted_logits.abs() + baseline_logits.abs()) / 2.0 + eps
    logit_diff_pct = 100 * num / den   # symmetrized relative change (%)

    max_change_per_node = logit_diff_pct.max(dim=1).values.cpu().numpy()

    ax.hist(max_change_per_node, bins=bins, color="skyblue", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Max Logit Change per Node (%)")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=9, wrap=True)

def plot_auc_binned_logit_change(
    ax, baseline_logits, corrupted_logits, y_true, 
    num_bins=4, title="ROC AUC by Logit Change Bins"
):
    """
    Plot ROC/AUC curves for bins of nodes grouped by *absolute logit change magnitude*.
    Each bin is defined based on the relative change between baseline and corrupted logits.
    Both "before" and "after" AUCs are computed on the same bin subset.

    Args:
        ax (matplotlib.axes.Axes): Axis to plot on.
        baseline_logits (torch.Tensor): Baseline validation logits [N, C].
        corrupted_logits (torch.Tensor): Corrupted validation logits [N, C].
        y_true (torch.Tensor): True labels for validation nodes [N].
        num_bins (int): Number of bins by change magnitude.
        title (str): Plot title.
    """
    eps = 1e-6
    # Relative symmetrized change (%)
    num = (corrupted_logits - baseline_logits).abs()
    den = (corrupted_logits.abs() + baseline_logits.abs()) / 2.0 + eps
    logit_diff_pct = 100 * num / den

    max_change_per_node = logit_diff_pct.max(dim=1).values.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    baseline_np = baseline_logits.detach().cpu()
    corrupted_np = corrupted_logits.detach().cpu()

    # Define bins by quantiles of *absolute logit change*
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(max_change_per_node, quantiles)

    for b in range(num_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (max_change_per_node >= lo) & (max_change_per_node < hi)
        if mask.sum() == 0:
            continue

        y_bin = y_true_np[mask]
        base_bin = baseline_np[mask]
        corr_bin = corrupted_np[mask]

        if base_bin.shape[1] == 2:  # binary classification
            y_score_base = F.softmax(base_bin, dim=1)[:, 1].numpy()
            y_score_corr = F.softmax(corr_bin, dim=1)[:, 1].numpy()

            fpr_b, tpr_b, _ = roc_curve(y_bin, y_score_base)
            auc_b = auc(fpr_b, tpr_b)

            fpr_c, tpr_c, _ = roc_curve(y_bin, y_score_corr)
            auc_c = auc(fpr_c, tpr_c)

            ax.plot(
                fpr_b, tpr_b, lw=1.5, linestyle="--",
                label=f"Bin {b+1}: {lo:.1f}-{hi:.1f}% | Before AUC={auc_b:.5f}"
            )
            ax.plot(
                fpr_c, tpr_c, lw=1.5,
                label=f"Bin {b+1}: {lo:.1f}-{hi:.1f}% | After AUC={auc_c:.5f}"
            )

            print(f"[BIN {b+1}] ΔAUC = {auc_c - auc_b:+.4f} | range={lo:.1f}-{hi:.1f}% | nodes={mask.sum()}")

    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=9, wrap=True)
    ax.legend(fontsize=7, loc="lower right")

def plot_rate_vs_degree(ax, y_true, y_pred, degrees, title="Rate vs Degree"):
    # Aggregate overall misclassification rate per integer degree
    deg_int = degrees.to(torch.long).cpu().numpy()
    misclassified = (y_true != y_pred).cpu().numpy().astype(int)

    max_deg = deg_int.max() if deg_int.size > 0 else 0
    counts = np.bincount(deg_int, minlength=max_deg + 1)
    mis_sums = np.bincount(deg_int, weights=misclassified, minlength=max_deg + 1)

    with np.errstate(divide='ignore', invalid='ignore'):
        rates = np.where(counts > 0, mis_sums / counts, np.nan)

    xs = np.arange(len(counts))
    ax.plot(xs, rates, marker='o', linewidth=1.5)
    ax.set_xlabel("Node Degree")
    ax.set_ylabel("Misclassification Rate (overall)")
    # Mention counts in the title
    nonzero_pairs = [(d, int(c)) for d, c in zip(xs, counts) if c > 0]
    ax.set_title(f"{title} | counts(sample): {nonzero_pairs[:8]}{' ...' if len(nonzero_pairs) > 8 else ''}", fontsize=9, wrap=True)


def plot_per_class_misclass(ax, y_true, y_pred, degrees, num_classes, title="Per-Class Misclassification vs Degree"):
    # Plot a line per class: misclassification rate vs integer degree
    deg_int = degrees.to(torch.long)
    xs_all = torch.arange(int(deg_int.max().item()) + 1) if deg_int.numel() > 0 else torch.arange(1)

    for c in range(num_classes):
        cls_mask = (y_true == c)
        if cls_mask.sum() == 0:
            continue
        deg_cls = deg_int[cls_mask]
        y_pred_cls = y_pred[cls_mask]
        max_deg_c = int(deg_cls.max().item()) if deg_cls.numel() > 0 else 0
        xs_c = torch.arange(max_deg_c + 1)
        counts = torch.bincount(deg_cls, minlength=max_deg_c + 1).cpu().numpy()
        mis = (y_pred_cls != c).to(torch.long)
        mis_sums = torch.bincount(deg_cls, weights=mis, minlength=max_deg_c + 1).cpu().numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            rates = np.where(counts > 0, mis_sums / counts, np.nan)
        ax.plot(xs_c.numpy(), rates, marker='o', linewidth=1.0, label=f"class {c}")

    ax.set_xlabel("Node Degree")
    ax.set_ylabel("Misclassification Rate (per class)")
    ax.set_title(title, fontsize=9, wrap=True)
    ax.legend(fontsize=8, ncol=2)


def plot_true_logit_probs(ax, val_logits, y_true, degrees, title="True Logit vs Degree"):
    probs = F.softmax(val_logits, dim=1)
    idx = torch.arange(y_true.size(0), device=y_true.device)
    true_probs = probs[idx, y_true]
    deg_int = degrees.to(torch.long)

    # Scatter colored by degree
    sc = ax.scatter(deg_int.cpu().numpy(), true_probs.cpu().numpy(), c=deg_int.cpu().numpy(), cmap='viridis', alpha=0.5, s=8)

    # Average line: mean probability per degree
    max_deg = int(deg_int.max().item()) if deg_int.numel() > 0 else 0
    xs = np.arange(max_deg + 1)
    sums = np.bincount(deg_int.cpu().numpy(), weights=true_probs.cpu().numpy(), minlength=max_deg + 1)
    counts = np.bincount(deg_int.cpu().numpy(), minlength=max_deg + 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        means = np.where(counts > 0, sums / counts, np.nan)
    ax.plot(xs, means, color='red', linewidth=2.0, label='avg')

    ax.set_xlabel("Node Degree")
    ax.set_ylabel("True-Class Probability (softmax)")
    ax.set_title(title, fontsize=9, wrap=True)
    ax.legend()
    try:
        from matplotlib import pyplot as _plt
        cbar = _plt.colorbar(sc, ax=ax)
        cbar.set_label('Degree')
    except Exception:
        pass


# --------------------------
# Corruption function
# --------------------------

def corrupt_features(data, features, combo, mask):
    """
    Return a shallow copy with selected features corrupted in the validation set.

    For each feature in 'combo':
      - If the feature is binary (unique values subset of {0,1}), randomly flips 0↔1.
      - If non-binary, shuffles its values among validation nodes.

    Args:
        data (torch_geometric.data.Data): Input graph data.
        features (list[str]): Optional list of feature names (for readability/logging).
        combo (list[int]): List of feature indices to corrupt.
        mask (torch.BoolTensor): Validation mask (True for nodes to corrupt).

    Returns:
        corrupted_data (torch_geometric.data.Data): Copy of data with corrupted x.
    """
    corrupted_data = data.clone()
    x = corrupted_data.x.clone()
    val_idx = mask.nonzero(as_tuple=True)[0]

    for f in combo:
        feat_vals = x[val_idx, f]
        unique_vals = torch.unique(feat_vals)

        # check for binary feature
        if set(unique_vals.tolist()) <= {0, 1}:
            # flip 0↔1 for binary features
            flip_mask = torch.rand(len(val_idx)) < 0.5
            x[val_idx[flip_mask], f] = 1 - x[val_idx[flip_mask], f]
            action = "flipped"
        else:
            # shuffle for continuous/non-binary
            perm = torch.randperm(len(val_idx))
            x[val_idx, f] = x[val_idx[perm], f]
            action = "shuffled"

        if features and f < len(features):
            feat_name = features[f]
        else:
            feat_name = f"f{f}"

        print(f"[corrupt_features] Feature {feat_name} ({action}) on {len(val_idx)} validation nodes")

    corrupted_data.x = x
    return corrupted_data



# --------------------------
# Main visualization function
# --------------------------

def _resolve_feature_indices(features_to_corrupt: List[Union[int, str]], feature_names: Optional[List[str]]) -> List[int]:
    resolved: List[int] = []
    for f in features_to_corrupt:
        if isinstance(f, int):
            resolved.append(f)
        elif isinstance(f, str):
            if feature_names is None:
                raise ValueError("Feature names not available on data; cannot resolve string feature '" + f + "'.")
            try:
                resolved.append(feature_names.index(f))
            except ValueError:
                raise ValueError(f"Feature name '{f}' not found. Available: {feature_names[:10]} ...")
        else:
            raise TypeError("Unsupported feature identifier type: " + str(type(f)))
    return resolved


def visualize_validation(config, model, whole_data, node_idx_2flip, suffix: str = "", file_suffix: str = ""):
    model.eval()

    # Setup directories
    vis_root = os.path.join(config['management']['output_folder_dir'], "visualization_plots")
    vis_dir = os.path.join(vis_root, f"{config['eval_params']['dataset']}_{config['pipeline_params']['model_name']}")
    os.makedirs(vis_dir, exist_ok=True)

    dataset = config['eval_params']['dataset']
    model_name = config['pipeline_params']['model_name']
    # Allow missing corruption config (e.g., during pretraining). Default to baseline only.
    features_to_corrupt_cfg = config.get('corruption', {}).get('features', [])
    feature_names = getattr(whole_data, 'feature_names', None)
    features_to_corrupt = _resolve_feature_indices(features_to_corrupt_cfg, feature_names)
    # Log which features will be considered for corruption
    if isinstance(features_to_corrupt_cfg, list) and len(features_to_corrupt_cfg) > 0:
        try:
            resolved_names = [feature_names[i] if feature_names is not None else str(i) for i in features_to_corrupt]
        except Exception:
            resolved_names = [str(i) for i in features_to_corrupt]
        logger.info(f"Corruption configured: raw={features_to_corrupt_cfg} | indices={features_to_corrupt} | names={resolved_names}")
    else:
        logger.info("Corruption configured: none (baseline only)")

    # Build corruption scenarios (powerset)
    corruption_scenarios = [()]  # baseline
    for r in range(1, len(features_to_corrupt) + 1):
        corruption_scenarios += list(itertools.combinations(features_to_corrupt, r))

    n = len(corruption_scenarios)

    # Create subplot figures in grid layout
    import math
    rows = int(math.ceil(math.sqrt(n)))
    cols = int(math.ceil(n / rows))
    fig_roc, axs_roc = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), constrained_layout=True)
    fig_deg, axs_deg = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), constrained_layout=True)
    fig_class, axs_class = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), constrained_layout=True)
    fig_probs, axs_probs = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), constrained_layout=True)
    fig_filtered, axs_filtered = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), constrained_layout=True)
    fig_hist, axs_hist = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), constrained_layout=True)
    fig_auc_high, axs_auc_high = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), constrained_layout=True)
    fig_auc_bins, axs_auc_bins = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), constrained_layout=True)

    # Normalize single-plot case
    def _axis_at(axs, i):
        if rows * cols == 1:
            return axs
        if not isinstance(axs, np.ndarray):
            return axs
        return axs.flatten()[i]

    # Run baseline ONCE before the loop
    with torch.no_grad():
        baseline_logits_full = model(**grab_input(whole_data))
    baseline_val_logits = baseline_logits_full[whole_data.val_mask]


    for idx, combo in enumerate(corruption_scenarios):
        # Corrupt ONLY validation nodes
        # Log current corruption combo
        try:
            combo_names = [feature_names[i] if feature_names is not None else str(i) for i in combo]
        except Exception:
            combo_names = [str(i) for i in combo]
        if len(combo) == 0:
            logger.info("Applying corruption scenario [baseline]: no features corrupted")
        else:
            logger.info(f"Applying corruption scenario: indices={list(combo)} | names={combo_names}")

        corrupted_data = corrupt_features(whole_data, feature_names, combo, whole_data.val_mask) \
                         if combo else whole_data

        # Forward pass
        with torch.no_grad():
            logits = model(**grab_input(corrupted_data))

        pred = logits.argmax(dim=-1)

        # Validation slice
        val_y_true = whole_data.y[whole_data.val_mask]
        val_y_pred = pred[whole_data.val_mask]
        val_logits = logits[whole_data.val_mask]

        if combo:
            diff_mean = (val_logits - baseline_val_logits).abs().mean().item()
            print(f"[DEBUG] Mean absolute logit diff for {combo_names}: {diff_mean:.4f}")


        # Degrees
        if hasattr(whole_data, 'edge_index') and whole_data.edge_index is not None:
            row_indices = whole_data.edge_index[0]
        elif hasattr(whole_data, 'adj_t') and whole_data.adj_t is not None:
            row_indices = whole_data.adj_t.storage.row()
        else:
            raise ValueError('Neither edge_index nor adj_t available')
        degrees = degree(row_indices, num_nodes=whole_data.num_nodes)[whole_data.val_mask]

        # --- Labels ---
        if not combo:
            corruption_label = "Baseline"
        else:
            try:
                combo_names = [feature_names[i] if feature_names is not None else str(i) for i in combo]
            except Exception:
                combo_names = [str(i) for i in combo]
            corruption_label = f"Features {combo_names}"
        base_title = f"Dataset={dataset} | Model={model_name} | Corr={corruption_label}"

        # --- ROC (binary only for now) ---
        roc_ax = _axis_at(axs_roc, idx)
        deg_ax = _axis_at(axs_deg, idx)
        cls_ax = _axis_at(axs_class, idx)
        prob_ax = _axis_at(axs_probs, idx)

        if logits.shape[1] == 2:
            y_true_bin = val_y_true.cpu().numpy()
            y_score_bin = F.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            plot_roc_auc(roc_ax, y_true_bin, y_score_bin, title=f"ROC | {base_title}")
        else:
            roc_ax.text(0.5, 0.5, "ROC only for binary", ha="center", va="center")
            roc_ax.set_title(f"ROC | {base_title}", fontsize=9, wrap=True)

        # --- Misclass vs degree ---
        plot_rate_vs_degree(deg_ax, val_y_true, val_y_pred, degrees, title=f"Rate vs Degree | {base_title}")

        # --- Per-class misclassification vs degree ---
        num_classes = logits.shape[1]
        plot_per_class_misclass(cls_ax, val_y_true, val_y_pred, degrees, num_classes, title=f"Per-Class vs Degree | {base_title}")

        # --- True logit probabilities scatter vs degree ---
        plot_true_logit_probs(prob_ax, val_logits, val_y_true, degrees, title=f"True Logit vs Degree | {base_title}")

        filtered_ax = _axis_at(axs_filtered, idx)
        hist_ax = _axis_at(axs_hist, idx)

        if combo:

            plot_filtered_logit_change_vs_degree(
            filtered_ax, baseline_val_logits, val_logits, degrees,
            title=f"Filtered Logit Change vs Degree | {base_title}"
        )

            plot_logit_change_histogram(
                hist_ax, baseline_val_logits, val_logits,
                title=f"Histogram of Logit Changes | {base_title}"
            )

        auc_high_ax = _axis_at(axs_auc_high, idx)
        plot_auc_high_change_nodes(
            auc_high_ax, baseline_val_logits, val_logits, val_y_true,
            top_pct=20,
            title=f"ROC AUC (High-Change Nodes) | {base_title}"
        )

        auc_bin_ax = _axis_at(axs_auc_bins, idx)
        
        plot_auc_binned_logit_change(
            auc_bin_ax, baseline_val_logits, val_logits, val_y_true,
            num_bins=4,
            title=f"ROC AUC by Bins | {base_title}"
        )


    # Save only combined figures
    # Hide any unused subplots to improve spacing
    def _hide_unused(axs):
        if isinstance(axs, np.ndarray) and axs.size > n:
            for ax in axs.flatten()[n:]:
                ax.set_visible(False)
    _hide_unused(axs_roc)
    _hide_unused(axs_deg)
    _hide_unused(axs_class)
    _hide_unused(axs_probs)

    fig_roc.savefig(os.path.join(vis_dir, f"roc_auc_all_{dataset}_{model_name}{file_suffix}.png"))
    fig_deg.savefig(os.path.join(vis_dir, f"rate_vs_degree_all_{dataset}_{model_name}{file_suffix}.png"))
    fig_class.savefig(os.path.join(vis_dir, f"perclass_misclass_all_{dataset}_{model_name}{file_suffix}.png"))
    fig_probs.savefig(os.path.join(vis_dir, f"true_logit_probs_all_{dataset}_{model_name}{file_suffix}.png"))
    fig_filtered.savefig(os.path.join(vis_dir, f"filtered_logit_change_vs_degree_all_{dataset}_{model_name}{file_suffix}.png"))
    fig_hist.savefig(os.path.join(vis_dir, f"logit_change_histograms_all_{dataset}_{model_name}{file_suffix}.png"))
    fig_auc_high.savefig(os.path.join(vis_dir, f"roc_auc_highchange_all_{dataset}_{model_name}{file_suffix}.png"))
    fig_auc_bins.savefig(os.path.join(vis_dir, f"roc_auc_binned_all_{dataset}_{model_name}{file_suffix}.png"))
    
    plt.close(fig_auc_bins)
    plt.close(fig_auc_high)
    plt.close(fig_filtered)
    plt.close(fig_hist)
    plt.close(fig_roc)
    plt.close(fig_deg)
    plt.close(fig_class)
    plt.close(fig_probs)

    sensitive_features = config['eval_params']['subgroup_features']

    with torch.no_grad():
        logits = model(**grab_input(whole_data))
    pred = logits.argmax(dim=-1)
    val_y_true = whole_data.y[whole_data.val_mask]
    val_y_pred = pred[whole_data.val_mask]
    unique_classes = torch.unique(val_y_true).cpu().numpy()

    for feat in sensitive_features:
        if feat not in feature_names:
            continue
        feat_idx = feature_names.index(feat)
        subgroup_vals = whole_data.x[whole_data.val_mask, feat_idx].cpu().numpy()
        unique_vals = np.unique(subgroup_vals)

        # Overall accuracy
        overall_accs = [
            (val_y_true[subgroup_vals == val] == val_y_pred[subgroup_vals == val]).float().mean().item()
            for val in unique_vals if (subgroup_vals == val).sum() > 0
        ]
        subgroup_labels = [str(val) for val in unique_vals if (subgroup_vals == val).sum() > 0]

        plt.figure(figsize=(8, 5))
        plt.bar(subgroup_labels, overall_accs, color="skyblue", edgecolor="black")
        plt.ylabel("Accuracy")
        plt.xlabel(f"{feat} subgroup")
        plt.title(f"Subgroup Accuracy by {feat}")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=30)
        plt.savefig(os.path.join(vis_dir, f"subgroup_accuracy_{feat}_{dataset}_{model_name}{file_suffix}.png"), bbox_inches="tight")
        plt.close()

        # Per-class accuracy
        if len(unique_classes) > 1:
            fig, ax = plt.subplots(figsize=(max(8, len(subgroup_labels) * 1.5), 6))
            x = np.arange(len(subgroup_labels))
            width = 0.8 / len(unique_classes)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes)))
            
            for i, class_label in enumerate(unique_classes):
                class_accs = []
                for val in [float(lbl) for lbl in subgroup_labels]:
                    class_mask = (subgroup_vals == val) & (val_y_true.cpu().numpy() == class_label)
                    acc = (val_y_true[class_mask] == val_y_pred[class_mask]).float().mean().item() if class_mask.sum() > 0 else 0
                    class_accs.append(acc)
                
                ax.bar(x + i * width, class_accs, width, label=f'Class {class_label}', 
                       color=colors[i], edgecolor='black', alpha=0.8)
            
            ax.set_xlabel(f"{feat} subgroup")
            ax.set_ylabel("Per-Class Accuracy")
            ax.set_title(f"Per-Class Accuracy by {feat}")
            ax.set_xticks(x + width * (len(unique_classes) - 1) / 2)
            ax.set_xticklabels(subgroup_labels, rotation=30)
            ax.legend()
            ax.set_ylim(0, 1.0)
            
            plt.savefig(os.path.join(vis_dir, f"per_class_accuracy_{feat}_{dataset}_{model_name}{file_suffix}.png"), bbox_inches="tight")
            plt.close()