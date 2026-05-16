from typing import Tuple

import torch

# Import from the seed-gnn directory
import sys
sys.path.append('/home/model_editing/gnn-editing-exploration/seed-gnn')
from edit_gnn.utils import prediction, select_edit_target_nodes as seed_select_edit_target_nodes


@torch.no_grad()
def _get_validation_view(model, whole_data):
    logits = prediction(model, whole_data)
    y_true = whole_data.y
    val_mask = whole_data.val_mask
    val_idx = val_mask.nonzero(as_tuple=True)[0]
    is_regression = bool(
        getattr(whole_data, "task_type", "") == "regression"
        or whole_data.y.dtype.is_floating_point
    )
    if is_regression:
        if logits.dim() == 2 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        val_pred = logits[val_mask]
        val_y_true = y_true[val_mask].to(val_pred.dtype)
        val_logits = val_pred
        residual = torch.abs(val_pred - val_y_true)
        return val_idx, val_logits, val_y_true, val_pred, residual, True

    val_logits = logits[val_mask]
    val_y_true = y_true[val_mask]
    val_pred = val_logits.argmax(dim=-1)
    mis_mask = val_pred.ne(val_y_true)
    return val_idx, val_logits, val_y_true, val_pred, mis_mask, False


def select_targets_default(model, whole_data, num_classes: int, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
    is_regression = bool(
        getattr(whole_data, "task_type", "") == "regression"
        or whole_data.y.dtype.is_floating_point
    )
    if is_regression:
        val_idx = whole_data.val_mask.nonzero(as_tuple=True)[0]
        if val_idx.numel() == 0:
            val_idx = whole_data.train_mask.nonzero(as_tuple=True)[0]
        if val_idx.numel() == 0:
            val_idx = torch.arange(whole_data.y.size(0), device=whole_data.y.device)
        k = min(num_samples, val_idx.size(0))
        perm = torch.randperm(val_idx.size(0), device=val_idx.device)[:k]
        chosen = val_idx[perm]
        target = whole_data.y[chosen]
        return chosen, target
    return seed_select_edit_target_nodes(
        model=model, whole_data=whole_data, num_classes=num_classes, num_samples=num_samples, from_valid_set=True
    )


def select_targets_hard_misclassified_valid(model, whole_data, num_classes: int, num_samples: int):
    val_idx, val_logits, val_y_true, val_pred, aux_mask, is_regression = _get_validation_view(model, whole_data)
    if is_regression:
        residual = aux_mask
        if residual.numel() == 0:
            return select_targets_default(model, whole_data, num_classes, num_samples)
        k = min(num_samples, residual.size(0))
        topk = torch.topk(residual, k=k, largest=True).indices
        chosen = val_idx[topk]
        target = whole_data.y[chosen]
        return chosen, target
    mis_mask = aux_mask
    if mis_mask.sum().item() == 0:
        return select_targets_default(model, whole_data, num_classes, num_samples)
    idx = torch.arange(val_logits.size(0), device=val_logits.device)
    true_logit = val_logits[idx, val_y_true]
    masked = val_logits.clone()
    masked[idx, val_y_true] = -1e9
    max_other, _ = masked.max(dim=1)
    margin = (max_other - true_logit)
    mis_margins = margin[mis_mask]
    mis_indices = val_idx[mis_mask]
    k = min(num_samples, mis_indices.size(0))
    topk = torch.topk(mis_margins, k=k, largest=True).indices
    chosen = mis_indices[topk]
    flipped_label = whole_data.y[chosen]
    return chosen, flipped_label


def select_targets_random_misclassified_valid(model, whole_data, num_classes: int, num_samples: int):
    val_idx, _, _, _, aux_mask, is_regression = _get_validation_view(model, whole_data)
    if is_regression:
        if val_idx.numel() == 0:
            return select_targets_default(model, whole_data, num_classes, num_samples)
        k = min(num_samples, val_idx.size(0))
        perm = torch.randperm(val_idx.size(0), device=val_idx.device)[:k]
        chosen = val_idx[perm]
        target = whole_data.y[chosen]
        return chosen, target
    mis_mask = aux_mask
    mis_indices = val_idx[mis_mask]
    if mis_indices.numel() == 0:
        return select_targets_default(model, whole_data, num_classes, num_samples)
    k = min(num_samples, mis_indices.size(0))
    perm = torch.randperm(mis_indices.size(0), device=mis_indices.device)[:k]
    chosen = mis_indices[perm]
    flipped_label = whole_data.y[chosen]
    return chosen, flipped_label


def select_targets_low_confidence_correct_valid(model, whole_data, num_classes: int, num_samples: int):
    val_idx, val_logits, val_y_true, val_pred, aux_mask, is_regression = _get_validation_view(model, whole_data)
    if is_regression:
        residual = aux_mask
        if residual.numel() == 0:
            return select_targets_default(model, whole_data, num_classes, num_samples)
        k = min(num_samples, residual.size(0))
        topk = torch.topk(residual, k=k, largest=False).indices
        chosen = val_idx[topk]
        target = whole_data.y[chosen]
        return chosen, target
    mis_mask = aux_mask
    corr_mask = ~mis_mask
    if corr_mask.sum().item() == 0:
        return select_targets_default(model, whole_data, num_classes, num_samples)
    idx = torch.arange(val_logits.size(0), device=val_logits.device)
    true_logit = val_logits[idx, val_y_true]
    masked = val_logits.clone()
    masked[idx, val_y_true] = -1e9
    max_other, _ = masked.max(dim=1)
    conf_margin = (true_logit - max_other)
    corr_margins = conf_margin[corr_mask]
    corr_indices = val_idx[corr_mask]
    k = min(num_samples, corr_indices.size(0))
    topk = torch.topk(corr_margins, k=k, largest=False).indices
    chosen = corr_indices[topk]
    flipped_label = whole_data.y[chosen]
    return chosen, flipped_label

def select_targets_high_confidence_correct_valid(model, whole_data, num_classes: int, num_samples: int):
    """
    Select high-confidence correct validation nodes: choose correct preds with the largest
    confidence margin (true_logit - max_other_logit).
    """
    val_idx, val_logits, val_y_true, val_pred, aux_mask, is_regression = _get_validation_view(model, whole_data)
    if is_regression:
        residual = aux_mask
        if residual.numel() == 0:
            return select_targets_default(model, whole_data, num_classes, num_samples)
        k = min(num_samples, residual.size(0))
        topk = torch.topk(residual, k=k, largest=True).indices
        chosen = val_idx[topk]
        target = whole_data.y[chosen]
        return chosen, target
    mis_mask = aux_mask
    corr_mask = ~mis_mask
    if corr_mask.sum().item() == 0:
        return select_targets_default(model, whole_data, num_classes, num_samples)
    idx = torch.arange(val_logits.size(0), device=val_logits.device)
    true_logit = val_logits[idx, val_y_true]
    masked = val_logits.clone()
    masked[idx, val_y_true] = -1e9
    max_other, _ = masked.max(dim=1)
    conf_margin = (true_logit - max_other)
    corr_margins = conf_margin[corr_mask]
    corr_indices = val_idx[corr_mask]
    k = min(num_samples, corr_indices.size(0))
    topk = torch.topk(corr_margins, k=k, largest=True).indices
    chosen = corr_indices[topk]
    flipped_label = whole_data.y[chosen]
    return chosen, flipped_label



def select_edit_targets_by_strategy(model, whole_data, num_classes: int, num_samples: int, strategy: str):
    strategy = (strategy or 'default').lower()
    is_regression = bool(
        getattr(whole_data, "task_type", "") == "regression"
        or whole_data.y.dtype.is_floating_point
    )
    if is_regression and strategy in ('default', 'random_misclassified', 'random_misclassified_valid'):
        return select_targets_default(model, whole_data, num_classes, num_samples)
    if strategy == 'default':
        return select_targets_default(model, whole_data, num_classes, num_samples)
    if strategy in ('hard_misclassified', 'hard_misclassified_valid'):
        return select_targets_hard_misclassified_valid(model, whole_data, num_classes, num_samples)
    if strategy in ('random_misclassified', 'random_misclassified_valid'):
        return select_targets_random_misclassified_valid(model, whole_data, num_classes, num_samples)
    if strategy in ('low_confidence_correct', 'low_confidence_correct_valid'):
        return select_targets_low_confidence_correct_valid(model, whole_data, num_classes, num_samples)
    if strategy in ('high_confidence_correct', 'high_confidence_correct_valid'):
        return select_targets_high_confidence_correct_valid(model, whole_data, num_classes, num_samples)
    return select_targets_default(model, whole_data, num_classes, num_samples)


