from typing import Dict

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

from edit_gnn.utils import prediction, grab_input
from torch_geometric.loader import NeighborLoader
import torch_geometric.typing as pyg_typing

from editing_pipelines.utils.gat_neighbor_eval import (
    gat_backbone_num_layers,
    should_use_gat_neighbor_loader,
)


def compute_full_auc_pr_by_split(model_obj, whole_data) -> Dict[str, float]:
    model_obj.eval()
    try:
        model_device = next(model_obj.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    data_eval = whole_data.to(model_device)

    use_neighbor_eval = should_use_gat_neighbor_loader(model_obj, data_eval)
    logits_full = None
    if use_neighbor_eval:
        if getattr(pyg_typing, "WITH_PYG_LIB", False) and getattr(pyg_typing, "WITH_TORCH_SPARSE", False):
            pyg_typing.WITH_PYG_LIB = False
        data_cpu = whole_data.to("cpu")
        masks = [m for m in [data_cpu.train_mask, data_cpu.val_mask, data_cpu.test_mask] if m is not None]
        input_nodes = torch.zeros(int(data_cpu.num_nodes), dtype=torch.bool)
        for mask in masks:
            input_nodes |= mask.cpu()
        nl = gat_backbone_num_layers(model_obj)
        hop_layers = nl if nl > 0 else int(getattr(model_obj, "num_layers", 2))
        loader = NeighborLoader(
            data_cpu,
            input_nodes=input_nodes,
            num_neighbors=[10] * hop_layers,
            batch_size=1024,
            shuffle=False,
        )
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(model_device)
                logits_batch = model_obj(**grab_input(batch))
                seed_bs = int(batch.batch_size)
                seed_nodes = batch.n_id[:seed_bs].detach().cpu().long()
                logits_seed = logits_batch[:seed_bs].detach().cpu()
                if logits_full is None:
                    logits_full = torch.full(
                        (int(data_cpu.num_nodes), int(logits_seed.size(-1))),
                        float("nan"),
                        dtype=torch.float32,
                    )
                logits_full[seed_nodes] = logits_seed
        if logits_full is None:
            return {"train": float("nan"), "val": float("nan"), "test": float("nan")}
        data_eval = data_cpu
    else:
        with torch.no_grad():
            logits_full = prediction(model_obj, data_eval).detach().cpu()
        data_eval = data_eval.to("cpu")

    is_regression = bool(getattr(data_eval, "task_type", "") == "regression" or data_eval.y.dtype.is_floating_point)
    if is_regression:
        return {"train": float("nan"), "val": float("nan"), "test": float("nan")}

    def _auc_pr(mask) -> float:
        if mask is None:
            return float("nan")
        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return float("nan")
        y_true = data_eval.y[idx].cpu().numpy()
        logits_idx = logits_full[idx]
        valid_rows = ~torch.isnan(logits_idx).any(dim=1)
        if valid_rows.sum().item() == 0:
            return float("nan")
        y_true = y_true[valid_rows.cpu().numpy()]
        probs = torch.softmax(logits_idx[valid_rows], dim=1).cpu().numpy()
        n_cls = probs.shape[1]
        if n_cls < 2:
            return float("nan")
        try:
            if n_cls == 2:
                return float(average_precision_score(y_true, probs[:, 1]))
            y_oh = label_binarize(y_true, classes=np.arange(n_cls))
            return float(average_precision_score(y_oh, probs, average="macro"))
        except Exception:
            return float("nan")

    return {
        "train": _auc_pr(data_eval.train_mask),
        "val": _auc_pr(data_eval.val_mask),
        "test": _auc_pr(data_eval.test_mask),
    }

