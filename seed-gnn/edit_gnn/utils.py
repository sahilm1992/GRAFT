import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from torch_geometric.data.data import Data

from models.base import BaseModel


def _is_regression_data(data: Data) -> bool:
    return bool(
        getattr(data, "task_type", "") == "regression"
        or data.y.dtype.is_floating_point
    )

def grab_input(data: Data, indices=None):
    x = data.x
    i = 1
    xs = [x]
    # for SIGN
    while hasattr(data, f'x{i}'):
        xs.append(getattr(data, f'x{i}'))
        i += 1
    if hasattr(data, 'adj_t') and data.adj_t is not None:
        return {"x": data.x, 'adj_t': data.adj_t}
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        return {"x": data.x, 'adj_t': data.edge_index}
    raise ValueError("Data object must contain either adj_t or edge_index.")


@torch.no_grad()
def prediction(model: BaseModel, data: Data):
    model.eval()
    input = grab_input(data)
    dev = next(model.parameters()).device
    x = input["x"]
    if x.device != dev:
        input["x"] = x.to(dev)
    adj = input.get("adj_t")
    if adj is not None and hasattr(adj, "device") and adj.device != dev:
        input["adj_t"] = adj.to(dev)
    return model(**input)

@torch.no_grad()
def compute_per_class_accuracy(logits, y, mask=None):
    """
    Returns dict: {class_id: accuracy}
    """
    dev = logits.device
    y = y.to(dev)
    if mask is not None:
        m = mask.to(dev)
        logits = logits[m]
        y = y[m]

    preds = logits.argmax(dim=-1)
    classes = torch.unique(y)

    per_class_acc = {}
    for c in classes:
        c = int(c.item())
        c_mask = (y == c)
        if c_mask.sum() == 0:
            per_class_acc[c] = None
        else:
            per_class_acc[c] = (
                preds[c_mask].eq(y[c_mask]).sum().item()
                / c_mask.sum().item()
            )

    return per_class_acc

def compute_metrics(logits, y, mask=None) -> Dict[str, float]:
    dev = logits.device
    y = y.to(dev)
    if mask is not None:
        m = mask.to(dev)
        logits, y = logits[m], y[m]

    if y.size(0) == 0:
        return {"acc": 0., "f1": 0., "balanced_recall": 0., "mcc": 0.}

    if y.dim() == 1:
        # Multiclass
        preds = logits.argmax(dim=-1)
        acc = preds.eq(y).sum().item() / y.size(0)
        
        num_classes = logits.size(-1)
        conf_matrix = torch.zeros(num_classes, num_classes, device=logits.device)
        for t, p in zip(y.view(-1), preds.view(-1)):
            conf_matrix[t.long(), p.long()] += 1
        
        tp = conf_matrix.diag()
        fp = conf_matrix.sum(dim=0) - tp
        fn = conf_matrix.sum(dim=1) - tp
        
        recall_per_class = tp / (tp + fn + 1e-8)
        balanced_recall = recall_per_class.mean().item()
        
        precision_per_class = tp / (tp + fp + 1e-8)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + 1e-8)
        f1 = f1_per_class.mean().item() # Macro-F1
        
        s = float(y.size(0))
        c = float(tp.sum().item())
        t_k = conf_matrix.sum(dim=1).to(torch.float64)
        p_k = conf_matrix.sum(dim=0).to(torch.float64)
        mcc_num = (c * s) - (t_k @ p_k).item()
        mcc_den = np.sqrt((s**2 - (p_k @ p_k).item()) * (s**2 - (t_k @ t_k).item()))
        mcc = mcc_num / (mcc_den + 1e-8)
        
        return {"acc": acc, "f1": f1, "balanced_recall": balanced_recall, "mcc": mcc}
    else:
        # Multi-label
        y_pred = logits > 0
        y_true = y > 0.5
        tp = (y_true & y_pred).sum().to(torch.float64)
        fp = (~y_true & y_pred).sum().to(torch.float64)
        fn = (y_true & ~y_pred).sum().to(torch.float64)
        tn = (~y_true & ~y_pred).sum().to(torch.float64)
        
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        balanced_recall = recall.item()
        
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (mcc_num / (mcc_den + 1e-8)).item()
        
        return {"acc": acc.item(), "f1": f1.item(), "balanced_recall": balanced_recall, "mcc": mcc}


def compute_micro_f1(logits, y, mask=None) -> float:
    return compute_metrics(logits, y, mask)["acc"]


class MetricsDict(dict):
    def __iter__(self):
        return iter(self["overall"])


@torch.no_grad()
def test(model: BaseModel, data: Data, specific_class: int = None):
    model.eval()
    out = prediction(model, data)
    is_regression = _is_regression_data(data)
    if is_regression:
        pred = out.squeeze(-1) if out.dim() == 2 and out.size(-1) == 1 else out
        y_true = data.y.to(device=pred.device, dtype=pred.dtype)

        def _reg_metrics(mask):
            m = mask.to(pred.device)
            pred_m = pred[m]
            y_m = y_true[m]
            if pred_m.numel() == 0:
                return {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan"), "r2": float("nan")}
            diff = pred_m - y_m
            mse = torch.mean(diff ** 2)
            mae = torch.mean(torch.abs(diff))
            rmse = torch.sqrt(mse)
            ss_res = torch.sum(diff ** 2)
            y_mean = torch.mean(y_m)
            ss_tot = torch.sum((y_m - y_mean) ** 2)
            r2 = float("nan") if ss_tot.item() <= 1e-12 else float(1.0 - (ss_res / (ss_tot + 1e-12)).item())
            return {
                "mae": float(mae.item()),
                "mse": float(mse.item()),
                "rmse": float(rmse.item()),
                "r2": r2,
            }

        train_metrics = _reg_metrics(data.train_mask)
        valid_metrics = _reg_metrics(data.val_mask)
        test_metrics = _reg_metrics(data.test_mask)

        return MetricsDict({
            "overall": (train_metrics["rmse"], valid_metrics["rmse"], test_metrics["rmse"]),
            "metrics": {
                "train": train_metrics,
                "val": valid_metrics,
                "test": test_metrics
            },
            "per_class": {
                "train": {},
                "val": {},
                "test": {}
            }
        })

    y_true = data.y
    train_mask = data.train_mask
    valid_mask = data.val_mask
    test_mask = data.test_mask
    if specific_class is not None:
        dev = out.device
        sel = data.y.to(dev) == specific_class
        out = out[sel]
        y_true = y_true.to(dev)[sel]
        train_mask = train_mask.to(dev)[sel]
        valid_mask = valid_mask.to(dev)[sel]
        test_mask = test_mask.to(dev)[sel]
    
    train_metrics = compute_metrics(out, y_true, train_mask)
    valid_metrics = compute_metrics(out, y_true, valid_mask)
    test_metrics = compute_metrics(out, y_true, test_mask)

    train_pc = compute_per_class_accuracy(out, y_true, train_mask)
    valid_pc = compute_per_class_accuracy(out, y_true, valid_mask)
    test_pc  = compute_per_class_accuracy(out, y_true, test_mask)

    return MetricsDict({
        "overall": (train_metrics["acc"], valid_metrics["acc"], test_metrics["acc"]),
        "metrics": {
            "train": train_metrics,
            "val": valid_metrics,
            "test": test_metrics
        },
        "per_class": {
            "train": train_pc,
            "val": valid_pc,
            "test": test_pc
        }
    })


def select_edit_target_nodes(model: BaseModel,
                            whole_data: Data,
                            num_classes: int,
                            num_samples: int,
                            from_valid_set: bool = True):
        model.eval()
        is_regression = _is_regression_data(whole_data)
        if is_regression:
            if from_valid_set:
                nodes_set = whole_data.val_mask.nonzero(as_tuple=False).view(-1)
            else:
                nodes_set = whole_data.train_mask.nonzero(as_tuple=False).view(-1)
            if nodes_set.numel() == 0:
                nodes_set = torch.arange(whole_data.y.size(0), device=whole_data.y.device)
            k = min(num_samples, int(nodes_set.numel()))
            perm = torch.randperm(int(nodes_set.numel()), device=nodes_set.device)[:k]
            node_idx_2flip = nodes_set[perm].view(-1, 1)
            flipped_label = whole_data.y[node_idx_2flip].view(-1, 1)
            return node_idx_2flip, flipped_label

        bef_edit_logits = prediction(model, whole_data)
        bef_edit_pred = bef_edit_logits.argmax(dim=-1)
        val_y_true = whole_data.y[whole_data.val_mask]
        val_y_pred = bef_edit_pred[whole_data.val_mask]
        if from_valid_set:
            nodes_set = whole_data.val_mask.nonzero().squeeze()
        else:
            nodes_set = whole_data.train_mask.nonzero().squeeze()

        wrong_pred_set = val_y_pred.ne(val_y_true).nonzero()
        val_node_idx_2flip = wrong_pred_set[torch.randperm(len(wrong_pred_set))[:num_samples]]
        node_idx_2flip = nodes_set[val_node_idx_2flip]
        flipped_label = whole_data.y[node_idx_2flip]

        return node_idx_2flip, flipped_label


def process_edit_results(
    bef_edit_results,
    raw_results,
    editable_param_count=None,
    total_param_count=None,
):
    bef_edit_tra_acc, bef_edit_val_acc, bef_edit_tst_acc = bef_edit_results
    average_success_rate = 0
    success_list = []
    average_dd = []
    highest_dd = []
    lowest_dd = []
    test_dd_std = 0
    selected_result = []

    train_acc, val_acc, test_acc, succeses, steps, mem, tot_time = zip(*raw_results)
    tra_drawdown = bef_edit_tra_acc - train_acc[-1]
    val_drawdown = bef_edit_val_acc - val_acc[-1]
    test_drawdown = test_drawdown = np.round((np.array([bef_edit_tst_acc] * len(test_acc)) - np.array(test_acc)), decimals = 3).tolist()
    average_dd = np.round(np.mean(np.array([bef_edit_tst_acc] * len(test_acc)) - np.array(test_acc)), decimals=3) * 100

    average_success_rate = np.round(np.mean(succeses), decimals = 3).tolist()
    success_list = np.round(np.array(succeses), decimals = 3).tolist()

    test_drawdown = [test_drawdown * 100] if not isinstance(test_drawdown, list) else [round(d * 100, 1) for d in test_drawdown]
    test_dd_std = np.std(test_drawdown)
    highest_dd = max(enumerate(test_drawdown), key=lambda x: x[1])
    lowest_dd = min(enumerate(test_drawdown), key=lambda x: x[1])
    
    selected_result = {}
    
    for i in [1, 10, 25, 50]:
        if len(test_drawdown) >= i:
            selected_result[str(i)] = (test_drawdown[i-1], success_list[i-1])
        else:
            selected_result[str(i)] = (None, None)
    mem_result = {}
    time_result = {}
    mem_result = {
        'max_memory': str(np.round(np.max(mem), decimals=3)) + 'MB'
    }

    for i in [1, 10, 25, 50]:
        if len(tot_time) >= i:
            time_result[str(i)] = str(np.round(tot_time[i-1], decimals=3))
        else:
            time_result[str(i)] = None
    time_result['total_time'] = np.sum(tot_time)

    results = dict(
        bef_edit_tra_acc=bef_edit_tra_acc,
        bef_edit_val_acc=bef_edit_val_acc,
        bef_edit_tst_acc=bef_edit_tst_acc,
        tra_drawdown=tra_drawdown * 100,
        val_drawdown=val_drawdown * 100,
        test_drawdown=test_drawdown,
        average_success_rate=average_success_rate,
        success_list=success_list,
        average_dd=average_dd,
        test_dd_std=test_dd_std,
        highest_dd=highest_dd,
        lowest_dd=lowest_dd,
        selected_result=selected_result,
        mean_complexity=np.mean(steps),
        memory_result=mem_result,
        time_result=time_result,
    )
    if editable_param_count is not None:
        results["editable_param_count"] = int(editable_param_count)
    if total_param_count is not None:
        results["total_param_count"] = int(total_param_count)
    return results


def success_rate(model, idx, label, whole_data):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    model.eval()

    input = grab_input(whole_data)
    is_regression = _is_regression_data(whole_data)
    if model.__class__.__name__ in ['GCN_MLP', 'SAGE_MLP', 'GAT_MLP', 'GIN_MLP', 'Polynormer_MLP']:
        out = model.fast_forward(input['x'][idx], idx)
        if is_regression:
            y_pred = out.squeeze(-1) if out.dim() == 2 and out.size(-1) == 1 else out
        else:
            y_pred = out.argmax(dim=-1)
    else:
        out = model(**input)
        if is_regression:
            pred_all = out.squeeze(-1) if out.dim() == 2 and out.size(-1) == 1 else out
            y_pred = pred_all[idx]
        else:
            y_pred = out.argmax(dim=-1)[idx]
    if is_regression:
        label_f = label.to(y_pred.dtype)
        scale = torch.std(label_f) if label_f.numel() > 1 else torch.abs(label_f).mean()
        tol = torch.clamp(scale * 0.1, min=1e-3)
        success = float((torch.abs(y_pred - label_f) <= tol).float().mean().item())
    else:
        success = int(y_pred.eq(label).sum()) / label.size(0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    return success
