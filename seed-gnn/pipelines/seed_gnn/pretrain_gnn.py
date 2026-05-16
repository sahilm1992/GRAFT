import os
import logging
import time
from copy import deepcopy
from pathlib import Path
logger = logging.getLogger("main")

from typing import Dict
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.typing as pyg_typing
from torch_geometric.data.data import Data
from torch_geometric.loader import NeighborLoader
import json
import models as models
from constants import SEED
from main_utils import set_seeds_all
from models.base import BaseModel
from data import get_data, prepare_dataset
from edit_gnn.utils import grab_input, test, compute_metrics, compute_per_class_accuracy
from pipelines.seed_gnn.utils import get_optimizer, save_model, visualize_validation
from pipelines.seed_gnn.seed_gnn_logging import Logger
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm

def get_split_class_counts(data: Data) -> Dict[str, Dict[int, int]]:
    """Returns class counts for train/val/test splits."""
    if data.y.dtype.is_floating_point:
        return {"train": {}, "val": {}, "test": {}}
    y = data.y.cpu()

    def count(mask):
        if mask is None:
            return {}
        labels = y[mask.cpu()]
        unique, counts = torch.unique(labels, return_counts=True)
        return {int(k): int(v) for k, v in zip(unique, counts)}

    return {
        "train": count(data.train_mask),
        "val": count(data.val_mask),
        "test": count(data.test_mask),
    }


def _checkpoint_prefix(config) -> str:
    model_name = config["pipeline_params"]["model_name"]
    num_layers = config["pipeline_params"].get("architecture", {}).get("num_layers")
    seed = config["management"].get("seed", SEED) # Get seed from management config

    if num_layers is None:
        return f"{model_name}_seed{seed}"
    return f"{model_name}_layers{num_layers}_seed{seed}"


def _to_regression_vector(output: torch.Tensor) -> torch.Tensor:
    if output.dim() == 2 and output.size(-1) == 1:
        return output.squeeze(-1)
    return output


def _compute_regression_metrics(pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor | None = None) -> Dict[str, float]:
    if mask is not None:
        pred = pred[mask]
        y_true = y_true[mask]
    if pred.numel() == 0:
        return {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    pred = pred.to(torch.float32)
    y_true = y_true.to(torch.float32)
    diff = pred - y_true
    mse = torch.mean(diff ** 2)
    mae = torch.mean(torch.abs(diff))
    rmse = torch.sqrt(mse)
    ss_res = torch.sum(diff ** 2)
    y_mean = torch.mean(y_true)
    ss_tot = torch.sum((y_true - y_mean) ** 2)
    if ss_tot.item() <= 1e-12:
        r2 = float("nan")
    else:
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)).item())
    return {
        "mae": float(mae.item()),
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "r2": r2,
    }


def train_loop(
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    train_data: Data,
    loss_op,
    is_regression: bool = False):
    device = next(model.parameters()).device
    train_data = train_data.to(device)
    model.train()
    optimizer.zero_grad()
    input = grab_input(train_data)
    out = model(**input)
    if is_regression:
        pred = _to_regression_vector(out[train_data.train_mask])
        target = train_data.y[train_data.train_mask].to(pred.dtype)
        loss = loss_op(pred, target)
    else:
        loss = loss_op(out[train_data.train_mask], train_data.y[train_data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def train_loop_neighbor(
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    train_loader: NeighborLoader,
    loss_op,
    device: torch.device,
    is_regression: bool = False,
):
    model.train()
    total_loss = 0.0
    total_steps = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        input_dict = grab_input(batch)
        out = model(**input_dict)
        # In NeighborLoader, the first `batch.batch_size` nodes are the seed nodes.
        if is_regression:
            pred = _to_regression_vector(out[:batch.batch_size])
            target = batch.y[:batch.batch_size].to(pred.dtype)
            loss = loss_op(pred, target)
        else:
            loss = loss_op(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        total_steps += 1
    return total_loss / max(total_steps, 1)


@torch.no_grad()
def predict_with_neighbor_loader(
    model: BaseModel,
    data: Data,
    num_layers: int,
    batch_size: int,
    num_neighbors: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    loader = NeighborLoader(
        data,
        input_nodes=None,
        num_neighbors=[num_neighbors] * num_layers,
        batch_size=batch_size,
        shuffle=False,
    )
    logits = None
    for batch in loader:
        batch = batch.to(device)
        out = model(**grab_input(batch))
        seed_bs = batch.batch_size
        seed_nodes = batch.n_id[:seed_bs].cpu()
        seed_out = out[:seed_bs].detach().cpu()
        if logits is None:
            logits = torch.zeros((data.num_nodes, seed_out.size(-1)), dtype=seed_out.dtype)
        logits[seed_nodes] = seed_out
    return logits


@torch.no_grad()
def test_with_neighbor_loader(
    model: BaseModel,
    data: Data,
    num_layers: int,
    batch_size: int,
    num_neighbors: int,
    device: torch.device,
):
    logits = predict_with_neighbor_loader(
        model=model,
        data=data,
        num_layers=num_layers,
        batch_size=batch_size,
        num_neighbors=num_neighbors,
        device=device,
    )
    y_true = data.y.cpu()
    train_mask = data.train_mask.cpu()
    valid_mask = data.val_mask.cpu()
    test_mask = data.test_mask.cpu()

    train_metrics = compute_metrics(logits, y_true, train_mask)
    valid_metrics = compute_metrics(logits, y_true, valid_mask)
    test_metrics = compute_metrics(logits, y_true, test_mask)

    train_pc = compute_per_class_accuracy(logits, y_true, train_mask)
    valid_pc = compute_per_class_accuracy(logits, y_true, valid_mask)
    test_pc = compute_per_class_accuracy(logits, y_true, test_mask)

    return {
        "overall": (train_metrics["acc"], valid_metrics["acc"], test_metrics["acc"]),
        "metrics": {
            "train": train_metrics,
            "val": valid_metrics,
            "test": test_metrics,
        },
        "per_class": {
            "train": train_pc,
            "val": valid_pc,
            "test": test_pc,
        },
    }, logits


@torch.no_grad()
def test_regression(model: BaseModel, data: Data):
    model.eval()
    pred = _to_regression_vector(model(**grab_input(data)))
    y_true = data.y
    train_mask = data.train_mask
    valid_mask = data.val_mask
    test_mask = data.test_mask

    train_metrics = _compute_regression_metrics(pred, y_true, train_mask)
    valid_metrics = _compute_regression_metrics(pred, y_true, valid_mask)
    test_metrics = _compute_regression_metrics(pred, y_true, test_mask)

    return {
        "overall": (train_metrics["rmse"], valid_metrics["rmse"], test_metrics["rmse"]),
        "metrics": {
            "train": train_metrics,
            "val": valid_metrics,
            "test": test_metrics,
        },
        "per_class": {
            "train": {},
            "val": {},
            "test": {},
        },
    }


@torch.no_grad()
def test_regression_with_neighbor_loader(
    model: BaseModel,
    data: Data,
    num_layers: int,
    batch_size: int,
    num_neighbors: int,
    device: torch.device,
):
    preds = _to_regression_vector(
        predict_with_neighbor_loader(
            model=model,
            data=data,
            num_layers=num_layers,
            batch_size=batch_size,
            num_neighbors=num_neighbors,
            device=device,
        )
    )
    y_true = data.y.cpu()
    train_mask = data.train_mask.cpu()
    valid_mask = data.val_mask.cpu()
    test_mask = data.test_mask.cpu()

    train_metrics = _compute_regression_metrics(preds, y_true, train_mask)
    valid_metrics = _compute_regression_metrics(preds, y_true, valid_mask)
    test_metrics = _compute_regression_metrics(preds, y_true, test_mask)

    return {
        "overall": (train_metrics["rmse"], valid_metrics["rmse"], test_metrics["rmse"]),
        "metrics": {
            "train": train_metrics,
            "val": valid_metrics,
            "test": test_metrics,
        },
        "per_class": {
            "train": {},
            "val": {},
            "test": {},
        },
    }, preds


def train(config, data_override: Data | None = None, backbone_runtime: float | None = None):
    set_seeds_all(config["management"]["seed"]) # Use seed from config
    MODEL_FAMILY = getattr(models, config['pipeline_params']['model_name'])
    if data_override is not None:
        data = data_override
        num_features = int(data.x.size(1))
        task_type = getattr(data, "task_type", "regression" if data.y.dtype.is_floating_point else "classification")
        if task_type == "regression":
            num_classes = 1
        else:
            num_classes = int(data.y.max().item() + 1)
    else:
        data, num_features, num_classes = get_data(
            config['management']['dataset_dir'],
            config['eval_params']['dataset'],
            config
        )
        task_type = getattr(data, "task_type", "regression" if data.y.dtype.is_floating_point else "classification")
    save_path = os.path.join(config['management']['pretrain_output_dir'], config['eval_params']['dataset'])

    model = MODEL_FAMILY(
        in_channels=num_features, 
        out_channels=num_classes, 
        load_pretrained_backbone = config['pipeline_params']['load_pretrained_backbone'], 
        saved_ckpt_path = save_path, 
        seed = config["management"]["seed"],
        **config['pipeline_params']['architecture']
    )
    model.cuda()
    logger.info(model)

    arch_config = config['pipeline_params'].get('architecture', {})
    neighbor_batch_size = int(arch_config.get('neighbor_batch_size', 0) or 0)
    use_neighbor_loader = neighbor_batch_size > 0 and hasattr(data, 'edge_index') and data.edge_index is not None
    num_layers = int(arch_config.get('num_layers', 2))
    neighbor_num_neighbors = int(arch_config.get('neighbor_num_neighbors', 10))

    train_loader = None
    eval_data = data.cpu()
    if use_neighbor_loader:
        # Some environments have pyg-lib sampler ABI mismatches (neighbor_sample
        # signature drift). Force torch-sparse backend for compatibility.
        if getattr(pyg_typing, "WITH_PYG_LIB", False) and getattr(pyg_typing, "WITH_TORCH_SPARSE", False):
            pyg_typing.WITH_PYG_LIB = False
            logger.info("Disabled pyg-lib neighbor sampler; using torch-sparse backend for compatibility.")
        train_loader = NeighborLoader(
            data,
            input_nodes=data.train_mask,
            num_neighbors=[neighbor_num_neighbors] * num_layers,
            batch_size=neighbor_batch_size,
            shuffle=True,
        )
        logger.info(
            f"Using NeighborLoader pretraining: batch_size={neighbor_batch_size}, "
            f"num_neighbors={neighbor_num_neighbors}, num_layers={num_layers}"
        )

    train_data, whole_data = prepare_dataset(config['pipeline_params'], data, remove_edge_index=True)
    split_class_counts = get_split_class_counts(whole_data)

    del data
    logger.info(f'training data: {train_data}')
    logger.info(f'whole data: {whole_data}')

    device = next(model.parameters()).device
    train_data = train_data.to(device)
    whole_data = whole_data.to(device)
    if not config['pipeline_params']['load_pretrained_backbone']:#or saved_ckpt_path is None or not os.path.exists(saved_ckpt_path):
        model.reset_parameters()
    optimizer = get_optimizer(config['pipeline_params'], model, pretrain=True)
    is_regression = (task_type == "regression")
    if is_regression:
        best_val = float("inf")
        selection_metric_name = "val_rmse"
        loss_op = F.mse_loss
        train_logger = Logger(primary_metric_name="rmse", greater_is_better=False)
    else:
        best_val = -1.
        selection_metric_name = "val_accuracy"
        loss_op = F.cross_entropy
        train_logger = Logger(primary_metric_name="accuracy", greater_is_better=True)
    best_ckpt_path = None
    best_ckpt_epoch = None
    checkpoint_prefix = _checkpoint_prefix(config)
    save_path = os.path.join(config['management']["pretrain_output_dir"], config['eval_params']['dataset'])

    def _compute_auc_pr_from_logits(logits: torch.Tensor, mask: torch.Tensor | None) -> float:
        if mask is None:
            return float("nan")
        # Align index devices for both logits indexing and label indexing.
        mask_for_idx = mask.to(logits.device) if mask.device != logits.device else mask
        idx = mask_for_idx.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return float("nan")
        y_idx = idx.to(whole_data.y.device) if idx.device != whole_data.y.device else idx
        y_true = whole_data.y[y_idx].cpu().numpy()
        probs = torch.softmax(logits[idx], dim=1).cpu().numpy()
        num_classes = probs.shape[1]
        if num_classes == 1:
            return float("nan")
        try:
            if num_classes == 2:
                return float(average_precision_score(y_true, probs[:, 1]))
            y_one_hot = label_binarize(y_true, classes=np.arange(num_classes))
            return float(average_precision_score(y_one_hot, probs, average="macro"))
        except ValueError:
            return float("nan")

    start_time = time.time()
    for epoch in tqdm(range(1, config['pipeline_params']['epochs'] + 1)):
        if not config['pipeline_params']['load_pretrained_backbone']:
            if use_neighbor_loader:
                train_loss = train_loop_neighbor(
                    model, optimizer, train_loader, loss_op, device, is_regression=is_regression
                )
            else:
                train_loss = train_loop(model, optimizer, train_data, loss_op, is_regression=is_regression)
        if use_neighbor_loader:
            if is_regression:
                result, sampled_logits = test_regression_with_neighbor_loader(
                    model=model,
                    data=eval_data,
                    num_layers=num_layers,
                    batch_size=neighbor_batch_size,
                    num_neighbors=neighbor_num_neighbors,
                    device=device,
                )
            else:
                result, sampled_logits = test_with_neighbor_loader(
                    model=model,
                    data=eval_data,
                    num_layers=num_layers,
                    batch_size=neighbor_batch_size,
                    num_neighbors=neighbor_num_neighbors,
                    device=device,
                )
        else:
            sampled_logits = None
            result = test_regression(model, whole_data) if is_regression else test(model, whole_data)
        train_logger.add_result(result)
        train_acc, valid_acc, test_acc = result["overall"]
        # save the model with the best valid acc
        improved = (valid_acc < best_val) if is_regression else (valid_acc > best_val)
        if improved:
            ckpt_path = save_model(model, save_path, checkpoint_prefix, epoch)
            logger.info(f"New best checkpoint at epoch {epoch}: overall={result['overall']}")
            best_val = valid_acc
            best_ckpt_path = ckpt_path
            best_ckpt_epoch = epoch # Store best_ckpt_epoch

        if epoch%50 == 0:
            val_metrics = result["metrics"]["val"]
            if is_regression:
                logger.info(
                    f"Epoch: {epoch:02d}, "
                    f"Train RMSE: {train_acc:.6f}, "
                    f"Valid RMSE: {valid_acc:.6f}, "
                    f"Valid MAE: {val_metrics['mae']:.6f}, "
                    f"Valid R2: {val_metrics['r2']:.6f}"
                )
            else:
                with torch.no_grad():
                    model.eval()
                    logits = sampled_logits if use_neighbor_loader else model(**grab_input(whole_data))
                    valid_auc_pr = _compute_auc_pr_from_logits(logits, whole_data.val_mask)
                logger.info(f'Epoch: {epoch:02d}, '
                            f'Train Acc: {100 * train_acc:.2f}%, '
                            f'Valid Acc: {100 * valid_acc:.2f}%, '
                            f'Valid F1: {100 * val_metrics["f1"]:.2f}%, '
                            f'Valid B-Recall: {100 * val_metrics["balanced_recall"]:.2f}%, '
                            f'Valid MCC: {val_metrics["mcc"]:.4f}, '
                            f'Valid AUC-PR: {100 * valid_auc_pr:.2f}%')
    pretrain_runtime = time.time() - start_time

    train_logger.print_statistics()
    metrics = train_logger.summarize()

    with torch.no_grad():
        if use_neighbor_loader:
            logits_full = predict_with_neighbor_loader(
                model=model,
                data=eval_data,
                num_layers=num_layers,
                batch_size=neighbor_batch_size,
                num_neighbors=neighbor_num_neighbors,
                device=device,
            )
        else:
            logits_full = model(**grab_input(whole_data))

    if is_regression:
        pred_full = _to_regression_vector(logits_full)
        metrics["final_train_mae"] = _compute_regression_metrics(pred_full, whole_data.y, whole_data.train_mask)["mae"]
        metrics["final_val_mae"] = _compute_regression_metrics(pred_full, whole_data.y, whole_data.val_mask)["mae"]
        metrics["final_test_mae"] = _compute_regression_metrics(pred_full, whole_data.y, whole_data.test_mask)["mae"]
        metrics["final_train_r2"] = _compute_regression_metrics(pred_full, whole_data.y, whole_data.train_mask)["r2"]
        metrics["final_val_r2"] = _compute_regression_metrics(pred_full, whole_data.y, whole_data.val_mask)["r2"]
        metrics["final_test_r2"] = _compute_regression_metrics(pred_full, whole_data.y, whole_data.test_mask)["r2"]
    else:
        metrics["final_train_auc_pr"] = _compute_auc_pr_from_logits(logits_full, whole_data.train_mask)
        metrics["final_val_auc_pr"] = _compute_auc_pr_from_logits(logits_full, whole_data.val_mask)
        metrics["final_test_auc_pr"] = _compute_auc_pr_from_logits(logits_full, whole_data.test_mask)

    

    metrics_json = {
        "experiment": {
            "exp_desc": config["management"]["exp_desc"],
            "task": config["management"]["task"],
            "seed": config["management"]["seed"] # Use seed from config
        },
        "data": {
            "dataset": config["eval_params"]["dataset"],
            "task_type": task_type,
            "num_nodes": int(whole_data.num_nodes),
            "num_features": num_features,
            "num_classes": num_classes,
            "class_distribution": split_class_counts,
            "corr_diagnostics": getattr(whole_data, "corr_diagnostics", {})
        },
        "model": {
            "name": config["pipeline_params"]["model_name"],
            "architecture": config["pipeline_params"]["architecture"],
            "load_pretrained_backbone": config["pipeline_params"]["load_pretrained_backbone"]
        },
        "training": {
            "epochs": config["pipeline_params"]["epochs"],
            "pretrain_runtime": pretrain_runtime,
            "backbone_pretrain_runtime": backbone_runtime,
            "metrics": metrics
        },
        "artifacts": {
            "checkpoint_dir": save_path,
            "best_checkpoint": {
                "path": best_ckpt_path,
                "epoch": best_ckpt_epoch,
                "selection_metric": selection_metric_name
            },
            "output_dir": config["management"]["output_folder_dir"]
        }

    }

    out_file = os.path.join(
        config["management"]["output_folder_dir"],
        f"metrics_pretrain_{_checkpoint_prefix(config)}.json" # Add suffix here
    )

    with open(out_file, "w") as f:
        json.dump(metrics_json, f, indent=2)
    return model, whole_data


def pretrain_gnn(config, data_override: Data | None = None, force_mlp_one_epoch: bool = True):
    save_path = os.path.join(config['management']['pretrain_output_dir'], config['eval_params']['dataset'])
    base_name = config["pipeline_params"]["model_name"].replace("_MLP", "")
    # Pass the full config to _checkpoint_prefix so it can get the seed and num_layers
    base_prefix = _checkpoint_prefix({**config, "pipeline_params": {**config["pipeline_params"], "model_name": base_name}})
    checkpoints = [str(x) for x in Path(save_path).glob(f"{base_prefix}_*.pt")]
    
    # For GNN + MLP, we need to pretrain the GNN backbone first, then the GNN + MLP
    backbone_runtime = None
    if '_MLP' in config['pipeline_params']['model_name'] and len(checkpoints) < 1:
        backbone_config = deepcopy(config)
        backbone_config['pipeline_params']['model_name'] = backbone_config['pipeline_params']['model_name'].replace('_MLP', '')
        backbone_config['pipeline_params']['load_pretrained_backbone'] = False
        
        start_bb = time.perf_counter()
        train(backbone_config, data_override=data_override)
        backbone_runtime = time.perf_counter() - start_bb
    
    # Control for 1 epoch training of the full model (e.g. MLP part)
    if config['pipeline_params'].get('force_mlp_one_epoch', force_mlp_one_epoch):
        config = deepcopy(config)
        config['pipeline_params']['epochs'] = 1
        logger.info("[Control] Forcing 1 epoch training for the full model.")

    model, whole_data = train(config, data_override=data_override, backbone_runtime=backbone_runtime)

    # If we just trained the backbone, we might want to ensure the runtime is captured.
    # The train() call above for the MLP model will save its own metrics.
    # If we want to inject the backbone_runtime into the final metrics, 
    # we should ideally pass it or handle it.

    print(model, whole_data)
    neighbor_batch_size = int(
        config.get('pipeline_params', {}).get('architecture', {}).get('neighbor_batch_size', 0) or 0
    )
    is_regression_data = bool(getattr(whole_data.y, "dtype", torch.long).is_floating_point)
    if neighbor_batch_size > 0:
        logger.info(
            "Skipping full-graph visualization because neighbor-batched pretraining is enabled "
            "(avoids full-graph GAT OOM during visualization forward pass)."
        )
    elif is_regression_data:
        logger.info("Skipping classification visualization for regression dataset.")
    else:
        visualize_validation(config,model, whole_data, [], " on Unedited Model", "")
