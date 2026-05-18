"""
LeastSquares Editor implementation.

Performs a one-shot ridge regression update on the final linear classifier
to enforce desired labels on selected nodes.
"""

import logging
import time
import json
from copy import deepcopy
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
import os
from pathlib import Path

import paths as graft_paths
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import networkx as nx
import torch_geometric
import torch_geometric.typing as pyg_typing
from torch_geometric.utils import degree as pyg_degree, add_self_loops, remove_self_loops
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from sklearn.decomposition import PCA
from collections import OrderedDict

from editing_pipelines.utils.gat_neighbor_eval import should_use_gat_neighbor_loader
from editing_pipelines.editors.base import BaseEditor
from edit_gnn.utils import prediction, test as seed_test, compute_metrics, compute_per_class_accuracy
from editing_pipelines.utils.selection import select_edit_targets_by_strategy
from editing_pipelines.utils.train_eval import test, success_rate
from editing_pipelines.utils.results import (
    save_misclassifications_txt,
    save_misclassification_summary_txt,
    compute_auc_per_feature_bucket,
    mean_median_prob_by_feature,
    perturb_feature_and_measure_probs,
)
from editing_pipelines.utils.visualization import (
    plot_auc_by_feature_with_counts,
    plot_mean_prob_by_feature,
    plot_rep_and_aug_distributions,
    plot_feature_sensitivity,
    plot_sensitivity_reduction,
    plot_transition_subset_distributions,
    plot_degree_vs_sensitivity 
)
from editing_pipelines.utils.lse_math_utils import (
    stack_rows,
    get_weight,
    set_weight,
    capture_layer_inputs,
)
from editing_pipelines.utils.lse_eval_utils import (
    compute_aug_confidences,
    compute_orig_confidences,
    evaluate_edit_effects,
)
from editing_pipelines.utils.metrics import compute_full_auc_pr_by_split


from editing_pipelines.utils.model_io import detect_backbone_module, log_forward_mode   


from edit_gnn.utils import grab_input  # noqa: E402
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from constants import SEED
from main_utils import set_seeds_all


logger = logging.getLogger("main")


class LeastSquaresEditor(BaseEditor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lambda_reg: float = config['pipeline_params'].get("leastsquares", {}).get("lambda_reg", 1e-6)
        self.bias_attr_idx: List[int] = config['pipeline_params'].get("leastsquares", {}).get("bias_attr_idx", [])
        logger.info(f"Initialized LeastSquaresEditor (λ={self.lambda_reg})")
        method_tag = self.config.get("management", {}).get("exp_desc", "leastsquares")
        seed_tag = self.config["pipeline_params"].get("leastsquares", {}).get("seed", SEED)
        suffix_parts = [method_tag, f"seed{seed_tag}"]
        ls_cfg = self.config["pipeline_params"].get("leastsquares", {})
        lambda_val = ls_cfg.get("lambda_reg")
        if lambda_val is not None:
            suffix_parts.append(f"lam{lambda_val}")
        top_frac = ls_cfg.get("top_fraction")
        if top_frac is not None:
            suffix_parts.append(f"top{top_frac}")
        if ls_cfg.get("only_correct", False):
            suffix_parts.append("onlycorrect")
        suffix = "_".join(str(part) for part in suffix_parts if part is not None)

        self.save_dir = str(
            Path(graft_paths.editing_pipelines_root_default())
            / "leastsquares"
            / "visualization_plots"
            / f"{self.config['eval_params']['dataset']}_"
              f"{self.config['pipeline_params']['model_name']}_{suffix}"
        )

        #         # self.load_model_and_data() # REMOVED - will be called in run_editing_experiment
        # backbone, bb_name = detect_backbone_module(self.model)
        # self.bb_name = bb_name
        # log_forward_mode(self.model, self.bb_name, logger)
        self.use_mlp_linears = bool(self.config['pipeline_params'].get('leastsquares', {}).get('use_mlp_linears', False))
        self.use_mean_steering_signal = bool(self.config['pipeline_params'].get('leastsquares', {}).get('mean_steering_signal', False))
        self.use_weighted_mean_signal = bool(self.config['pipeline_params'].get('leastsquares', {}).get('weighted_mean_signal', False))
        self.use_subspace = bool(ls_cfg.get('use_subspace', False))
        self.subspace_variance = float(ls_cfg.get('subspace_variance', 0.95))
        self.gamma_retain = float(ls_cfg.get('gamma_retain', 0.0))
        self.pr_alpha = float(ls_cfg.get('pr_alpha', 0.85))
        self.rank_mix_tau = float(ls_cfg.get('rank_mix_tau', 0.5))
        # device = self.get_device()
        # model = self.model.to(device)
        # self.bef_edit_results = seed_test(self.model, self.whole_data)
        # print(self.model.mlp_freezed)
        # print(f"Before edit results after loading model 0: {self.bef_edit_results['overall']}")

        # self.model.mlp_freezed = False
        # print(f"Model: {self.model.__class__.__name__}, MLP freezed: {self.model.mlp_freezed}")
        # self.bef_edit_results = seed_test(self.model, self.whole_data)
        # print(f"Before edit results after loading model 1: {self.bef_edit_results['overall']}")

        # for p in backbone.parameters():
        #     p.requires_grad = True
        # self.bef_edit_results = seed_test(self.model, self.whole_data)
        # print(f"Before edit results after loading model 2: {self.bef_edit_results['overall']}")
        # for p in self.model.MLP.parameters():
        #     p.requires_grad = True
        # self.bef_edit_results = seed_test(self.model, self.whole_data)
        # print(f"Before edit results after loading model 3: {self.bef_edit_results['overall']}")
        # log_forward_mode(self.model, self.bb_name, logger)

        # self.mu = 3 
    
        # Name of sensitive feature (dataset-specific mapping already fills this)
        self.sensitive_feature = self.config.get("pipeline_params", {}).get("sensitive_feature", "AGE")

        # Optional fixed values used for perturbation of the sensitive feature
        self.fixed_sensitive_values = self.config.get("pipeline_params", {}).get("fixed_sensitive_values", None)

        
        # print(f"Mu: {self.mu}")

    @staticmethod
    def _safe_corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
        """Pearson correlation with NaN-safe fallbacks for degenerate vectors."""
        if x is None or y is None:
            return float("nan")
        if x.numel() == 0 or y.numel() == 0 or x.numel() != y.numel():
            return float("nan")
        x = x.to(torch.float32)
        y = y.to(torch.float32)
        x_std = torch.std(x)
        y_std = torch.std(y)
        if torch.isnan(x_std) or torch.isnan(y_std) or x_std.item() == 0.0 or y_std.item() == 0.0:
            return float("nan")
        corr = torch.corrcoef(torch.stack([x, y]))[0, 1]
        return float(corr.item())

    @staticmethod
    def _rank_normalize(t: torch.Tensor) -> torch.Tensor:
        """Convert raw scores to percentile ranks in [0, 1]."""
        ranks = t.argsort().argsort().float()
        return ranks / ranks.max().clamp_min(1.0)

    def _is_regression_task(self, data=None) -> bool:
        data_obj = data if data is not None else getattr(self, "whole_data", None)
        if data_obj is None:
            return False
        task_type = str(getattr(data_obj, "task_type", "")).lower()
        if task_type:
            return task_type == "regression"
        y = getattr(data_obj, "y", None)
        return bool(y is not None and y.dtype.is_floating_point)

    def _print_polynormer_layers(self):
        model_name = self.model.__class__.__name__
        if "Polynormer" not in model_name:
            return
        print(f"[Polynormer Layer Dump] model={model_name}")
        for idx, (name, module) in enumerate(self.model.named_modules()):
            print(f"[Polynormer Layer Dump] {idx:03d} | {name or '<root>'} | {module.__class__.__name__}")

    def _to_regression_vector(self, output: torch.Tensor) -> torch.Tensor:
        if output.dim() == 2 and output.size(-1) == 1:
            return output.squeeze(-1)
        return output

    def _compute_regression_metrics(self, pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
        pred = pred[mask].to(torch.float32)
        y_true = y_true[mask].to(torch.float32)
        if pred.numel() == 0:
            return {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan"), "r2": float("nan")}
        diff = pred - y_true
        mse = torch.mean(diff ** 2)
        mae = torch.mean(torch.abs(diff))
        rmse = torch.sqrt(mse)
        ss_res = torch.sum(diff ** 2)
        y_mean = torch.mean(y_true)
        ss_tot = torch.sum((y_true - y_mean) ** 2)
        r2 = float("nan") if ss_tot.item() <= 1e-12 else float(1.0 - (ss_res / (ss_tot + 1e-12)).item())
        return {
            "mae": float(mae.item()),
            "mse": float(mse.item()),
            "rmse": float(rmse.item()),
            "r2": r2,
        }

    def _is_oom_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return isinstance(exc, torch.cuda.OutOfMemoryError) or ("out of memory" in msg)

    def _estimate_num_edges(self, data) -> int:
        try:
            edge_index = getattr(data, "edge_index", None)
            if edge_index is not None and hasattr(edge_index, "size"):
                return int(edge_index.size(1))
        except Exception:
            pass
        try:
            adj_t = getattr(data, "adj_t", None)
            if adj_t is not None:
                if hasattr(adj_t, "nnz"):
                    return int(adj_t.nnz())
                if torch.is_tensor(adj_t) and adj_t.dim() == 2:
                    return int(adj_t.size(1))
        except Exception:
            pass
        return -1

    def _empty_classification_eval(self):
        return {
            "overall": (float("nan"), float("nan"), float("nan")),
            "per_class": {"train": {}, "val": {}, "test": {}},
        }

    def _neighbor_eval_params(self) -> Tuple[int, int, int]:
        arch_cfg = self.config.get("pipeline_params", {}).get("architecture", {})
        num_layers = int(arch_cfg.get("num_layers", 2))
        # Match the pretraining defaults the user used when values are not explicitly set.
        batch_size = int(arch_cfg.get("neighbor_batch_size", 1024) or 1024)
        num_neighbors = int(arch_cfg.get("neighbor_num_neighbors", 10) or 10)
        return num_layers, batch_size, num_neighbors

    def _should_use_neighbor_eval(self, data) -> bool:
        model_name = str(self.config.get("pipeline_params", {}).get("model_name", ""))
        if not model_name.startswith("GAT"):
            return False
        return should_use_gat_neighbor_loader(self.model, data)

    @torch.no_grad()
    def _predict_with_neighbor_loader(
        self,
        model,
        data,
        num_layers: int,
        batch_size: int,
        num_neighbors: int,
    ) -> torch.Tensor:
        model.eval()
        device = next(model.parameters()).device
        eval_data = data.cpu()

        # Match pretraining workaround: force torch-sparse neighbor sampler when
        # pyg-lib ABI/signature drifts cause runtime failures.
        if getattr(pyg_typing, "WITH_PYG_LIB", False) and getattr(pyg_typing, "WITH_TORCH_SPARSE", False):
            pyg_typing.WITH_PYG_LIB = False
            logger.info("Disabled pyg-lib neighbor sampler; using torch-sparse backend for compatibility.")

        loader = NeighborLoader(
            eval_data,
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
                logits = torch.zeros((eval_data.num_nodes, seed_out.size(-1)), dtype=seed_out.dtype)
            logits[seed_nodes] = seed_out
        return logits

    @torch.no_grad()
    def _classification_eval_with_neighbor_loader(self, model, data):
        num_layers, batch_size, num_neighbors = self._neighbor_eval_params()

        logits = self._predict_with_neighbor_loader(
            model=model,
            data=data,
            num_layers=num_layers,
            batch_size=batch_size,
            num_neighbors=num_neighbors,
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
        }

    @torch.no_grad()
    def _evaluate_model(self, model, data):
        if not self._is_regression_task(data):
            dataset_name = str(self.config.get("eval_params", {}).get("dataset", ""))
            model_name = str(self.config.get("pipeline_params", {}).get("model_name", ""))
            num_edges = self._estimate_num_edges(data)
            should_skip = self._should_use_neighbor_eval(data)
            if should_skip:
                logger.warning(
                    "Using NeighborLoader classification eval for %s/%s (edges=%s) "
                    "to avoid full-graph GAT OOM.",
                    dataset_name,
                    model_name,
                    num_edges,
                )
                try:
                    return self._classification_eval_with_neighbor_loader(model, data)
                except Exception as exc:
                    msg = str(exc)
                    if self._is_oom_error(exc):
                        logger.warning("NeighborLoader classification eval also OOMed: %s", exc)
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        return self._empty_classification_eval()
                    if "pyg::neighbor_sample()" in msg and "edge_weight" in msg:
                        logger.warning(
                            "NeighborLoader sampler backend mismatch during eval (%s). "
                            "Falling back to empty base eval to continue editing.",
                            exc,
                        )
                        return self._empty_classification_eval()
                    raise
            try:
                return seed_test(model, data)
            except Exception as exc:
                if self._is_oom_error(exc):
                    logger.warning("Skipping classification seed_test due to OOM: %s", exc)
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    return self._empty_classification_eval()
                raise
        model.eval()
        pred = self._to_regression_vector(model(**grab_input(data)))
        y_true = data.y
        train_metrics = self._compute_regression_metrics(pred, y_true, data.train_mask)
        val_metrics = self._compute_regression_metrics(pred, y_true, data.val_mask)
        test_metrics = self._compute_regression_metrics(pred, y_true, data.test_mask)
        return {
            "overall": (train_metrics["rmse"], val_metrics["rmse"], test_metrics["rmse"]),
            "metrics": {"train": train_metrics, "val": val_metrics, "test": test_metrics},
            "per_class": {"train": {}, "val": {}, "test": {}},
        }



    @torch.no_grad()
    def run_degree_analysis(self):

        device = self.get_device()
        data = self.whole_data.to(device)
        model = self.model.eval()

        # Use the SAME sensitivity estimator used everywhere else
        val_df, test_df = perturb_feature_and_measure_probs(
            model,
            data,
            feature_name=self.sensitive_feature,
            K=10,
        )


        # Extract per-node quantities
        logits = model(**grab_input(data))
        probs = torch.softmax(logits, dim=-1)
        conf_all, _ = probs.max(dim=-1)   # shape: [N]

        # --- 3. Do analysis per split
        for split_name, df in [("val", val_df), ("test", test_df)]:
            node_ids = df["Node"].values                # indices into graph
            var_prob = df["VarProb"].values             # sensitivity
            deg = data.degree[node_ids].cpu().numpy()   # degree
            conf = conf_all[node_ids].cpu().numpy()     # confidence

            split_dir = os.path.join(self.save_dir, "degree_analysis", split_name)
            os.makedirs(split_dir, exist_ok=True)

            # plot_degree_vs_sensitivity(
            #     deg=deg,
            #     delta_probs=var_prob,
            #     conf=conf,
            #     save_dir=split_dir,
            # )

        # logger.info(f"[Degree Analysis] Saved to {os.path.join(self.save_dir, 'degree_analysis')}")

    @torch.no_grad()
    def _get_model_aware_subgraph(self, data, node_idx):
        """
        Returns:
            local_edge_index: [2, E]
            edge_weight: [E]
            num_nodes_subset: int
        """

        device = node_idx.device
        num_nodes_total = data.num_nodes
        num_nodes_subset = node_idx.numel()

        edge_index = data.edge_index

        # ===== Subgraph mapping =====
        global_to_local = torch.full((num_nodes_total,), -1, device=device, dtype=torch.long)
        global_to_local[node_idx] = torch.arange(num_nodes_subset, device=device)

        mask = (global_to_local[edge_index[0]] >= 0) & (global_to_local[edge_index[1]] >= 0)
        sub_edge_index = edge_index[:, mask]
        local_edge_index = global_to_local[sub_edge_index]

        # ===== Model-aware weighting =====
        backbone, bb_name = detect_backbone_module(self.model)

        if bb_name == "GCN":
            edge_index_sl, _ = add_self_loops(local_edge_index, num_nodes=num_nodes_subset)
            row, col = edge_index_sl
            deg = pyg_degree(col, num_nodes=num_nodes_subset)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            local_edge_index = edge_index_sl

        elif bb_name == "GAT":
            final_conv = backbone.convs[-1]

            try:
                U_final = capture_layer_inputs(
                    self.model,
                    data_obj=data,
                    layer=final_conv,
                    device=device
                )

                _, (_, alpha_gat) = final_conv(
                    U_final, data.edge_index, return_attention_weights=True
                )

                if alpha_gat.dim() > 1:
                    alpha_gat = alpha_gat.mean(dim=1)

                edge_weight = alpha_gat[mask]

            except Exception as e:
                logger.warning(f"GAT attention failed: {e}. Using uniform.")
                edge_weight = torch.ones(local_edge_index.size(1), device=device)

            row, col = local_edge_index
            deg = pyg_degree(row, num_nodes=num_nodes_subset)
            deg_inv = deg.pow(-1)
            deg_inv[deg_inv == float('inf')] = 0
            edge_weight = edge_weight * deg_inv[row]

        elif bb_name == "SAGE":
            row, col = local_edge_index
            deg = pyg_degree(row, num_nodes=num_nodes_subset)
            deg_inv = deg.pow(-1)
            deg_inv[deg_inv == float('inf')] = 0
            edge_weight = deg_inv[row]

        elif bb_name == "GIN":
            row, col = local_edge_index
            deg = pyg_degree(row, num_nodes=num_nodes_subset)
            deg_inv = deg.pow(-1)
            deg_inv[deg_inv == float('inf')] = 0
            edge_weight = deg_inv[row]

        else:
            row, col = local_edge_index
            deg = pyg_degree(row, num_nodes=num_nodes_subset)
            deg_inv = deg.pow(-1)
            deg_inv[deg_inv == float('inf')] = 0
            edge_weight = deg_inv[row]

        return local_edge_index, edge_weight, num_nodes_subset
    
    def _build_transition_matrix(self, local_edge_index, edge_weight, num_nodes, device):
        return torch.sparse_coo_tensor(
            local_edge_index,
            edge_weight,
            size=(num_nodes, num_nodes),
            device=device
        ).t()
    
    @torch.no_grad()
    def compute_model_aware_divrank(
        self,
        data,
        node_idx: torch.Tensor,
        alpha: float = 0.85,
        max_iter: int = 50,
        tol: float = 1e-6
    ) -> torch.Tensor:
        """
        Correct Graph-Aware DivRank (edge-reinforced random walk).

        Key idea:
        Transition probabilities are dynamically reinforced based on visit counts:
            P_ij(t) ∝ base_weight_ij * visit_count_j

        This encourages diversity by reducing repeated visits to the same nodes.
        """

        # =========================
        # MOVE TO CPU (stable)
        # =========================
        orig_device = node_idx.device
        node_idx = node_idx.cpu()
        data = data.cpu()
        device = torch.device("cpu")

        # =========================
        # SHARED GRAPH + WEIGHTS
        # =========================
        local_edge_index, edge_weight, num_nodes = self._get_model_aware_subgraph(data, node_idx)

        if num_nodes == 0:
            return torch.empty(0, device=orig_device)

        row, col = local_edge_index

        # =========================
        # INITIALIZATION
        # =========================
        rank = torch.full((num_nodes, 1), 1.0 / num_nodes, device=device)
        visit_count = torch.ones((num_nodes, 1), device=device)

        teleport = torch.full((num_nodes, 1), 1.0 / num_nodes, device=device)

        # =========================
        # ITERATIVE DIVRANK
        # =========================
        for _ in range(max_iter):
            prev_rank = rank

            # ---------------------------------
            # 🔥 EDGE-LEVEL REINFORCEMENT
            # ---------------------------------
            # Scale edges based on destination node visit count
            col_scale = visit_count.view(-1)  # [num_nodes]

            dynamic_weight = edge_weight * col_scale[col]

            # ---------------------------------
            # NORMALIZE (row-stochastic)
            # ---------------------------------
            deg = pyg_degree(row, num_nodes=num_nodes)
            deg_inv = deg.pow(-1)
            deg_inv[deg_inv == float('inf')] = 0

            dynamic_weight = dynamic_weight * deg_inv[row]

            # ---------------------------------
            # BUILD DYNAMIC TRANSITION MATRIX
            # ---------------------------------
            P_dyn = torch.sparse_coo_tensor(
                local_edge_index,
                dynamic_weight,
                size=(num_nodes, num_nodes),
                device=device
            ).t()

            # ---------------------------------
            # POWER UPDATE
            # ---------------------------------
            rank = alpha * torch.sparse.mm(P_dyn, rank) + (1 - alpha) * teleport

            # Normalize
            rank = rank / rank.sum().clamp_min(1e-12)

            # ---------------------------------
            # UPDATE VISIT COUNTS
            # ---------------------------------
            visit_count += rank

            # ---------------------------------
            # CONVERGENCE CHECK
            # ---------------------------------
            if torch.norm(rank - prev_rank, p=1) < tol:
                break

        return rank.view(-1).to(orig_device)

    @torch.no_grad()
    def compute_model_aware_pagerank(
        self,
        data,
        node_idx: torch.Tensor,
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-9
    ) -> torch.Tensor:
        """
        Modular, model-aware PageRank.

        Uses:
        - Shared subgraph extraction
        - Shared model-aware edge weighting
        - Shared transition matrix builder

        NOTE:
        - Transductive behavior is controlled outside this function
        by passing train+val indices (tv_idx).
        """

        orig_device = node_idx.device

        node_idx = node_idx.cpu()
        data = data.cpu()
        device = torch.device("cpu")

        # =========================
        # SHARED GRAPH + WEIGHTS
        # =========================
        local_edge_index, edge_weight, num_nodes = self._get_model_aware_subgraph(data, node_idx)

        if num_nodes == 0:
            return torch.empty(0, device=device)

        # =========================
        # TRANSITION MATRIX
        # =========================
        P = self._build_transition_matrix(
            local_edge_index,
            edge_weight,
            num_nodes,
            device
        )

        # =========================
        # INITIALIZATION
        # =========================
        rank = torch.full((num_nodes, 1), 1.0 / num_nodes, device=device)
        teleport = torch.full((num_nodes, 1), 1.0 / num_nodes, device=device)

        # =========================
        # POWER ITERATION
        # =========================
        for _ in range(max_iter):
            prev_rank = rank

            rank = alpha * torch.sparse.mm(P, rank) + (1 - alpha) * teleport

            # Convergence check
            if torch.norm(rank - prev_rank, p=1) < tol:
                break

        return rank.view(-1).to(orig_device)
    
    
    @torch.no_grad()
    def _compute_val_pagerank(self, data, val_idx: torch.Tensor, damping: float = 0.85, max_iter: int = 500, tol: float = 1e-8) -> torch.Tensor:
        """
        Compute PageRank on the validation-node induced subgraph.

        Returns:
            pr_val: Tensor of shape [num_val_nodes], aligned with val_idx order.
        """
        device = val_idx.device
        num_val = int(val_idx.numel())
        if num_val == 0:
            return torch.empty(0, device=device, dtype=torch.float32)

        if not hasattr(data, "edge_index") or data.edge_index is None:
            return torch.full((num_val,), 1.0 / max(num_val, 1), device=device, dtype=torch.float32)

        edge_index_cpu = data.edge_index.detach().cpu()
        if edge_index_cpu.numel() == 0:
            return torch.full((num_val,), 1.0 / max(num_val, 1), device=device, dtype=torch.float32)

        val_nodes = val_idx.detach().cpu().tolist()
        val_set = set(val_nodes)
        src_all = edge_index_cpu[0].tolist()
        dst_all = edge_index_cpu[1].tolist()
        val_edges = [(u, v) for (u, v) in zip(src_all, dst_all) if (u in val_set and v in val_set)]

        G = nx.DiGraph()
        G.add_nodes_from(val_nodes)  # keep isolated validation nodes in graph
        G.add_edges_from(val_edges)

        try:
            pr_dict = nx.pagerank(
                G,
                alpha=damping,
                max_iter=max_iter,
                tol=tol,
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning(
                "[PageRank] nx.pagerank failed to converge (damping=%.3f, "
                "max_iter=%d). Falling back to uniform.",
                damping, max_iter,
            )
            pr_dict = {node: 1.0 / max(num_val, 1) for node in val_nodes}
        pr_val = torch.tensor(
            [float(pr_dict.get(node, 0.0)) for node in val_nodes],
            dtype=torch.float32,
            device=device,
        )
        pr_val = pr_val / pr_val.sum().clamp_min(1e-12)
        return pr_val

    @torch.no_grad()
    def _compute_val_divrank(
        self,
        data,
        val_idx: torch.Tensor,
        alpha: float = 0.85,
        max_iter: int = 50,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """
        Non-model-aware DivRank on the validation-node induced subgraph.

        Uses uniform (degree-normalised) edge weights and the same
        edge-reinforced random-walk iteration as compute_model_aware_divrank,
        but without architecture-specific weighting.

        Returns:
            dr_val: Tensor of shape [num_val_nodes], aligned with val_idx order.
        """
        device = val_idx.device
        num_val = int(val_idx.numel())
        if num_val == 0:
            return torch.empty(0, device=device, dtype=torch.float32)

        if not hasattr(data, "edge_index") or data.edge_index is None:
            return torch.full((num_val,), 1.0 / max(num_val, 1), device=device, dtype=torch.float32)

        edge_index_cpu = data.edge_index.detach().cpu()
        if edge_index_cpu.numel() == 0:
            return torch.full((num_val,), 1.0 / max(num_val, 1), device=device, dtype=torch.float32)

        val_nodes = val_idx.detach().cpu()
        global_to_local = torch.full((int(data.num_nodes),), -1, dtype=torch.long)
        global_to_local[val_nodes] = torch.arange(num_val)

        src, dst = edge_index_cpu[0], edge_index_cpu[1]
        mask = (global_to_local[src] >= 0) & (global_to_local[dst] >= 0)
        local_row = global_to_local[src[mask]]
        local_col = global_to_local[dst[mask]]

        deg = pyg_degree(local_row, num_nodes=num_val).float()
        deg_inv = deg.pow(-1)
        deg_inv[deg_inv == float('inf')] = 0
        base_weight = deg_inv[local_row]

        rank = torch.full((num_val, 1), 1.0 / num_val)
        visit_count = torch.ones((num_val, 1))
        teleport = torch.full((num_val, 1), 1.0 / num_val)
        local_edge_index = torch.stack([local_row, local_col], dim=0)

        for _ in range(max_iter):
            prev_rank = rank

            col_scale = visit_count.view(-1)
            dyn_w = base_weight * col_scale[local_col]

            dyn_deg = pyg_degree(local_row, num_nodes=num_val).float()
            dyn_deg_inv = dyn_deg.pow(-1)
            dyn_deg_inv[dyn_deg_inv == float('inf')] = 0
            dyn_w = dyn_w * dyn_deg_inv[local_row]

            P_dyn = torch.sparse_coo_tensor(
                local_edge_index, dyn_w, size=(num_val, num_val)
            ).t()

            rank = alpha * torch.sparse.mm(P_dyn, rank) + (1 - alpha) * teleport
            rank = rank / rank.sum().clamp_min(1e-12)
            visit_count = visit_count + rank

            if torch.norm(rank - prev_rank, p=1) < tol:
                break

        return rank.view(-1).to(device)

    @torch.no_grad()
    def select_representative_examples(
        self,
        num_bins: int = 4,
        top_fraction: float = 0.25,
        sensitivity_based: bool = True,
        per_bin_selection: bool = False,
        only_correct: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select representative examples based on either:
          - confidence (high-confidence samples), or
          - sensitivity (large probability shift after sensitive-feature perturbation).

        Sensitive feature:
          - Name is taken from config: self.sensitive_feature
          - Fixed values for perturbation (if any) are taken from config:
                self.fixed_sensitive_values

        Arguments (ALL explicit, NOT from config):
          - num_bins:         number of quantile bins for the sensitive feature
          - top_fraction:     fraction of samples to keep (per bin or globally)
          - sensitivity_based:
                True  → rank by Δprob after perturbation
                False → rank by confidence on original data
          - per_bin_selection:
                True  → select top_fraction within each feature bin
                False → select top_fraction globally
          - only_correct:
                True  → consider only correctly classified VAL samples
                False → consider all VAL samples

        Returns:
            selected_indices (Tensor): indices of chosen nodes (w.r.t whole graph)
            selected_labels  (Tensor): corresponding true labels
        """

        ls_cfg = self.config.get("pipeline_params", {}).get("leastsquares", {})
        sensitivity_based = ls_cfg.get("sensitivity_based", sensitivity_based)
        only_correct = ls_cfg.get("only_correct", only_correct)
        top_fraction = ls_cfg.get("top_fraction", top_fraction)
        self.top_fraction = top_fraction

        model = self.model.eval()
        device = self.get_device()
        data = self.whole_data.to(device)
        # print("======================Representative Samples selection strategy==========================")
        # print("Sensitivity_based: ",sensitivity_based)
        # print("only_correct: ",only_correct)
        

        # --- 1. Sensitive feature name & fixed values from config ---
        feature_name = getattr(self, "sensitive_feature", None)
        if feature_name is None:
            raise ValueError("self.sensitive_feature is not set. "
                             "Make sure 'sensitive_feature' is in pipeline_params.")
        fixed_vals_cfg = getattr(self, "fixed_sensitive_values", None)

        # --- 2. Basic checks & feature index ---
        if not hasattr(data, "feature_names") or feature_name not in data.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in data.feature_names")
        feat_idx = data.feature_names.index(feature_name)
        
        # Print distinct values of the feature to be perturbed
        distinct_vals = torch.unique(data.x[:, feat_idx]).cpu().numpy()
        # print(f"[LeastSquaresEditor] Distinct values for feature '{feature_name}': {distinct_vals}")

        # --- 3. Model predictions and confidence on full graph ---
        is_regression = self._is_regression_task(data)
        use_neighbor_eval = (not is_regression) and self._should_use_neighbor_eval(data)
        if use_neighbor_eval:
            num_layers, batch_size, num_neighbors = self._neighbor_eval_params()
            logger.warning(
                "Using NeighborLoader forward for representative selection (%s/%s): "
                "num_layers=%s, batch_size=%s, num_neighbors=%s",
                self.config.get("eval_params", {}).get("dataset", ""),
                self.config.get("pipeline_params", {}).get("model_name", ""),
                num_layers,
                batch_size,
                num_neighbors,
            )
            logits = self._predict_with_neighbor_loader(
                model=model,
                data=data,
                num_layers=num_layers,
                batch_size=batch_size,
                num_neighbors=num_neighbors,
            ).to(device)
        else:
            logits = model(**grab_input(data))
        if is_regression:
            pred = self._to_regression_vector(logits)
            y_true = data.y.to(pred.dtype)
            residual = torch.abs(pred - y_true)
            conf = -residual
            probs = None
            preds = None
        else:
            probs = torch.softmax(logits, dim=-1)
            conf, preds = probs.max(dim=-1)
            y_true = data.y

        # --- 4. Validation subset ---
        val_mask = data.val_mask
        val_idx = val_mask.nonzero(as_tuple=False).view(-1)

        preds_val = preds[val_mask] if preds is not None else None
        y_true_val = y_true[val_mask]
        conf_val = conf[val_mask]
        probs_val = probs[val_mask] if probs is not None else None
        residual_val = residual[val_mask] if is_regression else None

        feature_vals_all = data.x[:, feat_idx]
        feature_vals_val = feature_vals_all[val_mask].float()

        # --- 5. Decide which VAL samples participate ---
        if only_correct:
            if is_regression:
                residual_quantile = float(ls_cfg.get("residual_quantile", 0.5))
                thresh = torch.quantile(residual_val, residual_quantile)
                base_mask = residual_val <= thresh
            else:
                base_mask = preds_val == y_true_val    # only correctly classified VAL samples
        else:
            base_mask = torch.ones_like(y_true_val, dtype=torch.bool)  # all VAL samples

        if base_mask.sum() == 0:
            raise ValueError("No validation samples selected with the given 'only_correct' setting.")

        sel_idx_val = val_idx[base_mask]          # indices in whole graph
        conf_sel = conf_val[base_mask]
        y_sel = y_true_val[base_mask]
        probs_sel = probs_val[base_mask] if probs_val is not None else None
        feature_vals_sel = feature_vals_val[base_mask]
        # ============================
        # DEGREE FILTER (NEW)
        # ============================
        ls_cfg = self.config.get("pipeline_params", {}).get("leastsquares", {})
        degree_mode = ls_cfg.get("degree_filter", None)   # None | "high" | "low"
        degree_fraction = ls_cfg.get("degree_fraction", 0.5)

        if degree_mode is not None:
            deg_all = data.degree.to(device)
            deg_sel = deg_all[sel_idx_val]

            sorted_deg, sort_idx = torch.sort(deg_sel, descending=True)

            k = max(1, int(degree_fraction * len(sort_idx)))

            if degree_mode == "high":
                keep_idx = sort_idx[:k]
            elif degree_mode == "low":
                keep_idx = sort_idx[-k:]
            else:
                raise ValueError(f"Unknown degree_mode: {degree_mode}")

            # Apply degree filter
            sel_idx_val = sel_idx_val[keep_idx]
            conf_sel = conf_sel[keep_idx]
            y_sel = y_sel[keep_idx]
            probs_sel = probs_sel[keep_idx]
            feature_vals_sel = feature_vals_sel[keep_idx]

            logger.info(
                f"[Degree Filter] mode={degree_mode}, kept {k}/{len(sort_idx)} nodes "
                f"({degree_fraction*100:.0f}%)"
            )

        

        # ==========================================================
        # CASE 1: Confidence-based selection (NO perturbation)
        # ==========================================================
        if not sensitivity_based:
            if per_bin_selection:
                # Quantile bins based on sensitive feature of selected VAL samples
                quantiles = torch.linspace(0, 1, num_bins + 1, device=device)
                bin_edges = torch.quantile(feature_vals_sel, quantiles)
                bin_ids = torch.bucketize(feature_vals_sel, bin_edges, right=False) - 1
                bin_ids = torch.clamp(bin_ids, 0, num_bins - 1)

                selected_indices = []
                for b in range(num_bins):
                    bin_mask = bin_ids == b
                    if bin_mask.sum() == 0:
                        continue
                    conf_bin = conf_sel[bin_mask]
                    idx_bin = sel_idx_val[bin_mask]
                    sorted_conf, sort_idx = torch.sort(conf_bin, descending=True)
                    num_to_select = max(1, int(top_fraction * len(sort_idx)))
                    top_idx = sort_idx[:num_to_select]
                    selected_indices.append(idx_bin[top_idx])

                if len(selected_indices) == 0:
                    raise ValueError("No representative examples selected (confidence-based).")
                selected_indices = torch.cat(selected_indices)
            else:
                # Global selection by confidence
                sorted_conf, sort_idx = torch.sort(conf_sel, descending=True)
                num_to_select = max(1, int(top_fraction * len(sort_idx)))
                selected_indices = sel_idx_val[sort_idx[:num_to_select]]

            selected_labels = y_true[selected_indices]

            # store for later use
            self.representative_examples = selected_indices
            self.representative_labels = selected_labels
            self.feature_name = feature_name
            self.num_bins = num_bins

            logger.info(
                f"[Representative Selection | Confidence-based] "
                f"Selected {selected_indices.numel()} examples "
                f"({top_fraction*100:.0f}% top confidence "
                f"{'per bin' if per_bin_selection else 'globally'}) "
                f"for feature '{feature_name}', only_correct={only_correct}."
            )
            return selected_indices.to(device), selected_labels.to(device)

        # ==========================================================
        # CASE 2: Sensitivity-based selection (perturb sensitive feature)
        # ==========================================================

        x_orig = data.x.clone()

        # 2a) Build perturbed values for the selected validation nodes
        if fixed_vals_cfg is not None and len(fixed_vals_cfg) > 0:
            anchors = torch.tensor(fixed_vals_cfg, dtype=torch.float32, device=device)
            n_sel = feature_vals_sel.numel()
            # randomly pick anchor values
            rand_idx = torch.randint(low=0, high=anchors.numel(), size=(n_sel,), device=device)
            new_vals = anchors[rand_idx]

            # try to avoid trivial no-change if anchors contain >1 unique value
            if torch.unique(anchors).numel() > 1:
                same_mask = (new_vals == feature_vals_sel)
                if same_mask.any():
                    alt_idx = torch.randint(low=0, high=anchors.numel(), size=(same_mask.sum(),), device=device)
                    new_vals[same_mask] = anchors[alt_idx]
        else:
            # No fixed values in config: sample uniformly from [min_val, max_val] over VAL
            finite_vals = feature_vals_val[torch.isfinite(feature_vals_val)]
            if finite_vals.numel() == 0:
                raise ValueError(f"Sensitive feature '{feature_name}' has no finite values in validation set.")
            min_val = finite_vals.min()
            max_val = finite_vals.max()
            new_vals = min_val + (max_val - min_val) * torch.rand_like(feature_vals_sel)

        # 2b) Create a perturbed copy of x, only for selected validation nodes
        x_pert = x_orig.clone()
        x_pert[sel_idx_val, feat_idx] = new_vals

        original_x = data.x
        data.x = x_pert
        if use_neighbor_eval:
            num_layers, batch_size, num_neighbors = self._neighbor_eval_params()
            logits_pert = self._predict_with_neighbor_loader(
                model=model,
                data=data,
                num_layers=num_layers,
                batch_size=batch_size,
                num_neighbors=num_neighbors,
            ).to(device)
        else:
            logits_pert = model(**grab_input(data))
        data.x = original_x  # restore
        if is_regression:
            pred_pert_sel = self._to_regression_vector(logits_pert)[sel_idx_val]
            pred_orig_sel = pred[sel_idx_val]
        else:
            probs_pert_sel = torch.softmax(logits_pert, dim=-1)[sel_idx_val]

        # --- 6. Compute probability shift (L1) ---
        if is_regression:
            delta_probs = torch.abs(pred_pert_sel - pred_orig_sel)
        else:
            delta_probs = torch.sum(torch.abs(probs_pert_sel - probs_sel), dim=-1)
        strategy_mode = ls_cfg.get("strategy_mode", "confidence")
        ranking_scores = delta_probs

        pr_sel = None
        centrality_metric_name = None
        if strategy_mode in ("sens_pr", "sens_subspace_retention_pr", "sens_subspace_pr"):
            pr_val = self._compute_val_pagerank(data, val_idx, damping=self.pr_alpha)
            node_to_val_pos = torch.full((int(data.num_nodes),), -1, device=device, dtype=torch.long)
            node_to_val_pos[val_idx] = torch.arange(val_idx.numel(), device=device, dtype=torch.long)
            sel_pos = node_to_val_pos[sel_idx_val]
            if (sel_pos < 0).any():
                raise RuntimeError("Failed to map selected validation nodes to PageRank positions.")
            sel_pos_idx = sel_pos.to(pr_val.device)
            pr_sel = pr_val[sel_pos_idx].to(delta_probs.device)
            centrality_metric_name = "pagerank"
        elif strategy_mode in ("sens_pr_graphaware", "sens_subspace_retention_pr_graphaware", "sens_subspace_pr_graphaware"):
            pr_tv = self.compute_model_aware_pagerank(data, val_idx, alpha=self.pr_alpha)
            node_to_tv_pos = torch.full((int(data.num_nodes),), -1, device=device, dtype=torch.long)
            node_to_tv_pos[val_idx] = torch.arange(val_idx.numel(), device=device, dtype=torch.long)
            sel_pos = node_to_tv_pos[sel_idx_val]
            if (sel_pos < 0).any():
                raise RuntimeError("Failed to map selected validation nodes to graph-aware PageRank positions.")
            sel_pos_idx = sel_pos.to(pr_tv.device)
            pr_sel = pr_tv[sel_pos_idx].to(delta_probs.device)
            centrality_metric_name = "graphaware_pagerank"

        elif strategy_mode in ("sens_divrank", "sens_subspace_retention_divrank", "sens_subspace_divrank"):
            dr_val = self._compute_val_divrank(data, val_idx, alpha=self.pr_alpha)
            node_to_val_pos = torch.full((int(data.num_nodes),), -1, device=device, dtype=torch.long)
            node_to_val_pos[val_idx] = torch.arange(val_idx.numel(), device=device, dtype=torch.long)
            sel_pos = node_to_val_pos[sel_idx_val]
            if (sel_pos < 0).any():
                raise RuntimeError("Failed to map selected validation nodes to DivRank positions.")
            sel_pos_idx = sel_pos.to(dr_val.device)
            pr_sel = dr_val[sel_pos_idx].to(delta_probs.device)
            centrality_metric_name = "divrank"

        elif strategy_mode in ("sens_divrank_graphaware", "sens_subspace_retention_divrank_graphaware", "sens_subspace_divrank_graphaware"):
            pr_tv = self.compute_model_aware_divrank(data, val_idx, alpha=self.pr_alpha)

            node_to_tv_pos = torch.full((int(data.num_nodes),), -1, device=device, dtype=torch.long)
            node_to_tv_pos[val_idx] = torch.arange(val_idx.numel(), device=device)

            sel_pos = node_to_tv_pos[sel_idx_val]
            if (sel_pos < 0).any():
                raise RuntimeError("Failed to map validation nodes to DivRank positions.")

            sel_pos_idx = sel_pos.to(pr_tv.device)
            pr_sel = pr_tv[sel_pos_idx].to(delta_probs.device)
            centrality_metric_name = "graphaware_divrank"

        # Persist diagnostics to be emitted in metrics JSON.
        self.selection_correlation_metrics = {
            "strategy_mode": strategy_mode,
            "centrality_metric": centrality_metric_name,
            "num_selected_validation_nodes": int(sel_idx_val.numel()),
            "sensitivity_graph_centrality_pearson": float("nan"),
            "sensitivity_graph_centrality_spearman": float("nan"),
        }

        if pr_sel is not None:
            a = self.rank_mix_tau
            rank_sens = self._rank_normalize(delta_probs)
            rank_cent = self._rank_normalize(pr_sel)
            ranking_scores = (1.0 - a) * rank_sens + a * rank_cent
            pearson_corr = self._safe_corrcoef(delta_probs, pr_sel)
            spearman_corr = self._safe_corrcoef(rank_sens, rank_cent)
            self.selection_correlation_metrics["sensitivity_graph_centrality_pearson"] = pearson_corr
            self.selection_correlation_metrics["sensitivity_graph_centrality_spearman"] = spearman_corr
            logger.info(
                "[Rank Mix] pr_alpha=%.3f, rank_mix_tau=%.3f, "
                "spearman(sens, pr)=%.3f",
                self.pr_alpha, a,
                spearman_corr,
            )
        # --- 7. Selection based on Δprob ---
        if per_bin_selection:
            quantiles = torch.linspace(0, 1, num_bins + 1, device=device)
            bin_edges = torch.quantile(feature_vals_sel, quantiles)
            bin_ids = torch.bucketize(feature_vals_sel, bin_edges, right=False) - 1
            bin_ids = torch.clamp(bin_ids, 0, num_bins - 1)

            selected_indices = []
            for b in range(num_bins):
                bin_mask = bin_ids == b
                if bin_mask.sum() == 0:
                    continue
                mask_idx = bin_mask.to(ranking_scores.device)
                score_bin = ranking_scores[mask_idx]
                idx_bin = sel_idx_val[mask_idx.to(sel_idx_val.device)]
                _, sort_idx = torch.sort(score_bin, descending=True)
                num_to_select = max(1, int(top_fraction * len(sort_idx)))
                top_idx = sort_idx[:num_to_select].to(idx_bin.device)
                selected_indices.append(idx_bin[top_idx])

            if len(selected_indices) == 0:
                raise ValueError("No representative examples selected (sensitivity-based).")
            selected_indices = torch.cat(selected_indices)
        else:
            # Global selection by Δprob
            _, sort_idx = torch.sort(ranking_scores, descending=True)
            num_to_select = max(1, int(top_fraction * len(sort_idx)))
            pick_idx = sort_idx[:num_to_select].to(sel_idx_val.device)
            selected_indices = sel_idx_val[pick_idx]

        selected_labels = y_true[selected_indices.to(y_true.device)]

        # --- 8. Plot confidence vs Δprob for diagnostics ---
        # os.makedirs(self.save_dir, exist_ok=True)
        # fig, ax = plt.subplots(figsize=(6, 4))
        # ax.scatter(conf_sel.cpu(), delta_probs.cpu(), alpha=0.6, s=20)
        # if is_regression:
        #     ax.set_xlabel("Initial Confidence Proxy (-|prediction error|)")
        #     ax.set_ylabel(f"Δ Prediction after {feature_name} Perturbation (L1)")
        #     ax.set_title(f"Sensitivity: Confidence Proxy vs Prediction Shift ({feature_name})")
        # else:
        #     ax.set_xlabel("Initial Confidence (Softmax max)")
        #     ax.set_ylabel(f"Δ Probability after {feature_name} Perturbation (L1)")
        #     ax.set_title(f"Sensitivity: Confidence vs Probability Shift ({feature_name})")
        # plt.tight_layout()
        # fig.savefig(os.path.join(self.save_dir, "perturbation_sensitivity_plot.png"), dpi=150)
        # plt.close(fig)

        # --- 9. Store internally for later use ---
        self.representative_examples = selected_indices
        self.representative_labels = selected_labels
        self.feature_name = feature_name
        self.num_bins = num_bins

        if strategy_mode in ("sens_pr", "sens_subspace_retention_pr", "sens_subspace_pr"):
            metric_name = "Δprob * PageRank"
        elif strategy_mode in ("sens_pr_graphaware", "sens_subspace_retention_pr_graphaware", "sens_subspace_pr_graphaware"):
            metric_name = "Δprob * GraphAwarePR"
        elif strategy_mode in ("sens_divrank", "sens_subspace_retention_divrank", "sens_subspace_divrank"):
            metric_name = "Δprob * DivRank"
        elif strategy_mode in ("sens_divrank_graphaware", "sens_subspace_retention_divrank_graphaware", "sens_subspace_divrank_graphaware"):
            metric_name = "Δprob * GraphAwareDivRank"
        else:
            metric_name = "Δprob"
        
        logger.info(
            f"[Representative Selection | Sensitivity-based] Selected {selected_indices.numel()} examples "
            f"({top_fraction*100:.0f}% top {metric_name} "
            f"{'per bin' if per_bin_selection else 'globally'}) "
            f"for feature '{feature_name}', only_correct={only_correct}."
        )

        return selected_indices.to(device), selected_labels.to(device)

    def curate_steering_examples(
    self,
    representative_examples,
    K: int = 6,
    ) -> List[torch.Tensor]:
        """
        Create K augmented feature matrices by perturbing ONLY the representative nodes
        along the sensitive feature.

        Sensitive feature name and (optional) fixed values are read from config:

            sensitive_feature = config["pipeline_params"].get("sensitive_feature", "AGE")
            fixed_values      = config["pipeline_params"]["leastsquares"].get("fixed_sensitive_values", None)

        Behaviour:
        - If fixed_values is provided and non-empty:
              For each augmented matrix, assign each representative node a value
              sampled from fixed_values (random permutation).
        - Else:
              For each augmented matrix, assign each representative node a random
              value uniformly drawn from [min(feature), max(feature)] across all nodes.

        Only the representative examples (self.representative_examples) are modified.
        """

        device = self.get_device()
        data = self.whole_data.to(device)
        x = data.x.clone()

        # --- 1. Fetch representative indices ---
        if representative_examples is not None:
            selected_idx = representative_examples
        else:
            selected_idx = getattr(self, "representative_examples", None)
        if selected_idx is None:
            raise RuntimeError("curate_steering_examples called before representative_examples were set.")
        if not torch.is_tensor(selected_idx):
            selected_idx = torch.as_tensor(selected_idx, dtype=torch.long, device=device)
        else:
            selected_idx = selected_idx.to(device=device, dtype=torch.long)

        # --- 2. Sensitive feature config ---
        sensitive_feature = self.sensitive_feature
        if not hasattr(data, "feature_names") or sensitive_feature not in data.feature_names:
            raise ValueError(f"Sensitive feature '{sensitive_feature}' not found in data.feature_names")

        feat_idx = int(data.feature_names.index(sensitive_feature))
        feature_vals = x[:, feat_idx]

        # Fixed values from config (if any)
        fixed_values = self.fixed_sensitive_values


        use_fixed_values = fixed_values is not None and len(fixed_values) > 0

        # --- 3. Prepare fixed values tensor if needed ---
        if use_fixed_values:
            # print("U")
            fixed_values = torch.tensor(fixed_values, dtype=torch.float32, device=device)
            if fixed_values.numel() < selected_idx.numel():
                # Repeat if fewer fixed values than selected examples
                repeat_factor = (selected_idx.numel() // fixed_values.numel()) + 1
                fixed_values = fixed_values.repeat(repeat_factor)
            # we will slice to size selected_idx.numel() later
        else:
            # For continuous-valued feature: use global min/max
            val_mask = data.val_mask
            val_feature_vals = feature_vals[val_mask]
            finite_vals = val_feature_vals[torch.isfinite(val_feature_vals)]
            if finite_vals.numel() == 0:
                raise ValueError(
                    f"[curate_steering_examples] Sensitive feature '{sensitive_feature}' "
                    "has no finite values on validation nodes."
                )
            min_v = finite_vals.min()
            max_v = finite_vals.max()
            if float(min_v) == float(max_v):
                logger.warning(
                    f"[curate_steering_examples] Sensitive feature '{sensitive_feature}' "
                    f"has constant value {float(min_v)}; random uniform will be degenerate."
                )

        # =====================================
        # CREATE K AUGMENTED MATRICES
        # =====================================
        debug_changes_total = 0
        aug_feature_matrices: List[torch.Tensor] = []

        for k in range(K):
            x_aug = x.clone()

            if use_fixed_values:
                # Random permutation of the fixed values, then assign first len(selected_idx)
                perm = torch.randperm(fixed_values.numel(), device=device)
                shuffled = fixed_values[perm][:selected_idx.numel()]
                for i, node in enumerate(selected_idx):
                    orig_val = float(x_aug[node, feat_idx])
                    new_val = float(shuffled[i])
                    if new_val != orig_val:
                        debug_changes_total += 1
                    x_aug[node, feat_idx] = shuffled[i]
            else:
                # Random uniform in [min_v, max_v] for each representative node
                rand_vals = torch.empty(selected_idx.numel(), device=device).uniform_(min_v, max_v)
                for i, node in enumerate(selected_idx):
                    orig_val = float(x_aug[node, feat_idx])
                    new_val = float(rand_vals[i])
                    if new_val != orig_val:
                        debug_changes_total += 1
                    x_aug[node, feat_idx] = rand_vals[i]

            aug_feature_matrices.append(x_aug)

        # === DEBUG STATS ===
        changed_nodes = 0
        for idx in selected_idx:
            orig = float(x[idx, feat_idx])
            new_vals = [float(X_aug[idx, feat_idx]) for X_aug in aug_feature_matrices]
            if any(v != orig for v in new_vals):
                changed_nodes += 1

        # print(f"\n[DEBUG Augmentation Validation]")
        # print(f"  Sensitive feature: {sensitive_feature}")
        # print(f"  Mode: {'FIXED_VALUES' if use_fixed_values else 'UNIFORM_MIN_MAX'}")
        # print(f"  Selected nodes: {selected_idx.numel()}")
        # print(f"  Nodes changed at least once: {changed_nodes}/{selected_idx.numel()} "
        #       f"({100*changed_nodes/selected_idx.numel():.1f}%)")
        # print(f"  Total changed assignments (over all K): {debug_changes_total}")

        sample_to_show = min(10, selected_idx.numel())
        # print("\n  Example changes:")
        for idx in selected_idx[:sample_to_show]:
            orig = float(x[idx, feat_idx])
            vals = [float(X_aug[idx, feat_idx]) for X_aug in aug_feature_matrices]
            # print(f"    {int(idx):5d}: {orig:.3f} -> {', '.join(f'{v:.3f}' for v in vals)}")

        self.X_aug_all = aug_feature_matrices
        return aug_feature_matrices


    @torch.no_grad()
    def select_edit_targets(self, K: int = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1. Select representative examples (edit targets)
        2. Create K steering/augmented feature matrices and store them in self.X_aug_all
        3. Select editable layers (populate self.editable_layers)

        Returns:
            node_idx_2flip, flipped_label
        """
        # Allow caller override of K or read from config
        if K is None:
            K = int(self.config.get("leastsquares", {}).get("num_aug", 6))

        ls_cfg = self.config["pipeline_params"]["leastsquares"]
        strategy_mode = ls_cfg.get("strategy_mode", "confidence")

        # ORIGINAL BEHAVIOUR
        node_idx_2flip, flipped_label = self.select_representative_examples(**kwargs)
        X_aug_all = self.curate_steering_examples(node_idx_2flip, K=K)
        self.X_aug_all = X_aug_all
        self.num_aug = K

        # layers unchanged
        self.select_edit_layers(only_linear=kwargs.get("only_linear", False))
        # logger.info(f"[select_edit_targets] Selected {node_idx_2flip.numel()} reps, created K = {len(X_aug_all)} aug matrices, {len(self.editable_layers)} editable layers.")
        # logger.info(f"Shape of X_aug matrices: {X_aug_all[0].shape}")

        return node_idx_2flip, flipped_label

    def compute_fisher_information(self, model, data, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute diagonal Fisher Information (Ω) over provided node indices, with parameter names."""
        model.eval()
        fisher = {}

        from edit_gnn.utils import grab_input
        out = model(**grab_input(data))
        if indices.dim() > 1:
            indices = indices.squeeze(dim=1)
        indices = indices.to(out.device)
        if self._is_regression_task(data):
            pred_sel = self._to_regression_vector(out[indices])
            y_sel = data.y[indices].to(pred_sel.dtype)
            loss = F.mse_loss(pred_sel, y_sel)
        else:
            criterion = nn.CrossEntropyLoss()
            y_sel = data.y[indices].to(out.device)
            loss = criterion(out[indices], y_sel)
        model.zero_grad()
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] = p.grad.detach() ** 2  # diagonal Fisher

        logger.info(f"Computed Fisher for {len(fisher)} parameters.")
        return fisher

    #need to write a generalized function for this later
    def select_edit_layers(self, **kwargs):
        # print("Selecting editable layers...")

        only_linear = kwargs.get('only_linear', None)
        if only_linear is None:
            only_linear = bool(self.config['pipeline_params']['leastsquares'].get('only_linear', False))
        self.only_linear = only_linear

        mlp_modules = set()
        if hasattr(self.model, 'MLP') and self.model.MLP is not None:
            mlp_modules = {m for m in self.model.MLP.modules() if isinstance(m, nn.Linear)}

        editable_layers = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and module in mlp_modules and not self.use_mlp_linears:
                continue
            if only_linear and isinstance(module, nn.Linear):
                editable_layers.append(module)
                continue

            if not only_linear:
                # Only keep modules for which _get_weight succeeds
                try:
                    _ = get_weight(module)
                    editable_layers.append(module)
                except Exception:
                    pass

        if len(editable_layers) == 0:
            raise RuntimeError("No editable layers found in the model!")

        self.editable_layers = editable_layers
        layer_names = [type(l).__name__ for l in editable_layers]
        logger.info(f"[Layer Selection] Editing enabled for {len(editable_layers)} layers: {layer_names} | only_linear={only_linear} | use_mlp_linears={self.use_mlp_linears}")
        return editable_layers
    
    # _evaluate_edit_effects moved to editing_pipelines.utils.lse_eval_utils.evaluate_edit_effects
 

    # MATH UTILS moved to editing_pipelines.utils.lse_math_utils

    @torch.no_grad()
    def _compute_D_U(
        self,
        layer: nn.Module,
        U_orig: torch.Tensor,
        U_aug: torch.Tensor,
        num_augments: int,
        aug_confidences: Optional[torch.Tensor] = None,
        orig_confidences: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given U (orig) and U' (aug) at layer input, compute:
            D = W U' - W U          (desired output change)
            U = U_orig - U_aug      (ΔU, the input perturbation)

        The edit S is found via  min_S ||ΔU S^T - D||² + λ||S||².
        This ensures (W+S)(u_orig - u_aug) ≈ 0, i.e. output invariance
        to the sensitive-feature perturbation.
        """
        W = get_weight(layer)  # [out, in]
        V_orig = U_orig @ W.t()
        V_aug = U_aug @ W.t()
        if self.use_mean_steering_signal:
            if U_orig.size(0) != U_aug.size(0):
                raise ValueError("U_orig and U_aug must have the same number of rows when computing mean steering signal.")
            if U_aug.size(0) % num_augments != 0:
                raise ValueError("num_augments must divide the stacked rows.")
            nodes_per_augment = U_aug.size(0) // num_augments
            V_aug_reshaped = V_aug.view(num_augments, nodes_per_augment, -1)
            # print(f"Shape of V_aug_reshaped: {V_aug_reshaped.shape}")
            V_orig_reshaped = V_orig.view(num_augments, nodes_per_augment, -1)
            # print(f"Shape of V_orig_reshaped: {V_orig_reshaped.shape}")
            V_orig_base = V_orig_reshaped[0]
            # print(f"Shape of V_orig_base: {V_orig_base.shape}")
            if self.use_weighted_mean_signal:
                if aug_confidences is None:
                    logger.warning("[Mean Steering] weighted_mean_signal set but no confidences provided; using unweighted mean.")
                    V_aug_sum = V_aug_reshaped.sum(dim=0)
                    V_mean_per_node = (V_orig_base + V_aug_sum) / (num_augments + 1)
                else:
                    if aug_confidences.shape != (num_augments, nodes_per_augment):
                        raise ValueError(
                            f"aug_confidences must have shape ({num_augments}, {nodes_per_augment}) "
                            f"but got {tuple(aug_confidences.shape)}."
                        )
                    if orig_confidences is None:
                        logger.warning("[Mean Steering] weighted_mean_signal set but no orig confidences; using unit weights for original.")
                        orig_weights = torch.ones(nodes_per_augment, device=V_aug.device, dtype=V_aug.dtype)
                    else:
                        if orig_confidences.shape != (nodes_per_augment,):
                            raise ValueError(
                                f"orig_confidences must have shape ({nodes_per_augment},) "
                                f"but got {tuple(orig_confidences.shape)}."
                            )
                        orig_weights = orig_confidences.to(V_aug.device).to(V_aug.dtype).clamp_min(0.0)
                    weights = aug_confidences.clamp_min(0.0)
                    weight_sum = weights.sum(dim=0)
                    weighted_aug_sum = (V_aug_reshaped * weights.unsqueeze(-1)).sum(dim=0)
                    denom = (orig_weights + weight_sum).clamp_min(1e-12).unsqueeze(-1)
                    V_mean_per_node = (V_orig_base * orig_weights.unsqueeze(-1) + weighted_aug_sum) / denom
            else:
                V_aug_sum = V_aug_reshaped.sum(dim=0)
                # print(f"Shape of V_aug_sum: {V_aug_sum.shape}")
                V_mean_per_node = (V_orig_base + V_aug_sum) / (num_augments + 1)
            # print(f"Shape of V_mean_per_node: {V_mean_per_node.shape}")
            V_mean_tiled = V_mean_per_node.repeat(num_augments, 1)
            D = V_aug - V_mean_tiled            # [m, out]
        else:
            D = V_aug - V_orig                    # [m, out]
        U = U_orig - U_aug                        # [m, in]  (ΔU)
        # print(f"Shape of D: {D.shape}")
        # print(f"Shape of U: {U.shape}")
        # print(f"L2 norm of U_orig - U_aug: {torch.norm(U_orig - U_aug)}")
        # print(f"V_orig: {V_orig}")
        # print(f"V_aug: {V_aug}")
        # print(f"D: {D}")
        # print(f"U: {U}")
        return D, U

    @torch.no_grad()
    def _solve_shift(self, D: torch.Tensor, U: torch.Tensor, lam: float):
        """
        Memory-efficient solve:
        S = ( (Uᵀ U + λ I)^{-1} Uᵀ D )ᵀ
        Shapes:
        U: (M, d_in)
        D: (M, d_out)
        returns S: (d_out, d_in)
        """
        finite_rows = torch.isfinite(U).all(dim=1) & torch.isfinite(D).all(dim=1)
        if not finite_rows.all():
            removed = int((~finite_rows).sum().item())
            logger.warning("[LeastSquares] Removing %d non-finite rows before ridge solve.", removed)
            U = U[finite_rows]
            D = D[finite_rows]
        if U.numel() == 0 or D.numel() == 0:
            logger.warning("[LeastSquares] No finite rows left for solve; returning zero shift.")
            return torch.zeros((D.size(1), U.size(1)), device=U.device, dtype=U.dtype)

        # feature-space ridge regression
        UtU = U.T @ U                # (d_in, d_in)
        UtD = U.T @ D                # (d_in, d_out)
        reg = lam * torch.eye(UtU.size(0), device=UtU.device, dtype=UtU.dtype)

        # solve (UtU + λI) X = UtD
        lhs = UtU + reg
        try:
            X = torch.linalg.solve(lhs, UtD)   # (d_in, d_out)
        except RuntimeError as err:
            err_msg = str(err).lower()
            # Some CUDA environments intermittently fail to initialize cuSOLVER.
            # Fall back to a CPU solve to keep editing robust.
            if "cusolver" in err_msg or "cublas" in err_msg or "cuda" in err_msg:
                logger.warning(
                    "[LeastSquares] CUDA linalg solve failed (%s). Falling back to CPU solve.",
                    err,
                )
                X = torch.linalg.solve(lhs.cpu(), UtD.cpu()).to(lhs.device)
            else:
                raise
        S = X.T                                   # (d_out, d_in)
        return S

    @torch.no_grad()
    def _solve_shift_subspace(
        self,
        D: torch.Tensor,
        U: torch.Tensor,
        U_orig_untiled: torch.Tensor,
        lam: float,
        gamma: float,
        variance_fraction: float = 0.95,
    ) -> torch.Tensor:
        """
        Subspace-constrained solve with self-retention penalty.

        Restricts the edit to the top-k right singular vectors of ΔU (keeping
        `variance_fraction` of the variance), and penalises edits that change
        the output for the original (unperturbed) representative activations.

        Solves:
            (X_invᵀ X_inv + γ X_retᵀ X_ret + λ I)⁻¹ (X_invᵀ D)
        where
            X_inv  = ΔU  @ V_k   (KM × k)
            X_ret  = U_orig @ V_k (M  × k)
        and V_k are the top-k right singular vectors of ΔU.

        Returns S: (d_out, d_in).
        """
        finite_rows = torch.isfinite(U).all(dim=1) & torch.isfinite(D).all(dim=1)
        if not finite_rows.all():
            removed = int((~finite_rows).sum().item())
            logger.warning("[LeastSquares] Removing %d non-finite rows before subspace solve.", removed)
            U = U[finite_rows]
            D = D[finite_rows]
        if U.numel() == 0 or D.numel() == 0:
            logger.warning("[LeastSquares] No finite rows left for subspace solve; returning zero shift.")
            return torch.zeros((D.size(1), U_orig_untiled.size(1)), device=U_orig_untiled.device, dtype=U_orig_untiled.dtype)

        orig_device = U.device
        try:
            _, s, Vt = torch.linalg.svd(U, full_matrices=False)
        except RuntimeError as err:
            err_msg = str(err).lower()
            if "cusolver" in err_msg or "cublas" in err_msg or "cuda" in err_msg:
                logger.warning(
                    "[LeastSquares] CUDA SVD failed (%s). Falling back to CPU.", err,
                )
                _, s, Vt = torch.linalg.svd(U.cpu(), full_matrices=False)
                s = s.to(orig_device)
                Vt = Vt.to(orig_device)
            else:
                raise

        s_sq = s ** 2
        total_var = s_sq.sum()
        cumvar = torch.cumsum(s_sq, dim=0)
        ratio = cumvar / total_var.clamp_min(1e-12)
        above = (ratio >= variance_fraction).nonzero(as_tuple=False)
        k = int(above[0].item()) + 1 if above.numel() > 0 else s.numel()
        k = max(1, min(k, s.numel()))

        V_k = Vt[:k].T  # (d_in, k)

        X_inv = U @ V_k                              # (KM, k)
        X_ret = U_orig_untiled @ V_k                  # (M, k)

        A = X_inv.T @ X_inv                           # (k, k)
        if gamma > 0:
            A = A + gamma * (X_ret.T @ X_ret)
        A = A + lam * torch.eye(k, device=A.device, dtype=A.dtype)

        rhs = X_inv.T @ D                             # (k, d_out)

        try:
            X_sol = torch.linalg.solve(A, rhs)        # (k, d_out)
        except RuntimeError as err:
            err_msg = str(err).lower()
            if "cusolver" in err_msg or "cublas" in err_msg or "cuda" in err_msg:
                logger.warning(
                    "[LeastSquares] CUDA linalg solve failed (%s). Falling back to CPU solve.",
                    err,
                )
                X_sol = torch.linalg.solve(A.cpu(), rhs.cpu()).to(A.device)
            else:
                raise

        S = (V_k @ X_sol).T                           # (d_out, d_in)

        logger.info(
            "[Subspace Solve] SVD rank=%d, kept k=%d (%.1f%% variance), gamma=%.4g",
            s.numel(), k, 100 * cumvar[k - 1] / total_var, gamma,
        )
        return S

    # Fairness helpers moved to editing_pipelines.utils.lse_eval_utils

    def _get_node_degrees(self, data) -> np.ndarray:
        num_nodes = int(data.num_nodes)
        if hasattr(data, "edge_index") and data.edge_index is not None:
            try:
                deg = pyg_degree(data.edge_index[0], num_nodes=num_nodes, dtype=torch.float32)
                return deg.cpu().numpy()
            except Exception:
                pass
        if hasattr(data, "adj_t") and data.adj_t is not None:
            try:
                dense_adj = data.adj_t.to_dense()
                deg = dense_adj.sum(dim=1).to(torch.float32).view(-1)
                return deg.cpu().numpy()
            except Exception:
                pass
        return np.zeros(num_nodes, dtype=np.float32)

    def _build_transition_subsets(
        self,
        mask: torch.Tensor,
        correct_before: torch.Tensor,
        correct_after: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        subsets = {"0->0": [], "0->1": [], "1->0": [], "1->1": []}
        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return {}
        for node in idx.tolist():
            before = int(correct_before[node].item())
            after = int(correct_after[node].item())
            key = f"{before}->{after}"
            subsets.setdefault(key, []).append(node)
        return {
            key: torch.tensor(values, dtype=torch.long, device=mask.device)
            for key, values in subsets.items()
            if len(values) > 0
        }

    # -----------------------------
    # Editing
    # -----------------------------

    def edit_layer(self, layer: nn.Module, S_star: torch.Tensor):
        """
        Edit the parameters of the selected layer.
        """
        W = get_weight(layer)
        set_weight(layer, W + S_star)

    def edit_model(self, **kwargs) -> None:
        model_before = deepcopy(self.model)
        self.model_before = model_before

        node_idx_2flip: torch.Tensor = kwargs['node_idx_2flip']
        flipped_label: torch.Tensor = kwargs['flipped_label']
        lambda_reg: float = kwargs.get('lambda_reg', self.lambda_reg)

        logger.info(f"Starting LeastSquares editing with {len(node_idx_2flip)} targets (λ={lambda_reg})")

        device = self.get_device()
        self.model = self.model.to(device)
        self.whole_data = self.whole_data.to(device)
        self.model.eval()

        # print(f"X_orig: {self.whole_data.x}")

        # print(f"X_aug_all: {self.X_aug_all}")

        self.weight_change_metrics = {}

        for i, layer in enumerate(self.editable_layers):
            # print(f"Editing layer: {layer.__class__.__name__}")
            
            # --- 1) Capture U (target nodes) under original input ---
            U_orig = capture_layer_inputs(
                self.model,
                data_obj=self.whole_data,
                layer=layer,
                node_idx=node_idx_2flip,
                device=self.get_device(),
            )  # shape: (M, d_in)
            if U_orig is None:
                logger.warning(f"[Edit] Skipping layer {layer.__class__.__name__} because original capture failed.")
                continue
            # print(f"Shape of U_orig: {U_orig.shape}")

            # --- 2) For each augmentation, forward pass entire graph ---
            U_aug_all = []
        # prefer previously created aug matrices, else create on-the-fly
            if hasattr(self, "X_aug_all") and len(self.X_aug_all) > 0:
                X_aug_all = self.X_aug_all
            else:
                num_aug = int(self.config.get("leastsquares", {}).get("num_aug", 6))
                X_aug_all = self.curate_steering_examples(K=num_aug)

            num_aug = len(X_aug_all)
            aug_confidences = None
            orig_confidences = None
            weighted_mode_active = self.use_weighted_mean_signal
            if self.use_mean_steering_signal and self.use_weighted_mean_signal:
                if self._should_use_neighbor_eval(self.whole_data):
                    logger.warning(
                        "[Edit] Disabling weighted mean steering for large-graph GAT run "
                        "to avoid full-graph confidence OOM. Falling back to unweighted mean steering."
                    )
                    weighted_mode_active = False
                else:
                    try:
                        aug_confidences = compute_aug_confidences(
                            self.model, self.whole_data, X_aug_all, device=self.get_device()
                        )
                        if aug_confidences is not None:
                            aug_confidences = aug_confidences[:, node_idx_2flip]
                        orig_confidences = compute_orig_confidences(
                            self.model, self.whole_data, device=self.get_device()
                        )
                        if orig_confidences is not None:
                            orig_confidences = orig_confidences[node_idx_2flip]
                    except Exception as exc:
                        if self._is_oom_error(exc):
                            logger.warning(
                                "[Edit] Weighted confidence computation OOMed (%s). "
                                "Falling back to unweighted mean steering for this run.",
                                exc,
                            )
                            weighted_mode_active = False
                            aug_confidences = None
                            orig_confidences = None
                            try:
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass
                        else:
                            raise
            for j in range(num_aug):
                X_aug = X_aug_all[j].to(self.get_device())
                U_aug = capture_layer_inputs(
                    self.model,
                    data_obj=self.whole_data,
                    layer=layer,
                    node_idx=node_idx_2flip,
                    override_x=X_aug,
                    device=self.get_device(),
                )  # shape: (M, d_in)
                if U_aug is None:
                    logger.warning(f"[Edit] Skipping layer {layer.__class__.__name__} because augmentation capture failed.")
                    U_aug_all = []
                    break
                # print(f"Shape of U_aug: {U_aug.shape}")
                U_aug_all.append(U_aug)
            if not U_aug_all:
                continue

            U_aug_stacked = torch.cat(U_aug_all, dim=0)  # (K*N, d_in)
            U_orig_tiled = U_orig.repeat(num_aug, 1)     # (K*N, d_in)

            # print(f"Shape of U_orig_tiled: {U_orig_tiled.shape}")
            # print(f"Shape of U_aug_stacked: {U_aug_stacked.shape}")

            # --- 3) Compute D for ALL nodes ---
            _orig_weighted_flag = self.use_weighted_mean_signal
            self.use_weighted_mean_signal = weighted_mode_active
            try:
                D, U = self._compute_D_U(
                    layer,
                    U_orig_tiled,
                    U_aug_stacked,
                    num_augments=num_aug,
                    aug_confidences=aug_confidences,
                    orig_confidences=orig_confidences,
                )
            finally:
                self.use_weighted_mean_signal = _orig_weighted_flag
            # D: (K*N, d_out), U: (K*N, d_in)
            
            # --- 4) Solve least squares shift ---
            if self.use_subspace:
                S = self._solve_shift_subspace(
                    D, U, U_orig,
                    lam=lambda_reg,
                    gamma=self.gamma_retain,
                    variance_fraction=self.subspace_variance,
                )
            else:
                S = self._solve_shift(D, U, lam=lambda_reg)

            # --- 5) apply shift ---
            before = get_weight(layer).detach().clone()
            self.edit_layer(layer=layer, S_star=S)
            after  = get_weight(layer).detach()

            # Debug objective terms for this layer:
            #   min_S ||U S^T - D|| + lambda * ||S||
            zero_shift = torch.zeros_like(S)
            su_minus_d_before = torch.norm(U @ zero_shift.t() - D).item()
            lambda_s_before = (lambda_reg * torch.norm(zero_shift)).item()
            su_minus_d_after = torch.norm(U @ S.t() - D).item()
            lambda_s_after = (lambda_reg * torch.norm(S)).item()
            
            # Calculate norms
            norm_orig = torch.norm(before.flatten()).item()
            norm_delta = torch.norm((after - before).flatten()).item()
            fractional_norm = norm_delta / (norm_orig + 1e-12)
            
            layer_name = f"layer_{i}_{layer.__class__.__name__}"
            self.weight_change_metrics[layer_name] = {
                "norm_orig": norm_orig,
                "norm_delta": norm_delta,
                "fractional_norm": fractional_norm
            }
            
            logger.info(f"[Edit] Layer {layer.__class__.__name__}: ||W_orig||₂ = {norm_orig:.6e}, ||ΔW||₂ = {norm_delta:.6e}, Fractional = {fractional_norm:.6f}")
            logger.info(
                "[Edit Objective] Layer %s | before: ||SU-D||=%.6e, lambda||S||=%.6e, total=%.6e | "
                "after: ||SU-D||=%.6e, lambda||S||=%.6e, total=%.6e",
                layer.__class__.__name__,
                su_minus_d_before,
                lambda_s_before,
                su_minus_d_before + lambda_s_before,
                su_minus_d_after,
                lambda_s_after,
                su_minus_d_after + lambda_s_after,
            )


        # Optionally save summaries
        if self._is_regression_task(self.whole_data):
            logger.info("Skipping misclassification summary artifacts for regression dataset.")
        else:
            try:
                save_misclassifications_txt(
                    self.config,
                    model_before=self.model_before,
                    model_after=self.model,
                    whole_data=self.whole_data,
                    method_name='leastsquares',
                    model_name=self.config['pipeline_params']['model_name'],
                    file_suffix='',
                )
                save_misclassification_summary_txt(
                    self.config,
                    model_before=self.model_before,
                    model_after=self.model,
                    whole_data=self.whole_data,
                    method_name='leastsquares',
                    model_name=self.config['pipeline_params']['model_name'],
                    file_suffix='',
                    edit_indices=node_idx_2flip,
                )
            except Exception as e:
                logger.warning(f"Failed to save misclassification summaries: {e}")

        logger.info("LeastSquares editing completed")
        return None

    def run_editing_experiment(self, **kwargs):
        """
        Top-level pipeline: select targets + steering examples, then run edit_model.
        """
        data_override = kwargs.get("data_override")
        self.load_model_and_data(data_override=data_override)
        
        backbone, bb_name = detect_backbone_module(self.model)
        self.bb_name = bb_name
        log_forward_mode(self.model, self.bb_name, logger)

        ls_cfg = self.config.get("pipeline_params", {}).get("leastsquares", {})
        seed = ls_cfg.get("seed", SEED)
        set_seeds_all(seed)
        device = self.get_device()
        self.model = self.model.to(device)
        if self.train_data is not None:
            self.train_data = self.train_data.to(device)
        self.whole_data = self.whole_data.to(device)
        self.is_regression = self._is_regression_task(self.whole_data)
        # self._print_polynormer_layers()
        if self.use_mlp_linears:
            logger.info("[MLP] Enabling MLP linear edits; unfreezing and fine-tuning the MLP path.")
            if hasattr(self.model, 'mlp_freezed'):
                self.model.mlp_freezed = False
            if hasattr(self.model, 'freeze_module'):
                self.model.freeze_module(train=False)
            self.fine_tune_if_needed()
        # print(f"Model: {self.model.__class__.__name__}, MLP freezed: {getattr(self.model, 'mlp_freezed', 'N/A')}")
        # print(f"========== Evaluating before edit ==========")
        # self.evaluate_before_edit()
        self.bef_edit_results = self._evaluate_model(self.model, self.whole_data)
        # print(f"Before edit results: {self.bef_edit_results}")
        # bef_edit_ft_results = self.fine_tune_if_needed()
        # self.evaluate_before_edit()
        # self.run_degree_analysis()


        # print(f"========== Evaluating before edit completed ==========")
        edit_start = time.perf_counter()
        # 1) Select targets and create steering examples
        kwargs["only_linear"] = kwargs.get("only_linear", False)
        node_idx_2flip, flipped_label = self.select_edit_targets(**kwargs)

        # 2) Run the edit
        lambda_reg = float(kwargs.get('lambda_reg', self.lambda_reg))

        self.edit_model(node_idx_2flip=node_idx_2flip, flipped_label=flipped_label, lambda_reg=lambda_reg)
        edit_runtime = time.perf_counter() - edit_start

        logger.info("LeastSquares editing completed")
        results_after = self._evaluate_model(self.model, self.whole_data)
        fairness_metrics = {}
        if not self.is_regression:
            if self._should_use_neighbor_eval(self.whole_data):
                logger.warning(
                    "Skipping fairness diagnostics for large-graph GAT run to avoid full-graph OOM."
                )
            else:
                try:
                    fairness_metrics = evaluate_edit_effects(self)
                except Exception as exc:
                    if self._is_oom_error(exc):
                        logger.warning(
                            "Skipping fairness diagnostics due to OOM: %s",
                            exc,
                        )
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                    else:
                        raise

        from pipelines.seed_gnn.pretrain_gnn import get_split_class_counts

        num_features = self.num_features
        num_classes = self.num_classes
        split_class_counts = get_split_class_counts(self.whole_data)

        metrics_before = deepcopy(self.bef_edit_results.get("metrics")) if isinstance(self.bef_edit_results, dict) else None
        metrics_after = deepcopy(results_after.get("metrics"))

        if not self.is_regression:
            try:
                auc_pr_before = compute_full_auc_pr_by_split(self.model_before, self.whole_data)
                auc_pr_after = compute_full_auc_pr_by_split(self.model, self.whole_data)
            except Exception as exc:
                if self._is_oom_error(exc):
                    logger.warning(
                        "Skipping full-graph AUC-PR due to OOM: %s",
                        exc,
                    )
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    auc_pr_before = {"train": float("nan"), "val": float("nan"), "test": float("nan")}
                    auc_pr_after = {"train": float("nan"), "val": float("nan"), "test": float("nan")}
                else:
                    raise
            if metrics_before:
                for split in ["train", "val", "test"]:
                    metrics_before[split]["auc_pr"] = auc_pr_before[split]
            if metrics_after:
                for split in ["train", "val", "test"]:
                    metrics_after[split]["auc_pr"] = auc_pr_after[split]

        feature_name = getattr(self, "sensitive_feature", "AGE")
        fixed_vals = getattr(self, "fixed_sensitive_values", None)

        def _empty_sens_df() -> pd.DataFrame:
            return pd.DataFrame(
                columns=["Node", "MeanProb", "VarProb", "RelVarProb", "FlipFraction"]
            )

        try:
            val_sens_before, test_sens_before = perturb_feature_and_measure_probs(
                self.model_before, self.whole_data, feature_name=feature_name,
                sensitive_feature_values=fixed_vals, compute_flips=not self.is_regression
            )
            val_sens_after, test_sens_after = perturb_feature_and_measure_probs(
                self.model, self.whole_data, feature_name=feature_name,
                sensitive_feature_values=fixed_vals, compute_flips=not self.is_regression
            )
        except Exception as exc:
            if self._is_oom_error(exc):
                logger.warning(
                    "Skipping perturbation sensitivity metrics due to OOM: %s",
                    exc,
                )
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                val_sens_before = _empty_sens_df()
                test_sens_before = _empty_sens_df()
                val_sens_after = _empty_sens_df()
                test_sens_after = _empty_sens_df()
            else:
                raise

        sensitivity_metrics = {
            "before": {
                "val": {
                    "mean_var": float(val_sens_before["VarProb"].mean()),
                    "mean_rel_var": float(val_sens_before["RelVarProb"].mean()),
                    "mean_flip_fraction": float(val_sens_before.get("FlipFraction", pd.Series([float("nan")])).mean()),
                },
                "test": {
                    "mean_var": float(test_sens_before["VarProb"].mean()),
                    "mean_rel_var": float(test_sens_before["RelVarProb"].mean()),
                    "mean_flip_fraction": float(test_sens_before.get("FlipFraction", pd.Series([float("nan")])).mean()),
                },
            },
            "after": {
                "val": {
                    "mean_var": float(val_sens_after["VarProb"].mean()),
                    "mean_rel_var": float(val_sens_after["RelVarProb"].mean()),
                    "mean_flip_fraction": float(val_sens_after.get("FlipFraction", pd.Series([float("nan")])).mean()),
                },
                "test": {
                    "mean_var": float(test_sens_after["VarProb"].mean()),
                    "mean_rel_var": float(test_sens_after["RelVarProb"].mean()),
                    "mean_flip_fraction": float(test_sens_after.get("FlipFraction", pd.Series([float("nan")])).mean()),
                },
            },
        }

        # Add percentage change from initial variance
        for split in ["val", "test"]:
            before_var = sensitivity_metrics["before"][split]["mean_var"]
            after_var = sensitivity_metrics["after"][split]["mean_var"]
            if before_var != 0:
                pct_change = ((after_var - before_var) / before_var) * 100
            else:
                pct_change = 0.0 if after_var == 0 else float('inf')
            sensitivity_metrics["after"][split]["pct_var_change"] = pct_change

        metrics_json = {
            "experiment": {
                "exp_desc": self.config["management"]["exp_desc"],
                "task": self.config["management"]["task"],
                "seed": SEED,
                "method": "leastsquares",
            },
            "data": {
                "dataset": self.config["eval_params"]["dataset"],
                "num_nodes": int(self.whole_data.num_nodes),
                "num_features": num_features,
                "num_classes": num_classes,
                "class_distribution": split_class_counts,
            },
            "model": {
                "name": self.config["pipeline_params"]["model_name"],
                "architecture": self.config["pipeline_params"]["architecture"],
            },
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "sensitivity_metrics": sensitivity_metrics,
            "selection_correlation_metrics": getattr(self, "selection_correlation_metrics", {}),
            "fairness_metrics": fairness_metrics,
            "edit_params": {
                "lambda_reg": lambda_reg,
                "num_targets": len(node_idx_2flip),
                "pr_alpha": self.pr_alpha,
                "rank_mix_tau": self.rank_mix_tau,
            },
            "weight_change_metrics": getattr(self, "weight_change_metrics", {}),
            "edit_runtime": edit_runtime,
        }

        ls_cfg = self.config["pipeline_params"].get("leastsquares", {})
        strategy_name = ls_cfg.get("strategy_mode", "unknown")

        base_name = ls_cfg.get("metrics_save_name", "metrics_edit")
        if base_name.endswith(".json"):
            base_name = base_name[:-5]
        deg_mode = ls_cfg.get("degree_filter", None)
        deg_frac = ls_cfg.get("degree_fraction", None)

        
        _uses_pr = ("_pr" in strategy_name or "_divrank" in strategy_name)

        suffixes = [
            strategy_name,
            f"lam{lambda_reg}",
        ]
        if deg_mode is not None:
            suffixes.append(f"deg{deg_mode}{int(deg_frac*100)}")
        if hasattr(self, "top_fraction"):
            suffixes.append(f"top{self.top_fraction}")
        if _uses_pr:
            suffixes.append(f"pra{self.pr_alpha}")
            suffixes.append(f"rmx{self.rank_mix_tau}")
        if "retention" in strategy_name:
            gamma_val = ls_cfg.get("gamma_retain", self.gamma_retain)
            suffixes.append(f"gamma{gamma_val}")

        num_layers = self.config.get("pipeline_params", {}).get("architecture", {}).get("num_layers")
        if num_layers is not None:
            suffixes.append(f"layers{num_layers}")

        suffixes.append(f"seed{ls_cfg.get('seed', SEED)}")

        save_name = base_name + "_" + "_".join(suffixes) + ".json"

        out_file = os.path.join(
            self.config["management"]["output_folder_dir"],
            save_name,
        )
        os.makedirs(self.config["management"]["output_folder_dir"], exist_ok=True)

        with open(out_file, "w") as f:
            json.dump(metrics_json, f, indent=2)
        logger.info(f"Saved edit metrics to {out_file}")
        self.attach_edit_checkpoint_artifacts(out_file)

        bef_acc, val_acc, test_acc = results_after["overall"]
        raw_results = [[bef_acc, val_acc, test_acc]]

        return raw_results, None
