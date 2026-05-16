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
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv
from torch_geometric.utils import degree as pyg_degree
from sklearn.decomposition import PCA
from collections import OrderedDict

from editing_pipelines.editors.base import BaseEditor
from edit_gnn.utils import prediction, test as seed_test
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
)


from editing_pipelines.utils.model_io import detect_backbone_module, log_forward_mode   


# Import from seed-gnn directory
import sys
sys.path.append('/home/model_editing/gnn-editing-exploration/seed-gnn')
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

        self.save_dir = (
            f"/home/model_editing/data/editing_pipelines/leastsquares/"
            f"visualization_plots/{self.config['eval_params']['dataset']}_"
            f"{self.config['pipeline_params']['model_name']}_{suffix}"
        )

        self.load_model_and_data()
        backbone, bb_name = detect_backbone_module(self.model)
        self.bb_name = bb_name
        log_forward_mode(self.model, self.bb_name, logger)
        self.use_mlp_linears = bool(self.config['pipeline_params'].get('leastsquares', {}).get('use_mlp_linears', False))
        self.use_mean_steering_signal = bool(self.config['pipeline_params'].get('leastsquares', {}).get('mean_steering_signal', False))
        device = self.get_device()
        model = self.model.to(device)
        self.bef_edit_results = seed_test(self.model, self.whole_data)
        # print(self.model.mlp_freezed)
        print(f"Before edit results after loading model 0: {self.bef_edit_results['overall']}")

        # self.model.mlp_freezed = False
        print(f"Model: {self.model.__class__.__name__}, MLP freezed: {self.model.mlp_freezed}")
        # self.bef_edit_results = seed_test(self.model, self.whole_data)
        # print(f"Before edit results after loading model 1: {self.bef_edit_results['overall']}")

        # for p in backbone.parameters():
        #     p.requires_grad = True
        # self.bef_edit_results = seed_test(self.model, self.whole_data)
        # print(f"Before edit results after loading model 2: {self.bef_edit_results['overall']}")
        # for p in self.model.MLP.parameters():
        #     p.requires_grad = True
        # self.bef_edit_results = seed_test(self.model, self.whole_data)
        print(f"Before edit results after loading model 3: {self.bef_edit_results['overall']}")
        log_forward_mode(self.model, self.bb_name, logger)

        self.mu = 3 
    
        # Name of sensitive feature (dataset-specific mapping already fills this)
        self.sensitive_feature = self.config.get("pipeline_params", {}).get("sensitive_feature", "AGE")

        # Optional fixed values used for perturbation of the sensitive feature
        self.fixed_sensitive_values = self.config.get("pipeline_params", {}).get("fixed_sensitive_values", None)

        
        print(f"Mu: {self.mu}")

    # @torch.no_grad()
    # def select_representative_examples(
    #     self,
    #     feature_name: str = "AGE",
    #     num_bins: int = 4,
    #     top_fraction: float = 0.25,
    #     **kwargs,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Select representative examples:
    #     - Correctly classified validation samples.
    #     - Split into quantile bins based on sensitive attribute (e.g., AGE).
    #     - Take top `top_fraction` highest-confidence examples from each bin.

    #     Returns:
    #         selected_indices (Tensor): indices of chosen nodes
    #         selected_labels (Tensor): their true labels
    #     """

    #     model = self.model.eval()
    #     device = self.get_device()
    #     data = self.whole_data.to(device)

    #     # --- 1. Model predictions and confidence ---
    #     logits = model(**grab_input(data))
    #     probs = torch.softmax(logits, dim=-1)
    #     conf, preds = probs.max(dim=-1)
    #     y_true = data.y

    #     # --- 2. Validation subset ---
    #     val_mask = data.val_mask
    #     val_idx = val_mask.nonzero(as_tuple=False).view(-1)
    #     preds_val = preds[val_mask]
    #     y_true_val = y_true[val_mask]
    #     conf_val = conf[val_mask]

    #     # --- 3. Correctly classified samples ---
    #     correct_mask = preds_val == y_true_val
    #     val_idx_correct = val_idx[correct_mask]
    #     conf_correct = conf_val[correct_mask]
    #     y_true_correct = y_true_val[correct_mask]

    #     # --- 4. Sensitive attribute values ---
    #     if not hasattr(data, "feature_names") or feature_name not in data.feature_names:
    #         raise ValueError(f"Feature '{feature_name}' not found in data.feature_names")
    #     feat_idx = data.feature_names.index(feature_name)
    #     feature_vals = data.x[:, feat_idx]
    #     feature_vals_correct = feature_vals[val_mask][correct_mask].float()

    #     # --- 5. Compute quantile bins ---
    #     quantiles = torch.linspace(0, 1, num_bins + 1, device=device)
    #     bin_edges = torch.quantile(feature_vals_correct, quantiles)
    #     bin_ids = torch.bucketize(feature_vals_correct, bin_edges, right=False) - 1
    #     bin_ids = torch.clamp(bin_ids, 0, num_bins - 1)

    #     # --- 6. Pick top fraction by confidence in each bin ---
    #     selected_indices = []
    #     for b in range(num_bins):
    #         bin_mask = bin_ids == b
    #         if bin_mask.sum() == 0:
    #             continue
    #         conf_bin = conf_correct[bin_mask]
    #         idx_bin = val_idx_correct[bin_mask]
    #         sorted_conf, sort_idx = torch.sort(conf_bin, descending=True)
    #         num_to_select = max(1, int(top_fraction * len(sort_idx)))
    #         top_idx = sort_idx[:num_to_select]
    #         selected_indices.append(idx_bin[top_idx])

    #     if len(selected_indices) == 0:
    #         raise ValueError("No representative examples selected.")
    #     selected_indices = torch.cat(selected_indices)
    #     selected_labels = y_true[selected_indices]

    #     # Store internally for later use
    #     self.representative_examples = selected_indices
    #     self.representative_labels = selected_labels
    #     self.feature_name = feature_name
    #     self.num_bins = num_bins

    #     logger.info(
    #         f"[Representative Selection] Selected {selected_indices.numel()} examples "
    #         f"({top_fraction*100:.0f}% top confidence per {num_bins} bins of {feature_name})."
    #     )

    #     return selected_indices.to(device), selected_labels.to(device)

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

        # --- 3. Model predictions and confidence on full graph ---
        logits = model(**grab_input(data))
        probs = torch.softmax(logits, dim=-1)
        conf, preds = probs.max(dim=-1)
        y_true = data.y

        # --- 4. Validation subset ---
        val_mask = data.val_mask
        val_idx = val_mask.nonzero(as_tuple=False).view(-1)

        preds_val = preds[val_mask]
        y_true_val = y_true[val_mask]
        conf_val = conf[val_mask]
        probs_val = probs[val_mask]

        feature_vals_all = data.x[:, feat_idx]
        feature_vals_val = feature_vals_all[val_mask].float()

        # --- 5. Decide which VAL samples participate ---
        if only_correct:
            base_mask = preds_val == y_true_val    # only correctly classified VAL samples
        else:
            base_mask = torch.ones_like(y_true_val, dtype=torch.bool)  # all VAL samples

        if base_mask.sum() == 0:
            raise ValueError("No validation samples selected with the given 'only_correct' setting.")

        sel_idx_val = val_idx[base_mask]          # indices in whole graph
        conf_sel = conf_val[base_mask]
        y_sel = y_true_val[base_mask]
        probs_sel = probs_val[base_mask]
        feature_vals_sel = feature_vals_val[base_mask]

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
            min_val = feature_vals_val.min()
            max_val = feature_vals_val.max()
            new_vals = min_val + (max_val - min_val) * torch.rand_like(feature_vals_sel)

        # 2b) Create a perturbed copy of x, only for selected validation nodes
        x_pert = x_orig.clone()
        x_pert[sel_idx_val, feat_idx] = new_vals

        original_x = data.x
        data.x = x_pert
        logits_pert = model(**grab_input(data))
        data.x = original_x  # restore
        probs_pert_sel = torch.softmax(logits_pert, dim=-1)[sel_idx_val]

        # --- 6. Compute probability shift (L1) ---
        delta_probs = torch.sum(torch.abs(probs_pert_sel - probs_sel), dim=-1)

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
                delta_bin = delta_probs[bin_mask]
                idx_bin = sel_idx_val[bin_mask]
                sorted_delta, sort_idx = torch.sort(delta_bin, descending=True)
                num_to_select = max(1, int(top_fraction * len(sort_idx)))
                top_idx = sort_idx[:num_to_select]
                selected_indices.append(idx_bin[top_idx])

            if len(selected_indices) == 0:
                raise ValueError("No representative examples selected (sensitivity-based).")
            selected_indices = torch.cat(selected_indices)
        else:
            # Global selection by Δprob
            sorted_delta, sort_idx = torch.sort(delta_probs, descending=True)
            num_to_select = max(1, int(top_fraction * len(sort_idx)))
            selected_indices = sel_idx_val[sort_idx[:num_to_select]]

        selected_labels = y_true[selected_indices]

        # --- 8. Plot confidence vs Δprob for diagnostics ---
        os.makedirs(self.save_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(conf_sel.cpu(), delta_probs.cpu(), alpha=0.6, s=20)
        ax.set_xlabel("Initial Confidence (Softmax max)")
        ax.set_ylabel(f"Δ Probability after {feature_name} Perturbation (L1)")
        ax.set_title(f"Sensitivity: Confidence vs Probability Shift ({feature_name})")
        plt.tight_layout()
        fig.savefig(os.path.join(self.save_dir, "perturbation_sensitivity_plot.png"), dpi=150)
        plt.close(fig)

        # --- 9. Store internally for later use ---
        self.representative_examples = selected_indices
        self.representative_labels = selected_labels
        self.feature_name = feature_name
        self.num_bins = num_bins

        logger.info(
            f"[Representative Selection | Sensitivity-based] Selected {selected_indices.numel()} examples "
            f"({top_fraction*100:.0f}% top Δprob "
            f"{'per bin' if per_bin_selection else 'globally'}) "
            f"for feature '{feature_name}', only_correct={only_correct}."
        )

        return selected_indices.to(device), selected_labels.to(device)

    def curate_steering_examples(
    self,
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
            print("U")
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
            min_v = val_feature_vals.min()
            max_v = val_feature_vals.max()
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

        print(f"\n[DEBUG Augmentation Validation]")
        print(f"  Sensitive feature: {sensitive_feature}")
        print(f"  Mode: {'FIXED_VALUES' if use_fixed_values else 'UNIFORM_MIN_MAX'}")
        print(f"  Selected nodes: {selected_idx.numel()}")
        print(f"  Nodes changed at least once: {changed_nodes}/{selected_idx.numel()} "
              f"({100*changed_nodes/selected_idx.numel():.1f}%)")
        print(f"  Total changed assignments (over all K): {debug_changes_total}")

        sample_to_show = min(10, selected_idx.numel())
        print("\n  Example changes:")
        for idx in selected_idx[:sample_to_show]:
            orig = float(x[idx, feat_idx])
            vals = [float(X_aug[idx, feat_idx]) for X_aug in aug_feature_matrices]
            print(f"    {int(idx):5d}: {orig:.3f} -> {', '.join(f'{v:.3f}' for v in vals)}")

        self.X_aug_all = aug_feature_matrices
        return aug_feature_matrices


    def compute_fisher_information(self, model, data, indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute diagonal Fisher Information (Ω) over provided node indices, with parameter names."""
        model.eval()
        fisher = {}

        from edit_gnn.utils import grab_input
        criterion = nn.CrossEntropyLoss()

        out = model(**grab_input(data))
        if indices.dim() > 1:
            indices = indices.squeeze(dim=1)
        indices = indices.to(out.device)
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
        print("Selecting editable layers...")

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
                    _ = self._get_weight(module)
                    editable_layers.append(module)
                except Exception:
                    pass

        if len(editable_layers) == 0:
            raise RuntimeError("No editable layers found in the model!")

        self.editable_layers = editable_layers
        layer_names = [type(l).__name__ for l in editable_layers]
        logger.info(f"[Layer Selection] Editing enabled for {len(editable_layers)} layers: {layer_names} | only_linear={only_linear} | use_mlp_linears={self.use_mlp_linears}")
        return editable_layers


    
    @torch.no_grad()
    def sanity_model_eval(self):
        device = self.get_device()
        self.model = self.model.to(device).eval()
        data = self.whole_data.to(device)

        logits = prediction(self.model, data)
        preds = logits.argmax(dim=-1)
        y = data.y

        def acc_and_n(mask, name):
            idx = mask.nonzero(as_tuple=False).view(-1)
            n = idx.numel()
            a = float('nan') if n == 0 else (preds[idx] == y[idx]).float().mean().item()
            logger.info(f"[{name}] n={n}, acc={a:.4f}")
            return a, n

        train_acc, ntr = acc_and_n(data.train_mask, "TRAIN")
        val_acc, nval  = acc_and_n(data.val_mask,   "VAL")
        test_acc, nte  = acc_and_n(data.test_mask,  "TEST")
        overall = (preds == y).float().mean().item()
        logger.info(f"[OVERALL] n={y.numel()}, acc={overall:.4f}")

        return dict(train=(train_acc, ntr), val=(val_acc, nval), test=(test_acc, nte), overall=overall)

    @torch.no_grad()
    def _evaluate_edit_effects(
        self,
        save_dir: Optional[str] = None,
        feature_name: Optional[str] = None,
        num_bins: int = 3,
        per_age: bool = False,   # if True, also dumps per-feature-value (no binning) CSVs
    ):
        """
        Evaluate edit effects using the pipeline's state and helpers, for an arbitrary
        sensitive feature (e.g. AGE, INCOME, GENDER, ...).

        Assumes:
        - `self.model_before` holds the pre-edit model (set in edit_model)
        - `self.model` holds the post-edit model
        - `self.representative_examples` are the selected targets
        - `self.X_aug_all` (optional) are K full feature matrices used for steering
        - `self.whole_data` has masks: val_mask, test_mask, labels: y, and feature_names incl. `feature_name`
        """

        # --- Setup & guards
        device = self.get_device()
        data = self.whole_data.to(device)

        os.makedirs(self.save_dir, exist_ok=True)
        if not hasattr(self, "model_before"):
            raise RuntimeError("`self.model_before` not found. Ensure edit_model saved a pre-edit snapshot.")

        model_before = self.model_before.to(device).eval()
        model_after  = self.model.to(device).eval()

        logits_before_full = prediction(model_before, data)
        logits_after_full = prediction(model_after, data)

        # Resolve sensitive feature name
        if feature_name is None:
            # Prefer self.feature_name (set by selection), else config, else default "AGE"
            feature_name = getattr(
                self,
                "feature_name",
                self.config.get("pipeline_params", {}).get("sensitive_feature", "AGE"),
            )

        assert hasattr(data, "feature_names") and feature_name in data.feature_names, \
            f"Feature '{feature_name}' not found in data.feature_names"

        # === Helper: get predictions, optionally with a temporary x override ===
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

        # === 1) Base predictions on original data (before & after)
        y_true = data.y
        preds_b = logits_before_full.argmax(dim=-1)
        preds_a = logits_after_full.argmax(dim=-1)
        correct_before = preds_b.eq(y_true)
        correct_after = preds_a.eq(y_true)
        split_subsets = {
            "VAL": self._build_transition_subsets(data.val_mask, correct_before, correct_after),
            "TEST": self._build_transition_subsets(data.test_mask, correct_before, correct_after),
        }
        feat_idx = data.feature_names.index(feature_name)

        # --- 1.5) Fairness metrics (test split)
        fairness_before = self._compute_fairness_metrics(
            model=model_before,
            preds=preds_b,
            data=data,
            feat_idx=feat_idx,
            mask=data.test_mask
        )
        fairness_after = self._compute_fairness_metrics(
            model=model_after,
            preds=preds_a,
            data=data,
            feat_idx=feat_idx,
            mask=data.test_mask
        )
        logger.info(
            f"[Fairness BEFORE] SP={fairness_before['sp']:.4f} "
            f"EO={fairness_before['eo']:.4f} "
            f"Counterfactual={fairness_before['counterfactual']:.4f} "
            f"Instability={fairness_before['instability']:.4f} "
            f"FeatureInstability={fairness_before['feature_instability']:.4f}"
        )
        logger.info(
            f"[Fairness AFTER]  SP={fairness_after['sp']:.4f} "
            f"EO={fairness_after['eo']:.4f} "
            f"Counterfactual={fairness_after['counterfactual']:.4f} "
            f"Instability={fairness_after['instability']:.4f} "
            f"FeatureInstability={fairness_after['feature_instability']:.4f}"
        )

        # === 2) Split-wise accuracy utility
        def acc_on_mask(preds: torch.Tensor, mask: torch.Tensor, name: str) -> Tuple[float, torch.Tensor]:
            idx = mask.nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                logger.warning(f"[{name}] mask empty.")
                return float("nan"), idx
            acc = (preds[idx] == y_true[idx]).float().mean().item()
            return acc, idx

        val_acc_b, val_idx   = acc_on_mask(preds_b, data.val_mask,  "VAL")
        val_acc_a, _         = acc_on_mask(preds_a, data.val_mask,  "VAL")
        test_acc_b, test_idx = acc_on_mask(preds_b, data.test_mask, "TEST")
        test_acc_a, _        = acc_on_mask(preds_a, data.test_mask, "TEST")

        logger.info(f"[VAL]  accuracy before={val_acc_b:.4f}  after={val_acc_a:.4f}  Δ={val_acc_a - val_acc_b:+.4f}")
        logger.info(f"[TEST] accuracy before={test_acc_b:.4f}  after={test_acc_a:.4f}  Δ={test_acc_a - test_acc_b:+.4f}")

        # Save split summary CSV
        pd.DataFrame(
            [
                {"Split": "VAL",  "Acc_Before": val_acc_b,  "Acc_After": val_acc_a,  "Delta": val_acc_a - val_acc_b},
                {"Split": "TEST", "Acc_Before": test_acc_b, "Acc_After": test_acc_a, "Delta": test_acc_a - test_acc_b},
            ]
        ).to_csv(os.path.join(self.save_dir, "split_accuracy_summary.csv"), index=False)

        feature_values_np = data.x[:, feat_idx].detach().cpu().numpy()
        degrees_np = self._get_node_degrees(data)
        logits_before_np = logits_before_full.detach().cpu().numpy()
        logits_after_np = logits_after_full.detach().cpu().numpy()
        prob_all_before = torch.softmax(logits_before_full, dim=-1)
        prob_all_after = torch.softmax(logits_after_full, dim=-1)
        node_indices = torch.arange(prob_all_before.size(0), device=prob_all_before.device)
        prob_selected_before = prob_all_before[node_indices, y_true].detach().cpu().numpy()
        prob_selected_after = prob_all_after[node_indices, y_true].detach().cpu().numpy()

        var_frames: Dict[str, Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]] = {}
        try:
            fixed_vals = getattr(self, "fixed_sensitive_values", None)
            val_sens_before_plot, test_sens_before_plot = perturb_feature_and_measure_probs(
                model_before, data, feature_name=feature_name,
                sensitive_feature_values=fixed_vals, compute_flips=True
            )
            val_sens_after_plot, test_sens_after_plot = perturb_feature_and_measure_probs(
                model_after, data, feature_name=feature_name,
                sensitive_feature_values=fixed_vals, compute_flips=True
            )
            var_frames = {
                "VAL": (val_sens_before_plot, val_sens_after_plot),
                "TEST": (test_sens_before_plot, test_sens_after_plot),
            }
        except Exception as err:
            logger.warning(f"[TransitionViz] Skipped variance-based plots due to error: {err}")
            var_frames = {}

        # === 3) Augmented accuracies on representative examples (averaged over K)
        aug_b = aug_a = None
        if hasattr(self, "X_aug_all") and isinstance(self.X_aug_all, list) and len(self.X_aug_all) > 0:
            if not hasattr(self, "representative_examples"):
                logger.warning("`representative_examples` missing; cannot compute augmented-set accuracy.")
            else:
                rep_idx = self.representative_examples.to(device=device, dtype=torch.long)

                acc_b_list, acc_a_list = [], []
                for i, X_aug in enumerate(self.X_aug_all):
                    X_aug = X_aug.to(device)
                    preds_b_aug = get_preds(model_before, data, override_x=X_aug)
                    preds_a_aug = get_preds(model_after,  data, override_x=X_aug)
                    acc_b_i = (preds_b_aug[rep_idx] == y_true[rep_idx]).float().mean().item()
                    acc_a_i = (preds_a_aug[rep_idx] == y_true[rep_idx]).float().mean().item()
                    acc_b_list.append(acc_b_i)
                    acc_a_list.append(acc_a_i)
                    logger.info(f"[AUG rep {i}] before={acc_b_i:.4f} after={acc_a_i:.4f} Δ={acc_a_i-acc_b_i:+.4f}")

                aug_b = float(np.mean(acc_b_list))
                aug_a = float(np.mean(acc_a_list))
                logger.info(
                    f"[AUG (representative mean over K={len(self.X_aug_all)})] "
                    f"before={aug_b:.4f} after={aug_a:.4f} Δ={aug_a-aug_b:+.4f}"
                )

                pd.DataFrame(
                    {
                        "AugIndex": list(range(len(self.X_aug_all))),
                        "Acc_Before": acc_b_list,
                        "Acc_After": acc_a_list,
                        "Delta": [a - b for a, b in zip(acc_a_list, acc_b_list)],
                    }
                ).to_csv(os.path.join(self.save_dir, "augmented_representatives_accuracy.csv"), index=False)
        else:
            logger.info("No `X_aug_all` found — skipping augmented representative accuracy.")

        # === 4) Transition tables (overall + per split)
        def transition_counts(preds_before: torch.Tensor, preds_after: torch.Tensor, idx: Optional[torch.Tensor] = None) -> pd.DataFrame:
            if idx is not None and idx.numel() > 0:
                b = preds_before[idx] == y_true[idx]
                a = preds_after[idx]  == y_true[idx]
            else:
                b = preds_before == y_true
                a = preds_after  == y_true
            labels = ["BeforeCorrect", "AfterCorrect", "Count"]
            arr = torch.stack([b, a], dim=1).cpu().numpy()
            bb_ff = np.logical_and(arr[:, 0] == False, arr[:, 1] == False).sum()  # F->F
            bb_ft = np.logical_and(arr[:, 0] == False, arr[:, 1] == True ).sum()  # F->T
            bb_tf = np.logical_and(arr[:, 0] == True , arr[:, 1] == False).sum()  # T->F
            bb_tt = np.logical_and(arr[:, 0] == True , arr[:, 1] == True ).sum()  # T->T
            df = pd.DataFrame(
                [
                    [False, False, bb_ff],
                    [False, True,  bb_ft],
                    [True,  False, bb_tf],
                    [True,  True,  bb_tt],
                ],
                columns=labels,
            )
            return df

        def save_transition(df: pd.DataFrame, tag: str):
            p = os.path.join(self.save_dir, f"transition_{tag}.csv")
            df.to_csv(p, index=False)
            mat = np.zeros((2, 2), dtype=int)
            for _, r in df.iterrows():
                i = 1 if r["BeforeCorrect"] else 0
                j = 1 if r["AfterCorrect"]  else 0
                mat[i, j] = int(r["Count"])
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(mat)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["Wrong", "Correct"])
            ax.set_yticks([0, 1]); ax.set_yticklabels(["Wrong", "Correct"])
            ax.set_xlabel("After"); ax.set_ylabel("Before")
            for (i, j), val in np.ndenumerate(mat):
                ax.text(j, i, str(val), ha="center", va="center")
            ax.set_title(f"Transition ({tag})")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(os.path.join(self.save_dir, f"transition_{tag}.png"), dpi=150)
            plt.close(fig)

        trans_all  = transition_counts(preds_b, preds_a, idx=None)
        trans_val  = transition_counts(preds_b, preds_a, idx=val_idx)
        trans_test = transition_counts(preds_b, preds_a, idx=test_idx)

        save_transition(trans_all,  "overall")
        save_transition(trans_val,  "val")
        save_transition(trans_test, "test")

        for split_name, idx_subset in zip(
            ["VAL", "TEST"],
            [val_idx, test_idx],
        ):
            subsets = split_subsets.get(split_name)
            if not subsets:
                continue
            self._plot_transition_subset_distributions(
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
            )

        # === 5) Feature-wise (formerly age-wise) analysis (VAL & TEST)
        feat_col = data.feature_names.index(feature_name)
        feat_vals = data.x[:, feat_col].detach().cpu().float().numpy()

        def feature_bins_quantiles(values: np.ndarray, idx: np.ndarray, n_bins: int):
            """Compute split-specific quantile edges and bin ids for the given indices."""
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
            df = pd.DataFrame(
                rows,
                columns=[bin_col, "Acc_Before", "Acc_After", "Count"],
            )
            df.to_csv(
                os.path.join(self.save_dir, f"{feature_name.lower()}_buckets_{tag}.csv"),
                index=False,
            )

            if len(df) > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                x = np.arange(len(df))
                ax.bar(x - 0.18, df["Acc_Before"].values, width=0.36, label="Before")
                ax.bar(x + 0.18, df["Acc_After"].values,  width=0.36, label="After")
                ax.set_xticks(x)
                ax.set_xticklabels(df[bin_col].tolist(), rotation=10, ha="right")
                ax.set_ylabel("Accuracy")
                ax.set_title(f"{split_name} {feature_name}-wise Accuracy")
                ax.legend()
                fig.tight_layout()
                fig.savefig(
                    os.path.join(self.save_dir, f"{feature_name.lower()}_buckets_{tag}.png"),
                    dpi=150,
                )
                plt.close(fig)
            return df

        val_feat_df  = bucket_acc_report("Validation", val_idx,  "val")
        test_feat_df = bucket_acc_report("Test",       test_idx, "test")

        # optional: per-feature-value dump (no binning), useful for debugging/regression
        if per_age:
            def per_feature_dump(idx_tensor: torch.Tensor, tag: str):
                idx = idx_tensor.detach().cpu().numpy()
                df = pd.DataFrame(
                    {
                        "Node": idx,
                        feature_name: feat_vals[idx],
                        "Correct_Before": (preds_b[idx] == y_true[idx]).cpu().numpy(),
                        "Correct_After":  (preds_a[idx] == y_true[idx]).cpu().numpy(),
                    }
                )
                df.to_csv(
                    os.path.join(self.save_dir, f"per_{feature_name.lower()}_{tag}.csv"),
                    index=False,
                )
            per_feature_dump(val_idx,  "val")
            per_feature_dump(test_idx, "test")

        # === 6) Final logging summary
        logger.info("\n[=== EDIT EVALUATION SUMMARY ===]")
        logger.info(f"VAL   acc  before={val_acc_b:.4f}  after={val_acc_a:.4f}  Δ={val_acc_a - val_acc_b:+.4f}")
        logger.info(f"TEST  acc  before={test_acc_b:.4f} after={test_acc_a:.4f} Δ={test_acc_a - test_acc_b:+.4f}")
        if aug_b is not None:
            logger.info(
                f"AUG (rep mean over K={len(self.X_aug_all)})  "
                f"before={aug_b:.4f} after={aug_a:.4f} Δ={aug_a - aug_b:+.4f}"
            )
        logger.info(f"Saved detailed CSVs/plots to: {self.save_dir}")

        # === 7) AUC by feature (equal-width & quantile bins)
        auc_before_eq = compute_auc_per_feature_bucket(
            model_before, data, feature_name=feature_name, mode="equal_width"
        )
        auc_after_eq  = compute_auc_per_feature_bucket(
            model_after,  data, feature_name=feature_name, mode="equal_width"
        )

        auc_before_eq.to_csv(
            os.path.join(self.save_dir, f"auc_{feature_name.lower()}_equal_before.csv"),
            index=False,
        )
        auc_after_eq.to_csv(
            os.path.join(self.save_dir, f"auc_{feature_name.lower()}_equal_after.csv"),
            index=False,
        )

        logger.info(f"[AUC by {feature_name} | equal] before:\n{auc_before_eq}")
        logger.info(f"[AUC by {feature_name} | equal] after:\n{auc_after_eq}")

        auc_before_q = compute_auc_per_feature_bucket(
            model_before, data, feature_name=feature_name, mode="quantile"
        )
        auc_after_q  = compute_auc_per_feature_bucket(
            model_after,  data, feature_name=feature_name, mode="quantile"
        )

        auc_before_q.to_csv(
            os.path.join(self.save_dir, f"auc_{feature_name.lower()}_quantile_before.csv"),
            index=False,
        )
        auc_after_q.to_csv(
            os.path.join(self.save_dir, f"auc_{feature_name.lower()}_quantile_after.csv"),
            index=False,
        )

        logger.info(f"[AUC by {feature_name} | quantile] before:\n{auc_before_q}")
        logger.info(f"[AUC by {feature_name} | quantile] after:\n{auc_after_q}")

        # === 8) Mean/Median probability vs feature (equal-width & quantile)
        mm_subset = self.config.get('pipeline_params', {}).get('leastsquares', {}).get('mean_median_subset', 'val_test')
        mm_prob_mode = self.config.get('pipeline_params', {}).get('leastsquares', {}).get('mean_median_prob_mode', 'positive')
        prob_label = "p(true class)" if mm_prob_mode == 'true_class' else "p(y=1)"

        mm_before_eq = mean_median_prob_by_feature(
            model_before,
            data,
            feature_name=feature_name,
            mode="equal_width",
            subset=mm_subset,
            prob_mode=mm_prob_mode,
        )
        mm_after_eq  = mean_median_prob_by_feature(
            model_after,
            data,
            feature_name=feature_name,
            mode="equal_width",
            subset=mm_subset,
            prob_mode=mm_prob_mode,
        )

        mm_before_eq.to_csv(
            os.path.join(self.save_dir, f"mean_median_prob_{feature_name.lower()}_equal_before.csv"),
            index=False,
        )
        mm_after_eq.to_csv(
            os.path.join(self.save_dir, f"mean_median_prob_{feature_name.lower()}_equal_after.csv"),
            index=False,
        )

        mm_before_q = mean_median_prob_by_feature(
            model_before,
            data,
            feature_name=feature_name,
            mode="quantile",
            subset=mm_subset,
            prob_mode=mm_prob_mode,
        )
        mm_after_q  = mean_median_prob_by_feature(
            model_after,
            data,
            feature_name=feature_name,
            mode="quantile",
            subset=mm_subset,
            prob_mode=mm_prob_mode,
        )

        mm_before_q.to_csv(
            os.path.join(self.save_dir, f"mean_median_prob_{feature_name.lower()}_quantile_before.csv"),
            index=False,
        )
        mm_after_q.to_csv(
            os.path.join(self.save_dir, f"mean_median_prob_{feature_name.lower()}_quantile_after.csv"),
            index=False,
        )

        # === 9) Sensitivity: perturb sensitive feature and measure variance of true-class probs
        # Use fixed_sensitive_values, if provided, else uniform in [min, max] on VAL set (as per our new helper)
        fixed_vals = getattr(self, "fixed_sensitive_values", None)

        val_before, test_before = perturb_feature_and_measure_probs(
            model_before,
            data,
            feature_name=feature_name,
            sensitive_feature_values=fixed_vals,
        )
        val_after,  test_after  = perturb_feature_and_measure_probs(
            model_after,
            data,
            feature_name=feature_name,
            sensitive_feature_values=fixed_vals,
        )

        val_before.to_csv(
            os.path.join(self.save_dir, f"{feature_name.lower()}_sensitivity_val_before.csv"),
            index=False,
        )
        test_before.to_csv(
            os.path.join(self.save_dir, f"{feature_name.lower()}_sensitivity_test_before.csv"),
            index=False,
        )
        val_after.to_csv(
            os.path.join(self.save_dir, f"{feature_name.lower()}_sensitivity_val_after.csv"),
            index=False,
        )
        test_after.to_csv(
            os.path.join(self.save_dir, f"{feature_name.lower()}_sensitivity_test_after.csv"),
            index=False,
        )

        # === 10) Per-node variance % drop (VAL & TEST) ===
        def compute_and_plot_var_drop(
            df_before: pd.DataFrame,
            df_after: pd.DataFrame,
            split: str,
        ) -> pd.DataFrame:
            merged = df_before.merge(df_after, on="Node", suffixes=("_before", "_after"))
            merged["VarDropAbs"] = merged["VarProb_before"] - merged["VarProb_after"]

            eps_thresh = 1e-10
            denom = merged["VarProb_before"].copy()
            small_mask = denom.abs() < eps_thresh
            denom[small_mask] = np.nan
            merged["VarDropPct"] = 100.0 * merged["VarDropAbs"] / denom
            merged["VarDropPct_clipped"] = merged["VarDropPct"].clip(-500, 500)

            merged.to_csv(
                os.path.join(self.save_dir, f"{feature_name.lower()}_var_drop_per_node_{split}.csv"),
                index=False,
            )

            # Histogram of percentage drops
            fig, ax = plt.subplots(figsize=(6, 4))
            vals = merged["VarDropPct_clipped"].dropna()
            ax.hist(vals, bins=50)
            ax.axvline(0.0, linestyle="--")
            ax.set_xlabel(
                "Variance drop (%)  (positive = variance decreased)\n"
                "(clipped to [-500, 500])"
            )
            ax.set_ylabel("Number of nodes")
            ax.set_title(
                f"Per-node variance drop under {feature_name} perturbations ({split.upper()})"
            )
            plt.tight_layout()
            fig.savefig(
                os.path.join(self.save_dir, f"{feature_name.lower()}_var_drop_hist_{split}.png"),
                dpi=150,
            )
            plt.close(fig)

            # Scatter before vs after variance
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(
                merged["VarProb_before"],
                merged["VarProb_after"],
                alpha=0.3,
                s=10,
            )
            min_val = min(merged["VarProb_before"].min(), merged["VarProb_after"].min())
            max_val = max(merged["VarProb_before"].max(), merged["VarProb_after"].max())
            ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
            ax.set_xlabel("VarProb before")
            ax.set_ylabel("VarProb after")
            ax.set_title(f"Per-node VarProb (before vs after) [{split.upper()}]")
            plt.tight_layout()
            fig.savefig(
                os.path.join(self.save_dir, f"{feature_name.lower()}_VarProb_before_vs_after_{split}.png"),
                dpi=150,
            )
            plt.close(fig)

            return merged

        val_var_drop  = compute_and_plot_var_drop(val_before,  val_after,  split="val")
        test_var_drop = compute_and_plot_var_drop(test_before, test_after, split="test")

        # === 11) High-level plots using generalized visualization helpers ===
        # AUC plots
        plot_auc_by_feature_with_counts(
            auc_before_eq,
            auc_after_eq,
            feature_name=feature_name,
            save_dir=self.save_dir,
            suffix="_equal",
        )
        plot_auc_by_feature_with_counts(
            auc_before_q,
            auc_after_q,
            feature_name=feature_name,
            save_dir=self.save_dir,
            suffix="_quantile",
        )

        # Mean/Median probability plots
        plot_mean_prob_by_feature(
            mm_before_eq,
            mm_after_eq,
            feature_name=feature_name,
            stat="mean",
            save_dir=self.save_dir,
            suffix="_equal",
            prob_label=prob_label,
        )
        plot_mean_prob_by_feature(
            mm_before_q,
            mm_after_q,
            feature_name=feature_name,
            stat="mean",
            save_dir=self.save_dir,
            suffix="_quantile",
            prob_label=prob_label,
        )
        plot_mean_prob_by_feature(
            mm_before_eq,
            mm_after_eq,
            feature_name=feature_name,
            stat="median",
            save_dir=self.save_dir,
            suffix="_equal",
            prob_label=prob_label,
        )
        plot_mean_prob_by_feature(
            mm_before_q,
            mm_after_q,
            feature_name=feature_name,
            stat="median",
            save_dir=self.save_dir,
            suffix="_quantile",
            prob_label=prob_label,
        )

        # Distributions for selected representatives and augmented values (use pre-edit model for confidence)
        plot_rep_and_aug_distributions(
            model_before,
            data,
            getattr(self, "representative_examples", []),
            getattr(self, "X_aug_all", []),
            feature_name,
            self.save_dir,
        )

        # Sensitivity plots (generalized)
        plot_feature_sensitivity(
            val_before,
            val_after,
            test_before,
            test_after,
            feature_name=feature_name,
            save_dir=self.save_dir,
        )
        plot_sensitivity_reduction(
            val_before,
            val_after,
            test_before,
            test_after,
            feature_name=feature_name,
            save_dir=self.save_dir,
        )

        return {"before": fairness_before, "after": fairness_after}
 
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

        # 1) representative examples
        node_idx_2flip, flipped_label = self.select_representative_examples(**kwargs)

        # 2) curated steering examples (K augmented full-feature matrices)
        X_aug_all = self.curate_steering_examples(K=K)
        # store for later use by edit_model
        self.X_aug_all = X_aug_all
        self.num_aug = K

        # 3) select editable layers and store them
        self.select_edit_layers(only_linear=kwargs.get("only_linear", False))

        logger.info(f"[select_edit_targets] Selected {node_idx_2flip.numel()} reps, created K = {len(X_aug_all)} aug matrices, {len(self.editable_layers)} editable layers.")
        logger.info(f"Shape of X_aug matrices: {X_aug_all[0].shape}")
        return node_idx_2flip, flipped_label

    # MATH UTILS

    def _stack_rows(self, xs: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(xs, dim=0) if len(xs) > 1 else xs[0]

    def _get_weight(self, layer: nn.Module) -> torch.Tensor:
        """
        Return the primary learnable weight matrix for a GNN layer.
        Supports GCNConv, GATConv, SAGEConv, GINConv, and Linear.
        """
        
        # Case 1: nn.Linear
        if isinstance(layer, nn.Linear):
            return layer.weight

        # Case 2: GCNConv and GraphConv types
        if hasattr(layer, "lin") and hasattr(layer.lin, "weight"):
            return layer.lin.weight

        # Case 3: GATConv used in Pokec — correct weight is lin_src.weight
        if isinstance(layer, GATConv):
            if hasattr(layer, "lin_src") and hasattr(layer.lin_src, "weight"):
                return layer.lin_src.weight

        # Case 4: SAGEConv uses separate linear transforms
        if isinstance(layer, SAGEConv):
            if hasattr(layer, "lin_l") and hasattr(layer.lin_l, "weight"):
                return layer.lin_l.weight

        # Case 5: GINConv — first MLP Linear
        if isinstance(layer, GINConv) and hasattr(layer, "nn"):
            for sub in layer.nn.modules():
                if isinstance(sub, nn.Linear):
                    return sub.weight

        raise AttributeError(
            f"[ERROR] Cannot extract weight for layer {layer.__class__.__name__}.\n"
            f"Available attrs: {list(layer.__dict__.keys())}"
        )



    def _set_weight(self, layer: nn.Module, new_w: torch.Tensor) -> None:
        w = self._get_weight(layer)
        assert w.shape == new_w.shape
        with torch.no_grad():
            w.copy_(new_w)

    @torch.no_grad()
    def _capture_layer_inputs(
        self,
        layer: nn.Module,
        data_obj,
        node_idx: torch.Tensor = None,
        override_x: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Run a forward pass, capture the *input* that hits `layer` for the nodes of interest.
        We assume the model reads `data_obj.x` as node features by convention.
        """
        device = self.get_device()
        # Optionally swap features for this pass (used for augmented set)
        if override_x is not None:
            original_x = data_obj.x
            data_obj.x = override_x

        captured: List[torch.Tensor] = []

        def pre_hook(_module, inp):
            x_in = inp[0]           # shape: (N, d_in)
            if node_idx is None:
                captured.append(x_in)     # <-- store all nodes
            else:
                captured.append(x_in[node_idx])

        handle = layer.register_forward_pre_hook(pre_hook)
        try:
            _ = prediction(self.model, data_obj)  # logits ignored; hook captures layer input
        finally:
            handle.remove()
            if override_x is not None:
                data_obj.x = original_x  # restore

        if len(captured) == 0:
            logger.warning(f"[Capture] Layer {layer.__class__.__name__} received 0 forward calls; skipping.")
            return None
        return captured[0]  # U_ℓ ∈ R^{m × d_in}

    @torch.no_grad()
    def _compute_D_U(
        self,
        layer: nn.Module,
        U_orig: torch.Tensor,
        U_aug: torch.Tensor,
        num_augments: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given U (orig) and U' (aug) at layer input, compute:
            D = W U' - W U
            U = U (orig)
        """
        W = self._get_weight(layer)  # [out, in]
        V_orig = U_orig @ W.t()
        V_aug = U_aug @ W.t()
        if self.use_mean_steering_signal:
            if U_orig.size(0) != U_aug.size(0):
                raise ValueError("U_orig and U_aug must have the same number of rows when computing mean steering signal.")
            if U_aug.size(0) % num_augments != 0:
                raise ValueError("num_augments must divide the stacked rows.")
            nodes_per_augment = U_aug.size(0) // num_augments
            V_aug_reshaped = V_aug.view(num_augments, nodes_per_augment, -1)
            print(f"Shape of V_aug_reshaped: {V_aug_reshaped.shape}")
            V_aug_sum = V_aug_reshaped.sum(dim=0)
            print(f"Shape of V_aug_sum: {V_aug_sum.shape}")
            V_orig_reshaped = V_orig.view(num_augments, nodes_per_augment, -1)
            print(f"Shape of V_orig_reshaped: {V_orig_reshaped.shape}")
            V_orig_base = V_orig_reshaped[0]
            print(f"Shape of V_orig_base: {V_orig_base.shape}")
            V_mean_per_node = (V_orig_base + V_aug_sum) / (num_augments + 1)
            print(f"Shape of V_mean_per_node: {V_mean_per_node.shape}")
            V_mean_tiled = V_mean_per_node.repeat(num_augments, 1)
            D = V_aug - V_mean_tiled            # [m, out]
        else:
            D = V_aug - V_orig                    # [m, out]
        U = U_orig                                # [m, in]
        print(f"Shape of D: {D.shape}")
        print(f"Shape of U: {U.shape}")
        print(f"L2 norm of U_orig - U_aug: {torch.norm(U_orig - U_aug)}")
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
        # feature-space ridge regression
        UtU = U.T @ U                # (d_in, d_in)
        UtD = U.T @ D                # (d_in, d_out)
        reg = lam * torch.eye(UtU.size(0), device=UtU.device, dtype=UtU.dtype)

        # solve (UtU + λI) X = UtD
        X = torch.linalg.solve(UtU + reg, UtD)   # (d_in, d_out)
        S = X.T                                   # (d_out, d_in)
        return S

    @torch.no_grad()
    def _compute_fairness_metrics(
        self,
        model,
        preds: torch.Tensor,
        data,
        feat_idx: int,
        mask: torch.Tensor,
    ) -> Dict[str, float]:
        def safe_mean(vec: torch.Tensor) -> float:
            return float(vec.float().mean().item()) if vec.numel() > 0 else float("nan")

        result = {
            "sp": float("nan"),
            "eo": float("nan"),
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

        groups = torch.sort(torch.unique(s_vals)).values
        if groups.numel() >= 2:
            g0, g1 = groups[0], groups[-1]
            def pos_rate(group_val):
                group_mask = s_vals == group_val
                return safe_mean(preds_mask[group_mask])
            sp = abs(pos_rate(g0) - pos_rate(g1))
            result["sp"] = sp

            def pos_rate_y1(group_val):
                group_mask = (s_vals == group_val) & (y_mask == 1)
                return safe_mean(preds_mask[group_mask])
            eo = abs(pos_rate_y1(g0) - pos_rate_y1(g1))
            result["eo"] = eo

        counterfactual = self._compute_counterfactual_fraction(model, data, feat_idx, mask, preds)
        result["counterfactual"] = counterfactual

        instability = self._compute_instability_fraction(model, data, preds, mask)
        result["instability"] = instability

        feat_instability = self._compute_feature_instability_fraction(model, data, preds, mask, feat_idx)
        result["feature_instability"] = feat_instability

        return result

    @torch.no_grad()
    def _compute_counterfactual_fraction(
        self,
        model,
        data,
        feat_idx: int,
        mask: torch.Tensor,
        base_preds: torch.Tensor,
    ) -> float:
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
    def _compute_instability_fraction(
        self,
        model,
        data,
        base_preds: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
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
    def _compute_feature_instability_fraction(
        self,
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

    def _plot_transition_subset_distributions(
        self,
        split_name: str,
        subset_nodes: Dict[str, torch.Tensor],
        feature_values: np.ndarray,
        degrees: np.ndarray,
        logits_before: np.ndarray,
        logits_after: np.ndarray,
        prob_before: np.ndarray,
        prob_after: np.ndarray,
        var_before_df: Optional[pd.DataFrame],
        var_after_df: Optional[pd.DataFrame],
        feature_name: str,
    ) -> None:
        if not subset_nodes:
            return
        subset_colors = {
            "0->0": "#781C6D",  # purple-ish
            "0->1": "#1a9850",
            "1->0": "#EA7317",  # orange
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

        output_dir = os.path.join(self.save_dir, "transition_analysis")
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
                    color=subset_colors[subset],
                    alpha=0.45,
                    label=subset_labels[subset],
                    density=False,
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

        _plot_hist(feature_values, f"{split_name} | {feature_name} distribution", f"{split_name.lower()}_{feature_name.lower()}_hist.png")
        _plot_hist(degrees, f"{split_name} | Degree distribution", f"{split_name.lower()}_degree_hist.png")

        logit_before_max = logits_before.max(axis=1)
        logit_after_max = logits_after.max(axis=1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        handles = []
        labels = []
        for subset in subset_order:
            idx = subset_idx.get(subset)
            if idx is None or idx.size == 0:
                continue
            sc0 = axes[0].scatter(
                logit_before_max[idx],
                logit_after_max[idx],
                color=subset_colors[subset],
                alpha=0.6,
                s=20,
                label=subset_labels[subset],
            )
            axes[1].scatter(
                prob_before[idx],
                prob_after[idx],
                color=subset_colors[subset],
                alpha=0.6,
                s=20,
                label=subset_labels[subset],
            )
            handles.append(sc0)
            labels.append(subset_labels[subset])
        for ax, metric in zip(axes, ["Logit", "Predicted Prob."]):
            ax.set_xlabel(f"{metric} Before")
            ax.set_ylabel(f"{metric} After")
            min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
            max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", linewidth=1)
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=4)
        fig.suptitle(f"{split_name} | Output shifts")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(output_dir, f"{split_name.lower()}_output_scatter.png"), dpi=150)
        plt.close(fig)

        def _build_var_map(df: Optional[pd.DataFrame]) -> Dict[int, float]:
            if df is None:
                return {}
            return {int(row["Node"]): float(row["VarProb"]) for _, row in df.iterrows()}

        var_before_map = _build_var_map(var_before_df)
        var_after_map = _build_var_map(var_after_df)
        if var_before_map and var_after_map:
            fig, ax = plt.subplots(figsize=(6, 5))
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

        prob_shift = prob_after - prob_before
        emb_norm_before = logits_before / (np.linalg.norm(logits_before, axis=1, keepdims=True) + 1e-8)
        emb_norm_after = logits_after / (np.linalg.norm(logits_after, axis=1, keepdims=True) + 1e-8)
        cos_dist = 1.0 - np.sum(emb_norm_before * emb_norm_after, axis=1)

        fig, ax = plt.subplots(figsize=(6, 5))
        plotted = False
        for subset in subset_order:
            idx = subset_idx.get(subset)
            if idx is None or idx.size == 0:
                continue
            ax.scatter(
                cos_dist[idx],
                np.abs(prob_shift[idx]),
                color=subset_colors[subset],
                alpha=0.6,
                s=20,
                label=subset_labels[subset],
            )
            plotted = True
        if plotted:
            ax.set_xlabel("Cosine Distance (Before vs After)")
            ax.set_ylabel("|Δ Predicted Prob.|")
            ax.set_title(f"{split_name} | Feature similarity vs probability shift")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{split_name.lower()}_cosine_prob_scatter.png"), dpi=150)
        plt.close(fig)

        combined_idx = np.concatenate([arr for arr in subset_idx.values()] + ([self.representative_examples.cpu().numpy()] if hasattr(self, "representative_examples") else []))
        if combined_idx.size >= 2:
            embeddings = np.concatenate(
                [logits_before[combined_idx], logits_after[combined_idx]],
                axis=0,
            )
            pca = PCA(n_components=2, random_state=0)
            pca.fit(embeddings)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for subset in subset_order:
                idx = subset_idx.get(subset)
                if idx is None or idx.size == 0:
                    continue
                coords_before = pca.transform(logits_before[idx])
                coords_after = pca.transform(logits_after[idx])
                axes[0].scatter(
                    coords_before[:, 0],
                    coords_before[:, 1],
                    color=subset_colors[subset],
                    alpha=0.6,
                    s=15,
                    label=subset_labels[subset],
                )
                axes[1].scatter(
                    coords_after[:, 0],
                    coords_after[:, 1],
                    color=subset_colors[subset],
                    alpha=0.6,
                    s=15,
                    label=subset_labels[subset],
                )
            if hasattr(self, "representative_examples"):
                rep_idx = self.representative_examples.detach().cpu().numpy()
                if rep_idx.size > 0:
                    coords_rep_before = pca.transform(logits_before[rep_idx])
                    coords_rep_after = pca.transform(logits_after[rep_idx])
                    axes[0].scatter(
                        coords_rep_before[:, 0],
                        coords_rep_before[:, 1],
                        color="black",
                        alpha=0.9,
                        s=25,
                        marker="x",
                        label="Rep Nodes",
                    )
                    axes[1].scatter(
                        coords_rep_after[:, 0],
                        coords_rep_after[:, 1],
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

    def _compute_feature_instability_fraction(
        self,
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
        sensitive_col = x_noisy[:, feat_idx]
        col_std = sensitive_col.std(unbiased=False)
        noise_scale = 0.01 * (col_std + 1e-6)
        noise = torch.randn_like(sensitive_col) * noise_scale
        x_noisy[:, feat_idx] = sensitive_col + noise
        data.x = x_noisy
        preds_noisy = prediction(model, data).argmax(dim=-1)
        data.x = original_x
        diff = (preds_noisy[idx] != base_preds[idx]).float().mean().item()
        return diff

    # -----------------------------
    # Editing
    # -----------------------------

    def edit_layer(self, layer: nn.Module, S_star: torch.Tensor, use_ewc=False, use_mu=True):
        """
        Edit the parameters of the selected layer, scaled by Fisher Information (if available).
        Matches the correct parameter using identity from model.named_parameters().
        """
        W = self._get_weight(layer)

        mu = self.mu if use_mu else 1

        # Ensure Fisher dict exists
        if not hasattr(self, "fisher_dict") or not isinstance(self.fisher_dict, dict) or not use_ewc:
            logger.warning("[Fisher] No fisher_dict found; applying unweighted edit.")
            self._set_weight(layer, W + mu * S_star)
            return

        # Find exact parameter name by tensor identity
        matched_name = None
        for name, param in self.model.named_parameters():
            if param is W:  # identity match
                matched_name = name
                break

        # If no match found, fall back to plain edit
        if matched_name is None:
            logger.warning(f"[Fisher] Could not find parameter name for {layer.__class__.__name__}; applying unweighted edit.")
            self._set_weight(layer, W + mu * S_star)
            return

        # Fetch corresponding Fisher tensor
        fisher_layer = self.fisher_dict.get(matched_name, None)
        if fisher_layer is None:
            logger.warning(f"[Fisher] No Fisher entry found for {matched_name}; applying unweighted edit.")
            weighted_shift = self.mu * S_star
        else:
            fisher_layer = fisher_layer.to(W.device)
            # Normalize to [0,1] range for stability
            f_norm = fisher_layer / (fisher_layer.max() + 1e-8)
            weighted_shift = mu * (S_star * (1.0 - f_norm))
            logger.info(f"[Fisher-weighted Edit] {matched_name}: "
                        f"maxF={fisher_layer.max():.3e}, meanF={fisher_layer.mean():.3e}, "
                        f"scale≈{mu:.2f}")

        # Apply update
        self._set_weight(layer, W + weighted_shift)

    def edit_model(self, **kwargs) -> List[List[Any]]:

        for n, p in self.model.named_parameters():
            if 'weight' in n:
                f = self.fisher_dict.get(n, None)
                if f is not None:
                    logger.info(f"  ✓ {n:40s} | Fisher shape={tuple(f.shape)} matches weight shape={tuple(p.shape)}")
                else:
                    logger.warning(f"  ✗ {n:40s} | No Fisher found.")
        print(f"Fisher dict keys: {self.fisher_dict.keys()}")
        logger.info("=============================")

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

        print(f"X_orig: {self.whole_data.x}")

        print(f"X_aug_all: {self.X_aug_all}")

        # Decide editing strategy: grouped parallel (GCN+MLP) by depth if available; otherwise fallback per-layer
        if getattr(self, 'parallel_layer_groups', None):
            for depth, group in enumerate(self.parallel_layer_groups):
                # If only_linear, edit only the MLP linear at this depth; else edit all modules in the group
                if getattr(self, 'only_linear', False):
                    # pick the nn.Linear modules that belong to MLP
                    mlp_lins = list(getattr(self.model.MLP, 'lins', [])) if hasattr(self.model, 'MLP') else []
                    layers_iter = [m for m in group if isinstance(m, nn.Linear) and m in mlp_lins]
                else:
                    layers_iter = list(group)
                logger.info(f"[Parallel Edit] Depth {depth} with {len(layers_iter)} layer(s) | only_linear={getattr(self, 'only_linear', False)}")

                # Prepare augmentations once per depth
                if hasattr(self, "X_aug_all") and len(self.X_aug_all) > 0:
                    X_aug_all = self.X_aug_all
                else:
                    num_aug = int(self.config.get("leastsquares", {}).get("num_aug", 6))
                    X_aug_all = self.curate_steering_examples(K=num_aug)

                num_aug = len(X_aug_all)

                # For each selected layer, compute S without applying, then apply all
                pending_applies = []  # list of (layer, S)
                for layer in layers_iter:
                    print(f"Editing layer: {layer.__class__.__name__} (depth {depth})")

                    # 1) Capture U under original input
                    U_orig = self._capture_layer_inputs(
                        layer=layer,
                        data_obj=self.whole_data,
                        node_idx=None
                    )
                    if U_orig is None:
                        logger.warning(f"[Edit Parallel] Skipping layer {layer.__class__.__name__} because original capture failed.")
                        continue

                    # 2) Capture U under each augmentation
                    U_aug_all = []
                    for i in range(num_aug):
                        X_aug = X_aug_all[i].to(self.get_device())
                        U_aug = self._capture_layer_inputs(
                            layer=layer,
                            data_obj=self.whole_data,
                            node_idx=None,
                            override_x=X_aug
                        )
                        if U_aug is None:
                            logger.warning(f"[Edit Parallel] Skipping layer {layer.__class__.__name__} due to missing augmentation capture.")
                            U_aug_all = []
                            break
                        U_aug_all.append(U_aug)
                    if not U_aug_all:
                        continue

                    U_aug_stacked = torch.cat(U_aug_all, dim=0)
                    U_orig_tiled = U_orig.repeat(num_aug, 1)

                    # 3) Compute D and solve S
                    D, U = self._compute_D_U(layer, U_orig_tiled, U_aug_stacked, num_augments=num_aug)
                    S = self._solve_shift(D, U, lam=lambda_reg)
                    pending_applies.append((layer, S))

                # Apply all shifts for this depth together
                for layer, S in pending_applies:
                    before = self._get_weight(layer).detach().clone()
                    self.edit_layer(layer=layer, S_star=S)
                    after = self._get_weight(layer).detach()
                    delta = torch.norm((after - before).flatten()).item()
                    logger.info(f"[Edit Parallel] Depth {depth} Layer {layer.__class__.__name__}: ||ΔW||₂ = {delta:.6e}, ||S||₂ = {torch.norm(S).item():.6e}")
        else:
            for layer in self.editable_layers:
                print(f"Editing layer: {layer.__class__.__name__}")
                
                # --- 1) Capture U (all nodes) under original input ---
                U_orig = self._capture_layer_inputs(
                    layer=layer,
                    data_obj=self.whole_data,
                    node_idx=None     # <- means keep ALL rows
                )  # shape: (N, d_in)
                if U_orig is None:
                    logger.warning(f"[Edit] Skipping layer {layer.__class__.__name__} because original capture failed.")
                    continue
                print(f"Shape of U_orig: {U_orig.shape}")

                # --- 2) For each augmentation, forward pass entire graph ---
                U_aug_all = []
            # prefer previously created aug matrices, else create on-the-fly
                if hasattr(self, "X_aug_all") and len(self.X_aug_all) > 0:
                    X_aug_all = self.X_aug_all
                else:
                    num_aug = int(self.config.get("leastsquares", {}).get("num_aug", 6))
                    X_aug_all = self.curate_steering_examples(K=num_aug)

                num_aug = len(X_aug_all)
                for i in range(num_aug):
                    X_aug = X_aug_all[i].to(self.get_device())
                    U_aug = self._capture_layer_inputs(
                        layer=layer,
                        data_obj=self.whole_data,
                        node_idx=None,      # <-- capture ALL nodes
                        override_x=X_aug
                    )  # shape: (N, d_in)
                    if U_aug is None:
                        logger.warning(f"[Edit] Skipping layer {layer.__class__.__name__} because augmentation capture failed.")
                        U_aug_all = []
                        break
                    print(f"Shape of U_aug: {U_aug.shape}")
                    U_aug_all.append(U_aug)
                if not U_aug_all:
                    continue

                U_aug_stacked = torch.cat(U_aug_all, dim=0)  # (K*N, d_in)
                U_orig_tiled = U_orig.repeat(num_aug, 1)     # (K*N, d_in)

                print(f"Shape of U_orig_tiled: {U_orig_tiled.shape}")
                print(f"Shape of U_aug_stacked: {U_aug_stacked.shape}")

                # --- 3) Compute D for ALL nodes ---
                D, U = self._compute_D_U(layer, U_orig_tiled, U_aug_stacked, num_augments=num_aug)
                # D: (K*N, d_out), U: (K*N, d_in)

                # --- 4) Solve least squares shift ---
                S = self._solve_shift(D, U, lam=lambda_reg)

                # --- 5) apply shift ---
                before = self._get_weight(layer).detach().clone()
                self.edit_layer(layer=layer, S_star=S)
                after  = self._get_weight(layer).detach()
                delta  = torch.norm((after - before).flatten()).item()
                logger.info(f"[Edit] Layer {layer.__class__.__name__}: ||ΔW||₂ = {delta:.6e}, ||S||₂ = {torch.norm(S).item():.6e}")


        # Optionally save summaries
        try:
            # model_before = deepcopy(self.model)  # no pre-edit snapshot available; keep same for interface
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
        results_after = seed_test(self.model, self.whole_data)
        fairness_metrics = self._evaluate_edit_effects()

        # Save metrics to JSON, similar to pretrain_gnn.py
        # try:
        from pipelines.seed_gnn.pretrain_gnn import get_split_class_counts
        
        num_features = self.num_features
        num_classes = self.num_classes
        split_class_counts = get_split_class_counts(self.whole_data)
        
        def compute_full_auc_pr(model_obj):
            model_obj.eval()
            with torch.no_grad():
                logits_full = prediction(model_obj, self.whole_data)
            
            def _auc_pr(mask):
                if mask is None: return float("nan")
                idx = mask.nonzero(as_tuple=False).view(-1)
                if idx.numel() == 0: return float("nan")
                y_true = self.whole_data.y[idx].cpu().numpy()
                probs = torch.softmax(logits_full[idx], dim=1).cpu().numpy()
                n_cls = probs.shape[1]
                if n_cls < 2: return float("nan")
                try:
                    if n_cls == 2:
                        return float(average_precision_score(y_true, probs[:, 1]))
                    y_oh = label_binarize(y_true, classes=np.arange(n_cls))
                    return float(average_precision_score(y_oh, probs, average="macro"))
                except Exception:
                    return float("nan")
            
            return {
                "train": _auc_pr(self.whole_data.train_mask),
                "val": _auc_pr(self.whole_data.val_mask),
                "test": _auc_pr(self.whole_data.test_mask)
            }

        # self.bef_edit_results is a MetricsDict (contains "overall", "metrics", "per_class")
        metrics_before = deepcopy(self.bef_edit_results.get("metrics")) if isinstance(self.bef_edit_results, dict) else None
        metrics_after = deepcopy(results_after.get("metrics"))

        auc_pr_before = compute_full_auc_pr(self.model_before)
        auc_pr_after = compute_full_auc_pr(self.model)

        if metrics_before:
            for split in ["train", "val", "test"]:
                metrics_before[split]["auc_pr"] = auc_pr_before[split]
        if metrics_after:
            for split in ["train", "val", "test"]:
                metrics_after[split]["auc_pr"] = auc_pr_after[split]
        
        # --- Sensitivity Metrics ---
        feature_name = getattr(self, "sensitive_feature", "AGE")
        fixed_vals = getattr(self, "fixed_sensitive_values", None)
        
        val_sens_before, test_sens_before = perturb_feature_and_measure_probs(
            self.model_before, self.whole_data, feature_name=feature_name,
            sensitive_feature_values=fixed_vals, compute_flips=True
        )
        val_sens_after, test_sens_after = perturb_feature_and_measure_probs(
            self.model, self.whole_data, feature_name=feature_name,
            sensitive_feature_values=fixed_vals, compute_flips=True
        )

        sensitivity_metrics = {
            "before": {
                "val": {
                    "mean_var": float(val_sens_before["VarProb"].mean()),
                    "mean_rel_var": float(val_sens_before["RelVarProb"].mean()),
                    "mean_flip_fraction": float(val_sens_before["FlipFraction"].mean())
                },
                "test": {
                    "mean_var": float(test_sens_before["VarProb"].mean()),
                    "mean_rel_var": float(test_sens_before["RelVarProb"].mean()),
                    "mean_flip_fraction": float(test_sens_before["FlipFraction"].mean())
                }
            },
            "after": {
                "val": {
                    "mean_var": float(val_sens_after["VarProb"].mean()),
                    "mean_rel_var": float(val_sens_after["RelVarProb"].mean()),
                    "mean_flip_fraction": float(val_sens_after["FlipFraction"].mean())
                },
                "test": {
                    "mean_var": float(test_sens_after["VarProb"].mean()),
                    "mean_rel_var": float(test_sens_after["RelVarProb"].mean()),
                    "mean_flip_fraction": float(test_sens_after["FlipFraction"].mean())
                }
            }
        }

        metrics_json = {
            "experiment": {
                "exp_desc": self.config["management"]["exp_desc"],
                "task": self.config["management"]["task"],
                "seed": SEED,
                "method": "leastsquares"
            },
            "data": {
                "dataset": self.config["eval_params"]["dataset"],
                "num_nodes": int(self.whole_data.num_nodes),
                "num_features": num_features,
                "num_classes": num_classes,
                "class_distribution": split_class_counts
            },
            "model": {
                "name": self.config["pipeline_params"]["model_name"],
                "architecture": self.config["pipeline_params"]["architecture"],
            },
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "sensitivity_metrics": sensitivity_metrics,
            "fairness_metrics": fairness_metrics,
            "edit_params": {
                "lambda_reg": lambda_reg,
                "num_targets": len(node_idx_2flip),
                "mu": self.mu
            }
        }
        
        ls_cfg = self.config["pipeline_params"].get("leastsquares", {})
        base_name = ls_cfg.get("metrics_save_name", "metrics_edit.json")
        if base_name.endswith(".json"):
            base_name = base_name[:-5]
        
        suffixes = []
        if getattr(self, "use_mean_steering_signal", False):
            suffixes.append("mean")
        if ls_cfg.get("sensitivity_based", False):
            suffixes.append("sensitivity")
        if ls_cfg.get("only_correct", False):
            suffixes.append("onlycorrect")
        
        # Add lambda and top_fraction to the name
        suffixes.append(f"lam{lambda_reg}")
        if hasattr(self, "top_fraction"):
            suffixes.append(f"top{self.top_fraction}")
        
        # Add seed to the name
        suffixes.append(f"seed{ls_cfg.get('seed', SEED)}")
        
        if suffixes:
            base_name += "_" + "_".join(suffixes)
        save_name = base_name + ".json"

        out_file = os.path.join(
            self.config["management"]["output_folder_dir"],
            save_name
        )
        
        with open(out_file, "w") as f:
            json.dump(metrics_json, f, indent=2)
        logger.info(f"Saved edit metrics to {out_file}")
        self.attach_edit_checkpoint_artifacts(out_file)
        
        # except Exception as e:
        #     logger.warning(f"Failed to save edit metrics JSON: {e}")

        bef_acc, val_acc, test_acc = results_after["overall"]
        raw_results = [[bef_acc, val_acc, test_acc]]
        return raw_results


    @torch.no_grad()
    def visualize_age_accuracy_before_edit(self, feature_name: str = "AGE"):
        """
        Before any editing, compute for each EXACT age value:
        - Overall accuracy per age
        - #correct and #incorrect nodes
        - Per-class accuracy per age

        Produces 3-panel plots for VAL and TEST splits (no CSV saving).
        """

        device = self.get_device()
        model = self.model.to(device).eval()
        data = self.whole_data.to(device)

        # Predictions before edit
        logits = prediction(model, data)
        preds = logits.argmax(dim=-1)
        y = data.y

        if not hasattr(data, "feature_names") or feature_name not in data.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in data.feature_names")

        age_col = data.feature_names.index(feature_name)
        ages = data.x[:, age_col].detach().cpu().numpy()
        preds_cpu = preds.detach().cpu().numpy()
        y_cpu = y.detach().cpu().numpy()
        num_classes = len(np.unique(y_cpu))

        os.makedirs(self.save_dir, exist_ok=True)

        def per_split(mask: torch.Tensor, split_tag: str, split_name: str):
            idx = mask.nonzero(as_tuple=False).view(-1).cpu().numpy()
            ages_split = ages[idx]
            preds_split = preds_cpu[idx]
            y_split = y_cpu[idx]

            unique_ages = np.unique(ages_split)

            rows = []
            per_class_dict = {}  # age → classwise accuracy list

            for a in unique_ages:
                m = ages_split == a
                total = m.sum()
                if total == 0:
                    continue

                correct = int((preds_split[m] == y_split[m]).sum())
                wrong = total - correct
                acc = correct / total

                # Per-class accuracy at this age
                cw = []
                for cls in range(num_classes):
                    mc = m & (y_split == cls)
                    tot_c = mc.sum()
                    if tot_c > 0:
                        cw_acc = (preds_split[mc] == y_split[mc]).sum() / tot_c
                    else:
                        cw_acc = np.nan
                    cw.append(cw_acc)
                per_class_dict[a] = cw

                rows.append([a, acc, correct, wrong, total])

            if not rows:
                logger.warning(f"No samples found for split={split_name} when computing age accuracy.")
                return

            df = pd.DataFrame(
                rows,
                columns=["Age", "Accuracy_Before", "NumCorrect_Before", "NumWrong_Before", "Total"],
            ).sort_values("Age")

            # ---- Plotting ----
            ages_sorted = df["Age"].values
            x = np.arange(len(ages_sorted))

            acc_vals = df["Accuracy_Before"].values * 100.0
            correct_vals = df["NumCorrect_Before"].values
            wrong_vals = df["NumWrong_Before"].values

            # Set dynamic step
            step = max(1, len(x) // 20)

            fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=True)

            # --- Panel 1: accuracy per age ---
            axes[0].bar(x, acc_vals)
            axes[0].set_ylabel("Accuracy (%)")
            axes[0].set_title(f"{split_name} accuracy by {feature_name}")
            axes[0].set_xticks(x[::step])
            axes[0].set_xticklabels(ages_sorted.astype(int)[::step], rotation=45, ha="right")
            axes[0].set_ylim(0, 100)

            # --- Panel 2: correct / incorrect counts ---
            axes[1].bar(x, correct_vals, label="Correct")
            axes[1].bar(x, wrong_vals, bottom=correct_vals, label="Incorrect")
            axes[1].set_ylabel("Count")
            axes[1].set_title(f"{split_name} Correct vs Incorrect")
            axes[1].set_xticks(x[::step])
            axes[1].set_xticklabels(ages_sorted.astype(int)[::step], rotation=45, ha="right")
            axes[1].legend()

            # --- Panel 3: per-class accuracy per age ---
            per_class_mat = np.array([per_class_dict[a] for a in ages_sorted]) * 100

            for ci in range(num_classes):
                axes[2].plot(x, per_class_mat[:, ci], marker="o", label=f"Class {ci}")

            axes[2].set_ylabel("Classwise Accuracy (%)")
            axes[2].set_title(f"{split_name} Classwise accuracy by Age")
            axes[2].set_xticks(x[::step])
            axes[2].set_xticklabels(ages_sorted.astype(int)[::step], rotation=45, ha="right")
            axes[2].set_ylim(0, 100)
            axes[2].legend()

            fig.tight_layout()
            fig.savefig(
                os.path.join(self.save_dir, f"age_exact_accuracy_before_{split_tag}.png"),
                dpi=150,
            )
            plt.close(fig)

        # Run for VAL and TEST
        per_split(data.val_mask, split_tag="val", split_name="Validation")
        per_split(data.test_mask, split_tag="test", split_name="Test")

        logger.info("Saved exact age-wise & classwise accuracy plots (before edit).")


    @torch.no_grad()
    def counterfactual_age_tests(
        self,
        feature_name: str = "AGE",
        young_threshold: float = 30.0,
        older_low: float = 40.0,
        older_high: float = 50.0,
    ):
        """
        VAL-ONLY counterfactual tests WITH CLASSWISE ACCURACY (no CSV saving)

        Test A: Perturb ALL young (< threshold) VAL nodes to random age in [older_low, older_high]
        Test B: Perturb ONLY young & wrong VAL nodes

        Produces:
            - Classwise accuracy before / after for both tests
            - Visual plots only (no CSV saving)
            - Printed summary
        """

        import numpy as np
        device = self.get_device()
        model = self.model.to(device).eval()
        data = self.whole_data.to(device)

        os.makedirs(self.save_dir, exist_ok=True)

        feat_idx = data.feature_names.index(feature_name)

        # ---- Baseline predictions ----
        logits_before = prediction(model, data)
        preds_before = logits_before.argmax(dim=-1)
        y = data.y

        # ---- VAL split ----
        val_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
        ages = data.x[:, feat_idx].detach().cpu().numpy()
        preds_cpu = preds_before.detach().cpu().numpy()
        y_cpu = y.detach().cpu().numpy()

        ages_val = ages[val_idx]
        preds_val_before = preds_cpu[val_idx]
        y_val = y_cpu[val_idx]

        # ---- Masks ----
        young_mask = ages_val < young_threshold
        wrong_mask = preds_val_before != y_val

        young_idx = val_idx[torch.from_numpy(young_mask)]
        young_wrong_idx = val_idx[torch.from_numpy(young_mask & wrong_mask)]

        num_young = len(young_idx)
        num_young_wrong = len(young_wrong_idx)

        print(f"[Counterfactual AGE tests] young_threshold={young_threshold}")
        print(f"  Young VAL nodes: {num_young}")
        print(f"  Young & wrong VAL nodes: {num_young_wrong}")

        # ---- Utility: classwise accuracy ----
        def classwise_accuracy(preds, labels, subset_idx):
            df = pd.DataFrame({"pred": preds[subset_idx], "true": labels[subset_idx]})
            results = []
            for cls in np.unique(labels[subset_idx]):
                mask = df["true"] == cls
                total = mask.sum()
                correct = (df.loc[mask, "pred"] == cls).sum()
                acc = correct / total if total > 0 else float("nan")
                results.append([cls, acc, correct, total])
            return pd.DataFrame(results, columns=["Class", "Acc", "Correct", "Total"])

        # ---- Baseline classwise ----
        cw_before_young = classwise_accuracy(preds_cpu, y_cpu, young_idx.cpu().numpy())
        cw_before_young_wrong = classwise_accuracy(preds_cpu, y_cpu, young_wrong_idx.cpu().numpy()) if num_young_wrong > 0 else None

        # ---- Test A: perturb ALL young ----
        original_x = data.x.clone()
        x_mod_all = original_x.clone()
        rand_age = torch.empty(data.x[:, feat_idx].shape, device=device).uniform_(older_low, older_high)
        x_mod_all[young_idx, feat_idx] = rand_age[young_idx]

        data.x = x_mod_all
        preds_after_all = prediction(model, data).argmax(dim=-1).detach().cpu().numpy()
        cw_after_young_all = classwise_accuracy(preds_after_all, y_cpu, young_idx.cpu().numpy())

        # ---- Test B: perturb ONLY young wrong ----
        data.x = original_x.clone()
        if num_young_wrong > 0:
            x_mod_wrong = original_x.clone()
            rand_age = torch.empty(data.x[:, feat_idx].shape, device=device).uniform_(older_low, older_high)
            x_mod_wrong[young_wrong_idx, feat_idx] = rand_age[young_wrong_idx]
            data.x = x_mod_wrong
            preds_after_wrong = prediction(model, data).argmax(dim=-1).detach().cpu().numpy()
            cw_after_young_wrong = classwise_accuracy(preds_after_wrong, y_cpu, young_wrong_idx.cpu().numpy())
        else:
            cw_after_young_wrong = None

        # Restore original
        data.x = original_x

        # ---- Plot comparison for ALL young ----
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(cw_before_young["Class"], cw_before_young["Acc"]*100, alpha=.6, label="Before")
        ax.bar(cw_after_young_all["Class"], cw_after_young_all["Acc"]*100, alpha=.6, label="After (ALL ↑)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Classwise accuracy change for VAL Young (<{young_threshold})")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(self.save_dir, "cw_young_all_val.png"), dpi=150)
        plt.close(fig)

        # ---- Plot for Young & Wrong ----
        if cw_before_young_wrong is not None:
            fig, ax = plt.subplots(figsize=(6,4))
            ax.bar(cw_before_young_wrong["Class"], cw_before_young_wrong["Acc"]*100, alpha=.6, label="Before")
            ax.bar(cw_after_young_wrong["Class"], cw_after_young_wrong["Acc"]*100, alpha=.6, label="After (WRONG ↑)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"Classwise accuracy change for Young & Wrong (<{young_threshold})")
            ax.legend()
            plt.tight_layout()
            fig.savefig(os.path.join(self.save_dir, "cw_young_wrong_val.png"), dpi=150)
            plt.close(fig)

        print("\n=== Counterfactual AGE classwise test summary ===")
        print("\n--- Young ALL ---")
        print(cw_before_young)
        print("\n--- Young ALL After ↑ ---")
        print(cw_after_young_all)

        if cw_before_young_wrong is not None:
            print("\n--- Young WRONG Before ---")
            print(cw_before_young_wrong)
            print("\n--- Young WRONG After ↑ ---")
            print(cw_after_young_wrong)

    @torch.no_grad()
    def quick_feature_shuffle_check(
        self,
        feature_name: str = None,
    ) -> dict:
        """
        Simple diagnostic:
        1) Compute VAL accuracy with original data.
        2) Shuffle the given feature among VAL nodes (permute values within VAL).
        3) Recompute VAL accuracy with shuffled feature.
        4) Report accuracy before/after and fraction of predictions that changed.

        Returns a dict with:
            {
            "feature_name": feature_name,
            "val_acc_before": ...,
            "val_acc_after": ...,
            "val_acc_drop": ...,
            "val_flip_fraction": ...
            }
        """
        device = self.get_device()
        model = self.model.to(device).eval()
        data = self.whole_data.to(device)

        # --- resolve which feature to test ---
        if feature_name is None:
            # try the configured sensitive_feature, else fall back to "fnlwgt"
            print("did not find a feature")
            feature_name = getattr(
                self,
                "sensitive_feature",
                self.config.get("pipeline_params", {}).get("sensitive_feature", "fnlwgt"),
            )

        if not hasattr(data, "feature_names") or feature_name not in data.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in data.feature_names")

        feat_idx = data.feature_names.index(feature_name)

        # --- baseline predictions ---
        logits_before = prediction(model, data)
        preds_before = logits_before.argmax(dim=-1)
        y = data.y

        val_mask = data.val_mask
        val_idx = val_mask.nonzero(as_tuple=False).view(-1)
        if val_idx.numel() == 0:
            raise RuntimeError("VAL mask is empty in quick_feature_shuffle_check.")

        # VAL accuracy before shuffling
        val_acc_before = (preds_before[val_idx] == y[val_idx]).float().mean().item()

        # --- shuffle feature values within VAL only ---
        x_orig = data.x.clone()
        feat_vals_val = x_orig[val_idx, feat_idx].clone()              # values to permute
        perm = torch.randperm(val_idx.numel(), device=device)
        shuffled_vals = feat_vals_val[perm]

        x_pert = x_orig.clone()
        x_pert[val_idx, feat_idx] = shuffled_vals

        # --- recompute predictions with shuffled feature ---
        original_x = data.x
        data.x = x_pert
        try:
            logits_after = prediction(model, data)
        finally:
            data.x = original_x  # restore no matter what

        preds_after = logits_after.argmax(dim=-1)

        # VAL accuracy after shuffling
        val_acc_after = (preds_after[val_idx] == y[val_idx]).float().mean().item()
        val_acc_drop = val_acc_before - val_acc_after

        # fraction of VAL nodes whose prediction changed
        val_flip_fraction = (preds_after[val_idx] != preds_before[val_idx]).float().mean().item()

        results = {
            "feature_name": feature_name,
            "val_acc_before": val_acc_before,
            "val_acc_after": val_acc_after,
            "val_acc_drop": val_acc_drop,
            "val_flip_fraction": val_flip_fraction,
        }

        logger.info(f"[Quick Shuffle Check] {feature_name} | "
                    f"VAL acc before={val_acc_before:.4f}, after={val_acc_after:.4f}, "
                    f"Δ={val_acc_drop:+.4f}, flip_frac={val_flip_fraction:.4f}")

        return results


    def run_editing_experiment(self, **kwargs):
        """
        Top-level pipeline: select targets + steering examples, then run edit_model.
        """
        ls_cfg = self.config.get("pipeline_params", {}).get("leastsquares", {})
        seed = ls_cfg.get("seed", SEED)
        set_seeds_all(seed)
        device = self.get_device()
        self.model = self.model.to(device)
        if self.train_data is not None:
            self.train_data = self.train_data.to(device)
        self.whole_data = self.whole_data.to(device)
        if self.use_mlp_linears:
            logger.info("[MLP] Enabling MLP linear edits; unfreezing and fine-tuning the MLP path.")
            if hasattr(self.model, 'mlp_freezed'):
                self.model.mlp_freezed = False
            if hasattr(self.model, 'freeze_module'):
                self.model.freeze_module(train=False)
            self.fine_tune_if_needed()
        print(f"Model: {self.model.__class__.__name__}, MLP freezed: {self.model.mlp_freezed}")
        print(f"========== Evaluating before edit ==========")
        # self.evaluate_before_edit()
        self.bef_edit_results = seed_test(self.model, self.whole_data)
        print(f"Before edit results: {self.bef_edit_results}")
        # bef_edit_ft_results = self.fine_tune_if_needed()
        # self.evaluate_before_edit()

        # self.quick_feature_shuffle_check(feature_name="fnlwgt")

        # --- NEW: age-wise accuracy visualization BEFORE any editing/fine-tuning ---
        # print("========== Age-wise accuracy BEFORE edit ==========")
        # self.visualize_age_accuracy_before_edit(feature_name="AGE")
        # print("========== Counterfactual AGE tests (before edit) ==========")
        # self.counterfactual_age_tests(feature_name="AGE", young_threshold=30.0, older_low=40.0, older_high=50.0)
        print(f"========== Evaluating before edit completed ==========")
        # 1) Select targets and create steering examples
        kwargs["only_linear"] = kwargs.get("only_linear", False)
        node_idx_2flip, flipped_label = self.select_edit_targets(**kwargs)

        # 2) Run the edit
        lambda_reg = float(kwargs.get('lambda_reg', self.lambda_reg))
        self.sanity_model_eval()
        logger.info("=== [Fisher Sanity Check] ===")
        # Unfreeze all layers temporarily
        for p in self.model.parameters():
            p.requires_grad = True

        for name, p in self.model.named_parameters():
            print(f"{name:40s} | requires_grad={p.requires_grad}")

        self.fisher_dict = self.compute_fisher_information(self.model, self.whole_data, self.representative_examples)
        raw_results = self.edit_model(node_idx_2flip=node_idx_2flip, flipped_label=flipped_label, lambda_reg=lambda_reg)

        return raw_results,None 





