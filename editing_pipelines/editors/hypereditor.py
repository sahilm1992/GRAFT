"""
Hyper Editor network implementation.
"""

import logging
from copy import deepcopy
from typing import List, Callable, Optional, Tuple, Dict, Any

import torch
from tqdm import tqdm

from editing_pipelines.editors.base import BaseEditor
from editing_pipelines.utils.model_io import get_optimizer
from editing_pipelines.utils.editing_ops import edit
from editing_pipelines.utils.train_eval import test, success_rate
from editing_pipelines.utils.selection import select_edit_targets_by_strategy
from editing_pipelines.utils.results import save_misclassifications_txt, save_misclassification_summary_txt
from editing_pipelines.utils.visualization import plot_misclassification_by_attributes_before_after, plot_targeted_edits_distribution



from main_utils import set_seeds_all

logger = logging.getLogger("main")

def age_bias_rule(
    data,
    preds: torch.Tensor,
    y_true: torch.Tensor,
    feature_vals: torch.Tensor,
    feature_name: str = "AGE",
    low_threshold: float = 20,
    high_threshold: float = 25,
    **kwargs
) -> torch.Tensor:
    """
    Identify biased misclassifications w.r.t age:
      - AGE < low_threshold and (true label = working=1, pred = not working=0)
      - AGE > high_threshold and (true label = not working=0, pred = working=1)
    """
    cond_young = (feature_vals < low_threshold) & (y_true == 1) & (preds == 0)
    cond_old = (feature_vals > high_threshold) & (y_true == 0) & (preds == 1)
    mask = cond_young | cond_old
    logger.info(
        f"[Age Bias Rule] Found {mask.sum().item()} biased samples "
        f"({feature_name}<{low_threshold} or >{high_threshold})"
    )
    return mask


class HyperNetwork(nn.Module):
    """
    MALMEN-style hypernetwork for GNN editing.
    Takes (u_j, grad_v_j) -> d_j
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, u, grad_v):
        x = torch.cat([u, grad_v], dim=-1)
        return self.net(x)


class HyperEditor(BaseEditor):

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hypernet = None
        self.optimizer = None
        logger.info("Initialized MALMEN-GNN HyperEditor")


    def initialize_hypernetwork(self, u_dim: int, v_dim: int, hyper_hidden_dim: int = 256, lr: float = 1e-4):
        # cant call in init function since u_dim and v_dim are not available yet
        self.hypernet = HyperNetwork(u_dim, v_dim, hyper_hidden_dim).to(self.config["device"])
        self.optimizer = optim.Adam(self.hypernet.parameters(), lr=lr)
        logger.info(f"Initialized hypernetwork: u_dim={u_dim}, v_dim={v_dim}, hidden={hidden_dim}, lr={lr}")


    # def select_edit_targets(self, num_targets: int = None):
    #     return super().select_edit_targets(num_targets)

    def train_hypernetwork(
        self,
        num_epochs: int = 10,
        lambda_reg: float = 1e-3,
        num_targets: int = None,
        select_kwargs: Dict[str, Any] = None,
    ):
        """
        Full MALMEN meta-training loop for GNN editing.
        """

        self.load_model_and_data()
        device = self.config["device"]
        model = self.model.to(device)
        model.eval()

        # 1. Get initial edit targets (nodes)
        node_idx, flipped_labels = self.select_edit_targets(num_targets=num_targets, **(select_kwargs or {}))
        logger.info(f"Meta-training with {node_idx.numel()} target nodes")

        # 2. Determine feature dimensions from model
        # forward pass to capture activation and output dims
        with torch.no_grad():
            out = model(self.whole_data.to(device))
        u_dim = model.convs[0].in_channels
        v_dim = model.convs[-1].out_channels

        # 3. Initialize hypernetwork if not already done
        if self.hypernet is None:
            self.initialize_hypernetwork(u_dim, v_dim)


        # Begin training

        for epoch in range(num_epochs):
            logger.info(f"[Epoch {epoch+1}/{num_epochs}]")

            # Inner loop: capture activations + grads for edit nodes
            model.zero_grad()
            for p in model.parameters():
                p.requires_grad = True

            # capture activations (keys) and grads (values)
            acts, grads = self._collect_edit_signals(model, node_idx, flipped_labels)
            D, U = self._compute_d_and_u(acts, grads)  # D (d', n), U (d, n)

            # Compute normal-equation edit
            S_star = self._solve_normal_equation(D, U, lambda_reg)
            delta_norm = torch.norm(S_star)

            # Apply edit (no grad flows into model)
            with torch.no_grad():
                self._apply_update_to_model(model, S_star)

            # Outer loop: compute meta-loss on edited nodes
            model.eval()
            with torch.no_grad():
                out = model(self.whole_data.to(device))
                logits = out[node_idx]
            loss_edit = nn.CrossEntropyLoss()(logits, flipped_labels.to(device))
            loss_reg = lambda_reg * delta_norm
            loss_meta = loss_edit + loss_reg

            # Optimize hypernetwork parameters
            self.optimizer.zero_grad()
            loss_meta.backward()
            self.optimizer.step()

            logger.info(f"Epoch {epoch+1}: L_edit={loss_edit:.4f}, L_reg={loss_reg:.4f}")

        logger.info("✅ Hypernetwork meta-training completed.")

    def _collect_edit_signals(self, model, node_idx, flipped_labels):
        """Run forward+backward on edit nodes and collect per-layer activations + grads"""
        device = self.config["device"]
        data = self.whole_data.to(device)
        acts, grads = {}, {}

        # register hooks on final GNN layer
        target_layer = model.convs[-1]
        def fwd_hook(mod, inp, out):
            acts["u"] = inp[0].detach()
        def bwd_hook(mod, gin, gout):
            grads["v"] = gout[0].detach()
        fh = target_layer.register_forward_hook(fwd_hook)
        bh = target_layer.register_full_backward_hook(bwd_hook)

        model.train()
        logits = model(data)
        loss = nn.CrossEntropyLoss()(logits[node_idx], flipped_labels.to(device))
        loss.backward()

        fh.remove(); bh.remove()

        u = acts["u"][node_idx]      # (k, d)
        grad_v = grads["v"][node_idx]  # (k, d')
        return u, grad_v

    def _compute_d_and_u(self, u, grad_v):
        """Compute D and U matrices via the hypernetwork"""
        D_cols, U_cols = [], []
        for j in range(u.shape[0]):
            u_j = u[j]
            grad_v_j = grad_v[j]
            d_j = self.hypernet(u_j, grad_v_j)
            D_cols.append(d_j)
            U_cols.append(u_j)
        D = torch.stack(D_cols, dim=1)
        U = torch.stack(U_cols, dim=1)
        return D, U

    def _solve_normal_equation(self, D, U, lambda_reg):
        """S* = D Uᵀ (U Uᵀ + λI)⁻¹"""
        UUT = U @ U.t()
        eye = torch.eye(UUT.shape[0], device=U.device)
        inv = torch.inverse(UUT + lambda_reg * eye)
        return (D @ U.t()) @ inv

    def _apply_update_to_model(self, model, S_star):
        """Add low-rank delta to final conv layer weights"""
        layer = model.convs[-1]
        with torch.no_grad():
            if hasattr(layer, "lin"):
                W = layer.lin.weight
            else:
                W = layer.weight
            if S_star.shape == W.shape:
                W.add_(S_star.to(W.device))
            else:
                logger.warning(f"Shape mismatch in update: {S_star.shape} vs {W.shape}")

    def edit_model(self, node_idx_2flip: torch.Tensor, flipped_label: torch.Tensor, max_num_step: int) -> List[List[Any]]:
        # return super().edit_model(node_idx_2flip, flipped_label, max_num_step)
        return None

    #need to import the prediction function for this, 
    def select_edit_targets(
        self,
        rule_fn: Optional[Callable] = None,
        feature_name: str = "AGE",
        num_targets: Optional[int] = None,
        **kwargs
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select edit targets using a customizable bias rule.
        Falls back to age_bias_rule() if no custom rule_fn is provided.

        Args:
            rule_fn: Callable returning a BoolTensor mask for val nodes.
            feature_name: Feature to use (default: 'AGE').
            num_targets: Max number of selected nodes (optional cap).
            **kwargs: Extra args (e.g., thresholds, parameters for rule_fn).
        """
        data = self.whole_data
        model = self.model.eval()
        device = next(model.parameters()).device

        # --- 1. Get model predictions ---
        logits = prediction(model, data)
        preds = logits.argmax(dim=-1)
        y_true = data.y

        # --- 2. Extract validation subset ---
        val_mask = data.val_mask
        val_idx = val_mask.nonzero(as_tuple=False).view(-1)
        preds_val = preds[val_mask]
        y_true_val = y_true[val_mask]

        # --- 3. Get feature values (e.g. AGE) ---
        if not hasattr(data, "feature_names") or feature_name not in data.feature_names:
            raise ValueError(f"Feature '{feature_name}' not found in data.feature_names")
        feat_idx = data.feature_names.index(feature_name)
        feature_vals = data.x[:, feat_idx][val_mask]

        # --- 4. Choose rule function ---
        rule_to_use = rule_fn or age_bias_rule

        # --- 5. Apply rule to get boolean mask ---
        selected_mask = rule_to_use(
            data=data,
            preds=preds_val,
            y_true=y_true_val,
            feature_vals=feature_vals,
            feature_name=feature_name,
            **kwargs
        )
        if not torch.is_tensor(selected_mask) or selected_mask.dtype != torch.bool:
            raise TypeError("rule_fn must return a torch.BoolTensor mask")

        # --- 6. Collect indices ---
        selected_indices = val_idx[selected_mask]
        logger.info(f"Total selected edit targets: {selected_indices.numel()}")

        # --- 7. Limit number of targets if specified ---
        if num_targets is not None and selected_indices.numel() > num_targets:
            logger.info(f"Limiting to {num_targets} edit targets")
            selected_indices = selected_indices[:num_targets]

        # --- 8. True labels (used for correction) ---
        flipped_labels = y_true[selected_indices]

        return selected_indices.to(device), flipped_labels.to(device)