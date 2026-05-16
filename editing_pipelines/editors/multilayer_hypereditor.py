"""
Hyper Editor network implementation.
"""

import logging
from copy import deepcopy
from typing import List, Callable, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from editing_pipelines.editors.base import BaseEditor
from editing_pipelines.utils.model_io import get_optimizer
from editing_pipelines.utils.editing_ops import edit
from editing_pipelines.utils.train_eval import test, success_rate
from editing_pipelines.utils.selection import select_edit_targets_by_strategy
from editing_pipelines.utils.results import save_misclassifications_txt, save_misclassification_summary_txt
from editing_pipelines.utils.visualization import (
    plot_misclassification_by_attributes_before_after,
    plot_targeted_edits_distribution,
    plot_validation_correct_confidence_histogram,
    visualize_validation,
)

# Import from seed-gnn directory
import sys
sys.path.append('/home/model_editing/gnn-editing-exploration/seed-gnn')
from main_utils import set_seeds_all
from edit_gnn.utils import prediction, grab_input

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
    Small MLP mapping (u_j, grad_v_j) -> d_j (value-difference).
    Each target layer will have its own HyperNetwork instance.
    """
    def __init__(self, u_dim: int, v_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(u_dim + v_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, v_dim)
        )

    def forward(self, u: torch.Tensor, grad_v: torch.Tensor) -> torch.Tensor:
        # u: (k, u_dim) ; grad_v: (k, v_dim) -> returns (k, v_dim)
        x = torch.cat([u, grad_v], dim=-1)
        return self.net(x)

class HyperEditor(BaseEditor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hypernets = {}       # layer_name -> HyperNetwork
        self.optimizer = None     # optimizer over all hypernets
        self.num_epochs = config.get("num_epochs", 10)
        self.lambda_reg = config.get("lambda_reg", 1e-3)
        self.hidden = config.get("hidden", 128)
        self.lr = config.get("lr", 1e-4)
        self.num_targets = config.get("num_targets", None)
        self.select_kwargs = config.get("select_kwargs", None)
        self.device = config.get("device", "cpu")
        self.lambda_loc = config.get("lambda_loc", 1e10)
        
        logger.info("Initialized Multi-layer MALMEN HyperEditor")

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
        
        model = self.model.eval()
        device = next(model.parameters()).device
        data = self.whole_data.to(device)

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
    # -------------------------
    # utility: which layers to edit
    # -------------------------
    def _get_target_layers(self) -> List[Tuple[str, nn.Module]]:
        """
        Return list of (name, module) pairs for all editable GNN layers.
        Supports nested architectures like GCN_MLP, GIN_MLP, GraphSAGE_MLP, etc.
        Only picks layers of type GCNConv, SAGEConv, GINConv, etc.
        """
        from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GraphConv

        target_types = (GCNConv, SAGEConv, GINConv, GraphConv)
        layers = []

        # recursive named_modules() catches nested layers
        for name, mod in self.model.named_modules():
            if isinstance(mod, target_types):
                layers.append((name, mod))

        if len(layers) == 0:
            # fallback: if model has a top-level conv list
            if hasattr(self.model, "convs"):
                for i, mod in enumerate(self.model.convs):
                    layers.append((f"convs.{i}", mod))
            elif hasattr(self.model, "GCN") and hasattr(self.model.GCN, "convs"):
                for i, mod in enumerate(self.model.GCN.convs):
                    layers.append((f"GCN.convs.{i}", mod))

        # print("Detected target layers:")
        # for lname, layer in layers:
        #     print(f"  {lname}: {layer}")

        return layers


    # -------------------------
    # initialize hypernets for all target layers
    # -------------------------
    def _ensure_hypernets(self, u_dims: Dict[str, int], v_dims: Dict[str, int], hidden: int = 128, lr: float = 1e-4, device: str = "cpu"):
        """Create (or keep) HyperNetworks for layers in target list. Also create optimizer."""
        new = {}
        params = []
        print(f"target layers: {self._get_target_layers()}")
        for lname, _ in self._get_target_layers():
            if lname in self.hypernets:
                new[lname] = self.hypernets[lname]
            else:
                u_dim = u_dims[lname]
                v_dim = v_dims[lname]
                hn = HyperNetwork(u_dim, v_dim, hidden).to(device)
                new[lname] = hn
            params += list(new[lname].parameters())
        # print(params)
        self.hypernets = new
        self.optimizer = optim.Adam(params, lr=lr)
        logger.info(f"Hypernets created for layers: {list(self.hypernets.keys())}")

    # -------------------------
    # collect inner signals (act keys and value-grads) for selected nodes
    # -------------------------
    def _collect_inner_signals(self, model: nn.Module, data: torch.Tensor, node_idx: torch.Tensor, device: str):
        """
        Runs forward and backward to collect:
            - U_selected[lname]: activations (keys) for selected nodes for each layer -> shape (d, n)
            - grad_v_selected[lname]: grads w.r.t layer outputs for selected nodes -> shape (d', n)
        Also captures U_all[lname]: activations for all nodes at each target layer -> shape (d, N)
        Returns dicts keyed by layer name.
        IMPORTANT: returned tensors are detached (no grad back to model).
        """
        model.train()
        for p in model.parameters():
            p.requires_grad = True

        U_selected = {}   # lname -> (d, n) (tensor)
        GradV_selected = {}  # lname -> (d', n)
        U_all = {}        # lname -> (d, N)

        hooks = []

        # forward hooks to capture activations for all nodes
        def make_fwd_hook(lname):
            def fwd_hook(module, inp, out):
                # inp[0] typically is node features at that layer: (N, d)
                if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], torch.Tensor):
                    act = inp[0]  # (N, d)
                elif isinstance(inp, torch.Tensor):
                    act = inp
                else:
                    # fallback: try using output 'out' as activation
                    act = out.detach()
                U_all[lname] = act.detach().cpu()
                # also store activation for selected nodes
                U_selected[lname] = act[node_idx].detach().cpu().t()  # make (d, n) for convenience
            return fwd_hook

        # backward hooks to capture grad_output for selected nodes
        def make_bwd_hook(lname):
            def bwd_hook(module, grad_input, grad_output):
                # grad_output[0] is grad wrt output (N, d_out)
                g = grad_output[0] if isinstance(grad_output, tuple) and len(grad_output) > 0 else grad_output
                if isinstance(g, torch.Tensor):
                    GradV_selected[lname] = g[node_idx].detach().cpu().t()  # (d', n)
            return bwd_hook

        # register hooks for all target layers
        for lname, module in self._get_target_layers():
            fh = module.register_forward_hook(make_fwd_hook(lname))
            bh = module.register_full_backward_hook(make_bwd_hook(lname))
            hooks += [fh, bh]

        # run forward and backward for loss on selected nodes only
        # logits = prediction(model, self.whole_data.to(device))
        input = grab_input(self.whole_data.to(device))
        logits = model(**input)
        labels = self.whole_data.y.to(device)
        loss = nn.CrossEntropyLoss()(logits[node_idx], labels[node_idx])
        loss.backward()

        # remove hooks
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

        # detach and move to device when used later; we return CPU tensors (safe)
        return U_all, U_selected, GradV_selected

    # -------------------------
    # compute D (d', n) per layer using hypernetworks
    # -------------------------
    def _compute_D_for_layer(self, lname: str, u_sel_cpu: torch.Tensor, grad_v_sel_cpu: torch.Tensor, device: str):
        """
        u_sel_cpu: (d, n) on CPU
        grad_v_sel_cpu: (d', n) on CPU
        returns D: (d', n) on device (requires_grad = True via hypernet)
        """
        hn = self.hypernets[lname]
        # convert to (n, d) shape for hypernet input
        u_sel = u_sel_cpu.t().to(device)       # (n, d)
        grad_v_sel = grad_v_sel_cpu.t().to(device)  # (n, d')
        D_rows = hn(u_sel, grad_v_sel)         # (n, d') -> but we want (d', n)
        D = D_rows.t()                         # (d', n)
        return D

    # -------------------------
    # compute delta_v_all for a layer: vectorized form
    # -------------------------
    def _compute_delta_v_all(self, D: torch.Tensor, U_sel_cpu: torch.Tensor, U_all_cpu: torch.Tensor, lambda_reg: float, device: str):
        """
        D: (d', n) (on device, depends on hypernet)
        U_sel_cpu: (d, n) on CPU (keys for selected nodes)
        U_all_cpu: (N, d) on CPU (!) -- note this was captured as (N, d) earlier; convert to (d, N)
        returns delta_V_all: (N, d') on device (to add to layer outputs)
        Derived vectorized expression:
            A = (U_sel U_sel^T + λ I)^(-1)   # (d, d)
            W = U_sel^T @ (A @ U_all)        # (n, N)
            delta_v_all = D @ W             # (d', N) -> transpose to (N, d')
        """
        # convert CPU captures -> device tensors with shapes needed
        U_sel = U_sel_cpu.to(device)          # (d, n)
        U_all = U_all_cpu.t().to(device)      # originally captured as (N, d) -> transpose to (d, N)
        # compute A = (U_sel @ U_sel^T + λ I)^-1
        uuT = U_sel @ U_sel.t()               # (d, d)
        eye = torch.eye(uuT.shape[0], device=device, dtype=uuT.dtype)
        try:
            A = torch.inverse(uuT + lambda_reg * eye)  # (d, d)
        except RuntimeError:
            logger.warning("uuT is not invertible, using pinverse")
            A = torch.pinverse(uuT + lambda_reg * eye)
        # W = U_sel.T @ (A @ U_all) -> (n, N)
        W = U_sel.t() @ (A @ U_all)           # (n, N)
        # delta_v_all = D @ W -> (d', N)
        delta = D @ W                         # (d', N)
        delta_T = delta.t()                   # (N, d')
        return delta_T

    # -------------------------
    # register forward hooks that add layer-specific deltas to outputs
    # -------------------------
    def _make_add_delta_hook(self, lname: str, delta_all: torch.Tensor, device: str):
        """
        delta_all: (N, d') on device. We'll add this to the forward output (N, d').
        """
        def hook(module, inp, out):
            # out is (N, d') ; add delta to it
            if not isinstance(out, torch.Tensor):
                # common case: out is Tensor
                return out
            return out + delta_all.to(out.device)
        return hook

    # -------------------------
    # inference: apply learned hypernets to perform an edit (multi-layer)
    # -------------------------
    def edit_model_multilayer(self, node_idx_2flip: torch.Tensor, flipped_label: torch.Tensor, lambda_reg: float = 1e-3, device: str = None):
        """
        One-shot inference edit using trained hypernets. This will compute D per layer, compute S*, and
        apply *in-place* updates to model parameters (useful for permanent edits).
        """
        if device is None:
            device = self.config.get("device", "cpu")
        self.load_model_and_data()
        model = self.model.to(device)
        model.train()

        # collect inner signals (we need U_sel and grad_v_sel for selected nodes)
        U_all_dict, U_sel_dict, GradV_sel_dict = self._collect_inner_signals(model, self.whole_data, node_idx_2flip.to(device), device)

        # for each layer, compute D via hypernet and form S*, then apply to module weights in-place
        for lname, module in self._get_target_layers():
            if lname not in U_sel_dict or lname not in GradV_sel_dict:
                continue
            U_sel_cpu = U_sel_dict[lname]           # (d, n)
            grad_v_sel_cpu = GradV_sel_dict[lname]  # (d', n)
            D = self._compute_D_for_layer(lname, U_sel_cpu, grad_v_sel_cpu, device)  # (d', n)

            # compute normal-equation S* = D U^T (U U^T + λI)^-1
            U_sel = U_sel_cpu.to(device)
            uuT = U_sel @ U_sel.t()
            eye = torch.eye(uuT.shape[0], device=device, dtype=uuT.dtype)
            try:
                inv = torch.inverse(uuT + lambda_reg * eye)
            except RuntimeError:
                inv = torch.pinverse(uuT + lambda_reg * eye)
            S_star = (D @ U_sel.t()) @ inv  # (d', d)

            # apply S_star to module weights (try common weight names)
            applied = False
            with torch.no_grad():
                if hasattr(module, "lin") and hasattr(module.lin, "weight"):
                    W = module.lin.weight
                    if W.shape == S_star.shape:
                        W.add_(S_star.to(W.device))
                        applied = True
                elif hasattr(module, "weight"):
                    W = module.weight
                    if W.shape == S_star.shape:
                        W.add_(S_star.to(W.device))
                        applied = True
                else:
                    for pname, p in module.named_parameters():
                        if p.shape == S_star.shape:
                            p.add_(S_star.to(p.device))
                            applied = True
                            break
            if not applied:
                logger.warning(f"Could not apply S* to layer {lname}; S* shape {S_star.shape}")

        # final evaluation
        self.model.eval()
        try:
            bef, val, test_acc = test(self.model, self.whole_data)
        except Exception:
            # fallback simple eval
            data_eval = self.whole_data.to(device)
            with torch.no_grad():
                out = self.model(data_eval)
                if isinstance(out, dict) and "logits" in out:
                    logits = out["logits"]
                else:
                    logits = out
                preds = logits.argmax(dim=-1)
                accs = []
                for mask_name in ["train_mask", "val_mask", "test_mask"]:
                    if hasattr(data_eval, mask_name):
                        mask = getattr(data_eval, mask_name)
                        if mask.sum() > 0:
                            acc = (preds[mask] == data_eval.y[mask]).float().mean().item()
                        else:
                            acc = float("nan")
                    else:
                        acc = float("nan")
                    accs.append(acc)
                bef, val, test_acc = accs

        logger.info(f"Post-edit eval (train,val,test): {bef:.4f}, {val:.4f}, {test_acc:.4f}")
        return

    def train_hypernetwork_multilayer_malmen(
        self,
        num_epochs: int = 10,
        lambda_reg: float = 1e-3,
        hidden: int = 128,
        lr: float = 1e-4,
        num_targets: int = None,
        select_kwargs: Dict[str, Any] = None,
        device: str = None,
    ):
        """
        MALMEN exact two-stage training (Algorithm 2) adapted to multi-layer GNNs.
        - Stage 0: Editor inference (compute D, U_sel, U_all, S*)
        - Stage 1: Backprop L_meta on edited LM -> cache grad_w_tilde for each layer
        - Stage 2: Compute grad_d via theorem and backprop proxy loss into hypernets

        Uses select_edit_targets() to pick node indices + labels (these supply L_meta).
        """

        if device is None:
            device = self.config.get("device", "cpu")
        self.load_model_and_data()
        model = self.model.to(device)
        model.eval()

        # select edit nodes (these supply y|x for L_meta per your instruction)
        node_idx_cpu, flipped_labels_cpu = self.select_edit_targets(num_targets=num_targets, **(select_kwargs or {}))
        if node_idx_cpu.numel() == 0:
            raise ValueError("No edit nodes selected.")
        node_idx = node_idx_cpu.to(device)
        flipped_labels = flipped_labels_cpu.to(device)
        logger.info(f"Training hypernets (MALMEN two-stage) on {node_idx.numel()} nodes")

        # 1) Ensure we have hypernets for each target layer (initialize if needed)
        # We'll derive u_dim and v_dim by a single inner signal collection
        U_all_dict, U_sel_dict, GradV_sel_dict = self._collect_inner_signals(model, self.whole_data, node_idx, device)
        u_dims, v_dims = {}, {}
        for lname, _ in self._get_target_layers():
            if lname in U_sel_dict and lname in GradV_sel_dict:
                u_dims[lname] = U_sel_dict[lname].shape[0]
                v_dims[lname] = GradV_sel_dict[lname].shape[0]
        self._ensure_hypernets(u_dims=u_dims, v_dims=v_dims, hidden=hidden, lr=lr, device=device)

        # optimizer exists from _ensure_hypernets
        optimizer = self.optimizer

        # main epochs
        for epoch in range(num_epochs):
            logger.info(f"[MALMEN Epoch {epoch+1}/{num_epochs}]")
            model.train()

            # ----------------------
            # STAGE 0: Editor Inference (compute D, U_sel, U_all, S*) and apply S* to model
            # ----------------------
            # We'll compute D (via hypernets), S_star per layer, and apply updates in-place,
            # but *do not* keep autograd connections from LM -> hypernets for Stage1.
            per_layer_info = {}  # lname -> dict with D (d',n, requires_grad), U_sel_cpu, U_all_cpu, S_star (device)
            orig_weights = {}    # to restore after stage2

            # capture inner signals again (we need up-to-date activations & grads)
            U_all_dict, U_sel_dict, GradV_sel_dict = self._collect_inner_signals(model, self.whole_data, node_idx, device)

            # compute D (hypernet outputs) for each layer (keep requires_grad True for hypernet training)
            
            for lname, module in self._get_target_layers():
                if lname not in U_sel_dict or lname not in GradV_sel_dict or lname not in U_all_dict:
                    continue
                U_sel_cpu = U_sel_dict[lname]           # (d, n) cpu
                grad_v_sel_cpu = GradV_sel_dict[lname]  # (d', n) cpu
                U_all_cpu = U_all_dict[lname]           # (N, d) cpu

                # D depends on hypernet -> must be computed with hypernets active (params require_grad True)
                D = self._compute_D_for_layer(lname, U_sel_cpu, grad_v_sel_cpu, device)  # (d', n) on device, requires_grad True
                # compute S* = D U^T (U U^T + λI)^-1  (on device)
                U_sel = U_sel_cpu.to(device)  # (d, n)
                uuT = U_sel @ U_sel.t()
                eye = torch.eye(uuT.shape[0], device=device, dtype=uuT.dtype)
                try:
                    inv = torch.inverse(uuT + lambda_reg * eye)
                except RuntimeError:
                    inv = torch.pinverse(uuT + lambda_reg * eye)
                S_star = (D @ U_sel.t()) @ inv  # (d', d) on device

                # save original weight tensor (copy) into orig_weights so we can restore after Stage 2
                # then add S* for param shift
                # Part of Algo 2, ln 1 - Inference stage
                with torch.no_grad():
                    applied = False
                    if hasattr(module, "lin") and hasattr(module.lin, "weight"):
                        orig_weights[lname] = module.lin.weight.detach().clone()
                        module.lin.weight.add_(S_star.to(module.lin.weight.device))
                        applied = True
                    elif hasattr(module, "weight"):
                        orig_weights[lname] = module.weight.detach().clone()
                        module.weight.add_(S_star.to(module.weight.device))
                        applied = True
                    else:
                        # attempt to locate matching parameter
                        for pname, p in module.named_parameters():
                            if p.shape == S_star.shape:
                                orig_weights[(lname, pname)] = p.detach().clone()
                                p.add_(S_star.to(p.device))
                                applied = True
                                break
                    if not applied:
                        logger.warning(f"[Stage0] Could not apply S* to layer {lname}; S* shape {S_star.shape}")

                # Algo 2, ln 2 - cache D, S_star, U_sel, U_all for each layer
                # cache D instead of grad_v
                per_layer_info[lname] = {
                    "D": D,  # requires_grad True, on device
                    "U_sel_cpu": U_sel_cpu,  # cpu
                    "U_all_cpu": U_all_cpu,  # cpu
                    "S_star": S_star.detach()  # store S* (detached)
                }

            # ----------------------
            # STAGE 1: Backprop L_meta on edited LM (hypernets frozen) -> cache grad_w_tilde per layer
            # ----------------------
            # Freeze hypernets; enable gradients for LM
            for hn in self.hypernets.values():
                for p in hn.parameters():
                    p.requires_grad = False
            for p in model.parameters():
                p.requires_grad = True
                if p.grad is not None:
                    p.grad.zero_()

            # Compute L_meta on edited model (per your instruction: use edit nodes for supervision)
            model.eval()
            with torch.enable_grad():
                input = grab_input(self.whole_data.to(device))
                logits = model(**input)

                # Algo 2, ln 3 - compute L_meta
                # ---- Compute L_gen (same as current CE loss) ----
                loss_gen = nn.CrossEntropyLoss()(logits[node_idx], flipped_labels)

                # ---- Compute L_loc (KL divergence on unaffected nodes) ----
                # define unaffected = complement of node_idx
                N = self.whole_data.num_nodes
                mask_all = torch.arange(N, device=device)
                unaffected_idx = mask_all[~torch.isin(mask_all, node_idx)]
                if unaffected_idx.numel() > 0:
                    # get unedited gnn predictions (p_W)
                    model.eval()
                    with torch.no_grad():
                        input = grab_input(self.whole_data.to(device))
                        logits_unedited = model(**input)
                        # if isinstance(logits_unedited, dict):
                        #     logits_unedited = logits_unedited["logits"]
                        p_W = torch.softmax(logits_unedited[unaffected_idx], dim=-1)
                    # get edited gnn predictions (p_Wtilde)
                    logits_edited = logits[unaffected_idx]
                    log_p_Wtilde = torch.log_softmax(logits_edited, dim=-1)
                    # KL(p_W || p_Wtilde)
                    loss_loc = torch.nn.functional.kl_div(log_p_Wtilde, p_W, reduction="batchmean")
                else:
                    loss_loc = torch.tensor(0.0, device=device)

                # ---- Total meta-loss ----
                lambda_loc = self.config["pipeline_params"].get("lambda_loc", 1.0)
                loss_meta = loss_gen + lambda_loc * loss_loc

                # loss_meta = nn.CrossEntropyLoss()(logits[node_idx], flipped_labels)  # use only selected nodes as per your instruction

                # Backprop only on gnn (hypernets are frozen)
                # Algo 2, ln 4 - Backprop L_meta on edited gnn
                loss_meta.backward()

            # Algo 2, ln 5 - cache gradients w.r.t. edited weights ∇_{W~} L_meta for each target layer
            grad_Wtilde = {}  # lname -> tensor (d', d)
            for lname, module in self._get_target_layers():
                # Try to find gradient in module params (lin.weight or weight)
                found = False
                if hasattr(module, "lin") and hasattr(module.lin, "weight"):
                    g = module.lin.weight.grad
                    if g is not None:
                        grad_Wtilde[lname] = g.detach().clone().to(device)
                        found = True
                elif hasattr(module, "weight") and module.weight.grad is not None:
                    grad_Wtilde[lname] = module.weight.grad.detach().clone().to(device)
                    found = True
                else:
                    # fallback: search parameters
                    for pname, p in module.named_parameters():
                        if p.grad is not None:
                            key = (lname, pname)
                            grad_Wtilde[key] = p.grad.detach().clone().to(device)
                            found = True
                if not found:
                    logger.warning(f"[Stage1] No grad cached for layer {lname}")

            # zero out gradients on gnn (we don't update gnn here)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            # ----------------------
            # STAGE 2: Compute ∇_{d} L_meta using Theorem 1 and backprop proxy loss into hypernets
            # ----------------------
            # Enable hypernet grads, freeze gnn params
            for hn in self.hypernets.values():
                for p in hn.parameters():
                    p.requires_grad = True
            for p in model.parameters():
                p.requires_grad = False

            # Build proxy loss: sum_{layers} sum_{j} (grad_d[:,j]^T * d[:,j])
            # Algo 2, ln 6-8
            proxy_terms = []
            for lname, info in per_layer_info.items():
                if lname not in grad_Wtilde:
                    continue
                D = info["D"]                # (d', n) on device, requires_grad True
                U_sel_cpu = info["U_sel_cpu"]  # (d, n) cpu
                U_sel = U_sel_cpu.to(device)   # (d, n) -- stacked keys
                # compute inv once
                uuT = U_sel @ U_sel.t()
                eye = torch.eye(uuT.shape[0], device=device, dtype=uuT.dtype)
                try:
                    inv = torch.inverse(uuT + lambda_reg * eye)
                except RuntimeError:
                    inv = torch.pinverse(uuT + lambda_reg * eye)

                # grad_Wtilde[lname] shape (d', d)
                G = grad_Wtilde[lname]  # (d', d)
                # M = G · inv  -> (d', d)
                M = G @ inv
                # grad_D = M @ U_sel  -> (d', n)
                grad_D = M @ U_sel  # (d', n)
                # proxy contribution: sum( grad_D * D )
                proxy_terms.append((grad_D * D).sum())

            # Algo 2, ln 9 - backprop proxy loss into hypernets
            if len(proxy_terms) == 0:
                logger.warning("No proxy terms found; skipping hypernet update this epoch.")
            else:
                proxy_loss = torch.stack(proxy_terms).sum()
                optimizer.zero_grad()
                proxy_loss.backward()
                optimizer.step()
                logger.info(f"Epoch {epoch+1}: proxy_loss={proxy_loss.item():.6g}, meta_loss={loss_meta.item():.6g}, loss_gen={loss_gen.item():.6g}, loss_loc={loss_loc.item():.6g}")

            # ----------------------
            # restore original gnn weights (undo S* applied)
            # ----------------------
            with torch.no_grad():
                for lname, module in self._get_target_layers():
                    if lname not in per_layer_info:
                        continue
                    S_star = per_layer_info[lname]["S_star"]
                    module_obj = dict(self._get_target_layers())[lname]
                    applied = False
                    if hasattr(module_obj, "lin") and hasattr(module_obj.lin, "weight"):
                        # restore
                        if lname in orig_weights:
                            module_obj.lin.weight.copy_(orig_weights[lname])
                            applied = True
                    elif hasattr(module_obj, "weight"):
                        if lname in orig_weights:
                            module_obj.weight.copy_(orig_weights[lname])
                            applied = True
                    else:
                        # attempt keys
                        key = (lname, "weight")
                        if key in orig_weights:
                            # attempt to find param and copy
                            for pname, p in module_obj.named_parameters():
                                if (lname, pname) in orig_weights:
                                    p.copy_(orig_weights[(lname, pname)])
                                    applied = True
                                    break
                    if not applied:
                        logger.warning(f"[Restore] Could not restore weights for {lname}")

            # end epoch loop

        logger.info("Finished MALMEN two-stage meta-training (multilayer).")
        return

    # -------------------------------
    # Train hypernetworks (Alg. 2)
    # -------------------------------
    def train_hypernetwork(self, **kwargs):
        """Wrapper around MALMEN-style meta-training."""
        logger.info("Starting hypernetwork meta-training (MALMEN-style)")
        num_epochs = kwargs.get("num_epochs", self.num_epochs)
        lambda_reg = kwargs.get("lambda_reg", self.lambda_reg)
        lambda_loc = kwargs.get("lambda_loc", self.lambda_loc)
        hidden = kwargs.get("hidden", self.hidden)
        lr = kwargs.get("lr", self.lr)
        num_targets = kwargs.get("num_targets", self.num_targets)
        select_kwargs = kwargs.get("select_kwargs", {})

        # call the MALMEN two-stage trainer (already implemented)
        self.train_hypernetwork_multilayer_malmen(
            num_epochs=num_epochs,
            lambda_reg=lambda_reg,
            hidden=hidden,
            lr=lr,
            num_targets=num_targets,
            select_kwargs=select_kwargs,
        )

    # -------------------------------
    # Perform actual edit (Alg. 1)
    # -------------------------------
    def edit_model(self, **kwargs) -> List[List[Any]]:
        """Perform inference-time edit using trained hypernets."""
        node_idx_2flip: torch.Tensor = kwargs["node_idx_2flip"]
        flipped_label: torch.Tensor = kwargs["flipped_label"]
        lambda_reg = kwargs.get("lambda_reg", self.lambda_reg)
        logger.info(f"Starting MALMEN-GNN editing with {len(node_idx_2flip)} targets")

        # apply learned hypernets to perform edit (Eq. 3–4)
        self.edit_model_multilayer(node_idx_2flip, flipped_label, lambda_reg=lambda_reg)

        # evaluate edited model
        bef_acc, val_acc, test_acc = test(self.model, self.whole_data)
        logger.info(f"Edited model accuracy -> Train: {bef_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        return [[bef_acc, val_acc, test_acc]]

    # -------------------------------
    # Save results and visualization
    # -------------------------------
    def save_and_plot_results(
        self,
        node_idx_2flip: torch.Tensor,
        model_before: torch.nn.Module,
        model_after: torch.nn.Module,
    ):
        """Save misclassification summary and visualize results."""
        logger.info("Saving and visualizing results")

        save_misclassifications_txt(
            self.config,
            model_before=model_before,
            model_after=model_after,
            whole_data=self.whole_data,
            method_name="hypergnn",
            model_name=self.config["pipeline_params"]["model_name"],
            file_suffix="",
        )

        save_misclassification_summary_txt(
            self.config,
            model_before=model_before,
            model_after=model_after,
            whole_data=self.whole_data,
            method_name="hypergnn",
            model_name=self.config["pipeline_params"]["model_name"],
            file_suffix="",
            edit_indices=node_idx_2flip,
        )

        plot_misclassification_by_attributes_before_after(
            self.config,
            model_before=model_before,
            model_after=model_after,
            whole_data=self.whole_data,
            method_name="hypergnn",
            model_name=self.config["pipeline_params"]["model_name"],
            file_suffix="",
        )

        plot_validation_correct_confidence_histogram(
            self.config,
            model_before=model_before,
            model_after=model_after,
            whole_data=self.whole_data,
            method_name="hypergnn",
            model_name=self.config["pipeline_params"]["model_name"],
            file_suffix="",
        )

        plot_targeted_edits_distribution(
            self.config,
            edited_node_idx=node_idx_2flip,
            whole_data=self.whole_data,
            method_name="hypergnn",
            model_name=self.config["pipeline_params"]["model_name"],
            file_suffix="",
        )

        visualize_validation(self.config, self.model, self.whole_data, node_idx_2flip, suffix=" (HyperGNN)")

    # -------------------------------
    # Orchestrate end-to-end editing experiment
    # -------------------------------
    def run_editing_experiment(self, **kwargs):
        """
        High-level wrapper: load model/data, train hypernetwork, apply edit,
        evaluate and visualize results.
        """
        logger.info(f"Running MALMEN-GNN editing experiment")

        # Load model + data
        self.load_model_and_data()
        bef_results = self.evaluate_before_edit()

        # Select targets
        node_idx_2flip, flipped_label = self.select_edit_targets(**kwargs)

        # Train hypernetwork using selected targets
        self.train_hypernetwork(
            num_targets=len(node_idx_2flip),
            select_kwargs=kwargs.get("select_kwargs", {}),
        )

        # Deepcopy model before applying edit
        model_before = deepcopy(self.model)

        # Apply one-shot edit using trained hypernets
        raw_results = self.edit_model(
            node_idx_2flip=node_idx_2flip,
            flipped_label=flipped_label,
        )

        # Save and visualize results
        # self.save_and_plot_results(node_idx_2flip, model_before, self.model)

        logger.info("MALMEN-GNN editing completed.")
        self.attach_edit_checkpoint_artifacts(None)
        return raw_results