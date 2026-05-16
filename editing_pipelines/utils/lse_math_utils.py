from __future__ import annotations

import logging
from typing import List, Optional

import torch
from torch import nn
from torch_geometric.nn import GATConv, SAGEConv, GINConv
from torch_geometric.loader import NeighborLoader
import torch_geometric.typing as pyg_typing

from edit_gnn.utils import prediction, grab_input

logger = logging.getLogger("main")


def stack_rows(xs: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat(xs, dim=0) if len(xs) > 1 else xs[0]


def get_weight(layer: nn.Module) -> torch.Tensor:
    """
    Return the primary learnable weight matrix for a GNN layer.
    Supports GCNConv, GATConv, SAGEConv, GINConv, and Linear.
    """
    if isinstance(layer, nn.Linear):
        return layer.weight

    if hasattr(layer, "lin") and hasattr(layer.lin, "weight"):
        return layer.lin.weight

    if isinstance(layer, GATConv):
        if hasattr(layer, "lin_src") and hasattr(layer.lin_src, "weight"):
            return layer.lin_src.weight

    if isinstance(layer, SAGEConv):
        if hasattr(layer, "lin_l") and hasattr(layer.lin_l, "weight"):
            return layer.lin_l.weight

    if isinstance(layer, GINConv) and hasattr(layer, "nn"):
        for sub in layer.nn.modules():
            if isinstance(sub, nn.Linear):
                return sub.weight

    raise AttributeError(
        f"[ERROR] Cannot extract weight for layer {layer.__class__.__name__}.\n"
        f"Available attrs: {list(layer.__dict__.keys())}"
    )


def set_weight(layer: nn.Module, new_w: torch.Tensor) -> None:
    w = get_weight(layer)
    assert w.shape == new_w.shape
    with torch.no_grad():
        w.copy_(new_w)


@torch.no_grad()
def capture_layer_inputs(
    model: nn.Module,
    data_obj,
    layer: nn.Module,
    node_idx: Optional[torch.Tensor] = None,
    override_x: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """
    Run a forward pass, capture the *input* that hits `layer` for the nodes of interest.
    We assume the model reads `data_obj.x` as node features by convention.
    """
    if device is not None:
        data_obj = data_obj.to(device)

    if override_x is not None:
        original_x = data_obj.x
        data_obj.x = override_x

    captured: List[torch.Tensor] = []

    def pre_hook(_module, inp):
        x_in = inp[0]
        if node_idx is None:
            captured.append(x_in)
        else:
            captured.append(x_in[node_idx])

    handle = layer.register_forward_pre_hook(pre_hook)
    try:
        _ = prediction(model, data_obj)
    except Exception as exc:
        msg = str(exc).lower()
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or ("out of memory" in msg)
        # For large-graph GAT, fall back to neighbor-sampled forward for the
        # specific target nodes so we can still capture layer inputs.
        if is_oom and node_idx is not None:
            logger.warning("[Capture] Full-graph forward OOM; using NeighborLoader fallback for capture.")
            captured.clear()

            # Some environments have pyg-lib sampler ABI mismatches; match
            # pretraining workaround by forcing torch-sparse backend.
            if getattr(pyg_typing, "WITH_PYG_LIB", False) and getattr(pyg_typing, "WITH_TORCH_SPARSE", False):
                pyg_typing.WITH_PYG_LIB = False

            model_device = next(model.parameters()).device
            target_idx_cpu = node_idx.view(-1).detach().cpu().to(torch.long)
            if target_idx_cpu.numel() == 0:
                raise RuntimeError("[Capture] Neighbor fallback requested with empty node_idx.")

            data_eval = data_obj.cpu()
            if override_x is not None:
                data_eval.x = override_x.detach().cpu()

            # Assume 2-layer default if model does not expose num_layers.
            num_layers = int(getattr(model, "num_layers", 2))
            loader = NeighborLoader(
                data_eval,
                input_nodes=target_idx_cpu,
                num_neighbors=[10] * num_layers,
                batch_size=min(1024, int(target_idx_cpu.numel())),
                shuffle=False,
            )

            current_seed_nodes = None
            current_seed_bs = 0
            captured_batches = []

            def pre_hook_neighbor(_module, inp):
                nonlocal current_seed_nodes, current_seed_bs
                if current_seed_nodes is None or current_seed_bs <= 0:
                    return
                x_in = inp[0]
                captured_batches.append(
                    (
                        current_seed_nodes.clone(),
                        x_in[:current_seed_bs].detach().cpu(),
                    )
                )

            handle.remove()
            handle = layer.register_forward_pre_hook(pre_hook_neighbor)
            for batch in loader:
                batch = batch.to(model_device)
                current_seed_bs = int(batch.batch_size)
                current_seed_nodes = batch.n_id[:current_seed_bs].detach().cpu().to(torch.long)
                _ = model(**grab_input(batch))

            # Rebuild outputs in requested node order.
            by_node = {}
            for seed_nodes, x_rows in captured_batches:
                for i in range(seed_nodes.numel()):
                    by_node[int(seed_nodes[i].item())] = x_rows[i]

            missing = [int(i.item()) for i in target_idx_cpu if int(i.item()) not in by_node]
            if missing:
                raise RuntimeError(f"[Capture] Neighbor fallback missing {len(missing)} target node captures.")

            captured_tensor = torch.stack([by_node[int(i.item())] for i in target_idx_cpu], dim=0).to(model_device)
            captured.append(captured_tensor)
        else:
            raise
    finally:
        handle.remove()
        if override_x is not None:
            data_obj.x = original_x

    if len(captured) == 0:
        logger.warning(f"[Capture] Layer {layer.__class__.__name__} received 0 forward calls; skipping.")
        return None
    return captured[0]
