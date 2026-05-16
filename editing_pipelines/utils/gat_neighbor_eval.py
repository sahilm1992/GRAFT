"""
When to avoid full-graph GAT inference and use NeighborLoader-style evaluation instead.
"""

from __future__ import annotations

import torch
from torch import nn


def estimate_num_edges(graph_data) -> int:
    try:
        edge_index = getattr(graph_data, "edge_index", None)
        if edge_index is not None and hasattr(edge_index, "size"):
            return int(edge_index.size(1))
    except Exception:
        pass
    try:
        adj_t = getattr(graph_data, "adj_t", None)
        if adj_t is not None:
            if hasattr(adj_t, "nnz"):
                return int(adj_t.nnz())
            if torch.is_tensor(adj_t) and adj_t.dim() == 2:
                return int(adj_t.size(1))
    except Exception:
        pass
    return -1


def gat_backbone_num_layers(model: nn.Module) -> int:
    """Number of GATConv layers in the backbone; 0 if not a GAT-style model."""
    name = model.__class__.__name__
    if not name.startswith("GAT"):
        return 0
    gat = getattr(model, "GAT", None)
    if gat is not None and hasattr(gat, "convs"):
        return len(gat.convs)
    if hasattr(model, "convs"):
        return len(model.convs)
    return max(1, int(getattr(model, "num_layers", 1)))


def should_use_gat_neighbor_loader(
    model: nn.Module,
    graph_data,
    edge_threshold: int = 10_000_000,
) -> bool:
    """
    Full-graph GAT forward is prohibitive for very large graphs or multi-layer GATs.
    Use batched neighbor sampling when this returns True.
    """
    if not model.__class__.__name__.startswith("GAT"):
        return False
    if gat_backbone_num_layers(model) > 1:
        return True
    return estimate_num_edges(graph_data) >= edge_threshold
