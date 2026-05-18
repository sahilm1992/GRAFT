import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from editing_pipelines._ensure_repo_paths import bootstrap  # noqa: E402

bootstrap()

import main_utils as main_utils
from editing_pipelines.utils.model_io import load_model, get_device, detect_backbone_module


def _infer_num_layers(model: torch.nn.Module, logger: object = None) -> int:
    try:
        backbone, bb_name = detect_backbone_module(model)
        if backbone is not None and hasattr(backbone, "convs"):
            num_layers = len(backbone.convs)
            if logger is not None:
                logger.info(f"Backbone detected: {bb_name}, layers={num_layers}")
            if num_layers > 0:
                return num_layers
    except Exception:
        pass
    if hasattr(model, "convs"):
        try:
            return len(model.convs)
        except Exception:
            pass
    for module in model.modules():
        if hasattr(module, "convs"):
            try:
                return len(module.convs)
            except Exception:
                continue
    try:
        from torch_geometric.nn import MessagePassing
        layers = [m for m in model.modules() if isinstance(m, MessagePassing)]
        if layers:
            return len(layers)
    except Exception:
        pass
    raise ValueError("Unable to infer number of GNN layers (L).")


def _resolve_edge_index(data) -> torch.Tensor:
    if hasattr(data, "edge_index") and data.edge_index is not None:
        return data.edge_index
    if hasattr(data, "adj_t") and data.adj_t is not None:
        row, col, _ = data.adj_t.coo()
        return torch.stack([row, col], dim=0)
    raise ValueError("Data must have edge_index or adj_t to compute hop distances.")


def _build_adjacency(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    edge_index = edge_index.detach().cpu()
    row = edge_index[0].tolist()
    col = edge_index[1].tolist()
    adj: List[set] = [set() for _ in range(num_nodes)]
    for r, c in zip(row, col):
        adj[r].add(c)
        adj[c].add(r)
    return [sorted(list(neigh)) for neigh in adj]


def get_k_hop_neighbors(node: int, adjacency: List[List[int]], max_hops: int) -> Dict[int, List[int]]:
    visited = {node}
    frontier = {node}
    hop_neighbors: Dict[int, List[int]] = {}
    for h in range(1, max_hops + 1):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(adjacency[n])
        next_frontier.difference_update(visited)
        hop_neighbors[h] = sorted(list(next_frontier))
        visited.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break
    return hop_neighbors


def _prepare_model_input(data) -> Dict[str, torch.Tensor]:
    if hasattr(data, "adj_t") and data.adj_t is not None:
        return {"x": data.x, "adj_t": data.adj_t}
    return {"x": data.x, "adj_t": data.edge_index}


def compute_sif(
    model: torch.nn.Module,
    data,
    num_samples: int = 50,
    logger: object = None,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    model.eval()
    device = next(model.parameters()).device

    data = data.clone()
    data.x = data.x.to(device).detach().clone()
    if hasattr(data, "adj_t") and data.adj_t is not None:
        data.adj_t = data.adj_t.to(device)
    if hasattr(data, "edge_index") and data.edge_index is not None:
        data.edge_index = data.edge_index.to(device)

    num_nodes = int(data.num_nodes)
    num_samples = min(num_samples, num_nodes)
    L = _infer_num_layers(model, logger=logger)
    if logger is not None:
        logger.info(f"Model class: {model.__class__.__name__}")
        logger.info(f"Inferred num_layers (L): {L}")
        logger.info(f"Num nodes: {num_nodes}, num_samples: {num_samples}")
        logger.info(f"Has adj_t: {hasattr(data, 'adj_t') and data.adj_t is not None}")
        logger.info(f"Has edge_index: {hasattr(data, 'edge_index') and data.edge_index is not None}")

    edge_index = _resolve_edge_index(data)
    if logger is not None:
        logger.info(f"edge_index shape: {tuple(edge_index.shape)}")
    adjacency = _build_adjacency(edge_index, num_nodes)
    if logger is not None:
        degrees = torch.tensor([len(n) for n in adjacency], dtype=torch.long)
        logger.info(
            "Adjacency stats: "
            f"min_deg={int(degrees.min().item())}, "
            f"max_deg={int(degrees.max().item())}, "
            f"mean_deg={float(degrees.float().mean().item()):.3f}"
        )

    sample_nodes = torch.randperm(num_nodes)[:num_samples].tolist()
    if logger is not None:
        logger.info(f"Sampled nodes (first 10): {sample_nodes[:10]}")
    per_hop_sums = torch.zeros(L, device=device)
    self_sum = 0.0
    valid_samples = 0

    for u in sample_nodes:
        hop_nodes = get_k_hop_neighbors(u, adjacency, L)
        if logger is not None and valid_samples < 3:
            hop_counts = {h: len(v) for h, v in hop_nodes.items()}
            logger.info(f"Node {u} hop counts: {hop_counts}")

        data.x.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        if data.x.grad is not None:
            data.x.grad.zero_()

        logits = model(**_prepare_model_input(data))

        hop_order = [0] + [h for h in range(1, L + 1) if hop_nodes.get(h)]
        if not hop_order:
            if logger is not None:
                logger.info(f"Node {u} skipped: no hops within L={L}")
            continue
        # for idx, h in enumerate(hop_order):
        #     model.zero_grad(set_to_none=True)
        #     if data.x.grad is not None:
        #         data.x.grad.zero_()
        #     if h == 0:
        #         nodes = torch.tensor([u], device=device)
        #     else:
        #         nodes = torch.tensor(hop_nodes[h], device=device)
        #     if nodes.numel() == 0:
        #         continue
        #     loss = logits[nodes].sum()
        #     retain_graph = idx != (len(hop_order) - 1)
        #     loss.backward(retain_graph=retain_graph)
        #     grad_u = data.x.grad[u]
        #     grad_norm = torch.norm(grad_u, p=2)
        #     if h == 0:
        #         self_sum += float(grad_norm.item())
        #     else:
        #         per_hop_sums[h - 1] += grad_norm

        # Pre-compute predicted classes once
        pred = logits.argmax(dim=1)

        for h in hop_order:

            if h == 0:
                nodes = [u]
            else:
                nodes = hop_nodes[h]

            if not nodes:
                continue

            hop_grad_total = 0.0

            for v in nodes:

                model.zero_grad(set_to_none=True)
                if data.x.grad is not None:
                    data.x.grad.zero_()

                # Backprop ONLY from predicted-class logit
                loss = logits[v, pred[v]]
                loss.backward(retain_graph=True)

                grad_u = data.x.grad[u]
                grad_norm = torch.norm(grad_u, p=2).item()

                hop_grad_total += grad_norm

            hop_avg = hop_grad_total / len(nodes)

            if h == 0:
                self_sum += hop_avg
            else:
                per_hop_sums[h - 1] += hop_avg

        valid_samples += 1
        data.x = data.x.detach()

    if valid_samples == 0:
        raise RuntimeError("No valid samples were processed for SIF computation.")

    per_hop_avg = per_hop_sums / float(valid_samples)
    self_influence = self_sum / float(valid_samples)
    propagated_influence = float(per_hop_avg.sum().item())
    sif_value = propagated_influence / (self_influence + 1e-12)

    return sif_value, {
        "per_hop": per_hop_avg.detach().cpu(),
        "self_influence": float(self_influence),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Structural Influence Factor (SIF) for a trained GNN.")
    parser.add_argument('--exp_desc', type=str, default="sif_metric", help='experiment description.')
    parser.add_argument('--pipeline_config_dir', type=str, required=True, help='file path of pipeline config.')
    parser.add_argument('--eval_config_dir', type=str, required=True, help='file path of eval config.')
    parser.add_argument('--output_folder_dir', default='results/', type=str, help='path of output result')
    parser.add_argument('--job_post_via', default='terminal', type=str, help='submission method (e.g., terminal, slurm_sbatch)')
    parser.add_argument('--pretrain_output_dir', default='ckpts/', type=str)
    parser.add_argument('--dataset_dir', default='datalake/', type=str)
    parser.add_argument('--task', type=str, default='pretrain')
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--save_name', type=str, default='sif_metric_')

    args = parser.parse_args()
    if args.output_folder_dir != '' and args.output_folder_dir[-1] != '/':
        args.output_folder_dir += '/'

    config = main_utils.register_args_and_configs(args)
    logger = main_utils.set_logger(args.output_folder_dir, args)

    device = get_device()
    logger.info(f"Using device: {device}")

    model, _, whole_data, _, _ = load_model(config)

    logger.info(f"Computing SIF with num_samples={args.num_samples}...")
    sif_value, details = compute_sif(
        model,
        whole_data,
        num_samples=args.num_samples,
        logger=logger,
    )

    logger.info(f"SIF: {sif_value:.6f}")
    logger.info(f"Self influence: {details['self_influence']:.6f}")
    logger.info(f"Per-hop influence: {details['per_hop'].tolist()}")

    output_dir = Path(config["management"]["output_folder_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"{args.save_name}{args.seed}.json"
    payload = {
        "sif": float(sif_value),
        "self_influence": float(details["self_influence"]),
        "per_hop": [float(v) for v in details["per_hop"].tolist()],
        "num_samples": int(args.num_samples),
    }
    with open(save_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"SIF metrics saved to {save_path}")


if __name__ == "__main__":
    main()
