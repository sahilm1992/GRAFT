import os
import json
import logging
from pathlib import Path

import torch

import models as models
from constants import SEED
from main_utils import set_seeds_all, parse_args, register_args_and_configs, set_logger
from data import get_data, prepare_dataset
from pipelines.seed_gnn.utils import get_optimizer
from pipelines.seed_gnn.modcirc_neuron_discovery import discover_gnn_circuits


logger = logging.getLogger("main")


def load_model_and_data(config):
    set_seeds_all(SEED)

    MODEL_FAMILY = getattr(models, config['pipeline_params']['model_name'])
    data, num_features, num_classes = get_data(
        config['management']['dataset_dir'], config['eval_params']['dataset'], config
    )

    save_path = os.path.join(
        config['management']['pretrain_output_dir'], config['eval_params']['dataset']
    )

    model = MODEL_FAMILY(
        in_channels=num_features,
        out_channels=num_classes,
        load_pretrained_backbone=True,
        saved_ckpt_path=save_path,
        **config['pipeline_params']['architecture']
    )
    model.cuda()

    # Ensure we have adjacency in sparse tensor format and train/val/test masks
    train_data, whole_data = prepare_dataset(config['pipeline_params'], data, remove_edge_index=True)
    del data
    return model, whole_data


def run_neuron_discovery(config):
    model, whole_data = load_model_and_data(config)

    feature_names = getattr(whole_data, 'feature_names', None)
    print(feature_names)
    features_cfg = config.get('corruption', {}).get('features', [])
    print(features_cfg)
    if not features_cfg:
        # default: try a few common features if names known; otherwise indices [0], [1]
        features_cfg = [[0], [1]]

    # Build target nodes from validation mask
    val_mask = whole_data.val_mask
    target_nodes = torch.nonzero(val_mask, as_tuple=False).view(-1)
    target_labels = whole_data.y[val_mask]

    # Save outputs
    out_root = os.path.join(
        config['management']['output_folder_dir'],
        'modcirc_results',
        f"{config['eval_params']['dataset']}_{config['pipeline_params']['model_name']}"
    )
    os.makedirs(out_root, exist_ok=True)
    # Run discovery
    circuit_nodes, node_importance, _ = discover_gnn_circuits(
        model=model,
        graph_data=whole_data,
        target_nodes=target_nodes,
        target_labels=target_labels,
        features_to_corrupt=[features_cfg],
        feature_names=feature_names,
        device='cuda',
        topk=config.get('modcirc', {}).get('topk', 10),
        visualize=True,
        save_plots=config['management']['output_folder_dir']+"/modcirc_results/circuit_analysis"
    )

    

    # Save circuit nodes summary
    summary_path = os.path.join(out_root, 'circuit_nodes.json')
    with open(summary_path, 'w') as f:
        json.dump(circuit_nodes, f, indent=2)

    # Save node importance per computational node
    importance_dir = os.path.join(out_root, 'node_importance')
    os.makedirs(importance_dir, exist_ok=True)
    for comp_node, tensor in node_importance.items():
        try:
            torch.save(tensor.detach().cpu(), os.path.join(importance_dir, f"{comp_node.replace('.', '_')}.pt"))
        except Exception:
            # Fallback to JSON list if tensor cannot be saved
            with open(os.path.join(importance_dir, f"{comp_node.replace('.', '_')}.json"), 'w') as f:
                json.dump(tensor.detach().cpu().tolist() if torch.is_tensor(tensor) else [], f)

    logger.info(f"Saved neuron discovery results to {out_root}")


def main():
    args = parse_args()
    config = register_args_and_configs(args)
    set_logger(args.output_folder_dir, args)
    run_neuron_discovery(config)


if __name__ == "__main__":
    main()


