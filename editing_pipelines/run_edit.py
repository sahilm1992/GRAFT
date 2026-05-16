#!/usr/bin/env python3
import os
import sys
import json
import argparse
import logging
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
SEED_GNN = os.path.join(ROOT, 'seed-gnn')

# Ensure imports
sys.path.insert(0, ROOT)
sys.path.insert(0, SEED_GNN)

from editing_pipelines import create_editor, get_method_description  # noqa: E402


def build_config(args):
    dataset_key = args.dataset.lower()
    default_hidden_channels = 32
    # Mapping from dataset to sensitive feature
    sensitive_feature_map = {
        "pokec": "AGE",
        "income": "fnlwgt",
        "bail": "WHITE",
        "credit": "Age",
        "yelp" : "feature_5",
        "tfinance" : "feature_8",
        "artnet-views": "feature_20_fraction",
        "twitch-views": "affiliate_status_1.0",
    }
    sensitive_feature_value_map = {
        "pokec": [15, 25, 35, 45, 55, 65],
        # "income": [0,1],
        "income": None, #auto for fnlwgt
        "bail": [0, 1],
        "credit": None,
        "yelp" : [0.39845687, 0.99998516],
        "tfinance" : None,
        "artnet-views": None,
        "twitch-views": None,
    }
    sensitive_feature = sensitive_feature_map.get(dataset_key, None)
    sensitive_feature_value = sensitive_feature_value_map.get(dataset_key, None)

    config = {
        'management': {
            'dataset_dir': args.dataset_dir,
            'pretrain_output_dir': args.pretrain_dir,
            'output_folder_dir': args.output_dir,
            'exp_desc': f"{args.method}_{args.dataset}_{args.model}",
            'task': 'edit',
            'seed': args.seed,
        },
        'eval_params': {
            'dataset': args.dataset,
            'num_targets': args.num_targets,
            'subgroup_features': [],
        },
        'pipeline_params': {
            'model_name': args.model,
            'optim': 'adam',
            'pretrain_lr': 0.01,
            'edit_lr': 0.001,
            'max_num_edit_steps': args.max_steps,
            'architecture': {
                'hidden_channels': default_hidden_channels,
                'num_layers': 2,
                'dropout': 0.1,
                "batch_norm": False,
                "residual": False,
            },
            'load_pretrained_backbone': getattr(args, 'load_pretrained_backbone', True),
            'force_mlp_one_epoch': True,
            'alpha': 0.25,
            'beta': 100,
            "normalize": True,
            "loop": True,
            "epochs": 500,
            "leastsquares": {
                "lambda_reg": 0.01,
                "bias_attr_idx": [],
                "use_mlp_linears": False,
                "skip_non_gnn_linears": True,
                "mean_steering_signal": False,
                "sensitivity_based": False,
                "only_correct": False,
                "top_fraction": 0.25,
                "seed": args.seed,
                "degree_filter": None,        # None | "high" | "low"
                "degree_fraction": 0.5,       # e.g. top/bottom 50%
                "metrics_save_name": "metrics_edit.json",
                "use_subspace": False,
                "subspace_variance": 0.95,
                "gamma_retain": 0.0,
                "pr_alpha": 0.85,
                "rank_mix_tau": 0.5,
            },
            "finetune": {
                "num_epochs": 50,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "top_fraction": 0.25,
                "seed": args.seed,
                "only_linear": False,
                "use_mlp_linears": False,
            },
            "sensitive_feature": sensitive_feature,
            "fixed_sensitive_values": sensitive_feature_value
        },
        'corruption': {
            'features': []
        }
    }
    strategy = getattr(args, "leastsquares_strategy", "confidence")
    if args.num_layers is not None:
        config['pipeline_params']['architecture']['num_layers'] = args.num_layers

    ls_cfg = config["pipeline_params"]["leastsquares"]
    if args.lambda_reg is not None:
        ls_cfg["lambda_reg"] = args.lambda_reg
    if args.top_fraction is not None:
        ls_cfg["top_fraction"] = args.top_fraction

    ft_cfg = config["pipeline_params"]["finetune"]
    if args.ft_epochs is not None:
        ft_cfg["num_epochs"] = args.ft_epochs
    if args.ft_lr is not None:
        ft_cfg["lr"] = args.ft_lr
    if args.top_fraction is not None:
        ft_cfg["top_fraction"] = args.top_fraction

    if getattr(args, 'pr_alpha', None) is not None:
        ls_cfg["pr_alpha"] = args.pr_alpha
    if getattr(args, 'rank_mix_tau', None) is not None:
        ls_cfg["rank_mix_tau"] = args.rank_mix_tau

    # Reset defaults
    ls_cfg["mean_steering_signal"] = False
    ls_cfg["sensitivity_based"] = False
    ls_cfg["only_correct"] = False
    ls_cfg["weighted_mean_signal"] = False
    ls_cfg["strategy_mode"] = strategy

    if strategy == "sensitivity_mean":
        ls_cfg["sensitivity_based"] = True
        ls_cfg["mean_steering_signal"] = True
    elif strategy == "confidence":
        ls_cfg["only_correct"] = True
    elif strategy == "sensitivity_wtd_mean":
        ls_cfg["sensitivity_based"] = True
        ls_cfg["mean_steering_signal"] = True
        ls_cfg["weighted_mean_signal"] = True
    elif strategy == "sens_pr":
        ls_cfg["sensitivity_based"] = True
        ls_cfg["mean_steering_signal"] = True
        ls_cfg["weighted_mean_signal"] = True
    elif strategy == "sens_pr_graphaware":
        ls_cfg["sensitivity_based"] = True
        ls_cfg["mean_steering_signal"] = True
        ls_cfg["weighted_mean_signal"] = True
    elif strategy == "sens_divrank":
        ls_cfg["sensitivity_based"] = True
        ls_cfg["mean_steering_signal"] = True
        ls_cfg["weighted_mean_signal"] = True
    elif strategy == "sens_divrank_graphaware":
        ls_cfg["sensitivity_based"] = True
        ls_cfg["mean_steering_signal"] = True
        ls_cfg["weighted_mean_signal"] = True
    elif strategy == "sens_subspace_retention":
        ls_cfg["sensitivity_based"] = True
        ls_cfg["mean_steering_signal"] = True
        ls_cfg["weighted_mean_signal"] = True
        ls_cfg["use_subspace"] = True
        ls_cfg["gamma_retain"] = 1.0
    elif strategy == "sens_subspace_retention_pr":
        ls_cfg["sensitivity_based"] = True
        ls_cfg["mean_steering_signal"] = True
        ls_cfg["weighted_mean_signal"] = True
        ls_cfg["use_subspace"] = True
        ls_cfg["gamma_retain"] = 1.0
    elif strategy == "sens_subspace_retention_pr_graphaware":
        ls_cfg["sensitivity_based"] = True
        ls_cfg["mean_steering_signal"] = True
        ls_cfg["weighted_mean_signal"] = True
        ls_cfg["use_subspace"] = True
        ls_cfg["gamma_retain"] = 1.0
    elif strategy == "sens_subspace_retention_divrank":
        ls_cfg["sensitivity_based"] = True
        ls_cfg["mean_steering_signal"] = True
        ls_cfg["weighted_mean_signal"] = True
        ls_cfg["use_subspace"] = True
        ls_cfg["gamma_retain"] = 1.0
    elif strategy == "sens_subspace_retention_divrank_graphaware":
        ls_cfg["sensitivity_based"] = True
        ls_cfg["mean_steering_signal"] = True
        ls_cfg["weighted_mean_signal"] = True
        ls_cfg["use_subspace"] = True
        ls_cfg["gamma_retain"] = 1.0
    elif strategy == "sens_subspace":
        ls_cfg["sensitivity_based"] = True
        ls_cfg["mean_steering_signal"] = True
        ls_cfg["weighted_mean_signal"] = True
        ls_cfg["use_subspace"] = True
        ls_cfg["gamma_retain"] = 0.0
    elif strategy == "sens_subspace_pr":
        ls_cfg["sensitivity_based"] = True
        ls_cfg["mean_steering_signal"] = True
        ls_cfg["weighted_mean_signal"] = True
        ls_cfg["use_subspace"] = True
        ls_cfg["gamma_retain"] = 0.0
    elif strategy == "sens_subspace_pr_graphaware":
        ls_cfg["sensitivity_based"] = True
        ls_cfg["mean_steering_signal"] = True
        ls_cfg["weighted_mean_signal"] = True
        ls_cfg["use_subspace"] = True
        ls_cfg["gamma_retain"] = 0.0
    elif strategy == "sens_subspace_divrank":
        ls_cfg["sensitivity_based"] = True
        ls_cfg["mean_steering_signal"] = True
        ls_cfg["weighted_mean_signal"] = True
        ls_cfg["use_subspace"] = True
        ls_cfg["gamma_retain"] = 0.0
    elif strategy == "sens_subspace_divrank_graphaware":
        ls_cfg["sensitivity_based"] = True
        ls_cfg["mean_steering_signal"] = True
        ls_cfg["weighted_mean_signal"] = True
        ls_cfg["use_subspace"] = True
        ls_cfg["gamma_retain"] = 0.0

    # Optional explicit override from CLI/env plumbing.
    if getattr(args, 'gamma_retain', None) is not None:
        ls_cfg["gamma_retain"] = args.gamma_retain

    return config


def main():
    parser = argparse.ArgumentParser(description='Run GNN editing experiment')
    parser.add_argument('--method', default='ewc', choices=['egnn', 'seed_gnn', 'ewc', 'hyper_gnn', 'leastsquares', 'finetune'])
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--model', default='GCN')
    parser.add_argument('--num-targets', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--leastsquares-strategy',
        type=str,
        default='confidence',
        choices=[
            'confidence',
            'sensitivity_mean',
            'sensitivity_wtd_mean',
            'sens_pr',
            'sens_pr_graphaware',
            'sens_divrank',
            'sens_divrank_graphaware',
            'sens_subspace_retention',
            'sens_subspace_retention_pr',
            'sens_subspace_retention_pr_graphaware',
            'sens_subspace_retention_divrank',
            'sens_subspace_retention_divrank_graphaware',
            'sens_subspace',
            'sens_subspace_pr',
            'sens_subspace_pr_graphaware',
            'sens_subspace_divrank',
            'sens_subspace_divrank_graphaware',
        ],
        help='Strategy controlling representative selection and steering signals.',
    )
    parser.add_argument('--strategy', default='', help='Target selection strategy')
    parser.add_argument('--dataset-dir', default=os.path.join(SEED_GNN))
    parser.add_argument('--pretrain-dir', default=os.path.join(SEED_GNN, 'pretrained_models'))
    parser.add_argument('--output-dir', default=os.path.join(ROOT, 'editing_pipelines', 'output'))
    parser.add_argument('--lambda-reg', type=float, default=None)
    parser.add_argument('--top-fraction', type=float, default=None)
    parser.add_argument('--num-layers', type=int, default=None)
    parser.add_argument('--ft-epochs', type=int, default=None)
    parser.add_argument('--ft-lr', type=float, default=None)
    parser.add_argument('--pr-alpha', type=float, default=None,
                        help='PageRank/DivRank damping factor (restart probability = 1 - pr_alpha)')
    parser.add_argument('--rank-mix-tau', type=float, default=None,
                        help='Rank-based mixing weight: 0 = pure sensitivity, 1 = pure PR/DivRank')
    parser.add_argument('--gamma-retain', type=float, default=None,
                        help='Retention strength for subspace-retention strategies.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    log = logging.getLogger('run_edit')

    config = build_config(args)
    print("LS STRATEGY =", args.leastsquares_strategy)


    ckpt_dir = Path(args.pretrain_dir) / args.dataset
    if not ckpt_dir.exists() or not any(ckpt_dir.glob(f"{args.model.replace('_MLP','')}_*.pt")):
        log.warning("No pretrained checkpoints found in %s. Ensure you pretrain the model first.", ckpt_dir)

    log.info('Method: %s | %s', args.method, get_method_description(args.method))
    log.info('Dataset: %s | Model: %s | Targets: %d | Steps: %d', args.dataset, args.model, args.num_targets, args.max_steps)
    if args.strategy:
        log.info('Target selection strategy: %s', args.strategy)

    editor = create_editor(args.method, config)
    select_kwargs = {}
    if args.strategy:
        select_kwargs['strategy'] = args.strategy
    raw_results, processed_results = editor.run_editing_experiment(
        select_kwargs=select_kwargs,
        edit_kwargs={}
    )

    print(json.dumps(processed_results, indent=2))


if __name__ == '__main__':
    main()


