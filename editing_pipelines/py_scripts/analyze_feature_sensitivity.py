import sys
import os
import json
import datetime
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from zoneinfo import ZoneInfo

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from editing_pipelines._ensure_repo_paths import bootstrap  # noqa: E402

bootstrap()

import main_utils as main_utils
from editing_pipelines.utils.model_io import load_model, get_device
from editing_pipelines.utils.results import perturb_feature_and_measure_probs
from edit_gnn.utils import prediction, test
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

def main():
    parser = argparse.ArgumentParser(description="Analyze feature sensitivity of a trained GNN.")
    parser.add_argument('--exp_desc', type=str, help='experiment description.')
    parser.add_argument('--pipeline_config_dir', type=str, required=True, help='file path of pipeline config.')
    parser.add_argument('--eval_config_dir', type=str, required=True, help='file path of eval config.')
    parser.add_argument('--output_folder_dir', default='results/', type=str, help='path of output result')
    parser.add_argument('--job_post_via', default='terminal', type=str, help='submission method (e.g., terminal, slurm_sbatch)')
    parser.add_argument('--pretrain_output_dir', default='ckpts/', type=str)
    parser.add_argument('--dataset_dir', default='datalake/', type=str)
    
    # Feature ablation arguments
    parser.add_argument('--use_feature_ablation', action='store_true')
    parser.add_argument('--feature_variant', type=str, default='full_features')
    parser.add_argument('--drop_features', type=str, nargs='*', default=[])

    # Sensitivity arguments
    parser.add_argument('--perturb_feature', type=str, help='Feature name to perturb. If None, tries to find sensitive_feature in config.')
    parser.add_argument('--num_seeds', type=int, default=100, help='Number of perturbations.')
    parser.add_argument('--prob_mode', type=str, default='positive', choices=['true_class', 'positive'], help='Probability mode: true_class or positive (class 1).')
    parser.add_argument('--task', type=str, default='pretrain')

    args = parser.parse_args()
    
    if args.output_folder_dir != '' and args.output_folder_dir[-1] != '/':
        args.output_folder_dir += '/'

    # Standard setup from seed-gnn
    config = main_utils.register_args_and_configs(args)
    logger = main_utils.set_logger(args.output_folder_dir, args)
    
    # Sync command line ablation args with config
    pp = config.setdefault("pipeline_params", {})
    if args.use_feature_ablation:
        pp["feature_variant"] = args.feature_variant
        pp["drop_features"] = args.drop_features
        pp["use_feature_ablated_ckpts"] = True
    
    feature_variant = pp.get("feature_variant", "full_features")
    use_feature_ablated = pp.get("use_feature_ablated_ckpts", False)
    
    # Resolve checkpoint directory
    ckpt_base = Path(config["management"]["pretrain_output_dir"]).resolve()
    if use_feature_ablated and "edit_ckpts_feature_ablated" not in ckpt_base.as_posix():
        ckpt_base = ckpt_base.parent / "edit_ckpts_feature_ablated"
    
    dataset = config["eval_params"]["dataset"]

    def has_checkpoints(path: Path) -> bool:
        if not path.exists():
            return False
        if path.is_file():
            return path.suffix == ".pt"
        for _ in path.rglob("*.pt"):
            return True
        return False

    dataset_dir = ckpt_base
    parts_lower = [part.lower() for part in ckpt_base.parts]
    if dataset.lower() not in parts_lower:
        dataset_dir = ckpt_base / dataset

    candidate_dirs = [
        dataset_dir / feature_variant,
        dataset_dir,
        ckpt_base / feature_variant,
        ckpt_base,
    ]

    resolved_ckpt_dir = None
    for cand in candidate_dirs:
        if has_checkpoints(cand):
            resolved_ckpt_dir = cand
            break
    if resolved_ckpt_dir is None:
        resolved_ckpt_dir = dataset_dir / feature_variant

    def descend_to_ckpts(path: Path) -> Path:
        current = path
        visited = set()
        while current.is_dir():
            if any(current.glob("*.pt")):
                break
            key = current.resolve()
            if key in visited:
                break
            visited.add(key)
            dataset_child = current / dataset
            if dataset_child.exists():
                current = dataset_child
                continue
            subdirs = [p for p in current.iterdir() if p.is_dir()]
            if len(subdirs) == 1:
                current = subdirs[0]
                continue
            break
        return current

    resolved_ckpt_dir = descend_to_ckpts(resolved_ckpt_dir)

    config["management"]["pretrain_output_dir"] = str(resolved_ckpt_dir)
    
    results_dir = os.path.join(config["management"]["output_folder_dir"], feature_variant)
    os.makedirs(results_dir, exist_ok=True)
    config["management"]["output_folder_dir"] = results_dir

    # Load model and data
    model, train_data, whole_data, nf, nc = load_model(config)
    
    # Calculate base accuracies
    print("Calculating base accuracies...")
    base_metrics = test(model, whole_data)
    train_acc, val_acc, test_acc = base_metrics["overall"]
    print(f"Base Accuracies - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

    with torch.no_grad():
        logits_full = prediction(model, whole_data)

    def compute_auc_pr(mask):
        if mask is None:
            return float("nan")
        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return float("nan")
        y_true = whole_data.y[idx].cpu().numpy()
        probs = torch.softmax(logits_full[idx], dim=1).cpu().numpy()
        num_classes = probs.shape[1]
        if num_classes == 1:
            return float("nan")
        if num_classes == 2:
            return float(average_precision_score(y_true, probs[:, 1]))
        y_one_hot = label_binarize(y_true, classes=np.arange(num_classes))
        return float(average_precision_score(y_one_hot, probs, average="macro"))

    train_auc_pr = compute_auc_pr(whole_data.train_mask)
    val_auc_pr = compute_auc_pr(whole_data.val_mask)
    test_auc_pr = compute_auc_pr(whole_data.test_mask)
    print(f"Base AUC-PR - Train: {train_auc_pr:.4f}, Val: {val_auc_pr:.4f}, Test: {test_auc_pr:.4f}")
    
    # Resolve which feature to perturb
    feature_name = args.perturb_feature
    if not feature_name:
        feature_name = pp.get("sensitive_feature")
        if not feature_name:
            if dataset.lower() == "bail": feature_name = "WHITE"
            elif dataset.lower() == "income": feature_name = "race"
            elif dataset.lower() == "pokec": feature_name = "AGE"
            else: feature_name = "AGE"

    # Reuse utility function
    print(f"Starting sensitivity analysis for feature '{feature_name}' with {args.num_seeds} seeds (mode={args.prob_mode})...")
    df_val, df_test = perturb_feature_and_measure_probs(
        model,
        whole_data,
        feature_name=feature_name,
        K=args.num_seeds,
        prob_mode=args.prob_mode,
        compute_flips=True
    )
    
    results_df = pd.concat([df_val, df_test]).sort_values("Node")
    
    # Save results
    save_path = os.path.join(results_dir, f"sensitivity_analysis_{feature_name}_n{args.num_seeds}_{args.prob_mode}.csv")
    results_df.to_csv(save_path, index=False)
    print(f"\nResults saved to {save_path}")
    
    summary_rows = []
    # Add Train row
    summary_rows.append({
        "Split": "TRAIN",
        "Accuracy": train_acc,
        "AUC_PR": train_auc_pr,
    })
    
    for split_name, split_df, acc in [("VAL", df_val, val_acc), ("TEST", df_test, test_acc)]:
        row = {
            "Split": split_name,
            "Accuracy": acc,
            "AUC_PR": val_auc_pr if split_name == "VAL" else test_auc_pr,
            "MeanVarProb": split_df["VarProb"].mean(),
            "MedianVarProb": split_df["VarProb"].median(),
        }
        if "CorrectFraction" in split_df:
            row["PostPerturbAcc"] = split_df["CorrectFraction"].mean()
        if "RelVarProb" in split_df:
            row["MeanRelVarProb"] = split_df["RelVarProb"].mean()
            row["MedianRelVarProb"] = split_df["RelVarProb"].median()
        if "FlipFraction" in split_df:
            row["MeanFlipFraction"] = split_df["FlipFraction"].mean()
            row["MedianFlipFraction"] = split_df["FlipFraction"].median()
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(results_dir, f"sensitivity_summary_{feature_name}_n{args.num_seeds}_{args.prob_mode}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")
    
    # Summary
    print(f"\n--- {feature_name} Sensitivity Summary ({args.num_seeds} seeds, {args.prob_mode} prob) ---")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Train AUC-PR: {train_auc_pr:.4f}")
    print(f"VAL Accuracy: {val_acc:.4f}")
    print(f"VAL AUC-PR: {val_auc_pr:.4f}")
    if "CorrectFraction" in df_val:
        print(f"VAL Post-Perturbation Accuracy: {df_val['CorrectFraction'].mean():.4f}")
    print(f"VAL Mean VarProb: {df_val['VarProb'].mean():.6f}")
    print(f"VAL Mean Flip Fraction: {df_val['FlipFraction'].mean():.4f}")
    print(f"TEST Accuracy: {test_acc:.4f}")
    print(f"TEST AUC-PR: {test_auc_pr:.4f}")
    if "CorrectFraction" in df_test:
        print(f"TEST Post-Perturbation Accuracy: {df_test['CorrectFraction'].mean():.4f}")
    print(f"TEST Mean VarProb: {df_test['VarProb'].mean():.6f}")
    print(f"TEST Mean Flip Fraction: {df_test['FlipFraction'].mean():.4f}")
    if "RelVarProb" in df_val:
        print(f"VAL Mean RelVarProb: {df_val['RelVarProb'].mean():.6f}")
    if "RelVarProb" in df_test:
        print(f"TEST Mean RelVarProb: {df_test['RelVarProb'].mean():.6f}")

if __name__ == "__main__":
    main()
