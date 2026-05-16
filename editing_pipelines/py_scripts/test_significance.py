#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path
from itertools import product
from tqdm import tqdm
import numpy as np
import torch
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from scipy import stats

import logging
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
# -------------------------------
# Path setup
# -------------------------------
ROOT = str(Path(__file__).resolve().parents[2])
SEED_GNN = os.path.join(ROOT, 'seed-gnn')
sys.path.insert(0, ROOT)
sys.path.insert(0, SEED_GNN)

from pipelines.seed_gnn.pretrain_gnn import train as pretrain_train  # noqa: E402
import pipelines.seed_gnn.pretrain_gnn as pretrain_module  # noqa: E402
from editing_pipelines.run_edit import build_config  # noqa: E402
from editing_pipelines import create_editor  # noqa: E402
from editing_pipelines.utils.results import perturb_feature_and_measure_probs  # noqa: E402
from editing_pipelines.utils.train_eval import test

# -------------------------------
# Helpers
# -------------------------------
def compute_mean_varprob(model, data):
    """Compute mean VarProb and RelVarProb for VAL/TEST splits."""
    val_df, test_df = perturb_feature_and_measure_probs(model, data,feature_name="AGE",
            sensitive_feature_values=[15,25,35,45,55,65], relative=True)
    def safe_mean(df, col):
        return float(df[col].mean()) if (len(df) > 0 and col in df.columns) else float("nan")
    return {
        "VAL": {"VarProb": safe_mean(val_df, "VarProb"), "RelVarProb": safe_mean(val_df, "RelVarProb")},
        "TEST": {"VarProb": safe_mean(test_df, "VarProb"), "RelVarProb": safe_mean(test_df, "RelVarProb")},
    }

def extract_val_test_acc(eval_result):
    """
    Normalize evaluation output into (val_acc, test_acc).
    Supports:
      - tuple/list: (train, val, test)
      - dict: {"overall": (train, val, test)} or {"val": ..., "test": ...}
    """
    if isinstance(eval_result, (list, tuple)):
        if len(eval_result) < 3:
            raise ValueError(f"Expected at least 3 values in eval tuple/list, got {len(eval_result)}")
        return float(eval_result[1]), float(eval_result[2])

    if isinstance(eval_result, dict):
        overall = eval_result.get("overall")
        if isinstance(overall, (list, tuple)) and len(overall) >= 3:
            return float(overall[1]), float(overall[2])

        # fallback if split metrics are provided directly
        if "val" in eval_result and "test" in eval_result:
            return float(eval_result["val"]), float(eval_result["test"])

    raise TypeError(
        "Unsupported eval result format. Expected tuple/list or dict with "
        "'overall' or 'val'/'test' keys."
    )


def paired_permutation_test(before, after, n_perm=10000, alternative="two-sided", rng=None):
    """Paired permutation test on mean(after - before)."""
    rng = np.random.default_rng(rng)
    before, after = np.array(before), np.array(after)
    deltas = after - before
    obs = deltas.mean()
    # handle degenerate case
    if deltas.size == 0:
        return 1.0, obs
    signs = rng.choice([1, -1], size=(n_perm, len(deltas)))
    perm_means = (signs * deltas).mean(axis=1)
    if alternative == "less":
        p = (np.sum(perm_means <= obs) + 1) / (n_perm + 1)
    elif alternative == "greater":
        p = (np.sum(perm_means >= obs) + 1) / (n_perm + 1)
    else:  # two-sided
        p = (np.sum(np.abs(perm_means) >= abs(obs)) + 1) / (n_perm + 1)
    return p, obs


# -------------------------------
# Main pipeline
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Variance + Accuracy significance test across seeds and methods")
    parser.add_argument('--dataset', default='pokec')
    parser.add_argument('--model', default='GIN_MLP')
    parser.add_argument('--methods', nargs='+', default=['leastsquares'])
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--seed-start', type=int, default=1)
    parser.add_argument('--dataset-dir', default=os.path.join(SEED_GNN))
    parser.add_argument('--pretrain-dir', default=os.path.join(SEED_GNN, 'pretrained_models'))
    parser.add_argument('--output-dir', default=os.path.join(ROOT, 'editing_pipelines', 'output'))
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num-targets', type=int, default=5)
    parser.add_argument('--leastsquares-strategy', type=str, default='confidence',
                        choices=['confidence', 'sensitivity_mean', 'sensitivity_wtd_mean', 'sens_pr', 'sens_pr_graphaware'])
    parser.add_argument('--strategy', default='', help='Target selection strategy')
    args = parser.parse_args()

    out_root = Path(args.output_dir) / "variance_significance_full"
    out_root.mkdir(parents=True, exist_ok=True)

    seed_values = list(range(args.seed_start, args.seed_start + args.N))
    baseline_means = {"VAL": [], "TEST": []}

    # -----------------------------
    # 1) Pretrain models for baseline
    # -----------------------------
    for s in tqdm(seed_values, desc="Baseline training"):
        seed_pretrain_root = str(Path(args.pretrain_dir) / f"{args.dataset}_seed{s}")
        config = build_config(argparse.Namespace(
            method="none", dataset=args.dataset, model=args.model,
            num_targets=args.num_targets, max_steps=10,
            strategy=args.strategy, 
            leastsquares_strategy=args.leastsquares_strategy,
            dataset_dir=args.dataset_dir,
            pretrain_dir=args.pretrain_dir, output_dir=args.output_dir,
            load_pretrained_backbone=False,
            seed=s,
        ))
        config['management']['pretrain_output_dir'] = seed_pretrain_root
        if args.epochs:
            config['pipeline_params']['epochs'] = args.epochs
        pretrain_module.SEED = s
        model, data = pretrain_train(config)
        model.eval()
        mv = compute_mean_varprob(model, data)
        baseline_means["VAL"].append(mv["VAL"]["VarProb"])
        baseline_means["TEST"].append(mv["TEST"]["VarProb"])
        del model, data
        torch.cuda.empty_cache()

    diffs = {split: np.array([abs(i-j) for i,j in product(baseline_means[split], baseline_means[split])])
             for split in ["VAL","TEST"]}

    # -----------------------------
    # 2) Run editing for each method
    # -----------------------------
    for method in args.methods:
        mv_before_all = {"VAL": [], "TEST": []}
        mv_after_all = {"VAL": [], "TEST": []}
        mv_before_all_rel = {"VAL": [], "TEST": []}
        mv_after_all_rel = {"VAL": [], "TEST": []}
        acc_before_all = {"VAL": [], "TEST": []}
        acc_after_all = {"VAL": [], "TEST": []}

        for s in tqdm(seed_values, desc=f"Editing seeds ({method})"):
            seed_pretrain_root = str(Path(args.pretrain_dir) / f"{args.dataset}_seed{s}")
            config = build_config(argparse.Namespace(
                method=method, dataset=args.dataset, model=args.model,
                num_targets=args.num_targets, max_steps=10,
                strategy=args.strategy,
                leastsquares_strategy=args.leastsquares_strategy,
                dataset_dir=args.dataset_dir,
                pretrain_dir=args.pretrain_dir, output_dir=args.output_dir,
                load_pretrained_backbone=False,
                seed=s,
            ))
            config['management']['pretrain_output_dir'] = seed_pretrain_root
            if args.epochs:
                config['pipeline_params']['epochs'] = args.epochs
            editor = create_editor(method, config)
            raw_results, _ = editor.run_editing_experiment(select_kwargs={}, edit_kwargs={})

            # Accuracies
            accs = test(editor.model_before, editor.whole_data)
            val_acc, test_acc = extract_val_test_acc(accs)
            acc_before_all["VAL"].append(val_acc)
            acc_before_all["TEST"].append(test_acc)

            accs = test(editor.model, editor.whole_data)
            val_acc, test_acc = extract_val_test_acc(accs)
            acc_after_all["VAL"].append(val_acc)
            acc_after_all["TEST"].append(test_acc)

            mv = compute_mean_varprob(editor.model, editor.whole_data)
            mvb = compute_mean_varprob(editor.model_before, editor.whole_data)
            for split in ["VAL","TEST"]:
                mv_before_all[split].append(mvb[split]["VarProb"])
                mv_after_all[split].append(mv[split]["VarProb"])
                mv_before_all_rel[split].append(mvb[split]["RelVarProb"])
                mv_after_all_rel[split].append(mv[split]["RelVarProb"])
            del editor
            torch.cuda.empty_cache()

        # -----------------------------
        # 3) Statistical tests
        # -----------------------------
        results = {"method": method}
        delta_mean = {split: abs(np.mean(mv_after_all[split]) - np.mean(mv_before_all[split])) for split in ["VAL","TEST"]}
        results["baseline_magnitude_test"] = {
            split: float(((diffs[split] >= delta_mean[split]).sum()+1)/(diffs[split].size+1))
            for split in ["VAL","TEST"]
        }

        wilco_results = {}
        for split in ["VAL","TEST"]:
            b,a = np.array(mv_before_all[split]), np.array(mv_after_all[split])
            deltas = a-b
            # Decide alternative based on mean direction (after - before)
            mean_delta = float(np.mean(deltas)) if deltas.size>0 else 0.0

            if deltas.size == 0:
                # degenerate
                stat, p = np.nan, 1.0
                cohen_d = np.nan
                r = np.nan
                sd_diff = np.nan
            else:
                if np.allclose(deltas, 0.0):
                    # all zero differences -> no evidence
                    try:
                        stat, p = wilcoxon(a, b, alternative='two-sided')
                    except Exception:
                        stat, p = np.nan, 1.0
                    sd_diff = float(deltas.std(ddof=1)) if deltas.size>1 else 0.0
                    cohen_d = np.nan if sd_diff == 0 else float(mean_delta / sd_diff)
                    r = 0.0
                else:
                    # pick alternative consistent with hypothesis: mean_delta < 0 => after < before => 'less'
                    if mean_delta < 0:
                        alt = 'less'
                    elif mean_delta > 0:
                        alt = 'greater'
                    else:
                        alt = 'two-sided'
                    try:
                        stat, p = wilcoxon(a, b, alternative=alt)
                    except ValueError:
                        # fallback to two-sided if scipy complains (e.g., small n, ties)
                        stat, p = np.nan, 1.0
                    sd_diff = float(deltas.std(ddof=1)) if deltas.size>1 else 0.0
                    cohen_d = float(mean_delta / sd_diff) if sd_diff > 0 else np.nan

                    # compute z for effect-size r, robust to p=0 or 1
                    eps = 1e-15
                    p_clamped = max(min(p, 1.0 - eps), eps)
                    if alt == 'two-sided':
                        z = stats.norm.ppf(1 - p_clamped / 2.0)
                    else:
                        # one-sided p: use the one-sided mapping
                        z = stats.norm.ppf(1 - p_clamped)
                    # if p is extremely small, z will be large; r = |z| / sqrt(n)
                    n = max(1, len(deltas))
                    r = abs(z) / np.sqrt(n)

            wilco_results[split] = {"stat":float(stat) if not np.isnan(stat) else stat,
                "p_one_sided":float(p),
                "mean_delta":float(mean_delta),
                "std_delta":float(sd_diff) if not np.isnan(sd_diff) else sd_diff,
                "cohen_d":float(cohen_d) if not np.isnan(cohen_d) else cohen_d,
                "wilcoxon_r":float(r) if not np.isnan(r) else r}
        results["paired_wilcoxon"] = wilco_results

        # permutation
        perm_results = {split: {"p_two_sided": float(paired_permutation_test(
            mv_before_all[split], mv_after_all[split], n_perm=5000, alternative="two-sided")[0])}
            for split in ["VAL","TEST"]}
        results["paired_permutation"] = perm_results

        # relative variance
        rel_results = {}
        for split in ["VAL","TEST"]:
            b,a=np.array(mv_before_all_rel[split]),np.array(mv_after_all_rel[split])
            # compute deltas and choose two-sided by default for relvar
            if b.size == 0:
                stat, p = np.nan, 1.0
                mean_delta_rel = 0.0
                sd_delta_rel = np.nan
            else:
                try:
                    stat, p = wilcoxon(b, a, alternative="two-sided")
                except Exception:
                    stat, p = np.nan, 1.0
                mean_delta_rel = float((a-b).mean())
                sd_delta_rel = float((a-b).std(ddof=1)) if (a-b).size>1 else 0.0

            rel_results[split]={"stat":float(stat) if not np.isnan(stat) else stat,"p_two_sided":float(p),
                "mean_delta":float(mean_delta_rel),"std_delta":float(sd_delta_rel)}
        results["paired_wilcoxon_relvar"]=rel_results

        # -----------------------------
        # 4) Accuracy significance (two-sided)
        # -----------------------------
        accuracy_significance = {}
        for split in ["VAL","TEST"]:

            print(f"Accuracy before: {acc_before_all[split]}")
            print(f"Accuracy after: {acc_after_all[split]}")
            before=np.array(acc_before_all[split]); after=np.array(acc_after_all[split])
            deltas=after-before; n=len(deltas)
            mean_delta=float(np.mean(deltas)) if n>0 else 0.0
            std_delta=float(np.std(deltas,ddof=1)) if n>1 else 0.0
            try:
                if n == 0:
                    stat, p = np.nan, 1.0
                else:
                    stat, p = wilcoxon(after, before, alternative="two-sided")
            except ValueError:
                stat, p = np.nan, 1.0
            cohen_d=float(mean_delta/std_delta) if std_delta>0 else np.nan
            # two-sided -> z from p/2
            eps = 1e-15
            p_clamped = max(min(p, 1.0 - eps), eps)
            z = stats.norm.ppf(1 - p_clamped/2.0) if p_clamped < 1.0 else 0.0
            r = float(abs(z)/np.sqrt(n)) if n>0 else np.nan
            accuracy_significance[split]={"mean_before":float(before.mean()) if before.size>0 else float("nan"),
                "mean_after":float(after.mean()) if after.size>0 else float("nan"),
                "mean_delta":mean_delta,"std_delta":std_delta,"p_wilcoxon_two_sided":float(p),
                "cohen_d":cohen_d,"wilcoxon_r":r}
            plt.figure(figsize=(4,3))
            plt.hist(deltas,bins=max(3, min(20, n)),color="skyblue",edgecolor="black",alpha=0.7)
            plt.axvline(0,color="red",linestyle="--")
            plt.title(f"{split} Δ Accuracy (two-sided)")
            plt.tight_layout()
            plt.savefig(out_root/f"accuracy_delta_distribution_{split}_{method}.png",dpi=150)
            plt.close()
        results["accuracy_significance"]=accuracy_significance

        # -----------------------------
        # 5) Save report
        # -----------------------------
        out_path = out_root / f"variance_significance_{args.dataset}_{args.model}_{method}.json"
        with open(out_path,"w") as f: json.dump(results,f,indent=2)
        print(f"✅ Saved: {out_path}")
        print(json.dumps(results,indent=2))


if __name__=="__main__":
    main()
