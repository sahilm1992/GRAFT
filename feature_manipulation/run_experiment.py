from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

ROOT = str(Path(__file__).resolve().parents[1])
SEED_GNN = str(Path(ROOT) / "seed-gnn")
sys.path.insert(0, ROOT)
sys.path.insert(0, SEED_GNN)

from editing_pipelines import create_editor  # noqa: E402
from editing_pipelines.utils.model_io import load_model  # noqa: E402
from data import get_data  # noqa: E402
from main_utils import set_seeds_all  # noqa: E402
from pipelines.seed_gnn.pretrain_gnn import pretrain_gnn, _checkpoint_prefix  # noqa: E402
from edit_gnn.utils import test as seed_test  # noqa: E402

from .manipulator import CorrelationMode, FeatureManipulationSpec, FeatureManipulator, FeatureType


def evaluate_model(model, data):
    return seed_test(model, data)


def _experiment_dir(root: Path, exp_name: str, train_corr: str, val_corr: str, test_corr: str) -> Path:
    folder = f"train_{train_corr}__val_{val_corr}__test_{test_corr}"
    return root / exp_name / folder


def _load_pretrain_metrics(metrics_path: Path) -> Optional[Dict[str, object]]:
    if not metrics_path.exists():
        return None
    try:
        with open(metrics_path, "r") as handle:
            return json.load(handle)
    except Exception:
        return None


def _build_feature_spec(
    feature: int | str,
    feature_type: FeatureType,
    p: float,
    alpha: float,
    sigma: float,
    seed: int,
) -> FeatureManipulationSpec:
    return FeatureManipulationSpec(
        feature=feature,
        feature_type=feature_type,
        p=p,
        alpha=alpha,
        sigma=sigma,
        seed=seed,
    )


def _compute_feature_label_corr(data, feature_idx: int, mask: torch.Tensor) -> float:
    idx = mask.nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        return float("nan")
    x = data.x[idx, feature_idx].to(torch.float32)
    y = data.y[idx].to(torch.float32)
    x = x - x.mean()
    y = y - y.mean()
    x_std = x.std(unbiased=False)
    y_std = y.std(unbiased=False)
    denom = x_std * y_std
    if denom == 0:
        return float("nan")
    return float((x * y).mean() / denom)


def _compute_all_feature_label_corrs(data, mask: torch.Tensor) -> list[float]:
    idx = mask.nonzero(as_tuple=False).view(-1)
    if idx.numel() == 0:
        return []
    x = data.x[idx].to(torch.float32)
    y = data.y[idx].to(torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean()
    x_std = x.std(dim=0, unbiased=False)
    y_std = y.std(unbiased=False)
    denom = x_std * y_std
    valid = denom != 0
    corrs = torch.full_like(x_std, float("nan"))
    if valid.any():
        corrs[valid] = (x[:, valid] * y.unsqueeze(1)).mean(dim=0) / denom[valid]
    return [float(v) for v in corrs]


def run_experiment(
    base_config: Dict[str, object],
    exp_name: str,
    feature: int | str,
    feature_type: FeatureType,
    p: float = 0.8,
    alpha: float = 1.0,
    sigma: float = 1.0,
    seed: int = 0,
    train_corrs: Iterable[CorrelationMode] = ("positive"),
    val_corrs: Iterable[CorrelationMode] = ("positive"),
    method: str = "leastsquares",
    select_kwargs: Optional[Dict[str, object]] = None,
    edit_kwargs: Optional[Dict[str, object]] = None,
    results_root: Optional[str] = None,
) -> Dict[str, Dict[str, object]]:
    set_seeds_all(seed)
    base_cfg = copy.deepcopy(base_config)
    if results_root:
        root_dir = Path(results_root)
    else:
        mgmt = base_cfg.get("management", {})
        candidate = mgmt.get("output_folder_dir") or mgmt.get("pretrain_output_dir")
        root_dir = Path(candidate) if candidate else Path(__file__).resolve().parent / "experiments"
    root_dir.mkdir(parents=True, exist_ok=True)
    data, _, _ = get_data(
        base_cfg["management"]["dataset_dir"],
        base_cfg["eval_params"]["dataset"],
        base_cfg,
    )

    spec = _build_feature_spec(feature, feature_type, p=p, alpha=alpha, sigma=sigma, seed=seed)
    feature_idx = FeatureManipulator(spec).resolve_feature_index(data)
    
    # Ensure force_mlp_one_epoch is set in base_cfg for the entire experiment
    base_cfg["pipeline_params"]["force_mlp_one_epoch"] = True
    
    base_corrs = {
        "train": _compute_feature_label_corr(data, feature_idx, data.train_mask),
        "val": _compute_feature_label_corr(data, feature_idx, data.val_mask),
        "test": _compute_feature_label_corr(data, feature_idx, data.test_mask),
    }
    results: Dict[str, Dict[str, object]] = {}

    for train_corr in train_corrs:
        # Create a base experiment directory for this train_corr
        train_exp_dir = root_dir / exp_name / f"train_{train_corr}"
        train_exp_dir.mkdir(parents=True, exist_ok=True)

        config = copy.deepcopy(base_cfg)
        # Temporary output dir for pretraining
        config["management"]["output_folder_dir"] = str(train_exp_dir)
        config["management"]["pretrain_output_dir"] = str(train_exp_dir)
        config["management"]["exp_desc"] = f"{exp_name}_pretrain_{train_corr}"

        # ---- Pretrain (train and val splits manipulated) ----
        train_data = data.clone()
        train_manipulator = FeatureManipulator(spec)
        train_manipulator.apply_split(train_data, "train", train_corr)
        train_manipulator.apply_split(train_data, "val", train_corr)
        train_corr_after = _compute_feature_label_corr(train_data, feature_idx, train_data.train_mask)
        val_corr_after_pre = _compute_feature_label_corr(train_data, feature_idx, train_data.val_mask)
        train_all_before = _compute_all_feature_label_corrs(data, data.train_mask)
        train_all_after = _compute_all_feature_label_corrs(train_data, train_data.train_mask)
        print(
            f"[corr monitor] pretrain | split=train | before={base_corrs['train']:.4f} "
            f"| after={train_corr_after:.4f}"
        )
        print(
            f"[corr monitor] pretrain | split=val | before={base_corrs['val']:.4f} "
            f"| after={val_corr_after_pre:.4f}"
        )
        print(f"[corr monitor] pretrain | all_before={train_all_before}")
        print(f"[corr monitor] pretrain | all_after={train_all_after}")
        pretrain_gnn(config, data_override=train_data)

        # Load pretrained checkpoint once for this train_corr
        pretrained_model, pretrain_data, pretrain_whole, _, _ = load_model(config, data_override=train_data)
        pretrain_eval = evaluate_model(pretrained_model, pretrain_whole)
        
        # We will share this pretrained model across all val_corrs
        for val_corr in val_corrs:
            test_corr = val_corr
            exp_dir = _experiment_dir(root_dir, exp_name, train_corr, val_corr, test_corr)
            exp_dir.mkdir(parents=True, exist_ok=True)

            # Update config for this specific val_corr
            config_edit = copy.deepcopy(config)
            config_edit["management"]["output_folder_dir"] = str(exp_dir)
            config_edit["management"]["pretrain_output_dir"] = str(train_exp_dir) # Point to pretrain dir
            config_edit["management"]["exp_desc"] = f"{exp_name}_{train_corr}_{val_corr}_{test_corr}"

            # Save pretrained.pt in the specific experiment dir
            pretrained_path = exp_dir / "pretrained.pt"
            torch.save(pretrained_model.state_dict(), pretrained_path)
            
            # Save pretrain metrics in the specific experiment dir
            metrics_pretrain_path = exp_dir / "metrics_pretrain.json"
            # Try to load the metrics from the train_exp_dir (where they were saved by pretrain_gnn)
            original_metrics_pretrain = train_exp_dir / f"metrics_pretrain_{_checkpoint_prefix(config)}.json"
            try:
                with open(original_metrics_pretrain, "r") as handle:
                    pretrain_payload = json.load(handle)
            except FileNotFoundError:
                pretrain_payload = {}
            
            pretrain_payload["eval_metrics"] = {
                "overall": list(pretrain_eval["overall"]),
                "metrics": pretrain_eval["metrics"],
                "per_class": pretrain_eval["per_class"],
            }
            with open(metrics_pretrain_path, "w") as handle:
                json.dump(pretrain_payload, handle, indent=2)

            # ---- Editing (val and test splits manipulated) ----
            edit_data = data.clone()
            edit_manipulator = FeatureManipulator(spec)
            
            # Use a different seed for the editing stage manipulation to avoid 
            # using the exact same random samples as pretraining if the corr is the same.
            edit_manipulator.apply_split(edit_data, "val", val_corr)
            edit_manipulator.apply_split(edit_data, "test", val_corr)
            
            val_corr_after = _compute_feature_label_corr(edit_data, feature_idx, edit_data.val_mask)
            test_corr_after_edit = _compute_feature_label_corr(edit_data, feature_idx, edit_data.test_mask)
            val_all_before = _compute_all_feature_label_corrs(data, data.val_mask)
            val_all_after = _compute_all_feature_label_corrs(edit_data, edit_data.val_mask)
            print(
                f"[corr monitor] editing | split=val | before={base_corrs['val']:.4f} "
                f"| after={val_corr_after:.4f}"
            )
            print(
                f"[corr monitor] editing | split=test | before={base_corrs['test']:.4f} "
                f"| after={test_corr_after_edit:.4f}"
            )
            print(f"[corr monitor] editing | all_before={val_all_before}")
            print(f"[corr monitor] editing | all_after={val_all_after}")

            editor = create_editor(method, config_edit)
            
            # SANITY CHECK: Compare data before and after editing experiment call
            data_before_ptr = id(edit_data)
            x_before_sum = edit_data.x.sum().item()
            
            raw_results, processed_results = editor.run_editing_experiment(
                select_kwargs=select_kwargs or {},
                edit_kwargs=edit_kwargs or {},
                data_override=edit_data,
            )
            
            # Check if editor.whole_data is the same object and has same content as edit_data
            data_after_ptr = id(editor.whole_data)
            x_after_sum = editor.whole_data.x.sum().item()
            
            print(f"[sanity check] Experiment: {train_corr}/{val_corr}")
            print(f"[sanity check] Data object ID: before={data_before_ptr}, after={data_after_ptr} (Match: {data_before_ptr == data_after_ptr})")
            print(f"[sanity check] Data X sum: before={x_before_sum:.4f}, after={x_after_sum:.4f} (Match: {abs(x_before_sum - x_after_sum) < 1e-4})")
            
            if abs(x_before_sum - x_after_sum) > 1e-4:
                print(f"WARNING: Data content changed during editing! This might explain metric jumps.")

            edited_path = exp_dir / "edited.pt"
            torch.save(editor.get_model().state_dict(), edited_path)

            # ---- Evaluation (test split already manipulated) ----
            test_corr_after = _compute_feature_label_corr(
                editor.whole_data, feature_idx, editor.whole_data.test_mask
            )
            test_all_before = _compute_all_feature_label_corrs(data, data.test_mask)
            test_all_after = _compute_all_feature_label_corrs(editor.whole_data, editor.whole_data.test_mask)
            print(
                f"[corr monitor] evaluation | split=test | before={base_corrs['test']:.4f} "
                f"| after={test_corr_after:.4f}"
            )
            print(f"[corr monitor] evaluation | all_before={test_all_before}")
            print(f"[corr monitor] evaluation | all_after={test_all_after}")
            eval_metrics = evaluate_model(editor.get_model(), editor.whole_data)

            metrics_path = exp_dir / "metrics.json"
            metrics_payload = {
                "experiment": {
                    "exp_name": exp_name,
                    "train_corr": train_corr,
                    "val_corr": val_corr,
                    "test_corr": test_corr,
                    "feature": feature,
                    "feature_type": feature_type,
                    "p": p,
                    "alpha": alpha,
                    "sigma": sigma,
                    "seed": seed,
                },
                "artifacts": {
                    "pretrained": str(pretrained_path),
                    "edited": str(edited_path),
                },
                "feature_label_corr": {
                    "train": {"before": base_corrs["train"], "after": train_corr_after},
                    "val": {"before": base_corrs["val"], "after": val_corr_after},
                    "test": {"before": base_corrs["test"], "after": test_corr_after},
                },
                "all_feature_label_corr": {
                    "train": {"before": train_all_before, "after": train_all_after},
                    "val": {"before": val_all_before, "after": val_all_after},
                    "test": {"before": test_all_before, "after": test_all_after},
                },
                "pretrain_metrics": _load_pretrain_metrics(exp_dir / "metrics_pretrain.json"),
                "edit_results": {
                    "raw": raw_results,
                    "processed": processed_results,
                },
                "eval_metrics": {
                    "overall": list(eval_metrics["overall"]),
                    "metrics": eval_metrics["metrics"],
                    "per_class": eval_metrics["per_class"],
                },
            }
            with open(metrics_path, "w") as handle:
                json.dump(metrics_payload, handle, indent=2)

            results[str(exp_dir)] = metrics_payload

    return results


RUN_EXPERIMENT_FN = run_experiment
EVAL_FN = evaluate_model
