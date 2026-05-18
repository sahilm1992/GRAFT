"""
EGNN Editor implementation.
"""

import logging
import time
import json
import os
from copy import deepcopy
from typing import Dict, List, Any
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm

from editing_pipelines.editors.base import BaseEditor
from editing_pipelines.utils.model_io import get_optimizer
from editing_pipelines.utils.train_eval import test, success_rate
from editing_pipelines.utils.selection import select_edit_targets_by_strategy
from editing_pipelines.utils.metrics import compute_full_auc_pr_by_split
from editing_pipelines.utils.results import (
    save_misclassifications_txt,
    save_misclassification_summary_txt,
    perturb_feature_and_measure_probs,
)
from editing_pipelines.utils.visualization import plot_misclassification_by_attributes_before_after, plot_targeted_edits_distribution, plot_validation_correct_confidence_histogram
from editing_pipelines.utils.lse_eval_utils import evaluate_edit_effects



from main_utils import set_seeds_all
from edit_gnn.utils import prediction, test as seed_test, grab_input
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from constants import SEED

logger = logging.getLogger("main")


class EGNNEditor(BaseEditor):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sensitive_feature = config.get("pipeline_params", {}).get("sensitive_feature", "AGE")
        self.fixed_sensitive_values = config.get("pipeline_params", {}).get("fixed_sensitive_values", None)
        self.save_dir = os.path.join(
            self.config["management"]["output_folder_dir"],
            "analysis",
            "egnn",
        )
        logger.info("Initialized EGNN Editor")

    def select_edit_targets(self, **kwargs):
        num_targets = kwargs.get('num_targets', self.config['eval_params']['num_targets'])
        strategy = kwargs.get('strategy') or self.config.get('target_selection', {}).get('strategy', 'default')
        strategy = strategy.lower()

        # Native EGNN parity default: random sampling from misclassified validation nodes.
        # If top_fraction is configured, derive target count from that fraction.
        if strategy == 'default':
            strategy = 'random_misclassified_valid'
            top_fraction = self.config.get("pipeline_params", {}).get("leastsquares", {}).get("top_fraction", None)
            if top_fraction is not None:
                try:
                    top_fraction = float(top_fraction)
                except (TypeError, ValueError):
                    top_fraction = None
            if top_fraction is not None:
                with torch.no_grad():
                    val_mask = self.whole_data.val_mask
                    val_count = int(val_mask.sum().item())
                if val_count > 0:
                    num_targets = max(1, int(top_fraction * val_count))
                logger.info(
                    "EGNN default target sampling uses top_fraction=%s -> num_targets=%s (val_nodes=%s)",
                    str(top_fraction),
                    str(num_targets),
                    str(val_count),
                )

        node_idx_2flip, flipped_label = select_edit_targets_by_strategy(
            self.model, self.whole_data, self.num_classes, num_targets, strategy
        )
        device = self.get_device()
        return node_idx_2flip.to(device), flipped_label.to(device)

    def _prepare_mlp_for_fast_forward(self):
        # EGNN edit ops call model.fast_forward() for *_MLP models.
        # If we skip full MLP pre-finetuning, we still must prepare cached GNN logits.
        if '_MLP' not in self.config['pipeline_params']['model_name']:
            return
        if not hasattr(self.model, "freeze_module"):
            return

        # Use combined mode expected by fast_forward (mlp_freezed must be False).
        self.model.freeze_module(train=False)
        with torch.no_grad():
            # Cache GNN backbone output with MLP path disabled.
            self.model.mlp_freezed = True
            self.model.gnn_output = self.model(**grab_input(self.whole_data)).detach().cpu()
            self.model.mlp_freezed = False

    def _is_regression_task(self) -> bool:
        return bool(
            getattr(self.whole_data, "task_type", "") == "regression"
            or self.whole_data.y.dtype.is_floating_point
        )

    def _to_regression_vector(self, output: torch.Tensor) -> torch.Tensor:
        if output.dim() == 2 and output.size(-1) == 1:
            return output.squeeze(-1)
        return output

    @torch.no_grad()
    def _evaluate_model(self, model_obj):
        if not self._is_regression_task():
            return seed_test(model_obj, self.whole_data)
        pred = self._to_regression_vector(prediction(model_obj, self.whole_data))
        y_true = self.whole_data.y.to(pred.dtype)

        def _metrics(mask):
            pred_m = pred[mask]
            y_m = y_true[mask]
            if pred_m.numel() == 0:
                return {"mae": float("nan"), "mse": float("nan"), "rmse": float("nan"), "r2": float("nan")}
            diff = pred_m - y_m
            mse = torch.mean(diff ** 2)
            mae = torch.mean(torch.abs(diff))
            rmse = torch.sqrt(mse)
            ss_res = torch.sum(diff ** 2)
            y_mean = torch.mean(y_m)
            ss_tot = torch.sum((y_m - y_mean) ** 2)
            r2 = float("nan") if ss_tot.item() <= 1e-12 else float(1.0 - (ss_res / (ss_tot + 1e-12)).item())
            return {"mae": float(mae.item()), "mse": float(mse.item()), "rmse": float(rmse.item()), "r2": r2}

        train_metrics = _metrics(self.whole_data.train_mask)
        val_metrics = _metrics(self.whole_data.val_mask)
        test_metrics = _metrics(self.whole_data.test_mask)
        return {
            "overall": (train_metrics["rmse"], val_metrics["rmse"], test_metrics["rmse"]),
            "metrics": {"train": train_metrics, "val": val_metrics, "test": test_metrics},
            "per_class": {"train": {}, "val": {}, "test": {}},
        }

    def _get_node_degrees(self, data) -> np.ndarray:
        num_nodes = int(data.num_nodes)
        if hasattr(data, "degree") and data.degree is not None:
            try:
                return data.degree.detach().cpu().to(torch.float32).numpy()
            except Exception:
                pass
        if hasattr(data, "edge_index") and data.edge_index is not None:
            try:
                src = data.edge_index[0]
                deg = torch.bincount(src, minlength=num_nodes).to(torch.float32)
                return deg.cpu().numpy()
            except Exception:
                pass
        return np.zeros(num_nodes, dtype=np.float32)

    def _build_transition_subsets(
        self,
        mask: torch.Tensor,
        correct_before: torch.Tensor,
        correct_after: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        subsets = {"0->0": [], "0->1": [], "1->0": [], "1->1": []}
        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return {}
        for node in idx.tolist():
            before = int(correct_before[node].item())
            after = int(correct_after[node].item())
            key = f"{before}->{after}"
            subsets.setdefault(key, []).append(node)
        return {
            key: torch.tensor(values, dtype=torch.long, device=mask.device)
            for key, values in subsets.items()
            if len(values) > 0
        }

    def edit_model(self, **kwargs) -> List[List[Any]]:
        node_idx_2flip: torch.Tensor = kwargs['node_idx_2flip']
        flipped_label: torch.Tensor = kwargs['flipped_label']
        logger.info(f"Starting EGNN editing with {len(node_idx_2flip)} targets")

        device = self.get_device()
        self.model = self.model.to(device)
        self.whole_data = self.whole_data.to(device)
        self.model_before = deepcopy(self.model)
        original_model = self.model_before
        model = deepcopy(self.model).to(device).train()
        is_regression = self._is_regression_task()
        max_num_step = kwargs.get('max_num_step', self.config['pipeline_params']['max_num_edit_steps'])
        optimizer = get_optimizer(self.config['pipeline_params'], model)
        # Build perturbed copies of selected validation targets and
        # enforce that perturbed copies keep the same label target.
        use_steering = bool(self.config.get("pipeline_params", {}).get("egnn_use_steering", True))

        X_aug_all = []
        if use_steering:
            logger.info("EGNN perturbation copies enabled")
            from editing_pipelines.editors.leastsquareseditor import LeastSquaresEditor
            ls_editor = LeastSquaresEditor(self.config)
            ls_editor.model = original_model
            ls_editor.whole_data = self.whole_data
            ls_editor.representative_examples = node_idx_2flip
            ls_editor.sensitive_feature = self.sensitive_feature
            ls_editor.fixed_sensitive_values = self.fixed_sensitive_values
            K = int(self.config.get("pipeline_params", {}).get("leastsquares", {}).get("num_aug", 6))
            X_aug_all = ls_editor.curate_steering_examples(representative_examples=node_idx_2flip, K=K)
        else:
            logger.info("EGNN steering disabled (original-style behavior)")

        raw_results = []
        for step_idx, (idx, f_label) in enumerate(
            tqdm(zip(node_idx_2flip, flipped_label), desc="EGNN Editing", total=len(node_idx_2flip))
        ):
            idx = idx.view(-1)
            f_label = f_label.view(-1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start_t = time.time()
            steps = 0
            current_success = 0.0

            for step in range(1, max_num_step + 1):
                optimizer.zero_grad()
                logits_clean = model(**grab_input(self.whole_data))
                if is_regression:
                    pred_clean = self._to_regression_vector(logits_clean)
                    edit_loss = F.mse_loss(pred_clean[idx], f_label.to(pred_clean.dtype))
                else:
                    edit_loss = F.cross_entropy(logits_clean[idx], f_label)
                aug_loss = torch.tensor(0.0, device=device)

                # Run the same EGNN-style target loss on each perturbed copy
                # using the same shared model parameters.
                if (
                    use_steering
                    and len(X_aug_all) > 0
                ):
                    for X_aug in X_aug_all:
                        X_aug = X_aug.to(device)
                        original_x = self.whole_data.x
                        self.whole_data.x = X_aug
                        try:
                            logits_aug = model(**grab_input(self.whole_data))
                        finally:
                            self.whole_data.x = original_x
                        if is_regression:
                            pred_aug = self._to_regression_vector(logits_aug)
                            aug_loss = aug_loss + F.mse_loss(pred_aug[idx], f_label.to(pred_aug.dtype))
                        else:
                            aug_loss = aug_loss + F.cross_entropy(logits_aug[idx], f_label)
                    aug_loss = aug_loss / max(len(X_aug_all), 1)

                num_views = 1 + (len(X_aug_all) if (use_steering and len(X_aug_all) > 0) else 0)
                total_loss = (edit_loss + (aug_loss * max(len(X_aug_all), 1) if num_views > 1 else 0.0)) / num_views
                total_loss.backward()
                optimizer.step()
                steps = step

                # Original EGNN behavior on clean input.
                current_target_success = success_rate(
                    model,
                    idx.view(-1),
                    f_label.view(-1),
                    self.whole_data,
                )
                # Apply same success check on every perturbed view.
                all_aug_success = 1.0
                if (
                    use_steering
                    and len(X_aug_all) > 0
                ):
                    for X_aug in X_aug_all:
                        X_aug = X_aug.to(device)
                        original_x = self.whole_data.x
                        self.whole_data.x = X_aug
                        try:
                            aug_success = success_rate(
                                model,
                                idx.view(-1),
                                f_label.view(-1),
                                self.whole_data,
                            )
                        finally:
                            self.whole_data.x = original_x
                        if aug_success != 1.0:
                            all_aug_success = 0.0
                            break

                if current_target_success == 1.0 and all_aug_success == 1.0:
                    break

            # Report cumulative success over edited-so-far nodes.
            current_success = success_rate(
                model,
                node_idx_2flip[:step_idx + 1].view(-1),
                flipped_label[:step_idx + 1].view(-1),
                self.whole_data,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                mem = 0.0
            tot_time = time.time() - start_t

            test_results = self._evaluate_model(model)["overall"]
            res = [*test_results, current_success, steps, mem, tot_time]
            raw_results.append(res)
        self.model = model.eval()
        logger.info("EGNN editing completed")
        if not is_regression:
            try:
                logger.info("Saving misclassification TXT")
                save_misclassifications_txt(
                    self.config,
                    model_before=self.model_before,
                    model_after=self.model,
                    whole_data=self.whole_data,
                    method_name='egnn',
                    model_name=self.config['pipeline_params']['model_name'],
                    file_suffix=''
                )
                save_misclassification_summary_txt(
                    self.config,
                    model_before=self.model_before,
                    model_after=self.model,
                    whole_data=self.whole_data,
                    method_name='egnn',
                    model_name=self.config['pipeline_params']['model_name'],
                    file_suffix='',
                    edit_indices=node_idx_2flip
                )

                logger.info("Plotting misclassification by attributes before and after editing")
                plot_misclassification_by_attributes_before_after(
                    self.config,
                    model_before=self.model_before,
                    model_after=self.model,
                    whole_data=self.whole_data,
                    method_name='egnn',
                    model_name=self.config['pipeline_params']['model_name'],
                    file_suffix=''
                )
                plot_validation_correct_confidence_histogram(
                    self.config,
                    model_before=self.model_before,
                    model_after=self.model,
                    whole_data=self.whole_data,
                    method_name='egnn',
                    model_name=self.config['pipeline_params']['model_name'],
                    file_suffix=''
                )
                plot_targeted_edits_distribution(
                    self.config,
                    edited_node_idx=node_idx_2flip,
                    whole_data=self.whole_data,
                    method_name='egnn',
                    model_name=self.config['pipeline_params']['model_name'],
                    file_suffix=''
                )
            except Exception as e:
                logger.warning(f"Failed to save misclassification TXT: {e}")
        return raw_results

    def _save_metrics(self, results_after, fairness_metrics, node_idx_2flip, edit_runtime):
        from pipelines.seed_gnn.pretrain_gnn import get_split_class_counts

        split_class_counts = get_split_class_counts(self.whole_data)

        is_regression = self._is_regression_task()
        metrics_before = deepcopy(self.bef_edit_results.get("metrics")) if isinstance(self.bef_edit_results, dict) else None
        metrics_after = deepcopy(results_after.get("metrics"))
        if not is_regression:
            auc_pr_before = compute_full_auc_pr_by_split(self.model_before, self.whole_data)
            auc_pr_after = compute_full_auc_pr_by_split(self.model, self.whole_data)
            if metrics_before:
                for split in ["train", "val", "test"]:
                    metrics_before[split]["auc_pr"] = auc_pr_before[split]
            if metrics_after:
                for split in ["train", "val", "test"]:
                    metrics_after[split]["auc_pr"] = auc_pr_after[split]

        feature_name = getattr(self, "sensitive_feature", "AGE")
        fixed_vals = getattr(self, "fixed_sensitive_values", None)
        val_sens_before, test_sens_before = perturb_feature_and_measure_probs(
            self.model_before, self.whole_data, feature_name=feature_name,
            sensitive_feature_values=fixed_vals, compute_flips=not is_regression
        )
        val_sens_after, test_sens_after = perturb_feature_and_measure_probs(
            self.model, self.whole_data, feature_name=feature_name,
            sensitive_feature_values=fixed_vals, compute_flips=not is_regression
        )

        sensitivity_metrics = {
            "before": {
                "val": {
                    "mean_var": float(val_sens_before["VarProb"].mean()),
                    "mean_rel_var": float(val_sens_before["RelVarProb"].mean()),
                    "mean_flip_fraction": float(val_sens_before["FlipFraction"].mean()) if "FlipFraction" in val_sens_before else float("nan"),
                },
                "test": {
                    "mean_var": float(test_sens_before["VarProb"].mean()),
                    "mean_rel_var": float(test_sens_before["RelVarProb"].mean()),
                    "mean_flip_fraction": float(test_sens_before["FlipFraction"].mean()) if "FlipFraction" in test_sens_before else float("nan"),
                },
            },
            "after": {
                "val": {
                    "mean_var": float(val_sens_after["VarProb"].mean()),
                    "mean_rel_var": float(val_sens_after["RelVarProb"].mean()),
                    "mean_flip_fraction": float(val_sens_after["FlipFraction"].mean()) if "FlipFraction" in val_sens_after else float("nan"),
                },
                "test": {
                    "mean_var": float(test_sens_after["VarProb"].mean()),
                    "mean_rel_var": float(test_sens_after["RelVarProb"].mean()),
                    "mean_flip_fraction": float(test_sens_after["FlipFraction"].mean()) if "FlipFraction" in test_sens_after else float("nan"),
                },
            },
        }
        for split in ["val", "test"]:
            before_var = sensitivity_metrics["before"][split]["mean_var"]
            after_var = sensitivity_metrics["after"][split]["mean_var"]
            if before_var != 0:
                pct_change = ((after_var - before_var) / before_var) * 100
            else:
                pct_change = 0.0 if after_var == 0 else float("inf")
            sensitivity_metrics["after"][split]["pct_var_change"] = pct_change

        metrics_json = {
            "experiment": {
                "exp_desc": self.config["management"]["exp_desc"],
                "task": self.config["management"]["task"],
                "seed": self.config["management"].get("seed", SEED),
                "method": "egnn",
            },
            "data": {
                "dataset": self.config["eval_params"]["dataset"],
                "num_nodes": int(self.whole_data.num_nodes),
                "num_features": self.num_features,
                "num_classes": self.num_classes,
                "class_distribution": split_class_counts,
            },
            "model": {
                "name": self.config["pipeline_params"]["model_name"],
                "architecture": self.config["pipeline_params"]["architecture"],
            },
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "sensitivity_metrics": sensitivity_metrics,
            "fairness_metrics": fairness_metrics,
            "edit_params": {
                "num_targets": int(len(node_idx_2flip)),
                "max_num_edit_steps": int(self.config["pipeline_params"]["max_num_edit_steps"]),
            },
            "weight_change_metrics": getattr(self, "weight_change_metrics", {}),
            "edit_runtime": edit_runtime,
        }

        seed = self.config["management"].get("seed", SEED)
        save_name = f"metrics_egnn_targets{len(node_idx_2flip)}_seed{seed}.json"
        out_file = os.path.join(self.config["management"]["output_folder_dir"], save_name)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(metrics_json, f, indent=2)
        logger.info(f"Saved EGNN metrics to {out_file}")
        self.attach_edit_checkpoint_artifacts(out_file)

    def run_editing_experiment(self, **kwargs):
        data_override = kwargs.get("data_override")
        self.load_model_and_data(data_override=data_override)

        seed = self.config["management"].get("seed", SEED)
        set_seeds_all(seed)
        device = self.get_device()
        self.model = self.model.to(device)
        self.whole_data = self.whole_data.to(device)

        self.bef_edit_results = self._evaluate_model(self.model)
        self._prepare_mlp_for_fast_forward()

        edit_start = time.perf_counter()
        node_idx_2flip, flipped_label = self.select_edit_targets(**kwargs)
        self.edit_model(node_idx_2flip=node_idx_2flip, flipped_label=flipped_label)
        edit_runtime = time.perf_counter() - edit_start

        results_after = self._evaluate_model(self.model)
        logger.info("Evaluating edit effects")
        # fairness_metrics = {} if self._is_regression_task() else evaluate_edit_effects(self)
        fairness_metrics = {}
        logger.info("Saving metrics")
        self._save_metrics(results_after, fairness_metrics, node_idx_2flip, edit_runtime)

        bef_acc, val_acc, test_acc = results_after["overall"]
        return [[bef_acc, val_acc, test_acc]], None


