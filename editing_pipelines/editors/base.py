"""
Base Editor class for GNN editing methods.
Provides common functionality shared by specific editors.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import os

import torch
from torch_geometric.data.data import Data

from editing_pipelines.utils.model_io import load_model
from editing_pipelines.utils.gat_neighbor_eval import should_use_gat_neighbor_loader
from editing_pipelines.utils.train_eval import test, finetune_gnn_mlp, seed_test
from editing_pipelines.utils.editing_ops import edit
from editing_pipelines.utils.visualization import visualize_validation
from editing_pipelines.utils.results import process_edit_results, process_raw_exp_results



from models.base import BaseModel

logger = logging.getLogger("main")


class BaseEditor(ABC):
    """Abstract base class for GNN editing methods."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.train_data = None
        self.whole_data = None
        self.num_features = None
        self.num_classes = None
        self.bef_edit_results = None

    def _edit_checkpoint_output_dir(self) -> Optional[str]:
        """
        Directory used only for post-edit ``.pt`` checkpoints.

        Defaults to ``{management.output_folder_dir}/checkpoints``. Override with
        ``management.edit_checkpoint_dir`` (absolute or relative path).
        """
        mgmt = self.config.get("management") or {}
        out_dir = mgmt.get("output_folder_dir")
        if not out_dir:
            return None
        explicit = mgmt.get("edit_checkpoint_dir")
        if explicit:
            return os.path.abspath(explicit)
        return os.path.abspath(os.path.join(out_dir, "checkpoints"))

    def _edited_checkpoint_filename(self, metrics_json_path: Optional[str]) -> str:
        """Basename for the saved ``.pt`` file (mirrors metrics slug without the ``metrics_`` prefix)."""
        if metrics_json_path:
            stem = Path(metrics_json_path).stem
            if stem.startswith("metrics_"):
                stem = stem[len("metrics_") :]
            return f"{stem}.pt" if stem else "edited.pt"
        mgmt = self.config.get("management") or {}
        seed = mgmt.get("seed", 0)
        model_name = self.config["pipeline_params"]["model_name"]
        arch = self.config["pipeline_params"].get("architecture") or {}
        num_layers = arch.get("num_layers", "?")
        short = model_name.replace("_MLP", "")
        return f"{short}_layers{num_layers}_seed{seed}.pt"

    def attach_edit_checkpoint_artifacts(self, metrics_json_path: Optional[str] = None) -> None:
        """
        Save edited model weights as a .pt file and record paths under ``artifacts`` using the same
        structure as ``metrics_pretrain_*.json`` (checkpoint_dir, best_checkpoint, output_dir).

        Checkpoints live under ``_edit_checkpoint_output_dir()`` (default:
        ``output_folder_dir/checkpoints``). The filename pairs with metrics JSON by stripping the
        leading ``metrics_`` from the stem (for example ``metrics_edit_foo.json`` → ``edit_foo.pt``).
        """
        if self.model is None:
            logger.warning("attach_edit_checkpoint_artifacts: no model; skipping checkpoint save")
            return
        mgmt = self.config.get("management") or {}
        out_dir = mgmt.get("output_folder_dir")
        if not out_dir:
            logger.warning("attach_edit_checkpoint_artifacts: missing output_folder_dir; skipping")
            return
        checkpoint_root = self._edit_checkpoint_output_dir()
        if checkpoint_root is None:
            return
        os.makedirs(checkpoint_root, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        abs_metrics = os.path.abspath(metrics_json_path) if metrics_json_path else None
        ckpt_path = os.path.join(checkpoint_root, self._edited_checkpoint_filename(metrics_json_path))

        state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state, ckpt_path)
        artifacts = {
            "checkpoint_dir": checkpoint_root,
            "best_checkpoint": {
                "path": os.path.abspath(ckpt_path),
                "epoch": 0,
                "selection_metric": "edited",
            },
            "output_dir": os.path.abspath(out_dir),
        }
        if abs_metrics is None:
            logger.info("Saved edited checkpoint to %s", ckpt_path)
            return
        try:
            with open(abs_metrics, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload["artifacts"] = artifacts
            with open(abs_metrics, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            logger.info("Saved edited checkpoint to %s; updated artifacts in %s", ckpt_path, abs_metrics)
        except Exception as exc:
            logger.warning("Checkpoint saved to %s but failed to patch metrics JSON %s: %s", ckpt_path, abs_metrics, exc)

    def get_device(self):
        force_cpu = os.getenv("FORCE_CPU", "0") == "1"
        has_visible_cuda = os.getenv("CUDA_VISIBLE_DEVICES", "") != ""
        if torch.cuda.is_available() and has_visible_cuda and not force_cpu:
            return torch.device("cuda")
        return torch.device("cpu")

    def _is_oom_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return isinstance(exc, torch.cuda.OutOfMemoryError) or ("out of memory" in msg)

    def _estimate_num_edges(self, graph_data: Data) -> int:
        try:
            edge_index = getattr(graph_data, "edge_index", None)
            if edge_index is not None and hasattr(edge_index, "size"):
                return int(edge_index.size(1))
        except Exception:
            pass
        try:
            adj_t = getattr(graph_data, "adj_t", None)
            if adj_t is not None and hasattr(adj_t, "nnz"):
                return int(adj_t.nnz())
        except Exception:
            pass
        return -1

    def _safe_seed_test(self):
        dataset_name = str(self.config.get("eval_params", {}).get("dataset", ""))
        model_name = str(self.config.get("pipeline_params", {}).get("model_name", ""))
        num_edges = self._estimate_num_edges(self.whole_data)
        arch = self.config.get("pipeline_params", {}).get("architecture") or {}
        try:
            num_layers = int(arch.get("num_layers", 2))
        except (TypeError, ValueError):
            num_layers = 2

        # Same policy as gat_neighbor_eval / load_model: skip full-graph seed_test only for
        # multi-layer GAT or very large graphs (not per-dataset).
        if not model_name.startswith("GAT"):
            should_skip = False
        else:
            should_skip = should_use_gat_neighbor_loader(self.model, self.whole_data)

        if should_skip:
            logger.warning(
                "Skipping seed_test for %s/%s (edges=%s, num_layers=%s) to avoid full-graph GAT OOM.",
                dataset_name,
                model_name,
                num_edges,
                num_layers,
            )
            return {
                "overall": (float("nan"), float("nan"), float("nan")),
                "per_class": {"train": {}, "val": {}, "test": {}},
            }

        try:
            return seed_test(self.model, self.whole_data)
        except Exception as exc:
            if self._is_oom_error(exc):
                logger.warning("Skipping seed_test due to OOM: %s", exc)
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                return {
                    "overall": (float("nan"), float("nan"), float("nan")),
                    "per_class": {"train": {}, "val": {}, "test": {}},
                }
            raise

    def load_model_and_data(self, data_override: Optional[Data] = None) -> Tuple[BaseModel, Data, Data, int, int]:
        logger.info(f"Loading model and data for {self.__class__.__name__}")
        self.model, self.train_data, self.whole_data, self.num_features, self.num_classes = load_model(
            self.config,
            data_override=data_override
        )
        logger.info(f"Model loaded: {self.model}")
        logger.info(f"Training data: {self.train_data}")
        logger.info(f"Whole data: {self.whole_data}")
        is_regression = bool(
            getattr(self.whole_data, "task_type", "") == "regression"
            or self.whole_data.y.dtype.is_floating_point
        )
        if is_regression:
            from edit_gnn.utils import grab_input  # type: ignore
            with torch.no_grad():
                pred = self.model(**grab_input(self.whole_data))
                if pred.dim() == 2 and pred.size(-1) == 1:
                    pred = pred.squeeze(-1)
                y = self.whole_data.y.to(pred.dtype)
                def _metrics(mask):
                    pred_m = pred[mask]
                    y_m = y[mask]
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
                    return {
                        "mae": float(mae.item()),
                        "mse": float(mse.item()),
                        "rmse": float(rmse.item()),
                        "r2": r2,
                    }
                train_metrics = _metrics(self.whole_data.train_mask)
                val_metrics = _metrics(self.whole_data.val_mask)
                test_metrics = _metrics(self.whole_data.test_mask)
                self.bef_edit_results = {
                    "overall": (train_metrics["rmse"], val_metrics["rmse"], test_metrics["rmse"]),
                    "metrics": {"train": train_metrics, "val": val_metrics, "test": test_metrics},
                    "per_class": {"train": {}, "val": {}, "test": {}},
                }
        else:
            self.bef_edit_results = self._safe_seed_test()
        print(f"Before edit results while loading model: {self.bef_edit_results['overall']}")
        return self.model, self.train_data, self.whole_data, self.num_features, self.num_classes

    def evaluate_before_edit(self) -> Tuple[float, float, float]:
        logger.info("Evaluating model before editing")
        self.bef_edit_results = self._safe_seed_test()
        bef_edit_train_acc, bef_edit_valid_acc, bef_edit_test_acc = self.bef_edit_results["overall"]
        logger.info(
            f"Before edit - Train acc: {bef_edit_train_acc:.4f}, Valid acc: {bef_edit_valid_acc:.4f}, Test acc: {bef_edit_test_acc:.4f}"
        )
        return self.bef_edit_results

    def fine_tune_if_needed(self) -> Optional[Tuple[float, float, float]]:
        if '_MLP' in self.config['pipeline_params']['model_name']:
            logger.info("Fine-tuning GNN+MLP model")
            return finetune_gnn_mlp(self.config, self.model, self.whole_data, self.train_data)
        return None

    @abstractmethod
    def select_edit_targets(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Must be implemented by child classes to select edit targets according to their strategy.
        Accepts arbitrary keyword args for editor-specific selection controls (e.g., strategy).
        """
        pass

    @abstractmethod
    def edit_model(self, **kwargs) -> List[List[Any]]:
        "Need to be implemented by the child class"
        pass

    def run_editing_experiment(
        self,
        num_targets: Optional[int] = None,
        max_num_step: Optional[int] = None,
        select_kwargs: Optional[Dict[str, Any]] = None,
        edit_kwargs: Optional[Dict[str, Any]] = None,
        data_override: Optional[Data] = None
    ) -> Tuple[List[List[Any]], Dict[str, Any]]:
        logger.info(f"Starting editing experiment with {self.__class__.__name__}")
        self.load_model_and_data(data_override=data_override)
        self.evaluate_before_edit()
        self.fine_tune_if_needed()
        # selection kwargs can specify num_targets/strategy/etc.
        sel_kwargs = dict(select_kwargs or {})
        if num_targets is not None:
            sel_kwargs.setdefault('num_targets', num_targets)
        node_idx_2flip, flipped_label = self.select_edit_targets(**sel_kwargs)
        if max_num_step is None:
            max_num_step = self.config['pipeline_params']['max_num_edit_steps']
        logger.info(f"Starting editing with {self.__class__.__name__}")
        ekw = dict(max_num_step=max_num_step)
        if edit_kwargs:
            ekw.update(edit_kwargs)
        ekw['node_idx_2flip'] = node_idx_2flip
        ekw['flipped_label'] = flipped_label
        raw_results = self.edit_model(**ekw)
        self.attach_edit_checkpoint_artifacts(None)
        raw_results = process_edit_results(self.bef_edit_results, raw_results)
        processed_results = process_raw_exp_results(raw_results)
        self.visualize_results(node_idx_2flip)
        logger.info(f"Editing experiment completed with {self.__class__.__name__}")
        return raw_results, processed_results

    def visualize_results(self, node_idx_2flip: torch.Tensor, suffix: str = ""):
        logger.info("Generating visualization plots")
        if suffix == "":
            suffix = f" ({self.__class__.__name__})"
        visualize_validation(self.config, self.model, self.whole_data, node_idx_2flip, suffix)

    def get_model(self) -> BaseModel:
        return self.model

    def get_data(self) -> Tuple[Data, Data]:
        return self.train_data, self.whole_data

    def get_config(self) -> Dict[str, Any]:
        return self.config

    def update_config(self, updates: Dict[str, Any]):
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        self.config = deep_update(self.config, updates)
        logger.info(f"Configuration updated: {updates}")


