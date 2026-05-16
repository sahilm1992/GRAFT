import os
import re
from pathlib import Path
from typing import List, Tuple, Optional
import torch.nn as nn
import torch
from torch_geometric.data import Data

# Import from the seed-gnn directory
import sys
sys.path.append('/home/model_editing/gnn-editing-exploration/seed-gnn')
import models as models
from data import get_data, prepare_dataset
from constants import SEED
from main_utils import set_seeds_all

from .gat_neighbor_eval import estimate_num_edges, should_use_gat_neighbor_loader


def get_device():
    force_cpu = os.getenv("FORCE_CPU", "0") == "1"
    has_visible_cuda = os.getenv("CUDA_VISIBLE_DEVICES", "") != ""
    if torch.cuda.is_available() and has_visible_cuda and not force_cpu:
        return torch.device("cuda")
    return torch.device("cpu")



def get_optimizer(model_config, model, pretrain: bool = False):
    if model_config['optim'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=model_config['pretrain_lr'] if pretrain else model_config['edit_lr']
        )
    elif model_config['optim'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=model_config['pretrain_lr'] if pretrain else model_config['edit_lr']
        )
    else:
        raise NotImplementedError
    return optimizer


def sorted_checkpoints(checkpoint_prefix, best_model_checkpoint, output_dir=None) -> List[str]:
    ordering_and_checkpoint_path = []
    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{checkpoint_prefix}_*")]
    for path in glob_checkpoints:
        import re as _re
        regex_match = _re.match(f".*{checkpoint_prefix}_([0-9]+)_seed([0-9]+)", path) # Match epoch and seed
        if regex_match and regex_match.groups():
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))
    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    if best_model_checkpoint is not None:
        best_model_index = checkpoints_sorted.index(str(Path(best_model_checkpoint)))
        checkpoints_sorted[best_model_index], checkpoints_sorted[-1] = (
            checkpoints_sorted[-1],
            checkpoints_sorted[best_model_index],
        )
    return checkpoints_sorted


def save_model(model, save_path, checkpoint_prefix: str, epoch: int):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Include seed in the filename
    best_model_checkpoint = os.path.join(save_path, f'{checkpoint_prefix}_{epoch}_seed{set_seeds_all.seed}.pt') # Use actual seed
    torch.save(model.state_dict(), best_model_checkpoint)
    checkpoints_sorted = sorted_checkpoints(checkpoint_prefix, best_model_checkpoint, save_path)
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - 1)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        os.remove(checkpoint)
    return best_model_checkpoint


def load_model(config, data_override: Optional[Data] = None):
    set_seeds_all(config["management"]["seed"]) # Use seed from config
    MODEL_FAMILY = getattr(models, config['pipeline_params']['model_name'])
    if data_override is not None:
        data = data_override
        num_features = int(data.x.size(1))
        task_type = str(getattr(data, "task_type", "")).lower()
        is_regression = (task_type == "regression") or bool(data.y.dtype.is_floating_point)
        num_classes = 1 if is_regression else int(data.y.max().item() + 1)
    else:
        data, num_features, num_classes = get_data(
            config['management']['dataset_dir'],
            config['eval_params']['dataset'],
            config
        )
    pretrain_root = Path(config['management']['pretrain_output_dir'])
    dataset_name = config['eval_params']['dataset']
    feature_variant = config['pipeline_params'].get("feature_variant", "full_features")
    if dataset_name not in pretrain_root.parts:
        save_path = pretrain_root / dataset_name
    else:
        save_path = pretrain_root
    save_path = Path(save_path)

    def dir_has_ckpt(path: Path) -> bool:
        return path.exists() and path.is_dir() and any(path.glob("*.pt"))

    def descend_to_ckpts(path: Path) -> Path:
        current = path
        visited = set()
        while current.exists() and current.is_dir():
            if dir_has_ckpt(current):
                break
            key = current.resolve()
            if key in visited:
                break
            visited.add(key)
            dataset_child = current / dataset_name
            if dataset_child.exists():
                current = dataset_child
                continue
            subdirs = [p for p in current.iterdir() if p.is_dir()]
            if len(subdirs) == 1:
                current = subdirs[0]
                continue
            break
        return current

    candidate_dirs = [
        save_path,
        save_path / dataset_name,
        save_path / feature_variant,
        save_path / dataset_name / feature_variant,
        save_path / feature_variant / dataset_name,
        pretrain_root / dataset_name / feature_variant,
        pretrain_root / dataset_name,
    ]

    resolved_ckpt_dir = None
    for cand in candidate_dirs:
        resolved = descend_to_ckpts(cand)
        if dir_has_ckpt(resolved):
            resolved_ckpt_dir = resolved
            break

    if resolved_ckpt_dir is None:
        resolved_ckpt_dir = save_path

    def pick_checkpoint_file(dir_path: Path, model_prefix: str, model_name: str, seed: int) -> Path:
        dir_path = Path(dir_path)
        if dir_path.is_file() and dir_path.suffix == ".pt":
            return dir_path
        if not dir_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {dir_path}")

        # Build candidate list by model-specific prefixes only.
        candidate_prefixes = [model_prefix]
        if model_name != model_prefix:
            candidate_prefixes.append(model_name)

        pt_files = []
        for prefix in candidate_prefixes:
            # Preferred naming conventions.
            pt_files.extend(dir_path.glob(f"{prefix}_*seed{seed}_*.pt"))
            pt_files.extend(dir_path.glob(f"{prefix}_*seed{seed}.pt"))
            # Legacy naming convention.
            pt_files.extend(dir_path.glob(f"{prefix}_*_{seed}.pt"))

        # Deduplicate while preserving deterministic order.
        pt_files = sorted(set(pt_files))
        if not pt_files:
            raise FileNotFoundError(
                f"No checkpoint files found under {dir_path} for model '{model_name}' "
                f"(prefix '{model_prefix}') and seed {seed}."
            )

        def filter_with_prefix(prefix: str, files):
            token = f"{prefix}_"
            results = []
            for f in files:
                stem = f.stem
                if not stem.startswith(token):
                    continue
                if not prefix.endswith("_MLP") and "_MLP" in stem[len(prefix):]:
                    continue
                results.append(f)
            return results

        filtered = filter_with_prefix(model_prefix, pt_files)
        if not filtered:
            filtered = filter_with_prefix(model_name, pt_files)
        
        # If we have multiple files and a seed, try to narrow down by seed string in filename
        if len(filtered) > 1 and seed is not None:
            seed_str = f"seed{seed}"
            seed_filtered = [f for f in filtered if seed_str in f.stem]
            if seed_filtered:
                filtered = seed_filtered

        if not filtered:
            raise FileNotFoundError(
                f"Found checkpoint files in {dir_path}, but none match the requested model "
                f"'{model_name}'/prefix '{model_prefix}'. Candidates: {[p.name for p in pt_files]}"
            )

        def sort_key(path: Path):
            stem = path.stem
            try:
                # Extract epoch (the number after the last underscore before .pt, or after seed)
                # Filenames: GAT_layers2_seed10_78.pt or GAT_layers2_78_seed10.pt
                match = re.search(r'_(\d+)(?:_seed\d+)?$', stem)
                if not match:
                    match = re.search(r'seed\d+_(\d+)$', stem)
                
                if match:
                    return int(match.group(1))
                return stem
            except Exception:
                return stem

        filtered.sort(key=sort_key)
        return filtered[-1]

    model_name = config['pipeline_params']['model_name']
    num_layers = config['pipeline_params'].get("architecture", {}).get("num_layers")
    model_prefix = f"{model_name}_layers{num_layers}" if num_layers is not None else model_name
    seed_val = config["management"]["seed"] # Get seed from config
    ckpt_file = pick_checkpoint_file(resolved_ckpt_dir, model_prefix, model_name, seed_val)

    def _load_state_dict_cpu(path: Path):
        try:
            return torch.load(str(path), map_location="cpu")
        except Exception:
            return None

    def infer_hidden_channels_from_ckpt(path: Path):
        """
        Infer hidden width from common first-layer tensors in checkpoint.
        Works across GCN/SAGE/GAT/GIN and *_MLP variants.
        """
        state_dict = _load_state_dict_cpu(path)
        if not isinstance(state_dict, dict):
            return None
        # Prefer tensors that directly encode hidden_channels (not 2*hidden).
        direct_candidates = [
            # GIN-specific direct hidden width
            "GIN.lin1.bias",
            "GIN.lin1.weight",
            "model.GIN.lin1.bias",
            "model.GIN.lin1.weight",
            # MLP fallback for *_MLP families
            "MLP.lins.0.bias",
            "MLP.lins.0.weight",
            "model.MLP.lins.0.bias",
            "model.MLP.lins.0.weight",
            # Other backbones
            "GCN.convs.0.bias",
            "GCN.convs.0.lin.weight",
            "SAGE.convs.0.lin_l.bias",
            "SAGE.convs.0.lin_l.weight",
            "GAT.convs.0.bias",
            "GAT.convs.0.lin.weight",
            "model.GCN.convs.0.bias",
            "model.GCN.convs.0.lin.weight",
            "model.SAGE.convs.0.lin_l.bias",
            "model.SAGE.convs.0.lin_l.weight",
            "model.GAT.convs.0.bias",
            "model.GAT.convs.0.lin.weight",
        ]
        for key in direct_candidates:
            if key not in state_dict:
                continue
            tensor = state_dict[key]
            if not hasattr(tensor, "shape"):
                continue
            if tensor.ndim == 1 and tensor.shape[0] > 0:
                return int(tensor.shape[0])
            if tensor.ndim >= 2 and tensor.shape[0] > 0:
                return int(tensor.shape[0])

        # GIN first MLP layer is typically 2*hidden -> hidden; convert to hidden.
        gin_doubled_candidates = [
            "GIN.convs.0.nn.0.bias",
            "GIN.convs.0.nn.0.weight",
            "model.GIN.convs.0.nn.0.bias",
            "model.GIN.convs.0.nn.0.weight",
        ]
        for key in gin_doubled_candidates:
            if key not in state_dict:
                continue
            tensor = state_dict[key]
            if not hasattr(tensor, "shape"):
                continue
            first_dim = int(tensor.shape[0]) if tensor.shape and tensor.shape[0] > 0 else None
            if first_dim is None:
                continue
            if first_dim % 2 == 0:
                return first_dim // 2
            return first_dim

        candidates = [
            # Last-resort generic hints
            "GIN.batch_norms.0.bias",
            "GIN.batch_norms.0.weight",
            "model.GIN.batch_norms.0.bias",
            "model.GIN.batch_norms.0.weight",
        ]
        for key in candidates:
            if key not in state_dict:
                continue
            tensor = state_dict[key]
            if not hasattr(tensor, "shape"):
                continue
            if tensor.ndim == 1 and tensor.shape[0] > 0:
                return int(tensor.shape[0])
            if tensor.ndim >= 2 and tensor.shape[0] > 0:
                return int(tensor.shape[0])
        return None

    def infer_gat_optimized_from_ckpt(path: Path, hidden_channels: int):
        """
        Infer whether checkpoint was trained with gat_optimized=True.
        Heuristic: first GAT conv bias size equals hidden_channels in optimized mode,
        and equals hidden_channels * heads in legacy concatenated mode.
        """
        try:
            state_dict = _load_state_dict_cpu(path)
            if not isinstance(state_dict, dict):
                return None
            candidates = [
                "GAT.convs.0.bias",
                "model.GAT.convs.0.bias",
                "convs.0.bias",
                "model.convs.0.bias",
            ]
            for key in candidates:
                if key in state_dict:
                    bias = state_dict[key]
                    if getattr(bias, "ndim", None) == 1:
                        return int(bias.shape[0]) == int(hidden_channels)
        except Exception:
            pass
        return None

    def infer_num_conv_layers_from_ckpt(path: Path, backbone_prefix: str) -> Optional[int]:
        """Infer GNN depth from backbone keys like GAT.convs.{i}.* in the checkpoint."""
        state_dict = _load_state_dict_cpu(path)
        if not isinstance(state_dict, dict):
            return None
        pat = re.compile(rf"^(?:model\.)?{re.escape(backbone_prefix)}\.convs\.(\d+)\.")
        max_idx = -1
        for k in state_dict.keys():
            m = pat.match(str(k))
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        if max_idx < 0:
            return None
        return max_idx + 1

    arch_cfg = dict(config['pipeline_params']['architecture'])

    # Align width with checkpoint for all tasks (classification used to skip this, which
    # breaks *_MLP loading when the JSON/run defaults disagree with pretraining).
    inferred_hidden = infer_hidden_channels_from_ckpt(ckpt_file)
    if inferred_hidden is not None:
        configured_hidden = arch_cfg.get("hidden_channels")
        if configured_hidden is None or int(configured_hidden) != int(inferred_hidden):
            print(
                f"[load_model] overriding hidden_channels from {configured_hidden} "
                f"to {inferred_hidden} based on checkpoint {ckpt_file.name}"
            )
            arch_cfg["hidden_channels"] = int(inferred_hidden)

    # Align depth with checkpoint so e.g. num_layers=1 in config does not collapse GAT
    # to a single output layer when weights are from a multi-layer pretrained model.
    backbone_prefix = model_name.replace("_MLP", "") if "_MLP" in model_name else model_name
    inferred_layers = infer_num_conv_layers_from_ckpt(ckpt_file, backbone_prefix)
    if inferred_layers is not None:
        configured_layers = arch_cfg.get("num_layers")
        if configured_layers is None or int(configured_layers) != int(inferred_layers):
            print(
                f"[load_model] overriding num_layers from {configured_layers} "
                f"to {inferred_layers} based on checkpoint {ckpt_file.name}"
            )
            arch_cfg["num_layers"] = int(inferred_layers)

    # GAT checkpoints can be trained in two layouts:
    # - optimized (concat=False) where first-layer bias is hidden_channels
    # - legacy concatenated where first-layer bias is hidden_channels * heads
    # Infer this from checkpoint regardless of task type to avoid shape mismatches.
    if model_name.startswith("GAT"):
        hidden_channels = arch_cfg.get("hidden_channels")
        inferred = infer_gat_optimized_from_ckpt(ckpt_file, hidden_channels)
        if inferred is not None:
            arch_cfg["gat_optimized"] = inferred
            print(f"[load_model] inferred gat_optimized={inferred} from checkpoint {ckpt_file.name}")

    # Keep nested config consistent with what we pass into from_pretrained.
    config["pipeline_params"]["architecture"].update(arch_cfg)

    model = MODEL_FAMILY.from_pretrained(
        in_channels=num_features,
        out_channels=num_classes,
        saved_ckpt_path=str(ckpt_file),
        seed=seed_val,
        **arch_cfg
    )
    model.to(get_device())
    train_data, whole_data = prepare_dataset(config['pipeline_params'], data, remove_edge_index=False)
    del data
    # Keep *_MLP models in backbone-only mode unless an editor explicitly opts into MLP usage.
    from edit_gnn.utils import test as seed_test  # type: ignore
    try:
        if '_MLP' in config['pipeline_params']['model_name']:
            if hasattr(model, "freeze_module"):
                model.freeze_module(train=True)
            elif hasattr(model, "mlp_freezed"):
                model.mlp_freezed = True
            if hasattr(model, "MLP") and hasattr(model, "freeze_layer"):
                model.freeze_layer(model.MLP, freeze=True)
            print("[load_model] kept *_MLP model in backbone-only mode (mlp_freezed=True).")
    except Exception as e:
        print(f"[WARN] Failed to enforce backbone-only *_MLP mode in loader: {e}")

    def _is_oom_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return isinstance(exc, torch.cuda.OutOfMemoryError) or ("out of memory" in msg)

    is_regression = bool(getattr(whole_data, "task_type", "") == "regression" or whole_data.y.dtype.is_floating_point)
    model_name = str(config["pipeline_params"].get("model_name", ""))
    dataset_name = str(config["eval_params"].get("dataset", ""))
    num_edges = estimate_num_edges(whole_data)
    # Same rule as everywhere else: skip diagnostic full-graph seed_test only when
    # should_use_gat_neighbor_loader (large graph or multi-layer GAT backbone).
    should_skip_seed_test = (
        (not is_regression)
        and model_name.startswith("GAT")
        and should_use_gat_neighbor_loader(model, whole_data)
    )
    try:
        if is_regression:
            from edit_gnn.utils import grab_input  # type: ignore
            with torch.no_grad():
                pred = model(**grab_input(whole_data))
                if pred.dim() == 2 and pred.size(-1) == 1:
                    pred = pred.squeeze(-1)
                y = whole_data.y.to(pred.dtype)

                def _rmse(mask):
                    if mask is None:
                        return float("nan")
                    vals = (pred[mask] - y[mask]) ** 2
                    if vals.numel() == 0:
                        return float("nan")
                    return float(torch.sqrt(vals.mean()).item())

                print(f"Final Base RMSE: {(_rmse(whole_data.train_mask), _rmse(whole_data.val_mask), _rmse(whole_data.test_mask))}")
        else:
            if should_skip_seed_test:
                print(
                    f"[WARN] Skipping seed_test base evaluation for {dataset_name}/{model_name} "
                    f"(edges={num_edges}; multi-layer or large-graph GAT; see gat_neighbor_eval) "
                    f"to avoid full-graph GAT OOM."
                )
                return model, train_data, whole_data, num_features, num_classes
            result_print = seed_test(model, whole_data)
            print(f"Final Base Accuracies: {result_print['overall']}")
    except Exception as exc:
        # Base-metric logging is diagnostic only; do not block editing runs on OOM.
        if _is_oom_error(exc):
            print(f"[WARN] Skipping base metric evaluation due to OOM: {exc}")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        else:
            raise
    return model, train_data, whole_data, num_features, num_classes


# -----------------------------
# LeastSquaresEditor Utilities
# -----------------------------
def detect_backbone_module(model) -> Tuple[Optional[nn.Module], Optional[str]]:
    """
    Detect the backbone submodule in *_MLP models generically (GCN/SAGE/GAT/GIN).
    Returns (module, name) or (None, None) if not found.
    """
    print("Detecting backbone module...")
    # Common explicit attribute names
    for name in ["GCN", "SAGE", "GAT", "GIN"]:
        print(model.__class__.__name__)
        if model.__class__.__name__.startswith(name):

            backbone = getattr(model, name)
            bb_name = name
            return backbone, bb_name
    # Fallback: find any attribute with 'convs' ModuleList
    try:
        for name, value in model.__dict__.items():
            if isinstance(value, nn.Module) and hasattr(value, "convs"):
                backbone = value
                bb_name = name
                return backbone, bb_name
    except Exception:
        pass
    return None, None

def log_forward_mode(model, bb_name, logger):
    if hasattr(model, "mlp_freezed"):
        path = f"{bb_name} only" if model.mlp_freezed else f"{bb_name} + MLP"
        logger.info(f"[Model Forward Mode] Active path = {path}")
    else:
        logger.info(f"[Model Forward Mode] (No mlp_freezed attribute, model is not {bb_name}_MLP)")
