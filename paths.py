"""Single place for checkout root (PATH_TO_GRAFT) and data-root layout (PATH_TO_DATA).

Shell drivers should ``source paths.sh``, which mirrors these defaults as exported env vars.

All dataset / checkpoint / pipeline artifacts are assumed to live under ``PATH_TO_DATA`` with
the subdirectory layout documented in ``paths.sh``.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT_FALLBACK = Path(__file__).resolve().parent


def path_to_graft() -> Path:
    return Path(os.environ.get("PATH_TO_GRAFT", str(_REPO_ROOT_FALLBACK))).expanduser().resolve()


def path_to_data() -> Path:
    return Path(os.environ.get("PATH_TO_DATA", "/home/model_editing/data")).expanduser().resolve()


def seed_gnn_dir() -> Path:
    return path_to_graft() / "seed-gnn"


def dataset_dir_default() -> Path:
    return path_to_data() / "seed_gnn_data" / "dataset"


def pretrain_edit_ckpts_dir_default() -> Path:
    return path_to_data() / "seed_gnn_data" / "edit_ckpts"


def seed_gnn_data_root_default() -> Path:
    return path_to_data() / "seed_gnn_data"


def seed_gnn_results_dir_default() -> Path:
    return path_to_data() / "seed_gnn_data" / "results" / "seed_gnn"


def editing_pipelines_root_default() -> Path:
    return path_to_data() / "editing_pipelines"


def time_ablation_root_default() -> Path:
    return path_to_data() / "editing_pipelines" / "time_ablation"


def ablations_dir_default() -> Path:
    return path_to_data() / "ablations"


def graphland_dataset_dir_default() -> Path:
    return path_to_data() / "graphland" / "graphland"
