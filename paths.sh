#!/usr/bin/env bash
# Central path defaults for GRAFT. Source from any driver script, e.g.:
#   REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"   # editing_pipelines/ -> repo root
#   # shellcheck source=../paths.sh
#   source "${REPO}/paths.sh"

_paths_dir="$(CDPATH="" cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH_TO_GRAFT="${PATH_TO_GRAFT:-$_paths_dir}"

export PATH_TO_DATA="${PATH_TO_DATA:-/home/model_editing/data}"

export DATASET_DIR="${DATASET_DIR:-${PATH_TO_DATA}/seed_gnn_data/dataset}"
export PRETRAIN_DIR="${PRETRAIN_DIR:-${PATH_TO_DATA}/seed_gnn_data/edit_ckpts}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${PATH_TO_DATA}/editing_pipelines}"
export ABLATIONS_DIR="${ABLATIONS_DIR:-${PATH_TO_DATA}/ablations}"
export OUTPUT_DIR_ROOT="${OUTPUT_DIR_ROOT:-${PATH_TO_DATA}/seed_gnn_data}"
export GRAPHLAND_DATASET_DIR="${GRAPHLAND_DATASET_DIR:-${PATH_TO_DATA}/graphland/graphland}"
