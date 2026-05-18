#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../paths.sh
source "${REPO}/paths.sh"

SEED_GNN="${PATH_TO_GRAFT}/seed-gnn"

# Positional/default args
DATASET="${1:-pokec}"
MODEL="${2:-GCN_MLP}"
N="${3:-2}"
EPOCHS="${4:-200}"
METHOD="${5:-leastsquares}"

# Derived dirs (can be overridden via env or extra args if needed)
DATASET_DIR="${DATASET_DIR:-$PATH_TO_DATA/seed_gnn_data/dataset}"
PRETRAIN_DIR="${PRETRAIN_DIR:-$PATH_TO_DATA/seed_gnn_data/edit_ckpts}"
OUTPUT_DIR="${OUTPUT_DIR:-$PATH_TO_DATA/editing_pipelines/stat_significance}"
STRATEGY="${STRATEGY:-}"
# Least-squares strategies: confidence | sensitivity_mean | sensitivity_wtd_mean | sens_pr | sens_pr_graphaware
LS_STRATEGY="${LS_STRATEGY:-sensitivity_wtd_mean}"

export PYTHONPATH="$PATH_TO_GRAFT:$SEED_GNN:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python "$PATH_TO_GRAFT/editing_pipelines/py_scripts/test_significance.py" \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --N "$N" \
  --epochs "$EPOCHS" \
  --methods "$METHOD" \
  --strategy "$STRATEGY" \
  --leastsquares-strategy "$LS_STRATEGY" \
  --dataset-dir "$DATASET_DIR" \
  --pretrain-dir "$PRETRAIN_DIR" \
  --output-dir "$OUTPUT_DIR" \
  # --plot \