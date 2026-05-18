#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../paths.sh
source "${REPO}/paths.sh"

SEED_GNN="${PATH_TO_GRAFT}/seed-gnn"

# Positional/default args
# 1) DATASET, 2) MODEL, 3) FEATURE (name or index), 4) BINS, 5) SPLIT
DATASET="${1:-income}"
MODEL="${2:-GCN_MLP}"
FEATURE_ARG="${3:-fnlwgt}"
BINS="${4:-30}"
SPLIT="${5:-whole}"

# Derived dirs (can be overridden via env)
DATASET_DIR="${DATASET_DIR:-$PATH_TO_DATA/seed_gnn_data/dataset}"
PRETRAIN_DIR="${PRETRAIN_DIR:-$PATH_TO_DATA/seed_gnn_data/edit_ckpts}"
OUTPUT_DIR="${OUTPUT_DIR:-$PATH_TO_DATA/editing_pipelines/visualizations}"

export PYTHONPATH="$PATH_TO_GRAFT:$SEED_GNN:${PYTHONPATH:-}"

# Determine whether FEATURE_ARG is an index (all digits) or a name
FEATURE_FLAG=()
if [[ "$FEATURE_ARG" =~ ^[0-9]+$ ]]; then
  FEATURE_FLAG=(--feature-index "$FEATURE_ARG")
else
  FEATURE_FLAG=(--feature-name "$FEATURE_ARG")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" python "$PATH_TO_GRAFT/editing_pipelines/py_scripts/visualize_dataset.py" \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --dataset-dir "$DATASET_DIR" \
  --pretrain-dir "$PRETRAIN_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --bins "$BINS" \
  --split "$SPLIT" \
  "${FEATURE_FLAG[@]}"

echo "Saved plots under: $OUTPUT_DIR/dataset_feature_distributions/${DATASET}_${MODEL}"


