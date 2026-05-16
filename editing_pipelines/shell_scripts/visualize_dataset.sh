#!/usr/bin/env bash
set -euo pipefail

# Root paths
ROOT="/home/model_editing/gnn-editing-exploration"
SEED_GNN="$ROOT/seed-gnn"
DATA_ROOT="/home/model_editing/data"

# Positional/default args
# 1) DATASET, 2) MODEL, 3) FEATURE (name or index), 4) BINS, 5) SPLIT
DATASET="${1:-income}"
MODEL="${2:-GCN_MLP}"
FEATURE_ARG="${3:-fnlwgt}"
BINS="${4:-30}"
SPLIT="${5:-whole}"

# Derived dirs (can be overridden via env)
DATASET_DIR="${DATASET_DIR:-$DATA_ROOT/seed_gnn_data/dataset}"
PRETRAIN_DIR="${PRETRAIN_DIR:-$DATA_ROOT/seed_gnn_data/edit_ckpts}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/editing_pipelines/visualizations}"

export PYTHONPATH="$ROOT:$SEED_GNN:${PYTHONPATH:-}"

# Determine whether FEATURE_ARG is an index (all digits) or a name
FEATURE_FLAG=()
if [[ "$FEATURE_ARG" =~ ^[0-9]+$ ]]; then
  FEATURE_FLAG=(--feature-index "$FEATURE_ARG")
else
  FEATURE_FLAG=(--feature-name "$FEATURE_ARG")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" python "$ROOT/editing_pipelines/py_scripts/visualize_dataset.py" \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --dataset-dir "$DATASET_DIR" \
  --pretrain-dir "$PRETRAIN_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --bins "$BINS" \
  --split "$SPLIT" \
  "${FEATURE_FLAG[@]}"

echo "Saved plots under: $OUTPUT_DIR/dataset_feature_distributions/${DATASET}_${MODEL}"


