#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash ./run_time_ablation.sh <dataset> <model> <method>
#
# Example:
#   bash ./run_time_ablation.sh pokec GCN_MLP leastsquares

if [[ $# -lt 3 ]]; then
  echo "Usage: bash ./run_time_ablation.sh <dataset> <model> <method>"
  echo "Methods: leastsquares | finetune | egnn | seedgnn | seed_gnn"
  exit 1
fi

DATASET="$1"
MODEL="$2"
METHOD_RAW="$3"

case "${METHOD_RAW}" in
  seedgnn) METHOD="seed_gnn" ;;
  seed_gnn|egnn|finetune|leastsquares) METHOD="${METHOD_RAW}" ;;
  *)
    echo "Unsupported method: ${METHOD_RAW}"
    echo "Supported: leastsquares | finetune | egnn | seedgnn | seed_gnn"
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EDIT_SH="${SCRIPT_DIR}/run_edit.sh"

DATASET_DIR="${DATASET_DIR:-/home/model_editing/data/seed_gnn_data/dataset}"
PRETRAIN_DIR="${PRETRAIN_DIR:-/home/model_editing/data/seed_gnn_data/edit_ckpts}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/model_editing/data/editing_pipelines/time_ablation}"

SEED=42
LAYERS=(2 3 5)
LS_STRATEGY="sens_subspace_retention_pr_graphaware"

# Standard hyperparameters used across your recent runs.
TOP_FRACTION="${TOP_FRACTION:-0.25}"
LAMBDA_REG="${LAMBDA_REG:-0.1}"
PR_ALPHA="${PR_ALPHA:-0.85}"
RANK_MIX_TAU="${RANK_MIX_TAU:-0.75}"
GAMMA_RETAIN="${GAMMA_RETAIN:-1.0}"

NUM_TARGETS="${NUM_TARGETS:-1000}"
MAX_STEPS="${MAX_STEPS:-50}"

echo "========================================"
echo "Running time ablation"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL}"
echo "Method: ${METHOD}"
echo "Layers: ${LAYERS[*]}"
echo "Seed: ${SEED}"
echo "Strategy: ${LS_STRATEGY}"
echo "Output root: ${OUTPUT_ROOT}"
echo "========================================"

TOTAL_RUNS="${#LAYERS[@]}"
CURRENT_RUN=0

for num_layers in "${LAYERS[@]}"; do
  CURRENT_RUN=$((CURRENT_RUN + 1))
  output_dir="${OUTPUT_ROOT}/${METHOD}/${DATASET}/${MODEL}/layers${num_layers}"

  echo ""
  echo "----------------------------------------"
  echo "Progress: ${CURRENT_RUN} / ${TOTAL_RUNS}"
  echo "Layer: ${num_layers}"
  echo "Output: ${output_dir}"
  echo "----------------------------------------"

  export TOP_FRACTION
  export NUM_LAYERS="${num_layers}"
  export LAMBDA_REG
  export PR_ALPHA
  export RANK_MIX_TAU
  export GAMMA_RETAIN

  # Finetune-specific knobs (used only by finetune method).
  export FT_EPOCHS="${FT_EPOCHS:-20}"
  export FT_LR="${FT_LR:-0.001}"

  bash "${RUN_EDIT_SH}" \
    "${METHOD}" \
    "${DATASET}" \
    "${MODEL}" \
    "${SEED}" \
    "${NUM_TARGETS}" \
    "${MAX_STEPS}" \
    "" \
    "${LS_STRATEGY}" \
    "${DATASET_DIR}" \
    "${PRETRAIN_DIR}" \
    "${output_dir}"
done

echo ""
echo "Done: time ablation runs finished."

