#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../paths.sh
source "${REPO}/paths.sh"

# Defaults
METHOD="${1:-leastsquares}"
DATASET="${2:-pokec}"
MODEL="${3:-GCN_MLP}"
SEED="${4:-42}"
NUM_TARGETS="${5:-1000}"
MAX_STEPS="${6:-50}"
TARGET_STRATEGY="${7:-}"
# Least-squares strategies: confidence | sensitivity_mean | sensitivity_wtd_mean | sens_pr | sens_pr_graphaware | sens_divrank | sens_divrank_graphaware | sens_subspace_retention | sens_subspace_retention_pr | sens_subspace_retention_pr_graphaware | sens_subspace_retention_divrank | sens_subspace_retention_divrank_graphaware | sens_subspace | sens_subspace_pr | sens_subspace_pr_graphaware | sens_subspace_divrank | sens_subspace_divrank_graphaware
LS_STRATEGY="${8:-sensitivity_mean}"

SEED_GNN="${PATH_TO_GRAFT}/seed-gnn"
DATASET_DIR="${9:-${DATASET_DIR}}"
PRETRAIN_DIR="${10:-${PRETRAIN_DIR}}"
OUTPUT_DIR="${11:-${OUTPUT_ROOT}/${METHOD}/${DATASET}/${MODEL}}"

export PYTHONPATH="${PATH_TO_GRAFT}:${SEED_GNN}:${PYTHONPATH:-}"

python "${PATH_TO_GRAFT}/editing_pipelines/run_edit.py" \
  --method "$METHOD" \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --seed "$SEED" \
  --num-targets "$NUM_TARGETS" \
  --max-steps "$MAX_STEPS" \
  ${TARGET_STRATEGY:+--strategy "$TARGET_STRATEGY"} \
  --leastsquares-strategy "$LS_STRATEGY" \
  --dataset-dir "$DATASET_DIR" \
  --pretrain-dir "$PRETRAIN_DIR" \
  --output-dir "$OUTPUT_DIR" \
  ${LAMBDA_REG:+--lambda-reg "$LAMBDA_REG"} \
  ${TOP_FRACTION:+--top-fraction "$TOP_FRACTION"} \
  ${NUM_LAYERS:+--num-layers "$NUM_LAYERS"} \
  ${GAMMA_RETAIN:+--gamma-retain "$GAMMA_RETAIN"} \
  ${FT_EPOCHS:+--ft-epochs "$FT_EPOCHS"} \
  ${FT_LR:+--ft-lr "$FT_LR"} \
  ${PR_ALPHA:+--pr-alpha "$PR_ALPHA"} \
  ${RANK_MIX_TAU:+--rank-mix-tau "$RANK_MIX_TAU"}

