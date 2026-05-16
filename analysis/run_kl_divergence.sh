#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-pokec}"
model="${2:-gat}"
feature="${3:-AGE}"
seed="${4:-42}"
output_root="${OUTPUT_ROOT:-/home/model_editing/data/editing_pipelines}"
dataset_dir="${DATASET_DIR:-/home/model_editing/data/seed_gnn_data/dataset}"
pretrain_dir="${PRETRAIN_DIR:-/home/model_editing/data/seed_gnn_data/edit_ckpts}"
shift 4 || true

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PY_SCRIPT="${SCRIPT_DIR}/kl_divergence_analysis.py"

output_dir="${output_root}/analysis/kl_divergence/${dataset}/${model}"

echo "Dataset: ${dataset}"
echo "Model: ${model}"
echo "Feature: ${feature}"
echo "Seed: ${seed}"
echo "Output: ${output_dir}"
echo "Dataset dir: ${dataset_dir}"
echo "Pretrain dir: ${pretrain_dir}"

"${PYTHON_BIN}" "${PY_SCRIPT}" \
    --exp_desc="kl_${dataset}_${model}_${feature}" \
    --pipeline_config_dir="${PROJECT_ROOT}/seed-gnn/config/pipeline_config/seed_gnn/${model}/${dataset}.json" \
    --eval_config_dir="${PROJECT_ROOT}/seed-gnn/config/eval_config/edit_gnn/${dataset}.json" \
    --output_folder_dir="${output_dir}/" \
    --pretrain_output_dir="${pretrain_dir}" \
    --dataset_dir="${dataset_dir}" \
    --seed="${seed}" \
    --feature-name="${feature}" \
    "$@"
