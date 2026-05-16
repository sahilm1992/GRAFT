#!/usr/bin/env bash

method="seed_gnn"
dataset="$1"
output_dir_root="$2"
dataset_dir="$3"
num_samples="${4:-50}"
shift 4 || true

if [ -z "$dataset" ] || [ -z "$output_dir_root" ] || [ -z "$dataset_dir" ]; then
    echo "Usage: $0 <dataset> <output_dir_root> <dataset_dir> [num_samples] [extra args...]"
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINES_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd -- "${PIPELINES_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PY_SCRIPT="${PIPELINES_ROOT}/py_scripts/compute_sif.py"

for model in gcn sage gin gat; do
    echo "Computing SIF for ${dataset} with ${model} (num_samples=${num_samples})"
    "${PYTHON_BIN}" "${PY_SCRIPT}" \
        --exp_desc="sif_${model}_${method}" \
        --pipeline_config_dir="${PROJECT_ROOT}/seed-gnn/config/pipeline_config/${method}/${model}/${dataset}.json" \
        --eval_config_dir="${PROJECT_ROOT}/seed-gnn/config/eval_config/edit_gnn/${dataset}.json" \
        --output_folder_dir="${output_dir_root}/results/${method}/${model}/${dataset}/" \
        --pretrain_output_dir="${output_dir_root}/edit_ckpts" \
        --dataset_dir="${dataset_dir}" \
        --num_samples="${num_samples}" \
        "$@"
done
