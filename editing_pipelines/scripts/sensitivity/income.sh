#!/usr/bin/env bash

method="seed_gnn"
dataset="income"
output_dir_root=$1
dataset_dir=$2
ablation=$3 # "none" or feature name like "race"
shift 3

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINES_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd -- "${PIPELINES_ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PY_SCRIPT="${PIPELINES_ROOT}/py_scripts/analyze_feature_sensitivity.py"

# Default perturb feature for income
perturb_feat="race"

# Handle ablation flags
if [ "$ablation" != "none" ] && [ ! -z "$ablation" ]; then
    variant="no_${ablation}"
    ablation_args="--use_feature_ablation --feature_variant ${variant} --drop_features ${ablation}"
else
    ablation_args=""
fi

for model in gcn sage gin gat; do
    echo "Running sensitivity analysis for ${dataset} with ${model} (Ablation: ${ablation})"
    "${PYTHON_BIN}" "${PY_SCRIPT}" \
        --exp_desc="sensitivity_${model}_${method}" \
        --pipeline_config_dir="${PROJECT_ROOT}/seed-gnn/config/pipeline_config/${method}/${model}/${dataset}.json" \
        --eval_config_dir="${PROJECT_ROOT}/seed-gnn/config/eval_config/edit_gnn/${dataset}.json" \
        --output_folder_dir="${output_dir_root}/results/${method}/${model}/${dataset}/" \
        --pretrain_output_dir="${output_dir_root}/edit_ckpts" \
        --dataset_dir="${dataset_dir}" \
        --perturb_feature="${perturb_feat}" \
        --num_seeds=100 \
        --prob_mode="positive" \
        ${ablation_args} \
        "$@"
done


