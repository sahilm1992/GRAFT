#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to this script
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

RUN_EDIT_SH="${SCRIPT_DIR}/run_edit.sh"

DATASET_DIR="${DATASET_DIR:-/home/model_editing/data/seed_gnn_data/dataset}"
PRETRAIN_DIR="${PRETRAIN_DIR:-/home/model_editing/data/seed_gnn_data/edit_ckpts}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/model_editing/data/editing_pipelines}"

# DEFAULT_LAMBDAS=("0.001 0.01 0.1 1 10 100")
DEFAULT_LAMBDAS=("500 1000 10000")
IFS=' ' read -r -a LAMBDAS <<< "${LAMBDAS_OVERRIDE:-${DEFAULT_LAMBDAS[*]}}"

DEFAULT_TOP_FRACTIONS=("0.1 0.25 0.5 0.75 1")
# DEFAULT_TOP_FRACTIONS=("0.25")
IFS=' ' read -r -a TOP_FRACTIONS <<< "${TOP_FRACTIONS_OVERRIDE:-${DEFAULT_TOP_FRACTIONS[*]}}"

DEFAULT_LAYERS=("2")
IFS=' ' read -r -a LAYERS <<< "${LAYERS_OVERRIDE:-${DEFAULT_LAYERS[*]}}"

METHOD="${METHOD:-leastsquares}"
DEFAULT_MODELS=("GCN_MLP" "SAGE_MLP" "GIN_MLP" "GAT_MLP")
# DEFAULT_MODELS=("GIN_MLP")
IFS=' ' read -r -a MODELS <<< "${MODELS_OVERRIDE:-${DEFAULT_MODELS[*]}}"
DEFAULT_SEEDS=("0" "10" "42")
# DEFAULT_SEEDS=("42")
IFS=' ' read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-${DEFAULT_SEEDS[*]}}"
NUM_TARGETS="${NUM_TARGETS:-1000}"
MAX_STEPS="${MAX_STEPS:-50}"
STRATEGY="${STRATEGY:-}"
# Least-squares strategies: confidence | sensitivity_mean | sensitivity_wtd_mean | sens_pr | sens_pr_graphaware | sens_divrank | sens_divrank_graphaware | sens_subspace_divrank | sens_subspace_retention_divrank | sens_subspace_divrank_graphaware | sens_subspace_retention_divrank_graphaware
LS_STRATEGY="${LS_STRATEGY:-sens_divrank}"

# Dataset → sensitive feature mapping for logging purposes
declare -A SENSITIVE_FEATURE_MAP=(
    ["bail"]="WHITE"
    ["income"]="fnlwgt"
    # ["credit"]="Age"
    ["pokec"]="AGE"
    ["yelp"]="feature_5"
    ["tfinance"]="feature_8"

)

# Optional fixed sensitive values (comma-separated) to surface in logs
declare -A FIXED_VALUE_MAP=(
    ["bail"]="0,1"
    ["pokec"]="15,25,35,45,55,65"
    ["yelp"]="0.39845687,0.99998516"
)

DATASETS=("income" "tfinance")
# DATASETS=("pokec" "bail" "yelp")


echo "========================================"
echo "Launching editing suite"
echo "Method: ${METHOD}"
echo "Models: ${MODELS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Lambdas: ${LAMBDAS[*]}"
echo "Top Fractions: ${TOP_FRACTIONS[*]}"
echo "Layers: ${LAYERS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Targets per run: ${NUM_TARGETS}"
echo "Max steps: ${MAX_STEPS}"
echo "Output root: ${OUTPUT_ROOT}"
echo "========================================"

for dataset in "${DATASETS[@]}"; do
    sensitive_feature="${SENSITIVE_FEATURE_MAP[$dataset]}"
    fixed_values="${FIXED_VALUE_MAP[$dataset]:-none}"
    echo ""
    echo "======== Dataset: ${dataset} | Sensitive: ${sensitive_feature} (values: ${fixed_values}) ========"

    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for lambda_reg in "${LAMBDAS[@]}"; do
                for top_fraction in "${TOP_FRACTIONS[@]}"; do
                    for num_layers in "${LAYERS[@]}"; do
                        output_dir="${OUTPUT_ROOT}/${METHOD}/${dataset}/${model}"

                        echo "----------------------------------------"
                        echo "Model: ${model} | Seed: ${seed} | Lambda: ${lambda_reg} | TopFrac: ${top_fraction} | Layers: ${num_layers}"
                        echo "Output: ${output_dir}"
                        echo "----------------------------------------"

                        # Export variables for run_edit.sh to pick up
                        export LAMBDA_REG="${lambda_reg}"
                        export TOP_FRACTION="${top_fraction}"
                        export NUM_LAYERS="${num_layers}"

                        bash "${RUN_EDIT_SH}" \
                            "${METHOD}" \
                            "${dataset}" \
                            "${model}" \
                            "${seed}" \
                            "${NUM_TARGETS}" \
                            "${MAX_STEPS}" \
                            "${STRATEGY}" \
                            "${LS_STRATEGY}" \
                            "${DATASET_DIR}" \
                            "${PRETRAIN_DIR}" \
                            "${output_dir}" \
                            --debug
                    done
                done
            done
        done
    done
done