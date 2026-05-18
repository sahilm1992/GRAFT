#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to this script
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../paths.sh
source "${REPO}/paths.sh"
RUN_EDIT_SH="${SCRIPT_DIR}/run_edit.sh"
# GNN depth: forwarded to run_edit.sh as --num-layers (run_edit.py architecture.num_layers).
#   NUM_LAYERS=1 ./run_seed_gnn.sh
#   LAYERS_OVERRIDE="1 2" ./run_seed_gnn.sh
# NUM_LAYERS wins over LAYERS_OVERRIDE. Default 2 matches run_edit.py.
DEFAULT_LAYERS=("2")
if [[ -n "${NUM_LAYERS:-}" ]]; then
    IFS=' ' read -r -a LAYERS <<< "${NUM_LAYERS}"
else
    IFS=' ' read -r -a LAYERS <<< "${LAYERS_OVERRIDE:-${DEFAULT_LAYERS[*]}}"
fi
SEED_GNN_EXPLICIT_LAYER_PATH=0
if [[ -n "${NUM_LAYERS:-}" || -n "${LAYERS_OVERRIDE:-}" ]]; then
    SEED_GNN_EXPLICIT_LAYER_PATH=1
fi

METHOD="seed_gnn"
DEFAULT_MODELS=( "GCN_MLP" "GIN_MLP" "SAGE_MLP" "Polynormer_MLP" "GAT_MLP")
IFS=' ' read -r -a MODELS <<< "${MODELS_OVERRIDE:-${DEFAULT_MODELS[*]}}"

DEFAULT_SEEDS=("42" "0" "10")
IFS=' ' read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-${DEFAULT_SEEDS[*]}}"
DEFAULT_TOP_FRACTIONS=("0.25")
IFS=' ' read -r -a TOP_FRACTIONS <<< "${TOP_FRACTIONS_OVERRIDE:-${DEFAULT_TOP_FRACTIONS[*]}}"

NUM_TARGETS="${NUM_TARGETS:-1000}"
MAX_STEPS="${MAX_STEPS:-50}"
STRATEGY="${STRATEGY:-}"
LS_STRATEGY="${LS_STRATEGY:-confidence}"

declare -A SENSITIVE_FEATURE_MAP=(
    ["bail"]="WHITE"
    ["income"]="fnlwgt"
    ["pokec"]="AGE"
    ["yelp"]="feature_5"
    ["tfinance"]="feature_8"
    ["artnet-views"]="feature_20_fraction"
    ["twitch-views"]="affiliate_status_1.0"
)

declare -A FIXED_VALUE_MAP=(
    ["bail"]="0,1"
    ["pokec"]="15,25,35,45,55,65"
    ["yelp"]="0.39845687,0.99998516"
)

# DATASETS=("tfinance")
DATASETS=("twitch-views" "artnet-views")
if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
    IFS=' ' read -r -a DATASETS <<< "${DATASETS_OVERRIDE}"
fi

echo "========================================"
echo "Launching SEED-GNN suite"
echo "Method: ${METHOD}"
echo "Models: ${MODELS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Targets per run: ${NUM_TARGETS}"
echo "Top fractions: ${TOP_FRACTIONS[*]}"
echo "Layers (NUM_LAYERS): ${LAYERS[*]}"
echo "Max steps: ${MAX_STEPS}"
echo "Output root: ${OUTPUT_ROOT}"
echo "========================================"

for dataset in "${DATASETS[@]}"; do
    sensitive_feature="${SENSITIVE_FEATURE_MAP[$dataset]:-unknown}"
    fixed_values="${FIXED_VALUE_MAP[$dataset]:-none}"
    echo ""
    echo "======== Dataset: ${dataset} | Sensitive: ${sensitive_feature} (values: ${fixed_values}) ========"

    for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            for top_fraction in "${TOP_FRACTIONS[@]}"; do
                for num_layers in "${LAYERS[@]}"; do
                    if [[ "${SEED_GNN_EXPLICIT_LAYER_PATH}" -eq 1 ]]; then
                        output_dir="${OUTPUT_ROOT}/${METHOD}/${dataset}/${model}/layers_${num_layers}/top_${top_fraction}"
                    else
                        output_dir="${OUTPUT_ROOT}/${METHOD}/${dataset}/${model}/top_${top_fraction}"
                    fi

                    echo "----------------------------------------"
                    echo "Model: ${model} | Seed: ${seed} | Top fraction: ${top_fraction} | Layers: ${num_layers}"
                    echo "Output: ${output_dir}"
                    echo "----------------------------------------"

                    unset LAMBDA_REG || true
                    export TOP_FRACTION="${top_fraction}"
                    export NUM_LAYERS="${num_layers}"
                    unset FT_EPOCHS || true
                    unset FT_LR || true

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
                        "${output_dir}"
                done
            done
        done
    done
done
