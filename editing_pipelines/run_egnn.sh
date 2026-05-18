#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to this script
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../paths.sh
source "${REPO}/paths.sh"
RUN_EDIT_SH="${SCRIPT_DIR}/run_edit.sh"

# GNN depth for checkpoint selection (--num-layers → run_edit.py architecture.num_layers).
# - Set NUM_LAYERS before invoking this script (e.g. NUM_LAYERS=1 ./run_egnn.sh) for a single value.
# - Or set LAYERS_OVERRIDE to one or more space-separated integers (e.g. LAYERS_OVERRIDE="1 2").
# If NUM_LAYERS is set, it wins over LAYERS_OVERRIDE. Default matches run_edit.py (2).
DEFAULT_LAYERS=("2")
if [[ -n "${NUM_LAYERS:-}" ]]; then
    IFS=' ' read -r -a LAYERS <<< "${NUM_LAYERS}"
else
    IFS=' ' read -r -a LAYERS <<< "${LAYERS_OVERRIDE:-${DEFAULT_LAYERS[*]}}"
fi

# When no layer override is set, keep the historical output layout (.../model/top_<frac>).
# Any explicit NUM_LAYERS or LAYERS_OVERRIDE uses .../model/layers_<n>/top_<frac> so sweeps do not clash.
EGNN_EXPLICIT_LAYERS=0
if [[ -n "${NUM_LAYERS:-}" || -n "${LAYERS_OVERRIDE:-}" ]]; then
    EGNN_EXPLICIT_LAYERS=1
fi

METHOD="egnn"
# DEFAULT_MODELS=("GCN_MLP" "SAGE_MLP" "GIseeN_MLP" "Polynormer_MLP" "GAT_MLP" )
DEFAULT_MODELS=("Polynormer" "GCN_MLP" "GIN_MLP")
IFS=' ' read -r -a MODELS <<< "${MODELS_OVERRIDE:-${DEFAULT_MODELS[*]}}"

DEFAULT_SEEDS=("0" "10" "42")
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
# DATASETS=("twitch-views" "artnet-views")
DATASETS=("pokec" "bail" "yelp")
if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
    IFS=' ' read -r -a DATASETS <<< "${DATASETS_OVERRIDE}"
fi

echo "========================================"
echo "Launching EGNN suite"
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
    for seed in "${SEEDS[@]}"; do
        for model in "${MODELS[@]}"; do
            for top_fraction in "${TOP_FRACTIONS[@]}"; do
                for num_layers in "${LAYERS[@]}"; do
                    if [[ "${EGNN_EXPLICIT_LAYERS}" -eq 1 ]]; then
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
