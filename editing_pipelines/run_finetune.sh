#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to this script
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../paths.sh
source "${REPO}/paths.sh"
RUN_EDIT_SH="${SCRIPT_DIR}/run_edit.sh"

METHOD="finetune"
# DEFAULT_MODELS=("GCN_MLP" "GAT_MLP" "SAGE_MLP" "GIN_MLP" "Polynormer_MLP")
# DEFAULT_MODELS=("GCN_MLP GIN_MLP Polynormer_MLP")
DEFAULT_MODELS=("GCN_MLP" "GIN_MLP")
IFS=' ' read -r -a MODELS <<< "${MODELS_OVERRIDE:-${DEFAULT_MODELS[*]}}"

# DEFAULT_SEEDS=("42" "0" "10")
DEFAULT_SEEDS=("0")
IFS=' ' read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-${DEFAULT_SEEDS[*]}}"

DEFAULT_TOP_FRACTIONS=("0.25")
IFS=' ' read -r -a TOP_FRACTIONS <<< "${TOP_FRACTIONS_OVERRIDE:-${DEFAULT_TOP_FRACTIONS[*]}}"

DEFAULT_FT_EPOCHS=("20")
IFS=' ' read -r -a FT_EPOCHS_LIST <<< "${FT_EPOCHS_OVERRIDE:-${DEFAULT_FT_EPOCHS[*]}}"

DEFAULT_FT_LRS=("0.01")
IFS=' ' read -r -a FT_LRS <<< "${FT_LR_OVERRIDE:-${DEFAULT_FT_LRS[*]}}"

# GNN depth: forwarded to run_edit.sh as --num-layers.
#   NUM_LAYERS=1 ./run_finetune.sh
#   LAYERS_OVERRIDE="1 2" ./run_finetune.sh
# NUM_LAYERS wins over LAYERS_OVERRIDE. Default 2 matches run_edit.py.
DEFAULT_LAYERS=("2 3 5")
if [[ -n "${NUM_LAYERS:-}" ]]; then
    IFS=' ' read -r -a LAYERS <<< "${NUM_LAYERS}"
else
    IFS=' ' read -r -a LAYERS <<< "${LAYERS_OVERRIDE:-${DEFAULT_LAYERS[*]}}"
fi
FINETUNE_EXPLICIT_LAYER_PATH=0
if [[ -n "${NUM_LAYERS:-}" || -n "${LAYERS_OVERRIDE:-}" ]]; then
    FINETUNE_EXPLICIT_LAYER_PATH=1
fi

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

# DATASETS=("pokec" "yelp" "bail")
DATASETS=("tfinance")
# DATASETS=("tfinance")
# DATASETS=("twitch-views" "artnet-views")
if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
    IFS=' ' read -r -a DATASETS <<< "${DATASETS_OVERRIDE}"
fi


echo "========================================"
echo "Launching Finetune suite"
echo "Method: ${METHOD}"
echo "Models: ${MODELS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Top Fractions: ${TOP_FRACTIONS[*]}"
echo "FT Epochs: ${FT_EPOCHS_LIST[*]}"
echo "FT LRs: ${FT_LRS[*]}"
echo "Layers (NUM_LAYERS): ${LAYERS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Targets per run: ${NUM_TARGETS}"
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
            for ft_epochs in "${FT_EPOCHS_LIST[@]}"; do
                for ft_lr in "${FT_LRS[@]}"; do
                    for top_fraction in "${TOP_FRACTIONS[@]}"; do
                        for num_layers in "${LAYERS[@]}"; do
                            if [[ "${FINETUNE_EXPLICIT_LAYER_PATH}" -eq 1 ]]; then
                                output_dir="${OUTPUT_ROOT}/${METHOD}/${dataset}/${model}/layers_${num_layers}"
                            else
                                output_dir="${OUTPUT_ROOT}/${METHOD}/${dataset}/${model}"
                            fi

                            echo "----------------------------------------"
                            echo "Model: ${model} | Seed: ${seed} | FTEpochs: ${ft_epochs} | FTLR: ${ft_lr} | TopFrac: ${top_fraction} | Layers: ${num_layers}"
                            echo "Output: ${output_dir}"
                            echo "----------------------------------------"

                            export TOP_FRACTION="${top_fraction}"
                            export NUM_LAYERS="${num_layers}"
                            export FT_EPOCHS="${ft_epochs}"
                            export FT_LR="${ft_lr}"
                            unset LAMBDA_REG || true

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
    done
done
