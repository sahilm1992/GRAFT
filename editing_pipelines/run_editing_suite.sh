#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to this script
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../paths.sh
source "${REPO}/paths.sh"
PROJECT_ROOT="${REPO}"

RUN_EDIT_SH="${SCRIPT_DIR}/run_edit.sh"

# DEFAULT_LAMBDAS=("100 10 1 0.1 0.01 0.001 ")
DEFAULT_LAMBDAS=("0.1")
IFS=' ' read -r -a LAMBDAS <<< "${LAMBDAS_OVERRIDE:-${DEFAULT_LAMBDAS[*]}}"

DEFAULT_TOP_FRACTIONS=("0.25")
# DEFAULT_TOP_FRACTIONS=("0.1" "0.25" "0.5")
IFS=' ' read -r -a TOP_FRACTIONS <<< "${TOP_FRACTIONS_OVERRIDE:-${DEFAULT_TOP_FRACTIONS[*]}}"

DEFAULT_LAYERS=("2 3 5")
IFS=' ' read -r -a LAYERS <<< "${LAYERS_OVERRIDE:-${DEFAULT_LAYERS[*]}}"

METHOD="${METHOD:-leastsquares}"
DEFAULT_MODELS=("GIN_MLP" "GCN_MLP")
# DEFAULT_MODELS=("GAT_MLP")
# DEFAULT_MODELS=("Polynormer" "GCN_MLP" "GAT_MLP" "GIN_MLP")
IFS=' ' read -r -a MODELS <<< "${MODELS_OVERRIDE:-${DEFAULT_MODELS[*]}}"
# DEFAULT_SEEDS=( "42" "0" "10")
DEFAULT_SEEDS=( "0")
# DEFAULT_SEEDS=("42")
IFS=' ' read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-${DEFAULT_SEEDS[*]}}"
NUM_TARGETS="${NUM_TARGETS:-1000}"
MAX_STEPS="${MAX_STEPS:-50}"
STRATEGY="${STRATEGY:-}"

# PageRank/DivRank hyperparameters (only used by *_pr* / *_divrank* strategies)
# DEFAULT_PR_ALPHAS=("0.5 0.85 0.95")
DEFAULT_PR_ALPHAS=("0.85")
IFS=' ' read -r -a PR_ALPHAS <<< "${PR_ALPHAS_OVERRIDE:-${DEFAULT_PR_ALPHAS[*]}}"
# DEFAULT_RANK_MIX_TAUS=("0.25 0.5 0.75")
DEFAULT_RANK_MIX_TAUS=("0.75")
IFS=' ' read -r -a RANK_MIX_TAUS <<< "${RANK_MIX_TAUS_OVERRIDE:-${DEFAULT_RANK_MIX_TAUS[*]}}"
# Least-squares strategies: confidence | sensitivity_mean | sensitivity_wtd_mean | sens_pr | sens_pr_graphaware | sens_divrank | sens_divrank_graphaware
#   Subspace (no retention): sens_subspace | sens_subspace_pr | sens_subspace_pr_graphaware | sens_subspace_divrank | sens_subspace_divrank_graphaware
#   Subspace + retention:    sens_subspace_retention | sens_subspace_retention_pr | sens_subspace_retention_pr_graphaware | sens_subspace_retention_divrank | sens_subspace_retention_divrank_graphaware
# Strategy groups (subspace families)
#   - graph_agnostic / graph-agnostic
#   - pr_based / pr-based
#   - divrank_based / divrank-based
#   - retention_only / retention-only
#   - non_subspace / non-subspace (graph-agnostic + PR + DivRank, excluding subspace methods)
# You can combine groups via LS_STRATEGY_GROUPS, e.g.
#   LS_STRATEGY_GROUPS="graph_agnostic pr_based"
#   LS_STRATEGY_GROUPS="divrank-based"
SUBSPACE_GRAPH_AGNOSTIC=(
    "sens_subspace_retention"
    "sens_subspace"
)
SUBSPACE_PR_BASED=(
    "sens_subspace_pr"
    "sens_subspace_retention_pr"
    "sens_subspace_pr_graphaware"
    "sens_subspace_retention_pr_graphaware"
)
SUBSPACE_DIVRANK_BASED=(
    "sens_subspace_divrank"
    "sens_subspace_retention_divrank"
    "sens_subspace_divrank_graphaware"
    "sens_subspace_retention_divrank_graphaware"
)
SUBSPACE_RETENTION_ONLY=(
    "sens_subspace_retention"
    "sens_subspace_retention_pr"
    "sens_subspace_retention_pr_graphaware"
    "sens_subspace_retention_divrank"
    "sens_subspace_retention_divrank_graphaware"
)
NONSUBSPACE_GRAPH_AGNOSTIC=(
    "confidence"
    "sensitivity_mean"
    "sensitivity_wtd_mean"
)
NONSUBSPACE_PR_BASED=(
    "sens_pr"
    "sens_pr_graphaware"
)
NONSUBSPACE_DIVRANK_BASED=(
    "sens_divrank"
    "sens_divrank_graphaware"
)
NONSUBSPACE_ALL=(
    "${NONSUBSPACE_GRAPH_AGNOSTIC[@]}"
    "${NONSUBSPACE_PR_BASED[@]}"
    "${NONSUBSPACE_DIVRANK_BASED[@]}"
)
FINAL_STRATEGIES=(
    "sens_subspace_retention_pr_graphaware"
)

# Keep previous default behavior (divrank-oriented subspace strategies).
DEFAULT_LS_STRATEGIES=("${FINAL_STRATEGIES[@]}")
LS_STRATEGY_GROUPS="${LS_STRATEGY_GROUPS:-}"

append_unique_strategy() {
    local candidate="$1"
    local existing
    for existing in "${LS_STRATEGIES[@]:-}"; do
        if [[ "${existing}" == "${candidate}" ]]; then
            return
        fi
    done
    LS_STRATEGIES+=("${candidate}")
}

add_group_strategies() {
    local group="$1"
    local s
    case "${group}" in
        graph_agnostic|graph-agnostic)
            for s in "${SUBSPACE_GRAPH_AGNOSTIC[@]}"; do append_unique_strategy "${s}"; done
            ;;
        pr_based|pr-based)
            for s in "${SUBSPACE_PR_BASED[@]}"; do append_unique_strategy "${s}"; done
            ;;
        divrank_based|divrank-based)
            for s in "${SUBSPACE_DIVRANK_BASED[@]}"; do append_unique_strategy "${s}"; done
            ;;
        retention_only|retention-only|subspace_retention|subspace-retention)
            for s in "${SUBSPACE_RETENTION_ONLY[@]}"; do append_unique_strategy "${s}"; done
            ;;
        non_subspace|non-subspace|nonsubspace|classic)
            for s in "${NONSUBSPACE_ALL[@]}"; do append_unique_strategy "${s}"; done
            ;;
        all|all_subspace|all-subspace)
            for s in "${SUBSPACE_GRAPH_AGNOSTIC[@]}"; do append_unique_strategy "${s}"; done
            for s in "${SUBSPACE_PR_BASED[@]}"; do append_unique_strategy "${s}"; done
            for s in "${SUBSPACE_DIVRANK_BASED[@]}"; do append_unique_strategy "${s}"; done
            ;;
        *)
            echo "Unknown LS strategy group: '${group}'"
            echo "Valid values: graph_agnostic, pr_based, divrank_based, retention_only, non_subspace, all_subspace"
            exit 1
            ;;
    esac
}

if [[ -n "${LS_STRATEGIES_OVERRIDE:-}" ]]; then
    IFS=' ' read -r -a LS_STRATEGIES <<< "${LS_STRATEGIES_OVERRIDE}"
elif [[ -n "${LS_STRATEGY:-}" ]]; then
    LS_STRATEGIES=("${LS_STRATEGY}")
elif [[ -n "${LS_STRATEGY_GROUPS}" ]]; then
    LS_STRATEGIES=()
    # Support both comma-separated and whitespace-separated groups.
    normalized_groups="${LS_STRATEGY_GROUPS//,/ }"
    IFS=' ' read -r -a REQUESTED_GROUPS <<< "${normalized_groups}"
    for group in "${REQUESTED_GROUPS[@]}"; do
        [[ -z "${group}" ]] && continue
        add_group_strategies "${group}"
    done
    if [[ ${#LS_STRATEGIES[@]} -eq 0 ]]; then
        echo "LS_STRATEGY_GROUPS was set but no strategies were selected."
        exit 1
    fi
else
    LS_STRATEGIES=("${DEFAULT_LS_STRATEGIES[@]}")
fi

# Dataset → sensitive feature mapping for logging purposes
# Python twin (keep keys/values aligned): editing_pipelines/suite_feature_maps.py
declare -A SENSITIVE_FEATURE_MAP=(
    ["bail"]="WHITE"
    ["income"]="fnlwgt"
    # ["credit"]="Age"
    ["pokec"]="AGE"
    ["yelp"]="feature_5"
    ["tfinance"]="feature_8"
    ["artnet-views"]="feature_20_fraction"
    ["twitch-views"]="affiliate_status_1.0"

)

# Optional fixed sensitive values (comma-separated) to surface in logs
# Python twin (keep keys aligned): editing_pipelines/suite_feature_maps.FIXED_SENSITIVE_VALUES_BY_DATASET
declare -A FIXED_VALUE_MAP=(
    ["bail"]="0,1"
    ["pokec"]="15,25,35,45,55,65"
    ["yelp"]="0.39845687,0.99998516"
)

# DATASETS=("yelp")
# DATASETS=("pokec" "bail")
# DATASETS=("pokec" "bail" "yelp" "tfinance")
DATASETS=( "tfinance")
# DATASETS=( "twitch-views")
if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
    IFS=' ' read -r -a DATASETS <<< "${DATASETS_OVERRIDE}"
fi


echo "========================================"
echo "Launching editing suite"
echo "Method: ${METHOD}"
echo "Models: ${MODELS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Lambdas: ${LAMBDAS[*]}"
echo "Top Fractions: ${TOP_FRACTIONS[*]}"
echo "Layers: ${LAYERS[*]}"
echo "LS Strategies: ${LS_STRATEGIES[*]}"
echo "Gamma retain: ${GAMMA_RETAIN:-<strategy-default>}"
echo "PR Alphas: ${PR_ALPHAS[*]}"
echo "Rank Mix Alphas: ${RANK_MIX_TAUS[*]}"
echo "Datasets: ${DATASETS[*]}"
echo "Targets per run: ${NUM_TARGETS}"
echo "Max steps: ${MAX_STEPS}"
echo "Output root: ${OUTPUT_ROOT}"
echo "========================================"

# Helper: does this strategy use PR/DivRank scoring?
uses_pr_or_divrank() {
    [[ "$1" == *_pr* ]] || [[ "$1" == *_divrank* ]]
}

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
                        for ls_strategy in "${LS_STRATEGIES[@]}"; do
                            output_dir="${OUTPUT_ROOT}/${METHOD}/${dataset}/${model}"

                            # Determine PR/DivRank sweep values for this strategy
                            if uses_pr_or_divrank "${ls_strategy}"; then
                                local_pr_alphas=("${PR_ALPHAS[@]}")
                                local_rmx_alphas=("${RANK_MIX_TAUS[@]}")
                            else
                                local_pr_alphas=("")
                                local_rmx_alphas=("")
                            fi

                            for pr_alpha in "${local_pr_alphas[@]}"; do
                                for rmx_alpha in "${local_rmx_alphas[@]}"; do

                                    echo "----------------------------------------"
                                    echo "Model: ${model} | Seed: ${seed} | Lambda: ${lambda_reg} | TopFrac: ${top_fraction} | Layers: ${num_layers} | LS Strategy: ${ls_strategy} | PR_α: ${pr_alpha:-default} | RankMix_α: ${rmx_alpha:-default}"
                                    echo "Output: ${output_dir}"
                                    echo "----------------------------------------"

                                    export LAMBDA_REG="${lambda_reg}"
                                    export TOP_FRACTION="${top_fraction}"
                                    export NUM_LAYERS="${num_layers}"
                                    export PR_ALPHA="${pr_alpha}"
                                    export RANK_MIX_TAU="${rmx_alpha}"
                                    export GAMMA_RETAIN="${GAMMA_RETAIN:-}"

                                    bash "${RUN_EDIT_SH}" \
                                        "${METHOD}" \
                                        "${dataset}" \
                                        "${model}" \
                                        "${seed}" \
                                        "${NUM_TARGETS}" \
                                        "${MAX_STEPS}" \
                                        "${STRATEGY}" \
                                        "${ls_strategy}" \
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
        done
    done
done