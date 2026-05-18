#!/usr/bin/env bash
# Pretrain seed-GNN models with one attribute removed per regression dataset, matching the
# sensitive-feature column used in editing_pipelines/run_editing_suite.sh.
#
# For each (dataset, architecture, seed, layer count) this calls
# scripts/pretrain/seed_gnn/<dataset>.sh with --use_feature_ablation and
# --drop_features set to that dataset's feature name.
#
# Output locations (see paths.sh → $OUTPUT_DIR_ROOT)
#
#   Checkpoints:
#     ${OUTPUT_DIR_ROOT}/edit_ckpts_feature_ablated/<dataset>/no_<feature>/
#
#   Pretrain metrics / logs:
#     ${OUTPUT_DIR_ROOT}/results/seed_gnn/<arch>/<dataset>/no_<feature>/
#
# Usage:
#   ./run_feature_drop_pretrain_regression.sh
#   DATASETS_OVERRIDE="twitch-views" ./run_feature_drop_pretrain_regression.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../paths.sh
source "${REPO}/paths.sh"
PRETRAIN_SCRIPT_DIR="${SCRIPT_DIR}/scripts/pretrain/seed_gnn"


IFS=' ' read -r -a DATASETS <<< "${DATASETS_OVERRIDE:-twitch-views}"

DEFAULT_MODELS=(Polynormer_MLP)
if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
  IFS=' ' read -r -a EDIT_MODELS <<< "${MODELS_OVERRIDE}"
else
  EDIT_MODELS=("${DEFAULT_MODELS[@]}")
fi

IFS=' ' read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-0 10 42}"
IFS=' ' read -r -a NUM_LAYERS_LIST <<< "${LAYERS_OVERRIDE:-2}"

# Dataset -> feature to drop (must match editing_pipelines/run_editing_suite.sh SENSITIVE_FEATURE_MAP)
declare -A DROP_FEATURE_FOR=(
  ["artnet-views"]="feature_20_fraction"
  ["twitch-views"]="affiliate_status_1.0"
)

declare -A MLP_TO_ARCH=(
  ["GCN_MLP"]="gcn"
  ["GIN_MLP"]="gin"
  ["SAGE_MLP"]="sage"
  ["GAT_MLP"]="gat"
  ["Polynormer_MLP"]="polynormer"
)

SEED_ARCHS=()
for m in "${EDIT_MODELS[@]}"; do
  arch="${MLP_TO_ARCH[$m]:-}"
  if [[ -z "$arch" ]]; then
    echo "Unknown model name '${m}' (expected one of: ${!MLP_TO_ARCH[*]})" >&2
    exit 1
  fi
  SEED_ARCHS+=("$arch")
done

EXTRA_FORWARD=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --)
      shift
      EXTRA_FORWARD+=("$@")
      break
      ;;
    *)
      echo "Unexpected argument: $1 (use -- before extra main.py flags)" >&2
      exit 1
      ;;
  esac
done

echo "========================================"
echo "Feature-drop pretrain suite (REGRESSION)"
echo "OUTPUT_DIR_ROOT: ${OUTPUT_DIR_ROOT}"
echo "DATASET_DIR:     ${DATASET_DIR}"
echo "Datasets:        ${DATASETS[*]}"
echo "Architectures:   ${SEED_ARCHS[*]} (from ${EDIT_MODELS[*]})"
echo "Seeds:           ${SEEDS[*]}"
echo "Num layers:      ${NUM_LAYERS_LIST[*]}"
echo "========================================"

for dataset in "${DATASETS[@]}"; do
  drop_feat="${DROP_FEATURE_FOR[$dataset]:-}"
  if [[ -z "$drop_feat" ]]; then
    echo "ERROR: No DROP_FEATURE_FOR entry for dataset '${dataset}'." >&2
    exit 1
  fi

  pretrain_sh="${PRETRAIN_SCRIPT_DIR}/${dataset}.sh"
  if [[ ! -f "$pretrain_sh" ]]; then
    echo "ERROR: Missing pretrain script ${pretrain_sh}" >&2
    exit 1
  fi

  feature_variant="no_${drop_feat}"

  echo ""
  echo "======== Dataset: ${dataset} | drop feature: ${drop_feat} | variant: ${feature_variant} ========"

  for seed in "${SEEDS[@]}"; do
    for layer in "${NUM_LAYERS_LIST[@]}"; do
      echo "Running pretrain | seed=${seed} | num_layers=${layer}"
      bash "${pretrain_sh}" \
        "${OUTPUT_DIR_ROOT}" \
        "${DATASET_DIR}" \
        --models "${SEED_ARCHS[@]}" \
        --use_feature_ablation \
        --feature_variant "${feature_variant}" \
        --drop_features "${drop_feat}" \
        --seed "${seed}" \
        --num_layers "${layer}" \
        "${EXTRA_FORWARD[@]}"
    done
  done
done

echo ""
echo "Done."
