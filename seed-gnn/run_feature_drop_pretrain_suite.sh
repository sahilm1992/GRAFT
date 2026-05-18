#!/usr/bin/env bash
# Pretrain seed-GNN models with one attribute removed per dataset, matching the
# sensitive-feature column used in editing_pipelines/run_editing_suite.sh.
#
# For each (dataset, architecture, seed, layer count) this calls
# scripts/pretrain/seed_gnn/<dataset>.sh with --use_feature_ablation and
# --drop_features set to that dataset's feature name. The feature_variant folder
# is no_<feature> (same convention as editing_pipelines/py_scripts/collect_all_metrics.py).
#
# Output locations (see paths.sh → $OUTPUT_DIR_ROOT)
#
#   Checkpoints:
#     ${OUTPUT_DIR_ROOT}/edit_ckpts_feature_ablated/<dataset>/no_<feature>/
#
#   Pretrain metrics / logs:
#     ${OUTPUT_DIR_ROOT}/results/seed_gnn/<arch>/<dataset>/no_<feature>/
#     e.g. metrics_pretrain.json under that directory.
#
# Environment overrides (same style as run_editing_suite.sh):
#   OUTPUT_DIR_ROOT   - root passed as $1 to per-dataset pretrain scripts (default below)
#   DATASET_DIR       - graph data root (default below)
#   DATASETS_OVERRIDE - space-separated dataset list
#   MODELS_OVERRIDE   - space-separated GCN_MLP-style names (default: all four)
#   SEEDS_OVERRIDE    - space-separated seeds (default: 0 10 42)
#   LAYERS_OVERRIDE   - space-separated layer counts (default: 2)
#
# Extra arguments after -- are forwarded to python main.py (e.g. --gat_optimized).
#
# Usage:
#   ./run_feature_drop_pretrain_suite.sh
#   OUTPUT_DIR_ROOT=~/data/seed_gnn_data ./run_feature_drop_pretrain_suite.sh
#   ./run_feature_drop_pretrain_suite.sh -- --gat_optimized

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../paths.sh
source "${REPO}/paths.sh"
PRETRAIN_SCRIPT_DIR="${SCRIPT_DIR}/scripts/pretrain/seed_gnn"


IFS=' ' read -r -a DATASETS <<< "${DATASETS_OVERRIDE:-pokec bail yelp tfinance}"

DEFAULT_MODELS=( GCN_MLP GIN_MLP SAGE_MLP GAT_MLP)
if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
  IFS=' ' read -r -a EDIT_MODELS <<< "${MODELS_OVERRIDE}"
else
  EDIT_MODELS=("${DEFAULT_MODELS[@]}")
fi

IFS=' ' read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-0 10 42}"
IFS=' ' read -r -a NUM_LAYERS_LIST <<< "${LAYERS_OVERRIDE:-2}"

# Dataset -> feature to drop (must match editing_pipelines/run_editing_suite.sh SENSITIVE_FEATURE_MAP)
declare -A DROP_FEATURE_FOR=(
  ["bail"]="WHITE"
  ["income"]="fnlwgt"
  ["pokec"]="AGE"
  ["yelp"]="feature_5"
  ["tfinance"]="feature_8"
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
    --gat_optimized)
      EXTRA_FORWARD+=("--gat_optimized")
      shift
      ;;
    --neighbor_batch_size)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Missing value for --neighbor_batch_size" >&2
        exit 1
      fi
      EXTRA_FORWARD+=("--neighbor_batch_size" "$2")
      shift 2
      ;;
    --neighbor_num_neighbors)
      if [[ -z "${2:-}" || "$2" == --* ]]; then
        echo "Missing value for --neighbor_num_neighbors" >&2
        exit 1
      fi
      EXTRA_FORWARD+=("--neighbor_num_neighbors" "$2")
      shift 2
      ;;
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
echo "Feature-drop pretrain suite (one attribute per dataset)"
echo "OUTPUT_DIR_ROOT: ${OUTPUT_DIR_ROOT}"
echo "DATASET_DIR:     ${DATASET_DIR}"
echo "Datasets:        ${DATASETS[*]}"
echo "Architectures:   ${SEED_ARCHS[*]} (from ${EDIT_MODELS[*]})"
echo "Seeds:           ${SEEDS[*]}"
echo "Num layers:      ${NUM_LAYERS_LIST[*]}"
echo "Checkpoints:     ${OUTPUT_DIR_ROOT}/edit_ckpts_feature_ablated/<dataset>/no_<feature>/"
echo "Results:         ${OUTPUT_DIR_ROOT}/results/seed_gnn/<arch>/<dataset>/no_<feature>/"
echo "========================================"

for dataset in "${DATASETS[@]}"; do
  drop_feat="${DROP_FEATURE_FOR[$dataset]:-}"
  if [[ -z "$drop_feat" ]]; then
    echo "ERROR: No DROP_FEATURE_FOR entry for dataset '${dataset}'. Add it next to run_editing_suite.sh's SENSITIVE_FEATURE_MAP." >&2
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
echo "Done. Outputs under ${OUTPUT_DIR_ROOT} as described in the header comment."
