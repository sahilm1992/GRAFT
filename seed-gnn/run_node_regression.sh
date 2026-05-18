#!/usr/bin/env bash
set -euo pipefail

# Run seed-GNN pretraining for node-regression datasets.
# Mirrors run.sh style (same argument parsing and per-dataset pretrain script call).
#
# Defaults:
#   datasets: twitch-views, artnet-views
#   models:   gcn, sage, gin, gat
#   seeds:    42 0 10
#   layers:   2
#
# Output:
#   checkpoints: <output_root>/edit_ckpts/<dataset>/
#   results:     <output_root>/results/seed_gnn/<model>/<dataset>/
#
# Optional env overrides:
#   OUTPUT_DIR_ROOT
#   DATASET_DIR
#   DATASETS_OVERRIDE           (space-separated dataset names)
#   SEEDS_OVERRIDE              (space-separated seeds)
#   LAYERS_OVERRIDE             (space-separated num_layers)
#
# Usage examples:
#   ./run_node_regression.sh
#   ./run_node_regression.sh --architectures gcn,sage
#   ./run_node_regression.sh --architectures gat --gat_optimized --neighbor_batch_size 1024 --neighbor_num_neighbors 10

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../paths.sh
source "${REPO}/paths.sh"
cd "${SCRIPT_DIR}"

DATASET_DIR="${DATASET_DIR:-${GRAPHLAND_DATASET_DIR}}"

IFS=' ' read -r -a datasets <<< "${DATASETS_OVERRIDE:- twitch-views artnet-views}"
IFS=' ' read -r -a seeds <<< "${SEEDS_OVERRIDE:-42 0 10}"
IFS=' ' read -r -a num_layers <<< "${LAYERS_OVERRIDE:-2}"
models=("gcn" "sage" "gin" "gat")

ARCHITECTURE_ARGS=()
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --architectures|--models)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                if [[ "$1" == *","* ]]; then
                    IFS=',' read -r -a parsed_models <<< "$1"
                    for parsed_model in "${parsed_models[@]}"; do
                        if [[ -n "$parsed_model" ]]; then
                            ARCHITECTURE_ARGS+=("$parsed_model")
                        fi
                    done
                else
                    ARCHITECTURE_ARGS+=("$1")
                fi
                shift
            done
            ;;
        --gat_optimized)
            EXTRA_ARGS+=("--gat_optimized")
            shift
            ;;
        --neighbor_batch_size)
            if [[ -z "${2:-}" || "$2" == --* ]]; then
                echo "Missing value for --neighbor_batch_size" >&2
                exit 1
            fi
            EXTRA_ARGS+=("--neighbor_batch_size" "$2")
            shift 2
            ;;
        --neighbor_num_neighbors)
            if [[ -z "${2:-}" || "$2" == --* ]]; then
                echo "Missing value for --neighbor_num_neighbors" >&2
                exit 1
            fi
            EXTRA_ARGS+=("--neighbor_num_neighbors" "$2")
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--architectures <arch1[,arch2,...]> [arch3 ...]] [--gat_optimized] [--neighbor_batch_size N] [--neighbor_num_neighbors K]" >&2
            exit 1
            ;;
    esac
done

if [ ${#ARCHITECTURE_ARGS[@]} -gt 0 ]; then
    models=("${ARCHITECTURE_ARGS[@]}")
fi

echo "========================================"
echo "Node-regression pretraining suite"
echo "Datasets: ${datasets[*]}"
echo "Models: ${models[*]}"
echo "Seeds: ${seeds[*]}"
echo "Num layers: ${num_layers[*]}"
echo "Dataset dir: ${DATASET_DIR}"
echo "Output root: ${OUTPUT_DIR_ROOT}"
echo "========================================"

for dataset in "${datasets[@]}"; do
    pretrain_script="scripts/pretrain/seed_gnn/${dataset}.sh"
    if [[ ! -f "${pretrain_script}" ]]; then
        echo "Missing dataset pretrain script: ${pretrain_script}" >&2
        exit 1
    fi
    for model in "${models[@]}"; do
        pipeline_cfg="config/pipeline_config/seed_gnn/${model}/${dataset}.json"
        eval_cfg="config/eval_config/edit_gnn/${dataset}.json"
        if [[ ! -f "${pipeline_cfg}" ]]; then
            echo "Missing pipeline config: ${pipeline_cfg}" >&2
            exit 1
        fi
        if [[ ! -f "${eval_cfg}" ]]; then
            echo "Missing eval config: ${eval_cfg}" >&2
            exit 1
        fi
        for seed in "${seeds[@]}"; do
            for layer in "${num_layers[@]}"; do
                echo "Running dataset=${dataset} model=${model} seed=${seed} layers=${layer}"
                bash "${pretrain_script}" \
                    "${OUTPUT_DIR_ROOT}" \
                    "${DATASET_DIR}" \
                    --models "${model}" \
                    "${EXTRA_ARGS[@]}" \
                    --seed "${seed}" \
                    --num_layers "${layer}"
            done
        done
    done
done
