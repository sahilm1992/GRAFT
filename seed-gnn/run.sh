#!/bin/bash

# List of datasets
# datasets=("amazoncomputers" "amazonphoto" "arxiv" "coauthorcs" "cora" "products" "income" "credit" "bail")
# rm -rf ~/data/seed_gnn_data/edit_ckpts/credit/*
# rm -rf ~/data/seed_gnn_data/edit_ckpts/income/*
# rm -rf ~/data/seed_gnn_data/results/seed_gnn/*/credit
# rm -rf ~/data/seed_gnn_data/results/seed_gnn/*/income

# rm -rf ~/data/seed_gnn_data/edit_ckpts/pokec/*
# rm -rf ~/data/seed_gnn_data/edit_ckpts/bail/full_features/bail/*

# rm -rf ~/data/seed_gnn_data/edit_ckpts_feature_ablated/pokec/full_features/pokec/*
# rm -rf ~/data/seed_gnn_data/edit_ckpts/income/full_features/income/*
# rm -rf ~/data/seed_gnn_data/edit_ckpts/pokec/full_features/pokec/*
models=("polynormer")
# models=("gcn")
# models=("gcn")
# datasets=("pokec" "income" "bail")

# datasets=("twitch-views")
seeds=("42" "0" "10" )
# seeds=("2" "3")
# num_layers=("5")
num_layers=("1")

# --- FEATURE ABLATION CONFIGURATION ---
USE_FEATURE_ABLATION=false # Set to true to enable
FEATURE_VARIANT="full_features"
DROP_FEATURES="WHITE"

FEATURE_ABLATION_ARGS=""
if [ "$USE_FEATURE_ABLATION" = true ]; then
    FEATURE_ABLATION_ARGS="--use_feature_ablation --feature_variant ${FEATURE_VARIANT} --drop_features ${DROP_FEATURES}"
fi

# Optional CLI override:
#   ./run.sh --architectures gcn,sage
#   ./run.sh --architectures gcn sage gin
#   ./run.sh --architectures gat --gat_optimized
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
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "Missing value for --neighbor_batch_size"
                exit 1
            fi
            EXTRA_ARGS+=("--neighbor_batch_size" "$2")
            shift 2
            ;;
        --neighbor_num_neighbors)
            if [[ -z "$2" || "$2" == --* ]]; then
                echo "Missing value for --neighbor_num_neighbors"
                exit 1
            fi
            EXTRA_ARGS+=("--neighbor_num_neighbors" "$2")
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--architectures <arch1[,arch2,...]> [arch3 ...]] [--gat_optimized] [--neighbor_batch_size N] [--neighbor_num_neighbors K]"
            exit 1
            ;;
    esac
done

if [ ${#ARCHITECTURE_ARGS[@]} -gt 0 ]; then
    models=("${ARCHITECTURE_ARGS[@]}")
fi

# --------------------------------------

# vanilla (no editing)

for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
        for layer in "${num_layers[@]}"; do
            echo "Running for dataset: $dataset | Seed: $seed | Num layers: $layer"
            bash scripts/pretrain/seed_gnn/${dataset}.sh \
                ~/data/seed_gnn_data \
                ~/data/seed_gnn_data/dataset \
                --models "${models[@]}" \
                ${FEATURE_ABLATION_ARGS} \
                "${EXTRA_ARGS[@]}" \
                --seed ${seed} \
                --num_layers ${layer}
        done
    done
done

# seed-gnn

# for dataset in "${datasets[@]}"; do
#     for model in "${models[@]}"; do
#         echo "Running for dataset: $dataset"
#         bash scripts/edit/seed_gnn/${model}/${dataset}.sh \
#             ~/data/seed_gnn_data \
#             ~/data/seed_gnn_data/dataset \
#             ${FEATURE_ABLATION_ARGS} \
#             ${SEED_ARGS}
#     done
# done


# modcirc

# for dataset in "${datasets[@]}"; do
#     for model in "${models[@]}"; do
#         echo "Running for dataset: $dataset"
#         bash scripts/modcirc/seed_gnn/${dataset}.sh \
#             ~/data/seed_gnn_data \
#             ~/data/seed_gnn_data/dataset \
#             $model \
#             ${SEED_ARGS}
#     done
# done

# seed-gnn

