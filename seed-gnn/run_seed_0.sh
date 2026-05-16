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
models=("gcn")
# models=("gcn")
# models=("gcn")
datasets=("pokec" "income" "bail")
# datasets=("bail")
# datasets=("web-fraud")

# --- FEATURE ABLATION CONFIGURATION ---
USE_FEATURE_ABLATION=false # Set to true to enable
FEATURE_VARIANT="full_features"
DROP_FEATURES="WHITE"
SEED=0 # Default seed

FEATURE_ABLATION_ARGS=""
if [ "$USE_FEATURE_ABLATION" = true ]; then
    FEATURE_ABLATION_ARGS="--use_feature_ablation --feature_variant ${FEATURE_VARIANT} --drop_features ${DROP_FEATURES}"
fi

SEED_ARGS="--seed ${SEED}"
# --------------------------------------

# vanilla (no editing)

for dataset in "${datasets[@]}"; do
    echo "Running for dataset: $dataset"
    bash scripts/pretrain/seed_gnn/${dataset}.sh \
        ~/data/seed_gnn_data \
        ~/data/seed_gnn_data/dataset \
        ${FEATURE_ABLATION_ARGS} \
        ${SEED_ARGS}
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

