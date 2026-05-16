#!/bin/bash

# Resolve project paths relative to this script so it works from anywhere
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
DATASET_DIR="${DATASET_DIR:-/home/model_editing/data/seed_gnn_data/dataset}"
OUTPUT_DIR_ROOT="${OUTPUT_DIR_ROOT:-/home/model_editing/data/seed_gnn_data}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"

# List of datasets to run
datasets=("bail" "income" "pokec")

for dataset in "${datasets[@]}"; do
    echo "========================================"
    echo "STARTING SIF SUITE FOR: $dataset"
    echo "Num samples: $NUM_SAMPLES"
    echo "========================================"

    bash "${SCRIPT_DIR}/shell_scripts/compute_sif.sh" \
        "${dataset}" \
        "${OUTPUT_DIR_ROOT}" \
        "${DATASET_DIR}" \
        "${NUM_SAMPLES}"
done
