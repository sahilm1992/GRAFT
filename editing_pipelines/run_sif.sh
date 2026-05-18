#!/bin/bash

# Resolve project paths relative to this script so it works from anywhere
REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../paths.sh
source "${REPO}/paths.sh"

# Configuration
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
