#!/bin/bash

# Resolve project paths relative to this script so it works from anywhere
REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../paths.sh
source "${REPO}/paths.sh"

# Configuration

# List of datasets to run
datasets=("bail" "income" "pokec")

# Set to "true" if you want to run sensitivity analysis on the ABLATED model
# Set to "false" if you want to run sensitivity analysis on the FULL (vanilla) model
RUN_ABLATION="false"

for dataset in "${datasets[@]}"; do
    # Map dataset to its primary sensitive feature
    case $dataset in
        "income")
            sensitive_feat="race"
            ;;
        "bail")
            sensitive_feat="WHITE"
            ;;
        "pokec")
            sensitive_feat="AGE"
            ;;
        *)
            sensitive_feat="Age"
            ;;
    esac

    # Determine the ablation argument for the dataset script
    if [ "$RUN_ABLATION" = "true" ]; then
        ablation=$sensitive_feat
    else
        ablation="none"
    fi

    echo "========================================"
    echo "STARTING SENSITIVITY SUITE FOR: $dataset"
    echo "Target Feature: $sensitive_feat"
    echo "Ablation Mode: $ablation"
    echo "========================================"
    
    bash "${SCRIPT_DIR}/scripts/sensitivity/${dataset}.sh" \
        "${OUTPUT_DIR_ROOT}" \
        "${DATASET_DIR}" \
        "${ablation}"
done
