#!/usr/bin/env bash
# Fix hyperparameters
export LAMBDAS_OVERRIDE="0.1"
export TOP_FRACTIONS_OVERRIDE="0.25"
export PR_ALPHAS_OVERRIDE="0.85"
export RANK_MIX_TAUS_OVERRIDE="0.75"

# Target dataset and specific strategies
export DATASETS_OVERRIDE="tfinance"
export LS_STRATEGIES_OVERRIDE="sens_subspace_divrank_graphaware sens_subspace_retention_divrank_graphaware"

# Run for all architectures and seeds
export MODELS_OVERRIDE="GCN_MLP GIN_MLP SAGE_MLP"
export SEEDS_OVERRIDE="0 10 42"

# Ensure we use the right script and pass through any GPU preference
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} bash run_editing_suite.sh
