#!/usr/bin/env bash
set -e
TOTAL_RUNS=7
CURRENT_RUN=0

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: LAMBDA_REG=0.1 TOP_FRACTION=0.25 NUM_LAYERS=2 PR_ALPHA=0.85 RANK_MIX_TAU=0.75 GAMMA_RETAIN=1.0 bash ./run_edit.sh leastsquares yelp Polynormer 42 1000 50 '' sens_subspace_retention_divrank_graphaware /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/leastsquares/yelp/Polynormer"
echo "========================================"
LAMBDA_REG=0.1 TOP_FRACTION=0.25 NUM_LAYERS=2 PR_ALPHA=0.85 RANK_MIX_TAU=0.75 GAMMA_RETAIN=1.0 bash ./run_edit.sh leastsquares yelp Polynormer 42 1000 50 '' sens_subspace_retention_divrank_graphaware /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/leastsquares/yelp/Polynormer

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: LAMBDA_REG=0.1 TOP_FRACTION=0.25 NUM_LAYERS=2 PR_ALPHA=0.85 RANK_MIX_TAU=0.75 GAMMA_RETAIN=1.0 bash ./run_edit.sh leastsquares tfinance Polynormer 0 1000 50 '' sens_subspace_retention_divrank_graphaware /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/leastsquares/tfinance/Polynormer"
echo "========================================"
LAMBDA_REG=0.1 TOP_FRACTION=0.25 NUM_LAYERS=2 PR_ALPHA=0.85 RANK_MIX_TAU=0.75 GAMMA_RETAIN=1.0 bash ./run_edit.sh leastsquares tfinance Polynormer 0 1000 50 '' sens_subspace_retention_divrank_graphaware /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/leastsquares/tfinance/Polynormer

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: LAMBDA_REG=0.1 TOP_FRACTION=0.25 NUM_LAYERS=2 PR_ALPHA=0.85 RANK_MIX_TAU=0.75 GAMMA_RETAIN=1.0 bash ./run_edit.sh leastsquares tfinance Polynormer 10 1000 50 '' sens_subspace_retention_divrank_graphaware /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/leastsquares/tfinance/Polynormer"
echo "========================================"
LAMBDA_REG=0.1 TOP_FRACTION=0.25 NUM_LAYERS=2 PR_ALPHA=0.85 RANK_MIX_TAU=0.75 GAMMA_RETAIN=1.0 bash ./run_edit.sh leastsquares tfinance Polynormer 10 1000 50 '' sens_subspace_retention_divrank_graphaware /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/leastsquares/tfinance/Polynormer

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: LAMBDA_REG=0.1 TOP_FRACTION=0.25 NUM_LAYERS=2 PR_ALPHA=0.85 RANK_MIX_TAU=0.75 GAMMA_RETAIN=1.0 bash ./run_edit.sh leastsquares tfinance Polynormer 42 1000 50 '' sens_subspace_retention_divrank_graphaware /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/leastsquares/tfinance/Polynormer"
echo "========================================"
LAMBDA_REG=0.1 TOP_FRACTION=0.25 NUM_LAYERS=2 PR_ALPHA=0.85 RANK_MIX_TAU=0.75 GAMMA_RETAIN=1.0 bash ./run_edit.sh leastsquares tfinance Polynormer 42 1000 50 '' sens_subspace_retention_divrank_graphaware /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/leastsquares/tfinance/Polynormer

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: LAMBDA_REG=0.1 TOP_FRACTION=0.25 NUM_LAYERS=2 PR_ALPHA=0.85 RANK_MIX_TAU=0.75 GAMMA_RETAIN=1.0 bash ./run_edit.sh leastsquares twitch-views Polynormer 0 1000 50 '' sens_subspace_retention_divrank_graphaware /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/leastsquares/twitch-views/Polynormer"
echo "========================================"
LAMBDA_REG=0.1 TOP_FRACTION=0.25 NUM_LAYERS=2 PR_ALPHA=0.85 RANK_MIX_TAU=0.75 GAMMA_RETAIN=1.0 bash ./run_edit.sh leastsquares twitch-views Polynormer 0 1000 50 '' sens_subspace_retention_divrank_graphaware /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/leastsquares/twitch-views/Polynormer

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: LAMBDA_REG=0.1 TOP_FRACTION=0.25 NUM_LAYERS=2 PR_ALPHA=0.85 RANK_MIX_TAU=0.75 GAMMA_RETAIN=1.0 bash ./run_edit.sh leastsquares twitch-views Polynormer 10 1000 50 '' sens_subspace_retention_divrank_graphaware /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/leastsquares/twitch-views/Polynormer"
echo "========================================"
LAMBDA_REG=0.1 TOP_FRACTION=0.25 NUM_LAYERS=2 PR_ALPHA=0.85 RANK_MIX_TAU=0.75 GAMMA_RETAIN=1.0 bash ./run_edit.sh leastsquares twitch-views Polynormer 10 1000 50 '' sens_subspace_retention_divrank_graphaware /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/leastsquares/twitch-views/Polynormer

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: LAMBDA_REG=0.1 TOP_FRACTION=0.25 NUM_LAYERS=2 PR_ALPHA=0.85 RANK_MIX_TAU=0.75 GAMMA_RETAIN=1.0 bash ./run_edit.sh leastsquares twitch-views Polynormer 42 1000 50 '' sens_subspace_retention_divrank_graphaware /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/leastsquares/twitch-views/Polynormer"
echo "========================================"
LAMBDA_REG=0.1 TOP_FRACTION=0.25 NUM_LAYERS=2 PR_ALPHA=0.85 RANK_MIX_TAU=0.75 GAMMA_RETAIN=1.0 bash ./run_edit.sh leastsquares twitch-views Polynormer 42 1000 50 '' sens_subspace_retention_divrank_graphaware /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/leastsquares/twitch-views/Polynormer
