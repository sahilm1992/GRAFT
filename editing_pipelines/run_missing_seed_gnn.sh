#!/usr/bin/env bash
set -e
TOTAL_RUNS=31
CURRENT_RUN=0

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn yelp Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/yelp/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn yelp Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/yelp/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn yelp Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/yelp/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn yelp Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/yelp/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn tfinance GCN_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/tfinance/GCN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn tfinance GCN_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/tfinance/GCN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn tfinance GIN_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/tfinance/GIN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn tfinance GIN_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/tfinance/GIN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn tfinance Polynormer_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/tfinance/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn tfinance Polynormer_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/tfinance/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn tfinance Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/tfinance/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn tfinance Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/tfinance/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn tfinance Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/tfinance/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn tfinance Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/tfinance/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn artnet-views GCN_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/artnet-views/GCN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn artnet-views GCN_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/artnet-views/GCN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn artnet-views GCN_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/artnet-views/GCN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn artnet-views GCN_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/artnet-views/GCN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn artnet-views GCN_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/artnet-views/GCN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn artnet-views GCN_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/artnet-views/GCN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn artnet-views Polynormer_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/artnet-views/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn artnet-views Polynormer_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/artnet-views/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn artnet-views Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/artnet-views/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn artnet-views Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/artnet-views/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn artnet-views Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/artnet-views/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn artnet-views Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/artnet-views/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views GCN_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/GCN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views GCN_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/GCN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views GCN_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/GCN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views GCN_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/GCN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views GCN_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/GCN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views GCN_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/GCN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views GIN_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/GIN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views GIN_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/GIN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views GIN_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/GIN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views GIN_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/GIN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views GIN_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/GIN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views GIN_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/GIN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views Polynormer_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views Polynormer_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/artnet-views/GAT_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/artnet-views/GAT_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/artnet-views/GAT_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/artnet-views/GAT_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/artnet-views/GAT_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/artnet-views/GAT_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/tfinance/GAT_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/tfinance/GAT_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/tfinance/GAT_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/tfinance/GAT_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/tfinance/GAT_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/tfinance/GAT_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/GAT_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/GAT_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/GAT_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/GAT_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/GAT_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh seed_gnn twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/seed_gnn/twitch-views/GAT_MLP/top_0.25
