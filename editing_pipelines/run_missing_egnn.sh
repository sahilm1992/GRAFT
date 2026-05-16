#!/usr/bin/env bash
set -e
TOTAL_RUNS=39
CURRENT_RUN=0

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn bail Polynormer_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/bail/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn bail Polynormer_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/bail/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn bail Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/bail/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn bail Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/bail/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn bail Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/bail/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn bail Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/bail/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn pokec Polynormer_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/pokec/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn pokec Polynormer_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/pokec/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn pokec Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/pokec/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn pokec Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/pokec/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn pokec Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/pokec/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn pokec Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/pokec/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn yelp Polynormer_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/yelp/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn yelp Polynormer_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/yelp/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn yelp Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/yelp/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn yelp Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/yelp/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn yelp Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/yelp/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn yelp Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/yelp/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn tfinance GIN_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/tfinance/GIN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn tfinance GIN_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/tfinance/GIN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn tfinance GIN_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/tfinance/GIN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn tfinance GIN_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/tfinance/GIN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn tfinance GIN_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/tfinance/GIN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn tfinance GIN_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/tfinance/GIN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn tfinance Polynormer_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/tfinance/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn tfinance Polynormer_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/tfinance/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn tfinance Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/tfinance/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn tfinance Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/tfinance/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn tfinance Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/tfinance/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn tfinance Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/tfinance/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn artnet-views GCN_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/artnet-views/GCN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn artnet-views GCN_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/artnet-views/GCN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn artnet-views GCN_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/artnet-views/GCN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn artnet-views GCN_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/artnet-views/GCN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn artnet-views GCN_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/artnet-views/GCN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn artnet-views GCN_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/artnet-views/GCN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn artnet-views Polynormer_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/artnet-views/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn artnet-views Polynormer_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/artnet-views/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn artnet-views Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/artnet-views/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn artnet-views Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/artnet-views/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn artnet-views Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/artnet-views/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn artnet-views Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/artnet-views/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views GCN_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/GCN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views GCN_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/GCN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views GCN_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/GCN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views GCN_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/GCN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views GCN_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/GCN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views GCN_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/GCN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views GIN_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/GIN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views GIN_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/GIN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views GIN_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/GIN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views GIN_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/GIN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views GIN_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/GIN_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views GIN_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/GIN_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views Polynormer_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views Polynormer_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views Polynormer_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/Polynormer_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views Polynormer_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/Polynormer_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/artnet-views/GAT_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/artnet-views/GAT_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/artnet-views/GAT_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/artnet-views/GAT_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/artnet-views/GAT_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/artnet-views/GAT_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/tfinance/GAT_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/tfinance/GAT_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/tfinance/GAT_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/tfinance/GAT_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/tfinance/GAT_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/tfinance/GAT_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/GAT_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/GAT_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/GAT_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/GAT_MLP/top_0.25

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/GAT_MLP/top_0.25"
echo "========================================"
TOP_FRACTION=0.25 bash ./run_edit.sh egnn twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/egnn/twitch-views/GAT_MLP/top_0.25
