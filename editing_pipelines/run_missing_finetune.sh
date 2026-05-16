#!/usr/bin/env bash
set -e
TOTAL_RUNS=108
CURRENT_RUN=0

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.1 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.1 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.01 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.01 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.001 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.001 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.0001 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.0001 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.1 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.1 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.01 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.01 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.001 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.001 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.0001 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.0001 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.1 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.1 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.01 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.01 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.001 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.001 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.0001 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.0001 bash ./run_edit.sh finetune artnet-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.1 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.1 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.01 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.01 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.001 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.001 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.0001 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.0001 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.1 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.1 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.01 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.01 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.001 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.001 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.0001 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.0001 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.1 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.1 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.01 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.01 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.001 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.001 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.0001 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.0001 bash ./run_edit.sh finetune artnet-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.1 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.1 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.01 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.01 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.001 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.001 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.0001 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.0001 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.1 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.1 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.01 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.01 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.001 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.001 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.0001 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.0001 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.1 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.1 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.01 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.01 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.001 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.001 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.0001 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.0001 bash ./run_edit.sh finetune artnet-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/artnet-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.1 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.1 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.01 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.01 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.001 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.001 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.0001 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.0001 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.1 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.1 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.01 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.01 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.001 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.001 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.0001 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.0001 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.1 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.1 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.01 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.01 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.001 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.001 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.0001 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.0001 bash ./run_edit.sh finetune tfinance GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.1 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.1 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.01 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.01 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.001 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.001 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.0001 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.0001 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.1 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.1 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.01 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.01 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.001 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.001 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.0001 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.0001 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.1 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.1 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.01 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.01 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.001 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.001 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.0001 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.0001 bash ./run_edit.sh finetune tfinance GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.1 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.1 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.01 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.01 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.001 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.001 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.0001 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.0001 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.1 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.1 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.01 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.01 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.001 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.001 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.0001 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.0001 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.1 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.1 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.01 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.01 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.001 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.001 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.0001 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.0001 bash ./run_edit.sh finetune tfinance GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/tfinance/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.1 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.1 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.01 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.01 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.001 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.001 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.0001 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.0001 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.1 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.1 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.01 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.01 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.001 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.001 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.0001 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.0001 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.1 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.1 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.01 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.01 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.001 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.001 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.0001 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.0001 bash ./run_edit.sh finetune twitch-views GAT_MLP 0 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.1 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.1 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.01 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.01 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.001 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.001 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.0001 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.0001 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.1 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.1 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.01 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.01 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.001 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.001 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.0001 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.0001 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.1 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.1 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.01 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.01 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.001 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.001 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.0001 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.0001 bash ./run_edit.sh finetune twitch-views GAT_MLP 10 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.1 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.1 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.01 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.01 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.001 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.001 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.0001 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=5 FT_LR=0.0001 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.1 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.1 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.01 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.01 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.001 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.001 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.0001 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=20 FT_LR=0.0001 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.1 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.1 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.01 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.01 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.001 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.001 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP

CURRENT_RUN=$((CURRENT_RUN + 1))
echo "========================================"
echo "Progress: $CURRENT_RUN / $TOTAL_RUNS"
echo "Running: TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.0001 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP"
echo "========================================"
TOP_FRACTION=0.25 NUM_LAYERS=2 FT_EPOCHS=50 FT_LR=0.0001 bash ./run_edit.sh finetune twitch-views GAT_MLP 42 1000 50 "" confidence /home/model_editing/data/seed_gnn_data/dataset /home/model_editing/data/seed_gnn_data/edit_ckpts /home/model_editing/data/editing_pipelines/finetune/twitch-views/GAT_MLP
