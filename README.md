# GRAFT (Gradient-Free Editing for GNNs)

GRAFT contains code and shell drivers for gradient-free editing experiments on graph neural networks.

## Overview

- **`run_*.sh` scripts** live under `ablations/`, `editing_pipelines/`, and `seed-gnn/` and are the usual entrypoints for batch jobs.
- **Defaults** assume data under `/home/model_editing/data/...`. Most scripts accept overrides (`DATASET_DIR`, `PRETRAIN_DIR`, `OUTPUT_ROOT`, `OUTPUT_DIR_ROOT`, etc.); check each file’s header for the exact list.
- **More detail** on layout and configs: [`editing_pipelines/README.md`](editing_pipelines/README.md), [`seed-gnn/README.md`](seed-gnn/README.md).

## Environment

The working environment is captured as conda env **`gnn_edit`**:

1. **Full conda solve (recommended)** — reproduces Python, CUDA-related conda packages, and pip extras:

   ```bash
   conda env create -f environment.yml
   conda activate gnn_edit
   ```

2. **Pip-only pin list** — [`requirements.txt`](requirements.txt) lists PyPI packages as pinned in that env. Some lines (PyTorch `+cu118`, `pyg-lib` / `torch-scatter`, etc.) normally need the official PyTorch / PyG wheel indexes; if `pip install -r requirements.txt` fails on those, install them the way [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pyg.org/) document for your CUDA version, or stick to `environment.yml`.

`environment.yml` was produced with `conda env export -n gnn_edit --no-builds` and **without** the `prefix:` field so paths stay machine-agnostic.

## Experiment scripts (`run_*.sh`)

### Editing pipelines (`editing_pipelines/`)

**Hub**

- **`run_edit.sh`** — Wraps [`run_edit.py`](editing_pipelines/run_edit.py): method, dataset, model, seed, target counts, max steps, strategies, and data/pretrain/output paths. Reads optional hyperparameters from the environment (`LAMBDA_REG`, `TOP_FRACTION`, `NUM_LAYERS`, `GAMMA_RETAIN`, `FT_EPOCHS`, `FT_LR`, `PR_ALPHA`, `RANK_MIX_TAU`). The script sets `PYTHONPATH` from a fixed root; change it if your clone is not in that location.

**Parameter sweeps** (loop and call `run_edit.sh`)

- **`run_editing_suite.sh`** — Large least-squares grid: λ, top fraction, depth, many `LS_STRATEGY` modes (or `LS_STRATEGY_GROUPS`), optional PageRank/DivRank sweeps (`PR_ALPHAS`, `RANK_MIX_TAUS`). Uses `run_edit.sh` with `--debug`.
- **`run_lambda.sh`** — Stress-tests regularization: large `LAMBDA_REG` values and multiple top fractions; default strategy `sens_divrank`; default datasets income and tfinance.
- **`run_seed_gnn.sh`** — Seed-GNN baseline over datasets, models, seeds, top fractions, depth (`NUM_LAYERS` / `LAYERS_OVERRIDE`).
- **`run_egnn.sh`** — EGNN baseline; same idea, optional `layers_<n>/top_<frac>` paths when layers are overridden.
- **`run_finetune.sh`** — Finetune baseline over models, seeds, top fraction, `FT_EPOCHS` / `FT_LR`, and depth.
- **`run_time_ablation.sh`** — Depth ablation for one `(dataset, model, method)`: seed 42, strategy `sens_subspace_retention_pr_graphaware`, layers 2/3/5.  
  Example: `bash run_time_ablation.sh <dataset> <model> <method>`.

**Auxiliary**

- **`run_sif.sh`** — Runs `shell_scripts/compute_sif.sh` for bail, income, pokec (`NUM_SAMPLES` default 50).
- **`run_sensitivity.sh`** — Dataset-specific sensitivity runs under `scripts/sensitivity/`; `RUN_ABLATION` toggles full vs. ablated-model analysis.

### Ablations (`ablations/`)

- **`run_feature_ablation_all_checkpoints.sh`** — Finds `metrics_*.json` under `OUTPUT_ROOT` for pokec / bail / yelp, GCN_MLP / GIN_MLP / Polynormer, and methods leastsquares / finetune / egnn / seed_gnn. Delegates to `launch_feature_ablation_from_metrics.py` → `feature_ablation_forward.py`. Supports `FEATURE_ABLATION_FILTER_*`, `DRY_RUN=1`, and `FEATURE_ABLATION_SKIP_EXISTING`.

### Seed-GNN pretraining (`seed-gnn/`)

- **`run_node_regression.sh`** — Pretrain on node-regression sets (e.g. twitch-views, artnet-views); sweeps gcn/sage/gin/gat, seeds, depth via `scripts/pretrain/seed_gnn/`.
- **`run_feature_drop_pretrain_suite.sh`** — Pretrain with one sensitive column dropped per dataset (default pokec/bail/yelp/tfinance), aligned with editing suite attribute names; outputs under `edit_ckpts_feature_ablated/.../no_<feature>/`. Arguments after `--` forward to `main.py`.
- **`run_feature_drop_pretrain_regression.sh`** — Same pattern for regression graphs and their sensitive columns.
