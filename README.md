# GRAFT (Gradient-Free Editing for GNNs)

GRAFT contains code and shell drivers for gradient-free editing experiments on graph neural networks.

## Overview

- `run_*.sh` scripts live under `ablations/`, `editing_pipelines/`, and `seed-gnn/` and are the usual entrypoints for batch jobs.
- **Path defaults**: Repo-root `[paths.py](paths.py)` and `[paths.sh](paths.sh)` set `PATH_TO_GRAFT` (default: the repository containing those files, typically `/home/model_editing/GRAFT`) and `PATH_TO_DATA` (default `/home/model_editing/data`). Shell drivers `source paths.sh`; override locations with env vars (`DATASET_DIR`, `PRETRAIN_DIR`, etc. still work as documented in each script).
- **Dataset preparation**: Import **raw** graphs **dataset-by-dataset** from the canonical repos, preprocess to match this codebase’s CSV layout under `DATASET_DIR` (see `[paths.sh](paths.sh)`), and respect each publisher’s license.  
**[GADBench](https://github.com/squareRoot3/GADBench/tree/master)** — anomaly / fraud benchmarks used here include **YelpChi** (mapped as **yelp**) and **T‑Finance** (mapped as **tfinance**); use its `[datasets](https://github.com/squareRoot3/GADBench/tree/master/datasets)` tree and preprocessing notebook as in that README.  
**[GraphLand](https://github.com/yandex-research/graphland/tree/main)** — regression targets such as **artnet-views** and **twitch-views** (`run_node_regression.sh`): follow `[data](https://github.com/yandex-research/graphland/tree/main/data)` plus `dataset.py` / `PyGDataset` in that repo.  
**Pokec** — take the release from **[GUIDE](https://github.com/michaelweihaosong/GUIDE)** (KDD 2022; *Group Equality Informed Individual Fairness in GNNs*): files live under `[dataset](https://github.com/michaelweihaosong/GUIDE/tree/main/dataset)` (**pokec** is distributed zipped there—unzip before use), then convert to this repo’s CSV layout under `DATASET_DIR`.  
**Bail** — take the fairness benchmark graphs from **[NIFTY](https://github.com/chirag-agarwall/nifty)** (*Towards a Unified Framework for Fair and Stable Graph Representation Learning*, [arXiv:2102.13186](https://arxiv.org/abs/2102.13186)): data live under `[dataset](https://github.com/chirag-agarwall/nifty/tree/main/dataset)` (that repo README notes large edge bundles may be zipped), then preprocess to match `DATASET_DIR` locally.
- **More detail** on layout and configs: `[editing_pipelines/README.md](editing_pipelines/README.md)`, `[seed-gnn/README.md](seed-gnn/README.md)`.

## Environment

The working environment is captured as conda env `**gnn_edit`**:

1. **Full conda solve (recommended)** — reproduces Python, CUDA-related conda packages, and pip extras:
  ```bash
   conda env create -f environment.yml
   conda activate gnn_edit
  ```
2. **Pip-only pin list** — `[requirements.txt](requirements.txt)` lists PyPI packages as pinned in that env. Some lines (PyTorch `+cu118`, `pyg-lib` / `torch-scatter`, etc.) normally need the official PyTorch / PyG wheel indexes; if `pip install -r requirements.txt` fails on those, install them the way [PyTorch](https://pytorch.org/get-started/locally/) and [PyG](https://pyg.org/) document for your CUDA version, or stick to `environment.yml`.

`environment.yml` was produced with `conda env export -n gnn_edit --no-builds` and **without** the `prefix:` field so paths stay machine-agnostic.

## Experiment scripts (`run_*.sh`)

### Editing pipelines (`editing_pipelines/`)

**Hub**

- `run_edit.sh` — Wraps `[run_edit.py](editing_pipelines/run_edit.py)`: method, dataset, model, seed, target counts, max steps, strategies, and data/pretrain/output paths. Reads optional hyperparameters from the environment (`LAMBDA_REG`, `TOP_FRACTION`, `NUM_LAYERS`, `GAMMA_RETAIN`, `FT_EPOCHS`, `FT_LR`, `PR_ALPHA`, `RANK_MIX_TAU`). Sets `PYTHONPATH` via `[paths.sh](paths.sh)`.

**Parameter sweeps** (loop and call `run_edit.sh`)

- `run_editing_suite.sh` — Large least-squares grid: λ, top fraction, depth, many `LS_STRATEGY` modes (or `LS_STRATEGY_GROUPS`), optional PageRank/DivRank sweeps (`PR_ALPHAS`, `RANK_MIX_TAUS`). Uses `run_edit.sh` with `--debug`.
- `run_seed_gnn.sh` — Seed-GNN baseline over datasets, models, seeds, top fractions, depth (`NUM_LAYERS` / `LAYERS_OVERRIDE`).
- `run_egnn.sh` — EGNN baseline; same idea, optional `layers_<n>/top_<frac>` paths when layers are overridden.
- `run_finetune.sh` — Finetune baseline over models, seeds, top fraction, `FT_EPOCHS` / `FT_LR`, and depth.

### Ablations (`ablations/`)

- `run_feature_ablation_all_checkpoints.sh` — Finds `metrics_*.json` under `OUTPUT_ROOT` for pokec / bail / yelp, GCN_MLP / GIN_MLP / Polynormer, and methods leastsquares / finetune / egnn / seed_gnn. Delegates to `launch_feature_ablation_from_metrics.py` → `feature_ablation_forward.py`. Supports `FEATURE_ABLATION_FILTER_*`, `DRY_RUN=1`, and `FEATURE_ABLATION_SKIP_EXISTING`.

### Seed-GNN pretraining (`seed-gnn/`)

- **[`run.sh`](seed-gnn/run.sh)** — Main vanilla pretraining loop: configure the `datasets`, `models`, `seeds`, and `num_layers` arrays plus optional **feature-ablation** block at the top of the script, then it calls `scripts/pretrain/seed_gnn/<dataset>.sh` per combination (two path arguments: output root and dataset dir appear as `~/data/seed_gnn_data` / `~/data/seed_gnn_data/dataset` in the defaults—adjust to match your `paths.sh` layout). Accepts **`--architectures`**, **`--gat_optimized`**, **`--neighbor_batch_size`**, and **`--neighbor_num_neighbors`** forwarded to each pretrain invocation.
- `run_node_regression.sh` — Pretrain on node-regression sets (e.g. twitch-views, artnet-views); sweeps gcn/sage/gin/gat, seeds, depth via `scripts/pretrain/seed_gnn/`.
- `run_feature_drop_pretrain_suite.sh` — Pretrain with one sensitive column dropped per dataset (default pokec/bail/yelp/tfinance), aligned with editing suite attribute names; outputs under `edit_ckpts_feature_ablated/.../no_<feature>/`. Arguments after `--` forward to `main.py`.
- `run_feature_drop_pretrain_regression.sh` — Same pattern for regression graphs and their sensitive columns.

