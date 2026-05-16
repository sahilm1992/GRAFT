#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEED_GNN_DIR="${ROOT_DIR}/seed-gnn"
DATA_DIR="/home/model_editing/data/seed_gnn_data/dataset"
OUTPUT_DIR="/home/model_editing/data/feature_manipulation"
CONFIG_DIR="${SEED_GNN_DIR}/config"

DATASET=""
FEATURE=""
FEATURE_TYPE="binary"
MODEL="gcn"
PIPELINE_METHOD="seed_gnn"
EDIT_METHOD="leastsquares"
EXP_NAME="feature_bias"
P="0.8"
ALPHA="1.0"
SIGMA="1.0"
SEED="42"
TRAIN_CORRS="positive,negative,zero"
VAL_CORRS="positive,negative,zero"

usage() {
  cat <<EOF
Usage: $(basename "$0") --dataset DATASET --feature FEATURE [options]

Required:
  --dataset DATASET           Dataset name (e.g., bail, income, pokec)
  --feature FEATURE           Feature name or index to manipulate

Options:
  --feature-type TYPE         binary|categorical|continuous (default: ${FEATURE_TYPE})
  --model MODEL               gcn|gat|gin|sage (default: ${MODEL})
  --pipeline-method METHOD    seed_gnn|egnn (default: ${PIPELINE_METHOD})
  --edit-method METHOD        leastsquares|egnn|seed_gnn|ewc|hyper_gnn (default: ${EDIT_METHOD})
  --exp-name NAME             Experiment name (default: ${EXP_NAME})
  --p FLOAT                   Bernoulli p for binary/categorical (default: ${P})
  --alpha FLOAT               Alpha for continuous feature (default: ${ALPHA})
  --sigma FLOAT               Sigma for continuous feature (default: ${SIGMA})
  --seed INT                  Random seed (default: ${SEED})
  --train-corrs LIST          Comma-separated list (default: ${TRAIN_CORRS})
  --val-corrs LIST            Comma-separated list (default: ${VAL_CORRS})

Outputs:
  feature_manipulation/experiments/[EXP_NAME]/train_*__val_*__test_*/
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --feature) FEATURE="$2"; shift 2 ;;
    --feature-type) FEATURE_TYPE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --pipeline-method) PIPELINE_METHOD="$2"; shift 2 ;;
    --edit-method) EDIT_METHOD="$2"; shift 2 ;;
    --exp-name) EXP_NAME="$2"; shift 2 ;;
    --p) P="$2"; shift 2 ;;
    --alpha) ALPHA="$2"; shift 2 ;;
    --sigma) SIGMA="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --train-corrs) TRAIN_CORRS="$2"; shift 2 ;;
    --val-corrs) VAL_CORRS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${DATASET}" || -z "${FEATURE}" ]]; then
  echo "Error: --dataset and --feature are required."
  usage
  exit 1
fi

PIPELINE_CONFIG="${CONFIG_DIR}/pipeline_config/${PIPELINE_METHOD}/${MODEL}/${DATASET}.json"
EVAL_CONFIG="${CONFIG_DIR}/eval_config/edit_gnn/${DATASET}.json"

if [[ ! -f "${PIPELINE_CONFIG}" ]]; then
  echo "Missing pipeline config: ${PIPELINE_CONFIG}"
  exit 1
fi
if [[ ! -f "${EVAL_CONFIG}" ]]; then
  echo "Missing eval config: ${EVAL_CONFIG}"
  exit 1
fi

env \
  ROOT_DIR="${ROOT_DIR}" \
  SEED_GNN_DIR="${SEED_GNN_DIR}" \
  DATA_DIR="${DATA_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  PIPELINE_CONFIG="${PIPELINE_CONFIG}" \
  EVAL_CONFIG="${EVAL_CONFIG}" \
  DATASET="${DATASET}" \
  FEATURE="${FEATURE}" \
  FEATURE_TYPE="${FEATURE_TYPE}" \
  MODEL="${MODEL}" \
  PIPELINE_METHOD="${PIPELINE_METHOD}" \
  EDIT_METHOD="${EDIT_METHOD}" \
  EXP_NAME="${EXP_NAME}" \
  P="${P}" \
  ALPHA="${ALPHA}" \
  SIGMA="${SIGMA}" \
  SEED="${SEED}" \
  TRAIN_CORRS="${TRAIN_CORRS}" \
  VAL_CORRS="${VAL_CORRS}" \
  PYTHONPATH="${ROOT_DIR}:${SEED_GNN_DIR}" \
  python - <<'PY'
import json
import os
import sys
from pathlib import Path

root_dir = Path(os.environ["ROOT_DIR"])
seed_gnn_dir = Path(os.environ["SEED_GNN_DIR"])
data_dir = Path(os.environ["DATA_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"])
pipeline_config_path = Path(os.environ["PIPELINE_CONFIG"])
eval_config_path = Path(os.environ["EVAL_CONFIG"])

raw_feature = os.environ["FEATURE"]
try:
    feature = int(raw_feature)
except ValueError:
    feature = raw_feature
feature_type = os.environ["FEATURE_TYPE"]
edit_method = os.environ["EDIT_METHOD"]
exp_name = os.environ["EXP_NAME"]
p = float(os.environ["P"])
alpha = float(os.environ["ALPHA"])
sigma = float(os.environ["SIGMA"])
seed = int(os.environ["SEED"])
train_corrs = os.environ["TRAIN_CORRS"].split(",")
val_corrs = os.environ["VAL_CORRS"].split(",")

sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(seed_gnn_dir))

from feature_manipulation import RUN_EXPERIMENT_FN

with open(pipeline_config_path, "r") as handle:
    pipeline_cfg = json.load(handle)
with open(eval_config_path, "r") as handle:
    eval_cfg = json.load(handle)

pipeline_params = pipeline_cfg.get("pipeline_params", {})
dataset_name = str(eval_cfg["eval_params"]["dataset"]).lower()

sensitive_feature_map = {
    "pokec": "AGE",
    "income": "race",
    "bail": "TIME",
    "credit": "Age",
}
sensitive_feature_value_map = {
    "pokec": [15, 25, 35, 45, 55, 65],
    "income": None,
    "bail": None,
    "credit": None,
}

pipeline_params.setdefault("leastsquares", {
    "lambda_reg": 10,
    "bias_attr_idx": [],
    "use_mlp_linears": False,
    "skip_non_gnn_linears": True,
    "mean_steering_signal": True,
    "sensitivity_based": True,
    "only_correct": False,
    "top_fraction": 0.25,
    "seed": seed,
    "metrics_save_name": "metrics_edit.json",
})
ls_cfg = pipeline_params["leastsquares"]
ls_cfg["mean_steering_signal"] = True
ls_cfg["sensitivity_based"] = True
ls_cfg["only_correct"] = False
ls_cfg["weighted_mean_signal"] = False
ls_cfg["strategy_mode"] = "sensitivity_mean"
pipeline_params.setdefault("sensitive_feature", sensitive_feature_map.get(dataset_name))
pipeline_params.setdefault("fixed_sensitive_values", sensitive_feature_value_map.get(dataset_name))
pipeline_params.setdefault("normalize", True)
pipeline_params.setdefault("loop", True)
pipeline_params.setdefault("epochs", 500)
pipeline_cfg["pipeline_params"] = pipeline_params

base_config = {
    "management": {
        "dataset_dir": str(data_dir),
        "pretrain_output_dir": str(output_dir / "experiments"),
        "output_folder_dir": str(output_dir / "experiments"),
        "exp_desc": exp_name,
        "task": "edit",
        "seed": seed,
    },
    "pipeline_params": pipeline_cfg["pipeline_params"],
    "eval_params": eval_cfg["eval_params"],
}
if "corruption" in eval_cfg:
    base_config["corruption"] = eval_cfg["corruption"]

RUN_EXPERIMENT_FN(
    base_config=base_config,
    exp_name=exp_name,
    feature=feature,
    feature_type=feature_type,
    p=p,
    alpha=alpha,
    sigma=sigma,
    seed=seed,
    train_corrs=train_corrs,
    val_corrs=val_corrs,
    method=edit_method,
)
PY
