#!/usr/bin/env bash
# Run ablations/feature_ablation_forward.py on every edited checkpoint (paired metrics_*.json)
# for pokec, bail, yelp × GCN_MLP, GIN_MLP, Polynormer × leastsquares, finetune, egnn, seed_gnn.
#
# Each job is dispatched via launch_feature_ablation_from_metrics.py, which reads experiment
# metadata from the metrics JSON, resolves the matching .pt (including metrics_edit.json_* slugs),
# and calls feature_ablation_forward.py with --edited-checkpoint.
#
# Environment (optional):
#   PYTHON     Python executable (default: python3) — must have torch / project deps
#   DATASET_DIR, PRETRAIN_DIR, OUTPUT_ROOT — same defaults as feature_ablation_forward.py
#   FEATURE_ABLATION_REQUIRE_TOP_FRACTION — default 0.25. Skip metrics unless top_fraction matches
#       (edit_params, _top0.25_ in filename, or …/top_0.25/ in path). Set to "0" to allow any top.
#   LEASTSQUARES_STRATEGY_ALLOWLIST — comma/space-separated strategy names to keep for leastsquares only.
#       Default (when unset): sens_subspace_retention_pr_graphaware
#       Disable filtering: export LEASTSQUARES_STRATEGY_ALLOWLIST=  (empty string)
#   LEASTSQUARES_STANDARD_HPARAMS — default on (unset). Set to "0" to allow any λ / top / PR / τ / γ.
#       When on, leastsquares metrics must match λ=0.1, top=0.25, pra=0.85, rmx=0.75, γ=1 if specified.
#   FINETUNE_STANDARD_HPARAMS — default on (unset). Set to "0" to allow any finetune lr/epochs.
#       When on, only metrics_finetune_lr0.01_ep20_* (authoritative parse in Python launcher).
#   FEATURE_ABLATION_DETERMINISTIC=1 — ablation JSON under ABLATIONS_DIR uses no timestamp (overwrites).
#   FEATURE_ABLATION_SKIP_EXISTING — default on (unset). Set to "0" to re-run even if output JSON exists.
#   ABLATIONS_DIR — where ablation JSON is written / checked (default: $PATH_TO_DATA/ablations).
#   FEATURE_ABLATION_FILTER_DATASET — optional comma list (e.g. pokec,yelp); unset = all three.
#   FEATURE_ABLATION_FILTER_MODEL — optional (e.g. GCN_MLP,GIN_MLP).
#   FEATURE_ABLATION_FILTER_METHOD — optional (e.g. seed_gnn,finetune). Folder names under OUTPUT_ROOT.
#   DRY_RUN=1           — only list metrics files that would run
#   CONTINUE_ON_ERROR=1 — keep going after a failure
#
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../paths.sh
source "${REPO}/paths.sh"
ABLATION_PY="${SCRIPT_DIR}/feature_ablation_forward.py"
LAUNCH_PY="${SCRIPT_DIR}/launch_feature_ablation_from_metrics.py"

PYTHON="${PYTHON:-python3}"

# When unset: require top_fraction == 0.25 for every method (see launch_feature_ablation_from_metrics.py).
FEATURE_ABLATION_REQUIRE_TOP_FRACTION="${FEATURE_ABLATION_REQUIRE_TOP_FRACTION-0.25}"
# When unset: only this LS strategy. Use LEASTSQUARES_STRATEGY_ALLOWLIST= to run all LS strategies.
LEASTSQUARES_STRATEGY_ALLOWLIST="${LEASTSQUARES_STRATEGY_ALLOWLIST-sens_subspace_retention_pr_graphaware}"
# When unset: enforce standard LS hyperparameters (see launch_feature_ablation_from_metrics.py).
LEASTSQUARES_STANDARD_HPARAMS="${LEASTSQUARES_STANDARD_HPARAMS-1}"

# When unset: finetune only lr=0.01, ep=20. Disable with FINETUNE_STANDARD_HPARAMS=0
FINETUNE_STANDARD_HPARAMS="${FINETUNE_STANDARD_HPARAMS-1}"
ABLATIONS_DIR="${ABLATIONS_DIR:-${PATH_TO_DATA}/ablations}"
FEATURE_ABLATION_SKIP_EXISTING="${FEATURE_ABLATION_SKIP_EXISTING-1}"

export ABLATION_PY DATASET_DIR PRETRAIN_DIR OUTPUT_ROOT PYTHON LEASTSQUARES_STRATEGY_ALLOWLIST LEASTSQUARES_STANDARD_HPARAMS FINETUNE_STANDARD_HPARAMS FEATURE_ABLATION_REQUIRE_TOP_FRACTION FEATURE_ABLATION_FILTER_DATASET FEATURE_ABLATION_FILTER_MODEL FEATURE_ABLATION_FILTER_METHOD ABLATIONS_DIR FEATURE_ABLATION_SKIP_EXISTING

# Optional comma-separated subsets (unset = full suite for that axis).
if [[ -n "${FEATURE_ABLATION_FILTER_DATASET:-}" ]]; then
  _ds="${FEATURE_ABLATION_FILTER_DATASET// /}"
  IFS=',' read -r -a DATASETS <<< "${_ds}"
else
  DATASETS=(pokec bail yelp)
fi
if [[ -n "${FEATURE_ABLATION_FILTER_MODEL:-}" ]]; then
  _mo="${FEATURE_ABLATION_FILTER_MODEL// /}"
  IFS=',' read -r -a MODELS <<< "${_mo}"
else
  MODELS=(GCN_MLP GIN_MLP Polynormer)
fi
if [[ -n "${FEATURE_ABLATION_FILTER_METHOD:-}" ]]; then
  _me="${FEATURE_ABLATION_FILTER_METHOD// /}"
  IFS=',' read -r -a METHODS <<< "${_me}"
else
  METHODS=(leastsquares finetune egnn seed_gnn)
fi

dry_run="${DRY_RUN:-0}"
cont="${CONTINUE_ON_ERROR:-0}"

# Fast path: skip leastsquares JSONs whose basename cannot match the allowlist (strategy slug in filename).
ls_metrics_matches_allowlist() {
  local path="$1"
  local allow="$2"
  [[ -z "${allow}" ]] && return 0
  local base bn
  base="$(basename "${path}")"
  bn="${base#metrics_}"
  local IFS=' ,'
  local tok
  for tok in ${allow}; do
    [[ -z "${tok}" ]] && continue
    [[ "${base}" == *"${tok}"* ]] || [[ "${bn}" == *"${tok}"* ]] && return 0
  done
  return 1
}

# Basename quick-check for standard LS hypers (authoritative parse is in the Python launcher).
ls_metrics_standard_hparams_basename() {
  local base="$1"
  [[ "${base}" == *"_lam0.1_"* ]] || return 1
  [[ "${base}" == *"_top0.25_"* ]] || return 1
  [[ "${base}" == *"_pra0.85_"* ]] || return 1
  [[ "${base}" == *"_rmx0.75_"* ]] || return 1
  if [[ "${base}" == *"_gamma"* ]]; then
    [[ "${base}" == *"_gamma1.0_"* ]] || [[ "${base}" == *"_gamma1_"* ]] || return 1
  fi
  return 0
}

# Finetune: only lr=0.01 and 20 epochs (matches metrics_finetune_lr0.01_ep20_* stem).
finetune_lr_ep_basename() {
  local bn="$1"
  [[ "${bn}" == *_lr0.01_ep20_* ]] || return 1
  return 0
}

total=0
ok=0
fail=0

for method in "${METHODS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      base="${OUTPUT_ROOT}/${method}/${ds}/${model}"
      [[ -d "$base" ]] || continue
      while IFS= read -r -d '' metrics; do
        mbase="$(basename "${metrics}")"
        if [[ "${method}" == "leastsquares" ]] && ! ls_metrics_matches_allowlist "${metrics}" "${LEASTSQUARES_STRATEGY_ALLOWLIST}"; then
          continue
        fi
        if [[ "${method}" == "leastsquares" && "${LEASTSQUARES_STANDARD_HPARAMS}" != "0" ]]; then
          ls_metrics_standard_hparams_basename "${mbase}" || continue
        fi
        if [[ "${method}" == "finetune" && "${FINETUNE_STANDARD_HPARAMS}" != "0" ]]; then
          finetune_lr_ep_basename "${mbase}" || continue
        fi
        total=$((total + 1))
        if [[ "${dry_run}" == "1" ]]; then
          echo "[dry-run] ${metrics}"
          ok=$((ok + 1))
          continue
        fi
        export METRICS_JSON="$metrics"
        set +e
        out="$("$PYTHON" "$LAUNCH_PY" 2>&1)"
        rc=$?
        set -e
        if [[ "$rc" -eq 0 ]]; then
          ok=$((ok + 1))
          printf '%s\n' "$out" | tail -n 3
        else
          fail=$((fail + 1))
          printf '%s\n' "$out"
          if [[ "${cont}" != "1" ]]; then
            echo "Failed (exit ${rc}) on ${metrics} — set CONTINUE_ON_ERROR=1 to keep going." >&2
            exit "$rc"
          fi
        fi
      done < <(find "$base" -type f -name 'metrics_*.json' -print0 | sort -z)
    done
  done
done

echo "Done. metrics files: ${total}  ok: ${ok}  failed: ${fail}" >&2
[[ "$fail" -eq 0 ]]
