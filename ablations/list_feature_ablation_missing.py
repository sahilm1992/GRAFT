#!/usr/bin/env python3
"""
List required feature-ablation runs (same eligibility as run_feature_ablation_all_checkpoints.sh +
launch_feature_ablation_from_metrics.py) and whether ABLATIONS_DIR already has a matching JSON.

Examples::

  # All missing (defaults: OUTPUT_ROOT / ABLATIONS_DIR from env or paths below)
  python ablations/list_feature_ablation_missing.py --only-missing

  # Filter (comma-separated or repeatable flags)
  python ablations/list_feature_ablation_missing.py --dataset yelp --model GCN_MLP --method seed_gnn
  python ablations/list_feature_ablation_missing.py --dataset pokec,bail --method egnn,seed_gnn

Environment (optional, same as batch driver): FEATURE_ABLATION_REQUIRE_TOP_FRACTION,
LEASTSQUARES_*, FINETUNE_STANDARD_HPARAMS, OUTPUT_ROOT, ABLATIONS_DIR.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import launch_feature_ablation_from_metrics as L

METHODS_ALL = ("leastsquares", "finetune", "egnn", "seed_gnn")
DATASETS_ALL = ("pokec", "bail", "yelp")
MODELS_ALL = ("GCN_MLP", "GIN_MLP", "Polynormer")


def _split_csv(vals: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not vals:
        return None
    out: set[str] = set()
    for v in vals:
        for part in re.split(r"[\s,]+", v.strip()):
            if part:
                out.add(part)
    return out


def _scan(
    output_root: Path,
    ablations_dir: Path,
    *,
    datasets: Optional[set[str]],
    models: Optional[set[str]],
    methods: Optional[set[str]],
) -> Tuple[List[Dict[str, Any]], Counter]:
    """Eligible rows (same filters as driver + launcher)."""
    skipped: Counter = Counter()
    rows: List[Dict[str, Any]] = []

    allow_ls = os.environ.get("LEASTSQUARES_STRATEGY_ALLOWLIST", "").strip()

    for method_dir in METHODS_ALL:
        if methods is not None and method_dir not in methods:
            continue
        for ds in DATASETS_ALL:
            if datasets is not None and ds not in datasets:
                continue
            for model in MODELS_ALL:
                if models is not None and model not in models:
                    continue
                base = output_root / method_dir / ds / model
                if not base.is_dir():
                    continue
                for metrics_path in sorted(base.rglob("metrics_*.json")):
                    if not metrics_path.is_file():
                        continue
                    mbase = metrics_path.name
                    if method_dir == "leastsquares" and not L.ls_metrics_matches_allowlist_filename(
                        mbase, allow_ls
                    ):
                        skipped["leastsquares_allowlist"] += 1
                        continue
                    if method_dir == "leastsquares" and os.environ.get(
                        "LEASTSQUARES_STANDARD_HPARAMS", "1"
                    ) != "0":
                        if not L.ls_metrics_standard_hparams_basename_ok(mbase):
                            skipped["leastsquares_basename_hp"] += 1
                            continue
                    if method_dir == "finetune" and os.environ.get(
                        "FINETUNE_STANDARD_HPARAMS", "1"
                    ) != "0":
                        if not L.finetune_metrics_basename_lr_ep_ok(mbase):
                            skipped["finetune_basename_lr_ep"] += 1
                            continue

                    try:
                        with open(metrics_path, encoding="utf-8") as f:
                            payload = json.load(f)
                    except Exception:
                        skipped["json_error"] += 1
                        continue

                    exp = payload.get("experiment") or {}
                    data = payload.get("data") or {}
                    model_o = payload.get("model") or {}
                    meth = (exp.get("method") or "").strip()
                    dataset = (data.get("dataset") or "").strip()
                    mname = (model_o.get("name") or "").strip()
                    seed = int(exp.get("seed", 0))

                    if meth not in METHODS_ALL:
                        skipped["wrong_method"] += 1
                        continue
                    ok_filt, _ = L.dmf_filter_ok(dataset, mname, meth)
                    if not ok_filt:
                        skipped["dmf_mismatch"] += 1
                        continue
                    ok_top, _ = L.top_fraction_requirement_ok(
                        payload, metrics_path.stem, metrics_path
                    )
                    if not ok_top:
                        skipped["top_fraction"] += 1
                        continue

                    ep = payload.get("edit_params") or {}
                    sel = payload.get("selection_correlation_metrics") or {}

                    if meth == "leastsquares":
                        strategy = sel.get("strategy_mode") or L.parse_ls_strategy_from_stem(
                            metrics_path.stem
                        )
                        if not strategy:
                            skipped["ls_no_strategy"] += 1
                            continue
                        if allow_ls:
                            allowed = {
                                x.strip()
                                for x in re.split(r"[\s,]+", allow_ls)
                                if x.strip()
                            }
                            if strategy not in allowed:
                                skipped["ls_strategy_json"] += 1
                                continue
                        if os.environ.get("LEASTSQUARES_STANDARD_HPARAMS", "1") != "0":
                            ok_hp, _ = L.least_squares_standard_hparams_ok(
                                payload, metrics_path.stem
                            )
                            if not ok_hp:
                                skipped["ls_std_hp"] += 1
                                continue

                    if meth == "finetune":
                        if os.environ.get("FINETUNE_STANDARD_HPARAMS", "1") != "0":
                            ok_ft, _ = L.finetune_lr_epochs_ok(payload, metrics_path.stem)
                            if not ok_ft:
                                skipped["finetune_std_hp"] += 1
                                continue

                    try:
                        ckpt = L.resolve_ckpt(metrics_path, payload)
                    except Exception as e:
                        rows.append(
                            {
                                "status": "blocked",
                                "method": meth,
                                "dataset": dataset,
                                "model": mname,
                                "seed": seed,
                                "metrics_path": str(metrics_path),
                                "ckpt_path": "",
                                "ablation_prefix": "",
                                "detail": str(e)[:500],
                            }
                        )
                        continue

                    prefix = L._ablation_basename_prefix(meth, dataset, mname, seed, ckpt)
                    existing = L.existing_feature_ablation_outputs(ablations_dir, prefix)
                    rows.append(
                        {
                            "status": "done" if existing else "missing",
                            "method": meth,
                            "dataset": dataset,
                            "model": mname,
                            "seed": seed,
                            "metrics_path": str(metrics_path),
                            "ckpt_path": str(ckpt),
                            "ablation_prefix": prefix,
                            "ablation_json": str(existing[0]) if existing else "",
                            "detail": "",
                        }
                    )

    return rows, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get("OUTPUT_ROOT", "/home/model_editing/data/editing_pipelines")
        ),
    )
    ap.add_argument(
        "--ablations-dir",
        type=Path,
        default=Path(os.environ.get("ABLATIONS_DIR", L.DEFAULT_ABLATIONS_DIR)),
    )
    ap.add_argument("--dataset", action="append", dest="datasets", help="e.g. yelp or pokec,bail")
    ap.add_argument("--model", action="append", dest="models")
    ap.add_argument("--method", action="append", dest="methods")
    ap.add_argument(
        "--only-missing",
        action="store_true",
        help="print only status=missing (excludes blocked)",
    )
    ap.add_argument(
        "--csv",
        action="store_true",
        help="machine-readable CSV to stdout",
    )
    args = ap.parse_args()

    os.environ.setdefault("FEATURE_ABLATION_REQUIRE_TOP_FRACTION", "0.25")
    os.environ.setdefault("LEASTSQUARES_STRATEGY_ALLOWLIST", "sens_subspace_retention_pr_graphaware")
    os.environ.setdefault("LEASTSQUARES_STANDARD_HPARAMS", "1")
    os.environ.setdefault("FINETUNE_STANDARD_HPARAMS", "1")

    datasets = _split_csv(args.datasets)
    models = _split_csv(args.models)
    methods = _split_csv(args.methods)

    out_root = args.output_root.expanduser().resolve()
    abl_dir = args.ablations_dir.expanduser().resolve()

    rows, skipped = _scan(
        out_root,
        abl_dir,
        datasets=datasets,
        models=models,
        methods=methods,
    )

    missing = [r for r in rows if r["status"] == "missing"]
    done = [r for r in rows if r["status"] == "done"]
    blocked = [r for r in rows if r["status"] == "blocked"]

    if args.csv:
        w = csv.DictWriter(
            sys.stdout,
            fieldnames=[
                "status",
                "method",
                "dataset",
                "model",
                "seed",
                "metrics_path",
                "ckpt_path",
                "ablation_prefix",
                "ablation_json",
                "detail",
            ],
        )
        w.writeheader()
        for r in rows:
            if args.only_missing and r["status"] != "missing":
                continue
            w.writerow(r)
        return 0

    print(f"output_root:   {out_root}  exists={out_root.is_dir()}")
    print(f"ablations_dir: {abl_dir}  exists={abl_dir.is_dir()}")
    print(
        f"checkpoint ok: {len(done) + len(missing)}  (done: {len(done)}, missing: {len(missing)}); "
        f"blocked ckpt resolve: {len(blocked)}"
    )
    print(f"skipped files (not in required set): {sum(skipped.values())}")
    for k, v in skipped.most_common():
        print(f"  {k}: {v}")

    print()
    if args.only_missing:
        print("MISSING:")
        for r in missing:
            print(
                f"  {r['method']}\t{r['dataset']}\t{r['model']}\tseed{r['seed']}\t{r['metrics_path']}"
            )
    else:
        print("MISSING:")
        for r in missing:
            print(
                f"  {r['method']}\t{r['dataset']}\t{r['model']}\tseed{r['seed']}\t{r['metrics_path']}"
            )
        if blocked:
            print("\nBLOCKED (checkpoint resolve failed):")
            for r in blocked:
                print(
                    f"  {r['method']}\t{r['dataset']}\t{r['model']}\tseed{r['seed']}\t{r['metrics_path']}"
                )
                print(f"    -> {r['detail']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
