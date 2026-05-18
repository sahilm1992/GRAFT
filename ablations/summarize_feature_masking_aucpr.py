#!/usr/bin/env python3
"""
Aggregate feature-ablation JSON files (output of feature_ablation_forward.py) into:

- **CSV** — machine-readable; full ``edited_checkpoint`` paths.
- **Markdown** — one table per **(dataset, model)**; rows = editing **strategy** and **seed**
  (plus **Checkpoint** when the same strategy/seed has multiple edited checkpoints); columns =
  AUC-PR masking metrics (train / val / test). Full flat export stays in the CSV.

Drops are **original AUC-PR − masked AUC-PR** (positive ⇒ worse under masking).
``edited_minus_pretrain_mask_drop`` is negative when the edit is *more* robust than pretrain.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

SPLITS = ("train", "val", "test")

METHOD_ORDER = ("leastsquares", "finetune", "egnn", "seed_gnn")
DATASET_ORDER = ("pokec", "bail", "yelp", "tfinance", "credit")
MODEL_ORDER = ("GCN_MLP", "GIN_MLP", "Polynormer", "SAGE_MLP", "GAT_MLP")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import paths as _paths  # noqa: E402


def _method_sort_key(m: str) -> Tuple[int, str]:
    try:
        return (METHOD_ORDER.index(m), m)
    except ValueError:
        return (len(METHOD_ORDER), m)


def _dataset_sort_key(d: str) -> Tuple[int, str]:
    try:
        return (DATASET_ORDER.index(d), d)
    except ValueError:
        return (len(DATASET_ORDER), d)


def _model_sort_key(m: str) -> Tuple[int, str]:
    try:
        return (MODEL_ORDER.index(m), m)
    except ValueError:
        return (len(MODEL_ORDER), m)


def _finite(x: Any) -> bool:
    try:
        v = float(x)
        return math.isfinite(v)
    except (TypeError, ValueError):
        return False


def _f(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def dataset_model_from_checkpoint(ckpt: str, method: str) -> Tuple[str, str]:
    """Expect .../<method>/<dataset>/<model>/... under editing_pipelines roots."""
    if not ckpt:
        return "", ""
    parts = Path(ckpt.replace("\\", "/")).parts
    try:
        i = parts.index(method)
    except ValueError:
        return "", ""
    if i + 2 >= len(parts):
        return "", ""
    return parts[i + 1], parts[i + 2]


def train_seed_from_ablation_filename(name: str) -> str:
    m = re.search(r"_seed(\d+)_", name)
    return m.group(1) if m else ""


def parse_ablation_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def aucpr_map(block: Mapping[str, Any]) -> Dict[str, float]:
    raw = block.get("auc_pr") or {}
    out: Dict[str, float] = {}
    for sp in SPLITS:
        out[sp] = _f(raw.get(sp))
    return out


def one_row(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    method = str(payload.get("editing_method") or "")
    ckpt = str(payload.get("edited_checkpoint") or "")
    ds, model = dataset_model_from_checkpoint(ckpt, method)
    if not ds:
        ds = ""
    if not model:
        model = ""

    variants = payload.get("variants") or {}
    base = variants.get("baseline_original_inputs") or {}
    mask = variants.get("masked_inputs") or {}

    pre_o = base.get("pretrain") or {}
    pre_m = mask.get("pretrain") or {}
    ed_o = base.get("edited") or {}
    ed_m = mask.get("edited") or {}

    task = str(pre_o.get("task") or ed_o.get("task") or "")

    row: Dict[str, Any] = {
        "ablation_json": path.name,
        "ablation_json_path": str(path.resolve()),
        "editing_method": method,
        "dataset": ds,
        "model": model,
        "train_seed": train_seed_from_ablation_filename(path.name),
        "edited_checkpoint": ckpt,
        "focused_feature": payload.get("focused_feature", ""),
        "mask_continuous_heuristic": "",
        "task": task,
    }
    mr = payload.get("mask_rule") or {}
    row["mask_continuous_heuristic"] = mr.get("continuous_heuristic_gt30_unique_values", "")

    ap_pre_o = aucpr_map(pre_o)
    ap_pre_m = aucpr_map(pre_m)
    ap_ed_o = aucpr_map(ed_o)
    ap_ed_m = aucpr_map(ed_m)

    for sp in SPLITS:
        po, pm = ap_pre_o[sp], ap_pre_m[sp]
        eo, em = ap_ed_o[sp], ap_ed_m[sp]
        row[f"pretrain_aucpr_original_{sp}"] = po
        row[f"pretrain_aucpr_masked_{sp}"] = pm
        row[f"pretrain_aucpr_drop_{sp}"] = po - pm if _finite(po) and _finite(pm) else float("nan")
        row[f"edited_aucpr_original_{sp}"] = eo
        row[f"edited_aucpr_masked_{sp}"] = em
        row[f"edited_aucpr_drop_{sp}"] = eo - em if _finite(eo) and _finite(em) else float("nan")
        # ----------------------------------------------------
        # Percentage AUC-PR drops
        # ----------------------------------------------------

        row[f"pretrain_aucpr_pct_drop_{sp}"] = (
            ((po - pm) / po) * 100.0
            if _finite(po) and _finite(pm) and abs(po) > 1e-12
            else float("nan")
        )

        row[f"edited_aucpr_pct_drop_{sp}"] = (
            ((eo - em) / eo) * 100.0
            if _finite(eo) and _finite(em) and abs(eo) > 1e-12
            else float("nan")
        )

    for sp in SPLITS:
        pd = row[f"pretrain_aucpr_drop_{sp}"]
        ed = row[f"edited_aucpr_drop_{sp}"]
        # Negative delta => edited drops *less* than pretrain (better robustness).
        row[f"edited_minus_pretrain_mask_drop_{sp}"] = (
            ed - pd if _finite(pd) and _finite(ed) else float("nan")
        )
        row[f"edited_more_robust_to_mask_{sp}"] = ""
        if _finite(pd) and _finite(ed):
            row[f"edited_more_robust_to_mask_{sp}"] = 1 if ed < pd else 0

    row["has_masked_inputs_branch"] = bool(mask)
    return row


def fieldnames() -> List[str]:
    head = [
        "ablation_json",
        "editing_method",
        "dataset",
        "model",
        "train_seed",
        "edited_checkpoint",
        "focused_feature",
        "mask_continuous_heuristic",
        "task",
        "has_masked_inputs_branch",
    ]
    tail: List[str] = []
    for sp in SPLITS:
        tail.extend(
            [
                f"pretrain_aucpr_original_{sp}",
                f"pretrain_aucpr_masked_{sp}",
                f"pretrain_aucpr_drop_{sp}",
                f"pretrain_aucpr_pct_drop_{sp}",
                f"edited_aucpr_original_{sp}",
                f"edited_aucpr_masked_{sp}",
                f"edited_aucpr_drop_{sp}",
                f"edited_aucpr_pct_drop_{sp}",
                f"edited_minus_pretrain_mask_drop_{sp}",
                f"edited_more_robust_to_mask_{sp}",
            ]
        )
    return head + tail


def _md_scalar(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and not math.isfinite(val):
        return ""
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, float):
        s = f"{val:.10g}"
        if "e" not in s.lower() and "." in s:
            s = s.rstrip("0").rstrip(".")
        return s if s != "" else "0"
    return str(val).replace("\n", " ").replace("|", "\\|")


def _grouped_table_headers(*, include_checkpoint: bool = False) -> List[str]:
    cols = ["Strategy", "Seed"]
    if include_checkpoint:
        cols.append("Checkpoint")
    for sp in SPLITS:
        cols.extend(
            [
                f"pre_drop_{sp}",
                f"pre_pct_{sp}",
                f"ed_drop_{sp}",
                f"ed_pct_{sp}",
                f"ed_minus_pre_{sp}",
                f"robust_{sp}",
            ]
        )
    return cols


def write_markdown_report_grouped(
    path: Path,
    *,
    ablations_dir: Path,
    rows: List[Dict[str, Any]],
    csv_name: str,
) -> None:
    lines: List[str] = []
    lines.append("# Feature masking — AUC-PR summary")
    lines.append("")
    lines.append(f"Source directory: `{ablations_dir}`")
    lines.append("")
    lines.append("- JSON inputs: `feature_ablation_*.json`")
    lines.append(f"- Runs summarized: **{len(rows)}**")
    lines.append(f"- Full column export: `{csv_name}`")
    lines.append("")
    lines.append("## How to read the tables")
    lines.append("")
    lines.append(
        "Sections are grouped by **dataset** and **model** (backbone). Each row is one editing run: "
        "**strategy** (method) and **seed**. If the same strategy and seed appear more than once, "
        "the **Checkpoint** column identifies the edited checkpoint file."
    )
    lines.append("")
    lines.append("| Column | Meaning |")
    lines.append("| --- | --- |")
    lines.append(
        "| `pre_drop_*` | Pretrained model: AUC-PR(original) − AUC-PR(masked). **> 0** ⇒ worse under masking. |"
    )
    lines.append("| `ed_drop_*` | Edited model: same. |")
    lines.append(
        "| `ed_minus_pre_*` | `ed_drop` − `pre_drop`. **< 0** ⇒ edit is *more* robust to masking than pretrain. |"
    )
    lines.append("| `robust_*` | `1` if `ed_drop < pre_drop` (strict). |")
    lines.append("| `Checkpoint` | Edited checkpoint basename (only in tables that need it). |")
    lines.append("")

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    skipped = 0
    for row in rows:
        ds = str(row.get("dataset") or "").strip()
        mo = str(row.get("model") or "").strip()
        if not ds or not mo:
            skipped += 1
            continue
        grouped[(ds, mo)].append(row)
    if skipped:
        print(
            f"[WARN] {skipped} rows missing dataset/model (omitted from grouped markdown)",
            file=sys.stderr,
        )

    keys_sorted = sorted(
        grouped.keys(),
        key=lambda t: (_dataset_sort_key(t[0]), _model_sort_key(t[1])),
    )

    for ds, mo in keys_sorted:
        g_rows = grouped[(ds, mo)]
        by_run: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        warned_dup: Set[Tuple[str, str, str, str, str]] = set()
        for r in g_rows:
            st = str(r.get("editing_method") or "")
            sd = str(r.get("train_seed") or "")
            ck = Path(str(r.get("edited_checkpoint") or "").replace("\\", "/")).name
            key = (st, sd, ck)
            if key in by_run:
                wk = (ds, mo, st, sd, ck)
                if wk not in warned_dup:
                    print(
                        f"[WARN] duplicate run {ds}/{mo} strategy={st} seed={sd} ckpt={ck}; keeping last",
                        file=sys.stderr,
                    )
                    warned_dup.add(wk)
            by_run[key] = r
        g_rows = list(by_run.values())
        g_rows.sort(
            key=lambda r: (
                _method_sort_key(str(r.get("editing_method") or "")),
                int(str(r.get("train_seed") or "0") or 0),
                Path(str(r.get("edited_checkpoint") or "").replace("\\", "/")).name,
            )
        )

        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for r in g_rows:
            pair_counts[
                (
                    str(r.get("editing_method") or ""),
                    str(r.get("train_seed") or ""),
                )
            ] += 1
        show_ckpt = any(c > 1 for c in pair_counts.values())

        lines.append(f"## {ds} — {mo}")
        lines.append("")
        feat0 = str(g_rows[0].get("focused_feature") or "") if g_rows else ""
        if feat0:
            lines.append(f"*Focused feature (masked column):* **{feat0}**")
            lines.append("")

        col_names = _grouped_table_headers(include_checkpoint=show_ckpt)
        lines.append("| " + " | ".join(col_names) + " |")
        lines.append("| " + " | ".join("---" for _ in col_names) + " |")
        for r in g_rows:
            cells: List[str] = [
                _md_scalar(r.get("editing_method")),
                _md_scalar(r.get("train_seed")),
            ]
            if show_ckpt:
                ck_name = Path(
                    str(r.get("edited_checkpoint") or "").replace("\\", "/")
                ).name
                cells.append(_md_scalar(ck_name or ""))
            for sp in SPLITS:
                cells.append(_md_scalar(r.get(f"pretrain_aucpr_drop_{sp}")))
                cells.append(_md_scalar(r.get(f"pretrain_aucpr_pct_drop_{sp}")))

                cells.append(_md_scalar(r.get(f"edited_aucpr_drop_{sp}")))
                cells.append(_md_scalar(r.get(f"edited_aucpr_pct_drop_{sp}")))
                cells.append(_md_scalar(r.get(f"edited_minus_pretrain_mask_drop_{sp}")))
                cells.append(_md_scalar(r.get(f"edited_more_robust_to_mask_{sp}")))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ablations-dir",
        type=Path,
        default=Path(os.environ.get("ABLATIONS_DIR", str(_paths.ablations_dir_default()))),
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Default: <ablations-dir>/feature_masking_aucpr_summary.csv",
    )
    ap.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Default: <ablations-dir>/feature_masking_aucpr_summary.md",
    )
    ap.add_argument(
        "--include-path-column",
        action="store_true",
        help="Also write ablation_json_path column in the CSV only.",
    )
    args = ap.parse_args()

    abl = args.ablations_dir.expanduser().resolve()
    if not abl.is_dir():
        print(f"Not a directory: {abl}", file=sys.stderr)
        return 2

    out_csv = (args.out_csv or (abl / "feature_masking_aucpr_summary.csv")).expanduser().resolve()
    out_md = (args.out_md or (abl / "feature_masking_aucpr_summary.md")).expanduser().resolve()

    paths = sorted(abl.glob("feature_ablation_*.json"))
    rows: List[Dict[str, Any]] = []
    for p in paths:
        try:
            payload = parse_ablation_json(p)
            row = one_row(p, payload)
            rows.append(row)
        except Exception as exc:
            print(f"[WARN] skip {p.name}: {exc}", file=sys.stderr)

    fn = fieldnames()
    if args.include_path_column:
        fn = fn[:1] + ["ablation_json_path"] + fn[1:]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fn, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            if not args.include_path_column:
                row = {k: row[k] for k in fn}
            else:
                row = {k: row.get(k, "") for k in fn}
            w.writerow(row)

    write_markdown_report_grouped(
        out_md,
        ablations_dir=abl,
        rows=rows,
        csv_name=out_csv.name,
    )

    print(f"Wrote {out_csv} ({len(rows)} rows from {len(paths)} JSON files)")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
