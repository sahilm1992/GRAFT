#!/usr/bin/env python3
"""
Invoke feature_ablation_forward.py for one pipeline metrics_*.json.

Resolves the edited .pt via artifacts.best_checkpoint.path when present; otherwise
matches checkpoints/metrics stem (handles metrics_edit.json_* vs edit_* and _gamma*_ slugs).

Environment:
  METRICS_JSON   — required path to metrics_*.json
  ABLATION_PY    — path to feature_ablation_forward.py
  DATASET_DIR, PRETRAIN_DIR, OUTPUT_ROOT, PYTHON — forwarded like the shell driver
  LEASTSQUARES_STRATEGY_ALLOWLIST — optional comma/space-separated list; if non-empty,
      least-squares runs proceed only when the inferred strategy is in this set.
  LEASTSQUARES_STANDARD_HPARAMS — if not "0", least-squares runs must match default suite
      hyperparameters: λ=0.1, top_fraction=0.25, pr_alpha=0.85, rank_mix_tau=0.75,
      and gamma_retain=1.0 when present in JSON/filename (omitted gamma is OK).
  FINETUNE_STANDARD_HPARAMS — if not "0", finetune runs must use lr=0.01 and num_epochs=20
      (from edit_params or metrics filename). Set to "0" to run all finetune variants.
  FEATURE_ABLATION_REQUIRE_TOP_FRACTION — default "0.25". All methods are skipped unless top_fraction
      matches (from edit_params, filename _top0.25_, or parent dir top_0.25). Set to "0" to disable.
  FEATURE_ABLATION_DETERMINISTIC — if "1"/"true"/"yes", pass --deterministic-ablation-filename so
      JSON output paths omit timestamps (re-runs overwrite).
  ABLATIONS_DIR — directory for feature ablation JSON (default matches feature_ablation_forward.py).
  FEATURE_ABLATION_SKIP_EXISTING — if not "0" (default: skip on), skip when any file exists matching
      feature_ablation_{method}_{dataset}_{model}_seed{seed}_{ck_slug}.json or ..._{ck_slug}_*.json.
  FEATURE_ABLATION_FILTER_DATASET — optional comma/space list (e.g. pokec,yelp). Subset of runs.
  FEATURE_ABLATION_FILTER_MODEL — optional (e.g. GCN_MLP,Polynormer).
  FEATURE_ABLATION_FILTER_METHOD — optional (e.g. seed_gnn,finetune). Matched to JSON experiment.method.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Default hyperparameters aligned with editing_pipelines/run_editing_suite.sh
LS_STD_LAMBDA = 0.1
LS_STD_TOP = 0.25
LS_STD_PRA = 0.85
LS_STD_RMX = 0.75
LS_STD_GAMMA = 1.0
FLOAT_TOL = 1e-5

FT_STD_LR = 0.01
FT_STD_EPOCHS = 20

DEFAULT_REQUIRE_TOP_FRACTION = 0.25


def _feq(a: float, b: float, tol: float = FLOAT_TOL) -> bool:
    return abs(float(a) - float(b)) < tol


def _parse_stem_float(stem: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, stem)
    if not m:
        return None
    return float(m.group(1))


def least_squares_standard_hparams_ok(payload: dict, stem: str) -> Tuple[bool, str]:
    """Require suite-default LS hypers (from edit_params or metrics filename)."""
    ep = payload.get("edit_params") or {}

    lam = ep.get("lambda_reg")
    if lam is None:
        lam = _parse_stem_float(stem, r"_lam([0-9.eE+-]+)_")
    if lam is None:
        return False, "lambda_reg missing"
    if not _feq(float(lam), LS_STD_LAMBDA):
        return False, f"lambda_reg {lam} != {LS_STD_LAMBDA}"

    top = ep.get("top_fraction")
    if top is None:
        top = _parse_stem_float(stem, r"_top([0-9.]+)_")
    if top is None:
        return False, "top_fraction missing"
    if not _feq(float(top), LS_STD_TOP):
        return False, f"top_fraction {top} != {LS_STD_TOP}"

    pra = ep.get("pr_alpha")
    if pra is None:
        pra = _parse_stem_float(stem, r"_pra([0-9.]+)_")
    if pra is None:
        return False, "pr_alpha missing"
    if not _feq(float(pra), LS_STD_PRA):
        return False, f"pr_alpha {pra} != {LS_STD_PRA}"

    rmx = ep.get("rank_mix_tau")
    if rmx is None:
        rmx = _parse_stem_float(stem, r"_rmx([0-9.]+)_")
    if rmx is None:
        return False, "rank_mix_tau missing"
    if not _feq(float(rmx), LS_STD_RMX):
        return False, f"rank_mix_tau {rmx} != {LS_STD_RMX}"

    g_ep = ep.get("gamma_retain")
    g_stem = parse_gamma_from_stem(stem)
    if g_ep is not None and not _feq(float(g_ep), LS_STD_GAMMA):
        return False, f"gamma_retain {g_ep} != {LS_STD_GAMMA}"
    if g_stem is not None and not _feq(float(g_stem), LS_STD_GAMMA):
        return False, f"filename gamma {g_stem} != {LS_STD_GAMMA}"

    return True, ""


def finetune_lr_epochs_ok(payload: dict, stem: str) -> Tuple[bool, str]:
    """Require lr=0.01 and num_epochs=20 (edit_params or metrics_finetune_lr*_ep* filename)."""
    ep = payload.get("edit_params") or {}

    lr = ep.get("lr")
    if lr is None:
        lr = _parse_stem_float(stem, r"_lr([0-9.eE+-]+)_")
    if lr is None:
        return False, "finetune lr missing"
    if not _feq(float(lr), FT_STD_LR):
        return False, f"finetune lr {lr} != {FT_STD_LR}"

    epochs = ep.get("num_epochs")
    if epochs is None:
        m = re.search(r"_ep([0-9]+)_", stem)
        epochs = int(m.group(1)) if m else None
    if epochs is None:
        return False, "finetune num_epochs missing"
    if int(epochs) != int(FT_STD_EPOCHS):
        return False, f"finetuneEpochs {epochs} != {FT_STD_EPOCHS}"

    return True, ""


def _find_edited_ckpt(metrics_path: Path) -> Path:
    ckpt_dir = metrics_path.parent / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"No checkpoints directory: {ckpt_dir}")

    stem = metrics_path.stem
    if not stem.startswith("metrics_"):
        raise ValueError(f"Expected metrics_*.json, got {metrics_path.name}")
    slug = stem[len("metrics_") :]

    candidates: List[Path] = []
    seen: set[str] = set()

    def add(p: Path) -> None:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            candidates.append(p)

    add(ckpt_dir / f"{slug}.pt")
    if slug.startswith("edit.json_"):
        alt_name = (f"edit_{slug[len('edit.json_'):]}").strip() + ".pt"
        add(ckpt_dir / alt_name)

    seed_m = re.search(r"_seed(\d+)$", slug)
    if seed_m:
        seed = seed_m.group(1)
        for p in sorted(ckpt_dir.glob(f"*seed{seed}.pt")):
            add(p)

    for p in candidates:
        if p.is_file():
            return p

    def norm(s: str) -> str:
        s = re.sub(r"_gamma[0-9.]+_", "_", s)
        return s.replace("edit.json_", "edit_")

    target = norm(slug)
    for p in ckpt_dir.glob("*.pt"):
        if norm(p.stem) == target:
            return p

    best: Optional[Tuple[int, Path]] = None
    for p in ckpt_dir.glob("*.pt"):
        n = norm(p.stem)
        if target in n or n in target:
            score = abs(len(n) - len(target))
            if best is None or score < best[0]:
                best = (score, p)
    if best is not None:
        return best[1]

    raise FileNotFoundError(
        f"No matching .pt under {ckpt_dir} for metrics slug {slug!r}."
    )


def resolve_ckpt(metrics_path: Path, payload: dict) -> Path:
    art = (payload.get("artifacts") or {}).get("best_checkpoint") or {}
    rel = art.get("path")
    if rel:
        p = Path(rel).resolve()
        if p.is_file():
            return p
    return _find_edited_ckpt(metrics_path)


def parse_ls_strategy_from_stem(stem: str) -> Optional[str]:
    m = re.search(r"metrics_edit(?:\.json)?_(.+?)_lam", stem)
    return m.group(1) if m else None


def parse_gamma_from_stem(stem: str) -> Optional[float]:
    m = re.search(r"_gamma([0-9.]+)_", stem)
    return float(m.group(1)) if m else None


def _parse_top_dir_segment(part: str) -> Optional[float]:
    if not part.startswith("top_"):
        return None
    try:
        return float(part[4:])
    except ValueError:
        return None


def infer_top_fraction(payload: dict, stem: str, metrics_path: Path) -> Optional[float]:
    """Resolve top_fraction from edit_params, metrics stem, or …/top_<frac>/… directory."""
    ep = payload.get("edit_params") or {}
    t = ep.get("top_fraction")
    if t is not None:
        return float(t)
    from_stem = _parse_stem_float(stem, r"_top([0-9.]+)_")
    if from_stem is not None:
        return float(from_stem)
    for part in metrics_path.parts:
        td = _parse_top_dir_segment(part)
        if td is not None:
            return td
    return None


def top_fraction_requirement_ok(payload: dict, stem: str, metrics_path: Path) -> Tuple[bool, str]:
    raw = os.environ.get("FEATURE_ABLATION_REQUIRE_TOP_FRACTION", str(DEFAULT_REQUIRE_TOP_FRACTION)).strip()
    if not raw or raw.lower() in ("0", "false", "off", "no", "none"):
        return True, ""
    try:
        need = float(raw)
    except ValueError:
        return False, f"invalid FEATURE_ABLATION_REQUIRE_TOP_FRACTION {raw!r}"
    got = infer_top_fraction(payload, stem, metrics_path)
    if got is None:
        return False, "top_fraction not in edit_params, filename (_top*_), or top_* parent directory"
    if not _feq(got, need):
        return False, f"top_fraction {got} != required {need}"
    return True, ""


def ls_metrics_matches_allowlist_filename(metrics_name: str, allow_raw: str) -> bool:
    """Mirror run_feature_ablation_all_checkpoints.sh ls_metrics_matches_allowlist (basename only)."""
    if not allow_raw.strip():
        return True
    base = metrics_name
    bn = base.removeprefix("metrics_")
    allowed = {x.strip() for x in re.split(r"[\s,]+", allow_raw) if x.strip()}
    return any(tok in base or tok in bn for tok in allowed)


def ls_metrics_standard_hparams_basename_ok(base: str) -> bool:
    """Mirror run_feature_ablation_all_checkpoints.sh ls_metrics_standard_hparams_basename."""
    if "_lam0.1_" not in base:
        return False
    if "_top0.25_" not in base:
        return False
    if "_pra0.85_" not in base:
        return False
    if "_rmx0.75_" not in base:
        return False
    if "_gamma" in base:
        if "_gamma1.0_" not in base and "_gamma1_" not in base:
            return False
    return True


def finetune_metrics_basename_lr_ep_ok(bn: str) -> bool:
    """Mirror finetune_lr_ep_basename in run_feature_ablation_all_checkpoints.sh."""
    return "_lr0.01_ep20_" in bn


def env_csv_allowlist(var: str) -> Optional[set[str]]:
    """None = no filter (allow all)."""
    raw = os.environ.get(var, "").strip()
    if not raw:
        return None
    return {x.strip() for x in re.split(r"[\s,]+", raw) if x.strip()}


def dmf_filter_ok(dataset: str, model: str, method: str) -> Tuple[bool, str]:
    allow_d = env_csv_allowlist("FEATURE_ABLATION_FILTER_DATASET")
    if allow_d is not None and dataset not in allow_d:
        return False, f"dataset {dataset!r} not in FEATURE_ABLATION_FILTER_DATASET"
    allow_mo = env_csv_allowlist("FEATURE_ABLATION_FILTER_MODEL")
    if allow_mo is not None and model not in allow_mo:
        return False, f"model {model!r} not in FEATURE_ABLATION_FILTER_MODEL"
    allow_me = env_csv_allowlist("FEATURE_ABLATION_FILTER_METHOD")
    if allow_me is not None and method not in allow_me:
        return False, f"method {method!r} not in FEATURE_ABLATION_FILTER_METHOD"
    return True, ""


DEFAULT_ABLATIONS_DIR = "/home/model_editing/data/ablations"


def _ablation_basename_prefix(method: str, dataset: str, model: str, seed: int, ckpt_path: Path) -> str:
    ck_slug = ckpt_path.stem[:120].replace(" ", "_")
    return f"feature_ablation_{method}_{dataset}_{model}_seed{seed}_{ck_slug}"


def existing_feature_ablation_outputs(ablations_dir: Path, prefix: str) -> List[Path]:
    """Match deterministic name or timestamped variants (same rule as feature_ablation_forward)."""
    if not ablations_dir.is_dir():
        return []
    found: List[Path] = []
    exact = ablations_dir / f"{prefix}.json"
    if exact.is_file():
        found.append(exact)
    found.extend(sorted(ablations_dir.glob(f"{prefix}_*.json")))
    return found


def main() -> int:
    metrics_path = Path(os.environ["METRICS_JSON"]).resolve()
    ablation_py = Path(os.environ["ABLATION_PY"]).resolve()
    dataset_dir = os.environ["DATASET_DIR"]
    pretrain_dir = os.environ["PRETRAIN_DIR"]
    output_root = os.environ["OUTPUT_ROOT"]
    py_exe = os.environ.get("PYTHON", "python3")

    with open(metrics_path, encoding="utf-8") as f:
        m = json.load(f)

    exp = m.get("experiment") or {}
    data = m.get("data") or {}
    model_o = m.get("model") or {}
    method = (exp.get("method") or "").strip()
    dataset = (data.get("dataset") or "").strip()
    model = (model_o.get("name") or "").strip()
    seed = int(exp.get("seed", 0))
    arch = model_o.get("architecture") or {}
    num_layers = int(arch.get("num_layers", 2))

    if method not in ("leastsquares", "finetune", "egnn", "seed_gnn"):
        print(f"skip unknown method {method!r} for {metrics_path}", file=sys.stderr)
        return 0

    ok_filt, reason_filt = dmf_filter_ok(dataset, model, method)
    if not ok_filt:
        print(f"skip dmf filter ({reason_filt})", file=sys.stderr)
        return 0

    ok_top, reason_top = top_fraction_requirement_ok(m, metrics_path.stem, metrics_path)
    if not ok_top:
        print(f"skip top_fraction ({reason_top})", file=sys.stderr)
        return 0

    ep = m.get("edit_params") or {}
    sel = m.get("selection_correlation_metrics") or {}

    if method == "leastsquares":
        strategy = sel.get("strategy_mode") or parse_ls_strategy_from_stem(metrics_path.stem)
        if not strategy:
            print(f"Cannot infer least-squares strategy for {metrics_path}", file=sys.stderr)
            return 2
        allow_raw = os.environ.get("LEASTSQUARES_STRATEGY_ALLOWLIST", "").strip()
        if allow_raw:
            allowed = {x.strip() for x in re.split(r"[\s,]+", allow_raw) if x.strip()}
            if strategy not in allowed:
                print(
                    f"skip leastsquares strategy {strategy!r} (not in LEASTSQUARES_STRATEGY_ALLOWLIST)",
                    file=sys.stderr,
                )
                return 0
        if os.environ.get("LEASTSQUARES_STANDARD_HPARAMS", "1") != "0":
            ok_hp, reason = least_squares_standard_hparams_ok(m, metrics_path.stem)
            if not ok_hp:
                print(f"skip leastsquares hyperparams ({reason})", file=sys.stderr)
                return 0

    if method == "finetune":
        if os.environ.get("FINETUNE_STANDARD_HPARAMS", "1") != "0":
            ok_ft, reason = finetune_lr_epochs_ok(m, metrics_path.stem)
            if not ok_ft:
                print(f"skip finetune hyperparams ({reason})", file=sys.stderr)
                return 0

    ckpt = resolve_ckpt(metrics_path, m)
    out_dir = ckpt.parent.parent

    ablations_dir = Path(
        os.environ.get("ABLATIONS_DIR", DEFAULT_ABLATIONS_DIR)
    ).expanduser().resolve()
    prefix = _ablation_basename_prefix(method, dataset, model, seed, ckpt)
    if os.environ.get("FEATURE_ABLATION_SKIP_EXISTING", "1") != "0":
        existing = existing_feature_ablation_outputs(ablations_dir, prefix)
        if existing:
            names = ", ".join(p.name for p in existing)
            print(f"skip existing ablation output(s): {names}", file=sys.stderr, flush=True)
            return 0

    cmd: List[str] = [
        py_exe,
        str(ablation_py),
        "--method",
        method,
        "--dataset",
        dataset,
        "--model",
        model,
        "--seed",
        str(seed),
        "--num-layers",
        str(num_layers),
        "--pretrain-dir",
        pretrain_dir,
        "--dataset-dir",
        dataset_dir,
        "--output-root",
        output_root,
        "--output-dir",
        str(out_dir),
        "--edited-checkpoint",
        str(ckpt),
        "--ablations-dir",
        str(ablations_dir),
    ]

    if os.environ.get("FEATURE_ABLATION_DETERMINISTIC", "").strip().lower() in ("1", "true", "yes"):
        cmd.append("--deterministic-ablation-filename")

    if method == "leastsquares":
        strategy = sel.get("strategy_mode") or parse_ls_strategy_from_stem(metrics_path.stem)
        if not strategy:
            print(f"Cannot infer least-squares strategy for {metrics_path}", file=sys.stderr)
            return 2
        cmd += ["--leastsquares-strategy", strategy]
        if ep.get("lambda_reg") is not None:
            cmd += ["--lambda-reg", str(ep["lambda_reg"])]
        if ep.get("top_fraction") is not None:
            cmd += ["--top-fraction", str(float(ep["top_fraction"]))]
        if ep.get("pr_alpha") is not None:
            cmd += ["--pr-alpha", str(ep["pr_alpha"])]
        if ep.get("rank_mix_tau") is not None:
            cmd += ["--rank-mix-tau", str(ep["rank_mix_tau"])]
        gamma = ep.get("gamma_retain")
        if gamma is None:
            g = parse_gamma_from_stem(metrics_path.stem)
            gamma = g
        if gamma is not None:
            cmd += ["--gamma-retain", str(gamma)]
    elif method == "finetune":
        if ep.get("top_fraction") is not None:
            cmd += ["--top-fraction", str(float(ep["top_fraction"]))]
        if ep.get("lr") is not None:
            cmd += ["--ft-lr", str(ep["lr"])]
        if ep.get("num_epochs") is not None:
            cmd += ["--ft-epochs", str(int(ep["num_epochs"]))]
    elif method in ("egnn", "seed_gnn"):
        tf = infer_top_fraction(m, metrics_path.stem, metrics_path)
        if tf is not None:
            cmd += ["--top-fraction", str(float(tf))]
        n = ep.get("num_targets")
        if n is None:
            mm = re.search(r"targets(\d+)", metrics_path.stem)
            n = int(mm.group(1)) if mm else None
        if n is not None:
            cmd += ["--edited-targets", str(int(n))]

    print("RUN:", " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


if __name__ == "__main__":
    raise SystemExit(main())
