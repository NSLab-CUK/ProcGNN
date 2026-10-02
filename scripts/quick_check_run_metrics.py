#!/usr/bin/env python3
"""Print key edge_all metrics from a training run (legacy / Frac primary / amount secondary).

Usage:
  python scripts/quick_check_run_metrics.py --run-dir outputs/.../process_kfold_P07_F01-...
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from process_graph.experiment.target_metric_names import (  # noqa: E402
    AMOUNT_R2_SHORT_ALIAS,
    EVAL_LEGACY_ANSWER_FRAC_R2,
    EVAL_PRIMARY_FRAC_R2_BY_PROCESS,
    EVAL_SECONDARY_AMOUNT_R2_BY_PROCESS,
    FRAC_R2_SHORT_ALIAS,
    METRIC_SCHEMA,
    pick_from_mapping,
    resolve_metric_key,
)
from process_graph.experiment.target_row_primary_metrics import (  # noqa: E402
    AMOUNT_SECONDARY_METRIC_NAME,
    PRIMARY_METRIC_NAME,
)

RUN_RE = re.compile(r"^process_kfold_P\d+_F\d+(?:-\d{8}-\d{6})?$")

OPTUNA_R2_KEYS = (
    "test/metric_target_r2",
    "metric_target_r2",
    "test/metric_tailgas_r2",
    "metric_tailgas_r2",
)
LEGACY_KEYS = (
    EVAL_LEGACY_ANSWER_FRAC_R2,
    f"test/{EVAL_LEGACY_ANSWER_FRAC_R2}",
    "test/legacy_answer_fraction_macro_r2",
    "legacy_answer_fraction_macro_r2",
)
FRAC_KEYS = (
    EVAL_PRIMARY_FRAC_R2_BY_PROCESS,
    f"test/{EVAL_PRIMARY_FRAC_R2_BY_PROCESS}",
    FRAC_R2_SHORT_ALIAS,
    f"test/{FRAC_R2_SHORT_ALIAS}",
)
AMOUNT_KEYS = (
    EVAL_SECONDARY_AMOUNT_R2_BY_PROCESS,
    f"test/{EVAL_SECONDARY_AMOUNT_R2_BY_PROCESS}",
    AMOUNT_SECONDARY_METRIC_NAME,
    AMOUNT_R2_SHORT_ALIAS,
)
VAL_BEST_BASES = (
    EVAL_PRIMARY_FRAC_R2_BY_PROCESS,
    EVAL_SECONDARY_AMOUNT_R2_BY_PROCESS,
    EVAL_LEGACY_ANSWER_FRAC_R2,
    FRAC_R2_SHORT_ALIAS,
    AMOUNT_R2_SHORT_ALIAS,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _weighted_answer_r2(h2: float, co2: float, wh: float = 5.0, wc: float = 3.0) -> float:
    vals: list[tuple[float, float]] = []
    if math.isfinite(h2):
        vals.append((wh, h2))
    if math.isfinite(co2):
        vals.append((wc, co2))
    if not vals:
        return float("nan")
    wsum = sum(w for w, _ in vals)
    return float(sum(w * r for w, r in vals) / wsum)


def _find_latest_run(root: Path) -> Path | None:
    candidates: list[Path] = []
    if RUN_RE.match(root.name) or (root / "metrics.json").is_file():
        return root if root.is_dir() else None
    if not root.is_dir():
        return None
    for p in root.rglob("*"):
        if p.is_dir() and (RUN_RE.match(p.name) or (p / "metrics.json").is_file()):
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.stat().st_mtime)


def _load_epoch_best(run_dir: Path) -> dict[str, float]:
    csv_path = run_dir / "metrics_per_epoch.csv"
    if not csv_path.is_file():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except (OSError, pd.errors.EmptyDataError):
        return {}
    if df.empty:
        return {}
    out: dict[str, float] = {}
    for base in VAL_BEST_BASES:
        col = f"val_{base}"
        if col not in df.columns:
            col = f"val_{resolve_metric_key(base)}"
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if not s.empty:
            out[f"best_{col}"] = float(s.max())
    return out


def _read_per_target_table(run_dir: Path, name: str) -> pd.DataFrame:
    for rel in (f"test/{name}", name):
        p = run_dir / rel
        if p.is_file():
            try:
                return pd.read_csv(p)
            except (OSError, pd.errors.EmptyDataError):
                return pd.DataFrame()
    return pd.DataFrame()


def _print_per_target_side_by_side(run_dir: Path) -> None:
    amt = _read_per_target_table(run_dir, "target_metrics_v4.csv")
    frac = _read_per_target_table(run_dir, "target_metrics_v4_frac.csv")
    if amt.empty and frac.empty:
        return
    print("\nPer-target_id (amount R2 vs Frac-primary R2):")
    ids: set[str] = set()
    if not amt.empty and "target_id" in amt.columns:
        ids |= set(amt["target_id"].astype(str))
    if not frac.empty and "target_id" in frac.columns:
        ids |= set(frac["target_id"].astype(str))
    for tid in sorted(ids):
        if tid.startswith("__ALL"):
            continue
        a_row = amt[amt["target_id"].astype(str) == tid].iloc[0] if not amt.empty else None
        f_row = frac[frac["target_id"].astype(str) == tid].iloc[0] if not frac.empty else None
        pid = species = feat = ""
        if f_row is not None:
            pid = str(f_row.get("process_id", ""))
            species = str(f_row.get("target_species", ""))
            feat = str(f_row.get("target_feature", f_row.get("target_feature_name", "")))
        elif a_row is not None:
            pid = str(a_row.get("process_id", ""))
            species = str(a_row.get("target_species", ""))
            feat = str(a_row.get("target_feature_name", ""))
        ar2 = float(pd.to_numeric(a_row.get("r2"), errors="coerce")) if a_row is not None else float("nan")
        fr2 = float(pd.to_numeric(f_row.get("r2"), errors="coerce")) if f_row is not None else float("nan")
        inc = str(f_row.get("included_in_primary", "")) if f_row is not None else ""
        skip = str(f_row.get("skip_reason", "") or "") if f_row is not None else ""
        print(
            f"  {pid} {tid} {species} feature={feat}  "
            f"amount_r2={ar2:.6f}  frac_primary_r2={fr2:.6f}  "
            f"included_in_primary={inc}  skip_reason={skip or '-'}"
        )
        if tid == "P06_T004" or "Stream17" in feat:
            print(f"    [Process6 Stream17] included_in_primary={inc} skip_reason={skip or '-'}")


def _report_run(run_dir: Path) -> None:
    run_dir = run_dir.resolve()
    print(f"\n=== run: {run_dir.relative_to(PROJECT_ROOT)} ===\n")

    test_m = _read_json(run_dir / "test" / "metrics.json")
    root_m = _read_json(run_dir / "metrics.json")
    merged = {**root_m, **test_m}

    h2 = pick_from_mapping(merged, *OPTUNA_R2_KEYS[:2])
    co2 = pick_from_mapping(merged, *OPTUNA_R2_KEYS[2:])
    legacy = pick_from_mapping(merged, *LEGACY_KEYS)
    frac_primary = pick_from_mapping(merged, *FRAC_KEYS)
    amount_sec = pick_from_mapping(merged, *AMOUNT_KEYS)
    w_r2 = _weighted_answer_r2(h2, co2)

    print("Test - legacy (Optuna ~0.9076, answer-edge Frac slots):")
    print(f"  metric_target_r2 (H2 Frac):      {h2:>10.6f}")
    print(f"  metric_tailgas_r2 (CO2 Frac):    {co2:>10.6f}")
    print(f"  answer_targets_r2 (w=5/3):     {w_r2:>10.6f}")
    print(f"  {EVAL_LEGACY_ANSWER_FRAC_R2}:   {legacy:>10.6f}")
    print()
    print("Test - Frac primary (per target_id, Frac_* only):")
    print(f"  {EVAL_PRIMARY_FRAC_R2_BY_PROCESS}: {frac_primary:>10.6f}")
    print(f"  ({FRAC_R2_SHORT_ALIAS})")
    print()
    print("Test - amount secondary (scale * Mole_Flow * Frac):")
    print(f"  {EVAL_SECONDARY_AMOUNT_R2_BY_PROCESS}: {amount_sec:>10.6f}")
    print(f"  ({AMOUNT_R2_SHORT_ALIAS})")

    pname = merged.get("primary_metric_name", merged.get("test/primary_metric_name", ""))
    schema = merged.get("metric_schema_version", merged.get("targetrow_metric_schema_version", ""))
    print(f"\nBundle: primary_metric_name={pname!r} schema={schema!r} (expected {METRIC_SCHEMA!r})")
    print(f"  included_target_ids: {merged.get('included_target_ids', '')}")
    print(f"  excluded_target_ids: {merged.get('excluded_target_ids', '')}")

    epoch_best = _load_epoch_best(run_dir)
    if epoch_best:
        print("\nBest validation (metrics_per_epoch.csv):")
        for k, v in sorted(epoch_best.items()):
            print(f"  {k}: {v:.6f}")

    _print_per_target_side_by_side(run_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick metric sanity check for one edge_all run.")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--root", type=str, default=None)
    args = parser.parse_args()

    if args.run_dir:
        run = (PROJECT_ROOT / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    elif args.root:
        root = (PROJECT_ROOT / args.root).resolve() if not Path(args.root).is_absolute() else Path(args.root)
        run = _find_latest_run(root)
    else:
        print("[error] pass --run-dir or --root", flush=True)
        return 1

    if run is None or not run.is_dir():
        print("[error] run directory not found", flush=True)
        return 1

    _report_run(run)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
