#!/usr/bin/env python3
"""Build a compact Process x target-edge-10D metric summary table."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _process_sort_key(value: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", str(value))
    return (int(match.group(1)) if match else 10_000, str(value))


def _infer_process(run_dir: Path) -> str:
    for part in reversed(run_dir.parts):
        if str(part).lower() == "all":
            return "All"
        match = re.fullmatch(r"Process(\d+)", part, flags=re.IGNORECASE)
        if match:
            return f"P{int(match.group(1))}"
        match = re.search(r"(?<![A-Za-z0-9])P(\d{1,2})(?!\d)", part, flags=re.IGNORECASE)
        if match:
            return f"P{int(match.group(1))}"
        match = re.search(r"(process_kfold_All|joint_all_processes)", part, flags=re.IGNORECASE)
        if match:
            return "All"
    return run_dir.name


def _best_epoch(run_dir: Path, monitor_metric: str) -> int | None:
    path = run_dir / "metrics_per_epoch.csv"
    if not path.is_file():
        return None
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    if frame.empty or "epoch" not in frame.columns:
        return None
    metric = monitor_metric
    if metric not in frame.columns:
        aliases = {
            "val_target_mean_r2": ["target_mean_r2", "val/target_mean_r2"],
            "val_target_r2": ["target_r2", "val/target_r2"],
            "val_target_edge_10d_r2_flatten": [
                "target_edge_10d_r2_flatten",
                "val/target_edge_10d_r2_flatten",
            ],
            "target_edge_10d_r2_flatten": [
                "val_target_edge_10d_r2_flatten",
                "val/target_edge_10d_r2_flatten",
            ],
            "flatten_r2": [
                "val_target_edge_10d_r2_flatten",
                "target_edge_10d_r2_flatten",
                "val/target_edge_10d_r2_flatten",
            ],
        }
        metric = next((cand for cand in aliases.get(monitor_metric, []) if cand in frame.columns), "")
    if metric and metric in frame.columns:
        values = pd.to_numeric(frame[metric], errors="coerce")
        if values.notna().any():
            idx = int(values.idxmax())
            return int(float(frame.loc[idx, "epoch"]))
    epochs = pd.to_numeric(frame["epoch"], errors="coerce").dropna()
    return int(float(epochs.iloc[-1])) if not epochs.empty else None


def _f(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _summary_row(summary_path: Path, monitor_metric: str) -> dict[str, Any]:
    run_dir = summary_path.parent
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    split = payload.get("splits", {}).get("val", {})
    return {
        "Process": _infer_process(run_dir),
        "Epoch": _best_epoch(run_dir, monitor_metric),
        "Target edges": int(_f(split.get("target_edge_10d_num_edges"))) if math.isfinite(_f(split.get("target_edge_10d_num_edges"))) else None,
        "MAE macro": _f(split.get("target_edge_10d_mae_property_macro")),
        "RMSE macro": _f(split.get("target_edge_10d_rmse_property_macro")),
        "R2 flatten": _f(split.get("target_edge_10d_r2_flatten")),
        "R2 edge macro": _f(split.get("target_edge_10d_r2_edge_macro")),
        "R2 unstable": int(_f(split.get("target_edge_10d_num_r2_unstable"))) if math.isfinite(_f(split.get("target_edge_10d_num_r2_unstable"))) else None,
        "run_dir": str(run_dir),
    }


def _latest_run_summaries(root: Path) -> list[Path]:
    summaries = list(root.rglob("target_edge_10d_summary.json"))
    latest_by_process: dict[str, Path] = {}
    for path in summaries:
        process = _infer_process(path.parent)
        old = latest_by_process.get(process)
        if old is None or path.stat().st_mtime > old.stat().st_mtime:
            latest_by_process[process] = path
    return [latest_by_process[k] for k in sorted(latest_by_process, key=_process_sort_key)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, help="Run root to scan, e.g. outputs/kfold_target_mean_r2_per_process")
    parser.add_argument("--out-csv", default="", help="Output CSV path. Defaults under output-root.")
    parser.add_argument("--monitor-metric", default="val_target_mean_r2")
    parser.add_argument("--include-run-dir", action="store_true")
    args = parser.parse_args()

    root = (PROJECT_ROOT / args.output_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"output root not found: {root}")
    rows = [_summary_row(path, args.monitor_metric) for path in _latest_run_summaries(root)]
    columns = [
        "Process",
        "Epoch",
        "Target edges",
        "MAE macro",
        "RMSE macro",
        "R2 flatten",
        "R2 edge macro",
        "R2 unstable",
    ]
    if args.include_run_dir:
        columns.append("run_dir")
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("Process", key=lambda s: s.map(_process_sort_key)).reset_index(drop=True)
    out_csv = Path(args.out_csv) if args.out_csv else root / "target_edge_10d_process_summary.csv"
    if not out_csv.is_absolute():
        out_csv = PROJECT_ROOT / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False, columns=[c for c in columns if c in frame.columns])
    print(frame[[c for c in columns if c in frame.columns]].to_string(index=False))
    print(f"[summary] wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
