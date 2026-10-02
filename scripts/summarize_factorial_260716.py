#!/usr/bin/env python3
"""Collect completed factorial_260716 metrics into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _latest_run_dir(root: Path) -> Path | None:
    if (root / "metrics_per_epoch.csv").is_file():
        return root
    if not root.is_dir():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir() and (p / "metrics_per_epoch.csv").is_file()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _pick(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in row:
            value = _num(row[key])
            if math.isfinite(value):
                return value
    return float("nan")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _best_epoch_row(metrics_path: Path, monitor: str = "val_target_mean_r2") -> dict[str, Any]:
    if not metrics_path.is_file():
        return {}
    try:
        df = pd.read_csv(metrics_path)
    except (OSError, pd.errors.EmptyDataError):
        return {}
    if df.empty:
        return {}
    if monitor in df.columns:
        metric = pd.to_numeric(df[monitor], errors="coerce")
        if metric.notna().any():
            return df.iloc[int(metric.idxmax())].to_dict()
    return df.iloc[-1].to_dict()


def _prop_r2(run_dir: Path, property_name: str) -> float:
    for rel in ("metrics_by_property.csv", "test/metrics_by_property.csv"):
        path = run_dir / rel
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError):
            continue
        prop_col = next((c for c in ("property_name", "property", "feature") if c in df.columns), None)
        r2_col = next((c for c in ("r2", "R2", "val_r2") if c in df.columns), None)
        if prop_col is None or r2_col is None:
            continue
        sub = df[df[prop_col].astype(str) == property_name]
        if not sub.empty:
            return _num(sub.iloc[0][r2_col])
    return float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/experiment/factorial_260716/manifest.csv")
    parser.add_argument("--output", default="outputs/factorial_260716/factorial_summary.csv")
    args = parser.parse_args()

    manifest_path = (PROJECT_ROOT / args.manifest).resolve()
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    out_rows: list[dict[str, Any]] = []
    for m in rows:
        output_root = (PROJECT_ROOT / m["output_root"]).resolve()
        run_dir = _latest_run_dir(output_root)
        epoch_row = _best_epoch_row(run_dir / "metrics_per_epoch.csv") if run_dir else {}
        info = _read_json(run_dir / "model_info.json") if run_dir else {}
        run_ctx = _read_json(run_dir / "run_context.json") if run_dir else {}
        out_rows.append(
            {
                "experiment_id": m["experiment_id"],
                "experiment_name": m["experiment_name"],
                "adapter_enabled": m["adapter_enabled"],
                "pinn_schedule_enabled": m["pinn_schedule_enabled"],
                "target_edge_weight": m["target_edge_weight"],
                "encoder_hidden_dim": m["encoder_hidden_dim"],
                "pi_head_input_dim": m["expected_pi_head_input_dim"],
                "run_dir": str(run_dir.relative_to(PROJECT_ROOT)).replace("\\", "/") if run_dir else "",
                "status": "completed_or_running" if run_dir else "missing",
                "best_epoch": epoch_row.get("epoch", ""),
                "target_mean_r2": _pick(epoch_row, "val_target_mean_r2", "target_mean_r2", "val/target_mean_r2"),
                "all_edge_mean_r2": _pick(epoch_row, "val_pi_all_edge_mean_r2", "pi_all_edge_mean_r2", "all_edge_property_mean_r2"),
                "non_target_mean_r2": _pick(epoch_row, "val_non_target_mean_r2", "non_target_mean_r2"),
                "edge_macro_r2": _pick(epoch_row, "val_target_edge_10d_r2_edge_macro", "target_edge_10d_r2_edge_macro"),
                "flatten_r2": _pick(epoch_row, "val_target_edge_10d_r2_flatten", "target_edge_10d_r2_flatten"),
                "density_r2": _prop_r2(run_dir, "Density") if run_dir else float("nan"),
                "enthalpy_r2": _prop_r2(run_dir, "Enthalpy") if run_dir else float("nan"),
                "parameter_count": info.get("total_parameters", run_ctx.get("total_parameters", "")),
                "peak_gpu_memory": epoch_row.get("peak_gpu_memory", epoch_row.get("gpu_peak_memory", "")),
                "epoch_time": epoch_row.get("epoch_time", epoch_row.get("epoch_seconds", "")),
            }
        )

    out_path = (PROJECT_ROOT / args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows: {out_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
