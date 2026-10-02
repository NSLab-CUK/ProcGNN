#!/usr/bin/env python3
"""Optuna end-to-end smoke: 1 trial, 2 epochs, tiny subset + skip startup audits.

Exit 0 if study completes and summary artifacts exist.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test scripts/tune_optuna.py")
    parser.add_argument(
        "--output-root",
        default="outputs/optuna_smoke_run",
        help="Fresh directory under project root (default: outputs/optuna_smoke_run)",
    )
    parser.add_argument("--study-name", default="optuna_smoke_edge_p01")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_root = (root / args.output_root / "Process1").resolve()

    cmd = [
        sys.executable,
        str(root / "scripts" / "tune_optuna.py"),
        "--config",
        "configs/experiment/optuna_smoke_edge_all.yaml",
        "--process-filter",
        "1",
        "--preset",
        "edge_all",
        "--n-trials",
        "1",
        "--max-epochs",
        "2",
        "--output-root",
        args.output_root,
        "--study-name",
        args.study_name,
        "--no-enqueue-best",
        "--skip-startup-debug",
        "--no-analysis-plots",
    ]
    print("[smoke-optuna] running:\n  " + " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(root))
    if rc != 0:
        print(f"[smoke-optuna] tune_optuna failed rc={rc}", file=sys.stderr)
        return rc

    summary = out_root / "optuna_trials_summary.csv"
    if not summary.is_file():
        print(f"[smoke-optuna] missing {summary}", file=sys.stderr)
        return 2

    trials = sorted(out_root.glob("trial_*"))
    if not trials:
        print(f"[smoke-optuna] no trial_* under {out_root}", file=sys.stderr)
        return 3
    last = trials[-1]
    metrics_json = last / "optuna_trial_metrics.json"
    hist_json = last / "train_val_history.json"
    for p in (metrics_json, hist_json):
        if not p.is_file():
            print(f"[smoke-optuna] missing {p}", file=sys.stderr)
            return 4

    payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    obj = payload.get("objective", {})
    val = float(obj.get("value", float("nan")))
    if not (val == val):  # NaN check
        print("[smoke-optuna] objective value is NaN", file=sys.stderr)
        return 5

    print(f"[smoke-optuna] ok objective={val:.6f}")
    print(f"  summary: {summary}")
    print(f"  trial:   {last}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
