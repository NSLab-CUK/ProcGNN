#!/usr/bin/env python3
"""Re-evaluate completed k-fold runs to refresh target-row primary metrics (no retrain)."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from ablation_targetrow_common import (  # noqa: E402
    PRIMARY_METRIC_NAME,
    discover_kfold_runs,
    extract_primary_metrics,
    preflight_recompute,
    validate_run_targetrow_complete,
)


def _find_base_config(run_dir: Path) -> Path | None:
    import json

    fold_dir = run_dir.parent
    meta = fold_dir / "runtime_overrides.json"
    if meta.is_file():
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            exp_root = Path(data.get("output_dir", ""))
            if not exp_root.is_absolute():
                exp_root = (PROJECT_ROOT / exp_root).resolve()
            proc_meta = exp_root.parent.parent / "run_meta.json"
            if proc_meta.is_file():
                rm = json.loads(proc_meta.read_text(encoding="utf-8"))
                bc = rm.get("base_config")
                if bc:
                    p = PROJECT_ROOT / bc
                    if p.is_file():
                        return p
        except (json.JSONDecodeError, OSError):
            pass
    exp_name = run_dir.parent.parent.parent.name
    cap = PROJECT_ROOT / "configs/experiment/capacity_ablation" / f"{exp_name}.yaml"
    if cap.is_file():
        return cap
    p = PROJECT_ROOT / "configs/experiment/process_surrogate_edge_all_v3.yaml"
    return p if p.is_file() else None


def _recompute_one(run_dir: Path, *, dry_run: bool) -> dict:
    cfg = _find_base_config(run_dir)
    pre = preflight_recompute(run_dir, cfg)
    out: dict = {
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)) if run_dir.is_relative_to(PROJECT_ROOT) else str(run_dir),
        "status": "pending",
        "failed_recompute_reason": pre.get("failed_recompute_reason", ""),
        "checkpoint": pre.get("checkpoint", ""),
        "config": pre.get("config", ""),
    }
    if not pre.get("ok"):
        out["status"] = "failed"
        return out

    overrides = run_dir.parent / "runtime_overrides.json"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/eval_process_surrogate_edge_all.py"),
        "--config",
        str(Path(out["config"]).relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "--checkpoint",
        out["checkpoint"],
        "--split",
        "all",
        "--output-dir",
        str(run_dir.resolve()),
        "--runtime-overrides-file",
        str(overrides.resolve()),
    ]
    if dry_run:
        out["status"] = "dry_run"
        out["command"] = " ".join(cmd)
        return out
    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            out["status"] = "failed"
            out["failed_recompute_reason"] = f"eval_rc={proc.returncode}:{(proc.stderr or proc.stdout or '')[:400]}"
            return out
        val = validate_run_targetrow_complete(run_dir)
        metrics = extract_primary_metrics(run_dir)
        out["status"] = "recomputed" if val.get("ok") else "failed_post_validate"
        if not val.get("ok"):
            out["failed_recompute_reason"] = val.get("reasons", "post_validate_failed")
        out[f"test_{PRIMARY_METRIC_NAME}"] = metrics.get(f"test_{PRIMARY_METRIC_NAME}")
    except (OSError, subprocess.SubprocessError) as exc:
        out["status"] = "failed"
        out["failed_recompute_reason"] = str(exc)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute target-row metrics for existing k-fold run_dirs.")
    parser.add_argument("--root", type=str, default="outputs/process_kfold_capacity_ablation")
    parser.add_argument("--out", type=str, default="", help="Write recompute_report.csv (default: <root>/summary)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = (PROJECT_ROOT / args.root).resolve()
    runs = discover_kfold_runs(root)
    rows = [_recompute_one(rec["run_dir"], dry_run=args.dry_run) for rec in runs]
    df = pd.DataFrame(rows)
    out_dir = Path(args.out) if args.out else root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "recompute_target_row_metrics_report.csv"
    df.to_csv(report_path, index=False)
    n_ok = int((df["status"] == "recomputed").sum()) if not df.empty else 0
    n_fail = int(df["status"].astype(str).str.startswith("failed").sum()) if not df.empty else 0
    print(
        f"[recompute] runs={len(rows)} recomputed={n_ok} failed={n_fail} report={report_path}",
        file=sys.stderr,
    )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
