#!/usr/bin/env python3
"""Audit capacity/method k-fold outputs for target-row primary metrics."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ablation_targetrow_common import (  # noqa: E402
    LEGACY_ANSWER_FRACTION_METRIC,
    PRIMARY_METRIC_NAME,
    TARGET_BALANCED_METRIC_NAME,
    TARGETROW_METRIC_SCHEMA_VERSION,
    discover_kfold_runs,
    extract_primary_metrics,
    read_target_v4_table,
    validate_run_targetrow_complete,
)

# Provenance policy expectations (include_in_main_verified_macro).
POLICY_INCLUDED = frozenset({"P05_T001", "P05_T002", "P05_T003"})
POLICY_EXCLUDED = frozenset(
    {"P02_T002", "P08_T001", "P08_T002", "P08_T003", "P06_T004"}
)


def _parse_ids(s: str) -> set[str]:
    if not s or (isinstance(s, float) and math.isnan(s)):
        return set()
    return {x.strip() for x in str(s).split(";") if x.strip()}


def _check_separate_co2_rows(df: pd.DataFrame, process_label: str, tids: tuple[str, str]) -> tuple[bool, str]:
    if df is None or df.empty:
        return False, "no_target_metrics_v4"
    sub = df[df["process_id"].astype(str) == process_label]
    found = set(sub["target_id"].astype(str).unique())
    need = set(tids)
    if not need.issubset(found):
        return False, f"missing_rows:{need - found}"
    co2 = sub[sub["target_id"].astype(str).isin(need)]
    if co2["target_id"].nunique() < 2:
        return False, "co2_not_separate_rows"
    if "target_species" in co2.columns and not (co2["target_species"].astype(str) == "CO2").all():
        return False, "species_not_co2"
    return True, "ok"


def _check_p07_h2o_edge(df: pd.DataFrame) -> tuple[bool, str]:
    if df is None or df.empty:
        return False, "no_data"
    row = df[df["target_id"].astype(str) == "P07_T003"]
    if row.empty:
        return False, "P07_T003_missing"
    r = row.iloc[0]
    edge = str(r.get("used_edge_id", r.get("canonical_answer_edge_id", "")))
    stream = str(r.get("used_stream_key_x", r.get("used_stream_key_y", r.get("target_stream", ""))))
    if edge != "P07_E017":
        return False, f"edge={edge}_expected_P07_E017"
    if "RE" not in stream.upper() and str(r.get("target_stream", "")) != "OUT_H2O":
        return False, f"stream={stream}"
    return True, "ok"


def _check_policy_ids_for_df(
    df: pd.DataFrame | None, inc: set[str], exc: set[str], pid: str
) -> list[dict]:
    """Policy checks only for targets that belong to this process fold table."""
    rows: list[dict] = []
    in_table: set[str] = set()
    if df is not None and not df.empty and "target_id" in df.columns:
        in_table = set(df["target_id"].astype(str).unique())
    for tid in POLICY_INCLUDED:
        if not tid.startswith(f"P{int(pid):02d}_"):
            continue
        ok = tid in inc and tid not in exc
        rows.append(
            {
                "check": f"policy_included_{tid}",
                "status": "OK" if ok else "FAIL",
                "detail": f"inc={tid in inc} exc={tid in exc}",
            }
        )
    for tid in POLICY_EXCLUDED:
        if tid in in_table:
            ok = tid not in inc and tid in exc
            rows.append(
                {
                    "check": f"policy_excluded_{tid}",
                    "status": "OK" if ok else "FAIL",
                    "detail": f"inc={tid in inc} exc={tid in exc} in_table=True",
                }
            )
        elif tid.startswith(f"P{int(pid):02d}_"):
            ok = tid not in inc
            rows.append(
                {
                    "check": f"policy_excluded_{tid}",
                    "status": "OK" if ok else "FAIL",
                    "detail": f"not_in_fold_table inc={tid in inc}",
                }
            )
    return rows


def _structural_checks_for_run(rec: dict, df: pd.DataFrame, m: dict) -> list[dict]:
    proc = str(rec["process_id"])
    pid = proc.replace("Process", "")
    checks: list[dict] = []
    if pid == "1":
        ok, det = _check_separate_co2_rows(df, "Process1", ("P01_T001", "P01_T003"))
        checks.append({"check": "P01_CO2_separate_rows", "status": "OK" if ok else "FAIL", "detail": det})
    if pid == "4":
        ok, det = _check_separate_co2_rows(df, "Process4", ("P04_T002", "P04_T003"))
        checks.append({"check": "P04_CO2_separate_rows", "status": "OK" if ok else "FAIL", "detail": det})
    if pid == "7":
        ok, det = _check_p07_h2o_edge(df)
        checks.append({"check": "P07_T003_RE_edge", "status": "OK" if ok else "FAIL", "detail": det})
    if pid == "5":
        inc = _parse_ids(str(m.get("included_target_ids", "")))
        exc = _parse_ids(str(m.get("excluded_target_ids", "")))
        for tid in ("P05_T001", "P05_T002", "P05_T003"):
            ok = tid in inc
            checks.append(
                {
                    "check": f"Process5_{tid}_included",
                    "status": "OK" if ok else "FAIL",
                    "detail": f"in_included={ok}",
                }
            )
    inc = _parse_ids(str(m.get("included_target_ids", "")))
    exc = _parse_ids(str(m.get("excluded_target_ids", "")))
    checks.extend(_check_policy_ids_for_df(df, inc, exc, pid))
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs/process_kfold_capacity_ablation")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Audit output dir (default: <root>/target_row_metric_audit)",
    )
    args = parser.parse_args()

    root = (PROJECT_ROOT / args.root).resolve()
    out_dir = Path(args.out) if args.out else root / "target_row_metric_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    inv_rows: list[dict] = []
    mismatch_rows: list[dict] = []
    structural_rows: list[dict] = []

    for rec in discover_kfold_runs(root):
        run_dir: Path = rec["run_dir"]
        val = validate_run_targetrow_complete(run_dir)
        m = extract_primary_metrics(run_dir)
        df = read_target_v4_table(run_dir)

        test_primary = m.get(f"test_{PRIMARY_METRIC_NAME}", float("nan"))
        test_legacy = m.get(f"test_{LEGACY_ANSWER_FRACTION_METRIC}", float("nan"))
        has_primary = math.isfinite(float(test_primary)) if test_primary is not None else False

        if val.get("ok"):
            metric_status = "ok_target_row_metric_available"
        elif val.get("reasons", "").startswith("schema_mismatch"):
            metric_status = "target_row_metric_recompute_needed"
        elif "legacy" in str(val.get("reasons", "")):
            metric_status = "legacy_only_no_target_row_metric"
        elif not val.get("has_target_metrics_v4"):
            metric_status = "target_metrics_v4_missing"
        elif not (run_dir / "metrics_per_epoch.csv").is_file():
            metric_status = "metrics_per_epoch_missing"
        else:
            metric_status = "target_row_metric_recompute_needed"

        inv = {
            "run_dir": str(run_dir.relative_to(PROJECT_ROOT)) if run_dir.is_relative_to(PROJECT_ROOT) else str(run_dir),
            "exp_name": rec["exp_name"],
            "process_id": rec["process_id"],
            "fold": rec["fold"],
            "has_metrics_per_epoch": (run_dir / "metrics_per_epoch.csv").is_file(),
            "has_target_metrics_v4": val.get("has_target_metrics_v4", False),
            "has_target_metrics_v4_summary": val.get("has_target_metrics_v4_summary", False),
            "has_primary_metric": has_primary,
            "primary_metric_column": f"test_{PRIMARY_METRIC_NAME}" if has_primary else "",
            "metric_schema_version": val.get("targetrow_metric_schema_version", ""),
            "primary_metric_name": val.get("primary_metric_name", ""),
            "has_legacy_metric": math.isfinite(float(test_legacy)) if test_legacy is not None else False,
            "has_target_row_rows": int(m.get("n_targets_total", 0)) > 0,
            "n_target_rows": int(m.get("n_targets_total", 0)),
            "included_target_ids": m.get("included_target_ids", ""),
            "excluded_target_ids": m.get("excluded_target_ids", ""),
            "metric_status": metric_status,
            "validation_ok": val.get("ok", False),
            "validation_reasons": val.get("reasons", ""),
        }
        inv_rows.append(inv)

        gap = float("nan")
        if has_primary and math.isfinite(float(test_legacy)):
            gap = float(test_primary) - float(test_legacy)
        mismatch_rows.append(
            {
                "run_dir": inv["run_dir"],
                "exp_name": rec["exp_name"],
                "process_id": rec["process_id"],
                "fold": rec["fold"],
                LEGACY_ANSWER_FRACTION_METRIC: test_legacy,
                PRIMARY_METRIC_NAME: test_primary,
                TARGET_BALANCED_METRIC_NAME: m.get(f"test_{TARGET_BALANCED_METRIC_NAME}", float("nan")),
                "metric_gap_legacy_vs_primary": gap,
                "diagnosis": metric_status,
            }
        )

        for chk in _structural_checks_for_run(rec, df, m):
            structural_rows.append({**inv, **chk})

    inv_df = pd.DataFrame(inv_rows)
    mis_df = pd.DataFrame(mismatch_rows)
    struct_df = pd.DataFrame(structural_rows)
    inv_df.to_csv(out_dir / "capacity_metric_inventory.csv", index=False)
    mis_df.to_csv(out_dir / "capacity_metric_mismatch_summary.csv", index=False)
    if not struct_df.empty:
        struct_df.to_csv(out_dir / "capacity_structural_checks.csv", index=False)

    n_runs = len(inv_rows)
    n_val_ok = int(inv_df["validation_ok"].sum()) if not inv_df.empty else 0
    n_primary = int(inv_df["has_primary_metric"].sum()) if not inv_df.empty else 0
    struct_fail = (
        int((struct_df["status"] == "FAIL").sum()) if not struct_df.empty and "status" in struct_df.columns else 0
    )
    overall = "OK" if n_runs > 0 and n_val_ok == n_runs and struct_fail == 0 else "FAIL"

    fold_lines: list[str] = []
    if not inv_df.empty:
        for _, r in inv_df.sort_values(["exp_name", "process_id", "fold"]).iterrows():
            fold_status = "OK" if r.get("validation_ok") else "FAIL"
            fold_lines.append(
                f"| {r['exp_name']} | {r['process_id']} | {r['fold']} | {fold_status} | "
                f"{r.get('metric_schema_version', '')} | {r.get('primary_metric_name', '')} | "
                f"{r.get('validation_reasons', '')} |"
            )
    fold_table = (
        "| exp | process | fold | status | schema | primary_metric | reasons |\n"
        "|-----|---------|------|--------|--------|----------------|----------|\n"
        + "\n".join(fold_lines)
        if fold_lines
        else "| (no runs) | | | | | | |"
    )

    struct_summary = ""
    if not struct_df.empty:
        fail_checks = struct_df[struct_df["status"] == "FAIL"]
        if not fail_checks.empty:
            struct_summary = "\n### Structural check failures\n\n```\n" + fail_checks[
                ["run_dir", "check", "detail"]
            ].head(20).to_string(index=False) + "\n```"

    md = f"""# Capacity ablation target-row metric audit

**Overall status: {overall}**

Root: `{root}`

## Summary

| Item | Count |
|------|-------|
| Runs scanned | {n_runs} |
| Validation OK (complete bundle) | {n_val_ok} |
| Primary metric present | {n_primary} |
| Structural check FAIL | {struct_fail} |

## Required schema

- `targetrow_metric_schema_version` = `{TARGETROW_METRIC_SCHEMA_VERSION}`
- `primary_metric_name` = `{PRIMARY_METRIC_NAME}`

## Per-fold status

{fold_table}

## Structural checks (global policy)

- Process1: `P01_T001` and `P01_T003` are separate CO2 target rows
- Process4: `P04_T002` and `P04_T003` are separate CO2 target rows
- Process7: `P07_T003` H2O on `P07_E017` / RE edge
- Process5: `P05_T001`, `P05_T002`, `P05_T003` in `included_target_ids`
- Policy excluded from main macro: `P02_T002`, `P08_T001`, `P08_T002`, `P08_T003`, `P06_T004`

{struct_summary}

## Optuna comparison

- Historical Optuna ~0.9076 = **legacy_answer_fraction_r2** (auxiliary).
- Capacity primary = **{PRIMARY_METRIC_NAME}** — do not compare directly.

## If FAIL

```bash
python scripts/recompute_capacity_ablation_target_row_metrics.py --root {root.relative_to(PROJECT_ROOT)}
python scripts/aggregate_capacity_ablation_results.py --root {root.relative_to(PROJECT_ROOT)}
```

## Files

- `capacity_metric_inventory.csv`
- `capacity_metric_mismatch_summary.csv`
- `capacity_structural_checks.csv` (per-run structural OK/FAIL)
"""
    (out_dir / "CAPACITY_ABLATION_TARGET_ROW_METRIC_AUDIT.md").write_text(md, encoding="utf-8")
    print(f"[audit] overall={overall} wrote {out_dir}", file=sys.stderr)
    return 0 if overall == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
