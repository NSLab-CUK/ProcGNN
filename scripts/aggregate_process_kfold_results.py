#!/usr/bin/env python3
"""Aggregate metrics from outputs/process_kfold/** K-fold runs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_RUN_DIR_RE = re.compile(r"^process_kfold_P(\d+)_F(\d+)$")


def _parse_run_dir(path: Path) -> tuple[str, int, int] | None:
    """Return (process_label, process_num, fold_1based) or None."""
    m = _RUN_DIR_RE.match(path.name)
    if not m:
        return None
    pnum = int(m.group(1))
    fold = int(m.group(2))
    return (f"Process{pnum}", pnum, fold)


def _safe_read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _pick_test_metrics(run_dir: Path) -> dict:
    """Prefer test/metrics.json; fall back to root metrics.json test/* keys."""
    tj = _safe_read_json(run_dir / "test" / "metrics.json")
    if isinstance(tj, dict) and tj:
        return tj
    mj = _safe_read_json(run_dir / "metrics.json")
    if not isinstance(mj, dict):
        return {}
    out = {}
    for k, v in mj.items():
        if k.startswith("test/"):
            out[k[5:]] = v
    return out


def _read_target_v4_table(run_dir: Path) -> pd.DataFrame | None:
    for rel in ("test/target_metrics_v4.csv", "target_metrics_v4.csv"):
        p = run_dir / rel
        if p.is_file():
            try:
                return pd.read_csv(p)
            except (OSError, pd.errors.EmptyDataError):
                return None
    return None


def _read_target_v4_summary(run_dir: Path) -> pd.DataFrame | None:
    for rel in ("test/target_metrics_v4_summary.csv", "target_metrics_v4_summary.csv"):
        p = run_dir / rel
        if p.is_file():
            try:
                return pd.read_csv(p)
            except (OSError, pd.errors.EmptyDataError):
                return None
    return None


def _macro_mean_target_v4_r2(run_dir: Path) -> float:
    """Primary v4 metric: mean of per-target R² (macro), not pooled flatten R²."""
    for rel in ("test/target_metrics_v4_summary.csv", "target_metrics_v4_summary.csv"):
        p = run_dir / rel
        if p.is_file():
            try:
                summ = pd.read_csv(p)
            except (OSError, pd.errors.EmptyDataError):
                continue
            if summ.empty or "target_id" not in summ.columns:
                continue
            macro = summ[summ["target_id"].astype(str) == "__ALL_macro_split__"]
            if macro.empty and "summary_kind" in summ.columns:
                macro = summ[summ["summary_kind"].astype(str) == "split_macro"]
            if not macro.empty and "r2" in macro.columns:
                v = pd.to_numeric(macro.iloc[0]["r2"], errors="coerce")
                if pd.notna(v):
                    return float(v)
    p = run_dir / "test" / "target_metrics_v4.csv"
    if not p.is_file():
        p = run_dir / "target_metrics_v4.csv"
    if not p.is_file():
        return float("nan")
    try:
        df = pd.read_csv(p)
    except (OSError, pd.errors.EmptyDataError):
        return float("nan")
    if df.empty or "target_id" not in df.columns or "r2" not in df.columns:
        return float("nan")
    if "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == "test"]
    sub = df[~df["target_id"].astype(str).str.startswith("__ALL")]
    if "status" in sub.columns:
        sub = sub[sub["status"].astype(str) == "ok"]
    if sub.empty:
        return float("nan")
    if "process_id" in sub.columns:
        proc_macros: list[float] = []
        for _, g in sub.groupby("process_id"):
            r2 = pd.to_numeric(g["r2"], errors="coerce").dropna()
            if not r2.empty:
                proc_macros.append(float(r2.mean()))
        if proc_macros:
            return float(sum(proc_macros) / len(proc_macros))
    r2 = pd.to_numeric(sub["r2"], errors="coerce").dropna()
    if r2.empty:
        return float("nan")
    return float(r2.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate process_kfold run metrics.")
    parser.add_argument("--output-root", type=str, default="outputs/process_kfold")
    args = parser.parse_args()

    root = (PROJECT_ROOT / args.output_root).resolve()
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    run_dirs: list[Path] = []
    if root.is_dir():
        for pdir in sorted(root.glob("Process*")):
            if not pdir.is_dir():
                continue
            for fdir in sorted(pdir.glob("fold_*")):
                if not fdir.is_dir():
                    continue
                for rdir in sorted(fdir.iterdir()):
                    if rdir.is_dir() and _parse_run_dir(rdir):
                        run_dirs.append(rdir)

    fold_rows: list[dict] = []
    target_parts: list[pd.DataFrame] = []
    missing: list[dict] = []

    for run_dir in run_dirs:
        parsed = _parse_run_dir(run_dir)
        if not parsed:
            continue
        plabel, pnum, fold = parsed
        tm = _pick_test_metrics(run_dir)
        status = "ok"
        if not tm:
            status = "missing_metrics"
            missing.append({"process_id": plabel, "fold": fold, "run_dir": str(run_dir), "reason": "no test metrics"})

        test_loss = tm.get("loss_total", tm.get("loss_edge", np.nan))
        edge_r2 = tm.get("edge_all_r2_orig", tm.get("r2", np.nan))

        mj_root = _safe_read_json(run_dir / "metrics.json") or {}
        tv_r2 = mj_root.get("test/target_v4_r2", np.nan)
        tv_macro_r2 = _macro_mean_target_v4_r2(run_dir)
        tv_mae = mj_root.get("test/target_v4_mae", np.nan)
        tv_rmse = mj_root.get("test/target_v4_rmse", np.nan)
        summ = _read_target_v4_summary(run_dir)
        n_targets_present = np.nan
        if summ is not None and not summ.empty and "n_targets_present" in summ.columns:
            try:
                n_targets_present = float(summ.iloc[0]["n_targets_present"])
            except (TypeError, ValueError, KeyError):
                pass

        fold_rows.append(
            {
                "process_id": plabel,
                "fold": fold,
                "run_dir": str(run_dir.relative_to(PROJECT_ROOT)) if run_dir.is_relative_to(PROJECT_ROOT) else str(run_dir),
                "test_loss": test_loss,
                "test_edge_all_r2": edge_r2,
                "test_target_v4_r2": tv_r2,
                "test_target_v4_macro_r2": tv_macro_r2,
                "test_target_v4_mae": tv_mae,
                "test_target_v4_rmse": tv_rmse,
                "n_targets_present": n_targets_present,
                "status": status,
            }
        )

        tv_df = _read_target_v4_table(run_dir)
        if tv_df is not None and not tv_df.empty:
            sub = tv_df.copy()
            sub["process_id"] = plabel
            sub["fold"] = fold
            target_parts.append(sub)

    process_fold_metrics = pd.DataFrame(fold_rows)
    if not process_fold_metrics.empty:
        process_fold_metrics = process_fold_metrics.sort_values(["process_id", "fold"])
    process_fold_metrics.to_csv(summary_dir / "process_fold_metrics.csv", index=False)

    # Per-target long table
    tgt_long = pd.DataFrame()
    if target_parts:
        tgt_long = pd.concat(target_parts, ignore_index=True)
        want = [
            "process_id",
            "fold",
            "target_id",
            "split",
            "target_species",
            "target_feature_name",
            "required_stream_key",
            "mae",
            "rmse",
            "r2",
            "n_samples",
        ]
        cols_keep = [c for c in want if c in tgt_long.columns]
        out_tgt = tgt_long[cols_keep].copy()
        if "required_stream_key" in out_tgt.columns:
            out_tgt.rename(columns={"required_stream_key": "stream"}, inplace=True)
        elif "stream" not in out_tgt.columns:
            out_tgt["stream"] = ""
        out_tgt.rename(
            columns={
                "target_feature_name": "target_name",
                "target_species": "species",
                "mae": "MAE",
                "rmse": "RMSE",
                "r2": "R2",
            },
            inplace=True,
        )
        out_tgt.to_csv(summary_dir / "target_metrics_by_process_fold.csv", index=False)
    else:
        pd.DataFrame(
            columns=[
                "process_id",
                "fold",
                "target_id",
                "stream",
                "species",
                "target_name",
                "MAE",
                "RMSE",
                "R2",
                "n_samples",
            ]
        ).to_csv(summary_dir / "target_metrics_by_process_fold.csv", index=False)
        tgt_long = pd.DataFrame()

    # Process-level summary (over folds)
    def _agg_process(df: pd.DataFrame, col: str) -> tuple[float, float]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return (np.nan, np.nan)
        return (float(s.mean()), float(s.std(ddof=0)))

    proc_summary: list[dict] = []
    if not process_fold_metrics.empty:
        for pid, g in process_fold_metrics.groupby("process_id"):
            ok = g[g["status"] == "ok"]
            m_r2, s_r2 = _agg_process(ok, "test_target_v4_r2")
            mm_r2, sm_r2 = _agg_process(ok, "test_target_v4_macro_r2")
            me_r2, se_r2 = _agg_process(ok, "test_edge_all_r2")
            proc_summary.append(
                {
                    "process_id": pid,
                    "mean_test_target_v4_r2": m_r2,
                    "std_test_target_v4_r2": s_r2,
                    "mean_test_target_v4_macro_r2": mm_r2,
                    "std_test_target_v4_macro_r2": sm_r2,
                    "mean_test_edge_all_r2": me_r2,
                    "std_test_edge_all_r2": se_r2,
                    "n_completed_folds": int((g["status"] == "ok").sum()),
                    "n_missing_folds": int((g["status"] != "ok").sum()),
                }
            )
    pd.DataFrame(proc_summary).to_csv(summary_dir / "process_summary_metrics.csv", index=False)

    # Target-level across folds
    tproc_rows: list[dict] = []
    if not tgt_long.empty and "target_id" in tgt_long.columns:
        metric_rows = tgt_long[~tgt_long["target_id"].astype(str).str.startswith("__ALL")]
        for (pid, tid), g in metric_rows.groupby(["process_id", "target_id"]):
            r2 = pd.to_numeric(g.get("r2"), errors="coerce").dropna()
            mae = pd.to_numeric(g.get("mae"), errors="coerce").dropna()
            rmse = pd.to_numeric(g.get("rmse"), errors="coerce").dropna()
            name = ""
            if "target_feature_name" in g.columns and not g.empty:
                name = str(g.iloc[0]["target_feature_name"])
            tproc_rows.append(
                {
                    "process_id": pid,
                    "target_id": tid,
                    "target_name": name,
                    "mean_R2": float(r2.mean()) if not r2.empty else np.nan,
                    "std_R2": float(r2.std(ddof=0)) if len(r2) > 1 else 0.0,
                    "mean_MAE": float(mae.mean()) if not mae.empty else np.nan,
                    "std_MAE": float(mae.std(ddof=0)) if len(mae) > 1 else 0.0,
                    "mean_RMSE": float(rmse.mean()) if not rmse.empty else np.nan,
                    "std_RMSE": float(rmse.std(ddof=0)) if len(rmse) > 1 else 0.0,
                    "n_completed_folds": int(len(g["fold"].unique())),
                }
            )
    pd.DataFrame(tproc_rows).to_csv(summary_dir / "target_metrics_by_process.csv", index=False)

    pd.DataFrame(missing).to_csv(summary_dir / "missing_or_failed_runs.csv", index=False)
    if missing:
        pd.DataFrame(missing).to_csv(summary_dir / "missing_runs.csv", index=False)

    exp_rows: list[dict] = []
    if not process_fold_metrics.empty:
        ok = process_fold_metrics[process_fold_metrics["status"] == "ok"]
        exp_rows.append(
            {
                "metric_version": "target_v4_all_targets_v1",
                "n_runs": int(len(ok)),
                "mean_test_target_v4_macro_r2": float(
                    pd.to_numeric(ok["test_target_v4_macro_r2"], errors="coerce").mean()
                ),
                "mean_legacy_edge_all_r2": float(
                    pd.to_numeric(ok["test_edge_all_r2"], errors="coerce").mean()
                ),
                "n_missing_runs": int((process_fold_metrics["status"] != "ok").sum()),
            }
        )
    pd.DataFrame(exp_rows).to_csv(summary_dir / "experiment_summary.csv", index=False)
    if not tgt_long.empty:
        tgt_long.to_csv(summary_dir / "target_v4_metrics_by_target.csv", index=False)

    def _df_md(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return "_(empty)_"
        try:
            return str(df.to_markdown(index=False))
        except Exception:
            return df.to_csv(index=False)

    # --- Markdown report ---
    flatten_note = (
        "`test/target_v4_r2` in metrics.json is the **flatten** aggregate (pooled samples across targets), "
        "matching `__ALL_flatten__` in target_metrics_v4.csv — not macro mean of per-target R². "
        "See `process_fold_metrics.csv` / `process_summary_metrics.csv` columns `test_target_v4_macro_r2` / "
        "`mean_test_target_v4_macro_r2` for the unweighted mean of per-target test R² (aligned with target_metrics_by_process.csv)."
    )
    macro_note = ""
    if not tgt_long.empty:
        metric_rows = tgt_long[~tgt_long["target_id"].astype(str).str.startswith("__ALL")]
        macro_by_pf: list[float] = []
        for (_, _), g in metric_rows.groupby(["process_id", "fold"]):
            r2 = pd.to_numeric(g["r2"], errors="coerce").dropna()
            if not r2.empty:
                macro_by_pf.append(float(r2.mean()))
        if macro_by_pf:
            macro_note = f"Across completed runs, mean of (**macro** = unweighted mean of per-target R²) over process×fold: **{float(np.mean(macro_by_pf)):.6f}** (std {float(np.std(macro_by_pf)):.6f}).\n"

    proc_tbl = ""
    if proc_summary:
        proc_tbl = _df_md(pd.DataFrame(proc_summary))

    low_targets = ""
    if tproc_rows:
        tt = pd.DataFrame(tproc_rows).sort_values("mean_R2").head(15)
        low_targets = _df_md(tt)

    miss_txt = ""
    if missing:
        miss_txt = _df_md(pd.DataFrame(missing))
    else:
        miss_txt = "_(none)_"

    report = f"""# Process K-fold results

## Execution settings

- Aggregated from: `{root}`
- `process_fold_metrics.csv`: one row per discovered run directory (`Process*/fold_*/*`).

## Interpretation

- {flatten_note}
- {macro_note}

## Per-process fold summary (test metrics)

{proc_tbl or "_(no runs found)_"}

## Targets with lowest mean R² (up to 15)

{low_targets or "_(no per-target table)_"}

## Missing / incomplete runs

{miss_txt}

## Caveats

- Edge-all R² is in **original scale** (`edge_all_r2_orig`) when present.
- Incomplete folds (early stop, crash) appear in `missing_runs.csv` or with `status != ok` in `process_fold_metrics.csv`.
- Compare macro vs flatten when judging whether a single poor target dominates pooled metrics.
"""
    (summary_dir / "process_kfold_report.md").write_text(report, encoding="utf-8")
    print(f"[done] wrote summary under {summary_dir}")


if __name__ == "__main__":
    main()
