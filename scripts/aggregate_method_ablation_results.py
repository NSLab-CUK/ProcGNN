#!/usr/bin/env python3
"""Aggregate method-ablation K-fold results and generate comparison / convergence plots."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from method_ablation_registry import CATALOG_BY_NAME, changed_config_keys  # noqa: E402


def _load_capacity_aggregate():
    global sys_path_inserted
    cap_path = PROJECT_ROOT / "scripts" / "aggregate_capacity_ablation_results.py"
    spec = importlib.util.spec_from_file_location("cap_agg", cap_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {cap_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cap = _load_capacity_aggregate()

_discover_runs = cap._discover_runs
_extract_fold_row_base = cap._extract_fold_row
_load_epoch_table = cap._load_epoch_table
_diagnose_convergence = cap._diagnose_convergence
_PlotContext = cap.PlotContext
_plot_fold_convergence = cap._plot_fold_convergence
_plot_process_summary = cap._plot_process_summary
_col_first = cap._col_first
_safe_read_json = cap._safe_read_json
_load_target_reference = cap._load_target_reference
_target_rows_for_run = cap._target_rows_for_run


def _read_target_v4_table(run_dir: Path) -> pd.DataFrame | None:
    for rel in ("test/target_metrics_v4.csv", "target_metrics_v4.csv"):
        p = run_dir / rel
        if p.is_file():
            try:
                return pd.read_csv(p)
            except (OSError, pd.errors.EmptyDataError):
                return None
    return None


def _extract_fold_row(rec: dict) -> dict:
    row = _extract_fold_row_base(rec)
    entry = CATALOG_BY_NAME.get(rec["experiment_name"], {})
    row["ablation_group"] = entry.get("ablation_group", "")
    row["changed_config_keys"] = "; ".join(changed_config_keys(entry))
    run_dir: Path = rec["run_dir"]
    skip_p = run_dir / "test" / "skipped_targets_v4.csv"
    if not skip_p.is_file():
        skip_p = run_dir / "skipped_targets_v4.csv"
    n_skip = 0
    if skip_p.is_file():
        try:
            sdf = pd.read_csv(skip_p)
            n_skip = int(len(sdf))
        except (OSError, pd.errors.EmptyDataError):
            n_skip = 0
    row["skipped_targets"] = n_skip
    row["legacy_target_h2_r2"] = row.get("target_h2_r2", np.nan)
    row["legacy_tailgas_co2_r2"] = row.get("tailgas_co2_r2", np.nan)
    return row


def _plot_method_comparisons(
    ctx: cap.PlotContext,
    exp_summary: pd.DataFrame,
    proc_summary: pd.DataFrame,
    diag_df: pd.DataFrame,
) -> None:
    if not ctx.make_plots or exp_summary.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = ctx.plot_format
    plot_dir = ctx.summary_dir / "plots"
    df = exp_summary.sort_values("mean_target_v4_macro_r2", ascending=False, na_position="last")
    baseline_row = df[df["experiment_name"] == "exp_B0_baseline"]
    baseline_r2 = (
        float(baseline_row.iloc[0]["mean_target_v4_macro_r2"]) if not baseline_row.empty else float("nan")
    )

    x = np.arange(len(df))
    colors = ["C3" if n == "exp_B0_baseline" else "C0" for n in df["experiment_name"]]

    try:
        fig, ax = plt.subplots(figsize=(max(12, len(df) * 0.42), 5.5))
        y = pd.to_numeric(df["mean_target_v4_macro_r2"], errors="coerce")
        yerr = pd.to_numeric(df.get("std_target_v4_macro_r2"), errors="coerce")
        ax.bar(x, y, yerr=yerr, capsize=3, color=colors, alpha=0.88)
        ax.set_xticks(x)
        ax.set_xticklabels(df["experiment_name"], rotation=55, ha="right", fontsize=8)
        ax.set_ylabel("mean_target_v4_macro_r2")
        ax.set_title("method_ablation_rank_target_v4_macro_r2")
        ax.grid(True, axis="y", alpha=0.3)
        ctx.savefig(fig, plot_dir / f"method_ablation_rank_target_v4_macro_r2.{ext}", plot_id="rank_v4")
    except Exception as exc:
        ctx.record_error(plot_id="rank_v4", path=str(plot_dir), error=str(exc))

    try:
        fig, ax = plt.subplots(figsize=(max(12, len(df) * 0.42), 5.5))
        delta = pd.to_numeric(df["mean_target_v4_macro_r2"], errors="coerce") - baseline_r2
        ax.bar(x, delta, color=colors, alpha=0.88)
        ax.axhline(0.0, color="k", linewidth=1.0, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(df["experiment_name"], rotation=55, ha="right", fontsize=8)
        ax.set_ylabel("delta macro R2 vs exp_B0_baseline")
        ax.set_title("ablation_delta_from_baseline")
        ax.grid(True, axis="y", alpha=0.3)
        ctx.savefig(fig, plot_dir / f"ablation_delta_from_baseline.{ext}", plot_id="delta_baseline")
    except Exception as exc:
        ctx.record_error(plot_id="delta_baseline", path=str(plot_dir), error=str(exc))

    if not proc_summary.empty and "mean_target_v4_macro_r2" in proc_summary.columns:
        try:
            pivot = proc_summary.pivot_table(
                index="process_id",
                columns="experiment_name",
                values="mean_target_v4_macro_r2",
                aggfunc="mean",
            )
            fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.45), max(5, pivot.shape[0] * 0.35)))
            im = ax.imshow(pivot.values.astype(float), aspect="auto", cmap="viridis")
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels(pivot.columns, rotation=55, ha="right", fontsize=7)
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels(pivot.index)
            ax.set_title("process_by_experiment_target_v4_r2_heatmap")
            fig.colorbar(im, ax=ax, label="mean_target_v4_macro_r2")
            ctx.savefig(fig, plot_dir / f"process_by_experiment_target_v4_r2_heatmap.{ext}", plot_id="heatmap")
        except Exception as exc:
            ctx.record_error(plot_id="heatmap", path=str(plot_dir), error=str(exc))

    if "ablation_group" in df.columns:
        try:
            gdf = df[df["experiment_name"] != "exp_B0_baseline"].copy()
            fig, ax = plt.subplots(figsize=(9, 5))
            for grp, g in gdf.groupby("ablation_group"):
                ax.scatter(
                    [grp] * len(g),
                    pd.to_numeric(g["mean_target_v4_macro_r2"], errors="coerce"),
                    label=grp,
                    s=60,
                    alpha=0.85,
                )
            if math.isfinite(baseline_r2):
                ax.axhline(baseline_r2, color="C3", linestyle="--", linewidth=1.2, label="baseline")
            ax.set_ylabel("mean_target_v4_macro_r2")
            ax.set_title("ablation_group_summary")
            ax.tick_params(axis="x", rotation=30)
            ax.legend(loc="best", fontsize=7)
            ax.grid(True, alpha=0.3)
            ctx.savefig(fig, plot_dir / f"ablation_group_summary.{ext}", plot_id="group_summary")
        except Exception as exc:
            ctx.record_error(plot_id="group_summary", path=str(plot_dir), error=str(exc))

    try:
        fig, ax = plt.subplots(figsize=(max(12, len(df) * 0.45), 5))
        w = 0.25
        for i, (col, label) in enumerate(
            (("mean_h2_r2", "H2"), ("mean_co2_r2", "CO2"), ("mean_h2o_r2", "H2O"))
        ):
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce")
            ax.bar(x + (i - 1) * w, vals, width=w, label=label, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(df["experiment_name"], rotation=55, ha="right", fontsize=8)
        ax.set_ylabel("mean species R2")
        ax.set_title("species_r2_by_experiment")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        ctx.savefig(fig, plot_dir / f"species_r2_by_experiment.{ext}", plot_id="species_r2")
    except Exception as exc:
        ctx.record_error(plot_id="species_r2", path=str(plot_dir), error=str(exc))

    if not diag_df.empty:
        try:
            agg = (
                diag_df.groupby("experiment_name")[["best_val_target_v4_macro_r2", "final_val_target_v4_macro_r2"]]
                .mean()
                .reset_index()
            )
            xx = np.arange(len(agg))
            fig, ax = plt.subplots(figsize=(max(10, len(agg) * 0.4), 5))
            ax.plot(xx, agg["best_val_target_v4_macro_r2"], "o-", label="best_val macro R2")
            ax.plot(xx, agg["final_val_target_v4_macro_r2"], "s--", label="final_val macro R2")
            ax.set_xticks(xx)
            ax.set_xticklabels(agg["experiment_name"], rotation=55, ha="right", fontsize=8)
            ax.set_title("convergence_best_vs_final_r2")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ctx.savefig(fig, plot_dir / f"convergence_best_vs_final_r2.{ext}", plot_id="best_vs_final")
        except Exception as exc:
            ctx.record_error(plot_id="best_vs_final", path=str(plot_dir), error=str(exc))

        try:
            counts = diag_df["converged_status"].value_counts()
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.bar(counts.index.astype(str), counts.values, color="C2", alpha=0.85)
            ax.set_title("convergence_status_count")
            ax.set_ylabel("fold count")
            ax.tick_params(axis="x", rotation=25)
            ctx.savefig(fig, plot_dir / f"convergence_status_count.{ext}", plot_id="conv_status")
        except Exception as exc:
            ctx.record_error(plot_id="conv_status", path=str(plot_dir), error=str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate method-ablation K-fold outputs.")
    parser.add_argument("--root", type=str, default="outputs/process_kfold_method_ablation")
    parser.add_argument("--make-plots", dest="make_plots", action="store_true", default=True)
    parser.add_argument("--no-plots", dest="make_plots", action="store_false")
    parser.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    root = (PROJECT_ROOT / args.root).resolve()
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    ctx = _PlotContext(
        make_plots=args.make_plots,
        plot_format=args.plot_format,
        dpi=args.dpi,
        summary_dir=summary_dir,
    )

    runs = _discover_runs(root)
    fold_rows = [_extract_fold_row(r) for r in runs]
    fold_df = pd.DataFrame(fold_rows)
    if not fold_df.empty:
        fold_df = fold_df.sort_values(["experiment_name", "process_id", "fold_id"])

    missing: list[dict] = []
    diag_rows: list[dict] = []
    target_ref = _load_target_reference()
    target_parts: list[pd.DataFrame] = []

    by_exp_proc: dict[tuple[str, str], list] = {}
    for rec in runs:
        run_dir: Path = rec["run_dir"]
        exp = rec["experiment_name"]
        proc = rec["process_id"]
        fold_id = rec["fold_id"]
        fold_plots = rec["fold_dir"] / "plots"

        if not fold_df.empty:
            match = fold_df[
                (fold_df["experiment_name"] == exp)
                & (fold_df["process_id"] == proc)
                & (fold_df["fold_id"] == fold_id)
            ]
            if not match.empty and match.iloc[0]["status"] != "ok":
                missing.append(
                    {
                        "experiment_name": exp,
                        "process_id": proc,
                        "fold_id": fold_id,
                        "run_dir": str(run_dir),
                        "reason": match.iloc[0]["status"],
                    }
                )

        epoch_df = _load_epoch_table(run_dir)
        diag = _diagnose_convergence(epoch_df)
        diag_rows.append(
            {
                "experiment_name": exp,
                "process_id": proc,
                "fold_id": fold_id,
                **{k: v for k, v in diag.items() if k != "r2_metric_used"},
                "r2_metric_column": diag.get("r2_metric_used", ""),
            }
        )
        try:
            _plot_fold_convergence(
                ctx,
                experiment_name=exp,
                process_id=proc,
                fold_id=fold_id,
                run_dir=run_dir,
                fold_plots_dir=fold_plots,
            )
        except Exception as exc:
            ctx.record_error(plot_id="fold_convergence", path=str(fold_plots), error=traceback.format_exc())

        if epoch_df is not None and not epoch_df.empty:
            by_exp_proc.setdefault((exp, proc), []).append((fold_id, epoch_df, run_dir))

        target_part = _target_rows_for_run(rec, target_ref)
        if not target_part.empty:
            target_parts.append(target_part)

    for (exp, proc), items in by_exp_proc.items():
        proc_plots = root / exp / proc / "plots"
        sample_df = items[0][1]
        r2_note = ""
        if _col_first(sample_df, ["val_val_target_v4_macro_r2", "val_target_v4_macro_r2"]) is None:
            r2_note = "fallback metric used for convergence_target_v4_r2_mean_std"
        try:
            _plot_process_summary(
                ctx,
                experiment_name=exp,
                process_id=proc,
                fold_dfs=[(fid, df) for fid, df, _ in items],
                proc_plots_dir=proc_plots,
                r2_fallback_note=r2_note,
            )
        except Exception as exc:
            ctx.record_error(plot_id="process_summary", path=str(proc_plots), error=traceback.format_exc())

    for exp_dir in sorted(root.glob("*/Process*")):
        if exp_dir.parent.name == "summary":
            continue
        meta = _safe_read_json(exp_dir / "run_meta.json")
        if isinstance(meta, dict) and meta.get("status") == "failed":
            missing.append(
                {
                    "experiment_name": exp_dir.parent.name,
                    "process_id": exp_dir.name,
                    "fold_id": "",
                    "reason": f"run_meta failed rc={meta.get('return_code')}",
                }
            )

    fold_df.to_csv(summary_dir / "experiment_process_summary.csv", index=False)

    exp_summary_rows: list[dict] = []
    if not fold_df.empty:
        for exp, g in fold_df.groupby("experiment_name"):
            entry = CATALOG_BY_NAME.get(exp, {})
            ok = g[g["status"] == "ok"]
            exp_summary_rows.append(
                {
                    "experiment_name": exp,
                    "ablation_group": entry.get("ablation_group", ""),
                    "changed_config_keys": "; ".join(changed_config_keys(entry)),
                    "n_processes_done": int(ok["process_id"].nunique()) if not ok.empty else 0,
                    "n_folds_done": int(len(ok)),
                    "mean_target_v4_macro_r2": float(
                        pd.to_numeric(ok["target_v4_macro_r2"], errors="coerce").mean()
                    )
                    if not ok.empty
                    else np.nan,
                    "std_target_v4_macro_r2": float(
                        pd.to_numeric(ok["target_v4_macro_r2"], errors="coerce").std(ddof=0)
                    )
                    if len(ok) > 1
                    else 0.0,
                    "mean_h2_r2": float(pd.to_numeric(ok["target_h2_r2"], errors="coerce").mean())
                    if not ok.empty
                    else np.nan,
                    "mean_co2_r2": float(pd.to_numeric(ok["tailgas_co2_r2"], errors="coerce").mean())
                    if not ok.empty
                    else np.nan,
                    "mean_answer_targets_r2": float(pd.to_numeric(ok.get("answer_targets_r2"), errors="coerce").mean())
                    if "answer_targets_r2" in ok.columns and not ok.empty
                    else np.nan,
                    "mean_best_val_answer_targets_r2": float(
                        pd.to_numeric(ok.get("best_val_answer_targets_r2"), errors="coerce").mean()
                    )
                    if "best_val_answer_targets_r2" in ok.columns and not ok.empty
                    else np.nan,
                    "mean_h2o_r2": float(pd.to_numeric(ok.get("target_h2o_r2"), errors="coerce").mean())
                    if "target_h2o_r2" in ok.columns and not ok.empty
                    else np.nan,
                    "mean_edge_all_r2": float(pd.to_numeric(ok["edge_all_r2"], errors="coerce").mean())
                    if not ok.empty
                    else np.nan,
                    "mean_legacy_target_h2_r2": float(
                        pd.to_numeric(ok.get("legacy_target_h2_r2"), errors="coerce").mean()
                    )
                    if not ok.empty
                    else np.nan,
                    "mean_legacy_tailgas_co2_r2": float(
                        pd.to_numeric(ok.get("legacy_tailgas_co2_r2"), errors="coerce").mean()
                    )
                    if not ok.empty
                    else np.nan,
                    "failed_runs": int((g["status"] != "ok").sum()),
                    "skipped_targets": int(pd.to_numeric(g.get("skipped_targets"), errors="coerce").sum()),
                }
            )
    exp_summary = pd.DataFrame(exp_summary_rows)
    exp_summary.to_csv(summary_dir / "experiment_summary.csv", index=False)

    if not exp_summary.empty:
        exp_summary.sort_values("mean_target_v4_macro_r2", ascending=False, na_position="last").to_csv(
            summary_dir / "experiment_rank_by_target_v4_r2.csv",
            index=False,
        )
        if "mean_answer_targets_r2" in exp_summary.columns:
            exp_summary.sort_values("mean_answer_targets_r2", ascending=False, na_position="last").to_csv(
                summary_dir / "experiment_rank_by_answer_targets_r2.csv",
                index=False,
            )
        if "mean_best_val_answer_targets_r2" in exp_summary.columns:
            exp_summary.sort_values("mean_best_val_answer_targets_r2", ascending=False, na_position="last").to_csv(
                summary_dir / "experiment_rank_by_best_val_answer_targets_r2.csv",
                index=False,
            )

    proc_summary_rows: list[dict] = []
    if not fold_df.empty:
        ok = fold_df[fold_df["status"] == "ok"]
        for (exp, proc), g in ok.groupby(["experiment_name", "process_id"]):
            r2 = pd.to_numeric(g["target_v4_macro_r2"], errors="coerce").dropna()
            answer_r2 = pd.to_numeric(g.get("answer_targets_r2"), errors="coerce").dropna()
            best_val_answer_r2 = pd.to_numeric(g.get("best_val_answer_targets_r2"), errors="coerce").dropna()
            proc_summary_rows.append(
                {
                    "experiment_name": exp,
                    "process_id": proc,
                    "n_folds": len(g),
                    "mean_target_v4_macro_r2": float(r2.mean()) if not r2.empty else np.nan,
                    "mean_answer_targets_r2": float(answer_r2.mean()) if not answer_r2.empty else np.nan,
                    "mean_best_val_answer_targets_r2": float(best_val_answer_r2.mean())
                    if not best_val_answer_r2.empty
                    else np.nan,
                }
            )
    proc_summary = pd.DataFrame(proc_summary_rows)

    tgt_long = pd.concat(target_parts, ignore_index=True) if target_parts else pd.DataFrame()
    if not tgt_long.empty:
        tgt_long.to_csv(summary_dir / "target_v4_metrics_by_target.csv", index=False)
        tgt_long.to_csv(summary_dir / "main_target_metrics_by_target.csv", index=False)
        group_cols = [
            c
            for c in (
                "experiment_name",
                "process_id",
                "target_id",
                "target_species",
                "target_feature_name",
                "target_stream_node",
                "target_formula",
                "canonical_answer_edge_id",
                "required_stream_key",
            )
            if c in tgt_long.columns
        ]
        value_cols = [c for c in ("r2", "mae", "rmse", "relative_accuracy_score", "n_samples") if c in tgt_long.columns]
        agg_spec = {}
        for c in value_cols:
            if c == "n_samples":
                agg_spec[c] = "sum"
            else:
                agg_spec[c] = ["mean", "std", "min", "max"]
        summary = tgt_long.groupby(group_cols, dropna=False).agg(agg_spec).reset_index()
        summary.columns = [
            "_".join([str(x) for x in col if str(x)]) if isinstance(col, tuple) else str(col)
            for col in summary.columns
        ]
        summary.to_csv(summary_dir / "main_target_metrics_by_target_summary.csv", index=False)
        wide = summary.pivot_table(
            index=["experiment_name", "process_id"],
            columns="target_id",
            values="r2_mean",
            aggfunc="first",
        ).reset_index()
        wide.to_csv(summary_dir / "main_target_r2_wide.csv", index=False)
    else:
        pd.DataFrame().to_csv(summary_dir / "target_v4_metrics_by_target.csv", index=False)
        pd.DataFrame().to_csv(summary_dir / "main_target_metrics_by_target.csv", index=False)
        pd.DataFrame().to_csv(summary_dir / "main_target_metrics_by_target_summary.csv", index=False)
        pd.DataFrame().to_csv(summary_dir / "main_target_r2_wide.csv", index=False)

    pd.DataFrame(missing).drop_duplicates().to_csv(summary_dir / "missing_or_failed_runs.csv", index=False)
    diag_df = pd.DataFrame(diag_rows)
    diag_df.to_csv(summary_dir / "convergence_diagnostics.csv", index=False)
    pd.DataFrame(ctx.plot_errors).to_csv(summary_dir / "plot_errors.csv", index=False)

    try:
        _plot_method_comparisons(ctx, exp_summary, proc_summary, diag_df)
    except Exception as exc:
        ctx.record_error(plot_id="method_comparisons", path=str(summary_dir / "plots"), error=traceback.format_exc())

    baseline_json = PROJECT_ROOT / "configs/experiment/method_ablation/baseline_resolved.json"
    report = f"""# Method ablation report

Aggregated from: `{root}`

## Primary metric

**target_v4_macro_r2** — stream-derived target amount R² (`Mole_Flow * Frac_*`).

For comparison with older Optuna results, use **best_val_answer_targets_r2** (best validation answer-target fraction R² over epochs) or **answer_targets_r2** for final test artifacts.

Baseline reference: `{baseline_json}`

## Summary CSVs

- `experiment_process_summary.csv` — per experiment × process × fold
- `experiment_summary.csv` — per experiment (includes `ablation_group`, `changed_config_keys`)
- `experiment_rank_by_target_v4_r2.csv`
- `experiment_rank_by_answer_targets_r2.csv`
- `experiment_rank_by_best_val_answer_targets_r2.csv`
- `target_v4_metrics_by_target.csv`
- `main_target_metrics_by_target.csv`
- `main_target_metrics_by_target_summary.csv`
- `main_target_r2_wide.csv`
- `convergence_diagnostics.csv`
- `missing_or_failed_runs.csv`
- `plot_errors.csv`

## Summary plots (`summary/plots/`)

- `method_ablation_rank_target_v4_macro_r2.png`
- `ablation_delta_from_baseline.png`
- `process_by_experiment_target_v4_r2_heatmap.png`
- `ablation_group_summary.png`
- `species_r2_by_experiment.png`
- `convergence_best_vs_final_r2.png`
- `convergence_status_count.png`

Fold/process convergence plots mirror capacity ablation under each `<EXP>/Process<P>/`.

## Convergence thresholds

See `configs/experiment/method_ablation/README.md` (same rules as capacity ablation).
"""
    (summary_dir / "README_method_ablation_report.md").write_text(report, encoding="utf-8")
    print(f"[done] summary written to {summary_dir}")


if __name__ == "__main__":
    main()
