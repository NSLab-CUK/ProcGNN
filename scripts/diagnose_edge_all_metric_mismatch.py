#!/usr/bin/env python3
"""Diagnose edge_all legacy-fraction R2 vs target_v4 amount R2 mismatches.

This script is read-only with respect to training outputs. It scans capacity
ablation run directories, inventories metric keys, compares legacy answer-edge
fraction R2 with v4 target amount R2, and writes CSV/PNG/MD diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_RE = re.compile(r"^process_kfold_P(?P<pid>\d+)_F(?P<fold>\d+)(?:-\d{8}-\d{6})?$")

TARGET_V4_R2_CANDIDATES = [
    "val_target_v4_macro_r2",
    "val_val_target_v4_macro_r2",
    "val/target_v4_macro_r2",
    "target_v4_macro_r2",
    "test/target_v4_macro_r2",
]
TARGET_V4_MAE_CANDIDATES = [
    "val_target_v4_macro_mae",
    "val_val_target_v4_macro_mae",
    "val/target_v4_macro_mae",
    "target_v4_macro_mae",
]
TARGET_V4_RMSE_CANDIDATES = [
    "val_target_v4_macro_rmse",
    "val_val_target_v4_macro_rmse",
    "val/target_v4_macro_rmse",
    "target_v4_macro_rmse",
]
LEGACY_R2_CANDIDATES = [
    "val_legacy_answer_fraction_macro_r2",
    "val/legacy_answer_fraction_macro_r2",
    "val_metric_answer_all_targets_mean_r2",
    "val/metric_answer_all_targets_mean_r2",
    "metric_answer_all_targets_mean_r2",
    "test/metric_answer_all_targets_mean_r2",
]


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _discover_runs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if exp_dir.name in {"summary", "metric_diagnostics"}:
            continue
        for proc_dir in sorted(exp_dir.glob("Process*")):
            if not proc_dir.is_dir():
                continue
            for fold_dir in sorted(proc_dir.glob("fold_*")):
                if not fold_dir.is_dir():
                    continue
                for run_dir in sorted(p for p in fold_dir.iterdir() if p.is_dir()):
                    m = RUN_RE.match(run_dir.name)
                    if not m:
                        continue
                    rows.append(
                        {
                            "exp_name": exp_dir.name,
                            "process_id": f"Process{int(m.group('pid'))}",
                            "fold": int(m.group("fold")),
                            "run_dir": run_dir,
                        }
                    )
    return rows


def _candidate_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in df.columns]


def _r2_related_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if "r2" in c.lower() or "r²" in c.lower()]


def _target_v4_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if "target_v4" in c.lower()]


def _answer_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if "answer" in c.lower() or "legacy" in c.lower()]


def _first_numeric(df: pd.DataFrame, candidates: list[str]) -> tuple[str, pd.Series | None]:
    for col in candidates:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                return col, s
    return "", None


def _last_best(df: pd.DataFrame, candidates: list[str]) -> tuple[str, float, float, float]:
    col, s = _first_numeric(df, candidates)
    if s is None or s.dropna().empty:
        return col, float("nan"), float("nan"), float("nan")
    finite = s.dropna()
    last = float(finite.iloc[-1])
    best_idx = finite.idxmax()
    best = float(finite.loc[best_idx])
    epoch = float(df.loc[best_idx, "epoch"]) if "epoch" in df.columns else float("nan")
    return col, last, best, epoch


def _macro_from_target_v4_csv(path: Path) -> float:
    df = _read_csv(path)
    if df.empty or "r2" not in df.columns:
        return float("nan")
    if "target_id" in df.columns:
        macro = df[df["target_id"].astype(str).eq("__ALL_macro_split__")]
        if macro.empty and "summary_kind" in df.columns:
            macro = df[df["summary_kind"].astype(str).eq("split_macro")]
        if not macro.empty:
            v = pd.to_numeric(macro.iloc[0]["r2"], errors="coerce")
            return float(v) if pd.notna(v) else float("nan")
        df = df[~df["target_id"].astype(str).str.startswith("__ALL")]
    if "status" in df.columns:
        ok = df[df["status"].astype(str).isin(["ok", "clamped"])]
        if not ok.empty:
            df = ok
    vals = pd.to_numeric(df["r2"], errors="coerce").dropna()
    return float(vals.mean()) if not vals.empty else float("nan")


def _macro_from_answer_edge_csv(path: Path) -> float:
    df = _read_csv(path)
    if df.empty:
        return float("nan")
    r2_cols = [c for c in df.columns if c.lower() in {"r2", "r2_orig"} or c.lower().endswith("_r2")]
    if not r2_cols:
        return float("nan")
    vals: list[float] = []
    for col in r2_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        vals.extend([float(v) for v in s])
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _json_metric(run_dir: Path, candidates: list[str]) -> tuple[str, float]:
    payloads = [_read_json(run_dir / "metrics.json")]
    for split in ("val", "test"):
        payloads.append(_read_json(run_dir / split / "metrics.json"))
    for payload in payloads:
        for key in candidates:
            if key in payload:
                try:
                    return key, float(payload[key])
                except (TypeError, ValueError):
                    pass
    return "", float("nan")


def _target_diagnosis(row: pd.Series) -> str:
    status = str(row.get("status", "") or row.get("r2_status", ""))
    true_std = float(pd.to_numeric(pd.Series([row.get("true_std")]), errors="coerce").iloc[0])
    pred_std = float(pd.to_numeric(pd.Series([row.get("pred_std")]), errors="coerce").iloc[0])
    std_ratio = float(pd.to_numeric(pd.Series([row.get("std_ratio")]), errors="coerce").iloc[0])
    r2 = float(pd.to_numeric(pd.Series([row.get("r2")]), errors="coerce").iloc[0])
    if "low_variance" in status or (math.isfinite(true_std) and true_std < 1e-6):
        return "low_variance_r2_unstable"
    if math.isfinite(std_ratio) and std_ratio < 0.2 and math.isfinite(true_std) and true_std > 1e-6:
        return "prediction_collapse_pred_std_lt_true_std"
    if math.isfinite(pred_std) and math.isfinite(true_std) and pred_std > true_std * 5.0:
        return "prediction_variance_too_high"
    if math.isfinite(r2) and r2 <= -0.95:
        return "r2_clamped_or_very_low"
    return "ok"


def _run_diagnosis(legacy_last: float, v4_last: float, v4_col: str, target_diag: list[str]) -> str:
    if not v4_col and not math.isfinite(v4_last):
        return "target_v4_missing"
    if target_diag and all(d == "low_variance_r2_unstable" for d in target_diag):
        return "low_variance_r2_unstable"
    if math.isfinite(legacy_last) and math.isfinite(v4_last):
        if legacy_last < 0.2 and v4_last >= 0.8:
            return "legacy_low_v4_ok"
        if legacy_last < 0.2 and v4_last < 0.2:
            if any("collapse" in d for d in target_diag):
                return "both_low_possible_underfit"
            return "scaler_or_mapping_suspect"
        if abs(v4_last - legacy_last) > 0.3:
            return "metric_column_name_mismatch"
    return "ok"


def _write_plots(out_dir: Path, mismatch: pd.DataFrame, target_diag: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not mismatch.empty:
        exp_mean = (
            mismatch.groupby("exp_name")[["last_target_v4_macro_r2", "last_legacy_answer_mean_r2"]]
            .mean(numeric_only=True)
            .sort_index()
        )
        if not exp_mean.empty:
            fig, ax = plt.subplots(figsize=(max(8, len(exp_mean) * 0.6), 4.5))
            exp_mean.plot(kind="bar", ax=ax)
            ax.set_ylabel("R2")
            ax.set_title("target_v4_macro_r2 vs legacy_answer_fraction_macro_r2 by exp")
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / "exp_target_v4_vs_legacy_r2.png", dpi=150)
            plt.close(fig)

        for metric, fname in (
            ("last_target_v4_macro_r2", "fold_target_v4_macro_r2.png"),
            ("last_legacy_answer_mean_r2", "fold_legacy_answer_fraction_r2.png"),
        ):
            fig, ax = plt.subplots(figsize=(10, 5))
            for (exp, proc), g in mismatch.groupby(["exp_name", "process_id"]):
                gg = g.sort_values("fold")
                ax.plot(gg["fold"], gg[metric], marker="o", label=f"{exp}/{proc}")
            ax.set_xlabel("fold")
            ax.set_ylabel(metric)
            ax.set_title(metric)
            ax.grid(True, alpha=0.3)
            if mismatch[["exp_name", "process_id"]].drop_duplicates().shape[0] <= 12:
                ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(plot_dir / fname, dpi=150)
            plt.close(fig)

    if not target_diag.empty:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(target_diag["true_std"], target_diag["pred_std"], alpha=0.7)
        maxv = pd.to_numeric(target_diag[["true_std", "pred_std"]].stack(), errors="coerce").max()
        if pd.notna(maxv) and maxv > 0:
            ax.plot([0, maxv], [0, maxv], "--", color="gray", linewidth=1)
        ax.set_xlabel("true_std")
        ax.set_ylabel("pred_std")
        ax.set_title("target true_std vs pred_std")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "target_true_std_vs_pred_std.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(target_diag["std_ratio"], target_diag["r2"], alpha=0.7)
        ax.set_xlabel("std_ratio")
        ax.set_ylabel("target R2")
        ax.set_title("target R2 vs std_ratio")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "target_r2_vs_std_ratio.png", dpi=150)
        plt.close(fig)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = [str(row.get(c, "")).replace("\n", " ") for c in df.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _reference_mapping_diagnostics() -> pd.DataFrame:
    ans = _read_csv(PROJECT_ROOT / "data/reference/v3/target_answer_edges.csv")
    edges = _read_csv(PROJECT_ROOT / "data/reference/v3/canonical_edges.csv")
    v4 = _read_csv(PROJECT_ROOT / "data/reference/v4/target_stream_targets.csv")
    if ans.empty or edges.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in ans.iterrows():
        pid_num = int(row["process_id"])
        edge_id = str(row["canonical_answer_edge_id"])
        match = edges[
            (pd.to_numeric(edges["process_id"], errors="coerce") == pid_num)
            & (edges["canonical_edge_id"].astype(str) == edge_id)
        ]
        dst = str(match.iloc[0]["dst_node"]) if not match.empty and "dst_node" in match.columns else ""
        v4_match = pd.DataFrame()
        if not v4.empty:
            task = str(row["task_name"])
            species = "H2" if task == "target_h2" else ("CO2" if task == "tailgas_co2" else "")
            stream_key = str(row.get("main_data_stream_key", ""))
            v4_match = v4[
                (pd.to_numeric(v4["process_id"], errors="coerce") == pid_num)
                & (v4["target_species"].astype(str) == species)
                & (v4["main_data_stream_key"].astype(str) == stream_key)
            ]
        rows.append(
            {
                "process_id": f"Process{pid_num}",
                "task_name": row.get("task_name", ""),
                "canonical_answer_edge_id": edge_id,
                "answer_dst_node": dst,
                "dst_is_v_output": dst == "V_OUTPUT",
                "target_column": row.get("target_column", ""),
                "main_data_stream_key": row.get("main_data_stream_key", ""),
                "v4_target_rows_matching_species_stream": int(len(v4_match)),
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    out_dir: Path,
    *,
    root: Path,
    inventory: pd.DataFrame,
    mismatch: pd.DataFrame,
    target_diag: pd.DataFrame,
    mapping_diag: pd.DataFrame,
) -> None:
    n_runs = len(mismatch)
    diag_counts = mismatch["diagnosis"].value_counts().to_dict() if "diagnosis" in mismatch.columns else {}
    target_counts = target_diag["diagnosis"].value_counts().to_dict() if "diagnosis" in target_diag.columns else {}
    v4_mean = (
        pd.to_numeric(mismatch["last_target_v4_macro_r2"], errors="coerce").mean()
        if "last_target_v4_macro_r2" in mismatch.columns
        else float("nan")
    )
    legacy_mean = (
        pd.to_numeric(mismatch["last_legacy_answer_mean_r2"], errors="coerce").mean()
        if "last_legacy_answer_mean_r2" in mismatch.columns
        else float("nan")
    )
    missing_v4 = int(mismatch["last_target_v4_macro_r2"].isna().sum()) if "last_target_v4_macro_r2" in mismatch else 0
    low_legacy_v4_ok = int((mismatch.get("diagnosis") == "legacy_low_v4_ok").sum()) if not mismatch.empty else 0
    v4_missing_mapping = (
        int((pd.to_numeric(mapping_diag.get("v4_target_rows_matching_species_stream"), errors="coerce") <= 0).sum())
        if not mapping_diag.empty and "v4_target_rows_matching_species_stream" in mapping_diag.columns
        else 0
    )
    formula_val = _read_csv(PROJECT_ROOT / "data/reference/v4/target_formula_validation.csv")
    special_cases = pd.DataFrame()
    if not formula_val.empty:
        special_cases = formula_val[
            (formula_val.get("formula_match_status", "").astype(str) != "exact_match")
            | (~formula_val.get("direct_target_exists", True).astype(bool))
        ].copy()
        special_cases = special_cases[
            special_cases.get("process_id", pd.Series(dtype=object)).astype(str).isin(["2", "5", "6"])
        ]
    special_preview = "No Process2/5/6 non-exact or synthetic/direct-missing target formula rows found."
    if not special_cases.empty:
        special_preview = _markdown_table(
            special_cases[
                [
                    c
                    for c in [
                        "process_id",
                        "target_id",
                        "target_feature_name",
                        "formula_type",
                        "direct_target_exists",
                        "formula_match_status",
                        "validation_note",
                    ]
                    if c in special_cases.columns
                ]
            ]
        )

    text = f"""# Edge-All Metric Root Cause Report

Generated by `scripts/diagnose_edge_all_metric_mismatch.py`

## Scope

- Root scanned: `{root}`
- Runs discovered: {n_runs}
- Inventory rows: {len(inventory)}
- Target v4 rows: {len(target_diag)}

## Main Finding

The terminal `mean_r2`/legacy value is the **legacy answer-edge fraction R²**
(`metric_answer_all_targets_mean_r2`), while the capacity-ablation main metric is
**target_v4_macro_r2**, the stream-derived target amount R² (`Mole_Flow * Frac_*`).
These are different metrics and can move differently.

## Aggregate Snapshot

- Mean last target_v4_macro_r2: {v4_mean if math.isfinite(float(v4_mean)) else float('nan'):.6f}
- Mean last legacy_answer_fraction_macro_r2: {legacy_mean if math.isfinite(float(legacy_mean)) else float('nan'):.6f}
- Runs missing target_v4 macro metric: {missing_v4}
- Runs diagnosed as `legacy_low_v4_ok`: {low_legacy_v4_ok}

## Diagnosis Counts

Run-level:

```json
{json.dumps(diag_counts, indent=2, ensure_ascii=False)}
```

Target-level:

```json
{json.dumps(target_counts, indent=2, ensure_ascii=False)}
```

## Metric Key Mapping

| Before / ambiguous | After / explicit |
|---|---|
| `val_r2` for edge_all | `val/target_v4_macro_r2` primary, legacy fallback only |
| `metric_answer_all_targets_mean_r2` | `legacy_answer_fraction_macro_r2` alias |
| `val_val_target_v4_macro_r2` | backward-compatible candidate for `val_target_v4_macro_r2` |
| terminal `mean_r2` | split into `target_v4_macro_r2` and `legacy_answer_fraction_macro_r2` |
| variance-sensitive R² interpretation | keep true R²; add `relative_accuracy_score` as auxiliary variance-insensitive score |

## Mapping Checks

- `target_metrics_v4.csv` is based on `data/reference/v4/target_stream_targets.csv`,
  which uses target amount formulas such as `Mole_Flow * Frac_H2` and optional scale factors.
- `data/reference/v3/target_answer_edges.csv` is the legacy answer-edge mapping used for
  answer-edge diagnostics.
- Answer-edge `dst_node == V_OUTPUT` failures: {0 if mapping_diag.empty else int((~mapping_diag['dst_is_v_output']).sum())}
- Legacy answer rows with no same species+stream row in v4 target table: {v4_missing_mapping}

### Process2/5/6 v4 validation notes

{special_preview}

## Interpretation

- If `legacy_answer_fraction_macro_r2` is low but `target_v4_macro_r2` is high, this is a
  metric display/selection issue, not evidence that target amount performance collapsed.
- If both are low and target rows show `pred_std << true_std`, suspect output collapse or underfit.
- If `true_std` is near zero, R² is unstable and should be interpreted with `r2_status`.
- If both are low without collapse, inspect target mapping and inverse scaling.
- `relative_accuracy_score = clip(1 - RMSE / mean(abs(true)), 0, 1)` is available as an auxiliary
  score when you want a positive magnitude-relative score that is less tied to target variance.
  It is not R² and should not replace `target_v4_macro_r2` for ranking unless explicitly chosen.

## Outputs

- `metric_key_inventory.csv`
- `metric_mismatch_summary.csv`
- `target_v4_variance_diagnostic.csv`
- `answer_edge_vs_target_v4_comparison.csv`
- `target_mapping_reference_diagnostic.csv`
- `plots/*.png`

## Re-run Commands

Run diagnostics:

```bash
python scripts/diagnose_edge_all_metric_mismatch.py \\
  --root outputs/process_kfold_capacity_ablation \\
  --out outputs/process_kfold_capacity_ablation/metric_diagnostics
```

Regenerate capacity ablation report:

```bash
python scripts/aggregate_capacity_ablation_results.py \\
  --root outputs/process_kfold_capacity_ablation \\
  --make-plots
```

## Retraining?

If target_v4 metrics are present in existing run outputs, retraining is not required for
metric separation or report regeneration. Retraining is only needed for runs where v4 artifacts
were never exported, or if diagnostics show both target_v4 and legacy metrics are genuinely low
due to model collapse/underfit.
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ROOT_CAUSE_REPORT.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose edge_all metric mismatch.")
    parser.add_argument("--root", default="outputs/process_kfold_capacity_ablation")
    parser.add_argument("--out", default="outputs/process_kfold_capacity_ablation/metric_diagnostics")
    args = parser.parse_args()

    root = (PROJECT_ROOT / args.root).resolve() if not Path(args.root).is_absolute() else Path(args.root)
    out_dir = (PROJECT_ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for rec in _discover_runs(root):
        run_dir: Path = rec["run_dir"]
        exp = str(rec["exp_name"])
        proc = str(rec["process_id"])
        fold = int(rec["fold"])
        metrics_epoch = _read_csv(run_dir / "metrics_per_epoch.csv")

        r2_cols = _r2_related_columns(metrics_epoch) if not metrics_epoch.empty else []
        tv_cols = _target_v4_columns(metrics_epoch) if not metrics_epoch.empty else []
        ans_cols = _answer_columns(metrics_epoch) if not metrics_epoch.empty else []
        legacy_col = v4_col = ""
        last_legacy = best_legacy = best_legacy_epoch = float("nan")
        last_v4 = best_v4 = best_v4_epoch = float("nan")

        if not metrics_epoch.empty:
            legacy_col, last_legacy, best_legacy, best_legacy_epoch = _last_best(metrics_epoch, LEGACY_R2_CANDIDATES)
            v4_col, last_v4, best_v4, best_v4_epoch = _last_best(metrics_epoch, TARGET_V4_R2_CANDIDATES)

        if not math.isfinite(last_v4):
            for rel in ("val/target_metrics_v4_summary.csv", "val/target_metrics_v4.csv", "test/target_metrics_v4_summary.csv", "test/target_metrics_v4.csv"):
                last_v4 = _macro_from_target_v4_csv(run_dir / rel)
                if math.isfinite(last_v4):
                    v4_col = rel
                    break
        if not math.isfinite(last_v4):
            key, val = _json_metric(run_dir, TARGET_V4_R2_CANDIDATES)
            if math.isfinite(val):
                v4_col, last_v4 = key, val

        if not math.isfinite(last_legacy):
            for rel in ("val/answer_edge_metrics.csv", "test/answer_edge_metrics.csv", "answer_edge_metrics.csv"):
                last_legacy = _macro_from_answer_edge_csv(run_dir / rel)
                if math.isfinite(last_legacy):
                    legacy_col = rel
                    break
        if not math.isfinite(last_legacy):
            key, val = _json_metric(run_dir, LEGACY_R2_CANDIDATES)
            if math.isfinite(val):
                legacy_col, last_legacy = key, val

        inventory_rows.append(
            {
                "run_dir": _rel(run_dir),
                "exp_name": exp,
                "process_id": proc,
                "fold": fold,
                "r2_related_columns": "|".join(r2_cols),
                "target_v4_columns": "|".join(tv_cols),
                "answer_all_targets_columns": "|".join(ans_cols),
                "monitor_inferred_column": v4_col or legacy_col,
                "best_epoch_by_legacy_r2": best_legacy_epoch,
                "best_epoch_by_target_v4_macro_r2": best_v4_epoch,
            }
        )

        run_target_diags: list[str] = []
        for split in ("val", "test"):
            tv = _read_csv(run_dir / split / "target_metrics_v4.csv")
            if tv.empty:
                continue
            for _, row in tv.iterrows():
                tid = str(row.get("target_id", ""))
                if tid.startswith("__ALL"):
                    continue
                diag = _target_diagnosis(row)
                run_target_diags.append(diag)
                target_rows.append(
                    {
                        "run_dir": _rel(run_dir),
                        "exp_name": exp,
                        "process_id": proc,
                        "fold": fold,
                        "split": split,
                        "target_id": tid,
                        "target_species": row.get("target_species", ""),
                        "mae": row.get("mae", float("nan")),
                        "rmse": row.get("rmse", float("nan")),
                        "r2": row.get("r2", float("nan")),
                        "relative_accuracy_score": row.get("relative_accuracy_score", float("nan")),
                        "relative_rmse": row.get("relative_rmse", float("nan")),
                        "r2_raw": row.get("r2_raw", float("nan")),
                        "r2_clamped": row.get("r2_clamped", row.get("r2", float("nan"))),
                        "sse": row.get("sse", float("nan")),
                        "sst": row.get("sst", float("nan")),
                        "sst_eff": row.get("sst_eff", float("nan")),
                        "true_mean": row.get("true_mean", float("nan")),
                        "true_std": row.get("true_std", float("nan")),
                        "pred_mean": row.get("pred_mean", float("nan")),
                        "pred_std": row.get("pred_std", float("nan")),
                        "std_ratio": row.get("std_ratio", float("nan")),
                        "status": row.get("status", row.get("r2_status", "")),
                        "diagnosis": diag,
                    }
                )

        gap = last_v4 - last_legacy if math.isfinite(last_v4) and math.isfinite(last_legacy) else float("nan")
        diag = _run_diagnosis(last_legacy, last_v4, v4_col, run_target_diags)
        mismatch_rows.append(
            {
                "run_dir": _rel(run_dir),
                "exp_name": exp,
                "process_id": proc,
                "fold": fold,
                "last_legacy_answer_mean_r2": last_legacy,
                "last_target_v4_macro_r2": last_v4,
                "best_legacy_answer_mean_r2": best_legacy,
                "best_target_v4_macro_r2": best_v4,
                "gap_between_legacy_and_v4": gap,
                "legacy_metric_source": legacy_col,
                "target_v4_metric_source": v4_col,
                "diagnosis": diag,
            }
        )
        comparison_rows.append(
            {
                "run_dir": _rel(run_dir),
                "exp_name": exp,
                "process_id": proc,
                "fold": fold,
                "legacy_answer_fraction_macro_r2": last_legacy,
                "target_v4_macro_r2": last_v4,
                "gap_target_v4_minus_legacy": gap,
                "interpretation": "metric_display_issue" if diag == "legacy_low_v4_ok" else diag,
            }
        )

    inventory = pd.DataFrame(inventory_rows)
    mismatch = pd.DataFrame(mismatch_rows)
    target_diag = pd.DataFrame(target_rows)
    comparison = pd.DataFrame(comparison_rows)
    mapping_diag = _reference_mapping_diagnostics()

    inventory.to_csv(out_dir / "metric_key_inventory.csv", index=False)
    mismatch.to_csv(out_dir / "metric_mismatch_summary.csv", index=False)
    target_diag.to_csv(out_dir / "target_v4_variance_diagnostic.csv", index=False)
    comparison.to_csv(out_dir / "answer_edge_vs_target_v4_comparison.csv", index=False)
    mapping_diag.to_csv(out_dir / "target_mapping_reference_diagnostic.csv", index=False)

    _write_plots(out_dir, mismatch, target_diag)
    _write_report(
        out_dir,
        root=root,
        inventory=inventory,
        mismatch=mismatch,
        target_diag=target_diag,
        mapping_diag=mapping_diag,
    )
    print(f"[diagnose] wrote diagnostics to {out_dir}")
    print(f"[diagnose] runs={len(mismatch)} target_rows={len(target_diag)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
