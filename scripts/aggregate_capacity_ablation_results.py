#!/usr/bin/env python3
"""Aggregate capacity-ablation K-fold results and generate comparison / convergence plots."""
from __future__ import annotations

import argparse
import json
import math
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RUN_DIR_RE = re.compile(r"^process_kfold_P(\d+)_F(\d+)(?:-\d{8}-\d{6})?$")
TARGET_REF_PATH = PROJECT_ROOT / "data" / "reference" / "v4" / "target_stream_targets.csv"

# Convergence diagnosis thresholds (reference-only; see README).
SLOPE_EPS = 1e-4
OVERFIT_GAP_THRESHOLD = 0.1
MIN_EPOCHS_FOR_DIAGNOSIS = 20
STABLE_R2_SLOPE_THRESH = 1e-4
STABLE_LOSS_SLOPE_THRESH = 1e-5
UNSTABLE_R2_STD_THRESH = 0.05

PRIMARY_R2_COLUMNS = [
    "val_process_balanced_main_target_r2_main_verified",
    "val_val_process_balanced_main_target_r2_main_verified",
    "val/process_balanced_main_target_r2_main_verified",
    "process_balanced_main_target_r2_main_verified",
    "test_process_balanced_main_target_r2_main_verified",
]
TARGET_V4_MACRO_R2_COLUMNS = PRIMARY_R2_COLUMNS + [
    "val_target_balanced_main_target_r2_main_verified",
    "val_target_v4_macro_r2",
    "val_val_target_v4_macro_r2",
    "val/target_v4_macro_r2",
    "target_v4_macro_r2",
]
LEGACY_ANSWER_FRACTION_R2_COLUMNS = [
    "val_legacy_answer_fraction_macro_r2",
    "val/legacy_answer_fraction_macro_r2",
    "val_metric_answer_all_targets_mean_r2",
    "val/metric_answer_all_targets_mean_r2",
    "metric_answer_all_targets_mean_r2",
]

LAYER_SWEEP = {"exp_L3": 3, "exp_L4": 4, "exp_L5": 5, "exp_L6": 6}
HIDDEN_SWEEP = {"exp_H384": 384, "exp_H448": 448, "exp_H512": 512}
ATTN_SWEEP = {"exp_A256": 256, "exp_A384": 384, "exp_H512": 512}
EMBED_SWEEP = ["exp_E_small", "exp_E_base", "exp_E_large"]
MLP_SWEEP = ["exp_M_base", "exp_M_shallow", "exp_M_update_deep", "exp_M_input_deep"]


def _safe_read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _parse_run_dir(path: Path) -> tuple[int, int] | None:
    m = _RUN_DIR_RE.match(path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _pick_test_metrics(run_dir: Path) -> dict:
    tj = _safe_read_json(run_dir / "test" / "metrics.json")
    if isinstance(tj, dict) and tj:
        return tj
    mj = _safe_read_json(run_dir / "metrics.json")
    if not isinstance(mj, dict):
        return {}
    return {k[5:]: v for k, v in mj.items() if str(k).startswith("test/")}


def _macro_mean_target_v4_r2(
    run_dir: Path,
    *,
    flag_col: str | None = "include_in_main_verified_macro",
    summary_target_id: str = "__ALL_macro_main_verified__",
    summary_kind: str = "split_macro_main_verified",
) -> float:
    for rel in ("test/target_metrics_v4_summary.csv", "target_metrics_v4_summary.csv"):
        p = run_dir / rel
        if not p.is_file():
            continue
        try:
            summ = pd.read_csv(p)
        except (OSError, pd.errors.EmptyDataError):
            continue
        if summ.empty or "target_id" not in summ.columns:
            continue
        macro = summ[summ["target_id"].astype(str) == summary_target_id]
        if macro.empty and "summary_kind" in summ.columns:
            macro = summ[summ["summary_kind"].astype(str) == summary_kind]
        if macro.empty and summary_target_id == "__ALL_macro_main_verified__":
            macro = summ[summ["target_id"].astype(str) == "__ALL_macro_split__"]
            if macro.empty and "summary_kind" in summ.columns:
                macro = summ[summ["summary_kind"].astype(str) == "split_macro"]
        if not macro.empty and "r2" in macro.columns:
            v = pd.to_numeric(macro.iloc[0]["r2"], errors="coerce")
            if pd.notna(v):
                return float(v)
    df = _read_target_v4_table(run_dir)
    if df is None or df.empty or "r2" not in df.columns:
        return float("nan")
    sub = df[~df["target_id"].astype(str).str.startswith("__ALL")] if "target_id" in df.columns else df
    if flag_col and flag_col in sub.columns:
        sub = sub[sub[flag_col].astype(str).str.lower().isin(("true", "1", "yes"))]
    elif "status" in sub.columns and flag_col is None:
        sub = sub[sub["status"].astype(str) == "ok"]
    r2 = pd.to_numeric(sub.get("r2"), errors="coerce").dropna()
    return float(r2.mean()) if not r2.empty else float("nan")


def _provenance_counts(run_dir: Path) -> dict:
    df = _read_target_v4_table(run_dir)
    if df is None or df.empty or "target_id" not in df.columns:
        return {}
    sub = df[~df["target_id"].astype(str).str.startswith("__ALL")]
    n_total = int(sub["target_id"].nunique())
    out: dict = {"n_targets_total": n_total}
    if "include_in_main_verified_macro" in sub.columns:
        ok = sub[sub["include_in_main_verified_macro"].astype(str).str.lower().isin(("true", "1", "yes"))]
        out["n_targets_main_verified"] = int(ok["target_id"].nunique())
        excl = sub[~sub["target_id"].isin(ok["target_id"])]
        out["n_targets_excluded"] = int(excl["target_id"].nunique())
        out["excluded_target_ids"] = ";".join(sorted(excl["target_id"].astype(str).unique()))
    if "include_in_formula_macro" in sub.columns:
        form = sub[sub["include_in_formula_macro"].astype(str).str.lower().isin(("true", "1", "yes"))]
        out["n_targets_formula_available"] = int(form["target_id"].nunique())
    return out


def _metric_from_main_targets(run_dir: Path) -> dict:
    mj = _safe_read_json(run_dir / "metrics_main_targets.json")
    if not isinstance(mj, dict):
        return {}
    out: dict = {}
    for k in ("target_v4_macro_r2", "target_h2_r2", "tailgas_co2_r2", "target_v4_macro_rmse"):
        if k in mj:
            out[k] = mj[k]
        if f"test/{k}" in mj:
            out[k] = mj[f"test/{k}"]
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


def _load_target_reference() -> pd.DataFrame:
    if not TARGET_REF_PATH.is_file():
        return pd.DataFrame()
    try:
        ref = pd.read_csv(TARGET_REF_PATH)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()
    if ref.empty:
        return ref
    ref = ref.copy()
    ref["process_id"] = "Process" + pd.to_numeric(ref["process_id"], errors="coerce").astype("Int64").astype(str)
    return ref


def _target_rows_for_run(rec: dict, target_ref: pd.DataFrame) -> pd.DataFrame:
    run_dir: Path = rec["run_dir"]
    df = _read_target_v4_table(run_dir)
    if df is None or df.empty or "target_id" not in df.columns:
        return pd.DataFrame()
    out = df[~df["target_id"].astype(str).str.startswith("__ALL")].copy()
    if out.empty:
        return out
    out.insert(0, "run_dir", str(run_dir.relative_to(PROJECT_ROOT)) if run_dir.is_relative_to(PROJECT_ROOT) else str(run_dir))
    out.insert(0, "fold_id", int(rec["fold_id"]))
    out.insert(0, "experiment_name", str(rec["experiment_name"]))
    if not target_ref.empty:
        meta_cols = [
            "process_id",
            "target_id",
            "target_stream_node",
            "target_formula",
            "formula_type",
            "canonical_answer_edge_id",
            "main_data_stream_key",
            "source_note",
            "mapping_note",
        ]
        keep = [c for c in meta_cols if c in target_ref.columns]
        out = out.merge(target_ref[keep], on=["process_id", "target_id"], how="left", suffixes=("", "_ref"))
    return out


def _discover_runs(root: Path) -> list[dict]:
    rows: list[dict] = []
    if not root.is_dir():
        return rows
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == "summary":
            continue
        exp_name = exp_dir.name
        for proc_dir in sorted(exp_dir.glob("Process*")):
            if not proc_dir.is_dir():
                continue
            process_id = proc_dir.name
            for fold_dir in sorted(proc_dir.glob("fold_*")):
                if not fold_dir.is_dir():
                    continue
                fold_m = re.match(r"fold_(\d+)", fold_dir.name)
                fold_id = int(fold_m.group(1)) if fold_m else 0
                for child in sorted(fold_dir.iterdir()):
                    if not child.is_dir():
                        continue
                    parsed = _parse_run_dir(child)
                    if not parsed:
                        continue
                    _, fold_from_name = parsed
                    fold_id = fold_from_name
                    rows.append(
                        {
                            "experiment_name": exp_name,
                            "process_id": process_id,
                            "fold_id": fold_id,
                            "fold_dir": fold_dir,
                            "run_dir": child,
                        }
                    )
    return rows


def _extract_fold_row(rec: dict) -> dict:
    from ablation_targetrow_common import (
        LEGACY_ANSWER_FRACTION_METRIC,
        PRIMARY_METRIC_NAME,
        TARGET_BALANCED_METRIC_NAME,
        best_epoch_by_primary,
        extract_primary_metrics,
    )

    run_dir: Path = rec["run_dir"]
    tm = _pick_test_metrics(run_dir)
    mj = _safe_read_json(run_dir / "metrics.json") or {}
    mt = _metric_from_main_targets(run_dir)
    status = "ok" if tm or mj or mt else "missing_metrics"
    primary = extract_primary_metrics(run_dir)
    best_epoch, best_val_primary = best_epoch_by_primary(run_dir)
    tv_macro_formula = _macro_mean_target_v4_r2(
        run_dir,
        flag_col="include_in_formula_macro",
        summary_target_id="__ALL_macro_formula_available__",
        summary_kind="split_macro_formula_available",
    )
    tv_macro_all = _macro_mean_target_v4_r2(
        run_dir,
        flag_col=None,
        summary_target_id="__ALL_macro_split__",
        summary_kind="split_macro",
    )
    h2 = tm.get("metric_target_r2", mj.get("test/metric_target_r2", mt.get("target_h2_r2", np.nan)))
    co2 = tm.get("metric_tailgas_r2", mj.get("test/metric_tailgas_r2", mt.get("tailgas_co2_r2", np.nan)))
    legacy_macro = primary.get(f"test_{LEGACY_ANSWER_FRACTION_METRIC}", np.nan)
    if pd.isna(legacy_macro):
        legacy_macro = primary.get(f"val_{LEGACY_ANSWER_FRACTION_METRIC}", np.nan)
    edge_r2 = tm.get("edge_all_r2_orig", tm.get("r2", mj.get("test/edge_all_r2_orig", np.nan)))
    tv_rmse = mj.get("test/target_v4_rmse", mt.get("target_v4_macro_rmse", np.nan))
    train_time = mj.get("train_time_sec", mj.get("elapsed_sec", np.nan))
    h2o = tm.get("test/target_v4_h2o_mean_r2", mt.get("target_v4_h2o_mean_r2", np.nan))
    h2_f = float(h2) if pd.notna(h2) else np.nan
    co2_f = float(co2) if pd.notna(co2) else np.nan
    answer_vals = [v for v in (h2_f, co2_f) if math.isfinite(v)]
    answer_targets_r2 = float(sum(answer_vals) / len(answer_vals)) if answer_vals else np.nan
    best_val_answer_targets_r2 = _best_val_answer_targets_r2(run_dir)
    test_primary = primary.get(f"test_{PRIMARY_METRIC_NAME}", np.nan)
    val_tb = primary.get(f"val_{TARGET_BALANCED_METRIC_NAME}", np.nan)
    test_tb = primary.get(f"test_{TARGET_BALANCED_METRIC_NAME}", np.nan)
    val_legacy = primary.get(f"val_{LEGACY_ANSWER_FRACTION_METRIC}", np.nan)
    schema_ver = primary.get("targetrow_metric_schema_version", "target_row_primary_v1")
    return {
        "exp_name": rec["experiment_name"],
        "experiment_name": rec["experiment_name"],
        "process_id": rec["process_id"],
        "fold": rec["fold_id"],
        "fold_id": rec["fold_id"],
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)) if run_dir.is_relative_to(PROJECT_ROOT) else str(run_dir),
        "metric_schema_version": schema_ver,
        "targetrow_metric_schema_version": schema_ver,
        "primary_metric_name": PRIMARY_METRIC_NAME,
        "best_epoch_by_primary_metric": best_epoch,
        "process_balanced_main_target_r2_main_verified": test_primary,
        "target_balanced_main_target_r2_main_verified": test_tb,
        "legacy_answer_fraction_r2": legacy_macro,
        "included_target_ids": primary.get("included_target_ids", ""),
        "excluded_target_ids": primary.get("excluded_target_ids", ""),
        "excluded_target_reasons": primary.get("excluded_target_reasons", ""),
        "n_targets_main_verified": primary.get("n_targets_main_verified", np.nan),
        "n_targets_excluded": primary.get("n_targets_excluded", np.nan),
        f"val_{PRIMARY_METRIC_NAME}": primary.get(f"val_{PRIMARY_METRIC_NAME}", best_val_primary),
        f"test_{PRIMARY_METRIC_NAME}": test_primary,
        f"val_{TARGET_BALANCED_METRIC_NAME}": val_tb,
        f"test_{TARGET_BALANCED_METRIC_NAME}": test_tb,
        f"val_{LEGACY_ANSWER_FRACTION_METRIC}": val_legacy,
        f"test_{LEGACY_ANSWER_FRACTION_METRIC}": legacy_macro,
        "target_v4_macro_r2": test_primary,
        "target_v4_macro_r2_main_verified": primary.get(f"test_{TARGET_BALANCED_METRIC_NAME}", np.nan),
        "target_v4_macro_r2_formula_available": tv_macro_formula,
        "target_v4_macro_r2_all_reported": tv_macro_all,
        **{k: v for k, v in primary.items() if k.startswith("n_") or k.endswith("_target_ids") or k.endswith("_reasons") or k.endswith("_counts") or k.endswith("_by_species")},
        "target_h2_r2": h2,
        "tailgas_co2_r2": co2,
        "answer_targets_r2": answer_targets_r2,
        "best_val_answer_targets_r2": best_val_answer_targets_r2,
        "legacy_answer_fraction_macro_r2": legacy_macro,
        "target_h2o_r2": h2o,
        "edge_all_r2": edge_r2,
        "target_v4_macro_rmse": tv_rmse,
        "train_time_sec": train_time,
        "status": status,
    }


def _load_epoch_table(run_dir: Path) -> pd.DataFrame | None:
    csv_path = run_dir / "metrics_per_epoch.csv"
    if csv_path.is_file():
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                return df
        except (OSError, pd.errors.EmptyDataError):
            pass
    hist = _safe_read_json(run_dir / "train_val_history.json")
    if not isinstance(hist, dict):
        return None
    train = hist.get("train") or {}
    val = hist.get("val") or {}
    epochs = train.get("epoch") or []
    if not epochs:
        return None
    rows = []
    val_by_ep = {int(v): i for i, v in enumerate(val.get("epoch") or [])}
    for i, ep in enumerate(epochs):
        row = {"epoch": int(ep), "train_loss": float(train.get("loss_total", [np.nan] * len(epochs))[i])}
        if "train_lr" in train:
            row["train_lr"] = float(train["train_lr"][i])
        vi = val_by_ep.get(int(ep))
        if vi is not None:
            row["val_loss"] = float(val.get("loss_total", [np.nan] * len(val.get("epoch", [])))[vi])
            for k, arr in val.items():
                if k == "epoch":
                    continue
                row[f"val_{k}"] = float(arr[vi]) if vi < len(arr) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _best_val_answer_targets_r2(run_dir: Path) -> float:
    """Optuna-compatible objective: best validation answer-target fraction R² over epochs."""
    df = _load_epoch_table(run_dir)
    if df is None or df.empty:
        return float("nan")
    col = _col_first(
        df,
        [
            "val_answer_targets_r2_weighted",
            "val_answer_targets_r2_mean",
            "val_legacy_answer_fraction_macro_r2",
            "val_metric_answer_all_targets_mean_r2",
            "val/metric_answer_all_targets_mean_r2",
        ],
    )
    if not col:
        return float("nan")
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.max()) if not vals.empty else float("nan")


def _col_first(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def _series_numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)


def _slope_last_n(y: np.ndarray, n: int = 10) -> float:
    y = y[np.isfinite(y)]
    if len(y) < 2:
        return float("nan")
    k = min(n, len(y))
    x = np.arange(k, dtype=float)
    return float(np.polyfit(x, y[-k:], 1)[0])


def _diagnose_convergence(df: pd.DataFrame) -> dict:
    if df is None or df.empty or "epoch" not in df.columns:
        return {
            "n_epochs": 0,
            "best_epoch": np.nan,
            "final_epoch": np.nan,
            "best_val_target_v4_macro_r2": np.nan,
            "final_val_target_v4_macro_r2": np.nan,
            "final_minus_best_r2": np.nan,
            "best_val_loss": np.nan,
            "final_val_loss": np.nan,
            "final_minus_best_loss": np.nan,
            "train_loss_final": np.nan,
            "val_loss_final": np.nan,
            "generalization_gap_loss": np.nan,
            "r2_slope_last_10_epochs": np.nan,
            "loss_slope_last_10_epochs": np.nan,
            "converged_status": "insufficient_data",
            "warning_reason": "no epoch metrics",
        }
    r2_col = _col_first(
        df,
        TARGET_V4_MACRO_R2_COLUMNS + LEGACY_ANSWER_FRACTION_R2_COLUMNS + ["val_metric_target_r2"],
    )
    loss_col = _col_first(df, ["val_loss", "val_loss_total"])
    if loss_col is None:
        loss_col = _col_first(df, [c for c in df.columns if c.startswith("val_loss")])
    train_loss_col = _col_first(df, ["train_loss", "loss_total"])

    epochs = _series_numeric(df, "epoch")
    n_epochs = int(np.nanmax(epochs)) if len(epochs) else 0
    final_epoch = float(epochs[-1]) if len(epochs) else np.nan

    r2 = _series_numeric(df, r2_col) if r2_col else np.full(len(df), np.nan)
    val_loss = _series_numeric(df, loss_col) if loss_col else np.full(len(df), np.nan)
    train_loss = _series_numeric(df, train_loss_col) if train_loss_col else np.full(len(df), np.nan)

    finite_r2 = np.isfinite(r2)
    best_idx = int(np.nanargmax(r2[finite_r2])) if finite_r2.any() else -1
    if best_idx >= 0:
        r2_finite_idx = np.where(finite_r2)[0][best_idx]
        best_val_r2 = float(r2[r2_finite_idx])
        best_epoch = float(epochs[r2_finite_idx])
    else:
        best_val_r2 = float("nan")
        best_epoch = float("nan")

    final_val_r2 = float(r2[-1]) if len(r2) else float("nan")
    final_minus_best_r2 = (
        final_val_r2 - best_val_r2 if math.isfinite(final_val_r2) and math.isfinite(best_val_r2) else float("nan")
    )

    finite_vl = np.isfinite(val_loss)
    if finite_vl.any():
        vl_idx = int(np.nanargmin(val_loss[finite_vl]))
        vi = np.where(finite_vl)[0][vl_idx]
        best_val_loss = float(val_loss[vi])
        best_loss_epoch = float(epochs[vi])
    else:
        best_val_loss = float("nan")
        best_loss_epoch = float("nan")

    final_val_loss = float(val_loss[-1]) if len(val_loss) else float("nan")
    final_minus_best_loss = (
        final_val_loss - best_val_loss
        if math.isfinite(final_val_loss) and math.isfinite(best_val_loss)
        else float("nan")
    )
    train_final = float(train_loss[-1]) if len(train_loss) else float("nan")
    val_final = final_val_loss
    gen_gap = (
        val_final - train_final if math.isfinite(val_final) and math.isfinite(train_final) else float("nan")
    )

    r2_slope = _slope_last_n(r2, 10)
    loss_slope = _slope_last_n(val_loss, 10)

    status = "insufficient_data"
    warning = ""
    if n_epochs >= MIN_EPOCHS_FOR_DIAGNOSIS and finite_r2.sum() >= MIN_EPOCHS_FOR_DIAGNOSIS:
        r2_tail = r2[np.isfinite(r2)][-10:]
        loss_tail = val_loss[np.isfinite(val_loss)][-10:]
        r2_std = float(np.std(r2_tail)) if len(r2_tail) > 1 else 0.0
        if r2_std > UNSTABLE_R2_STD_THRESH:
            status = "unstable"
            warning = f"high val R2 std in last 10 epochs ({r2_std:.4f})"
        elif r2_slope > STABLE_R2_SLOPE_THRESH and loss_slope <= STABLE_LOSS_SLOPE_THRESH:
            status = "still_improving"
            warning = "val_target_v4_macro_r2 still rising in last 10 epochs"
        elif (
            math.isfinite(gen_gap)
            and gen_gap > OVERFIT_GAP_THRESHOLD
            and loss_slope > SLOPE_EPS
            and r2_slope < -STABLE_R2_SLOPE_THRESH
        ):
            status = "overfitting"
            warning = "val loss rising / val R2 falling while train improves"
        elif abs(r2_slope) <= STABLE_R2_SLOPE_THRESH and abs(loss_slope) <= STABLE_LOSS_SLOPE_THRESH:
            status = "converged"
        else:
            status = "converged"
            warning = "mixed signals; review plots"
    elif n_epochs > 0:
        warning = f"fewer than {MIN_EPOCHS_FOR_DIAGNOSIS} epochs with val R2"

    return {
        "n_epochs": n_epochs,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "best_val_target_v4_macro_r2": best_val_r2,
        "final_val_target_v4_macro_r2": final_val_r2,
        "final_minus_best_r2": final_minus_best_r2,
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "final_minus_best_loss": final_minus_best_loss,
        "train_loss_final": train_final,
        "val_loss_final": val_final,
        "generalization_gap_loss": gen_gap,
        "r2_slope_last_10_epochs": r2_slope,
        "loss_slope_last_10_epochs": loss_slope,
        "converged_status": status,
        "warning_reason": warning,
        "r2_metric_used": r2_col or "",
        "best_loss_epoch": best_loss_epoch,
    }


class PlotContext:
    def __init__(self, *, make_plots: bool, plot_format: str, dpi: int, summary_dir: Path):
        self.make_plots = make_plots
        self.plot_format = plot_format.lstrip(".")
        self.dpi = dpi
        self.summary_dir = summary_dir
        self.plot_errors: list[dict] = []
        self._plt = None

    def _matplotlib(self):
        if self._plt is None:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            self._plt = plt
        return self._plt

    def record_error(self, *, plot_id: str, path: str, error: str) -> None:
        self.plot_errors.append({"plot_id": plot_id, "path": path, "error": error})

    def savefig(self, fig, path: Path, *, plot_id: str) -> None:
        if not self.make_plots:
            plt = self._matplotlib()
            plt.close(fig)
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        except Exception as exc:
            self.record_error(plot_id=plot_id, path=str(path), error=str(exc))
        finally:
            self._matplotlib().close(fig)


def _plot_fold_convergence(
    ctx: PlotContext,
    *,
    experiment_name: str,
    process_id: str,
    fold_id: int,
    run_dir: Path,
    fold_plots_dir: Path,
) -> None:
    df = _load_epoch_table(run_dir)
    if df is None or df.empty:
        ctx.record_error(
            plot_id="fold_convergence",
            path=str(fold_plots_dir),
            error=f"no metrics_per_epoch or train_val_history for {run_dir}",
        )
        return
    plt = ctx._matplotlib()
    ext = ctx.plot_format
    title_suffix = f"{experiment_name} | {process_id} | fold_{fold_id:02d}"

    train_col = _col_first(df, ["train_loss"])
    val_loss_col = _col_first(df, ["val_loss", "val_loss_total"])
    if train_col:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(df["epoch"], df[train_col], label="train_loss")
        if val_loss_col:
            ax.plot(df["epoch"], df[val_loss_col], "o-", label=val_loss_col, markersize=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(f"convergence_loss — {title_suffix}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ctx.savefig(fig, fold_plots_dir / f"convergence_loss.{ext}", plot_id="convergence_loss")

    r2_col = _col_first(
        df,
        TARGET_V4_MACRO_R2_COLUMNS + LEGACY_ANSWER_FRACTION_R2_COLUMNS + ["val_metric_target_r2"],
    )
    if r2_col:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(df["epoch"], df[r2_col], "o-", label=r2_col, markersize=3)
        if r2_col != "val_metric_target_r2" and "val_metric_tailgas_r2" in df.columns:
            ax.plot(df["epoch"], df["val_metric_tailgas_r2"], "s-", label="val_metric_tailgas_r2", markersize=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel("R²")
        ax.set_title(f"convergence_r2 — {title_suffix}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ctx.savefig(fig, fold_plots_dir / f"convergence_r2.{ext}", plot_id="convergence_r2")

    mae_col = _col_first(df, ["val_val_target_v4_macro_mae", "val_target_v4_macro_mae"])
    rmse_col = _col_first(df, ["val_val_target_v4_macro_rmse", "val_target_v4_macro_rmse"])
    if mae_col or rmse_col:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if mae_col:
            ax.plot(df["epoch"], df[mae_col], "o-", label=mae_col, markersize=3)
        if rmse_col:
            ax.plot(df["epoch"], df[rmse_col], "s-", label=rmse_col, markersize=3)
        if not mae_col and not rmse_col:
            edge_mae = _col_first(df, ["val_edge_all_mae"])
            if edge_mae:
                ax.plot(df["epoch"], df[edge_mae], label=edge_mae)
        ax.set_xlabel("epoch")
        ax.set_ylabel("error")
        ax.set_title(f"convergence_mae_rmse — {title_suffix}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ctx.savefig(fig, fold_plots_dir / f"convergence_mae_rmse.{ext}", plot_id="convergence_mae_rmse")

    lr_col = _col_first(df, ["train_lr", "val_train_lr"])
    grad_col = _col_first(df, [c for c in df.columns if "grad" in c.lower()])
    if lr_col or grad_col:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if lr_col:
            ax.plot(df["epoch"], df[lr_col], label=lr_col)
        if grad_col:
            ax2 = ax.twinx()
            ax2.plot(df["epoch"], df[grad_col], "r--", label=grad_col)
            ax2.set_ylabel("grad norm")
            ax2.legend(loc="upper right")
        ax.set_xlabel("epoch")
        ax.set_title(f"learning_diagnostics — {title_suffix}")
        ax.set_ylabel("learning rate")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        ctx.savefig(fig, fold_plots_dir / f"learning_diagnostics.{ext}", plot_id="learning_diagnostics")


def _align_fold_curves(dfs: list[pd.DataFrame], col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return epochs grid, mean, std across folds (pad with nan)."""
    if not dfs or col is None:
        return np.array([]), np.array([]), np.array([])
    max_ep = max(int(df["epoch"].max()) for df in dfs if "epoch" in df.columns)
    grid = np.arange(1, max_ep + 1)
    stack = []
    for df in dfs:
        ep = _series_numeric(df, "epoch").astype(int)
        y = _series_numeric(df, col)
        mapped = np.full(len(grid), np.nan)
        for e, v in zip(ep, y):
            if 1 <= e <= len(grid) and math.isfinite(v):
                mapped[e - 1] = v
        stack.append(mapped)
    arr = np.vstack(stack)
    return grid, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _plot_process_summary(
    ctx: PlotContext,
    *,
    experiment_name: str,
    process_id: str,
    fold_dfs: list[tuple[int, pd.DataFrame]],
    proc_plots_dir: Path,
    r2_fallback_note: str,
) -> None:
    if not fold_dfs:
        return
    plt = ctx._matplotlib()
    ext = ctx.plot_format
    dfs = [df for _, df in fold_dfs]
    title = f"{experiment_name} | {process_id}"

    train_col = "train_loss"
    val_col = _col_first(dfs[0], ["val_loss", "val_loss_total"]) if dfs else None
    if train_col in dfs[0].columns:
        g, mean_t, std_t = _align_fold_curves(dfs, train_col)
        g2, mean_v, std_v = _align_fold_curves(dfs, val_col) if val_col else (np.array([]), np.array([]), np.array([]))
        if len(g):
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(g, mean_t, label=f"train_loss mean (n={len(dfs)})")
            ax.fill_between(g, mean_t - std_t, mean_t + std_t, alpha=0.2)
            if len(g2):
                ax.plot(g2, mean_v, label=f"{val_col} mean")
                ax.fill_between(g2, mean_v - std_v, mean_v + std_v, alpha=0.2)
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.set_title(f"convergence_loss_mean_std — {title}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ctx.savefig(fig, proc_plots_dir / f"convergence_loss_mean_std.{ext}", plot_id="process_loss_mean_std")

    r2_col = _col_first(
        dfs[0],
        TARGET_V4_MACRO_R2_COLUMNS + LEGACY_ANSWER_FRACTION_R2_COLUMNS,
    )
    if r2_col:
        g, mean_r, std_r = _align_fold_curves(dfs, r2_col)
        if len(g):
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(g, mean_r, label=r2_col)
            ax.fill_between(g, mean_r - std_r, mean_r + std_r, alpha=0.2)
            ax.set_xlabel("epoch")
            ax.set_ylabel("R²")
            subtitle = r2_fallback_note or ""
            ax.set_title(f"convergence_target_v4_r2_mean_std — {title}\n{subtitle}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ctx.savefig(
                fig,
                proc_plots_dir / f"convergence_target_v4_r2_mean_std.{ext}",
                plot_id="process_v4_r2_mean_std",
            )

    # combined species plot
    fig, ax = plt.subplots(figsize=(9, 5))
    any_line = False
    for species, col_candidates in (
        ("H2", ["val_val_target_v4_h2_mean_r2", "val_target_v4_h2_mean_r2"]),
        ("CO2", ["val_val_target_v4_co2_mean_r2", "val_target_v4_co2_mean_r2"]),
        ("H2O", ["val_val_target_v4_h2o_mean_r2", "val_target_v4_h2o_mean_r2"]),
    ):
        col = _col_first(dfs[0], col_candidates)
        if not col:
            continue
        g, mean_r, _ = _align_fold_curves(dfs, col)
        if len(g):
            ax.plot(g, mean_r, label=f"{species} ({col})")
            any_line = True
    if any_line:
        ax.set_xlabel("epoch")
        ax.set_ylabel("mean R² across folds")
        ax.set_title(f"convergence_h2_co2_h2o_r2_mean — {title}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ctx.savefig(
            fig,
            proc_plots_dir / f"convergence_h2_co2_h2o_r2_mean.{ext}",
            plot_id="process_species_r2_mean",
        )
    else:
        plt.close(fig)


def _plot_experiment_comparisons(
    ctx: PlotContext,
    exp_summary: pd.DataFrame,
    proc_summary: pd.DataFrame,
) -> None:
    if not ctx.make_plots or exp_summary.empty:
        return
    plt = ctx._matplotlib()
    ext = ctx.plot_format
    plot_dir = ctx.summary_dir / "plots"

    rank_col = "mean_test_process_balanced_main_target_r2_main_verified"
    if rank_col not in exp_summary.columns:
        rank_col = "mean_target_v4_macro_r2"
    df = exp_summary.sort_values(rank_col, ascending=False, na_position="last")
    if rank_col in df.columns:
        fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.45), 5))
        x = np.arange(len(df))
        y = pd.to_numeric(df[rank_col], errors="coerce")
        std_col = rank_col.replace("mean_", "std_")
        yerr = pd.to_numeric(df.get(std_col), errors="coerce")
        ax.bar(x, y, yerr=yerr, capsize=3, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(df["experiment_name"], rotation=45, ha="right")
        ax.set_ylabel(rank_col)
        ax.set_title("experiment_rank_primary_targetrow_r2")
        ax.grid(True, axis="y", alpha=0.3)
        ctx.savefig(fig, plot_dir / f"experiment_rank_target_v4_macro_r2.{ext}", plot_id="exp_rank_v4")

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.5), 5))
    x = np.arange(len(df))
    w = 0.25
    for i, (col, label) in enumerate(
        (
            ("mean_h2_r2", "H2"),
            ("mean_co2_r2", "CO2"),
            ("mean_h2o_r2", "H2O"),
        )
    ):
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        ax.bar(x + (i - 1) * w, vals, width=w, label=label, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df["experiment_name"], rotation=45, ha="right")
    ax.set_ylabel("mean R²")
    ax.set_title("experiment_rank_h2_co2_h2o_r2")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ctx.savefig(fig, plot_dir / f"experiment_rank_h2_co2_h2o_r2.{ext}", plot_id="exp_rank_species")

    if not proc_summary.empty and "mean_target_v4_macro_r2" in proc_summary.columns:
        pivot = proc_summary.pivot_table(
            index="process_id",
            columns="experiment_name",
            values="mean_target_v4_macro_r2",
            aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 0.5), max(5, pivot.shape[0] * 0.4)))
        im = ax.imshow(pivot.values.astype(float), aspect="auto", cmap="viridis")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        ax.set_title("process_by_experiment_target_v4_r2_heatmap")
        fig.colorbar(im, ax=ax, label="mean_target_v4_macro_r2")
        ctx.savefig(fig, plot_dir / f"process_by_experiment_target_v4_r2_heatmap.{ext}", plot_id="heatmap")

    def _sweep_plot(sweep_map: dict, xlabel: str, fname: str, plot_id: str) -> None:
        sub = df[df["experiment_name"].isin(sweep_map.keys())].copy()
        if sub.empty:
            return
        sub["x"] = sub["experiment_name"].map(sweep_map)
        sub = sub.sort_values("x")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(
            sub["x"],
            pd.to_numeric(sub["mean_target_v4_macro_r2"], errors="coerce"),
            yerr=pd.to_numeric(sub.get("std_target_v4_macro_r2"), errors="coerce"),
            fmt="o-",
            capsize=4,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("mean_target_v4_macro_r2")
        ax.set_title(fname.replace("_", " "))
        ax.grid(True, alpha=0.3)
        ctx.savefig(fig, plot_dir / f"{fname}.{ext}", plot_id=plot_id)

    _sweep_plot(LAYER_SWEEP, "num_layers", "layer_sweep_target_v4_r2", "layer_sweep")
    _sweep_plot(HIDDEN_SWEEP, "hidden_dim", "hidden_sweep_target_v4_r2", "hidden_sweep")
    _sweep_plot(ATTN_SWEEP, "attn_hidden_dim", "attention_sweep_target_v4_r2", "attn_sweep")

    for sweep_list, fname in ((EMBED_SWEEP, "embedding_dim_comparison_target_v4_r2"), (MLP_SWEEP, "mlp_depth_comparison_target_v4_r2")):
        sub = df[df["experiment_name"].isin(sweep_list)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(7, len(sub) * 0.6), 4.5))
        ax.bar(sub["experiment_name"], pd.to_numeric(sub["mean_target_v4_macro_r2"], errors="coerce"))
        ax.set_ylabel("mean_target_v4_macro_r2")
        ax.set_title(fname.replace("_", " "))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, axis="y", alpha=0.3)
        ctx.savefig(fig, plot_dir / f"{fname}.{ext}", plot_id=fname)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate capacity ablation K-fold outputs.")
    parser.add_argument("--root", type=str, default="outputs/process_kfold_capacity_ablation")
    parser.add_argument("--make-plots", dest="make_plots", action="store_true", default=True)
    parser.add_argument("--no-plots", dest="make_plots", action="store_false")
    parser.add_argument("--plot-format", type=str, default="png", choices=["png", "pdf"])
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    root = (PROJECT_ROOT / args.root).resolve()
    summary_dir = root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    ctx = PlotContext(make_plots=args.make_plots, plot_format=args.plot_format, dpi=args.dpi, summary_dir=summary_dir)

    runs = _discover_runs(root)
    fold_rows = [_extract_fold_row(r) for r in runs]
    fold_df = pd.DataFrame(fold_rows)
    if not fold_df.empty:
        fold_df = fold_df.sort_values(["experiment_name", "process_id", "fold_id"])

    missing: list[dict] = []
    diag_rows: list[dict] = []
    target_ref = _load_target_reference()
    target_parts: list[pd.DataFrame] = []

    # Per-fold / per-process plots
    by_exp_proc: dict[tuple[str, str], list[tuple[int, pd.DataFrame, Path]]] = {}
    for rec in runs:
        run_dir: Path = rec["run_dir"]
        exp = rec["experiment_name"]
        proc = rec["process_id"]
        fold_id = rec["fold_id"]
        fold_plots = rec["fold_dir"] / "plots"
        if fold_df.empty:
            continue
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
            ctx.record_error(plot_id="fold_convergence_bundle", path=str(fold_plots), error=traceback.format_exc())
            missing.append(
                {
                    "experiment_name": exp,
                    "process_id": proc,
                    "fold_id": fold_id,
                    "reason": f"plot_error: {exc}",
                }
            )
        if epoch_df is not None and not epoch_df.empty:
            by_exp_proc.setdefault((exp, proc), []).append((fold_id, epoch_df, run_dir))
        target_part = _target_rows_for_run(rec, target_ref)
        if not target_part.empty:
            target_parts.append(target_part)

    for (exp, proc), items in by_exp_proc.items():
        proc_plots = root / exp / proc / "plots"
        r2_note = ""
        sample_df = items[0][1]
        if _col_first(sample_df, TARGET_V4_MACRO_R2_COLUMNS) is None:
            r2_note = "fallback: legacy answer fraction R2 (auxiliary, not main target amount R2)"
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

    # run_meta failures
    for exp_dir in sorted(root.glob("*/Process*")):
        meta_path = exp_dir / "run_meta.json"
        meta = _safe_read_json(meta_path)
        if isinstance(meta, dict) and meta.get("status") == "failed":
            missing.append(
                {
                    "experiment_name": exp_dir.parent.name,
                    "process_id": exp_dir.name,
                    "fold_id": "",
                    "run_dir": str(meta_path),
                    "reason": f"run_meta failed rc={meta.get('return_code')}",
                }
            )

    fold_df.to_csv(summary_dir / "experiment_process_summary.csv", index=False)
    fold_df.to_csv(summary_dir / "capacity_ablation_summary_targetrow.csv", index=False)
    fold_df.to_csv(root / "capacity_ablation_summary_targetrow.csv", index=False)

    target_long = pd.concat(target_parts, ignore_index=True) if target_parts else pd.DataFrame()
    if not target_long.empty:
        target_long.to_csv(summary_dir / "main_target_metrics_by_target.csv", index=False)
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
            if c in target_long.columns
        ]
        value_cols = [c for c in ("r2", "mae", "rmse", "relative_accuracy_score", "n_samples") if c in target_long.columns]
        agg_spec = {}
        for c in value_cols:
            if c == "n_samples":
                agg_spec[c] = "sum"
            else:
                agg_spec[c] = ["mean", "std", "min", "max"]
        summary = target_long.groupby(group_cols, dropna=False).agg(agg_spec).reset_index()
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
        pd.DataFrame().to_csv(summary_dir / "main_target_metrics_by_target.csv", index=False)
        pd.DataFrame().to_csv(summary_dir / "main_target_metrics_by_target_summary.csv", index=False)
        pd.DataFrame().to_csv(summary_dir / "main_target_r2_wide.csv", index=False)

    exp_summary_rows: list[dict] = []
    if not fold_df.empty:
        ok = fold_df[fold_df["status"] == "ok"]
        for exp, g in fold_df.groupby("experiment_name"):
            g_ok = ok[ok["experiment_name"] == exp] if not ok.empty else g.iloc[0:0]
            exp_summary_rows.append(
                {
                    "experiment_name": exp,
                    "n_processes_done": int(g_ok["process_id"].nunique()) if not g_ok.empty else 0,
                    "n_folds_done": int(len(g_ok)),
                    "mean_test_process_balanced_main_target_r2_main_verified": float(
                        pd.to_numeric(
                            g_ok.get("test_process_balanced_main_target_r2_main_verified", g_ok.get("target_v4_macro_r2")),
                            errors="coerce",
                        ).mean()
                    )
                    if not g_ok.empty
                    else np.nan,
                    "std_test_process_balanced_main_target_r2_main_verified": float(
                        pd.to_numeric(
                            g_ok.get("test_process_balanced_main_target_r2_main_verified", g_ok.get("target_v4_macro_r2")),
                            errors="coerce",
                        ).std(ddof=0)
                    )
                    if len(g_ok) > 1
                    else 0.0,
                    "mean_target_v4_macro_r2": float(
                        pd.to_numeric(g_ok.get("target_v4_macro_r2"), errors="coerce").mean()
                    )
                    if not g_ok.empty and "target_v4_macro_r2" in g_ok.columns
                    else np.nan,
                    "mean_h2_r2": float(pd.to_numeric(g_ok["target_h2_r2"], errors="coerce").mean())
                    if not g_ok.empty
                    else np.nan,
                    "mean_co2_r2": float(pd.to_numeric(g_ok["tailgas_co2_r2"], errors="coerce").mean())
                    if not g_ok.empty
                    else np.nan,
                    "mean_answer_targets_r2": float(
                        pd.to_numeric(g_ok["answer_targets_r2"], errors="coerce").mean()
                    )
                    if "answer_targets_r2" in g_ok.columns and not g_ok.empty
                    else np.nan,
                    "mean_best_val_answer_targets_r2": float(
                        pd.to_numeric(g_ok.get("best_val_answer_targets_r2"), errors="coerce").mean()
                    )
                    if "best_val_answer_targets_r2" in g_ok.columns and not g_ok.empty
                    else np.nan,
                    "mean_legacy_answer_fraction_macro_r2": float(
                        pd.to_numeric(g_ok.get("legacy_answer_fraction_macro_r2"), errors="coerce").mean()
                    )
                    if "legacy_answer_fraction_macro_r2" in g_ok.columns and not g_ok.empty
                    else np.nan,
                    "mean_h2o_r2": float(pd.to_numeric(g_ok.get("target_h2o_r2"), errors="coerce").mean())
                    if "target_h2o_r2" in g_ok.columns and not g_ok.empty
                    else np.nan,
                    "mean_edge_all_r2": float(pd.to_numeric(g_ok["edge_all_r2"], errors="coerce").mean())
                    if not g_ok.empty
                    else np.nan,
                    "failed_runs": int((g["status"] != "ok").sum()),
                }
            )
    exp_summary = pd.DataFrame(exp_summary_rows)
    exp_summary.to_csv(summary_dir / "experiment_summary.csv", index=False)

    proc_summary_rows: list[dict] = []
    if not fold_df.empty:
        ok = fold_df[fold_df["status"] == "ok"]
        for (exp, proc), g in ok.groupby(["experiment_name", "process_id"]):
            r2 = pd.to_numeric(
                g.get("test_process_balanced_main_target_r2_main_verified", g.get("target_v4_macro_r2")),
                errors="coerce",
            ).dropna()
            answer_r2 = pd.to_numeric(g.get("answer_targets_r2"), errors="coerce").dropna()
            best_val_answer_r2 = pd.to_numeric(g.get("best_val_answer_targets_r2"), errors="coerce").dropna()
            proc_summary_rows.append(
                {
                    "experiment_name": exp,
                    "process_id": proc,
                    "n_folds": int(len(g)),
                    "mean_test_process_balanced_main_target_r2_main_verified": float(r2.mean()) if not r2.empty else np.nan,
                    "std_test_process_balanced_main_target_r2_main_verified": float(r2.std(ddof=0)) if len(r2) > 1 else 0.0,
                    "mean_target_v4_macro_r2": float(r2.mean()) if not r2.empty else np.nan,
                    "mean_answer_targets_r2": float(answer_r2.mean()) if not answer_r2.empty else np.nan,
                    "mean_best_val_answer_targets_r2": float(best_val_answer_r2.mean())
                    if not best_val_answer_r2.empty
                    else np.nan,
                }
            )
    proc_summary = pd.DataFrame(proc_summary_rows)
    proc_summary.to_csv(summary_dir / "process_by_experiment_summary.csv", index=False)

    if not exp_summary.empty:
        rank_col = "mean_test_process_balanced_main_target_r2_main_verified"
        if rank_col not in exp_summary.columns:
            rank_col = "mean_target_v4_macro_r2"
        rank = exp_summary.sort_values(rank_col, ascending=False, na_position="last")
        rank.to_csv(summary_dir / "experiment_rank_by_primary_targetrow_r2.csv", index=False)
        rank.to_csv(summary_dir / "experiment_rank_by_target_v4_r2.csv", index=False)
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

    pd.DataFrame(missing).drop_duplicates().to_csv(summary_dir / "missing_or_failed_runs.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(summary_dir / "convergence_diagnostics.csv", index=False)
    pd.DataFrame(ctx.plot_errors).to_csv(summary_dir / "plot_errors.csv", index=False)

    try:
        _plot_experiment_comparisons(ctx, exp_summary, proc_summary)
    except Exception as exc:
        ctx.record_error(plot_id="experiment_comparisons", path=str(summary_dir / "plots"), error=traceback.format_exc())
        pd.DataFrame(ctx.plot_errors).to_csv(summary_dir / "plot_errors.csv", index=False)

    report = f"""# Capacity ablation report (target-row primary metrics)

Aggregated from: `{root}`

## Primary metric

- **process_balanced_main_target_r2_main_verified**: for each process, mean per-target_id amount R² (main-verified rows only), then mean across processes. This is the **main** ranking metric for capacity ablation.
- **target_balanced_main_target_r2_main_verified** (auxiliary): arithmetic mean over all main-verified target rows (not process-balanced).
- **legacy_answer_fraction_r2** (auxiliary): Frac_* on legacy H2/CO2 answer slots — comparable to historical Optuna `answer_targets_r2` (~0.9076); **not** comparable to the primary metric.
- **target_v4_macro_r2** / **target_v4_macro_r2_main_verified**: deprecated aliases; do not use for new comparisons.
- Same species with different stream or `target_feature` remain **separate target_id rows** (e.g. Process1 CO2×2).
- Targets with mapping_conflict / mismatch / unresolved provenance are **excluded** from the primary metric; see `excluded_target_ids` and `excluded_target_reasons`.

## Optuna vs capacity ablation

- Optuna best ~0.9076 = **legacy_answer_fraction_r2** (H2+CO2 weighted Frac R²).
- Capacity ablation primary = **process_balanced_main_target_r2_main_verified**.
- Do not compare these numbers directly; use `val_legacy_answer_fraction_r2` / `test_legacy_answer_fraction_r2` for legacy comparison.

## Skip-existing / stale metrics

If runs were completed before this metric schema, use `scripts/recompute_capacity_ablation_target_row_metrics.py` or re-run with a new `--output-root`. Rows without primary columns may be `target_row_metric_recompute_needed` in the audit report.

## Summary tables

| File | Description |
|------|-------------|
| `experiment_process_summary.csv` | Per experiment × process × fold metrics |
| `experiment_summary.csv` | Per experiment aggregates |
| `capacity_ablation_summary_targetrow.csv` | Fold-level summary (primary + auxiliary columns) |
| `experiment_rank_by_primary_targetrow_r2.csv` | Experiments sorted by mean **process_balanced** test R² |
| `experiment_rank_by_target_v4_r2.csv` | Deprecated alias of primary rank |
| `experiment_rank_by_answer_targets_r2.csv` | Experiments sorted by Optuna-compatible answer target R² |
| `experiment_rank_by_best_val_answer_targets_r2.csv` | Experiments sorted by best validation answer target R² |
| `main_target_metrics_by_target.csv` | Per fold × process × target_id v4 metrics |
| `main_target_metrics_by_target_summary.csv` | Per experiment × process × target_id mean/std metrics |
| `main_target_r2_wide.csv` | Wide table of per-target R² |
| `convergence_diagnostics.csv` | Automated convergence hints (reference only) |
| `missing_or_failed_runs.csv` | Missing metrics or failed jobs |
| `plot_errors.csv` | Plot generation failures |

## Plots

### Training (existing, under each run directory)

`train_process_surrogate.py` writes to `<run_dir>/plots/`:

- `loss_total_train_val.png`
- `loss_train_components.png`, `loss_val_components.png`
- `val_r2_target_h2_tailgas_co2.png`, `val_mae_answer_edges.png`, `val_edge_all_mae_mse_norm.png`

### Capacity ablation (this aggregator)

- Fold: `<EXP>/Process<P>/fold_<K>/plots/convergence_*.png`
- Process: `<EXP>/Process<P>/plots/convergence_*_mean_std.png`
- Summary: `summary/plots/experiment_rank_*.png`, sweep plots, heatmap

## Convergence thresholds (reference)

- `SLOPE_EPS={SLOPE_EPS}`, `OVERFIT_GAP_THRESHOLD={OVERFIT_GAP_THRESHOLD}`, `MIN_EPOCHS_FOR_DIAGNOSIS={MIN_EPOCHS_FOR_DIAGNOSIS}`
- `still_improving` → consider increasing `max_epochs` beyond 100
- `overfitting` → review dropout, weight decay, early stopping, residual settings

## learning_diagnostics

Written only when `train_lr` or grad norm columns exist in `metrics_per_epoch.csv`; otherwise see README (not available).
"""
    (summary_dir / "README_capacity_ablation_report.md").write_text(report, encoding="utf-8")
    (root / "CAPACITY_ABLATION_TARGETROW_REPORT.md").write_text(report, encoding="utf-8")
    print(f"[done] summary written to {summary_dir}")


if __name__ == "__main__":
    main()
