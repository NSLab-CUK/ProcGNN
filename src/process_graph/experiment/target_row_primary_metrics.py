"""Primary target-row metrics for edge_all (capacity/method/k-fold ablations)."""

from __future__ import annotations

import logging
import math
import warnings
from typing import Any, Mapping

import pandas as pd

from .target_metric_names import (
    AMOUNT_R2_SHORT_ALIAS,
    EVAL_FRAC_R2_MEAN_CO2,
    EVAL_FRAC_R2_MEAN_H2,
    EVAL_FRAC_R2_MEAN_H2O,
    EVAL_LEGACY_ANSWER_FRAC_R2,
    EVAL_LEGACY_ANSWER_FRAC_R2_MACRO,
    EVAL_PRIMARY_FRAC_R2_BY_PROCESS,
    EVAL_PRIMARY_FRAC_R2_BY_TARGET,
    EVAL_PRIMARY_FRAC_R2_FORMULA_BY_PROCESS,
    EVAL_PRIMARY_FRAC_R2_FORMULA_BY_TARGET,
    EVAL_SECONDARY_AMOUNT_R2_BY_PROCESS,
    EVAL_SECONDARY_AMOUNT_R2_BY_TARGET,
    EVAL_SECONDARY_AMOUNT_R2_FORMULA_BY_PROCESS,
    EVAL_SECONDARY_AMOUNT_R2_FORMULA_BY_TARGET,
    EVAL_TARGET_ROWS_EVALUATED,
    EVAL_TARGET_ROWS_SKIPPED,
    EVAL_TARGET_ROWS_TOTAL,
    FRAC_R2_SHORT_ALIAS,
    METRIC_SCHEMA,
    SUMMARY_AMOUNT_R2_BY_PROCESS,
    SUMMARY_AMOUNT_R2_BY_TARGET,
    SUMMARY_FRAC_R2_BY_PROCESS,
    SUMMARY_FRAC_R2_BY_TARGET,
    SUMMARY_FRAC_R2_FORMULA_BY_PROCESS,
    SUMMARY_FRAC_R2_FORMULA_BY_TARGET,
    SUMMARY_ROW_ID_FRAC_PROCESS,
    SUMMARY_ROW_ID_FRAC_TARGET,
    resolve_metric_key,
)

logger = logging.getLogger(__name__)

TARGETROW_METRIC_SCHEMA_VERSION = METRIC_SCHEMA
PRIMARY_METRIC_NAME = EVAL_PRIMARY_FRAC_R2_BY_PROCESS
TARGET_BALANCED_METRIC_NAME = EVAL_PRIMARY_FRAC_R2_BY_TARGET
PRIMARY_METRIC_FORMULA_AVAILABLE = EVAL_PRIMARY_FRAC_R2_FORMULA_BY_PROCESS
TARGET_BALANCED_METRIC_FORMULA_AVAILABLE = EVAL_PRIMARY_FRAC_R2_FORMULA_BY_TARGET
AMOUNT_SECONDARY_METRIC_NAME = EVAL_SECONDARY_AMOUNT_R2_BY_PROCESS
AMOUNT_TARGET_BALANCED_METRIC_NAME = EVAL_SECONDARY_AMOUNT_R2_BY_TARGET
LEGACY_ANSWER_FRACTION_METRIC = EVAL_LEGACY_ANSWER_FRAC_R2
FRAC_METRIC_ALIAS = FRAC_R2_SHORT_ALIAS
AMOUNT_METRIC_ALIAS = AMOUNT_R2_SHORT_ALIAS

DEPRECATED_MAIN_ALIASES = (
    "target_v4_macro_r2",
    "target_v4_macro_r2_main_verified",
    "process_balanced_main_target_frac_r2_main_verified",
    "answer_targets_r2",
    "metric_answer_all_targets_mean_r2",
)

SUMMARY_KIND_PROCESS_BALANCED_MAIN = SUMMARY_FRAC_R2_BY_PROCESS
SUMMARY_KIND_TARGET_BALANCED_MAIN = SUMMARY_FRAC_R2_BY_TARGET
SUMMARY_KIND_PROCESS_BALANCED_FORMULA = SUMMARY_FRAC_R2_FORMULA_BY_PROCESS
SUMMARY_KIND_TARGET_BALANCED_FORMULA = SUMMARY_FRAC_R2_FORMULA_BY_TARGET

PROCESS_BALANCED_SUMMARY_TARGET_ID = SUMMARY_ROW_ID_FRAC_PROCESS
TARGET_BALANCED_SUMMARY_TARGET_ID = SUMMARY_ROW_ID_FRAC_TARGET


def _truthy_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(("true", "1", "yes", "t"))


def _filter_main_verified(usable: pd.DataFrame) -> pd.DataFrame:
    if "include_in_main_verified_macro" in usable.columns:
        return usable[_truthy_series(usable["include_in_main_verified_macro"])]
    return usable


def _filter_formula_available(usable: pd.DataFrame) -> pd.DataFrame:
    if "include_in_formula_macro" in usable.columns:
        return usable[_truthy_series(usable["include_in_formula_macro"])]
    return usable


def _filter_included_in_primary(usable: pd.DataFrame) -> pd.DataFrame:
    if "included_in_primary" in usable.columns:
        return usable[_truthy_series(usable["included_in_primary"])]
    return _filter_main_verified(usable)


def _per_target_rows(metric_df: pd.DataFrame) -> pd.DataFrame:
    if metric_df.empty or "target_id" not in metric_df.columns:
        return metric_df.iloc[0:0]
    sub = metric_df[~metric_df["target_id"].astype(str).str.startswith("__ALL")].copy()
    if "r2" in sub.columns:
        sub = sub[pd.to_numeric(sub["r2"], errors="coerce").notna()]
    return sub


def compute_target_balanced_r2(
    metric_df: pd.DataFrame,
    *,
    policy: str = "main_verified",
    use_included_in_primary: bool = False,
) -> float:
    """Arithmetic mean of per-target_id R² (same as legacy split_macro_main_verified)."""
    sub = _per_target_rows(metric_df)
    if use_included_in_primary:
        sub = _filter_included_in_primary(sub)
    elif policy == "formula_available":
        sub = _filter_formula_available(sub)
    else:
        sub = _filter_main_verified(sub)
    if sub.empty:
        return float("nan")
    r2s = pd.to_numeric(sub["r2"], errors="coerce").dropna()
    return float(r2s.mean()) if not r2s.empty else float("nan")


def compute_process_balanced_r2(
    metric_df: pd.DataFrame,
    *,
    policy: str = "main_verified",
    use_included_in_primary: bool = False,
) -> float:
    """Mean of per-process target-row R² means (main metric for multi-process eval)."""
    sub = _per_target_rows(metric_df)
    if use_included_in_primary:
        sub = _filter_included_in_primary(sub)
    elif policy == "formula_available":
        sub = _filter_formula_available(sub)
    else:
        sub = _filter_main_verified(sub)
    if sub.empty or "process_id" not in sub.columns:
        return compute_target_balanced_r2(metric_df, policy=policy)
    proc_means: list[float] = []
    for _, g in sub.groupby("process_id", dropna=False):
        r2s = pd.to_numeric(g["r2"], errors="coerce").dropna()
        if not r2s.empty:
            proc_means.append(float(r2s.mean()))
    return float(sum(proc_means) / len(proc_means)) if proc_means else float("nan")


def build_balanced_macro_summary_rows(
    metric_df: pd.DataFrame,
    *,
    split: str,
    provenance_loaded: bool,
    use_included_in_primary: bool = False,
) -> list[dict[str, Any]]:
    """Append process- and target-balanced macro rows (primary ablation metrics)."""
    usable = _per_target_rows(metric_df)
    if usable.empty:
        return []

    rows: list[dict[str, Any]] = []

    def _row(kind: str, target_id: str, r2: float, n_targets: int, n_processes: int) -> dict[str, Any]:
        if kind in (SUMMARY_KIND_PROCESS_BALANCED_MAIN, SUMMARY_KIND_TARGET_BALANCED_MAIN):
            family = "primary_frac"
        elif kind in (SUMMARY_KIND_PROCESS_BALANCED_FORMULA, SUMMARY_KIND_TARGET_BALANCED_FORMULA):
            family = "primary_frac"
        elif "amount" in kind:
            family = "secondary_amount"
        else:
            family = "legacy"
        return {
            "split": split,
            "process_id": "__ALL__",
            "target_id": target_id,
            "r2": r2,
            "n_targets_present": n_targets,
            "n_processes": n_processes,
            "summary_kind": kind,
            "metric_family": family,
            "provenance_policy_loaded": provenance_loaded,
        }

    if use_included_in_primary:
        main_sub = _filter_included_in_primary(usable)
        form_sub = main_sub
    else:
        main_sub = _filter_main_verified(usable)
        form_sub = _filter_formula_available(usable)
    if provenance_loaded or use_included_in_primary:
        tb_main = compute_target_balanced_r2(
            metric_df, policy="main_verified", use_included_in_primary=use_included_in_primary
        )
        pb_main = compute_process_balanced_r2(
            metric_df, policy="main_verified", use_included_in_primary=use_included_in_primary
        )
        if math.isfinite(tb_main):
            rows.append(
                _row(
                    SUMMARY_KIND_TARGET_BALANCED_MAIN,
                    TARGET_BALANCED_SUMMARY_TARGET_ID,
                    tb_main,
                    int(main_sub["target_id"].nunique()),
                    int(main_sub["process_id"].nunique()) if "process_id" in main_sub.columns else 0,
                )
            )
        if math.isfinite(pb_main):
            rows.append(
                _row(
                    SUMMARY_KIND_PROCESS_BALANCED_MAIN,
                    PROCESS_BALANCED_SUMMARY_TARGET_ID,
                    pb_main,
                    int(main_sub["target_id"].nunique()),
                    int(main_sub["process_id"].nunique()) if "process_id" in main_sub.columns else 0,
                )
            )
        tb_form = compute_target_balanced_r2(
            metric_df, policy="formula_available", use_included_in_primary=use_included_in_primary
        )
        pb_form = compute_process_balanced_r2(
            metric_df, policy="formula_available", use_included_in_primary=use_included_in_primary
        )
        if math.isfinite(tb_form):
            rows.append(
                _row(
                    SUMMARY_KIND_TARGET_BALANCED_FORMULA,
                    "__ALL_target_balanced_formula_available__",
                    tb_form,
                    int(form_sub["target_id"].nunique()),
                    int(form_sub["process_id"].nunique()) if "process_id" in form_sub.columns else 0,
                )
            )
        if math.isfinite(pb_form):
            rows.append(
                _row(
                    SUMMARY_KIND_PROCESS_BALANCED_FORMULA,
                    "__ALL_process_balanced_formula_available__",
                    pb_form,
                    int(form_sub["target_id"].nunique()),
                    int(form_sub["process_id"].nunique()) if "process_id" in form_sub.columns else 0,
                )
            )
    else:
        warnings.warn(
            "v4 provenance not loaded; process_balanced_main_target_r2 uses all evaluated targets (deprecated).",
            stacklevel=2,
        )
        tb = compute_target_balanced_r2(metric_df, policy="main_verified")
        pb = compute_process_balanced_r2(metric_df, policy="main_verified")
        if math.isfinite(tb):
            rows.append(
                _row(
                    SUMMARY_KIND_TARGET_BALANCED_MAIN,
                    TARGET_BALANCED_SUMMARY_TARGET_ID,
                    tb,
                    int(usable["target_id"].nunique()),
                    int(usable["process_id"].nunique()) if "process_id" in usable.columns else 0,
                )
            )
        if math.isfinite(pb):
            rows.append(
                _row(
                    SUMMARY_KIND_PROCESS_BALANCED_MAIN,
                    PROCESS_BALANCED_SUMMARY_TARGET_ID,
                    pb,
                    int(usable["target_id"].nunique()),
                    int(usable["process_id"].nunique()) if "process_id" in usable.columns else 0,
                )
            )
    return rows


def _pick_summary_r2(summary_df: pd.DataFrame, kind: str, split: str = "") -> float:
    if summary_df.empty:
        return float("nan")
    sub = summary_df[summary_df.get("summary_kind", "").astype(str) == kind]
    if split and "split" in sub.columns:
        sub = sub[sub["split"].astype(str) == split]
    if sub.empty:
        return float("nan")
    return float(pd.to_numeric(sub.iloc[-1]["r2"], errors="coerce"))


def payload_from_balanced_macros(
    summary_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    *,
    split_name: str = "",
    legacy_answer_fraction_r2: float | None = None,
    use_included_in_primary: bool = False,
) -> dict[str, Any]:
    """Flat payload keys for metrics.json / val scalars (split-prefixed when split_name set)."""
    out: dict[str, Any] = {}
    sp = split_name or ""

    def _set(base: str, val: float) -> None:
        if not math.isfinite(val):
            return
        out[base] = val
        if sp:
            out[f"{sp}/{base}"] = val

    pb_main = _pick_summary_r2(summary_df, SUMMARY_KIND_PROCESS_BALANCED_MAIN, sp)
    tb_main = _pick_summary_r2(summary_df, SUMMARY_KIND_TARGET_BALANCED_MAIN, sp)
    if not math.isfinite(pb_main):
        pb_main = compute_process_balanced_r2(
            metric_df, use_included_in_primary=use_included_in_primary
        )
    if not math.isfinite(tb_main):
        tb_main = compute_target_balanced_r2(
            metric_df, use_included_in_primary=use_included_in_primary
        )

    _set(PRIMARY_METRIC_NAME, pb_main)
    _set(TARGET_BALANCED_METRIC_NAME, tb_main)
    _set(FRAC_METRIC_ALIAS, tb_main if math.isfinite(tb_main) else pb_main)
    _set(PRIMARY_METRIC_FORMULA_AVAILABLE, _pick_summary_r2(summary_df, SUMMARY_KIND_PROCESS_BALANCED_FORMULA, sp))
    _set(TARGET_BALANCED_METRIC_FORMULA_AVAILABLE, _pick_summary_r2(summary_df, SUMMARY_KIND_TARGET_BALANCED_FORMULA, sp))

    if math.isfinite(pb_main):
        _set(FRAC_R2_SHORT_ALIAS, pb_main)
    if legacy_answer_fraction_r2 is not None and math.isfinite(legacy_answer_fraction_r2):
        _set(LEGACY_ANSWER_FRACTION_METRIC, legacy_answer_fraction_r2)
        _set(EVAL_LEGACY_ANSWER_FRAC_R2_MACRO, legacy_answer_fraction_r2)

    excl = extract_target_exclusion_metadata(metric_df, use_included_in_primary=use_included_in_primary)
    out.update(excl)
    out["targetrow_metric_schema_version"] = TARGETROW_METRIC_SCHEMA_VERSION
    out["metric_schema_version"] = TARGETROW_METRIC_SCHEMA_VERSION
    out["primary_metric_name"] = PRIMARY_METRIC_NAME
    if sp:
        for k, v in list(excl.items()):
            if k.startswith("n_") or k.endswith("_target_ids") or k.endswith("_reasons") or k.startswith("n_"):
                out[f"{sp}/{k}"] = v
        out[f"{sp}/targetrow_metric_schema_version"] = TARGETROW_METRIC_SCHEMA_VERSION
        out[f"{sp}/primary_metric_name"] = PRIMARY_METRIC_NAME

    return out


def payload_from_amount_secondary_macros(
    summary_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    *,
    split_name: str = "",
) -> dict[str, Any]:
    """Amount (Mole_Flow * Frac) diagnostics — not used for early stopping after v2."""
    out: dict[str, Any] = {}
    sp = split_name or ""

    def _set(base: str, val: float) -> None:
        if not math.isfinite(val):
            return
        out[base] = val
        if sp:
            out[f"{sp}/{base}"] = val

    pb = _pick_summary_r2(summary_df, SUMMARY_KIND_PROCESS_BALANCED_MAIN, sp)
    tb = _pick_summary_r2(summary_df, SUMMARY_KIND_TARGET_BALANCED_MAIN, sp)
    if not math.isfinite(pb):
        pb = compute_process_balanced_r2(metric_df)
    if not math.isfinite(tb):
        tb = compute_target_balanced_r2(metric_df)
    _set(AMOUNT_SECONDARY_METRIC_NAME, pb)
    _set(AMOUNT_TARGET_BALANCED_METRIC_NAME, tb)
    _set(AMOUNT_METRIC_ALIAS, tb if math.isfinite(tb) else pb)
    if math.isfinite(pb):
        _set(AMOUNT_R2_SHORT_ALIAS, tb if math.isfinite(tb) else pb)
    return out


def extract_target_exclusion_metadata(
    metric_df: pd.DataFrame,
    *,
    use_included_in_primary: bool = False,
) -> dict[str, Any]:
    sub = _per_target_rows(metric_df)
    out: dict[str, Any] = {
        "n_targets_total": int(sub["target_id"].nunique()) if not sub.empty else 0,
        "n_main_verified_targets": 0,
        "n_included_targets": 0,
        "n_targets_formula_available": 0,
        "n_targets_excluded": 0,
        "n_excluded_targets": 0,
        "included_target_ids": "",
        "excluded_target_ids": "",
        "excluded_target_reasons": "",
        "targetrow_metric_schema_version": TARGETROW_METRIC_SCHEMA_VERSION,
        "metric_schema_version": TARGETROW_METRIC_SCHEMA_VERSION,
        "primary_metric_name": PRIMARY_METRIC_NAME,
    }
    if sub.empty:
        return out
    if use_included_in_primary and "included_in_primary" in sub.columns:
        ok = sub[_truthy_series(sub["included_in_primary"])]
        excl = sub[~sub["target_id"].isin(ok["target_id"])]
        out["n_main_verified_targets"] = int(ok["target_id"].nunique())
        out["n_included_targets"] = out["n_main_verified_targets"]
        out["n_targets_excluded"] = int(excl["target_id"].nunique())
        out["n_excluded_targets"] = out["n_targets_excluded"]
        out["included_target_ids"] = ";".join(sorted(ok["target_id"].astype(str).unique()))
        out["excluded_target_ids"] = ";".join(sorted(excl["target_id"].astype(str).unique()))
        if "skip_reason" in excl.columns:
            reasons = [
                f"{r['target_id']}:{r['skip_reason']}"
                for _, r in excl.iterrows()
                if pd.notna(r.get("skip_reason")) and str(r.get("skip_reason", "")).strip()
            ]
            if reasons:
                out["excluded_target_reasons"] = ";".join(reasons)
    elif "include_in_main_verified_macro" in sub.columns:
        ok = sub[_truthy_series(sub["include_in_main_verified_macro"])]
        excl = sub[~sub["target_id"].isin(ok["target_id"])]
        out["n_main_verified_targets"] = int(ok["target_id"].nunique())
        out["n_included_targets"] = out["n_main_verified_targets"]
        out["n_targets_excluded"] = int(excl["target_id"].nunique())
        out["n_excluded_targets"] = out["n_targets_excluded"]
        out["included_target_ids"] = ";".join(sorted(ok["target_id"].astype(str).unique()))
        out["excluded_target_ids"] = ";".join(sorted(excl["target_id"].astype(str).unique()))
        if "exclusion_reason" in excl.columns:
            reasons = [
                f"{r['target_id']}:{r['exclusion_reason']}"
                for _, r in excl.iterrows()
                if pd.notna(r.get("exclusion_reason"))
            ]
            out["excluded_target_reasons"] = ";".join(reasons)
    else:
        out["n_main_verified_targets"] = int(sub["target_id"].nunique())
        out["n_included_targets"] = out["n_main_verified_targets"]
        out["included_target_ids"] = ";".join(sorted(sub["target_id"].astype(str).unique()))
    if "include_in_formula_macro" in sub.columns:
        form = sub[_truthy_series(sub["include_in_formula_macro"])]
        out["n_targets_formula_available"] = int(form["target_id"].nunique())
    if "process_id" in sub.columns:
        counts = sub.groupby("process_id")["target_id"].nunique().to_dict()
        out["process_target_counts"] = ";".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda x: str(x[0])))
    if "target_species" in sub.columns:
        sc = sub.groupby("target_species")["target_id"].nunique().to_dict()
        out["target_count_by_species"] = ";".join(f"{k}:{v}" for k, v in sorted(sc.items(), key=lambda x: str(x[0])))
    return out


def resolve_monitor_epoch_log_key(
    monitor_metric: str,
    task_mode: str,
) -> tuple[str, str]:
    """Map train.monitor_metric to epoch_log key; edge_all prefers process-balanced primary."""
    metric = str(monitor_metric or "val_loss")
    r2_mode = "max"
    if task_mode != "edge_all":
        if metric == "val_r2":
            return "val/metric_target_r2", r2_mode
        if metric in ("val_loss",):
            return "val/loss", "min"
        return "val/loss", "min"

    explicit: dict[str, str] = {
        "val_target_edge_property_mean_r2": "val/target_edge_property_mean_r2",
        "target_edge_property_mean_r2": "val/target_edge_property_mean_r2",
        "val_target_stream_feature_macro_r2": "val/target_stream_feature_macro_r2",
        "target_stream_feature_macro_r2": "val/target_stream_feature_macro_r2",
        "val_target_edge_10d_r2_property_macro": "val/target_edge_10d_r2_property_macro",
        "target_edge_10d_r2_property_macro": "val/target_edge_10d_r2_property_macro",
        "val_target_edge_10d_r2_edge_macro": "val/target_edge_10d_r2_edge_macro",
        "target_edge_10d_r2_edge_macro": "val/target_edge_10d_r2_edge_macro",
        f"val_{PRIMARY_METRIC_NAME}": f"val/{PRIMARY_METRIC_NAME}",
        "val_eval_primary_frac_r2_by_process": f"val/{PRIMARY_METRIC_NAME}",
        PRIMARY_METRIC_NAME: f"val/{PRIMARY_METRIC_NAME}",
        f"val_{TARGET_BALANCED_METRIC_NAME}": f"val/{TARGET_BALANCED_METRIC_NAME}",
        "val_eval_primary_frac_r2_by_target": f"val/{TARGET_BALANCED_METRIC_NAME}",
        TARGET_BALANCED_METRIC_NAME: f"val/{TARGET_BALANCED_METRIC_NAME}",
        f"val_{FRAC_METRIC_ALIAS}": f"val/{FRAC_METRIC_ALIAS}",
        FRAC_METRIC_ALIAS: f"val/{FRAC_METRIC_ALIAS}",
        f"val_{AMOUNT_SECONDARY_METRIC_NAME}": f"val/{AMOUNT_SECONDARY_METRIC_NAME}",
        "val_eval_secondary_amount_r2_by_process": f"val/{AMOUNT_SECONDARY_METRIC_NAME}",
        AMOUNT_SECONDARY_METRIC_NAME: f"val/{AMOUNT_SECONDARY_METRIC_NAME}",
        f"val_{AMOUNT_METRIC_ALIAS}": f"val/{AMOUNT_METRIC_ALIAS}",
        AMOUNT_METRIC_ALIAS: f"val/{AMOUNT_METRIC_ALIAS}",
    }
    for old_key, canon in (
        ("val_process_balanced_main_target_frac_r2_main_verified", PRIMARY_METRIC_NAME),
        ("val_target_balanced_main_target_frac_r2_main_verified", TARGET_BALANCED_METRIC_NAME),
        ("val_target_v4_frac_main_verified_r2", FRAC_METRIC_ALIAS),
        ("val_process_balanced_main_target_r2_main_verified", AMOUNT_SECONDARY_METRIC_NAME),
        ("val_target_v4_main_verified_r2", AMOUNT_METRIC_ALIAS),
        ("val_target_v4_macro_r2_main_verified", AMOUNT_METRIC_ALIAS),
    ):
        explicit[old_key] = f"val/{canon}"
    if metric in explicit:
        return explicit[metric], r2_mode
    resolved = resolve_metric_key(metric)
    if resolved != metric and resolved in (
        PRIMARY_METRIC_NAME,
        TARGET_BALANCED_METRIC_NAME,
        FRAC_METRIC_ALIAS,
        AMOUNT_SECONDARY_METRIC_NAME,
        AMOUNT_METRIC_ALIAS,
    ):
        return f"val/{resolved}", r2_mode

    if metric == "val_r2":
        return f"val/{PRIMARY_METRIC_NAME}", r2_mode

    if metric == "val_mae":
        return "val/answer_targets_mae", "min"
    if metric == "val_rmse":
        return "val/target_h2_rmse", "min"
    return "val/answer_targets_mae", "min"


def pick_primary_r2_from_summary_csv(
    summary_df: pd.DataFrame,
    *,
    policy: str = "main_verified",
    balanced: str = "process",
) -> float:
    kind = SUMMARY_KIND_PROCESS_BALANCED_MAIN if balanced == "process" else SUMMARY_KIND_TARGET_BALANCED_MAIN
    if policy == "formula_available":
        kind = (
            SUMMARY_KIND_PROCESS_BALANCED_FORMULA
            if balanced == "process"
            else SUMMARY_KIND_TARGET_BALANCED_FORMULA
        )
    v = _pick_summary_r2(summary_df, kind)
    if math.isfinite(v):
        return v
    # Legacy fallbacks
    legacy_kind = "split_macro_main_verified" if policy == "main_verified" else "split_macro_formula_available"
    return _pick_summary_r2(summary_df, legacy_kind)


def warn_deprecated_main_metric_usage(context: str, column: str) -> None:
    logger.warning(
        "[%s] Deprecated main metric column %r; use %s instead.",
        context,
        column,
        PRIMARY_METRIC_NAME,
    )


def build_metric_summary_bundle_row(payload: Mapping[str, Any], summary_df: pd.DataFrame) -> dict[str, Any]:
    """One wide row appended to target_metrics_v4_summary.csv with bundle metadata + macro R²."""
    from .target_v4_provenance import payload_from_policy_macros

    sp = str(payload.get("split", "") or "")
    if not sp and not summary_df.empty and "split" in summary_df.columns:
        sp = str(summary_df["split"].iloc[0])

    def _r2(kind: str, fallback_key: str) -> float:
        v = _pick_summary_r2(summary_df, kind, sp)
        if math.isfinite(v):
            return v
        raw = payload.get(fallback_key, payload.get(f"test/{fallback_key}", float("nan")))
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float("nan")

    row: dict[str, Any] = {
        "split": sp,
        "process_id": "__ALL__",
        "target_id": "__METRIC_BUNDLE__",
        "summary_kind": "metric_bundle_metadata",
        "primary_metric_name": payload.get("primary_metric_name", PRIMARY_METRIC_NAME),
        "metric_schema_version": payload.get(
            "metric_schema_version", payload.get("targetrow_metric_schema_version", TARGETROW_METRIC_SCHEMA_VERSION)
        ),
        "targetrow_metric_schema_version": payload.get(
            "targetrow_metric_schema_version", TARGETROW_METRIC_SCHEMA_VERSION
        ),
        "included_target_ids": payload.get("included_target_ids", ""),
        "excluded_target_ids": payload.get("excluded_target_ids", ""),
        "excluded_target_reasons": payload.get("excluded_target_reasons", ""),
        "n_targets_main_verified": payload.get("n_main_verified_targets", payload.get("n_included_targets", 0)),
        "n_targets_formula_available": payload.get("n_targets_formula_available", 0),
        "n_targets_excluded": payload.get("n_targets_excluded", payload.get("n_excluded_targets", 0)),
        PRIMARY_METRIC_NAME: _r2(SUMMARY_KIND_PROCESS_BALANCED_MAIN, PRIMARY_METRIC_NAME),
        TARGET_BALANCED_METRIC_NAME: _r2(SUMMARY_KIND_TARGET_BALANCED_MAIN, TARGET_BALANCED_METRIC_NAME),
        FRAC_R2_SHORT_ALIAS: _r2(SUMMARY_KIND_TARGET_BALANCED_MAIN, FRAC_R2_SHORT_ALIAS),
        AMOUNT_SECONDARY_METRIC_NAME: float(
            payload.get(AMOUNT_SECONDARY_METRIC_NAME, payload.get(f"test/{AMOUNT_SECONDARY_METRIC_NAME}", float("nan")))
            or float("nan")
        ),
        AMOUNT_TARGET_BALANCED_METRIC_NAME: float(
            payload.get(AMOUNT_TARGET_BALANCED_METRIC_NAME, float("nan")) or float("nan")
        ),
        AMOUNT_R2_SHORT_ALIAS: float(payload.get(AMOUNT_METRIC_ALIAS, float("nan")) or float("nan")),
        PRIMARY_METRIC_FORMULA_AVAILABLE: _r2(
            SUMMARY_KIND_PROCESS_BALANCED_FORMULA, PRIMARY_METRIC_FORMULA_AVAILABLE
        ),
        TARGET_BALANCED_METRIC_FORMULA_AVAILABLE: _r2(
            SUMMARY_KIND_TARGET_BALANCED_FORMULA, TARGET_BALANCED_METRIC_FORMULA_AVAILABLE
        ),
        LEGACY_ANSWER_FRACTION_METRIC: float(
            payload.get(
                LEGACY_ANSWER_FRACTION_METRIC,
                payload.get(EVAL_LEGACY_ANSWER_FRAC_R2_MACRO, float("nan")),
            )
            or float("nan")
        ),
    }
    return row


def provenance_fields_for_epoch_log(val_metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Fields for metrics_per_epoch.csv (strings allowed)."""
    row: dict[str, Any] = {
        "val_targetrow_metric_schema_version": TARGETROW_METRIC_SCHEMA_VERSION,
        "val_primary_metric_name": PRIMARY_METRIC_NAME,
    }
    for src, dst in (
        ("included_target_ids", "val_included_target_ids"),
        ("excluded_target_ids", "val_excluded_target_ids"),
        ("excluded_target_reasons", "val_excluded_target_reasons"),
        ("val_n_main_verified_targets", "val_n_included_targets"),
        ("val_n_targets_excluded", "val_n_excluded_targets"),
        ("n_main_verified_targets", "val_n_included_targets"),
        ("n_targets_excluded", "val_n_excluded_targets"),
    ):
        if src in val_metrics and val_metrics[src] not in (None, ""):
            row[dst] = val_metrics[src]
    n_inc = val_metrics.get("val_n_main_verified_targets", val_metrics.get("n_main_verified_targets"))
    n_exc = val_metrics.get("val_n_targets_excluded", val_metrics.get("n_targets_excluded"))
    if n_inc is not None:
        try:
            ni = int(n_inc)
            row["val_n_included_targets"] = ni
            row["val_n_main_verified_targets"] = ni
        except (TypeError, ValueError):
            pass
    if n_exc is not None:
        try:
            row["val_n_excluded_targets"] = int(n_exc)
        except (TypeError, ValueError):
            pass
    return row


def epoch_csv_column_names() -> dict[str, str]:
    """val_* column names written to metrics_per_epoch.csv."""
    return {
        "primary_frac_by_process": f"val_{PRIMARY_METRIC_NAME}",
        "primary_frac_by_target": f"val_{TARGET_BALANCED_METRIC_NAME}",
        "frac_alias": f"val_{FRAC_METRIC_ALIAS}",
        "secondary_amount_by_process": f"val_{AMOUNT_SECONDARY_METRIC_NAME}",
        "secondary_amount_by_target": f"val_{AMOUNT_TARGET_BALANCED_METRIC_NAME}",
        "amount_alias": f"val_{AMOUNT_METRIC_ALIAS}",
        "primary_frac_formula_by_process": f"val_{PRIMARY_METRIC_FORMULA_AVAILABLE}",
        "primary_frac_formula_by_target": f"val_{TARGET_BALANCED_METRIC_FORMULA_AVAILABLE}",
        "legacy_answer_frac": f"val_{LEGACY_ANSWER_FRACTION_METRIC}",
        "n_main_verified": "val_n_main_verified_targets",
        "excluded_target_ids": "val_excluded_target_ids",
    }


EPOCH_LOG_SCALAR_BASES: tuple[str, ...] = (
    PRIMARY_METRIC_NAME,
    TARGET_BALANCED_METRIC_NAME,
    FRAC_METRIC_ALIAS,
    PRIMARY_METRIC_FORMULA_AVAILABLE,
    TARGET_BALANCED_METRIC_FORMULA_AVAILABLE,
    AMOUNT_SECONDARY_METRIC_NAME,
    AMOUNT_TARGET_BALANCED_METRIC_NAME,
    AMOUNT_METRIC_ALIAS,
    EVAL_SECONDARY_AMOUNT_R2_FORMULA_BY_PROCESS,
    EVAL_SECONDARY_AMOUNT_R2_FORMULA_BY_TARGET,
    "eval_amount_r2_all_target_rows",
    EVAL_LEGACY_ANSWER_FRAC_R2,
    EVAL_LEGACY_ANSWER_FRAC_R2_MACRO,
    EVAL_FRAC_R2_MEAN_H2,
    EVAL_FRAC_R2_MEAN_CO2,
    EVAL_FRAC_R2_MEAN_H2O,
    EVAL_TARGET_ROWS_EVALUATED,
    EVAL_TARGET_ROWS_SKIPPED,
    EVAL_TARGET_ROWS_TOTAL,
    "eval_target_row_relative_accuracy_mean",
    "eval_target_row_mae_mean",
    "eval_target_row_rmse_mean",
    "eval_target_row_r2_macro",
)


def _lookup_val_scalar(val_metrics: Mapping[str, Any], base: str) -> float:
    from .target_metric_names import CANONICAL_TO_DEPRECATED, pick_from_mapping

    keys: list[str] = [base, f"val/{base}", f"val_{base}"]
    for old in CANONICAL_TO_DEPRECATED.get(base, ()):
        keys.extend([old, f"val/{old}", f"val_{old}"])
    if base == EVAL_LEGACY_ANSWER_FRAC_R2:
        keys.extend(["legacy_answer_fraction_r2", "val_legacy_answer_fraction_r2", "metric_answer_all_targets_mean_r2"])
    if base == EVAL_LEGACY_ANSWER_FRAC_R2_MACRO:
        keys.extend(["legacy_answer_fraction_macro_r2", "val_legacy_answer_fraction_macro_r2"])
    return float(pick_from_mapping(val_metrics, *keys, default=float("nan")))


def write_edge_all_epoch_metric_columns(row: dict[str, Any], val_metrics: Mapping[str, Any]) -> None:
    """Copy readable eval_* scalars into metrics_per_epoch row (val_* columns)."""
    for base in EPOCH_LOG_SCALAR_BASES:
        v = _lookup_val_scalar(val_metrics, base)
        if math.isfinite(v):
            row[f"val_{base}"] = v
    row.update(provenance_fields_for_epoch_log(val_metrics))
