"""V4 stream-derived target metrics (all target_id per process) for edge_all evaluation.

Uses original-scale edge properties only (y_true_orig / y_pred_orig from edge_predictions).
Legacy H2/CO2 answer-edge metrics remain separate in edge_all_reporting / train_utils.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Iterable, Mapping, Sequence

import pandas as pd

from ..constants import STREAM_EDGE_FEATURE_SLOTS
from ..data.stream_keys import canonicalize_stream_key
from .target_row_primary_metrics import (
    AMOUNT_METRIC_ALIAS,
    AMOUNT_SECONDARY_METRIC_NAME,
    AMOUNT_TARGET_BALANCED_METRIC_NAME,
    FRAC_METRIC_ALIAS,
    LEGACY_ANSWER_FRACTION_METRIC,
    PRIMARY_METRIC_NAME,
    build_balanced_macro_summary_rows,
    build_metric_summary_bundle_row,
    payload_from_amount_secondary_macros,
    payload_from_balanced_macros,
)
from .target_metric_names import (
    EVAL_FRAC_R2_MEAN_CO2,
    EVAL_FRAC_R2_MEAN_H2,
    EVAL_FRAC_R2_MEAN_H2O,
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
    METRIC_KIND_AMOUNT,
    METRIC_KIND_FRAC,
    METRIC_VERSION_AMOUNT_ROWS,
    METRIC_VERSION_FRAC_ROWS,
)
from .target_v4_provenance import (
    build_policy_macro_summary_rows,
    load_provenance_policy,
    merge_provenance_into_metrics,
    payload_from_policy_macros,
    resolve_provenance_path,
)

METRIC_VERSION = METRIC_VERSION_AMOUNT_ROWS
FRAC_METRIC_VERSION = METRIC_VERSION_FRAC_ROWS
STREAM17_SKIP_REASON = "no_predicted_stream17_edge"
EPS = 1e-8
MOLE_FLOW_PROPERTY = "Mole_Flow"

SPECIES_TO_FRAC_PROPERTY: dict[str, str] = {
    "H2": "Frac_H2",
    "CO2": "Frac_CO2",
    "H2O": "Frac_H2O",
    "CH4": "Frac_CH4",
    "CO": "Frac_CO",
    "O2": "Frac_O2",
    "N2": "Frac_N2",
}

EDGE_STREAM_PROPERTY_TO_INDEX: dict[str, int] = {
    name: idx for idx, name in enumerate(STREAM_EDGE_FEATURE_SLOTS)
}

FRAC_PROPERTY_NAMES: tuple[str, ...] = tuple(
    c for c in STREAM_EDGE_FEATURE_SLOTS if c.startswith("Frac_")
)


@dataclass(frozen=True)
class TargetSpecV4:
    process_id: str
    process_num: int
    target_id: str
    target_name: str
    target_species: str
    stream_key: str
    required_stream_key: str
    canonical_edge_id: str
    scale_factor: float
    formula_type: str
    target_formula: str
    target_stream_node: str
    source_row: str


@dataclass
class TargetV4EvalResult:
    target_metrics_df: pd.DataFrame
    summary_df: pd.DataFrame
    skipped_df: pd.DataFrame
    predictions_df: pd.DataFrame
    mapping_df: pd.DataFrame
    physical_validity_df: pd.DataFrame
    payload: dict[str, Any]
    frac_target_metrics_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    frac_summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    frac_skipped_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    frac_predictions_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)


def frac_property_for_species(species: str) -> str | None:
    return SPECIES_TO_FRAC_PROPERTY.get(str(species).strip())


def _is_stream17_exhaust_target(spec: TargetSpecV4) -> bool:
    return "Stream17" in spec.target_name or (
        spec.required_stream_key == "17" and "Stream17" in spec.target_formula
    )


def _select_spec_edge_rows(piv: pd.DataFrame, spec: TargetSpecV4) -> pd.DataFrame:
    sub = piv[(piv["process_id"].astype(str) == spec.process_id)].copy()
    sub = sub[sub["stream_key_norm"] == spec.required_stream_key]
    if spec.canonical_edge_id:
        sub = sub[sub["canonical_edge_id"].astype(str) == spec.canonical_edge_id]
    return sub


def _metric_rows_from_predictions(
    pred_df: pd.DataFrame,
    flat_specs: Sequence[TargetSpecV4],
    *,
    split_name: str,
    train_cfg: Any | None,
    metric_kind: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Build per-target_id metric rows from long-form prediction rows (amount or frac)."""
    metric_rows: list[dict[str, Any]] = []
    skipped_extra: list[dict[str, Any]] = []
    if pred_df.empty:
        return pd.DataFrame(), skipped_extra

    for (split, process_id, target_id), group in pred_df.groupby(
        ["split", "process_id", "target_id"], dropna=False
    ):
        mb = _metric_bundle(group["target_true_value"], group["target_pred_value"])
        first = group.iloc[0]
        spec_match = next(
            (s for s in flat_specs if s.target_id == target_id and s.process_id == process_id),
            None,
        )
        frac_col = frac_property_for_species(str(first.get("target_species", "")))
        r2_val = float(mb["r2"])
        skip_reason = str(first.get("skip_reason", "") or "")
        included = bool(first.get("included_in_primary", False))
        if skip_reason:
            included = False
        elif not math.isfinite(r2_val):
            included = False

        target_stream = str(first.get("target_stream", first.get("required_stream_key", "")))
        if spec_match is not None:
            target_stream = spec_match.target_stream_node or spec_match.required_stream_key
        row: dict[str, Any] = {
            "split": split,
            "process_id": process_id,
            "target_id": target_id,
            "target_species": first["target_species"],
            "target_feature": first.get("target_feature", first.get("target_feature_name", "")),
            "target_feature_name": first.get("target_feature_name", ""),
            "target_stream": target_stream,
            "used_edge_id": spec_match.canonical_edge_id if spec_match else first.get("canonical_answer_edge_id", ""),
            "canonical_answer_edge_id": spec_match.canonical_edge_id if spec_match else first.get("canonical_answer_edge_id", ""),
            "used_stream_key": first.get("required_stream_key", ""),
            "frac_property": frac_col or "",
            "mae": float(mb["mae"]),
            "rmse": float(mb["rmse"]),
            "r2": r2_val,
            "n_samples": int(mb["n_samples"]),
            "included_in_primary": included,
            "skip_reason": skip_reason,
            "status": str(mb.get("r2_status", "ok")),
            "metric_kind": metric_kind,
        }
        if metric_kind == METRIC_KIND_AMOUNT:
            row.update(
                {
                    "used_mole_flow_slot": MOLE_FLOW_PROPERTY,
                    "used_frac_slot": frac_col or "",
                    "used_formula": spec_match.target_formula if spec_match else "",
                    "formula_status": spec_match.formula_type if spec_match else "",
                    "scale": float(first.get("scale_factor", 1.0)),
                    "scale_factor": float(first.get("scale_factor", 1.0)),
                    "required_stream_key": first.get("required_stream_key", ""),
                    "relative_accuracy_score": float(mb["relative_accuracy_score"]),
                    "relative_rmse": float(mb["relative_rmse"]),
                    "r2_raw": float(mb["r2_raw"]),
                    "r2_clamped": float(mb["r2_clamped"]),
                    "true_mean": float(mb["true_mean"]),
                    "pred_mean": float(mb["pred_mean"]),
                    "n_valid_samples": int(mb["n_samples"]),
                }
            )
            if spec_match is not None and train_cfg is not None:
                from .target_row_spec import final_weight_for_spec, target_row_from_v4_spec

                lw_val = final_weight_for_spec(target_row_from_v4_spec(spec_match), train_cfg)
                row["loss_weight_applied"] = bool(lw_val > 1.0)
                row["loss_weight_value"] = float(lw_val)
        metric_rows.append(row)
    return pd.DataFrame(metric_rows), skipped_extra


def normalize_stream_key(value: Any) -> str:
    return canonicalize_stream_key(value)


def load_target_stream_targets_v4(path: Path) -> dict[str, list[TargetSpecV4]]:
    if not path.is_file():
        raise FileNotFoundError(f"target_stream_targets.csv not found: {path}")
    frame = pd.read_csv(path, dtype={"process_id": int})
    required = {
        "process_id",
        "target_id",
        "target_species",
        "target_feature_name",
        "canonical_answer_edge_id",
        "main_data_stream_key",
    }
    missing_cols = required - set(frame.columns)
    if missing_cols:
        raise ValueError(f"{path} missing columns: {sorted(missing_cols)}")

    by_process: dict[str, list[TargetSpecV4]] = defaultdict(list)
    seen: set[tuple[int, str]] = set()
    for row in frame.to_dict(orient="records"):
        pnum = int(row["process_id"])
        tid = str(row["target_id"]).strip()
        key = (pnum, tid)
        if key in seen:
            raise ValueError(f"duplicate target_id={tid!r} for process_id={pnum} in {path}")
        seen.add(key)
        species = str(row["target_species"]).strip()
        if species not in SPECIES_TO_FRAC_PROPERTY:
            continue
        req = normalize_stream_key(row.get("required_stream_key") or row.get("main_data_stream_key"))
        try:
            sf = float(row.get("scale_factor", 1.0))
        except (TypeError, ValueError):
            sf = float("nan")
        by_process[f"Process{pnum}"].append(
            TargetSpecV4(
                process_id=f"Process{pnum}",
                process_num=pnum,
                target_id=tid,
                target_name=str(row.get("target_feature_name", "")),
                target_species=species,
                stream_key=normalize_stream_key(row.get("main_data_stream_key")),
                required_stream_key=req,
                canonical_edge_id=str(row.get("canonical_answer_edge_id", "")).strip(),
                scale_factor=sf,
                formula_type=str(row.get("formula_type", "")),
                target_formula=str(row.get("target_formula", "")),
                target_stream_node=str(row.get("target_stream_node", "")),
                source_row=str(row.get("source_row", "")),
            )
        )
    return dict(by_process)


def load_target_stream_targets_v4_all(path: Path) -> list[TargetSpecV4]:
    out: list[TargetSpecV4] = []
    for specs in load_target_stream_targets_v4(path).values():
        out.extend(specs)
    return out


def _parse_scale_factor(value: Any) -> tuple[float | None, str | None]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 1.0, None
    try:
        sf = float(value)
    except (TypeError, ValueError):
        return None, f"invalid scale_factor={value!r}"
    if not math.isfinite(sf):
        return None, f"non-finite scale_factor={value!r}"
    return sf, None


def _smape(y_true: pd.Series, y_pred: pd.Series, eps: float = EPS) -> float:
    if len(y_true) == 0:
        return 0.0
    den = y_true.abs() + y_pred.abs() + eps
    return float((2.0 * (y_true - y_pred).abs() / den).mean() * 100.0)


def _r2_details(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    rel_floor = 1e-4
    n = int(len(y_true))
    if n < 1:
        return {
            "r2_raw": 0.0,
            "r2_clamped": 0.0,
            "sse": 0.0,
            "sst": 0.0,
            "sst_eff": 0.0,
            "r2_status": "no_samples",
        }
    if n < 2:
        ss_res = float(((y_true - y_pred) ** 2).sum())
        if ss_res <= 1e-18:
            return {
                "r2_raw": 1.0,
                "r2_clamped": 1.0,
                "sse": ss_res,
                "sst": 0.0,
                "sst_eff": 0.0,
                "r2_status": "single_sample_perfect",
            }
        y0 = float(y_true.iloc[0])
        ymax = max(abs(y0), 1e-30)
        raw = 1.0 - ss_res / (ymax * ymax + 1e-30)
        return {
            "r2_raw": raw,
            "r2_clamped": max(0.0, min(1.0, raw)),
            "sse": ss_res,
            "sst": 0.0,
            "sst_eff": ymax * ymax + 1e-30,
            "r2_status": "single_sample",
        }
    ss_res = float(((y_true - y_pred) ** 2).sum())
    y_mean = float(y_true.mean())
    ss_tot = float(((y_true - y_mean) ** 2).sum())
    scale = max(float((y_true.astype(float) ** 2).mean()), y_mean * y_mean, 1e-30)
    if not math.isfinite(ss_tot):
        ss_tot = 0.0
    ss_tot_eff = max(ss_tot, rel_floor * scale, 1e-30)
    r2 = 1.0 - ss_res / ss_tot_eff
    if not math.isfinite(r2):
        r2 = 0.0
    r2_clamped = max(-1.0, min(1.0, r2))
    status = "ok"
    if ss_tot <= rel_floor * scale:
        status = "low_variance_r2_unstable"
    if r2 != r2_clamped:
        status = f"{status}_clamped" if status != "ok" else "clamped"
    return {
        "r2_raw": r2,
        "r2_clamped": r2_clamped,
        "sse": ss_res,
        "sst": ss_tot,
        "sst_eff": ss_tot_eff,
        "r2_status": status,
    }


def _r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(_r2_details(y_true, y_pred)["r2_clamped"])


def _metric_bundle(y_true: pd.Series, y_pred: pd.Series, r2_eps: float = 1e-6) -> dict[str, Any]:
    n = int(len(y_true))
    if n == 0:
        return {
            "true_mean": 0.0,
            "pred_mean": 0.0,
            "mean_bias": 0.0,
            "true_std": 0.0,
            "pred_std": 0.0,
            "std_ratio": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "relative_accuracy_score": 0.0,
            "relative_rmse": 0.0,
            "r2_raw": 0.0,
            "r2_clamped": 0.0,
            "sse": 0.0,
            "sst": 0.0,
            "sst_eff": 0.0,
            "n_samples": 0,
            "r2_unstable": True,
            "r2_status": "no_samples",
        }
    yt = y_true.astype(float)
    yp = y_pred.astype(float)
    true_mean = float(yt.mean())
    pred_mean = float(yp.mean())
    true_std = float(yt.std(ddof=0))
    pred_std = float(yp.std(ddof=0))
    std_ratio = float(pred_std / true_std) if true_std > r2_eps else 0.0
    mae = float((yt - yp).abs().mean())
    rmse = float(((yt - yp) ** 2).mean()) ** 0.5
    mean_abs_true = float(yt.abs().mean())
    relative_rmse = float(rmse / (mean_abs_true + EPS))
    relative_accuracy_score = max(0.0, min(1.0, 1.0 - relative_rmse))
    r2_detail = _r2_details(yt, yp)
    r2_val = float(r2_detail["r2_clamped"])
    r2_unstable = bool(true_std < r2_eps or str(r2_detail["r2_status"]).startswith("low_variance"))
    return {
        "true_mean": true_mean,
        "pred_mean": pred_mean,
        "mean_bias": float(pred_mean - true_mean),
        "true_std": true_std,
        "pred_std": pred_std,
        "std_ratio": std_ratio,
        "mae": mae,
        "rmse": rmse,
        "r2": r2_val,
        "relative_accuracy_score": relative_accuracy_score,
        "relative_rmse": relative_rmse,
        "r2_raw": float(r2_detail["r2_raw"]),
        "r2_clamped": float(r2_detail["r2_clamped"]),
        "sse": float(r2_detail["sse"]),
        "sst": float(r2_detail["sst"]),
        "sst_eff": float(r2_detail["sst_eff"]),
        "n_samples": n,
        "r2_unstable": r2_unstable,
        "r2_status": str(r2_detail["r2_status"]),
    }


def assert_edge_predictions_original_scale(
    edge_predictions: pd.DataFrame,
    *,
    strict: bool = False,
) -> list[str]:
    """Sanity-check that y_*_orig columns look like physical units, not z-scores."""
    warnings: list[str] = []
    if edge_predictions is None or edge_predictions.empty:
        return warnings
    if "y_true_orig" not in edge_predictions.columns or "y_pred_orig" not in edge_predictions.columns:
        raise ValueError("edge_predictions must contain y_true_orig and y_pred_orig for v4 metrics.")
    sup = edge_predictions[edge_predictions.get("y_edge_mask", 1.0) > 0.0]
    if sup.empty:
        return warnings
    for prop in FRAC_PROPERTY_NAMES:
        sub = sup[sup["property_name"].astype(str) == prop]
        if sub.empty:
            continue
        for col in ("y_true_orig", "y_pred_orig"):
            s = pd.to_numeric(sub[col], errors="coerce").dropna()
            if s.empty:
                continue
            frac_high = float((s > 1.5).mean())
            frac_neg = float((s < -0.1).mean())
            if frac_high > 0.05 or frac_neg > 0.05:
                msg = (
                    f"v4 scale check: {col} {prop} has {frac_high:.1%} >1.5 or {frac_neg:.1%} <-0.1 "
                    "(possible normalized values)"
                )
                warnings.append(msg)
                if strict:
                    raise ValueError(msg)
    mole = sup[sup["property_name"].astype(str) == MOLE_FLOW_PROPERTY]
    if not mole.empty:
        for col in ("y_true_orig", "y_pred_orig"):
            s = pd.to_numeric(mole[col], errors="coerce").dropna()
            if s.empty:
                continue
            neg_rate = float((s < 0.0).mean())
            med = float(s.median())
            if neg_rate > 0.01:
                warnings.append(f"v4 scale check: {col} Mole_Flow negative rate={neg_rate:.3f}")
            if strict and med < 0.5 and float(s.abs().max()) < 5.0:
                raise ValueError(
                    f"v4 scale check: {col} Mole_Flow median={med:.4f} looks normalized (strict mode)"
                )
    return warnings


def compute_target_v4_metrics_from_original_scale(
    *,
    edge_predictions: pd.DataFrame,
    target_specs: Sequence[TargetSpecV4] | Mapping[str, Sequence[TargetSpecV4]],
    split_name: str = "",
    base_metrics: Mapping[str, Any] | None = None,
    train_cfg: Any | None = None,
) -> TargetV4EvalResult:
    """Compute per-target_id v4 metrics from long-form edge_predictions (original scale only)."""
    if isinstance(target_specs, Mapping):
        flat_specs: list[TargetSpecV4] = []
        for specs in target_specs.values():
            flat_specs.extend(specs)
    else:
        flat_specs = list(target_specs)

    warnings = assert_edge_predictions_original_scale(edge_predictions, strict=False)
    payload: dict[str, Any] = dict(base_metrics or {})
    payload["metric_version"] = METRIC_VERSION

    if edge_predictions is None or edge_predictions.empty:
        empty = pd.DataFrame()
        payload["target_v4_warnings"] = ["edge_predictions empty"]
        return TargetV4EvalResult(
            target_metrics_df=empty,
            summary_df=empty,
            skipped_df=empty,
            predictions_df=empty,
            mapping_df=empty,
            physical_validity_df=empty,
            payload=payload,
            warnings=list(payload["target_v4_warnings"]),
            frac_target_metrics_df=empty,
            frac_summary_df=empty,
        )

    sup = edge_predictions[edge_predictions["y_edge_mask"] > 0.0].copy()
    if sup.empty:
        empty = pd.DataFrame()
        payload["target_v4_warnings"] = ["no masked edge rows in edge_predictions"]
        return TargetV4EvalResult(
            target_metrics_df=empty,
            summary_df=empty,
            skipped_df=empty,
            predictions_df=empty,
            mapping_df=empty,
            physical_validity_df=empty,
            payload=payload,
            warnings=list(payload["target_v4_warnings"]),
            frac_target_metrics_df=empty,
            frac_summary_df=empty,
        )

    sup["stream_key_norm"] = sup["main_data_stream_key"].map(normalize_stream_key)
    key_cols = [
        "split",
        "process_id",
        "sample_id",
        "canonical_edge_id",
        "main_data_stream_key",
        "stream_key_norm",
    ]
    piv = sup.pivot_table(
        index=key_cols,
        columns="property_name",
        values=["y_true_orig", "y_pred_orig"],
        aggfunc="first",
    ).reset_index()
    piv.columns = [
        "_".join([c for c in col if c]).strip("_") if isinstance(col, tuple) else col
        for col in piv.columns
    ]

    pred_rows: list[dict[str, Any]] = []
    pred_rows_frac: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    frac_skipped_rows: list[dict[str, Any]] = []
    phys_rows: list[dict[str, Any]] = []

    for spec in flat_specs:
        frac_col = frac_property_for_species(spec.target_species)
        if frac_col is None:
            reason = f"unsupported_species:{spec.target_species}"
            skipped_rows.append(
                {
                    "process_id": spec.process_id,
                    "target_id": spec.target_id,
                    "target_name": spec.target_name,
                    "reason": reason,
                }
            )
            frac_skipped_rows.append(
                {
                    "process_id": spec.process_id,
                    "target_id": spec.target_id,
                    "target_name": spec.target_name,
                    "reason": reason,
                    "metric_kind": METRIC_KIND_FRAC,
                }
            )
            continue

        true_frac = f"y_true_orig_{frac_col}"
        pred_frac = f"y_pred_orig_{frac_col}"
        if true_frac not in piv.columns or pred_frac not in piv.columns:
            reason = f"missing_frac_columns:{frac_col}"
            skipped_rows.append(
                {
                    "process_id": spec.process_id,
                    "target_id": spec.target_id,
                    "target_name": spec.target_name,
                    "reason": reason,
                }
            )
            frac_skipped_rows.append(
                {
                    "process_id": spec.process_id,
                    "target_id": spec.target_id,
                    "target_name": spec.target_name,
                    "reason": reason,
                    "metric_kind": METRIC_KIND_FRAC,
                }
            )
            continue

        sub = _select_spec_edge_rows(piv, spec)
        stream17 = _is_stream17_exhaust_target(spec)
        sf, sf_err = _parse_scale_factor(spec.scale_factor)

        mapping_status = "ok" if len(sub) else "missing_stream_rows"
        mapping_rows.append(
            {
                "process_id": spec.process_num,
                "process_label": spec.process_id,
                "target_id": spec.target_id,
                "target_stream_node": spec.target_stream_node,
                "target_species": spec.target_species,
                "target_feature_name": spec.target_name,
                "target_formula": spec.target_formula,
                "formula_type": spec.formula_type,
                "canonical_answer_edge_id": spec.canonical_edge_id,
                "main_data_stream_key": spec.stream_key,
                "required_stream_key": spec.required_stream_key,
                "scale_factor": sf if sf_err is None else float("nan"),
                "n_edge_rows": int(len(sub)),
                "mapping_status": mapping_status,
            }
        )
        if sub.empty:
            reason = (
                STREAM17_SKIP_REASON
                if stream17
                else f"no_rows_for_stream:{spec.required_stream_key}"
            )
            skipped_rows.append(
                {
                    "process_id": spec.process_id,
                    "target_id": spec.target_id,
                    "target_name": spec.target_name,
                    "reason": reason,
                }
            )
            frac_skipped_rows.append(
                {
                    "process_id": spec.process_id,
                    "target_id": spec.target_id,
                    "target_name": spec.target_name,
                    "reason": reason,
                    "metric_kind": METRIC_KIND_FRAC,
                }
            )
            continue

        tf = pd.to_numeric(sub[true_frac], errors="coerce")
        pf = pd.to_numeric(sub[pred_frac], errors="coerce")
        frac_valid = tf.notna() & pf.notna()
        if int(frac_valid.sum()) == 0:
            reason = "no_valid_frac_rows"
            skipped_rows.append(
                {
                    "process_id": spec.process_id,
                    "target_id": spec.target_id,
                    "target_name": spec.target_name,
                    "reason": reason,
                }
            )
            frac_skipped_rows.append(
                {
                    "process_id": spec.process_id,
                    "target_id": spec.target_id,
                    "target_name": spec.target_name,
                    "reason": reason,
                    "metric_kind": METRIC_KIND_FRAC,
                }
            )
            continue

        sub_frac = sub.loc[frac_valid].copy()
        sub_frac["target_true_frac"] = tf[frac_valid].astype(float)
        sub_frac["target_pred_frac"] = pf[frac_valid].astype(float)

        # --- Frac primary (no Mole_Flow, no scale_factor) ---
        for _, row in sub_frac.iterrows():
            pred_rows_frac.append(
                {
                    "split": row.get("split", split_name),
                    "process_id": spec.process_id,
                    "sample_id": row["sample_id"],
                    "target_id": spec.target_id,
                    "target_species": spec.target_species,
                    "target_feature": spec.target_name,
                    "target_feature_name": spec.target_name,
                    "target_stream": spec.target_stream_node or spec.required_stream_key,
                    "required_stream_key": spec.required_stream_key,
                    "canonical_answer_edge_id": spec.canonical_edge_id,
                    "source_canonical_edge_id": row["canonical_edge_id"],
                    "frac_property": frac_col,
                    "target_true_value": float(row["target_true_frac"]),
                    "target_pred_value": float(row["target_pred_frac"]),
                    "included_in_primary": True,
                    "skip_reason": "",
                }
            )

        # --- Amount secondary (Mole_Flow * Frac * scale_factor) ---
        true_mole = "y_true_orig_Mole_Flow"
        pred_mole = "y_pred_orig_Mole_Flow"
        missing_mole = [c for c in (true_mole, pred_mole) if c not in piv.columns]
        if sf_err or missing_mole:
            skipped_rows.append(
                {
                    "process_id": spec.process_id,
                    "target_id": spec.target_id,
                    "target_name": spec.target_name,
                    "reason": sf_err or f"missing_pivot_columns:{missing_mole}",
                }
            )
        else:
            tm = pd.to_numeric(sub[true_mole], errors="coerce")
            pm = pd.to_numeric(sub[pred_mole], errors="coerce")
            amount_valid = tm.notna() & pm.notna() & frac_valid
            if int(amount_valid.sum()) == 0:
                skipped_rows.append(
                    {
                        "process_id": spec.process_id,
                        "target_id": spec.target_id,
                        "target_name": spec.target_name,
                        "reason": "no_valid_amount_rows",
                    }
                )
            else:
                sub_amt = sub.loc[amount_valid].copy()
                sub_amt["target_true_value"] = (
                    float(sf) * tm[amount_valid].astype(float) * tf[amount_valid].astype(float)
                )
                sub_amt["target_pred_value"] = (
                    float(sf) * pm[amount_valid].astype(float) * pf[amount_valid].astype(float)
                )
                neg_mole = float((pm[amount_valid] < 0.0).mean())
                frac_below = float((pf[amount_valid] < 0.0).mean())
                frac_above = float((pf[amount_valid] > 1.0).mean())
                phys_rows.append(
                    {
                        "process_id": spec.process_id,
                        "target_id": spec.target_id,
                        "target_species": spec.target_species,
                        "frac_property_name": frac_col,
                        "negative_mole_flow_rate": neg_mole,
                        "frac_below_zero_rate": frac_below,
                        "frac_above_one_rate": frac_above,
                        "mole_flow_true_mean": float(tm[amount_valid].mean()),
                        "mole_flow_pred_mean": float(pm[amount_valid].mean()),
                        "frac_true_mean": float(tf[amount_valid].mean()),
                        "frac_pred_mean": float(pf[amount_valid].mean()),
                    }
                )
                for _, row in sub_amt.iterrows():
                    pred_rows.append(
                        {
                            "split": row.get("split", split_name),
                            "process_id": spec.process_id,
                            "sample_id": row["sample_id"],
                            "target_id": spec.target_id,
                            "target_species": spec.target_species,
                            "target_feature_name": spec.target_name,
                            "required_stream_key": spec.required_stream_key,
                            "canonical_answer_edge_id": spec.canonical_edge_id,
                            "source_canonical_edge_id": row["canonical_edge_id"],
                            "scale_factor": float(sf),
                            "target_true_value": float(row["target_true_value"]),
                            "target_pred_value": float(row["target_pred_value"]),
                        }
                    )

    pred_df = pd.DataFrame(pred_rows)
    pred_frac_df = pd.DataFrame(pred_rows_frac)
    mapping_df = pd.DataFrame(mapping_rows)
    skipped_df = pd.DataFrame(skipped_rows)
    phys_df = pd.DataFrame(phys_rows)

    metric_df, _ = _metric_rows_from_predictions(
        pred_df, flat_specs, split_name=split_name, train_cfg=train_cfg, metric_kind=METRIC_KIND_AMOUNT
    )
    metric_frac_df, _ = _metric_rows_from_predictions(
        pred_frac_df, flat_specs, split_name=split_name, train_cfg=None, metric_kind=METRIC_KIND_FRAC
    )
    prov_path = resolve_provenance_path()
    provenance = load_provenance_policy(prov_path)
    provenance_loaded = not provenance.empty
    if provenance_loaded:
        metric_df = merge_provenance_into_metrics(metric_df, provenance)
        metric_frac_df = merge_provenance_into_metrics(metric_frac_df, provenance)
        payload["target_v4_provenance_policy_path"] = str(prov_path)
        payload["target_v4_provenance_loaded"] = True
        if "include_in_main_verified_macro" in provenance.columns:
            excluded = provenance[~provenance["include_in_main_verified_macro"]]["target_id"].astype(str).tolist()
            payload["target_v4_excluded_target_ids"] = excluded
    else:
        payload["target_v4_provenance_loaded"] = False
        warnings.append(
            "v4_target_provenance_decisions.csv not found; macro R² uses all evaluated targets (deprecated)."
        )

    if not metric_frac_df.empty:
        if provenance_loaded and "include_in_main_verified_macro" in metric_frac_df.columns:
            inc_mask = metric_frac_df["include_in_main_verified_macro"].astype(str).str.lower().isin(
                ("true", "1", "yes", "t")
            )
            r2_ok = pd.to_numeric(metric_frac_df["r2"], errors="coerce").map(math.isfinite)
            metric_frac_df["included_in_primary"] = inc_mask & r2_ok
            bad = ~(inc_mask & r2_ok)
            if bad.any():
                existing = metric_frac_df.loc[bad, "skip_reason"].astype(str)
                metric_frac_df.loc[bad, "skip_reason"] = existing.where(
                    existing.str.strip() != "",
                    "excluded_from_main_verified_macro",
                )
        else:
            r2_ok = pd.to_numeric(metric_frac_df["r2"], errors="coerce").map(math.isfinite)
            metric_frac_df["included_in_primary"] = r2_ok
            metric_frac_df.loc[~r2_ok, "skip_reason"] = metric_frac_df.loc[~r2_ok, "skip_reason"].where(
                metric_frac_df.loc[~r2_ok, "skip_reason"].astype(str).str.strip() != "",
                "non_finite_r2",
            )

    summary_rows = _build_v4_summary_rows(
        metric_df,
        pred_df,
        split_name=split_name,
        provenance_loaded=provenance_loaded,
    )
    summary_df = pd.DataFrame(summary_rows)
    frac_summary_rows = _build_v4_summary_rows(
        metric_frac_df,
        pred_frac_df,
        split_name=split_name,
        provenance_loaded=provenance_loaded,
        use_included_in_primary=True,
    )
    frac_summary_df = pd.DataFrame(frac_summary_rows)

    payload.update(_payload_from_v4_summary(summary_df, metric_df, split_name=split_name))
    payload.update(_payload_from_v4_summary(frac_summary_df, metric_frac_df, split_name=split_name))
    payload.update(payload_from_policy_macros(summary_df, split_name=split_name))
    legacy_macro = payload.get("metric_answer_all_targets_mean_r2")
    if legacy_macro is None:
        legacy_macro = payload.get("legacy_answer_fraction_macro_r2")
    payload.update(
        payload_from_balanced_macros(
            frac_summary_df,
            metric_frac_df,
            split_name=split_name,
            legacy_answer_fraction_r2=float(legacy_macro) if legacy_macro is not None else None,
            use_included_in_primary=True,
        )
    )
    payload.update(
        payload_from_amount_secondary_macros(
            summary_df,
            metric_df,
            split_name=split_name,
        )
    )
    payload["target_v4_mapping"] = mapping_rows
    payload["target_v4_warnings"] = warnings
    payload["target_v4_num_targets_total"] = len(flat_specs)
    payload["target_v4_num_targets_evaluated"] = int(
        metric_df["target_id"].nunique() if not metric_df.empty else 0
    )
    payload["target_v4_num_targets_skipped"] = len(skipped_rows)
    if not summary_df.empty and summary_df.get("summary_kind", pd.Series(dtype=str)).astype(str).str.contains(
        "species_mean_reporting_only", na=False
    ).any():
        warnings.append(
            "species_mean_reporting_only rows (__SPECIES_*) are diagnostic only; "
            "primary macro metrics use per-target_id row means."
        )
    payload["eval_metric_granularity_amount"] = "per_target_row"
    payload["eval_metric_granularity_frac"] = "per_target_row_frac_species"
    payload["eval_metric_version_amount"] = METRIC_VERSION
    payload["eval_metric_version_frac"] = FRAC_METRIC_VERSION
    payload["legacy_answer_fraction_macro_r2_note"] = (
        "Use metric_target_r2/metric_tailgas_r2 or legacy_answer_fraction_macro_r2; "
        "not comparable to Frac primary or amount secondary macros."
    )
    n_frac_skip = len(frac_skipped_rows)
    if n_frac_skip:
        payload["target_v4_frac_skipped_count"] = n_frac_skip
        warnings.append(f"frac_primary: {n_frac_skip} target row(s) skipped (see target_metrics_v4_frac / skipped)")

    return TargetV4EvalResult(
        target_metrics_df=metric_df,
        summary_df=summary_df,
        skipped_df=skipped_df,
        predictions_df=pred_df,
        mapping_df=mapping_df,
        physical_validity_df=phys_df,
        payload=payload,
        frac_target_metrics_df=metric_frac_df,
        frac_summary_df=frac_summary_df,
        frac_skipped_df=pd.DataFrame(frac_skipped_rows),
        frac_predictions_df=pred_frac_df,
        warnings=warnings,
    )


def _build_v4_summary_rows(
    metric_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    *,
    split_name: str = "",
    provenance_loaded: bool = False,
    use_included_in_primary: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if metric_df.empty:
        return rows

    usable = metric_df[pd.to_numeric(metric_df["r2"], errors="coerce").notna()].copy()

    for (split, process_id), g in usable.groupby(["split", "process_id"], dropna=False):
        r2s = pd.to_numeric(g["r2"], errors="coerce").dropna()
        scores = (
            pd.to_numeric(g["relative_accuracy_score"], errors="coerce").dropna()
            if "relative_accuracy_score" in g.columns
            else pd.Series(dtype=float)
        )
        macro_r2 = float(r2s.mean()) if not r2s.empty else float("nan")
        rows.append(
            {
                "split": split,
                "process_id": process_id,
                "target_id": "__ALL_macro_process__",
                "mae": float(pd.to_numeric(g["mae"], errors="coerce").mean()),
                "rmse": float(pd.to_numeric(g["rmse"], errors="coerce").mean()),
                "r2": macro_r2,
                "relative_accuracy_score": float(scores.mean()) if not scores.empty else float("nan"),
                "n_samples": int(pd.to_numeric(g["n_samples"], errors="coerce").sum()),
                "n_targets_present": int(len(g)),
                "summary_kind": "process_macro",
            }
        )

    for split, g in usable.groupby("split", dropna=False):
        proc_macros: list[float] = []
        for _, pg in g.groupby("process_id"):
            r2s = pd.to_numeric(pg["r2"], errors="coerce").dropna()
            if not r2s.empty:
                proc_macros.append(float(r2s.mean()))
        macro_r2 = float(sum(proc_macros) / len(proc_macros)) if proc_macros else float("nan")
        scores = (
            pd.to_numeric(g["relative_accuracy_score"], errors="coerce").dropna()
            if "relative_accuracy_score" in g.columns
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "split": split,
                "process_id": "__ALL__",
                "target_id": "__ALL_macro_split__",
                "mae": float(pd.to_numeric(g["mae"], errors="coerce").mean()),
                "rmse": float(pd.to_numeric(g["rmse"], errors="coerce").mean()),
                "r2": macro_r2,
                "relative_accuracy_score": float(scores.mean()) if not scores.empty else float("nan"),
                "n_samples": int(pd.to_numeric(g["n_samples"], errors="coerce").sum()),
                "n_targets_present": int(g["target_id"].nunique()),
                "n_processes": int(g["process_id"].nunique()),
                "summary_kind": "split_macro",
            }
        )

    if not pred_df.empty:
        sp = split_name or (str(metric_df["split"].iloc[0]) if "split" in metric_df.columns else "")
        sub = pred_df if not sp else pred_df[pred_df["split"].astype(str) == sp]
        if not sub.empty:
            mb = _metric_bundle(sub["target_true_value"], sub["target_pred_value"])
            rows.append(
                {
                    "split": sp,
                    "process_id": "__ALL__",
                    "target_id": "__ALL_flatten__",
                    "mae": float(mb["mae"]),
                    "rmse": float(mb["rmse"]),
                    "r2": float(mb["r2"]),
                    "relative_accuracy_score": float(mb["relative_accuracy_score"]),
                    "n_samples": int(mb["n_samples"]),
                    "n_targets_present": int(sub["target_id"].nunique()),
                    "summary_kind": "flatten_pooled",
                }
            )

    for species in ("H2", "CO2", "H2O"):
        sg = usable[usable["target_species"].astype(str) == species]
        if sg.empty:
            continue
        r2s = pd.to_numeric(sg["r2"], errors="coerce").dropna()
        scores = (
            pd.to_numeric(sg["relative_accuracy_score"], errors="coerce").dropna()
            if "relative_accuracy_score" in sg.columns
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "split": split_name or str(sg["split"].iloc[0]),
                "process_id": "__ALL__",
                "target_id": f"__SPECIES_{species}__",
                "target_species": species,
                "r2": float(r2s.mean()) if not r2s.empty else float("nan"),
                "relative_accuracy_score": float(scores.mean()) if not scores.empty else float("nan"),
                "n_targets_present": int(sg["target_id"].nunique()),
                "summary_kind": "species_mean_reporting_only",
            }
        )
    sp = split_name or (str(usable["split"].iloc[0]) if "split" in usable.columns and not usable.empty else "")
    rows.extend(
        build_policy_macro_summary_rows(usable, split=sp, provenance_loaded=provenance_loaded)
    )
    rows.extend(
        build_balanced_macro_summary_rows(
            metric_df,
            split=sp,
            provenance_loaded=provenance_loaded,
            use_included_in_primary=use_included_in_primary,
        )
    )

    return rows


def _payload_from_v4_summary(
    summary_df: pd.DataFrame,
    metric_df: pd.DataFrame,
    *,
    split_name: str = "",
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if summary_df.empty:
        return out

    def _pick_macro(kind: str, sp: str) -> float:
        sub = summary_df[
            (summary_df.get("summary_kind", "") == kind)
            & (summary_df["target_id"].astype(str).str.contains("macro", na=False))
        ]
        if sp and "split" in sub.columns:
            sub = sub[sub["split"].astype(str) == sp]
        if sub.empty:
            return float("nan")
        return float(pd.to_numeric(sub.iloc[-1]["r2"], errors="coerce"))

    def _pick_summary_value(frame: pd.DataFrame, kind: str, sp: str, column: str) -> float:
        if column not in frame.columns:
            return float("nan")
        sub = frame[
            (frame.get("summary_kind", "") == kind)
            & (frame["target_id"].astype(str).str.contains("macro", na=False))
        ]
        if sp and "split" in sub.columns:
            sub = sub[sub["split"].astype(str) == sp]
        if sub.empty:
            return float("nan")
        return float(pd.to_numeric(sub.iloc[-1][column], errors="coerce"))

    splits = [split_name] if split_name else sorted(metric_df["split"].astype(str).unique().tolist())
    for sp in splits:
        macro_main = _pick_macro("split_macro_main_verified", sp)
        if not math.isfinite(macro_main):
            macro_main = _pick_macro("split_macro", sp)
        if math.isfinite(macro_main):
            out[f"{sp}/target_v4_macro_r2_main_verified"] = macro_main
            out[f"{sp}/target_v4_macro_r2"] = macro_main
        macro_formula = _pick_macro("split_macro_formula_available", sp)
        if math.isfinite(macro_formula):
            out[f"{sp}/target_v4_macro_r2_formula_available"] = macro_formula
        macro_all = _pick_macro("split_macro_all_reported", sp)
        if math.isfinite(macro_all):
            out[f"{sp}/target_v4_macro_r2_all_reported"] = macro_all
        macro = macro_main if math.isfinite(macro_main) else _pick_macro("split_macro", sp)
        if math.isfinite(macro):
            out[f"{sp}/target_v4_macro_r2"] = macro
        macro_score = _pick_summary_value(summary_df, "split_macro", sp, "relative_accuracy_score")
        if math.isfinite(macro_score):
            out[f"{sp}/target_v4_macro_relative_accuracy_score"] = macro_score
        proc_sub = summary_df[
            (summary_df.get("summary_kind", "") == "process_macro")
            & (summary_df["split"].astype(str) == sp)
        ]
        if not proc_sub.empty:
            out[f"{sp}/target_v4_process_macro_r2_mean"] = float(
                pd.to_numeric(proc_sub["r2"], errors="coerce").mean()
            )
        flat = summary_df[
            (summary_df["target_id"].astype(str) == "__ALL_flatten__")
            & (summary_df["split"].astype(str) == sp)
        ]
        if not flat.empty:
            out[f"{sp}/target_v4_r2"] = float(flat.iloc[0]["r2"])
            out[f"{sp}/target_v4_mae"] = float(flat.iloc[0]["mae"])
            out[f"{sp}/target_v4_rmse"] = float(flat.iloc[0]["rmse"])
            if "relative_accuracy_score" in flat.columns:
                out[f"{sp}/target_v4_relative_accuracy_score"] = float(flat.iloc[0]["relative_accuracy_score"])

        met_sp = metric_df[metric_df["split"].astype(str) == sp] if sp else metric_df
        if not met_sp.empty:
            out[f"{sp}/target_v4_num_targets_evaluated"] = int(met_sp["target_id"].nunique())

        for species, key in (("H2", "target_v4_h2_mean_r2"), ("CO2", "target_v4_co2_mean_r2"), ("H2O", "target_v4_h2o_mean_r2")):
            sg = summary_df[
                (summary_df["target_id"].astype(str) == f"__SPECIES_{species}__")
                & (summary_df["split"].astype(str) == sp)
            ]
            if not sg.empty:
                v = float(pd.to_numeric(sg.iloc[0]["r2"], errors="coerce"))
                if math.isfinite(v):
                    out[f"{sp}/{key}"] = v
                if "relative_accuracy_score" in sg.columns:
                    sv = float(pd.to_numeric(sg.iloc[0]["relative_accuracy_score"], errors="coerce"))
                    if math.isfinite(sv):
                        out[f"{sp}/{key.replace('_mean_r2', '_mean_relative_accuracy_score')}"] = sv

    if split_name:
        sp = split_name
        if f"{sp}/target_v4_macro_r2" in out:
            out["target_v4_macro_r2"] = out[f"{sp}/target_v4_macro_r2"]
            out["test/target_v4_macro_r2"] = out[f"{sp}/target_v4_macro_r2"]
        if f"{sp}/target_v4_macro_relative_accuracy_score" in out:
            out["target_v4_macro_relative_accuracy_score"] = out[f"{sp}/target_v4_macro_relative_accuracy_score"]
            out["test/target_v4_macro_relative_accuracy_score"] = out[
                f"{sp}/target_v4_macro_relative_accuracy_score"
            ]
    return out


def write_target_v4_split_artifacts(
    out_dir: Path,
    result: TargetV4EvalResult,
    *,
    write_predictions: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result.target_metrics_df.to_csv(out_dir / "target_metrics_v4.csv", index=False)
    if not result.frac_target_metrics_df.empty:
        result.frac_target_metrics_df.to_csv(out_dir / "target_metrics_v4_frac.csv", index=False)
    bundle_row = build_metric_summary_bundle_row(result.payload, result.summary_df)
    summary_out = pd.concat(
        [result.summary_df, pd.DataFrame([bundle_row])],
        ignore_index=True,
    )
    summary_out.to_csv(out_dir / "target_metrics_v4_summary.csv", index=False)
    if not result.frac_summary_df.empty:
        frac_bundle = build_metric_summary_bundle_row(result.payload, result.frac_summary_df)
        frac_summary_out = pd.concat(
            [result.frac_summary_df, pd.DataFrame([frac_bundle])],
            ignore_index=True,
        )
        frac_summary_out.to_csv(out_dir / "target_metrics_v4_frac_summary.csv", index=False)
    # Explicit aliases for the user-facing main target metrics:
    # one row per process target_id (e.g. P07_T001/P07_T002/P07_T003).
    result.target_metrics_df.to_csv(out_dir / "main_target_metrics_by_target.csv", index=False)
    summary_out.to_csv(out_dir / "main_target_metrics_summary.csv", index=False)
    result.skipped_df.to_csv(out_dir / "skipped_targets_v4.csv", index=False)
    if not result.frac_skipped_df.empty:
        result.frac_skipped_df.to_csv(out_dir / "skipped_targets_v4_frac.csv", index=False)
    result.mapping_df.to_csv(out_dir / "target_mapping_v4.csv", index=False)
    if write_predictions and not result.predictions_df.empty:
        result.predictions_df.to_csv(out_dir / "target_predictions_v4.csv", index=False)
    if not result.physical_validity_df.empty:
        result.physical_validity_df.to_csv(out_dir / "target_v4_physical_validity.csv", index=False)
    (out_dir / "target_metrics_v4.json").write_text(
        json.dumps(result.payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    meta_row = {
        "split": result.payload.get("split", ""),
        "metric_schema_version": result.payload.get("metric_schema_version", ""),
        "targetrow_metric_schema_version": result.payload.get(
            "targetrow_metric_schema_version", ""
        ),
        "primary_metric_name": result.payload.get("primary_metric_name", ""),
        "included_target_ids": result.payload.get("included_target_ids", ""),
        "excluded_target_ids": result.payload.get("excluded_target_ids", ""),
        "excluded_target_reasons": result.payload.get("excluded_target_reasons", ""),
        "n_targets_main_verified": result.payload.get("n_main_verified_targets", 0),
        "n_targets_formula_available": result.payload.get("n_targets_formula_available", 0),
        "n_targets_excluded": result.payload.get("n_targets_excluded", 0),
        LEGACY_ANSWER_FRACTION_METRIC: result.payload.get(LEGACY_ANSWER_FRACTION_METRIC),
    }
    (out_dir / "target_row_metric_meta.json").write_text(
        json.dumps(meta_row, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def augment_metrics_with_legacy_aliases(metrics: dict[str, Any]) -> dict[str, Any]:
    """Add legacy_* keys alongside existing H2/CO2 metric keys (non-destructive)."""
    out = dict(metrics)
    if "metric_answer_all_targets_mean_r2" in out:
        v = out["metric_answer_all_targets_mean_r2"]
        out.setdefault("legacy_answer_fraction_r2", v)
        out.setdefault("legacy_answer_fraction_macro_r2", v)
    if "metric_target_r2" not in out and "target_h2_r2_orig" in out:
        out["metric_target_r2"] = out["target_h2_r2_orig"]
        out["target_h2_r2"] = out["target_h2_r2_orig"]
    if "metric_tailgas_r2" not in out and "tailgas_co2_r2_orig" in out:
        out["metric_tailgas_r2"] = out["tailgas_co2_r2_orig"]
        out["tailgas_co2_r2"] = out["tailgas_co2_r2_orig"]
    pairs = (
        ("target_h2_r2", "legacy_target_h2_r2"),
        ("target_h2_r2_orig", "legacy_target_h2_r2"),
        ("metric_target_r2", "legacy_target_h2_r2"),
        ("metric_target_h2_r2", "legacy_target_h2_r2"),
        ("target_h2_mae", "legacy_target_h2_mae"),
        ("target_h2_mae_orig", "legacy_target_h2_mae"),
        ("metric_target_h2_mae", "legacy_target_h2_mae"),
        ("target_h2_rmse", "legacy_target_h2_rmse"),
        ("target_h2_rmse_orig", "legacy_target_h2_rmse"),
        ("metric_target_h2_rmse", "legacy_target_h2_rmse"),
        ("tailgas_co2_r2", "legacy_tailgas_co2_r2"),
        ("tailgas_co2_r2_orig", "legacy_tailgas_co2_r2"),
        ("metric_tailgas_r2", "legacy_tailgas_co2_r2"),
        ("metric_tailgas_co2_r2", "legacy_tailgas_co2_r2"),
        ("tailgas_co2_mae", "legacy_tailgas_co2_mae"),
        ("tailgas_co2_mae_orig", "legacy_tailgas_co2_mae"),
        ("metric_tailgas_co2_mae", "legacy_tailgas_co2_mae"),
        ("tailgas_co2_rmse", "legacy_tailgas_co2_rmse"),
        ("tailgas_co2_rmse_orig", "legacy_tailgas_co2_rmse"),
        ("metric_tailgas_co2_rmse", "legacy_tailgas_co2_rmse"),
    )
    for src, dst in pairs:
        if src in out and dst not in out:
            try:
                out[dst] = float(out[src])
            except (TypeError, ValueError):
                out[dst] = out[src]
    if "edge_all_frac_h2_r2_orig" in out:
        out["main_target_global_frac_h2_r2"] = out["edge_all_frac_h2_r2_orig"]
    if "target_v4_macro_r2_per_target_mean" not in out:
        for k, v in list(out.items()):
            if str(k).endswith("target_v4_macro_r2_per_target_mean"):
                out["target_v4_macro_r2_per_target_mean"] = v
                break
    return out


@dataclass
class _TargetV4AccumulatorBucket:
    sse: float = 0.0
    sae: float = 0.0
    smape_sum: float = 0.0
    sum_y: float = 0.0
    sum_y2: float = 0.0
    sum_abs_y: float = 0.0
    n: int = 0


class TargetV4EpochAccumulator:
    """Online accumulation of v4 target metrics during validation (scalars only)."""

    def __init__(self, specs_by_process: Mapping[str, Sequence[TargetSpecV4]]) -> None:
        self.specs_by_process = {k: list(v) for k, v in specs_by_process.items()}
        self.buckets: DefaultDict[tuple[str, str, str], _TargetV4AccumulatorBucket] = defaultdict(
            _TargetV4AccumulatorBucket
        )
        self.frac_buckets: DefaultDict[tuple[str, str, str], _TargetV4AccumulatorBucket] = defaultdict(
            _TargetV4AccumulatorBucket
        )
        self.split_name: str = "val"

    def update_batch(
        self,
        *,
        export,
        y_pred_orig,
        y_true_orig,
        y_edge_mask,
        edge_target_columns: Sequence[str],
        split_name: str = "val",
    ) -> None:
        import torch

        self.split_name = split_name
        cols = list(edge_target_columns)
        col_idx = {str(c): j for j, c in enumerate(cols)}
        mole_j = col_idx.get(MOLE_FLOW_PROPERTY)
        if mole_j is None:
            return
        n_edges = int(y_pred_orig.shape[0])
        if export is None or len(export.process_id) != n_edges:
            return
        ypo = y_pred_orig.detach().cpu().numpy()
        yto = y_true_orig.detach().cpu().numpy()
        mask = y_edge_mask.detach().cpu().numpy()

        for e in range(n_edges):
            if float(mask[e]) <= 0.0:
                continue
            pid = str(export.process_id[e])
            stream_norm = normalize_stream_key(export.main_data_stream_key[e])
            specs = self.specs_by_process.get(pid, [])
            if not specs:
                continue
            pred_mole = float(ypo[e, mole_j])
            true_mole = float(yto[e, mole_j])

            for spec in specs:
                if stream_norm != spec.required_stream_key:
                    continue
                if spec.canonical_edge_id and str(export.canonical_edge_id[e]) != spec.canonical_edge_id:
                    continue
                frac_col = SPECIES_TO_FRAC_PROPERTY.get(spec.target_species)
                if frac_col is None:
                    continue
                fj = col_idx.get(frac_col)
                if fj is None:
                    continue
                sf, err = _parse_scale_factor(spec.scale_factor)
                if err or sf is None:
                    continue
                pred_frac = float(ypo[e, fj])
                true_frac = float(yto[e, fj])
                key = (split_name, pid, spec.target_id)

                if not (math.isnan(pred_frac) or math.isnan(true_frac)):
                    err_f = pred_frac - true_frac
                    bf = self.frac_buckets[key]
                    bf.sse += err_f * err_f
                    bf.sae += abs(err_f)
                    den_f = abs(true_frac) + abs(pred_frac) + EPS
                    bf.smape_sum += 100.0 * 2.0 * abs(err_f) / den_f
                    bf.sum_y += true_frac
                    bf.sum_y2 += true_frac * true_frac
                    bf.sum_abs_y += abs(true_frac)
                    bf.n += 1

                if not (
                    math.isnan(pred_mole)
                    or math.isnan(true_mole)
                    or math.isnan(pred_frac)
                    or math.isnan(true_frac)
                ):
                    pred_t = sf * pred_mole * pred_frac
                    true_t = sf * true_mole * true_frac
                    err_v = pred_t - true_t
                    b = self.buckets[key]
                    b.sse += err_v * err_v
                    b.sae += abs(err_v)
                    den = abs(true_t) + abs(pred_t) + EPS
                    b.smape_sum += 100.0 * 2.0 * abs(err_v) / den
                    b.sum_y += true_t
                    b.sum_y2 += true_t * true_t
                    b.sum_abs_y += abs(true_t)
                    b.n += 1

    def finalize_scalars(self) -> dict[str, float]:
        def _answer_r2(sse: float, sum_y: float, sum_y2: float, n: float) -> float:
            rel_floor = 1e-4
            if n < 1.0:
                return 0.0
            if n < 2.0:
                if sse <= 1e-18:
                    return 1.0
                ymax = max(abs(float(sum_y)), 1e-30)
                return max(0.0, min(1.0, 1.0 - float(sse) / (ymax * ymax + 1e-30)))
            mean_y = sum_y / n
            sst = float(sum_y2) - float(n) * mean_y * mean_y
            scale = max(abs(float(sum_y2) / n), mean_y * mean_y, 1e-30)
            if not math.isfinite(sst):
                sst = 0.0
            sst_eff = max(float(sst), rel_floor * scale, 1e-30)
            r2 = 1.0 - float(sse) / sst_eff
            if not math.isfinite(r2):
                return 0.0
            return max(-1.0, min(1.0, r2))

        per_target_r2: list[float] = []
        per_target_r2_main_verified: list[float] = []
        per_target_r2_formula: list[float] = []
        per_target_score: list[float] = []
        per_target_mae: list[float] = []
        per_target_rmse: list[float] = []
        species_r2: DefaultDict[str, list[float]] = defaultdict(list)
        species_score: DefaultDict[str, list[float]] = defaultdict(list)
        process_macros: DefaultDict[str, list[float]] = defaultdict(list)
        process_macros_main: DefaultDict[str, list[float]] = defaultdict(list)
        process_macros_formula: DefaultDict[str, list[float]] = defaultdict(list)
        process_scores: DefaultDict[str, list[float]] = defaultdict(list)
        n_eval = 0
        n_skip = 0

        provenance = load_provenance_policy()
        prov_by_tid: dict[str, dict[str, Any]] = {}
        if not provenance.empty and "target_id" in provenance.columns:
            for _, pr in provenance.iterrows():
                prov_by_tid[str(pr["target_id"])] = pr.to_dict()

        specs_total = sum(len(v) for v in self.specs_by_process.values())
        seen_targets: set[tuple[str, str]] = set()

        for (split, pid, tid), b in self.buckets.items():
            seen_targets.add((pid, tid))
            if b.n < 1:
                n_skip += 1
                continue
            mae = b.sae / b.n
            rmse = math.sqrt(b.sse / b.n) if b.n > 0 else 0.0
            r2 = _answer_r2(b.sse, b.sum_y, b.sum_y2, float(b.n))
            mean_abs_y = b.sum_abs_y / b.n if b.n > 0 else 0.0
            score = max(0.0, min(1.0, 1.0 - (rmse / (mean_abs_y + EPS))))
            per_target_mae.append(mae)
            per_target_rmse.append(rmse)
            per_target_r2.append(float(r2))
            per_target_score.append(float(score))
            process_macros[pid].append(float(r2))
            process_scores[pid].append(float(score))
            pr = prov_by_tid.get(tid, {})
            inc_main = bool(pr.get("include_in_main_verified_macro", True)) if pr else True
            inc_formula = bool(pr.get("include_in_formula_macro", inc_main)) if pr else True
            if inc_main:
                per_target_r2_main_verified.append(float(r2))
                process_macros_main[pid].append(float(r2))
            if inc_formula:
                per_target_r2_formula.append(float(r2))
                process_macros_formula[pid].append(float(r2))
            n_eval += 1
            for spec in self.specs_by_process.get(pid, []):
                if spec.target_id == tid:
                    species_r2[spec.target_species].append(float(r2))
                    species_score[spec.target_species].append(float(score))
                    break

        n_skip = max(0, specs_total - len(seen_targets))

        def _mean(xs: list[float]) -> float:
            fin = [x for x in xs if math.isfinite(x)]
            return float(sum(fin) / len(fin)) if fin else float("nan")

        proc_macro_vals = [_mean(v) for v in process_macros.values()]
        proc_macro_main_vals = [_mean(v) for v in process_macros_main.values()]
        proc_macro_formula_vals = [_mean(v) for v in process_macros_formula.values()]
        proc_score_vals = [_mean(v) for v in process_scores.values()]
        macro_main = (
            _mean(proc_macro_main_vals)
            if proc_macro_main_vals
            else (_mean(per_target_r2_main_verified) if per_target_r2_main_verified else float("nan"))
        )
        macro_target_balanced_main = (
            _mean(per_target_r2_main_verified) if per_target_r2_main_verified else float("nan")
        )
        macro_formula = (
            _mean(per_target_r2_formula) if per_target_r2_formula else macro_main
        )
        macro_process_balanced_formula = (
            _mean(proc_macro_formula_vals)
            if proc_macro_formula_vals
            else macro_formula
        )
        macro_all = _mean(proc_macro_vals) if proc_macro_vals else _mean(per_target_r2)
        if provenance.empty:
            macro_primary = macro_all
        else:
            macro_primary = macro_main if math.isfinite(macro_main) else float("nan")

        frac_per_target_r2_main: list[float] = []
        frac_process_macros_main: DefaultDict[str, list[float]] = defaultdict(list)
        frac_seen: set[tuple[str, str]] = set()
        for (split, pid, tid), b in self.frac_buckets.items():
            frac_seen.add((pid, tid))
            if b.n < 1:
                continue
            r2f = _answer_r2(b.sse, b.sum_y, b.sum_y2, float(b.n))
            if not math.isfinite(r2f):
                continue
            pr = prov_by_tid.get(tid, {})
            inc_main = bool(pr.get("include_in_main_verified_macro", True)) if pr else True
            if inc_main:
                frac_per_target_r2_main.append(float(r2f))
                frac_process_macros_main[pid].append(float(r2f))

        frac_proc_main_vals = [_mean(v) for v in frac_process_macros_main.values()]
        frac_macro_main = (
            _mean(frac_proc_main_vals)
            if frac_proc_main_vals
            else (_mean(frac_per_target_r2_main) if frac_per_target_r2_main else float("nan"))
        )
        frac_target_balanced_main = (
            _mean(frac_per_target_r2_main) if frac_per_target_r2_main else float("nan")
        )
        frac_macro_formula = frac_target_balanced_main
        frac_process_balanced_formula = frac_macro_main

        meta_rows: list[dict[str, Any]] = []
        for (split, pid, tid), b in self.frac_buckets.items():
            if b.n < 1:
                continue
            pr = prov_by_tid.get(tid, {})
            r2f = _answer_r2(b.sse, b.sum_y, b.sum_y2, float(b.n))
            meta_rows.append(
                {
                    "target_id": tid,
                    "process_id": pid,
                    "r2": r2f,
                    "included_in_primary": bool(
                        pr.get("include_in_main_verified_macro", True) if pr else True
                    )
                    and math.isfinite(r2f),
                    "include_in_main_verified_macro": bool(pr.get("include_in_main_verified_macro", True))
                    if pr
                    else True,
                    "include_in_formula_macro": bool(pr.get("include_in_formula_macro", True)) if pr else True,
                    "exclusion_reason": pr.get("exclusion_reason", ""),
                }
            )
        from .target_row_primary_metrics import extract_target_exclusion_metadata

        prov_meta = extract_target_exclusion_metadata(
            pd.DataFrame(meta_rows), use_included_in_primary=True
        )

        out: dict[str, Any] = {
            f"val_{EVAL_PRIMARY_FRAC_R2_BY_PROCESS}": frac_macro_main,
            f"val_{EVAL_PRIMARY_FRAC_R2_BY_TARGET}": frac_target_balanced_main,
            f"val_{FRAC_R2_SHORT_ALIAS}": frac_target_balanced_main,
            f"val_{EVAL_PRIMARY_FRAC_R2_FORMULA_BY_PROCESS}": frac_process_balanced_formula,
            f"val_{EVAL_PRIMARY_FRAC_R2_FORMULA_BY_TARGET}": frac_macro_formula,
            f"val_{EVAL_SECONDARY_AMOUNT_R2_BY_PROCESS}": macro_main,
            f"val_{EVAL_SECONDARY_AMOUNT_R2_BY_TARGET}": macro_target_balanced_main,
            f"val_{AMOUNT_METRIC_ALIAS}": macro_target_balanced_main,
            f"val_{EVAL_SECONDARY_AMOUNT_R2_FORMULA_BY_PROCESS}": macro_process_balanced_formula,
            f"val_{EVAL_SECONDARY_AMOUNT_R2_FORMULA_BY_TARGET}": macro_formula,
            f"val_eval_amount_r2_all_target_rows": macro_all,
            f"val_eval_target_row_r2_macro": frac_macro_main if math.isfinite(frac_macro_main) else macro_primary,
            f"val_n_main_verified_targets": float(len(frac_per_target_r2_main)),
            f"val_n_included_targets": float(len(frac_per_target_r2_main)),
            f"val_n_excluded_targets": float(
                max(0, len(frac_seen) - len(frac_per_target_r2_main))
            ),
            f"val_n_amount_main_verified_targets": float(len(per_target_r2_main_verified)),
            f"val_eval_target_row_relative_accuracy_mean": _mean(proc_score_vals)
            if proc_score_vals
            else _mean(per_target_score),
            f"val_eval_target_row_mae_mean": _mean(per_target_mae),
            f"val_eval_target_row_rmse_mean": _mean(per_target_rmse),
            f"val_{EVAL_FRAC_R2_MEAN_H2}": _mean(species_r2.get("H2", [])),
            f"val_{EVAL_FRAC_R2_MEAN_CO2}": _mean(species_r2.get("CO2", [])),
            f"val_{EVAL_FRAC_R2_MEAN_H2O}": _mean(species_r2.get("H2O", [])),
            f"val_{EVAL_TARGET_ROWS_EVALUATED}": float(len(seen_targets)),
            f"val_{EVAL_TARGET_ROWS_SKIPPED}": float(n_skip),
            f"val_{EVAL_TARGET_ROWS_TOTAL}": float(specs_total),
        }
        for mk, mv in prov_meta.items():
            if mk in out:
                continue
            if isinstance(mv, (int, float)) and not isinstance(mv, bool):
                out[f"val_{mk}"] = float(mv)
            elif isinstance(mv, str):
                out[f"val_{mk}"] = mv
        prefix = f"{self.split_name}/"
        for k, v in list(out.items()):
            if k.startswith("val_"):
                out[prefix + k[4:]] = v
        return out


def build_target_v4_epoch_accumulator(targets_path: Path) -> TargetV4EpochAccumulator:
    return TargetV4EpochAccumulator(load_target_stream_targets_v4(targets_path))
