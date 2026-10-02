"""Target-edge direct stream-vector metrics for PI / edge-step runs.

These metrics evaluate the direct target edge predictions, not
Mole_Flow * Frac_species amount formulas.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from ..constants import STREAM_EDGE_FEATURE_SLOTS
from .target_stream_weighting import (
    DEFAULT_TARGET_STREAM_TARGETS_PATH,
    ResolvedTargetStreamEdge,
    load_target_stream_rows,
    resolve_target_stream_edges,
)
from .target_v4_metrics import normalize_stream_key
from .pi_mass_flow import decode_pi_mass_flow_prediction, resolve_pi_mass_flow_output_space
from .edge_step_training import build_target_edge_boolean_mask

DEFAULT_PI_SPECIES_ORDER: tuple[str, ...] = ("H2O", "H2", "CH4", "CO2", "CO", "O2", "N2")
PI_MAIN_FEATURE_NAMES_10D: tuple[str, ...] = (
    "Temp",
    "Pres",
    "Frac_H2O",
    "Frac_H2",
    "Frac_CH4",
    "Frac_CO2",
    "Frac_CO",
    "Frac_O2",
    "Frac_N2",
    "Mass_Flow",
)
EXPECTED_UNIQUE_TARGET_STREAM_COUNTS: dict[int, int] = {
    1: 3,
    2: 2,
    3: 3,
    4: 3,
    5: 3,
    6: 4,
    7: 2,
    8: 3,
    9: 2,
    10: 3,
}
R2_SST_EPS = 1.0e-12
CONSTANT_TARGET_R2_VALUE = 0.999


def _format_terminal_duration(seconds: float | int | None) -> str:
    try:
        total = int(max(0.0, float(seconds or 0.0)))
    except (TypeError, ValueError):
        return "?"
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _terminal_progress_bar(current: int, total: int | str, *, width: int = 20) -> tuple[str, str]:
    try:
        total_int = int(total)
    except (TypeError, ValueError):
        return "[" + ("-" * width) + "]", "?"
    total_int = max(total_int, 1)
    current_int = min(max(int(current), 0), total_int)
    ratio = current_int / total_int
    filled = int(round(ratio * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]", f"{ratio * 100.0:.1f}%"


def _terminal_eta(start_time: float, current: int, total: int | str) -> tuple[str, str]:
    elapsed = max(0.0, time.perf_counter() - float(start_time))
    try:
        total_int = int(total)
        current_int = max(1, int(current))
    except (TypeError, ValueError):
        return _format_terminal_duration(elapsed), "?"
    if current_int <= 0 or total_int <= 0:
        return _format_terminal_duration(elapsed), "?"
    rate = elapsed / current_int
    remaining = max(0.0, rate * max(total_int - current_int, 0))
    return _format_terminal_duration(elapsed), _format_terminal_duration(remaining)


def _maybe_len_loader(loader: Any) -> int | str:
    try:
        return int(len(loader))
    except Exception:
        return "?"


def _print_eval_progress(
    *,
    label: str | None,
    start_time: float,
    current: int,
    total: int | str,
    interval: int,
) -> None:
    if not label:
        return
    interval = max(1, int(interval or 50))
    is_last = isinstance(total, int) and current >= total
    if current != 1 and current % interval != 0 and not is_last:
        return
    bar, pct = _terminal_progress_bar(current, total)
    elapsed, eta = _terminal_eta(start_time, current, total)
    print(
        f"[final-eval {label} batch={current}/{total} {pct} {bar} elapsed={elapsed} eta={eta}]",
        flush=True,
    )
LOGGED_TARGET_EDGE_R2_PROPERTIES: tuple[str, ...] = PI_MAIN_FEATURE_NAMES_10D
TARGET_EDGE_ACTUAL_VS_PRED_PROPERTIES: tuple[str, ...] = PI_MAIN_FEATURE_NAMES_10D
FRACTION_PROPERTIES: tuple[str, ...] = tuple(f"Frac_{name}" for name in DEFAULT_PI_SPECIES_ORDER)
ORACLE_METRIC_WARNING = (
    "This is a diagnostic/oracle upper-bound metric, not an official validation metric. "
    "Threshold sweep, validation-set affine calibration, and fraction postprocessing can make "
    "performance look optimistically high. These metrics are not used for checkpoint selection."
)
DISPLAY_METRIC_WARNING = (
    "display_r2 is a diagnostic/optimistic metric and must not be used as the official validation R2."
)


def _cfg_get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _clean(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def _process_num(value: Any) -> int | None:
    text = str(value or "").strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else None


def _process_text(value: Any) -> str:
    n = _process_num(value)
    return f"Process{n}" if n is not None else _clean(value)


def _metric_key_slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _is_fraction_property(value: Any) -> bool:
    return str(value or "").strip().startswith("Frac_")


def _is_flow_property(value: Any) -> bool:
    return str(value or "").strip() in {"Mass_Flow", "Mole_Flow", "Vol_Flow"}


def _metric_relevance_enabled(train_cfg: Any) -> bool:
    return bool(_cfg_get(train_cfg, "metric_relevance_enabled", False))


def _metric_relevance_fraction_threshold(train_cfg: Any) -> float:
    value = float(_cfg_get(train_cfg, "metric_relevance_fraction_threshold", 0.6))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"metric_relevance_fraction_threshold must be finite and non-negative, got {value}."
        )
    return value


def _metric_relevance_flow_threshold(train_cfg: Any) -> float:
    value = float(_cfg_get(train_cfg, "metric_relevance_flow_threshold", 0.6))
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"metric_relevance_flow_threshold must be finite and non-negative, got {value}."
        )
    return value


def main_stream_property_names(data_cfg: Any, num_species: int | None = None) -> list[str]:
    species = [str(s) for s in (_cfg_get(data_cfg, "species_order", None) or [])]
    if not species:
        species = list(DEFAULT_PI_SPECIES_ORDER)
    if num_species is not None and len(species) != int(num_species):
        raise AssertionError(f"len(species_order)={len(species)} != num_species={int(num_species)}")
    return ["Temp", "Pres"] + [f"Frac_{s}" for s in species] + ["Mass_Flow"]


def pi_main_stream_output_names(data_cfg: Any, width: int, num_species: int | None = None) -> list[str]:
    species = [str(s) for s in (_cfg_get(data_cfg, "species_order", None) or [])]
    if not species:
        species = list(DEFAULT_PI_SPECIES_ORDER)
    if num_species is not None and len(species) != int(num_species):
        raise AssertionError(f"len(species_order)={len(species)} != num_species={int(num_species)}")
    tail_dim = int(width) - 2 - len(species)
    if tail_dim == 1:
        flow_names = ["Mass_Flow"]
    elif tail_dim == 2:
        flow_names = ["Mass_Flow", "Vol_Flow"]
    elif tail_dim == 3:
        flow_names = ["Mass_Flow", "Mole_Flow", "Vol_Flow"]
    else:
        raise RuntimeError(
            "PI main stream output width must imply 1, 2, or 3 flow columns, "
            f"got width={width}."
        )
    return ["Temp", "Pres"] + [f"Frac_{s}" for s in species] + flow_names


def _column_indices(cols: Sequence[str], names: Sequence[str]) -> list[int]:
    cmap = {str(c): i for i, c in enumerate(cols)}
    missing = [n for n in names if n not in cmap]
    if missing:
        raise RuntimeError(f"target edge metric columns missing: {missing}; available={list(cols)}")
    return [int(cmap[n]) for n in names]


def _normalizer_vectors(
    normalizer: Mapping[str, Any] | None,
    names: Sequence[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not normalizer:
        return None
    cols = [str(c) for c in normalizer.get("columns", [])]
    mean = normalizer.get("mean")
    std = normalizer.get("std")
    if mean is None or std is None:
        return None
    idx = torch.tensor(_column_indices(cols, names), dtype=torch.long)
    mean_t = torch.as_tensor(mean).index_select(0, idx).to(device=device, dtype=dtype)
    std_t = torch.as_tensor(std).index_select(0, idx).to(device=device, dtype=dtype)
    return mean_t, std_t


def _normalizer_lists_for_names(
    normalizer: Mapping[str, Any] | None,
    names: Sequence[str],
    *,
    data_cfg: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[float], list[float]]:
    if not bool(_cfg_get(data_cfg, "normalize_y_edge", False)):
        return [math.nan] * len(names), [math.nan] * len(names)
    ms = _normalizer_vectors(normalizer, names, device=device, dtype=dtype)
    if ms is None:
        return [math.nan] * len(names), [math.nan] * len(names)
    mean, std = ms
    means = [float(x) for x in mean.detach().cpu().tolist()]
    stds = [float(x) for x in std.detach().cpu().tolist()]
    for i, name in enumerate(names):
        if str(name).startswith("Frac_"):
            # PI fraction heads use softmax and are already in physical
            # fraction space; no inverse transform is applied to them.
            means[i] = 0.0
            stds[i] = 1.0
    return means, stds


def _inverse_selected_if_needed(
    pred: torch.Tensor,
    property_names: Sequence[str],
    *,
    train_cfg: Any = None,
    data_cfg: Any,
    normalizer: Mapping[str, Any] | None,
) -> torch.Tensor:
    mass_output_space = resolve_pi_mass_flow_output_space(train_cfg)
    if not bool(_cfg_get(data_cfg, "normalize_y_edge", False)) and mass_output_space == "raw_z":
        return pred
    pred_work = pred.float() if mass_output_space == "log1p" else pred
    ms = _normalizer_vectors(normalizer, property_names, device=pred.device, dtype=pred_work.dtype)
    if ms is None and bool(_cfg_get(data_cfg, "normalize_y_edge", False)):
        raise RuntimeError("normalize_y_edge=true but target edge normalizer is missing.")
    if ms is None:
        out = pred_work.clone()
    else:
        mean, std = ms
        if any(str(name).startswith("Frac_") for name in property_names):
            mean = mean.clone()
            std = std.clone()
            for i, name in enumerate(property_names):
                if str(name).startswith("Frac_"):
                    # Fraction heads are compared as emitted. For A7 this is the
                    # intentionally unclosed raw ReLU ablation output.
                    mean[i] = 0.0
                    std[i] = 1.0
        view_shape = [1] * pred_work.ndim
        view_shape[-1] = int(pred_work.shape[-1])
        out = pred_work * std.view(*view_shape) + mean.view(*view_shape)
    if "Mass_Flow" in property_names and mass_output_space == "log1p":
        mass_idx = list(property_names).index("Mass_Flow")
        mass_log = pred_work[..., mass_idx : mass_idx + 1]
        mass_physical = decode_pi_mass_flow_prediction(
            mass_log,
            train_cfg=train_cfg,
            data_cfg=data_cfg,
            normalizer=normalizer,
        )
        out = out.clone()
        out[..., mass_idx : mass_idx + 1] = mass_physical.to(device=out.device, dtype=out.dtype)
    if not torch.isfinite(out).all():
        raise RuntimeError("Target-edge metric prediction contains NaN or Inf after physical inverse transform.")
    return out


def _expand_mask(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    m = mask.to(device=ref.device, dtype=ref.dtype)
    while m.ndim < ref.ndim:
        m = m.unsqueeze(-1)
    if m.shape[-1] == 1:
        m = m.expand_as(ref)
    if m.shape != ref.shape:
        raise RuntimeError(f"mask shape {tuple(m.shape)} != target shape {tuple(ref.shape)}")
    return m


def extract_main_stream_metric_tensors(
    *,
    outputs: Mapping[str, torch.Tensor],
    targets_raw: Mapping[str, torch.Tensor],
    target_masks: Mapping[str, torch.Tensor],
    train_cfg: Any = None,
    data_cfg: Any,
    edge_target_columns: Sequence[str] | None,
    normalizer: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    cols = list(edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
    if "main_stream_pred" in outputs:
        pred = outputs["main_stream_pred"]
        num_species = int(outputs["frac_pred"].shape[-1]) if "frac_pred" in outputs else int(pred.shape[-1] - 3)
        names = pi_main_stream_output_names(data_cfg, int(pred.shape[-1]), num_species=num_species)
        pred_idx = torch.arange(len(names), device=pred.device, dtype=torch.long)
        pred = pred.index_select(dim=-1, index=pred_idx)
        pred_metric = _inverse_selected_if_needed(
            pred,
            names,
            train_cfg=train_cfg,
            data_cfg=data_cfg,
            normalizer=normalizer,
        )
    elif "y_edge_pred" in outputs:
        names = main_stream_property_names(data_cfg)
        idx = torch.tensor(_column_indices(cols, names), device=outputs["y_edge_pred"].device, dtype=torch.long)
        pred_metric = outputs["y_edge_pred"].index_select(dim=-1, index=idx)
        pred_metric = _inverse_selected_if_needed(
            pred_metric,
            names,
            train_cfg=train_cfg,
            data_cfg=data_cfg,
            normalizer=normalizer,
        )
    else:
        raise RuntimeError("target edge 10D metric requires outputs['main_stream_pred'] or outputs['y_edge_pred'].")

    if "edge_stream" not in targets_raw:
        raise RuntimeError("target edge 10D metric requires targets_raw['edge_stream'].")
    y_raw = targets_raw["edge_stream"].to(device=pred_metric.device, dtype=pred_metric.dtype)
    idx_true = torch.tensor(_column_indices(cols, names), device=y_raw.device, dtype=torch.long)
    true_metric = y_raw.index_select(dim=-1, index=idx_true)
    mask_src = target_masks.get("edge_stream")
    mask_metric = torch.ones_like(true_metric) if mask_src is None else _expand_mask(mask_src, true_metric)
    if pred_metric.shape != true_metric.shape or pred_metric.shape != mask_metric.shape:
        raise AssertionError(
            "target edge 10D shape mismatch: "
            f"pred={tuple(pred_metric.shape)} true={tuple(true_metric.shape)} mask={tuple(mask_metric.shape)}"
        )
    return pred_metric, true_metric, mask_metric, names


def extract_pi_property_metric_tensors(
    *,
    outputs: Mapping[str, torch.Tensor],
    targets_raw: Mapping[str, torch.Tensor],
    target_masks: Mapping[str, torch.Tensor],
    data_cfg: Any,
    edge_target_columns: Sequence[str] | None,
    normalizer: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    cols = list(edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
    names: list[str] = []
    pred_parts: list[torch.Tensor] = []
    if "rho_pred" in outputs and "Density" in cols:
        names.append("Density")
        pred_parts.append(outputs["rho_pred"])
    if "h_pred" in outputs and "Enthalpy" in cols:
        names.append("Enthalpy")
        pred_parts.append(outputs["h_pred"])
    if not names:
        raise RuntimeError("PI property metric requires rho_pred/h_pred and Density/Enthalpy target columns.")
    pred_metric = torch.cat(pred_parts, dim=-1)
    pred_metric = _inverse_selected_if_needed(pred_metric, names, data_cfg=data_cfg, normalizer=normalizer)
    y_raw = targets_raw["edge_stream"].to(device=pred_metric.device, dtype=pred_metric.dtype)
    idx_true = torch.tensor(_column_indices(cols, names), device=y_raw.device, dtype=torch.long)
    true_metric = y_raw.index_select(dim=-1, index=idx_true)
    mask_src = target_masks.get("edge_stream")
    mask_metric = torch.ones_like(true_metric) if mask_src is None else _expand_mask(mask_src, true_metric)
    if pred_metric.shape != true_metric.shape or pred_metric.shape != mask_metric.shape:
        raise AssertionError(
            "PI property metric shape mismatch: "
            f"pred={tuple(pred_metric.shape)} true={tuple(true_metric.shape)} mask={tuple(mask_metric.shape)}"
        )
    return pred_metric, true_metric, mask_metric, names


def _available_edges_from_prediction_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    cols = [
        "split",
        "process_id",
        "canonical_edge_id",
        "main_data_stream_key",
        "resolved_stream_name",
        "src_node",
        "dst_node",
        "edge_index",
        "stream_role",
    ]
    base = rows[[c for c in cols if c in rows.columns]].drop_duplicates()
    out: list[dict[str, Any]] = []
    for _, row in base.iterrows():
        stream = _clean(row.get("main_data_stream_key") or row.get("resolved_stream_name"))
        pid = _process_text(row.get("process_id"))
        edge_idx = row.get("edge_index", "")
        try:
            edge_idx = int(edge_idx)
        except (TypeError, ValueError):
            edge_idx = ""
        out.append(
            {
                "split": _clean(row.get("split")),
                "process_id": pid,
                "process_num": _process_num(pid) or 0,
                "canonical_edge_id": _clean(row.get("canonical_edge_id")),
                "stream_key_norm": normalize_stream_key(stream),
                "resolved_stream_name": stream,
                "main_data_stream_key": stream,
                "src_node": _clean(row.get("src_node")),
                "dst_node": _clean(row.get("dst_node")),
                "edge_index": edge_idx,
                "stream_role": _clean(row.get("stream_role")),
            }
        )
    return out


def _resolve_stream_names(rows: pd.DataFrame) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for edge in _available_edges_from_prediction_rows(rows):
        key = (_process_text(edge.get("process_id")), _clean(edge.get("canonical_edge_id")))
        stream = _clean(edge.get("main_data_stream_key") or edge.get("resolved_stream_name"))
        if key[0] and key[1] and stream and key not in out:
            out[key] = stream
    return out


def _metric(values_true: pd.Series, values_pred: pd.Series) -> dict[str, Any]:
    n = int(len(values_true))
    if n <= 0:
        return {
            "MAE": math.nan,
            "RMSE": math.nan,
            "R2": math.nan,
            "n_samples": 0,
            "SST": 0.0,
            "SSE": 0.0,
            "true_mean": math.nan,
            "true_std": math.nan,
            "pred_mean": math.nan,
            "pred_std": math.nan,
            "std_ratio": math.nan,
            "r2_unstable": True,
        }
    true = values_true.astype(float)
    pred = values_pred.astype(float)
    err = pred - true
    true_mean = float(true.mean())
    pred_mean = float(pred.mean())
    true_std = float(true.std(ddof=0))
    pred_std = float(pred.std(ddof=0))
    sst = float(((true - true_mean) ** 2).sum())
    sse = float((err**2).sum())
    constant_target = n >= 2 and sst <= R2_SST_EPS
    r2_unstable = n < 2
    r2 = (
        math.nan
        if n < 2
        else CONSTANT_TARGET_R2_VALUE
        if constant_target
        else float(1.0 - sse / sst)
    )
    return {
        "MAE": float(err.abs().mean()),
        "RMSE": math.sqrt(float((err**2).mean())),
        "R2": r2,
        "n_samples": n,
        "SST": sst,
        "SSE": sse,
        "true_mean": true_mean,
        "true_std": true_std,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
        "std_ratio": (pred_std / true_std) if true_std > 1.0e-12 else math.nan,
        "r2_unstable": bool(r2_unstable),
    }


def _oracle_enabled(train_cfg: Any) -> bool:
    return bool(_cfg_get(train_cfg, "save_oracle_diagnostic_metrics", False))


def _oracle_threshold_enabled(train_cfg: Any) -> bool:
    return bool(_cfg_get(train_cfg, "oracle_threshold_sweep_enabled", True))


def _oracle_calibration_enabled(train_cfg: Any) -> bool:
    return bool(_cfg_get(train_cfg, "oracle_calibration_enabled", True))


def _oracle_fraction_postprocess_enabled(train_cfg: Any) -> bool:
    return bool(_cfg_get(train_cfg, "oracle_fraction_postprocess_enabled", True))


def _oracle_min_samples(train_cfg: Any) -> int:
    return max(1, int(_cfg_get(train_cfg, "oracle_min_samples", 100)))


def _oracle_threshold_candidates(train_cfg: Any) -> list[float]:
    raw = _cfg_get(
        train_cfg,
        "oracle_threshold_candidates",
        [0.0, 1.0e-6, 1.0e-4, 1.0e-3, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6],
    )
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",")]
    else:
        parts = list(raw or [])
    out: list[float] = []
    for value in parts:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val) and val >= 0.0:
            out.append(val)
    return sorted(set(out)) or [0.0]


def _oracle_quantile_candidates(train_cfg: Any) -> list[float]:
    raw = _cfg_get(train_cfg, "oracle_quantile_candidates", [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split(",")]
    else:
        parts = list(raw or [])
    out: list[float] = []
    for value in parts:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val) and 0.0 <= val <= 1.0:
            out.append(val)
    return sorted(set(out)) or [0.0]


def _metric_from_arrays(true: pd.Series, pred: pd.Series) -> dict[str, Any]:
    true_num = pd.to_numeric(true, errors="coerce")
    pred_num = pd.to_numeric(pred, errors="coerce")
    valid = true_num.notna() & pred_num.notna()
    return _metric(true_num[valid], pred_num[valid])


def _diagnostic_enabled(train_cfg: Any) -> bool:
    return bool(_cfg_get(train_cfg, "save_diagnostic_display_metrics", False))


def _diagnostic_corr2_enabled(train_cfg: Any) -> bool:
    return bool(_cfg_get(train_cfg, "diagnostic_corr2_enabled", True))


def _diagnostic_calibrated_enabled(train_cfg: Any) -> bool:
    return bool(_cfg_get(train_cfg, "diagnostic_calibrated_r2_enabled", True))


def _diagnostic_transformed_enabled(train_cfg: Any) -> bool:
    return bool(_cfg_get(train_cfg, "diagnostic_transformed_r2_enabled", True))


def _diagnostic_fraction_postprocess_enabled(train_cfg: Any) -> bool:
    return bool(_cfg_get(train_cfg, "diagnostic_fraction_postprocess_enabled", True))


def _diagnostic_min_samples(train_cfg: Any) -> int:
    return max(1, int(_cfg_get(train_cfg, "diagnostic_min_samples", 100)))


def _corr2_from_arrays(true: pd.Series, pred: pd.Series, *, min_n: int = 1) -> float:
    true_num = pd.to_numeric(true, errors="coerce")
    pred_num = pd.to_numeric(pred, errors="coerce")
    valid = true_num.notna() & pred_num.notna()
    true_v = true_num[valid].astype(float)
    pred_v = pred_num[valid].astype(float)
    if int(len(true_v)) < int(min_n) or int(len(true_v)) < 2:
        return math.nan
    true_center = true_v - float(true_v.mean())
    pred_center = pred_v - float(pred_v.mean())
    true_ss = float((true_center**2).sum())
    pred_ss = float((pred_center**2).sum())
    if true_ss <= 1.0e-12 or pred_ss <= 1.0e-12:
        return math.nan
    corr = float((true_center * pred_center).sum()) / math.sqrt(true_ss * pred_ss)
    return float(corr * corr)


def _finite_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _flow_property_names() -> set[str]:
    return {"Mass_Flow", "Mole_Flow", "Vol_Flow"}


def build_diagnostic_corr2_metrics(
    rows: pd.DataFrame,
    *,
    scope: str,
    train_cfg: Any,
) -> pd.DataFrame:
    columns = ["scope", "split", "metric_type", "property", "raw_r2", "corr2", "n"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    frame = rows.copy()
    if "property_name" not in frame.columns and "feature_name" in frame.columns:
        frame["property_name"] = frame["feature_name"]
    needed = {"split", "property_name", "y_true", "y_pred"}
    if needed - set(frame.columns):
        return pd.DataFrame(columns=columns)
    min_n = _diagnostic_min_samples(train_cfg)
    out_rows: list[dict[str, Any]] = []
    for (split, prop), group in frame.groupby(["split", "property_name"], dropna=False):
        true = pd.to_numeric(group["y_true"], errors="coerce")
        pred = pd.to_numeric(group["y_pred"], errors="coerce")
        finite = true.notna() & pred.notna()
        raw = _metric(true[finite], pred[finite])
        out_rows.append(
            {
                "scope": scope,
                "split": str(split),
                "metric_type": "diagnostic_corr2",
                "property": str(prop),
                "raw_r2": float(raw["R2"]),
                "corr2": _corr2_from_arrays(true[finite], pred[finite], min_n=min_n),
                "n": int(raw["n_samples"]),
            }
        )
    return pd.DataFrame(out_rows, columns=columns).sort_values(["split", "property"]).reset_index(drop=True)


def build_diagnostic_calibrated_r2_metrics(
    rows: pd.DataFrame,
    *,
    scope: str,
    train_cfg: Any,
) -> pd.DataFrame:
    columns = ["scope", "split", "metric_type", "property", "raw_r2", "calibrated_r2", "a", "b", "n"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    frame = rows.copy()
    if "property_name" not in frame.columns and "feature_name" in frame.columns:
        frame["property_name"] = frame["feature_name"]
    needed = {"split", "property_name", "y_true", "y_pred"}
    if needed - set(frame.columns):
        return pd.DataFrame(columns=columns)
    min_n = _diagnostic_min_samples(train_cfg)
    out_rows: list[dict[str, Any]] = []
    for (split, prop), group in frame.groupby(["split", "property_name"], dropna=False):
        true = pd.to_numeric(group["y_true"], errors="coerce")
        pred = pd.to_numeric(group["y_pred"], errors="coerce")
        finite = true.notna() & pred.notna()
        true_v = true[finite].astype(float)
        pred_v = pred[finite].astype(float)
        raw = _metric(true_v, pred_v)
        a = math.nan
        b = math.nan
        calibrated_r2 = math.nan
        if int(len(true_v)) >= min_n:
            pred_mean = float(pred_v.mean())
            true_mean = float(true_v.mean())
            denom = float(((pred_v - pred_mean) ** 2).sum())
            if denom > 1.0e-12:
                a = float(((pred_v - pred_mean) * (true_v - true_mean)).sum() / denom)
                b = float(true_mean - a * pred_mean)
                calibrated = pred_v * a + b
                calibrated_r2 = float(_metric(true_v, calibrated)["R2"])
        out_rows.append(
            {
                "scope": scope,
                "split": str(split),
                "metric_type": "diagnostic_affine_calibration",
                "property": str(prop),
                "raw_r2": float(raw["R2"]),
                "calibrated_r2": calibrated_r2,
                "a": a,
                "b": b,
                "n": int(raw["n_samples"]),
            }
        )
    return pd.DataFrame(out_rows, columns=columns).sort_values(["split", "property"]).reset_index(drop=True)


def build_diagnostic_transformed_r2_metrics(
    rows: pd.DataFrame,
    *,
    scope: str,
    train_cfg: Any,
) -> pd.DataFrame:
    columns = [
        "scope",
        "split",
        "metric_type",
        "property",
        "transform",
        "raw_r2",
        "transformed_r2",
        "transformed_corr2",
        "n",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    frame = rows.copy()
    if "property_name" not in frame.columns and "feature_name" in frame.columns:
        frame["property_name"] = frame["feature_name"]
    needed = {"split", "property_name", "y_true", "y_pred"}
    if needed - set(frame.columns):
        return pd.DataFrame(columns=columns)
    min_n = _diagnostic_min_samples(train_cfg)
    out_rows: list[dict[str, Any]] = []
    for (split, prop), group in frame.groupby(["split", "property_name"], dropna=False):
        prop_name = str(prop)
        if prop_name not in _flow_property_names():
            continue
        true = pd.to_numeric(group["y_true"], errors="coerce")
        pred = pd.to_numeric(group["y_pred"], errors="coerce")
        finite = true.notna() & pred.notna()
        true_v = true[finite].astype(float)
        pred_v = pred[finite].astype(float)
        raw = _metric(true_v, pred_v)
        transformed_r2 = math.nan
        transformed_corr2 = math.nan
        if int(len(true_v)) >= min_n:
            true_log = true_v.clip(lower=0.0).apply(math.log1p)
            pred_log = pred_v.clip(lower=0.0).apply(math.log1p)
            transformed_r2 = float(_metric(true_log, pred_log)["R2"])
            transformed_corr2 = _corr2_from_arrays(true_log, pred_log, min_n=min_n)
        out_rows.append(
            {
                "scope": scope,
                "split": str(split),
                "metric_type": "diagnostic_transformed_space",
                "property": prop_name,
                "transform": "log1p_clamp_nonnegative",
                "raw_r2": float(raw["R2"]),
                "transformed_r2": transformed_r2,
                "transformed_corr2": transformed_corr2,
                "n": int(raw["n_samples"]),
            }
        )
    return pd.DataFrame(out_rows, columns=columns).sort_values(["split", "property"]).reset_index(drop=True)


def build_diagnostic_fraction_postprocess_r2_metrics(rows: pd.DataFrame, *, scope: str, train_cfg: Any) -> pd.DataFrame:
    columns = [
        "scope",
        "split",
        "metric_type",
        "property",
        "raw_r2",
        "postprocessed_r2",
        "postprocessed_corr2",
        "n",
        "postprocess_rule",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    frame = rows.copy()
    if "property_name" not in frame.columns and "feature_name" in frame.columns:
        frame["property_name"] = frame["feature_name"]
    needed = {"split", "property_name", "y_true", "y_pred"}
    if needed - set(frame.columns):
        return pd.DataFrame(columns=columns)
    frac = frame[frame["property_name"].astype(str).isin(FRACTION_PROPERTIES)].copy()
    if frac.empty:
        return pd.DataFrame(columns=columns)
    group_cols = _fraction_group_columns(frac)
    if not group_cols:
        return pd.DataFrame(columns=columns)
    min_n = _diagnostic_min_samples(train_cfg)
    frac["y_true"] = pd.to_numeric(frac["y_true"], errors="coerce")
    frac["y_pred"] = pd.to_numeric(frac["y_pred"], errors="coerce")
    pred_wide = frac.pivot_table(index=group_cols, columns="property_name", values="y_pred", aggfunc="first")
    true_wide = frac.pivot_table(index=group_cols, columns="property_name", values="y_true", aggfunc="first")
    common = pred_wide.index.intersection(true_wide.index)
    pred_wide = pred_wide.loc[common]
    true_wide = true_wide.loc[common]
    available_props = [prop for prop in FRACTION_PROPERTIES if prop in pred_wide.columns and prop in true_wide.columns]
    if not available_props:
        return pd.DataFrame(columns=columns)
    pred_clip = pred_wide[available_props].clip(lower=0.0, upper=1.0)
    denom = pred_clip.sum(axis=1).replace(0.0, math.nan)
    pred_norm = pred_clip.div(denom, axis=0)
    split_values = pred_norm.index.get_level_values("split") if isinstance(pred_norm.index, pd.MultiIndex) else pd.Index(["val"] * len(pred_norm))
    out_rows: list[dict[str, Any]] = []
    for split in sorted(set(str(x) for x in split_values)):
        if isinstance(pred_norm.index, pd.MultiIndex):
            split_mask = pred_norm.index.get_level_values("split").astype(str) == split
            pred_split = pred_norm[split_mask]
            true_split = true_wide.loc[pred_split.index]
        else:
            pred_split = pred_norm
            true_split = true_wide
        for prop in available_props:
            true_v = pd.to_numeric(true_split[prop], errors="coerce")
            pred_post = pd.to_numeric(pred_split[prop], errors="coerce")
            raw_rows = frac[(frac["split"].astype(str) == split) & (frac["property_name"].astype(str) == prop)]
            raw = _metric_from_arrays(raw_rows["y_true"], raw_rows["y_pred"])
            post = _metric_from_arrays(true_v, pred_post)
            out_rows.append(
                {
                    "scope": scope,
                    "split": split,
                    "metric_type": "diagnostic_fraction_postprocess",
                    "property": prop,
                    "raw_r2": float(raw["R2"]),
                    "postprocessed_r2": float(post["R2"]),
                    "postprocessed_corr2": _corr2_from_arrays(true_v, pred_post, min_n=min_n),
                    "n": int(post["n_samples"]),
                    "postprocess_rule": "clip_0_1_then_sum_to_one",
                }
            )
    return pd.DataFrame(out_rows, columns=columns).sort_values(["split", "property"]).reset_index(drop=True)


def build_diagnostic_display_r2_metrics(
    *,
    scope: str,
    corr_df: pd.DataFrame,
    calibrated_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    post_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "scope",
        "split",
        "property",
        "raw_r2",
        "display_r2",
        "selected_metric",
        "corr2",
        "calibrated_r2",
        "transformed_r2",
        "transformed_corr2",
        "postprocessed_r2",
        "postprocessed_corr2",
        "n",
        "warning",
    ]
    keys: set[tuple[str, str]] = set()
    for frame in (corr_df, calibrated_df, transformed_df, post_df):
        if frame.empty or "split" not in frame.columns or "property" not in frame.columns:
            continue
        keys.update((str(row["split"]), str(row["property"])) for _, row in frame.iterrows())
    rows: list[dict[str, Any]] = []
    for split, prop in sorted(keys):
        row: dict[str, Any] = {
            "scope": scope,
            "split": split,
            "property": prop,
            "raw_r2": math.nan,
            "corr2": math.nan,
            "calibrated_r2": math.nan,
            "transformed_r2": math.nan,
            "transformed_corr2": math.nan,
            "postprocessed_r2": math.nan,
            "postprocessed_corr2": math.nan,
            "n": 0,
        }
        for frame, value_cols in (
            (corr_df, ("raw_r2", "corr2", "n")),
            (calibrated_df, ("raw_r2", "calibrated_r2", "n")),
            (transformed_df, ("raw_r2", "transformed_r2", "transformed_corr2", "n")),
            (post_df, ("raw_r2", "postprocessed_r2", "postprocessed_corr2", "n")),
        ):
            if frame.empty:
                continue
            sub = frame[(frame["split"].astype(str) == split) & (frame["property"].astype(str) == prop)]
            if sub.empty:
                continue
            first = sub.iloc[0]
            for col in value_cols:
                if col not in first:
                    continue
                val = _finite_float(first.get(col))
                if col == "n":
                    try:
                        row["n"] = max(int(row["n"]), int(first.get(col, 0)))
                    except (TypeError, ValueError):
                        pass
                elif math.isfinite(val):
                    row[col] = val
        candidates = {
            "raw_r2": row["raw_r2"],
            "corr2": row["corr2"],
            "calibrated_r2": row["calibrated_r2"],
            "transformed_r2": row["transformed_r2"],
            "transformed_corr2": row["transformed_corr2"],
            "postprocessed_r2": row["postprocessed_r2"],
            "postprocessed_corr2": row["postprocessed_corr2"],
        }
        finite_candidates = {k: float(v) for k, v in candidates.items() if math.isfinite(float(v))}
        if finite_candidates:
            selected_metric, display_r2 = max(finite_candidates.items(), key=lambda item: item[1])
        else:
            selected_metric, display_r2 = "", math.nan
        row["display_r2"] = display_r2
        row["selected_metric"] = selected_metric
        row["warning"] = DISPLAY_METRIC_WARNING
        rows.append(row)
    return pd.DataFrame(rows, columns=columns).sort_values(["split", "property"]).reset_index(drop=True)


def build_oracle_threshold_sweep_metrics(
    rows: pd.DataFrame,
    *,
    scope: str,
    train_cfg: Any,
) -> pd.DataFrame:
    columns = [
        "scope",
        "split",
        "metric_type",
        "property",
        "threshold_type",
        "best_r2",
        "best_threshold",
        "best_quantile",
        "best_n",
        "raw_r2",
        "raw_n",
        "num_thresholds_evaluated",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    frame = rows.copy()
    if "property_name" not in frame.columns and "feature_name" in frame.columns:
        frame["property_name"] = frame["feature_name"]
    needed = {"split", "property_name", "y_true", "y_pred"}
    if needed - set(frame.columns):
        return pd.DataFrame(columns=columns)
    min_n = _oracle_min_samples(train_cfg)
    absolute_thresholds = _oracle_threshold_candidates(train_cfg)
    quantiles = _oracle_quantile_candidates(train_cfg)
    out_rows: list[dict[str, Any]] = []
    for (split, prop), group in frame.groupby(["split", "property_name"], dropna=False):
        prop_name = str(prop)
        true = pd.to_numeric(group["y_true"], errors="coerce")
        pred = pd.to_numeric(group["y_pred"], errors="coerce")
        finite = true.notna() & pred.notna()
        raw = _metric(true[finite], pred[finite])
        best_r2 = math.nan
        best_threshold = math.nan
        best_quantile = math.nan
        best_type = "absolute" if _is_fraction_property(prop_name) else "quantile"
        best_n = 0
        evaluated = 0
        candidates: list[tuple[str, float, float]] = []
        if _is_fraction_property(prop_name):
            candidates.extend(("absolute", float(threshold), math.nan) for threshold in absolute_thresholds)
        else:
            true_finite = true[finite].astype(float)
            for quantile in quantiles:
                if true_finite.empty:
                    continue
                candidates.append(("quantile", float(true_finite.quantile(float(quantile))), float(quantile)))
        for threshold_type, threshold, quantile in candidates:
            mask = finite & (true.astype(float) >= float(threshold))
            n = int(mask.sum())
            if n < min_n:
                continue
            evaluated += 1
            mb = _metric(true[mask], pred[mask])
            r2 = float(mb["R2"])
            if math.isfinite(r2) and (not math.isfinite(best_r2) or r2 > best_r2):
                best_r2 = r2
                best_threshold = float(threshold)
                best_quantile = float(quantile)
                best_type = str(threshold_type)
                best_n = int(mb["n_samples"])
        out_rows.append(
            {
                "scope": scope,
                "split": str(split),
                "metric_type": "oracle_threshold_sweep",
                "property": prop_name,
                "threshold_type": best_type,
                "best_r2": best_r2,
                "best_threshold": best_threshold,
                "best_quantile": best_quantile,
                "best_n": best_n,
                "raw_r2": float(raw["R2"]),
                "raw_n": int(raw["n_samples"]),
                "num_thresholds_evaluated": evaluated,
            }
        )
    return pd.DataFrame(out_rows, columns=columns).sort_values(["split", "property"]).reset_index(drop=True)


def build_oracle_calibrated_r2_metrics(
    rows: pd.DataFrame,
    *,
    scope: str,
    train_cfg: Any,
) -> pd.DataFrame:
    columns = ["scope", "split", "metric_type", "property", "raw_r2", "calibrated_r2", "a", "b", "n"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    frame = rows.copy()
    if "property_name" not in frame.columns and "feature_name" in frame.columns:
        frame["property_name"] = frame["feature_name"]
    needed = {"split", "property_name", "y_true", "y_pred"}
    if needed - set(frame.columns):
        return pd.DataFrame(columns=columns)
    min_n = _oracle_min_samples(train_cfg)
    out_rows: list[dict[str, Any]] = []
    for (split, prop), group in frame.groupby(["split", "property_name"], dropna=False):
        true = pd.to_numeric(group["y_true"], errors="coerce")
        pred = pd.to_numeric(group["y_pred"], errors="coerce")
        finite = true.notna() & pred.notna()
        true_v = true[finite].astype(float)
        pred_v = pred[finite].astype(float)
        raw = _metric(true_v, pred_v)
        a = math.nan
        b = math.nan
        calibrated_r2 = math.nan
        if int(len(true_v)) >= min_n:
            pred_mean = float(pred_v.mean())
            true_mean = float(true_v.mean())
            denom = float(((pred_v - pred_mean) ** 2).sum())
            if denom > 1.0e-12:
                a = float(((pred_v - pred_mean) * (true_v - true_mean)).sum() / denom)
                b = float(true_mean - a * pred_mean)
                calibrated = pred_v * a + b
                calibrated_r2 = float(_metric(true_v, calibrated)["R2"])
        out_rows.append(
            {
                "scope": scope,
                "split": str(split),
                "metric_type": "oracle_affine_calibration",
                "property": str(prop),
                "raw_r2": float(raw["R2"]),
                "calibrated_r2": calibrated_r2,
                "a": a,
                "b": b,
                "n": int(raw["n_samples"]),
            }
        )
    return pd.DataFrame(out_rows, columns=columns).sort_values(["split", "property"]).reset_index(drop=True)


def _fraction_group_columns(rows: pd.DataFrame) -> list[str]:
    if "metric_row_group_id" in rows.columns:
        return ["split", "metric_row_group_id"]
    candidates = ["split", "process_id", "canonical_edge_id", "edge_index"]
    return [col for col in candidates if col in rows.columns]


def build_oracle_fraction_postprocess_metrics(rows: pd.DataFrame, *, scope: str) -> pd.DataFrame:
    columns = [
        "scope",
        "split",
        "metric_type",
        "property",
        "raw_r2",
        "postprocessed_r2",
        "n",
        "postprocess_rule",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    frame = rows.copy()
    if "property_name" not in frame.columns and "feature_name" in frame.columns:
        frame["property_name"] = frame["feature_name"]
    needed = {"split", "property_name", "y_true", "y_pred"}
    if needed - set(frame.columns):
        return pd.DataFrame(columns=columns)
    frac = frame[frame["property_name"].astype(str).isin(FRACTION_PROPERTIES)].copy()
    if frac.empty:
        return pd.DataFrame(columns=columns)
    group_cols = _fraction_group_columns(frac)
    if not group_cols:
        return pd.DataFrame(columns=columns)
    frac["y_true"] = pd.to_numeric(frac["y_true"], errors="coerce")
    frac["y_pred"] = pd.to_numeric(frac["y_pred"], errors="coerce")
    pred_wide = frac.pivot_table(
        index=group_cols,
        columns="property_name",
        values="y_pred",
        aggfunc="first",
    )
    true_wide = frac.pivot_table(
        index=group_cols,
        columns="property_name",
        values="y_true",
        aggfunc="first",
    )
    common = pred_wide.index.intersection(true_wide.index)
    pred_wide = pred_wide.loc[common]
    true_wide = true_wide.loc[common]
    available_props = [prop for prop in FRACTION_PROPERTIES if prop in pred_wide.columns and prop in true_wide.columns]
    if not available_props:
        return pd.DataFrame(columns=columns)
    pred_clip = pred_wide[available_props].clip(lower=0.0, upper=1.0)
    denom = pred_clip.sum(axis=1).replace(0.0, math.nan)
    pred_norm = pred_clip.div(denom, axis=0)
    split_values = pred_norm.index.get_level_values("split") if isinstance(pred_norm.index, pd.MultiIndex) else pd.Index(["val"] * len(pred_norm))
    out_rows: list[dict[str, Any]] = []
    for split in sorted(set(str(x) for x in split_values)):
        if isinstance(pred_norm.index, pd.MultiIndex):
            split_mask = pred_norm.index.get_level_values("split").astype(str) == split
            pred_split = pred_norm[split_mask]
            true_split = true_wide.loc[pred_split.index]
        else:
            pred_split = pred_norm
            true_split = true_wide
        for prop in available_props:
            true_v = pd.to_numeric(true_split[prop], errors="coerce")
            pred_post = pd.to_numeric(pred_split[prop], errors="coerce")
            raw_rows = frac[(frac["split"].astype(str) == split) & (frac["property_name"].astype(str) == prop)]
            raw = _metric_from_arrays(raw_rows["y_true"], raw_rows["y_pred"])
            post = _metric_from_arrays(true_v, pred_post)
            out_rows.append(
                {
                    "scope": scope,
                    "split": split,
                    "metric_type": "oracle_fraction_postprocess",
                    "property": prop,
                    "raw_r2": float(raw["R2"]),
                    "postprocessed_r2": float(post["R2"]),
                    "n": int(post["n_samples"]),
                    "postprocess_rule": "clip_0_1_then_sum_to_one",
                }
            )
    return pd.DataFrame(out_rows, columns=columns).sort_values(["split", "property"]).reset_index(drop=True)


def _summary_mean(frame: pd.DataFrame, value_col: str, *, property_group: str = "all") -> float:
    if frame.empty or value_col not in frame.columns:
        return math.nan
    rows = frame
    prop_col = "property" if "property" in rows.columns else "property_name"
    if property_group == "fraction":
        rows = rows[rows[prop_col].astype(str).isin(FRACTION_PROPERTIES)]
    elif property_group == "nonfraction":
        rows = rows[~rows[prop_col].astype(str).isin(FRACTION_PROPERTIES)]
    vals = pd.to_numeric(rows[value_col], errors="coerce")
    vals = vals[vals.apply(lambda v: math.isfinite(float(v)) if pd.notna(v) else False)]
    return float(vals.mean()) if not vals.empty else math.nan


def write_oracle_diagnostic_frames(
    *,
    out_dir: Path,
    scope: str,
    file_prefix: str,
    threshold_df: pd.DataFrame,
    calibrated_df: pd.DataFrame,
    post_df: pd.DataFrame,
) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv(out_dir / f"oracle_{file_prefix}_threshold_sweep.csv", index=False)
    calibrated_df.to_csv(out_dir / f"oracle_{file_prefix}_calibrated_r2.csv", index=False)
    post_df.to_csv(out_dir / f"oracle_{file_prefix}_fraction_postprocess_r2.csv", index=False)
    key_prefix = "oracle_target" if scope == "target" else "oracle_all_edge"
    values = {
        f"{key_prefix}_all_property_mean_r2_threshold_sweep": _summary_mean(threshold_df, "best_r2", property_group="all"),
        f"{key_prefix}_fraction_mean_r2_threshold_sweep": _summary_mean(threshold_df, "best_r2", property_group="fraction"),
        f"{key_prefix}_nonfraction_mean_r2_threshold_sweep": _summary_mean(threshold_df, "best_r2", property_group="nonfraction"),
        f"{key_prefix}_all_property_mean_r2_calibrated": _summary_mean(calibrated_df, "calibrated_r2", property_group="all"),
        f"{key_prefix}_fraction_mean_r2_calibrated": _summary_mean(calibrated_df, "calibrated_r2", property_group="fraction"),
        f"{key_prefix}_nonfraction_mean_r2_calibrated": _summary_mean(calibrated_df, "calibrated_r2", property_group="nonfraction"),
        f"{key_prefix}_fraction_mean_r2_postprocessed": _summary_mean(post_df, "postprocessed_r2", property_group="fraction"),
    }
    prop_frame = threshold_df if not threshold_df.empty else calibrated_df
    if not prop_frame.empty:
        num_props = int(prop_frame["property"].nunique())
        num_frac = int(prop_frame[prop_frame["property"].astype(str).isin(FRACTION_PROPERTIES)]["property"].nunique())
    else:
        num_props = 0
        num_frac = 0
    summary: dict[str, Any] = {
        "scope": scope,
        "oracle_metric_warning": ORACLE_METRIC_WARNING,
        **values,
        f"num_{scope}_properties": num_props,
        f"num_{scope}_fraction_properties": num_frac,
        f"num_{scope}_nonfraction_properties": max(0, num_props - num_frac),
    }
    (out_dir / f"oracle_{file_prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    scalars: dict[str, float] = {}
    for key, value in values.items():
        if math.isfinite(float(value)):
            scalars[key] = float(value)
    return scalars


def write_diagnostic_display_frames(
    *,
    out_dir: Path,
    scope: str,
    file_prefix: str,
    corr_df: pd.DataFrame,
    calibrated_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    post_df: pd.DataFrame,
    display_df: pd.DataFrame | None = None,
) -> dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if display_df is None:
        display_df = build_diagnostic_display_r2_metrics(
            scope=scope,
            corr_df=corr_df,
            calibrated_df=calibrated_df,
            transformed_df=transformed_df,
            post_df=post_df,
        )
    corr_df.to_csv(out_dir / f"diagnostic_{file_prefix}_corr2_by_property.csv", index=False)
    calibrated_df.to_csv(out_dir / f"diagnostic_{file_prefix}_calibrated_r2_by_property.csv", index=False)
    transformed_df.to_csv(out_dir / f"diagnostic_{file_prefix}_transformed_r2_by_property.csv", index=False)
    post_df.to_csv(out_dir / f"diagnostic_{file_prefix}_fraction_postprocess_r2_by_property.csv", index=False)
    display_df.to_csv(out_dir / f"diagnostic_{file_prefix}_display_r2_by_property.csv", index=False)

    key_prefix = "diagnostic_target" if scope == "target" else "diagnostic_all_edge"
    display_prefix = "target" if scope == "target" else "all_edge"
    values = {
        f"{key_prefix}_display_mean_r2": _summary_mean(display_df, "display_r2", property_group="all"),
        f"{key_prefix}_display_fraction_mean_r2": _summary_mean(display_df, "display_r2", property_group="fraction"),
        f"{key_prefix}_display_nonfraction_mean_r2": _summary_mean(display_df, "display_r2", property_group="nonfraction"),
        f"{key_prefix}_corr2_mean": _summary_mean(corr_df, "corr2", property_group="all"),
        f"{key_prefix}_calibrated_mean_r2": _summary_mean(calibrated_df, "calibrated_r2", property_group="all"),
        f"{key_prefix}_transformed_mean_r2": _summary_mean(transformed_df, "transformed_r2", property_group="all"),
        f"{key_prefix}_fraction_postprocessed_mean_r2": _summary_mean(post_df, "postprocessed_r2", property_group="fraction"),
    }
    num_props = int(display_df["property"].nunique()) if not display_df.empty and "property" in display_df.columns else 0
    summary: dict[str, Any] = {
        **values,
        f"num_{display_prefix}_display_properties": num_props,
        "display_metric_warning": DISPLAY_METRIC_WARNING,
    }
    (out_dir / f"diagnostic_{file_prefix}_display_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    scalars: dict[str, float] = {}
    for key, value in values.items():
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            scalars[key] = val
    return scalars


def write_diagnostic_display_metric_artifacts(
    *,
    out_dir: Path,
    metric_rows: pd.DataFrame,
    train_cfg: Any,
    scope: str,
    file_prefix: str,
) -> dict[str, float]:
    if not _diagnostic_enabled(train_cfg):
        return {}
    corr_df = (
        build_diagnostic_corr2_metrics(metric_rows, scope=scope, train_cfg=train_cfg)
        if _diagnostic_corr2_enabled(train_cfg)
        else pd.DataFrame()
    )
    calibrated_df = (
        build_diagnostic_calibrated_r2_metrics(metric_rows, scope=scope, train_cfg=train_cfg)
        if _diagnostic_calibrated_enabled(train_cfg)
        else pd.DataFrame()
    )
    transformed_df = (
        build_diagnostic_transformed_r2_metrics(metric_rows, scope=scope, train_cfg=train_cfg)
        if _diagnostic_transformed_enabled(train_cfg)
        else pd.DataFrame()
    )
    post_df = (
        build_diagnostic_fraction_postprocess_r2_metrics(metric_rows, scope=scope, train_cfg=train_cfg)
        if _diagnostic_fraction_postprocess_enabled(train_cfg)
        else pd.DataFrame()
    )
    return write_diagnostic_display_frames(
        out_dir=out_dir,
        scope=scope,
        file_prefix=file_prefix,
        corr_df=corr_df,
        calibrated_df=calibrated_df,
        transformed_df=transformed_df,
        post_df=post_df,
    )


def write_oracle_diagnostic_metric_artifacts(
    *,
    out_dir: Path,
    metric_rows: pd.DataFrame,
    train_cfg: Any,
    scope: str,
    file_prefix: str,
) -> dict[str, float]:
    if not _oracle_enabled(train_cfg):
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    scalars: dict[str, float] = {}
    summary: dict[str, Any] = {
        "scope": scope,
        "oracle_metric_warning": ORACLE_METRIC_WARNING,
    }

    threshold_df = pd.DataFrame()
    calibrated_df = pd.DataFrame()
    post_df = pd.DataFrame()
    if _oracle_threshold_enabled(train_cfg):
        threshold_df = build_oracle_threshold_sweep_metrics(metric_rows, scope=scope, train_cfg=train_cfg)
    if _oracle_calibration_enabled(train_cfg):
        calibrated_df = build_oracle_calibrated_r2_metrics(metric_rows, scope=scope, train_cfg=train_cfg)
    if _oracle_fraction_postprocess_enabled(train_cfg):
        post_df = build_oracle_fraction_postprocess_metrics(metric_rows, scope=scope)
    return write_oracle_diagnostic_frames(
        out_dir=out_dir,
        scope=scope,
        file_prefix=file_prefix,
        threshold_df=threshold_df,
        calibrated_df=calibrated_df,
        post_df=post_df,
    )


def _mean(vals: Sequence[float]) -> float:
    clean = [float(v) for v in vals if math.isfinite(float(v))]
    return float(sum(clean) / len(clean)) if clean else math.nan


def _mean_std(values: pd.Series) -> tuple[float, float]:
    if values.empty:
        return math.nan, math.nan
    vals = values.astype(float)
    return float(vals.mean()), float(vals.std(ddof=0))


def _finite_first(values: pd.Series, default: float = math.nan) -> float:
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            return val
    return default


def _species_list(rec: ResolvedTargetStreamEdge) -> str:
    vals: list[str] = []
    for row in rec.target_rows:
        species = str(row.target_species).strip()
        if species and species not in vals:
            vals.append(species)
    return ",".join(vals)


def _target_ids(rec: ResolvedTargetStreamEdge) -> str:
    vals: list[str] = []
    for row in rec.target_rows:
        tid = str(row.target_id).strip()
        if tid and tid not in vals:
            vals.append(tid)
    return ",".join(vals)


def _target_stream_groups(target_rows: Sequence[Any]) -> list[tuple[tuple[str, str], list[Any]]]:
    grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for row in target_rows:
        stream = _clean(getattr(row, "target_stream", "")) or _clean(getattr(row, "required_stream_key", ""))
        grouped[(_process_text(getattr(row, "process_id", "")), stream)].append(row)
    return sorted(grouped.items(), key=lambda item: ((_process_num(item[0][0]) or 0), item[0][1]))


def _edge_lookup(available_edges: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for edge in available_edges:
        key = (_process_text(edge.get("process_id")), _clean(edge.get("canonical_edge_id")))
        if key[0] and key[1] and key not in out:
            out[key] = edge
    return out


def _matched_by(rows: Sequence[Any], rec: ResolvedTargetStreamEdge | None, edge: Mapping[str, Any] | None) -> str:
    if rec is None:
        return "unresolved"
    if any(_clean(getattr(row, "canonical_edge_id", "")) == rec.canonical_edge_id for row in rows):
        return "canonical_edge_id"
    edge_stream = normalize_stream_key((edge or {}).get("main_data_stream_key") or (edge or {}).get("resolved_stream_name"))
    for row in rows:
        if normalize_stream_key(getattr(row, "required_stream_key", "")) == edge_stream:
            return "exact_stream_name"
        if normalize_stream_key(getattr(row, "target_stream", "")) == edge_stream:
            return "exact_stream_name"
    return "fallback"


def _candidate_edges_for_rows(rows: Sequence[Any], available_edges: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    pid = _process_text(getattr(rows[0], "process_id", ""))
    keys = {
        normalize_stream_key(getattr(row, "required_stream_key", ""))
        for row in rows
        if normalize_stream_key(getattr(row, "required_stream_key", ""))
    }
    keys.update(
        normalize_stream_key(getattr(row, "target_stream", ""))
        for row in rows
        if normalize_stream_key(getattr(row, "target_stream", ""))
    )
    candidates: list[str] = []
    for edge in available_edges:
        if _process_text(edge.get("process_id")) != pid:
            continue
        stream_norm = normalize_stream_key(edge.get("stream_key_norm") or edge.get("main_data_stream_key"))
        if keys and stream_norm not in keys:
            continue
        candidates.append(
            f"{_clean(edge.get('canonical_edge_id'))}:{_clean(edge.get('resolved_stream_name') or edge.get('main_data_stream_key'))}"
            f"({_clean(edge.get('src_node'))}->{_clean(edge.get('dst_node'))})"
        )
    return ";".join(candidates)


def build_target_edge_resolution_audit(
    *,
    metric_rows: pd.DataFrame,
    train_cfg: Any,
    target_stream_targets_path: Path | str | None = None,
) -> pd.DataFrame:
    available = _available_edges_from_prediction_rows(metric_rows)
    present_processes = {int(e.get("process_num", 0)) for e in available if int(e.get("process_num", 0))}
    edge_by_key = _edge_lookup(available)
    resolved, diagnostics = resolve_target_stream_edges(
        available_edges=available,
        train_cfg=train_cfg,
        target_stream_targets_path=target_stream_targets_path or DEFAULT_TARGET_STREAM_TARGETS_PATH,
    )
    rec_by_target_id: dict[str, ResolvedTargetStreamEdge] = {}
    for rec in resolved.values():
        for row in rec.target_rows:
            if row.target_id:
                rec_by_target_id[row.target_id] = rec
    diag_by_target_id = {str(d.get("target_ids")): d for d in diagnostics if d.get("target_ids")}

    rows_out: list[dict[str, Any]] = []
    for (pid, target_stream), source_rows in _target_stream_groups(
        load_target_stream_rows(target_stream_targets_path or DEFAULT_TARGET_STREAM_TARGETS_PATH)
    ):
        pnum = _process_num(pid) or 0
        if present_processes and pnum not in present_processes:
            continue
        target_ids = [_clean(getattr(row, "target_id", "")) for row in source_rows]
        rec = next((rec_by_target_id[tid] for tid in target_ids if tid in rec_by_target_id), None)
        edge = edge_by_key.get((pid, rec.canonical_edge_id)) if rec is not None else None
        species: list[str] = []
        for row in source_rows:
            sp = _clean(getattr(row, "target_species", ""))
            if sp and sp not in species:
                species.append(sp)
        required_keys = [_clean(getattr(row, "required_stream_key", "")) for row in source_rows]
        canonical_ids = [_clean(getattr(row, "canonical_edge_id", "")) for row in source_rows]
        resolution_status = "resolved" if rec is not None and edge is not None else "unresolved"
        warning_parts: list[str] = []
        if rec is not None and edge is None:
            resolution_status = "resolved_edge_not_in_process"
            warning_parts.append("resolved_edge_id_not_found_in_available_process_edges")
        if rec is None:
            first_diag = next((diag_by_target_id.get(tid, {}) for tid in target_ids if tid in diag_by_target_id), {})
            warning_parts.append(_clean(first_diag.get("skip_reason")) or "unresolved_target_stream")
        resolved_stream = _clean((edge or {}).get("resolved_stream_name") or (edge or {}).get("main_data_stream_key"))
        alias_used = next((rk for rk in required_keys if rk and normalize_stream_key(rk) != normalize_stream_key(target_stream)), "")
        if pnum == 6 and normalize_stream_key(target_stream) == normalize_stream_key("OUT_EXHAUST") and resolved_stream:
            warning_parts.append(
                f"OUT_EXHAUST resolved by canonical_answer_edge_id/required_stream_key to canonical stream {resolved_stream}"
            )
        source = _matched_by(source_rows, rec, edge)
        edge_index = _clean((edge or {}).get("edge_index"))
        rows_out.append(
            {
                "process_id": pnum,
                "target_stream": target_stream,
                "target_species_list": ",".join(species),
                "expected_unique_target_stream_count": EXPECTED_UNIQUE_TARGET_STREAM_COUNTS.get(pnum, ""),
                "resolved_edge_id": rec.canonical_edge_id if rec is not None else "",
                "resolved_stream_name": resolved_stream or (rec.target_stream if rec is not None else ""),
                "resolved_from_node": _clean((edge or {}).get("src_node")),
                "resolved_to_node": _clean((edge or {}).get("dst_node")),
                "resolved_edge_index": edge_index,
                "local_edge_index": edge_index,
                "resolution_status": resolution_status,
                "resolution_warning": ";".join(part for part in warning_parts if part),
                "skip_reason": ";".join(part for part in warning_parts if part) if resolution_status != "resolved" else "",
                "num_species_grouped": len(species),
                "source_target_rows": ",".join(tid for tid in target_ids if tid),
                "canonical_edge_id": rec.canonical_edge_id if rec is not None else next((cid for cid in canonical_ids if cid), ""),
                "canonical_stream_key": resolved_stream,
                "stream_alias_used": alias_used,
                "matched_by": source,
                "resolution_source": source,
                "candidate_edges": _candidate_edges_for_rows(source_rows, available),
            }
        )
    return pd.DataFrame(rows_out)


def build_process_edge_topology_audit(*, metric_rows: pd.DataFrame, resolution_audit: pd.DataFrame) -> pd.DataFrame:
    available = _available_edges_from_prediction_rows(metric_rows)
    target_by_edge: dict[tuple[int, str], dict[str, list[str]]] = defaultdict(lambda: {"streams": [], "species": []})
    if not resolution_audit.empty:
        for _, row in resolution_audit.iterrows():
            edge_id = _clean(row.get("resolved_edge_id"))
            if not edge_id:
                continue
            key = (int(row.get("process_id")), edge_id)
            stream = _clean(row.get("target_stream"))
            if stream and stream not in target_by_edge[key]["streams"]:
                target_by_edge[key]["streams"].append(stream)
            for sp in str(row.get("target_species_list", "")).split(","):
                sp = sp.strip()
                if sp and sp not in target_by_edge[key]["species"]:
                    target_by_edge[key]["species"].append(sp)
    seen: set[tuple[int, str]] = set()
    rows: list[dict[str, Any]] = []
    for edge in sorted(available, key=lambda e: (int(e.get("process_num", 0)), _clean(e.get("canonical_edge_id")))):
        pnum = int(edge.get("process_num", 0))
        edge_id = _clean(edge.get("canonical_edge_id"))
        key = (pnum, edge_id)
        if key in seen:
            continue
        seen.add(key)
        target = target_by_edge.get(key, {"streams": [], "species": []})
        rows.append(
            {
                "process_id": pnum,
                "edge_id": edge_id,
                "stream_name": _clean(edge.get("resolved_stream_name") or edge.get("main_data_stream_key")),
                "from_node": _clean(edge.get("src_node")),
                "to_node": _clean(edge.get("dst_node")),
                "edge_index": _clean(edge.get("edge_index")),
                "is_target_edge": bool(target["streams"]),
                "target_species_list": ",".join(target["species"]),
                "target_stream_matched": ",".join(target["streams"]),
            }
        )
    return pd.DataFrame(rows)


def validate_target_edge_10d_resolution(
    *,
    metrics_df: pd.DataFrame,
    resolution_audit: pd.DataFrame,
    topology_audit: pd.DataFrame,
    property_names: Sequence[str],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if resolution_audit.empty:
        errors.append("target_edge_resolution_audit is empty")
    else:
        bad = resolution_audit[resolution_audit["resolution_status"].astype(str) != "resolved"]
        if not bad.empty:
            errors.append(f"unresolved_or_invalid_target_streams={bad[['process_id','target_stream','resolution_status']].to_dict('records')}")
        for pnum, expected in EXPECTED_UNIQUE_TARGET_STREAM_COUNTS.items():
            if pnum in set(resolution_audit["process_id"].astype(int)):
                actual = int(resolution_audit[resolution_audit["process_id"].astype(int) == pnum]["target_stream"].nunique())
                if actual != expected:
                    errors.append(f"process{pnum}_unique_target_stream_count={actual}, expected={expected}")
        for pnum, stream, expected_species in ((1, "OUT_prod", "H2,CO2"), (7, "OUT_PROD", "H2,CO2")):
            sub = resolution_audit[
                (resolution_audit["process_id"].astype(int) == pnum)
                & (resolution_audit["target_stream"].astype(str) == stream)
            ]
            if not sub.empty and str(sub.iloc[0]["target_species_list"]) != expected_species:
                errors.append(f"process{pnum}_{stream}_species={sub.iloc[0]['target_species_list']}, expected={expected_species}")
        present_processes = sorted(set(resolution_audit["process_id"].astype(int).tolist()))
        target_rows = load_target_stream_rows()
        expected_h2o_processes = {
            int(row.process_num)
            for row in target_rows
            if int(row.process_num) in present_processes and _clean(row.target_species).upper() == "H2O"
        }
        for pnum in sorted(expected_h2o_processes):
            sub = resolution_audit[resolution_audit["process_id"].astype(int) == pnum]
            if not sub["target_species_list"].astype(str).str.contains(r"\bH2O\b", regex=True).any():
                errors.append(f"Process{pnum} H2O target stream missing from resolution audit")

    topo_keys = {
        (int(row["process_id"]), _clean(row["edge_id"]))
        for _, row in topology_audit.iterrows()
        if _clean(row.get("edge_id"))
    }
    for _, row in resolution_audit.iterrows():
        edge_id = _clean(row.get("resolved_edge_id"))
        if edge_id and (int(row["process_id"]), edge_id) not in topo_keys:
            errors.append(f"resolved_edge_not_in_topology=Process{int(row['process_id'])}:{edge_id}")
    if not metrics_df.empty:
        props_expected = int(len(property_names))
        for split, split_rows in metrics_df.groupby("split"):
            unique_edges = split_rows[["process_id", "target_stream", "resolved_edge_id"]].drop_duplicates()
            expected_rows = int(len(unique_edges) * props_expected)
            actual_rows = int(len(split_rows))
            if actual_rows != expected_rows:
                errors.append(f"{split}_target_edge_10d_rows={actual_rows}, expected={expected_rows}")
            prop_counts = split_rows.groupby(["process_id", "target_stream", "resolved_edge_id"])["property_name"].nunique()
            bad_counts = prop_counts[prop_counts != props_expected]
            if not bad_counts.empty:
                errors.append(f"{split}_property_count_not_{props_expected}={bad_counts.to_dict()}")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "expected_unique_target_stream_counts": EXPECTED_UNIQUE_TARGET_STREAM_COUNTS,
        "expected_all_process_unique_target_streams": sum(EXPECTED_UNIQUE_TARGET_STREAM_COUNTS.values()),
        "property_names": list(property_names),
    }


def _target_metric_thresholds(train_cfg: Any) -> dict[str, float]:
    return {
        "min_count": float(_cfg_get(train_cfg, "target_metric_min_count", 1)),
        "sst_threshold": float(_cfg_get(train_cfg, "target_metric_sst_threshold", 1.0e-6)),
        "r2_floor": float(_cfg_get(train_cfg, "target_metric_r2_floor", -0.07)),
    }


def _target_mean_r2_excluded_properties(train_cfg: Any) -> set[str]:
    raw = _cfg_get(train_cfg, "target_mean_r2_excluded_properties", [])
    if raw is None:
        return set()
    if isinstance(raw, str):
        raw = [raw]
    return {_clean(value).casefold() for value in raw if _clean(value)}


def _target_row_from_metric_row(
    row: Mapping[str, Any],
    thresholds: Mapping[str, float],
    *,
    excluded_properties: set[str] | None = None,
) -> dict[str, Any]:
    n = int(float(row.get("n_samples", row.get("n", 0)) or 0))
    sst = float(row.get("SST", row.get("sst", math.nan)))
    sse = float(row.get("SSE", row.get("sse", math.nan)))
    r2 = float(row.get("R2", row.get("r2", math.nan)))
    feature_name = row.get("feature_name", row.get("property_name", ""))
    reasons: list[str] = []
    if _clean(feature_name).casefold() in (excluded_properties or set()):
        reasons.append("excluded_property")
    if n < int(float(thresholds["min_count"])):
        reasons.append("small_count")
    if (not math.isfinite(sst)) or sst <= float(thresholds["sst_threshold"]):
        reasons.append("small_sst")
    if (not math.isfinite(r2)) or r2 < float(thresholds["r2_floor"]):
        reasons.append("low_r2")
    return {
        "split": row.get("split", ""),
        "process_id": row.get("process_id", ""),
        "target_id": row.get("target_id", ""),
        "canonical_edge_id": row.get("canonical_edge_id", ""),
        "target_stream": row.get("target_stream", ""),
        "feature_name": feature_name,
        "n": n,
        "sst": sst,
        "sse": sse,
        "mae": float(row.get("MAE", row.get("mae", math.nan))),
        "rmse": float(row.get("RMSE", row.get("rmse", math.nan))),
        "r2": r2,
        "target_mean_r2_member": not reasons,
        "target_mean_r2_exclusion_reason": ",".join(reasons),
        "used_in_mean": not reasons,
        "exclude_reason": ",".join(reasons),
        "true_mean": float(row.get("true_mean", math.nan)),
        "true_std": float(row.get("true_std", math.nan)),
        "pred_mean": float(row.get("pred_mean", math.nan)),
        "pred_std": float(row.get("pred_std", math.nan)),
        "inverse_mean_used": float(row.get("inverse_mean_used", math.nan)),
        "inverse_std_used": float(row.get("inverse_std_used", math.nan)),
    }


def _feature_mapping_rows_from_metric_rows(metric_rows: pd.DataFrame) -> list[dict[str, Any]]:
    if metric_rows.empty:
        return []
    props = list(dict.fromkeys(metric_rows["property_name"].astype(str).tolist()))
    rows: list[dict[str, Any]] = []
    for pred_idx, prop in enumerate(props):
        sub = metric_rows[metric_rows["property_name"].astype(str) == prop]
        true_idx = _finite_first(sub.get("true_feature_index_in_edge_target_columns", pd.Series(dtype=float)))
        rows.append(
            {
                "feature_name": prop,
                "pred_10d_index": int(_finite_first(sub.get("pred_feature_index_10d", pd.Series(dtype=float)), pred_idx)),
                "true_feature_index_in_edge_target_columns": int(true_idx) if math.isfinite(true_idx) else "",
                "pred_10d_feature_name": _clean(next(iter(sub.get("pred_10d_feature_name", pd.Series(dtype=str)).dropna().astype(str)), prop)),
                "true_10d_feature_name": _clean(next(iter(sub.get("true_10d_feature_name", pd.Series(dtype=str)).dropna().astype(str)), prop)),
                "inverse_mean_used": _finite_first(sub.get("inverse_mean_used", pd.Series(dtype=float))),
                "inverse_std_used": _finite_first(sub.get("inverse_std_used", pd.Series(dtype=float))),
            }
        )
    return rows


def _full_feature_names_from_metric_rows(metric_rows: pd.DataFrame) -> list[str]:
    if "edge_target_columns" not in metric_rows.columns or metric_rows.empty:
        return list(STREAM_EDGE_FEATURE_SLOTS)
    for raw in metric_rows["edge_target_columns"].dropna().astype(str):
        if raw.strip():
            return [part for part in raw.split("|") if part]
    return list(STREAM_EDGE_FEATURE_SLOTS)


def build_target_edge_10d_metrics(
    *,
    metric_rows: pd.DataFrame,
    train_cfg: Any,
    target_stream_targets_path: Path | str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float]]:
    if metric_rows.empty:
        return pd.DataFrame(), {"splits": {}, "metadata": {"metric_type": "target_edge_10d"}}, {}
    rows = metric_rows.copy()
    rows = rows[rows.get("mask", 0.0).astype(float) > 0.0].copy()
    available = _available_edges_from_prediction_rows(metric_rows)
    resolved, _ = resolve_target_stream_edges(
        available_edges=available,
        train_cfg=train_cfg,
        target_stream_targets_path=target_stream_targets_path or DEFAULT_TARGET_STREAM_TARGETS_PATH,
    )
    stream_names = _resolve_stream_names(metric_rows)
    seen_props = list(dict.fromkeys(metric_rows["property_name"].astype(str).tolist()))
    default_props = main_stream_property_names(None)
    property_names = [p for p in default_props if p in set(seen_props)]
    property_names.extend([p for p in seen_props if p not in set(property_names)])
    thresholds = _target_metric_thresholds(train_cfg)
    excluded_properties = _target_mean_r2_excluded_properties(train_cfg)
    csv_rows: list[dict[str, Any]] = []
    target_metric_rows: list[dict[str, Any]] = []
    split_summary: dict[str, Any] = {}
    for split in sorted(set(metric_rows["split"].astype(str).tolist())):
        split_rows = rows[rows["split"].astype(str) == split]
        property_r2_values: list[float] = []
        property_mae_values: list[float] = []
        property_rmse_values: list[float] = []
        edge_r2_values: list[float] = []
        flat_true: list[float] = []
        flat_pred: list[float] = []
        target_mean_values: list[float] = []
        unstable_count = 0
        used_in_mean_count = 0
        excluded_count = 0
        valid_values = 0
        edge_count = 0
        for key, rec in sorted(resolved.items()):
            pid, edge_id = key
            edge_rows = split_rows[
                (split_rows["process_id"].astype(str).map(_process_text) == pid)
                & (split_rows["canonical_edge_id"].astype(str) == edge_id)
            ]
            if edge_rows.empty:
                continue
            edge_count += 1
            edge_true: list[float] = []
            edge_pred: list[float] = []
            resolved_stream_name = stream_names.get(key, rec.target_stream)
            for prop in property_names:
                prop_rows = edge_rows[edge_rows["property_name"].astype(str) == prop]
                mb = _metric(prop_rows["y_true"], prop_rows["y_pred"])
                valid_values += int(mb["n_samples"])
                if bool(mb["r2_unstable"]):
                    unstable_count += 1
                norm_true_mean, norm_true_std = (
                    _mean_std(prop_rows["norm_y_true"])
                    if "norm_y_true" in prop_rows.columns and not prop_rows.empty
                    else (math.nan, math.nan)
                )
                norm_pred_mean, norm_pred_std = (
                    _mean_std(prop_rows["norm_y_pred"])
                    if "norm_y_pred" in prop_rows.columns and not prop_rows.empty
                    else (math.nan, math.nan)
                )
                inv_mean = _finite_first(prop_rows["inverse_mean_used"]) if "inverse_mean_used" in prop_rows.columns and not prop_rows.empty else math.nan
                inv_std = _finite_first(prop_rows["inverse_std_used"]) if "inverse_std_used" in prop_rows.columns and not prop_rows.empty else math.nan
                base_row = {
                    "split": split,
                    "process_id": rec.process_num,
                    "target_id": _target_ids(rec),
                    "target_stream": rec.target_stream,
                    "canonical_edge_id": rec.canonical_edge_id,
                    "resolved_edge_id": rec.canonical_edge_id,
                    "resolved_stream_name": resolved_stream_name,
                    "target_species_list": _species_list(rec),
                    "metric_scope": "target_edge_10d",
                    "feature_name": prop,
                    "property_name": prop,
                    **mb,
                    "n": mb["n_samples"],
                    "sst": mb["SST"],
                    "sse": mb["SSE"],
                    "mae": mb["MAE"],
                    "rmse": mb["RMSE"],
                    "r2": mb["R2"],
                    "norm_true_mean": norm_true_mean,
                    "norm_true_std": norm_true_std,
                    "norm_pred_mean": norm_pred_mean,
                    "norm_pred_std": norm_pred_std,
                    "orig_true_mean": mb["true_mean"],
                    "orig_true_std": mb["true_std"],
                    "orig_pred_mean": mb["pred_mean"],
                    "orig_pred_std": mb["pred_std"],
                    "inverse_mean_used": inv_mean,
                    "inverse_std_used": inv_std,
                }
                csv_rows.append(base_row)
                target_row = _target_row_from_metric_row(
                    base_row,
                    thresholds,
                    excluded_properties=excluded_properties,
                )
                target_metric_rows.append(target_row)
                if bool(target_row["target_mean_r2_member"]) and math.isfinite(float(target_row["r2"])):
                    if math.isfinite(float(target_row["mae"])):
                        property_mae_values.append(float(target_row["mae"]))
                    if math.isfinite(float(target_row["rmse"])):
                        property_rmse_values.append(float(target_row["rmse"]))
                    property_r2_values.append(float(target_row["r2"]))
                    target_mean_values.append(float(target_row["r2"]))
                    if not prop_rows.empty:
                        vals_true = prop_rows["y_true"].astype(float).tolist()
                        vals_pred = prop_rows["y_pred"].astype(float).tolist()
                        edge_true.extend(vals_true)
                        edge_pred.extend(vals_pred)
                        flat_true.extend(vals_true)
                        flat_pred.extend(vals_pred)
                    used_in_mean_count += 1
                else:
                    excluded_count += 1
            if len(edge_true) >= 2:
                emb = _metric(pd.Series(edge_true), pd.Series(edge_pred))
                edge_metric_member = _target_row_from_metric_row(
                    {
                        "split": split,
                        "process_id": rec.process_num,
                        "canonical_edge_id": rec.canonical_edge_id,
                        "target_stream": rec.target_stream,
                        "feature_name": "__edge_flatten__",
                        **emb,
                    },
                    thresholds,
                )
                if bool(edge_metric_member["target_mean_r2_member"]) and math.isfinite(float(edge_metric_member["r2"])):
                    edge_r2_values.append(float(emb["R2"]))
        flat_mb = _metric(pd.Series(flat_true), pd.Series(flat_pred))
        split_summary[split] = {
            "target_edge_10d_mae_property_macro": _mean(property_mae_values),
            "target_edge_10d_rmse_property_macro": _mean(property_rmse_values),
            "target_edge_10d_r2_property_macro": _mean(property_r2_values),
            "target_edge_10d_r2_edge_macro": _mean(edge_r2_values),
            "target_edge_10d_r2_flatten": flat_mb["R2"],
            "target_edge_10d_num_edges": edge_count,
            "target_edge_10d_num_properties": len(property_names),
            "target_edge_10d_num_valid_values": valid_values,
            "target_edge_10d_num_r2_unstable": unstable_count,
            "target_edge_10d_used_in_mean_count": used_in_mean_count,
            "target_edge_10d_excluded_count": excluded_count,
            "target_mean_r2": _mean(target_mean_values),
            "target_r2": _mean(target_mean_values),
        }
    feature_mapping = _feature_mapping_rows_from_metric_rows(metric_rows)
    payload = {
        "metadata": {
            "metric_type": "target_edge_10d",
            "metric_scope": "target_edge_10d",
            "target_mapping_source": str(target_stream_targets_path or DEFAULT_TARGET_STREAM_TARGETS_PATH),
            "property_names": property_names,
            "full_y_edge_feature_names": _full_feature_names_from_metric_rows(metric_rows),
            "pi_main_feature_names_10d": list(property_names),
            "pi_main_feature_indices_in_y_edge": [row.get("true_feature_index_in_edge_target_columns", "") for row in feature_mapping],
            "pred_10d_feature_names": [row.get("pred_10d_feature_name", "") for row in feature_mapping],
            "true_10d_feature_names": [row.get("true_10d_feature_name", "") for row in feature_mapping],
            "species_only_metrics_are_primary": False,
            "target_metric_thresholds": thresholds,
            "target_mean_r2_excluded_properties": sorted(excluded_properties),
        },
        "splits": split_summary,
        "target_r2": target_metric_rows,
        "feature_mapping": feature_mapping,
    }
    scalars = target_edge_10d_scalar_metrics(payload)
    return pd.DataFrame(csv_rows), payload, scalars


def target_edge_10d_scalar_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    splits = payload.get("splits", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(splits, Mapping):
        return out
    for split, data in splits.items():
        if not isinstance(data, Mapping):
            continue
        for key, value in data.items():
            if isinstance(value, bool):
                val = float(value)
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                val = float(value)
            else:
                continue
            out[f"{split}_{key}"] = val
            if str(split) == "val":
                out[f"val_{key}"] = val
                out[f"eval_{key}"] = val
                out[key] = val
    return out


def build_target_edge_r2_by_property(rows: pd.DataFrame) -> pd.DataFrame:
    columns = ["split", "metric_scope", "property_name", "n", "SST", "SSE", "R2"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    frame = rows.copy()
    if "property_name" not in frame.columns and "feature_name" in frame.columns:
        frame["property_name"] = frame["feature_name"]
    needed = {"split", "property_name", "n", "true_mean", "sst", "sse"}
    missing = needed - set(frame.columns)
    if missing:
        return pd.DataFrame(columns=columns)
    out_rows: list[dict[str, Any]] = []
    for (split, prop), group in frame.groupby(["split", "property_name"], dropna=False):
        valid = group.copy()
        valid["n"] = pd.to_numeric(valid["n"], errors="coerce").fillna(0.0)
        valid = valid[valid["n"] > 0]
        if valid.empty:
            continue
        n_values = pd.to_numeric(valid["n"], errors="coerce").fillna(0.0)
        n_total = float(n_values.sum())
        if n_total <= 0:
            continue
        means = pd.to_numeric(valid["true_mean"], errors="coerce").fillna(0.0)
        global_mean = float((means * n_values).sum() / n_total)
        sst = float(
            (
                pd.to_numeric(valid["sst"], errors="coerce").fillna(0.0)
                + n_values * (means - global_mean) ** 2
            ).sum()
        )
        sse = float(pd.to_numeric(valid["sse"], errors="coerce").fillna(0.0).sum())
        out_rows.append(
            {
                "split": str(split),
                "metric_scope": "target_edges_by_property",
                "property_name": str(prop),
                "n": n_total,
                "SST": sst,
                "SSE": sse,
                "R2": (
                    math.nan
                    if n_total < 2
                    else float(1.0 - sse / sst)
                    if sst > R2_SST_EPS
                    else CONSTANT_TARGET_R2_VALUE
                ),
            }
        )
    return pd.DataFrame(out_rows, columns=columns).sort_values(["split", "property_name"]).reset_index(drop=True)


def build_target_edge_internal_metrics_by_property(rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize pooled and equal-edge target metrics without stability filtering."""

    columns = [
        "split",
        "metric_scope",
        "property_name",
        "edge_count",
        "n",
        "MAE",
        "RMSE",
        "pooled_R2",
        "edge_macro_R2",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    frame = rows.copy()
    if "property_name" not in frame.columns and "feature_name" in frame.columns:
        frame["property_name"] = frame["feature_name"]
    pooled = build_target_edge_r2_by_property(frame)
    out_rows: list[dict[str, Any]] = []
    for (split, prop), group in frame.groupby(["split", "property_name"], dropna=False):
        n_values = pd.to_numeric(group.get("n"), errors="coerce").fillna(0.0)
        valid_n = n_values > 0
        if not bool(valid_n.any()):
            continue
        valid = group.loc[valid_n]
        n_values = n_values.loc[valid_n]
        n_total = float(n_values.sum())
        mae_values = pd.to_numeric(valid.get("mae"), errors="coerce")
        sse_values = pd.to_numeric(valid.get("sse"), errors="coerce")
        r2_values = pd.to_numeric(valid.get("r2"), errors="coerce")
        finite_r2 = r2_values[r2_values.map(lambda value: math.isfinite(float(value)))]
        pooled_row = pooled[
            (pooled["split"].astype(str) == str(split))
            & (pooled["property_name"].astype(str) == str(prop))
        ]
        pooled_r2 = float(pooled_row.iloc[0]["R2"]) if not pooled_row.empty else math.nan
        out_rows.append(
            {
                "split": str(split),
                "metric_scope": "target_edges_internal_by_property",
                "property_name": str(prop),
                "edge_count": int(len(valid)),
                "n": n_total,
                "MAE": float((mae_values.fillna(0.0) * n_values).sum() / n_total),
                "RMSE": float(math.sqrt(max(0.0, float(sse_values.fillna(0.0).sum()) / n_total))),
                "pooled_R2": pooled_r2,
                "edge_macro_R2": float(finite_r2.mean()) if len(finite_r2) else math.nan,
            }
        )
    return pd.DataFrame(out_rows, columns=columns).sort_values(
        ["split", "property_name"]
    ).reset_index(drop=True)


def target_edge_internal_metrics_scalars(rows: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for _, row in rows.iterrows():
        split = str(row["split"])
        slug = _metric_key_slug(row["property_name"])
        for column, metric_name in (
            ("pooled_R2", "pooled_r2"),
            ("edge_macro_R2", "edge_macro_r2"),
            ("MAE", "mae"),
            ("RMSE", "rmse"),
        ):
            try:
                value = float(row[column])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                out[f"{split}_target_internal_{metric_name}_{slug}"] = value
    return out


def target_edge_r2_by_property_scalars(rows: pd.DataFrame, *, train_cfg: Any = None) -> dict[str, float]:
    out: dict[str, float] = {}
    val_r2_values: list[float] = []
    for _, row in rows.iterrows():
        try:
            value = float(row["R2"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        if str(row["property_name"]) not in LOGGED_TARGET_EDGE_R2_PROPERTIES:
            continue
        split = str(row["split"])
        key = f"{split}_target_edge_r2_{_metric_key_slug(row['property_name'])}"
        out[key] = value
        if split == "val":
            out[key[4:] if key.startswith("val_") else key] = value
            if str(row["property_name"]) in PI_MAIN_FEATURE_NAMES_10D:
                val_r2_values.append(value)
    if val_r2_values:
        raw_mean = float(sum(val_r2_values) / len(val_r2_values))
        out["val_target_edge_property_mean_r2"] = raw_mean
        out["target_edge_property_mean_r2"] = raw_mean
    return out


def build_target_edge_r2_by_property_relevant(
    rows: pd.DataFrame,
    *,
    train_cfg: Any,
) -> pd.DataFrame:
    columns = [
        "split",
        "metric_scope",
        "property_name",
        "relevance_rule",
        "n",
        "SST",
        "SSE",
        "MAE",
        "RMSE",
        "R2",
        "true_mean",
        "pred_mean",
    ]
    if rows.empty or not _metric_relevance_enabled(train_cfg):
        return pd.DataFrame(columns=columns)
    frame = rows.copy()
    if "property_name" not in frame.columns and "feature_name" in frame.columns:
        frame["property_name"] = frame["feature_name"]
    needed = {"split", "property_name", "y_true", "y_pred"}
    missing = needed - set(frame.columns)
    if missing:
        return pd.DataFrame(columns=columns)
    fraction_threshold = _metric_relevance_fraction_threshold(train_cfg)
    flow_threshold = _metric_relevance_flow_threshold(train_cfg)
    out_rows: list[dict[str, Any]] = []
    for (split, prop), group in frame.groupby(["split", "property_name"], dropna=False):
        prop_name = str(prop)
        valid = group.copy()
        true = pd.to_numeric(valid["y_true"], errors="coerce")
        pred = pd.to_numeric(valid["y_pred"], errors="coerce")
        mask = true.notna() & pred.notna()
        if _is_fraction_property(prop_name):
            mask = mask & (true.astype(float) > fraction_threshold)
            rule = f"true>{fraction_threshold:g}"
        elif _is_flow_property(prop_name):
            mask = mask & (true.astype(float) > flow_threshold)
            rule = f"true>{flow_threshold:g}"
        else:
            rule = "finite"
        true_v = true[mask].astype(float)
        pred_v = pred[mask].astype(float)
        n = int(len(true_v))
        if n <= 0:
            out_rows.append(
                {
                    "split": str(split),
                    "metric_scope": "target_edges_by_property_relevant",
                    "property_name": prop_name,
                    "relevance_rule": rule,
                    "n": 0,
                    "SST": 0.0,
                    "SSE": 0.0,
                    "MAE": math.nan,
                    "RMSE": math.nan,
                    "R2": math.nan,
                    "true_mean": math.nan,
                    "pred_mean": math.nan,
                }
            )
            continue
        err = pred_v - true_v
        true_mean = float(true_v.mean())
        pred_mean = float(pred_v.mean())
        sst = float(((true_v - true_mean) ** 2).sum())
        sse = float((err**2).sum())
        out_rows.append(
            {
                "split": str(split),
                "metric_scope": "target_edges_by_property_relevant",
                "property_name": prop_name,
                "relevance_rule": rule,
                "n": n,
                "SST": sst,
                "SSE": sse,
                "MAE": float(err.abs().mean()),
                "RMSE": math.sqrt(float((err**2).mean())),
                "R2": (
                    math.nan
                    if n < 2
                    else float(1.0 - sse / sst)
                    if sst > R2_SST_EPS
                    else CONSTANT_TARGET_R2_VALUE
                ),
                "true_mean": true_mean,
                "pred_mean": pred_mean,
            }
        )
    return pd.DataFrame(out_rows, columns=columns).sort_values(["split", "property_name"]).reset_index(drop=True)


def target_edge_r2_by_property_relevant_scalars(rows: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    r2_values: list[float] = []
    for _, row in rows.iterrows():
        try:
            value = float(row["R2"])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        if str(row["property_name"]) not in LOGGED_TARGET_EDGE_R2_PROPERTIES:
            continue
        split = str(row["split"])
        slug = _metric_key_slug(row["property_name"])
        key = f"{split}_target_edge_r2_{slug}"
        out[key] = value
        if split == "val":
            out[key[4:] if key.startswith("val_") else key] = value
            if str(row["property_name"]) in PI_MAIN_FEATURE_NAMES_10D:
                r2_values.append(value)
    if r2_values:
        out["val_target_edge_property_mean_r2"] = float(sum(r2_values) / len(r2_values))
        out["target_edge_property_mean_r2"] = out["val_target_edge_property_mean_r2"]
    return out


def write_metric_rows_actual_vs_pred_plots(
    *,
    out_dir: Path,
    metric_rows: pd.DataFrame,
    properties: Sequence[str],
    root_name: str = "actual_vs_predicted",
) -> list[Path]:
    if metric_rows.empty:
        return []

    # Plotting is optional and imported lazily so normal training pays no import cost.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rows = metric_rows.copy()
    rows = rows[rows["property_name"].astype(str).isin(set(properties))]
    if "mask" in rows.columns:
        rows = rows[pd.to_numeric(rows["mask"], errors="coerce").fillna(0.0) > 0.0]
    if rows.empty:
        return []

    written: list[Path] = []
    for (split, property_name), group in rows.groupby(["split", "property_name"], sort=True):
        actual = pd.to_numeric(group["y_true"], errors="coerce")
        predicted = pd.to_numeric(group["y_pred"], errors="coerce")
        finite = actual.notna() & predicted.notna()
        actual = actual[finite]
        predicted = predicted[finite]
        if actual.empty:
            continue

        split_dir = out_dir / root_name / _metric_key_slug(split)
        split_dir.mkdir(parents=True, exist_ok=True)
        pairs: list[tuple[str, pd.Series, pd.Series, str, str]] = [
            ("", actual, predicted, "Actual", "Predicted")
        ]
        if str(property_name) in {"Mass_Flow", "Mole_Flow", "Vol_Flow"}:
            pairs.append(
                (
                    "_log1p",
                    actual.clip(lower=0.0).map(math.log1p),
                    predicted.clip(lower=0.0).map(math.log1p),
                    "log1p(Actual)",
                    "log1p(Predicted)",
                )
            )

        for suffix, x_values, y_values, x_label, y_label in pairs:
            lower = float(min(x_values.min(), y_values.min()))
            upper = float(max(x_values.max(), y_values.max()))
            if not math.isfinite(lower) or not math.isfinite(upper):
                continue
            span = upper - lower
            pad = max(span * 0.05, 1.0e-9)
            axis_min = lower - pad
            axis_max = upper + pad
            mae = float((x_values - y_values).abs().mean())
            sst = float(((x_values - x_values.mean()) ** 2).sum())
            sse = float(((x_values - y_values) ** 2).sum())
            r2 = (
                math.nan
                if len(x_values) < 2
                else float(1.0 - sse / sst)
                if sst > R2_SST_EPS
                else CONSTANT_TARGET_R2_VALUE
            )

            fig, ax = plt.subplots(figsize=(6.0, 6.0))
            ax.scatter(x_values, y_values, s=10, alpha=0.45, edgecolors="none")
            ax.plot([axis_min, axis_max], [axis_min, axis_max], color="black", linewidth=1.0)
            ax.set_xlim(axis_min, axis_max)
            ax.set_ylim(axis_min, axis_max)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f"{property_name} ({split})")
            r2_text = f"{r2:.5f}" if math.isfinite(r2) else "nan"
            ax.text(
                0.03,
                0.97,
                f"R2={r2_text}\nMAE={mae:.6g}\nn={len(x_values)}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
            ax.grid(alpha=0.2)
            fig.tight_layout()
            path = split_dir / f"{_metric_key_slug(property_name)}{suffix}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            written.append(path)
    return written


def write_target_edge_actual_vs_pred_plots(
    *,
    out_dir: Path,
    metric_rows: pd.DataFrame,
    resolution_audit: pd.DataFrame,
) -> list[Path]:
    if metric_rows.empty or resolution_audit.empty:
        return []

    rows = filter_metric_rows_to_resolved_target_edges(metric_rows, resolution_audit)
    return write_metric_rows_actual_vs_pred_plots(
        out_dir=out_dir,
        metric_rows=rows,
        properties=TARGET_EDGE_ACTUAL_VS_PRED_PROPERTIES,
        root_name="actual_vs_predicted",
    )


def filter_metric_rows_to_resolved_target_edges(
    metric_rows: pd.DataFrame,
    resolution_audit: pd.DataFrame,
) -> pd.DataFrame:
    if metric_rows.empty or resolution_audit.empty:
        return metric_rows.iloc[0:0].copy()
    resolved_edges = {
        (int(row["process_id"]), _clean(row["resolved_edge_id"]))
        for _, row in resolution_audit.iterrows()
        if _clean(row.get("resolution_status")) == "resolved"
        and _process_num(row.get("process_id")) is not None
        and _clean(row.get("resolved_edge_id"))
    }
    if not resolved_edges:
        return metric_rows.iloc[0:0].copy()
    rows = metric_rows.copy()
    rows["_process_num"] = rows["process_id"].map(_process_num)
    rows["_edge_key"] = list(zip(rows["_process_num"], rows["canonical_edge_id"].map(_clean)))
    rows = rows[rows["_edge_key"].isin(resolved_edges)].copy()
    return rows.drop(columns=["_process_num", "_edge_key"], errors="ignore")


class TargetEdge10DAccumulator:
    def __init__(self, *, train_cfg: Any, target_stream_targets_path: Path | str | None = None) -> None:
        self.train_cfg = train_cfg
        self.target_stream_targets_path = target_stream_targets_path
        self.rows: list[dict[str, Any]] = []
        self._row_group_counter = 0

    def update_batch(
        self,
        *,
        export: Any,
        pred_main: torch.Tensor,
        true_main: torch.Tensor,
        mask_main: torch.Tensor,
        property_names: Sequence[str],
        split_name: str,
        edge_target_columns: Sequence[str] | None = None,
        true_feature_indices: Sequence[int] | None = None,
        inverse_mean: Sequence[float] | None = None,
        inverse_std: Sequence[float] | None = None,
    ) -> None:
        if export is None:
            return
        pred = pred_main.detach().cpu()
        true = true_main.detach().cpu()
        mask = mask_main.detach().cpu()
        if pred.ndim == 3:
            pred = pred.reshape(-1, pred.shape[-1])
            true = true.reshape(-1, true.shape[-1])
            mask = mask.reshape(-1, mask.shape[-1])
        if pred.shape != true.shape or pred.shape != mask.shape:
            raise AssertionError(f"target edge metric shape mismatch: {pred.shape}, {true.shape}, {mask.shape}")
        edge_cols = list(edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
        true_indices = list(true_feature_indices) if true_feature_indices is not None else _column_indices(edge_cols, property_names)
        inv_mean = [float(x) for x in inverse_mean] if inverse_mean is not None else [math.nan] * len(property_names)
        inv_std = [float(x) for x in inverse_std] if inverse_std is not None else [math.nan] * len(property_names)
        n = int(pred.shape[0])
        process_ids = list(getattr(export, "process_id", [""] * n) or [""] * n)
        edge_ids = list(getattr(export, "canonical_edge_id", [""] * n) or [""] * n)
        streams = list(getattr(export, "main_data_stream_key", [""] * n) or [""] * n)
        src_nodes = list(getattr(export, "src_node", [""] * n) or [""] * n)
        dst_nodes = list(getattr(export, "dst_node", [""] * n) or [""] * n)
        roles = list(getattr(export, "stream_role", [""] * n) or [""] * n)
        for e in range(n):
            row_group_id = self._row_group_counter
            self._row_group_counter += 1
            for j, prop in enumerate(property_names):
                norm_true = (
                    (float(true[e, j]) - inv_mean[j]) / inv_std[j]
                    if math.isfinite(inv_mean[j]) and math.isfinite(inv_std[j]) and inv_std[j] != 0.0
                    else math.nan
                )
                norm_pred = (
                    (float(pred[e, j]) - inv_mean[j]) / inv_std[j]
                    if math.isfinite(inv_mean[j]) and math.isfinite(inv_std[j]) and inv_std[j] != 0.0
                    else math.nan
                )
                self.rows.append(
                    {
                        "split": split_name,
                        "process_id": _process_text(process_ids[e]),
                        "canonical_edge_id": _clean(edge_ids[e]),
                        "main_data_stream_key": _clean(streams[e]),
                        "resolved_stream_name": _clean(streams[e]),
                        "src_node": _clean(src_nodes[e]),
                        "dst_node": _clean(dst_nodes[e]),
                        "edge_index": int(e),
                        "stream_role": _clean(roles[e]),
                        "metric_row_group_id": int(row_group_id),
                        "property_name": str(prop),
                        "y_true": float(true[e, j]),
                        "y_pred": float(pred[e, j]),
                        "norm_y_true": norm_true,
                        "norm_y_pred": norm_pred,
                        "mask": float(mask[e, j]),
                        "edge_target_columns": "|".join(edge_cols),
                        "pred_feature_index_10d": int(j),
                        "pred_10d_feature_name": str(prop),
                        "true_feature_index_in_edge_target_columns": int(true_indices[j]),
                        "true_10d_feature_name": str(prop),
                        "inverse_mean_used": inv_mean[j],
                        "inverse_std_used": inv_std[j],
                    }
                )

    def finalize(self) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float]]:
        return build_target_edge_10d_metrics(
            metric_rows=pd.DataFrame(self.rows),
            train_cfg=self.train_cfg,
            target_stream_targets_path=self.target_stream_targets_path,
        )

    def finalize_scalars(self) -> dict[str, float]:
        _, _, scalars = self.finalize()
        return scalars


def write_target_edge_10d_metric_artifacts(
    *,
    out_dir: Path,
    train_cfg: Any,
    target_stream_targets_path: Path | str | None = None,
    metric_rows: pd.DataFrame | None = None,
    edge_predictions: pd.DataFrame | None = None,
    save_actual_vs_pred_plots: bool | None = None,
) -> dict[str, float]:
    if metric_rows is None:
        metric_rows = pd.DataFrame() if edge_predictions is None else _metric_rows_from_edge_predictions(edge_predictions)
    csv_df, payload, scalars = build_target_edge_10d_metrics(
        metric_rows=metric_rows,
        train_cfg=train_cfg,
        target_stream_targets_path=target_stream_targets_path,
    )
    property_names = list(payload.get("metadata", {}).get("property_names", main_stream_property_names(None)))
    resolution_audit = build_target_edge_resolution_audit(
        metric_rows=metric_rows,
        train_cfg=train_cfg,
        target_stream_targets_path=target_stream_targets_path,
    )
    topology_audit = build_process_edge_topology_audit(metric_rows=metric_rows, resolution_audit=resolution_audit)
    validation = validate_target_edge_10d_resolution(
        metrics_df=csv_df,
        resolution_audit=resolution_audit,
        topology_audit=topology_audit,
        property_names=property_names,
    )
    payload["resolution_audit"] = validation
    out_dir.mkdir(parents=True, exist_ok=True)
    should_save_plots = (
        bool(_cfg_get(train_cfg, "save_target_edge_actual_vs_pred_plots", False))
        if save_actual_vs_pred_plots is None
        else bool(save_actual_vs_pred_plots)
    )
    if should_save_plots:
        plot_paths = write_target_edge_actual_vs_pred_plots(
            out_dir=out_dir,
            metric_rows=metric_rows,
            resolution_audit=resolution_audit,
        )
        plot_root = out_dir / "actual_vs_predicted"
        plot_root.mkdir(parents=True, exist_ok=True)
        (plot_root / "manifest.json").write_text(
            json.dumps(
                {"files": [str(path.relative_to(out_dir)) for path in plot_paths]},
                indent=2,
            ),
            encoding="utf-8",
        )
    csv_df.to_csv(out_dir / "target_edge_10d_metrics.csv", index=False)
    feature_mapping_df = pd.DataFrame(payload.get("feature_mapping", []))
    feature_mapping_df.to_csv(out_dir / "target_edge_10d_feature_mapping.csv", index=False)
    (out_dir / "target_edge_10d_feature_mapping.json").write_text(
        json.dumps(
            {
                "full_y_edge_feature_names": payload.get("metadata", {}).get("full_y_edge_feature_names", []),
                "pi_main_feature_names_10d": payload.get("metadata", {}).get("pi_main_feature_names_10d", []),
                "pi_main_feature_indices_in_14d": payload.get("metadata", {}).get("pi_main_feature_indices_in_y_edge", []),
                "pred_10d_feature_names": payload.get("metadata", {}).get("pred_10d_feature_names", []),
                "true_10d_feature_names": payload.get("metadata", {}).get("true_10d_feature_names", []),
                "rows": payload.get("feature_mapping", []),
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    target_r2_df = pd.DataFrame(payload.get("target_r2", []))
    target_r2_df.to_csv(out_dir / "target_r2.csv", index=False)
    target_r2_df.to_csv(out_dir / "target_edge_feature_metrics.csv", index=False)
    target_r2_by_property_raw_df = build_target_edge_r2_by_property(target_r2_df)
    target_r2_by_property_raw_df.to_csv(out_dir / "target_edge_r2_by_property_raw.csv", index=False)
    target_internal_df = build_target_edge_internal_metrics_by_property(target_r2_df)
    target_internal_df.to_csv(out_dir / "target_edge_internal_metrics_by_property.csv", index=False)
    target_metric_rows = filter_metric_rows_to_resolved_target_edges(metric_rows, resolution_audit)
    target_r2_by_property_raw_df.to_csv(out_dir / "target_edge_r2_by_property.csv", index=False)
    property_scalars = target_edge_r2_by_property_scalars(target_r2_by_property_raw_df, train_cfg=train_cfg)
    # The by-property aggregate is a fallback for configurations where the
    # direct target-row mean is unavailable. Do not overwrite a valid direct
    # mean: it carries the configured count/SST/R2/property exclusion policy.
    direct_mean_keys = {
        "val_target_mean_r2",
        "target_mean_r2",
        "val_target_r2",
        "target_r2",
    }
    for key, value in property_scalars.items():
        if key in direct_mean_keys and key in scalars and math.isfinite(float(scalars[key])):
            continue
        scalars[key] = value
    scalars.update(target_edge_internal_metrics_scalars(target_internal_df))
    scalars.update(
        write_oracle_diagnostic_metric_artifacts(
            out_dir=out_dir,
            metric_rows=target_metric_rows,
            train_cfg=train_cfg,
            scope="target",
            file_prefix="target_edge",
        )
    )
    scalars.update(
        write_diagnostic_display_metric_artifacts(
            out_dir=out_dir,
            metric_rows=target_metric_rows,
            train_cfg=train_cfg,
            scope="target",
            file_prefix="target_edge",
        )
    )
    (out_dir / "target_r2.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "metric_type": "target_r2",
                    "target_mapping_source": str(target_stream_targets_path or DEFAULT_TARGET_STREAM_TARGETS_PATH),
                    "target_metric_thresholds": payload.get("metadata", {}).get("target_metric_thresholds", {}),
                    "relevant_r2_scope": "deprecated; official target_edge_r2_by_property.csv is raw/unfiltered",
                    "metric_relevance_enabled": _metric_relevance_enabled(train_cfg),
                    "metric_relevance_fraction_threshold": _metric_relevance_fraction_threshold(train_cfg),
                    "metric_relevance_flow_threshold": _metric_relevance_flow_threshold(train_cfg),
                },
                "splits": {
                    split: {"target_r2": data.get("target_r2"), "target_mean_r2": data.get("target_mean_r2")}
                    for split, data in (payload.get("splits", {}) or {}).items()
                    if isinstance(data, Mapping)
                },
                "rows": payload.get("target_r2", []),
                "property_rows": target_r2_by_property_raw_df.to_dict(orient="records"),
                "property_raw_rows": target_r2_by_property_raw_df.to_dict(orient="records"),
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    (out_dir / "target_edge_feature_metrics.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "metric_type": "target_edge_feature_metrics",
                    "target_mapping_source": str(target_stream_targets_path or DEFAULT_TARGET_STREAM_TARGETS_PATH),
                    "target_metric_thresholds": payload.get("metadata", {}).get("target_metric_thresholds", {}),
                    "feature_names": payload.get("metadata", {}).get("property_names", []),
                    "r2_scope": "split_accumulated_target_edge_direct_features",
                },
                "splits": {
                    split: {"target_r2": data.get("target_r2"), "target_mean_r2": data.get("target_mean_r2")}
                    for split, data in (payload.get("splits", {}) or {}).items()
                    if isinstance(data, Mapping)
                },
                "rows": payload.get("target_r2", []),
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    try:
        target_r2_df.to_excel(out_dir / "target_r2.xlsx", index=False)
        target_r2_df.to_excel(out_dir / "target_edge_feature_metrics.xlsx", index=False)
    except Exception:
        pass
    resolution_audit.to_csv(out_dir / "target_edge_resolution_audit.csv", index=False)
    topology_audit.to_csv(out_dir / "process_edge_topology_audit.csv", index=False)
    (out_dir / "target_edge_10d_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    (out_dir / "target_edge_resolution_validation.json").write_text(
        json.dumps(validation, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    _print_target_edge_list(csv_df)
    if not validation.get("ok", False):
        print(f"[TargetEdgeMetric][WARN] resolution audit errors: {validation.get('errors', [])}", flush=True)
    return scalars


def _target_edge_property_scalar_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    splits = payload.get("splits", {}) if isinstance(payload, Mapping) else {}
    if not isinstance(splits, Mapping):
        return out
    for split, data in splits.items():
        if not isinstance(data, Mapping):
            continue
        for key, value in data.items():
            if isinstance(value, bool):
                val = float(value)
            elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                val = float(value)
            else:
                continue
            if key == "target_mean_r2":
                prop_key = "target_edge_derived_property_mean_r2"
            elif key == "target_r2":
                prop_key = "target_edge_derived_property_r2"
            elif key.startswith("target_edge_10d"):
                prop_key = key.replace("target_edge_10d", "target_edge_derived_property")
            else:
                continue
            out[f"{split}_{prop_key}"] = val
            if str(split) == "val":
                out[f"val_{prop_key}"] = val
                out[f"eval_{prop_key}"] = val
                out[prop_key] = val
    return out


def write_target_edge_property_metric_artifacts(
    *,
    out_dir: Path,
    train_cfg: Any,
    target_stream_targets_path: Path | str | None = None,
    metric_rows: pd.DataFrame | None = None,
) -> dict[str, float]:
    metric_rows = pd.DataFrame() if metric_rows is None else metric_rows
    csv_df, payload, _ = build_target_edge_10d_metrics(
        metric_rows=metric_rows,
        train_cfg=train_cfg,
        target_stream_targets_path=target_stream_targets_path,
    )
    if not csv_df.empty and "metric_scope" in csv_df.columns:
        csv_df = csv_df.copy()
        csv_df["metric_scope"] = "target_edge_property"
    metadata = payload.setdefault("metadata", {})
    metadata["metric_type"] = "target_edge_property"
    metadata["metric_scope"] = "target_edge_property"
    metadata["property_names"] = list(dict.fromkeys(metric_rows.get("property_name", pd.Series(dtype=str)).astype(str).tolist()))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(out_dir / "target_edge_property_metrics.csv", index=False)
    (out_dir / "target_edge_property_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    return _target_edge_property_scalar_metrics(payload)


@torch.no_grad()
def collect_target_edge_10d_metric_rows_from_loader(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    train_cfg: Any,
    data_cfg: Any,
    use_amp: bool,
    split_name: str,
    target_stream_targets_path: Path | str | None = None,
    normalizer: Mapping[str, Any] | None = None,
    progress_label: str | None = None,
    progress_interval: int = 50,
) -> pd.DataFrame:
    was_training = model.training
    model.eval()
    acc = TargetEdge10DAccumulator(train_cfg=train_cfg, target_stream_targets_path=target_stream_targets_path)
    total_batches = _maybe_len_loader(loader)
    started_at = time.perf_counter()
    for batch_idx, batch in enumerate(loader):
        batch_data = {k: v.to(device) for k, v in batch.model_kwargs.items()}
        targets = {k: v.to(device) for k, v in batch.targets.items()}
        target_masks = {k: v.to(device) for k, v in batch.target_masks.items()}
        task_inputs = {h: {k: v.to(device) for k, v in p.items()} for h, p in batch.task_inputs.items()}
        edge_cols = list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
        target_edge_mask, _target_mask_info = build_target_edge_boolean_mask(
            train_cfg=train_cfg,
            edge_export_meta=batch.edge_export_meta,
            edge_target_columns=edge_cols,
            n_edges=int(targets["edge_stream"].shape[0]),
            device=device,
        )
        model_batch_data = dict(batch_data)
        model_batch_data["target_edge_mask"] = target_edge_mask
        amp_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=amp_device, enabled=bool(use_amp)):
            outputs = model(model_batch_data, task_inputs=task_inputs)
        pred_main, true_main, mask_main, prop_names = extract_main_stream_metric_tensors(
            outputs=outputs,
            targets_raw=targets,
            target_masks=target_masks,
            train_cfg=train_cfg,
            data_cfg=data_cfg,
            edge_target_columns=edge_cols,
            normalizer=normalizer,
        )
        inv_mean, inv_std = _normalizer_lists_for_names(
            normalizer,
            prop_names,
            data_cfg=data_cfg,
            device=pred_main.device,
            dtype=pred_main.dtype,
        )
        acc.update_batch(
            export=batch.edge_export_meta,
            pred_main=pred_main,
            true_main=true_main,
            mask_main=mask_main,
            property_names=prop_names,
            split_name=split_name,
            edge_target_columns=edge_cols,
            inverse_mean=inv_mean,
            inverse_std=inv_std,
        )
        _print_eval_progress(
            label=progress_label,
            start_time=started_at,
            current=int(batch_idx) + 1,
            total=total_batches,
            interval=progress_interval,
        )
    model.train(was_training)
    return pd.DataFrame(acc.rows)


@torch.no_grad()
def collect_target_edge_property_metric_rows_from_loader(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    train_cfg: Any,
    data_cfg: Any,
    use_amp: bool,
    split_name: str,
    target_stream_targets_path: Path | str | None = None,
    normalizer: Mapping[str, Any] | None = None,
    progress_label: str | None = None,
    progress_interval: int = 50,
) -> pd.DataFrame:
    was_training = model.training
    model.eval()
    acc = TargetEdge10DAccumulator(train_cfg=train_cfg, target_stream_targets_path=target_stream_targets_path)
    total_batches = _maybe_len_loader(loader)
    started_at = time.perf_counter()
    for batch_idx, batch in enumerate(loader):
        batch_data = {k: v.to(device) for k, v in batch.model_kwargs.items()}
        targets = {k: v.to(device) for k, v in batch.targets.items()}
        target_masks = {k: v.to(device) for k, v in batch.target_masks.items()}
        task_inputs = {h: {k: v.to(device) for k, v in p.items()} for h, p in batch.task_inputs.items()}
        edge_cols = list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
        target_edge_mask, _target_mask_info = build_target_edge_boolean_mask(
            train_cfg=train_cfg,
            edge_export_meta=batch.edge_export_meta,
            edge_target_columns=edge_cols,
            n_edges=int(targets["edge_stream"].shape[0]),
            device=device,
        )
        model_batch_data = dict(batch_data)
        model_batch_data["target_edge_mask"] = target_edge_mask
        amp_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=amp_device, enabled=bool(use_amp)):
            outputs = model(model_batch_data, task_inputs=task_inputs)
        pred_prop, true_prop, mask_prop, prop_names = extract_pi_property_metric_tensors(
            outputs=outputs,
            targets_raw=targets,
            target_masks=target_masks,
            data_cfg=data_cfg,
            edge_target_columns=edge_cols,
            normalizer=normalizer,
        )
        inv_mean, inv_std = _normalizer_lists_for_names(
            normalizer,
            prop_names,
            data_cfg=data_cfg,
            device=pred_prop.device,
            dtype=pred_prop.dtype,
        )
        acc.update_batch(
            export=batch.edge_export_meta,
            pred_main=pred_prop,
            true_main=true_prop,
            mask_main=mask_prop,
            property_names=prop_names,
            split_name=split_name,
            edge_target_columns=edge_cols,
            inverse_mean=inv_mean,
            inverse_std=inv_std,
        )
        _print_eval_progress(
            label=progress_label,
            start_time=started_at,
            current=int(batch_idx) + 1,
            total=total_batches,
            interval=progress_interval,
        )
    model.train(was_training)
    return pd.DataFrame(acc.rows)


def _metric_rows_from_edge_predictions(edge_predictions: pd.DataFrame) -> pd.DataFrame:
    if edge_predictions.empty:
        return pd.DataFrame()
    required = {
        "split",
        "process_id",
        "canonical_edge_id",
        "main_data_stream_key",
        "property_name",
        "y_true_orig",
        "y_pred_orig",
        "y_edge_mask",
    }
    missing = sorted(required - set(edge_predictions.columns))
    if missing:
        raise RuntimeError(f"edge_predictions missing columns for target edge 10D metrics: {missing}")
    out = edge_predictions.rename(
        columns={
            "y_true_orig": "y_true",
            "y_pred_orig": "y_pred",
            "y_edge_mask": "mask",
        }
    ).copy()
    names = set(main_stream_property_names(None))
    out = out[out["property_name"].astype(str).isin(names)].copy()
    cols = [
        "split",
        "process_id",
        "canonical_edge_id",
        "main_data_stream_key",
        "property_name",
        "y_true",
        "y_pred",
        "mask",
    ]
    for optional in ("src_node", "dst_node", "edge_index", "stream_role", "resolved_stream_name"):
        if optional in out.columns:
            cols.append(optional)
    return out[cols]


def _print_target_edge_list(csv_df: pd.DataFrame) -> None:
    if csv_df.empty:
        print("[TargetEdgeMetric] no target edge 10D rows generated", flush=True)
        return
    base = csv_df[["process_id", "target_stream", "resolved_edge_id", "resolved_stream_name", "target_species_list"]].drop_duplicates()
    for pid in sorted(base["process_id"].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        print(f"[TargetEdgeMetric] Process {pid}", flush=True)
        sub = base[base["process_id"] == pid]
        for _, row in sub.iterrows():
            print(
                "  "
                f"target_stream={row['target_stream']}, "
                f"resolved_edge_id={row['resolved_edge_id']}, "
                f"resolved_stream_name={row['resolved_stream_name']}, "
                f"target_species_list={row['target_species_list']}",
                flush=True,
            )

