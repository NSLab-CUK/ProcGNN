"""Target stream edge-level loss weighting and feature metrics for edge_all."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from ..constants import STREAM_EDGE_FEATURE_SLOTS
from .target_v4_metrics import normalize_stream_key

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TARGET_STREAM_TARGETS_PATH = PROJECT_ROOT / "data/reference/v4/target_stream_targets.csv"


@dataclass(frozen=True)
class TargetStreamRow:
    process_id: str
    process_num: int
    target_id: str
    target_stream: str
    target_species: str
    target_feature: str
    target_formula: str
    canonical_edge_id: str
    required_stream_key: str
    source: str = "target_stream_targets.csv"
    weight: float | None = None
    fallback_stream_key: str = ""


@dataclass
class ResolvedTargetStreamEdge:
    process_id: str
    process_num: int
    canonical_edge_id: str
    target_stream: str
    edge_weight: float
    weight_source: str
    target_rows: list[TargetStreamRow] = field(default_factory=list)
    yaml_rows: list[TargetStreamRow] = field(default_factory=list)


def _process_num(value: Any) -> int | None:
    text = str(value or "").strip()
    m = re.search(r"(\d+)", text)
    return int(m.group(1)) if m else None


def _process_text(value: Any) -> str:
    n = _process_num(value)
    return f"Process{n}" if n is not None else str(value or "").strip()


def _clean(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def _infer_process_from_row(row: Mapping[str, Any]) -> str:
    for key in ("process_id", "process", "process_num"):
        if _clean(row.get(key)):
            return _process_text(row.get(key))
    for key in ("target_id", "canonical_answer_edge_id"):
        n = _process_num(row.get(key))
        if n is not None:
            return f"Process{n}"
    return ""


def _row_from_mapping(row: Mapping[str, Any], *, source: str) -> TargetStreamRow:
    pid = _infer_process_from_row(row)
    pnum = _process_num(pid) or 0
    weight_raw = row.get("weight", None)
    weight: float | None
    try:
        weight = float(weight_raw) if weight_raw is not None and _clean(weight_raw) else None
    except (TypeError, ValueError):
        weight = None
    return TargetStreamRow(
        process_id=pid,
        process_num=pnum,
        target_id=_clean(row.get("target_id")),
        target_stream=_clean(row.get("target_stream") or row.get("target_stream_node")),
        target_species=_clean(row.get("target_species")),
        target_feature=_clean(row.get("target_feature") or row.get("target_feature_name")),
        target_formula=_clean(row.get("target_formula")),
        canonical_edge_id=_clean(row.get("canonical_answer_edge_id") or row.get("canonical_edge_id")),
        required_stream_key=normalize_stream_key(
            row.get("required_stream_key") or row.get("main_data_stream_key") or row.get("target_stream")
        ),
        source=source,
        weight=weight,
        fallback_stream_key=normalize_stream_key(row.get("fallback_stream_key")),
    )


def load_target_stream_rows(path: Path | str | None = None) -> list[TargetStreamRow]:
    csv_path = Path(path) if path else DEFAULT_TARGET_STREAM_TARGETS_PATH
    if not csv_path.is_file():
        return []
    frame = pd.read_csv(csv_path, dtype={"process_id": int})
    rows: list[TargetStreamRow] = []
    for rec in frame.to_dict(orient="records"):
        rows.append(_row_from_mapping(rec, source="target_stream_targets.csv"))
    return rows


def _yaml_weight_rows(train_cfg: Any) -> list[TargetStreamRow]:
    raw = getattr(train_cfg, "target_stream_loss_weights", None) or []
    records: list[Mapping[str, Any]] = []
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if isinstance(value, Mapping):
                rec = dict(value)
                rec.setdefault("canonical_answer_edge_id", key)
                records.append(rec)
            else:
                records.append({"canonical_answer_edge_id": key, "weight": value})
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        records = [r for r in raw if isinstance(r, Mapping)]
    return [_row_from_mapping(r, source="yaml") for r in records]


def _available_edges_from_export_meta(export_meta: Any) -> list[dict[str, Any]]:
    if export_meta is None:
        return []
    n = len(getattr(export_meta, "process_id", []) or [])
    edge_ids = list(getattr(export_meta, "canonical_edge_id", [""] * n) or [""] * n)
    streams = list(getattr(export_meta, "main_data_stream_key", [""] * n) or [""] * n)
    out: list[dict[str, Any]] = []
    for i in range(n):
        pid = _process_text(export_meta.process_id[i])
        out.append(
            {
                "edge_index": i,
                "process_id": pid,
                "process_num": _process_num(pid) or 0,
                "canonical_edge_id": _clean(edge_ids[i]),
                "stream_key_norm": normalize_stream_key(streams[i]),
            }
        )
    return out


def _available_edges_from_prediction_df(edge_predictions: pd.DataFrame) -> list[dict[str, Any]]:
    if edge_predictions.empty:
        return []
    cols = ["split", "process_id", "canonical_edge_id", "main_data_stream_key"]
    base = edge_predictions[[c for c in cols if c in edge_predictions.columns]].drop_duplicates()
    out: list[dict[str, Any]] = []
    for _, row in base.iterrows():
        pid = _process_text(row.get("process_id"))
        out.append(
            {
                "edge_index": None,
                "split": _clean(row.get("split")),
                "process_id": pid,
                "process_num": _process_num(pid) or 0,
                "canonical_edge_id": _clean(row.get("canonical_edge_id")),
                "stream_key_norm": normalize_stream_key(row.get("main_data_stream_key")),
            }
        )
    return out


def _match_row_to_edges(row: TargetStreamRow, available: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], str]:
    same_process = [e for e in available if _process_text(e.get("process_id")) == row.process_id]
    if not same_process:
        return [], "no_edges_for_process"
    if row.canonical_edge_id:
        matches = [e for e in same_process if _clean(e.get("canonical_edge_id")) == row.canonical_edge_id]
        if matches:
            return matches, ""
        return [], "canonical_answer_edge_id_not_found_in_process"
    for stream_key in (row.fallback_stream_key, row.required_stream_key, normalize_stream_key(row.target_stream)):
        if not stream_key:
            continue
        matches = [e for e in same_process if normalize_stream_key(e.get("stream_key_norm")) == stream_key]
        if matches:
            return matches, ""
    return [], "no_canonical_edge_for_target_stream"


def _apply_conflict(old: float, new: float, policy: str) -> float:
    policy_l = str(policy or "max").strip().lower()
    if policy_l == "max":
        return max(float(old), float(new))
    if policy_l in {"last", "override"}:
        return float(new)
    if policy_l == "first":
        return float(old)
    return max(float(old), float(new))


def resolve_target_stream_edges(
    *,
    available_edges: Sequence[Mapping[str, Any]],
    train_cfg: Any,
    target_stream_targets_path: Path | str | None = None,
) -> tuple[dict[tuple[str, str], ResolvedTargetStreamEdge], list[dict[str, Any]]]:
    default_weight = float(getattr(train_cfg, "target_stream_loss_weight", 5.0))
    conflict_policy = str(getattr(train_cfg, "target_stream_weight_conflict_policy", "max"))
    allow_yaml_only = bool(getattr(train_cfg, "allow_yaml_only_target_streams", False))
    csv_rows = load_target_stream_rows(target_stream_targets_path)
    yaml_rows = _yaml_weight_rows(train_cfg)
    resolved: dict[tuple[str, str], ResolvedTargetStreamEdge] = {}
    diagnostics: list[dict[str, Any]] = []

    for row in csv_rows:
        matches, skip = _match_row_to_edges(row, available_edges)
        if not matches:
            diagnostics.append(_diag_row(row, matched=False, skip_reason=skip, conflict_policy=conflict_policy))
            continue
        for edge in matches:
            key = (_process_text(edge.get("process_id")), _clean(edge.get("canonical_edge_id")))
            if not key[1]:
                diagnostics.append(_diag_row(row, matched=False, skip_reason="missing_canonical_edge_id", conflict_policy=conflict_policy))
                continue
            if key not in resolved:
                resolved[key] = ResolvedTargetStreamEdge(
                    process_id=key[0],
                    process_num=_process_num(key[0]) or 0,
                    canonical_edge_id=key[1],
                    target_stream=row.target_stream or row.required_stream_key,
                    edge_weight=default_weight,
                    weight_source="target_stream_targets_csv",
                    target_rows=[],
                )
            if row.target_id not in {r.target_id for r in resolved[key].target_rows}:
                resolved[key].target_rows.append(row)

    csv_edge_keys = set(resolved.keys())
    for row in yaml_rows:
        matches, skip = _match_row_to_edges(row, available_edges)
        if not matches:
            diagnostics.append(_diag_row(row, matched=False, skip_reason=skip, conflict_policy=conflict_policy))
            continue
        if row.weight is None:
            diagnostics.append(_diag_row(row, matched=False, skip_reason="missing_yaml_weight", conflict_policy=conflict_policy))
            continue
        for edge in matches:
            key = (_process_text(edge.get("process_id")), _clean(edge.get("canonical_edge_id")))
            if not key[1]:
                diagnostics.append(_diag_row(row, matched=False, skip_reason="missing_canonical_edge_id", conflict_policy=conflict_policy))
                continue
            if key not in csv_edge_keys and not allow_yaml_only:
                diagnostics.append(_diag_row(row, matched=False, skip_reason="yaml_only_target_stream_not_allowed", conflict_policy=conflict_policy))
                continue
            if key not in resolved:
                resolved[key] = ResolvedTargetStreamEdge(
                    process_id=key[0],
                    process_num=_process_num(key[0]) or 0,
                    canonical_edge_id=key[1],
                    target_stream=row.target_stream or row.fallback_stream_key or row.required_stream_key,
                    edge_weight=float(row.weight),
                    weight_source="per_process_yaml",
                    target_rows=[],
                )
            else:
                if resolved[key].weight_source == "per_process_yaml":
                    resolved[key].edge_weight = _apply_conflict(
                        resolved[key].edge_weight,
                        float(row.weight),
                        conflict_policy,
                    )
                else:
                    resolved[key].edge_weight = float(row.weight)
                resolved[key].weight_source = "per_process_yaml"
            resolved[key].yaml_rows.append(row)
    return resolved, diagnostics


def _diag_row(
    row: TargetStreamRow,
    *,
    matched: bool,
    skip_reason: str = "",
    conflict_policy: str = "max",
) -> dict[str, Any]:
    return {
        "split": "",
        "epoch": "",
        "step": "",
        "process_id": row.process_id,
        "target_stream": row.target_stream or row.required_stream_key or row.fallback_stream_key,
        "canonical_edge_id": row.canonical_edge_id,
        "matched_edge_batch_index": "",
        "target_ids": row.target_id,
        "target_species_list": row.target_species,
        "num_target_rows_for_same_stream": 1,
        "edge_weight": row.weight if row.weight is not None else "",
        "weight_source": row.source,
        "conflict_policy": conflict_policy,
        "num_supervised_features": 0,
        "used_feature_names": "",
        "matched": bool(matched),
        "skip_reason": skip_reason,
    }


def build_target_stream_loss_weight_tensor(
    *,
    export_meta: Any,
    edge_target_columns: Sequence[str],
    train_cfg: Any,
    target_stream_targets_path: Path | str | None = None,
) -> tuple[torch.Tensor | None, list[dict[str, Any]], dict[str, float]]:
    if export_meta is None or not bool(getattr(train_cfg, "use_target_stream_loss_weighting", False)):
        return None, [], {}
    available = _available_edges_from_export_meta(export_meta)
    resolved, diagnostics = resolve_target_stream_edges(
        available_edges=available,
        train_cfg=train_cfg,
        target_stream_targets_path=target_stream_targets_path,
    )
    n_edges = len(available)
    n_cols = len(list(edge_target_columns or STREAM_EDGE_FEATURE_SLOTS))
    w = torch.ones((n_edges, n_cols), dtype=torch.float32)
    edge_to_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for edge in available:
        edge_to_indices[(_process_text(edge.get("process_id")), _clean(edge.get("canonical_edge_id")))].append(int(edge["edge_index"]))
    for key, rec in resolved.items():
        for idx in edge_to_indices.get(key, []):
            w[idx, :] = float(rec.edge_weight)
            diagnostics.append(
                {
                    "split": "",
                    "epoch": "",
                    "step": "",
                    "process_id": rec.process_id,
                    "target_stream": rec.target_stream,
                    "canonical_edge_id": rec.canonical_edge_id,
                    "matched_edge_batch_index": idx,
                    "target_ids": ";".join(sorted({r.target_id for r in rec.target_rows if r.target_id})),
                    "target_species_list": ";".join(sorted({r.target_species for r in rec.target_rows if r.target_species})),
                    "num_target_rows_for_same_stream": len(rec.target_rows),
                    "edge_weight": float(rec.edge_weight),
                    "weight_source": rec.weight_source,
                    "conflict_policy": str(getattr(train_cfg, "target_stream_weight_conflict_policy", "max")),
                    "num_supervised_features": "",
                    "used_feature_names": "",
                    "matched": True,
                    "skip_reason": "",
                }
            )
    summary = {
        "num_target_stream_edges_weighted": float(len(resolved)),
        "num_target_stream_edges_with_yaml_override": float(sum(1 for r in resolved.values() if r.weight_source == "per_process_yaml")),
        "num_target_stream_rows_skipped": float(sum(1 for d in diagnostics if not bool(d.get("matched")))),
        "target_stream_loss_weight": float(getattr(train_cfg, "target_stream_loss_weight", 5.0)),
    }
    return (w if torch.any(w != 1.0) else None), diagnostics, summary


def _metric_bundle(true: pd.Series, pred: pd.Series) -> dict[str, Any]:
    n = int(len(true))
    if n <= 0:
        return {"mae": None, "rmse": None, "r2": None, "n": 0}
    err = pred.astype(float) - true.astype(float)
    mae = float(err.abs().mean())
    rmse = math.sqrt(float((err**2).mean()))
    if n < 2:
        r2 = None
    else:
        sst = float(((true.astype(float) - true.astype(float).mean()) ** 2).sum())
        r2 = None if sst <= 1.0e-12 else float(1.0 - float((err**2).sum()) / sst)
    return {"mae": mae, "rmse": rmse, "r2": r2, "n": n}


def build_target_stream_feature_metrics(
    *,
    edge_predictions: pd.DataFrame,
    train_cfg: Any,
    target_stream_targets_path: Path | str | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    available = _available_edges_from_prediction_df(edge_predictions)
    resolved, skipped = resolve_target_stream_edges(
        available_edges=available,
        train_cfg=train_cfg,
        target_stream_targets_path=target_stream_targets_path,
    )
    rows_csv: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    splits_payload: dict[str, Any] = {}
    sup = edge_predictions[edge_predictions.get("y_edge_mask", 0) > 0.0].copy() if not edge_predictions.empty else pd.DataFrame()
    split_names = sorted(set(edge_predictions.get("split", pd.Series(dtype=str)).astype(str).tolist())) if not edge_predictions.empty else []
    for split in split_names:
        split_df = sup[sup["split"].astype(str) == split] if not sup.empty else pd.DataFrame()
        proc_payload: dict[str, Any] = {}
        split_macro_r2: list[float] = []
        for key, rec in sorted(resolved.items()):
            pid, edge_id = key
            g_edge = split_df[
                (split_df["process_id"].astype(str) == pid)
                & (split_df["canonical_edge_id"].astype(str) == edge_id)
            ] if not split_df.empty else pd.DataFrame()
            features: dict[str, Any] = {}
            stream_macro_r2: list[float] = []
            stream_macro_mae: list[float] = []
            stream_macro_rmse: list[float] = []
            used_features: list[str] = []
            for prop in STREAM_EDGE_FEATURE_SLOTS:
                g = g_edge[g_edge["property_name"].astype(str) == prop] if not g_edge.empty else pd.DataFrame()
                mb = _metric_bundle(g["y_true_orig"], g["y_pred_orig"]) if not g.empty else {"mae": None, "rmse": None, "r2": None, "n": 0}
                features[prop] = mb
                if int(mb["n"]) > 0:
                    used_features.append(prop)
                if mb["r2"] is not None:
                    stream_macro_r2.append(float(mb["r2"]))
                if mb["mae"] is not None:
                    stream_macro_mae.append(float(mb["mae"]))
                if mb["rmse"] is not None:
                    stream_macro_rmse.append(float(mb["rmse"]))
                rows_csv.append(
                    {
                        "split": split,
                        "process_id": rec.process_num,
                        "target_stream": rec.target_stream,
                        "canonical_edge_id": rec.canonical_edge_id,
                        "target_ids": ";".join(sorted({r.target_id for r in rec.target_rows if r.target_id})),
                        "target_species_list": ";".join(sorted({r.target_species for r in rec.target_rows if r.target_species})),
                        "feature_name": prop,
                        "mae": mb["mae"],
                        "rmse": mb["rmse"],
                        "r2": mb["r2"],
                        "n": mb["n"],
                        "matched": True,
                        "edge_weight": rec.edge_weight,
                        "weight_source": rec.weight_source,
                        "skip_reason": "",
                    }
                )
            macro = {
                "mae": (sum(stream_macro_mae) / len(stream_macro_mae)) if stream_macro_mae else None,
                "rmse": (sum(stream_macro_rmse) / len(stream_macro_rmse)) if stream_macro_rmse else None,
                "r2": (sum(stream_macro_r2) / len(stream_macro_r2)) if stream_macro_r2 else None,
            }
            if macro["r2"] is not None:
                split_macro_r2.append(float(macro["r2"]))
            proc = proc_payload.setdefault(str(rec.process_num), {"target_streams": {}})
            stream_key = rec.target_stream or rec.canonical_edge_id
            if stream_key in proc["target_streams"]:
                stream_key = f"{stream_key}::{rec.canonical_edge_id}"
            proc["target_streams"][stream_key] = {
                "canonical_edge_id": rec.canonical_edge_id,
                "matched": True,
                "edge_weight": rec.edge_weight,
                "weight_source": rec.weight_source,
                "target_rows": [
                    {
                        "target_id": r.target_id,
                        "target_species": r.target_species,
                        "target_feature": r.target_feature,
                        "target_formula": r.target_formula,
                    }
                    for r in rec.target_rows
                ],
                "features": features,
                "macro": macro,
            }
            diag_rows.append(
                {
                    "split": split,
                    "epoch": "final",
                    "step": "",
                    "process_id": rec.process_num,
                    "target_stream": rec.target_stream,
                    "canonical_edge_id": rec.canonical_edge_id,
                    "matched_edge_batch_index": "",
                    "target_ids": ";".join(sorted({r.target_id for r in rec.target_rows if r.target_id})),
                    "target_species_list": ";".join(sorted({r.target_species for r in rec.target_rows if r.target_species})),
                    "num_target_rows_for_same_stream": len(rec.target_rows),
                    "edge_weight": rec.edge_weight,
                    "weight_source": rec.weight_source,
                    "conflict_policy": str(getattr(train_cfg, "target_stream_weight_conflict_policy", "max")),
                    "num_supervised_features": len(used_features),
                    "used_feature_names": ";".join(used_features),
                    "matched": True,
                    "skip_reason": "",
                }
            )
        for d in skipped:
            d2 = dict(d)
            d2["split"] = split
            diag_rows.append(d2)
            rows_csv.append(
                {
                    "split": split,
                    "process_id": _process_num(d.get("process_id")) or d.get("process_id", ""),
                    "target_stream": d.get("target_stream", ""),
                    "canonical_edge_id": d.get("canonical_edge_id", ""),
                    "target_ids": d.get("target_ids", ""),
                    "target_species_list": d.get("target_species_list", ""),
                    "feature_name": "",
                    "mae": None,
                    "rmse": None,
                    "r2": None,
                    "n": 0,
                    "matched": False,
                    "edge_weight": d.get("edge_weight", ""),
                    "weight_source": d.get("weight_source", ""),
                    "skip_reason": d.get("skip_reason", ""),
                }
            )
        splits_payload[split] = {
            "processes": proc_payload,
            "macro": {
                "target_stream_feature_macro_r2": (sum(split_macro_r2) / len(split_macro_r2)) if split_macro_r2 else None
            },
        }
    payload = {
        "metadata": {
            "metric_type": "target_stream_feature_prediction",
            "scale": "original",
            "derived_amount_metrics_are_primary": False,
            "target_stream_loss_weight": float(getattr(train_cfg, "target_stream_loss_weight", 5.0)),
            "target_stream_weighting_mode": str(getattr(train_cfg, "target_stream_weighting_mode", "unique_target_stream_edge_by_process")),
            "target_stream_weight_conflict_policy": str(getattr(train_cfg, "target_stream_weight_conflict_policy", "max")),
            "supports_per_process_yaml_weights": True,
        },
        "splits": splits_payload,
    }
    return payload, pd.DataFrame(rows_csv), pd.DataFrame(diag_rows)


def target_stream_feature_scalar_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    splits = payload.get("splits", {}) if isinstance(payload, Mapping) else {}
    if isinstance(splits, Mapping):
        for split, data in splits.items():
            macro = data.get("macro", {}) if isinstance(data, Mapping) else {}
            val = macro.get("target_stream_feature_macro_r2")
            if isinstance(val, (int, float)) and math.isfinite(float(val)):
                out[f"{split}_target_stream_feature_macro_r2"] = float(val)
                if str(split) == "val":
                    out["val_target_stream_feature_macro_r2"] = float(val)
                    out["eval_target_stream_feature_macro_r2"] = float(val)
    return out


def write_target_stream_feature_metric_artifacts(
    *,
    out_dir: Path,
    edge_predictions: pd.DataFrame,
    train_cfg: Any,
    target_stream_targets_path: Path | str | None = None,
) -> dict[str, float]:
    payload, csv_df, diag_df = build_target_stream_feature_metrics(
        edge_predictions=edge_predictions,
        train_cfg=train_cfg,
        target_stream_targets_path=target_stream_targets_path,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "target_stream_feature_metrics.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    csv_df.to_csv(out_dir / "target_stream_feature_metrics.csv", index=False)
    diag_df.to_csv(out_dir / "target_stream_loss_weighting_diagnostics.csv", index=False)
    return target_stream_feature_scalar_metrics(payload)


class TargetStreamFeatureAccumulator:
    """Small validation-time accumulator for target stream feature macro metrics."""

    def __init__(self, *, train_cfg: Any, target_stream_targets_path: Path | str | None = None) -> None:
        self.train_cfg = train_cfg
        self.target_stream_targets_path = target_stream_targets_path
        self.rows: list[dict[str, Any]] = []

    def update_batch(
        self,
        *,
        export: Any,
        y_pred_orig: torch.Tensor,
        y_true_orig: torch.Tensor,
        y_edge_mask: torch.Tensor,
        edge_target_columns: Sequence[str],
        split_name: str,
    ) -> None:
        cols = list(edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
        pred = y_pred_orig.detach().cpu()
        true = y_true_orig.detach().cpu()
        mask = y_edge_mask.detach().cpu()
        if mask.ndim == 1:
            mask = mask.view(-1, 1).expand_as(pred)
        for e in range(int(pred.shape[0])):
            for j, prop in enumerate(cols):
                self.rows.append(
                    {
                        "split": split_name,
                        "process_id": _process_text(export.process_id[e]),
                        "canonical_edge_id": _clean(export.canonical_edge_id[e]),
                        "main_data_stream_key": _clean(export.main_data_stream_key[e]),
                        "property_name": str(prop),
                        "y_true_orig": float(true[e, j]),
                        "y_pred_orig": float(pred[e, j]),
                        "y_edge_mask": float(mask[e, j]),
                    }
                )

    def finalize_scalars(self) -> dict[str, float]:
        if not self.rows:
            return {}
        payload, _, _ = build_target_stream_feature_metrics(
            edge_predictions=pd.DataFrame(self.rows),
            train_cfg=self.train_cfg,
            target_stream_targets_path=self.target_stream_targets_path,
        )
        scalars = target_stream_feature_scalar_metrics(payload)
        out: dict[str, float] = {}
        for key, val in scalars.items():
            if key.startswith("val_"):
                out[key] = val
                out[key.replace("val_", "eval_", 1)] = val
            elif key.startswith("eval_"):
                out[key] = val
        if "val_target_stream_feature_macro_r2" in out:
            out["target_stream_feature_macro_r2"] = out["val_target_stream_feature_macro_r2"]
        return out
