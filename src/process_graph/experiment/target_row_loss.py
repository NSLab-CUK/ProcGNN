"""Target-row and incident-edge auxiliary losses for edge_all training."""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch

from ..constants import STREAM_EDGE_FEATURE_SLOTS
from .target_row_amounts import compute_target_row_amounts
from .target_row_spec import TargetRowSpec, load_target_row_specs
from .target_v4_metrics import normalize_stream_key

_TARGET_ROW_SPECS_CACHE: list[TargetRowSpec] | None = None


def _process_id_text(value: Any) -> str:
    text = str(value).strip()
    if text.isdigit():
        return f"Process{int(text)}"
    return text


def _specs(train_cfg: Any | None = None) -> list[TargetRowSpec]:
    global _TARGET_ROW_SPECS_CACHE
    if _TARGET_ROW_SPECS_CACHE is not None:
        return list(_TARGET_ROW_SPECS_CACHE)
    _TARGET_ROW_SPECS_CACHE = load_target_row_specs(train_cfg=train_cfg)
    return list(_TARGET_ROW_SPECS_CACHE)


def _mask_like(y_mask: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    m = y_mask.to(device=y_pred.device, dtype=y_pred.dtype)
    while m.ndim < y_pred.ndim:
        m = m.unsqueeze(-1)
    return m.expand_as(y_pred)


def _loss_elements(y_pred: torch.Tensor, y_true: torch.Tensor, loss_type: str) -> torch.Tensor:
    if loss_type == "mse":
        return (y_pred - y_true) ** 2
    if loss_type == "l1":
        return (y_pred - y_true).abs()
    if loss_type == "smooth_l1":
        return torch.nn.functional.smooth_l1_loss(y_pred, y_true, reduction="none")
    raise ValueError(f"Unsupported loss_type for target-row loss: {loss_type!r}")


def _masked_mean_from_weight(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    weight: torch.Tensor,
    *,
    loss_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    w = weight.to(device=y_pred.device, dtype=y_pred.dtype)
    while w.ndim < y_pred.ndim:
        w = w.unsqueeze(-1)
    w = w.expand_as(y_pred)
    denom = w.sum()
    if float(denom.detach().cpu()) <= 0.0:
        z = y_pred.sum() * 0.0
        return z, denom
    return (_loss_elements(y_pred, y_true, loss_type) * w).sum() / denom, denom


def _target_weight_for_spec(spec: TargetRowSpec, train_cfg: Any | None) -> float:
    tw_map = getattr(train_cfg, "target_weight_by_id", None) if train_cfg is not None else None
    if isinstance(tw_map, Mapping):
        return float(tw_map.get(spec.target_id, spec.target_weight))
    return float(getattr(spec, "target_weight", 1.0))


def _answer_weight_for_species(spec: TargetRowSpec, train_cfg: Any | None) -> float:
    if train_cfg is None:
        return float(getattr(spec, "species_weight", 1.0))
    default = float(getattr(train_cfg, "answer_edge_weight", 1.0))
    species = str(spec.species).strip().lower()
    if species == "h2":
        return float(getattr(train_cfg, "answer_edge_weight_h2", default))
    if species == "co2":
        return float(getattr(train_cfg, "answer_edge_weight_co2", default))
    if species == "h2o":
        return float(getattr(train_cfg, "answer_edge_weight_h2o", default))
    return default


def _internal_weight_for_spec(spec: TargetRowSpec, train_cfg: Any | None, *, use_answer_weight: bool) -> float:
    weight = _target_weight_for_spec(spec, train_cfg)
    if use_answer_weight:
        weight *= _answer_weight_for_species(spec, train_cfg)
    return float(weight)


def _edge_matches_spec(*, edge_i: int, export_meta: Any, spec: TargetRowSpec) -> bool:
    if _process_id_text(export_meta.process_id[edge_i]) != str(spec.process_id):
        return False
    edge_ids = getattr(export_meta, "canonical_edge_id", None)
    if edge_ids is not None and str(getattr(spec, "edge_id", "")).strip():
        if str(edge_ids[edge_i]).strip() != str(spec.edge_id).strip():
            return False
    stream = normalize_stream_key(getattr(export_meta, "main_data_stream_key", [""])[edge_i])
    req = normalize_stream_key(getattr(spec, "required_stream_key", spec.target_stream))
    return bool(not req or stream == req)


def _matching_edge_indices(export_meta: Any, spec: TargetRowSpec, n_edges: int) -> list[int]:
    return [i for i in range(n_edges) if _edge_matches_spec(edge_i=i, export_meta=export_meta, spec=spec)]


def _column_index(edge_target_columns: Sequence[str], name: str) -> int | None:
    cols = [str(c) for c in edge_target_columns]
    try:
        return cols.index(str(name))
    except ValueError:
        return None


def resolve_all_edge_r2_config(train_cfg: Any) -> dict[str, Any]:
    defaults = {
        "loss_weight": 0.0,
        "min_count": 8,
        "sst_threshold": 1.0e-4,
        "loss_cap": 0.0,
    }
    new_keys = {
        "loss_weight": "all_edge_r2_loss_weight",
        "min_count": "all_edge_r2_min_count",
        "sst_threshold": "all_edge_r2_sst_threshold",
        "loss_cap": "all_edge_r2_loss_cap",
    }
    legacy_keys = {
        "loss_weight": "primary_frac_r2_loss_weight",
        "min_count": "primary_frac_r2_min_count",
        "sst_threshold": "primary_frac_r2_sst_threshold",
        "loss_cap": "primary_frac_r2_loss_cap",
    }
    has_new = any(getattr(train_cfg, key, None) is not None for key in new_keys.values())
    if has_new:
        out = {name: getattr(train_cfg, key, defaults[name]) for name, key in new_keys.items()}
        source = "new_all_edge_r2_keys"
    else:
        has_legacy = any(hasattr(train_cfg, key) for key in legacy_keys.values())
        if has_legacy:
            out = {name: getattr(train_cfg, key, defaults[name]) for name, key in legacy_keys.items()}
            source = "legacy_primary_frac_r2_keys"
        else:
            out = dict(defaults)
            source = "default"
    return {
        "loss_weight": float(out["loss_weight"]),
        "min_count": int(out["min_count"]),
        "sst_threshold": float(out["sst_threshold"]),
        "loss_cap": float(out["loss_cap"]),
        "source": source,
    }


def compute_all_edge_r2_loss(
    *,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor,
    train_cfg: Any,
    element_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    cfg = resolve_all_edge_r2_config(train_cfg)
    mask = _mask_like(y_mask, y_pred)
    if element_weight is not None:
        ew = _mask_like(element_weight, y_pred)
        mask = mask * ew
    terms: list[torch.Tensor] = []
    valid = 0
    for j in range(int(y_pred.shape[1])):
        mj = mask[:, j]
        sel = mj > 0
        count = int(sel.sum().detach().cpu())
        if count < int(cfg["min_count"]):
            continue
        yt = y_true[:, j][sel]
        yp = y_pred[:, j][sel]
        sst = ((yt - yt.mean()) ** 2).sum()
        if float(sst.detach().cpu()) <= float(cfg["sst_threshold"]):
            continue
        sse = ((yp - yt) ** 2).sum()
        term = sse / sst.clamp_min(float(cfg["sst_threshold"]))
        cap = float(cfg["loss_cap"])
        if cap > 0.0:
            term = term.clamp_max(cap)
        terms.append(term)
        valid += 1
    if not terms:
        loss = y_pred.sum() * 0.0
    else:
        loss = torch.stack(terms).mean()
    n = float((mask > 0).sum().detach().cpu())
    return loss, {
        "r2_loss_scope": "edge_all",
        "r2_loss_num_elements": n,
        "edge_all_loss_num_elements": n,
        "r2_loss_valid_feature_count": valid,
        "primary_frac_r2_valid_feature_count": valid,
        "all_edge_r2_min_count": int(cfg["min_count"]),
        "all_edge_r2_sst_threshold": float(cfg["sst_threshold"]),
        "all_edge_r2_loss_cap": float(cfg["loss_cap"]),
        "all_edge_r2_config_source": str(cfg["source"]),
    }


def build_primary_frac_mask(
    *,
    y_pred: torch.Tensor,
    y_mask: torch.Tensor,
    export_meta: Any,
    edge_target_columns: Sequence[str],
    train_cfg: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    mask = torch.zeros_like(y_pred, dtype=y_pred.dtype, device=y_pred.device)
    if export_meta is None:
        return mask, {"primary_frac_count": 0.0, "target_loss_num_elements": 0.0}
    n_edges = int(y_pred.shape[0])
    specs = _specs(train_cfg)
    for spec in specs:
        if spec.skip_v4_loss():
            continue
        col = _column_index(edge_target_columns, spec.frac_slot)
        if col is None:
            continue
        for e in _matching_edge_indices(export_meta, spec, n_edges):
            m = y_mask[e, col] if y_mask.ndim == 2 else y_mask[e]
            if float(m.detach().cpu()) > 0.0:
                mask[e, col] = 1.0
    count = float(mask.sum().detach().cpu())
    return mask, {"primary_frac_count": count, "target_loss_num_elements": count}


def _masked_r2_loss_on_mask(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
    train_cfg: Any,
) -> tuple[torch.Tensor, int]:
    min_count = int(getattr(train_cfg, "primary_frac_r2_min_count", 8))
    threshold = float(getattr(train_cfg, "primary_frac_r2_sst_threshold", 1.0e-4))
    terms: list[torch.Tensor] = []
    for j in range(int(y_pred.shape[1])):
        mj = mask[:, j] > 0
        if int(mj.sum().detach().cpu()) < min_count:
            continue
        yt = y_true[:, j][mj]
        yp = y_pred[:, j][mj]
        sst = ((yt - yt.mean()) ** 2).sum()
        if float(sst.detach().cpu()) <= threshold:
            continue
        terms.append(((yp - yt) ** 2).sum() / sst.clamp_min(threshold))
    if not terms:
        return y_pred.sum() * 0.0, 0
    return torch.stack(terms).mean(), len(terms)


def compute_primary_frac_losses(
    *,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor,
    export_meta: Any,
    edge_target_columns: Sequence[str],
    train_cfg: Any,
    loss_type: str | None = None,
    y_edge_mean: torch.Tensor | None = None,
    y_edge_std: torch.Tensor | None = None,
    compute_r2_loss: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    mask, info = build_primary_frac_mask(
        y_pred=y_pred,
        y_mask=y_mask,
        export_meta=export_meta,
        edge_target_columns=edge_target_columns,
        train_cfg=train_cfg,
    )
    lt = loss_type or str(getattr(train_cfg, "loss_type_edge_all", "smooth_l1"))
    loss_frac, _ = _masked_mean_from_weight(y_pred, y_true, mask, loss_type=lt)
    if compute_r2_loss:
        loss_r2, valid = _masked_r2_loss_on_mask(y_pred, y_true, mask, train_cfg)
    else:
        loss_r2, valid = y_pred.sum() * 0.0, 0
    info["primary_frac_r2_valid_feature_count"] = valid
    info["r2_loss_valid_feature_count"] = valid
    pred_vals = y_pred[mask > 0]
    true_vals = y_true[mask > 0]
    if pred_vals.numel() > 1:
        ps = float(pred_vals.detach().std(unbiased=False).cpu())
        ts = float(true_vals.detach().std(unbiased=False).cpu())
    else:
        ps = ts = 0.0
    info.update(
        {
            "primary_frac_pred_std_norm": ps,
            "primary_frac_true_std_norm": ts,
            "primary_frac_std_ratio_norm": ps / ts if ts > 0 else 0.0,
            "primary_frac_pred_std_orig": ps,
            "primary_frac_true_std_orig": ts,
            "primary_frac_std_ratio_orig": ps / ts if ts > 0 else 0.0,
        }
    )
    return loss_frac, loss_r2, info


def compute_target_frac_feature_loss(
    *,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor,
    export_meta: Any,
    edge_target_columns: Sequence[str],
    train_cfg: Any,
    loss_type: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if export_meta is None:
        return y_pred.sum() * 0.0, {"n_terms": 0}
    loss, _, info = compute_primary_frac_losses(
        y_pred=y_pred,
        y_true=y_true,
        y_mask=y_mask,
        export_meta=export_meta,
        edge_target_columns=edge_target_columns,
        train_cfg=train_cfg,
        loss_type=loss_type,
    )
    info["n_terms"] = int(info.get("primary_frac_count", 0))
    return loss, info


def _incident_edges_for_target(
    *,
    target_edge: int,
    edge_index: torch.Tensor,
    export_meta: Any,
    train_cfg: Any,
) -> dict[str, Any]:
    src_idx = int(edge_index[0, target_edge].detach().cpu())
    dst_idx = int(edge_index[1, target_edge].detach().cpu())
    src_node = ""
    dst_node = ""
    if hasattr(export_meta, "src_node") and len(getattr(export_meta, "src_node", [])) > target_edge:
        src_node = str(export_meta.src_node[target_edge])
    if hasattr(export_meta, "dst_node") and len(getattr(export_meta, "dst_node", [])) > target_edge:
        dst_node = str(export_meta.dst_node[target_edge])
    virtuals = {str(v).strip() for v in getattr(train_cfg, "target_incident_virtual_node_names", ["V_INPUT", "V_OUTPUT"])}
    src_is_virtual = bool(src_node and src_node in virtuals)
    dst_is_virtual = bool(dst_node and dst_node in virtuals)
    exclude_virtual = bool(getattr(train_cfg, "target_incident_exclude_virtual_nodes", True))
    include_target = bool(getattr(train_cfg, "target_incident_include_target_edge", True))

    n_edges = int(edge_index.shape[1])
    active_nodes: list[int] = []
    if not (exclude_virtual and src_is_virtual):
        active_nodes.append(src_idx)
    if not (exclude_virtual and dst_is_virtual):
        active_nodes.append(dst_idx)

    before = ((edge_index[0] == src_idx) | (edge_index[1] == src_idx) | (edge_index[0] == dst_idx) | (edge_index[1] == dst_idx))
    after = torch.zeros(n_edges, dtype=torch.bool, device=edge_index.device)
    for node in active_nodes:
        after |= (edge_index[0] == node) | (edge_index[1] == node)
    if include_target:
        after[target_edge] = True
    before_rows = torch.nonzero(before, as_tuple=False).flatten().detach().cpu().tolist()
    after_rows = torch.nonzero(after, as_tuple=False).flatten().detach().cpu().tolist()
    return {
        "src_node": src_node,
        "dst_node": dst_node,
        "src_is_virtual": src_is_virtual,
        "dst_is_virtual": dst_is_virtual,
        "before_rows": [int(x) for x in before_rows],
        "after_rows": [int(x) for x in after_rows],
        "num_before": len(before_rows),
        "num_after": len(after_rows),
        "num_removed": max(0, len(before_rows) - len(after_rows)),
    }


def _target_incident_terms(
    *,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor,
    export_meta: Any,
    edge_index: torch.Tensor | None,
    train_cfg: Any,
    loss_type: str,
    with_loss: bool,
) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    if export_meta is None or edge_index is None:
        return [], []
    n_edges = int(y_pred.shape[0])
    if int(edge_index.shape[1]) != n_edges:
        return [], []
    use_answer = bool(getattr(train_cfg, "target_incident_use_answer_edge_weight", False))
    rows: list[dict[str, Any]] = []
    terms: list[torch.Tensor] = []
    for spec in _specs(train_cfg):
        if spec.skip_v4_loss():
            continue
        matches = _matching_edge_indices(export_meta, spec, n_edges)
        if not matches:
            rows.append({"spec": spec, "matched": False, "target_edge": -1})
            continue
        for target_edge in matches:
            inc = _incident_edges_for_target(
                target_edge=target_edge,
                edge_index=edge_index,
                export_meta=export_meta,
                train_cfg=train_cfg,
            )
            edge_rows = inc["after_rows"]
            edge_tensor = torch.as_tensor(edge_rows, device=y_pred.device, dtype=torch.long)
            sub_mask = _mask_like(y_mask[edge_tensor], y_pred[edge_tensor])
            num_cells = float((sub_mask > 0).sum().detach().cpu())
            internal_weight = _internal_weight_for_spec(spec, train_cfg, use_answer_weight=use_answer)
            if with_loss and num_cells > 0:
                base_loss, _ = _masked_mean_from_weight(
                    y_pred[edge_tensor],
                    y_true[edge_tensor],
                    sub_mask,
                    loss_type=loss_type,
                )
                terms.append(base_loss * float(internal_weight))
            rows.append(
                {
                    "spec": spec,
                    "matched": True,
                    "target_edge": int(target_edge),
                    "edge_rows": edge_rows,
                    "internal_weight": float(internal_weight),
                    "num_supervised_cells": int(num_cells),
                    **inc,
                }
            )
    return rows, terms


def _balanced_reduce(rows: list[dict[str, Any]], terms: list[torch.Tensor], train_cfg: Any, ref: torch.Tensor) -> torch.Tensor:
    if not terms:
        return ref.sum() * 0.0
    matched_rows = [r for r in rows if r.get("matched") and int(r.get("num_supervised_cells", 0)) > 0]
    if len(matched_rows) != len(terms):
        return torch.stack(terms).mean()
    balancing = str(getattr(train_cfg, "target_feature_loss_balancing", "process_target_balanced")).lower()
    if balancing != "process_target_balanced":
        return torch.stack(terms).mean()
    grouped: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
    for row, term in zip(matched_rows, terms):
        spec = row["spec"]
        grouped[(str(spec.process_id), str(spec.target_id))].append(term)
    by_process: dict[str, list[torch.Tensor]] = defaultdict(list)
    for (pid, _tid), vals in grouped.items():
        by_process[pid].append(torch.stack(vals).mean())
    process_terms = [torch.stack(vals).mean() for vals in by_process.values()]
    return torch.stack(process_terms).mean()


def compute_target_incident_edge_loss(
    *,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor,
    export_meta: Any,
    edge_index: torch.Tensor | None,
    train_cfg: Any,
    loss_type: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    lt = loss_type or str(getattr(train_cfg, "loss_type_edge_all", "smooth_l1"))
    rows, terms = _target_incident_terms(
        y_pred=y_pred,
        y_true=y_true,
        y_mask=y_mask,
        export_meta=export_meta,
        edge_index=edge_index,
        train_cfg=train_cfg,
        loss_type=lt,
        with_loss=True,
    )
    loss = _balanced_reduce(rows, terms, train_cfg, y_pred)
    matched = [r for r in rows if r.get("matched")]
    included_edges = sorted({int(e) for r in matched for e in r.get("edge_rows", [])})
    n_edges = int(y_pred.shape[0])
    inner_weights = [float(r.get("internal_weight", 0.0)) for r in matched]
    primary_w = float(getattr(train_cfg, "primary_frac_loss_weight", 1.0))
    return loss, {
        "primary_frac_count": float(len(included_edges)),
        "target_loss_num_elements": float(sum(int(r.get("num_supervised_cells", 0)) for r in matched)),
        "target_incident_edge_count": float(len(included_edges)),
        "target_incident_supervised_cell_count": float(sum(int(r.get("num_supervised_cells", 0)) for r in matched)),
        "target_incident_target_row_count": float(len(rows)),
        "target_incident_matched_edge_count": float(len(matched)),
        "target_incident_virtual_endpoint_excluded_count": float(sum(int(r.get("num_removed", 0)) for r in matched)),
        "target_incident_target_edge_included_count": float(sum(1 for r in matched if int(r.get("target_edge", -1)) in r.get("edge_rows", []))),
        "target_incident_virtual_filter_applied": float(1 if bool(getattr(train_cfg, "target_incident_exclude_virtual_nodes", True)) else 0),
        "target_incident_apply_answer_weight": float(1 if bool(getattr(train_cfg, "target_incident_use_answer_edge_weight", False)) else 0),
        "target_incident_inner_weight_mean": float(sum(inner_weights) / len(inner_weights)) if inner_weights else 0.0,
        "target_incident_inner_weight_max": float(max(inner_weights)) if inner_weights else 0.0,
        "target_incident_effective_weight_mean": float(primary_w * (sum(inner_weights) / len(inner_weights))) if inner_weights else 0.0,
        "target_incident_coverage_edge_frac": float(len(included_edges) / n_edges) if n_edges > 0 else 0.0,
    }


def compute_v4_target_row_loss(
    *,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor,
    export_meta: Any,
    edge_target_columns: Sequence[str],
    train_cfg: Any,
    loss_type: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    specs = _specs(train_cfg)
    cols = list(edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
    mask = torch.zeros_like(y_pred)
    for spec in specs:
        col = _column_index(cols, spec.frac_slot)
        if col is None:
            continue
        weight = _answer_weight_for_species(spec, train_cfg)
        for e in _matching_edge_indices(export_meta, spec, int(y_pred.shape[0])):
            m = y_mask[e, col] if y_mask.ndim == 2 else y_mask[e]
            if float(m.detach().cpu()) > 0.0:
                mask[e, col] = float(weight)
    loss, denom = _masked_mean_from_weight(
        y_pred,
        y_true,
        mask,
        loss_type=loss_type or str(getattr(train_cfg, "loss_type_edge_all", "smooth_l1")),
    )
    ids = [s.target_id for s in specs]
    return loss, {"target_ids_in_batch": ids, "n_terms": float(denom.detach().cpu())}


def compute_v4_target_row_amount_loss(
    *,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor,
    export_meta: Any,
    edge_target_columns: Sequence[str],
    train_cfg: Any,
    loss_type: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    specs = _specs(train_cfg)
    batch = compute_target_row_amounts(
        y_pred=y_pred,
        y_true=y_true,
        y_mask=y_mask,
        export_meta=export_meta,
        edge_target_columns=edge_target_columns,
        specs=specs,
    )
    vals = []
    for pred, true, m in zip(batch.pred_amount, batch.true_amount, batch.mask):
        if float(m) <= 0.0:
            continue
        vals.append((float(pred), float(true)))
    if not vals:
        return y_pred.sum() * 0.0, {"n_terms": 0}
    yp = torch.as_tensor([p for p, _ in vals], device=y_pred.device, dtype=y_pred.dtype)
    yt = torch.as_tensor([t for _, t in vals], device=y_pred.device, dtype=y_pred.dtype)
    loss = _loss_elements(yp, yt, loss_type or str(getattr(train_cfg, "loss_type_edge_all", "smooth_l1"))).mean()
    return loss, {"n_terms": len(vals), "target_ids_in_batch": list(batch.target_ids)}


def _loss_scalar(loss_scalars: Mapping[str, Any], name: str) -> float:
    value = loss_scalars.get(name, float("nan"))
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _ratio(num: float, den: float) -> float:
    if not math.isfinite(den) or abs(den) <= 1.0e-12:
        return float("nan")
    return num / den


def build_target_incident_diagnostic_rows(
    *,
    split: str,
    epoch: int,
    step: int,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor,
    export_meta: Any,
    edge_index: torch.Tensor | None,
    train_cfg: Any,
    loss_scalars: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows, _terms = _target_incident_terms(
        y_pred=y_pred,
        y_true=y_true,
        y_mask=y_mask,
        export_meta=export_meta,
        edge_index=edge_index,
        train_cfg=train_cfg,
        loss_type=str(getattr(train_cfg, "loss_type_edge_all", "smooth_l1")),
        with_loss=False,
    )
    r2_cfg = resolve_all_edge_r2_config(train_cfg)
    primary_w = float(getattr(train_cfg, "primary_frac_loss_weight", 1.0))
    matched_total = sum(1 for r in rows if r.get("matched"))
    out: list[dict[str, Any]] = []
    for row in rows:
        spec = row["spec"]
        matched = bool(row.get("matched"))
        edge_rows = [int(e) for e in row.get("edge_rows", [])]
        edge_ids = []
        for e in edge_rows:
            edge_ids.append(str(getattr(export_meta, "canonical_edge_id", [""] * int(y_pred.shape[0]))[e]))
        internal_weight = float(row.get("internal_weight", 0.0)) if matched else 0.0
        target_edge = int(row.get("target_edge", -1))
        target_edge_id = ""
        if matched and target_edge >= 0:
            target_edge_id = str(getattr(export_meta, "canonical_edge_id", [""])[target_edge])
        out.append(
            {
                "split": split,
                "epoch": int(epoch),
                "step": int(step),
                "process_id": str(spec.process_id),
                "target_id": str(spec.target_id),
                "target_species": str(spec.species),
                "canonical_answer_edge_id": str(spec.edge_id),
                "target_edge_batch_index": target_edge if matched else "",
                "target_edge_canonical_id": target_edge_id,
                "src_node": row.get("src_node", ""),
                "dst_node": row.get("dst_node", ""),
                "src_is_virtual": bool(row.get("src_is_virtual", False)),
                "dst_is_virtual": bool(row.get("dst_is_virtual", False)),
                "target_incident_exclude_virtual_nodes": bool(getattr(train_cfg, "target_incident_exclude_virtual_nodes", True)),
                "target_incident_include_target_edge": bool(getattr(train_cfg, "target_incident_include_target_edge", True)),
                "num_incident_edges_before_virtual_filter": int(row.get("num_before", 0)),
                "num_incident_edges_after_virtual_filter": int(row.get("num_after", 0)),
                "num_edges_removed_by_virtual_filter": int(row.get("num_removed", 0)),
                "num_supervised_cells": int(row.get("num_supervised_cells", 0)),
                "internal_weight": internal_weight,
                "primary_frac_loss_weight": primary_w,
                "effective_weight": internal_weight * primary_w,
                "included_edge_batch_indices": str(edge_rows),
                "included_edge_canonical_ids": str(edge_ids),
                "loss_edge_all": _loss_scalar(loss_scalars, "loss_edge_all"),
                "loss_target_incident_edges": _loss_scalar(loss_scalars, "loss_target_incident_edges"),
                "weighted_edge_all": _loss_scalar(loss_scalars, "weighted_edge_all"),
                "weighted_target_incident_edges": _loss_scalar(loss_scalars, "weighted_target_incident_edges"),
                "loss_all_edge_r2": _loss_scalar(loss_scalars, "loss_all_edge_r2"),
                "weighted_all_edge_r2": _loss_scalar(loss_scalars, "weighted_all_edge_r2"),
                "loss_total_final": _loss_scalar(loss_scalars, "loss_total_final"),
                "ratio_weighted_target_incident_to_edge_all": _ratio(
                    _loss_scalar(loss_scalars, "weighted_target_incident_edges"),
                    _loss_scalar(loss_scalars, "weighted_edge_all"),
                ),
                "ratio_weighted_all_edge_r2_to_edge_all": _ratio(
                    _loss_scalar(loss_scalars, "weighted_all_edge_r2"),
                    _loss_scalar(loss_scalars, "weighted_edge_all"),
                ),
                "matched_target_edges": int(matched_total),
                "amount_auxiliary_enabled": bool(_loss_scalar(loss_scalars, "loss_amount_auxiliary") != 0.0),
                "amount_auxiliary_contributes_to_total": False,
                "weighted_amount_auxiliary_disabled": _loss_scalar(loss_scalars, "weighted_amount_auxiliary_disabled"),
                "all_edge_r2_loss_weight": float(r2_cfg["loss_weight"]),
                "all_edge_r2_min_count": int(r2_cfg["min_count"]),
                "all_edge_r2_sst_threshold": float(r2_cfg["sst_threshold"]),
                "all_edge_r2_loss_cap": float(r2_cfg["loss_cap"]),
                "all_edge_r2_config_source": str(r2_cfg["source"]),
            }
        )
    return out


def warn_if_duplicate_legacy_weighting(train_cfg: Any) -> None:
    if bool(getattr(train_cfg, "use_legacy_answer_weighting", False)) and bool(
        getattr(train_cfg, "target_incident_use_answer_edge_weight", False)
    ):
        warnings.warn(
            "Both legacy answer element weighting and target_incident_use_answer_edge_weight are enabled; "
            "target incident auxiliary may receive duplicated answer weights.",
            RuntimeWarning,
            stacklevel=2,
        )
