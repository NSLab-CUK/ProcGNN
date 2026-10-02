"""Per-target_id Frac species values from edge tensors (primary supervision)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from .target_row_spec import TargetRowSpec
from .target_v4_metrics import normalize_stream_key


def _process_num(text: object) -> int | None:
    digits = "".join(ch for ch in str(text) if ch.isdigit())
    if not digits:
        return None
    return int(digits)


@dataclass
class TargetRowFracBatch:
    target_ids: list[str] = field(default_factory=list)
    pred_frac: list[float] = field(default_factory=list)
    true_frac: list[float] = field(default_factory=list)
    mask: list[float] = field(default_factory=list)
    edge_indices: list[int] = field(default_factory=list)
    process_ids: list[str] = field(default_factory=list)
    frac_slots: list[str] = field(default_factory=list)


def _edge_matches_spec(*, edge_i: int, export_meta, spec: TargetRowSpec) -> bool:
    pid = str(export_meta.process_id[edge_i])
    if pid != spec.process_id and _process_num(pid) != int(spec.process_num):
        return False
    stream = normalize_stream_key(export_meta.main_data_stream_key[edge_i])
    if stream != spec.required_stream_key:
        return False
    edge_ids = getattr(export_meta, "canonical_edge_id", None)
    if edge_ids is not None and spec.edge_id:
        if str(edge_ids[edge_i]).strip() != spec.edge_id:
            return False
    return True


def compute_target_row_fracs(
    *,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor,
    export_meta,
    edge_target_columns: Sequence[str],
    specs: Sequence[TargetRowSpec],
) -> TargetRowFracBatch:
    """pred_frac[target_id] = Frac_species on spec.edge_id (same scale as y_pred/y_true)."""
    cols = [str(c) for c in edge_target_columns]
    col_idx = {c: i for i, c in enumerate(cols)}
    n_edges = int(y_pred.shape[0])
    if export_meta is None or len(export_meta.process_id) != n_edges:
        return TargetRowFracBatch()

    batch_pids = {str(export_meta.process_id[i]) for i in range(n_edges)}
    active_specs = [s for s in specs if s.process_id in batch_pids and not s.skip_v4_loss()]

    out = TargetRowFracBatch()
    for spec in active_specs:
        fj = col_idx.get(spec.frac_slot)
        if fj is None:
            continue
        for e in range(n_edges):
            if not _edge_matches_spec(edge_i=e, export_meta=export_meta, spec=spec):
                continue
            if y_mask.ndim > 1 and y_mask.shape[1] > fj:
                mval = float(y_mask[e, fj].item())
            elif y_mask.ndim == 1:
                mval = float(y_mask[e].item())
            else:
                mval = 0.0
            if mval <= 0.0:
                continue
            pf = float(y_pred[e, fj].detach().cpu())
            tf = float(y_true[e, fj].detach().cpu())
            if any(math.isnan(x) for x in (pf, tf)):
                continue
            out.target_ids.append(spec.target_id)
            out.pred_frac.append(pf)
            out.true_frac.append(tf)
            out.mask.append(1.0)
            out.edge_indices.append(e)
            out.process_ids.append(spec.process_id)
            out.frac_slots.append(spec.frac_slot)
    return out
