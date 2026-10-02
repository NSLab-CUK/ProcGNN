"""Per-target_id amount computation from edge tensors (never pooled by species)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from .target_row_spec import TargetRowSpec
from .target_v4_metrics import normalize_stream_key


@dataclass
class TargetRowAmountBatch:
    """Amounts for one forward pass (flat edge batch)."""

    target_ids: list[str] = field(default_factory=list)
    pred_amount: list[float] = field(default_factory=list)
    true_amount: list[float] = field(default_factory=list)
    mask: list[float] = field(default_factory=list)
    edge_indices: list[int] = field(default_factory=list)
    process_ids: list[str] = field(default_factory=list)

    def as_dicts(self) -> list[dict[str, Any]]:
        rows = []
        for i, tid in enumerate(self.target_ids):
            rows.append(
                {
                    "target_id": tid,
                    "pred_amount": self.pred_amount[i],
                    "true_amount": self.true_amount[i],
                    "mask": self.mask[i],
                    "edge_index": self.edge_indices[i],
                    "process_id": self.process_ids[i],
                }
            )
        return rows


def _edge_matches_spec(
    *,
    edge_i: int,
    export_meta,
    spec: TargetRowSpec,
) -> bool:
    pid = str(export_meta.process_id[edge_i])
    if pid != spec.process_id:
        return False
    stream = normalize_stream_key(export_meta.main_data_stream_key[edge_i])
    if stream != spec.required_stream_key:
        return False
    edge_ids = getattr(export_meta, "canonical_edge_id", None)
    if edge_ids is not None and spec.edge_id:
        if str(edge_ids[edge_i]).strip() != spec.edge_id:
            return False
    return True


def compute_target_row_amounts(
    *,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor,
    export_meta,
    edge_target_columns: Sequence[str],
    specs: Sequence[TargetRowSpec],
    y_pred_is_orig: bool = True,
) -> TargetRowAmountBatch:
    """Compute pred/true amount per target_id; each target_id is a separate key.

    pred_amount[target_id] = scale * Mole_Flow * Frac_species on spec.edge_id
    """
    cols = [str(c) for c in edge_target_columns]
    col_idx = {c: i for i, c in enumerate(cols)}
    mi = col_idx.get(specs[0].mole_flow_slot if specs else "Mole_Flow")
    if mi is None:
        mi = col_idx.get("Mole_Flow")
    if mi is None:
        return TargetRowAmountBatch()

    n_edges = int(y_pred.shape[0])
    if export_meta is None or len(export_meta.process_id) != n_edges:
        return TargetRowAmountBatch()

    batch_pids = {str(export_meta.process_id[i]) for i in range(n_edges)}
    active_specs = [s for s in specs if s.process_id in batch_pids and not s.skip_v4_loss()]

    out = TargetRowAmountBatch()
    device = y_pred.device
    dtype = y_pred.dtype

    for spec in active_specs:
        fj = col_idx.get(spec.frac_slot)
        if fj is None:
            continue
        for e in range(n_edges):
            if not _edge_matches_spec(edge_i=e, export_meta=export_meta, spec=spec):
                continue
            mval = float(y_mask[e].item()) if y_mask.ndim == 1 else float(y_mask[e].sum().item())
            if mval <= 0.0:
                continue
            pm = float(y_pred[e, mi].detach().cpu())
            tm = float(y_true[e, mi].detach().cpu())
            pf = float(y_pred[e, fj].detach().cpu())
            tf = float(y_true[e, fj].detach().cpu())
            if any(math.isnan(x) for x in (pm, tm, pf, tf)):
                continue
            pred_amt = spec.scale * pm * pf
            true_amt = spec.scale * tm * tf
            out.target_ids.append(spec.target_id)
            out.pred_amount.append(pred_amt)
            out.true_amount.append(true_amt)
            out.mask.append(1.0)
            out.edge_indices.append(e)
            out.process_ids.append(spec.process_id)

    return out


def amounts_by_target_id(batch: TargetRowAmountBatch) -> dict[str, list[tuple[float, float, float]]]:
    """Group batch rows by target_id (multiple graph instances => multiple values per id)."""
    grouped: dict[str, list[tuple[float, float, float]]] = {}
    for tid, pred, true, mask in zip(
        batch.target_ids, batch.pred_amount, batch.true_amount, batch.mask
    ):
        if mask <= 0:
            continue
        grouped.setdefault(tid, []).append((pred, true, mask))
    return grouped
