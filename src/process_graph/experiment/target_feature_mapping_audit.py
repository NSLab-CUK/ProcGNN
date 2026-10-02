"""Audit target_id -> batch edge index -> Frac feature index mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch

from ..constants import STREAM_EDGE_FEATURE_SLOTS
from .stream_vector_policy import SPECIES_TO_FEATURE_INDEX, STREAM_FEATURE_NAMES
from .target_row_fracs import _edge_matches_spec
from .target_row_spec import TargetRowSpec, load_target_row_specs


def _feature_mask_at(y_mask: torch.Tensor, edge_i: int, feat_j: int) -> float:
    if y_mask.ndim > 1 and y_mask.shape[1] > feat_j:
        return float(y_mask[edge_i, feat_j].item())
    if y_mask.ndim == 1:
        return float(y_mask[edge_i].item())
    return 0.0


def build_target_feature_mapping_audit_rows(
    *,
    export_meta,
    y_mask: torch.Tensor | None = None,
    edge_target_columns: Sequence[str] | None = None,
    train_cfg: Any | None = None,
    v4_path: Path | str | None = None,
) -> pd.DataFrame:
    """One row per v4 target row (reference), with batch-edge resolution when export_meta given."""
    cols = list(edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
    col_idx = {c: i for i, c in enumerate(cols)}
    specs = load_target_row_specs(v4_path=v4_path, train_cfg=train_cfg)
    n_edges = len(export_meta.process_id) if export_meta is not None else 0

    rows: list[dict[str, Any]] = []
    for spec in specs:
        feat_name = spec.frac_slot
        feat_idx = col_idx.get(feat_name, SPECIES_TO_FEATURE_INDEX.get(spec.species, -1))
        batch_edge_index = -1
        mask_val = float("nan")
        skip_reason = ""
        if spec.skip_v4_loss():
            skip_reason = spec.exclusion_reason or spec.formula_status or "skip_v4_loss"
        elif not spec.include_in_main_verified_macro:
            skip_reason = "excluded_from_main_verified_macro"

        if export_meta is not None and n_edges > 0:
            for e in range(n_edges):
                if _edge_matches_spec(edge_i=e, export_meta=export_meta, spec=spec):
                    batch_edge_index = int(e)
                    if y_mask is not None:
                        mask_val = _feature_mask_at(y_mask, e, int(feat_idx))
                    break
            if batch_edge_index < 0:
                skip_reason = skip_reason or "edge_not_in_batch"
            elif y_mask is not None and mask_val <= 0.0:
                stream_key = str(getattr(export_meta, "main_data_stream_key", [""])[batch_edge_index])
                if not str(stream_key).strip():
                    skip_reason = skip_reason or "empty_stream_key_mask_zero"
                else:
                    skip_reason = skip_reason or "feature_mask_zero"

        rows.append(
            {
                "process_id": spec.process_id,
                "target_id": spec.target_id,
                "canonical_answer_edge_id": spec.edge_id,
                "batch_edge_index": batch_edge_index,
                "target_species": spec.species,
                "target_feature_name": feat_name,
                "target_feature_index": int(feat_idx),
                "included_in_primary": bool(spec.include_in_main_verified_macro and not spec.skip_v4_loss()),
                "skip_reason": skip_reason,
                "required_stream_key": spec.required_stream_key,
            }
        )
    return pd.DataFrame(rows)


def write_target_feature_mapping_audit(
    output_dir: Path,
    *,
    export_meta=None,
    y_mask: torch.Tensor | None = None,
    edge_target_columns: Sequence[str] | None = None,
    train_cfg: Any | None = None,
) -> Path:
    out_dir = Path(output_dir)
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / "target_feature_mapping_audit.csv"
    df = build_target_feature_mapping_audit_rows(
        export_meta=export_meta,
        y_mask=y_mask,
        edge_target_columns=edge_target_columns,
        train_cfg=train_cfg,
    )
    meta = pd.DataFrame(
        [
            {
                "stream_target_dim": len(STREAM_FEATURE_NAMES),
                "stream_feature_names": ";".join(STREAM_FEATURE_NAMES),
            }
        ]
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        meta.to_csv(f, index=False)
        df.to_csv(f, index=False)
    return path
