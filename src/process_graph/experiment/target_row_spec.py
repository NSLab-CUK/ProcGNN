"""Atomic v4 target row identity (target_id + edge_id), not species-level grouping."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .target_v4_metrics import SPECIES_TO_FRAC_PROPERTY, normalize_stream_key
from .target_v4_provenance import load_provenance_policy, resolve_provenance_path
from .train_utils import resolve_answer_edge_species_weights

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_V4_TARGETS_PATH = PROJECT_ROOT / "data/reference/v4/target_stream_targets.csv"
MOLE_FLOW_SLOT = "Mole_Flow"

SKIP_LOSS_STATUSES = frozenset(
    {
        "mismatch",
        "unresolved",
        "unresolved_main_column",
        "mapping_conflict",
        "special_case_unresolved",
    }
)


@dataclass(frozen=True)
class TargetRowSpec:
    """One supervision unit = one v4 target_id row."""

    process_id: str
    process_num: int
    target_id: str
    target_feature: str
    target_stream: str
    edge_id: str
    species: str
    frac_slot: str
    mole_flow_slot: str
    formula: str
    scale: float
    formula_status: str = ""
    target_provenance: str = ""
    include_in_main_verified_macro: bool = True
    include_in_formula_macro: bool = True
    exclusion_reason: str = ""
    target_weight: float = 1.0
    species_weight: float = 1.0

    @property
    def required_stream_key(self) -> str:
        return normalize_stream_key(self.target_stream)

    @property
    def final_weight(self) -> float:
        return float(self.target_weight) * float(self.species_weight)

    def skip_v4_loss(self) -> bool:
        if self.exclusion_reason and str(self.exclusion_reason).strip().lower() not in ("", "nan", "none"):
            return True
        fs = str(self.formula_status).strip().lower()
        return fs in SKIP_LOSS_STATUSES


def _species_frac(species: str) -> str:
    slot = SPECIES_TO_FRAC_PROPERTY.get(str(species).strip())
    if slot is None:
        raise ValueError(f"unsupported target species for frac slot: {species!r}")
    return slot


def _parse_scale(raw: Any) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1.0


def _provenance_for_target(provenance: pd.DataFrame, target_id: str) -> dict[str, Any]:
    if provenance.empty or "target_id" not in provenance.columns:
        return {}
    sub = provenance[provenance["target_id"].astype(str) == str(target_id)]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


def load_target_row_specs(
    *,
    v4_path: Path | str | None = None,
    provenance_path: Path | str | None = None,
    train_cfg: Any | None = None,
    target_weight_by_id: Mapping[str, float] | None = None,
) -> list[TargetRowSpec]:
    """Load all v4 targets as atomic TargetRowSpec rows (one per target_id)."""
    csv_path = Path(v4_path) if v4_path else DEFAULT_V4_TARGETS_PATH
    if not csv_path.is_file():
        raise FileNotFoundError(f"target_stream_targets.csv not found: {csv_path}")

    provenance = load_provenance_policy(provenance_path or resolve_provenance_path())
    species_w: dict[str, float] = {}
    if train_cfg is not None:
        species_w = resolve_answer_edge_species_weights(train_cfg)
    tw_by_id = dict(target_weight_by_id or {})
    if train_cfg is not None:
        extra = getattr(train_cfg, "target_weight_by_id", None) or {}
        if isinstance(extra, Mapping):
            tw_by_id = {**tw_by_id, **{str(k): float(v) for k, v in extra.items()}}

    frame = pd.read_csv(csv_path, dtype={"process_id": int})
    specs: list[TargetRowSpec] = []
    seen: set[tuple[int, str]] = set()

    for row in frame.to_dict(orient="records"):
        pnum = int(row["process_id"])
        tid = str(row["target_id"]).strip()
        key = (pnum, tid)
        if key in seen:
            raise ValueError(f"duplicate target_id={tid!r} for process_id={pnum}")
        seen.add(key)
        species = str(row.get("target_species", "")).strip()
        if species not in SPECIES_TO_FRAC_PROPERTY:
            continue

        prov = _provenance_for_target(provenance, tid)
        fs = str(prov.get("formula_status", row.get("formula_type", ""))).strip()
        excl = str(prov.get("exclusion_reason", "")).strip()
        inc_main = bool(prov.get("include_in_main_verified_macro", True)) if prov else True
        inc_formula = bool(prov.get("include_in_formula_macro", inc_main)) if prov else True
        if provenance.empty:
            inc_main = True
            inc_formula = True

        stream_node = str(row.get("target_stream_node", ""))
        req_stream = normalize_stream_key(row.get("required_stream_key") or row.get("main_data_stream_key"))

        specs.append(
            TargetRowSpec(
                process_id=f"Process{pnum}",
                process_num=pnum,
                target_id=tid,
                target_feature=str(row.get("target_feature_name", "")),
                target_stream=req_stream or stream_node,
                edge_id=str(row.get("canonical_answer_edge_id", "")).strip(),
                species=species,
                frac_slot=_species_frac(species),
                mole_flow_slot=MOLE_FLOW_SLOT,
                formula=str(row.get("target_formula", "")),
                scale=_parse_scale(row.get("scale_factor", 1.0)),
                formula_status=fs,
                target_provenance=str(prov.get("target_provenance", prov.get("decision_status", ""))),
                include_in_main_verified_macro=inc_main,
                include_in_formula_macro=inc_formula,
                exclusion_reason=excl,
                target_weight=float(tw_by_id.get(tid, 1.0)),
                species_weight=float(species_w.get(species, 1.0)) if species_w else 1.0,
            )
        )
    return specs


def load_target_row_specs_by_process(
    **kwargs: Any,
) -> dict[str, list[TargetRowSpec]]:
    rows = load_target_row_specs(**kwargs)
    out: dict[str, list[TargetRowSpec]] = {}
    for spec in rows:
        out.setdefault(spec.process_id, []).append(spec)
    return out


def target_row_from_v4_spec(v4_spec: Any, *, provenance_row: Mapping[str, Any] | None = None) -> TargetRowSpec:
    """Bridge TargetSpecV4 -> TargetRowSpec for metrics."""
    prov = provenance_row or {}
    species = str(v4_spec.target_species)
    return TargetRowSpec(
        process_id=str(v4_spec.process_id),
        process_num=int(v4_spec.process_num),
        target_id=str(v4_spec.target_id),
        target_feature=str(v4_spec.target_name),
        target_stream=str(v4_spec.required_stream_key),
        edge_id=str(v4_spec.canonical_edge_id),
        species=species,
        frac_slot=_species_frac(species),
        mole_flow_slot=MOLE_FLOW_SLOT,
        formula=str(v4_spec.target_formula),
        scale=float(v4_spec.scale_factor),
        formula_status=str(prov.get("formula_status", v4_spec.formula_type)),
        target_provenance=str(prov.get("target_provenance", "")),
        include_in_main_verified_macro=bool(prov.get("include_in_main_verified_macro", True)),
        include_in_formula_macro=bool(prov.get("include_in_formula_macro", True)),
        exclusion_reason=str(prov.get("exclusion_reason", "")),
    )


def final_weight_for_spec(spec: TargetRowSpec, train_cfg: Any | None = None) -> float:
    if train_cfg is None:
        return spec.final_weight
    species_w = resolve_answer_edge_species_weights(train_cfg)
    base = float(getattr(train_cfg, "answer_edge_weight", 5.0))
    sw = float(species_w.get(spec.species, base))
    tw_map = getattr(train_cfg, "target_weight_by_id", None)
    if isinstance(tw_map, Mapping):
        tw = float(tw_map.get(spec.target_id, spec.target_weight))
    else:
        tw = float(spec.target_weight)
    return tw * sw
