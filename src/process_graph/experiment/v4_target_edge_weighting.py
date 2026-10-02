"""V4 target-row edge loss weighting for edge_all (species Frac slots on canonical edges)."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from ..constants import STREAM_EDGE_FEATURE_SLOTS
from .target_v4_metrics import SPECIES_TO_FRAC_PROPERTY, normalize_stream_key
from .train_utils import resolve_answer_edge_species_weights

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_V4_TARGETS_PATH = PROJECT_ROOT / "data/reference/v4/target_stream_targets.csv"

_LEGACY_WARNED = False


@dataclass(frozen=True)
class V4TargetWeightRow:
    process_id: str
    target_id: str
    target_species: str
    required_stream_key: str
    canonical_edge_id: str
    required_properties: tuple[str, ...]


def load_v4_target_weight_rows(path: Path | None = None) -> list[V4TargetWeightRow]:
    csv_path = path or DEFAULT_V4_TARGETS_PATH
    if not csv_path.is_file():
        return []
    frame = pd.read_csv(csv_path, dtype={"process_id": int})
    rows: list[V4TargetWeightRow] = []
    for rec in frame.to_dict(orient="records"):
        species = str(rec.get("target_species", "")).strip()
        if species not in SPECIES_TO_FRAC_PROPERTY:
            continue
        try:
            props = json.loads(str(rec.get("required_properties", "[]")))
            if not isinstance(props, list):
                props = []
        except json.JSONDecodeError:
            props = []
        rows.append(
            V4TargetWeightRow(
                process_id=f"Process{int(rec['process_id'])}",
                target_id=str(rec.get("target_id", "")),
                target_species=species,
                required_stream_key=normalize_stream_key(
                    rec.get("required_stream_key") or rec.get("main_data_stream_key")
                ),
                canonical_edge_id=str(rec.get("canonical_answer_edge_id", "")).strip(),
                required_properties=tuple(str(p) for p in props),
            )
        )
    return rows


def species_frac_property(species: str) -> str | None:
    return SPECIES_TO_FRAC_PROPERTY.get(str(species).strip())


def build_v4_target_loss_weight_tensor(
    *,
    export_meta,
    edge_target_columns: Sequence[str],
    train_cfg: Any,
    v4_targets_path: Path | None = None,
) -> tuple[torch.Tensor | None, list[str]]:
    """Build per-element loss weights for edge_all training/eval.

    When ``use_v4_target_edge_weighting`` is true (default) and mode is ``target_rows``,
    apply species weights only on the Frac_* slot of each v4 target's canonical edge.

    When disabled, fall back to legacy stream-key + ``required_properties`` weighting.
    """
    global _LEGACY_WARNED
    if export_meta is None:
        return None, []

    species_weights = resolve_answer_edge_species_weights(train_cfg)
    default_w = float(getattr(train_cfg, "answer_edge_weight", 5.0))
    if max(species_weights.values(), default=default_w) <= 1.0:
        return None, []

    use_v4 = bool(getattr(train_cfg, "use_v4_target_edge_weighting", True))
    use_legacy = bool(getattr(train_cfg, "use_legacy_answer_weighting", False))
    mode = str(getattr(train_cfg, "v4_target_weighting_mode", "target_row")).strip().lower()
    if mode in ("target_rows", "target_row"):
        mode = "target_row"
    if not use_v4:
        if not _LEGACY_WARNED:
            warnings.warn(
                "use_v4_target_edge_weighting=false: H2O/CO2/H2 loss uses legacy "
                "stream-key + required_properties matching (no canonical_edge_id). "
                "Set use_v4_target_edge_weighting=true for per-target-row routing.",
                stacklevel=2,
            )
            _LEGACY_WARNED = True
        return _build_legacy_stream_property_weights(
            export_meta=export_meta,
            edge_target_columns=edge_target_columns,
            species_weights=species_weights,
            default_w=default_w,
            v4_targets_path=v4_targets_path,
        )

    if bool(getattr(train_cfg, "use_v4_target_row_loss", True)) and not use_legacy:
        return None, [
            "element-level v4 weighting skipped: use_v4_target_row_loss=true "
            "(set use_legacy_answer_weighting=true to enable edge_stream_loss_weight tensor)"
        ]

    if not use_legacy and not use_v4:
        return None, []

    if mode != "target_row":
        return _build_legacy_stream_property_weights(
            export_meta=export_meta,
            edge_target_columns=edge_target_columns,
            species_weights=species_weights,
            default_w=default_w,
            v4_targets_path=v4_targets_path,
        )

    rows = load_v4_target_weight_rows(v4_targets_path)
    if not rows:
        return None, [f"v4 target table not found or empty: {v4_targets_path or DEFAULT_V4_TARGETS_PATH}"]

    cols = [str(c) for c in edge_target_columns]
    col_idx = {c: i for i, c in enumerate(cols)}
    n_edges = len(export_meta.process_id)
    w = torch.ones((n_edges, len(cols)), dtype=torch.float32)
    warn_msgs: list[str] = []
    missing_frac: set[str] = set()

    edge_ids = getattr(export_meta, "canonical_edge_id", None)
    if edge_ids is None:
        warn_msgs.append("edge_export_meta missing canonical_edge_id; v4 target_rows weighting skipped")
        return None, warn_msgs

    for edge_i, (pid, stream_key_raw, edge_id_raw) in enumerate(
        zip(export_meta.process_id, export_meta.main_data_stream_key, edge_ids)
    ):
        stream_key = normalize_stream_key(stream_key_raw)
        edge_id = str(edge_id_raw).strip()
        for row in rows:
            if str(pid) != row.process_id:
                continue
            if stream_key != row.required_stream_key:
                continue
            if row.canonical_edge_id and edge_id != row.canonical_edge_id:
                continue
            sw = float(species_weights.get(row.target_species, default_w))
            if sw <= 1.0:
                continue
            frac_col = species_frac_property(row.target_species)
            if frac_col is None:
                continue
            j = col_idx.get(frac_col)
            if j is None:
                missing_frac.add(frac_col)
                continue
            w[edge_i, j] = max(float(w[edge_i, j].item()), sw)

    if missing_frac:
        warn_msgs.append(
            f"v4 target_rows weighting skipped unknown Frac columns: {sorted(missing_frac)}"
        )
    if torch.any(w > 1.0):
        return w, warn_msgs
    return None, warn_msgs


def _build_legacy_stream_property_weights(
    *,
    export_meta,
    edge_target_columns: Sequence[str],
    species_weights: Mapping[str, float],
    default_w: float,
    v4_targets_path: Path | None,
) -> tuple[torch.Tensor | None, list[str]]:
    """Legacy: match (process_id, stream_key) and weight all required_properties."""
    path = v4_targets_path or DEFAULT_V4_TARGETS_PATH
    if not path.is_file():
        return None, [f"v4 target table not found: {path}"]
    try:
        df = pd.read_csv(path, dtype={"process_id": int})
    except Exception as exc:
        return None, [f"failed to read v4 target table: {exc}"]

    cols = [str(c) for c in edge_target_columns]
    col_idx = {c: i for i, c in enumerate(cols)}
    n_edges = len(export_meta.process_id)
    w = torch.ones((n_edges, len(cols)), dtype=torch.float32)
    missing_props: set[str] = set()
    specs_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    warn_msgs: list[str] = []

    for rec in df.to_dict(orient="records"):
        species = str(rec.get("target_species", ""))
        sw = float(species_weights.get(species, default_w))
        if sw <= 1.0:
            continue
        stream_key = normalize_stream_key(rec.get("required_stream_key") or rec.get("main_data_stream_key"))
        try:
            props = json.loads(str(rec.get("required_properties", "[]")))
            if not isinstance(props, list):
                props = []
        except json.JSONDecodeError:
            warn_msgs.append(f"invalid required_properties for target_id={rec.get('target_id')}")
            continue
        specs_by_key.setdefault((f"Process{int(rec['process_id'])}", stream_key), []).append(
            {"species_weight": sw, "required_properties": [str(p) for p in props]}
        )

    for edge_i, (pid, stream_key_raw) in enumerate(
        zip(export_meta.process_id, export_meta.main_data_stream_key)
    ):
        stream_key = normalize_stream_key(stream_key_raw)
        for spec in specs_by_key.get((str(pid), stream_key), []):
            sw = float(spec["species_weight"])
            for prop in spec["required_properties"]:
                j = col_idx.get(str(prop))
                if j is None:
                    missing_props.add(str(prop))
                    continue
                w[edge_i, j] = max(float(w[edge_i, j].item()), sw)

    if missing_props:
        warn_msgs.append(
            f"legacy stream weighting skipped unknown properties: {sorted(missing_props)}"
        )
    if torch.any(w > 1.0):
        return w, warn_msgs
    return None, warn_msgs


def loss_weight_for_species(train_cfg: Any, species: str) -> float:
    species_weights = resolve_answer_edge_species_weights(train_cfg)
    base = float(getattr(train_cfg, "answer_edge_weight", 5.0))
    return float(species_weights.get(str(species).strip(), base))
