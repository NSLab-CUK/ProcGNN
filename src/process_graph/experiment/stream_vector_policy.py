"""Stream target vector layout for single edge-all head (edge_all)."""

from __future__ import annotations

from typing import Any, Sequence

from ..constants import STREAM_EDGE_FEATURE_SLOTS

STREAM_FEATURE_NAMES: tuple[str, ...] = tuple(STREAM_EDGE_FEATURE_SLOTS)
STREAM_TARGET_DIM: int = len(STREAM_FEATURE_NAMES)


def feature_index(name: str) -> int:
    try:
        return STREAM_FEATURE_NAMES.index(str(name))
    except ValueError as exc:
        raise KeyError(f"stream feature {name!r} not in STREAM_EDGE_FEATURE_SLOTS") from exc


FRAC_H2_INDEX = feature_index("Frac_H2")
FRAC_CO2_INDEX = feature_index("Frac_CO2")
FRAC_H2O_INDEX = feature_index("Frac_H2O")
MOLE_FLOW_INDEX = feature_index("Mole_Flow")

SPECIES_TO_FEATURE_INDEX: dict[str, int] = {
    "H2": FRAC_H2_INDEX,
    "CO2": FRAC_CO2_INDEX,
    "H2O": FRAC_H2O_INDEX,
}


def build_stream_vector_policy(*, edge_feature_mask_used: bool = True) -> dict[str, Any]:
    return {
        "stream_target_dim": STREAM_TARGET_DIM,
        "stream_feature_names": list(STREAM_FEATURE_NAMES),
        "frac_h2_index": FRAC_H2_INDEX,
        "frac_co2_index": FRAC_CO2_INDEX,
        "frac_h2o_index": FRAC_H2O_INDEX,
        "mole_flow_index": MOLE_FLOW_INDEX,
        "edge_feature_mask_used": bool(edge_feature_mask_used),
        "empty_stream_key_edges_kept": True,
        "prediction_head": "single_edge_all_head",
    }


def format_stream_feature_index_table() -> str:
    lines = ["stream_feature_names (index: name):"]
    for i, n in enumerate(STREAM_FEATURE_NAMES):
        lines.append(f"  {i}: {n}")
    return "\n".join(lines)
