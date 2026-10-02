from __future__ import annotations

import re
import sys
import os
from typing import Dict, List, Set, Tuple

from .constants import HX_ROLE_TO_IDX, OPER_FEATURE_SLOTS, ROLE_TO_IDX, UNIT_TO_IDX
from .schema import CellSpec, GraphSample, ResolveContext


def build_incoming_node_map(process_spec) -> Dict[str, List[str]]:
    node_names = process_spec.node_names
    src_list, dst_list = process_spec.edge_index

    incoming: Dict[str, List[str]] = {name: [] for name in node_names}
    for src, dst in zip(src_list, dst_list):
        src_name = node_names[src]
        dst_name = node_names[dst]
        incoming[dst_name].append(src_name)

    return incoming


def _safe_float(value) -> float:
    if value is None:
        raise ValueError("Cannot convert None to float.")
    return float(value)


PROCESS_COLUMN_ALIASES: Dict[str, Dict[str, str]] = {
    "Process2": {
        "CH4_Flow": "CH4_Total",
    },
    "Process6": {
        "P_Burner": "P_Comb",
        "FUEL_Flow": "FUEL_CH4_Flow",
    },
    "Process8": {
        "SC_Ratio": "S/C_ratio",
        "P_Reaction": "P_Rxn",
        "T_HEAT1": "T_Heat1",
    },
    "Process9": {
        "Q_BURNER": "Q_Burner",
        "PROD_Mole": "PROD_H2_Mole",
    },
    "Process10": {
        "Q_BURNER": "Q_BN",
    },
}

_WARNED_UNRESOLVED_REFS: set[tuple[str, str]] = set()
_LINEAR_REF = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*\*\s*([A-Za-z0-9_]+)\s*$")


def _warn_unresolved_ref(process_id: str, ref: str) -> None:
    if os.environ.get("PROCESS_GRAPH_RESOLVER_WARNINGS", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    key = (str(process_id), str(ref))
    if key in _WARNED_UNRESOLVED_REFS:
        return
    _WARNED_UNRESOLVED_REFS.add(key)
    print(
        f"[resolver][warning] unresolved reference for {process_id}: {ref!r} -> mask=0 (value=0).",
        file=sys.stderr,
    )


def _lookup_reference(data_row: Dict[str, float], ref: str, process_id: str) -> Tuple[float, int]:
    ref_clean = str(ref).strip()
    if ref_clean in data_row and data_row[ref_clean] is not None:
        return _safe_float(data_row[ref_clean]), 1

    m = _LINEAR_REF.match(ref_clean)
    if m:
        coef = float(m.group(1))
        base_col = m.group(2).strip()
        if base_col in data_row and data_row[base_col] is not None:
            return coef * _safe_float(data_row[base_col]), 1

    aliases = PROCESS_COLUMN_ALIASES.get(str(process_id), {})
    alias_col = aliases.get(ref_clean)
    if alias_col and alias_col in data_row and data_row[alias_col] is not None:
        return _safe_float(data_row[alias_col]), 1

    ref_fold = ref_clean.casefold()
    for col_name, value in data_row.items():
        if str(col_name).casefold() == ref_fold and value is not None:
            return _safe_float(value), 1

    _warn_unresolved_ref(process_id, ref_clean)
    return 0.0, 0


def resolve_passthrough_pressure_from_upstream(
    context: ResolveContext,
    node_name: str,
    visited: Set[str] | None = None,
) -> Tuple[float, int]:
    if visited is None:
        visited = set()

    if node_name in visited:
        return 0.0, 0
    visited.add(node_name)

    process_spec = context.process_spec
    upstream_nodes = context.incoming_map.get(node_name, [])

    for upstream_name in upstream_nodes:
        upstream_node = process_spec.nodes[upstream_name]
        press_spec = upstream_node.features.get("press", CellSpec(kind="missing"))

        if press_spec.kind == "constant":
            return float(press_spec.value), 1

        if press_spec.kind == "reference":
            ref = str(press_spec.value)
            value, mask = _lookup_reference(context.data_row, ref, context.process_spec.process_id)
            if mask == 1:
                return value, mask

        if press_spec.kind == "passthrough":
            value, mask = resolve_passthrough_pressure_from_upstream(
                context=context,
                node_name=upstream_name,
                visited=visited,
            )
            if mask == 1:
                return value, mask

    return 0.0, 0


def resolve_feature_value(cell_spec: CellSpec, context: ResolveContext) -> Tuple[float, int]:
    if cell_spec.kind == "missing":
        return 0.0, 0

    if cell_spec.kind == "constant":
        return float(cell_spec.value), 1

    if cell_spec.kind == "reference":
        ref = str(cell_spec.value)
        return _lookup_reference(context.data_row, ref, context.process_spec.process_id)

    if cell_spec.kind == "passthrough":
        hint_ref = cell_spec.meta.get("hint_ref")
        if hint_ref is not None:
            value, mask = _lookup_reference(context.data_row, hint_ref, context.process_spec.process_id)
            if mask == 1:
                return value, mask

        if context.passthrough_policy == "inherit_upstream":
            return resolve_passthrough_pressure_from_upstream(
                context=context,
                node_name=context.node_name,
            )

        return 0.0, 1

    raise ValueError(f"Unknown CellKind: {cell_spec.kind}")


def build_targets(decoder_spec: dict[str, dict[str, str]], data_row: dict) -> dict[str, dict[str, float]]:
    targets: dict[str, dict[str, float]] = {}

    for category, node_to_target in decoder_spec.items():
        targets[category] = {}
        for node_name, raw_target_name in node_to_target.items():
            if raw_target_name in data_row and data_row[raw_target_name] is not None:
                targets[category][node_name] = float(data_row[raw_target_name])

    return targets


def build_graph_sample(process_spec, data_row: dict, passthrough_policy: str = "zero") -> GraphSample:
    x_role: List[int] = []
    x_unit: List[int] = []
    x_hx_role: List[int] = []
    x_oper: List[List[float]] = []
    x_oper_mask: List[List[int]] = []

    incoming_map = build_incoming_node_map(process_spec)

    for node_name in process_spec.node_names:
        node = process_spec.nodes[node_name]

        x_role.append(ROLE_TO_IDX[node.role])
        x_unit.append(UNIT_TO_IDX[node.type_canonical])
        x_hx_role.append(HX_ROLE_TO_IDX[node.hx_role])

        feat_values: List[float] = []
        feat_masks: List[int] = []

        for slot in OPER_FEATURE_SLOTS:
            cell_spec = node.features[slot]
            context = ResolveContext(
                process_spec=process_spec,
                data_row=data_row,
                incoming_map=incoming_map,
                node_name=node_name,
                slot=slot,
                passthrough_policy=passthrough_policy,
            )
            value, mask = resolve_feature_value(cell_spec, context)
            feat_values.append(value)
            feat_masks.append(mask)

        x_oper.append(feat_values)
        x_oper_mask.append(feat_masks)

    targets = build_targets(process_spec.decoder_spec, data_row)

    return GraphSample(
        process_id=process_spec.process_id,
        node_names=process_spec.node_names,
        edge_index=process_spec.edge_index,
        x_role=x_role,
        x_unit=x_unit,
        x_hx_role=x_hx_role,
        x_oper=x_oper,
        x_oper_mask=x_oper_mask,
        targets=targets,
    )
