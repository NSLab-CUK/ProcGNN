from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from .constants import CellKind, HXRole, NodeRole, PassthroughPolicy


@dataclass
class CellSpec:
    kind: CellKind
    value: Optional[Union[str, float]] = None
    meta: Dict[str, str] = field(default_factory=dict)


@dataclass
class NodeSpec:
    name: str
    role: NodeRole
    type_raw: str
    type_canonical: str
    hx_role: HXRole = None
    features: Dict[str, CellSpec] = field(default_factory=dict)


@dataclass
class ProcessSpec:
    process_id: str
    node_names: List[str]
    edge_index: List[List[int]]
    nodes: Dict[str, NodeSpec]
    decoder_spec: Dict[str, Dict[str, str]]
    # Optional: node sheet column BFP_READOUT marks which sink node is used for fixed-task readout.
    # If the workbook's second sheet marks output sinks in column A with yellow fill, those nodes
    # override target (H2) / tailgas (CO2) readout when resolved (see parser.parse_process_file).
    # Keys are "target" and/or "tailgas"; values are graph node names (e.g. OUT_PROD).
    fixed_readout: Dict[str, str] = field(default_factory=dict)


@dataclass
class ResolveContext:
    process_spec: ProcessSpec
    data_row: Dict[str, float]
    incoming_map: Dict[str, List[str]]
    node_name: str
    slot: str
    passthrough_policy: PassthroughPolicy = "zero"


@dataclass
class GraphSample:
    process_id: str
    node_names: List[str]
    edge_index: List[List[int]]

    x_role: List[int]
    x_unit: List[int]
    x_hx_role: List[int]
    x_oper: List[List[float]]
    x_oper_mask: List[List[int]]

    targets: Dict[str, Dict[str, float]]

    edge_stream_role: List[int] = field(default_factory=list)
    edge_property_stream_role: List[int] = field(default_factory=list)
    edge_stream_id: List[int] = field(default_factory=list)
    edge_oper: List[List[float]] = field(default_factory=list)
    edge_oper_mask: List[List[int]] = field(default_factory=list)
    edge_stream_names: List[str] = field(default_factory=list)
    # Per-edge 1.0 if stream features joined successfully, 0.0 if missing key or join failed (v3 / future loaders).
    edge_feature_mask: List[float] = field(default_factory=list)
    # Context masks are graph-structure metadata. Context edges participate in
    # message passing but are excluded from prediction supervision and PINN.
    node_is_context: List[float] = field(default_factory=list)
    node_pinn_mask: List[float] = field(default_factory=list)
    edge_is_context: List[float] = field(default_factory=list)
    edge_is_predictable: List[float] = field(default_factory=list)
    edge_is_supervised: List[float] = field(default_factory=list)
    edge_is_target: List[float] = field(default_factory=list)
    edge_pinn_mask: List[float] = field(default_factory=list)
    # Optional topology-only HX pass-through relation. Indices are graph-local
    # here and are converted to batch-global edge rows by collate_graph_batch.
    hx_pair_current_edge_index: List[int] = field(default_factory=list)
    hx_paired_edge_index: List[int] = field(default_factory=list)
    hx_pair_mask: List[float] = field(default_factory=list)
    hx_pair_side: List[int] = field(default_factory=list)
    # Optional graph-level direct feed branch input. Values use the canonical
    # CH4, AIR, WATER order and are normalized with the x_oper train scaler.
    graph_feed_values: List[float] = field(default_factory=list)
    graph_feed_mask: List[int] = field(default_factory=list)
    # Structural edge inputs only. Must not contain target stream values from Process_Streams.csv.
    edge_struct_attr: List[List[float]] = field(default_factory=list)
    y_edge_true: List[List[float]] = field(default_factory=list)
    y_edge_mask: List[float] = field(default_factory=list)
    canonical_edge_ids: List[str] = field(default_factory=list)
    # Local edge row index (0..num_edges-1) per fixed slot; from target_answer_edges.csv, not heuristics.
    answer_edge_pos: Dict[str, int] = field(default_factory=dict)
    answer_edge_ids: Dict[str, str] = field(default_factory=dict)
    edge_target_columns: List[str] = field(default_factory=list)
    # Graph-node aligned optional physical terms for node-level PINN balances.
    node_q: List[float] = field(default_factory=list)
    node_w: List[float] = field(default_factory=list)
    node_qw_valid_mask: List[float] = field(default_factory=list)
    node_balance_exclude_mass: List[float] = field(default_factory=list)
    node_balance_exclude_component: List[float] = field(default_factory=list)
    node_balance_exclude_atom: List[float] = field(default_factory=list)
    node_balance_exclude_energy: List[float] = field(default_factory=list)
