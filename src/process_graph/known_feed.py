from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .constants import OPER_FEATURE_SLOTS
from .schema import CellSpec, ProcessSpec


KNOWN_FEED_NAMES: tuple[str, ...] = ("CH4", "AIR", "WATER")
KNOWN_FEED_FEATURE_NAMES: tuple[str, ...] = (
    "feed_ch4_flow",
    "feed_air_flow",
    "feed_water_flow",
)
KNOWN_FEED_RATIO_FEATURE_NAMES: tuple[str, ...] = (
    "log_air_ch4_ratio",
    "log_water_ch4_ratio",
)
_KNOWN_FEED_SLOT: dict[str, str] = {
    "CH4": "ch4",
    "AIR": "air",
    "WATER": "h2o",
}


@dataclass(frozen=True)
class KnownFeedCondition:
    enabled: bool = False
    mode: str = "v_input_node"
    feed_names: tuple[str, ...] = KNOWN_FEED_NAMES
    include_mask: bool = True
    preprocessing: str = "existing_operating_scaler"
    missing_v_input: str = "warn_and_mask"
    create_only_when_available: bool = True
    reuse_existing_source_role: bool = True
    connect_using_graph_metadata: bool = True
    context_only: bool = True
    supervised: bool = False
    target: bool = False
    include_in_pinn: bool = False
    include_in_global_pooling: bool = True
    ratio_features_enabled: bool = False
    ratio_epsilon: float = 1.0e-6
    ratio_attach_to_v_input: bool = True
    ratio_attach_to_burner: bool = True


@dataclass(frozen=True)
class KnownFeedBinding:
    feed_name: str
    operating_slot: str
    source_node_name: str | None
    cell_spec: CellSpec | None

    @property
    def source_reference(self) -> str | None:
        if self.cell_spec is None or self.cell_spec.kind != "reference":
            return None
        return str(self.cell_spec.value)


@dataclass(frozen=True)
class IndependentFeedTopologyBinding:
    feed_name: str
    feed_node_name: str
    source_node_name: str
    destination_node_name: str
    source_canonical_edge_id: str
    context_edge_id: str
    stream_key: str


def parse_known_feed_condition(raw: Any) -> KnownFeedCondition:
    if raw is None:
        raw = {}
    if isinstance(raw, KnownFeedCondition):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError("model.known_feed_condition must be a mapping.")

    enabled = bool(raw.get("enabled", False))
    mode = str(raw.get("mode", "v_input_node")).strip().lower()
    feed_names = tuple(str(x).strip().upper() for x in raw.get("feed_names", KNOWN_FEED_NAMES))
    include_mask = bool(raw.get("include_mask", True))
    preprocessing = str(raw.get("preprocessing", "existing_operating_scaler")).strip().lower()
    missing_v_input = str(raw.get("missing_v_input", "warn_and_mask")).strip().lower()
    feed_nodes = raw.get("feed_nodes", {}) or {}
    feed_edges = raw.get("feed_edges", {}) or {}
    ratio_features = raw.get("ratio_features", {}) or {}
    if not isinstance(feed_nodes, Mapping):
        raise TypeError("known_feed_condition.feed_nodes must be a mapping.")
    if not isinstance(feed_edges, Mapping):
        raise TypeError("known_feed_condition.feed_edges must be a mapping.")
    if not isinstance(ratio_features, Mapping):
        raise TypeError("known_feed_condition.ratio_features must be a mapping.")
    create_only_when_available = bool(
        feed_nodes.get("create_only_when_available", True)
    )
    reuse_existing_source_role = bool(
        feed_nodes.get("reuse_existing_source_role", True)
    )
    connect_using_graph_metadata = bool(
        feed_nodes.get("connect_using_graph_metadata", True)
    )
    context_only = bool(feed_edges.get("context_only", True))
    supervised = bool(feed_edges.get("supervised", False))
    target = bool(feed_edges.get("target", False))
    include_in_pinn = bool(feed_edges.get("include_in_pinn", False))
    include_in_global_pooling = bool(
        feed_edges.get("include_in_global_pooling", True)
    )
    ratio_features_enabled = bool(ratio_features.get("enabled", False))
    ratio_transform = str(ratio_features.get("transform", "log_ratio")).strip().lower()
    ratio_epsilon = float(ratio_features.get("epsilon", 1.0e-6))
    ratio_attach_to_v_input = bool(ratio_features.get("attach_to_v_input", True))
    ratio_attach_to_burner = bool(ratio_features.get("attach_to_burner", True))

    if enabled and mode not in {"v_input_node", "independent_feed_nodes"}:
        raise ValueError(
            "known_feed_condition.mode must be 'v_input_node' or "
            "'independent_feed_nodes'."
        )
    if enabled and feed_names != KNOWN_FEED_NAMES:
        raise ValueError(
            "known_feed_condition.feed_names must be exactly [CH4, AIR, WATER] in that "
            f"order, got {list(feed_names)}."
        )
    if enabled and not include_mask:
        raise ValueError("known_feed_condition.include_mask must be true.")
    if enabled and preprocessing != "existing_operating_scaler":
        raise ValueError(
            "known_feed_condition.preprocessing must be 'existing_operating_scaler'."
        )
    if enabled and mode == "v_input_node" and missing_v_input not in {
        "warn_and_mask",
        "error",
    }:
        raise ValueError(
            "known_feed_condition.missing_v_input must be 'warn_and_mask' or 'error'."
        )
    if enabled and mode == "independent_feed_nodes":
        f2_clean_ablation = {
            "feed_nodes.create_only_when_available": create_only_when_available,
            "feed_nodes.reuse_existing_source_role": reuse_existing_source_role,
            "feed_nodes.connect_using_graph_metadata": connect_using_graph_metadata,
            "feed_edges.context_only": context_only,
            "feed_edges.supervised=false": not supervised,
            "feed_edges.target=false": not target,
            "feed_edges.include_in_pinn=false": not include_in_pinn,
            "feed_edges.include_in_global_pooling": include_in_global_pooling,
        }
        invalid = [name for name, valid in f2_clean_ablation.items() if not valid]
        if invalid:
            raise ValueError(
                "independent_feed_nodes clean ablation requires: "
                + ", ".join(invalid)
            )
    if ratio_features_enabled:
        if not enabled:
            raise ValueError(
                "known_feed_condition.ratio_features requires known_feed_condition.enabled=true."
            )
        if ratio_transform != "log_ratio":
            raise ValueError(
                "known_feed_condition.ratio_features.transform must be 'log_ratio'."
            )
        if not math.isfinite(ratio_epsilon) or ratio_epsilon <= 0.0:
            raise ValueError(
                "known_feed_condition.ratio_features.epsilon must be finite and positive."
            )
        if not (ratio_attach_to_v_input or ratio_attach_to_burner):
            raise ValueError(
                "known_feed_condition.ratio_features must attach to V_INPUT, BURNER, or both."
            )
    return KnownFeedCondition(
        enabled=enabled,
        mode=mode,
        feed_names=feed_names,
        include_mask=include_mask,
        preprocessing=preprocessing,
        missing_v_input=missing_v_input,
        create_only_when_available=create_only_when_available,
        reuse_existing_source_role=reuse_existing_source_role,
        connect_using_graph_metadata=connect_using_graph_metadata,
        context_only=context_only,
        supervised=supervised,
        target=target,
        include_in_pinn=include_in_pinn,
        include_in_global_pooling=include_in_global_pooling,
        ratio_features_enabled=ratio_features_enabled,
        ratio_epsilon=ratio_epsilon,
        ratio_attach_to_v_input=ratio_attach_to_v_input,
        ratio_attach_to_burner=ratio_attach_to_burner,
    )


def known_feed_feature_names(raw: Any = None) -> tuple[str, ...]:
    cfg = parse_known_feed_condition(raw)
    if not cfg.enabled:
        return ()
    ratio_names = KNOWN_FEED_RATIO_FEATURE_NAMES if cfg.ratio_features_enabled else ()
    return KNOWN_FEED_FEATURE_NAMES + ratio_names


def operating_feature_names(raw: Any = None) -> tuple[str, ...]:
    cfg = parse_known_feed_condition(raw)
    base = tuple(OPER_FEATURE_SLOTS)
    return base + known_feed_feature_names(cfg) if cfg.enabled else base


def operating_feature_dim(raw: Any = None) -> int:
    return len(operating_feature_names(raw))


def _node_tokens(node_name: str) -> set[str]:
    normalized = str(node_name).upper().replace("-", "_")
    return {token for token in normalized.split("_") if token}


def _pick_source_binding(
    process_spec: ProcessSpec,
    *,
    feed_name: str,
    operating_slot: str,
) -> KnownFeedBinding:
    candidates: list[tuple[str, CellSpec]] = []
    for node in process_spec.nodes.values():
        if node.role != "source":
            continue
        cell_spec = node.features.get(operating_slot)
        if cell_spec is None or cell_spec.kind == "missing":
            continue
        candidates.append((node.name, cell_spec))

    if feed_name == "CH4":
        preferred = [
            item
            for item in candidates
            if "CH4" in _node_tokens(item[0]) and "FUEL" not in _node_tokens(item[0])
        ]
    elif feed_name == "AIR":
        preferred = [item for item in candidates if "AIR" in _node_tokens(item[0])]
    else:
        preferred = [
            item
            for item in candidates
            if _node_tokens(item[0]).intersection({"WATER", "H2O"})
        ]

    if len(preferred) == 1:
        node_name, cell_spec = preferred[0]
        return KnownFeedBinding(feed_name, operating_slot, node_name, cell_spec)
    if not preferred and len(candidates) == 1:
        node_name, cell_spec = candidates[0]
        return KnownFeedBinding(feed_name, operating_slot, node_name, cell_spec)
    if len(preferred) > 1 or len(candidates) > 1:
        details = [
            f"{node_name}:{cell_spec.kind}={cell_spec.value!r}"
            for node_name, cell_spec in (preferred or candidates)
        ]
        raise ValueError(
            f"Ambiguous {feed_name} source feed in {process_spec.process_id}: {details}"
        )
    return KnownFeedBinding(feed_name, operating_slot, None, None)


def known_feed_bindings(process_spec: ProcessSpec) -> tuple[KnownFeedBinding, ...]:
    return tuple(
        _pick_source_binding(
            process_spec,
            feed_name=feed_name,
            operating_slot=_KNOWN_FEED_SLOT[feed_name],
        )
        for feed_name in KNOWN_FEED_NAMES
    )


def known_feed_config_dict(cfg: KnownFeedCondition) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "enabled": bool(cfg.enabled),
        "mode": cfg.mode,
        "feed_names": list(cfg.feed_names),
        "include_mask": bool(cfg.include_mask),
        "preprocessing": cfg.preprocessing,
        "missing_v_input": cfg.missing_v_input,
    }
    if cfg.mode == "independent_feed_nodes":
        payload["feed_nodes"] = {
            "create_only_when_available": bool(cfg.create_only_when_available),
            "reuse_existing_source_role": bool(cfg.reuse_existing_source_role),
            "connect_using_graph_metadata": bool(cfg.connect_using_graph_metadata),
        }
        payload["feed_edges"] = {
            "context_only": bool(cfg.context_only),
            "supervised": bool(cfg.supervised),
            "target": bool(cfg.target),
            "include_in_pinn": bool(cfg.include_in_pinn),
            "include_in_global_pooling": bool(cfg.include_in_global_pooling),
        }
    if cfg.ratio_features_enabled:
        payload["ratio_features"] = {
            "enabled": True,
            "transform": "log_ratio",
            "epsilon": float(cfg.ratio_epsilon),
            "attach_to_v_input": bool(cfg.ratio_attach_to_v_input),
            "attach_to_burner": bool(cfg.ratio_attach_to_burner),
        }
    return payload


def known_feed_ratio_values(
    feed_values: Sequence[float],
    feed_masks: Sequence[int],
    cfg: KnownFeedCondition,
) -> tuple[list[float], list[int]]:
    """Return log(AIR/CH4) and log(WATER/CH4) with validity masks."""
    if not cfg.ratio_features_enabled:
        return [], []
    if len(feed_values) != len(KNOWN_FEED_NAMES) or len(feed_masks) != len(KNOWN_FEED_NAMES):
        raise ValueError("Known-feed ratio features require CH4, AIR, and WATER values/masks.")
    ch4 = float(feed_values[0])
    ch4_valid = bool(feed_masks[0]) and math.isfinite(ch4) and ch4 > cfg.ratio_epsilon
    values: list[float] = []
    masks: list[int] = []
    for numerator_index in (1, 2):
        numerator = float(feed_values[numerator_index])
        valid = (
            ch4_valid
            and bool(feed_masks[numerator_index])
            and math.isfinite(numerator)
            and numerator >= 0.0
        )
        if valid:
            values.append(
                math.log(
                    (numerator + cfg.ratio_epsilon)
                    / (ch4 + cfg.ratio_epsilon)
                )
            )
            masks.append(1)
        else:
            values.append(0.0)
            masks.append(0)
    return values, masks


def independent_feed_node_name(feed_name: str) -> str:
    normalized = str(feed_name).strip().upper()
    if normalized not in KNOWN_FEED_NAMES:
        raise ValueError(f"Unknown feed name {feed_name!r}.")
    return f"V_FEED_{normalized}"


def independent_feed_context_edge_id(process_id: int, feed_name: str) -> str:
    normalized = str(feed_name).strip().upper()
    if normalized not in KNOWN_FEED_NAMES:
        raise ValueError(f"Unknown feed name {feed_name!r}.")
    return f"P{int(process_id):02d}_F2_FEED_{normalized}"


def build_independent_feed_topology_bindings(
    *,
    process_id: int,
    bindings: Sequence[KnownFeedBinding],
    canonical_edge_rows: Sequence[Mapping[str, Any]],
) -> tuple[IndependentFeedTopologyBinding, ...]:
    """Resolve F2 destinations from canonical source metadata, never node-name guesses."""
    output: list[IndependentFeedTopologyBinding] = []
    seen_nodes: set[str] = set()
    seen_edges: set[str] = set()
    for binding in bindings:
        if binding.source_node_name is None or binding.cell_spec is None:
            continue
        source_name = str(binding.source_node_name).strip().upper()
        matches = [
            row
            for row in canonical_edge_rows
            if str(row.get("src_node_raw", "")).strip().upper() == source_name
            and str(row.get("src_node", "")).strip().upper() == "V_INPUT"
        ]
        if len(matches) != 1:
            edge_ids = [
                str(row.get("canonical_edge_id", "")) for row in matches
            ]
            raise ValueError(
                f"Process{int(process_id)} {binding.feed_name}: expected exactly one "
                "canonical V_INPUT edge whose src_node_raw matches "
                f"{binding.source_node_name!r}, got {edge_ids}."
            )
        row = matches[0]
        destination = str(row.get("dst_node", "")).strip()
        source_edge_id = str(row.get("canonical_edge_id", "")).strip()
        if not destination or not source_edge_id:
            raise ValueError(
                f"Process{int(process_id)} {binding.feed_name}: incomplete canonical "
                "source-edge metadata."
            )
        node_name = independent_feed_node_name(binding.feed_name)
        edge_id = independent_feed_context_edge_id(process_id, binding.feed_name)
        if node_name in seen_nodes or edge_id in seen_edges:
            raise ValueError(
                f"Duplicate F2 topology identity for Process{int(process_id)}: "
                f"node={node_name!r} edge={edge_id!r}."
            )
        seen_nodes.add(node_name)
        seen_edges.add(edge_id)
        output.append(
            IndependentFeedTopologyBinding(
                feed_name=binding.feed_name,
                feed_node_name=node_name,
                source_node_name=binding.source_node_name,
                destination_node_name=destination,
                source_canonical_edge_id=source_edge_id,
                context_edge_id=edge_id,
                stream_key=str(row.get("main_data_stream_key", "")).strip(),
            )
        )
    return tuple(output)


def append_known_feed_values(
    base_values: Sequence[float],
    *,
    feed_values: Sequence[float],
) -> list[float]:
    if len(feed_values) != len(KNOWN_FEED_FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(KNOWN_FEED_FEATURE_NAMES)} feed values, got {len(feed_values)}."
        )
    return [float(x) for x in base_values] + [float(x) for x in feed_values]
