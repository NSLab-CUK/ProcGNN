from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class HXPairRelationConfig:
    enabled: bool = False
    source: str = "topology_metadata"
    metadata_path: str = "data/reference/v3/hx_edge_pairs.csv"
    require_one_to_one: bool = True
    require_symmetric_pair: bool = True
    side_embedding_dim: int = 16
    hidden_dim: int = 256
    output_dim: int = 384
    dropout: float = 0.1
    gate_init: float = 0.05
    apply_inside_flow_gnn: bool = False


def parse_hx_pair_relation(raw: Any = None) -> HXPairRelationConfig:
    if raw is None:
        raw = {}
    if isinstance(raw, HXPairRelationConfig):
        return raw
    if not isinstance(raw, Mapping):
        raise TypeError("model.hx_pair_relation must be a mapping.")

    relation_mlp = raw.get("relation_mlp", {}) or {}
    residual = raw.get("residual", {}) or {}
    if not isinstance(relation_mlp, Mapping):
        raise TypeError("model.hx_pair_relation.relation_mlp must be a mapping.")
    if not isinstance(residual, Mapping):
        raise TypeError("model.hx_pair_relation.residual must be a mapping.")

    side_embedding_dim = int(raw.get("side_embedding_dim", 16))
    output_dim = int(relation_mlp.get("output_dim", 384))
    expected_input_dim = 2 * output_dim + side_embedding_dim
    configured_input_dim = int(
        relation_mlp.get("input_dim", expected_input_dim)
    )
    if configured_input_dim != expected_input_dim:
        raise ValueError(
            "hx_pair_relation.relation_mlp.input_dim must equal "
            "2 * output_dim + side_embedding_dim: "
            f"{configured_input_dim} != {expected_input_dim}."
        )

    config = HXPairRelationConfig(
        enabled=bool(raw.get("enabled", False)),
        source=str(raw.get("source", "topology_metadata")).strip().lower(),
        metadata_path=str(
            raw.get("metadata_path", "data/reference/v3/hx_edge_pairs.csv")
        ).strip(),
        require_one_to_one=bool(raw.get("require_one_to_one", True)),
        require_symmetric_pair=bool(raw.get("require_symmetric_pair", True)),
        side_embedding_dim=side_embedding_dim,
        hidden_dim=int(relation_mlp.get("hidden_dim", 256)),
        output_dim=output_dim,
        dropout=float(relation_mlp.get("dropout", 0.1)),
        gate_init=float(residual.get("gate_init", 0.05)),
        apply_inside_flow_gnn=bool(raw.get("apply_inside_flow_gnn", False)),
    )
    if not config.enabled:
        return config
    if config.source != "topology_metadata":
        raise ValueError("hx_pair_relation.source must be 'topology_metadata'.")
    if not config.metadata_path:
        raise ValueError("hx_pair_relation.metadata_path must not be empty.")
    if not config.require_one_to_one or not config.require_symmetric_pair:
        raise ValueError(
            "HX pair relation requires one-to-one and symmetric pair validation."
        )
    if config.side_embedding_dim <= 0 or config.hidden_dim <= 0 or config.output_dim <= 0:
        raise ValueError("HX pair relation dimensions must be positive.")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("hx_pair_relation dropout must be in [0, 1).")
    if not math.isfinite(config.gate_init) or not 0.0 < config.gate_init < 1.0:
        raise ValueError("hx_pair_relation gate_init must be finite and in (0, 1).")
    if config.apply_inside_flow_gnn:
        raise ValueError(
            "R0-HXPair applies the relation only to the final edge descriptor; "
            "apply_inside_flow_gnn must be false."
        )
    if not bool(residual.get("enabled", True)) or not bool(
        residual.get("gate_enabled", True)
    ):
        raise ValueError("HX pair relation requires a gated residual update.")
    apply_to = tuple(str(value) for value in raw.get("apply_to", ["final_edge_embedding"]))
    if apply_to != ("final_edge_embedding",):
        raise ValueError(
            "hx_pair_relation.apply_to must be ['final_edge_embedding']."
        )
    return config


def hx_pair_config_dict(config: HXPairRelationConfig) -> dict[str, Any]:
    if not config.enabled:
        return {"enabled": False}
    return {
        "enabled": True,
        "source": config.source,
        "metadata_path": config.metadata_path,
        "require_one_to_one": config.require_one_to_one,
        "require_symmetric_pair": config.require_symmetric_pair,
        "side_embedding_dim": config.side_embedding_dim,
        "relation_mlp": {
            "input_dim": 2 * config.output_dim + config.side_embedding_dim,
            "hidden_dim": config.hidden_dim,
            "output_dim": config.output_dim,
            "dropout": config.dropout,
        },
        "residual": {
            "enabled": True,
            "gate_enabled": True,
            "gate_init": config.gate_init,
        },
        "apply_to": ["final_edge_embedding"],
        "apply_inside_flow_gnn": False,
    }
