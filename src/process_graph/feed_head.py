from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .known_feed import KNOWN_FEED_NAMES


@dataclass(frozen=True)
class FeedHeadConditioningConfig:
    enabled: bool = False
    features: tuple[str, ...] = KNOWN_FEED_NAMES
    use_scaled_values: bool = True
    use_masks: bool = True
    include_ratios: bool = False
    input_dim: int = 6
    hidden_dim: int = 32
    output_dim: int = 64
    dropout: float = 0.0
    broadcast_to_prediction_edges: bool = True
    concat_position: str = "after_legacy_edge_descriptor"


def parse_feed_head_conditioning(raw: Any = None) -> FeedHeadConditioningConfig:
    if raw is None:
        return FeedHeadConditioningConfig()
    if not isinstance(raw, Mapping):
        raise TypeError("model.feed_head_conditioning must be a mapping.")
    encoder = raw.get("encoder", {}) or {}
    if not isinstance(encoder, Mapping):
        raise TypeError("model.feed_head_conditioning.encoder must be a mapping.")
    cfg = FeedHeadConditioningConfig(
        enabled=bool(raw.get("enabled", False)),
        features=tuple(str(v).strip().upper() for v in raw.get("features", KNOWN_FEED_NAMES)),
        use_scaled_values=bool(raw.get("use_scaled_values", True)),
        use_masks=bool(raw.get("use_masks", True)),
        include_ratios=bool(raw.get("include_ratios", False)),
        input_dim=int(raw.get("input_dim", 6)),
        hidden_dim=int(encoder.get("hidden_dim", 32)),
        output_dim=int(encoder.get("output_dim", 64)),
        dropout=float(encoder.get("dropout", 0.0)),
        broadcast_to_prediction_edges=bool(raw.get("broadcast_to_prediction_edges", True)),
        concat_position=str(raw.get("concat_position", "after_legacy_edge_descriptor")),
    )
    if not cfg.enabled:
        return cfg
    if cfg.features != KNOWN_FEED_NAMES:
        raise ValueError(
            "feed_head_conditioning.features must be exactly [CH4, AIR, WATER] "
            f"in canonical order, got {list(cfg.features)}."
        )
    if not cfg.use_scaled_values or not cfg.use_masks or cfg.include_ratios:
        raise ValueError(
            "R0-FeedHead requires scaled values and masks, and forbids ratios in the direct branch."
        )
    if cfg.input_dim != 2 * len(KNOWN_FEED_NAMES):
        raise ValueError("feed_head_conditioning.input_dim must be 6.")
    if cfg.hidden_dim <= 0 or cfg.output_dim <= 0:
        raise ValueError("feed_head_conditioning encoder dimensions must be positive.")
    if cfg.dropout != 0.0:
        raise ValueError("R0-FeedHead encoder dropout must be 0.0.")
    if not cfg.broadcast_to_prediction_edges:
        raise ValueError("feed_head_conditioning.broadcast_to_prediction_edges must be true.")
    if cfg.concat_position != "after_legacy_edge_descriptor":
        raise ValueError(
            "feed_head_conditioning.concat_position must be 'after_legacy_edge_descriptor'."
        )
    return cfg


def feed_head_config_dict(cfg: FeedHeadConditioningConfig) -> dict[str, Any]:
    return {
        "enabled": cfg.enabled,
        "features": list(cfg.features),
        "use_scaled_values": cfg.use_scaled_values,
        "use_masks": cfg.use_masks,
        "include_ratios": cfg.include_ratios,
        "input_dim": cfg.input_dim,
        "encoder": {
            "hidden_dim": cfg.hidden_dim,
            "output_dim": cfg.output_dim,
            "dropout": cfg.dropout,
        },
        "broadcast_to_prediction_edges": cfg.broadcast_to_prediction_edges,
        "concat_position": cfg.concat_position,
    }
