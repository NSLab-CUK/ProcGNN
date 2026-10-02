"""Registry for method-ablation experiments (one-factor-at-a-time)."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

# Resolved baseline model fields (v3 edge_all + capacity-sized encoder; see README).
BASELINE_MODEL: dict[str, Any] = {
    "use_edge_stream_head": False,
    "use_edge_decoder": True,
    "edge_decoder_hidden_dim": 128,
    "stream_target_dim": 12,
    "edge_struct_dim": 5,
    "hidden_dim": 512,
    "num_layers": 5,
    "attn_hidden_dim": 512,
    "role_emb_dim": 96,
    "unit_emb_dim": 64,
    "use_hx_role_embedding": False,
    "input_mlp_layers": 2,
    "diff_mlp_layers": 2,
    "update_mlp_layers": 2,
    "final_mlp_layers": 2,
    "diff_mode": "concat",
    "fusion_mode": "concat",
    "global_pool": "concat_set2set",
    "use_edge_features": True,
    "initial_residual_mode": "per_layer",
    "initial_residual_alpha": 0.05,
    "layer_residual_mode": "interp",
    "layer_residual_alpha": 0.5,
    "layer_residual_norm": False,
    "use_final_projection": True,
}

# Keys touched only by capacity ablation — never changed in method ablation.
CAPACITY_KEYS = frozenset(
    {
        "hidden_dim",
        "num_layers",
        "attn_hidden_dim",
        "role_emb_dim",
        "unit_emb_dim",
        "use_hx_role_embedding",
        "input_mlp_layers",
        "diff_mlp_layers",
        "update_mlp_layers",
        "final_mlp_layers",
    }
)


def _entry(
    name: str,
    group: str,
    overrides: dict[str, Any],
    *,
    listed_default: bool = True,
    optional: bool = False,
    notes: str = "",
    baseline_equivalent_label: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "ablation_group": group,
        "model_overrides": overrides,
        "listed_default": listed_default,
        "optional": optional,
        "notes": notes,
        "baseline_equivalent_label": baseline_equivalent_label,
    }


# Full catalog (including baseline-equivalent configs for documentation).
METHOD_EXPERIMENT_CATALOG: list[dict[str, Any]] = [
    _entry("exp_B0_baseline", "baseline", {}),
    _entry("exp_R1_layer_residual_none", "layer_residual", {"layer_residual_mode": "none"}),
    _entry(
        "exp_R2_layer_residual_interp",
        "layer_residual",
        {"layer_residual_mode": "interp", "layer_residual_alpha": 0.5, "layer_residual_norm": False},
        listed_default=False,
        baseline_equivalent_label="same as exp_B0_baseline",
    ),
    _entry(
        "exp_R3_layer_residual_interp_norm",
        "layer_residual",
        {"layer_residual_mode": "interp", "layer_residual_alpha": 0.5, "layer_residual_norm": True},
        listed_default=False,
        optional=True,
    ),
    _entry("exp_I0_initial_none", "initial_residual", {"initial_residual_mode": "none"}),
    _entry(
        "exp_I1_initial_per_layer",
        "initial_residual",
        {"initial_residual_mode": "per_layer", "initial_residual_alpha": 0.05},
        baseline_equivalent_label="same as exp_B0_baseline",
    ),
    _entry(
        "exp_I2_initial_final",
        "initial_residual",
        {"initial_residual_mode": "final", "initial_residual_alpha": 0.05},
    ),
    _entry(
        "exp_I3_initial_both",
        "initial_residual",
        {"initial_residual_mode": "both", "initial_residual_alpha": 0.05},
    ),
    _entry(
        "exp_D0_diff_concat",
        "differential",
        {"diff_mode": "concat"},
        listed_default=False,
        baseline_equivalent_label="same as exp_B0_baseline",
        notes="diff coupling comparison, not no-differential",
    ),
    _entry("exp_D1_diff_add", "differential", {"diff_mode": "add"}),
    _entry(
        "exp_EF0_edge_features_on",
        "edge_features",
        {"use_edge_features": True},
        listed_default=False,
        baseline_equivalent_label="same as exp_B0_baseline",
    ),
    _entry(
        "exp_EF1_edge_features_off",
        "edge_features",
        {"use_edge_features": False},
        notes="Blocked at train time: use_edge_decoder=True requires use_edge_features=True in ProcessSurrogateModel.",
    ),
    _entry(
        "exp_F0_fusion_concat",
        "fusion",
        {"fusion_mode": "concat"},
        listed_default=False,
        baseline_equivalent_label="same as exp_B0_baseline",
    ),
    _entry("exp_F1_fusion_sum", "fusion", {"fusion_mode": "sum"}),
    _entry("exp_F2_fusion_mean", "fusion", {"fusion_mode": "mean"}),
    _entry("exp_F3_fusion_weighted", "fusion", {"fusion_mode": "weighted"}, notes="Adds learnable fusion gate parameters."),
    _entry(
        "exp_G0_global_concat_set2set",
        "global_pool",
        {"global_pool": "concat_set2set"},
        listed_default=False,
        baseline_equivalent_label="same as exp_B0_baseline",
    ),
    _entry("exp_G1_global_mean", "global_pool", {"global_pool": "mean"}),
    _entry("exp_G2_global_sum", "global_pool", {"global_pool": "sum"}),
    _entry("exp_G3_global_max", "global_pool", {"global_pool": "max"}),
    _entry("exp_G4_global_attention", "global_pool", {"global_pool": "attention"}),
    _entry(
        "exp_P0_final_projection_on",
        "final_projection",
        {"use_final_projection": True},
        listed_default=False,
        baseline_equivalent_label="same as exp_B0_baseline",
    ),
    _entry(
        "exp_P1_final_projection_off",
        "final_projection",
        {"use_final_projection": False},
        listed_default=False,
        optional=True,
        notes="edge_all EdgeDecoder uses local_node_embeddings; global_embedding still used in decoder.",
    ),
]

DEFAULT_METHOD_EXPERIMENTS: list[str] = [
    e["name"]
    for e in METHOD_EXPERIMENT_CATALOG
    if e.get("listed_default", True) and not e.get("baseline_equivalent_label")
]

OPTIONAL_METHOD_EXPERIMENTS: list[str] = [e["name"] for e in METHOD_EXPERIMENT_CATALOG if e.get("optional")]

CATALOG_BY_NAME: dict[str, dict[str, Any]] = {e["name"]: e for e in METHOD_EXPERIMENT_CATALOG}


def merged_model_overrides(entry: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(BASELINE_MODEL)
    out.update(entry.get("model_overrides") or {})
    return out


def changed_config_keys(entry: dict[str, Any]) -> list[str]:
    ov = entry.get("model_overrides") or {}
    keys = []
    for k, v in ov.items():
        if k in CAPACITY_KEYS:
            continue
        if BASELINE_MODEL.get(k) != v:
            keys.append(f"model.{k}={v!r}")
    return keys


def is_baseline_equivalent(entry: dict[str, Any]) -> bool:
    if entry.get("baseline_equivalent_label"):
        return True
    return merged_model_overrides(entry) == BASELINE_MODEL


def should_run_experiment(name: str, *, include_optional: bool) -> bool:
    entry = CATALOG_BY_NAME[name]
    if entry.get("optional") and not include_optional:
        return False
    if is_baseline_equivalent(entry) and name != "exp_B0_baseline":
        return False
    return True


def resolve_experiment_names(names: list[str] | None, *, include_optional: bool) -> list[str]:
    if names is None:
        raw = list(DEFAULT_METHOD_EXPERIMENTS)
        if include_optional:
            raw.extend(OPTIONAL_METHOD_EXPERIMENTS)
    else:
        raw = list(names)
    out: list[str] = []
    seen: set[str] = set()
    for name in raw:
        if name not in CATALOG_BY_NAME:
            raise ValueError(f"unknown method ablation experiment: {name}")
        if not should_run_experiment(name, include_optional=include_optional):
            print(f"[skip] {name}: baseline-equivalent or not included (see README)")
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    if "exp_B0_baseline" not in out and (names is None or "exp_B0_baseline" in (names or [])):
        out.insert(0, "exp_B0_baseline")
    elif names is None and "exp_B0_baseline" not in out:
        out.insert(0, "exp_B0_baseline")
    return out
