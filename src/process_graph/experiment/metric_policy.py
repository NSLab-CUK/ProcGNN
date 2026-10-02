"""Official metric / loss policy for edge_all (single head, target-feature emphasis)."""

from __future__ import annotations

from typing import Any, Mapping

from .stream_vector_policy import build_stream_vector_policy
from .target_metric_names import (
    AMOUNT_R2_SHORT_ALIAS,
    EVAL_LEGACY_ANSWER_FRAC_R2,
    EVAL_PRIMARY_FRAC_R2_BY_PROCESS,
    EVAL_PRIMARY_FRAC_R2_BY_TARGET,
    EVAL_SECONDARY_AMOUNT_R2_BY_PROCESS,
    FRAC_R2_SHORT_ALIAS,
    pick_from_mapping,
)
from .target_row_primary_metrics import AMOUNT_TARGET_BALANCED_METRIC_NAME

AMOUNT_SECONDARY_METRIC_NAME = EVAL_SECONDARY_AMOUNT_R2_BY_PROCESS

OFFICIAL_METRIC = "target_edge_property_mean_r2"
EARLY_STOP_METRIC = f"val_{OFFICIAL_METRIC}"

TARGET_EDGE_R2_PROPERTY_NAMES: tuple[str, ...] = (
    "Temp",
    "Pres",
    "Mass_Flow",
    "Frac_H2O",
    "Frac_H2",
    "Frac_CH4",
    "Frac_CO2",
    "Frac_CO",
    "Frac_O2",
    "Frac_N2",
)

PI_ALL_EDGE_R2_PROPERTY_NAMES: tuple[str, ...] = (
    *TARGET_EDGE_R2_PROPERTY_NAMES,
)

PI_R2_PROPERTY_NAMES: tuple[str, ...] = PI_ALL_EDGE_R2_PROPERTY_NAMES


def _metric_slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


_COMPACT_EDGE_ALL_METRICS_KEYS: tuple[str, ...] = (
    "epoch",
    "train_lr",
    "had_validation",
    "train_loss_total",
    "train_loss_total_final",
    "train_loss",
    "train_epoch_time_sec",
    "train_samples_per_sec",
    "train_optimizer_steps_per_sec",
    "train_optimizer_step_count",
    "train_optimizer_steps_total",
    "train_optimizer_budget_reached",
    "train_peak_gpu_memory_mb",
    "train_peak_gpu_reserved_mb",
    "train_loss_edge_all",
    "train_edge_step_loss_mean",
    "train_edge_step_pi_loss_mean",
    "train_target_edge_loss_mean",
    "train_non_target_edge_loss_mean",
    "train_grad_norm",
    "train_edge_step_target_edge_count",
    "train_edge_step_default_edge_count",
    "train_target_edge_weight_mean",
    "train_default_edge_weight_mean",
    "train_loss_main_mean",
    "train_loss_rho_mean",
    "train_loss_h_mean",
    "train_loss_target_frac_feature",
    "train_loss_target_row_amount",
    "train_loss_node_mass",
    "train_loss_node_component",
    "train_loss_node_atom",
    "train_weighted_node_mass",
    "train_weighted_node_component",
    "train_weighted_node_atom",
    "train_node_mass_valid_count",
    "train_node_component_valid_count",
    "train_node_atom_valid_count",
    "train_node_mass_residual_mean",
    "train_node_component_residual_mean",
    "train_node_atom_residual_mean",
    "train_node_mass_residual_max",
    "train_node_component_residual_max",
    "train_node_atom_residual_max",
    "val_loss_total",
    "val_loss_total_final",
    "val_loss",
    "val_loss_edge_all",
    "val_loss_target_frac_feature",
    "val_loss_target_row_amount",
    "val_loss_node_mass",
    "val_loss_node_component",
    "val_loss_node_atom",
    "val_node_mass_valid_count",
    "val_node_component_valid_count",
    "val_node_atom_valid_count",
    "val_node_mass_residual_mean",
    "val_node_component_residual_mean",
    "val_node_atom_residual_mean",
    "val_node_mass_residual_max",
    "val_node_component_residual_max",
    "val_node_atom_residual_max",
    "val_target_r2",
    "val_target_mean_r2",
    "val_target_edge_property_mean_r2",
    "val_target_edge_10d_mae_property_macro",
    "val_target_edge_10d_rmse_property_macro",
    "val_target_edge_10d_r2_flatten",
    "val_target_edge_10d_r2_edge_macro",
    "val_target_edge_10d_num_r2_unstable",
    "val_target_edge_10d_num_edges",
    "val_target_edge_10d_num_properties",
    "val_target_edge_10d_num_valid_values",
    "val_pi_all_edge_property_mean_r2",
    f"val_{EVAL_PRIMARY_FRAC_R2_BY_PROCESS}",
    "legacy_metric_target_r2",
    "diag_edge_all_mae",
)


def _compact_edge_all_metrics_per_epoch_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Keep metrics_per_epoch focused on key losses and reliable R2 signals."""
    keep = set(_COMPACT_EDGE_ALL_METRICS_KEYS)
    for name in TARGET_EDGE_R2_PROPERTY_NAMES:
        keep.add(f"val_target_edge_r2_{_metric_slug(name)}")
    for name in PI_ALL_EDGE_R2_PROPERTY_NAMES:
        keep.add(f"val_pi_all_edge_r2_{_metric_slug(name)}")
    return {key: row[key] for key in row if key in keep}


ALIAS_POLICY: dict[str, dict[str, Any]] = {
    "val_r2": {
        "resolves_to": EARLY_STOP_METRIC,
        "deprecated": True,
        "note": "edge_all: use val_target_edge_property_mean_r2",
    },
    "loss_v4_targets": {
        "resolves_to": "loss_target_row_amount",
        "deprecated": True,
        "note": "amount auxiliary only; not included in default loss_total",
    },
    "loss_target_row_frac": {
        "resolves_to": "loss_target_frac_feature",
        "deprecated": True,
    },
    "loss_v4_targets_frac": {
        "resolves_to": "loss_target_frac_feature",
        "deprecated": True,
    },
    FRAC_R2_SHORT_ALIAS: {
        "resolves_to": EVAL_PRIMARY_FRAC_R2_BY_PROCESS,
        "deprecated": True,
    },
}


def _bool(train_cfg: Any, name: str, default: bool) -> bool:
    return bool(getattr(train_cfg, name, default))


def _float(train_cfg: Any, name: str, default: float) -> float:
    return float(getattr(train_cfg, name, default))


def resolve_loss_lambdas(train_cfg: Any) -> tuple[bool, bool, bool, float, float]:
    """Return use_edge_all, use_target_feature, use_amount, lambda_feature, lambda_amount."""
    use_edge = _bool(train_cfg, "use_edge_all_loss", True)
    scope = str(getattr(train_cfg, "target_feature_loss_scope", "target_frac_feature")).strip().lower()
    use_feat = _bool(
        train_cfg,
        "use_target_feature_loss",
        _bool(train_cfg, "use_target_row_frac_loss", True),
    )
    if scope in {"none", "off", "disabled", "target_stream_edge_weighted_edge_all"}:
        use_feat = False
    use_amount = _bool(train_cfg, "use_target_row_amount_loss", False)
    if _bool(train_cfg, "use_v4_target_row_loss", False) and not use_amount:
        use_amount = True
    lam_feat = _float(
        train_cfg,
        "primary_frac_loss_weight",
        _float(
            train_cfg,
            "lambda_target_feature",
            _float(train_cfg, "lambda_target_row_frac", 1.0),
        ),
    )
    _legacy_lam_feat = _float(
        train_cfg,
        "lambda_target_feature",
        _float(train_cfg, "lambda_target_row_frac", 1.0),
    )
    lam_amount = _float(train_cfg, "lambda_target_row_amount", 0.0)
    if lam_amount <= 0.0:
        use_amount = False
    if lam_feat <= 0.0:
        use_feat = False
    return use_edge, use_feat, use_amount, lam_feat, lam_amount


def build_loss_policy(train_cfg: Any) -> dict[str, Any]:
    from .target_row_loss import resolve_all_edge_r2_config

    use_edge, use_feat, use_amount_requested, lam_feat, lam_amount = resolve_loss_lambdas(train_cfg)
    r2_cfg = resolve_all_edge_r2_config(train_cfg)
    space = str(getattr(train_cfg, "target_feature_loss_space", "normalized"))
    balancing = str(getattr(train_cfg, "target_feature_loss_balancing", "process_target_balanced"))
    legacy_lam_feat = _float(train_cfg, "lambda_target_feature", _float(train_cfg, "lambda_target_row_frac", 1.0))
    return {
        "prediction_head": "single_edge_all_head",
        "target_feature_loss_source": "shared_pred_edge_tensor",
        "base_loss": "loss_edge_all_masked",
        "target_feature_loss": str(getattr(train_cfg, "target_feature_loss_scope", "target_frac_feature")),
        "target_feature_source": "data/reference/v4/target_stream_targets.csv",
        "target_feature_loss_space": space,
        "target_feature_loss_scale_checked": True,
        "target_feature_loss_balancing": balancing,
        "amount_loss_enabled": False,
        "amount_loss_requested": use_amount_requested,
        "lambda_target_feature": lam_feat,
        "primary_frac_loss_weight": _float(train_cfg, "primary_frac_loss_weight", lam_feat),
        "all_edge_r2_loss_weight": float(r2_cfg["loss_weight"]),
        "all_edge_r2_min_count": int(r2_cfg["min_count"]),
        "all_edge_r2_sst_threshold": float(r2_cfg["sst_threshold"]),
        "all_edge_r2_loss_cap": float(r2_cfg["loss_cap"]),
        "all_edge_r2_config_source": str(r2_cfg["source"]),
        "primary_frac_r2_loss_weight": _float(train_cfg, "primary_frac_r2_loss_weight", 0.0),
        "legacy_lambda_target_feature": legacy_lam_feat,
        "lambda_target_row_amount": lam_amount,
        "lambda_target_row_frac": lam_feat,
        "use_edge_all_loss": use_edge,
        "use_target_feature_loss": use_feat,
        "r2_loss_scope": "edge_all",
        "formula": "weighted_edge_all + all_edge_r2_loss_weight*loss_all_edge_r2",
        "amount_auxiliary": "diagnostic_only",
        "use_target_stream_loss_weighting": _bool(train_cfg, "use_target_stream_loss_weighting", False),
        "target_stream_loss_weight": _float(train_cfg, "target_stream_loss_weight", 5.0),
        "target_stream_weighting_mode": str(getattr(train_cfg, "target_stream_weighting_mode", "unique_target_stream_edge_by_process")),
        "target_stream_weight_conflict_policy": str(getattr(train_cfg, "target_stream_weight_conflict_policy", "max")),
        "allow_yaml_only_target_streams": _bool(train_cfg, "allow_yaml_only_target_streams", False),
        "loss_primary_frac_deprecated": True,
        "loss_primary_frac_contributes_to_total": use_feat,
        "target_incident_exclude_virtual_nodes": _bool(train_cfg, "target_incident_exclude_virtual_nodes", True),
        "target_incident_include_target_edge": _bool(train_cfg, "target_incident_include_target_edge", True),
        "target_incident_use_answer_edge_weight": _bool(train_cfg, "target_incident_use_answer_edge_weight", False),
    }


def build_metric_policy(train_cfg: Any) -> dict[str, Any]:
    monitor = str(getattr(train_cfg, "monitor_metric", EARLY_STOP_METRIC))
    mode = str(getattr(train_cfg, "monitor_mode", "max"))
    return {
        "official_metric": OFFICIAL_METRIC,
        "official_metric_description": (
            "arithmetic mean of the 10 pooled Target Property R2 values"
        ),
        "early_stop_metric": monitor if monitor.startswith("val_") else EARLY_STOP_METRIC,
        "early_stop_resolved": monitor if monitor.startswith("val_") else EARLY_STOP_METRIC,
        "primary_metric_family": "target_stream_feature_prediction",
        "secondary_metric_family": "derived_amount_diagnostic",
        "legacy_metric_family": "answer_edge_slot_frac",
        "monitor_mode": mode,
    }


def print_metric_loss_policy(
    train_cfg: Any,
    *,
    verbosity: str = "verbose",
    stream_target_dim: int | None = None,
) -> None:
    mp = build_metric_policy(train_cfg)
    lp = build_loss_policy(train_cfg)
    sv = build_stream_vector_policy()
    if stream_target_dim is not None:
        sv = dict(sv)
        sv["stream_target_dim"] = int(stream_target_dim)
    if str(verbosity or "verbose").strip().lower() != "verbose":
        print(
            "[MetricPolicy] "
            f"official={mp['official_metric']} early_stop={mp['early_stop_resolved']} "
            f"stream_dim={sv['stream_target_dim']} "
            f"target_weighting={lp['use_target_stream_loss_weighting']} "
            f"target_weight={lp['target_stream_loss_weight']} "
            f"r2_weight={lp['all_edge_r2_loss_weight']}",
            flush=True,
        )
        return
    print(f"[MetricPolicy] official_metric={mp['official_metric']}", flush=True)
    print(f"[MetricPolicy] early_stop_metric={mp['early_stop_resolved']}", flush=True)
    print("[HeadPolicy] prediction_head=edge_all_edge_decoder (see [model][edge_head] for single/grouped_property)", flush=True)
    print(
        f"[StreamVectorPolicy] stream_target_dim={sv['stream_target_dim']}, "
        f"features={sv['stream_feature_names']}",
        flush=True,
    )
    print(
        "[LossPolicy] loss_total=weighted_edge_all + all_edge_r2_loss_weight*loss_all_edge_r2 "
        f"(target_stream_weighting={lp['use_target_stream_loss_weighting']}, "
        f"target_stream_loss_weight={lp['target_stream_loss_weight']}, "
        f"lambda_r2={lp['all_edge_r2_loss_weight']}, "
        f"r2_loss_scope={lp['r2_loss_scope']}, r2_config_source={lp['all_edge_r2_config_source']})",
        flush=True,
    )
    print(
        "[LossPolicy] target_stream_edge_weighting="
        f"mode={lp['target_stream_weighting_mode']} "
        f"conflict_policy={lp['target_stream_weight_conflict_policy']} "
        f"allow_yaml_only={lp['allow_yaml_only_target_streams']} "
        f"loss_primary_frac_contributes_to_total={lp['loss_primary_frac_contributes_to_total']}",
        flush=True,
    )
    print(
        f"[LossPolicy] amount_auxiliary={lp['amount_auxiliary']}"
        f" requested={lp['amount_loss_requested']} lambda_amount={lp['lambda_target_row_amount']}",
        flush=True,
    )


def build_edge_all_metrics_per_epoch_row(
    *,
    epoch: int,
    train_loss_total: float,
    lr: float,
    train_components: Mapping[str, float],
    val_metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Canonical metrics_per_epoch.csv row for edge_all."""
    from .target_row_primary_metrics import write_edge_all_epoch_metric_columns

    def _lc(keys: tuple[str, ...], default: float = float("nan")) -> float:
        for k in keys:
            if k in train_components:
                return float(train_components[k])
        return default

    row: dict[str, Any] = {
        "epoch": int(epoch),
        "train_lr": float(lr),
        "had_validation": 1 if val_metrics is not None else 0,
        "train_loss_total": float(train_loss_total),
        "train_loss_total_final": _lc(("loss_total_final", "loss_total"), float(train_loss_total)),
        "train_loss": float(train_loss_total),
        "train_epoch_time_sec": _lc(("epoch_time_sec",)),
        "train_samples_per_sec": _lc(("samples_per_sec",)),
        "train_optimizer_steps_per_sec": _lc(("optimizer_steps_per_sec",)),
        "train_peak_gpu_memory_mb": _lc(("peak_gpu_memory_mb",), 0.0),
        "train_peak_gpu_reserved_mb": _lc(("peak_gpu_reserved_mb",), 0.0),
        "train_loss_edge_all": _lc(("loss_edge_all", "loss_edge")),
        "train_loss_primary_frac": _lc(("loss_primary_frac", "loss_target_frac_feature", "loss_target_row_frac")),
        "train_loss_target_incident_edges": _lc(("loss_target_incident_edges", "loss_primary_frac", "loss_target_frac_feature")),
        "train_loss_all_edge_r2": _lc(("loss_all_edge_r2", "r2_loss_all", "loss_primary_frac_r2"), 0.0),
        "train_weighted_edge_all": _lc(("weighted_edge_all", "loss_edge_all", "loss_edge"), 0.0),
        "train_weighted_primary_frac": _lc(("weighted_primary_frac",), 0.0),
        "train_weighted_target_incident_edges": _lc(("weighted_target_incident_edges", "weighted_primary_frac"), 0.0),
        "train_weighted_all_edge_r2": _lc(("weighted_all_edge_r2",), 0.0),
        "train_target_stream_loss_weighting_enabled": _lc(("target_stream_loss_weighting_enabled",), 0.0),
        "train_num_target_stream_edges_weighted": _lc(("num_target_stream_edges_weighted",), 0.0),
        "train_num_target_stream_edges_with_yaml_override": _lc(("num_target_stream_edges_with_yaml_override",), 0.0),
        "train_num_target_stream_rows_skipped": _lc(("num_target_stream_rows_skipped", "target_stream_edges_skipped"), 0.0),
        "train_loss_primary_frac_deprecated": _lc(("loss_primary_frac_deprecated",), 1.0),
        "train_loss_primary_frac_contributes_to_total": _lc(("loss_primary_frac_contributes_to_total",), 0.0),
        "train_amount_metrics_are_primary": _lc(("amount_metrics_are_primary",), 0.0),
        "all_edge_r2_loss_weight": _lc(("all_edge_r2_loss_weight",), 0.0),
        "all_edge_r2_min_count": _lc(("all_edge_r2_min_count",), 0.0),
        "all_edge_r2_sst_threshold": _lc(("all_edge_r2_sst_threshold",), 0.0),
        "all_edge_r2_loss_cap": _lc(("all_edge_r2_loss_cap",), 0.0),
        "all_edge_r2_config_source": str(train_components.get("all_edge_r2_config_source", "unknown")),
        "train_loss_amount_auxiliary": _lc(("loss_amount_auxiliary", "loss_target_row_amount"), 0.0),
        "train_amount_auxiliary_enabled": _lc(("amount_auxiliary_enabled",), 0.0),
        "train_amount_auxiliary_contributes_to_total": _lc(("amount_auxiliary_contributes_to_total",), 0.0),
        "train_weighted_amount_auxiliary_disabled": _lc(("weighted_amount_auxiliary_disabled",), 0.0),
        "train_loss_extra_unexpected": _lc(("loss_extra_unexpected",), 0.0),
        "train_loss_primary_frac_r2": _lc(("loss_all_edge_r2", "r2_loss_all", "loss_primary_frac_r2"), 0.0),
        "r2_loss_scope": "edge_all",
        "train_r2_loss_num_elements": _lc(("r2_loss_num_elements",), 0.0),
        "train_target_loss_num_elements": _lc(("target_loss_num_elements", "primary_frac_count"), 0.0),
        "train_edge_all_loss_num_elements": _lc(("edge_all_loss_num_elements",), 0.0),
        "train_r2_loss_valid_feature_count": _lc(("r2_loss_valid_feature_count", "primary_frac_r2_valid_feature_count"), 0.0),
        "train_primary_frac_count": _lc(("primary_frac_count",), 0.0),
        "train_target_incident_edge_count": _lc(("target_incident_edge_count",), 0.0),
        "train_target_incident_supervised_cell_count": _lc(("target_incident_supervised_cell_count",), 0.0),
        "train_target_incident_target_row_count": _lc(("target_incident_target_row_count",), 0.0),
        "train_target_incident_matched_edge_count": _lc(("target_incident_matched_edge_count",), 0.0),
        "train_target_incident_virtual_endpoint_excluded_count": _lc(("target_incident_virtual_endpoint_excluded_count",), 0.0),
        "train_target_incident_effective_weight_mean": _lc(("target_incident_effective_weight_mean",), 0.0),
        "train_target_incident_coverage_edge_frac": _lc(("target_incident_coverage_edge_frac",), 0.0),
        "train_target_incident_loss_ratio_total": _lc(("target_incident_loss_ratio_total",), 0.0),
        "train_target_incident_loss_ratio_edge": _lc(("target_incident_loss_ratio_edge",), 0.0),
        "train_primary_frac_r2_valid_feature_count": _lc(("primary_frac_r2_valid_feature_count",), 0.0),
        "train_primary_frac_pred_std_norm": _lc(("primary_frac_pred_std_norm",), 0.0),
        "train_primary_frac_true_std_norm": _lc(("primary_frac_true_std_norm",), 0.0),
        "train_primary_frac_std_ratio_norm": _lc(("primary_frac_std_ratio_norm",), 0.0),
        "train_primary_frac_pred_std_orig": _lc(("primary_frac_pred_std_orig",), 0.0),
        "train_primary_frac_true_std_orig": _lc(("primary_frac_true_std_orig",), 0.0),
        "train_primary_frac_std_ratio_orig": _lc(("primary_frac_std_ratio_orig",), 0.0),
        "train_loss_target_frac_feature": _lc(
            ("loss_target_frac_feature", "loss_target_row_frac", "loss_v4_targets_frac")
        ),
        "train_loss_target_row_amount": _lc(
            ("loss_target_row_amount", "loss_v4_targets_amount", "loss_v4_targets"), 0.0
        ),
        "train_loss_node_mass": _lc(("loss_node_mass",)),
        "train_loss_node_component": _lc(("loss_node_component",)),
        "train_loss_node_atom": _lc(("loss_node_atom",)),
        "train_weighted_node_mass": _lc(("weighted_node_mass",)),
        "train_weighted_node_component": _lc(("weighted_node_component",)),
        "train_weighted_node_atom": _lc(("weighted_node_atom",)),
        "train_node_mass_valid_count": _lc(("node_mass_valid_count",), 0.0),
        "train_node_component_valid_count": _lc(("node_component_valid_count",), 0.0),
        "train_node_atom_valid_count": _lc(("node_atom_valid_count",), 0.0),
        "train_node_mass_residual_mean": _lc(("node_mass_residual_mean",)),
        "train_node_component_residual_mean": _lc(("node_component_residual_mean",)),
        "train_node_atom_residual_mean": _lc(("node_atom_residual_mean",)),
        "train_node_mass_residual_max": _lc(("node_mass_residual_max",)),
        "train_node_component_residual_max": _lc(("node_component_residual_max",)),
        "train_node_atom_residual_max": _lc(("node_atom_residual_max",)),
        "train_edge_step_loss_mean": _lc(("edge_step_loss_mean",), float("nan")),
        "train_edge_step_pi_loss_mean": _lc(("edge_step_pi_loss_mean",), float("nan")),
        "train_target_edge_loss_mean": _lc(("target_edge_loss_mean",), float("nan")),
        "train_non_target_edge_loss_mean": _lc(("non_target_edge_loss_mean",), float("nan")),
        "train_edge_update_count": _lc(("edge_update_count",), 0.0),
        "train_optimizer_step_count": _lc(("optimizer_step_count",), 0.0),
        "train_optimizer_steps_total": _lc(("optimizer_steps_total",), 0.0),
        "train_optimizer_budget_reached": _lc(("optimizer_budget_reached",), 0.0),
        "train_skipped_edge_count": _lc(("skipped_edge_count",), 0.0),
        "train_num_edges_per_batch": _lc(("num_edges_per_batch",), 0.0),
        "train_edge_step_forward_count": _lc(("edge_step_forward_count",), 0.0),
        "train_edge_step_pi_forward_count": _lc(("edge_step_pi_forward_count",), 0.0),
        "train_grad_norm": _lc(("grad_norm",), float("nan")),
        "train_edge_step_target_edge_count": _lc(("edge_step_target_edge_count",), 0.0),
        "train_edge_step_default_edge_count": _lc(("edge_step_default_edge_count",), 0.0),
        "train_target_edge_weight_mean": _lc(("target_edge_weight_mean",), 0.0),
        "train_default_edge_weight_mean": _lc(("default_edge_weight_mean",), 0.0),
        "train_edge_pred_output_dim": _lc(("edge_pred_output_dim",), 0.0),
        "train_loss_main_mean": _lc(("loss_main_mean",), 0.0),
        "train_loss_rho_mean": _lc(("loss_rho_mean",), 0.0),
        "train_loss_h_mean": _lc(("loss_h_mean",), 0.0),
        "train_loss_volume_mean": _lc(("loss_volume_mean",), 0.0),
        "train_loss_enthalpy_flow_mean": _lc(("loss_enthalpy_flow_mean",), 0.0),
        "train_loss_atom_mean": _lc(("loss_atom_mean",), 0.0),
        "train_loss_energy_mean": _lc(("loss_energy_mean",), 0.0),
        "train_loss_rho_max": _lc(("loss_rho_max",), 0.0),
        "train_loss_h_max": _lc(("loss_h_max",), 0.0),
        "train_loss_volume_max": _lc(("loss_volume_max",), 0.0),
        "train_loss_enthalpy_flow_max": _lc(("loss_enthalpy_flow_max",), 0.0),
        "train_edge_weight_min": _lc(("edge_weight_min",), 0.0),
        "train_edge_weight_max": _lc(("edge_weight_max",), 0.0),
        "train_edge_weight_mean": _lc(("edge_weight_mean",), 0.0),
    }
    if val_metrics is None:
        return _compact_edge_all_metrics_per_epoch_row(row)

    def _vc(keys: tuple[str, ...], default: float = float("nan")) -> float:
        for k in keys:
            if k in val_metrics:
                try:
                    return float(val_metrics[k])
                except (TypeError, ValueError):
                    pass
        return default

    row["val_loss_total"] = _vc(("loss_total",))
    row["val_loss_total_final"] = _vc(("loss_total_final", "loss_total"))
    row["val_loss"] = row["val_loss_total"]
    row["val_loss_edge_all"] = _vc(("loss_edge_all", "loss_edge"))
    row["val_loss_primary_frac"] = _vc(("loss_primary_frac", "loss_target_frac_feature", "loss_target_row_frac"))
    row["val_loss_target_incident_edges"] = _vc(("loss_target_incident_edges", "loss_primary_frac", "loss_target_frac_feature"))
    row["val_loss_all_edge_r2"] = _vc(("loss_all_edge_r2", "r2_loss_all", "loss_primary_frac_r2"), 0.0)
    row["val_weighted_edge_all"] = _vc(("weighted_edge_all", "loss_edge_all", "loss_edge"), 0.0)
    row["val_weighted_primary_frac"] = _vc(("weighted_primary_frac",), 0.0)
    row["val_weighted_target_incident_edges"] = _vc(("weighted_target_incident_edges", "weighted_primary_frac"), 0.0)
    row["val_weighted_all_edge_r2"] = _vc(("weighted_all_edge_r2",), 0.0)
    row["val_target_stream_loss_weighting_enabled"] = _vc(("target_stream_loss_weighting_enabled",), 0.0)
    row["val_num_target_stream_edges_weighted"] = _vc(("num_target_stream_edges_weighted",), 0.0)
    row["val_num_target_stream_edges_with_yaml_override"] = _vc(("num_target_stream_edges_with_yaml_override",), 0.0)
    row["val_num_target_stream_rows_skipped"] = _vc(("num_target_stream_rows_skipped", "target_stream_edges_skipped"), 0.0)
    row["val_loss_primary_frac_deprecated"] = _vc(("loss_primary_frac_deprecated",), 1.0)
    row["val_loss_primary_frac_contributes_to_total"] = _vc(("loss_primary_frac_contributes_to_total",), 0.0)
    row["val_amount_metrics_are_primary"] = _vc(("amount_metrics_are_primary",), 0.0)
    row["val_target_r2"] = _vc(("target_r2", "val_target_r2"), float("nan"))
    row["val_target_mean_r2"] = _vc(("target_mean_r2", "val_target_mean_r2"), float("nan"))
    row["val_target_edge_property_mean_r2"] = _vc(
        ("target_edge_property_mean_r2", "val_target_edge_property_mean_r2"),
        float("nan"),
    )
    row["val_target_edge_10d_mae_property_macro"] = _vc(("target_edge_10d_mae_property_macro", "val_target_edge_10d_mae_property_macro"), float("nan"))
    row["val_target_edge_10d_rmse_property_macro"] = _vc(("target_edge_10d_rmse_property_macro", "val_target_edge_10d_rmse_property_macro"), float("nan"))
    row["val_target_edge_10d_r2_flatten"] = _vc(("target_edge_10d_r2_flatten", "val_target_edge_10d_r2_flatten"), float("nan"))
    row["val_target_edge_10d_r2_edge_macro"] = _vc(("target_edge_10d_r2_edge_macro", "val_target_edge_10d_r2_edge_macro"), float("nan"))
    row["val_target_edge_10d_num_r2_unstable"] = _vc(("target_edge_10d_num_r2_unstable", "val_target_edge_10d_num_r2_unstable"), 0.0)
    row["val_target_edge_10d_num_edges"] = _vc(("target_edge_10d_num_edges", "val_target_edge_10d_num_edges"), 0.0)
    row["val_target_edge_10d_num_properties"] = _vc(("target_edge_10d_num_properties", "val_target_edge_10d_num_properties"), 0.0)
    row["val_target_edge_10d_num_valid_values"] = _vc(("target_edge_10d_num_valid_values", "val_target_edge_10d_num_valid_values"), 0.0)
    row["val_pi_all_edge_property_mean_r2"] = _vc(("pi_all_edge_property_mean_r2",), float("nan"))
    for property_name in PI_ALL_EDGE_R2_PROPERTY_NAMES:
        slug = _metric_slug(property_name)
        metric_key = f"pi_all_edge_r2_{slug}"
        if metric_key in val_metrics:
            row[f"val_{metric_key}"] = _vc((metric_key,), float("nan"))
    for property_name in TARGET_EDGE_R2_PROPERTY_NAMES:
        slug = _metric_slug(property_name)
        metric_keys = (f"val_target_edge_r2_{slug}", f"target_edge_r2_{slug}")
        if any(metric_key in val_metrics for metric_key in metric_keys):
            row[f"val_target_edge_r2_{slug}"] = _vc(metric_keys, float("nan"))
    row["val_all_edge_r2_loss_weight"] = _vc(("all_edge_r2_loss_weight",), 0.0)
    row["val_all_edge_r2_min_count"] = _vc(("all_edge_r2_min_count",), 0.0)
    row["val_all_edge_r2_sst_threshold"] = _vc(("all_edge_r2_sst_threshold",), 0.0)
    row["val_all_edge_r2_loss_cap"] = _vc(("all_edge_r2_loss_cap",), 0.0)
    row["val_loss_amount_auxiliary"] = _vc(("loss_amount_auxiliary", "loss_target_row_amount"), 0.0)
    row["val_amount_auxiliary_enabled"] = _vc(("amount_auxiliary_enabled",), 0.0)
    row["val_amount_auxiliary_contributes_to_total"] = _vc(("amount_auxiliary_contributes_to_total",), 0.0)
    row["val_weighted_amount_auxiliary_disabled"] = _vc(("weighted_amount_auxiliary_disabled",), 0.0)
    row["val_loss_extra_unexpected"] = _vc(("loss_extra_unexpected",), 0.0)
    row["val_loss_primary_frac_r2"] = _vc(("loss_all_edge_r2", "r2_loss_all", "loss_primary_frac_r2"), 0.0)
    row["val_r2_loss_num_elements"] = _vc(("r2_loss_num_elements",), 0.0)
    row["val_target_loss_num_elements"] = _vc(("target_loss_num_elements", "primary_frac_count"), 0.0)
    row["val_edge_all_loss_num_elements"] = _vc(("edge_all_loss_num_elements",), 0.0)
    row["val_r2_loss_valid_feature_count"] = _vc(("r2_loss_valid_feature_count", "primary_frac_r2_valid_feature_count"), 0.0)
    for dk in (
        "primary_frac_count",
        "target_incident_edge_count",
        "target_incident_supervised_cell_count",
        "target_incident_target_row_count",
        "target_incident_matched_edge_count",
        "target_incident_virtual_endpoint_excluded_count",
        "target_incident_effective_weight_mean",
        "target_incident_coverage_edge_frac",
        "target_incident_loss_ratio_total",
        "target_incident_loss_ratio_edge",
        "primary_frac_r2_valid_feature_count",
        "primary_frac_pred_std_norm",
        "primary_frac_true_std_norm",
        "primary_frac_std_ratio_norm",
        "primary_frac_pred_std_orig",
        "primary_frac_true_std_orig",
        "primary_frac_std_ratio_orig",
    ):
        row[f"val_{dk}"] = _vc((dk,), 0.0)
    row["val_loss_target_frac_feature"] = _vc(
        ("loss_target_frac_feature", "loss_target_row_frac", "loss_v4_targets_frac")
    )
    row["val_loss_target_row_amount"] = _vc(
        ("loss_target_row_amount", "loss_v4_targets_amount", "loss_v4_targets"), 0.0
    )
    row["val_loss_node_mass"] = _vc(("loss_node_mass",))
    row["val_loss_node_component"] = _vc(("loss_node_component",))
    row["val_loss_node_atom"] = _vc(("loss_node_atom",))
    row["val_node_mass_valid_count"] = _vc(("node_mass_valid_count",), 0.0)
    row["val_node_component_valid_count"] = _vc(("node_component_valid_count",), 0.0)
    row["val_node_atom_valid_count"] = _vc(("node_atom_valid_count",), 0.0)
    row["val_node_mass_residual_mean"] = _vc(("node_mass_residual_mean",))
    row["val_node_component_residual_mean"] = _vc(("node_component_residual_mean",))
    row["val_node_atom_residual_mean"] = _vc(("node_atom_residual_mean",))
    row["val_node_mass_residual_max"] = _vc(("node_mass_residual_max",))
    row["val_node_component_residual_max"] = _vc(("node_component_residual_max",))
    row["val_node_atom_residual_max"] = _vc(("node_atom_residual_max",))

    write_edge_all_epoch_metric_columns(row, val_metrics)
    for target_r2_alias in (
        "val_target_stream_feature_macro_r2",
        "val_eval_primary_frac_r2_by_target",
    ):
        row.pop(target_r2_alias, None)

    h2 = pick_from_mapping(val_metrics, "metric_target_r2", default=float("nan"))
    c2 = pick_from_mapping(val_metrics, "metric_tailgas_r2", default=float("nan"))
    if h2 == h2:
        row["legacy_metric_target_r2"] = float(h2)
    if c2 == c2:
        row["legacy_metric_tailgas_r2"] = float(c2)
    mae_vals = []
    for k in ("target_h2_mae", "tailgas_co2_mae"):
        v = pick_from_mapping(val_metrics, k, default=float("nan"))
        if v == v:
            mae_vals.append(float(v))
    if mae_vals:
        row["diag_answer_targets_mae"] = sum(mae_vals) / len(mae_vals)
    for dk in ("edge_all_mse", "edge_all_mae"):
        v = pick_from_mapping(val_metrics, dk, default=float("nan"))
        if v == v:
            row[f"diag_{dk}"] = float(v)
    return _compact_edge_all_metrics_per_epoch_row(row)


def pick_official_val_r2(val_metrics: Mapping[str, Any]) -> float:
    return float(
        pick_from_mapping(
            val_metrics,
            EVAL_PRIMARY_FRAC_R2_BY_PROCESS,
            f"val_{EVAL_PRIMARY_FRAC_R2_BY_PROCESS}",
            FRAC_R2_SHORT_ALIAS,
            default=float("nan"),
        )
    )


def enrich_metrics_json(payload: dict[str, Any], train_cfg: Any) -> dict[str, Any]:
    mp = build_metric_policy(train_cfg)
    lp = build_loss_policy(train_cfg)
    sv = build_stream_vector_policy()
    out = dict(payload)
    out["metric_policy"] = mp
    out["loss_policy"] = lp
    out["stream_vector_policy"] = sv
    out["alias_policy"] = ALIAS_POLICY

    def _blk(keys: tuple[str, ...]) -> dict[str, Any]:
        blk: dict[str, Any] = {}
        for k in keys:
            v = pick_from_mapping(out, k, f"val/{k}", f"test/{k}", f"val_{k}", f"test_{k}", default=float("nan"))
            if v == v:
                blk[k] = float(v)
        return blk

    pooled_target_property_keys = tuple(
        f"target_edge_r2_{_metric_slug(name)}" for name in TARGET_EDGE_R2_PROPERTY_NAMES
    )
    out["primary"] = _blk((OFFICIAL_METRIC, *pooled_target_property_keys))
    out["primary_frac_diagnostic"] = _blk(
        (EVAL_PRIMARY_FRAC_R2_BY_PROCESS, EVAL_PRIMARY_FRAC_R2_BY_TARGET)
    )
    out["secondary"] = _blk((EVAL_SECONDARY_AMOUNT_R2_BY_PROCESS, AMOUNT_TARGET_BALANCED_METRIC_NAME))
    out["legacy"] = _blk((EVAL_LEGACY_ANSWER_FRAC_R2,))
    diag_keys = (
        "eval_target_rows_count",
        "eval_target_rows_skipped_count",
        "eval_target_rows_total",
        "included_target_ids",
        "excluded_target_ids",
    )
    out["diagnostics"] = {k: out[k] for k in diag_keys if k in out}
    out["official_metric_name"] = OFFICIAL_METRIC
    return out
