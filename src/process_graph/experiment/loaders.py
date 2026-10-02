from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Mapping, Type, TypeVar

from .schema import (
    DECODER_CATEGORIES,
    AuxTaskConfig,
    DataConfig,
    DecoderTaskConfig,
    EpochBalancedSamplerConfig,
    EpochSamplerMassFlowTailConfig,
    EpochSamplerHardTargetEdgesConfig,
    EpochSamplerMixtureConfig,
    EpochSamplerQuantileBalanceConfig,
    EpochSamplerSparsePositiveConfig,
    ExperimentConfig,
    FixedTaskConfig,
    MassFlowPhysicalAuxiliaryConfig,
    ModelYamlConfig,
    NodePinnOptimizationConfig,
    RarePositiveSamplerConfig,
    RareTargetEdgeAugmentationConfig,
    SampleHybridTargetEdgeStepPIConfig,
    TargetEdgeAugmentationConfig,
    TrainConfig,
)
from .yaml_utils import deep_merge, read_yaml_mapping
from ..constants import STREAM_EDGE_FEATURE_SLOTS
from ..known_feed import known_feed_config_dict, parse_known_feed_condition
from ..hx_pair import hx_pair_config_dict, parse_hx_pair_relation
from ..feed_head import feed_head_config_dict, parse_feed_head_conditioning

T = TypeVar("T")


def project_root_from_here() -> Path:
    """Return repository root assuming `src/process_graph/experiment/loaders.py` layout."""

    return Path(__file__).resolve().parents[3]


def _as_dataclass(cls: Type[T], data: Mapping[str, Any]) -> T:
    allowed = {f.name for f in fields(cls)}
    selected = {k: v for k, v in data.items() if k in allowed}
    return cls(**selected)


def _validate_quantile_list(values: Any, *, name: str, min_len: int = 2) -> list[float]:
    if not isinstance(values, list) or len(values) < min_len:
        raise ValueError(f"{name} must be a list with at least {min_len} values.")
    result = [float(value) for value in values]
    if any((not math.isfinite(value) or value < 0.0 or value > 1.0) for value in result):
        raise ValueError(f"{name} values must be finite probabilities in [0, 1].")
    if result != sorted(result):
        raise ValueError(f"{name} must be sorted in ascending order.")
    return result


def load_model_yaml_config(path: Path, overrides: Mapping[str, Any] | None = None) -> ModelYamlConfig:
    payload = read_yaml_mapping(path)
    if overrides:
        deep_merge(payload, dict(overrides))
    if "use_oper_mask" in payload and "oper_mask_mode" not in payload:
        legacy = bool(payload.get("use_oper_mask"))
        payload["oper_mask_mode"] = "concat" if legacy else "ignore"
        print(
            "[config][deprecated] model.use_oper_mask is deprecated; "
            "use model.oper_mask_mode (concat/ignore/gated). Applied alias conversion."
        )
    if "final_projection" in payload and "use_final_projection" not in payload:
        payload["use_final_projection"] = bool(payload.get("final_projection"))
        print(
            "[config][deprecated] model.final_projection is deprecated; "
            "use model.use_final_projection. Applied alias conversion."
        )
    if "input_residual" in payload and "initial_residual_mode" not in payload:
        payload["initial_residual_mode"] = "final" if bool(payload.get("input_residual")) else "none"
        print(
            "[config][deprecated] model.input_residual is deprecated; "
            "use model.initial_residual_mode (none/final/per_layer/both). Applied alias conversion."
        )
    if payload.get("readout_feature_source") == "h_loc":
        payload["readout_feature_source"] = "local"
        print(
            "[config][deprecated] model.readout_feature_source='h_loc' is deprecated; "
            "use 'local'. Applied alias conversion."
        )
    return _as_dataclass(ModelYamlConfig, payload)


def load_train_config(path: Path, overrides: Mapping[str, Any] | None = None) -> TrainConfig:
    payload = read_yaml_mapping(path)
    if overrides:
        deep_merge(payload, dict(overrides))
    legacy_keys = {"loss_type"}
    if any(k in payload for k in legacy_keys):
        raise ValueError(
            "Deprecated train config keys detected (loss_type). "
            "Use loss_type_target / loss_type_tailgas / loss_type_decoder."
        )
    mass_output_space = str(payload.get("pi_mass_flow_output_space", "raw_z") or "raw_z").strip().lower()
    if mass_output_space not in {"raw_z", "log1p"}:
        raise ValueError(
            "train.pi_mass_flow_output_space must be 'raw_z' or 'log1p', "
            f"got {mass_output_space!r}."
        )
    if mass_output_space == "log1p" and bool(payload.get("use_log1p_mass_flow_loss", False)):
        raise ValueError(
            "train.pi_mass_flow_output_space='log1p' cannot be combined with "
            "use_log1p_mass_flow_loss=true (legacy A2 raw-z path)."
        )
    mass_flow_transform = str(
        payload.get("mass_flow_transform", "log1p") or "log1p"
    ).strip().lower()
    if mass_flow_transform not in {"log1p", "tempered_log", "scaled_log"}:
        raise ValueError(
            "train.mass_flow_transform must be 'log1p', 'tempered_log', or 'scaled_log', "
            f"got {mass_flow_transform!r}."
        )
    mass_flow_log_tau = float(payload.get("mass_flow_log_tau", 1.0))
    if not math.isfinite(mass_flow_log_tau) or mass_flow_log_tau <= 0.0:
        raise ValueError(
            "train.mass_flow_log_tau must be finite and positive, "
            f"got {mass_flow_log_tau}."
        )
    mass_flow_log_scale = float(payload.get("mass_flow_log_scale", 1.0))
    if not math.isfinite(mass_flow_log_scale) or mass_flow_log_scale <= 0.0:
        raise ValueError(
            "train.mass_flow_log_scale must be finite and positive, "
            f"got {mass_flow_log_scale}."
        )
    mass_flow_log_eps = float(payload.get("mass_flow_log_eps", 1.0e-8))
    if not math.isfinite(mass_flow_log_eps) or mass_flow_log_eps <= 0.0:
        raise ValueError(
            "train.mass_flow_log_eps must be finite and positive, "
            f"got {mass_flow_log_eps}."
        )
    mass_log_weight = float(payload.get("mass_flow_log_loss_weight", 1.0))
    if not math.isfinite(mass_log_weight) or mass_log_weight <= 0.0:
        raise ValueError(
            f"train.mass_flow_log_loss_weight must be finite and positive, got {mass_log_weight}."
        )
    fraction_loss_type = str(payload.get("pi_fraction_loss_type", "legacy") or "legacy").strip().lower()
    if fraction_loss_type not in {"legacy", "clr", "log"}:
        raise ValueError(
            "train.pi_fraction_loss_type must be 'legacy', 'clr', or 'log', "
            f"got {fraction_loss_type!r}."
        )
    clr_eps = float(payload.get("pi_fraction_clr_eps", 1.0e-6))
    if not math.isfinite(clr_eps) or clr_eps <= 0.0:
        raise ValueError(f"train.pi_fraction_clr_eps must be finite and positive, got {clr_eps}.")
    clr_weight = float(payload.get("pi_fraction_clr_loss_weight", 1.0))
    if not math.isfinite(clr_weight) or clr_weight < 0.0:
        raise ValueError(
            f"train.pi_fraction_clr_loss_weight must be finite and non-negative, got {clr_weight}."
        )
    log_eps = float(payload.get("pi_fraction_log_eps", 1.0e-6))
    if not math.isfinite(log_eps) or log_eps <= 0.0:
        raise ValueError(f"train.pi_fraction_log_eps must be finite and positive, got {log_eps}.")
    log_weight = float(payload.get("pi_fraction_log_loss_weight", 1.0))
    if not math.isfinite(log_weight) or log_weight <= 0.0:
        raise ValueError(
            f"train.pi_fraction_log_loss_weight must be finite and positive, got {log_weight}."
        )
    hybrid_raw = payload.get("sample_hybrid_target_edge_step_pi")
    if hybrid_raw is not None:
        if not isinstance(hybrid_raw, Mapping):
            raise TypeError("train.sample_hybrid_target_edge_step_pi must be a mapping.")
        hybrid_cfg = _as_dataclass(SampleHybridTargetEdgeStepPIConfig, hybrid_raw)
        reduction = str(hybrid_cfg.non_target_reduction).strip().lower()
        if reduction != "edge_group_macro_mean":
            raise ValueError(
                "train.sample_hybrid_target_edge_step_pi.non_target_reduction "
                "currently supports only 'edge_group_macro_mean'."
            )
        order = str(hybrid_cfg.target_update_order).strip().lower()
        if order not in {"canonical_edge_id", "canonical_edge_index"}:
            raise ValueError(
                "train.sample_hybrid_target_edge_step_pi.target_update_order must be "
                "'canonical_edge_id' or 'canonical_edge_index'."
            )
        hybrid_cfg.non_target_reduction = reduction
        hybrid_cfg.target_update_order = order
        payload["sample_hybrid_target_edge_step_pi"] = hybrid_cfg
    epoch_sampler_raw = payload.get("epoch_sampler")
    if epoch_sampler_raw is not None:
        if not isinstance(epoch_sampler_raw, Mapping):
            raise TypeError("train.epoch_sampler must be a mapping.")
        sampler_payload = dict(epoch_sampler_raw)
        mixture_raw = sampler_payload.get("mixture", {})
        sparse_raw = sampler_payload.get("sparse_positive", {})
        quantile_raw = sampler_payload.get("quantile_balance", {})
        tail_raw = sampler_payload.get("mass_flow_tail", {})
        hard_edges_raw = sampler_payload.get("hard_target_edges", {})
        if mixture_raw is not None and not isinstance(mixture_raw, Mapping):
            raise TypeError("train.epoch_sampler.mixture must be a mapping.")
        if sparse_raw is not None and not isinstance(sparse_raw, Mapping):
            raise TypeError("train.epoch_sampler.sparse_positive must be a mapping.")
        if quantile_raw is not None and not isinstance(quantile_raw, Mapping):
            raise TypeError("train.epoch_sampler.quantile_balance must be a mapping.")
        if tail_raw is not None and not isinstance(tail_raw, Mapping):
            raise TypeError("train.epoch_sampler.mass_flow_tail must be a mapping.")
        if hard_edges_raw is not None and not isinstance(hard_edges_raw, Mapping):
            raise TypeError("train.epoch_sampler.hard_target_edges must be a mapping.")
        mixture_cfg = _as_dataclass(EpochSamplerMixtureConfig, dict(mixture_raw or {}))
        mixture_values = {
            "uniform": float(mixture_cfg.uniform),
            "sparse_positive": float(mixture_cfg.sparse_positive),
            "quantile_balance": float(mixture_cfg.quantile_balance),
            "mass_flow_tail": float(mixture_cfg.mass_flow_tail),
            "hard_target_edges": float(mixture_cfg.hard_target_edges),
        }
        if any((not math.isfinite(value) or value < 0.0) for value in mixture_values.values()):
            raise ValueError("train.epoch_sampler.mixture values must be finite and non-negative.")
        if abs(sum(mixture_values.values()) - 1.0) > 1.0e-6:
            raise ValueError("train.epoch_sampler.mixture values must sum to 1.0.")
        sparse_cfg = _as_dataclass(EpochSamplerSparsePositiveConfig, dict(sparse_raw or {}))
        sparse_cfg.properties = [str(name) for name in sparse_cfg.properties]
        if not sparse_cfg.properties:
            raise ValueError("train.epoch_sampler.sparse_positive.properties must not be empty.")
        sparse_cfg.positive_quantiles = _validate_quantile_list(
            sparse_cfg.positive_quantiles,
            name="train.epoch_sampler.sparse_positive.positive_quantiles",
        )
        if not math.isfinite(float(sparse_cfg.zero_threshold)) or float(sparse_cfg.zero_threshold) < 0.0:
            raise ValueError("train.epoch_sampler.sparse_positive.zero_threshold must be non-negative.")
        quantile_cfg = _as_dataclass(EpochSamplerQuantileBalanceConfig, dict(quantile_raw or {}))
        quantile_cfg.properties = [str(name) for name in quantile_cfg.properties]
        if not quantile_cfg.properties:
            raise ValueError("train.epoch_sampler.quantile_balance.properties must not be empty.")
        quantile_cfg.quantiles = _validate_quantile_list(
            quantile_cfg.quantiles,
            name="train.epoch_sampler.quantile_balance.quantiles",
        )
        tail_cfg = _as_dataclass(EpochSamplerMassFlowTailConfig, dict(tail_raw or {}))
        tail_cfg.quantiles = _validate_quantile_list(
            tail_cfg.quantiles,
            name="train.epoch_sampler.mass_flow_tail.quantiles",
        )
        tail_cfg.quantile_scope = str(tail_cfg.quantile_scope).strip().lower()
        if tail_cfg.quantile_scope not in {"process", "global"}:
            raise ValueError("train.epoch_sampler.mass_flow_tail.quantile_scope must be 'process' or 'global'.")
        hard_edges_cfg = _as_dataclass(EpochSamplerHardTargetEdgesConfig, dict(hard_edges_raw or {}))
        if hard_edges_cfg.enabled:
            if not isinstance(hard_edges_cfg.rules, list) or not hard_edges_cfg.rules:
                raise ValueError("train.epoch_sampler.hard_target_edges.rules must be a non-empty list when enabled.")
            validated_rules: list[dict[str, Any]] = []
            for idx, raw_rule in enumerate(hard_edges_cfg.rules):
                if not isinstance(raw_rule, Mapping):
                    raise TypeError(f"train.epoch_sampler.hard_target_edges.rules[{idx}] must be a mapping.")
                rule = dict(raw_rule)
                source = str(rule.get("source", "target_edge_property")).strip().lower()
                if source not in {"target_edge_property", "main_row_ratio"}:
                    raise ValueError(
                        f"train.epoch_sampler.hard_target_edges.rules[{idx}].source "
                        "must be 'target_edge_property' or 'main_row_ratio'."
                    )
                prop = str(rule.get("property", "")).strip()
                edge_ids = [str(edge).strip() for edge in rule.get("edge_ids", []) if str(edge).strip()]
                if not prop:
                    raise ValueError(f"train.epoch_sampler.hard_target_edges.rules[{idx}].property is required.")
                if source == "target_edge_property" and not edge_ids:
                    raise ValueError(f"train.epoch_sampler.hard_target_edges.rules[{idx}].edge_ids must not be empty.")
                min_value = float(rule.get("min_value", 0.0))
                max_value = float(rule.get("max_value", float("inf")))
                if not (math.isfinite(min_value) and (math.isfinite(max_value) or max_value == float("inf"))):
                    raise ValueError(
                        f"train.epoch_sampler.hard_target_edges.rules[{idx}] min/max must be finite numbers."
                    )
                if max_value < min_value:
                    raise ValueError(
                        f"train.epoch_sampler.hard_target_edges.rules[{idx}].max_value must be >= min_value."
                    )
                rule["property"] = prop
                rule["source"] = source
                rule["edge_ids"] = edge_ids
                rule["min_value"] = min_value
                rule["max_value"] = max_value
                rule["name"] = str(rule.get("name", f"{prop}_hard_edges")).strip() or f"{prop}_hard_edges"
                if source == "main_row_ratio":
                    process_ids = [int(value) for value in rule.get("process_ids", [])]
                    numerator_column = str(rule.get("numerator_column", "")).strip()
                    denominator_column = str(rule.get("denominator_column", "")).strip()
                    transform = str(rule.get("transform", "raw_ratio")).strip().lower()
                    epsilon = float(rule.get("epsilon", 1.0e-6))
                    if not process_ids:
                        raise ValueError(
                            f"train.epoch_sampler.hard_target_edges.rules[{idx}].process_ids "
                            "must not be empty for main_row_ratio."
                        )
                    if not numerator_column or not denominator_column:
                        raise ValueError(
                            f"train.epoch_sampler.hard_target_edges.rules[{idx}] requires "
                            "numerator_column and denominator_column for main_row_ratio."
                        )
                    if transform not in {"raw_ratio", "log_ratio"}:
                        raise ValueError(
                            f"train.epoch_sampler.hard_target_edges.rules[{idx}].transform "
                            "must be 'raw_ratio' or 'log_ratio'."
                        )
                    if not math.isfinite(epsilon) or epsilon <= 0.0:
                        raise ValueError(
                            f"train.epoch_sampler.hard_target_edges.rules[{idx}].epsilon "
                            "must be finite and positive."
                        )
                    rule["process_ids"] = process_ids
                    rule["numerator_column"] = numerator_column
                    rule["denominator_column"] = denominator_column
                    rule["transform"] = transform
                    rule["epsilon"] = epsilon
                validated_rules.append(rule)
            hard_edges_cfg.rules = validated_rules
        sampler_cfg = _as_dataclass(EpochBalancedSamplerConfig, sampler_payload)
        sampler_cfg.mixture = mixture_cfg
        sampler_cfg.sparse_positive = sparse_cfg
        sampler_cfg.quantile_balance = quantile_cfg
        sampler_cfg.mass_flow_tail = tail_cfg
        sampler_cfg.hard_target_edges = hard_edges_cfg
        sampler_cfg.mode = str(getattr(sampler_cfg, "mode", "mixture") or "mixture").strip().lower()
        if sampler_cfg.mode not in {
            "random",
            "mixture",
            "scheduled_predefined_hard",
            "base_plus_hard_fill",
        }:
            raise ValueError(
                "train.epoch_sampler.mode must be 'random', 'mixture', "
                "'scheduled_predefined_hard', or 'base_plus_hard_fill'."
            )
        for key in ("warmup_end_epoch", "transition_end_epoch"):
            if int(getattr(sampler_cfg, key)) < 0:
                raise ValueError(f"train.epoch_sampler.{key} must be >= 0.")
        for key in ("transition_hard_ratio", "final_hard_ratio"):
            value = float(getattr(sampler_cfg, key))
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"train.epoch_sampler.{key} must be a finite ratio in [0, 1].")
        if (
            sampler_cfg.mode in {"scheduled_predefined_hard", "base_plus_hard_fill"}
            and not bool(getattr(hard_edges_cfg, "enabled", False))
        ):
            raise ValueError(
                "train.epoch_sampler.hard_target_edges.enabled must be true "
                "when mode uses hard target edges."
            )
        if not math.isfinite(float(sampler_cfg.epoch_fraction)) or float(sampler_cfg.epoch_fraction) <= 0.0:
            raise ValueError("train.epoch_sampler.epoch_fraction must be finite and positive.")
        hard_fill_total_size = int(getattr(sampler_cfg, "hard_fill_total_size", 0))
        hard_fill_ratio = float(getattr(sampler_cfg, "hard_fill_ratio", 0.0))
        if hard_fill_total_size < 0:
            raise ValueError("train.epoch_sampler.hard_fill_total_size must be >= 0.")
        if not math.isfinite(hard_fill_ratio) or hard_fill_ratio < 0.0 or hard_fill_ratio > 1.0:
            raise ValueError("train.epoch_sampler.hard_fill_ratio must be a finite ratio in [0, 1].")
        if int(sampler_cfg.max_repeat_per_sample_per_epoch) < 1:
            raise ValueError("train.epoch_sampler.max_repeat_per_sample_per_epoch must be >= 1.")
        payload["epoch_sampler"] = sampler_cfg
    closure_weight = float(payload.get("pi_fraction_closure_loss_weight", 0.0))
    if not math.isfinite(closure_weight) or closure_weight < 0.0:
        raise ValueError(
            "train.pi_fraction_closure_loss_weight must be finite and non-negative, "
            f"got {closure_weight}."
        )
    if fraction_loss_type in {"clr", "log"}:
        if bool(payload.get("pi_normalize_fraction_loss", False)):
            raise ValueError(
                f"train.pi_fraction_loss_type={fraction_loss_type!r} cannot be combined with "
                "pi_normalize_fraction_loss=true."
            )
        if not bool(payload.get("use_zero_flow_fraction_mask", False)):
            raise ValueError(
                f"train.pi_fraction_loss_type={fraction_loss_type!r} requires "
                "use_zero_flow_fraction_mask=true."
            )
    fraction_component_penalty = payload.get("pi_fraction_component_penalty")
    if fraction_component_penalty is not None:
        if not isinstance(fraction_component_penalty, Mapping):
            raise TypeError("train.pi_fraction_component_penalty must be a mapping.")
        enabled = bool(fraction_component_penalty.get("enabled", False))
        loss_type = str(fraction_component_penalty.get("loss_type", "smooth_l1") or "smooth_l1").strip().lower()
        if loss_type not in {"smooth_l1", "huber", "mse", "l2", "l1", "mae"}:
            raise ValueError(
                "train.pi_fraction_component_penalty.loss_type must be smooth_l1, mse, or l1 "
                f"compatible, got {loss_type!r}."
            )
        eps = float(fraction_component_penalty.get("eps", 1.0e-8))
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("train.pi_fraction_component_penalty.eps must be finite and positive.")
        rampup = fraction_component_penalty.get("rampup", {}) or {}
        if not isinstance(rampup, Mapping):
            raise TypeError("train.pi_fraction_component_penalty.rampup must be a mapping.")
        ramp_fraction = float(rampup.get("fraction", 0.2))
        if not math.isfinite(ramp_fraction) or ramp_fraction < 0.0:
            raise ValueError("train.pi_fraction_component_penalty.rampup.fraction must be finite and non-negative.")
        for block_name in ("co_positive", "soft_zero_fp", "co2_positive", "diagnostics"):
            block = fraction_component_penalty.get(block_name, {}) or {}
            if not isinstance(block, Mapping):
                raise TypeError(f"train.pi_fraction_component_penalty.{block_name} must be a mapping.")
        if enabled and fraction_loss_type != "clr":
            print(
                "[config][warning] pi_fraction_component_penalty is enabled with "
                f"pi_fraction_loss_type={fraction_loss_type!r}; this is allowed but the current "
                "ablation plan expects CLR baseline fraction loss."
            )
    mw_unit_scale = payload.get("mw_unit_scale")
    if mw_unit_scale is not None and (
        not math.isfinite(float(mw_unit_scale)) or float(mw_unit_scale) <= 0.0
    ):
        raise ValueError(f"train.mw_unit_scale must be finite and positive, got {mw_unit_scale}.")
    volume_unit_scale = float(payload.get("volume_unit_scale", 1.0))
    if not math.isfinite(volume_unit_scale) or volume_unit_scale <= 0.0:
        raise ValueError(
            f"train.volume_unit_scale must be finite and positive, got {volume_unit_scale}."
        )
    for key in (
        "rho_residual_abs_clip",
        "h_normalized_residual_abs_clip",
        "volume_normalized_residual_abs_clip",
    ):
        value = payload.get(key)
        if value is not None and (not math.isfinite(float(value)) or float(value) <= 0.0):
            raise ValueError(f"train.{key} must be finite and positive when set, got {value}.")
    for key in ("lambda_node_mass", "lambda_node_component", "lambda_node_atom", "lambda_node_energy"):
        value = payload.get(key)
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0.0):
            raise ValueError(f"train.{key} must be finite and non-negative, got {value}.")
    node_balance_pi = payload.get("node_balance_pi")
    if node_balance_pi is not None and not isinstance(node_balance_pi, Mapping):
        raise TypeError("train.node_balance_pi must be a mapping.")
    node_pinn_optimization_raw = payload.get("node_pinn_optimization")
    if node_pinn_optimization_raw is not None:
        if not isinstance(node_pinn_optimization_raw, Mapping):
            raise TypeError("train.node_pinn_optimization must be a mapping.")
        payload["node_pinn_optimization"] = _as_dataclass(
            NodePinnOptimizationConfig,
            node_pinn_optimization_raw,
        )
    mass_flow_aux_raw = payload.get("mass_flow_physical_auxiliary")
    if mass_flow_aux_raw is not None:
        if not isinstance(mass_flow_aux_raw, Mapping):
            raise TypeError(
                "train.mass_flow_physical_auxiliary must be a mapping."
            )
        payload["mass_flow_physical_auxiliary"] = _as_dataclass(
            MassFlowPhysicalAuxiliaryConfig,
            mass_flow_aux_raw,
        )
    cfg = _as_dataclass(TrainConfig, payload)
    cfg.mass_flow_transform = mass_flow_transform
    cfg.mass_flow_log_tau = mass_flow_log_tau
    cfg.mass_flow_log_scale = mass_flow_log_scale
    cfg.mass_flow_log_eps = mass_flow_log_eps
    sampler_raw = payload.get("rare_positive_sampler")
    if sampler_raw is not None:
        if not isinstance(sampler_raw, Mapping):
            raise TypeError("train.rare_positive_sampler must be a mapping.")
        sampler_cfg = _as_dataclass(RarePositiveSamplerConfig, sampler_raw)
        allowed_components = {
            "Frac_H2O",
            "Frac_H2",
            "Frac_CH4",
            "Frac_CO2",
            "Frac_CO",
            "Frac_O2",
            "Frac_N2",
        }
        components = [str(name) for name in sampler_cfg.components]
        if not components:
            raise ValueError("train.rare_positive_sampler.components must not be empty.")
        unknown = sorted(set(components) - allowed_components)
        if unknown:
            raise ValueError(
                "train.rare_positive_sampler.components contains unsupported fraction columns: "
                f"{unknown}."
            )
        if len(set(components)) != len(components):
            raise ValueError("train.rare_positive_sampler.components must not contain duplicates.")
        sampler_cfg.components = components
        if not math.isfinite(float(sampler_cfg.positive_threshold)) or float(
            sampler_cfg.positive_threshold
        ) < 0.0:
            raise ValueError(
                "train.rare_positive_sampler.positive_threshold must be finite and non-negative."
            )
        if not math.isfinite(float(sampler_cfg.max_weight)) or float(sampler_cfg.max_weight) < 1.0:
            raise ValueError("train.rare_positive_sampler.max_weight must be finite and >= 1.")
        if str(sampler_cfg.mode).strip().lower() != "any_target_edge":
            raise ValueError(
                "train.rare_positive_sampler.mode currently supports only 'any_target_edge'."
            )
        sampler_cfg.mode = "any_target_edge"
        cfg.rare_positive_sampler = sampler_cfg
    augmentation_raw = payload.get("rare_target_edge_augmentation")
    if augmentation_raw is not None:
        if not isinstance(augmentation_raw, Mapping):
            raise TypeError("train.rare_target_edge_augmentation must be a mapping.")
        augmentation_cfg = _as_dataclass(
            RareTargetEdgeAugmentationConfig,
            augmentation_raw,
        )
        allowed_components = {
            "Frac_H2O",
            "Frac_H2",
            "Frac_CH4",
            "Frac_CO2",
            "Frac_CO",
            "Frac_O2",
            "Frac_N2",
        }
        components = [str(name) for name in augmentation_cfg.components]
        if not components:
            raise ValueError(
                "train.rare_target_edge_augmentation.components must not be empty."
            )
        unknown = sorted(set(components) - allowed_components)
        if unknown:
            raise ValueError(
                "train.rare_target_edge_augmentation.components contains unsupported "
                f"fraction columns: {unknown}."
            )
        if len(set(components)) != len(components):
            raise ValueError(
                "train.rare_target_edge_augmentation.components must not contain duplicates."
            )
        augmentation_cfg.components = components
        threshold = float(augmentation_cfg.positive_threshold)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError(
                "train.rare_target_edge_augmentation.positive_threshold must be "
                "finite and non-negative."
            )
        high_thresholds = {
            str(name): float(value)
            for name, value in augmentation_cfg.high_positive_thresholds.items()
        }
        unknown_high = sorted(set(high_thresholds) - set(components))
        if unknown_high:
            raise ValueError(
                "train.rare_target_edge_augmentation.high_positive_thresholds "
                f"contains components not selected for augmentation: {unknown_high}."
            )
        if any(not math.isfinite(value) or value < 0.0 for value in high_thresholds.values()):
            raise ValueError(
                "train.rare_target_edge_augmentation.high_positive_thresholds values "
                "must be finite and non-negative."
            )
        augmentation_cfg.high_positive_thresholds = high_thresholds
        if int(augmentation_cfg.default_factor) < 1:
            raise ValueError(
                "train.rare_target_edge_augmentation.default_factor must be >= 1."
            )
        if int(augmentation_cfg.high_positive_factor) < 1:
            raise ValueError(
                "train.rare_target_edge_augmentation.high_positive_factor must be >= 1."
            )
        ratio = float(augmentation_cfg.max_augmented_ratio)
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError(
                "train.rare_target_edge_augmentation.max_augmented_ratio must be "
                "between 0 and 1."
            )
        if not bool(augmentation_cfg.target_edges_only):
            raise ValueError(
                "train.rare_target_edge_augmentation.target_edges_only=false is not supported."
            )
        cfg.rare_target_edge_augmentation = augmentation_cfg
    balanced_raw = payload.get("target_edge_augmentation")
    if balanced_raw is not None:
        if not isinstance(balanced_raw, Mapping):
            raise TypeError("train.target_edge_augmentation must be a mapping.")
        balanced_cfg = _as_dataclass(TargetEdgeAugmentationConfig, balanced_raw)
        allowed_components = {
            "Frac_H2O",
            "Frac_H2",
            "Frac_CH4",
            "Frac_CO2",
            "Frac_CO",
            "Frac_O2",
            "Frac_N2",
        }
        allowed_aug_properties = {*allowed_components, "Mass_Flow"}
        properties = [str(name) for name in balanced_cfg.target_properties]
        if not properties:
            raise ValueError("train.target_edge_augmentation.target_properties must not be empty.")
        mode = str(balanced_cfg.selection_mode).strip().lower()
        unknown = sorted(set(properties) - allowed_aug_properties)
        if unknown:
            raise ValueError(
                "train.target_edge_augmentation.target_properties contains unsupported "
                f"fraction columns: {unknown}."
            )
        if len(set(properties)) != len(properties):
            raise ValueError(
                "train.target_edge_augmentation.target_properties must not contain duplicates."
            )
        balanced_cfg.target_properties = properties
        if int(balanced_cfg.max_augmented_items) < 0:
            raise ValueError(
                "train.target_edge_augmentation.max_augmented_items must be non-negative."
            )
        if balanced_cfg.max_augmented_ratio is not None:
            ratio = float(balanced_cfg.max_augmented_ratio)
            if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
                raise ValueError(
                    "train.target_edge_augmentation.max_augmented_ratio must be "
                    f"between 0 and 1, got {ratio}."
                )
            balanced_cfg.max_augmented_ratio = ratio
        if int(balanced_cfg.factor) < 1:
            raise ValueError("train.target_edge_augmentation.factor must be >= 1.")
        if mode not in {"balanced_property_edge", "adaptive_bins"}:
            raise ValueError(
                "train.target_edge_augmentation.selection_mode currently supports only "
                "'balanced_property_edge' or 'adaptive_bins'."
            )
        balanced_cfg.selection_mode = mode
        if mode == "balanced_property_edge":
            fraction_properties = [name for name in properties if name in allowed_components]
            if len(fraction_properties) != len(properties):
                raise ValueError(
                    "train.target_edge_augmentation.selection_mode='balanced_property_edge' "
                    "supports only Frac_* target_properties."
                )
            quotas = {str(name): int(value) for name, value in balanced_cfg.property_quota.items()}
            if set(quotas) != set(properties):
                raise ValueError(
                    "train.target_edge_augmentation.property_quota keys must exactly match "
                    "target_properties."
                )
            if any(value < 0 for value in quotas.values()):
                raise ValueError(
                    "train.target_edge_augmentation.property_quota values must be non-negative."
                )
            if (
                int(balanced_cfg.max_augmented_items) > 0
                and sum(quotas.values()) > int(balanced_cfg.max_augmented_items)
            ):
                raise ValueError(
                    "train.target_edge_augmentation.property_quota sum must not exceed "
                    "max_augmented_items."
                )
            balanced_cfg.property_quota = quotas
            if int(balanced_cfg.max_per_target_edge) < 1:
                raise ValueError(
                    "train.target_edge_augmentation.max_per_target_edge must be >= 1."
                )
            thresholds = {
                str(name): float(value)
                for name, value in balanced_cfg.min_positive_threshold.items()
            }
            if set(thresholds) != set(properties):
                raise ValueError(
                    "train.target_edge_augmentation.min_positive_threshold keys must exactly "
                    "match target_properties."
                )
            if any(not math.isfinite(value) or value < 0.0 for value in thresholds.values()):
                raise ValueError(
                    "train.target_edge_augmentation.min_positive_threshold values must be "
                    "finite and non-negative."
                )
            balanced_cfg.min_positive_threshold = thresholds
        else:
            if int(balanced_cfg.repeat_cap_per_original) < 1:
                raise ValueError(
                    "train.target_edge_augmentation.repeat_cap_per_original must be >= 1."
                )
            bins = list(balanced_cfg.adaptive_bins or [])
            if not bins:
                raise ValueError(
                    "train.target_edge_augmentation.selection_mode='adaptive_bins' requires "
                    "adaptive_bins."
                )
            total_quota = 0
            for idx, item in enumerate(bins):
                if not isinstance(item, Mapping):
                    raise TypeError(
                        f"train.target_edge_augmentation.adaptive_bins[{idx}] must be a mapping."
                    )
                name = str(item.get("name", "")).strip()
                scope = str(item.get("scope", "")).strip().lower()
                prop = str(item.get("property", "")).strip()
                quota = int(item.get("quota", 0))
                if not name:
                    raise ValueError(
                        f"train.target_edge_augmentation.adaptive_bins[{idx}].name is required."
                    )
                if scope not in {"target", "all"}:
                    raise ValueError(
                        f"train.target_edge_augmentation.adaptive_bins[{idx}].scope must be "
                        "'target' or 'all'."
                    )
                if prop not in allowed_aug_properties:
                    raise ValueError(
                        f"train.target_edge_augmentation.adaptive_bins[{idx}].property "
                        f"is unsupported: {prop!r}."
                    )
                if quota < 0:
                    raise ValueError(
                        f"train.target_edge_augmentation.adaptive_bins[{idx}].quota "
                        "must be non-negative."
                    )
                if prop == "Mass_Flow":
                    qlow = float(item.get("quantile_low", -1.0))
                    qhigh = float(item.get("quantile_high", -1.0))
                    if not (math.isfinite(qlow) and math.isfinite(qhigh) and 0.0 <= qlow < qhigh <= 1.0):
                        raise ValueError(
                            f"train.target_edge_augmentation.adaptive_bins[{idx}] Mass_Flow "
                            "rules require 0 <= quantile_low < quantile_high <= 1."
                        )
                else:
                    lo = float(item.get("min_value", -1.0))
                    hi = float(item.get("max_value", -1.0))
                    if not (math.isfinite(lo) and math.isfinite(hi) and 0.0 <= lo < hi):
                        raise ValueError(
                            f"train.target_edge_augmentation.adaptive_bins[{idx}] fraction "
                            "rules require 0 <= min_value < max_value."
                        )
                total_quota += quota
            if (
                int(balanced_cfg.max_augmented_items) > 0
                and total_quota > int(balanced_cfg.max_augmented_items)
            ):
                raise ValueError(
                    "train.target_edge_augmentation adaptive_bins quota sum must not exceed "
                    "max_augmented_items."
                )
            priority_edges = {
                str(edge_id): float(weight)
                for edge_id, weight in balanced_cfg.mass_flow_priority_edges.items()
            }
            if any(not math.isfinite(weight) or weight < 0.0 for weight in priority_edges.values()):
                raise ValueError(
                    "train.target_edge_augmentation.mass_flow_priority_edges values must be "
                    "finite and non-negative."
                )
            balanced_cfg.mass_flow_priority_edges = priority_edges
        if not bool(balanced_cfg.keep_original_items):
            raise ValueError(
                "train.target_edge_augmentation.keep_original_items=false is not supported."
            )
        apply_to = [str(split).strip().lower() for split in balanced_cfg.apply_to]
        if apply_to != ["train"]:
            raise ValueError(
                "train.target_edge_augmentation.apply_to must be exactly ['train']."
            )
        balanced_cfg.apply_to = apply_to
        cfg.target_edge_augmentation = balanced_cfg
    return cfg


def load_data_config(path: Path, overrides: Mapping[str, Any] | None = None) -> DataConfig:
    payload = read_yaml_mapping(path)
    if overrides:
        deep_merge(payload, dict(overrides))
    legacy_keys = {"target_categories", "use_target_categories"}
    if any(k in payload for k in legacy_keys):
        raise ValueError(
            "Deprecated data config keys detected (target_categories/use_target_categories). "
            "Use fixed_tasks + aux_task structure."
        )

    cfg = _as_dataclass(DataConfig, payload)

    fixed_tasks_raw = payload.get("fixed_tasks")
    if isinstance(fixed_tasks_raw, Mapping):
        parsed_fixed: Dict[str, FixedTaskConfig] = {}
        for key, value in fixed_tasks_raw.items():
            if not isinstance(value, Mapping):
                raise TypeError(f"data.fixed_tasks.{key} must be a mapping.")
            parsed_fixed[str(key)] = _as_dataclass(FixedTaskConfig, value)
        cfg.fixed_tasks = parsed_fixed

    aux_task_raw = payload.get("aux_task")
    if isinstance(aux_task_raw, Mapping):
        cfg.aux_task = _as_dataclass(AuxTaskConfig, aux_task_raw)

    decoder_tasks_raw = payload.get("decoder_tasks")
    if isinstance(decoder_tasks_raw, Mapping):
        parsed_decoder: Dict[str, DecoderTaskConfig] = {}
        for key, value in decoder_tasks_raw.items():
            if key not in DECODER_CATEGORIES:
                raise ValueError(
                    f"data.decoder_tasks.{key!r} is not a recognised decoder category. "
                    f"Allowed: {DECODER_CATEGORIES}."
                )
            if not isinstance(value, Mapping):
                raise TypeError(f"data.decoder_tasks.{key} must be a mapping.")
            parsed_decoder[str(key)] = _as_dataclass(DecoderTaskConfig, value)
        for key in DECODER_CATEGORIES:
            parsed_decoder.setdefault(key, DecoderTaskConfig())
        cfg.decoder_tasks = parsed_decoder

    required_fixed = {"target", "tailgas"}
    missing = required_fixed - set(cfg.fixed_tasks.keys())
    if missing:
        raise ValueError(f"fixed_tasks must include target and tailgas. Missing: {sorted(missing)}")
    if not cfg.fixed_tasks["target"].enabled or not cfg.fixed_tasks["tailgas"].enabled:
        raise ValueError("fixed_tasks.target and fixed_tasks.tailgas must both be enabled=true.")
    if cfg.fixed_tasks["target"].target_name != "target":
        raise ValueError("fixed_tasks.target.target_name must be 'target'.")
    if cfg.fixed_tasks["tailgas"].target_name != "tailgas":
        raise ValueError("fixed_tasks.tailgas.target_name must be 'tailgas'.")
    decoder_on = any(t.enabled for t in cfg.decoder_tasks.values())
    if cfg.aux_task.enabled and decoder_on:
        raise ValueError(
            "Enable either data.aux_task or data.decoder_tasks, not both at the same time."
        )
    if cfg.aux_task.enabled and cfg.aux_task.target_name != "aux":
        raise ValueError("aux_task.target_name must be 'aux'.")
    if cfg.task_mode not in ("multitask", "target_only", "edge_all"):
        raise ValueError(
            f"data.task_mode must be 'multitask', 'target_only', or 'edge_all', got {cfg.task_mode!r}."
        )
    if cfg.task_mode == "edge_all":
        if not getattr(cfg, "use_canonical_graph_spec_v3", False):
            raise ValueError("data.task_mode='edge_all' requires use_canonical_graph_spec_v3=true.")
        if cfg.topology_mode != "stream_edge":
            raise ValueError("data.task_mode='edge_all' requires topology_mode='stream_edge'.")
        if decoder_on:
            raise ValueError("data.task_mode='edge_all' cannot be combined with decoder_tasks enabled=true.")
        if cfg.aux_task.enabled:
            raise ValueError("data.task_mode='edge_all' cannot be combined with aux_task.enabled=true.")
    if cfg.task_mode == "target_only" and cfg.aux_task.enabled:
        raise ValueError("data.task_mode='target_only' cannot be combined with aux_task.enabled=true.")
    if cfg.topology_mode not in ("excel_node", "stream_edge"):
        raise ValueError(
            f"data.topology_mode must be 'excel_node' or 'stream_edge', got {cfg.topology_mode!r}."
        )
    ref = str(getattr(cfg, "edge_all_reference_dir", "") or "").strip()
    if ref:
        cfg.canonical_graph_spec_v3_dir = ref
    sdir = str(getattr(cfg, "edge_all_stream_dir", "") or "").strip()
    if sdir:
        cfg.stream_data_dir = sdir
    return cfg


def _merge_edge_all_block(
    edge_all_block: Mapping[str, Any] | None,
    *,
    model_overrides: dict[str, Any],
    train_overrides: dict[str, Any],
    data_overrides: dict[str, Any],
) -> None:
    """Merge optional experiment YAML `edge_all:` subtree into overrides."""
    if not isinstance(edge_all_block, Mapping):
        return
    model_overrides.update(edge_all_block.get("model", {}) or {})
    data_overrides.update(edge_all_block.get("data", {}) or {})
    loss_blk = edge_all_block.get("loss", {}) or {}
    if loss_blk.get("type") == "masked_mse":
        train_overrides.setdefault("loss_type_edge_all", "mse")
    elif loss_blk.get("type") == "masked_smooth_l1":
        train_overrides.setdefault("loss_type_edge_all", "smooth_l1")
    elif loss_blk.get("type") == "masked_l1":
        train_overrides.setdefault("loss_type_edge_all", "l1")
    if "use_y_edge_mask" in loss_blk:
        train_overrides["edge_all_use_y_edge_mask"] = bool(loss_blk["use_y_edge_mask"])
    if loss_blk:
        if "normalize_targets" in loss_blk:
            data_overrides["normalize_y_edge"] = bool(loss_blk.get("normalize_targets", True))
        else:
            data_overrides.setdefault("normalize_y_edge", True)
    norm_blk = edge_all_block.get("normalization", {}) or {}
    if str(norm_blk.get("scaler", "standard")).lower() != "standard":
        raise ValueError("edge_all.normalization.scaler must be 'standard' (only supported scaler).")
    if not bool(norm_blk.get("fit_on_train_only", True)):
        raise ValueError("edge_all.normalization.fit_on_train_only must be true.")
    if not bool(norm_blk.get("mask_aware", True)):
        raise ValueError("edge_all.normalization.mask_aware must be true.")
    dbg = edge_all_block.get("debug", {}) or {}
    if "print_first_batch" in dbg:
        train_overrides["edge_all_print_first_batch"] = bool(dbg["print_first_batch"])
    if "assert_no_leakage" in dbg:
        train_overrides["edge_all_assert_no_leakage"] = bool(dbg["assert_no_leakage"])
    if "export_predictions" in dbg:
        train_overrides["edge_all_export_predictions"] = bool(dbg["export_predictions"])


def load_experiment_config(path: Path) -> ExperimentConfig:
    root = project_root_from_here()
    payload = read_yaml_mapping(path)
    overrides = payload.pop("overrides", {}) or {}
    if not isinstance(overrides, Mapping):
        raise TypeError("experiment.overrides must be a mapping")
    edge_all_nested = payload.pop("edge_all", None)

    experiment = _as_dataclass(ExperimentConfig, payload)
    experiment.project_root = root

    model_path = (root / experiment.model_config).resolve()
    train_path = (root / experiment.train_config).resolve()
    data_path = (root / experiment.data_config).resolve()

    model_overrides: dict[str, Any] = dict(overrides.get("model", {}) or {})
    train_overrides: dict[str, Any] = dict(overrides.get("train", {}) or {})
    data_overrides: dict[str, Any] = dict(overrides.get("data", {}) or {})
    _merge_edge_all_block(edge_all_nested, model_overrides=model_overrides, train_overrides=train_overrides, data_overrides=data_overrides)

    if not isinstance(model_overrides, Mapping):
        raise TypeError("overrides.model must be a mapping")
    if not isinstance(train_overrides, Mapping):
        raise TypeError("overrides.train must be a mapping")
    if not isinstance(data_overrides, Mapping):
        raise TypeError("overrides.data must be a mapping")

    experiment.model = load_model_yaml_config(model_path, model_overrides)
    experiment.train = load_train_config(train_path, train_overrides)
    experiment.data = load_data_config(data_path, data_overrides)
    known_feed_cfg = parse_known_feed_condition(
        getattr(experiment.model, "known_feed_condition", None)
    )
    experiment.model.known_feed_condition = known_feed_config_dict(known_feed_cfg)
    experiment.data.known_feed_condition = known_feed_config_dict(known_feed_cfg)
    hx_pair_cfg = parse_hx_pair_relation(
        getattr(experiment.model, "hx_pair_relation", None)
    )
    experiment.model.hx_pair_relation = hx_pair_config_dict(hx_pair_cfg)
    experiment.data.hx_pair_relation = hx_pair_config_dict(hx_pair_cfg)
    feed_head_cfg = parse_feed_head_conditioning(
        getattr(experiment.model, "feed_head_conditioning", None)
    )
    experiment.model.feed_head_conditioning = feed_head_config_dict(feed_head_cfg)
    experiment.data.feed_head_conditioning = feed_head_config_dict(feed_head_cfg)
    if (
        getattr(experiment.data, "task_mode", "multitask") == "edge_all"
        and getattr(experiment.model, "edge_head_type", "single") == "grouped_property"
    ):
        cols_mode = str(getattr(experiment.data, "edge_target_columns_mode", "auto")).strip().lower()
        if cols_mode == "explicit":
            cols = [str(c) for c in (getattr(experiment.data, "edge_target_columns", None) or [])]
            expected = list(STREAM_EDGE_FEATURE_SLOTS)
            if cols != expected:
                raise ValueError(
                    "model.edge_head_type='grouped_property' currently reassembles branch outputs in "
                    "STREAM_EDGE_FEATURE_SLOTS order. When data.edge_target_columns_mode='explicit', "
                    f"data.edge_target_columns must equal {expected}, got {cols}."
                )
    return experiment
