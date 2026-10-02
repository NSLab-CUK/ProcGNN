"""PINN-style edge-wise training utilities for the PI grouped edge head."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import torch
import torch.nn.functional as F

from ..constants import STREAM_EDGE_FEATURE_SLOTS
from .edge_step_training import build_edge_step_edge_weight_vector, build_target_edge_boolean_mask
from .node_balance_pi import compute_node_balance_pinn_losses, resolve_node_balance_config
from .pi_mass_flow import (
    decode_pi_mass_flow_prediction,
    resolve_mass_flow_log_eps,
    resolve_mass_flow_log_scale,
    resolve_mass_flow_log_tau,
    resolve_mass_flow_transform,
    resolve_pi_mass_flow_output_space,
    transform_mass_flow,
)
from .target_edge_10d_metrics import CONSTANT_TARGET_R2_VALUE, R2_SST_EPS

DEFAULT_PI_SPECIES_ORDER: tuple[str, ...] = ("H2O", "H2", "CH4", "CO2", "CO", "O2", "N2")
DEFAULT_MOLECULAR_WEIGHT_G_PER_MOL: dict[str, float] = {
    "H2O": 18.01528,
    "H2": 2.01588,
    "CH4": 16.04246,
    "CO2": 44.0095,
    "CO": 28.0101,
    "O2": 31.9988,
    "N2": 28.0134,
}


def _cfg_get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _terminal_log_verbosity(train_cfg: Any) -> str:
    value = str(_cfg_get(train_cfg, "terminal_log_verbosity", "compact") or "compact").strip().lower()
    if value in {"debug", "full"}:
        return "verbose"
    if value in {"silent", "none"}:
        return "quiet"
    if value not in {"quiet", "compact", "verbose"}:
        return "compact"
    return value


def _format_terminal_duration(seconds: float | int | None) -> str:
    try:
        total = int(max(0.0, float(seconds or 0.0)))
    except (TypeError, ValueError):
        return "?"
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _terminal_progress_bar(current: int, total: int | str, *, width: int = 20) -> tuple[str, str]:
    try:
        total_int = int(total)
    except (TypeError, ValueError):
        return "[" + ("-" * width) + "]", "?"
    total_int = max(total_int, 1)
    current_int = min(max(int(current), 0), total_int)
    ratio = current_int / total_int
    filled = int(round(ratio * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]", f"{ratio * 100.0:.1f}%"


def _terminal_eta(start_time: float, current: int, total: int | str) -> tuple[str, str]:
    elapsed = max(0.0, time.perf_counter() - float(start_time))
    try:
        total_int = int(total)
        current_int = max(1, int(current))
    except (TypeError, ValueError):
        return _format_terminal_duration(elapsed), "?"
    if current_int <= 0 or total_int <= 0:
        return _format_terminal_duration(elapsed), "?"
    rate = elapsed / current_int
    remaining = max(0.0, rate * max(total_int - current_int, 0))
    return _format_terminal_duration(elapsed), _format_terminal_duration(remaining)


def _select_edge_row(tensor: torch.Tensor, edge_id: int, feature_dim: int | None = None) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor[:, int(edge_id), :]
    if tensor.ndim == 2:
        return tensor[int(edge_id) : int(edge_id) + 1, :]
    if tensor.ndim == 1:
        row = tensor[int(edge_id) : int(edge_id) + 1].view(1, 1)
        if feature_dim is not None and int(feature_dim) > 1:
            row = row.expand(1, int(feature_dim))
        return row
    raise ValueError(f"expected 1D/2D/3D tensor, got shape={tuple(tensor.shape)}")


def _select_edge_rows(
    tensor: torch.Tensor,
    edge_indices: Sequence[int] | torch.Tensor,
    feature_dim: int | None = None,
) -> torch.Tensor:
    if torch.is_tensor(edge_indices):
        idx = edge_indices.to(device=tensor.device, dtype=torch.long).reshape(-1)
    else:
        idx = torch.tensor([int(i) for i in edge_indices], device=tensor.device, dtype=torch.long)
    if tensor.ndim == 3:
        if idx.numel() != 1:
            raise ValueError("3D PI tensors support one canonical edge id at a time.")
        return tensor[:, int(idx.item()), :]
    if tensor.ndim == 2:
        return tensor.index_select(0, idx)
    if tensor.ndim == 1:
        row = tensor.index_select(0, idx).view(-1, 1)
        if feature_dim is not None and int(feature_dim) > 1:
            row = row.expand(row.shape[0], int(feature_dim))
        return row
    raise ValueError(f"expected 1D/2D/3D tensor, got shape={tuple(tensor.shape)}")


def _edge_count_from_target(target: torch.Tensor) -> int:
    if target.ndim == 3:
        return int(target.shape[1])
    if target.ndim in {1, 2}:
        return int(target.shape[0])
    raise ValueError(f"expected target tensor [E,D] or [B,E,D], got shape={tuple(target.shape)}")


def _loss_raw(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    lt = str(loss_type or "smooth_l1").strip().lower()
    if lt in {"smooth_l1", "huber"}:
        return F.smooth_l1_loss(pred, target, reduction="none")
    if lt in {"mse", "l2"}:
        return F.mse_loss(pred, target, reduction="none")
    if lt in {"l1", "mae"}:
        return F.l1_loss(pred, target, reduction="none")
    if lt in {"percentage", "percent", "mape"}:
        return (pred - target).abs() / target.abs().clamp_min(1.0e-6)
    raise ValueError(f"Unsupported PI main loss type: {loss_type}")


def resolve_pi_main_loss_type(train_cfg: Any) -> str:
    """Return the unreduced loss type for PI main_stream_pred supervision."""
    explicit = str(_cfg_get(train_cfg, "pi_main_loss_type", "") or "").strip()
    if explicit:
        return explicit
    return str(_cfg_get(train_cfg, "loss_type_edge_all", "smooth_l1") or "smooth_l1").strip()


def resolve_pi_fraction_loss_type(train_cfg: Any) -> str:
    value = str(_cfg_get(train_cfg, "pi_fraction_loss_type", "legacy") or "legacy").strip().lower()
    if value not in {"legacy", "clr", "log"}:
        raise ValueError(f"pi_fraction_loss_type must be 'legacy', 'clr', or 'log', got {value!r}.")
    if value in {"clr", "log"} and bool(_cfg_get(train_cfg, "pi_normalize_fraction_loss", False)):
        raise ValueError(
            f"{value.upper()} fraction loss cannot be combined with pi_normalize_fraction_loss=true."
        )
    if value in {"clr", "log"} and not bool(_cfg_get(train_cfg, "use_zero_flow_fraction_mask", False)):
        raise ValueError(f"{value.upper()} fraction loss requires use_zero_flow_fraction_mask=true.")
    return value


def _clr_transform(values: torch.Tensor, eps: float) -> torch.Tensor:
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError(f"CLR eps must be finite and positive, got {eps}.")
    logged = torch.log(values.float().clamp_min(float(eps)))
    clr = logged - logged.mean(dim=-1, keepdim=True)
    if not torch.isfinite(clr).all():
        raise RuntimeError("CLR transform produced NaN or Inf.")
    return clr


def _log_fraction_transform(values: torch.Tensor, eps: float) -> torch.Tensor:
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError(f"Fraction log eps must be finite and positive, got {eps}.")
    logged = torch.log(values.float().clamp_min(float(eps)))
    if not torch.isfinite(logged).all():
        raise RuntimeError("Fraction log transform produced NaN or Inf.")
    return logged


def build_pi_main_criterion(train_cfg: Any):
    loss_type = resolve_pi_main_loss_type(train_cfg)
    return lambda p, t: _loss_raw(p, t, loss_type)


def _cfg_mapping(obj: Any, name: str) -> Mapping[str, Any]:
    value = _cfg_get(obj, name, {}) or {}
    return value if isinstance(value, Mapping) else {}


def _sub_mapping(obj: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = obj.get(name, {}) or {}
    return value if isinstance(value, Mapping) else {}


def _finite_nonnegative_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    return out if math.isfinite(out) else float(default)


def _fraction_component_index(species: Sequence[str], component_name: str) -> int | None:
    name = str(component_name).strip()
    if name.startswith("Frac_"):
        name = name[5:]
    for i, species_name in enumerate(species):
        if str(species_name) == name:
            return int(i)
    return None


def _fraction_penalty_ramp_weight(
    penalty_cfg: Mapping[str, Any],
    *,
    global_step: int | None,
    total_train_steps: int | None,
) -> float:
    ramp_cfg = _sub_mapping(penalty_cfg, "rampup")
    if not bool(ramp_cfg.get("enabled", True)):
        return 1.0
    fraction = _finite_nonnegative_float(ramp_cfg.get("fraction", 0.2), 0.2)
    if fraction <= 0.0:
        return 1.0
    if global_step is None or total_train_steps is None or int(total_train_steps) <= 0:
        return 1.0
    denom = max(1.0, float(int(total_train_steps)) * fraction)
    return float(min(1.0, max(0.0, float(int(global_step)) / denom)))


def _weighted_fraction_loss_per_sample(
    element_loss: torch.Tensor,
    weight: torch.Tensor,
    valid: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    w = torch.where(valid, weight.to(device=element_loss.device, dtype=element_loss.dtype), torch.zeros_like(element_loss))
    finite = torch.isfinite(element_loss) & torch.isfinite(w)
    w = torch.where(finite, w, torch.zeros_like(w))
    denom = w.sum().clamp_min(float(eps))
    valid_sample = valid & finite
    valid_count = valid_sample.to(dtype=element_loss.dtype).sum().clamp_min(1.0)
    sample_loss = torch.where(
        valid_sample,
        element_loss * w * valid_count / denom,
        torch.zeros_like(element_loss),
    )
    return sample_loss, valid_sample, w


def _compute_fraction_component_penalty(
    *,
    pred_frac: torch.Tensor,
    true_frac: torch.Tensor,
    frac_mask: torch.Tensor,
    row_valid: torch.Tensor,
    species: Sequence[str],
    train_cfg: Any,
    global_step: int | None,
    total_train_steps: int | None,
    collect_debug: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    penalty_cfg = _cfg_mapping(train_cfg, "pi_fraction_component_penalty")
    enabled = bool(penalty_cfg.get("enabled", False))
    zero = pred_frac.sum(dim=-1).float() * 0.0
    valid_zero = torch.zeros_like(zero, dtype=torch.bool)
    if not enabled:
        return zero, valid_zero, {
            "frac_penalty_enabled": False,
            "frac_penalty_ramp_weight": 0.0,
            "loss_frac_penalty_total": 0.0,
            "loss_co_pos": 0.0,
            "loss_ch4_fp": 0.0,
            "loss_co2_fp": 0.0,
            "loss_h2_fp": 0.0,
            "loss_co2_pos": 0.0,
        }

    eps = _finite_nonnegative_float(penalty_cfg.get("eps", 1.0e-8), 1.0e-8)
    eps = max(eps, 1.0e-12)
    loss_type = str(penalty_cfg.get("loss_type", "smooth_l1") or "smooth_l1").strip().lower()
    ramp_weight = _fraction_penalty_ramp_weight(
        penalty_cfg,
        global_step=global_step,
        total_train_steps=total_train_steps,
    )
    base_valid = row_valid.bool().unsqueeze(-1) & frac_mask.bool()
    finite = torch.isfinite(pred_frac) & torch.isfinite(true_frac)
    base_valid = base_valid & finite
    total = zero.clone()
    valid_any = valid_zero.clone()
    debug: dict[str, Any] = {
        "frac_penalty_enabled": True,
        "frac_penalty_ramp_weight": ramp_weight,
        "loss_frac_penalty_total": 0.0,
        "loss_co_pos": 0.0,
        "loss_ch4_fp": 0.0,
        "loss_co2_fp": 0.0,
        "loss_h2_fp": 0.0,
        "loss_co2_pos": 0.0,
        "co_pos_gate_mean": 0.0,
        "co_pos_gate_sum": 0.0,
        "co_pos_effective_count": 0.0,
        "co_true_positive_count_true_gt_001": 0.0,
        "co_pred_mean_on_positive": 0.0,
        "co_true_mean_on_positive": 0.0,
        "zero_gate_ch4_mean": 0.0,
        "zero_gate_co2_mean": 0.0,
        "zero_gate_h2_mean": 0.0,
        "fp_ch4_active_count_pred_gt_margin": 0.0,
        "fp_co2_active_count_pred_gt_margin": 0.0,
        "fp_h2_active_count_pred_gt_margin": 0.0,
        "co2_pos_gate_mean": 0.0,
        "co2_pos_gate_sum": 0.0,
        "co2_true_high_count_true_gt_005": 0.0,
        "co2_true_high_count_true_gt_02": 0.0,
        "co2_pred_mean_on_high": 0.0,
    }

    def component_loss_raw(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if loss_type in {"smooth_l1", "huber"}:
            return F.smooth_l1_loss(pred, target, reduction="none")
        if loss_type in {"mse", "l2"}:
            return F.mse_loss(pred, target, reduction="none")
        if loss_type in {"l1", "mae"}:
            return F.l1_loss(pred, target, reduction="none")
        raise ValueError(f"Unsupported pi_fraction_component_penalty.loss_type={loss_type!r}.")

    co_cfg = _sub_mapping(penalty_cfg, "co_positive")
    if bool(co_cfg.get("enabled", False)):
        idx = _fraction_component_index(species, "Frac_CO")
        if idx is not None:
            pred = pred_frac[..., idx].float()
            true = true_frac[..., idx].float()
            valid = base_valid[..., idx]
            threshold = _finite_nonnegative_float(co_cfg.get("threshold", 0.01), 0.01)
            temperature = max(_finite_nonnegative_float(co_cfg.get("temperature", 0.005), 0.005), eps)
            lam = _finite_nonnegative_float(co_cfg.get("lambda", 2.0), 2.0)
            gate = torch.sigmoid((true - threshold) / temperature)
            element_loss = component_loss_raw(pred, true)
            loss, valid_loss, weight = _weighted_fraction_loss_per_sample(element_loss, gate, valid, eps=eps)
            contribution = float(ramp_weight) * lam * loss
            total = total + contribution
            valid_any = valid_any | valid_loss
            if collect_debug:
                positive = valid & (true > 0.01)
                weight_max = weight.max() if weight.numel() else weight.new_tensor(0.0)
                debug.update(
                    {
                        "loss_co_pos": _mean_valid(loss, valid_loss),
                        "weighted_loss_co_pos": _mean_valid(contribution, valid_loss),
                        "co_pos_gate_mean": float(gate[valid].mean().detach().cpu()) if bool(valid.any()) else 0.0,
                        "co_pos_gate_sum": float(weight.sum().detach().cpu()),
                        "co_pos_effective_count": (
                            float((weight.sum() / weight_max.clamp_min(eps)).detach().cpu())
                            if bool((weight_max > 0.0).detach().cpu())
                            else 0.0
                        ),
                        "co_true_positive_count_true_gt_001": int(positive.sum().detach().cpu()),
                        "co_pred_mean_on_positive": float(pred[positive].mean().detach().cpu()) if bool(positive.any()) else 0.0,
                        "co_true_mean_on_positive": float(true[positive].mean().detach().cpu()) if bool(positive.any()) else 0.0,
                    }
                )

    fp_cfg = _sub_mapping(penalty_cfg, "soft_zero_fp")
    if bool(fp_cfg.get("enabled", False)):
        eps_true = max(_finite_nonnegative_float(fp_cfg.get("eps_true", 0.001), 0.001), eps)
        for species_name, debug_stem, default_lambda, default_margin in (
            ("Frac_CH4", "ch4", 1.0, 0.005),
            ("Frac_CO2", "co2", 1.0, 0.005),
            ("Frac_H2", "h2", 0.3, 0.01),
        ):
            idx = _fraction_component_index(species, species_name)
            if idx is None:
                continue
            pred = pred_frac[..., idx].float()
            true = true_frac[..., idx].float()
            valid = base_valid[..., idx]
            lam = _finite_nonnegative_float(fp_cfg.get(f"{debug_stem}_lambda", default_lambda), default_lambda)
            margin = _finite_nonnegative_float(fp_cfg.get(f"{debug_stem}_margin", default_margin), default_margin)
            zero_gate = eps_true / (true + eps_true)
            element_loss = F.relu(pred - margin).pow(2)
            loss, valid_loss, weight = _weighted_fraction_loss_per_sample(element_loss, zero_gate, valid, eps=eps)
            contribution = float(ramp_weight) * lam * loss
            total = total + contribution
            valid_any = valid_any | valid_loss
            if collect_debug:
                debug[f"loss_{debug_stem}_fp"] = _mean_valid(loss, valid_loss)
                debug[f"weighted_loss_{debug_stem}_fp"] = _mean_valid(contribution, valid_loss)
                debug[f"zero_gate_{debug_stem}_mean"] = (
                    float(zero_gate[valid].mean().detach().cpu()) if bool(valid.any()) else 0.0
                )
                debug[f"fp_{debug_stem}_active_count_pred_gt_margin"] = int((valid & (pred > margin)).sum().detach().cpu())

    co2_cfg = _sub_mapping(penalty_cfg, "co2_positive")
    if bool(co2_cfg.get("enabled", False)):
        idx = _fraction_component_index(species, "Frac_CO2")
        if idx is not None:
            pred = pred_frac[..., idx].float()
            true = true_frac[..., idx].float()
            valid = base_valid[..., idx]
            lam = _finite_nonnegative_float(co2_cfg.get("lambda", 1.0), 1.0)
            alpha = _finite_nonnegative_float(co2_cfg.get("alpha", 2.0), 2.0)
            scale = max(_finite_nonnegative_float(co2_cfg.get("scale", 0.2), 0.2), eps)
            gate = torch.clamp(true / scale, min=0.0, max=1.0)
            weight = alpha * gate
            element_loss = component_loss_raw(pred, true)
            loss, valid_loss, effective_weight = _weighted_fraction_loss_per_sample(element_loss, weight, valid, eps=eps)
            contribution = float(ramp_weight) * lam * loss
            total = total + contribution
            valid_any = valid_any | valid_loss
            if collect_debug:
                high_005 = valid & (true > 0.05)
                high_02 = valid & (true > 0.2)
                debug.update(
                    {
                        "loss_co2_pos": _mean_valid(loss, valid_loss),
                        "weighted_loss_co2_pos": _mean_valid(contribution, valid_loss),
                        "co2_pos_gate_mean": float(gate[valid].mean().detach().cpu()) if bool(valid.any()) else 0.0,
                        "co2_pos_gate_sum": float(effective_weight.sum().detach().cpu()),
                        "co2_true_high_count_true_gt_005": int(high_005.sum().detach().cpu()),
                        "co2_true_high_count_true_gt_02": int(high_02.sum().detach().cpu()),
                        "co2_pred_mean_on_high": float(pred[high_005].mean().detach().cpu()) if bool(high_005.any()) else 0.0,
                    }
                )

    total = torch.where(valid_any, total, zero)
    if not torch.isfinite(total).all():
        raise RuntimeError("Fraction component penalty produced NaN or Inf.")
    if collect_debug:
        debug["loss_frac_penalty_total"] = _mean_valid(total, valid_any)
        debug["frac_penalty_valid_count"] = int(valid_any.sum().detach().cpu())
    return total, valid_any, debug


def masked_mse_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    loss_raw = (pred - target) ** 2
    loss_raw = loss_raw * mask
    valid_count = mask.sum(dim=-1)
    valid = valid_count > 0
    sample_loss = loss_raw.sum(dim=-1) / valid_count.clamp_min(float(eps))
    sample_loss = torch.where(valid, sample_loss, torch.zeros_like(sample_loss))
    return sample_loss, valid


def masked_normalized_mse_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1.0e-8,
    scale_floor: float | None = None,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    residual_abs_clip: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Symmetric normalization remains finite for physically valid zero-flow rows.
    scale = pred.abs() + target.abs()
    if scale_floor is not None:
        scale = torch.maximum(scale, scale.new_full(scale.shape, float(scale_floor)))
    scale = scale + float(eps)
    residual = (pred - target) / scale
    finite = torch.isfinite(residual)
    mask = mask.to(dtype=torch.bool) & finite
    residual = torch.where(finite, residual, torch.zeros_like(residual))
    if residual_abs_clip is not None:
        limit = float(residual_abs_clip)
        residual = residual.clamp(min=-limit, max=limit)
    loss_name = str(loss_type).strip().lower()
    if loss_name in {"squared", "mse"}:
        return masked_mse_per_sample(residual, torch.zeros_like(residual), mask, eps=eps)
    if loss_name in {"huber", "smooth_l1"}:
        return masked_regression_per_sample(
            residual,
            torch.zeros_like(residual),
            mask,
            criterion=torch.nn.HuberLoss(delta=float(huber_delta), reduction="none"),
            eps=eps,
        )
    if loss_name in {"l1", "mae"}:
        return masked_regression_per_sample(
            residual,
            torch.zeros_like(residual),
            mask,
            criterion=torch.nn.L1Loss(reduction="none"),
            eps=eps,
        )
    raise ValueError(f"Unsupported normalized loss_type={loss_type!r}.")


def masked_residual_regression_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    criterion: Any,
    residual_abs_clip: float | None = None,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    residual = pred - target
    finite = torch.isfinite(residual)
    safe_mask = mask.to(dtype=torch.bool) & finite
    residual = torch.where(finite, residual, torch.zeros_like(residual))
    if residual_abs_clip is not None:
        limit = float(residual_abs_clip)
        residual = residual.clamp(min=-limit, max=limit)
    return masked_regression_per_sample(
        residual,
        torch.zeros_like(residual),
        safe_mask,
        criterion=criterion,
        eps=eps,
    )


def _edge_groups_from_export_meta(edge_export_meta: Any, n_edges: int) -> list[tuple[str, list[int]]]:
    """Group flattened batched rows by process + canonical edge.

    The collate path flattens [B, E, D] edge targets into [B*E, D].  In
    edge_step_pi we still want one optimizer update per canonical edge in the
    current batch, with sample-wise losses inside that update.
    """
    if edge_export_meta is None:
        return [(str(i), [i]) for i in range(int(n_edges))]
    edge_ids = list(getattr(edge_export_meta, "canonical_edge_id", []) or [])
    proc_ids = list(getattr(edge_export_meta, "process_id", []) or [])
    predictable = list(getattr(edge_export_meta, "is_predictable", []) or [])
    if len(edge_ids) != int(n_edges) or len(proc_ids) != int(n_edges):
        return [(str(i), [i]) for i in range(int(n_edges))]
    use_predictable_mask = len(predictable) == int(n_edges)
    groups: dict[tuple[str, str], list[int]] = {}
    for i, (pid, eid) in enumerate(zip(proc_ids, edge_ids)):
        if use_predictable_mask and float(predictable[i]) <= 0.5:
            continue
        key = (str(pid), str(eid))
        groups.setdefault(key, []).append(i)
    return [(f"{pid}:{eid}", idxs) for (pid, eid), idxs in groups.items()]


def masked_regression_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    criterion: Any,
    loss_weight: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    loss_raw = criterion(pred, target)
    if loss_raw.ndim == 0:
        raise RuntimeError("criterion must return unreduced loss. Use reduction='none'.")
    if loss_weight is not None:
        if loss_weight.shape != loss_raw.shape:
            raise RuntimeError(
                f"loss_weight shape {tuple(loss_weight.shape)} does not match loss shape {tuple(loss_raw.shape)}."
            )
        loss_raw = loss_raw * loss_weight.to(device=loss_raw.device, dtype=loss_raw.dtype)
    loss_raw = loss_raw * mask
    valid_count = mask.sum(dim=-1)
    valid = valid_count > 0
    sample_loss = loss_raw.sum(dim=-1) / valid_count.clamp_min(float(eps))
    sample_loss = torch.where(valid, sample_loss, torch.zeros_like(sample_loss))
    return sample_loss, valid


def get_mass_physical_weight(
    *,
    epoch: int,
    start_epoch: int,
    end_epoch: int,
    max_weight: float,
    start_weight: float = 0.0,
    end_weight: float | None = None,
    schedule_type: str = "linear_warmup",
) -> float:
    """Resolve the 1-based epoch weight for the physical Mass_Flow term."""
    epoch = int(epoch)
    start_epoch = int(start_epoch)
    end_epoch = int(end_epoch)
    max_weight = float(max_weight)
    start_weight = float(start_weight)
    end_weight = max_weight if end_weight is None else float(end_weight)
    mode = str(schedule_type).strip().lower()
    if mode in {"constant", "none"}:
        return max_weight
    if mode != "linear_warmup":
        raise ValueError(f"Unsupported Mass_Flow physical schedule_type={mode!r}.")
    if start_epoch < 1 or end_epoch < start_epoch:
        raise ValueError(
            "Mass_Flow physical schedule uses 1-based epochs and requires "
            "1 <= start_epoch <= end_epoch."
        )
    if epoch < start_epoch:
        return 0.0
    if epoch >= end_epoch:
        return end_weight
    if end_epoch == start_epoch:
        return end_weight
    progress = float(epoch - start_epoch) / float(end_epoch - start_epoch)
    return start_weight + progress * (end_weight - start_weight)


def resolve_mass_flow_physical_scale(
    *,
    train_cfg: Any,
    normalizer: Mapping[str, Any] | None,
) -> tuple[float, str]:
    """Resolve one run-level Mass_Flow scale using train-split statistics only."""
    cfg = _cfg_get(train_cfg, "mass_flow_physical_auxiliary", None)
    if normalizer is None:
        raise RuntimeError(
            "mass_flow_physical_auxiliary.enabled=true requires train-split normalizer statistics."
        )
    method = str(_cfg_get(cfg, "scale_method", "train_std")).strip().lower()
    minimum_scale = float(_cfg_get(cfg, "minimum_scale", 1.0e-8))
    resolved = _cfg_get(cfg, "resolved_scale", None)
    if resolved is not None:
        scale = float(resolved)
        source = f"persisted_{method}"
    elif method in {"train_std", "std"}:
        columns = [str(value) for value in normalizer.get("columns", [])]
        std = normalizer.get("std")
        if "Mass_Flow" not in columns or std is None:
            raise RuntimeError("Mass_Flow train-split std is missing from the normalizer.")
        scale = float(torch.as_tensor(std).reshape(-1)[columns.index("Mass_Flow")].item())
        source = "train_std"
    elif method == "train_iqr":
        p25 = normalizer.get("mass_flow_physical_p25")
        p75 = normalizer.get("mass_flow_physical_p75")
        if p25 is None or p75 is None:
            raise RuntimeError(
                "scale_method='train_iqr' requires train-split Mass_Flow P25/P75 statistics."
            )
        scale = float(p75) - float(p25)
        source = "train_iqr"
    elif method == "p95_minus_p05":
        p05 = normalizer.get("mass_flow_physical_p05")
        p95 = normalizer.get("mass_flow_physical_p95")
        if p05 is None or p95 is None:
            raise RuntimeError(
                "scale_method='p95_minus_p05' requires train-split Mass_Flow P05/P95 statistics."
            )
        scale = float(p95) - float(p05)
        source = "train_p95_minus_p05"
    elif method == "fixed":
        fixed = _cfg_get(cfg, "fixed_scale", None)
        if fixed is None:
            raise RuntimeError("scale_method='fixed' requires fixed_scale.")
        scale = float(fixed)
        source = "fixed"
    else:
        raise ValueError(f"Unsupported Mass_Flow physical auxiliary scale_method={method!r}.")
    if not math.isfinite(scale) or scale < minimum_scale:
        raise RuntimeError(
            "Mass_Flow physical auxiliary scale must be finite and >= "
            f"minimum_scale ({minimum_scale:g}); got {scale!r} from {source}."
        )
    return scale, source


def compute_mass_flow_physical_auxiliary_per_sample(
    pred_physical: torch.Tensor,
    target_physical: torch.Tensor,
    mask: torch.Tensor,
    *,
    train_cfg: Any,
    normalizer: Mapping[str, Any] | None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Train-scale-normalized physical Mass_Flow Huber loss."""
    cfg = _cfg_get(train_cfg, "mass_flow_physical_auxiliary", None)
    enabled = bool(_cfg_get(cfg, "enabled", False))
    zero = pred_physical.reshape(-1) * 0.0
    invalid = torch.zeros_like(zero, dtype=torch.bool)
    if not enabled:
        return zero, invalid, {
            "mass_flow_physical_aux_enabled": False,
            "mass_flow_physical_aux_weight": 0.0,
            "mass_flow_physical_aux_scale": math.nan,
            "mass_flow_physical_aux_scale_method": "disabled",
        }
    method = str(_cfg_get(cfg, "scale_method", "train_std")).strip().lower()
    epsilon = float(_cfg_get(cfg, "epsilon", 1.0e-8))
    scale, scale_source = resolve_mass_flow_physical_scale(
        train_cfg=train_cfg,
        normalizer=normalizer,
    )
    residual = (pred_physical.float() - target_physical.float()) / (scale + epsilon)
    residual_nonfinite_count = int((~torch.isfinite(residual)).sum().detach().cpu())
    valid = (
        (mask.reshape(-1) > 0)
        & torch.isfinite(residual.reshape(-1))
        & torch.isfinite(pred_physical.reshape(-1))
        & torch.isfinite(target_physical.reshape(-1))
    )
    residual = torch.where(torch.isfinite(residual), residual, torch.zeros_like(residual))
    clip = _cfg_get(cfg, "residual_clip", 10.0)
    if clip is not None:
        residual = residual.clamp(min=-float(clip), max=float(clip))
    delta = float(_cfg_get(cfg, "delta", 1.0))
    abs_residual = residual.abs()
    loss = torch.where(
        abs_residual <= delta,
        0.5 * residual.square(),
        delta * (abs_residual - 0.5 * delta),
    ).reshape(-1)
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    epoch = int(_cfg_get(cfg, "current_epoch", 1))
    weight = get_mass_physical_weight(
        epoch=epoch,
        start_epoch=int(_cfg_get(cfg, "schedule_start_epoch", 3)),
        end_epoch=int(_cfg_get(cfg, "schedule_end_epoch", 7)),
        max_weight=float(_cfg_get(cfg, "weight", 0.10)),
        start_weight=float(_cfg_get(cfg, "schedule_start_weight", 0.0)),
        end_weight=_cfg_get(cfg, "schedule_end_weight", None),
        schedule_type=str(_cfg_get(cfg, "schedule_type", "linear_warmup")),
    )
    valid_values = valid.reshape(-1)
    pred_valid = pred_physical.reshape(-1)[valid_values]
    target_valid = target_physical.reshape(-1)[valid_values]
    loss_valid = loss.reshape(-1)[valid_values]
    tail_debug: dict[str, float] = {}
    flat_target = target_physical.reshape(-1)
    flat_loss = loss.reshape(-1)
    for label in ("p95", "p99"):
        threshold = normalizer.get(f"mass_flow_physical_{label}") if normalizer else None
        if threshold is None or not math.isfinite(float(threshold)):
            tail_debug[f"mass_flow_physical_loss_{label}_sum"] = 0.0
            tail_debug[f"mass_flow_physical_{label}_count"] = 0.0
            continue
        tail_valid = valid_values & (flat_target >= float(threshold))
        tail_debug[f"mass_flow_physical_loss_{label}_sum"] = (
            float(flat_loss[tail_valid].sum().detach().cpu()) if bool(tail_valid.any()) else 0.0
        )
        tail_debug[f"mass_flow_physical_{label}_count"] = float(tail_valid.sum().detach().cpu())
    return loss, valid, {
        "mass_flow_physical_aux_enabled": True,
        "mass_flow_physical_aux_weight": weight,
        "mass_flow_physical_aux_max_weight": float(_cfg_get(cfg, "weight", 0.10)),
        "mass_flow_physical_aux_epoch": epoch,
        "mass_flow_physical_aux_scale": scale,
        "mass_flow_physical_aux_scale_method": method,
        "mass_flow_physical_aux_scale_source": scale_source,
        "mass_flow_pred_physical_mean": (
            float(pred_valid.mean().detach().cpu()) if pred_valid.numel() else math.nan
        ),
        "mass_flow_true_physical_mean": (
            float(target_valid.mean().detach().cpu()) if target_valid.numel() else math.nan
        ),
        "mass_flow_physical_loss_max": (
            float(loss_valid.max().detach().cpu()) if loss_valid.numel() else 0.0
        ),
        "mass_flow_physical_nonfinite_count": residual_nonfinite_count,
        **tail_debug,
    }


def resolve_pi_species_order(data_cfg: Any, train_cfg: Any, outputs: Mapping[str, torch.Tensor]) -> list[str]:
    raw = list(_cfg_get(data_cfg, "species_order", None) or _cfg_get(train_cfg, "species_order", None) or [])
    num_species = int(outputs["frac_pred"].shape[-1])
    if raw:
        species = [str(s) for s in raw]
        if len(species) != num_species:
            raise AssertionError(f"len(species_order)={len(species)} != num_species={num_species}")
        return species
    if num_species == len(DEFAULT_PI_SPECIES_ORDER):
        return list(DEFAULT_PI_SPECIES_ORDER)
    frac_slots = [s.replace("Frac_", "") for s in STREAM_EDGE_FEATURE_SLOTS if s.startswith("Frac_")]
    if len(frac_slots) >= num_species:
        return frac_slots[:num_species]
    raise RuntimeError("Cannot infer PI species_order; set data.species_order.")


def _column_indices(cols: Sequence[str], names: Sequence[str]) -> list[int]:
    col_map = {str(c): i for i, c in enumerate(cols)}
    missing = [n for n in names if n not in col_map]
    if missing:
        raise RuntimeError(f"PI target columns missing: {missing}; available={list(cols)}")
    return [int(col_map[n]) for n in names]


def _expand_edge_mask(mask: torch.Tensor, width: int, target: torch.Tensor) -> torch.Tensor:
    m = mask.to(device=target.device, dtype=target.dtype)
    if m.ndim == target.ndim - 1:
        m = m.unsqueeze(-1)
    if m.shape[-1] == 1:
        m = m.expand(*target.shape[:-1], int(width))
    if m.shape != target.shape:
        raise RuntimeError(f"mask shape {tuple(m.shape)} does not match target shape {tuple(target.shape)}")
    return m


def _take_columns(x: torch.Tensor, idx: Sequence[int]) -> torch.Tensor:
    index = torch.tensor(list(idx), device=x.device, dtype=torch.long)
    return x.index_select(dim=-1, index=index)


def get_pi_targets(
    *,
    targets: Mapping[str, torch.Tensor],
    target_masks: Mapping[str, torch.Tensor],
    outputs: Mapping[str, torch.Tensor],
    train_cfg: Any,
    data_cfg: Any,
    edge_target_columns: Sequence[str] | None = None,
) -> dict[str, torch.Tensor | None]:
    if "edge_stream" not in targets:
        raise RuntimeError("PI loss requires targets['edge_stream'].")
    if "edge_stream" not in target_masks:
        raise RuntimeError("PI loss requires target_masks['edge_stream'].")
    cols = list(edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
    y = targets["edge_stream"].to(device=outputs["main_stream_pred"].device, dtype=outputs["main_stream_pred"].dtype)
    edge_mask = target_masks["edge_stream"].to(device=y.device, dtype=y.dtype)
    species = resolve_pi_species_order(data_cfg, train_cfg, outputs)
    tail_dim = int(outputs["main_stream_pred"].shape[-1]) - 2 - len(species)
    if tail_dim == 1:
        flow_names = ["Mass_Flow"]
    elif tail_dim == 2:
        flow_names = ["Mass_Flow", "Vol_Flow"]
    elif tail_dim == 3:
        flow_names = ["Mass_Flow", "Mole_Flow", "Vol_Flow"]
    else:
        raise RuntimeError(
            "PI main_stream_pred tail must have 1, 2, or 3 flow columns, "
            f"got width={int(outputs['main_stream_pred'].shape[-1])}, species={len(species)}."
        )
    main_names = ["Temp", "Pres"] + [f"Frac_{s}" for s in species] + flow_names
    main_idx = _column_indices(cols, main_names)
    main_target = _take_columns(y, main_idx)
    main_mask = _expand_edge_mask(edge_mask, main_target.shape[-1], main_target)

    out: dict[str, torch.Tensor | None] = {"main_target": main_target, "main_mask": main_mask}
    for key, col, use_key in (
        ("rho", "Density", "use_rho_loss"),
        ("h", "Enthalpy", "use_h_loss"),
        ("mole_flow", "Mole_Flow", "use_mole_flow_diagnostic"),
        ("volume", "Vol_Flow", "use_volume_loss"),
        ("enthalpy_flow", "Enthalpy_Flow", "use_enthalpy_flow_loss"),
    ):
        enabled = bool(_cfg_get(train_cfg, use_key, False))
        if col in cols:
            val = _take_columns(y, [cols.index(col)])
            mask = _expand_edge_mask(edge_mask, 1, val)
            out[f"{key}_target"] = val
            out[f"{key}_mask"] = mask
        elif enabled:
            raise RuntimeError(f"{use_key}=true but PI target column {col!r} is missing.")
        else:
            out[f"{key}_target"] = None
            out[f"{key}_mask"] = None

    assert outputs["main_stream_pred"].shape == main_target.shape
    assert main_mask.shape == main_target.shape
    if bool(_cfg_get(train_cfg, "use_rho_loss", True)):
        assert outputs["rho_pred"].shape == out["rho_target"].shape
        assert out["rho_mask"].shape == out["rho_target"].shape
    if bool(_cfg_get(train_cfg, "use_h_loss", True)):
        assert outputs["h_pred"].shape == out["h_target"].shape
        assert out["h_mask"].shape == out["h_target"].shape
    num_species = int(outputs["frac_pred"].shape[-1])
    assert outputs["main_stream_pred"].shape[-1] in {
        2 + num_species + 1,
        2 + num_species + 2,
        2 + num_species + 3,
    }
    return out


def _normalizer_col(normalizer: Mapping[str, Any] | None, col: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not normalizer:
        return None
    columns = [str(c) for c in normalizer.get("columns", [])]
    if col not in columns:
        return None
    idx = columns.index(col)
    mean = normalizer.get("mean")
    std = normalizer.get("std")
    if mean is None or std is None:
        return None
    return mean[idx], std[idx]


def _normalizer_vectors_for_names(
    normalizer: Mapping[str, Any] | None,
    names: Sequence[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not normalizer:
        raise RuntimeError("PI fraction loss normalization requires a train-set normalizer.")
    columns = [str(c) for c in normalizer.get("columns", [])]
    mean = normalizer.get("mean")
    std = normalizer.get("std")
    if not columns or mean is None or std is None:
        raise RuntimeError("PI fraction loss normalizer requires columns, mean, and std.")
    indices = _column_indices(columns, names)
    mean_t = torch.as_tensor(mean)
    std_t = torch.as_tensor(std)
    if mean_t.ndim != 1 or std_t.ndim != 1 or mean_t.shape != std_t.shape:
        raise RuntimeError(
            "PI fraction loss normalizer mean/std must be matching 1D tensors, "
            f"got mean={tuple(mean_t.shape)} std={tuple(std_t.shape)}."
        )
    index = torch.tensor(indices, device=mean_t.device, dtype=torch.long)
    frac_mean = mean_t.index_select(0, index).to(device=device, dtype=dtype)
    frac_std = std_t.to(device=mean_t.device).index_select(0, index).to(device=device, dtype=dtype)
    return frac_mean, frac_std


def _inverse_if_needed(x: torch.Tensor, col: str, normalizer: Mapping[str, Any] | None, data_cfg: Any) -> torch.Tensor:
    if not bool(_cfg_get(data_cfg, "normalize_y_edge", False)):
        return x
    ms = _normalizer_col(normalizer, col)
    if ms is None:
        raise RuntimeError(f"Cannot inverse-transform PI column {col!r}; normalizer is missing it.")
    mean, std = ms
    return x * std.to(device=x.device, dtype=x.dtype) + mean.to(device=x.device, dtype=x.dtype)


def _ensure_pi_output_keys(
    outputs: Mapping[str, torch.Tensor],
    *,
    train_cfg: Any,
    data_cfg: Any,
    edge_target_columns: Sequence[str] | None,
    normalizer: Mapping[str, Any] | None,
) -> Mapping[str, torch.Tensor]:
    """Expose direct 12/14D single-head output through the PI loss key contract."""
    if "main_stream_pred" in outputs:
        return outputs
    y_edge_pred = outputs.get("y_edge_pred")
    if y_edge_pred is None:
        return outputs
    cols = list(edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
    raw_species = list(_cfg_get(data_cfg, "species_order", None) or _cfg_get(train_cfg, "species_order", None) or [])
    if raw_species:
        species = [str(s) for s in raw_species]
    else:
        species = [str(c).replace("Frac_", "") for c in cols if str(c).startswith("Frac_")]
    flow_names = [name for name in ("Mass_Flow", "Mole_Flow", "Vol_Flow") if name in cols]
    if "Mass_Flow" not in flow_names:
        raise RuntimeError("single-head PI compatibility requires a Mass_Flow prediction column.")
    if len(flow_names) not in {1, 3}:
        raise RuntimeError(
            "single-head PI compatibility requires either [Mass_Flow] or "
            f"[Mass_Flow, Mole_Flow, Vol_Flow], got {flow_names}."
        )
    main_names = ["Temp", "Pres"] + [f"Frac_{s}" for s in species] + flow_names
    mapped: dict[str, torch.Tensor] = dict(outputs)
    mapped["main_stream_pred"] = _take_columns(y_edge_pred, _column_indices(cols, main_names))
    mapped["T_pred"] = _take_columns(y_edge_pred, [cols.index("Temp")])
    mapped["P_pred"] = _take_columns(y_edge_pred, [cols.index("Pres")])
    frac_columns = [f"Frac_{s}" for s in species]
    frac_raw = _take_columns(y_edge_pred, _column_indices(cols, frac_columns))
    frac_physical = torch.cat(
        [_inverse_if_needed(frac_raw[..., i : i + 1], name, normalizer, data_cfg) for i, name in enumerate(frac_columns)],
        dim=-1,
    )
    mapped["frac_pred"] = frac_physical
    mapped["mass_flow_pred"] = _take_columns(y_edge_pred, [cols.index("Mass_Flow")])
    if "Mole_Flow" in cols:
        mapped["mole_flow_pred"] = _take_columns(y_edge_pred, [cols.index("Mole_Flow")])
    if "Vol_Flow" in cols:
        mapped["volume_flow_pred"] = _take_columns(y_edge_pred, [cols.index("Vol_Flow")])
    if "Density" in cols:
        mapped["rho_pred"] = _take_columns(y_edge_pred, [cols.index("Density")])
    elif bool(_cfg_get(train_cfg, "use_rho_loss", True)):
        raise RuntimeError("single-head PI compatibility requires Density when use_rho_loss=true.")
    if "Enthalpy" in cols:
        mapped["h_pred"] = _take_columns(y_edge_pred, [cols.index("Enthalpy")])
    elif bool(_cfg_get(train_cfg, "use_h_loss", True)):
        raise RuntimeError("single-head PI compatibility requires Enthalpy when use_h_loss=true.")
    return mapped


def _is_fraction_property(name: str) -> bool:
    return str(name).startswith("Frac_")


def _normalize_target_if_needed(
    y: torch.Tensor,
    normalizer: Mapping[str, Any] | None,
    data_cfg: Any,
    edge_target_columns: Sequence[str] | None = None,
) -> torch.Tensor:
    if not bool(_cfg_get(data_cfg, "normalize_y_edge", False)):
        return y
    if not normalizer or normalizer.get("mean") is None or normalizer.get("std") is None:
        raise RuntimeError("normalize_y_edge=true but PI normalizer is missing mean/std.")
    cols = list(edge_target_columns or [])
    mean = torch.as_tensor(normalizer["mean"])
    std = torch.as_tensor(normalizer["std"])
    normalizer_cols = [str(c) for c in normalizer.get("columns", [])]
    if cols:
        if len(cols) != int(y.shape[-1]):
            raise RuntimeError(
                f"edge_target_columns width {len(cols)} does not match target width {int(y.shape[-1])}."
            )
        if normalizer_cols:
            index = torch.tensor(_column_indices(normalizer_cols, cols), device=mean.device, dtype=torch.long)
            mean = mean.index_select(0, index)
            std = std.to(device=mean.device).index_select(0, index)
    if mean.ndim != 1 or std.ndim != 1 or int(mean.shape[0]) != int(y.shape[-1]) or mean.shape != std.shape:
        raise RuntimeError(
            "PI target normalizer shape mismatch: "
            f"target={tuple(y.shape)} mean={tuple(mean.shape)} std={tuple(std.shape)}."
        )
    mean = mean.to(device=y.device, dtype=y.dtype)
    std = std.to(device=y.device, dtype=y.dtype).clamp_min(1.0e-8)
    out = (y - mean.view(*([1] * (y.ndim - 1)), -1)) / std.view(*([1] * (y.ndim - 1)), -1)
    if cols and len(cols) == int(y.shape[-1]):
        # PI fraction heads already emit physical mole fractions.
        frac_idx = [i for i, col in enumerate(cols) if _is_fraction_property(col)]
        if frac_idx:
            index = torch.tensor(frac_idx, device=y.device, dtype=torch.long)
            out = out.clone()
            out.index_copy_(-1, index, y.index_select(-1, index))
    return out


def _prepare_pi_main_loss_tensors(
    *,
    pred_main: torch.Tensor,
    target_main: torch.Tensor,
    mask_main: torch.Tensor,
    outputs: Mapping[str, torch.Tensor],
    train_cfg: Any,
    data_cfg: Any,
    normalizer: Mapping[str, Any] | None,
    global_step: int | None = None,
    total_train_steps: int | None = None,
    collect_debug: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
]:
    fraction_loss_type = resolve_pi_fraction_loss_type(train_cfg)
    use_clr_fraction_loss = fraction_loss_type == "clr"
    use_log_fraction_loss = fraction_loss_type == "log"
    normalize_frac = bool(_cfg_get(train_cfg, "pi_normalize_fraction_loss", False))
    use_zero_flow_mask = bool(_cfg_get(train_cfg, "use_zero_flow_fraction_mask", False))
    use_mass_log_loss = bool(_cfg_get(train_cfg, "use_log1p_mass_flow_loss", False))
    mass_output_space = resolve_pi_mass_flow_output_space(train_cfg)
    use_direct_mass_log = mass_output_space == "log1p"
    use_fraction_component_penalty = bool(_cfg_mapping(train_cfg, "pi_fraction_component_penalty").get("enabled", False))
    if (
        not use_clr_fraction_loss
        and not use_log_fraction_loss
        and not normalize_frac
        and not use_zero_flow_mask
        and not use_mass_log_loss
        and not use_direct_mass_log
        and not use_fraction_component_penalty
    ):
        zero_frac = pred_main[..., 0].float() * 0.0
        valid_frac = torch.zeros_like(zero_frac, dtype=torch.bool)
        return (
            pred_main,
            target_main,
            mask_main,
            torch.ones_like(pred_main),
            zero_frac,
            valid_frac,
            zero_frac,
            valid_frac,
            zero_frac,
            valid_frac,
            zero_frac,
            valid_frac,
            {
            "fraction_loss_type": "legacy",
            "fraction_loss_normalized": False,
            "fraction_clr_loss_enabled": False,
            "fraction_log_loss_enabled": False,
            "frac_penalty_enabled": False,
            "zero_flow_fraction_mask_enabled": False,
            "mass_flow_log1p_loss_enabled": False,
            "mass_flow_direct_log_enabled": False,
            },
        )

    species = resolve_pi_species_order(data_cfg, train_cfg, outputs)
    num_species = int(outputs["frac_pred"].shape[-1])
    if len(species) != num_species:
        raise AssertionError(f"len(species_order)={len(species)} != num_species={num_species}")
    tail_dim = int(pred_main.shape[-1]) - 2 - num_species
    if tail_dim not in {1, 2, 3}:
        raise RuntimeError(
            f"PI main flow tail must have 1, 2, or 3 columns, got tail_dim={tail_dim}."
        )
    mass_tail_slice = slice(2 + num_species, 2 + num_species + 1)
    expected_width = 2 + num_species + tail_dim
    if int(pred_main.shape[-1]) != expected_width or pred_main.shape != target_main.shape:
        raise RuntimeError(
            "PI main loss shape mismatch before fraction normalization: "
            f"pred={tuple(pred_main.shape)} target={tuple(target_main.shape)} expected_width={expected_width}."
        )

    frac_names = [f"Frac_{name}" for name in species]
    frac_slice = slice(2, 2 + num_species)
    pred_frac = pred_main[..., frac_slice].float()
    true_frac = target_main[..., frac_slice].float()
    mask_for_loss = mask_main.float().clone()
    loss_weight_for_loss = torch.ones_like(pred_main, dtype=torch.float32)
    frac_mask = mask_for_loss[..., frac_slice] > 0
    row_valid = frac_mask.any(dim=-1)
    zero_flow_mask_enabled = False
    zero_flow_valid_rows = 0
    zero_flow_masked_rows = 0
    mass_flow_for_mask_min = math.nan
    mass_flow_for_mask_max = math.nan
    if use_zero_flow_mask:
        zero_flow_mask_enabled = True
        eps = float(_cfg_get(train_cfg, "zero_flow_fraction_mask_eps", _cfg_get(train_cfg, "eps", 1.0e-8)))
        if not math.isfinite(eps) or eps < 0.0:
            raise ValueError(f"zero_flow_fraction_mask_eps must be finite and non-negative, got {eps}.")
        mass_target = target_main[..., mass_tail_slice].float()
        try:
            mass_mean, mass_std = _normalizer_vectors_for_names(
                normalizer,
                ["Mass_Flow"],
                device=pred_main.device,
                dtype=torch.float32,
            )
            if bool(_cfg_get(data_cfg, "normalize_y_edge", False)):
                mass_target = mass_target * mass_std.view(*([1] * (mass_target.ndim - 1)), 1) + mass_mean.view(
                    *([1] * (mass_target.ndim - 1)), 1
                )
        except Exception:
            # Fallback for tests or configs without Mass_Flow stats: zero-sum fractions
            # indicate rows where composition is physically undefined.
            mass_target = true_frac.sum(dim=-1, keepdim=True)
        row_valid = (mass_target.squeeze(-1) > eps) & frac_mask.any(dim=-1)
        if use_clr_fraction_loss or use_log_fraction_loss:
            row_valid = row_valid & (true_frac.sum(dim=-1) > eps)
        frac_mask = frac_mask & row_valid.unsqueeze(-1)
        mask_for_loss[..., frac_slice] = mask_for_loss[..., frac_slice] * row_valid.unsqueeze(-1).to(mask_for_loss.dtype)
        if collect_debug:
            zero_flow_valid_rows = int(row_valid.sum().detach().cpu())
            zero_flow_masked_rows = int((frac_mask.any(dim=-1).logical_not() & (mask_main[..., frac_slice] > 0).any(dim=-1)).sum().detach().cpu())
        if collect_debug and mass_target.numel():
            mass_flow_for_mask_min = float(mass_target.min().detach().cpu())
            mass_flow_for_mask_max = float(mass_target.max().detach().cpu())
    elif collect_debug:
        zero_flow_valid_rows = int(row_valid.sum().detach().cpu())

    valid_pred = pred_frac[frac_mask]
    valid_true = true_frac[frac_mask]
    runtime_physical_checks = bool(_cfg_get(train_cfg, "pi_runtime_physical_checks", True))
    if runtime_physical_checks:
        tolerance = 1.0e-4
        if valid_pred.numel() and float(valid_pred.min().detach().cpu()) < -tolerance:
            raise RuntimeError("PI fraction prediction is negative before loss preparation.")
        if (
            not use_clr_fraction_loss
            and valid_pred.numel()
            and float(valid_pred.max().detach().cpu()) > 1.0 + tolerance
        ):
            raise RuntimeError("PI fraction prediction is not in physical [0, 1] scale before loss normalization.")
        if valid_true.numel() and (
            float(valid_true.min().detach().cpu()) < -tolerance
            or float(valid_true.max().detach().cpu()) > 1.0 + tolerance
        ):
            raise RuntimeError("PI fraction target is not in physical [0, 1] scale before loss normalization.")

    pred_frac_sum = pred_frac.sum(dim=-1)
    relu_l1_denominator = outputs.get("relu_l1_denominator")
    relu_l1_all_zero_mask = outputs.get("relu_l1_all_zero_mask")
    if collect_debug and relu_l1_denominator is not None and relu_l1_all_zero_mask is not None:
        denominator = relu_l1_denominator.float().reshape(-1)
        all_zero_mask = relu_l1_all_zero_mask.bool().reshape(-1)
        relu_l1_all_zero_row_count = int(all_zero_mask.sum().detach().cpu())
        relu_l1_all_zero_row_ratio = float(all_zero_mask.float().mean().detach().cpu())
        relu_l1_denominator_min = float(denominator.min().detach().cpu()) if denominator.numel() else math.nan
        relu_l1_denominator_mean = float(denominator.mean().detach().cpu()) if denominator.numel() else math.nan
    else:
        relu_l1_all_zero_row_count = 0
        relu_l1_all_zero_row_ratio = 0.0
        relu_l1_denominator_min = math.nan
        relu_l1_denominator_mean = math.nan
    pred_frac_sum_mean = float(pred_frac_sum.mean().detach().cpu()) if collect_debug and pred_frac_sum.numel() else math.nan
    pred_frac_sum_min = float(pred_frac_sum.min().detach().cpu()) if collect_debug and pred_frac_sum.numel() else math.nan
    pred_frac_sum_max = float(pred_frac_sum.max().detach().cpu()) if collect_debug and pred_frac_sum.numel() else math.nan
    pred_frac_negative_count = int((pred_frac < 0.0).sum().detach().cpu()) if collect_debug else 0
    pred_frac_nan_count = int(torch.isnan(pred_frac).sum().detach().cpu()) if collect_debug else 0
    pred_frac_inf_count = int(torch.isinf(pred_frac).sum().detach().cpu()) if collect_debug else 0

    frac_loss_mean_min = math.nan
    frac_loss_mean_max = math.nan
    frac_loss_std_min = math.nan
    frac_loss_std_max = math.nan
    if normalize_frac:
        frac_mean, frac_std = _normalizer_vectors_for_names(
            normalizer,
            frac_names,
            device=pred_main.device,
            dtype=torch.float32,
        )
        std_min = float(_cfg_get(train_cfg, "pi_fraction_loss_std_min", 1.0e-3))
        if not math.isfinite(std_min) or std_min <= 0.0:
            raise ValueError(f"pi_fraction_loss_std_min must be finite and positive, got {std_min}.")
        frac_std = frac_std.clamp_min(std_min)
        if frac_mean.shape != (num_species,) or frac_std.shape != (num_species,):
            raise RuntimeError(
                f"PI fraction normalizer shape mismatch: mean={tuple(frac_mean.shape)} "
                f"std={tuple(frac_std.shape)} expected=({num_species},)."
            )
        view_shape = [1] * (pred_frac.ndim - 1) + [num_species]
        frac_mean_view = frac_mean.view(*view_shape)
        frac_std_view = frac_std.view(*view_shape)
        pred_frac_for_loss = (pred_frac - frac_mean_view) / frac_std_view
        true_frac_for_loss = (true_frac - frac_mean_view) / frac_std_view
        if collect_debug:
            frac_loss_mean_min = float(frac_mean.min().detach().cpu())
            frac_loss_mean_max = float(frac_mean.max().detach().cpu())
            frac_loss_std_min = float(frac_std.min().detach().cpu())
            frac_loss_std_max = float(frac_std.max().detach().cpu())
    else:
        pred_frac_for_loss = pred_frac
        true_frac_for_loss = true_frac

    target_frac_sum = true_frac.sum(dim=-1)
    valid_target_frac_sum = target_frac_sum[row_valid]
    if collect_debug and valid_target_frac_sum.numel():
        target_frac_sum_mean = float(valid_target_frac_sum.mean().detach().cpu())
        target_frac_sum_min = float(valid_target_frac_sum.min().detach().cpu())
        target_frac_sum_max = float(valid_target_frac_sum.max().detach().cpu())
        target_frac_sum_abs_error = (valid_target_frac_sum - 1.0).abs()
        target_frac_sum_abs_error_mean = float(target_frac_sum_abs_error.mean().detach().cpu())
        target_frac_sum_bad_count = int((target_frac_sum_abs_error > 1.0e-3).sum().detach().cpu())
    else:
        target_frac_sum_mean = math.nan
        target_frac_sum_min = math.nan
        target_frac_sum_max = math.nan
        target_frac_sum_abs_error_mean = math.nan
        target_frac_sum_bad_count = 0

    if use_clr_fraction_loss:
        clr_eps = float(_cfg_get(train_cfg, "pi_fraction_clr_eps", 1.0e-6))
        pred_clr = _clr_transform(pred_frac, clr_eps)
        true_clr = _clr_transform(true_frac, clr_eps)
        clr_component_mask = (mask_main[..., frac_slice] > 0) & row_valid.unsqueeze(-1)
        clr_loss_raw = F.smooth_l1_loss(pred_clr, true_clr, reduction="none")
        clr_mask_float = clr_component_mask.to(device=clr_loss_raw.device, dtype=clr_loss_raw.dtype)
        clr_count = clr_mask_float.sum(dim=-1)
        valid_frac_clr = clr_count > 0
        loss_frac_clr = (clr_loss_raw * clr_mask_float).sum(dim=-1) / clr_count.clamp_min(1.0)
        loss_frac_clr = torch.where(
            valid_frac_clr,
            loss_frac_clr,
            pred_frac.sum(dim=-1) * 0.0,
        )
        frac_sum = pred_frac.sum(dim=-1)
        closure_loss_raw = F.smooth_l1_loss(frac_sum, torch.ones_like(frac_sum), reduction="none")
        valid_frac_closure = row_valid
        loss_frac_closure = torch.where(
            valid_frac_closure,
            closure_loss_raw,
            pred_frac.sum(dim=-1) * 0.0,
        )
        # CLR is a separate term; do not also include Frac_* in the main 10D loss.
        mask_for_loss[..., frac_slice] = 0.0
    else:
        clr_eps = math.nan
        pred_clr = None
        true_clr = None
        loss_frac_clr = pred_frac.sum(dim=-1) * 0.0
        valid_frac_clr = torch.zeros_like(loss_frac_clr, dtype=torch.bool)
        loss_frac_closure = pred_frac.sum(dim=-1) * 0.0
        valid_frac_closure = torch.zeros_like(loss_frac_closure, dtype=torch.bool)

    if use_log_fraction_loss:
        log_eps = float(_cfg_get(train_cfg, "pi_fraction_log_eps", 1.0e-6))
        pred_frac_log = _log_fraction_transform(pred_frac, log_eps)
        true_frac_log = _log_fraction_transform(true_frac, log_eps)
        log_component_mask = (mask_main[..., frac_slice] > 0) & row_valid.unsqueeze(-1)
        log_loss_raw = F.smooth_l1_loss(pred_frac_log, true_frac_log, reduction="none")
        log_mask_float = log_component_mask.to(device=log_loss_raw.device, dtype=log_loss_raw.dtype)
        log_count = log_mask_float.sum(dim=-1)
        valid_frac_log = log_count > 0
        loss_frac_log = (log_loss_raw * log_mask_float).sum(dim=-1) / log_count.clamp_min(1.0)
        loss_frac_log = torch.where(
            valid_frac_log,
            loss_frac_log,
            pred_frac.sum(dim=-1) * 0.0,
        )
        # Log regression is a separate term; do not also supervise Frac_* in main 10D.
        mask_for_loss[..., frac_slice] = 0.0
    else:
        log_eps = math.nan
        pred_frac_log = None
        true_frac_log = None
        loss_frac_log = pred_frac.sum(dim=-1) * 0.0
        valid_frac_log = torch.zeros_like(loss_frac_log, dtype=torch.bool)

    loss_frac_penalty, valid_frac_penalty, frac_penalty_debug = _compute_fraction_component_penalty(
        pred_frac=pred_frac,
        true_frac=true_frac,
        frac_mask=frac_mask,
        row_valid=row_valid,
        species=species,
        train_cfg=train_cfg,
        global_step=global_step,
        total_train_steps=total_train_steps,
        collect_debug=collect_debug,
    )

    mass_log_loss_enabled = False
    mass_log_mean_value = math.nan
    mass_log_std_value = math.nan
    mass_pred_physical_min = math.nan
    mass_pred_physical_max = math.nan
    mass_target_physical_min = math.nan
    mass_target_physical_max = math.nan
    mass_target_physical_mean = math.nan
    mass_target_physical_std = math.nan
    mass_target_log_min = math.nan
    mass_target_log_max = math.nan
    mass_target_log_mean = math.nan
    mass_target_log_std = math.nan
    mass_pred_log_min = math.nan
    mass_pred_log_max = math.nan
    mass_pred_log_mean = math.nan
    mass_pred_log_std = math.nan
    mass_log_loss_weight = 1.0
    pred_tail_for_loss = pred_main[..., 2 + num_species :].float()
    target_tail_for_loss = target_main[..., 2 + num_species :].float()

    if use_direct_mass_log:
        if int(pred_tail_for_loss.shape[-1]) not in {1, 2, 3} or pred_tail_for_loss.shape != target_tail_for_loss.shape:
            raise RuntimeError(
                "PI direct-log Mass_Flow expects one, two, or three main tail columns, "
                f"got pred_tail={tuple(pred_tail_for_loss.shape)} target_tail={tuple(target_tail_for_loss.shape)}."
            )
        target_mass_physical = target_tail_for_loss[..., 0:1]
        if bool(_cfg_get(data_cfg, "normalize_y_edge", False)):
            mass_mean, mass_std = _normalizer_vectors_for_names(
                normalizer,
                ["Mass_Flow"],
                device=pred_main.device,
                dtype=torch.float32,
            )
            view_shape = [1] * (target_mass_physical.ndim - 1) + [1]
            target_mass_physical = (
                target_mass_physical * mass_std.view(*view_shape) + mass_mean.view(*view_shape)
            )
        target_mass_log = transform_mass_flow(
            target_mass_physical,
            train_cfg=train_cfg,
        )
        if not torch.isfinite(pred_tail_for_loss).all():
            raise RuntimeError("PI direct-log Mass_Flow prediction contains NaN or Inf in the loss path.")
        if not torch.isfinite(target_mass_log).all():
            raise RuntimeError("PI direct-log Mass_Flow target contains NaN or Inf after log1p.")
        target_tail_for_loss = torch.cat(
            (target_mass_log, target_tail_for_loss[..., 1:]),
            dim=-1,
        )
        mass_log_loss_weight = float(_cfg_get(train_cfg, "mass_flow_log_loss_weight", 1.0))
        if not math.isfinite(mass_log_loss_weight) or mass_log_loss_weight <= 0.0:
            raise ValueError(
                f"mass_flow_log_loss_weight must be finite and positive, got {mass_log_loss_weight}."
            )
        loss_weight_for_loss[..., mass_tail_slice] = mass_log_loss_weight
        if collect_debug and target_mass_physical.numel():
            mass_target_physical_min = float(target_mass_physical.min().detach().cpu())
            mass_target_physical_max = float(target_mass_physical.max().detach().cpu())
            mass_target_physical_mean = float(target_mass_physical.mean().detach().cpu())
            mass_target_physical_std = float(target_mass_physical.std(unbiased=False).detach().cpu())
            mass_target_log_min = float(target_mass_log.min().detach().cpu())
            mass_target_log_max = float(target_mass_log.max().detach().cpu())
            mass_target_log_mean = float(target_mass_log.mean().detach().cpu())
            mass_target_log_std = float(target_mass_log.std(unbiased=False).detach().cpu())
            mass_pred_log = pred_tail_for_loss[..., 0:1]
            mass_pred_log_min = float(mass_pred_log.min().detach().cpu())
            mass_pred_log_max = float(mass_pred_log.max().detach().cpu())
            mass_pred_log_mean = float(mass_pred_log.mean().detach().cpu())
            mass_pred_log_std = float(mass_pred_log.std(unbiased=False).detach().cpu())
    elif use_mass_log_loss:
        mass_log_loss_enabled = True
        if int(pred_tail_for_loss.shape[-1]) not in {1, 2, 3} or pred_tail_for_loss.shape != target_tail_for_loss.shape:
            raise RuntimeError(
                "PI log1p Mass_Flow loss expects one, two, or three main tail columns, "
                f"got pred_tail={tuple(pred_tail_for_loss.shape)} target_tail={tuple(target_tail_for_loss.shape)}."
            )
        mass_log_mean_raw = None if normalizer is None else normalizer.get("mass_flow_log1p_mean")
        mass_log_std_raw = None if normalizer is None else normalizer.get("mass_flow_log1p_std")
        if mass_log_mean_raw is None or mass_log_std_raw is None:
            raise RuntimeError("use_log1p_mass_flow_loss=true requires train-set mass_flow_log1p_mean/std.")
        mass_log_mean = torch.as_tensor(mass_log_mean_raw, device=pred_main.device, dtype=torch.float32).reshape(())
        mass_log_std = torch.as_tensor(mass_log_std_raw, device=pred_main.device, dtype=torch.float32).reshape(())
        std_min = float(_cfg_get(train_cfg, "log1p_mass_flow_loss_std_min", 1.0e-6))
        if not math.isfinite(std_min) or std_min <= 0.0:
            raise ValueError(f"log1p_mass_flow_loss_std_min must be finite and positive, got {std_min}.")
        mass_log_std = mass_log_std.clamp_min(std_min)
        pred_mass_physical = pred_tail_for_loss[..., 0:1]
        target_mass_physical = target_tail_for_loss[..., 0:1]
        if bool(_cfg_get(data_cfg, "normalize_y_edge", False)):
            mass_mean, mass_std = _normalizer_vectors_for_names(
                normalizer,
                ["Mass_Flow"],
                device=pred_main.device,
                dtype=torch.float32,
            )
            view_shape = [1] * (pred_mass_physical.ndim - 1) + [1]
            pred_mass_physical = pred_mass_physical * mass_std.view(*view_shape) + mass_mean.view(*view_shape)
            target_mass_physical = target_mass_physical * mass_std.view(*view_shape) + mass_mean.view(*view_shape)
        if collect_debug and pred_mass_physical.numel():
            mass_pred_physical_min = float(pred_mass_physical.min().detach().cpu())
            mass_pred_physical_max = float(pred_mass_physical.max().detach().cpu())
            mass_target_physical_min = float(target_mass_physical.min().detach().cpu())
            mass_target_physical_max = float(target_mass_physical.max().detach().cpu())
        pred_mass_log = transform_mass_flow(
            pred_mass_physical,
            train_cfg=train_cfg,
        )
        target_mass_log = transform_mass_flow(
            target_mass_physical,
            train_cfg=train_cfg,
        )
        pred_tail_for_loss = torch.cat(
            ((pred_mass_log - mass_log_mean) / mass_log_std, pred_tail_for_loss[..., 1:]),
            dim=-1,
        )
        target_tail_for_loss = torch.cat(
            ((target_mass_log - mass_log_mean) / mass_log_std, target_tail_for_loss[..., 1:]),
            dim=-1,
        )
        if collect_debug:
            mass_log_mean_value = float(mass_log_mean.detach().cpu())
            mass_log_std_value = float(mass_log_std.detach().cpu())

    pred_for_loss = torch.cat(
        (pred_main[..., :2].float(), pred_frac_for_loss, pred_tail_for_loss),
        dim=-1,
    )
    target_for_loss = torch.cat(
        (target_main[..., :2].float(), true_frac_for_loss, target_tail_for_loss),
        dim=-1,
    )
    if pred_for_loss.shape != pred_main.shape or target_for_loss.shape != target_main.shape:
        raise AssertionError("PI fraction loss normalization changed the main tensor shape.")
    if mask_for_loss.shape != mask_main.shape:
        raise AssertionError("PI zero-flow fraction mask changed the main mask shape.")
    if loss_weight_for_loss.shape != pred_main.shape:
        raise AssertionError("PI Mass_Flow loss weighting changed the main tensor shape.")
    if normalize_frac and (frac_mean.device != pred_for_loss.device or frac_std.device != pred_for_loss.device):
        raise AssertionError("PI fraction loss normalizer tensors are on the wrong device.")

    return (
        pred_for_loss,
        target_for_loss,
        mask_for_loss,
        loss_weight_for_loss,
        loss_frac_clr,
        valid_frac_clr,
        loss_frac_log,
        valid_frac_log,
        loss_frac_closure,
        valid_frac_closure,
        loss_frac_penalty,
        valid_frac_penalty,
        {
        "fraction_loss_type": fraction_loss_type,
        "fraction_loss_normalized": normalize_frac,
        "fraction_clr_loss_enabled": use_clr_fraction_loss,
        "fraction_log_loss_enabled": use_log_fraction_loss,
        **frac_penalty_debug,
        "fraction_clr_eps": clr_eps,
        "fraction_log_eps": log_eps,
        "fraction_clr_valid_sample_count": int(valid_frac_clr.sum().detach().cpu()) if collect_debug else 0,
        "fraction_log_valid_sample_count": int(valid_frac_log.sum().detach().cpu()) if collect_debug else 0,
        "fraction_clr_loss_mean": _mean_valid(loss_frac_clr, valid_frac_clr) if collect_debug else 0.0,
        "fraction_log_loss_mean": _mean_valid(loss_frac_log, valid_frac_log) if collect_debug else 0.0,
        "fraction_closure_loss_mean": _mean_valid(loss_frac_closure, valid_frac_closure) if collect_debug else 0.0,
        "fraction_closure_valid_sample_count": int(valid_frac_closure.sum().detach().cpu()) if collect_debug else 0,
        "fraction_clr_pred_component_mean_abs_max": (
            float(pred_clr.mean(dim=-1).abs().max().detach().cpu())
            if collect_debug and pred_clr is not None and pred_clr.numel()
            else math.nan
        ),
        "fraction_clr_target_component_mean_abs_max": (
            float(true_clr.mean(dim=-1).abs().max().detach().cpu())
            if collect_debug and true_clr is not None and true_clr.numel()
            else math.nan
        ),
        "zero_flow_fraction_mask_enabled": zero_flow_mask_enabled,
        "zero_flow_fraction_valid_rows": zero_flow_valid_rows,
        "zero_flow_fraction_masked_rows": zero_flow_masked_rows,
        "zero_flow_fraction_mass_min": mass_flow_for_mask_min,
        "zero_flow_fraction_mass_max": mass_flow_for_mask_max,
        "target_fraction_sum_mean": target_frac_sum_mean,
        "target_fraction_sum_min": target_frac_sum_min,
        "target_fraction_sum_max": target_frac_sum_max,
        "target_fraction_sum_abs_error_mean": target_frac_sum_abs_error_mean,
        "target_fraction_sum_abs_error_gt_1e3_count": target_frac_sum_bad_count,
        "target_fraction_sum_valid_row_count": int(valid_target_frac_sum.numel()) if collect_debug else 0,
        "relu_l1_all_zero_row_count": relu_l1_all_zero_row_count,
        "relu_l1_all_zero_row_ratio": relu_l1_all_zero_row_ratio,
        "relu_l1_denominator_min": relu_l1_denominator_min,
        "relu_l1_denominator_mean": relu_l1_denominator_mean,
        "pred_frac_sum_mean": pred_frac_sum_mean,
        "pred_frac_sum_min": pred_frac_sum_min,
        "pred_frac_sum_max": pred_frac_sum_max,
        "pred_frac_negative_count": pred_frac_negative_count,
        "pred_frac_nan_count": pred_frac_nan_count,
        "pred_frac_inf_count": pred_frac_inf_count,
        "mass_flow_log1p_loss_enabled": mass_log_loss_enabled,
        "mass_flow_log1p_mean": mass_log_mean_value,
        "mass_flow_log1p_std": mass_log_std_value,
        "mass_flow_log1p_pred_physical_min": mass_pred_physical_min,
        "mass_flow_log1p_pred_physical_max": mass_pred_physical_max,
        "mass_flow_log1p_target_physical_min": mass_target_physical_min,
        "mass_flow_log1p_target_physical_max": mass_target_physical_max,
        "mass_flow_direct_log_enabled": use_direct_mass_log,
        "mass_flow_output_space": mass_output_space,
        "mass_flow_transform": resolve_mass_flow_transform(train_cfg),
        "mass_flow_log_tau": resolve_mass_flow_log_tau(train_cfg),
        "mass_flow_log_scale": resolve_mass_flow_log_scale(train_cfg),
        "mass_flow_log_eps": resolve_mass_flow_log_eps(train_cfg),
        "mass_flow_log_loss_weight": mass_log_loss_weight,
        "mass_flow_direct_target_physical_min": mass_target_physical_min,
        "mass_flow_direct_target_physical_max": mass_target_physical_max,
        "mass_flow_direct_target_physical_mean": mass_target_physical_mean,
        "mass_flow_direct_target_physical_std": mass_target_physical_std,
        "mass_flow_direct_target_log_min": mass_target_log_min,
        "mass_flow_direct_target_log_max": mass_target_log_max,
        "mass_flow_direct_target_log_mean": mass_target_log_mean,
        "mass_flow_direct_target_log_std": mass_target_log_std,
        "mass_flow_direct_pred_log_min": mass_pred_log_min,
        "mass_flow_direct_pred_log_max": mass_pred_log_max,
        "mass_flow_direct_pred_log_mean": mass_pred_log_mean,
        "mass_flow_direct_pred_log_std": mass_pred_log_std,
        "frac_pred_physical_min": float(valid_pred.min().detach().cpu()) if collect_debug and valid_pred.numel() else math.nan,
        "frac_pred_physical_max": float(valid_pred.max().detach().cpu()) if collect_debug and valid_pred.numel() else math.nan,
        "frac_target_physical_min": float(valid_true.min().detach().cpu()) if collect_debug and valid_true.numel() else math.nan,
        "frac_target_physical_max": float(valid_true.max().detach().cpu()) if collect_debug and valid_true.numel() else math.nan,
        "frac_loss_mean_min": frac_loss_mean_min,
        "frac_loss_mean_max": frac_loss_mean_max,
        "frac_loss_std_min": frac_loss_std_min,
        "frac_loss_std_max": frac_loss_std_max,
        },
    )


def _mw_tensor(species_order: Sequence[str], data_cfg: Any, train_cfg: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    raw = dict(DEFAULT_MOLECULAR_WEIGHT_G_PER_MOL)
    raw.update({str(k): float(v) for k, v in (dict(_cfg_get(data_cfg, "molecular_weight", {}) or {})).items()})
    missing = [s for s in species_order if s not in raw]
    if missing:
        raise RuntimeError(f"Missing molecular_weight entries for species: {missing}")
    vals = torch.tensor([float(raw[s]) for s in species_order], device=device, dtype=dtype)
    explicit_scale = _cfg_get(train_cfg, "mw_unit_scale", None)
    if explicit_scale is not None:
        scale = float(explicit_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"mw_unit_scale must be finite and positive, got {scale}.")
        vals = vals * scale
    else:
        unit = str(_cfg_get(train_cfg, "molecular_weight_unit", "g_per_mol"))
        if unit == "g_per_mol":
            vals = vals / 1000.0
        elif unit in {"kg_per_mol", "kg_per_kmol"}:
            pass
        else:
            raise ValueError(f"Unsupported molecular_weight_unit={unit!r}")
    return vals


def build_pi_physical_outputs(
    outputs: Mapping[str, torch.Tensor],
    *,
    train_cfg: Any,
    data_cfg: Any,
    normalizer: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor | None]:
    main = outputs["main_stream_pred"]
    num_species = int(outputs["frac_pred"].shape[-1])
    species = resolve_pi_species_order(data_cfg, train_cfg, outputs)
    T = _inverse_if_needed(outputs["T_pred"], "Temp", normalizer, data_cfg)
    P = _inverse_if_needed(outputs["P_pred"], "Pres", normalizer, data_cfg)
    # A7 intentionally passes raw ReLU fractions through unchanged. In that
    # ablation mw_bar is diagnostic, not a physically closed mixture average.
    frac = outputs["frac_pred"]
    mass_flow = decode_pi_mass_flow_prediction(
        outputs["mass_flow_pred"],
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        normalizer=normalizer,
    )
    rho = (
        _inverse_if_needed(outputs["rho_pred"], "Density", normalizer, data_cfg)
        if "rho_pred" in outputs
        else None
    )
    h = (
        _inverse_if_needed(outputs["h_pred"], "Enthalpy", normalizer, data_cfg)
        if "h_pred" in outputs
        else None
    )
    eps = float(_cfg_get(train_cfg, "eps", 1.0e-8))
    mw = _mw_tensor(species, data_cfg, train_cfg, main.device, main.dtype)
    mw_bar = (frac * mw.view(*([1] * (frac.ndim - 1)), num_species)).sum(dim=-1, keepdim=True)
    if "mole_flow_pred" in outputs:
        mole_flow = _inverse_if_needed(outputs["mole_flow_pred"], "Mole_Flow", normalizer, data_cfg)
    else:
        mole_flow = mass_flow / mw_bar.clamp_min(eps)
    species_mole_flow = mole_flow * frac
    volume_unit_scale = float(_cfg_get(train_cfg, "volume_unit_scale", 1.0))
    if not math.isfinite(volume_unit_scale) or volume_unit_scale <= 0.0:
        raise ValueError(f"volume_unit_scale must be finite and positive, got {volume_unit_scale}.")
    if "volume_flow_pred" in outputs:
        volume_flow = _inverse_if_needed(outputs["volume_flow_pred"], "Vol_Flow", normalizer, data_cfg)
    elif rho is not None:
        volume_flow = mass_flow / rho.clamp_min(eps) * volume_unit_scale
    else:
        volume_flow = None
    enthalpy_flow = None
    if h is not None:
        h_basis = str(_cfg_get(train_cfg, "h_basis", "mass_specific"))
        if h_basis == "mass_specific":
            enthalpy_flow = mass_flow * h
        elif h_basis == "molar":
            enthalpy_flow = mole_flow * h
        else:
            raise ValueError(f"Unsupported h_basis={h_basis!r}")
    return {
        "T": T,
        "P": P,
        "frac": frac,
        "mass_flow": mass_flow,
        "rho": rho,
        "h": h,
        "mw_bar": mw_bar,
        "mole_flow": mole_flow,
        "species_mole_flow": species_mole_flow,
        "volume_flow": volume_flow,
        "enthalpy_flow": enthalpy_flow,
    }


def build_pi_physical_targets(
    pi_targets: Mapping[str, torch.Tensor | None],
    *,
    train_cfg: Any,
    data_cfg: Any,
    normalizer: Mapping[str, Any] | None = None,
) -> dict[str, torch.Tensor | None]:
    def inv(key: str, col: str) -> torch.Tensor | None:
        val = pi_targets.get(f"{key}_target")
        if val is None:
            return None
        return _inverse_if_needed(val, col, normalizer, data_cfg)

    return {
        "rho": inv("rho", "Density"),
        "h": inv("h", "Enthalpy"),
        "mole_flow": inv("mole_flow", "Mole_Flow"),
        "volume_flow": inv("volume", "Vol_Flow"),
        "enthalpy_flow": inv("enthalpy_flow", "Enthalpy_Flow"),
    }


@dataclass
class _RegressionMoments:
    n: int = 0
    sum_true: float = 0.0
    sum_pred: float = 0.0
    sum_true_sq: float = 0.0
    sum_pred_sq: float = 0.0
    sum_true_pred: float = 0.0
    sse: float = 0.0
    sae: float = 0.0
    smape_sum: float = 0.0

    def update(self, true: torch.Tensor, pred: torch.Tensor, valid: torch.Tensor) -> None:
        t = true[valid].detach().double()
        p = pred[valid].detach().double()
        if t.numel() == 0:
            return
        err = p - t
        self.n += int(t.numel())
        self.sum_true += float(t.sum().item())
        self.sum_pred += float(p.sum().item())
        self.sum_true_sq += float((t * t).sum().item())
        self.sum_pred_sq += float((p * p).sum().item())
        self.sum_true_pred += float((t * p).sum().item())
        self.sse += float((err * err).sum().item())
        self.sae += float(err.abs().sum().item())
        # Symmetric mean absolute percentage error is accumulated before the
        # final mean so metric collection stays streaming-friendly.  The
        # epsilon makes the 0/0 case finite and matches target_v4_metrics.
        den = t.abs() + p.abs() + 1.0e-8
        self.smape_sum += float((2.0 * err.abs() / den).sum().item())

    def row(self, property_name: str) -> dict[str, Any]:
        n = int(self.n)
        true_mean = self.sum_true / n if n else math.nan
        pred_mean = self.sum_pred / n if n else math.nan
        sst = max(self.sum_true_sq - n * true_mean * true_mean, 0.0) if n else 0.0
        true_var = max(self.sum_true_sq / n - true_mean * true_mean, 0.0) if n else math.nan
        pred_var = max(self.sum_pred_sq / n - pred_mean * pred_mean, 0.0) if n else math.nan
        constant_target = n >= 2 and sst <= R2_SST_EPS
        return {
            "split": "val",
            "metric_scope": "all_supervised_edges_by_property",
            "property_name": property_name,
            "prediction_kind": "derived" if property_name in {"Mole_Flow", "Vol_Flow"} else "direct_head",
            "n": n,
            "SST": sst,
            "SSE": self.sse,
            "MAE": self.sae / n if n else math.nan,
            "sMAPE_pct": self.smape_sum / n * 100.0 if n else math.nan,
            "RMSE": math.sqrt(self.sse / n) if n else math.nan,
            "R2": (
                math.nan
                if n < 2
                else CONSTANT_TARGET_R2_VALUE
                if constant_target
                else 1.0 - self.sse / sst
            ),
            "true_mean": true_mean,
            "true_std": math.sqrt(true_var) if n else math.nan,
            "pred_mean": pred_mean,
            "pred_std": math.sqrt(pred_var) if n else math.nan,
            "mean_signed_error": pred_mean - true_mean if n else math.nan,
        }

    def calibrated_row(self, property_name: str, *, scope: str) -> dict[str, Any]:
        base = self.row(property_name)
        n = int(self.n)
        a = math.nan
        b = math.nan
        calibrated_r2 = math.nan
        if n >= 2:
            true_mean = self.sum_true / n
            pred_mean = self.sum_pred / n
            pred_var_sum = self.sum_pred_sq - n * pred_mean * pred_mean
            true_var_sum = self.sum_true_sq - n * true_mean * true_mean
            cov_sum = self.sum_true_pred - n * true_mean * pred_mean
            if pred_var_sum > 1.0e-12 and true_var_sum > 1.0e-12:
                a = cov_sum / pred_var_sum
                b = true_mean - a * pred_mean
                calibrated_sse = true_var_sum - (cov_sum * cov_sum / pred_var_sum)
                calibrated_sse = max(float(calibrated_sse), 0.0)
                calibrated_r2 = 1.0 - calibrated_sse / true_var_sum
        return {
            "scope": scope,
            "split": base.get("split", "val"),
            "metric_type": "oracle_affine_calibration",
            "property": property_name,
            "raw_r2": base.get("R2", math.nan),
            "calibrated_r2": calibrated_r2,
            "a": a,
            "b": b,
            "n": n,
        }

    def corr2_row(self, property_name: str, *, scope: str, min_n: int = 1) -> dict[str, Any]:
        base = self.row(property_name)
        n = int(self.n)
        corr2 = math.nan
        if n >= max(2, int(min_n)):
            true_mean = self.sum_true / n
            pred_mean = self.sum_pred / n
            true_var_sum = self.sum_true_sq - n * true_mean * true_mean
            pred_var_sum = self.sum_pred_sq - n * pred_mean * pred_mean
            cov_sum = self.sum_true_pred - n * true_mean * pred_mean
            if true_var_sum > 1.0e-12 and pred_var_sum > 1.0e-12:
                corr = cov_sum / math.sqrt(true_var_sum * pred_var_sum)
                corr2 = float(corr * corr)
        return {
            "scope": scope,
            "split": base.get("split", "val"),
            "metric_type": "diagnostic_corr2",
            "property": property_name,
            "raw_r2": base.get("R2", math.nan),
            "corr2": corr2,
            "n": n,
        }

    def diagnostic_calibrated_row(self, property_name: str, *, scope: str, min_n: int = 1) -> dict[str, Any]:
        row = self.calibrated_row(property_name, scope=scope)
        row["metric_type"] = "diagnostic_affine_calibration"
        if int(row.get("n", 0)) < int(min_n):
            row["calibrated_r2"] = math.nan
            row["a"] = math.nan
            row["b"] = math.nan
        return row

    def transformed_row(
        self,
        property_name: str,
        *,
        scope: str,
        raw_r2: float = math.nan,
        min_n: int = 1,
    ) -> dict[str, Any]:
        base = self.row(property_name)
        corr_row = self.corr2_row(property_name, scope=scope, min_n=min_n)
        transformed_r2 = float(base.get("R2", math.nan)) if int(self.n) >= int(min_n) else math.nan
        return {
            "scope": scope,
            "split": base.get("split", "val"),
            "metric_type": "diagnostic_transformed_space",
            "property": property_name,
            "transform": "log1p_clamp_nonnegative",
            "raw_r2": raw_r2,
            "transformed_r2": transformed_r2,
            "transformed_corr2": corr_row.get("corr2", math.nan),
            "n": int(self.n),
        }


@dataclass
class _ThresholdOracleStat:
    threshold_type: str
    threshold_label: str
    quantile: float = math.nan
    moments: _RegressionMoments = field(default_factory=_RegressionMoments)
    threshold_sum: float = 0.0
    threshold_count: int = 0

    def update(self, true: torch.Tensor, pred: torch.Tensor, valid: torch.Tensor, *, threshold: float) -> None:
        self.threshold_sum += float(threshold)
        self.threshold_count += 1
        self.moments.update(true, pred, valid & (true >= float(threshold)))

    def row(self, property_name: str, *, scope: str, raw_r2: float) -> dict[str, Any]:
        base = self.moments.row(property_name)
        threshold = self.threshold_sum / self.threshold_count if self.threshold_count else math.nan
        return {
            "scope": scope,
            "split": base.get("split", "val"),
            "metric_type": "oracle_threshold_sweep",
            "property": property_name,
            "threshold_type": self.threshold_type,
            "best_r2": base.get("R2", math.nan),
            "best_threshold": threshold,
            "best_quantile": self.quantile,
            "best_n": int(base.get("n", 0)),
            "raw_r2": raw_r2,
            "raw_n": int(base.get("n", 0)),
            "num_thresholds_evaluated": int(self.threshold_count > 0),
        }


class PIAllEdgePropertyR2Accumulator:
    """Streaming original-scale R2 diagnostics for every supervised PI edge."""

    def __init__(self, train_cfg: Any = None, *, store_scatter_samples: bool | None = None) -> None:
        self.train_cfg = train_cfg
        self.stats: dict[str, _RegressionMoments] = {}
        self.relevant_stats: dict[str, _RegressionMoments] = {}
        self.prediction_kinds: dict[str, str] = {}
        self.relevance_enabled = bool(_cfg_get(train_cfg, "metric_relevance_enabled", False))
        self.diagnostic_enabled = bool(_cfg_get(train_cfg, "save_diagnostic_display_metrics", False))
        self.diagnostic_corr2_enabled = bool(_cfg_get(train_cfg, "diagnostic_corr2_enabled", True))
        self.diagnostic_calibrated_enabled = bool(_cfg_get(train_cfg, "diagnostic_calibrated_r2_enabled", True))
        self.diagnostic_transformed_enabled = bool(_cfg_get(train_cfg, "diagnostic_transformed_r2_enabled", True))
        self.diagnostic_fraction_postprocess_enabled = bool(
            _cfg_get(train_cfg, "diagnostic_fraction_postprocess_enabled", True)
        )
        self.diagnostic_min_samples = max(1, int(_cfg_get(train_cfg, "diagnostic_min_samples", 100)))
        self.display_log_flow_stats: dict[str, _RegressionMoments] = {}
        self.mass_flow_tail_stats: dict[str, _RegressionMoments] = {}
        self.mass_flow_tail_thresholds: dict[str, float] = {}
        self.mass_flow_p95_pred_true_ratios: list[float] = []
        self.mass_flow_p95_signed_error_sum = 0.0
        self.mass_flow_p95_signed_error_count = 0
        self.display_fraction_rows: list[dict[str, Any]] = []
        self._display_row_group_counter = 0
        self.store_oracle_samples = bool(_cfg_get(train_cfg, "save_oracle_diagnostic_metrics", False))
        self.oracle_threshold_enabled = bool(_cfg_get(train_cfg, "oracle_threshold_sweep_enabled", True))
        self.oracle_calibration_enabled = bool(_cfg_get(train_cfg, "oracle_calibration_enabled", True))
        self.oracle_fraction_postprocess_enabled = bool(_cfg_get(train_cfg, "oracle_fraction_postprocess_enabled", True))
        self.oracle_min_samples = max(1, int(_cfg_get(train_cfg, "oracle_min_samples", 100)))
        raw_thresholds = _cfg_get(
            train_cfg,
            "oracle_threshold_candidates",
            [0.0, 1.0e-6, 1.0e-4, 1.0e-3, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6],
        )
        self.oracle_abs_thresholds = sorted(
            {
                float(v)
                for v in (raw_thresholds if not isinstance(raw_thresholds, str) else raw_thresholds.split(","))
                if math.isfinite(float(v)) and float(v) >= 0.0
            }
        ) or [0.0]
        raw_quantiles = _cfg_get(train_cfg, "oracle_quantile_candidates", [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
        self.oracle_quantiles = sorted(
            {
                float(v)
                for v in (raw_quantiles if not isinstance(raw_quantiles, str) else raw_quantiles.split(","))
                if math.isfinite(float(v)) and 0.0 <= float(v) <= 1.0
            }
        ) or [0.0]
        self.oracle_threshold_stats: dict[str, dict[str, _ThresholdOracleStat]] = {}
        self.oracle_rows: list[dict[str, Any]] = []
        self._oracle_row_group_counter = 0
        if store_scatter_samples is None:
            store_scatter_samples = bool(_cfg_get(train_cfg, "save_pi_all_edge_actual_vs_pred_plots", False))
        self.store_scatter_samples = bool(store_scatter_samples)
        max_points = int(_cfg_get(train_cfg, "actual_vs_pred_max_points_per_property", 50000))
        self.max_scatter_points_per_property = max(max_points, 0)
        self.scatter_rows: list[dict[str, Any]] = []
        self._scatter_counts: dict[str, int] = {}
        self.zero_flow_fraction_mask_enabled = bool(
            _cfg_get(train_cfg, "use_zero_flow_fraction_mask", False)
        )
        self.zero_flow_fraction_mask_eps = float(
            _cfg_get(train_cfg, "zero_flow_fraction_mask_eps", _cfg_get(train_cfg, "eps", 1.0e-8))
        )
        if (
            not math.isfinite(self.zero_flow_fraction_mask_eps)
            or self.zero_flow_fraction_mask_eps < 0.0
        ):
            raise ValueError(
                "zero_flow_fraction_mask_eps must be finite and non-negative, "
                f"got {self.zero_flow_fraction_mask_eps}."
            )
        self.zero_flow_fraction_rows_seen = 0
        self.zero_flow_fraction_rows_masked = 0
        fraction_threshold = float(_cfg_get(train_cfg, "metric_relevance_fraction_threshold", 0.6))
        if not math.isfinite(fraction_threshold) or fraction_threshold < 0.0:
            raise ValueError(
                f"metric_relevance_fraction_threshold must be finite and non-negative, got {fraction_threshold}."
            )
        flow_threshold = float(_cfg_get(train_cfg, "metric_relevance_flow_threshold", 0.6))
        if not math.isfinite(flow_threshold) or flow_threshold < 0.0:
            raise ValueError(
                f"metric_relevance_flow_threshold must be finite and non-negative, got {flow_threshold}."
            )
        self.relevance_fraction_threshold = fraction_threshold
        self.relevance_flow_threshold = flow_threshold

    def update_batch(
        self,
        *,
        outputs: Mapping[str, torch.Tensor],
        targets_raw: Mapping[str, torch.Tensor],
        target_masks: Mapping[str, torch.Tensor],
        train_cfg: Any,
        data_cfg: Any,
        edge_target_columns: Sequence[str],
        normalizer: Mapping[str, Any] | None,
        split_name: str = "val",
    ) -> None:
        cols = list(edge_target_columns)
        phys = build_pi_physical_outputs(
            outputs,
            train_cfg=train_cfg,
            data_cfg=data_cfg,
            normalizer=normalizer,
        )
        species = resolve_pi_species_order(data_cfg, train_cfg, outputs)
        reduced_hierarchical_output = (
            outputs.get("level1_pred") is not None
            and outputs.get("mass_flow_pred") is not None
            and outputs.get("mole_flow_pred") is None
            and outputs.get("rho_pred") is None
            and outputs.get("h_pred") is None
            and int(outputs["main_stream_pred"].shape[-1]) in {10, 11}
        )
        pred_by_name: dict[str, torch.Tensor] = {
            "Temp": phys["T"],
            "Pres": phys["P"],
            "Mass_Flow": phys["mass_flow"],
        }
        if phys.get("volume_flow") is not None and (
            outputs.get("volume_flow_pred") is not None
            or not reduced_hierarchical_output
        ):
            pred_by_name["Vol_Flow"] = phys["volume_flow"]
        if not reduced_hierarchical_output:
            pred_by_name["Mole_Flow"] = phys["mole_flow"]
        if phys.get("rho") is not None:
            pred_by_name["Density"] = phys["rho"]
        if phys.get("h") is not None:
            pred_by_name["Enthalpy"] = phys["h"]
        self.prediction_kinds.update(
            {
                "Temp": "direct_head",
                "Pres": "direct_head",
                "Mass_Flow": "direct_head",
            }
        )
        if "Vol_Flow" in pred_by_name:
            self.prediction_kinds["Vol_Flow"] = (
                "direct_head" if outputs.get("volume_flow_pred") is not None else "derived"
            )
        if not reduced_hierarchical_output:
            self.prediction_kinds["Mole_Flow"] = (
                "direct_head" if outputs.get("mole_flow_pred") is not None else "derived"
            )
        if phys.get("rho") is not None:
            self.prediction_kinds["Density"] = "direct_head"
        if phys.get("h") is not None:
            self.prediction_kinds["Enthalpy"] = "direct_head"
        for idx, species_name in enumerate(species):
            name = f"Frac_{species_name}"
            pred_by_name[name] = phys["frac"][..., idx : idx + 1]
            self.prediction_kinds[name] = "direct_head"
        names = [name for name in cols if name in pred_by_name]
        pred = torch.cat([pred_by_name[name] for name in names], dim=-1)
        true_all = targets_raw["edge_stream"].to(device=pred.device, dtype=pred.dtype)
        true = _take_columns(true_all, [cols.index(name) for name in names])
        mask_src = target_masks.get("edge_stream")
        mask = torch.ones_like(true) if mask_src is None else _expand_edge_mask(mask_src, len(names), true)
        fraction_indices = [idx for idx, name in enumerate(names) if str(name).startswith("Frac_")]
        if self.zero_flow_fraction_mask_enabled and fraction_indices:
            if "Mass_Flow" not in cols:
                raise RuntimeError(
                    "use_zero_flow_fraction_mask=true requires Mass_Flow in edge_target_columns "
                    "for all-edge fraction metrics."
                )
            mass_true = true_all[..., cols.index("Mass_Flow")].to(device=pred.device, dtype=pred.dtype)
            flowing_row = torch.isfinite(mass_true) & (mass_true > self.zero_flow_fraction_mask_eps)
            fraction_index = torch.tensor(fraction_indices, device=mask.device, dtype=torch.long)
            fraction_mask_before = mask.index_select(dim=-1, index=fraction_index) > 0
            supervised_fraction_row = fraction_mask_before.any(dim=-1)
            self.zero_flow_fraction_rows_seen += int(supervised_fraction_row.sum().detach().cpu())
            self.zero_flow_fraction_rows_masked += int(
                (supervised_fraction_row & ~flowing_row).sum().detach().cpu()
            )
            mask = mask.clone()
            mask[..., fraction_indices] = (
                mask[..., fraction_indices] * flowing_row.unsqueeze(-1).to(mask.dtype)
            )
        pred = pred.reshape(-1, len(names))
        true = true.reshape(-1, len(names))
        mask = mask.reshape(-1, len(names))
        oracle_group_ids: list[int] = []
        if self.store_oracle_samples:
            oracle_group_ids = list(range(self._oracle_row_group_counter, self._oracle_row_group_counter + pred.shape[0]))
            self._oracle_row_group_counter += int(pred.shape[0])
        display_group_ids: list[int] = []
        if self.diagnostic_enabled and self.diagnostic_fraction_postprocess_enabled:
            display_group_ids = list(range(self._display_row_group_counter, self._display_row_group_counter + pred.shape[0]))
            self._display_row_group_counter += int(pred.shape[0])
        for idx, name in enumerate(names):
            valid = (mask[:, idx] > 0) & torch.isfinite(true[:, idx]) & torch.isfinite(pred[:, idx])
            self.stats.setdefault(name, _RegressionMoments()).update(true[:, idx], pred[:, idx], valid)
            if str(name) == "Mass_Flow" and normalizer is not None:
                for label in ("p90", "p95", "p99"):
                    threshold_raw = normalizer.get(f"mass_flow_physical_{label}")
                    if threshold_raw is None:
                        continue
                    threshold = float(threshold_raw)
                    if not math.isfinite(threshold):
                        continue
                    self.mass_flow_tail_thresholds[label] = threshold
                    tail_valid = valid & (true[:, idx] >= threshold)
                    self.mass_flow_tail_stats.setdefault(label, _RegressionMoments()).update(
                        true[:, idx],
                        pred[:, idx],
                        tail_valid,
                    )
                    if label == "p95" and bool(tail_valid.any()):
                        true_tail = true[:, idx][tail_valid].detach().float().cpu()
                        pred_tail = pred[:, idx][tail_valid].detach().float().cpu()
                        ratio_valid = true_tail.abs() > 1.0e-12
                        self.mass_flow_p95_pred_true_ratios.extend(
                            (pred_tail[ratio_valid] / true_tail[ratio_valid]).tolist()
                        )
                        error = pred_tail - true_tail
                        self.mass_flow_p95_signed_error_sum += float(error.sum().item())
                        self.mass_flow_p95_signed_error_count += int(error.numel())
            if (
                self.diagnostic_enabled
                and self.diagnostic_transformed_enabled
                and str(name) in {"Mass_Flow", "Mole_Flow", "Vol_Flow"}
                and bool(valid.any())
            ):
                true_log = torch.log1p(true[:, idx].clamp_min(0.0))
                pred_log = torch.log1p(pred[:, idx].clamp_min(0.0))
                self.display_log_flow_stats.setdefault(name, _RegressionMoments()).update(true_log, pred_log, valid)
            if self.store_oracle_samples and self.oracle_threshold_enabled and bool(valid.any()):
                prop_threshold_stats = self.oracle_threshold_stats.setdefault(name, {})
                if str(name).startswith("Frac_"):
                    for threshold in self.oracle_abs_thresholds:
                        stat_key = f"absolute:{threshold:g}"
                        stat = prop_threshold_stats.setdefault(
                            stat_key,
                            _ThresholdOracleStat(
                                threshold_type="absolute",
                                threshold_label=stat_key,
                                quantile=math.nan,
                            ),
                        )
                        stat.update(true[:, idx], pred[:, idx], valid, threshold=float(threshold))
                else:
                    true_valid_for_q = true[:, idx][valid].detach().float()
                    if true_valid_for_q.numel() > 0:
                        for quantile in self.oracle_quantiles:
                            threshold_tensor = torch.quantile(true_valid_for_q, float(quantile))
                            threshold = float(threshold_tensor.detach().cpu().item())
                            stat_key = f"quantile:{quantile:g}"
                            stat = prop_threshold_stats.setdefault(
                                stat_key,
                                _ThresholdOracleStat(
                                    threshold_type="quantile",
                                    threshold_label=stat_key,
                                    quantile=float(quantile),
                                ),
                            )
                            stat.update(true[:, idx], pred[:, idx], valid, threshold=threshold)
            if (
                self.store_oracle_samples
                and self.oracle_fraction_postprocess_enabled
                and str(name).startswith("Frac_")
                and bool(valid.any())
            ):
                true_valid = true[:, idx][valid].detach().cpu()
                pred_valid = pred[:, idx][valid].detach().cpu()
                valid_indices = torch.nonzero(valid.detach().cpu(), as_tuple=False).reshape(-1).tolist()
                for local_i, y_true, y_pred in zip(valid_indices, true_valid.tolist(), pred_valid.tolist()):
                    self.oracle_rows.append(
                        {
                            "split": str(split_name),
                            "metric_scope": "all_supervised_edges_by_property",
                            "metric_row_group_id": int(oracle_group_ids[int(local_i)]),
                            "property_name": str(name),
                            "y_true": float(y_true),
                            "y_pred": float(y_pred),
                            "mask": 1.0,
                        }
                    )
            if (
                self.diagnostic_enabled
                and self.diagnostic_fraction_postprocess_enabled
                and str(name).startswith("Frac_")
                and bool(valid.any())
            ):
                true_valid = true[:, idx][valid].detach().cpu()
                pred_valid = pred[:, idx][valid].detach().cpu()
                valid_indices = torch.nonzero(valid.detach().cpu(), as_tuple=False).reshape(-1).tolist()
                for local_i, y_true, y_pred in zip(valid_indices, true_valid.tolist(), pred_valid.tolist()):
                    self.display_fraction_rows.append(
                        {
                            "split": str(split_name),
                            "metric_scope": "all_supervised_edges_by_property",
                            "metric_row_group_id": int(display_group_ids[int(local_i)]),
                            "property_name": str(name),
                            "y_true": float(y_true),
                            "y_pred": float(y_pred),
                            "mask": 1.0,
                        }
                    )
            if self.store_scatter_samples and self.max_scatter_points_per_property > 0:
                remaining = self.max_scatter_points_per_property - self._scatter_counts.get(name, 0)
                if remaining > 0 and bool(valid.any()):
                    true_valid = true[:, idx][valid].detach().cpu()
                    pred_valid = pred[:, idx][valid].detach().cpu()
                    take = min(int(true_valid.numel()), int(remaining))
                    for y_true, y_pred in zip(true_valid[:take].tolist(), pred_valid[:take].tolist()):
                        self.scatter_rows.append(
                            {
                                "split": str(split_name),
                                "metric_scope": "all_supervised_edges_by_property",
                                "property_name": str(name),
                                "y_true": float(y_true),
                                "y_pred": float(y_pred),
                                "mask": 1.0,
                            }
                        )
                    self._scatter_counts[name] = self._scatter_counts.get(name, 0) + take
            if self.relevance_enabled:
                if str(name).startswith("Frac_"):
                    relevant_valid = valid & (true[:, idx] > self.relevance_fraction_threshold)
                elif str(name) in {"Mass_Flow", "Mole_Flow", "Vol_Flow"}:
                    relevant_valid = valid & (true[:, idx] > self.relevance_flow_threshold)
                else:
                    relevant_valid = valid
                self.relevant_stats.setdefault(name, _RegressionMoments()).update(
                    true[:, idx],
                    pred[:, idx],
                    relevant_valid,
                )

    def rows(self) -> list[dict[str, Any]]:
        order = list(STREAM_EDGE_FEATURE_SLOTS) + ["Density", "Enthalpy"]
        rows = [self.stats[name].row(name) for name in order if name in self.stats]
        for row in rows:
            name = str(row["property_name"])
            row["prediction_kind"] = self.prediction_kinds.get(name, row["prediction_kind"])
        return rows

    def relevant_rows(self) -> list[dict[str, Any]]:
        order = list(STREAM_EDGE_FEATURE_SLOTS) + ["Density", "Enthalpy"]
        rows: list[dict[str, Any]] = []
        for name in order:
            if name not in self.relevant_stats:
                continue
            row = self.relevant_stats[name].row(name)
            row["prediction_kind"] = self.prediction_kinds.get(name, row["prediction_kind"])
            row["metric_scope"] = "all_supervised_edges_by_property_relevant"
            row["relevance_rule"] = (
                f"true>{self.relevance_fraction_threshold:g}"
                if str(name).startswith("Frac_")
                else f"true>{self.relevance_flow_threshold:g}"
                if str(name) in {"Mass_Flow", "Mole_Flow", "Vol_Flow"}
                else "finite"
            )
            rows.append(row)
        return rows

    def oracle_threshold_rows(self) -> list[dict[str, Any]]:
        if not self.store_oracle_samples or not self.oracle_threshold_enabled:
            return []
        order = list(STREAM_EDGE_FEATURE_SLOTS) + ["Density", "Enthalpy"]
        rows: list[dict[str, Any]] = []
        for name in order:
            stats_by_threshold = self.oracle_threshold_stats.get(name, {})
            if not stats_by_threshold:
                continue
            raw_row = self.stats.get(name, _RegressionMoments()).row(name)
            raw_r2 = float(raw_row.get("R2", math.nan))
            raw_n = int(raw_row.get("n", 0))
            candidates = []
            for stat in stats_by_threshold.values():
                row = stat.row(name, scope="all_edge", raw_r2=raw_r2)
                row["raw_n"] = raw_n
                if int(row.get("best_n", 0)) >= self.oracle_min_samples:
                    candidates.append(row)
            if not candidates:
                continue
            best = max(
                candidates,
                key=lambda row: float(row.get("best_r2", float("-inf")))
                if math.isfinite(float(row.get("best_r2", math.nan)))
                else float("-inf"),
            )
            best["num_thresholds_evaluated"] = len(candidates)
            rows.append(best)
        return rows

    def oracle_calibrated_rows(self) -> list[dict[str, Any]]:
        if not self.store_oracle_samples or not self.oracle_calibration_enabled:
            return []
        order = list(STREAM_EDGE_FEATURE_SLOTS) + ["Density", "Enthalpy"]
        rows: list[dict[str, Any]] = []
        for name in order:
            moments = self.stats.get(name)
            if moments is None:
                continue
            rows.append(moments.calibrated_row(name, scope="all_edge"))
        return rows

    def diagnostic_corr2_rows(self) -> list[dict[str, Any]]:
        if not self.diagnostic_enabled or not self.diagnostic_corr2_enabled:
            return []
        order = list(STREAM_EDGE_FEATURE_SLOTS) + ["Density", "Enthalpy"]
        rows: list[dict[str, Any]] = []
        for name in order:
            moments = self.stats.get(name)
            if moments is None:
                continue
            rows.append(moments.corr2_row(name, scope="all_edge", min_n=self.diagnostic_min_samples))
        return rows

    def diagnostic_calibrated_rows(self) -> list[dict[str, Any]]:
        if not self.diagnostic_enabled or not self.diagnostic_calibrated_enabled:
            return []
        order = list(STREAM_EDGE_FEATURE_SLOTS) + ["Density", "Enthalpy"]
        rows: list[dict[str, Any]] = []
        for name in order:
            moments = self.stats.get(name)
            if moments is None:
                continue
            rows.append(moments.diagnostic_calibrated_row(name, scope="all_edge", min_n=self.diagnostic_min_samples))
        return rows

    def diagnostic_transformed_rows(self) -> list[dict[str, Any]]:
        if not self.diagnostic_enabled or not self.diagnostic_transformed_enabled:
            return []
        rows: list[dict[str, Any]] = []
        for name in ("Mass_Flow", "Mole_Flow", "Vol_Flow"):
            moments = self.display_log_flow_stats.get(name)
            if moments is None:
                continue
            raw_r2 = self.stats.get(name, _RegressionMoments()).row(name).get("R2", math.nan)
            rows.append(
                moments.transformed_row(
                    name,
                    scope="all_edge",
                    raw_r2=float(raw_r2),
                    min_n=self.diagnostic_min_samples,
                )
            )
        return rows

    def write_artifacts(self, out_dir: Path) -> dict[str, float]:
        import json
        import pandas as pd

        logged_properties = {
            "Temp",
            "Pres",
            "Mass_Flow",
            "Mole_Flow",
            "Vol_Flow",
            "Frac_H2O",
            "Frac_H2",
            "Frac_CH4",
            "Frac_CO2",
            "Frac_CO",
            "Frac_O2",
            "Frac_N2",
            "Density",
            "Enthalpy",
        }
        rows = self.rows()
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_frame = pd.DataFrame(rows)
        raw_frame.to_csv(out_dir / "pi_all_edge_property_r2_raw.csv", index=False)
        relevant_rows = self.relevant_rows() if self.relevance_enabled else []
        output_rows = rows
        if relevant_rows:
            pd.DataFrame(relevant_rows).to_csv(out_dir / "pi_all_edge_property_r2_relevance_deprecated.csv", index=False)
        frame = pd.DataFrame(output_rows)
        frame.to_csv(out_dir / "pi_all_edge_property_r2.csv", index=False)
        if self.mass_flow_tail_stats:
            tail_rows: list[dict[str, Any]] = []
            for label in ("p90", "p95", "p99"):
                moments = self.mass_flow_tail_stats.get(label)
                if moments is None:
                    continue
                row = moments.row("Mass_Flow")
                row.update(
                    {
                        "metric_scope": "mass_flow_train_threshold_tail",
                        "train_threshold_label": label.upper(),
                        "train_threshold": self.mass_flow_tail_thresholds.get(label, math.nan),
                    }
                )
                if label == "p95":
                    ratios = sorted(self.mass_flow_p95_pred_true_ratios)
                    row["median_pred_true_ratio"] = (
                        float(ratios[len(ratios) // 2]) if ratios else math.nan
                    )
                    row["mean_signed_error"] = (
                        self.mass_flow_p95_signed_error_sum / self.mass_flow_p95_signed_error_count
                        if self.mass_flow_p95_signed_error_count
                        else math.nan
                    )
                tail_rows.append(row)
            pd.DataFrame(tail_rows).to_csv(out_dir / "mass_flow_tail_metrics.csv", index=False)
        payload = {
            "metadata": {
                "metric_scope": "all_supervised_edges_by_property",
                "metric_scope_note": (
                    "pi_all_edge_property_r2.csv uses supervised physical-scale values. "
                    "When zero_flow_fraction_mask_enabled=true, Frac_* metrics exclude rows whose "
                    "true Mass_Flow is not greater than zero_flow_fraction_mask_eps. Relevance "
                    "filtering is deprecated and never replaces the official all-edge property file"
                ),
                "metric_relevance_enabled": self.relevance_enabled,
                "metric_relevance_fraction_threshold": self.relevance_fraction_threshold,
                "metric_relevance_flow_threshold": self.relevance_flow_threshold,
                "zero_flow_fraction_mask_enabled": self.zero_flow_fraction_mask_enabled,
                "zero_flow_fraction_mask_eps": self.zero_flow_fraction_mask_eps,
                "zero_flow_fraction_rows_seen": self.zero_flow_fraction_rows_seen,
                "zero_flow_fraction_rows_masked": self.zero_flow_fraction_rows_masked,
                "prediction_kinds": dict(sorted(self.prediction_kinds.items())),
            },
            "rows": output_rows,
            "raw_rows": rows,
        }
        (out_dir / "pi_all_edge_property_r2.json").write_text(
            json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8"
        )
        if self.store_scatter_samples and self.scatter_rows:
            from .target_edge_10d_metrics import write_metric_rows_actual_vs_pred_plots

            plot_frame = pd.DataFrame(self.scatter_rows)
            plot_paths = write_metric_rows_actual_vs_pred_plots(
                out_dir=out_dir,
                metric_rows=plot_frame,
                properties=tuple(logged_properties),
                root_name="all_edge_actual_vs_predicted",
            )
            plot_root = out_dir / "all_edge_actual_vs_predicted"
            plot_root.mkdir(parents=True, exist_ok=True)
            (plot_root / "manifest.json").write_text(
                json.dumps(
                    {"files": [str(path.relative_to(out_dir)) for path in plot_paths]},
                    indent=2,
                ),
                encoding="utf-8",
            )
        if self.store_oracle_samples:
            from .target_edge_10d_metrics import (
                build_oracle_fraction_postprocess_metrics,
                write_oracle_diagnostic_frames,
            )

            threshold_df = pd.DataFrame(self.oracle_threshold_rows())
            calibrated_df = pd.DataFrame(self.oracle_calibrated_rows())
            post_df = (
                build_oracle_fraction_postprocess_metrics(pd.DataFrame(self.oracle_rows), scope="all_edge")
                if self.oracle_rows and self.oracle_fraction_postprocess_enabled
                else pd.DataFrame()
            )
            scalars_oracle = write_oracle_diagnostic_frames(
                out_dir=out_dir,
                scope="all_edge",
                file_prefix="pi_all_edge",
                threshold_df=threshold_df,
                calibrated_df=calibrated_df,
                post_df=post_df,
            )
        if self.diagnostic_enabled:
            from .target_edge_10d_metrics import (
                build_diagnostic_display_r2_metrics,
                build_diagnostic_fraction_postprocess_r2_metrics,
                write_diagnostic_display_frames,
            )

            corr_df = pd.DataFrame(self.diagnostic_corr2_rows())
            calibrated_display_df = pd.DataFrame(self.diagnostic_calibrated_rows())
            transformed_df = pd.DataFrame(self.diagnostic_transformed_rows())
            post_display_df = (
                build_diagnostic_fraction_postprocess_r2_metrics(
                    pd.DataFrame(self.display_fraction_rows),
                    scope="all_edge",
                    train_cfg=self.train_cfg,
                )
                if self.display_fraction_rows and self.diagnostic_fraction_postprocess_enabled
                else pd.DataFrame()
            )
            display_df = build_diagnostic_display_r2_metrics(
                scope="all_edge",
                corr_df=corr_df,
                calibrated_df=calibrated_display_df,
                transformed_df=transformed_df,
                post_df=post_display_df,
            )
            scalars_diagnostic = write_diagnostic_display_frames(
                out_dir=out_dir,
                scope="all_edge",
                file_prefix="pi_all_edge",
                corr_df=corr_df,
                calibrated_df=calibrated_display_df,
                transformed_df=transformed_df,
                post_df=post_display_df,
                display_df=display_df,
            )
        scalars: dict[str, float] = {}
        r2_values: list[float] = []
        for row in output_rows:
            value = float(row["R2"])
            if not math.isfinite(value):
                continue
            if str(row["property_name"]) not in logged_properties:
                continue
            slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in row["property_name"]).strip("_")
            scalars[f"pi_all_edge_r2_{slug}"] = value
            r2_values.append(value)
        if r2_values:
            scalars["pi_all_edge_property_mean_r2"] = float(sum(r2_values) / len(r2_values))
        if self.store_oracle_samples:
            scalars.update(scalars_oracle)
        if self.diagnostic_enabled:
            scalars.update(scalars_diagnostic)
        display = ", ".join(
            f"{row['property_name']}={float(row['R2']):.5f}"
            for row in output_rows
            if str(row["property_name"]) in logged_properties
        )
        if display:
            print(f"[R2][all-edges/by-property] {display}", flush=True)
        if self.zero_flow_fraction_mask_enabled:
            print(
                "[R2][all-edges/fraction-mask] "
                f"flow_rule=true_Mass_Flow>{self.zero_flow_fraction_mask_eps:g} "
                f"excluded={self.zero_flow_fraction_rows_masked}/"
                f"{self.zero_flow_fraction_rows_seen}",
                flush=True,
            )
        return scalars


def _mean_valid(x: torch.Tensor, valid: torch.Tensor) -> float:
    return float(x[valid].mean().detach().cpu().item()) if bool(valid.any()) else 0.0


def _max_valid(x: torch.Tensor, valid: torch.Tensor) -> float:
    return float(x[valid].max().detach().cpu().item()) if bool(valid.any()) else 0.0


def compute_single_edge_pinn_loss(
    *,
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    target_masks: Mapping[str, torch.Tensor],
    edge_id: int,
    edge_indices: Sequence[int] | torch.Tensor | None = None,
    train_cfg: Any,
    data_cfg: Any,
    edge_target_columns: Sequence[str] | None = None,
    normalizer: Mapping[str, Any] | None = None,
    criterion_main: Any | None = None,
    edge_weight_vector: torch.Tensor | None = None,
    global_step: int | None = None,
    total_train_steps: int | None = None,
    debug: bool = False,
    collect_log: bool = True,
    pi_targets_cache: Mapping[str, torch.Tensor | None] | None = None,
    phys_outputs_cache: Mapping[str, torch.Tensor] | None = None,
    phys_targets_cache: Mapping[str, torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    eps = float(_cfg_get(train_cfg, "eps", 1.0e-8))
    pi_t = (
        dict(pi_targets_cache)
        if pi_targets_cache is not None
        else get_pi_targets(
            targets=targets,
            target_masks=target_masks,
            outputs=outputs,
            train_cfg=train_cfg,
            data_cfg=data_cfg,
            edge_target_columns=edge_target_columns,
        )
    )
    idxs = [int(edge_id)] if edge_indices is None else edge_indices
    pred_main_e = _select_edge_rows(outputs["main_stream_pred"], idxs)
    target_main_e = _select_edge_rows(pi_t["main_target"], idxs, int(pred_main_e.shape[-1]))
    mask_main_e = _select_edge_rows(pi_t["main_mask"], idxs, int(pred_main_e.shape[-1]))
    rho_pred = outputs.get("rho_pred")
    rho_pred_e = (
        _select_edge_rows(rho_pred, idxs)
        if rho_pred is not None
        else None
    )
    rho_target_e = _select_edge_rows(pi_t["rho_target"], idxs, 1) if pi_t["rho_target"] is not None else None
    rho_mask_e = _select_edge_rows(pi_t["rho_mask"], idxs, 1) if pi_t["rho_mask"] is not None else None
    h_pred = outputs.get("h_pred")
    h_pred_e = (
        _select_edge_rows(h_pred, idxs)
        if h_pred is not None
        else None
    )
    h_target_e = _select_edge_rows(pi_t["h_target"], idxs, 1) if pi_t["h_target"] is not None else None
    h_mask_e = _select_edge_rows(pi_t["h_mask"], idxs, 1) if pi_t["h_mask"] is not None else None

    if criterion_main is None:
        criterion_main = build_pi_main_criterion(train_cfg)

    loss_outputs = dict(outputs)
    loss_outputs["frac_pred"] = _select_edge_rows(outputs["frac_pred"], idxs)
    for diagnostic_key in ("relu_l1_denominator", "relu_l1_all_zero_mask"):
        diagnostic_value = outputs.get(diagnostic_key)
        if diagnostic_value is not None:
            loss_outputs[diagnostic_key] = _select_edge_rows(diagnostic_value, idxs)

    (
        pred_main_for_loss,
        target_main_for_loss,
        mask_main_for_loss,
        main_loss_weight,
        loss_frac_clr,
        valid_frac_clr,
        loss_frac_log,
        valid_frac_log,
        loss_frac_closure,
        valid_frac_closure,
        loss_frac_penalty,
        valid_frac_penalty,
        fraction_loss_debug,
    ) = _prepare_pi_main_loss_tensors(
        pred_main=pred_main_e,
        target_main=target_main_e,
        mask_main=mask_main_e,
        outputs=loss_outputs,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        normalizer=normalizer,
        global_step=global_step,
        total_train_steps=total_train_steps,
        collect_debug=bool(collect_log or debug),
    )
    mass_feature_idx = 2 + int(loss_outputs["frac_pred"].shape[-1])
    mass_aux_cfg = _cfg_get(train_cfg, "mass_flow_physical_auxiliary", None)
    mass_log_weight = float(_cfg_get(mass_aux_cfg, "log_weight", 1.0))
    main_loss_weight = main_loss_weight.clone()
    main_loss_weight[..., mass_feature_idx : mass_feature_idx + 1] *= mass_log_weight
    loss_main, valid_main = masked_regression_per_sample(
        pred_main_for_loss,
        target_main_for_loss,
        mask_main_for_loss,
        criterion=criterion_main,
        loss_weight=main_loss_weight,
        eps=eps,
    )
    mass_aux_cfg = _cfg_get(train_cfg, "mass_flow_physical_auxiliary", None)
    mass_aux_enabled = bool(_cfg_get(mass_aux_cfg, "enabled", False))
    needs_physical = bool(
        collect_log
        or debug
        or mass_aux_enabled
        or _cfg_get(train_cfg, "use_rho_loss", False)
        or _cfg_get(train_cfg, "use_h_loss", False)
        or _cfg_get(train_cfg, "use_volume_loss", False)
        or _cfg_get(train_cfg, "use_enthalpy_flow_loss", False)
    )
    phys_o = (
        dict(phys_outputs_cache)
        if phys_outputs_cache is not None
        else build_pi_physical_outputs(outputs, train_cfg=train_cfg, data_cfg=data_cfg, normalizer=normalizer)
        if needs_physical
        else {}
    )
    if bool(collect_log or debug) and bool(fraction_loss_debug.get("mass_flow_direct_log_enabled", False)):
        mass_physical_e = _select_edge_rows(phys_o["mass_flow"], idxs, 1)
        fraction_loss_debug.update(
            {
                "mass_flow_direct_pred_physical_min": float(mass_physical_e.min().detach().cpu()),
                "mass_flow_direct_pred_physical_max": float(mass_physical_e.max().detach().cpu()),
                "mass_flow_direct_pred_physical_mean": float(mass_physical_e.mean().detach().cpu()),
                "mass_flow_direct_pred_physical_std": float(mass_physical_e.std(unbiased=False).detach().cpu()),
                "mass_flow_direct_pred_physical_finite": bool(torch.isfinite(mass_physical_e).all()),
            }
        )
    phys_t = (
        dict(phys_targets_cache)
        if phys_targets_cache is not None
        else build_pi_physical_targets(pi_t, train_cfg=train_cfg, data_cfg=data_cfg, normalizer=normalizer)
        if needs_physical
        else {}
    )
    mass_mask_e = mask_main_for_loss[..., mass_feature_idx : mass_feature_idx + 1]
    mass_log_raw = criterion_main(
        pred_main_for_loss[..., mass_feature_idx : mass_feature_idx + 1],
        target_main_for_loss[..., mass_feature_idx : mass_feature_idx + 1],
    )
    mass_log_raw = (
        mass_log_raw
        * main_loss_weight[..., mass_feature_idx : mass_feature_idx + 1].to(mass_log_raw)
    ).reshape(-1)
    valid_mass_log = (
        (mass_mask_e.reshape(-1) > 0)
        & torch.isfinite(mass_log_raw)
    )
    mass_log_raw = torch.where(valid_mass_log, mass_log_raw, torch.zeros_like(mass_log_raw))
    if mass_aux_enabled:
        target_mass_physical_e = _inverse_if_needed(
            target_main_e[..., mass_feature_idx : mass_feature_idx + 1],
            "Mass_Flow",
            normalizer,
            data_cfg,
        )
        mass_physical_loss, valid_mass_physical, mass_aux_debug = (
            compute_mass_flow_physical_auxiliary_per_sample(
                _select_edge_rows(phys_o["mass_flow"], idxs, 1),
                target_mass_physical_e,
                mass_mask_e,
                train_cfg=train_cfg,
                normalizer=normalizer,
            )
        )
    else:
        mass_physical_loss = torch.zeros_like(mass_log_raw)
        valid_mass_physical = torch.zeros_like(valid_mass_log)
        mass_aux_debug = {
            "mass_flow_physical_aux_enabled": False,
            "mass_flow_physical_aux_weight": 0.0,
            "mass_flow_physical_aux_scale": math.nan,
            "mass_flow_physical_aux_scale_method": "disabled",
        }
    mass_aux_weight = float(mass_aux_debug["mass_flow_physical_aux_weight"])
    if bool(mass_aux_debug["mass_flow_physical_aux_enabled"]):
        feature_count = mask_main_for_loss.sum(dim=-1).clamp_min(eps)
        loss_main = loss_main + mass_aux_weight * mass_physical_loss / feature_count
        valid_main = valid_main | valid_mass_physical
    mass_total_raw = mass_log_raw + mass_aux_weight * mass_physical_loss
    derived_mole_ratio = math.nan
    if bool(collect_log or debug) and phys_t.get("mole_flow") is not None:
        pred_mole_e = _select_edge_rows(phys_o["mole_flow"], idxs, 1)
        true_mole_e = _select_edge_rows(phys_t["mole_flow"], idxs, 1)
        ratio_valid = torch.isfinite(pred_mole_e) & torch.isfinite(true_mole_e) & (true_mole_e.abs() > eps)
        if bool(ratio_valid.any()):
            derived_mole_ratio = float((pred_mole_e[ratio_valid] / true_mole_e[ratio_valid]).mean().detach().cpu())
    derived_volume_ratio = math.nan
    if bool(collect_log or debug) and phys_t.get("volume_flow") is not None and phys_o.get("volume_flow") is not None:
        pred_volume_e = _select_edge_rows(phys_o["volume_flow"], idxs, 1)
        true_volume_e = _select_edge_rows(phys_t["volume_flow"], idxs, 1)
        ratio_valid = torch.isfinite(pred_volume_e) & torch.isfinite(true_volume_e) & (true_volume_e.abs() > eps)
        if bool(ratio_valid.any()):
            derived_volume_ratio = float(
                (pred_volume_e[ratio_valid] / true_volume_e[ratio_valid]).mean().detach().cpu()
            )

    zero = torch.zeros_like(loss_main)
    valid_zero = torch.zeros_like(valid_main, dtype=torch.bool)
    if bool(_cfg_get(train_cfg, "use_rho_loss", True)):
        if rho_pred_e is None or phys_o["rho"] is None:
            raise RuntimeError(
                "use_rho_loss=true requires a rho_pred output, but the active "
                "head does not predict Density."
            )
        if rho_target_e is None or rho_mask_e is None or phys_t["rho"] is None:
            raise RuntimeError("use_rho_loss=true but rho target is missing.")
        loss_rho, valid_rho = masked_residual_regression_per_sample(
            _select_edge_rows(phys_o["rho"], idxs),
            _select_edge_rows(phys_t["rho"], idxs, 1),
            rho_mask_e,
            criterion=criterion_main,
            residual_abs_clip=_cfg_get(train_cfg, "rho_residual_abs_clip", None),
            eps=eps,
        )
    else:
        loss_rho, valid_rho = zero.clone(), valid_zero.clone()
    if bool(_cfg_get(train_cfg, "use_h_loss", True)):
        if h_pred_e is None or phys_o["h"] is None:
            raise RuntimeError(
                "use_h_loss=true requires an h_pred output, but the active "
                "head does not predict Enthalpy."
            )
        if h_target_e is None or h_mask_e is None or phys_t["h"] is None:
            raise RuntimeError("use_h_loss=true but h target is missing.")
        h_loss_fn = (
            masked_normalized_mse_per_sample
            if bool(_cfg_get(train_cfg, "h_loss_normalized", True))
            else masked_regression_per_sample
        )
        h_loss_kwargs = {} if bool(_cfg_get(train_cfg, "h_loss_normalized", True)) else {"criterion": criterion_main}
        if bool(_cfg_get(train_cfg, "h_loss_normalized", True)):
            h_loss_kwargs = {
                "scale_floor": _cfg_get(train_cfg, "h_loss_scale_floor", None),
                "loss_type": str(_cfg_get(train_cfg, "h_normalized_loss_type", "mse")),
                "huber_delta": float(_cfg_get(train_cfg, "h_normalized_huber_delta", 1.0)),
                "residual_abs_clip": _cfg_get(train_cfg, "h_normalized_residual_abs_clip", None),
            }
        loss_h, valid_h = h_loss_fn(
            _select_edge_rows(phys_o["h"], idxs),
            _select_edge_rows(phys_t["h"], idxs, 1),
            h_mask_e,
            eps=eps,
            **h_loss_kwargs,
        )
    else:
        loss_h, valid_h = zero.clone(), valid_zero.clone()
    if bool(_cfg_get(train_cfg, "use_volume_loss", False)) and pi_t["volume_mask"] is not None and phys_t.get("volume_flow") is not None:
        volume_loss_fn = (
            masked_normalized_mse_per_sample
            if bool(_cfg_get(train_cfg, "volume_loss_normalized", True))
            else masked_mse_per_sample
        )
        volume_loss_kwargs = (
            {
                "scale_floor": _cfg_get(train_cfg, "volume_loss_scale_floor", None),
                "loss_type": str(_cfg_get(train_cfg, "volume_normalized_loss_type", "mse")),
                "huber_delta": float(_cfg_get(train_cfg, "volume_normalized_huber_delta", 1.0)),
                "residual_abs_clip": _cfg_get(train_cfg, "volume_normalized_residual_abs_clip", None),
            }
            if bool(_cfg_get(train_cfg, "volume_loss_normalized", True))
            else {}
        )
        loss_volume, valid_volume = volume_loss_fn(
            _select_edge_rows(phys_o["volume_flow"], idxs),
            _select_edge_rows(phys_t["volume_flow"], idxs, 1),
            _select_edge_rows(pi_t["volume_mask"], idxs, 1),
            eps=eps,
            **volume_loss_kwargs,
        )
    elif bool(_cfg_get(train_cfg, "use_volume_loss", False)):
        raise RuntimeError("use_volume_loss=true but volume target is missing.")
    else:
        loss_volume, valid_volume = zero.clone(), valid_zero.clone()
    if bool(_cfg_get(train_cfg, "use_enthalpy_flow_loss", False)):
        if pi_t["enthalpy_flow_mask"] is None or phys_t["enthalpy_flow"] is None:
            raise RuntimeError("use_enthalpy_flow_loss=true but enthalpy flow target is missing.")
        loss_enthalpy_flow, valid_enthalpy_flow = masked_mse_per_sample(
            _select_edge_rows(phys_o["enthalpy_flow"], idxs),
            _select_edge_rows(phys_t["enthalpy_flow"], idxs, 1),
            _select_edge_rows(pi_t["enthalpy_flow_mask"], idxs, 1),
            eps=eps,
        )
    else:
        loss_enthalpy_flow, valid_enthalpy_flow = zero.clone(), valid_zero.clone()
    # Node atom/energy balances are computed once per graph batch, never inside
    # this per-edge update.
    loss_atom, valid_atom = zero.clone(), valid_zero.clone()
    loss_energy, valid_energy = zero.clone(), valid_zero.clone()

    lm = float(_cfg_get(train_cfg, "lambda_main", 1.0))
    lfrac_clr = (
        float(_cfg_get(train_cfg, "pi_fraction_clr_loss_weight", 1.0))
        if bool(fraction_loss_debug.get("fraction_clr_loss_enabled", False))
        else 0.0
    )
    if not math.isfinite(lfrac_clr) or lfrac_clr < 0.0:
        raise ValueError(f"pi_fraction_clr_loss_weight must be finite and non-negative, got {lfrac_clr}.")
    lfrac_log = (
        float(_cfg_get(train_cfg, "pi_fraction_log_loss_weight", 1.0))
        if bool(fraction_loss_debug.get("fraction_log_loss_enabled", False))
        else 0.0
    )
    if not math.isfinite(lfrac_log) or lfrac_log < 0.0:
        raise ValueError(f"pi_fraction_log_loss_weight must be finite and non-negative, got {lfrac_log}.")
    lfrac_closure = (
        float(_cfg_get(train_cfg, "pi_fraction_closure_loss_weight", 0.0))
        if bool(fraction_loss_debug.get("fraction_clr_loss_enabled", False))
        else 0.0
    )
    if not math.isfinite(lfrac_closure) or lfrac_closure < 0.0:
        raise ValueError(
            f"pi_fraction_closure_loss_weight must be finite and non-negative, got {lfrac_closure}."
        )
    lrho = float(_cfg_get(train_cfg, "lambda_rho", 0.01))
    lh = float(_cfg_get(train_cfg, "lambda_h", 0.01))
    lv = float(_cfg_get(train_cfg, "lambda_volume", 0.0))
    leh = float(_cfg_get(train_cfg, "lambda_enthalpy_flow", 0.0))
    la = 0.0
    le = 0.0
    valid_any = torch.zeros_like(valid_main, dtype=torch.bool)
    if lm > 0:
        valid_any = valid_any | valid_main
    if lfrac_clr > 0:
        valid_any = valid_any | valid_frac_clr
    if lfrac_log > 0:
        valid_any = valid_any | valid_frac_log
    if lfrac_closure > 0:
        valid_any = valid_any | valid_frac_closure
    if bool(fraction_loss_debug.get("frac_penalty_enabled", False)):
        valid_any = valid_any | valid_frac_penalty
    if lrho > 0 and bool(_cfg_get(train_cfg, "use_rho_loss", True)):
        valid_any = valid_any | valid_rho
    if lh > 0 and bool(_cfg_get(train_cfg, "use_h_loss", True)):
        valid_any = valid_any | valid_h
    if lv > 0 and bool(_cfg_get(train_cfg, "use_volume_loss", False)):
        valid_any = valid_any | valid_volume
    if leh > 0 and bool(_cfg_get(train_cfg, "use_enthalpy_flow_loss", False)):
        valid_any = valid_any | valid_enthalpy_flow
    if la > 0 and bool(_cfg_get(train_cfg, "use_atom_balance_loss", False)):
        valid_any = valid_any | valid_atom
    if le > 0 and bool(_cfg_get(train_cfg, "use_energy_balance_loss", False)):
        valid_any = valid_any | valid_energy
    if int(valid_any.sum().item()) == 0:
        return None, {"edge_id": int(edge_id), "skip_reason": "no_valid_target_for_edge"}

    base_loss_per_sample = (
        lm * loss_main
        + lfrac_clr * loss_frac_clr
        + lfrac_log * loss_frac_log
        + lfrac_closure * loss_frac_closure
        + lrho * loss_rho
        + lh * loss_h
        + lv * loss_volume
        + leh * loss_enthalpy_flow
        + la * loss_atom
        + le * loss_energy
    )
    loss_per_sample = base_loss_per_sample + loss_frac_penalty
    if edge_weight_vector is None:
        edge_weight = torch.full_like(loss_per_sample, float(_cfg_get(train_cfg, "edge_weight_default", 1.0)))
    elif edge_weight_vector.ndim == 1:
        edge_weight = _select_edge_rows(edge_weight_vector, idxs).reshape(-1).to(loss_per_sample.device, loss_per_sample.dtype)
    else:
        edge_weight = _select_edge_rows(edge_weight_vector, idxs).reshape(-1).to(loss_per_sample.device, loss_per_sample.dtype)
    weighted_loss = loss_per_sample * edge_weight
    loss_e = weighted_loss[valid_any].mean()
    mass_gradient_payload: dict[str, torch.Tensor] = {}
    if bool(_cfg_get(mass_aux_cfg, "gradient_diagnostics", False)):
        feature_count = mask_main_for_loss.sum(dim=-1).clamp_min(eps)
        mass_valid = valid_mass_log | valid_mass_physical
        if bool(mass_valid.any()):
            mass_gradient_payload = {
                "_mass_log_objective_tensor": (
                    lm * mass_log_raw / feature_count * edge_weight
                )[mass_valid].mean(),
                "_mass_physical_objective_tensor": (
                    lm * mass_aux_weight * mass_physical_loss / feature_count * edge_weight
                )[mass_valid].mean(),
            }
    if (not torch.isfinite(loss_e).all()) or bool(torch.isnan(loss_e)):
        return None, {"edge_id": int(edge_id), "skip_reason": "non_finite_loss"}
    if debug:
        assert pred_main_e.shape == target_main_e.shape == mask_main_e.shape
        if bool(_cfg_get(train_cfg, "use_rho_loss", True)):
            assert rho_pred_e is not None and rho_pred_e.ndim == 2
        if bool(_cfg_get(train_cfg, "use_h_loss", True)):
            assert h_pred_e is not None and h_pred_e.ndim == 2
        assert torch.isfinite(loss_e).all()
    ew_valid = edge_weight[valid_any]
    is_target = ew_valid != float(_cfg_get(train_cfg, "edge_weight_default", 1.0))
    if not bool(collect_log or debug):
        return loss_e, {
            "edge_id": int(edge_id),
            "loss_total_after_weight": 0.0,
            "valid_sample_count": 0,
            "is_target_edge": False,
            "updated_or_skipped": "updated",
            "skip_reason": "",
        }
    return loss_e, {
        "edge_id": int(edge_id),
        "pred_main_e_shape": tuple(pred_main_e.shape),
        "target_main_e_shape": tuple(target_main_e.shape),
        "mask_main_e_shape": tuple(mask_main_e.shape),
        "rho_pred_e_shape": tuple(rho_pred_e.shape) if rho_pred_e is not None else (),
        "h_pred_e_shape": tuple(h_pred_e.shape) if h_pred_e is not None else (),
        "main_mask_sum": float(mask_main_e.sum().detach().cpu().item()),
        "rho_mask_sum": float(rho_mask_e.sum().detach().cpu().item()) if rho_mask_e is not None else 0.0,
        "h_mask_sum": float(h_mask_e.sum().detach().cpu().item()) if h_mask_e is not None else 0.0,
        "loss_main_before_weight": _mean_valid(loss_main, valid_main),
        "mass_log_loss": _mean_valid(mass_log_raw, valid_mass_log),
        "mass_physical_loss": _mean_valid(mass_physical_loss, valid_mass_physical),
        "mass_physical_weight": mass_aux_weight,
        "mass_log_weight": mass_log_weight,
        "mass_total_loss": _mean_valid(
            mass_total_raw,
            valid_mass_log | valid_mass_physical,
        ),
        "loss_fraction_clr_before_weight": _mean_valid(loss_frac_clr, valid_frac_clr),
        "loss_fraction_log_before_weight": _mean_valid(loss_frac_log, valid_frac_log),
        "loss_fraction_closure_before_weight": _mean_valid(loss_frac_closure, valid_frac_closure),
        "loss_rho_before_weight": _mean_valid(loss_rho, valid_rho),
        "loss_h_before_weight": _mean_valid(loss_h, valid_h),
        "weighted_loss_main_before_edge_weight": _mean_valid(lm * loss_main, valid_main),
        "weighted_loss_fraction_clr_before_edge_weight": _mean_valid(
            lfrac_clr * loss_frac_clr,
            valid_frac_clr,
        ),
        "weighted_loss_fraction_log_before_edge_weight": _mean_valid(
            lfrac_log * loss_frac_log,
            valid_frac_log,
        ),
        "weighted_loss_fraction_closure_before_edge_weight": _mean_valid(
            lfrac_closure * loss_frac_closure,
            valid_frac_closure,
        ),
        "weighted_loss_rho_before_edge_weight": _mean_valid(lrho * loss_rho, valid_rho),
        "weighted_loss_h_before_edge_weight": _mean_valid(lh * loss_h, valid_h),
        "loss_volume_before_weight": _mean_valid(loss_volume, valid_volume),
        "loss_enthalpy_flow_before_weight": _mean_valid(loss_enthalpy_flow, valid_enthalpy_flow),
        "loss_atom_before_weight": _mean_valid(loss_atom, valid_atom),
        "loss_energy_before_weight": _mean_valid(loss_energy, valid_energy),
        "loss_rho_max": _max_valid(loss_rho, valid_rho),
        "loss_h_max": _max_valid(loss_h, valid_h),
        "loss_volume_max": _max_valid(loss_volume, valid_volume),
        "loss_enthalpy_flow_max": _max_valid(loss_enthalpy_flow, valid_enthalpy_flow),
        "loss_total_after_weight": float(loss_e.detach().cpu().item()),
        "loss_base_before_frac_penalty": _mean_valid(base_loss_per_sample, valid_any),
        "loss_total_after_frac_penalty": _mean_valid(loss_per_sample, valid_any),
        "loss_frac_penalty_total": _mean_valid(loss_frac_penalty, valid_frac_penalty),
        "frac_penalty_to_base_ratio": (
            _mean_valid(loss_frac_penalty, valid_frac_penalty)
            / max(abs(_mean_valid(base_loss_per_sample, valid_any)), 1.0e-12)
        ),
        "derived_mole_flow_to_target_ratio_mean": derived_mole_ratio,
        "derived_volume_flow_to_target_ratio_mean": derived_volume_ratio,
        "valid_sample_count": int(valid_any.sum().detach().cpu().item()),
        "edge_weight_min": float(ew_valid.min().detach().cpu().item()),
        "edge_weight_max": float(ew_valid.max().detach().cpu().item()),
        "edge_weight_mean": float(ew_valid.mean().detach().cpu().item()),
        "target_sample_count_for_this_edge": int(is_target.sum().detach().cpu().item()),
        "is_target_edge": bool(is_target.any().detach().cpu().item()),
        "updated_or_skipped": "updated",
        "skip_reason": "",
        **mass_aux_debug,
        **fraction_loss_debug,
        **mass_gradient_payload,
    }


def _grad_l2_norm(parameters: Any) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(torch.sum(g * g).cpu().item())
    return float(total**0.5)


def _hierarchical_gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    """Return one synchronized gradient-norm snapshot for the reduced PI path."""

    raw_model = getattr(model, "module", model)
    if getattr(getattr(raw_model, "edge_decoder", None), "hierarchical_pi_head", None) is None:
        return {}
    group_squares: dict[str, torch.Tensor] = {}
    for name, parameter in raw_model.named_parameters():
        if parameter.grad is None:
            continue
        groups: list[str] = []
        if name.startswith("encoder.edge_input_encoder."):
            groups.append("edge_encoder")
            if ".role_embedding." in name:
                groups.append("edge_role_embedding")
            elif ".stream_embedding." in name:
                groups.append("stream_id_embedding")
            elif ".structural_encoder." in name:
                groups.append("edge_structural_encoder")
            elif ".input_mlp." in name:
                groups.append("edge_fusion")
        elif name.startswith("encoder.input_encoder.role_embedding."):
            groups.append("node_role_embedding")
        elif name.startswith("encoder.input_encoder.unit_embedding."):
            groups.append("node_unit_embedding")
        elif name.startswith("encoder.input_encoder.operating_encoder."):
            groups.append("operating_encoder")
        elif name.startswith("encoder.input_encoder.input_mlp."):
            groups.append("node_fusion")
        elif name.startswith("encoder.layers."):
            groups.append("gnn")
        elif (
            name.startswith("encoder.set2set_pool.")
            or name.startswith("encoder.attention_pool.")
        ):
            groups.append("global_pool")
        elif name.startswith("encoder.final_projection."):
            groups.append("node_final_projection")
        elif name.startswith("edge_decoder.hierarchical_global_projection."):
            groups.append("global_projection")
        elif (
            name.startswith("edge_decoder.hierarchical_pi_head.shared_")
        ):
            groups.append("shared_edge_decoder")
        elif name.startswith("edge_decoder.hierarchical_pi_head.condition_head."):
            groups.append("condition_head")
        elif name.startswith("edge_decoder.hierarchical_pi_head.fraction_head."):
            groups.append("fraction_head")
        elif name.startswith("edge_decoder.hierarchical_pi_head.mass_head."):
            groups.append("mass_flow_head")
        elif name.startswith("edge_decoder.hierarchical_pi_head.volume_head."):
            groups.append("volume_flow_head")
        else:
            continue
        square = parameter.grad.detach().float().square().sum()
        for group in groups:
            group_squares[group] = (
                group_squares.get(group, square.new_zeros(())) + square
            )
    if not group_squares:
        return {}
    names = sorted(group_squares)
    values = torch.stack([group_squares[name].sqrt() for name in names]).detach().cpu().tolist()
    return {name: float(value) for name, value in zip(names, values)}


def _mass_dual_gradient_group(parameter_name: str) -> str | None:
    if parameter_name.startswith("edge_decoder.hierarchical_pi_head.mass_head."):
        return "mass_flow_head"
    if parameter_name.startswith("edge_decoder.hierarchical_pi_head.shared_"):
        return "shared_edge_decoder"
    if parameter_name.startswith("encoder.layers."):
        return "flow_gnn"
    if parameter_name.startswith("encoder.input_encoder."):
        return "node_encoder"
    if parameter_name.startswith("encoder.edge_input_encoder."):
        return "edge_encoder"
    return None


def mass_dual_space_gradient_diagnostics(
    *,
    model: torch.nn.Module,
    mass_log_objective: torch.Tensor,
    mass_physical_objective: torch.Tensor,
) -> dict[str, float]:
    """Compare log/physical Mass gradients without taking an optimizer step."""
    raw_model = getattr(model, "module", model)
    named = [
        (name, parameter)
        for name, parameter in raw_model.named_parameters()
        if parameter.requires_grad
    ]
    params = [parameter for _, parameter in named]
    log_grads = torch.autograd.grad(
        mass_log_objective,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    physical_grads = torch.autograd.grad(
        mass_physical_objective,
        params,
        retain_graph=True,
        allow_unused=True,
    )
    log_squares: dict[str, float] = {}
    physical_squares: dict[str, float] = {}
    dot = 0.0
    log_total = 0.0
    physical_total = 0.0
    for (name, _), grad_log, grad_physical in zip(named, log_grads, physical_grads):
        group = _mass_dual_gradient_group(name)
        if grad_log is not None:
            log_square = float(grad_log.detach().float().square().sum().cpu())
            log_total += log_square
            if group is not None:
                log_squares[group] = log_squares.get(group, 0.0) + log_square
        if grad_physical is not None:
            physical_square = float(grad_physical.detach().float().square().sum().cpu())
            physical_total += physical_square
            if group is not None:
                physical_squares[group] = physical_squares.get(group, 0.0) + physical_square
        if grad_log is not None and grad_physical is not None:
            dot += float(
                (grad_log.detach().float() * grad_physical.detach().float()).sum().cpu()
            )
    result = {
        "mass_grad_log_total": math.sqrt(log_total),
        "mass_grad_physical_total": math.sqrt(physical_total),
        "mass_grad_log_physical_cosine": (
            dot / max(math.sqrt(log_total * physical_total), 1.0e-30)
        ),
    }
    for group in sorted(set(log_squares) | set(physical_squares)):
        result[f"mass_grad_log_{group}"] = math.sqrt(log_squares.get(group, 0.0))
        result[f"mass_grad_physical_{group}"] = math.sqrt(
            physical_squares.get(group, 0.0)
        )
    return result


def _accumulate_hierarchical_gradient_diagnostics(
    diagnostics: MutableMapping[str, float],
    *,
    update_role: str,
    pre_clip: Mapping[str, float],
    pre_clip_total: float,
    grad_clip: float | None,
) -> None:
    clip_limit = float(grad_clip or 0.0)
    clip_scale = (
        min(1.0, clip_limit / (float(pre_clip_total) + 1.0e-12))
        if clip_limit > 0.0 and math.isfinite(float(pre_clip_total))
        else 1.0
    )
    prefix = f"grad_{update_role}"
    diagnostics[f"{prefix}_update_count"] = diagnostics.get(f"{prefix}_update_count", 0.0) + 1.0
    if clip_scale < 1.0:
        diagnostics[f"{prefix}_clip_count"] = diagnostics.get(f"{prefix}_clip_count", 0.0) + 1.0
    for group, value in pre_clip.items():
        pre_key = f"{prefix}_pre_clip_{group}_sum"
        post_key = f"{prefix}_post_clip_{group}_sum"
        diagnostics[pre_key] = diagnostics.get(pre_key, 0.0) + float(value)
        diagnostics[post_key] = diagnostics.get(post_key, 0.0) + float(value) * clip_scale


@dataclass
class EdgeStepPIBatchResult:
    loss_items: MutableMapping[str, torch.Tensor]
    debug_rows: list[dict[str, Any]] = field(default_factory=list)


_EDGE_STEP_PI_SUM_KEYS: tuple[tuple[str, str], ...] = (
    ("loss_main_mean", "loss_main_before_weight"),
    ("mass_log_loss_mean", "mass_log_loss"),
    ("mass_physical_loss_mean", "mass_physical_loss"),
    ("mass_total_loss_mean", "mass_total_loss"),
    ("mass_physical_weight_mean", "mass_physical_weight"),
    ("mass_log_weight_mean", "mass_log_weight"),
    ("mass_pred_physical_mean", "mass_flow_pred_physical_mean"),
    ("mass_true_physical_mean", "mass_flow_true_physical_mean"),
    ("mass_physical_loss_p95_sum", "mass_flow_physical_loss_p95_sum"),
    ("mass_physical_p95_count", "mass_flow_physical_p95_count"),
    ("mass_physical_loss_p99_sum", "mass_flow_physical_loss_p99_sum"),
    ("mass_physical_p99_count", "mass_flow_physical_p99_count"),
    ("loss_fraction_clr_mean", "loss_fraction_clr_before_weight"),
    ("weighted_loss_fraction_clr_mean", "weighted_loss_fraction_clr_before_edge_weight"),
    ("loss_fraction_log_mean", "loss_fraction_log_before_weight"),
    ("weighted_loss_fraction_log_mean", "weighted_loss_fraction_log_before_edge_weight"),
    ("loss_fraction_closure_mean", "loss_fraction_closure_before_weight"),
    ("weighted_loss_fraction_closure_mean", "weighted_loss_fraction_closure_before_edge_weight"),
    ("loss_frac_penalty_total", "loss_frac_penalty_total"),
    ("loss_base_before_frac_penalty", "loss_base_before_frac_penalty"),
    ("loss_total_after_frac_penalty", "loss_total_after_frac_penalty"),
    ("frac_penalty_ramp_weight", "frac_penalty_ramp_weight"),
    ("frac_penalty_to_base_ratio", "frac_penalty_to_base_ratio"),
    ("loss_co_pos", "loss_co_pos"),
    ("weighted_loss_co_pos", "weighted_loss_co_pos"),
    ("co_pos_gate_mean", "co_pos_gate_mean"),
    ("co_pos_gate_sum", "co_pos_gate_sum"),
    ("co_pos_effective_count", "co_pos_effective_count"),
    ("co_true_positive_count_true_gt_001", "co_true_positive_count_true_gt_001"),
    ("co_pred_mean_on_positive", "co_pred_mean_on_positive"),
    ("co_true_mean_on_positive", "co_true_mean_on_positive"),
    ("loss_ch4_fp", "loss_ch4_fp"),
    ("loss_co2_fp", "loss_co2_fp"),
    ("loss_h2_fp", "loss_h2_fp"),
    ("weighted_loss_ch4_fp", "weighted_loss_ch4_fp"),
    ("weighted_loss_co2_fp", "weighted_loss_co2_fp"),
    ("weighted_loss_h2_fp", "weighted_loss_h2_fp"),
    ("zero_gate_ch4_mean", "zero_gate_ch4_mean"),
    ("zero_gate_co2_mean", "zero_gate_co2_mean"),
    ("zero_gate_h2_mean", "zero_gate_h2_mean"),
    ("fp_ch4_active_count_pred_gt_margin", "fp_ch4_active_count_pred_gt_margin"),
    ("fp_co2_active_count_pred_gt_margin", "fp_co2_active_count_pred_gt_margin"),
    ("fp_h2_active_count_pred_gt_margin", "fp_h2_active_count_pred_gt_margin"),
    ("loss_co2_pos", "loss_co2_pos"),
    ("weighted_loss_co2_pos", "weighted_loss_co2_pos"),
    ("co2_pos_gate_mean", "co2_pos_gate_mean"),
    ("co2_pos_gate_sum", "co2_pos_gate_sum"),
    ("co2_true_high_count_true_gt_005", "co2_true_high_count_true_gt_005"),
    ("co2_true_high_count_true_gt_02", "co2_true_high_count_true_gt_02"),
    ("co2_pred_mean_on_high", "co2_pred_mean_on_high"),
    ("loss_rho_mean", "loss_rho_before_weight"),
    ("loss_h_mean", "loss_h_before_weight"),
    ("loss_volume_mean", "loss_volume_before_weight"),
    ("loss_enthalpy_flow_mean", "loss_enthalpy_flow_before_weight"),
    ("loss_atom_mean", "loss_atom_before_weight"),
    ("loss_energy_mean", "loss_energy_before_weight"),
    ("edge_weight_mean", "edge_weight_mean"),
    ("relu_l1_all_zero_row_count", "relu_l1_all_zero_row_count"),
    ("relu_l1_all_zero_row_ratio", "relu_l1_all_zero_row_ratio"),
    ("relu_l1_denominator_mean", "relu_l1_denominator_mean"),
    ("pred_frac_sum_mean", "pred_frac_sum_mean"),
    ("pred_frac_negative_count", "pred_frac_negative_count"),
    ("pred_frac_nan_count", "pred_frac_nan_count"),
    ("pred_frac_inf_count", "pred_frac_inf_count"),
)


_EDGE_STEP_PI_MAX_KEYS: tuple[tuple[str, str], ...] = (
    ("loss_rho_max", "loss_rho_max"),
    ("loss_h_max", "loss_h_max"),
    ("loss_volume_max", "loss_volume_max"),
    ("loss_enthalpy_flow_max", "loss_enthalpy_flow_max"),
    ("edge_weight_min", "edge_weight_min"),
    ("edge_weight_max", "edge_weight_max"),
    ("relu_l1_denominator_min", "relu_l1_denominator_min"),
    ("pred_frac_sum_min", "pred_frac_sum_min"),
    ("pred_frac_sum_max", "pred_frac_sum_max"),
)


def _accumulate_edge_step_pi_log(
    sums: MutableMapping[str, float],
    maxes: MutableMapping[str, float],
    log: Mapping[str, Any],
) -> None:
    for key, src in _EDGE_STEP_PI_SUM_KEYS:
        sums[key] = sums.get(key, 0.0) + float(log.get(src, 0.0))
    for key, src in _EDGE_STEP_PI_MAX_KEYS:
        val = float(log.get(src, 0.0))
        if key.endswith("_min"):
            maxes[key] = min(maxes.get(key, val), val)
        else:
            maxes[key] = max(maxes.get(key, val), val)


def _backward_clip_optimizer_step(
    *,
    loss: torch.Tensor,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    use_amp: bool,
    grad_clip: float | None,
    train_cfg: Any,
    gradient_diagnostics: MutableMapping[str, float] | None = None,
    gradient_update_role: str | None = None,
) -> tuple[float, bool]:
    grad_norm = 0.0
    if bool(use_amp):
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        module_grad_norms = (
            _hierarchical_gradient_norms(model)
            if gradient_diagnostics is not None and gradient_update_role
            else {}
        )
        if grad_clip is not None and float(grad_clip) > 0:
            grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            grad_norm = float(grad_norm_t.detach().cpu())
        else:
            grad_norm = _grad_l2_norm(model.parameters())
        if module_grad_norms and gradient_diagnostics is not None and gradient_update_role:
            _accumulate_hierarchical_gradient_diagnostics(
                gradient_diagnostics,
                update_role=gradient_update_role,
                pre_clip=module_grad_norms,
                pre_clip_total=grad_norm,
                grad_clip=grad_clip,
            )
        step_applied = math.isfinite(grad_norm)
        if step_applied:
            scaler.step(optimizer)
        else:
            optimizer.zero_grad(set_to_none=True)
        scaler.update()
    else:
        loss.backward()
        module_grad_norms = (
            _hierarchical_gradient_norms(model)
            if gradient_diagnostics is not None and gradient_update_role
            else {}
        )
        if grad_clip is not None and float(grad_clip) > 0:
            grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            grad_norm = float(grad_norm_t.detach().cpu())
        else:
            grad_norm = _grad_l2_norm(model.parameters())
        if module_grad_norms and gradient_diagnostics is not None and gradient_update_role:
            _accumulate_hierarchical_gradient_diagnostics(
                gradient_diagnostics,
                update_role=gradient_update_role,
                pre_clip=module_grad_norms,
                pre_clip_total=grad_norm,
                grad_clip=grad_clip,
            )
        step_applied = math.isfinite(grad_norm)
        if step_applied:
            optimizer.step()
        else:
            optimizer.zero_grad(set_to_none=True)
    if (
        step_applied
        and scheduler is not None
        and str(_cfg_get(train_cfg, "scheduler_step_unit", "epoch")).lower() == "optimizer_step"
    ):
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
    return grad_norm, step_applied


def _edge_group_is_target(
    *,
    edge_indices: Sequence[int],
    edge_weight_vector: torch.Tensor,
    edge_weight_default: float,
) -> bool:
    idx = torch.tensor([int(i) for i in edge_indices], device=edge_weight_vector.device, dtype=torch.long)
    weights = edge_weight_vector.index_select(0, idx)
    default = torch.as_tensor(edge_weight_default, device=weights.device, dtype=weights.dtype)
    return bool((~torch.isclose(weights, default.expand_as(weights))).any().detach().cpu().item())


def _edge_group_is_target_cpu(
    *,
    edge_indices: Sequence[int],
    target_edge_mask: Sequence[bool],
) -> bool:
    n = len(target_edge_mask)
    return any(0 <= int(i) < n and bool(target_edge_mask[int(i)]) for i in edge_indices)


def _sort_edge_groups(
    edge_groups: list[tuple[str, list[int]]],
    *,
    order: str,
) -> list[tuple[str, list[int]]]:
    if str(order).strip().lower() == "canonical_edge_index":
        return sorted(edge_groups, key=lambda item: (min(int(i) for i in item[1]), str(item[0])))
    return sorted(edge_groups, key=lambda item: (str(item[0]), min(int(i) for i in item[1])))


def _sample_hybrid_cfg(train_cfg: Any) -> Any:
    return _cfg_get(train_cfg, "sample_hybrid_target_edge_step_pi", {}) or {}


def _node_pinn_optimization_cfg(train_cfg: Any) -> Any:
    return _cfg_get(train_cfg, "node_pinn_optimization", {}) or {}


def compose_joint_node_pinn_loss(
    supervised_anchor_loss: torch.Tensor,
    scheduled_node_loss: torch.Tensor,
    *,
    supervised_anchor_weight: float,
    node_outer_weight: float,
) -> torch.Tensor:
    """Compose the joint objective.

    ``scheduled_node_loss`` already includes the epoch PINN schedule because
    the caller receives the scaled training config.
    """

    return (
        float(supervised_anchor_weight) * supervised_anchor_loss
        + float(node_outer_weight) * scheduled_node_loss
    )


def _scalar(value: float, device: torch.device) -> torch.Tensor:
    return torch.tensor(float(value), dtype=torch.float32, device=device)


def _collapse_scale_values(
    *,
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    target_masks: Mapping[str, torch.Tensor],
    train_cfg: Any,
    data_cfg: Any,
    edge_target_columns: Sequence[str] | None,
    normalizer: Mapping[str, Any] | None,
) -> dict[str, torch.Tensor]:
    columns = list(edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
    physical = build_pi_physical_outputs(
        outputs,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        normalizer=normalizer,
    )
    y = targets["edge_stream"].to(device=physical["mass_flow"].device, dtype=physical["mass_flow"].dtype)
    edge_mask = target_masks["edge_stream"].to(device=y.device).reshape(-1) > 0.5
    values: dict[str, torch.Tensor] = {}
    for name, pred_key in (("mass_flow", "mass_flow"), ("enthalpy", "h")):
        column = "Mass_Flow" if name == "mass_flow" else "Enthalpy"
        pred_value = physical.get(pred_key)
        if column not in columns or pred_value is None:
            values[f"{name}_pred_abs"] = y.new_empty((0,))
            values[f"{name}_true_abs"] = y.new_empty((0,))
            continue
        true_value = _take_columns(y, [columns.index(column)])
        true_value = _inverse_if_needed(true_value, column, normalizer, data_cfg)
        valid = edge_mask & torch.isfinite(true_value.reshape(-1)) & torch.isfinite(pred_value.reshape(-1))
        values[f"{name}_pred_abs"] = pred_value.reshape(-1)[valid].detach().abs()
        values[f"{name}_true_abs"] = true_value.reshape(-1)[valid].detach().abs()
    return values


def _median_ratio(pred_abs: torch.Tensor, true_abs: torch.Tensor) -> tuple[float, float, float]:
    if pred_abs.numel() == 0 or true_abs.numel() == 0:
        return 0.0, 0.0, 0.0
    pred_median = float(torch.median(pred_abs).cpu())
    true_median = float(torch.median(true_abs).cpu())
    ratio = pred_median / max(true_median, 1.0e-12)
    return pred_median, true_median, ratio


def _gradient_group(name: str) -> str:
    lowered = name.lower()
    if "mass_flow" in lowered or "flow_head" in lowered:
        return "mass_flow_head"
    if "enthalpy" in lowered or "h_head" in lowered:
        return "enthalpy_head"
    if any(token in lowered for token in ("encoder", "gnn", "message", "node_")):
        return "shared_encoder"
    return "other"


def _autograd_group_norms(
    model: torch.nn.Module,
    loss: torch.Tensor,
) -> dict[str, float]:
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(
        loss,
        [parameter for _, parameter in named_parameters],
        retain_graph=True,
        allow_unused=True,
    )
    sums = {"total": 0.0, "shared_encoder": 0.0, "mass_flow_head": 0.0, "enthalpy_head": 0.0, "other": 0.0}
    for (name, _), gradient in zip(named_parameters, gradients):
        if gradient is None:
            continue
        squared = float(gradient.detach().float().square().sum().cpu())
        sums["total"] += squared
        sums[_gradient_group(name)] += squared
    return {key: math.sqrt(value) for key, value in sums.items()}


def _profile_now(device: torch.device, enabled: bool) -> float:
    if enabled and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _profile_add(timings: MutableMapping[str, float], key: str, start: float, device: torch.device, enabled: bool) -> None:
    if not enabled:
        return
    end = _profile_now(device, enabled)
    timings[key] = timings.get(key, 0.0) + float(end - start)


def _write_fraction_penalty_diagnostics_csv(out_dir: Path, metric_rows: Any) -> None:
    try:
        import pandas as pd
    except Exception:
        return
    rows = metric_rows if hasattr(metric_rows, "columns") else pd.DataFrame(metric_rows)
    if rows is None or rows.empty or "property_name" not in rows.columns:
        return
    props = ["Frac_CO", "Frac_CH4", "Frac_CO2", "Frac_H2"]
    fp_threshold = {"Frac_CO": 0.005, "Frac_CH4": 0.005, "Frac_CO2": 0.005, "Frac_H2": 0.01}
    pos_threshold = {"Frac_CO": 0.01, "Frac_CH4": 0.01, "Frac_CO2": 0.05, "Frac_H2": 0.01}
    summary: list[dict[str, Any]] = []
    for split, split_df in rows.groupby("split", dropna=False):
        for prop in props:
            mask = split_df.get("mask")
            valid_mask = mask.astype(float) > 0 if mask is not None else True
            sub = split_df[(split_df["property_name"] == prop) & valid_mask]
            if sub.empty:
                summary.append(
                    {
                        "split": split,
                        "property": prop,
                        "count": 0,
                        "true_mean": math.nan,
                        "pred_mean": math.nan,
                        "true_std": math.nan,
                        "pred_std": math.nan,
                        "pred_std_over_true_std": math.nan,
                        "near_zero_count": 0,
                        "near_zero_pred_mean": math.nan,
                        "near_zero_false_positive_rate": math.nan,
                        "positive_count": 0,
                        "positive_mae": math.nan,
                        "positive_bias": math.nan,
                    }
                )
                continue
            true = sub["y_true"].astype(float)
            pred = sub["y_pred"].astype(float)
            true_std = float(true.std(ddof=0))
            pred_std = float(pred.std(ddof=0))
            near_zero = true < 1.0e-3
            false_positive = near_zero & (pred > fp_threshold[prop])
            positive = true > pos_threshold[prop]
            positive_err = pred[positive] - true[positive]
            base = {
                "split": split,
                "property": prop,
                "count": int(len(sub)),
                "true_mean": float(true.mean()),
                "pred_mean": float(pred.mean()),
                "true_std": true_std,
                "pred_std": pred_std,
                "pred_std_over_true_std": pred_std / true_std if true_std > 0.0 else math.nan,
                "near_zero_count": int(near_zero.sum()),
                "near_zero_pred_mean": float(pred[near_zero].mean()) if bool(near_zero.any()) else math.nan,
                "near_zero_false_positive_rate": (
                    float(false_positive.sum()) / float(near_zero.sum()) if int(near_zero.sum()) > 0 else math.nan
                ),
                "positive_count": int(positive.sum()),
                "positive_mae": float(positive_err.abs().mean()) if bool(positive.any()) else math.nan,
                "positive_bias": float(positive_err.mean()) if bool(positive.any()) else math.nan,
            }
            summary.append(base)
            if prop == "Frac_CO2":
                high = true > 0.2
                high_err = pred[high] - true[high]
                summary.append(
                    {
                        **base,
                        "property": "Frac_CO2_true_gt_02",
                        "count": int(high.sum()),
                        "true_mean": float(true[high].mean()) if bool(high.any()) else math.nan,
                        "pred_mean": float(pred[high].mean()) if bool(high.any()) else math.nan,
                        "positive_count": int(high.sum()),
                        "positive_mae": float(high_err.abs().mean()) if bool(high.any()) else math.nan,
                        "positive_bias": float(high_err.mean()) if bool(high.any()) else math.nan,
                    }
                )
    if summary:
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary).to_csv(out_dir / "fraction_penalty_diagnostics.csv", index=False)


def compute_pi_node_balance_losses(
    *,
    outputs: Mapping[str, torch.Tensor],
    batch_data: Mapping[str, torch.Tensor],
    train_cfg: Any,
    data_cfg: Any,
    normalizer: Mapping[str, Any] | None,
    edge_target_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    required = ("edge_index", "edge_batch", "batch", "x_unit")
    missing = [key for key in required if key not in batch_data]
    if missing:
        raise RuntimeError(f"Node PINN losses require batch tensors: {missing}")
    physical = build_pi_physical_outputs(
        outputs,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        normalizer=normalizer,
    )
    species = resolve_pi_species_order(data_cfg, train_cfg, outputs)
    physical_targets: dict[str, torch.Tensor] | None = None
    y_edge_true = batch_data.get("y_edge_true")
    if y_edge_true is not None:
        cols = list(edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
        if int(y_edge_true.shape[-1]) == len(cols):
            device = physical["mass_flow"].device
            dtype = physical["mass_flow"].dtype
            y = y_edge_true.to(device=device, dtype=dtype)
            mass_idx = cols.index("Mass_Flow") if "Mass_Flow" in cols else None
            enthalpy_idx = cols.index("Enthalpy") if "Enthalpy" in cols else None
            frac_idx = [cols.index(f"Frac_{s}") for s in species if f"Frac_{s}" in cols]
            if mass_idx is not None and len(frac_idx) == len(species):
                mass_true = y[..., mass_idx : mass_idx + 1]
                frac_true = y.index_select(-1, torch.tensor(frac_idx, device=device, dtype=torch.long))
                physical_targets = {"mass_flow": mass_true, "frac": frac_true}
                if enthalpy_idx is not None:
                    h_true = y[..., enthalpy_idx : enthalpy_idx + 1]
                    h_basis = str(_cfg_get(train_cfg, "h_basis", "mass_specific"))
                    if h_basis == "mass_specific":
                        physical_targets["enthalpy_flow"] = mass_true * h_true
                    elif h_basis == "molar":
                        mw = _mw_tensor(species, data_cfg, train_cfg, device, dtype)
                        mw_bar = (frac_true * mw.view(1, -1)).sum(dim=-1, keepdim=True)
                        mole_true = mass_true / mw_bar.clamp_min(float(_cfg_get(train_cfg, "eps", 1.0e-8)))
                        physical_targets["enthalpy_flow"] = mole_true * h_true
    return compute_node_balance_pinn_losses(
        physical_outputs=physical,
        edge_index=batch_data["edge_index"],
        edge_batch_or_graph_id=batch_data["edge_batch"],
        node_batch_or_graph_id=batch_data["batch"],
        node_unit_type=batch_data["x_unit"],
        species_order=species,
        train_cfg=train_cfg,
        physical_targets=physical_targets,
        node_q=batch_data.get("node_q"),
        node_w=batch_data.get("node_w"),
        node_qw_valid_mask=batch_data.get("node_qw_valid_mask"),
        node_balance_exclude_mass=batch_data.get("node_balance_exclude_mass"),
        node_balance_exclude_component=batch_data.get("node_balance_exclude_component"),
        node_balance_exclude_atom=batch_data.get("node_balance_exclude_atom"),
        node_balance_exclude_energy=batch_data.get("node_balance_exclude_energy"),
        edge_pinn_mask=batch_data.get("edge_pinn_mask"),
    )


def train_one_batch_sample_hybrid_target_edge_step_pi(
    *,
    model: torch.nn.Module,
    batch_data: Any,
    task_inputs: Any,
    targets: Mapping[str, torch.Tensor],
    target_masks: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    train_cfg: Any,
    data_cfg: Any,
    device: torch.device,
    use_amp: bool,
    grad_clip: float | None,
    edge_export_meta: Any,
    edge_target_columns: Sequence[str] | None,
    normalizer: Mapping[str, Any] | None = None,
    rng: random.Random | None = None,
    global_step: int | None = None,
    total_train_steps: int | None = None,
    pinn_schedule_multiplier: float = 1.0,
    optimizer_step_budget: int | None = None,
    debug: bool = False,
) -> EdgeStepPIBatchResult:
    """Train one batch containing exactly one graph sample.

    Update schedule:
      1. all valid non-target canonical edge groups as one macro-mean update
      2. each target canonical edge group as one independent update
      3. optional node PINN update as one independent update

    Each optimizer update performs a fresh forward pass.  This intentionally
    avoids retain_graph=True and avoids reusing stale activations after
    optimizer.step().
    """

    edge_batch = batch_data.get("edge_batch") if isinstance(batch_data, Mapping) else None
    if edge_batch is None:
        raise RuntimeError("sample_hybrid_target_edge_step_pi requires batch_data['edge_batch'].")
    graph_ids = torch.unique(edge_batch.detach())
    if int(graph_ids.numel()) != 1:
        raise RuntimeError(
            "sample_hybrid_target_edge_step_pi requires batch_size=1, "
            f"but received {int(graph_ids.numel())} graph samples."
        )

    n_edges = _edge_count_from_target(targets["edge_stream"])
    profile_timing = False
    timings: dict[str, float] = {}
    timing_start = _profile_now(device, False)
    edge_groups = _edge_groups_from_export_meta(edge_export_meta, n_edges)
    hybrid_cfg = _sample_hybrid_cfg(train_cfg)
    mass_aux_cfg = _cfg_get(train_cfg, "mass_flow_physical_auxiliary", None)
    profile_timing = bool(_cfg_get(hybrid_cfg, "profile_timing", False))
    if profile_timing:
        timing_start = _profile_now(device, profile_timing)
    order = str(_cfg_get(hybrid_cfg, "target_update_order", "canonical_edge_id")).strip().lower()
    allow_duplicates = bool(_cfg_get(hybrid_cfg, "allow_duplicate_canonical_edge_rows", True))
    log_details = bool(_cfg_get(hybrid_cfg, "log_sample_update_details", True))
    collect_update_diagnostics = bool(
        _cfg_get(hybrid_cfg, "collect_update_diagnostics", log_details)
        or log_details
        or debug
        or _cfg_get(mass_aux_cfg, "gradient_diagnostics", False)
    )
    use_existing_target_weight = bool(_cfg_get(hybrid_cfg, "use_existing_target_edge_weight", False))
    if not allow_duplicates:
        duplicate_groups = [(label, idxs) for label, idxs in edge_groups if len(idxs) > 1]
        if duplicate_groups:
            preview = ", ".join(f"{label}:{len(idxs)}" for label, idxs in duplicate_groups[:5])
            raise RuntimeError(
                "sample_hybrid_target_edge_step_pi found duplicate canonical edge rows "
                f"while allow_duplicate_canonical_edge_rows=false: {preview}"
            )

    edge_weight_vector, weight_info = build_edge_step_edge_weight_vector(
        train_cfg=train_cfg,
        edge_export_meta=edge_export_meta,
        edge_target_columns=edge_target_columns,
        n_edges=n_edges,
        device=device,
    )
    target_edge_mask, _target_mask_info = build_target_edge_boolean_mask(
        train_cfg=train_cfg,
        edge_export_meta=edge_export_meta,
        edge_target_columns=edge_target_columns,
        n_edges=n_edges,
        device=device,
    )
    model_batch_data = dict(batch_data)
    model_batch_data["target_edge_mask"] = target_edge_mask
    edge_weight_default = float(_cfg_get(train_cfg, "edge_weight_default", 1.0))
    target_edge_mask_cpu = target_edge_mask.detach().cpu().tolist()
    non_target_groups: list[tuple[str, list[int]]] = []
    target_groups: list[tuple[str, list[int]]] = []
    for label, idxs in edge_groups:
        idx_list = [int(i) for i in idxs]
        if _edge_group_is_target_cpu(
            edge_indices=idx_list,
            target_edge_mask=target_edge_mask_cpu,
        ):
            target_groups.append((label, idx_list))
        else:
            non_target_groups.append((label, idx_list))
    target_groups = _sort_edge_groups(target_groups, order=order)
    non_target_groups = _sort_edge_groups(non_target_groups, order="canonical_edge_index")
    _profile_add(timings, "profile_grouping_sec", timing_start, device, profile_timing)

    criterion_main = build_pi_main_criterion(train_cfg)
    losses: list[float] = []
    target_losses: list[float] = []
    non_target_losses: list[float] = []
    debug_rows: list[dict[str, Any]] = []
    sums: dict[str, float] = {}
    maxes: dict[str, float] = {}
    skipped = 0
    edge_optimizer_steps = 0
    non_target_optimizer_steps = 0
    target_optimizer_steps = 0

    if optimizer_step_budget is not None and int(optimizer_step_budget) < 1:
        raise ValueError("optimizer_step_budget must be positive when provided")

    def _optimizer_budget_available() -> bool:
        if optimizer_step_budget is None:
            return True
        return (edge_optimizer_steps + node_optimizer_steps) < int(optimizer_step_budget)

    node_optimizer_steps = 0
    node_outputs: Mapping[str, torch.Tensor] | None = None
    forward_count = 0
    grad_norm_last = 0.0
    component_log_count = 0
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    last_outputs: Mapping[str, torch.Tensor] | None = None
    neutral_edge_weight_vector = torch.full_like(edge_weight_vector, edge_weight_default)
    gradient_diagnostics: dict[str, float] = {}
    log_module_gradient_norms = bool(_cfg_get(hybrid_cfg, "log_module_gradient_norms", False))
    mass_gradient_epoch = int(_cfg_get(mass_aux_cfg, "current_epoch", 1))
    mass_gradient_pending = bool(
        _cfg_get(mass_aux_cfg, "gradient_diagnostics", False)
        and _cfg_get(mass_aux_cfg, "_gradient_diagnostics_collected_epoch", None)
        != mass_gradient_epoch
    )
    cached_mass_gradient_diagnostics = _cfg_get(
        mass_aux_cfg,
        "_gradient_diagnostics_values",
        {},
    )
    if isinstance(cached_mass_gradient_diagnostics, Mapping):
        gradient_diagnostics.update(
            {
                str(key): float(value)
                for key, value in cached_mass_gradient_diagnostics.items()
            }
        )

    edge_loss_needs_physical = bool(
        collect_update_diagnostics
        or debug
        or _cfg_get(mass_aux_cfg, "enabled", False)
        or _cfg_get(train_cfg, "use_rho_loss", False)
        or _cfg_get(train_cfg, "use_h_loss", False)
        or _cfg_get(train_cfg, "use_volume_loss", False)
        or _cfg_get(train_cfg, "use_enthalpy_flow_loss", False)
    )
    static_pi_targets: Mapping[str, torch.Tensor | None] | None = None
    static_phys_targets: Mapping[str, torch.Tensor | None] | None = None

    if non_target_groups and _optimizer_budget_available():
        optimizer.zero_grad(set_to_none=True)
        timing_start = _profile_now(device, profile_timing)
        with torch.amp.autocast(device_type=amp_device, enabled=bool(use_amp)):
            outputs = model(model_batch_data, task_inputs=task_inputs)
            outputs = _ensure_pi_output_keys(
                outputs,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                edge_target_columns=edge_target_columns,
                normalizer=normalizer,
            )
            last_outputs = outputs
            forward_count += 1
            if static_pi_targets is None:
                static_pi_targets = get_pi_targets(
                    targets=targets, target_masks=target_masks, outputs=outputs,
                    train_cfg=train_cfg, data_cfg=data_cfg,
                    edge_target_columns=edge_target_columns,
                )
                if edge_loss_needs_physical:
                    static_phys_targets = build_pi_physical_targets(
                        static_pi_targets, train_cfg=train_cfg, data_cfg=data_cfg,
                        normalizer=normalizer,
                    )
            pi_targets_cache = static_pi_targets
            phys_outputs_cache = (
                build_pi_physical_outputs(
                    outputs, train_cfg=train_cfg, data_cfg=data_cfg,
                    normalizer=normalizer,
                )
                if edge_loss_needs_physical else None
            )
            phys_targets_cache = static_phys_targets
            group_losses: list[torch.Tensor] = []
            group_logs: list[dict[str, Any]] = []
            group_mass_log_objectives: list[torch.Tensor] = []
            group_mass_physical_objectives: list[torch.Tensor] = []
            for group_label, edge_indices in non_target_groups:
                edge_id = int(edge_indices[0])
                loss_e, log = compute_single_edge_pinn_loss(
                    outputs=outputs,
                    targets=targets,
                    target_masks=target_masks,
                    edge_id=edge_id,
                    edge_indices=edge_indices,
                    train_cfg=train_cfg,
                    data_cfg=data_cfg,
                    edge_target_columns=edge_target_columns,
                    normalizer=normalizer,
                    criterion_main=criterion_main,
                    edge_weight_vector=edge_weight_vector,
                    global_step=global_step,
                    total_train_steps=total_train_steps,
                    debug=debug,
                    collect_log=collect_update_diagnostics,
                    pi_targets_cache=pi_targets_cache,
                    phys_outputs_cache=phys_outputs_cache,
                    phys_targets_cache=phys_targets_cache,
                )
                if loss_e is None:
                    skipped += 1
                    if log_details:
                        debug_rows.append(
                            dict(
                                log,
                                edge_group=group_label,
                                edge_group_size=len(edge_indices),
                                update_role="non_target_macro",
                                updated_or_skipped="skipped",
                            )
                        )
                    continue
                mass_log_objective = log.pop("_mass_log_objective_tensor", None)
                mass_physical_objective = log.pop(
                    "_mass_physical_objective_tensor",
                    None,
                )
                if mass_log_objective is not None and mass_physical_objective is not None:
                    group_mass_log_objectives.append(mass_log_objective)
                    group_mass_physical_objectives.append(mass_physical_objective)
                group_losses.append(loss_e)
                group_logs.append(dict(log, edge_group=group_label, edge_group_size=len(edge_indices)))
        _profile_add(timings, "profile_non_target_forward_loss_sec", timing_start, device, profile_timing)
        if group_losses:
            loss_non_target = torch.stack(group_losses).mean()
            if (
                mass_gradient_pending
                and group_mass_log_objectives
                and group_mass_physical_objectives
            ):
                mass_gradient_values = mass_dual_space_gradient_diagnostics(
                    model=model,
                    mass_log_objective=torch.stack(group_mass_log_objectives).mean(),
                    mass_physical_objective=torch.stack(
                        group_mass_physical_objectives
                    ).mean(),
                )
                gradient_diagnostics.update(mass_gradient_values)
                setattr(
                    mass_aux_cfg,
                    "_gradient_diagnostics_values",
                    dict(mass_gradient_values),
                )
                setattr(
                    mass_aux_cfg,
                    "_gradient_diagnostics_collected_epoch",
                    mass_gradient_epoch,
                )
                mass_gradient_pending = False
            timing_start = _profile_now(device, profile_timing)
            grad_norm_last, step_applied = _backward_clip_optimizer_step(
                loss=loss_non_target,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                use_amp=use_amp,
                grad_clip=grad_clip,
                train_cfg=train_cfg,
                gradient_diagnostics=gradient_diagnostics if log_module_gradient_norms else None,
                gradient_update_role="non_target",
            )
            _profile_add(timings, "profile_non_target_backward_step_sec", timing_start, device, profile_timing)
            if not step_applied:
                skipped += len(group_logs)
            else:
                loss_value = float(loss_non_target.detach().cpu().item())
                losses.append(loss_value)
                non_target_losses.append(loss_value)
                edge_optimizer_steps += 1
                non_target_optimizer_steps += 1
                for log in group_logs:
                    _accumulate_edge_step_pi_log(sums, maxes, log)
                    component_log_count += 1
                    if log_details:
                        debug_rows.append(
                            dict(
                                log,
                                update_role="non_target_macro",
                                macro_loss=loss_value,
                                grad_norm=grad_norm_last,
                                updated_or_skipped="updated",
                            )
                        )
        elif log_details:
            debug_rows.append(
                {
                    "edge_id": -1,
                    "edge_group": "non_target_macro",
                    "edge_group_size": 0,
                    "update_role": "non_target_macro",
                    "updated_or_skipped": "skipped",
                    "skip_reason": "no_valid_non_target_edge_group",
                }
            )

    for group_label, edge_indices in target_groups:
        if not _optimizer_budget_available():
            break
        edge_id = int(edge_indices[0])
        optimizer.zero_grad(set_to_none=True)
        timing_start = _profile_now(device, profile_timing)
        with torch.amp.autocast(device_type=amp_device, enabled=bool(use_amp)):
            outputs = model(model_batch_data, task_inputs=task_inputs)
            outputs = _ensure_pi_output_keys(
                outputs,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                edge_target_columns=edge_target_columns,
                normalizer=normalizer,
            )
            last_outputs = outputs
            forward_count += 1
            if static_pi_targets is None:
                static_pi_targets = get_pi_targets(
                    targets=targets, target_masks=target_masks, outputs=outputs,
                    train_cfg=train_cfg, data_cfg=data_cfg,
                    edge_target_columns=edge_target_columns,
                )
                if edge_loss_needs_physical:
                    static_phys_targets = build_pi_physical_targets(
                        static_pi_targets, train_cfg=train_cfg, data_cfg=data_cfg,
                        normalizer=normalizer,
                    )
            pi_targets_cache = static_pi_targets
            phys_outputs_cache = (
                build_pi_physical_outputs(
                    outputs, train_cfg=train_cfg, data_cfg=data_cfg,
                    normalizer=normalizer,
                )
                if edge_loss_needs_physical else None
            )
            phys_targets_cache = static_phys_targets
            loss_e, log = compute_single_edge_pinn_loss(
                outputs=outputs,
                targets=targets,
                target_masks=target_masks,
                edge_id=edge_id,
                edge_indices=edge_indices,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                edge_target_columns=edge_target_columns,
                normalizer=normalizer,
                criterion_main=criterion_main,
                edge_weight_vector=edge_weight_vector if use_existing_target_weight else neutral_edge_weight_vector,
                global_step=global_step,
                total_train_steps=total_train_steps,
                debug=debug,
                collect_log=collect_update_diagnostics,
                pi_targets_cache=pi_targets_cache,
                phys_outputs_cache=phys_outputs_cache,
                phys_targets_cache=phys_targets_cache,
            )
        _profile_add(timings, "profile_target_forward_loss_sec", timing_start, device, profile_timing)
        if loss_e is None:
            skipped += 1
            if log_details:
                debug_rows.append(
                    dict(
                        log,
                        edge_group=group_label,
                        edge_group_size=len(edge_indices),
                        update_role="target_edge",
                        target_weight_applied=use_existing_target_weight,
                        updated_or_skipped="skipped",
                    )
                )
            continue
        timing_start = _profile_now(device, profile_timing)
        grad_norm_last, step_applied = _backward_clip_optimizer_step(
            loss=loss_e,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=grad_clip,
            train_cfg=train_cfg,
            gradient_diagnostics=gradient_diagnostics if log_module_gradient_norms else None,
            gradient_update_role="target",
        )
        _profile_add(timings, "profile_target_backward_step_sec", timing_start, device, profile_timing)
        if not step_applied:
            skipped += 1
            continue
        lv = float(loss_e.detach().cpu().item())
        losses.append(lv)
        target_losses.append(lv)
        edge_optimizer_steps += 1
        target_optimizer_steps += 1
        log = dict(log)
        log["is_target_edge"] = True
        log["target_weight_applied"] = bool(use_existing_target_weight)
        _accumulate_edge_step_pi_log(sums, maxes, log)
        component_log_count += 1
        if log_details:
            debug_rows.append(
                dict(
                    log,
                    edge_group=group_label,
                    edge_group_size=len(edge_indices),
                    update_role="target_edge",
                    grad_norm=grad_norm_last,
                    updated_or_skipped="updated",
                )
            )

    node_cfg = resolve_node_balance_config(train_cfg)
    node_optimization_cfg = _node_pinn_optimization_cfg(train_cfg)
    node_update_mode = str(_cfg_get(node_optimization_cfg, "update_mode", "separate")).strip().lower()
    anchor_weight = float(_cfg_get(node_optimization_cfg, "supervised_anchor_weight", 1.0))
    node_outer_weight = float(_cfg_get(node_optimization_cfg, "node_outer_weight", 1.0))
    anchor_apply_target_weight = bool(_cfg_get(node_optimization_cfg, "anchor_apply_target_weight", False))
    if node_update_mode not in {"separate", "joint_with_supervised_anchor"}:
        raise ValueError(f"Unsupported node_pinn_optimization.update_mode={node_update_mode!r}.")
    anchor_raw_loss = torch.zeros((), dtype=torch.float32, device=device)
    anchor_weighted_loss = anchor_raw_loss
    node_raw_loss = torch.zeros((), dtype=torch.float32, device=device)
    node_outer_weighted_loss = node_raw_loss
    joint_loss_value = torch.zeros((), dtype=torch.float32, device=device)
    joint_update_count = 0
    joint_skip_count = 0
    joint_skip_reason = "not_joint_mode"
    log_gradient_diagnostics = bool(_cfg_get(node_optimization_cfg, "log_gradient_diagnostics", False))
    node_contributes = any(
        (
            node_cfg["use_node_mass"] and node_cfg["lambda_node_mass"] > 0.0,
            node_cfg["use_node_component"] and node_cfg["lambda_node_component"] > 0.0,
            node_cfg["use_node_atom"] and node_cfg["lambda_node_atom"] > 0.0,
            node_cfg["use_node_energy"] and node_cfg["lambda_node_energy"] > 0.0,
        )
    )
    if node_contributes and _optimizer_budget_available():
        optimizer.zero_grad(set_to_none=True)
        timing_start = _profile_now(device, profile_timing)
        with torch.amp.autocast(device_type=amp_device, enabled=bool(use_amp)):
            node_outputs = model(model_batch_data, task_inputs=task_inputs)
            node_outputs = _ensure_pi_output_keys(
                node_outputs,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                edge_target_columns=edge_target_columns,
                normalizer=normalizer,
            )
            node_result = compute_pi_node_balance_losses(
                outputs=node_outputs,
                batch_data=batch_data,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                normalizer=normalizer,
                edge_target_columns=edge_target_columns,
            )
            node_weighted_loss = node_result["weighted_node_total"]
            if node_update_mode == "joint_with_supervised_anchor":
                if static_pi_targets is None:
                    static_pi_targets = get_pi_targets(
                        targets=targets, target_masks=target_masks,
                        outputs=node_outputs, train_cfg=train_cfg,
                        data_cfg=data_cfg, edge_target_columns=edge_target_columns,
                    )
                    if edge_loss_needs_physical:
                        static_phys_targets = build_pi_physical_targets(
                            static_pi_targets, train_cfg=train_cfg,
                            data_cfg=data_cfg, normalizer=normalizer,
                        )
                pi_targets_cache = static_pi_targets
                phys_outputs_cache = (
                    build_pi_physical_outputs(
                        node_outputs, train_cfg=train_cfg, data_cfg=data_cfg,
                        normalizer=normalizer,
                    )
                    if edge_loss_needs_physical else None
                )
                phys_targets_cache = static_phys_targets
                anchor_group_losses: list[torch.Tensor] = []
                anchor_edge_weights = edge_weight_vector if anchor_apply_target_weight else neutral_edge_weight_vector
                for _group_label, edge_indices in edge_groups:
                    anchor_edge_loss, _ = compute_single_edge_pinn_loss(
                        outputs=node_outputs,
                        targets=targets,
                        target_masks=target_masks,
                        edge_id=int(edge_indices[0]),
                        edge_indices=edge_indices,
                        train_cfg=train_cfg,
                        data_cfg=data_cfg,
                        edge_target_columns=edge_target_columns,
                        normalizer=normalizer,
                        criterion_main=criterion_main,
                        edge_weight_vector=anchor_edge_weights,
                        global_step=global_step,
                        total_train_steps=total_train_steps,
                        debug=debug,
                        collect_log=False,
                        pi_targets_cache=pi_targets_cache,
                        phys_outputs_cache=phys_outputs_cache,
                        phys_targets_cache=phys_targets_cache,
                    )
                    if anchor_edge_loss is not None:
                        anchor_group_losses.append(anchor_edge_loss)
                if anchor_group_losses:
                    anchor_raw_loss = torch.stack(anchor_group_losses).mean()
                    anchor_weighted_loss = anchor_weight * anchor_raw_loss
                    node_raw_loss = (
                        node_weighted_loss / float(pinn_schedule_multiplier)
                        if float(pinn_schedule_multiplier) > 0.0
                        else torch.zeros_like(node_weighted_loss)
                    )
                    node_outer_weighted_loss = node_outer_weight * node_weighted_loss
                    joint_loss_value = compose_joint_node_pinn_loss(
                        anchor_raw_loss,
                        node_weighted_loss,
                        supervised_anchor_weight=anchor_weight,
                        node_outer_weight=node_outer_weight,
                    )
        _profile_add(timings, "profile_node_forward_loss_sec", timing_start, device, profile_timing)
        forward_count += 1
        node_diag = node_result["diagnostics"]
        node_has_valid = any(
            (
                node_diag["node_mass_contributes_to_total"] and node_diag["node_mass_valid_count"] > 0,
                node_diag["node_component_contributes_to_total"] and node_diag["node_component_valid_count"] > 0,
                node_diag["node_atom_contributes_to_total"] and node_diag["node_atom_valid_count"] > 0,
                node_diag["node_energy_contributes_to_total"] and node_diag["node_energy_valid_count"] > 0,
            )
        )
        if node_update_mode == "joint_with_supervised_anchor":
            anchor_valid = bool(anchor_group_losses) and bool(torch.isfinite(anchor_raw_loss))
            node_valid = node_has_valid and bool(torch.isfinite(node_weighted_loss))
            joint_valid = anchor_valid and node_valid and bool(torch.isfinite(joint_loss_value))
            if joint_valid:
                if log_gradient_diagnostics:
                    for prefix, diagnostic_loss in (
                        ("anchor", anchor_weighted_loss),
                        ("node", node_outer_weighted_loss),
                        ("joint", joint_loss_value),
                    ):
                        for group, value in _autograd_group_norms(model, diagnostic_loss).items():
                            gradient_diagnostics[f"node_pinn_grad_{prefix}_{group}"] = value
                timing_start = _profile_now(device, profile_timing)
                grad_norm_last, step_applied = _backward_clip_optimizer_step(
                    loss=joint_loss_value,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    use_amp=use_amp,
                    grad_clip=grad_clip,
                    train_cfg=train_cfg,
                    gradient_diagnostics=gradient_diagnostics if log_module_gradient_norms else None,
                    gradient_update_role="node_joint",
                )
                _profile_add(timings, "profile_node_backward_step_sec", timing_start, device, profile_timing)
                node_optimizer_steps = int(step_applied)
                joint_update_count = int(step_applied)
                joint_skip_count = int(not step_applied)
                joint_skip_reason = "none" if step_applied else "gradient_step_rejected"
            else:
                node_optimizer_steps = 0
                joint_skip_count = 1
                if not anchor_valid:
                    joint_skip_reason = "no_valid_supervised_anchor"
                elif not node_valid:
                    joint_skip_reason = "invalid_node_pinn"
                else:
                    joint_skip_reason = "nonfinite_joint_loss"
        elif node_has_valid and bool(torch.isfinite(node_weighted_loss)):
            timing_start = _profile_now(device, profile_timing)
            grad_norm_last, step_applied = _backward_clip_optimizer_step(
                loss=node_weighted_loss,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                use_amp=use_amp,
                grad_clip=grad_clip,
                train_cfg=train_cfg,
                gradient_diagnostics=gradient_diagnostics if log_module_gradient_norms else None,
                gradient_update_role="node",
            )
            _profile_add(timings, "profile_node_backward_step_sec", timing_start, device, profile_timing)
            node_optimizer_steps = int(step_applied)
        else:
            node_optimizer_steps = 0
    else:
        if last_outputs is None:
            with torch.no_grad(), torch.amp.autocast(device_type=amp_device, enabled=bool(use_amp)):
                last_outputs = model(model_batch_data, task_inputs=task_inputs)
                last_outputs = _ensure_pi_output_keys(
                    last_outputs,
                    train_cfg=train_cfg,
                    data_cfg=data_cfg,
                    edge_target_columns=edge_target_columns,
                    normalizer=normalizer,
                )
            forward_count += 1
        with torch.no_grad():
            node_result = compute_pi_node_balance_losses(
                outputs=last_outputs,
                batch_data=batch_data,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                normalizer=normalizer,
                edge_target_columns=edge_target_columns,
            )
        node_weighted_loss = node_result["weighted_node_total"]
        node_diag = node_result["diagnostics"]
        node_optimizer_steps = 0

    mass_pred_median = mass_true_median = mass_ratio = 0.0
    h_pred_median = h_true_median = h_ratio = 0.0
    if bool(_cfg_get(train_cfg, "compute_collapse_diagnostics", True)):
        diagnostic_outputs = node_outputs if node_outputs is not None else last_outputs
        collapse_values = _collapse_scale_values(
            outputs=diagnostic_outputs,
            targets=targets,
            target_masks=target_masks,
            train_cfg=train_cfg,
            data_cfg=data_cfg,
            edge_target_columns=edge_target_columns,
            normalizer=normalizer,
        )
        mass_pred_median, mass_true_median, mass_ratio = _median_ratio(
            collapse_values["mass_flow_pred_abs"],
            collapse_values["mass_flow_true_abs"],
        )
        h_pred_median, h_true_median, h_ratio = _median_ratio(
            collapse_values["enthalpy_pred_abs"],
            collapse_values["enthalpy_true_abs"],
        )

    denom = max(component_log_count, 1)
    mean_loss = sum(losses) / len(losses) if losses else 0.0
    node_weighted_value = float(node_weighted_loss.detach().cpu())
    total_after_node = mean_loss + (
        float(joint_loss_value.detach().cpu())
        if node_update_mode == "joint_with_supervised_anchor" and joint_update_count
        else node_weighted_value
    )
    zero = torch.zeros((), dtype=torch.float32, device=device)
    items: MutableMapping[str, torch.Tensor] = {
        "loss_total": _scalar(total_after_node, device),
        "loss_total_final": _scalar(total_after_node, device),
        "loss_edge_all": _scalar(mean_loss, device),
        "loss_edge_supervised": _scalar(mean_loss, device),
        "edge_step_pi_loss_mean": _scalar(mean_loss, device),
        "sample_hybrid_target_edge_step_pi_loss_mean": _scalar(mean_loss, device),
        "total_loss_before_node_pinn": _scalar(mean_loss, device),
        "total_loss_after_node_pinn": _scalar(total_after_node, device),
        "target_edge_loss_mean": _scalar(sum(target_losses) / len(target_losses) if target_losses else 0.0, device),
        "non_target_edge_loss_mean": _scalar(sum(non_target_losses) / len(non_target_losses) if non_target_losses else 0.0, device),
        "edge_update_count": _scalar(edge_optimizer_steps, device),
        "optimizer_step_count": _scalar(edge_optimizer_steps + node_optimizer_steps, device),
        "non_target_optimizer_step_count": _scalar(non_target_optimizer_steps, device),
        "target_optimizer_step_count": _scalar(target_optimizer_steps, device),
        "node_optimizer_step_count": _scalar(node_optimizer_steps, device),
        "optimizer_step_budget": _scalar(
            -1 if optimizer_step_budget is None else int(optimizer_step_budget), device
        ),
        "optimizer_step_budget_exhausted": _scalar(
            0.0 if optimizer_step_budget is None else float(not _optimizer_budget_available()), device
        ),
        "joint_node_optimizer_step_count": _scalar(joint_update_count, device),
        "joint_node_skip_count": _scalar(joint_skip_count, device),
        "node_pinn_supervised_anchor_raw": anchor_raw_loss.detach(),
        "node_pinn_supervised_anchor_weighted": anchor_weighted_loss.detach(),
        "node_pinn_raw": node_raw_loss.detach(),
        "node_pinn_outer_weighted": node_outer_weighted_loss.detach(),
        "node_pinn_joint_loss": joint_loss_value.detach(),
        "node_pinn_schedule_multiplier": _scalar(pinn_schedule_multiplier, device),
        "node_pinn_outer_weight": _scalar(node_outer_weight, device),
        "node_pinn_anchor_weight": _scalar(anchor_weight, device),
        "node_pinn_joint_mode": _scalar(1.0 if node_update_mode == "joint_with_supervised_anchor" else 0.0, device),
        "collapse_mass_flow_pred_abs_median": _scalar(mass_pred_median, device),
        "collapse_mass_flow_true_abs_median": _scalar(mass_true_median, device),
        "collapse_mass_flow_pred_true_ratio": _scalar(mass_ratio, device),
        "collapse_enthalpy_pred_abs_median": _scalar(h_pred_median, device),
        "collapse_enthalpy_true_abs_median": _scalar(h_true_median, device),
        "collapse_enthalpy_pred_true_ratio": _scalar(h_ratio, device),
        "skipped_edge_count": _scalar(skipped, device),
        "num_edges_per_batch": _scalar(len(edge_groups), device),
        "num_edge_rows_per_batch": _scalar(n_edges, device),
        "data_iteration_count": _scalar(1, device),
        "graph_sample_count": _scalar(1, device),
        "edge_step_pi_forward_count": _scalar(forward_count, device),
        "grad_norm": _scalar(grad_norm_last, device),
        "edge_step_target_edge_count": _scalar(len(target_groups), device),
        "edge_step_default_edge_count": _scalar(len(non_target_groups), device),
        "sample_hybrid_target_group_count": _scalar(len(target_groups), device),
        "sample_hybrid_non_target_group_count": _scalar(len(non_target_groups), device),
        "target_edge_weight_mean": _scalar(weight_info.get("target_edge_weight_mean", 0.0), device),
        "default_edge_weight_mean": _scalar(
            weight_info.get("default_edge_weight_mean", _cfg_get(train_cfg, "edge_weight_default", 1.0)),
            device,
        ),
        "target_edge_existing_weight_applied": _scalar(1.0 if use_existing_target_weight else 0.0, device),
        "loss_all_edge_r2": zero.clone(),
        "weighted_all_edge_r2": zero.clone(),
        "loss_primary_frac": zero.clone(),
        "weighted_primary_frac": zero.clone(),
        "loss_primary_frac_contributes_to_total": zero.clone(),
    }
    for key, val in sums.items():
        items[key] = _scalar(val / denom, device)
    for label in ("p95", "p99"):
        count = float(sums.get(f"mass_physical_{label}_count", 0.0))
        loss_sum = float(sums.get(f"mass_physical_loss_{label}_sum", 0.0))
        items[f"mass_physical_loss_{label}_mean"] = _scalar(
            loss_sum / count if count > 0.0 else 0.0,
            device,
        )
    for key, val in maxes.items():
        items[key] = _scalar(val, device)
    for key, val in node_diag.items():
        if isinstance(val, (bool, int, float)):
            items[key] = _scalar(float(val), device)
    items["loss_rho"] = items.get("loss_rho_mean", zero.clone())
    items["loss_h"] = items.get("loss_h_mean", zero.clone())
    items["loss_volume"] = items.get("loss_volume_mean", zero.clone())
    items["lambda_volume"] = _scalar(float(_cfg_get(train_cfg, "lambda_volume", 0.0)), device)
    items["weighted_node_mass"] = _scalar(float(node_result["weighted_node_mass"].detach().cpu()), device)
    items["weighted_node_component"] = _scalar(float(node_result["weighted_node_component"].detach().cpu()), device)
    items["weighted_node_atom"] = _scalar(float(node_result["weighted_node_atom"].detach().cpu()), device)
    items["weighted_node_energy"] = _scalar(float(node_result["weighted_node_energy"].detach().cpu()), device)
    for key, val in timings.items():
        items[key] = _scalar(val, device)
    gradient_roles = ("non_target", "target", "node_joint", "node")
    for role in gradient_roles:
        prefix = f"grad_{role}"
        count = float(gradient_diagnostics.get(f"{prefix}_update_count", 0.0))
        if count <= 0.0:
            continue
        items[f"{prefix}_clip_ratio"] = _scalar(
            float(gradient_diagnostics.get(f"{prefix}_clip_count", 0.0)) / count,
            device,
        )
        for key, val in gradient_diagnostics.items():
            marker = f"{prefix}_"
            if not key.startswith(marker) or not key.endswith("_sum"):
                continue
            items[key[: -len("_sum")]] = _scalar(float(val) / count, device)
    for key, val in gradient_diagnostics.items():
        if key.endswith("_sum"):
            continue
        items[key] = _scalar(val, device)
    if joint_skip_reason != "none":
        debug_rows.append(
            {
                "update_role": "node_pinn_joint",
                "updated_or_skipped": "skipped" if joint_skip_count else "not_applicable",
                "skip_reason": joint_skip_reason,
            }
        )
    return EdgeStepPIBatchResult(loss_items=items, debug_rows=debug_rows)


def train_one_batch_edge_step_pi(
    *,
    model: torch.nn.Module,
    batch_data: Any,
    task_inputs: Any,
    targets: Mapping[str, torch.Tensor],
    target_masks: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    train_cfg: Any,
    data_cfg: Any,
    device: torch.device,
    use_amp: bool,
    grad_clip: float | None,
    edge_export_meta: Any,
    edge_target_columns: Sequence[str] | None,
    normalizer: Mapping[str, Any] | None = None,
    rng: random.Random | None = None,
    global_step: int | None = None,
    total_train_steps: int | None = None,
    debug: bool = False,
) -> EdgeStepPIBatchResult:
    n_edges = _edge_count_from_target(targets["edge_stream"])
    edge_groups = _edge_groups_from_export_meta(edge_export_meta, n_edges)
    if bool(_cfg_get(train_cfg, "shuffle_edges_each_batch", True)):
        (rng or random).shuffle(edge_groups)
    edge_weight_vector, weight_info = build_edge_step_edge_weight_vector(
        train_cfg=train_cfg,
        edge_export_meta=edge_export_meta,
        edge_target_columns=edge_target_columns,
        n_edges=n_edges,
        device=device,
    )
    target_edge_mask, _target_mask_info = build_target_edge_boolean_mask(
        train_cfg=train_cfg,
        edge_export_meta=edge_export_meta,
        edge_target_columns=edge_target_columns,
        n_edges=n_edges,
        device=device,
    )
    model_batch_data = dict(batch_data)
    model_batch_data["target_edge_mask"] = target_edge_mask
    criterion_main = build_pi_main_criterion(train_cfg)
    losses: list[float] = []
    target_losses: list[float] = []
    non_target_losses: list[float] = []
    debug_rows: list[dict[str, Any]] = []
    sums: dict[str, float] = {}
    maxes: dict[str, float] = {}
    skipped = 0
    updated = 0
    forward_count = 0
    grad_norm_last = 0.0
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    last_outputs: Mapping[str, torch.Tensor] | None = None
    for group_label, edge_indices in edge_groups:
        edge_id = int(edge_indices[0])
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=amp_device, enabled=bool(use_amp)):
            outputs = model(model_batch_data, task_inputs=task_inputs)
            outputs = _ensure_pi_output_keys(
                outputs,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                edge_target_columns=edge_target_columns,
                normalizer=normalizer,
            )
            last_outputs = outputs
            forward_count += 1
            pi_targets_cache = get_pi_targets(
                targets=targets,
                target_masks=target_masks,
                outputs=outputs,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                edge_target_columns=edge_target_columns,
            )
            phys_outputs_cache = build_pi_physical_outputs(
                outputs,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                normalizer=normalizer,
            )
            phys_targets_cache = build_pi_physical_targets(
                pi_targets_cache,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                normalizer=normalizer,
            )
            loss_e, log = compute_single_edge_pinn_loss(
                outputs=outputs,
                targets=targets,
                target_masks=target_masks,
                edge_id=edge_id,
                edge_indices=edge_indices,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                edge_target_columns=edge_target_columns,
                normalizer=normalizer,
                criterion_main=criterion_main,
                edge_weight_vector=edge_weight_vector,
                global_step=global_step,
                total_train_steps=total_train_steps,
                debug=debug,
                pi_targets_cache=pi_targets_cache,
                phys_outputs_cache=phys_outputs_cache,
                phys_targets_cache=phys_targets_cache,
            )
        if loss_e is None:
            skipped += 1
            debug_rows.append(dict(log, edge_group=group_label, edge_group_size=len(edge_indices), updated_or_skipped="skipped"))
            continue
        grad_norm_last, step_applied = _backward_clip_optimizer_step(
            loss=loss_e,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=grad_clip,
            train_cfg=train_cfg,
        )
        if not step_applied:
            skipped += 1
            continue
        updated += 1
        lv = float(loss_e.detach().cpu().item())
        losses.append(lv)
        (target_losses if bool(log.get("is_target_edge")) else non_target_losses).append(lv)
        for key, src in (
            ("loss_main_mean", "loss_main_before_weight"),
            ("mass_log_loss_mean", "mass_log_loss"),
            ("mass_physical_loss_mean", "mass_physical_loss"),
            ("mass_total_loss_mean", "mass_total_loss"),
            ("mass_physical_weight_mean", "mass_physical_weight"),
            ("mass_log_weight_mean", "mass_log_weight"),
            ("mass_pred_physical_mean", "mass_flow_pred_physical_mean"),
            ("mass_true_physical_mean", "mass_flow_true_physical_mean"),
            ("mass_physical_loss_p95_sum", "mass_flow_physical_loss_p95_sum"),
            ("mass_physical_p95_count", "mass_flow_physical_p95_count"),
            ("mass_physical_loss_p99_sum", "mass_flow_physical_loss_p99_sum"),
            ("mass_physical_p99_count", "mass_flow_physical_p99_count"),
            ("loss_fraction_clr_mean", "loss_fraction_clr_before_weight"),
            ("weighted_loss_fraction_clr_mean", "weighted_loss_fraction_clr_before_edge_weight"),
            ("loss_fraction_log_mean", "loss_fraction_log_before_weight"),
            ("weighted_loss_fraction_log_mean", "weighted_loss_fraction_log_before_edge_weight"),
            ("loss_fraction_closure_mean", "loss_fraction_closure_before_weight"),
            (
                "weighted_loss_fraction_closure_mean",
                "weighted_loss_fraction_closure_before_edge_weight",
            ),
            ("loss_frac_penalty_total", "loss_frac_penalty_total"),
            ("loss_base_before_frac_penalty", "loss_base_before_frac_penalty"),
            ("loss_total_after_frac_penalty", "loss_total_after_frac_penalty"),
            ("frac_penalty_ramp_weight", "frac_penalty_ramp_weight"),
            ("frac_penalty_to_base_ratio", "frac_penalty_to_base_ratio"),
            ("loss_co_pos", "loss_co_pos"),
            ("weighted_loss_co_pos", "weighted_loss_co_pos"),
            ("co_pos_gate_mean", "co_pos_gate_mean"),
            ("co_pos_gate_sum", "co_pos_gate_sum"),
            ("co_pos_effective_count", "co_pos_effective_count"),
            ("co_true_positive_count_true_gt_001", "co_true_positive_count_true_gt_001"),
            ("co_pred_mean_on_positive", "co_pred_mean_on_positive"),
            ("co_true_mean_on_positive", "co_true_mean_on_positive"),
            ("loss_ch4_fp", "loss_ch4_fp"),
            ("loss_co2_fp", "loss_co2_fp"),
            ("loss_h2_fp", "loss_h2_fp"),
            ("weighted_loss_ch4_fp", "weighted_loss_ch4_fp"),
            ("weighted_loss_co2_fp", "weighted_loss_co2_fp"),
            ("weighted_loss_h2_fp", "weighted_loss_h2_fp"),
            ("zero_gate_ch4_mean", "zero_gate_ch4_mean"),
            ("zero_gate_co2_mean", "zero_gate_co2_mean"),
            ("zero_gate_h2_mean", "zero_gate_h2_mean"),
            ("fp_ch4_active_count_pred_gt_margin", "fp_ch4_active_count_pred_gt_margin"),
            ("fp_co2_active_count_pred_gt_margin", "fp_co2_active_count_pred_gt_margin"),
            ("fp_h2_active_count_pred_gt_margin", "fp_h2_active_count_pred_gt_margin"),
            ("loss_co2_pos", "loss_co2_pos"),
            ("weighted_loss_co2_pos", "weighted_loss_co2_pos"),
            ("co2_pos_gate_mean", "co2_pos_gate_mean"),
            ("co2_pos_gate_sum", "co2_pos_gate_sum"),
            ("co2_true_high_count_true_gt_005", "co2_true_high_count_true_gt_005"),
            ("co2_true_high_count_true_gt_02", "co2_true_high_count_true_gt_02"),
            ("co2_pred_mean_on_high", "co2_pred_mean_on_high"),
            ("loss_rho_mean", "loss_rho_before_weight"),
            ("loss_h_mean", "loss_h_before_weight"),
            ("loss_volume_mean", "loss_volume_before_weight"),
            ("loss_enthalpy_flow_mean", "loss_enthalpy_flow_before_weight"),
            ("loss_atom_mean", "loss_atom_before_weight"),
            ("loss_energy_mean", "loss_energy_before_weight"),
            ("edge_weight_mean", "edge_weight_mean"),
            ("relu_l1_all_zero_row_count", "relu_l1_all_zero_row_count"),
            ("relu_l1_all_zero_row_ratio", "relu_l1_all_zero_row_ratio"),
            ("relu_l1_denominator_mean", "relu_l1_denominator_mean"),
            ("pred_frac_sum_mean", "pred_frac_sum_mean"),
            ("pred_frac_negative_count", "pred_frac_negative_count"),
            ("pred_frac_nan_count", "pred_frac_nan_count"),
            ("pred_frac_inf_count", "pred_frac_inf_count"),
        ):
            sums[key] = sums.get(key, 0.0) + float(log.get(src, 0.0))
        for key, src in (
            ("loss_rho_max", "loss_rho_max"),
            ("loss_h_max", "loss_h_max"),
            ("loss_volume_max", "loss_volume_max"),
            ("loss_enthalpy_flow_max", "loss_enthalpy_flow_max"),
            ("edge_weight_min", "edge_weight_min"),
            ("edge_weight_max", "edge_weight_max"),
            ("relu_l1_denominator_min", "relu_l1_denominator_min"),
            ("pred_frac_sum_min", "pred_frac_sum_min"),
            ("pred_frac_sum_max", "pred_frac_sum_max"),
        ):
            val = float(log.get(src, 0.0))
            if key.endswith("_min"):
                maxes[key] = min(maxes.get(key, val), val)
            else:
                maxes[key] = max(maxes.get(key, val), val)
        debug_rows.append(dict(log, edge_group=group_label, edge_group_size=len(edge_indices), grad_norm=grad_norm_last))

    node_cfg = resolve_node_balance_config(train_cfg)
    node_contributes = any(
        (
            node_cfg["use_node_mass"] and node_cfg["lambda_node_mass"] > 0.0,
            node_cfg["use_node_component"] and node_cfg["lambda_node_component"] > 0.0,
            node_cfg["use_node_atom"] and node_cfg["lambda_node_atom"] > 0.0,
            node_cfg["use_node_energy"] and node_cfg["lambda_node_energy"] > 0.0,
        )
    )
    if node_contributes:
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=amp_device, enabled=bool(use_amp)):
            node_outputs = model(model_batch_data, task_inputs=task_inputs)
            node_outputs = _ensure_pi_output_keys(
                node_outputs,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                edge_target_columns=edge_target_columns,
                normalizer=normalizer,
            )
            node_result = compute_pi_node_balance_losses(
                outputs=node_outputs,
                batch_data=batch_data,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                normalizer=normalizer,
                edge_target_columns=edge_target_columns,
            )
            node_weighted_loss = node_result["weighted_node_total"]
        forward_count += 1
        node_diag = node_result["diagnostics"]
        node_has_valid = any(
            (
                node_diag["node_mass_contributes_to_total"] and node_diag["node_mass_valid_count"] > 0,
                node_diag["node_component_contributes_to_total"] and node_diag["node_component_valid_count"] > 0,
                node_diag["node_atom_contributes_to_total"] and node_diag["node_atom_valid_count"] > 0,
                node_diag["node_energy_contributes_to_total"] and node_diag["node_energy_valid_count"] > 0,
            )
        )
        if node_has_valid and bool(torch.isfinite(node_weighted_loss)):
            grad_norm_last, step_applied = _backward_clip_optimizer_step(
                loss=node_weighted_loss,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                use_amp=use_amp,
                grad_clip=grad_clip,
                train_cfg=train_cfg,
            )
            node_optimizer_steps = int(step_applied)
        else:
            node_optimizer_steps = 0
    else:
        if last_outputs is None:
            with torch.no_grad(), torch.amp.autocast(device_type=amp_device, enabled=bool(use_amp)):
                last_outputs = model(model_batch_data, task_inputs=task_inputs)
                last_outputs = _ensure_pi_output_keys(
                    last_outputs,
                    train_cfg=train_cfg,
                    data_cfg=data_cfg,
                    edge_target_columns=edge_target_columns,
                    normalizer=normalizer,
                )
            forward_count += 1
        with torch.no_grad():
            node_result = compute_pi_node_balance_losses(
                outputs=last_outputs,
                batch_data=batch_data,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                normalizer=normalizer,
                edge_target_columns=edge_target_columns,
            )
        node_weighted_loss = node_result["weighted_node_total"]
        node_diag = node_result["diagnostics"]
        node_optimizer_steps = 0

    denom = max(updated, 1)
    mean_loss = sum(losses) / len(losses) if losses else 0.0
    node_weighted_value = float(node_weighted_loss.detach().cpu())
    total_after_node = mean_loss + node_weighted_value
    zero = torch.zeros((), dtype=torch.float32, device=device)
    items: MutableMapping[str, torch.Tensor] = {
        "loss_total": _scalar(total_after_node, device),
        "loss_total_final": _scalar(total_after_node, device),
        "loss_edge_all": _scalar(mean_loss, device),
        "loss_edge_supervised": _scalar(mean_loss, device),
        "edge_step_pi_loss_mean": _scalar(mean_loss, device),
        "total_loss_before_node_pinn": _scalar(mean_loss, device),
        "total_loss_after_node_pinn": _scalar(total_after_node, device),
        "target_edge_loss_mean": _scalar(sum(target_losses) / len(target_losses) if target_losses else 0.0, device),
        "non_target_edge_loss_mean": _scalar(sum(non_target_losses) / len(non_target_losses) if non_target_losses else 0.0, device),
        "edge_update_count": _scalar(updated, device),
        "optimizer_step_count": _scalar(updated + node_optimizer_steps, device),
        "node_optimizer_step_count": _scalar(node_optimizer_steps, device),
        "skipped_edge_count": _scalar(skipped, device),
        "num_edges_per_batch": _scalar(len(edge_groups), device),
        "num_edge_rows_per_batch": _scalar(n_edges, device),
        "edge_step_pi_forward_count": _scalar(forward_count, device),
        "grad_norm": _scalar(grad_norm_last, device),
        "edge_step_target_edge_count": _scalar(weight_info.get("target_edge_count", 0), device),
        "edge_step_default_edge_count": _scalar(weight_info.get("default_edge_count", 0), device),
        "target_edge_weight_mean": _scalar(weight_info.get("target_edge_weight_mean", 0.0), device),
        "default_edge_weight_mean": _scalar(
            weight_info.get("default_edge_weight_mean", _cfg_get(train_cfg, "edge_weight_default", 1.0)),
            device,
        ),
        "loss_all_edge_r2": zero.clone(),
        "weighted_all_edge_r2": zero.clone(),
        "loss_primary_frac": zero.clone(),
        "weighted_primary_frac": zero.clone(),
        "loss_primary_frac_contributes_to_total": zero.clone(),
    }
    for key, val in sums.items():
        items[key] = _scalar(val / denom, device)
    for label in ("p95", "p99"):
        count = float(sums.get(f"mass_physical_{label}_count", 0.0))
        loss_sum = float(sums.get(f"mass_physical_loss_{label}_sum", 0.0))
        items[f"mass_physical_loss_{label}_mean"] = _scalar(
            loss_sum / count if count > 0.0 else 0.0,
            device,
        )
    for key, val in maxes.items():
        items[key] = _scalar(val, device)
    for key, val in node_diag.items():
        if isinstance(val, (bool, int, float)):
            items[key] = _scalar(float(val), device)
    items["loss_rho"] = items.get("loss_rho_mean", zero.clone())
    items["loss_h"] = items.get("loss_h_mean", zero.clone())
    items["loss_volume"] = items.get("loss_volume_mean", zero.clone())
    items["lambda_volume"] = _scalar(float(_cfg_get(train_cfg, "lambda_volume", 0.0)), device)
    items["weighted_node_mass"] = _scalar(float(node_result["weighted_node_mass"].detach().cpu()), device)
    items["weighted_node_component"] = _scalar(float(node_result["weighted_node_component"].detach().cpu()), device)
    items["weighted_node_atom"] = _scalar(float(node_result["weighted_node_atom"].detach().cpu()), device)
    items["weighted_node_energy"] = _scalar(float(node_result["weighted_node_energy"].detach().cpu()), device)
    return EdgeStepPIBatchResult(loss_items=items, debug_rows=debug_rows)


@torch.no_grad()
def evaluate_pi_epoch(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    train_cfg: Any,
    data_cfg: Any,
    use_amp: bool,
    normalizer: Mapping[str, Any] | None = None,
    target_stream_targets_path: str | None = None,
    target_edge_10d_out_dir: Any | None = None,
) -> dict[str, float]:
    from .target_edge_10d_metrics import (
        TargetEdge10DAccumulator,
        extract_main_stream_metric_tensors,
        extract_pi_property_metric_tensors,
        _normalizer_lists_for_names,
    )

    was_training = model.training
    model.eval()
    total = 0.0
    count = 0
    node_weighted_total = 0.0
    node_batch_count = 0
    node_diagnostic_sums: dict[str, float] = {}
    compute_collapse_diagnostics = bool(_cfg_get(train_cfg, "compute_collapse_diagnostics", True))
    collapse_value_lists: dict[str, list[torch.Tensor]] = {
        "mass_flow_pred_abs": [],
        "mass_flow_true_abs": [],
        "enthalpy_pred_abs": [],
        "enthalpy_true_abs": [],
    }
    edge_component_sums: dict[str, float] = {}
    edge_component_count = 0
    te10_acc = (
        TargetEdge10DAccumulator(train_cfg=train_cfg, target_stream_targets_path=target_stream_targets_path)
        if target_stream_targets_path
        else None
    )
    compute_property_metrics = bool(_cfg_get(train_cfg, "compute_target_edge_property_metrics", False))
    prop_acc = (
        TargetEdge10DAccumulator(train_cfg=train_cfg, target_stream_targets_path=target_stream_targets_path)
        if target_stream_targets_path and compute_property_metrics
        else None
    )
    all_edge_r2_acc = PIAllEdgePropertyR2Accumulator(train_cfg=train_cfg, store_scatter_samples=False)
    validation_started_at = time.perf_counter()
    terminal_progress_enabled = _terminal_log_verbosity(train_cfg) != "quiet"
    try:
        num_val_batches: int | str = len(loader)
    except TypeError:
        num_val_batches = "?"
    val_progress_interval = max(100, int(_cfg_get(train_cfg, "log_interval", 10) or 10) * 10)
    compute_validation_pinn_loss = bool(_cfg_get(train_cfg, "compute_validation_pinn_loss", True))
    for batch_idx, batch in enumerate(loader):
        batch_data = {k: v.to(device) for k, v in batch.model_kwargs.items()}
        targets_raw = {k: v.to(device) for k, v in batch.targets.items()}
        targets = dict(targets_raw)
        target_masks = {k: v.to(device) for k, v in batch.target_masks.items()}
        if "edge_stream" in targets:
            targets = dict(targets)
            targets["edge_stream"] = _normalize_target_if_needed(
                targets["edge_stream"],
                normalizer,
                data_cfg,
                list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
            )
        task_inputs = {h: {k: v.to(device) for k, v in p.items()} for h, p in batch.task_inputs.items()}
        n_edges = _edge_count_from_target(targets["edge_stream"])
        target_edge_mask, _target_mask_info = build_target_edge_boolean_mask(
            train_cfg=train_cfg,
            edge_export_meta=batch.edge_export_meta,
            edge_target_columns=list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
            n_edges=n_edges,
            device=device,
        )
        model_batch_data = dict(batch_data)
        model_batch_data["target_edge_mask"] = target_edge_mask
        amp_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=amp_device, enabled=bool(use_amp)):
            outputs = model(model_batch_data, task_inputs=task_inputs)
            outputs = _ensure_pi_output_keys(
                outputs,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                edge_target_columns=list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
                normalizer=normalizer,
            )
        all_edge_r2_acc.update_batch(
            outputs=outputs,
            targets_raw=targets_raw,
            target_masks=target_masks,
            train_cfg=train_cfg,
            data_cfg=data_cfg,
            edge_target_columns=list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
            normalizer=normalizer,
        )
        if compute_collapse_diagnostics:
            batch_collapse_values = _collapse_scale_values(
                outputs=outputs,
                targets=targets,
                target_masks=target_masks,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                edge_target_columns=list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
                normalizer=normalizer,
            )
            for key, value in batch_collapse_values.items():
                if value.numel() > 0:
                    collapse_value_lists[key].append(value.cpu())
        if te10_acc is not None and batch.edge_export_meta is not None:
            pred_main, true_main, mask_main, prop_names = extract_main_stream_metric_tensors(
                outputs=outputs,
                targets_raw=targets_raw,
                target_masks=target_masks,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                edge_target_columns=list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
                normalizer=normalizer,
            )
            edge_cols = list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
            inv_mean, inv_std = _normalizer_lists_for_names(
                normalizer,
                prop_names,
                data_cfg=data_cfg,
                device=pred_main.device,
                dtype=pred_main.dtype,
            )
            te10_acc.update_batch(
                export=batch.edge_export_meta,
                pred_main=pred_main,
                true_main=true_main,
                mask_main=mask_main,
                property_names=prop_names,
                split_name="val",
                edge_target_columns=edge_cols,
                inverse_mean=inv_mean,
                inverse_std=inv_std,
            )
            if prop_acc is not None:
                pred_prop, true_prop, mask_prop, prop_prop_names = extract_pi_property_metric_tensors(
                    outputs=outputs,
                    targets_raw=targets_raw,
                    target_masks=target_masks,
                    data_cfg=data_cfg,
                    edge_target_columns=list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
                    normalizer=normalizer,
                )
                prop_inv_mean, prop_inv_std = _normalizer_lists_for_names(
                    normalizer,
                    prop_prop_names,
                    data_cfg=data_cfg,
                    device=pred_prop.device,
                    dtype=pred_prop.dtype,
                )
                prop_acc.update_batch(
                    export=batch.edge_export_meta,
                    pred_main=pred_prop,
                    true_main=true_prop,
                    mask_main=mask_prop,
                    property_names=prop_prop_names,
                    split_name="val",
                    edge_target_columns=edge_cols,
                    inverse_mean=prop_inv_mean,
                    inverse_std=prop_inv_std,
                )
        if compute_validation_pinn_loss:
            edge_weight_vector, _ = build_edge_step_edge_weight_vector(
                train_cfg=train_cfg,
                edge_export_meta=batch.edge_export_meta,
                edge_target_columns=list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
                n_edges=n_edges,
                device=device,
            )
            pi_targets_cache = get_pi_targets(
                targets=targets,
                target_masks=target_masks,
                outputs=outputs,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                edge_target_columns=list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
            )
            phys_outputs_cache = build_pi_physical_outputs(
                outputs,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                normalizer=normalizer,
            )
            phys_targets_cache = build_pi_physical_targets(
                pi_targets_cache,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                normalizer=normalizer,
            )
            for edge_id in range(n_edges):
                loss_e, edge_log = compute_single_edge_pinn_loss(
                    outputs=outputs,
                    targets=targets,
                    target_masks=target_masks,
                    edge_id=edge_id,
                    train_cfg=train_cfg,
                    data_cfg=data_cfg,
                    edge_target_columns=list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
                    normalizer=normalizer,
                    edge_weight_vector=edge_weight_vector,
                    pi_targets_cache=pi_targets_cache,
                    phys_outputs_cache=phys_outputs_cache,
                    phys_targets_cache=phys_targets_cache,
                )
                if loss_e is not None:
                    total += float(loss_e.detach().cpu().item())
                    count += 1
                    edge_component_count += 1
                    for out_key, log_key in (
                        ("loss_rho", "loss_rho_before_weight"),
                        ("loss_h", "loss_h_before_weight"),
                        ("loss_volume", "loss_volume_before_weight"),
                    ):
                        edge_component_sums[out_key] = edge_component_sums.get(out_key, 0.0) + float(
                            edge_log.get(log_key, 0.0)
                        )
            node_result = compute_pi_node_balance_losses(
                outputs=outputs,
                batch_data=batch_data,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                normalizer=normalizer,
                edge_target_columns=list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
            )
            node_weighted_total += float(node_result["weighted_node_total"].detach().cpu())
            node_batch_count += 1
            for key, value in node_result["diagnostics"].items():
                if isinstance(value, (bool, int, float)):
                    node_diagnostic_sums[key] = node_diagnostic_sums.get(key, 0.0) + float(value)
        current_batch = int(batch_idx) + 1
        should_log_progress = (
            terminal_progress_enabled
            and (
                current_batch == 1
                or current_batch % val_progress_interval == 0
                or (isinstance(num_val_batches, int) and current_batch >= num_val_batches)
            )
        )
        if should_log_progress:
            progress_bar, progress_pct = _terminal_progress_bar(current_batch, num_val_batches)
            elapsed_text, eta_text = _terminal_eta(validation_started_at, current_batch, num_val_batches)
            print(
                f"[validation batch={current_batch}/{num_val_batches} {progress_pct} {progress_bar} "
                f"elapsed={elapsed_text} eta={eta_text}]",
                flush=True,
            )
    model.train(was_training)
    edge_loss = total / max(count, 1)
    node_loss = node_weighted_total / max(node_batch_count, 1)
    loss = edge_loss + node_loss
    out = {
        "loss_total": loss,
        "loss_total_final": loss,
        "loss_edge": edge_loss,
        "loss_edge_all": edge_loss,
        "loss_edge_supervised": edge_loss,
        "edge_step_pi_loss_mean": edge_loss,
        "total_loss_before_node_pinn": edge_loss,
        "total_loss_after_node_pinn": loss,
        "loss_all_edge_r2": 0.0,
        "weighted_all_edge_r2": 0.0,
        "loss_primary_frac": 0.0,
        "weighted_primary_frac": 0.0,
        "eval_primary_frac_r2_by_process": 0.0,
        "eval_primary_frac_r2_by_target": 0.0,
        "eval_primary_frac_r2_formula_ok_by_process": 0.0,
        "eval_secondary_amount_r2_by_process": 0.0,
        "eval_target_rows_count": 0.0,
        "target_stream_feature_macro_r2": 0.0,
        "amount_metrics_are_primary": 0.0,
        "validation_pinn_loss_computed": 1.0 if compute_validation_pinn_loss else 0.0,
    }
    for name in ("mass_flow", "enthalpy"):
        pred_values = (
            torch.cat(collapse_value_lists[f"{name}_pred_abs"])
            if collapse_value_lists[f"{name}_pred_abs"]
            else torch.empty(0)
        )
        true_values = (
            torch.cat(collapse_value_lists[f"{name}_true_abs"])
            if collapse_value_lists[f"{name}_true_abs"]
            else torch.empty(0)
        )
        pred_median, true_median, ratio = _median_ratio(pred_values, true_values)
        out[f"collapse_{name}_pred_abs_median"] = pred_median
        out[f"collapse_{name}_true_abs_median"] = true_median
        out[f"collapse_{name}_pred_true_ratio"] = ratio
    for key, value in node_diagnostic_sums.items():
        out[key] = value / max(node_batch_count, 1)
    for key, value in edge_component_sums.items():
        out[key] = value / max(edge_component_count, 1)
    if te10_acc is not None:
        try:
            if target_edge_10d_out_dir is not None:
                import pandas as pd

                from .target_edge_10d_metrics import (
                    write_target_edge_10d_metric_artifacts,
                    write_target_edge_property_metric_artifacts,
                )

                te10_metric_rows = pd.DataFrame(te10_acc.rows)
                te10_scalars = write_target_edge_10d_metric_artifacts(
                    out_dir=target_edge_10d_out_dir,
                    metric_rows=te10_metric_rows,
                    train_cfg=train_cfg,
                    target_stream_targets_path=target_stream_targets_path,
                    save_actual_vs_pred_plots=False,
                )
                penalty_cfg = _cfg_mapping(train_cfg, "pi_fraction_component_penalty")
                diagnostics_cfg = _sub_mapping(penalty_cfg, "diagnostics")
                if bool(diagnostics_cfg.get("save_eval_csv", False)):
                    _write_fraction_penalty_diagnostics_csv(Path(target_edge_10d_out_dir), te10_metric_rows)
                if prop_acc is not None:
                    te10_scalars.update(
                        write_target_edge_property_metric_artifacts(
                            out_dir=target_edge_10d_out_dir,
                            metric_rows=pd.DataFrame(prop_acc.rows),
                            train_cfg=train_cfg,
                            target_stream_targets_path=target_stream_targets_path,
                        )
                    )
            else:
                te10_scalars = te10_acc.finalize_scalars()
            out.update(te10_scalars)
        except Exception:
            pass
    if target_edge_10d_out_dir is not None:
        out.update(all_edge_r2_acc.write_artifacts(Path(target_edge_10d_out_dir)))
    return out


@torch.no_grad()
def collect_pi_all_edge_property_metrics_from_loader(
    *,
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    train_cfg: Any,
    data_cfg: Any,
    use_amp: bool,
    split_name: str,
    normalizer: Mapping[str, Any] | None = None,
    store_scatter_samples: bool = False,
    accumulator: PIAllEdgePropertyR2Accumulator | None = None,
    progress_label: str | None = None,
    progress_interval: int = 50,
    target_edge_only: bool = False,
) -> PIAllEdgePropertyR2Accumulator:
    was_training = model.training
    model.eval()
    acc = accumulator or PIAllEdgePropertyR2Accumulator(
        train_cfg=train_cfg,
        store_scatter_samples=store_scatter_samples,
    )
    try:
        total_batches: int | str = int(len(loader))
    except Exception:
        total_batches = "?"
    started_at = time.perf_counter()
    for batch_idx, batch in enumerate(loader):
        batch_data = {k: v.to(device) for k, v in batch.model_kwargs.items()}
        targets_raw = {k: v.to(device) for k, v in batch.targets.items()}
        target_masks = {k: v.to(device) for k, v in batch.target_masks.items()}
        task_inputs = {h: {k: v.to(device) for k, v in p.items()} for h, p in batch.task_inputs.items()}
        n_edges = _edge_count_from_target(targets_raw["edge_stream"])
        target_edge_mask, _target_mask_info = build_target_edge_boolean_mask(
            train_cfg=train_cfg,
            edge_export_meta=batch.edge_export_meta,
            edge_target_columns=list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
            n_edges=n_edges,
            device=device,
        )
        if target_edge_only:
            # The standard target mask carries label availability.  Intersect
            # it with the predictable-edge mask so the accumulator reports
            # only the same target edges used by the paper's target metrics.
            stream_mask = target_masks.get("edge_stream")
            edge_mask = target_edge_mask.to(dtype=targets_raw["edge_stream"].dtype)
            if stream_mask is None:
                target_masks["edge_stream"] = edge_mask
            else:
                # target_masks is commonly [E] and _expand_edge_mask expands
                # it later.  Only append feature dimensions when the original
                # supervision mask already carries them; otherwise [E, 1] ×
                # [E] would incorrectly broadcast to [E, E].
                while edge_mask.ndim < stream_mask.ndim:
                    edge_mask = edge_mask.unsqueeze(-1)
                target_masks["edge_stream"] = stream_mask * edge_mask
        model_batch_data = dict(batch_data)
        model_batch_data["target_edge_mask"] = target_edge_mask
        amp_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=amp_device, enabled=bool(use_amp)):
            outputs = model(model_batch_data, task_inputs=task_inputs)
            outputs = _ensure_pi_output_keys(
                outputs,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                edge_target_columns=list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
                normalizer=normalizer,
            )
        acc.update_batch(
            outputs=outputs,
            targets_raw=targets_raw,
            target_masks=target_masks,
            train_cfg=train_cfg,
            data_cfg=data_cfg,
            edge_target_columns=list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
            normalizer=normalizer,
            split_name=split_name,
        )
        if progress_label:
            current = int(batch_idx) + 1
            interval = max(1, int(progress_interval or 50))
            is_last = isinstance(total_batches, int) and current >= total_batches
            if current == 1 or current % interval == 0 or is_last:
                progress_bar, progress_pct = _terminal_progress_bar(current, total_batches)
                elapsed_text, eta_text = _terminal_eta(started_at, current, total_batches)
                print(
                    f"[final-eval {progress_label} batch={current}/{total_batches} "
                    f"{progress_pct} {progress_bar} elapsed={elapsed_text} eta={eta_text}]",
                    flush=True,
                )
    model.train(was_training)
    return acc






