"""Mass_Flow output-space helpers for the PI grouped-property head."""

from __future__ import annotations

import math
import warnings
from typing import Any, Mapping

import torch

PI_MASS_FLOW_OUTPUT_SPACES: tuple[str, ...] = ("raw_z", "log1p")
MASS_FLOW_TRANSFORMS: tuple[str, ...] = ("log1p", "tempered_log", "scaled_log")
_EXPM1_MAX_LOG_FLOAT32 = math.log(torch.finfo(torch.float32).max) - 2.0
_warned_expm1_clamp = False


def _cfg_get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def resolve_pi_mass_flow_output_space(train_cfg: Any) -> str:
    value = str(_cfg_get(train_cfg, "pi_mass_flow_output_space", "raw_z") or "raw_z").strip().lower()
    if value not in PI_MASS_FLOW_OUTPUT_SPACES:
        raise ValueError(
            f"pi_mass_flow_output_space must be one of {PI_MASS_FLOW_OUTPUT_SPACES}, got {value!r}."
        )
    if value == "log1p" and bool(_cfg_get(train_cfg, "use_log1p_mass_flow_loss", False)):
        raise ValueError(
            "pi_mass_flow_output_space='log1p' directly predicts log1p(Mass_Flow); "
            "use_log1p_mass_flow_loss is the legacy A2 raw-z path and must be false."
        )
    return value


def resolve_mass_flow_transform(train_cfg: Any) -> str:
    value = str(_cfg_get(train_cfg, "mass_flow_transform", "log1p") or "log1p").strip().lower()
    if value not in MASS_FLOW_TRANSFORMS:
        raise ValueError(
            f"mass_flow_transform must be one of {MASS_FLOW_TRANSFORMS}, got {value!r}."
        )
    return value


def resolve_mass_flow_log_tau(train_cfg: Any) -> float:
    tau = float(_cfg_get(train_cfg, "mass_flow_log_tau", 1.0))
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError(f"mass_flow_log_tau must be finite and positive, got {tau}.")
    return tau


def resolve_mass_flow_log_scale(train_cfg: Any) -> float:
    scale = float(_cfg_get(train_cfg, "mass_flow_log_scale", 1.0))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"mass_flow_log_scale must be finite and positive, got {scale}.")
    return scale


def resolve_mass_flow_log_eps(train_cfg: Any) -> float:
    eps = float(_cfg_get(train_cfg, "mass_flow_log_eps", 1.0e-8))
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"mass_flow_log_eps must be finite and positive, got {eps}.")
    return eps


def _mass_flow_effective_tau(train_cfg: Any) -> float:
    if resolve_mass_flow_transform(train_cfg) == "log1p":
        return 1.0
    return resolve_mass_flow_log_tau(train_cfg)


def transform_mass_flow(
    mass_flow: torch.Tensor,
    *,
    train_cfg: Any,
) -> torch.Tensor:
    """Transform physical Mass_Flow into the configured training space."""
    mass = mass_flow.float().clamp_min(0.0)
    transform = resolve_mass_flow_transform(train_cfg)
    if transform == "scaled_log":
        scale = resolve_mass_flow_log_scale(train_cfg)
        eps = resolve_mass_flow_log_eps(train_cfg)
        transformed = scale * torch.log(mass + eps)
    else:
        tau = _mass_flow_effective_tau(train_cfg)
        transformed = torch.log1p(tau * mass) / tau
    if not torch.isfinite(transformed).all():
        raise RuntimeError("Mass_Flow transform produced NaN or Inf.")
    return transformed


def transform_mass_flow_scalar(
    value: float,
    *,
    transform: str = "log1p",
    tau: float = 1.0,
    scale: float = 1.0,
    eps: float = 1.0e-8,
) -> float:
    cfg = {
        "mass_flow_transform": transform,
        "mass_flow_log_tau": tau,
        "mass_flow_log_scale": scale,
        "mass_flow_log_eps": eps,
    }
    mass = max(float(value), 0.0)
    if resolve_mass_flow_transform(cfg) == "scaled_log":
        return resolve_mass_flow_log_scale(cfg) * math.log(mass + resolve_mass_flow_log_eps(cfg))
    effective_tau = _mass_flow_effective_tau(cfg)
    return math.log1p(effective_tau * mass) / effective_tau


def inverse_transform_mass_flow(
    transformed: torch.Tensor,
    *,
    train_cfg: Any,
) -> torch.Tensor:
    """Decode configured Mass_Flow training space into physical Mass_Flow."""
    value = transformed.float()
    if not torch.isfinite(value).all():
        raise RuntimeError("PI direct Mass_Flow prediction contains NaN or Inf before inverse transform.")
    transform = resolve_mass_flow_transform(train_cfg)
    if transform == "scaled_log":
        scale = resolve_mass_flow_log_scale(train_cfg)
        eps = resolve_mass_flow_log_eps(train_cfg)
        exponent = value / scale
        divisor = 1.0
        offset = eps
    else:
        tau = _mass_flow_effective_tau(train_cfg)
        exponent = tau * value
        divisor = tau
        offset = 0.0
    global _warned_expm1_clamp
    if bool((exponent > _EXPM1_MAX_LOG_FLOAT32).any()):
        if not _warned_expm1_clamp:
            warnings.warn(
                "PI direct Mass_Flow prediction exceeded the safe float32 inverse-transform "
                "range; clamping only for physical metric/export decoding.",
                RuntimeWarning,
                stacklevel=2,
            )
            _warned_expm1_clamp = True
        exponent = exponent.clamp_max(_EXPM1_MAX_LOG_FLOAT32)
    if transform == "scaled_log":
        physical = (torch.exp(exponent) / divisor - offset).clamp_min(0.0)
    else:
        physical = (torch.expm1(exponent) / divisor).clamp_min(0.0)
    if not torch.isfinite(physical).all():
        raise RuntimeError("PI direct Mass_Flow physical decoding produced NaN or Inf.")
    return physical


def decode_pi_mass_flow_prediction(
    prediction: torch.Tensor,
    *,
    train_cfg: Any,
    data_cfg: Any,
    normalizer: Mapping[str, Any] | None,
) -> torch.Tensor:
    """Decode the PI Mass_Flow head into physical Mass_Flow."""
    output_space = resolve_pi_mass_flow_output_space(train_cfg)
    if output_space == "raw_z":
        normalize_y_edge = bool(_cfg_get(data_cfg, "normalize_y_edge", False))
        if not normalize_y_edge:
            return prediction
        if not normalizer:
            raise RuntimeError("normalize_y_edge=true but the PI Mass_Flow normalizer is missing.")
        columns = [str(c) for c in normalizer.get("columns", [])]
        if "Mass_Flow" not in columns or normalizer.get("mean") is None or normalizer.get("std") is None:
            raise RuntimeError("Cannot inverse-transform raw-z PI Mass_Flow; normalizer statistics are missing.")
        idx = columns.index("Mass_Flow")
        mean = torch.as_tensor(normalizer["mean"])[idx].to(device=prediction.device, dtype=prediction.dtype)
        std = torch.as_tensor(normalizer["std"])[idx].to(device=prediction.device, dtype=prediction.dtype)
        return prediction * std + mean

    return inverse_transform_mass_flow(prediction, train_cfg=train_cfg)


def checkpoint_pi_mass_flow_output_space(payload: Mapping[str, Any]) -> str:
    explicit = payload.get("pi_mass_flow_output_space")
    if explicit is not None:
        return str(explicit).strip().lower()
    config = payload.get("config")
    if isinstance(config, Mapping):
        train = config.get("train")
        if isinstance(train, Mapping) and train.get("pi_mass_flow_output_space") is not None:
            return str(train["pi_mass_flow_output_space"]).strip().lower()
    # Checkpoints written before A4 used raw-z Mass_Flow semantics.
    return "raw_z"


def checkpoint_mass_flow_transform(payload: Mapping[str, Any]) -> tuple[str, float, float, float]:
    config = payload.get("config")
    train = config.get("train") if isinstance(config, Mapping) else None
    explicit_transform = payload.get("mass_flow_transform")
    if explicit_transform is None and isinstance(train, Mapping):
        explicit_transform = train.get("mass_flow_transform")
    transform = str(explicit_transform or "log1p").strip().lower()
    explicit_tau = payload.get("mass_flow_log_tau")
    if explicit_tau is None and isinstance(train, Mapping):
        explicit_tau = train.get("mass_flow_log_tau")
    tau = float(1.0 if explicit_tau is None else explicit_tau)
    explicit_scale = payload.get("mass_flow_log_scale")
    if explicit_scale is None and isinstance(train, Mapping):
        explicit_scale = train.get("mass_flow_log_scale")
    scale = float(1.0 if explicit_scale is None else explicit_scale)
    explicit_eps = payload.get("mass_flow_log_eps")
    if explicit_eps is None and isinstance(train, Mapping):
        explicit_eps = train.get("mass_flow_log_eps")
    eps = float(1.0e-8 if explicit_eps is None else explicit_eps)
    return transform, tau, scale, eps


def assert_checkpoint_pi_mass_flow_compatible(
    payload: Mapping[str, Any],
    *,
    train_cfg: Any,
    checkpoint_path: Any = "",
) -> None:
    current = resolve_pi_mass_flow_output_space(train_cfg)
    stored = checkpoint_pi_mass_flow_output_space(payload)
    if stored not in PI_MASS_FLOW_OUTPUT_SPACES:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path!s} has invalid pi_mass_flow_output_space={stored!r}."
        )
    if stored != current:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path!s} uses pi_mass_flow_output_space={stored!r}, "
            f"but the current run uses {current!r}. Refusing to reinterpret the Mass_Flow head."
        )
    if current == "log1p":
        current_transform = resolve_mass_flow_transform(train_cfg)
        current_tau = resolve_mass_flow_log_tau(train_cfg)
        current_scale = resolve_mass_flow_log_scale(train_cfg)
        current_eps = resolve_mass_flow_log_eps(train_cfg)
        stored_transform, stored_tau, stored_scale, stored_eps = checkpoint_mass_flow_transform(payload)
        if stored_transform not in MASS_FLOW_TRANSFORMS:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path!s} has invalid mass_flow_transform="
                f"{stored_transform!r}."
            )
        tau_mismatch = (
            current_transform == "tempered_log"
            and not math.isclose(stored_tau, current_tau, rel_tol=0.0, abs_tol=1.0e-12)
        )
        scale_mismatch = (
            current_transform == "scaled_log"
            and not math.isclose(stored_scale, current_scale, rel_tol=0.0, abs_tol=1.0e-12)
        )
        eps_mismatch = (
            current_transform == "scaled_log"
            and not math.isclose(stored_eps, current_eps, rel_tol=0.0, abs_tol=1.0e-20)
        )
        if stored_transform != current_transform or tau_mismatch or scale_mismatch or eps_mismatch:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path!s} uses Mass_Flow transform "
                f"{stored_transform!r} with tau={stored_tau:g}, scale={stored_scale:g}, eps={stored_eps:g}, "
                f"but the current run uses {current_transform!r} with tau={current_tau:g}, "
                f"scale={current_scale:g}, eps={current_eps:g}. Refusing to reinterpret "
                "the Mass_Flow head."
            )
