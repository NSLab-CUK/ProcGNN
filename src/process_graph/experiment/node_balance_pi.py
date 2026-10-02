"""Node-level conservation diagnostics and optional losses for the PI edge head."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from ..constants import UNIT_VOCAB

PI_SPECIES_ORDER: tuple[str, ...] = ("H2O", "H2", "CH4", "CO2", "CO", "O2", "N2")
PI_ATOM_ORDER: tuple[str, ...] = ("C", "H", "O", "N")
PI_ATOM_COUNTS: dict[str, tuple[float, float, float, float]] = {
    "H2O": (0.0, 2.0, 1.0, 0.0),
    "H2": (0.0, 2.0, 0.0, 0.0),
    "CH4": (1.0, 4.0, 0.0, 0.0),
    "CO2": (1.0, 0.0, 2.0, 0.0),
    "CO": (1.0, 0.0, 1.0, 0.0),
    "O2": (0.0, 0.0, 2.0, 0.0),
    "N2": (0.0, 0.0, 0.0, 2.0),
}

NONREACTIVE_NODE_TYPES: tuple[str, ...] = (
    "mixer",
    "splitter",
    "psa",
    "flash",
    "hx_dt",
    "hx_hot",
    "hx_cold",
    "heater",
    "cooler",
    "compressor",
    "pump",
    "turbine",
)
REACTIVE_NODE_TYPES: tuple[str, ...] = ("smr_reactor", "wgs_reactor", "burner")
HX_NODE_TYPES: tuple[str, ...] = ("hx_dt", "hx_hot", "hx_cold")
BOUNDARY_NODE_TYPES: tuple[str, ...] = ("input_virtual", "output_virtual")


def _cfg_get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _cfg_mapping(obj: Any, name: str) -> Mapping[str, Any]:
    value = _cfg_get(obj, name, {}) or {}
    return value if isinstance(value, Mapping) else {}


def _block(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = cfg.get(name, {}) or {}
    return value if isinstance(value, Mapping) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"Expected a finite numeric value, got {value!r}.")
    return out


def resolve_node_balance_config(train_cfg: Any) -> dict[str, Any]:
    nested = _cfg_mapping(train_cfg, "node_balance_pi")
    nested_enabled = bool(nested.get("enabled", True))
    zero = _block(nested, "zero_flow")
    mass = _block(nested, "mass")
    component = _block(nested, "component")
    atom = _block(nested, "atom")
    energy = _block(nested, "energy")
    mass_prediction_filter = _block(mass, "prediction_filter")
    energy_prediction_filter = _block(energy, "prediction_filter")
    energy_consistency = _block(energy, "consistency_filter")
    diagnostics = _block(nested, "diagnostics")
    energy_has_nested_block = "energy" in nested

    def enabled(block: Mapping[str, Any], flat_key: str, default: bool) -> bool:
        return nested_enabled and bool(block.get("enabled", _cfg_get(train_cfg, flat_key, default)))

    def weight(block: Mapping[str, Any], flat_key: str, default: float) -> float:
        value = block.get("weight", _cfg_get(train_cfg, flat_key, default))
        out = float(value or 0.0)
        if not math.isfinite(out) or out < 0.0:
            raise ValueError(f"{flat_key}/node_balance_pi weight must be finite and non-negative, got {value}.")
        return out

    eps_default = float(_cfg_get(train_cfg, "eps", 1.0e-8))
    zero_flow_threshold = float(
        zero.get(
            "threshold",
            _cfg_get(train_cfg, "zero_flow_threshold", _cfg_get(train_cfg, "zero_flow_fraction_mask_eps", 1.0e-8)),
        )
    )
    if not math.isfinite(zero_flow_threshold) or zero_flow_threshold < 0.0:
        raise ValueError(f"zero_flow_threshold must be finite and non-negative, got {zero_flow_threshold}.")

    cfg = {
        "compute_diagnostics": bool(diagnostics.get("enabled", _cfg_get(train_cfg, "compute_node_balance_diagnostics", True))),
        "zero_flow_threshold": zero_flow_threshold,
        "zero_flow_nan_enthalpy_as_zero": bool(zero.get("zero_flow_nan_enthalpy_as_zero", True)),
        "use_node_mass": enabled(mass, "use_node_mass_balance_loss", False),
        "lambda_node_mass": weight(mass, "lambda_node_mass", 0.0),
        "node_mass_relative": bool(mass.get("relative", _cfg_get(train_cfg, "node_mass_balance_relative", True))),
        "node_mass_epsilon": float(mass.get("epsilon", eps_default)),
        "node_mass_scale_floor": _float_or_none(mass.get("scale_floor")),
        "node_mass_loss_type": str(mass.get("loss_type", "squared")).strip().lower(),
        "node_mass_huber_delta": float(mass.get("huber_delta", 1.0e-3)),
        "node_mass_residual_abs_clip": _float_or_none(mass.get("residual_abs_clip")),
        "node_mass_pred_residual_abs_max": _float_or_none(
            mass_prediction_filter.get("max_abs_residual", mass.get("max_pred_residual_abs"))
        ),
        "use_node_component": enabled(component, "use_node_component_balance_loss", False),
        "lambda_node_component": weight(component, "lambda_node_component", 0.0),
        "node_component_relative": bool(component.get("relative", True)),
        "node_component_epsilon": float(component.get("epsilon", eps_default)),
        "node_component_scale_floor": _float_or_none(component.get("scale_floor")),
        "node_component_flow_threshold": float(component.get("component_flow_threshold", 0.0) or 0.0),
        "node_component_loss_type": str(component.get("loss_type", "squared")).strip().lower(),
        "node_component_huber_delta": float(component.get("huber_delta", 1.0e-3)),
        "node_component_residual_abs_clip": _float_or_none(component.get("residual_abs_clip")),
        "use_node_atom": enabled(atom, "use_node_atom_balance_loss", _cfg_get(train_cfg, "use_atom_balance_loss", False)),
        "lambda_node_atom": weight(atom, "lambda_node_atom", _cfg_get(train_cfg, "lambda_atom", 0.0)),
        "node_atom_relative": bool(atom.get("relative", _cfg_get(train_cfg, "node_atom_balance_relative", True))),
        "node_atom_epsilon": float(atom.get("epsilon", eps_default)),
        "node_atom_scale_floor": _float_or_none(atom.get("scale_floor")),
        "node_atom_flow_threshold": float(atom.get("atom_flow_threshold", 0.0) or 0.0),
        "node_atom_loss_type": str(atom.get("loss_type", "squared")).strip().lower(),
        "node_atom_huber_delta": float(atom.get("huber_delta", 1.0e-3)),
        "node_atom_residual_abs_clip": _float_or_none(atom.get("residual_abs_clip")),
        "use_node_energy": enabled(energy, "use_node_energy_balance_loss", _cfg_get(train_cfg, "use_energy_balance_loss", False)),
        "lambda_node_energy": weight(energy, "lambda_node_energy", _cfg_get(train_cfg, "lambda_energy", 0.0)),
        "node_energy_relative": bool(energy.get("relative", _cfg_get(train_cfg, "node_energy_balance_relative", True))),
        "node_energy_epsilon": float(energy.get("epsilon", eps_default)),
        "node_energy_scale_floor": _float_or_none(energy.get("scale_floor")),
        "node_energy_loss_type": str(energy.get("loss_type", "huber")).strip().lower(),
        "node_energy_huber_delta": float(energy.get("huber_delta", 1.0e-3)),
        "node_energy_residual_abs_clip": _float_or_none(energy.get("residual_abs_clip")),
        "node_energy_pred_residual_abs_max": _float_or_none(
            energy_prediction_filter.get("max_abs_residual", energy.get("max_pred_residual_abs"))
        ),
        "node_energy_valid_unit_types": tuple(
            str(v).strip().lower()
            for v in (
                energy.get(
                    "valid_unit_types",
                    [] if energy_has_nested_block else _cfg_get(train_cfg, "node_energy_valid_unit_types", []),
                )
                or []
            )
            if str(v).strip()
        ),
        "node_energy_exclude_unit_types": tuple(
            str(v).strip().lower()
            for v in (
                energy.get(
                    "exclude_unit_types",
                    [] if energy_has_nested_block else _cfg_get(train_cfg, "node_energy_exclude_unit_types", []),
                )
                or []
            )
            if str(v).strip()
        ),
        "node_energy_use_qw": bool(energy.get("use_process_main_qw", True)),
        "node_energy_signed_work_input": bool(energy.get("signed_work_input", True)),
        "hx_total_balance_only": bool(energy.get("hx_total_balance_only", True)),
        "use_q_hx": bool(energy.get("use_q_hx", False)),
        "node_energy_consistency_filter_enabled": bool(energy_consistency.get("enabled", False)),
        "node_energy_consistency_threshold": float(
            energy_consistency.get("max_true_normalized_residual", 1.0e-2)
        ),
        "node_energy_consistency_mode": str(energy_consistency.get("mode", "exclude")).strip().lower(),
        "eps": eps_default,
    }
    for key, value in cfg.items():
        if key.endswith(("epsilon", "huber_delta")) and (not math.isfinite(float(value)) or float(value) <= 0.0):
            raise ValueError(f"{key} must be finite and positive, got {value}.")
        if key.endswith("flow_threshold") and (not math.isfinite(float(value)) or float(value) < 0.0):
            raise ValueError(f"{key} must be finite and non-negative, got {value}.")
        if key.endswith("loss_type") and value not in {"squared", "mse", "huber", "smooth_l1", "l1", "mae"}:
            raise ValueError(f"Unsupported node balance loss_type={value!r}.")
        if key.endswith("pred_residual_abs_max") and value is not None and float(value) <= 0.0:
            raise ValueError(f"{key} must be positive when set, got {value}.")
        if key.endswith("residual_abs_clip") and value is not None and float(value) <= 0.0:
            raise ValueError(f"{key} must be positive when set, got {value}.")
    if cfg["node_energy_consistency_filter_enabled"]:
        threshold = float(cfg["node_energy_consistency_threshold"])
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(
                "node_balance_pi.energy.consistency_filter.max_true_normalized_residual "
                f"must be finite and positive, got {threshold}."
            )
        if cfg["node_energy_consistency_mode"] != "exclude":
            raise ValueError(
                "node_balance_pi.energy.consistency_filter.mode currently supports only "
                f"'exclude', got {cfg['node_energy_consistency_mode']!r}."
            )
    if cfg["use_q_hx"] and cfg["hx_total_balance_only"]:
        raise ValueError("node_balance_pi.energy.use_q_hx=true conflicts with hx_total_balance_only=true.")
    return cfg


def build_pi_atom_matrix(
    species_order: Sequence[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    species = tuple(str(value) for value in species_order)
    missing = [value for value in species if value not in PI_ATOM_COUNTS]
    if missing:
        raise RuntimeError(f"Missing atom-count rows for species: {missing}")
    return torch.tensor([PI_ATOM_COUNTS[value] for value in species], device=device, dtype=dtype)


def _unit_name_mask(node_unit_type: torch.Tensor, values: Sequence[str]) -> torch.Tensor:
    wanted = {str(value).strip().lower() for value in values if str(value).strip()}
    aliases = {
        "reactor": {"smr_reactor", "wgs_reactor"},
        "heat_exchanger": {"hx_dt", "hx_hot", "hx_cold"},
        "hx": {"hx_dt", "hx_hot", "hx_cold"},
    }
    expanded = set(wanted)
    for value in list(wanted):
        expanded.update(aliases.get(value, set()))
    ids = [idx for idx, name in enumerate(UNIT_VOCAB) if str(name).lower() in expanded]
    mask = torch.zeros_like(node_unit_type, dtype=torch.bool)
    for idx in ids:
        mask |= node_unit_type == int(idx)
    return mask


def _as_edge_matrix(x: torch.Tensor, *, num_edges: int, name: str) -> torch.Tensor:
    out = x.float().reshape(num_edges, -1)
    if out.shape[0] != num_edges:
        raise RuntimeError(f"{name} first dimension must match num_edges={num_edges}, got {tuple(out.shape)}")
    return out


def derive_component_molar_flows(
    mass_flow: torch.Tensor,
    fractions: torch.Tensor,
    molecular_weights: torch.Tensor,
    eps: float,
) -> dict[str, torch.Tensor]:
    mass = mass_flow.float().reshape(fractions.shape[0], -1)
    if mass.shape[-1] != 1:
        raise RuntimeError(f"mass_flow must have one column, got {tuple(mass.shape)}")
    frac = fractions.float()
    if frac.shape[-1] != int(molecular_weights.numel()):
        raise RuntimeError(
            f"fraction width {frac.shape[-1]} does not match molecular weights {int(molecular_weights.numel())}."
        )
    mw = molecular_weights.to(device=frac.device, dtype=frac.dtype)
    mixture_mw = (frac * mw.view(1, -1)).sum(dim=-1, keepdim=True)
    total_molar_flow = mass / mixture_mw.clamp_min(float(eps))
    component_molar_flow = total_molar_flow * frac
    return {
        "mixture_mw": mixture_mw,
        "total_molar_flow": total_molar_flow,
        "component_molar_flow": component_molar_flow,
    }


def derive_atom_flows(component_molar_flow: torch.Tensor, atom_matrix: torch.Tensor) -> torch.Tensor:
    return component_molar_flow.float() @ atom_matrix.to(
        device=component_molar_flow.device,
        dtype=component_molar_flow.dtype,
    )


def _node_sum(edge_values: torch.Tensor, *, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    out = torch.zeros((num_nodes, edge_values.shape[-1]), device=edge_values.device, dtype=edge_values.dtype)
    out.index_add_(0, index, edge_values)
    return out


def _incident_bad(edge_values: torch.Tensor, *, src: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
    finite_edge = torch.isfinite(edge_values).all(dim=-1)
    bad = (~finite_edge).to(dtype=torch.long)
    out = torch.zeros(num_nodes, device=edge_values.device, dtype=torch.long)
    out.index_add_(0, src, bad)
    out.index_add_(0, dst, bad)
    return out > 0


def _loss_values(residual: torch.Tensor, loss_type: str, huber_delta: float) -> torch.Tensor:
    if loss_type in {"squared", "mse"}:
        return residual.square()
    if loss_type in {"huber", "smooth_l1"}:
        return F.huber_loss(residual, torch.zeros_like(residual), delta=float(huber_delta), reduction="none")
    if loss_type in {"l1", "mae"}:
        return residual.abs()
    raise ValueError(f"Unsupported node balance loss_type={loss_type!r}.")


def _masked_loss(
    residual: torch.Tensor,
    valid: torch.Tensor,
    loss_type: str,
    huber_delta: float,
    residual_abs_clip: float | None = None,
) -> torch.Tensor:
    if valid.ndim == 1 and residual.ndim == 2:
        valid = valid.unsqueeze(-1).expand_as(residual)
    finite = torch.isfinite(residual)
    safe_residual = torch.where(finite, residual, torch.zeros_like(residual))
    valid = valid & finite
    selected = safe_residual[valid]
    if selected.numel() == 0:
        return safe_residual.sum() * 0.0
    if residual_abs_clip is not None:
        limit = float(residual_abs_clip)
        selected = selected.clamp(min=-limit, max=limit)
    return _loss_values(selected, loss_type, huber_delta).mean()


def _reduce_balance(
    pred_values: torch.Tensor,
    true_values: torch.Tensor,
    *,
    src: torch.Tensor,
    dst: torch.Tensor,
    num_nodes: int,
    base_valid: torch.Tensor,
    relative: bool,
    eps: float,
    scale_floor: float | None,
    valid_scale_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_safe = torch.where(torch.isfinite(pred_values), pred_values, torch.zeros_like(pred_values))
    true_safe = torch.where(torch.isfinite(true_values), true_values, torch.zeros_like(true_values)).detach()
    pred_in = _node_sum(pred_safe, index=dst, num_nodes=num_nodes)
    pred_out = _node_sum(pred_safe, index=src, num_nodes=num_nodes)
    true_in_abs = _node_sum(true_safe.abs(), index=dst, num_nodes=num_nodes)
    true_out_abs = _node_sum(true_safe.abs(), index=src, num_nodes=num_nodes)
    scale = true_in_abs + true_out_abs
    if scale_floor is not None:
        scale = torch.maximum(scale, scale.new_full(scale.shape, float(scale_floor)))
    scale = scale + float(eps)
    residual = pred_in - pred_out
    if relative:
        residual = residual / scale.detach()
    bad = _incident_bad(pred_values, src=src, dst=dst, num_nodes=num_nodes) | _incident_bad(
        true_values, src=src, dst=dst, num_nodes=num_nodes
    )
    valid = base_valid.unsqueeze(-1).expand_as(residual) & ~bad.unsqueeze(-1)
    if valid_scale_threshold > 0.0:
        valid &= scale > float(valid_scale_threshold)
    return residual, valid, scale.detach()


def _stats(prefix: str, valid: torch.Tensor, residual: torch.Tensor, touched_count: int) -> dict[str, Any]:
    selected = residual[valid] if valid.shape == residual.shape else residual[valid.reshape(-1)]
    abs_selected = selected.abs()
    # These additive moments make post-hoc, held-out conservation evaluation
    # exact across variable-size process graphs.  The thresholds apply to the
    # normalized residual used by the configured relative balance equation.
    thresholds = (("0p01", 1.0e-2), ("0p05", 5.0e-2), ("0p10", 1.0e-1))
    return {
        f"{prefix}_valid_count": int(valid.sum().detach().cpu()),
        f"{prefix}_skip_count": int(touched_count) - int(valid.any(dim=-1).sum().detach().cpu())
        if valid.ndim == 2
        else int(touched_count) - int(valid.sum().detach().cpu()),
        f"{prefix}_residual_mean": float(abs_selected.mean().detach().cpu()) if selected.numel() else 0.0,
        f"{prefix}_residual_max": float(abs_selected.max().detach().cpu()) if selected.numel() else 0.0,
        f"{prefix}_residual_abs_sum": float(abs_selected.sum().detach().cpu()),
        f"{prefix}_residual_sq_sum": float(selected.square().sum().detach().cpu()),
        **{
            f"{prefix}_residual_abs_le_{label}_count": int((abs_selected <= threshold).sum().detach().cpu())
            for label, threshold in thresholds
        },
    }


def _node_optional(batch_tensor: torch.Tensor | None, num_nodes: int, device: torch.device) -> torch.Tensor | None:
    if batch_tensor is None:
        return None
    out = batch_tensor.to(device=device, dtype=torch.float32).reshape(num_nodes, -1)
    if out.shape[-1] != 1:
        raise ValueError(f"Expected one value per node, got {tuple(out.shape)}.")
    return out


def compute_node_balance_pinn_losses(
    *,
    physical_outputs: Mapping[str, torch.Tensor],
    edge_index: torch.Tensor,
    edge_batch_or_graph_id: torch.Tensor,
    node_batch_or_graph_id: torch.Tensor,
    node_unit_type: torch.Tensor,
    species_order: Sequence[str],
    train_cfg: Any,
    physical_targets: Mapping[str, torch.Tensor] | None = None,
    node_q: torch.Tensor | None = None,
    node_w: torch.Tensor | None = None,
    node_qw_valid_mask: torch.Tensor | None = None,
    node_balance_exclude_mass: torch.Tensor | None = None,
    node_balance_exclude_component: torch.Tensor | None = None,
    node_balance_exclude_atom: torch.Tensor | None = None,
    node_balance_exclude_energy: torch.Tensor | None = None,
    edge_pinn_mask: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute node-level PI losses without changing edge-step semantics."""
    cfg = resolve_node_balance_config(train_cfg)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2,E], got {tuple(edge_index.shape)}.")
    src_all = edge_index[0].long()
    dst_all = edge_index[1].long()
    edge_graph_all = edge_batch_or_graph_id.long().reshape(-1)
    node_graph = node_batch_or_graph_id.long().reshape(-1)
    node_units = node_unit_type.long().reshape(-1)
    original_num_edges = int(src_all.numel())
    num_nodes = int(node_graph.numel())
    if edge_graph_all.numel() != original_num_edges:
        raise ValueError("edge_batch_or_graph_id length does not match edge_index.")
    if node_units.numel() != num_nodes:
        raise ValueError("node_unit_type length does not match node_batch_or_graph_id.")
    if original_num_edges and (
        int(src_all.min()) < 0
        or int(dst_all.min()) < 0
        or int(src_all.max()) >= num_nodes
        or int(dst_all.max()) >= num_nodes
    ):
        raise ValueError("edge_index references a node outside node_batch_or_graph_id.")
    if edge_pinn_mask is None:
        pinn_edge_mask = torch.ones(
            original_num_edges, device=edge_index.device, dtype=torch.bool
        )
    else:
        pinn_edge_mask = (
            edge_pinn_mask.to(device=edge_index.device).reshape(-1) > 0.5
        )
        if pinn_edge_mask.numel() != original_num_edges:
            raise ValueError(
                "edge_pinn_mask length does not match edge_index: "
                f"{pinn_edge_mask.numel()} vs {original_num_edges}."
            )
    pinn_edge_index = torch.nonzero(pinn_edge_mask, as_tuple=False).reshape(-1)
    src = src_all.index_select(0, pinn_edge_index)
    dst = dst_all.index_select(0, pinn_edge_index)
    edge_graph = edge_graph_all.index_select(0, pinn_edge_index)
    num_edges = int(src.numel())
    if num_edges and (
        not torch.equal(node_graph.index_select(0, src), edge_graph)
        or not torch.equal(node_graph.index_select(0, dst), edge_graph)
    ):
        raise RuntimeError("edge_index/edge_batch would mix edges across graph or sample boundaries.")

    species = [str(s) for s in species_order]
    if species != list(PI_SPECIES_ORDER):
        raise RuntimeError(f"node-level PINN expects species_order={list(PI_SPECIES_ORDER)}, got {species}.")

    device = edge_index.device
    in_degree = torch.zeros(num_nodes, device=device, dtype=torch.long)
    out_degree = torch.zeros_like(in_degree)
    ones = torch.ones(num_edges, device=device, dtype=torch.long)
    in_degree.index_add_(0, dst, ones)
    out_degree.index_add_(0, src, ones)
    touched = (in_degree + out_degree) > 0
    internal = touched & (in_degree > 0) & (out_degree > 0) & ~_unit_name_mask(node_units, BOUNDARY_NODE_TYPES)
    nonreactive = internal & _unit_name_mask(node_units, NONREACTIVE_NODE_TYPES)
    reactive = internal & _unit_name_mask(node_units, REACTIVE_NODE_TYPES)
    hx_nodes = _unit_name_mask(node_units, HX_NODE_TYPES)

    mass_pred = _as_edge_matrix(
        physical_outputs["mass_flow"],
        num_edges=original_num_edges,
        name="mass_flow",
    ).index_select(0, pinn_edge_index)
    frac_pred = _as_edge_matrix(
        physical_outputs["frac"],
        num_edges=original_num_edges,
        name="frac",
    ).index_select(0, pinn_edge_index)
    energy_enabled = bool(cfg["use_node_energy"] and cfg["lambda_node_energy"] > 0.0)
    hflow_output = physical_outputs.get("enthalpy_flow")
    energy_available = hflow_output is not None
    if energy_enabled and hflow_output is None:
        raise RuntimeError(
            "Node energy PINN is enabled, but the selected edge head does not provide Enthalpy."
        )
    hflow_pred = (
        _as_edge_matrix(
            hflow_output,
            num_edges=original_num_edges,
            name="enthalpy_flow",
        ).index_select(0, pinn_edge_index)
        if hflow_output is not None
        else torch.zeros_like(mass_pred)
    )
    if physical_targets is not None and physical_targets.get("mass_flow") is not None:
        mass_true = _as_edge_matrix(
            physical_targets["mass_flow"],
            num_edges=original_num_edges,
            name="target_mass_flow",
        ).index_select(0, pinn_edge_index)
    else:
        mass_true = mass_pred.detach()
    if physical_targets is not None and physical_targets.get("frac") is not None:
        frac_true = _as_edge_matrix(
            physical_targets["frac"],
            num_edges=original_num_edges,
            name="target_frac",
        ).index_select(0, pinn_edge_index)
    else:
        frac_true = frac_pred.detach()
    if energy_available and physical_targets is not None and physical_targets.get("enthalpy_flow") is not None:
        hflow_true = _as_edge_matrix(
            physical_targets["enthalpy_flow"],
            num_edges=original_num_edges,
            name="target_enthalpy_flow",
        ).index_select(0, pinn_edge_index)
    else:
        hflow_true = hflow_pred.detach()

    active_true = mass_true > float(cfg["zero_flow_threshold"])
    if bool(cfg["zero_flow_nan_enthalpy_as_zero"]):
        hflow_true = torch.where(active_true | torch.isfinite(hflow_true), hflow_true, torch.zeros_like(hflow_true))

    molecular_weights = torch.tensor(
        [18.01528, 2.01588, 16.04246, 44.0095, 28.0101, 31.9988, 28.0134],
        device=device,
        dtype=mass_pred.dtype,
    )
    comp_pred = derive_component_molar_flows(
        mass_pred,
        frac_pred,
        molecular_weights,
        float(cfg["node_component_epsilon"]),
    )["component_molar_flow"]
    comp_true = derive_component_molar_flows(
        mass_true.detach(),
        frac_true.detach(),
        molecular_weights,
        float(cfg["node_component_epsilon"]),
    )["component_molar_flow"]
    atom_matrix = build_pi_atom_matrix(species, device=device, dtype=comp_pred.dtype)
    atom_pred = derive_atom_flows(comp_pred, atom_matrix)
    atom_true = derive_atom_flows(comp_true.detach(), atom_matrix)

    excl_mass = _node_optional(node_balance_exclude_mass, num_nodes, device)
    excl_component = _node_optional(node_balance_exclude_component, num_nodes, device)
    excl_atom = _node_optional(node_balance_exclude_atom, num_nodes, device)
    excl_energy = _node_optional(node_balance_exclude_energy, num_nodes, device)
    mass_base = internal if excl_mass is None else internal & (excl_mass.reshape(-1) < 0.5)
    component_base = nonreactive if excl_component is None else nonreactive & (excl_component.reshape(-1) < 0.5)
    atom_base = reactive if excl_atom is None else reactive & (excl_atom.reshape(-1) < 0.5)
    energy_base = internal if excl_energy is None else internal & (excl_energy.reshape(-1) < 0.5)
    if cfg["node_energy_valid_unit_types"]:
        energy_base &= _unit_name_mask(node_units, cfg["node_energy_valid_unit_types"])
    if cfg["node_energy_exclude_unit_types"]:
        energy_base &= ~_unit_name_mask(node_units, cfg["node_energy_exclude_unit_types"])

    mass_residual, mass_valid, _ = _reduce_balance(
        mass_pred,
        mass_true.detach(),
        src=src,
        dst=dst,
        num_nodes=num_nodes,
        base_valid=mass_base,
        relative=cfg["node_mass_relative"],
        eps=cfg["node_mass_epsilon"],
        scale_floor=cfg["node_mass_scale_floor"],
    )
    mass_pre_prediction_valid = mass_valid.clone()
    mass_prediction_mask = torch.ones_like(mass_valid, dtype=torch.bool)
    if cfg["node_mass_pred_residual_abs_max"] is not None:
        mass_prediction_mask = mass_residual.detach().abs() <= float(cfg["node_mass_pred_residual_abs_max"])
        mass_valid = mass_valid & mass_prediction_mask
    component_residual, component_valid, _ = _reduce_balance(
        comp_pred,
        comp_true.detach(),
        src=src,
        dst=dst,
        num_nodes=num_nodes,
        base_valid=component_base,
        relative=cfg["node_component_relative"],
        eps=cfg["node_component_epsilon"],
        scale_floor=cfg["node_component_scale_floor"],
        valid_scale_threshold=cfg["node_component_flow_threshold"],
    )
    atom_residual, atom_valid, _ = _reduce_balance(
        atom_pred,
        atom_true.detach(),
        src=src,
        dst=dst,
        num_nodes=num_nodes,
        base_valid=atom_base,
        relative=cfg["node_atom_relative"],
        eps=cfg["node_atom_epsilon"],
        scale_floor=cfg["node_atom_scale_floor"],
        valid_scale_threshold=cfg["node_atom_flow_threshold"],
    )

    if energy_available:
        q = _node_optional(node_q, num_nodes, device)
        w = _node_optional(node_w, num_nodes, device)
        qw_valid = _node_optional(node_qw_valid_mask, num_nodes, device)
        if not cfg["node_energy_use_qw"]:
            q = None
            w = None
            qw_valid = None
        if q is None:
            q = torch.zeros((num_nodes, 1), device=device, dtype=hflow_pred.dtype)
        if w is None:
            w = torch.zeros((num_nodes, 1), device=device, dtype=hflow_pred.dtype)
        if qw_valid is None:
            qw_valid = torch.ones((num_nodes, 1), device=device, dtype=hflow_pred.dtype)
        q = q.detach()
        w = w.detach()
        if cfg["hx_total_balance_only"] and not cfg["use_q_hx"]:
            q = torch.where(hx_nodes.reshape(-1, 1), torch.zeros_like(q), q)
        if not cfg["node_energy_signed_work_input"]:
            w = -w
        energy_bad = _incident_bad(hflow_pred, src=src, dst=dst, num_nodes=num_nodes) | _incident_bad(
            hflow_true, src=src, dst=dst, num_nodes=num_nodes
        )
        pred_in_h = _node_sum(
            torch.where(torch.isfinite(hflow_pred), hflow_pred, torch.zeros_like(hflow_pred)),
            index=dst,
            num_nodes=num_nodes,
        )
        pred_out_h = _node_sum(
            torch.where(torch.isfinite(hflow_pred), hflow_pred, torch.zeros_like(hflow_pred)),
            index=src,
            num_nodes=num_nodes,
        )
        true_safe_h = torch.where(torch.isfinite(hflow_true), hflow_true, torch.zeros_like(hflow_true)).detach()
        true_in_h = _node_sum(true_safe_h, index=dst, num_nodes=num_nodes)
        true_out_h = _node_sum(true_safe_h, index=src, num_nodes=num_nodes)
        true_scale = _node_sum(true_safe_h.abs(), index=dst, num_nodes=num_nodes) + _node_sum(
            true_safe_h.abs(), index=src, num_nodes=num_nodes
        )
        energy_scale = true_scale + q.abs() + w.abs()
        if cfg["node_energy_scale_floor"] is not None:
            energy_scale = torch.maximum(
                energy_scale,
                energy_scale.new_full(energy_scale.shape, float(cfg["node_energy_scale_floor"])),
            )
        energy_scale = energy_scale + float(cfg["node_energy_epsilon"])
        energy_residual = pred_in_h + q + w - pred_out_h
        energy_true_residual = true_in_h + q + w - true_out_h
        if cfg["node_energy_relative"]:
            energy_residual = energy_residual / energy_scale.detach()
            energy_true_residual = energy_true_residual / energy_scale.detach()
        energy_structural_valid = energy_base.reshape(-1, 1) & (qw_valid > 0.5) & ~energy_bad.reshape(-1, 1)
        energy_consistency_mask = torch.ones_like(energy_structural_valid, dtype=torch.bool)
        if cfg["node_energy_consistency_filter_enabled"]:
            threshold = float(cfg["node_energy_consistency_threshold"])
            energy_consistency_mask = energy_true_residual.detach().abs() <= threshold
        energy_prediction_mask = torch.ones_like(energy_structural_valid, dtype=torch.bool)
        if cfg["node_energy_pred_residual_abs_max"] is not None:
            energy_prediction_mask = energy_residual.detach().abs() <= float(cfg["node_energy_pred_residual_abs_max"])
        energy_valid = energy_structural_valid & energy_consistency_mask & energy_prediction_mask
    else:
        energy_residual = mass_pred.new_zeros((num_nodes, 1))
        energy_true_residual = energy_residual.detach()
        qw_valid = torch.ones((num_nodes, 1), device=device, dtype=mass_pred.dtype)
        energy_structural_valid = torch.zeros((num_nodes, 1), device=device, dtype=torch.bool)
        energy_consistency_mask = energy_structural_valid.clone()
        energy_prediction_mask = energy_structural_valid.clone()
        energy_valid = energy_structural_valid.clone()

    mass_loss = _masked_loss(
        mass_residual,
        mass_valid,
        cfg["node_mass_loss_type"],
        cfg["node_mass_huber_delta"],
        cfg["node_mass_residual_abs_clip"],
    )
    component_loss = _masked_loss(
        component_residual,
        component_valid,
        cfg["node_component_loss_type"],
        cfg["node_component_huber_delta"],
        cfg["node_component_residual_abs_clip"],
    )
    atom_loss = _masked_loss(
        atom_residual,
        atom_valid,
        cfg["node_atom_loss_type"],
        cfg["node_atom_huber_delta"],
        cfg["node_atom_residual_abs_clip"],
    )
    energy_loss = _masked_loss(
        energy_residual,
        energy_valid,
        cfg["node_energy_loss_type"],
        cfg["node_energy_huber_delta"],
        cfg["node_energy_residual_abs_clip"],
    )

    weighted_mass = (
        mass_loss * cfg["lambda_node_mass"]
        if cfg["use_node_mass"] and cfg["lambda_node_mass"] > 0.0
        else torch.zeros_like(mass_loss)
    )
    weighted_component = (
        component_loss * cfg["lambda_node_component"]
        if cfg["use_node_component"] and cfg["lambda_node_component"] > 0.0
        else torch.zeros_like(component_loss)
    )
    weighted_atom = (
        atom_loss * cfg["lambda_node_atom"]
        if cfg["use_node_atom"] and cfg["lambda_node_atom"] > 0.0
        else torch.zeros_like(atom_loss)
    )
    weighted_energy = (
        energy_loss * cfg["lambda_node_energy"]
        if cfg["use_node_energy"] and cfg["lambda_node_energy"] > 0.0
        else torch.zeros_like(energy_loss)
    )
    weighted_total = weighted_mass + weighted_component + weighted_atom + weighted_energy

    if not cfg["compute_diagnostics"]:
        contributes = {
            "node_mass_contributes_to_total": bool(cfg["use_node_mass"] and cfg["lambda_node_mass"] > 0),
            "node_component_contributes_to_total": bool(
                cfg["use_node_component"] and cfg["lambda_node_component"] > 0
            ),
            "node_atom_contributes_to_total": bool(cfg["use_node_atom"] and cfg["lambda_node_atom"] > 0),
            "node_energy_contributes_to_total": bool(
                cfg["use_node_energy"] and cfg["lambda_node_energy"] > 0
            ),
        }
        return {
            "loss_node_mass": mass_loss,
            "loss_node_component": component_loss,
            "loss_node_atom": atom_loss,
            "loss_node_energy": energy_loss,
            "loss_node_pi_total": weighted_total,
            "weighted_node_mass": weighted_mass,
            "weighted_node_component": weighted_component,
            "weighted_node_atom": weighted_atom,
            "weighted_node_energy": weighted_energy,
            "weighted_node_total": weighted_total,
            "diagnostics": {
                "loss_node_mass": float(mass_loss.detach().cpu()),
                "loss_node_component": float(component_loss.detach().cpu()),
                "loss_node_atom": float(atom_loss.detach().cpu()),
                "loss_node_energy": float(energy_loss.detach().cpu()),
                "loss_node_pi_total": float(weighted_total.detach().cpu()),
                "node_mass_valid_count": int(mass_valid.sum().detach().cpu()),
                "node_component_valid_count": int(component_valid.sum().detach().cpu()),
                "node_atom_valid_count": int(atom_valid.sum().detach().cpu()),
                "node_energy_valid_count": int(energy_valid.sum().detach().cpu()),
                **contributes,
            },
        }

    touched_count = int(touched.sum().detach().cpu())
    diagnostics: dict[str, Any] = {
        "loss_node_mass": float(mass_loss.detach().cpu()),
        "loss_node_component": float(component_loss.detach().cpu()),
        "loss_node_atom": float(atom_loss.detach().cpu()),
        "loss_node_energy": float(energy_loss.detach().cpu()),
        "loss_node_pi_total": float(weighted_total.detach().cpu()),
        "lambda_node_mass": cfg["lambda_node_mass"],
        "lambda_node_component": cfg["lambda_node_component"],
        "lambda_node_atom": cfg["lambda_node_atom"],
        "lambda_node_energy": cfg["lambda_node_energy"],
        "node_mass_residual_abs_clip": cfg["node_mass_residual_abs_clip"],
        "node_component_residual_abs_clip": cfg["node_component_residual_abs_clip"],
        "node_atom_residual_abs_clip": cfg["node_atom_residual_abs_clip"],
        "node_energy_residual_abs_clip": cfg["node_energy_residual_abs_clip"],
        "node_mass_contributes_to_total": bool(cfg["use_node_mass"] and cfg["lambda_node_mass"] > 0),
        "node_component_contributes_to_total": bool(cfg["use_node_component"] and cfg["lambda_node_component"] > 0),
        "node_atom_contributes_to_total": bool(cfg["use_node_atom"] and cfg["lambda_node_atom"] > 0),
        "node_energy_contributes_to_total": bool(cfg["use_node_energy"] and cfg["lambda_node_energy"] > 0),
        "node_energy_consistency_filter_enabled": bool(cfg["node_energy_consistency_filter_enabled"]),
        "node_energy_consistency_threshold": float(cfg["node_energy_consistency_threshold"]),
        "node_energy_structural_valid_count": int(energy_structural_valid.sum().detach().cpu()),
        "node_energy_consistent_count": int((energy_structural_valid & energy_consistency_mask).sum().detach().cpu()),
        "node_energy_consistency_rejected_count": int(
            (energy_structural_valid & ~energy_consistency_mask).sum().detach().cpu()
        ),
        "node_energy_consistency_accept_ratio": (
            float((energy_structural_valid & energy_consistency_mask).sum().detach().cpu())
            / max(float(energy_structural_valid.sum().detach().cpu()), 1.0)
        ),
        "node_mass_pred_residual_filter_threshold": (
            float(cfg["node_mass_pred_residual_abs_max"])
            if cfg["node_mass_pred_residual_abs_max"] is not None
            else 0.0
        ),
        "node_mass_pred_residual_rejected_count": int(
            (mass_pre_prediction_valid & ~mass_prediction_mask).sum().detach().cpu()
        )
        if cfg["node_mass_pred_residual_abs_max"] is not None
        else 0,
        "node_energy_pred_residual_filter_threshold": (
            float(cfg["node_energy_pred_residual_abs_max"])
            if cfg["node_energy_pred_residual_abs_max"] is not None
            else 0.0
        ),
        "node_energy_pred_residual_rejected_count": int(
            (energy_structural_valid & energy_consistency_mask & ~energy_prediction_mask).sum().detach().cpu()
        )
        if cfg["node_energy_pred_residual_abs_max"] is not None
        else 0,
        "node_energy_true_residual_abs_mean": float(
            energy_true_residual.detach().abs()[energy_structural_valid].mean().cpu()
        )
        if bool(energy_structural_valid.any())
        else 0.0,
        "node_energy_true_residual_abs_max": float(
            energy_true_residual.detach().abs()[energy_structural_valid].max().cpu()
        )
        if bool(energy_structural_valid.any())
        else 0.0,
        "node_energy_true_residual_abs_p95": float(
            torch.quantile(energy_true_residual.detach().abs()[energy_structural_valid], 0.95).cpu()
        )
        if bool(energy_structural_valid.any())
        else 0.0,
        "node_energy_pred_residual_abs_mean": float(energy_residual.detach().abs()[energy_valid].mean().cpu())
        if bool(energy_valid.any())
        else 0.0,
        "node_graph_count": int(torch.unique(node_graph).numel()),
        "node_pinn_edge_count": int(num_edges),
        "node_pinn_edge_excluded_count": int(original_num_edges - num_edges),
        "node_touched_count": touched_count,
        "node_boundary_count": int(_unit_name_mask(node_units, BOUNDARY_NODE_TYPES).sum().detach().cpu()),
        "node_nonreactive_count": int(nonreactive.sum().detach().cpu()),
        "node_reactive_count": int(reactive.sum().detach().cpu()),
        "node_hx_count": int(hx_nodes.sum().detach().cpu()),
        "node_qw_invalid_count": int((qw_valid.reshape(-1) <= 0.5).sum().detach().cpu()),
        "node_excluded_mass_count": int(excl_mass.sum().detach().cpu()) if excl_mass is not None else 0,
        "node_excluded_component_count": int(excl_component.sum().detach().cpu()) if excl_component is not None else 0,
        "node_excluded_atom_count": int(excl_atom.sum().detach().cpu()) if excl_atom is not None else 0,
        "node_excluded_energy_count": int(excl_energy.sum().detach().cpu()) if excl_energy is not None else 0,
        **_stats("node_mass", mass_valid, mass_residual, touched_count),
        **_stats("node_component", component_valid, component_residual, touched_count),
        **_stats("node_atom", atom_valid, atom_residual, touched_count),
        **_stats("node_energy", energy_valid, energy_residual, touched_count),
    }
    return {
        "loss_node_mass": mass_loss,
        "loss_node_component": component_loss,
        "loss_node_atom": atom_loss,
        "loss_node_energy": energy_loss,
        "loss_node_pi_total": weighted_total,
        "weighted_node_mass": weighted_mass,
        "weighted_node_component": weighted_component,
        "weighted_node_atom": weighted_atom,
        "weighted_node_energy": weighted_energy,
        "weighted_node_total": weighted_total,
        "diagnostics": diagnostics,
    }
