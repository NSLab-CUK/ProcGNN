from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR

from ..constants import STREAM_EDGE_FEATURE_SLOTS
from .schema import DECODER_CATEGORIES, DataConfig, TrainConfig


def build_optimizer(model: nn.Module, cfg: TrainConfig):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found for optimizer.")
    if cfg.optimizer == "adam":
        return Adam(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unsupported optimizer '{cfg.optimizer}'.")


def build_scheduler(optimizer, cfg: TrainConfig):
    if cfg.scheduler == "none":
        return None
    if cfg.scheduler == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg.scheduler_t_max,
            eta_min=cfg.scheduler_eta_min,
        )
    if cfg.scheduler == "step":
        return StepLR(
            optimizer,
            step_size=cfg.scheduler_step_size,
            gamma=cfg.scheduler_gamma,
        )
    if cfg.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            min_lr=cfg.scheduler_min_lr,
        )
    raise ValueError(f"Unsupported scheduler '{cfg.scheduler}'.")


def build_loss_fn(loss_type: str):
    if loss_type == "mse":
        return nn.functional.mse_loss
    if loss_type == "l1":
        return nn.functional.l1_loss
    if loss_type == "smooth_l1":
        return nn.functional.smooth_l1_loss
    if loss_type == "percentage":

        def percentage_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            denom = target.detach().abs().clamp_min(1e-6)
            return ((pred - target).abs() / denom).mean()

        return percentage_loss
    raise ValueError(f"Unsupported loss_type '{loss_type}'.")


def masked_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_fn,
) -> torch.Tensor:
    mask = mask.to(dtype=pred.dtype)
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(pred)
    valid = mask > 0
    if not valid.any():
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return loss_fn(pred[valid], target[valid])


def compute_three_slot_loss(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    target_masks: Dict[str, torch.Tensor],
    cfg: TrainConfig,
    *,
    aux_masked: bool,
) -> Dict[str, torch.Tensor]:
    required = {"target", "tailgas", "aux"}
    missing_pred = required - set(preds.keys())
    missing_target = required - set(targets.keys())
    if missing_pred:
        raise RuntimeError(f"Missing prediction slots: {sorted(missing_pred)}")
    if missing_target:
        raise RuntimeError(f"Missing target slots: {sorted(missing_target)}")

    criterion_target = build_loss_fn(cfg.loss_type_target)
    criterion_tailgas = build_loss_fn(cfg.loss_type_tailgas)
    criterion_aux = build_loss_fn(cfg.loss_type_aux)

    loss_target = criterion_target(preds["target"], targets["target"])
    loss_tailgas = criterion_tailgas(preds["tailgas"], targets["tailgas"])
    if aux_masked:
        aux_mask = target_masks.get("aux")
        if aux_mask is None:
            raise RuntimeError("aux is configured as masked but target_masks['aux'] is missing.")
        loss_aux = masked_loss(preds["aux"], targets["aux"], aux_mask, criterion_aux)
    else:
        loss_aux = criterion_aux(preds["aux"], targets["aux"])

    lambda_target = float(cfg.task_loss_weights.get("target", 1.0))
    lambda_tailgas = float(cfg.task_loss_weights.get("tailgas", 1.0))
    lambda_aux = float(cfg.task_loss_weights.get("aux", 0.3))
    total = lambda_target * loss_target + lambda_tailgas * loss_tailgas + lambda_aux * loss_aux
    out = {"loss_target": loss_target, "loss_tailgas": loss_tailgas, "loss_aux": loss_aux}

    loss_es, term_es = _edge_stream_loss_weighted(preds, targets, target_masks, cfg)
    if float(cfg.edge_stream_loss_weight) > 0.0 and "edge_stream" in preds:
        out["loss_edge_stream"] = loss_es
        total = total + term_es
    out["loss_total"] = total
    return out


def decoder_tasks_enabled(data_cfg: DataConfig) -> bool:
    return any(t.enabled for t in data_cfg.decoder_tasks.values())


def loss_metric_keys(data_cfg: DataConfig, train_cfg: Optional[TrainConfig] = None) -> list[str]:
    if getattr(data_cfg, "task_mode", "multitask") == "edge_all":
        return [
            "loss_total",
            "loss_total_final",
            "loss_edge_all",
            "loss_edge",
            "loss_primary_frac",
            "loss_all_edge_r2",
            "weighted_edge_all",
            "weighted_primary_frac",
            "weighted_all_edge_r2",
            "loss_amount_auxiliary",
            "loss_target_row_amount",
            "weighted_amount_auxiliary_disabled",
            "edge_step_loss_mean",
            "edge_step_pi_loss_mean",
            "sample_hybrid_target_edge_step_pi_loss_mean",
            "target_edge_loss_mean",
            "non_target_edge_loss_mean",
            "edge_update_count",
            "optimizer_step_count",
            "non_target_optimizer_step_count",
            "target_optimizer_step_count",
            "node_optimizer_step_count",
            "skipped_edge_count",
            "num_edges_per_batch",
            "num_edge_rows_per_batch",
            "data_iteration_count",
            "edge_step_forward_count",
            "edge_step_pi_forward_count",
            "graph_sample_count",
            "sample_hybrid_target_group_count",
            "sample_hybrid_non_target_group_count",
            "target_edge_existing_weight_applied",
            "profile_grouping_sec",
            "profile_non_target_forward_loss_sec",
            "profile_non_target_backward_step_sec",
            "profile_target_forward_loss_sec",
            "profile_target_backward_step_sec",
            "profile_node_forward_loss_sec",
            "profile_node_backward_step_sec",
            "grad_norm",
            "edge_step_target_edge_count",
            "loss_main_mean",
            "loss_rho_mean",
            "loss_h_mean",
            "loss_volume_mean",
            "loss_enthalpy_flow_mean",
            "loss_atom_mean",
            "loss_energy_mean",
            "loss_rho_max",
            "loss_h_max",
            "loss_volume_max",
            "loss_enthalpy_flow_max",
            "edge_weight_min",
            "edge_weight_max",
            "edge_weight_mean",
            "edge_pred_output_dim",
            "loss_extra_unexpected",
        ]
    keys = ["loss_total", "loss_target"]
    if getattr(data_cfg, "task_mode", "multitask") == "multitask":
        keys.append("loss_tailgas")
    if decoder_tasks_enabled(data_cfg):
        for category in DECODER_CATEGORIES:
            cfg = data_cfg.decoder_tasks.get(category)
            if cfg is not None and cfg.enabled:
                keys.append(f"loss_{category}")
    elif data_cfg.aux_task.enabled:
        keys.append("loss_aux")
    elif train_cfg is not None:
        if float(getattr(train_cfg, "edge_stream_loss_weight", 0.0)) > 0.0 and getattr(
            data_cfg, "use_canonical_graph_spec_v3", False
        ):
            keys.append("loss_edge_stream")
    return keys


def _edge_stream_loss_weighted(
    preds: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    target_masks: Mapping[str, torch.Tensor],
    train_cfg: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked edge-stream loss and scalar term to add to loss_total."""
    ref = next(iter(preds.values())) if preds else None
    if ref is None:
        z = torch.zeros(())
        return z, z
    z = torch.zeros((), device=ref.device, dtype=ref.dtype)
    w = float(getattr(train_cfg, "edge_stream_loss_weight", 0.0))
    if w <= 0.0 or "edge_stream" not in preds or "edge_stream" not in targets:
        return z, z
    mask = target_masks.get("edge_stream")
    if mask is None:
        raise RuntimeError("edge_stream predictions/targets are present but target_masks['edge_stream'] is missing.")
    criterion = build_loss_fn(getattr(train_cfg, "loss_type_edge_stream", "mse"))
    loss_es = masked_loss(preds["edge_stream"], targets["edge_stream"], mask, criterion)
    lam = float(train_cfg.task_loss_weights.get("edge_stream", 1.0))
    return loss_es, w * lam * loss_es


def compute_fixed_only_loss(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    train_cfg: TrainConfig,
    *,
    target_only: bool = False,
    target_masks: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    masks = target_masks or {}
    if "target" not in preds or "target" not in targets:
        raise RuntimeError("Missing 'target' in preds or targets.")
    criterion_target = build_loss_fn(train_cfg.loss_type_target)
    loss_target = criterion_target(preds["target"], targets["target"])
    lambda_target = float(train_cfg.task_loss_weights.get("target", 1.0))
    total = lambda_target * loss_target
    out: Dict[str, torch.Tensor] = {"loss_target": loss_target}

    if not target_only:
        if "tailgas" not in preds or "tailgas" not in targets:
            raise RuntimeError("Missing 'tailgas' in preds or targets.")
        criterion_tailgas = build_loss_fn(train_cfg.loss_type_tailgas)
        loss_tailgas = criterion_tailgas(preds["tailgas"], targets["tailgas"])
        lambda_tailgas = float(train_cfg.task_loss_weights.get("tailgas", 1.0))
        total = total + lambda_tailgas * loss_tailgas
        out["loss_tailgas"] = loss_tailgas

    loss_es, term_es = _edge_stream_loss_weighted(preds, targets, masks, train_cfg)
    if float(getattr(train_cfg, "edge_stream_loss_weight", 0.0)) > 0.0 and "edge_stream" in preds:
        out["loss_edge_stream"] = loss_es
        total = total + term_es
    out["loss_total"] = total
    return out


def compute_multitask_decoder_loss(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    target_masks: Dict[str, torch.Tensor],
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
) -> Dict[str, torch.Tensor]:
    target_only = getattr(data_cfg, "task_mode", "multitask") == "target_only"
    items = compute_fixed_only_loss(
        preds,
        targets,
        train_cfg,
        target_only=target_only,
        target_masks=target_masks,
    )
    total = items["loss_total"]
    ref = total
    for category in DECODER_CATEGORIES:
        cat_cfg = data_cfg.decoder_tasks.get(category)
        if cat_cfg is None or not cat_cfg.enabled:
            continue
        pred = preds.get(category)
        target = targets.get(category)
        if pred is None or target is None:
            raise RuntimeError(f"Decoder category '{category}' is enabled but preds/targets are missing.")
        loss_type = cat_cfg.loss_type or train_cfg.loss_type_decoder
        criterion = build_loss_fn(loss_type)
        loss_cat = pred.sum() * 0.0 if int(target.numel()) == 0 else criterion(pred, target)
        items[f"loss_{category}"] = loss_cat
        lam = float(train_cfg.task_loss_weights.get(category, 0.3))
        total = total + lam * loss_cat
        ref = loss_cat
    if not torch.is_tensor(ref):
        raise RuntimeError("Internal loss construction error.")
    items["loss_total"] = total
    return items


def resolve_answer_edge_species_weights(train_cfg: Any) -> dict[str, float]:
    default = float(getattr(train_cfg, "answer_edge_weight", 5.0))
    return {
        "H2": float(getattr(train_cfg, "answer_edge_weight_h2", default)),
        "CO2": float(getattr(train_cfg, "answer_edge_weight_co2", default)),
        "H2O": float(getattr(train_cfg, "answer_edge_weight_h2o", default)),
    }


def _assert_finite(name: str, t: torch.Tensor) -> None:
    if not torch.isfinite(t).all():
        raise RuntimeError(f"{name} contains NaN or Inf.")


def masked_edge_regression_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_type: str = "mse",
    element_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean loss over elements where mask is positive."""
    _assert_finite("y_edge_pred", pred)
    _assert_finite("y_edge_true", true)
    if pred.shape != true.shape:
        raise RuntimeError(
            f"y_edge_pred shape {tuple(pred.shape)} != y_edge_true shape {tuple(true.shape)}."
        )
    m = mask.to(dtype=pred.dtype, device=pred.device)
    while m.ndim < pred.ndim:
        m = m.unsqueeze(-1)
    try:
        m = m.expand_as(pred)
    except RuntimeError as exc:
        raise RuntimeError(
            f"y_edge_mask shape {tuple(mask.shape)} cannot broadcast to pred shape {tuple(pred.shape)}."
        ) from exc
    if element_weight is not None:
        w = element_weight.to(dtype=pred.dtype, device=pred.device)
        while w.ndim < pred.ndim:
            w = w.unsqueeze(-1)
        try:
            w = w.expand_as(pred)
        except RuntimeError as exc:
            raise RuntimeError(
                f"edge_stream_loss_weight shape {tuple(element_weight.shape)} cannot broadcast to pred shape {tuple(pred.shape)}."
            ) from exc
        m = m * w
    denom = m.sum()
    if denom.item() <= 0:
        raise RuntimeError("masked_edge_regression_loss: mask sum is zero (no supervised edge elements).")
    diff = pred - true
    if loss_type == "mse":
        return (m * (diff**2)).sum() / denom, denom
    if loss_type == "l1":
        return (m * diff.abs()).sum() / denom, denom
    if loss_type == "smooth_l1":
        huber = torch.nn.functional.smooth_l1_loss(pred, true, reduction="none")
        return (m * huber).sum() / denom, denom
    raise ValueError(f"Unsupported loss_type for edge_all: {loss_type!r} (use mse, l1, or smooth_l1).")


def masked_edge_mae(pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(dtype=pred.dtype, device=pred.device)
    while m.ndim < pred.ndim:
        m = m.unsqueeze(-1)
    m = m.expand_as(pred)
    denom = m.sum()
    if denom.item() <= 0:
        return pred.sum() * 0.0
    return (m * (pred - true).abs()).sum() / denom


def resolve_frac_h2_co2_column_indices(edge_target_columns: Sequence[str]) -> tuple[int, int]:
    cols = [str(c) for c in edge_target_columns]
    h2_cands = [i for i, c in enumerate(cols) if "H2" in c.upper() and "H2O" not in c.upper()]
    if len(h2_cands) != 1:
        raise RuntimeError(f"Expected exactly one H2 (non-H2O) column in edge_target_columns; got {h2_cands}.")
    co2_cands = [i for i, c in enumerate(cols) if "CO2" in c.upper()]
    if len(co2_cands) != 1:
        raise RuntimeError(f"Expected exactly one CO2 column in edge_target_columns; got {co2_cands}.")
    return int(h2_cands[0]), int(co2_cands[0])


def validate_edge_all_batch(
    *,
    y_edge_pred: torch.Tensor,
    y_edge_true: torch.Tensor,
    y_edge_mask: torch.Tensor,
    edge_struct: torch.Tensor,
    edge_struct_dim: int,
    answer_edge_pos,
) -> None:
    if y_edge_pred.shape != y_edge_true.shape:
        raise RuntimeError(
            f"y_edge_pred shape {tuple(y_edge_pred.shape)} != y_edge_true shape {tuple(y_edge_true.shape)}."
        )
    e = y_edge_pred.size(0)
    if edge_struct.shape[0] != e or edge_struct.shape[-1] != int(edge_struct_dim):
        raise RuntimeError(
            f"edge_struct_attr expected shape ({e}, {edge_struct_dim}), got {tuple(edge_struct.shape)}."
        )
    if answer_edge_pos is not None:
        for bi, slot_map in enumerate(answer_edge_pos):
            for slot, pos in slot_map.items():
                pi = int(pos)
                if pi < 0 or pi >= e:
                    raise RuntimeError(
                        f"answer_edge_pos out of range: batch graph index={bi} slot={slot!r} pos={pi} "
                        f"num_edges={e}."
                    )
                if float(y_edge_mask[pi].item()) <= 0.0:
                    raise RuntimeError(
                        f"answer edge y_edge_mask is 0: batch graph index={bi} slot={slot!r} global_pos={pi}."
                    )


def compute_edge_all_training_loss(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    target_masks: Dict[str, torch.Tensor],
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    *,
    edge_export_meta=None,
    edge_target_columns: Sequence[str] | None = None,
    edge_index: torch.Tensor | None = None,
    y_edge_mean: torch.Tensor | None = None,
    y_edge_std: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    if "y_edge_pred" not in preds:
        raise RuntimeError("edge_all mode requires preds['y_edge_pred'].")
    if "edge_stream" not in targets:
        raise RuntimeError("edge_all mode requires targets['edge_stream'] (batched y_edge_true).")
    if "edge_stream" not in target_masks:
        raise RuntimeError("edge_all mode requires target_masks['edge_stream'].")

    from .metric_policy import resolve_loss_lambdas
    from .target_row_loss import (
        compute_all_edge_r2_loss,
        compute_primary_frac_losses,
        compute_v4_target_row_amount_loss,
        resolve_all_edge_r2_config,
        warn_if_duplicate_legacy_weighting,
    )
    from .target_stream_weighting import build_target_stream_loss_weight_tensor

    warn_if_duplicate_legacy_weighting(train_cfg)
    lt = str(getattr(train_cfg, "loss_type_edge_all", "mse"))
    use_edge, use_feat, _use_amount_requested, _lam_feat, lam_amount = resolve_loss_lambdas(train_cfg)
    lam_primary = float(getattr(train_cfg, "primary_frac_loss_weight", _lam_feat))
    all_edge_r2_cfg = resolve_all_edge_r2_config(train_cfg)
    lam_primary_r2 = float(all_edge_r2_cfg["loss_weight"])
    if lam_primary <= 0.0 and lam_primary_r2 <= 0.0:
        use_feat = False

    z = preds["y_edge_pred"].sum() * 0.0
    cols = list(edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
    stream_weight_summary: dict[str, float] = {}
    stream_weight_diags: list[dict[str, Any]] = []
    if bool(getattr(train_cfg, "use_target_stream_loss_weighting", False)) and edge_export_meta is not None:
        ts_w, stream_weight_diags, stream_weight_summary = build_target_stream_loss_weight_tensor(
            export_meta=edge_export_meta,
            edge_target_columns=cols,
            train_cfg=train_cfg,
        )
        if ts_w is not None:
            current_w = target_masks.get("edge_stream_loss_weight")
            ts_w = ts_w.to(device=preds["y_edge_pred"].device, dtype=preds["y_edge_pred"].dtype)
            target_masks = dict(target_masks)
            target_masks["edge_stream_loss_weight"] = ts_w if current_w is None else ts_w * current_w.to(device=ts_w.device, dtype=ts_w.dtype)

    loss_edge_all = z
    if use_edge:
        loss_edge_all, _ = masked_edge_regression_loss(
            preds["y_edge_pred"],
            targets["edge_stream"],
            target_masks["edge_stream"],
            loss_type=lt,
            element_weight=target_masks.get("edge_stream_loss_weight"),
        )

    loss_primary_frac = z
    loss_all_edge_r2 = z
    loss_amount = z
    primary_info: dict[str, float | int] = {}
    all_r2_info: dict[str, Any] = {}
    cols = list(edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
    primary_info.update(stream_weight_summary)
    primary_info["loss_primary_frac_deprecated"] = 1.0
    primary_info["loss_primary_frac_contributes_to_total"] = 0.0
    primary_info["amount_metrics_are_primary"] = 0.0
    primary_info["target_stream_loss_weighting_enabled"] = 1.0 if bool(getattr(train_cfg, "use_target_stream_loss_weighting", False)) else 0.0
    primary_info["target_stream_edges_skipped"] = float(sum(1 for d in stream_weight_diags if not bool(d.get("matched"))))

    if lam_primary_r2 > 0.0:
        loss_all_edge_r2, all_r2_info = compute_all_edge_r2_loss(
            y_pred=preds["y_edge_pred"],
            y_true=targets["edge_stream"],
            y_mask=target_masks["edge_stream"],
            train_cfg=train_cfg,
            element_weight=None,
        )
    if edge_export_meta is not None:
        if use_feat:
            loss_primary_frac, _target_subset_r2_unused, target_info = compute_primary_frac_losses(
                y_pred=preds["y_edge_pred"],
                y_true=targets["edge_stream"],
                y_mask=target_masks["edge_stream"],
                export_meta=edge_export_meta,
                edge_target_columns=cols,
                train_cfg=train_cfg,
                loss_type=lt,
                y_edge_mean=y_edge_mean,
                y_edge_std=y_edge_std,
                compute_r2_loss=False,
            )
            primary_info.update(target_info)
        loss_amount, _ = compute_v4_target_row_amount_loss(
            y_pred=preds["y_edge_pred"],
            y_true=targets["edge_stream"],
            y_mask=target_masks["edge_stream"],
            export_meta=edge_export_meta,
            edge_target_columns=cols,
            train_cfg=train_cfg,
            loss_type=lt,
        )

    if all_r2_info:
        for key in (
            "r2_loss_num_elements",
            "edge_all_loss_num_elements",
            "r2_loss_valid_feature_count",
            "primary_frac_r2_valid_feature_count",
            "all_edge_r2_min_count",
            "all_edge_r2_sst_threshold",
            "all_edge_r2_loss_cap",
        ):
            primary_info[key] = all_r2_info.get(key, 0.0)
    primary_info["all_edge_r2_loss_weight"] = float(all_edge_r2_cfg["loss_weight"])
    primary_info["all_edge_r2_min_count"] = int(all_edge_r2_cfg["min_count"])
    primary_info["all_edge_r2_sst_threshold"] = float(all_edge_r2_cfg["sst_threshold"])
    primary_info["all_edge_r2_loss_cap"] = float(all_edge_r2_cfg["loss_cap"])

    weighted_edge_all = loss_edge_all
    weighted_primary_frac = lam_primary * loss_primary_frac
    weighted_all_edge_r2 = lam_primary_r2 * loss_all_edge_r2
    weighted_amount_auxiliary_disabled = lam_amount * loss_amount
    loss_total_final = weighted_edge_all + weighted_primary_frac + weighted_all_edge_r2
    loss_extra_unexpected = loss_total_final - (weighted_edge_all + weighted_primary_frac + weighted_all_edge_r2)

    def _scalar(name: str) -> torch.Tensor:
        return torch.as_tensor(primary_info.get(name, 0.0), device=z.device, dtype=z.dtype)

    return {
        "loss_edge_all": loss_edge_all,
        "loss_edge": loss_edge_all,
        "loss_primary_frac": loss_primary_frac,
        "loss_all_edge_r2": loss_all_edge_r2,
        "r2_loss_all": loss_all_edge_r2,
        "loss_primary_frac_r2": loss_all_edge_r2,
        "loss_target_frac_feature": loss_primary_frac,
        "loss_target_row_frac": loss_primary_frac,
        "loss_target_row_amount": loss_amount,
        "loss_amount_auxiliary": loss_amount,
        "loss_v4_targets_frac": loss_primary_frac,
        "loss_v4_targets_amount": loss_amount,
        "loss_v4_targets": loss_amount,
        "weighted_edge_all": weighted_edge_all,
        "weighted_primary_frac": weighted_primary_frac,
        "weighted_all_edge_r2": weighted_all_edge_r2,
        "weighted_amount_auxiliary_disabled": weighted_amount_auxiliary_disabled,
        "loss_extra_unexpected": loss_extra_unexpected,
        "loss_total_final": loss_total_final,
        "target_stream_loss_weighting_enabled": _scalar("target_stream_loss_weighting_enabled"),
        "num_target_stream_edges_weighted": _scalar("num_target_stream_edges_weighted"),
        "num_target_stream_edges_with_yaml_override": _scalar("num_target_stream_edges_with_yaml_override"),
        "num_target_stream_rows_skipped": _scalar("num_target_stream_rows_skipped"),
        "target_stream_edges_skipped": _scalar("target_stream_edges_skipped"),
        "target_stream_loss_weight": _scalar("target_stream_loss_weight"),
        "loss_primary_frac_deprecated": _scalar("loss_primary_frac_deprecated"),
        "loss_primary_frac_contributes_to_total": _scalar("loss_primary_frac_contributes_to_total"),
        "amount_metrics_are_primary": _scalar("amount_metrics_are_primary"),
        "primary_frac_count": _scalar("primary_frac_count"),
        "all_edge_r2_loss_weight": _scalar("all_edge_r2_loss_weight"),
        "all_edge_r2_min_count": _scalar("all_edge_r2_min_count"),
        "all_edge_r2_sst_threshold": _scalar("all_edge_r2_sst_threshold"),
        "all_edge_r2_loss_cap": _scalar("all_edge_r2_loss_cap"),
        "primary_frac_r2_valid_feature_count": _scalar("primary_frac_r2_valid_feature_count"),
        "r2_loss_num_elements": _scalar("r2_loss_num_elements"),
        "target_loss_num_elements": _scalar("target_loss_num_elements"),
        "edge_all_loss_num_elements": _scalar("edge_all_loss_num_elements"),
        "r2_loss_valid_feature_count": _scalar("r2_loss_valid_feature_count"),
        "primary_frac_pred_std_norm": _scalar("primary_frac_pred_std_norm"),
        "primary_frac_true_std_norm": _scalar("primary_frac_true_std_norm"),
        "primary_frac_std_ratio_norm": _scalar("primary_frac_std_ratio_norm"),
        "primary_frac_pred_std_orig": _scalar("primary_frac_pred_std_orig"),
        "primary_frac_true_std_orig": _scalar("primary_frac_true_std_orig"),
        "primary_frac_std_ratio_orig": _scalar("primary_frac_std_ratio_orig"),
        "loss_total": loss_total_final,
    }


def compute_training_loss(
    preds: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    target_masks: Dict[str, torch.Tensor],
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    *,
    edge_export_meta=None,
    edge_target_columns: Sequence[str] | None = None,
    edge_index: torch.Tensor | None = None,
    y_edge_mean: torch.Tensor | None = None,
    y_edge_std: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    if getattr(data_cfg, "task_mode", "multitask") == "edge_all":
        return compute_edge_all_training_loss(
            preds,
            targets,
            target_masks,
            train_cfg,
            data_cfg,
            edge_export_meta=edge_export_meta,
            edge_target_columns=edge_target_columns,
            edge_index=edge_index,
            y_edge_mean=y_edge_mean,
            y_edge_std=y_edge_std,
        )
    if decoder_tasks_enabled(data_cfg):
        return compute_multitask_decoder_loss(preds, targets, target_masks, train_cfg, data_cfg)
    if data_cfg.aux_task.enabled:
        return compute_three_slot_loss(
            preds,
            targets,
            target_masks,
            train_cfg,
            aux_masked=bool(data_cfg.aux_task.masked and data_cfg.aux_task.task_name == "heat_duty"),
        )
    return compute_fixed_only_loss(
        preds,
        targets,
        train_cfg,
        target_only=getattr(data_cfg, "task_mode", "multitask") == "target_only",
        target_masks=target_masks,
    )


def _edge_all_normalize(
    y: torch.Tensor,
    mean: torch.Tensor | None,
    std: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    if mean is None or std is None:
        return y
    m = mean.to(device=device, dtype=y.dtype).view(1, -1)
    s = std.to(device=device, dtype=y.dtype).view(1, -1)
    return (y - m) / s


@torch.no_grad()
def evaluate_edge_all_epoch(
    model,
    loader,
    device,
    train_cfg,
    data_cfg,
    use_amp: bool,
    *,
    y_edge_mean: torch.Tensor | None,
    y_edge_std: torch.Tensor | None,
    edge_struct_dim: int,
    target_stream_targets_path: str | None = None,
    compute_target_v4_scalars: bool = True,
) -> dict[str, float]:
    from pathlib import Path

    from .target_edge_10d_metrics import TargetEdge10DAccumulator, main_stream_property_names
    from .target_stream_weighting import TargetStreamFeatureAccumulator
    from .target_v4_metrics import TargetV4EpochAccumulator, build_target_v4_epoch_accumulator

    _was_training_edge_val = model.training
    model.eval()
    v4_acc: TargetV4EpochAccumulator | None = None
    if compute_target_v4_scalars and target_stream_targets_path:
        v4_path = Path(target_stream_targets_path)
        if v4_path.is_file():
            v4_acc = build_target_v4_epoch_accumulator(v4_path)
    ts_acc: TargetStreamFeatureAccumulator | None = None
    if target_stream_targets_path and bool(getattr(train_cfg, "use_target_stream_loss_weighting", False)):
        ts_acc = TargetStreamFeatureAccumulator(train_cfg=train_cfg, target_stream_targets_path=target_stream_targets_path)
    te10_acc: TargetEdge10DAccumulator | None = None
    if target_stream_targets_path:
        te10_acc = TargetEdge10DAccumulator(train_cfg=train_cfg, target_stream_targets_path=target_stream_targets_path)

    steps = 0
    loss_total_sum = 0.0
    loss_edge_sum = 0.0
    loss_frac_sum = 0.0
    loss_r2_sum = 0.0
    loss_amount_sum = 0.0
    weighted_edge_sum = 0.0
    weighted_frac_sum = 0.0
    weighted_r2_sum = 0.0
    weighted_amount_disabled_sum = 0.0
    loss_extra_unexpected_sum = 0.0
    diag_sum: dict[str, float] = defaultdict(float)
    mae_norm_sum = 0.0
    sse_norm_sum = 0.0
    n_norm = 0.0
    h2_sae_o = h2_sse_o = h2_n = 0.0
    co2_sae_o = co2_sse_o = co2_n = 0.0
    h2_sum_y = h2_sum_y2 = 0.0
    co2_sum_y = co2_sum_y2 = 0.0

    for batch in loader:
        batch_data = {k: v.to(device) for k, v in batch.model_kwargs.items()}
        targets = {k: v.to(device) for k, v in batch.targets.items()}
        target_masks = {k: v.to(device) for k, v in batch.target_masks.items()}
        task_inputs = {
            head: {k: v.to(device) for k, v in payload.items()}
            for head, payload in batch.task_inputs.items()
        }
        y_raw = targets["edge_stream"]
        y_n = _edge_all_normalize(y_raw, y_edge_mean, y_edge_std, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = model(batch_data, task_inputs=task_inputs)
            loss_items = compute_edge_all_training_loss(
                preds,
                {"edge_stream": y_n},
                target_masks,
                train_cfg,
                data_cfg,
                edge_export_meta=batch.edge_export_meta,
                edge_target_columns=list(batch.edge_target_columns or []),
                edge_index=batch_data.get("edge_index"),
                y_edge_mean=y_edge_mean,
                y_edge_std=y_edge_std,
            )

        loss_total_sum += float(loss_items["loss_total"].detach().cpu())
        loss_edge_sum += float(loss_items.get("loss_edge_all", loss_items["loss_edge"]).detach().cpu())
        loss_frac_sum += float(
            loss_items.get("loss_target_frac_feature", loss_items["loss_target_row_frac"]).detach().cpu()
        )
        loss_r2_sum += float(
            loss_items.get("loss_all_edge_r2", loss_items.get("loss_primary_frac_r2", loss_items["loss_total"] * 0))
            .detach()
            .cpu()
        )
        loss_amount_sum += float(loss_items["loss_target_row_amount"].detach().cpu())
        weighted_edge_sum += float(loss_items["weighted_edge_all"].detach().cpu())
        weighted_frac_sum += float(loss_items["weighted_primary_frac"].detach().cpu())
        weighted_r2_sum += float(loss_items["weighted_all_edge_r2"].detach().cpu())
        weighted_amount_disabled_sum += float(loss_items["weighted_amount_auxiliary_disabled"].detach().cpu())
        loss_extra_unexpected_sum += float(loss_items["loss_extra_unexpected"].detach().cpu())
        for dk in (
            "primary_frac_count",
            "primary_frac_r2_valid_feature_count",
            "primary_frac_pred_std_norm",
            "primary_frac_true_std_norm",
            "primary_frac_std_ratio_norm",
            "primary_frac_pred_std_orig",
            "primary_frac_true_std_orig",
            "primary_frac_std_ratio_orig",
            "r2_loss_num_elements",
            "target_loss_num_elements",
            "edge_all_loss_num_elements",
            "r2_loss_valid_feature_count",
            "all_edge_r2_loss_weight",
            "all_edge_r2_min_count",
            "all_edge_r2_sst_threshold",
            "all_edge_r2_loss_cap",
            "target_stream_loss_weighting_enabled",
            "num_target_stream_edges_weighted",
            "num_target_stream_edges_with_yaml_override",
            "num_target_stream_rows_skipped",
            "target_stream_edges_skipped",
            "target_stream_loss_weight",
            "loss_primary_frac_deprecated",
            "loss_primary_frac_contributes_to_total",
            "amount_metrics_are_primary",
        ):
            if dk in loss_items:
                diag_sum[dk] += float(loss_items[dk].detach().cpu())

        pred = preds["y_edge_pred"]
        mask = target_masks["edge_stream"]
        mae_n = masked_edge_mae(pred, y_n, mask)
        mae_norm_sum += float(mae_n.detach().cpu())
        m = mask.to(dtype=pred.dtype, device=pred.device)
        while m.ndim < pred.ndim:
            m = m.unsqueeze(-1)
        m = m.expand_as(pred)
        diff = (pred - y_n) * m
        sse_norm_sum += float((diff**2).sum().detach().cpu())
        n_norm += float(m.sum().detach().cpu())

        edge_struct = batch_data.get("edge_struct_attr")
        if edge_struct is None:
            edge_struct = batch_data.get("edge_oper")
        validate_edge_all_batch(
            y_edge_pred=pred,
            y_edge_true=y_n,
            y_edge_mask=mask,
            edge_struct=edge_struct,
            edge_struct_dim=edge_struct_dim,
            answer_edge_pos=batch.answer_edge_pos,
        )

        if v4_acc is not None and batch.edge_export_meta is not None:
            if y_edge_mean is not None and y_edge_std is not None:
                mean = y_edge_mean.to(device=device, dtype=pred.dtype).view(1, -1)
                std = y_edge_std.to(device=device, dtype=pred.dtype).view(1, -1)
                y_pred_orig = pred * std + mean
            else:
                y_pred_orig = pred
            v4_acc.update_batch(
                export=batch.edge_export_meta,
                y_pred_orig=y_pred_orig,
                y_true_orig=y_raw,
                y_edge_mask=mask,
                edge_target_columns=list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
                split_name="val",
            )
            if ts_acc is not None:
                ts_acc.update_batch(
                    export=batch.edge_export_meta,
                    y_pred_orig=y_pred_orig,
                    y_true_orig=y_raw,
                    y_edge_mask=mask,
                    edge_target_columns=list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
                    split_name="val",
                )
            if te10_acc is not None:
                cols = list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
                prop_names = main_stream_property_names(data_cfg)
                col_map = {str(c): i for i, c in enumerate(cols)}
                idx = torch.tensor([col_map[p] for p in prop_names], device=y_pred_orig.device, dtype=torch.long)
                pred_main = y_pred_orig.index_select(dim=-1, index=idx)
                true_main = y_raw.index_select(dim=-1, index=idx)
                m = mask.to(device=true_main.device, dtype=true_main.dtype)
                while m.ndim < true_main.ndim:
                    m = m.unsqueeze(-1)
                if m.shape[-1] == 1:
                    m = m.expand_as(true_main)
                te10_acc.update_batch(
                    export=batch.edge_export_meta,
                    pred_main=pred_main,
                    true_main=true_main,
                    mask_main=m,
                    property_names=prop_names,
                    split_name="val",
                )

        cols = list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
        h2_i, co2_i = resolve_frac_h2_co2_column_indices(cols)
        if batch.answer_edge_pos:
            for pos_map in batch.answer_edge_pos:
                if "target" in pos_map:
                    r = int(pos_map["target"])
                    ph = pred[r, h2_i]
                    th_o = y_raw[r, h2_i]
                    ph_o = ph * y_edge_std[h2_i] + y_edge_mean[h2_i] if y_edge_mean is not None and y_edge_std is not None else ph
                    d = ph_o - th_o
                    yv = float(th_o.detach().cpu())
                    h2_sum_y += yv
                    h2_sum_y2 += yv * yv
                    h2_sae_o += float(d.abs().detach().cpu())
                    h2_sse_o += float((d**2).detach().cpu())
                    h2_n += 1.0
                if "tailgas" in pos_map:
                    r = int(pos_map["tailgas"])
                    pc = pred[r, co2_i]
                    tc_o = y_raw[r, co2_i]
                    pc_o = pc * y_edge_std[co2_i] + y_edge_mean[co2_i] if y_edge_mean is not None and y_edge_std is not None else pc
                    d = pc_o - tc_o
                    yv = float(tc_o.detach().cpu())
                    co2_sum_y += yv
                    co2_sum_y2 += yv * yv
                    co2_sae_o += float(d.abs().detach().cpu())
                    co2_sse_o += float((d**2).detach().cpu())
                    co2_n += 1.0
        steps += 1

    if steps == 0:
        z = 0.0
        model.train(_was_training_edge_val)
        return {
            "loss_total": z,
            "loss_total_final": z,
            "loss_edge_all": z,
            "loss_edge": z,
            "loss_primary_frac": z,
            "loss_all_edge_r2": z,
            "r2_loss_all": z,
            "loss_primary_frac_r2": z,
            "loss_target_frac_feature": z,
            "loss_target_row_frac": z,
            "loss_target_row_amount": z,
            "loss_amount_auxiliary": z,
            "loss_v4_targets_frac": z,
            "loss_v4_targets_amount": z,
            "loss_v4_targets": z,
            "weighted_edge_all": z,
            "weighted_primary_frac": z,
            "weighted_all_edge_r2": z,
            "weighted_amount_auxiliary_disabled": z,
            "loss_extra_unexpected": z,
            "loss_target": z,
            "loss_tailgas": z,
            "edge_all_mse": z,
            "edge_all_mae": z,
            "target_h2_mae": z,
            "target_h2_rmse": z,
            "tailgas_co2_mae": z,
            "tailgas_co2_rmse": z,
            "metric_target_r2": z,
            "metric_tailgas_r2": z,
            "primary_frac_count": z,
            "primary_frac_r2_valid_feature_count": z,
            "r2_loss_num_elements": z,
            "target_loss_num_elements": z,
            "edge_all_loss_num_elements": z,
            "r2_loss_valid_feature_count": z,
        }

    out: dict[str, float] = {
        "loss_total": loss_total_sum / steps,
        "loss_total_final": loss_total_sum / steps,
        "loss_edge_all": loss_edge_sum / steps,
        "loss_edge": loss_edge_sum / steps,
        "loss_primary_frac": loss_frac_sum / steps,
        "loss_all_edge_r2": loss_r2_sum / steps,
        "r2_loss_all": loss_r2_sum / steps,
        "loss_primary_frac_r2": loss_r2_sum / steps,
        "loss_target_frac_feature": loss_frac_sum / steps,
        "loss_target_row_frac": loss_frac_sum / steps,
        "loss_target_row_amount": loss_amount_sum / steps,
        "loss_amount_auxiliary": loss_amount_sum / steps,
        "loss_v4_targets_frac": loss_frac_sum / steps,
        "loss_v4_targets_amount": loss_amount_sum / steps,
        "loss_v4_targets": loss_amount_sum / steps,
        "weighted_edge_all": weighted_edge_sum / steps,
        "weighted_primary_frac": weighted_frac_sum / steps,
        "weighted_all_edge_r2": weighted_r2_sum / steps,
        "weighted_amount_auxiliary_disabled": weighted_amount_disabled_sum / steps,
        "loss_extra_unexpected": loss_extra_unexpected_sum / steps,
        "loss_target": 0.0,
        "loss_tailgas": 0.0,
        "edge_all_mae": mae_norm_sum / steps,
        "edge_all_mse": (sse_norm_sum / n_norm) if n_norm > 0 else 0.0,
    }
    count_keys = {"primary_frac_count", "r2_loss_num_elements", "target_loss_num_elements", "edge_all_loss_num_elements"}
    for dk, val in diag_sum.items():
        out[dk] = val if dk in count_keys else val / steps

    def _r2(sse: float, sum_y: float, sum_y2: float, n: float) -> float:
        if n < 2.0:
            return 1.0 if sse <= 1e-18 and n > 0.0 else 0.0
        sst = sum_y2 - (sum_y * sum_y) / n
        if abs(sst) <= 1e-12:
            return 1.0 if sse <= 1e-18 else 0.0
        return 1.0 - sse / sst

    out["metric_target_h2_mae"] = (h2_sae_o / h2_n) if h2_n > 0 else 0.0
    out["metric_target_h2_rmse"] = (h2_sse_o / h2_n) ** 0.5 if h2_n > 0 else 0.0
    out["metric_tailgas_co2_mae"] = (co2_sae_o / co2_n) if co2_n > 0 else 0.0
    out["metric_tailgas_co2_rmse"] = (co2_sse_o / co2_n) ** 0.5 if co2_n > 0 else 0.0
    out["metric_target_r2"] = _r2(h2_sse_o, h2_sum_y, h2_sum_y2, h2_n)
    out["metric_tailgas_r2"] = _r2(co2_sse_o, co2_sum_y, co2_sum_y2, co2_n)
    out["target_h2_mae"] = out["metric_target_h2_mae"]
    out["target_h2_rmse"] = out["metric_target_h2_rmse"]
    out["target_h2_r2"] = out["metric_target_r2"]
    out["tailgas_co2_mae"] = out["metric_tailgas_co2_mae"]
    out["tailgas_co2_rmse"] = out["metric_tailgas_co2_rmse"]
    out["tailgas_co2_r2"] = out["metric_tailgas_r2"]
    out["diagnostic_target_species_frac_h2_mae"] = out["metric_target_h2_mae"]
    out["diagnostic_target_species_frac_h2_rmse"] = out["metric_target_h2_rmse"]
    out["diagnostic_target_species_frac_h2_r2"] = out["metric_target_r2"]
    out["diagnostic_target_species_frac_co2_mae"] = out["metric_tailgas_co2_mae"]
    out["diagnostic_target_species_frac_co2_rmse"] = out["metric_tailgas_co2_rmse"]
    out["diagnostic_target_species_frac_co2_r2"] = out["metric_tailgas_r2"]

    if v4_acc is not None:
        try:
            out.update(v4_acc.finalize_scalars())
        except Exception:
            pass
    if ts_acc is not None:
        try:
            out.update(ts_acc.finalize_scalars())
        except Exception:
            pass
    if te10_acc is not None:
        try:
            out.update(te10_acc.finalize_scalars())
        except Exception:
            pass
    model.train(_was_training_edge_val)
    return out
