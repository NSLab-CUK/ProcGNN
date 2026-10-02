from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from ..constants import STREAM_EDGE_FEATURE_SLOTS
from .target_stream_weighting import build_target_stream_loss_weight_tensor


@dataclass
class EdgeStepBatchResult:
    loss: torch.Tensor
    loss_items: dict[str, torch.Tensor]
    debug_rows: list[dict[str, Any]]


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _temporary_target_stream_cfg(train_cfg: Any, *, target_weight: float) -> Any:
    class _Cfg:
        pass

    cfg = _Cfg()
    for name in dir(train_cfg):
        if name.startswith("_"):
            continue
        try:
            setattr(cfg, name, getattr(train_cfg, name))
        except Exception:
            pass
    if isinstance(train_cfg, Mapping):
        for key, value in train_cfg.items():
            setattr(cfg, str(key), value)
    setattr(cfg, "use_target_stream_loss_weighting", True)
    setattr(cfg, "target_stream_loss_weight", float(target_weight))
    setattr(cfg, "edge_stream_loss_weight", float(target_weight))
    return cfg


def _edge_groups_from_export_meta(edge_export_meta: Any, n_edges: int) -> list[tuple[str, list[int]]]:
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
        groups.setdefault((str(pid), str(eid)), []).append(int(i))
    return [(f"{pid}:{eid}", idxs) for (pid, eid), idxs in groups.items()]


def build_target_edge_boolean_mask(
    *,
    train_cfg: Any,
    edge_export_meta: Any,
    edge_target_columns: Sequence[str] | None,
    n_edges: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return target-edge identity independent of supervision weight values."""

    mask = torch.zeros((int(n_edges),), dtype=torch.bool, device=device)
    source = "none"
    summary: Mapping[str, Any] = {}
    if edge_export_meta is not None:
        cols = list(edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
        ts_cfg = _temporary_target_stream_cfg(train_cfg, target_weight=2.0)
        w2d, _, summary = build_target_stream_loss_weight_tensor(
            export_meta=edge_export_meta,
            edge_target_columns=cols,
            train_cfg=ts_cfg,
        )
        if w2d is not None:
            edge_w = w2d.max(dim=1).values.to(device=device, dtype=torch.float32)
            n = min(int(n_edges), int(edge_w.numel()))
            if n > 0:
                mask[:n] = edge_w[:n] > 1.0
            source = "target_stream_mapping"
        if source == "none":
            answer_names = list(getattr(edge_export_meta, "answer_task_names", []) or [])
            if answer_names:
                n = min(int(n_edges), len(answer_names))
                flags = torch.tensor(
                    [bool(str(x).strip()) for x in answer_names[:n]],
                    dtype=torch.bool,
                    device=device,
                )
                if bool(flags.any()):
                    mask[:n][flags] = True
                    source = "answer_task_names_fallback"
    return mask, {
        "target_mask_source": source,
        "target_edge_count": int(mask.sum().item()),
        "default_edge_count": int((~mask).sum().item()),
        "target_stream_summary": dict(summary or {}),
    }


def build_edge_step_edge_weight_vector(
    *,
    train_cfg: Any,
    edge_export_meta: Any,
    edge_target_columns: Sequence[str] | None,
    n_edges: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    default_weight = float(_cfg_get(train_cfg, "edge_weight_default", 1.0))
    target_weight = float(_cfg_get(train_cfg, "edge_weight_target_edge", 5.0))
    weights = torch.full((int(n_edges),), default_weight, dtype=torch.float32, device=device)
    target_mask, mask_info = build_target_edge_boolean_mask(
        train_cfg=train_cfg,
        edge_export_meta=edge_export_meta,
        edge_target_columns=edge_target_columns,
        n_edges=n_edges,
        device=device,
    )
    if bool(target_mask.any()):
        weights[target_mask] = target_weight
    default_mask = ~target_mask
    return weights, {
        "weight_source": str(mask_info.get("target_mask_source", "none")),
        "target_edge_count": int(target_mask.sum().item()),
        "default_edge_count": int(default_mask.sum().item()),
        "default_weight": default_weight,
        "target_weight": target_weight,
        "target_edge_weight_mean": float(weights[target_mask].mean().item()) if bool(target_mask.any()) else 0.0,
        "default_edge_weight_mean": float(weights[default_mask].mean().item()) if bool(default_mask.any()) else default_weight,
        "target_stream_summary": dict(mask_info.get("target_stream_summary", {}) or {}),
    }


def _select_edge_rows(tensor: torch.Tensor, edge_indices: Sequence[int]) -> torch.Tensor:
    idx = torch.tensor([int(i) for i in edge_indices], device=tensor.device, dtype=torch.long)
    return tensor.index_select(0, idx)


def _edge_loss_elementwise(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    loss_type = str(loss_type or "smooth_l1").strip().lower()
    if loss_type in {"smooth_l1", "huber"}:
        return F.smooth_l1_loss(pred, target, reduction="none")
    if loss_type == "l1":
        return F.l1_loss(pred, target, reduction="none")
    if loss_type == "mse":
        return F.mse_loss(pred, target, reduction="none")
    raise ValueError(f"Unsupported edge_step loss_type_edge_all={loss_type!r}.")


def compute_single_edge_loss(
    *,
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    target_masks: Mapping[str, torch.Tensor],
    edge_id: int,
    train_cfg: Any,
    edge_weight_vector: torch.Tensor | None = None,
    edge_indices: Sequence[int] | None = None,
    debug: bool = False,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    pred_all = outputs.get("y_edge_pred")
    if pred_all is None:
        pred_all = outputs.get("main_stream_pred")
    if pred_all is None:
        raise KeyError("edge_step requires outputs['y_edge_pred'] or outputs['main_stream_pred'].")
    true_all = targets["edge_stream"]
    mask_all = target_masks["edge_stream"]
    idxs = [int(edge_id)] if edge_indices is None else [int(i) for i in edge_indices]
    pred_e = _select_edge_rows(pred_all, idxs)
    true_e = _select_edge_rows(true_all, idxs).to(device=pred_e.device, dtype=pred_e.dtype)
    mask_e = _select_edge_rows(mask_all, idxs).to(device=pred_e.device, dtype=pred_e.dtype)
    mask_sum = mask_e.sum()
    log: dict[str, Any] = {
        "edge_id": int(edge_id),
        "edge_indices": ",".join(str(i) for i in idxs),
        "pred_e_shape": tuple(pred_e.shape),
        "mask_sum": float(mask_sum.detach().cpu().item()),
    }
    if not bool(mask_sum.detach().cpu().item() > 0.0):
        log["skipped_edge_reason"] = "mask_sum_zero"
        return None, log
    loss_type = str(_cfg_get(train_cfg, "loss_type_edge_all", "smooth_l1"))
    elem = _edge_loss_elementwise(pred_e, true_e, loss_type)
    sample_loss = (elem * mask_e).sum() / mask_sum.clamp_min(torch.finfo(pred_e.dtype).eps)
    if edge_weight_vector is None:
        edge_weight = pred_e.new_tensor(1.0)
    elif edge_weight_vector.ndim == 1:
        edge_weight = _select_edge_rows(edge_weight_vector.to(device=pred_e.device, dtype=pred_e.dtype), idxs).mean()
    else:
        edge_weight = _select_edge_rows(edge_weight_vector.to(device=pred_e.device, dtype=pred_e.dtype), idxs).mean()
    weighted_loss = sample_loss * edge_weight
    default_weight = float(_cfg_get(train_cfg, "edge_weight_default", 1.0))
    log.update(
        {
            "loss_before_edge_weight": float(sample_loss.detach().cpu().item()),
            "edge_weight_mean": float(edge_weight.detach().cpu().item()),
            "is_target_edge": bool(float(edge_weight.detach().cpu().item()) != default_weight),
            "loss_after_edge_weight": float(weighted_loss.detach().cpu().item()),
        }
    )
    return weighted_loss, log


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
) -> float:
    if bool(use_amp) and scaler is not None and bool(getattr(scaler, "is_enabled", lambda: False)()):
        scaler.scale(loss).backward()
        if grad_clip is not None and float(grad_clip) > 0.0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        else:
            grad_norm = torch.tensor(0.0, device=loss.device)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if grad_clip is not None and float(grad_clip) > 0.0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        else:
            grad_norm = torch.tensor(0.0, device=loss.device)
        optimizer.step()
    if scheduler is not None and str(_cfg_get(train_cfg, "scheduler_step_unit", "epoch")) == "optimizer_step":
        scheduler.step()
    try:
        return float(grad_norm.detach().cpu().item())
    except AttributeError:
        return float(grad_norm)


def train_one_batch_edge_step(
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
    device: torch.device,
    use_amp: bool,
    grad_clip: float | None,
    edge_export_meta: Any,
    edge_target_columns: Sequence[str] | None,
    rng: random.Random | None = None,
    debug: bool = False,
) -> EdgeStepBatchResult:
    n_edges = int(targets["edge_stream"].shape[0])
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
    losses: list[float] = []
    debug_rows: list[dict[str, Any]] = []
    skipped = 0
    updated = 0
    forward_count = 0
    grad_norm_last = 0.0
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    for group_label, edge_indices in edge_groups:
        edge_id = int(edge_indices[0])
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=amp_device, enabled=bool(use_amp)):
            outputs = model(batch_data, task_inputs=task_inputs)
            forward_count += 1
            loss_e, log = compute_single_edge_loss(
                outputs=outputs,
                targets=targets,
                target_masks=target_masks,
                edge_id=edge_id,
                edge_indices=edge_indices,
                train_cfg=train_cfg,
                edge_weight_vector=edge_weight_vector,
                debug=debug,
            )
        if loss_e is None:
            skipped += 1
            debug_rows.append(dict(log, edge_group=group_label, edge_group_size=len(edge_indices), updated_or_skipped="skipped"))
            continue
        grad_norm_last = _backward_clip_optimizer_step(
            loss=loss_e,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=grad_clip,
            train_cfg=train_cfg,
        )
        updated += 1
        losses.append(float(loss_e.detach().cpu().item()))
        debug_rows.append(
            dict(
                log,
                edge_group=group_label,
                edge_group_size=len(edge_indices),
                grad_norm=grad_norm_last,
                updated_or_skipped="updated",
            )
        )
    mean_loss = sum(losses) / len(losses) if losses else 0.0
    loss_tensor = torch.tensor(float(mean_loss), dtype=torch.float32, device=device)
    loss_items = {
        "edge_step_loss_mean": loss_tensor,
        "edge_update_count": torch.tensor(float(updated), dtype=torch.float32, device=device),
        "optimizer_step_count": torch.tensor(float(updated), dtype=torch.float32, device=device),
        "skipped_edge_count": torch.tensor(float(skipped), dtype=torch.float32, device=device),
        "edge_step_forward_count": torch.tensor(float(forward_count), dtype=torch.float32, device=device),
        "edge_step_target_edge_count": torch.tensor(float(weight_info.get("target_edge_count", 0)), dtype=torch.float32, device=device),
        "edge_step_default_edge_count": torch.tensor(float(weight_info.get("default_edge_count", 0)), dtype=torch.float32, device=device),
        "grad_norm": torch.tensor(float(grad_norm_last), dtype=torch.float32, device=device),
    }
    return EdgeStepBatchResult(loss=loss_tensor, loss_items=loss_items, debug_rows=debug_rows)
