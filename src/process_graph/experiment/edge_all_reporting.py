from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import pandas as pd
import torch

from ..constants import STREAM_EDGE_FEATURE_SLOTS
from ..data.tabular_dataset import V3_SLOT_TO_TASK_NAME
from .edge_all_answer_columns import resolve_answer_metric_spec, resolve_answer_target_column_index
from .schema import DataConfig, TrainConfig
from .target_v4_metrics import (
    augment_metrics_with_legacy_aliases,
    compute_target_v4_metrics_from_original_scale,
    load_target_stream_targets_v4,
    write_target_v4_split_artifacts,
)
from .train_utils import (
    _edge_all_normalize,
    compute_edge_all_training_loss,
    resolve_answer_edge_species_weights,
    validate_edge_all_batch,
)


# Long-form edge×property table: keep only columns needed for v4 / baselines / offline scripts.
# (Full per-cell errors live only in RAM during evaluate_edge_all_detailed.)
EDGE_PREDICTIONS_EXPORT_COLUMNS: tuple[str, ...] = (
    "split",
    "process_id",
    "sample_id",
    "canonical_edge_id",
    "main_data_stream_key",
    "stream_role",
    "is_input_edge",
    "is_output_edge",
    "is_internal_edge",
    "property_name",
    "y_true_norm",
    "y_pred_norm",
    "y_true_orig",
    "y_pred_orig",
    "y_edge_mask",
    "is_answer_edge",
    "answer_task_name",
)


def edge_predictions_export_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Subset columns for CSV export (smaller files than the in-memory evaluation frame)."""
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()
    cols = [c for c in EDGE_PREDICTIONS_EXPORT_COLUMNS if c in df.columns]
    if len(cols) < 8:
        return df.copy()
    return df.loc[:, cols].copy()


@dataclass
class EdgeAllEvalArtifacts:
    metrics: dict[str, float]
    edge_predictions: pd.DataFrame
    metrics_by_process: pd.DataFrame
    metrics_by_property: pd.DataFrame
    metrics_by_stream_role: pd.DataFrame
    answer_edge_metrics: pd.DataFrame
    mapping_logs: list[dict[str, Any]]


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den and abs(den) > 0.0 else 0.0


def _rmse(sse: float, n: float) -> float:
    return math.sqrt(sse / n) if n > 0 else 0.0


def _r2(y_true: pd.Series, y_pred: pd.Series, eps: float = 1e-12) -> float:
    """R²; NaN 없이 유한값. ``train_utils.evaluate_edge_all_epoch`` 의 상대 SST 바닥과 동일 계열."""
    rel_floor = 1e-4
    n = int(len(y_true))
    if n < 1:
        return 0.0
    if n < 2:
        ss_res = float(((y_true - y_pred) ** 2).sum())
        if ss_res <= 1e-18:
            return 1.0
        y0 = float(y_true.iloc[0])
        ymax = max(abs(y0), 1e-30)
        return max(0.0, min(1.0, 1.0 - ss_res / (ymax * ymax + 1e-30)))
    ss_res = float(((y_true - y_pred) ** 2).sum())
    y_mean = float(y_true.mean())
    ss_tot = float(((y_true - y_mean) ** 2).sum())
    scale = max(float((y_true.astype(float) ** 2).mean()), y_mean * y_mean, 1e-30)
    if not math.isfinite(ss_tot):
        ss_tot = 0.0
    ss_tot_eff = max(ss_tot, rel_floor * scale, 1e-30)
    r2 = 1.0 - (ss_res / ss_tot_eff)
    if not math.isfinite(r2):
        return 0.0
    return max(-1.0, min(1.0, r2))


def _metric_bundle(y_true: pd.Series, y_pred: pd.Series, r2_eps: float = 1e-6) -> dict[str, float | int | bool]:
    n = int(len(y_true))
    if n == 0:
        return {
            "true_mean": 0.0,
            "pred_mean": 0.0,
            "mean_bias": 0.0,
            "true_std": 0.0,
            "pred_std": 0.0,
            "std_ratio": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "n_samples": 0,
            "r2_unstable": True,
        }
    yt = y_true.astype(float)
    yp = y_pred.astype(float)
    true_mean = float(yt.mean())
    pred_mean = float(yp.mean())
    true_std = float(yt.std(ddof=0))
    pred_std = float(yp.std(ddof=0))
    err = yp - yt
    mae = float(err.abs().mean())
    rmse = math.sqrt(float((err * err).mean()))
    r2_unstable = bool(true_std < r2_eps)
    r2 = float(_r2(yt, yp))
    return {
        "true_mean": true_mean,
        "pred_mean": pred_mean,
        "mean_bias": pred_mean - true_mean,
        "true_std": true_std,
        "pred_std": pred_std,
        "std_ratio": _safe_ratio(pred_std, true_std),
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
        "n_samples": n,
        "r2_unstable": r2_unstable,
    }


def _smape(y_true: pd.Series, y_pred: pd.Series, eps: float = 1e-12) -> float:
    denom = (y_true.abs() + y_pred.abs()).clip(lower=eps)
    return float((2.0 * (y_pred - y_true).abs() / denom).mean())


def _process_num(pid: str) -> int:
    p = str(pid).strip()
    if p.lower().startswith("process"):
        return int(p[7:])
    return int(p)


def _normalize_main_data_stream_key(value: Any) -> str:
    """Align stream keys across CSV, graph export, and training (digit keys → canonical str)."""
    if value is None or pd.isna(value):
        return ""
    raw = str(value).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    return str(int(raw)) if raw.isdigit() else raw


def _build_edge_stream_loss_weight_tensor(
    *,
    export_meta,
    edge_target_columns: Sequence[str],
    train_cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, list[str]]:
    if bool(getattr(train_cfg, "use_target_stream_loss_weighting", False)):
        return None, []
    from .v4_target_edge_weighting import build_v4_target_loss_weight_tensor

    w_cpu, warn_msgs = build_v4_target_loss_weight_tensor(
        export_meta=export_meta,
        edge_target_columns=edge_target_columns,
        train_cfg=train_cfg,
    )
    if w_cpu is None:
        return None, warn_msgs
    return w_cpu.to(device=device, dtype=dtype), warn_msgs


@torch.no_grad()
def evaluate_edge_all_detailed(
    *,
    model,
    loader: Iterable,
    device: torch.device,
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    y_edge_mean: torch.Tensor | None,
    y_edge_std: torch.Tensor | None,
    edge_struct_dim: int,
    target_answer_edges_path: Path,
    split_name: str,
) -> EdgeAllEvalArtifacts:
    model.eval()
    answers = pd.read_csv(target_answer_edges_path, dtype={"process_id": int})
    weight_warn_emitted = False

    rows: list[dict[str, Any]] = []
    mapping_logs: list[dict[str, Any]] = []
    loss_sum = 0.0
    step_count = 0

    for batch in loader:
        batch_data = {k: v.to(device) for k, v in batch.model_kwargs.items()}
        targets = {k: v.to(device) for k, v in batch.targets.items()}
        target_masks = {k: v.to(device) for k, v in batch.target_masks.items()}
        task_inputs = {h: {k: v.to(device) for k, v in p.items()} for h, p in batch.task_inputs.items()}
        y_true_orig = targets["edge_stream"]
        y_true_norm = _edge_all_normalize(y_true_orig, y_edge_mean, y_edge_std, device)
        preds = model(batch_data, task_inputs=task_inputs)
        y_pred_norm = preds["y_edge_pred"]
        cols = list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
        weighted_masks = dict(target_masks)
        w_tensor, w_warn = _build_edge_stream_loss_weight_tensor(
            export_meta=batch.edge_export_meta,
            edge_target_columns=cols,
            train_cfg=train_cfg,
            device=device,
            dtype=target_masks["edge_stream"].dtype,
        )
        if w_tensor is not None:
            weighted_masks["edge_stream_loss_weight"] = w_tensor
        if w_warn and not weight_warn_emitted:
            for msg in w_warn:
                mapping_logs.append(
                    {
                        "split": split_name,
                        "process_id": "",
                        "task_name": "v4_target_weighting",
                        "target_column": "",
                        "mapped_edge_target_column": "",
                        "edge_target_column_idx": -1,
                        "mapping_rule": msg,
                    }
                )
            weight_warn_emitted = True
        li = compute_edge_all_training_loss(
            {"y_edge_pred": y_pred_norm},
            {"edge_stream": y_true_norm},
            weighted_masks,
            train_cfg,
            data_cfg,
            edge_export_meta=batch.edge_export_meta,
            edge_target_columns=cols,
            edge_index=batch_data.get("edge_index"),
        )
        loss_sum += float(li["loss_total"].detach().cpu())
        step_count += 1

        edge_struct = batch_data.get("edge_struct_attr")
        if edge_struct is None:
            edge_struct = batch_data.get("edge_oper")
        validate_edge_all_batch(
            y_edge_pred=y_pred_norm,
            y_edge_true=y_true_norm,
            y_edge_mask=target_masks["edge_stream"],
            edge_struct=edge_struct,
            edge_struct_dim=edge_struct_dim,
            answer_edge_pos=batch.answer_edge_pos,
        )

        if y_pred_norm.shape != y_true_norm.shape:
            raise RuntimeError(f"y_edge_pred shape {tuple(y_pred_norm.shape)} != y_edge_true shape {tuple(y_true_norm.shape)}")

        if y_edge_mean is not None and y_edge_std is not None:
            m = y_edge_mean.to(device=device, dtype=y_pred_norm.dtype).view(1, -1)
            s = y_edge_std.to(device=device, dtype=y_pred_norm.dtype).view(1, -1)
            y_pred_orig = y_pred_norm * s + m
        else:
            y_pred_orig = y_pred_norm

        n_edges = int(y_pred_norm.shape[0])
        export = batch.edge_export_meta
        if export is None:
            raise RuntimeError("edge_export_meta is required for edge_all export.")
        if len(export.process_id) != n_edges:
            raise RuntimeError("edge_export_meta length does not match batched edge rows.")

        # Explicit answer-column mapping per process in the batch; fail if missing.
        proc_col_map: dict[str, dict[str, tuple[int, str, str]]] = {}
        for pid in sorted(set(export.process_id)):
            pnum = _process_num(pid)
            proc_rows = answers[answers["process_id"] == pnum]
            if len(proc_rows) == 0:
                raise RuntimeError(f"target_answer_edges has no rows for process_id={pnum}")
            slot_map: dict[str, tuple[int, str, str]] = {}
            for slot, task_name in V3_SLOT_TO_TASK_NAME.items():
                r = proc_rows[proc_rows["task_name"] == task_name]
                if len(r) != 1:
                    raise RuntimeError(
                        f"target_answer_edges must have exactly one row for process_id={pnum} task_name={task_name}, got {len(r)}"
                    )
                tc = str(r.iloc[0]["target_column"])
                idx, rule = resolve_answer_target_column_index(
                    task_name=task_name, target_column=tc, edge_target_columns=cols
                )
                slot_map[slot] = (idx, tc, rule)
                mapping_logs.append(
                    {
                        "split": split_name,
                        "process_id": pid,
                        "task_name": task_name,
                        "target_column": tc,
                        "mapped_edge_target_column": cols[idx],
                        "edge_target_column_idx": idx,
                        "mapping_rule": rule,
                    }
                )
            proc_col_map[pid] = slot_map

        # answer_edge_pos must index y_edge_pred rows directly.
        if batch.answer_edge_pos:
            for graph_idx, slot_map in enumerate(batch.answer_edge_pos):
                for slot, pos in slot_map.items():
                    pi = int(pos)
                    if pi < 0 or pi >= n_edges:
                        raise RuntimeError(f"answer_edge_pos out of range: graph={graph_idx} slot={slot} pos={pi} n_edges={n_edges}")
                    if float(target_masks["edge_stream"][pi].item()) <= 0.0:
                        raise RuntimeError(f"answer edge y_edge_mask is 0: graph={graph_idx} slot={slot} pos={pi}")

        y_mask = target_masks["edge_stream"].detach().cpu().numpy().tolist()
        ypn = y_pred_norm.detach().cpu().numpy()
        ytn = y_true_norm.detach().cpu().numpy()
        ypo = y_pred_orig.detach().cpu().numpy()
        yto = y_true_orig.detach().cpu().numpy()
        context_mask = list(getattr(export, "is_context", []) or [])
        use_context_mask = len(context_mask) == n_edges

        for e in range(n_edges):
            if use_context_mask and float(context_mask[e]) > 0.5:
                continue
            task_names = str(export.answer_task_names[e] or "")
            task_set = {x for x in task_names.split(";") if x}
            for j, prop in enumerate(cols):
                yto_ij = float(yto[e, j])
                ypo_ij = float(ypo[e, j])
                ytn_ij = float(ytn[e, j])
                ypn_ij = float(ypn[e, j])
                signed_e_o = ypo_ij - yto_ij
                ae = abs(signed_e_o)
                se = ae * ae
                se_n = ypn_ij - ytn_ij
                ae_n = abs(se_n)
                se2_n = se_n * se_n
                denom_smape = abs(ypo_ij) + abs(yto_ij) + 1e-12
                smape_p = 2.0 * ae / denom_smape
                yt_abs = max(abs(yto_ij), 1e-12)
                rel_signed = signed_e_o / yt_abs
                rows.append(
                    {
                        "split": split_name,
                        "process_id": export.process_id[e],
                        "sample_id": export.sample_id[e],
                        "canonical_edge_id": export.canonical_edge_id[e],
                        "main_data_stream_key": export.main_data_stream_key[e],
                        "stream_role": export.stream_role[e],
                        "is_input_edge": float(export.is_input_edge[e]),
                        "is_output_edge": float(export.is_output_edge[e]),
                        "is_internal_edge": float(export.is_internal_edge[e]),
                        "property_name": str(prop),
                        "y_true_norm": float(ytn[e, j]),
                        "y_pred_norm": float(ypn[e, j]),
                        "y_true_orig": float(yto[e, j]),
                        "y_pred_orig": float(ypo[e, j]),
                        "signed_error_orig": float(signed_e_o),
                        "abs_error_orig": float(ae),
                        "squared_error_orig": float(se),
                        "signed_error_norm": float(se_n),
                        "abs_error_norm": float(ae_n),
                        "squared_error_norm": float(se2_n),
                        "smape_point_orig": float(smape_p),
                        "rel_signed_error_orig": float(rel_signed),
                        "y_edge_mask": float(y_mask[e]),
                        "is_answer_edge": 1 if task_set else 0,
                        "answer_task_name": ";".join(sorted(task_set)),
                    }
                )

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        z = 0.0
        empty = pd.DataFrame()
        return EdgeAllEvalArtifacts(
            metrics={"loss_total": z, "loss_edge": z, "edge_all_mse_norm": z, "edge_all_mae_norm": z, "edge_all_rmse_orig": z, "edge_all_mae_orig": z},
            edge_predictions=pred_df,
            metrics_by_process=empty,
            metrics_by_property=empty,
            metrics_by_stream_role=empty,
            answer_edge_metrics=empty,
            mapping_logs=mapping_logs,
        )

    sup = pred_df[pred_df["y_edge_mask"] > 0.0].copy()
    mse_norm = float(((sup["y_pred_norm"] - sup["y_true_norm"]) ** 2).mean()) if len(sup) else 0.0
    mae_norm = float((sup["y_pred_norm"] - sup["y_true_norm"]).abs().mean()) if len(sup) else 0.0
    rmse_orig = math.sqrt(float(sup["squared_error_orig"].mean())) if len(sup) else 0.0
    mae_orig = float(sup["abs_error_orig"].mean()) if len(sup) else 0.0

    by_property = []
    for prop, g in sup.groupby("property_name"):
        mb = _metric_bundle(g["y_true_orig"], g["y_pred_orig"])
        by_property.append(
            {
                "property_name": prop,
                **mb,
                "property_mae_orig": float(g["abs_error_orig"].mean()),
                "property_rmse_orig": math.sqrt(float(g["squared_error_orig"].mean())),
                "property_r2_orig": _r2(g["y_true_orig"], g["y_pred_orig"]),
                "property_smape_orig": _smape(g["y_true_orig"], g["y_pred_orig"]),
                "count": int(len(g)),
            }
        )
    by_property_df = pd.DataFrame(by_property).sort_values("property_name").reset_index(drop=True)

    by_process = []
    for pid, g in sup.groupby("process_id"):
        mb = _metric_bundle(g["y_true_orig"], g["y_pred_orig"])
        by_process.append(
            {
                "process_id": pid,
                **mb,
                "edge_all_mae_orig": float(g["abs_error_orig"].mean()),
                "edge_all_rmse_orig": math.sqrt(float(g["squared_error_orig"].mean())),
                "count": int(len(g)),
            }
        )
    by_process_df = pd.DataFrame(by_process).sort_values("process_id").reset_index(drop=True)

    role_rows = []
    for role_name, g in sup.groupby("stream_role"):
        mb = _metric_bundle(g["y_true_orig"], g["y_pred_orig"])
        role_rows.append(
            {
                "stream_role": role_name,
                **mb,
                "edge_all_mae_orig": float(g["abs_error_orig"].mean()),
                "edge_all_rmse_orig": math.sqrt(float(g["squared_error_orig"].mean())),
                "count": int(len(g)),
            }
        )
    by_role_df = pd.DataFrame(role_rows).sort_values("stream_role").reset_index(drop=True)

    # Required rollups: input/internal/output.
    role_agg = []
    for role, col in (("input", "is_input_edge"), ("internal", "is_internal_edge"), ("output", "is_output_edge")):
        g = sup[sup[col] > 0.0]
        mb = _metric_bundle(g["y_true_orig"], g["y_pred_orig"])
        role_agg.append(
            {
                "stream_role_group": role,
                **mb,
                "edge_all_mae_orig": float(g["abs_error_orig"].mean()) if len(g) else 0.0,
                "edge_all_rmse_orig": math.sqrt(float(g["squared_error_orig"].mean())) if len(g) else 0.0,
                "count": int(len(g)),
            }
        )
    by_role_df = pd.concat([by_role_df, pd.DataFrame(role_agg)], ignore_index=True)

    answer_rows = []
    answer_sup = sup[sup["is_answer_edge"] > 0].copy()
    if not answer_sup.empty and "answer_task_name" in answer_sup.columns:
        answer_sup["__task_tokens"] = answer_sup["answer_task_name"].astype(str).str.split(";")
        answer_exp = answer_sup.explode("__task_tokens")
        answer_exp["task_name"] = answer_exp["__task_tokens"].astype(str).str.strip()
        answer_exp = answer_exp[answer_exp["task_name"] != ""]
        for task_name, g in answer_exp.groupby("task_name", dropna=False):
            g_prop = _filter_answer_task_property(str(task_name), g)
            if g_prop.empty:
                g_prop = g
            mb = _metric_bundle(g_prop["y_true_orig"], g_prop["y_pred_orig"])
            answer_rows.append(
                {
                    "task_name": str(task_name),
                    "mapped_property_name": _DEFAULT_TASK_META.get(str(task_name), {}).get(
                        "mapped_property_name", ""
                    ),
                    **mb,
                    f"{task_name}_mae_orig": float(g_prop["abs_error_orig"].mean()) if len(g_prop) else 0.0,
                    f"{task_name}_rmse_orig": math.sqrt(float(g_prop["squared_error_orig"].mean()))
                    if len(g_prop)
                    else 0.0,
                    f"{task_name}_r2_orig": float(mb["r2"]),
                    "count": int(len(g_prop)),
                }
            )
    answer_df = pd.DataFrame(answer_rows)

    overall = _metric_bundle(sup["y_true_orig"], sup["y_pred_orig"])
    metrics = {
        "loss_total": (loss_sum / step_count) if step_count > 0 else 0.0,
        "loss_edge": (loss_sum / step_count) if step_count > 0 else 0.0,
        "edge_all_mse_norm": mse_norm,
        "edge_all_mae_norm": mae_norm,
        "edge_all_rmse_orig": rmse_orig,
        "edge_all_mae_orig": mae_orig,
        "edge_all_r2_orig": float(overall["r2"]),
        "true_mean": float(overall["true_mean"]),
        "pred_mean": float(overall["pred_mean"]),
        "mean_bias": float(overall["mean_bias"]),
        "true_std": float(overall["true_std"]),
        "pred_std": float(overall["pred_std"]),
        "std_ratio": float(overall["std_ratio"]),
        "mae": float(overall["mae"]),
        "rmse": float(overall["rmse"]),
        "r2": float(overall["r2"]),
        "n_samples": int(overall["n_samples"]),
        "r2_unstable": bool(overall["r2_unstable"]),
    }
    for row in answer_rows:
        t = str(row["task_name"])
        metrics[f"{t}_mae_orig"] = float(row.get(f"{t}_mae_orig", 0.0))
        metrics[f"{t}_rmse_orig"] = float(row.get(f"{t}_rmse_orig", 0.0))
        metrics[f"{t}_r2_orig"] = float(row.get(f"{t}_r2_orig", 0.0))
        if t == "target_h2":
            metrics["target_h2_mae"] = metrics[f"{t}_mae_orig"]
            metrics["target_h2_rmse"] = metrics[f"{t}_rmse_orig"]
            metrics["target_h2_r2"] = metrics[f"{t}_r2_orig"]
        elif t == "tailgas_co2":
            metrics["tailgas_co2_mae"] = metrics[f"{t}_mae_orig"]
            metrics["tailgas_co2_rmse"] = metrics[f"{t}_rmse_orig"]
            metrics["tailgas_co2_r2"] = metrics[f"{t}_r2_orig"]

    if not by_property_df.empty and "property_name" in by_property_df.columns:
        frac_row = by_property_df[by_property_df["property_name"].astype(str) == "Frac_H2"]
        if not frac_row.empty:
            metrics["edge_all_frac_h2_r2_orig"] = float(frac_row.iloc[0].get("property_r2_orig", frac_row.iloc[0].get("r2", 0.0)))

    metrics = augment_metrics_with_legacy_aliases(metrics)

    return EdgeAllEvalArtifacts(
        metrics=metrics,
        edge_predictions=pred_df,
        metrics_by_process=by_process_df,
        metrics_by_property=by_property_df,
        metrics_by_stream_role=by_role_df,
        answer_edge_metrics=answer_df,
        mapping_logs=mapping_logs,
    )


def write_edge_all_artifacts(
    *,
    out_dir: Path,
    artifacts: EdgeAllEvalArtifacts,
    scaler_columns: Sequence[str],
    scaler_mean: torch.Tensor | None,
    scaler_std: torch.Tensor | None,
    scaler_counts: Sequence[int] | None,
    scaler_source: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts.metrics_by_process.to_csv(out_dir / "metrics_by_process.csv", index=False)
    artifacts.metrics_by_property.to_csv(out_dir / "metrics_by_property.csv", index=False)
    artifacts.metrics_by_stream_role.to_csv(out_dir / "metrics_by_stream_role.csv", index=False)
    artifacts.answer_edge_metrics.to_csv(out_dir / "answer_edge_metrics.csv", index=False)

    metrics_payload = dict(artifacts.metrics)
    metrics_payload["answer_column_mappings"] = artifacts.mapping_logs
    if scaler_source:
        metrics_payload["scaler_source"] = scaler_source
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    if scaler_mean is not None and scaler_std is not None:
        mean_np = scaler_mean.detach().cpu().numpy().tolist()
        std_np = scaler_std.detach().cpu().numpy().tolist()
    else:
        mean_np = [0.0 for _ in scaler_columns]
        std_np = [1.0 for _ in scaler_columns]
    counts = list(scaler_counts) if scaler_counts is not None else [0 for _ in scaler_columns]
    pd.DataFrame(
        {
            "property": list(scaler_columns),
            "mean": mean_np,
            "std": std_np,
            "masked_fit_count": counts,
            "scaler_source": [scaler_source for _ in scaler_columns],
        }
    ).to_csv(out_dir / "scaler_stats.csv", index=False)
    write_edge_all_standard_metrics_json(out_dir=out_dir, artifacts=artifacts)


# Legacy/default hints for common fixed slots.
_DEFAULT_TASK_META: dict[str, dict[str, str]] = {
    "target_h2": {"edge_slot": "target", "mapped_property_name": "Frac_H2"},
    "tailgas_co2": {"edge_slot": "tailgas", "mapped_property_name": "Frac_CO2"},
}


def _filter_answer_task_property(task_name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only the mapped Frac_* column for a main answer task (not all edge properties)."""
    if frame.empty or "property_name" not in frame.columns:
        return frame
    meta = _DEFAULT_TASK_META.get(str(task_name).strip())
    if meta and meta.get("mapped_property_name"):
        return frame[frame["property_name"].astype(str) == str(meta["mapped_property_name"])]
    return frame[frame["property_name"].astype(str).str.startswith("Frac_")]


def _split_answer_tasks(raw: Any) -> list[str]:
    return [tok.strip() for tok in str(raw or "").split(";") if tok and tok.strip()]


def _collect_main_tasks(artifacts: EdgeAllEvalArtifacts) -> list[str]:
    tasks: set[str] = set()
    ans_df = artifacts.answer_edge_metrics
    if ans_df is not None and not ans_df.empty and "task_name" in ans_df.columns:
        tasks |= {str(t).strip() for t in ans_df["task_name"].tolist() if str(t).strip()}
    for row in artifacts.mapping_logs or []:
        task = str((row or {}).get("task_name", "")).strip()
        if task and task != "v4_target_weighting":
            tasks.add(task)
    pred = artifacts.edge_predictions
    if pred is not None and not pred.empty and "answer_task_name" in pred.columns:
        for raw in pred["answer_task_name"].tolist():
            tasks |= set(_split_answer_tasks(raw))
    return sorted(tasks)


def _jsonable_scalar(value: Any) -> float | int | bool | str | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        fv = float(value)
        return fv if math.isfinite(fv) else None
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return None


def _metric_bundle_to_json(mb: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "mae": _jsonable_scalar(mb.get("mae")),
        "rmse": _jsonable_scalar(mb.get("rmse")),
        "r2": _jsonable_scalar(mb.get("r2")),
        "smape": _jsonable_scalar(
            mb.get("smape") if "smape" in mb else mb.get("property_smape_orig")
        ),
        "true_mean": _jsonable_scalar(mb.get("true_mean")),
        "pred_mean": _jsonable_scalar(mb.get("pred_mean")),
        "mean_bias": _jsonable_scalar(mb.get("mean_bias")),
        "true_std": _jsonable_scalar(mb.get("true_std")),
        "pred_std": _jsonable_scalar(mb.get("pred_std")),
        "std_ratio": _jsonable_scalar(mb.get("std_ratio")),
        "n_samples": _jsonable_scalar(mb.get("n_samples")),
        "r2_unstable": bool(mb.get("r2_unstable", False)),
    }


def _mean_over_property_columns(
    prop_df: pd.DataFrame,
    *,
    value_cols: Sequence[str],
) -> dict[str, Any]:
    if prop_df is None or prop_df.empty:
        return {"n_properties": 0}
    out: dict[str, Any] = {"n_properties": int(len(prop_df))}
    counts = prop_df["count"].astype(float) if "count" in prop_df.columns else pd.Series([1.0] * len(prop_df))
    wsum = float(counts.sum())
    for col in value_cols:
        if col not in prop_df.columns:
            continue
        series = pd.to_numeric(prop_df[col], errors="coerce")
        out[f"{col}_unweighted_mean"] = _jsonable_scalar(series.mean())
        if wsum > 0:
            out[f"{col}_count_weighted_mean"] = _jsonable_scalar((series * counts).sum() / wsum)
    return out


def build_main_targets_metrics_json(
    artifacts: EdgeAllEvalArtifacts,
    *,
    split: str = "",
) -> dict[str, Any]:
    """Per main answer target (H2, CO2, other Frac_* on answer edges) metrics for JSON export."""
    metrics = artifacts.metrics
    primary: dict[str, Any] = {}
    task_rows: dict[str, Mapping[str, Any]] = {}
    ans_df = artifacts.answer_edge_metrics
    if ans_df is not None and not ans_df.empty:
        for _, row in ans_df.iterrows():
            task = str(row.get("task_name", "")).strip()
            if task:
                task_rows[task] = row
    mapping_by_task: dict[str, list[dict[str, Any]]] = {}
    for row in artifacts.mapping_logs or []:
        task = str((row or {}).get("task_name", "")).strip()
        if not task or task == "v4_target_weighting":
            continue
        mapping_by_task.setdefault(task, []).append(
            {
                "process_id": str((row or {}).get("process_id", "")),
                "target_column": str((row or {}).get("target_column", "")),
                "mapped_edge_target_column": str((row or {}).get("mapped_edge_target_column", "")),
                "edge_target_column_idx": _jsonable_scalar((row or {}).get("edge_target_column_idx")),
                "mapping_rule": str((row or {}).get("mapping_rule", "")),
            }
        )

    answer_channels: list[dict[str, Any]] = []
    pred = artifacts.edge_predictions
    if pred is not None and not pred.empty and "is_answer_edge" in pred.columns:
        ans = pred[(pred["y_edge_mask"] > 0.0) & (pred["is_answer_edge"] > 0)].copy()
        if not ans.empty and "answer_task_name" in ans.columns:
            ans["__task_tokens"] = ans["answer_task_name"].astype(str).str.split(";")
            ans_exp = ans.explode("__task_tokens")
            ans_exp["task_name"] = ans_exp["__task_tokens"].astype(str).str.strip()
            ans_exp = ans_exp[ans_exp["task_name"] != ""]
            for (task_raw, prop), g in ans_exp.groupby(["task_name", "property_name"], dropna=False):
                task_s = str(task_raw).strip()
                if not task_s:
                    continue
                mb = _metric_bundle(g["y_true_orig"], g["y_pred_orig"])
                answer_channels.append(
                    {
                        "task_name": task_s,
                        "property_name": str(prop),
                        "n_samples": int(len(g)),
                        **_metric_bundle_to_json(mb),
                    }
                )

    channel_by_task: dict[str, dict[str, Any]] = {}
    for ch in answer_channels:
        task_s = str(ch.get("task_name", "")).strip()
        prop = str(ch.get("property_name", "")).strip()
        meta = _DEFAULT_TASK_META.get(task_s, {})
        mapped = str(meta.get("mapped_property_name", "")).strip()
        if task_s and mapped and prop == mapped:
            channel_by_task[task_s] = ch

    for task in _collect_main_tasks(artifacts):
        row = task_rows.get(task)
        default_meta = _DEFAULT_TASK_META.get(task, {})
        ch = channel_by_task.get(task)
        task_maps = mapping_by_task.get(task, [])
        unique_mapped_cols = sorted({m["mapped_edge_target_column"] for m in task_maps if m["mapped_edge_target_column"]})
        unique_target_cols = sorted({m["target_column"] for m in task_maps if m["target_column"]})
        entry: dict[str, Any] = {
            "task_name": task,
            "edge_slot": default_meta.get("edge_slot"),
            "mapped_property_name": default_meta.get("mapped_property_name"),
            "mapped_edge_target_columns": unique_mapped_cols,
            "target_columns": unique_target_cols,
            "mae_orig": _jsonable_scalar(
                ch.get("mae") if ch else (row.get(f"{task}_mae_orig", row.get("mae")) if row is not None else metrics.get(f"{task}_mae_orig"))
            ),
            "rmse_orig": _jsonable_scalar(
                ch.get("rmse") if ch else (row.get(f"{task}_rmse_orig", row.get("rmse")) if row is not None else metrics.get(f"{task}_rmse_orig"))
            ),
            "r2_orig": _jsonable_scalar(
                ch.get("r2") if ch else (row.get(f"{task}_r2_orig", row.get("r2")) if row is not None else metrics.get(f"{task}_r2_orig"))
            ),
            "n_samples": _jsonable_scalar(
                ch.get("n_samples") if ch else (row.get("count", row.get("n_samples")) if row is not None else None)
            ),
            "per_process_mappings": task_maps,
        }
        if not entry["mapped_property_name"] and len(unique_mapped_cols) == 1:
            entry["mapped_property_name"] = unique_mapped_cols[0]
        primary[task] = entry

    answer_metrics_flat: dict[str, Any] = {}
    for key, val in sorted(metrics.items()):
        ks = str(key)
        if ks.startswith("answer_target_") or ks.startswith("answer_tailgas_"):
            answer_metrics_flat[ks] = _jsonable_scalar(val)
        if ks in ("target_h2_mae", "target_h2_rmse", "target_h2_r2", "target_h2_smape"):
            answer_metrics_flat[ks] = _jsonable_scalar(val)
        if ks in ("tailgas_co2_mae", "tailgas_co2_rmse", "tailgas_co2_r2", "tailgas_co2_smape"):
            answer_metrics_flat[ks] = _jsonable_scalar(val)

    return {
        "split": split,
        "definition": (
            "primary_answer_tasks: all answer tasks discovered from answer edges/mapping logs "
            "(process-specific target columns included); "
            "answer_edge_by_channel: all answer-edge task×property groups; "
            "answer_metrics_flat: scalar keys from detailed eval (incl. Frac_* channels)."
        ),
        "primary_answer_tasks": primary,
        "answer_edge_by_channel": answer_channels,
        "answer_metrics_flat": answer_metrics_flat,
    }


def build_edge_features_mean_json(
    artifacts: EdgeAllEvalArtifacts,
    *,
    split: str = "",
) -> dict[str, Any]:
    """Mean metrics over all edge feature columns (STREAM_EDGE properties on masked edges)."""
    prop_df = artifacts.metrics_by_property
    metrics = artifacts.metrics
    per_property: list[dict[str, Any]] = []
    if prop_df is not None and not prop_df.empty:
        for _, row in prop_df.iterrows():
            per_property.append(
                {
                    "property_name": str(row.get("property_name", "")),
                    "count": _jsonable_scalar(row.get("count")),
                    "mae_orig": _jsonable_scalar(row.get("property_mae_orig", row.get("mae"))),
                    "rmse_orig": _jsonable_scalar(row.get("property_rmse_orig", row.get("rmse"))),
                    "r2_orig": _jsonable_scalar(row.get("property_r2_orig", row.get("r2"))),
                    "smape_orig": _jsonable_scalar(row.get("property_smape_orig")),
                }
            )

    value_cols = ("property_mae_orig", "property_rmse_orig", "property_r2_orig", "property_smape_orig")
    mean_all = _mean_over_property_columns(prop_df, value_cols=value_cols)
    frac_df = (
        prop_df[prop_df["property_name"].astype(str).str.startswith("Frac_")]
        if prop_df is not None and not prop_df.empty
        else pd.DataFrame()
    )
    mean_frac = _mean_over_property_columns(frac_df, value_cols=value_cols)

    overall = {
        "edge_all_mae_orig": _jsonable_scalar(metrics.get("edge_all_mae_orig")),
        "edge_all_rmse_orig": _jsonable_scalar(metrics.get("edge_all_rmse_orig")),
        "edge_all_r2_orig": _jsonable_scalar(metrics.get("edge_all_r2_orig")),
        "edge_all_mae_norm": _jsonable_scalar(metrics.get("edge_all_mae_norm")),
        "edge_all_mse_norm": _jsonable_scalar(metrics.get("edge_all_mse_norm")),
        "n_samples": _jsonable_scalar(metrics.get("n_samples")),
    }

    return {
        "split": split,
        "definition": (
            "overall_*: all masked edge×property points pooled; "
            "mean_over_edge_features: arithmetic mean of per-property metrics "
            "(unweighted and count-weighted); frac_* subset is Frac_H2, Frac_CO2, etc. only."
        ),
        "overall_all_edge_properties": overall,
        "per_property": per_property,
        "mean_over_edge_features": mean_all,
        "mean_over_frac_features_only": mean_frac,
    }


def write_edge_all_standard_metrics_json(
    *,
    out_dir: Path,
    artifacts: EdgeAllEvalArtifacts,
    split: str = "",
) -> None:
    """Write metrics_main_targets.json and metrics_edge_features_mean.json (always after eval)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    split_name = split
    if not split_name:
        ep = artifacts.edge_predictions
        if ep is not None and not ep.empty and "split" in ep.columns:
            split_name = str(ep["split"].iloc[0])
    main_payload = build_main_targets_metrics_json(artifacts, split=split_name)
    edge_mean_payload = build_edge_features_mean_json(artifacts, split=split_name)
    (out_dir / "metrics_main_targets.json").write_text(
        json.dumps(main_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "metrics_edge_features_mean.json").write_text(
        json.dumps(edge_mean_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_edge_all_metrics_json_all_splits(
    out_root: Path,
    splits: Mapping[str, EdgeAllEvalArtifacts],
) -> None:
    """Run-level JSON: all splits for main targets and edge-feature means."""
    if not splits:
        return
    out_root.mkdir(parents=True, exist_ok=True)
    main_by_split = {sp: build_main_targets_metrics_json(art, split=sp) for sp, art in splits.items()}
    edge_mean_by_split = {sp: build_edge_features_mean_json(art, split=sp) for sp, art in splits.items()}
    (out_root / "metrics_main_targets_all_splits.json").write_text(
        json.dumps({"splits": main_by_split}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_root / "metrics_edge_features_mean_all_splits.json").write_text(
        json.dumps({"splits": edge_mean_by_split}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _pick_val_history_best_index(
    val: Mapping[str, Any],
    *,
    w_h2: float = 5.0,
    w_co2: float = 3.0,
) -> int | None:
    h2_s = val.get("metric_target_r2") or val.get("target_h2_r2") or []
    co2_s = val.get("metric_tailgas_r2") or val.get("tailgas_co2_r2") or []
    if not isinstance(h2_s, list) or not isinstance(co2_s, list) or not h2_s or not co2_s:
        return None
    n = min(len(h2_s), len(co2_s))
    wh, wc = float(w_h2), float(w_co2)
    wsum = wh + wc
    if wsum <= 0:
        wh, wc, wsum = 1.0, 1.0, 2.0
    best_i: int | None = None
    best_v = float("-inf")
    for i in range(n):
        a, b = float(h2_s[i]), float(co2_s[i])
        if not (math.isfinite(a) and math.isfinite(b)):
            if math.isfinite(a):
                v = a
            elif math.isfinite(b):
                v = b
            else:
                continue
        else:
            v = (wh * a + wc * b) / wsum
        if v > best_v:
            best_v = v
            best_i = i
    return best_i


def _val_series_at(val: Mapping[str, Any], idx: int, *keys: str) -> float | None:
    for key in keys:
        arr = val.get(key)
        if isinstance(arr, list) and len(arr) > idx:
            try:
                fv = float(arr[idx])
                return fv if math.isfinite(fv) else None
            except (TypeError, ValueError):
                pass
    return None


def build_main_targets_json_from_val_history(
    val: Mapping[str, Any],
    *,
    split_label: str = "val",
    w_h2: float = 5.0,
    w_co2: float = 3.0,
) -> dict[str, Any]:
    """Fallback when full post-train export is skipped: use val epoch series from training."""
    best_i = _pick_val_history_best_index(val, w_h2=w_h2, w_co2=w_co2)
    epochs = val.get("epoch") or []
    best_epoch = float(epochs[best_i]) if best_i is not None and isinstance(epochs, list) and best_i < len(epochs) else None

    def _task_at(task: str, slot: str, prop: str) -> dict[str, Any]:
        if best_i is None:
            return {"task_name": task, "edge_slot": slot, "mapped_property_name": prop}
        return {
            "task_name": task,
            "edge_slot": slot,
            "mapped_property_name": prop,
            "mae_orig": _val_series_at(
                val,
                best_i,
                f"target_h2_mae" if task == "target_h2" else "tailgas_co2_mae",
                f"metric_target_h2_mae" if task == "target_h2" else "metric_tailgas_co2_mae",
            ),
            "rmse_orig": _val_series_at(
                val,
                best_i,
                f"target_h2_rmse" if task == "target_h2" else "tailgas_co2_rmse",
                f"metric_target_h2_rmse" if task == "target_h2" else "metric_tailgas_co2_rmse",
            ),
            "r2_orig": _val_series_at(
                val,
                best_i,
                "metric_target_r2" if task == "target_h2" else "metric_tailgas_r2",
                "target_h2_r2" if task == "target_h2" else "tailgas_co2_r2",
            ),
            "smape_orig": _val_series_at(
                val,
                best_i,
                f"target_h2_smape" if task == "target_h2" else "tailgas_co2_smape",
                f"metric_target_h2_smape" if task == "target_h2" else "metric_tailgas_co2_smape",
            ),
        }

    primary = {
        "target_h2": _task_at("target_h2", "target", "Frac_H2"),
        "tailgas_co2": _task_at("tailgas_co2", "tailgas", "Frac_CO2"),
    }

    answer_flat: dict[str, Any] = {}
    answer_channels: list[dict[str, Any]] = []
    if best_i is not None:
        channel_metrics: dict[tuple[str, str], dict[str, Any]] = {}
        for key, arr in val.items():
            if not isinstance(arr, list) or best_i >= len(arr):
                continue
            ks = str(key)
            if ks.startswith("answer_target_") or ks.startswith("answer_tailgas_"):
                answer_flat[ks] = _jsonable_scalar(arr[best_i])
            if ks in (
                "target_h2_mae",
                "target_h2_rmse",
                "metric_target_r2",
                "target_h2_smape",
                "tailgas_co2_mae",
                "tailgas_co2_rmse",
                "metric_tailgas_r2",
                "tailgas_co2_smape",
            ):
                answer_flat[ks] = _jsonable_scalar(arr[best_i])
            for slot, prefix in (("target", "answer_target"), ("tailgas", "answer_tailgas")):
                if not ks.startswith(f"{prefix}_"):
                    continue
                rest = ks[len(prefix) + 1 :]
                if rest.endswith("_mae"):
                    prop, metric = rest[: -len("_mae")], "mae"
                elif rest.endswith("_rmse"):
                    prop, metric = rest[: -len("_rmse")], "rmse"
                elif rest.endswith("_r2"):
                    prop, metric = rest[: -len("_r2")], "r2"
                elif rest.endswith("_smape"):
                    prop, metric = rest[: -len("_smape")], "smape"
                else:
                    continue
                channel_metrics.setdefault((slot, prop), {})[metric] = _jsonable_scalar(arr[best_i])
        for (slot, prop), mb in sorted(channel_metrics.items()):
            answer_channels.append(
                {
                    "edge_slot": slot,
                    "property_name": prop,
                    "mae": mb.get("mae"),
                    "rmse": mb.get("rmse"),
                    "r2": mb.get("r2"),
                    "smape": mb.get("smape"),
                }
            )

    return {
        "split": split_label,
        "source": "train_val_history",
        "best_val_row_index": best_i,
        "best_epoch": best_epoch,
        "objective_weights": {"h2": float(w_h2), "co2": float(w_co2)},
        "primary_answer_tasks": primary,
        "answer_edge_by_channel": answer_channels,
        "answer_metrics_flat": answer_flat,
        "note": "Post-train detailed eval export was not run; metrics are from validation during training.",
    }


def build_edge_features_mean_json_from_val_history(
    val: Mapping[str, Any],
    *,
    split_label: str = "val",
    w_h2: float = 5.0,
    w_co2: float = 3.0,
) -> dict[str, Any]:
    """Val-history fallback: only pooled normalized edge loss (no per-property breakdown)."""
    best_i = _pick_val_history_best_index(val, w_h2=w_h2, w_co2=w_co2)
    epochs = val.get("epoch") or []
    best_epoch = float(epochs[best_i]) if best_i is not None and isinstance(epochs, list) and best_i < len(epochs) else None
    overall: dict[str, Any] = {}
    if best_i is not None:
        overall = {
            "edge_all_mae_norm": _val_series_at(val, best_i, "edge_all_mae"),
            "edge_all_mse_norm": _val_series_at(val, best_i, "edge_all_mse"),
        }
    return {
        "split": split_label,
        "source": "train_val_history",
        "best_val_row_index": best_i,
        "best_epoch": best_epoch,
        "overall_all_edge_properties": overall,
        "per_property": [],
        "mean_over_edge_features": {"n_properties": 0},
        "mean_over_frac_features_only": {"n_properties": 0},
        "note": (
            "Per-property edge-feature means require post-train export "
            "(edge_all_export_predictions=true) and metrics_edge_features_mean.json under train/val/test."
        ),
    }


def write_main_targets_json_from_val_history(
    output_dir: Path,
    val: Mapping[str, Any],
    *,
    w_h2: float = 5.0,
    w_co2: float = 3.0,
) -> None:
    """Write metrics_main_targets.json from val history when detailed export is unavailable."""
    payload = build_main_targets_json_from_val_history(val, w_h2=w_h2, w_co2=w_co2)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics_main_targets.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_edge_features_mean_json_from_val_history(
    output_dir: Path,
    val: Mapping[str, Any],
    *,
    w_h2: float = 5.0,
    w_co2: float = 3.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_edge_features_mean_json_from_val_history(val, w_h2=w_h2, w_co2=w_co2)
    (output_dir / "metrics_edge_features_mean.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_edge_all_metrics_json_from_val_history(
    output_dir: Path,
    val: Mapping[str, Any],
    *,
    w_h2: float = 5.0,
    w_co2: float = 3.0,
) -> None:
    """Root-level JSON when full edge_all export was skipped."""
    write_main_targets_json_from_val_history(output_dir, val, w_h2=w_h2, w_co2=w_co2)
    write_edge_features_mean_json_from_val_history(output_dir, val, w_h2=w_h2, w_co2=w_co2)


def write_edge_all_combined_metrics_csvs(
    out_dir: Path,
    splits: Mapping[str, EdgeAllEvalArtifacts],
    merged_edge_predictions: pd.DataFrame,
) -> None:
    """Merged CSV bundle: overall scalars, rollup tables, per-edge×property summary (not full cell dump)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    chunks: list[pd.DataFrame] = []
    for split_name, art in splits.items():
        for k, v in art.metrics.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "split": split_name,
                    "scope": "overall",
                    "table": "aggregate",
                    "metric": k,
                    "value": fv,
                }
            )
        for table_name, df in (
            ("metrics_by_process", art.metrics_by_process),
            ("metrics_by_property", art.metrics_by_property),
            ("metrics_by_stream_role", art.metrics_by_stream_role),
            ("answer_edge_metrics", art.answer_edge_metrics),
        ):
            if df is None or df.empty:
                continue
            t = df.copy()
            t.insert(0, "rollup_table", table_name)
            t.insert(0, "split", split_name)
            chunks.append(t)
    if rows:
        pd.DataFrame(rows).to_csv(out_dir / "edge_all_overall_metrics_by_split.csv", index=False)
    if chunks:
        pd.concat(chunks, ignore_index=True).to_csv(out_dir / "edge_all_rollups_all_splits.csv", index=False)
    if not merged_edge_predictions.empty:
        # Long-form per-cell rollup from in-memory merged predictions (edge_predictions.csv not written).
        sup = merged_edge_predictions[merged_edge_predictions["y_edge_mask"] > 0.0].copy()
        if not sup.empty:
            per_rows: list[dict[str, Any]] = []
            for (sp, cid, prop), g in sup.groupby(["split", "canonical_edge_id", "property_name"], sort=False):
                mb = _metric_bundle(g["y_true_orig"], g["y_pred_orig"])
                per_rows.append(
                    {
                        "split": str(sp),
                        "canonical_edge_id": cid,
                        "property_name": prop,
                        "n_samples": int(len(g)),
                        **mb,
                        "mean_abs_error_orig": float(g["abs_error_orig"].mean()),
                        "mean_squared_error_orig": float(g["squared_error_orig"].mean()),
                        "mean_smape_point_orig": float(g["smape_point_orig"].mean()),
                    }
                )
            pd.DataFrame(per_rows).sort_values(["split", "canonical_edge_id", "property_name"]).to_csv(
                out_dir / "edge_all_per_edge_per_property_metrics.csv", index=False
            )


def write_edge_all_eval_figures(plot_dir: Path, edge_predictions: pd.DataFrame) -> None:
    """PNG summaries from merged edge_predictions (train/val/test)."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    if edge_predictions.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    df = edge_predictions[edge_predictions["y_edge_mask"] > 0.0].copy()
    if df.empty:
        return

    props = sorted(df["property_name"].astype(str).unique())
    splits_present = [s for s in ("train", "val", "test") if s in set(df["split"].astype(str))]
    if props and splits_present:
        x = np.arange(len(props))
        width = min(0.28, 0.8 / max(len(splits_present), 1))
        fig, ax = plt.subplots(figsize=(max(8.0, len(props) * 0.55), 5.2))
        for i, sp in enumerate(splits_present):
            sub = df[df["split"].astype(str) == sp]
            mae = [
                float(sub[sub["property_name"] == p]["abs_error_orig"].mean())
                if not sub[sub["property_name"] == p].empty
                else 0.0
                for p in props
            ]
            ax.bar(x + (i - (len(splits_present) - 1) / 2) * width, mae, width, label=sp)
        ax.set_xticks(x)
        ax.set_xticklabels(props, rotation=45, ha="right")
        ax.set_ylabel("MAE (orig scale)")
        ax.set_title("Mean |error| by property and split")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "edge_all_mae_by_property_and_split.png", dpi=150)
        plt.close(fig)

    val = df[df["split"].astype(str) == "val"]
    if not val.empty:
        vprops = sorted(val["property_name"].astype(str).unique())[:12]
        n = len(vprops)
        if n:
            ncols = 4
            nrows = int(np.ceil(n / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False)
            for idx, prop in enumerate(vprops):
                r, c = divmod(idx, ncols)
                ax = axes[r][c]
                pv = val[val["property_name"] == prop]
                if len(pv) > 5000:
                    pv = pv.sample(5000, random_state=0)
                ax.scatter(pv["y_true_orig"], pv["y_pred_orig"], s=4, alpha=0.25)
                tmin = float(min(pv["y_true_orig"].min(), pv["y_pred_orig"].min()))
                tmax = float(max(pv["y_true_orig"].max(), pv["y_pred_orig"].max()))
                if tmax > tmin:
                    ax.plot([tmin, tmax], [tmin, tmax], "r--", lw=0.9)
                ax.set_title(str(prop))
                ax.set_xlabel("true")
                ax.set_ylabel("pred")
                ax.grid(True, alpha=0.3)
            for j in range(n, nrows * ncols):
                r, c = divmod(j, ncols)
                axes[r][c].set_visible(False)
            fig.suptitle("Validation: true vs pred (per property; subsampled if large)")
            fig.tight_layout()
            fig.savefig(plot_dir / "edge_all_val_true_vs_pred_grid.png", dpi=150)
            plt.close(fig)

        e = val["signed_error_orig"].astype(float)
        if len(e) > 20000:
            e = e.sample(20000, random_state=1)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.hist(e, bins=60, color="C0", alpha=0.75, edgecolor="none")
        ax.set_xlabel("signed error (pred - true), orig scale")
        ax.set_ylabel("count")
        ax.set_title("Validation signed error distribution")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "edge_all_val_signed_error_hist.png", dpi=150)
        plt.close(fig)


def write_edge_all_extra_diagnostics(
    *,
    out_dir: Path,
    edge_predictions: pd.DataFrame,
) -> None:
    sup = edge_predictions[edge_predictions["y_edge_mask"] > 0.0].copy()
    # Canonical edge metrics
    by_edge = []
    for cid, g in sup.groupby("canonical_edge_id"):
        mb = _metric_bundle(g["y_true_orig"], g["y_pred_orig"])
        by_edge.append({"canonical_edge_id": cid, **mb})
    pd.DataFrame(by_edge).sort_values("canonical_edge_id").to_csv(
        out_dir / "metrics_by_canonical_edge.csv", index=False
    )

    # Target-specific diagnostics (mapped property + task)
    task_prop = {"target_h2": "Frac_H2", "tailgas_co2": "Frac_CO2"}
    for task, prop in task_prop.items():
        d = sup[
            (sup["answer_task_name"].str.contains(task, regex=False))
            & (sup["property_name"] == prop)
        ].copy()
        if d.empty:
            diag = pd.DataFrame([{"task_name": task, "mapped_property_name": prop, **_metric_bundle(pd.Series(dtype=float), pd.Series(dtype=float))}])
        else:
            diag = pd.DataFrame(
                [
                    {
                        "task_name": task,
                        "mapped_property_name": prop,
                        "answer_edge_id_mode": str(d["canonical_edge_id"].mode().iloc[0]),
                        "main_data_stream_key_mode": str(d["main_data_stream_key"].mode().iloc[0]),
                        **_metric_bundle(d["y_true_orig"], d["y_pred_orig"]),
                        "y_true_min": float(d["y_true_orig"].min()),
                        "y_true_max": float(d["y_true_orig"].max()),
                        "y_pred_min": float(d["y_pred_orig"].min()),
                        "y_pred_max": float(d["y_pred_orig"].max()),
                    }
                ]
            )
        diag_path = out_dir / f"{task}_diagnostic.csv"
        diag.to_csv(diag_path, index=False)

        # Detailed per-sample diagnostics for target_h2 (under-dispersion audit).
        if task == "target_h2":
            dd = d.copy()
            if not dd.empty:
                dd["error"] = dd["y_pred_orig"] - dd["y_true_orig"]
                dd["abs_error"] = dd["error"].abs()
                dd["error_norm"] = dd["y_pred_norm"] - dd["y_true_norm"]
                dd["true_percentile"] = dd["y_true_orig"].rank(method="average", pct=True)
                dd["pred_percentile"] = dd["y_pred_orig"].rank(method="average", pct=True)
                dd = dd[
                    [
                        "sample_id",
                        "process_id",
                        "canonical_edge_id",
                        "main_data_stream_key",
                        "stream_role",
                        "property_name",
                        "y_true_orig",
                        "y_pred_orig",
                        "error",
                        "abs_error",
                        "y_true_norm",
                        "y_pred_norm",
                        "error_norm",
                        "true_percentile",
                        "pred_percentile",
                    ]
                ].rename(columns={"property_name": "property"})
                dd.to_csv(out_dir / "target_h2_diagnostic.csv", index=False)

                # Top errors
                dd.sort_values("abs_error", ascending=False).head(20).to_csv(
                    out_dir / "target_h2_top_errors.csv", index=False
                )

                # True-value quantile bins
                binned = d.copy()
                binned["true_bin"] = pd.qcut(
                    binned["y_true_orig"], q=5, labels=False, duplicates="drop"
                )
                binned = binned.dropna(subset=["true_bin"])
                rows: list[dict] = []
                if binned.empty:
                    # qcut returns all-NaN when y_true is nearly constant; still emit one summary row.
                    mb = _metric_bundle(d["y_true_orig"], d["y_pred_orig"])
                    rows.append(
                        {
                            "true_bin": 0,
                            "bin_count": int(len(d)),
                            "mae": float(mb["mae"]),
                            "rmse": float(mb["rmse"]),
                            "bias": float(mb["mean_bias"]),
                            "pred_std": float(mb["pred_std"]),
                            "true_std": float(mb["true_std"]),
                            "std_ratio": float(mb["std_ratio"]),
                            "true_min": float(d["y_true_orig"].min()),
                            "true_max": float(d["y_true_orig"].max()),
                        }
                    )
                else:
                    for b, g in binned.groupby("true_bin"):
                        mb = _metric_bundle(g["y_true_orig"], g["y_pred_orig"])
                        rows.append(
                            {
                                "true_bin": int(b),
                                "bin_count": int(len(g)),
                                "mae": float(mb["mae"]),
                                "rmse": float(mb["rmse"]),
                                "bias": float(mb["mean_bias"]),
                                "pred_std": float(mb["pred_std"]),
                                "true_std": float(mb["true_std"]),
                                "std_ratio": float(mb["std_ratio"]),
                                "true_min": float(g["y_true_orig"].min()),
                                "true_max": float(g["y_true_orig"].max()),
                            }
                        )
                pd.DataFrame(rows).sort_values("true_bin").to_csv(
                    out_dir / "target_h2_error_by_true_bin.csv", index=False
                )
            else:
                pd.DataFrame().to_csv(out_dir / "target_h2_top_errors.csv", index=False)
                pd.DataFrame().to_csv(out_dir / "target_h2_error_by_true_bin.csv", index=False)

    # output role + frac only
    frac_cols = {c for c in STREAM_EDGE_FEATURE_SLOTS if c.startswith("Frac_")}
    of = sup[(sup["stream_role"] == "output") & (sup["property_name"].isin(frac_cols))].copy()
    out = pd.DataFrame([{"scope": "output_frac", **_metric_bundle(of["y_true_orig"], of["y_pred_orig"])}])
    out.to_csv(out_dir / "output_frac_diagnostic.csv", index=False)

    # Prediction dispersion by property (true vs pred std + ratio).
    disp_rows: list[dict[str, Any]] = []
    for prop, g in sup.groupby("property_name"):
        tstd = float(g["y_true_orig"].astype(float).std(ddof=0))
        pstd = float(g["y_pred_orig"].astype(float).std(ddof=0))
        disp_rows.append(
            {
                "property_name": str(prop),
                "true_std": tstd,
                "pred_std": pstd,
                "std_ratio": _safe_ratio(pstd, tstd),
                "n_samples": int(len(g)),
            }
        )
    pd.DataFrame(disp_rows).sort_values("property_name").to_csv(
        out_dir / "prediction_dispersion_by_property.csv", index=False
    )


def write_baseline_comparison(
    *,
    out_dir: Path,
    edge_predictions: pd.DataFrame,
    current_metrics_by_split: dict[str, dict[str, float]],
) -> dict[str, Any]:
    sup = edge_predictions[edge_predictions["y_edge_mask"] > 0.0].copy()
    train = sup[sup["split"] == "train"].copy()
    val_test = sup[sup["split"].isin(["val", "test"])].copy()

    prop_mean = train.groupby("property_name")["y_true_orig"].mean()
    role_prop_mean = train.groupby(["stream_role", "property_name"])["y_true_orig"].mean()
    key_prop_mean = train.groupby(["main_data_stream_key", "property_name"])["y_true_orig"].mean()
    edge_prop_mean = train.groupby(["canonical_edge_id", "property_name"])["y_true_orig"].mean()

    def _apply_baseline(df: pd.DataFrame, name: str) -> pd.DataFrame:
        out = df.copy()
        if name == "global_train_property_mean":
            out["y_pred_baseline"] = out["property_name"].map(prop_mean)
        elif name == "stream_role_property_mean":
            out["y_pred_baseline"] = [
                role_prop_mean.get((r, p), prop_mean.get(p, 0.0))
                for r, p in zip(out["stream_role"], out["property_name"])
            ]
        elif name == "stream_key_property_mean":
            out["y_pred_baseline"] = [
                key_prop_mean.get((k, p), prop_mean.get(p, 0.0))
                for k, p in zip(out["main_data_stream_key"], out["property_name"])
            ]
        else:
            out["y_pred_baseline"] = [
                edge_prop_mean.get((c, p), prop_mean.get(p, 0.0))
                for c, p in zip(out["canonical_edge_id"], out["property_name"])
            ]
        return out

    baseline_names = [
        "global_train_property_mean",
        "stream_role_property_mean",
        "stream_key_property_mean",
        "canonical_edge_property_mean",
    ]
    baseline_payload: dict[str, Any] = {"current_model": current_metrics_by_split}
    rows_bp: list[dict[str, Any]] = []
    rows_bs: list[dict[str, Any]] = []
    rows_ba: list[dict[str, Any]] = []

    for bn in baseline_names:
        bdf = _apply_baseline(val_test, bn)
        by_split = {}
        for split, g in bdf.groupby("split"):
            mb = _metric_bundle(g["y_true_orig"], g["y_pred_baseline"])
            by_split[split] = {
                "edge_all_mae_orig": float(mb["mae"]),
                "edge_all_rmse_orig": float(mb["rmse"]),
                "edge_all_r2_orig": float(mb["r2"]),
                "true_std": float(mb["true_std"]),
                "r2_unstable": bool(mb["r2_unstable"]),
            }
            # by property
            for p, gp in g.groupby("property_name"):
                pm = _metric_bundle(gp["y_true_orig"], gp["y_pred_baseline"])
                rows_bp.append({"baseline": bn, "split": split, "property_name": p, **pm})
            # by stream role
            for r, gr in g.groupby("stream_role"):
                rm = _metric_bundle(gr["y_true_orig"], gr["y_pred_baseline"])
                rows_bs.append({"baseline": bn, "split": split, "stream_role": r, **rm})
            # answer edges
            ans = g[g["is_answer_edge"] == 1]
            for task in ("target_h2", "tailgas_co2"):
                gt = ans[ans["answer_task_name"].str.contains(task, regex=False)]
                tm = _metric_bundle(gt["y_true_orig"], gt["y_pred_baseline"])
                rows_ba.append({"baseline": bn, "split": split, "task_name": task, **tm})
                by_split[split][f"{task}_mae_orig"] = float(tm["mae"])
                by_split[split][f"{task}_rmse_orig"] = float(tm["rmse"])
                by_split[split][f"{task}_r2_orig"] = float(tm["r2"])
                by_split[split][f"{task}_true_std"] = float(tm["true_std"])
                by_split[split][f"{task}_r2_unstable"] = bool(tm["r2_unstable"])
        baseline_payload[bn] = by_split

    # Auto win summary (test split)
    wins: dict[str, Any] = {}
    cm_test = current_metrics_by_split.get("test", {})
    for bn in baseline_names:
        b_test = baseline_payload.get(bn, {}).get("test", {})
        wins[bn] = {
            "edge_all_mae_win": float(cm_test.get("edge_all_mae_orig", 1e30)) < float(b_test.get("edge_all_mae_orig", 1e30)),
            "edge_all_rmse_win": float(cm_test.get("edge_all_rmse_orig", 1e30)) < float(b_test.get("edge_all_rmse_orig", 1e30)),
            "edge_all_r2_ref": [float(cm_test.get("edge_all_r2_orig", 0.0)), float(b_test.get("edge_all_r2_orig", 0.0))],
            "edge_all_r2_unstable": bool(cm_test.get("r2_unstable", False) or b_test.get("r2_unstable", False)),
        }
    baseline_payload["win_summary"] = wins

    (out_dir / "baseline_metrics.json").write_text(json.dumps(baseline_payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows_bp).to_csv(out_dir / "baseline_metrics_by_property.csv", index=False)
    pd.DataFrame(rows_bs).to_csv(out_dir / "baseline_metrics_by_stream_role.csv", index=False)
    pd.DataFrame(rows_ba).to_csv(out_dir / "baseline_answer_edge_metrics.csv", index=False)
    return baseline_payload


def compute_answer_metrics_v2(
    *,
    out_dir: Path,
    edge_predictions: pd.DataFrame,
    target_answer_edges_path: Path,
    base_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute answer metrics using target_column-aware primary amount metrics."""
    sup = edge_predictions[edge_predictions["y_edge_mask"] > 0.0].copy()
    answers = pd.read_csv(target_answer_edges_path, dtype={"process_id": int})
    p1 = answers[answers["process_id"] == 1].copy()
    tasks = [t for t in ("target_h2", "tailgas_co2") if t in set(p1["task_name"].astype(str))]
    if not tasks:
        raise RuntimeError("No Process1 target tasks found for v2 metrics.")

    mapping_rows: list[dict[str, Any]] = []
    for task in tasks:
        tc = str(p1[p1["task_name"] == task].iloc[0]["target_column"])
        spec = resolve_answer_metric_spec(task_name=task, target_column=tc)
        mapping_rows.append(
            {
                "task_name": task,
                "target_column": tc,
                "primary_metric": spec.primary_metric,
                "formula": spec.formula,
                "auxiliary_metric": spec.auxiliary_metric,
                "scale_factor": spec.scale_factor,
            }
        )
    mapping_df = pd.DataFrame(mapping_rows)

    work = sup[
        (sup["process_id"] == "Process1")
        & (sup["is_answer_edge"] == 1)
        & (sup["answer_task_name"].isin(tasks))
    ].copy()
    key_cols = ["split", "process_id", "sample_id", "canonical_edge_id", "answer_task_name"]
    piv = work.pivot_table(
        index=key_cols,
        columns="property_name",
        values=["y_true_orig", "y_pred_orig"],
        aggfunc="first",
    ).reset_index()
    piv.columns = [
        "_".join([c for c in col if c]).strip("_") if isinstance(col, tuple) else col
        for col in piv.columns
    ]

    required = ["Mole_Flow", "Frac_H2", "Frac_CO2"]
    missing_cols = [c for c in required if f"y_true_orig_{c}" not in piv.columns or f"y_pred_orig_{c}" not in piv.columns]
    if missing_cols:
        raise RuntimeError(f"Cannot compute v2 derived metrics, missing properties: {missing_cols}")

    rows_pred: list[dict[str, Any]] = []
    rows_metric: list[dict[str, Any]] = []
    metric_payload: dict[str, Any] = dict(base_metrics or {})
    metric_payload["answer_metric_mapping_v2"] = mapping_rows

    for task in tasks:
        spec_row = mapping_df[mapping_df["task_name"] == task].iloc[0]
        sf = float(spec_row["scale_factor"])
        aux = str(spec_row["auxiliary_metric"])
        sub = piv[piv["answer_task_name"] == task].copy()
        if task == "target_h2":
            true_amt = sf * sub["y_true_orig_Mole_Flow"] * sub["y_true_orig_Frac_H2"]
            pred_amt = sf * sub["y_pred_orig_Mole_Flow"] * sub["y_pred_orig_Frac_H2"]
        else:
            true_amt = sf * sub["y_true_orig_Mole_Flow"] * sub["y_true_orig_Frac_CO2"]
            pred_amt = sf * sub["y_pred_orig_Mole_Flow"] * sub["y_pred_orig_Frac_CO2"]
        sub["true_amount"] = true_amt
        sub["pred_amount"] = pred_amt
        sub["abs_error_amount"] = (pred_amt - true_amt).abs()
        sub["error_amount"] = pred_amt - true_amt
        rows_pred.extend(sub.to_dict(orient="records"))

        frac_rows = work[(work["answer_task_name"] == task) & (work["property_name"] == aux)].copy()
        for split in ("train", "val", "test"):
            gs = sub[sub["split"] == split]
            if len(gs):
                mb = _metric_bundle(gs["true_amount"], gs["pred_amount"])
                corr = gs["true_amount"].corr(gs["pred_amount"]) if len(gs) > 1 else 0.0
                rows_metric.append(
                    {
                        "task_name": task,
                        "split": split,
                        "metric_type": "amount_primary",
                        "metric_name": "amount",
                        "mae": float(mb["mae"]),
                        "rmse": float(mb["rmse"]),
                        "r2": float(mb["r2"]),
                        "std_ratio": float(mb["std_ratio"]),
                        "corr": float(corr) if pd.notna(corr) else 0.0,
                    }
                )
                metric_payload[f"{split}/{task}_amount_mae"] = float(mb["mae"])
                metric_payload[f"{split}/{task}_amount_rmse"] = float(mb["rmse"])
                metric_payload[f"{split}/{task}_amount_r2"] = float(mb["r2"])
                metric_payload[f"{split}/{task}_amount_std_ratio"] = float(mb["std_ratio"])
                metric_payload[f"{split}/{task}_amount_corr"] = float(corr) if pd.notna(corr) else 0.0
            fs = frac_rows[frac_rows["split"] == split]
            if len(fs):
                fm = _metric_bundle(fs["y_true_orig"], fs["y_pred_orig"])
                rows_metric.append(
                    {
                        "task_name": task,
                        "split": split,
                        "metric_type": "fraction_aux",
                        "metric_name": aux,
                        "mae": float(fm["mae"]),
                        "rmse": float(fm["rmse"]),
                        "r2": float(fm["r2"]),
                        "std_ratio": float(fm["std_ratio"]),
                        "corr": float(fs["y_true_orig"].corr(fs["y_pred_orig"])) if len(fs) > 1 else 0.0,
                    }
                )
                metric_payload[f"{split}/{task}_frac_mae"] = float(fm["mae"])
                metric_payload[f"{split}/{task}_frac_rmse"] = float(fm["rmse"])
                metric_payload[f"{split}/{task}_frac_r2"] = float(fm["r2"])
                metric_payload[f"{split}/{task}_frac_std_ratio"] = float(fm["std_ratio"])

    pd.DataFrame(rows_metric).to_csv(out_dir / "answer_edge_metrics_v2.csv", index=False)
    pd.DataFrame(rows_pred).to_csv(out_dir / "derived_answer_metrics_v2.csv", index=False)
    (out_dir / "metrics_v2.json").write_text(json.dumps(metric_payload, indent=2), encoding="utf-8")
    return metric_payload


def _normalize_stream_key(value: Any) -> str:
    return _normalize_main_data_stream_key(value)


def compute_target_metrics_v4(
    *,
    out_dir: Path,
    edge_predictions: pd.DataFrame,
    target_stream_targets_path: Path,
    base_metrics: dict[str, Any] | None = None,
    train_cfg: TrainConfig | None = None,
) -> dict[str, Any]:
    """Compute all v4 target_id metrics from original-scale edge_predictions (see target_v4_metrics.py)."""
    split_name = ""
    if (
        edge_predictions is not None
        and not edge_predictions.empty
        and "split" in edge_predictions.columns
    ):
        splits = edge_predictions["split"].astype(str).dropna().unique().tolist()
        split_name = splits[0] if len(splits) == 1 else ""

    specs_by_process = load_target_stream_targets_v4(target_stream_targets_path)
    result = compute_target_v4_metrics_from_original_scale(
        edge_predictions=edge_predictions,
        target_specs=specs_by_process,
        split_name=split_name,
        base_metrics=augment_metrics_with_legacy_aliases(dict(base_metrics or {})),
        train_cfg=train_cfg,
    )
    write_target_v4_split_artifacts(out_dir, result, write_predictions=True)

    pred_df = result.predictions_df
    if not pred_df.empty:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(
                pred_df["target_true_value"].astype(float),
                pred_df["target_pred_value"].astype(float),
                s=8,
                alpha=0.35,
            )
            tmin = float(min(pred_df["target_true_value"].min(), pred_df["target_pred_value"].min()))
            tmax = float(max(pred_df["target_true_value"].max(), pred_df["target_pred_value"].max()))
            ax.plot([tmin, tmax], [tmin, tmax], "r--", linewidth=1.0)
            ax.set_xlabel("true")
            ax.set_ylabel("pred")
            ax.set_title("Target v4 true vs pred (all)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "target_true_vs_pred_scatter_v4.png", dpi=150)
            plt.close(fig)

            for target_id, g in pred_df.groupby("target_id"):
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(
                    g["target_true_value"].astype(float),
                    g["target_pred_value"].astype(float),
                    s=10,
                    alpha=0.4,
                )
                gmin = float(min(g["target_true_value"].min(), g["target_pred_value"].min()))
                gmax = float(max(g["target_true_value"].max(), g["target_pred_value"].max()))
                ax.plot([gmin, gmax], [gmin, gmax], "r--", linewidth=1.0)
                ax.set_xlabel("true")
                ax.set_ylabel("pred")
                ax.set_title(f"Target v4 true vs pred: {target_id}")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                safe_name = str(target_id).replace("/", "_").replace("\\", "_")
                fig.savefig(out_dir / f"target_scatter_{safe_name}.png", dpi=150)
                plt.close(fig)
        except Exception:
            pass

    return result.payload
