#!/usr/bin/env python3
"""Precompute node-level true energy consistency masks.

The output CSV can be passed through
data.node_energy_consistency_cache_path. Dataset loading will OR rows marked
exclude_energy=1 into node_balance_exclude_energy.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from process_graph.constants import UNIT_TO_IDX  # noqa: E402
from process_graph.data.tabular_dataset import ProcessGraphTabularDataset  # noqa: E402
from process_graph.experiment.loaders import load_experiment_config  # noqa: E402
from process_graph.experiment.node_balance_pi import resolve_node_balance_config  # noqa: E402

BOUNDARY_NODE_TYPES = {"input_virtual", "output_virtual"}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _fmt_bool(value: bool) -> str:
    return "true" if bool(value) else "false"


def _unit_names(x_unit: Iterable[int]) -> list[str]:
    rev = {int(v): str(k) for k, v in UNIT_TO_IDX.items()}
    return [rev.get(int(v), "") for v in x_unit]


def _node_sum(values: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    out = torch.zeros((num_nodes, 1), dtype=values.dtype)
    if values.numel():
        out.index_add_(0, index.to(dtype=torch.long), values.reshape(-1, 1))
    return out


def _as_node_column(values: object, num_nodes: int, default: float) -> torch.Tensor:
    if values is None:
        return torch.full((num_nodes, 1), float(default), dtype=torch.float64)
    vals = list(values)  # type: ignore[arg-type]
    if len(vals) != num_nodes:
        return torch.full((num_nodes, 1), float(default), dtype=torch.float64)
    return torch.tensor(vals, dtype=torch.float64).reshape(num_nodes, 1)


def _compute_record_rows(record: object, *, threshold: float, node_cfg: dict[str, object]) -> list[dict[str, object]]:
    graph = record.graph  # type: ignore[attr-defined]
    meta = dict(record.sample_meta)  # type: ignore[attr-defined]
    cols = list(graph.edge_target_columns or [])
    if "Mass_Flow" not in cols or "Enthalpy" not in cols:
        raise RuntimeError(
            "node energy consistency precompute requires Mass_Flow and Enthalpy in edge_target_columns."
        )
    mass_idx = cols.index("Mass_Flow")
    h_idx = cols.index("Enthalpy")
    y = torch.tensor(graph.y_edge_true, dtype=torch.float64)
    edge_mask = torch.tensor(graph.y_edge_mask, dtype=torch.bool).reshape(-1, 1)
    mass = y[:, mass_idx : mass_idx + 1]
    h = y[:, h_idx : h_idx + 1]

    h_basis = str(node_cfg.get("h_basis", "mass_specific"))
    if h_basis != "mass_specific":
        raise RuntimeError(f"precompute currently supports h_basis='mass_specific', got {h_basis!r}.")
    hflow = mass * h
    active_true = mass > float(node_cfg.get("zero_flow_threshold", 1.0e-12))
    if bool(node_cfg.get("zero_flow_nan_enthalpy_as_zero", True)):
        hflow = torch.where(active_true | torch.isfinite(hflow), hflow, torch.zeros_like(hflow))

    edge_index = torch.tensor(graph.edge_index, dtype=torch.long)
    src = edge_index[0]
    dst = edge_index[1]
    num_nodes = len(graph.node_names)
    num_edges = int(y.shape[0])
    if int(src.numel()) != num_edges or int(dst.numel()) != num_edges:
        raise RuntimeError("edge_index and y_edge_true edge count mismatch.")

    unit_names = _unit_names(graph.x_unit)
    in_degree = torch.zeros(num_nodes, dtype=torch.long)
    out_degree = torch.zeros(num_nodes, dtype=torch.long)
    ones = torch.ones(num_edges, dtype=torch.long)
    if num_edges:
        in_degree.index_add_(0, dst, ones)
        out_degree.index_add_(0, src, ones)
    touched = (in_degree + out_degree) > 0
    boundary = torch.tensor([name in BOUNDARY_NODE_TYPES for name in unit_names], dtype=torch.bool)
    internal = touched & (in_degree > 0) & (out_degree > 0) & ~boundary

    excl_energy = _as_node_column(getattr(graph, "node_balance_exclude_energy", None), num_nodes, 0.0) > 0.5
    energy_base = internal.reshape(-1, 1) & ~excl_energy
    valid_types = tuple(str(x) for x in node_cfg.get("node_energy_valid_unit_types", ()) or ())
    exclude_types = tuple(str(x) for x in node_cfg.get("node_energy_exclude_unit_types", ()) or ())
    if valid_types:
        valid_mask = torch.tensor([name in valid_types for name in unit_names], dtype=torch.bool).reshape(-1, 1)
        energy_base &= valid_mask
    if exclude_types:
        exclude_mask = torch.tensor([name in exclude_types for name in unit_names], dtype=torch.bool).reshape(-1, 1)
        energy_base &= ~exclude_mask

    q = _as_node_column(getattr(graph, "node_q", None), num_nodes, 0.0)
    w = _as_node_column(getattr(graph, "node_w", None), num_nodes, 0.0)
    qw_valid = _as_node_column(getattr(graph, "node_qw_valid_mask", None), num_nodes, 1.0)
    if not bool(node_cfg.get("node_energy_use_qw", False)):
        q.zero_()
        w.zero_()
        qw_valid.fill_(1.0)

    hflow_valid = torch.isfinite(hflow) & edge_mask
    safe_h = torch.where(hflow_valid, hflow, torch.zeros_like(hflow))
    true_in = _node_sum(safe_h, dst, num_nodes)
    true_out = _node_sum(safe_h, src, num_nodes)
    scale = _node_sum(safe_h.abs(), dst, num_nodes) + _node_sum(safe_h.abs(), src, num_nodes) + q.abs() + w.abs()
    scale_floor = node_cfg.get("node_energy_scale_floor")
    if scale_floor is not None:
        scale = torch.maximum(scale, torch.full_like(scale, float(scale_floor)))
    eps = float(node_cfg.get("node_energy_epsilon", 1.0e-8))
    residual = true_in + q + w - true_out
    if bool(node_cfg.get("node_energy_relative", True)):
        residual = residual / (scale + eps)
    structural_valid = energy_base & (qw_valid > 0.5)

    rows: list[dict[str, object]] = []
    for i, node_name in enumerate(graph.node_names):
        valid = bool(structural_valid[i].item())
        value = float(abs(residual[i].item())) if math.isfinite(float(residual[i].item())) else float("nan")
        consistent = bool(valid and math.isfinite(value) and value <= threshold)
        exclude = bool(valid and (not math.isfinite(value) or value > threshold))
        rows.append(
            {
                "process_id": str(meta.get("process_id", graph.process_id)),
                "sample_id": str(meta.get("sample_id", "")),
                "node_name": str(node_name),
                "node_index": int(i),
                "unit_type": unit_names[i],
                "true_energy_residual": float(residual[i].item()),
                "true_energy_residual_abs": value,
                "threshold": float(threshold),
                "structural_valid": _fmt_bool(valid),
                "energy_consistent": _fmt_bool(consistent),
                "exclude_energy": int(exclude),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute node-level true energy residual masks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split-manifest", action="append", default=[], help="Split manifest to scan. Repeat for train/val/test.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=1.0e-2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if not math.isfinite(float(args.threshold)) or float(args.threshold) < 0.0:
        raise ValueError("--threshold must be finite and non-negative.")

    experiment = load_experiment_config(_resolve(args.config))
    data_cfg = copy.deepcopy(experiment.data)
    data_cfg.node_energy_consistency_cache_path = ""
    data_cfg.normalize_x_oper = False
    data_cfg.normalize_targets = False
    node_cfg = resolve_node_balance_config(experiment.train)
    node_cfg["h_basis"] = str(getattr(experiment.train, "h_basis", "mass_specific"))

    manifests = [Path(p) for p in args.split_manifest]
    if not manifests:
        raw = str(getattr(data_cfg, "train_split_manifest_path", "") or "").strip()
        if raw:
            manifests = [Path(raw)]
    if not manifests:
        raise ValueError("Provide --split-manifest or set data.train_split_manifest_path in config.")

    all_rows: list[dict[str, object]] = []
    for manifest in manifests:
        dataset = ProcessGraphTabularDataset(
            Path(data_cfg.train_data_path),
            data_cfg,
            experiment.project_root,
            split_manifest=_resolve(manifest),
        )
        n = len(dataset) if int(args.limit) <= 0 else min(len(dataset), int(args.limit))
        for i in range(n):
            all_rows.extend(_compute_record_rows(dataset[i], threshold=float(args.threshold), node_cfg=node_cfg))
        print(f"[node-energy-precompute] manifest={manifest} samples={n} rows={len(all_rows)}", flush=True)

    out = _resolve(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(out, index=False)
    valid = frame[frame["structural_valid"].astype(str).str.lower() == "true"]
    excluded = int(pd.to_numeric(frame["exclude_energy"], errors="coerce").fillna(0).sum())
    print(
        f"[node-energy-precompute] wrote {out} total_rows={len(frame)} "
        f"structural_valid={len(valid)} exclude_energy={excluded}",
        flush=True,
    )


if __name__ == "__main__":
    main()
