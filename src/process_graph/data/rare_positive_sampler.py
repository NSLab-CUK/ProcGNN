from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from process_graph.data.stream_keys import (
    canonicalize_process_id,
    canonicalize_stream_key,
)


@dataclass(frozen=True)
class RarePositiveWeightResult:
    weights: torch.Tensor
    summary: dict[str, Any]


def _id_key(value: Any) -> str:
    return canonicalize_stream_key(value)


def _process_number(value: Any) -> int:
    normalized = canonicalize_process_id(value)
    if not normalized:
        raise ValueError(f"Cannot resolve process number from {value!r}.")
    return int(normalized[1:])


def _base_dataset_indices(dataset: Dataset) -> tuple[Dataset, list[int]]:
    indices = list(range(len(dataset)))
    base = dataset
    while isinstance(base, Subset):
        indices = [int(base.indices[index]) for index in indices]
        base = base.dataset
    return base, indices


def _target_stream_keys_by_process(
    *,
    project_root: Path,
    data_cfg: Any,
) -> dict[int, set[str]]:
    target_path = project_root / "data/reference/v4/target_stream_targets.csv"
    canonical_dir = project_root / str(
        getattr(data_cfg, "canonical_graph_spec_v3_dir", "data/reference/v3")
    )
    edge_path = canonical_dir / "canonical_edges.csv"
    if not target_path.is_file():
        raise FileNotFoundError(f"Rare-positive target mapping not found: {target_path}")
    if not edge_path.is_file():
        raise FileNotFoundError(f"Rare-positive canonical edge table not found: {edge_path}")

    targets = pd.read_csv(target_path, usecols=["process_id", "canonical_answer_edge_id"])
    edges = pd.read_csv(
        edge_path,
        usecols=["process_id", "canonical_edge_id", "main_data_stream_key"],
    )
    target_edge_ids = {
        (int(row.process_id), str(row.canonical_answer_edge_id))
        for row in targets.itertuples(index=False)
    }
    result: dict[int, set[str]] = {}
    for row in edges.itertuples(index=False):
        key = (int(row.process_id), str(row.canonical_edge_id))
        if key not in target_edge_ids:
            continue
        stream_key = _id_key(row.main_data_stream_key)
        if stream_key:
            result.setdefault(int(row.process_id), set()).add(stream_key)
    return result


def compute_rare_positive_sample_weights(
    dataset: Dataset,
    *,
    sampler_cfg: Any,
    project_root: Path,
) -> RarePositiveWeightResult:
    if str(getattr(sampler_cfg, "mode", "")).strip().lower() != "any_target_edge":
        raise ValueError("Rare-positive sampler currently supports only mode='any_target_edge'.")
    components = [str(name) for name in getattr(sampler_cfg, "components", [])]
    threshold = float(getattr(sampler_cfg, "positive_threshold", 1.0e-4))
    max_weight = float(getattr(sampler_cfg, "max_weight", 10.0))
    if not components:
        raise ValueError("Rare-positive sampler components must not be empty.")

    base, base_indices = _base_dataset_indices(dataset)
    frame = getattr(base, "frame", None)
    data_cfg = getattr(base, "data_cfg", None)
    if not isinstance(frame, pd.DataFrame) or data_cfg is None:
        raise TypeError(
            "Rare-positive sampler requires ProcessGraphTabularDataset or a Subset wrapping it."
        )
    if str(getattr(data_cfg, "task_mode", "")).strip().lower() != "edge_all":
        raise ValueError("Rare-positive sampler is supported only for data.task_mode='edge_all'.")
    process_column = str(getattr(data_cfg, "process_id_column", "process_id"))
    if process_column not in frame.columns or "ID" not in frame.columns:
        raise KeyError(
            f"Rare-positive sampler requires dataset columns {process_column!r} and 'ID'."
        )

    selected = frame.iloc[base_indices][[process_column, "ID"]].copy().reset_index(drop=True)
    selected["_process_num"] = selected[process_column].map(_process_number)
    selected["_sample_key"] = selected["ID"].map(_id_key)
    if bool(selected["_sample_key"].eq("").any()):
        raise ValueError("Rare-positive sampler found empty sample IDs in the train dataset.")

    target_streams = _target_stream_keys_by_process(
        project_root=project_root,
        data_cfg=data_cfg,
    )
    positive = pd.DataFrame(False, index=range(len(selected)), columns=components)
    missing_sample_count = 0
    stream_dir = project_root / str(
        getattr(data_cfg, "stream_data_dir", "data/main_data_Streams")
    )

    for process_num, sample_rows in selected.groupby("_process_num", sort=True):
        process_num = int(process_num)
        streams = target_streams.get(process_num, set())
        if not streams:
            raise RuntimeError(
                f"Rare-positive sampler resolved no target streams for process {process_num}."
            )
        stream_path = stream_dir / f"{process_num}.Process_Streams.csv"
        if not stream_path.is_file():
            raise FileNotFoundError(f"Rare-positive Process_Streams file not found: {stream_path}")
        usecols = ["ID", "Stream_Name", *components]
        stream_frame = pd.read_csv(stream_path, usecols=usecols)
        stream_frame["_sample_key"] = stream_frame["ID"].map(_id_key)
        stream_frame["_stream_key"] = stream_frame["Stream_Name"].map(_id_key)
        wanted_ids = set(sample_rows["_sample_key"])
        stream_frame = stream_frame[
            stream_frame["_sample_key"].isin(wanted_ids)
            & stream_frame["_stream_key"].isin(streams)
        ]
        maxima = stream_frame.groupby("_sample_key", sort=False)[components].max()
        flags = maxima.gt(threshold)
        for row_index, sample_key in zip(sample_rows.index, sample_rows["_sample_key"]):
            if sample_key not in flags.index:
                missing_sample_count += 1
                continue
            positive.loc[int(row_index), components] = flags.loc[sample_key, components].to_numpy(
                dtype=bool
            )

    weights = torch.ones(len(selected), dtype=torch.double)
    component_summary: dict[str, dict[str, float | int]] = {}
    for component in components:
        flags = torch.as_tensor(positive[component].to_numpy(dtype=bool))
        positive_count = int(flags.sum().item())
        frequency = positive_count / max(1, len(selected))
        factor = 1.0 / math.sqrt(frequency + 1.0e-12)
        weights[flags] *= factor
        component_summary[component] = {
            "positive_count": positive_count,
            "positive_frequency": float(frequency),
            "positive_weight_factor": float(factor),
        }
    weights.clamp_(min=1.0, max=max_weight)

    process_weight = pd.DataFrame(
        {
            "process_id": selected["_process_num"].astype(int),
            "weight": weights.numpy(),
        }
    ).groupby("process_id", sort=True)["weight"].sum()
    total_weight = float(process_weight.sum())
    process_share = {
        str(int(process_id)): float(value / total_weight)
        for process_id, value in process_weight.items()
    }
    summary = {
        "enabled": True,
        "mode": "any_target_edge",
        "num_samples": int(len(selected)),
        "components": component_summary,
        "positive_threshold": threshold,
        "max_weight": max_weight,
        "missing_sample_count": int(missing_sample_count),
        "weight_min": float(weights.min().item()) if weights.numel() else math.nan,
        "weight_mean": float(weights.mean().item()) if weights.numel() else math.nan,
        "weight_max": float(weights.max().item()) if weights.numel() else math.nan,
        "expected_process_sampling_share": process_share,
    }
    return RarePositiveWeightResult(weights=weights, summary=summary)
