from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
from torch.utils.data import Dataset, Subset

from process_graph.data.tabular_dataset import GraphSampleRecord
from process_graph.data.stream_keys import (
    canonicalize_process_id,
    canonicalize_stream_key,
)


@dataclass(frozen=True)
class TargetEdgeAugmentationRecord:
    original_index: int
    process_id: int
    sample_key: str
    canonical_edge_id: str
    positive_components: tuple[str, ...]
    high_positive_components: tuple[str, ...]
    component_values: tuple[tuple[str, float], ...]
    copy_index: int
    selected_property: str | None = None
    selected_bin: str | None = None
    selected_scope: str | None = None


@dataclass(frozen=True)
class TargetEdgeAugmentationBuild:
    dataset: "TargetEdgeAugmentedDataset"
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


def _target_edges_by_process(
    *,
    project_root: Path,
    data_cfg: Any,
) -> tuple[dict[int, list[tuple[str, str]]], int]:
    target_path = project_root / "data/reference/v4/target_stream_targets.csv"
    canonical_dir = project_root / str(
        getattr(data_cfg, "canonical_graph_spec_v3_dir", "data/reference/v3")
    )
    edge_path = canonical_dir / "canonical_edges.csv"
    if not target_path.is_file():
        raise FileNotFoundError(f"Target-edge augmentation mapping not found: {target_path}")
    if not edge_path.is_file():
        raise FileNotFoundError(f"Canonical edge table not found: {edge_path}")

    targets = pd.read_csv(
        target_path,
        usecols=["process_id", "canonical_answer_edge_id"],
    )
    target_ids = {
        (int(row.process_id), str(row.canonical_answer_edge_id).strip())
        for row in targets.itertuples(index=False)
    }
    edges = pd.read_csv(
        edge_path,
        usecols=["process_id", "canonical_edge_id", "main_data_stream_key"],
    )
    result: dict[int, list[tuple[str, str]]] = {}
    resolved: set[tuple[int, str]] = set()
    for row in edges.itertuples(index=False):
        process_id = int(row.process_id)
        edge_id = str(row.canonical_edge_id).strip()
        key = (process_id, edge_id)
        if key not in target_ids or key in resolved:
            continue
        stream_key = _id_key(row.main_data_stream_key)
        if not stream_key:
            continue
        result.setdefault(process_id, []).append((edge_id, stream_key))
        resolved.add(key)
    missing = sorted(target_ids - resolved)
    if missing:
        raise RuntimeError(
            "Target-edge augmentation could not resolve canonical stream keys for "
            f"{len(missing)} target edges: {missing[:5]}."
        )
    for process_id in result:
        result[process_id].sort()
    return result, len(resolved)


def _edge_display_names_by_process(
    *,
    project_root: Path,
    data_cfg: Any,
) -> dict[tuple[int, str], str]:
    """Return human-readable edge/stream labels for augmentation filters/audits."""
    result: dict[tuple[int, str], str] = {}
    canonical_dir = project_root / str(
        getattr(data_cfg, "canonical_graph_spec_v3_dir", "data/reference/v3")
    )
    edge_path = canonical_dir / "canonical_edges.csv"
    if edge_path.is_file():
        usecols = [
            "process_id",
            "canonical_edge_id",
            "main_data_stream_key",
            "stream_name_norm",
        ]
        edges = pd.read_csv(edge_path, usecols=[c for c in usecols if c in pd.read_csv(edge_path, nrows=0).columns])
        for row in edges.itertuples(index=False):
            process_id = int(getattr(row, "process_id"))
            edge_id = str(getattr(row, "canonical_edge_id")).strip()
            candidates = []
            for column in ("stream_name_norm", "main_data_stream_key"):
                if hasattr(row, column):
                    value = str(getattr(row, column)).strip()
                    if value and value.lower() not in {"nan", "<na>"}:
                        candidates.append(value)
            result[(process_id, edge_id)] = candidates[0] if candidates else edge_id

    target_path = project_root / "data/reference/v4/target_stream_targets.csv"
    if target_path.is_file():
        targets = pd.read_csv(
            target_path,
            usecols=["process_id", "canonical_answer_edge_id", "target_stream_node"],
        )
        for row in targets.itertuples(index=False):
            process_id = int(row.process_id)
            edge_id = str(row.canonical_answer_edge_id).strip()
            stream_name = str(row.target_stream_node).strip()
            if stream_name and stream_name.lower() not in {"nan", "<na>"}:
                result[(process_id, edge_id)] = stream_name
    return result


class TargetEdgeAugmentedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        augmentation_records: list[TargetEdgeAugmentationRecord],
    ) -> None:
        self.dataset = dataset
        self.augmentation_records = list(augmentation_records)
        self.original_length = len(dataset)

    def __len__(self) -> int:
        return self.original_length + len(self.augmentation_records)

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < self.original_length:
            return self.dataset[index]

        augmentation = self.augmentation_records[index - self.original_length]
        record = self.dataset[augmentation.original_index]
        if not isinstance(record, GraphSampleRecord):
            raise TypeError(
                "TargetEdgeAugmentedDataset expects GraphSampleRecord items, got "
                f"{type(record).__name__}."
            )
        edge_ids = list(record.graph.canonical_edge_ids)
        try:
            edge_index = edge_ids.index(augmentation.canonical_edge_id)
        except ValueError as exc:
            raise RuntimeError(
                f"Augmented target edge {augmentation.canonical_edge_id!r} is absent from "
                f"sample process={record.graph.process_id!r}."
            ) from exc
        original_mask = list(record.graph.y_edge_mask)
        if len(original_mask) != len(edge_ids):
            raise RuntimeError(
                "Target-edge augmentation found inconsistent canonical_edge_ids/y_edge_mask "
                f"lengths: {len(edge_ids)} != {len(original_mask)}."
            )
        if float(original_mask[edge_index]) <= 0.0:
            raise RuntimeError(
                f"Augmented target edge {augmentation.canonical_edge_id!r} is not supervised "
                "in the original item."
            )
        augmented_mask = [0.0] * len(original_mask)
        augmented_mask[edge_index] = 1.0
        augmented_graph = replace(record.graph, y_edge_mask=augmented_mask)
        sample_meta = dict(record.sample_meta)
        sample_meta.update(
            {
                "is_target_edge_augmented": True,
                "augmented_target_edge_id": augmentation.canonical_edge_id,
                "augmented_positive_components": list(augmentation.positive_components),
                "augmented_selected_property": augmentation.selected_property,
                "augmented_selected_bin": augmentation.selected_bin,
                "augmented_selected_scope": augmentation.selected_scope,
            }
        )
        return replace(record, graph=augmented_graph, sample_meta=sample_meta)


def _edges_by_process(
    *,
    project_root: Path,
    data_cfg: Any,
    target_only: bool,
) -> tuple[dict[int, list[tuple[str, str]]], int]:
    if target_only:
        return _target_edges_by_process(project_root=project_root, data_cfg=data_cfg)

    canonical_dir = project_root / str(
        getattr(data_cfg, "canonical_graph_spec_v3_dir", "data/reference/v3")
    )
    edge_path = canonical_dir / "canonical_edges.csv"
    if not edge_path.is_file():
        raise FileNotFoundError(f"Canonical edge table not found: {edge_path}")
    edges = pd.read_csv(
        edge_path,
        usecols=["process_id", "canonical_edge_id", "main_data_stream_key"],
    )
    result: dict[int, list[tuple[str, str]]] = {}
    for row in edges.itertuples(index=False):
        stream_key = _id_key(row.main_data_stream_key)
        if not stream_key:
            continue
        result.setdefault(int(row.process_id), []).append(
            (str(row.canonical_edge_id).strip(), stream_key)
        )
    for process_id in result:
        result[process_id].sort()
    return result, sum(len(items) for items in result.values())


def build_rare_target_edge_augmented_dataset(
    dataset: Dataset,
    *,
    augmentation_cfg: Any,
    project_root: Path,
) -> TargetEdgeAugmentationBuild:
    base, base_indices = _base_dataset_indices(dataset)
    frame = getattr(base, "frame", None)
    data_cfg = getattr(base, "data_cfg", None)
    if not isinstance(frame, pd.DataFrame) or data_cfg is None:
        raise TypeError(
            "Target-edge augmentation requires ProcessGraphTabularDataset or a Subset "
            "wrapping it."
        )
    if str(getattr(data_cfg, "task_mode", "")).strip().lower() != "edge_all":
        raise ValueError("Target-edge augmentation is supported only for data.task_mode='edge_all'.")

    components = [str(name) for name in augmentation_cfg.components]
    threshold = float(augmentation_cfg.positive_threshold)
    high_thresholds = {
        str(name): float(value)
        for name, value in augmentation_cfg.high_positive_thresholds.items()
    }
    default_factor = int(augmentation_cfg.default_factor)
    high_factor = int(augmentation_cfg.high_positive_factor)
    max_augmented = int(
        math.floor(len(dataset) * float(augmentation_cfg.max_augmented_ratio))
    )
    process_column = str(getattr(data_cfg, "process_id_column", "process_id"))
    if process_column not in frame.columns or "ID" not in frame.columns:
        raise KeyError(
            f"Target-edge augmentation requires columns {process_column!r} and 'ID'."
        )

    selected = frame.iloc[base_indices][[process_column, "ID"]].copy().reset_index(drop=True)
    selected["_dataset_index"] = range(len(selected))
    selected["_process_num"] = selected[process_column].map(_process_number)
    selected["_sample_key"] = selected["ID"].map(_id_key)
    if bool(selected["_sample_key"].eq("").any()):
        raise ValueError("Target-edge augmentation found empty sample IDs in train data.")

    target_edges, target_edge_count = _target_edges_by_process(
        project_root=project_root,
        data_cfg=data_cfg,
    )
    stream_dir = project_root / str(
        getattr(data_cfg, "stream_data_dir", "data/main_data_Streams")
    )
    candidates: list[tuple[int, float, TargetEdgeAugmentationRecord, int]] = []
    candidate_component_counts: Counter[str] = Counter()
    candidate_positive_edge_count = 0
    missing_sample_edge_count = 0
    rng = random.Random(int(augmentation_cfg.seed))

    for process_num, sample_rows in selected.groupby("_process_num", sort=True):
        process_num = int(process_num)
        process_target_edges = target_edges.get(process_num, [])
        if not process_target_edges:
            raise RuntimeError(
                f"Target-edge augmentation resolved no target edges for process {process_num}."
            )
        stream_path = stream_dir / f"{process_num}.Process_Streams.csv"
        if not stream_path.is_file():
            raise FileNotFoundError(f"Process_Streams file not found: {stream_path}")
        stream_frame = pd.read_csv(
            stream_path,
            usecols=["ID", "Stream_Name", *components],
        )
        stream_frame["_sample_key"] = stream_frame["ID"].map(_id_key)
        stream_frame["_stream_key"] = stream_frame["Stream_Name"].map(_id_key)
        wanted_ids = set(sample_rows["_sample_key"])
        wanted_streams = {stream_key for _, stream_key in process_target_edges}
        stream_frame = stream_frame[
            stream_frame["_sample_key"].isin(wanted_ids)
            & stream_frame["_stream_key"].isin(wanted_streams)
        ]
        values_by_sample_stream = stream_frame.groupby(
            ["_sample_key", "_stream_key"],
            sort=False,
        )[components].max()

        for original_index, sample_key in sample_rows[
            ["_dataset_index", "_sample_key"]
        ].itertuples(index=False, name=None):
            sample_key = str(sample_key)
            original_index = int(original_index)
            for edge_id, stream_key in process_target_edges:
                lookup_key = (sample_key, stream_key)
                if lookup_key not in values_by_sample_stream.index:
                    missing_sample_edge_count += 1
                    continue
                values = {
                    component: float(values_by_sample_stream.loc[lookup_key, component])
                    for component in components
                }
                positive_components = tuple(
                    component
                    for component in components
                    if math.isfinite(values[component]) and values[component] > threshold
                )
                if not positive_components:
                    continue
                high_components = tuple(
                    component
                    for component, high_threshold in high_thresholds.items()
                    if component in positive_components and values[component] > high_threshold
                )
                factor = high_factor if high_components else default_factor
                candidate_positive_edge_count += 1
                if "Frac_CO2" in high_components:
                    priority = 0
                elif "Frac_CH4" in positive_components:
                    priority = 1
                elif "Frac_CO" in positive_components:
                    priority = 2
                else:
                    priority = 3
                candidate_component_counts.update(positive_components)
                for copy_index in range(factor):
                    augmentation = TargetEdgeAugmentationRecord(
                        original_index=original_index,
                        process_id=process_num,
                        sample_key=sample_key,
                        canonical_edge_id=edge_id,
                        positive_components=positive_components,
                        high_positive_components=high_components,
                        component_values=tuple(
                            (component, values[component]) for component in components
                        ),
                        copy_index=copy_index,
                    )
                    candidates.append((priority, rng.random(), augmentation, factor))

    candidates.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2].process_id,
            item[2].canonical_edge_id,
            item[2].sample_key,
            item[2].copy_index,
        )
    )
    selected_records = [item[2] for item in candidates[:max_augmented]]
    selected_component_counts: Counter[str] = Counter()
    process_counts: Counter[str] = Counter()
    edge_counts: Counter[str] = Counter()
    for record in selected_records:
        selected_component_counts.update(record.positive_components)
        process_counts[str(record.process_id)] += 1
        edge_counts[record.canonical_edge_id] += 1

    summary = {
        "enabled": True,
        "target_edges_only": True,
        "target_mapping_unique_edge_count": int(target_edge_count),
        "original_sample_count": int(len(dataset)),
        "candidate_positive_edge_count": int(candidate_positive_edge_count),
        "candidate_augmented_item_count_before_cap": int(len(candidates)),
        "augmented_sample_count": int(len(selected_records)),
        "augmented_ratio": float(len(selected_records) / max(1, len(dataset))),
        "max_augmented_ratio": float(augmentation_cfg.max_augmented_ratio),
        "component_candidate_edge_counts": dict(sorted(candidate_component_counts.items())),
        "component_augmented_item_counts": dict(sorted(selected_component_counts.items())),
        "process_augmented_item_counts": dict(
            sorted(process_counts.items(), key=lambda item: int(item[0]))
        ),
        "top_augmented_target_edges": [
            {"canonical_edge_id": edge_id, "count": int(count)}
            for edge_id, count in edge_counts.most_common(20)
        ],
        "missing_sample_target_edge_count": int(missing_sample_edge_count),
        "examples": [
            {
                "original_index": int(record.original_index),
                "process_id": int(record.process_id),
                "sample_key": record.sample_key,
                "canonical_edge_id": record.canonical_edge_id,
                "positive_components": list(record.positive_components),
                "high_positive_components": list(record.high_positive_components),
                "component_values": dict(record.component_values),
            }
            for record in selected_records[:3]
        ],
        "seed": int(augmentation_cfg.seed),
    }
    return TargetEdgeAugmentationBuild(
        dataset=TargetEdgeAugmentedDataset(dataset, selected_records),
        summary=summary,
    )


def _round_edge_quotas(total: int, ratios: dict[str, float]) -> dict[str, int]:
    if total <= 0 or not ratios:
        return {}
    positive = {edge: max(0.0, float(weight)) for edge, weight in ratios.items()}
    weight_sum = sum(positive.values())
    if weight_sum <= 0.0:
        return {}
    raw = {edge: total * weight / weight_sum for edge, weight in positive.items()}
    quotas = {edge: int(math.floor(value)) for edge, value in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(raw, key=lambda edge: (raw[edge] - quotas[edge], edge), reverse=True)
    for edge in order[:remainder]:
        quotas[edge] += 1
    return {edge: quota for edge, quota in quotas.items() if quota > 0}


def _resolve_max_augmented_items(dataset_len: int, augmentation_cfg: Any) -> int:
    """Resolve absolute/relative augmentation cap for the current train dataset."""
    configured_items = int(getattr(augmentation_cfg, "max_augmented_items", 0))
    ratio = getattr(augmentation_cfg, "max_augmented_ratio", None)
    if ratio is None:
        return max(0, configured_items)
    ratio_items = int(math.floor(max(0, int(dataset_len)) * float(ratio)))
    if configured_items > 0:
        return max(0, min(configured_items, ratio_items))
    return max(0, ratio_items)


def _select_from_candidates(
    *,
    candidates: list[TargetEdgeAugmentationRecord],
    quota: int,
    repeat_cap: int,
    rng: random.Random,
    sample_repeat_counts: Counter[int],
    selected_records: list[TargetEdgeAugmentationRecord],
    selected_rule_counts: Counter[str],
    rule_key: str,
    max_total: int,
    edge_id: str | None = None,
) -> int:
    if quota <= 0 or not candidates or len(selected_records) >= max_total:
        return 0
    pool = [
        record
        for record in candidates
        if edge_id is None or record.canonical_edge_id == edge_id
    ]
    if not pool:
        return 0
    rng.shuffle(pool)
    selected = 0
    per_record_copy_counts: Counter[tuple[int, str, str, str]] = Counter()
    cursor = 0
    stalled_passes = 0
    while selected < quota and len(selected_records) < max_total and stalled_passes < 2:
        made_progress = False
        for _ in range(len(pool)):
            candidate = pool[cursor % len(pool)]
            cursor += 1
            if sample_repeat_counts[int(candidate.original_index)] >= repeat_cap:
                continue
            copy_key = (
                int(candidate.original_index),
                candidate.sample_key,
                candidate.canonical_edge_id,
                rule_key,
            )
            copy_index = per_record_copy_counts[copy_key]
            per_record_copy_counts[copy_key] += 1
            selected_records.append(replace(candidate, copy_index=copy_index))
            sample_repeat_counts[int(candidate.original_index)] += 1
            selected_rule_counts[rule_key] += 1
            selected += 1
            made_progress = True
            if selected >= quota or len(selected_records) >= max_total:
                break
        stalled_passes = 0 if made_progress else stalled_passes + 1
    return selected


def _rule_edge_filter(rule: dict[str, Any]) -> set[str]:
    values = rule.get("edge_ids", rule.get("canonical_edge_ids", [])) or []
    if isinstance(values, str):
        values = [values]
    return {str(value).strip() for value in values if str(value).strip()}


def _rule_stream_tokens(rule: dict[str, Any]) -> tuple[str, ...]:
    values = rule.get("stream_contains", rule.get("stream_name_contains", [])) or []
    if isinstance(values, str):
        values = [values]
    return tuple(str(value).strip().lower() for value in values if str(value).strip())


def _matches_adaptive_rule(
    *,
    value: float,
    row: dict[str, Any],
    rule: dict[str, Any],
    prop: str,
    mass_thresholds: dict[str, tuple[float, float]],
) -> tuple[bool, Any, Any]:
    edge_filter = _rule_edge_filter(rule)
    if edge_filter and str(row["canonical_edge_id"]) not in edge_filter:
        return False, "", ""
    stream_tokens = _rule_stream_tokens(rule)
    if stream_tokens:
        stream_name = str(row.get("stream_name", "")).lower()
        edge_id = str(row.get("canonical_edge_id", "")).lower()
        if not any(token in stream_name or token in edge_id for token in stream_tokens):
            return False, "", ""
    if prop == "Mass_Flow":
        lo, hi = mass_thresholds[str(rule["name"])]
        return bool(value >= lo and value <= hi), lo, hi
    lo = float(rule["min_value"])
    hi = float(rule["max_value"])
    mode = str(rule.get("match_mode", "positive")).strip().lower()
    if mode in {"negative", "zero", "near_zero"}:
        matched = value >= lo and value < hi
    else:
        matched = value > lo and value <= hi
    return bool(matched), lo, hi


def build_adaptive_target_edge_augmented_dataset(
    dataset: Dataset,
    *,
    augmentation_cfg: Any,
    project_root: Path,
) -> TargetEdgeAugmentationBuild:
    base, base_indices = _base_dataset_indices(dataset)
    frame = getattr(base, "frame", None)
    data_cfg = getattr(base, "data_cfg", None)
    if not isinstance(frame, pd.DataFrame) or data_cfg is None:
        raise TypeError(
            "Adaptive target-edge augmentation requires ProcessGraphTabularDataset or "
            "a Subset wrapping it."
        )
    if str(getattr(data_cfg, "task_mode", "")).strip().lower() != "edge_all":
        raise ValueError("Target-edge augmentation is supported only for data.task_mode='edge_all'.")

    rules = [dict(item) for item in list(getattr(augmentation_cfg, "adaptive_bins", []) or [])]
    max_items = _resolve_max_augmented_items(len(dataset), augmentation_cfg)
    repeat_cap = int(getattr(augmentation_cfg, "repeat_cap_per_original", 20))
    if max_items <= 0 or not rules:
        summary = {
            "enabled": True,
            "selection_mode": "adaptive_bins",
            "original_sample_count": int(len(dataset)),
            "augmented_sample_count": 0,
            "final_dataset_count": int(len(dataset)),
            "reason": "max_augmented_items<=0 or adaptive_bins empty",
        }
        return TargetEdgeAugmentationBuild(
            dataset=TargetEdgeAugmentedDataset(dataset, []),
            summary=summary,
        )

    process_column = str(getattr(data_cfg, "process_id_column", "process_id"))
    if process_column not in frame.columns or "ID" not in frame.columns:
        raise KeyError(
            f"Target-edge augmentation requires columns {process_column!r} and 'ID'."
        )
    selected = frame.iloc[base_indices][[process_column, "ID"]].copy().reset_index(drop=True)
    selected["_dataset_index"] = range(len(selected))
    selected["_process_num"] = selected[process_column].map(_process_number)
    selected["_sample_key"] = selected["ID"].map(_id_key)
    if bool(selected["_sample_key"].eq("").any()):
        raise ValueError("Target-edge augmentation found empty sample IDs in train data.")

    needed_properties = sorted({str(rule["property"]) for rule in rules})
    target_edges, target_edge_count = _edges_by_process(
        project_root=project_root,
        data_cfg=data_cfg,
        target_only=True,
    )
    all_edges, all_edge_count = _edges_by_process(
        project_root=project_root,
        data_cfg=data_cfg,
        target_only=False,
    )
    stream_dir = project_root / str(
        getattr(data_cfg, "stream_data_dir", "data/main_data_Streams")
    )
    edge_display_names = _edge_display_names_by_process(
        project_root=project_root,
        data_cfg=data_cfg,
    )

    # First collect train-only physical values. Quantile thresholds therefore cannot
    # leak validation/test distribution into the augmentation decision.
    candidate_rows: list[dict[str, Any]] = []
    missing_sample_edge_count = 0
    for process_num, sample_rows in selected.groupby("_process_num", sort=True):
        process_num = int(process_num)
        process_edges_by_scope = {
            "target": target_edges.get(process_num, []),
            "all": all_edges.get(process_num, []),
        }
        stream_path = stream_dir / f"{process_num}.Process_Streams.csv"
        if not stream_path.is_file():
            raise FileNotFoundError(f"Process_Streams file not found: {stream_path}")
        stream_frame = pd.read_csv(
            stream_path,
            usecols=["ID", "Stream_Name", *needed_properties],
        )
        stream_frame["_sample_key"] = stream_frame["ID"].map(_id_key)
        stream_frame["_stream_key"] = stream_frame["Stream_Name"].map(_id_key)
        wanted_ids = set(sample_rows["_sample_key"])
        wanted_streams = {
            stream_key
            for edges in process_edges_by_scope.values()
            for _, stream_key in edges
        }
        stream_frame = stream_frame[
            stream_frame["_sample_key"].isin(wanted_ids)
            & stream_frame["_stream_key"].isin(wanted_streams)
        ]
        values_by_sample_stream = (
            stream_frame.groupby(["_sample_key", "_stream_key"], sort=False)[
                needed_properties
            ]
            .max()
            .reset_index()
        )
        sample_index_frame = sample_rows[["_dataset_index", "_sample_key"]].copy()
        sample_index_frame["_sample_key"] = sample_index_frame["_sample_key"].astype(str)
        for scope, edges in process_edges_by_scope.items():
            if not edges:
                continue
            edge_frame = pd.DataFrame(edges, columns=["canonical_edge_id", "_stream_key"])
            merged = values_by_sample_stream.merge(edge_frame, on="_stream_key", how="inner")
            if merged.empty:
                missing_sample_edge_count += len(sample_rows) * len(edges)
                continue
            merged = merged.merge(sample_index_frame, on="_sample_key", how="inner")
            missing_sample_edge_count += max(0, len(sample_rows) * len(edges) - len(merged))
            for _, row in merged.iterrows():
                values = {
                    prop: float(row[prop])
                    for prop in needed_properties
                }
                candidate_rows.append(
                    {
                        "scope": scope,
                        "process_id": process_num,
                        "original_index": int(row["_dataset_index"]),
                        "sample_key": str(row["_sample_key"]),
                        "canonical_edge_id": str(row["canonical_edge_id"]),
                        "stream_name": str(
                            edge_display_names.get(
                                (process_num, str(row["canonical_edge_id"])),
                                str(row["canonical_edge_id"]),
                            )
                        ),
                        "values": values,
                    }
                )

    mass_thresholds: dict[str, tuple[float, float]] = {}
    for rule in rules:
        prop = str(rule["property"])
        if prop != "Mass_Flow":
            continue
        scope = str(rule["scope"]).strip().lower()
        rule_name = str(rule["name"])
        values = [
            float(row["values"][prop])
            for row in candidate_rows
            if row["scope"] == scope and math.isfinite(float(row["values"][prop]))
        ]
        if not values:
            mass_thresholds[rule_name] = (math.inf, -math.inf)
            continue
        series = pd.Series(values, dtype="float64")
        qlow = float(rule["quantile_low"])
        qhigh = float(rule["quantile_high"])
        mass_thresholds[rule_name] = (
            float(series.quantile(qlow)),
            float(series.quantile(qhigh)),
        )

    rule_candidates: dict[str, list[TargetEdgeAugmentationRecord]] = {
        str(rule["name"]): [] for rule in rules
    }
    rule_candidate_counts: Counter[str] = Counter()
    for rule in rules:
        name = str(rule["name"])
        scope = str(rule["scope"]).strip().lower()
        prop = str(rule["property"])
        selected_bin = str(rule.get("bin", name))
        for row in candidate_rows:
            if row["scope"] != scope:
                continue
            value = float(row["values"][prop])
            if not math.isfinite(value):
                continue
            matched, lo, hi = _matches_adaptive_rule(
                value=value,
                row=row,
                rule=rule,
                prop=prop,
                mass_thresholds=mass_thresholds,
            )
            if not matched:
                continue
            component_values = tuple(
                (needed_prop, float(row["values"][needed_prop]))
                for needed_prop in needed_properties
            )
            record = TargetEdgeAugmentationRecord(
                original_index=int(row["original_index"]),
                process_id=int(row["process_id"]),
                sample_key=str(row["sample_key"]),
                canonical_edge_id=str(row["canonical_edge_id"]),
                positive_components=(prop,),
                high_positive_components=(),
                component_values=component_values,
                copy_index=0,
                selected_property=prop,
                selected_bin=selected_bin,
                selected_scope=scope,
            )
            rule_candidates[name].append(record)
        rule_candidate_counts[name] = len(rule_candidates[name])

    rng = random.Random(int(augmentation_cfg.seed))
    priority_edges = {
        str(edge_id): float(weight)
        for edge_id, weight in getattr(augmentation_cfg, "mass_flow_priority_edges", {}).items()
    }
    selected_records: list[TargetEdgeAugmentationRecord] = []
    sample_repeat_counts: Counter[int] = Counter()
    selected_rule_counts: Counter[str] = Counter()
    shortfalls: Counter[str] = Counter()
    spillover_counts: Counter[str] = Counter()
    rule_by_name = {str(rule["name"]): rule for rule in rules}

    for rule in rules:
        if len(selected_records) >= max_items:
            break
        name = str(rule["name"])
        quota = int(rule.get("quota", 0))
        rule_repeat_cap = int(rule.get("repeat_cap_per_original", repeat_cap))
        candidates = rule_candidates[name]
        selected_for_rule = 0
        if (
            str(rule["property"]) == "Mass_Flow"
            and str(rule["scope"]).strip().lower() == "target"
            and priority_edges
        ):
            edge_quotas = _round_edge_quotas(quota, priority_edges)
            for edge_id, edge_quota in edge_quotas.items():
                selected_for_rule += _select_from_candidates(
                    candidates=candidates,
                    quota=edge_quota,
                    repeat_cap=rule_repeat_cap,
                    rng=rng,
                    sample_repeat_counts=sample_repeat_counts,
                    selected_records=selected_records,
                    selected_rule_counts=selected_rule_counts,
                    rule_key=name,
                    max_total=max_items,
                    edge_id=edge_id,
                )
                if len(selected_records) >= max_items:
                    break
            remaining = quota - selected_for_rule
        else:
            remaining = quota
        if remaining > 0 and len(selected_records) < max_items:
            selected_for_rule += _select_from_candidates(
                candidates=candidates,
                quota=remaining,
                repeat_cap=rule_repeat_cap,
                rng=rng,
                sample_repeat_counts=sample_repeat_counts,
                selected_records=selected_records,
                selected_rule_counts=selected_rule_counts,
                rule_key=name,
                max_total=max_items,
            )
        shortfall = max(0, quota - selected_for_rule)
        if shortfall > 0:
            shortfalls[name] += shortfall
            spillover_to = str(rule.get("spillover_to", "")).strip()
            if spillover_to and spillover_to in rule_by_name and len(selected_records) < max_items:
                added = _select_from_candidates(
                    candidates=rule_candidates[spillover_to],
                    quota=shortfall,
                    repeat_cap=int(rule_by_name[spillover_to].get("repeat_cap_per_original", repeat_cap)),
                    rng=rng,
                    sample_repeat_counts=sample_repeat_counts,
                    selected_records=selected_records,
                    selected_rule_counts=selected_rule_counts,
                    rule_key=spillover_to,
                    max_total=max_items,
                )
                spillover_counts[f"{name}->{spillover_to}"] += added

    property_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()
    edge_counts: Counter[str] = Counter()
    process_counts: Counter[str] = Counter()
    property_bin_edge_counts: Counter[tuple[str, str, str, str]] = Counter()
    property_bin_stream_edge_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    repeat_groups: dict[int, set[str]] = defaultdict(set)
    for record in selected_records:
        prop = str(record.selected_property or "")
        bin_name = str(record.selected_bin or "")
        scope = str(record.selected_scope or "")
        stream_name = str(edge_display_names.get((int(record.process_id), record.canonical_edge_id), ""))
        property_counts[prop] += 1
        scope_counts[scope] += 1
        edge_counts[record.canonical_edge_id] += 1
        process_counts[str(record.process_id)] += 1
        property_bin_edge_counts[(scope, prop, bin_name, record.canonical_edge_id)] += 1
        property_bin_stream_edge_counts[(scope, prop, bin_name, stream_name, record.canonical_edge_id)] += 1
        repeat_groups[int(record.original_index)].add(bin_name)

    bin_count_rows = []
    for rule in rules:
        name = str(rule["name"])
        prop = str(rule["property"])
        scope = str(rule["scope"]).strip().lower()
        bin_name = str(rule.get("bin", name))
        lo = hi = ""
        if prop == "Mass_Flow":
            lo, hi = mass_thresholds.get(name, ("", ""))
        else:
            lo, hi = float(rule["min_value"]), float(rule["max_value"])
        bin_count_rows.append(
            {
                "rule": name,
                "scope": scope,
                "property": prop,
                "bin": bin_name,
                "quota": int(rule.get("quota", 0)),
                "candidate_count": int(rule_candidate_counts[name]),
                "selected_count": int(selected_rule_counts[name]),
                "shortfall": int(shortfalls[name]),
                "skipped_by_cap": int(shortfalls[name]),
                "lower_bound": lo,
                "upper_bound": hi,
                "spillover_to": str(rule.get("spillover_to", "")),
                "spillover_in": int(
                    sum(
                        count
                        for key, count in spillover_counts.items()
                        if key.endswith(f"->{name}")
                    )
                ),
                "spillover_out": int(
                    sum(
                        count
                        for key, count in spillover_counts.items()
                        if key.startswith(f"{name}->")
                    )
                ),
                "repeat_cap": int(rule.get("repeat_cap_per_original", repeat_cap)),
                "match_mode": str(rule.get("match_mode", "positive")),
                "edge_ids": ",".join(sorted(_rule_edge_filter(rule))),
                "stream_contains": ",".join(_rule_stream_tokens(rule)),
            }
        )
    edge_count_rows = [
        {
            "scope": scope,
            "property": prop,
            "bin": bin_name,
            "stream_name": stream_name,
            "canonical_edge_id": edge_id,
            "selected_count": int(count),
        }
        for (scope, prop, bin_name, stream_name, edge_id), count in sorted(property_bin_stream_edge_counts.items())
    ]
    repeat_rows = [
        {
            "original_index": int(index),
            "repeat_count": int(count),
            "groups_selected_from": ",".join(sorted(repeat_groups.get(int(index), set()))),
        }
        for index, count in sample_repeat_counts.most_common(100)
    ]
    selected_debug_keys = {
        (
            int(record.original_index),
            record.canonical_edge_id,
            str(record.selected_property or ""),
            str(record.selected_bin or ""),
            str(record.selected_scope or ""),
        )
        for record in selected_records
    }
    candidate_debug_rows = []
    for rule in rules:
        name = str(rule["name"])
        for record in rule_candidates[name][:500]:
            values = dict(record.component_values)
            selected_key = (
                int(record.original_index),
                record.canonical_edge_id,
                str(record.selected_property or ""),
                str(record.selected_bin or ""),
                str(record.selected_scope or ""),
            )
            candidate_debug_rows.append(
                {
                    "group_name": name,
                    "property": str(record.selected_property or ""),
                    "value": values.get(str(record.selected_property or ""), ""),
                    "process_id": int(record.process_id),
                    "canonical_edge_id": record.canonical_edge_id,
                    "stream_name": edge_display_names.get((int(record.process_id), record.canonical_edge_id), ""),
                    "original_index": int(record.original_index),
                    "selected": int(selected_key in selected_debug_keys),
                }
            )

    summary = {
        "enabled": True,
        "selection_mode": "adaptive_bins",
        "target_mapping_unique_edge_count": int(target_edge_count),
        "all_edge_mapping_unique_edge_count": int(all_edge_count),
        "original_sample_count": int(len(dataset)),
        "augmented_sample_count": int(len(selected_records)),
        "final_dataset_count": int(len(dataset) + len(selected_records)),
        "max_augmented_items": int(max_items),
        "configured_max_augmented_items": int(getattr(augmentation_cfg, "max_augmented_items", 0)),
        "max_augmented_ratio": (
            None
            if getattr(augmentation_cfg, "max_augmented_ratio", None) is None
            else float(getattr(augmentation_cfg, "max_augmented_ratio"))
        ),
        "repeat_cap_per_original": int(repeat_cap),
        "rule_repeat_caps": {
            str(rule["name"]): int(rule.get("repeat_cap_per_original", repeat_cap))
            for rule in rules
        },
        "observed_max_repeat_per_original": int(max(sample_repeat_counts.values(), default=0)),
        "rule_candidate_counts": dict(sorted(rule_candidate_counts.items())),
        "rule_selected_counts": dict(sorted(selected_rule_counts.items())),
        "rule_shortfalls": dict(sorted(shortfalls.items())),
        "spillover_counts": dict(sorted(spillover_counts.items())),
        "property_augmented_item_counts": dict(sorted(property_counts.items())),
        "scope_augmented_item_counts": dict(sorted(scope_counts.items())),
        "process_augmented_item_counts": dict(
            sorted(process_counts.items(), key=lambda item: int(item[0]))
        ),
        "top_augmented_edges": [
            {"canonical_edge_id": edge_id, "count": int(count)}
            for edge_id, count in edge_counts.most_common(30)
        ],
        "mass_flow_priority_edges": priority_edges,
        "missing_sample_edge_count": int(missing_sample_edge_count),
        "keep_original_items": True,
        "apply_to": ["train"],
        "seed": int(augmentation_cfg.seed),
        "csv_tables": {
            "adaptive_augmentation_bin_counts": bin_count_rows,
            "adaptive_augmentation_edge_counts": edge_count_rows,
            "adaptive_augmentation_repeat_counts": repeat_rows,
            "adaptive_augmentation_candidate_debug": candidate_debug_rows,
        },
        "examples": [
            {
                "original_index": int(record.original_index),
                "process_id": int(record.process_id),
                "sample_key": record.sample_key,
                "canonical_edge_id": record.canonical_edge_id,
                "selected_scope": record.selected_scope,
                "selected_property": record.selected_property,
                "selected_bin": record.selected_bin,
                "component_values": dict(record.component_values),
            }
            for record in selected_records[:5]
        ],
    }
    return TargetEdgeAugmentationBuild(
        dataset=TargetEdgeAugmentedDataset(dataset, selected_records),
        summary=summary,
    )


def build_balanced_target_edge_augmented_dataset(
    dataset: Dataset,
    *,
    augmentation_cfg: Any,
    project_root: Path,
) -> TargetEdgeAugmentationBuild:
    if str(getattr(augmentation_cfg, "selection_mode", "")).strip().lower() == "adaptive_bins":
        return build_adaptive_target_edge_augmented_dataset(
            dataset,
            augmentation_cfg=augmentation_cfg,
            project_root=project_root,
        )

    base, base_indices = _base_dataset_indices(dataset)
    frame = getattr(base, "frame", None)
    data_cfg = getattr(base, "data_cfg", None)
    if not isinstance(frame, pd.DataFrame) or data_cfg is None:
        raise TypeError(
            "Target-edge augmentation requires ProcessGraphTabularDataset or a Subset "
            "wrapping it."
        )
    if str(getattr(data_cfg, "task_mode", "")).strip().lower() != "edge_all":
        raise ValueError("Target-edge augmentation is supported only for data.task_mode='edge_all'.")

    properties = [str(name) for name in augmentation_cfg.target_properties]
    thresholds = {
        str(name): float(value)
        for name, value in augmentation_cfg.min_positive_threshold.items()
    }
    quotas = {
        str(name): int(value)
        for name, value in augmentation_cfg.property_quota.items()
    }
    factor = int(augmentation_cfg.factor)
    max_items = _resolve_max_augmented_items(len(dataset), augmentation_cfg)
    max_per_edge = int(augmentation_cfg.max_per_target_edge)
    process_column = str(getattr(data_cfg, "process_id_column", "process_id"))
    if process_column not in frame.columns or "ID" not in frame.columns:
        raise KeyError(
            f"Target-edge augmentation requires columns {process_column!r} and 'ID'."
        )

    selected = frame.iloc[base_indices][[process_column, "ID"]].copy().reset_index(drop=True)
    selected["_dataset_index"] = range(len(selected))
    selected["_process_num"] = selected[process_column].map(_process_number)
    selected["_sample_key"] = selected["ID"].map(_id_key)
    if bool(selected["_sample_key"].eq("").any()):
        raise ValueError("Target-edge augmentation found empty sample IDs in train data.")

    target_edges, target_edge_count = _target_edges_by_process(
        project_root=project_root,
        data_cfg=data_cfg,
    )
    stream_dir = project_root / str(
        getattr(data_cfg, "stream_data_dir", "data/main_data_Streams")
    )
    pools: dict[str, dict[str, list[TargetEdgeAugmentationRecord]]] = {
        name: defaultdict(list) for name in properties
    }
    candidate_counts: Counter[str] = Counter()
    missing_sample_edge_count = 0

    for process_num, sample_rows in selected.groupby("_process_num", sort=True):
        process_num = int(process_num)
        process_target_edges = target_edges.get(process_num, [])
        if not process_target_edges:
            raise RuntimeError(
                f"Target-edge augmentation resolved no target edges for process {process_num}."
            )
        stream_path = stream_dir / f"{process_num}.Process_Streams.csv"
        if not stream_path.is_file():
            raise FileNotFoundError(f"Process_Streams file not found: {stream_path}")
        stream_frame = pd.read_csv(
            stream_path,
            usecols=["ID", "Stream_Name", *properties],
        )
        stream_frame["_sample_key"] = stream_frame["ID"].map(_id_key)
        stream_frame["_stream_key"] = stream_frame["Stream_Name"].map(_id_key)
        wanted_ids = set(sample_rows["_sample_key"])
        wanted_streams = {stream_key for _, stream_key in process_target_edges}
        stream_frame = stream_frame[
            stream_frame["_sample_key"].isin(wanted_ids)
            & stream_frame["_stream_key"].isin(wanted_streams)
        ]
        values_by_sample_stream = stream_frame.groupby(
            ["_sample_key", "_stream_key"],
            sort=False,
        )[properties].max()

        for original_index, sample_key in sample_rows[
            ["_dataset_index", "_sample_key"]
        ].itertuples(index=False, name=None):
            original_index = int(original_index)
            sample_key = str(sample_key)
            for edge_id, stream_key in process_target_edges:
                lookup_key = (sample_key, stream_key)
                if lookup_key not in values_by_sample_stream.index:
                    missing_sample_edge_count += 1
                    continue
                values = {
                    name: float(values_by_sample_stream.loc[lookup_key, name])
                    for name in properties
                }
                positive = tuple(
                    name
                    for name in properties
                    if math.isfinite(values[name]) and values[name] > thresholds[name]
                )
                if not positive:
                    continue
                component_values = tuple((name, values[name]) for name in properties)
                for selected_property in positive:
                    candidate_counts[selected_property] += 1
                    base_record = TargetEdgeAugmentationRecord(
                        original_index=original_index,
                        process_id=process_num,
                        sample_key=sample_key,
                        canonical_edge_id=edge_id,
                        positive_components=positive,
                        high_positive_components=(),
                        component_values=component_values,
                        copy_index=0,
                        selected_property=selected_property,
                    )
                    pools[selected_property][edge_id].append(base_record)

    rng = random.Random(int(augmentation_cfg.seed))
    expanded_pools: dict[str, dict[str, list[TargetEdgeAugmentationRecord]]] = {
        name: {} for name in properties
    }
    edge_orders: dict[str, list[str]] = {}
    for property_name in properties:
        for edge_id, records in pools[property_name].items():
            expanded = [
                replace(record, copy_index=copy_index)
                for record in records
                for copy_index in range(factor)
            ]
            rng.shuffle(expanded)
            expanded_pools[property_name][edge_id] = expanded
        edge_order = sorted(expanded_pools[property_name])
        rng.shuffle(edge_order)
        edge_orders[property_name] = edge_order

    selected_records: list[TargetEdgeAugmentationRecord] = []
    property_counts: Counter[str] = Counter()
    edge_counts: Counter[str] = Counter()
    property_edge_counts: Counter[tuple[str, str]] = Counter()
    process_counts: Counter[str] = Counter()
    property_sample_edge_counts: Counter[tuple[str, int, str, str]] = Counter()
    edge_cursor = {name: 0 for name in properties}

    while len(selected_records) < max_items:
        made_progress = False
        for property_name in properties:
            if len(selected_records) >= max_items:
                break
            if property_counts[property_name] >= quotas[property_name]:
                continue
            edge_order = edge_orders[property_name]
            if not edge_order:
                continue
            selected_record: TargetEdgeAugmentationRecord | None = None
            for _ in range(len(edge_order)):
                cursor = edge_cursor[property_name] % len(edge_order)
                edge_cursor[property_name] += 1
                edge_id = edge_order[cursor]
                if property_edge_counts[(property_name, edge_id)] >= max_per_edge:
                    continue
                queue = expanded_pools[property_name][edge_id]
                while queue:
                    candidate = queue.pop()
                    property_sample_edge_key = (
                        property_name,
                        candidate.original_index,
                        candidate.sample_key,
                        candidate.canonical_edge_id,
                    )
                    if property_sample_edge_counts[property_sample_edge_key] < factor:
                        selected_record = candidate
                        break
                if selected_record is not None:
                    break
            if selected_record is None:
                continue
            selected_records.append(selected_record)
            property_counts[property_name] += 1
            edge_counts[selected_record.canonical_edge_id] += 1
            property_edge_counts[
                (property_name, selected_record.canonical_edge_id)
            ] += 1
            process_counts[str(selected_record.process_id)] += 1
            property_sample_edge_counts[
                (
                    property_name,
                    selected_record.original_index,
                    selected_record.sample_key,
                    selected_record.canonical_edge_id,
                )
            ] += 1
            made_progress = True
        if not made_progress:
            break

    quota_shortfall = {
        name: int(max(0, quotas[name] - property_counts[name]))
        for name in properties
    }
    summary = {
        "enabled": True,
        "selection_mode": "balanced_property_edge",
        "target_mapping_unique_edge_count": int(target_edge_count),
        "original_sample_count": int(len(dataset)),
        "augmented_sample_count": int(len(selected_records)),
        "final_dataset_count": int(len(dataset) + len(selected_records)),
        "max_augmented_items": max_items,
        "configured_max_augmented_items": int(getattr(augmentation_cfg, "max_augmented_items", 0)),
        "max_augmented_ratio": (
            None
            if getattr(augmentation_cfg, "max_augmented_ratio", None) is None
            else float(getattr(augmentation_cfg, "max_augmented_ratio"))
        ),
        "factor": factor,
        "max_per_target_edge": max_per_edge,
        "property_quota": quotas,
        "property_candidate_edge_counts": {
            name: int(candidate_counts[name]) for name in properties
        },
        "property_augmented_item_counts": {
            name: int(property_counts[name]) for name in properties
        },
        "property_quota_shortfall": quota_shortfall,
        "process_augmented_item_counts": dict(
            sorted(process_counts.items(), key=lambda item: int(item[0]))
        ),
        "target_edge_augmented_item_counts": dict(sorted(edge_counts.items())),
        "property_target_edge_augmented_item_counts": {
            f"{property_name}|{edge_id}": int(count)
            for (property_name, edge_id), count in sorted(property_edge_counts.items())
        },
        "observed_max_per_property_target_edge": int(
            max(property_edge_counts.values(), default=0)
        ),
        "observed_max_total_per_target_edge": int(max(edge_counts.values(), default=0)),
        "missing_sample_target_edge_count": int(missing_sample_edge_count),
        "keep_original_items": True,
        "apply_to": ["train"],
        "seed": int(augmentation_cfg.seed),
        "examples": [
            {
                "original_index": int(record.original_index),
                "process_id": int(record.process_id),
                "sample_key": record.sample_key,
                "canonical_edge_id": record.canonical_edge_id,
                "selected_property": record.selected_property,
                "positive_components": list(record.positive_components),
                "component_values": dict(record.component_values),
            }
            for record in selected_records[:3]
        ],
    }
    return TargetEdgeAugmentationBuild(
        dataset=TargetEdgeAugmentedDataset(dataset, selected_records),
        summary=summary,
    )
