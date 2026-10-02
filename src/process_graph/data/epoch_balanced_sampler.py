from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from torch.utils.data import Sampler

from process_graph.data.target_edge_augmentation import (
    _base_dataset_indices,
    _edges_by_process,
    _id_key,
    _process_number,
    _target_edges_by_process,
)


def _largest_remainder(total: int, weights: dict[str, float]) -> dict[str, int]:
    if total <= 0:
        return {key: 0 for key in weights}
    wsum = sum(max(0.0, float(value)) for value in weights.values())
    if wsum <= 0.0:
        return {key: 0 for key in weights}
    raw = {key: total * max(0.0, float(value)) / wsum for key, value in weights.items()}
    quotas = {key: int(math.floor(value)) for key, value in raw.items()}
    rem = total - sum(quotas.values())
    order = sorted(raw, key=lambda key: (raw[key] - quotas[key], key), reverse=True)
    for key in order[:rem]:
        quotas[key] += 1
    return quotas


def _quantile_bins(values: list[float], quantiles: list[float]) -> list[tuple[float, float, str]]:
    finite = pd.Series([float(v) for v in values if math.isfinite(float(v))])
    if finite.empty:
        return []
    bounds = [float(finite.quantile(float(q))) for q in quantiles]
    bins: list[tuple[float, float, str]] = []
    for lo_q, hi_q, lo, hi in zip(quantiles[:-1], quantiles[1:], bounds[:-1], bounds[1:]):
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi < lo:
            continue
        if hi == lo and len(set(bounds)) > 1:
            continue
        bins.append((lo, hi, f"q{lo_q:.2f}_{hi_q:.2f}"))
    return bins


class EpochBalancedSampler(Sampler[int]):
    """Select a small, process-balanced, distribution-aware train subset per epoch."""

    def __init__(
        self,
        dataset: Any,
        *,
        sampler_cfg: Any,
        project_root: Path,
        output_dir: Path | None = None,
        fold_id: int = 1,
    ) -> None:
        self.dataset = dataset
        self.cfg = sampler_cfg
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.fold_id = int(fold_id)
        self.epoch = 0
        self.selection_counts: Counter[int] = Counter()
        self.history_rows: list[dict[str, Any]] = []
        self.base, self.base_indices = _base_dataset_indices(dataset)
        self.frame = getattr(self.base, "frame", None)
        self.data_cfg = getattr(self.base, "data_cfg", None)
        if not isinstance(self.frame, pd.DataFrame) or self.data_cfg is None:
            raise TypeError("EpochBalancedSampler requires ProcessGraphTabularDataset or a Subset wrapping it.")
        if str(getattr(self.data_cfg, "task_mode", "")).strip().lower() != "edge_all":
            raise ValueError("EpochBalancedSampler supports only data.task_mode='edge_all'.")
        self.process_column = str(getattr(self.data_cfg, "process_id_column", "process_id"))
        if self.process_column not in self.frame.columns or "ID" not in self.frame.columns:
            raise KeyError(f"EpochBalancedSampler requires columns {self.process_column!r} and 'ID'.")
        self.sample_info = self._build_sample_info()
        self.process_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, info in self.sample_info.items():
            self.process_to_indices[int(info["process_num"])].append(int(idx))
        for values in self.process_to_indices.values():
            values.sort()
        self.process_ids = sorted(self.process_to_indices)
        # Reuse the projected Process_Streams frame across target, all-edge,
        # and hard-pool builders. These paths previously reread each large CSV.
        self._stream_frame_cache: dict[int, pd.DataFrame] = {}
        self.mode = str(getattr(self.cfg, "mode", "mixture") or "mixture").strip().lower()
        configured_base_size = int(getattr(self.cfg, "base_epoch_size", 0) or 0)
        self.base_epoch_size = (
            min(configured_base_size, int(len(dataset)))
            if configured_base_size > 0
            else max(1, int(math.ceil(float(self.cfg.epoch_fraction) * len(dataset))))
        )
        if self.mode == "base_plus_hard_fill":
            requested = int(getattr(self.cfg, "hard_fill_total_size", 0) or 0)
            self.epoch_size = max(self.base_epoch_size, requested) if requested > 0 else self.base_epoch_size
            self.epoch_size = min(int(self.epoch_size), int(len(dataset)))
        else:
            self.epoch_size = int(self.base_epoch_size)
        self.feature_values = (
            {}
            if self.mode in {"random", "scheduled_predefined_hard"}
            else self._build_feature_values()
        )
        self.buckets = {} if self.mode == "random" else self._build_buckets()
        self.dedicated_mass_tail_pools = (
            self._build_dedicated_mass_tail_pools()
            if bool(getattr(self.cfg.mass_flow_tail, "enabled", False))
            else {}
        )

    def _build_sample_info(self) -> dict[int, dict[str, Any]]:
        selected = self.frame.iloc[self.base_indices][[self.process_column, "ID"]].copy().reset_index(drop=True)
        selected["_dataset_index"] = range(len(selected))
        selected["_process_num"] = selected[self.process_column].map(_process_number)
        selected["_sample_key"] = selected["ID"].map(_id_key)
        if bool(selected["_sample_key"].eq("").any()):
            raise ValueError("EpochBalancedSampler found empty sample IDs in train data.")
        result: dict[int, dict[str, Any]] = {}
        for row in selected.to_dict("records"):
            result[int(row["_dataset_index"])] = {
                "process_num": int(row["_process_num"]),
                "sample_key": str(row["_sample_key"]),
            }
        return result

    def _sampler_stream_properties(self) -> list[str]:
        properties = (
            set(self.cfg.sparse_positive.properties)
            | set(self.cfg.quantile_balance.properties)
            | {str(self.cfg.mass_flow_tail.property)}
        )
        hard_cfg = getattr(self.cfg, "hard_target_edges", None)
        if hard_cfg is not None:
            for rule in list(getattr(hard_cfg, "rules", []) or []):
                if str(rule.get("source", "target_edge_property")).strip().lower() != "target_edge_property":
                    continue
                property_name = str(rule.get("property", "")).strip()
                if property_name:
                    properties.add(property_name)
        return sorted(name for name in properties if name)

    def _read_process_stream_frame(self, process_num: int) -> pd.DataFrame:
        process_num = int(process_num)
        cached = self._stream_frame_cache.get(process_num)
        if cached is not None:
            return cached
        stream_dir = self.project_root / str(
            getattr(self.data_cfg, "stream_data_dir", "data/main_data_Streams")
        )
        stream_path = stream_dir / f"{process_num}.Process_Streams.csv"
        if not stream_path.is_file():
            raise FileNotFoundError(f"Process_Streams file not found: {stream_path}")
        usecols = {"ID", "Stream_Name", *self._sampler_stream_properties()}
        stream = pd.read_csv(stream_path, usecols=lambda col: col in usecols)
        stream["_sample_key"] = stream["ID"].map(_id_key)
        stream["_stream_key"] = stream["Stream_Name"].map(_id_key)
        wanted_ids = {
            info["sample_key"]
            for info in self.sample_info.values()
            if int(info["process_num"]) == process_num
        }
        stream = stream[stream["_sample_key"].isin(wanted_ids)].copy()
        self._stream_frame_cache[process_num] = stream
        return stream

    def _read_stream_values(self, *, target_only: bool) -> pd.DataFrame:
        edge_map, _ = (
            _target_edges_by_process(project_root=self.project_root, data_cfg=self.data_cfg)
            if target_only
            else _edges_by_process(project_root=self.project_root, data_cfg=self.data_cfg, target_only=False)
        )
        frames: list[pd.DataFrame] = []
        for process_num in self.process_ids:
            stream = self._read_process_stream_frame(process_num)
            wanted_streams = {stream_key for _, stream_key in edge_map.get(process_num, [])}
            stream = stream[stream["_stream_key"].isin(wanted_streams)].copy()
            stream["_process_num"] = int(process_num)
            frames.append(stream)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _build_feature_values(self) -> dict[int, dict[str, float]]:
        target_streams = self._read_stream_values(target_only=True)
        all_streams = self._read_stream_values(target_only=False)
        result: dict[int, dict[str, float]] = {idx: {} for idx in self.sample_info}
        lookup = {
            (int(info["process_num"]), str(info["sample_key"])): idx for idx, info in self.sample_info.items()
        }

        def merge_values(frame: pd.DataFrame, *, prefix: str) -> None:
            if frame.empty:
                return
            value_cols = [
                col for col in frame.columns if col not in {"ID", "Stream_Name", "_sample_key", "_stream_key", "_process_num"}
            ]
            grouped = frame.groupby(["_process_num", "_sample_key"], sort=False)[value_cols].max(numeric_only=True)
            for key, row in grouped.iterrows():
                idx = lookup.get((int(key[0]), str(key[1])))
                if idx is None:
                    continue
                for col in value_cols:
                    val = row.get(col)
                    try:
                        fval = float(val)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(fval):
                        result[idx][f"{prefix}:{col}"] = fval

        merge_values(target_streams, prefix="target")
        merge_values(all_streams, prefix="all")
        return result

    def _build_buckets(self) -> dict[str, dict[int, dict[str, list[int]]]]:
        buckets: dict[str, dict[int, dict[str, list[int]]]] = {
            "uniform": defaultdict(lambda: {"all": []}),
            "sparse_positive": defaultdict(lambda: defaultdict(list)),
            "quantile_balance": defaultdict(lambda: defaultdict(list)),
            "mass_flow_tail": defaultdict(lambda: defaultdict(list)),
            "hard_target_edges": defaultdict(lambda: defaultdict(list)),
        }
        for process_num, indices in self.process_to_indices.items():
            buckets["uniform"][process_num]["all"] = list(indices)

        if self.mode == "scheduled_predefined_hard":
            hard_cfg = getattr(self.cfg, "hard_target_edges", None)
            if hard_cfg is not None and bool(getattr(hard_cfg, "enabled", False)):
                self._populate_hard_target_edge_buckets(buckets["hard_target_edges"])
            return buckets

        zero_threshold = float(self.cfg.sparse_positive.zero_threshold)
        for prop in self.cfg.sparse_positive.properties:
            values = [
                self.feature_values[idx].get(f"target:{prop}", float("nan"))
                for idx in self.sample_info
            ]
            positive = [float(v) for v in values if math.isfinite(float(v)) and float(v) > zero_threshold]
            bins = _quantile_bins(positive, list(self.cfg.sparse_positive.positive_quantiles))
            for idx, info in self.sample_info.items():
                value = self.feature_values[idx].get(f"target:{prop}", float("nan"))
                if not math.isfinite(float(value)) or float(value) <= zero_threshold:
                    continue
                process_num = int(info["process_num"])
                for lo, hi, label in bins:
                    if float(value) >= lo and float(value) <= hi:
                        buckets["sparse_positive"][process_num][f"{prop}:{label}"].append(idx)
                        break

        for prop in self.cfg.quantile_balance.properties:
            for process_num, indices in self.process_to_indices.items():
                vals = [self.feature_values[idx].get(f"all:{prop}", float("nan")) for idx in indices]
                bins = _quantile_bins([float(v) for v in vals if math.isfinite(float(v))], list(self.cfg.quantile_balance.quantiles))
                for idx in indices:
                    value = self.feature_values[idx].get(f"all:{prop}", float("nan"))
                    if not math.isfinite(float(value)):
                        continue
                    for lo, hi, label in bins:
                        if float(value) >= lo and float(value) <= hi:
                            buckets["quantile_balance"][process_num][f"{prop}:{label}"].append(idx)
                            break

        tail_prop = str(self.cfg.mass_flow_tail.property)
        if str(self.cfg.mass_flow_tail.quantile_scope).lower() == "global":
            global_bins = _quantile_bins(
                [
                    self.feature_values[idx].get(f"all:{tail_prop}", float("nan"))
                    for idx in self.sample_info
                ],
                list(self.cfg.mass_flow_tail.quantiles),
            )
        else:
            global_bins = []
        for process_num, indices in self.process_to_indices.items():
            bins = global_bins or _quantile_bins(
                [
                    self.feature_values[idx].get(f"all:{tail_prop}", float("nan"))
                    for idx in indices
                ],
                list(self.cfg.mass_flow_tail.quantiles),
            )
            for idx in indices:
                value = self.feature_values[idx].get(f"all:{tail_prop}", float("nan"))
                if not math.isfinite(float(value)):
                    continue
                for lo, hi, label in bins:
                    if float(value) >= lo and float(value) <= hi:
                        buckets["mass_flow_tail"][process_num][f"{tail_prop}:{label}"].append(idx)
                        break
        hard_cfg = getattr(self.cfg, "hard_target_edges", None)
        if hard_cfg is not None and bool(getattr(hard_cfg, "enabled", False)):
            self._populate_hard_target_edge_buckets(buckets["hard_target_edges"])
        return buckets

    def _build_dedicated_mass_tail_pools(self) -> dict[str, dict[str, list[int]]]:
        cfg = self.cfg.mass_flow_tail
        property_name = str(cfg.property)
        quantiles = list(cfg.quantiles)
        edge_map, _ = _edges_by_process(
            project_root=self.project_root,
            data_cfg=self.data_cfg,
            target_only=False,
        )
        stream_dir = self.project_root / str(
            getattr(self.data_cfg, "stream_data_dir", "data/main_data_Streams")
        )
        records: dict[int, dict[str, list[tuple[int, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        process_values: dict[int, list[float]] = defaultdict(list)
        all_values: list[float] = []
        for process_num in self.process_ids:
            wanted_ids = {
                info["sample_key"]
                for info in self.sample_info.values()
                if int(info["process_num"]) == int(process_num)
            }
            sample_lookup = {
                str(info["sample_key"]): idx
                for idx, info in self.sample_info.items()
                if int(info["process_num"]) == int(process_num)
            }
            stream_path = stream_dir / f"{process_num}.Process_Streams.csv"
            if not stream_path.is_file():
                raise FileNotFoundError(
                    f"Process_Streams file not found: {stream_path}"
                )
            stream = pd.read_csv(
                stream_path,
                usecols=lambda col: col in {"ID", "Stream_Name", property_name},
            )
            stream["_sample_key"] = stream["ID"].map(_id_key)
            stream["_stream_key"] = stream["Stream_Name"].map(_id_key)
            stream = stream[stream["_sample_key"].isin(wanted_ids)]
            for edge_id, stream_key in edge_map.get(process_num, []):
                edge_rows = stream[stream["_stream_key"] == str(stream_key)]
                if property_name not in edge_rows.columns:
                    continue
                for row in edge_rows[["_sample_key", property_name]].itertuples(
                    index=False, name=None
                ):
                    idx = sample_lookup.get(str(row[0]))
                    try:
                        value = float(row[1])
                    except (TypeError, ValueError):
                        continue
                    if idx is None or not math.isfinite(value):
                        continue
                    records[int(process_num)][str(edge_id)].append((idx, value))
                    all_values.append(value)
                    process_values[int(process_num)].append(value)

        global_bins = _quantile_bins(all_values, quantiles)
        result: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        min_count = int(getattr(cfg, "canonical_edge_min_count", 1))
        for process_num, by_edge in records.items():
            bins = (
                global_bins
                if str(getattr(cfg, "quantile_scope", "process")).lower() == "global"
                else _quantile_bins(process_values[process_num], quantiles)
            )
            for edge_id, values in by_edge.items():
                if len(values) < min_count:
                    continue
                for idx, value in values:
                    for bin_idx, (lo, hi, bin_label) in enumerate(bins):
                        in_bin = value >= lo and (
                            value <= hi if bin_idx == len(bins) - 1 else value < hi
                        )
                        if not in_bin:
                            continue
                        if bool(getattr(cfg, "canonical_edge_balanced", True)):
                            label = f"P{int(process_num):02d}:{edge_id}"
                        elif bool(getattr(cfg, "process_balanced", True)):
                            label = f"P{int(process_num):02d}"
                        else:
                            label = "all"
                        result[bin_label][label].append(int(idx))
                        break
        return {
            bin_label: {
                label: sorted(set(pool)) for label, pool in by_label.items()
            }
            for bin_label, by_label in result.items()
        }

    def _select_dedicated_mass_tail(
        self,
        *,
        rng: random.Random,
        selected: set[int],
    ) -> tuple[list[int], Counter[str], Counter[str], int]:
        cfg = self.cfg.mass_flow_tail
        target = int(getattr(cfg, "target_items_per_epoch", 0))
        chosen_all: list[int] = []
        bin_counts: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()
        for bin_label, quota in dict(getattr(cfg, "bin_quotas", {}) or {}).items():
            chosen, _, labels = self._choose_from_bucket(
                rng=rng,
                pool_by_label=dict(
                    self.dedicated_mass_tail_pools.get(str(bin_label), {})
                ),
                quota=int(quota),
                selected=selected,
                allow_replacement=bool(getattr(cfg, "allow_duplicate_samples", False)),
            )
            chosen_all.extend(chosen)
            bin_counts[str(bin_label)] += len(chosen)
            label_counts.update(
                {f"{bin_label}:{key}": value for key, value in labels.items()}
            )
        shortfall = max(0, target - len(chosen_all))
        if shortfall:
            fallback_pool: dict[str, list[int]] = defaultdict(list)
            for by_label in self.dedicated_mass_tail_pools.values():
                for label, pool in by_label.items():
                    fallback_pool[label].extend(pool)
            fallback, _, labels = self._choose_from_bucket(
                rng=rng,
                pool_by_label=dict(fallback_pool),
                quota=shortfall,
                selected=selected,
                allow_replacement=bool(getattr(cfg, "allow_duplicate_samples", False)),
            )
            chosen_all.extend(fallback)
            bin_counts["tail_pool_fallback"] += len(fallback)
            label_counts.update(
                {f"fallback:{key}": value for key, value in labels.items()}
            )
        return chosen_all, bin_counts, label_counts, max(0, target - len(chosen_all))

    def _populate_hard_target_edge_buckets(self, bucket: dict[int, dict[str, list[int]]]) -> None:
        hard_cfg = getattr(self.cfg, "hard_target_edges", None)
        rules = list(getattr(hard_cfg, "rules", []) or [])
        if not rules:
            return
        ratio_rules = [
            rule
            for rule in rules
            if str(rule.get("source", "target_edge_property")).strip().lower()
            == "main_row_ratio"
        ]
        if ratio_rules:
            selected_frame = self.frame.iloc[self.base_indices].copy().reset_index(drop=True)
            selected_frame["_dataset_index"] = range(len(selected_frame))
            selected_frame["_process_num"] = selected_frame[self.process_column].map(
                _process_number
            )
            for rule in ratio_rules:
                numerator_column = str(rule["numerator_column"])
                denominator_column = str(rule["denominator_column"])
                missing = [
                    name
                    for name in (numerator_column, denominator_column)
                    if name not in selected_frame.columns
                ]
                if missing:
                    raise KeyError(
                        f"main_row_ratio rule {rule.get('name')!r} is missing Main columns: {missing}"
                    )
                process_ids = {int(value) for value in rule.get("process_ids", [])}
                numerator = pd.to_numeric(
                    selected_frame[numerator_column], errors="coerce"
                )
                denominator = pd.to_numeric(
                    selected_frame[denominator_column], errors="coerce"
                )
                epsilon = float(rule.get("epsilon", 1.0e-6))
                valid = (
                    selected_frame["_process_num"].isin(process_ids)
                    & numerator.notna()
                    & denominator.notna()
                    & numerator.ge(0.0)
                    & denominator.gt(epsilon)
                )
                ratio = numerator / denominator
                if str(rule.get("transform", "raw_ratio")).strip().lower() == "log_ratio":
                    ratio = ratio.map(
                        lambda value: math.log(float(value))
                        if math.isfinite(float(value)) and float(value) > 0.0
                        else math.nan
                    )
                selected_rows = selected_frame[
                    valid
                    & ratio.between(
                        float(rule.get("min_value", 0.0)),
                        float(rule.get("max_value", float("inf"))),
                        inclusive="both",
                    )
                ]
                if selected_rows.empty:
                    continue
                prop = str(rule.get("property", "")).strip()
                label = str(rule.get("name", f"{prop}_input_regime")).strip()
                bucket_label = f"{label}:input_regime:{prop}"
                for process_num, process_rows in selected_rows.groupby(
                    "_process_num", sort=True
                ):
                    bucket[int(process_num)][bucket_label].extend(
                        int(value) for value in process_rows["_dataset_index"].tolist()
                    )

        rules = [
            rule
            for rule in rules
            if str(rule.get("source", "target_edge_property")).strip().lower()
            == "target_edge_property"
        ]
        if not rules:
            return
        needed_properties = sorted({str(rule.get("property", "")).strip() for rule in rules if str(rule.get("property", "")).strip()})
        if not needed_properties:
            return
        edge_map, _ = _target_edges_by_process(project_root=self.project_root, data_cfg=self.data_cfg)
        rules_by_edge: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rule in rules:
            for edge_id in rule.get("edge_ids", []) or []:
                rules_by_edge[str(edge_id).strip()].append(rule)
        for process_num in self.process_ids:
            process_edges = {
                str(edge_id).strip(): str(stream_key).strip()
                for edge_id, stream_key in edge_map.get(process_num, [])
                if str(edge_id).strip() in rules_by_edge
            }
            if not process_edges:
                continue
            stream = self._read_process_stream_frame(process_num)
            sample_lookup = {
                str(info["sample_key"]): idx
                for idx, info in self.sample_info.items()
                if int(info["process_num"]) == int(process_num)
            }
            for edge_id, stream_key in process_edges.items():
                edge_rows = stream[stream["_stream_key"] == stream_key]
                if edge_rows.empty:
                    continue
                for rule in rules_by_edge[edge_id]:
                    prop = str(rule.get("property", "")).strip()
                    if prop not in edge_rows.columns:
                        continue
                    min_value = float(rule.get("min_value", 0.0))
                    max_value = float(rule.get("max_value", float("inf")))
                    label = str(rule.get("name", f"{prop}_hard_edges")).strip() or f"{prop}_hard_edges"
                    selected_rows = edge_rows[
                        pd.to_numeric(edge_rows[prop], errors="coerce").between(
                            min_value,
                            max_value,
                            inclusive="both",
                        )
                    ]
                    if selected_rows.empty:
                        continue
                    bucket_label = f"{label}:{edge_id}:{prop}"
                    for sample_key in selected_rows["_sample_key"].astype(str):
                        idx = sample_lookup.get(sample_key)
                        if idx is not None:
                            bucket[int(process_num)][bucket_label].append(int(idx))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self.epoch_size)

    def _process_quotas(self) -> dict[int, int]:
        return self._process_quotas_for(int(self.epoch_size))

    def _process_quotas_for(self, total: int) -> dict[int, int]:
        total = max(0, int(total))
        base = total // max(1, len(self.process_ids))
        rem = total % max(1, len(self.process_ids))
        quotas = {pid: base for pid in self.process_ids}
        if rem:
            start = self.epoch % len(self.process_ids)
            order = self.process_ids[start:] + self.process_ids[:start]
            for pid in order[:rem]:
                quotas[pid] += 1
        return quotas

    def _scheduled_hard_ratio(self) -> float:
        displayed_epoch = int(self.epoch) + 1
        warmup_end = int(getattr(self.cfg, "warmup_end_epoch", 5))
        transition_end = int(getattr(self.cfg, "transition_end_epoch", 7))
        if displayed_epoch <= warmup_end:
            return 0.0
        if displayed_epoch <= transition_end:
            return float(getattr(self.cfg, "transition_hard_ratio", 0.20))
        return float(getattr(self.cfg, "final_hard_ratio", 0.30))

    def _choose_from_bucket(
        self,
        *,
        rng: random.Random,
        pool_by_label: dict[str, list[int]],
        quota: int,
        selected: set[int],
        allow_replacement: bool,
    ) -> tuple[list[int], int, Counter[str]]:
        chosen: list[int] = []
        replacement_count = 0
        label_counts: Counter[str] = Counter()
        labels = sorted(label for label, pool in pool_by_label.items() if pool)
        if not labels or quota <= 0:
            return chosen, replacement_count, label_counts
        cursor = rng.randrange(len(labels))
        attempts = 0
        while len(chosen) < quota and attempts < quota * max(10, len(labels) * 2):
            label = labels[cursor % len(labels)]
            cursor += 1
            attempts += 1
            pool = list(pool_by_label[label])
            if not pool:
                continue
            rng.shuffle(pool)
            pick = None
            for idx in pool:
                if idx not in selected:
                    pick = idx
                    break
            if pick is None:
                if not allow_replacement:
                    continue
                pick = pool[0]
                replacement_count += 1
            chosen.append(int(pick))
            selected.add(int(pick))
            label_counts[label] += 1
        return chosen, replacement_count, label_counts

    def _write_stats(self, row: dict[str, Any], selected: list[int]) -> None:
        self.history_rows.append(row)
        if str(row.get("mode", "")).strip().lower() in {"scheduled_predefined_hard", "base_plus_hard_fill"}:
            print(
                "[EpochSampler] "
                + json.dumps(
                    {
                        "epoch": int(row.get("epoch", self.epoch + 1)),
                        "mode": row.get("mode", "scheduled_predefined_hard"),
                        "base": int(row.get("base_selected_count", 0)),
                        "total": int(row.get("epoch_train_size", len(selected))),
                        "random_ratio": float(row.get("random_ratio", 0.0)),
                        "hard_ratio": float(row.get("hard_ratio", 0.0)),
                        "random": int(row.get("random_selected_count", 0)),
                        "hard": int(row.get("hard_selected_count", 0)),
                        "hard_fallback_to_random": int(row.get("hard_fallback_to_random_count", 0)),
                        "unique": int(row.get("selected_unique_count", len(set(selected)))),
                        "duplicates": int(row.get("duplicate_sample_count", len(selected) - len(set(selected)))),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if self.output_dir is None:
            return
        out = self.output_dir
        out.mkdir(parents=True, exist_ok=True)
        epoch_path = out / f"sampling_stats_epoch_{self.epoch + 1:04d}.json"
        epoch_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
        pd.DataFrame(self.history_rows).to_csv(out / "sampling_history.csv", index=False)
        if bool(getattr(self.cfg, "log_sample_selection_counts", True)):
            count_rows = [
                {
                    "dataset_index": int(idx),
                    "process_id": int(self.sample_info[idx]["process_num"]),
                    "sample_key": self.sample_info[idx]["sample_key"],
                    "selection_count": int(count),
                }
                for idx, count in sorted(self.selection_counts.items())
            ]
            pd.DataFrame(count_rows).to_csv(out / "sample_selection_counts.csv", index=False)

    def __iter__(self) -> Iterable[int]:
        seed = int(self.cfg.sampler_seed) + self.fold_id * 100000 + int(self.epoch)
        rng = random.Random(seed)
        if self.mode == "random":
            return iter(self._iter_random(rng=rng, seed=seed))
        if self.mode == "scheduled_predefined_hard":
            return iter(self._iter_scheduled_predefined_hard(rng=rng, seed=seed))
        if self.mode == "base_plus_hard_fill":
            return iter(self._iter_base_plus_hard_fill(rng=rng, seed=seed))
        selected: set[int] = set()
        output: list[int] = []
        replacement_count = 0
        process_counts: Counter[str] = Counter()
        bucket_counts: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()
        process_quotas = self._process_quotas()
        mixture = {
            "uniform": float(self.cfg.mixture.uniform),
            "sparse_positive": float(self.cfg.mixture.sparse_positive),
            "quantile_balance": float(self.cfg.mixture.quantile_balance),
            "mass_flow_tail": float(self.cfg.mixture.mass_flow_tail),
            "hard_target_edges": float(getattr(self.cfg.mixture, "hard_target_edges", 0.0)),
        }
        for process_num in self.process_ids:
            pquota = int(process_quotas.get(process_num, 0))
            bquotas = _largest_remainder(pquota, mixture)
            for bucket in ("uniform", "sparse_positive", "quantile_balance", "mass_flow_tail", "hard_target_edges"):
                allow_replacement = (
                    bool(getattr(self.cfg, "replacement_for_uniform", False))
                    if bucket == "uniform"
                    else bool(getattr(self.cfg, "replacement_for_rare_buckets", True))
                )
                chosen, repl, counts = self._choose_from_bucket(
                    rng=rng,
                    pool_by_label=dict(self.buckets[bucket].get(process_num, {})),
                    quota=int(bquotas.get(bucket, 0)),
                    selected=selected,
                    allow_replacement=allow_replacement and not bool(getattr(self.cfg, "unique_within_epoch", True)),
                )
                output.extend(chosen)
                replacement_count += repl
                bucket_counts[bucket] += len(chosen)
                label_counts.update({f"{bucket}:{k}": v for k, v in counts.items()})
            missing = pquota - sum(1 for idx in output if self.sample_info[idx]["process_num"] == process_num)
            if missing > 0:
                fallback_pool = [idx for idx in self.process_to_indices[process_num] if idx not in selected]
                rng.shuffle(fallback_pool)
                fill = fallback_pool[:missing]
                output.extend(fill)
                selected.update(fill)
                bucket_counts["fallback_uniform"] += len(fill)
        if len(output) < self.epoch_size:
            rest = [idx for idx in range(len(self.dataset)) if idx not in selected]
            rng.shuffle(rest)
            fill = rest[: self.epoch_size - len(output)]
            output.extend(fill)
            selected.update(fill)
            bucket_counts["global_fallback"] += len(fill)
        output = output[: self.epoch_size]
        rng.shuffle(output)
        for idx in output:
            self.selection_counts[int(idx)] += 1
            process_counts[str(self.sample_info[int(idx)]["process_num"])] += 1
        coverage = len(self.selection_counts) / max(1, len(self.dataset))
        stat = {
            "fold_id": int(self.fold_id),
            "epoch": int(self.epoch + 1),
            "seed": int(seed),
            "train_pool_size": int(len(self.dataset)),
            "epoch_train_size": int(self.epoch_size),
            "selected_unique_count": int(len(set(output))),
            "replacement_count": int(replacement_count + len(output) - len(set(output))),
            "process_counts": dict(sorted(process_counts.items(), key=lambda item: int(item[0]))),
            "bucket_counts": dict(sorted(bucket_counts.items())),
            "bucket_label_counts": dict(sorted(label_counts.items())),
            "cumulative_unique_selected": int(len(self.selection_counts)),
            "cumulative_unique_coverage": float(coverage),
        }
        self._write_stats(stat, output)
        return iter(output)

    def _iter_random(self, *, rng: random.Random, seed: int) -> list[int]:
        """Draw a unique, globally uniform subset from the full train pool."""
        output = rng.sample(range(len(self.dataset)), k=int(self.epoch_size))
        process_counts: Counter[str] = Counter()
        for idx in output:
            self.selection_counts[int(idx)] += 1
            process_counts[str(self.sample_info[int(idx)]["process_num"])] += 1
        coverage = len(self.selection_counts) / max(1, len(self.dataset))
        stat = {
            "fold_id": int(self.fold_id),
            "epoch": int(self.epoch + 1),
            "mode": "random",
            "seed": int(seed),
            "train_pool_size": int(len(self.dataset)),
            "epoch_train_size": int(self.epoch_size),
            "selected_unique_count": int(len(output)),
            "duplicate_sample_count": 0,
            "replacement_count": 0,
            "process_counts": dict(
                sorted(process_counts.items(), key=lambda item: int(item[0]))
            ),
            "bucket_counts": {"uniform_random": int(len(output))},
            "bucket_label_counts": {},
            "cumulative_unique_selected": int(len(self.selection_counts)),
            "cumulative_unique_coverage": float(coverage),
        }
        self._write_stats(stat, output)
        return output

    def _iter_base_plus_hard_fill(self, *, rng: random.Random, seed: int) -> list[int]:
        selected: set[int] = set()
        output: list[int] = []
        replacement_count = 0
        hard_fallback_to_random = 0
        process_counts: Counter[str] = Counter()
        process_base_counts: Counter[str] = Counter()
        process_hard_counts: Counter[str] = Counter()
        process_hard_fallback_counts: Counter[str] = Counter()
        bucket_counts: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()

        tail_chosen, tail_bins, tail_labels, tail_shortfall = (
            self._select_dedicated_mass_tail(rng=rng, selected=selected)
        )
        output.extend(tail_chosen)
        bucket_counts["mandatory_mass_flow_tail"] += len(tail_chosen)
        bucket_counts.update(
            {
                f"mandatory_mass_flow_tail:{key}": value
                for key, value in tail_bins.items()
            }
        )
        label_counts.update(
            {
                f"mandatory_mass_flow_tail:{key}": value
                for key, value in tail_labels.items()
            }
        )
        base_remaining = max(0, int(self.base_epoch_size) - len(output))
        base_quotas = self._process_quotas_for(base_remaining)
        mixture = {
            "uniform": float(self.cfg.mixture.uniform),
            "sparse_positive": float(self.cfg.mixture.sparse_positive),
            "quantile_balance": float(self.cfg.mixture.quantile_balance),
            "mass_flow_tail": float(self.cfg.mixture.mass_flow_tail),
            "hard_target_edges": float(getattr(self.cfg.mixture, "hard_target_edges", 0.0)),
        }
        if bool(getattr(self.cfg.mass_flow_tail, "enabled", False)):
            mixture["mass_flow_tail"] = 0.0
        if bool(getattr(self.cfg, "hard_fill_base_exclude_hard_target_edges", True)):
            mixture["hard_target_edges"] = 0.0
        if sum(max(0.0, value) for value in mixture.values()) <= 0.0:
            mixture = {"uniform": 1.0, "sparse_positive": 0.0, "quantile_balance": 0.0, "mass_flow_tail": 0.0, "hard_target_edges": 0.0}

        for process_num in self.process_ids:
            pquota = int(base_quotas.get(process_num, 0))
            process_start_count = sum(
                1
                for idx in output
                if self.sample_info[int(idx)]["process_num"] == process_num
            )
            bquotas = _largest_remainder(pquota, mixture)
            for bucket in ("uniform", "sparse_positive", "quantile_balance", "mass_flow_tail", "hard_target_edges"):
                allow_replacement = (
                    bool(getattr(self.cfg, "replacement_for_uniform", False))
                    if bucket == "uniform"
                    else bool(getattr(self.cfg, "replacement_for_rare_buckets", True))
                )
                chosen, repl, counts = self._choose_from_bucket(
                    rng=rng,
                    pool_by_label=dict(self.buckets[bucket].get(process_num, {})),
                    quota=int(bquotas.get(bucket, 0)),
                    selected=selected,
                    allow_replacement=allow_replacement and not bool(getattr(self.cfg, "unique_within_epoch", True)),
                )
                output.extend(chosen)
                replacement_count += repl
                bucket_counts[f"base_{bucket}"] += len(chosen)
                label_counts.update({f"base_{bucket}:{k}": v for k, v in counts.items()})
            process_end_count = sum(
                1
                for idx in output
                if self.sample_info[int(idx)]["process_num"] == process_num
            )
            missing = pquota - (process_end_count - process_start_count)
            if missing > 0:
                fallback_pool = [idx for idx in self.process_to_indices[process_num] if idx not in selected]
                rng.shuffle(fallback_pool)
                fill = fallback_pool[:missing]
                output.extend(fill)
                selected.update(fill)
                bucket_counts["base_fallback_uniform"] += len(fill)
            process_base_counts[str(process_num)] = sum(
                1 for idx in output if self.sample_info[int(idx)]["process_num"] == process_num
            )

        total_size = int(self.epoch_size)
        target_hard_total = max(0, int(round(total_size * float(getattr(self.cfg, "hard_fill_ratio", 0.40)))))
        hard_needed = min(max(0, total_size - len(output)), target_hard_total)
        global_hard_pool: dict[str, list[int]] = defaultdict(list)
        for process_num in self.process_ids:
            for label, pool in self.buckets["hard_target_edges"].get(process_num, {}).items():
                if pool:
                    global_hard_pool[f"P{int(process_num):02d}:{label}"].extend(int(idx) for idx in pool)
        property_quotas = dict(
            getattr(
                getattr(self.cfg, "hard_target_edges", None),
                "property_quotas",
                {},
            )
            or {}
        )
        hard_chosen: list[int] = []
        hard_repl = 0
        hard_labels: Counter[str] = Counter()
        hard_property_counts: Counter[str] = Counter()
        hard_property_shortfalls: Counter[str] = Counter()
        hard_rule_counts: Counter[str] = Counter()
        hard_rule_shortfalls: Counter[str] = Counter()
        allow_hard_replacement = bool(
            getattr(self.cfg, "replacement_for_rare_buckets", True)
        ) and not bool(getattr(self.cfg, "unique_within_epoch", True))
        if property_quotas:
            configured_total = 0
            rule_quotas = dict(
                getattr(
                    getattr(self.cfg, "hard_target_edges", None),
                    "rule_quotas",
                    {},
                )
                or {}
            )
            rules_by_name = {
                str(rule.get("name", "")): rule
                for rule in list(
                    getattr(
                        getattr(self.cfg, "hard_target_edges", None),
                        "rules",
                        [],
                    )
                    or []
                )
            }
            for property_name, requested in property_quotas.items():
                property_name = str(property_name)
                requested = max(0, int(requested))
                configured_total += requested
                property_pool = {
                    label: pool
                    for label, pool in global_hard_pool.items()
                    if label.rsplit(":", 1)[-1] == property_name
                }
                property_start = len(hard_chosen)
                quota_rule_names: set[str] = set()
                for rule_name, rule_requested in rule_quotas.items():
                    rule = rules_by_name.get(str(rule_name), {})
                    if str(rule.get("property", "")) != property_name:
                        continue
                    quota_rule_names.add(str(rule_name))
                    rule_requested = max(0, int(rule_requested))
                    rule_pool = {
                        label: pool
                        for label, pool in property_pool.items()
                        if len(label.split(":", 3)) == 4
                        and label.split(":", 3)[1] == str(rule_name)
                    }
                    chosen, repl, labels = self._choose_from_bucket(
                        rng=rng,
                        pool_by_label=rule_pool,
                        quota=min(
                            rule_requested,
                            max(0, requested - (len(hard_chosen) - property_start)),
                            max(0, hard_needed - len(hard_chosen)),
                        ),
                        selected=selected,
                        allow_replacement=allow_hard_replacement,
                    )
                    hard_chosen.extend(chosen)
                    hard_repl += repl
                    hard_labels.update(labels)
                    hard_rule_counts[str(rule_name)] += len(chosen)
                    hard_rule_shortfalls[str(rule_name)] += max(
                        0, rule_requested - len(chosen)
                    )

                property_remaining = max(
                    0, requested - (len(hard_chosen) - property_start)
                )
                if property_remaining:
                    unallocated_property_pool = {
                        label: pool
                        for label, pool in property_pool.items()
                        if len(label.split(":", 3)) != 4
                        or label.split(":", 3)[1] not in quota_rule_names
                    }
                    chosen, repl, labels = self._choose_from_bucket(
                        rng=rng,
                        pool_by_label=unallocated_property_pool,
                        quota=min(
                            property_remaining,
                            max(0, hard_needed - len(hard_chosen)),
                        ),
                        selected=selected,
                        allow_replacement=allow_hard_replacement,
                    )
                    hard_chosen.extend(chosen)
                    hard_repl += repl
                    hard_labels.update(labels)

                property_selected = len(hard_chosen) - property_start
                hard_property_counts[property_name] += property_selected
                hard_property_shortfalls[property_name] += max(
                    0, requested - property_selected
                )
            generic_quota = max(
                0,
                hard_needed - min(hard_needed, configured_total),
            )
            if generic_quota:
                chosen, repl, labels = self._choose_from_bucket(
                    rng=rng,
                    pool_by_label=dict(global_hard_pool),
                    quota=generic_quota,
                    selected=selected,
                    allow_replacement=allow_hard_replacement,
                )
                hard_chosen.extend(chosen)
                hard_repl += repl
                hard_labels.update(labels)
        else:
            hard_chosen, hard_repl, hard_labels = self._choose_from_bucket(
                rng=rng,
                pool_by_label=dict(global_hard_pool),
                quota=hard_needed,
                selected=selected,
                allow_replacement=allow_hard_replacement,
            )
        output.extend(hard_chosen)
        replacement_count += hard_repl
        bucket_counts["hard_fill"] += len(hard_chosen)
        label_counts.update({f"hard_fill:{k}": v for k, v in hard_labels.items()})
        for idx in hard_chosen:
            process_hard_counts[str(self.sample_info[int(idx)]["process_num"])] += 1
        shortfall = max(0, hard_needed - len(hard_chosen))
        if shortfall:
            hard_fallback_to_random += shortfall

        if len(output) < total_size:
            rest = [idx for idx in range(len(self.dataset)) if idx not in selected]
            rng.shuffle(rest)
            fill = rest[: total_size - len(output)]
            output.extend(fill)
            selected.update(fill)
            bucket_counts["hard_shortfall_random_fallback"] += len(fill)

        output = output[:total_size]
        rng.shuffle(output)
        for idx in output:
            self.selection_counts[int(idx)] += 1
            process_counts[str(self.sample_info[int(idx)]["process_num"])] += 1
        coverage = len(self.selection_counts) / max(1, len(self.dataset))
        stat = {
            "fold_id": int(self.fold_id),
            "epoch": int(self.epoch + 1),
            "mode": "base_plus_hard_fill",
            "seed": int(seed),
            "train_pool_size": int(len(self.dataset)),
            "base_epoch_train_size": int(self.base_epoch_size),
            "epoch_train_size": int(self.epoch_size),
            "base_selected_count": int(sum(process_base_counts.values())),
            "random_ratio": float(0.0),
            "hard_ratio": float(getattr(self.cfg, "hard_fill_ratio", 0.40)),
            "hard_target_count": int(target_hard_total),
            "hard_property_counts": dict(sorted(hard_property_counts.items())),
            "hard_property_shortfalls": dict(
                sorted(hard_property_shortfalls.items())
            ),
            "hard_rule_counts": dict(sorted(hard_rule_counts.items())),
            "hard_rule_shortfalls": dict(sorted(hard_rule_shortfalls.items())),
            "mass_flow_tail_requested_count": int(
                getattr(self.cfg.mass_flow_tail, "target_items_per_epoch", 0)
            ),
            "mass_flow_tail_selected_unique_count": int(len(set(tail_chosen))),
            "mass_flow_tail_shortfall_count": int(tail_shortfall),
            "mass_flow_tail_bin_counts": dict(sorted(tail_bins.items())),
            "random_selected_count": int(len(output) - bucket_counts.get("hard_fill", 0)),
            "hard_selected_count": int(bucket_counts.get("hard_fill", 0)),
            "hard_fallback_to_random_count": int(hard_fallback_to_random + bucket_counts.get("hard_shortfall_random_fallback", 0)),
            "selected_unique_count": int(len(set(output))),
            "duplicate_sample_count": int(len(output) - len(set(output))),
            "replacement_count": int(replacement_count + len(output) - len(set(output))),
            "process_counts": dict(sorted(process_counts.items(), key=lambda item: int(item[0]))),
            "process_base_counts": dict(sorted(process_base_counts.items(), key=lambda item: int(item[0]))),
            "process_hard_counts": dict(sorted(process_hard_counts.items(), key=lambda item: int(item[0]))),
            "process_hard_fallback_counts": dict(sorted(process_hard_fallback_counts.items(), key=lambda item: int(item[0]))),
            "bucket_counts": dict(sorted(bucket_counts.items())),
            "bucket_label_counts": dict(sorted(label_counts.items())),
            "cumulative_unique_selected": int(len(self.selection_counts)),
            "cumulative_unique_coverage": float(coverage),
        }
        self._write_stats(stat, output)
        return output

    def _iter_scheduled_predefined_hard(self, *, rng: random.Random, seed: int) -> list[int]:
        selected: set[int] = set()
        output: list[int] = []
        process_quotas = self._process_quotas()
        hard_ratio = self._scheduled_hard_ratio()
        random_ratio = 1.0 - hard_ratio
        replacement_count = 0
        hard_fallback_to_random = 0
        process_counts: Counter[str] = Counter()
        process_random_counts: Counter[str] = Counter()
        process_hard_counts: Counter[str] = Counter()
        process_hard_fallback_counts: Counter[str] = Counter()
        bucket_counts: Counter[str] = Counter()
        label_counts: Counter[str] = Counter()

        for process_num in self.process_ids:
            pquota = int(process_quotas.get(process_num, 0))
            quotas = _largest_remainder(pquota, {"random": random_ratio, "hard": hard_ratio})
            hard_quota = int(quotas.get("hard", 0))
            random_quota = int(quotas.get("random", 0))

            hard_chosen, hard_repl, hard_labels = self._choose_from_bucket(
                rng=rng,
                pool_by_label=dict(self.buckets["hard_target_edges"].get(process_num, {})),
                quota=hard_quota,
                selected=selected,
                allow_replacement=bool(getattr(self.cfg, "replacement_for_rare_buckets", True))
                and not bool(getattr(self.cfg, "unique_within_epoch", True)),
            )
            output.extend(hard_chosen)
            replacement_count += hard_repl
            bucket_counts["hard"] += len(hard_chosen)
            process_hard_counts[str(process_num)] += len(hard_chosen)
            label_counts.update({f"hard:{k}": v for k, v in hard_labels.items()})

            hard_shortfall = max(0, hard_quota - len(hard_chosen))
            if hard_shortfall:
                random_quota += hard_shortfall
                hard_fallback_to_random += hard_shortfall
                process_hard_fallback_counts[str(process_num)] += hard_shortfall

            random_pool = {
                "all": [
                    idx
                    for idx in self.process_to_indices[process_num]
                    if idx not in selected or not bool(getattr(self.cfg, "unique_within_epoch", True))
                ]
            }
            random_chosen, random_repl, random_labels = self._choose_from_bucket(
                rng=rng,
                pool_by_label=random_pool,
                quota=random_quota,
                selected=selected,
                allow_replacement=bool(getattr(self.cfg, "replacement_for_uniform", False))
                and not bool(getattr(self.cfg, "unique_within_epoch", True)),
            )
            output.extend(random_chosen)
            replacement_count += random_repl
            bucket_counts["random"] += len(random_chosen)
            process_random_counts[str(process_num)] += len(random_chosen)
            label_counts.update({f"random:{k}": v for k, v in random_labels.items()})

            missing = pquota - sum(1 for idx in output if self.sample_info[idx]["process_num"] == process_num)
            if missing > 0:
                fallback_pool = [idx for idx in self.process_to_indices[process_num] if idx not in selected]
                rng.shuffle(fallback_pool)
                fill = fallback_pool[:missing]
                output.extend(fill)
                selected.update(fill)
                bucket_counts["process_random_fallback"] += len(fill)
                process_random_counts[str(process_num)] += len(fill)
                hard_fallback_to_random += len(fill)
                process_hard_fallback_counts[str(process_num)] += len(fill)

        if len(output) < self.epoch_size:
            rest = [idx for idx in range(len(self.dataset)) if idx not in selected]
            rng.shuffle(rest)
            fill = rest[: self.epoch_size - len(output)]
            output.extend(fill)
            selected.update(fill)
            bucket_counts["global_fallback"] += len(fill)
            for idx in fill:
                process_random_counts[str(self.sample_info[int(idx)]["process_num"])] += 1
        output = output[: self.epoch_size]
        rng.shuffle(output)
        for idx in output:
            process_key = str(self.sample_info[int(idx)]["process_num"])
            self.selection_counts[int(idx)] += 1
            process_counts[process_key] += 1

        process_quota_rows = {
            str(pid): {
                "total_quota": int(quota),
                "random_quota": int(_largest_remainder(int(quota), {"random": random_ratio, "hard": hard_ratio}).get("random", 0)),
                "hard_quota": int(_largest_remainder(int(quota), {"random": random_ratio, "hard": hard_ratio}).get("hard", 0)),
                "random_selected": int(process_random_counts.get(str(pid), 0)),
                "hard_selected": int(process_hard_counts.get(str(pid), 0)),
                "hard_fallback_to_random": int(process_hard_fallback_counts.get(str(pid), 0)),
                "selected_total": int(process_counts.get(str(pid), 0)),
            }
            for pid, quota in sorted(process_quotas.items())
        }
        coverage = len(self.selection_counts) / max(1, len(self.dataset))
        stat = {
            "fold_id": int(self.fold_id),
            "epoch": int(self.epoch + 1),
            "mode": "scheduled_predefined_hard",
            "seed": int(seed),
            "train_pool_size": int(len(self.dataset)),
            "epoch_train_size": int(self.epoch_size),
            "random_ratio": float(random_ratio),
            "hard_ratio": float(hard_ratio),
            "random_selected_count": int(bucket_counts.get("random", 0) + bucket_counts.get("process_random_fallback", 0) + bucket_counts.get("global_fallback", 0)),
            "hard_selected_count": int(bucket_counts.get("hard", 0)),
            "hard_fallback_to_random_count": int(hard_fallback_to_random),
            "selected_unique_count": int(len(set(output))),
            "duplicate_sample_count": int(len(output) - len(set(output))),
            "replacement_count": int(replacement_count + len(output) - len(set(output))),
            "process_counts": dict(sorted(process_counts.items(), key=lambda item: int(item[0]))),
            "process_quota_counts": process_quota_rows,
            "bucket_counts": dict(sorted(bucket_counts.items())),
            "bucket_label_counts": dict(sorted(label_counts.items())),
            "cumulative_unique_selected": int(len(self.selection_counts)),
            "cumulative_unique_coverage": float(coverage),
        }
        self._write_stats(stat, output)
        return output
