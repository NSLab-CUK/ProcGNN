#!/usr/bin/env python3
"""Run nested-subset transfer versus scratch data-efficiency experiments.

The script reuses the existing single-process full-unseen split, training entry
point, completed pretrain checkpoints, and zero-shot artifacts. Only the
held-out-process train manifest is reduced; the target test manifest is never
changed. Transfer and scratch runs reference the exact same subset manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_single_process_full_unseen_experiments as full_unseen  # noqa: E402
from process_graph.experiment.loaders import load_experiment_config  # noqa: E402

REQUIRED_MANIFEST_COLUMNS = ("process_id", "sample_id", "merged_row_index")
DEFAULT_RATIOS = (
    0.02,
    0.04,
    0.06,
    0.08,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _rel(path: Path) -> str:
    return full_unseen._rel(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")


def _read_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_MANIFEST_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"manifest is empty: {path}")
    indices = pd.to_numeric(frame["merged_row_index"], errors="raise").astype(int)
    if indices.duplicated().any():
        raise ValueError(f"manifest has duplicate merged_row_index values: {path}")
    return frame


def _ratio_count(pool_size: int, ratio: float) -> int:
    rounded = int((Decimal(str(ratio)) * Decimal(int(pool_size))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return min(int(pool_size), max(1, rounded))


def _ratio_label(ratio: float) -> str:
    percent = int((Decimal(str(ratio)) * Decimal(100)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return f"ratio_{percent:03d}"


def _validate_ratios(values: Sequence[float]) -> list[float]:
    ratios = sorted({float(value) for value in values})
    if not ratios or any(not math.isfinite(value) or value <= 0.0 or value > 1.0 for value in ratios):
        raise ValueError("data ratios must be finite values in (0, 1]")
    labels = [_ratio_label(value) for value in ratios]
    if len(labels) != len(set(labels)):
        raise ValueError(f"data ratios produce duplicate output labels: {ratios}")
    return ratios


def _indices_hash(values: Iterable[int]) -> str:
    text = "\n".join(str(int(value)) for value in values)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _process_key(value: Any) -> int:
    text = str(value).strip().lower().replace("process", "").strip()
    return int(float(text))


def _sample_key(value: Any) -> str:
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    return str(int(numeric)) if numeric.is_integer() else text


def _excluded_sample_keys(path: Path | None) -> set[tuple[int, str]]:
    if path is None:
        return set()
    if not path.is_file():
        raise FileNotFoundError(f"excluded graph samples file not found: {path}")
    frame = pd.read_csv(path)
    missing = {"process_id", "sample_id"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return {
        (_process_key(row.process_id), _sample_key(row.sample_id))
        for row in frame.itertuples(index=False)
    }


def _remove_excluded_samples(
    frame: pd.DataFrame,
    excluded_keys: set[tuple[int, str]],
) -> tuple[pd.DataFrame, int]:
    if not excluded_keys:
        return frame, 0
    keep = [
        (_process_key(row.process_id), _sample_key(row.sample_id)) not in excluded_keys
        for row in frame.itertuples(index=False)
    ]
    filtered = frame.loc[keep].reset_index(drop=True)
    return filtered, int(len(frame) - len(filtered))


def _write_stable_csv(frame: pd.DataFrame, path: Path, *, force: bool) -> None:
    csv_text = frame.to_csv(index=False)
    if path.is_file():
        try:
            current_frame = pd.read_csv(path)
            same_content = list(current_frame.columns) == list(frame.columns) and current_frame.equals(
                frame.reset_index(drop=True)
            )
        except Exception:
            same_content = False
        if not same_content and not force:
            raise RuntimeError(f"existing subset differs from requested deterministic subset: {path}")
        if same_content:
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(csv_text)


@dataclass(frozen=True)
class SubsetSpec:
    heldout_process: int
    fold: int
    ratio: float
    ratio_label: str
    transfer_pool_size: int
    sample_count: int
    subset_manifest: Path
    subset_indices_path: Path
    subset_hash: str
    target_val_manifest: Path
    target_val_size: int
    target_test_manifest: Path
    target_test_size: int
    permutation_seed: int


def prepare_nested_subsets(
    *,
    split_root: Path,
    output_root: Path,
    heldout_processes: Sequence[int],
    folds: Sequence[int],
    ratios: Sequence[float],
    seed: int,
    force: bool,
    excluded_graph_samples_path: Path | None = None,
) -> tuple[list[SubsetSpec], pd.DataFrame]:
    specs: list[SubsetSpec] = []
    pool_rows: list[dict[str, Any]] = []
    subset_root = output_root / "subsets"
    excluded_keys = _excluded_sample_keys(excluded_graph_samples_path)
    for heldout in heldout_processes:
        for fold in folds:
            fold_dir = split_root / f"heldout_P{heldout:02d}" / f"fold_{fold:02d}"
            source_manifest = fold_dir / "target_train.csv"
            target_val = fold_dir / "target_val.csv"
            target_test = fold_dir / "target_test.csv"
            raw_pool = _read_manifest(source_manifest)
            raw_val = _read_manifest(target_val)
            raw_test = _read_manifest(target_test)
            pool, pool_excluded_count = _remove_excluded_samples(raw_pool, excluded_keys)
            val, val_excluded_count = _remove_excluded_samples(raw_val, excluded_keys)
            test, test_excluded_count = _remove_excluded_samples(raw_test, excluded_keys)
            pool_indices = set(pd.to_numeric(pool["merged_row_index"], errors="raise").astype(int).tolist())
            val_indices = set(pd.to_numeric(val["merged_row_index"], errors="raise").astype(int).tolist())
            test_indices = set(pd.to_numeric(test["merged_row_index"], errors="raise").astype(int).tolist())
            overlaps = {
                "train_val": pool_indices & val_indices,
                "train_test": pool_indices & test_indices,
                "val_test": val_indices & test_indices,
            }
            if any(overlaps.values()):
                raise RuntimeError(
                    f"heldout P{heldout:02d} fold {fold}: target split overlap "
                    + ", ".join(f"{name}={len(rows)}" for name, rows in overlaps.items())
                )

            permutation_seed = int(seed) + int(heldout) * 1000 + int(fold)
            permutation = np.random.default_rng(permutation_seed).permutation(len(pool))
            ordered = pool.iloc[permutation].reset_index(drop=True)
            ordered_indices = pd.to_numeric(ordered["merged_row_index"], errors="raise").astype(int).tolist()
            out_dir = subset_root / f"heldout_P{heldout:02d}" / f"fold_{fold:02d}"
            order_frame = ordered.copy()
            order_frame.insert(0, "subset_rank", np.arange(len(order_frame), dtype=np.int64))
            _write_stable_csv(order_frame, out_dir / "permutation.csv", force=force)

            previous_indices: set[int] = set()
            ratio_rows: list[dict[str, Any]] = []
            fold_specs: list[SubsetSpec] = []
            for ratio in ratios:
                count = _ratio_count(len(pool), ratio)
                label = _ratio_label(ratio)
                subset = ordered.iloc[:count].copy()
                indices = pd.to_numeric(subset["merged_row_index"], errors="raise").astype(int).tolist()
                index_set = set(indices)
                if not previous_indices.issubset(index_set):
                    raise AssertionError(f"nested subset invariant failed for P{heldout:02d} F{fold:02d} {label}")
                previous_indices = index_set
                manifest_path = out_dir / f"{label}.csv"
                indices_path = out_dir / f"{label}_indices.json"
                _write_stable_csv(subset, manifest_path, force=force)
                subset_hash = _indices_hash(indices)
                _write_json(
                    indices_path,
                    {
                        "heldout_process": int(heldout),
                        "fold": int(fold),
                        "sample_ratio": float(ratio),
                        "sample_count": int(count),
                        "transfer_pool_size": int(len(pool)),
                        "permutation_seed": int(permutation_seed),
                        "merged_row_indices_sha256": subset_hash,
                        "merged_row_indices": indices,
                    },
                )
                spec = SubsetSpec(
                    heldout_process=int(heldout),
                    fold=int(fold),
                    ratio=float(ratio),
                    ratio_label=label,
                    transfer_pool_size=int(len(pool)),
                    sample_count=int(count),
                    subset_manifest=manifest_path,
                    subset_indices_path=indices_path,
                    subset_hash=subset_hash,
                    target_val_manifest=target_val,
                    target_val_size=int(len(val)),
                    target_test_manifest=target_test,
                    target_test_size=int(len(test)),
                    permutation_seed=int(permutation_seed),
                )
                specs.append(spec)
                fold_specs.append(spec)
                ratio_rows.append(
                    {
                        "sample_ratio": float(ratio),
                        "sample_count": int(count),
                        "subset_manifest": _rel(manifest_path),
                        "subset_indices": _rel(indices_path),
                        "subset_hash": subset_hash,
                    }
                )

            nested_ok = all(
                set(
                    json.loads(spec.subset_indices_path.read_text(encoding="utf-8"))["merged_row_indices"]
                ).issubset(
                    set(
                        json.loads(fold_specs[index + 1].subset_indices_path.read_text(encoding="utf-8"))[
                            "merged_row_indices"
                        ]
                    )
                )
                for index, spec in enumerate(fold_specs[:-1])
            )
            metadata = {
                "heldout_process": int(heldout),
                "fold": int(fold),
                "seed": int(seed),
                "permutation_seed": int(permutation_seed),
                "source_transfer_manifest": _rel(source_manifest),
                "raw_transfer_pool_size": int(len(raw_pool)),
                "transfer_pool_size": int(len(pool)),
                "transfer_pool_excluded_count": int(pool_excluded_count),
                "target_test_manifest": _rel(target_test),
                "target_val_manifest": _rel(target_val),
                "validation_manifest": _rel(target_val),
                "test_manifest": _rel(target_test),
                "target_val_size": int(len(val)),
                "raw_target_val_size": int(len(raw_val)),
                "target_val_excluded_count": int(val_excluded_count),
                "target_test_size": int(len(test)),
                "raw_target_test_size": int(len(raw_test)),
                "target_test_excluded_count": int(test_excluded_count),
                "excluded_graph_samples_path": (
                    _rel(excluded_graph_samples_path) if excluded_graph_samples_path is not None else ""
                ),
                "validation_and_test_are_same_manifest": False,
                "train_val_disjoint": True,
                "train_test_disjoint": True,
                "val_test_disjoint": True,
                "train_indices_sha256": _indices_hash(sorted(pool_indices)),
                "validation_indices_sha256": _indices_hash(sorted(val_indices)),
                "test_indices_sha256": _indices_hash(sorted(test_indices)),
                "nested_subset": bool(nested_ok),
                "ratios": ratio_rows,
            }
            _write_json(out_dir / "subset_metadata.json", metadata)
            pool_rows.append(
                {
                    "process_id": int(heldout),
                    "fold": int(fold),
                    "raw_transfer_pool_size": int(len(raw_pool)),
                    "transfer_pool_size": int(len(pool)),
                    "transfer_pool_excluded_count": int(pool_excluded_count),
                    "validation_size": int(len(val)),
                    "test_size": int(len(test)),
                    "target_val_excluded_count": int(val_excluded_count),
                    "target_test_excluded_count": int(test_excluded_count),
                    "validation_and_test_are_same_manifest": False,
                    "train_val_disjoint": True,
                    "train_test_disjoint": True,
                    "val_test_disjoint": True,
                    "nested_subset": bool(nested_ok),
                    **{
                        f"n_{int(round(ratio * 100)):03d}": _ratio_count(len(pool), ratio)
                        for ratio in ratios
                    },
                }
            )
    pool_frame = pd.DataFrame(pool_rows).sort_values(["process_id", "fold"]).reset_index(drop=True)
    aggregate_dir = output_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    pool_path = aggregate_dir / "transfer_pool_sizes.csv"
    if pool_path.is_file():
        existing_pool = pd.read_csv(pool_path)
        combined_pool = pd.concat((existing_pool, pool_frame), ignore_index=True)
        combined_pool = combined_pool.drop_duplicates(["process_id", "fold"], keep="last")
    else:
        combined_pool = pool_frame
    combined_pool = combined_pool.sort_values(["process_id", "fold"]).reset_index(drop=True)
    combined_pool.to_csv(pool_path, index=False)
    process_stats = (
        combined_pool.groupby("process_id")["transfer_pool_size"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )
    process_stats.to_csv(aggregate_dir / "transfer_pool_size_by_process.csv", index=False)
    return specs, pool_frame


def _latest_artifact_dir(stage_dir: Path) -> Path | None:
    metrics_path = full_unseen._latest_file(stage_dir, "metrics.json")
    return metrics_path.parent if metrics_path is not None else None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _target_property_rows(artifact_dir: Path, split: str) -> pd.DataFrame:
    path = artifact_dir / "target_edge_r2_by_property.csv"
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "split" in frame.columns:
        frame = frame[frame["split"].astype(str) == str(split)].copy()
    return frame


def _target_mean_r2(
    artifact_dir: Path,
    *,
    split: str,
    excluded_properties: set[str],
    metrics_payload: Mapping[str, Any],
) -> float:
    frame = _target_property_rows(artifact_dir, split)
    if not frame.empty and {"property_name", "R2"}.issubset(frame.columns):
        names = frame["property_name"].astype(str).str.casefold()
        values = pd.to_numeric(frame["R2"], errors="coerce")
        keep = values.notna() & np.isfinite(values) & ~names.isin(excluded_properties)
        if bool(keep.any()):
            return float(values[keep].mean())
    candidates = (
        f"{split}_target_edge_property_mean_r2",
        "val_target_edge_property_mean_r2",
        "test_target_edge_property_mean_r2",
        "target_edge_property_mean_r2",
    )
    for key in candidates:
        try:
            value = float(metrics_payload[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value

    # Compatibility fallback for older artifacts that predate pooled-property metrics.
    target_rows_path = artifact_dir / "target_r2.csv"
    if target_rows_path.is_file():
        frame = pd.read_csv(target_rows_path)
        if "split" in frame.columns:
            frame = frame[frame["split"].astype(str) == str(split)].copy()
        feature_column = "feature_name" if "feature_name" in frame.columns else "property_name"
        r2_column = "r2" if "r2" in frame.columns else "R2"
        member_column = "target_mean_r2_member" if "target_mean_r2_member" in frame.columns else "used_in_mean"
        if {feature_column, r2_column, member_column}.issubset(frame.columns):
            names = frame[feature_column].astype(str).str.casefold()
            values = pd.to_numeric(frame[r2_column], errors="coerce")
            members = frame[member_column].astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})
            keep = members & values.notna() & np.isfinite(values) & ~names.isin(excluded_properties)
            if bool(keep.any()):
                return float(values[keep].mean())
    candidates = (
        f"{split}_target_mean_r2",
        "val_target_mean_r2",
        "test_target_mean_r2",
        "target_mean_r2",
        "eval_target_mean_r2",
    )
    for key in candidates:
        try:
            value = float(metrics_payload[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return math.nan


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _training_time_sec(run_dir: Path) -> float:
    status = _load_json(run_dir / "status.json")
    try:
        value = float(status.get("duration_sec"))
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    return math.nan


def _actual_optimizer_steps(artifact_dir: Path) -> int:
    """Read the cumulative optimizer-step count written for the final epoch."""
    path = artifact_dir / "metrics_per_epoch.csv"
    if not path.is_file():
        return 0
    frame = pd.read_csv(path)
    for column in (
        "train_optimizer_steps_total",
        "train/optimizer_steps_total",
        "optimizer_steps_total",
    ):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if not values.empty:
            return int(round(float(values.max())))
    return 0


def write_patience_pilot_diagnostics(output_root: Path) -> None:
    """Summarize the fixed patience pilot without changing its hyperparameters."""
    epoch_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for process_id in (1, 3, 6):
        for fold_dir in sorted((output_root / f"heldout_P{process_id:02d}").glob("fold_*")):
            fold = int(fold_dir.name.split("_")[-1])
            for mode in ("transfer", "scratch"):
                for ratio_label in ("ratio_005", "ratio_010", "ratio_025"):
                    run_dir = fold_dir / mode / ratio_label
                    artifact_dir = _latest_artifact_dir(run_dir)
                    metrics_path = artifact_dir / "metrics_per_epoch.csv" if artifact_dir else None
                    if metrics_path is None or not metrics_path.is_file():
                        continue
                    frame = pd.read_csv(metrics_path)
                    metric_column = next(
                        (
                            name
                            for name in (
                                "val_target_edge_property_mean_r2",
                                "val_target_mean_r2",
                            )
                            if name in frame.columns
                        ),
                        None,
                    )
                    if metric_column is None:
                        continue
                    metric = pd.to_numeric(frame[metric_column], errors="coerce")
                    valid = metric.dropna()
                    best_row_index = int(valid.idxmax()) if not valid.empty else int(len(frame) - 1)
                    best_epoch = int(frame.loc[best_row_index, "epoch"])
                    stopped_epoch = int(pd.to_numeric(frame["epoch"], errors="raise").max())
                    metadata = _load_json(run_dir / "run_metadata.json")
                    max_epochs = int(metadata.get("max_epochs", stopped_epoch))
                    patience = int(metadata.get("early_stopping_patience", 3))
                    last_values = valid.tail(2).tolist()
                    rising_before_stop = len(last_values) == 2 and last_values[-1] > last_values[-2]
                    stopped_early = stopped_epoch < max_epochs
                    premature_risk = bool(
                        mode == "scratch"
                        and stopped_early
                        and (stopped_epoch <= 6 or rising_before_stop)
                    )
                    for _, row in frame.iterrows():
                        epoch_rows.append(
                            {
                                "process_id": process_id,
                                "fold": fold,
                                "mode": mode,
                                "sample_ratio": int(ratio_label[-3:]) / 100.0,
                                "epoch": int(row["epoch"]),
                                "train_loss": _safe_float(row.get("train_loss")),
                                "validation_loss": _safe_float(row.get("val_loss")),
                                "val_target_edge_property_mean_r2": _safe_float(
                                    row.get(metric_column)
                                ),
                                "best_epoch": best_epoch,
                                "stopped_epoch": stopped_epoch,
                            }
                        )
                    expected_steps = int(metadata.get("total_optimizer_steps", 0) or 0)
                    actual_steps = _actual_optimizer_steps(artifact_dir)
                    if actual_steps <= 0:
                        raise RuntimeError(
                            f"optimizer-step count missing for {run_dir}: actual={actual_steps}"
                        )
                    if expected_steps > 0 and actual_steps > expected_steps:
                        raise RuntimeError(
                            f"optimizer-step cap exceeded for {run_dir}: "
                            f"cap={expected_steps} actual={actual_steps}"
                        )
                    summary_rows.append(
                        {
                            "process_id": process_id,
                            "fold": fold,
                            "mode": mode,
                            "sample_ratio": int(ratio_label[-3:]) / 100.0,
                            "max_epochs": max_epochs,
                            "patience": patience,
                            "best_epoch": best_epoch,
                            "stopped_epoch": stopped_epoch,
                            "stopped_early": stopped_early,
                            "rising_immediately_before_stop": rising_before_stop,
                            "premature_stop_risk": premature_risk,
                        }
                    )
    aggregate_dir = output_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(epoch_rows).to_csv(aggregate_dir / "patience_pilot_epoch_history.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(aggregate_dir / "patience_pilot_summary.csv", index=False)


def _append_property_rows(
    output: list[dict[str, Any]],
    *,
    artifact_dir: Path,
    split: str,
    base: Mapping[str, Any],
) -> None:
    frame = _target_property_rows(artifact_dir, split)
    for _, row in frame.iterrows():
        try:
            r2 = float(row.get("R2"))
        except (TypeError, ValueError):
            r2 = math.nan
        output.append(
            {
                **base,
                "metric_split": str(split),
                "property_name": str(row.get("property_name", "")),
                "r2": r2,
                "n": row.get("n", ""),
                "sst": row.get("SST", ""),
                "sse": row.get("SSE", ""),
            }
        )


def _append_edge_rows(
    output: list[dict[str, Any]],
    *,
    artifact_dir: Path,
    split: str,
    base: Mapping[str, Any],
) -> None:
    path = artifact_dir / "target_r2.csv"
    if not path.is_file():
        return
    frame = pd.read_csv(path)
    if "split" in frame.columns:
        frame = frame[frame["split"].astype(str) == str(split)].copy()
    wanted = (
        "process_id",
        "target_id",
        "canonical_edge_id",
        "target_stream",
        "feature_name",
        "n",
        "sst",
        "sse",
        "mae",
        "rmse",
        "r2",
        "used_in_mean",
        "exclude_reason",
    )
    for _, row in frame.iterrows():
        output.append({**base, "metric_split": str(split), **{key: row.get(key, "") for key in wanted}})


def aggregate_results(
    *,
    output_root: Path,
    existing_unseen_root: Path,
    heldout_processes: Sequence[int],
    folds: Sequence[int],
    ratios: Sequence[float],
    excluded_properties: set[str],
) -> None:
    run_rows: list[dict[str, Any]] = []
    property_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []

    for heldout in heldout_processes:
        for fold in folds:
            subset_metadata = _load_json(
                output_root / "subsets" / f"heldout_P{heldout:02d}" / f"fold_{fold:02d}" / "subset_metadata.json"
            )
            transfer_pool_size = int(subset_metadata.get("transfer_pool_size", 0))
            zero_dir = existing_unseen_root / f"heldout_P{heldout:02d}" / f"fold_{fold:02d}" / "zero_shot"
            zero_artifact = _latest_artifact_dir(zero_dir)
            if zero_artifact is not None:
                payload = _load_json(zero_artifact / "metrics.json")
                score = _target_mean_r2(
                    zero_artifact,
                    split="test",
                    excluded_properties=excluded_properties,
                    metrics_payload=payload,
                )
                base = {
                    "process_id": int(heldout),
                    "fold": int(fold),
                    "mode": "zero_shot",
                    "sample_ratio": 0.0,
                    "sample_count": 0,
                    "transfer_pool_size": transfer_pool_size,
                }
                run_rows.append(
                    {
                        **base,
                        "target_edge_property_mean_r2": score,
                        "val_target_edge_property_mean_r2": math.nan,
                        "test_target_edge_property_mean_r2": score,
                        "best_epoch": int(payload.get("best_epoch", -1)),
                        "final_epoch": int(payload.get("final_epoch", 0)),
                        "training_time_sec": 0.0,
                        "pretrained_checkpoint": str(
                            _load_json(zero_dir / "evaluation_metadata.json").get("checkpoint_path", "")
                        ),
                        "subset_manifest": "",
                        "subset_hash": "",
                        "run_dir": _rel(zero_dir),
                        "artifact_dir": _rel(zero_artifact),
                        "metric_source": "existing_zero_shot_target_edge_property_mean_r2",
                        "validation_and_test_are_same_manifest": False,
                    }
                )
                _append_property_rows(property_rows, artifact_dir=zero_artifact, split="test", base=base)
                _append_edge_rows(edge_rows, artifact_dir=zero_artifact, split="test", base=base)

            for ratio in ratios:
                label = _ratio_label(ratio)
                for mode in ("transfer", "scratch"):
                    run_dir = output_root / f"heldout_P{heldout:02d}" / f"fold_{fold:02d}" / mode / label
                    if not full_unseen._completed(run_dir, kind="train"):
                        continue
                    artifact_dir = _latest_artifact_dir(run_dir)
                    if artifact_dir is None:
                        continue
                    metadata = _load_json(run_dir / "run_metadata.json")
                    payload = _load_json(artifact_dir / "metrics.json")
                    scaler_provenance = _load_json(artifact_dir / "scaler_fit_provenance.json")
                    expected_subset_hash = str(metadata.get("subset_hash", ""))
                    actual_scaler_hash = str(scaler_provenance.get("merged_row_indices_sha256", ""))
                    if not expected_subset_hash or actual_scaler_hash != expected_subset_hash:
                        raise RuntimeError(
                            f"scaler fit provenance mismatch for {run_dir}: "
                            f"expected={expected_subset_hash} actual={actual_scaler_hash}"
                        )
                    expected_steps = int(metadata.get("total_optimizer_steps", 0) or 0)
                    actual_steps = _actual_optimizer_steps(artifact_dir)
                    if actual_steps <= 0:
                        raise RuntimeError(
                            f"optimizer-step count missing for {run_dir}: actual={actual_steps}"
                        )
                    if expected_steps > 0 and actual_steps > expected_steps:
                        raise RuntimeError(
                            f"optimizer-step cap exceeded for {run_dir}: "
                            f"cap={expected_steps} actual={actual_steps}"
                        )
                    termination_condition = (
                        "optimizer_step_cap"
                        if expected_steps > 0 and actual_steps >= expected_steps
                        else "max_epochs"
                    )
                    val_score = _target_mean_r2(
                        artifact_dir,
                        split="val",
                        excluded_properties=excluded_properties,
                        metrics_payload=payload,
                    )
                    test_score = _target_mean_r2(
                        artifact_dir,
                        split="test",
                        excluded_properties=excluded_properties,
                        metrics_payload=payload,
                    )
                    base = {
                        "process_id": int(heldout),
                        "fold": int(fold),
                        "mode": mode,
                        "sample_ratio": float(ratio),
                        "sample_count": int(metadata.get("sample_count", 0)),
                        "transfer_pool_size": int(metadata.get("transfer_pool_size", 0)),
                    }
                    run_rows.append(
                        {
                            **base,
                            "target_edge_property_mean_r2": test_score,
                            "val_target_edge_property_mean_r2": val_score,
                            "test_target_edge_property_mean_r2": test_score,
                            "best_epoch": int(payload.get("best_epoch", -1)),
                            "final_epoch": int(payload.get("final_epoch", 0)),
                            "training_time_sec": _training_time_sec(run_dir),
                            "total_optimizer_steps": actual_steps,
                            "optimizer_step_cap": expected_steps,
                            "optimizer_step_cap_reached": bool(
                                expected_steps > 0 and actual_steps == expected_steps
                            ),
                            "termination_condition": termination_condition,
                            "pretrained_checkpoint": str(metadata.get("pretrained_checkpoint", "")),
                            "subset_manifest": str(metadata.get("subset_manifest", "")),
                            "subset_hash": str(metadata.get("subset_hash", "")),
                            "scaler_fit_indices_sha256": actual_scaler_hash,
                            "scaler_fit_is_current_training_subset": True,
                            "run_dir": _rel(run_dir),
                            "artifact_dir": _rel(artifact_dir),
                            "metric_source": "independent_target_test_target_edge_property_mean_r2",
                            "validation_and_test_are_same_manifest": False,
                        }
                    )
                    _append_property_rows(property_rows, artifact_dir=artifact_dir, split="test", base=base)
                    _append_edge_rows(edge_rows, artifact_dir=artifact_dir, split="test", base=base)

    aggregate_dir = output_root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(run_rows)
    runs.to_csv(aggregate_dir / "run_results.csv", index=False)
    pd.DataFrame(property_rows).to_csv(aggregate_dir / "property_results.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(aggregate_dir / "edge_results.csv", index=False)
    if runs.empty:
        print(f"[data-efficiency][aggregate] no completed results under {_rel(output_root)}", flush=True)
        return

    curve = (
        runs.groupby(["mode", "sample_ratio"])["target_edge_property_mean_r2"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["mode", "sample_ratio"])
    )
    curve.to_csv(aggregate_dir / "learning_curve_summary.csv", index=False)
    process_curve = (
        runs.groupby(["process_id", "mode", "sample_ratio"])["target_edge_property_mean_r2"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["process_id", "mode", "sample_ratio"])
    )
    process_curve.to_csv(aggregate_dir / "process_learning_curves.csv", index=False)

    trained = runs[runs["mode"].isin(["transfer", "scratch"])].copy()
    paired = trained.pivot_table(
        index=["process_id", "fold", "sample_ratio", "sample_count", "transfer_pool_size"],
        columns="mode",
        values="target_edge_property_mean_r2",
        aggfunc="first",
    ).reset_index()
    if {"transfer", "scratch"}.issubset(paired.columns):
        paired = paired.dropna(subset=["transfer", "scratch"]).copy()
        paired["transfer_gain"] = paired["transfer"] - paired["scratch"]
        paired.to_csv(aggregate_dir / "transfer_gain_by_run.csv", index=False)
        gain_summary = (
            paired.groupby("sample_ratio")["transfer_gain"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        gain_summary.to_csv(aggregate_dir / "transfer_gain_summary.csv", index=False)

    zero = runs[runs["mode"] == "zero_shot"][["process_id", "fold", "target_edge_property_mean_r2"]].rename(
        columns={"target_edge_property_mean_r2": "zero_shot_r2"}
    )
    transfer = runs[runs["mode"] == "transfer"].copy()
    adaptation = transfer.merge(zero, on=["process_id", "fold"], how="inner")
    if not adaptation.empty:
        adaptation["adaptation_gain"] = (
            adaptation["target_edge_property_mean_r2"] - adaptation["zero_shot_r2"]
        )
        adaptation.to_csv(aggregate_dir / "adaptation_gain_by_run.csv", index=False)
        adaptation_summary = (
            adaptation.groupby("sample_ratio")["adaptation_gain"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        adaptation_summary.to_csv(aggregate_dir / "adaptation_gain_summary.csv", index=False)

    properties = pd.DataFrame(property_rows)
    if not properties.empty:
        property_summary = (
            properties.groupby(["mode", "sample_ratio", "property_name"])["r2"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        property_summary.to_csv(aggregate_dir / "property_summary.csv", index=False)
    write_patience_pilot_diagnostics(output_root)
    print(f"[data-efficiency][aggregate] wrote {_rel(aggregate_dir)} runs={len(runs)}", flush=True)


def _print_design_audit(
    *,
    specs: Sequence[SubsetSpec],
    existing_unseen_root: Path,
) -> None:
    grouped: dict[tuple[int, int], list[SubsetSpec]] = {}
    for spec in specs:
        grouped.setdefault((spec.heldout_process, spec.fold), []).append(spec)
    for (heldout, fold), fold_specs in grouped.items():
        checkpoint = full_unseen._latest_completed_checkpoint(
            existing_unseen_root / f"heldout_P{heldout:02d}" / "pretrain"
        )
        first = fold_specs[0]
        print(
            f"[data-efficiency][design] Heldout Process {heldout} / Fold {fold} "
            f"transfer_pool_size={first.transfer_pool_size}",
            flush=True,
        )
        for spec in fold_specs:
            print(f"  ratio {spec.ratio:.2f} -> n={spec.sample_count}", flush=True)
        print("  nested subset = true", flush=True)
        print(f"  Transfer checkpoint = {_rel(checkpoint) if checkpoint else 'MISSING'}", flush=True)
        print(f"  Transfer-{fold_specs[0].ratio:.0%} train size = {fold_specs[0].sample_count}", flush=True)
        print(f"  Scratch-{fold_specs[0].ratio:.0%} train size = {fold_specs[0].sample_count}", flush=True)
        print("  same subset indices = true", flush=True)
        print(f"  validation size = {first.target_val_size}", flush=True)
        print(f"  test size = {first.target_test_size}", flush=True)
        print("  train/validation/test disjoint = true", flush=True)
        print("  validation/test manifest identical = false", flush=True)
        print("  subset-only training exposure = true", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run epoch-capped Proposed data-efficiency experiments.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--split-root", default="data/splits/single_process_full_unseen_60_20_20")
    parser.add_argument("--merged-csv", default="data/datasets_v3/process_main_merged.csv")
    parser.add_argument("--existing-unseen-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--heldout-processes", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--data-ratios", type=float, nargs="+", default=list(DEFAULT_RATIOS))
    parser.add_argument("--modes", nargs="+", choices=("transfer", "scratch"), default=["transfer"])
    parser.add_argument("--finetune-mode", choices=("full", "head_only"), default="full")
    parser.add_argument(
        "--max-epochs", type=int, default=30,
        help="Maximum epoch horizon; the optimizer-step cap can stop training earlier.",
    )
    parser.add_argument(
        "--total-optimizer-steps", type=int, default=80000,
        help="Maximum applied optimizer-step cap; training may stop at max epochs first.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--monitor-metric", default="val_target_edge_property_mean_r2")
    parser.add_argument("--monitor-mode", choices=("min", "max"), default="max")
    parser.add_argument("--seed", type=int, default=260716)
    parser.add_argument("--gpu-ids", nargs="+", default=["0"])
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--force-reevaluate", action="store_true")
    parser.add_argument("--skip-startup-debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--skip-final-aggregation",
        action="store_true",
        help="Do not rewrite shared aggregate files after this partitioned worker finishes.",
    )
    parser.add_argument(
        "--patience-pilot",
        action="store_true",
        help="Run the fixed P01/P03/P06 fold-1 x 5/10/25%% x transfer/scratch patience diagnostic.",
    )
    args = parser.parse_args()

    if args.patience_pilot:
        args.heldout_processes = [1, 3, 6]
        args.folds = [1]
        args.data_ratios = [0.05, 0.10, 0.25]
        args.modes = ["transfer", "scratch"]

    ratios = _validate_ratios(args.data_ratios)
    heldout_processes = [int(value) for value in args.heldout_processes]
    folds = [int(value) for value in args.folds]
    if any(value < 1 or value > 10 for value in heldout_processes):
        parser.error("--heldout-processes values must be in 1..10")
    if any(value < 1 or value > 5 for value in folds):
        parser.error("--folds values must be in 1..5")
    if "scratch" in args.modes and args.finetune_mode != "full":
        parser.error("scratch comparison requires --finetune-mode full")
    if int(args.early_stopping_patience) < 1:
        parser.error("--early-stopping-patience must be >= 1")
    if int(args.total_optimizer_steps) < 1:
        parser.error("--total-optimizer-steps must be >= 1")

    base_config = _resolve(args.base_config)
    split_root = _resolve(args.split_root)
    existing_unseen_root = _resolve(args.existing_unseen_root)
    output_root = _resolve(args.output_root)
    merged_csv = _resolve(args.merged_csv)
    for path, label in (
        (base_config, "base config"),
        (merged_csv, "merged CSV"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    output_root.mkdir(parents=True, exist_ok=True)

    experiment = load_experiment_config(base_config)
    excluded_graph_samples_raw = str(getattr(experiment.data, "excluded_graph_samples_path", "") or "").strip()
    excluded_graph_samples_path = _resolve(excluded_graph_samples_raw) if excluded_graph_samples_raw else None
    excluded_properties = {
        str(name).strip().casefold()
        for name in getattr(experiment.train, "target_mean_r2_excluded_properties", [])
        if str(name).strip()
    }
    specs, pool_frame = prepare_nested_subsets(
        split_root=split_root,
        output_root=output_root,
        heldout_processes=heldout_processes,
        folds=folds,
        ratios=ratios,
        seed=int(args.seed),
        force=bool(args.force_reevaluate),
        excluded_graph_samples_path=excluded_graph_samples_path,
    )
    process_stats = (
        pool_frame.groupby("process_id")["transfer_pool_size"].agg(["mean", "min", "max"]).reset_index()
    )
    print("[data-efficiency][pool-size-by-process]", flush=True)
    print(process_stats.to_string(index=False), flush=True)
    if args.dry_run or args.prepare_only:
        _print_design_audit(specs=specs, existing_unseen_root=existing_unseen_root)

    design = {
        "base_config": _rel(base_config),
        "split_root": _rel(split_root),
        "merged_csv": _rel(merged_csv),
        "existing_unseen_root": _rel(existing_unseen_root),
        "output_root": _rel(output_root),
        "heldout_processes": heldout_processes,
        "folds": folds,
        "data_ratios": ratios,
        "modes": list(args.modes),
        "seed": int(args.seed),
        "max_epochs": int(args.max_epochs),
        "total_optimizer_steps": int(args.total_optimizer_steps),
        "termination_condition": "max_epochs_or_optimizer_step_cap",
        "optimizer_step_policy": "upper_cap",
        "early_stopping_enabled": False,
        "checkpoint_selection": "best_validation_checkpoint_within_epoch_horizon",
        "nested_subsets": True,
        "transfer_scratch_share_manifest": True,
        "epoch_sample_size_policy": "min_subset_size_1000_without_replacement",
        "existing_pretrain_reused": True,
        "existing_zero_shot_reused": True,
        "existing_transfer_reused": False,
        "existing_transfer_reuse_reason": (
            "old runs sampled 2000 rows per epoch from the full transfer pool; "
            "they did not restrict total accessible unique target-process data"
        ),
        "validation_and_test_are_same_manifest": False,
        "preprocessing_fit_scope": "current_ratio_training_subset_only",
        "target_mean_r2_excluded_properties": sorted(excluded_properties),
        "maximum_new_training_runs": len(specs) * len(args.modes),
    }
    design_path = output_root / "experiment_design.json"
    existing_design = _load_json(design_path)
    if existing_design:
        stable_keys = (
            "base_config", "split_root", "merged_csv", "existing_unseen_root", "seed",
            "max_epochs", "total_optimizer_steps", "termination_condition",
        )
        conflicts = [
            key for key in stable_keys
            if key in existing_design and existing_design.get(key) != design.get(key)
        ]
        if conflicts and not args.force_reevaluate:
            raise RuntimeError(f"existing experiment design conflicts on keys: {conflicts}")
        for key in ("heldout_processes", "folds", "data_ratios", "modes"):
            design[key] = sorted(set(existing_design.get(key, [])) | set(design.get(key, [])))
        design["maximum_new_training_runs"] = (
            len(design["heldout_processes"])
            * len(design["folds"])
            * len(design["data_ratios"])
            * len(design["modes"])
        )
    _write_json(design_path, design)

    if args.aggregate_only:
        aggregate_results(
            output_root=output_root,
            existing_unseen_root=existing_unseen_root,
            heldout_processes=heldout_processes,
            folds=folds,
            ratios=ratios,
            excluded_properties=excluded_properties,
        )
        return
    if args.prepare_only:
        return

    specs_by_key = {(spec.heldout_process, spec.fold, spec.ratio_label): spec for spec in specs}
    tasks: list[full_unseen.Task] = []
    for heldout in heldout_processes:
        source_ids = full_unseen._source_ids(heldout)
        split_dir = split_root / f"heldout_P{heldout:02d}"
        pretrain_dir = existing_unseen_root / f"heldout_P{heldout:02d}" / "pretrain"
        checkpoint = full_unseen._latest_completed_checkpoint(pretrain_dir)
        if "transfer" in args.modes and checkpoint is None:
            raise FileNotFoundError(f"completed pretrain checkpoint not found for heldout P{heldout:02d}: {pretrain_dir}")
        for fold in folds:
            target_val = split_dir / f"fold_{fold:02d}" / "target_val.csv"
            target_test = split_dir / f"fold_{fold:02d}" / "target_test.csv"
            for ratio in ratios:
                spec = specs_by_key[(heldout, fold, _ratio_label(ratio))]
                # Keep the fixed accessible subset, but cap one epoch's exposure.
                epoch_sample_count = min(int(spec.sample_count), 1000)
                for mode in args.modes:
                    run_dir = output_root / f"heldout_P{heldout:02d}" / f"fold_{fold:02d}" / mode / spec.ratio_label
                    runtime_path = run_dir / "runtime_overrides.json"
                    full_unseen._write_runtime_overrides(
                        runtime_path,
                        seed=int(args.seed),
                        experiment_name=f"data_eff_P{heldout:02d}_F{fold:02d}_{mode}_{spec.ratio_label}",
                        output_dir=run_dir,
                        merged_csv=_rel(merged_csv),
                        process_ids=[heldout],
                        train_manifest=spec.subset_manifest,
                        val_manifest=target_val,
                        test_manifest=target_test,
                        max_epochs=int(args.max_epochs),
                        monitor_metric=str(args.monitor_metric),
                        monitor_mode=str(args.monitor_mode),
                        learning_rate=args.learning_rate,
                        batch_size=args.batch_size,
                        epoch_sampler={
                            "enabled": True,
                            "mode": "random",
                            "base_epoch_size": int(epoch_sample_count),
                            "unique_within_epoch": True,
                            "replacement_for_uniform": False,
                            "mass_flow_tail": {"enabled": False},
                            "hard_target_edges": {"enabled": False},
                        },
                        early_stopping_patience=int(args.early_stopping_patience),
                        final_target_edge_metric_splits=["val", "test"],
                        train_overrides={
                            "early_stopping": False,
                            "max_optimizer_steps": int(args.total_optimizer_steps),
                            "save_last_checkpoint": False,
                            "checkpoint_include_optimizer_state": False,
                            "pinn_weight_schedule": {
                                **dict(getattr(experiment.train, "pinn_weight_schedule", {}) or {}),
                                "fixed_step_reference_epochs": 30,
                            },
                        },
                    )
                    pretrained = checkpoint if mode == "transfer" else None
                    tasks.append(
                        full_unseen.Task(
                            name=f"P{heldout:02d}_F{fold:02d}_{mode}_{spec.ratio_label}",
                            kind="train",
                            run_dir=run_dir,
                            cmd=full_unseen._train_cmd(
                                base_config=base_config,
                                runtime_path=runtime_path,
                                max_epochs=int(args.max_epochs),
                                pretrained=pretrained,
                                finetune_mode="full",
                                skip_startup_debug=bool(args.skip_startup_debug),
                                no_training_plots=True,
                            ),
                            metadata={
                                "stage": "transfer_data_efficiency",
                                "heldout_process": int(heldout),
                                "source_process_ids": source_ids,
                                "fold": int(fold),
                                "mode": mode,
                                "sample_ratio": float(ratio),
                                "sample_count": int(spec.sample_count),
                                "transfer_pool_size": int(spec.transfer_pool_size),
                                "subset_manifest": _rel(spec.subset_manifest),
                                "subset_indices": _rel(spec.subset_indices_path),
                                "subset_hash": spec.subset_hash,
                                "permutation_seed": int(spec.permutation_seed),
                                "validation_manifest": _rel(target_val),
                                "test_manifest": _rel(target_test),
                                "validation_and_test_are_same_manifest": False,
                                "preprocessing_fit_manifest": _rel(spec.subset_manifest),
                                "scaler_fit_indices_sha256": spec.subset_hash,
                                "scaler_fit_is_current_training_subset": True,
                                "pretrained_checkpoint": _rel(checkpoint) if pretrained is not None else "",
                                "random_seed": int(args.seed),
                                "max_epochs": int(args.max_epochs),
                                "early_stopping_enabled": False,
                                "total_optimizer_steps": int(args.total_optimizer_steps),
                                "termination_condition": "max_epochs_or_optimizer_step_cap",
                                "optimizer_step_policy": "upper_cap",
                                "epoch_sample_size": int(epoch_sample_count),
                                "epoch_sampling": "random_without_replacement_from_fixed_subset",
                            },
                        )
                    )

    print(
        f"[data-efficiency] tasks={len(tasks)} expected_max={len(specs) * len(args.modes)} "
        f"pretrain_runs=0 zero_shot_runs=0",
        flush=True,
    )
    full_unseen._run_tasks(
        tasks,
        gpu_ids=[str(value) for value in args.gpu_ids],
        max_parallel=int(args.max_parallel),
        resume_existing=bool(args.resume_existing),
        force_reevaluate=bool(args.force_reevaluate),
        dry_run=bool(args.dry_run),
    )
    if not args.dry_run and not args.skip_final_aggregation:
        aggregate_results(
            output_root=output_root,
            existing_unseen_root=existing_unseen_root,
            heldout_processes=heldout_processes,
            folds=folds,
            ratios=ratios,
            excluded_properties=excluded_properties,
        )


if __name__ == "__main__":
    main()
