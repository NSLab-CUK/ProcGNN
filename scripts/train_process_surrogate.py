from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
import traceback
from functools import partial
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset
from tqdm import tqdm

if not hasattr(argparse, "BooleanOptionalAction"):
    class _BooleanOptionalAction(argparse.Action):
        def __init__(
            self,
            option_strings,
            dest,
            default=None,
            required=False,
            help=None,
            metavar=None,
        ):
            _option_strings = []
            for option_string in option_strings:
                _option_strings.append(option_string)
                if option_string.startswith("--"):
                    option_string = "--no-" + option_string[2:]
                    _option_strings.append(option_string)
            super().__init__(
                option_strings=_option_strings,
                dest=dest,
                nargs=0,
                default=default,
                required=required,
                help=help,
                metavar=metavar,
            )

        def __call__(self, parser, namespace, values, option_string=None):
            value = not (option_string or "").startswith("--no-")
            setattr(namespace, self.dest, value)

    argparse.BooleanOptionalAction = _BooleanOptionalAction

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _atomic_torch_save(payload: Any, destination: Path) -> None:
    """Write a torch artifact without leaving a truncated destination file."""
    destination = Path(destination)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

from process_graph.data.tabular_dataset import (  # noqa: E402
    PROCESS_FIXED_TARGET_COLUMNS,
    ProcessGraphTabularDataset,
    collate_graph_batch,
    compute_category_target_normalizer,
    compute_oper_normalizer,
    compute_oper_normalizer_from_x_oper,
    compute_target_normalizer,
    compute_y_edge_log1p_column_scaler,
    compute_y_edge_scaler,
)
from process_graph.data.stream_keys import (  # noqa: E402
    STREAM_KEY_CANONICALIZATION_VERSION,
    canonicalize_stream_key,
)
from process_graph.data.rare_positive_sampler import (  # noqa: E402
    compute_rare_positive_sample_weights,
)
from process_graph.data.epoch_balanced_sampler import EpochBalancedSampler  # noqa: E402
from process_graph.data.target_edge_augmentation import (  # noqa: E402
    build_balanced_target_edge_augmented_dataset,
    build_rare_target_edge_augmented_dataset,
)
from process_graph.constants import STREAM_EDGE_FEATURE_SLOTS  # noqa: E402
from process_graph.constants import OPER_FEATURE_SLOTS  # noqa: E402
from process_graph.known_feed import (  # noqa: E402
    known_feed_config_dict,
    operating_feature_names,
    parse_known_feed_condition,
)
from process_graph.parser import parse_process_file  # noqa: E402
from process_graph.resolver import build_graph_sample  # noqa: E402
from process_graph.experiment.edge_all_answer_columns import resolve_answer_target_column_index  # noqa: E402
from process_graph.experiment.edge_all_leakage import assert_no_edge_all_target_leakage  # noqa: E402
from process_graph.experiment.edge_all_reporting import (  # noqa: E402
    compute_target_metrics_v4,
    evaluate_edge_all_detailed,
    write_baseline_comparison,
    write_edge_all_artifacts,
    write_edge_all_combined_metrics_csvs,
    write_edge_all_eval_figures,
    write_edge_all_extra_diagnostics,
    write_edge_all_metrics_json_all_splits,
    write_edge_all_metrics_json_from_val_history,
)
from process_graph.debug_ndjson import (  # noqa: E402
    DEFAULT_NDJSON_LOG_PATH,
    get_session_ndjson_path,
    set_session_ndjson_path,
    write_training_debug_event,
)
from process_graph.experiment.config_builders import build_task_specs, model_yaml_to_encoder_config  # noqa: E402
from process_graph.experiment.target_stream_weighting import write_target_stream_feature_metric_artifacts  # noqa: E402
from process_graph.experiment.target_edge_10d_metrics import (  # noqa: E402
    collect_target_edge_10d_metric_rows_from_loader,
    write_target_edge_10d_metric_artifacts,
)
from process_graph.experiment.edge_step_training import train_one_batch_edge_step  # noqa: E402
from process_graph.experiment.edge_step_training import build_target_edge_boolean_mask  # noqa: E402
from process_graph.experiment.edge_step_pi_training import (  # noqa: E402
    _normalize_target_if_needed,
    collect_pi_all_edge_property_metrics_from_loader,
    evaluate_pi_epoch,
    get_pi_targets,
    resolve_mass_flow_physical_scale,
    train_one_batch_sample_hybrid_target_edge_step_pi,
    train_one_batch_edge_step_pi,
)
from process_graph.experiment.pi_mass_flow import (  # noqa: E402
    assert_checkpoint_pi_mass_flow_compatible,
    resolve_mass_flow_log_eps,
    resolve_mass_flow_log_scale,
    resolve_mass_flow_log_tau,
    resolve_mass_flow_transform,
    resolve_pi_mass_flow_output_space,
)
from process_graph.experiment.loaders import load_experiment_config  # noqa: E402
from process_graph.experiment.train_utils import (  # noqa: E402
    _edge_all_normalize,
    build_optimizer,
    build_scheduler,
    compute_training_loss,
    decoder_tasks_enabled,
    evaluate_edge_all_epoch,
    loss_metric_keys,
    masked_edge_mae,
    masked_edge_regression_loss,
    resolve_answer_edge_species_weights,
    validate_edge_all_batch,
)
from process_graph.experiment.optuna_warmstart import (  # noqa: E402
    discover_best_hyperparameters_json,
    flat_optuna_params_to_runtime_overrides,
    load_best_params_from_json,
)
from process_graph.experiment.yaml_utils import deep_merge  # noqa: E402
from process_graph.models import ProcessSurrogateModel  # noqa: E402
from process_graph.experiment.metric_policy import enrich_metrics_json  # noqa: E402
from process_graph.experiment.target_row_loss import (  # noqa: E402
    build_target_incident_diagnostic_rows,
    resolve_all_edge_r2_config,
)
from process_graph.experiment.edge_all_supervision_log import (  # noqa: E402
    edge_all_supervision_banner,
    format_epoch_detail_edge_all_line,
    format_epoch_summary_edge_all_parts,
    format_val_metric_edge_all_line,
)

# val_metrics may include provenance strings (semicolon-separated target ids).
_VAL_METRIC_NON_SCALAR_SUBSTRINGS = (
    "target_ids",
    "target_reasons",
    "schema_version",
    "primary_metric_name",
    "excluded_target",
)

_Y_EDGE_SCALER_CACHE_VERSION = 5


def _cuda_synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def _benchmark_inference_loader(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    train_cfg: Any,
    warmup_runs: int,
    measured_runs: int,
    max_batches: int = 0,
    use_amp: bool = False,
    target_edges_only: bool = False,
    verify_target_edge_values: bool = False,
) -> dict[str, Any]:
    """Measure loader + host-to-device + model-forward inference cost.

    ``target_edges_only`` retains full encoder message passing but restricts the
    hierarchical stream-property decoder to target edges.  This is an explicit
    deployment-time mode; it never changes the default all-predictable-edge
    model path.
    """

    warmup_runs = max(0, int(warmup_runs))
    measured_runs = max(1, int(measured_runs))
    max_batches = max(0, int(max_batches))
    was_training = model.training
    model.eval()
    prediction_shapes: dict[str, list[int]] = {}

    def _run_once() -> tuple[int, int, int, int]:
        sample_count = 0
        batch_count = 0
        decoded_edge_count = 0
        predictable_edge_count = 0
        for batch in loader:
            if max_batches and batch_count >= max_batches:
                break
            batch_data = {key: value.to(device) for key, value in batch.model_kwargs.items()}
            task_inputs = {
                head: {key: value.to(device) for key, value in payload.items()}
                for head, payload in batch.task_inputs.items()
            }
            edge_target = batch.targets.get("edge_stream")
            if edge_target is not None:
                n_edges = int(edge_target.shape[0])
                target_edge_mask, _ = build_target_edge_boolean_mask(
                    train_cfg=train_cfg,
                    edge_export_meta=batch.edge_export_meta,
                    edge_target_columns=list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
                    n_edges=n_edges,
                    device=device,
                )
                batch_data["target_edge_mask"] = target_edge_mask
                predictable_mask = batch_data.get("edge_is_predictable")
                if predictable_mask is not None:
                    predictable_mask = predictable_mask.to(
                        device=device, dtype=torch.bool
                    ).reshape(-1)
                    if predictable_mask.numel() != n_edges:
                        raise ValueError(
                            "edge_is_predictable must align with edge targets during inference benchmarking."
                        )
                else:
                    predictable_mask = torch.ones(
                        n_edges, device=device, dtype=torch.bool
                    )
                predictable_edge_count += int(predictable_mask.sum().item())
                if target_edges_only:
                    decoder_mask = predictable_mask & target_edge_mask.to(
                        device=device, dtype=torch.bool
                    ).reshape(-1)
                    batch_data["edge_prediction_mask"] = decoder_mask
                    decoded_edge_count += int(decoder_mask.sum().item())
                else:
                    decoded_edge_count += int(predictable_mask.sum().item())
            elif target_edges_only:
                raise RuntimeError(
                    "--inference-target-edges-only requires edge_stream targets to build the target-edge mask."
                )
            amp_device = "cuda" if device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=amp_device, enabled=bool(use_amp)):
                outputs = model(batch_data, task_inputs=task_inputs)
            if not prediction_shapes and isinstance(outputs, Mapping):
                prediction_shapes.update(
                    {
                        str(key): list(value.shape)
                        for key, value in outputs.items()
                        if isinstance(value, torch.Tensor)
                    }
                )
            sample_count += len(batch.sample_meta)
            batch_count += 1
        return sample_count, batch_count, decoded_edge_count, predictable_edge_count

    def _verify_target_only_values() -> dict[str, float | int]:
        """Show that selecting target decoder rows preserves target predictions."""

        for batch in loader:
            batch_data = {key: value.to(device) for key, value in batch.model_kwargs.items()}
            task_inputs = {
                head: {key: value.to(device) for key, value in payload.items()}
                for head, payload in batch.task_inputs.items()
            }
            edge_target = batch.targets.get("edge_stream")
            if edge_target is None:
                raise RuntimeError("target-edge-only verification requires edge_stream targets.")
            n_edges = int(edge_target.shape[0])
            target_edge_mask, _ = build_target_edge_boolean_mask(
                train_cfg=train_cfg,
                edge_export_meta=batch.edge_export_meta,
                edge_target_columns=list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
                n_edges=n_edges,
                device=device,
            )
            predictable_mask = batch_data.get("edge_is_predictable")
            if predictable_mask is None:
                predictable_mask = torch.ones(n_edges, device=device, dtype=torch.bool)
            else:
                predictable_mask = predictable_mask.to(device=device, dtype=torch.bool).reshape(-1)
            selected_mask = predictable_mask & target_edge_mask.to(device=device, dtype=torch.bool).reshape(-1)
            full_data = dict(batch_data)
            full_data["target_edge_mask"] = target_edge_mask
            selected_data = dict(full_data)
            selected_data["edge_prediction_mask"] = selected_mask
            amp_device = "cuda" if device.type == "cuda" else "cpu"
            with torch.amp.autocast(device_type=amp_device, enabled=bool(use_amp)):
                full_outputs = model(full_data, task_inputs=task_inputs)
                selected_outputs = model(selected_data, task_inputs=task_inputs)
            compared = 0
            max_abs_diff = 0.0
            max_abs_diff_by_output: dict[str, float] = {}
            max_abs_reference_by_output: dict[str, float] = {}
            max_relative_diff_by_output: dict[str, float] = {}
            for name, full_value in full_outputs.items():
                selected_value = selected_outputs.get(name)
                if (
                    not isinstance(full_value, torch.Tensor)
                    or not isinstance(selected_value, torch.Tensor)
                    or full_value.ndim < 1
                    or selected_value.shape != full_value.shape
                    or full_value.shape[0] != n_edges
                ):
                    continue
                if bool(selected_mask.any()):
                    difference = (full_value[selected_mask] - selected_value[selected_mask]).abs()
                    output_max_abs_diff = float(difference.max().item())
                    max_abs_diff = max(max_abs_diff, output_max_abs_diff)
                    max_abs_diff_by_output[str(name)] = output_max_abs_diff
                    reference = full_value[selected_mask].abs()
                    max_abs_reference_by_output[str(name)] = float(
                        reference.max().item()
                    )
                    relative_difference = difference / reference.clamp_min(1e-8)
                    max_relative_diff_by_output[str(name)] = float(
                        relative_difference.max().item()
                    )
                    compared += int(difference.numel())
            return {
                "target_edge_value_comparison_count": compared,
                "target_edge_value_max_abs_diff": max_abs_diff,
                "target_edge_value_max_abs_diff_by_output": max_abs_diff_by_output,
                "target_edge_value_max_abs_reference_by_output": max_abs_reference_by_output,
                "target_edge_value_max_relative_diff_by_output": max_relative_diff_by_output,
                "target_main_stream_max_abs_diff": max_abs_diff_by_output.get(
                    "main_stream_pred", float("nan")
                ),
                "target_edges_verified": int(selected_mask.sum().item()),
            }
        raise RuntimeError("target-edge-only verification received an empty loader.")

    for _ in range(warmup_runs):
        _run_once()
    _cuda_synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    durations: list[float] = []
    sample_count = 0
    batch_count = 0
    decoded_edge_count = 0
    predictable_edge_count = 0
    for _ in range(measured_runs):
        _cuda_synchronize(device)
        started = time.perf_counter()
        run_samples, run_batches, run_decoded_edges, run_predictable_edges = _run_once()
        _cuda_synchronize(device)
        durations.append(time.perf_counter() - started)
        sample_count = run_samples
        batch_count = run_batches
        decoded_edge_count = run_decoded_edges
        predictable_edge_count = run_predictable_edges

    mean_sec = sum(durations) / len(durations)
    variance = sum((value - mean_sec) ** 2 for value in durations) / len(durations)
    std_sec = math.sqrt(max(variance, 0.0))
    median_sec = float(statistics.median(durations))
    result = {
        "timing_scope": "data_loader_iteration_plus_host_to_device_plus_model_forward",
        "edge_decoder_scope": (
            "target_edges_only" if target_edges_only else "all_predictable_edges"
        ),
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "max_batches": max_batches,
        "inference_batch_size": int(getattr(loader, "batch_size", 0) or 0),
        "samples_per_run": sample_count,
        "batches_per_run": batch_count,
        "decoded_edges_per_run": decoded_edge_count,
        "predictable_edges_per_run": predictable_edge_count,
        "inference_total_sec_mean": mean_sec,
        "inference_total_sec_std": std_sec,
        "inference_time_per_sample_ms": (mean_sec * 1000.0 / sample_count) if sample_count else float("nan"),
        "inference_time_std_ms": (std_sec * 1000.0 / sample_count) if sample_count else float("nan"),
        "inference_time_median_ms": (
            median_sec * 1000.0 / sample_count if sample_count else float("nan")
        ),
        "throughput_samples_per_sec": (sample_count / mean_sec) if mean_sec > 0.0 else float("nan"),
        "inference_peak_gpu_allocated_mb": (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
            if device.type == "cuda"
            else 0.0
        ),
        "inference_peak_gpu_reserved_mb": (
            float(torch.cuda.max_memory_reserved(device)) / (1024.0**2)
            if device.type == "cuda"
            else 0.0
        ),
        "device": str(device),
        "prediction_shapes": prediction_shapes,
    }
    if target_edges_only and verify_target_edge_values:
        result.update(_verify_target_only_values())
    if was_training:
        model.train()
    return result


def _path_signature(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"path": str(path or ""), "missing": True}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _manifest_fit_provenance(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {"manifest_path": str(path or ""), "missing": True}
    frame = pd.read_csv(path, usecols=["merged_row_index"])
    indices = pd.to_numeric(frame["merged_row_index"], errors="raise").astype(int).tolist()
    digest = hashlib.sha256(
        "\n".join(str(index) for index in indices).encode("utf-8")
    ).hexdigest()
    return {
        "manifest_path": str(path.resolve()),
        "sample_count": int(len(indices)),
        "merged_row_indices_sha256": digest,
        "train_split_only": True,
    }


def _y_edge_scaler_cache_path(
    *,
    experiment: Any,
    train_csv: Path,
    train_manifest_path: Path | None,
    dataset_len: int,
    stream_target_dim: int,
    include_mass_flow_transform: bool | None = None,
    edge_target_columns: list[str] | None = None,
) -> Path:
    train_cfg = getattr(experiment, "train", None)
    if include_mass_flow_transform is None:
        include_mass_flow_transform = bool(
            getattr(train_cfg, "use_log1p_mass_flow_loss", False)
            or (train_cfg is not None and resolve_pi_mass_flow_output_space(train_cfg) == "log1p")
        )
    process_ids = [int(x) for x in (getattr(experiment.data, "edge_all_processes", None) or [])]
    stream_root = (experiment.project_root / experiment.data.stream_data_dir).resolve()
    reference_root = (
        experiment.project_root
        / getattr(experiment.data, "canonical_graph_spec_v3_dir", "data/reference/v3")
    ).resolve()
    payload = {
        "version": _Y_EDGE_SCALER_CACHE_VERSION if include_mass_flow_transform else 2,
        "stream_key_canonicalization_version": STREAM_KEY_CANONICALIZATION_VERSION,
        "train_csv": _path_signature(train_csv),
        "train_manifest": _path_signature(train_manifest_path),
        "dataset_len": int(dataset_len),
        "stream_target_dim": int(stream_target_dim),
        "process_ids": process_ids,
        "edge_target_columns": list(
            edge_target_columns
            if edge_target_columns is not None
            else (getattr(experiment.data, "edge_target_columns", None) or [])
        ),
        "stream_files": [
            _path_signature(stream_root / f"{process_id}.Process_Streams.csv")
            for process_id in process_ids
        ],
        "reference_files": [
            _path_signature(reference_root / name)
            for name in ("canonical_nodes.csv", "canonical_edges.csv", "target_answer_edges.csv")
        ],
        "excluded_graph_samples": _path_signature(
            (
                experiment.project_root
                / str(getattr(experiment.data, "excluded_graph_samples_path", "") or "")
            ).resolve()
            if str(getattr(experiment.data, "excluded_graph_samples_path", "") or "").strip()
            else None
        ),
    }
    if include_mass_flow_transform:
        payload["mass_flow_transform"] = resolve_mass_flow_transform(train_cfg)
        payload["mass_flow_log_tau"] = resolve_mass_flow_log_tau(train_cfg)
        payload["mass_flow_log_scale"] = resolve_mass_flow_log_scale(train_cfg)
        payload["mass_flow_log_eps"] = resolve_mass_flow_log_eps(train_cfg)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:20]
    return experiment.project_root / "outputs" / "cache" / "y_edge_scalers" / f"{digest}.pt"


def _oper_normalizer_cache_path(
    *,
    experiment: Any,
    train_csv: Path,
    train_manifest_path: Path | None,
    dataset_len: int,
    stream_target_dim: int,
) -> Path:
    del stream_target_dim
    model_cfg = getattr(experiment, "model", None)
    known_feed_cfg = parse_known_feed_condition(
        getattr(model_cfg, "known_feed_condition", None)
    )
    process_ids = [int(x) for x in (getattr(experiment.data, "edge_all_processes", None) or [])]
    reference_root = (
        experiment.project_root
        / getattr(experiment.data, "canonical_graph_spec_v3_dir", "data/reference/v3")
    ).resolve()
    payload = {
        "version": 2,
        "train_csv": _path_signature(train_csv),
        "train_manifest": _path_signature(train_manifest_path),
        "dataset_len": int(dataset_len),
        "process_ids": process_ids,
        "reference_files": [
            _path_signature(reference_root / name)
            for name in ("canonical_nodes.csv", "canonical_edges.csv")
        ],
        "excluded_graph_samples": _path_signature(
            (
                experiment.project_root
                / str(getattr(experiment.data, "excluded_graph_samples_path", "") or "")
            ).resolve()
            if str(getattr(experiment.data, "excluded_graph_samples_path", "") or "").strip()
            else None
        ),
        "known_feed_condition": known_feed_config_dict(known_feed_cfg),
        "operating_feature_names": list(operating_feature_names(known_feed_cfg)),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:20]
    return experiment.project_root / "outputs" / "cache" / "oper_normalizers" / f"{digest}.pt"


def _legacy_oper_normalizer_cache_path(
    *,
    experiment: Any,
    train_csv: Path,
    train_manifest_path: Path | None,
    dataset_len: int,
    stream_target_dim: int,
) -> Path:
    """Locate caches written before operating stats were decoupled from output shape."""
    y_edge_path = _y_edge_scaler_cache_path(
        experiment=experiment,
        train_csv=train_csv,
        train_manifest_path=train_manifest_path,
        dataset_len=dataset_len,
        stream_target_dim=stream_target_dim,
        include_mass_flow_transform=False,
    )
    known_feed_cfg = parse_known_feed_condition(
        getattr(getattr(experiment, "model", None), "known_feed_condition", None)
    )
    if known_feed_cfg.enabled:
        payload = {
            "base_cache_key": y_edge_path.name,
            "known_feed_condition": known_feed_config_dict(known_feed_cfg),
            "operating_feature_names": list(operating_feature_names(known_feed_cfg)),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()[:20]
        return (
            experiment.project_root
            / "outputs"
            / "cache"
            / "oper_normalizers"
            / f"{digest}.pt"
        )
    return experiment.project_root / "outputs" / "cache" / "oper_normalizers" / y_edge_path.name


def _try_float_metric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        fv = float(value)
        return fv if math.isfinite(fv) else None
    return None


def _coerce_val_metric_scalars(val_metrics: Mapping[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in val_metrics.items():
        ks = str(key)
        if any(sub in ks for sub in _VAL_METRIC_NON_SCALAR_SUBSTRINGS):
            continue
        fv = _try_float_metric(value)
        if fv is None and not isinstance(value, str):
            try:
                fv = float(value)
                if not math.isfinite(fv):
                    fv = None
            except (TypeError, ValueError):
                fv = None
        if fv is not None:
            out[ks] = fv
    return out


def _merge_edge_all_val_scalars_into_epoch_log(
    epoch_log: dict[str, Any], val_metrics: Mapping[str, Any]
) -> None:
    """Copy numeric edge_all eval keys into epoch_log; skip provenance strings."""
    numeric_prefixes = (
        "val_target_v4_",
        "val_process_balanced_",
        "val_target_balanced_",
        "val_legacy_",
        "val_target_stream_",
        "val_target_edge_10d_",
        "val_target_edge_r2_",
        "val_target_edge_property_mean_r2",
        "val_target_r2",
        "val_target_mean_r2",
        "val_n_",
        "val/eval_",
        "eval_",
    )
    for vk, vv in val_metrics.items():
        vks = str(vk)
        if any(sub in vks for sub in _VAL_METRIC_NON_SCALAR_SUBSTRINGS):
            continue
        if not (
            vks.startswith(numeric_prefixes)
            or vks.startswith("val/target_v4_")
            or vks.startswith("val/target_stream_")
            or vks.startswith("val/target_edge_10d_")
            or vks.startswith("val/target_edge_r2_")
            or vks.startswith("pi_all_edge_")
            or vks.startswith("target_edge_r2_")
            or vks.startswith("target_edge_property_mean_r2")
            or vks.startswith("val/target_r2")
            or vks.startswith("val/target_mean_r2")
            or vks.startswith("val_eval_")
        ):
            continue
        fv = _try_float_metric(vv)
        if fv is None:
            continue
        clean = vks.replace("val_", "", 1) if vks.startswith("val_") else vks
        if clean.startswith("val/"):
            clean = clean.replace("val/", "", 1)
        epoch_log[f"val/{clean}"] = fv
        epoch_log[f"val_{clean.replace('/', '_')}"] = fv
        epoch_log[vk] = fv


def _copy_edge_all_val_train_loss_scalars(
    epoch_log: dict[str, Any], val_metrics: Mapping[str, Any]
) -> None:
    """Copy validation loss/std diagnostics used by terminal epoch summaries."""
    keys = (
        "loss_edge_all",
        "loss_primary_frac",
        "loss_primary_frac_r2",
        "loss_target_frac_feature",
        "loss_target_row_frac",
        "loss_target_row_amount",
        "primary_frac_count",
        "loss_node_mass",
        "loss_node_component",
        "loss_node_atom",
        "node_mass_valid_count",
        "node_component_valid_count",
        "node_atom_valid_count",
        "node_mass_residual_mean",
        "node_component_residual_mean",
        "node_atom_residual_mean",
        "node_mass_residual_max",
        "node_component_residual_max",
        "node_atom_residual_max",
        "primary_frac_r2_valid_feature_count",
        "primary_frac_pred_std_norm",
        "primary_frac_true_std_norm",
        "primary_frac_std_ratio_norm",
        "primary_frac_pred_std_orig",
        "primary_frac_true_std_orig",
        "primary_frac_std_ratio_orig",
        "edge_all_mae",
        "edge_all_mse",
        "metric_target_r2",
        "metric_tailgas_r2",
        "metric_answer_all_targets_mean_r2",
        "target_h2_mae",
        "target_h2_rmse",
        "target_h2_r2",
        "tailgas_co2_mae",
        "tailgas_co2_rmse",
        "tailgas_co2_r2",
    )
    for key in keys:
        if key not in val_metrics:
            continue
        fv = _try_float_metric(val_metrics[key])
        if fv is None:
            continue
        epoch_log[f"val/{key}"] = fv
        epoch_log[f"val_{key}"] = fv


def _is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_ddp() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_ddp() else 1


def _is_main() -> bool:
    return _rank() == 0


def _ensure_output_dir(output_dir: Path) -> None:
    """Re-create run output dir before writes (network mounts may drop empty dirs)."""
    if _is_main():
        output_dir.mkdir(parents=True, exist_ok=True)


def _terminal_log_verbosity(train_cfg: Any) -> str:
    value = str(getattr(train_cfg, "terminal_log_verbosity", "compact") or "compact").strip().lower()
    if value in {"debug", "full"}:
        return "verbose"
    if value in {"silent", "none"}:
        return "quiet"
    if value not in {"quiet", "compact", "verbose"}:
        return "compact"
    return value


def _terminal_log_is_verbose(train_cfg: Any) -> bool:
    return _terminal_log_verbosity(train_cfg) == "verbose"


def _terminal_log_is_quiet(train_cfg: Any) -> bool:
    return _terminal_log_verbosity(train_cfg) == "quiet"


def _format_terminal_duration(seconds: float | int | None) -> str:
    try:
        total = int(max(0.0, float(seconds or 0.0)))
    except (TypeError, ValueError):
        return "?"
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _terminal_progress_bar(current: int, total: int | str, *, width: int = 20) -> tuple[str, str]:
    try:
        total_int = int(total)
    except (TypeError, ValueError):
        return "[" + ("-" * width) + "]", "?"
    total_int = max(total_int, 1)
    current_int = min(max(int(current), 0), total_int)
    ratio = current_int / total_int
    filled = int(round(ratio * width))
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]", f"{ratio * 100.0:.1f}%"


def _terminal_eta(start_time: float, current: int, total: int | str) -> tuple[str, str]:
    elapsed = max(0.0, time.perf_counter() - float(start_time))
    try:
        total_int = int(total)
        current_int = max(1, int(current))
    except (TypeError, ValueError):
        return _format_terminal_duration(elapsed), "?"
    if current_int <= 0 or total_int <= 0:
        return _format_terminal_duration(elapsed), "?"
    rate = elapsed / current_int
    remaining = max(0.0, rate * max(total_int - current_int, 0))
    return _format_terminal_duration(elapsed), _format_terminal_duration(remaining)


def _representation_stats(tensor: torch.Tensor) -> dict[str, float | int]:
    """Compact latent diagnostics for dimension-ablation forward audits."""
    x = tensor.detach().float()
    if x.ndim != 2 or x.numel() == 0:
        return {"rows": int(x.shape[0]) if x.ndim else 0, "dim": 0}
    feature_std = x.std(dim=0, unbiased=False)
    row_norm = x.norm(dim=-1)
    centered = x - x.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    singular_sum = singular.sum()
    if float(singular_sum.item()) > 0.0:
        probabilities = singular / singular_sum
        effective_rank = torch.exp(
            -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
        )
        effective_rank_value = float(effective_rank.cpu().item())
    else:
        effective_rank_value = 0.0
    return {
        "rows": int(x.shape[0]),
        "dim": int(x.shape[1]),
        "mean": float(x.mean().cpu().item()),
        "std": float(x.std(unbiased=False).cpu().item()),
        "norm_mean": float(row_norm.mean().cpu().item()),
        "norm_p95": float(torch.quantile(row_norm, 0.95).cpu().item()),
        "feature_std_mean": float(feature_std.mean().cpu().item()),
        "dead_dimension_ratio": float((feature_std <= 1.0e-8).float().mean().cpu().item()),
        "near_zero_variance_ratio": float(
            (feature_std <= 1.0e-4).float().mean().cpu().item()
        ),
        "effective_rank": effective_rank_value,
    }


def _print_edge_head_summary(model: ProcessSurrogateModel, *, train_cfg: Any | None = None) -> None:
    edge_decoder = getattr(model, "edge_decoder", None)
    if edge_decoder is None or not hasattr(edge_decoder, "describe_head"):
        return
    summary = edge_decoder.describe_head()
    verbosity = _terminal_log_verbosity(train_cfg) if train_cfg is not None else "compact"
    if verbosity == "quiet":
        return
    if summary.get("hierarchical_reduced_pi"):
        count = lambda module: sum(  # noqa: E731
            p.numel() for p in module.parameters()
        ) if module is not None else 0
        encoder = model.encoder
        head = edge_decoder.hierarchical_pi_head
        print(
            "[model][dimension-audit] "
            f"node_hidden={encoder.config.hidden_dim} "
            f"edge_hidden={encoder.config.edge_hidden_dim or encoder.config.hidden_dim} "
            f"gnn={encoder.config.flow_gnn_architecture} "
            f"differential={encoder.config.diff_mode} "
            f"set2set={summary.get('hierarchical_global_dim')} "
            f"descriptor={summary.get('hierarchical_descriptor_dim')} "
            f"decoder={summary.get('hierarchical_decoder_intermediate_dim')}"
            f"->{summary.get('hierarchical_shared_dim')} "
            f"branches=({summary.get('hierarchical_condition_hidden_dim')}/"
            f"{summary.get('hierarchical_fraction_hidden_dim')}/"
            f"{summary.get('hierarchical_mass_hidden_dim')}) "
            f"level2={summary.get('hierarchical_level2_hidden_dim')} "
            f"params(total={count(model)},"
            f"node={count(encoder.input_encoder)},"
            f"gnn={count(encoder.layers)},"
            f"edge={count(encoder.edge_input_encoder)},"
            f"pool={count(encoder.set2set_pool or encoder.attention_pool)},"
            f"global_proj={count(edge_decoder.hierarchical_global_projection)},"
            f"shared={count(head.shared_input) + count(head.shared_update) + count(head.shared_output_norm)},"
            f"condition={count(head.condition_head)},"
            f"fraction={count(head.fraction_head)},"
            f"mass={count(head.mass_head)},"
            f"volume={count(head.volume_head)})",
            flush=True,
        )
    if verbosity != "verbose":
        features = summary.get("stream_feature_order") or []
        frac_features = summary.get("grouped_frac_order") or []
        flow_features = summary.get("grouped_flow_order") or []
        cond_features = summary.get("grouped_cond_order") or []
        print(
            "[model][edge_head] "
            f"type={summary.get('edge_head_type')} "
            f"output_dim={summary.get('final_output_dim')} "
            f"params={summary.get('total_edge_head_parameters')} "
            f"features={features} "
            f"groups(frac={frac_features}, flow={flow_features}, cond={cond_features})",
            flush=True,
        )
        return
    print("[model][edge_head]", flush=True)
    for key in (
        "edge_head_type",
        "stream_target_dim",
        "grouped_property",
        "frac_head_output_dim",
        "flow_head_output_dim",
        "cond_head_output_dim",
        "final_output_dim",
        "stream_feature_order",
        "grouped_frac_order",
        "grouped_frac_indices",
        "grouped_flow_order",
        "grouped_flow_indices",
        "grouped_cond_order",
        "grouped_cond_indices",
        "single_head_parameters",
        "grouped_input_proj_parameters",
        "shared_proj_parameters",
        "frac_head_parameters",
        "flow_head_parameters",
        "cond_head_parameters",
        "total_edge_head_parameters",
        "hierarchical_descriptor_dim",
        "hierarchical_global_dim",
        "hierarchical_edge_embedding_dim",
        "hierarchical_decoder_intermediate_dim",
        "hierarchical_shared_dim",
        "hierarchical_condition_hidden_dim",
        "hierarchical_fraction_hidden_dim",
        "hierarchical_mass_hidden_dim",
        "hierarchical_level2_hidden_dim",
        "hierarchical_shared_residual",
    ):
        print(f"  {key}: {summary.get(key)}", flush=True)


def setup_ddp() -> int:
    """Initialise DDP from torchrun environment variables. Returns local rank."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp() -> None:
    if _is_ddp():
        dist.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed + _rank())
    torch.manual_seed(seed + _rank())


def _collate_for_experiment(data_cfg: Any):
    """Batch collation: target_only omits tailgas tensors entirely."""
    if getattr(data_cfg, "task_mode", "multitask") == "target_only":
        return partial(collate_graph_batch, fixed_slots=("target",))
    return collate_graph_batch


def _data_loader_worker_kwargs(train_cfg: Any) -> dict[str, Any]:
    """Keep worker processes alive instead of respawning them every epoch."""
    num_workers = int(getattr(train_cfg, "num_workers", 0) or 0)
    if num_workers <= 0:
        return {}
    return {
        "persistent_workers": bool(getattr(train_cfg, "persistent_workers", True)),
        "prefetch_factor": max(
            1,
            int(getattr(train_cfg, "prefetch_factor", 2) or 2),
        ),
    }


def _evaluation_batch_size(train_cfg: Any) -> int:
    """Use a larger eval batch without changing batch-size-one training."""
    configured = int(getattr(train_cfg, "eval_batch_size", 0) or 0)
    return configured if configured > 0 else int(train_cfg.batch_size)


def _sample_validation_loader(
    dataset: Any,
    *,
    train_cfg: Any,
    data_cfg: Any,
    epoch: int,
    use_ddp: bool,
) -> tuple[DataLoader, DistributedSampler | None, int | None]:
    """Build an epoch-specific random validation subset loader when requested."""
    sample_size = int(getattr(train_cfg, "val_random_sample_size", 0) or 0)
    if sample_size <= 0 or sample_size >= len(dataset):
        sampler = DistributedSampler(dataset, shuffle=False) if use_ddp else None
        loader = DataLoader(
            dataset,
            batch_size=_evaluation_batch_size(train_cfg),
            shuffle=False,
            sampler=sampler,
            num_workers=train_cfg.num_workers,
            pin_memory=train_cfg.pin_memory,
            collate_fn=_collate_for_experiment(data_cfg),
            **_data_loader_worker_kwargs(train_cfg),
        )
        return loader, sampler, None

    seed = int(getattr(train_cfg, "val_random_sample_seed", 42))
    if bool(getattr(train_cfg, "val_random_sample_each_epoch", True)):
        seed += int(epoch)
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), sample_size)
    subset = Subset(dataset, indices)
    sampler = DistributedSampler(subset, shuffle=False) if use_ddp else None
    loader = DataLoader(
        subset,
        batch_size=_evaluation_batch_size(train_cfg),
        shuffle=False,
        sampler=sampler,
        num_workers=train_cfg.num_workers,
        pin_memory=train_cfg.pin_memory,
        collate_fn=_collate_for_experiment(data_cfg),
        **_data_loader_worker_kwargs(train_cfg),
    )
    return loader, sampler, int(sample_size)


def _normalize_stream_key(value: Any) -> str:
    return canonicalize_stream_key(value)


def _edge_all_loss_kwargs(
    batch,
    data_cfg,
    *,
    y_edge_mean: torch.Tensor | None = None,
    y_edge_std: torch.Tensor | None = None,
) -> dict[str, Any]:
    if getattr(data_cfg, "task_mode", "multitask") != "edge_all":
        return {}
    out: dict[str, Any] = {
        "edge_export_meta": batch.edge_export_meta,
        "edge_target_columns": list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
    }
    edge_index = batch.model_kwargs.get("edge_index")
    if edge_index is not None:
        out["edge_index"] = edge_index
    if y_edge_mean is not None:
        out["y_edge_mean"] = y_edge_mean
    if y_edge_std is not None:
        out["y_edge_std"] = y_edge_std
    return out


def _apply_v4_target_loss_weights(
    batch,
    target_masks: dict[str, torch.Tensor],
    train_cfg: Any,
    device: torch.device,
) -> None:
    """Legacy element-level edge_stream_loss_weight (only if use_legacy_answer_weighting)."""
    if bool(getattr(train_cfg, "use_target_stream_loss_weighting", False)):
        return
    if not bool(getattr(train_cfg, "use_legacy_answer_weighting", False)):
        return
    if "edge_stream" not in target_masks or batch.edge_export_meta is None:
        return
    from process_graph.experiment.v4_target_edge_weighting import build_v4_target_loss_weight_tensor

    cols = list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
    w_cpu, _ = build_v4_target_loss_weight_tensor(
        export_meta=batch.edge_export_meta,
        edge_target_columns=cols,
        train_cfg=train_cfg,
    )
    if w_cpu is not None:
        target_masks["edge_stream_loss_weight"] = w_cpu.to(
            device=device, dtype=target_masks["edge_stream"].dtype
        )


def pick_device(name: str, local_rank: int = 0) -> torch.device:
    if name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda", local_rank)
        if _is_main():
            print("[WARN] CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device '{name}'. Use 'cuda' or 'cpu'.")


def resolve_split_filter(csv_path: Path, split_column: str, wanted: str) -> str | None:
    frame = pd.read_csv(csv_path, nrows=1)
    if split_column in frame.columns:
        return wanted
    return None


def _jsonable_config(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _jsonable_config(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable_config(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "__dataclass_fields__"):
        return _jsonable_config(asdict(obj))
    return str(obj)


def _safe_read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _hpo_pick_metric(metrics: Mapping[str, Any], *names: str) -> float:
    for name in names:
        for key in (name, f"test/{name}", f"val/{name}"):
            if key in metrics:
                value = _safe_float(metrics[key])
                if math.isfinite(value):
                    return value
    return float("nan")


def _hpo_read_last_epoch(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return {}
    if df.empty:
        return {}
    row = df.iloc[-1].to_dict()
    return {str(k): _jsonable_config(v) for k, v in row.items()}


def _pinn_weight_schedule_scale(
    train_cfg: Any,
    epoch_index: int,
    optimizer_step_index: int | None = None,
) -> float:
    schedule = getattr(train_cfg, "pinn_weight_schedule", None) or {}
    if not isinstance(schedule, Mapping) or not bool(schedule.get("enabled", False)):
        return 1.0
    max_optimizer_steps = int(getattr(train_cfg, "max_optimizer_steps", 0) or 0)
    if max_optimizer_steps > 0 and optimizer_step_index is not None:
        reference_epochs = max(1, int(schedule.get("fixed_step_reference_epochs", 30) or 30))
        progress = min(max(float(optimizer_step_index) / float(max_optimizer_steps), 0.0), 1.0)
        epoch_index = min(reference_epochs - 1, int(progress * reference_epochs))
    stages = schedule.get("stages")
    if isinstance(stages, list) and stages:
        user_epoch = int(epoch_index) + 1
        for idx, raw_stage in enumerate(stages):
            if not isinstance(raw_stage, Mapping):
                raise TypeError(f"train.pinn_weight_schedule.stages[{idx}] must be a mapping.")
            start_epoch = int(raw_stage.get("start_epoch", 1) or 1)
            end_epoch_raw = raw_stage.get("end_epoch")
            end_epoch = None if end_epoch_raw is None else int(end_epoch_raw)
            multiplier = float(raw_stage.get("multiplier", raw_stage.get("scale", 1.0)))
            if not math.isfinite(multiplier) or multiplier < 0.0:
                raise ValueError(
                    f"train.pinn_weight_schedule.stages[{idx}].multiplier must be finite and non-negative."
                )
            if user_epoch >= start_epoch and (end_epoch is None or user_epoch <= end_epoch):
                return multiplier
        return 1.0
    mode = str(schedule.get("mode", "linear_ramp")).strip().lower()
    start = float(schedule.get("start_scale", 0.0))
    end = float(schedule.get("end_scale", 1.0))
    if mode in {"constant", "none"}:
        return end
    if mode not in {"linear", "linear_ramp", "ramp"}:
        raise ValueError(f"Unsupported train.pinn_weight_schedule.mode={mode!r}.")
    warmup_epochs = max(1, int(schedule.get("warmup_epochs", schedule.get("epochs", 5)) or 5))
    if warmup_epochs <= 1:
        progress = 1.0
    else:
        progress = min(max(float(epoch_index) / float(warmup_epochs - 1), 0.0), 1.0)
    return start + (end - start) * progress


def _scaled_pinn_train_config(train_cfg: Any, scale: float) -> Any:
    if abs(float(scale) - 1.0) < 1.0e-12:
        return train_cfg
    out = copy.deepcopy(train_cfg)
    for name in (
        "lambda_node_mass",
        "lambda_node_component",
        "lambda_node_atom",
        "lambda_node_energy",
    ):
        if hasattr(out, name):
            setattr(out, name, float(getattr(out, name) or 0.0) * float(scale))
    node_pi = getattr(out, "node_balance_pi", None)
    if isinstance(node_pi, Mapping):
        node_pi = copy.deepcopy(dict(node_pi))
        for block_name in ("mass", "component", "atom", "energy"):
            block = node_pi.get(block_name)
            if isinstance(block, Mapping) and "weight" in block:
                block = dict(block)
                block["weight"] = float(block.get("weight") or 0.0) * float(scale)
                node_pi[block_name] = block
        out.node_balance_pi = node_pi
    return out


def _restore_mass_flow_physical_scale_from_checkpoint(
    payload: Mapping[str, Any],
    *,
    train_cfg: Any,
    checkpoint_path: Path,
) -> None:
    """Use the run-level physical scale persisted by the training checkpoint."""
    cfg = getattr(train_cfg, "mass_flow_physical_auxiliary", None)
    if not bool(getattr(cfg, "enabled", False)):
        return
    metadata = payload.get("mass_flow_physical_scale")
    if not isinstance(metadata, Mapping):
        return
    scale = float(metadata.get("scale", math.nan))
    minimum_scale = float(getattr(cfg, "minimum_scale", 1.0e-8))
    if not math.isfinite(scale) or scale < minimum_scale:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} contains invalid Mass_Flow physical scale {scale!r}."
        )
    cfg.resolved_scale = scale


def _hpo_read_focus_property_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return []
    if df.empty or "property_name" not in df.columns:
        return []
    focus = {"Mass_Flow", "Mole_Flow", "Vol_Flow", "Frac_H2", "Frac_CO2", "Frac_H2O"}
    sub = df[df["property_name"].astype(str).isin(focus)].copy()
    return [{str(k): _jsonable_config(v) for k, v in row.items()} for row in sub.to_dict(orient="records")]


def _write_hpo_run_context(
    output_dir: Path,
    *,
    experiment: Any,
    experiment_path: Path,
    best_metric: float | None,
    best_epoch: int,
    final_epoch: int,
    best_val_loss: float | None,
    final_val_loss: float | None,
    eval_checkpoint_path: Path | None,
) -> None:
    """Persist compact run context for later manual/LLM-guided hyperparameter search."""
    metrics = _safe_read_json(output_dir / "metrics.json")
    loss_summary = _safe_read_json(output_dir / "loss_contribution_summary.json")
    last_epoch = _hpo_read_last_epoch(output_dir / "metrics_per_epoch.csv")
    focus_props = _hpo_read_focus_property_metrics(output_dir / "metrics_by_property.csv")
    if not focus_props:
        focus_props = _hpo_read_focus_property_metrics(output_dir / "test" / "metrics_by_property.csv")

    model_cfg = experiment.model
    train_cfg = experiment.train
    data_cfg = experiment.data
    process_ids = getattr(data_cfg, "edge_all_processes", None) or []
    objective_candidates = {
        "process_balanced_main_target_frac_r2_main_verified": _hpo_pick_metric(
            metrics,
            "process_balanced_main_target_frac_r2_main_verified",
            "eval_primary_frac_r2_by_process",
        ),
        "target_balanced_main_target_frac_r2_main_verified": _hpo_pick_metric(
            metrics,
            "target_balanced_main_target_frac_r2_main_verified",
            "eval_primary_frac_r2_by_target",
        ),
        "legacy_answer_fraction_macro_r2": _hpo_pick_metric(
            metrics,
            "legacy_answer_fraction_macro_r2",
            "metric_answer_all_targets_mean_r2",
        ),
        "edge_all_r2_orig": _hpo_pick_metric(metrics, "edge_all_r2_orig"),
        "edge_all_mae_orig": _hpo_pick_metric(metrics, "edge_all_mae_orig"),
    }
    payload: dict[str, Any] = {
        "purpose": "Run-level context for future hyperparameter/weight optimization.",
        "run_dir": str(output_dir),
        "experiment_yaml": str(experiment_path),
        "process_ids": [int(x) for x in process_ids if str(x).strip()],
        "seed": int(getattr(experiment, "seed", 0)),
        "device": str(getattr(experiment, "device", "")),
        "best_metric_value": float(best_metric) if best_metric is not None else float("nan"),
        "best_epoch": int(best_epoch),
        "final_epoch": int(final_epoch),
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else float("nan"),
        "final_val_loss": float(final_val_loss) if final_val_loss is not None else float("nan"),
        "evaluation_checkpoint_path": str(eval_checkpoint_path) if eval_checkpoint_path is not None else "",
        "objective_candidates": objective_candidates,
        "hparams": {
            "learning_rate": float(getattr(train_cfg, "learning_rate", 0.0)),
            "batch_size": int(getattr(train_cfg, "batch_size", 0)),
            "gradient_clip_norm": float(getattr(train_cfg, "gradient_clip_norm", 0.0)),
            "weight_decay": float(getattr(train_cfg, "weight_decay", 0.0)),
            "scheduler": str(getattr(train_cfg, "scheduler", "")),
            "edge_head_type": str(getattr(model_cfg, "edge_head_type", "single")),
            "edge_decoder_hidden_dim": int(getattr(model_cfg, "edge_decoder_hidden_dim", 0)),
            "edge_head_dropout": float(getattr(model_cfg, "edge_head_dropout", 0.0)),
            "edge_head_shared_dims": list(getattr(model_cfg, "edge_head_shared_dims", []) or []),
            "edge_head_frac_dims": list(getattr(model_cfg, "edge_head_frac_dims", []) or []),
            "edge_head_flow_dims": list(getattr(model_cfg, "edge_head_flow_dims", []) or []),
            "edge_head_cond_dims": list(getattr(model_cfg, "edge_head_cond_dims", []) or []),
            "edge_oper_dim": int(getattr(model_cfg, "edge_oper_dim", 0)),
            "edge_struct_dim": int(getattr(model_cfg, "edge_struct_dim", 0)),
            "edge_struct_include_join_flags": bool(getattr(data_cfg, "edge_struct_include_join_flags", True)),
            "primary_frac_loss_weight": float(getattr(train_cfg, "primary_frac_loss_weight", 0.0)),
            "primary_frac_r2_loss_weight": float(getattr(train_cfg, "primary_frac_r2_loss_weight", 0.0)),
            "primary_frac_r2_min_count": int(getattr(train_cfg, "primary_frac_r2_min_count", 0)),
            "primary_frac_r2_sst_threshold": float(getattr(train_cfg, "primary_frac_r2_sst_threshold", 0.0)),
            "primary_frac_r2_loss_cap": float(getattr(train_cfg, "primary_frac_r2_loss_cap", 0.0)),
            "answer_edge_weight_h2": float(getattr(train_cfg, "answer_edge_weight_h2", 0.0)),
            "answer_edge_weight_co2": float(getattr(train_cfg, "answer_edge_weight_co2", 0.0)),
            "answer_edge_weight_h2o": float(getattr(train_cfg, "answer_edge_weight_h2o", 0.0)),
            "target_feature_loss_balancing": str(getattr(train_cfg, "target_feature_loss_balancing", "")),
        },
        "loss_contribution_summary": loss_summary,
        "last_epoch_metrics": last_epoch,
        "focus_property_metrics": focus_props,
        "metric_files": {
            "metrics_per_epoch": str(output_dir / "metrics_per_epoch.csv"),
            "metrics_json": str(output_dir / "metrics.json"),
            "target_metrics_v4": str(output_dir / "target_metrics_v4.csv"),
            "target_metrics_v4_frac": str(output_dir / "target_metrics_v4_frac.csv"),
            "metrics_by_property": str(output_dir / "metrics_by_property.csv"),
            "loss_contribution_by_epoch": str(output_dir / "loss_contribution_by_epoch.csv"),
        },
    }
    out_json = output_dir / "hpo_run_context.json"
    out_csv = output_dir / "hpo_metric_snapshot.csv"
    out_json.write_text(json.dumps(_jsonable_config(payload), indent=2), encoding="utf-8")
    flat_row = {
        "run_dir": str(output_dir),
        "experiment_yaml": str(experiment_path),
        "process_ids": ",".join(str(x) for x in payload["process_ids"]),
        "best_metric_value": payload["best_metric_value"],
        "best_epoch": payload["best_epoch"],
        "final_epoch": payload["final_epoch"],
        **{f"hparam_{k}": _jsonable_config(v) for k, v in payload["hparams"].items()},
        **{f"objective_{k}": v for k, v in objective_candidates.items()},
    }
    pd.DataFrame([flat_row]).to_csv(out_csv, index=False)
    print(f"[hpo] wrote context: {out_json}")
    print(f"[hpo] wrote snapshot: {out_csv}")


def _merge_overrides(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    deep_merge(merged, dict(override))
    return merged


def _parse_override_value(text: str) -> Any:
    value = str(text)
    low = value.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        return json.loads(value)
    except Exception:
        return value


def _set_nested_override(payload: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = [p for p in str(dotted_key).split(".") if p]
    if not parts:
        raise ValueError("empty override key")
    cur: dict[str, Any] = payload
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _parse_cli_overrides(items: list[str] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--override must be KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        if key == "train.max_epochs":
            key = "train.epochs"
        _set_nested_override(payload, key, _parse_override_value(raw))
    return payload


def _normalize_runtime_override_aliases(payload: Mapping[str, Any]) -> dict[str, Any]:
    out = _merge_overrides({}, payload)
    loss = out.pop("loss", None)
    if isinstance(loss, Mapping):
        train = out.setdefault("train", {})
        edge_weight = loss.get("edge_weight")
        if isinstance(edge_weight, Mapping):
            if "default" in edge_weight:
                train["edge_weight_default"] = edge_weight["default"]
            if "target_edge" in edge_weight:
                train["edge_weight_target_edge"] = edge_weight["target_edge"]
        if "use_all_edge_r2_loss_in_edge_step" in loss:
            train["use_all_edge_r2_loss_in_edge_step"] = loss["use_all_edge_r2_loss_in_edge_step"]
        for key in (
            "use_total_loss_in_edge_step_pi",
            "use_all_edge_r2_loss_in_edge_step_pi",
            "use_primary_frac_loss_in_edge_step_pi",
            "use_pinn_loss",
            "lambda_main",
            "lambda_rho",
            "lambda_h",
            "lambda_volume",
            "lambda_enthalpy_flow",
            "lambda_atom",
            "lambda_energy",
            "compute_node_balance_diagnostics",
            "use_node_mass_balance_loss",
            "lambda_node_mass",
            "node_mass_balance_relative",
            "use_node_component_balance_loss",
            "lambda_node_component",
            "node_component_balance_relative",
            "node_balance_pi",
            "use_node_atom_balance_loss",
            "lambda_node_atom",
            "node_atom_balance_relative",
            "use_node_energy_balance_loss",
            "lambda_node_energy",
            "node_energy_balance_relative",
            "use_rho_loss",
            "use_h_loss",
            "use_volume_loss",
            "use_enthalpy_flow_loss",
            "use_atom_balance_loss",
            "use_energy_balance_loss",
            "pinn_loss_reduction",
            "eps",
            "mw_unit_scale",
            "volume_unit_scale",
            "volume_loss_normalized",
        ):
            if key in loss:
                train[key] = loss[key]
    physics = out.pop("physics", None)
    if isinstance(physics, Mapping):
        train = out.setdefault("train", {})
        for key in ("h_basis", "molecular_weight_unit"):
            if key in physics:
                train[key] = physics[key]
    return out


def _write_edge_step_debug_rows(output_dir: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    _ensure_output_dir(output_dir)
    path = output_dir / "edge_step_first_batch_debug.csv"
    pd.DataFrame([dict(r) for r in rows]).to_csv(path, index=False)
    print(f"[edge_step][debug] wrote {path}", flush=True)


def _write_edge_step_pi_debug_rows(output_dir: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    _ensure_output_dir(output_dir)
    path = output_dir / "edge_step_pi_first_batch_debug.csv"
    pd.DataFrame([dict(r) for r in rows]).to_csv(path, index=False)
    print(f"[edge_step_pi][debug] wrote {path}", flush=True)


def _write_edge_step_pi_batch_summary(
    output_dir: Path,
    loss_items: Mapping[str, Any],
) -> None:
    _ensure_output_dir(output_dir)
    payload: dict[str, Any] = {}
    for key, value in loss_items.items():
        if torch.is_tensor(value) and value.numel() == 1:
            payload[str(key)] = float(value.detach().cpu())
        elif isinstance(value, (bool, int, float, str)):
            payload[str(key)] = value
    path = output_dir / "edge_step_pi_first_batch_summary.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    print(f"[edge_step_pi][debug] wrote {path}", flush=True)


def _debug_ndjson(
    *,
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: Mapping[str, Any],
) -> None:
    # Legacy hypothesis logging used to append to a repo-wide file even during
    # normal training. A session path exists only for explicit debug runs.
    if get_session_ndjson_path() is None:
        return
    write_training_debug_event(
        session_id="8909e4",
        run_id=run_id,
        hypothesis_id=hypothesis_id,
        location=location,
        message=message,
        data=data,
    )


def _apply_cli_debug_to_experiment(args: argparse.Namespace, experiment: ExperimentConfig) -> None:
    if getattr(args, "debug", False):
        experiment.train.debug_mode = True
    if getattr(args, "debug_epochs", None) is not None:
        experiment.train.debug_epochs = int(args.debug_epochs)
    if getattr(args, "debug_vector_batches", None) is not None:
        experiment.train.debug_vector_batches = int(args.debug_vector_batches)
    if getattr(args, "debug_task_mode", None):
        experiment.train.debug_task_mode = args.debug_task_mode  # type: ignore[assignment]
    if getattr(args, "debug_quick_epochs", None) is not None and int(args.debug_quick_epochs) > 0:
        experiment.train.epochs = int(args.debug_quick_epochs)


def _configure_debug_session_ndjson(args: argparse.Namespace, experiment: ExperimentConfig) -> None:
    """NDJSON is written to `logs/debug-8909e4.log` and optionally to a per-run session file."""
    set_session_ndjson_path(None)
    if getattr(args, "debug_log_file", ""):
        path = Path(args.debug_log_file)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        else:
            path = path.resolve()
        set_session_ndjson_path(path)
        if _is_main():
            print(f"[debug] NDJSON session log (extra): {path}")
        return
    if getattr(args, "debug", False) or experiment.train.debug_mode:
        sess_dir = (experiment.project_root / experiment.output_dir / "debug_sessions").resolve()
        sess_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{experiment.experiment_name}_seed{experiment.seed}_lp{os.getppid()}"
        path = sess_dir / f"{stem}_rank{_rank()}.ndjson"
        set_session_ndjson_path(path)
        if _is_main():
            print(f"[debug] NDJSON session log (per rank): {path}")
            print(f"[debug] NDJSON default log (all ranks): {DEFAULT_NDJSON_LOG_PATH}")


def _save_training_curves(
    train_history: dict[str, list],
    val_history: dict[str, list],
    plot_dir: Path,
    *,
    best_epoch: int | None = None,
    best_val_loss: float | None = None,
    evaluation_checkpoint_path: str = "",
) -> None:
    """Save PNG figures for epoch-mean train losses and validation losses (rank-0 only)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs = train_history.get("epoch", [])
    if not epochs:
        return

    # --- Total: train vs val ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_history["loss_total"], label="train (total)", color="C0", linewidth=1.5)
    val_ep = val_history.get("epoch", [])
    if val_ep:
        ax.plot(
            val_ep,
            val_history["loss_total"],
            "o-",
            label="val (total)",
            color="C1",
            linewidth=1.5,
            markersize=4,
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if best_epoch is not None and best_epoch > 0:
        ax.axvline(float(best_epoch), color="C3", linestyle="--", linewidth=1.2, label=f"best epoch={best_epoch}")
    title = "Total loss"
    if best_epoch is not None and best_epoch > 0 and best_val_loss is not None:
        title += f" | best_epoch={best_epoch}, best_val_loss={best_val_loss:.6f}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "loss_total_train_val.png", dpi=150)
    plt.close(fig)

    # --- Per-task train (epoch mean) ---
    comp_keys = [
        k
        for k in train_history
        if k not in {"epoch", "loss_total"} and k.startswith("loss_") and train_history[k]
    ]
    if comp_keys:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, key in enumerate(sorted(comp_keys)):
            label = key.replace("loss_", "", 1)
            ax.plot(epochs, train_history[key], label=label, linewidth=1.2)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss (batch mean)")
        ax.set_title("Train loss by task (epoch mean)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "loss_train_components.png", dpi=150)
        plt.close(fig)

    # --- Per-task val ---
    val_comp = [
        k
        for k in val_history
        if k not in {"epoch", "loss_total"} and k.startswith("loss_") and val_history[k]
    ]
    if val_ep and val_comp:
        fig, ax = plt.subplots(figsize=(10, 5))
        for key in sorted(val_comp):
            label = key.replace("loss_", "", 1)
            ax.plot(val_ep, val_history[key], "o-", label=label, linewidth=1.2, markersize=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Validation loss by task")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "loss_val_components.png", dpi=150)
        plt.close(fig)

    (plot_dir / "training_curve_summary.json").write_text(
        json.dumps(
            {
                "best_epoch": int(best_epoch) if best_epoch is not None else -1,
                "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
                "evaluation_checkpoint_path": str(evaluation_checkpoint_path or ""),
                "selection_basis": "best_checkpoint"
                if str(evaluation_checkpoint_path or "").endswith("best.pt")
                else "last_checkpoint",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[plots] saved under {plot_dir}")


def _save_metrics_per_epoch_plots(output_dir: Path) -> None:
    """Save/update lightweight plots directly from metrics_per_epoch.csv."""
    metrics_path = output_dir / "metrics_per_epoch.csv"
    if not metrics_path.is_file():
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
    except Exception as exc:
        print(f"[plots][warn] matplotlib/pandas unavailable; skipping metrics_per_epoch plots: {exc}", flush=True)
        return

    try:
        df = pd.read_csv(metrics_path)
    except Exception as exc:
        print(f"[plots][warn] failed to read {metrics_path}: {exc}", flush=True)
        return
    if df.empty or "epoch" not in df.columns:
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs = pd.to_numeric(df["epoch"], errors="coerce")

    def _series(col: str):
        if col not in df.columns:
            return None
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() == 0:
            return None
        return values

    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        for col, label, style in (
            ("train_loss_total", "train loss", "-"),
            ("train_edge_step_pi_loss_mean", "train edge_step_pi", "--"),
            ("val_loss_total", "val loss", "o-"),
            ("val_loss", "val loss", "o-"),
        ):
            values = _series(col)
            if values is None:
                continue
            if col == "val_loss" and _series("val_loss_total") is not None:
                continue
            ax.plot(epochs, values, style, label=label, linewidth=1.5, markersize=3)
            plotted = True
        if plotted:
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.set_title("Train / Validation Loss")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(plot_dir / "loss_total_train_val.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[plots][warn] failed to save loss_total_train_val.png: {exc}", flush=True)

    try:
        component_cols = [
            ("train_loss_main_mean", "main"),
            ("train_loss_rho_mean", "rho raw"),
            ("train_loss_h_mean", "h raw"),
            ("train_edge_step_pi_loss_mean", "weighted total"),
        ]
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        for col, label in component_cols:
            values = _series(col)
            if values is None:
                continue
            ax.plot(epochs, values, label=label, linewidth=1.3)
            plotted = True
        if plotted:
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_title("edge_step_pi Raw Loss Components")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(plot_dir / "edge_step_pi_raw_component_losses.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[plots][warn] failed to save edge_step_pi component plot: {exc}", flush=True)

    try:
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        for col, label in (
            ("val_target_edge_10d_r2_edge_macro", "R2 edge macro"),
            ("val_target_edge_10d_r2_flatten", "R2 flatten"),
            ("val_target_edge_10d_r2_property_macro", "R2 property macro"),
        ):
            values = _series(col)
            if values is None:
                continue
            ax.plot(epochs, values, "o-", label=label, linewidth=1.3, markersize=3)
            plotted = True
        if plotted:
            ax.set_xlabel("epoch")
            ax.set_ylabel("R2")
            ax.set_title("Target Edge 10D Validation R2")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(plot_dir / "target_edge_10d_validation.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[plots][warn] failed to save target_edge_10d_validation.png: {exc}", flush=True)


def _format_epoch_summary_line(
    *,
    epoch: int,
    total_epochs: int,
    mean_train: float,
    lr: float,
    epoch_log: Mapping[str, float],
    edge_all: bool,
    target_only: bool,
    best_metric_name: str,
    had_validation: bool,
) -> str:
    """One-line epoch summary for terminal (train loss + val metrics when available)."""
    parts = [
        f"[epoch {epoch + 1}/{total_epochs}]",
        f"train_loss={mean_train:.6f}",
        f"lr={lr:.2e}",
    ]
    if had_validation and "val/loss" in epoch_log:
        parts.append(f"val_loss={float(epoch_log['val/loss']):.6f}")
    if had_validation and edge_all:
        parts.extend(format_epoch_summary_edge_all_parts(epoch_log))
    elif had_validation:
        _tr2 = epoch_log.get("val/metric_target_r2")
        if _tr2 is not None and math.isfinite(float(_tr2)):
            parts.append(f"val_target_r2={float(_tr2):.6f}")
        if not target_only:
            _cr2 = epoch_log.get("val/metric_tailgas_r2")
            if _cr2 is not None and math.isfinite(float(_cr2)):
                parts.append(f"val_tailgas_r2={float(_cr2):.6f}")
    elif not had_validation:
        parts.append("val=skipped")
    _bsf = epoch_log.get("best_so_far")
    if _bsf is not None and math.isfinite(float(_bsf)):
        parts.append(f"best({best_metric_name})={float(_bsf):.6f}")
    return " | ".join(parts)


def _format_pinn_conservation_precision_line(epoch_log: Mapping[str, float]) -> str | None:
    """High-precision report for the three active node-conservation terms."""
    names = ("mass", "component", "atom")

    def _values(prefix: str, suffix: str) -> list[float]:
        return [float(epoch_log.get(f"{prefix}/{suffix.format(name=name)}", float("nan"))) for name in names]

    train_loss = _values("train", "loss_node_{name}")
    if not any(math.isfinite(value) for value in train_loss):
        return None
    train_weighted = _values("train", "weighted_node_{name}")
    train_residual = _values("train", "node_{name}_residual_mean")
    train_count = _values("train", "node_{name}_valid_count")
    val_loss = _values("val", "loss_node_{name}")

    def _triplet(values: list[float], fmt: str = ".12e") -> str:
        return "/".join(format(value, fmt) if math.isfinite(value) else "NA" for value in values)

    parts = [
        "[pinn-conservation mass/component/atom]",
        f"train_raw={_triplet(train_loss)}",
        f"train_weighted={_triplet(train_weighted)}",
        f"train_abs_residual_mean={_triplet(train_residual)}",
        f"train_valid_per_batch={_triplet(train_count, '.2f')}",
    ]
    if any(math.isfinite(value) for value in val_loss):
        parts.append(f"val_raw={_triplet(val_loss)}")
    else:
        parts.append("val_raw=disabled")
    return " | ".join(parts)


def _append_metrics_per_epoch_csv(
    output_dir: Path,
    *,
    edge_all: bool,
    target_only: bool,
    epoch: int,
    train_loss: float,
    lr: float,
    val_metrics: Mapping[str, float] | None,
    answer_weight_h2: float,
    answer_weight_co2: float,
    train_components: Mapping[str, float] | None = None,
) -> None:
    """매 에폭 train + (해당 에폭에 val이 있으면) val 지표를 metrics_per_epoch.csv에 한 줄 추가."""
    path = output_dir / "metrics_per_epoch.csv"
    if edge_all:
        from process_graph.experiment.metric_policy import build_edge_all_metrics_per_epoch_row

        tc = dict(train_components or {})
        row = build_edge_all_metrics_per_epoch_row(
            epoch=epoch,
            train_loss_total=train_loss,
            lr=lr,
            train_components=tc,
            val_metrics=val_metrics,
        )
        df = pd.DataFrame([row])
        header = not path.is_file()
        df.to_csv(path, mode="a", index=False, header=header, encoding="utf-8")
        diag_keys = (
            "loss_edge_all",
            "loss_target_incident_edges",
            "weighted_target_incident_edges",
            "loss_all_edge_r2",
            "loss_total_final",
            "target_incident_edge_count",
            "target_incident_supervised_cell_count",
            "target_incident_target_row_count",
            "target_incident_matched_edge_count",
            "target_incident_virtual_endpoint_excluded_count",
            "target_incident_target_edge_included_count",
            "target_incident_virtual_filter_applied",
            "target_incident_apply_answer_weight",
            "target_incident_inner_weight_mean",
            "target_incident_inner_weight_max",
            "target_incident_effective_weight_mean",
            "target_incident_coverage_edge_frac",
            "target_incident_loss_ratio_total",
            "target_incident_loss_ratio_edge",
            "target_stream_loss_weighting_enabled",
            "num_target_stream_edges_weighted",
            "num_target_stream_edges_with_yaml_override",
            "num_target_stream_rows_skipped",
            "loss_primary_frac_deprecated",
            "loss_primary_frac_contributes_to_total",
            "amount_metrics_are_primary",
        )
        diag_row: dict[str, Any] = {"epoch": int(epoch), "train_lr": float(lr)}
        for key in diag_keys:
            if key in tc:
                diag_row[f"train_{key}"] = float(tc[key])
            if val_metrics is not None and key in val_metrics:
                try:
                    diag_row[f"val_{key}"] = float(val_metrics[key])
                except (TypeError, ValueError):
                    diag_row[f"val_{key}"] = val_metrics[key]
        dpath = output_dir / "incident_loss_epoch_summary.csv"
        pd.DataFrame([diag_row]).to_csv(
            dpath,
            mode="a",
            index=False,
            header=not dpath.is_file(),
            encoding="utf-8",
        )
        return

    row: dict[str, Any] = {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "train_lr": float(lr),
        "had_validation": 1 if val_metrics is not None else 0,
    }
    if val_metrics is not None:
        for k, v in sorted(val_metrics.items()):
            try:
                row[f"val_{k}"] = float(v)
            except (TypeError, ValueError):
                row[f"val_{k}"] = v
        if edge_all:
            h2 = float(val_metrics.get("metric_target_r2", float("nan")))
            c2 = float(val_metrics.get("metric_tailgas_r2", float("nan")))
            wh, wc = float(answer_weight_h2), float(answer_weight_co2)
            wsum = wh + wc
            if wsum <= 0:
                wh = wc = 1.0
                wsum = 2.0
            if math.isfinite(h2) and math.isfinite(c2):
                row["val_answer_targets_r2_weighted"] = (wh * h2 + wc * c2) / wsum
            elif math.isfinite(h2):
                row["val_answer_targets_r2_weighted"] = h2
            elif math.isfinite(c2):
                row["val_answer_targets_r2_weighted"] = c2
            else:
                row["val_answer_targets_r2_weighted"] = float("nan")
            mr = float(val_metrics.get("metric_answer_all_targets_mean_r2", float("nan")))
            if math.isfinite(mr):
                row["val_answer_targets_r2_mean"] = mr
                row["val_legacy_answer_fraction_macro_r2"] = mr
            try:
                from process_graph.experiment.target_row_primary_metrics import write_edge_all_epoch_metric_columns

                write_edge_all_epoch_metric_columns(row, val_metrics)
            except ImportError:
                pass
        elif not target_only:
            h2 = float(val_metrics.get("metric_target_r2", float("nan")))
            c2 = float(val_metrics.get("metric_tailgas_r2", float("nan")))
            wh, wc = float(answer_weight_h2), float(answer_weight_co2)
            wsum = wh + wc
            if wsum <= 0:
                wh = wc = 1.0
                wsum = 2.0
            if math.isfinite(h2) and math.isfinite(c2):
                row["val_answer_targets_r2_weighted"] = (wh * h2 + wc * c2) / wsum
            elif math.isfinite(h2):
                row["val_answer_targets_r2_weighted"] = h2
            elif math.isfinite(c2):
                row["val_answer_targets_r2_weighted"] = c2
            else:
                row["val_answer_targets_r2_weighted"] = float("nan")
        else:
            row["val_answer_targets_r2_weighted"] = float(val_metrics.get("metric_target_r2", float("nan")))
    df = pd.DataFrame([row])
    header = not path.is_file()
    df.to_csv(path, mode="a", index=False, header=header, encoding="utf-8")


def _append_incident_loss_diagnostic_rows(
    output_dir: Path,
    *,
    split: str,
    epoch: int,
    step: int,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    y_mask: torch.Tensor,
    export_meta: Any,
    edge_index: torch.Tensor | None,
    train_cfg: Any,
    loss_scalars: Mapping[str, Any],
) -> None:
    if export_meta is None or edge_index is None:
        return
    rows = build_target_incident_diagnostic_rows(
        split=split,
        epoch=epoch,
        step=step,
        y_pred=y_pred,
        y_true=y_true,
        y_mask=y_mask,
        export_meta=export_meta,
        edge_index=edge_index,
        train_cfg=train_cfg,
        loss_scalars=loss_scalars,
    )
    if not rows:
        return
    matched = max(int(r.get("matched_target_edges", 0) or 0) for r in rows)
    max_eff = max(float(r.get("effective_weight", 0.0) or 0.0) for r in rows)
    if matched <= 0:
        print(f"[incident_loss][warn] matched_target_edges=0 epoch={epoch} step={step}", flush=True)
    if max_eff > 10.0:
        print(f"[incident_loss][warn] effective_weight>10 max={max_eff:.3f} epoch={epoch} step={step}", flush=True)
    path = output_dir / "incident_loss_diagnostics.csv"
    _ensure_output_dir(output_dir)
    pd.DataFrame(rows).to_csv(
        path,
        mode="a",
        index=False,
        header=not path.is_file(),
        encoding="utf-8",
    )


def _save_val_r2_and_mae_plots(
    val_history: dict[str, list],
    plot_dir: Path,
    *,
    edge_all: bool,
    target_only: bool,
) -> None:
    """검증 R²(타깃/테일가스) 및 MAE 곡선 PNG (--no-training-plots 와 무관하게 호출 가능)."""
    val_ep = val_history.get("epoch") or []
    if not val_ep:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    if edge_all:
        h2r = val_history.get("metric_target_r2") or val_history.get("target_h2_r2")
        c2r = val_history.get("metric_tailgas_r2") or val_history.get("tailgas_co2_r2")
        if h2r and len(h2r) == len(val_ep):
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            ax.plot(val_ep, h2r, "o-", label="val metric_target_r2 (H2 answer)", color="C0", linewidth=1.2)
            if c2r and len(c2r) == len(val_ep):
                ax.plot(val_ep, c2r, "s-", label="val metric_tailgas_r2 (CO2 answer)", color="C1", linewidth=1.2)
            mr = val_history.get("metric_answer_all_targets_mean_r2")
            if mr and len(mr) == len(val_ep):
                ax.plot(val_ep, mr, "d-", label="val mean R2 (all answer targets)", color="C2", linewidth=1.2)
            ax.set_xlabel("epoch")
            ax.set_ylabel("R2")
            ax.set_title("Validation R2 (answer edges)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / "val_r2_target_h2_tailgas_co2.png", dpi=150)
            plt.close(fig)
        mae_h2 = val_history.get("target_h2_mae")
        mae_co2 = val_history.get("tailgas_co2_mae")
        amae = val_history.get("answer_targets_mae")
        if mae_h2 and len(mae_h2) == len(val_ep):
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            ax.plot(val_ep, mae_h2, "o-", label="target_h2_mae", color="C0")
            if mae_co2 and len(mae_co2) == len(val_ep):
                ax.plot(val_ep, mae_co2, "s-", label="tailgas_co2_mae", color="C1")
            if amae and len(amae) == len(val_ep):
                ax.plot(val_ep, amae, "^--", label="answer_targets_mae", color="C3")
            ax.set_xlabel("epoch")
            ax.set_ylabel("MAE")
            ax.set_title("Validation MAE (answer edges)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / "val_mae_answer_edges.png", dpi=150)
            plt.close(fig)
        emae = val_history.get("edge_all_mae")
        emse = val_history.get("edge_all_mse")
        if emae and len(emae) == len(val_ep):
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            ax.plot(val_ep, emae, "o-", label="edge_all_mae (norm)", color="C2")
            if emse and len(emse) == len(val_ep):
                ax2 = ax.twinx()
                ax2.plot(val_ep, emse, "s--", label="edge_all_mse (norm)", color="C4", alpha=0.85)
                ax2.set_ylabel("MSE (norm)")
                lines1, lab1 = ax.get_legend_handles_labels()
                lines2, lab2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, lab1 + lab2, loc="upper right")
            else:
                ax.legend()
            ax.set_xlabel("epoch")
            ax.set_ylabel("MAE (norm)")
            ax.set_title("Validation edge_all MAE / MSE (normalized space)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / "val_edge_all_mae_mse_norm.png", dpi=150)
            plt.close(fig)
    else:
        tr = val_history.get("metric_target_r2")
        if tr and len(tr) == len(val_ep):
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            ax.plot(val_ep, tr, "o-", label="val metric_target_r2", color="C0")
            if not target_only:
                tgr = val_history.get("metric_tailgas_r2")
                if tgr and len(tgr) == len(val_ep):
                    ax.plot(val_ep, tgr, "s-", label="val metric_tailgas_r2", color="C1")
            ax.set_xlabel("epoch")
            ax.set_ylabel("R2")
            ax.set_title("Validation R2 (multitask)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(plot_dir / "val_r2_multitask.png", dpi=150)
            plt.close(fig)


def _prepare_fixed_target_frame(
    csv_path: Path,
    process_id_column: str,
    split_column: str,
    split_filter: str | None,
    fixed_tasks: Mapping[str, Any],
    *,
    allowed_row_indices: set[int] | None = None,
) -> pd.DataFrame:
    columns = {process_id_column, split_column}
    for cfg in fixed_tasks.values():
        target_variable = getattr(cfg, "target_variable", None)
        if target_variable:
            columns.add(str(target_variable))
    for slot_map in PROCESS_FIXED_TARGET_COLUMNS.values():
        for column in slot_map.values():
            columns.add(str(column))
    frame = pd.read_csv(csv_path, usecols=lambda c: c in columns)
    if allowed_row_indices is not None:
        idxs = sorted(i for i in allowed_row_indices if 0 <= i < len(frame))
        frame = frame.iloc[idxs].copy()
    elif split_filter is not None and split_column in frame.columns:
        frame = frame[frame[split_column].astype(str) == split_filter].copy()
    return frame


def _read_split_manifest_indices(project_root: Path, relative_or_abs: str) -> set[int]:
    rel = (relative_or_abs or "").strip()
    if not rel:
        return set()
    path = Path(rel)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"split manifest not found: {path}")
    man = pd.read_csv(path)
    for cand in ("merged_row_index", "row_index", "csv_row_index"):
        if cand in man.columns:
            return {int(x) for x in man[cand].tolist()}
    raise ValueError(f"{path} must contain merged_row_index (or row_index / csv_row_index).")


def _split_manifest_row_count(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"split manifest not found: {path}")
    return int(len(pd.read_csv(path)))


def _resolve_fixed_target_column_for_process(
    process_id: str,
    slot_name: str,
    fixed_tasks: Mapping[str, Any],
) -> str | None:
    mapped = PROCESS_FIXED_TARGET_COLUMNS.get(str(process_id), {}).get(slot_name)
    if mapped:
        return mapped
    cfg = fixed_tasks.get(slot_name)
    if cfg is None:
        return None
    column = getattr(cfg, "target_variable", None)
    return str(column) if column else None


def _collect_routed_fixed_target_values(
    frame: pd.DataFrame,
    *,
    process_id_column: str,
    slot_name: str,
    fixed_tasks: Mapping[str, Any],
) -> pd.Series:
    values: list[float] = []
    for process_id, group in frame.groupby(process_id_column):
        column = _resolve_fixed_target_column_for_process(str(process_id), slot_name, fixed_tasks)
        if not column or column not in group.columns:
            continue
        series = pd.to_numeric(group[column], errors="coerce").dropna().astype(float)
        if not series.empty:
            values.extend(series.tolist())
    return pd.Series(values, dtype="float64")


def _compute_fixed_slot_clip_maxes(
    frame: pd.DataFrame,
    *,
    process_id_column: str,
    fixed_tasks: Mapping[str, Any],
) -> dict[str, float]:
    clip_max: dict[str, float] = {}
    for slot_name, cfg in fixed_tasks.items():
        percentile = getattr(cfg, "clip_max_percentile", None)
        if percentile is None:
            continue
        values = _collect_routed_fixed_target_values(
            frame,
            process_id_column=process_id_column,
            slot_name=slot_name,
            fixed_tasks=fixed_tasks,
        )
        if values.empty:
            continue
        clip_max[slot_name] = float(values.quantile(float(percentile) / 100.0))
    return clip_max


def _build_x_oper_audit(project_root: Path, out_dir: Path) -> pd.DataFrame:
    spec_dir = project_root / "data" / "process_specs" / "raw"
    main_dir = project_root / "data" / "main_data"
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for pid in range(1, 11):
        process_id = f"Process{pid}"
        spec_path = spec_dir / f"{process_id}_Adjacency_Matrix.xlsx"
        main_path = main_dir / f"{pid}.Process_Main.csv"
        if not spec_path.is_file() or not main_path.is_file():
            continue
        spec = parse_process_file(str(spec_path), process_id=process_id)
        frame = pd.read_csv(main_path)
        alias_map = {
            "Process2": {"CH4_Flow": "CH4_Total"},
            "Process6": {"P_Burner": "P_Comb", "FUEL_Flow": "FUEL_CH4_Flow"},
            "Process8": {"SC_Ratio": "S/C_ratio", "P_Reaction": "P_Rxn", "T_HEAT1": "T_Heat1"},
            "Process9": {"Q_BURNER": "Q_Burner", "PROD_Mole": "PROD_H2_Mole"},
            "Process10": {"Q_BURNER": "Q_BN"},
        }.get(process_id, {})
        linear_pat = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*\*\s*([A-Za-z0-9_]+)\s*$")
        for node_name in spec.node_names:
            node = spec.nodes[node_name]
            for oper_name in OPER_FEATURE_SLOTS:
                cell_spec = node.features.get(oper_name)
                source_column: str | None = None
                if cell_spec is not None and cell_spec.kind == "reference":
                    source_column = str(cell_spec.value)
                elif cell_spec is not None and cell_spec.kind == "constant":
                    source_column = f"CONST:{cell_spec.value}"
                elif cell_spec is not None and cell_spec.kind == "passthrough":
                    source_column = f"PASSTHROUGH:{cell_spec.meta.get('hint_ref', '')}".strip(":")
                n_rows = int(len(frame))
                if cell_spec is None or cell_spec.kind == "missing":
                    s = pd.Series([0.0] * n_rows, dtype="float64")
                    missing_count = n_rows
                elif cell_spec.kind == "constant":
                    s = pd.Series([float(cell_spec.value)] * n_rows, dtype="float64")
                    missing_count = 0
                elif cell_spec.kind == "passthrough":
                    hint_ref = str(cell_spec.meta.get("hint_ref", "") or "")
                    if hint_ref and hint_ref in frame.columns:
                        s = pd.to_numeric(frame[hint_ref], errors="coerce").fillna(0.0).astype(float)
                    else:
                        s = pd.Series([0.0] * n_rows, dtype="float64")
                    missing_count = 0
                else:  # reference
                    ref = str(cell_spec.value).strip()
                    matched_col = None
                    coef = 1.0
                    m = linear_pat.match(ref)
                    if m:
                        coef = float(m.group(1))
                        ref = m.group(2).strip()
                    if ref in frame.columns:
                        matched_col = ref
                    elif ref in alias_map and alias_map[ref] in frame.columns:
                        matched_col = alias_map[ref]
                    else:
                        ref_fold = ref.casefold()
                        for c in frame.columns:
                            if str(c).casefold() == ref_fold:
                                matched_col = str(c)
                                break
                    if matched_col is None:
                        s = pd.Series([0.0] * n_rows, dtype="float64")
                        missing_count = n_rows
                    else:
                        source_column = matched_col if coef == 1.0 else f"{coef}*{matched_col}"
                        s = coef * pd.to_numeric(frame[matched_col], errors="coerce").fillna(0.0).astype(float)
                        missing_count = int(pd.to_numeric(frame[matched_col], errors="coerce").isna().sum())
                zero_count = int((s == 0.0).sum())
                if source_column is None or source_column.startswith("PASSTHROUGH") or missing_count > 0:
                    warnings.append(
                        f"{process_id}:{node_name}:{oper_name} source={source_column!r} missing_count={missing_count}"
                    )
                slot_index = int(OPER_FEATURE_SLOTS.index(oper_name))
                std_val = float(s.std(ddof=0)) if len(s) else 0.0
                rows.append(
                    {
                        "process_id": process_id,
                        "node_id": node_name,
                        "slot_index": slot_index,
                        "oper_name": oper_name,
                        "source_column": source_column,
                        "mean": float(s.mean()) if len(s) else 0.0,
                        "std": std_val,
                        "min": float(s.min()) if len(s) else 0.0,
                        "max": float(s.max()) if len(s) else 0.0,
                        "missing_count": missing_count,
                        "zero_count": zero_count,
                        "zero_ratio": float(zero_count / max(len(s), 1)),
                        "constant_flag": bool(std_val <= 1e-12),
                        "missing_flag": bool(missing_count > 0),
                        "n_rows": int(len(s)),
                    }
                )
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_df = pd.DataFrame(rows)
    if not audit_df.empty:
        audit_df.to_csv(out_dir / "x_oper_audit_all.csv", index=False)
        audit_df.to_csv(out_dir / "x_oper_slot_audit_all.csv", index=False)
        for process_id, g in audit_df.groupby("process_id"):
            pid_num = str(process_id).replace("Process", "")
            g.to_csv(out_dir / f"process_{pid_num}_x_oper_audit.csv", index=False)
    if warnings:
        warn_path = out_dir / "x_oper_audit_warnings.txt"
        warn_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
        for msg in warnings[:80]:
            print(f"[x_oper_audit][warning] {msg}")
    return audit_df


def _build_target_and_answer_audits(project_root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    v4_path = project_root / "data" / "reference" / "v4" / "target_stream_targets.csv"
    v3_edges_path = project_root / "data" / "reference" / "v3" / "canonical_edges.csv"
    v3_answers_path = project_root / "data" / "reference" / "v3" / "target_answer_edges.csv"
    main_dir = project_root / "data" / "main_data"
    stream_dir = project_root / "data" / "main_data_Streams"
    tfv_path = project_root / "data" / "reference" / "v4" / "target_formula_validation.csv"
    if not v4_path.is_file():
        return pd.DataFrame(), pd.DataFrame()

    tdf = pd.read_csv(v4_path)
    tfv = pd.read_csv(tfv_path) if tfv_path.is_file() else pd.DataFrame()
    edges = pd.read_csv(v3_edges_path)
    answers = pd.read_csv(v3_answers_path)
    t_rows: list[dict[str, Any]] = []
    a_rows: list[dict[str, Any]] = []
    linear_pat = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*\*\s*([A-Za-z0-9_]+)\s*$")

    def _truthy(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        return s in {"1", "true", "yes", "y"}

    answer_lookup = {
        (int(r["process_id"]), str(r["task_name"])): str(r["canonical_answer_edge_id"])
        for _, r in answers.iterrows()
    }

    for rec in tdf.to_dict(orient="records"):
        pid = int(rec["process_id"])
        process_id = f"Process{pid}"
        main_path = main_dir / f"{pid}.Process_Main.csv"
        stream_path = stream_dir / f"{pid}.Process_Streams.csv"
        main_cols = set(pd.read_csv(main_path, nrows=1).columns) if main_path.is_file() else set()
        stream_keys = (
            set(
                pd.read_csv(stream_path, usecols=["Stream_Name"])["Stream_Name"]
                .map(canonicalize_stream_key)
                .tolist()
            )
            if stream_path.is_file()
            else set()
        )
        target_col = str(rec.get("target_feature_name", ""))
        target_col_main = target_col
        m = linear_pat.match(target_col)
        if m:
            target_col_main = m.group(2).strip()
        required_key = _normalize_stream_key(rec.get("required_stream_key") or rec.get("main_data_stream_key"))
        cand_edge = edges[
            (edges["process_id"].astype(int) == pid)
            & (edges["main_data_stream_key"].astype(str).map(_normalize_stream_key) == required_key)
        ]
        canonical_edge_exists = len(cand_edge) > 0
        answer_edge_id_registry = str(rec.get("canonical_answer_edge_id", ""))
        species = str(rec.get("target_species", "")).upper()
        task_name = "target_h2" if species == "H2" else ("tailgas_co2" if species == "CO2" else "")
        answer_edge_id_matched = answer_lookup.get((pid, task_name), answer_edge_id_registry)
        answer_row = edges[
            (edges["process_id"].astype(int) == pid)
            & (edges["canonical_edge_id"].astype(str) == answer_edge_id_matched)
        ]
        dst_node = str(answer_row.iloc[0]["dst_node"]) if len(answer_row) else ""
        src_node = str(answer_row.iloc[0]["src_node"]) if len(answer_row) else ""
        matched_stream_key = str(answer_row.iloc[0]["main_data_stream_key"]) if len(answer_row) else ""
        is_output_edge = bool(
            len(answer_row)
            and str(answer_row.iloc[0].get("is_output_edge", "")).strip().lower() in {"1", "true", "yes"}
        )
        v_output_connected = bool(len(answer_row) and dst_node == "V_OUTPUT")
        is_valid_answer_edge = bool(len(answer_row) and v_output_connected)
        warning = ""
        tfv_match = tfv[
            (tfv.get("process_id", pd.Series(dtype=int)).astype(int) == pid)
            & (tfv.get("target_feature_name", pd.Series(dtype=str)).astype(str) == target_col)
        ] if not tfv.empty else pd.DataFrame()
        direct_exists = bool(tfv_match.iloc[0]["direct_target_exists"]) if len(tfv_match) else (target_col_main in main_cols)
        if target_col_main and target_col_main not in main_cols and direct_exists:
            warning = f"missing_main_column:{target_col}"
        if required_key and required_key not in stream_keys:
            warning = (warning + ";" if warning else "") + f"missing_stream_key:{required_key}"
        if not canonical_edge_exists:
            warning = (warning + ";" if warning else "") + "missing_canonical_edge_for_required_key"
        if answer_edge_id_registry != answer_edge_id_matched:
            warning = (warning + ";" if warning else "") + (
                f"registry_answer_edge_mismatch:{answer_edge_id_registry}->{answer_edge_id_matched}"
            )
        if not is_valid_answer_edge:
            warning = (warning + ";" if warning else "") + "invalid_answer_edge"
        t_rows.append(
            {
                "process_id": process_id,
                "target_id": rec.get("target_id"),
                "target_column": target_col,
                "target_column_main": target_col_main,
                "target_stream": rec.get("target_stream_node"),
                "target_species": species,
                "formula": rec.get("target_formula"),
                "main_column_exists": target_col_main in main_cols if target_col_main else False,
                "stream_key_exists": required_key in stream_keys if required_key else False,
                "canonical_edge_exists": canonical_edge_exists,
                "canonical_answer_edge_id": answer_edge_id_registry,
                "matched_edge_id": answer_edge_id_matched,
                "src_node": src_node,
                "dst_node": dst_node,
                "main_data_stream_key": matched_stream_key,
                "is_output_edge": is_output_edge,
                "v_output_connected": v_output_connected,
                "is_valid_answer_edge": is_valid_answer_edge,
                "warning": warning,
                "target_source": "main" if direct_exists else "streams",
            }
        )

    for rec in answers.to_dict(orient="records"):
        pid = int(rec["process_id"])
        process_id = f"Process{pid}"
        edge_id = str(rec.get("canonical_answer_edge_id", ""))
        match = edges[
            (edges["process_id"].astype(int) == pid)
            & (edges["canonical_edge_id"].astype(str) == edge_id)
        ]
        dst_node = str(match.iloc[0]["dst_node"]) if len(match) else ""
        a_rows.append(
            {
                "process_id": process_id,
                "task_name": rec.get("task_name"),
                "target_column": rec.get("target_column"),
                "answer_edge_id": edge_id,
                "edge_exists": bool(len(match)),
                "dst_node": dst_node,
                "is_valid_answer_edge": bool(len(match) and dst_node == "V_OUTPUT"),
                "warning": "" if bool(len(match) and dst_node == "V_OUTPUT") else "invalid_or_missing_answer_edge",
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    tdf_out = pd.DataFrame(t_rows)
    adf_out = pd.DataFrame(a_rows)
    tdf_out.to_csv(out_dir / "target_mapping_audit.csv", index=False)
    adf_out.to_csv(out_dir / "answer_edge_audit.csv", index=False)
    return tdf_out, adf_out


def _build_main_tabular_sanity_baseline(project_root: Path, out_dir: Path) -> pd.DataFrame:
    try:
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split
    except Exception as exc:  # pragma: no cover
        print(f"[sanity_baseline][warning] sklearn unavailable: {exc}")
        return pd.DataFrame()
    tfv_path = project_root / "data" / "reference" / "v4" / "target_formula_validation.csv"
    if not tfv_path.is_file():
        return pd.DataFrame()
    targets = pd.read_csv(tfv_path)
    rows: list[dict[str, Any]] = []
    for pid in range(1, 11):
        process_id = f"Process{pid}"
        main_path = project_root / "data" / "main_data" / f"{pid}.Process_Main.csv"
        if not main_path.is_file():
            continue
        frame = pd.read_csv(main_path)
        proc_targets = targets[
            (targets["process_id"].astype(int) == pid)
            & (targets["direct_target_exists"].astype(bool))
            & (targets["target_feature_name"].astype(str).isin(frame.columns))
        ]
        drop_cols = {"ID"}
        drop_cols.update([c for c in frame.columns if c.startswith("Q_") or c.startswith("W_") or c.startswith("VolFlow")])
        for _, tr in proc_targets.iterrows():
            y_col = str(tr["target_feature_name"])
            if y_col not in frame.columns:
                continue
            numeric_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
            feature_cols = [c for c in numeric_cols if c not in drop_cols and c != y_col]
            if len(feature_cols) < 3:
                continue
            X = frame[feature_cols].fillna(0.0)
            y = frame[y_col].astype(float)
            xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            mdl = ExtraTreesRegressor(n_estimators=80, random_state=42, n_jobs=1)
            mdl.fit(xtr, ytr)
            pred = mdl.predict(xte)
            rows.append(
                {
                    "process_id": process_id,
                    "target_column": y_col,
                    "n_features": len(feature_cols),
                    "n_train": len(xtr),
                    "n_test": len(xte),
                    "r2": float(r2_score(yte, pred)),
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df.to_csv(out_dir / "main_tabular_sanity_baseline.csv", index=False)
    return out_df


def _build_main_full_vs_gnn_xoper_baseline(
    project_root: Path,
    out_dir: Path,
    process_ids: list[int] | None = None,
) -> pd.DataFrame:
    try:
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split
    except Exception as exc:  # pragma: no cover
        print(f"[baseline_compare][warning] sklearn unavailable: {exc}")
        return pd.DataFrame()

    pids = process_ids or list(range(1, 11))
    tfv_path = project_root / "data" / "reference" / "v4" / "target_formula_validation.csv"
    spec_dir = project_root / "data" / "process_specs" / "raw"
    main_dir = project_root / "data" / "main_data"
    if not tfv_path.is_file():
        return pd.DataFrame()

    tfv = pd.read_csv(tfv_path)
    lin_pat = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*\*\s*([A-Za-z0-9_]+)\s*$")
    rows: list[dict[str, Any]] = []

    for pid in pids:
        process_id = f"Process{pid}"
        main_path = main_dir / f"{pid}.Process_Main.csv"
        spec_path = spec_dir / f"{process_id}_Adjacency_Matrix.xlsx"
        if not main_path.is_file() or not spec_path.is_file():
            continue
        df = pd.read_csv(main_path)
        if df.empty:
            continue
        spec = parse_process_file(str(spec_path), process_id=process_id)

        # Build graph x_oper feature matrix per sample
        x_oper_vecs: list[list[float]] = []
        valid_idx: list[int] = []
        for i, rec in enumerate(df.to_dict(orient="records")):
            data_row = {
                k: float(v)
                for k, v in rec.items()
                if pd.notna(v) and isinstance(v, (int, float))
            }
            g = build_graph_sample(spec, data_row, passthrough_policy="zero")
            v = torch.tensor(g.x_oper, dtype=torch.float32).flatten().tolist()
            x_oper_vecs.append(v)
            valid_idx.append(i)
        X_xoper = pd.DataFrame(x_oper_vecs, index=valid_idx)

        proc_targets = tfv[
            (tfv["process_id"].astype(int) == pid)
            & (tfv["direct_target_exists"].astype(bool))
        ].copy()
        for _, tr in proc_targets.iterrows():
            target_id = str(tr["target_id"])
            y_expr = str(tr["target_feature_name"])
            coef = 1.0
            y_col = y_expr
            m = lin_pat.match(y_expr)
            if m:
                coef = float(m.group(1))
                y_col = m.group(2).strip()
            if y_col not in df.columns:
                continue
            y = coef * pd.to_numeric(df[y_col], errors="coerce")
            use_idx = y.dropna().index
            if len(use_idx) < 100:
                continue
            yv = y.loc[use_idx].astype(float)

            # A. Main full-input baseline
            drop_cols = {"ID", y_col}
            drop_cols.update(
                [c for c in df.columns if c.startswith("Q_") or c.startswith("W_") or c.startswith("VolFlow")]
            )
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            feat_cols = [c for c in num_cols if c not in drop_cols]
            if len(feat_cols) < 3:
                continue
            X_main = df.loc[use_idx, feat_cols].fillna(0.0)

            # B. GNN x_oper-only baseline
            X_op = X_xoper.loc[use_idx].fillna(0.0)
            if X_op.shape[1] < 3:
                continue

            xtr_m, xte_m, ytr, yte = train_test_split(X_main, yv, test_size=0.2, random_state=42)
            xtr_o, xte_o, ytr_o, yte_o = train_test_split(X_op, yv, test_size=0.2, random_state=42)

            mdl_m = ExtraTreesRegressor(n_estimators=80, random_state=42, n_jobs=1)
            mdl_o = ExtraTreesRegressor(n_estimators=80, random_state=42, n_jobs=1)
            mdl_m.fit(xtr_m, ytr)
            mdl_o.fit(xtr_o, ytr_o)
            r2_main = float(r2_score(yte, mdl_m.predict(xte_m)))
            r2_xoper = float(r2_score(yte_o, mdl_o.predict(xte_o)))
            rows.append(
                {
                    "process_id": process_id,
                    "target_id": target_id,
                    "target_column": y_expr,
                    "main_full_r2": r2_main,
                    "gnn_xoper_r2": r2_xoper,
                    "gap": r2_main - r2_xoper,
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df.to_csv(out_dir / "baseline_main_full_vs_gnn_xoper.csv", index=False)
    return out_df


@torch.no_grad()
def _dump_batch_debug_by_process(
    *,
    project_root: Path,
    experiment: ExperimentConfig,
    oper_mean,
    oper_std,
    target_mean,
    target_std,
    fixed_slot_clip_max,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for pid in range(1, 11):
        data_cfg = replace(
            experiment.data,
            edge_all_processes=[pid],
            edge_all_train_processes=None,
            edge_all_val_processes=None,
            edge_all_test_processes=None,
        )
        ds = ProcessGraphTabularDataset(
            experiment.data.train_data_path,
            data_cfg,
            project_root,
            split_filter="train",
            oper_mean=oper_mean,
            oper_std=oper_std,
            target_mean=target_mean,
            target_std=target_std,
            fixed_slot_clip_max=fixed_slot_clip_max,
        )
        if len(ds) == 0:
            continue
        ld = DataLoader(
            ds,
            batch_size=min(16, len(ds)),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=_collate_for_experiment(data_cfg),
        )
        b = next(iter(ld))
        x_oper = b.model_kwargs["x_oper"].float()
        y_target = b.targets.get("target", torch.zeros(0))
        y_edge = b.targets.get("edge_stream", torch.zeros(0))
        edge_mask = b.target_masks.get("edge_stream", torch.zeros(0))
        target_mask = b.target_masks.get("target", torch.zeros(0))
        payload = {
            "process_id": f"Process{pid}",
            "x_oper": {
                "mean": float(x_oper.mean()),
                "std": float(x_oper.std(unbiased=False)),
                "min": float(x_oper.min()),
                "max": float(x_oper.max()),
            },
            "node_x_oper_preview": x_oper[: min(10, x_oper.shape[0]), :].tolist(),
            "y_target": {
                "mean": float(y_target.mean()) if y_target.numel() else 0.0,
                "std": float(y_target.std(unbiased=False)) if y_target.numel() else 0.0,
                "min": float(y_target.min()) if y_target.numel() else 0.0,
                "max": float(y_target.max()) if y_target.numel() else 0.0,
            },
            "y_edge": {
                "mean": float(y_edge.mean()) if y_edge.numel() else 0.0,
                "std": float(y_edge.std(unbiased=False)) if y_edge.numel() else 0.0,
                "min": float(y_edge.min()) if y_edge.numel() else 0.0,
                "max": float(y_edge.max()) if y_edge.numel() else 0.0,
            },
            "edge_mask_coverage": float(edge_mask.mean()) if edge_mask.numel() else 0.0,
            "target_mask_coverage": float(target_mask.mean()) if target_mask.numel() else 0.0,
        }
        (out_dir / f"batch_debug_process_{pid}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )


def _analyze_fixed_target_distribution(
    frame: pd.DataFrame,
    *,
    process_id_column: str,
    fixed_tasks: Mapping[str, Any],
    batch_size: int,
    output_dir: Path,
    top_k: int,
) -> None:
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    analysis: dict[str, Any] = {"fixed_targets": {}}
    for slot_name, cfg in fixed_tasks.items():
        values = _collect_routed_fixed_target_values(
            frame,
            process_id_column=process_id_column,
            slot_name=slot_name,
            fixed_tasks=fixed_tasks,
        )
        if values.empty:
            continue
        series = values.reset_index(drop=True)
        batch_rows: list[dict[str, float | int]] = []
        for batch_idx in range(0, len(series), batch_size):
            batch = series.iloc[batch_idx : batch_idx + batch_size]
            if batch.empty:
                continue
            batch_rows.append(
                {
                    "batch_index": int(batch_idx // batch_size),
                    "size": int(len(batch)),
                    "min": float(batch.min()),
                    "max": float(batch.max()),
                    "mean": float(batch.mean()),
                    "std": float(batch.std(ddof=0)),
                    "span": float(batch.max() - batch.min()),
                }
            )

        batch_df = pd.DataFrame(batch_rows)
        batch_csv = debug_dir / f"{slot_name}_batch_stats.csv"
        if not batch_df.empty:
            batch_df.to_csv(batch_csv, index=False)
            top_batches = (
                batch_df.sort_values(["std", "span"], ascending=False)
                .head(max(int(top_k), 1))
                .to_dict(orient="records")
            )
        else:
            top_batches = []

        stats = {
            "routing_mode": "process_fixed_target_columns",
            "count": int(series.shape[0]),
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "p95": float(series.quantile(0.95)),
            "p99": float(series.quantile(0.99)),
            "max_to_p99_ratio": float(series.max() / max(float(series.quantile(0.99)), 1e-12)),
            "batch_stats_csv": str(batch_csv),
            "top_batches_by_std": top_batches,
        }
        analysis["fixed_targets"][slot_name] = stats
        print(
            f"[target-stats] {slot_name}: count={stats['count']} min={stats['min']:.6f} "
            f"max={stats['max']:.6f} mean={stats['mean']:.6f} std={stats['std']:.6f} "
            f"p95={stats['p95']:.6f} p99={stats['p99']:.6f} max/p99={stats['max_to_p99_ratio']:.3f}"
        )
        for item in top_batches:
            print(
                f"[target-batch] {slot_name} batch={item['batch_index']} size={item['size']} "
                f"min={item['min']:.6f} max={item['max']:.6f} mean={item['mean']:.6f} "
                f"std={item['std']:.6f}"
            )

    summary_path = debug_dir / "fixed_target_analysis.json"
    summary_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"[target-stats] saved analysis to {summary_path}")


def _log_problematic_batch(
    *,
    output_dir: Path,
    epoch: int,
    global_step: int,
    batch_idx: int,
    loss_items: Mapping[str, torch.Tensor],
    preds: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    sample_meta: list[dict[str, Any]],
    grad_norm: float | None,
) -> None:
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    sample_rows: list[dict[str, Any]] = []
    pred_target = preds.get("target")
    pred_tailgas = preds.get("tailgas")
    tgt_target = targets.get("target")
    tgt_tailgas = targets.get("tailgas")
    for i, meta in enumerate(sample_meta):
        row = dict(meta)
        if pred_target is not None and i < pred_target.shape[0]:
            row["pred_target"] = float(pred_target[i].detach().cpu().view(-1)[0])
        if pred_tailgas is not None and i < pred_tailgas.shape[0]:
            row["pred_tailgas"] = float(pred_tailgas[i].detach().cpu().view(-1)[0])
        if tgt_target is not None and i < tgt_target.shape[0]:
            row["used_target"] = float(tgt_target[i].detach().cpu().view(-1)[0])
        if tgt_tailgas is not None and i < tgt_tailgas.shape[0]:
            row["used_tailgas"] = float(tgt_tailgas[i].detach().cpu().view(-1)[0])
        sample_rows.append(row)

    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "batch_index": int(batch_idx),
        "grad_norm": None if grad_norm is None else float(grad_norm),
        "loss_items": {k: float(v.detach().cpu()) for k, v in loss_items.items()},
        "samples": sample_rows,
    }
    log_path = debug_dir / "problem_batches.jsonl"
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
    print(
        f"[debug] problematic batch logged: epoch={epoch} step={global_step} "
        f"batch={batch_idx} file={log_path}"
    )


def _check_edge_all_loss_formula(
    *,
    output_dir: Path,
    epoch: int,
    global_step: int,
    batch_idx: int,
    loss_items: MutableMapping[str, torch.Tensor],
    preds: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    sample_meta: list[dict[str, Any]],
) -> None:
    expected = (
        loss_items.get("weighted_edge_all", loss_items["loss_total"] * 0)
        + loss_items.get("weighted_primary_frac", loss_items["loss_total"] * 0)
        + loss_items.get("weighted_all_edge_r2", loss_items["loss_total"] * 0)
    )
    final = loss_items.get("loss_total_final", loss_items["loss_total"])
    diff = final - expected
    loss_items["loss_formula_expected"] = expected.detach()
    loss_items["loss_formula_diff"] = diff.detach()
    if abs(float(diff.detach().cpu())) <= 1.0e-5:
        return

    loss_items["loss_extra_unexpected"] = diff.detach()
    print(
        "[loss][warn] loss_total_final != weighted edge+pfrac+r2 "
        f"epoch={epoch} step={global_step} batch={batch_idx} "
        f"final={float(final.detach().cpu()):.6f} expected={float(expected.detach().cpu()):.6f} "
        f"diff={float(diff.detach().cpu()):.6f} "
        f"amount_aux={float(loss_items.get('loss_amount_auxiliary', loss_items['loss_total'] * 0).detach().cpu()):.6f} "
        f"disabled_amount_weighted={float(loss_items.get('weighted_amount_auxiliary_disabled', loss_items['loss_total'] * 0).detach().cpu()):.6f}",
        flush=True,
    )
    _log_problematic_batch(
        output_dir=output_dir,
        epoch=epoch,
        global_step=global_step,
        batch_idx=batch_idx,
        loss_items=loss_items,
        preds=preds,
        targets=targets,
        sample_meta=sample_meta,
        grad_norm=None,
    )


def _r2_from_pred_target(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.numel() == 0 or target.numel() == 0:
        return float("nan")
    target_det = target.detach()
    pred_det = pred.detach()
    sse = torch.sum((pred_det - target_det) ** 2)
    target_mean = torch.mean(target_det)
    sst = torch.sum((target_det - target_mean) ** 2)
    sst_val = float(sst.cpu())
    if sst_val <= 1e-12:
        return float("nan")
    return 1.0 - float((sse / sst).cpu())


def _slot_denormalize(
    x: torch.Tensor,
    *,
    slot_name: str,
    mean_map: Mapping[str, float] | None,
    std_map: Mapping[str, float] | None,
    fixed_cfg: Any,
    normalize_targets: bool,
) -> torch.Tensor:
    out = x.detach().clone()
    if normalize_targets:
        mean = float((mean_map or {}).get(slot_name, 0.0))
        std = float((std_map or {}).get(slot_name, 1.0))
        out = out * (std if std != 0 else 1.0) + mean
    if getattr(fixed_cfg, "transform", "none") == "log1p":
        out = torch.expm1(out)
    return out


def _safe_std(x: torch.Tensor) -> float:
    if x.numel() <= 1:
        return 0.0
    return float(x.detach().float().std(unbiased=False).cpu())


def _safe_corrcoef(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() <= 1 or y.numel() <= 1:
        return float("nan")
    x1 = x.detach().view(-1).float()
    y1 = y.detach().view(-1).float()
    x1 = x1 - x1.mean()
    y1 = y1 - y1.mean()
    denom = torch.sqrt((x1.pow(2).sum()) * (y1.pow(2).sum()))
    if float(denom.cpu()) <= 1e-12:
        return float("nan")
    return float((x1 * y1).sum().cpu() / denom.cpu())


def _safe_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() == 0 or y.numel() == 0:
        return float("nan")
    x1 = x.detach().view(-1).float()
    y1 = y.detach().view(-1).float()
    denom = torch.linalg.norm(x1) * torch.linalg.norm(y1)
    if float(denom.cpu()) <= 1e-12:
        return float("nan")
    return float(torch.dot(x1, y1).cpu() / denom.cpu())


def _grad_l2_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total += float(param_norm.item()) ** 2
    return total ** 0.5


def _grad_l2_norm_prefix(model: torch.nn.Module, name_prefix: str) -> float:
    total = 0.0
    for name, p in model.named_parameters():
        if not name.startswith(name_prefix) or p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total += float(param_norm.item()) ** 2
    return total**0.5


def _init_metric_accumulator() -> dict[str, float]:
    return {"sae": 0.0, "sse": 0.0, "sum_y": 0.0, "sum_y2": 0.0, "n": 0.0}


def _accumulate_task_metrics(
    acc: dict[str, float],
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    slot_name: str,
    target_mean: Mapping[str, float] | None,
    target_std: Mapping[str, float] | None,
    data_cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_eval = _slot_denormalize(
        pred,
        slot_name=slot_name,
        mean_map=target_mean,
        std_map=target_std,
        fixed_cfg=data_cfg.fixed_tasks[slot_name],
        normalize_targets=data_cfg.normalize_targets,
    )
    target_eval = _slot_denormalize(
        target,
        slot_name=slot_name,
        mean_map=target_mean,
        std_map=target_std,
        fixed_cfg=data_cfg.fixed_tasks[slot_name],
        normalize_targets=data_cfg.normalize_targets,
    )
    diff = (pred_eval - target_eval).detach().view(-1)
    y = target_eval.detach().view(-1)
    acc["sae"] += float(diff.abs().sum().cpu())
    acc["sse"] += float((diff**2).sum().cpu())
    acc["sum_y"] += float(y.sum().cpu())
    acc["sum_y2"] += float((y**2).sum().cpu())
    acc["n"] += float(y.numel())
    return pred_eval, target_eval


def _finalize_task_metrics(acc: Mapping[str, float]) -> dict[str, float]:
    n = float(acc.get("n", 0.0))
    if n <= 0:
        return {"mae": 0.0, "rmse": 0.0, "r2": float("nan")}
    sae = float(acc.get("sae", 0.0))
    sse = float(acc.get("sse", 0.0))
    sum_y = float(acc.get("sum_y", 0.0))
    sum_y2 = float(acc.get("sum_y2", 0.0))
    mae = sae / n
    rmse = (sse / n) ** 0.5
    sst = sum_y2 - (sum_y * sum_y) / n
    eps = 1e-12
    sst_eff = max(float(sst), eps)
    r2 = 1.0 - (sse / sst_eff)
    return {"mae": mae, "rmse": rmse, "r2": r2}


@torch.no_grad()
def evaluate_epoch(
    model: ProcessSurrogateModel,
    loader: DataLoader,
    device: torch.device,
    train_cfg: Any,
    data_cfg: Any,
    use_amp: bool,
    *,
    target_mean: Mapping[str, float] | None = None,
    target_std: Mapping[str, float] | None = None,
) -> dict[str, float]:
    _was_training_eval_epoch = model.training
    model.eval()
    target_only = getattr(data_cfg, "task_mode", "multitask") == "target_only"
    keys = loss_metric_keys(data_cfg, train_cfg)
    totals = {k: 0.0 for k in keys}
    metric_slots = ("target",) if target_only else ("target", "tailgas")
    metric_accum = {s: _init_metric_accumulator() for s in metric_slots}
    collect_debug = bool(getattr(train_cfg, "debug_mode", False))
    pred_store: dict[str, list[torch.Tensor]] = {s: [] for s in metric_slots}
    true_store: dict[str, list[torch.Tensor]] = {s: [] for s in metric_slots}
    edge_stream_sae = 0.0
    edge_stream_sse = 0.0
    edge_stream_n = 0.0
    steps = 0
    for batch in loader:
        batch_data = {k: v.to(device) for k, v in batch.model_kwargs.items()}
        targets = {k: v.to(device) for k, v in batch.targets.items()}
        target_masks = {k: v.to(device) for k, v in batch.target_masks.items()}
        if getattr(data_cfg, "task_mode", "multitask") == "edge_all":
            _apply_v4_target_loss_weights(batch, target_masks, train_cfg, device)
        task_inputs = {
            head: {k: v.to(device) for k, v in payload.items()}
            for head, payload in batch.task_inputs.items()
        }
        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = model(batch_data, task_inputs=task_inputs)
            loss_items = compute_training_loss(
                preds,
                targets,
                target_masks,
                train_cfg,
                data_cfg,
                **_edge_all_loss_kwargs(batch, data_cfg),
            )
        for k in keys:
            if k in loss_items:
                totals[k] += float(loss_items[k].detach().cpu())
        pred_target = preds.get("target")
        gt_target = targets.get("target")
        if pred_target is not None and gt_target is not None and pred_target.numel() > 0:
            pred_eval, true_eval = _accumulate_task_metrics(
                metric_accum["target"],
                pred=pred_target,
                target=gt_target,
                slot_name="target",
                target_mean=target_mean,
                target_std=target_std,
                data_cfg=data_cfg,
            )
            if collect_debug:
                pred_store["target"].append(pred_eval.detach().cpu())
                true_store["target"].append(true_eval.detach().cpu())
        if not target_only:
            pred_tailgas = preds.get("tailgas")
            gt_tailgas = targets.get("tailgas")
            if pred_tailgas is not None and gt_tailgas is not None and pred_tailgas.numel() > 0:
                pred_eval, true_eval = _accumulate_task_metrics(
                    metric_accum["tailgas"],
                    pred=pred_tailgas,
                    target=gt_tailgas,
                    slot_name="tailgas",
                    target_mean=target_mean,
                    target_std=target_std,
                    data_cfg=data_cfg,
                )
                if collect_debug:
                    pred_store["tailgas"].append(pred_eval.detach().cpu())
                    true_store["tailgas"].append(true_eval.detach().cpu())
        if (
            float(getattr(train_cfg, "edge_stream_loss_weight", 0.0)) > 0.0
            and getattr(data_cfg, "use_canonical_graph_spec_v3", False)
            and preds.get("edge_stream") is not None
            and targets.get("edge_stream") is not None
        ):
            em = target_masks.get("edge_stream")
            if em is not None:
                p_es = preds["edge_stream"]
                t_es = targets["edge_stream"]
                m = em.to(dtype=p_es.dtype)
                while m.ndim < p_es.ndim:
                    m = m.unsqueeze(-1)
                m = m.expand_as(p_es)
                valid = m > 0
                if valid.any():
                    diff = (p_es - t_es)[valid].detach()
                    edge_stream_sae += float(diff.abs().sum().cpu())
                    edge_stream_sse += float((diff**2).sum().cpu())
                    edge_stream_n += float(diff.numel())
        steps += 1
    if steps == 0:
        out = {k: 0.0 for k in keys}
        out.update(
            {
                "metric_target_mae": 0.0,
                "metric_target_rmse": 0.0,
                "metric_target_r2": float("nan"),
            }
        )
        if not target_only:
            out.update(
                {
                    "metric_tailgas_mae": 0.0,
                    "metric_tailgas_rmse": 0.0,
                    "metric_tailgas_r2": float("nan"),
                }
            )
        out["metric_edge_stream_mae"] = 0.0
        out["metric_edge_stream_rmse"] = 0.0
        model.train(_was_training_eval_epoch)
        return out
    out = {k: totals[k] / steps for k in keys}
    for prefix in metric_slots:
        stats = _finalize_task_metrics(metric_accum[prefix])
        out[f"metric_{prefix}_mae"] = stats["mae"]
        out[f"metric_{prefix}_rmse"] = stats["rmse"]
        out[f"metric_{prefix}_r2"] = stats["r2"]
    if collect_debug:
        for prefix in metric_slots:
            if pred_store[prefix]:
                pred_full = torch.cat(pred_store[prefix], dim=0)
                true_full = torch.cat(true_store[prefix], dim=0)
                out[f"debug_{prefix}_pred_mean"] = float(pred_full.mean())
                out[f"debug_{prefix}_pred_std"] = _safe_std(pred_full)
                out[f"debug_{prefix}_true_mean"] = float(true_full.mean())
                out[f"debug_{prefix}_true_std"] = _safe_std(true_full)
                pred_orig = _slot_denormalize(
                    pred_full,
                    slot_name=prefix,
                    mean_map=target_mean,
                    std_map=target_std,
                    fixed_cfg=data_cfg.fixed_tasks[prefix],
                    normalize_targets=data_cfg.normalize_targets,
                )
                true_orig = _slot_denormalize(
                    true_full,
                    slot_name=prefix,
                    mean_map=target_mean,
                    std_map=target_std,
                    fixed_cfg=data_cfg.fixed_tasks[prefix],
                    normalize_targets=data_cfg.normalize_targets,
                )
                out[f"debug_{prefix}_pred_orig_mean"] = float(pred_orig.mean())
                out[f"debug_{prefix}_pred_orig_std"] = _safe_std(pred_orig)
                out[f"debug_{prefix}_true_orig_mean"] = float(true_orig.mean())
                out[f"debug_{prefix}_true_orig_std"] = _safe_std(true_orig)
    if edge_stream_n > 0.0:
        out["metric_edge_stream_mae"] = edge_stream_sae / edge_stream_n
        out["metric_edge_stream_rmse"] = (edge_stream_sse / edge_stream_n) ** 0.5
    else:
        out["metric_edge_stream_mae"] = 0.0
        out["metric_edge_stream_rmse"] = 0.0
    model.train(_was_training_eval_epoch)
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train process surrogate from YAML experiment config.")
    parser.add_argument("--config", type=str, default="configs/experiment/process_surrogate_base.yaml")
    parser.add_argument("--best-metric-name", type=str, default="val/loss")
    parser.add_argument("--best-metric-mode", type=str, choices=("min", "max"), default="min")
    parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping on validation metric.")
    parser.add_argument("--early-stopping-metric", type=str, default="val/loss")
    parser.add_argument("--early-stopping-mode", type=str, choices=("min", "max"), default="min")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-5)
    parser.add_argument(
        "--runtime-overrides-file",
        type=str,
        default="",
        help="Internal JSON file used to pass runtime overrides into a spawned trial process.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Runtime override as dotted KEY=VALUE, e.g. --override train.backward_mode=edge_step.",
    )
    parser.add_argument(
        "--warmstart-best-json",
        type=str,
        default="",
        help=(
            "Optuna ``best_hyperparameters.json`` 등: ``best_params`` 를 train/model 초기값으로 병합. "
            "우선순위는 실험 YAML < 본 옵션 < ``--runtime-overrides-file``."
        ),
    )
    parser.add_argument(
        "--auto-warmstart-best",
        action="store_true",
        help="``--process-filter``>0 일 때 ``outputs/optuna/Process{N}/best_hyperparameters.json`` 등 자동 탐색 후 병합.",
    )
    parser.add_argument(
        "--torch-distributed-debug",
        type=str,
        choices=("off", "info", "detail"),
        default="off",
        help="Set TORCH_DISTRIBUTED_DEBUG for DDP runs.",
    )
    parser.add_argument("--no-training-plots", action="store_true")
    parser.add_argument(
        "--no-save-model-weights",
        action="store_true",
        help="Do not write model checkpoint weights (checkpoints/best.pt or checkpoints/last.pt). Metrics/artifacts still run.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable train.debug_mode (verbose prints + NDJSON session file under output_dir/debug_sessions/).",
    )
    parser.add_argument(
        "--debug-epochs",
        type=int,
        default=None,
        help="Override train.debug_epochs when set.",
    )
    parser.add_argument(
        "--debug-vector-batches",
        type=int,
        default=None,
        help="Override train.debug_vector_batches when set.",
    )
    parser.add_argument(
        "--debug-task-mode",
        type=str,
        choices=("all", "target_only", "tailgas_only"),
        default=None,
        help="Override train.debug_task_mode when set.",
    )
    parser.add_argument(
        "--debug-quick-epochs",
        type=int,
        default=0,
        help="If >0, override train.epochs for a short debug run (e.g. 1 or 2).",
    )
    parser.add_argument(
        "--debug-log-file",
        type=str,
        default="",
        help="Append NDJSON here in addition to logs/debug-8909e4.log (relative paths are under project root).",
    )
    parser.add_argument(
        "--edge-all-forward-debug",
        action="store_true",
        help="task_mode=edge_all: run one train batch (forward + loss), print shapes, exit 0.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=0,
        help="If >0, override train.epochs (short runs / K-fold driver).",
    )
    parser.add_argument(
        "--train-split-manifest",
        type=str,
        default="",
        help="CSV with merged_row_index column; restricts train rows (same as data.train_split_manifest_path).",
    )
    parser.add_argument(
        "--val-split-manifest",
        type=str,
        default="",
        help="CSV with merged_row_index column; restricts val rows.",
    )
    parser.add_argument(
        "--test-split-manifest",
        type=str,
        default="",
        help="CSV with merged_row_index column; restricts test rows.",
    )
    parser.add_argument(
        "--process-filter",
        type=int,
        default=0,
        help="If >0, set data.edge_all_processes to this single process id (edge_all).",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=str,
        default="",
        help="Initialize model weights from a checkpoint without restoring optimizer/scheduler state.",
    )
    parser.add_argument(
        "--pretrained-load-mode",
        type=str,
        choices=("model_only",),
        default="model_only",
        help="Transfer-learning checkpoint load mode. Currently only model_only is supported.",
    )
    parser.add_argument(
        "--pretrained-strict-load",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use strict model_state_dict loading for --pretrained-checkpoint.",
    )
    parser.add_argument(
        "--finetune-mode",
        type=str,
        choices=("full", "head_only"),
        default="full",
        help="full trains all parameters; head_only freezes the encoder and trains prediction heads only.",
    )
    parser.add_argument(
        "--skip-startup-debug",
        action="store_true",
        help=(
            "Skip heavy startup steps: outputs/debug audits (x_oper, target mapping, baselines) "
            "and per-process batch JSON dumps. Training is unchanged; use when startup appears to hang."
        ),
    )
    parser.add_argument(
        "--efficiency-benchmark",
        action="store_true",
        help="Measure repeated best-checkpoint inference on the test loader and write JSON metadata.",
    )
    parser.add_argument(
        "--inference-only-benchmark",
        action="store_true",
        help=(
            "Load --pretrained-checkpoint and benchmark test-loader inference without running "
            "any training epoch. Writes inference_benchmark.json and a one-row CSV."
        ),
    )
    parser.add_argument(
        "--inference-target-edges-only",
        action="store_true",
        help=(
            "For inference benchmarking, keep full-graph encoding but decode only target edges. "
            "Requires edge-stream targets and leaves the default full-stream path unchanged."
        ),
    )
    parser.add_argument("--inference-warmup-runs", type=int, default=5)
    parser.add_argument("--inference-measured-runs", type=int, default=20)
    parser.add_argument(
        "--inference-max-batches",
        type=int,
        default=0,
        help="Optional inference benchmark batch cap; 0 uses the complete test loader.",
    )
    return parser


def _monitor_metric_to_epoch_log_key(experiment: ExperimentConfig) -> tuple[str, str]:
    """Map ``train.monitor_metric`` to epoch_log key and min/max mode."""
    from process_graph.experiment.target_row_primary_metrics import resolve_monitor_epoch_log_key

    task_mode = str(getattr(experiment.data, "task_mode", "multitask"))
    metric = str(getattr(experiment.train, "monitor_metric", "val_loss"))
    mode = str(getattr(experiment.train, "monitor_mode", "min"))
    if metric in {"val_target_r2", "target_r2"}:
        return "val/target_r2", "max"
    if metric in {"val_target_mean_r2", "target_mean_r2"}:
        return "val/target_mean_r2", "max"
    if metric in {
        "val_target_edge_property_mean_r2",
        "target_edge_property_mean_r2",
    }:
        return "val/target_edge_property_mean_r2", "max"
    if metric in {
        "val_target_edge_10d_r2_flatten",
        "target_edge_10d_r2_flatten",
        "flatten_r2",
    }:
        return "val/target_edge_10d_r2_flatten", "max"
    if task_mode == "edge_all" and ("r2" in metric.lower() or metric.startswith("val_eval_primary")):
        key, r2_mode = resolve_monitor_epoch_log_key(metric, task_mode)
        return key, r2_mode
    if metric == "val_r2" or metric.startswith("val_process_balanced") or metric.startswith("val_target_balanced"):
        r2_mode = "max"
        if task_mode == "edge_all":
            return resolve_monitor_epoch_log_key(metric, task_mode)
        return "val/metric_target_r2", r2_mode
    if metric == "val_mae":
        if task_mode == "edge_all":
            return "val/answer_targets_mae", mode
        return "val/metric_target_mae", mode
    if metric == "val_rmse":
        if task_mode == "edge_all":
            return "val/target_h2_rmse", mode
        return "val/metric_target_rmse", mode
    # val_loss
    if task_mode == "edge_all":
        return "val/loss", mode
    return "val/loss", mode


def _apply_train_checkpoint_policy_from_config(
    args: argparse.Namespace,
    experiment: ExperimentConfig,
) -> None:
    """Apply ``process_train.yaml`` early-stopping / monitor_metric to CLI args."""
    metric_key, mode = _monitor_metric_to_epoch_log_key(experiment)
    args.best_metric_name = metric_key
    args.best_metric_mode = mode
    tr = experiment.train
    if bool(tr.early_stopping):
        args.early_stopping = True
        args.early_stopping_metric = metric_key
        args.early_stopping_mode = mode
        args.early_stopping_patience = int(tr.early_stopping_patience)


def _apply_runtime_overrides(experiment: ExperimentConfig, runtime_overrides: Mapping[str, Any]) -> None:
    for key in (
        "experiment_name",
        "seed",
        "device",
        "eval_only",
        "output_dir",
        "save_dir",
        "log_dir",
    ):
        if key in runtime_overrides and hasattr(experiment, key):
            setattr(experiment, key, runtime_overrides[key])

    train_override = runtime_overrides.get("train", {})
    if isinstance(train_override, Mapping):
        for key, value in train_override.items():
            if key == "task_loss_weights" and isinstance(value, Mapping):
                merged = dict(experiment.train.task_loss_weights)
                merged.update({str(k): float(v) for k, v in value.items()})
                experiment.train.task_loss_weights = merged
                continue
            if key == "epoch_sampler" and isinstance(value, Mapping):
                for sampler_key, sampler_value in value.items():
                    if hasattr(experiment.train.epoch_sampler, sampler_key):
                        setattr(
                            experiment.train.epoch_sampler,
                            sampler_key,
                            sampler_value,
                        )
                continue
            if hasattr(experiment.train, key):
                setattr(experiment.train, key, value)

    model_override = runtime_overrides.get("model", {})
    if isinstance(model_override, Mapping):
        for key, value in model_override.items():
            if hasattr(experiment.model, key):
                setattr(experiment.model, key, value)

    data_override = runtime_overrides.get("data", {})
    if isinstance(data_override, Mapping):
        for key, value in data_override.items():
            if hasattr(experiment.data, key):
                setattr(experiment.data, key, value)
    known_feed_cfg = parse_known_feed_condition(
        getattr(experiment.model, "known_feed_condition", None)
    )
    experiment.model.known_feed_condition = known_feed_config_dict(known_feed_cfg)
    experiment.data.known_feed_condition = known_feed_config_dict(known_feed_cfg)


def _metric_improved(current: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    if mode == "min":
        return current < best
    return current > best


def _metric_improved_with_delta(current: float, best: float | None, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "min":
        return current < (best - float(min_delta))
    return current > (best + float(min_delta))


def _nonfinite_train_skip_budget(train_loader: DataLoader, train_cfg: Any) -> int:
    custom = int(getattr(train_cfg, "max_nonfinite_train_batches_per_epoch", 0) or 0)
    if custom > 0:
        return custom
    n = len(train_loader)
    return max(500, 5 * max(n, 1))


def _clone_to_cpu(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _clone_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)


def _snapshot_training_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return {
        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "optimizer": _clone_to_cpu(optimizer.state_dict()),
    }


def _restore_training_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    snapshot: Mapping[str, Any] | None,
) -> bool:
    if not snapshot:
        return False
    model_state = snapshot.get("model")
    optimizer_state = snapshot.get("optimizer")
    if not isinstance(model_state, Mapping) or not isinstance(optimizer_state, Mapping):
        return False
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    return True


def _scale_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    *,
    factor: float,
    min_lr: float,
) -> tuple[list[float], list[float]]:
    old_lrs: list[float] = []
    new_lrs: list[float] = []
    f = max(float(factor), 0.0)
    floor = max(float(min_lr), 0.0)
    for group in optimizer.param_groups:
        old = float(group.get("lr", 0.0))
        new = max(old * f, floor)
        group["lr"] = new
        old_lrs.append(old)
        new_lrs.append(new)
    return old_lrs, new_lrs


def _find_nonfinite_state_names(model: torch.nn.Module, *, limit: int = 8) -> list[str]:
    bad: list[str] = []
    for name, tensor in model.state_dict().items():
        if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
            continue
        if not torch.isfinite(tensor).all():
            bad.append(str(name))
            if len(bad) >= int(limit):
                break
    return bad


def _write_training_run_failed_artifact(
    output_dir: Path,
    exc: BaseException,
    *,
    experiment_path: Path,
    final_epoch_completed: int,
    edge_all: bool,
    batch_size: int,
    learning_rate: float,
) -> None:
    payload: dict[str, Any] = {
        "status": "failed",
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exc(),
        "experiment_yaml": str(experiment_path),
        "final_epoch_completed": int(final_epoch_completed),
        "edge_all": bool(edge_all),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
    }
    if torch.cuda.is_available():
        payload["cuda_oom"] = isinstance(exc, torch.cuda.OutOfMemoryError)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "training_run_failed.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[train] wrote failure artifact: {output_dir / 'training_run_failed.json'}")
    except OSError as oerr:
        print(f"[train] could not write training_run_failed.json: {oerr}")


def _resolve_checkpoint_path(path_str: str) -> Path:
    path = Path(str(path_str).strip())
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"pretrained checkpoint not found: {path}")
    return path


def _target_adapter_missing_prefixes(model: torch.nn.Module) -> tuple[str, ...]:
    edge_decoder = getattr(model, "edge_decoder", None)
    pi_head = getattr(edge_decoder, "pi_head", None)
    if pi_head is None or not bool(getattr(pi_head, "target_branch_hidden_adapter_enabled", False)):
        return ()
    return (
        "edge_decoder.pi_head.target_condition_adapter.",
        "edge_decoder.pi_head.target_fraction_adapter.",
        "edge_decoder.pi_head.target_flow_adapter.",
    )


def _allowed_removed_head_prefixes(model: torch.nn.Module) -> tuple[str, ...]:
    edge_decoder = getattr(model, "edge_decoder", None)
    hierarchical_head = getattr(edge_decoder, "hierarchical_pi_head", None)
    if hierarchical_head is None or bool(
        getattr(hierarchical_head, "predict_volume_flow", True)
    ):
        return ()
    return ("edge_decoder.hierarchical_pi_head.volume_head.",)


def _load_state_dict_allowing_target_adapter_missing(
    model: torch.nn.Module,
    state: Mapping[str, Any],
    *,
    strict: bool,
) -> Any:
    adapter_prefixes = _target_adapter_missing_prefixes(model)
    removed_head_prefixes = _allowed_removed_head_prefixes(model)
    if not bool(strict):
        return model.load_state_dict(state, strict=False)
    if not adapter_prefixes and not removed_head_prefixes:
        try:
            return model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            edge_decoder = getattr(model, "edge_decoder", None)
            if getattr(edge_decoder, "hierarchical_pi_head", None) is not None:
                raise RuntimeError(
                    "Checkpoint is incompatible with hierarchical_reduced_pi. "
                    "This head changes descriptor dimensions and the supervised output schema "
                    "(11D), so legacy checkpoints cannot be partially loaded. Start a fresh run "
                    "or provide a checkpoint created by the exact same dimension architecture."
                ) from exc
            raise
    try:
        return model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        load_result = model.load_state_dict(state, strict=False)
        missing = list(getattr(load_result, "missing_keys", []) or [])
        unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
        bad_missing = [
            key
            for key in missing
            if not any(str(key).startswith(prefix) for prefix in adapter_prefixes)
        ]
        bad_unexpected = [
            key
            for key in unexpected
            if not any(str(key).startswith(prefix) for prefix in removed_head_prefixes)
        ]
        if bad_missing or bad_unexpected:
            raise exc
        return load_result


def _load_model_only_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    strict: bool,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"checkpoint payload must be a mapping: {checkpoint_path}")
    state = payload.get("model_state_dict")
    if state is None:
        state = payload.get("model")
    if not isinstance(state, Mapping):
        raise KeyError(
            f"checkpoint has no model_state_dict/model mapping for model-only load: {checkpoint_path}"
        )
    operating_weight_suffix = "encoder.input_encoder.operating_encoder.0.weight"
    checkpoint_operating_weight = next(
        (
            value
            for key, value in state.items()
            if str(key).endswith(operating_weight_suffix)
            and isinstance(value, torch.Tensor)
        ),
        None,
    )
    current_state = model.state_dict()
    current_operating_weight = next(
        (
            value
            for key, value in current_state.items()
            if str(key).endswith(operating_weight_suffix)
        ),
        None,
    )
    if (
        checkpoint_operating_weight is not None
        and current_operating_weight is not None
        and tuple(checkpoint_operating_weight.shape)
        != tuple(current_operating_weight.shape)
    ):
        raise RuntimeError(
            "Operating encoder checkpoint shape mismatch: "
            f"checkpoint={tuple(checkpoint_operating_weight.shape)} "
            f"current={tuple(current_operating_weight.shape)}. F0 uses 13 values + "
            "13 masks (26 inputs), while F1 uses 16 values + 16 masks (32 inputs). "
            "Clean F0/F1 ablations must train from scratch; automatic partial loading "
            "is intentionally disabled."
        )
    load_result = _load_state_dict_allowing_target_adapter_missing(
        model,
        state,
        strict=bool(strict),
    )
    missing = list(getattr(load_result, "missing_keys", []) or [])
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    return {
        "checkpoint_path": str(checkpoint_path),
        "strict": bool(strict),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "checkpoint_epoch": payload.get("epoch", None),
        "checkpoint_best_metric": payload.get("best_metric", None),
        "checkpoint_config_path": payload.get("config_path", None),
        "loaded_tensor_count": len(state),
    }


def _set_finetune_mode(model: ProcessSurrogateModel, mode: str) -> dict[str, Any]:
    mode = str(mode or "full").strip().lower()
    if mode not in {"full", "head_only"}:
        raise ValueError(f"unknown finetune mode: {mode}")
    for param in model.parameters():
        param.requires_grad = True
    trainable_modules: list[str] = []
    if mode == "head_only":
        for param in model.encoder.parameters():
            param.requires_grad = False
        if model.edge_decoder is not None:
            for param in model.edge_decoder.parameters():
                param.requires_grad = True
            trainable_modules.append("edge_decoder")
        if model.edge_stream_head is not None:
            for param in model.edge_stream_head.parameters():
                param.requires_grad = True
            trainable_modules.append("edge_stream_head")
        if len(model.heads) > 0:
            for param in model.heads.parameters():
                param.requires_grad = True
            trainable_modules.extend([f"heads.{name}" for name in model.heads.keys()])
        if not trainable_modules:
            raise RuntimeError("finetune-mode=head_only found no trainable prediction head modules.")
    else:
        trainable_modules.append("all")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return {
        "finetune_mode": mode,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "frozen_params": int(frozen_params),
        "trainable_modules": trainable_modules,
    }


def _load_runtime_overrides_file(path_str: str) -> dict[str, Any]:
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"runtime overrides file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"runtime overrides file must contain a JSON object: {path}")
    return payload


def train_one_run(
    args: argparse.Namespace,
    *,
    runtime_overrides: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    if args.torch_distributed_debug != "off" and "TORCH_DISTRIBUTED_DEBUG" not in os.environ:
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = args.torch_distributed_debug.upper()
    use_ddp = "LOCAL_RANK" in os.environ
    local_rank = 0
    if use_ddp:
        local_rank = setup_ddp()

    experiment_path = Path(args.config)
    if not experiment_path.is_absolute():
        experiment_path = (PROJECT_ROOT / experiment_path).resolve()

    experiment = load_experiment_config(experiment_path)
    file_overrides = _normalize_runtime_override_aliases(_load_runtime_overrides_file(args.runtime_overrides_file))
    cli_overrides = _normalize_runtime_override_aliases(_parse_cli_overrides(getattr(args, "override", [])))
    warm_nested: dict[str, Any] = {}
    ws_arg = (getattr(args, "warmstart_best_json", "") or "").strip()
    warm_path: Path | None = None
    if ws_arg:
        wp = Path(ws_arg)
        warm_path = wp if wp.is_absolute() else (PROJECT_ROOT / wp).resolve()
        wflat = load_best_params_from_json(warm_path)
    elif bool(getattr(args, "auto_warmstart_best", False)) and int(getattr(args, "process_filter", 0) or 0) > 0:
        warm_path = discover_best_hyperparameters_json(
            project_root=PROJECT_ROOT, process_filter=int(args.process_filter)
        )
        wflat = load_best_params_from_json(warm_path) if warm_path is not None else {}
    else:
        wflat = {}
    if wflat:
        preset = "edge_all" if str(getattr(experiment.data, "task_mode", "")) == "edge_all" else "multitask"
        warm_nested = flat_optuna_params_to_runtime_overrides(wflat, preset=preset)
        if _is_main() and warm_path is not None:
            print(f"[warmstart] merged {len(wflat)} flat params from {warm_path}")
    merged_runtime_overrides = _merge_overrides(
        warm_nested,
        _merge_overrides(file_overrides, _merge_overrides(_normalize_runtime_override_aliases(runtime_overrides or {}), cli_overrides)),
    )
    if merged_runtime_overrides:
        _apply_runtime_overrides(experiment, merged_runtime_overrides)
    if getattr(args, "max_epochs", 0) and int(args.max_epochs) > 0:
        experiment.train.epochs = int(args.max_epochs)
    if getattr(args, "process_filter", 0) and int(args.process_filter) > 0:
        experiment.data.edge_all_processes = [int(args.process_filter)]
    _apply_train_checkpoint_policy_from_config(args, experiment)
    cli_tr = getattr(args, "train_split_manifest", "") or ""
    cli_va = getattr(args, "val_split_manifest", "") or ""
    cli_te = getattr(args, "test_split_manifest", "") or ""
    if cli_tr:
        experiment.data.train_split_manifest_path = cli_tr
    if cli_va:
        experiment.data.val_split_manifest_path = cli_va
    if cli_te:
        experiment.data.test_split_manifest_path = cli_te
    _apply_cli_debug_to_experiment(args, experiment)
    _configure_debug_session_ndjson(args, experiment)
    set_seed(experiment.seed)
    device = pick_device(experiment.device, local_rank)
    ddp_find_unused_parameters = True

    edge_all = getattr(experiment.data, "task_mode", "multitask") == "edge_all"
    if edge_all:
        if not getattr(experiment.data, "use_canonical_graph_spec_v3", False):
            raise RuntimeError("task_mode=edge_all requires data.use_canonical_graph_spec_v3=True.")
        if getattr(experiment.data, "topology_mode", "") != "stream_edge":
            raise RuntimeError("task_mode=edge_all requires data.topology_mode='stream_edge'.")
        if not getattr(experiment.model, "use_edge_decoder", False):
            raise RuntimeError("task_mode=edge_all requires model.use_edge_decoder=True.")
        if _is_main():
            from process_graph.experiment.target_v4_provenance import resolve_provenance_path

            prov = resolve_provenance_path()
            if not _terminal_log_is_quiet(experiment.train):
                print(
                    edge_all_supervision_banner(
                        experiment.train,
                        provenance_loaded=prov is not None,
                        verbosity=_terminal_log_verbosity(experiment.train),
                    ),
                    flush=True,
                )
            from process_graph.experiment.metric_policy import print_metric_loss_policy

            print_metric_loss_policy(
                experiment.train,
                verbosity=_terminal_log_verbosity(experiment.train),
                stream_target_dim=int(getattr(experiment.model, "stream_target_dim", 12)),
            )

    # Joint multi-process edge_all: raise default capacity when training one model on many processes.
    # Skip when (a) --process-filter > 0, or (b) data.edge_all_processes lists exactly one id (K-fold / narrow YAML).
    _pf = int(getattr(args, "process_filter", 0) or 0)
    _plist = getattr(experiment.data, "edge_all_processes", None)
    _single_process_edge_all = _pf > 0 or (
        isinstance(_plist, (list, tuple)) and len(_plist) == 1
    )
    if edge_all and not _single_process_edge_all:
        hd_floor, ed_floor = 384, 384
        explicit_dimension_design = bool(
            getattr(experiment.model, "dimension_design", {}) or {}
        )
        old_h = int(experiment.model.hidden_dim)
        if old_h < hd_floor and not explicit_dimension_design:
            experiment.model.hidden_dim = hd_floor
            if _is_main():
                print(f"[model] all-process edge_all: hidden_dim {old_h} -> {hd_floor}")
        old_ed = int(getattr(experiment.model, "edge_decoder_hidden_dim", 0) or 0)
        explicit_hierarchical_head = (
            str(getattr(experiment.model, "edge_head_type", "")).strip().lower()
            == "hierarchical_reduced_pi"
        )
        if old_ed < ed_floor and not explicit_hierarchical_head:
            experiment.model.edge_decoder_hidden_dim = ed_floor
            if _is_main():
                print(f"[model] all-process edge_all: edge_decoder_hidden_dim {old_ed} -> {ed_floor}")

    encoder_cfg = model_yaml_to_encoder_config(experiment.model, experiment.data)
    task_specs = build_task_specs(experiment.model, experiment.data)
    model = ProcessSurrogateModel(encoder_config=encoder_cfg, task_specs=task_specs).to(device)
    if use_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=ddp_find_unused_parameters,
        )
    raw_model = model.module if use_ddp else model
    transfer_init_metadata: dict[str, Any] = {
        "pretrained_checkpoint": "",
        "pretrained_load_mode": str(getattr(args, "pretrained_load_mode", "model_only")),
        "pretrained_strict_load": bool(getattr(args, "pretrained_strict_load", True)),
        "finetune_mode": str(getattr(args, "finetune_mode", "full")),
    }
    pretrained_arg = str(getattr(args, "pretrained_checkpoint", "") or "").strip()
    if pretrained_arg:
        if str(getattr(args, "pretrained_load_mode", "model_only")) != "model_only":
            raise ValueError("--pretrained-load-mode currently supports only model_only.")
        pretrained_path = _resolve_checkpoint_path(pretrained_arg)
        if edge_all:
            checkpoint_payload = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            if not isinstance(checkpoint_payload, Mapping):
                raise TypeError(f"checkpoint payload must be a mapping: {pretrained_path}")
            assert_checkpoint_pi_mass_flow_compatible(
                checkpoint_payload,
                train_cfg=experiment.train,
                checkpoint_path=pretrained_path,
            )
        transfer_init_metadata.update(
            _load_model_only_checkpoint(
                raw_model,
                pretrained_path,
                strict=bool(getattr(args, "pretrained_strict_load", True)),
            )
        )
        if _is_main():
            print(
                "[transfer] loaded pretrained model weights "
                f"path={pretrained_path} strict={transfer_init_metadata['strict']} "
                f"missing={len(transfer_init_metadata.get('missing_keys', []))} "
                f"unexpected={len(transfer_init_metadata.get('unexpected_keys', []))}",
                flush=True,
            )
    freeze_metadata = _set_finetune_mode(raw_model, str(getattr(args, "finetune_mode", "full")))
    transfer_init_metadata.update(freeze_metadata)
    if _is_main():
        print(
            "[transfer] finetune_mode="
            f"{freeze_metadata['finetune_mode']} trainable_params={freeze_metadata['trainable_params']} "
            f"frozen_params={freeze_metadata['frozen_params']} "
            f"trainable_modules={','.join(freeze_metadata['trainable_modules'])}",
            flush=True,
        )
    if edge_all and _is_main():
        _print_edge_head_summary(raw_model, train_cfg=experiment.train)
    backward_mode = str(getattr(experiment.train, "backward_mode", "batch_total") or "batch_total").strip()
    if backward_mode not in {"batch_total", "edge_step", "edge_step_pi", "sample_hybrid_target_edge_step_pi"}:
        raise ValueError(f"Unknown backward_mode: {backward_mode}")
    if edge_all and backward_mode == "sample_hybrid_target_edge_step_pi":
        if use_ddp:
            raise RuntimeError(
                "sample_hybrid_target_edge_step_pi currently supports single-GPU/single-process training only."
            )
        if int(getattr(experiment.train, "batch_size", 1) or 1) != 1:
            raise ValueError(
                "sample_hybrid_target_edge_step_pi requires batch_size=1, "
                f"got {getattr(experiment.train, 'batch_size', None)}."
            )
        if int(getattr(experiment.train, "gradient_accumulation_steps", 1) or 1) != 1:
            raise ValueError(
                "sample_hybrid_target_edge_step_pi requires gradient_accumulation_steps=1, "
                f"got {getattr(experiment.train, 'gradient_accumulation_steps', None)}."
            )
    if edge_all and backward_mode in {"edge_step", "edge_step_pi", "sample_hybrid_target_edge_step_pi"}:
        max_optimizer_steps = int(getattr(experiment.train, "max_optimizer_steps", 0) or 0)
        if max_optimizer_steps > 0 and backward_mode != "sample_hybrid_target_edge_step_pi":
            raise ValueError(
                "train.max_optimizer_steps requires "
                "backward_mode=sample_hybrid_target_edge_step_pi so every internal "
                "optimizer.step boundary can be budgeted exactly."
            )
        if int(getattr(experiment.train, "gradient_accumulation_steps", 1) or 1) != 1:
            if _is_main():
                print(
                    f"[WARN] {backward_mode} mode disables gradient accumulation; forcing gradient_accumulation_steps=1.",
                    flush=True,
                )
            experiment.train.gradient_accumulation_steps = 1
        if backward_mode == "edge_step" and bool(getattr(experiment.train, "use_all_edge_r2_loss_in_edge_step", False)):
            if _is_main():
                print("[WARN] edge_step ignores all-edge R2 auxiliary loss; forcing it off.", flush=True)
            experiment.train.use_all_edge_r2_loss_in_edge_step = False
        if backward_mode in {"edge_step_pi", "sample_hybrid_target_edge_step_pi"}:
            for attr in (
                "use_total_loss_in_edge_step_pi",
                "use_all_edge_r2_loss_in_edge_step_pi",
                "use_primary_frac_loss_in_edge_step_pi",
            ):
                if bool(getattr(experiment.train, attr, False)):
                    if _is_main():
                        print(f"[WARN] edge_step_pi ignores {attr}; forcing it off.", flush=True)
                    setattr(experiment.train, attr, False)
        if _is_main() and not _terminal_log_is_quiet(experiment.train):
            if _terminal_log_is_verbose(experiment.train):
                print(
                    f"[WARN] {backward_mode} mode performs optimizer.step per valid edge. "
                    "Total optimizer updates per epoch may be much larger than batch_total.",
                    flush=True,
                )
            else:
                print(
                    f"[train-mode] {backward_mode}: non-target mean step + per-target-edge steps + node step",
                    flush=True,
                )
            if backward_mode in {"edge_step_pi", "sample_hybrid_target_edge_step_pi"}:
                if _terminal_log_is_verbose(experiment.train):
                    print(
                        f"[edge_step_pi] scheduler_step_unit={getattr(experiment.train, 'scheduler_step_unit', 'epoch')} "
                        f"shuffle_edges_each_batch={getattr(experiment.train, 'shuffle_edges_each_batch', True)} "
                        f"edge_weight_default={getattr(experiment.train, 'edge_weight_default', 1.0)} "
                        f"edge_weight_target_edge={getattr(experiment.train, 'edge_weight_target_edge', 5.0)} "
                        f"normalize_y_edge={getattr(experiment.data, 'normalize_y_edge', False)} "
                        f"pi_main_loss_type={getattr(experiment.train, 'pi_main_loss_type', '') or getattr(experiment.train, 'loss_type_edge_all', 'smooth_l1')} "
                        f"lambda_main={getattr(experiment.train, 'lambda_main', 1.0)} "
                        f"lambda_rho={getattr(experiment.train, 'lambda_rho', 0.01)} "
                        f"lambda_h={getattr(experiment.train, 'lambda_h', 0.01)} "
                        f"lambda_volume={getattr(experiment.train, 'lambda_volume', 0.0)} "
                        f"lambda_enthalpy_flow={getattr(experiment.train, 'lambda_enthalpy_flow', 0.0)} "
                        f"lambda_atom={getattr(experiment.train, 'lambda_atom', 0.0)} "
                        f"lambda_energy={getattr(experiment.train, 'lambda_energy', 0.0)} "
                        f"lambda_node_mass={getattr(experiment.train, 'lambda_node_mass', 0.0)} "
                        f"lambda_node_component={getattr(experiment.train, 'lambda_node_component', 0.0)} "
                        f"lambda_node_atom={getattr(experiment.train, 'lambda_node_atom', None)} "
                        f"lambda_node_energy={getattr(experiment.train, 'lambda_node_energy', None)} "
                        f"mw_unit_scale={getattr(experiment.train, 'mw_unit_scale', None)} "
                        f"volume_unit_scale={getattr(experiment.train, 'volume_unit_scale', 1.0)}",
                        flush=True,
                    )
                else:
                    print(
                        "[edge_step_pi] "
                        f"target_weight={getattr(experiment.train, 'edge_weight_target_edge', 5.0)} "
                        f"loss={getattr(experiment.train, 'pi_main_loss_type', '') or getattr(experiment.train, 'loss_type_edge_all', 'smooth_l1')} "
                        f"pinn(node_mass={getattr(experiment.train, 'lambda_node_mass', 0.0)}, "
                        f"node_component={getattr(experiment.train, 'lambda_node_component', 0.0)}, "
                        f"node_atom={getattr(experiment.train, 'lambda_node_atom', None)}, "
                        f"node_energy={getattr(experiment.train, 'lambda_node_energy', None)}, "
                        f"rho={getattr(experiment.train, 'lambda_rho', 0.01)}, "
                        f"h={getattr(experiment.train, 'lambda_h', 0.01)}, "
                        f"volume={getattr(experiment.train, 'lambda_volume', 0.0)})",
                        flush=True,
                    )
                mass_aux_cfg = getattr(experiment.train, "mass_flow_physical_auxiliary", None)
                if bool(getattr(mass_aux_cfg, "enabled", False)):
                    print(
                        "[mass-flow-physical-aux] "
                        f"log_weight={float(getattr(mass_aux_cfg, 'log_weight', 1.0))} "
                        f"max_weight={float(getattr(mass_aux_cfg, 'weight', 0.10))} "
                        f"scale={getattr(mass_aux_cfg, 'scale_method', 'train_std')} "
                        f"delta={float(getattr(mass_aux_cfg, 'delta', 1.0))} "
                        f"clip={getattr(mass_aux_cfg, 'residual_clip', 10.0)} "
                        f"schedule={getattr(mass_aux_cfg, 'schedule_type', 'linear_warmup')} "
                        f"epochs={getattr(mass_aux_cfg, 'schedule_start_epoch', 3)}"
                        f"..{getattr(mass_aux_cfg, 'schedule_end_epoch', 7)} (1-based)",
                        flush=True,
                    )
                    node_opt_cfg = getattr(experiment.train, "node_pinn_optimization", None)
                    if str(getattr(node_opt_cfg, "update_mode", "separate")) == "separate":
                        print(
                            "[WARN] Mass_Flow physical auxiliary is part of supervised edge loss, "
                            "while Node PINN still uses a separate optimizer step.",
                            flush=True,
                        )
                if backward_mode == "sample_hybrid_target_edge_step_pi":
                    hybrid_cfg = getattr(experiment.train, "sample_hybrid_target_edge_step_pi", None)
                    if _terminal_log_is_verbose(experiment.train):
                        print(
                            "[sample_hybrid_target_edge_step_pi] "
                            f"batch_size={getattr(experiment.train, 'batch_size', None)} "
                            f"non_target_reduction={getattr(hybrid_cfg, 'non_target_reduction', 'edge_group_macro_mean')} "
                            f"target_update_order={getattr(hybrid_cfg, 'target_update_order', 'canonical_edge_id')} "
                            f"use_existing_target_edge_weight={getattr(hybrid_cfg, 'use_existing_target_edge_weight', False)} "
                            f"require_batch_size_one={getattr(hybrid_cfg, 'require_batch_size_one', True)} "
                            f"allow_duplicate_canonical_edge_rows={getattr(hybrid_cfg, 'allow_duplicate_canonical_edge_rows', True)}",
                            flush=True,
                        )
                    node_opt_cfg = getattr(experiment.train, "node_pinn_optimization", None)
                    if node_opt_cfg is not None:
                        print(
                            "[node-pinn-optimization] "
                            f"mode={getattr(node_opt_cfg, 'update_mode', 'separate')} "
                            f"anchor_weight={float(getattr(node_opt_cfg, 'supervised_anchor_weight', 1.0))} "
                            f"node_outer_weight={float(getattr(node_opt_cfg, 'node_outer_weight', 1.0))} "
                            f"anchor_apply_target_weight="
                            f"{bool(getattr(node_opt_cfg, 'anchor_apply_target_weight', False))} "
                            f"gradient_diagnostics="
                            f"{bool(getattr(node_opt_cfg, 'log_gradient_diagnostics', False))}",
                            flush=True,
                        )
                    else:
                        print(
                            "[sample_hybrid] "
                            f"batch_size={getattr(experiment.train, 'batch_size', None)} "
                            f"non_target={getattr(hybrid_cfg, 'non_target_reduction', 'edge_group_macro_mean')} "
                            f"target_order={getattr(hybrid_cfg, 'target_update_order', 'canonical_edge_id')}",
                            flush=True,
                        )
            else:
                print(
                    f"[edge_step] scheduler_step_unit={getattr(experiment.train, 'scheduler_step_unit', 'epoch')} "
                    f"shuffle_edges_each_batch={getattr(experiment.train, 'shuffle_edges_each_batch', True)} "
                    f"edge_weight_default={getattr(experiment.train, 'edge_weight_default', 1.0)} "
                    f"edge_weight_target_edge={getattr(experiment.train, 'edge_weight_target_edge', 5.0)}",
                    flush=True,
                )
        has_bn = any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in raw_model.modules())
        if has_bn and _is_main():
            print(
                f"[WARN] BatchNorm detected under {backward_mode} mode. Repeated forward passes on the same batch "
                "may over-update running statistics.",
                flush=True,
            )

    train_csv = (experiment.project_root / experiment.data.train_data_path).resolve()
    if not train_csv.is_file():
        raise FileNotFoundError(f"Training CSV not found at {train_csv}.")

    train_manifest_rel = str(getattr(experiment.data, "train_split_manifest_path", "") or "").strip()
    val_manifest_rel = str(getattr(experiment.data, "val_split_manifest_path", "") or "").strip()
    test_manifest_rel = str(getattr(experiment.data, "test_split_manifest_path", "") or "").strip()

    train_split = resolve_split_filter(train_csv, experiment.data.split_column, "train")
    train_allowed_idx: set[int] | None = None
    if train_manifest_rel:
        train_allowed_idx = _read_split_manifest_indices(experiment.project_root, train_manifest_rel)
        train_split = None
    fixed_target_frame = _prepare_fixed_target_frame(
        train_csv,
        experiment.data.process_id_column,
        experiment.data.split_column,
        train_split,
        experiment.data.fixed_tasks,
        allowed_row_indices=train_allowed_idx,
    )
    _clip_tasks = experiment.data.fixed_tasks
    if getattr(experiment.data, "task_mode", "multitask") == "target_only":
        _clip_tasks = {"target": experiment.data.fixed_tasks["target"]}
    fixed_slot_clip_max = _compute_fixed_slot_clip_maxes(
        fixed_target_frame,
        process_id_column=experiment.data.process_id_column,
        fixed_tasks=_clip_tasks,
    )

    stats_cfg = replace(experiment.data, normalize_x_oper=False, normalize_targets=False)
    train_manifest_path = (
        (experiment.project_root / train_manifest_rel).resolve() if train_manifest_rel else None
    )
    train_fit_provenance = _manifest_fit_provenance(train_manifest_path)
    run_name = f"{experiment.experiment_name}-{datetime.now():%Y%m%d-%H%M%S}"
    debug_run_id = f"{run_name}-rank{_rank()}"
    run_tag = run_name
    output_dir = (experiment.project_root / experiment.output_dir / run_tag).resolve()
    ckpt_dir = (experiment.project_root / experiment.save_dir / run_tag).resolve()
    base_train_ds = ProcessGraphTabularDataset(
        experiment.data.train_data_path,
        stats_cfg,
        experiment.project_root,
        split_filter=train_split,
        split_manifest=train_manifest_path,
        fixed_slot_clip_max=fixed_slot_clip_max,
    )
    oper_mean = oper_std = None
    oper_feature_names = list(
        operating_feature_names(getattr(experiment.model, "known_feed_condition", None))
    )
    target_mean = target_std = None
    if experiment.data.normalize_x_oper or experiment.data.normalize_targets:
        stats_len = len(base_train_ds)
        if getattr(args, "edge_all_forward_debug", False):
            stats_len = min(stats_len, 512)
        oper_cache_path = _oper_normalizer_cache_path(
            experiment=experiment,
            train_csv=train_csv,
            train_manifest_path=train_manifest_path,
            dataset_len=stats_len,
            stream_target_dim=int(getattr(encoder_cfg, "stream_target_dim", 12)),
        )
        legacy_oper_cache_paths = []
        for legacy_dim in {
            int(getattr(encoder_cfg, "stream_target_dim", 12)),
            10,
            11,
            12,
            14,
        }:
            legacy_path = _legacy_oper_normalizer_cache_path(
                experiment=experiment,
                train_csv=train_csv,
                train_manifest_path=train_manifest_path,
                dataset_len=stats_len,
                stream_target_dim=legacy_dim,
            )
            if legacy_path != oper_cache_path and legacy_path not in legacy_oper_cache_paths:
                legacy_oper_cache_paths.append(legacy_path)
        oper_cache: dict[str, Any] | None = None
        if experiment.data.normalize_x_oper:
            oper_cache_read_path = next(
                (path for path in [oper_cache_path, *legacy_oper_cache_paths] if path.is_file()),
                oper_cache_path,
            )
            if oper_cache_read_path.is_file():
                try:
                    loaded = torch.load(oper_cache_read_path, map_location="cpu", weights_only=False)
                    loaded_mean = (
                        torch.as_tensor(loaded.get("mean"), dtype=torch.float32)
                        if isinstance(loaded, dict) and loaded.get("mean") is not None
                        else None
                    )
                    loaded_columns = (
                        [str(x) for x in loaded.get("columns", [])]
                        if isinstance(loaded, dict)
                        else []
                    )
                    compatible_columns = not loaded_columns or loaded_columns == oper_feature_names
                    if (
                        isinstance(loaded, dict)
                        and int(loaded.get("dataset_len", -1)) == stats_len
                        and loaded_mean is not None
                        and int(loaded_mean.numel()) == len(oper_feature_names)
                        and compatible_columns
                    ):
                        oper_cache = loaded
                        oper_mean = loaded_mean
                        oper_std = torch.as_tensor(loaded["std"], dtype=torch.float32)
                        print(
                            f"[oper-normalizer-cache] hit; skipped {stats_len} graph builds: {oper_cache_read_path}",
                            flush=True,
                        )
                        if oper_cache_read_path != oper_cache_path:
                            oper_cache_path.parent.mkdir(parents=True, exist_ok=True)
                            tmp_cache_path = oper_cache_path.with_suffix(".tmp")
                            torch.save(loaded, tmp_cache_path)
                            os.replace(tmp_cache_path, oper_cache_path)
                            print(
                                f"[oper-normalizer-cache] migrated legacy cache to stable key: {oper_cache_path}",
                                flush=True,
                            )
                except Exception as exc:
                    print(f"[oper-normalizer-cache][warn] ignored invalid cache: {exc}", flush=True)
        if experiment.data.normalize_x_oper and oper_cache is None:
            print(
                f"[oper-normalizer-cache] miss; fitting on {stats_len} train samples once: "
                f"{oper_cache_path}",
                flush=True,
            )
            t0 = time.perf_counter()
            try:
                oper_mean, oper_std = compute_oper_normalizer_from_x_oper(
                    base_train_ds.iter_x_oper_for_normalizer(limit=stats_len)
                )
                print(
                    f"[oper-normalizer-cache] fast stats path completed in {time.perf_counter() - t0:.2f}s",
                    flush=True,
                )
            except NotImplementedError:
                oper_mean, oper_std = compute_oper_normalizer(base_train_ds[idx] for idx in range(stats_len))
                print(
                    f"[oper-normalizer-cache] graph-build fallback completed in {time.perf_counter() - t0:.2f}s",
                    flush=True,
                )
            oper_cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_cache_path = oper_cache_path.with_suffix(".tmp")
            torch.save(
                {
                    "mean": oper_mean.cpu(),
                    "std": oper_std.cpu(),
                    "columns": oper_feature_names,
                    "known_feed_condition": known_feed_config_dict(
                        parse_known_feed_condition(
                            getattr(experiment.model, "known_feed_condition", None)
                        )
                    ),
                    "dataset_len": stats_len,
                },
                tmp_cache_path,
            )
            os.replace(tmp_cache_path, oper_cache_path)
            print(f"[oper-normalizer-cache] wrote {oper_cache_path}", flush=True)
        if experiment.data.normalize_x_oper and _is_main():
            output_dir.mkdir(parents=True, exist_ok=True)
            oper_metadata = {
                "columns": oper_feature_names,
                "mean": oper_mean.cpu(),
                "std": oper_std.cpu(),
                "train_split_only": True,
                "fit_provenance": train_fit_provenance,
                "known_feed_condition": known_feed_config_dict(
                    parse_known_feed_condition(
                        getattr(experiment.model, "known_feed_condition", None)
                    )
                ),
            }
            torch.save(oper_metadata, output_dir / "oper_normalizer.pt")
            (output_dir / "oper_normalizer_metadata.json").write_text(
                json.dumps(
                    {
                        **oper_metadata,
                        "mean": oper_mean.detach().cpu().tolist(),
                        "std": oper_std.detach().cpu().tolist(),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        if experiment.data.normalize_targets:
            stats_source = [base_train_ds[idx] for idx in range(stats_len)]
            if decoder_tasks_enabled(experiment.data):
                enabled_cats = [name for name, cfg in experiment.data.decoder_tasks.items() if cfg.enabled]
                target_mean, target_std = compute_category_target_normalizer(stats_source, enabled_cats)
            elif experiment.data.aux_task.enabled:
                target_mean, target_std = compute_target_normalizer(stats_source, ["target", "tailgas", "aux"])
            else:
                _norm_slots = (
                    ["target"]
                    if getattr(experiment.data, "task_mode", "multitask") == "target_only"
                    else ["target", "tailgas"]
                )
                target_mean, target_std = compute_target_normalizer(stats_source, _norm_slots)

    train_dataset = ProcessGraphTabularDataset(
        experiment.data.train_data_path,
        experiment.data,
        experiment.project_root,
        split_filter=train_split,
        split_manifest=train_manifest_path,
        oper_mean=oper_mean,
        oper_std=oper_std,
        target_mean=target_mean,
        target_std=target_std,
        fixed_slot_clip_max=fixed_slot_clip_max,
    )
    if experiment.train.tiny_overfit_samples > 0:
        tiny_n = min(int(experiment.train.tiny_overfit_samples), len(train_dataset))
        tiny_indices = list(range(tiny_n))
        train_dataset = Subset(train_dataset, tiny_indices)
    if edge_all:
        cap_tr = getattr(experiment.data, "edge_all_cap_train_samples", None)
        if cap_tr is not None and int(cap_tr) > 0:
            cap_n = min(int(cap_tr), len(train_dataset))
            if cap_n > 0:
                train_dataset = Subset(train_dataset, list(range(cap_n)))
    train_scaler_dataset = train_dataset
    rare_positive_summary: dict[str, Any] | None = None
    rare_target_edge_augmentation_summary: dict[str, Any] | None = None
    balanced_target_edge_augmentation_summary: dict[str, Any] | None = None
    rare_sampler_cfg = getattr(experiment.train, "rare_positive_sampler", None)
    rare_sampler_enabled = bool(
        rare_sampler_cfg is not None and getattr(rare_sampler_cfg, "enabled", False)
    )
    epoch_sampler_cfg = getattr(experiment.train, "epoch_sampler", None)
    epoch_sampler_enabled = bool(
        epoch_sampler_cfg is not None and getattr(epoch_sampler_cfg, "enabled", False)
    )
    rare_target_edge_augmentation_cfg = getattr(
        experiment.train,
        "rare_target_edge_augmentation",
        None,
    )
    rare_target_edge_augmentation_enabled = bool(
        rare_target_edge_augmentation_cfg is not None
        and getattr(rare_target_edge_augmentation_cfg, "enabled", False)
    )
    balanced_target_edge_augmentation_cfg = getattr(
        experiment.train,
        "target_edge_augmentation",
        None,
    )
    balanced_target_edge_augmentation_enabled = bool(
        balanced_target_edge_augmentation_cfg is not None
        and getattr(balanced_target_edge_augmentation_cfg, "enabled", False)
    )
    augmentation_mode_count = sum(
        (
            rare_sampler_enabled,
            epoch_sampler_enabled,
            rare_target_edge_augmentation_enabled,
            balanced_target_edge_augmentation_enabled,
        )
    )
    if augmentation_mode_count > 1:
        raise RuntimeError(
            "rare_positive_sampler, epoch_sampler, rare_target_edge_augmentation, and "
            "target_edge_augmentation are mutually exclusive. Enable only one sampling/augmentation mode."
        )
    if rare_target_edge_augmentation_enabled:
        if use_ddp:
            raise RuntimeError(
                "rare_target_edge_augmentation.enabled=true is not compatible with DDP. "
                "Run single-process training or disable the augmentation."
            )
        augmentation_build = build_rare_target_edge_augmented_dataset(
            train_dataset,
            augmentation_cfg=rare_target_edge_augmentation_cfg,
            project_root=experiment.project_root,
        )
        train_dataset = augmentation_build.dataset
        rare_target_edge_augmentation_summary = dict(augmentation_build.summary)
        print(
            "[rare-target-edge-augmentation] "
            + json.dumps(rare_target_edge_augmentation_summary, sort_keys=True),
            flush=True,
        )
    if balanced_target_edge_augmentation_enabled:
        if use_ddp:
            raise RuntimeError(
                "target_edge_augmentation.enabled=true is not compatible with DDP. "
                "Run single-process training or disable the augmentation."
            )
        augmentation_build = build_balanced_target_edge_augmented_dataset(
            train_dataset,
            augmentation_cfg=balanced_target_edge_augmentation_cfg,
            project_root=experiment.project_root,
        )
        train_dataset = augmentation_build.dataset
        balanced_target_edge_augmentation_summary = dict(augmentation_build.summary)
        print(
            "[target-edge-augmentation] "
            + json.dumps(balanced_target_edge_augmentation_summary, sort_keys=True),
            flush=True,
        )
    if rare_sampler_enabled:
        if use_ddp:
            raise RuntimeError(
                "rare_positive_sampler.enabled=true is not compatible with the current "
                "DistributedSampler path. Disable it or run single-process training."
            )
        rare_result = compute_rare_positive_sample_weights(
            train_dataset,
            sampler_cfg=rare_sampler_cfg,
            project_root=experiment.project_root,
        )
        sampler_generator = torch.Generator()
        sampler_generator.manual_seed(int(experiment.seed))
        train_sampler = WeightedRandomSampler(
            rare_result.weights,
            num_samples=len(train_dataset),
            replacement=bool(getattr(rare_sampler_cfg, "replacement", True)),
            generator=sampler_generator,
        )
        rare_positive_summary = dict(rare_result.summary)
        rare_positive_summary["replacement"] = bool(
            getattr(rare_sampler_cfg, "replacement", True)
        )
        print(
            "[rare-positive-sampler] "
            + json.dumps(rare_positive_summary, sort_keys=True),
            flush=True,
        )
    elif epoch_sampler_enabled:
        if use_ddp:
            raise RuntimeError(
                "epoch_sampler.enabled=true is currently single-process only. "
                "Disable it or run without DDP."
            )
        fold_id = 1
        match = re.search(r"fold[_-]?(\d+)", str(output_dir), flags=re.IGNORECASE)
        if match is not None:
            fold_id = int(match.group(1))
        train_sampler = EpochBalancedSampler(
            train_dataset,
            sampler_cfg=epoch_sampler_cfg,
            project_root=experiment.project_root,
            output_dir=output_dir / "epoch_sampler",
            fold_id=fold_id,
        )
        print(
            "[epoch-sampler] "
            + json.dumps(
                {
                    "enabled": True,
                    "train_pool_size": len(train_dataset),
                    "epoch_fraction": float(getattr(epoch_sampler_cfg, "epoch_fraction", 0.01)),
                    "epoch_train_size": len(train_sampler),
                    "process_count": len(getattr(train_sampler, "process_ids", [])),
                    "sampler_seed": int(getattr(epoch_sampler_cfg, "sampler_seed", experiment.seed)),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        train_sampler = DistributedSampler(train_dataset, shuffle=True) if use_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=experiment.train.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=experiment.train.num_workers,
        pin_memory=experiment.train.pin_memory,
        collate_fn=_collate_for_experiment(experiment.data),
        **_data_loader_worker_kwargs(experiment.train),
    )

    val_csv = (experiment.project_root / experiment.data.val_data_path).resolve()
    val_loader: DataLoader | None = None
    val_sampler: DistributedSampler | None = None
    val_dataset_for_eval: Any | None = None
    test_loader: DataLoader | None = None
    if experiment.train.tiny_overfit_samples > 0:
        val_dataset_for_eval = train_dataset
        val_sampler = DistributedSampler(train_dataset, shuffle=False) if use_ddp else None
        val_loader = DataLoader(
            train_dataset,
            batch_size=_evaluation_batch_size(experiment.train),
            shuffle=False,
            sampler=val_sampler,
            num_workers=experiment.train.num_workers,
            pin_memory=experiment.train.pin_memory,
            collate_fn=_collate_for_experiment(experiment.data),
            **_data_loader_worker_kwargs(experiment.train),
        )
    elif val_csv.is_file():
        val_split = resolve_split_filter(val_csv, experiment.data.split_column, "val")
        val_manifest_path = (
            (experiment.project_root / val_manifest_rel).resolve() if val_manifest_rel else None
        )
        if val_manifest_path is not None:
            manifest_rows = _split_manifest_row_count(val_manifest_path)
            if manifest_rows <= 0:
                raise RuntimeError(f"validation split manifest is empty: {val_manifest_path}")
            val_split = None
        val_dataset = ProcessGraphTabularDataset(
            experiment.data.val_data_path,
            experiment.data,
            experiment.project_root,
            split_filter=val_split,
            split_manifest=val_manifest_path,
            oper_mean=oper_mean,
            oper_std=oper_std,
            target_mean=target_mean,
            target_std=target_std,
            fixed_slot_clip_max=fixed_slot_clip_max,
        )
        if val_manifest_path is not None and len(val_dataset) <= 0:
            raise RuntimeError(
                "validation split manifest produced zero dataset rows: "
                f"{val_manifest_path}"
            )
        if len(val_dataset) > 0:
            if edge_all:
                cap_val = getattr(experiment.data, "edge_all_cap_val_samples", None)
                if cap_val is not None and int(cap_val) > 0:
                    cap_vn = min(int(cap_val), len(val_dataset))
                    if cap_vn > 0:
                        val_dataset = Subset(val_dataset, list(range(cap_vn)))
            val_dataset_for_eval = val_dataset
            val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None
            val_loader = DataLoader(
                val_dataset,
                batch_size=_evaluation_batch_size(experiment.train),
                shuffle=False,
                sampler=val_sampler,
                num_workers=experiment.train.num_workers,
                pin_memory=experiment.train.pin_memory,
                collate_fn=_collate_for_experiment(experiment.data),
                **_data_loader_worker_kwargs(experiment.train),
            )
    test_csv = (experiment.project_root / experiment.data.test_data_path).resolve()
    if test_csv.is_file():
        test_split = resolve_split_filter(test_csv, experiment.data.split_column, "test")
        test_manifest_path = (
            (experiment.project_root / test_manifest_rel).resolve() if test_manifest_rel else None
        )
        if test_manifest_path is not None:
            manifest_rows = _split_manifest_row_count(test_manifest_path)
            if manifest_rows <= 0:
                raise RuntimeError(f"test split manifest is empty: {test_manifest_path}")
            test_split = None
        test_dataset = ProcessGraphTabularDataset(
            experiment.data.test_data_path,
            experiment.data,
            experiment.project_root,
            split_filter=test_split,
            split_manifest=test_manifest_path,
            oper_mean=oper_mean,
            oper_std=oper_std,
            target_mean=target_mean,
            target_std=target_std,
            fixed_slot_clip_max=fixed_slot_clip_max,
        )
        if test_manifest_path is not None and len(test_dataset) <= 0:
            raise RuntimeError(
                "test split manifest produced zero dataset rows: "
                f"{test_manifest_path}"
            )
        if edge_all:
            cap_test = getattr(experiment.data, "edge_all_cap_test_samples", None)
            if cap_test is not None and int(cap_test) > 0 and len(test_dataset) > int(cap_test):
                test_dataset = Subset(test_dataset, list(range(int(cap_test))))
        if len(test_dataset) > 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=_evaluation_batch_size(experiment.train),
                shuffle=False,
                num_workers=experiment.train.num_workers,
                pin_memory=experiment.train.pin_memory,
                collate_fn=_collate_for_experiment(experiment.data),
                **_data_loader_worker_kwargs(experiment.train),
            )

    named_params = [name for name, _ in raw_model.named_parameters()]
    # region agent log
    _debug_ndjson(
        run_id=debug_run_id,
        hypothesis_id="H1_H2",
        location="scripts/train_process_surrogate.py:train_one_run",
        message="ddp_setup_and_param_index_map",
        data={
            "rank": _rank(),
            "use_ddp": use_ddp,
            "find_unused_parameters": ddp_find_unused_parameters,
            "decoder_on": decoder_tasks_enabled(experiment.data),
            "aux_on": bool(experiment.data.aux_task.enabled),
            "task_specs": [spec.name for spec in task_specs],
            "use_role_embedding": bool(experiment.model.use_role_embedding),
            "use_unit_embedding": bool(experiment.model.use_unit_embedding),
            "use_hx_role_embedding": bool(experiment.model.use_hx_role_embedding),
            "torch_distributed_debug": os.environ.get("TORCH_DISTRIBUTED_DEBUG", ""),
            "param_idx_0": named_params[0] if len(named_params) > 0 else "",
            "param_idx_1": named_params[1] if len(named_params) > 1 else "",
            "param_idx_2": named_params[2] if len(named_params) > 2 else "",
            "param_idx_3": named_params[3] if len(named_params) > 3 else "",
        },
    )
    # endregion
    save_model_weights = not bool(getattr(args, "no_save_model_weights", False))
    if _is_main():
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "transfer_initialization.json").write_text(
            json.dumps(transfer_init_metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if rare_positive_summary is not None:
            (output_dir / "rare_positive_sampler_summary.json").write_text(
                json.dumps(rare_positive_summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if rare_target_edge_augmentation_summary is not None:
            (output_dir / "rare_target_edge_augmentation_summary.json").write_text(
                json.dumps(
                    rare_target_edge_augmentation_summary,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        if balanced_target_edge_augmentation_summary is not None:
            balanced_target_edge_augmentation_csv_tables = dict(
                balanced_target_edge_augmentation_summary.pop("csv_tables", {}) or {}
            )
            (output_dir / "target_edge_augmentation_summary.json").write_text(
                json.dumps(
                    balanced_target_edge_augmentation_summary,
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            for table_name, rows in balanced_target_edge_augmentation_csv_tables.items():
                if not rows:
                    continue
                pd.DataFrame(rows).to_csv(
                    output_dir / f"{table_name}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
        if save_model_weights:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        global_debug_dir = (experiment.project_root / "outputs" / "debug").resolve()
        if not getattr(args, "skip_startup_debug", False):
            _build_x_oper_audit(experiment.project_root, global_debug_dir)
            _build_target_and_answer_audits(experiment.project_root, global_debug_dir)
            _build_main_tabular_sanity_baseline(experiment.project_root, global_debug_dir)
            _build_main_full_vs_gnn_xoper_baseline(
                experiment.project_root, global_debug_dir, process_ids=[1, 2, 5, 8]
            )
        else:
            print(
                "[train] --skip-startup-debug: skipping x_oper/target/baseline audits "
                "(they can take many minutes on full Main CSVs).",
                flush=True,
            )
        if (
            experiment.train.debug_mode
            or getattr(args, "debug", False)
            or getattr(args, "debug_log_file", "")
        ):
            dbg_note = output_dir / "debug" / "NDJSON_LOG_PATHS.txt"
            dbg_note.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                "NDJSON debug logs (for offline analysis):",
                f"- default (all ranks interleaved): {DEFAULT_NDJSON_LOG_PATH}",
            ]
            sess = get_session_ndjson_path()
            if sess is not None:
                lines.append(f"- this run session (rank {_rank()}): {sess}")
            lines.append("")
            lines.append("Other artifacts under this output_dir:")
            lines.append(f"- {output_dir / 'debug' / 'fixed_target_analysis.json'}")
            lines.append(f"- {output_dir / 'debug' / 'problem_batches.jsonl'} (when loss >= debug_loss_threshold)")
            lines.append(f"- {output_dir / 'debug' / '*_batch_stats.csv'}")
            dbg_note.write_text("\n".join(lines), encoding="utf-8")
        _ft_analysis = experiment.data.fixed_tasks
        if getattr(experiment.data, "task_mode", "multitask") == "target_only":
            _ft_analysis = {"target": experiment.data.fixed_tasks["target"]}
        if not edge_all:
            _analyze_fixed_target_distribution(
                fixed_target_frame,
                process_id_column=experiment.data.process_id_column,
                fixed_tasks=_ft_analysis,
                batch_size=experiment.train.batch_size,
                output_dir=output_dir,
                top_k=experiment.train.debug_top_batches_to_report,
            )

    y_edge_mean = None
    y_edge_std = None
    y_edge_cols: list[str] | None = None
    y_edge_counts: list[int] | None = None
    y_edge_warnings: list[str] = []
    y_edge_mass_log_mean = None
    y_edge_mass_log_std = None
    y_edge_mass_log_count = 0
    y_edge_mass_log_warnings: list[str] = []
    y_edge_physical_quantiles: dict[str, dict[str, float]] = {}
    mass_scale_metadata: dict[str, Any] | None = None
    if edge_all and getattr(experiment.data, "normalize_y_edge", False):
        lim = len(train_scaler_dataset)
        if getattr(args, "edge_all_forward_debug", False):
            lim = min(lim, 32)
        source_edge_columns = list(getattr(experiment.data, "edge_target_columns", None) or [])
        # The hierarchical reduced head predicts an 11D subset, while the
        # dataset and physical inverse transforms intentionally retain the
        # complete 14D source schema. Fit/cache statistics for the source
        # columns; get_pi_targets performs the name-based 14D -> 11D mapping.
        yt_dim = (
            len(source_edge_columns)
            if source_edge_columns
            else int(getattr(encoder_cfg, "stream_target_dim", 12))
        )
        scaler_cache_path = _y_edge_scaler_cache_path(
            experiment=experiment,
            train_csv=train_csv,
            train_manifest_path=train_manifest_path,
            dataset_len=lim,
            stream_target_dim=yt_dim,
        )
        scaler_cache: dict[str, Any] | None = None
        need_mass_log_scaler = bool(getattr(experiment.train, "use_log1p_mass_flow_loss", False))
        mass_flow_transform = resolve_mass_flow_transform(experiment.train)
        mass_flow_log_tau = resolve_mass_flow_log_tau(experiment.train)
        mass_flow_log_scale = resolve_mass_flow_log_scale(experiment.train)
        mass_flow_log_eps = resolve_mass_flow_log_eps(experiment.train)
        if scaler_cache_path.is_file():
            try:
                loaded = torch.load(scaler_cache_path, map_location="cpu", weights_only=False)
                if (
                    isinstance(loaded, dict)
                    and len(loaded.get("columns", [])) == yt_dim
                    and int(loaded.get("dataset_len", -1)) == lim
                    and (
                        not need_mass_log_scaler
                        or (
                            loaded.get("mass_flow_log1p_mean") is not None
                            and loaded.get("mass_flow_log1p_std") is not None
                            and int(loaded.get("mass_flow_log1p_count", 0)) > 0
                            and math.isfinite(float(torch.as_tensor(loaded["mass_flow_log1p_mean"]).reshape(())))
                            and math.isfinite(float(torch.as_tensor(loaded["mass_flow_log1p_std"]).reshape(())))
                            and float(torch.as_tensor(loaded["mass_flow_log1p_std"]).reshape(())) > 0.0
                            and str(loaded.get("mass_flow_transform", "log1p")) == mass_flow_transform
                            and math.isclose(float(loaded.get("mass_flow_log_tau", 1.0)), mass_flow_log_tau, rel_tol=0.0, abs_tol=1.0e-12)
                            and math.isclose(float(loaded.get("mass_flow_log_scale", 1.0)), mass_flow_log_scale, rel_tol=0.0, abs_tol=1.0e-12)
                            and math.isclose(float(loaded.get("mass_flow_log_eps", 1.0e-8)), mass_flow_log_eps, rel_tol=0.0, abs_tol=1.0e-20)
                        )
                    )
                ):
                    scaler_cache = loaded
            except Exception as exc:
                print(f"[edge_all][scaler-cache][warn] ignored invalid cache: {exc}", flush=True)
        if scaler_cache is None and source_edge_columns:
            legacy_source_columns = [
                "Temp", "Pres", "Vol_Flow", "Mole_Flow", "Mass_Flow",
                "Frac_H2O", "Frac_H2", "Frac_CH4", "Frac_CO2", "Frac_CO",
                "Frac_O2", "Frac_N2", "Enthalpy", "Density",
            ]
            if set(source_edge_columns).issubset(legacy_source_columns):
                legacy_cache_path = _y_edge_scaler_cache_path(
                    experiment=experiment,
                    train_csv=train_csv,
                    train_manifest_path=train_manifest_path,
                    dataset_len=lim,
                    stream_target_dim=len(legacy_source_columns),
                    edge_target_columns=legacy_source_columns,
                )
                if legacy_cache_path != scaler_cache_path and legacy_cache_path.is_file():
                    try:
                        legacy = torch.load(legacy_cache_path, map_location="cpu", weights_only=False)
                        legacy_columns = [str(x) for x in legacy.get("columns", [])]
                        legacy_indices = [legacy_columns.index(name) for name in source_edge_columns]
                        valid_mass_log = (
                            not need_mass_log_scaler
                            or (
                                legacy.get("mass_flow_log1p_mean") is not None
                                and legacy.get("mass_flow_log1p_std") is not None
                                and int(legacy.get("mass_flow_log1p_count", 0)) > 0
                                and str(legacy.get("mass_flow_transform", "log1p")) == mass_flow_transform
                                and math.isclose(float(legacy.get("mass_flow_log_tau", 1.0)), mass_flow_log_tau, rel_tol=0.0, abs_tol=1.0e-12)
                                and math.isclose(float(legacy.get("mass_flow_log_scale", 1.0)), mass_flow_log_scale, rel_tol=0.0, abs_tol=1.0e-12)
                                and math.isclose(float(legacy.get("mass_flow_log_eps", 1.0e-8)), mass_flow_log_eps, rel_tol=0.0, abs_tol=1.0e-20)
                            )
                        )
                        if int(legacy.get("dataset_len", -1)) == lim and valid_mass_log:
                            legacy_counts = list(legacy.get("masked_value_counts_per_column", []))
                            scaler_cache = dict(legacy)
                            scaler_cache["mean"] = torch.as_tensor(legacy["mean"])[legacy_indices].clone()
                            scaler_cache["std"] = torch.as_tensor(legacy["std"])[legacy_indices].clone()
                            scaler_cache["columns"] = list(source_edge_columns)
                            scaler_cache["masked_value_counts_per_column"] = (
                                [int(legacy_counts[index]) for index in legacy_indices]
                                if len(legacy_counts) == len(legacy_columns) else []
                            )
                            scaler_cache_path.parent.mkdir(parents=True, exist_ok=True)
                            tmp_cache_path = scaler_cache_path.with_suffix(".tmp")
                            torch.save(scaler_cache, tmp_cache_path)
                            os.replace(tmp_cache_path, scaler_cache_path)
                            print(
                                "[edge_all][scaler-cache] reused matching columns from legacy "
                                f"cache and wrote {scaler_cache_path}",
                                flush=True,
                            )
                    except (KeyError, ValueError, TypeError, RuntimeError) as exc:
                        print(
                            f"[edge_all][scaler-cache][warn] could not migrate legacy cache: {exc}",
                            flush=True,
                        )
        if scaler_cache is None:
            print(
                f"[edge_all][scaler-cache] miss; fitting on {lim} train samples once: "
                f"{scaler_cache_path}",
                flush=True,
            )
            t0 = time.perf_counter()
            try:
                if not hasattr(train_scaler_dataset, "fit_y_edge_scaler_fast"):
                    raise NotImplementedError("train scaler dataset does not expose fit_y_edge_scaler_fast")
                y_edge_fit, mass_log_fit = train_scaler_dataset.fit_y_edge_scaler_fast(
                    limit=lim,
                    stream_target_dim=yt_dim,
                    log_column_name="Mass_Flow" if need_mass_log_scaler else None,
                    mass_flow_transform=mass_flow_transform,
                    mass_flow_log_tau=mass_flow_log_tau,
                    mass_flow_log_scale=mass_flow_log_scale,
                    mass_flow_log_eps=mass_flow_log_eps,
                    physical_quantile_column_name="Mass_Flow",
                    physical_quantile_probs=(0.05, 0.25, 0.75, 0.90, 0.95, 0.99),
                )
                print(
                    f"[edge_all][scaler-cache] fast stats path completed in {time.perf_counter() - t0:.2f}s",
                    flush=True,
                )
            except NotImplementedError:
                y_edge_fit = compute_y_edge_scaler(
                    (train_scaler_dataset[i] for i in range(lim)),
                    stream_target_dim=yt_dim,
                )
                mass_log_fit = None
                print(
                    f"[edge_all][scaler-cache] graph-build fallback completed in {time.perf_counter() - t0:.2f}s",
                    flush=True,
                )
            y_edge_mean, y_edge_std = y_edge_fit.mean, y_edge_fit.std
            y_edge_cols = list(y_edge_fit.columns)
            y_edge_counts = list(y_edge_fit.masked_value_counts_per_column)
            y_edge_warnings = list(y_edge_fit.warnings)
            y_edge_physical_quantiles = dict(y_edge_fit.physical_quantiles)
            if need_mass_log_scaler:
                if mass_log_fit is None:
                    mass_log_fit = compute_y_edge_log1p_column_scaler(
                        (train_scaler_dataset[i] for i in range(lim)),
                        column_name="Mass_Flow",
                        stream_target_dim=yt_dim,
                        mass_flow_transform=mass_flow_transform,
                        mass_flow_log_tau=mass_flow_log_tau,
                        mass_flow_log_scale=mass_flow_log_scale,
                        mass_flow_log_eps=mass_flow_log_eps,
                    )
                y_edge_mass_log_mean = mass_log_fit.mean
                y_edge_mass_log_std = mass_log_fit.std
                y_edge_mass_log_count = int(mass_log_fit.count)
                y_edge_mass_log_warnings = list(mass_log_fit.warnings)
            scaler_cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_cache_path = scaler_cache_path.with_suffix(".tmp")
            torch.save(
                {
                    "mean": y_edge_mean.cpu(),
                    "std": y_edge_std.cpu(),
                    "columns": y_edge_cols,
                    "masked_value_counts_per_column": y_edge_counts,
                    "warnings": y_edge_warnings,
                    "dataset_len": lim,
                    "mass_flow_log1p_mean": None if y_edge_mass_log_mean is None else y_edge_mass_log_mean.cpu(),
                    "mass_flow_log1p_std": None if y_edge_mass_log_std is None else y_edge_mass_log_std.cpu(),
                    "mass_flow_log1p_count": y_edge_mass_log_count,
                    "mass_flow_log1p_warnings": y_edge_mass_log_warnings,
                    "mass_flow_transform": mass_flow_transform,
                    "mass_flow_log_tau": mass_flow_log_tau,
                    "mass_flow_log_scale": mass_flow_log_scale,
                    "mass_flow_log_eps": mass_flow_log_eps,
                    "physical_quantiles": y_edge_physical_quantiles,
                    "stream_key_canonicalization_version": STREAM_KEY_CANONICALIZATION_VERSION,
                },
                tmp_cache_path,
            )
            os.replace(tmp_cache_path, scaler_cache_path)
            print(f"[edge_all][scaler-cache] wrote {scaler_cache_path}", flush=True)
        else:
            y_edge_mean = torch.as_tensor(scaler_cache["mean"], dtype=torch.float32)
            y_edge_std = torch.as_tensor(scaler_cache["std"], dtype=torch.float32)
            y_edge_cols = [str(x) for x in scaler_cache["columns"]]
            y_edge_counts = [int(x) for x in scaler_cache.get("masked_value_counts_per_column", [])]
            y_edge_warnings = [str(x) for x in scaler_cache.get("warnings", [])]
            y_edge_physical_quantiles = dict(scaler_cache.get("physical_quantiles", {}))
            if need_mass_log_scaler and scaler_cache.get("mass_flow_log1p_mean") is not None:
                y_edge_mass_log_mean = torch.as_tensor(scaler_cache["mass_flow_log1p_mean"], dtype=torch.float32)
                y_edge_mass_log_std = torch.as_tensor(scaler_cache["mass_flow_log1p_std"], dtype=torch.float32)
                y_edge_mass_log_count = int(scaler_cache.get("mass_flow_log1p_count", 0))
                y_edge_mass_log_warnings = [str(x) for x in scaler_cache.get("mass_flow_log1p_warnings", [])]
            print(
                f"[edge_all][scaler-cache] hit; skipped {lim} graph builds: {scaler_cache_path}",
                flush=True,
            )
        mass_aux_cfg = getattr(experiment.train, "mass_flow_physical_auxiliary", None)
        if bool(getattr(mass_aux_cfg, "enabled", False)):
            mass_scale_normalizer = {
                "mean": y_edge_mean,
                "std": y_edge_std,
                "columns": y_edge_cols or source_edge_columns,
                "mass_flow_physical_p05": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p05"),
                "mass_flow_physical_p25": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p25"),
                "mass_flow_physical_p75": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p75"),
                "mass_flow_physical_p95": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p95"),
            }
            resolved_mass_scale, resolved_mass_scale_source = resolve_mass_flow_physical_scale(
                train_cfg=experiment.train,
                normalizer=mass_scale_normalizer,
            )
            mass_aux_cfg.resolved_scale = float(resolved_mass_scale)
            mass_scale_metadata = {
                "enabled": True,
                "scale": float(resolved_mass_scale),
                "scale_method": str(getattr(mass_aux_cfg, "scale_method", "train_std")),
                "scale_source": str(resolved_mass_scale_source),
                "train_split_only": True,
                "log_weight": float(getattr(mass_aux_cfg, "log_weight", 1.0)),
                "physical_max_weight": float(getattr(mass_aux_cfg, "weight", 0.10)),
                "schedule_type": str(getattr(mass_aux_cfg, "schedule_type", "linear_warmup")),
                "schedule_start_epoch_1_based": int(getattr(mass_aux_cfg, "schedule_start_epoch", 3)),
                "schedule_end_epoch_1_based": int(getattr(mass_aux_cfg, "schedule_end_epoch", 7)),
                "schedule_start_weight": float(getattr(mass_aux_cfg, "schedule_start_weight", 0.0)),
                "schedule_end_weight": float(
                    getattr(mass_aux_cfg, "schedule_end_weight", None)
                    if getattr(mass_aux_cfg, "schedule_end_weight", None) is not None
                    else getattr(mass_aux_cfg, "weight", 0.10)
                ),
            }
            if _is_main():
                (output_dir / "mass_dual_space_loss_metadata.json").write_text(
                    json.dumps(mass_scale_metadata, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(
                    "[mass-dual-space] "
                    f"scale={resolved_mass_scale:.8g} source={resolved_mass_scale_source} "
                    f"log_weight={mass_scale_metadata['log_weight']:.4g} "
                    f"physical_max_weight={mass_scale_metadata['physical_max_weight']:.4g} "
                    f"schedule={mass_scale_metadata['schedule_type']} "
                    f"epochs={mass_scale_metadata['schedule_start_epoch_1_based']}"
                    f"..{mass_scale_metadata['schedule_end_epoch_1_based']} (1-based)",
                    flush=True,
                )
        if _is_main():
            torch.save(
                {
                    "mean": y_edge_mean.cpu(),
                    "std": y_edge_std.cpu(),
                    "columns": y_edge_cols,
                    "mass_flow_log1p_mean": None if y_edge_mass_log_mean is None else y_edge_mass_log_mean.cpu(),
                    "mass_flow_log1p_std": None if y_edge_mass_log_std is None else y_edge_mass_log_std.cpu(),
                    "mass_flow_log1p_count": y_edge_mass_log_count,
                    "mass_flow_transform": mass_flow_transform,
                    "mass_flow_log_tau": mass_flow_log_tau,
                    "mass_flow_log_scale": mass_flow_log_scale,
                    "mass_flow_log_eps": mass_flow_log_eps,
                    "physical_quantiles": y_edge_physical_quantiles,
                    "fit_provenance": train_fit_provenance,
                },
                output_dir / "y_edge_scaler.pt",
            )
            (output_dir / "scaler_fit_provenance.json").write_text(
                json.dumps(train_fit_provenance, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "property": y_edge_cols,
                    "mean": y_edge_mean.cpu().numpy(),
                    "std": y_edge_std.cpu().numpy(),
                    "masked_fit_count": y_edge_counts,
                }
            ).to_csv(output_dir / "scaler_stats.csv", index=False)
            if y_edge_warnings:
                (output_dir / "scaler_fit_warnings.txt").write_text(
                    "\n".join(y_edge_warnings) + "\n", encoding="utf-8"
                )
            if y_edge_mass_log_mean is not None and y_edge_mass_log_std is not None:
                pd.DataFrame(
                    {
                        "property": ["Mass_Flow"],
                        "log1p_mean": [float(y_edge_mass_log_mean)],
                        "log1p_std": [float(y_edge_mass_log_std)],
                        "masked_fit_count": [int(y_edge_mass_log_count)],
                    }
                ).to_csv(output_dir / "mass_flow_log1p_scaler_stats.csv", index=False)
                if y_edge_mass_log_warnings:
                    (output_dir / "mass_flow_log1p_scaler_warnings.txt").write_text(
                        "\n".join(y_edge_mass_log_warnings) + "\n", encoding="utf-8"
                    )
            primary_scaler_parts: list[str] = []
            for _pname in ("Frac_H2", "Frac_CO2", "Frac_H2O"):
                if _pname in y_edge_cols:
                    _j = y_edge_cols.index(_pname)
                    primary_scaler_parts.append(
                        f"{_pname}[idx={_j}, mean={float(y_edge_mean[_j]):.6g}, "
                        f"std={float(y_edge_std[_j]):.6g}, n={int(y_edge_counts[_j]) if y_edge_counts else 0}]"
                    )
            if primary_scaler_parts:
                fraction_loss_type = str(
                    getattr(experiment.train, "pi_fraction_loss_type", "legacy") or "legacy"
                ).strip().lower()
                if fraction_loss_type == "clr":
                    fraction_loss_note = (
                        "CLR loss uses physical fractions without z-score; scaler stats are diagnostics only; "
                    )
                elif bool(getattr(experiment.train, "pi_normalize_fraction_loss", False)):
                    fraction_loss_note = "normalized loss uses train-split z-score; "
                else:
                    fraction_loss_note = "physical-scale fraction loss; scaler stats are diagnostics only; "
                print(
                    "[edge_all][scaler][primary] "
                    + fraction_loss_note
                    + "; ".join(primary_scaler_parts),
                    flush=True,
                )

    if edge_all and getattr(experiment.train, "edge_all_assert_no_leakage", False):
        if _is_main():
            peek = next(iter(train_loader))
            assert_no_edge_all_target_leakage(
                peek,
                edge_struct_dim=int(encoder_cfg.edge_struct_dim),
                edge_target_columns=peek.edge_target_columns,
            )
        if use_ddp:
            dist.barrier()

    if (
        edge_all
        and _is_main()
        and bool(getattr(experiment.train, "edge_all_export_target_feature_mapping_audit", False))
    ):
        from process_graph.experiment.target_feature_mapping_audit import (
            write_target_feature_mapping_audit,
        )

        try:
            _audit_peek = next(iter(train_loader))
            _audit_path = write_target_feature_mapping_audit(
                output_dir,
                export_meta=_audit_peek.edge_export_meta,
                y_mask=_audit_peek.target_masks.get("edge_stream"),
                edge_target_columns=list(_audit_peek.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
                train_cfg=experiment.train,
            )
        except StopIteration:
            _audit_path = write_target_feature_mapping_audit(
                output_dir,
                train_cfg=experiment.train,
            )
        print(f"[StreamVectorPolicy] target_feature_mapping_audit={_audit_path}", flush=True)

    if _is_main() and not getattr(args, "skip_startup_debug", False):
        global_debug_dir = (experiment.project_root / "outputs" / "debug").resolve()
        _dump_batch_debug_by_process(
            project_root=experiment.project_root,
            experiment=experiment,
            oper_mean=oper_mean,
            oper_std=oper_std,
            target_mean=target_mean,
            target_std=target_std,
            fixed_slot_clip_max=fixed_slot_clip_max,
            out_dir=global_debug_dir,
        )

    if edge_all and getattr(args, "edge_all_forward_debug", False):
        if use_ddp:
            raise RuntimeError("--edge-all-forward-debug requires a single-process run (no torchrun DDP).")
        if not _is_main():
            return {}
        peek = next(iter(train_loader))
        assert_no_edge_all_target_leakage(
            peek,
            edge_struct_dim=int(encoder_cfg.edge_struct_dim),
            edge_target_columns=peek.edge_target_columns,
        )
        batch_data = {k: v.to(device) for k, v in peek.model_kwargs.items()}
        targets = {k: v.to(device) for k, v in peek.targets.items()}
        target_masks = {k: v.to(device) for k, v in peek.target_masks.items()}
        _apply_v4_target_loss_weights(peek, target_masks, experiment.train, device)
        task_inputs = {
            head: {k: v.to(device) for k, v in payload.items()}
            for head, payload in peek.task_inputs.items()
        }
        y_raw = targets["edge_stream"]
        y_n = _edge_all_normalize(y_raw, y_edge_mean, y_edge_std, device)
        raw_model.eval()
        with torch.no_grad():
            forward_result = raw_model(
                batch_data,
                task_inputs=task_inputs,
                return_encoder_output=True,
            )
            preds = forward_result["predictions"]
            encoder_output = forward_result["encoder_output"]
            if "y_edge_pred" in preds:
                loss_items = compute_training_loss(
                    preds,
                    {**targets, "edge_stream": y_n},
                    target_masks,
                    experiment.train,
                    experiment.data,
                    **_edge_all_loss_kwargs(
                        peek,
                        experiment.data,
                        y_edge_mean=y_edge_mean,
                        y_edge_std=y_edge_std,
                    ),
                )
                yp = preds["y_edge_pred"]
                yt = y_n
                ym = target_masks["edge_stream"]
            elif "edge_stream" in preds:
                preds = dict(preds)
                preds["y_edge_pred"] = preds["edge_stream"]
                loss_items = compute_training_loss(
                    preds,
                    {**targets, "edge_stream": y_n},
                    target_masks,
                    experiment.train,
                    experiment.data,
                    **_edge_all_loss_kwargs(
                        peek,
                        experiment.data,
                        y_edge_mean=y_edge_mean,
                        y_edge_std=y_edge_std,
                    ),
                )
                yp = preds["y_edge_pred"]
                yt = y_n
                ym = target_masks["edge_stream"]
            elif "main_stream_pred" in preds:
                pi_targets = get_pi_targets(
                    targets={**targets, "edge_stream": y_n},
                    target_masks=target_masks,
                    outputs=preds,
                    train_cfg=experiment.train,
                    data_cfg=experiment.data,
                    edge_target_columns=list(peek.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
                )
                yp = preds["main_stream_pred"]
                yt = pi_targets["main_target"]
                ym = pi_targets["main_mask"]
                loss_edge, _ = masked_edge_regression_loss(
                    yp,
                    yt,
                    ym,
                    loss_type=str(getattr(experiment.train, "pi_main_loss_type", getattr(experiment.train, "loss_type_edge_all", "mse"))),
                )
                loss_items = {"loss_total": loss_edge, "loss_edge": loss_edge}
            else:
                raise RuntimeError(
                    "edge_all_forward_debug requires one of preds['y_edge_pred'], preds['edge_stream'], "
                    "or preds['main_stream_pred']."
                )
        struct = batch_data.get("edge_struct_attr")
        if struct is None:
            struct = batch_data.get("edge_oper")
        validate_edge_all_batch(
            y_edge_pred=yp,
            y_edge_true=yt,
            y_edge_mask=target_masks["edge_stream"],
            edge_struct=struct,
            edge_struct_dim=int(encoder_cfg.edge_struct_dim),
            answer_edge_pos=peek.answer_edge_pos,
        )
        masked_edge_regression_loss(
            yp,
            yt,
            ym,
            loss_type=str(getattr(experiment.train, "loss_type_edge_all", "mse")),
        )
        print("[edge_all_forward_debug]")
        print(f"  y_edge_pred={tuple(yp.shape)} y_edge_true={tuple(yt.shape)} mask={tuple(ym.shape)}")
        print(
            f"  loss_total={float(loss_items['loss_total'].detach().cpu()):.6f} "
            f"loss_edge={float(loss_items.get('loss_edge', loss_items['loss_total']).detach().cpu()):.6f} "
            f"loss_primary_frac={float(loss_items.get('loss_primary_frac', loss_items['loss_total'] * 0).detach().cpu()):.6f} "
            f"loss_all_edge_r2={float(loss_items.get('loss_all_edge_r2', loss_items['loss_total'] * 0).detach().cpu()):.6f} "
            f"primary_frac_count={float(loss_items.get('primary_frac_count', loss_items['loss_total'] * 0).detach().cpu()):.0f} "
            f"r2_loss_num_elements={float(loss_items.get('r2_loss_num_elements', loss_items['loss_total'] * 0).detach().cpu()):.0f} "
            f"target_loss_num_elements={float(loss_items.get('target_loss_num_elements', loss_items['loss_total'] * 0).detach().cpu()):.0f}"
        )
        print(
            f"  supervision: primary_frac_loss_weight={getattr(experiment.train, 'primary_frac_loss_weight', 0.0)} "
            f"all_edge_r2_loss_weight={resolve_all_edge_r2_config(experiment.train)['loss_weight']} "
            f"all_edge_r2_min_count={resolve_all_edge_r2_config(experiment.train)['min_count']} "
            f"all_edge_r2_config_source={resolve_all_edge_r2_config(experiment.train)['source']} "
            f"use_legacy_answer_weighting={getattr(experiment.train, 'use_legacy_answer_weighting', False)}"
        )
        if peek.answer_edge_pos:
            print(f"  answer_edge_pos[0]={peek.answer_edge_pos[0]} (legacy slots; v4 metrics use target_id rows)")
        print(
            f"  edge_struct_attr min/max=({float(struct.min().detach().cpu()):.6f}, {float(struct.max().detach().cpu()):.6f})"
        )
        print(
            f"  pred mean/std={float(yp.mean().detach().cpu()):.6f}/{_safe_std(yp):.6f} "
            f"true(norm) mean/std={float(y_n.mean().detach().cpu()):.6f}/{_safe_std(y_n):.6f}"
        )
        representation_stats = {
            "node_initial": _representation_stats(
                encoder_output.initial_node_embeddings
            ),
            "node_local_final": _representation_stats(
                encoder_output.local_node_embeddings
            ),
            "global": _representation_stats(encoder_output.global_embedding),
        }
        for layer_index, layer_tensor in enumerate(
            encoder_output.layer_node_embeddings, start=1
        ):
            representation_stats[f"gnn_layer_{layer_index}"] = (
                _representation_stats(layer_tensor)
            )
        if encoder_output.edge_embeddings is not None:
            representation_stats["edge"] = _representation_stats(
                encoder_output.edge_embeddings
            )
        if "shared_edge_latent" in preds:
            representation_stats["shared_edge_latent"] = _representation_stats(
                preds["shared_edge_latent"]
            )
        print(
            "  representation="
            + ", ".join(
                f"{name}[dim={stats.get('dim')},rank={float(stats.get('effective_rank', 0.0)):.2f},"
                f"dead={float(stats.get('dead_dimension_ratio', 0.0)):.3f}]"
                for name, stats in representation_stats.items()
            )
        )
        eb = batch_data["edge_batch"].detach().cpu().tolist()
        n_graphs = int(batch_data["batch"].max().item()) + 1 if batch_data["batch"].numel() else 0
        ec = Counter(eb)
        per_graph_edges = [int(ec[i]) for i in range(n_graphs)]
        ap_shapes: list[dict[str, Any]] = []
        if peek.answer_edge_pos:
            for gi, pos_map in enumerate(peek.answer_edge_pos):
                off = sum(per_graph_edges[:gi]) if gi < len(per_graph_edges) else 0
                local_map = {slot: int(gpos) - int(off) for slot, gpos in pos_map.items()}
                pred_row_shapes = {slot: list(yp[int(gpos)].shape) for slot, gpos in pos_map.items()}
                ap_shapes.append(
                    {
                        "graph_index": gi,
                        "num_edges_this_graph": per_graph_edges[gi] if gi < len(per_graph_edges) else None,
                        "global_answer_positions": dict(pos_map),
                        "local_answer_edge_index": local_map,
                        "y_edge_pred_row_shape_by_slot": pred_row_shapes,
                    }
                )
        dbg_path = output_dir / "first_batch_debug.json"
        dbg_path.write_text(
            json.dumps(
                {
                    "mode": "edge_all_forward_debug",
                    "y_edge_pred_shape": list(yp.shape),
                    "y_edge_true_shape": list(y_raw.shape),
                    "y_edge_mask_shape": list(target_masks["edge_stream"].shape),
                    "edge_struct_shape": list(struct.shape),
                    "edge_struct_min": float(struct.min().detach().cpu()),
                    "edge_struct_max": float(struct.max().detach().cpu()),
                    "num_graphs": n_graphs,
                    "per_graph_num_edges": per_graph_edges,
                    "answer_edge_pos_global": peek.answer_edge_pos,
                    "answer_edge_pos_batch_audit": ap_shapes,
                    "edge_target_columns": list(peek.edge_target_columns or []),
                    "representation_stats": representation_stats,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  wrote {dbg_path}")
        return {}

    if getattr(args, "inference_only_benchmark", False):
        if use_ddp:
            raise RuntimeError("--inference-only-benchmark requires a single-process (non-DDP) run.")
        if not pretrained_arg:
            raise ValueError("--inference-only-benchmark requires --pretrained-checkpoint.")
        if test_loader is None:
            raise RuntimeError("--inference-only-benchmark requires a non-empty test loader.")
        if not _is_main():
            cleanup_ddp()
            return {}

        output_dir.mkdir(parents=True, exist_ok=True)
        inference_payload = _benchmark_inference_loader(
            model=model,
            loader=test_loader,
            device=device,
            train_cfg=experiment.train,
            warmup_runs=args.inference_warmup_runs,
            measured_runs=args.inference_measured_runs,
            max_batches=args.inference_max_batches,
            use_amp=bool(experiment.train.mixed_precision and device.type == "cuda"),
            target_edges_only=bool(args.inference_target_edges_only),
            verify_target_edge_values=bool(args.inference_target_edges_only),
        )
        inference_payload.update(
            {
                "benchmark_mode": "checkpoint_only",
                "checkpoint_path": str(pretrained_path.resolve()),
                "checkpoint_load_strict": bool(getattr(args, "pretrained_strict_load", True)),
                "config_path": str(experiment_path.resolve()),
                "test_split_manifest_path": str(test_manifest_path.resolve()) if test_manifest_path else "",
                "test_dataset_size": int(len(test_loader.dataset)),
                "total_parameters": int(sum(p.numel() for p in raw_model.parameters())),
                "trainable_parameters": int(sum(p.numel() for p in raw_model.parameters() if p.requires_grad)),
                "precision": "amp" if bool(experiment.train.mixed_precision and device.type == "cuda") else "float32",
            }
        )
        benchmark_json = output_dir / "inference_benchmark.json"
        benchmark_csv = output_dir / "inference_benchmark_summary.csv"
        benchmark_json.write_text(
            json.dumps(inference_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        pd.DataFrame([inference_payload]).to_csv(benchmark_csv, index=False, encoding="utf-8-sig")
        print(
            "[inference-only-benchmark] "
            f"samples_per_run={inference_payload['samples_per_run']} "
            f"batches_per_run={inference_payload['batches_per_run']} "
            f"latency_ms_per_sample={inference_payload['inference_time_per_sample_ms']:.6f} "
            f"median_ms_per_sample={inference_payload['inference_time_median_ms']:.6f} "
            f"throughput={inference_payload['throughput_samples_per_sec']:.3f}/s\n"
            f"[inference-only-benchmark] wrote {benchmark_json}",
            flush=True,
        )
        cleanup_ddp()
        return {
            "best_metric": float("nan"),
            "best_epoch": float("nan"),
            "output_dir": str(output_dir),
        }

    optimizer = build_optimizer(raw_model, experiment.train)
    scheduler = build_scheduler(optimizer, experiment.train)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=experiment.train.mixed_precision and device.type == "cuda",
    )
    use_amp = bool(experiment.train.mixed_precision and device.type == "cuda")
    active_decoder_cats = [name for name, cfg in experiment.data.decoder_tasks.items() if cfg.enabled]
    train_history: dict[str, list] = defaultdict(list)
    val_history: dict[str, list] = defaultdict(list)
    best_metric: float | None = None
    best_epoch = -1
    best_val_loss: float | None = None
    final_val_loss: float | None = None
    final_epoch = 0
    early_stop_epoch = -1
    epochs_without_improve = 0
    stopped_early = False
    early_stop_best: float | None = None
    best_ckpt_path = ckpt_dir / "best.pt"
    last_ckpt_path = ckpt_dir / "last.pt"
    eval_ckpt_path: Path | None = last_ckpt_path if save_model_weights else None

    try:
        model.train()
        global_step = 0
        optimizer_steps_total = 0
        optimizer_budget_reached = False
        monitor_fallback_warned = False
        target_only = getattr(experiment.data, "task_mode", "multitask") == "target_only"
        if _is_main() and experiment.train.debug_mode and not edge_all:
            # nn.ModuleDict does not always implement dict-like .get (e.g. older PyTorch); use __contains__.
            _heads = raw_model.heads
            target_head = _heads["target"] if "target" in _heads else None
            tailgas_head = _heads["tailgas"] if "tailgas" in _heads else None
            target_head_id = id(target_head) if target_head is not None else -1
            tailgas_head_id = id(tailgas_head) if tailgas_head is not None else -1
            target_head_params = (
                [name for name, _ in _heads["target"].named_parameters()] if "target" in _heads else []
            )
            tailgas_head_params = (
                [name for name, _ in _heads["tailgas"].named_parameters()] if "tailgas" in _heads else []
            )
            print("[DEBUG][routing] task head routing summary")
            print(
                f"[DEBUG][routing] target_column={experiment.data.fixed_tasks['target'].target_variable}"
            )
            if not target_only:
                print(
                    f"[DEBUG][routing] tailgas_column={experiment.data.fixed_tasks['tailgas'].target_variable}"
                )
            print(
                f"[DEBUG][routing] target_head_id={target_head_id} tailgas_head_id={tailgas_head_id} "
                f"same_head={target_head_id == tailgas_head_id}"
            )
            print(f"[DEBUG][routing] target_head_params={target_head_params}")
            if not target_only:
                print(f"[DEBUG][routing] tailgas_head_params={tailgas_head_params}")
            print(
                f"[DEBUG][routing] target_scaler(mean={float((target_mean or {}).get('target', 0.0)):.6f}, "
                f"std={float((target_std or {}).get('target', 1.0)):.6f})"
            )
            if not target_only:
                print(
                    f"[DEBUG][routing] tailgas_scaler(mean={float((target_mean or {}).get('tailgas', 0.0)):.6f}, "
                    f"std={float((target_std or {}).get('tailgas', 1.0)):.6f})"
                )
            sanity_record = train_dataset[0] if len(train_dataset) > 0 else None
            if sanity_record is not None:
                graph = sanity_record.graph
                print("[SANITY CHECK]")
                print(f"input: {graph.x_oper[0] if graph.x_oper else []}")
                print(f"target: {sanity_record.slot_targets.get('target')}")
                if not target_only:
                    print(f"tailgas: {sanity_record.slot_targets.get('tailgas')}")
        _cuda_synchronize(device)
        fit_wall_started_at = time.perf_counter()
        best_checkpoint_wall_sec = float("nan")
        for epoch in range(experiment.train.epochs):
            final_epoch = int(epoch + 1)
            raw_model.set_training_epoch(final_epoch)
            _cuda_synchronize(device)
            epoch_started_at = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            pinn_schedule_scale = _pinn_weight_schedule_scale(
                experiment.train, epoch, optimizer_steps_total
            )
            active_train_cfg = _scaled_pinn_train_config(experiment.train, pinn_schedule_scale)
            active_mass_aux_cfg = getattr(active_train_cfg, "mass_flow_physical_auxiliary", None)
            if active_mass_aux_cfg is not None:
                active_mass_aux_cfg.current_epoch = final_epoch
            if (
                _is_main()
                and bool(getattr(experiment.train, "pinn_weight_schedule", None) or {})
                and not _terminal_log_is_quiet(experiment.train)
            ):
                print(
                    f"[pinn-weight-schedule] epoch={epoch + 1}/{experiment.train.epochs} "
                    f"scale={pinn_schedule_scale:.4f}",
                    flush=True,
                )
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            epoch_comp_sums: defaultdict[str, Any] = defaultdict(float)
            edge_all_loss_prop_norm: defaultdict[str, float] = defaultdict(float)
            edge_all_loss_prop_raw: defaultdict[str, float] = defaultdict(float)
            edge_all_loss_prop_count: defaultdict[str, float] = defaultdict(float)
            edge_all_loss_group_norm: defaultdict[str, float] = defaultdict(float)
            metric_slots = ("target",) if target_only else ("target", "tailgas")
            epoch_metric_accum = {s: _init_metric_accumulator() for s in metric_slots}
            epoch_loss = 0.0
            steps = 0
            nonfinite_skips = 0
            nonfinite_skip_budget = _nonfinite_train_skip_budget(train_loader, experiment.train)
            epoch_aborted_nonfinite = False
            epoch_nonfinite_reason = ""
            epoch_recovery_lrs: tuple[list[float], list[float]] | None = None
            epoch_start_snapshot = (
                _snapshot_training_state(raw_model, optimizer)
                if edge_all and bool(getattr(experiment.train, "recover_nonfinite_edge_all", True))
                else None
            )
            batch_target_mae = float("nan")
            batch_target_rmse = float("nan")
            batch_target_r2 = float("nan")
            batch_tailgas_mae = float("nan")
            batch_tailgas_rmse = float("nan")
            batch_tailgas_r2 = float("nan")
            batch_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{experiment.train.epochs}",
                leave=True,
                disable=(not _is_main()) or str(os.environ.get("TQDM_DISABLE", "")).strip().lower() in {"1", "true", "yes", "on"},
            )
            try:
                num_epoch_batches = len(train_loader)
            except TypeError:
                num_epoch_batches = "?"
            if (
                _is_main()
                and bool(getattr(batch_pbar, "disable", False))
                and not _terminal_log_is_quiet(experiment.train)
            ):
                bar, pct = _terminal_progress_bar(0, num_epoch_batches)
                print(
                    f"[epoch {epoch + 1}/{experiment.train.epochs}] train_start "
                    f"batches={num_epoch_batches} {pct} {bar} elapsed=0s eta=?",
                    flush=True,
                )
            for batch_idx, batch in enumerate(batch_pbar):
                max_optimizer_steps = int(getattr(experiment.train, "max_optimizer_steps", 0) or 0)
                if max_optimizer_steps > 0 and optimizer_steps_total >= max_optimizer_steps:
                    optimizer_budget_reached = True
                    break
                if int(getattr(experiment.train, "max_train_batches", 0) or 0) > 0 and batch_idx >= int(
                    getattr(experiment.train, "max_train_batches", 0)
                ):
                    break
                global_step += 1
                # Edge-step trainers clear gradients before every internal
                # optimizer update. Clearing all parameters here as well was a
                # redundant full-model traversal once per graph sample.
                if not (
                    edge_all
                    and backward_mode
                    in {"edge_step", "edge_step_pi", "sample_hybrid_target_edge_step_pi"}
                ):
                    optimizer.zero_grad(set_to_none=True)
                batch_data = {k: v.to(device) for k, v in batch.model_kwargs.items()}
                targets = {k: v.to(device) for k, v in batch.targets.items()}
                target_masks = {k: v.to(device) for k, v in batch.target_masks.items()}
                if edge_all:
                    _apply_v4_target_loss_weights(batch, target_masks, active_train_cfg, device)
                task_inputs = {
                    head: {k: v.to(device) for k, v in payload.items()}
                    for head, payload in batch.task_inputs.items()
                }
                if epoch == 0 and batch_idx == 0 and _is_main():
                    routing_preview = [
                        {
                            "process_id": meta.get("process_id", ""),
                            "row_index": meta.get("row_index", -1),
                            "target_node_index": meta.get("target_node_index", -1),
                            "target_node_name": meta.get("target_node_name", ""),
                            "target_best_node_index": meta.get("target_best_node_index", -1),
                            "target_best_node_name": meta.get("target_best_node_name", ""),
                            "target_best_differs": meta.get("target_best_differs", False),
                            "tailgas_node_index": meta.get("tailgas_node_index", -1),
                            "tailgas_node_name": meta.get("tailgas_node_name", ""),
                            "tailgas_best_node_index": meta.get("tailgas_best_node_index", -1),
                            "tailgas_best_node_name": meta.get("tailgas_best_node_name", ""),
                            "tailgas_best_differs": meta.get("tailgas_best_differs", False),
                            "target_edge_index": meta.get("target_edge_index", -1),
                            "target_edge_stream_name": meta.get("target_edge_stream_name", ""),
                            "target_edge_src_node": meta.get("target_edge_src_node", ""),
                            "target_edge_dst_node": meta.get("target_edge_dst_node", ""),
                            "target_edge_role": meta.get("target_edge_role", ""),
                            "tailgas_edge_index": meta.get("tailgas_edge_index", -1),
                            "tailgas_edge_stream_name": meta.get("tailgas_edge_stream_name", ""),
                            "tailgas_edge_src_node": meta.get("tailgas_edge_src_node", ""),
                            "tailgas_edge_dst_node": meta.get("tailgas_edge_dst_node", ""),
                            "tailgas_edge_role": meta.get("tailgas_edge_role", ""),
                        }
                        for meta in batch.sample_meta[:5]
                    ]
                    # region agent log
                    _debug_ndjson(
                        run_id=debug_run_id,
                        hypothesis_id="H4",
                        location="scripts/train_process_surrogate.py:train_loop",
                        message="first_batch_fixed_slot_routing_preview",
                        data={"samples": routing_preview},
                    )
                    # endregion
                if edge_all and backward_mode == "edge_step":
                    y_raw = targets["edge_stream"]
                    y_n = _edge_all_normalize(y_raw, y_edge_mean, y_edge_std, device)
                    edge_step_rng = random.Random(int(experiment.seed) + int(epoch) * 1000003 + int(batch_idx))
                    edge_step_debug = bool(getattr(experiment.train, "debug_edge_step", False))
                    result = train_one_batch_edge_step(
                        model=model,
                        batch_data=batch_data,
                        task_inputs=task_inputs,
                        targets={**targets, "edge_stream": y_n},
                        target_masks=target_masks,
                        train_cfg=active_train_cfg,
                        optimizer=optimizer,
                        scaler=scaler,
                        use_amp=use_amp,
                        device=device,
                        grad_clip=float(getattr(active_train_cfg, "gradient_clip_norm", 0.0) or 0.0),
                        edge_export_meta=batch.edge_export_meta,
                        edge_target_columns=list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS),
                        scheduler=scheduler,
                        rng=edge_step_rng,
                        debug=edge_step_debug,
                    )
                    loss_items = result.loss_items
                    loss = loss_items.get("loss_total_final", loss_items["loss_total"])
                    if _is_main():
                        for _k, _v in loss_items.items():
                            if torch.is_tensor(_v):
                                if _v.numel() == 1:
                                    epoch_comp_sums[str(_k)] += _v.detach()
                        if epoch == 0 and batch_idx == 0:
                            _write_edge_step_debug_rows(output_dir, result.debug_rows)
                            if edge_step_debug:
                                first_debug_row = result.debug_rows[0] if result.debug_rows else {}
                                print("[edge_step][debug] first batch shapes", flush=True)
                                print(f"  edge_pred full shape={first_debug_row.get('edge_pred_full_shape')}", flush=True)
                                print(f"  single pred_e shape={first_debug_row.get('single_pred_e_shape')}", flush=True)
                                print(f"  single target_e shape={first_debug_row.get('single_target_e_shape')}", flush=True)
                                print(f"  single mask_e shape={first_debug_row.get('single_mask_e_shape')}", flush=True)
                                print(f"  output dimension unchanged={first_debug_row.get('edge_pred_output_dim')}", flush=True)
                    epoch_loss += float(loss.detach().cpu())
                    steps += 1
                    if _is_main():
                        postfix = {
                            "edge_step_loss_mean": f"{loss_items['edge_step_loss_mean'].item():.4f}",
                            "target_edge_loss_mean": f"{loss_items['target_edge_loss_mean'].item():.4f}",
                            "non_target_edge_loss_mean": f"{loss_items['non_target_edge_loss_mean'].item():.4f}",
                            "edge_update_count": f"{loss_items['edge_update_count'].item():.0f}",
                            "skipped_edge_count": f"{loss_items['skipped_edge_count'].item():.0f}",
                            "gnorm": f"{loss_items['grad_norm'].item():.3f}",
                        }
                        batch_pbar.set_postfix(**postfix)
                        if global_step % experiment.train.log_interval == 0:
                            print(
                                f"[metric step={global_step}] edge_step "
                                f"edge_step_loss_mean={loss_items['edge_step_loss_mean'].item():.6f} "
                                f"target_edge_loss_mean={loss_items['target_edge_loss_mean'].item():.6f} "
                                f"non_target_edge_loss_mean={loss_items['non_target_edge_loss_mean'].item():.6f} "
                                f"edge_update_count={loss_items['edge_update_count'].item():.0f} "
                                f"optimizer_step_count={loss_items['optimizer_step_count'].item():.0f} "
                                f"skipped_edge_count={loss_items['skipped_edge_count'].item():.0f} "
                                f"num_edges_per_batch={loss_items['num_edges_per_batch'].item():.0f} "
                                f"grad_norm={loss_items['grad_norm'].item():.6f}",
                                flush=True,
                            )
                    continue
                if edge_all and backward_mode == "sample_hybrid_target_edge_step_pi":
                    batch_pinn_schedule_scale = _pinn_weight_schedule_scale(
                        experiment.train, epoch, optimizer_steps_total
                    )
                    batch_train_cfg = _scaled_pinn_train_config(
                        experiment.train, batch_pinn_schedule_scale
                    )
                    batch_mass_aux_cfg = getattr(batch_train_cfg, "mass_flow_physical_auxiliary", None)
                    if batch_mass_aux_cfg is not None:
                        batch_mass_aux_cfg.current_epoch = final_epoch
                    y_raw = targets["edge_stream"]
                    cols = list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
                    pi_normalizer = None
                    if y_edge_mean is not None and y_edge_std is not None:
                        pi_normalizer = {
                            "mean": y_edge_mean,
                            "std": y_edge_std,
                            "columns": y_edge_cols or cols,
                            "mass_flow_log1p_mean": y_edge_mass_log_mean,
                            "mass_flow_log1p_std": y_edge_mass_log_std,
                            "mass_flow_log1p_count": y_edge_mass_log_count,
                            "mass_flow_physical_p05": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p05"),
                            "mass_flow_physical_p25": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p25"),
                            "mass_flow_physical_p75": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p75"),
                            "mass_flow_physical_p90": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p90"),
                            "mass_flow_physical_p95": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p95"),
                            "mass_flow_physical_p99": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p99"),
                        }
                    y_pi = _normalize_target_if_needed(
                        y_raw,
                        pi_normalizer,
                        experiment.data,
                        cols,
                    )
                    edge_step_rng = random.Random(int(experiment.seed) + int(epoch) * 1000003 + int(batch_idx))
                    edge_step_pi_debug = bool(getattr(batch_train_cfg, "debug_edge_step_pi", False))
                    result = train_one_batch_sample_hybrid_target_edge_step_pi(
                        model=model,
                        batch_data=batch_data,
                        task_inputs=task_inputs,
                        targets={**targets, "edge_stream": y_pi},
                        target_masks=target_masks,
                        train_cfg=batch_train_cfg,
                        data_cfg=experiment.data,
                        optimizer=optimizer,
                        scaler=scaler,
                        use_amp=use_amp,
                        device=device,
                        grad_clip=float(getattr(active_train_cfg, "gradient_clip_norm", 0.0) or 0.0),
                        edge_export_meta=batch.edge_export_meta,
                        edge_target_columns=cols,
                        scheduler=scheduler,
                        normalizer=pi_normalizer,
                        rng=edge_step_rng,
                        global_step=global_step,
                        total_train_steps=max(1, int(len(train_loader)) * int(experiment.train.epochs)),
                        pinn_schedule_multiplier=batch_pinn_schedule_scale,
                        optimizer_step_budget=(
                            max_optimizer_steps - optimizer_steps_total
                            if max_optimizer_steps > 0
                            else None
                        ),
                        debug=edge_step_pi_debug,
                    )
                    loss_items = result.loss_items
                    optimizer_steps_total += int(
                        round(float(loss_items["optimizer_step_count"].detach().cpu().item()))
                    )
                    if max_optimizer_steps > 0 and optimizer_steps_total >= max_optimizer_steps:
                        optimizer_budget_reached = True
                    loss = loss_items.get("loss_total_final", loss_items["loss_total"])
                    if _is_main():
                        for _k, _v in loss_items.items():
                            if torch.is_tensor(_v):
                                if _v.numel() == 1:
                                    epoch_comp_sums[str(_k)] += _v.detach()
                        if edge_step_pi_debug and epoch == 0 and batch_idx == 0:
                            _write_edge_step_pi_debug_rows(output_dir, result.debug_rows)
                            _write_edge_step_pi_batch_summary(output_dir, loss_items)
                            if edge_step_pi_debug:
                                first_debug_row = result.debug_rows[0] if result.debug_rows else {}
                                print("[sample_hybrid_target_edge_step_pi][debug] first batch shapes", flush=True)
                                print(f"  pred_main_e shape={first_debug_row.get('pred_main_e_shape')}", flush=True)
                                print(f"  target_main_e shape={first_debug_row.get('target_main_e_shape')}", flush=True)
                                print(f"  mask_main_e shape={first_debug_row.get('mask_main_e_shape')}", flush=True)
                                print(f"  rho_pred_e shape={first_debug_row.get('rho_pred_e_shape')}", flush=True)
                                print(f"  h_pred_e shape={first_debug_row.get('h_pred_e_shape')}", flush=True)
                    epoch_loss += float(loss.detach().cpu())
                    steps += 1
                    if _is_main():
                        zero_post = torch.tensor(0.0, device=device)
                        if not bool(getattr(batch_pbar, "disable", False)):
                            postfix = {
                                "hybrid_pi": f"{loss_items['edge_step_pi_loss_mean'].item():.4f}",
                                "main": f"{loss_items.get('loss_main_mean', zero_post).item():.4f}",
                                "rho": f"{loss_items.get('loss_rho_mean', zero_post).item():.4f}",
                                "h": f"{loss_items.get('loss_h_mean', zero_post).item():.4f}",
                                "target_steps": f"{loss_items['target_optimizer_step_count'].item():.0f}",
                                "non_target_steps": f"{loss_items['non_target_optimizer_step_count'].item():.0f}",
                                "node_steps": f"{loss_items['node_optimizer_step_count'].item():.0f}",
                                "skipped": f"{loss_items['skipped_edge_count'].item():.0f}",
                                "gnorm": f"{loss_items['grad_norm'].item():.3f}",
                            }
                            batch_pbar.set_postfix(**postfix)
                        if global_step % experiment.train.log_interval == 0 and not _terminal_log_is_quiet(experiment.train):
                            if _terminal_log_is_verbose(experiment.train):
                                print(
                                    f"[metric step={global_step}] sample_hybrid_target_edge_step_pi "
                                    f"loss={loss_items['edge_step_pi_loss_mean'].item():.6f} "
                                    f"loss_main_mean={loss_items.get('loss_main_mean', zero_post).item():.6f} "
                                    f"loss_rho_mean={loss_items.get('loss_rho_mean', zero_post).item():.6f} "
                                    f"loss_h_mean={loss_items.get('loss_h_mean', zero_post).item():.6f} "
                                    f"loss_node_mass={loss_items.get('loss_node_mass', zero_post).item():.12e} "
                                    f"loss_node_component={loss_items.get('loss_node_component', zero_post).item():.12e} "
                                    f"loss_node_atom={loss_items.get('loss_node_atom', zero_post).item():.12e} "
                                    f"loss_node_energy={loss_items.get('loss_node_energy', zero_post).item():.6f} "
                                    f"loss_volume={loss_items.get('loss_volume_mean', zero_post).item():.6f} "
                                    f"non_target_optimizer_step_count={loss_items['non_target_optimizer_step_count'].item():.0f} "
                                    f"target_optimizer_step_count={loss_items['target_optimizer_step_count'].item():.0f} "
                                    f"node_optimizer_step_count={loss_items['node_optimizer_step_count'].item():.0f} "
                                    f"optimizer_step_count={loss_items['optimizer_step_count'].item():.0f} "
                                    f"target_group_count={loss_items['sample_hybrid_target_group_count'].item():.0f} "
                                    f"non_target_group_count={loss_items['sample_hybrid_non_target_group_count'].item():.0f} "
                                    f"edge_update_count={loss_items['edge_update_count'].item():.0f} "
                                    f"skipped_edge_count={loss_items['skipped_edge_count'].item():.0f} "
                                    f"num_edges_per_batch={loss_items['num_edges_per_batch'].item():.0f} "
                                    f"forward_count={loss_items['edge_step_pi_forward_count'].item():.0f} "
                                    f"target_edge_existing_weight_applied={loss_items['target_edge_existing_weight_applied'].item():.0f} "
                                    f"grad_norm={loss_items['grad_norm'].item():.6f}",
                                    flush=True,
                                )
                            else:
                                progress_bar, progress_pct = _terminal_progress_bar(batch_idx + 1, num_epoch_batches)
                                elapsed_text, eta_text = _terminal_eta(
                                    epoch_started_at,
                                    batch_idx + 1,
                                    num_epoch_batches,
                                )
                                print(
                                    f"[train e={epoch + 1}/{experiment.train.epochs} "
                                    f"step={batch_idx + 1}/{num_epoch_batches} {progress_pct} {progress_bar} "
                                    f"elapsed={elapsed_text} eta={eta_text} global={global_step}] "
                                    f"loss={loss_items['edge_step_pi_loss_mean'].item():.4f} "
                                    f"edge=t{loss_items.get('target_edge_loss_mean', zero_post).item():.3f}"
                                    f"/nt{loss_items.get('non_target_edge_loss_mean', zero_post).item():.3f} "
                                    f"mass=log{loss_items.get('mass_log_loss_mean', zero_post).item():.3f}"
                                    f"+w{loss_items.get('mass_physical_weight_mean', zero_post).item():.3f}"
                                    f"*phys{loss_items.get('mass_physical_loss_mean', zero_post).item():.3f}"
                                    f"={loss_items.get('mass_total_loss_mean', zero_post).item():.3f} "
                                    f"node_w=({loss_items.get('weighted_node_mass', zero_post).item():.6e}/"
                                    f"{loss_items.get('weighted_node_component', zero_post).item():.6e}/"
                                    f"{loss_items.get('weighted_node_atom', zero_post).item():.6e}/"
                                    f"{loss_items.get('weighted_node_energy', zero_post).item():.3f}) "
                                    f"vol={loss_items.get('loss_volume_mean', zero_post).item():.3f} "
                                    f"joint={loss_items.get('node_pinn_joint_loss', zero_post).item():.3f} "
                                    f"anchor={loss_items.get('node_pinn_supervised_anchor_raw', zero_post).item():.3f} "
                                    f"scale_ratio=m{loss_items.get('collapse_mass_flow_pred_true_ratio', zero_post).item():.3f}"
                                    f"/h{loss_items.get('collapse_enthalpy_pred_true_ratio', zero_post).item():.3f} "
                                    f"updates=opt{loss_items['optimizer_step_count'].item():.0f}"
                                    f"/t{loss_items['target_optimizer_step_count'].item():.0f}"
                                    f"/n{loss_items['non_target_optimizer_step_count'].item():.0f}"
                                    f"/node{loss_items['node_optimizer_step_count'].item():.0f} "
                                    f"groups=t{loss_items['sample_hybrid_target_group_count'].item():.0f}"
                                    f"/nt{loss_items['sample_hybrid_non_target_group_count'].item():.0f} "
                                    f"fwd={loss_items['edge_step_pi_forward_count'].item():.0f} "
                                    f"skip={loss_items['skipped_edge_count'].item():.0f} "
                                    f"gnorm={loss_items['grad_norm'].item():.3f}",
                                    flush=True,
                                )
                    if optimizer_budget_reached:
                        break
                    continue
                if edge_all and backward_mode == "edge_step_pi":
                    y_raw = targets["edge_stream"]
                    cols = list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
                    pi_normalizer = None
                    if y_edge_mean is not None and y_edge_std is not None:
                        pi_normalizer = {
                            "mean": y_edge_mean,
                            "std": y_edge_std,
                            "columns": y_edge_cols or cols,
                            "mass_flow_log1p_mean": y_edge_mass_log_mean,
                            "mass_flow_log1p_std": y_edge_mass_log_std,
                            "mass_flow_log1p_count": y_edge_mass_log_count,
                            "mass_flow_physical_p05": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p05"),
                            "mass_flow_physical_p25": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p25"),
                            "mass_flow_physical_p75": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p75"),
                            "mass_flow_physical_p90": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p90"),
                            "mass_flow_physical_p95": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p95"),
                            "mass_flow_physical_p99": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p99"),
                        }
                    y_pi = _normalize_target_if_needed(
                        y_raw,
                        pi_normalizer,
                        experiment.data,
                        cols,
                    )
                    edge_step_rng = random.Random(int(experiment.seed) + int(epoch) * 1000003 + int(batch_idx))
                    edge_step_pi_debug = bool(getattr(active_train_cfg, "debug_edge_step_pi", False))
                    result = train_one_batch_edge_step_pi(
                        model=model,
                        batch_data=batch_data,
                        task_inputs=task_inputs,
                        targets={**targets, "edge_stream": y_pi},
                        target_masks=target_masks,
                        train_cfg=active_train_cfg,
                        data_cfg=experiment.data,
                        optimizer=optimizer,
                        scaler=scaler,
                        use_amp=use_amp,
                        device=device,
                        grad_clip=float(getattr(active_train_cfg, "gradient_clip_norm", 0.0) or 0.0),
                        edge_export_meta=batch.edge_export_meta,
                        edge_target_columns=cols,
                        scheduler=scheduler,
                        normalizer=pi_normalizer,
                        rng=edge_step_rng,
                        global_step=global_step,
                        total_train_steps=max(1, int(len(train_loader)) * int(experiment.train.epochs)),
                        debug=edge_step_pi_debug,
                    )
                    loss_items = result.loss_items
                    loss = loss_items.get("loss_total_final", loss_items["loss_total"])
                    if _is_main():
                        for _k, _v in loss_items.items():
                            if torch.is_tensor(_v):
                                if _v.numel() == 1:
                                    epoch_comp_sums[str(_k)] += _v.detach()
                        if edge_step_pi_debug and epoch == 0 and batch_idx == 0:
                            _write_edge_step_pi_debug_rows(output_dir, result.debug_rows)
                            _write_edge_step_pi_batch_summary(output_dir, loss_items)
                            if edge_step_pi_debug:
                                first_debug_row = result.debug_rows[0] if result.debug_rows else {}
                                print("[edge_step_pi][debug] first batch shapes", flush=True)
                                print(f"  pred_main_e shape={first_debug_row.get('pred_main_e_shape')}", flush=True)
                                print(f"  target_main_e shape={first_debug_row.get('target_main_e_shape')}", flush=True)
                                print(f"  mask_main_e shape={first_debug_row.get('mask_main_e_shape')}", flush=True)
                                print(f"  rho_pred_e shape={first_debug_row.get('rho_pred_e_shape')}", flush=True)
                                print(f"  h_pred_e shape={first_debug_row.get('h_pred_e_shape')}", flush=True)
                    epoch_loss += float(loss.detach().cpu())
                    steps += 1
                    if _is_main():
                        zero_post = torch.tensor(0.0, device=device)
                        if not bool(getattr(batch_pbar, "disable", False)):
                            postfix = {
                                "edge_step_pi": f"{loss_items['edge_step_pi_loss_mean'].item():.4f}",
                                "main": f"{loss_items.get('loss_main_mean', zero_post).item():.4f}",
                                "rho": f"{loss_items.get('loss_rho_mean', zero_post).item():.4f}",
                                "h": f"{loss_items.get('loss_h_mean', zero_post).item():.4f}",
                                "node_mass": f"{loss_items.get('loss_node_mass', zero_post).item():.4f}",
                                "node_component": f"{loss_items.get('loss_node_component', zero_post).item():.4f}",
                                "node_atom": f"{loss_items.get('loss_node_atom', zero_post).item():.4f}",
                                "node_energy": f"{loss_items.get('loss_node_energy', zero_post).item():.4f}",
                                "edge_updates": f"{loss_items['edge_update_count'].item():.0f}",
                                "skipped": f"{loss_items['skipped_edge_count'].item():.0f}",
                                "gnorm": f"{loss_items['grad_norm'].item():.3f}",
                            }
                            batch_pbar.set_postfix(**postfix)
                        if global_step % experiment.train.log_interval == 0:
                            print(
                                f"[metric step={global_step}] edge_step_pi "
                                f"loss={loss_items['edge_step_pi_loss_mean'].item():.6f} "
                                f"loss_main_mean={loss_items.get('loss_main_mean', zero_post).item():.6f} "
                                f"loss_rho_mean={loss_items.get('loss_rho_mean', zero_post).item():.6f} "
                                f"loss_h_mean={loss_items.get('loss_h_mean', zero_post).item():.6f} "
                                f"loss_node_mass={loss_items.get('loss_node_mass', zero_post).item():.12e} "
                                f"loss_node_component={loss_items.get('loss_node_component', zero_post).item():.12e} "
                                f"loss_node_atom={loss_items.get('loss_node_atom', zero_post).item():.12e} "
                                f"loss_node_energy={loss_items.get('loss_node_energy', zero_post).item():.6f} "
                                f"loss_volume={loss_items.get('loss_volume_mean', zero_post).item():.6f} "
                                f"node_mass_valid_count={loss_items.get('node_mass_valid_count', zero_post).item():.0f} "
                                f"node_component_valid_count={loss_items.get('node_component_valid_count', zero_post).item():.0f} "
                                f"node_atom_valid_count={loss_items.get('node_atom_valid_count', zero_post).item():.0f} "
                                f"node_energy_valid_count={loss_items.get('node_energy_valid_count', zero_post).item():.0f} "
                                f"node_mass_skip_count={loss_items.get('node_mass_skip_count', zero_post).item():.0f} "
                                f"node_component_skip_count={loss_items.get('node_component_skip_count', zero_post).item():.0f} "
                                f"node_atom_skip_count={loss_items.get('node_atom_skip_count', zero_post).item():.0f} "
                                f"node_energy_skip_count={loss_items.get('node_energy_skip_count', zero_post).item():.0f} "
                                f"total_before_node={loss_items.get('total_loss_before_node_pinn', zero_post).item():.6f} "
                                f"total_after_node={loss_items.get('total_loss_after_node_pinn', zero_post).item():.6f} "
                                f"loss_frac_penalty_total={loss_items.get('loss_frac_penalty_total', zero_post).item():.6f} "
                                f"frac_penalty_ramp_weight={loss_items.get('frac_penalty_ramp_weight', zero_post).item():.6f} "
                                f"frac_penalty_to_base_ratio={loss_items.get('frac_penalty_to_base_ratio', zero_post).item():.6f} "
                                f"loss_rho_max={loss_items.get('loss_rho_max', zero_post).item():.6f} "
                                f"loss_h_max={loss_items.get('loss_h_max', zero_post).item():.6f} "
                                f"edge_update_count={loss_items['edge_update_count'].item():.0f} "
                                f"optimizer_step_count={loss_items['optimizer_step_count'].item():.0f} "
                                f"skipped_edge_count={loss_items['skipped_edge_count'].item():.0f} "
                                f"num_edges_per_batch={loss_items['num_edges_per_batch'].item():.0f} "
                                f"forward_count={loss_items['edge_step_pi_forward_count'].item():.0f} "
                                f"grad_norm={loss_items['grad_norm'].item():.6f}",
                                flush=True,
                            )
                    continue
                with torch.amp.autocast("cuda", enabled=use_amp):
                    if edge_all:
                        y_raw = targets["edge_stream"]
                        y_n = _edge_all_normalize(y_raw, y_edge_mean, y_edge_std, device)
                        preds = model(batch_data, task_inputs=task_inputs)
                        if not torch.isfinite(preds["y_edge_pred"]).all():
                            nonfinite_skips += 1
                            if _is_main():
                                print(
                                    f"[train][warn] non-finite y_edge_pred epoch={epoch + 1} batch={batch_idx} "
                                    f"skip={nonfinite_skips}/{nonfinite_skip_budget}"
                                )
                            optimizer.zero_grad(set_to_none=True)
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if edge_all and epoch_start_snapshot is not None:
                                _restore_training_state(raw_model, optimizer, epoch_start_snapshot)
                                epoch_recovery_lrs = _scale_optimizer_lr(
                                    optimizer,
                                    factor=float(getattr(experiment.train, "nonfinite_recovery_lr_factor", 0.5)),
                                    min_lr=float(
                                        getattr(
                                            experiment.train,
                                            "nonfinite_recovery_min_lr",
                                            getattr(experiment.train, "scheduler_min_lr", 1.0e-6),
                                        )
                                    ),
                                )
                                scaler = torch.amp.GradScaler(
                                    "cuda",
                                    enabled=experiment.train.mixed_precision and device.type == "cuda",
                                )
                                epoch_aborted_nonfinite = True
                                epoch_nonfinite_reason = "y_edge_pred"
                                if _is_main():
                                    print(
                                        "[train][recover] non-finite y_edge_pred detected; restored epoch-start "
                                        f"model/optimizer state, lr {epoch_recovery_lrs[0]} -> {epoch_recovery_lrs[1]}, "
                                        "aborting this epoch before validation.",
                                        flush=True,
                                    )
                                break
                            if nonfinite_skips > nonfinite_skip_budget:
                                raise RuntimeError(
                                    f"Too many non-finite edge_all predictions in one epoch "
                                    f"({nonfinite_skips} > {nonfinite_skip_budget}). "
                                    "Lower learning_rate, use smooth_l1 instead of mse, or reduce model size / batch_size."
                                )
                            continue
                        loss_items = compute_training_loss(
                            preds,
                            {**targets, "edge_stream": y_n},
                            target_masks,
                            experiment.train,
                            experiment.data,
                            **_edge_all_loss_kwargs(
                                batch,
                                experiment.data,
                                y_edge_mean=y_edge_mean,
                                y_edge_std=y_edge_std,
                            ),
                        )
                        _check_edge_all_loss_formula(
                            output_dir=output_dir,
                            epoch=epoch + 1,
                            global_step=global_step,
                            batch_idx=batch_idx,
                            loss_items=loss_items,
                            preds=preds,
                            targets=targets,
                            sample_meta=batch.sample_meta,
                        )
                        if _is_main():
                            _append_incident_loss_diagnostic_rows(
                                output_dir,
                                split="train",
                                epoch=epoch + 1,
                                step=global_step,
                                y_pred=preds["y_edge_pred"],
                                y_true=y_n,
                                y_mask=target_masks["edge_stream"],
                                export_meta=batch.edge_export_meta,
                                edge_index=batch_data.get("edge_index"),
                                train_cfg=experiment.train,
                                loss_scalars=loss_items,
                            )
                        loss = loss_items.get("loss_total_final", loss_items["loss_total"])
                        # loss contribution diagnostics (no model change)
                        cols = list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
                        ym = target_masks["edge_stream"]
                        m2 = ym.unsqueeze(-1)
                        se_norm = ((preds["y_edge_pred"] - y_n) ** 2) * m2
                        if y_edge_mean is not None and y_edge_std is not None:
                            _m = y_edge_mean.to(device=device, dtype=preds["y_edge_pred"].dtype).view(1, -1)
                            _s = y_edge_std.to(device=device, dtype=preds["y_edge_pred"].dtype).view(1, -1)
                            yp_orig = preds["y_edge_pred"] * _s + _m
                        else:
                            yp_orig = preds["y_edge_pred"]
                        se_raw = ((yp_orig - y_raw) ** 2) * m2
                        prop_loss_norm = se_norm.sum(dim=0).detach().cpu().tolist()
                        prop_loss_raw = se_raw.sum(dim=0).detach().cpu().tolist()
                        prop_count_scalar = float(ym.sum().detach().cpu())
                        for j, pn in enumerate(cols):
                            edge_all_loss_prop_norm[str(pn)] += float(prop_loss_norm[j])
                            edge_all_loss_prop_raw[str(pn)] += float(prop_loss_raw[j])
                            edge_all_loss_prop_count[str(pn)] += prop_count_scalar
                        # Group-level ratios
                        flow_idx = [i for i, c in enumerate(cols) if c in {"Vol_Flow", "Mass_Flow", "Mole_Flow"}]
                        frac_idx = [i for i, c in enumerate(cols) if str(c).startswith("Frac_")]
                        if flow_idx:
                            edge_all_loss_group_norm["flow_norm"] += float(
                                se_norm[:, flow_idx].sum().detach().cpu()
                            )
                        if frac_idx:
                            edge_all_loss_group_norm["frac_norm"] += float(
                                se_norm[:, frac_idx].sum().detach().cpu()
                            )
                        export_meta = batch.edge_export_meta
                        if export_meta is not None:
                            out_mask = torch.tensor(
                                [float(v) for v in export_meta.is_output_edge],
                                device=device,
                                dtype=preds["y_edge_pred"].dtype,
                            ).unsqueeze(-1)
                            edge_all_loss_group_norm["output_norm"] += float(
                                (se_norm * out_mask).sum().detach().cpu()
                            )
                            ans_mask = torch.tensor(
                                [1.0 if str(v or "").strip() else 0.0 for v in export_meta.answer_task_names],
                                device=device,
                                dtype=preds["y_edge_pred"].dtype,
                            ).unsqueeze(-1)
                            edge_all_loss_group_norm["answer_norm"] += float(
                                (se_norm * ans_mask).sum().detach().cpu()
                            )
                    else:
                        preds = model(batch_data, task_inputs=task_inputs)
                        pt = preds.get("target")
                        if pt is not None and not torch.isfinite(pt).all():
                            nonfinite_skips += 1
                            if _is_main():
                                print(
                                    f"[train][warn] non-finite pred target epoch={epoch + 1} batch={batch_idx} "
                                    f"skip={nonfinite_skips}/{nonfinite_skip_budget}"
                                )
                            optimizer.zero_grad(set_to_none=True)
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if nonfinite_skips > nonfinite_skip_budget:
                                raise RuntimeError(
                                    f"Too many non-finite multitask predictions in one epoch ({nonfinite_skips} > {nonfinite_skip_budget})."
                                )
                            continue
                        if not target_only:
                            ptl = preds.get("tailgas")
                            if ptl is not None and not torch.isfinite(ptl).all():
                                nonfinite_skips += 1
                                if _is_main():
                                    print(
                                        f"[train][warn] non-finite pred tailgas epoch={epoch + 1} batch={batch_idx} "
                                        f"skip={nonfinite_skips}/{nonfinite_skip_budget}"
                                    )
                                optimizer.zero_grad(set_to_none=True)
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                if nonfinite_skips > nonfinite_skip_budget:
                                    raise RuntimeError(
                                        f"Too many non-finite multitask predictions in one epoch ({nonfinite_skips} > {nonfinite_skip_budget})."
                                    )
                                continue
                        loss_items = compute_training_loss(
                            preds, targets, target_masks, experiment.train, experiment.data
                        )
                        loss_target = loss_items["loss_target"]
                        if target_only:
                            loss = loss_items["loss_total"]
                        else:
                            loss_tailgas = loss_items["loss_tailgas"]
                            if experiment.train.debug_task_mode == "target_only":
                                loss = loss_target
                            elif experiment.train.debug_task_mode == "tailgas_only":
                                loss = loss_tailgas
                            else:
                                loss = loss_items["loss_total"]
                debug_epoch = experiment.train.debug_mode and epoch < int(experiment.train.debug_epochs)
                if debug_epoch and edge_all and batch_idx == 0 and _is_main():
                    enc_dbg = raw_model.encode(batch_data)
                    struct = batch_data.get("edge_struct_attr")
                    if struct is None:
                        struct = batch_data.get("edge_oper")
                    yp = preds["y_edge_pred"]
                    ym = target_masks["edge_stream"]
                    validate_edge_all_batch(
                        y_edge_pred=yp,
                        y_edge_true=y_n,
                        y_edge_mask=ym,
                        edge_struct=struct,
                        edge_struct_dim=int(encoder_cfg.edge_struct_dim),
                        answer_edge_pos=batch.answer_edge_pos,
                    )
                    cols = list(batch.edge_target_columns or [])
                    print("[model/debug][edge_all] first batch")
                    print(f"  h_node shape={tuple(enc_dbg.node_embeddings.shape)}")
                    print(f"  edge_struct_attr shape={tuple(struct.shape)}")
                    print(f"  h_src shape={tuple(enc_dbg.node_embeddings[batch_data['edge_index'][0]].shape)}")
                    print(f"  h_dst shape={tuple(enc_dbg.node_embeddings[batch_data['edge_index'][1]].shape)}")
                    dec_in = torch.cat(
                        [
                            enc_dbg.node_embeddings[batch_data["edge_index"][0]],
                            enc_dbg.node_embeddings[batch_data["edge_index"][1]],
                            struct,
                        ],
                        dim=-1,
                    )
                    print(f"  edge_decoder input shape={tuple(dec_in.shape)}")
                    print(f"  y_edge_pred shape={tuple(yp.shape)}")
                    print(f"  y_edge_true shape={tuple(y_raw.shape)}")
                    print(f"  y_edge_mask shape={tuple(ym.shape)}")
                    print(f"  y_edge_mask sum={float(ym.sum().detach().cpu()):.1f}")
                    print(f"  edge_target_columns={cols}")
                    print(f"  answer_edge_pos={batch.answer_edge_pos}")
                    ans_path = (
                        experiment.project_root / experiment.data.canonical_graph_spec_v3_dir / "target_answer_edges.csv"
                    ).resolve()
                    answers_df = pd.read_csv(ans_path) if ans_path.is_file() else None
                    h2_i = co2_i = None
                    if answers_df is not None and batch.answer_edge_pos and batch.sample_meta:
                        meta0 = batch.sample_meta[0]
                        pid = str(meta0.get("process_id", ""))
                        pnum = int(pid.replace("Process", "").strip()) if pid.startswith("Process") else int(pid)
                        ap0 = batch.answer_edge_pos[0]
                        for slot, task_key in (("target", "target_h2"), ("tailgas", "tailgas_co2")):
                            if slot not in ap0:
                                continue
                            rows = answers_df[
                                (answers_df["process_id"] == pnum) & (answers_df["task_name"] == task_key)
                            ]
                            if len(rows) != 1:
                                continue
                            tc = str(rows.iloc[0]["target_column"])
                            ci, rule = resolve_answer_target_column_index(
                                task_name=task_key,
                                target_column=tc,
                                edge_target_columns=cols,
                            )
                            print(f"  answer_column_map {task_key}: target_answer_edges={tc!r} -> edge_col_idx={ci} ({rule})")
                            if task_key == "target_h2":
                                h2_i = ci
                            else:
                                co2_i = ci
                        if "target" in ap0 and h2_i is not None:
                            r = int(ap0["target"])
                            print(
                                f"  target_h2 pred/true sample: pred={float(yp[r, h2_i].detach().cpu()):.6f} "
                                f"true={float(y_raw[r, h2_i].detach().cpu()):.6f} (norm true={float(y_n[r, h2_i].detach().cpu()):.6f})"
                            )
                        if "tailgas" in ap0 and co2_i is not None:
                            r = int(ap0["tailgas"])
                            print(
                                f"  tailgas_co2 pred/true sample: pred={float(yp[r, co2_i].detach().cpu()):.6f} "
                                f"true={float(y_raw[r, co2_i].detach().cpu()):.6f} (norm true={float(y_n[r, co2_i].detach().cpu()):.6f})"
                            )
                    _, denom = masked_edge_regression_loss(
                        yp, y_n, ym, loss_type=str(getattr(experiment.train, "loss_type_edge_all", "mse"))
                    )
                    mae_n = masked_edge_mae(yp, y_n, ym)
                    print("[loss/debug][edge_all] first batch")
                    print(f"  loss_edge={float(loss_items['loss_edge'].detach().cpu()):.6f}")
                    print(f"  masked element count={float(denom.detach().cpu()):.1f}")
                    print(
                        f"  pred mean/std={float(yp.mean().detach().cpu()):.6f}/{_safe_std(yp):.6f} "
                        f"true(norm) mean/std={float(y_n.mean().detach().cpu()):.6f}/{_safe_std(y_n):.6f}"
                    )
                    print(
                        f"  pred has NaN/Inf={not torch.isfinite(yp).all().item()} "
                        f"true has NaN/Inf={not torch.isfinite(y_raw).all().item()}"
                    )
                    print(f"  mae(norm)={float(mae_n.detach().cpu()):.6f}")
                    if getattr(experiment.train, "edge_all_print_first_batch", False):
                        eb = batch_data["edge_batch"].detach().cpu().tolist()
                        n_graphs = int(batch_data["batch"].max().item()) + 1 if batch_data["batch"].numel() else 0
                        ec = Counter(eb)
                        per_graph_edges = [int(ec[i]) for i in range(n_graphs)]
                        ap_shapes = []
                        if batch.answer_edge_pos:
                            for gi, pos_map in enumerate(batch.answer_edge_pos):
                                off = sum(per_graph_edges[:gi]) if gi < len(per_graph_edges) else 0
                                local_map = {slot: int(gpos) - int(off) for slot, gpos in pos_map.items()}
                                pred_row_shapes = {
                                    slot: list(yp[int(gpos)].shape) for slot, gpos in pos_map.items()
                                }
                                ap_shapes.append(
                                    {
                                        "graph_index": gi,
                                        "num_edges_this_graph": per_graph_edges[gi]
                                        if gi < len(per_graph_edges)
                                        else None,
                                        "global_answer_positions": dict(pos_map),
                                        "local_answer_edge_index": local_map,
                                        "y_edge_pred_row_shape_by_slot": pred_row_shapes,
                                    }
                                )
                        dbg_path = output_dir / "first_batch_debug.json"
                        dbg_path.write_text(
                            json.dumps(
                                {
                                    "y_edge_pred_shape": list(yp.shape),
                                    "y_edge_true_shape": list(y_raw.shape),
                                    "y_edge_mask_shape": list(ym.shape),
                                    "edge_struct_shape": list(struct.shape),
                                    "num_graphs": n_graphs,
                                    "per_graph_num_edges": per_graph_edges,
                                    "answer_edge_pos_global": batch.answer_edge_pos,
                                    "answer_edge_pos_batch_audit": ap_shapes,
                                    "edge_target_columns": cols,
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                        print(f"  wrote {dbg_path}")
                if debug_epoch and not edge_all:
                    if batch_idx == 0 and _is_main() and not target_only:
                        if experiment.train.debug_task_mode == "target_only":
                            print(
                                "[DEBUG][note] debug_task_mode=target_only: backward는 loss_target만 사용합니다. "
                                "tailgas head 가중치는 거의 갱신되지 않아 pred_tailgas가 배치마다 상수에 가깝고 corr=nan·tg_* 지표는 "
                                "정상 학습 여부로 보지 마세요."
                            )
                        elif experiment.train.debug_task_mode == "tailgas_only":
                            print(
                                "[DEBUG][note] debug_task_mode=tailgas_only: backward는 loss_tailgas만 사용합니다. "
                                "target head는 거의 갱신되지 않아 pred_target이 상수에 가깝고 t_* 지표는 참고용입니다."
                            )
                    y_target = targets["target"]
                    pred_target = preds["target"]
                    target_var = float(y_target.detach().float().var(unbiased=False).cpu()) if y_target.numel() > 1 else 0.0
                    tailgas_var = 0.0
                    print(
                        f"[DEBUG][target] batch={batch_idx} min={float(y_target.min().detach().cpu()):.6f} "
                        f"max={float(y_target.max().detach().cpu()):.6f} mean={float(y_target.mean().detach().cpu()):.6f} std={_safe_std(y_target):.6f}"
                    )
                    if not target_only:
                        y_tailgas = targets["tailgas"]
                        pred_tailgas = preds["tailgas"]
                        tailgas_var = float(y_tailgas.detach().float().var(unbiased=False).cpu()) if y_tailgas.numel() > 1 else 0.0
                        print(
                            f"[DEBUG][tailgas] batch={batch_idx} min={float(y_tailgas.min().detach().cpu()):.6f} "
                            f"max={float(y_tailgas.max().detach().cpu()):.6f} mean={float(y_tailgas.mean().detach().cpu()):.6f} std={_safe_std(y_tailgas):.6f}"
                        )
                    print(f"[DEBUG] pred_target shape: {tuple(pred_target.shape)}")
                    print(f"[DEBUG] y_target shape: {tuple(y_target.shape)}")
                    if not target_only:
                        print(f"[DEBUG] pred_tailgas shape: {tuple(pred_tailgas.shape)}")
                        print(f"[DEBUG] y_tailgas shape: {tuple(y_tailgas.shape)}")
                        print(
                            f"[DEBUG][label-check] allclose={torch.allclose(y_target, y_tailgas)} "
                            f"corr={_safe_corrcoef(y_target, y_tailgas):.6f} "
                            f"cos={_safe_cosine_similarity(y_target, y_tailgas):.6f} "
                            f"y_target_id={id(y_target)} y_tailgas_id={id(y_tailgas)}"
                        )
                        print(
                            f"[DEBUG][pred-check] allclose={torch.allclose(pred_target, pred_tailgas)} "
                            f"corr={_safe_corrcoef(pred_target, pred_tailgas):.6f} "
                            f"cos={_safe_cosine_similarity(pred_target, pred_tailgas):.6f} "
                            f"pred_target_id={id(pred_target)} pred_tailgas_id={id(pred_tailgas)} "
                            f"pred_target_std={_safe_std(pred_target):.6f} pred_tailgas_std={_safe_std(pred_tailgas):.6f}"
                        )
                    if batch_idx < int(experiment.train.debug_vector_batches):
                        print(f"[DEBUG][y_target[:5]] {y_target[:5].view(-1).detach().cpu().tolist()}")
                        print(f"[DEBUG][pred_target[:5]] {pred_target[:5].view(-1).detach().cpu().tolist()}")
                        if not target_only:
                            print(f"[DEBUG][y_tailgas[:5]] {y_tailgas[:5].view(-1).detach().cpu().tolist()}")
                            print(f"[DEBUG][pred_tailgas[:5]] {pred_tailgas[:5].view(-1).detach().cpu().tolist()}")
                        for meta in batch.sample_meta[:3]:
                            if meta.get("target_edge_stream_name"):
                                route = (
                                    "[DEBUG][edge-route] "
                                    f"process={meta.get('process_id')} row={meta.get('row_index')} "
                                    f"target={meta.get('target_edge_stream_name')} "
                                    f"({meta.get('target_edge_src_node')}->{meta.get('target_edge_dst_node')}, "
                                    f"role={meta.get('target_edge_role')})"
                                )
                            else:
                                route = (
                                    "[DEBUG][node-route] "
                                    f"process={meta.get('process_id')} row={meta.get('row_index')} "
                                    f"target={meta.get('target_node_name')} "
                                    f"(best={meta.get('target_best_node_name')}, mode={meta.get('target_route_mode')}, "
                                    f"reasons={meta.get('target_route_reasons')})"
                                )
                            if not target_only:
                                if meta.get("tailgas_edge_stream_name"):
                                    route += (
                                        f" tailgas={meta.get('tailgas_edge_stream_name')} "
                                        f"({meta.get('tailgas_edge_src_node')}->{meta.get('tailgas_edge_dst_node')}, "
                                        f"role={meta.get('tailgas_edge_role')})"
                                    )
                                else:
                                    route += (
                                        f" tailgas={meta.get('tailgas_node_name')} "
                                        f"(best={meta.get('tailgas_best_node_name')}, mode={meta.get('tailgas_route_mode')}, "
                                        f"reasons={meta.get('tailgas_route_reasons')})"
                                    )
                            print(route)
                    metric_in = (
                        f"[DEBUG][metric-input] target_pred_id={id(pred_target)} target_true_id={id(y_target)}"
                    )
                    if not target_only:
                        metric_in += (
                            f" tailgas_pred_id={id(pred_tailgas)} tailgas_true_id={id(y_tailgas)}"
                        )
                    print(metric_in)
                    heat_mask = target_masks.get("heat_duty")
                    if heat_mask is not None:
                        heat_mask_sum = float(heat_mask.sum().detach().cpu())
                        print(f"[DEBUG] heat mask sum: {heat_mask_sum:.6f}")
                        if heat_mask_sum == 0.0:
                            print("[DEBUG] heat mask sum is zero for this batch")
                    y_target_orig = _slot_denormalize(
                        y_target,
                        slot_name="target",
                        mean_map=target_mean,
                        std_map=target_std,
                        fixed_cfg=experiment.data.fixed_tasks["target"],
                        normalize_targets=experiment.data.normalize_targets,
                    )
                    pred_target_orig = _slot_denormalize(
                        pred_target,
                        slot_name="target",
                        mean_map=target_mean,
                        std_map=target_std,
                        fixed_cfg=experiment.data.fixed_tasks["target"],
                        normalize_targets=experiment.data.normalize_targets,
                    )
                    if not target_only:
                        y_tailgas_orig = _slot_denormalize(
                            y_tailgas,
                            slot_name="tailgas",
                            mean_map=target_mean,
                            std_map=target_std,
                            fixed_cfg=experiment.data.fixed_tasks["tailgas"],
                            normalize_targets=experiment.data.normalize_targets,
                        )
                        pred_tailgas_orig = _slot_denormalize(
                            pred_tailgas,
                            slot_name="tailgas",
                            mean_map=target_mean,
                            std_map=target_std,
                            fixed_cfg=experiment.data.fixed_tasks["tailgas"],
                            normalize_targets=experiment.data.normalize_targets,
                        )
                    print(
                        f"[DEBUG][norm-stats] y_target(mean={float(y_target.mean().detach().cpu()):.6f}, std={_safe_std(y_target):.6f}) "
                        f"pred_target(mean={float(pred_target.mean().detach().cpu()):.6f}, std={_safe_std(pred_target):.6f})"
                    )
                    if not target_only:
                        print(
                            f"[DEBUG][norm-stats] y_tailgas(mean={float(y_tailgas.mean().detach().cpu()):.6f}, std={_safe_std(y_tailgas):.6f}) "
                            f"pred_tailgas(mean={float(pred_tailgas.mean().detach().cpu()):.6f}, std={_safe_std(pred_tailgas):.6f})"
                        )
                    print(
                        f"[DEBUG][orig-stats] y_target(mean={float(y_target_orig.mean().detach().cpu()):.6f}, std={_safe_std(y_target_orig):.6f}) "
                        f"pred_target(mean={float(pred_target_orig.mean().detach().cpu()):.6f}, std={_safe_std(pred_target_orig):.6f})"
                    )
                    if not target_only:
                        print(
                            f"[DEBUG][orig-stats] y_tailgas(mean={float(y_tailgas_orig.mean().detach().cpu()):.6f}, std={_safe_std(y_tailgas_orig):.6f}) "
                            f"pred_tailgas(mean={float(pred_tailgas_orig.mean().detach().cpu()):.6f}, std={_safe_std(pred_tailgas_orig):.6f})"
                        )
                    scaler_msg = (
                        f"[DEBUG][scaler-check] target_mean={float((target_mean or {}).get('target', 0.0)):.6f} "
                        f"target_std={float((target_std or {}).get('target', 1.0)):.6f}"
                    )
                    if not target_only:
                        scaler_msg += (
                            f" tailgas_mean={float((target_mean or {}).get('tailgas', 0.0)):.6f} "
                            f"tailgas_std={float((target_std or {}).get('tailgas', 1.0)):.6f} "
                            f"same_scaler_values={float((target_mean or {}).get('target', 0.0)) == float((target_mean or {}).get('tailgas', 0.0)) and float((target_std or {}).get('target', 1.0)) == float((target_std or {}).get('tailgas', 1.0))}"
                        )
                    print(scaler_msg)
                if edge_all:
                    batch_target_mae = float("nan")
                    batch_target_rmse = float("nan")
                    batch_target_r2 = float("nan")
                    batch_tailgas_mae = float("nan")
                    batch_tailgas_rmse = float("nan")
                    batch_tailgas_r2 = float("nan")
                else:
                    with torch.no_grad():
                        pred_target_eval, y_target_eval = _accumulate_task_metrics(
                            epoch_metric_accum["target"],
                            pred=preds["target"],
                            target=targets["target"],
                            slot_name="target",
                            target_mean=target_mean,
                            target_std=target_std,
                            data_cfg=experiment.data,
                        )
                        if not target_only:
                            pred_tailgas_eval, y_tailgas_eval = _accumulate_task_metrics(
                                epoch_metric_accum["tailgas"],
                                pred=preds["tailgas"],
                                target=targets["tailgas"],
                                slot_name="tailgas",
                                target_mean=target_mean,
                                target_std=target_std,
                                data_cfg=experiment.data,
                            )
                        target_stats = _finalize_task_metrics(
                            {
                                "sae": float((pred_target_eval - y_target_eval).abs().sum().cpu()),
                                "sse": float(((pred_target_eval - y_target_eval) ** 2).sum().cpu()),
                                "sum_y": float(y_target_eval.sum().cpu()),
                                "sum_y2": float((y_target_eval**2).sum().cpu()),
                                "n": float(y_target_eval.numel()),
                            }
                        )
                        batch_target_mae = target_stats["mae"]
                        batch_target_rmse = target_stats["rmse"]
                        batch_target_r2 = target_stats["r2"]
                        if not target_only:
                            tailgas_stats = _finalize_task_metrics(
                                {
                                    "sae": float((pred_tailgas_eval - y_tailgas_eval).abs().sum().cpu()),
                                    "sse": float(((pred_tailgas_eval - y_tailgas_eval) ** 2).sum().cpu()),
                                    "sum_y": float(y_tailgas_eval.sum().cpu()),
                                    "sum_y2": float((y_tailgas_eval**2).sum().cpu()),
                                    "n": float(y_tailgas_eval.numel()),
                                }
                            )
                            batch_tailgas_mae = tailgas_stats["mae"]
                            batch_tailgas_rmse = tailgas_stats["rmse"]
                            batch_tailgas_r2 = tailgas_stats["r2"]
                        if debug_epoch:
                            valid_target = int(targets["target"].numel())
                            if batch_target_r2 != batch_target_r2:
                                print(
                                    f"[DEBUG][r2-nan] batch={batch_idx} task=target valid={valid_target} "
                                    f"var={target_var:.12f} process_ids={[m.get('process_id') for m in batch.sample_meta[:5]]}"
                                )
                            if not target_only:
                                valid_tailgas = int(targets["tailgas"].numel())
                                if batch_tailgas_r2 != batch_tailgas_r2:
                                    print(
                                        f"[DEBUG][r2-nan] batch={batch_idx} task=tailgas valid={valid_tailgas} "
                                        f"var={tailgas_var:.12f} process_ids={[m.get('process_id') for m in batch.sample_meta[:5]]}"
                                    )
                if not torch.isfinite(loss):
                    nonfinite_skips += 1
                    if _is_main():
                        print(
                            f"[train][warn] non-finite loss epoch={epoch + 1} batch={batch_idx} "
                            f"skip={nonfinite_skips}/{nonfinite_skip_budget}"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if edge_all and epoch_start_snapshot is not None:
                        _restore_training_state(raw_model, optimizer, epoch_start_snapshot)
                        epoch_recovery_lrs = _scale_optimizer_lr(
                            optimizer,
                            factor=float(getattr(experiment.train, "nonfinite_recovery_lr_factor", 0.5)),
                            min_lr=float(
                                getattr(
                                    experiment.train,
                                    "nonfinite_recovery_min_lr",
                                    getattr(experiment.train, "scheduler_min_lr", 1.0e-6),
                                )
                            ),
                        )
                        scaler = torch.amp.GradScaler(
                            "cuda",
                            enabled=experiment.train.mixed_precision and device.type == "cuda",
                        )
                        epoch_aborted_nonfinite = True
                        epoch_nonfinite_reason = "loss"
                        if _is_main():
                            print(
                                "[train][recover] non-finite loss detected; restored epoch-start "
                                f"model/optimizer state, lr {epoch_recovery_lrs[0]} -> {epoch_recovery_lrs[1]}, "
                                "aborting this epoch before validation.",
                                flush=True,
                            )
                        break
                    if nonfinite_skips > nonfinite_skip_budget:
                        raise RuntimeError(
                            f"Too many non-finite losses in one epoch ({nonfinite_skips} > {nonfinite_skip_budget}). "
                            "Lower learning_rate, change loss_type, or reduce batch_size."
                        )
                    continue
                if _is_main():
                    for _k, _v in loss_items.items():
                        if _v.numel() == 1:
                            epoch_comp_sums[str(_k)] += _v.detach()
                grad_norm: float | None = None
                scaler.scale(loss).backward()
                if experiment.train.gradient_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), experiment.train.gradient_clip_norm
                        )
                    )
                    if edge_all and not math.isfinite(grad_norm) and epoch_start_snapshot is not None:
                        optimizer.zero_grad(set_to_none=True)
                        _restore_training_state(raw_model, optimizer, epoch_start_snapshot)
                        epoch_recovery_lrs = _scale_optimizer_lr(
                            optimizer,
                            factor=float(getattr(experiment.train, "nonfinite_recovery_lr_factor", 0.5)),
                            min_lr=float(
                                getattr(
                                    experiment.train,
                                    "nonfinite_recovery_min_lr",
                                    getattr(experiment.train, "scheduler_min_lr", 1.0e-6),
                                )
                            ),
                        )
                        scaler = torch.amp.GradScaler(
                            "cuda",
                            enabled=experiment.train.mixed_precision and device.type == "cuda",
                        )
                        epoch_aborted_nonfinite = True
                        epoch_nonfinite_reason = "grad_norm"
                        if _is_main():
                            print(
                                "[train][recover] non-finite gradient norm detected; restored epoch-start "
                                f"model/optimizer state, lr {epoch_recovery_lrs[0]} -> {epoch_recovery_lrs[1]}, "
                                "aborting this epoch before validation.",
                                flush=True,
                            )
                        break
                total_norm = _grad_l2_norm(model)
                if debug_epoch:
                    print(f"[DEBUG] grad_norm={total_norm:.12f}")
                    if float(loss.detach().cpu()) == 0.0:
                        print("[WARN] Zero loss batch detected")
                        print(f"target std: {_safe_std(targets['target']):.12f}")
                        if not target_only:
                            print(f"tailgas std: {_safe_std(targets['tailgas']):.12f}")
                if steps == 0:
                    no_grad_names = [
                        name
                        for name, param in raw_model.named_parameters()
                        if param.requires_grad and param.grad is None
                    ]
                    # region agent log
                    _debug_ndjson(
                        run_id=debug_run_id,
                        hypothesis_id="H1_H3",
                        location="scripts/train_process_surrogate.py:train_loop",
                        message="first_step_grad_audit",
                        data={
                            "rank": _rank(),
                            "epoch": epoch + 1,
                            "batch_idx": batch_idx,
                            "pred_keys": sorted(list(preds.keys())),
                            "target_keys": sorted(list(targets.keys())),
                            "task_input_keys": sorted(list(task_inputs.keys())),
                            "loss_total": float(loss.detach().cpu()),
                            "no_grad_param_names": no_grad_names,
                        },
                    )
                    # endregion
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += float(loss.detach().cpu())
                steps += 1

                if _is_main():
                    if edge_all:
                        z_loss = loss_items["loss_total"] * 0
                        postfix = {
                            "loss_total_final": f"{loss.item():.4f}",
                            "loss_edge_all": f"{loss_items.get('loss_edge_all', loss_items['loss_edge']).item():.4f}",
                            "loss_primary_frac": f"{loss_items.get('loss_primary_frac', z_loss).item():.4f}",
                            "loss_all_edge_r2": f"{loss_items.get('loss_all_edge_r2', z_loss).item():.4f}",
                            "weighted_edge_all": f"{loss_items.get('weighted_edge_all', z_loss).item():.4f}",
                            "weighted_primary_frac": f"{loss_items.get('weighted_primary_frac', z_loss).item():.4f}",
                            "weighted_all_edge_r2": f"{loss_items.get('weighted_all_edge_r2', z_loss).item():.4f}",
                            "loss_amount_auxiliary": f"{loss_items.get('loss_amount_auxiliary', z_loss).item():.4f}",
                            "loss_extra_unexpected": f"{loss_items.get('loss_extra_unexpected', z_loss).item():.4g}",
                        }
                        if "primary_frac_count" in loss_items:
                            postfix["pcnt"] = f"{loss_items['primary_frac_count'].item():.0f}"
                        if "primary_frac_std_ratio_norm" in loss_items:
                            postfix["pstd"] = f"{loss_items['primary_frac_std_ratio_norm'].item():.2f}"
                    else:
                        postfix = {
                            "loss": f"{loss.item():.4f}",
                            "tgt": f"{loss_items['loss_target'].item():.4f}",
                            "t_mae": f"{batch_target_mae:.3f}",
                            "t_rmse": f"{batch_target_rmse:.3f}",
                            "t_r2": f"{batch_target_r2:.3f}",
                        }
                    if not edge_all and not target_only:
                        postfix["tg"] = f"{loss_items['loss_tailgas'].item():.4f}"
                        postfix["tg_mae"] = f"{batch_tailgas_mae:.3f}"
                        postfix["tg_rmse"] = f"{batch_tailgas_rmse:.3f}"
                        postfix["tg_r2"] = f"{batch_tailgas_r2:.3f}"
                    if decoder_tasks_enabled(experiment.data):
                        for cat in active_decoder_cats:
                            k = f"loss_{cat}"
                            if k in loss_items:
                                postfix[cat] = f"{loss_items[k].item():.4f}"
                    elif "loss_aux" in loss_items:
                        postfix["aux"] = f"{loss_items['loss_aux'].item():.4f}"
                    if "loss_edge_stream" in loss_items:
                        postfix["es"] = f"{loss_items['loss_edge_stream'].item():.4f}"
                    if grad_norm is not None:
                        postfix["gnorm"] = f"{grad_norm:.3f}"
                    batch_pbar.set_postfix(**postfix)
                    if loss.item() >= experiment.train.debug_loss_threshold:
                        _log_problematic_batch(
                            output_dir=output_dir,
                            epoch=epoch + 1,
                            global_step=global_step,
                            batch_idx=batch_idx,
                            loss_items=loss_items,
                            preds=preds,
                            targets=targets,
                            sample_meta=batch.sample_meta,
                            grad_norm=grad_norm,
                        )
                    if global_step % experiment.train.log_interval == 0:
                        if edge_all:
                            z_loss = loss_items["loss_total"] * 0
                            le = float(
                                loss_items.get("loss_edge_all", loss_items.get("loss_edge", z_loss)).detach().cpu()
                            )
                            lp = float(loss_items.get("loss_primary_frac", z_loss).detach().cpu())
                            lr2 = float(
                                loss_items.get("loss_all_edge_r2", z_loss).detach().cpu()
                            )
                            we = float(loss_items.get("weighted_edge_all", z_loss).detach().cpu())
                            wp = float(loss_items.get("weighted_primary_frac", z_loss).detach().cpu())
                            wr2 = float(loss_items.get("weighted_all_edge_r2", z_loss).detach().cpu())
                            la = float(loss_items.get("loss_amount_auxiliary", z_loss).detach().cpu())
                            lx = float(loss_items.get("loss_extra_unexpected", z_loss).detach().cpu())
                            lf = float(
                                loss_items.get("loss_total_final", loss_items["loss_total"]).detach().cpu()
                            )
                            pcnt = float(loss_items.get("primary_frac_count", z_loss).detach().cpu())
                            pstd = float(
                                loss_items.get("primary_frac_std_ratio_norm", z_loss).detach().cpu()
                            )
                            msg = (
                                f"[metric step={global_step}] edge_all primary-focused "
                                f"loss_total_final={lf:.6f} "
                                f"loss_edge_all={le:.6f} loss_primary_frac={lp:.6f} "
                                f"loss_all_edge_r2={lr2:.6f} "
                                f"weighted_edge_all={we:.6f} weighted_primary_frac={wp:.6f} "
                                f"weighted_all_edge_r2={wr2:.6f} "
                                f"loss_amount_auxiliary={la:.6f} loss_extra_unexpected={lx:.6f} "
                                f"primary_count={pcnt:.0f} primary_std_ratio_norm={pstd:.3f}"
                            )
                        else:
                            msg = (
                                f"[metric step={global_step}] "
                                f"target(mae={batch_target_mae:.6f}, rmse={batch_target_rmse:.6f}, r2={batch_target_r2:.6f})"
                            )
                            if not target_only:
                                msg += (
                                    f" tailgas(mae={batch_tailgas_mae:.6f}, rmse={batch_tailgas_rmse:.6f}, "
                                    f"r2={batch_tailgas_r2:.6f})"
                                )
                        print(msg)
            mean_train = epoch_loss / max(steps, 1)
            _cuda_synchronize(device)
            training_elapsed_sec = max(
                time.perf_counter() - epoch_started_at,
                1.0e-12,
            )
            graph_samples = float(
                epoch_comp_sums.get("graph_sample_count", steps)
            )
            optimizer_steps = float(
                epoch_comp_sums.get("optimizer_step_count", steps)
            )
            peak_gpu_memory_mb = (
                float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
                if device.type == "cuda"
                else 0.0
            )
            peak_gpu_reserved_mb = (
                float(torch.cuda.max_memory_reserved(device)) / (1024.0**2)
                if device.type == "cuda"
                else 0.0
            )
            if steps > 0:
                for _k, _v in list(epoch_comp_sums.items()):
                    if torch.is_tensor(_v):
                        epoch_comp_sums[_k] = float(_v.detach().cpu())
            if _is_main() and steps > 0:
                inv = 1.0 / float(steps)
                train_history["epoch"].append(float(epoch + 1))
                for _k in sorted(epoch_comp_sums.keys()):
                    train_history[_k].append(epoch_comp_sums[_k] * inv)
                if edge_all:
                    total_norm = float(sum(edge_all_loss_prop_norm.values()))
                    flow_norm = float(edge_all_loss_group_norm.get("flow_norm", 0.0))
                    frac_norm = float(edge_all_loss_group_norm.get("frac_norm", 0.0))
                    out_norm = float(edge_all_loss_group_norm.get("output_norm", 0.0))
                    ans_norm = float(edge_all_loss_group_norm.get("answer_norm", 0.0))
                    contrib_rows = []
                    for pn in sorted(edge_all_loss_prop_norm.keys()):
                        raw = float(edge_all_loss_prop_raw.get(pn, 0.0))
                        norm = float(edge_all_loss_prop_norm.get(pn, 0.0))
                        cnt = float(edge_all_loss_prop_count.get(pn, 0.0))
                        contrib_rows.append(
                            {
                                "epoch": int(epoch + 1),
                                "property_name": pn,
                                "property_raw_loss": raw,
                                "property_normalized_loss": norm,
                                "property_sample_count": int(cnt),
                                "property_loss_ratio": (norm / total_norm) if total_norm > 0 else 0.0,
                                "flow_loss_ratio": (flow_norm / total_norm) if total_norm > 0 else 0.0,
                                "frac_loss_ratio": (frac_norm / total_norm) if total_norm > 0 else 0.0,
                                "answer_edge_loss_ratio": (ans_norm / total_norm) if total_norm > 0 else 0.0,
                                "output_role_loss_ratio": (out_norm / total_norm) if total_norm > 0 else 0.0,
                            }
                        )
                    cpath = output_dir / "loss_contribution_by_epoch.csv"
                    mode = "w" if epoch == 0 else "a"
                    _ensure_output_dir(output_dir)
                    pd.DataFrame(contrib_rows).to_csv(cpath, index=False, mode=mode, header=(epoch == 0))
            if edge_all and not epoch_aborted_nonfinite and epoch_start_snapshot is not None:
                bad_state = _find_nonfinite_state_names(raw_model)
                if bad_state:
                    _restore_training_state(raw_model, optimizer, epoch_start_snapshot)
                    epoch_recovery_lrs = _scale_optimizer_lr(
                        optimizer,
                        factor=float(getattr(experiment.train, "nonfinite_recovery_lr_factor", 0.5)),
                        min_lr=float(
                            getattr(
                                experiment.train,
                                "nonfinite_recovery_min_lr",
                                getattr(experiment.train, "scheduler_min_lr", 1.0e-6),
                            )
                        ),
                    )
                    scaler = torch.amp.GradScaler(
                        "cuda",
                        enabled=experiment.train.mixed_precision and device.type == "cuda",
                    )
                    epoch_aborted_nonfinite = True
                    epoch_nonfinite_reason = "model_state:" + ",".join(bad_state)
                    if _is_main():
                        print(
                            "[train][recover] non-finite model state detected after epoch; restored epoch-start "
                            f"model/optimizer state, lr {epoch_recovery_lrs[0]} -> {epoch_recovery_lrs[1]}, "
                            "validation will use the restored finite state.",
                            flush=True,
                        )
            if scheduler is not None and not epoch_aborted_nonfinite:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(mean_train)
                elif edge_all and backward_mode in {
                    "edge_step",
                    "edge_step_pi",
                    "sample_hybrid_target_edge_step_pi",
                } and str(
                    getattr(experiment.train, "scheduler_step_unit", "epoch")
                ).lower() == "optimizer_step":
                    pass
                else:
                    scheduler.step()
            if edge_all:
                epoch_log = {
                    "epoch": float(epoch + 1),
                    "train/loss": mean_train,
                    "train_loss": mean_train,
                    "train/loss_total": mean_train,
                    "train_loss_total": mean_train,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/pinn_weight_schedule_scale": float(pinn_schedule_scale),
                    "train_pinn_weight_schedule_scale": float(pinn_schedule_scale),
                    "train/epoch_time_sec": training_elapsed_sec,
                    "train/samples_per_sec": graph_samples / training_elapsed_sec,
                    "train/optimizer_steps_per_sec": optimizer_steps
                    / training_elapsed_sec,
                    "train/peak_gpu_memory_mb": peak_gpu_memory_mb,
                    "train/peak_gpu_reserved_mb": peak_gpu_reserved_mb,
                }
                if epoch_aborted_nonfinite:
                    epoch_log["train/nonfinite_recovered"] = 1.0
                    epoch_log["train_nonfinite_recovered"] = 1.0
                    epoch_log["train/nonfinite_recovery_reason"] = epoch_nonfinite_reason
                    if epoch_recovery_lrs is not None:
                        epoch_log["train/nonfinite_recovery_lr_old"] = float(epoch_recovery_lrs[0][0])
                        epoch_log["train/nonfinite_recovery_lr_new"] = float(epoch_recovery_lrs[1][0])
                inv = 1.0 / max(steps, 1)
                for _lk in (
                    "loss_edge_all",
                            "loss_edge",
                    "loss_primary_frac",
                            "loss_target_incident_edges",
                            "loss_all_edge_r2",
                            "r2_loss_all",
                            "loss_primary_frac_r2",
                            "loss_target_frac_feature",
                            "loss_target_row_frac",
                            "loss_target_row_amount",
                            "primary_frac_count",
                            "target_incident_edge_count",
                            "target_incident_supervised_cell_count",
                            "target_incident_target_row_count",
                            "target_incident_matched_edge_count",
                            "target_incident_virtual_endpoint_excluded_count",
                            "target_incident_target_edge_included_count",
                            "target_incident_virtual_filter_applied",
                            "target_incident_apply_answer_weight",
                            "target_incident_inner_weight_mean",
                            "target_incident_inner_weight_max",
                            "target_incident_effective_weight_mean",
                            "target_incident_coverage_edge_frac",
                            "target_incident_loss_ratio_total",
                            "target_incident_loss_ratio_edge",
                            "amount_auxiliary_enabled",
                            "amount_auxiliary_contributes_to_total",
                            "all_edge_r2_loss_weight",
                            "all_edge_r2_min_count",
                            "all_edge_r2_sst_threshold",
                            "all_edge_r2_loss_cap",
                            "primary_frac_r2_valid_feature_count",
                            "r2_loss_num_elements",
                            "target_loss_num_elements",
                            "edge_all_loss_num_elements",
                            "r2_loss_valid_feature_count",
                            "primary_frac_pred_std_norm",
                    "primary_frac_true_std_norm",
                    "primary_frac_std_ratio_norm",
                    "primary_frac_pred_std_orig",
                    "primary_frac_true_std_orig",
                    "primary_frac_std_ratio_orig",
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
                    "edge_step_default_edge_count",
                    "target_edge_weight_mean",
                    "default_edge_weight_mean",
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
                    "loss_node_mass",
                    "loss_node_component",
                    "loss_node_atom",
                    "weighted_node_mass",
                    "weighted_node_component",
                    "weighted_node_atom",
                    "node_mass_valid_count",
                    "node_component_valid_count",
                    "node_atom_valid_count",
                    "node_mass_residual_mean",
                    "node_component_residual_mean",
                    "node_atom_residual_mean",
                    "node_mass_residual_max",
                    "node_component_residual_max",
                    "node_atom_residual_max",
                    "loss_total",
                ):
                    if _lk in epoch_comp_sums:
                        v = epoch_comp_sums[_lk] * inv
                        epoch_log[f"train/{_lk}"] = v
                        epoch_log[f"train_{_lk}"] = v
            else:
                epoch_log = {
                    "epoch": float(epoch + 1),
                    "train/loss": mean_train,
                    "train_loss": mean_train,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch_time_sec": training_elapsed_sec,
                    "train/samples_per_sec": graph_samples / training_elapsed_sec,
                    "train/optimizer_steps_per_sec": optimizer_steps
                    / training_elapsed_sec,
                    "train/peak_gpu_memory_mb": peak_gpu_memory_mb,
                    "train/peak_gpu_reserved_mb": peak_gpu_reserved_mb,
                    "train/metric_target_mae": _finalize_task_metrics(epoch_metric_accum["target"])["mae"],
                    "train/metric_target_rmse": _finalize_task_metrics(epoch_metric_accum["target"])["rmse"],
                    "train/metric_target_r2": _finalize_task_metrics(epoch_metric_accum["target"])["r2"],
                }
                if not target_only:
                    epoch_log["train/metric_tailgas_mae"] = _finalize_task_metrics(epoch_metric_accum["tailgas"])["mae"]
                    epoch_log["train/metric_tailgas_rmse"] = _finalize_task_metrics(epoch_metric_accum["tailgas"])["rmse"]
                    epoch_log["train/metric_tailgas_r2"] = _finalize_task_metrics(epoch_metric_accum["tailgas"])["r2"]
            epoch_log["train/optimizer_steps_total"] = float(optimizer_steps_total)
            epoch_log["train_optimizer_steps_total"] = float(optimizer_steps_total)
            epoch_log["train/optimizer_budget_reached"] = float(optimizer_budget_reached)
            epoch_log["train_optimizer_budget_reached"] = float(optimizer_budget_reached)
            do_val = val_loader is not None and (
                (epoch + 1) % max(experiment.train.val_interval, 1) == 0
                or epoch + 1 == experiment.train.epochs
                or optimizer_budget_reached
            )
            epoch_val_metrics: dict[str, float] | None = None
            if do_val:
                validation_started_at = time.perf_counter()
                active_val_loader = val_loader
                active_val_sampler = val_sampler
                sampled_val_size: int | None = None
                if val_dataset_for_eval is not None:
                    active_val_loader, active_val_sampler, sampled_val_size = _sample_validation_loader(
                        val_dataset_for_eval,
                        train_cfg=experiment.train,
                        data_cfg=experiment.data,
                        epoch=epoch,
                        use_ddp=use_ddp,
                    )
                if active_val_sampler is not None:
                    active_val_sampler.set_epoch(epoch)
                if _is_main() and not _terminal_log_is_quiet(experiment.train):
                    try:
                        num_val_batches = len(active_val_loader) if active_val_loader is not None else 0
                    except TypeError:
                        num_val_batches = "?"
                    sample_note = (
                        f" sampled={sampled_val_size}/{len(val_dataset_for_eval)}"
                        if sampled_val_size is not None and val_dataset_for_eval is not None
                        else ""
                    )
                    print(
                        f"[epoch {epoch + 1}/{experiment.train.epochs}] validation_start "
                        f"batches={num_val_batches}{sample_note}",
                        flush=True,
                    )
                if active_val_loader is None:
                    raise RuntimeError("validation requested but validation loader is not available")
                if edge_all:
                    if backward_mode in {"edge_step_pi", "sample_hybrid_target_edge_step_pi"}:
                        pi_val_normalizer = None
                        if y_edge_mean is not None and y_edge_std is not None:
                            pi_val_normalizer = {
                                "mean": y_edge_mean,
                                "std": y_edge_std,
                                "columns": y_edge_cols or STREAM_EDGE_FEATURE_SLOTS,
                                "mass_flow_log1p_mean": y_edge_mass_log_mean,
                                "mass_flow_log1p_std": y_edge_mass_log_std,
                                "mass_flow_log1p_count": y_edge_mass_log_count,
                                "mass_flow_physical_p05": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p05"),
                                "mass_flow_physical_p25": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p25"),
                                "mass_flow_physical_p75": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p75"),
                                "mass_flow_physical_p90": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p90"),
                                "mass_flow_physical_p95": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p95"),
                                "mass_flow_physical_p99": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p99"),
                            }
                        val_metrics = evaluate_pi_epoch(
                            model=model,
                            loader=active_val_loader,
                            device=device,
                            train_cfg=active_train_cfg,
                            data_cfg=experiment.data,
                            use_amp=use_amp,
                            normalizer=pi_val_normalizer,
                            target_stream_targets_path=str((experiment.project_root / "data/reference/v4/target_stream_targets.csv").resolve()),
                            target_edge_10d_out_dir=output_dir,
                        )
                    else:
                        targets_v4_csv = (
                            experiment.project_root / "data/reference/v4/target_stream_targets.csv"
                        ).resolve()
                        val_metrics = evaluate_edge_all_epoch(
                            model,
                            active_val_loader,
                            device,
                            active_train_cfg,
                            experiment.data,
                            use_amp,
                            y_edge_mean=y_edge_mean,
                            y_edge_std=y_edge_std,
                            edge_struct_dim=int(encoder_cfg.edge_struct_dim),
                            target_stream_targets_path=str(targets_v4_csv)
                            if targets_v4_csv.is_file()
                            else None,
                            compute_target_v4_scalars=True,
                        )
                else:
                    val_metrics = evaluate_epoch(
                        model,
                        active_val_loader,
                        device,
                        active_train_cfg,
                        experiment.data,
                        use_amp,
                        target_mean=target_mean,
                        target_std=target_std,
                    )
                epoch_val_metrics = _coerce_val_metric_scalars(val_metrics)
                validation_elapsed_sec = (
                    time.perf_counter() - validation_started_at
                )
                epoch_log["val/loss"] = val_metrics["loss_total"]
                epoch_log["val_loss"] = val_metrics["loss_total"]
                epoch_log["val/epoch_time_sec"] = validation_elapsed_sec
                epoch_log["val_epoch_time_sec"] = validation_elapsed_sec
                if _is_main() and not _terminal_log_is_quiet(experiment.train):
                    _target_property_mean = val_metrics.get(
                        "target_edge_property_mean_r2",
                        val_metrics.get("val_target_edge_property_mean_r2", float("nan")),
                    )
                    _all_edge_mean = val_metrics.get(
                        "pi_all_edge_property_mean_r2",
                        val_metrics.get("val_pi_all_edge_property_mean_r2", float("nan")),
                    )
                    print(
                        f"[epoch {epoch + 1}/{experiment.train.epochs}] validation_done "
                        f"elapsed={_format_terminal_duration(validation_elapsed_sec)} "
                        f"val_loss={float(val_metrics['loss_total']):.6f} "
                        f"target_edge_property_mean_r2={float(_target_property_mean):.6f} "
                        f"all_edge_property_mean_r2={float(_all_edge_mean):.6f}",
                        flush=True,
                    )
                if not edge_all:
                    epoch_log["val/loss_target"] = val_metrics["loss_target"]
                    epoch_log["val_loss_target"] = val_metrics["loss_target"]
                    if not target_only:
                        epoch_log["val/loss_tailgas"] = val_metrics["loss_tailgas"]
                        epoch_log["val_loss_tailgas"] = val_metrics["loss_tailgas"]
                    epoch_log["val/metric_target_mae"] = val_metrics["metric_target_mae"]
                    epoch_log["val_metric_target_mae"] = val_metrics["metric_target_mae"]
                    epoch_log["val/metric_target_rmse"] = val_metrics["metric_target_rmse"]
                    epoch_log["val_metric_target_rmse"] = val_metrics["metric_target_rmse"]
                    epoch_log["val/metric_target_r2"] = val_metrics["metric_target_r2"]
                    epoch_log["val_metric_target_r2"] = val_metrics["metric_target_r2"]
                    if not target_only:
                        epoch_log["val/metric_tailgas_mae"] = val_metrics["metric_tailgas_mae"]
                        epoch_log["val_metric_tailgas_mae"] = val_metrics["metric_tailgas_mae"]
                        epoch_log["val/metric_tailgas_rmse"] = val_metrics["metric_tailgas_rmse"]
                        epoch_log["val_metric_tailgas_rmse"] = val_metrics["metric_tailgas_rmse"]
                        epoch_log["val/metric_tailgas_r2"] = val_metrics["metric_tailgas_r2"]
                        epoch_log["val_metric_tailgas_r2"] = val_metrics["metric_tailgas_r2"]
                else:
                    epoch_log["val/loss_edge"] = val_metrics.get("loss_edge", val_metrics["loss_total"])
                    epoch_log["val_loss_edge"] = val_metrics.get("loss_edge", val_metrics["loss_total"])
                    if "loss_v4_targets" in val_metrics:
                        epoch_log["val/loss_v4_targets"] = float(val_metrics["loss_v4_targets"])
                        epoch_log["val_loss_v4_targets"] = float(val_metrics["loss_v4_targets"])
                    epoch_log["val/edge_all_mse"] = val_metrics.get("edge_all_mse", 0.0)
                    epoch_log["val/edge_all_mae"] = val_metrics.get("edge_all_mae", 0.0)
                    epoch_log["val/target_h2_mae"] = val_metrics.get("target_h2_mae", 0.0)
                    epoch_log["val/target_h2_rmse"] = val_metrics.get("target_h2_rmse", 0.0)
                    epoch_log["val/tailgas_co2_mae"] = val_metrics.get("tailgas_co2_mae", 0.0)
                    epoch_log["val/tailgas_co2_rmse"] = val_metrics.get("tailgas_co2_rmse", 0.0)
                    # Answer-edge: H2, CO2 및 동일 스트림의 기타 Frac_* (예: Frac_H2O) MAE 동일 가중 평균
                    if backward_mode not in {"edge_step_pi", "sample_hybrid_target_edge_step_pi"}:
                        mae_vals: list[float] = []
                        for _legacy_key in ("target_h2_mae", "tailgas_co2_mae"):
                            if _legacy_key in val_metrics:
                                try:
                                    mae_vals.append(float(val_metrics[_legacy_key]))
                                except (TypeError, ValueError):
                                    pass
                        for _k, _v in val_metrics.items():
                            _ks = str(_k)
                            if (
                                (_ks.startswith("answer_target_") or _ks.startswith("answer_tailgas_"))
                                and _ks.endswith("_mae")
                            ):
                                try:
                                    mae_vals.append(float(_v))
                                except (TypeError, ValueError):
                                    pass
                        if mae_vals:
                            epoch_log["val/answer_targets_mae"] = float(sum(mae_vals) / max(len(mae_vals), 1))
                    _mr2 = val_metrics.get("metric_answer_all_targets_mean_r2")
                    if _mr2 is not None and math.isfinite(float(_mr2)):
                        epoch_log["val/metric_answer_all_targets_mean_r2"] = float(_mr2)
                        epoch_log["val/legacy_answer_fraction_macro_r2"] = float(_mr2)
                        epoch_log["val_legacy_answer_fraction_macro_r2"] = float(_mr2)
                        epoch_log["val/legacy_answer_fraction_r2"] = float(_mr2)
                        epoch_log["val_legacy_answer_fraction_r2"] = float(_mr2)
                    _copy_edge_all_val_train_loss_scalars(epoch_log, val_metrics)
                    _merge_edge_all_val_scalars_into_epoch_log(epoch_log, val_metrics)
                if "loss_edge_stream" in val_metrics:
                    epoch_log["val/loss_edge_stream"] = val_metrics["loss_edge_stream"]
                    epoch_log["val_loss_edge_stream"] = val_metrics["loss_edge_stream"]
                if "metric_edge_stream_mae" in val_metrics:
                    epoch_log["val/metric_edge_stream_mae"] = val_metrics["metric_edge_stream_mae"]
                    epoch_log["val_metric_edge_stream_mae"] = val_metrics["metric_edge_stream_mae"]
                    epoch_log["val/metric_edge_stream_rmse"] = val_metrics["metric_edge_stream_rmse"]
                    epoch_log["val_metric_edge_stream_rmse"] = val_metrics["metric_edge_stream_rmse"]
                if _is_main():
                    if edge_all:
                        vmsg = format_val_metric_edge_all_line(val_metrics)
                    else:
                        vmsg = (
                            f"   [val-metric] "
                            f"target(mae={val_metrics['metric_target_mae']:.6f}, "
                            f"rmse={val_metrics['metric_target_rmse']:.6f}, "
                            f"r2={val_metrics['metric_target_r2']:.6f})"
                        )
                        if not target_only:
                            vmsg += (
                                f" tailgas(mae={val_metrics['metric_tailgas_mae']:.6f}, "
                                f"rmse={val_metrics['metric_tailgas_rmse']:.6f}, "
                                f"r2={val_metrics['metric_tailgas_r2']:.6f})"
                            )
                        if float(getattr(experiment.train, "edge_stream_loss_weight", 0.0)) > 0.0 and getattr(
                            experiment.data, "use_canonical_graph_spec_v3", False
                        ):
                            vmsg += (
                                f" edge_stream(mae={val_metrics['metric_edge_stream_mae']:.6f}, "
                                f"rmse={val_metrics['metric_edge_stream_rmse']:.6f})"
                            )
                    print(vmsg)
                    if experiment.train.debug_mode and not edge_all:
                        dmsg = (
                            f"   [val-debug] "
                            f"target(pred_std={val_metrics.get('debug_target_pred_std', float('nan')):.6f}, "
                            f"y_std={val_metrics.get('debug_target_true_std', float('nan')):.6f}, "
                            f"pred_orig_std={val_metrics.get('debug_target_pred_orig_std', float('nan')):.6f}, "
                            f"y_orig_std={val_metrics.get('debug_target_true_orig_std', float('nan')):.6f})"
                        )
                        if not target_only:
                            dmsg += (
                                f" tailgas(pred_std={val_metrics.get('debug_tailgas_pred_std', float('nan')):.6f}, "
                                f"y_std={val_metrics.get('debug_tailgas_true_std', float('nan')):.6f}, "
                                f"pred_orig_std={val_metrics.get('debug_tailgas_pred_orig_std', float('nan')):.6f}, "
                                f"y_orig_std={val_metrics.get('debug_tailgas_true_orig_std', float('nan')):.6f})"
                            )
                        print(dmsg)
                for _vk, _vv in val_metrics.items():
                    if _vk in epoch_val_metrics:
                        val_history[str(_vk)].append(epoch_val_metrics[str(_vk)])
                    elif isinstance(_vv, (int, float)) and math.isfinite(float(_vv)):
                        val_history[str(_vk)].append(float(_vv))
                if edge_all and "val/answer_targets_mae" in epoch_log:
                    val_history["answer_targets_mae"].append(float(epoch_log["val/answer_targets_mae"]))
                val_history["epoch"].append(float(epoch + 1))
            if _is_main():
                tc = (
                    {k: float(epoch_comp_sums[k]) * (1.0 / max(steps, 1)) for k in epoch_comp_sums}
                    if steps > 0
                    else {}
                )
                if edge_all:
                    tc["all_edge_r2_config_source"] = resolve_all_edge_r2_config(experiment.train)["source"]
                tc["epoch_time_sec"] = float(training_elapsed_sec)
                tc["samples_per_sec"] = float(graph_samples / training_elapsed_sec)
                tc["optimizer_steps_per_sec"] = float(optimizer_steps / training_elapsed_sec)
                tc["peak_gpu_memory_mb"] = float(peak_gpu_memory_mb)
                tc["peak_gpu_reserved_mb"] = float(peak_gpu_reserved_mb)
                tc["optimizer_steps_total"] = float(optimizer_steps_total)
                tc["optimizer_budget_reached"] = float(optimizer_budget_reached)
                _append_metrics_per_epoch_csv(
                    output_dir,
                    edge_all=edge_all,
                    target_only=target_only,
                    epoch=int(epoch + 1),
                    train_loss=float(mean_train),
                    lr=float(optimizer.param_groups[0]["lr"]),
                    val_metrics=epoch_val_metrics,
                    answer_weight_h2=float(getattr(experiment.train, "answer_edge_weight_h2", 1.0)),
                    answer_weight_co2=float(getattr(experiment.train, "answer_edge_weight_co2", 1.0)),
                    train_components=tc,
                )
                if not args.no_training_plots:
                    _save_metrics_per_epoch_plots(output_dir)
            tracked_metric = epoch_log.get(args.best_metric_name)
            if (
                tracked_metric is None
                and edge_all
                and args.best_metric_name == "val/target_edge_property_mean_r2"
            ):
                raise RuntimeError(
                    "Configured monitor val_target_edge_property_mean_r2 was not emitted. "
                    "Refusing to select a checkpoint with a differently named fallback metric."
                )
            if tracked_metric is None and edge_all:
                from process_graph.experiment.target_row_primary_metrics import (
                    LEGACY_ANSWER_FRACTION_METRIC,
                    PRIMARY_METRIC_NAME,
                    TARGET_BALANCED_METRIC_NAME,
                )

                _fallback_key = ""
                for key in (
                    f"val/{PRIMARY_METRIC_NAME}",
                    f"val/{TARGET_BALANCED_METRIC_NAME}",
                    "val/process_balanced_main_target_r2_formula_available",
                    f"val/{LEGACY_ANSWER_FRACTION_METRIC}",
                    "val/legacy_answer_fraction_macro_r2",
                    "val/target_v4_macro_r2_main_verified",
                ):
                    tracked_metric = epoch_log.get(key)
                    if tracked_metric is not None:
                        _fallback_key = key
                        break
                if (
                    tracked_metric is not None
                    and _is_main()
                    and not monitor_fallback_warned
                    and "legacy" in _fallback_key
                ):
                    print(
                        "[WARN] target-row main metric not found. Falling back to "
                        "legacy_answer_fraction_r2. This is not the primary v4 main-target metric.",
                        flush=True,
                    )
                    monitor_fallback_warned = True
            if tracked_metric is not None and _metric_improved(tracked_metric, best_metric, args.best_metric_mode):
                best_metric = tracked_metric
                best_epoch = epoch + 1
                best_val_loss = float(epoch_log.get("val/loss", tracked_metric))
                if _is_main() and save_model_weights:
                    best_checkpoint_payload = {
                            "model": raw_model.state_dict(),
                            "model_state_dict": raw_model.state_dict(),
                            "experiment": str(experiment_path),
                            "metric_name": args.best_metric_name,
                            "metric_value": float(best_metric),
                            "epoch": best_epoch,
                            "best_val_loss": float(best_val_loss),
                            "pi_mass_flow_output_space": resolve_pi_mass_flow_output_space(experiment.train),
                            "mass_flow_transform": resolve_mass_flow_transform(experiment.train),
                            "mass_flow_log_tau": resolve_mass_flow_log_tau(experiment.train),
                            "mass_flow_log_scale": resolve_mass_flow_log_scale(experiment.train),
                            "mass_flow_log_eps": resolve_mass_flow_log_eps(experiment.train),
                            "mass_flow_physical_scale": (
                                None if mass_scale_metadata is None else dict(mass_scale_metadata)
                            ),
                            "oper_normalizer": (
                                None
                                if oper_mean is None or oper_std is None
                                else {
                                    "columns": list(oper_feature_names),
                                    "mean": oper_mean.detach().cpu(),
                                    "std": oper_std.detach().cpu(),
                                    "train_split_only": True,
                                }
                            ),
                            "transfer_initialization": dict(transfer_init_metadata),
                            "config": _jsonable_config(asdict(experiment)),
                        }
                    if bool(getattr(experiment.train, "checkpoint_include_optimizer_state", True)):
                        best_checkpoint_payload["optimizer_state_dict"] = optimizer.state_dict()
                    _atomic_torch_save(best_checkpoint_payload, best_ckpt_path)
                    _cuda_synchronize(device)
                    best_checkpoint_wall_sec = time.perf_counter() - fit_wall_started_at
            if "val/loss" in epoch_log:
                final_val_loss = float(epoch_log["val/loss"])
            if args.early_stopping:
                es_metric_value = epoch_log.get(args.early_stopping_metric)
                if (
                    es_metric_value is None
                    and edge_all
                    and args.early_stopping_metric == "val/target_edge_property_mean_r2"
                ):
                    raise RuntimeError(
                        "Configured early-stopping metric val_target_edge_property_mean_r2 "
                        "was not emitted; no fallback metric will be substituted."
                    )
                if es_metric_value is None and edge_all:
                    from process_graph.experiment.target_row_primary_metrics import (
                        LEGACY_ANSWER_FRACTION_METRIC,
                        PRIMARY_METRIC_NAME,
                        TARGET_BALANCED_METRIC_NAME,
                    )

                    for key in (
                        f"val/{PRIMARY_METRIC_NAME}",
                        f"val/{TARGET_BALANCED_METRIC_NAME}",
                        f"val/{LEGACY_ANSWER_FRACTION_METRIC}",
                        "val/legacy_answer_fraction_macro_r2",
                    ):
                        es_metric_value = epoch_log.get(key)
                        if es_metric_value is not None:
                            break
                if es_metric_value is not None:
                    if _metric_improved_with_delta(
                        float(es_metric_value),
                        early_stop_best,
                        args.early_stopping_mode,
                        args.early_stopping_min_delta,
                    ):
                        early_stop_best = float(es_metric_value)
                        epochs_without_improve = 0
                    else:
                        epochs_without_improve += 1
                    if epochs_without_improve >= int(args.early_stopping_patience):
                        stopped_early = True
                        early_stop_epoch = int(epoch + 1)
                        if _is_main():
                            print(
                                f"[early-stopping] stop_epoch={early_stop_epoch} best_epoch={best_epoch} "
                                f"patience={int(args.early_stopping_patience)} min_delta={float(args.early_stopping_min_delta):.2e}"
                            )
                        epoch_log["best_so_far"] = float(best_metric) if best_metric is not None else float("nan")
                        break
            epoch_log["best_so_far"] = float(best_metric) if best_metric is not None else float("nan")
            if _is_main():
                epoch_summary = _format_epoch_summary_line(
                    epoch=epoch,
                    total_epochs=experiment.train.epochs,
                    mean_train=float(mean_train),
                    lr=float(optimizer.param_groups[0]["lr"]),
                    epoch_log=epoch_log,
                    edge_all=edge_all,
                    target_only=target_only,
                    best_metric_name=str(args.best_metric_name),
                    had_validation=do_val,
                )
                print(epoch_summary, flush=True)
                if edge_all:
                    print(format_epoch_detail_edge_all_line(epoch_log), flush=True)
            if optimizer_budget_reached:
                if _is_main():
                    print(
                        f"[fixed-optimizer-budget] reached exact total={optimizer_steps_total} "
                        f"at epoch={epoch + 1}; checkpoint selection remains validation-based.",
                        flush=True,
                    )
                break

        if _is_main() and getattr(experiment.train, "probe_final_grad_norms", False) and not edge_all:
            raw_model.train()
            optimizer.zero_grad(set_to_none=True)
            probe_batch = next(iter(train_loader))
            pb_data = {k: v.to(device) for k, v in probe_batch.model_kwargs.items()}
            pb_targets = {k: v.to(device) for k, v in probe_batch.targets.items()}
            pb_masks = {k: v.to(device) for k, v in probe_batch.target_masks.items()}
            pb_task_inputs = {
                head: {k: v.to(device) for k, v in payload.items()}
                for head, payload in probe_batch.task_inputs.items()
            }
            with torch.amp.autocast("cuda", enabled=use_amp):
                pb_preds = model(pb_data, task_inputs=pb_task_inputs)
                pb_losses = compute_training_loss(
                    pb_preds, pb_targets, pb_masks, experiment.train, experiment.data
                )
                pb_loss = pb_losses["loss_total"]
            if scaler is not None:
                scaler.scale(pb_loss).backward()
                scaler.unscale_(optimizer)
            else:
                pb_loss.backward()
            probe_payload = {
                "encoder_grad_l2": _grad_l2_norm_prefix(raw_model, "encoder"),
                "target_head_grad_l2": _grad_l2_norm_prefix(raw_model, "heads.target"),
            }
            if not target_only and "tailgas" in raw_model.heads:
                probe_payload["tailgas_head_grad_l2"] = _grad_l2_norm_prefix(raw_model, "heads.tailgas")
            if getattr(raw_model, "edge_stream_head", None) is not None:
                probe_payload["edge_stream_head_grad_l2"] = _grad_l2_norm_prefix(
                    raw_model, "edge_stream_head"
                )
            (output_dir / "grad_norm_probe.json").write_text(
                json.dumps(probe_payload, indent=2), encoding="utf-8"
            )
            optimizer.zero_grad(set_to_none=True)

        if _is_main() and save_model_weights:
            save_last_checkpoint = bool(getattr(experiment.train, "save_last_checkpoint", True))
            if save_last_checkpoint or not best_ckpt_path.is_file():
                last_checkpoint_payload = {
                    "model": raw_model.state_dict(),
                    "model_state_dict": raw_model.state_dict(),
                    "experiment": str(experiment_path),
                    "epoch": int(final_epoch),
                    "best_val_loss": float(best_val_loss) if best_val_loss is not None else float("nan"),
                    "pi_mass_flow_output_space": resolve_pi_mass_flow_output_space(experiment.train),
                    "mass_flow_transform": resolve_mass_flow_transform(experiment.train),
                    "mass_flow_log_tau": resolve_mass_flow_log_tau(experiment.train),
                    "mass_flow_log_scale": resolve_mass_flow_log_scale(experiment.train),
                    "mass_flow_log_eps": resolve_mass_flow_log_eps(experiment.train),
                    "mass_flow_physical_scale": (
                        None if mass_scale_metadata is None else dict(mass_scale_metadata)
                    ),
                    "oper_normalizer": (
                        None
                        if oper_mean is None or oper_std is None
                        else {
                            "columns": list(oper_feature_names),
                            "mean": oper_mean.detach().cpu(),
                            "std": oper_std.detach().cpu(),
                            "train_split_only": True,
                        }
                    ),
                    "transfer_initialization": dict(transfer_init_metadata),
                    "config": _jsonable_config(asdict(experiment)),
                }
                if bool(getattr(experiment.train, "checkpoint_include_optimizer_state", True)):
                    last_checkpoint_payload["optimizer_state_dict"] = optimizer.state_dict()
                _atomic_torch_save(last_checkpoint_payload, last_ckpt_path)
                print(f"[saved] {last_ckpt_path}")
            else:
                print("[checkpoint] save_last_checkpoint=false; using validation-selected best.pt", flush=True)
            eval_ckpt_path = best_ckpt_path if best_ckpt_path.is_file() else last_ckpt_path
            eval_payload = torch.load(eval_ckpt_path, map_location=device)
            assert_checkpoint_pi_mass_flow_compatible(
                eval_payload,
                train_cfg=experiment.train,
                checkpoint_path=eval_ckpt_path,
            )
            _restore_mass_flow_physical_scale_from_checkpoint(
                eval_payload,
                train_cfg=experiment.train,
                checkpoint_path=eval_ckpt_path,
            )
            eval_state = eval_payload.get("model_state_dict") or eval_payload.get("model")
            if eval_state is None:
                raise RuntimeError(f"Checkpoint {eval_ckpt_path} does not contain model weights.")
            _load_state_dict_allowing_target_adapter_missing(raw_model, eval_state, strict=True)
            _cuda_synchronize(device)
            end_to_end_training_sec = time.perf_counter() - fit_wall_started_at
            if args.efficiency_benchmark:
                epoch_frame = pd.read_csv(output_dir / "metrics_per_epoch.csv")
                optimization_times = pd.to_numeric(
                    epoch_frame.get("train_epoch_time_sec"), errors="coerce"
                ).dropna()
                efficiency_training_payload = {
                    "optimization_time_sec": float(optimization_times.sum()),
                    "end_to_end_training_time_sec": float(end_to_end_training_sec),
                    "end_to_end_time_until_best_sec": float(best_checkpoint_wall_sec),
                    "optimization_time_definition": "sum of synchronized optimizer epoch loops",
                    "end_to_end_time_definition": (
                        "first training epoch start through validation, early stopping, checkpoint "
                        "serialization, best-checkpoint selection and reload"
                    ),
                    "best_epoch": int(best_epoch),
                    "stopped_epoch": int(final_epoch),
                    "stopped_early": bool(stopped_early),
                    "early_stopping_patience": int(args.early_stopping_patience),
                }
                (output_dir / "computational_efficiency_training.json").write_text(
                    json.dumps(efficiency_training_payload, indent=2), encoding="utf-8"
                )
            print(
                f"[train-summary] final_epoch={int(final_epoch)} best_epoch={int(best_epoch)} "
                f"final_val_loss={float(final_val_loss) if final_val_loss is not None else float('nan'):.6f} "
                f"best_val_loss={float(best_val_loss) if best_val_loss is not None else float('nan'):.6f} "
                f"evaluation_checkpoint_path={eval_ckpt_path}"
            )
            if args.efficiency_benchmark and test_loader is not None:
                inference_payload = _benchmark_inference_loader(
                    model=model,
                    loader=test_loader,
                    device=device,
                    train_cfg=experiment.train,
                    warmup_runs=args.inference_warmup_runs,
                    measured_runs=args.inference_measured_runs,
                    max_batches=args.inference_max_batches,
                    use_amp=use_amp,
                    target_edges_only=bool(args.inference_target_edges_only),
                    verify_target_edge_values=bool(args.inference_target_edges_only),
                )
                inference_payload.update(
                    {
                        "total_parameters": int(sum(p.numel() for p in raw_model.parameters())),
                        "trainable_parameters": int(
                            sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
                        ),
                        "checkpoint_path": str(eval_ckpt_path),
                        "precision": "amp" if use_amp else "float32",
                    }
                )
                (output_dir / "computational_efficiency_inference.json").write_text(
                    json.dumps(inference_payload, indent=2), encoding="utf-8"
                )
                print(
                    "[efficiency][inference] "
                    f"samples={inference_payload['samples_per_run']} "
                    f"latency_ms={inference_payload['inference_time_per_sample_ms']:.6f} "
                    f"throughput={inference_payload['throughput_samples_per_sec']:.3f}/s",
                    flush=True,
                )
        elif _is_main():
            print("[checkpoint] --no-save-model-weights: skipped writing best.pt/last.pt", flush=True)
            print(
                f"[train-summary] final_epoch={int(final_epoch)} best_epoch={int(best_epoch)} "
                f"final_val_loss={float(final_val_loss) if final_val_loss is not None else float('nan'):.6f} "
                f"best_val_loss={float(best_val_loss) if best_val_loss is not None else float('nan'):.6f} "
                "evaluation_checkpoint_path=<disabled>",
                flush=True,
            )
        if _is_main():
            if (
                edge_all
                and _is_main()
                and backward_mode not in {"edge_step_pi", "sample_hybrid_target_edge_step_pi"}
                and bool(getattr(experiment.train, "edge_all_export_predictions", True))
            ):
                answers_csv = (
                    experiment.project_root / experiment.data.canonical_graph_spec_v3_dir / "target_answer_edges.csv"
                ).resolve()
                targets_v4_csv = (experiment.project_root / "data/reference/v4/target_stream_targets.csv").resolve()
                split_artifacts: dict[str, Any] = {}
                for split_name, ld in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
                    if ld is None:
                        continue
                    art = evaluate_edge_all_detailed(
                        model=model,
                        loader=ld,
                        device=device,
                        train_cfg=experiment.train,
                        data_cfg=experiment.data,
                        y_edge_mean=y_edge_mean,
                        y_edge_std=y_edge_std,
                        edge_struct_dim=int(encoder_cfg.edge_struct_dim),
                        target_answer_edges_path=answers_csv,
                        split_name=split_name,
                    )
                    split_dir = output_dir / split_name
                    write_edge_all_artifacts(
                        out_dir=split_dir,
                        artifacts=art,
                        scaler_columns=y_edge_cols or list(STREAM_EDGE_FEATURE_SLOTS),
                        scaler_mean=y_edge_mean,
                        scaler_std=y_edge_std,
                        scaler_counts=y_edge_counts,
                    )
                    if targets_v4_csv.is_file():
                        compute_target_metrics_v4(
                            out_dir=split_dir,
                            edge_predictions=art.edge_predictions,
                            target_stream_targets_path=targets_v4_csv,
                            base_metrics=art.metrics,
                            train_cfg=experiment.train,
                        )
                    split_artifacts[split_name] = art
                if split_artifacts:
                    merged_edge_predictions = pd.concat(
                        [v.edge_predictions for v in split_artifacts.values()], ignore_index=True
                    )
                    metrics_payload = {}
                    for sp, art in split_artifacts.items():
                        for k, v in art.metrics.items():
                            metrics_payload[f"{sp}/{k}"] = float(v)
                        legacy = art.metrics.get("metric_answer_all_targets_mean_r2")
                        if legacy is not None:
                            metrics_payload[f"{sp}/legacy_answer_fraction_macro_r2"] = float(legacy)
                            metrics_payload[f"{sp}_legacy_answer_fraction_macro_r2"] = float(legacy)
                    metrics_payload["final_epoch"] = int(final_epoch)
                    metrics_payload["best_epoch"] = int(best_epoch)
                    metrics_payload["final_val_loss"] = float(final_val_loss) if final_val_loss is not None else float("nan")
                    metrics_payload["best_val_loss"] = float(best_val_loss) if best_val_loss is not None else float("nan")
                    metrics_payload["evaluation_checkpoint_path"] = str(eval_ckpt_path) if eval_ckpt_path is not None else ""
                    metrics_payload["stopped_early"] = bool(stopped_early)
                    metrics_payload["early_stop_epoch"] = int(early_stop_epoch)
                    if targets_v4_csv.is_file():
                        v4_payload = compute_target_metrics_v4(
                            out_dir=output_dir,
                            edge_predictions=merged_edge_predictions,
                            target_stream_targets_path=targets_v4_csv,
                            base_metrics=metrics_payload,
                            train_cfg=experiment.train,
                        )
                        for k, v in v4_payload.items():
                            if isinstance(v, (int, float, bool)):
                                metrics_payload[k] = float(v)
                        ts_payload = write_target_stream_feature_metric_artifacts(
                            out_dir=output_dir,
                            edge_predictions=merged_edge_predictions,
                            train_cfg=experiment.train,
                            target_stream_targets_path=targets_v4_csv,
                        )
                        for k, v in ts_payload.items():
                            metrics_payload[k] = float(v)
                        te10_payload = write_target_edge_10d_metric_artifacts(
                            out_dir=output_dir,
                            edge_predictions=merged_edge_predictions,
                            train_cfg=experiment.train,
                            target_stream_targets_path=targets_v4_csv,
                        )
                        for k, v in te10_payload.items():
                            metrics_payload[k] = float(v)
                    (output_dir / "metrics.json").write_text(
                        json.dumps(enrich_metrics_json(metrics_payload, experiment.train), indent=2),
                        encoding="utf-8",
                    )
                    # extra diagnostics / baselines (logging-only)
                    write_edge_all_extra_diagnostics(
                        out_dir=output_dir,
                        edge_predictions=merged_edge_predictions,
                    )
                    write_edge_all_combined_metrics_csvs(
                        output_dir,
                        split_artifacts,
                        merged_edge_predictions,
                    )
                    write_edge_all_metrics_json_all_splits(output_dir, split_artifacts)
                    write_edge_all_eval_figures(
                        output_dir / "plots" / "edge_all_eval",
                        merged_edge_predictions,
                    )
                    current_by_split = {
                        sp: {
                            "edge_all_mae_orig": float(art.metrics.get("edge_all_mae_orig", 0.0)),
                            "edge_all_rmse_orig": float(art.metrics.get("edge_all_rmse_orig", 0.0)),
                            "edge_all_r2_orig": float(art.metrics.get("edge_all_r2_orig", 0.0)),
                            "target_h2_mae_orig": float(art.metrics.get("target_h2_mae_orig", 0.0)),
                            "target_h2_rmse_orig": float(art.metrics.get("target_h2_rmse_orig", 0.0)),
                            "target_h2_r2_orig": float(art.metrics.get("target_h2_r2_orig", 0.0)),
                            "tailgas_co2_mae_orig": float(art.metrics.get("tailgas_co2_mae_orig", 0.0)),
                            "tailgas_co2_rmse_orig": float(art.metrics.get("tailgas_co2_rmse_orig", 0.0)),
                            "tailgas_co2_r2_orig": float(art.metrics.get("tailgas_co2_r2_orig", 0.0)),
                            "true_std": float(art.metrics.get("true_std", 0.0)),
                            "r2_unstable": bool(art.metrics.get("r2_unstable", False)),
                        }
                        for sp, art in split_artifacts.items()
                    }
                    baseline_payload = write_baseline_comparison(
                        out_dir=output_dir,
                        edge_predictions=merged_edge_predictions,
                        current_metrics_by_split=current_by_split,
                    )

                    # loss contribution summary
                    lc_path = output_dir / "loss_contribution_by_epoch.csv"
                    if lc_path.is_file():
                        lc = pd.read_csv(lc_path)
                        if not lc.empty:
                            last = lc[lc["epoch"] == lc["epoch"].max()]
                            summary = {
                                "flow_loss_ratio_last_epoch": float(last["flow_loss_ratio"].iloc[0]),
                                "frac_loss_ratio_last_epoch": float(last["frac_loss_ratio"].iloc[0]),
                                "answer_edge_loss_ratio_last_epoch": float(last["answer_edge_loss_ratio"].iloc[0]),
                                "output_role_loss_ratio_last_epoch": float(last["output_role_loss_ratio"].iloc[0]),
                                "flow_loss_ratio_mean": float(lc["flow_loss_ratio"].mean()),
                                "frac_loss_ratio_mean": float(lc["frac_loss_ratio"].mean()),
                                "answer_edge_loss_ratio_mean": float(lc["answer_edge_loss_ratio"].mean()),
                                "output_role_loss_ratio_mean": float(lc["output_role_loss_ratio"].mean()),
                            }
                            (output_dir / "loss_contribution_summary.json").write_text(
                                json.dumps(summary, indent=2), encoding="utf-8"
                            )
                    # Required console summary
                    test_cur = current_by_split.get("test", {})
                    test_can = baseline_payload.get("canonical_edge_property_mean", {}).get("test", {})
                    tdiag = pd.read_csv(output_dir / "target_h2_diagnostic.csv") if (output_dir / "target_h2_diagnostic.csv").is_file() else pd.DataFrame()
                    cdiag = pd.read_csv(output_dir / "tailgas_co2_diagnostic.csv") if (output_dir / "tailgas_co2_diagnostic.csv").is_file() else pd.DataFrame()
                    lsum = {}
                    if (output_dir / "loss_contribution_summary.json").is_file():
                        lsum = json.loads((output_dir / "loss_contribution_summary.json").read_text(encoding="utf-8"))
                    unstable = []
                    for _, prow in pd.read_csv(output_dir / "baseline_answer_edge_metrics.csv").iterrows():
                        if bool(prow.get("r2_unstable", False)):
                            unstable.append(f"{prow.get('baseline')}::{prow.get('split')}::{prow.get('task_name')}")
                    print("[edge_all summary]")
                    print(
                        f"  current test edge_all: MAE={test_cur.get('edge_all_mae_orig', float('nan')):.6f} "
                        f"RMSE={test_cur.get('edge_all_rmse_orig', float('nan')):.6f} "
                        f"R2={test_cur.get('edge_all_r2_orig', float('nan')):.6f}"
                    )
                    print(
                        f"  canonical_edge baseline test edge_all: MAE={test_can.get('edge_all_mae_orig', float('nan')):.6f} "
                        f"RMSE={test_can.get('edge_all_rmse_orig', float('nan')):.6f} "
                        f"R2={test_can.get('edge_all_r2_orig', float('nan')):.6f}"
                    )
                    if not tdiag.empty:
                        tr_true = tdiag["y_true_orig"].astype(float)
                        tr_pred = tdiag["y_pred_orig"].astype(float)
                        tr_err = tr_pred - tr_true
                        tr_mae = float(tr_err.abs().mean())
                        tr_rmse = float((tr_err.pow(2).mean()) ** 0.5)
                        tr_std = float(tr_true.std(ddof=0))
                        tr_pred_std = float(tr_pred.std(ddof=0))
                        tr_r2 = float(1.0 - (tr_err.pow(2).sum() / ((tr_true - tr_true.mean()).pow(2).sum()))) if float(((tr_true - tr_true.mean()).pow(2).sum())) > 0 else float("nan")
                        print(
                            f"  target_h2: MAE={tr_mae:.6f} "
                            f"RMSE={tr_rmse:.6f} R2={tr_r2:.6f} "
                            f"true_std={tr_std:.6f} pred_std={tr_pred_std:.6f} "
                            f"std_ratio={(tr_pred_std / tr_std) if tr_std > 0 else float('nan'):.6f}"
                        )
                    if not cdiag.empty:
                        cr = cdiag.iloc[0]
                        print(
                            f"  tailgas_co2: MAE={float(cr.get('mae_orig', float('nan'))):.6f} "
                            f"RMSE={float(cr.get('rmse_orig', float('nan'))):.6f} "
                            f"true_std={float(cr.get('true_std', float('nan'))):.6f} pred_std={float(cr.get('pred_std', float('nan'))):.6f}"
                        )
                    print(
                        f"  Flow loss ratio={float(lsum.get('flow_loss_ratio_last_epoch', float('nan'))):.6f} "
                        f"Frac loss ratio={float(lsum.get('frac_loss_ratio_last_epoch', float('nan'))):.6f} "
                        f"answer edge loss ratio={float(lsum.get('answer_edge_loss_ratio_last_epoch', float('nan'))):.6f}"
                    )
                    print(
                        f"  output role loss ratio={float(lsum.get('output_role_loss_ratio_last_epoch', float('nan'))):.6f}"
                    )
                    print(f"  R2 unstable targets={unstable}")
                    tsv4 = output_dir / "test" / "target_metrics_v4.csv"
                    if tsv4.is_file():
                        tdf = pd.read_csv(tsv4)
                        focus_ids = {"FUELGAS_CO2", "PROD_H2", "PROD_CO2", "RESTEAM_H2O"}
                        focus = tdf[tdf["target_id"].astype(str).isin(focus_ids)]
                        for _, row in focus.iterrows():
                            print(
                                f"  [target-v4] {row.get('target_id')} R2={float(row.get('r2', float('nan'))):.6f} "
                                f"MAE={float(row.get('mae', float('nan'))):.6f} RMSE={float(row.get('rmse', float('nan'))):.6f}"
                            )
                    print(f"  generated files under: {output_dir}")
                if save_model_weights and test_loader is not None and last_ckpt_path.is_file():
                    def _eval_checkpoint(ckpt_path: Path):
                        payload = torch.load(ckpt_path, map_location=device)
                        assert_checkpoint_pi_mass_flow_compatible(
                            payload,
                            train_cfg=experiment.train,
                            checkpoint_path=ckpt_path,
                        )
                        _restore_mass_flow_physical_scale_from_checkpoint(
                            payload,
                            train_cfg=experiment.train,
                            checkpoint_path=ckpt_path,
                        )
                        state = payload.get("model_state_dict") or payload.get("model")
                        if state is None:
                            raise RuntimeError(f"Checkpoint {ckpt_path} missing model state.")
                        _load_state_dict_allowing_target_adapter_missing(raw_model, state, strict=True)
                        return evaluate_edge_all_detailed(
                            model=model,
                            loader=test_loader,
                            device=device,
                            train_cfg=experiment.train,
                            data_cfg=experiment.data,
                            y_edge_mean=y_edge_mean,
                            y_edge_std=y_edge_std,
                            edge_struct_dim=int(encoder_cfg.edge_struct_dim),
                            target_answer_edges_path=answers_csv,
                            split_name="test",
                        )

                    final_art = _eval_checkpoint(last_ckpt_path)
                    pd.DataFrame([final_art.metrics]).to_csv(output_dir / "test_metrics_final_checkpoint.csv", index=False)
                    best_art = _eval_checkpoint(best_ckpt_path) if best_ckpt_path.is_file() else final_art
                    pd.DataFrame([best_art.metrics]).to_csv(output_dir / "test_metrics_best_checkpoint.csv", index=False)
                    compare_rows = []
                    for metric_key in sorted(set(final_art.metrics.keys()) | set(best_art.metrics.keys())):
                        fval = float(final_art.metrics.get(metric_key, float("nan")))
                        bval = float(best_art.metrics.get(metric_key, float("nan")))
                        compare_rows.append(
                            {
                                "metric": metric_key,
                                "final_checkpoint": fval,
                                "best_checkpoint": bval,
                                "delta_best_minus_final": bval - fval,
                            }
                        )
                    # Property-level R2 comparison
                    if not final_art.metrics_by_property.empty or not best_art.metrics_by_property.empty:
                        fprop = final_art.metrics_by_property.set_index("property_name")
                        bprop = best_art.metrics_by_property.set_index("property_name")
                        for prop in sorted(set(fprop.index) | set(bprop.index)):
                            fval = float(fprop.loc[prop]["property_r2_orig"]) if prop in fprop.index else float("nan")
                            bval = float(bprop.loc[prop]["property_r2_orig"]) if prop in bprop.index else float("nan")
                            compare_rows.append(
                                {
                                    "metric": f"property_r2::{prop}",
                                    "final_checkpoint": fval,
                                    "best_checkpoint": bval,
                                    "delta_best_minus_final": bval - fval,
                                }
                            )
                    # Answer-edge R2 comparison
                    if not final_art.answer_edge_metrics.empty or not best_art.answer_edge_metrics.empty:
                        fans = final_art.answer_edge_metrics.set_index("task_name")
                        bans = best_art.answer_edge_metrics.set_index("task_name")
                        for task_name in sorted(set(fans.index) | set(bans.index)):
                            fval = float(fans.loc[task_name]["r2"]) if task_name in fans.index else float("nan")
                            bval = float(bans.loc[task_name]["r2"]) if task_name in bans.index else float("nan")
                            compare_rows.append(
                                {
                                    "metric": f"answer_edge_r2::{task_name}",
                                    "final_checkpoint": fval,
                                    "best_checkpoint": bval,
                                    "delta_best_minus_final": bval - fval,
                                }
                            )
                    # Target-v4 R2 comparison (test split).
                    if targets_v4_csv.is_file():
                        fv4 = compute_target_metrics_v4(
                            out_dir=(output_dir / "_tmp_final_v4"),
                            edge_predictions=final_art.edge_predictions,
                            target_stream_targets_path=targets_v4_csv,
                            base_metrics=None,
                            train_cfg=experiment.train,
                        )
                        bv4 = compute_target_metrics_v4(
                            out_dir=(output_dir / "_tmp_best_v4"),
                            edge_predictions=best_art.edge_predictions,
                            target_stream_targets_path=targets_v4_csv,
                            base_metrics=None,
                            train_cfg=experiment.train,
                        )
                        f_csv = output_dir / "_tmp_final_v4" / "target_metrics_v4.csv"
                        b_csv = output_dir / "_tmp_best_v4" / "target_metrics_v4.csv"
                        if f_csv.is_file() and b_csv.is_file():
                            fdf = pd.read_csv(f_csv)
                            bdf = pd.read_csv(b_csv)
                            fdf = fdf[fdf["split"].astype(str) == "test"]
                            bdf = bdf[bdf["split"].astype(str) == "test"]
                            fk = fdf.set_index("target_id")
                            bk = bdf.set_index("target_id")
                            for target_id in sorted(set(fk.index) | set(bk.index)):
                                fval = float(fk.loc[target_id]["r2"]) if target_id in fk.index else float("nan")
                                bval = float(bk.loc[target_id]["r2"]) if target_id in bk.index else float("nan")
                                compare_rows.append(
                                    {
                                        "metric": f"target_v4_r2::{target_id}",
                                        "final_checkpoint": fval,
                                        "best_checkpoint": bval,
                                        "delta_best_minus_final": bval - fval,
                                    }
                                )
                    pd.DataFrame(compare_rows).to_csv(output_dir / "checkpoint_comparison_summary.csv", index=False)
            if edge_all and _is_main() and backward_mode in {"edge_step_pi", "sample_hybrid_target_edge_step_pi"}:
                targets_v4_csv = (experiment.project_root / "data/reference/v4/target_stream_targets.csv").resolve()
                pi_normalizer = None
                if y_edge_mean is not None and y_edge_std is not None:
                    pi_normalizer = {
                        "mean": y_edge_mean,
                        "std": y_edge_std,
                        "columns": y_edge_cols or list(STREAM_EDGE_FEATURE_SLOTS),
                        "mass_flow_log1p_mean": y_edge_mass_log_mean,
                        "mass_flow_log1p_std": y_edge_mass_log_std,
                        "mass_flow_log1p_count": y_edge_mass_log_count,
                        "mass_flow_physical_p05": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p05"),
                        "mass_flow_physical_p25": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p25"),
                        "mass_flow_physical_p75": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p75"),
                        "mass_flow_physical_p90": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p90"),
                        "mass_flow_physical_p95": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p95"),
                        "mass_flow_physical_p99": y_edge_physical_quantiles.get("Mass_Flow", {}).get("p99"),
                    }
                metric_rows = None
                final_metric_splits = getattr(experiment.train, "final_target_edge_metric_splits", ["val"])
                if isinstance(final_metric_splits, str):
                    final_metric_splits = [final_metric_splits]
                final_metric_split_set = {str(name).strip().lower() for name in final_metric_splits if str(name).strip()}
                save_best_plots = bool(
                    getattr(experiment.train, "save_best_actual_vs_pred_plots", False)
                    or getattr(experiment.train, "save_target_edge_actual_vs_pred_plots", False)
                    or getattr(experiment.train, "save_pi_all_edge_actual_vs_pred_plots", False)
                )
                save_all_edge_plots = bool(
                    getattr(experiment.train, "save_pi_all_edge_actual_vs_pred_plots", False)
                    or save_best_plots
                )
                all_edge_metric_acc = None
                for split_name, ld in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
                    if ld is None:
                        continue
                    if split_name.lower() not in final_metric_split_set:
                        continue
                    next_metric_rows = collect_target_edge_10d_metric_rows_from_loader(
                        model=model,
                        loader=ld,
                        device=device,
                        train_cfg=experiment.train,
                        data_cfg=experiment.data,
                        use_amp=use_amp,
                        split_name=split_name,
                        target_stream_targets_path=targets_v4_csv if targets_v4_csv.is_file() else None,
                        normalizer=pi_normalizer,
                        progress_label=f"{split_name}/target-edge",
                        progress_interval=50,
                    )
                    metric_rows = (
                        next_metric_rows
                        if metric_rows is None
                        else pd.concat((metric_rows, next_metric_rows), ignore_index=True)
                    )
                    all_edge_metric_acc = collect_pi_all_edge_property_metrics_from_loader(
                        model=model,
                        loader=ld,
                        device=device,
                        train_cfg=experiment.train,
                        data_cfg=experiment.data,
                        use_amp=use_amp,
                        split_name=split_name,
                        normalizer=pi_normalizer,
                        store_scatter_samples=save_all_edge_plots,
                        accumulator=all_edge_metric_acc,
                        progress_label=f"{split_name}/all-edge",
                        progress_interval=50,
                    )
                if metric_rows is not None:
                    te10_payload = write_target_edge_10d_metric_artifacts(
                        out_dir=output_dir,
                        metric_rows=metric_rows,
                        train_cfg=experiment.train,
                        target_stream_targets_path=targets_v4_csv if targets_v4_csv.is_file() else None,
                        save_actual_vs_pred_plots=save_best_plots,
                    )
                    if all_edge_metric_acc is not None:
                        te10_payload.update(all_edge_metric_acc.write_artifacts(output_dir))
                    metrics_path = output_dir / "metrics.json"
                    metrics_payload: dict[str, Any] = {}
                    if metrics_path.is_file():
                        try:
                            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                        except Exception:
                            metrics_payload = {}
                    metrics_payload.update({k: float(v) for k, v in te10_payload.items()})
                    metrics_payload["final_epoch"] = int(final_epoch)
                    metrics_payload["best_epoch"] = int(best_epoch)
                    metrics_payload["final_val_loss"] = float(final_val_loss) if final_val_loss is not None else float("nan")
                    metrics_payload["best_val_loss"] = float(best_val_loss) if best_val_loss is not None else float("nan")
                    metrics_payload["evaluation_checkpoint_path"] = str(eval_ckpt_path) if eval_ckpt_path is not None else ""
                    metrics_path.write_text(
                        json.dumps(enrich_metrics_json(metrics_payload, experiment.train), indent=2),
                        encoding="utf-8",
                    )
                    print(f"  target_edge_10d metrics: {output_dir / 'target_edge_10d_metrics.csv'}", flush=True)
    except KeyboardInterrupt as exc:
        if _is_main():
            ex = locals().get("experiment")
            od = locals().get("output_dir")
            opt = locals().get("optimizer")
            if od is not None and isinstance(od, Path) and ex is not None:
                try:
                    lr_now = float(opt.param_groups[0]["lr"]) if opt is not None else float(  # type: ignore[union-attr]
                        getattr(ex.train, "learning_rate", 0.0)
                    )
                except Exception:
                    lr_now = float(getattr(ex.train, "learning_rate", 0.0))
                _write_training_run_failed_artifact(
                    od,
                    exc,
                    experiment_path=experiment_path,
                    final_epoch_completed=int(locals().get("final_epoch", 0)),
                    edge_all=bool(locals().get("edge_all", False)),
                    batch_size=int(getattr(ex.train, "batch_size", 0)),
                    learning_rate=lr_now,
                )
        raise
    except Exception as exc:
        if _is_main():
            ex = locals().get("experiment")
            od = locals().get("output_dir")
            opt = locals().get("optimizer")
            if od is not None and isinstance(od, Path) and ex is not None:
                try:
                    lr_now = float(opt.param_groups[0]["lr"]) if opt is not None else float(  # type: ignore[union-attr]
                        getattr(ex.train, "learning_rate", 0.0)
                    )
                except Exception:
                    lr_now = float(getattr(ex.train, "learning_rate", 0.0))
                _write_training_run_failed_artifact(
                    od,
                    exc,
                    experiment_path=experiment_path,
                    final_epoch_completed=int(locals().get("final_epoch", 0)),
                    edge_all=bool(locals().get("edge_all", False)),
                    batch_size=int(getattr(ex.train, "batch_size", 0)),
                    learning_rate=lr_now,
                )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise
    finally:
        if _is_main() and not args.no_training_plots and train_history.get("epoch"):
            _save_training_curves(
                train_history,
                val_history,
                output_dir / "plots",
                best_epoch=(int(best_epoch) if best_epoch > 0 else None),
                best_val_loss=best_val_loss,
                evaluation_checkpoint_path=str(eval_ckpt_path),
            )
        if _is_main() and train_history.get("epoch"):
            hist_path = output_dir / "train_val_history.json"
            serializable_train = {k: [float(x) for x in v] for k, v in train_history.items()}
            serializable_val = {k: [float(x) for x in v] for k, v in val_history.items()}
            hist_path.write_text(
                json.dumps({"train": serializable_train, "val": serializable_val}, indent=2),
                encoding="utf-8",
            )
        if _is_main() and locals().get("edge_all") and val_history.get("epoch"):
            _od = locals().get("output_dir")
            _ex = locals().get("experiment")
            if isinstance(_od, Path) and _ex is not None:
                _main_json = _od / "metrics_main_targets.json"
                _edge_mean_json = _od / "metrics_edge_features_mean.json"
                if not _main_json.is_file() or not _edge_mean_json.is_file():
                    _wh = float(getattr(_ex.train, "answer_edge_weight_h2", 5.0))
                    _wc = float(getattr(_ex.train, "answer_edge_weight_co2", 5.0))
                    write_edge_all_metrics_json_from_val_history(
                        _od,
                        val_history,
                        w_h2=_wh,
                        w_co2=_wc,
                    )
        if _is_main() and val_history.get("epoch"):
            _save_val_r2_and_mae_plots(
                val_history,
                output_dir / "plots",
                edge_all=edge_all,
                target_only=target_only,
            )
        if _is_main():
            print(f"[metrics] per-epoch CSV: {output_dir / 'metrics_per_epoch.csv'}")
            try:
                _bm = best_metric
                _ex = experiment
                _od = output_dir
            except NameError:
                _bm = None
                _ex = None
                _od = None
            if (
                _bm is not None
                and _ex is not None
                and _od is not None
                and isinstance(_od, Path)
            ):
                hp_path = _od / "best_hyperparameters.json"
                hp_payload = {
                    "best_metric_name": args.best_metric_name,
                    "best_metric_value": float(_bm),
                    "best_epoch": int(best_epoch),
                    "experiment_yaml": str(experiment_path),
                    "config": _jsonable_config(asdict(_ex)),
                }
                hp_path.parent.mkdir(parents=True, exist_ok=True)
                hp_path.write_text(json.dumps(hp_payload, indent=2), encoding="utf-8")
                try:
                    _write_hpo_run_context(
                        _od,
                        experiment=_ex,
                        experiment_path=experiment_path,
                        best_metric=float(_bm),
                        best_epoch=int(best_epoch),
                        final_epoch=int(locals().get("final_epoch", 0)),
                        best_val_loss=locals().get("best_val_loss"),
                        final_val_loss=locals().get("final_val_loss"),
                        eval_checkpoint_path=locals().get("eval_ckpt_path"),
                    )
                except Exception as hpo_exc:
                    print(f"[hpo][warn] failed to write hpo_run_context.json: {hpo_exc}")
        cleanup_ddp()

    return {
        "best_metric": float(best_metric) if best_metric is not None else float("nan"),
        "best_epoch": float(best_epoch),
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    train_one_run(args)


if __name__ == "__main__":
    main()
