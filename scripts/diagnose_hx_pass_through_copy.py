from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from process_graph.data.tabular_dataset import ProcessGraphTabularDataset  # noqa: E402
from process_graph.experiment.config_builders import (  # noqa: E402
    build_task_specs,
    model_yaml_to_encoder_config,
)
from process_graph.experiment.edge_step_pi_training import (  # noqa: E402
    _ensure_pi_output_keys,
)
from process_graph.experiment.edge_step_training import (  # noqa: E402
    build_target_edge_boolean_mask,
)
from process_graph.experiment.loaders import load_experiment_config  # noqa: E402
from process_graph.experiment.target_edge_10d_metrics import (  # noqa: E402
    extract_main_stream_metric_tensors,
)
from process_graph.models import ProcessSurrogateModel  # noqa: E402
from scripts.train_process_surrogate import _collate_for_experiment  # noqa: E402


FRACTION_PROPERTIES = [
    "Frac_H2O",
    "Frac_H2",
    "Frac_CH4",
    "Frac_CO2",
    "Frac_CO",
    "Frac_O2",
    "Frac_N2",
]
EDGE_IDS = {
    "wrong": "P03_E005",
    "paired": "P03_E010",
    "outlet": "P03_E014",
}
DEFAULT_RUN_NAMES = ("f1_hard", "f1_random", "f2_hard", "f2_random")


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _metric_stats(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float | int]:
    true = np.asarray(list(y_true), dtype=np.float64)
    pred = np.asarray(list(y_pred), dtype=np.float64)
    valid = np.isfinite(true) & np.isfinite(pred)
    true = true[valid]
    pred = pred[valid]
    if true.size == 0:
        return {
            "n": 0,
            "sst": float("nan"),
            "sse": float("nan"),
            "r2": float("nan"),
            "mae": float("nan"),
            "true_mean": float("nan"),
            "pred_mean": float("nan"),
            "true_std": float("nan"),
            "pred_std": float("nan"),
            "pred_std_over_true_std": float("nan"),
            "bias": float("nan"),
        }
    residual = pred - true
    sse = float(np.dot(residual, residual))
    centered = true - float(true.mean())
    sst = float(np.dot(centered, centered))
    true_std = float(true.std(ddof=0))
    pred_std = float(pred.std(ddof=0))
    return {
        "n": int(true.size),
        "sst": sst,
        "sse": sse,
        "r2": float(1.0 - sse / sst) if sst > 0.0 else float("nan"),
        "mae": float(np.abs(residual).mean()),
        "true_mean": float(true.mean()),
        "pred_mean": float(pred.mean()),
        "true_std": true_std,
        "pred_std": pred_std,
        "pred_std_over_true_std": (
            float(pred_std / true_std) if true_std > 0.0 else float("nan")
        ),
        "bias": float(residual.mean()),
    }


def _pearson(x: Iterable[float], y: Iterable[float]) -> float:
    left = np.asarray(list(x), dtype=np.float64)
    right = np.asarray(list(y), dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.size < 2 or float(left.std()) == 0.0 or float(right.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _discover_run(run_root: Path, name: str) -> dict[str, Path]:
    base = run_root / name
    checkpoints = sorted(base.rglob("best.pt"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise FileNotFoundError(f"No best.pt found under {base}")
    checkpoint = checkpoints[-1]
    run_tag = checkpoint.parent.name
    artifact_candidates = [
        path
        for path in base.iterdir()
        if path.is_dir() and path.name == run_tag
    ]
    if len(artifact_candidates) != 1:
        raise RuntimeError(
            f"Expected one artifact directory named {run_tag} under {base}, "
            f"found {len(artifact_candidates)}"
        )
    artifact_dir = artifact_candidates[0]
    scaler_path = artifact_dir / "y_edge_scaler.pt"
    raw_metric_path = artifact_dir / "target_edge_r2_by_property_raw.csv"
    detail_metric_path = artifact_dir / "target_edge_feature_metrics.csv"
    for required in (scaler_path, raw_metric_path, detail_metric_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    return {
        "checkpoint": checkpoint,
        "artifact_dir": artifact_dir,
        "scaler": scaler_path,
        "raw_metrics": raw_metric_path,
        "detail_metrics": detail_metric_path,
    }


def _local_config_path(checkpoint_payload: Mapping[str, Any]) -> Path:
    raw_path = str(checkpoint_payload.get("experiment", "") or "")
    basename = Path(raw_path).name
    if not basename:
        raise RuntimeError("Checkpoint does not record its experiment config path.")
    matches = list((PROJECT_ROOT / "configs").rglob(basename))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one local config named {basename}, found {len(matches)}: {matches}"
        )
    return matches[0].resolve()


def _make_p03_manifest(source_manifest: Path, destination: Path) -> pd.DataFrame:
    frame = pd.read_csv(source_manifest)
    required = {"process_id", "sample_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"Validation manifest is missing columns: {missing}")
    process_num = (
        frame["process_id"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(int)
    )
    p03 = frame.loc[process_num.eq(3)].copy()
    if p03.empty:
        raise RuntimeError(f"No Process3 rows found in validation manifest: {source_manifest}")
    if p03["sample_id"].astype(str).duplicated().any():
        duplicates = p03.loc[
            p03["sample_id"].astype(str).duplicated(keep=False), "sample_id"
        ].tolist()
        raise RuntimeError(f"Duplicate Process3 sample IDs in validation manifest: {duplicates[:10]}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    p03.to_csv(destination, index=False)
    return p03


def _restore_experiment(
    checkpoint_payload: Mapping[str, Any],
) -> tuple[Any, Path]:
    config_path = _local_config_path(checkpoint_payload)
    experiment = load_experiment_config(config_path)
    resolved = checkpoint_payload.get("config", {})
    if isinstance(resolved, Mapping):
        experiment.seed = int(resolved.get("seed", experiment.seed))
        resolved_data = resolved.get("data", {})
        if isinstance(resolved_data, Mapping):
            # Apply only path/cap fields whose runtime override is meaningful for
            # evaluation. Replacing the full nested DataConfig would turn
            # dataclass members such as fixed_tasks into plain dictionaries.
            for key in (
                "train_data_path",
                "val_data_path",
                "test_data_path",
                "train_split_manifest_path",
                "val_split_manifest_path",
                "test_split_manifest_path",
                "edge_all_cap_val_samples",
            ):
                if key in resolved_data and hasattr(experiment.data, key):
                    setattr(experiment.data, key, resolved_data[key])
    return experiment, config_path


def _edge_position(meta: Any, edge_id: str) -> int:
    matches = [
        idx
        for idx, value in enumerate(list(meta.canonical_edge_id))
        if str(value) == edge_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {edge_id} row for sample {list(meta.sample_id)[:1]}, "
            f"found {len(matches)}"
        )
    return int(matches[0])


def _run_checkpoint(
    *,
    name: str,
    paths: Mapping[str, Path],
    p03_manifest_path: Path,
    device: torch.device,
    max_samples: int,
    progress_interval: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    checkpoint_payload = torch.load(
        paths["checkpoint"], map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint_payload, Mapping):
        raise TypeError(f"Checkpoint payload must be a mapping: {paths['checkpoint']}")
    experiment, config_path = _restore_experiment(checkpoint_payload)
    checkpoint_manifest = _resolve(experiment.data.val_split_manifest_path)
    source_manifest = pd.read_csv(checkpoint_manifest)
    expected_ids = set(
        source_manifest.loc[
            source_manifest["process_id"].astype(str).eq("Process3"), "sample_id"
        ].astype(str)
    )
    diagnostic_manifest = pd.read_csv(p03_manifest_path)
    diagnostic_ids = set(diagnostic_manifest["sample_id"].astype(str))
    if diagnostic_ids != expected_ids:
        raise RuntimeError(
            f"{name}: diagnostic P03 manifest does not match checkpoint validation split."
        )

    oper_normalizer = checkpoint_payload.get("oper_normalizer")
    if not isinstance(oper_normalizer, Mapping):
        raise RuntimeError(f"{name}: checkpoint has no oper_normalizer.")
    oper_mean = torch.as_tensor(oper_normalizer["mean"], dtype=torch.float32)
    oper_std = torch.as_tensor(oper_normalizer["std"], dtype=torch.float32)
    y_edge_scaler = torch.load(paths["scaler"], map_location="cpu", weights_only=False)

    dataset = ProcessGraphTabularDataset(
        experiment.data.val_data_path,
        experiment.data,
        experiment.project_root,
        split_filter=None,
        split_manifest=p03_manifest_path,
        oper_mean=oper_mean,
        oper_std=oper_std,
    )
    if max_samples > 0:
        sample_count = min(int(max_samples), len(dataset))
        dataset = torch.utils.data.Subset(dataset, list(range(sample_count)))
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=_collate_for_experiment(experiment.data),
    )

    encoder_config = model_yaml_to_encoder_config(experiment.model, experiment.data)
    model = ProcessSurrogateModel(
        encoder_config=encoder_config,
        task_specs=build_task_specs(experiment.model, experiment.data),
    ).to(device)
    state_dict = checkpoint_payload.get("model_state_dict") or checkpoint_payload.get("model")
    if not isinstance(state_dict, Mapping):
        raise RuntimeError(f"{name}: checkpoint has no model state.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    rows: list[dict[str, Any]] = []
    observed_samples: set[str] = set()
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            batch_data = {
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in batch.model_kwargs.items()
            }
            targets_raw = {
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in batch.targets.items()
            }
            target_masks = {
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in batch.target_masks.items()
            }
            task_inputs = {
                head: {
                    key: value.to(device, non_blocking=device.type == "cuda")
                    for key, value in payload.items()
                }
                for head, payload in batch.task_inputs.items()
            }
            n_edges = int(batch_data["edge_index"].shape[1])
            target_edge_mask, _ = build_target_edge_boolean_mask(
                train_cfg=experiment.train,
                edge_export_meta=batch.edge_export_meta,
                edge_target_columns=list(batch.edge_target_columns),
                n_edges=n_edges,
                device=device,
            )
            batch_data["target_edge_mask"] = target_edge_mask
            outputs = model(batch_data, task_inputs=task_inputs)
            outputs = _ensure_pi_output_keys(
                outputs,
                train_cfg=experiment.train,
                data_cfg=experiment.data,
                edge_target_columns=list(batch.edge_target_columns),
                normalizer=y_edge_scaler,
            )
            pred, true, valid_mask, property_names = extract_main_stream_metric_tensors(
                outputs=outputs,
                targets_raw=targets_raw,
                target_masks=target_masks,
                train_cfg=experiment.train,
                data_cfg=experiment.data,
                edge_target_columns=list(batch.edge_target_columns),
                normalizer=y_edge_scaler,
            )
            pred = pred.detach().cpu()
            true = true.detach().cpu()
            valid_mask = valid_mask.detach().cpu().bool()
            positions = {
                key: _edge_position(batch.edge_export_meta, edge_id)
                for key, edge_id in EDGE_IDS.items()
            }
            sample_ids = {
                str(batch.edge_export_meta.sample_id[position])
                for position in positions.values()
            }
            if len(sample_ids) != 1:
                raise RuntimeError(
                    f"{name}: E005/E010/E014 sample keys do not match: {sample_ids}"
                )
            sample_id = next(iter(sample_ids))
            if sample_id in observed_samples:
                raise RuntimeError(f"{name}: duplicate paired sample {sample_id}")
            observed_samples.add(sample_id)

            mass_idx = property_names.index("Mass_Flow")
            co_idx = property_names.index("Frac_CO")
            co_true = float(true[positions["outlet"], co_idx])
            if co_true < 1.0e-4:
                co_bin = "near_zero"
            elif co_true < 0.01:
                co_bin = "transition"
            else:
                co_bin = "high"
            for property_name in FRACTION_PROPERTIES:
                property_idx = property_names.index(property_name)
                rows.append(
                    {
                        "checkpoint": name,
                        "process_id": "Process3",
                        "graph_id": sample_id,
                        "sample_id": sample_id,
                        "edge_semantic_id": EDGE_IDS["outlet"],
                        "property_name": property_name,
                        "e005_edge_tensor_index": positions["wrong"],
                        "e010_edge_tensor_index": positions["paired"],
                        "e014_edge_tensor_index": positions["outlet"],
                        "e010_true": float(true[positions["paired"], property_idx]),
                        "e010_pred": float(pred[positions["paired"], property_idx]),
                        "e014_true": float(true[positions["outlet"], property_idx]),
                        "e014_pred_original": float(pred[positions["outlet"], property_idx]),
                        "e014_pred_from_e010": float(pred[positions["paired"], property_idx]),
                        "e014_pred_from_wrong_pair": float(
                            pred[positions["wrong"], property_idx]
                        ),
                        "mass_flow_true": float(true[positions["outlet"], mass_idx]),
                        "fraction_valid_mask": bool(
                            valid_mask[positions["outlet"], property_idx]
                            and valid_mask[positions["paired"], property_idx]
                        ),
                        "co_bin": co_bin,
                    }
                )
            if progress_interval > 0 and (
                batch_index % progress_interval == 0 or batch_index == len(loader)
            ):
                elapsed = time.perf_counter() - started
                rate = batch_index / max(elapsed, 1.0e-12)
                eta = (len(loader) - batch_index) / max(rate, 1.0e-12)
                print(
                    f"[hx-pairing] {name} {batch_index}/{len(loader)} "
                    f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
    if len(observed_samples) != len(loader):
        raise RuntimeError(
            f"{name}: expected {len(loader)} unique samples, got {len(observed_samples)}"
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metadata = {
        "checkpoint": name,
        "checkpoint_path": str(paths["checkpoint"]),
        "artifact_dir": str(paths["artifact_dir"]),
        "config_path": str(config_path),
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", -1)),
        "validation_manifest": str(checkpoint_manifest),
        "validation_p03_samples": len(observed_samples),
        "device": str(device),
        "resolved_model": _jsonable(asdict(experiment.model)),
    }
    return pd.DataFrame(rows), metadata


def _build_fraction_metrics(samples: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    prediction_columns = {
        "P03_E010": "e010_pred",
        "P03_E014": "e014_pred_original",
    }
    for checkpoint, checkpoint_frame in samples.groupby("checkpoint", sort=False):
        for property_name, property_frame in checkpoint_frame.groupby(
            "property_name", sort=False
        ):
            valid = property_frame["fraction_valid_mask"].astype(bool)
            property_frame = property_frame.loc[valid]
            for edge_id, prediction_column in prediction_columns.items():
                target_column = "e010_true" if edge_id == "P03_E010" else "e014_true"
                stats = _metric_stats(
                    property_frame[target_column],
                    property_frame[prediction_column],
                )
                records.append(
                    {
                        "checkpoint": checkpoint,
                        "edge_semantic_id": edge_id,
                        "property_name": property_name,
                        **stats,
                    }
                )
    return pd.DataFrame(records)


def _build_copy_metrics(
    samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    copy_rows: list[dict[str, Any]] = []
    wrong_rows: list[dict[str, Any]] = []
    for checkpoint, checkpoint_frame in samples.groupby("checkpoint", sort=False):
        for property_name, property_frame in checkpoint_frame.groupby(
            "property_name", sort=False
        ):
            valid = property_frame["fraction_valid_mask"].astype(bool)
            property_frame = property_frame.loc[valid]
            original = _metric_stats(
                property_frame["e014_true"], property_frame["e014_pred_original"]
            )
            copied = _metric_stats(
                property_frame["e014_true"], property_frame["e014_pred_from_e010"]
            )
            wrong = _metric_stats(
                property_frame["e014_true"],
                property_frame["e014_pred_from_wrong_pair"],
            )
            copy_rows.append(
                {
                    "checkpoint": checkpoint,
                    "property_name": property_name,
                    **{f"original_{key}": value for key, value in original.items()},
                    **{f"copied_{key}": value for key, value in copied.items()},
                    "r2_change": float(copied["r2"]) - float(original["r2"]),
                    "sse_change": float(copied["sse"]) - float(original["sse"]),
                }
            )
            wrong_rows.append(
                {
                    "checkpoint": checkpoint,
                    "property_name": property_name,
                    **{f"original_{key}": value for key, value in original.items()},
                    **{f"correct_pair_{key}": value for key, value in copied.items()},
                    **{f"wrong_pair_{key}": value for key, value in wrong.items()},
                }
            )
    return pd.DataFrame(copy_rows), pd.DataFrame(wrong_rows)


def _build_bin_metrics(samples: pd.DataFrame) -> pd.DataFrame:
    co = samples.loc[samples["property_name"].eq("Frac_CO")].copy()
    rows: list[dict[str, Any]] = []
    for (checkpoint, bin_name), frame in co.groupby(
        ["checkpoint", "co_bin"], sort=False
    ):
        for prediction_name, column in (
            ("original", "e014_pred_original"),
            ("copied_from_e010", "e014_pred_from_e010"),
        ):
            stats = _metric_stats(frame["e014_true"], frame[column])
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "co_bin": bin_name,
                    "prediction": prediction_name,
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def _build_shuffle_metrics(
    samples: pd.DataFrame,
    *,
    seed: int,
    repeats: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for checkpoint_index, (checkpoint, checkpoint_frame) in enumerate(
        samples.groupby("checkpoint", sort=False)
    ):
        for property_index, (property_name, property_frame) in enumerate(
            checkpoint_frame.groupby("property_name", sort=False)
        ):
            property_frame = property_frame.sort_values("sample_id", kind="stable")
            true = property_frame["e014_true"].to_numpy(dtype=np.float64)
            paired = property_frame["e014_pred_from_e010"].to_numpy(dtype=np.float64)
            valid = property_frame["fraction_valid_mask"].to_numpy(dtype=bool)
            true = true[valid]
            paired = paired[valid]
            correct_stats = _metric_stats(true, paired)
            repeat_r2: list[float] = []
            for repeat in range(repeats):
                rng = np.random.default_rng(
                    int(seed) + checkpoint_index * 1000 + property_index * 100 + repeat
                )
                shuffled = paired[rng.permutation(len(paired))]
                stats = _metric_stats(true, shuffled)
                repeat_r2.append(float(stats["r2"]))
                rows.append(
                    {
                        "checkpoint": checkpoint,
                        "property_name": property_name,
                        "repeat": repeat,
                        "seed": (
                            int(seed)
                            + checkpoint_index * 1000
                            + property_index * 100
                            + repeat
                        ),
                        "correct_pair_r2": correct_stats["r2"],
                        **{f"shuffled_{key}": value for key, value in stats.items()},
                    }
                )
            mean_r2 = float(np.nanmean(repeat_r2))
            std_r2 = float(np.nanstd(repeat_r2))
            for row in rows[-repeats:]:
                row["shuffled_r2_mean"] = mean_r2
                row["shuffled_r2_std"] = std_r2
    return pd.DataFrame(rows)


def _ground_truth_equality(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    first_checkpoint = next(iter(samples["checkpoint"].unique()))
    frame = samples.loc[samples["checkpoint"].eq(first_checkpoint)]
    for property_name, property_frame in frame.groupby("property_name", sort=False):
        difference = np.abs(
            property_frame["e010_true"].to_numpy(dtype=np.float64)
            - property_frame["e014_true"].to_numpy(dtype=np.float64)
        )
        rows.append(
            {
                "property_name": property_name,
                "sample_count": int(len(property_frame)),
                "max_abs_difference": float(difference.max(initial=0.0)),
                "mismatch_count_at_1e_10": int(np.count_nonzero(difference > 1.0e-10)),
                "mismatch_count_at_1e_8": int(np.count_nonzero(difference > 1.0e-8)),
                "equal_ratio_at_1e_10": float(np.mean(difference <= 1.0e-10)),
                "equal_ratio_at_1e_8": float(np.mean(difference <= 1.0e-8)),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_target_co(
    *,
    samples: pd.DataFrame,
    run_paths: Mapping[str, Mapping[str, Path]],
    copy_metrics: pd.DataFrame,
    full_p03: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for checkpoint, paths in run_paths.items():
        raw = pd.read_csv(paths["raw_metrics"])
        co_row = raw.loc[raw["property_name"].eq("Frac_CO")]
        if len(co_row) != 1:
            raise RuntimeError(f"{checkpoint}: expected one aggregate Frac_CO row.")
        co_row = co_row.iloc[0]
        property_r2 = raw["R2"].to_numpy(dtype=np.float64)
        p03_copy = copy_metrics.loc[
            (copy_metrics["checkpoint"].eq(checkpoint))
            & (copy_metrics["property_name"].eq("Frac_CO"))
        ]
        if len(p03_copy) != 1:
            raise RuntimeError(f"{checkpoint}: expected one P03 Frac_CO copy metric row.")
        p03_copy = p03_copy.iloc[0]
        detail = pd.read_csv(paths["detail_metrics"])
        r2_column = "R2" if "R2" in detail.columns else "r2"
        detail_row = detail.loc[
            detail["canonical_edge_id"].eq("P03_E014")
            & detail["feature_name"].eq("Frac_CO")
        ]
        if len(detail_row) != 1:
            raise RuntimeError(
                f"{checkpoint}: expected one stored P03_E014 Frac_CO metric row."
            )
        detail_row = detail_row.iloc[0]
        stored_p03_sse = float(detail_row["sse"])
        measured_p03_sse = float(p03_copy["original_sse"])
        if full_p03 and not math.isclose(
            stored_p03_sse, measured_p03_sse, rel_tol=2.0e-5, abs_tol=1.0e-8
        ):
            raise RuntimeError(
                f"{checkpoint}: reproduced P03 SSE {measured_p03_sse} does not match "
                f"stored best-checkpoint SSE {stored_p03_sse}."
            )
        if full_p03:
            copied_sse = float(p03_copy["copied_sse"])
            counterfactual_sse = float(co_row["SSE"]) - stored_p03_sse + copied_sse
            counterfactual_r2 = 1.0 - counterfactual_sse / float(co_row["SST"])
            counterfactual_mean_r2 = float(
                (
                    property_r2.sum()
                    - float(co_row["R2"])
                    + counterfactual_r2
                )
                / len(property_r2)
            )
        else:
            copied_sse = float("nan")
            counterfactual_sse = float("nan")
            counterfactual_r2 = float("nan")
            counterfactual_mean_r2 = float("nan")
        rows.append(
            {
                "checkpoint": checkpoint,
                "full_p03_validation": bool(full_p03),
                "target_co_n": int(co_row["n"]),
                "target_co_sst": float(co_row["SST"]),
                "original_target_co_sse": float(co_row["SSE"]),
                "original_target_co_r2": float(co_row["R2"]),
                "stored_p03_e014_co_r2": float(detail_row[r2_column]),
                "stored_p03_e014_co_sse": stored_p03_sse,
                "measured_p03_e014_co_sse": measured_p03_sse,
                "copied_p03_e014_co_sse": copied_sse,
                "counterfactual_target_co_sse": counterfactual_sse,
                "counterfactual_target_co_r2": counterfactual_r2,
                "target_co_r2_change": counterfactual_r2 - float(co_row["R2"]),
                "original_p03_sse_fraction": stored_p03_sse / float(co_row["SSE"]),
                "counterfactual_p03_sse_fraction": (
                    copied_sse / counterfactual_sse
                    if full_p03 and counterfactual_sse > 0.0
                    else float("nan")
                ),
                "original_target_mean_r2": float(property_r2.mean()),
                "counterfactual_target_mean_r2": counterfactual_mean_r2,
                "target_mean_r2_change": counterfactual_mean_r2
                - float(property_r2.mean()),
            }
        )
    return pd.DataFrame(rows)


def _checkpoint_summary(
    metadata: list[dict[str, Any]],
    samples: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    co = samples.loc[samples["property_name"].eq("Frac_CO")]
    for item in metadata:
        checkpoint = item["checkpoint"]
        frame = co.loc[co["checkpoint"].eq(checkpoint)]
        rows.append(
            {
                key: value
                for key, value in item.items()
                if key != "resolved_model"
            }
            | {
                "e010_e014_co_prediction_correlation": _pearson(
                    frame["e010_pred"], frame["e014_pred_original"]
                ),
                "e010_e014_co_prediction_mae": float(
                    np.abs(
                        frame["e010_pred"].to_numpy(dtype=np.float64)
                        - frame["e014_pred_original"].to_numpy(dtype=np.float64)
                    ).mean()
                ),
                "missing_e005_samples": 0,
                "missing_e010_samples": 0,
                "missing_e014_samples": 0,
                "duplicate_e005_samples": 0,
                "duplicate_e010_samples": 0,
                "duplicate_e014_samples": 0,
                "sample_key_mismatch_count": 0,
            }
        )
    return pd.DataFrame(rows)


def _fraction_copy_summary(copy_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for checkpoint, frame in copy_metrics.groupby("checkpoint", sort=False):
        rows.append(
            {
                "checkpoint": checkpoint,
                "property_count": int(len(frame)),
                "original_fraction_macro_r2": float(frame["original_r2"].mean()),
                "copied_fraction_macro_r2": float(frame["copied_r2"].mean()),
                "fraction_macro_r2_change": float(
                    frame["copied_r2"].mean() - frame["original_r2"].mean()
                ),
                "original_fraction_total_sse": float(frame["original_sse"].sum()),
                "copied_fraction_total_sse": float(frame["copied_sse"].sum()),
                "fraction_total_sse_change": float(
                    frame["copied_sse"].sum() - frame["original_sse"].sum()
                ),
                "fraction_total_sse_relative_change": float(
                    frame["copied_sse"].sum() / frame["original_sse"].sum() - 1.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _decision(
    fraction_metrics: pd.DataFrame,
    copy_metrics: pd.DataFrame,
    wrong_metrics: pd.DataFrame,
    shuffle_metrics: pd.DataFrame,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for checkpoint in copy_metrics["checkpoint"].unique():
        e010_co = fraction_metrics.loc[
            fraction_metrics["checkpoint"].eq(checkpoint)
            & fraction_metrics["edge_semantic_id"].eq("P03_E010")
            & fraction_metrics["property_name"].eq("Frac_CO"),
            "r2",
        ].iloc[0]
        copy_co = copy_metrics.loc[
            copy_metrics["checkpoint"].eq(checkpoint)
            & copy_metrics["property_name"].eq("Frac_CO")
        ].iloc[0]
        wrong_co = wrong_metrics.loc[
            wrong_metrics["checkpoint"].eq(checkpoint)
            & wrong_metrics["property_name"].eq("Frac_CO")
        ].iloc[0]
        shuffled_co = shuffle_metrics.loc[
            shuffle_metrics["checkpoint"].eq(checkpoint)
            & shuffle_metrics["property_name"].eq("Frac_CO")
        ]
        rows.append(
            {
                "checkpoint": checkpoint,
                "e010_co_r2": float(e010_co),
                "e014_original_co_r2": float(copy_co["original_r2"]),
                "e014_copied_co_r2": float(copy_co["copied_r2"]),
                "e014_wrong_pair_co_r2": float(wrong_co["wrong_pair_r2"]),
                "shuffled_co_r2_mean": float(shuffled_co["shuffled_r2_mean"].iloc[0]),
                "copy_co_r2_gain": float(copy_co["r2_change"]),
                "fraction_r2_gains": {
                    property_name: float(value)
                    for property_name, value in copy_metrics.loc[
                        copy_metrics["checkpoint"].eq(checkpoint),
                        ["property_name", "r2_change"],
                    ].itertuples(index=False, name=None)
                },
            }
        )
    meaningful_upstream = np.mean(
        [row["e010_co_r2"] > 0.30 for row in rows]
    )
    meaningful_copy = np.mean(
        [row["copy_co_r2_gain"] > 0.10 for row in rows]
    )
    correct_beats_controls = np.mean(
        [
            row["e014_copied_co_r2"]
            > max(row["e014_wrong_pair_co_r2"], row["shuffled_co_r2_mean"]) + 0.10
            for row in rows
        ]
    )
    if meaningful_upstream >= 0.75 and meaningful_copy >= 0.75 and correct_beats_controls >= 0.75:
        verdict = "HX pairing delivery failure is the primary cause."
    elif meaningful_upstream < 0.50 and meaningful_copy < 0.50:
        verdict = "Upstream E010 prediction failure is the primary cause."
    elif meaningful_copy >= 0.50 and correct_beats_controls >= 0.50:
        verdict = "Upstream prediction and HX pairing delivery failures both contribute."
    else:
        verdict = "The current diagnostic is insufficient for a single-cause conclusion."
    return {
        "verdict": verdict,
        "checkpoint_decisions": rows,
        "decision_rates": {
            "meaningful_upstream_rate": float(meaningful_upstream),
            "meaningful_copy_gain_rate": float(meaningful_copy),
            "correct_pair_beats_controls_rate": float(correct_beats_controls),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose P03 HX3 pass-through delivery by copying the same-sample "
            "P03_E010 fraction prediction to P03_E014."
        )
    )
    parser.add_argument(
        "--run-root",
        default="outputs/0731_f_sampler_ablation_fixed",
        help="Root containing f1_hard, f1_random, f2_hard, and f2_random.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hx_pairing_diagnostic",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=list(DEFAULT_RUN_NAMES),
        choices=list(DEFAULT_RUN_NAMES),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--shuffle-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=260731)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Diagnostic smoke limit. Use 0 for the required full P03 validation.",
    )
    parser.add_argument("--progress-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    run_root = _resolve(args.run_root)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    run_paths = {
        name: _discover_run(run_root, name)
        for name in args.runs
    }

    first_payload = torch.load(
        run_paths[args.runs[0]]["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    first_experiment, _ = _restore_experiment(first_payload)
    validation_manifest = _resolve(first_experiment.data.val_split_manifest_path)
    p03_manifest_path = output_dir / "p03_validation_manifest.csv"
    p03_manifest = _make_p03_manifest(validation_manifest, p03_manifest_path)
    expected_p03_count = len(p03_manifest)
    if args.max_samples > 0:
        print(
            f"[hx-pairing][smoke] limiting each checkpoint to {args.max_samples} "
            "P03 samples; aggregate counterfactual metrics will be marked unavailable.",
            flush=True,
        )

    sample_frames: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []
    for name in args.runs:
        frame, checkpoint_metadata = _run_checkpoint(
            name=name,
            paths=run_paths[name],
            p03_manifest_path=p03_manifest_path,
            device=device,
            max_samples=max(0, int(args.max_samples)),
            progress_interval=max(0, int(args.progress_interval)),
        )
        sample_frames.append(frame)
        metadata.append(checkpoint_metadata)
    samples = pd.concat(sample_frames, ignore_index=True)

    equality = _ground_truth_equality(samples)
    if (
        int(equality["mismatch_count_at_1e_10"].sum()) != 0
        or int(equality["mismatch_count_at_1e_8"].sum()) != 0
    ):
        raise RuntimeError(
            "P03_E010/P03_E014 fraction ground truth is not identical; "
            "counterfactual evaluation aborted."
        )
    fraction_metrics = _build_fraction_metrics(samples)
    copy_metrics, wrong_metrics = _build_copy_metrics(samples)
    bin_metrics = _build_bin_metrics(samples)
    shuffle_metrics = _build_shuffle_metrics(
        samples,
        seed=int(args.seed),
        repeats=max(5, int(args.shuffle_repeats)),
    )
    full_p03 = int(args.max_samples) <= 0 and len(
        samples.loc[samples["checkpoint"].eq(args.runs[0]), "sample_id"].unique()
    ) == expected_p03_count
    aggregate_metrics = _aggregate_target_co(
        samples=samples,
        run_paths=run_paths,
        copy_metrics=copy_metrics,
        full_p03=full_p03,
    )
    checkpoint_summary = _checkpoint_summary(metadata, samples)
    fraction_copy_summary = _fraction_copy_summary(copy_metrics)
    decision = _decision(
        fraction_metrics,
        copy_metrics,
        wrong_metrics,
        shuffle_metrics,
    )

    checkpoint_summary.to_csv(output_dir / "checkpoint_summary.csv", index=False)
    fraction_copy_summary.to_csv(
        output_dir / "p03_fraction_copy_summary.csv", index=False
    )
    fraction_metrics.to_csv(
        output_dir / "p03_e010_e014_fraction_metrics.csv", index=False
    )
    bin_metrics.to_csv(output_dir / "p03_co_bin_metrics.csv", index=False)
    aggregate_metrics.to_csv(
        output_dir / "target_co_counterfactual_metrics.csv", index=False
    )
    wrong_metrics.to_csv(output_dir / "wrong_pair_control.csv", index=False)
    shuffle_metrics.to_csv(output_dir / "shuffled_pair_control.csv", index=False)
    samples.to_csv(output_dir / "sample_level_predictions.csv", index=False)
    equality.to_csv(output_dir / "ground_truth_pairing_validation.csv", index=False)

    summary = {
        "purpose": "P03 HX3 same-sample pass-through prediction copy diagnostic",
        "created_at_unix": time.time(),
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "seed": int(args.seed),
        "shuffle_repeats": max(5, int(args.shuffle_repeats)),
        "max_samples": int(args.max_samples),
        "full_p03_validation": bool(full_p03),
        "p03_validation_sample_count": int(expected_p03_count),
        "evaluated_sample_count_per_checkpoint": {
            checkpoint: int(
                samples.loc[
                    samples["checkpoint"].eq(checkpoint), "sample_id"
                ].nunique()
            )
            for checkpoint in args.runs
        },
        "edge_ids": EDGE_IDS,
        "fraction_properties": FRACTION_PROPERTIES,
        "ground_truth_equality": equality.to_dict(orient="records"),
        "checkpoint_metadata": metadata,
        "fraction_copy_summary": fraction_copy_summary.to_dict(orient="records"),
        "aggregate_target_co": aggregate_metrics.to_dict(orient="records"),
        "decision": decision,
    }
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"[hx-pairing][done] wrote diagnostic artifacts to {output_dir}",
        flush=True,
    )
    print(f"[hx-pairing][verdict] {decision['verdict']}", flush=True)


if __name__ == "__main__":
    main()
