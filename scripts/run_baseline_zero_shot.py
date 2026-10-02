#!/usr/bin/env python3
"""Leakage-safe zero-shot evaluation for graph baseline models.

For each held-out process, the model and every learned normalization statistic
are fitted on *source-process train data only*.  Target-process validation and
test samples are never used for fitting or calibration; their topology is used
only to instantiate the schema-compatible decoder before loading the frozen
source state.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_final_baseline_experiments as baseline  # noqa: E402
import run_baseline_transfer_data_efficiency as transfer  # noqa: E402


DEFAULT_MODELS = ("GCN", "GIN", "GAT", "Graphormer", "SAT", "GraphToSFILES")
DEFAULT_SPLIT_ROOT = "data/splits/single_process_full_unseen_60_20_20"
DEFAULT_OUTPUT_ROOT = "outputs/0819final/baselines_zero_shot_0910"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _restore_node_scaler(model: Any, state: Mapping[str, Any]) -> None:
    """Restore source-only node normalization without calling ``fit`` on target."""
    scaler = model.scaler
    scaler.mean = None if state.get("mean") is None else np.asarray(state["mean"], dtype=np.float64)
    scaler.scale = None if state.get("scale") is None else np.asarray(state["scale"], dtype=np.float64)
    scaler.eps = float(state.get("eps", 1.0e-12))
    scaler.flat_value_mode = bool(state.get("flat_value_mode", False))


class _PropertyBlockPreprocessor:
    """A source-only 10-property scaler compatible with variable edge counts.

    Rows have ``n_target_edges * 10`` columns and therefore cannot be
    concatenated across processes.  Every contiguous 10D property block has
    the identical schema, so the scaler is fitted after reshaping all source
    training targets to ``[-1, 10]`` and then restored to the original shape.
    """

    property_width = 10

    def __init__(self, scaler: Any) -> None:
        self.scaler = scaler

    @classmethod
    def fit_source(cls, source_data: Mapping[int, Mapping[str, Any]]) -> "_PropertyBlockPreprocessor":
        from process_graph.baselines.preprocessing import ArrayStandardizer

        blocks = []
        for item in source_data.values():
            values = np.asarray(item["arrays"]["train"].y, dtype=np.float64)
            if values.ndim != 2 or values.shape[1] % cls.property_width:
                raise ValueError(f"source target shape must be [N, E*10], got {values.shape}")
            blocks.append(values.reshape(-1, cls.property_width))
        return cls(ArrayStandardizer().fit(np.concatenate(blocks, axis=0)))

    def _transform(self, values: np.ndarray, inverse: bool) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] % self.property_width:
            raise ValueError(f"target shape must be [N, E*10], got {array.shape}")
        flat = array.reshape(-1, self.property_width)
        transformed = self.scaler.inverse_transform(flat) if inverse else self.scaler.transform(flat)
        return transformed.reshape(array.shape)

    def transform_y(self, values: np.ndarray) -> np.ndarray:
        return self._transform(values, inverse=False)

    def inverse_y(self, values: np.ndarray) -> np.ndarray:
        return self._transform(values, inverse=True)

    def state_dict(self) -> dict[str, Any]:
        return {"kind": "source_property_block_standardizer", "property_width": self.property_width, "y_scaler": self.scaler.state_dict()}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "_PropertyBlockPreprocessor":
        from process_graph.baselines.preprocessing import ArrayStandardizer

        if state.get("kind") != "source_property_block_standardizer":
            raise ValueError("zero-shot cache has an incompatible preprocessing state")
        if int(state.get("property_width", -1)) != cls.property_width:
            raise ValueError("zero-shot cache property width is incompatible")
        return cls(ArrayStandardizer.from_state_dict(state.get("y_scaler", {})))


def _global_source_preprocessor(source_data: Mapping[int, Mapping[str, Any]]) -> _PropertyBlockPreprocessor:
    """Fit one 10D property scaler across source-process *training* blocks only."""
    return _PropertyBlockPreprocessor.fit_source(source_data)


def _source_cache(
    *, model_code: str, heldout: int, source_ids: Sequence[int], heldout_root: Path,
    cache_root: Path, input_policy: str, pretrain_epochs: int, patience: int,
    batch_size: int, seed: int, force_rebuild: bool,
) -> dict[str, Any]:
    """Train/cache a source-only model whose normalizers are also source-only."""
    cache_path = cache_root / "zero_shot_pretrain" / model_code / f"heldout_P{heldout:02d}.pt"
    if cache_path.is_file() and not force_rebuild:
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        required = {"state_dict", "node_scaler", "preprocessor"}
        if required.issubset(payload):
            print(f"[zero-shot][SOURCE-CACHE] {model_code} P{heldout:02d}", flush=True)
            return payload

    graph_mode = transfer._graph_mode(model_code)
    source_views = transfer._materialize_views(
        root=cache_root / "zero_shot_views" / f"heldout_P{heldout:02d}", fold=1,
        manifests={"train": heldout_root / "source_train.csv", "val": heldout_root / "source_val.csv"},
    )
    source_data = transfer._load_arrays(
        process_ids=source_ids, fold=1, views_root=source_views, splits=("train", "val"),
        input_policy=input_policy, graph_mode=graph_mode,
    )
    shared_preprocessor = _global_source_preprocessor(source_data)
    graph_input = model_code != "MLP"
    train_x = transfer._inputs(source_data, "train", graph_input)
    val_x = transfer._inputs(source_data, "val", graph_input)
    train_y = {pid: shared_preprocessor.transform_y(item["arrays"]["train"].y) for pid, item in source_data.items()}
    val_y = {pid: shared_preprocessor.transform_y(item["arrays"]["val"].y) for pid, item in source_data.items()}
    model = transfer._new_model(
        model_code,
        transfer._model_config(model_code=model_code, epochs=pretrain_epochs, patience=patience, batch_size=batch_size, seed=seed + heldout),
    )
    print(f"[zero-shot][SOURCE-START] {model_code} P{heldout:02d} source_processes={list(source_ids)}", flush=True)
    if model_code in {"Graphormer", "SAT", "GraphToSFILES"}:
        model.fit_joint({pid: (train_x[pid], train_y[pid]) for pid in source_ids}, {pid: (val_x[pid], val_y[pid]) for pid in source_ids})
    else:
        model.fit(train_x, train_y, val_data=(val_x, val_y))
    payload = {
        "state_dict": copy.deepcopy(model.model.state_dict()),
        "node_scaler": model.scaler.state_dict(),
        "preprocessor": shared_preprocessor.state_dict(),
        "model_code": model_code,
        "heldout_process": heldout,
        "source_process_ids": list(source_ids),
        "normalization_fit_scope": "source_process_train_only",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(f"[zero-shot][SOURCE-DONE] {model_code} P{heldout:02d}", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def _initialize_schema_only(model: Any, model_code: str, process_id: int, target: Mapping[str, Any], preprocessor: Any) -> None:
    """Build target output shape only; no target optimization or scaler fitting is retained."""
    graphs = target["arrays"]["val"].graph_samples if model.requires_graph_input else target["x"]["val"]
    y = preprocessor.transform_y(target["arrays"]["val"].y)
    if model_code in {"Graphormer", "SAT", "GraphToSFILES"}:
        model.fit_joint({process_id: (graphs, y)}, None)
    else:
        model.fit({process_id: graphs}, {process_id: y}, None)


def _evaluate_zero_shot(*, model: Any, model_code: str, process_id: int, fold: int, data: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    """Write normal evaluation artifacts without inventing a target ``train`` split."""
    from process_graph.baselines.training.evaluator import write_evaluation_artifacts

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {}
    for split in ("val", "test"):
        arrays = data["arrays"][split]
        value = arrays.graph_samples if model.requires_graph_input else data["x"][split]
        prediction = data["preprocessor"].inverse_y(model.predict_process(process_id, value))
        metrics[split] = write_evaluation_artifacts(out_dir=output_dir, arrays=arrays, y_pred=prediction, fold=fold, split=split)
        baseline._print_target_edge_r2(output_dir=output_dir, split=split, model_code=model_code, process_id=process_id, fold=fold)
    # The target schema is known, but this evaluator deliberately has no target
    # training split.  Preserve that fact instead of labeling validation rows as training rows.
    schema_arrays = data["arrays"]["val"]
    schema_rows = data["rows"]["val"]
    baseline._write_json(output_dir / "feature_schema.json", {
        "feature_names": list(schema_arrays.feature_names), "feature_builder": data["builder"].state_dict(),
        "feature_audit": schema_rows.feature_audit or {}, "input_policy": schema_rows.input_policy,
        "fit_scope": "source_process_train_only; target rows not used to fit preprocessing",
    })
    baseline._write_json(output_dir / "target_schema.json", {
        "target_names": list(schema_arrays.target_names), "target_schema": schema_rows.target_schema or [],
    })
    baseline._write_json(output_dir / "preprocessing.json", data["preprocessor"].state_dict())
    model.save(output_dir / "model.pt")
    return metrics


def _run_one(
    *, model_code: str, heldout: int, fold: int, split_root: Path, output_root: Path,
    cache_root: Path, input_policy: str, pretrain_epochs: int, patience: int,
    batch_size: int, seed: int, resume: bool, force_rebuild_source: bool,
) -> None:
    run_dir = output_root / model_code / f"heldout_P{heldout:02d}" / f"fold_{fold:02d}"
    if resume and baseline._completed(run_dir):
        print(f"[zero-shot][SKIP] {model_code} P{heldout:02d} F{fold:02d}", flush=True)
        return
    heldout_root = split_root / f"heldout_P{heldout:02d}"
    metadata = _load_json(heldout_root / "metadata.json")
    source_ids = [int(value) for value in metadata.get("source_process_ids", [])]
    if not source_ids:
        raise RuntimeError(f"source_process_ids missing: {heldout_root / 'metadata.json'}")
    source = _source_cache(
        model_code=model_code, heldout=heldout, source_ids=source_ids, heldout_root=heldout_root,
        cache_root=cache_root, input_policy=input_policy, pretrain_epochs=pretrain_epochs,
        patience=patience, batch_size=batch_size, seed=seed, force_rebuild=force_rebuild_source,
    )
    target_views = transfer._materialize_views(
        root=run_dir / "split_views", fold=fold,
        # The baseline builder requires a train manifest to select the permitted
        # operating-reference columns.  Supplying source_train here keeps that
        # audit (including proxy exclusion) source-only; target train is never
        # materialized or read by this zero-shot protocol.
        manifests={
            "train": heldout_root / "source_train.csv",
            "val": heldout_root / f"fold_{fold:02d}" / "target_val.csv",
            "test": heldout_root / f"fold_{fold:02d}" / "target_test.csv",
        },
    )
    # ``_materialize_views`` partitions source_train by its own process IDs,
    # whereas the baseline builder looks specifically for Process<heldout>.
    # Install the unmodified source-only manifest at that lookup location.
    source_reference_manifest = target_views / f"Process{heldout}" / f"fold_{fold:02d}" / "train.csv"
    source_reference_manifest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(heldout_root / "source_train.csv", source_reference_manifest)
    target = transfer._load_arrays(
        process_ids=[heldout], fold=fold, views_root=target_views, splits=("val", "test"),
        input_policy=input_policy, graph_mode=transfer._graph_mode(model_code),
    )[heldout]
    source_preprocessor = _PropertyBlockPreprocessor.from_state_dict(source["preprocessor"])
    # Use source-only scaling at inference.  The temporary target scaler created
    # by the shape builder is overwritten before any prediction is made.
    target["preprocessor"] = source_preprocessor
    model = transfer._new_model(
        model_code,
        transfer._model_config(model_code=model_code, epochs=0, patience=patience, batch_size=batch_size, seed=seed + heldout * 1000 + fold),
    )
    _initialize_schema_only(model, model_code, heldout, target, source_preprocessor)
    model.model.load_state_dict(source["state_dict"], strict=True)
    _restore_node_scaler(model, source["node_scaler"])
    started = time.perf_counter()
    metrics = _evaluate_zero_shot(model=model, model_code=model_code, process_id=heldout, fold=fold, data=target, output_dir=run_dir)
    seconds = time.perf_counter() - started
    summary = {
        "scope": "single_process_unseen_zero_shot", "mode": "zero_shot",
        "model_code": model_code, "heldout_process": heldout, "fold": fold,
        "source_process_ids": source_ids, "input_policy": input_policy,
        "graph_mode": transfer._graph_mode(model_code),
        "pretrained_state": str(cache_root / "zero_shot_pretrain" / model_code / f"heldout_P{heldout:02d}.pt"),
        "preprocessing_fit_scope": "source_process_train_only",
        "target_feature_reference_fit_scope": "source_process_train_only",
        "target_training_samples_used": 0, "target_validation_samples_used_for_calibration": 0,
        "validation_manifest": str(heldout_root / f"fold_{fold:02d}" / "target_val.csv"),
        "test_manifest": str(heldout_root / f"fold_{fold:02d}" / "target_test.csv"),
        "validation_and_test_are_same_manifest": False,
        "val_target_edge_property_mean_r2": baseline._metric_value(metrics["val"]),
        "test_target_edge_property_mean_r2": baseline._metric_value(metrics["test"]),
        "evaluation_seconds": seconds,
        "parameter_count": int(sum(item.numel() for item in model.model.parameters())),
    }
    summary.update(baseline._property_metric_scalars(run_dir, "val"))
    summary.update(baseline._property_metric_scalars(run_dir, "test"))
    baseline._write_json(run_dir / "run_summary.json", summary)
    baseline._write_json(run_dir / "status.json", {"status": "completed"})
    print(f"[zero-shot][DONE] {model_code} P{heldout:02d} F{fold:02d} seconds={seconds:.1f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate graph baselines under source-only zero-shot transfer")
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=list(DEFAULT_MODELS))
    parser.add_argument("--heldout-processes", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--split-root", default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-root", default="outputs/0819final/baselines_transfer_0909_cache")
    parser.add_argument("--baseline-root", default=str(baseline.DEFAULT_BASELINE_ROOT))
    parser.add_argument("--input-policy", choices=("process_spec_refs",), default="process_spec_refs")
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=260716)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--force-rebuild-source", action="store_true")
    args = parser.parse_args()
    baseline_root = transfer._resolve(args.baseline_root)
    baseline._install_baselines(baseline_root)
    sources = baseline._verify_baseline_implementation(baseline_root)
    split_root, output_root, cache_root = transfer._resolve(args.split_root), transfer._resolve(args.output_root), transfer._resolve(args.cache_root)
    design = {
        "models": args.models, "heldout_processes": args.heldout_processes, "folds": args.folds,
        "expected_zero_shot_evaluations": len(args.models) * len(args.heldout_processes) * len(args.folds),
        "expected_source_pretrains": len(args.models) * len(args.heldout_processes),
        "protocol": "source_pretrain_and_all_normalization_on_source_train_only; frozen evaluation_on_heldout_process",
        "target_training_samples_used": 0, "target_validation_samples_used_for_calibration": 0,
        "baseline_sources": sources,
    }
    _write_json(output_root / "experiment_design.json", design)
    print(json.dumps(design, indent=2), flush=True)
    for model_code in args.models:
        for heldout in args.heldout_processes:
            for fold in args.folds:
                _run_one(model_code=model_code, heldout=int(heldout), fold=int(fold), split_root=split_root, output_root=output_root, cache_root=cache_root, input_policy=args.input_policy, pretrain_epochs=args.pretrain_epochs, patience=args.patience, batch_size=args.batch_size, seed=args.seed, resume=args.resume_existing, force_rebuild_source=args.force_rebuild_source)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
