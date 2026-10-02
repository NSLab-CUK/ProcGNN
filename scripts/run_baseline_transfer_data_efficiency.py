#!/usr/bin/env python3
"""Leakage-safe transfer data-efficiency experiments for graph baselines.

Each held-out process is source-pretrained once using the same source manifests
as the proposed-model transfer experiment.  Every target ratio then starts
from that immutable source state, fits its feature/target and node scalers on
that ratio only, selects the best epoch on the fixed target validation split,
and reports the untouched target test split.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import run_final_baseline_experiments as baseline  # noqa: E402

DEFAULT_SPLIT_ROOT = "data/splits/single_process_full_unseen_60_20_20"
DEFAULT_SUBSET_ROOT = "outputs/0819final/data_efficiency/subsets"
DEFAULT_OUTPUT_ROOT = "outputs/0819final/baselines_transfer_0909"
DEFAULT_MODELS = ("GCN", "GIN", "GAT", "Graphormer", "SAT", "GraphToSFILES")
DEFAULT_RATIOS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _ratio_label(value: float) -> str:
    return f"ratio_{int(round(float(value) * 100)):03d}"


def _graph_mode(model_code: str) -> str:
    if model_code == "GraphToSFILES":
        return "node_topology_raw_edges"
    if model_code in {"Graphormer", "SAT"}:
        return "node_topology"
    return "node_topology_only"


def _model_config(*, model_code: str, epochs: int, patience: int, batch_size: int, seed: int) -> dict[str, Any]:
    config: dict[str, Any] = {
        "hidden_dim": 64, "num_layers": 3, "gat_num_heads": 4,
        "lr": 1.0e-3, "weight_decay": 0.0, "epochs": int(epochs),
        "batch_size": int(batch_size), "patience": int(patience), "seed": int(seed),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if model_code in {"Graphormer", "SAT", "GraphToSFILES"}:
        config.update({
            "name": model_code.lower(), "hidden_dim": 128, "num_layers": 4,
            "num_heads": 4, "ffn_dim": 512, "dropout": 0.1,
            "attention_dropout": 0.1, "max_degree": 16,
            "max_spatial_distance": 16,
        })
        if model_code == "SAT":
            config["structure_hops"] = 2
        if model_code == "GraphToSFILES":
            config.update({"name": "graph_to_sfiles_combined", "num_layers": 6, "num_heads": 8, "lap_pe_dim": 8})
    return config


def _new_model(model_code: str, config: Mapping[str, Any]) -> Any:
    from process_graph.baselines.models.joint import JointProcessBaseline
    from process_graph.baselines.models.graphormer import GraphormerBaseline
    from process_graph.baselines.models.sat import SATBaseline
    from process_graph.baselines.models.graph_to_sfiles_combined import GraphToSFILESCombinedBaseline

    if model_code == "Graphormer":
        return GraphormerBaseline(config)
    if model_code == "SAT":
        return SATBaseline(config)
    if model_code == "GraphToSFILES":
        return GraphToSFILESCombinedBaseline(config)
    return JointProcessBaseline(baseline.MULTI_MODELS[model_code], config)


def _materialize_views(*, root: Path, fold: int, manifests: Mapping[str, Path]) -> Path:
    """Give the baseline builder the expected ProcessN/foldNN manifest layout."""
    target = root / f"fold_{fold:02d}"
    for split, source in manifests.items():
        if not source.is_file():
            raise FileNotFoundError(f"missing {split} manifest: {source}")
        frame = pd.read_csv(source)
        if "process_id" not in frame.columns:
            raise KeyError(f"{source} has no process_id")
        ids = frame["process_id"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)
        for process_id in sorted(ids.unique()):
            destination = root / f"Process{process_id}" / f"fold_{fold:02d}" / f"{split}.csv"
            subset = frame.loc[ids.eq(process_id)].reset_index(drop=True)
            destination.parent.mkdir(parents=True, exist_ok=True)
            subset.to_csv(destination, index=False)
    return root


def _load_arrays(
    *, process_ids: Sequence[int], fold: int, views_root: Path, splits: Sequence[str],
    input_policy: str, graph_mode: str, fit_builder: Any | None = None,
) -> dict[int, dict[str, Any]]:
    from process_graph.baselines.dataset import build_target_edge_rows, materialize_arrays
    from process_graph.baselines.preprocessing import BaselinePreprocessor

    loaded: dict[int, dict[str, Any]] = {}
    for process_id in process_ids:
        kwargs = {
            "project_root": PROJECT_ROOT,
            "experiment_config_path": PROJECT_ROOT / "configs/experiment/process_surrogate_edge_all_v3.yaml",
            "splits_dir": views_root,
            "fold": int(fold), "process_ids": [int(process_id)],
            "input_policy": input_policy, "graph_mode": graph_mode,
        }
        rows = {split: build_target_edge_rows(split=split, **kwargs) for split in splits}
        first = next(iter(splits))
        arrays: dict[str, Any] = {}
        builder = fit_builder
        for split in splits:
            arrays[split], builder = materialize_arrays(rows[split], builder, fit=(split == first and builder is None))
        preprocessor = BaselinePreprocessor().fit(arrays[first].x, arrays[first].y)
        loaded[int(process_id)] = {
            "rows": rows, "arrays": arrays, "builder": builder, "preprocessor": preprocessor,
            "x": {split: preprocessor.transform_x(arrays[split].x) for split in splits},
            "y": {split: preprocessor.transform_y(arrays[split].y) for split in splits if split == first},
        }
    return loaded


def _inputs(data: Mapping[int, Mapping[str, Any]], split: str, graph_input: bool) -> dict[int, Any]:
    return {pid: item["arrays"][split].graph_samples if graph_input else item["x"][split] for pid, item in data.items()}


def _targets(data: Mapping[int, Mapping[str, Any]], split: str) -> dict[int, np.ndarray]:
    return {pid: item["preprocessor"].transform_y(item["arrays"][split].y) for pid, item in data.items()}


def _initialize_target_model(model: Any, model_code: str, process_id: int, data: Mapping[str, Any]) -> None:
    """Construct target-shaped head/scaler without taking an optimisation step."""
    graph_input = bool(model.requires_graph_input)
    train_x = data["arrays"]["train"].graph_samples if graph_input else data["x"]["train"]
    train_y = data["preprocessor"].transform_y(data["arrays"]["train"].y)
    if model_code in {"Graphormer", "SAT", "GraphToSFILES"}:
        model.fit_joint({process_id: (train_x, train_y)}, None)
    else:
        model.fit({process_id: train_x}, {process_id: train_y}, None)


def _fit_warm_start(
    *, model: Any, model_code: str, process_id: int, data: Mapping[str, Any], epochs: int, patience: int, seed: int,
) -> None:
    """Fine-tune an already constructed model; no reinitialization and no scaler refit."""
    if model.model is None:
        raise RuntimeError("warm-start target model was not initialized")
    graph_input = bool(model.requires_graph_input)
    train_x = data["arrays"]["train"].graph_samples if graph_input else data["x"]["train"]
    val_x = data["arrays"]["val"].graph_samples if graph_input else data["x"]["val"]
    train_y = data["preprocessor"].transform_y(data["arrays"]["train"].y)
    val_y = data["preprocessor"].transform_y(data["arrays"]["val"].y)
    optimizer = torch.optim.Adam(model.model.parameters(), lr=float(model.lr), weight_decay=float(model.weight_decay))
    rng = np.random.default_rng(int(seed))
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    model.history = []
    for epoch in range(1, int(epochs) + 1):
        indices = rng.permutation(len(train_y))
        losses: list[float] = []
        model.model.train()
        for start in range(0, len(indices), int(model.batch_size)):
            index = indices[start:start + int(model.batch_size)]
            batch_x = np.asarray(train_x)[index] if not graph_input else [train_x[int(i)] for i in index]
            prediction = model._predict_tensor(process_id, batch_x, training=True) if model_code == "GraphToSFILES" else model._predict_tensor(process_id, batch_x)
            target = torch.as_tensor(np.asarray(train_y)[index], dtype=torch.float32, device=model.device)
            loss = F.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.model.eval()
        validation: list[float] = []
        with torch.no_grad():
            for start in range(0, len(val_y), int(model.batch_size)):
                batch_x = val_x[start:start + int(model.batch_size)]
                prediction = model._predict_tensor(process_id, batch_x)
                target = torch.as_tensor(np.asarray(val_y[start:start + int(model.batch_size)]), dtype=torch.float32, device=model.device)
                validation.append(float(F.mse_loss(prediction, target).cpu()))
        val_loss = float(np.mean(validation)) if validation else float(np.mean(losses))
        model.history.append({"epoch": float(epoch), "train_loss": float(np.mean(losses)), "val_loss": val_loss})
        model.epochs_completed = epoch
        if val_loss < best_loss:
            best_loss = val_loss
            model.best_epoch = epoch
            best_state = copy.deepcopy(model.model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= int(patience):
                break
    if best_state is not None:
        model.model.load_state_dict(best_state)


def _evaluate(*, model: Any, model_code: str, process_id: int, fold: int, data: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    from process_graph.baselines.training.evaluator import write_evaluation_artifacts

    output_dir.mkdir(parents=True, exist_ok=True)
    graph_input = bool(model.requires_graph_input)
    metrics: dict[str, Any] = {}
    for split in ("val", "test"):
        arrays = data["arrays"][split]
        value = arrays.graph_samples if graph_input else data["x"][split]
        predictions = data["preprocessor"].inverse_y(model.predict_process(process_id, value))
        metrics[split] = write_evaluation_artifacts(out_dir=output_dir, arrays=arrays, y_pred=predictions, fold=fold, split=split)
        baseline._print_target_edge_r2(output_dir=output_dir, split=split, model_code=model_code, process_id=process_id, fold=fold)
    baseline._write_feature_schema(output_dir, data)
    baseline._write_json(output_dir / "preprocessing.json", data["preprocessor"].state_dict())
    model.save(output_dir / "model.pt")
    return metrics


def _run_one(
    *, model_code: str, heldout: int, fold: int, ratio: float, split_root: Path, subset_root: Path,
    output_root: Path, cache_root: Path, input_policy: str, epochs: int, pretrain_epochs: int,
    patience: int, batch_size: int, seed: int, resume: bool,
) -> None:
    label = _ratio_label(ratio)
    run_dir = output_root / model_code / f"heldout_P{heldout:02d}" / f"fold_{fold:02d}" / label
    if resume and baseline._completed(run_dir):
        print(f"[transfer-baseline][SKIP] {model_code} P{heldout:02d} F{fold:02d} {label}", flush=True)
        return
    heldout_root = split_root / f"heldout_P{heldout:02d}"
    metadata = _load_json(heldout_root / "metadata.json")
    source_ids = [int(value) for value in metadata.get("source_process_ids", [])]
    if not source_ids:
        raise RuntimeError(f"source_process_ids missing: {heldout_root / 'metadata.json'}")
    graph_mode = _graph_mode(model_code)
    state_path = cache_root / "pretrain" / model_code / f"heldout_P{heldout:02d}.pt"
    if state_path.is_file():
        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        source_state = payload["state_dict"]
        print(f"[transfer-baseline][PRETRAIN-CACHE] {model_code} P{heldout:02d}", flush=True)
    else:
        # Materializing 54k source graph samples is expensive.  It is done
        # only for the one source-pretrain cache, never once per ratio.
        source_views = _materialize_views(
            root=cache_root / "views" / f"heldout_P{heldout:02d}", fold=1,
            manifests={"train": heldout_root / "source_train.csv", "val": heldout_root / "source_val.csv"},
        )
        # A source test split is intentionally never loaded or evaluated.
        source_data = _load_arrays(
            process_ids=source_ids, fold=1, views_root=source_views,
            splits=("train", "val"), input_policy=input_policy, graph_mode=graph_mode,
        )
        source_model = _new_model(model_code, _model_config(model_code=model_code, epochs=pretrain_epochs, patience=patience, batch_size=batch_size, seed=seed + heldout))
        graph_input = bool(source_model.requires_graph_input)
        train_x, val_x = _inputs(source_data, "train", graph_input), _inputs(source_data, "val", graph_input)
        train_y, val_y = _targets(source_data, "train"), _targets(source_data, "val")
        print(f"[transfer-baseline][PRETRAIN-START] {model_code} P{heldout:02d} source_processes={source_ids}", flush=True)
        if model_code in {"Graphormer", "SAT", "GraphToSFILES"}:
            source_model.fit_joint({pid: (train_x[pid], train_y[pid]) for pid in source_ids}, {pid: (val_x[pid], val_y[pid]) for pid in source_ids})
        else:
            source_model.fit(train_x, train_y, val_data=(val_x, val_y))
        source_state = copy.deepcopy(source_model.model.state_dict())
        state_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": source_state, "model_code": model_code, "heldout_process": heldout, "source_process_ids": source_ids}, state_path)
        print(f"[transfer-baseline][PRETRAIN-DONE] {model_code} P{heldout:02d}", flush=True)
        del source_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    target_views = _materialize_views(
        root=run_dir / "split_views", fold=fold,
        manifests={
            "train": subset_root / f"heldout_P{heldout:02d}" / f"fold_{fold:02d}" / f"{label}.csv",
            "val": heldout_root / f"fold_{fold:02d}" / "target_val.csv",
            "test": heldout_root / f"fold_{fold:02d}" / "target_test.csv",
        },
    )
    target = _load_arrays(process_ids=[heldout], fold=fold, views_root=target_views, splits=("train", "val", "test"), input_policy=input_policy, graph_mode=graph_mode)[heldout]
    target_model = _new_model(model_code, _model_config(model_code=model_code, epochs=0, patience=patience, batch_size=batch_size, seed=seed + heldout * 1000 + fold))
    _initialize_target_model(target_model, model_code, heldout, target)
    target_model.model.load_state_dict(source_state, strict=True)
    started = time.perf_counter()
    _fit_warm_start(model=target_model, model_code=model_code, process_id=heldout, data=target, epochs=epochs, patience=patience, seed=seed + heldout * 1000 + fold)
    seconds = time.perf_counter() - started
    metrics = _evaluate(model=target_model, model_code=model_code, process_id=heldout, fold=fold, data=target, output_dir=run_dir)
    subset_meta = _load_json(subset_root / f"heldout_P{heldout:02d}" / f"fold_{fold:02d}" / f"{label}_indices.json")
    summary = {
        "scope": "single_process_transfer_data_efficiency", "model_code": model_code,
        "heldout_process": heldout, "fold": fold, "mode": "transfer", "sample_ratio": ratio,
        "sample_count": int(len(target["arrays"]["train"].y)), "transfer_pool_size": subset_meta.get("transfer_pool_size"),
        "subset_hash": subset_meta.get("merged_row_indices_sha256"), "source_process_ids": source_ids,
        "input_policy": input_policy, "graph_mode": graph_mode,
        "pretrained_state": str(state_path), "preprocessing_fit_scope": "current_ratio_training_subset_only",
        "validation_manifest": str(heldout_root / f"fold_{fold:02d}" / "target_val.csv"),
        "test_manifest": str(heldout_root / f"fold_{fold:02d}" / "target_test.csv"),
        "validation_and_test_are_same_manifest": False,
        "val_target_edge_property_mean_r2": baseline._metric_value(metrics["val"]),
        "test_target_edge_property_mean_r2": baseline._metric_value(metrics["test"]),
        "training_seconds": seconds, "best_epoch": int(target_model.best_epoch), "epochs_completed": int(target_model.epochs_completed),
        "parameter_count": int(sum(item.numel() for item in target_model.model.parameters())),
    }
    summary.update(baseline._property_metric_scalars(run_dir, "val"))
    summary.update(baseline._property_metric_scalars(run_dir, "test"))
    baseline._write_json(run_dir / "run_summary.json", summary)
    baseline._write_json(run_dir / "status.json", {"status": "completed"})
    print(f"[transfer-baseline][DONE] {model_code} P{heldout:02d} F{fold:02d} {label} seconds={seconds:.1f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GNN/transformer baseline transfer data-efficiency protocol")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=list(DEFAULT_MODELS))
    parser.add_argument("--heldout-processes", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--data-ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--split-root", default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--subset-root", default=DEFAULT_SUBSET_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-root", default="outputs/0819final/baselines_transfer_0909_cache")
    parser.add_argument("--baseline-root", default=str(baseline.DEFAULT_BASELINE_ROOT))
    parser.add_argument("--input-policy", default="process_spec_refs", choices=("process_spec_refs",))
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=260716)
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()
    baseline_root = _resolve(args.baseline_root)
    baseline._install_baselines(baseline_root)
    sources = baseline._verify_baseline_implementation(baseline_root)
    split_root, subset_root = _resolve(args.split_root), _resolve(args.subset_root)
    output_root, cache_root = _resolve(args.output_root), _resolve(args.cache_root)
    plan = {
        "models": args.models, "heldout_processes": args.heldout_processes, "folds": args.folds, "data_ratios": args.data_ratios,
        "expected_finetune_runs": len(args.models) * len(args.heldout_processes) * len(args.folds) * len(args.data_ratios),
        "expected_source_pretrains": len(args.models) * len(args.heldout_processes),
        "transfer_protocol": "source_pretrain_once_then_independent_ratio_finetune",
        "preprocessing_fit_scope": "current_ratio_training_subset_only",
        "validation_and_test_are_same_manifest": False, "baseline_sources": sources,
    }
    _write_json(output_root / "experiment_design.json", plan)
    print(json.dumps(plan, indent=2), flush=True)
    for model_code in args.models:
        for heldout in args.heldout_processes:
            for fold in args.folds:
                for ratio in args.data_ratios:
                    _run_one(model_code=model_code, heldout=int(heldout), fold=int(fold), ratio=float(ratio), split_root=split_root, subset_root=subset_root, output_root=output_root, cache_root=cache_root, input_policy=args.input_policy, epochs=args.max_epochs, pretrain_epochs=args.pretrain_epochs, patience=args.patience, batch_size=args.batch_size, seed=args.seed, resume=args.resume_existing)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
