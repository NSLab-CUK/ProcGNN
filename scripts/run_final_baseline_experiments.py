from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_ROOT = PROJECT_ROOT.parent / "화학공정Baselines"

SINGLE_MODELS: dict[str, str] = {
    "M1": "configs/baseline/m1_svr.yaml",
    "M2": "configs/baseline/m2_random_forest.yaml",
    "M3": "configs/baseline/m3_xgboost.yaml",
    "B1": "configs/baseline/b1_kriging_kr31.yaml",
    "B2": "configs/baseline/b2_cubic_rbf.yaml",
    "B3": "configs/baseline/b3_gtl_ann.yaml",
    "B4": "configs/baseline/b4_cumene_efficiency_ann.yaml",
    "B5": "configs/baseline/b5_cumene_destruction_ann.yaml",
    "B6": "configs/baseline/b6_reusable_distillation_ann.yaml",
    "B7": "configs/baseline/b7_distillation_boundary_gp.yaml",
    "G1": "configs/baseline/g1_flowsheet_gcn.yaml",
    "G2": "configs/baseline/g2_flowsheet_gin_final.yaml",
    "G3": "configs/baseline/g3_flowsheet_gat_final.yaml",
    "Graphormer": "graphormer",
    "SAT": "sat",
    "GraphToSFILES": "graph_to_sfiles_combined",
}
MULTI_MODELS = {"B6": "mlp", "GCN": "gcn", "GIN": "gin", "GAT": "gat", "Graphormer": "graphormer", "SAT": "sat", "GraphToSFILES": "graph_to_sfiles_combined"}
MLP_BASELINE_MODELS = {"B3", "B4", "B5", "B6"}
TARGET_EDGE_PROPERTY_NAMES = (
    "Temp", "Pres", "Frac_H2O", "Frac_H2", "Frac_CH4", "Frac_CO2",
    "Frac_CO", "Frac_O2", "Frac_N2", "Mass_Flow",
)
R2_SST_EPS = 1.0e-12
CONSTANT_TARGET_R2_VALUE = 0.0


def _effective_input_policy(requested_policy: str, model_code: str) -> str:
    """Apply the reduced-control sensitivity only to ANN/MLP baselines."""

    if requested_policy == "core_controls" and model_code not in MLP_BASELINE_MODELS:
        return "process_spec_refs"
    return requested_policy


def _property_slug(name: str) -> str:
    return str(name).strip().lower()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _property_frame_from_predictions(output_dir: Path, split: str) -> pd.DataFrame:
    path = output_dir / f"{split}_predictions.csv"
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    predictions = pd.read_csv(path)
    required = {"property_name", "true_value", "predicted_value"}
    if not required.issubset(predictions.columns):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for property_name, group in predictions.groupby("property_name", sort=False):
        true = pd.to_numeric(group["true_value"], errors="coerce").to_numpy(float)
        pred = pd.to_numeric(group["predicted_value"], errors="coerce").to_numpy(float)
        valid = np.isfinite(true) & np.isfinite(pred)
        true = true[valid]
        pred = pred[valid]
        n_valid = int(len(true))
        true_mean = float(true.mean()) if n_valid else float("nan")
        sst = float(((true - true_mean) ** 2).sum()) if n_valid else float("nan")
        sse = float(((true - pred) ** 2).sum()) if n_valid else float("nan")
        if n_valid < 2:
            r2 = float("nan")
            status = "insufficient_finite_pairs"
        elif sst <= R2_SST_EPS:
            r2 = CONSTANT_TARGET_R2_VALUE
            status = "constant_target_assigned_0.0"
        else:
            r2 = float(1.0 - sse / sst)
            status = "ok"
        rows.append({
            "property_name": str(property_name), "R2": r2,
            "R2_status": status, "R2_n_valid": n_valid,
            "R2_n_total": int(len(group)), "R2_SST": sst, "R2_SSE": sse,
            "true_mean": true_mean,
        })
    return pd.DataFrame(rows)


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_baseline_config(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    config = _read_yaml(path)
    base = config.pop("base_config", None)
    return _merge(_read_yaml(root / base), config) if base else config


def _completed(path: Path) -> bool:
    if not (path / "status.json").is_file():
        return False
    try:
        status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return status.get("status") == "completed" and (path / "test_metrics.json").is_file()


def _install_baselines(root: Path) -> None:
    source = str((root / "src").resolve())
    if source not in sys.path:
        sys.path.insert(0, source)


def _verify_baseline_implementation(root: Path) -> dict[str, str]:
    """Fail fast unless the imported external baseline contains leakage fixes."""
    import process_graph.baselines.dataset as dataset_module
    import process_graph.baselines.metrics as metrics_module
    import process_graph.baselines.models.simple_gnn as gnn_module
    import process_graph.baselines.models.graphormer as graphormer_module
    import process_graph.baselines.models.sat as sat_module
    import process_graph.baselines.models.graph_to_sfiles_combined as combined_module

    expected_root = (root / "src").resolve()
    modules = {
        "dataset": Path(dataset_module.__file__).resolve(),
        "metrics": Path(metrics_module.__file__).resolve(),
        "simple_gnn": Path(gnn_module.__file__).resolve(),
        "graphormer": Path(graphormer_module.__file__).resolve(),
        "sat": Path(sat_module.__file__).resolve(),
        "graph_to_sfiles_combined": Path(combined_module.__file__).resolve(),
    }
    for name, path in modules.items():
        if not path.is_relative_to(expected_root):
            raise RuntimeError(
                f"baseline import provenance mismatch for {name}: imported={path} "
                f"expected_under={expected_root}"
            )
    required_markers = {
        "dataset": (
            "feature_audit", "train_target_near_affine_proxy",
            "_strip_graph_edges_and_targets", "core_controls",
        ),
        "metrics": (
            "flatten_r2_physical_scale_diagnostic_only", "R2_force_finite",
            "R2_status", "CONSTANT_TARGET_R2_VALUE",
        ),
        "simple_gnn": ("target-edge query or slot embedding",),
        "graphormer": (
            "directed-topology Graphormer",
            "edge attributes",
            "concat(h_source, h_destination)",
        ),
        "sat": (
            "Structure-Aware Transformer",
            "directed GIN",
            "concat(h_source, h_destination)",
        ),
        "graph_to_sfiles_combined": (
            "Graph-to-SFILES Combined encoder",
            "GenericSharedStreamHead",
            "proposed_head_used\": False",
        ),
    }
    for name, markers in required_markers.items():
        text = modules[name].read_text(encoding="utf-8")
        missing = [marker for marker in markers if marker not in text]
        if missing:
            raise RuntimeError(
                f"external baseline implementation is stale: file={modules[name]} "
                f"missing_markers={missing}. Synchronize the leakage-safe baseline checkout."
            )
    payload = {name: str(path) for name, path in modules.items()}
    print(f"[baseline-source] root={root}", flush=True)
    for name, path in payload.items():
        print(f"[baseline-source] {name}={path}", flush=True)
    print("[baseline-source] leakage_safe=true finite_r2_diagnostics=true", flush=True)
    return payload


def _write_feature_schema(output_dir: Path, data: dict[str, Any]) -> None:
    train_rows = data["rows"]["train"]
    arrays = data["arrays"]["train"]
    builder = data["builder"]
    _write_json(output_dir / "feature_schema.json", {
        "feature_names": list(arrays.feature_names),
        "feature_builder": builder.state_dict(),
        "feature_audit": train_rows.feature_audit or {},
        "input_policy": train_rows.input_policy,
    })
    _write_json(output_dir / "target_schema.json", {
        "target_names": list(arrays.target_names),
        "target_schema": train_rows.target_schema or [],
    })


def _prepare_views(split_root: Path, destination: Path, folds: list[int]) -> Path:
    """Materialize per-process views from the canonical joint fold manifests."""
    view_root = destination / "process_split_views"
    for fold in folds:
        fold_name = f"fold_{int(fold):02d}"
        source_fold = split_root / fold_name
        for split in ("train", "val", "test"):
            source = source_fold / f"{split}.csv"
            if not source.is_file():
                raise FileNotFoundError(f"missing joint split manifest: {source}")
            frame = pd.read_csv(source)
            if "process_id" not in frame.columns:
                raise KeyError(f"missing process_id column: {source}")
            process_numbers = (
                frame["process_id"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)
            )
            for process_id in sorted(process_numbers.unique()):
                target = view_root / f"Process{process_id}" / fold_name / f"{split}.csv"
                subset = frame.loc[process_numbers == process_id]
                if target.is_file():
                    try:
                        existing = pd.read_csv(target, usecols=["merged_row_index"])
                        expected = subset[["merged_row_index"]]
                        if existing.equals(expected.reset_index(drop=True)):
                            continue
                    except Exception:
                        pass
                target.parent.mkdir(parents=True, exist_ok=True)
                subset.to_csv(target, index=False)
    return view_root


def _load_process_fold(
    *,
    process_id: int,
    fold: int,
    views_root: Path,
    max_train_samples: int | None,
    max_val_samples: int | None,
    max_test_samples: int | None,
    input_policy: str,
    graph_mode: str = "node_only",
) -> dict[str, Any]:
    from process_graph.baselines.dataset import assert_no_sample_leakage, build_target_edge_rows, materialize_arrays
    from process_graph.baselines.preprocessing import BaselinePreprocessor

    kwargs = {
        "project_root": PROJECT_ROOT,
        "experiment_config_path": PROJECT_ROOT / "configs/experiment/process_surrogate_edge_all_v3.yaml",
        "splits_dir": views_root,
        "fold": int(fold),
        "process_ids": [int(process_id)],
        "input_policy": str(input_policy),
        "graph_mode": str(graph_mode),
    }
    rows = {
        "train": build_target_edge_rows(split="train", max_samples=max_train_samples, **kwargs),
        "val": build_target_edge_rows(split="val", max_samples=max_val_samples, **kwargs),
        "test": build_target_edge_rows(split="test", max_samples=max_test_samples, **kwargs),
    }
    assert_no_sample_leakage({name: value.rows for name, value in rows.items()})
    train, builder = materialize_arrays(rows["train"], fit=True)
    val, _ = materialize_arrays(rows["val"], builder)
    test, _ = materialize_arrays(rows["test"], builder)
    preprocessor = BaselinePreprocessor().fit(train.x, train.y)
    return {
        "rows": rows,
        "arrays": {"train": train, "val": val, "test": test},
        "builder": builder,
        "preprocessor": preprocessor,
        "x": {
            "train": preprocessor.transform_x(train.x),
            "val": preprocessor.transform_x(val.x),
            "test": preprocessor.transform_x(test.x),
        },
        "y": {
            "train": preprocessor.transform_y(train.y),
            "val": preprocessor.transform_y(val.y),
        },
    }


def _metric_value(payload: dict[str, Any]) -> float:
    try:
        return float(payload.get("target_edge_property_mean_r2", float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _refresh_joint_property_artifacts(output_dir: Path, split: str) -> float | None:
    """Pool every process and target edge by property before computing joint R2."""
    paths = sorted(output_dir.glob(f"Process*/{split}_property_metrics.csv"))
    if not paths:
        return None
    frames = [pd.read_csv(path) for path in paths]
    rows: list[dict[str, Any]] = []
    property_r2: dict[str, float] = {}
    for property_name in TARGET_EDGE_PROPERTY_NAMES:
        pieces = []
        for frame in frames:
            if "property_name" not in frame.columns:
                continue
            selected = frame.loc[frame["property_name"].astype(str).eq(property_name)].copy()
            if not selected.empty:
                pieces.append(selected)
        combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        required = {"R2_n_valid", "R2_SST", "R2_SSE", "true_mean"}
        if combined.empty or not required.issubset(combined.columns):
            n_total = 0
            sst = sse = true_mean = r2 = float("nan")
            status = "insufficient_finite_pairs"
        else:
            n_values = pd.to_numeric(combined["R2_n_valid"], errors="coerce").fillna(0.0)
            means = pd.to_numeric(combined["true_mean"], errors="coerce")
            within_sst = pd.to_numeric(combined["R2_SST"], errors="coerce")
            within_sse = pd.to_numeric(combined["R2_SSE"], errors="coerce")
            valid = (n_values > 0) & means.notna() & within_sst.notna() & within_sse.notna()
            n_values = n_values.loc[valid]
            means = means.loc[valid]
            within_sst = within_sst.loc[valid]
            within_sse = within_sse.loc[valid]
            n_total = int(n_values.sum())
            if n_total > 0:
                true_mean = float((means * n_values).sum() / n_total)
                sst = float((within_sst + n_values * (means - true_mean) ** 2).sum())
                sse = float(within_sse.sum())
            else:
                true_mean = sst = sse = float("nan")
            if n_total < 2:
                r2 = float("nan")
                status = "insufficient_finite_pairs"
            elif sst <= R2_SST_EPS:
                r2 = CONSTANT_TARGET_R2_VALUE
                status = "constant_target_assigned_0.0"
            else:
                r2 = float(1.0 - sse / sst)
                status = "ok"
        property_r2[property_name] = r2
        rows.append({
            "property_name": property_name,
            "R2": r2,
            "R2_status": status,
            "R2_n_valid": n_total,
            "R2_SST": sst,
            "R2_SSE": sse,
            "true_mean": true_mean,
        })
    frame = pd.DataFrame(rows)
    _write_csv(output_dir / f"{split}_property_metrics.csv", frame)
    finite = [value for value in property_r2.values() if np.isfinite(value)]
    mean_r2 = float(np.mean(finite)) if finite else float("nan")
    metrics_path = output_dir / f"{split}_metrics.json"
    payload = _load_json(metrics_path)
    payload.update({
        "primary_metric_name": "target_edge_property_mean_r2",
        "primary_metric_value": mean_r2,
        "target_edge_property_mean_r2": mean_r2,
        "target_edge_property_r2": property_r2,
        "target_edge_property_r2_valid_count": len(finite),
        "target_edge_property_r2_total_count": len(TARGET_EDGE_PROPERTY_NAMES),
        "target_edge_property_pooling": "all_processes_all_target_edges_by_property",
    })
    _write_json(metrics_path, payload)
    return mean_r2


def _artifact_target_edge_property_mean_r2(output_dir: Path, split: str) -> float:
    joint_value = _refresh_joint_property_artifacts(output_dir, split)
    if joint_value is not None:
        return joint_value
    metrics_path = output_dir / f"{split}_metrics.json"
    property_path = output_dir / f"{split}_property_metrics.csv"
    if not property_path.is_file():
        frame = _property_frame_from_predictions(output_dir, split)
        if frame.empty:
            return _metric_value(_load_json(metrics_path))
    else:
        try:
            frame = pd.read_csv(property_path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            frame = _property_frame_from_predictions(output_dir, split)
            if frame.empty:
                return _metric_value(_load_json(metrics_path))
    if not {"property_name", "R2"}.issubset(frame.columns):
        return float("nan")
    pooled = {name: float("nan") for name in TARGET_EDGE_PROPERTY_NAMES}
    for index, row in frame.iterrows():
        name = str(row.get("property_name", ""))
        if name not in pooled:
            continue
        try:
            property_r2 = float(row.get("R2", float("nan")))
        except (TypeError, ValueError):
            property_r2 = float("nan")
        n_valid = pd.to_numeric(pd.Series([row.get("R2_n_valid")]), errors="coerce").iloc[0]
        sst = pd.to_numeric(pd.Series([row.get("R2_SST")]), errors="coerce").iloc[0]
        if np.isfinite(n_valid) and n_valid >= 2 and np.isfinite(sst) and sst <= R2_SST_EPS:
            property_r2 = CONSTANT_TARGET_R2_VALUE
            frame.at[index, "R2"] = property_r2
            if "R2_status" in frame.columns:
                frame.at[index, "R2_status"] = "constant_target_assigned_0.0"
        pooled[name] = property_r2
    _write_csv(property_path, frame)
    finite = [property_r2 for property_r2 in pooled.values() if np.isfinite(property_r2)]
    value = float(np.mean(finite)) if finite else float("nan")
    payload = _load_json(metrics_path)
    payload.update({
        "primary_metric_name": "target_edge_property_mean_r2",
        "primary_metric_value": value,
        "target_edge_property_mean_r2": value,
        "target_edge_property_r2": pooled,
        "target_edge_property_r2_valid_count": len(finite),
        "target_edge_property_r2_total_count": len(TARGET_EDGE_PROPERTY_NAMES),
        "target_edge_property_pooling": "all_target_edges_by_property",
    })
    _write_json(metrics_path, payload)
    return value


def _property_metric_scalars(output_dir: Path, split: str) -> dict[str, float]:
    _artifact_target_edge_property_mean_r2(output_dir, split)
    metrics_path = output_dir / f"{split}_metrics.json"
    payload = _load_json(metrics_path)
    values = payload.get("target_edge_property_r2", {}) or {}
    return {
        f"{split}_target_edge_r2_{_property_slug(name)}": float(
            values.get(name, float("nan"))
        )
        for name in TARGET_EDGE_PROPERTY_NAMES
    }


def _fit_with_compact_convergence_log(model: Any, train_x: Any, train_y: Any, *, val_data: Any) -> None:
    """Run fit without flooding the terminal with repeated sklearn warnings."""
    try:
        from sklearn.exceptions import ConvergenceWarning
    except ImportError:  # pragma: no cover - sklearn is required by these baselines
        ConvergenceWarning = Warning  # type: ignore[assignment,misc]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train_x, train_y, val_data=val_data)
    convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
    for item in caught:
        if not issubclass(item.category, ConvergenceWarning):
            warnings.warn(str(item.message), item.category, stacklevel=2)
    if convergence:
        print(
            f"[baseline-fit][convergence-warning] count={len(convergence)} "
            "optimizer reached its configured iteration limit; metrics below are still exported",
            flush=True,
        )


def _format_r2(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    return f"{number:.5f}" if np.isfinite(number) else "nan"


def _print_target_edge_r2(
    *, output_dir: Path, split: str, model_code: str, process_id: int, fold: int
) -> None:
    main_value = _artifact_target_edge_property_mean_r2(output_dir, split)
    property_path = output_dir / f"{split}_property_metrics.csv"
    if property_path.is_file():
        property_frame = pd.read_csv(property_path)
        property_frame = property_frame[
            property_frame["property_name"].astype(str).isin(TARGET_EDGE_PROPERTY_NAMES)
        ]
        by_name = property_frame.set_index("property_name")["R2"].to_dict()
        values = " ".join(
            f"{name}={_format_r2(by_name.get(name))}"
            for name in TARGET_EDGE_PROPERTY_NAMES
        )
        print(
            f"[baseline-r2][{split}][model={model_code} process=P{process_id:02d} "
            f"fold={fold}] target_edge_property_mean_r2={_format_r2(main_value)}",
            flush=True,
        )
        print(f"  [pooled-by-property] {values}", flush=True)

    path = output_dir / f"{split}_target_edge_property_metrics.csv"
    if not path.is_file():
        print(f"[baseline-r2][{split}] missing={path}", flush=True)
        return
    frame = pd.read_csv(path)
    required = {"target_edge_id", "property_name", "R2"}
    if not required.issubset(frame.columns):
        print(f"[baseline-r2][{split}] invalid_columns={path}", flush=True)
        return
    for edge_id, rows in frame.groupby("target_edge_id", sort=False, dropna=False):
        edge_label = str(edge_id) if pd.notna(edge_id) and str(edge_id) else "target"
        values = " ".join(
            f"{row.property_name}={_format_r2(row.R2)}"
            for row in rows.itertuples(index=False)
        )
        print(f"  [edge-diagnostic {edge_label}] {values}", flush=True)


def _single_run(
    *,
    model_code: str,
    process_id: int,
    fold: int,
    data: dict[str, Any],
    output_dir: Path,
    baseline_root: Path,
    max_epochs: int | None,
    patience: int,
    batch_size: int,
    seed: int,
    resume: bool,
) -> dict[str, Any]:
    from process_graph.baselines.common import runtime_environment
    from process_graph.baselines.registry import create_model
    from process_graph.baselines.models.graphormer import GraphormerBaseline
    from process_graph.baselines.models.sat import SATBaseline
    from process_graph.baselines.models.graph_to_sfiles_combined import GraphToSFILESCombinedBaseline
    from process_graph.baselines.training.evaluator import write_evaluation_artifacts

    if resume and _completed(output_dir):
        print(
            f"[baseline-run][resume] model={model_code} process=P{process_id:02d} fold={fold}",
            flush=True,
        )
        _print_target_edge_r2(
            output_dir=output_dir, split="val", model_code=model_code,
            process_id=process_id, fold=fold,
        )
        _print_target_edge_r2(
            output_dir=output_dir, split="test", model_code=model_code,
            process_id=process_id, fold=fold,
        )
        summary = _load_json(output_dir / "run_summary.json")
        summary.update({
            "scope": "single_process", "model_code": model_code,
            "process_id": process_id, "fold": fold,
            "input_policy": data["rows"]["train"].input_policy,
        })
        summary["val_target_edge_property_mean_r2"] = (
            _artifact_target_edge_property_mean_r2(output_dir, "val")
        )
        summary["test_target_edge_property_mean_r2"] = (
            _artifact_target_edge_property_mean_r2(output_dir, "test")
        )
        summary.update(_property_metric_scalars(output_dir, "val"))
        summary.update(_property_metric_scalars(output_dir, "test"))
        _write_json(output_dir / "run_summary.json", summary)
        return summary
    print(
        f"[baseline-run][start] model={model_code} process=P{process_id:02d} fold={fold}",
        flush=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    transformer_baseline = model_code in {"Graphormer", "SAT", "GraphToSFILES"}
    if transformer_baseline:
        params = {
            "name": model_code.lower(), "hidden_dim": 128, "num_layers": 4,
            "num_heads": 4, "ffn_dim": 512, "dropout": 0.1,
            "attention_dropout": 0.1, "max_degree": 16,
            "max_spatial_distance": 16, "lr": 1.0e-3,
            "weight_decay": 0.0, "epochs": int(max_epochs or 30),
            "batch_size": int(batch_size), "patience": int(patience), "seed": int(seed),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        if model_code == "GraphToSFILES":
            params.update({"name": "graph_to_sfiles_combined", "num_layers": 6, "num_heads": 8, "ffn_dim": 512, "lap_pe_dim": 8})
            model = GraphToSFILESCombinedBaseline(params)
        elif model_code == "SAT":
            params["structure_hops"] = 2
            model = SATBaseline(params)
        else:
            model = GraphormerBaseline(params)
        model_name = model_code.lower()
    else:
        config = _load_baseline_config(baseline_root, SINGLE_MODELS[model_code])
        params = dict(config["model"].get("params", {}))
        if max_epochs is not None and model_code.startswith(("B", "G")):
            params["epochs"] = int(max_epochs)
        params["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        model = create_model(str(config["model"]["name"]), params)
        model_name = str(config["model"]["name"])
    arrays = data["arrays"]
    prep = data["preprocessor"]
    graph_input = bool(getattr(model, "requires_graph_input", False))
    train_x = arrays["train"].graph_samples if graph_input else data["x"]["train"]
    val_x = arrays["val"].graph_samples if graph_input else data["x"]["val"]
    test_x = arrays["test"].graph_samples if graph_input else data["x"]["test"]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    if transformer_baseline:
        model.fit_joint(
            {process_id: (train_x, data["y"]["train"])},
            {process_id: (val_x, data["y"]["val"])},
        )
    else:
        _fit_with_compact_convergence_log(
            model, train_x, data["y"]["train"], val_data=(val_x, data["y"]["val"])
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    peak_memory_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
    if transformer_baseline:
        val_pred = prep.inverse_y(model.predict_process(process_id, val_x))
        test_pred = prep.inverse_y(model.predict_process(process_id, test_x))
    else:
        val_pred = prep.inverse_y(model.predict(val_x))
        test_pred = prep.inverse_y(model.predict(test_x))
    val_metrics = write_evaluation_artifacts(out_dir=output_dir, arrays=arrays["val"], y_pred=val_pred, fold=fold, split="val")
    test_metrics = write_evaluation_artifacts(out_dir=output_dir, arrays=arrays["test"], y_pred=test_pred, fold=fold, split="test")
    _print_target_edge_r2(
        output_dir=output_dir, split="val", model_code=model_code,
        process_id=process_id, fold=fold,
    )
    _print_target_edge_r2(
        output_dir=output_dir, split="test", model_code=model_code,
        process_id=process_id, fold=fold,
    )
    model.save(output_dir / "model")
    _write_json(output_dir / "model_metadata.json", model.metadata())
    _write_json(output_dir / "preprocessing.json", prep.state_dict())
    _write_feature_schema(output_dir, data)
    summary = {
        "scope": "single_process",
        "model_code": model_code,
        "model_name": model_name,
        "process_id": process_id,
        "fold": fold,
        "input_policy": data["rows"]["train"].input_policy,
        "input_dim": int(arrays["train"].x.shape[1]),
        "output_dim": int(arrays["train"].y.shape[1]),
        "train_size": len(arrays["train"].y),
        "val_size": len(arrays["val"].y),
        "test_size": len(arrays["test"].y),
        "val_target_edge_property_mean_r2": _metric_value(val_metrics),
        "test_target_edge_property_mean_r2": _metric_value(test_metrics),
        **_property_metric_scalars(output_dir, "val"),
        **_property_metric_scalars(output_dir, "test"),
        "training_seconds": training_seconds,
        "peak_gpu_memory_mb": peak_memory_mb,
        "environment": runtime_environment(),
    }
    _write_json(output_dir / "run_summary.json", summary)
    _write_json(output_dir / "status.json", {"status": "completed"})
    return summary


def _multi_run(
    *,
    model_code: str,
    fold: int,
    process_data: dict[int, dict[str, Any]],
    output_dir: Path,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    resume: bool,
) -> dict[str, Any]:
    from process_graph.baselines.models.joint import JointProcessBaseline
    from process_graph.baselines.models.graphormer import GraphormerBaseline
    from process_graph.baselines.models.sat import SATBaseline
    from process_graph.baselines.models.graph_to_sfiles_combined import GraphToSFILESCombinedBaseline
    from process_graph.baselines.training.evaluator import write_evaluation_artifacts

    if resume and _completed(output_dir):
        print(f"[baseline-run][resume] model={model_code} joint fold={fold}", flush=True)
        for pid in sorted(process_data):
            process_dir = output_dir / f"Process{pid}"
            for split in ("val", "test"):
                _print_target_edge_r2(
                    output_dir=process_dir, split=split, model_code=model_code,
                    process_id=pid, fold=fold,
                )
        summary = _load_json(output_dir / "run_summary.json")
        summary.update({
            "scope": "multi_process_joint", "model_code": model_code,
            "fold": fold, "process_count": len(process_data),
            "input_policy": next(iter(process_data.values()))["rows"]["train"].input_policy,
        })
        summary["val_target_edge_property_mean_r2"] = (
            _artifact_target_edge_property_mean_r2(output_dir, "val")
        )
        summary["test_target_edge_property_mean_r2"] = (
            _artifact_target_edge_property_mean_r2(output_dir, "test")
        )
        summary["target_edge_property_pooling"] = (
            "all_processes_all_target_edges_by_property"
        )
        summary.update(_property_metric_scalars(output_dir, "val"))
        summary.update(_property_metric_scalars(output_dir, "test"))
        _write_json(output_dir / "run_summary.json", summary)
        return summary
    print(f"[baseline-run][start] model={model_code} joint fold={fold}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = {
        "hidden_dim": 64,
        "num_layers": 3,
        "gat_num_heads": 4,
        "lr": 1.0e-3,
        "weight_decay": 0.0,
        "epochs": int(max_epochs),
        "batch_size": int(batch_size),
        "patience": int(patience),
        "seed": int(seed),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if model_code in {"Graphormer", "SAT", "GraphToSFILES"}:
        model_config.update({
            "name": model_code.lower(),
            "hidden_dim": 128,
            "num_layers": 4,
            "num_heads": 4,
            "ffn_dim": 512,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "max_degree": 16,
            "max_spatial_distance": 16,
            "batch_size": int(batch_size),
        })
        if model_code == "Graphormer":
            model = GraphormerBaseline(model_config)
        elif model_code == "SAT":
            model_config["structure_hops"] = 2
            model = SATBaseline(model_config)
        else:
            model_config.update({
                "name": "graph_to_sfiles_combined",
                "num_layers": 6,
                "num_heads": 8,
                "ffn_dim": 512,
                "lap_pe_dim": 8,
            })
            model = GraphToSFILESCombinedBaseline(model_config)
    else:
        model = JointProcessBaseline(MULTI_MODELS[model_code], model_config)
    graph_input = model.requires_graph_input
    x_train = {
        pid: (data["arrays"]["train"].graph_samples if graph_input else data["x"]["train"])
        for pid, data in process_data.items()
    }
    x_val = {
        pid: (data["arrays"]["val"].graph_samples if graph_input else data["x"]["val"])
        for pid, data in process_data.items()
    }
    y_train = {pid: data["y"]["train"] for pid, data in process_data.items()}
    y_val = {pid: data["y"]["val"] for pid, data in process_data.items()}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    if model_code in {"Graphormer", "SAT", "GraphToSFILES"}:
        model.fit_joint(
            {pid: (x_train[pid], y_train[pid]) for pid in process_data},
            {pid: (x_val[pid], y_val[pid]) for pid in process_data},
        )
    else:
        _fit_with_compact_convergence_log(model, x_train, y_train, val_data=(x_val, y_val))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    peak_memory_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
    process_rows = []
    for pid, data in sorted(process_data.items()):
        prep = data["preprocessor"]
        arrays = data["arrays"]
        val_input = arrays["val"].graph_samples if graph_input else data["x"]["val"]
        test_input = arrays["test"].graph_samples if graph_input else data["x"]["test"]
        val_pred = prep.inverse_y(model.predict_process(pid, val_input))
        test_pred = prep.inverse_y(model.predict_process(pid, test_input))
        process_dir = output_dir / f"Process{pid}"
        val_metrics = write_evaluation_artifacts(out_dir=process_dir, arrays=arrays["val"], y_pred=val_pred, fold=fold, split="val")
        test_metrics = write_evaluation_artifacts(out_dir=process_dir, arrays=arrays["test"], y_pred=test_pred, fold=fold, split="test")
        _write_feature_schema(process_dir, data)
        _print_target_edge_r2(
            output_dir=process_dir, split="val", model_code=model_code,
            process_id=pid, fold=fold,
        )
        _print_target_edge_r2(
            output_dir=process_dir, split="test", model_code=model_code,
            process_id=pid, fold=fold,
        )
        process_rows.append({
            "process_id": pid,
            "fold": fold,
            "val_target_edge_property_mean_r2": _metric_value(val_metrics),
            "test_target_edge_property_mean_r2": _metric_value(test_metrics),
        })
    pd.DataFrame(process_rows).to_csv(output_dir / "process_metrics.csv", index=False)
    val_joint_r2 = _artifact_target_edge_property_mean_r2(output_dir, "val")
    test_joint_r2 = _artifact_target_edge_property_mean_r2(output_dir, "test")
    for split, value in (("val", val_joint_r2), ("test", test_joint_r2)):
        property_frame = pd.read_csv(output_dir / f"{split}_property_metrics.csv")
        by_name = property_frame.set_index("property_name")["R2"].to_dict()
        values = " ".join(
            f"{name}={_format_r2(by_name.get(name))}"
            for name in TARGET_EDGE_PROPERTY_NAMES
        )
        print(
            f"[baseline-r2][{split}][model={model_code} joint fold={fold}] "
            f"target_edge_property_mean_r2={_format_r2(value)}",
            flush=True,
        )
        print(f"  [pooled-by-property] {values}", flush=True)
    metadata = model.metadata()
    model.save(output_dir / "model.pt")
    _write_json(output_dir / "model_metadata.json", metadata)
    summary = {
        "scope": "multi_process_joint",
        "model_code": model_code,
        "fold": fold,
        "process_count": len(process_data),
        "input_policy": next(iter(process_data.values()))["rows"]["train"].input_policy,
        "val_target_edge_property_mean_r2": val_joint_r2,
        "test_target_edge_property_mean_r2": test_joint_r2,
        "target_edge_property_pooling": "all_processes_all_target_edges_by_property",
        **_property_metric_scalars(output_dir, "val"),
        **_property_metric_scalars(output_dir, "test"),
        "training_seconds": training_seconds,
        "peak_gpu_memory_mb": peak_memory_mb,
        "parameter_count": metadata["parameter_count"],
        "best_epoch": metadata["best_epoch"],
        "epochs_completed": metadata["epochs_completed"],
    }
    test_metrics_path = output_dir / "test_metrics.json"
    test_metrics_payload = _load_json(test_metrics_path)
    test_metrics_payload["target_edge_property_mean_r2"] = summary[
        "test_target_edge_property_mean_r2"
    ]
    _write_json(test_metrics_path, test_metrics_payload)
    _write_json(output_dir / "run_summary.json", summary)
    _write_json(output_dir / "status.json", {"status": "completed"})
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Final paper single- and multi-process baseline runner")
    parser.add_argument("--scope", choices=("single", "multi"), required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--process-ids", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--split-root", default="data/splits/all_processes_full100k_outer5_grouped_60_20_20")
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--output-root", default="outputs/final_paper/baselines")
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--input-policy",
        choices=("process_spec_refs", "core_controls", "static_only"),
        default="process_spec_refs",
        help=(
            "baseline input scope; core_controls additionally removes P_*, SP*, "
            "and dT_* settings from ANN/MLP baselines only; GNN inputs stay unchanged"
        ),
    )
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _aggregate_persisted_runs(output_root: Path, scope: str) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    if scope == "single":
        paths = sorted((output_root / "single").glob("*/Process*/fold_*/run_summary.json"))
    else:
        paths = sorted((output_root / "multi").glob("*/fold_*/run_summary.json"))
    rows = []
    for path in paths:
        row = _load_json(path)
        run_dir = path.parent
        if scope == "single":
            row.update({
                "scope": "single_process",
                "model_code": run_dir.parent.parent.name,
                "process_id": int(run_dir.parent.name.replace("Process", "")),
                "fold": int(run_dir.name.replace("fold_", "")),
            })
        else:
            row.update({
                "scope": "multi_process_joint",
                "model_code": run_dir.parent.name,
                "fold": int(run_dir.name.replace("fold_", "")),
            })
        if scope == "single":
            val_value = _artifact_target_edge_property_mean_r2(run_dir, "val")
            test_value = _artifact_target_edge_property_mean_r2(run_dir, "test")
        else:
            val_value = _artifact_target_edge_property_mean_r2(run_dir, "val")
            test_value = _artifact_target_edge_property_mean_r2(run_dir, "test")
        row["val_target_edge_property_mean_r2"] = val_value
        row["test_target_edge_property_mean_r2"] = test_value
        row.setdefault("training_seconds", float("nan"))
        row.setdefault("peak_gpu_memory_mb", float("nan"))
        row.update(_property_metric_scalars(run_dir, "val"))
        row.update(_property_metric_scalars(run_dir, "test"))
        _write_json(path, row)
        rows.append(row)
    frame = pd.DataFrame(rows)
    _write_csv(output_root / f"{scope}_run_summary.csv", frame)
    if frame.empty:
        _write_csv(output_root / f"{scope}_summary.csv", pd.DataFrame())
        return frame
    aggregate_columns: dict[str, tuple[str, str]] = {
        "test_target_edge_property_mean_r2_mean": ("test_target_edge_property_mean_r2", "mean"),
        "test_target_edge_property_mean_r2_std": ("test_target_edge_property_mean_r2", "std"),
        "training_seconds_mean": ("training_seconds", "mean"),
        "peak_gpu_memory_mb_mean": ("peak_gpu_memory_mb", "mean"),
        "runs": ("fold", "count"),
    }
    for name in TARGET_EDGE_PROPERTY_NAMES:
        column = f"test_target_edge_r2_{_property_slug(name)}"
        aggregate_columns[f"{column}_mean"] = (column, "mean")
        aggregate_columns[f"{column}_std"] = (column, "std")
    summary = frame.groupby(["model_code"]).agg(**aggregate_columns).reset_index()
    _write_csv(output_root / f"{scope}_summary.csv", summary)
    return frame


def main() -> None:
    args = _parser().parse_args()
    baseline_root = Path(args.baseline_root).resolve()
    _install_baselines(baseline_root)
    baseline_sources = _verify_baseline_implementation(baseline_root)
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    initial_provenance = {
        "baseline_root": str(baseline_root),
        "modules": baseline_sources,
        "leakage_safe_required": True,
        "requested_input_policy": args.input_policy,
    }
    _write_json(output_root / "baseline_source_provenance.json", initial_provenance)
    _write_json(output_root / f"{args.scope}_source_provenance.json", initial_provenance)
    if args.aggregate_only:
        frame = _aggregate_persisted_runs(output_root, args.scope)
        print(f"[baseline-aggregate] scope={args.scope} runs={len(frame)}", flush=True)
        return
    views_root = _prepare_views((PROJECT_ROOT / args.split_root).resolve(), output_root, args.folds)
    valid = SINGLE_MODELS if args.scope == "single" else MULTI_MODELS
    invalid = sorted(set(args.models) - set(valid))
    if invalid:
        raise ValueError(f"invalid {args.scope} model codes {invalid}; available={sorted(valid)}")
    input_policy_by_model = {
        model_code: _effective_input_policy(args.input_policy, model_code)
        for model_code in args.models
    }
    model_provenance = {
        "baseline_root": str(baseline_root),
        "modules": baseline_sources,
        "leakage_safe_required": True,
        "requested_input_policy": args.input_policy,
        "input_policy_by_model": input_policy_by_model,
    }
    _write_json(output_root / "baseline_source_provenance.json", model_provenance)
    _write_json(output_root / f"{args.scope}_source_provenance.json", model_provenance)
    expected = len(args.models) * len(args.folds) * (len(args.process_ids) if args.scope == "single" else 1)
    plan = {
        "scope": args.scope,
        "models": args.models,
        "process_ids": args.process_ids,
        "folds": args.folds,
        "expected_runs": expected,
        "requested_input_policy": args.input_policy,
        "input_policy_by_model": input_policy_by_model,
        "core_controls_model_scope": sorted(MLP_BASELINE_MODELS),
        "core_controls_exclusions": (
            ["P_* pressure settings", "SP* composition/ratio setpoints", "dT_* HX approach settings"]
            if args.input_policy == "core_controls"
            else []
        ),
        "graphsage_in_final_registry": False,
        "generic_gnn_prediction_unit": "one_target_edge",
        "generic_gnn_shared_head": "global_sum(node_states)->Linear(hidden,hidden)->ReLU->Linear(hidden,10); shared across target edges",
        "generic_gnn_process_id_routing": False,
        "primary_metric": "target_edge_property_mean_r2",
        "primary_metric_policy": "pool_all_target_edges_by_property_then_mean_finite_10d_r2",
        "target_properties": ["Temp", "Pres", "Frac_H2O", "Frac_H2", "Frac_CH4", "Frac_CO2", "Frac_CO", "Frac_O2", "Frac_N2", "Mass_Flow"],
    }
    _write_json(output_root / f"{args.scope}_plan.json", plan)
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    summaries = []
    for fold in args.folds:
        process_data_by_policy: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
        for model_code in args.models:
            effective_policy = input_policy_by_model[model_code]
            graph_mode = (
                "node_topology_raw_edges" if model_code == "GraphToSFILES"
                else "node_topology" if model_code in {"Graphormer", "SAT"}
                else "node_topology_only" if model_code in {"G1", "G2", "G3", "GCN", "GIN", "GAT"}
                else "node_only"
            )
            cache_key = (effective_policy, graph_mode)
            if cache_key not in process_data_by_policy:
                process_data_by_policy[cache_key] = {
                    pid: _load_process_fold(
                        process_id=pid,
                        fold=fold,
                        views_root=views_root,
                        max_train_samples=args.max_train_samples,
                        max_val_samples=args.max_val_samples,
                        max_test_samples=args.max_test_samples,
                        input_policy=effective_policy,
                        graph_mode=graph_mode,
                    )
                    for pid in args.process_ids
                }
            process_data = process_data_by_policy[cache_key]
            if args.scope == "single":
                for pid in args.process_ids:
                    summaries.append(_single_run(
                        model_code=model_code,
                        process_id=pid,
                        fold=fold,
                        data=process_data[pid],
                        output_dir=output_root / "single" / model_code / f"Process{pid}" / f"fold_{fold:02d}",
                        baseline_root=baseline_root,
                        max_epochs=args.max_epochs,
                        patience=args.patience,
                        batch_size=args.batch_size,
                        seed=args.seed,
                        resume=args.resume_existing,
                    ))
                    gc.collect()
            else:
                summaries.append(_multi_run(
                    model_code=model_code,
                    fold=fold,
                    process_data=process_data,
                    output_dir=output_root / "multi" / model_code / f"fold_{fold:02d}",
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    resume=args.resume_existing,
                ))
                gc.collect()
    _aggregate_persisted_runs(output_root, args.scope)


if __name__ == "__main__":
    main()
