"""Measure multi-process baseline inference latency on the held-out balanced 10-sample set.

The script intentionally does not train any model.  It reuses every preserved
smoke checkpoint that is available in this checkout and reconstructs the two
missing generic-GNN architectures (GIN/GAT) solely to measure their forward
latency.  The output records checkpoint provenance per model so that the
architecture-only measurements cannot be mistaken for final trained weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_NAMES = (
    "Temp", "Pres", "Frac_H2O", "Frac_H2", "Frac_CH4", "Frac_CO2",
    "Frac_CO", "Frac_O2", "Frac_N2", "Mass_Flow",
)
MODEL_ORDER = ("B6", "GCN", "GIN", "GAT", "Graphormer", "SAT", "GraphToSFILES")


def _process_number(value: Any) -> int:
    text = "".join(char for char in str(value) if char.isdigit())
    return int(text) if text else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--baseline-root",
        default=str(PROJECT_ROOT.parent / "화학공정Baselines"),
        help="Checkout containing process_graph.baselines source code.",
    )
    parser.add_argument(
        "--manifest",
        default=str(
            PROJECT_ROOT
            / "outputs/inference_benchmark_manifests_20260916"
            / "fold01_test_one_sample_per_process.csv"
        ),
    )
    parser.add_argument(
        "--split-root",
        default=str(
            PROJECT_ROOT / "data/splits/all_processes_full100k_outer5_grouped_60_20_20"
        ),
    )
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--measured-runs", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _materialize_benchmark_views(
    *, split_root: Path, manifest: pd.DataFrame, output_dir: Path
) -> Path:
    """Create a reproducible one-held-out-sample-per-process view for fold 1."""

    view_root = output_dir / "process_split_views"
    fold_dir = split_root / "fold_01"
    selected = {
        _process_number(row.process_id): str(row.sample_id).strip()
        for row in manifest.itertuples(index=False)
    }
    for split in ("train", "val", "test"):
        source = fold_dir / f"{split}.csv"
        if not source.is_file():
            raise FileNotFoundError(f"missing canonical split manifest: {source}")
        frame = pd.read_csv(source)
        if "process_id" not in frame.columns or "sample_id" not in frame.columns:
            raise KeyError(f"{source} must contain process_id and sample_id")
        process_numbers = frame["process_id"].map(_process_number)
        for process_id, sample_id in sorted(selected.items()):
            subset = frame.loc[process_numbers.eq(process_id)].copy()
            if split == "test":
                sample_key = subset["sample_id"].astype(str).str.replace(r"\.0$", "", regex=True)
                subset = subset.loc[sample_key.eq(sample_id)].copy()
                if len(subset) != 1:
                    raise ValueError(
                        f"expected exactly one test row for Process{process_id} sample {sample_id}; "
                        f"found {len(subset)}"
                    )
            target = view_root / f"Process{process_id}" / "fold_01" / f"{split}.csv"
            target.parent.mkdir(parents=True, exist_ok=True)
            subset.to_csv(target, index=False)
    manifest.to_csv(output_dir / "benchmark_input_manifest.csv", index=False, encoding="utf-8-sig")
    return view_root


def _build_inputs(
    *, views_root: Path
) -> tuple[dict[int, np.ndarray], dict[int, Any], dict[int, Any], dict[int, int]]:
    """Build exactly the baseline input representations for the selected ten samples."""

    from process_graph.baselines.dataset import build_target_edge_rows, materialize_arrays
    from process_graph.baselines.preprocessing import BaselinePreprocessor

    common = {
        "project_root": PROJECT_ROOT,
        "experiment_config_path": PROJECT_ROOT / "configs/experiment/process_surrogate_edge_all_v3.yaml",
        "splits_dir": views_root,
        "fold": 1,
    }
    mlp_inputs: dict[int, np.ndarray] = {}
    graph_inputs: dict[int, Any] = {}
    combined_inputs: dict[int, Any] = {}
    output_widths: dict[int, int] = {}

    for process_id in range(1, 11):
        print(f"[prepare] Process{process_id}/10", flush=True)
        def make_rows(split: str, policy: str, graph_mode: str, max_samples: int | None):
            return build_target_edge_rows(
                split=split,
                process_ids=[process_id],
                input_policy=policy,
                graph_mode=graph_mode,
                max_samples=max_samples,
                **common,
            )

        # B6 checkpoint was created under the core-controls sensitivity policy.
        # The fitted objects here only create a correctly shaped input tensor;
        # they are outside the timed section and are not used for accuracy evaluation.
        mlp_test_rows = make_rows("test", "core_controls", "node_only", None)
        mlp_test, _ = materialize_arrays(mlp_test_rows, fit=True)
        mlp_preprocessor = BaselinePreprocessor().fit(mlp_test.x, mlp_test.y)
        mlp_x = mlp_preprocessor.transform_x(mlp_test.x)
        if len(mlp_x) != 1:
            raise AssertionError(f"Process{process_id}: expected one MLP test input, got {len(mlp_x)}")
        mlp_inputs[process_id] = np.asarray(mlp_x, dtype=np.float32)

        # Transformer decoders consume the target-edge endpoints.  Generic
        # GNNs safely ignore that extra metadata.  Graph-to-SFILES instead uses
        # the raw-edge view, so it receives a separately materialized graph.
        endpoint_rows = make_rows("test", "process_spec_refs", "node_topology", None)
        raw_edge_rows = make_rows("test", "process_spec_refs", "node_topology_raw_edges", None)
        if endpoint_rows.graph_samples is None or len(endpoint_rows.graph_samples) != 1:
            raise AssertionError(f"Process{process_id}: expected one endpoint graph test input")
        if raw_edge_rows.graph_samples is None or len(raw_edge_rows.graph_samples) != 1:
            raise AssertionError(f"Process{process_id}: expected one raw-edge graph test input")
        graph_inputs[process_id] = endpoint_rows.graph_samples[0]
        combined_inputs[process_id] = raw_edge_rows.graph_samples[0]
        output_widths[process_id] = int(len(endpoint_rows.target_names))

    return mlp_inputs, graph_inputs, combined_inputs, output_widths


def _restore_node_scaler(instance: Any, state: dict[str, Any]) -> None:
    instance.scaler.mean = None if state.get("mean") is None else np.asarray(state["mean"], dtype=np.float64)
    instance.scaler.scale = None if state.get("scale") is None else np.asarray(state["scale"], dtype=np.float64)
    instance.scaler.flat_value_mode = bool(state.get("flat_value_mode", False))


def _configure_output_shapes(instance: Any, widths: dict[int, int]) -> None:
    instance.output_dims = {str(pid): int(width) for pid, width in widths.items()}
    instance.target_edge_counts = {str(pid): int(width // len(TARGET_NAMES)) for pid, width in widths.items()}
    if any(width % len(TARGET_NAMES) for width in widths.values()):
        raise ValueError(f"target widths are not 10-property blocks: {widths}")


def _load_b6(device: torch.device) -> tuple[Any, dict[str, Any]]:
    from process_graph.baselines.models.joint import JointProcessBaseline, _JointMLP

    path = PROJECT_ROOT / "outputs/_core_controls_smoke_0907/multi/B6/fold_01/model.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    config = {"hidden_dim": metadata["hidden_dim"], "num_layers": metadata["num_layers"], "device": str(device)}
    model = JointProcessBaseline("mlp", config)
    model.input_dims = {str(key): int(value) for key, value in metadata["input_dims"].items()}
    model.output_dims = {str(key): int(value) for key, value in metadata["output_dims"].items()}
    model.model = _JointMLP(model.input_dims, model.output_dims, model.hidden_dim, model.num_layers).to(device)
    model.model.load_state_dict(payload["state_dict"])
    model.model.eval()
    model.parameter_count = int(metadata["parameter_count"])
    return model, {"path": str(path), "provenance": "saved_smoke_checkpoint_all_10_processes"}


def _load_joint_gnn(
    model_type: str, device: torch.device, widths: dict[int, int]
) -> tuple[Any, dict[str, Any]]:
    from process_graph.baselines.models.joint import JointProcessBaseline, _JointGNN

    checkpoint = PROJECT_ROOT / "outputs/_core_controls_model_scope_smoke_0907/multi/GCN/fold_01/model.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    base_metadata = payload["metadata"]
    config = {
        "hidden_dim": int(base_metadata["hidden_dim"]),
        "num_layers": int(base_metadata["num_layers"]),
        "gat_num_heads": 4,
        "device": str(device),
    }
    model = JointProcessBaseline(model_type, config)
    model.input_dims = {str(pid): int(next(iter(base_metadata["input_dims"].values()))) for pid in widths}
    _configure_output_shapes(model, widths)
    _restore_node_scaler(model, payload["scaler"])
    node_dim = int(next(iter(model.input_dims.values())))
    model.model = _JointGNN(
        node_input_dim=node_dim,
        hidden_dim=model.hidden_dim,
        num_layers=model.num_layers,
        gnn_type=model_type,
        gat_num_heads=model.gat_num_heads,
    ).to(device)
    if model_type == "gcn":
        # The preserved checkpoint predates the ``stream_head.`` container
        # introduced in the current baseline source.  Parameters are otherwise
        # identical, so remap only that serialization prefix before strict load.
        expected = set(model.model.state_dict())
        restored = {
            (
                f"stream_head.{key}"
                if key not in expected and f"stream_head.{key}" in expected
                else key
            ): value
            for key, value in payload["state_dict"].items()
        }
        model.model.load_state_dict(restored, strict=True)
        provenance = "saved_smoke_checkpoint_process_1; executed on the 10-process held-out inputs"
    else:
        torch.manual_seed(42)
        # The computational graph, tensor shapes, and input distribution determine latency;
        # no GIN/GAT checkpoint survives in this workspace.
        for module in model.model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        provenance = "architecture_only_deterministic_weights; no saved multi-process checkpoint"
    model.model.eval()
    model.parameter_count = int(sum(parameter.numel() for parameter in model.model.parameters()))
    return model, {"path": str(checkpoint) if model_type == "gcn" else "", "provenance": provenance}


def _load_pickled_transformer(
    name: str, device: torch.device, widths: dict[int, int]
) -> tuple[Any, dict[str, Any]]:
    relative = {
        "Graphormer": "outputs/graphormer_smoke_all10/multi/Graphormer/fold_01/model.pt/model.pkl",
        "SAT": "outputs/sat_smoke/multi/SAT/fold_01/model.pt/model.pkl",
    }[name]
    path = PROJECT_ROOT / relative
    with path.open("rb") as handle:
        model = pickle.load(handle)
    model.device = device
    model.model = model.model.to(device)
    model.model.eval()
    _configure_output_shapes(model, widths)
    return model, {
        "path": str(path),
        "provenance": (
            "saved_smoke_checkpoint_all_10_processes"
            if name == "Graphormer"
            else "saved_smoke_checkpoint_process_1; executed on the 10-process held-out inputs"
        ),
    }


def _load_combined(device: torch.device, widths: dict[int, int]) -> tuple[Any, dict[str, Any]]:
    from process_graph.baselines.models.graph_to_sfiles_combined import (
        GraphToSFILESCombinedBaseline,
        _CombinedNet,
    )

    path = PROJECT_ROOT / "outputs/_combined_smoke/multi/GraphToSFILES/fold_01/model.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    state = payload["state_dict"]
    node_dim = int(state["node_projection.weight"].shape[1])
    edge_dim = int(state["edge_projection.weight"].shape[1])
    config = {
        "name": "graph_to_sfiles_combined",
        "hidden_dim": int(metadata["hidden_dim"]),
        "num_layers": int(metadata["num_layers"]),
        "num_heads": int(metadata["num_heads"]),
        "ffn_dim": int(metadata["ffn_dim"]),
        "lap_pe_dim": int(metadata["lap_pe_dim"]),
        "batch_size": 1,
        "device": str(device),
    }
    model = GraphToSFILESCombinedBaseline(config)
    model.node_input_dim = node_dim
    model.edge_input_dim = edge_dim
    _configure_output_shapes(model, widths)
    model.model = _CombinedNet(
        node_dim,
        edge_dim,
        model.hidden_dim,
        model.lap_pe_dim,
        model.num_layers,
        model.num_heads,
        model.ffn_dim,
        model.dropout,
        model.attention_dropout,
    ).to(device)
    model.model.load_state_dict(state)
    model.model.eval()
    _restore_node_scaler(model, payload["node_scaler"])
    model.parameter_count = int(metadata["parameter_count"])
    model.trainable_parameter_count = int(metadata["trainable_parameter_count"])
    return model, {
        "path": str(path),
        "provenance": "saved_smoke_checkpoint_process_1; executed on the 10-process held-out inputs",
    }


@torch.inference_mode()
def _benchmark(
    *, name: str, run_one: Callable[[int], torch.Tensor], device: torch.device,
    warmup_runs: int, measured_runs: int, parameter_count: int, provenance: dict[str, Any]
) -> dict[str, Any]:
    process_ids = list(range(1, 11))
    for _ in range(warmup_runs):
        for process_id in process_ids:
            _ = run_one(process_id)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    durations: list[float] = []
    prediction_shape: list[int] = []
    for _ in range(measured_runs):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for process_id in process_ids:
            output = run_one(process_id)
            prediction_shape = list(output.shape)
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
    values = np.asarray(durations, dtype=np.float64)
    total_mean = float(values.mean())
    total_std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return {
        "model": name,
        "timing_scope": "selected_cpu_sample_to_host_to_device_plus_model_specific_batch_assembly_plus_forward",
        "warmup_runs": int(warmup_runs),
        "measured_runs": int(measured_runs),
        "samples_per_run": len(process_ids),
        "inference_batch_size": 1,
        "inference_total_sec_mean": total_mean,
        "inference_total_sec_std": total_std,
        "inference_time_per_sample_ms": float(1000.0 * total_mean / len(process_ids)),
        "inference_time_std_ms": float(1000.0 * total_std / len(process_ids)),
        "inference_time_median_ms": float(1000.0 * np.median(values) / len(process_ids)),
        "throughput_samples_per_sec": float(len(process_ids) / total_mean),
        "inference_peak_gpu_allocated_mb": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
        "inference_peak_gpu_reserved_mb": float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)),
        "device": str(device),
        "total_parameters": int(parameter_count),
        "prediction_shape_last_sample": prediction_shape,
        "checkpoint_path": provenance["path"],
        "checkpoint_provenance": provenance["provenance"],
    }


def _proposed_reference() -> dict[str, Any] | None:
    path = (
        PROJECT_ROOT
        / "outputs/inference_benchmark_proposed_fold01_balanced10_20260916"
        / "proposed_inference_benchmark_fold01_balanced10-20260916-202150"
        / "inference_benchmark.json"
    )
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    record = dict(record)
    record["model"] = "Proposed"
    record["checkpoint_provenance"] = "final_proposed_checkpoint_strict_load"
    return record


def _write_readme(output_dir: Path, records: list[dict[str, Any]]) -> None:
    rows = "\n".join(
        f"| {r['model']} | {r['inference_time_per_sample_ms']:.3f} | {r['inference_time_std_ms']:.3f} | {r['throughput_samples_per_sec']:.2f} | {r['checkpoint_provenance']} |"
        for r in records
    )
    text = f"""# Multi-process baseline inference benchmark

Hardware: CUDA GPU measured at runtime.  Protocol: one held-out Fold-1 test
sample from each of P01--P10 (10 samples/run), batch size 1, {records[0]['warmup_runs']}
warm-up runs, and {records[0]['measured_runs']} measured runs.  Each recorded
run includes the 10 sequential single-sample forwards.  The input manifest is
`benchmark_input_manifest.csv`.

| Model | ms/sample | SD (ms/sample) | samples/s | provenance |
|---|---:|---:|---:|---|
{rows}

## Reporting constraint

No final five-fold multi-process baseline checkpoint is preserved in this
checkout.  Therefore these measurements are valid runtime measurements of the
implemented architectures and the explicitly named saved smoke checkpoints,
but must **not** be labelled as final trained multi-process model latency in a
paper.  GIN and GAT are architecture-only timings because no checkpoint was
found; model weights do not change tensor shapes or the forward computational
graph.  Re-run or restore the final multi-process checkpoints before using a
final paper efficiency table.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a comparable GPU inference benchmark")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be a CUDA device")
    torch.cuda.set_device(device)

    baseline_root = Path(args.baseline_root).resolve()
    baseline_source = baseline_root / "src"
    if not baseline_source.is_dir():
        raise FileNotFoundError(f"baseline source directory not found: {baseline_source}")
    sys.path.insert(0, str(baseline_source))

    manifest_path = Path(args.manifest).resolve()
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 10 or set(manifest.columns) < {"process_id", "sample_id"}:
        raise ValueError("manifest must contain exactly 10 rows with process_id and sample_id")
    if sorted(manifest["process_id"].map(_process_number).tolist()) != list(range(1, 11)):
        raise ValueError("manifest must contain exactly one row for every Process1 through Process10")

    output_dir.mkdir(parents=True)
    shutil.copy2(manifest_path, output_dir / "source_balanced_manifest.csv")
    views_root = _materialize_benchmark_views(
        split_root=Path(args.split_root).resolve(), manifest=manifest, output_dir=output_dir
    )
    mlp_inputs, graph_inputs, combined_inputs, widths = _build_inputs(views_root=views_root)

    b6, b6_prov = _load_b6(device)
    for process_id, x in mlp_inputs.items():
        expected = int(b6.input_dims[str(process_id)])
        if x.shape[1] != expected:
            raise ValueError(f"B6 Process{process_id}: input width {x.shape[1]} != checkpoint width {expected}")

    gcn, gcn_prov = _load_joint_gnn("gcn", device, widths)
    gin, gin_prov = _load_joint_gnn("gin", device, widths)
    gat, gat_prov = _load_joint_gnn("gat", device, widths)
    graphormer, graphormer_prov = _load_pickled_transformer("Graphormer", device, widths)
    sat, sat_prov = _load_pickled_transformer("SAT", device, widths)
    combined, combined_prov = _load_combined(device, widths)

    cases: list[tuple[str, Callable[[int], torch.Tensor], int, dict[str, Any]]] = [
        ("B6", lambda pid: b6._predict_tensor(pid, mlp_inputs[pid]), b6.parameter_count, b6_prov),
        ("GCN", lambda pid: gcn._predict_tensor(pid, [graph_inputs[pid]]), gcn.parameter_count, gcn_prov),
        ("GIN", lambda pid: gin._predict_tensor(pid, [graph_inputs[pid]]), gin.parameter_count, gin_prov),
        ("GAT", lambda pid: gat._predict_tensor(pid, [graph_inputs[pid]]), gat.parameter_count, gat_prov),
        ("Graphormer", lambda pid: graphormer._predict_tensor(pid, [graph_inputs[pid]]), graphormer.parameter_count, graphormer_prov),
        ("SAT", lambda pid: sat._predict_tensor(pid, [graph_inputs[pid]]), sat.parameter_count, sat_prov),
        ("GraphToSFILES", lambda pid: combined._predict_tensor(pid, [combined_inputs[pid]]), combined.parameter_count, combined_prov),
    ]
    records: list[dict[str, Any]] = []
    for name, run_one, parameter_count, provenance in cases:
        record = _benchmark(
            name=name,
            run_one=run_one,
            device=device,
            warmup_runs=args.warmup_runs,
            measured_runs=args.measured_runs,
            parameter_count=parameter_count,
            provenance=provenance,
        )
        records.append(record)
        print(
            f"[benchmark] {name}: {record['inference_time_per_sample_ms']:.3f} ms/sample "
            f"({record['throughput_samples_per_sec']:.2f} samples/s)",
            flush=True,
        )
    proposed = _proposed_reference()
    if proposed is not None:
        records.append(proposed)
    _write_json(
        output_dir / "multi_baseline_inference_benchmark.json",
        {
            "protocol": {
                "hardware": torch.cuda.get_device_name(device),
                "device": str(device),
                "manifest": str(manifest_path),
                "sample_selection": "one held-out Fold-1 test sample per process, P01 through P10",
                "inference_batch_size": 1,
                "warmup_runs": args.warmup_runs,
                "measured_runs": args.measured_runs,
                "timing_scope": "selected_cpu_sample_to_host_to_device_plus_model_specific_batch_assembly_plus_forward",
            },
            "records": records,
        },
    )
    _write_csv(output_dir / "multi_baseline_inference_benchmark.csv", records)
    _write_readme(output_dir, records)
    print(f"[benchmark] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
