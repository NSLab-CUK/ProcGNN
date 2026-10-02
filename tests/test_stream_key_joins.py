from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from process_graph.data.stream_keys import (
    CanonicalStreamKeyCollisionError,
    MissingRequiredStreamKeyError,
    build_canonical_stream_row_index,
    canonicalize_process_id,
    canonicalize_stream_key,
)
from process_graph.data.tabular_dataset import ProcessGraphTabularDataset
from process_graph.data.tabular_dataset import collate_graph_batch
from process_graph.experiment.config_builders import (
    build_task_specs,
    model_yaml_to_encoder_config,
)
from process_graph.experiment.loaders import load_experiment_config
from process_graph.models.process_surrogate import ProcessSurrogateModel


P03_RECOVERY_EDGES = {
    "P03_E004",
    "P03_E007",
    "P03_E008",
    "P03_E009",
    "P03_E012",
    "P03_E016",
    "P03_E018",
    "P03_E019",
    "P03_E020",
}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, ""),
        (float("nan"), ""),
        ("", ""),
        ("  ", ""),
        ("01", "1"),
        ("001", "1"),
        ("1", "1"),
        (1, "1"),
        (1.0, "1"),
        (" 01 ", "1"),
        ("\u00a001\u2009", "1"),
        ("PROD", "PROD"),
        ("P03_E014", "P03_E014"),
        ("R1-FEED", "R1-FEED"),
        ("A\u00a0 B", "A B"),
    ],
)
def test_canonicalize_stream_key(raw, expected) -> None:
    assert canonicalize_stream_key(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [3, "3", "03", "P3", "P03", "Process3", "process03"],
)
def test_canonicalize_process_id(raw) -> None:
    assert canonicalize_process_id(raw) == "P03"


def test_canonical_stream_collision_is_rejected() -> None:
    rows = pd.DataFrame(
        {
            "ID": [1, 1],
            "Stream_Name": ["01", "1"],
            "Mass_Flow": [10.0, 20.0],
        }
    )
    with pytest.raises(
        CanonicalStreamKeyCollisionError,
        match="Canonical stream-key collision",
    ):
        build_canonical_stream_row_index(
            rows,
            process_id="P03",
            sample_id=1,
            csv_path="3.Process_Streams.csv",
        )


def test_all_processes_required_edges_join_and_p03_values_match_source() -> None:
    experiment = load_experiment_config(
        PROJECT_ROOT
        / "configs/experiment/pinn/known_feed_260730/f0_e2_baseline.yaml"
    )
    cfg = replace(
        experiment.data,
        normalize_x_oper=False,
        normalize_targets=False,
        edge_all_processes=list(range(1, 11)),
    )
    dataset = ProcessGraphTabularDataset(
        experiment.data.train_data_path,
        cfg,
        experiment.project_root,
    )
    process_col = cfg.process_id_column

    for process_id in range(1, 11):
        process_label = f"Process{process_id}"
        index = int(
            dataset.frame.index[
                dataset.frame[process_col].astype(str).eq(process_label)
            ][0]
        )
        record = dataset[index]
        graph = record.graph
        edges = dataset._v3_edges_for_pid(process_id)
        keyed_edge_ids = {
            str(row["canonical_edge_id"])
            for _, row in edges.iterrows()
            if canonicalize_stream_key(row.get("main_data_stream_key"))
        }
        for edge_id in keyed_edge_ids:
            edge_index = graph.canonical_edge_ids.index(edge_id)
            assert graph.edge_feature_mask[edge_index] == 1.0
            assert graph.y_edge_mask[edge_index] == 1.0
            assert graph.edge_is_supervised[edge_index] == 1.0

        if process_id == 3:
            row_id = dataset.frame.iloc[index]["ID"]
            stream_rows = dataset._stream_rows_for_process_id(
                process_label,
                row_id,
            )
            source = build_canonical_stream_row_index(
                stream_rows,
                process_id=process_label,
                sample_id=row_id,
                csv_path=dataset._stream_csv_path("3"),
            )
            edge_columns = list(graph.edge_target_columns)
            for edge_id in P03_RECOVERY_EDGES:
                edge_index = graph.canonical_edge_ids.index(edge_id)
                edge_row = edges[
                    edges["canonical_edge_id"].astype(str).eq(edge_id)
                ].iloc[0]
                stream_key = canonicalize_stream_key(
                    edge_row["main_data_stream_key"]
                )
                stream_row = source[stream_key]
                expected = [
                    float(stream_row[column])
                    if column in stream_row.index
                    and not pd.isna(stream_row[column])
                    else 0.0
                    for column in edge_columns
                ]
                assert graph.y_edge_true[edge_index] == pytest.approx(
                    expected,
                    rel=1.0e-6,
                )

        dataset._stream_process_frame_cache.clear()
        dataset._stream_rows_cache.clear()


def test_train_val_test_collate_forward_loss_backward_and_metric_smoke() -> None:
    experiment = load_experiment_config(
        PROJECT_ROOT
        / "configs/experiment/pinn/known_feed_260730/f0_e2_baseline.yaml"
    )
    cfg = replace(
        experiment.data,
        normalize_x_oper=False,
        normalize_targets=False,
    )
    model = ProcessSurrogateModel(
        model_yaml_to_encoder_config(experiment.model, cfg),
        task_specs=build_task_specs(experiment.model, cfg),
    )
    output_columns = [
        "Temp",
        "Pres",
        *[f"Frac_{species}" for species in experiment.model.species_order],
        "Mass_Flow",
        "Vol_Flow",
    ]

    metric_sum = 0.0
    metric_count = 0
    for split_name in ("train", "val", "test"):
        manifest = PROJECT_ROOT / str(
            getattr(cfg, f"{split_name}_split_manifest_path")
        )
        dataset = ProcessGraphTabularDataset(
            getattr(cfg, f"{split_name}_data_path"),
            cfg,
            experiment.project_root,
            split_manifest=manifest,
        )
        batch = collate_graph_batch([dataset[0]])
        predictions = model(
            batch.model_kwargs,
            task_inputs=batch.task_inputs,
        )["main_stream_pred"]
        source_columns = list(batch.edge_target_columns)
        target_indices = [source_columns.index(column) for column in output_columns]
        targets = batch.model_kwargs["y_edge_true"][:, target_indices]
        valid = batch.model_kwargs["y_edge_mask"].bool()
        assert bool(valid.any())
        loss = F.smooth_l1_loss(predictions[valid], targets[valid])
        assert torch.isfinite(loss)
        if split_name == "train":
            model.zero_grad(set_to_none=True)
            loss.backward()
            assert any(
                parameter.grad is not None
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        with torch.no_grad():
            metric_sum += float(
                torch.abs(predictions[valid] - targets[valid]).sum().item()
            )
            metric_count += int(predictions[valid].numel())
        assert metric_count > 0

    assert metric_sum / metric_count >= 0.0


def test_missing_required_source_row_fails_with_actionable_context() -> None:
    experiment = load_experiment_config(
        PROJECT_ROOT
        / "configs/experiment/pinn/known_feed_260730/f0_e2_baseline.yaml"
    )
    cfg = replace(
        experiment.data,
        normalize_x_oper=False,
        normalize_targets=False,
        edge_all_processes=[3],
    )
    dataset = ProcessGraphTabularDataset(
        experiment.data.train_data_path,
        cfg,
        experiment.project_root,
    )
    ids = pd.to_numeric(dataset.frame["ID"], errors="coerce")
    index = int(
        dataset.frame.index[
            dataset.frame[cfg.process_id_column].astype(str).eq("Process3")
            & ids.eq(7593)
        ][0]
    )
    with pytest.raises(MissingRequiredStreamKeyError) as caught:
        dataset[index]
    message = str(caught.value)
    for expected in (
        "process_id='P03'",
        "sample_id",
        "canonical_edge_id='P03_E015'",
        "raw_graph_key='11'",
        "canonical_graph_key='11'",
        "available_csv_keys",
        "3.Process_Streams.csv",
    ):
        assert expected in message


def test_explicit_excluded_graph_sample_is_removed_before_stream_join(tmp_path: Path) -> None:
    experiment = load_experiment_config(
        PROJECT_ROOT
        / "configs/experiment/pinn/known_feed_260730/f0_e2_baseline.yaml"
    )
    excluded_path = tmp_path / "excluded_graph_samples.csv"
    excluded_path.write_text(
        "process_id,sample_id,reason\n3,7593,missing source rows\n",
        encoding="utf-8",
    )
    cfg = replace(
        experiment.data,
        normalize_x_oper=False,
        normalize_targets=False,
        edge_all_processes=[3],
        excluded_graph_samples_path=str(excluded_path),
    )
    dataset = ProcessGraphTabularDataset(
        experiment.data.train_data_path,
        cfg,
        experiment.project_root,
    )
    ids = pd.to_numeric(dataset.frame["ID"], errors="coerce")
    process3 = dataset.frame[cfg.process_id_column].astype(str).eq("Process3")
    assert not bool((process3 & ids.eq(7593)).any())
    assert len(dataset) == 9999
