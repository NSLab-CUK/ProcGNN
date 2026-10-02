from __future__ import annotations

import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from process_graph.constants import OPER_FEATURE_SLOTS
from process_graph.data.tabular_dataset import (
    ProcessGraphTabularDataset,
    collate_graph_batch,
    compute_oper_normalizer_from_x_oper,
)
from process_graph.experiment.config_builders import model_yaml_to_encoder_config
from process_graph.experiment.loaders import load_experiment_config
from process_graph.known_feed import (
    KNOWN_FEED_FEATURE_NAMES,
    KNOWN_FEED_RATIO_FEATURE_NAMES,
    known_feed_bindings,
)
from process_graph.models.process_surrogate import ProcessSurrogateModel


CONFIG_DIR = PROJECT_ROOT / "configs" / "experiment" / "pinn" / "known_feed_260730"
BASE_E2 = (
    PROJECT_ROOT
    / "configs"
    / "experiment"
    / "pinn"
    / "head_dimension_260729"
    / "e2_balanced384_focus_sampler.yaml"
)


@pytest.fixture(scope="module")
def f1_experiment():
    return load_experiment_config(CONFIG_DIR / "f1_e2_vinput_feed.yaml")


@pytest.fixture(scope="module")
def p03_f1_dataset(f1_experiment):
    cfg = replace(
        f1_experiment.data,
        normalize_x_oper=False,
        normalize_targets=False,
        edge_all_processes=[3],
    )
    return ProcessGraphTabularDataset(
        f1_experiment.data.train_data_path,
        cfg,
        f1_experiment.project_root,
    )


@pytest.fixture(scope="module")
def p03_f1_ratio_dataset(f1_experiment):
    feed_cfg = dict(f1_experiment.data.known_feed_condition)
    feed_cfg["ratio_features"] = {
        "enabled": True,
        "transform": "log_ratio",
        "epsilon": 1.0e-6,
        "attach_to_v_input": True,
        "attach_to_burner": True,
    }
    cfg = replace(
        f1_experiment.data,
        known_feed_condition=feed_cfg,
        normalize_x_oper=False,
        normalize_targets=False,
        edge_all_processes=[3],
    )
    return ProcessGraphTabularDataset(
        f1_experiment.data.train_data_path,
        cfg,
        f1_experiment.project_root,
    )


def test_f0_is_exact_e2_baseline_and_f1_changes_only_operating_width(f1_experiment) -> None:
    base = load_experiment_config(BASE_E2)
    f0 = load_experiment_config(CONFIG_DIR / "f0_e2_baseline.yaml")
    f1 = f1_experiment

    base_model = asdict(base.model)
    f0_model = asdict(f0.model)
    base_model.pop("known_feed_condition")
    f0_model.pop("known_feed_condition")
    assert f0_model == base_model
    assert asdict(f0.train) == asdict(base.train)
    base_data = asdict(base.data)
    f0_data = asdict(f0.data)
    base_data.pop("known_feed_condition")
    f0_data.pop("known_feed_condition")
    assert f0_data == base_data

    f0_encoder = model_yaml_to_encoder_config(f0.model, f0.data)
    f1_encoder = model_yaml_to_encoder_config(f1.model, f1.data)
    assert f0_encoder.oper_dim == len(OPER_FEATURE_SLOTS) == 13
    assert f1_encoder.oper_dim == 16
    assert f0_encoder.operating_hidden_dim == f1_encoder.operating_hidden_dim == 64
    assert f0_encoder.hidden_dim == f1_encoder.hidden_dim == 384

    f0_model_obj = ProcessSurrogateModel(f0_encoder, task_specs=[])
    f1_model_obj = ProcessSurrogateModel(f1_encoder, task_specs=[])
    assert f0_model_obj.encoder.input_encoder.operating_encoder[0].in_features == 26
    assert f1_model_obj.encoder.input_encoder.operating_encoder[0].in_features == 32
    assert f1_model_obj.encoder.input_encoder.operating_encoder[0].out_features == 64
    assert f1_model_obj.encoder.input_encoder.input_mlp.net[0].in_features == 120
    f0_reload = ProcessSurrogateModel(f0_encoder, task_specs=[])
    f0_reload.load_state_dict(f0_model_obj.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="size mismatch"):
        f1_model_obj.load_state_dict(f0_model_obj.state_dict(), strict=True)


def test_raw_feed_values_are_only_on_canonical_v_input(p03_f1_dataset) -> None:
    record = p03_f1_dataset[0]
    graph = record.graph
    row = p03_f1_dataset.frame.iloc[0]
    v_input_index = graph.node_names.index("V_INPUT")
    feed_start = len(OPER_FEATURE_SLOTS)

    assert len(graph.x_oper[v_input_index]) == 16
    assert graph.x_oper[v_input_index][feed_start:] == pytest.approx(
        [float(row["CH4_Flow"]), float(row["AIR_Flow"]), float(row["WATER_Flow"])],
        rel=1.0e-6,
    )
    assert graph.x_oper_mask[v_input_index][feed_start:] == [1, 1, 1]
    for node_index in range(len(graph.node_names)):
        if node_index == v_input_index:
            continue
        assert graph.x_oper[node_index][feed_start:] == [0.0, 0.0, 0.0]
        assert graph.x_oper_mask[node_index][feed_start:] == [0, 0, 0]

    bindings = known_feed_bindings(p03_f1_dataset._load_spec("Process3"))
    assert [binding.source_reference for binding in bindings] == [
        "CH4_Flow",
        "AIR_Flow",
        "WATER_Flow",
    ]
    assert all(
        binding.source_node_name
        and p03_f1_dataset._load_spec("Process3").nodes[binding.source_node_name].role
        == "source"
        for binding in bindings
    )


def test_log_feed_ratios_are_attached_only_to_v_input_and_burner(
    p03_f1_ratio_dataset,
) -> None:
    record = p03_f1_ratio_dataset[0]
    graph = record.graph
    row = p03_f1_ratio_dataset.frame.iloc[0]
    v_input_index = graph.node_names.index("V_INPUT")
    burner_index = graph.node_names.index("BURNER")
    tail_start = len(OPER_FEATURE_SLOTS)
    ratio_start = tail_start + len(KNOWN_FEED_FEATURE_NAMES)
    expected_ratios = torch.log(
        torch.tensor(
            [float(row["AIR_Flow"]), float(row["WATER_Flow"])],
            dtype=torch.float64,
        ).add(1.0e-6)
        / (float(row["CH4_Flow"]) + 1.0e-6)
    ).tolist()

    assert len(graph.x_oper[v_input_index]) == (
        len(OPER_FEATURE_SLOTS)
        + len(KNOWN_FEED_FEATURE_NAMES)
        + len(KNOWN_FEED_RATIO_FEATURE_NAMES)
    )
    assert graph.x_oper[v_input_index][ratio_start:] == pytest.approx(expected_ratios)
    assert graph.x_oper_mask[v_input_index][ratio_start:] == [1, 1]
    assert graph.x_oper[burner_index][tail_start:ratio_start] == [0.0, 0.0, 0.0]
    assert graph.x_oper_mask[burner_index][tail_start:ratio_start] == [0, 0, 0]
    assert graph.x_oper[burner_index][ratio_start:] == pytest.approx(expected_ratios)
    assert graph.x_oper_mask[burner_index][ratio_start:] == [1, 1]
    for node_index, node_name in enumerate(graph.node_names):
        if node_name in {"V_INPUT", "BURNER"}:
            continue
        assert graph.x_oper[node_index][ratio_start:] == [0.0, 0.0]
        assert graph.x_oper_mask[node_index][ratio_start:] == [0, 0]


def test_log_feed_ratios_use_train_featurewise_normalization(
    p03_f1_ratio_dataset,
) -> None:
    raw = list(p03_f1_ratio_dataset.iter_x_oper_for_normalizer(limit=4))
    mean, std = compute_oper_normalizer_from_x_oper(raw)
    ratio_start = len(OPER_FEATURE_SLOTS) + len(KNOWN_FEED_FEATURE_NAMES)
    expected = []
    for index in range(4):
        row = p03_f1_ratio_dataset.frame.iloc[index]
        expected.append(
            [
                math.log((float(row["AIR_Flow"]) + 1.0e-6) / (float(row["CH4_Flow"]) + 1.0e-6)),
                math.log((float(row["WATER_Flow"]) + 1.0e-6) / (float(row["CH4_Flow"]) + 1.0e-6)),
            ]
        )
    expected_tensor = torch.tensor(expected, dtype=torch.float32)
    assert mean.shape[0] == ratio_start + 2
    assert torch.allclose(mean[ratio_start:], expected_tensor.mean(dim=0), rtol=1.0e-5)
    assert torch.all(std[ratio_start:] > 0)


def test_feed_normalizer_is_train_featurewise_not_diluted_by_non_v_input_nodes(
    p03_f1_dataset,
) -> None:
    raw = list(p03_f1_dataset.iter_x_oper_for_normalizer(limit=4))
    mean, std = compute_oper_normalizer_from_x_oper(raw)
    feed_start = len(OPER_FEATURE_SLOTS)
    expected = torch.tensor(
        [
            [
                float(p03_f1_dataset.frame.iloc[i]["CH4_Flow"]),
                float(p03_f1_dataset.frame.iloc[i]["AIR_Flow"]),
                float(p03_f1_dataset.frame.iloc[i]["WATER_Flow"]),
            ]
            for i in range(4)
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(mean[feed_start:], expected.mean(dim=0), rtol=1.0e-5)
    assert torch.all(std[feed_start:] > 0)

    record = p03_f1_dataset[0]
    original_cfg = p03_f1_dataset.data_cfg
    original_mean = p03_f1_dataset.oper_mean
    original_std = p03_f1_dataset.oper_std
    try:
        p03_f1_dataset.data_cfg = replace(original_cfg, normalize_x_oper=True)
        p03_f1_dataset.oper_mean = mean
        p03_f1_dataset.oper_std = std
        normalized = p03_f1_dataset._apply_normalization(record.graph)
    finally:
        p03_f1_dataset.data_cfg = original_cfg
        p03_f1_dataset.oper_mean = original_mean
        p03_f1_dataset.oper_std = original_std
    v_input_index = record.graph.node_names.index("V_INPUT")
    scaled = torch.tensor(normalized.x_oper[v_input_index][feed_start:])
    restored = scaled * std[feed_start:] + mean[feed_start:]
    assert torch.allclose(
        restored,
        torch.tensor(record.graph.x_oper[v_input_index][feed_start:]),
        rtol=1.0e-5,
    )
    for node_index in range(len(normalized.node_names)):
        if node_index != v_input_index:
            assert normalized.x_oper[node_index][feed_start:] == [0.0, 0.0, 0.0]


def test_batching_keeps_each_graph_feed_on_its_own_v_input(p03_f1_dataset) -> None:
    records = [p03_f1_dataset[0], p03_f1_dataset[1]]
    batch = collate_graph_batch(records)
    offset = 0
    feed_start = len(OPER_FEATURE_SLOTS)
    for record in records:
        local_v_input = record.graph.node_names.index("V_INPUT")
        global_v_input = offset + local_v_input
        assert batch.model_kwargs["x_oper"][global_v_input, feed_start:].tolist() == pytest.approx(
            record.graph.x_oper[local_v_input][feed_start:]
        )
        graph_id = int(batch.model_kwargs["batch"][global_v_input].item())
        assert graph_id == records.index(record)
        offset += len(record.graph.node_names)


def test_real_zero_and_missing_are_distinguished_by_feed_mask(p03_f1_dataset) -> None:
    spec = p03_f1_dataset._load_spec("Process3")
    incoming = {name: [] for name in spec.node_names}
    zero_values, zero_masks = p03_f1_dataset._resolve_known_feed_values(
        pid=3,
        spec=spec,
        data_row={"CH4_Flow": 0.0, "AIR_Flow": 0.0, "WATER_Flow": 0.0},
        incoming_map=incoming,
    )
    missing_values, missing_masks = p03_f1_dataset._resolve_known_feed_values(
        pid=3,
        spec=spec,
        data_row={},
        incoming_map=incoming,
    )
    assert zero_values == [0.0, 0.0, 0.0]
    assert zero_masks == [1, 1, 1]
    assert missing_values == [0.0, 0.0, 0.0]
    assert missing_masks == [0, 0, 0]


def test_process_aware_source_mapping_handles_alias_and_absent_air(
    p03_f1_dataset,
) -> None:
    process2 = p03_f1_dataset._load_spec("Process2")
    process5 = p03_f1_dataset._load_spec("Process5")
    assert [binding.source_reference for binding in known_feed_bindings(process2)] == [
        "CH4_Total",
        "AIR_Flow",
        "WATER_Flow",
    ]
    assert [binding.source_reference for binding in known_feed_bindings(process5)] == [
        "CH4_Flow",
        None,
        "WATER_Flow",
    ]
    values, masks = p03_f1_dataset._resolve_known_feed_values(
        pid=5,
        spec=process5,
        data_row={"CH4_Flow": 10.0, "WATER_Flow": 20.0},
        incoming_map={name: [] for name in process5.node_names},
    )
    assert values == [10.0, 0.0, 20.0]
    assert masks == [1, 0, 1]


def test_p03_e014_prediction_is_connected_to_feed_inputs(
    f1_experiment,
    p03_f1_dataset,
) -> None:
    record = p03_f1_dataset[0]
    batch = collate_graph_batch([record])
    encoder_cfg = model_yaml_to_encoder_config(f1_experiment.model, f1_experiment.data)
    model = ProcessSurrogateModel(encoder_cfg, task_specs=[]).eval()

    model_kwargs = dict(batch.model_kwargs)
    x_oper = model_kwargs["x_oper"].clone().requires_grad_(True)
    model_kwargs["x_oper"] = x_oper
    prediction = model(model_kwargs)["main_stream_pred"]
    edge_index = record.graph.canonical_edge_ids.index("P03_E014")
    co_index = 2 + list(f1_experiment.model.species_order).index("CO")
    co_prediction = prediction[edge_index, co_index]
    gradient = torch.autograd.grad(co_prediction, x_oper)[0]
    v_input_index = record.graph.node_names.index("V_INPUT")
    feed_gradient = gradient[v_input_index, -len(KNOWN_FEED_FEATURE_NAMES) :]
    assert torch.isfinite(feed_gradient).all()

    perturbed_kwargs = dict(batch.model_kwargs)
    perturbed = perturbed_kwargs["x_oper"].clone()
    perturbed[v_input_index, -2] += 1.0
    perturbed_kwargs["x_oper"] = perturbed
    with torch.no_grad():
        perturbed_prediction = model(perturbed_kwargs)["main_stream_pred"][
            edge_index, co_index
        ]
    assert torch.isfinite(perturbed_prediction)
    assert not torch.equal(co_prediction.detach(), perturbed_prediction)
