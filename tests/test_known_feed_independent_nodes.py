from __future__ import annotations

import sys
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from process_graph.constants import OPER_FEATURE_SLOTS, ROLE_TO_IDX, UNIT_TO_IDX
from process_graph.data.tabular_dataset import (
    ProcessGraphTabularDataset,
    collate_graph_batch,
    compute_oper_normalizer_from_x_oper,
)
from process_graph.experiment.config_builders import model_yaml_to_encoder_config
from process_graph.experiment.edge_step_pi_training import (
    _edge_groups_from_export_meta,
)
from process_graph.experiment.loaders import load_experiment_config
from process_graph.experiment.node_balance_pi import compute_node_balance_pinn_losses
from process_graph.known_feed import KNOWN_FEED_NAMES
from process_graph.models.process_surrogate import ProcessSurrogateModel


CONFIG_DIR = PROJECT_ROOT / "configs" / "experiment" / "pinn" / "known_feed_260730"


@pytest.fixture(scope="module")
def f0_experiment():
    return load_experiment_config(CONFIG_DIR / "f0_e2_baseline.yaml")


@pytest.fixture(scope="module")
def f2_experiment():
    return load_experiment_config(CONFIG_DIR / "f2_e2_independent_feed_nodes.yaml")


@pytest.fixture(scope="module")
def f0_dataset(f0_experiment):
    cfg = replace(
        f0_experiment.data,
        normalize_x_oper=False,
        normalize_targets=False,
        edge_all_processes=[3, 5],
    )
    return ProcessGraphTabularDataset(
        f0_experiment.data.train_data_path,
        cfg,
        f0_experiment.project_root,
    )


@pytest.fixture(scope="module")
def f2_dataset(f2_experiment):
    cfg = replace(
        f2_experiment.data,
        normalize_x_oper=False,
        normalize_targets=False,
        edge_all_processes=[3, 5],
    )
    return ProcessGraphTabularDataset(
        f2_experiment.data.train_data_path,
        cfg,
        f2_experiment.project_root,
    )


def _record_for_process(dataset: ProcessGraphTabularDataset, process_id: str):
    process_col = dataset.data_cfg.process_id_column
    index = int(
        dataset.frame.index[
            dataset.frame[process_col].astype(str).eq(process_id)
        ][0]
    )
    return dataset[index]


def _shortest_path(
    graph,
    start: str,
    goal: str,
    *,
    directed: bool,
) -> int | None:
    adjacency = {name: set() for name in graph.node_names}
    for src, dst in zip(graph.edge_index[0], graph.edge_index[1]):
        src_name = graph.node_names[int(src)]
        dst_name = graph.node_names[int(dst)]
        adjacency[src_name].add(dst_name)
        if not directed:
            adjacency[dst_name].add(src_name)
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        node, distance = queue.popleft()
        if node == goal:
            return distance
        for neighbor in adjacency[node]:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append((neighbor, distance + 1))
    return None


def test_f2_config_changes_only_known_feed_representation(
    f0_experiment,
    f2_experiment,
) -> None:
    f0_model = asdict(f0_experiment.model)
    f2_model = asdict(f2_experiment.model)
    f0_model.pop("known_feed_condition")
    f2_model.pop("known_feed_condition")
    assert f2_model == f0_model
    assert asdict(f2_experiment.train) == asdict(f0_experiment.train)

    f0_data = asdict(f0_experiment.data)
    f2_data = asdict(f2_experiment.data)
    f0_data.pop("known_feed_condition")
    f2_data.pop("known_feed_condition")
    assert f2_data == f0_data

    f0_encoder = model_yaml_to_encoder_config(
        f0_experiment.model, f0_experiment.data
    )
    f2_encoder = model_yaml_to_encoder_config(
        f2_experiment.model, f2_experiment.data
    )
    assert f0_encoder.oper_dim == 13
    assert f2_encoder.oper_dim == 16
    assert f0_encoder.hidden_dim == f2_encoder.hidden_dim == 384
    assert f0_encoder.operating_hidden_dim == f2_encoder.operating_hidden_dim == 64


def test_p03_raw_values_identity_topology_and_context_masks(
    f0_dataset,
    f2_dataset,
) -> None:
    f0 = _record_for_process(f0_dataset, "Process3").graph
    f2_record = _record_for_process(f2_dataset, "Process3")
    f2 = f2_record.graph
    row = f2_dataset.frame.iloc[f2_record.sample_meta["row_index"]]
    feed_start = len(OPER_FEATURE_SLOTS)

    assert f2.node_names[: len(f0.node_names)] == f0.node_names
    assert f2.canonical_edge_ids[: len(f0.canonical_edge_ids)] == f0.canonical_edge_ids
    assert f2.edge_index[0][: len(f0.canonical_edge_ids)] == f0.edge_index[0]
    assert f2.edge_index[1][: len(f0.canonical_edge_ids)] == f0.edge_index[1]
    assert f2.y_edge_mask[: len(f0.canonical_edge_ids)] == f0.y_edge_mask
    assert f2.edge_is_target[: len(f0.canonical_edge_ids)] == f0.edge_is_target
    assert torch.allclose(
        torch.tensor(
            [
                values[:feed_start]
                for values in f2.x_oper[: len(f0.node_names)]
            ]
        ),
        torch.tensor(f0.x_oper),
    )

    expected = {
        "V_FEED_CH4": ("CH4_Flow", 0, "C2", "P03_F2_FEED_CH4"),
        "V_FEED_AIR": ("AIR_Flow", 1, "MIX2", "P03_F2_FEED_AIR"),
        "V_FEED_WATER": ("WATER_Flow", 2, "P1", "P03_F2_FEED_WATER"),
    }
    for node_name, (column, slot, destination, edge_id) in expected.items():
        node_index = f2.node_names.index(node_name)
        assert f2.x_role[node_index] == ROLE_TO_IDX["source"]
        assert f2.x_unit[node_index] == UNIT_TO_IDX["stream"]
        assert f2.node_is_context[node_index] == 1.0
        assert f2.node_pinn_mask[node_index] == 0.0
        assert f2.x_oper[node_index][feed_start + slot] == pytest.approx(
            float(row[column])
        )
        assert f2.x_oper_mask[node_index][feed_start + slot] == 1
        for other_slot in set(range(3)) - {slot}:
            assert f2.x_oper[node_index][feed_start + other_slot] == 0.0
            assert f2.x_oper_mask[node_index][feed_start + other_slot] == 0

        edge_index = f2.canonical_edge_ids.index(edge_id)
        assert f2.node_names[f2.edge_index[0][edge_index]] == node_name
        assert f2.node_names[f2.edge_index[1][edge_index]] == destination
        assert f2.edge_is_context[edge_index] == 1.0
        assert f2.edge_is_predictable[edge_index] == 0.0
        assert f2.edge_is_supervised[edge_index] == 0.0
        assert f2.edge_is_target[edge_index] == 0.0
        assert f2.edge_pinn_mask[edge_index] == 0.0
        assert f2.y_edge_mask[edge_index] == 0.0

    v_input_index = f2.node_names.index("V_INPUT")
    assert f2.x_oper[v_input_index][feed_start:] == [0.0, 0.0, 0.0]
    assert f2.x_oper_mask[v_input_index][feed_start:] == [0, 0, 0]
    assert len(f2.node_names) == len(f0.node_names) + 3
    assert len(f2.canonical_edge_ids) == len(f0.canonical_edge_ids) + 3

    target_index_f0 = f0.canonical_edge_ids.index("P03_E014")
    target_index_f2 = f2.canonical_edge_ids.index("P03_E014")
    assert target_index_f2 == target_index_f0
    assert (
        f2.node_names[f2.edge_index[0][target_index_f2]],
        f2.node_names[f2.edge_index[1][target_index_f2]],
    ) == (
        f0.node_names[f0.edge_index[0][target_index_f0]],
        f0.node_names[f0.edge_index[1][target_index_f0]],
    )


def test_missing_air_omits_p05_feed_node_and_edge(f0_dataset, f2_dataset) -> None:
    f0 = _record_for_process(f0_dataset, "Process5").graph
    f2 = _record_for_process(f2_dataset, "Process5").graph
    assert "V_FEED_AIR" not in f2.node_names
    assert "P05_F2_FEED_AIR" not in f2.canonical_edge_ids
    assert f2.node_names[-2:] == ["V_FEED_CH4", "V_FEED_WATER"]
    assert f2.canonical_edge_ids[-2:] == [
        "P05_F2_FEED_CH4",
        "P05_F2_FEED_WATER",
    ]
    assert len(f2.node_names) == len(f0.node_names) + 2
    assert len(f2.canonical_edge_ids) == len(f0.canonical_edge_ids) + 2


def test_feed_normalizer_uses_only_each_active_feed_slot(f2_dataset) -> None:
    raw = list(f2_dataset.iter_x_oper_for_normalizer(limit=4))
    mean, std = compute_oper_normalizer_from_x_oper(raw)
    feed_start = len(OPER_FEATURE_SLOTS)
    expected = torch.tensor(
        [
            [
                float(f2_dataset.frame.iloc[i]["CH4_Flow"]),
                float(f2_dataset.frame.iloc[i]["AIR_Flow"]),
                float(f2_dataset.frame.iloc[i]["WATER_Flow"]),
            ]
            for i in range(4)
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(mean[feed_start:], torch.nanmean(expected, dim=0), rtol=1.0e-5)
    assert torch.all(std[feed_start:] > 0)

    record = _record_for_process(f2_dataset, "Process3")
    original_cfg = f2_dataset.data_cfg
    original_mean = f2_dataset.oper_mean
    original_std = f2_dataset.oper_std
    try:
        f2_dataset.data_cfg = replace(original_cfg, normalize_x_oper=True)
        f2_dataset.oper_mean = mean
        f2_dataset.oper_std = std
        normalized = f2_dataset._apply_normalization(record.graph)
    finally:
        f2_dataset.data_cfg = original_cfg
        f2_dataset.oper_mean = original_mean
        f2_dataset.oper_std = original_std

    for feed_slot, feed_name in enumerate(KNOWN_FEED_NAMES):
        node_index = record.graph.node_names.index(f"V_FEED_{feed_name}")
        scaled = torch.tensor(normalized.x_oper[node_index][feed_start:])
        restored = scaled[feed_slot] * std[feed_start + feed_slot] + mean[
            feed_start + feed_slot
        ]
        assert restored == pytest.approx(
            record.graph.x_oper[node_index][feed_start + feed_slot],
            rel=1.0e-5,
        )
        for other_slot in set(range(3)) - {feed_slot}:
            assert normalized.x_oper[node_index][feed_start + other_slot] == 0.0


def test_batch_offsets_do_not_create_cross_graph_edges(f2_dataset) -> None:
    records = [
        _record_for_process(f2_dataset, "Process3"),
        _record_for_process(f2_dataset, "Process5"),
    ]
    batch = collate_graph_batch(records)
    edge_index = batch.model_kwargs["edge_index"]
    node_batch = batch.model_kwargs["batch"]
    edge_batch = batch.model_kwargs["edge_batch"]
    assert torch.equal(node_batch[edge_index[0]], edge_batch)
    assert torch.equal(node_batch[edge_index[1]], edge_batch)
    assert int(batch.model_kwargs["node_is_context"].sum().item()) == 5
    assert int(batch.model_kwargs["edge_is_context"].sum().item()) == 5
    assert int((batch.model_kwargs["edge_pinn_mask"] <= 0.5).sum().item()) == 5
    groups = _edge_groups_from_export_meta(
        batch.edge_export_meta,
        int(edge_index.shape[1]),
    )
    grouped_indices = {index for _, indices in groups for index in indices}
    context_indices = {
        index
        for index, value in enumerate(batch.edge_export_meta.is_context)
        if float(value) > 0.5
    }
    assert grouped_indices.isdisjoint(context_indices)


def test_p03_feed_nodes_reach_p03_e014_with_bidirectional_flowgnn(
    f2_dataset,
) -> None:
    graph = _record_for_process(f2_dataset, "Process3").graph
    target_edge_index = graph.canonical_edge_ids.index("P03_E014")
    target_source = graph.node_names[graph.edge_index[0][target_edge_index]]
    expected_directed = {
        "V_FEED_CH4": 3,
        "V_FEED_AIR": 2,
        "V_FEED_WATER": 7,
    }
    expected_undirected = {
        "V_FEED_CH4": 3,
        "V_FEED_AIR": 2,
        "V_FEED_WATER": 4,
    }
    for feed_name in KNOWN_FEED_NAMES:
        node_name = f"V_FEED_{feed_name}"
        assert _shortest_path(
            graph, node_name, target_source, directed=True
        ) == expected_directed[node_name]
        assert _shortest_path(
            graph, node_name, target_source, directed=False
        ) == expected_undirected[node_name]
        assert expected_undirected[node_name] <= 5


def test_context_edges_are_exactly_excluded_from_node_pinn(
    f0_experiment,
    f0_dataset,
    f2_dataset,
) -> None:
    torch.manual_seed(11)
    f0 = _record_for_process(f0_dataset, "Process3").graph
    f2 = _record_for_process(f2_dataset, "Process3").graph
    e0 = len(f0.canonical_edge_ids)
    e2 = len(f2.canonical_edge_ids)
    base_mass = torch.rand(e0, 1) + 1.0
    base_frac = torch.softmax(torch.randn(e0, 7), dim=-1)
    context_mass = torch.full((e2 - e0, 1), 1.0e12)
    context_frac = torch.softmax(torch.randn(e2 - e0, 7), dim=-1)

    common = {
        "edge_batch_or_graph_id": torch.zeros(e0, dtype=torch.long),
        "node_batch_or_graph_id": torch.zeros(len(f0.node_names), dtype=torch.long),
        "node_unit_type": torch.tensor(f0.x_unit, dtype=torch.long),
        "species_order": ("H2O", "H2", "CH4", "CO2", "CO", "O2", "N2"),
        "train_cfg": f0_experiment.train,
        "physical_targets": {"mass_flow": base_mass, "frac": base_frac},
        "node_balance_exclude_mass": torch.tensor(f0.node_balance_exclude_mass),
        "node_balance_exclude_component": torch.tensor(f0.node_balance_exclude_component),
        "node_balance_exclude_atom": torch.tensor(f0.node_balance_exclude_atom),
        "node_balance_exclude_energy": torch.tensor(f0.node_balance_exclude_energy),
    }
    result_f0 = compute_node_balance_pinn_losses(
        physical_outputs={"mass_flow": base_mass, "frac": base_frac},
        edge_index=torch.tensor(f0.edge_index, dtype=torch.long),
        **common,
    )
    result_f2 = compute_node_balance_pinn_losses(
        physical_outputs={
            "mass_flow": torch.cat([base_mass, context_mass], dim=0),
            "frac": torch.cat([base_frac, context_frac], dim=0),
        },
        physical_targets={
            "mass_flow": torch.cat([base_mass, context_mass], dim=0),
            "frac": torch.cat([base_frac, context_frac], dim=0),
        },
        edge_index=torch.tensor(f2.edge_index, dtype=torch.long),
        edge_batch_or_graph_id=torch.zeros(e2, dtype=torch.long),
        node_batch_or_graph_id=torch.zeros(len(f2.node_names), dtype=torch.long),
        node_unit_type=torch.tensor(f2.x_unit, dtype=torch.long),
        species_order=common["species_order"],
        train_cfg=f0_experiment.train,
        node_balance_exclude_mass=torch.tensor(f2.node_balance_exclude_mass),
        node_balance_exclude_component=torch.tensor(f2.node_balance_exclude_component),
        node_balance_exclude_atom=torch.tensor(f2.node_balance_exclude_atom),
        node_balance_exclude_energy=torch.tensor(f2.node_balance_exclude_energy),
        edge_pinn_mask=torch.tensor(f2.edge_pinn_mask),
    )
    for key in (
        "loss_node_mass",
        "loss_node_component",
        "loss_node_atom",
        "weighted_node_total",
    ):
        assert torch.equal(result_f2[key], result_f0[key])
    assert result_f2["diagnostics"]["node_pinn_edge_count"] == e0
    assert result_f2["diagnostics"]["node_pinn_edge_excluded_count"] == e2 - e0
    for key in (
        "node_mass_valid_count",
        "node_component_valid_count",
        "node_atom_valid_count",
        "node_energy_valid_count",
    ):
        assert result_f2["diagnostics"][key] == result_f0["diagnostics"][key]


def test_feed_perturbations_reach_flowgnn_layers_and_p03_e014(
    f2_experiment,
    f2_dataset,
) -> None:
    torch.manual_seed(17)
    record = _record_for_process(f2_dataset, "Process3")
    batch = collate_graph_batch([record])
    model = ProcessSurrogateModel(
        model_yaml_to_encoder_config(f2_experiment.model, f2_experiment.data),
        task_specs=[],
    )
    model.eval()
    batch_data = dict(batch.model_kwargs)
    with torch.no_grad():
        baseline = model(batch_data, return_encoder_output=True)
    baseline_encoder = baseline["encoder_output"]
    baseline_predictions = baseline["predictions"]

    target_edge_index = record.graph.canonical_edge_ids.index("P03_E014")
    target_source = int(record.graph.edge_index[0][target_edge_index])
    target_destination = int(record.graph.edge_index[1][target_edge_index])
    co_index = 2 + list(f2_experiment.model.species_order).index("CO")
    feed_start = len(OPER_FEATURE_SLOTS)

    for feed_slot, feed_name in enumerate(KNOWN_FEED_NAMES):
        node_name = f"V_FEED_{feed_name}"
        node_index = record.graph.node_names.index(node_name)
        context_edge_index = record.graph.canonical_edge_ids.index(
            f"P03_F2_FEED_{feed_name}"
        )
        destination_index = int(record.graph.edge_index[1][context_edge_index])
        perturbed_data = dict(batch_data)
        perturbed_data["x_oper"] = batch_data["x_oper"].clone()
        raw_value = float(
            perturbed_data["x_oper"][node_index, feed_start + feed_slot]
        )
        perturbed_data["x_oper"][node_index, feed_start + feed_slot] += max(
            abs(raw_value) * 0.2,
            1.0,
        )
        with torch.no_grad():
            perturbed = model(perturbed_data, return_encoder_output=True)
        perturbed_encoder = perturbed["encoder_output"]
        perturbed_predictions = perturbed["predictions"]

        assert not torch.equal(
            perturbed_encoder.initial_node_embeddings[node_index],
            baseline_encoder.initial_node_embeddings[node_index],
        )
        assert not torch.equal(
            perturbed_encoder.layer_node_embeddings[0][destination_index],
            baseline_encoder.layer_node_embeddings[0][destination_index],
        )
        assert not torch.equal(
            perturbed_encoder.local_node_embeddings[target_source],
            baseline_encoder.local_node_embeddings[target_source],
        )
        assert not torch.equal(
            perturbed_encoder.local_node_embeddings[target_destination],
            baseline_encoder.local_node_embeddings[target_destination],
        )
        assert not torch.equal(
            perturbed_encoder.global_embedding,
            baseline_encoder.global_embedding,
        )
        assert (
            perturbed_predictions["main_stream_pred"][
                target_edge_index, co_index
            ].item()
            != baseline_predictions["main_stream_pred"][
                target_edge_index, co_index
            ].item()
        )


def test_full_forward_gradient_and_optimizer_smoke(
    f2_experiment,
    f2_dataset,
) -> None:
    torch.manual_seed(7)
    record = _record_for_process(f2_dataset, "Process3")
    batch = collate_graph_batch([record])
    model = ProcessSurrogateModel(
        model_yaml_to_encoder_config(f2_experiment.model, f2_experiment.data),
        task_specs=[],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    batch_data = dict(batch.model_kwargs)
    x_oper = batch_data["x_oper"].clone().requires_grad_(True)
    batch_data["x_oper"] = x_oper
    outputs = model(batch_data)
    prediction = outputs["main_stream_pred"]
    assert prediction.shape == (len(record.graph.canonical_edge_ids), 11)
    assert torch.isfinite(prediction).all()

    target_edge_index = record.graph.canonical_edge_ids.index("P03_E014")
    co_index = 2 + list(f2_experiment.model.species_order).index("CO")
    co_prediction = prediction[target_edge_index, co_index]
    feed_gradient = torch.autograd.grad(
        co_prediction, x_oper, retain_graph=True
    )[0]
    feed_start = len(OPER_FEATURE_SLOTS)
    for feed_slot, feed_name in enumerate(KNOWN_FEED_NAMES):
        node_index = record.graph.node_names.index(f"V_FEED_{feed_name}")
        scalar_gradient = feed_gradient[node_index, feed_start + feed_slot]
        assert torch.isfinite(scalar_gradient)
        assert float(scalar_gradient.abs()) > 0.0

    supervised = batch.model_kwargs["y_edge_mask"].reshape(-1, 1)
    context = batch.model_kwargs["edge_is_context"].reshape(-1, 1)
    smoke_loss = (prediction.square() * supervised).sum() / supervised.sum().clamp_min(1.0)
    assert torch.all(supervised[context > 0.5] == 0.0)
    optimizer.zero_grad(set_to_none=True)
    smoke_loss.backward()
    optimizer.step()
    assert torch.isfinite(smoke_loss)
