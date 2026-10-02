from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from process_graph.experiment.config_builders import model_yaml_to_encoder_config
from process_graph.experiment.loaders import load_experiment_config
from process_graph.data.tabular_dataset import GraphSampleRecord, collate_graph_batch
from process_graph.hx_pair import parse_hx_pair_relation
from process_graph.models.edge_decoder import HXPairRelationAdapter
from process_graph.models.process_surrogate import ProcessSurrogateModel
from process_graph.schema import GraphSample


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/experiment/pinn/co_ratio_regime_260803"
R0 = CONFIG_DIR / "c1_f1_ratio.yaml"
HX = CONFIG_DIR / "c1_f1_ratio_r0_hx_pair.yaml"


def test_r0_is_preserved_and_hx_is_strictly_opt_in() -> None:
    raw_r0 = yaml.safe_load(R0.read_text(encoding="utf-8"))
    raw_hx = yaml.safe_load(HX.read_text(encoding="utf-8"))
    assert raw_r0["overrides"]["train"] == raw_hx["overrides"]["train"]
    assert raw_r0["overrides"]["data"] == raw_hx["overrides"]["data"]
    base_model = deepcopy(raw_r0["overrides"]["model"])
    hx_model = deepcopy(raw_hx["overrides"]["model"])
    relation = hx_model.pop("hx_pair_relation")
    assert base_model == hx_model
    assert relation["enabled"] is True
    assert relation["relation_mlp"] == {
        "input_dim": 784,
        "hidden_dim": 256,
        "output_dim": 384,
        "dropout": 0.1,
    }

    base_exp = load_experiment_config(R0)
    hx_exp = load_experiment_config(HX)
    base_cfg = model_yaml_to_encoder_config(base_exp.model, base_exp.data)
    hx_cfg = model_yaml_to_encoder_config(hx_exp.model, hx_exp.data)
    base = ProcessSurrogateModel(base_cfg)
    hx = ProcessSurrogateModel(hx_cfg)
    assert base_cfg.hx_pair_relation_enabled is False
    assert hx_cfg.hx_pair_relation_enabled is True
    assert not any("hx_pair_relation" in key for key in base.state_dict())
    assert any("hx_pair_relation" in key for key in hx.state_dict())
    assert sum(p.numel() for p in hx.edge_decoder.hx_pair_relation.parameters()) == 300_977


def test_parser_rejects_wrong_shape_or_nonfinal_application() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        parse_hx_pair_relation(
            {
                "enabled": True,
                "side_embedding_dim": 16,
                "relation_mlp": {"input_dim": 785, "output_dim": 384},
            }
        )
    with pytest.raises(ValueError, match="apply_to"):
        parse_hx_pair_relation(
            {"enabled": True, "apply_to": ["inside_flow_gnn"]}
        )


def _adapter_inputs() -> tuple[torch.Tensor, ...]:
    # Edge 1 participates in two adjacent HX units. Edge 3 is non-HX.
    edges = torch.randn(4, 8, requires_grad=True)
    current = torch.tensor([0, 1, 1, 2])
    paired = torch.tensor([1, 0, 2, 1])
    mask = torch.ones(4)
    side = torch.tensor([1, 2, 1, 2])
    graph_ids = torch.zeros(4, dtype=torch.long)
    return edges, current, paired, mask, side, graph_ids


def test_adapter_identity_sensitivity_shape_gate_and_gradients() -> None:
    torch.manual_seed(7)
    adapter = HXPairRelationAdapter(
        edge_dim=8,
        side_embedding_dim=4,
        hidden_dim=6,
        output_dim=8,
        dropout=0.0,
        gate_init=0.05,
    )
    edges, current, paired, mask, side, graph_ids = _adapter_inputs()
    output = adapter(edges, current, paired, mask, side, edge_graph_ids=graph_ids)
    assert output.shape == edges.shape
    assert torch.equal(output[3], edges[3])
    torch.testing.assert_close(adapter.gate, torch.tensor(0.05))

    changed = edges.detach().clone()
    changed[0] += 2.0
    changed_output = adapter(
        changed, current, paired, mask, side, edge_graph_ids=graph_ids
    )
    assert not torch.allclose(output[1].detach(), changed_output[1])
    torch.testing.assert_close(output[2].detach(), changed_output[2])

    output.square().mean().backward()
    assert edges.grad is not None and torch.isfinite(edges.grad).all()
    for parameter in adapter.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_adapter_rejects_asymmetry_and_cross_graph_pairs() -> None:
    adapter = HXPairRelationAdapter(
        edge_dim=8, side_embedding_dim=4, hidden_dim=6, output_dim=8, dropout=0.0
    )
    edges, current, paired, mask, side, graph_ids = _adapter_inputs()
    with pytest.raises(ValueError, match="symmetric"):
        adapter(edges, current[:-1], paired[:-1], mask[:-1], side[:-1])
    bad_graph_ids = graph_ids.clone()
    bad_graph_ids[1] = 1
    with pytest.raises(ValueError, match="graph boundaries"):
        adapter(
            edges,
            current,
            paired,
            mask,
            side,
            edge_graph_ids=bad_graph_ids,
        )


def _record(process_id: str) -> GraphSampleRecord:
    graph = GraphSample(
        process_id=process_id,
        node_names=["A", "HX1", "B"],
        edge_index=[[0, 1], [1, 2]],
        x_role=[0, 0, 0],
        x_unit=[0, 0, 0],
        x_hx_role=[0, 0, 0],
        x_oper=[[0.0], [0.0], [0.0]],
        x_oper_mask=[[1], [1], [1]],
        targets={},
        hx_pair_current_edge_index=[0, 1],
        hx_paired_edge_index=[1, 0],
        hx_pair_mask=[1.0, 1.0],
        hx_pair_side=[1, 2],
    )
    return GraphSampleRecord(
        graph=graph,
        slot_targets={},
        slot_masks={},
        slot_node_index={},
        category_node_indices={},
        category_values={},
        sample_meta={},
    )


def test_collate_offsets_pair_relations_without_cross_batch_leakage() -> None:
    batch = collate_graph_batch([_record("Process1"), _record("Process2")])
    kwargs = batch.model_kwargs
    assert kwargs["hx_pair_current_edge_index"].tolist() == [0, 1, 2, 3]
    assert kwargs["hx_paired_edge_index"].tolist() == [1, 0, 3, 2]
    edge_batch = kwargs["edge_batch"]
    assert torch.equal(
        edge_batch[kwargs["hx_pair_current_edge_index"]],
        edge_batch[kwargs["hx_paired_edge_index"]],
    )


def test_static_metadata_has_full_hx_coverage_and_no_label_leakage() -> None:
    edges = pd.read_csv(ROOT / "data/reference/v3/canonical_edges.csv")
    pairs = pd.read_csv(ROOT / "data/reference/v3/hx_edge_pairs.csv")
    assert list(pairs.columns) == [
        "process_id",
        "hx_node",
        "pair_id",
        "inlet_edge_id",
        "outlet_edge_id",
        "source",
    ]
    forbidden = {"target", "prediction", "loss", "frac", "mass_flow", "y_true"}
    assert not any(token in column.lower() for column in pairs.columns for token in forbidden)
    assert not pairs.duplicated(["process_id", "hx_node", "inlet_edge_id"]).any()
    assert not pairs.duplicated(["process_id", "hx_node", "outlet_edge_id"]).any()

    hx_units = set()
    for row in pairs.itertuples(index=False):
        process_edges = edges[edges["process_id"] == row.process_id].set_index(
            "canonical_edge_id"
        )
        assert process_edges.loc[row.inlet_edge_id, "dst_node"] == row.hx_node
        assert process_edges.loc[row.outlet_edge_id, "src_node"] == row.hx_node
        hx_units.add((int(row.process_id), str(row.hx_node)))
    expected_hx_units = set()
    for row in edges.itertuples(index=False):
        if str(row.src_node).startswith("HX"):
            expected_hx_units.add((int(row.process_id), str(row.src_node)))
        if str(row.dst_node).startswith("HX"):
            expected_hx_units.add((int(row.process_id), str(row.dst_node)))
    assert hx_units == expected_hx_units
    assert len(hx_units) == 35
    assert len(pairs) == 70
    assert 8 not in set(pairs["process_id"])
