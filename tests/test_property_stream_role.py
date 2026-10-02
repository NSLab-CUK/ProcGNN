from __future__ import annotations

import csv
from pathlib import Path

import torch

from process_graph.constants import (
    PROPERTY_STREAM_ROLE_TO_IDX,
    property_stream_role_id,
)
from process_graph.models.edge_decoder import EdgeDecoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARALLEL_PAIRS = (
    ("P04_E019", "P04_E020"),
    ("P06_E022", "P06_E023"),
    ("P07_E016", "P07_E017"),
    ("P08_E015", "P08_E016"),
)


def _canonical_rows() -> dict[str, dict[str, str]]:
    path = PROJECT_ROOT / "data" / "reference" / "v3" / "canonical_edges.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["canonical_edge_id"]: row for row in csv.DictReader(handle)}


def _decoder(*, enabled: bool) -> EdgeDecoder:
    return EdgeDecoder(
        node_hidden_dim=4,
        global_dim=8,
        edge_struct_dim=3,
        out_dim=14,
        hidden_dim=8,
        activation="relu",
        dropout=0.0,
        edge_head_type="pi_grouped_property",
        edge_head_dropout=0.0,
        cond_head_output_dim=2,
        frac_head_output_dim=7,
        mass_head_output_dim=3,
        property_head_hidden_dim=8,
        property_head_num_layers=2,
        fraction_activation="softmax",
        fraction_temperature=0.5,
        species_order=("H2O", "H2", "CH4", "CO2", "CO", "O2", "N2"),
        property_stream_role_enabled=enabled,
        property_stream_role_embedding_dim=16,
        property_stream_role_unknown_role_id=0,
    )


def _inputs() -> tuple[torch.Tensor, ...]:
    node_emb = torch.randn(2, 4)
    edge_index = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
    edge_struct = torch.ones(2, 3)
    global_emb = torch.randn(1, 8)
    batch = torch.zeros(2, dtype=torch.long)
    roles = torch.tensor(
        [
            PROPERTY_STREAM_ROLE_TO_IDX["output_product"],
            PROPERTY_STREAM_ROLE_TO_IDX["output_h2o"],
        ],
        dtype=torch.long,
    )
    return node_emb, edge_index, edge_struct, global_emb, batch, roles


def test_parallel_output_pairs_have_distinct_property_roles() -> None:
    rows = _canonical_rows()
    for edge_a, edge_b in PARALLEL_PAIRS:
        row_a, row_b = rows[edge_a], rows[edge_b]
        assert row_a["src_node"] == row_b["src_node"]
        assert row_a["dst_node"] == row_b["dst_node"]
        assert row_a["stream_role"] == row_b["stream_role"] == "output"
        role_a = property_stream_role_id(
            row_a["stream_role"], row_a["dst_node_raw"]
        )
        role_b = property_stream_role_id(
            row_b["stream_role"], row_b["dst_node_raw"]
        )
        assert role_a != role_b


def test_role_enabled_shapes_gradients_and_shuffle_effect() -> None:
    torch.manual_seed(7)
    decoder = _decoder(enabled=True).eval()
    inputs = _inputs()
    roles = inputs[-1]
    output = decoder(*inputs[:-1], property_stream_role=roles)
    shuffled = decoder(*inputs[:-1], property_stream_role=roles.flip(0))

    assert decoder.in_dim == 19
    assert decoder.property_head_input_dim == 35
    assert output["main_stream_pred"].shape == (2, 12)
    assert output["condition_pred"].shape == (2, 2)
    assert output["frac_pred"].shape == (2, 7)
    assert output["flow_pred"].shape == (2, 3)
    assert output["rho_pred"].shape == (2, 1)
    assert output["h_pred"].shape == (2, 1)
    assert not torch.allclose(
        output["main_stream_pred"], shuffled["main_stream_pred"]
    )

    output["main_stream_pred"].sum().backward()
    embedding = decoder.property_stream_role_embedding
    assert embedding is not None
    assert embedding.weight.grad is not None
    assert torch.isfinite(embedding.weight.grad).all()
    assert embedding.weight.grad.norm() > 0
    assert decoder.pi_head is not None
    for branch in (
        decoder.pi_head.condition_head,
        decoder.pi_head.fraction_head,
        decoder.pi_head.mass_flow_head,
    ):
        grad = branch[0].weight.grad
        assert grad is not None
        assert torch.isfinite(grad).all()


def test_role_disabled_is_exactly_role_independent() -> None:
    torch.manual_seed(11)
    decoder = _decoder(enabled=False).eval()
    inputs = _inputs()
    output_a = decoder(*inputs[:-1], property_stream_role=inputs[-1])
    output_b = decoder(*inputs[:-1], property_stream_role=inputs[-1].flip(0))

    assert decoder.in_dim == 19
    assert decoder.property_head_input_dim == 19
    assert decoder.property_stream_role_embedding is None
    for key in output_a:
        assert torch.equal(output_a[key], output_b[key])


def test_production_property_head_dimensions() -> None:
    decoder_off = EdgeDecoder(
        node_hidden_dim=512,
        global_dim=1024,
        edge_struct_dim=3,
        out_dim=14,
        hidden_dim=384,
        activation="relu",
        dropout=0.0,
        edge_head_type="pi_grouped_property",
        cond_head_output_dim=2,
        frac_head_output_dim=7,
        mass_head_output_dim=3,
        property_head_hidden_dim=128,
        property_head_num_layers=2,
    )
    decoder_on = EdgeDecoder(
        node_hidden_dim=512,
        global_dim=1024,
        edge_struct_dim=3,
        out_dim=14,
        hidden_dim=384,
        activation="relu",
        dropout=0.0,
        edge_head_type="pi_grouped_property",
        cond_head_output_dim=2,
        frac_head_output_dim=7,
        mass_head_output_dim=3,
        property_head_hidden_dim=128,
        property_head_num_layers=2,
        property_stream_role_enabled=True,
        property_stream_role_embedding_dim=16,
    )
    assert decoder_off.property_head_input_dim == 2051
    assert decoder_on.property_head_input_dim == 2067
    assert decoder_off.pi_head is not None
    assert decoder_on.pi_head is not None
    assert decoder_off.pi_head.rho_head[0].in_features == 9
    assert decoder_off.pi_head.h_head[0].in_features == 9
    assert decoder_on.pi_head.rho_head[0].in_features == 9
    assert decoder_on.pi_head.h_head[0].in_features == 9
