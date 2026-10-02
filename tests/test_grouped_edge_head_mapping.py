from __future__ import annotations

import torch
from torch import nn

from process_graph.constants import STREAM_EDGE_FEATURE_SLOTS
from process_graph.experiment.loaders import load_experiment_config
from process_graph.models.edge_decoder import EdgeDecoder


class _FixedHead(nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.values.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1)


def test_grouped_property_head_reassembles_current_stream_feature_order() -> None:
    decoder = EdgeDecoder(
        node_hidden_dim=4,
        global_dim=3,
        edge_struct_dim=5,
        out_dim=12,
        hidden_dim=8,
        activation="relu",
        dropout=0.0,
        edge_head_type="grouped_property",
    )
    decoder.grouped_input_proj = nn.Identity()
    decoder.shared_proj = nn.Identity()
    decoder.frac_head = _FixedHead([10, 11, 12, 13, 14, 15, 16])
    decoder.flow_head = _FixedHead([20, 21, 22])
    decoder.cond_head = _FixedHead([30, 31])

    out = decoder._forward_grouped_property(torch.zeros(2, 8))
    row = out[0]
    expected = {
        "Frac_CH4": 10,
        "Frac_CO": 11,
        "Frac_CO2": 12,
        "Frac_H2": 13,
        "Frac_H2O": 14,
        "Frac_N2": 15,
        "Frac_O2": 16,
        "Mass_Flow": 20,
        "Mole_Flow": 21,
        "Vol_Flow": 22,
        "Pres": 30,
        "Temp": 31,
    }
    for name, value in expected.items():
        idx = STREAM_EDGE_FEATURE_SLOTS.index(name)
        assert float(row[idx]) == float(value)


def test_grouped_property_config_rejects_custom_explicit_target_order(tmp_path) -> None:
    model_yaml = tmp_path / "model.yaml"
    train_yaml = tmp_path / "train.yaml"
    data_yaml = tmp_path / "data.yaml"
    exp_yaml = tmp_path / "experiment.yaml"
    model_yaml.write_text(
        "\n".join(
            [
                "use_edge_decoder: true",
                "edge_head_type: grouped_property",
                "stream_target_dim: 12",
            ]
        ),
        encoding="utf-8",
    )
    train_yaml.write_text("", encoding="utf-8")
    data_yaml.write_text(
        "\n".join(
            [
                "topology_mode: stream_edge",
                "use_canonical_graph_spec_v3: true",
                "task_mode: edge_all",
                "fixed_tasks:",
                "  target:",
                "    enabled: true",
                "    target_name: target",
                "  tailgas:",
                "    enabled: true",
                "    target_name: tailgas",
                "edge_target_columns_mode: explicit",
                "edge_target_columns:",
                "  - Frac_CH4",
                "  - Frac_CO",
                "  - Frac_CO2",
                "  - Frac_H2",
                "  - Frac_H2O",
                "  - Frac_N2",
                "  - Frac_O2",
                "  - Mass_Flow",
                "  - Mole_Flow",
                "  - Pres",
                "  - Temp",
                "  - Vol_Flow",
            ]
        ),
        encoding="utf-8",
    )
    exp_yaml.write_text(
        "\n".join(
            [
                "model_config: " + model_yaml.as_posix(),
                "train_config: " + train_yaml.as_posix(),
                "data_config: " + data_yaml.as_posix(),
            ]
        ),
        encoding="utf-8",
    )

    try:
        load_experiment_config(exp_yaml)
    except ValueError as exc:
        assert "grouped_property" in str(exc)
        assert "edge_target_columns" in str(exc)
    else:
        raise AssertionError("grouped_property should reject custom explicit edge_target_columns order")
