from process_graph.data.tabular_dataset import (
    V3_EDGE_STRUCT_FEATURE_COLUMNS,
    V3_EDGE_STRUCT_ROLE_COLUMNS,
    _edge_struct_feature_columns,
    _edge_struct_values,
)
from process_graph.experiment.schema import DataConfig


def test_edge_struct_default_keeps_join_flags() -> None:
    cfg = DataConfig()

    assert _edge_struct_feature_columns(cfg) == V3_EDGE_STRUCT_FEATURE_COLUMNS
    assert _edge_struct_values(
        is_input_edge=True,
        is_output_edge=False,
        is_internal_edge=False,
        has_stream_key=1.0,
        join_ok=1.0,
        data_cfg=cfg,
    ) == [1.0, 0.0, 0.0, 1.0, 1.0]


def test_edge_struct_can_drop_join_flags() -> None:
    cfg = DataConfig(edge_struct_include_join_flags=False)

    assert _edge_struct_feature_columns(cfg) == V3_EDGE_STRUCT_ROLE_COLUMNS
    assert _edge_struct_values(
        is_input_edge=True,
        is_output_edge=False,
        is_internal_edge=False,
        has_stream_key=1.0,
        join_ok=1.0,
        data_cfg=cfg,
    ) == [1.0, 0.0, 0.0]
