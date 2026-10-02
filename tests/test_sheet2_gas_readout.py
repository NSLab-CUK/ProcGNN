"""Unit tests for H2 / CO2 readout node resolution from highlighted sink lists."""

from process_graph.parser import resolve_gas_readout_nodes_from_highlight_list


def test_process2_like_order():
    yellow = ["OUT_PROD", "OUT_EXHAUST"]
    nodes = ["IN_1", "R1", "OUT_PROD", "OUT_EXHAUST"]
    got = resolve_gas_readout_nodes_from_highlight_list(yellow, nodes)
    assert got["target"] == "OUT_PROD"
    assert got["tailgas"] == "OUT_EXHAUST"


def test_process6_co2_before_exhaust_in_list():
    yellow = ["OUT_PROD", "OUT_H2O", "OUT_CO2", "OUT_EXHAUST"]
    nodes = list(yellow)
    got = resolve_gas_readout_nodes_from_highlight_list(yellow, nodes)
    assert got["target"] == "OUT_PROD"
    assert got["tailgas"] == "OUT_CO2"


def test_process1_case_and_exhaust():
    yellow = ["OUT_exhaust", "OUT_prod", "OUT_RESTEAM"]
    nodes = ["OUT_exhaust", "OUT_prod", "OUT_RESTEAM"]
    got = resolve_gas_readout_nodes_from_highlight_list(yellow, nodes)
    assert got["target"] == "OUT_prod"
    assert got["tailgas"] == "OUT_exhaust"


def test_process7_only_prod_h2o():
    yellow = ["OUT_PROD", "OUT_H2O"]
    nodes = ["OUT_PROD", "OUT_H2O"]
    got = resolve_gas_readout_nodes_from_highlight_list(yellow, nodes)
    assert got["target"] == "OUT_PROD"
    assert got["tailgas"] == "OUT_H2O"


def test_empty_highlights():
    assert resolve_gas_readout_nodes_from_highlight_list([], ["A"]) == {}
    assert resolve_gas_readout_nodes_from_highlight_list(["X"], ["Y"]) == {}
