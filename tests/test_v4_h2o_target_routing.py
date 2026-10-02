"""Tests for v4 H2O target-row loss weighting (canonical edge + Frac_H2O slot)."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from process_graph.constants import STREAM_EDGE_FEATURE_SLOTS
from process_graph.experiment.v4_target_edge_weighting import build_v4_target_loss_weight_tensor


@dataclass
class _FakeExport:
    process_id: list[str]
    main_data_stream_key: list[str]
    canonical_edge_id: list[str]


def _train_cfg(*, use_v4: bool = True, h2o_w: float = 5.0, legacy_element: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        answer_edge_weight=5.0,
        answer_edge_weight_h2=5.0,
        answer_edge_weight_co2=5.0,
        answer_edge_weight_h2o=h2o_w,
        use_v4_target_edge_weighting=use_v4,
        v4_target_weighting_mode="target_row",
        use_v4_target_row_loss=not legacy_element,
        use_legacy_answer_weighting=legacy_element,
    )


def test_h2o_weight_on_edge_b_frac_h2o_only(tmp_path):
    """Mock batch: H2/CO2 on edge A, H2O on edge B — H2O weight only on B / Frac_H2O."""
    v4_csv = tmp_path / "targets.csv"
    v4_csv.write_text(
        "process_id,target_id,target_species,target_feature_name,canonical_answer_edge_id,"
        "main_data_stream_key,required_stream_key,required_properties,scale_factor,formula_type\n"
        '99,P99_T001,H2,PROD_H2_Mole,E_EDGE_A,PROD,PROD,"[""Mole_Flow"", ""Frac_H2""]",1,derived\n'
        '99,P99_T002,CO2,PROD_CO2_Mole,E_EDGE_A,PROD,PROD,"[""Mole_Flow"", ""Frac_CO2""]",1,derived\n'
        '99,P99_T003,H2O,RE_H2O_Mole,E_EDGE_B,RE,RE,"[""Mole_Flow"", ""Frac_H2O""]",1,derived\n',
        encoding="utf-8",
    )
    export = _FakeExport(
        process_id=["Process99", "Process99"],
        main_data_stream_key=["PROD", "RE"],
        canonical_edge_id=["E_EDGE_A", "E_EDGE_B"],
    )
    w, _ = build_v4_target_loss_weight_tensor(
        export_meta=export,
        edge_target_columns=STREAM_EDGE_FEATURE_SLOTS,
        train_cfg=_train_cfg(legacy_element=True),
        v4_targets_path=v4_csv,
    )
    assert w is not None
    cols = list(STREAM_EDGE_FEATURE_SLOTS)
    h2o_i = cols.index("Frac_H2O")
    h2_i = cols.index("Frac_H2")
    co2_i = cols.index("Frac_CO2")
    # Edge A (PROD): H2 and CO2 weighted, not H2O
    assert float(w[0, h2_i]) >= 5.0
    assert float(w[0, co2_i]) >= 5.0
    assert float(w[0, h2o_i]) == 1.0
    # Edge B (RE): H2O weighted only on Frac_H2O
    assert float(w[1, h2o_i]) >= 5.0
    assert float(w[1, h2_i]) == 1.0
    assert float(w[1, co2_i]) == 1.0


def test_legacy_stream_mode_weights_mole_and_frac_on_stream(tmp_path):
    v4_csv = tmp_path / "targets.csv"
    v4_csv.write_text(
        "process_id,target_id,target_species,target_feature_name,canonical_answer_edge_id,"
        "main_data_stream_key,required_stream_key,required_properties,scale_factor,formula_type\n"
        '99,P99_T003,H2O,RE_H2O_Mole,E_EDGE_B,RE,RE,"[""Mole_Flow"", ""Frac_H2O""]",1,derived\n',
        encoding="utf-8",
    )
    export = _FakeExport(
        process_id=["Process99"],
        main_data_stream_key=["RE"],
        canonical_edge_id=["E_EDGE_B"],
    )
    cfg = _train_cfg(use_v4=False)
    w, _ = build_v4_target_loss_weight_tensor(
        export_meta=export,
        edge_target_columns=STREAM_EDGE_FEATURE_SLOTS,
        train_cfg=cfg,
        v4_targets_path=v4_csv,
    )
    assert w is not None
    cols = list(STREAM_EDGE_FEATURE_SLOTS)
    assert float(w[0, cols.index("Mole_Flow")]) >= 5.0
    assert float(w[0, cols.index("Frac_H2O")]) >= 5.0
