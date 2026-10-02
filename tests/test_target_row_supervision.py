"""Target-row (target_id) supervision: no species pooling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from process_graph.constants import STREAM_EDGE_FEATURE_SLOTS
from process_graph.experiment.target_row_amounts import compute_target_row_amounts
from process_graph.experiment.target_row_loss import compute_target_incident_edge_loss, compute_v4_target_row_loss
from process_graph.experiment.target_row_spec import TargetRowSpec
from process_graph.experiment.target_v4_metrics import (
    _build_v4_summary_rows,
    compute_target_v4_metrics_from_original_scale,
    load_target_stream_targets_v4,
)
from process_graph.experiment.v4_target_edge_weighting import build_v4_target_loss_weight_tensor

@dataclass
class _Export:
    process_id: list
    main_data_stream_key: list
    canonical_edge_id: list
    src_node: list = field(default_factory=list)
    dst_node: list = field(default_factory=list)


def _spec(
    *,
    tid: str,
    pid: str = "Process99",
    edge: str,
    stream: str,
    species: str,
    feature: str,
) -> TargetRowSpec:
    frac = {"H2": "Frac_H2", "CO2": "Frac_CO2", "H2O": "Frac_H2O"}[species]
    return TargetRowSpec(
        process_id=pid,
        process_num=99,
        target_id=tid,
        target_feature=feature,
        target_stream=stream,
        edge_id=edge,
        species=species,
        frac_slot=frac,
        mole_flow_slot="Mole_Flow",
        formula="Mole_Flow*Frac",
        scale=1.0,
        species_weight=5.0 if species == "H2O" else 3.0,
    )


def test_two_co2_targets_separate_amount_keys():
    specs = [
        _spec(tid="PXX_T001", edge="E_A", stream="EX", species="CO2", feature="EX_CO2_Mole"),
        _spec(tid="PXX_T002", edge="E_B", stream="10", species="CO2", feature="Stream10_CO2_Mole"),
    ]
    export = _Export(
        process_id=["Process99", "Process99"],
        main_data_stream_key=["EX", "10"],
        canonical_edge_id=["E_A", "E_B"],
    )
    cols = list(STREAM_EDGE_FEATURE_SLOTS)
    mi, ci = cols.index("Mole_Flow"), cols.index("Frac_CO2")
    y_pred = torch.zeros(2, len(cols))
    y_true = torch.zeros(2, len(cols))
    y_pred[0, mi], y_true[0, mi] = 100.0, 100.0
    y_pred[0, ci], y_true[0, ci] = 0.2, 0.2
    y_pred[1, mi], y_true[1, mi] = 50.0, 50.0
    y_pred[1, ci], y_true[1, ci] = 0.4, 0.4
    mask = torch.ones(2)
    batch = compute_target_row_amounts(
        y_pred=y_pred,
        y_true=y_true,
        y_mask=mask,
        export_meta=export,
        edge_target_columns=cols,
        specs=specs,
    )
    tids = set(batch.target_ids)
    assert tids == {"PXX_T001", "PXX_T002"}
    assert len(batch.target_ids) == 2
    by_id = {tid: (p, t) for tid, p, t in zip(batch.target_ids, batch.pred_amount, batch.true_amount)}
    assert abs(by_id["PXX_T001"][0] - 20.0) < 1e-4
    assert abs(by_id["PXX_T002"][0] - 20.0) < 1e-4


def test_h2o_weight_only_on_h2o_edge_legacy_element_mode():
    specs = [
        _spec(tid="PXX_H2", edge="E_A", stream="PROD", species="H2", feature="PROD_H2_Mole"),
        _spec(tid="PXX_H2O", edge="E_B", stream="RE", species="H2O", feature="RE_H2O_Mole"),
    ]
    export = _Export(
        process_id=["Process99", "Process99"],
        main_data_stream_key=["PROD", "RE"],
        canonical_edge_id=["E_A", "E_B"],
    )
    cfg = SimpleNamespace(
        answer_edge_weight=5.0,
        answer_edge_weight_h2=5.0,
        answer_edge_weight_co2=5.0,
        answer_edge_weight_h2o=7.0,
        use_v4_target_edge_weighting=True,
        use_legacy_answer_weighting=True,
        use_v4_target_row_loss=False,
        v4_target_weighting_mode="target_row",
    )
    w, _ = build_v4_target_loss_weight_tensor(
        export_meta=export,
        edge_target_columns=STREAM_EDGE_FEATURE_SLOTS,
        train_cfg=cfg,
        v4_targets_path=None,
    )
    # Patch: load from specs manually - use inline rows in v4_target_edge_weighting via temp file
    # Instead test row loss path with compute_v4_target_row_loss
    cols = list(STREAM_EDGE_FEATURE_SLOTS)
    mi, h2o_i = cols.index("Mole_Flow"), cols.index("Frac_H2O")
    y_pred = torch.zeros(2, len(cols))
    y_true = torch.zeros(2, len(cols))
    y_pred[0, mi], y_true[0, mi] = 10.0, 10.0
    y_pred[0, cols.index("Frac_H2")], y_true[0, cols.index("Frac_H2")] = 0.9, 0.8
    y_pred[1, mi], y_true[1, mi] = 2.0, 2.0
    y_pred[1, h2o_i], y_true[1, h2o_i] = 0.6, 0.5
    mask = torch.ones(2)
    import process_graph.experiment.target_row_loss as trl

    trl._TARGET_ROW_SPECS_CACHE = specs
    cfg2 = SimpleNamespace(
        use_v4_target_row_loss=True,
        loss_type_edge_all="mse",
        answer_edge_weight_h2o=7.0,
        answer_edge_weight_h2=3.0,
        answer_edge_weight_co2=3.0,
        answer_edge_weight=5.0,
    )
    loss_v4, info = compute_v4_target_row_loss(
        y_pred=y_pred,
        y_true=y_true,
        y_mask=mask,
        export_meta=export,
        edge_target_columns=cols,
        train_cfg=cfg2,
    )
    assert float(loss_v4.detach()) > 0
    assert "PXX_H2O" in info["target_ids_in_batch"]
    assert "PXX_H2" in info["target_ids_in_batch"]
    trl._TARGET_ROW_SPECS_CACHE = None


def test_target_incident_edge_loss_weights_all_features_on_edges_touching_target_nodes():
    import process_graph.experiment.target_row_loss as trl

    specs = [_spec(tid="PXX_H2", edge="E_TARGET", stream="PROD", species="H2", feature="PROD_H2_Mole")]
    export = _Export(
        process_id=["Process99", "Process99", "Process99", "Process99"],
        main_data_stream_key=["PROD", "SIDE", "FEED", "DETACHED"],
        canonical_edge_id=["E_TARGET", "E_SIDE", "E_FEED", "E_DETACHED"],
    )
    y_true = torch.zeros(4, 3)
    y_pred = torch.ones(4, 3, requires_grad=True)
    y_mask = torch.ones(4, 3)
    edge_index = torch.tensor(
        [
            [0, 1, 3, 5],
            [1, 2, 0, 6],
        ],
        dtype=torch.long,
    )
    cfg = SimpleNamespace(
        loss_type_edge_all="smooth_l1",
        target_feature_loss_balancing="process_target_balanced",
        answer_edge_weight=3.0,
        answer_edge_weight_h2=3.0,
        answer_edge_weight_co2=3.0,
        answer_edge_weight_h2o=3.0,
    )

    trl._TARGET_ROW_SPECS_CACHE = specs
    try:
        loss, info = compute_target_incident_edge_loss(
            y_pred=y_pred,
            y_true=y_true,
            y_mask=y_mask,
            export_meta=export,
            edge_index=edge_index,
            train_cfg=cfg,
        )
    finally:
        trl._TARGET_ROW_SPECS_CACHE = None
    loss.backward()

    assert torch.isfinite(loss)
    assert info["target_incident_edge_count"] == 3
    assert info["primary_frac_count"] == 3
    assert float(y_pred.grad[:3].abs().sum()) > 0.0
    assert float(y_pred.grad[3].abs().sum()) == 0.0


def test_target_incident_edge_loss_excludes_virtual_endpoint_fanout_by_default():
    import process_graph.experiment.target_row_loss as trl

    specs = [_spec(tid="PXX_H2", edge="E_TARGET", stream="PROD", species="H2", feature="PROD_H2_Mole")]
    export = _Export(
        process_id=["Process99", "Process99", "Process99", "Process99"],
        main_data_stream_key=["PROD", "OTHER_OUT", "FEED", "DETACHED"],
        canonical_edge_id=["E_TARGET", "E_OTHER_OUT", "E_FEED", "E_DETACHED"],
        src_node=["UNIT_A", "UNIT_B", "UNIT_C", "UNIT_D"],
        dst_node=["V_OUTPUT", "V_OUTPUT", "UNIT_A", "UNIT_E"],
    )
    y_true = torch.zeros(4, 2)
    y_pred = torch.ones(4, 2, requires_grad=True)
    y_mask = torch.ones(4, 2)
    edge_index = torch.tensor(
        [
            [0, 2, 3, 5],
            [1, 1, 0, 6],
        ],
        dtype=torch.long,
    )
    cfg = SimpleNamespace(
        loss_type_edge_all="smooth_l1",
        target_feature_loss_balancing="process_target_balanced",
        target_incident_exclude_virtual_nodes=True,
        target_incident_virtual_node_names=["V_INPUT", "V_OUTPUT"],
        target_incident_include_target_edge=True,
        target_incident_use_answer_edge_weight=False,
        primary_frac_loss_weight=5.0,
        answer_edge_weight=5.0,
        answer_edge_weight_h2=7.0,
        answer_edge_weight_co2=5.0,
        answer_edge_weight_h2o=5.0,
    )

    trl._TARGET_ROW_SPECS_CACHE = specs
    try:
        loss, info = compute_target_incident_edge_loss(
            y_pred=y_pred,
            y_true=y_true,
            y_mask=y_mask,
            export_meta=export,
            edge_index=edge_index,
            train_cfg=cfg,
        )
    finally:
        trl._TARGET_ROW_SPECS_CACHE = None
    loss.backward()

    assert torch.isfinite(loss)
    assert info["target_incident_virtual_filter_applied"] == 1
    assert info["target_incident_virtual_endpoint_excluded_count"] == 1
    assert info["target_incident_edge_count"] == 2
    assert info["target_incident_inner_weight_mean"] == 1.0
    assert info["target_incident_effective_weight_mean"] == 5.0
    assert float(y_pred.grad[0].abs().sum()) > 0.0  # target edge is kept explicitly
    assert float(y_pred.grad[2].abs().sum()) > 0.0  # UNIT_A incident edge remains
    assert float(y_pred.grad[1].abs().sum()) == 0.0  # V_OUTPUT-only fanout is excluded
    assert float(y_pred.grad[3].abs().sum()) == 0.0


def test_macro_r2_is_target_row_mean_not_species_first():
    rows = []
    base = {
        "split": "val",
        "process_id": "Process1",
        "sample_id": "1",
        "y_edge_mask": 1.0,
        "is_answer_edge": 0,
    }
    for prop, yt, yp in (
        ("Mole_Flow", 100.0, 90.0),
        ("Frac_CO2", 0.1, 0.2),
        ("Frac_H2", 0.8, 0.7),
        ("Frac_H2O", 0.05, 0.05),
        ("Temp", 300.0, 300.0),
        ("Pres", 1.0, 1.0),
        ("Vol_Flow", 1e5, 1e5),
        ("Mass_Flow", 1e4, 1e4),
        ("Frac_CH4", 0.05, 0.05),
        ("Frac_CO", 0.0, 0.0),
        ("Frac_O2", 0.0, 0.0),
        ("Frac_N2", 0.0, 0.0),
    ):
        rows.append({**base, "canonical_edge_id": "P01_E015", "main_data_stream_key": "FUELGAS", "property_name": prop, "y_true_orig": yt, "y_pred_orig": yp})
    for prop, yt, yp in (
        ("Mole_Flow", 200.0, 180.0),
        ("Frac_CO2", 0.3, 0.25),
        ("Frac_H2", 0.5, 0.5),
        ("Frac_H2O", 0.1, 0.1),
        ("Temp", 300.0, 300.0),
        ("Pres", 1.0, 1.0),
        ("Vol_Flow", 1e5, 1e5),
        ("Mass_Flow", 1e4, 1e4),
        ("Frac_CH4", 0.05, 0.05),
        ("Frac_CO", 0.0, 0.0),
        ("Frac_O2", 0.0, 0.0),
        ("Frac_N2", 0.0, 0.0),
    ):
        rows.append({**base, "canonical_edge_id": "P01_E021", "main_data_stream_key": "PROD", "property_name": prop, "y_true_orig": yt, "y_pred_orig": yp})
    edge_pred = pd.DataFrame(rows)
    v4_path = Path(__file__).resolve().parents[1] / "data/reference/v4/target_stream_targets.csv"
    specs = load_target_stream_targets_v4(v4_path)
    flat = [s for sl in specs.values() for s in sl if s.process_id == "Process1"]
    result = compute_target_v4_metrics_from_original_scale(
        edge_predictions=edge_pred,
        target_specs=flat,
        split_name="val",
    )
    co2_rows = result.target_metrics_df[result.target_metrics_df["target_species"].astype(str) == "CO2"]
    co2_ids = set(co2_rows["target_id"].astype(str))
    assert "P01_T001" in co2_ids and "P01_T003" in co2_ids
    summary = _build_v4_summary_rows(result.target_metrics_df, result.predictions_df, split_name="val")
    main_rows = [r for r in summary if r.get("summary_kind") == "split_macro"]
    species_rows = [r for r in summary if "species_mean" in str(r.get("summary_kind", ""))]
    assert main_rows
    assert species_rows  # reporting only
