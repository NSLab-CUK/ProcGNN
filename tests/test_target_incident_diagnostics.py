from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from process_graph.experiment.target_row_loss import (
    build_target_incident_diagnostic_rows,
    compute_target_incident_edge_loss,
    resolve_all_edge_r2_config,
)
from process_graph.experiment.target_row_spec import TargetRowSpec
from process_graph.experiment.train_utils import compute_edge_all_training_loss


@dataclass
class _Export:
    process_id: list[str]
    main_data_stream_key: list[str]
    canonical_edge_id: list[str]
    src_node: list[str] = field(default_factory=list)
    dst_node: list[str] = field(default_factory=list)


def _spec() -> TargetRowSpec:
    return TargetRowSpec(
        process_id="Process99",
        process_num=99,
        target_id="P99_T001",
        target_feature="PROD_H2_Mole",
        target_stream="PROD",
        edge_id="E_TARGET",
        species="H2",
        frac_slot="Frac_H2",
        mole_flow_slot="Mole_Flow",
        formula="Mole_Flow*Frac_H2",
        scale=1.0,
    )


def _cfg(**kwargs):
    base = dict(
        use_edge_all_loss=True,
        use_target_feature_loss=True,
        use_target_row_frac_loss=True,
        use_target_row_amount_loss=True,
        lambda_target_feature=1.0,
        lambda_target_row_frac=1.0,
        lambda_target_row_amount=10.0,
        primary_frac_loss_weight=5.0,
        all_edge_r2_loss_weight=0.05,
        all_edge_r2_min_count=2,
        all_edge_r2_sst_threshold=1.0e-6,
        all_edge_r2_loss_cap=5.0,
        loss_type_edge_all="smooth_l1",
        target_feature_loss_scope="target_incident_edges",
        target_feature_loss_balancing="process_target_balanced",
        target_incident_exclude_virtual_nodes=True,
        target_incident_virtual_node_names=["V_INPUT", "V_OUTPUT"],
        target_incident_include_target_edge=True,
        target_incident_use_answer_edge_weight=False,
        answer_edge_weight=1.0,
        answer_edge_weight_h2=1.0,
        answer_edge_weight_co2=1.0,
        answer_edge_weight_h2o=1.0,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _case_tensors():
    export = _Export(
        process_id=["Process99", "Process99", "Process99", "Process99"],
        main_data_stream_key=["PROD", "OTHER_OUT", "FEED", "DETACHED"],
        canonical_edge_id=["E_TARGET", "E_OTHER_OUT", "E_FEED", "E_DETACHED"],
        src_node=["UNIT_A", "UNIT_B", "UNIT_C", "UNIT_D"],
        dst_node=["V_OUTPUT", "V_OUTPUT", "UNIT_A", "UNIT_E"],
    )
    edge_index = torch.tensor([[0, 2, 3, 5], [1, 1, 0, 6]], dtype=torch.long)
    y_true = torch.zeros(4, 3)
    y_pred = torch.ones(4, 3, requires_grad=True)
    y_mask = torch.ones(4, 3)
    return export, edge_index, y_pred, y_true, y_mask


def test_virtual_output_fanout_excluded_and_target_edge_kept(monkeypatch):
    import process_graph.experiment.target_row_loss as trl

    export, edge_index, y_pred, y_true, y_mask = _case_tensors()
    monkeypatch.setattr(trl, "_TARGET_ROW_SPECS_CACHE", [_spec()])
    loss, info = compute_target_incident_edge_loss(
        y_pred=y_pred,
        y_true=y_true,
        y_mask=y_mask,
        export_meta=export,
        edge_index=edge_index,
        train_cfg=_cfg(),
    )
    loss.backward()

    assert info["target_incident_edge_count"] == 2
    assert info["target_incident_virtual_endpoint_excluded_count"] == 1
    assert float(y_pred.grad[0].abs().sum()) > 0.0
    assert float(y_pred.grad[1].abs().sum()) == 0.0
    assert float(y_pred.grad[2].abs().sum()) > 0.0


def test_exclude_virtual_false_uses_original_endpoint_incident_scope(monkeypatch):
    import process_graph.experiment.target_row_loss as trl

    export, edge_index, y_pred, y_true, y_mask = _case_tensors()
    monkeypatch.setattr(trl, "_TARGET_ROW_SPECS_CACHE", [_spec()])
    loss, info = compute_target_incident_edge_loss(
        y_pred=y_pred,
        y_true=y_true,
        y_mask=y_mask,
        export_meta=export,
        edge_index=edge_index,
        train_cfg=_cfg(target_incident_exclude_virtual_nodes=False),
    )
    loss.backward()

    assert info["target_incident_edge_count"] == 3
    assert float(y_pred.grad[1].abs().sum()) > 0.0


def test_diagnostics_effective_weight_and_loss_keys(monkeypatch):
    import process_graph.experiment.target_row_loss as trl

    export, edge_index, y_pred, y_true, y_mask = _case_tensors()
    monkeypatch.setattr(trl, "_TARGET_ROW_SPECS_CACHE", [_spec()])
    cfg = _cfg(target_incident_use_answer_edge_weight=False, answer_edge_weight_h2=5.0)
    items = compute_edge_all_training_loss(
        {"y_edge_pred": y_pred},
        {"edge_stream": y_true},
        {"edge_stream": y_mask},
        cfg,
        data_cfg=SimpleNamespace(task_mode="edge_all"),
        edge_export_meta=export,
        edge_target_columns=["Mole_Flow", "Frac_H2", "Frac_CO2"],
        edge_index=edge_index,
    )
    rows = build_target_incident_diagnostic_rows(
        split="train",
        epoch=1,
        step=1,
        y_pred=y_pred,
        y_true=y_true,
        y_mask=y_mask,
        export_meta=export,
        edge_index=edge_index,
        train_cfg=cfg,
        loss_scalars=items,
    )

    assert "loss_target_incident_edges" in items
    assert "weighted_target_incident_edges" in items
    assert "loss_primary_frac" in items
    assert rows[0]["internal_weight"] == 1.0
    assert rows[0]["effective_weight"] == 5.0
    assert rows[0]["dst_is_virtual"] is True
    assert rows[0]["num_edges_removed_by_virtual_filter"] == 1
    assert "E_TARGET" in rows[0]["included_edge_canonical_ids"]


def test_all_edge_r2_new_keys_override_and_legacy_fallback():
    cfg_new = SimpleNamespace(
        all_edge_r2_loss_weight=0.07,
        all_edge_r2_min_count=11,
        all_edge_r2_sst_threshold=2.0e-3,
        all_edge_r2_loss_cap=3.0,
        primary_frac_r2_loss_weight=0.99,
        primary_frac_r2_min_count=99,
    )
    resolved = resolve_all_edge_r2_config(cfg_new)
    assert resolved["source"] == "new_all_edge_r2_keys"
    assert resolved["loss_weight"] == 0.07
    assert resolved["min_count"] == 11

    cfg_old = SimpleNamespace(
        primary_frac_r2_loss_weight=0.02,
        primary_frac_r2_min_count=4,
        primary_frac_r2_sst_threshold=5.0e-4,
        primary_frac_r2_loss_cap=2.0,
    )
    fallback = resolve_all_edge_r2_config(cfg_old)
    assert fallback["source"] == "legacy_primary_frac_r2_keys"
    assert fallback["loss_weight"] == 0.02
    assert fallback["min_count"] == 4


def test_amount_auxiliary_does_not_contribute_to_total(monkeypatch):
    import process_graph.experiment.target_row_loss as trl

    export, edge_index, y_pred, y_true, y_mask = _case_tensors()
    monkeypatch.setattr(trl, "_TARGET_ROW_SPECS_CACHE", [_spec()])

    def fake_amount_loss(**kwargs):
        return kwargs["y_pred"].sum() * 0.0 + 99.0, {"n_terms": 1}

    monkeypatch.setattr(trl, "compute_v4_target_row_amount_loss", fake_amount_loss)
    cfg = _cfg(use_target_feature_loss=False, primary_frac_loss_weight=0.0, all_edge_r2_loss_weight=0.0)
    items = compute_edge_all_training_loss(
        {"y_edge_pred": y_pred},
        {"edge_stream": y_true},
        {"edge_stream": y_mask},
        cfg,
        data_cfg=SimpleNamespace(task_mode="edge_all"),
        edge_export_meta=export,
        edge_target_columns=["Mole_Flow", "Frac_H2", "Frac_CO2"],
        edge_index=edge_index,
    )
    expected = items["weighted_edge_all"] + items["weighted_primary_frac"] + items["weighted_all_edge_r2"]
    assert torch.allclose(items["loss_total_final"], expected)
    assert float(items["loss_amount_auxiliary"]) == 99.0
    assert float(items["weighted_amount_auxiliary_disabled"]) == 990.0
    assert float(items["amount_auxiliary_contributes_to_total"]) == 0.0
