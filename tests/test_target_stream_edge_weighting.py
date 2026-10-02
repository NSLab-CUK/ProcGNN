from __future__ import annotations

from types import SimpleNamespace

import torch

from process_graph.experiment.target_stream_weighting import build_target_stream_loss_weight_tensor
from process_graph.experiment.train_utils import compute_edge_all_training_loss


class Meta(SimpleNamespace):
    pass


def _meta() -> Meta:
    return Meta(
        process_id=["Process6", "Process6", "Process7", "Process6"],
        sample_id=["s1", "s1", "s2", "s1"],
        dataset_split=["train"] * 4,
        canonical_edge_id=["P06_E022", "P06_E021", "P07_E016", "P06_E999"],
        main_data_stream_key=["PROD", "RE", "PROD", "PROD"],
        stream_role=["product"] * 4,
        is_input_edge=[0.0] * 4,
        is_output_edge=[1.0] * 4,
        is_internal_edge=[0.0] * 4,
        answer_task_names=[""] * 4,
    )


def _cfg(**kw):
    base = dict(
        use_target_stream_loss_weighting=True,
        target_stream_loss_weight=5.0,
        target_stream_weighting_mode="unique_target_stream_edge_by_process",
        target_stream_weight_conflict_policy="max",
        allow_yaml_only_target_streams=False,
        target_stream_loss_weights=[],
        loss_type_edge_all="smooth_l1",
        use_edge_all_loss=True,
        use_target_feature_loss=False,
        target_feature_loss_scope="target_stream_edge_weighted_edge_all",
        all_edge_r2_loss_weight=0.0,
        all_edge_r2_min_count=8,
        all_edge_r2_sst_threshold=1e-3,
        all_edge_r2_loss_cap=5.0,
        primary_frac_loss_weight=5.0,
        lambda_target_row_amount=0.0,
        use_target_row_amount_loss=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_target_stream_edge_weight_applies_to_all_features_once_and_same_process_only():
    cols = ["Temp", "Pres", "Vol_Flow", "Mole_Flow", "Mass_Flow", "Frac_H2O", "Frac_H2", "Frac_CH4", "Frac_CO2", "Frac_CO", "Frac_O2", "Frac_N2"]
    w, diags, summary = build_target_stream_loss_weight_tensor(
        export_meta=_meta(),
        edge_target_columns=cols,
        train_cfg=_cfg(),
    )
    assert w is not None
    assert torch.all(w[0] == 5.0)
    assert torch.all(w[1] == 5.0)
    assert torch.all(w[2] == 5.0)  # Process7 PROD is independently a target stream.
    assert torch.all(w[3] == 1.0)  # Same stream name in Process6 but wrong canonical edge.
    assert float(summary["num_target_stream_edges_weighted"]) == 3.0
    p7 = [d for d in diags if d.get("canonical_edge_id") == "P07_E016" and d.get("matched")]
    assert p7 and int(p7[0]["num_target_rows_for_same_stream"]) == 2


def test_yaml_override_conflict_policy_and_yaml_only_gate():
    cfg = _cfg(
        target_stream_loss_weights=[
            {"target_id": "P06_T001", "canonical_answer_edge_id": "P06_E022", "weight": 3.0},
            {"target_id": "P06_T001B", "canonical_answer_edge_id": "P06_E022", "weight": 7.0},
            {"target_id": "P06_X", "canonical_answer_edge_id": "P06_E999", "weight": 9.0},
        ]
    )
    w, diags, summary = build_target_stream_loss_weight_tensor(
        export_meta=_meta(),
        edge_target_columns=["Temp", "Pres"],
        train_cfg=cfg,
    )
    assert w is not None
    assert torch.all(w[0] == 7.0)
    assert torch.all(w[3] == 1.0)
    assert any(d.get("skip_reason") == "yaml_only_target_stream_not_allowed" for d in diags)
    assert float(summary["num_target_stream_edges_with_yaml_override"]) >= 1.0

    cfg.allow_yaml_only_target_streams = True
    w2, _, _ = build_target_stream_loss_weight_tensor(
        export_meta=_meta(),
        edge_target_columns=["Temp", "Pres"],
        train_cfg=cfg,
    )
    assert w2 is not None
    assert torch.all(w2[3] == 9.0)


def test_species_is_metadata_not_routing_and_fallback_is_same_process_only():
    cfg = _cfg(
        allow_yaml_only_target_streams=True,
        target_stream_loss_weights=[
            {"target_id": "P06_NEW", "fallback_stream_key": "PROD", "target_species": "Xe", "weight": 4.0},
        ],
    )
    w, _, _ = build_target_stream_loss_weight_tensor(
        export_meta=_meta(),
        edge_target_columns=["Temp", "Frac_H2", "Frac_CO2"],
        train_cfg=cfg,
    )
    assert w is not None
    assert torch.all(w[0] == 4.0)  # YAML override replaces the CSV default, even when lower.
    assert torch.all(w[2] == 5.0)  # Different process PROD is not affected by P06_NEW.


def test_loss_total_excludes_deprecated_primary_frac_auxiliary():
    y_pred = torch.zeros((2, 3), dtype=torch.float32)
    y_true = torch.ones((2, 3), dtype=torch.float32)
    mask = torch.ones((2, 3), dtype=torch.float32)
    meta = Meta(
        process_id=["Process6", "Process6"],
        canonical_edge_id=["P06_E022", "P06_E999"],
        main_data_stream_key=["PROD", "PROD"],
    )
    out = compute_edge_all_training_loss(
        {"y_edge_pred": y_pred},
        {"edge_stream": y_true},
        {"edge_stream": mask},
        _cfg(),
        SimpleNamespace(),
        edge_export_meta=meta,
        edge_target_columns=["Temp", "Pres", "Vol_Flow"],
    )
    assert float(out["weighted_primary_frac"]) == 0.0
    assert float(out["loss_primary_frac_contributes_to_total"]) == 0.0
    assert torch.allclose(out["loss_total_final"], out["weighted_edge_all"] + out["weighted_all_edge_r2"])
