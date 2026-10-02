"""Tests for target-row Frac loss and metric policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from process_graph.experiment.metric_policy import (
    OFFICIAL_METRIC,
    build_loss_policy,
    build_metric_policy,
    resolve_loss_lambdas,
)
from process_graph.experiment.target_row_loss import compute_target_frac_feature_loss


@dataclass
class _Cfg:
    use_edge_all_loss: bool = True
    use_target_feature_loss: bool = True
    use_target_row_frac_loss: bool = True
    use_target_row_amount_loss: bool = False
    lambda_target_feature: float = 1.0
    lambda_target_row_frac: float = 1.0
    lambda_target_row_amount: float = 0.0
    loss_type_edge_all: str = "smooth_l1"
    use_v4_target_row_loss: bool = False


def test_metric_policy_official_metric():
    mp = build_metric_policy(_Cfg())
    assert mp["official_metric"] == OFFICIAL_METRIC
    assert mp["early_stop_resolved"] == f"val_{OFFICIAL_METRIC}"


def test_loss_policy_single_head_and_amount_disabled():
    lp = build_loss_policy(_Cfg())
    assert lp["prediction_head"] == "single_edge_all_head"
    assert lp["amount_loss_enabled"] is False
    assert lp["lambda_target_feature"] == 1.0
    assert lp["lambda_target_row_amount"] == 0.0


def test_resolve_loss_lambdas_returns_five_tuple():
    use_edge, use_feat, use_amount, lam_f, lam_a = resolve_loss_lambdas(_Cfg())
    assert use_edge is True
    assert use_feat is True
    assert use_amount is False
    assert lam_f == 1.0
    assert lam_a == 0.0


def test_metrics_per_epoch_row_required_columns():
    from process_graph.experiment.metric_policy import (
        EARLY_STOP_METRIC,
        build_edge_all_metrics_per_epoch_row,
    )
    from process_graph.experiment.target_metric_names import (
        EVAL_LEGACY_ANSWER_FRAC_R2,
        EVAL_PRIMARY_FRAC_R2_BY_PROCESS,
        EVAL_PRIMARY_FRAC_R2_BY_TARGET,
        EVAL_SECONDARY_AMOUNT_R2_BY_PROCESS,
    )

    row = build_edge_all_metrics_per_epoch_row(
        epoch=1,
        train_loss_total=10.0,
        lr=1e-4,
        train_components={
            "loss_edge_all": 1.0,
            "loss_target_frac_feature": 2.0,
            "loss_target_row_amount": 0.0,
            "edge_step_target_edge_count": 3.0,
            "target_edge_weight_mean": 5.0,
            "default_edge_weight_mean": 1.0,
            "epoch_time_sec": 12.5,
            "peak_gpu_memory_mb": 456.0,
            "peak_gpu_reserved_mb": 768.0,
        },
        val_metrics={
            "loss_total": 9.0,
            "loss_edge_all": 1.0,
            "loss_target_frac_feature": 2.0,
            "loss_target_row_amount": 0.0,
            EVAL_PRIMARY_FRAC_R2_BY_PROCESS: 0.42,
            EVAL_PRIMARY_FRAC_R2_BY_TARGET: 0.40,
            EVAL_SECONDARY_AMOUNT_R2_BY_PROCESS: 0.30,
            EVAL_LEGACY_ANSWER_FRAC_R2: 0.25,
            "metric_target_r2": 0.2,
            "target_r2": 0.62,
            "target_mean_r2": 0.61,
            "target_stream_feature_macro_r2": 0.11,
            "edge_all_mae": 0.1,
        },
    )
    for col in (
        "train_loss_edge_all",
        "train_loss_target_frac_feature",
        "train_loss_target_row_amount",
        "val_loss_edge_all",
        "val_loss_target_frac_feature",
        f"val_{EVAL_PRIMARY_FRAC_R2_BY_PROCESS}",
        "legacy_metric_target_r2",
        "val_target_r2",
        "val_target_mean_r2",
        "train_edge_step_target_edge_count",
        "train_target_edge_weight_mean",
        "train_default_edge_weight_mean",
        "diag_edge_all_mae",
        "train_epoch_time_sec",
        "train_peak_gpu_memory_mb",
        "train_peak_gpu_reserved_mb",
    ):
        assert col in row, col
    assert row["val_target_r2"] == 0.62
    assert row["val_target_mean_r2"] == 0.61
    assert row["train_edge_step_target_edge_count"] == 3.0
    assert row["train_target_edge_weight_mean"] == 5.0
    assert row["train_default_edge_weight_mean"] == 1.0
    assert row["train_epoch_time_sec"] == 12.5
    assert row["train_peak_gpu_memory_mb"] == 456.0
    assert row["train_peak_gpu_reserved_mb"] == 768.0
    assert "val_target_stream_feature_macro_r2" not in row
    assert f"val_{EVAL_PRIMARY_FRAC_R2_BY_TARGET}" not in row
    assert EARLY_STOP_METRIC == "val_target_edge_property_mean_r2"


def test_target_frac_feature_loss_zero_without_meta():
    cfg = _Cfg()
    y = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    loss, info = compute_target_frac_feature_loss(
        y_pred=y,
        y_true=y,
        y_mask=torch.ones(1),
        export_meta=None,
        edge_target_columns=["Mole_Flow", "Frac_H2", "Frac_CO2"],
        train_cfg=cfg,
    )
    assert float(loss.detach()) == 0.0
    assert info["n_terms"] == 0


def test_edge_all_total_excludes_amount_auxiliary(monkeypatch):
    from process_graph.data.tabular_dataset import EdgeBatchExportMeta
    from process_graph.experiment import target_row_loss
    from process_graph.experiment.train_utils import compute_edge_all_training_loss

    cfg = _Cfg(
        use_target_feature_loss=False,
        use_target_row_frac_loss=False,
        use_target_row_amount_loss=True,
        lambda_target_feature=0.0,
        lambda_target_row_frac=0.0,
        lambda_target_row_amount=10.0,
    )
    cfg.primary_frac_loss_weight = 0.0
    cfg.primary_frac_r2_loss_weight = 0.0

    def fake_amount_loss(**kwargs):
        ref = kwargs["y_pred"]
        return ref.sum() * 0.0 + 99.0, {"n_terms": 1}

    monkeypatch.setattr(target_row_loss, "compute_v4_target_row_amount_loss", fake_amount_loss)

    y_pred = torch.zeros(2, 3)
    y_true = torch.ones(2, 3)
    y_mask = torch.ones(2)
    meta = EdgeBatchExportMeta(
        process_id=["Process1", "Process1"],
        sample_id=["S0", "S1"],
        dataset_split=["train", "train"],
        canonical_edge_id=["P01_E001", "P01_E001"],
        main_data_stream_key=["A", "A"],
        stream_role=["product", "product"],
        is_input_edge=[0.0, 0.0],
        is_output_edge=[1.0, 1.0],
        is_internal_edge=[0.0, 0.0],
        answer_task_names=["target", "target"],
    )

    items = compute_edge_all_training_loss(
        {"y_edge_pred": y_pred},
        {"edge_stream": y_true},
        {"edge_stream": y_mask},
        cfg,
        data_cfg=object(),
        edge_export_meta=meta,
        edge_target_columns=["Mole_Flow", "Frac_H2", "Frac_CO2"],
    )

    expected = (
        items["weighted_edge_all"]
        + items["weighted_primary_frac"]
        + items["weighted_all_edge_r2"]
    )
    assert torch.allclose(items["loss_total"], expected)
    assert torch.allclose(items["loss_total_final"], expected)
    assert float(items["loss_amount_auxiliary"]) == 99.0
    assert float(items["weighted_amount_auxiliary_disabled"]) == 990.0
    assert float(items["loss_extra_unexpected"]) == 0.0

