from __future__ import annotations

import torch

from process_graph.data.tabular_dataset import EdgeBatchExportMeta
from process_graph.experiment.target_row_loss import (
    build_primary_frac_mask,
    compute_all_edge_r2_loss,
    compute_primary_frac_losses,
)


class _Cfg:
    loss_type_edge_all = "smooth_l1"
    primary_frac_r2_min_count = 8
    primary_frac_r2_sst_threshold = 1.0e-4


def _meta(n: int) -> EdgeBatchExportMeta:
    return EdgeBatchExportMeta(
        process_id=["Process1"] * n,
        sample_id=[f"S{i}" for i in range(n)],
        dataset_split=["train"] * n,
        canonical_edge_id=["P01_E021"] * n,
        main_data_stream_key=["PROD"] * n,
        stream_role=["product"] * n,
        is_input_edge=[0.0] * n,
        is_output_edge=[1.0] * n,
        is_internal_edge=[0.0] * n,
        answer_task_names=["target"] * n,
    )


def _meta_numeric_process_id(n: int) -> EdgeBatchExportMeta:
    meta = _meta(n)
    meta.process_id[:] = ["1"] * n
    return meta


def test_primary_frac_mask_enables_h2_and_co2_on_same_answer_edge() -> None:
    cols = [
        "Frac_CH4",
        "Frac_CO",
        "Frac_CO2",
        "Frac_H2",
        "Frac_H2O",
        "Frac_N2",
        "Frac_O2",
        "Mass_Flow",
        "Mole_Flow",
        "Pres",
        "Temp",
        "Vol_Flow",
    ]
    y_pred = torch.zeros(8, 12)
    y_mask = torch.ones(8, 12)

    mask, info = build_primary_frac_mask(
        y_pred=y_pred,
        y_mask=y_mask,
        export_meta=_meta(8),
        edge_target_columns=cols,
        train_cfg=_Cfg(),
    )

    assert info["primary_frac_count"] == 16
    assert torch.all(mask[:, 3] == 1.0)  # P01_T002 H2 -> Frac_H2
    assert torch.all(mask[:, 2] == 1.0)  # P01_T003 CO2 -> Frac_CO2
    assert torch.all(mask[:, 4] == 0.0)  # P01_E021 has no H2O target row


def test_primary_frac_mask_accepts_numeric_process_ids() -> None:
    cols = [
        "Frac_CH4",
        "Frac_CO",
        "Frac_CO2",
        "Frac_H2",
        "Frac_H2O",
        "Frac_N2",
        "Frac_O2",
        "Mass_Flow",
        "Mole_Flow",
        "Pres",
        "Temp",
        "Vol_Flow",
    ]
    y_pred = torch.zeros(8, 12)
    y_mask = torch.ones(8, 12)

    mask, info = build_primary_frac_mask(
        y_pred=y_pred,
        y_mask=y_mask,
        export_meta=_meta_numeric_process_id(8),
        edge_target_columns=cols,
        train_cfg=_Cfg(),
    )

    assert info["primary_frac_count"] == 16
    assert torch.all(mask[:, 3] == 1.0)
    assert torch.all(mask[:, 2] == 1.0)


def test_primary_frac_losses_are_finite_and_backward_flows() -> None:
    cols = [
        "Frac_CH4",
        "Frac_CO",
        "Frac_CO2",
        "Frac_H2",
        "Frac_H2O",
        "Frac_N2",
        "Frac_O2",
        "Mass_Flow",
        "Mole_Flow",
        "Pres",
        "Temp",
        "Vol_Flow",
    ]
    y_true = torch.zeros(8, 12)
    y_true[:, 2] = torch.linspace(-1.0, 1.0, 8)
    y_true[:, 3] = torch.linspace(1.0, -1.0, 8)
    y_pred = (y_true + 0.1).clone().detach().requires_grad_(True)
    y_mask = torch.ones(8, 12)

    loss_frac, loss_r2, info = compute_primary_frac_losses(
        y_pred=y_pred,
        y_true=y_true,
        y_mask=y_mask,
        export_meta=_meta(8),
        edge_target_columns=cols,
        train_cfg=_Cfg(),
        y_edge_mean=torch.zeros(12),
        y_edge_std=torch.ones(12),
    )
    loss = loss_frac + loss_r2
    loss.backward()

    assert torch.isfinite(loss_frac)
    assert torch.isfinite(loss_r2)
    assert info["primary_frac_count"] == 16
    assert info["primary_frac_r2_valid_feature_count"] == 2
    assert y_pred.grad is not None
    assert float(y_pred.grad[:, 2].abs().sum() + y_pred.grad[:, 3].abs().sum()) > 0.0


def test_all_edge_r2_loss_uses_supervised_edge_mask_not_target_subset() -> None:
    y_true = torch.zeros(8, 4)
    y_true[:, 0] = torch.linspace(-1.0, 1.0, 8)
    y_true[:, 1] = torch.linspace(1.0, -1.0, 8)
    y_true[:, 2] = torch.linspace(-0.5, 0.5, 8)
    y_pred = (y_true + 0.1).clone().detach().requires_grad_(True)
    y_mask = torch.ones(8, 4)
    y_mask[:, 3] = 0.0

    loss_r2, info = compute_all_edge_r2_loss(
        y_pred=y_pred,
        y_true=y_true,
        y_mask=y_mask,
        train_cfg=_Cfg(),
    )
    loss_r2.backward()

    assert torch.isfinite(loss_r2)
    assert info["r2_loss_scope"] == "edge_all"
    assert info["r2_loss_num_elements"] == 24
    assert info["edge_all_loss_num_elements"] == 24
    assert info["r2_loss_valid_feature_count"] == 3
    assert y_pred.grad is not None
    assert float(y_pred.grad[:, 0].abs().sum()) > 0.0
    assert float(y_pred.grad[:, 1].abs().sum()) > 0.0
    assert float(y_pred.grad[:, 2].abs().sum()) > 0.0
    assert float(y_pred.grad[:, 3].abs().sum()) == 0.0
