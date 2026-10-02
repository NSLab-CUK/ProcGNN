from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from process_graph.data.tabular_dataset import compute_y_edge_log1p_column_scaler
from process_graph.experiment.pi_mass_flow import (
    assert_checkpoint_pi_mass_flow_compatible,
    decode_pi_mass_flow_prediction,
    inverse_transform_mass_flow,
    transform_mass_flow,
)
from process_graph.experiment.edge_step_pi_training import (
    compute_mass_flow_physical_auxiliary_per_sample,
    get_mass_physical_weight,
    mass_dual_space_gradient_diagnostics,
    resolve_mass_flow_physical_scale,
)


@pytest.mark.parametrize("tau", [0.01, 0.05, 0.1])
def test_tempered_log_round_trip_recovers_physical_mass_flow(tau):
    cfg = SimpleNamespace(
        mass_flow_transform="tempered_log",
        mass_flow_log_tau=tau,
    )
    mass = torch.tensor([0.0, 1.0, 10.0, 1.0e3, 1.0e6], dtype=torch.float32)

    transformed = transform_mass_flow(mass, train_cfg=cfg)
    restored = inverse_transform_mass_flow(transformed, train_cfg=cfg)

    assert torch.allclose(restored, mass, rtol=2.0e-5, atol=1.0e-5)


def test_smaller_tau_compresses_large_mass_flow_less_than_log1p():
    mass = torch.tensor([1.0e5], dtype=torch.float32)
    log1p = transform_mass_flow(
        mass,
        train_cfg=SimpleNamespace(
            mass_flow_transform="log1p",
            mass_flow_log_tau=1.0,
        ),
    )
    tau01 = transform_mass_flow(
        mass,
        train_cfg=SimpleNamespace(
            mass_flow_transform="tempered_log",
            mass_flow_log_tau=0.1,
        ),
    )
    tau005 = transform_mass_flow(
        mass,
        train_cfg=SimpleNamespace(
            mass_flow_transform="tempered_log",
            mass_flow_log_tau=0.05,
        ),
    )
    tau001 = transform_mass_flow(
        mass,
        train_cfg=SimpleNamespace(
            mass_flow_transform="tempered_log",
            mass_flow_log_tau=0.01,
        ),
    )

    assert tau001.item() > tau005.item() > tau01.item() > log1p.item()


def test_log1p_transform_retains_legacy_formula():
    mass = torch.tensor([0.0, 9.0, 1000.0], dtype=torch.float32)
    cfg = SimpleNamespace(mass_flow_transform="log1p", mass_flow_log_tau=0.01)

    assert torch.equal(transform_mass_flow(mass, train_cfg=cfg), torch.log1p(mass))


def test_scaled_log_round_trip_recovers_physical_mass_flow():
    cfg = SimpleNamespace(
        mass_flow_transform="scaled_log",
        mass_flow_log_scale=2.0,
        mass_flow_log_eps=1.0e-8,
    )
    mass = torch.tensor([0.0, 1.0, 100.0, 1.0e6], dtype=torch.float32)

    restored = inverse_transform_mass_flow(
        transform_mass_flow(mass, train_cfg=cfg),
        train_cfg=cfg,
    )

    assert torch.allclose(restored, mass, rtol=2.0e-5, atol=1.0e-5)


def _physical_aux_cfg(*, enabled=True, scale_method="std"):
    return SimpleNamespace(
        mass_flow_physical_auxiliary=SimpleNamespace(
            enabled=enabled,
            weight=0.10,
            loss="huber",
            delta=1.0,
            residual_clip=10.0,
            scale_method=scale_method,
            minimum_scale=1.0e-8,
            epsilon=1.0e-8,
        )
    )


def test_mass_flow_physical_auxiliary_is_zero_at_exact_prediction_and_has_gradient():
    pred = torch.tensor([[100.0], [130.0]], requires_grad=True)
    target = torch.tensor([[100.0], [100.0]])
    normalizer = {"columns": ["Mass_Flow"], "std": torch.tensor([100.0])}

    loss, valid, _ = compute_mass_flow_physical_auxiliary_per_sample(
        pred,
        target,
        torch.ones_like(target),
        train_cfg=_physical_aux_cfg(),
        normalizer=normalizer,
    )
    loss.sum().backward()

    assert valid.tolist() == [True, True]
    assert loss[0].item() == pytest.approx(0.0)
    assert loss[1].item() > 0.0
    assert pred.grad is not None
    assert pred.grad[1].item() > 0.0


def test_mass_flow_physical_auxiliary_penalizes_larger_tail_underprediction_more():
    normalizer = {"columns": ["Mass_Flow"], "std": torch.tensor([100.0])}
    loss, _, _ = compute_mass_flow_physical_auxiliary_per_sample(
        torch.tensor([[90.0], [900.0]]),
        torch.tensor([[100.0], [1000.0]]),
        torch.ones(2, 1),
        train_cfg=_physical_aux_cfg(),
        normalizer=normalizer,
    )

    assert loss[1].item() > loss[0].item()


def test_mass_flow_physical_auxiliary_disabled_is_exact_zero():
    pred = torch.tensor([[1.0]], requires_grad=True)
    loss, valid, debug = compute_mass_flow_physical_auxiliary_per_sample(
        pred,
        torch.tensor([[1000.0]]),
        torch.ones(1, 1),
        train_cfg=_physical_aux_cfg(enabled=False),
        normalizer=None,
    )

    assert loss.item() == 0.0
    assert not valid.any()
    assert debug["mass_flow_physical_aux_enabled"] is False


def test_mass_flow_physical_auxiliary_uses_only_supplied_train_scale():
    pred = torch.tensor([[200.0]])
    target = torch.tensor([[100.0]])
    loss_small_scale, _, _ = compute_mass_flow_physical_auxiliary_per_sample(
        pred,
        target,
        torch.ones_like(target),
        train_cfg=_physical_aux_cfg(),
        normalizer={"columns": ["Mass_Flow"], "std": torch.tensor([50.0])},
    )
    loss_large_scale, _, _ = compute_mass_flow_physical_auxiliary_per_sample(
        pred,
        target,
        torch.ones_like(target),
        train_cfg=_physical_aux_cfg(),
        normalizer={"columns": ["Mass_Flow"], "std": torch.tensor([500.0])},
    )

    assert loss_small_scale.item() > loss_large_scale.item()


def test_mass_flow_physical_weight_uses_one_based_linear_warmup():
    weights = [
        get_mass_physical_weight(
            epoch=epoch,
            start_epoch=3,
            end_epoch=7,
            max_weight=0.10,
        )
        for epoch in (1, 3, 5, 7, 9)
    ]

    assert weights == pytest.approx([0.0, 0.0, 0.05, 0.10, 0.10])


def test_mass_flow_physical_scale_supports_train_iqr_and_fixed():
    iqr_cfg = _physical_aux_cfg(scale_method="train_iqr")
    iqr_scale, iqr_source = resolve_mass_flow_physical_scale(
        train_cfg=iqr_cfg,
        normalizer={
            "columns": ["Mass_Flow"],
            "std": torch.tensor([999.0]),
            "mass_flow_physical_p25": 100.0,
            "mass_flow_physical_p75": 450.0,
        },
    )
    fixed_cfg = _physical_aux_cfg(scale_method="fixed")
    fixed_cfg.mass_flow_physical_auxiliary.fixed_scale = 1234.0
    fixed_scale, fixed_source = resolve_mass_flow_physical_scale(
        train_cfg=fixed_cfg,
        normalizer={"columns": ["Mass_Flow"], "std": torch.tensor([999.0])},
    )

    assert iqr_scale == pytest.approx(350.0)
    assert iqr_source == "train_iqr"
    assert fixed_scale == pytest.approx(1234.0)
    assert fixed_source == "fixed"


def test_mass_flow_physical_loss_backpropagates_through_canonical_inverse():
    cfg = SimpleNamespace(
        pi_mass_flow_output_space="log1p",
        mass_flow_transform="scaled_log",
        mass_flow_log_scale=2.0,
        mass_flow_log_eps=1.0e-8,
        mass_flow_physical_auxiliary=SimpleNamespace(
            enabled=True,
            weight=0.10,
            loss="huber",
            delta=1.0,
            residual_clip=10.0,
            scale_method="train_std",
            minimum_scale=1.0e-8,
            epsilon=1.0e-8,
            current_epoch=7,
            schedule_type="linear_warmup",
            schedule_start_epoch=3,
            schedule_end_epoch=7,
            schedule_start_weight=0.0,
            schedule_end_weight=0.10,
        ),
    )
    true_physical = torch.tensor([[1000.0]])
    pred_transformed = transform_mass_flow(
        torch.tensor([[800.0]]),
        train_cfg=cfg,
    ).detach().requires_grad_(True)
    pred_physical = inverse_transform_mass_flow(pred_transformed, train_cfg=cfg)
    loss, valid, debug = compute_mass_flow_physical_auxiliary_per_sample(
        pred_physical,
        true_physical,
        torch.ones_like(true_physical),
        train_cfg=cfg,
        normalizer={"columns": ["Mass_Flow"], "std": torch.tensor([500.0])},
    )
    (debug["mass_flow_physical_aux_weight"] * loss[valid].mean()).backward()

    assert pred_transformed.grad is not None
    assert torch.isfinite(pred_transformed.grad).all()
    assert pred_transformed.grad.abs().sum().item() > 0.0


def test_mass_dual_gradient_diagnostics_does_not_write_parameter_grads():
    class TinyMassModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Module()
            self.encoder.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
            self.edge_decoder = torch.nn.Module()
            self.edge_decoder.hierarchical_pi_head = torch.nn.Module()
            self.edge_decoder.hierarchical_pi_head.mass_head = torch.nn.Linear(2, 1)

        def forward(self, x):
            x = self.encoder.layers[0](x)
            return self.edge_decoder.hierarchical_pi_head.mass_head(x)

    model = TinyMassModel()
    pred = model(torch.tensor([[1.0, -0.5]]))
    log_objective = pred.square().mean()
    physical_objective = (pred - 1.0).square().mean()
    diagnostics = mass_dual_space_gradient_diagnostics(
        model=model,
        mass_log_objective=log_objective,
        mass_physical_objective=physical_objective,
    )

    assert diagnostics["mass_grad_log_total"] > 0.0
    assert diagnostics["mass_grad_physical_total"] > 0.0
    assert "mass_grad_log_physical_cosine" in diagnostics
    assert all(parameter.grad is None for parameter in model.parameters())


def test_direct_tempered_metric_decode_returns_physical_mass_flow():
    expected = torch.tensor([0.0, 15.0, 250000.0], dtype=torch.float32)
    train_cfg = SimpleNamespace(
        pi_mass_flow_output_space="log1p",
        use_log1p_mass_flow_loss=False,
        mass_flow_transform="tempered_log",
        mass_flow_log_tau=0.05,
    )
    encoded = transform_mass_flow(expected, train_cfg=train_cfg)

    decoded = decode_pi_mass_flow_prediction(
        encoded,
        train_cfg=train_cfg,
        data_cfg=SimpleNamespace(normalize_y_edge=True),
        normalizer=None,
    )

    assert torch.allclose(decoded, expected, rtol=2.0e-5, atol=1.0e-5)


def test_checkpoint_rejects_different_tempered_tau():
    train_cfg = SimpleNamespace(
        pi_mass_flow_output_space="log1p",
        use_log1p_mass_flow_loss=False,
        mass_flow_transform="tempered_log",
        mass_flow_log_tau=0.05,
    )

    with pytest.raises(RuntimeError, match="Refusing to reinterpret"):
        assert_checkpoint_pi_mass_flow_compatible(
            {
                "pi_mass_flow_output_space": "log1p",
                "mass_flow_transform": "tempered_log",
                "mass_flow_log_tau": 0.1,
            },
            train_cfg=train_cfg,
            checkpoint_path="tau01.pt",
        )


def test_tempered_log_scaler_fits_transformed_train_values():
    columns = ["Temp", "Pres", "Mass_Flow"]
    records = [
        SimpleNamespace(
            graph=SimpleNamespace(
                edge_target_columns=columns,
                y_edge_true=[[0.0, 0.0, value]],
                y_edge_mask=[1.0],
                process_id=f"train_{index}",
            )
        )
        for index, value in enumerate((0.0, 99.0))
    ]
    fit = compute_y_edge_log1p_column_scaler(
        records,
        column_name="Mass_Flow",
        stream_target_dim=len(columns),
        mass_flow_transform="tempered_log",
        mass_flow_log_tau=0.1,
    )
    expected = torch.log1p(0.1 * torch.tensor([0.0, 99.0])) / 0.1

    assert fit.count == 2
    assert fit.mean.item() == pytest.approx(float(expected.mean()))
    assert fit.std.item() == pytest.approx(float(expected.std(unbiased=False)))
