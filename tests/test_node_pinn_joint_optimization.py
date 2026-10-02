from __future__ import annotations

import pytest
import torch
from pathlib import Path

from process_graph.experiment.edge_step_pi_training import compose_joint_node_pinn_loss
from process_graph.experiment.loaders import load_experiment_config


def test_joint_objective_uses_scheduled_node_loss_once() -> None:
    anchor = torch.tensor(2.0, requires_grad=True)
    raw_node = torch.tensor(4.0, requires_grad=True)
    schedule = 0.5

    loss = compose_joint_node_pinn_loss(
        anchor,
        schedule * raw_node,
        supervised_anchor_weight=1.5,
        node_outer_weight=0.25,
    )

    assert loss.item() == pytest.approx(1.5 * 2.0 + 0.25 * 0.5 * 4.0)
    loss.backward()
    assert anchor.grad.item() == pytest.approx(1.5)
    assert raw_node.grad.item() == pytest.approx(0.25 * 0.5)


def test_joint_objective_preserves_supervised_gradient() -> None:
    parameter = torch.tensor(3.0, requires_grad=True)
    anchor = (parameter - 1.0).square()
    node = (parameter + 2.0).square()
    loss = compose_joint_node_pinn_loss(
        anchor,
        node,
        supervised_anchor_weight=1.0,
        node_outer_weight=0.05,
    )
    loss.backward()
    assert parameter.grad.item() == pytest.approx(2.0 * (3.0 - 1.0) + 0.05 * 2.0 * (3.0 + 2.0))


def test_existing_config_defaults_to_separate_mode() -> None:
    experiment = load_experiment_config(
        Path("configs/experiment/pinn/model_260716_pi_tw40_legacy_fracfocus_f01.yaml")
    )
    assert experiment.train.node_pinn_optimization.update_mode == "separate"
    assert experiment.train.node_pinn_optimization.supervised_anchor_weight == pytest.approx(1.0)
    assert experiment.train.node_pinn_optimization.node_outer_weight == pytest.approx(1.0)
