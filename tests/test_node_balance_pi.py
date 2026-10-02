from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from process_graph.constants import UNIT_TO_IDX
from process_graph.experiment.edge_step_pi_training import (
    _mw_tensor,
    build_pi_physical_outputs,
    masked_normalized_mse_per_sample,
    masked_residual_regression_per_sample,
)
from process_graph.experiment.loaders import load_experiment_config
from process_graph.experiment.node_balance_pi import (
    PI_ATOM_ORDER,
    build_pi_atom_matrix,
    compute_node_balance_pinn_losses,
)


SPECIES = ["H2O", "H2", "CH4", "CO2", "CO", "O2", "N2"]


def _cfg(**overrides):
    values = {
        "eps": 1.0e-8,
        "mw_unit_scale": 1.0,
        "molecular_weight_unit": "g_per_mol",
        "volume_unit_scale": 1000.0,
        "h_basis": "mass_specific",
        "pi_mass_flow_output_space": "raw_z",
        "use_log1p_mass_flow_loss": False,
        "use_node_mass_balance_loss": False,
        "lambda_node_mass": 0.0,
        "node_mass_balance_relative": True,
        "use_node_component_balance_loss": False,
        "lambda_node_component": 0.0,
        "node_component_balance_relative": True,
        "use_node_atom_balance_loss": False,
        "lambda_node_atom": 0.0,
        "node_atom_balance_relative": True,
        "use_node_energy_balance_loss": False,
        "lambda_node_energy": 0.0,
        "node_energy_balance_relative": True,
        "node_energy_valid_unit_types": [],
        "node_energy_exclude_unit_types": ["heater", "cooler", "reactor"],
        "node_energy_use_q": False,
        "node_balance_pi": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _data_cfg():
    return SimpleNamespace(
        normalize_y_edge=False,
        species_order=SPECIES,
        molecular_weight={},
    )


def _physical(mass, species_mole_flow, enthalpy=None):
    mass_t = torch.as_tensor(mass, dtype=torch.float32).reshape(-1, 1)
    species_t = torch.as_tensor(species_mole_flow, dtype=torch.float32).reshape(-1, len(SPECIES))
    frac = species_t / species_t.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return {
        "mass_flow": mass_t,
        "frac": frac,
        "species_mole_flow": species_t,
        "enthalpy_flow": (
            torch.as_tensor(enthalpy, dtype=torch.float32).reshape(-1, 1)
            if enthalpy is not None
            else mass_t
        ),
    }


def _single_internal_graph():
    return {
        "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        "edge_batch": torch.zeros(2, dtype=torch.long),
        "node_batch": torch.zeros(3, dtype=torch.long),
        "node_units": torch.tensor(
            [
                UNIT_TO_IDX["input_virtual"],
                UNIT_TO_IDX["mixer"],
                UNIT_TO_IDX["output_virtual"],
            ],
            dtype=torch.long,
        ),
    }


def test_species_order_mw_and_atom_matrix_are_aligned():
    mw = _mw_tensor(SPECIES, _data_cfg(), _cfg(), torch.device("cpu"), torch.float32)
    atom = build_pi_atom_matrix(SPECIES, device=torch.device("cpu"), dtype=torch.float32)
    assert PI_ATOM_ORDER == ("C", "H", "O", "N")
    assert mw.tolist() == pytest.approx(
        [18.01528, 2.01588, 16.04246, 44.0095, 28.0101, 31.9988, 28.0134]
    )
    assert atom.tolist() == [
        [0.0, 2.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [1.0, 4.0, 0.0, 0.0],
        [1.0, 0.0, 2.0, 0.0],
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 2.0],
    ]


def test_mw_and_volume_scales_match_stream_dataset_units():
    frac = torch.zeros((1, len(SPECIES)))
    frac[:, SPECIES.index("CH4")] = 1.0
    outputs = {
        "main_stream_pred": torch.zeros((1, 10)),
        "T_pred": torch.ones((1, 1)),
        "P_pred": torch.ones((1, 1)),
        "frac_pred": frac,
        "mass_flow_pred": torch.tensor([[16.04246]]),
        "rho_pred": torch.tensor([[2.0]]),
        "h_pred": torch.tensor([[3.0]]),
    }
    physical = build_pi_physical_outputs(
        outputs,
        train_cfg=_cfg(),
        data_cfg=_data_cfg(),
        normalizer=None,
    )
    assert physical["mole_flow"].item() == pytest.approx(1.0)
    assert physical["volume_flow"].item() == pytest.approx(8021.23)


def test_normalized_volume_loss_is_bounded_for_zero_target():
    loss, valid = masked_normalized_mse_per_sample(
        torch.tensor([[1000.0]]),
        torch.tensor([[0.0]]),
        torch.ones((1, 1)),
    )
    assert valid.tolist() == [True]
    assert torch.isfinite(loss).all()
    assert loss.item() == pytest.approx(1.0)


def test_edge_pinn_residual_guards_keep_extreme_and_nonfinite_values_safe():
    pred = torch.tensor([[1.0e20], [float("inf")]], requires_grad=True)
    loss, valid = masked_residual_regression_per_sample(
        pred,
        torch.zeros_like(pred),
        torch.ones_like(pred, dtype=torch.bool),
        criterion=torch.nn.HuberLoss(delta=0.5, reduction="none"),
        residual_abs_clip=10.0,
    )
    assert valid.tolist() == [True, False]
    assert torch.isfinite(loss).all()
    assert loss[0].item() == pytest.approx(4.875)
    loss.sum().backward()
    assert torch.isfinite(pred.grad).all()


def test_node_pinn_residual_clip_bounds_extreme_mass_loss():
    graph = _single_internal_graph()
    one_ch4 = [[0, 0, 1, 0, 0, 0, 0]] * 2
    result = compute_node_balance_pinn_losses(
        physical_outputs=_physical([1.0e20, 1.0], one_ch4),
        physical_targets=_physical([1.0, 1.0], one_ch4),
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=graph["node_units"],
        species_order=SPECIES,
        train_cfg=_cfg(
            node_balance_pi={
                "enabled": True,
                "mass": {
                    "enabled": True,
                    "weight": 0.75,
                    "relative": True,
                    "scale_floor": 1.0,
                    "loss_type": "huber",
                    "huber_delta": 0.5,
                    "residual_abs_clip": 10.0,
                },
            }
        ),
    )
    assert result["loss_node_mass"].item() == pytest.approx(4.875)
    assert result["weighted_node_total"].item() == pytest.approx(0.75 * 4.875)


def test_node_mass_balance_zero_and_nonzero():
    graph = _single_internal_graph()
    one_ch4 = [[0, 0, 1, 0, 0, 0, 0]] * 2
    balanced = compute_node_balance_pinn_losses(
        physical_outputs=_physical([10.0, 10.0], one_ch4),
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=graph["node_units"],
        species_order=SPECIES,
        train_cfg=_cfg(),
    )
    unbalanced = compute_node_balance_pinn_losses(
        physical_outputs=_physical([10.0, 8.0], one_ch4),
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=graph["node_units"],
        species_order=SPECIES,
        train_cfg=_cfg(),
    )
    assert balanced["loss_node_mass"].item() == pytest.approx(0.0)
    assert balanced["diagnostics"]["node_mass_valid_count"] == 1
    assert unbalanced["loss_node_mass"].item() > 0.0


def test_node_mass_prediction_filter_excludes_extreme_pred_residual():
    graph = _single_internal_graph()
    one_ch4 = [[0, 0, 1, 0, 0, 0, 0]] * 2
    unfiltered = compute_node_balance_pinn_losses(
        physical_outputs=_physical([1000.0, 1.0], one_ch4),
        physical_targets=_physical([1.0, 1.0], one_ch4),
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=graph["node_units"],
        species_order=SPECIES,
        train_cfg=_cfg(
            use_node_mass_balance_loss=True,
            lambda_node_mass=1.0,
            node_balance_pi={
                "mass": {
                    "enabled": True,
                    "weight": 1.0,
                    "loss_type": "huber",
                    "huber_delta": 1.0e-3,
                }
            },
        ),
    )
    filtered = compute_node_balance_pinn_losses(
        physical_outputs=_physical([1000.0, 1.0], one_ch4),
        physical_targets=_physical([1.0, 1.0], one_ch4),
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=graph["node_units"],
        species_order=SPECIES,
        train_cfg=_cfg(
            use_node_mass_balance_loss=True,
            lambda_node_mass=1.0,
            node_balance_pi={
                "mass": {
                    "enabled": True,
                    "weight": 1.0,
                    "loss_type": "huber",
                    "huber_delta": 1.0e-3,
                    "prediction_filter": {"max_abs_residual": 10.0},
                }
            },
        ),
    )
    assert unfiltered["loss_node_mass"].item() > 0.0
    assert filtered["diagnostics"]["node_mass_valid_count"] == 0
    assert filtered["diagnostics"]["node_mass_pred_residual_rejected_count"] == 1
    assert filtered["loss_node_mass"].item() == pytest.approx(0.0, abs=1.0e-8)


def test_node_energy_balance_and_default_reactor_exclusion():
    graph = _single_internal_graph()
    flows = [[0, 0, 1, 0, 0, 0, 0]] * 2
    balanced = compute_node_balance_pinn_losses(
        physical_outputs=_physical([10, 10], flows, enthalpy=[50, 50]),
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=graph["node_units"],
        species_order=SPECIES,
        train_cfg=_cfg(),
    )
    unbalanced = compute_node_balance_pinn_losses(
        physical_outputs=_physical([10, 10], flows, enthalpy=[50, 40]),
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=graph["node_units"],
        species_order=SPECIES,
        train_cfg=_cfg(),
    )
    assert balanced["loss_node_energy"].item() == pytest.approx(0.0)
    assert unbalanced["loss_node_energy"].item() > 0.0

    reactor_units = graph["node_units"].clone()
    reactor_units[1] = UNIT_TO_IDX["smr_reactor"]
    included = compute_node_balance_pinn_losses(
        physical_outputs=_physical([10, 10], flows, enthalpy=[50, 40]),
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=reactor_units,
        species_order=SPECIES,
        train_cfg=_cfg(node_energy_exclude_unit_types=[]),
    )
    assert included["loss_node_energy"].item() > 0.0
    assert included["diagnostics"]["node_energy_valid_count"] == 1

    hx_units = graph["node_units"].clone()
    hx_units[1] = UNIT_TO_IDX["hx_hot"]
    hx_excluded = compute_node_balance_pinn_losses(
        physical_outputs=_physical([10, 10], flows, enthalpy=[50, 40]),
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=hx_units,
        species_order=SPECIES,
        train_cfg=_cfg(node_energy_exclude_unit_types=["heat_exchanger"]),
    )
    assert hx_excluded["diagnostics"]["node_energy_valid_count"] == 0


def test_node_energy_prediction_filter_keeps_true_consistency_but_rejects_extreme_prediction():
    graph = _single_internal_graph()
    flows = [[0, 0, 1, 0, 0, 0, 0]] * 2
    train_cfg = _cfg(
        use_node_energy_balance_loss=True,
        lambda_node_energy=1.0,
        node_balance_pi={
            "energy": {
                "enabled": True,
                "weight": 1.0,
                "loss_type": "huber",
                "huber_delta": 1.0e-3,
                "use_process_main_qw": True,
                "consistency_filter": {
                    "enabled": True,
                    "max_true_normalized_residual": 1.0e-2,
                },
                "prediction_filter": {"max_abs_residual": 10.0},
            }
        },
    )
    result = compute_node_balance_pinn_losses(
        physical_outputs=_physical([10.0, 10.0], flows, enthalpy=[100000.0, 1.0]),
        physical_targets=_physical([10.0, 10.0], flows, enthalpy=[10.0, 10.0]),
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=graph["node_units"],
        species_order=SPECIES,
        train_cfg=train_cfg,
    )
    assert result["diagnostics"]["node_energy_consistent_count"] == 1
    assert result["diagnostics"]["node_energy_pred_residual_rejected_count"] == 1
    assert result["diagnostics"]["node_energy_valid_count"] == 0
    assert result["loss_node_energy"].item() == pytest.approx(0.0, abs=1.0e-8)


def test_node_atom_balance_reaction_and_nonconserving_case():
    # CO + H2O -> CO2 + H2 conserves C/H/O/N.
    edge_index = torch.tensor([[0, 1, 2, 2], [2, 2, 3, 4]], dtype=torch.long)
    units = torch.tensor(
        [
            UNIT_TO_IDX["input_virtual"],
            UNIT_TO_IDX["input_virtual"],
            UNIT_TO_IDX["smr_reactor"],
            UNIT_TO_IDX["output_virtual"],
            UNIT_TO_IDX["output_virtual"],
        ]
    )
    flows = torch.zeros((4, len(SPECIES)))
    flows[0, SPECIES.index("CO")] = 1.0
    flows[1, SPECIES.index("H2O")] = 1.0
    flows[2, SPECIES.index("CO2")] = 1.0
    flows[3, SPECIES.index("H2")] = 1.0
    mw = [18.01528, 2.01588, 16.04246, 44.0095, 28.0101, 31.9988, 28.0134]
    conserved = compute_node_balance_pinn_losses(
        physical_outputs=_physical(
            [
                mw[SPECIES.index("CO")],
                mw[SPECIES.index("H2O")],
                mw[SPECIES.index("CO2")],
                mw[SPECIES.index("H2")],
            ],
            flows,
        ),
        edge_index=edge_index,
        edge_batch_or_graph_id=torch.zeros(4, dtype=torch.long),
        node_batch_or_graph_id=torch.zeros(5, dtype=torch.long),
        node_unit_type=units,
        species_order=SPECIES,
        train_cfg=_cfg(),
    )
    assert conserved["loss_node_atom"].item() == pytest.approx(0.0, abs=1.0e-8)

    flows[0].zero_()
    flows[0, SPECIES.index("CH4")] = 1.0
    nonconserving = compute_node_balance_pinn_losses(
        physical_outputs=_physical(
            [
                mw[SPECIES.index("CH4")],
                mw[SPECIES.index("H2O")],
                mw[SPECIES.index("CO2")],
                mw[SPECIES.index("H2")],
            ],
            flows,
        ),
        edge_index=edge_index,
        edge_batch_or_graph_id=torch.zeros(4, dtype=torch.long),
        node_batch_or_graph_id=torch.zeros(5, dtype=torch.long),
        node_unit_type=units,
        species_order=SPECIES,
        train_cfg=_cfg(),
    )
    assert nonconserving["loss_node_atom"].item() > 0.0


def test_component_balance_applies_to_nonreactive_and_excludes_reactive():
    edge_index = torch.tensor([[0, 1, 2], [2, 2, 3]], dtype=torch.long)
    units = torch.tensor(
        [
            UNIT_TO_IDX["input_virtual"],
            UNIT_TO_IDX["input_virtual"],
            UNIT_TO_IDX["mixer"],
            UNIT_TO_IDX["output_virtual"],
        ],
        dtype=torch.long,
    )
    flows = torch.zeros((3, len(SPECIES)))
    flows[:, SPECIES.index("H2")] = torch.tensor([1.0, 2.0, 3.0])
    h2_mw = 2.01588
    out = compute_node_balance_pinn_losses(
        physical_outputs=_physical([h2_mw, 2 * h2_mw, 3 * h2_mw], flows),
        edge_index=edge_index,
        edge_batch_or_graph_id=torch.zeros(3, dtype=torch.long),
        node_batch_or_graph_id=torch.zeros(4, dtype=torch.long),
        node_unit_type=units,
        species_order=SPECIES,
        train_cfg=_cfg(use_node_component_balance_loss=True, lambda_node_component=1.0),
    )
    assert out["loss_node_component"].item() == pytest.approx(0.0, abs=1.0e-8)
    assert out["diagnostics"]["node_component_valid_count"] >= 1

    units[2] = UNIT_TO_IDX["smr_reactor"]
    excluded = compute_node_balance_pinn_losses(
        physical_outputs=_physical([h2_mw, 2 * h2_mw, 3 * h2_mw], flows),
        edge_index=edge_index,
        edge_batch_or_graph_id=torch.zeros(3, dtype=torch.long),
        node_batch_or_graph_id=torch.zeros(4, dtype=torch.long),
        node_unit_type=units,
        species_order=SPECIES,
        train_cfg=_cfg(use_node_component_balance_loss=True, lambda_node_component=1.0),
    )
    assert excluded["diagnostics"]["node_component_valid_count"] == 0


def test_energy_balance_uses_q_w_and_ignores_hx_q():
    graph = _single_internal_graph()
    flows = [[0, 0, 1, 0, 0, 0, 0]] * 2
    heater = compute_node_balance_pinn_losses(
        physical_outputs=_physical([1, 1], flows, enthalpy=[10, 15]),
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=graph["node_units"],
        species_order=SPECIES,
        train_cfg=_cfg(use_node_energy_balance_loss=True, lambda_node_energy=1.0),
        node_q=torch.tensor([0.0, 5.0, 0.0]),
        node_w=torch.zeros(3),
        node_qw_valid_mask=torch.ones(3),
    )
    assert heater["loss_node_energy"].item() == pytest.approx(0.0, abs=1.0e-8)

    compressor = compute_node_balance_pinn_losses(
        physical_outputs=_physical([1, 1], flows, enthalpy=[10, 13]),
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=graph["node_units"],
        species_order=SPECIES,
        train_cfg=_cfg(use_node_energy_balance_loss=True, lambda_node_energy=1.0),
        node_q=torch.zeros(3),
        node_w=torch.tensor([0.0, 3.0, 0.0]),
        node_qw_valid_mask=torch.ones(3),
    )
    assert compressor["loss_node_energy"].item() == pytest.approx(0.0, abs=1.0e-8)

    hx_units = graph["node_units"].clone()
    hx_units[1] = UNIT_TO_IDX["hx_dt"]
    hx = compute_node_balance_pinn_losses(
        physical_outputs=_physical([1, 1], flows, enthalpy=[20, 20]),
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=hx_units,
        species_order=SPECIES,
        train_cfg=_cfg(use_node_energy_balance_loss=True, lambda_node_energy=1.0),
        node_q=torch.tensor([0.0, 8.0, 0.0]),
        node_w=torch.zeros(3),
        node_qw_valid_mask=torch.ones(3),
    )
    assert hx["loss_node_energy"].item() == pytest.approx(0.0, abs=1.0e-8)


def test_energy_consistency_filter_keeps_consistent_true_nodes():
    graph = _single_internal_graph()
    flows = [[0, 0, 1, 0, 0, 0, 0]] * 2
    result = compute_node_balance_pinn_losses(
        physical_outputs=_physical([1, 1], flows, enthalpy=[10, 15]),
        physical_targets=_physical([1, 1], flows, enthalpy=[10, 15]),
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=graph["node_units"],
        species_order=SPECIES,
        train_cfg=_cfg(
            use_node_energy_balance_loss=True,
            lambda_node_energy=1.0,
            node_balance_pi={
                "energy": {
                    "enabled": True,
                    "weight": 1.0,
                    "consistency_filter": {
                        "enabled": True,
                        "max_true_normalized_residual": 1.0e-2,
                        "mode": "exclude",
                    },
                }
            },
        ),
        node_q=torch.tensor([0.0, 5.0, 0.0]),
        node_w=torch.zeros(3),
        node_qw_valid_mask=torch.ones(3),
    )
    assert result["loss_node_energy"].item() == pytest.approx(0.0, abs=1.0e-8)
    assert result["diagnostics"]["node_energy_structural_valid_count"] == 1
    assert result["diagnostics"]["node_energy_consistent_count"] == 1
    assert result["diagnostics"]["node_energy_consistency_rejected_count"] == 0


def test_energy_consistency_filter_excludes_inconsistent_true_nodes():
    graph = _single_internal_graph()
    flows = [[0, 0, 1, 0, 0, 0, 0]] * 2
    cfg = _cfg(
        use_node_energy_balance_loss=True,
        lambda_node_energy=1.0,
        node_balance_pi={
            "energy": {
                "enabled": True,
                "weight": 1.0,
                "consistency_filter": {
                    "enabled": True,
                    "max_true_normalized_residual": 1.0e-2,
                    "mode": "exclude",
                },
            }
        },
    )
    filtered = compute_node_balance_pinn_losses(
        physical_outputs=_physical([1, 1], flows, enthalpy=[10, 30]),
        physical_targets=_physical([1, 1], flows, enthalpy=[10, 10]),
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=graph["node_units"],
        species_order=SPECIES,
        train_cfg=cfg,
        node_q=torch.tensor([0.0, 5.0, 0.0]),
        node_w=torch.zeros(3),
        node_qw_valid_mask=torch.ones(3),
    )
    unfiltered = compute_node_balance_pinn_losses(
        physical_outputs=_physical([1, 1], flows, enthalpy=[10, 30]),
        physical_targets=_physical([1, 1], flows, enthalpy=[10, 10]),
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=graph["node_units"],
        species_order=SPECIES,
        train_cfg=_cfg(use_node_energy_balance_loss=True, lambda_node_energy=1.0),
        node_q=torch.tensor([0.0, 5.0, 0.0]),
        node_w=torch.zeros(3),
        node_qw_valid_mask=torch.ones(3),
    )
    assert filtered["diagnostics"]["node_energy_structural_valid_count"] == 1
    assert filtered["diagnostics"]["node_energy_consistent_count"] == 0
    assert filtered["diagnostics"]["node_energy_consistency_rejected_count"] == 1
    assert filtered["loss_node_energy"].item() == pytest.approx(0.0, abs=1.0e-8)
    assert torch.isfinite(filtered["loss_node_energy"])
    assert unfiltered["loss_node_energy"].item() > 0.0


def test_energy_consistency_filter_boundary_is_inclusive():
    graph = _single_internal_graph()
    flows = [[0, 0, 1, 0, 0, 0, 0]] * 2
    # r_true = 15 - H_out, scale = 10 + H_out + 5.
    # r_true / scale = 0.01, so the <= threshold boundary must be accepted.
    target_hout = 14.702970297029701
    result = compute_node_balance_pinn_losses(
        physical_outputs=_physical([1, 1], flows, enthalpy=[10, target_hout]),
        physical_targets=_physical([1, 1], flows, enthalpy=[10, target_hout]),
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=graph["node_units"],
        species_order=SPECIES,
        train_cfg=_cfg(
            use_node_energy_balance_loss=True,
            lambda_node_energy=1.0,
            node_balance_pi={
                "energy": {
                    "enabled": True,
                    "weight": 1.0,
                    "consistency_filter": {
                        "enabled": True,
                        "max_true_normalized_residual": 1.0e-2,
                        "mode": "exclude",
                    },
                }
            },
        ),
        node_q=torch.tensor([0.0, 5.0, 0.0]),
        node_w=torch.zeros(3),
        node_qw_valid_mask=torch.ones(3),
    )
    assert result["diagnostics"]["node_energy_consistent_count"] == 1
    assert result["diagnostics"]["node_energy_true_residual_abs_max"] == pytest.approx(1.0e-2)


def test_energy_consistency_filter_keeps_prediction_gradient_only():
    graph = _single_internal_graph()
    flows = [[0, 0, 1, 0, 0, 0, 0]] * 2
    pred = _physical([1, 1], flows, enthalpy=[10, 14])
    pred["enthalpy_flow"].requires_grad_(True)
    true = _physical([1, 1], flows, enthalpy=[10, 15])
    true["enthalpy_flow"].requires_grad_(True)
    q = torch.tensor([0.0, 5.0, 0.0], requires_grad=True)
    result = compute_node_balance_pinn_losses(
        physical_outputs=pred,
        physical_targets=true,
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=graph["node_units"],
        species_order=SPECIES,
        train_cfg=_cfg(
            use_node_energy_balance_loss=True,
            lambda_node_energy=1.0,
            node_balance_pi={
                "energy": {
                    "enabled": True,
                    "weight": 1.0,
                    "consistency_filter": {
                        "enabled": True,
                        "max_true_normalized_residual": 1.0e-2,
                        "mode": "exclude",
                    },
                }
            },
        ),
        node_q=q,
        node_w=torch.zeros(3),
        node_qw_valid_mask=torch.ones(3),
    )
    result["loss_node_energy"].backward()
    assert pred["enthalpy_flow"].grad is not None
    assert pred["enthalpy_flow"].grad.abs().sum().item() > 0.0
    assert true["enthalpy_flow"].grad is None
    assert q.grad is None


def test_node_losses_do_not_mix_graphs_and_validate_group_ids():
    edge_index = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.long)
    edge_batch = torch.tensor([0, 0, 1, 1])
    node_batch = torch.tensor([0, 0, 0, 1, 1, 1])
    units = torch.tensor(
        [
            UNIT_TO_IDX["input_virtual"],
            UNIT_TO_IDX["mixer"],
            UNIT_TO_IDX["output_virtual"],
        ]
        * 2
    )
    flows = [[0, 0, 1, 0, 0, 0, 0]] * 4
    result = compute_node_balance_pinn_losses(
        physical_outputs=_physical([10, 10, 7, 7], flows),
        edge_index=edge_index,
        edge_batch_or_graph_id=edge_batch,
        node_batch_or_graph_id=node_batch,
        node_unit_type=units,
        species_order=SPECIES,
        train_cfg=_cfg(),
    )
    assert result["loss_node_mass"].item() == pytest.approx(0.0)
    assert result["diagnostics"]["node_graph_count"] == 2
    assert result["diagnostics"]["node_mass_valid_count"] == 2

    bad_edge_batch = edge_batch.clone()
    bad_edge_batch[-1] = 0
    with pytest.raises(RuntimeError, match="mix edges"):
        compute_node_balance_pinn_losses(
            physical_outputs=_physical([10, 10, 7, 7], flows),
            edge_index=edge_index,
            edge_batch_or_graph_id=bad_edge_batch,
            node_batch_or_graph_id=node_batch,
            node_unit_type=units,
            species_order=SPECIES,
            train_cfg=_cfg(),
        )


def test_node_loss_weights_and_zero_valid_nodes_are_finite():
    graph = _single_internal_graph()
    flows = [[0, 0, 1, 0, 0, 0, 0]] * 2
    off = compute_node_balance_pinn_losses(
        physical_outputs=_physical([10, 8], flows),
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=graph["node_units"],
        species_order=SPECIES,
        train_cfg=_cfg(lambda_node_mass=2.0, use_node_mass_balance_loss=False),
    )
    on = compute_node_balance_pinn_losses(
        physical_outputs=_physical([10, 8], flows),
        edge_index=graph["edge_index"],
        edge_batch_or_graph_id=graph["edge_batch"],
        node_batch_or_graph_id=graph["node_batch"],
        node_unit_type=graph["node_units"],
        species_order=SPECIES,
        train_cfg=_cfg(lambda_node_mass=2.0, use_node_mass_balance_loss=True),
    )
    assert off["weighted_node_total"].item() == pytest.approx(0.0)
    assert on["weighted_node_total"].item() == pytest.approx(2.0 * on["loss_node_mass"].item())

    only_virtual = compute_node_balance_pinn_losses(
        physical_outputs=_physical([1.0], [[0, 0, 1, 0, 0, 0, 0]]),
        edge_index=torch.tensor([[0], [1]]),
        edge_batch_or_graph_id=torch.zeros(1, dtype=torch.long),
        node_batch_or_graph_id=torch.zeros(2, dtype=torch.long),
        node_unit_type=torch.tensor(
            [UNIT_TO_IDX["input_virtual"], UNIT_TO_IDX["output_virtual"]]
        ),
        species_order=SPECIES,
        train_cfg=_cfg(
            use_node_mass_balance_loss=True,
            lambda_node_mass=1.0,
            use_node_atom_balance_loss=True,
            lambda_node_atom=1.0,
        ),
    )
    assert torch.isfinite(only_virtual["weighted_node_total"])
    assert only_virtual["weighted_node_total"].item() == pytest.approx(0.0)
    assert only_virtual["diagnostics"]["node_mass_valid_count"] == 0


def test_a6_pinn_all_config_preserves_a6_and_enables_all_losses():
    root = Path(__file__).resolve().parents[1]
    base = load_experiment_config(
        root
        / "configs/experiment/pinn/process_surrogate_edge_all_v3_pi_all_10pct_softmax_temp05_fracclr_massdirectlog.yaml"
    )
    config = load_experiment_config(
        root
        / "configs/experiment/pinn/process_surrogate_edge_all_v3_pi_all_10pct_softmax_temp05_fracclr_massdirectlog_pinn_all.yaml"
    )
    assert config.model == base.model
    assert config.data == base.data
    assert config.model.fraction_activation == "softmax"
    assert config.model.fraction_temperature == pytest.approx(0.5)
    assert config.train.pi_fraction_loss_type == "clr"
    assert config.train.pi_fraction_clr_loss_weight == pytest.approx(0.5)
    assert config.train.pi_mass_flow_output_space == "log1p"
    assert config.train.compute_node_balance_diagnostics is True
    assert config.train.use_node_mass_balance_loss is True
    assert config.train.lambda_node_mass == pytest.approx(1.0e-4)
    assert config.train.use_node_atom_balance_loss is True
    assert config.train.lambda_node_atom == pytest.approx(1.0e-5)
    assert config.train.use_node_energy_balance_loss is True
    assert config.train.lambda_node_energy == pytest.approx(1.0e-6)
    assert config.train.use_volume_loss is True
    assert config.train.lambda_volume == pytest.approx(1.0e-6)
    assert config.train.mw_unit_scale == pytest.approx(1.0)
    assert config.train.volume_unit_scale == pytest.approx(1000.0)
    assert config.train.volume_loss_normalized is True
    assert config.train.node_energy_use_q is False


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("m", (True, False, False, False)),
        ("a", (False, True, False, False)),
        ("v", (False, False, False, True)),
        ("e", (False, False, True, False)),
        ("ma", (True, True, False, False)),
        ("mv", (True, False, False, True)),
        ("mav", (True, True, False, True)),
    ],
)
def test_a6_pinn_ablation_configs_enable_only_requested_losses(name, expected):
    root = Path(__file__).resolve().parents[1]
    base = load_experiment_config(
        root
        / "configs/experiment/pinn/process_surrogate_edge_all_v3_pi_all_10pct_softmax_temp05_fracclr_massdirectlog.yaml"
    )
    config = load_experiment_config(root / f"configs/experiment/pinn/a6_pinn_{name}.yaml")
    resolved = {
        "mass": bool(
            config.train.use_node_mass_balance_loss and config.train.lambda_node_mass > 0
        ),
        "atom": bool(
            config.train.use_node_atom_balance_loss and config.train.lambda_node_atom > 0
        ),
        "energy": bool(
            config.train.use_node_energy_balance_loss and config.train.lambda_node_energy > 0
        ),
        "volume": bool(config.train.use_volume_loss and config.train.lambda_volume > 0),
    }
    assert tuple(resolved.values()) == expected
    assert config.model == base.model
    assert config.data == base.data
    assert config.train.compute_node_balance_diagnostics is True
    assert config.train.mw_unit_scale == pytest.approx(1.0)
    assert config.train.volume_unit_scale == pytest.approx(1000.0)
    assert config.train.volume_loss_normalized is True
    assert config.train.node_energy_use_q is False


@pytest.mark.parametrize(
    ("name", "weights"),
    [
        ("m", (0.5, 0.0, 0.0, 0.0)),
        ("a", (0.0, 0.25, 0.0, 0.0)),
        ("v", (0.0, 0.0, 0.0, 0.10)),
        ("e", (0.0, 0.0, 0.01, 0.0)),
        ("ma", (0.5, 0.25, 0.0, 0.0)),
        ("mv", (0.5, 0.0, 0.0, 0.10)),
        ("mav", (0.5, 0.25, 0.0, 0.10)),
    ],
)
def test_a6_pinn_strong_ablation_configs_preserve_structure(name, weights):
    root = Path(__file__).resolve().parents[1]
    base = load_experiment_config(root / f"configs/experiment/pinn/a6_pinn_{name}.yaml")
    strong = load_experiment_config(
        root / f"configs/experiment/pinn/a6_pinn_{name}_strong.yaml"
    )
    assert strong.model == base.model
    assert strong.data == base.data
    assert (
        strong.train.lambda_node_mass,
        strong.train.lambda_node_atom,
        strong.train.lambda_node_energy,
        strong.train.lambda_volume,
    ) == pytest.approx(weights)
    assert strong.train.use_node_mass_balance_loss == base.train.use_node_mass_balance_loss
    assert strong.train.use_node_atom_balance_loss == base.train.use_node_atom_balance_loss
    assert strong.train.use_node_energy_balance_loss == base.train.use_node_energy_balance_loss
    assert strong.train.use_volume_loss == base.train.use_volume_loss


def test_a6_pinn_all_strong_config_preserves_structure():
    root = Path(__file__).resolve().parents[1]
    base = load_experiment_config(
        root
        / "configs/experiment/pinn/process_surrogate_edge_all_v3_pi_all_10pct_softmax_temp05_fracclr_massdirectlog_pinn_all.yaml"
    )
    strong = load_experiment_config(
        root
        / "configs/experiment/pinn/process_surrogate_edge_all_v3_pi_all_10pct_softmax_temp05_fracclr_massdirectlog_pinn_all_strong.yaml"
    )
    assert strong.model == base.model
    assert strong.data == base.data
    assert (
        strong.train.lambda_node_mass,
        strong.train.lambda_node_atom,
        strong.train.lambda_node_energy,
        strong.train.lambda_volume,
    ) == pytest.approx((0.5, 0.25, 0.01, 0.10))
    assert strong.train.use_node_mass_balance_loss is True
    assert strong.train.use_node_atom_balance_loss is True
    assert strong.train.use_node_energy_balance_loss is True
    assert strong.train.use_volume_loss is True
