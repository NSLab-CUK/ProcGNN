from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd
import torch
import yaml
from PIL import Image
from torch import nn

from scripts.final_experiment_registry import load_registry, logical_counts
from scripts.run_final_shap_explainability import RawTargetPrediction
from scripts.run_final_sensitivity_experiments import DEPTHS, MULTIPLIERS, PIN_TERMS
from scripts.train_process_surrogate import _pinn_weight_schedule_scale


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_final_registry_has_exact_models_schema_and_counts() -> None:
    registry = load_registry()
    experiments = registry["experiments"]
    assert registry["target_schema"]["dimension"] == 10
    assert "Vol_Flow" not in registry["target_schema"]["properties"]
    assert experiments["single_process_baselines"]["models"] == [
        "M1", "M2", "M3", "B1", "B2", "B3", "B4", "B5", "B6", "B7",
        "G1", "G2", "G3",
    ]
    assert experiments["single_process_baselines"]["final_gnn_mapping"] == {
        "G1": "GCN", "G2": "GIN", "G3": "GAT",
    }
    assert experiments["single_process_baselines"]["gnn_prediction_unit"] == "one_target_edge"
    assert experiments["single_process_baselines"]["gnn_head"] == "shared_10_property_head"
    assert "GraphSAGE" not in experiments["single_process_baselines"]["models"]
    assert experiments["multi_process_comparison"]["models"] == [
        "B6", "GCN", "GIN", "GAT", "Proposed",
    ]
    assert experiments["multi_process_comparison"]["generic_gnn_head"] == "shared_across_all_processes_and_target_edges"
    assert experiments["multi_process_comparison"]["generic_gnn_process_id_routing"] == "forbidden"
    assert experiments["proposed_data_efficiency"]["ratios"] == pytest.approx(
        [0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    assert experiments["proposed_data_efficiency"]["total_optimizer_steps"] == 80_000
    assert experiments["proposed_data_efficiency"]["max_epochs"] == 30
    assert experiments["proposed_data_efficiency"]["optimizer_step_policy"] == "upper_cap"
    assert experiments["proposed_data_efficiency"]["termination_condition"] == "max_epochs_or_optimizer_step_cap"
    assert logical_counts(registry) == {
        "proposed_single_process": 50,
        "single_process_baselines": 650,
        "multi_process_comparison": 25,
        "proposed_zero_shot": 50,
        "proposed_data_efficiency": 650,
        "computational_efficiency": 25,
        "explainability_shap": 10,
        "sensitivity_depth": 35,
        "sensitivity_pin": 45,
    }


def test_sensitivity_is_depth_one_to_seven_and_active_pin_oat_only() -> None:
    assert DEPTHS == tuple(range(1, 8))
    assert MULTIPLIERS == (0.5, 1.0, 2.0)
    assert set(PIN_TERMS) == {"node_mass", "node_component", "node_atom"}
    assert 0.0 not in MULTIPLIERS


def test_final_10d_loss_weights_use_clean_unit_fraction_and_mass_coefficients() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/experiment/pinn/model_260805_10d_frac1.yaml").read_text(encoding="utf-8")
    )
    train = config["overrides"]["train"]
    assert config["overrides"]["data"]["edge_target_columns"] == [
        "Temp", "Pres", "Frac_H2O", "Frac_H2", "Frac_CH4",
        "Frac_CO2", "Frac_CO", "Frac_O2", "Frac_N2", "Mass_Flow",
    ]
    assert train["pi_fraction_log_loss_weight"] == pytest.approx(1.0)
    assert train["node_balance_pi"]["mass"]["weight"] == pytest.approx(1.0)
    assert train["lambda_node_mass"] == pytest.approx(1.0)
    assert train["early_stopping"] is True
    assert train["early_stopping_patience"] == 5
    assert train["monitor_metric"] == "val_target_edge_property_mean_r2"
    assert train["epoch_sampler"]["base_epoch_size"] == 1000
    assert train["epoch_sampler"]["hard_fill_total_size"] == 2000
    assert train["epoch_sampler"]["hard_fill_ratio"] == pytest.approx(0.5)
    assert train["epoch_sampler"]["hard_target_edges"]["property_quotas"] == {
        "Frac_CO": 400,
        "Mass_Flow": 400,
        "Frac_CH4": 200,
    }


def test_fixed_step_pin_schedule_uses_common_step_fraction() -> None:
    cfg = SimpleNamespace(
        max_optimizer_steps=80_000,
        pinn_weight_schedule={
            "enabled": True,
            "fixed_step_reference_epochs": 30,
            "stages": [
                {"start_epoch": 1, "end_epoch": 5, "multiplier": 0.0},
                {"start_epoch": 6, "end_epoch": 7, "multiplier": 0.5},
                {"start_epoch": 8, "end_epoch": None, "multiplier": 1.0},
            ],
        },
    )
    assert _pinn_weight_schedule_scale(cfg, 0, 0) == 0.0
    assert _pinn_weight_schedule_scale(cfg, 99, 13_333) == 0.0
    assert _pinn_weight_schedule_scale(cfg, 0, 16_000) == 0.5
    assert _pinn_weight_schedule_scale(cfg, 0, 20_000) == 1.0
    assert _pinn_weight_schedule_scale(cfg, 0, 80_000) == 1.0


class _DummyTargetModel(nn.Module):
    def forward(self, batch, task_inputs=None):
        value = batch["x_oper"].sum() + batch["graph_feed_values"].sum()
        row = torch.stack([value + float(index) for index in range(10)])
        return {"main_stream_pred": torch.stack([row, row + 1.0])}


def test_raw_shap_wrapper_preserves_base_gradient_and_target_selection() -> None:
    static = {
        "x_oper": torch.zeros(2, 18),
        "x_oper_mask": torch.ones(2, 18),
        "graph_feed_values": torch.zeros(1, 3),
        "graph_feed_mask": torch.ones(1, 3),
        "graph_feed_input": torch.zeros(1, 6),
    }
    wrapper = RawTargetPrediction(
        model=_DummyTargetModel(), static_batch=static, task_inputs={},
        node_count=2, oper_dim=18, target_edge_index=1, property_index=6,
        oper_mean=torch.zeros(18), oper_std=torch.ones(18),
    )
    raw = torch.zeros(1, 39, requires_grad=True)
    result = wrapper(raw)
    assert result.shape == (1, 1)
    assert result.item() == pytest.approx(7.0)
    result.sum().backward()
    assert raw.grad is not None
    assert torch.all(raw.grad != 0)


def test_final_master_does_not_reference_graphsage_or_old_ratios() -> None:
    text = (PROJECT_ROOT / "scripts/run_final_experiments.py").read_text(encoding="utf-8")
    assert "GraphSAGE" not in text
    assert '"0.05"' not in text
    assert '"0.25"' not in text
    assert "run_final_shap_experiments.py" in text
    assert '"proposed_single_process"' in text


def test_shap_coordinates_cover_every_canonical_node_and_image_bounds() -> None:
    canonical = pd.read_csv(PROJECT_ROOT / "data/reference/canonical_nodes.csv")
    coordinates = pd.read_csv(PROJECT_ROOT / "data/process_overall_img/unit_coordinates.csv")
    canonical["process_number"] = canonical["process_id"].str.extract(r"(\d+)").astype(int)
    assert not coordinates.duplicated(["process_id", "node_name"]).any()
    assert len(coordinates) == len(canonical) == 174
    for process_id in range(1, 11):
        expected = set(canonical.loc[canonical["process_number"].eq(process_id), "node_name"])
        selected = coordinates.loc[coordinates["process_id"].eq(process_id)]
        assert set(selected["node_name"]) == expected
        with Image.open(PROJECT_ROOT / f"data/process_overall_img/{process_id}번.png") as image:
            width, height = image.size
        assert selected["x"].between(0, width - 1).all()
        assert selected["y"].between(0, height - 1).all()
