from types import SimpleNamespace

import json
import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from process_graph.constants import STREAM_EDGE_FEATURE_SLOTS
from process_graph.data.tabular_dataset import compute_y_edge_log1p_column_scaler
from process_graph.models.edge_decoder import PIGroupedPropertyHead
from process_graph.experiment.edge_step_pi_training import (
    DEFAULT_MOLECULAR_WEIGHT_G_PER_MOL,
    PIAllEdgePropertyR2Accumulator,
    build_pi_physical_outputs,
    compute_single_edge_pinn_loss,
    _clr_transform,
    _edge_group_is_target,
    _log_fraction_transform,
    _normalize_target_if_needed,
    _sort_edge_groups,
    _RegressionMoments,
)
from process_graph.experiment.loaders import load_experiment_config, load_train_config
from process_graph.experiment.pi_mass_flow import assert_checkpoint_pi_mass_flow_compatible
from process_graph.experiment.target_edge_10d_metrics import extract_main_stream_metric_tensors


def test_all_edge_constant_target_r2_is_high():
    moments = _RegressionMoments()
    moments.update(
        torch.zeros(4),
        torch.ones(4),
        torch.ones(4, dtype=torch.bool),
    )

    row = moments.row("Frac_O2")

    assert row["R2"] == 0.999
    assert "r2_constant_target_substitute" not in row


def test_pi_all_edge_property_r2_includes_direct_and_derived_values(tmp_path):
    species = ["H2O", "H2", "CH4", "CO2", "CO", "O2", "N2"]
    frac = torch.tensor(
        [
            [0.10, 0.20, 0.10, 0.20, 0.10, 0.10, 0.20],
            [0.11, 0.19, 0.11, 0.19, 0.11, 0.11, 0.18],
            [0.12, 0.18, 0.12, 0.18, 0.12, 0.12, 0.16],
            [0.13, 0.17, 0.13, 0.17, 0.13, 0.13, 0.14],
        ],
        dtype=torch.float32,
    )
    temp = torch.arange(300.0, 304.0).unsqueeze(-1)
    pres = torch.arange(10.0, 14.0).unsqueeze(-1)
    mass = torch.arange(1.0, 5.0).unsqueeze(-1)
    rho = torch.arange(2.0, 6.0).unsqueeze(-1)
    enthalpy = torch.arange(100.0, 104.0).unsqueeze(-1)
    mw = torch.tensor([DEFAULT_MOLECULAR_WEIGHT_G_PER_MOL[s] / 1000.0 for s in species])
    mole = mass / (frac * mw).sum(dim=-1, keepdim=True)
    volume = mass / rho
    columns = list(STREAM_EDGE_FEATURE_SLOTS) + ["Enthalpy", "Density"]
    true = torch.cat([temp, pres, volume, mole, mass, frac, enthalpy, rho], dim=-1)
    outputs = {
        "T_pred": temp,
        "P_pred": pres,
        "frac_pred": frac,
        "mass_flow_pred": mass,
        "rho_pred": rho,
        "h_pred": enthalpy,
        "main_stream_pred": torch.cat([temp, pres, frac, mass], dim=-1),
    }
    data_cfg = SimpleNamespace(
        species_order=species,
        molecular_weight=DEFAULT_MOLECULAR_WEIGHT_G_PER_MOL,
        normalize_y_edge=False,
    )
    train_cfg = SimpleNamespace(
        eps=1.0e-8,
        molecular_weight_unit="g_per_mol",
        h_basis="mass_specific",
        metric_relevance_fraction_threshold=1.0e-3,
    )
    acc = PIAllEdgePropertyR2Accumulator(train_cfg=train_cfg)
    acc.update_batch(
        outputs=outputs,
        targets_raw={"edge_stream": true},
        target_masks={"edge_stream": torch.ones(4)},
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=None,
    )
    scalars = acc.write_artifacts(tmp_path)
    rows = pd.read_csv(tmp_path / "pi_all_edge_property_r2.csv")

    assert set(rows["property_name"]) == set(columns)
    assert set(rows[rows["property_name"].isin(["Mole_Flow", "Vol_Flow"])]["prediction_kind"]) == {"derived"}
    assert scalars["pi_all_edge_r2_mole_flow"] == 1.0
    assert scalars["pi_all_edge_r2_vol_flow"] == 1.0
    assert scalars["pi_all_edge_r2_frac_ch4"] == 1.0
    assert scalars["pi_all_edge_r2_density"] == 1.0
    assert scalars["pi_all_edge_r2_enthalpy"] == 1.0


def test_pi_all_edge_oracle_artifacts_are_separate_from_raw_metrics(tmp_path):
    species = ["H2O", "H2", "CH4", "CO2", "CO", "O2", "N2"]
    frac = torch.tensor(
        [
            [0.20, 0.20, 0.10, 0.10, 0.10, 0.10, 0.20],
            [0.19, 0.21, 0.11, 0.09, 0.09, 0.11, 0.20],
            [0.18, 0.22, 0.12, 0.08, 0.08, 0.12, 0.20],
            [0.17, 0.23, 0.13, 0.07, 0.07, 0.13, 0.20],
        ],
        dtype=torch.float32,
    )
    temp = torch.arange(300.0, 304.0).unsqueeze(-1)
    pres = torch.arange(10.0, 14.0).unsqueeze(-1)
    mass = torch.arange(1.0, 5.0).unsqueeze(-1)
    rho = torch.arange(2.0, 6.0).unsqueeze(-1)
    enthalpy = torch.arange(100.0, 104.0).unsqueeze(-1)
    mw = torch.tensor([DEFAULT_MOLECULAR_WEIGHT_G_PER_MOL[s] / 1000.0 for s in species])
    mole = mass / (frac * mw).sum(dim=-1, keepdim=True)
    volume = mass / rho
    columns = list(STREAM_EDGE_FEATURE_SLOTS) + ["Enthalpy", "Density"]
    true = torch.cat([temp, pres, volume, mole, mass, frac, enthalpy, rho], dim=-1)
    outputs = {
        "T_pred": temp,
        "P_pred": pres,
        "frac_pred": frac * 0.95,
        "mass_flow_pred": mass,
        "rho_pred": rho,
        "h_pred": enthalpy,
        "main_stream_pred": torch.cat([temp, pres, frac * 0.95, mass], dim=-1),
    }
    data_cfg = SimpleNamespace(
        species_order=species,
        molecular_weight=DEFAULT_MOLECULAR_WEIGHT_G_PER_MOL,
        normalize_y_edge=False,
    )
    train_cfg = SimpleNamespace(
        eps=1.0e-8,
        molecular_weight_unit="g_per_mol",
        h_basis="mass_specific",
        metric_relevance_enabled=True,
        metric_relevance_fraction_threshold=0.6,
        save_oracle_diagnostic_metrics=True,
        oracle_min_samples=2,
        oracle_threshold_candidates=[0.0, 0.1],
    )
    acc = PIAllEdgePropertyR2Accumulator(train_cfg=train_cfg)
    acc.update_batch(
        outputs=outputs,
        targets_raw={"edge_stream": true},
        target_masks={"edge_stream": torch.ones(4)},
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=None,
        split_name="val",
    )
    acc.write_artifacts(tmp_path)

    main = pd.read_csv(tmp_path / "pi_all_edge_property_r2.csv")
    assert "relevance_rule" not in main.columns
    assert (tmp_path / "pi_all_edge_property_r2_relevance_deprecated.csv").is_file()
    assert (tmp_path / "oracle_pi_all_edge_threshold_sweep.csv").is_file()
    assert (tmp_path / "oracle_pi_all_edge_calibrated_r2.csv").is_file()
    assert (tmp_path / "oracle_pi_all_edge_fraction_postprocess_r2.csv").is_file()
    assert (tmp_path / "oracle_pi_all_edge_summary.json").is_file()
    threshold = pd.read_csv(tmp_path / "oracle_pi_all_edge_threshold_sweep.csv")
    assert {"Mass_Flow", "Temp", "Frac_CH4"} <= set(threshold["property"])
    assert threshold[threshold["property"] == "Mass_Flow"].iloc[0]["threshold_type"] == "quantile"
    assert threshold[threshold["property"] == "Frac_CH4"].iloc[0]["threshold_type"] == "absolute"
    calibrated = pd.read_csv(tmp_path / "oracle_pi_all_edge_calibrated_r2.csv")
    assert {"Mass_Flow", "Temp", "Frac_CH4"} <= set(calibrated["property"])


def test_pi_all_edge_diagnostic_display_artifacts_are_separate_from_raw_metrics(tmp_path):
    species = ["H2O", "H2", "CH4", "CO2", "CO", "O2", "N2"]
    frac = torch.tensor(
        [
            [0.20, 0.20, 0.10, 0.10, 0.10, 0.10, 0.20],
            [0.19, 0.21, 0.11, 0.09, 0.09, 0.11, 0.20],
            [0.18, 0.22, 0.12, 0.08, 0.08, 0.12, 0.20],
            [0.17, 0.23, 0.13, 0.07, 0.07, 0.13, 0.20],
        ],
        dtype=torch.float32,
    )
    temp = torch.arange(300.0, 304.0).unsqueeze(-1)
    pres = torch.arange(10.0, 14.0).unsqueeze(-1)
    mass = torch.arange(1.0, 5.0).unsqueeze(-1)
    rho = torch.arange(2.0, 6.0).unsqueeze(-1)
    enthalpy = torch.arange(100.0, 104.0).unsqueeze(-1)
    mw = torch.tensor([DEFAULT_MOLECULAR_WEIGHT_G_PER_MOL[s] / 1000.0 for s in species])
    mole = mass / (frac * mw).sum(dim=-1, keepdim=True)
    volume = mass / rho
    columns = list(STREAM_EDGE_FEATURE_SLOTS) + ["Enthalpy", "Density"]
    true = torch.cat([temp, pres, volume, mole, mass, frac, enthalpy, rho], dim=-1)
    outputs = {
        "T_pred": temp * 1.1 + 2.0,
        "P_pred": pres * 1.1 + 2.0,
        "frac_pred": frac * 0.95,
        "mass_flow_pred": mass * 1.2 + 0.5,
        "rho_pred": rho * 1.1 + 0.1,
        "h_pred": enthalpy * 1.1 + 0.1,
        "main_stream_pred": torch.cat([temp * 1.1 + 2.0, pres * 1.1 + 2.0, frac * 0.95, mass * 1.2 + 0.5], dim=-1),
    }
    data_cfg = SimpleNamespace(
        species_order=species,
        molecular_weight=DEFAULT_MOLECULAR_WEIGHT_G_PER_MOL,
        normalize_y_edge=False,
    )
    train_cfg = SimpleNamespace(
        eps=1.0e-8,
        molecular_weight_unit="g_per_mol",
        h_basis="mass_specific",
        save_diagnostic_display_metrics=True,
        diagnostic_min_samples=2,
    )
    acc = PIAllEdgePropertyR2Accumulator(train_cfg=train_cfg)
    acc.update_batch(
        outputs=outputs,
        targets_raw={"edge_stream": true},
        target_masks={"edge_stream": torch.ones(4)},
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=None,
        split_name="val",
    )
    acc.write_artifacts(tmp_path)

    main = pd.read_csv(tmp_path / "pi_all_edge_property_r2.csv")
    assert "display_r2" not in main.columns
    assert (tmp_path / "diagnostic_pi_all_edge_corr2_by_property.csv").is_file()
    assert (tmp_path / "diagnostic_pi_all_edge_calibrated_r2_by_property.csv").is_file()
    assert (tmp_path / "diagnostic_pi_all_edge_transformed_r2_by_property.csv").is_file()
    assert (tmp_path / "diagnostic_pi_all_edge_fraction_postprocess_r2_by_property.csv").is_file()
    assert (tmp_path / "diagnostic_pi_all_edge_display_r2_by_property.csv").is_file()
    assert (tmp_path / "diagnostic_pi_all_edge_display_summary.json").is_file()

    corr = pd.read_csv(tmp_path / "diagnostic_pi_all_edge_corr2_by_property.csv")
    assert {"Mass_Flow", "Temp", "Frac_CH4"} <= set(corr["property"])
    transformed = pd.read_csv(tmp_path / "diagnostic_pi_all_edge_transformed_r2_by_property.csv")
    assert {"Mass_Flow", "Mole_Flow", "Vol_Flow"} <= set(transformed["property"])
    assert "Temp" not in set(transformed["property"])
    post = pd.read_csv(tmp_path / "diagnostic_pi_all_edge_fraction_postprocess_r2_by_property.csv")
    assert set(post["property"]).issubset({f"Frac_{sp}" for sp in species})
    display = pd.read_csv(tmp_path / "diagnostic_pi_all_edge_display_r2_by_property.csv")
    assert {"display_r2", "selected_metric", "warning"} <= set(display.columns)


def test_pi_all_edge_actual_vs_predicted_plots_are_optional(tmp_path):
    species = ["H2O", "H2", "CH4", "CO2", "CO", "O2", "N2"]
    frac = torch.full((3, len(species)), 1.0 / len(species), dtype=torch.float32)
    temp = torch.arange(300.0, 303.0).unsqueeze(-1)
    pres = torch.arange(10.0, 13.0).unsqueeze(-1)
    mass = torch.arange(1.0, 4.0).unsqueeze(-1)
    rho = torch.arange(2.0, 5.0).unsqueeze(-1)
    enthalpy = torch.arange(100.0, 103.0).unsqueeze(-1)
    mw = torch.tensor([DEFAULT_MOLECULAR_WEIGHT_G_PER_MOL[s] / 1000.0 for s in species])
    mole = mass / (frac * mw).sum(dim=-1, keepdim=True)
    volume = mass / rho
    columns = list(STREAM_EDGE_FEATURE_SLOTS) + ["Enthalpy", "Density"]
    target_by_name = {
        "Temp": temp,
        "Pres": pres,
        "Vol_Flow": volume,
        "Mole_Flow": mole,
        "Mass_Flow": mass,
        "Enthalpy": enthalpy,
        "Density": rho,
    }
    for idx, sp in enumerate(species):
        target_by_name[f"Frac_{sp}"] = frac[:, idx : idx + 1]
    true = torch.cat([target_by_name[name] for name in columns], dim=-1)
    outputs = {
        "T_pred": temp,
        "P_pred": pres,
        "frac_pred": frac,
        "mass_flow_pred": mass,
        "rho_pred": rho,
        "h_pred": enthalpy,
        "main_stream_pred": torch.cat([temp, pres, frac, mass], dim=-1),
    }
    data_cfg = SimpleNamespace(
        species_order=species,
        molecular_weight=DEFAULT_MOLECULAR_WEIGHT_G_PER_MOL,
        normalize_y_edge=False,
    )
    train_cfg = SimpleNamespace(
        eps=1.0e-8,
        molecular_weight_unit="g_per_mol",
        h_basis="mass_specific",
        save_pi_all_edge_actual_vs_pred_plots=True,
    )
    acc = PIAllEdgePropertyR2Accumulator(train_cfg=train_cfg, store_scatter_samples=True)
    acc.update_batch(
        outputs=outputs,
        targets_raw={"edge_stream": true},
        target_masks={"edge_stream": torch.ones(3)},
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=None,
        split_name="val",
    )
    acc.write_artifacts(tmp_path)
    plot_root = tmp_path / "all_edge_actual_vs_predicted" / "val"
    assert (tmp_path / "all_edge_actual_vs_predicted" / "manifest.json").is_file()
    assert (plot_root / "mass_flow.png").is_file()
    assert (plot_root / "mole_flow.png").is_file()
    assert (plot_root / "vol_flow.png").is_file()


def test_pi_all_edge_relevant_r2_filters_zero_flow_rows(tmp_path):
    species = ["H2O", "H2", "CH4", "CO2", "CO", "O2", "N2"]
    frac = torch.full((2, len(species)), 1.0 / len(species), dtype=torch.float32)
    temp = torch.full((2, 1), 300.0, dtype=torch.float32)
    pres = torch.full((2, 1), 10.0, dtype=torch.float32)
    pred_mass = torch.tensor([[100.0], [10.0]], dtype=torch.float32)
    true_mass = torch.tensor([[0.0], [10.0]], dtype=torch.float32)
    rho = torch.ones(2, 1, dtype=torch.float32)
    enthalpy = torch.ones(2, 1, dtype=torch.float32)
    columns = list(STREAM_EDGE_FEATURE_SLOTS) + ["Enthalpy", "Density"]
    target_by_name = {
        "Temp": temp,
        "Pres": pres,
        "Vol_Flow": true_mass,
        "Mole_Flow": true_mass,
        "Mass_Flow": true_mass,
        "Enthalpy": enthalpy,
        "Density": rho,
    }
    for idx, sp in enumerate(species):
        target_by_name[f"Frac_{sp}"] = frac[:, idx : idx + 1]
    true = torch.cat([target_by_name[name] for name in columns], dim=-1)
    outputs = {
        "T_pred": temp,
        "P_pred": pres,
        "frac_pred": frac,
        "mass_flow_pred": pred_mass,
        "rho_pred": rho,
        "h_pred": enthalpy,
        "main_stream_pred": torch.cat([temp, pres, frac, pred_mass], dim=-1),
    }
    data_cfg = SimpleNamespace(
        species_order=species,
        molecular_weight=DEFAULT_MOLECULAR_WEIGHT_G_PER_MOL,
        normalize_y_edge=False,
    )
    train_cfg = SimpleNamespace(
        eps=1.0e-8,
        molecular_weight_unit="g_per_mol",
        h_basis="mass_specific",
        metric_relevance_enabled=True,
        metric_relevance_flow_threshold=1.0e-8,
    )
    acc = PIAllEdgePropertyR2Accumulator(train_cfg=train_cfg)
    acc.update_batch(
        outputs=outputs,
        targets_raw={"edge_stream": true},
        target_masks={"edge_stream": torch.ones(2)},
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=None,
    )
    acc.write_artifacts(tmp_path)
    main = pd.read_csv(tmp_path / "pi_all_edge_property_r2.csv")
    main_mass = main[main["property_name"] == "Mass_Flow"].iloc[0]
    assert int(main_mass["n"]) == 2
    relevant = pd.read_csv(tmp_path / "pi_all_edge_property_r2_relevance_deprecated.csv")
    mass = relevant[relevant["property_name"] == "Mass_Flow"].iloc[0]
    assert int(mass["n"]) == 1
    assert mass["relevance_rule"] == "true>1e-08"


def test_pi_all_edge_fraction_metrics_exclude_zero_flow_rows_only(tmp_path):
    species = ["H2O", "H2", "CH4", "CO2", "CO", "O2", "N2"]
    true_frac = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.2],
            [0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1],
        ],
        dtype=torch.float32,
    )
    pred_frac = true_frac.clone()
    pred_frac[0] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    temp = torch.arange(300.0, 303.0).unsqueeze(-1)
    pres = torch.arange(10.0, 13.0).unsqueeze(-1)
    true_mass = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
    pred_mass = torch.tensor([[5.0], [1.0], [2.0]], dtype=torch.float32)
    rho = torch.ones(3, 1, dtype=torch.float32)
    enthalpy = torch.ones(3, 1, dtype=torch.float32)
    columns = list(STREAM_EDGE_FEATURE_SLOTS) + ["Enthalpy", "Density"]
    target_by_name = {
        "Temp": temp,
        "Pres": pres,
        "Vol_Flow": true_mass,
        "Mole_Flow": true_mass,
        "Mass_Flow": true_mass,
        "Enthalpy": enthalpy,
        "Density": rho,
    }
    for idx, sp in enumerate(species):
        target_by_name[f"Frac_{sp}"] = true_frac[:, idx : idx + 1]
    true = torch.cat([target_by_name[name] for name in columns], dim=-1)
    outputs = {
        "T_pred": temp,
        "P_pred": pres,
        "frac_pred": pred_frac,
        "mass_flow_pred": pred_mass,
        "rho_pred": rho,
        "h_pred": enthalpy,
        "main_stream_pred": torch.cat([temp, pres, pred_frac, pred_mass], dim=-1),
    }
    data_cfg = SimpleNamespace(
        species_order=species,
        molecular_weight=DEFAULT_MOLECULAR_WEIGHT_G_PER_MOL,
        normalize_y_edge=False,
    )
    train_cfg = SimpleNamespace(
        eps=1.0e-8,
        molecular_weight_unit="g_per_mol",
        h_basis="mass_specific",
        use_zero_flow_fraction_mask=True,
        zero_flow_fraction_mask_eps=1.0e-8,
    )
    acc = PIAllEdgePropertyR2Accumulator(
        train_cfg=train_cfg,
        store_scatter_samples=True,
    )
    acc.update_batch(
        outputs=outputs,
        targets_raw={"edge_stream": true},
        target_masks={"edge_stream": torch.ones(3)},
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=None,
        split_name="val",
    )
    acc.write_artifacts(tmp_path)

    rows = pd.read_csv(tmp_path / "pi_all_edge_property_r2.csv").set_index("property_name")
    assert int(rows.loc["Frac_H2O", "n"]) == 2
    assert rows.loc["Frac_H2O", "R2"] == pytest.approx(1.0)
    assert int(rows.loc["Mass_Flow", "n"]) == 3
    assert rows.loc["Mass_Flow", "R2"] < 1.0
    fraction_scatter = [
        row for row in acc.scatter_rows if row["property_name"] == "Frac_H2O"
    ]
    assert len(fraction_scatter) == 2
    assert acc.zero_flow_fraction_rows_seen == 3
    assert acc.zero_flow_fraction_rows_masked == 1

    payload = json.loads((tmp_path / "pi_all_edge_property_r2.json").read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    assert metadata["zero_flow_fraction_mask_enabled"] is True
    assert metadata["zero_flow_fraction_rows_masked"] == 1


def test_pi_all_edge_fraction_metrics_keep_zero_flow_rows_when_mask_disabled(tmp_path):
    species = ["H2O", "H2", "CH4", "CO2", "CO", "O2", "N2"]
    true_frac = torch.zeros(2, len(species), dtype=torch.float32)
    true_frac[1, 0] = 0.5
    pred_frac = true_frac.clone()
    pred_frac[0, 0] = 1.0
    temp = torch.ones(2, 1)
    pres = torch.ones(2, 1)
    mass = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    columns = list(STREAM_EDGE_FEATURE_SLOTS)
    target_by_name = {
        "Temp": temp,
        "Pres": pres,
        "Vol_Flow": mass,
        "Mole_Flow": mass,
        "Mass_Flow": mass,
    }
    for idx, sp in enumerate(species):
        target_by_name[f"Frac_{sp}"] = true_frac[:, idx : idx + 1]
    true = torch.cat([target_by_name[name] for name in columns], dim=-1)
    outputs = {
        "T_pred": temp,
        "P_pred": pres,
        "frac_pred": pred_frac,
        "mass_flow_pred": mass,
        "rho_pred": torch.ones(2, 1),
        "h_pred": torch.ones(2, 1),
        "main_stream_pred": torch.cat([temp, pres, pred_frac, mass], dim=-1),
    }
    data_cfg = SimpleNamespace(
        species_order=species,
        molecular_weight=DEFAULT_MOLECULAR_WEIGHT_G_PER_MOL,
        normalize_y_edge=False,
    )
    train_cfg = SimpleNamespace(
        eps=1.0e-8,
        molecular_weight_unit="g_per_mol",
        h_basis="mass_specific",
        use_zero_flow_fraction_mask=False,
    )
    acc = PIAllEdgePropertyR2Accumulator(train_cfg=train_cfg)
    acc.update_batch(
        outputs=outputs,
        targets_raw={"edge_stream": true},
        target_masks={"edge_stream": torch.ones(2)},
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=None,
    )
    acc.write_artifacts(tmp_path)

    rows = pd.read_csv(tmp_path / "pi_all_edge_property_r2.csv").set_index("property_name")
    assert int(rows.loc["Frac_H2O", "n"]) == 2
    assert rows.loc["Frac_H2O", "R2"] < 1.0


def test_pi_aux_rho_h_losses_follow_pi_main_loss_type():
    species = ["H2O", "H2", "CH4", "CO2", "CO", "O2", "N2"]
    columns = list(STREAM_EDGE_FEATURE_SLOTS) + ["Density", "Enthalpy"]
    frac = torch.tensor([[0.10, 0.20, 0.10, 0.20, 0.10, 0.10, 0.20]], dtype=torch.float32)
    temp = torch.tensor([[300.0]], dtype=torch.float32)
    pres = torch.tensor([[10.0]], dtype=torch.float32)
    mass = torch.tensor([[5.0]], dtype=torch.float32)
    main_pred = torch.cat([temp, pres, frac, mass], dim=-1)
    target_by_name = {
        "Temp": temp,
        "Pres": pres,
        "Vol_Flow": torch.tensor([[1.0]], dtype=torch.float32),
        "Mole_Flow": torch.tensor([[1.0]], dtype=torch.float32),
        "Mass_Flow": mass,
        "Density": torch.tensor([[2.0]], dtype=torch.float32),
        "Enthalpy": torch.tensor([[2.0]], dtype=torch.float32),
    }
    for i, sp in enumerate(species):
        target_by_name[f"Frac_{sp}"] = frac[:, i : i + 1]
    target = torch.cat([target_by_name[name] for name in columns], dim=-1)
    outputs = {
        "T_pred": temp,
        "P_pred": pres,
        "frac_pred": frac,
        "mass_flow_pred": mass,
        "rho_pred": torch.zeros_like(temp),
        "h_pred": torch.zeros_like(temp),
        "main_stream_pred": main_pred,
    }
    train_cfg = SimpleNamespace(
        pi_main_loss_type="smooth_l1",
        lambda_main=0.0,
        lambda_rho=1.0,
        lambda_h=1.0,
        use_rho_loss=True,
        use_h_loss=True,
        use_volume_loss=False,
        use_enthalpy_flow_loss=False,
        use_atom_balance_loss=False,
        use_energy_balance_loss=False,
        edge_weight_default=1.0,
        eps=1.0e-8,
        molecular_weight_unit="g_per_mol",
        h_basis="mass_specific",
        h_loss_normalized=False,
    )
    data_cfg = SimpleNamespace(
        species_order=species,
        molecular_weight=DEFAULT_MOLECULAR_WEIGHT_G_PER_MOL,
        normalize_y_edge=False,
    )

    loss, log = compute_single_edge_pinn_loss(
        outputs=outputs,
        targets={"edge_stream": target},
        target_masks={"edge_stream": torch.ones(1)},
        edge_id=0,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=None,
    )

    assert loss is not None
    assert float(loss) == 3.0
    assert log["loss_rho_before_weight"] == 1.5
    assert log["loss_h_before_weight"] == 1.5


def test_pi_target_normalization_keeps_fraction_targets_physical():
    columns = ["Temp", "Pres", "Frac_H2O", "Frac_H2", "Frac_CH4", "Frac_CO2", "Frac_CO", "Mass_Flow"]
    y = torch.tensor([[[300.0, 10.0, 0.2, 0.3, 0.1, 0.05, 0.01, 1000.0]]], dtype=torch.float32)
    normalizer = {
        "mean": torch.tensor([200.0, 5.0, 0.4, 0.2, 0.2, 0.1, 0.012, 500.0], dtype=torch.float32),
        "std": torch.tensor([50.0, 5.0, 0.2, 0.1, 0.1, 0.05, 0.027, 100.0], dtype=torch.float32),
        "columns": columns,
    }
    data_cfg = SimpleNamespace(normalize_y_edge=True)

    out = _normalize_target_if_needed(y, normalizer, data_cfg, columns)

    assert torch.allclose(out[..., 0], torch.tensor([[2.0]]))
    assert torch.allclose(out[..., 1], torch.tensor([[1.0]]))
    assert torch.allclose(out[..., 7], torch.tensor([[5.0]]))
    assert torch.allclose(out[..., 2:7], y[..., 2:7])


def test_pi_fraction_loss_normalizes_prediction_and_target_with_train_stats():
    species = ["H2O", "H2"]
    columns = ["Temp", "Pres", "Frac_H2O", "Frac_H2", "Mass_Flow"]
    frac = torch.tensor([[0.3, 0.7]], dtype=torch.float32, requires_grad=True)
    zeros = torch.zeros((1, 1), dtype=torch.float32)
    main_pred = torch.cat((zeros, zeros, frac, zeros), dim=-1)
    main_target = torch.tensor([[0.0, 0.0, 0.2, 0.8, 0.0]], dtype=torch.float32)
    outputs = {
        "T_pred": zeros,
        "P_pred": zeros,
        "frac_pred": frac,
        "mass_flow_pred": zeros,
        "rho_pred": zeros,
        "h_pred": zeros,
        "main_stream_pred": main_pred,
    }
    train_cfg = SimpleNamespace(
        pi_main_loss_type="smooth_l1",
        pi_normalize_fraction_loss=True,
        pi_fraction_loss_std_min=1.0e-3,
        lambda_main=1.0,
        lambda_rho=0.0,
        lambda_h=0.0,
        lambda_volume=0.0,
        lambda_enthalpy_flow=0.0,
        lambda_atom=0.0,
        lambda_energy=0.0,
        use_rho_loss=False,
        use_h_loss=False,
        use_volume_loss=False,
        use_enthalpy_flow_loss=False,
        use_atom_balance_loss=False,
        use_energy_balance_loss=False,
        edge_weight_default=1.0,
        eps=1.0e-8,
        molecular_weight_unit="g_per_mol",
        h_basis="mass_specific",
    )
    data_cfg = SimpleNamespace(
        species_order=species,
        molecular_weight={"H2O": 18.01528, "H2": 2.01588},
        normalize_y_edge=False,
    )
    normalizer = {
        "columns": columns,
        "mean": torch.tensor([0.0, 0.0, 0.1, 0.5, 0.0], dtype=torch.float32),
        "std": torch.tensor([1.0, 1.0, 0.1, 0.2, 1.0], dtype=torch.float32),
    }

    loss, log = compute_single_edge_pinn_loss(
        outputs=outputs,
        targets={"edge_stream": main_target},
        target_masks={"edge_stream": torch.ones(1)},
        edge_id=0,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )

    assert loss is not None
    assert torch.allclose(loss, torch.tensor(0.125), atol=1.0e-6)
    assert log["fraction_loss_normalized"] is True
    assert log["frac_pred_physical_min"] == pytest.approx(0.3)
    assert log["frac_pred_physical_max"] == pytest.approx(0.7)
    assert log["frac_target_physical_min"] == pytest.approx(0.2)
    assert log["frac_target_physical_max"] == pytest.approx(0.8)
    assert log["frac_loss_std_min"] == pytest.approx(0.1)

    loss.backward()
    assert frac.grad is not None
    assert torch.isfinite(frac.grad).all()
    assert torch.any(frac.grad != 0.0)


def _pi_fraction_mask_fixture(*, frac_target, mass_target, use_mask: bool):
    species = ["H2O", "H2"]
    columns = ["Temp", "Pres", "Frac_H2O", "Frac_H2", "Mass_Flow"]
    frac = torch.tensor([[0.3, 0.7]], dtype=torch.float32, requires_grad=True)
    zeros = torch.zeros((1, 1), dtype=torch.float32)
    main_pred = torch.cat((zeros, zeros, frac, torch.tensor([[float(mass_target)]], dtype=torch.float32)), dim=-1)
    main_target = torch.tensor([[0.0, 0.0, float(frac_target[0]), float(frac_target[1]), float(mass_target)]], dtype=torch.float32)
    outputs = {
        "T_pred": zeros,
        "P_pred": zeros,
        "frac_pred": frac,
        "mass_flow_pred": torch.tensor([[float(mass_target)]], dtype=torch.float32),
        "rho_pred": zeros,
        "h_pred": zeros,
        "main_stream_pred": main_pred,
    }
    train_cfg = SimpleNamespace(
        pi_main_loss_type="smooth_l1",
        pi_normalize_fraction_loss=True,
        pi_fraction_loss_std_min=1.0e-3,
        use_zero_flow_fraction_mask=use_mask,
        zero_flow_fraction_mask_eps=1.0e-8,
        lambda_main=1.0,
        lambda_rho=0.0,
        lambda_h=0.0,
        lambda_volume=0.0,
        lambda_enthalpy_flow=0.0,
        lambda_atom=0.0,
        lambda_energy=0.0,
        use_rho_loss=False,
        use_h_loss=False,
        use_volume_loss=False,
        use_enthalpy_flow_loss=False,
        use_atom_balance_loss=False,
        use_energy_balance_loss=False,
        edge_weight_default=1.0,
        eps=1.0e-8,
        molecular_weight_unit="g_per_mol",
        h_basis="mass_specific",
    )
    data_cfg = SimpleNamespace(
        species_order=species,
        molecular_weight={"H2O": 18.01528, "H2": 2.01588},
        normalize_y_edge=False,
    )
    normalizer = {
        "columns": columns,
        "mean": torch.tensor([0.0, 0.0, 0.1, 0.5, 0.0], dtype=torch.float32),
        "std": torch.tensor([1.0, 1.0, 0.1, 0.2, 1.0], dtype=torch.float32),
    }
    return outputs, main_target, train_cfg, data_cfg, normalizer, columns


def test_zero_flow_fraction_mask_excludes_undefined_fraction_rows():
    outputs, target, train_cfg, data_cfg, normalizer, columns = _pi_fraction_mask_fixture(
        frac_target=(0.0, 0.0),
        mass_target=0.0,
        use_mask=True,
    )

    loss, log = compute_single_edge_pinn_loss(
        outputs=outputs,
        targets={"edge_stream": target},
        target_masks={"edge_stream": torch.ones(1)},
        edge_id=0,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )

    assert loss is not None
    assert torch.isfinite(loss)
    assert torch.allclose(loss, torch.tensor(0.0))
    assert log["zero_flow_fraction_mask_enabled"] is True
    assert log["zero_flow_fraction_valid_rows"] == 0
    assert log["zero_flow_fraction_masked_rows"] == 1


def test_zero_flow_fraction_mask_keeps_positive_mass_rows():
    outputs, target, train_cfg, data_cfg, normalizer, columns = _pi_fraction_mask_fixture(
        frac_target=(0.2, 0.8),
        mass_target=1.0,
        use_mask=True,
    )

    loss, log = compute_single_edge_pinn_loss(
        outputs=outputs,
        targets={"edge_stream": target},
        target_masks={"edge_stream": torch.ones(1)},
        edge_id=0,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )

    assert loss is not None
    assert torch.allclose(loss, torch.tensor(0.125), atol=1.0e-6)
    assert log["zero_flow_fraction_valid_rows"] == 1
    assert log["zero_flow_fraction_masked_rows"] == 0


def test_zero_flow_fraction_mask_off_preserves_previous_fraction_loss():
    outputs, target, train_cfg, data_cfg, normalizer, columns = _pi_fraction_mask_fixture(
        frac_target=(0.0, 0.0),
        mass_target=0.0,
        use_mask=False,
    )

    loss, log = compute_single_edge_pinn_loss(
        outputs=outputs,
        targets={"edge_stream": target},
        target_masks={"edge_stream": torch.ones(1)},
        edge_id=0,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )

    assert loss is not None
    assert torch.isfinite(loss)
    assert float(loss) > 0.0
    assert log["zero_flow_fraction_mask_enabled"] is False


def test_mass_flow_log1p_scaler_uses_train_records_only():
    columns = ["Temp", "Pres", "Frac_H2O", "Frac_H2", "Mass_Flow"]
    train_records = [
        SimpleNamespace(
            graph=SimpleNamespace(
                edge_target_columns=columns,
                y_edge_true=[[0.0, 0.0, 0.5, 0.5, 0.0]],
                y_edge_mask=[1.0],
                process_id="train_a",
            )
        ),
        SimpleNamespace(
            graph=SimpleNamespace(
                edge_target_columns=columns,
                y_edge_true=[[0.0, 0.0, 0.5, 0.5, 9.0]],
                y_edge_mask=[1.0],
                process_id="train_b",
            )
        ),
    ]
    val_record_not_used = SimpleNamespace(
        graph=SimpleNamespace(
            edge_target_columns=columns,
            y_edge_true=[[0.0, 0.0, 0.5, 0.5, 9999.0]],
            y_edge_mask=[1.0],
            process_id="val",
        )
    )

    fit = compute_y_edge_log1p_column_scaler(
        train_records,
        column_name="Mass_Flow",
        stream_target_dim=len(columns),
    )

    expected_vals = torch.log1p(torch.tensor([0.0, 9.0], dtype=torch.float32))
    assert fit.count == 2
    assert fit.mean.item() == pytest.approx(float(expected_vals.mean()))
    assert fit.std.item() == pytest.approx(float(expected_vals.std(unbiased=False)))
    assert fit.mean.item() != pytest.approx(float(torch.log1p(torch.tensor([0.0, 9.0, 9999.0])).mean()))
    assert val_record_not_used.graph.process_id == "val"


def _pi_mass_log_fixture(
    *,
    use_mass_log: bool,
    pred_mass_norm: float = 2.0,
    target_mass_norm: float = 0.0,
    output_space: str = "raw_z",
    mass_log_weight: float = 1.0,
):
    species = ["H2O", "H2"]
    columns = ["Temp", "Pres", "Frac_H2O", "Frac_H2", "Mass_Flow", "Density", "Enthalpy"]
    frac = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    zeros = torch.zeros((1, 1), dtype=torch.float32)
    mass_pred = torch.tensor([[float(pred_mass_norm)]], dtype=torch.float32)
    mass_target = torch.tensor([[float(target_mass_norm)]], dtype=torch.float32)
    outputs = {
        "T_pred": zeros,
        "P_pred": zeros,
        "frac_pred": frac,
        "mass_flow_pred": mass_pred,
        "rho_pred": zeros,
        "h_pred": zeros,
        "main_stream_pred": torch.cat((zeros, zeros, frac, mass_pred), dim=-1),
    }
    target = torch.cat((zeros, zeros, frac, mass_target, zeros, zeros), dim=-1)
    train_cfg = SimpleNamespace(
        pi_main_loss_type="smooth_l1",
        pi_normalize_fraction_loss=False,
        use_zero_flow_fraction_mask=False,
        use_log1p_mass_flow_loss=use_mass_log,
        log1p_mass_flow_loss_std_min=1.0e-6,
        pi_mass_flow_output_space=output_space,
        mass_flow_log_loss_weight=mass_log_weight,
        lambda_main=1.0,
        lambda_rho=0.0,
        lambda_h=0.0,
        lambda_volume=0.0,
        lambda_enthalpy_flow=0.0,
        lambda_atom=0.0,
        lambda_energy=0.0,
        use_rho_loss=False,
        use_h_loss=False,
        use_volume_loss=False,
        use_enthalpy_flow_loss=False,
        use_atom_balance_loss=False,
        use_energy_balance_loss=False,
        edge_weight_default=1.0,
        eps=1.0e-8,
        molecular_weight_unit="g_per_mol",
        h_basis="mass_specific",
    )
    data_cfg = SimpleNamespace(
        species_order=species,
        molecular_weight={"H2O": 18.01528, "H2": 2.01588},
        normalize_y_edge=True,
    )
    normalizer = {
        "columns": columns,
        "mean": torch.tensor([0.0, 0.0, 0.0, 0.0, 5.0, 1.0, 100.0], dtype=torch.float32),
        "std": torch.tensor([1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 10.0], dtype=torch.float32),
        "mass_flow_log1p_mean": torch.tensor(math.log1p(5.0), dtype=torch.float32),
        "mass_flow_log1p_std": torch.tensor(1.0, dtype=torch.float32),
        "mass_flow_log1p_count": 2,
    }
    return outputs, target, train_cfg, data_cfg, normalizer, columns


def test_log1p_mass_flow_loss_is_finite_with_negative_physical_prediction():
    outputs, target, train_cfg, data_cfg, normalizer, columns = _pi_mass_log_fixture(
        use_mass_log=True,
        pred_mass_norm=-100.0,
        target_mass_norm=2.0,
    )

    loss, log = compute_single_edge_pinn_loss(
        outputs=outputs,
        targets={"edge_stream": target},
        target_masks={"edge_stream": torch.ones(1)},
        edge_id=0,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )

    assert loss is not None
    assert torch.isfinite(loss)
    assert log["mass_flow_log1p_loss_enabled"] is True
    assert log["mass_flow_log1p_pred_physical_min"] < 0.0


def test_log1p_mass_flow_loss_keeps_physical_metric_outputs():
    outputs, _target, train_cfg, data_cfg, normalizer, _columns = _pi_mass_log_fixture(
        use_mass_log=True,
        pred_mass_norm=2.0,
        target_mass_norm=0.0,
    )

    phys = build_pi_physical_outputs(outputs, train_cfg=train_cfg, data_cfg=data_cfg, normalizer=normalizer)

    assert phys["mass_flow"].item() == pytest.approx(9.0)


def test_log1p_mass_flow_loss_off_preserves_raw_mass_loss():
    outputs, target, train_cfg, data_cfg, normalizer, columns = _pi_mass_log_fixture(
        use_mass_log=False,
        pred_mass_norm=2.0,
        target_mass_norm=0.0,
    )

    loss, log = compute_single_edge_pinn_loss(
        outputs=outputs,
        targets={"edge_stream": target},
        target_masks={"edge_stream": torch.ones(1)},
        edge_id=0,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )

    assert loss is not None
    assert torch.allclose(loss, torch.tensor(0.3), atol=1.0e-6)
    assert log["mass_flow_log1p_loss_enabled"] is False


def test_direct_log_mass_flow_loss_compares_head_output_to_unscaled_log1p_target():
    expected_log = math.log1p(9.0)
    outputs, target, train_cfg, data_cfg, normalizer, columns = _pi_mass_log_fixture(
        use_mass_log=False,
        pred_mass_norm=expected_log,
        target_mass_norm=2.0,
        output_space="log1p",
    )

    loss, log = compute_single_edge_pinn_loss(
        outputs=outputs,
        targets={"edge_stream": target},
        target_masks={"edge_stream": torch.ones(1)},
        edge_id=0,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )

    assert loss is not None
    assert loss.item() == pytest.approx(0.0, abs=1.0e-7)
    assert log["mass_flow_direct_log_enabled"] is True
    assert log["mass_flow_direct_target_physical_mean"] == pytest.approx(9.0)
    assert log["mass_flow_direct_target_log_mean"] == pytest.approx(expected_log)
    assert log["mass_flow_direct_pred_log_mean"] == pytest.approx(expected_log)
    assert log["mass_flow_direct_pred_physical_mean"] == pytest.approx(9.0)


def test_direct_log_mass_flow_weight_scales_only_mass_loss_term():
    common = dict(
        use_mass_log=False,
        pred_mass_norm=0.0,
        target_mass_norm=2.0,
        output_space="log1p",
    )
    fixture_one = _pi_mass_log_fixture(**common, mass_log_weight=1.0)
    fixture_two = _pi_mass_log_fixture(**common, mass_log_weight=2.0)

    losses = []
    for outputs, target, train_cfg, data_cfg, normalizer, columns in (fixture_one, fixture_two):
        loss, _ = compute_single_edge_pinn_loss(
            outputs=outputs,
            targets={"edge_stream": target},
            target_masks={"edge_stream": torch.ones(1)},
            edge_id=0,
            train_cfg=train_cfg,
            data_cfg=data_cfg,
            edge_target_columns=columns,
            normalizer=normalizer,
        )
        assert loss is not None
        losses.append(loss.item())

    assert losses[1] == pytest.approx(2.0 * losses[0])


def test_direct_log_mass_flow_metric_and_physical_output_use_expm1():
    expected_mass = 9.0
    outputs, target, train_cfg, data_cfg, normalizer, columns = _pi_mass_log_fixture(
        use_mass_log=False,
        pred_mass_norm=math.log1p(expected_mass),
        target_mass_norm=2.0,
        output_space="log1p",
    )
    target_raw = target.clone()
    target_raw[:, columns.index("Mass_Flow")] = expected_mass

    phys = build_pi_physical_outputs(outputs, train_cfg=train_cfg, data_cfg=data_cfg, normalizer=normalizer)
    pred_metric, true_metric, _, names = extract_main_stream_metric_tensors(
        outputs=outputs,
        targets_raw={"edge_stream": target_raw},
        target_masks={"edge_stream": torch.ones(1)},
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )
    mass_idx = names.index("Mass_Flow")

    assert phys["mass_flow"].item() == pytest.approx(expected_mass)
    assert pred_metric[0, mass_idx].item() == pytest.approx(expected_mass)
    assert true_metric[0, mass_idx].item() == pytest.approx(expected_mass)


def test_direct_log_mass_flow_metric_stays_float32_for_large_amp_prediction():
    expected_mass = 87593.125
    outputs, target, train_cfg, data_cfg, normalizer, columns = _pi_mass_log_fixture(
        use_mass_log=False,
        pred_mass_norm=math.log1p(expected_mass),
        target_mass_norm=0.0,
        output_space="log1p",
    )
    outputs = {key: value.half() for key, value in outputs.items()}
    target_raw = target.clone()
    target_raw[:, columns.index("Mass_Flow")] = expected_mass

    pred_metric, _, _, names = extract_main_stream_metric_tensors(
        outputs=outputs,
        targets_raw={"edge_stream": target_raw},
        target_masks={"edge_stream": torch.ones(1)},
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )
    mass_idx = names.index("Mass_Flow")

    assert pred_metric.dtype == torch.float32
    assert torch.isfinite(pred_metric).all()
    assert pred_metric[0, mass_idx].item() == pytest.approx(expected_mass, rel=3.0e-3)


def test_direct_log_checkpoint_rejects_raw_z_semantics():
    train_cfg = SimpleNamespace(pi_mass_flow_output_space="log1p", use_log1p_mass_flow_loss=False)

    with pytest.raises(RuntimeError, match="Refusing to reinterpret"):
        assert_checkpoint_pi_mass_flow_compatible(
            {"pi_mass_flow_output_space": "raw_z"},
            train_cfg=train_cfg,
            checkpoint_path="old.pt",
        )

    assert_checkpoint_pi_mass_flow_compatible(
        {"pi_mass_flow_output_space": "log1p"},
        train_cfg=train_cfg,
        checkpoint_path="a4.pt",
    )


def test_pi_grouped_property_head_sigmoid_fraction_activation_does_not_softmax():
    torch.manual_seed(0)
    head = PIGroupedPropertyHead(
        input_dim=5,
        num_species=3,
        property_head_hidden_dim=4,
        property_head_num_layers=1,
        fraction_activation="sigmoid",
    )
    out = head(torch.randn(6, 5))
    frac = out["frac_pred"]

    assert torch.all(frac >= 0.0)
    assert torch.all(frac <= 1.0)
    assert not torch.allclose(frac.sum(dim=-1), torch.ones(frac.shape[0]), atol=1.0e-5)


def test_pi_grouped_property_head_softmax_fraction_temperature():
    cold = PIGroupedPropertyHead(
        input_dim=5,
        num_species=3,
        property_head_hidden_dim=4,
        property_head_num_layers=1,
        fraction_activation="softmax",
        fraction_temperature=0.5,
    )
    base = PIGroupedPropertyHead(
        input_dim=5,
        num_species=3,
        property_head_hidden_dim=4,
        property_head_num_layers=1,
        fraction_activation="softmax",
        fraction_temperature=1.0,
    )
    for head in (cold, base):
        final_linear = head.fraction_head[-1]
        with torch.no_grad():
            final_linear.weight.zero_()
            final_linear.bias.copy_(torch.tensor([0.0, 1.0, 2.0]))

    x = torch.randn(2, 5)
    cold_frac = cold(x)["frac_pred"]
    base_frac = base(x)["frac_pred"]

    assert torch.allclose(base_frac, torch.softmax(torch.tensor([[0.0, 1.0, 2.0]]), dim=-1).expand_as(base_frac))
    assert torch.allclose(cold_frac, torch.softmax(torch.tensor([[0.0, 2.0, 4.0]]), dim=-1).expand_as(cold_frac))
    assert torch.allclose(cold_frac.sum(dim=-1), torch.ones(cold_frac.shape[0]))
    assert torch.all(cold_frac[:, -1] > base_frac[:, -1])


def test_pi_grouped_property_head_relu_l1_fraction_activation():
    head = PIGroupedPropertyHead(
        input_dim=5,
        num_species=3,
        property_head_hidden_dim=4,
        property_head_num_layers=1,
        fraction_activation="relu_l1",
    )
    final_linear = head.fraction_head[-1]
    with torch.no_grad():
        final_linear.weight.zero_()
        final_linear.bias.copy_(torch.tensor([-1.0, 0.0, 2.0]))

    outputs = head(torch.randn(4, 5))
    frac = outputs["frac_pred"]

    assert torch.allclose(frac, torch.tensor([[0.0, 0.0, 1.0]]).expand_as(frac))
    assert torch.allclose(frac.sum(dim=-1), torch.ones(frac.shape[0]))
    assert torch.all(outputs["relu_l1_denominator"] == 2.0)
    assert not torch.any(outputs["relu_l1_all_zero_mask"])

    with torch.no_grad():
        final_linear.bias.fill_(-1.0)
    zero_outputs = head(torch.randn(2, 5))
    zero_case = zero_outputs["frac_pred"]
    assert torch.allclose(zero_case, torch.full_like(zero_case, 1.0 / 3.0))
    assert torch.all(zero_outputs["relu_l1_denominator"] == 0.0)
    assert torch.all(zero_outputs["relu_l1_all_zero_mask"])


def test_pi_grouped_property_head_softplus_l1_fraction_activation():
    torch.manual_seed(0)
    head = PIGroupedPropertyHead(
        input_dim=5,
        num_species=3,
        property_head_hidden_dim=4,
        property_head_num_layers=1,
        fraction_activation="softplus_l1",
    )
    inputs = torch.randn(6, 5)
    frac = head(inputs)["frac_pred"]

    assert torch.all(frac > 0.0)
    assert torch.allclose(frac.sum(dim=-1), torch.ones(frac.shape[0]), atol=1.0e-6)

    frac[:, 0].sum().backward()
    grad = head.fraction_head[-1].weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert torch.any(grad != 0.0)


def test_a7_pi_grouped_property_head_pure_relu_is_not_l1_normalized():
    head = PIGroupedPropertyHead(
        input_dim=5,
        num_species=3,
        property_head_hidden_dim=4,
        property_head_num_layers=1,
        fraction_activation="relu",
    )
    final_linear = head.fraction_head[-1]
    with torch.no_grad():
        final_linear.weight.zero_()
        final_linear.bias.copy_(torch.tensor([-1.0, 2.0, 3.0]))

    frac = head(torch.randn(2, 5))["frac_pred"]

    assert torch.all(frac >= 0.0)
    assert torch.allclose(frac, torch.tensor([[0.0, 2.0, 3.0]]).expand_as(frac))
    assert torch.all(frac.sum(dim=-1) == 5.0)
    assert torch.any(frac > 1.0)

    with torch.no_grad():
        final_linear.bias.fill_(-1.0)
    assert torch.count_nonzero(head(torch.randn(2, 5))["frac_pred"]) == 0


def test_a7_all_negative_relu_logits_have_finite_clr_loss_but_zero_gradient():
    head = PIGroupedPropertyHead(
        input_dim=5,
        num_species=3,
        property_head_hidden_dim=4,
        property_head_num_layers=1,
        fraction_activation="relu",
    )
    final_linear = head.fraction_head[-1]
    with torch.no_grad():
        final_linear.weight.zero_()
        final_linear.bias.fill_(-1.0)
    pred = head(torch.randn(2, 5))["frac_pred"]
    target = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32).expand_as(pred)

    loss = torch.nn.functional.smooth_l1_loss(
        _clr_transform(pred, 1.0e-6),
        _clr_transform(target, 1.0e-6),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert final_linear.bias.grad is not None
    assert torch.count_nonzero(final_linear.bias.grad) == 0


def test_a7_clr_is_invariant_to_common_fraction_scale():
    closed = torch.tensor([[0.1, 0.2, 0.7]], dtype=torch.float32)
    scaled = closed * 10.0

    assert torch.allclose(
        _clr_transform(closed, 1.0e-6),
        _clr_transform(scaled, 1.0e-6),
        atol=1.0e-6,
    )


def _enable_a6_clr(train_cfg):
    train_cfg.pi_fraction_loss_type = "clr"
    train_cfg.pi_fraction_clr_eps = 1.0e-6
    train_cfg.pi_fraction_clr_loss_weight = 0.5
    train_cfg.pi_normalize_fraction_loss = False
    train_cfg.use_zero_flow_fraction_mask = True
    train_cfg.zero_flow_fraction_mask_eps = 1.0e-8


def _enable_fraction_log(train_cfg):
    train_cfg.pi_fraction_loss_type = "log"
    train_cfg.pi_fraction_log_eps = 1.0e-6
    train_cfg.pi_fraction_log_loss_weight = 0.5
    train_cfg.pi_normalize_fraction_loss = False
    train_cfg.use_zero_flow_fraction_mask = True
    train_cfg.zero_flow_fraction_mask_eps = 1.0e-8


def test_a6_clr_transform_is_centered_finite_and_has_finite_gradient():
    frac = torch.tensor([[0.0, 0.2, 0.8]], dtype=torch.float32, requires_grad=True)

    clr = _clr_transform(frac, 1.0e-6)
    loss = torch.nn.functional.smooth_l1_loss(clr, torch.zeros_like(clr))
    loss.backward()

    assert torch.isfinite(clr).all()
    assert torch.allclose(clr.mean(dim=-1), torch.zeros(1), atol=1.0e-6)
    assert frac.grad is not None
    assert torch.isfinite(frac.grad).all()


def test_fraction_log_transform_is_finite_with_zero_and_has_finite_gradient():
    frac = torch.tensor([[0.0, 0.2, 0.8]], dtype=torch.float32, requires_grad=True)

    logged = _log_fraction_transform(frac, 1.0e-6)
    loss = torch.nn.functional.smooth_l1_loss(logged, torch.zeros_like(logged))
    loss.backward()

    assert torch.isfinite(logged).all()
    assert frac.grad is not None
    assert torch.isfinite(frac.grad).all()


def test_fraction_log_loss_is_separate_from_main_and_clr():
    outputs, target, train_cfg, data_cfg, normalizer, columns = _pi_mass_log_fixture(
        use_mass_log=False,
        pred_mass_norm=math.log1p(9.0),
        target_mass_norm=2.0,
        output_space="log1p",
    )
    _enable_fraction_log(train_cfg)
    frac = torch.tensor([[0.8, 0.2]], dtype=torch.float32, requires_grad=True)
    outputs["frac_pred"] = frac
    outputs["main_stream_pred"] = torch.cat(
        (outputs["T_pred"], outputs["P_pred"], frac, outputs["mass_flow_pred"]),
        dim=-1,
    )

    loss, log = compute_single_edge_pinn_loss(
        outputs=outputs,
        targets={"edge_stream": target},
        target_masks={"edge_stream": torch.ones(1)},
        edge_id=0,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )
    assert loss is not None
    loss.backward()

    assert log["fraction_loss_type"] == "log"
    assert log["fraction_log_loss_enabled"] is True
    assert log["fraction_clr_loss_enabled"] is False
    assert log["loss_fraction_log_before_weight"] > 0.0
    assert log["loss_fraction_clr_before_weight"] == pytest.approx(0.0)
    assert log["loss_main_before_weight"] == pytest.approx(0.0, abs=1.0e-7)
    assert loss.item() == pytest.approx(
        0.5 * log["loss_fraction_log_before_weight"],
        rel=1.0e-6,
    )
    assert frac.grad is not None
    assert torch.isfinite(frac.grad).all()


def test_fraction_log_loss_masks_zero_flow_rows_without_nan():
    outputs, target, train_cfg, data_cfg, normalizer, columns = _pi_mass_log_fixture(
        use_mass_log=False,
        pred_mass_norm=0.0,
        target_mass_norm=-2.5,
        output_space="log1p",
    )
    _enable_fraction_log(train_cfg)
    target[:, 2:4] = 0.0

    loss, log = compute_single_edge_pinn_loss(
        outputs=outputs,
        targets={"edge_stream": target},
        target_masks={"edge_stream": torch.ones(1)},
        edge_id=0,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )

    assert loss is not None
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1.0e-7)
    assert log["fraction_log_valid_sample_count"] == 0
    assert log["loss_fraction_log_before_weight"] == pytest.approx(0.0)
    assert log["zero_flow_fraction_masked_rows"] == 1


def test_fraction_log_loss_handles_zero_target_component():
    outputs, target, train_cfg, data_cfg, normalizer, columns = _pi_mass_log_fixture(
        use_mass_log=False,
        pred_mass_norm=math.log1p(9.0),
        target_mass_norm=2.0,
        output_space="log1p",
    )
    _enable_fraction_log(train_cfg)
    target[:, 2:4] = torch.tensor([[0.0, 1.0]])

    loss, log = compute_single_edge_pinn_loss(
        outputs=outputs,
        targets={"edge_stream": target},
        target_masks={"edge_stream": torch.ones(1)},
        edge_id=0,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )

    assert loss is not None
    assert torch.isfinite(loss)
    assert log["fraction_log_valid_sample_count"] == 1
    assert log["target_fraction_sum_mean"] == pytest.approx(1.0)


def test_sample_hybrid_target_edge_step_pi_config_loads_and_validates(tmp_path):
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        """
backward_mode: sample_hybrid_target_edge_step_pi
batch_size: 1
gradient_accumulation_steps: 1
sample_hybrid_target_edge_step_pi:
  enabled: true
  non_target_reduction: edge_group_macro_mean
  target_update_order: canonical_edge_index
  use_existing_target_edge_weight: false
  require_batch_size_one: true
  allow_duplicate_canonical_edge_rows: true
  log_sample_update_details: true
""".strip(),
        encoding="utf-8",
    )

    cfg = load_train_config(cfg_path)

    assert cfg.backward_mode == "sample_hybrid_target_edge_step_pi"
    assert cfg.batch_size == 1
    assert cfg.gradient_accumulation_steps == 1
    assert cfg.sample_hybrid_target_edge_step_pi.enabled is True
    assert cfg.sample_hybrid_target_edge_step_pi.target_update_order == "canonical_edge_index"
    assert cfg.sample_hybrid_target_edge_step_pi.use_existing_target_edge_weight is False


def test_sample_hybrid_rejects_unknown_target_update_order(tmp_path):
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(
        """
sample_hybrid_target_edge_step_pi:
  target_update_order: random_order
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target_update_order"):
        load_train_config(cfg_path)


def test_sample_hybrid_target_detection_uses_isclose_and_deterministic_order():
    weights = torch.tensor([1.0, 1.0 + 1.0e-8, 40.0, 1.0], dtype=torch.float32)

    assert _edge_group_is_target(edge_indices=[0, 1], edge_weight_vector=weights, edge_weight_default=1.0) is False
    assert _edge_group_is_target(edge_indices=[2], edge_weight_vector=weights, edge_weight_default=1.0) is True

    groups = [("P01:E010", [10]), ("P01:E002", [2]), ("P01:E001", [5])]
    by_id = _sort_edge_groups(groups, order="canonical_edge_id")
    by_index = _sort_edge_groups(groups, order="canonical_edge_index")

    assert [name for name, _ in by_id] == ["P01:E001", "P01:E002", "P01:E010"]
    assert [name for name, _ in by_index] == ["P01:E002", "P01:E001", "P01:E010"]


@pytest.mark.parametrize(("temp", "token"), [(0.5, "temp05"), (0.3, "temp03"), (0.2, "temp02")])
def test_fraction_log_strong_yaml_loads_temperature_and_preserves_direct_mass(temp, token):
    config = load_experiment_config(
        Path(
            "configs/experiment/pinn/"
            f"process_surrogate_edge_all_v3_pi_all_10pct_softmax_{token}_"
            "fraclog_massdirectlog_pinn_all_strong.yaml"
        )
    )

    assert config.model.fraction_activation == "softmax"
    assert config.model.fraction_temperature == pytest.approx(temp)
    assert config.train.pi_fraction_loss_type == "log"
    assert config.train.pi_fraction_log_eps == pytest.approx(1.0e-6)
    assert config.train.pi_fraction_log_loss_weight == pytest.approx(0.5)
    assert config.train.pi_mass_flow_output_space == "log1p"
    assert config.train.use_log1p_mass_flow_loss is False
    assert config.train.lambda_node_mass == pytest.approx(0.5)
    assert config.train.lambda_node_atom == pytest.approx(0.25)
    assert config.train.lambda_node_energy == pytest.approx(0.01)
    assert config.train.lambda_volume == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("filename", "log_weight"),
    [
        (
            "process_surrogate_edge_all_v3_pi_all_10pct_relu_l1_"
            "fraclog_massdirectlog_pinn_all_strong.yaml",
            0.5,
        ),
        (
            "process_surrogate_edge_all_v3_pi_all_10pct_relu_l1_"
            "fraclog01_massdirectlog_pinn_all_strong.yaml",
            0.1,
        ),
    ],
)
def test_relu_l1_fraction_log_strong_yaml_preserves_direct_mass_and_pinn_weights(
    filename,
    log_weight,
):
    path = Path("configs/experiment/pinn") / filename
    config = load_experiment_config(path)
    yaml_text = path.read_text(encoding="utf-8")

    assert config.model.fraction_activation == "relu_l1"
    assert "fraction_temperature:" not in yaml_text
    assert config.train.pi_fraction_loss_type == "log"
    assert config.train.pi_fraction_log_eps == pytest.approx(1.0e-6)
    assert config.train.pi_fraction_log_loss_weight == pytest.approx(log_weight)
    assert config.train.pi_normalize_fraction_loss is False
    assert config.train.use_zero_flow_fraction_mask is True
    assert config.train.pi_mass_flow_output_space == "log1p"
    assert config.train.use_log1p_mass_flow_loss is False
    assert config.train.lambda_node_mass == pytest.approx(0.5)
    assert config.train.lambda_node_atom == pytest.approx(0.25)
    assert config.train.lambda_node_energy == pytest.approx(0.01)
    assert config.train.lambda_volume == pytest.approx(0.10)


def test_relu_l1_all_zero_fallback_has_finite_log_loss_and_diagnostics():
    outputs, target, train_cfg, data_cfg, normalizer, columns = _pi_mass_log_fixture(
        use_mass_log=False,
        pred_mass_norm=math.log1p(9.0),
        target_mass_norm=2.0,
        output_space="log1p",
    )
    _enable_fraction_log(train_cfg)
    head = PIGroupedPropertyHead(
        input_dim=5,
        num_species=2,
        property_head_hidden_dim=4,
        property_head_num_layers=1,
        fraction_activation="relu_l1",
    )
    final_linear = head.fraction_head[-1]
    with torch.no_grad():
        final_linear.weight.zero_()
        final_linear.bias.fill_(-1.0)
    head_outputs = head(torch.zeros(1, 5))
    outputs["frac_pred"] = head_outputs["frac_pred"]
    outputs["relu_l1_denominator"] = head_outputs["relu_l1_denominator"]
    outputs["relu_l1_all_zero_mask"] = head_outputs["relu_l1_all_zero_mask"]
    outputs["main_stream_pred"] = torch.cat(
        (
            outputs["T_pred"],
            outputs["P_pred"],
            outputs["frac_pred"],
            outputs["mass_flow_pred"],
        ),
        dim=-1,
    )

    loss, log = compute_single_edge_pinn_loss(
        outputs=outputs,
        targets={"edge_stream": target},
        target_masks={"edge_stream": torch.ones(1)},
        edge_id=0,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )

    assert loss is not None
    assert torch.isfinite(loss)
    assert log["fraction_loss_type"] == "log"
    assert log["fraction_log_loss_enabled"] is True
    assert log["fraction_clr_loss_enabled"] is False
    assert log["relu_l1_all_zero_row_count"] == 1
    assert log["relu_l1_all_zero_row_ratio"] == pytest.approx(1.0)
    assert log["relu_l1_denominator_min"] == pytest.approx(0.0)
    assert log["relu_l1_denominator_mean"] == pytest.approx(0.0)
    assert log["pred_frac_sum_mean"] == pytest.approx(1.0)
    assert log["pred_frac_sum_min"] == pytest.approx(1.0)
    assert log["pred_frac_sum_max"] == pytest.approx(1.0)
    assert log["pred_frac_negative_count"] == 0
    assert log["pred_frac_nan_count"] == 0
    assert log["pred_frac_inf_count"] == 0
    assert log["loss_main_before_weight"] == pytest.approx(0.0, abs=1.0e-7)
    assert log["mass_flow_direct_log_enabled"] is True


def test_a6_clr_is_separate_from_main_and_does_not_use_fraction_zscore_stats():
    outputs, target, train_cfg, data_cfg, normalizer, columns = _pi_mass_log_fixture(
        use_mass_log=False,
        pred_mass_norm=math.log1p(9.0),
        target_mass_norm=2.0,
        output_space="log1p",
    )
    _enable_a6_clr(train_cfg)
    normalizer["mean"][2:4] = torch.nan
    normalizer["std"][2:4] = torch.nan

    loss, log = compute_single_edge_pinn_loss(
        outputs=outputs,
        targets={"edge_stream": target},
        target_masks={"edge_stream": torch.ones(1)},
        edge_id=0,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )

    assert loss is not None
    assert loss.item() == pytest.approx(0.0, abs=1.0e-7)
    assert log["fraction_loss_type"] == "clr"
    assert log["fraction_clr_loss_enabled"] is True
    assert log["fraction_loss_normalized"] is False
    assert log["fraction_clr_valid_sample_count"] == 1
    assert log["fraction_clr_pred_component_mean_abs_max"] < 1.0e-6
    assert log["mass_flow_direct_log_enabled"] is True


def test_a6_clr_fraction_loss_has_finite_gradient():
    outputs, target, train_cfg, data_cfg, normalizer, columns = _pi_mass_log_fixture(
        use_mass_log=False,
        pred_mass_norm=math.log1p(9.0),
        target_mass_norm=2.0,
        output_space="log1p",
    )
    _enable_a6_clr(train_cfg)
    frac = torch.tensor([[0.8, 0.2]], dtype=torch.float32, requires_grad=True)
    outputs["frac_pred"] = frac
    outputs["main_stream_pred"] = torch.cat(
        (outputs["T_pred"], outputs["P_pred"], frac, outputs["mass_flow_pred"]),
        dim=-1,
    )

    loss, log = compute_single_edge_pinn_loss(
        outputs=outputs,
        targets={"edge_stream": target},
        target_masks={"edge_stream": torch.ones(1)},
        edge_id=0,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )
    assert loss is not None
    loss.backward()

    assert loss.item() > 0.0
    assert log["loss_fraction_clr_before_weight"] > 0.0
    assert frac.grad is not None
    assert torch.isfinite(frac.grad).all()
    assert torch.any(frac.grad != 0.0)


def test_a6_zero_flow_all_invalid_fraction_rows_return_zero_without_nan():
    outputs, target, train_cfg, data_cfg, normalizer, columns = _pi_mass_log_fixture(
        use_mass_log=False,
        pred_mass_norm=0.0,
        target_mass_norm=-2.5,
        output_space="log1p",
    )
    _enable_a6_clr(train_cfg)
    train_cfg.pi_fraction_closure_loss_weight = 1.0
    target[:, 2:4] = 0.0

    loss, log = compute_single_edge_pinn_loss(
        outputs=outputs,
        targets={"edge_stream": target},
        target_masks={"edge_stream": torch.ones(1)},
        edge_id=0,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )

    assert loss is not None
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1.0e-7)
    assert log["fraction_clr_valid_sample_count"] == 0
    assert log["loss_fraction_clr_before_weight"] == pytest.approx(0.0)
    assert log["loss_fraction_closure_before_weight"] == pytest.approx(0.0)
    assert log["zero_flow_fraction_masked_rows"] == 1


def test_a6_yaml_loads_clr_softmax_and_direct_log_mass():
    config = load_experiment_config(
        Path(
            "configs/experiment/pinn/"
            "process_surrogate_edge_all_v3_pi_all_10pct_softmax_temp05_fracclr_massdirectlog.yaml"
        )
    )

    assert config.model.fraction_activation == "softmax"
    assert config.model.fraction_temperature == pytest.approx(0.5)
    assert config.train.pi_fraction_loss_type == "clr"
    assert config.train.pi_fraction_clr_loss_weight == pytest.approx(0.5)
    assert config.train.pi_normalize_fraction_loss is False
    assert config.train.use_zero_flow_fraction_mask is True
    assert config.train.pi_mass_flow_output_space == "log1p"
    assert config.train.use_log1p_mass_flow_loss is False


def test_a7_closure_penalty_is_optional_and_valid_row_only():
    outputs, target, train_cfg, data_cfg, normalizer, columns = _pi_mass_log_fixture(
        use_mass_log=False,
        pred_mass_norm=math.log1p(9.0),
        target_mass_norm=2.0,
        output_space="log1p",
    )
    _enable_a6_clr(train_cfg)
    frac = torch.tensor([[0.4, 0.4]], dtype=torch.float32)
    outputs["frac_pred"] = frac
    outputs["main_stream_pred"] = torch.cat(
        (outputs["T_pred"], outputs["P_pred"], frac, outputs["mass_flow_pred"]),
        dim=-1,
    )

    train_cfg.pi_fraction_closure_loss_weight = 0.0
    loss_off, log_off = compute_single_edge_pinn_loss(
        outputs=outputs,
        targets={"edge_stream": target},
        target_masks={"edge_stream": torch.ones(1)},
        edge_id=0,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )
    train_cfg.pi_fraction_closure_loss_weight = 1.0
    loss_on, log_on = compute_single_edge_pinn_loss(
        outputs=outputs,
        targets={"edge_stream": target},
        target_masks={"edge_stream": torch.ones(1)},
        edge_id=0,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )

    assert loss_off is not None and loss_on is not None
    assert loss_off.item() == pytest.approx(0.0, abs=1.0e-7)
    assert log_off["loss_fraction_closure_before_weight"] == pytest.approx(0.02)
    assert loss_on.item() == pytest.approx(0.02)
    assert log_on["weighted_loss_fraction_closure_before_edge_weight"] == pytest.approx(0.02)


def test_mass_flow_physical_aux_disabled_preserves_single_edge_loss_exactly():
    outputs, target, train_cfg, data_cfg, normalizer, columns = _pi_mass_log_fixture(
        use_mass_log=False,
        pred_mass_norm=math.log1p(9.0),
        target_mass_norm=2.0,
        output_space="log1p",
    )
    loss_without_key, _ = compute_single_edge_pinn_loss(
        outputs=outputs,
        targets={"edge_stream": target},
        target_masks={"edge_stream": torch.ones(1)},
        edge_id=0,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )
    train_cfg.mass_flow_physical_auxiliary = SimpleNamespace(enabled=False)
    loss_disabled, log_disabled = compute_single_edge_pinn_loss(
        outputs=outputs,
        targets={"edge_stream": target},
        target_masks={"edge_stream": torch.ones(1)},
        edge_id=0,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )

    assert loss_without_key is not None and loss_disabled is not None
    assert torch.equal(loss_without_key, loss_disabled)
    assert log_disabled["mass_physical_loss"] == pytest.approx(0.0)


def test_a7_raw_relu_fraction_is_used_unchanged_by_downstream_and_metric():
    outputs, target, train_cfg, data_cfg, normalizer, columns = _pi_mass_log_fixture(
        use_mass_log=False,
        pred_mass_norm=math.log1p(9.0),
        target_mass_norm=2.0,
        output_space="log1p",
    )
    _enable_a6_clr(train_cfg)
    raw_frac = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
    outputs["frac_pred"] = raw_frac
    outputs["main_stream_pred"] = torch.cat(
        (outputs["T_pred"], outputs["P_pred"], raw_frac, outputs["mass_flow_pred"]),
        dim=-1,
    )

    phys = build_pi_physical_outputs(outputs, train_cfg=train_cfg, data_cfg=data_cfg, normalizer=normalizer)
    scaled_outputs = dict(outputs)
    scaled_outputs["frac_pred"] = raw_frac * 2.0
    scaled_outputs["main_stream_pred"] = torch.cat(
        (
            outputs["T_pred"],
            outputs["P_pred"],
            scaled_outputs["frac_pred"],
            outputs["mass_flow_pred"],
        ),
        dim=-1,
    )
    phys_scaled = build_pi_physical_outputs(
        scaled_outputs,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        normalizer=normalizer,
    )
    target_raw = target.clone()
    target_raw[:, columns.index("Mass_Flow")] = 9.0
    pred_metric, _, _, names = extract_main_stream_metric_tensors(
        outputs=outputs,
        targets_raw={"edge_stream": target_raw},
        target_masks={"edge_stream": torch.ones(1)},
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )

    assert torch.equal(phys["frac"], raw_frac)
    assert torch.allclose(phys_scaled["mw_bar"], phys["mw_bar"] * 2.0)
    assert torch.allclose(phys_scaled["mole_flow"], phys["mole_flow"] / 2.0)
    frac_indices = [names.index("Frac_H2O"), names.index("Frac_H2")]
    assert torch.equal(pred_metric[:, frac_indices], raw_frac)


def test_a7_yaml_loads_pure_relu_clr_without_closure_and_direct_log_mass():
    config = load_experiment_config(
        Path(
            "configs/experiment/pinn/"
            "process_surrogate_edge_all_v3_pi_all_10pct_relu_fracclr_massdirectlog.yaml"
        )
    )

    assert config.model.fraction_activation == "relu"
    assert config.train.pi_fraction_loss_type == "clr"
    assert config.train.pi_fraction_clr_loss_weight == pytest.approx(1.0)
    assert config.train.pi_fraction_closure_loss_weight == pytest.approx(0.0)
    assert config.train.pi_normalize_fraction_loss is False
    assert config.train.use_zero_flow_fraction_mask is True
    assert config.train.pi_mass_flow_output_space == "log1p"
    assert config.train.use_log1p_mass_flow_loss is False

