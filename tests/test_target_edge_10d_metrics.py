from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

from process_graph.experiment.target_edge_10d_metrics import (
    TargetEdge10DAccumulator,
    build_process_edge_topology_audit,
    build_target_edge_10d_metrics,
    build_target_edge_r2_by_property,
    build_target_edge_r2_by_property_relevant,
    build_target_edge_resolution_audit,
    main_stream_property_names,
    target_edge_r2_by_property_scalars,
    validate_target_edge_10d_resolution,
    write_target_edge_10d_metric_artifacts,
    write_target_edge_property_metric_artifacts,
    extract_main_stream_metric_tensors,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V4_CSV = PROJECT_ROOT / "data/reference/v4/target_stream_targets.csv"
PROPS = main_stream_property_names(SimpleNamespace(species_order=["H2O", "H2", "CH4", "CO2", "CO", "O2", "N2"]))


def _cfg(**kw):
    base = dict(
        target_stream_loss_weight=5.0,
        target_stream_weight_conflict_policy="max",
        allow_yaml_only_target_streams=False,
        target_stream_loss_weights=[],
        target_metric_min_count=1,
        target_metric_sst_threshold=1.0e-6,
        target_metric_r2_floor=-0.07,
        target_mean_r2_excluded_properties=[],
    )
    base.update(kw)
    return SimpleNamespace(
        **base,
    )


def _metric_rows(split: str) -> pd.DataFrame:
    edges = [
        ("Process1", "P01_E015", "FUELGAS"),
        ("Process1", "P01_E021", "PROD"),
        ("Process1", "P01_E031", "RESTEAM"),
        ("Process7", "P07_E016", "PROD"),
        ("Process7", "P07_E017", "RE"),
    ]
    rows = []
    for edge_i, (pid, edge_id, stream) in enumerate(edges):
        for sample_i in range(3):
            for prop_i, prop in enumerate(PROPS):
                true = float(edge_i * 10 + sample_i + prop_i * 0.01)
                rows.append(
                    {
                        "split": split,
                        "process_id": pid,
                        "canonical_edge_id": edge_id,
                        "main_data_stream_key": stream,
                        "resolved_stream_name": stream,
                        "property_name": prop,
                        "y_true": true,
                        "y_pred": true + 0.1,
                        "mask": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def _metric_rows_from_v4(process_ids: list[int], split: str = "val") -> pd.DataFrame:
    targets = pd.read_csv(V4_CSV)
    rows = []
    for _, row in targets[targets["process_id"].isin(process_ids)].drop_duplicates(
        ["process_id", "canonical_answer_edge_id"]
    ).iterrows():
        pnum = int(row["process_id"])
        edge_id = str(row["canonical_answer_edge_id"])
        stream = str(row["main_data_stream_key"])
        for sample_i in range(3):
            for prop_i, prop in enumerate(PROPS):
                true = float(pnum * 100 + sample_i + prop_i * 0.01)
                rows.append(
                    {
                        "split": split,
                        "process_id": f"Process{pnum}",
                        "canonical_edge_id": edge_id,
                        "main_data_stream_key": stream,
                        "resolved_stream_name": stream,
                        "src_node": f"UNIT_{pnum}_{edge_id}",
                        "dst_node": "V_OUTPUT",
                        "edge_index": prop_i,
                        "stream_role": "output",
                        "property_name": prop,
                        "y_true": true,
                        "y_pred": true + 0.1,
                        "mask": 1.0,
                    }
                )
    return pd.DataFrame(rows)


def test_target_edge_10d_groups_species_rows_by_unique_target_stream():
    rows = _metric_rows("val")
    csv_df, payload, scalars = build_target_edge_10d_metrics(
        metric_rows=rows,
        train_cfg=_cfg(),
        target_stream_targets_path=V4_CSV,
    )

    p1 = csv_df[csv_df["process_id"] == 1]
    p7 = csv_df[csv_df["process_id"] == 7]
    assert set(p1["target_stream"]) == {"OUT_exhaust", "OUT_prod", "OUT_RESTEAM"}
    assert set(p7["target_stream"]) == {"OUT_PROD", "OUT_H2O"}
    assert p1[["target_stream", "resolved_edge_id"]].drop_duplicates().shape[0] == 3
    assert p7[["target_stream", "resolved_edge_id"]].drop_duplicates().shape[0] == 2
    assert p1[p1["target_stream"] == "OUT_prod"]["target_species_list"].iloc[0] == "H2,CO2"
    assert p7[p7["target_stream"] == "OUT_PROD"]["target_species_list"].iloc[0] == "H2,CO2"
    assert "H2O" in set(csv_df["target_species_list"])
    assert len(p1) == 3 * 10
    assert len(p7) == 2 * 10
    assert list(p1[p1["target_stream"] == "OUT_prod"]["property_name"]) == PROPS
    assert "val_target_edge_10d_r2_property_macro" in scalars
    assert payload["metadata"]["species_only_metrics_are_primary"] is False


def test_target_edge_10d_writer_creates_validation_and_test_csv(tmp_path):
    rows = pd.concat([_metric_rows("val"), _metric_rows("test")], ignore_index=True)
    scalars = write_target_edge_10d_metric_artifacts(
        out_dir=tmp_path,
        metric_rows=rows,
        train_cfg=_cfg(),
        target_stream_targets_path=V4_CSV,
    )
    path = tmp_path / "target_edge_10d_metrics.csv"
    assert path.is_file()
    out = pd.read_csv(path)
    required = {
        "split",
        "process_id",
        "target_stream",
        "resolved_edge_id",
        "resolved_stream_name",
        "target_species_list",
        "metric_scope",
        "property_name",
        "MAE",
        "RMSE",
        "R2",
        "n_samples",
        "true_mean",
        "true_std",
        "pred_mean",
        "pred_std",
        "orig_true_mean",
        "orig_true_std",
        "orig_pred_mean",
        "orig_pred_std",
        "inverse_mean_used",
        "inverse_std_used",
        "std_ratio",
        "r2_unstable",
    }
    assert required.issubset(out.columns)
    assert set(out["split"]) == {"val", "test"}
    assert set(out["metric_scope"]) == {"target_edge_10d"}
    assert out["resolved_edge_id"].astype(str).str.len().min() > 0
    assert "val_target_edge_10d_r2_edge_macro" in scalars
    assert "target_h2" not in "".join(out.columns)
    assert "tailgas_co2" not in "".join(out.columns)
    assert (tmp_path / "target_edge_resolution_audit.csv").is_file()
    assert (tmp_path / "process_edge_topology_audit.csv").is_file()
    assert (tmp_path / "target_edge_resolution_validation.json").is_file()
    target_path = tmp_path / "target_r2.csv"
    assert target_path.is_file()
    feature_path = tmp_path / "target_edge_feature_metrics.csv"
    assert feature_path.is_file()
    target_rows = pd.read_csv(target_path)
    feature_rows = pd.read_csv(feature_path)
    required_target = {
        "split",
        "process_id",
        "target_id",
        "canonical_edge_id",
        "target_stream",
        "feature_name",
        "n",
        "sst",
        "sse",
        "mae",
        "rmse",
        "r2",
        "used_in_mean",
        "exclude_reason",
        "true_mean",
        "true_std",
        "pred_mean",
        "pred_std",
        "inverse_mean_used",
        "inverse_std_used",
    }
    assert required_target.issubset(target_rows.columns)
    assert required_target.issubset(feature_rows.columns)
    assert len(target_rows) == len(out)
    assert len(feature_rows) == len(out)
    assert (tmp_path / "target_r2.json").is_file()
    assert (tmp_path / "target_edge_feature_metrics.json").is_file()
    assert (tmp_path / "target_edge_10d_feature_mapping.csv").is_file()
    assert (tmp_path / "target_edge_10d_feature_mapping.json").is_file()
    assert (tmp_path / "target_edge_r2_by_property.csv").is_file()
    assert "val_target_edge_r2_temp" in scalars


def test_target_edge_actual_vs_predicted_plots_are_optional_and_physical(tmp_path):
    rows = pd.concat([_metric_rows("val"), _metric_rows("test")], ignore_index=True)
    extra_flow_rows = []
    mass_rows = rows[rows["property_name"] == "Mass_Flow"]
    for prop, offset in (("Mole_Flow", 1000.0), ("Vol_Flow", 2000.0)):
        flow_rows = mass_rows.copy()
        flow_rows["property_name"] = prop
        flow_rows["y_true"] = pd.to_numeric(flow_rows["y_true"]) + offset
        flow_rows["y_pred"] = pd.to_numeric(flow_rows["y_pred"]) + offset
        extra_flow_rows.append(flow_rows)
    rows = pd.concat([rows, *extra_flow_rows], ignore_index=True)
    write_target_edge_10d_metric_artifacts(
        out_dir=tmp_path,
        metric_rows=rows,
        train_cfg=_cfg(save_target_edge_actual_vs_pred_plots=True),
        target_stream_targets_path=V4_CSV,
    )

    plot_root = tmp_path / "actual_vs_predicted"
    expected = {
        "temp.png",
        "pres.png",
        "frac_h2o.png",
        "frac_h2.png",
        "frac_ch4.png",
        "frac_co2.png",
        "frac_co.png",
        "frac_o2.png",
        "frac_n2.png",
        "mass_flow.png",
        "mass_flow_log1p.png",
    }
    assert {path.name for path in (plot_root / "val").glob("*.png")} == expected
    assert {path.name for path in (plot_root / "test").glob("*.png")} == expected
    assert (plot_root / "manifest.json").is_file()


def test_target_edge_r2_by_property_combines_edges_before_r2():
    rows = pd.DataFrame(
        [
            {"split": "val", "feature_name": "Temp", "n": 2, "true_mean": 0.5, "sst": 0.5, "sse": 0.0},
            {"split": "val", "feature_name": "Temp", "n": 2, "true_mean": 10.5, "sst": 0.5, "sse": 2.0},
        ]
    )
    out = build_target_edge_r2_by_property(rows)
    assert len(out) == 1
    assert out.iloc[0]["n"] == 4
    assert out.iloc[0]["SST"] == 101.0
    assert out.iloc[0]["R2"] == 1.0 - 2.0 / 101.0


def test_target_edge_relevant_property_r2_filters_fraction_samples():
    rows = pd.DataFrame(
        [
            {"split": "val", "property_name": "Frac_CO2", "y_true": 0.0, "y_pred": 0.9},
            {"split": "val", "property_name": "Frac_CO2", "y_true": 5.0e-4, "y_pred": 0.9},
            {"split": "val", "property_name": "Frac_CO2", "y_true": 0.2, "y_pred": 0.25},
            {"split": "val", "property_name": "Frac_CO2", "y_true": 0.4, "y_pred": 0.35},
            {"split": "val", "property_name": "Mass_Flow", "y_true": 0.0, "y_pred": 100.0},
            {"split": "val", "property_name": "Mass_Flow", "y_true": 10.0, "y_pred": 12.0},
            {"split": "val", "property_name": "Mass_Flow", "y_true": 20.0, "y_pred": 18.0},
        ]
    )

    relevant = build_target_edge_r2_by_property_relevant(
        rows,
        train_cfg=_cfg(
            metric_relevance_enabled=True,
            metric_relevance_fraction_threshold=1.0e-3,
            metric_relevance_flow_threshold=0.1,
        ),
    )

    frac = relevant[relevant["property_name"] == "Frac_CO2"].iloc[0]
    assert int(frac["n"]) == 2
    assert abs(float(frac["SSE"]) - 0.005) < 1.0e-12
    assert abs(float(frac["SST"]) - 0.02) < 1.0e-12
    assert abs(float(frac["R2"]) - 0.75) < 1.0e-12

    mass = relevant[relevant["property_name"] == "Mass_Flow"].iloc[0]
    assert int(mass["n"]) == 2
    assert mass["relevance_rule"] == "true>0.1"
    assert abs(float(mass["R2"]) - 0.84) < 1.0e-12


def test_target_edge_property_artifact_stays_raw_when_relevance_is_configured(tmp_path):
    rows = pd.DataFrame(
        [
            {
                "split": "val",
                "process_id": "Process1",
                "canonical_edge_id": "P01_E021",
                "main_data_stream_key": "PROD",
                "resolved_stream_name": "PROD",
                "property_name": "Frac_CO2",
                "y_true": 0.10,
                "y_pred": 0.12,
                "mask": 1.0,
                "edge_index": 0,
            },
            {
                "split": "val",
                "process_id": "Process1",
                "canonical_edge_id": "P01_E021",
                "main_data_stream_key": "PROD",
                "resolved_stream_name": "PROD",
                "property_name": "Frac_CO2",
                "y_true": 0.20,
                "y_pred": 0.22,
                "mask": 1.0,
                "edge_index": 1,
            },
            {
                "split": "val",
                "process_id": "Process1",
                "canonical_edge_id": "P01_NON_TARGET",
                "main_data_stream_key": "NON_TARGET",
                "resolved_stream_name": "NON_TARGET",
                "property_name": "Frac_CO2",
                "y_true": 0.80,
                "y_pred": 0.80,
                "mask": 1.0,
                "edge_index": 2,
            },
            {
                "split": "val",
                "process_id": "Process1",
                "canonical_edge_id": "P01_NON_TARGET",
                "main_data_stream_key": "NON_TARGET",
                "resolved_stream_name": "NON_TARGET",
                "property_name": "Frac_CO2",
                "y_true": 0.90,
                "y_pred": 0.90,
                "mask": 1.0,
                "edge_index": 3,
            },
        ]
    )

    write_target_edge_10d_metric_artifacts(
        out_dir=tmp_path,
        metric_rows=rows,
        train_cfg=_cfg(
            metric_relevance_enabled=True,
            metric_relevance_fraction_threshold=0.6,
            metric_relevance_flow_threshold=0.6,
        ),
        target_stream_targets_path=V4_CSV,
    )

    main = pd.read_csv(tmp_path / "target_edge_r2_by_property.csv")
    co2 = main[main["property_name"] == "Frac_CO2"].iloc[0]
    assert co2["metric_scope"] == "target_edges_by_property"
    assert int(co2["n"]) == 2

    raw = pd.read_csv(tmp_path / "target_edge_r2_by_property_raw.csv")
    raw_co2 = raw[raw["property_name"] == "Frac_CO2"].iloc[0]
    assert int(raw_co2["n"]) == 2
    assert not (tmp_path / "oracle_target_edge_threshold_sweep.csv").exists()


def test_target_edge_oracle_artifacts_are_separate_from_raw_metrics(tmp_path):
    rows = []
    props = ["Frac_H2O", "Frac_H2", "Frac_CH4", "Frac_CO2", "Frac_CO", "Frac_O2", "Frac_N2"]
    for sample_i in range(4):
        for prop_i, prop in enumerate(props):
            true = 0.05 * (prop_i + 1)
            pred = true * (0.8 + 0.05 * sample_i)
            rows.append(
                {
                    "split": "val",
                    "process_id": "Process1",
                    "canonical_edge_id": "P01_E021",
                    "main_data_stream_key": "PROD",
                    "resolved_stream_name": "PROD",
                    "property_name": prop,
                    "y_true": true,
                    "y_pred": pred,
                    "mask": 1.0,
                    "edge_index": sample_i,
                    "metric_row_group_id": sample_i,
                }
            )

    write_target_edge_10d_metric_artifacts(
        out_dir=tmp_path,
        metric_rows=pd.DataFrame(rows),
        train_cfg=_cfg(
            save_oracle_diagnostic_metrics=True,
            oracle_min_samples=2,
            oracle_threshold_candidates=[0.0, 0.1, 0.2],
        ),
        target_stream_targets_path=V4_CSV,
    )

    assert (tmp_path / "oracle_target_edge_threshold_sweep.csv").is_file()
    assert (tmp_path / "oracle_target_edge_calibrated_r2.csv").is_file()
    assert (tmp_path / "oracle_target_edge_fraction_postprocess_r2.csv").is_file()
    assert (tmp_path / "oracle_target_edge_summary.json").is_file()
    main = pd.read_csv(tmp_path / "target_edge_r2_by_property.csv")
    assert "best_threshold" not in main.columns
    oracle = pd.read_csv(tmp_path / "oracle_target_edge_threshold_sweep.csv")
    assert set(oracle["metric_type"]) == {"oracle_threshold_sweep"}
    assert {"threshold_type", "best_quantile"} <= set(oracle.columns)


def test_target_edge_diagnostic_display_artifacts_are_separate_from_raw_metrics(tmp_path):
    rows = []
    props = [
        "Frac_H2O",
        "Frac_H2",
        "Frac_CH4",
        "Frac_CO2",
        "Frac_CO",
        "Frac_O2",
        "Frac_N2",
        "Mass_Flow",
        "Temp",
    ]
    for sample_i in range(4):
        group_id = sample_i
        for prop_i, prop in enumerate(props):
            true = 0.05 * (prop_i + 1) + 0.01 * sample_i
            if prop == "Mass_Flow":
                true = 10.0 + sample_i * 5.0
            elif prop == "Temp":
                true = 300.0 + sample_i
            pred = true * 1.2 + 0.5
            rows.append(
                {
                    "split": "val",
                    "process_id": "Process1",
                    "canonical_edge_id": "P01_E021",
                    "main_data_stream_key": "PROD",
                    "resolved_stream_name": "PROD",
                    "property_name": prop,
                    "y_true": true,
                    "y_pred": pred,
                    "mask": 1.0,
                    "edge_index": sample_i,
                    "metric_row_group_id": group_id,
                }
            )

    write_target_edge_10d_metric_artifacts(
        out_dir=tmp_path,
        metric_rows=pd.DataFrame(rows),
        train_cfg=_cfg(save_diagnostic_display_metrics=True, diagnostic_min_samples=2),
        target_stream_targets_path=V4_CSV,
    )

    main = pd.read_csv(tmp_path / "target_edge_r2_by_property.csv")
    assert "display_r2" not in main.columns
    assert (tmp_path / "diagnostic_target_edge_corr2_by_property.csv").is_file()
    assert (tmp_path / "diagnostic_target_edge_calibrated_r2_by_property.csv").is_file()
    assert (tmp_path / "diagnostic_target_edge_transformed_r2_by_property.csv").is_file()
    assert (tmp_path / "diagnostic_target_edge_fraction_postprocess_r2_by_property.csv").is_file()
    assert (tmp_path / "diagnostic_target_edge_display_r2_by_property.csv").is_file()
    assert (tmp_path / "diagnostic_target_edge_display_summary.json").is_file()

    corr = pd.read_csv(tmp_path / "diagnostic_target_edge_corr2_by_property.csv")
    assert {"Mass_Flow", "Temp", "Frac_CH4"} <= set(corr["property"])
    transformed = pd.read_csv(tmp_path / "diagnostic_target_edge_transformed_r2_by_property.csv")
    assert "Mass_Flow" in set(transformed["property"])
    assert "Temp" not in set(transformed["property"])
    post = pd.read_csv(tmp_path / "diagnostic_target_edge_fraction_postprocess_r2_by_property.csv")
    assert set(post["property"]).issubset({p for p in props if p.startswith("Frac_")})
    display = pd.read_csv(tmp_path / "diagnostic_target_edge_display_r2_by_property.csv")
    assert {"display_r2", "selected_metric", "warning"} <= set(display.columns)
    assert "calibrated_r2" in set(display["selected_metric"]) or "corr2" in set(display["selected_metric"])


def test_target_edge_10d_accumulator_one_edge_ten_property_rows():
    acc = TargetEdge10DAccumulator(train_cfg=_cfg(), target_stream_targets_path=V4_CSV)
    export = SimpleNamespace(
        process_id=["Process1"],
        canonical_edge_id=["P01_E021"],
        main_data_stream_key=["PROD"],
    )
    pred = torch.ones(1, len(PROPS))
    true = torch.zeros(1, len(PROPS))
    mask = torch.ones(1, len(PROPS))
    acc.update_batch(
        export=export,
        pred_main=pred,
        true_main=true,
        mask_main=mask,
        property_names=PROPS,
        split_name="val",
    )
    csv_df, _, _ = acc.finalize()
    assert len(csv_df) == 10
    assert csv_df["target_stream"].iloc[0] == "OUT_prod"
    assert csv_df["target_species_list"].iloc[0] == "H2,CO2"


def test_target_edge_metric_does_not_inverse_softmax_fraction_predictions():
    species = ["H2O", "H2", "CH4", "CO2", "CO", "O2", "N2"]
    data_cfg = SimpleNamespace(species_order=species, normalize_y_edge=True)
    names = main_stream_property_names(data_cfg)
    columns = ["Temp", "Pres", "Vol_Flow", "Mole_Flow", "Mass_Flow"] + [f"Frac_{s}" for s in species]
    columns += ["Enthalpy", "Density"]
    frac = torch.tensor([[0.20, 0.30, 0.10, 0.05, 0.01, 0.04, 0.30]], dtype=torch.float32)
    pred = torch.cat(
        [
            torch.tensor([[2.0, 1.0]], dtype=torch.float32),
            frac,
            torch.tensor([[5.0]], dtype=torch.float32),
        ],
        dim=-1,
    )
    y_raw = torch.zeros(1, len(columns), dtype=torch.float32)
    for i, name in enumerate(names):
        y_raw[0, columns.index(name)] = pred[0, i]
    normalizer = {
        "columns": columns,
        "mean": torch.tensor([200.0, 5.0, 0.0, 0.0, 500.0, 0.4, 0.2, 0.2, 0.1, 0.012, 0.02, 0.1, 0.0, 0.0]),
        "std": torch.tensor([50.0, 5.0, 1.0, 1.0, 100.0, 0.2, 0.1, 0.1, 0.05, 0.027, 0.05, 0.2, 1.0, 1.0]),
    }
    outputs = {
        "main_stream_pred": pred,
        "frac_pred": frac,
    }

    pred_metric, _, _, prop_names = extract_main_stream_metric_tensors(
        outputs=outputs,
        targets_raw={"edge_stream": y_raw},
        target_masks={"edge_stream": torch.ones(1)},
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=normalizer,
    )

    frac_start = prop_names.index("Frac_H2O")
    assert torch.allclose(pred_metric[:, frac_start : frac_start + len(species)], frac)
    assert torch.allclose(pred_metric[:, 0], torch.tensor([300.0]))
    assert torch.allclose(pred_metric[:, 1], torch.tensor([10.0]))
    assert torch.allclose(pred_metric[:, -1], torch.tensor([1000.0]))


def test_target_edge_metric_keeps_12d_flow_predictions():
    species = ["H2O", "H2", "CH4", "CO2", "CO", "O2", "N2"]
    data_cfg = SimpleNamespace(species_order=species, normalize_y_edge=False)
    frac = torch.tensor([[0.20, 0.30, 0.10, 0.05, 0.01, 0.04, 0.30]], dtype=torch.float32)
    pred = torch.cat(
        [
            torch.tensor([[300.0, 10.0]], dtype=torch.float32),
            frac,
            torch.tensor([[1000.0, 50.0, 25.0]], dtype=torch.float32),
        ],
        dim=-1,
    )
    columns = ["Temp", "Pres", "Vol_Flow", "Mole_Flow", "Mass_Flow"] + [f"Frac_{s}" for s in species]
    columns += ["Enthalpy", "Density"]
    y_raw = torch.zeros(1, len(columns), dtype=torch.float32)
    expected = {
        "Temp": 300.0,
        "Pres": 10.0,
        "Mass_Flow": 1000.0,
        "Mole_Flow": 50.0,
        "Vol_Flow": 25.0,
    }
    for name, value in expected.items():
        y_raw[0, columns.index(name)] = value
    for idx, sp in enumerate(species):
        y_raw[0, columns.index(f"Frac_{sp}")] = frac[0, idx]

    pred_metric, true_metric, _, prop_names = extract_main_stream_metric_tensors(
        outputs={"main_stream_pred": pred, "frac_pred": frac},
        targets_raw={"edge_stream": y_raw},
        target_masks={"edge_stream": torch.ones(1)},
        data_cfg=data_cfg,
        edge_target_columns=columns,
        normalizer=None,
    )

    assert prop_names == ["Temp", "Pres"] + [f"Frac_{s}" for s in species] + [
        "Mass_Flow",
        "Mole_Flow",
        "Vol_Flow",
    ]
    for name, value in expected.items():
        idx = prop_names.index(name)
        assert torch.allclose(pred_metric[:, idx], torch.tensor([value]))
        assert torch.allclose(true_metric[:, idx], torch.tensor([value]))


def test_target_mean_r2_uses_configured_inclusion_rules():
    rows = []
    specs = [
        ("Temp", [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]),
        ("Pres", [10.0, 10.0, 10.0, 10.0], [9.0, 9.0, 9.0, 9.0]),
        ("Mass_Flow", [0.0, 1.0, 2.0, 3.0], [10.0, 10.0, 10.0, 10.0]),
        ("Frac_H2", [0.0, 1.0], [0.0, 1.0]),
    ]
    for prop, true_vals, pred_vals in specs:
        for i, (yt, yp) in enumerate(zip(true_vals, pred_vals)):
            rows.append(
                {
                    "split": "val",
                    "process_id": "Process1",
                    "canonical_edge_id": "P01_E021",
                    "main_data_stream_key": "PROD",
                    "resolved_stream_name": "PROD",
                    "property_name": prop,
                    "y_true": yt,
                    "y_pred": yp,
                    "mask": 1.0,
                    "edge_index": i,
                }
            )
    _, payload, scalars = build_target_edge_10d_metrics(
        metric_rows=pd.DataFrame(rows),
        train_cfg=_cfg(target_metric_min_count=4, target_metric_sst_threshold=1.0e-3, target_metric_r2_floor=-1.0),
        target_stream_targets_path=V4_CSV,
    )
    target_rows = pd.DataFrame(payload["target_r2"])
    assert set(target_rows["feature_name"]) == {"Temp", "Pres", "Mass_Flow", "Frac_H2"}
    assert bool(target_rows[target_rows["feature_name"] == "Temp"].iloc[0]["target_mean_r2_member"]) is True
    assert bool(target_rows[target_rows["feature_name"] == "Pres"].iloc[0]["target_mean_r2_member"]) is False
    assert bool(target_rows[target_rows["feature_name"] == "Mass_Flow"].iloc[0]["target_mean_r2_member"]) is False
    assert bool(target_rows[target_rows["feature_name"] == "Frac_H2"].iloc[0]["target_mean_r2_member"]) is False
    assert scalars["val_target_mean_r2"] == 1.0
    assert scalars["target_mean_r2"] == 1.0
    assert scalars["val_target_edge_10d_mae_property_macro"] == 0.0
    assert scalars["val_target_edge_10d_rmse_property_macro"] == 0.0
    assert scalars["val_target_edge_10d_r2_property_macro"] == 1.0
    assert scalars["val_target_edge_10d_r2_edge_macro"] == 1.0
    assert scalars["val_target_edge_10d_r2_flatten"] == 1.0


def test_target_mean_r2_can_exclude_property_without_removing_its_metric(tmp_path):
    rows = []
    for prop, pred_vals in (
        ("Temp", [0.0, 1.0, 2.0, 3.0]),
        ("Vol_Flow", [3.0, 2.0, 1.0, 0.0]),
    ):
        for i, (yt, yp) in enumerate(zip([0.0, 1.0, 2.0, 3.0], pred_vals)):
            rows.append(
                {
                    "split": "val",
                    "process_id": "Process1",
                    "canonical_edge_id": "P01_E021",
                    "main_data_stream_key": "PROD",
                    "resolved_stream_name": "PROD",
                    "property_name": prop,
                    "y_true": yt,
                    "y_pred": yp,
                    "mask": 1.0,
                    "edge_index": i,
                }
            )
    csv_df, payload, scalars = build_target_edge_10d_metrics(
        metric_rows=pd.DataFrame(rows),
        train_cfg=_cfg(target_mean_r2_excluded_properties=["Vol_Flow"]),
        target_stream_targets_path=V4_CSV,
    )
    assert set(csv_df["property_name"]) == {"Temp", "Vol_Flow"}
    target_rows = pd.DataFrame(payload["target_r2"])
    vol_row = target_rows[target_rows["feature_name"] == "Vol_Flow"].iloc[0]
    assert bool(vol_row["target_mean_r2_member"]) is False
    assert "excluded_property" in vol_row["target_mean_r2_exclusion_reason"].split(",")
    assert scalars["val_target_mean_r2"] == 1.0
    written_scalars = write_target_edge_10d_metric_artifacts(
        out_dir=tmp_path,
        metric_rows=pd.DataFrame(rows),
        train_cfg=_cfg(target_mean_r2_excluded_properties=["Vol_Flow"]),
        target_stream_targets_path=V4_CSV,
    )
    assert written_scalars["val_target_mean_r2"] == 1.0


def test_property_metrics_do_not_overwrite_direct_target_mean_r2(tmp_path):
    direct_rows = _metric_rows_from_v4([2])
    property_rows = direct_rows[direct_rows["property_name"].isin(["Temp", "Pres"])].copy()
    property_rows["property_name"] = property_rows["property_name"].map(
        {"Temp": "Density", "Pres": "Enthalpy"}
    )
    property_rows["y_pred"] = property_rows["y_true"] + 0.001

    direct_scalars = write_target_edge_10d_metric_artifacts(
        out_dir=tmp_path,
        metric_rows=direct_rows,
        train_cfg=_cfg(),
        target_stream_targets_path=V4_CSV,
    )
    property_scalars = write_target_edge_property_metric_artifacts(
        out_dir=tmp_path,
        metric_rows=property_rows,
        train_cfg=_cfg(),
        target_stream_targets_path=V4_CSV,
    )
    combined = dict(direct_scalars)
    combined.update(property_scalars)

    assert "val_target_mean_r2" not in property_scalars
    assert "target_mean_r2" not in property_scalars
    assert "val_target_edge_property_mean_r2" not in property_scalars
    assert "val_target_edge_derived_property_mean_r2" in property_scalars
    assert combined["val_target_mean_r2"] == direct_scalars["val_target_mean_r2"]
    assert combined["val_target_edge_property_mean_r2"] == direct_scalars[
        "val_target_edge_property_mean_r2"
    ]


def test_target_edge_property_mean_r2_uses_exactly_the_10_main_properties():
    rows = pd.DataFrame(
        [
            {"split": "val", "property_name": name, "R2": 1.0}
            for name in (
                "Temp", "Pres", "Mass_Flow", "Frac_H2O", "Frac_H2", "Frac_CH4",
                "Frac_CO2", "Frac_CO", "Frac_O2", "Frac_N2",
            )
        ]
        + [
            {"split": "val", "property_name": "Mole_Flow", "R2": -100.0},
            {"split": "val", "property_name": "Vol_Flow", "R2": -100.0},
        ]
    )

    scalars = target_edge_r2_by_property_scalars(rows, train_cfg=_cfg())

    assert scalars["val_target_edge_property_mean_r2"] == 1.0


def test_constant_target_property_r2_is_high():
    rows = pd.DataFrame(
        [
            {
                "split": "val",
                "property_name": "Frac_O2",
                "n": 100,
                "true_mean": 0.0,
                "sst": 0.0,
                "sse": 25.0,
            }
        ]
    )

    result = build_target_edge_r2_by_property(rows).iloc[0]

    assert result["R2"] == 0.999
    assert "r2_constant_target_substitute" not in result.index


def test_target_edge_resolution_audit_expected_processes():
    cases = [
        ([1], 3, 30, {"OUT_prod": "H2,CO2", "OUT_RESTEAM": "H2O"}),
        ([2], 2, 20, {"OUT_PROD": "H2", "OUT_EXHAUST": "CO2"}),
        ([6], 4, 40, {"OUT_PROD": "H2", "OUT_H2O": "H2O", "OUT_CO2": "CO2", "OUT_EXHAUST": "CO2"}),
        ([7], 2, 20, {"OUT_PROD": "H2,CO2", "OUT_H2O": "H2O"}),
    ]
    for process_ids, expected_edges, expected_rows, species_by_stream in cases:
        rows = _metric_rows_from_v4(process_ids)
        csv_df, payload, _ = build_target_edge_10d_metrics(
            metric_rows=rows,
            train_cfg=_cfg(),
            target_stream_targets_path=V4_CSV,
        )
        audit = build_target_edge_resolution_audit(
            metric_rows=rows,
            train_cfg=_cfg(),
            target_stream_targets_path=V4_CSV,
        )
        topology = build_process_edge_topology_audit(metric_rows=rows, resolution_audit=audit)
        validation = validate_target_edge_10d_resolution(
            metrics_df=csv_df,
            resolution_audit=audit,
            topology_audit=topology,
            property_names=payload["metadata"]["property_names"],
        )
        assert validation["ok"], validation
        assert len(audit) == expected_edges
        assert len(csv_df[csv_df["split"] == "val"]) == expected_rows
        assert audit["resolved_edge_id"].astype(str).str.len().min() > 0
        assert set(audit["resolution_status"]) == {"resolved"}
        for stream, species in species_by_stream.items():
            sub = audit[audit["target_stream"] == stream]
            assert not sub.empty
            assert sub.iloc[0]["target_species_list"] == species
        if process_ids == [6]:
            ex = audit[audit["target_stream"] == "OUT_EXHAUST"].iloc[0]
            assert ex["resolved_edge_id"] == "P06_E013"
            assert str(ex["resolved_stream_name"]) == "17"
            assert "canonical stream 17" in str(ex["resolution_warning"])


def test_target_edge_resolution_audit_all_processes_28_edges_280_rows():
    rows = _metric_rows_from_v4(list(range(1, 11)))
    csv_df, payload, _ = build_target_edge_10d_metrics(
        metric_rows=rows,
        train_cfg=_cfg(),
        target_stream_targets_path=V4_CSV,
    )
    audit = build_target_edge_resolution_audit(
        metric_rows=rows,
        train_cfg=_cfg(),
        target_stream_targets_path=V4_CSV,
    )
    topology = build_process_edge_topology_audit(metric_rows=rows, resolution_audit=audit)
    validation = validate_target_edge_10d_resolution(
        metrics_df=csv_df,
        resolution_audit=audit,
        topology_audit=topology,
        property_names=payload["metadata"]["property_names"],
    )
    assert validation["ok"], validation
    assert len(audit) == 28
    assert len(csv_df[csv_df["split"] == "val"]) == 280
    assert csv_df["property_name"].nunique() == 10
    assert set(audit["resolution_status"]) == {"resolved"}
    assert audit["target_species_list"].astype(str).str.contains("H2O").any()

