import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from scripts.run_process_kfold_experiments import _best_target_mean_r2
from scripts.run_process_kfold_experiments import _best_monitor_metric
from scripts.train_process_surrogate import (
    _oper_normalizer_cache_path,
    _y_edge_scaler_cache_path,
)
from scripts.tune_optuna import (
    _objective_series_and_name,
    _optuna_trial_train_overrides,
    build_optuna_trial_report,
)


def test_optuna_target_mean_r2_controls_monitor_and_objective(tmp_path):
    train_overrides = _optuna_trial_train_overrides({}, optuna_objective="target_mean_r2")
    assert train_overrides["monitor_metric"] == "val_target_mean_r2"
    assert train_overrides["monitor_mode"] == "max"

    val = {"epoch": [1.0, 2.0], "val_target_mean_r2": [0.2, 0.7]}
    series, name = _objective_series_and_name(
        val,
        "edge_all",
        optuna_objective="target_mean_r2",
        w_h2=5.0,
        w_co2=5.0,
    )
    assert series == [0.2, 0.7]
    assert name == "optuna_objective_target_mean_r2"

    (tmp_path / "train_val_history.json").write_text(
        json.dumps({"val": val}), encoding="utf-8"
    )
    report = build_optuna_trial_report(
        tmp_path,
        preset="edge_all",
        w_h2=5.0,
        w_co2=5.0,
        optuna_objective="target_mean_r2",
    )
    assert report["objective"]["objective_metric_key"] == "val_target_mean_r2"
    assert report["objective"]["value"] == 0.7
    assert report["objective"]["best_epoch"] == 2.0


def test_optuna_target_property_mean_controls_monitor_and_objective(tmp_path):
    train_overrides = _optuna_trial_train_overrides(
        {}, optuna_objective="target_edge_property_mean_r2"
    )
    assert train_overrides["monitor_metric"] == "val_target_edge_property_mean_r2"
    assert train_overrides["monitor_mode"] == "max"

    val = {
        "epoch": [1.0, 2.0],
        "val_target_edge_property_mean_r2": [0.4, 0.9],
    }
    series, name = _objective_series_and_name(
        val,
        "edge_all",
        optuna_objective="target_edge_property_mean_r2",
        w_h2=5.0,
        w_co2=5.0,
    )
    assert series == [0.4, 0.9]
    assert name == "optuna_objective_target_edge_property_mean_r2"


def test_kfold_summary_uses_best_target_mean_r2(tmp_path):
    pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "val_target_mean_r2": [-1.0, 0.8, 0.3],
            "val_eval_primary_frac_r2_by_process": [0.9, 0.1, 0.2],
        }
    ).to_csv(tmp_path / "metrics_per_epoch.csv", index=False)
    assert _best_target_mean_r2(tmp_path) == (2, 0.8)


def test_kfold_summary_uses_best_target_property_mean_r2(tmp_path):
    pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "val_target_edge_property_mean_r2": [0.2, 0.91, 0.7],
            "val_target_mean_r2": [0.95, 0.1, 0.2],
        }
    ).to_csv(tmp_path / "metrics_per_epoch.csv", index=False)
    assert _best_monitor_metric(
        tmp_path, "val_target_edge_property_mean_r2"
    ) == (2, 0.91, "val_target_edge_property_mean_r2")


def test_all_process_10pct_config_points_to_materialized_dataset():
    import yaml

    project_root = Path(__file__).resolve().parents[1]
    config_path = (
        project_root
        / "configs/experiment/pinn/process_surrogate_edge_all_v3_pi_all_10pct.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    data = payload["overrides"]["data"]
    train = payload["overrides"]["train"]
    assert data["edge_all_processes"] == list(range(1, 11))
    assert data["train_data_path"].endswith("process_main_merged_all_processes_10pct.csv")
    assert data["train_split_manifest_path"].endswith("All/fold_01/train.csv")
    assert train["monitor_metric"] == "val_target_mean_r2"
    assert train["monitor_mode"] == "max"


def test_y_edge_scaler_cache_key_tracks_dataset_and_manifest(tmp_path):
    train_csv = tmp_path / "train.csv"
    manifest = tmp_path / "train_manifest.csv"
    train_csv.write_text("ID,process_id\n1,Process1\n", encoding="utf-8")
    manifest.write_text("merged_row_index\n0\n", encoding="utf-8")
    experiment = SimpleNamespace(
        project_root=tmp_path,
        data=SimpleNamespace(
            edge_all_processes=[1],
            stream_data_dir="streams",
            canonical_graph_spec_v3_dir="reference",
            edge_target_columns=["Temp", "Pres"],
        ),
    )
    first = _y_edge_scaler_cache_path(
        experiment=experiment,
        train_csv=train_csv,
        train_manifest_path=manifest,
        dataset_len=1,
        stream_target_dim=2,
    )
    manifest.write_text("merged_row_index\n0\n1\n", encoding="utf-8")
    second = _y_edge_scaler_cache_path(
        experiment=experiment,
        train_csv=train_csv,
        train_manifest_path=manifest,
        dataset_len=2,
        stream_target_dim=2,
    )
    assert first != second
    assert first.parent == tmp_path / "outputs/cache/y_edge_scalers"
    oper = _oper_normalizer_cache_path(
        experiment=experiment,
        train_csv=train_csv,
        train_manifest_path=manifest,
        dataset_len=2,
        stream_target_dim=2,
    )
    assert oper.parent == tmp_path / "outputs/cache/oper_normalizers"
    oper_other_output_dim = _oper_normalizer_cache_path(
        experiment=experiment,
        train_csv=train_csv,
        train_manifest_path=manifest,
        dataset_len=2,
        stream_target_dim=11,
    )
    assert oper == oper_other_output_dim
