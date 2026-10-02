from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_final_baseline_experiments import (  # noqa: E402
    _aggregate_persisted_runs,
    _artifact_target_edge_property_mean_r2,
    _refresh_joint_property_artifacts,
)


def test_artifact_metric_uses_pooled_property_rows_and_exact_10d_set(tmp_path: Path):
    pd.DataFrame(
        [
            {"property_name": "Temp", "R2": 0.8},
            {"property_name": "Pres", "R2": 0.6},
            {"property_name": "Frac_H2", "R2": float("nan")},
            {"property_name": "Vol_Flow", "R2": -100.0},
        ]
    ).to_csv(tmp_path / "test_property_metrics.csv", index=False)

    assert _artifact_target_edge_property_mean_r2(tmp_path, "test") == 0.7
    payload = json.loads((tmp_path / "test_metrics.json").read_text(encoding="utf-8"))
    assert len(payload["target_edge_property_r2"]) == 10
    assert payload["target_edge_property_r2_valid_count"] == 2


def test_existing_constant_property_artifact_is_migrated_to_zero_r2(tmp_path: Path):
    pd.DataFrame([
        {
            "property_name": "Frac_O2", "R2": float("nan"),
            "R2_status": "constant_target", "R2_n_valid": 20,
            "R2_SST": 0.0, "R2_SSE": 0.0,
        }
    ]).to_csv(tmp_path / "test_property_metrics.csv", index=False)

    assert _artifact_target_edge_property_mean_r2(tmp_path, "test") == 0.0
    payload = json.loads((tmp_path / "test_metrics.json").read_text(encoding="utf-8"))
    assert payload["target_edge_property_r2"]["Frac_O2"] == 0.0


def test_joint_metric_pools_processes_before_r2_and_exports_all_10(tmp_path: Path):
    for process, temp_mean, temp_sse in ((1, 0.5, 0.0), (2, 10.5, 2.0)):
        process_dir = tmp_path / f"Process{process}"
        process_dir.mkdir()
        pd.DataFrame([
            {
                "property_name": "Temp", "R2_n_valid": 2,
                "R2_SST": 0.5, "R2_SSE": temp_sse, "true_mean": temp_mean,
            },
            {
                "property_name": "Pres", "R2_n_valid": 2,
                "R2_SST": 0.0, "R2_SSE": 0.0, "true_mean": 1.0,
            },
        ]).to_csv(process_dir / "test_property_metrics.csv", index=False)

    value = _refresh_joint_property_artifacts(tmp_path, "test")
    payload = json.loads(
        (tmp_path / "test_metrics.json").read_text(encoding="utf-8")
    )
    expected_temp = 1.0 - 2.0 / 101.0
    assert payload["target_edge_property_r2"]["Temp"] == expected_temp
    assert payload["target_edge_property_r2"]["Pres"] == 0.0
    assert len(payload["target_edge_property_r2"]) == 10
    assert value == expected_temp / 2.0
    assert payload["target_edge_property_pooling"] == (
        "all_processes_all_target_edges_by_property"
    )


def test_multi_aggregate_keeps_10_property_columns_and_joint_pooling(tmp_path: Path):
    run_dir = tmp_path / "multi" / "GAT" / "fold_01"
    for process, temp_mean, temp_sse in ((1, 0.5, 0.0), (2, 10.5, 2.0)):
        process_dir = run_dir / f"Process{process}"
        process_dir.mkdir(parents=True)
        for split in ("val", "test"):
            pd.DataFrame([{
                "property_name": "Temp", "R2_n_valid": 2,
                "R2_SST": 0.5, "R2_SSE": temp_sse, "true_mean": temp_mean,
                "R2": 1.0 - temp_sse / 0.5,
            }]).to_csv(process_dir / f"{split}_property_metrics.csv", index=False)
    (run_dir / "run_summary.json").write_text(json.dumps({
        "model_code": "GAT", "fold": 1, "training_seconds": 1.0,
        "peak_gpu_memory_mb": 2.0,
    }), encoding="utf-8")

    frame = _aggregate_persisted_runs(tmp_path, "multi")
    assert frame.loc[0, "test_target_edge_r2_temp"] == 1.0 - 2.0 / 101.0
    for name in (
        "temp", "pres", "frac_h2o", "frac_h2", "frac_ch4", "frac_co2",
        "frac_co", "frac_o2", "frac_n2", "mass_flow",
    ):
        assert f"test_target_edge_r2_{name}" in frame.columns
