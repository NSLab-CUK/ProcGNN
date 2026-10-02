"""Tests for run_dir target-row validation."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from process_graph.experiment.target_row_primary_metrics import TARGETROW_METRIC_SCHEMA_VERSION


def test_validate_complete_minimal(tmp_path: Path) -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from ablation_targetrow_common import PRIMARY_METRIC_NAME, validate_run_targetrow_complete

    run = tmp_path / "run"
    test_dir = run / "test"
    test_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "target_id": "P05_T001",
                "process_id": "Process5",
                "r2": 0.9,
                "include_in_main_verified_macro": True,
            }
        ]
    ).to_csv(test_dir / "target_metrics_v4.csv", index=False)
    pd.DataFrame(
        [
            {
                "summary_kind": "process_balanced_macro_main_verified",
                "target_id": "__ALL_process_balanced_main_verified__",
                "r2": 0.9,
            }
        ]
    ).to_csv(test_dir / "target_metrics_v4_summary.csv", index=False)
    metrics = {
        "targetrow_metric_schema_version": TARGETROW_METRIC_SCHEMA_VERSION,
        "primary_metric_name": PRIMARY_METRIC_NAME,
        "test/process_balanced_main_target_r2_main_verified": 0.9,
        "included_target_ids": "P05_T001",
        "excluded_target_ids": "",
    }
    (run / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    val = validate_run_targetrow_complete(run)
    assert val["ok"] is True
