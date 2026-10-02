"""Tests for process- vs target-balanced primary metrics."""
from __future__ import annotations

import pandas as pd

from process_graph.experiment.target_row_primary_metrics import (
    PRIMARY_METRIC_NAME,
    compute_process_balanced_r2,
    compute_target_balanced_r2,
    extract_target_exclusion_metadata,
)


def test_process_balanced_equals_target_for_single_process():
    metric_df = pd.DataFrame(
        [
            {
                "process_id": "Process1",
                "target_id": "P01_T001",
                "r2": 0.9,
                "include_in_main_verified_macro": True,
            },
            {
                "process_id": "Process1",
                "target_id": "P01_T003",
                "r2": 0.7,
                "include_in_main_verified_macro": True,
            },
        ]
    )
    tb = compute_target_balanced_r2(metric_df)
    pb = compute_process_balanced_r2(metric_df)
    assert abs(tb - 0.8) < 1e-6
    assert abs(pb - 0.8) < 1e-6


def test_process_balanced_differs_when_two_processes():
    metric_df = pd.DataFrame(
        [
            {"process_id": "Process1", "target_id": "P01_T001", "r2": 1.0, "include_in_main_verified_macro": True},
            {"process_id": "Process2", "target_id": "P02_T001", "r2": 0.0, "include_in_main_verified_macro": True},
        ]
    )
    tb = compute_target_balanced_r2(metric_df)
    pb = compute_process_balanced_r2(metric_df)
    assert abs(tb - 0.5) < 1e-6
    assert abs(pb - 0.5) < 1e-6  # (1+0)/2 processes


def test_primary_metric_name():
    assert PRIMARY_METRIC_NAME == "eval_primary_frac_r2_by_process"


def test_extract_metadata_counts_included_in_primary_rows():
    metric_df = pd.DataFrame(
        [
            {"process_id": "Process4", "target_id": "P04_T001", "r2": 0.1, "included_in_primary": True},
            {"process_id": "Process4", "target_id": "P04_T002", "r2": 0.2, "included_in_primary": True},
            {"process_id": "Process4", "target_id": "P04_T003", "r2": 0.3, "included_in_primary": True},
        ]
    )

    meta = extract_target_exclusion_metadata(metric_df, use_included_in_primary=True)

    assert meta["n_targets_total"] == 3
    assert meta["n_included_targets"] == 3
    assert meta["n_main_verified_targets"] == 3
    assert meta["included_target_ids"] == "P04_T001;P04_T002;P04_T003"
    assert meta["n_excluded_targets"] == 0
