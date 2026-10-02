from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_transfer_data_efficiency_experiments import (  # noqa: E402
    _actual_optimizer_steps,
    _ratio_count,
    _target_mean_r2,
    aggregate_results,
    prepare_nested_subsets,
)


def _manifest(count: int, process_id: int = 1, start: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "process_id": [f"Process{process_id}"] * count,
            "sample_id": list(range(start, start + count)),
            "merged_row_index": list(range(start, start + count)),
        }
    )


def test_ratio_count_uses_half_up_rounding():
    assert _ratio_count(8001, 0.05) == 400
    assert _ratio_count(8001, 0.10) == 800
    assert _ratio_count(8001, 0.25) == 2000
    assert _ratio_count(8001, 0.50) == 4001
    assert _ratio_count(8001, 1.00) == 8001


def test_actual_optimizer_steps_reads_final_cumulative_count(tmp_path: Path):
    pd.DataFrame(
        {"epoch": [1, 2, 3], "train_optimizer_steps_total": [1200, 2500, 4000]}
    ).to_csv(tmp_path / "metrics_per_epoch.csv", index=False)

    assert _actual_optimizer_steps(tmp_path) == 4000


def test_nested_subsets_are_shared_manifests_and_exclude_invalid_graph(tmp_path: Path):
    split_root = tmp_path / "splits"
    fold_dir = split_root / "heldout_P01" / "fold_01"
    fold_dir.mkdir(parents=True)
    _manifest(20).to_csv(fold_dir / "target_train.csv", index=False)
    _manifest(5, start=50).to_csv(fold_dir / "target_val.csv", index=False)
    _manifest(5, start=100).to_csv(fold_dir / "target_test.csv", index=False)
    excluded = tmp_path / "excluded.csv"
    pd.DataFrame(
        {"process_id": [1], "sample_id": [3], "reason": ["test"], "missing_stream_keys": [""]}
    ).to_csv(excluded, index=False)

    specs, pools = prepare_nested_subsets(
        split_root=split_root,
        output_root=tmp_path / "out",
        heldout_processes=[1],
        folds=[1],
        ratios=[0.25, 0.5, 1.0],
        seed=123,
        force=False,
        excluded_graph_samples_path=excluded,
    )

    assert pools.iloc[0]["raw_transfer_pool_size"] == 20
    assert pools.iloc[0]["transfer_pool_size"] == 19
    assert [spec.sample_count for spec in specs] == [5, 10, 19]
    index_sets = []
    for spec in specs:
        payload = json.loads(spec.subset_indices_path.read_text(encoding="utf-8"))
        index_sets.append(set(payload["merged_row_indices"]))
        assert 3 not in index_sets[-1]
    assert index_sets[0] < index_sets[1] < index_sets[2]

    second_specs, _ = prepare_nested_subsets(
        split_root=split_root,
        output_root=tmp_path / "out",
        heldout_processes=[1],
        folds=[1],
        ratios=[0.25, 0.5, 1.0],
        seed=123,
        force=False,
        excluded_graph_samples_path=excluded,
    )
    assert [spec.subset_hash for spec in second_specs] == [spec.subset_hash for spec in specs]
    metadata = json.loads(
        (tmp_path / "out" / "subsets" / "heldout_P01" / "fold_01" / "subset_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["validation_and_test_are_same_manifest"] is False
    assert metadata["train_val_disjoint"] is True
    assert metadata["train_test_disjoint"] is True
    assert metadata["val_test_disjoint"] is True


def test_aggregate_target_mean_uses_membership_and_excludes_vol_flow(tmp_path: Path):
    pd.DataFrame(
        [
            {"split": "test", "feature_name": "Temp", "r2": 0.8, "used_in_mean": True},
            {"split": "test", "feature_name": "Mass_Flow", "r2": 0.6, "used_in_mean": True},
            {"split": "test", "feature_name": "Vol_Flow", "r2": 0.9, "used_in_mean": True},
            {"split": "test", "feature_name": "Frac_CO", "r2": -4.0, "used_in_mean": False},
        ]
    ).to_csv(tmp_path / "target_r2.csv", index=False)

    value = _target_mean_r2(
        tmp_path,
        split="test",
        excluded_properties={"vol_flow"},
        metrics_payload={"test_target_mean_r2": 123.0},
    )
    assert value == 0.7


@pytest.mark.parametrize(
    ("actual_steps", "expected_termination"),
    [(4, "optimizer_step_cap"), (3, "max_epochs")],
)
def test_aggregate_results_accepts_step_cap_or_max_epochs(
    tmp_path: Path,
    actual_steps: int,
    expected_termination: str,
):
    output_root = tmp_path / "out"
    artifact = output_root / "heldout_P01" / "fold_01" / "transfer" / "ratio_010" / "artifact"
    artifact.mkdir(parents=True)
    (artifact / "best.pt").write_bytes(b"checkpoint")
    pd.DataFrame(
        {"epoch": [1, 2], "train_optimizer_steps_total": [2, actual_steps]}
    ).to_csv(artifact / "metrics_per_epoch.csv", index=False)
    (artifact / "metrics.json").write_text('{"best_epoch": 2, "final_epoch": 2}', encoding="utf-8")
    (artifact / "scaler_fit_provenance.json").write_text(
        '{"merged_row_indices_sha256": "subset-hash"}', encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"split": "val", "property_name": "Temp", "R2": 0.7},
            {"split": "test", "property_name": "Temp", "R2": 0.8},
        ]
    ).to_csv(artifact / "target_edge_r2_by_property.csv", index=False)

    run_dir = artifact.parent
    (run_dir / "status.json").write_text(
        '{"status": "completed", "duration_sec": 12.5}', encoding="utf-8"
    )
    (run_dir / "run_metadata.json").write_text(
        '{"subset_hash": "subset-hash", "total_optimizer_steps": 4}', encoding="utf-8"
    )
    subset_dir = output_root / "subsets" / "heldout_P01" / "fold_01"
    subset_dir.mkdir(parents=True)
    (subset_dir / "subset_metadata.json").write_text(
        '{"transfer_pool_size": 10}', encoding="utf-8"
    )

    aggregate_results(
        output_root=output_root,
        existing_unseen_root=tmp_path / "unseen",
        heldout_processes=[1],
        folds=[1],
        ratios=[0.1],
        excluded_properties=set(),
    )

    row = pd.read_csv(output_root / "aggregate" / "run_results.csv").iloc[0]
    assert row["target_edge_property_mean_r2"] == 0.8
    assert row["total_optimizer_steps"] == actual_steps
    assert row["termination_condition"] == expected_termination
