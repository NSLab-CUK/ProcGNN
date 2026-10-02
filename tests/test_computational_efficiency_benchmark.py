from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from run_computational_efficiency_benchmark import (  # noqa: E402
    PRIMARY_MODELS,
    _combine_baseline_fold,
    _prepare_process_split_views,
)


def test_primary_model_set_is_the_required_graph_comparison():
    assert PRIMARY_MODELS == ["gcn", "graphsage", "gin", "proposed"]


def test_process_views_preserve_joint_split_membership(tmp_path: Path):
    split_root = tmp_path / "splits"
    fold = split_root / "fold_01"
    fold.mkdir(parents=True)
    for split in ("train", "val", "test"):
        pd.DataFrame(
            {
                "process_id": ["Process1", "Process2", "Process1"],
                "sample_id": [1, 2, 3],
                "merged_row_index": [10, 20, 30],
            }
        ).to_csv(fold / f"{split}.csv", index=False)

    view_root = _prepare_process_split_views(
        split_root=split_root, output_root=tmp_path / "out", folds=[1]
    )

    p1 = pd.read_csv(view_root / "Process1" / "fold_01" / "train.csv")
    p2 = pd.read_csv(view_root / "Process2" / "fold_01" / "train.csv")
    assert p1["merged_row_index"].tolist() == [10, 30]
    assert p2["merged_row_index"].tolist() == [20]


def test_per_process_gnn_times_are_summed_epoch_by_epoch():
    rows = []
    for pid in range(1, 11):
        rows.append(
            {
                "process_id": pid,
                "train_time_total_sec": 6.0,
                "optimization_time_sec": 6.0,
                "end_to_end_training_time_sec": 8.0,
                "end_to_end_time_until_best_sec": 7.0,
                "train_time_until_best_sec": 5.0,
                "train_time_per_epoch_mean_sec": 2.0,
                "epoch_times_sec": [1.0, 2.0, 3.0],
                "inference_total_sec": 1.0,
                "sample_count": 10,
                "total_parameters": 100,
                "trainable_parameters": 100,
                "num_epochs": 3,
                "peak_gpu_allocated_mb": 20.0,
                "peak_gpu_reserved_mb": 30.0,
                "inference_peak_gpu_memory_mb": 15.0,
                "inference_peak_gpu_reserved_mb": 18.0,
                "inference_time_median_ms": 90.0,
                "test_target_mean_r2": 0.8,
                "device": "cuda:0",
                "inference_batch_size": 4,
                "early_stopping_patience": 3,
            }
        )

    combined = _combine_baseline_fold("gcn", 1, rows)

    assert combined["train_time_total_sec"] == 60.0
    assert combined["optimization_time_sec"] == 60.0
    assert combined["end_to_end_training_time_sec"] == 80.0
    assert combined["train_time_per_epoch_mean_sec"] == 20.0
    assert math.isclose(combined["train_time_per_epoch_std_sec"], 10.0 * math.sqrt(2.0 / 3.0))
    assert combined["total_parameters"] == 1000
    assert combined["inference_time_per_sample_ms"] == 100.0
    assert combined["test_target_mean_r2"] == 0.8
