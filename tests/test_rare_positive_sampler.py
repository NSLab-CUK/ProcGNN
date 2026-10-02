from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset, Subset

from process_graph.data.rare_positive_sampler import compute_rare_positive_sample_weights
from process_graph.experiment.loaders import load_experiment_config


class _FrameDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, data_cfg: SimpleNamespace) -> None:
        self.frame = frame
        self.data_cfg = data_cfg

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        raise AssertionError("The vectorized sampler must not build graph items.")


def _write_fixture(tmp_path):
    (tmp_path / "data/reference/v4").mkdir(parents=True)
    (tmp_path / "data/reference/v3").mkdir(parents=True)
    (tmp_path / "data/streams").mkdir(parents=True)
    pd.DataFrame(
        [
            {"process_id": 1, "canonical_answer_edge_id": "P01_E001"},
            {"process_id": 2, "canonical_answer_edge_id": "P02_E001"},
        ]
    ).to_csv(tmp_path / "data/reference/v4/target_stream_targets.csv", index=False)
    pd.DataFrame(
        [
            {
                "process_id": 1,
                "canonical_edge_id": "P01_E001",
                "main_data_stream_key": "TARGET",
            },
            {
                "process_id": 2,
                "canonical_edge_id": "P02_E001",
                "main_data_stream_key": "TARGET",
            },
        ]
    ).to_csv(tmp_path / "data/reference/v3/canonical_edges.csv", index=False)
    pd.DataFrame(
        [
            {"ID": 1, "Stream_Name": "TARGET", "Frac_CH4": 0.2, "Frac_CO2": 0.3},
            {"ID": 1, "Stream_Name": "OTHER", "Frac_CH4": 0.9, "Frac_CO2": 0.9},
            {"ID": 2, "Stream_Name": "TARGET", "Frac_CH4": 0.0, "Frac_CO2": 0.0},
        ]
    ).to_csv(tmp_path / "data/streams/1.Process_Streams.csv", index=False)
    pd.DataFrame(
        [
            {"ID": 1, "Stream_Name": "TARGET", "Frac_CH4": 0.0, "Frac_CO2": 0.0},
            {"ID": 2, "Stream_Name": "TARGET", "Frac_CH4": 0.0, "Frac_CO2": 0.0},
            # A positive non-target stream must not affect the sample weight.
            {"ID": 2, "Stream_Name": "OTHER", "Frac_CH4": 0.8, "Frac_CO2": 0.8},
        ]
    ).to_csv(tmp_path / "data/streams/2.Process_Streams.csv", index=False)


def test_rare_positive_weights_use_only_target_streams_and_clamp(tmp_path):
    _write_fixture(tmp_path)
    frame = pd.DataFrame(
        [
            {"process_id": "Process1", "ID": 1},
            {"process_id": "Process1", "ID": 2},
            {"process_id": "Process2", "ID": 1},
            {"process_id": "Process2", "ID": 2},
        ]
    )
    data_cfg = SimpleNamespace(
        task_mode="edge_all",
        process_id_column="process_id",
        canonical_graph_spec_v3_dir="data/reference/v3",
        stream_data_dir="data/streams",
    )
    sampler_cfg = SimpleNamespace(
        components=["Frac_CH4", "Frac_CO2"],
        positive_threshold=1.0e-4,
        max_weight=3.0,
        mode="any_target_edge",
    )
    result = compute_rare_positive_sample_weights(
        _FrameDataset(frame, data_cfg),
        sampler_cfg=sampler_cfg,
        project_root=tmp_path,
    )

    assert torch.allclose(result.weights, torch.tensor([3.0, 1.0, 1.0, 1.0], dtype=torch.double))
    assert result.summary["components"]["Frac_CH4"]["positive_frequency"] == pytest.approx(0.25)
    assert result.summary["components"]["Frac_CO2"]["positive_frequency"] == pytest.approx(0.25)
    assert result.summary["missing_sample_count"] == 0


def test_rare_positive_weights_follow_subset_order(tmp_path):
    _write_fixture(tmp_path)
    frame = pd.DataFrame(
        [
            {"process_id": "Process1", "ID": 1},
            {"process_id": "Process1", "ID": 2},
            {"process_id": "Process2", "ID": 1},
            {"process_id": "Process2", "ID": 2},
        ]
    )
    data_cfg = SimpleNamespace(
        task_mode="edge_all",
        process_id_column="process_id",
        canonical_graph_spec_v3_dir="data/reference/v3",
        stream_data_dir="data/streams",
    )
    sampler_cfg = SimpleNamespace(
        components=["Frac_CH4"],
        positive_threshold=1.0e-4,
        max_weight=10.0,
        mode="any_target_edge",
    )
    dataset = Subset(_FrameDataset(frame, data_cfg), [2, 0])
    result = compute_rare_positive_sample_weights(
        dataset,
        sampler_cfg=sampler_cfg,
        project_root=tmp_path,
    )

    assert torch.allclose(
        result.weights,
        torch.tensor([1.0, 2**0.5], dtype=torch.double),
    )


def test_rare_positive_sampler_is_opt_in_in_experiment_yaml():
    base = load_experiment_config(
        Path(
            "configs/experiment/pinn/"
            "process_surrogate_edge_all_v3_pi_all_10pct_softmax_temp05_fracclr_massdirectlog.yaml"
        )
    )
    enabled = load_experiment_config(
        Path(
            "configs/experiment/pinn/"
            "process_surrogate_edge_all_v3_pi_all_10pct_softmax_temp05_fracclr_massdirectlog_rarepos.yaml"
        )
    )

    assert base.train.rare_positive_sampler.enabled is False
    assert enabled.train.rare_positive_sampler.enabled is True
    assert enabled.train.rare_positive_sampler.components == [
        "Frac_CH4",
        "Frac_CO",
        "Frac_CO2",
    ]
    assert enabled.train.rare_positive_sampler.mode == "any_target_edge"
    assert enabled.train.rare_positive_sampler.replacement is True
