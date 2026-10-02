from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from torch.utils.data import Dataset

from process_graph.data.tabular_dataset import GraphSampleRecord
from process_graph.data.target_edge_augmentation import (
    TargetEdgeAugmentedDataset,
    build_balanced_target_edge_augmented_dataset,
    build_rare_target_edge_augmented_dataset,
)
from process_graph.experiment.loaders import load_experiment_config
from process_graph.schema import GraphSample


def _graph_record(sample_id: int) -> GraphSampleRecord:
    graph = GraphSample(
        process_id="Process1",
        node_names=["A", "B"],
        edge_index=[[0, 0], [1, 1]],
        x_role=[0, 0],
        x_unit=[0, 0],
        x_hx_role=[0, 0],
        x_oper=[[float(sample_id)], [2.0]],
        x_oper_mask=[[1], [1]],
        targets={},
        y_edge_true=[[0.0] * 14, [1.0] * 14],
        y_edge_mask=[1.0, 1.0],
        canonical_edge_ids=["P01_E001", "P01_E002"],
        edge_target_columns=[
            "Temp",
            "Pres",
            "Vol_Flow",
            "Mole_Flow",
            "Mass_Flow",
            "Frac_H2O",
            "Frac_H2",
            "Frac_CH4",
            "Frac_CO2",
            "Frac_CO",
            "Frac_O2",
            "Frac_N2",
            "Enthalpy",
            "Density",
        ],
    )
    return GraphSampleRecord(
        graph=graph,
        slot_targets={},
        slot_masks={},
        slot_node_index={},
        category_node_indices={},
        category_values={},
        sample_meta={"ID": sample_id},
    )


class _RecordDataset(Dataset):
    def __init__(self) -> None:
        self.frame = pd.DataFrame(
            [
                {"process_id": "Process1", "ID": 1},
                {"process_id": "Process1", "ID": 2},
            ]
        )
        self.data_cfg = SimpleNamespace(
            task_mode="edge_all",
            process_id_column="process_id",
            canonical_graph_spec_v3_dir="data/reference/v3",
            stream_data_dir="data/streams",
        )
        self.records = [_graph_record(1), _graph_record(2)]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> GraphSampleRecord:
        return self.records[index]


def _write_fixture(tmp_path: Path) -> None:
    (tmp_path / "data/reference/v4").mkdir(parents=True)
    (tmp_path / "data/reference/v3").mkdir(parents=True)
    (tmp_path / "data/streams").mkdir(parents=True)
    pd.DataFrame(
        [{"process_id": 1, "canonical_answer_edge_id": "P01_E001"}]
    ).to_csv(tmp_path / "data/reference/v4/target_stream_targets.csv", index=False)
    pd.DataFrame(
        [
            {
                "process_id": 1,
                "canonical_edge_id": "P01_E001",
                "main_data_stream_key": "TARGET",
            },
            {
                "process_id": 1,
                "canonical_edge_id": "P01_E002",
                "main_data_stream_key": "OTHER",
            },
        ]
    ).to_csv(tmp_path / "data/reference/v3/canonical_edges.csv", index=False)
    pd.DataFrame(
        [
            {
                "ID": 1,
                "Stream_Name": "TARGET",
                "Frac_CH4": 0.2,
                "Frac_CO2": 0.1,
                "Frac_CO": 0.0,
            },
            {
                "ID": 2,
                "Stream_Name": "TARGET",
                "Frac_CH4": 0.2,
                "Frac_CO2": 0.0,
                "Frac_CO": 0.0,
            },
        ]
    ).to_csv(tmp_path / "data/streams/1.Process_Streams.csv", index=False)


def _config(**overrides):
    values = {
        "components": ["Frac_CH4", "Frac_CO2", "Frac_CO"],
        "positive_threshold": 1.0e-4,
        "high_positive_thresholds": {"Frac_CO2": 0.05},
        "default_factor": 2,
        "high_positive_factor": 3,
        "max_augmented_ratio": 0.5,
        "target_edges_only": True,
        "seed": 42,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_target_edge_augmentation_prioritizes_high_co2_and_masks_one_edge(tmp_path):
    _write_fixture(tmp_path)
    original = _RecordDataset()
    build = build_rare_target_edge_augmented_dataset(
        original,
        augmentation_cfg=_config(),
        project_root=tmp_path,
    )

    assert isinstance(build.dataset, TargetEdgeAugmentedDataset)
    assert len(original) == 2
    assert len(build.dataset) == 3
    assert build.summary["target_mapping_unique_edge_count"] == 1
    assert build.summary["augmented_sample_count"] == 1
    assert build.summary["augmented_ratio"] == 0.5

    original_item = original[0]
    augmented = build.dataset[2]
    assert augmented.graph.y_edge_mask == [1.0, 0.0]
    assert original_item.graph.y_edge_mask == [1.0, 1.0]
    assert augmented.graph.y_edge_true is original_item.graph.y_edge_true
    assert augmented.graph.x_oper is original_item.graph.x_oper
    assert augmented.graph.canonical_edge_ids is original_item.graph.canonical_edge_ids
    assert augmented.sample_meta["is_target_edge_augmented"] is True
    assert augmented.sample_meta["augmented_target_edge_id"] == "P01_E001"
    assert augmented.sample_meta["augmented_positive_components"] == [
        "Frac_CH4",
        "Frac_CO2",
    ]


def test_target_edge_augmentation_zero_ratio_keeps_original_dataset_length(tmp_path):
    _write_fixture(tmp_path)
    original = _RecordDataset()
    build = build_rare_target_edge_augmented_dataset(
        original,
        augmentation_cfg=_config(max_augmented_ratio=0.0),
        project_root=tmp_path,
    )

    assert len(build.dataset) == len(original)
    assert build.summary["augmented_sample_count"] == 0


def test_target_edge_augmentation_config_is_separate_and_opt_in():
    baseline = load_experiment_config(
        Path(
            "configs/experiment/pinn/"
            "process_surrogate_edge_all_v3_pi_all_10pct_softmax_temp05_"
            "fracclr_massdirectlog.yaml"
        )
    )
    enabled = load_experiment_config(
        Path(
            "configs/experiment/pinn/"
            "process_surrogate_edge_all_v3_pi_all_10pct_softmax_temp05_"
            "fracclr_massdirectlog_targetedgeaug.yaml"
        )
    )

    assert baseline.train.rare_positive_sampler.enabled is False
    assert baseline.train.rare_target_edge_augmentation.enabled is False
    assert enabled.train.rare_positive_sampler.enabled is False
    assert enabled.train.rare_target_edge_augmentation.enabled is True
    assert enabled.train.rare_target_edge_augmentation.high_positive_thresholds == {
        "Frac_CO2": 0.05
    }


def test_balanced_target_edge_augmentation_honors_property_quotas_and_edge_cap(tmp_path):
    (tmp_path / "data/reference/v4").mkdir(parents=True)
    (tmp_path / "data/reference/v3").mkdir(parents=True)
    (tmp_path / "data/streams").mkdir(parents=True)
    pd.DataFrame(
        [
            {"process_id": 1, "canonical_answer_edge_id": "P01_E001"},
            {"process_id": 1, "canonical_answer_edge_id": "P01_E002"},
        ]
    ).to_csv(tmp_path / "data/reference/v4/target_stream_targets.csv", index=False)
    pd.DataFrame(
        [
            {
                "process_id": 1,
                "canonical_edge_id": "P01_E001",
                "main_data_stream_key": "TARGET_A",
            },
            {
                "process_id": 1,
                "canonical_edge_id": "P01_E002",
                "main_data_stream_key": "TARGET_B",
            },
        ]
    ).to_csv(tmp_path / "data/reference/v3/canonical_edges.csv", index=False)
    stream_rows = []
    for sample_id in range(1, 7):
        for stream_name in ("TARGET_A", "TARGET_B"):
            stream_rows.append(
                {
                    "ID": sample_id,
                    "Stream_Name": stream_name,
                    "Frac_CO2": 0.2,
                    "Frac_CH4": 0.1,
                    "Frac_CO": 0.05,
                }
            )
    pd.DataFrame(stream_rows).to_csv(
        tmp_path / "data/streams/1.Process_Streams.csv",
        index=False,
    )

    dataset = _RecordDataset()
    dataset.frame = pd.DataFrame(
        [{"process_id": "Process1", "ID": sample_id} for sample_id in range(1, 7)]
    )
    dataset.records = [_graph_record(sample_id) for sample_id in range(1, 7)]
    config = SimpleNamespace(
        target_properties=["Frac_CO2", "Frac_CH4", "Frac_CO"],
        min_positive_threshold={
            "Frac_CO2": 0.05,
            "Frac_CH4": 0.02,
            "Frac_CO": 0.01,
        },
        property_quota={"Frac_CO2": 3, "Frac_CH4": 2, "Frac_CO": 1},
        factor=3,
        max_augmented_items=6,
        max_per_target_edge=3,
        seed=42,
    )

    build = build_balanced_target_edge_augmented_dataset(
        dataset,
        augmentation_cfg=config,
        project_root=tmp_path,
    )

    assert len(build.dataset) == 12
    assert build.summary["property_augmented_item_counts"] == {
        "Frac_CO2": 3,
        "Frac_CH4": 2,
        "Frac_CO": 1,
    }
    assert build.summary["property_quota_shortfall"] == {
        "Frac_CO2": 0,
        "Frac_CH4": 0,
        "Frac_CO": 0,
    }
    assert build.summary["observed_max_per_property_target_edge"] <= 3
    assert max(
        build.summary["property_target_edge_augmented_item_counts"].values()
    ) <= 3
    for offset in range(6):
        augmented = build.dataset[len(dataset) + offset]
        assert sum(augmented.graph.y_edge_mask) == 1.0
        assert augmented.sample_meta["augmented_selected_property"] in {
            "Frac_CO2",
            "Frac_CH4",
            "Frac_CO",
        }
