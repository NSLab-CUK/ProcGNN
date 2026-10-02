from types import SimpleNamespace

import pandas as pd

from process_graph.data.epoch_balanced_sampler import EpochBalancedSampler
from process_graph.experiment.schema import (
    EpochBalancedSamplerConfig,
    EpochSamplerHardTargetEdgesConfig,
    EpochSamplerMassFlowTailConfig,
    EpochSamplerMixtureConfig,
    EpochSamplerQuantileBalanceConfig,
    EpochSamplerSparsePositiveConfig,
)


class _TinyDataset:
    def __init__(self, frame, data_cfg):
        self.frame = frame
        self.data_cfg = data_cfg

    def __len__(self):
        return len(self.frame)


def test_epoch_sampler_random_mode_is_global_unique_and_reproducible(tmp_path):
    rows = [
        {"process_id": "Process1" if index < 90 else "Process2", "ID": index}
        for index in range(100)
    ]
    dataset = _TinyDataset(
        pd.DataFrame(rows),
        SimpleNamespace(
            task_mode="edge_all",
            process_id_column="process_id",
            stream_data_dir="unused",
            canonical_graph_spec_v3_dir="unused",
        ),
    )
    cfg = EpochBalancedSamplerConfig(
        enabled=True,
        mode="random",
        base_epoch_size=20,
        sampler_seed=123,
    )
    sampler = EpochBalancedSampler(
        dataset,
        sampler_cfg=cfg,
        project_root=tmp_path,
        output_dir=tmp_path / "stats",
    )

    sampler.set_epoch(0)
    first = list(iter(sampler))
    sampler.set_epoch(0)
    second = list(iter(sampler))
    sampler.set_epoch(1)
    third = list(iter(sampler))

    assert first == second
    assert first != third
    assert len(first) == 20
    assert len(set(first)) == 20
    stats = pd.read_json(
        tmp_path / "stats/sampling_stats_epoch_0001.json",
        typ="series",
    )
    assert stats["mode"] == "random"
    assert stats["bucket_counts"] == {"uniform_random": 20}


def test_main_row_ratio_rule_builds_p03_transition_bucket_without_targets(tmp_path):
    rows = []
    for sample_id in range(60):
        ratio = 2.0 + 0.05 * sample_id
        rows.append(
            {
                "process_id": "Process3",
                "ID": sample_id,
                "CH4_Flow": 10.0,
                "AIR_Flow": 10.0 * ratio,
            }
        )
    rows.extend(
        {
            "process_id": "Process4",
            "ID": sample_id,
            "CH4_Flow": 10.0,
            "AIR_Flow": 35.0,
        }
        for sample_id in range(40)
    )
    dataset = _TinyDataset(
        pd.DataFrame(rows),
        SimpleNamespace(
            task_mode="edge_all",
            process_id_column="process_id",
            stream_data_dir="unused",
            canonical_graph_spec_v3_dir="unused",
        ),
    )
    cfg = EpochBalancedSamplerConfig(
        enabled=True,
        mode="scheduled_predefined_hard",
        base_epoch_size=20,
        sampler_seed=123,
        hard_target_edges=EpochSamplerHardTargetEdgesConfig(
            enabled=True,
            rules=[
                {
                    "name": "p03_air_ch4_transition",
                    "source": "main_row_ratio",
                    "property": "Frac_CO",
                    "process_ids": [3],
                    "numerator_column": "AIR_Flow",
                    "denominator_column": "CH4_Flow",
                    "transform": "raw_ratio",
                    "epsilon": 1.0e-6,
                    "min_value": 3.0,
                    "max_value": 4.2,
                }
            ],
        ),
    )
    sampler = EpochBalancedSampler(
        dataset,
        sampler_cfg=cfg,
        project_root=tmp_path,
    )

    label = "p03_air_ch4_transition:input_regime:Frac_CO"
    selected = sampler.buckets["hard_target_edges"][3][label]
    selected_ids = [int(dataset.frame.iloc[index]["ID"]) for index in selected]
    assert selected_ids == list(range(20, 45))
    assert sampler.buckets["hard_target_edges"].get(4, {}).get(label, []) == []


def test_epoch_balanced_sampler_is_process_balanced_unique_and_reproducible(tmp_path):
    project_root = tmp_path
    (project_root / "data/reference/v3").mkdir(parents=True)
    (project_root / "data/reference/v4").mkdir(parents=True)
    (project_root / "streams").mkdir(parents=True)
    pd.DataFrame(
        [
            {"process_id": 1, "canonical_edge_id": "P01_E001", "main_data_stream_key": "S1"},
            {"process_id": 1, "canonical_edge_id": "P01_E002", "main_data_stream_key": "S2"},
            {"process_id": 2, "canonical_edge_id": "P02_E001", "main_data_stream_key": "S1"},
            {"process_id": 2, "canonical_edge_id": "P02_E002", "main_data_stream_key": "S2"},
        ]
    ).to_csv(project_root / "data/reference/v3/canonical_edges.csv", index=False)
    pd.DataFrame(
        [
            {"process_id": 1, "canonical_answer_edge_id": "P01_E001"},
            {"process_id": 2, "canonical_answer_edge_id": "P02_E001"},
        ]
    ).to_csv(project_root / "data/reference/v4/target_stream_targets.csv", index=False)
    rows = []
    for pid in (1, 2):
        stream_rows = []
        for sample_id in range(10):
            rows.append({"process_id": f"Process{pid}", "ID": sample_id})
            for stream_name in ("S1", "S2"):
                stream_rows.append(
                    {
                        "ID": sample_id,
                        "Stream_Name": stream_name,
                        "Temp": 300.0 + sample_id,
                        "Pres": 10.0 + sample_id,
                        "Mass_Flow": 1.0 + pid * sample_id,
                        "Frac_H2O": 0.1,
                        "Frac_H2": 0.2 + 0.01 * sample_id,
                        "Frac_CH4": 0.001 * sample_id,
                        "Frac_CO2": 0.002 * sample_id,
                        "Frac_CO": 0.003 * sample_id,
                        "Frac_O2": 0.0,
                        "Frac_N2": 0.7,
                    }
                )
        pd.DataFrame(stream_rows).to_csv(project_root / f"streams/{pid}.Process_Streams.csv", index=False)
    data_cfg = SimpleNamespace(
        task_mode="edge_all",
        process_id_column="process_id",
        stream_data_dir="streams",
        canonical_graph_spec_v3_dir="data/reference/v3",
    )
    dataset = _TinyDataset(pd.DataFrame(rows), data_cfg)
    cfg = EpochBalancedSamplerConfig(
        enabled=True,
        epoch_fraction=0.5,
        sampler_seed=123,
        mixture=EpochSamplerMixtureConfig(uniform=0.4, sparse_positive=0.3, quantile_balance=0.2, mass_flow_tail=0.1),
        sparse_positive=EpochSamplerSparsePositiveConfig(
            properties=["Frac_CH4", "Frac_CO", "Frac_CO2", "Frac_H2"],
            positive_quantiles=[0.0, 0.5, 0.9, 1.0],
        ),
        quantile_balance=EpochSamplerQuantileBalanceConfig(
            properties=["Temp", "Pres", "Mass_Flow", "Frac_H2"],
            quantiles=[0.0, 0.5, 1.0],
        ),
        mass_flow_tail=EpochSamplerMassFlowTailConfig(
            property="Mass_Flow",
            quantiles=[0.8, 0.95, 1.0],
        ),
    )
    sampler = EpochBalancedSampler(dataset, sampler_cfg=cfg, project_root=project_root, output_dir=tmp_path / "stats")
    sampler.set_epoch(0)
    first = list(iter(sampler))
    sampler.set_epoch(0)
    second = list(iter(sampler))

    assert first == second
    assert len(first) == 10
    assert len(set(first)) == 10
    counts = pd.Series([dataset.frame.iloc[i]["process_id"] for i in first]).value_counts().to_dict()
    assert counts == {"Process1": 5, "Process2": 5}
    assert (tmp_path / "stats/sampling_stats_epoch_0001.json").is_file()


def test_epoch_balanced_sampler_can_focus_hard_target_edges(tmp_path):
    project_root = tmp_path
    (project_root / "data/reference/v3").mkdir(parents=True)
    (project_root / "data/reference/v4").mkdir(parents=True)
    (project_root / "streams").mkdir(parents=True)
    pd.DataFrame(
        [
            {"process_id": 1, "canonical_edge_id": "P01_E001", "main_data_stream_key": "TARGET"},
            {"process_id": 1, "canonical_edge_id": "P01_E002", "main_data_stream_key": "OTHER"},
            {"process_id": 2, "canonical_edge_id": "P02_E001", "main_data_stream_key": "TARGET"},
            {"process_id": 2, "canonical_edge_id": "P02_E002", "main_data_stream_key": "OTHER"},
        ]
    ).to_csv(project_root / "data/reference/v3/canonical_edges.csv", index=False)
    pd.DataFrame(
        [
            {"process_id": 1, "canonical_answer_edge_id": "P01_E001"},
            {"process_id": 2, "canonical_answer_edge_id": "P02_E001"},
        ]
    ).to_csv(project_root / "data/reference/v4/target_stream_targets.csv", index=False)

    rows = []
    for pid in (1, 2):
        stream_rows = []
        for sample_id in range(10):
            rows.append({"process_id": f"Process{pid}", "ID": sample_id})
            stream_rows.append(
                {
                    "ID": sample_id,
                    "Stream_Name": "TARGET",
                    "Frac_CO2": 0.0 if sample_id < 5 else 0.25,
                }
            )
            stream_rows.append(
                {
                    "ID": sample_id,
                    "Stream_Name": "OTHER",
                    "Frac_CO2": 0.5,
                }
            )
        pd.DataFrame(stream_rows).to_csv(project_root / f"streams/{pid}.Process_Streams.csv", index=False)
    data_cfg = SimpleNamespace(
        task_mode="edge_all",
        process_id_column="process_id",
        stream_data_dir="streams",
        canonical_graph_spec_v3_dir="data/reference/v3",
    )
    dataset = _TinyDataset(pd.DataFrame(rows), data_cfg)
    cfg = EpochBalancedSamplerConfig(
        enabled=True,
        epoch_fraction=0.5,
        sampler_seed=123,
        mixture=EpochSamplerMixtureConfig(
            uniform=0.0,
            sparse_positive=0.0,
            quantile_balance=0.0,
            mass_flow_tail=0.0,
            hard_target_edges=1.0,
        ),
        hard_target_edges=EpochSamplerHardTargetEdgesConfig(
            enabled=True,
            rules=[
                {
                    "name": "weak_co2",
                    "property": "Frac_CO2",
                    "min_value": 0.001,
                    "max_value": 1.0,
                    "edge_ids": ["P01_E001", "P02_E001"],
                }
            ],
        ),
    )
    sampler = EpochBalancedSampler(dataset, sampler_cfg=cfg, project_root=project_root, output_dir=tmp_path / "stats")
    selected = list(iter(sampler))

    assert len(selected) == 10
    assert all(int(dataset.frame.iloc[index]["ID"]) >= 5 for index in selected)
    stats = pd.read_json(tmp_path / "stats/sampling_stats_epoch_0001.json", typ="series")
    assert stats["bucket_counts"]["hard_target_edges"] == 10


def test_base_plus_hard_fill_guarantees_unique_mass_flow_tail_quota(tmp_path):
    project_root = tmp_path
    (project_root / "data/reference/v3").mkdir(parents=True)
    (project_root / "data/reference/v4").mkdir(parents=True)
    (project_root / "streams").mkdir(parents=True)
    pd.DataFrame(
        [
            {"process_id": pid, "canonical_edge_id": f"P{pid:02d}_E001", "main_data_stream_key": "S1"}
            for pid in (1, 2)
        ]
    ).to_csv(project_root / "data/reference/v3/canonical_edges.csv", index=False)
    pd.DataFrame(
        [
            {"process_id": pid, "canonical_answer_edge_id": f"P{pid:02d}_E001"}
            for pid in (1, 2)
        ]
    ).to_csv(project_root / "data/reference/v4/target_stream_targets.csv", index=False)
    rows = []
    for pid in (1, 2):
        stream_rows = []
        for sample_id in range(100):
            rows.append({"process_id": f"Process{pid}", "ID": sample_id})
            stream_rows.append(
                {
                    "ID": sample_id,
                    "Stream_Name": "S1",
                    "Temp": 300.0,
                    "Pres": 10.0,
                    "Mass_Flow": float(sample_id + pid * 100),
                    "Frac_CH4": 0.1,
                    "Frac_CO": 0.1,
                    "Frac_CO2": 0.1,
                    "Frac_H2": 0.1,
                }
            )
        pd.DataFrame(stream_rows).to_csv(
            project_root / f"streams/{pid}.Process_Streams.csv",
            index=False,
        )
    dataset = _TinyDataset(
        pd.DataFrame(rows),
        SimpleNamespace(
            task_mode="edge_all",
            process_id_column="process_id",
            stream_data_dir="streams",
            canonical_graph_spec_v3_dir="data/reference/v3",
        ),
    )
    cfg = EpochBalancedSamplerConfig(
        enabled=True,
        mode="base_plus_hard_fill",
        epoch_fraction=0.1,
        hard_fill_total_size=20,
        hard_fill_ratio=0.0,
        sampler_seed=321,
        mixture=EpochSamplerMixtureConfig(
            uniform=0.5,
            sparse_positive=0.0,
            quantile_balance=0.5,
            mass_flow_tail=0.0,
            hard_target_edges=0.0,
        ),
        quantile_balance=EpochSamplerQuantileBalanceConfig(
            properties=["Mass_Flow"],
            quantiles=[0.0, 0.5, 1.0],
        ),
        mass_flow_tail=EpochSamplerMassFlowTailConfig(
            enabled=True,
            target_items_per_epoch=6,
            property="Mass_Flow",
            quantiles=[0.90, 0.95, 0.99, 1.0],
            bin_quotas={"q0.90_0.95": 2, "q0.95_0.99": 2, "q0.99_1.00": 2},
            quantile_scope="global",
            canonical_edge_min_count=1,
        ),
    )
    sampler = EpochBalancedSampler(
        dataset,
        sampler_cfg=cfg,
        project_root=project_root,
        output_dir=tmp_path / "stats",
    )
    sampler.set_epoch(0)
    first = list(iter(sampler))
    sampler.set_epoch(0)
    second = list(iter(sampler))
    stats = pd.read_json(tmp_path / "stats/sampling_stats_epoch_0001.json", typ="series")

    assert first == second
    assert len(first) == 20
    assert len(set(first)) == 20
    assert stats["mass_flow_tail_requested_count"] == 6
    assert stats["mass_flow_tail_selected_unique_count"] == 6
    assert stats["mass_flow_tail_shortfall_count"] == 0


def test_base_plus_hard_fill_honors_property_quotas(tmp_path):
    project_root = tmp_path
    (project_root / "data/reference/v3").mkdir(parents=True)
    (project_root / "data/reference/v4").mkdir(parents=True)
    (project_root / "streams").mkdir(parents=True)
    edge_rows = [
        {"process_id": 1, "canonical_edge_id": "P01_E001", "main_data_stream_key": "CO"},
        {"process_id": 1, "canonical_edge_id": "P01_E002", "main_data_stream_key": "MASS"},
        {"process_id": 1, "canonical_edge_id": "P01_E003", "main_data_stream_key": "CH4"},
    ]
    pd.DataFrame(edge_rows).to_csv(
        project_root / "data/reference/v3/canonical_edges.csv", index=False
    )
    pd.DataFrame(
        [
            {"process_id": 1, "canonical_answer_edge_id": row["canonical_edge_id"]}
            for row in edge_rows
        ]
    ).to_csv(
        project_root / "data/reference/v4/target_stream_targets.csv", index=False
    )

    frame_rows = []
    stream_rows = []
    for sample_id in range(60):
        frame_rows.append({"process_id": "Process1", "ID": sample_id})
        stream_rows.extend(
            [
                {
                    "ID": sample_id,
                    "Stream_Name": "CO",
                    "Frac_CO": 0.2 if sample_id < 20 else 0.0,
                    "Mass_Flow": 0.0,
                    "Frac_CH4": 0.0,
                },
                {
                    "ID": sample_id,
                    "Stream_Name": "MASS",
                    "Frac_CO": 0.0,
                    "Mass_Flow": (
                        300000.0
                        if 20 <= sample_id < 30
                        else 700000.0
                        if 30 <= sample_id < 40
                        else 0.0
                    ),
                    "Frac_CH4": 0.0,
                },
                {
                    "ID": sample_id,
                    "Stream_Name": "CH4",
                    "Frac_CO": 0.0,
                    "Mass_Flow": 0.0,
                    "Frac_CH4": 0.2 if 40 <= sample_id else 0.0,
                },
            ]
        )
    pd.DataFrame(stream_rows).to_csv(
        project_root / "streams/1.Process_Streams.csv", index=False
    )
    dataset = _TinyDataset(
        pd.DataFrame(frame_rows),
        SimpleNamespace(
            task_mode="edge_all",
            process_id_column="process_id",
            stream_data_dir="streams",
            canonical_graph_spec_v3_dir="data/reference/v3",
        ),
    )
    cfg = EpochBalancedSamplerConfig(
        enabled=True,
        mode="base_plus_hard_fill",
        epoch_fraction=0.5,
        base_epoch_size=4,
        hard_fill_total_size=20,
        hard_fill_ratio=0.8,
        sampler_seed=321,
        mixture=EpochSamplerMixtureConfig(
            uniform=1.0,
            sparse_positive=0.0,
            quantile_balance=0.0,
            mass_flow_tail=0.0,
            hard_target_edges=0.0,
        ),
        hard_target_edges=EpochSamplerHardTargetEdgesConfig(
            enabled=True,
            property_quotas={"Frac_CO": 6, "Mass_Flow": 7, "Frac_CH4": 3},
            rule_quotas={"mass_high": 4, "mass_extreme": 3},
            rules=[
                {
                    "name": "co",
                    "property": "Frac_CO",
                    "min_value": 0.1,
                    "max_value": 1.0,
                    "edge_ids": ["P01_E001"],
                },
                {
                    "name": "mass_high",
                    "property": "Mass_Flow",
                    "min_value": 100000.0,
                    "max_value": 499999.999999,
                    "edge_ids": ["P01_E002"],
                },
                {
                    "name": "mass_extreme",
                    "property": "Mass_Flow",
                    "min_value": 500000.0,
                    "max_value": 1.0e20,
                    "edge_ids": ["P01_E002"],
                },
                {
                    "name": "ch4",
                    "property": "Frac_CH4",
                    "min_value": 0.1,
                    "max_value": 1.0,
                    "edge_ids": ["P01_E003"],
                },
            ],
        ),
    )
    sampler = EpochBalancedSampler(
        dataset,
        sampler_cfg=cfg,
        project_root=project_root,
        output_dir=tmp_path / "stats",
    )
    selected = list(iter(sampler))
    stats = pd.read_json(tmp_path / "stats/sampling_stats_epoch_0001.json", typ="series")

    assert len(selected) == 20
    assert len(set(selected)) == 20
    assert stats["hard_property_counts"] == {
        "Frac_CH4": 3,
        "Frac_CO": 6,
        "Mass_Flow": 7,
    }
    assert stats["hard_property_shortfalls"] == {
        "Frac_CH4": 0,
        "Frac_CO": 0,
        "Mass_Flow": 0,
    }
    assert stats["hard_rule_counts"] == {
        "mass_extreme": 3,
        "mass_high": 4,
    }
    assert stats["hard_rule_shortfalls"] == {
        "mass_extreme": 0,
        "mass_high": 0,
    }
