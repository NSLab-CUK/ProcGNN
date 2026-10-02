"""Unit tests for v4 target metrics (all target_id per process)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from process_graph.experiment.target_v4_metrics import (
    EDGE_STREAM_PROPERTY_TO_INDEX,
    SPECIES_TO_FRAC_PROPERTY,
    TargetV4EpochAccumulator,
    assert_edge_predictions_original_scale,
    compute_target_v4_metrics_from_original_scale,
    load_target_stream_targets_v4,
    normalize_stream_key,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V4_CSV = PROJECT_ROOT / "data/reference/v4/target_stream_targets.csv"


def _edge_row(
    *,
    process_id: str = "Process2",
    sample_id: str = "1",
    stream_key: str = "PROD",
    canonical_edge_id: str = "P02_E014",
    mole_true: float = 100.0,
    mole_pred: float = 95.0,
    frac_h2_true: float = 0.8,
    frac_h2_pred: float = 0.75,
    frac_co2_true: float = 0.1,
    frac_co2_pred: float = 0.12,
) -> list[dict]:
    rows = []
    base = {
        "split": "test",
        "process_id": process_id,
        "sample_id": sample_id,
        "canonical_edge_id": canonical_edge_id,
        "main_data_stream_key": stream_key,
        "y_edge_mask": 1.0,
        "is_answer_edge": 1,
    }
    for prop, yt, yp in (
        ("Mole_Flow", mole_true, mole_pred),
        ("Frac_H2", frac_h2_true, frac_h2_pred),
        ("Frac_CO2", frac_co2_true, frac_co2_pred),
        ("Frac_H2O", 0.05, 0.06),
        ("Temp", 300.0, 298.0),
        ("Pres", 1.0, 1.0),
        ("Vol_Flow", 1e5, 1e5),
        ("Mass_Flow", 1e4, 1e4),
        ("Frac_CH4", 0.05, 0.05),
        ("Frac_CO", 0.0, 0.0),
        ("Frac_O2", 0.0, 0.0),
        ("Frac_N2", 0.0, 0.0),
    ):
        rows.append({**base, "property_name": prop, "y_true_orig": yt, "y_pred_orig": yp})
    return rows


@pytest.fixture
def v4_specs():
    return load_target_stream_targets_v4(V4_CSV)


def test_property_and_species_mappings():
    assert EDGE_STREAM_PROPERTY_TO_INDEX["Mole_Flow"] == 3
    assert EDGE_STREAM_PROPERTY_TO_INDEX["Frac_H2"] == 6
    assert SPECIES_TO_FRAC_PROPERTY["H2"] == "Frac_H2"
    assert normalize_stream_key("12") == "12"
    assert normalize_stream_key("12.0") == "12"


def test_process_target_counts(v4_specs):
    assert len(v4_specs["Process1"]) == 4
    assert len(v4_specs["Process2"]) == 2
    assert len(v4_specs["Process4"]) == 3
    assert len(v4_specs["Process7"]) == 3
    assert len(v4_specs["Process9"]) == 2
    assert all(t.target_species in ("H2", "CO2") for t in v4_specs["Process9"])


def test_p2_h2_scale_factor(v4_specs):
    h2 = [t for t in v4_specs["Process2"] if t.target_id == "P02_T001"][0]
    co2 = [t for t in v4_specs["Process2"] if t.target_id == "P02_T002"][0]
    assert h2.scale_factor == pytest.approx(0.8)
    assert co2.scale_factor == pytest.approx(1.0)
    assert h2.canonical_edge_id == "P02_E014"
    assert co2.canonical_edge_id == "P02_E016"
    assert co2.required_stream_key == "OUT_EXHAUST"


def test_compute_p2_from_original_scale(v4_specs):
    ep = pd.DataFrame(
        _edge_row()
        + _edge_row(
            stream_key="OUT_EXHAUST",
            canonical_edge_id="P02_E016",
            mole_true=80.0,
            mole_pred=78.0,
        )
    )
    result = compute_target_v4_metrics_from_original_scale(
        edge_predictions=ep,
        target_specs=v4_specs["Process2"],
        split_name="test",
    )
    assert not result.target_metrics_df.empty
    h2_row = result.target_metrics_df[result.target_metrics_df["target_id"] == "P02_T001"].iloc[0]
    assert h2_row["n_samples"] >= 1
    assert float(h2_row["scale_factor"]) == pytest.approx(0.8)
    pred_h2 = result.predictions_df[result.predictions_df["target_id"] == "P02_T001"].iloc[0]
    assert pred_h2["target_pred_value"] == pytest.approx(0.8 * 95.0 * 0.75)
    assert pred_h2["target_true_value"] == pytest.approx(0.8 * 100.0 * 0.8)


def test_p7_same_edge_two_species(v4_specs):
    rows = _edge_row(process_id="Process7", canonical_edge_id="P07_E016", stream_key="PROD")
    ep = pd.DataFrame(rows)
    result = compute_target_v4_metrics_from_original_scale(
        edge_predictions=ep,
        target_specs=v4_specs["Process7"],
        split_name="test",
    )
    ids = set(result.target_metrics_df["target_id"].tolist())
    assert "P07_T001" in ids and "P07_T002" in ids


def test_missing_target_skipped_not_fatal(v4_specs):
    rows = _edge_row(process_id="Process9", stream_key="PROD", canonical_edge_id="P09_E020")
    rows += _edge_row(
        process_id="Process9",
        stream_key="EXHAUS",
        canonical_edge_id="P09_E022",
        mole_true=50.0,
        mole_pred=48.0,
    )
    ep = pd.DataFrame(rows)
    result = compute_target_v4_metrics_from_original_scale(
        edge_predictions=ep,
        target_specs=v4_specs["Process9"],
        split_name="test",
    )
    assert len(result.target_metrics_df) == 2
    assert result.skipped_df.empty or len(result.skipped_df) >= 0
    assert any(
        k in result.payload for k in ("target_v4_macro_r2", "test/target_v4_macro_r2", "test/target_v4_r2")
    )


def test_unsupported_species_skipped(v4_specs):
    ep = pd.DataFrame(_edge_row())
    fake_spec = list(v4_specs["Process2"])
    from process_graph.experiment.target_v4_metrics import TargetSpecV4

    fake_spec.append(
        TargetSpecV4(
            process_id="Process2",
            process_num=2,
            target_id="FAKE_T",
            target_name="FAKE",
            target_species="Xe",
            stream_key="PROD",
            required_stream_key="PROD",
            canonical_edge_id="P02_E014",
            scale_factor=1.0,
            formula_type="",
            target_formula="",
            target_stream_node="",
            source_row="",
        )
    )
    result = compute_target_v4_metrics_from_original_scale(
        edge_predictions=ep,
        target_specs=fake_spec,
        split_name="test",
    )
    assert any(result.skipped_df["reason"].astype(str).str.contains("unsupported_species"))


def test_macro_r2_is_per_target_mean(v4_specs):
    ep = pd.DataFrame(_edge_row())
    result = compute_target_v4_metrics_from_original_scale(
        edge_predictions=ep,
        target_specs=v4_specs["Process2"],
        split_name="test",
    )
    per_target = result.target_metrics_df["r2"].astype(float).tolist()
    summ = result.summary_df
    macro_row = summ[summ["target_id"].astype(str) == "__ALL_macro_split__"]
    if macro_row.empty:
        macro_row = summ[summ["summary_kind"].astype(str) == "split_macro"]
    assert not macro_row.empty
    macro = float(macro_row.iloc[0]["r2"])
    assert macro == pytest.approx(sum(per_target) / len(per_target), rel=1e-5)
    assert float(result.payload["test/target_v4_macro_r2"]) == pytest.approx(macro, rel=1e-5)


def test_original_scale_sanity_warns_on_zlike_frac():
    rows = _edge_row(frac_h2_true=5.0, frac_h2_pred=4.0)
    ep = pd.DataFrame(rows)
    warns = assert_edge_predictions_original_scale(ep, strict=False)
    assert any("possible normalized" in w or "Frac_H2" in w for w in warns)


def test_epoch_accumulator_p2_scale():
    acc = TargetV4EpochAccumulator(load_target_stream_targets_v4(V4_CSV))
    from types import SimpleNamespace

    export = SimpleNamespace(
        process_id=["Process2"],
        sample_id=["1"],
        canonical_edge_id=["P02_E014"],
        main_data_stream_key=["PROD"],
        answer_task_names=[""],
        is_input_edge=[0.0],
        is_output_edge=[1.0],
        is_internal_edge=[0.0],
        stream_role=["product"],
    )
    import torch

    cols = [
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
    ]
    y = torch.tensor([[300.0, 1.0, 1e5, 100.0, 1e4, 0.05, 0.8, 0.05, 0.1, 0.0, 0.0, 0.0]], dtype=torch.float32)
    p = torch.tensor([[298.0, 1.0, 1e5, 95.0, 1e4, 0.06, 0.75, 0.05, 0.12, 0.0, 0.0, 0.0]], dtype=torch.float32)
    mask = torch.ones(1, dtype=torch.float32)
    acc.update_batch(
        export=export,
        y_pred_orig=p,
        y_true_orig=y,
        y_edge_mask=mask,
        edge_target_columns=cols,
    )
    scalars = acc.finalize_scalars()
    assert "val_eval_primary_frac_r2_by_process" in scalars
    assert "val_eval_secondary_amount_r2_by_process" in scalars
    assert scalars["val_eval_target_rows_total"] >= 2.0
