from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from process_graph.constants import STREAM_EDGE_FEATURE_SLOTS
from process_graph.experiment.target_stream_weighting import build_target_stream_feature_metrics


def _cfg(**kw):
    base = dict(
        use_target_stream_loss_weighting=True,
        target_stream_loss_weight=5.0,
        target_stream_weighting_mode="unique_target_stream_edge_by_process",
        target_stream_weight_conflict_policy="max",
        allow_yaml_only_target_streams=False,
        target_stream_loss_weights=[
            {"target_id": "P06_T001", "canonical_answer_edge_id": "P06_E022", "weight": 8.0},
        ],
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _predictions() -> pd.DataFrame:
    rows = []
    for split in ("val", "test"):
        for prop in STREAM_EDGE_FEATURE_SLOTS:
            mask = 0.0 if prop == "Frac_N2" else 1.0
            for sample in range(2):
                rows.append(
                    {
                        "split": split,
                        "process_id": "Process6",
                        "canonical_edge_id": "P06_E022",
                        "main_data_stream_key": "PROD",
                        "property_name": prop,
                        "y_true_orig": float(sample + 1),
                        "y_pred_orig": float(sample + 1),
                        "y_edge_mask": mask,
                    }
                )
        rows.append(
            {
                "split": split,
                "process_id": "Process6",
                "canonical_edge_id": "P06_E999",
                "main_data_stream_key": "PROD",
                "property_name": "Temp",
                "y_true_orig": 1.0,
                "y_pred_orig": 99.0,
                "y_edge_mask": 1.0,
            }
        )
    return pd.DataFrame(rows)


def test_target_stream_feature_metrics_json_shape_metadata_and_original_scale():
    payload, csv_df, diag_df = build_target_stream_feature_metrics(
        edge_predictions=_predictions(),
        train_cfg=_cfg(),
    )
    assert payload["metadata"]["metric_type"] == "target_stream_feature_prediction"
    assert payload["metadata"]["scale"] == "original"
    assert payload["metadata"]["derived_amount_metrics_are_primary"] is False
    prod = payload["splits"]["test"]["processes"]["6"]["target_streams"]["OUT_PROD"]
    assert prod["canonical_edge_id"] == "P06_E022"
    assert prod["edge_weight"] == 8.0
    assert prod["weight_source"] == "per_process_yaml"
    assert set(prod["features"]) == set(STREAM_EDGE_FEATURE_SLOTS)
    assert prod["features"]["Temp"]["n"] == 2
    assert prod["features"]["Temp"]["mae"] == 0.0
    assert prod["features"]["Frac_N2"]["n"] == 0
    assert prod["features"]["Frac_N2"]["mae"] is None
    assert prod["features"]["Frac_N2"]["rmse"] is None
    assert prod["features"]["Frac_N2"]["r2"] is None
    assert "target_stream_feature_macro_r2" in payload["splits"]["test"]["macro"]
    assert not csv_df.empty
    assert {"split", "process_id", "target_stream", "feature_name", "edge_weight", "weight_source"}.issubset(csv_df.columns)
    assert not diag_df.empty
    used = diag_df[(diag_df["matched"] == True) & (diag_df["target_stream"] == "OUT_PROD")].iloc[0]["used_feature_names"]
    assert "Temp" in used and "Frac_H2" in used and "Frac_CO2" in used
