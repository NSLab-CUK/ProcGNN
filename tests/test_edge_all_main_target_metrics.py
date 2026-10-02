"""Regression tests for main-target (H2/CO2) R² on answer edges."""

from __future__ import annotations

import pandas as pd

from process_graph.experiment.edge_all_reporting import (
    _DEFAULT_TASK_META,
    _filter_answer_task_property,
    _metric_bundle,
)


def _synthetic_answer_edge_rows() -> pd.DataFrame:
    rows = []
    props = {
        "Mole_Flow": (1000.0, 900.0),
        "Frac_H2": (0.8, 0.75),
        "Frac_CO2": (0.1, 0.12),
        "Temp": (300.0, 298.0),
    }
    for prop, (yt, yp) in props.items():
        rows.append(
            {
                "property_name": prop,
                "y_true_orig": yt,
                "y_pred_orig": yp,
                "answer_task_name": "target_h2",
            }
        )
    return pd.DataFrame(rows)


def test_filter_answer_task_property_keeps_frac_only():
    frame = _synthetic_answer_edge_rows()
    selected = _filter_answer_task_property("target_h2", frame)
    assert set(selected["property_name"].astype(str)) == {"Frac_H2"}
    assert _DEFAULT_TASK_META["target_h2"]["mapped_property_name"] == "Frac_H2"


def test_main_target_h2_r2_differs_from_pooled_all_properties():
    frame = _synthetic_answer_edge_rows()
    g_all = frame[frame["answer_task_name"].astype(str).str.contains("target_h2")]
    g_frac = _filter_answer_task_property("target_h2", g_all)
    r2_all = float(_metric_bundle(g_all["y_true_orig"], g_all["y_pred_orig"])["r2"])
    r2_frac = float(_metric_bundle(g_frac["y_true_orig"], g_frac["y_pred_orig"])["r2"])
    assert r2_frac > 0.9
    assert r2_all < r2_frac
