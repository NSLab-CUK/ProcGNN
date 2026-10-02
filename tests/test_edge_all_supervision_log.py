from __future__ import annotations

from process_graph.experiment.edge_all_supervision_log import (
    format_epoch_detail_edge_all_line,
    format_epoch_summary_edge_all_parts,
    format_val_metric_edge_all_line,
)


def _target_edge_10d_metrics() -> dict[str, float]:
    metrics = {
        "loss_total": 3.742545,
        "loss_edge": 3.742545,
        "target_edge_10d_mae_property_macro": 0.12,
        "target_edge_10d_rmse_property_macro": 0.34,
        "target_edge_10d_r2_property_macro": 0.56,
        "target_edge_10d_r2_edge_macro": 0.67,
        "target_edge_10d_r2_flatten": 0.45,
        "target_r2": 0.56,
        "target_mean_r2": 0.56,
        "target_edge_property_mean_r2": 0.755,
        "target_edge_10d_num_edges": 3.0,
        "target_edge_10d_num_properties": 10.0,
        "target_edge_10d_num_valid_values": 300.0,
        "target_edge_10d_num_r2_unstable": 0.0,
        "eval_primary_frac_r2_by_process": 0.0,
        "eval_target_rows_count": 0.0,
    }
    for index, slug in enumerate(
        (
            "temp",
            "pres",
            "frac_h2o",
            "frac_h2",
            "frac_ch4",
            "frac_co2",
            "frac_co",
            "frac_o2",
            "frac_n2",
            "mass_flow",
        )
    ):
        metrics[f"target_edge_r2_{slug}"] = 0.71 + index * 0.01
    return metrics


def test_val_metric_formatter_prefers_target_edge_10d_over_legacy_primary_frac():
    line = format_val_metric_edge_all_line(_target_edge_10d_metrics())

    assert "[val-target-property-r2][edge_all]" in line
    assert "target_edge_property_mean_r2=0.755000" in line
    assert "Temp=0.71000" in line
    assert "Mass_Flow=0.80000" in line
    assert "target_mean_r2" not in line
    assert "target_edges=3" in line
    assert "legacy_primary_frac=diagnostic_only" in line
    assert "target_rows=0" not in line
    assert "[val-primary][edge_all]" not in line


def test_epoch_detail_formatter_prefers_target_edge_10d():
    epoch_log = {
        "train/edge_step_pi_loss_mean": 2.8,
        "val/loss": 3.7,
        **{f"val/{k}": v for k, v in _target_edge_10d_metrics().items()},
    }

    line = format_epoch_detail_edge_all_line(epoch_log)

    assert "[epoch-target-property-r2]" in line
    assert "target_edge_property_mean_r2=0.75500" in line
    assert "Temp=0.71000" in line
    assert "Mass_Flow=0.80000" in line
    assert "target_mean_r2" not in line
    assert "target_edges=3" in line
    assert "legacy_primary_frac=diagnostic_only" in line
    assert "FracR2" not in line
    assert "rows=0" not in line


def test_epoch_summary_parts_prefers_target_edge_10d():
    epoch_log = {f"val/{k}": v for k, v in _target_edge_10d_metrics().items()}

    parts = format_epoch_summary_edge_all_parts(epoch_log)
    text = " ".join(parts)

    assert "target_edge_10d_edges=3" in text
    assert "target_edge_property_mean_r2=0.755000" in text
    assert "target_mean_r2" not in text
    assert "primary_frac_r2_by_process" not in text
