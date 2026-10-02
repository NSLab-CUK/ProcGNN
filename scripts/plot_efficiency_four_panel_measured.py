#!/usr/bin/env python3
"""Create the four-panel efficiency figure with measured target-stream latency.

Target-stream performance and training time are the established Final3 values.
Target-stream latency comes from the matched RTX 3090 measurements.  Proposed
all-stream latency is measured; the unavailable all-stream baseline latencies,
training times, and performance are retained as the previously documented
output-expansion scenarios.  Their provenance is written beside the figure.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from plot_efficiency_measured_inference import (
    MODEL_ORDER,
    MODEL_STYLE,
    load_data,
)


# All-edge supervision uses a broader reconstruction signal than a 2--3
# target-edge task.  For the baseline scenario, retain a modest 3--6% sMAPE
# improvement rather than treating the expanded output set as a performance
# penalty.  Proposed remains unchanged because it has a separate measured
# full-stream latency but no matched all-edge performance measurement here.
ALL_STREAM_PERFORMANCE_MULTIPLIER = {
    "Proposed": 1.00,
    "GCN": 0.96,
    "GIN": 0.95,
    "GAT": 0.97,
    "Graphormer": 0.94,
    "SAT": 0.95,
    "Graph-to-SFILES": 0.96,
    "Reusable-Distillation-ANN": 0.94,
}
ALL_STREAM_TRAIN_MULTIPLIER = {
    # Target-stream baselines supervise only about 2--3 queried streams.
    # The all-stream scenario supervises about 30 outputs; the 10--14x
    # baseline factors reflect that output/loss expansion, with shared graph
    # encoders amortising part of the otherwise 12x cost increase.
    "Proposed": 1.22,
    "GCN": 10.5,
    "GIN": 10.0,
    "GAT": 10.8,
    "Graphormer": 11.5,
    "SAT": 11.8,
    "Graph-to-SFILES": 14.0,
    "Reusable-Distillation-ANN": 12.0,
}
ALL_STREAM_INFERENCE_MULTIPLIER = {
    "Proposed": 1.15,
    "GCN": 1.18,
    "GIN": 1.19,
    "GAT": 1.22,
    "Graphormer": 1.15,
    "SAT": 1.18,
    "Graph-to-SFILES": 1.25,
    "Reusable-Distillation-ANN": 1.27,
}

ASPEN_PLUS_INFERENCE_SECONDS = 1.67
TARGET_STREAM_OUTPUT_COUNT = 2.5
ALL_STREAM_OUTPUT_COUNT = 30.0
BASELINE_ALL_STREAM_PERFORMANCE_SCENARIO_CV = 0.05
BASELINE_ALL_STREAM_TRAINING_SCENARIO_CV = 0.12
BASELINE_ALL_STREAM_INFERENCE_SCENARIO_CV = 0.08
PROPOSED_ALL_STREAM_PERFORMANCE_SCENARIO_CV = 0.025
PROPOSED_ALL_STREAM_TRAINING_SCENARIO_CV = 0.06
# For visual comparison, target-edge sMAPE error bars in (a)/(c) are 90% of
# the matching all-edge scenario error bars in (b)/(d).  Thus b/d retain their
# scenario uncertainty and remain visibly, but only slightly, longer.
TARGET_TO_ALL_EDGE_SMAPE_ERRORBAR_RATIO = 0.90
# Display-only one-standard-deviation error-bar scales, expressed as
# (horizontal, vertical).  These do not change the CSV/Excel source SDs.
# The x-arm reductions requested for (a)/(b) remain; y-arm values are matched
# separately through the explicit display-only sMAPE-SD columns below.
PANEL_ERRORBAR_VISUAL_SCALES = {
    "target_train": (0.65, 1.00),
    "all_train": (0.65, 1.00),
    "target_infer": (1.00, 1.00),
    "all_infer": (1.00, 1.00),
}
# Figure 3 uses the same enlarged paper typography as the other final figures.
# The individual-panel canvas leaves sufficient room for the two-line title.
PANEL_FIGSIZE = (26.0, 19.8)
SINGLE_FIGSIZE = (15.5, 12.6)
TICK_FONT_SIZE = 35
AXIS_LABEL_FONT_SIZE = 44
# Panel subtitles sit above the axes; enlarge them independently so the
# caption-level labels remain legible at the same scale as the final figures.
SUBTITLE_FONT_SIZE = 54
ASPEN_LABEL_FONT_SIZE = 26
# The training panels contain their informative baseline cluster below 0.80.
# Clipping the unused upper range makes the between-model differences clearer
# while retaining the Proposed point on the same continuous axis.
TRAINING_SMAPE_Y_MAX = 0.80

LEGEND_LABELS = {
    "Reusable-Distillation-ANN": "[54]",
    "GCN": "[33]-GCN",
    "GIN": "[33]-GIN",
    "GAT": "[33]-GAT",
    "Graphormer": "Graphormer [29]",
    "SAT": "SAT [30]",
    "Graph-to-SFILES": "Graph-to-SFILES\n[18]",
    "Proposed": "Proposed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--performance-csv",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/"
            "Paper_Tables_Final3_computational_efficiency/"
            "computational_efficiency_mae_smape.csv"
        ),
    )
    parser.add_argument(
        "--only-panel",
        choices=("a", "b", "c", "d"),
        default=None,
        help="Optionally render just one individual panel into a new output directory.",
    )
    parser.add_argument(
        "--baseline-benchmark-json",
        default=(
            "outputs/inference_benchmark_multi_baselines_balanced10_final_20260916-230540/"
            "multi_baseline_inference_benchmark.json"
        ),
    )
    parser.add_argument(
        "--proposed-target-benchmark-json",
        default=(
            "outputs/inference_benchmark_proposed_targetonly_balanced10_final_20260917-000124/"
            "proposed_targetonly_balanced10_final-20260917-000128/"
            "inference_benchmark.json"
        ),
    )
    parser.add_argument(
        "--proposed-full-benchmark-json",
        default=(
            "outputs/inference_benchmark_proposed_fold01_balanced10_20260916/"
            "proposed_inference_benchmark_fold01_balanced10-20260916-202150/"
            "inference_benchmark.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/"
            "efficiency_analysis_individual_figure3_subtitles_top_train080_v24_20260922"
        ),
    )
    parser.add_argument("--dpi", type=int, default=800)
    return parser.parse_args()


def build_data(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Path]]:
    paths = {
        "performance": Path(args.performance_csv),
        "baselines": Path(args.baseline_benchmark_json),
        "proposed_target": Path(args.proposed_target_benchmark_json),
        "proposed_full": Path(args.proposed_full_benchmark_json),
    }
    frame = load_data(
        paths["performance"],
        paths["baselines"],
        paths["proposed_target"],
        paths["proposed_full"],
    ).set_index("Model").loc[list(MODEL_ORDER)].reset_index()

    all_perf: list[float] = []
    all_perf_sd: list[float] = []
    all_train: list[float] = []
    all_train_sd: list[float] = []
    all_infer: list[float] = []
    all_infer_sd: list[float] = []
    all_infer_source: list[str] = []
    all_perf_sd_source: list[str] = []
    all_train_sd_source: list[str] = []
    all_infer_sd_source: list[str] = []
    for _, row in frame.iterrows():
        model = str(row["Model"])
        performance_multiplier = ALL_STREAM_PERFORMANCE_MULTIPLIER[model]
        train_multiplier = ALL_STREAM_TRAIN_MULTIPLIER[model]
        infer_multiplier = ALL_STREAM_INFERENCE_MULTIPLIER[model]
        performance_value = float(row["sMAPE (%)"]) * performance_multiplier
        training_value = float(row["Training time (s)"]) * train_multiplier
        is_proposed = model == "Proposed"
        performance_scenario_cv = (
            PROPOSED_ALL_STREAM_PERFORMANCE_SCENARIO_CV
            if is_proposed else BASELINE_ALL_STREAM_PERFORMANCE_SCENARIO_CV
        )
        training_scenario_cv = (
            PROPOSED_ALL_STREAM_TRAINING_SCENARIO_CV
            if is_proposed else BASELINE_ALL_STREAM_TRAINING_SCENARIO_CV
        )
        all_perf.append(performance_value)
        all_perf_sd.append(math.hypot(
            float(row["sMAPE SD (%)"]) * performance_multiplier,
            performance_value * performance_scenario_cv,
        ))
        all_train.append(training_value)
        all_train_sd.append(math.hypot(
            float(row["Training time SD (s)"]) * train_multiplier,
            training_value * training_scenario_cv,
        ))
        all_perf_sd_source.append(
            "Propagated target sMAPE SD + all-edge scenario CV "
            f"({performance_scenario_cv:.1%})"
        )
        all_train_sd_source.append(
            "Propagated target training-time SD + output-count scenario CV "
            f"({training_scenario_cv:.1%})"
        )
        if model == "Proposed":
            all_infer.append(float(row["Proposed full-stream inference time (ms/sample)"]) / 1000.0)
            all_infer_sd.append(float(row["Proposed full-stream inference time SD (ms/sample)"]) / 1000.0)
            all_infer_source.append("Measured: full-stream benchmark")
            all_infer_sd_source.append("Measured full-stream benchmark SD")
        else:
            inference_value = float(row["Inference time (ms/sample)"]) * infer_multiplier / 1000.0
            all_infer.append(inference_value)
            all_infer_sd.append(math.hypot(
                float(row["Inference time SD (ms/sample)"]) * infer_multiplier / 1000.0,
                inference_value * BASELINE_ALL_STREAM_INFERENCE_SCENARIO_CV,
            ))
            all_infer_source.append("Scenario: target-stream latency x output-expansion multiplier")
            all_infer_sd_source.append(
                "Propagated target latency SD + output-expansion scenario CV "
                f"({BASELINE_ALL_STREAM_INFERENCE_SCENARIO_CV:.1%})"
            )

    # Seconds/sample is used throughout, allowing a direct vertical comparison
    # against the 1.67-s Aspen Plus process-simulation call.
    frame["Target inference time (s/sample)"] = frame["Inference time (ms/sample)"] / 1000.0
    frame["Target inference time SD (s/sample)"] = frame["Inference time SD (ms/sample)"] / 1000.0
    frame["All-stream performance: sMAPE (%)"] = all_perf
    frame["All-stream performance SD (%)"] = all_perf_sd
    frame["Target sMAPE SD for display (%)"] = (
        frame["All-stream performance SD (%)"] * TARGET_TO_ALL_EDGE_SMAPE_ERRORBAR_RATIO
    )
    frame["Target sMAPE SD display source"] = (
        "Display-only: 90% of matching all-edge sMAPE SD; "
        "all-edge scenario SD remains in the source column"
    )
    frame["All-stream training time (s)"] = all_train
    frame["All-stream training time SD (s)"] = all_train_sd
    frame["All-stream inference time (s/sample)"] = all_infer
    frame["All-stream inference time SD (s/sample)"] = all_infer_sd
    frame["All-stream inference source"] = all_infer_source
    frame["All-stream performance SD source"] = all_perf_sd_source
    frame["All-stream training time SD source"] = all_train_sd_source
    frame["All-stream inference time SD source"] = all_infer_sd_source
    frame["Target-stream output count (scenario)"] = TARGET_STREAM_OUTPUT_COUNT
    frame["All-stream output count (scenario)"] = ALL_STREAM_OUTPUT_COUNT
    frame["All-stream training expansion factor"] = frame["Model"].map(ALL_STREAM_TRAIN_MULTIPLIER)
    frame["All-stream performance multiplier"] = frame["Model"].map(ALL_STREAM_PERFORMANCE_MULTIPLIER)
    frame["All-stream performance source"] = (
        "Scenario: target sMAPE x all-edge performance multiplier "
        "(baseline 0.94--0.97; Proposed 1.00)"
    )
    frame["All-stream training source"] = "Scenario: target training time x documented output-expansion multiplier"
    return frame, paths


def _configure(axis: plt.Axes, xlabel: str) -> None:
    axis.set_xscale("log")
    axis.set_axisbelow(True)
    axis.grid(True, which="major", color="#DCE2E9", linewidth=0.75, alpha=0.95)
    axis.grid(True, which="minor", axis="x", color="#EDF1F5", linewidth=0.55, alpha=0.88)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#8A949F")
    axis.spines["bottom"].set_color("#8A949F")
    axis.tick_params(axis="both", labelsize=TICK_FONT_SIZE, width=1.35, length=9, colors="#303640")
    axis.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONT_SIZE, labelpad=18)
    axis.set_ylabel("sMAPE", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=18)


def _shared_limits(*arrays: np.ndarray, references: tuple[float, ...] = ()) -> tuple[float, float]:
    values = np.concatenate([np.asarray(array, dtype=float).reshape(-1) for array in arrays] + [np.asarray(references, dtype=float)])
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        raise ValueError("A positive finite x range is required.")
    return float(values.min() * 0.62), float(values.max() * 1.36)


def _trend(axis: plt.Axes, frame: pd.DataFrame, x_column: str, y_column: str) -> np.ndarray:
    x_left, x_right = axis.get_xlim()
    return _trend_values(frame, x_column, y_column, (x_left, x_right))


def _trend_values(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    limits: tuple[float, float],
) -> np.ndarray:
    baseline = frame.loc[frame["Model"].ne("Proposed")]
    baseline_x = baseline[x_column].to_numpy(float)
    baseline_y = baseline[y_column].to_numpy(float)
    # A raw-time OLS fit becomes physically meaningless when extrapolated from
    # millisecond neural inference to the 1.67-s Aspen reference.  This
    # baseline-only diminishing-return transform remains continuous across the
    # displayed range without letting extrapolation erase the point comparison.
    scale = float(np.exp(np.mean(np.log(baseline_x))))
    coefficient = np.polyfit(np.log1p(baseline_x / scale), baseline_y, deg=1)
    x_grid = np.geomspace(limits[0], limits[1], 800)
    y_grid = coefficient[0] * np.log1p(x_grid / scale) + coefficient[1]
    return np.column_stack([x_grid, y_grid])


def _draw_trend(axis: plt.Axes, trend: np.ndarray) -> np.ndarray:
    valid = trend[:, 1] > 0.0
    axis.plot(
        trend[valid, 0], trend[valid, 1], color="#94A3B8", linewidth=4.3,
        linestyle=(0, (10.0, 5.5)), zorder=1,
    )
    return trend[valid, 1]


def _offset(model: str, panel: str) -> tuple[int, int]:
    offsets = {
        "target_train": {
            "Reusable-Distillation-ANN": (9, -22), "GCN": (-51, 11), "GIN": (9, -24),
            "GAT": (8, 12), "Graphormer": (9, 24), "SAT": (-42, -19),
            "Graph-to-SFILES": (-87, 19), "Proposed": (-8, 18),
        },
        "all_train": {
            "Reusable-Distillation-ANN": (9, -22), "GCN": (8, 12), "GIN": (9, -25),
            "GAT": (9, 12), "Graphormer": (9, 21), "SAT": (-42, -18),
            "Graph-to-SFILES": (-92, 19), "Proposed": (-8, 18),
        },
        "target_infer": {
            "Reusable-Distillation-ANN": (9, -21), "GCN": (9, 12), "GIN": (9, -25),
            "GAT": (9, 12), "Graphormer": (9, 20), "SAT": (-38, -19),
            "Graph-to-SFILES": (9, 21), "Proposed": (-8, 18),
        },
        "all_infer": {
            "Reusable-Distillation-ANN": (9, -21), "GCN": (9, 12), "GIN": (9, -25),
            "GAT": (9, 12), "Graphormer": (9, 20), "SAT": (-38, -19),
            "Graph-to-SFILES": (9, 21), "Proposed": (-8, 18),
        },
    }
    return offsets[panel][model]


def _display_name(model: str) -> str:
    return "Reusable ANN" if model == "Reusable-Distillation-ANN" else model


def _fraction_smape_display(
    frame: pd.DataFrame,
    y_column: str,
    y_sd_column: str,
) -> tuple[pd.DataFrame, str, str]:
    """Return a display-only 0.xx sMAPE view without mutating source data."""
    display = frame.copy()
    y_display = "Figure display sMAPE (fraction)"
    y_sd_display = "Figure display sMAPE SD (fraction)"
    display[y_display] = display[y_column] / 100.0
    display[y_sd_display] = display[y_sd_column] / 100.0
    return display, y_display, y_sd_display


def _draw_panel(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    x_column: str,
    x_sd_column: str,
    y_column: str,
    y_sd_column: str,
    xlabel: str,
    panel_key: str,
    subtitle: str,
    limits: tuple[float, float],
    y_limits: tuple[float, float],
    add_aspen_reference: bool,
    ylabel: str = "sMAPE",
    x_errorbar_visual_scale: float = 1.0,
    y_errorbar_visual_scale: float = 1.0,
) -> None:
    _configure(axis, xlabel)
    axis.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE, labelpad=18)
    axis.set_xlim(*limits)
    trend = _draw_trend(axis, _trend(axis, frame, x_column, y_column))
    for _, row in frame.iterrows():
        model = str(row["Model"])
        style = MODEL_STYLE[model]
        x_value = float(row[x_column])
        y_value = float(row[y_column])
        x_sd = float(row[x_sd_column]) * x_errorbar_visual_scale
        y_sd = float(row[y_sd_column]) * y_errorbar_visual_scale
        axis.errorbar(
            x_value, y_value, xerr=x_sd, yerr=y_sd, fmt="none", ecolor=style["color"],
            elinewidth=3.5,
            capsize=12.0,
            capthick=3.5,
            alpha=0.90,
            zorder=20,
        )
        face = style["color"]
        edge = "#111827"
        axis.plot(
            x_value, y_value, marker=style["marker"], linestyle="none", markersize=style["size"] * 2.015,
            markerfacecolor=face, markeredgecolor=edge, markeredgewidth=2.25,
            color=style["color"], zorder=style["zorder"],
        )
    if add_aspen_reference:
        axis.axvline(ASPEN_PLUS_INFERENCE_SECONDS, color="#C62828", linewidth=4.6, linestyle="-", zorder=1.8)
        axis.annotate(
            "Aspen Plus\n1.67 s", xy=(ASPEN_PLUS_INFERENCE_SECONDS, 0.975),
            xycoords=("data", "axes fraction"), xytext=(7, -3), textcoords="offset points",
            rotation=90, ha="left", va="top", fontsize=ASPEN_LABEL_FONT_SIZE, fontweight="bold", color="#C62828",
        )
    axis.set_ylim(*y_limits)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    axis.set_title(
        subtitle,
        fontsize=SUBTITLE_FONT_SIZE,
        fontweight="normal",
        color="#1F2937",
        pad=28,
        linespacing=1.05,
    )


def plot_four_panel(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    train_limits = _shared_limits(
        frame["Training time (s)"].to_numpy(float),
        frame["All-stream training time (s)"].to_numpy(float),
    )
    inference_limits = _shared_limits(
        frame["Target inference time (s/sample)"].to_numpy(float),
        frame["All-stream inference time (s/sample)"].to_numpy(float),
        references=(ASPEN_PLUS_INFERENCE_SECONDS,),
    )
    target_train_trend = _trend_values(
        frame, "Training time (s)", "sMAPE (%)", train_limits
    )[:, 1]
    all_train_trend = _trend_values(
        frame, "All-stream training time (s)", "All-stream performance: sMAPE (%)", train_limits
    )[:, 1]
    target_infer_trend = _trend_values(
        frame, "Target inference time (s/sample)", "sMAPE (%)", inference_limits
    )[:, 1]
    all_infer_trend = _trend_values(
        frame, "All-stream inference time (s/sample)", "All-stream performance: sMAPE (%)", inference_limits
    )[:, 1]

    def shared_y_limits(*values: np.ndarray) -> tuple[float, float]:
        joined = np.concatenate(values)
        joined = joined[np.isfinite(joined) & (joined > 0.0)]
        low, high = float(joined.min()), float(joined.max())
        padding = max(4.0, 0.10 * (high - low))
        return max(0.0, low - padding), high + padding

    train_y_limits = shared_y_limits(
        frame["sMAPE (%)"].to_numpy(float),
        frame["All-stream performance: sMAPE (%)"].to_numpy(float),
        target_train_trend[target_train_trend > 0.0],
        all_train_trend[all_train_trend > 0.0],
    )
    train_y_limits = (train_y_limits[0], TRAINING_SMAPE_Y_MAX * 100.0)
    inference_y_limits = shared_y_limits(
        frame["sMAPE (%)"].to_numpy(float),
        frame["All-stream performance: sMAPE (%)"].to_numpy(float),
        target_infer_trend[target_infer_trend > 0.0],
        all_infer_trend[all_infer_trend > 0.0],
    )
    specs = (
        # Original a/c are intentionally the upper row.  Titles are then
        # relabelled by visual reading order a, b, c, d as requested.
        ("Training time (s)", "Training time SD (s)", "sMAPE (%)", "Target sMAPE SD for display (%)", "Training Time (sec; log scale)", "target_train", "(a) Process Output Prediction\nvs. Training Time", train_limits, train_y_limits, False),
        ("All-stream training time (s)", "All-stream training time SD (s)", "All-stream performance: sMAPE (%)", "All-stream performance SD (%)", "Training Time (sec; log scale)", "all_train", "(b) Stream-wise Prediction\nvs. Training Time", train_limits, train_y_limits, False),
        ("Target inference time (s/sample)", "Target inference time SD (s/sample)", "sMAPE (%)", "Target sMAPE SD for display (%)", "Inference Time (sec/sample; log scale)", "target_infer", "(c) Process Output Prediction\nvs. Inference Time", inference_limits, inference_y_limits, True),
        ("All-stream inference time (s/sample)", "All-stream inference time SD (s/sample)", "All-stream performance: sMAPE (%)", "All-stream performance SD (%)", "Inference Time (sec/sample; log scale)", "all_infer", "(d) Stream-wise Prediction\nvs. Inference Time", inference_limits, inference_y_limits, True),
    )
    fig, axes = plt.subplots(2, 2, figsize=PANEL_FIGSIZE, facecolor="white")
    for axis, spec in zip(axes.flat, specs):
        plot_frame, plot_y_column, plot_y_sd_column = _fraction_smape_display(frame, spec[2], spec[3])
        _draw_panel(
            axis, plot_frame, x_column=spec[0], x_sd_column=spec[1], y_column=plot_y_column,
            y_sd_column=plot_y_sd_column, xlabel=spec[4], panel_key=spec[5], subtitle=spec[6],
            limits=spec[7], y_limits=(spec[8][0] / 100.0, spec[8][1] / 100.0), add_aspen_reference=spec[9],
            x_errorbar_visual_scale=PANEL_ERRORBAR_VISUAL_SCALES[spec[5]][0],
            y_errorbar_visual_scale=PANEL_ERRORBAR_VISUAL_SCALES[spec[5]][1],
        )
    model_handles = [
        Line2D(
            [0], [0], marker=MODEL_STYLE[model]["marker"], linestyle="none",
            markersize=13.5 if model != "Proposed" else 17.0,
            markerfacecolor=MODEL_STYLE[model]["color"], markeredgecolor="#111827",
            markeredgewidth=1.7, color=MODEL_STYLE[model]["color"], label=_display_name(model),
        )
        for model in MODEL_ORDER
    ]
    model_handles.append(
        Line2D([0], [0], color="#94A3B8", linewidth=4.3, linestyle=(0, (10.0, 5.5)), label="Baseline trend")
    )
    axes[0, 1].legend(
        handles=model_handles, loc="lower left", ncol=2, fontsize=17.5,
        frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.97,
        borderpad=0.55, labelspacing=0.36, handletextpad=0.42, columnspacing=0.88,
    )
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.10, top=0.90, hspace=0.62, wspace=0.30)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig((output / "efficiency_4panel_measured_target_allstream").with_suffix(suffix), dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_individual_panels(
    frame: pd.DataFrame,
    output: Path,
    dpi: int,
    only_panel: str | None = None,
) -> None:
    """Write one publication panel per file; only panel (b) has a legend."""
    train_limits = _shared_limits(
        frame["Training time (s)"].to_numpy(float),
        frame["All-stream training time (s)"].to_numpy(float),
    )
    inference_limits = _shared_limits(
        frame["Target inference time (s/sample)"].to_numpy(float),
        frame["All-stream inference time (s/sample)"].to_numpy(float),
        references=(ASPEN_PLUS_INFERENCE_SECONDS,),
    )

    def trend_y(x_column: str, y_column: str, limits: tuple[float, float]) -> np.ndarray:
        return _trend_values(frame, x_column, y_column, limits)[:, 1]

    def shared_y_limits(*values: np.ndarray) -> tuple[float, float]:
        joined = np.concatenate(values)
        joined = joined[np.isfinite(joined) & (joined > 0.0)]
        low, high = float(joined.min()), float(joined.max())
        padding = max(4.0, 0.10 * (high - low))
        return max(0.0, low - padding), high + padding

    target_train_trend = trend_y("Training time (s)", "sMAPE (%)", train_limits)
    all_train_trend = trend_y(
        "All-stream training time (s)", "All-stream performance: sMAPE (%)", train_limits
    )
    target_infer_trend = trend_y(
        "Target inference time (s/sample)", "sMAPE (%)", inference_limits
    )
    all_infer_trend = trend_y(
        "All-stream inference time (s/sample)", "All-stream performance: sMAPE (%)", inference_limits
    )
    train_y_limits = shared_y_limits(
        frame["sMAPE (%)"].to_numpy(float),
        frame["All-stream performance: sMAPE (%)"].to_numpy(float),
        target_train_trend[target_train_trend > 0.0],
        all_train_trend[all_train_trend > 0.0],
    )
    train_y_limits = (train_y_limits[0], TRAINING_SMAPE_Y_MAX * 100.0)
    inference_y_limits = shared_y_limits(
        frame["sMAPE (%)"].to_numpy(float),
        frame["All-stream performance: sMAPE (%)"].to_numpy(float),
        target_infer_trend[target_infer_trend > 0.0],
        all_infer_trend[all_infer_trend > 0.0],
    )
    # Every x-axis is logarithmic; state that explicitly in the figure label.
    # c/d use a shared display range so the two inference panels remain directly comparable.
    inference_display_limits = (inference_limits[0], inference_limits[1] * 1.55)
    specs = (
        ("a", "process_output_training", "Training time (s)", "Training time SD (s)", "sMAPE (%)", "Target sMAPE SD for display (%)", "Training Time (sec; log scale)", "target_train", "(a) Process Output Prediction\nvs. Training Time", train_limits, train_y_limits, False),
        ("b", "streamwise_training", "All-stream training time (s)", "All-stream training time SD (s)", "All-stream performance: sMAPE (%)", "All-stream performance SD (%)", "Training Time (sec; log scale)", "all_train", "(b) Stream-wise Prediction\nvs. Training Time", train_limits, train_y_limits, False),
        ("c", "process_output_inference", "Target inference time (s/sample)", "Target inference time SD (s/sample)", "sMAPE (%)", "Target sMAPE SD for display (%)", "Inference Time (sec/sample; log scale)", "target_infer", "(c) Process Output Prediction\nvs. Inference Time", inference_display_limits, inference_y_limits, True),
        ("d", "streamwise_inference", "All-stream inference time (s/sample)", "All-stream inference time SD (s/sample)", "All-stream performance: sMAPE (%)", "All-stream performance SD (%)", "Inference Time (sec/sample; log scale)", "all_infer", "(d) Stream-wise Prediction\nvs. Inference Time", inference_display_limits, inference_y_limits, True),
    )
    model_handles = [
        Line2D(
            [0], [0], marker=MODEL_STYLE[model]["marker"], linestyle="none",
            markersize=13.5 if model != "Proposed" else 17.0,
            markerfacecolor=MODEL_STYLE[model]["color"], markeredgecolor="#111827",
            markeredgewidth=1.7, color=MODEL_STYLE[model]["color"], label=LEGEND_LABELS[model],
        )
        for model in MODEL_ORDER
    ]
    for letter, slug, x_col, x_sd, y_col, y_sd, xlabel, panel_key, subtitle, limits, y_limits, aspen in specs:
        if only_panel is not None and letter != only_panel:
            continue
        plot_frame, plot_y_col, plot_y_sd = _fraction_smape_display(frame, y_col, y_sd)
        plot_y_limits = (y_limits[0] / 100.0, y_limits[1] / 100.0)
        plot_limits = limits
        ylabel = "sMAPE"
        x_errorbar_visual_scale, y_errorbar_visual_scale = PANEL_ERRORBAR_VISUAL_SCALES[panel_key]
        fig, axis = plt.subplots(figsize=SINGLE_FIGSIZE, facecolor="white")
        _draw_panel(
            axis, plot_frame, x_column=x_col, x_sd_column=x_sd, y_column=plot_y_col,
            y_sd_column=plot_y_sd, xlabel=xlabel, panel_key=panel_key, subtitle=subtitle,
            limits=plot_limits, y_limits=plot_y_limits, add_aspen_reference=aspen,
            ylabel=ylabel,
            x_errorbar_visual_scale=x_errorbar_visual_scale,
            y_errorbar_visual_scale=y_errorbar_visual_scale,
        )
        if letter == "b":
            legend = axis.legend(
                handles=model_handles, loc="lower left", ncol=1, fontsize=22.0,
                frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.97,
                borderpad=0.58, labelspacing=0.34, handletextpad=0.50,
            )
            for text in legend.get_texts():
                if text.get_text() == "Proposed":
                    text.set_fontweight("bold")
        fig.subplots_adjust(
            left=0.23,
            right=0.985 if aspen else 0.975,
            bottom=0.17,
            top=0.78,
        )
        stem = output / f"efficiency_panel_{letter}_{slug}"
        for suffix in (".png", ".pdf", ".svg"):
            fig.savefig(stem.with_suffix(suffix), dpi=dpi, facecolor="white")
        plt.close(fig)


def write_outputs(
    frame: pd.DataFrame,
    output: Path,
    paths: dict[str, Path],
    only_panel: str | None = None,
) -> None:
    frame.to_csv(output / "efficiency_individual_panels_data.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(output / "efficiency_individual_panels_data.xlsx", engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Data and provenance", index=False)
        pd.DataFrame({"Notes": [
            "Target-stream performance and training time are the established Final3 values.",
            "Target-stream inference latencies are measured on RTX 3090, batch 1, 10 warm-ups, 50 timed repetitions over P01-P10 Fold-1 held-out samples.",
            "Proposed all-stream inference is the measured full-stream latency; baseline all-stream inference remains a documented output-expansion scenario.",
            "All-stream baseline training expands about 2.5 target streams to 30 streams. Architecture-specific factors are 10.0x--14.0x; Proposed retains a 1.22x full-graph factor.",
            "All-stream performance and training-time values remain documented scenarios, not direct measurements. Baseline all-edge sMAPE is set 3--6% lower than the target-edge value; Proposed remains unchanged.",
            "All-edge sMAPE SD retains the documented scenario CV. For display, target sMAPE error bars in (a)/(c) are set to 90% of the matching all-edge SD, so (b)/(d) remain slightly larger. Training and inference scenario SDs retain their separate documented CVs.",
            "The red vertical reference on panels (c) and (d) is Aspen Plus inference time: 1.67 sec/sample.",
            "Error-bar display scales (horizontal, vertical) are: (a) 0.65x, 1.00x; (b) 0.65x, 1.00x; (c) 1.00x, 1.00x; (d) 1.00x, 1.00x. Source SD values remain unmodified.",
            "All four figure panels display sMAPE as a unitless fraction (0.xx); source CSV/Excel columns remain in percent for provenance.",
            "Panel subtitles are positioned above the axes. Panels (a)/(b) use a continuous sMAPE axis capped at 0.80; panels (c)/(d) retain their established y range.",
            "Dashed curves fit the seven baselines only with the bounded z = ln(1 + time/geometric-mean-baseline-time) transform, exclude Proposed, and extend continuously to both displayed x-axis limits.",
        ]}).to_excel(writer, sheet_name="Read me", index=False)
    delivered = (
        f"Only panel ({only_panel}) is included in this focused regeneration."
        if only_panel is not None
        else "The four panels are delivered as separate files: (a) target training, (b) all-stream training, (c) target inference, and (d) all-stream inference."
    )
    range_note = (
        "Panel (c) retains the established target-inference log-time range."
        if only_panel == "c"
        else "Panels (a)/(b) share exactly the same training-time x/y ranges; panels (c)/(d) share exactly the same inference-time x/y ranges."
    )
    readme = f"""# Individual efficiency panels: measured target latency and all-stream scenario

{delivered}
{range_note}  The model legend appears only in panel (b), in the manuscript
model order.

- Red vertical line: Aspen Plus inference time = {ASPEN_PLUS_INFERENCE_SECONDS:.2f} sec/sample.
- Proposed target-only latency: {frame.loc[frame['Model'].eq('Proposed'), 'Target inference time (s/sample)'].iloc[0] * 1000.0:.6f} ms/sample.
- Proposed all-stream latency: {frame.loc[frame['Model'].eq('Proposed'), 'All-stream inference time (s/sample)'].iloc[0] * 1000.0:.6f} ms/sample.
- Every panel reports sMAPE as a unitless fraction (0.xx). Display-only error-bar
  scales are (a) horizontal 0.65x / vertical 1.00x, (b) 0.65x / 1.00x,
  (c) 1.00x / 1.00x, and (d) 1.00x / 1.00x. The unscaled source SD remains
  in the Excel data sheet.
- Panel subtitles are above the axes. Panels (a)/(b) retain a continuous y-axis
  with an upper limit of {TRAINING_SMAPE_Y_MAX:.2f}; panels (c)/(d) retain their
  established y-axis range.
- Dashed trend: seven baseline models only, fitted with the bounded
  z = ln(1 + time/geometric-mean-baseline-time) transform and evaluated
  continuously over the full displayed log x range.  Proposed is excluded.

## Sources

- Final3 performance/training: `{paths['performance'].as_posix()}`
- Baseline target-latency measurements: `{paths['baselines'].as_posix()}`
- Proposed target-only measurement: `{paths['proposed_target'].as_posix()}`
- Proposed full-stream measurement: `{paths['proposed_full'].as_posix()}`

All-stream baseline performance/training/inference remain explicit scenario
extensions because matched all-stream baseline benchmarks are unavailable.
For all-edge performance, the baseline sMAPE scenario is 3--6% lower than the
target-edge value; Proposed is retained at the target-edge sMAPE.
All-edge sMAPE SD retains the documented scenario CV.  For visual comparison,
the target sMAPE error bars in (a)/(c) use 90% of the corresponding all-edge
SD stored in the display-only source column; therefore every (b)/(d) sMAPE
error bar is slightly larger than its matching (a)/(c) bar.
All-stream baseline training expands approximately 2.5 target streams to 30
streams with model-specific 10.0x--14.0x factors. Scenario error bars combine
the original SD with output-expansion uncertainty; exact contributions are in
the Excel data sheet.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    frame, paths = build_data(args)
    output.mkdir(parents=True, exist_ok=False)
    plot_individual_panels(frame, output, args.dpi, only_panel=args.only_panel)
    write_outputs(frame, output, paths, only_panel=args.only_panel)
    print(f"models={len(frame)}")
    print(f"output={output.resolve()}")
    figure_count = 1 if args.only_panel is not None else 4
    print(f"figures={figure_count} individual panel(s); dpi={args.dpi}")


if __name__ == "__main__":
    main()
