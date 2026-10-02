#!/usr/bin/env python3
"""Plot empirical target-stream efficiency with measured inference latency.

The training-time/performance panel uses the established Final3 multi-process
table.  The inference-time/performance panel uses the 10-process, batch-one
latency measurements collected on the RTX 3090.  The baseline-only trend is a
visual linear fit in original time coordinates, drawn continuously across the
entire displayed logarithmic horizontal axis; Proposed is never used to fit it.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


MODEL_ORDER = (
    "Reusable-Distillation-ANN",
    "GCN",
    "GIN",
    "GAT",
    "Graphormer",
    "SAT",
    "Graph-to-SFILES",
    "Proposed",
)

BENCHMARK_TO_DISPLAY = {
    "B6": "Reusable-Distillation-ANN",
    "GCN": "GCN",
    "GIN": "GIN",
    "GAT": "GAT",
    "Graphormer": "Graphormer",
    "SAT": "SAT",
    "GraphToSFILES": "Graph-to-SFILES",
    "Proposed": "Proposed",
}

MODEL_STYLE = {
    "Proposed": {"color": "#111111", "marker": "*", "size": 17.0, "zorder": 8},
    "GCN": {"color": "#0072B2", "marker": "o", "size": 8.4, "zorder": 5},
    "GIN": {"color": "#E69F00", "marker": "s", "size": 8.0, "zorder": 5},
    "GAT": {"color": "#009E73", "marker": "^", "size": 8.5, "zorder": 5},
    "Graphormer": {"color": "#CC79A7", "marker": "D", "size": 8.0, "zorder": 5},
    "SAT": {"color": "#D55E00", "marker": "v", "size": 8.5, "zorder": 5},
    "Graph-to-SFILES": {"color": "#56B4E9", "marker": "P", "size": 8.5, "zorder": 5},
    "Reusable-Distillation-ANN": {"color": "#6B7280", "marker": "X", "size": 8.4, "zorder": 5},
}

SINGLE_FIGSIZE = (10.5, 7.2)
PANEL_FIGSIZE = (20.8, 7.2)


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
        help="Recorded in the data table for scope transparency; not used for the target-only comparison panel.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/"
            "efficiency_analysis_measured_target_inference_20260917"
        ),
    )
    parser.add_argument("--dpi", type=int, default=1200)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def load_data(
    performance_path: Path,
    baseline_path: Path,
    proposed_target_path: Path,
    proposed_full_path: Path,
) -> pd.DataFrame:
    if not performance_path.is_file():
        raise FileNotFoundError(performance_path)
    performance = pd.read_csv(performance_path)
    required = {
        "Model",
        "Training time (s)",
        "Training time SD (s)",
        "sMAPE (%)",
        "sMAPE SD (%)",
    }
    missing = required.difference(performance.columns)
    if missing:
        raise ValueError(f"Performance CSV lacks columns: {sorted(missing)}")
    performance = performance.copy()
    performance["Model"] = performance["Model"].replace(
        {"GraphToSFILES": "Graph-to-SFILES"}
    )
    performance = performance.set_index("Model")
    absent = [model for model in MODEL_ORDER if model not in performance.index]
    if absent:
        raise ValueError(f"Performance CSV lacks models: {absent}")

    baseline_payload = _read_json(baseline_path)
    records = baseline_payload.get("records")
    if not isinstance(records, list):
        raise TypeError(f"Benchmark records missing from: {baseline_path}")
    inference: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        raw_name = str(record.get("model", ""))
        model = BENCHMARK_TO_DISPLAY.get(raw_name)
        if model is None or model == "Proposed":
            continue
        inference[model] = record

    target_proposed = _read_json(proposed_target_path)
    full_proposed = _read_json(proposed_full_path)
    inference["Proposed"] = target_proposed
    missing_benchmarks = [model for model in MODEL_ORDER if model not in inference]
    if missing_benchmarks:
        raise ValueError(f"Missing measured inference records for: {missing_benchmarks}")

    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        perf_row = performance.loc[model]
        bench = inference[model]
        latency = float(bench["inference_time_per_sample_ms"])
        latency_sd = float(bench["inference_time_std_ms"])
        if not (math.isfinite(latency) and latency > 0.0):
            raise ValueError(f"Invalid inference latency for {model}: {latency}")
        rows.append(
            {
                "Model": model,
                "sMAPE (%)": float(perf_row["sMAPE (%)"]),
                "sMAPE SD (%)": float(perf_row["sMAPE SD (%)"]),
                "Training time (s)": float(perf_row["Training time (s)"]),
                "Training time SD (s)": float(perf_row["Training time SD (s)"]),
                "Inference time (ms/sample)": latency,
                "Inference time SD (ms/sample)": latency_sd,
                "Inference throughput (samples/s)": float(bench["throughput_samples_per_sec"]),
                "Inference timing scope": str(bench.get("timing_scope", "")),
                "Inference checkpoint provenance": str(bench.get("checkpoint_provenance", "")),
                "Inference benchmark source": str(
                    proposed_target_path if model == "Proposed" else baseline_path
                ),
                "Proposed full-stream inference time (ms/sample)": (
                    float(full_proposed["inference_time_per_sample_ms"])
                    if model == "Proposed"
                    else math.nan
                ),
                "Proposed full-stream inference time SD (ms/sample)": (
                    float(full_proposed["inference_time_std_ms"])
                    if model == "Proposed"
                    else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _configure_axis(axis: plt.Axes, xlabel: str) -> None:
    axis.set_xscale("log")
    axis.set_axisbelow(True)
    axis.grid(True, which="major", color="#DCE2E9", linewidth=0.8, alpha=0.95)
    axis.grid(True, which="minor", axis="x", color="#EDF1F5", linewidth=0.6, alpha=0.88)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#8A949F")
    axis.spines["bottom"].set_color("#8A949F")
    axis.tick_params(axis="both", labelsize=17, width=1.25, length=7, colors="#303640")
    axis.set_xlabel(xlabel, fontsize=20, labelpad=11)
    axis.set_ylabel("sMAPE (%)", fontsize=20, labelpad=11)


def _x_limits(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values) & (values > 0.0)]
    if finite.size == 0:
        raise ValueError("Positive finite x values are required.")
    return float(np.min(finite) * 0.72), float(np.max(finite) * 1.38)


def _add_baseline_trend(axis: plt.Axes, frame: pd.DataFrame, x_column: str) -> np.ndarray:
    baseline = frame.loc[frame["Model"].ne("Proposed")]
    x = baseline[x_column].to_numpy(float)
    y = baseline["sMAPE (%)"].to_numpy(float)
    coefficient = np.polyfit(x, y, deg=1)
    x_left, x_right = axis.get_xlim()
    x_grid = np.geomspace(x_left, x_right, 600)
    y_grid = coefficient[0] * x_grid + coefficient[1]
    axis.plot(
        x_grid,
        y_grid,
        color="#94A3B8",
        linewidth=2.35,
        linestyle=(0, (5.5, 3.0)),
        zorder=1,
        label="Baseline trend",
    )
    return y_grid


def _annotation_offset(model: str, panel: str) -> tuple[int, int]:
    offsets = {
        "training": {
            "Reusable-Distillation-ANN": (-4, -27), "GCN": (-49, 15),
            "GIN": (11, -28), "GAT": (12, 14), "Graphormer": (11, 28),
            "SAT": (-41, -23), "Graph-to-SFILES": (-102, 24), "Proposed": (-8, 17),
        },
        "inference": {
            "Reusable-Distillation-ANN": (10, -25), "GCN": (10, 15),
            "GIN": (10, -27), "GAT": (10, 15), "Graphormer": (10, 18),
            "SAT": (-40, -24), "Graph-to-SFILES": (10, 18), "Proposed": (-10, 18),
        },
    }
    return offsets[panel][model]


def _display_name(model: str) -> str:
    return model.replace("Reusable-Distillation-ANN", "Reusable ANN")


def _plot_points(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    x_column: str,
    x_sd_column: str,
    panel: str,
) -> None:
    for _, row in frame.iterrows():
        model = str(row["Model"])
        style = MODEL_STYLE[model]
        x_value = float(row[x_column])
        y_value = float(row["sMAPE (%)"])
        x_sd = float(row[x_sd_column])
        y_sd = float(row["sMAPE SD (%)"])
        axis.errorbar(
            x_value,
            y_value,
            xerr=x_sd,
            yerr=y_sd,
            fmt="none",
            ecolor=style["color"],
            elinewidth=1.45,
            capsize=4.5,
            capthick=1.35,
            alpha=0.75,
            zorder=2,
        )
        axis.plot(
            x_value,
            y_value,
            marker=style["marker"],
            linestyle="none",
            markersize=style["size"],
            markerfacecolor=style["color"],
            markeredgecolor="#111111",
            markeredgewidth=1.25 if model != "Proposed" else 1.1,
            color=style["color"],
            zorder=style["zorder"],
        )
        axis.annotate(
            _display_name(model),
            xy=(x_value, y_value),
            xytext=_annotation_offset(model, panel),
            textcoords="offset points",
            fontsize=13.5 if model != "Proposed" else 15.0,
            fontweight="bold" if model == "Proposed" else "normal",
            color="#111111",
            ha="right" if model == "Proposed" else "left",
            va="center",
            zorder=10,
        )


def _finish_axis(axis: plt.Axes, *, y_values: np.ndarray, subtitle: str, compact: bool) -> None:
    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    padding = max(3.4, 0.07 * (y_max - y_min))
    axis.set_ylim(max(0.0, y_min - padding), y_max + padding)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    trend_handle = Line2D([0], [0], color="#94A3B8", linewidth=2.35, linestyle=(0, (5.5, 3.0)))
    axis.legend(
        [trend_handle],
        ["Baseline-only trend"],
        loc="upper left",
        fontsize=12.5 if not compact else 11.0,
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        framealpha=0.96,
        borderpad=0.45,
        handlelength=2.4,
    )
    axis.text(
        0.5,
        -0.34 if not compact else -0.31,
        subtitle,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=21 if not compact else 17.5,
        fontweight="bold",
        color="#1F2937",
    )


def plot_one(
    frame: pd.DataFrame,
    *,
    x_column: str,
    x_sd_column: str,
    xlabel: str,
    panel: str,
    subtitle: str,
    stem: Path,
    dpi: int,
) -> None:
    fig, axis = plt.subplots(figsize=SINGLE_FIGSIZE, facecolor="white")
    _configure_axis(axis, xlabel)
    x_values = frame[x_column].to_numpy(float)
    axis.set_xlim(*_x_limits(x_values))
    trend = _add_baseline_trend(axis, frame, x_column)
    _plot_points(axis, frame, x_column=x_column, x_sd_column=x_sd_column, panel=panel)
    _finish_axis(
        axis,
        y_values=np.concatenate([frame["sMAPE (%)"].to_numpy(float), trend]),
        subtitle=subtitle,
        compact=False,
    )
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.31, top=0.96)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(stem.with_suffix(suffix), dpi=dpi, facecolor="white")
    plt.close(fig)


def plot_panel(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    specs = (
        (
            "Training time (s)", "Training time SD (s)", "Joint Training Time (s)",
            "training", "(a) Performance vs. Training Time",
        ),
        (
            "Inference time (ms/sample)", "Inference time SD (ms/sample)", "Inference Time (ms/sample)",
            "inference", "(b) Performance vs. Inference Time",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=PANEL_FIGSIZE, facecolor="white")
    for axis, (x_col, x_sd, xlabel, panel, subtitle) in zip(axes, specs):
        _configure_axis(axis, xlabel)
        axis.set_xlim(*_x_limits(frame[x_col].to_numpy(float)))
        trend = _add_baseline_trend(axis, frame, x_col)
        _plot_points(axis, frame, x_column=x_col, x_sd_column=x_sd, panel=panel)
        _finish_axis(
            axis,
            y_values=np.concatenate([frame["sMAPE (%)"].to_numpy(float), trend]),
            subtitle=subtitle,
            compact=True,
        )
    fig.subplots_adjust(left=0.075, right=0.988, bottom=0.29, top=0.96, wspace=0.23)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig((output / "efficiency_measured_1x2_panel").with_suffix(suffix), dpi=dpi, facecolor="white")
    plt.close(fig)


def write_readme(frame: pd.DataFrame, output: Path, paths: dict[str, Path]) -> None:
    proposed = frame.loc[frame["Model"].eq("Proposed")].iloc[0]
    text = f"""# Measured target-stream efficiency analysis

This package supersedes the earlier latency *estimates* only for the measured
target-stream inference panel.

- Hardware: NVIDIA RTX 3090, CUDA:0.
- Protocol: one held-out Fold-1 sample from each of P01-P10, batch size 1,
  10 warm-ups, and 50 timed repetitions.
- Proposed target-edge-only latency: {proposed['Inference time (ms/sample)']:.6f} +/- {proposed['Inference time SD (ms/sample)']:.6f} ms/sample.
- Proposed full-stream latency: {proposed['Proposed full-stream inference time (ms/sample)']:.6f} +/- {proposed['Proposed full-stream inference time SD (ms/sample)']:.6f} ms/sample.  It is retained in the data table but is not used in the target-output comparison panel.
- The dashed trend is fitted using the seven baselines only in original-time
  coordinates, then evaluated continuously from the left to the right display
  limit on the log-scaled x-axis.  Proposed is excluded from its fit.

## Provenance

- Performance/training-time table: `{paths['performance'].as_posix()}`
- Baseline latency benchmark: `{paths['baselines'].as_posix()}`
- Proposed target-only benchmark: `{paths['proposed_target'].as_posix()}`
- Proposed full-stream benchmark: `{paths['proposed_full'].as_posix()}`

The baseline benchmark JSON records checkpoint provenance per row.  Some
baseline architectures were timed with smoke or architecture-only weights; the
latency is useful for this matched implementation/hardware comparison, but the
checkpoint caveat remains in `efficiency_measured_data.csv` and must be
resolved before describing every baseline point as a final-checkpoint latency.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = {
        "performance": Path(args.performance_csv),
        "baselines": Path(args.baseline_benchmark_json),
        "proposed_target": Path(args.proposed_target_benchmark_json),
        "proposed_full": Path(args.proposed_full_benchmark_json),
    }
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    frame = load_data(
        paths["performance"],
        paths["baselines"],
        paths["proposed_target"],
        paths["proposed_full"],
    )
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "efficiency_measured_data.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(output / "efficiency_measured_data.xlsx", engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Measured target inference", index=False)
    plot_one(
        frame,
        x_column="Training time (s)",
        x_sd_column="Training time SD (s)",
        xlabel="Joint Training Time (s)",
        panel="training",
        subtitle="(a) Performance vs. Training Time",
        stem=output / "efficiency_vs_training_time",
        dpi=args.dpi,
    )
    plot_one(
        frame,
        x_column="Inference time (ms/sample)",
        x_sd_column="Inference time SD (ms/sample)",
        xlabel="Inference Time (ms/sample)",
        panel="inference",
        subtitle="(b) Performance vs. Inference Time",
        stem=output / "efficiency_vs_inference_time",
        dpi=args.dpi,
    )
    plot_panel(frame, output, args.dpi)
    write_readme(frame, output, paths)
    print(f"models={len(frame)}")
    print(f"output={output.resolve()}")
    print("figures=3; formats=png,pdf,svg; dpi=" + str(args.dpi))


if __name__ == "__main__":
    main()
