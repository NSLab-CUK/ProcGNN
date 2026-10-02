#!/usr/bin/env python3
"""Create a transparent observed-versus-estimated efficiency-analysis package.

Target-stream mean sMAPE and training time are read from the existing Final3
artifact.  No matched inference-latency or all-stream benchmark is archived,
so those values are deterministic scenario estimates explicitly marked as such
in every tabular and figure artifact.  The estimates are for figure planning
only and must be replaced by matched-hardware measurements before publication.
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
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


MODEL_ORDER = [
    "Proposed",
    "GCN",
    "GIN",
    "GAT",
    "Graphormer",
    "SAT",
    "Graph-to-SFILES",
    "Reusable-Distillation-ANN",
]

SOURCE_MODEL_NAME = {
    "Graph-to-SFILES": "GraphToSFILES",
}

MODEL_STYLE = {
    "Proposed": {"color": "#111111", "marker": "*", "size": 16.0, "zorder": 8},
    "GCN": {"color": "#0072B2", "marker": "o", "size": 7.8, "zorder": 5},
    "GIN": {"color": "#E69F00", "marker": "s", "size": 7.4, "zorder": 5},
    "GAT": {"color": "#009E73", "marker": "^", "size": 8.0, "zorder": 5},
    "Graphormer": {"color": "#CC79A7", "marker": "D", "size": 7.4, "zorder": 5},
    "SAT": {"color": "#D55E00", "marker": "v", "size": 8.0, "zorder": 5},
    "Graph-to-SFILES": {"color": "#56B4E9", "marker": "P", "size": 8.0, "zorder": 5},
    "Reusable-Distillation-ANN": {"color": "#6B7280", "marker": "X", "size": 7.8, "zorder": 5},
}

# Match the publication transfer/sensitivity figures: 10.5 x 7.2 in for a
# stand-alone panel.  The composite keeps this same aspect ratio at 2 x scale.
PAPER_SINGLE_FIGSIZE = (10.5, 7.2)
PAPER_PANEL_FIGSIZE = (16.8, 11.52)

# Estimates are intentionally fixed, documented, and conservative.  They are
# not sampled and do not masquerade as uncertainty measurements.
ALL_STREAM_ERROR_MULTIPLIER = {
    "Proposed": 1.08,
    "GCN": 1.24,
    "GIN": 1.22,
    "GAT": 1.26,
    "Graphormer": 1.14,
    "SAT": 1.17,
    "Graph-to-SFILES": 1.28,
    "Reusable-Distillation-ANN": 1.32,
}
ALL_STREAM_TRAIN_MULTIPLIER = {
    "Proposed": 1.22,
    "GCN": 1.30,
    "GIN": 1.28,
    "GAT": 1.32,
    "Graphormer": 1.25,
    "SAT": 1.27,
    "Graph-to-SFILES": 1.38,
    "Reusable-Distillation-ANN": 1.35,
}
TARGET_INFERENCE_SEC_PER_1000 = {
    "Proposed": 1.85,
    "GCN": 0.35,
    "GIN": 0.42,
    "GAT": 0.58,
    "Graphormer": 1.15,
    "SAT": 1.32,
    "Graph-to-SFILES": 0.95,
    "Reusable-Distillation-ANN": 0.18,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--observed-source",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/"
            "Paper_Tables_Final3_computational_efficiency/"
            "computational_efficiency_mae_smape.csv"
        ),
        help="Final3 observed target-stream sMAPE/training-time CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/"
            "efficiency_analysis_observed_estimated_20260916"
        ),
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def _model_from_source(value: str) -> str:
    return "Graph-to-SFILES" if value == "GraphToSFILES" else value


def _source_name(model: str) -> str:
    return SOURCE_MODEL_NAME.get(model, model)


def load_observed(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "Model", "Training time (s)", "Training time SD (s)", "sMAPE (%)", "sMAPE SD (%)",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Observed source lacks columns: {sorted(missing)}")
    frame = frame.copy()
    frame["Model"] = frame["Model"].astype(str).map(_model_from_source)
    frame = frame.set_index("Model")
    absent = [model for model in MODEL_ORDER if model not in frame.index]
    if absent:
        raise ValueError(f"Observed source lacks models: {absent}")
    frame = frame.loc[MODEL_ORDER].reset_index()
    for column in required.difference({"Model"}):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (frame[["Training time (s)", "sMAPE (%)"]] <= 0).any().any():
        raise ValueError("Observed time and sMAPE must be strictly positive for log-scale plots.")
    return frame


def build_final_data(observed: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in observed.itertuples(index=False):
        model = str(row.Model)
        # Attribute names in itertuples are unstable for parenthesized headers;
        # values are obtained from the original columns below to stay explicit.
        record = {
            "Model": model,
            "Target-stream performance: mean sMAPE (%)": float(
                observed.loc[observed["Model"].eq(model), "sMAPE (%)"].iloc[0]
            ),
            "Target-stream performance SD (%)": float(
                observed.loc[observed["Model"].eq(model), "sMAPE SD (%)"].iloc[0]
            ),
            "All-stream performance: mean sMAPE (%)": float(
                observed.loc[observed["Model"].eq(model), "sMAPE (%)"].iloc[0]
                * ALL_STREAM_ERROR_MULTIPLIER[model]
            ),
            "All-stream performance SD (%)": math.nan,
            "Training time, target-stream (s)": float(
                observed.loc[observed["Model"].eq(model), "Training time (s)"].iloc[0]
            ),
            "Training time, target-stream SD (s)": float(
                observed.loc[observed["Model"].eq(model), "Training time SD (s)"].iloc[0]
            ),
            "Inference time, target-stream (s per 1,000 samples)": TARGET_INFERENCE_SEC_PER_1000[model],
            "Inference time, target-stream SD": math.nan,
            "Training time, all-stream (s)": float(
                observed.loc[observed["Model"].eq(model), "Training time (s)"].iloc[0]
                * ALL_STREAM_TRAIN_MULTIPLIER[model]
            ),
            "Training time, all-stream SD": math.nan,
            "Inference time, all-stream (s per 1,000 samples)": (
                TARGET_INFERENCE_SEC_PER_1000[model] * ALL_STREAM_INFERENCE_MULTIPLIER[model]
            ),
            "Inference time, all-stream SD": math.nan,
            "Target-stream performance source": "Observed: Final3 5-fold mean sMAPE",
            "Target-stream training source": "Observed: Final3 5-fold completed-run metadata",
            "Target-stream inference source": "Estimated: no archived matched inference-latency benchmark",
            "All-stream performance source": "Estimated: target sMAPE × documented all-stream degradation multiplier",
            "All-stream training source": "Estimated: target training time × documented output-expansion multiplier",
            "All-stream inference source": "Estimated: target inference time × documented output-expansion multiplier",
            "All-stream error multiplier": ALL_STREAM_ERROR_MULTIPLIER[model],
            "All-stream training multiplier": ALL_STREAM_TRAIN_MULTIPLIER[model],
            "All-stream inference multiplier": ALL_STREAM_INFERENCE_MULTIPLIER[model],
            "Observed target data source": source_path.as_posix(),
            "Data source": "Observed target performance/training + estimated inference/all-stream",
        }
        rows.append(record)
    return pd.DataFrame(rows)


def _configure_axis(axis: plt.Axes, *, xlabel: str) -> None:
    axis.set_xscale("log")
    axis.grid(True, which="major", color="#DCE2E9", linewidth=0.75, alpha=0.95)
    axis.grid(True, which="minor", axis="x", color="#EDF1F5", linewidth=0.55, alpha=0.85)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#8A949F")
    axis.spines["bottom"].set_color("#8A949F")
    axis.tick_params(axis="both", labelsize=18, width=1.2, length=7, colors="#303640")
    axis.set_xlabel(xlabel, fontsize=20, labelpad=12)
    axis.set_ylabel("sMAPE (%)", fontsize=20, labelpad=12)


def _trend(axis: plt.Axes, x: np.ndarray, y: np.ndarray, *, display_x: np.ndarray) -> None:
    # This is deliberately a baseline-only guide.  Proposed is plotted and
    # annotated, but excluded because its much larger time range would dominate
    # the least-squares fit.  Fitting in original time coordinates makes the
    # guide curve on the logarithmic horizontal axis.  `display_x` spans all
    # plotted models, so the baseline-only guide remains continuous through
    # the Proposed model's time range for a direct visual comparison.
    coef = np.polyfit(x, y, deg=1)
    x_grid = np.geomspace(float(np.min(display_x)) * 0.82, float(np.max(display_x)) * 1.22, 360)
    y_grid = coef[0] * x_grid + coef[1]
    axis.plot(x_grid, y_grid, color="#94A3B8", linewidth=2.1, linestyle=(0, (5, 3)), zorder=1)


def _annotate(axis: plt.Axes, model: str, x: float, y: float, *, panel: str) -> None:
    # Offset choices keep the dense baseline cluster legible without changing
    # a point's location or relying on a fitted label-placement package.
    offsets = {
        "target_train": {
            "Reusable-Distillation-ANN": (10, -30), "GCN": (-76, 12), "GIN": (10, -31),
            "GAT": (21, 12), "Graphormer": (20, 29), "SAT": (-54, -23),
            "Graph-to-SFILES": (-84, 25), "Proposed": (-18, 19),
        },
        "target_infer": {
            "Reusable-Distillation-ANN": (10, -25), "GCN": (10, 15), "GIN": (10, -27),
            "GAT": (10, 15), "Graphormer": (10, 17), "SAT": (10, -24),
            "Graph-to-SFILES": (10, 17), "Proposed": (-18, 19),
        },
        "all_train": {
            "Reusable-Distillation-ANN": (10, -25), "GCN": (10, 16), "GIN": (10, -29),
            "GAT": (10, 17), "Graphormer": (10, 16), "SAT": (38, -24),
            "Graph-to-SFILES": (10, 17), "Proposed": (-18, 19),
        },
        "all_infer": {
            "Reusable-Distillation-ANN": (10, -25), "GCN": (10, 16), "GIN": (10, -29),
            "GAT": (10, 17), "Graphormer": (10, 16), "SAT": (10, -24),
            "Graph-to-SFILES": (10, 17), "Proposed": (-18, 19),
        },
    }
    display = model.replace("Reusable-Distillation-ANN", "Reusable ANN").replace("Graph-to-SFILES", "Graph-to-SFILES")
    axis.annotate(
        display,
        xy=(x, y),
        xytext=offsets[panel][model],
        textcoords="offset points",
        fontsize=13.0 if model != "Proposed" else 14.5,
        fontweight="bold" if model == "Proposed" else "normal",
        color="#111111",
        ha="right" if model == "Proposed" else "left",
        va="center",
        zorder=10,
    )


def _format_title(title: str) -> str:
    """Keep publication titles large without allowing long strings to clip."""
    return title.replace(" vs. ", "\nvs. ")


def _source_key(axis: plt.Axes, text: str, *, estimated: bool) -> None:
    face = "white" if estimated else "#334155"
    edge = "#334155"
    marker = "o" if estimated else "o"
    handle = Line2D(
        [0], [0], marker=marker, linestyle="none", markersize=10.5,
        markerfacecolor=face, markeredgecolor=edge, markeredgewidth=1.0,
        label=text,
    )
    axis.legend(
        handles=[handle], loc="upper left", fontsize=13.0, frameon=True,
        facecolor="white", edgecolor="#CBD5E1", framealpha=0.96,
        handletextpad=0.45, borderpad=0.45,
    )


def plot_one(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    x_sd_column: str | None,
    y_sd_column: str | None,
    title: str,
    panel_key: str,
    source_text: str,
    estimated_coordinate: bool,
    stem: Path,
    dpi: int,
) -> None:
    fig, axis = plt.subplots(figsize=PAPER_SINGLE_FIGSIZE, facecolor="white")
    _configure_axis(axis, xlabel=("Training time (s)" if "Training" in x_column else "Inference time (s)"))
    x = frame[x_column].to_numpy(float)
    y = frame[y_column].to_numpy(float)
    baseline = frame.loc[frame["Model"].ne("Proposed")]
    _trend(
        axis, baseline[x_column].to_numpy(float), baseline[y_column].to_numpy(float),
        display_x=x,
    )
    for _, row in frame.iterrows():
        model = str(row["Model"])
        style = MODEL_STYLE[model]
        x_value, y_value = float(row[x_column]), float(row[y_column])
        xerr = float(row[x_sd_column]) if x_sd_column and pd.notna(row[x_sd_column]) else None
        yerr = float(row[y_sd_column]) if y_sd_column and pd.notna(row[y_sd_column]) else None
        # A dark perimeter makes the filled (observed) versus white (estimated)
        # provenance visible even for model families whose symbols are circles.
        edge = style["color"] if estimated_coordinate else "#111827"
        face = "white" if estimated_coordinate else style["color"]
        if xerr is not None or yerr is not None:
            axis.errorbar(
                x_value, y_value, xerr=xerr, yerr=yerr, fmt="none", ecolor=style["color"],
                elinewidth=1.4, capsize=4.5, capthick=1.4, alpha=0.70, zorder=2,
            )
        axis.plot(
            x_value, y_value, marker=style["marker"], linestyle="none", markersize=style["size"],
            markerfacecolor=face, markeredgecolor=edge, markeredgewidth=1.45 if model != "Proposed" else 1.2,
            color=style["color"], zorder=style["zorder"],
        )
        _annotate(axis, model, x_value, y_value, panel=panel_key)
    axis.set_title(_format_title(title), loc="left", fontsize=24, fontweight="bold", color="#1F2937", pad=12)
    low, high = float(np.min(y)), float(np.max(y))
    margin = max(3.3, 0.10 * (high - low))
    axis.set_ylim(max(0.0, low - margin), high + margin)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    _source_key(axis, source_text, estimated=estimated_coordinate)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.18, top=0.84)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def plot_panel(frame: pd.DataFrame, output: Path, dpi: int) -> None:
    specs = [
        ("Training time, target-stream (s)", "Target-stream performance: mean sMAPE (%)", "Training time, target-stream SD (s)", "Target-stream performance SD (%)", "(a) Target-stream performance vs. training time", "target_train", False),
        ("Inference time, target-stream (s per 1,000 samples)", "Target-stream performance: mean sMAPE (%)", None, "Target-stream performance SD (%)", "(b) Target-stream performance vs. inference time", "target_infer", True),
        ("Training time, all-stream (s)", "All-stream performance: mean sMAPE (%)", None, None, "(c) All-stream performance vs. training time", "all_train", True),
        ("Inference time, all-stream (s per 1,000 samples)", "All-stream performance: mean sMAPE (%)", None, None, "(d) All-stream performance vs. inference time", "all_infer", True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=PAPER_PANEL_FIGSIZE, facecolor="white")
    for axis, (x_col, y_col, x_sd, y_sd, title, panel, estimated) in zip(axes.flat, specs):
        _configure_axis(axis, xlabel=("Training time (s)" if "Training" in x_col else "Inference time (s)"))
        x = frame[x_col].to_numpy(float)
        y = frame[y_col].to_numpy(float)
        baseline = frame.loc[frame["Model"].ne("Proposed")]
        _trend(
            axis, baseline[x_col].to_numpy(float), baseline[y_col].to_numpy(float),
            display_x=x,
        )
        for _, row in frame.iterrows():
            model = str(row["Model"])
            style = MODEL_STYLE[model]
            x_value, y_value = float(row[x_col]), float(row[y_col])
            xerr = float(row[x_sd]) if x_sd and pd.notna(row[x_sd]) else None
            yerr = float(row[y_sd]) if y_sd and pd.notna(row[y_sd]) else None
            if xerr is not None or yerr is not None:
                axis.errorbar(x_value, y_value, xerr=xerr, yerr=yerr, fmt="none", ecolor=style["color"], elinewidth=1.25, capsize=3.5, alpha=0.70, zorder=2)
            axis.plot(
                x_value, y_value, marker=style["marker"], linestyle="none", markersize=style["size"] * 1.18,
                markerfacecolor="white" if estimated else style["color"],
                markeredgecolor=style["color"] if estimated else "#111827", markeredgewidth=1.35,
                color=style["color"], zorder=style["zorder"],
            )
            _annotate(axis, model, x_value, y_value, panel=panel)
        low, high = float(np.min(y)), float(np.max(y))
        axis.set_ylim(max(0.0, low - max(3.3, .10 * (high - low))), high + max(3.3, .10 * (high - low)))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.set_title(_format_title(title), loc="left", fontsize=17.5, fontweight="bold", color="#1F2937", pad=8)
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.10, top=0.95, hspace=0.32, wspace=0.23)
    stem = output / "efficiency_2x2_panel"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def write_excel(summary: pd.DataFrame, output: Path) -> None:
    workbook_path = output / "efficiency_summary_table.xlsx"
    display = summary[[
        "Model",
        "Target-stream performance: mean sMAPE (%)",
        "All-stream performance: mean sMAPE (%)",
        "Training time, target-stream (s)",
        "Inference time, target-stream (s per 1,000 samples)",
        "Training time, all-stream (s)",
        "Inference time, all-stream (s per 1,000 samples)",
        "Data source",
    ]].copy()
    display.columns = [
        "Model", "Target Perf.\nmean sMAPE (%)", "All-stream Perf.\nmean sMAPE (%)",
        "Train time\n(target, s)", "Inference time\n(target, s / 1,000 samples)",
        "Train time\n(all-stream, s)", "Inference time\n(all-stream, s / 1,000 samples)",
        "Data source",
    ]
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        display.to_excel(writer, sheet_name="Efficiency summary", index=False, startrow=1)
        summary.to_excel(writer, sheet_name="Final data and provenance", index=False)
        notes = pd.DataFrame({"Notes": [
            "Target-stream sMAPE and training time are observed Final3 5-fold values.",
            "No archived matched inference-latency or all-stream benchmark exists in the current workspace.",
            "All estimated values are explicitly labeled and are for figure planning only; replace them with matched benchmark measurements before manuscript submission.",
            "All inferred latency values are cumulative batch-1 inference time for 1,000 graph samples.",
        ]})
        notes.to_excel(writer, sheet_name="Read me", index=False)
    book = load_workbook(workbook_path)
    sheet = book["Efficiency summary"]
    sheet["A1"] = "Efficiency analysis summary — observed target-stream values and explicitly estimated all-stream/latency values"
    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=8)
    title = sheet["A1"]
    title.font = Font(name="Arial", size=12, bold=True, color="FFFFFF")
    title.fill = PatternFill("solid", fgColor="1F4E78")
    title.alignment = Alignment(horizontal="center")
    for cell in sheet[2]:
        cell.font = Font(name="Arial", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="2F75B5")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in sheet.iter_rows(min_row=3):
        for cell in row:
            cell.alignment = Alignment(vertical="center", wrap_text=True)
    sheet.freeze_panes = "A3"
    widths = [29, 18, 20, 16, 24, 18, 26, 22]
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.row_dimensions[1].height = 28
    sheet.row_dimensions[2].height = 36
    for name in ("Final data and provenance", "Read me"):
        current = book[name]
        current.freeze_panes = "A2"
        for column in range(1, current.max_column + 1):
            current.column_dimensions[get_column_letter(column)].width = min(40, max(14, len(str(current.cell(1, column).value)) + 2))
    book.save(workbook_path)


def write_log(frame: pd.DataFrame, observed_source: Path, output: Path) -> None:
    estimates = pd.DataFrame({
        "Model": MODEL_ORDER,
        "All-stream error multiplier": [ALL_STREAM_ERROR_MULTIPLIER[m] for m in MODEL_ORDER],
        "All-stream training multiplier": [ALL_STREAM_TRAIN_MULTIPLIER[m] for m in MODEL_ORDER],
        "Target inference estimate (s / 1,000 samples)": [TARGET_INFERENCE_SEC_PER_1000[m] for m in MODEL_ORDER],
        "All-stream inference multiplier": [ALL_STREAM_INFERENCE_MULTIPLIER[m] for m in MODEL_ORDER],
    })
    multiplier_lines = "\n".join(
        f"| {row.Model} | {row._1:.2f}× | {row._2:.2f}× | {row._3:.2f} | {row._4:.2f}× |"
        for row in estimates.itertuples(index=False)
    )
    text = f"""# Efficiency analysis: observed and estimated data log

## Observed values

The following target-stream values are directly observed and were copied from
`{observed_source.as_posix()}`:

- Mean sMAPE (%) and sMAPE SD (%) for all eight models.
- Training time mean and SD (s) for all eight models.

The source is the Final3 plotting table, which combines the reported
multi-process average sMAPE and five-fold completed-run training metadata.
Those archived run times are observations, but they were not newly measured in
a controlled matched-hardware timing benchmark.

## Unavailable values and estimation status

No archived, matched-hardware inference-latency benchmark was found for these
eight models.  No all-stream (target plus intermediate stream) baseline run was
found either.  Therefore all target-stream inference values and every
all-stream value in this package are **estimated**, never observed.  The
inference basis is cumulative batch-1 latency for 1,000 graph samples, chosen
only to express sub-second model calls in readable seconds.

## Deterministic estimation rules

| Model | All-stream error / target error | All-stream train / target train | Target inference estimate (s / 1,000) | All-stream inference / target inference |
|---|---:|---:|---:|---:|
{multiplier_lines}

All-stream error = observed target mean sMAPE × error multiplier.  All-stream
training time = observed target training time × training multiplier.  All-stream
inference time = estimated target inference time × inference multiplier.

The degradation ranges follow the requested architecture-aware prior: Proposed
has the smallest error degradation (8%) because its message-passing/readout is
stream-aware; Graphormer/SAT are intermediate (14–17%); vanilla GNNs are higher
(22–26%); and Graph-to-SFILES/Reusable-ANN are highest (28–32%).  Training and
inference multipliers reflect added output supervision/decoding without treating
these estimates as measured speedups or uncertainty intervals.

## Figure interpretation draft

Across the target-stream setting, the proposed model occupies the low-error end
of the time–performance plane, while its training time is materially higher
than that of the baselines.  The descriptive log-time fits illustrate diminishing
error improvement with additional compute and do not constitute a causal scaling
law.  Under the clearly labeled all-stream extension scenario, the proposed
model has the smallest assumed error degradation, consistent with a richer
stream-aware prediction scope.  These all-stream and latency panels are figure
planning artifacts; matched all-stream and inference benchmarks must replace
their estimates before they are presented as empirical results.
"""
    (output / "efficiency_estimation_log.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source = Path(args.observed_source)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    observed = load_observed(source)
    frame = build_final_data(observed, source)
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "efficiency_final_data.csv", index=False, encoding="utf-8-sig")

    plot_one(
        frame, x_column="Training time, target-stream (s)", y_column="Target-stream performance: mean sMAPE (%)",
        x_sd_column="Training time, target-stream SD (s)", y_sd_column="Target-stream performance SD (%)",
        title="(a) Target-stream performance vs. training time", panel_key="target_train",
        source_text="Observed target-stream: 5-fold mean +/- SD", estimated_coordinate=False,
        stem=output / "efficiency_target_train", dpi=args.dpi,
    )
    plot_one(
        frame, x_column="Inference time, target-stream (s per 1,000 samples)", y_column="Target-stream performance: mean sMAPE (%)",
        x_sd_column=None, y_sd_column="Target-stream performance SD (%)",
        title="(b) Target-stream performance vs. inference time", panel_key="target_infer",
        source_text="Observed performance; estimated latency", estimated_coordinate=True,
        stem=output / "efficiency_target_infer", dpi=args.dpi,
    )
    plot_one(
        frame, x_column="Training time, all-stream (s)", y_column="All-stream performance: mean sMAPE (%)",
        x_sd_column=None, y_sd_column=None,
        title="(c) All-stream performance vs. training time", panel_key="all_train",
        source_text="Estimated all-stream extension", estimated_coordinate=True,
        stem=output / "efficiency_allstream_train", dpi=args.dpi,
    )
    plot_one(
        frame, x_column="Inference time, all-stream (s per 1,000 samples)", y_column="All-stream performance: mean sMAPE (%)",
        x_sd_column=None, y_sd_column=None,
        title="(d) All-stream performance vs. inference time", panel_key="all_infer",
        source_text="Estimated all-stream extension", estimated_coordinate=True,
        stem=output / "efficiency_allstream_infer", dpi=args.dpi,
    )
    plot_panel(frame, output, args.dpi)
    write_excel(frame, output)
    write_log(frame, source, output)
    summary = frame[[
        "Model", "Target-stream performance: mean sMAPE (%)", "All-stream performance: mean sMAPE (%)",
        "Training time, target-stream (s)", "Inference time, target-stream (s per 1,000 samples)",
        "Training time, all-stream (s)", "Inference time, all-stream (s per 1,000 samples)", "Data source",
    ]]
    summary.to_csv(output / "efficiency_summary_table.csv", index=False, encoding="utf-8-sig")
    print(f"models={len(frame)}; output={output.resolve()}; formats=png,pdf; dpi={args.dpi}")


if __name__ == "__main__":
    main()
