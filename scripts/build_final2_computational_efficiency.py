#!/usr/bin/env python3
"""Build paper-ready computational-efficiency tables and figures from Final2.

Only completed-run metadata already shown in ``Computational_Eff`` are used.
The script deliberately does not estimate missing parameter counts or fabricate
surrogate-versus-simulator latency values.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


PAIR = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*(?:±|\+/-)\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


def parse_pair(value: object) -> tuple[float, float]:
    if not isinstance(value, str):
        return math.nan, math.nan
    match = PAIR.match(value.replace(",", ""))
    if not match:
        return math.nan, math.nan
    return float(match.group(1)), float(match.group(2))


def parse_count(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip().replace(",", "").isdigit():
        return float(value.replace(",", ""))
    return math.nan


def load_source(source: Path) -> pd.DataFrame:
    # Final3 is a valid workbook but lacks a reliable worksheet dimension in
    # read-only mode, so retain normal mode for deterministic row discovery.
    book = load_workbook(source, data_only=True, read_only=False)
    if "Computational_Eff" not in book.sheetnames:
        raise KeyError(f"Computational_Eff sheet is missing from {source}")
    sheet = book["Computational_Eff"]
    rows: list[dict[str, object]] = []
    for row in range(3, sheet.max_row + 1):
        model = sheet.cell(row, 1).value
        if not isinstance(model, str) or not model.strip():
            continue
        r2, r2_sd = parse_pair(sheet.cell(row, 2).value)
        train_s, train_s_sd = parse_pair(sheet.cell(row, 3).value)
        memory_mb, memory_mb_sd = parse_pair(sheet.cell(row, 4).value)
        if not math.isfinite(train_s):
            continue
        rows.append(
            {
                "Model": model,
                "Validation target-edge property mean R2": r2,
                "Validation target-edge property mean R2 SD": r2_sd,
                "Training time (s)": train_s,
                "Training time SD (s)": train_s_sd,
                "Peak GPU memory (MB)": memory_mb,
                "Peak GPU memory SD (MB)": memory_mb_sd,
                "Trainable parameters": parse_count(sheet.cell(row, 5).value),
                "Trainable parameters display": sheet.cell(row, 5).value,
                "Folds": sheet.cell(row, 6).value,
                "Theoretical complexity": sheet.cell(row, 7).value,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no completed-run computational-efficiency records")
    return frame


def add_target_mae_nmae(source: Path, resources: pd.DataFrame) -> pd.DataFrame:
    """Attach authoritative target-edge average MAE/NMAE from Multi_Process."""
    book = load_workbook(source, data_only=True, read_only=False)
    if "Multi_Process" not in book.sheetnames:
        raise KeyError(f"Multi_Process sheet is missing from {source}")
    sheet = book["Multi_Process"]
    mae_column = nmae_column = None
    for column in range(2, sheet.max_column + 1):
        heading = sheet.cell(2, column).value
        if heading == "MAE ± SD (reported)":
            mae_column = column
        elif heading == "NMAE ± uncertainty":
            nmae_column = column
    if mae_column is None or nmae_column is None:
        raise RuntimeError("Final3 corrected average MAE/NMAE columns were not found")

    performance: list[dict[str, object]] = []
    for row in range(3, sheet.max_row + 1):
        model = sheet.cell(row, 1).value
        if not isinstance(model, str) or "all edge" in model.lower():
            continue
        mae, mae_sd = parse_pair(sheet.cell(row, mae_column).value)
        nmae, nmae_sd = parse_pair(sheet.cell(row, nmae_column).value)
        if math.isfinite(mae) and math.isfinite(nmae):
            performance.append(
                {
                    "Model": model,
                    "Target-edge mean MAE": mae,
                    "Target-edge mean MAE SD": mae_sd,
                    "Target-edge mean NMAE": nmae,
                    "Target-edge mean NMAE SD": nmae_sd,
                }
            )
    merged = resources.merge(pd.DataFrame(performance), on="Model", how="left", validate="one_to_one")
    missing = merged.loc[merged["Target-edge mean MAE"].isna(), "Model"].tolist()
    if missing:
        raise RuntimeError(f"missing Multi_Process MAE/NMAE for resource rows: {missing}")
    return merged


def fmt_pair(mean: float, sd: float, digits: int = 4) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def write_excel(frame: pd.DataFrame, destination: Path) -> None:
    book = Workbook()
    sheet = book.active
    sheet.title = "Computational_Efficiency"
    headers = [
        "Model",
        "Target-edge mean\nMAE ± SD",
        "Target-edge mean\nNMAE ± uncertainty",
        "Training time (s)\nmean ± SD",
        "Peak GPU memory (MB)\nmean ± SD",
        "Trainable parameters",
        "Folds",
        "Theoretical complexity",
    ]
    fill = PatternFill("solid", fgColor="1F4E78")
    for column, header in enumerate(headers, 1):
        cell = sheet.cell(1, column, header)
        cell.fill = fill
        cell.font = Font(name="Arial", size=10, color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    sheet.row_dimensions[1].height = 34

    for row_index, row in enumerate(frame.itertuples(index=False), 2):
        values = [
            row[0],
            fmt_pair(row[11], row[12]),
            fmt_pair(row[13], row[14]),
            fmt_pair(row[3], row[4], 1),
            fmt_pair(row[5], row[6], 1),
            row[8] if isinstance(row[8], str) else f"{int(row[7]):,}",
            row[9],
            row[10],
        ]
        for column, value in enumerate(values, 1):
            cell = sheet.cell(row_index, column, value)
            cell.font = Font(name="Arial", size=10)
            cell.alignment = Alignment(
                horizontal="left" if column in {1, 7} else "center",
                vertical="center",
                wrap_text=column == 8,
            )
        sheet.row_dimensions[row_index].height = 36 if len(str(values[-1])) > 48 else 22

    note_row = len(frame) + 3
    sheet.merge_cells(start_row=note_row, start_column=1, end_row=note_row, end_column=8)
    note = sheet.cell(
        note_row,
        1,
        "Note. Target-edge MAE/NMAE are corrected Multi_Process reported averages. Training time and GPU memory are completed-run metadata. "
        "The proposed-model parameter count is not stored in the authoritative source and is therefore reported as N/A.",
    )
    note.font = Font(name="Arial", size=9, italic=True, color="666666")
    note.alignment = Alignment(wrap_text=True, vertical="center")
    sheet.row_dimensions[note_row].height = 30
    widths = [30, 23, 25, 24, 27, 22, 9, 66]
    for column, width in enumerate(widths, 1):
        sheet.column_dimensions[get_column_letter(column)].width = width
    sheet.freeze_panes = "A2"
    sheet.sheet_view.showGridLines = False

    status = book.create_sheet("Inference_vs_Simulator")
    status.append(["Comparison", "Status", "Reason"])
    status.append(
        [
            "Surrogate inference vs rigorous process simulation",
            "Not available",
            "No paired latency artifact for the trained surrogate and rigorous simulator exists under outputs/0819final. "
            "It must be measured in a dedicated matched-hardware benchmark before publication.",
        ]
    )
    for cell in status[1]:
        cell.fill = fill
        cell.font = Font(name="Arial", size=10, color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for cell in status[2]:
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    status.column_dimensions["A"].width = 47
    status.column_dimensions["B"].width = 18
    status.column_dimensions["C"].width = 98
    status.row_dimensions[2].height = 46
    status.sheet_view.showGridLines = False
    book.save(destination)


def plot_time_performance(frame: pd.DataFrame, directory: Path, metric: str, metric_sd: str, stem_name: str) -> None:
    fig, axis = plt.subplots(figsize=(10.8, 6.3))
    colors = ["#2563EB" if model != "Proposed" else "#DC2626" for model in frame["Model"]]
    markers = {
        "GAT": "o", "GCN": "s", "GIN": "^", "GraphToSFILES": "D",
        "Graphormer": "P", "Reusable-Distillation-ANN": "v", "Proposed": "*", "SAT": "X",
    }
    for (_, row), color in zip(frame.iterrows(), colors):
        axis.errorbar(
            row["Training time (s)"],
            row[metric],
            xerr=row["Training time SD (s)"],
            yerr=row[metric_sd],
            fmt=markers.get(row["Model"], "o"),
            markersize=7,
            color=color,
            capsize=3,
            zorder=3,
            label=row["Model"],
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Joint-training time per fold (s, log scale)")
    axis.set_ylabel(f"Target-edge mean {metric.replace('Target-edge mean ', '')} (log scale; lower is better)")
    axis.set_title(f"Error–training-time trade-off ({metric.replace('Target-edge mean ', '')}, multi-process)")
    axis.grid(alpha=0.25, which="both")
    axis.legend(title="Model", fontsize=8, title_fontsize=9, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.subplots_adjust(right=0.74)
    fig.savefig(directory / f"{stem_name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{stem_name}.svg", bbox_inches="tight")
    plt.close(fig)


def plot_resources(frame: pd.DataFrame, directory: Path) -> None:
    display = frame.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(display))
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.7), sharey=True, constrained_layout=True)
    specs = [
        ("Training time (s)", "Training time per fold (s)", "#2563EB"),
        ("Peak GPU memory (MB)", "Peak allocated GPU memory (MB)", "#059669"),
        ("Trainable parameters", "Trainable parameters", "#7C3AED"),
    ]
    for axis, (column, label, color) in zip(axes, specs):
        valid = display[column].notna()
        axis.barh(y[valid], display.loc[valid, column], color=color, alpha=0.88)
        axis.set_xscale("log")
        axis.set_xlabel(label + " (log scale)")
        axis.grid(axis="x", alpha=0.25, which="both")
        axis.set_title(label)
        if column == "Trainable parameters":
            for index in np.where(~valid)[0]:
                axis.text(
                    0.03,
                    index,
                    "N/A",
                    transform=axis.get_yaxis_transform(),
                    va="center",
                    fontsize=9,
                    color="#666666",
                )
    axes[0].set_yticks(y, display["Model"])
    axes[0].set_ylabel("Model")
    figure.suptitle("Joint-training resource profile (multi-process setting)", fontsize=13)
    figure.savefig(directory / "joint_training_resource_profile.png", dpi=300, bbox_inches="tight")
    figure.savefig(directory / "joint_training_resource_profile.svg", bbox_inches="tight")
    plt.close(figure)


def write_readme(directory: Path, frame: pd.DataFrame, source: Path) -> None:
    text = f"""# Computational-efficiency artifacts

Source workbook: `{source.as_posix()}` (`Computational_Eff` and `Multi_Process` worksheets).

The table and figures cover the {len(frame)} shared deep-learning and graph-based models in the multi-process setting. Performance is the target-edge mean MAE or NMAE from the corrected reported Multi_Process averages (lower is better). Training time and peak GPU memory are 5-fold mean ± sample SD recorded in completed-run metadata rather than a newly controlled benchmark.

`mae_vs_training_time.*` and `nmae_vs_training_time.*` place training time on the horizontal log axis and the corresponding target-edge error metric on the vertical log axis. `joint_training_resource_profile.*` separately displays training time, peak GPU memory, and stored parameter count. The proposed-model parameter count is unavailable in the authoritative workbook, so it remains `N/A` rather than being inferred from another run.

## Inference-versus-simulator comparison

No paired surrogate-inference and rigorous-simulation latency measurements were found in the final output root. Consequently, no such quantitative comparison or plot is published here. The text of Section 5.6 may state the architectural expectation of direct-forward-pass inference, but it must not claim a numerical speed-up until a matched-hardware timing benchmark is saved.

## Paper-safe Section 5.6 wording

> We compare the proposed model with shared deep learning-based and graph-based baselines in the multi-process setting. Predictive performance is reported as target-edge mean MAE and NMAE, while training time and peak allocated GPU memory are reported as five-fold mean ± sample standard deviation from completed-run metadata. Because the proposed model includes bidirectional propagation, flow-aware attention, differential unit encoding, and graph-level readout, its training resource usage is reported separately from predictive error. Parameter counts are shown only where stored by the corresponding experiment metadata. A quantitative comparison between surrogate inference and rigorous process simulation is deferred until paired latency measurements under matched hardware are available.
"""
    (directory / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("outputs/0819final/Final2_checked.xlsx"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/0819final/Final2_check_computational_efficiency"),
    )
    args = parser.parse_args()
    frame = add_target_mae_nmae(args.source, load_source(args.source))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = args.output_dir / "tables"
    tables.mkdir(exist_ok=True)
    frame.to_csv(tables / "computational_efficiency_multi_process.csv", index=False, float_format="%.4f")
    pd.DataFrame(
        [{
            "Comparison": "Surrogate inference vs rigorous process simulation",
            "Status": "Not available",
            "Reason": "No paired latency artifact exists in the final output root; do not report a numerical speed-up.",
        }]
    ).to_csv(tables / "inference_vs_simulation_status.csv", index=False)
    write_excel(frame, args.output_dir / f"{args.source.stem}_computational_efficiency_tables.xlsx")
    plot_time_performance(frame, args.output_dir, "Target-edge mean MAE", "Target-edge mean MAE SD", "mae_vs_training_time")
    plot_time_performance(frame, args.output_dir, "Target-edge mean NMAE", "Target-edge mean NMAE SD", "nmae_vs_training_time")
    plot_resources(frame, args.output_dir)
    write_readme(args.output_dir, frame, args.source)
    print(f"models={len(frame)}; output={args.output_dir}")


if __name__ == "__main__":
    main()
