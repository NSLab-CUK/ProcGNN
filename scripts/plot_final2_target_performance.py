#!/usr/bin/env python3
"""Create target-edge-only tables and figures from Final2_checked.xlsx.

The source workbook is treated as the final authority: values are read from
its displayed mean +/- SD cells and are never recomputed from run artefacts.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator
import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


SOURCE_SHEETS = [
    "Single_Process", "Multi_Process", "Zero_Shot", "Transfer_Data_Eff",
    "Depth_Sensitivity", "PIN_Sensitivity", "Ablation",
]
METRICS = ["MAE", "NMAE"]  # Preserve exact final-workbook labels.
PIN_WEIGHTS = {
    ("Mass", "Low"): 0.5, ("Mass", "Default"): 1.0, ("Mass", "High"): 2.0,
    ("Component", "Low"): 7.5e-8, ("Component", "Default"): 1.5e-7, ("Component", "High"): 3.0e-7,
    ("Atom", "Low"): 0.1, ("Atom", "Default"): 0.2, ("Atom", "High"): 0.4,
}
NUMBER = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
TRANSFER_LABEL = re.compile(r"^(.*?)\s*\((\d+)%\)$")
DEPTH_LABEL = re.compile(r"Depth\s+(\d+)")
PIN_LABEL = re.compile(r"(?:Proposed\s*[\u2014-]\s*)?(Mass|Component|Atom)\s+(Low|Default|High)")


def _pair(value: object) -> tuple[float, float]:
    """Extract mean and SD from a final-workbook cell such as '1.2 +/- 0.3'."""
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value), math.nan
    if not isinstance(value, str):
        return math.nan, math.nan
    pieces = re.split(r"\s*(?:\u00b1|\+/-)\s*", value.strip(), maxsplit=1)
    match = NUMBER.match(pieces[0])
    mean = float(match.group(1)) if match else math.nan
    sd = math.nan
    if len(pieces) == 2:
        match = NUMBER.match(pieces[1])
        sd = float(match.group(1)) if match else math.nan
    return mean, sd


def _configuration(sheet: str, label: str) -> str:
    clean = label.replace("(Target edge)", "").strip()
    clean = re.sub(r"^Proposed\s*[\u2014-]\s*", "", clean)
    return clean or sheet


def _extract_standard_sheets(source: Path) -> tuple[pd.DataFrame, list[str]]:
    book = load_workbook(source, data_only=True, read_only=False)
    rows: list[dict[str, object]] = []
    properties: list[str] = []
    for sheet_name in SOURCE_SHEETS:
        if sheet_name not in book.sheetnames:
            continue
        sheet = book[sheet_name]
        # Locate paired MAE/NMAE columns from the two-level header instead of
        # assuming properties begin at column 2.  Final3 adds three PINN pass
        # rate columns before Temp in the single- and multi-process sheets.
        property_columns: list[tuple[str, int]] = []
        for column in range(2, sheet.max_column):
            property_name = sheet.cell(1, column).value
            metric = sheet.cell(2, column).value
            next_metric = sheet.cell(2, column + 1).value
            if not isinstance(property_name, str) or property_name.strip().lower() == "average":
                continue
            if metric == "MAE" and next_metric == "NMAE":
                property_name = property_name.strip()
                property_columns.append((property_name, column))
                if property_name not in properties:
                    properties.append(property_name)
        if not property_columns:
            continue
        for row in range(3, sheet.max_row + 1):
            model_label = sheet.cell(row, 1).value
            if not isinstance(model_label, str) or not model_label.strip():
                continue
            # Explicit user requirement: never publish all-edge rows here.
            if "all edge" in model_label.lower():
                continue
            for property_name, column in property_columns:
                mae_mean, mae_sd = _pair(sheet.cell(row, column).value)
                nmae_mean, nmae_sd = _pair(sheet.cell(row, column + 1).value)
                for metric, mean, sd in (("MAE", mae_mean, mae_sd), ("NMAE", nmae_mean, nmae_sd)):
                    if not math.isfinite(mean):
                        continue
                    rows.append({
                        "source_sheet": sheet_name,
                        "configuration": _configuration(sheet_name, model_label),
                        "model": re.sub(r"\s*\([^)]*\)$", "", model_label).strip(),
                        "metric": metric,
                        "property": property_name,
                        "mean": round(mean, 4),
                        "std": round(sd, 4) if math.isfinite(sd) else math.nan,
                    })
    if not rows:
        raise RuntimeError("no target-edge standard metric rows found in source workbook")
    return pd.DataFrame(rows), properties


def _transfer_frame(detail: pd.DataFrame) -> pd.DataFrame:
    transfer = detail[detail["source_sheet"].eq("Transfer_Data_Eff")].copy()
    parsed = transfer["configuration"].str.extract(TRANSFER_LABEL)
    transfer["model"] = parsed[0]
    transfer["transfer_percent"] = pd.to_numeric(parsed[1], errors="coerce")
    transfer = transfer.dropna(subset=["transfer_percent"]).copy()
    transfer["transfer_percent"] = transfer["transfer_percent"].astype(int)

    zero = detail[detail["source_sheet"].eq("Zero_Shot")].copy()
    zero["model"] = zero["configuration"].str.replace(r"\s*\(Target edge\)\s*$", "", regex=True)
    zero["transfer_percent"] = 0
    zero = zero[zero["model"].isin(transfer["model"].unique())]
    return pd.concat([zero, transfer], ignore_index=True).sort_values(
        ["metric", "property", "model", "transfer_percent"]
    )


def _sensitivity_frame(detail: pd.DataFrame) -> pd.DataFrame:
    depth = detail[detail["source_sheet"].eq("Depth_Sensitivity")].copy()
    depth["sensitivity"] = "Depth"
    depth["setting"] = depth["configuration"].str.extract(DEPTH_LABEL)[0]
    depth = depth.dropna(subset=["setting"])
    depth["setting_order"] = pd.to_numeric(depth["setting"], errors="coerce")

    pin = detail[detail["source_sheet"].eq("PIN_Sensitivity")].copy()
    parsed = pin["configuration"].str.extract(PIN_LABEL)
    pin["sensitivity"] = parsed[0]
    pin["setting"] = parsed[1]
    pin = pin.dropna(subset=["sensitivity", "setting"])
    pin["setting_order"] = pin["setting"].map({"Low": 1, "Default": 2, "High": 3})
    pin["weight"] = [PIN_WEIGHTS[(name, setting)] for name, setting in zip(pin["sensitivity"], pin["setting"])]
    depth["weight"] = math.nan
    return pd.concat([depth, pin], ignore_index=True)


def _wide_metric_table(frame: pd.DataFrame, metric: str, row_columns: list[str], properties: list[str]) -> pd.DataFrame:
    sub = frame[frame["metric"].eq(metric)].copy()
    wide = sub.pivot_table(index=row_columns, columns="property", values="mean", aggfunc="first")
    wide = wide.reindex(columns=properties)
    wide["Average"] = wide.mean(axis=1)
    wide = wide.reset_index()
    # Weight is an experimental setting rather than a performance result: do
    # not round e.g. 7.5e-8 to 0.0000. All performance columns remain 4 dp.
    for column in [*properties, "Average"]:
        wide[column] = wide[column].round(4)
    return wide


def _write_frame(sheet, frame: pd.DataFrame, title: str) -> None:
    sheet.append([title])
    sheet.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(frame.columns))
    title_cell = sheet.cell(1, 1)
    title_cell.font = Font(bold=True, color="FFFFFF", size=12)
    title_cell.fill = PatternFill("solid", fgColor="1F4E78")
    title_cell.alignment = Alignment(horizontal="left")
    sheet.append(list(frame.columns))
    for cell in sheet[2]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="5B9BD5")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for values in frame.itertuples(index=False, name=None):
        sheet.append(list(values))
    for row in sheet.iter_rows(min_row=3, max_row=sheet.max_row):
        for cell in row:
            cell.alignment = Alignment(vertical="center", wrap_text=False)
            if isinstance(cell.value, (float, np.floating)):
                header = sheet.cell(2, cell.column).value
                cell.number_format = "0.00E+00" if header == "weight" else "0.0000"
    sheet.freeze_panes = "A3"
    sheet.auto_filter.ref = f"A2:{get_column_letter(sheet.max_column)}{sheet.max_row}"
    for col in range(1, sheet.max_column + 1):
        values = [str(sheet.cell(row, col).value or "") for row in range(1, min(sheet.max_row, 50) + 1)]
        sheet.column_dimensions[get_column_letter(col)].width = min(max(max(map(len, values)) + 2, 12), 30)
    sheet.row_dimensions[1].height = 22
    sheet.row_dimensions[2].height = 32


def _save_figure(figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def _sensitivity_plots(sensitivity: pd.DataFrame, figure_dir: Path) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    for metric in METRICS:
        depth = sensitivity[(sensitivity["sensitivity"].eq("Depth")) & sensitivity["metric"].eq(metric)]
        depth_average = depth.groupby(["setting", "setting_order"], as_index=False)["mean"].mean().sort_values("setting_order")
        fig, axis = plt.subplots(figsize=(6.6, 4.2))
        axis.plot(depth_average["setting_order"], depth_average["mean"].round(4), marker="o", linewidth=2.2, color="#1F77B4")
        axis.set(xlabel="Number of message-passing layers", ylabel=f"Mean {metric} across 10 properties", title=f"Depth sensitivity ({metric}, target edges)")
        axis.set_xticks(depth_average["setting_order"])
        axis.grid(alpha=0.28)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.4f}"))
        stem = figure_dir / f"depth_sensitivity_{metric.lower()}_target"
        _save_figure(fig, stem)
        manifest.append({"figure": stem.name, "scope": "Depth sensitivity", "metric": metric})

        fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9), sharey=True)
        pin = sensitivity[(sensitivity["sensitivity"].isin(["Mass", "Component", "Atom"])) & sensitivity["metric"].eq(metric)]
        for axis, name in zip(axes, ["Mass", "Component", "Atom"]):
            group = pin[pin["sensitivity"].eq(name)].sort_values("weight")
            average = group.groupby(["setting", "weight"], as_index=False)["mean"].mean().sort_values("weight")
            axis.plot(average["weight"], average["mean"].round(4), marker="o", linewidth=2.1)
            axis.set_xscale("log")
            axis.set_xticks(average["weight"])
            axis.set_xticklabels([f"{weight:.1e}" if weight < 1.0e-4 else f"{weight:g}" for weight in average["weight"]])
            # The three actual settings must be the only visible x labels;
            # automatic log minor labels otherwise overlap them.
            axis.xaxis.set_minor_locator(NullLocator())
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set_title(f"{name} conservation")
            axis.set_xlabel(f"$\\lambda_{{{name.lower()}}}$")
            axis.grid(alpha=0.28, which="both")
            axis.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.4f}"))
        axes[0].set_ylabel(f"Mean {metric} across 10 properties")
        fig.suptitle(f"PIN-weight sensitivity ({metric}, target edges)", y=1.03)
        fig.tight_layout()
        # Use a new stem because earlier generic Low/Default/High figures may
        # be open remotely; this filename makes the corrected x-axis explicit.
        stem = figure_dir / f"pinn_weight_sensitivity_{metric.lower()}_target_actual_weights"
        _save_figure(fig, stem)
        manifest.append({"figure": stem.name, "scope": "PIN-weight sensitivity", "metric": metric})
    return manifest


def _property_slug(name: str) -> str:
    return {
        "H₂O": "h2o", "H₂": "h2", "CH₄": "ch4", "CO₂": "co2", "O₂": "o2", "N₂": "n2",
    }.get(name, re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_"))


def _transfer_plots(transfer: pd.DataFrame, figure_dir: Path, properties: list[str]) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    model_order = ["Proposed", "GCN", "GIN", "GAT", "Graphormer", "SAT", "GraphToSFILES"]
    for metric in METRICS:
        for property_name in properties:
            sub = transfer[(transfer["metric"].eq(metric)) & transfer["property"].eq(property_name)].copy()
            fig, axis = plt.subplots(figsize=(7.2, 4.5))
            for model in model_order:
                group = sub[sub["model"].eq(model)].sort_values("transfer_percent")
                if group.empty:
                    continue
                axis.plot(group["transfer_percent"], group["mean"].round(4), marker="o", markersize=3.5, linewidth=2.0, label=model)
            axis.set(
                xlabel="Target-process training data (%)",
                ylabel=f"{metric} (log scale; lower is better)",
                title=f"Transfer performance: {property_name} ({metric}, target edges)",
            )
            axis.set_xticks([0, 2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90])
            axis.set_xlim(-1, 91)
            positive = sub.loc[sub["mean"] > 0, "mean"]
            if not positive.empty:
                axis.set_yscale("log")
            axis.grid(alpha=0.25, which="both")
            axis.legend(frameon=False, fontsize=7, ncol=2, loc="best")
            stem = figure_dir / f"transfer_{_property_slug(property_name)}_{metric.lower()}_target"
            _save_figure(fig, stem)
            manifest.append({"figure": stem.name, "scope": f"Transfer {property_name}", "metric": metric})
    return manifest


def _write_readme(path: Path, source: Path, figure_count: int) -> None:
    path.write_text(
        "# Final target-only performance tables and figures\n\n"
        f"- Source of truth: `{source.as_posix()}`\n"
        "- Scope: target-edge rows only. Every row labeled `All edge` was excluded.\n"
        "- Metrics: `MAE` and `NMAE`, preserving the exact labels in the final workbook.\n"
        "- Values: stored source means and standard deviations; values supplied to tables and plots were rounded to four decimal places.\n"
        "- Sensitivity figures show the arithmetic mean across the 10 output properties. PIN panels use each term's actual loss-weight scale: mass 0.5/1.0/2.0, component 7.5e-8/1.5e-7/3.0e-7, and atom 0.1/0.2/0.4.\n"
        "- Transfer figures show each property separately from 0% (zero-shot) to 90%. A log y-axis is used because source values span multiple orders of magnitude.\n"
        f"- Created figures: {figure_count} figures, each saved as PNG and SVG.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="outputs/0819final/Final2_checked.xlsx")
    parser.add_argument("--output-dir", default="outputs/0819final/Final2_check_target_plots")
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    figure_dir, table_dir = output / "figures", output / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    detail, properties = _extract_standard_sheets(source)
    transfer = _transfer_frame(detail)
    sensitivity = _sensitivity_frame(detail)
    if transfer.empty or sensitivity.empty:
        raise RuntimeError("required target-only transfer or sensitivity data are missing")

    sensitivity_average = (
        sensitivity.groupby(["sensitivity", "setting", "setting_order", "metric"], as_index=False)["mean"].mean()
        .sort_values(["sensitivity", "setting_order", "metric"])
        .round(4)
    )
    transfer_average = (
        transfer.groupby(["model", "transfer_percent", "metric"], as_index=False)["mean"].mean()
        .sort_values(["model", "transfer_percent", "metric"])
        .round(4)
    )
    detail.round({"mean": 4, "std": 4}).to_csv(table_dir / "target_property_performance_long.csv", index=False)
    sensitivity_average.to_csv(table_dir / "sensitivity_target_mean_performance.csv", index=False)
    transfer.round({"mean": 4, "std": 4}).to_csv(table_dir / "transfer_target_property_performance.csv", index=False)
    transfer_average.to_csv(table_dir / "transfer_target_average_performance.csv", index=False)

    book = Workbook()
    book.remove(book.active)
    readme = book.create_sheet("README")
    readme.append(["Final2 target-edge-only performance tables"])
    readme["A1"].font = Font(bold=True, size=14)
    for text in [
        f"Source workbook: {source.as_posix()}",
        "All rows labeled All edge are excluded.",
        "Metric labels are preserved exactly from the final workbook: MAE and NMAE.",
        "All numeric values are rounded to four decimal places.",
        "Transfer tables include the 0% zero-shot point and 2% through 90% transfer points.",
    ]:
        readme.append([text])
    readme.column_dimensions["A"].width = 105
    _write_frame(book.create_sheet("Sensitivity_MAE"), _wide_metric_table(sensitivity, "MAE", ["sensitivity", "setting", "weight", "setting_order"], properties), "Sensitivity: MAE (target edges)")
    _write_frame(book.create_sheet("Sensitivity_NMAE"), _wide_metric_table(sensitivity, "NMAE", ["sensitivity", "setting", "weight", "setting_order"], properties), "Sensitivity: NMAE (target edges)")
    _write_frame(book.create_sheet("Transfer_MAE"), _wide_metric_table(transfer, "MAE", ["model", "transfer_percent"], properties), "Transfer: MAE by property (target edges)")
    _write_frame(book.create_sheet("Transfer_NMAE"), _wide_metric_table(transfer, "NMAE", ["model", "transfer_percent"], properties), "Transfer: NMAE by property (target edges)")
    _write_frame(book.create_sheet("All_Target_Long"), detail.round({"mean": 4, "std": 4}), "All extracted target-edge performance values")
    workbook_path = output / f"{source.stem}_target_only_tables.xlsx"
    book.save(workbook_path)

    manifest = _sensitivity_plots(sensitivity, figure_dir) + _transfer_plots(transfer, figure_dir, properties)
    pd.DataFrame(manifest).to_csv(output / "figure_manifest.csv", index=False)
    _write_readme(output / "README.md", source, len(manifest))
    print(f"source={source}")
    print(f"target_property_rows={len(detail)}")
    print(f"transfer_rows={len(transfer)}; transfer_points={sorted(transfer.transfer_percent.unique())}")
    print(f"sensitivity_rows={len(sensitivity)}")
    print(f"figures={len(manifest)}; output={output}")


if __name__ == "__main__":
    main()
