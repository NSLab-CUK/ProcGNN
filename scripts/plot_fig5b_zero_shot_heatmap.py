"""Generate only Fig. 5b from existing 0% unseen-flowsheet results.

This script reads the manuscript transfer-results workbook, uses the same
representative models selected for Fig. 5a, and creates a zero-shot sMAPE
heatmap.  No training, fine-tuning, evaluation, or synthetic values are used.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

import plot_fig5a_mean_smape_vs_finetuning as fig5a


ROOT = Path(__file__).resolve().parents[1]
SOURCE_WORKBOOK = (
    ROOT
    / "outputs"
    / "0819final"
    / "Paper_Tables_Final3_figures"
    / "final"
    / "experiment_results"
    / "experiment_results.xlsx"
)
SOURCE_SHEET = "Transfer_sMAPE"
FIG5A_SELECTION_REFERENCE = ROOT / "scripts" / "plot_fig5a_mean_smape_vs_finetuning.py"
OUT_DIR = (
    ROOT
    / "outputs"
    / "0819final"
    / "Paper_Tables_Final3_figures"
    / "final"
    / "Fig5b_zero_shot_heatmap"
)

TARGET_PROPERTIES = (
    "H2O",
    "H2",
    "CH4",
    "CO2",
    "CO",
    "O2",
    "N2",
    "Temperature",
    "Pressure",
    "Mass Flow",
)
PROPERTY_LABELS = {
    "H2O": r"H$_2$O",
    "H2": r"H$_2$",
    "CH4": r"CH$_4$",
    "CO2": r"CO$_2$",
    "CO": "CO",
    "O2": r"O$_2$",
    "N2": r"N$_2$",
    "Temperature": "Temperature",
    "Pressure": "Pressure",
    "Mass Flow": "Mass Flow",
}
MODEL_LABELS = {
    "Proposed": "Proposed",
    "GCN": "GCN",
    "Graphormer": "Graphormer",
    "GraphToSFILES": "Graph-to-SFILES",
}
COLORMAP = "viridis"


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def normalize_property(value: object) -> str:
    """Normalize the workbook's Unicode-subscript headings without guessing aliases."""
    subscript_map = str.maketrans({chr(0x2082): "2", chr(0x2084): "4"})
    normalized = str(value).strip().translate(subscript_map)
    return {"Temp": "Temperature", "Pres": "Pressure"}.get(normalized, normalized)


def load_zero_shot_source() -> tuple[pd.DataFrame, list[str]]:
    """Read only the existing 0% target-property sMAPE source rows."""
    if not SOURCE_WORKBOOK.is_file():
        raise FileNotFoundError(f"Missing source workbook: {SOURCE_WORKBOOK}")
    workbook = pd.ExcelFile(SOURCE_WORKBOOK)
    if SOURCE_SHEET not in workbook.sheet_names:
        raise KeyError(f"Missing worksheet {SOURCE_SHEET!r}; found {workbook.sheet_names}")
    source = pd.read_excel(workbook, sheet_name=SOURCE_SHEET)
    required = {"model", "transfer_percent", "property", "sMAPE_mean_pct"}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"{SOURCE_SHEET} is missing required columns: {missing}")

    source = source.copy()
    source["Model"] = source["model"].astype(str).str.strip()
    source["Fine-tuning ratio (%)"] = pd.to_numeric(source["transfer_percent"], errors="coerce")
    source["sMAPE (%)"] = pd.to_numeric(source["sMAPE_mean_pct"], errors="coerce")
    source["Source property"] = source["property"].astype(str).str.strip()
    source["Property"] = source["Source property"].map(normalize_property)
    source = source.loc[
        source["Fine-tuning ratio (%)"].eq(0)
        & source["Property"].isin(TARGET_PROPERTIES)
    ].copy()
    source["Fine-tuning ratio (%)"] = source["Fine-tuning ratio (%)"].astype(int)
    source["sMAPE (ratio)"] = source["sMAPE (%)"] / 100.0
    if source.empty:
        raise ValueError("No 0% zero-shot target-property sMAPE rows exist in the source workbook.")
    if source["sMAPE (ratio)"].isna().any() or (source["sMAPE (ratio)"] < 0).any():
        raise ValueError("Zero-shot source has missing or negative sMAPE values.")
    if source.duplicated(["Model", "Property"]).any():
        duplicates = source.loc[source.duplicated(["Model", "Property"], keep=False), ["Model", "Property"]]
        raise ValueError(f"Expected one 0% sMAPE value per model/property; duplicates found:\n{duplicates}")

    models = source["Model"].drop_duplicates().tolist()
    expected = set(TARGET_PROPERTIES)
    for model in models:
        present = set(source.loc[source["Model"].eq(model), "Property"])
        missing_properties = sorted(expected.difference(present))
        if missing_properties:
            raise ValueError(f"0% source is incomplete for {model}: missing {missing_properties}")
    return source, models


def all_model_validation(source: pd.DataFrame, selected_models: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Find displayed-subset and all-model minima for every target property."""
    records: list[dict[str, object]] = []
    missing_global_winners: list[str] = []
    for property_name in TARGET_PROPERTIES:
        all_rows = source.loc[source["Property"].eq(property_name)].copy()
        displayed_rows = all_rows.loc[all_rows["Model"].isin(selected_models)].copy()
        if displayed_rows.empty:
            raise ValueError(f"No displayed model has a zero-shot value for {property_name}")
        all_min = float(all_rows["sMAPE (ratio)"].min())
        displayed_min = float(displayed_rows["sMAPE (ratio)"].min())
        all_best_models = all_rows.loc[np.isclose(all_rows["sMAPE (ratio)"], all_min), "Model"].tolist()
        displayed_best_models = displayed_rows.loc[
            np.isclose(displayed_rows["sMAPE (ratio)"], displayed_min), "Model"
        ].tolist()
        omitted_winners = [model for model in all_best_models if model not in selected_models]
        if omitted_winners:
            missing_global_winners.extend(omitted_winners)
        proposed_value = float(
            all_rows.loc[all_rows["Model"].eq("Proposed"), "sMAPE (ratio)"].iloc[0]
        )
        proposed_is_best = bool(np.isclose(proposed_value, all_min))
        records.append(
            {
                "Property": property_name,
                "Displayed minimum sMAPE": displayed_min,
                "Displayed best model(s)": "; ".join(displayed_best_models),
                "All-model minimum sMAPE": all_min,
                "All-model best model(s)": "; ".join(all_best_models),
                "Global best omitted from Fig5a subset": "; ".join(omitted_winners),
                "Proposed sMAPE": proposed_value,
                "Proposed is all-model best": proposed_is_best,
            }
        )
    return pd.DataFrame(records), list(dict.fromkeys(missing_global_winners))


def determine_display_models(source: pd.DataFrame, validation: pd.DataFrame, source_models: list[str]) -> list[str]:
    """Use Fig. 5a's subset and add an omitted global winner only if necessary."""
    selected = list(fig5a.SELECTED_MODELS)
    if "Proposed" not in selected:
        raise ValueError("Fig. 5a model selection unexpectedly omits Proposed.")
    unavailable = [model for model in selected if model not in source_models]
    if unavailable:
        raise ValueError(f"Fig. 5a-selected model(s) lack 0% zero-shot results: {unavailable}")

    global_winners = []
    for cell in validation["All-model best model(s)"]:
        global_winners.extend([model for model in str(cell).split("; ") if model])
    additions = [model for model in dict.fromkeys(global_winners) if model not in selected]
    if additions:
        print(
            "WARNING: an all-model property winner is absent from the Fig. 5a subset; "
            f"adding it to Fig. 5b to avoid a misleading best-value display: {additions}"
        )
        selected.extend(additions)
    return selected


def scale_limits(source: pd.DataFrame) -> tuple[float, float]:
    """Derive reusable linear limits from all existing 0% values, not the subset."""
    maximum = float(source["sMAPE (ratio)"].max())
    # A zero lower bound and rounded-up 0.1 upper bound keep lower error
    # visually better and let Fig. 5c reuse an unclipped, explicit scale.
    vmin = 0.0
    vmax = float(math.ceil(maximum / 0.1) * 0.1)
    if not vmax > vmin:
        raise ValueError(f"Invalid heatmap scale limits: vmin={vmin}, vmax={vmax}")
    return vmin, vmax


def build_csv(source: pd.DataFrame, validation: pd.DataFrame, displayed_models: list[str]) -> pd.DataFrame:
    """Save all inspected values plus the column-minimum checks used for annotation."""
    lookup = validation.set_index("Property")
    values = source.copy()
    values["Record type"] = "source_zero_shot_value"
    values["Displayed in Fig5b"] = values["Model"].isin(displayed_models)
    values["Displayed model minimum"] = values.apply(
        lambda row: bool(np.isclose(row["sMAPE (ratio)"], lookup.loc[row["Property"], "Displayed minimum sMAPE"])),
        axis=1,
    )
    values["All-model minimum"] = values.apply(
        lambda row: bool(np.isclose(row["sMAPE (ratio)"], lookup.loc[row["Property"], "All-model minimum sMAPE"])),
        axis=1,
    )
    values["All-model best model(s)"] = values["Property"].map(lookup["All-model best model(s)"])
    values["Displayed best model(s)"] = values["Property"].map(lookup["Displayed best model(s)"])
    values["Source workbook"] = relative_path(SOURCE_WORKBOOK)
    values["Source worksheet"] = SOURCE_SHEET

    summary = validation.copy()
    summary.insert(0, "Record type", "column_minimum_summary")
    summary["Model"] = ""
    summary["Source property"] = ""
    summary["Fine-tuning ratio (%)"] = 0
    summary["sMAPE (%)"] = np.nan
    summary["sMAPE (ratio)"] = np.nan
    summary["Displayed in Fig5b"] = pd.NA
    summary["Displayed model minimum"] = pd.NA
    summary["All-model minimum"] = pd.NA
    summary["Source workbook"] = relative_path(SOURCE_WORKBOOK)
    summary["Source worksheet"] = SOURCE_SHEET

    fields = [
        "Record type",
        "Model",
        "Property",
        "Source property",
        "Fine-tuning ratio (%)",
        "sMAPE (%)",
        "sMAPE (ratio)",
        "Displayed in Fig5b",
        "Displayed model minimum",
        "All-model minimum",
        "Displayed minimum sMAPE",
        "Displayed best model(s)",
        "All-model minimum sMAPE",
        "All-model best model(s)",
        "Global best omitted from Fig5a subset",
        "Proposed sMAPE",
        "Proposed is all-model best",
        "Source workbook",
        "Source worksheet",
    ]
    for field in fields:
        if field not in values:
            values[field] = pd.NA
        if field not in summary:
            summary[field] = pd.NA
    output = pd.concat(
        [values.reindex(columns=fields).astype(object), summary.reindex(columns=fields).astype(object)],
        ignore_index=True,
    )
    order = {property_name: index for index, property_name in enumerate(TARGET_PROPERTIES)}
    output["_property_order"] = output["Property"].map(order)
    return output.sort_values(["Record type", "Model", "_property_order"], kind="stable").drop(columns="_property_order")


def plot_heatmap(
    source: pd.DataFrame,
    validation: pd.DataFrame,
    displayed_models: list[str],
    vmin: float,
    vmax: float,
    pdf_path: Path,
    png_path: Path,
) -> None:
    """Plot a labelled, column-best-marked heatmap using the persisted scale."""
    matrix = (
        source.loc[source["Model"].isin(displayed_models)]
        .pivot(index="Model", columns="Property", values="sMAPE (ratio)")
        .reindex(index=displayed_models, columns=TARGET_PROPERTIES)
    )
    if matrix.isna().any().any():
        missing = matrix.isna()
        raise ValueError(f"Cannot plot incomplete zero-shot heatmap:\n{missing[missing.any(axis=1)]}")

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.labelcolor": "#20252B",
            "xtick.color": "#20252B",
            "ytick.color": "#20252B",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    ):
        figure_size = (7.60, 3.45)
        fig, axis = plt.subplots(figsize=figure_size)
        image = axis.imshow(matrix.to_numpy(float), cmap=COLORMAP, vmin=vmin, vmax=vmax, aspect="auto")
        axis.set_title(
            "(b) Zero-Shot Prediction on the Unseen Flowsheet",
            fontsize=14.5,
            fontweight="normal",
            color="#20252B",
            pad=16,
        )
        axis.set_xticks(range(len(TARGET_PROPERTIES)), [PROPERTY_LABELS[item] for item in TARGET_PROPERTIES])
        axis.set_yticks(range(len(displayed_models)), [MODEL_LABELS.get(item, item) for item in displayed_models])
        axis.set_xlabel("Target property", fontsize=11.6, labelpad=8)
        axis.set_ylabel("Model", fontsize=11.6, labelpad=8)
        axis.tick_params(axis="x", labelsize=10.0, length=0, pad=6)
        axis.tick_params(axis="y", labelsize=10.0, length=0, pad=6)
        # Only the three long non-chemical property labels need rotation;
        # keeping the formulas horizontal preserves quick column scanning.
        for tick_label in axis.get_xticklabels()[-3:]:
            tick_label.set_rotation(30)
            tick_label.set_ha("right")
            tick_label.set_rotation_mode("anchor")

        # Light cell divisions clarify the matrix without adding a numerical
        # 0/1-style grid or obscuring the colour mapping.
        axis.set_xticks(np.arange(-0.5, len(TARGET_PROPERTIES), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(displayed_models), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.15)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)

        normalization = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        global_minimum = validation.set_index("Property")["All-model minimum sMAPE"]
        for row_index, model in enumerate(displayed_models):
            for column_index, property_name in enumerate(TARGET_PROPERTIES):
                value = float(matrix.loc[model, property_name])
                is_global_best = bool(np.isclose(value, global_minimum[property_name]))
                text_color = "white" if normalization(value) < 0.48 else "#20252B"
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9.25,
                    fontweight="bold" if is_global_best else "normal",
                    zorder=4,
                )
                if is_global_best:
                    axis.add_patch(
                        Rectangle(
                            (column_index - 0.5, row_index - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor="#111111",
                            linewidth=2.1,
                            zorder=5,
                        )
                    )

        colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.028)
        colorbar.set_label("sMAPE", fontsize=11.2, labelpad=8)
        colorbar.ax.tick_params(labelsize=9.5, length=3.0)
        colorbar.outline.set_edgecolor("#9AA7B4")
        colorbar.outline.set_linewidth(0.65)
        # The expanded left margin keeps “Graph-to-SFILES” intact, while the
        # deeper lower margin accommodates the rotated long property labels.
        fig.subplots_adjust(left=0.215, right=0.915, top=0.835, bottom=0.275)
        fig.savefig(pdf_path, format="pdf", facecolor="white")
        fig.savefig(png_path, format="png", dpi=600, facecolor="white")
        plt.close(fig)
    print(f"Figure size: {figure_size[0]:.2f} x {figure_size[1]:.2f} inches")


def write_scale_json(path: Path, source: pd.DataFrame, displayed_models: list[str], vmin: float, vmax: float) -> None:
    """Persist Fig. 5b's exact scale for Fig. 5c reuse."""
    payload = {
        "figure": "Fig. 5b",
        "title": "Zero-Shot Prediction on the Unseen Flowsheet",
        "value_definition": "sMAPE as a unitless ratio; source percentage values divided by 100",
        "normalization": "linear",
        "colormap": COLORMAP,
        "vmin": vmin,
        "vmax": vmax,
        "scale_basis": "vmin=0; vmax is the all-model 0% maximum rounded upward to the nearest 0.1",
        "all_model_zero_shot_maximum": float(source["sMAPE (ratio)"].max()),
        "displayed_models": displayed_models,
        "all_evaluated_models": source["Model"].drop_duplicates().tolist(),
        "target_properties": list(TARGET_PROPERTIES),
        "source_workbook": relative_path(SOURCE_WORKBOOK),
        "source_worksheet": SOURCE_SHEET,
        "reuse_instruction": "Reuse vmin, vmax, colormap, and linear normalization unchanged for Fig. 5c.",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def print_validation(validation: pd.DataFrame) -> None:
    """Print the all-model comparison required to audit the displayed subset."""
    print("\nColumn-minimum validation (sMAPE ratio):")
    print(
        validation.loc[
            :,
            [
                "Property",
                "Displayed minimum sMAPE",
                "Displayed best model(s)",
                "All-model minimum sMAPE",
                "All-model best model(s)",
            ],
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    proposed_wins = validation.loc[validation["Proposed is all-model best"], "Property"].tolist()
    non_wins = validation.loc[~validation["Proposed is all-model best"], "Property"].tolist()
    print(f"\nProposed best count across ALL evaluated models: {len(proposed_wins)}/{len(validation)}")
    print(f"Properties Proposed wins: {', '.join(proposed_wins) if proposed_wins else 'None'}")
    print(f"Properties Proposed does not win: {', '.join(non_wins) if non_wins else 'None'}")
    for property_name in non_wins:
        row = validation.loc[validation["Property"].eq(property_name)].iloc[0]
        competitors = [
            model
            for model in str(row["All-model best model(s)"]).split("; ")
            if model != "Proposed"
        ]
        print(f"Best competitor for {property_name}: {'; '.join(competitors)} ({row['All-model minimum sMAPE']:.6f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    source, source_models = load_zero_shot_source()
    initial_models = list(fig5a.SELECTED_MODELS)
    validation, omitted_global_winners = all_model_validation(source, initial_models)
    displayed_models = determine_display_models(source, validation, source_models)
    # Recalculate displayed minima if an exceptional omitted global winner was
    # added above; all-model minima and Proposed win status remain unchanged.
    validation, remaining_omitted = all_model_validation(source, displayed_models)
    if remaining_omitted:
        raise RuntimeError(f"Global zero-shot winners are still absent from the heatmap: {remaining_omitted}")
    vmin, vmax = scale_limits(source)

    print("Source data files used:")
    print(f"- {relative_path(SOURCE_WORKBOOK)} (worksheet: {SOURCE_SHEET})")
    print("Model-selection reference used:")
    print(f"- {relative_path(FIG5A_SELECTION_REFERENCE)}")
    print("0% source models:")
    print("- " + ", ".join(source_models))
    print("Fig. 5b displayed models:")
    print("- " + ", ".join(displayed_models))
    if omitted_global_winners:
        print(f"WARNING: Fig. 5a subset initially omitted global winner(s): {omitted_global_winners}")
    else:
        print("All property-wise global zero-shot winners are included in the Fig. 5a subset.")
    print(f"Saved heatmap scale: vmin={vmin:.6f}, vmax={vmax:.6f}, cmap={COLORMAP}, normalization=linear")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / "Fig5b_zero_shot_heatmap.pdf"
    png_path = args.output_dir / "Fig5b_zero_shot_heatmap_600dpi.png"
    csv_path = args.output_dir / "Fig5b_zero_shot_heatmap.csv"
    scale_path = args.output_dir / "Fig5_heatmap_scale.json"
    build_csv(source, validation, displayed_models).to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_heatmap(source, validation, displayed_models, vmin, vmax, pdf_path, png_path)
    write_scale_json(scale_path, source, displayed_models, vmin, vmax)
    print_validation(validation)
    print("\nOutput paths:")
    for path in (pdf_path, png_path, csv_path, scale_path):
        print(f"- {relative_path(path)}")


if __name__ == "__main__":
    main()
