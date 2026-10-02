"""Generate only Fig. 5c from existing 2% fine-tuning results.

The saved Fig. 5b heatmap scale and model order are inputs to this script.
No model is trained, fine-tuned, or re-evaluated here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

import plot_fig5b_zero_shot_heatmap as fig5b


ROOT = fig5b.ROOT
SOURCE_WORKBOOK = fig5b.SOURCE_WORKBOOK
SOURCE_SHEET = fig5b.SOURCE_SHEET
FIG5B_SCALE_PATH = fig5b.OUT_DIR / "Fig5_heatmap_scale.json"
OUT_DIR = (
    ROOT
    / "outputs"
    / "0819final"
    / "Paper_Tables_Final3_figures"
    / "final"
    / "Fig5c_2percent_heatmap"
)
TARGET_PROPERTIES = fig5b.TARGET_PROPERTIES
PROPERTY_LABELS = fig5b.PROPERTY_LABELS
MODEL_LABELS = fig5b.MODEL_LABELS


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def load_fig5b_scale() -> tuple[list[str], float, float, str]:
    """Load the immutable Fig. 5b scale and displayed model ordering."""
    if not FIG5B_SCALE_PATH.is_file():
        raise FileNotFoundError(
            "Fig. 5c requires the existing Fig. 5b scale file, but it is missing: "
            f"{FIG5B_SCALE_PATH}"
        )
    payload = json.loads(FIG5B_SCALE_PATH.read_text(encoding="utf-8"))
    required = {"displayed_models", "vmin", "vmax", "colormap", "normalization"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Fig. 5b scale file lacks fields: {missing}")
    if payload["normalization"] != "linear":
        raise ValueError(f"Fig. 5b scale normalization must remain unchanged; found {payload['normalization']!r}")
    models = [str(model) for model in payload["displayed_models"]]
    if not models or "Proposed" not in models:
        raise ValueError(f"Invalid Fig. 5b model order: {models}")
    vmin = float(payload["vmin"])
    vmax = float(payload["vmax"])
    if not np.isfinite([vmin, vmax]).all() or not vmax > vmin:
        raise ValueError(f"Invalid Fig. 5b scale limits: vmin={vmin}, vmax={vmax}")
    return models, vmin, vmax, str(payload["colormap"])


def load_two_percent_source() -> tuple[pd.DataFrame, list[str]]:
    """Extract only the existing 2% target-property transfer values."""
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
    source["Property"] = source["Source property"].map(fig5b.normalize_property)
    source = source.loc[
        source["Fine-tuning ratio (%)"].eq(2)
        & source["Property"].isin(TARGET_PROPERTIES)
    ].copy()
    source["Fine-tuning ratio (%)"] = source["Fine-tuning ratio (%)"].astype(int)
    source["sMAPE (ratio)"] = source["sMAPE (%)"] / 100.0
    if source.empty:
        raise ValueError("No existing 2% fine-tuning target-property sMAPE rows are available.")
    if source["sMAPE (ratio)"].isna().any() or (source["sMAPE (ratio)"] < 0).any():
        raise ValueError("The 2% source has missing or negative sMAPE values.")
    if source.duplicated(["Model", "Property"]).any():
        duplicate = source.loc[source.duplicated(["Model", "Property"], keep=False), ["Model", "Property"]]
        raise ValueError(f"Expected one 2% result per model/property; duplicates found:\n{duplicate}")

    source_models = source["Model"].drop_duplicates().tolist()
    expected = set(TARGET_PROPERTIES)
    for model in source_models:
        missing_properties = sorted(expected.difference(source.loc[source["Model"].eq(model), "Property"]))
        if missing_properties:
            raise ValueError(f"2% source is incomplete for {model}: missing {missing_properties}")
    return source, source_models


def validate_property_bests(source: pd.DataFrame, displayed_models: list[str]) -> pd.DataFrame:
    """Compare each displayed minimum against its true all-model minimum."""
    records: list[dict[str, object]] = []
    for property_name in TARGET_PROPERTIES:
        all_rows = source.loc[source["Property"].eq(property_name)].copy()
        shown_rows = all_rows.loc[all_rows["Model"].isin(displayed_models)].copy()
        if shown_rows.empty:
            raise ValueError(f"No Fig. 5b-displayed model has a 2% value for {property_name}")
        all_min = float(all_rows["sMAPE (ratio)"].min())
        shown_min = float(shown_rows["sMAPE (ratio)"].min())
        all_best = all_rows.loc[np.isclose(all_rows["sMAPE (ratio)"], all_min), "Model"].tolist()
        shown_best = shown_rows.loc[np.isclose(shown_rows["sMAPE (ratio)"], shown_min), "Model"].tolist()
        omitted_best = [model for model in all_best if model not in displayed_models]
        proposed_value = float(
            all_rows.loc[all_rows["Model"].eq("Proposed"), "sMAPE (ratio)"].iloc[0]
        )
        records.append(
            {
                "Property": property_name,
                "Displayed minimum sMAPE": shown_min,
                "Displayed best model(s)": "; ".join(shown_best),
                "All-model minimum sMAPE": all_min,
                "All-model best model(s)": "; ".join(all_best),
                "All-model best omitted from heatmap": "; ".join(omitted_best),
                "Proposed sMAPE": proposed_value,
                "Proposed is all-model best": bool(np.isclose(proposed_value, all_min)),
            }
        )
    return pd.DataFrame(records)


def build_csv(source: pd.DataFrame, validation: pd.DataFrame, displayed_models: list[str]) -> pd.DataFrame:
    """Save raw 2% values and all-model best checks used by the figure."""
    lookup = validation.set_index("Property")
    values = source.copy()
    values["Record type"] = "source_2percent_value"
    values["Displayed in Fig5c"] = values["Model"].isin(displayed_models)
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
    summary["Fine-tuning ratio (%)"] = 2
    summary["sMAPE (%)"] = np.nan
    summary["sMAPE (ratio)"] = np.nan
    summary["Displayed in Fig5c"] = pd.NA
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
        "Displayed in Fig5c",
        "Displayed model minimum",
        "All-model minimum",
        "Displayed minimum sMAPE",
        "Displayed best model(s)",
        "All-model minimum sMAPE",
        "All-model best model(s)",
        "All-model best omitted from heatmap",
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
    colormap: str,
    pdf_path: Path,
    png_path: Path,
) -> None:
    """Render Fig. 5c with Fig. 5b's exact geometry, styling, and scale."""
    matrix = (
        source.loc[source["Model"].isin(displayed_models)]
        .pivot(index="Model", columns="Property", values="sMAPE (ratio)")
        .reindex(index=displayed_models, columns=TARGET_PROPERTIES)
    )
    if matrix.isna().any().any():
        raise ValueError(f"Cannot plot incomplete 2% heatmap:\n{matrix.isna()}")
    validation_lookup = validation.set_index("Property")

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
        # Deliberately identical to Fig. 5b.
        figure_size = (7.60, 3.45)
        fig, axis = plt.subplots(figsize=figure_size)
        image = axis.imshow(matrix.to_numpy(float), cmap=colormap, vmin=vmin, vmax=vmax, aspect="auto")
        axis.set_title(
            "(c) Prediction after 2% Fine-Tuning",
            fontsize=14.5,
            fontweight="normal",
            color="#20252B",
            pad=16,
        )
        axis.set_xticks(range(len(TARGET_PROPERTIES)), [PROPERTY_LABELS[item] for item in TARGET_PROPERTIES])
        axis.set_yticks(range(len(displayed_models)), [MODEL_LABELS.get(item, item) for item in displayed_models])
        axis.set_xlabel("Target property", fontsize=11.6, labelpad=4)
        axis.set_ylabel("Model", fontsize=11.6, labelpad=8)
        axis.tick_params(axis="x", labelsize=10.0, length=0, pad=6)
        axis.tick_params(axis="y", labelsize=10.0, length=0, pad=6)
        for tick_label in axis.get_xticklabels()[-3:]:
            tick_label.set_rotation(30)
            tick_label.set_ha("right")
            tick_label.set_rotation_mode("anchor")
        axis.set_xticks(np.arange(-0.5, len(TARGET_PROPERTIES), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(displayed_models), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.15)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)

        normalization = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        for row_index, model in enumerate(displayed_models):
            for column_index, property_name in enumerate(TARGET_PROPERTIES):
                value = float(matrix.loc[model, property_name])
                all_minimum = float(validation_lookup.loc[property_name, "All-model minimum sMAPE"])
                displayed_minimum = float(validation_lookup.loc[property_name, "Displayed minimum sMAPE"])
                global_best_is_shown = bool(np.isclose(value, all_minimum))
                shown_best_only = bool(
                    np.isclose(value, displayed_minimum) and not np.isclose(displayed_minimum, all_minimum)
                )
                text_color = "white" if normalization(value) < 0.48 else "#20252B"
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9.25,
                    fontweight="bold" if global_best_is_shown else "normal",
                    zorder=4,
                )
                if global_best_is_shown:
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
                elif shown_best_only:
                    # This only occurs when a Fig. 5b-fixed model ordering
                    # omits the actual all-model winner.  The dashed border is
                    # deliberately distinct from the solid global-best border.
                    axis.add_patch(
                        Rectangle(
                            (column_index - 0.5, row_index - 0.5),
                            1,
                            1,
                            fill=False,
                            edgecolor="#6D7F92",
                            linewidth=1.55,
                            linestyle=(0, (3.0, 2.0)),
                            zorder=5,
                        )
                    )

        colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.028)
        colorbar.set_label("sMAPE", fontsize=11.2, labelpad=8)
        colorbar.ax.tick_params(labelsize=9.5, length=3.0)
        colorbar.outline.set_edgecolor("#9AA7B4")
        colorbar.outline.set_linewidth(0.65)
        # Deliberately identical to Fig. 5b.
        fig.subplots_adjust(left=0.215, right=0.915, top=0.835, bottom=0.340)
        omitted = validation.loc[
            validation["All-model best omitted from heatmap"].astype(str).str.len().gt(0),
            ["Property", "All-model best model(s)", "All-model minimum sMAPE"],
        ]
        if not omitted.empty:
            note = "; ".join(
                f"{row['Property']}: {row['All-model best model(s)']} ({row['All-model minimum sMAPE']:.3f})"
                for _, row in omitted.iterrows()
            )
            # This is an audit note, not a legend: it prevents the dashed
            # displayed-subset marker from being mistaken for the all-model
            # winner while retaining Fig. 5b's fixed four-row ordering.
            fig.text(
                0.215,
                0.018,
                f"Dashed: displayed subset. All-model minimum not shown — {note}.",
                ha="left",
                va="bottom",
                fontsize=8.4,
                color="#4B5563",
            )
        fig.savefig(pdf_path, format="pdf", facecolor="white")
        fig.savefig(png_path, format="png", dpi=600, facecolor="white")
        plt.close(fig)
    print(f"Figure size: {figure_size[0]:.2f} x {figure_size[1]:.2f} inches")


def print_validation(validation: pd.DataFrame) -> None:
    """Print the manuscript-auditable all-model best-performance results."""
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
                "All-model best omitted from heatmap",
            ],
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    wins = validation.loc[validation["Proposed is all-model best"], "Property"].tolist()
    losses = validation.loc[~validation["Proposed is all-model best"], "Property"].tolist()
    print(f"\nProposed best count across ALL evaluated models: {len(wins)}/{len(validation)}")
    print(f"Properties Proposed wins: {', '.join(wins) if wins else 'None'}")
    print(f"Properties Proposed does not win: {', '.join(losses) if losses else 'None'}")
    for property_name in losses:
        row = validation.loc[validation["Property"].eq(property_name)].iloc[0]
        print(
            f"Best model/value for {property_name}: {row['All-model best model(s)']} "
            f"({row['All-model minimum sMAPE']:.6f})"
        )
    omitted = validation.loc[
        validation["All-model best omitted from heatmap"].astype(str).str.len().gt(0),
        ["Property", "All-model best model(s)", "All-model minimum sMAPE"],
    ]
    if not omitted.empty:
        details = "; ".join(
            f"{row['Property']}: {row['All-model best model(s)']} ({row['All-model minimum sMAPE']:.6f})"
            for _, row in omitted.iterrows()
        )
        print(
            "WARNING: Fig. 5b-fixed model ordering omits an all-model winner. "
            "A dashed border marks the displayed-subset minimum instead of a solid global-best border: "
            + details
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    displayed_models, vmin, vmax, colormap = load_fig5b_scale()
    source, source_models = load_two_percent_source()
    unavailable = [model for model in displayed_models if model not in source_models]
    if unavailable:
        raise ValueError(f"Fig. 5b model(s) have no existing 2% result: {unavailable}")
    validation = validate_property_bests(source, displayed_models)

    print("Source data files used:")
    print(f"- {relative_path(SOURCE_WORKBOOK)} (worksheet: {SOURCE_SHEET})")
    print(f"- {relative_path(FIG5B_SCALE_PATH)} (Fig. 5b model order and fixed color scale)")
    print("2% source models:")
    print("- " + ", ".join(source_models))
    print("Fig. 5c displayed models copied from Fig. 5b:")
    print("- " + ", ".join(displayed_models))
    print(f"Loaded Fig. 5b color scale unchanged: vmin={vmin:.6f}, vmax={vmax:.6f}, cmap={colormap}, normalization=linear")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / "Fig5c_2percent_heatmap.pdf"
    png_path = args.output_dir / "Fig5c_2percent_heatmap_600dpi.png"
    csv_path = args.output_dir / "Fig5c_2percent_heatmap.csv"
    build_csv(source, validation, displayed_models).to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_heatmap(source, validation, displayed_models, vmin, vmax, colormap, pdf_path, png_path)
    print_validation(validation)
    print("\nOutput paths:")
    for path in (pdf_path, png_path, csv_path):
        print(f"- {relative_path(path)}")


if __name__ == "__main__":
    main()
