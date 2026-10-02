"""Generate only Fig. 5a from existing transfer-learning result data.

The script reads the manuscript workbook's ``Transfer_sMAPE`` sheet, computes
unweighted means across the ten required target properties, and draws the
0--10% fine-tuning panel.  It does not train, fine-tune, or evaluate a model;
it also does not create the requested zero-shot or 2% heatmaps.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
PROTOCOL_EVIDENCE = ROOT / "scripts" / "plot_transfer_baseline_fixed_figures.py"
OUT_DIR = (
    ROOT
    / "outputs"
    / "0819final"
    / "Paper_Tables_Final3_figures"
    / "final"
    / "Fig5a_mean_smape_vs_finetuning"
)

# The source includes longer-range measurements as well.  Fig. 5a is the
# manuscript's 0--10% adaptation panel and uses the existing points only.
REQUESTED_RATIOS = (0, 2, 4, 6, 8, 10)
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

# No conventional/non-graph transfer baseline is present in the source data.
# These three existing baseline families give a compact, meaningful comparison.
SELECTED_MODELS = ("Proposed", "GCN", "Graphormer", "GraphToSFILES")
MODEL_RATIONALES = {
    "Proposed": "Proposed process-graph model.",
    "GCN": "Generic message-passing GNN baseline.",
    "Graphormer": "Global-attention graph-transformer baseline.",
    "GraphToSFILES": "Graph-to-sequence structural comparator.",
}
MODEL_STYLE = {
    "Proposed": {"label": "Proposed", "color": "#111111", "marker": "*", "lw": 2.25, "ms": 11.5, "zorder": 5},
    "GCN": {"label": "GCN", "color": "#2A6FBB", "marker": "o", "lw": 1.85, "ms": 6.6, "zorder": 3},
    "Graphormer": {"label": "Graphormer", "color": "#C5659A", "marker": "D", "lw": 1.85, "ms": 6.3, "zorder": 3},
    "GraphToSFILES": {"label": "Graph-to-SFILES", "color": "#42B6D7", "marker": "P", "lw": 1.85, "ms": 6.6, "zorder": 3},
}


def relative_path(path: Path) -> str:
    """Return a repository-relative POSIX path for reproducibility fields."""
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def normalize_property(value: object) -> str:
    """Map workbook property spellings to the ten requested property names."""
    # The workbook correctly stores Unicode subscripts.  Building the mapping
    # from code points avoids shell/editor codepage ambiguity on Windows.
    subscript_map = str.maketrans({chr(0x2082): "2", chr(0x2084): "4"})
    normalized = str(value).strip().translate(subscript_map)
    return {"Temp": "Temperature", "Pres": "Pressure"}.get(normalized, normalized)


def load_source() -> tuple[pd.DataFrame, list[int], list[str]]:
    """Read existing per-property transfer sMAPE rows and validate schema."""
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
    source["model"] = source["model"].astype(str).str.strip()
    source["transfer_percent"] = pd.to_numeric(source["transfer_percent"], errors="coerce")
    source["sMAPE_mean_pct"] = pd.to_numeric(source["sMAPE_mean_pct"], errors="coerce")
    source["Source property"] = source["property"].astype(str).str.strip()
    source["Property"] = source["Source property"].map(normalize_property)
    source = source.dropna(subset=["transfer_percent", "sMAPE_mean_pct"])
    source["transfer_percent"] = source["transfer_percent"].astype(int)

    available_ratios = sorted(source["transfer_percent"].unique().tolist())
    source_models = source["model"].drop_duplicates().tolist()
    available_properties = source["Property"].drop_duplicates().tolist()
    missing_properties = sorted(set(TARGET_PROPERTIES).difference(available_properties))
    if missing_properties:
        raise ValueError(f"Transfer source lacks required properties: {missing_properties}")
    return source, available_ratios, source_models


def extract_figure_scope(source: pd.DataFrame, available_ratios: list[int]) -> tuple[pd.DataFrame, list[int]]:
    """Keep only recorded 0--10% target-property values for this subfigure."""
    ratios = [ratio for ratio in REQUESTED_RATIOS if ratio in available_ratios]
    absent = [ratio for ratio in REQUESTED_RATIOS if ratio not in available_ratios]
    if absent:
        print(f"WARNING: requested 0--10% ratios absent from the source and omitted: {absent}")
    if 0 not in ratios:
        raise ValueError("The existing zero-shot (0%) point is required for Fig. 5a but is unavailable.")
    if len(ratios) < 2:
        raise ValueError(f"Only {ratios} are available in the requested Fig. 5a ratio range.")

    scoped = source.loc[
        source["transfer_percent"].isin(ratios)
        & source["Property"].isin(TARGET_PROPERTIES)
    ].copy()
    if scoped.empty:
        raise ValueError("No target-property transfer sMAPE rows exist in the requested 0--10% scope.")
    return scoped, ratios


def aggregate_ten_property_means(scoped: pd.DataFrame, ratios: list[int], models: list[str]) -> pd.DataFrame:
    """Compute a mean only when exactly one value exists for every target property."""
    records: list[dict[str, object]] = []
    expected = set(TARGET_PROPERTIES)
    for model in models:
        for ratio in ratios:
            group = scoped.loc[
                scoped["model"].eq(model) & scoped["transfer_percent"].eq(ratio)
            ].copy()
            counts = group["Property"].value_counts(dropna=False)
            present = set(counts.index.tolist())
            missing = sorted(expected.difference(present))
            duplicated = sorted(counts[counts.gt(1)].index.tolist())
            complete = not missing and not duplicated and len(group) == len(TARGET_PROPERTIES)
            if not complete:
                print(
                    "WARNING: incomplete model-ratio combination; mean not computed: "
                    f"model={model}, ratio={ratio}, missing={missing}, duplicated={duplicated}, rows={len(group)}"
                )
            records.append(
                {
                    "Record type": "aggregate_mean" if complete else "incomplete_model_ratio",
                    "Model": model,
                    "Fine-tuning ratio (%)": ratio,
                    "Property": "",
                    "sMAPE (%)": math.nan,
                    "sMAPE (ratio)": math.nan,
                    "Number of properties used": int(len(group["Property"].unique())),
                    "Complete 10-property coverage": complete,
                    "Missing properties": "; ".join(missing),
                    "Duplicated properties": "; ".join(duplicated),
                    "Mean sMAPE (%)": float(group["sMAPE_mean_pct"].mean()) if complete else math.nan,
                    "Mean sMAPE (ratio)": float(group["sMAPE_mean_pct"].mean() / 100.0) if complete else math.nan,
                    "Used in Fig5a": model in SELECTED_MODELS and complete,
                    "Source workbook": relative_path(SOURCE_WORKBOOK),
                    "Source worksheet": SOURCE_SHEET,
                }
            )
    return pd.DataFrame(records)


def build_output_csv(scoped: pd.DataFrame, aggregates: pd.DataFrame) -> pd.DataFrame:
    """Preserve source values and the exact mean values used for plotting."""
    source_rows = scoped.loc[:, ["model", "transfer_percent", "Property", "Source property", "sMAPE_mean_pct"]].copy()
    source_rows = source_rows.rename(
        columns={
            "model": "Model",
            "transfer_percent": "Fine-tuning ratio (%)",
            "sMAPE_mean_pct": "sMAPE (%)",
        }
    )
    source_rows.insert(0, "Record type", "source_property")
    source_rows["Number of properties used"] = pd.NA
    source_rows["Complete 10-property coverage"] = pd.NA
    source_rows["Missing properties"] = ""
    source_rows["Duplicated properties"] = ""
    source_rows["Mean sMAPE (%)"] = pd.NA
    source_rows["sMAPE (ratio)"] = source_rows["sMAPE (%)"] / 100.0
    source_rows["Mean sMAPE (ratio)"] = pd.NA
    source_rows["Used in Fig5a"] = source_rows["Model"].isin(SELECTED_MODELS)
    source_rows["Source workbook"] = relative_path(SOURCE_WORKBOOK)
    source_rows["Source worksheet"] = SOURCE_SHEET

    fields = [
        "Record type",
        "Model",
        "Fine-tuning ratio (%)",
        "Property",
        "Source property",
        "sMAPE (%)",
        "sMAPE (ratio)",
        "Number of properties used",
        "Complete 10-property coverage",
        "Missing properties",
        "Duplicated properties",
        "Mean sMAPE (%)",
        "Mean sMAPE (ratio)",
        "Used in Fig5a",
        "Source workbook",
        "Source worksheet",
    ]
    aggregate_rows = aggregates.copy()
    aggregate_rows["Source property"] = ""
    for field in fields:
        if field not in source_rows:
            source_rows[field] = pd.NA
        if field not in aggregate_rows:
            aggregate_rows[field] = pd.NA
    # Retain source-only and aggregate-only fields without an all-NA dtype
    # inference warning from newer pandas releases.
    output = pd.concat(
        [
            source_rows.reindex(columns=fields).astype(object),
            aggregate_rows.reindex(columns=fields).astype(object),
        ],
        ignore_index=True,
    )
    return output.sort_values(["Record type", "Model", "Fine-tuning ratio (%)", "Property"], kind="stable")


def plot_figure(aggregates: pd.DataFrame, ratios: list[int], pdf_path: Path, png_path: Path) -> None:
    """Draw the requested single-panel data-efficiency line figure."""
    plot_data = aggregates.loc[
        aggregates["Used in Fig5a"] & aggregates["Complete 10-property coverage"]
    ].copy()
    expected_rows = len(SELECTED_MODELS) * len(ratios)
    if len(plot_data) != expected_rows:
        raise ValueError(
            f"Fig. 5a needs {expected_rows} complete selected model-ratio means, found {len(plot_data)}."
        )

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.2,
            "axes.labelcolor": "#20252B",
            "xtick.color": "#20252B",
            "ytick.color": "#20252B",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(7.60, 5.05))
    fig.patch.set_facecolor("white")
    axis.set_facecolor("white")

    for model in SELECTED_MODELS:
        group = plot_data.loc[plot_data["Model"].eq(model)].sort_values("Fine-tuning ratio (%)")
        style = MODEL_STYLE[model]
        axis.plot(
            group["Fine-tuning ratio (%)"],
            group["Mean sMAPE (ratio)"],
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linewidth=style["lw"],
            markersize=style["ms"],
            markeredgewidth=0.85,
            markeredgecolor="white" if model != "Proposed" else "#111111",
            zorder=style["zorder"],
        )

    all_means = plot_data["Mean sMAPE (ratio)"].to_numpy(float)
    y_top = float(math.ceil((all_means.max() * 1.10) / 0.02) * 0.02)
    y_bottom = 0.0
    axis.set_ylim(y_bottom, y_top)
    axis.set_xlim(min(ratios) - 0.45, max(ratios) + 0.70)
    axis.set_xticks(ratios, ["0\n(Zero-shot)"] + [str(ratio) for ratio in ratios[1:]])
    axis.set_xlabel("Fine-Tuning Data Ratio (%)", fontsize=13.1, labelpad=9)
    axis.set_ylabel("Mean sMAPE", fontsize=13.1, labelpad=8)
    axis.set_title(
        "(a) Data Efficiency for Unseen-Flowsheet Adaptation",
        fontsize=15.1,
        fontweight="normal",
        color="#20252B",
        pad=17,
    )
    axis.tick_params(axis="both", labelsize=10.9, length=3.5, width=0.75)
    axis.grid(axis="y", color="#E3E8ED", linewidth=0.62, alpha=0.80)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#AEB7C2")
    axis.spines["bottom"].set_color("#AEB7C2")

    # Inspect the data before selecting a fixed legend location.  The upper
    # right is valid only when late-ratio curves remain below the legend box.
    late_max = float(
        plot_data.loc[plot_data["Fine-tuning ratio (%)"].ge(4), "Mean sMAPE (ratio)"].max()
    )
    legend_lower_bound = y_top - 0.31 * (y_top - y_bottom)
    if late_max < legend_lower_bound:
        legend_location = "upper right (inside axes; data-free)"
        axis.legend(
            loc="upper right",
            bbox_to_anchor=(0.982, 0.982),
            frameon=True,
            facecolor="white",
            edgecolor="#CBD3DB",
            framealpha=0.96,
            fontsize=10.2,
            handlelength=2.25,
            handletextpad=0.60,
            labelspacing=0.48,
            borderpad=0.56,
        )
    else:
        legend_location = "outside right (late-ratio data occupy upper-right)"
        axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.012, 1.0),
            frameon=True,
            facecolor="white",
            edgecolor="#CBD3DB",
            framealpha=0.96,
            fontsize=10.2,
            handlelength=2.25,
            handletextpad=0.60,
            labelspacing=0.48,
            borderpad=0.56,
        )
    print(
        "Legend placement inspection: "
        f"late-ratio maximum={late_max:.6f}, legend lower boundary={legend_lower_bound:.6f}; "
        f"{legend_location}."
    )

    fig.subplots_adjust(left=0.135, right=0.970, top=0.865, bottom=0.165)
    fig.savefig(pdf_path, format="pdf", facecolor="white")
    fig.savefig(png_path, format="png", dpi=600, facecolor="white")
    plt.close(fig)


def print_validation(aggregates: pd.DataFrame, ratios: list[int]) -> None:
    """Print required completeness and proposed-model checks."""
    print("\nValidation: model / ratio / mean sMAPE / number of properties used")
    table = aggregates.loc[:, ["Model", "Fine-tuning ratio (%)", "Mean sMAPE (%)", "Number of properties used"]]
    print(table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    proposed = aggregates.loc[
        aggregates["Model"].eq("Proposed") & aggregates["Complete 10-property coverage"]
    ].set_index("Fine-tuning ratio (%)")
    required = [ratio for ratio in (0, 2, 10) if ratio in proposed.index]
    if len(required) != 3:
        print("WARNING: Proposed 0%, 2%, and/or 10% summary is unavailable because a recorded ratio is incomplete.")
        return
    zero = float(proposed.loc[0, "Mean sMAPE (%)"])
    two = float(proposed.loc[2, "Mean sMAPE (%)"])
    ten = float(proposed.loc[10, "Mean sMAPE (%)"])
    print("\nProposed-model summary")
    print(f"0% zero-shot mean sMAPE: {zero:.6f}%")
    print(f"2% mean sMAPE: {two:.6f}%")
    print(f"10% mean sMAPE: {ten:.6f}%")
    print(f"Relative change, 0% to 2%: {100.0 * (two - zero) / zero:.6f}%")
    print(f"Relative change, 0% to 10%: {100.0 * (ten - zero) / zero:.6f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate only Fig. 5a from existing transfer sMAPE results.")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    source, available_ratios, source_models = load_source()
    scoped, ratios = extract_figure_scope(source, available_ratios)
    missing_selected = [model for model in SELECTED_MODELS if model not in source_models]
    if missing_selected:
        raise ValueError(f"Requested representative models are absent from source data: {missing_selected}")
    aggregates = aggregate_ten_property_means(scoped, ratios, source_models)

    print("Source data files used:")
    print(f"- {relative_path(SOURCE_WORKBOOK)} (worksheet: {SOURCE_SHEET})")
    print("Protocol evidence inspected for the 0% zero-shot label (not numerical input):")
    print(f"- {relative_path(PROTOCOL_EVIDENCE)}")
    print(f"Available transfer ratios in source: {available_ratios}")
    print(f"Fig. 5a recorded ratios used: {ratios}")
    print(f"Target properties averaged (exactly {len(TARGET_PROPERTIES)}): {list(TARGET_PROPERTIES)}")
    print("Selected models:")
    for model in SELECTED_MODELS:
        print(f"- {model}: {MODEL_RATIONALES[model]}")
    print("No conventional/non-graph transfer model is available in this source; it is not substituted or invented.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / "Fig5a_mean_smape_vs_finetuning.pdf"
    png_path = args.output_dir / "Fig5a_mean_smape_vs_finetuning_600dpi.png"
    csv_path = args.output_dir / "Fig5a_mean_smape_vs_finetuning.csv"
    output_csv = build_output_csv(scoped, aggregates)
    output_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
    plot_figure(aggregates, ratios, pdf_path, png_path)
    print_validation(aggregates, ratios)
    print("\nFigure size: 7.60 x 5.05 inches")
    print(f"Output PDF: {relative_path(pdf_path)}")
    print(f"Output PNG (600 dpi): {relative_path(png_path)}")
    print(f"Output processed values CSV: {relative_path(csv_path)}")


if __name__ == "__main__":
    main()
