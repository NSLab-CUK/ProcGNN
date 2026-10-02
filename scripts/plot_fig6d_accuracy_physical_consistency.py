"""Create ONLY Fig. 6d from measured physics-ablation summary results."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
SOURCE_WORKBOOK = ROOT / "outputs" / "0819final" / "Paper_Tables_Final3_figures" / "final" / "experiment_results" / "experiment_results.xlsx"
OUTPUT_DIR = ROOT / "outputs" / "0819final" / "Paper_Tables_Final3_figures" / "final" / "Fig6d_accuracy_physical_consistency"

SETTING_MAP = {
    "No PINN": ("None", "No physics"),
    "Mass only": ("M", "Mass only"),
    "Component only": ("C", "Component only"),
    "Atom only": ("E", "Elemental only"),
    "Mass + Component": ("M+C", "Mass + Component"),
    "Mass + Atom": ("M+E", "Mass + Elemental"),
    "Component + Atom": ("C+E", "Component + Elemental"),
    "Full PINN": ("Full", "Full physics"),
}
ORDER = ["No PINN", "Mass only", "Component only", "Atom only", "Mass + Component", "Mass + Atom", "Component + Atom", "Full PINN"]


def _parse_mean_sd(value: object) -> tuple[float, float]:
    match = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*±\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", str(value))
    if match is None:
        raise ValueError(f"Expected 'mean ± SD', received {value!r}")
    return float(match.group(1)), float(match.group(2))


def _load_data() -> pd.DataFrame:
    performance = pd.read_excel(SOURCE_WORKBOOK, sheet_name="Ablation_PINN_10D")
    constraint = pd.read_excel(SOURCE_WORKBOOK, sheet_name="Constraint_PINN")
    if "Average is the arithmetic mean across the 10 property means" not in str(performance.iloc[-1, 0]):
        raise RuntimeError("The current performance table does not document the expected overall-sMAPE averaging rule.")
    if str(performance.iloc[0, -1]).strip() != "sMAPE (%)":
        raise RuntimeError("The final Ablation_PINN_10D column is not the expected average sMAPE field.")
    performance_rows: dict[str, tuple[float, float]] = {}
    for row in performance.iloc[1:9].itertuples(index=False):
        variant = str(row[0])
        if variant not in SETTING_MAP:
            raise RuntimeError(f"Unexpected physics-ablation variant in performance table: {variant}")
        performance_rows[variant] = _parse_mean_sd(row[-1])
    needed = {"Model", "Conservation term", "Tolerance", "Mean satisfaction rate", "Standard deviation"}
    if missing := needed.difference(constraint.columns):
        raise RuntimeError(f"Constraint_PINN lacks: {sorted(missing)}")
    constraint = constraint.copy()
    constraint["threshold"] = constraint["Tolerance"].astype(str).str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0].astype(float) / 100.0
    rows: list[dict[str, object]] = []
    for variant in ORDER:
        display, setting = SETTING_MAP[variant]
        values = {}
        for term in ("Mass", "Component", "Atom"):
            part = constraint.loc[(constraint["Model"].astype(str) == ({"No PINN": "w/o Physics-informed loss", "Full PINN": "Proposed (multi-process)"}.get(variant, variant))) & (constraint["Conservation term"].astype(str) == term) & np.isclose(constraint["threshold"], 0.05), ["Mean satisfaction rate", "Standard deviation"]]
            part = part.drop_duplicates()
            if len(part) != 1:
                raise RuntimeError(f"Expected one 5% {term} constraint row for {variant}, found {len(part)}")
            values[term] = (float(part.iloc[0, 0]), float(part.iloc[0, 1]))
        smape_percent, displayed_average_sd = performance_rows[variant]
        # The performance workbook documents an arithmetic mean of property means,
        # but has no fold-aligned overall-sMAPE series.  Its displayed ± value is
        # not used as a fold-wise overall error bar.
        mean_satisfaction = float(np.mean([values["Mass"][0], values["Component"][0], values["Atom"][0]]))
        rows.append({
            "Setting": setting,
            "Label": display,
            "Overall_sMAPE": smape_percent / 100.0,
            "sMAPE_SD": np.nan,
            "sMAPE_displayed_property_average_SD": displayed_average_sd / 100.0,
            "Mass_satisfaction_5": values["Mass"][0],
            "Component_satisfaction_5": values["Component"][0],
            "Elemental_satisfaction_5": values["Atom"][0],
            "Overall_or_mean_satisfaction_5": mean_satisfaction,
            "Satisfaction_SD": np.nan,
            "Mass_satisfaction_5_SD": values["Mass"][1],
            "Component_satisfaction_5_SD": values["Component"][1],
            "Elemental_satisfaction_5_SD": values["Atom"][1],
            "Metric_type": "DERIVED PROXY: arithmetic mean of Mass, Component, and Elemental satisfaction at 5%; no pooled counts stored for every ablation setting.",
            "Source_file": str(SOURCE_WORKBOOK),
        })
    data = pd.DataFrame(rows)
    dominated = []
    for current in data.itertuples(index=False):
        better = (data["Overall_sMAPE"] <= current.Overall_sMAPE) & (data["Overall_or_mean_satisfaction_5"] >= current.Overall_or_mean_satisfaction_5) & ((data["Overall_sMAPE"] < current.Overall_sMAPE) | (data["Overall_or_mean_satisfaction_5"] > current.Overall_or_mean_satisfaction_5))
        dominated.append(bool(better.any()))
    data["Pareto_status"] = np.where(dominated, "Dominated", "Non-dominated")
    return data


def _plot(data: pd.DataFrame) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(6.65, 4.30), constrained_layout=False)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    colors = {"No physics": "#C44E52", "Full physics": "#1F77B4"}
    offsets = {
        "None": (7, -9), "M": (7, 7), "C": (7, -10), "E": (7, 7),
        "M+C": (7, -10), "M+E": (7, 7), "C+E": (7, 7), "Full": (-27, 9),
    }
    for row in data.itertuples(index=False):
        emphasized = row.Setting in colors
        color = colors.get(row.Setting, "#6D7886")
        axis.scatter(
            row.Overall_sMAPE, row.Overall_or_mean_satisfaction_5,
            s=68 if emphasized else 43, color=color, edgecolor="#20252B" if emphasized else "white",
            linewidth=1.1 if emphasized else 0.65, zorder=4,
        )
        dx, dy = offsets[row.Label]
        axis.annotate(row.Label, (row.Overall_sMAPE, row.Overall_or_mean_satisfaction_5), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color="#2F3742", weight="bold" if emphasized else "normal", zorder=5)
    axis.set_xlabel("Overall sMAPE", fontsize=10.0, labelpad=6)
    axis.set_ylabel("Mean conservation satisfaction at 5%", fontsize=10.0, labelpad=6)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_xlim(data["Overall_sMAPE"].min() - 0.003, data["Overall_sMAPE"].max() + 0.004)
    axis.set_ylim(0.28, 0.84)
    axis.set_title("(d) Prediction Accuracy and Physical Consistency", x=0.4383, fontsize=12.0, pad=13, weight="normal")
    axis.tick_params(axis="both", labelsize=9.0, length=3.0, width=0.8, colors="#30343B")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#59616D")
        axis.spines[spine].set_linewidth(0.8)
    figure.text(0.5, 0.025, "Physical-consistency score is the unweighted mean of the three measured 5% satisfaction rates.\nNo pooled constraint count is stored for every ablation setting.", ha="center", va="bottom", fontsize=7.0, color="#4B5563")
    figure.subplots_adjust(left=0.145, right=0.955, top=0.86, bottom=0.235)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "Fig6d_accuracy_physical_consistency.pdf"
    png_path = OUTPUT_DIR / "Fig6d_accuracy_physical_consistency_600dpi.png"
    figure.savefig(pdf_path, facecolor="white")
    figure.savefig(png_path, dpi=600, facecolor="white")
    plt.close(figure)
    return pdf_path, png_path


def main() -> None:
    if not SOURCE_WORKBOOK.is_file():
        raise FileNotFoundError(SOURCE_WORKBOOK)
    data = _load_data()
    pdf_path, png_path = _plot(data)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "Fig6d_accuracy_physical_consistency.csv"
    data.to_csv(csv_path, index=False)
    print("SOURCE FILE PATHS USED:")
    print(f"- {SOURCE_WORKBOOK}")
    print("OVERALL sMAPE: existing arithmetic mean across the ten target-property means; converted from the table's percent scale to a decimal fraction.")
    print("PHYSICAL CONSISTENCY: DERIVED PROXY = unweighted mean of the three measured 5% satisfaction rates; no per-setting pooled constraint counts or fold-aligned overall SDs are stored.")
    print("\nABLATION DATA:")
    print(data[["Setting", "Overall_sMAPE", "Mass_satisfaction_5", "Component_satisfaction_5", "Elemental_satisfaction_5", "Overall_or_mean_satisfaction_5", "Pareto_status"]].to_string(index=False))
    print("\nNON-DOMINATED SETTINGS:")
    print(", ".join(data.loc[data["Pareto_status"] == "Non-dominated", "Setting"]))
    print("\nFIGURE SIZE: 6.65 x 4.30 in")
    print("OUTPUT PATHS:")
    for path in (pdf_path, png_path, csv_path):
        print(f"- {path}")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.titleweight": "normal", "pdf.fonttype": 42, "ps.fonttype": 42})
    main()
