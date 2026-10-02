"""Create only Fig. 7b and Fig. 7c from the validated Fig. 7a timing CSV.

No timing or model evaluation is run.  Fig. 7c joins the Fig. 7a target-output
timing rows with the curated target-stream accuracy column in the current
Efficiency worksheet.  All-stream scenario extensions are deliberately not
used.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG7A_CSV = ROOT / "outputs/0819final/Paper_Tables_Final3_figures/final/Fig7a_inference_time/Fig7a_inference_time.csv"
WORKBOOK = ROOT / "outputs/0819final/Paper_Tables_Final3_figures/final/experiment_results/experiment_results.xlsx"
EFFICIENCY_CSV = ROOT / "outputs/0819final/Paper_Tables_Final3_figures/final/final_model_results/data/efficiency/plot_data.csv"
OUTPUT_7B = ROOT / "outputs/0819final/Paper_Tables_Final3_figures/final/Fig7b_speedup"
OUTPUT_7C = ROOT / "outputs/0819final/Paper_Tables_Final3_figures/final/Fig7c_accuracy_vs_inference"

SPEEDUP_ORDER = (
    "Proposed",
    "Graph-to-SFILES",
    "SAT",
    "Graphormer",
    "GAT",
    "GCN",
    "GIN",
    "Reusable-Distillation-ANN",
)
TRADEOFF_ORDER = (
    "Reusable-Distillation-ANN",
    "GIN",
    "GAT",
    "GCN",
    "Graphormer",
    "SAT",
    "Graph-to-SFILES",
    "Proposed",
)


def _timing_data() -> tuple[pd.DataFrame, float]:
    if not FIG7A_CSV.is_file():
        raise FileNotFoundError(f"Validated Fig. 7a timing file is required: {FIG7A_CSV}")
    timing = pd.read_csv(FIG7A_CSV)
    required = {"System", "Figure_included", "Time_s_per_flowsheet_sample", "Time_SD_s_per_flowsheet_sample", "Source_file", "Timing_scope", "Checkpoint_provenance"}
    missing = required.difference(timing.columns)
    if missing:
        raise ValueError(f"Fig. 7a timing CSV lacks: {sorted(missing)}")
    included = timing.loc[timing["Figure_included"].astype(bool)].copy()
    aspen = included.loc[included["System"].eq("Aspen Plus")]
    if len(aspen) != 1:
        raise ValueError("Fig. 7a must contain exactly one included Aspen Plus reference")
    aspen_time = float(aspen.iloc[0]["Time_s_per_flowsheet_sample"])
    if not np.isfinite(aspen_time) or aspen_time <= 0.0:
        raise ValueError("Invalid Aspen Plus time in Fig. 7a timing CSV")
    models = included.loc[included["System"].ne("Aspen Plus")].copy()
    if set(models["System"]) != set(TRADEOFF_ORDER):
        raise ValueError("Fig. 7a model set no longer matches the curated target-stream comparison models")
    return models, aspen_time


def _speedup_data(models: pd.DataFrame, aspen_time: float) -> pd.DataFrame:
    data = models.set_index("System").loc[list(SPEEDUP_ORDER)].reset_index().copy()
    data["Aspen_time_s_per_flowsheet_sample"] = aspen_time
    data["Speed_up_over_Aspen_x"] = aspen_time / data["Time_s_per_flowsheet_sample"].astype(float)
    data["Calculation"] = "Speed-up = Fig. 7a Aspen Plus time / Fig. 7a model inference time"
    data["Aspen_reference_status"] = "Existing manuscript reference; incomplete provenance as documented in Fig. 7a"
    return data


def _accuracy_data(models: pd.DataFrame) -> pd.DataFrame:
    for path in (WORKBOOK, EFFICIENCY_CSV):
        if not path.is_file():
            raise FileNotFoundError(path)
    # The workbook is the authoritative final-results package; the CSV is
    # cross-checked because it is the specific source indexed by that workbook.
    workbook = pd.read_excel(WORKBOOK, sheet_name="Efficiency").set_index("Model")
    efficiency = pd.read_csv(EFFICIENCY_CSV).set_index("Model")
    required = {"sMAPE (%)", "sMAPE SD (%)", "Inference time (ms/sample)"}
    for name, source in (("workbook", workbook), ("efficiency CSV", efficiency)):
        missing = required.difference(source.columns)
        if missing:
            raise ValueError(f"{name} lacks: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for timing in models.set_index("System").loc[list(TRADEOFF_ORDER)].itertuples():
        model = timing.Index
        if model not in workbook.index or model not in efficiency.index:
            raise ValueError(f"No curated target-stream accuracy for {model}")
        workbook_row = workbook.loc[model]
        csv_row = efficiency.loc[model]
        for column in ("sMAPE (%)", "sMAPE SD (%)", "Inference time (ms/sample)"):
            if not np.isclose(float(workbook_row[column]), float(csv_row[column]), rtol=0.0, atol=1e-12):
                raise ValueError(f"Workbook/CSV mismatch for {model} {column}")
        expected_time = float(workbook_row["Inference time (ms/sample)"]) / 1000.0
        if not np.isclose(float(timing.Time_s_per_flowsheet_sample), expected_time, rtol=0.0, atol=1e-12):
            raise ValueError(f"Fig. 7a/Efficiency timing mismatch for {model}")
        rows.append(
            {
                "Model": model,
                "Inference_time_s_per_flowsheet_sample": float(timing.Time_s_per_flowsheet_sample),
                "Inference_time_SD_s_per_flowsheet_sample": float(timing.Time_SD_s_per_flowsheet_sample),
                "Mean_sMAPE": float(workbook_row["sMAPE (%)"]) / 100.0,
                "Mean_sMAPE_SD": float(workbook_row["sMAPE SD (%)"]) / 100.0,
                "Accuracy_metric_source_scale": "Efficiency target-stream sMAPE (%), converted to unitless fraction for display",
                "Common_evaluation_setting": "Curated multi-process target-stream comparison; Fig. 7a batch-one latency and Efficiency target-stream performance",
                "Pairing_status": "Included: timing matches Efficiency target-inference value exactly",
                "Timing_scope": timing.Timing_scope,
                "Timing_checkpoint_provenance": timing.Checkpoint_provenance,
                "Timing_source_file": timing.Source_file,
                "Accuracy_source_file": str(WORKBOOK),
                "Efficiency_csv_source_file": str(EFFICIENCY_CSV),
            }
        )
    data = pd.DataFrame(rows)
    dominated: list[bool] = []
    for current in data.itertuples(index=False):
        dominates_current = (
            (data["Inference_time_s_per_flowsheet_sample"] <= current.Inference_time_s_per_flowsheet_sample)
            & (data["Mean_sMAPE"] <= current.Mean_sMAPE)
            & (
                (data["Inference_time_s_per_flowsheet_sample"] < current.Inference_time_s_per_flowsheet_sample)
                | (data["Mean_sMAPE"] < current.Mean_sMAPE)
            )
        )
        dominated.append(bool(dominates_current.any()))
    data["Pareto_status"] = np.where(dominated, "Pareto-dominated", "Non-dominated")
    return data


def _speedup_label(value: float) -> str:
    if value >= 1000.0:
        return f"{value / 1000.0:.2g}k×"
    if value >= 100.0:
        return f"{value:.0f}×"
    return f"{value:.1f}×"


def _plot_speedup(data: pd.DataFrame) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(6.65, 4.30), constrained_layout=False)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    positions = np.arange(len(data))
    for position, row in zip(positions, data.itertuples(index=False), strict=True):
        color = "#111111" if row.System == "Proposed" else "#4C78A8"
        axis.scatter(row.Speed_up_over_Aspen_x, position, s=53 if row.System == "Proposed" else 40, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        axis.annotate(_speedup_label(float(row.Speed_up_over_Aspen_x)), (row.Speed_up_over_Aspen_x, position), xytext=(6, 0), textcoords="offset points", ha="left", va="center", fontsize=7.5, color="#303640")
    axis.axvline(1.0, color="#AEB6C2", linewidth=0.85, linestyle=(0, (2, 3)), zorder=1)
    axis.annotate("Aspen Plus = 1×", (1.0, 0.30), xytext=(6, 0), textcoords="offset points", ha="left", va="center", fontsize=7.1, color="#667085")
    axis.set_xscale("log")
    axis.set_xlim(1.0, max(float(data["Speed_up_over_Aspen_x"].max()) * 2.1, 10.0))
    axis.set_yticks(positions, data["System"])
    axis.invert_yaxis()
    axis.set_xlabel("Speed-up over Aspen Plus (×; log scale)", fontsize=10.0, labelpad=7)
    axis.set_title("(b) Speed-Up over Aspen Plus", x=0.3008, fontsize=12.0, pad=13, weight="normal")
    axis.grid(axis="x", which="major", color="#E4E8ED", linewidth=0.7)
    axis.tick_params(axis="both", labelsize=8.8, length=3.0, width=0.8, colors="#30343B")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#59616D")
        axis.spines[spine].set_linewidth(0.8)
    figure.text(0.5, 0.025, "Computed exclusively from the Fig. 7a timing dataset. Aspen Plus remains an incomplete-provenance reference; see Fig. 7a CSV.", ha="center", va="bottom", fontsize=6.8, color="#4B5563")
    figure.subplots_adjust(left=0.30, right=0.955, top=0.86, bottom=0.215)
    OUTPUT_7B.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_7B / "Fig7b_speedup.pdf"
    png_path = OUTPUT_7B / "Fig7b_speedup_600dpi.png"
    figure.savefig(pdf_path, facecolor="white")
    figure.savefig(png_path, dpi=600, facecolor="white")
    plt.close(figure)
    return pdf_path, png_path


def _label_offsets(data: pd.DataFrame) -> list[tuple[float, float]]:
    """Use collision-resolved offsets for the known close baseline cluster."""
    offsets = {
        "Reusable-Distillation-ANN": (8, -10),
        "GIN": (-27, 14),
        "GAT": (9, -21),
        "GCN": (-34, 14),
        "Graphormer": (-76, 14),
        "SAT": (12, 8),
        "Graph-to-SFILES": (12, -21),
        "Proposed": (-54, 10),
    }
    missing = set(data["Model"]).difference(offsets)
    if missing:
        raise ValueError(f"No collision-resolved annotation offsets for: {sorted(missing)}")
    return [offsets[model] for model in data["Model"]]


def _plot_tradeoff(data: pd.DataFrame) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(6.65, 4.30), constrained_layout=False)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    offsets = _label_offsets(data)
    for row, offset in zip(data.itertuples(index=False), offsets, strict=True):
        is_pareto = row.Pareto_status == "Non-dominated"
        color = "#111111" if row.Model == "Proposed" else "#4C78A8"
        axis.scatter(row.Inference_time_s_per_flowsheet_sample, row.Mean_sMAPE, s=54 if is_pareto else 40, color=color, edgecolor="#20252B" if is_pareto else "white", linewidth=0.85, zorder=4)
        axis.annotate(row.Model, (row.Inference_time_s_per_flowsheet_sample, row.Mean_sMAPE), xytext=offset, textcoords="offset points", fontsize=7.5, color="#303640", weight="bold" if is_pareto else "normal", zorder=5)
    frontier = data.loc[data["Pareto_status"].eq("Non-dominated")].sort_values("Inference_time_s_per_flowsheet_sample")
    if len(frontier) >= 2:
        axis.plot(frontier["Inference_time_s_per_flowsheet_sample"], frontier["Mean_sMAPE"], color="#7D8794", linewidth=0.9, linestyle=(0, (3, 3)), zorder=2)
    axis.set_xscale("log")
    axis.set_xlim(1.5e-4, 7e-2)
    y_values = data["Mean_sMAPE"].to_numpy(float)
    axis.set_ylim(max(0.0, y_values.min() - 0.06), y_values.max() + 0.05)
    axis.set_xlabel("Inference time (s per flowsheet sample; log scale)", fontsize=10.0, labelpad=7)
    axis.set_ylabel("Mean sMAPE", fontsize=10.0, labelpad=6)
    axis.set_title("(c) Prediction Accuracy versus Inference Time", x=0.4383, fontsize=12.0, pad=13, weight="normal")
    axis.grid(axis="x", which="major", color="#E4E8ED", linewidth=0.7)
    axis.tick_params(axis="both", labelsize=8.8, length=3.0, width=0.8, colors="#30343B")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#59616D")
        axis.spines[spine].set_linewidth(0.8)
    figure.text(0.5, 0.025, "Target-stream, multi-process comparison only: measured Fig. 7a latency paired with the matching Efficiency mean sMAPE.\nDashed segment joins the mathematically non-dominated settings.", ha="center", va="bottom", fontsize=6.8, color="#4B5563")
    figure.subplots_adjust(left=0.145, right=0.955, top=0.86, bottom=0.215)
    OUTPUT_7C.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_7C / "Fig7c_accuracy_vs_inference.pdf"
    png_path = OUTPUT_7C / "Fig7c_accuracy_vs_inference_600dpi.png"
    figure.savefig(pdf_path, facecolor="white")
    figure.savefig(png_path, dpi=600, facecolor="white")
    plt.close(figure)
    return pdf_path, png_path


def main() -> None:
    models, aspen_time = _timing_data()
    speedup = _speedup_data(models, aspen_time)
    accuracy = _accuracy_data(models)
    speedup_pdf, speedup_png = _plot_speedup(speedup)
    tradeoff_pdf, tradeoff_png = _plot_tradeoff(accuracy)
    speedup_csv = OUTPUT_7B / "Fig7b_speedup.csv"
    tradeoff_csv = OUTPUT_7C / "Fig7c_accuracy_vs_inference.csv"
    speedup.to_csv(speedup_csv, index=False)
    accuracy.to_csv(tradeoff_csv, index=False)
    print("SOURCE FILE PATHS USED:")
    for path in (FIG7A_CSV, WORKBOOK, EFFICIENCY_CSV):
        print(f"- {path}")
    print("\nFIG. 7B SPEED-UP VALUES:")
    print(speedup[["System", "Time_s_per_flowsheet_sample", "Speed_up_over_Aspen_x"]].to_string(index=False))
    print("\nFIG. 7C MATCHED MODEL DATA:")
    print(accuracy[["Model", "Inference_time_s_per_flowsheet_sample", "Mean_sMAPE", "Pareto_status"]].to_string(index=False))
    print("\nNON-DOMINATED SETTINGS:")
    print(", ".join(accuracy.loc[accuracy["Pareto_status"].eq("Non-dominated"), "Model"]))
    print("\nFIGURE SIZE: 6.65 x 4.30 in for Fig. 7b and Fig. 7c")
    print("OUTPUT PATHS:")
    for path in (speedup_pdf, speedup_png, speedup_csv, tradeoff_pdf, tradeoff_png, tradeoff_csv):
        print(f"- {path}")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.titleweight": "normal", "pdf.fonttype": 42, "ps.fonttype": 42})
    main()
