"""Create only Fig. 7a from archived timing records; no timing is rerun.

The neural values are measured batch-one latency per held-out flowsheet-graph
sample.  Aspen Plus is an existing 1.67-s manuscript reference only: its
timing provenance is incomplete and is reported explicitly rather than treated
as a matched benchmark.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "outputs/0819final/Paper_Tables_Final3_figures/final/experiment_results/experiment_results.xlsx"
BASELINE_JSON = ROOT / "outputs/inference_benchmark_multi_baselines_balanced10_final_20260916-230540/multi_baseline_inference_benchmark.json"
PROPOSED_TARGET_JSON = ROOT / "outputs/inference_benchmark_proposed_targetonly_balanced10_final_20260917-000124/proposed_targetonly_balanced10_final-20260917-000128/inference_benchmark.json"
PROPOSED_FULL_JSON = ROOT / "outputs/inference_benchmark_proposed_fold01_balanced10_20260916/proposed_inference_benchmark_fold01_balanced10-20260916-202150/inference_benchmark.json"
MANIFEST = ROOT / "outputs/inference_benchmark_manifests_20260916/fold01_test_one_sample_per_process.csv"
ASPEN_REFERENCE_SCRIPT = ROOT / "scripts/plot_efficiency_four_panel_measured.py"
OUTPUT_DIR = ROOT / "outputs/0819final/Paper_Tables_Final3_figures/final/Fig7a_inference_time"


DISPLAY_NAME = {
    "B6": "Reusable-Distillation-ANN",
    "GCN": "GCN",
    "GIN": "GIN",
    "GAT": "GAT",
    "Graphormer": "Graphormer",
    "SAT": "SAT",
    "GraphToSFILES": "Graph-to-SFILES",
}
PLOT_ORDER = (
    "Aspen Plus",
    "Proposed",
    "Reusable-Distillation-ANN",
    "GCN",
    "GIN",
    "GAT",
    "Graphormer",
    "SAT",
    "Graph-to-SFILES",
)


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return result


def _read_aspen_seconds(path: Path) -> float:
    """Read the manuscript reference from code instead of re-entering a value."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ASPEN_PLUS_INFERENCE_SECONDS":
                    value = ast.literal_eval(node.value)
                    if isinstance(value, (int, float)) and value > 0:
                        return float(value)
    raise ValueError(f"ASPEN_PLUS_INFERENCE_SECONDS was not found in {path}")


def _check_workbook_timing(timing: pd.DataFrame) -> None:
    workbook = pd.read_excel(WORKBOOK, sheet_name="Efficiency")
    workbook = workbook.set_index("Model")
    for row in timing.loc[timing["Figure_included"] & timing["System"].ne("Aspen Plus")].itertuples(index=False):
        if row.System not in workbook.index:
            raise ValueError(f"{row.System} is absent from the current Efficiency worksheet")
        expected_seconds = float(workbook.loc[row.System, "Inference time (ms/sample)"]) / 1000.0
        if not np.isclose(row.Time_s_per_flowsheet_sample, expected_seconds, rtol=0.0, atol=1e-12):
            raise ValueError(f"Workbook timing mismatch for {row.System}: {expected_seconds} vs {row.Time_s_per_flowsheet_sample}")


def _records() -> pd.DataFrame:
    for path in (WORKBOOK, BASELINE_JSON, PROPOSED_TARGET_JSON, PROPOSED_FULL_JSON, MANIFEST, ASPEN_REFERENCE_SCRIPT):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = pd.read_csv(MANIFEST)
    if len(manifest) != 10 or set(manifest["process_id"]) != {f"Process{index}" for index in range(1, 11)}:
        raise ValueError("The timing manifest is not the expected one-sample-per-P01–P10 record")

    baseline = _read_json(BASELINE_JSON)
    protocol = baseline.get("protocol")
    if not isinstance(protocol, dict):
        raise TypeError("Baseline timing protocol is missing")
    baseline_records = baseline.get("records")
    if not isinstance(baseline_records, list):
        raise TypeError("Baseline timing records are missing")

    rows: list[dict[str, object]] = []
    aspen_seconds = _read_aspen_seconds(ASPEN_REFERENCE_SCRIPT)
    rows.append(
        {
            "System": "Aspen Plus",
            "Figure_included": True,
            "Time_s_per_flowsheet_sample": aspen_seconds,
            "Time_SD_s_per_flowsheet_sample": np.nan,
            "Original_value": aspen_seconds,
            "Original_unit": "s/sample",
            "Unit_conversion": "None; existing manuscript reference is already seconds per sample.",
            "Timing_scope": "Existing manuscript reference described as a process-simulation call.",
            "Sample_count_per_run": np.nan,
            "Batch_size": np.nan,
            "Warmup_runs": np.nan,
            "Measured_runs": np.nan,
            "Hardware": "Not recorded",
            "Device": "Not recorded",
            "Checkpoint_provenance": "Not applicable",
            "Source_file": str(ASPEN_REFERENCE_SCRIPT),
            "Timing_status": "Existing manuscript reference; incomplete provenance",
            "CPU_metadata": "Not recorded",
            "Core_count": "Not recorded",
            "Aspen_version": "Not recorded",
            "Warm_or_cold_start": "Not recorded",
            "Convergence_time_included": "Not recorded",
            "Repetition_metadata": "Not recorded",
            "Metadata_warning": "WARNING: Aspen timing is not a fully validated matched benchmark.",
        }
    )

    for record in baseline_records:
        if not isinstance(record, dict):
            continue
        model = DISPLAY_NAME.get(str(record.get("model", "")))
        if model is None:
            continue
        milliseconds = float(record["inference_time_per_sample_ms"])
        standard_deviation_ms = float(record["inference_time_std_ms"])
        samples_per_run = int(record["samples_per_run"])
        if samples_per_run != len(manifest) or int(record["inference_batch_size"]) != 1:
            raise ValueError(f"Invalid sample/batch protocol for {model}")
        rows.append(
            {
                "System": model,
                "Figure_included": True,
                "Time_s_per_flowsheet_sample": milliseconds / 1000.0,
                "Time_SD_s_per_flowsheet_sample": standard_deviation_ms / 1000.0,
                "Original_value": milliseconds,
                "Original_unit": "ms/sample",
                "Unit_conversion": "milliseconds per batch-one flowsheet sample / 1000",
                "Timing_scope": str(record["timing_scope"]),
                "Sample_count_per_run": samples_per_run,
                "Batch_size": int(record["inference_batch_size"]),
                "Warmup_runs": int(record["warmup_runs"]),
                "Measured_runs": int(record["measured_runs"]),
                "Hardware": str(protocol.get("hardware", "Not recorded")),
                "Device": str(record.get("device", protocol.get("device", "Not recorded"))),
                "Checkpoint_provenance": str(record.get("checkpoint_provenance", "Not recorded")),
                "Source_file": str(BASELINE_JSON),
                "Timing_status": "Measured timing record",
                "CPU_metadata": "Not applicable (GPU benchmark)",
                "Core_count": "Not applicable (GPU benchmark)",
                "Aspen_version": "Not applicable",
                "Warm_or_cold_start": "10 warm-up runs before timed runs",
                "Convergence_time_included": "Not applicable",
                "Repetition_metadata": "50 timed repetitions; reported SD available",
                "Metadata_warning": "",
            }
        )

    proposed_target = _read_json(PROPOSED_TARGET_JSON)
    target_ms = float(proposed_target["inference_time_per_sample_ms"])
    target_sd_ms = float(proposed_target["inference_time_std_ms"])
    if int(proposed_target["samples_per_run"]) != len(manifest) or int(proposed_target["inference_batch_size"]) != 1:
        raise ValueError("Invalid proposed target timing protocol")
    rows.append(
        {
            "System": "Proposed",
            "Figure_included": True,
            "Time_s_per_flowsheet_sample": target_ms / 1000.0,
            "Time_SD_s_per_flowsheet_sample": target_sd_ms / 1000.0,
            "Original_value": target_ms,
            "Original_unit": "ms/sample",
            "Unit_conversion": "milliseconds per batch-one flowsheet sample / 1000",
            "Timing_scope": str(proposed_target["timing_scope"]),
            "Sample_count_per_run": int(proposed_target["samples_per_run"]),
            "Batch_size": int(proposed_target["inference_batch_size"]),
            "Warmup_runs": int(proposed_target["warmup_runs"]),
            "Measured_runs": int(proposed_target["measured_runs"]),
            "Hardware": "Not recorded in this JSON (device recorded)",
            "Device": str(proposed_target.get("device", "Not recorded")),
            "Checkpoint_provenance": "final proposed target-output checkpoint",
            "Source_file": str(PROPOSED_TARGET_JSON),
            "Timing_status": "Measured timing record",
            "CPU_metadata": "Not applicable (GPU benchmark)",
            "Core_count": "Not applicable (GPU benchmark)",
            "Aspen_version": "Not applicable",
            "Warm_or_cold_start": "10 warm-up runs before timed runs",
            "Convergence_time_included": "Not applicable",
            "Repetition_metadata": "50 timed repetitions; reported SD available",
            "Metadata_warning": "",
        }
    )

    # The full-stream proposed timing is retained in the CSV, but it is not
    # plotted because no measured full-stream latency is archived for every
    # baseline; a scenario extension would not be a valid comparison.
    proposed_full = _read_json(PROPOSED_FULL_JSON)
    rows.append(
        {
            "System": "Proposed (full-stream; excluded)",
            "Figure_included": False,
            "Time_s_per_flowsheet_sample": float(proposed_full["inference_time_per_sample_ms"]) / 1000.0,
            "Time_SD_s_per_flowsheet_sample": float(proposed_full["inference_time_std_ms"]) / 1000.0,
            "Original_value": float(proposed_full["inference_time_per_sample_ms"]),
            "Original_unit": "ms/sample",
            "Unit_conversion": "milliseconds per batch-one flowsheet sample / 1000",
            "Timing_scope": str(proposed_full["timing_scope"]),
            "Sample_count_per_run": int(proposed_full["samples_per_run"]),
            "Batch_size": int(proposed_full["inference_batch_size"]),
            "Warmup_runs": int(proposed_full["warmup_runs"]),
            "Measured_runs": int(proposed_full["measured_runs"]),
            "Hardware": "Not recorded in this JSON (device recorded)",
            "Device": str(proposed_full.get("device", "Not recorded")),
            "Checkpoint_provenance": "final proposed full-stream checkpoint",
            "Source_file": str(PROPOSED_FULL_JSON),
            "Timing_status": "Measured but excluded from plot: no matched measured full-stream baseline timings",
            "CPU_metadata": "Not applicable (GPU benchmark)",
            "Core_count": "Not applicable (GPU benchmark)",
            "Aspen_version": "Not applicable",
            "Warm_or_cold_start": "10 warm-up runs before timed runs",
            "Convergence_time_included": "Not applicable",
            "Repetition_metadata": "50 timed repetitions; reported SD available",
            "Metadata_warning": "Not plotted to avoid comparing it against target-output baseline timings.",
        }
    )
    frame = pd.DataFrame(rows)
    _check_workbook_timing(frame)
    return frame


def _format_seconds(value: float) -> str:
    if value >= 1.0:
        return f"{value:.2f} s"
    return f"{value * 1000.0:.2f} ms"


def _plot(data: pd.DataFrame) -> tuple[Path, Path]:
    plotted = data.loc[data["Figure_included"]].set_index("System").loc[list(PLOT_ORDER)].reset_index()
    figure, axis = plt.subplots(figsize=(6.65, 4.30), constrained_layout=False)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    positions = np.arange(len(plotted))
    colors = ["#C44E52" if name == "Aspen Plus" else "#111111" if name == "Proposed" else "#4C78A8" for name in plotted["System"]]
    for position, row, color in zip(positions, plotted.itertuples(index=False), colors, strict=True):
        time = float(row.Time_s_per_flowsheet_sample)
        standard_deviation = float(row.Time_SD_s_per_flowsheet_sample) if pd.notna(row.Time_SD_s_per_flowsheet_sample) else np.nan
        if np.isfinite(standard_deviation):
            axis.errorbar(time, position, xerr=standard_deviation, color="#5E6875", capsize=2.4, linewidth=0.85, zorder=2)
        axis.scatter(time, position, s=54 if row.System in {"Aspen Plus", "Proposed"} else 40, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        axis.annotate(_format_seconds(time), (time, position), xytext=(6, 0), textcoords="offset points", va="center", ha="left", fontsize=7.4, color="#303640")
    axis.set_xscale("log")
    axis.set_yticks(positions, plotted["System"])
    axis.invert_yaxis()
    axis.set_xlabel("Inference / evaluation time (s per flowsheet sample; log scale)", fontsize=10.0, labelpad=7)
    # The wider left margin accommodates long system labels; offset the axes
    # title so its visual centre remains the centre of the full panel.
    axis.set_title("(a) Inference Time per Flowsheet Evaluation", x=0.3358, fontsize=12.0, pad=13, weight="normal")
    axis.grid(axis="x", which="major", color="#E4E8ED", linewidth=0.7)
    axis.tick_params(axis="both", labelsize=8.8, length=3.0, width=0.8, colors="#30343B")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#59616D")
        axis.spines[spine].set_linewidth(0.8)
    axis.set_xlim(1e-4, 3.0)
    figure.text(0.5, 0.025, "Model timings: 10 held-out P01–P10 flowsheet-graph samples, batch size 1, 10 warm-ups, and 50 timed repetitions.\nAspen Plus: the archived 1.67 s/sample reference lacks hardware, version, start-state, convergence, and repetition metadata.", ha="center", va="bottom", fontsize=6.7, color="#4B5563")
    figure.subplots_adjust(left=0.27, right=0.955, top=0.86, bottom=0.235)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "Fig7a_inference_time.pdf"
    png_path = OUTPUT_DIR / "Fig7a_inference_time_600dpi.png"
    figure.savefig(pdf_path, facecolor="white")
    figure.savefig(png_path, dpi=600, facecolor="white")
    plt.close(figure)
    return pdf_path, png_path


def main() -> None:
    data = _records()
    pdf_path, png_path = _plot(data)
    csv_path = OUTPUT_DIR / "Fig7a_inference_time.csv"
    data.to_csv(csv_path, index=False)
    print("SOURCE FILE PATHS USED:")
    for path in (WORKBOOK, BASELINE_JSON, PROPOSED_TARGET_JSON, PROPOSED_FULL_JSON, MANIFEST, ASPEN_REFERENCE_SCRIPT):
        print(f"- {path}")
    print("\nTIMING VALUES USED (seconds per batch-one flowsheet-graph sample):")
    print(data.loc[data["Figure_included"], ["System", "Time_s_per_flowsheet_sample", "Time_SD_s_per_flowsheet_sample", "Timing_status"]].to_string(index=False))
    print("\nASPEN METADATA CHECK:")
    aspen = data.loc[data["System"].eq("Aspen Plus")].iloc[0]
    for field in ("CPU_metadata", "Core_count", "Aspen_version", "Warm_or_cold_start", "Convergence_time_included", "Repetition_metadata"):
        print(f"- {field}: {aspen[field]}")
    print(f"- warning: {aspen['Metadata_warning']}")
    print("\nFIGURE SIZE: 6.65 x 4.30 in")
    print("OUTPUT PATHS:")
    for path in (pdf_path, png_path, csv_path):
        print(f"- {path}")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.titleweight": "normal", "pdf.fonttype": 42, "ps.fonttype": 42})
    main()
