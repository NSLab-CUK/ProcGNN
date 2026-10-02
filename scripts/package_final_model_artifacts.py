#!/usr/bin/env python3
"""Build the canonical, publication-ready final-model artifact package.

The source render directories are preserved.  This script copies only the
latest approved figures and their supporting tables into a stable directory
whose names do not contain dates, experiment versions, or styling notes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = Path("outputs/0819final/Paper_Tables_Final3_figures")

TARGET_SOURCE = FIGURE_ROOT / "final/shap_target_material_27_flowsheets_20260922"
TARGET_SUMMARY_SOURCE = FIGURE_ROOT / "final/shap_target_material_27_summary_20260922"
TRANSFER_SOURCE = FIGURE_ROOT / "final/transfer_learning_figures"
ABSOLUTE_SHAP_SOURCE = FIGURE_ROOT / "final/shap_absolute_abc_no_legend_subtitles_top_20260922"
EFFICIENCY_SOURCE = FIGURE_ROOT / "efficiency_analysis_individual_figure3_subtitles_top_train080_v24_20260922"

DEFAULT_OUTPUT = FIGURE_ROOT / "final/final_model_results"
FORMATS = ("png", "pdf", "svg")


TARGET_EDGE_PANELS = {
    "H2": (
        ("a", "P01", "E021"), ("b", "P02", "E014"),
        ("c", "P03", "E025"), ("d", "P04", "E019"),
        ("e", "P05", "E016"), ("f", "P06", "E022"),
        ("g", "P07", "E016"), ("h", "P08", "E015"),
        ("i", "P09", "E020"), ("j", "P10", "E029"),
    ),
    "CO2": (
        ("a", "P01", "E015"), ("b", "P02", "E016"),
        ("c", "P03", "E014"), ("d", "P04", "E009"),
        ("e", "P05", "E017"), ("f", "P06", "E023"),
        ("g", "P07", "E016"), ("h", "P08", "E016"),
        ("i", "P09", "E022"), ("j", "P10", "E015"),
    ),
    "H2O": (
        ("a", "P01", "E031"), ("b", "P03", "E023"),
        ("c", "P05", "E015"), ("d", "P06", "E021"),
        ("e", "P07", "E017"), ("f", "P08", "E014"),
        ("g", "P10", "E027"),
    ),
}

TRANSFER_PROPERTIES = (
    ("a", "h2o", "H2O"),
    ("b", "h2", "H2"),
    ("c", "ch4", "CH4"),
    ("d", "co2", "CO2"),
    ("e", "co", "CO"),
    ("f", "o2", "O2"),
    ("g", "n2", "N2"),
    ("h", "temp", "temperature"),
    ("i", "pres", "pressure"),
    ("j", "mass_flow", "mass_flow"),
)

ABSOLUTE_SHAP_PANELS = (
    (
        "mean_absolute_relative_shap_by_node_type_all_processes",
        "a_unit_type",
    ),
    (
        "mean_absolute_relative_shap_by_operating_variable_all_processes",
        "b_operating_variable",
    ),
    (
        "mean_absolute_relative_shap_by_feed_type_existing_processes",
        "c_feed_type",
    ),
)

EFFICIENCY_PANELS = (
    ("efficiency_panel_a_process_output_training", "a_training_process_output"),
    ("efficiency_panel_b_streamwise_training", "b_training_stream_wise"),
    ("efficiency_panel_c_process_output_inference", "c_inference_process_output"),
    ("efficiency_panel_d_streamwise_inference", "d_inference_stream_wise"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Canonical package directory, relative to the repository root unless absolute.",
    )
    parser.add_argument(
        "--refresh-transfer",
        action="store_true",
        help="Replace only transfer figures and their data in an existing canonical package.",
    )
    return parser.parse_args()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_copy(
    source: Path,
    destination: Path,
    *,
    package_root: Path,
    category: str,
    item: str,
    records: list[dict[str, str]],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(source, destination)
    except OSError as error:
        # CopyFile2 can reject an otherwise readable file on the mapped
        # research drive when an image preview has a mapped view open.  Copy
        # through a sibling temporary file and atomically replace only after
        # the full byte stream is available.
        temporary = destination.with_name(f"{destination.name}.copying")
        try:
            temporary.unlink(missing_ok=True)
            with source.open("rb") as source_handle, temporary.open("wb") as destination_handle:
                shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
            shutil.copystat(source, temporary)
            temporary.replace(destination)
        except OSError as fallback_error:
            temporary.unlink(missing_ok=True)
            raise OSError(
                f"Could not copy source={source} to destination={destination}: "
                f"CopyFile2={error}; streamed fallback={fallback_error}"
            ) from fallback_error
    records.append(
        {
            "category": category,
            "item": item,
            "format": destination.suffix.lstrip(".") or "data",
            "destination": destination.relative_to(package_root).as_posix(),
            "source": source.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(destination),
        }
    )


def _figure_jobs(output: Path) -> list[tuple[Path, Path, str, str]]:
    jobs: list[tuple[Path, Path, str, str]] = []

    for species, panels in TARGET_EDGE_PANELS.items():
        for panel, process, edge in panels:
            source_stem = TARGET_SOURCE / f"{species}_target_edges/shap_signed_flowsheet_{panel}_{process}_{edge}"
            destination_stem = output / f"figures/target_edge_shap/{species}/{panel}_{process}_{edge}"
            for extension in FORMATS:
                jobs.append((source_stem.with_suffix(f".{extension}"), destination_stem.with_suffix(f".{extension}"), "target_edge_shap", f"{species}_{process}_{edge}"))

    for range_slug in ("0_10", "0_100"):
        for panel, source_property, destination_property in TRANSFER_PROPERTIES:
            source_stem = TRANSFER_SOURCE / f"transfer_individual/transfer_{source_property}_smape_{range_slug}"
            destination_stem = output / f"figures/transfer_learning/range_{range_slug}/{panel}_{destination_property}"
            for extension in FORMATS:
                jobs.append((source_stem.with_suffix(f".{extension}"), destination_stem.with_suffix(f".{extension}"), "transfer_learning", f"range_{range_slug}_{destination_property}"))

    for source_name, destination_name in ABSOLUTE_SHAP_PANELS:
        for extension in FORMATS:
            jobs.append((ABSOLUTE_SHAP_SOURCE / f"{source_name}.{extension}", output / f"figures/absolute_shap/{destination_name}.{extension}", "absolute_shap", destination_name))

    for source_name, destination_name in EFFICIENCY_PANELS:
        for extension in FORMATS:
            jobs.append((EFFICIENCY_SOURCE / f"{source_name}.{extension}", output / f"figures/efficiency/{destination_name}.{extension}", "efficiency", destination_name))

    return jobs


def _data_jobs(output: Path) -> list[tuple[Path, Path, str, str]]:
    jobs: list[tuple[Path, Path, str, str]] = []

    for species in TARGET_EDGE_PANELS:
        jobs.append((TARGET_SOURCE / f"{species}_target_edges/shap_target_edge_node_summary.csv", output / f"data/target_edge_shap/{species}_node_summary.csv", "target_edge_shap_data", f"{species}_node_summary"))
    target_files = (
        (TARGET_SOURCE / "shap_target_edge_node_summary.csv", "node_summary.csv"),
        (TARGET_SOURCE / "target_edge_species_map.csv", "target_edge_map.csv"),
        (TARGET_SOURCE / "shap_overlay_manifest.json", "figure_manifest.json"),
        (TARGET_SOURCE / "IMPLEMENTATION_DETAILS.md", "implementation_details.md"),
        (TARGET_SUMMARY_SOURCE / "selected_sample_level_shap.csv", "sample_level_shap.csv"),
        (TARGET_SUMMARY_SOURCE / "shap_target_material_node_summary.csv", "material_node_summary.csv"),
        (TARGET_SUMMARY_SOURCE / "target_material_map.csv", "target_material_map.csv"),
        (TARGET_SUMMARY_SOURCE / "target_material_shap_coverage.csv", "coverage.csv"),
        (TARGET_SUMMARY_SOURCE / "target_material_shap_manifest.json", "analysis_manifest.json"),
    )
    for source, destination_name in target_files:
        jobs.append((source, output / f"data/target_edge_shap/{destination_name}", "target_edge_shap_data", destination_name))

    for source_name, destination_name in (
        ("transfer_smape_plot_data.csv", "plot_data.csv"),
        ("figure_manifest.csv", "figure_manifest.csv"),
    ):
        jobs.append((TRANSFER_SOURCE / source_name, output / f"data/transfer_learning/{destination_name}", "transfer_learning_data", destination_name))

    for source_name, destination_name in (
        ("absolute_shap_node_type_mean_all_processes.csv", "unit_type.csv"),
        ("absolute_shap_operating_variable_mean_all_processes.csv", "operating_variable.csv"),
        ("absolute_shap_feed_type_mean_all_processes.csv", "feed_type.csv"),
        ("absolute_shap_panels_display_data.csv", "all_panels.csv"),
    ):
        jobs.append((ABSOLUTE_SHAP_SOURCE / source_name, output / f"data/absolute_shap/{destination_name}", "absolute_shap_data", destination_name))

    for source_name, destination_name in (
        ("efficiency_individual_panels_data.csv", "plot_data.csv"),
        ("efficiency_individual_panels_data.xlsx", "plot_data.xlsx"),
    ):
        jobs.append((EFFICIENCY_SOURCE / source_name, output / f"data/efficiency/{destination_name}", "efficiency_data", destination_name))

    return jobs


def _readme() -> str:
    return """# Final model artifacts

이 폴더는 최종 모델을 기준으로 확정된 논문용 그림과 근거 데이터를 한곳에 정리한 canonical package입니다. 날짜, 실험 버전, 임시 스타일 설정은 파일명에서 제거했으며, 원본 렌더링 폴더는 수정하지 않았습니다.

## Final model

- Checkpoint: `outputs/0819final/sensitivity_10d_clean/pin/node_mass_low/All/fold_01/checkpoints/process_kfold_All_F01-20260903-021159/best.pt`
- Evaluation split: `data/splits/all_processes_full100k_outer5_grouped_60_20_20/fold_01/test.csv`
- Full target-edge SHAP settings and libraries: `data/target_edge_shap/implementation_details.md`

## Contents

| Directory | Description | Figure count |
|---|---|---:|
| `figures/target_edge_shap/` | Target-edge-material signed SHAP flowsheets: H2 10, CO2 10, H2O 7 | 27 |
| `figures/transfer_learning/` | Ten properties for the 0-10 and 0-100 ranges | 20 |
| `figures/absolute_shap/` | Mean absolute SHAP by unit type, operating variable, and feed type | 3 |
| `figures/efficiency/` | Training/inference efficiency panels for process-output and stream-wise prediction | 4 |

Total: 54 figures. Every figure is provided as PNG, PDF, and SVG.

## File naming

- Panel prefixes (`a_`, `b_`, ...) preserve the order printed in each figure family.
- Target-edge SHAP uses `{panel}_{process}_{edge}`, for example `a_P01_E021`.
- Transfer figures are separated into `range_0_10` and `range_0_100`; filenames contain only the panel and property.
- No canonical filename contains a date, run number, checkpoint timestamp, font multiplier, or internal revision number.

## Final display settings

- Target-edge SHAP: one explanation per process-edge-material target; common signed relative-SHAP colour range of +/-0.26.
- Transfer learning: subtitles above every panel; typography is 0.85x of the preceding final transfer set. The shared model legend appears only in panel (e), CO, inside the upper-right data-free plotting area. No baseline-mean zoom inset is included.
- Absolute SHAP: subtitles above the panels, no legend, and common blue mean-|SHAP| bars with standard-deviation whiskers.
- Efficiency: all subtitles above the panels. Panels (a) and (b) use a continuous sMAPE axis capped at 0.80. Panels (c) and (d) keep the established y-axis range and the Aspen Plus 1.67 s/sample reference.

## Supporting data

The `data/` directory mirrors the four figure families. It contains the plotted CSV/XLSX values, target maps, coverage tables, and manifests. `SOURCE_MANIFEST.csv` records the original source path and SHA-256 checksum for every copied file.

## Rebuild

Run from the repository root:

```powershell
python scripts/package_final_model_artifacts.py
```

The packager refuses to overwrite an existing canonical directory. Remove or archive an older package explicitly before rebuilding.
"""


def _write_source_manifest(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("category", "item", "format", "destination", "source", "sha256"),
        )
        writer.writeheader()
        writer.writerows(records)


def _copy_jobs(
    jobs: list[tuple[Path, Path, str, str]],
    *,
    output: Path,
) -> list[dict[str, str]]:
    missing = [str(_absolute(source)) for source, _, _, _ in jobs if not _absolute(source).is_file()]
    if missing:
        raise FileNotFoundError("Missing source artifacts:\n" + "\n".join(missing))
    records: list[dict[str, str]] = []
    for source, destination, category, item in jobs:
        _record_copy(
            _absolute(source),
            destination,
            package_root=output,
            category=category,
            item=item,
            records=records,
        )
    return records


def main() -> None:
    args = parse_args()
    output = _absolute(args.output_dir)
    if args.refresh_transfer:
        if not output.is_dir():
            raise FileNotFoundError(f"Canonical package does not exist: {output}")
        transfer_jobs = [
            job for job in (_figure_jobs(output) + _data_jobs(output))
            if job[2].startswith("transfer_learning")
        ]
        refreshed_records = _copy_jobs(transfer_jobs, output=output)
        manifest = output / "SOURCE_MANIFEST.csv"
        with manifest.open(encoding="utf-8-sig", newline="") as handle:
            existing_records = list(csv.DictReader(handle))
        retained_records = [
            record for record in existing_records
            if not record["category"].startswith("transfer_learning")
        ]
        _write_source_manifest(manifest, retained_records + refreshed_records)
        (output / "README.md").write_text(_readme(), encoding="utf-8")
        print(f"output={output}")
        print(f"refreshed_transfer_files={len(refreshed_records)}")
        return
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing final package: {output}")

    jobs = _figure_jobs(output) + _data_jobs(output)
    output.mkdir(parents=True, exist_ok=False)
    records = _copy_jobs(jobs, output=output)

    manifest = output / "SOURCE_MANIFEST.csv"
    _write_source_manifest(manifest, records)
    (output / "README.md").write_text(_readme(), encoding="utf-8")

    figure_records = [record for record in records if not record["category"].endswith("_data")]
    print(f"output={output}")
    print(f"figures={len(figure_records) // len(FORMATS)}")
    print(f"figure_files={len(figure_records)}")
    print(f"supporting_files={len(records) - len(figure_records)}")


if __name__ == "__main__":
    main()
