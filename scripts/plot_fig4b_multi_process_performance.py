"""Generate only Fig. 4b from the manuscript's multi-process Table 3.

The script imports shared Figure 4 styling/extraction helpers, but writes only
the Fig. 4b outputs below. No model training or evaluation is performed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import plot_fig4a_single_process_performance as fig4


ROOT = fig4.ROOT
SOURCE_WORKBOOK = (
    ROOT
    / "outputs"
    / "0819final"
    / "Paper_Tables_Final3_figures"
    / "final"
    / "experiment_results"
    / "experiment_results.xlsx"
)
SOURCE_SHEET = "Table3_Multi_10D"
OUT_DIR = (
    ROOT
    / "outputs"
    / "0819final"
    / "Paper_Tables_Final3_figures"
    / "final"
    / "Fig4b_multi_process_performance"
)


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def prepare_outputs(overwrite: bool) -> dict[str, Path]:
    outputs = {
        "pdf": OUT_DIR / "Fig4b_multi_process_performance.pdf",
        "png": OUT_DIR / "Fig4b_multi_process_performance_600dpi.png",
        "csv": OUT_DIR / "Fig4b_multi_process_performance.csv",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing Fig. 4b outputs: {names}. Use --overwrite.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()
    return outputs


def print_validation(source, comparison, source_properties: list[str]) -> None:
    print("\nTarget-property headings in current Table 3:")
    print("- " + ", ".join(source_properties))
    baseline_count = int(
        source["Comparison eligibility"].eq("baseline").groupby(source["Model"]).first().sum()
    )
    print(f"Model rows extracted: {source['Model'].nunique()} (baseline candidates: {baseline_count})")
    print("\nProperty comparison (sMAPE %):")
    display = comparison.loc[
        :,
        [
            "Property",
            "Proposed sMAPE (%)",
            "Best baseline",
            "Best baseline sMAPE (%)",
            "Relative reduction (%)",
        ],
    ]
    print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    proposed_best_count = int(comparison["Proposed is best"].sum())
    non_best = comparison.loc[~comparison["Proposed is best"], "Property"].tolist()
    average_reduction = float(comparison["Relative reduction (%)"].mean())
    print(f"\nProperties where Proposed is best: {proposed_best_count}/{len(comparison)}")
    print(f"Properties where Proposed is not best: {', '.join(non_best) if non_best else 'None'}")
    print(f"Average relative reduction: {average_reduction:.4f}%")
    print(
        "WARNING: The current Table 3 caption does not state an independent count of properties "
        "where Proposed is best; the displayed sMAPE means yield the reported count above."
    )


def check_axis_compatibility(comparison) -> None:
    values = comparison[["Proposed sMAPE (ratio)", "Best baseline sMAPE (ratio)"]].to_numpy(float).ravel()
    if not np.all(np.isfinite(values)):
        raise ValueError("Fig. 4b comparison contains non-finite sMAPE values.")
    maximum = float(values.max())
    minimum = float(values.min())
    # Fig. 4a uses a 0.0005--0.40 log-scale sMAPE-ratio range. Existing Fig. 4b data exceed
    # that upper bound, so a shared range would conceal its largest pairs.
    if minimum < 0.0005 or maximum > 0.40:
        print(
            "WARNING: Shared Fig. 4a y limits (0.0005-0.40 log-scale ratio) are inappropriate "
            f"for Fig. 4b values ({minimum:.6f}-{maximum:.6f}); using an independent readable axis."
        )
    else:
        print("Fig. 4b values fit the Fig. 4a y limits; shared limits are feasible.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace only the three Fig. 4b outputs")
    args = parser.parse_args()

    if not SOURCE_WORKBOOK.is_file():
        raise FileNotFoundError(f"Current manuscript workbook is missing: {SOURCE_WORKBOOK}")
    # Shared helpers resolve their data path at runtime; set their scope to
    # Table 3 before extraction, without invoking or recreating Fig. 4a.
    fig4.SOURCE_WORKBOOK = SOURCE_WORKBOOK
    fig4.SOURCE_SHEET = SOURCE_SHEET

    print("Source files used:")
    print(f"- {relative_path(SOURCE_WORKBOOK)} (worksheet: {SOURCE_SHEET})")
    source, source_properties = fig4.extract_source_table()
    source, comparison = fig4.make_comparison(source)
    check_axis_compatibility(comparison)
    outputs = prepare_outputs(args.overwrite)
    fig4.output_table(source, comparison).to_csv(outputs["csv"], index=False, float_format="%.10g")
    figure_size = fig4.draw_figure(
        comparison,
        outputs["pdf"],
        outputs["png"],
        title="(b) Multi-Process Prediction",
    )
    print_validation(source, comparison, source_properties)
    print(f"\nFigure size: {figure_size[0]:.2f} x {figure_size[1]:.2f} inches")
    print("Output paths:")
    for path in outputs.values():
        print(f"- {relative_path(path)}")


if __name__ == "__main__":
    main()
