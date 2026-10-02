"""Replace generic baseline labels in the final paper-performance workbook.

The numerical cells, formulas, and table layout are intentionally left unchanged.
Only the first-column model labels are mapped to the concrete baseline names used
in the manuscript.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from openpyxl import load_workbook


MODEL_LABELS = {
    "Kriging": "Kriging-KR31",
    "RBF": "Cubic RBF",
    "Neural Baseline 1": "GTL-ANN",
    "Neural Baseline 2": "Cumene-Efficiency-ANN",
    "Neural Baseline 3": "Cumene-Destruction-ANN",
    "Neural Baseline 4": "Reusable-Distillation-ANN",
    "Neural Baseline 5": "Distillation-Boundary-GP",
    "MLP (B6)": "Reusable-Distillation-ANN",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workbook",
        type=Path,
        default=Path("outputs/0819final/Final2_checked.xlsx"),
    )
    args = parser.parse_args()
    if not args.workbook.is_file():
        raise FileNotFoundError(args.workbook)

    workbook = load_workbook(args.workbook)
    changes: list[tuple[str, str, str]] = []
    for worksheet in workbook.worksheets:
        for row in worksheet.iter_rows(min_col=1, max_col=1):
            cell = row[0]
            if cell.value in MODEL_LABELS:
                old = cell.value
                cell.value = MODEL_LABELS[old]
                changes.append((worksheet.title, old, cell.value))

    workbook.save(args.workbook)
    print(f"Updated {len(changes)} model-label cells in {args.workbook}")
    for sheet, old, new in changes:
        print(f"[{sheet}] {old} -> {new}")


if __name__ == "__main__":
    main()
