#!/usr/bin/env python3
"""Restyle the Computational_Eff worksheet to match the paper result tables.

This only changes the worksheet presentation.  It retains the completed-run
metadata values already stored in the workbook.
"""
from __future__ import annotations

import argparse
import os
from copy import copy
from pathlib import Path

import openpyxl
from openpyxl.styles import Alignment, Font


HEADERS = [
    "Model",
    "Validation target-edge\nproperty mean R\u00b2",
    "Training time (s)\nmean \u00b1 SD",
    "Peak GPU memory (MB)\nmean \u00b1 SD",
    "Parameter count",
    "Folds",
    "Theoretical complexity",
]


def _copy_style(source, target) -> None:
    if source.has_style:
        target._style = copy(source._style)
    if source.alignment:
        target.alignment = copy(source.alignment)
    if source.number_format:
        target.number_format = source.number_format


def _display(value: object) -> object:
    """Repair legacy ASCII fallback only; do not alter numeric content."""
    if isinstance(value, str):
        return (
            value.replace(" ? ", " \u00b1 ")
            .replace(" \ufffd ", " \u00b1 ")
            .replace("R?", "R\u00b2")
            .replace("R\ufffd", "R\u00b2")
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workbook",
        default="outputs/0819final/ALL_EXPERIMENT_TABLES_0910_Final.xlsx",
    )
    args = parser.parse_args()
    workbook = Path(args.workbook)

    book = openpyxl.load_workbook(workbook)
    if "Computational_Eff" not in book.sheetnames or "Zero_Shot" not in book.sheetnames:
        raise KeyError("expected Computational_Eff and Zero_Shot worksheets")
    sheet = book["Computational_Eff"]
    reference = book["Zero_Shot"]

    # The source table has data at rows 5 onward.  Reading it first keeps this
    # idempotent even when the script is run more than once.
    rows = []
    for row in range(1, sheet.max_row + 1):
        model = sheet.cell(row, 1).value
        if model in {"Model", None}:
            continue
        values = [sheet.cell(row, column).value for column in range(1, 8)]
        if isinstance(values[0], str) and values[0] in {
            "Computational efficiency from existing completed-run metadata",
            "Computational efficiency",
        }:
            continue
        if all(value is None for value in values[1:]):
            continue
        rows.append([_display(value) for value in values])
    if not rows:
        raise RuntimeError("no computational-efficiency data rows found")

    # Rebuild solely this sheet. It is a presentation-only table with the same
    # two-level header vocabulary used by the other result sheets.
    for merged in list(sheet.merged_cells.ranges):
        sheet.unmerge_cells(str(merged))
    sheet.delete_rows(1, sheet.max_row)

    for column, header in enumerate(HEADERS, start=1):
        target = sheet.cell(1, column, header)
        _copy_style(reference.cell(1, 1 if column == 1 else 2), target)
        target.font = copy(reference.cell(1, 1 if column == 1 else 2).font)
        target.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        sheet.merge_cells(start_row=1, start_column=column, end_row=2, end_column=column)

    sheet.row_dimensions[1].height = 28
    sheet.row_dimensions[2].height = 6
    for row_index, values in enumerate(rows, start=3):
        for column, value in enumerate(values, start=1):
            target = sheet.cell(row_index, column, value)
            _copy_style(reference.cell(3, 1 if column == 1 else 2), target)
            target.alignment = Alignment(
                horizontal="left" if column in {1, 7} else "center",
                vertical="center",
                wrap_text=column == 7,
            )
        sheet.row_dimensions[row_index].height = 32 if len(str(values[6])) > 48 else 22

    note_row = len(rows) + 4
    sheet.merge_cells(start_row=note_row, start_column=1, end_row=note_row, end_column=7)
    note = sheet.cell(
        note_row,
        1,
        "Note. All values are 5-fold mean \u00b1 sample SD. Training time and peak allocated GPU memory are recorded completed-run metadata, not a newly controlled benchmark.",
    )
    note.font = Font(name="Arial", size=9, italic=True, color="666666")
    note.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
    sheet.row_dimensions[note_row].height = 30

    widths = {"A": 22, "B": 28, "C": 24, "D": 29, "E": 20, "F": 10, "G": 68}
    for column, width in widths.items():
        sheet.column_dimensions[column].width = width
    sheet.freeze_panes = "A3"
    sheet.sheet_view.showGridLines = False

    temporary = workbook.with_name(f".{workbook.name}.computational_eff.tmp")
    book.save(temporary)
    os.replace(temporary, workbook)
    print(f"formatted {workbook}: {len(rows)} computational-efficiency rows")


if __name__ == "__main__":
    main()
