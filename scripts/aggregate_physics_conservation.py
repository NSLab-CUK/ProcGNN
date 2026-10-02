"""Aggregate fold-level held-out conservation reports into a paper-ready table."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


METRICS = (
    "normalized_abs_residual_mae",
    "normalized_residual_rmse",
    "satisfaction_rate_abs_le_0.01",
    "satisfaction_rate_abs_le_0.05",
    "satisfaction_rate_abs_le_0.10",
)


def _summary(values: list[float]) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan, math.nan
    return mean(finite), stdev(finite) if len(finite) > 1 else 0.0


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}e}" if math.isfinite(value) else "NA"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate held-out physics-conservation fold reports.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.input_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("physics_conservation_metrics.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row["source"] = str(path)
                rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No physics_conservation_metrics.csv found below {root}")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["term"])].append(row)
    summary_rows: list[dict[str, object]] = []
    for (model, term), group in sorted(grouped.items()):
        output_row: dict[str, object] = {"model": model, "term": term, "folds": len(group)}
        for metric in METRICS:
            values = [float(row.get(metric, "nan")) for row in group]
            avg, std = _summary(values)
            output_row[f"{metric}_mean"] = avg
            output_row[f"{metric}_std"] = std
        output_row["valid_residual_count_total"] = sum(
            int(float(row.get("valid_residual_count", "0"))) for row in group
        )
        summary_rows.append(output_row)

    fields = list(summary_rows[0])
    with (output / "physics_conservation_5fold_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    lines = [
        "# Held-out physics conservation",
        "",
        "Each entry is fold mean ± sample standard deviation. Lower MAE/RMSE is better; higher satisfaction is better.",
        "",
        "| Model | Term | NMAE | NRMSE | abs(epsilon) <= 1% | abs(epsilon) <= 5% | abs(epsilon) <= 10% | Valid residuals |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {model} | {term} | {mae} ± {mae_std} | {rmse} ± {rmse_std} | {p01:.2%} ± {p01std:.2%} | {p05:.2%} ± {p05std:.2%} | {p10:.2%} ± {p10std:.2%} | {count} |".format(
                model=row["model"],
                term=row["term"],
                mae=_fmt(float(row["normalized_abs_residual_mae_mean"])),
                mae_std=_fmt(float(row["normalized_abs_residual_mae_std"])),
                rmse=_fmt(float(row["normalized_residual_rmse_mean"])),
                rmse_std=_fmt(float(row["normalized_residual_rmse_std"])),
                p01=float(row["satisfaction_rate_abs_le_0.01_mean"]),
                p01std=float(row["satisfaction_rate_abs_le_0.01_std"]),
                p05=float(row["satisfaction_rate_abs_le_0.05_mean"]),
                p05std=float(row["satisfaction_rate_abs_le_0.05_std"]),
                p10=float(row["satisfaction_rate_abs_le_0.10_mean"]),
                p10std=float(row["satisfaction_rate_abs_le_0.10_std"]),
                count=int(row["valid_residual_count_total"]),
            )
        )
    (output / "physics_conservation_5fold_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {output / 'physics_conservation_5fold_summary.md'}")


if __name__ == "__main__":
    main()
