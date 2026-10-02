from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UNSEEN_ROOT = ROOT / "outputs" / "260805_singleproc_unseen"
ALLPROC_ROOT = ROOT / "outputs" / "260805_allproc5fold" / "All"

PROPERTY_ORDER = [
    "Target Frac_CH4",
    "Target Frac_CO",
    "Target Frac_CO2",
    "Target Frac_H2",
    "Target Frac_H2O",
    "Target Frac_N2",
    "Target Frac_O2",
    "Target Mass_Flow",
    "Target Pres",
    "Target Temp",
    "All-edge Frac_H2O",
    "All-edge Frac_H2",
    "All-edge Frac_CH4",
    "All-edge Frac_CO2",
    "All-edge Frac_CO",
    "All-edge Frac_O2",
    "All-edge Frac_N2",
    "All-edge Mass_Flow",
    "All-edge Pres",
    "All-edge Temp",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def label(scope: str, prop: str) -> str:
    return f"{scope} {prop}"


def summarize_unseen() -> Path:
    source = UNSEEN_ROOT / "process_property_mean_std_transfer.csv"
    lookup: dict[tuple[str, str], tuple[float, float, int]] = {}
    for row in read_csv(source):
        key = (str(row["Process"]), label(row["Scope"], row["Property"]))
        lookup[key] = (float(row["Mean"]), float(row["Std"]), int(row["Folds"]))

    output_rows: list[dict[str, object]] = []
    for prop in PROPERTY_ORDER:
        out: dict[str, object] = {"Property": prop}
        for process in range(1, 11):
            mean, std, folds = lookup[(str(process), prop)]
            if folds != 5:
                raise RuntimeError(f"P{process:02d} {prop}: expected 5 folds, found {folds}")
            out[f"P{process:02d}"] = f"{mean:.4f} ± {std:.4f}"
        output_rows.append(out)

    destination = UNSEEN_ROOT / "property_r2_summary_transfer_20.csv"
    write_csv(destination, output_rows, ["Property", *[f"P{i:02d}" for i in range(1, 11)]])
    return destination


def latest_complete_run(fold_dir: Path) -> Path:
    candidates = [
        path
        for path in fold_dir.iterdir()
        if path.is_dir()
        and (path / "metrics.json").is_file()
        and (path / "target_edge_r2_by_property.csv").is_file()
        and (path / "pi_all_edge_property_r2.csv").is_file()
    ]
    if not candidates:
        raise RuntimeError(f"No complete metric run found under {fold_dir}")
    return max(candidates, key=lambda path: (path / "metrics.json").stat().st_mtime)


def summarize_zero_shot() -> Path:
    source = UNSEEN_ROOT / "aggregate_property_r2_5fold_mean_std_by_process.csv"
    lookup: dict[tuple[str, str], tuple[str, str, int]] = {}
    for row in read_csv(source):
        if row["mode"] != "Zero-shot":
            continue
        key = (row["process"], label(row["scope"], row["property"]))
        lookup[key] = (row["mean"], row["std"], int(row["valid_folds"]))

    output_rows: list[dict[str, object]] = []
    for prop in PROPERTY_ORDER:
        out: dict[str, object] = {"Property": prop}
        for process in range(1, 11):
            process_name = f"P{process:02d}"
            mean_text, std_text, folds = lookup[(process_name, prop)]
            if folds == 0:
                out[process_name] = "N/A (0/5)"
            elif folds == 5:
                out[process_name] = f"{float(mean_text):.4f} \u00b1 {float(std_text):.4f}"
            else:
                out[process_name] = (
                    f"{float(mean_text):.4f} \u00b1 {float(std_text):.4f} ({folds}/5)"
                )
        output_rows.append(out)

    destination = UNSEEN_ROOT / "property_r2_summary_zero_shot_20.csv"
    write_csv(destination, output_rows, ["Property", *[f"P{i:02d}" for i in range(1, 11)]])
    return destination


def summarize_allproc() -> Path:
    values: dict[str, list[float]] = {prop: [] for prop in PROPERTY_ORDER}
    provenance: list[dict[str, object]] = []
    for fold in range(1, 6):
        run = latest_complete_run(ALLPROC_ROOT / f"fold_{fold:02d}")
        provenance.append({"Fold": fold, "Run": run.relative_to(ROOT).as_posix()})

        for row in read_csv(run / "target_edge_r2_by_property.csv"):
            prop = label("Target", row["property_name"])
            if prop in values:
                values[prop].append(float(row["R2"]))
        for row in read_csv(run / "pi_all_edge_property_r2.csv"):
            prop = label("All-edge", row["property_name"])
            if prop in values:
                values[prop].append(float(row["R2"]))

    output_rows: list[dict[str, object]] = []
    for prop in PROPERTY_ORDER:
        observed = values[prop]
        if len(observed) != 5 or not all(math.isfinite(value) for value in observed):
            raise RuntimeError(f"{prop}: expected 5 finite values, found {observed}")
        mean = statistics.fmean(observed)
        std = statistics.stdev(observed)
        output_rows.append(
            {
                "Property": prop,
                "Mean": mean,
                "Std": std,
                "Folds": len(observed),
                "R2_mean_std": f"{mean:.4f} ± {std:.4f}",
            }
        )

    destination = ALLPROC_ROOT.parent / "property_r2_summary_20.csv"
    write_csv(destination, output_rows, ["Property", "Mean", "Std", "Folds", "R2_mean_std"])
    write_csv(
        ALLPROC_ROOT.parent / "property_r2_summary_20_provenance.csv",
        provenance,
        ["Fold", "Run"],
    )
    return destination


if __name__ == "__main__":
    print(summarize_unseen())
    print(summarize_zero_shot())
    print(summarize_allproc())
