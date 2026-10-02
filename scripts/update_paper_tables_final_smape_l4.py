"""Add target-edge sMAPE columns to Paper_Tables_Final.xlsx.

Baseline sMAPE values are measured from saved test predictions.  Proposed
single-process layer-4 values are estimated by applying the property-wise
ratio observed between the joint depth-4 sensitivity run and the joint
layer-5 run.  Proposed multi-process layer-4 values use the matching depth-4
sensitivity outputs directly.  Estimated cells are marked with a dagger.
"""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill


PROPERTIES = [
    "Temp",
    "Pres",
    "Frac_H2O",
    "Frac_H2",
    "Frac_CH4",
    "Frac_CO2",
    "Frac_CO",
    "Frac_O2",
    "Frac_N2",
    "Mass_Flow",
]

BASELINES = [
    ("baselines", "M1", "M1", "SVR"),
    ("baselines", "M2", "M2", "Random Forest"),
    ("baselines", "M3", "M3", "XGBoost"),
    ("baselines", "B1", "B1", "Kriging-KR31"),
    ("baselines", "B2", "B2", "Cubic RBF"),
    ("baselines", "B3", "B3", "GTL-ANN"),
    ("baselines", "B4", "B4", "Cumene-Efficiency-ANN"),
    ("baselines", "B5", "B5", "Cumene-Destruction-ANN"),
    ("baselines", "B6", "B6", "Reusable-Distillation-ANN"),
    ("baselines", "B7", "B7", "Distillation-Boundary-GP"),
    ("baselines_topology_sfiles_0908", "G1", "GCN", "GCN"),
    ("baselines_topology_sfiles_0908", "G2", "GIN", "GIN"),
    ("baselines_topology_sfiles_0908", "G3", "GAT", "GAT"),
    ("baselines_topology_sfiles_0908", "GraphToSFILES", "GraphToSFILES", "GraphToSFILES"),
    ("baselines_transformer_0908", "Graphormer", "Graphormer", "Graphormer"),
    ("baselines_transformer_0908", "SAT", "SAT", "SAT"),
]


@dataclass(frozen=True)
class PredictionJob:
    model: str
    scope: str
    fold: int
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", required=True)
    parser.add_argument("--server-root", action="append", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--source-summary", required=True)
    return parser.parse_args()


def _prediction_jobs(server_roots: list[Path]) -> list[PredictionJob]:
    jobs: list[PredictionJob] = []
    for family, single_key, multi_key, model in BASELINES:
        for root in server_roots:
            final_root = root / "outputs" / "0819final"
            single = final_root / family / "single" / single_key
            for path in single.glob("Process*/fold_*/test_predictions.csv"):
                fold = int(path.parent.name.split("_")[-1])
                jobs.append(PredictionJob(model, "single", fold, path))
            multi = final_root / family / "multi" / multi_key
            for path in multi.glob("fold_*/Process*/test_predictions.csv"):
                fold = int(path.parents[1].name.split("_")[-1])
                jobs.append(PredictionJob(model, "multi", fold, path))
    return jobs


def _read_prediction(job: PredictionJob) -> tuple[PredictionJob, dict[str, tuple[float, int]]]:
    frame = pd.read_csv(
        job.path,
        usecols=["property_name", "true_value", "predicted_value"],
    )
    true = pd.to_numeric(frame["true_value"], errors="coerce").to_numpy(float)
    pred = pd.to_numeric(frame["predicted_value"], errors="coerce").to_numpy(float)
    prop = frame["property_name"].astype(str).to_numpy()
    valid = np.isfinite(true) & np.isfinite(pred)
    values = 100.0 * 2.0 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1.0e-8)
    result: dict[str, tuple[float, int]] = {}
    for name in PROPERTIES:
        mask = valid & (prop == name)
        result[name] = (float(values[mask].sum()), int(mask.sum()))
    return job, result


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def _baseline_stats(
    jobs: list[PredictionJob], workers: int
) -> tuple[dict[str, dict[str, dict[str, tuple[float, float]]]], list[dict[str, object]]]:
    completed: list[tuple[PredictionJob, dict[str, tuple[float, int]]]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for index, result in enumerate(pool.map(_read_prediction, jobs), start=1):
            completed.append(result)
            if index % 100 == 0 or index == len(jobs):
                print(f"[baseline sMAPE] {index}/{len(jobs)}", flush=True)

    run_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    multi_fold_sums: dict[tuple[str, int, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    audit: list[dict[str, object]] = []
    for job, result in completed:
        if job.scope == "single":
            macro: list[float] = []
            for prop, (total, count) in result.items():
                value = total / count if count else math.nan
                if math.isfinite(value):
                    run_values[(job.scope, job.model, prop)].append(value)
                    macro.append(value)
            if len(macro) == len(PROPERTIES):
                run_values[(job.scope, job.model, "Average")].append(float(np.mean(macro)))
        else:
            for prop, (total, count) in result.items():
                target = multi_fold_sums[(job.model, job.fold, prop)]
                target[0] += total
                target[1] += count
        audit.append(
            {
                "scope": job.scope,
                "model": job.model,
                "fold": job.fold,
                "path": str(job.path),
            }
        )

    for (model, fold, prop), (total, count) in multi_fold_sums.items():
        if count:
            run_values[("multi", model, prop)].append(total / count)
    multi_models = {key[1] for key in run_values if key[0] == "multi"}
    for model in multi_models:
        per_property = {
            prop: run_values[("multi", model, prop)] for prop in PROPERTIES
        }
        folds = len(next(iter(per_property.values()), []))
        if folds and all(len(values) == folds for values in per_property.values()):
            for index in range(folds):
                run_values[("multi", model, "Average")].append(
                    float(np.mean([per_property[prop][index] for prop in PROPERTIES]))
                )

    stats: dict[str, dict[str, dict[str, tuple[float, float]]]] = defaultdict(dict)
    for (scope, model, prop), values in run_values.items():
        stats[scope].setdefault(model, {})[prop] = _mean_std(values)
    return dict(stats), audit


def _posthoc_rows(paths: list[Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["property_name", "sMAPE_pct"])
        frame["run"] = str(path)
        rows.append(frame)
    if not rows:
        return pd.DataFrame(columns=["property_name", "sMAPE_pct", "run"])
    return pd.concat(rows, ignore_index=True)


def _proposed_stats(server_roots: list[Path]) -> tuple[
    dict[str, dict[str, tuple[float, float]]], pd.DataFrame
]:
    single_paths: list[Path] = []
    joint_paths: list[Path] = []
    depth4_paths: list[Path] = []
    for root in server_roots:
        base = root / "outputs" / "0819final" / "posthoc_target_smape"
        single_paths.extend(base.glob("proposed_single_process/Process*/fold_*/target_property_smape.csv"))
        joint_paths.extend(base.glob("proposed_joint_10d_clean/All/fold_*/target_property_smape.csv"))
        depth4_paths.extend(base.glob("sensitivity_10d_clean/depth/depth_4/All/fold_*/target_property_smape.csv"))
    single = _posthoc_rows(single_paths)
    joint = _posthoc_rows(joint_paths)
    depth4 = _posthoc_rows(depth4_paths)
    if single["run"].nunique() != 50 or joint["run"].nunique() != 5 or depth4["run"].nunique() != 5:
        raise RuntimeError(
            "Expected 50 single, 5 joint, and 5 depth-4 Proposed runs; got "
            f"{single['run'].nunique()}, {joint['run'].nunique()}, {depth4['run'].nunique()}."
        )

    joint_mean = joint.groupby("property_name")["sMAPE_pct"].mean()
    depth4_mean = depth4.groupby("property_name")["sMAPE_pct"].mean()
    factor = depth4_mean / joint_mean
    single["estimated_L4_sMAPE_pct"] = single.apply(
        lambda row: float(row["sMAPE_pct"] * factor[row["property_name"]]), axis=1
    )
    stats: dict[str, dict[str, tuple[float, float]]] = {"single": {}, "multi": {}}
    for prop in PROPERTIES:
        stats["single"][prop] = _mean_std(
            single.loc[single["property_name"] == prop, "estimated_L4_sMAPE_pct"].tolist()
        )
        stats["multi"][prop] = _mean_std(
            depth4.loc[depth4["property_name"] == prop, "sMAPE_pct"].tolist()
        )
    single_macro = single.pivot(index="run", columns="property_name", values="estimated_L4_sMAPE_pct")
    depth4_macro = depth4.pivot(index="run", columns="property_name", values="sMAPE_pct")
    stats["single"]["Average"] = _mean_std(single_macro[PROPERTIES].mean(axis=1).tolist())
    stats["multi"]["Average"] = _mean_std(depth4_macro[PROPERTIES].mean(axis=1).tolist())

    provenance = pd.DataFrame({
        "property_name": PROPERTIES,
        "joint_L5_sMAPE_mean_pct": [joint_mean[prop] for prop in PROPERTIES],
        "depth4_sMAPE_mean_pct": [depth4_mean[prop] for prop in PROPERTIES],
        "L4_over_L5_factor": [factor[prop] for prop in PROPERTIES],
        "single_L5_sMAPE_mean_pct": [single.loc[single.property_name == prop, "sMAPE_pct"].mean() for prop in PROPERTIES],
        "estimated_single_L4_sMAPE_mean_pct": [stats["single"][prop][0] for prop in PROPERTIES],
        "estimated_multi_L4_sMAPE_mean_pct": [stats["multi"][prop][0] for prop in PROPERTIES],
    })
    return stats, provenance


def _copy_cell_style(source: openpyxl.cell.Cell, target: openpyxl.cell.Cell) -> None:
    if source.has_style:
        target._style = copy(source._style)
    target.number_format = source.number_format
    target.font = copy(source.font)
    target.fill = copy(source.fill)
    target.border = copy(source.border)
    target.alignment = copy(source.alignment)
    target.protection = copy(source.protection)


def _augment_metric_sheet(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    *,
    metric_start: int,
    footer_row: int,
) -> list[int]:
    group_count = 11
    if any(ws.cell(2, col).value == "sMAPE (%)" for col in range(1, ws.max_column + 1)):
        return [metric_start + 3 * index + 2 for index in range(group_count)]
    headers = [ws.cell(1, metric_start + 2 * index).value for index in range(group_count)]
    for merged in list(ws.merged_cells.ranges):
        ws.unmerge_cells(str(merged))
    for index in reversed(range(group_count)):
        ws.insert_cols(metric_start + 2 * index + 2, 1)
    smape_columns: list[int] = []
    for index, header in enumerate(headers):
        start = metric_start + 3 * index
        smape_col = start + 2
        smape_columns.append(smape_col)
        for row in range(1, ws.max_row + 1):
            _copy_cell_style(ws.cell(row, smape_col - 1), ws.cell(row, smape_col))
        ws.cell(1, start).value = header
        ws.cell(2, start).value = "MAE"
        ws.cell(2, start + 1).value = "NMAE"
        ws.cell(2, smape_col).value = "sMAPE (%)"
        ws.cell(2, smape_col).comment = Comment(
            "Target-edge sMAPE on the physical scale. Dagger-marked Proposed values are layer-4 estimates.",
            "Codex",
        )
        ws.merge_cells(start_row=1, start_column=start, end_row=1, end_column=smape_col)
        ws.column_dimensions[openpyxl.utils.get_column_letter(smape_col)].width = 18
    for col in range(1, metric_start):
        ws.merge_cells(start_row=1, start_column=col, end_row=2, end_column=col)
    ws.merge_cells(start_row=footer_row, start_column=1, end_row=footer_row, end_column=ws.max_column)
    return smape_columns


def _format_metric(value: tuple[float, float], estimated: bool = False) -> str:
    mean, std = value
    suffix = " †" if estimated else ""
    return f"{mean:.4f} ± {std:.4f}{suffix}"


def _fill_sheet(
    ws: openpyxl.worksheet.worksheet.Worksheet,
    *,
    metric_start: int,
    footer_row: int,
    values: dict[str, dict[str, tuple[float, float]]],
    estimated_models: set[str],
) -> None:
    columns = _augment_metric_sheet(ws, metric_start=metric_start, footer_row=footer_row)
    property_keys = PROPERTIES + ["Average"]
    for row in range(3, footer_row):
        model = ws.cell(row, 1).value
        if model not in values:
            continue
        for col, prop in zip(columns, property_keys):
            metric = values[model].get(prop)
            ws.cell(row, col).value = (
                _format_metric(metric, estimated=model in estimated_models) if metric else None
            )
    note = str(ws.cell(footer_row, 1).value or "").rstrip()
    addition = (
        " sMAPE is target-edge, physical-scale percentage error. "
        "† Proposed layer-4 sMAPE is estimated from saved layer-5 outputs using the property-wise "
        "depth-4/joint-layer-5 correction documented in sMAPE_L4_Estimation. Blank cells indicate no matching sMAPE artifact."
    )
    if "sMAPE is target-edge" not in note:
        ws.cell(footer_row, 1).value = note + addition
    ws.cell(footer_row, 1).alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[footer_row].height = 60


def _provenance_sheet(wb: openpyxl.Workbook, provenance: pd.DataFrame, audit: list[dict[str, object]]) -> None:
    title = "sMAPE_L4_Estimation"
    if title in wb.sheetnames:
        del wb[title]
    ws = wb.create_sheet(title)
    notes = [
        "Metric: target-edge sMAPE (%) = 100 * mean(2|prediction-truth|/(|prediction|+|truth|+1e-8)); lower is better.",
        "Baseline values are measured from saved test_predictions.csv files and are not layer-adjusted.",
        "Proposed multi-process layer-4 values use the matching depth_4 sensitivity outputs (5 folds).",
        "Proposed single-process layer-4 values are estimates: each layer-5 run is multiplied by the property-wise ratio mean(depth_4 joint)/mean(layer-5 joint).",
        "Dagger-marked values are estimates for comparison with the layer-4 MAE/NMAE table; they must not be described as directly measured layer-4 sMAPE.",
    ]
    for row, text in enumerate(notes, start=1):
        ws.cell(row, 1).value = text
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=7)
    header_row = len(notes) + 2
    for col, name in enumerate(provenance.columns, start=1):
        cell = ws.cell(header_row, col, name)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
    for row_index, record in enumerate(provenance.itertuples(index=False), start=header_row + 1):
        for col, value in enumerate(record, start=1):
            ws.cell(row_index, col, value)
            if col > 1:
                ws.cell(row_index, col).number_format = "0.0000"
    ws.cell(header_row + len(provenance) + 2, 1).value = "Baseline prediction files used"
    ws.cell(header_row + len(provenance) + 2, 2).value = len(audit)
    ws.freeze_panes = f"A{header_row + 1}"
    for col in range(1, 8):
        ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 29


def main() -> None:
    args = parse_args()
    workbook_path = Path(args.workbook).resolve()
    server_roots = [Path(value).resolve() for value in args.server_root]
    jobs = _prediction_jobs(server_roots)
    baseline_stats, audit = _baseline_stats(jobs, args.workers)
    proposed_stats, provenance = _proposed_stats(server_roots)

    source = pd.DataFrame(audit)
    source.to_csv(args.source_summary, index=False, encoding="utf-8-sig")
    print(f"[sources] baseline_prediction_files={len(source)} -> {args.source_summary}", flush=True)

    workbook = openpyxl.load_workbook(workbook_path)
    single_values = dict(baseline_stats.get("single", {}))
    single_values["Proposed"] = proposed_stats["single"]
    multi_values = dict(baseline_stats.get("multi", {}))
    multi_values["Proposed"] = proposed_stats["multi"]
    _fill_sheet(
        workbook["singleprocess_10D"], metric_start=2, footer_row=22,
        values=single_values, estimated_models={"Proposed"},
    )
    _fill_sheet(
        workbook["multiprocess_10D"], metric_start=2, footer_row=13,
        values=multi_values, estimated_models={"Proposed"},
    )
    _fill_sheet(
        workbook["ablation_10D"], metric_start=6, footer_row=11,
        values={"Full Model": proposed_stats["multi"]}, estimated_models={"Full Model"},
    )
    _fill_sheet(
        workbook["pinn_ablation_10D"], metric_start=5, footer_row=12,
        values={"Full PINN": proposed_stats["multi"]}, estimated_models={"Full PINN"},
    )
    _provenance_sheet(workbook, provenance, audit)
    temporary = workbook_path.with_name(workbook_path.stem + ".tmp.xlsx")
    workbook.save(temporary)
    os.replace(temporary, workbook_path)
    print(f"[saved] {workbook_path}", flush=True)


if __name__ == "__main__":
    main()
