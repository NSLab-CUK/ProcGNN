"""Aggregate target-edge sMAPE (%) from completed Final3-era evaluations.

Baseline runs already write physical-scale ``test_predictions.csv`` files and
are therefore recomputed directly.  Proposed runs are incorporated when
``evaluate_target_smape.py`` has written ``target_property_smape.csv`` next to
their checkpoint evaluation output.  The output keeps every fold/process/run
in a long audit table and also writes paper-ready mean +/- sample-SD summaries.
"""

from __future__ import annotations

import argparse
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EPS = 1.0e-8
MODEL_NAMES = {
    "M1": "SVR",
    "M2": "Random Forest",
    "M3": "XGBoost",
    "B1": "Kriging-KR31",
    "B2": "Cubic RBF",
    "B3": "GTL-ANN",
    "B4": "Cumene-Efficiency-ANN",
    "B5": "Cumene-Destruction-ANN",
    "B6": "Reusable-Distillation-ANN",
    "B7": "Distillation-Boundary-GP",
    "G1": "GCN",
    "G2": "GIN",
    "G3": "GAT",
}


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _part_after(parts: tuple[str, ...], marker: str) -> str | None:
    lowered = [part.lower() for part in parts]
    try:
        return parts[lowered.index(marker.lower()) + 1]
    except (ValueError, IndexError):
        return None


def _find_match(parts: Iterable[str], pattern: str) -> str | None:
    regex = re.compile(pattern, re.IGNORECASE)
    for part in parts:
        match = regex.fullmatch(part)
        if match:
            return match.group(1)
    return None


def _infer_baseline_metadata(path: Path) -> dict[str, str]:
    parts = path.parts
    text = "/".join(parts).lower()
    if "baselines_transfer_0909" in text:
        family = "Transfer data efficiency"
    elif "baselines_zero_shot_0910" in text:
        family = "Zero-shot"
    elif "baselines_topology_sfiles_0908" in text:
        family = "Single process" if any(part.lower() == "single" for part in parts) else "Multi process"
    else:
        family = "Single process" if any(part.lower() == "single" for part in parts) else "Multi process"
    model = _part_after(parts, "single") or _part_after(parts, "multi")
    if model is None:
        # Joint runs are generally <root>/<model>/joint/fold_NN/.
        try:
            model = parts[[part.lower() for part in parts].index("joint") - 1]
        except ValueError:
            # Zero-shot/transfer baselines use <root>/<model>/heldout_PNN.
            model = None
            for root_name in ("baselines_zero_shot_0910", "baselines_transfer_0909"):
                try:
                    model = parts[[part.lower() for part in parts].index(root_name) + 1]
                    break
                except (ValueError, IndexError):
                    continue
            if model is None:
                model = "Unknown"
    return {
        "family": family,
        "model_code": str(model),
        "model": MODEL_NAMES.get(str(model), str(model)),
        "process": _find_match(parts, r"Process(\d+)") or "",
        "heldout_process": _find_match(parts, r"heldout_P(\d+)") or "",
        "fold": _find_match(parts, r"fold_(\d+)") or "",
        "transfer_ratio": _find_match(parts, r"ratio_(\d+)") or "",
        "experiment_setting": "",
        "run_label": "",
        "source_kind": "baseline_prediction_csv",
    }


def _read_one_baseline(path: Path) -> tuple[list[dict[str, object]], dict[str, str] | None]:
    try:
        frame = pd.read_csv(path, usecols=["property_name", "true_value", "predicted_value"])
        frame["true_value"] = pd.to_numeric(frame["true_value"], errors="coerce")
        frame["predicted_value"] = pd.to_numeric(frame["predicted_value"], errors="coerce")
        meta = _infer_baseline_metadata(path)
        rows: list[dict[str, object]] = []
        for property_name, group in frame.groupby("property_name", sort=False):
            group = group.loc[group["true_value"].map(math.isfinite) & group["predicted_value"].map(math.isfinite)]
            if group.empty:
                continue
            true = group["true_value"].abs()
            pred = group["predicted_value"].abs()
            smape = (2.0 * (group["predicted_value"] - group["true_value"]).abs() / (true + pred + EPS)).mean() * 100.0
            rows.append({
                **meta,
                "property_name": str(property_name),
                "n": int(len(group)),
                "sMAPE_pct": float(smape),
                "source_path": str(path),
            })
        return rows, None
    except Exception as exc:  # Preserve a complete audit rather than silently dropping a run.
        return [], {"source_path": str(path), "error": f"{type(exc).__name__}: {exc}"}


def _baseline_rows(paths: list[Path], *, workers: int) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        for index, (file_rows, failure) in enumerate(executor.map(_read_one_baseline, paths), start=1):
            rows.extend(file_rows)
            if failure is not None:
                failures.append(failure)
            if index == 1 or index % 100 == 0 or index == len(paths):
                print(f"[baseline sMAPE] {index}/{len(paths)} prediction files", flush=True)
    return rows, failures


def _proposed_rows(paths: list[Path]) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
            required = {"property_name", "sMAPE_pct"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"missing columns: {sorted(missing)}")
            parts = path.parts
            text = "/".join(parts).lower()
            if "data_efficiency" in text:
                family = "Transfer data efficiency"
            elif "proposed_unseen" in text:
                family = "Zero-shot"
            elif "sensitivity_10d_clean" in text:
                family = "Sensitivity/Ablation"
            elif "proposed_single_process" in text:
                family = "Single process"
            else:
                family = "Multi process"
            for row in frame.itertuples(index=False):
                value = float(getattr(row, "sMAPE_pct"))
                if not math.isfinite(value):
                    continue
                rows.append({
                    "family": family,
                    "model_code": "Proposed",
                    "model": "Proposed",
                    "process": _find_match(parts, r"Process(\d+)") or "",
                    "heldout_process": _find_match(parts, r"heldout_P(\d+)") or "",
                    "fold": _find_match(parts, r"fold_(\d+)") or "",
                    "transfer_ratio": _find_match(parts, r"ratio_(\d+)") or "",
                    "experiment_setting": (
                        "/".join(parts[parts.index("sensitivity_10d_clean") + 1 : -1])
                        if "sensitivity_10d_clean" in parts else ""
                    ),
                    "run_label": str(getattr(row, "model", "Proposed")),
                    "property_name": str(getattr(row, "property_name")),
                    "n": int(getattr(row, "n", 0)),
                    "sMAPE_pct": value,
                    "source_kind": "proposed_posthoc_evaluation",
                    "source_path": str(path),
                })
        except Exception as exc:
            failures.append({"source_path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return rows, failures


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "family", "model_code", "model", "process", "heldout_process",
        "transfer_ratio", "experiment_setting", "property_name", "source_kind",
    ]
    return (
        frame.groupby(group_cols, dropna=False, sort=True)
        .agg(
            completed_runs=("sMAPE_pct", "count"),
            total_n=("n", "sum"),
            sMAPE_mean_pct=("sMAPE_pct", "mean"),
            sMAPE_std_pct=("sMAPE_pct", "std"),
        )
        .reset_index()
    )


def _format_summary(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["sMAPE_pct_mean_sd"] = output.apply(
        lambda row: (
            f"{float(row.sMAPE_mean_pct):.4f} ± "
            f"{0.0 if pd.isna(row.sMAPE_std_pct) else float(row.sMAPE_std_pct):.4f}"
        ),
        axis=1,
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate existing target-edge sMAPE (%) artifacts.")
    parser.add_argument("--final-root", default="outputs/0819final")
    parser.add_argument("--proposed-smape-root", default="outputs/0819final/posthoc_target_smape")
    parser.add_argument("--output-root", default="outputs/0819final/Final3_percentage_metrics")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent CSV readers for remote output storage.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    final_root = _resolve(args.final_root)
    proposed_root = _resolve(args.proposed_smape_root)
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_roots = [
        final_root / "baselines",
        final_root / "baselines_topology_sfiles_0908",
        final_root / "baselines_transfer_0909",
        final_root / "baselines_zero_shot_0910",
    ]
    baseline_paths = sorted({path for root in baseline_roots if root.is_dir() for path in root.rglob("test_predictions.csv")})
    proposed_paths = sorted(proposed_root.rglob("target_property_smape.csv")) if proposed_root.is_dir() else []
    print(f"[discover] baseline prediction files={len(baseline_paths)} proposed sMAPE files={len(proposed_paths)}", flush=True)

    baseline, baseline_failures = _baseline_rows(baseline_paths, workers=args.workers)
    proposed, proposed_failures = _proposed_rows(proposed_paths)
    detail = pd.DataFrame([*baseline, *proposed])
    if detail.empty:
        raise RuntimeError("No usable test prediction or proposed sMAPE files were found.")
    detail = detail.sort_values(["family", "model", "process", "heldout_process", "transfer_ratio", "fold", "property_name"])
    summary = _summary(detail)
    formatted = _format_summary(summary)
    detail.to_csv(output_root / "target_smape_detail.csv", index=False)
    summary.to_csv(output_root / "target_smape_summary.csv", index=False)
    formatted.to_csv(output_root / "target_smape_summary_formatted.csv", index=False)
    pd.DataFrame([*baseline_failures, *proposed_failures]).to_csv(output_root / "read_failures.csv", index=False)
    with pd.ExcelWriter(output_root / "Final3_target_sMAPE.xlsx", engine="openpyxl") as writer:
        formatted.to_excel(writer, sheet_name="Target_sMAPE_summary", index=False)
        detail.to_excel(writer, sheet_name="Target_sMAPE_detail", index=False)
        pd.DataFrame([*baseline_failures, *proposed_failures]).to_excel(writer, sheet_name="Read_failures", index=False)
    (output_root / "README.md").write_text(
        "# Final3 target-edge sMAPE (%)\n\n"
        "Formula: `100 * mean(2*abs(y_pred-y_true)/(abs(y_pred)+abs(y_true)+1e-8))`. "
        "Lower is better; values are percentages (0--200).\n\n"
        "Baseline rows are recomputed from saved `test_predictions.csv`. Proposed rows appear after "
        "`scripts/evaluate_target_smape.py` creates per-checkpoint target artifacts under the selected proposed root.\n",
        encoding="utf-8",
    )
    print(f"[done] detail_rows={len(detail)} summary_rows={len(summary)} output={output_root}", flush=True)


if __name__ == "__main__":
    main()
