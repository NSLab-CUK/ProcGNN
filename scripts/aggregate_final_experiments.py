from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROPERTIES = (
    "temp", "pres", "frac_h2o", "frac_h2", "frac_ch4",
    "frac_co2", "frac_co", "frac_o2", "frac_n2", "mass_flow",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _latest(root: Path, name: str) -> Path | None:
    paths = list(root.rglob(name)) if root.exists() else []
    return max(paths, key=lambda path: path.stat().st_mtime) if paths else None


def _metric_row(root: Path, **fields: Any) -> dict[str, Any] | None:
    path = _latest(root, "metrics.json")
    if path is None:
        return None
    payload = _json(path)
    row = {
        **fields,
        "target_mean_r2": payload.get("test_target_mean_r2", payload.get("target_mean_r2", np.nan)),
        "metrics_path": str(path),
    }
    for prop in PROPERTIES:
        row[f"target_r2_{prop}"] = payload.get(
            f"test_target_edge_r2_{prop}", payload.get(f"target_edge_r2_{prop}", np.nan)
        )
    return row


def _fold_rows(root: Path, **fields: Any) -> list[dict[str, Any]]:
    rows = []
    for fold in range(1, 6):
        candidates = [
            path for path in root.rglob("metrics.json")
            if f"fold_{fold:02d}" in {part.lower() for part in path.parts}
        ]
        if not candidates:
            continue
        path = max(candidates, key=lambda item: item.stat().st_mtime)
        payload = _json(path)
        row = {
            **fields, "fold": fold,
            "target_mean_r2": payload.get("test_target_mean_r2", payload.get("target_mean_r2", np.nan)),
            "metrics_path": str(path),
        }
        for prop in PROPERTIES:
            row[f"target_r2_{prop}"] = payload.get(
                f"test_target_edge_r2_{prop}", payload.get(f"target_edge_r2_{prop}", np.nan)
            )
        rows.append(row)
    return rows


def _process_fold_rows(root: Path, **fields: Any) -> list[dict[str, Any]]:
    rows = []
    for process_dir in sorted(root.glob("Process*")):
        try:
            process_id = int(process_dir.name.replace("Process", ""))
        except ValueError:
            continue
        for fold in range(1, 6):
            candidates = list((process_dir / f"fold_{fold:02d}").rglob("metrics.json"))
            if not candidates:
                continue
            path = max(candidates, key=lambda item: item.stat().st_mtime)
            payload = _json(path)
            row = {
                **fields,
                "process_id": process_id,
                "fold": fold,
                "target_mean_r2": payload.get(
                    "test_target_edge_property_mean_r2",
                    payload.get("target_edge_property_mean_r2", np.nan),
                ),
                "metrics_path": str(path),
            }
            for prop in PROPERTIES:
                row[f"target_r2_{prop}"] = payload.get(
                    f"test_target_edge_r2_{prop}", payload.get(f"target_edge_r2_{prop}", np.nan)
                )
            rows.append(row)
    return rows


def _summarize(frame: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.groupby(group, dropna=False).agg(
        target_mean_r2_mean=("target_mean_r2", "mean"),
        target_mean_r2_std=("target_mean_r2", "std"),
        runs=("target_mean_r2", "count"),
    ).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate every final-paper experiment family")
    parser.add_argument("--root", default="outputs/final_paper")
    parser.add_argument(
        "--baseline-root",
        default=None,
        help="Baseline artifact root; defaults to <root>/baselines.",
    )
    args = parser.parse_args()
    root = _resolve(args.root)
    baseline_root = (
        _resolve(args.baseline_root) if args.baseline_root else root / "baselines"
    )
    summary_root = root / "summary"
    raw_root = summary_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    single_source = baseline_root / "single_summary.csv"
    if single_source.is_file():
        shutil.copy2(single_source, summary_root / "single_process_baseline_summary.csv")
    else:
        pd.DataFrame().to_csv(summary_root / "single_process_baseline_summary.csv", index=False)

    single_comparison_rows: list[dict[str, Any]] = []
    baseline_runs = baseline_root / "single_run_summary.csv"
    if baseline_runs.is_file():
        frame = pd.read_csv(baseline_runs)
        for row in frame.to_dict("records"):
            single_comparison_rows.append({
                "model": row.get("model_code"),
                "process_id": row.get("process_id"),
                "fold": row.get("fold"),
                "target_mean_r2": row.get("test_target_edge_property_mean_r2"),
            })
    single_comparison_rows.extend(
        _process_fold_rows(root / "proposed_single_process", model="Proposed")
    )
    single_comparison_raw = pd.DataFrame(single_comparison_rows)
    single_comparison_raw.to_csv(raw_root / "single_process_comparison_by_fold.csv", index=False)
    _summarize(single_comparison_raw, ["model"]).to_csv(
        summary_root / "single_process_comparison_summary.csv", index=False
    )

    multi_rows: list[dict[str, Any]] = []
    generic = baseline_root / "multi_run_summary.csv"
    if generic.is_file():
        frame = pd.read_csv(generic)
        for row in frame.to_dict("records"):
            multi_rows.append({
                "model": row.get("model_code"), "fold": row.get("fold"),
                "target_mean_r2": row.get("test_target_mean_r2"),
            })
    multi_rows.extend(_fold_rows(root / "proposed_joint_10d_clean", model="Proposed"))
    multi_raw = pd.DataFrame(multi_rows)
    multi_raw.to_csv(raw_root / "multi_process_by_fold.csv", index=False)
    _summarize(multi_raw, ["model"]).to_csv(summary_root / "multi_process_summary.csv", index=False)

    zero_registry = root / "proposed_unseen/aggregate/run_registry.csv"
    if zero_registry.is_file():
        zero_raw = pd.read_csv(zero_registry)
        if "stage" in zero_raw:
            zero_raw = zero_raw.loc[zero_raw["stage"].astype(str).eq("zero_shot")].copy()
    else:
        zero_raw = pd.DataFrame()
    zero_raw.to_csv(raw_root / "zero_shot_runs.csv", index=False)
    if not zero_raw.empty and {"heldout_process", "target_mean_r2"}.issubset(zero_raw.columns):
        zero_summary = zero_raw.groupby("heldout_process").agg(
            target_mean_r2_mean=("target_mean_r2", "mean"),
            target_mean_r2_std=("target_mean_r2", "std"), folds=("fold", "count"),
        ).reset_index()
    else:
        zero_summary = pd.DataFrame()
    zero_summary.to_csv(summary_root / "zero_shot_summary.csv", index=False)

    data_eff_source = root / "data_efficiency/aggregate/learning_curve_summary.csv"
    if data_eff_source.is_file():
        shutil.copy2(data_eff_source, summary_root / "data_efficiency_summary.csv")
    else:
        pd.DataFrame().to_csv(summary_root / "data_efficiency_summary.csv", index=False)

    efficiency_source = root / "computational_efficiency/computational_efficiency_summary.csv"
    if efficiency_source.is_file():
        shutil.copy2(efficiency_source, summary_root / "computational_efficiency_summary.csv")
    else:
        pd.DataFrame().to_csv(summary_root / "computational_efficiency_summary.csv", index=False)

    shap_rows = []
    for path in sorted((root / "explainability_shap").glob("P??/global_feature_importance.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "process_id", int(path.parent.name[1:]))
        shap_rows.append(frame)
    shap_summary = pd.concat(shap_rows, ignore_index=True) if shap_rows else pd.DataFrame()
    shap_summary.to_csv(summary_root / "explainability_summary.csv", index=False)

    depth_rows = []
    for depth in range(1, 8):
        case_root = (
            root / "proposed_joint_10d_clean"
            if depth == 5
            else root / f"sensitivity_10d_clean/depth/depth_{depth}"
        )
        depth_rows.extend(_fold_rows(case_root, depth=depth, reused_default=(depth == 5)))
    depth_raw = pd.DataFrame(depth_rows)
    depth_raw.to_csv(raw_root / "depth_sensitivity_by_fold.csv", index=False)
    _summarize(depth_raw, ["depth", "reused_default"]).to_csv(
        summary_root / "depth_sensitivity_summary.csv", index=False
    )

    defaults = {"node_mass": 1.0, "node_component": 1.5e-7, "node_atom": 0.2}
    pin_rows = []
    for term, weight in defaults.items():
        for label, multiplier in (("low", 0.5), ("default", 1.0), ("high", 2.0)):
            case_root = (
                root / "proposed_joint_10d_clean"
                if label == "default"
                else root / f"sensitivity_10d_clean/pin/{term}_{label}"
            )
            pin_rows.extend(_fold_rows(
                case_root, pin_term=term, case=label, multiplier=multiplier,
                weight=weight * multiplier, reused_default=(label == "default"),
            ))
    pin_raw = pd.DataFrame(pin_rows)
    pin_raw.to_csv(raw_root / "pin_sensitivity_by_fold.csv", index=False)
    _summarize(pin_raw, ["pin_term", "case", "multiplier", "weight", "reused_default"]).to_csv(
        summary_root / "pin_sensitivity_summary.csv", index=False
    )
    print(f"[final-aggregate] wrote {summary_root}")


if __name__ == "__main__":
    main()
