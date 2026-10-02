from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from final_experiment_registry import load_registry, logical_counts

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _json_status(path: Path) -> str:
    if not path.is_file():
        return "missing"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status", "missing"))
    except Exception:
        return "failed"


def _count_status(root: Path) -> tuple[int, int]:
    completed = failed = 0
    for path in root.rglob("status.json") if root.exists() else []:
        try:
            state = json.loads(path.read_text(encoding="utf-8")).get("status")
        except Exception:
            state = "failed"
        completed += int(state == "completed")
        failed += int(state == "failed")
    return completed, failed


def _count_kfold(root: Path) -> int:
    return sum(
        1 for fold in (root / "All").glob("fold_*")
        if any(fold.rglob("metrics.json")) and any(fold.rglob("best.pt"))
    ) if root.exists() else 0


def _count_process_kfold(root: Path) -> int:
    return sum(
        1 for fold in root.glob("Process*/fold_*")
        if any(fold.rglob("metrics.json")) and any(fold.rglob("best.pt"))
    ) if root.exists() else 0


def _count_zero_shot(root: Path) -> int:
    return sum(
        1 for path in root.glob("heldout_P*/fold_*/zero_shot/evaluation_metadata.json")
        if path.is_file()
    ) if root.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Final experiment completion matrix")
    parser.add_argument("--output-root", default="outputs/final_paper")
    parser.add_argument(
        "--baseline-root",
        default=None,
        help="Baseline artifact root; defaults to <output-root>/baselines.",
    )
    args = parser.parse_args()
    registry = load_registry()
    expected = logical_counts(registry)
    root = (PROJECT_ROOT / args.output_root).resolve()
    baseline_root = (
        (PROJECT_ROOT / args.baseline_root).resolve()
        if args.baseline_root
        else root / "baselines"
    )
    paths = {
        "proposed_single_process": root / "proposed_single_process",
        "single_process_baselines": baseline_root / "single",
        "multi_process_comparison": baseline_root / "multi",
        "proposed_zero_shot": root / "proposed_unseen",
        "proposed_data_efficiency": root / "data_efficiency",
        "computational_efficiency": root / "computational_efficiency",
        "explainability_shap": root / "explainability_shap",
        "sensitivity_depth": root / "sensitivity_10d_clean/depth",
        "sensitivity_pin": root / "sensitivity_10d_clean/pin",
    }
    rows = []
    for experiment, count in expected.items():
        completed, failed = _count_status(paths[experiment])
        blocked = 0
        if experiment == "proposed_single_process":
            completed = _count_process_kfold(paths[experiment])
        elif experiment == "multi_process_comparison":
            completed += _count_kfold(root / "proposed_joint_10d_clean")
        elif experiment == "proposed_zero_shot":
            completed = _count_zero_shot(paths[experiment])
        elif experiment == "computational_efficiency":
            efficiency = paths[experiment] / "efficiency_by_fold.csv"
            completed = len(pd.read_csv(efficiency)) if efficiency.is_file() else 0
        elif experiment == "explainability_shap":
            completed = sum(
                1 for path in paths[experiment].glob("P??")
                if (path / "sample_level_shap.csv").is_file()
                and _json_status(path / "flowsheet_status.json") == "completed"
            )
            blocked = sum(
                1 for path in paths[experiment].glob("P??")
                if _json_status(path / "flowsheet_status.json") == "blocked"
            )
        elif experiment == "sensitivity_depth":
            completed = _count_kfold(root / "proposed_joint_10d_clean") + sum(
                _count_kfold(case) for case in paths[experiment].glob("depth_*")
            )
        elif experiment == "sensitivity_pin":
            completed = 3 * _count_kfold(root / "proposed_joint_10d_clean") + sum(
                _count_kfold(case) for case in paths[experiment].glob("*_*")
            )
        rows.append({
            "experiment": experiment, "expected": count, "completed": completed,
            "missing": max(0, count - completed), "failed": failed, "blocked": blocked,
            "legacy": 0, "path": str(paths[experiment]),
        })
    frame = pd.DataFrame(rows)
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "master_experiment_status.csv", index=False)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
