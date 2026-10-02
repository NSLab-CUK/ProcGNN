from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _discover_checkpoint(root: Path, fold: int) -> Path:
    fold_tokens = {f"fold_{fold:02d}", f"fold_{fold}"}
    candidates = [
        path for path in root.rglob("best.pt")
        if any(token in {part.lower() for part in path.parts} for token in fold_tokens)
    ]
    if not candidates:
        raise FileNotFoundError(f"No Proposed joint best.pt for fold {fold} under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _completed(path: Path) -> bool:
    files_ready = all((path / name).is_file() for name in (
        "sample_level_shap.csv", "global_feature_importance.csv", "node_importance.csv",
        "topk_nodes.csv", "stream_importance_status.csv", "flowsheet_status.json",
    ))
    if not files_ready:
        return False
    try:
        flowsheet = json.loads((path / "flowsheet_status.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return flowsheet.get("status") == "completed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final target-only SHAP for each Process")
    parser.add_argument("--proposed-root", default="outputs/final_paper/proposed_joint_10d_clean")
    parser.add_argument("--split-root", default="data/splits/all_processes_full100k_outer5_grouped_60_20_20")
    parser.add_argument("--output-root", default="outputs/final_paper/explainability_shap")
    parser.add_argument("--process-ids", nargs="+", type=int, default=list(range(1, 11)))
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--background-size", type=int, default=32)
    parser.add_argument("--explain-samples", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--coordinate-file", default="data/process_overall_img/unit_coordinates.csv")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--require-flowsheet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proposed_root = _resolve(args.proposed_root)
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/validate_shap_flowsheet_coordinates.py"),
            "--coordinates", str(args.coordinate_file),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    checkpoint = _discover_checkpoint(proposed_root, args.fold)
    manifest = _resolve(args.split_root) / f"fold_{args.fold:02d}" / "test.csv"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    plan = {
        "checkpoint": str(checkpoint), "fold": args.fold,
        "process_ids": args.process_ids, "expected_runs": len(args.process_ids),
        "manifest": str(manifest), "target_scope": "target_predictions_only",
    }
    (output_root / "explainability_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    for process_id in args.process_ids:
        process_output = output_root / f"P{process_id:02d}"
        if args.resume_existing and _completed(process_output):
            print(f"[shap][skip] P{process_id:02d}", flush=True)
            continue
        command = [
            sys.executable, str(PROJECT_ROOT / "scripts/run_final_shap_explainability.py"),
            "--checkpoint", str(checkpoint), "--manifest", str(manifest),
            "--process-id", str(process_id), "--output-root", str(output_root),
            "--background-size", str(args.background_size),
            "--explain-samples", str(args.explain_samples),
            "--device", str(args.device), "--coordinate-file", str(args.coordinate_file),
        ]
        if args.require_flowsheet:
            command.append("--require-flowsheet")
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
