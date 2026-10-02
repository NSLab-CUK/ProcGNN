from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPLEXITY = {
    "MLP": "O(B*d_in*d + B*L*d^2)",
    "GCN": "O(L*(|E|*d + |V|*d^2))",
    "GIN": "O(L*(|E|*d + |V|*d^2))",
    "GAT": "O(L*H*(|E|*d_h + |V|*d_h^2)), H*d_h=d",
    "Graphormer": "O(L*(|V|^2*d + |V|*d^2)); structural attention bias",
    "SAT": "O(L*(|V|^2*d + |V|*d^2) + L*|V|*K*d); K-hop structure encoder",
    "GraphToSFILES": "O(L*(|V|^2*d + |V|*d^2) + SFILES sequence encoder)",
    "Proposed": "O(L*(|E|*d^2 + |V|*d^2) + Set2Set + edge-readout); PIN adds training-only passes",
}


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _generic_rows(roots: list[Path]) -> list[dict]:
    """Read completed joint-baseline metadata from one or more output roots.

    The final baselines were intentionally launched in separate output roots
    (standard, transformer, and topology/SFILES).  A model/fold is retained
    once, selecting the newest metadata artifact if roots overlap.
    """
    chosen: dict[tuple[str, int], tuple[Path, dict]] = {}
    # The final registry uses the literature code B6 for the shared
    # multi-process MLP, while the paper-facing label is simply MLP.
    model_dirs = {
        "B6": "MLP", "GCN": "GCN", "GIN": "GIN", "GAT": "GAT",
        "Graphormer": "Graphormer", "SAT": "SAT", "GraphToSFILES": "GraphToSFILES",
    }
    for root in roots:
        for model_dir, model in model_dirs.items():
            for path in sorted((root / "multi" / model_dir).glob("fold_*/run_summary.json")):
                payload = _json(path)
                fold = int(payload.get("fold", -1))
                key = (model, fold)
                if key not in chosen or path.stat().st_mtime > chosen[key][0].stat().st_mtime:
                    chosen[key] = (path, payload)
    rows = []
    for (model, fold), (path, payload) in sorted(chosen.items()):
        rows.append({
                "model": model, "fold": int(payload.get("fold", -1)),
                # Proposed production artifacts expose the final pooled
                # property metric on validation only.  Use that same split and
                # namespace for every efficiency row instead of mixing the
                # legacy strict target_mean_r2 with baseline test metrics.
                "target_edge_property_mean_r2": payload.get(
                    "val_target_edge_property_mean_r2", np.nan
                ),
                "performance_metric": "val_target_edge_property_mean_r2",
                "total_training_time_sec": payload.get("training_seconds", np.nan),
                "peak_gpu_allocated_mb": payload.get("peak_gpu_memory_mb", np.nan),
                "parameter_count": payload.get("parameter_count", np.nan),
                "theoretical_complexity": COMPLEXITY[model],
                "source_run_summary": str(path),
            })
    return rows


def _proposed_rows(root: Path) -> list[dict]:
    rows = []
    for fold_dir in sorted((root / "All").glob("fold_*")):
        metrics_files = sorted(fold_dir.rglob("metrics_per_epoch.csv"), key=lambda path: path.stat().st_mtime)
        metrics_jsons = sorted(fold_dir.rglob("metrics.json"), key=lambda path: path.stat().st_mtime)
        if not metrics_files:
            continue
        frame = pd.read_csv(metrics_files[-1])
        payload = _json(metrics_jsons[-1]) if metrics_jsons else {}
        time_col = pd.to_numeric(frame.get("train_epoch_time_sec"), errors="coerce")
        memory_col = pd.to_numeric(frame.get("train_peak_gpu_memory_mb"), errors="coerce")
        rows.append({
            "model": "Proposed", "fold": int(fold_dir.name.split("_")[-1]),
            "target_edge_property_mean_r2": payload.get(
                "val_target_edge_property_mean_r2",
                payload.get("target_edge_property_mean_r2", np.nan),
            ),
            "performance_metric": "val_target_edge_property_mean_r2",
            "total_training_time_sec": float(time_col.sum()) if hasattr(time_col, "sum") else np.nan,
            "peak_gpu_allocated_mb": float(memory_col.max()) if hasattr(memory_col, "max") else np.nan,
            "parameter_count": payload.get("parameter_count", np.nan),
            "theoretical_complexity": COMPLEXITY["Proposed"],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate final joint-model efficiency measurements")
    parser.add_argument(
        "--baseline-root", nargs="+", default=["outputs/final_paper/baselines"],
        help="One or more baseline output roots; newest duplicate model/fold is retained.",
    )
    parser.add_argument("--proposed-root", default="outputs/final_paper/proposed_joint_10d_clean")
    parser.add_argument("--output-root", default="outputs/final_paper/computational_efficiency")
    args = parser.parse_args()
    baseline_roots = [
        (PROJECT_ROOT / root).resolve() if not Path(root).is_absolute() else Path(root).resolve()
        for root in args.baseline_root
    ]
    rows = _generic_rows(baseline_roots)
    rows += _proposed_rows((PROJECT_ROOT / args.proposed_root).resolve())
    output = (PROJECT_ROOT / args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(output / "efficiency_by_fold.csv", index=False)
    if raw.empty:
        print("[efficiency] no completed final runs found")
        return
    summary = raw.groupby("model").agg(
        target_edge_property_mean_r2_mean=("target_edge_property_mean_r2", "mean"),
        target_edge_property_mean_r2_std=("target_edge_property_mean_r2", "std"),
        performance_metric=("performance_metric", "first"),
        training_time_mean=("total_training_time_sec", "mean"), training_time_std=("total_training_time_sec", "std"),
        peak_gpu_memory_mean=("peak_gpu_allocated_mb", "mean"), peak_gpu_memory_std=("peak_gpu_allocated_mb", "std"),
        parameter_count_mean=("parameter_count", "mean"), parameter_count_std=("parameter_count", "std"),
        folds=("fold", "count"), theoretical_complexity=("theoretical_complexity", "first"),
    ).reset_index()
    summary.to_csv(output / "computational_efficiency_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
