from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET_PROPERTY_COLUMNS = [
    "val_target_edge_r2_temp",
    "val_target_edge_r2_pres",
    "val_target_edge_r2_frac_h2o",
    "val_target_edge_r2_frac_h2",
    "val_target_edge_r2_frac_ch4",
    "val_target_edge_r2_frac_co2",
    "val_target_edge_r2_frac_co",
    "val_target_edge_r2_frac_o2",
    "val_target_edge_r2_frac_n2",
    "val_target_edge_r2_mass_flow",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive the final data-efficiency optimizer-step budget")
    parser.add_argument("--existing-transfer-root", default="outputs/260805_singleproc_unseen")
    parser.add_argument("--samples-per-epoch", type=int, default=2000)
    parser.add_argument("--output-root", default="outputs/final_paper/data_efficiency/budget_audit")
    args = parser.parse_args()
    root = (PROJECT_ROOT / args.existing_transfer_root).resolve()
    rows = []
    for run_dir in sorted(root.glob("heldout_P*/fold_*/transfer_full")):
        paths = sorted(run_dir.rglob("metrics_per_epoch.csv"), key=lambda path: path.stat().st_mtime)
        if not paths:
            continue
        frame = pd.read_csv(paths[-1])
        if frame.empty:
            continue
        if "val_target_edge_property_mean_r2" in frame:
            values = pd.to_numeric(
                frame["val_target_edge_property_mean_r2"], errors="coerce"
            )
            metric_source = "val_target_edge_property_mean_r2"
        elif all(column in frame for column in TARGET_PROPERTY_COLUMNS):
            values = frame[TARGET_PROPERTY_COLUMNS].apply(
                pd.to_numeric, errors="coerce"
            ).mean(axis=1, skipna=False)
            metric_source = "reconstructed_10_target_property_r2_mean"
        elif "val_target_mean_r2" in frame:
            values = pd.to_numeric(frame["val_target_mean_r2"], errors="coerce")
            metric_source = "legacy_val_target_mean_r2_fallback"
        else:
            continue
        if not values.notna().any():
            continue
        best_index = values.idxmax()
        best_epoch = int(frame.loc[best_index, "epoch"])
        target_updates = pd.to_numeric(
            frame.get("train_edge_step_target_edge_count"), errors="coerce"
        ).median()
        updates_per_sample = int(round(float(target_updates))) + 2
        rows.append({
            "run_dir": str(run_dir), "best_epoch": best_epoch,
            "final_epoch": int(frame["epoch"].max()),
            "target_updates_per_sample": int(round(float(target_updates))),
            "estimated_optimizer_updates_per_sample": updates_per_sample,
            "estimated_optimizer_steps_at_best": best_epoch * args.samples_per_epoch * updates_per_sample,
            "metric_source": metric_source,
        })
    if not rows:
        raise RuntimeError(f"No completed transfer metric histories found under {root}")
    raw = pd.DataFrame(rows)
    representative_best_epoch = int(round(float(raw["best_epoch"].median())))
    representative_updates = int(round(float(raw["estimated_optimizer_updates_per_sample"].median())))
    selected = representative_best_epoch * int(args.samples_per_epoch) * representative_updates
    output = (PROJECT_ROOT / args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "source_transfer_runs.csv", index=False)
    report = {
        "source_root": str(root), "completed_runs": len(raw),
        "best_epoch_median": representative_best_epoch,
        "best_epoch_mean": float(raw["best_epoch"].mean()),
        "samples_per_epoch": int(args.samples_per_epoch),
        "optimizer_updates_per_sample_median": representative_updates,
        "update_formula": "1 non-target macro + N target-edge + 1 node-anchor update",
        "selected_total_optimizer_steps": selected,
        "artifact_limit": (
            "Legacy histories did not persist applied optimizer-step counters; the estimate is "
            "reconstructed from best epoch and the deterministic sample-hybrid update structure."
        ),
    }
    (output / "fixed_step_budget.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
