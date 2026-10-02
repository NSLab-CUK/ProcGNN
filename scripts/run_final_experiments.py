from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from final_experiment_registry import load_registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _commands() -> dict[str, list[list[str]]]:
    py = sys.executable
    proposed = "configs/experiment/pinn/model_260805_10d_frac1.yaml"
    joint_split = "data/splits/all_processes_full100k_outer5_grouped_60_20_20"
    merged = "data/datasets_v3/process_main_merged.csv"
    all_processes = [str(value) for value in range(1, 11)]
    folds = [str(value) for value in range(1, 6)]
    common_baseline = [py, "scripts/run_final_baseline_experiments.py"]
    final_root = "outputs/0819final"
    baseline_root = f"{final_root}/baselines"
    proposed_joint = [
        py, "scripts/run_process_kfold_experiments.py", "--base-config", proposed,
        "--splits-dir", joint_split, "--merged-csv", merged, "--process-ids", *all_processes,
        "--joint-all-processes", "--only-folds", *folds, "--max-epochs", "30",
        "--monitor-metric", "val_target_edge_property_mean_r2", "--monitor-mode", "max",
        "--output-root", f"{final_root}/proposed_joint_10d_clean", "--skip-existing", "--skip-startup-debug",
    ]
    return {
        "proposed_single_process": [[
            py, "scripts/run_process_kfold_experiments.py", "--base-config", proposed,
            "--splits-dir", joint_split, "--merged-csv", merged,
            "--process-ids", *all_processes, "--only-folds", *folds,
            "--max-epochs", "30", "--monitor-metric", "val_target_edge_property_mean_r2",
            "--monitor-mode", "max", "--output-root", f"{final_root}/proposed_single_process",
            "--skip-existing", "--skip-startup-debug",
        ]],
        "single_process_baselines": [[
            *common_baseline, "--scope", "single", "--models",
            "M1", "M2", "M3", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "G1", "G2", "G3",
            "--process-ids", *all_processes, "--folds", *folds,
            "--output-root", baseline_root, "--max-epochs", "30", "--patience", "5", "--resume-existing",
        ]],
        "multi_process_comparison": [[
            *common_baseline, "--scope", "multi", "--models", "B6", "GCN", "GIN", "GAT",
            "--process-ids", *all_processes, "--folds", *folds,
            "--output-root", baseline_root, "--max-epochs", "30",
            "--patience", "5", "--resume-existing",
        ], proposed_joint],
        "proposed_zero_shot": [[
            py, "scripts/run_single_process_full_unseen_experiments.py", "--base-config", proposed,
            "--split-root", "data/splits/single_process_full_unseen_60_20_20", "--merged-csv", merged,
            "--output-root", f"{final_root}/proposed_unseen", "--heldout-processes", *all_processes,
            "--folds", *folds, "--pretrain-source-fold", "1", "--mode", "pretrain",
            "--max-epochs-pretrain", "30", "--monitor-metric", "val_target_edge_property_mean_r2", "--monitor-mode", "max",
            "--gpu-ids", "0", "--max-parallel", "1", "--resume-existing", "--skip-startup-debug",
        ], [
            py, "scripts/run_single_process_full_unseen_experiments.py", "--base-config", proposed,
            "--split-root", "data/splits/single_process_full_unseen_60_20_20", "--merged-csv", merged,
            "--output-root", f"{final_root}/proposed_unseen", "--heldout-processes", *all_processes,
            "--folds", *folds, "--pretrain-source-fold", "1", "--mode", "zero_shot",
            "--gpu-ids", "0", "--max-parallel", "1", "--resume-existing", "--skip-startup-debug",
        ]],
        "proposed_data_efficiency": [[
            py, "scripts/run_transfer_data_efficiency_experiments.py", "--base-config", proposed,
            "--split-root", "data/splits/single_process_full_unseen_60_20_20", "--merged-csv", merged,
            "--existing-unseen-root", f"{final_root}/proposed_unseen",
            "--output-root", f"{final_root}/data_efficiency", "--heldout-processes", *all_processes,
            "--folds", *folds, "--data-ratios", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9",
            "--modes", "transfer", "--total-optimizer-steps", "80000", "--max-epochs", "30",
            "--early-stopping-patience", "5", "--monitor-metric", "val_target_edge_property_mean_r2",
            "--gpu-ids", "0", "--max-parallel", "1", "--resume-existing", "--skip-startup-debug",
        ]],
        "computational_efficiency": [[
            py, "scripts/aggregate_final_efficiency.py",
            "--baseline-root", baseline_root,
            "--proposed-root", f"{final_root}/proposed_joint_10d_clean",
            "--output-root", f"{final_root}/computational_efficiency",
        ]],
        "explainability_shap": [[
            py, "scripts/run_final_shap_experiments.py",
            "--proposed-root", f"{final_root}/proposed_joint_10d_clean",
            "--split-root", joint_split,
            "--output-root", f"{final_root}/explainability_shap",
            "--process-ids", *all_processes, "--fold", "1",
            "--background-size", "32", "--explain-samples", "16",
            "--require-flowsheet", "--resume-existing",
        ]],
        "sensitivity": [[
            py, "scripts/run_final_sensitivity_experiments.py", "--mode", "all", "--folds", *folds,
            "--base-config", proposed,
            "--default-result-root", f"{final_root}/proposed_joint_10d_clean",
            "--output-root", f"{final_root}/sensitivity_10d_clean",
            "--max-epochs", "30", "--resume-existing",
        ]],
        "aggregation": [[
            py, "scripts/aggregate_final_experiments.py", "--root", final_root,
            "--baseline-root", baseline_root,
        ], [
            py, "scripts/final_experiment_status.py", "--output-root", final_root,
            "--baseline-root", baseline_root,
        ]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Final paper experiment master runner")
    parser.add_argument("--families", nargs="+", default=["all"])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--plan-json",
        default="outputs/0819final/final_command_plan.json",
    )
    args = parser.parse_args()
    load_registry()
    commands = _commands()
    selected = list(commands) if "all" in args.families else args.families
    unknown = sorted(set(selected) - set(commands))
    if unknown:
        raise ValueError(f"unknown final families: {unknown}")
    plan = {name: commands[name] for name in selected}
    plan_path = (PROJECT_ROOT / args.plan_json).resolve()
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    for family in selected:
        for command in commands[family]:
            print(f"[{family}] {subprocess.list2cmdline(command)}", flush=True)
            if args.execute:
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
