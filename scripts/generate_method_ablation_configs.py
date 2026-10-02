#!/usr/bin/env python3
"""Generate configs/experiment/method_ablation/*.yaml from method_ablation_registry."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from method_ablation_registry import (  # noqa: E402
    BASELINE_MODEL,
    METHOD_EXPERIMENT_CATALOG,
    changed_config_keys,
    merged_model_overrides,
)

ROOT = PROJECT_ROOT / "configs/experiment/method_ablation"

COMMON_DATA = {
    "task_mode": "edge_all",
    "normalize_y_edge": True,
    "normalize_targets": False,
    "train_data_path": "data/datasets_v3/train.csv",
    "val_data_path": "data/datasets_v3/val.csv",
    "test_data_path": "data/datasets_v3/test.csv",
    "edge_all_processes": list(range(1, 11)),
    "edge_all_split_mode": "seen_process",
}
COMMON_TRAIN = {
    "val_interval": 1,
    "loss_type_edge_all": "smooth_l1",
    "answer_edge_weight": 5.0,
    "answer_edge_weight_h2": 5.0,
    "answer_edge_weight_co2": 5.0,
    "answer_edge_weight_h2o": 5.0,
    "edge_all_print_first_batch": True,
    "edge_all_assert_no_leakage": True,
}
EDGE_ALL = {
    "data": {
        "reference_dir": "data/reference/v3",
        "stream_dir": "data/main_data_Streams",
        "processes": list(range(1, 11)),
        "split_mode": "seen_process",
    },
    "loss": {"type": "masked_smooth_l1", "use_y_edge_mask": True, "normalize_targets": True},
    "normalization": {"scaler": "standard", "fit_on_train_only": True, "mask_aware": True},
    "debug": {"print_first_batch": True, "assert_no_leakage": True, "export_predictions": True},
}


def main() -> None:
    (ROOT / "optional").mkdir(parents=True, exist_ok=True)
    for entry in METHOD_EXPERIMENT_CATALOG:
        name = entry["name"]
        model_ov = merged_model_overrides(entry)
        out_root = f"outputs/process_kfold_method_ablation/{name}"
        payload = {
            "experiment_name": f"method_ablation_{name}",
            "seed": 42,
            "device": "cuda",
            "eval_only": False,
            "model_config": "configs/model/process_encoder.yaml",
            "train_config": "configs/train/process_train.yaml",
            "data_config": "configs/data/process_data.yaml",
            "output_dir": out_root,
            "save_dir": f"{out_root}/checkpoints",
            "log_dir": f"{out_root}/logs",
            "resume_path": "",
            "overrides": {"model": model_ov, "data": COMMON_DATA, "train": COMMON_TRAIN},
            "edge_all": EDGE_ALL,
        }
        sub = ROOT / ("optional" if entry.get("optional") else "")
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / f"{name}.yaml"
        chg = changed_config_keys(entry) or ["(none)"]
        lines = [
            f"# Method ablation (one-factor): {name}",
            f"# Group: {entry['ablation_group']}",
            f"# Changed vs baseline: {', '.join(chg)}",
        ]
        if entry.get("baseline_equivalent_label"):
            lines.append(f"# Baseline-equivalent: {entry['baseline_equivalent_label']}")
        if entry.get("notes"):
            lines.append(f"# Note: {entry['notes']}")
        lines.append("")
        with path.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            yaml.dump(payload, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        print(f"wrote {path.relative_to(PROJECT_ROOT)}")

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from process_graph.experiment.loaders import load_experiment_config

    exp = load_experiment_config((ROOT / "exp_B0_baseline.yaml").resolve())
    resolved = {
        "source": "load_experiment_config(configs/experiment/method_ablation/exp_B0_baseline.yaml)",
        "capacity_ablation_summary": (
            "See outputs/process_kfold_capacity_ablation/summary/experiment_summary.csv when available."
        ),
        "model": {k: getattr(exp.model, k) for k in sorted(BASELINE_MODEL) if hasattr(exp.model, k)},
        "data_task_mode": getattr(exp.data, "task_mode", None),
    }
    (ROOT / "baseline_resolved.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    print(f"wrote {ROOT.relative_to(PROJECT_ROOT)}/baseline_resolved.json")


if __name__ == "__main__":
    main()
