from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    PROJECT_ROOT
    / "configs"
    / "experiment"
    / "pinn"
    / "model_260716_pi_tw40_legacy_fracfocus_f01.yaml"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "configs" / "experiment" / "pinn" / "ablation_260728"
)


# R: property-stream Role16
# J: joint Node PINN update with a supervised anchor and direct H supervision
# M: dual-space Mass Flow loss plus 350 mandatory tail samples
ARMS: dict[str, dict[str, bool]] = {
    "a0_r0_j0_m0": {"role16": False, "joint": False, "mass_aux": False, "tail350": False},
    "a1_r1_j0_m0": {"role16": True, "joint": False, "mass_aux": False, "tail350": False},
    "a2_r0_j1_m0": {"role16": False, "joint": True, "mass_aux": False, "tail350": False},
    "a3_r0_j0_m1": {"role16": False, "joint": False, "mass_aux": True, "tail350": True},
}


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.setdefault(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping at {key!r}, got {type(value).__name__}.")
    return value


def build_arm(base: dict[str, Any], arm_name: str, factors: dict[str, bool]) -> dict[str, Any]:
    payload = copy.deepcopy(base)
    run_root = f"outputs/0728_ablation3/{arm_name}"
    payload["experiment_name"] = f"m260716_ablation3_{arm_name}"
    payload["output_dir"] = run_root
    payload["save_dir"] = f"{run_root}/checkpoints"
    payload["log_dir"] = f"{run_root}/logs"
    payload["resume_path"] = ""

    overrides = _mapping(payload, "overrides")
    model = _mapping(overrides, "model")
    train = _mapping(overrides, "train")

    model["target_branch_hidden_adapter"] = {"enabled": False}
    model["property_stream_role"] = {
        "enabled": bool(factors["role16"]),
        "embedding_dim": 16,
        "unknown_role_id": 0,
    }

    joint = bool(factors["joint"])
    train["node_pinn_optimization"] = {
        "update_mode": "joint_with_supervised_anchor" if joint else "separate",
        "supervised_anchor_weight": 1.0,
        "node_outer_weight": 0.05 if joint else 1.0,
        "anchor_apply_target_weight": False,
        "log_gradient_diagnostics": False,
    }
    train["lambda_h"] = 0.02 if joint else 1.0e-11

    train["mass_flow_physical_auxiliary"] = {
        "enabled": bool(factors["mass_aux"]),
        "weight": 0.10,
        "loss": "huber",
        "delta": 1.0,
        "residual_clip": 10.0,
        "scale_method": "std",
        "minimum_scale": 1.0e-8,
        "epsilon": 1.0e-8,
    }

    epoch_sampler = _mapping(train, "epoch_sampler")
    tail_enabled = bool(factors["tail350"])
    epoch_sampler["mass_flow_tail"] = {
        "enabled": tail_enabled,
        "target_items_per_epoch": 350 if tail_enabled else 0,
        "property": "Mass_Flow",
        "quantiles": [0.90, 0.95, 0.99, 1.0],
        "bin_quotas": {
            "q0.90_0.95": 100,
            "q0.95_0.99": 150,
            "q0.99_1.00": 100,
        },
        "quantile_scope": "global",
        "process_balanced": True,
        "canonical_edge_balanced": True,
        "allow_duplicate_samples": False,
        "canonical_edge_min_count": 50,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the 260728 three-method ablation experiment YAML files."
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    base_path = args.base_config.resolve()
    output_dir = args.output_dir.resolve()
    with base_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle)
    if not isinstance(base, dict):
        raise TypeError(f"Base config must contain a mapping: {base_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for arm_name, factors in ARMS.items():
        payload = build_arm(base, arm_name, factors)
        destination = output_dir / f"{arm_name}.yaml"
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
        print(destination.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
