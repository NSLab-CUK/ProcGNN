from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPTHS = tuple(range(1, 8))
PIN_TERMS = {
    "node_mass": ("mass", "lambda_node_mass"),
    "node_component": ("component", "lambda_node_component"),
    "node_atom": ("atom", "lambda_node_atom"),
}
MULTIPLIERS = (0.5, 1.0, 2.0)


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_variant(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    if path.is_file() and path.read_text(encoding="utf-8") == serialized:
        return
    path.write_text(serialized, encoding="utf-8")


def _command(config: Path, output: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, str(PROJECT_ROOT / "scripts/run_process_kfold_experiments.py"),
        "--base-config", str(config), "--splits-dir", str(args.split_root),
        "--merged-csv", str(args.merged_csv), "--process-ids", *map(str, range(1, 11)),
        "--joint-all-processes", "--only-folds", *map(str, args.folds),
        "--max-epochs", str(args.max_epochs),
        "--monitor-metric", "val_target_edge_property_mean_r2",
        "--monitor-mode", "max", "--output-root", str(output), "--skip-startup-debug",
    ] + (["--skip-existing"] if args.resume_existing else [])


def _default_reference(
    root: Path | None,
    folds: list[int],
    *,
    require_no_volume_head: bool,
    require_sender_flow_attention: bool,
) -> dict[str, Any]:
    if root is None:
        return {"default_result_root": None, "default_checkpoints": []}
    checkpoints: list[str] = []
    missing: list[int] = []
    for fold in folds:
        matches = sorted((root / "All" / f"fold_{fold:02d}").rglob("best.pt"))
        if not matches:
            missing.append(int(fold))
        else:
            checkpoint = matches[-1]
            if require_no_volume_head or require_sender_flow_attention:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                state = payload.get("model_state_dict", payload.get("model", payload))
                volume_keys = [str(key) for key in state if "volume_head" in str(key)]
                if require_no_volume_head and volume_keys:
                    raise ValueError(
                        "10D sensitivity cannot reuse an 11D checkpoint containing volume_head: "
                        f"{checkpoint}"
                    )
                if require_sender_flow_attention:
                    config = payload.get("config", {})
                    attention_group = (
                        config.get("model", {})
                        .get("flow_gnn", {})
                        .get("forward_attention_group")
                    )
                    if attention_group != "source":
                        raise ValueError(
                            "sender-normalized Flow Attention sensitivity requires a "
                            "checkpoint with model.flow_gnn.forward_attention_group=source, "
                            f"got {attention_group!r}: {checkpoint}"
                        )
            checkpoints.append(str(checkpoint))
    if missing:
        raise FileNotFoundError(
            f"default sensitivity result root is missing best.pt for folds {missing}: {root}"
        )
    return {"default_result_root": str(root), "default_checkpoints": checkpoints}


def main() -> None:
    parser = argparse.ArgumentParser(description="Final depth and one-at-a-time PIN-weight sensitivity")
    parser.add_argument("--mode", choices=("depth", "pin", "all"), default="all")
    parser.add_argument("--base-config", default="configs/experiment/pinn/model_260805_10d_frac1.yaml")
    parser.add_argument("--split-root", default="data/splits/all_processes_full100k_outer5_grouped_60_20_20")
    parser.add_argument("--merged-csv", default="data/datasets_v3/process_main_merged.csv")
    parser.add_argument("--output-root", default="outputs/final_paper/sensitivity")
    parser.add_argument(
        "--default-result-root",
        default=None,
        help="Existing default-depth/default-PIN joint 5-fold root reused as the multiplier=1 reference.",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--cases",
        nargs="+",
        default=None,
        help=(
            "Run only named cases, e.g. node_mass_low node_atom_high or depth_7. "
            "This permits sharding non-default cases without copying the default checkpoints."
        ),
    )
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.base_config = (PROJECT_ROOT / args.base_config).resolve()
    args.split_root = (PROJECT_ROOT / args.split_root).resolve()
    args.merged_csv = (PROJECT_ROOT / args.merged_csv).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    base = _load(args.base_config)
    target_columns = list(base.get("overrides", {}).get("data", {}).get("edge_target_columns", []))
    default_root = (PROJECT_ROOT / args.default_result_root).resolve() if args.default_result_root else None
    depth_cases = {f"depth_{depth}" for depth in DEPTHS}
    pin_cases = {
        f"{term}_{label}"
        for term in PIN_TERMS
        for label in ("low", "default", "high")
    }
    allowed_cases = (
        depth_cases | pin_cases if args.mode == "all"
        else depth_cases if args.mode == "depth"
        else pin_cases
    )
    requested_cases = set(args.cases or allowed_cases)
    unknown_cases = requested_cases - allowed_cases
    if unknown_cases:
        raise ValueError(
            f"cases incompatible with mode={args.mode!r}: {sorted(unknown_cases)}; "
            f"allowed={sorted(allowed_cases)}"
        )
    needs_default_reference = bool(
        requested_cases & {"depth_5", "node_mass_default", "node_component_default", "node_atom_default"}
    )
    default_reference = (
        _default_reference(
            default_root,
            args.folds,
            require_no_volume_head="Vol_Flow" not in target_columns,
            require_sender_flow_attention=True,
        )
        if needs_default_reference
        else {"default_result_root": None, "default_checkpoints": []}
    )
    configs = output_root / "resolved_configs"
    plans: list[dict[str, Any]] = []

    if args.mode in {"depth", "all"}:
        for depth in DEPTHS:
            if f"depth_{depth}" not in requested_cases:
                continue
            if depth == 5:
                plans.append({
                    "family": "depth", "case": "depth_5", "reused_default": True,
                    **default_reference,
                })
                continue
            config = copy.deepcopy(base)
            config["experiment_name"] = f"final_depth_{depth}"
            config["overrides"]["model"]["num_layers"] = int(depth)
            config_path = configs / f"depth_{depth}.yaml"
            _write_variant(config_path, config)
            command = _command(config_path, output_root / "depth" / f"depth_{depth}", args)
            plans.append({"family": "depth", "case": f"depth_{depth}", "reused_default": False, "command": command})

    if args.mode in {"pin", "all"}:
        train = base["overrides"]["train"]
        for term, (block, legacy_key) in PIN_TERMS.items():
            default = float(train["node_balance_pi"][block]["weight"])
            for multiplier in MULTIPLIERS:
                label = {0.5: "low", 1.0: "default", 2.0: "high"}[multiplier]
                if f"{term}_{label}" not in requested_cases:
                    continue
                if multiplier == 1.0:
                    plans.append({
                        "family": "pin", "case": f"{term}_{label}", "reused_default": True,
                        "weight": default, **default_reference,
                    })
                    continue
                config = copy.deepcopy(base)
                weight = default * multiplier
                config["experiment_name"] = f"final_pin_{term}_{label}"
                variant_train = config["overrides"]["train"]
                variant_train["node_balance_pi"][block]["weight"] = weight
                variant_train[legacy_key] = weight
                config_path = configs / f"pin_{term}_{label}.yaml"
                _write_variant(config_path, config)
                command = _command(config_path, output_root / "pin" / f"{term}_{label}", args)
                plans.append({"family": "pin", "case": f"{term}_{label}", "reused_default": False, "weight": weight, "command": command})

    output_root.mkdir(parents=True, exist_ok=True)
    plan_path = output_root / "sensitivity_plan.json"
    plan_path.write_text(json.dumps(plans, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"cases": len(plans), "new_training_cases": sum(not row["reused_default"] for row in plans), "folds": args.folds}, indent=2))
    for row in plans:
        if row.get("command"):
            print(subprocess.list2cmdline(row["command"]))
            if not args.dry_run:
                subprocess.run(row["command"], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
