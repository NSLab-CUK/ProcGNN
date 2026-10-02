from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = PROJECT_ROOT / "configs/final_experiments/final_protocol_260811.yaml"


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    target = payload.get("target_schema", {})
    if int(target.get("dimension", 0)) != len(target.get("properties", [])):
        raise ValueError("target schema dimension and property list disagree")
    if "Vol_Flow" in target.get("properties", []):
        raise ValueError("final target schema must not contain Vol_Flow")
    single = payload["experiments"]["single_process_baselines"]
    required_single = ["M1", "M2", "M3", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "G1", "G2", "G3"]
    if single.get("models") != required_single:
        raise ValueError(f"single-process final registry must contain exactly {required_single}")
    if list(single.get("display_names", {})) != required_single:
        raise ValueError("single-process baseline display names must cover all 13 models in registry order")
    if single.get("final_gnn_mapping") != {"G1": "GCN", "G2": "GIN", "G3": "GAT"}:
        raise ValueError("final GNN mapping must be G1=GCN, G2=GIN, G3=GAT")
    if single.get("gnn_prediction_unit") != "one_target_edge" or single.get("gnn_head") != "shared_10_property_head":
        raise ValueError("single-process GNN must use one target edge per shared 10-property prediction")
    multi = payload["experiments"]["multi_process_comparison"]
    if multi.get("models") != ["B6", "GCN", "GIN", "GAT", "Proposed"]:
        raise ValueError("multi-process comparison registry is not the final five-model set")
    if multi.get("generic_gnn_head") != "shared_across_all_processes_and_target_edges":
        raise ValueError("multi-process generic GNNs must use one shared target-edge head")
    if multi.get("generic_gnn_process_id_routing") != "forbidden":
        raise ValueError("process-ID routing is forbidden for multi-process generic GNNs")
    return payload


def logical_counts(registry: dict[str, Any]) -> dict[str, int]:
    experiments = registry["experiments"]
    return {
        "proposed_single_process": int(experiments["proposed_single_process"]["expected_runs"]),
        "single_process_baselines": int(experiments["single_process_baselines"]["expected_runs"]),
        "multi_process_comparison": int(experiments["multi_process_comparison"]["expected_runs"]),
        "proposed_zero_shot": int(experiments["proposed_zero_shot"]["expected_evaluation_runs"]),
        "proposed_data_efficiency": int(experiments["proposed_data_efficiency"]["expected_runs"]),
        "computational_efficiency": int(experiments["computational_efficiency"]["expected_measurements"]),
        "explainability_shap": int(experiments["explainability_shap"]["expected_process_runs"]),
        "sensitivity_depth": int(experiments["sensitivity_depth"]["logical_runs"]),
        "sensitivity_pin": int(experiments["sensitivity_pin"]["logical_runs"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and print the final paper experiment registry")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    registry = load_registry(Path(args.registry).resolve())
    output = {
        "protocol_id": registry["protocol_id"],
        "proposed_config": registry["proposed_config"],
        "target_schema": registry["target_schema"],
        "logical_counts": logical_counts(registry),
        "excluded": registry["excluded_from_final_registry"],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
