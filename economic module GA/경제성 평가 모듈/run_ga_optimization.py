"""Small, auditable GA over GNN-predicted streams for P01--P10."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gnn_economic import DEFAULT_CHECKPOINT, DEFAULT_RUNTIME_OVERRIDES, GNNEconomicAdapter, process_ids


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a constrained GNN-to-economic GA for one or all SMR processes.")
    parser.add_argument("--processes", default="all", help="all or comma-separated IDs, e.g. P03,P05")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default="configs/experiment/pinn/model_260805_10d_frac1.yaml")
    parser.add_argument("--runtime-overrides", default=None, help="Defaults to the paired final-model runtime_overrides.json.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--template-id", type=int, default=5000)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-method", choices=("ga", "random"), default="ga",
                        help="Evolutionary GA or equal-budget uniform random-search control.")
    parser.add_argument("--include-feed-flow", action="store_true", help="Also vary CH4/WATER/AIR/FUEL scale variables.")
    parser.add_argument("--max-specific-co2", type=float, default=None)
    parser.add_argument("--co2-max-fraction", type=float, default=None,
                        help="CO2 cap as a fraction of this process's masked-GNN template baseline.")
    parser.add_argument("--h2-min-fraction", type=float, default=0.90)
    parser.add_argument("--h2-max-fraction", type=float, default=1.10)
    parser.add_argument("--bound-low-quantile", type=float, default=0.01)
    parser.add_argument("--bound-high-quantile", type=float, default=0.99)
    parser.add_argument("--unsafe-input-proxies", action="store_true", help="Use result/proxy columns as GNN inputs; invalid for optimization.")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _penalty(result: dict[str, Any], *, baseline_h2: float, h2_min_fraction: float,
             h2_max_fraction: float, max_specific_co2: float | None) -> tuple[float, dict[str, float]]:
    violations: dict[str, float] = {}
    for name, status in result["constraints"].items():
        amount = max(0.0, float(status["violation"]))
        if amount:
            violations[str(name)] = amount
    indicators = result["indicators"]
    h2 = float(indicators["h2_production_kg_h"])
    low = baseline_h2 * h2_min_fraction
    high = baseline_h2 * h2_max_fraction
    if h2 < low:
        violations["h2_capacity_low"] = (low - h2) / max(baseline_h2, 1e-8)
    if h2 > high:
        violations["h2_capacity_high"] = (h2 - high) / max(baseline_h2, 1e-8)
    if max_specific_co2 is not None:
        co2 = float(result["objectives"]["specific_co2_kg_per_kg_h2"])
        if not math.isfinite(co2):
            violations["specific_co2_max"] = 1e3
        elif co2 > max_specific_co2:
            violations["specific_co2_max"] = (co2 - max_specific_co2) / max(max_specific_co2, 1e-8)
    return float(sum(violations.values())), violations


def _run_process(adapter: GNNEconomicAdapter, process_id: int, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    variables = adapter.decision_columns(process_id, include_feed_flow=args.include_feed_flow)
    if not variables:
        raise RuntimeError(f"P{process_id:02d}: no variable, non-proxy decision columns were found.")
    bounds = adapter.bounds(
        process_id, variables, low=args.bound_low_quantile, high=args.bound_high_quantile,
    )
    rng = np.random.default_rng(args.seed + process_id)
    base = adapter.evaluate(process_id)
    baseline_h2 = float(base["indicators"]["h2_production_kg_h"])
    if not math.isfinite(baseline_h2) or baseline_h2 <= 1.0:
        raise RuntimeError(f"P{process_id:02d}: GNN baseline has unusable H2 production {baseline_h2}.")
    baseline_co2 = float(base["objectives"]["specific_co2_kg_per_kg_h2"])
    co2_cap = args.max_specific_co2
    if args.co2_max_fraction is not None:
        if not math.isfinite(baseline_co2) or baseline_co2 < 0.0:
            raise RuntimeError(f"P{process_id:02d}: baseline specific CO2 is unusable: {baseline_co2}.")
        co2_cap = baseline_co2 * args.co2_max_fraction
    lower = np.asarray([bounds[v][0] for v in variables], dtype=float)
    upper = np.asarray([bounds[v][1] for v in variables], dtype=float)
    population = rng.uniform(lower, upper, size=(args.population, len(variables)))
    population[0] = np.asarray([float(adapter._template_rows[process_id][v]) for v in variables])
    population[0] = np.clip(population[0], lower, upper)
    history: list[dict[str, Any]] = []

    def assess(vector: np.ndarray, generation: int, index: int) -> tuple[float, dict[str, Any], dict[str, float]]:
        candidate = {name: float(value) for name, value in zip(variables, vector)}
        result = adapter.evaluate(process_id, candidate)
        constraint_sum, violations = _penalty(
            result, baseline_h2=baseline_h2, h2_min_fraction=args.h2_min_fraction,
            h2_max_fraction=args.h2_max_fraction, max_specific_co2=co2_cap,
        )
        lcoh = float(result["objectives"]["lcoh_usd_per_kg"])
        score = lcoh + 1_000.0 * constraint_sum if math.isfinite(lcoh) else 1e12
        history.append({
            "generation": generation, "member": index, "score": score,
            "lcoh_usd_per_kg": lcoh,
            "specific_co2_kg_per_kg_h2": float(result["objectives"]["specific_co2_kg_per_kg_h2"]),
            "h2_production_kg_h": float(result["indicators"]["h2_production_kg_h"]),
            "feasible": not violations, "violation_sum": constraint_sum,
            "violations": violations, "candidate": candidate,
        })
        return score, result, violations

    best_score, best_result, best_vector = float("inf"), None, None
    best_feasible_lcoh, best_feasible_result, best_feasible_vector = float("inf"), None, None
    for generation in range(args.generations + 1):
        scored = [assess(vector, generation, idx) for idx, vector in enumerate(population)]
        scores = np.asarray([entry[0] for entry in scored])
        idx = int(np.argmin(scores))
        if scores[idx] < best_score:
            best_score, best_result, best_vector = float(scores[idx]), scored[idx][1], population[idx].copy()
        for index, (score, result, violations) in enumerate(scored):
            lcoh = float(result["objectives"]["lcoh_usd_per_kg"])
            if not violations and math.isfinite(lcoh) and lcoh < best_feasible_lcoh:
                best_feasible_lcoh = lcoh
                best_feasible_result = result
                best_feasible_vector = population[index].copy()
        if generation == args.generations:
            break
        if args.search_method == "random":
            population = rng.uniform(lower, upper, size=(args.population, len(variables)))
            continue
        elite_idx = np.argsort(scores)[:2]
        children = [population[i].copy() for i in elite_idx]
        while len(children) < args.population:
            pick = rng.integers(0, args.population, size=6)
            p1, p2 = population[pick[:3][np.argmin(scores[pick[:3]])]], population[pick[3:][np.argmin(scores[pick[3:]])]]
            child = np.where(rng.random(len(variables)) < 0.5, p1, p2)
            mutate = rng.random(len(variables)) < 0.25
            child = child + mutate * rng.normal(0.0, 0.10, size=len(variables)) * (upper - lower)
            children.append(np.clip(child, lower, upper))
        population = np.asarray(children[:args.population])
    assert best_result is not None and best_vector is not None
    feasible_rows = [row for row in history if row["feasible"]]
    return {
        "process_id": f"P{process_id:02d}", "baseline": base,
        "variables": variables, "bounds": bounds,
        "co2_cap_kg_per_kg_h2": co2_cap,
        "evaluation_counts": {"total": len(history), "feasible": len(feasible_rows)},
        # Retained for compatibility.  Paper analyses must use best_feasible,
        # never this penalised result when it remains infeasible.
        "best_score": best_score,
        "best_candidate": {name: float(value) for name, value in zip(variables, best_vector)},
        "best_result": best_result,
        "best_feasible": (
            None if best_feasible_result is None else {
                "lcoh_usd_per_kg": best_feasible_lcoh,
                "candidate": {name: float(value) for name, value in zip(variables, best_feasible_vector)},
                "result": best_feasible_result,
            }
        ),
    }, history


def main() -> None:
    args = _args()
    if args.population < 4 or args.generations < 1:
        raise ValueError("population must be >=4 and generations must be >=1.")
    if not 0.0 < args.h2_min_fraction <= args.h2_max_fraction:
        raise ValueError("Require 0 < h2-min-fraction <= h2-max-fraction.")
    if args.max_specific_co2 is not None and args.co2_max_fraction is not None:
        raise ValueError("Use only one of --max-specific-co2 and --co2-max-fraction.")
    if args.max_specific_co2 is not None and args.max_specific_co2 < 0.0:
        raise ValueError("--max-specific-co2 must be non-negative.")
    if args.co2_max_fraction is not None and args.co2_max_fraction <= 0.0:
        raise ValueError("--co2-max-fraction must be positive.")
    if not 0.0 <= args.bound_low_quantile < args.bound_high_quantile <= 1.0:
        raise ValueError("Require 0 <= bound-low-quantile < bound-high-quantile <= 1.")
    processes = process_ids(args.processes)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "economic_ga" / f"ga_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    policy = "unsafe" if args.unsafe_input_proxies else "masked_proxy"
    runtime_overrides = args.runtime_overrides or DEFAULT_RUNTIME_OVERRIDES
    with GNNEconomicAdapter(
        checkpoint=args.checkpoint, config=args.config, runtime_overrides=runtime_overrides,
        device=args.device, template_id=args.template_id, input_policy=policy,
    ) as adapter:
        summaries, all_history = [], []
        audits = []
        for pid in processes:
            audits.append(adapter.input_audit(pid))
            summary, history = _run_process(adapter, pid, args)
            summaries.append(summary)
            all_history.extend([{**row, "process_id": f"P{pid:02d}"} for row in history])
            best_feasible = summary["best_feasible"]
            if best_feasible is None:
                status = "no feasible candidate"
            else:
                status = f"best feasible LCOH={best_feasible['lcoh_usd_per_kg']:.4g}"
            print(
                f"[{args.search_method.upper()}] P{pid:02d} {status}; "
                f"feasible={summary['evaluation_counts']['feasible']}/{summary['evaluation_counts']['total']}",
                flush=True,
            )
        provenance = {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "checkpoint": {"path": str(adapter.checkpoint_path), "sha256": _sha256(adapter.checkpoint_path)},
            "config": {"path": str(adapter.config_path), "sha256": _sha256(adapter.config_path)},
            "runtime_overrides": (
                None if adapter.runtime_overrides_path is None else {
                    "path": str(adapter.runtime_overrides_path), "sha256": _sha256(adapter.runtime_overrides_path),
                }
            ),
            "y_edge_scaler": {"path": str(adapter.y_edge_scaler_path), "sha256": _sha256(adapter.y_edge_scaler_path)},
        }
    with (output_dir / "ga_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe({
            "arguments": vars(args), "input_policy": policy, "input_audits": audits,
            "processes": summaries, "provenance": provenance,
        }), handle, indent=2, ensure_ascii=False)
    rows = []
    for row in all_history:
        flattened = {k: v for k, v in row.items() if k not in {"candidate", "violations"}}
        # Process_Main schemas differ and can contain columns distinguished
        # only by case (for example T_Cool1 vs T_COOL1).  A JSON field keeps
        # one all-process history CSV portable to case-insensitive readers.
        flattened["candidate_json"] = json.dumps(_json_safe(row["candidate"]), ensure_ascii=False)
        flattened["violations_json"] = json.dumps(_json_safe(row["violations"]), ensure_ascii=False)
        rows.append(flattened)
    if rows:
        with (output_dir / "ga_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader(); writer.writerows(rows)
    print(f"[GA] wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
