"""Aggregate repeated, constrained GA runs into paper-ready audit tables.

Only candidates satisfying every economic, capacity, and optional CO2 constraint
are compared.  A penalised incumbent is deliberately never substituted for a
feasible result.  The script also exports the three best feasible GA operating
points per process/scenario for independent Aspen re-checking.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize repeated GNN-economic GA study arms.")
    parser.add_argument("--root", required=True, help="Study root containing */*/seed_*/ga_summary.json files.")
    parser.add_argument("--output-dir", default=None, help="Defaults to <root>/paper_summary.")
    parser.add_argument("--top-aspen-candidates", type=int, default=3)
    return parser.parse_args()


def _float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _stat(values: Iterable[float]) -> dict[str, float | int | None]:
    numbers = [float(v) for v in values if math.isfinite(float(v))]
    if not numbers:
        return {"n": 0, "mean": None, "std": None, "median": None, "min": None, "max": None}
    ordered = sorted(numbers)
    midpoint = len(ordered) // 2
    median = ordered[midpoint] if len(ordered) % 2 else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    return {
        "n": len(numbers), "mean": mean(numbers), "std": stdev(numbers) if len(numbers) > 1 else 0.0,
        "median": median, "min": ordered[0], "max": ordered[-1],
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_identity(root: Path, summary_path: Path, payload: dict[str, Any]) -> tuple[str, str, str]:
    """Read scenario/method/replicate from runner layout, with safe manual fallback."""
    relative = summary_path.parent.relative_to(root)
    pieces = relative.parts
    arguments = payload.get("arguments", {})
    scenario = pieces[0] if len(pieces) >= 3 else "manual"
    method = str(arguments.get("search_method", pieces[1] if len(pieces) >= 3 else "unknown"))
    replicate = pieces[2] if len(pieces) >= 3 else f"seed_{arguments.get('seed', 'unknown')}"
    return scenario, method, replicate


def main() -> None:
    args = _args()
    if args.top_aspen_candidates < 1:
        raise ValueError("--top-aspen-candidates must be >= 1.")
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Study root not found: {root}")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else root / "paper_summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    long_rows: list[dict[str, object]] = []
    unsafe_runs: list[str] = []
    provenance_hashes: set[str] = set()
    for summary_path in sorted(root.rglob("ga_summary.json")):
        if output_dir in summary_path.parents:
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        scenario, method, replicate = _run_identity(root, summary_path, payload)
        arguments = payload.get("arguments", {})
        if payload.get("input_policy") != "masked_proxy":
            unsafe_runs.append(str(summary_path))
        checkpoint_hash = ((payload.get("provenance") or {}).get("checkpoint") or {}).get("sha256")
        if checkpoint_hash:
            provenance_hashes.add(str(checkpoint_hash))
        for process in payload.get("processes", []):
            baseline = process.get("baseline", {})
            baseline_objectives = baseline.get("objectives", {})
            baseline_indicators = baseline.get("indicators", {})
            best = process.get("best_feasible")
            result = best.get("result", {}) if isinstance(best, dict) else {}
            objectives = result.get("objectives", {})
            indicators = result.get("indicators", {})
            counts = process.get("evaluation_counts", {})
            baseline_lcoh = _float(baseline_objectives.get("lcoh_usd_per_kg"))
            best_lcoh = _float(best.get("lcoh_usd_per_kg")) if isinstance(best, dict) else None
            row: dict[str, object] = {
                "run_id": str(summary_path.parent.relative_to(root)),
                "scenario": scenario, "search_method": method, "replicate": replicate,
                "seed": arguments.get("seed"), "process_id": process.get("process_id"),
                "input_policy": payload.get("input_policy"),
                "evaluation_budget": counts.get("total"), "feasible_candidates": counts.get("feasible"),
                "feasible_rate": (
                    float(counts["feasible"]) / float(counts["total"])
                    if _float(counts.get("total")) not in (None, 0.0) else None
                ),
                "baseline_lcoh_usd_per_kg": baseline_lcoh,
                "baseline_specific_co2_kg_per_kg_h2": _float(baseline_objectives.get("specific_co2_kg_per_kg_h2")),
                "baseline_h2_production_kg_h": _float(baseline_indicators.get("h2_production_kg_h")),
                "co2_cap_kg_per_kg_h2": _float(process.get("co2_cap_kg_per_kg_h2")),
                "has_feasible_solution": best_lcoh is not None,
                "best_feasible_lcoh_usd_per_kg": best_lcoh,
                "best_feasible_specific_co2_kg_per_kg_h2": _float(objectives.get("specific_co2_kg_per_kg_h2")),
                "best_feasible_h2_production_kg_h": _float(indicators.get("h2_production_kg_h")),
                "lcoh_improvement_pct": (
                    100.0 * (baseline_lcoh - best_lcoh) / baseline_lcoh
                    if baseline_lcoh is not None and baseline_lcoh > 0.0 and best_lcoh is not None else None
                ),
                "candidate_json": json.dumps(best.get("candidate"), ensure_ascii=False) if isinstance(best, dict) else "",
            }
            long_rows.append(row)
    if not long_rows:
        raise RuntimeError(f"No ga_summary.json files found below {root}")
    _write_csv(output_dir / "paper_runs_long.csv", long_rows)

    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in long_rows:
        groups[(str(row["scenario"]), str(row["search_method"]), str(row["process_id"]))].append(row)
    aggregate_rows: list[dict[str, object]] = []
    for (scenario, method, process_id), rows in sorted(groups.items()):
        solved = [row for row in rows if row["has_feasible_solution"]]
        lcoh = _stat([float(row["best_feasible_lcoh_usd_per_kg"]) for row in solved])
        aggregate_rows.append({
            "scenario": scenario, "search_method": method, "process_id": process_id,
            "replicates": len(rows), "runs_with_feasible_solution": len(solved),
            "run_feasibility_rate": len(solved) / len(rows),
            "candidate_feasible_rate_mean": _stat([
                float(row["feasible_rate"]) for row in rows if row["feasible_rate"] is not None
            ])["mean"],
            "evaluation_budget_per_run": rows[0]["evaluation_budget"],
            "baseline_lcoh_usd_per_kg": _stat([
                float(row["baseline_lcoh_usd_per_kg"]) for row in rows if row["baseline_lcoh_usd_per_kg"] is not None
            ])["mean"],
            "best_feasible_lcoh_mean": lcoh["mean"], "best_feasible_lcoh_std": lcoh["std"],
            "best_feasible_lcoh_median": lcoh["median"],
            "lcoh_improvement_pct_mean": _stat([
                float(row["lcoh_improvement_pct"]) for row in solved if row["lcoh_improvement_pct"] is not None
            ])["mean"],
            "specific_co2_mean": _stat([
                float(row["best_feasible_specific_co2_kg_per_kg_h2"])
                for row in solved if row["best_feasible_specific_co2_kg_per_kg_h2"] is not None
            ])["mean"],
        })
    _write_csv(output_dir / "paper_aggregate.csv", aggregate_rows)

    paired: dict[tuple[str, str, object], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in long_rows:
        if row["search_method"] in {"ga", "random"}:
            paired[(str(row["scenario"]), str(row["process_id"]), row["seed"])][str(row["search_method"])] = row
    paired_rows: list[dict[str, object]] = []
    grouped_diffs: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (scenario, process_id, seed), pair in sorted(paired.items()):
        ga = pair.get("ga")
        random = pair.get("random")
        if ga is None or random is None:
            continue
        ga_lcoh, random_lcoh = ga["best_feasible_lcoh_usd_per_kg"], random["best_feasible_lcoh_usd_per_kg"]
        difference = float(ga_lcoh) - float(random_lcoh) if ga_lcoh is not None and random_lcoh is not None else None
        if difference is not None:
            grouped_diffs[(scenario, process_id)].append(difference)
        paired_rows.append({
            "scenario": scenario, "process_id": process_id, "seed": seed,
            "ga_has_feasible_solution": ga["has_feasible_solution"],
            "random_has_feasible_solution": random["has_feasible_solution"],
            "ga_minus_random_lcoh_usd_per_kg": difference,
            "ga_wins_lower_lcoh": difference is not None and difference < 0.0,
        })
    _write_csv(output_dir / "paper_paired_ga_vs_random.csv", paired_rows)

    candidates: list[dict[str, object]] = []
    ga_groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in long_rows:
        if row["search_method"] == "ga" and row["has_feasible_solution"]:
            ga_groups[(str(row["scenario"]), str(row["process_id"]))].append(row)
    for (scenario, process_id), rows in sorted(ga_groups.items()):
        for rank, row in enumerate(sorted(rows, key=lambda item: float(item["best_feasible_lcoh_usd_per_kg"]))[:args.top_aspen_candidates], start=1):
            candidates.append({
                "scenario": scenario, "process_id": process_id, "rank_by_gnn_lcoh": rank,
                "run_id": row["run_id"], "seed": row["seed"],
                "candidate_json": row["candidate_json"],
                "gnn_lcoh_usd_per_kg": row["best_feasible_lcoh_usd_per_kg"],
                "gnn_specific_co2_kg_per_kg_h2": row["best_feasible_specific_co2_kg_per_kg_h2"],
                "gnn_h2_production_kg_h": row["best_feasible_h2_production_kg_h"],
                "aspen_recheck_status": "pending",
            })
    _write_csv(output_dir / "aspen_recheck_candidates.csv", candidates)

    report = {
        "study_root": str(root), "runs": len({row["run_id"] for row in long_rows}),
        "process_run_rows": len(long_rows), "input_policy_required_for_claims": "masked_proxy",
        "unsafe_run_paths": unsafe_runs, "checkpoint_hashes": sorted(provenance_hashes),
        "artifact_files": {
            "run_level": "paper_runs_long.csv", "aggregate": "paper_aggregate.csv",
            "paired_control": "paper_paired_ga_vs_random.csv", "aspen_queue": "aspen_recheck_candidates.csv",
        },
        "interpretation_guard": (
            "Rows are GNN-screened only. Report an optimization result as Aspen-validated only after "
            "the matching operating point in aspen_recheck_candidates.csv has been independently simulated."
        ),
    }
    (output_dir / "paper_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SUMMARY] wrote {output_dir}")
    print(f"[SUMMARY] runs={report['runs']} process-run rows={report['process_run_rows']} Aspen queue={len(candidates)}")
    if unsafe_runs:
        print(f"[WARNING] {len(unsafe_runs)} unsafe-proxy runs found; exclude them from paper claims.")


if __name__ == "__main__":
    main()
