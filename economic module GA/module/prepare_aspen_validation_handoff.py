#!/usr/bin/env python3
"""Create a self-contained Aspen-validation handoff from a completed GA study.

The handoff intentionally preserves the original GA summaries and creates
human-readable, flattened operating-condition tables.  It never presents a
GNN-screened result as Aspen-validated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_DIRNAME = "paper_summary"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", required=True, help="Completed GA study root.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <study-root>/aspen_validation_handoff. Must not already exist.",
    )
    return parser.parse_args()


def _process_sort_key(value: object) -> int:
    text = str(value).strip().upper()
    return int(text.removeprefix("P"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty JSON object.")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must decode to an object.")
    return parsed


def _float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _load_run_payloads(study_root: Path, candidates: pd.DataFrame) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for run_id in sorted(candidates["run_id"].astype(str).unique()):
        summary = study_root / run_id / "ga_summary.json"
        if not summary.is_file():
            raise FileNotFoundError(f"Candidate source summary is missing: {summary}")
        payload = json.loads(summary.read_text(encoding="utf-8"))
        if payload.get("input_policy") != "masked_proxy":
            raise ValueError(f"{summary}: expected masked_proxy input policy.")
        payloads[run_id] = payload
    return payloads


def _process_summary(payload: dict[str, Any], process_id: str) -> dict[str, Any]:
    matches = [row for row in payload.get("processes", []) if str(row.get("process_id")) == process_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one {process_id} process summary, found {len(matches)}.")
    result = matches[0]
    if not isinstance(result.get("best_feasible"), dict):
        raise ValueError(f"{process_id} has no feasible solution in a queued run.")
    return result


def _template_conditions(
    process_id: str,
    run_payload: dict[str, Any],
    template_id: int,
) -> tuple[pd.Series, list[str], str]:
    number = _process_sort_key(process_id)
    source = PROJECT_ROOT / "data" / "main_data" / f"{number}.Process_Main.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Process template source is missing: {source}")
    frame = pd.read_csv(source)
    selected = frame.loc[pd.to_numeric(frame["ID"], errors="coerce").eq(template_id)]
    if len(selected) != 1:
        raise ValueError(f"{source}: expected exactly one template ID={template_id}, found {len(selected)}.")
    audits = [row for row in run_payload.get("input_audits", []) if str(row.get("process_id")) == process_id]
    if len(audits) != 1:
        raise ValueError(f"{process_id}: expected exactly one input audit.")
    safe_columns = [str(value) for value in audits[0].get("safe_condition_columns", [])]
    missing = sorted(set(safe_columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{process_id}: safe condition columns absent from template: {missing}")
    return selected.iloc[0], safe_columns, source.relative_to(PROJECT_ROOT).as_posix()


def _candidate_id(row: pd.Series) -> str:
    return (
        f"{str(row['process_id']).upper()}_"
        f"rank{int(row['rank_by_gnn_lcoh']):02d}_"
        f"seed{int(row['seed'])}"
    )


def _readme(
    *,
    study_root: Path,
    package_name: str,
    candidate_count: int,
    priority_count: int,
    backup_count: int,
    no_feasible: list[str],
    template_id: int,
    generations: int,
    population: int,
    seeds: list[int],
) -> str:
    no_feasible_text = ", ".join(no_feasible) if no_feasible else "None"
    return f"""# Aspen validation handoff

## Status and scope

This package contains **GNN-screened GA candidates only**.  It is not Aspen-validated.
Do not report the GNN LCOH, specific-CO2, H2-production, feasibility, or improvement
as a final optimization result until the matching Aspen case has converged and been
checked independently.

- Source GA root: `{study_root.as_posix()}`
- Package directory: `{package_name}`
- Input policy: `masked_proxy` (not `unsafe`)
- Template operating row: `ID={template_id}` for every process
- GA configuration: population={population}, generations={generations}, seeds={seeds}
- Per process/run budget: {population} x ({generations} + 1) = {population * (generations + 1)} GNN/economic evaluations
- Aspen queue: {candidate_count} candidates ({priority_count} rank-1 priority; {backup_count} rank-2/3 backup)
- No GNN-feasible candidate: {no_feasible_text}

## Files to use

1. `01_priority_rank1_candidates.csv`: validate these one candidate per eligible process first.
2. `02_backup_rank2_rank3_candidates.csv`: use when the rank-1 case fails to converge,
   violates a constraint, or its Aspen ranking is not retained.
3. `03_all_candidate_operating_conditions_wide.csv`: all safe Process_Main operating
   conditions after applying the GA candidate.  Blank cells belong to another process.
4. `04_all_candidate_decision_variables_long.csv`: only the variables changed by GA,
   with the template value and delta for an audit trail.
5. `05_gnn_predicted_indicators.csv` and `06_gnn_constraint_checklist.csv`: GNN-side
   expectations to compare with Aspen.  They are reference values, not validation.
6. `07_aspen_results_template.csv`: fill this file after each Aspen run.  Preserve the
   candidate ID exactly so results can be joined back to the GA record.
7. `08_ga_aggregate_all_processes.csv`: full 10-process GA outcome, including the
   process without a feasible GNN-screened candidate.
8. `Aspen_validation_handoff.xlsx`: the same handoff tables in one workbook.

## Aspen re-check procedure

1. Open the original Aspen model for the indicated process and begin from its matching
   template/base case.  Apply every listed value in the candidate operating-condition
   table using the **native Aspen units** of that case.
2. GA changed only the variables listed in file 04.  The other safe operating conditions
   are held at their `ID={template_id}` template values in file 03.  Keep unlisted Aspen
   specifications at the original template/base-case settings.
3. Converge the model, export the required stream results, and independently recompute
   LCOH, specific CO2, H2 production, and all process constraints under the same
   economic assumptions.  Record actual values and pass/fail status in file 07.
4. Record non-convergence, operating-range violations, or ranking changes explicitly.
   A failed or non-converged candidate must not be called optimized.

## Important limits

- Column names are original `Process_Main` operating-condition names.  Units were not
  stored in the GA artifact; confirm each name and unit against the original Aspen case.
- Feed-flow scaling was disabled (`include_feed_flow=false`); feed values therefore stay
  at the template values supplied in file 03.
- P03 has no feasible GNN-screened point in any of the three long-GA seeds and is not
  included in the Aspen queue.
- Source summaries and GA provenance are preserved unchanged in `provenance/`.
"""


def main() -> None:
    args = _args()
    study_root = Path(args.study_root).resolve()
    if not study_root.is_dir():
        raise FileNotFoundError(study_root)
    output = Path(args.output_dir).resolve() if args.output_dir else study_root / "aspen_validation_handoff"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing handoff: {output}")

    summary_dir = study_root / SUMMARY_DIRNAME
    candidates_path = summary_dir / "aspen_recheck_candidates.csv"
    aggregate_path = summary_dir / "paper_aggregate.csv"
    run_long_path = summary_dir / "paper_runs_long.csv"
    report_path = summary_dir / "paper_report.json"
    required = [candidates_path, aggregate_path, run_long_path, report_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required paper-summary files: {missing}")

    candidates = pd.read_csv(candidates_path)
    required_candidate_columns = {
        "process_id", "rank_by_gnn_lcoh", "run_id", "seed", "candidate_json",
        "gnn_lcoh_usd_per_kg", "gnn_specific_co2_kg_per_kg_h2", "gnn_h2_production_kg_h",
    }
    absent = required_candidate_columns.difference(candidates.columns)
    if absent:
        raise ValueError(f"Aspen candidate queue is missing columns: {sorted(absent)}")
    candidates["process_id"] = candidates["process_id"].astype(str)
    candidates["rank_by_gnn_lcoh"] = pd.to_numeric(candidates["rank_by_gnn_lcoh"], errors="raise").astype(int)
    candidates["seed"] = pd.to_numeric(candidates["seed"], errors="raise").astype(int)
    candidates = candidates.sort_values(
        ["process_id", "rank_by_gnn_lcoh", "seed"], key=lambda series: (
            series.map(_process_sort_key) if series.name == "process_id" else series
        )
    ).reset_index(drop=True)
    if candidates.duplicated(["process_id", "rank_by_gnn_lcoh"]).any():
        raise ValueError("Aspen queue has duplicate process/rank candidates.")

    runs = pd.read_csv(run_long_path)
    required_run_columns = {
        "process_id", "run_id", "seed", "baseline_h2_production_kg_h",
        "baseline_lcoh_usd_per_kg", "has_feasible_solution",
    }
    absent = required_run_columns.difference(runs.columns)
    if absent:
        raise ValueError(f"Run-level table is missing columns: {sorted(absent)}")
    runs["process_id"] = runs["process_id"].astype(str)
    runs["seed"] = pd.to_numeric(runs["seed"], errors="raise").astype(int)
    queue = candidates.merge(
        runs,
        on=["process_id", "run_id", "seed"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_run"),
    )
    if queue["baseline_h2_production_kg_h"].isna().any():
        raise ValueError("Could not join one or more queued candidates to run-level metadata.")
    if not queue["has_feasible_solution"].fillna(False).astype(bool).all():
        raise ValueError("Aspen queue contains a run without a feasible GNN candidate.")

    payloads = _load_run_payloads(study_root, queue)
    first_payload = payloads[str(queue.iloc[0]["run_id"])]
    arguments = first_payload.get("arguments", {})
    expected = {
        "template_id": int(arguments.get("template_id", -1)),
        "population": int(arguments.get("population", -1)),
        "generations": int(arguments.get("generations", -1)),
        "include_feed_flow": bool(arguments.get("include_feed_flow", True)),
    }
    if expected["template_id"] < 0 or expected["population"] < 1 or expected["generations"] < 0:
        raise ValueError("Could not read GA configuration from the first run summary.")
    for run_id, payload in payloads.items():
        run_args = payload.get("arguments", {})
        observed = {
            "template_id": int(run_args.get("template_id", -1)),
            "population": int(run_args.get("population", -1)),
            "generations": int(run_args.get("generations", -1)),
            "include_feed_flow": bool(run_args.get("include_feed_flow", True)),
        }
        if observed != expected:
            raise ValueError(f"Inconsistent GA arguments in {run_id}: {observed} != {expected}")
    if expected["include_feed_flow"]:
        raise ValueError("This handoff expects the fixed-feed-flow GA study, but include_feed_flow=true.")

    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()

    queue_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    operating_rows: list[dict[str, Any]] = []
    indicator_rows: list[dict[str, Any]] = []
    constraint_rows: list[dict[str, Any]] = []

    for _, row in queue.iterrows():
        process_id = str(row["process_id"])
        run_id = str(row["run_id"])
        process = _process_summary(payloads[run_id], process_id)
        candidate = _json_mapping(row["candidate_json"], label=f"{process_id}/{run_id} candidate_json")
        best = process["best_feasible"]
        best_candidate = best.get("candidate", {})
        if candidate != best_candidate:
            raise ValueError(f"{process_id}/{run_id}: queue candidate differs from ga_summary best_feasible.")
        template, safe_columns, template_source = _template_conditions(
            process_id, payloads[run_id], expected["template_id"]
        )
        candidate_id = _candidate_id(row)
        tier = "priority_rank_1" if int(row["rank_by_gnn_lcoh"]) == 1 else "backup_rank_2_or_3"
        base = {
            "candidate_id": candidate_id,
            "validation_tier": tier,
            "process_id": process_id,
            "rank_by_gnn_lcoh": int(row["rank_by_gnn_lcoh"]),
            "seed": int(row["seed"]),
            "source_run_id": run_id,
            "template_id": expected["template_id"],
            "template_source": template_source,
            "gnn_lcoh_usd_per_kg": _float(row["gnn_lcoh_usd_per_kg"]),
            "gnn_specific_co2_kg_per_kg_h2": _float(row["gnn_specific_co2_kg_per_kg_h2"]),
            "gnn_h2_production_kg_h": _float(row["gnn_h2_production_kg_h"]),
            "baseline_h2_production_kg_h": _float(row["baseline_h2_production_kg_h"]),
            "baseline_lcoh_usd_per_kg": _float(row["baseline_lcoh_usd_per_kg"]),
        }
        queue_rows.append({**base, "optimized_variable_count": len(candidate)})

        for name in sorted(candidate):
            template_value = _float(template[name])
            candidate_value = _float(candidate[name])
            decision_rows.append({
                **base,
                "variable_name": str(name),
                "template_value": template_value,
                "candidate_value": candidate_value,
                "candidate_minus_template": (
                    candidate_value - template_value
                    if candidate_value is not None and template_value is not None else None
                ),
                "units": "confirm against original Aspen case (not stored in GA artifact)",
            })
        for name in safe_columns:
            template_value = _float(template[name])
            candidate_value = _float(candidate.get(name, template[name]))
            operating_rows.append({
                **base,
                "variable_name": str(name),
                "application_role": "optimized_by_ga" if name in candidate else "fixed_template",
                "template_value": template_value,
                "candidate_value": candidate_value,
                "units": "confirm against original Aspen case (not stored in GA artifact)",
            })

        result = best.get("result", {})
        indicators = result.get("indicators", {})
        indicator_record = dict(base)
        for name, value in indicators.items():
            numeric = _float(value)
            if numeric is not None:
                indicator_record[f"gnn_{name}"] = numeric
        indicator_rows.append(indicator_record)

        for name, status in result.get("constraints", {}).items():
            constraint_rows.append({
                **base,
                "constraint_name": str(name),
                "constraint_kind": "economic_module",
                "gnn_expected_value": None,
                "gnn_lower_bound": None,
                "gnn_upper_bound": None,
                "gnn_ok": bool(status.get("ok")),
                "gnn_violation": _float(status.get("violation")),
                "aspen_ok": "PENDING",
                "aspen_value": None,
                "aspen_notes": "",
            })
        baseline_h2 = float(base["baseline_h2_production_kg_h"])
        h2_value = _float(indicators.get("h2_production_kg_h"))
        for name, lower, upper in (
            ("h2_production_lower_bound", 0.9 * baseline_h2, None),
            ("h2_production_upper_bound", None, 1.1 * baseline_h2),
        ):
            is_ok = (h2_value >= lower) if lower is not None else (h2_value <= upper)
            constraint_rows.append({
                **base,
                "constraint_name": name,
                "constraint_kind": "ga_h2_production_window",
                "gnn_expected_value": h2_value,
                "gnn_lower_bound": lower,
                "gnn_upper_bound": upper,
                "gnn_ok": bool(is_ok),
                "gnn_violation": 0.0 if is_ok else None,
                "aspen_ok": "PENDING",
                "aspen_value": None,
                "aspen_notes": "",
            })

    queue_frame = pd.DataFrame(queue_rows).sort_values(
        ["process_id", "rank_by_gnn_lcoh"], key=lambda series: (
            series.map(_process_sort_key) if series.name == "process_id" else series
        )
    ).reset_index(drop=True)
    priority = queue_frame.loc[queue_frame["rank_by_gnn_lcoh"].eq(1)].copy()
    backup = queue_frame.loc[queue_frame["rank_by_gnn_lcoh"].gt(1)].copy()
    decisions = pd.DataFrame(decision_rows)
    operating = pd.DataFrame(operating_rows)
    indicators = pd.DataFrame(indicator_rows)
    constraints = pd.DataFrame(constraint_rows)

    meta_columns = list(queue_frame.columns)
    operating_wide = operating.pivot(index="candidate_id", columns="variable_name", values="candidate_value").reset_index()
    operating_wide = queue_frame.merge(operating_wide, on="candidate_id", how="left", validate="one_to_one")

    result_template = queue_frame.copy()
    result_template["aspen_run_status"] = "PENDING"
    result_template["aspen_case_file"] = ""
    result_template["aspen_units_confirmed"] = "PENDING"
    result_template["aspen_converged"] = "PENDING"
    result_template["aspen_lcoh_usd_per_kg"] = None
    result_template["aspen_specific_co2_kg_per_kg_h2"] = None
    result_template["aspen_h2_production_kg_h"] = None
    result_template["aspen_constraints_all_pass"] = "PENDING"
    result_template["lcoh_aspen_minus_gnn"] = None
    result_template["specific_co2_aspen_minus_gnn"] = None
    result_template["h2_production_aspen_minus_gnn"] = None
    result_template["aspen_rank_within_process"] = None
    result_template["final_candidate_decision"] = "PENDING"
    result_template["aspen_notes"] = ""

    aggregate = pd.read_csv(aggregate_path)
    aggregate["process_id"] = aggregate["process_id"].astype(str)
    aggregate["aspen_handoff_status"] = aggregate["process_id"].map(
        lambda process: "rank_1_priority_plus_rank_2_3_backup" if process in set(queue_frame["process_id"])
        else "no_gnn_feasible_candidate__no_aspen_case_requested"
    )
    aggregate = aggregate.sort_values("process_id", key=lambda series: series.map(_process_sort_key)).reset_index(drop=True)
    no_feasible = aggregate.loc[
        pd.to_numeric(aggregate["runs_with_feasible_solution"], errors="coerce").fillna(0).eq(0)
    ].copy()

    _write_csv(priority, output / "01_priority_rank1_candidates.csv")
    _write_csv(backup, output / "02_backup_rank2_rank3_candidates.csv")
    _write_csv(operating_wide, output / "03_all_candidate_operating_conditions_wide.csv")
    _write_csv(decisions, output / "04_all_candidate_decision_variables_long.csv")
    _write_csv(indicators, output / "05_gnn_predicted_indicators.csv")
    _write_csv(constraints, output / "06_gnn_constraint_checklist.csv")
    _write_csv(result_template, output / "07_aspen_results_template.csv")
    _write_csv(aggregate, output / "08_ga_aggregate_all_processes.csv")
    _write_csv(no_feasible, output / "09_no_gnn_feasible_candidate.csv")

    with pd.ExcelWriter(output / "Aspen_validation_handoff.xlsx", engine="openpyxl") as writer:
        priority.to_excel(writer, sheet_name="01_Priority_rank1", index=False)
        backup.to_excel(writer, sheet_name="02_Backup_rank2_3", index=False)
        operating_wide.to_excel(writer, sheet_name="03_Operating_conditions", index=False)
        decisions.to_excel(writer, sheet_name="04_Decision_variables", index=False)
        indicators.to_excel(writer, sheet_name="05_GNN_indicators", index=False)
        constraints.to_excel(writer, sheet_name="06_GNN_constraints", index=False)
        result_template.to_excel(writer, sheet_name="07_Aspen_results", index=False)
        aggregate.to_excel(writer, sheet_name="08_GA_summary", index=False)
        no_feasible.to_excel(writer, sheet_name="09_No_feasible", index=False)

    source_files = [candidates_path, aggregate_path, run_long_path, report_path]
    for source in source_files:
        shutil.copy2(source, provenance / source.name)
    for run_id in sorted(payloads):
        source = study_root / run_id / "ga_summary.json"
        shutil.copy2(source, provenance / f"{run_id.replace('/', '_')}_ga_summary.json")

    manifest = {
        "study_root": str(study_root),
        "input_policy": "masked_proxy",
        "template_id": expected["template_id"],
        "include_feed_flow": expected["include_feed_flow"],
        "population": expected["population"],
        "generations": expected["generations"],
        "seeds": sorted(int(value) for value in queue_frame["seed"].unique()),
        "candidate_count": int(len(queue_frame)),
        "priority_candidate_count": int(len(priority)),
        "backup_candidate_count": int(len(backup)),
        "no_feasible_processes": no_feasible["process_id"].astype(str).tolist(),
        "source_sha256": {source.name: _sha256(source) for source in source_files},
        "interpretation_guard": (
            "All candidate predictions are GNN-screened only. Aspen results must be recorded "
            "before an optimization claim or improvement value is reported."
        ),
    }
    (output / "handoff_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        _readme(
            study_root=study_root,
            package_name=output.name,
            candidate_count=len(queue_frame),
            priority_count=len(priority),
            backup_count=len(backup),
            no_feasible=no_feasible["process_id"].astype(str).tolist(),
            template_id=expected["template_id"],
            generations=expected["generations"],
            population=expected["population"],
            seeds=sorted(int(value) for value in queue_frame["seed"].unique()),
        ),
        encoding="utf-8",
    )
    print(
        f"handoff={output}; candidates={len(queue_frame)}; "
        f"priority={len(priority)}; backup={len(backup)}; no_feasible={len(no_feasible)}"
    )


if __name__ == "__main__":
    main()
