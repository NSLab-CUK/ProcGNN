#!/usr/bin/env python3
"""Build v4 target provenance decisions and reference patch proposals from validation evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v4_validation_common import MANDATORY_SCAN_TARGETS, PROJECT_ROOT, load_target_formula_table, resolve_path

OK_FORMULA_STATUSES = frozenset({"ok", "ok_with_tolerance"})
TOLERANCE_SCAN = frozenset({"exact_match", "tolerance_match"})


def _load_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _best_scan_row(scan_top: pd.DataFrame, target_id: str) -> pd.Series | None:
    sub = scan_top[scan_top["target_id"].astype(str) == target_id]
    if sub.empty:
        return None
    tol = sub[sub["candidate_status"].isin(TOLERANCE_SCAN)]
    if not tol.empty:
        return tol.sort_values("rank_by_diff_abs_mean").iloc[0]
    return sub.sort_values("rank_by_diff_abs_mean").iloc[0]


def _decide_row(
    formula_row: pd.Series,
    val_row: pd.Series | None,
    ref_row: pd.Series | None,
    scan_best: pd.Series | None,
    *,
    include_streams_only_in_formula_macro: bool = True,
) -> dict[str, Any]:
    tid = str(formula_row["target_id"])
    pid = int(formula_row["process_id"])
    target_feature = str(formula_row["target_feature"])
    declared_stream = str(formula_row["target_stream"])
    val_status = str(val_row.get("status", "")) if val_row is not None else ""
    main_all_nan = val_status == "unresolved_main_column" or (
        val_row is not None and int(val_row.get("n_valid", 0) or 0) == 0 and val_status != "ok_streams_only_no_main"
    )
    if tid == "P08_T003":
        main_all_nan = True

    ref_edge = str(formula_row.get("reference_edge_id", ""))
    ref_sk = str(ref_row.get("canonical_main_data_stream_key", "")) if ref_row is not None else ""
    ans_sk = str(ref_row.get("target_answer_main_data_stream_key", "")) if ref_row is not None else ""

    best_sk = ""
    best_max = float("nan")
    best_mr = float("nan")
    best_r2 = float("nan")
    scan_status = ""
    if scan_best is not None:
        best_sk = str(scan_best.get("candidate_stream_key", ""))
        best_max = float(scan_best.get("diff_abs_max", float("nan")))
        best_mr = float(scan_best.get("mismatch_rate", float("nan")))
        best_r2 = float(scan_best.get("r2_main_vs_calc", float("nan")))
        scan_status = str(scan_best.get("candidate_status", ""))

    declared_ok = val_status in OK_FORMULA_STATUSES
    alternate_ok = scan_status in TOLERANCE_SCAN and best_sk

    decision_status = "unresolved"
    recommended = "inspect_raw_data_definition"
    include_main = False
    include_formula = False
    include_streams_only = False
    exclusion = ""
    main_formula_status = val_status or "not_validated"

    if val_status == "ok_streams_only_no_main" or (formula_row.get("special_case") and tid == "P06_T004"):
        decision_status = "streams_only_no_main"
        recommended = "convert_to_streams_only_target"
        include_streams_only = include_streams_only_in_formula_macro
        exclusion = "no_main_column_for_cross_check"
        main_formula_status = "streams_only_no_main"
    elif main_all_nan:
        decision_status = "main_all_nan_streams_available"
        recommended = "exclude_from_v4_macro_until_label_fixed"
        exclusion = "main_target_column_all_nan"
        if scan_best is not None and scan_status not in ("main_all_nan", "unresolved_candidate_columns"):
            include_streams_only = include_streams_only_in_formula_macro
    elif declared_ok:
        decision_status = (
            "main_verified_ok" if val_status == "ok" else "main_verified_ok_with_tolerance"
        )
        recommended = "keep_current_mapping"
        include_main = True
        include_formula = True
    elif alternate_ok:
        decision_status = "mapping_conflict_exact_alternate_found"
        if tid == "P02_T002" and best_sk.upper() == "PROD":
            recommended = "update_target_stream_to_PROD"
        else:
            recommended = "update_target_stream_to_alternate"
        include_main = False
        include_formula = False
        exclusion = f"declared_stream_mismatch;alternate={best_sk}"
    elif val_status == "mismatch":
        if scan_best is not None and scan_status in TOLERANCE_SCAN:
            decision_status = "mapping_conflict_exact_alternate_found"
            recommended = "update_target_stream_to_alternate"
            exclusion = f"declared_mismatch;alternate={best_sk}"
        else:
            decision_status = "exclude_until_fixed"
            recommended = "exclude_from_v4_macro_until_label_fixed"
            exclusion = "mismatch_no_tolerance_alternate"
    else:
        decision_status = "unresolved"
        recommended = "require_manual_review"
        exclusion = val_status or "unknown"

    # Policy overrides for known cases
    if tid == "P02_T002" and alternate_ok and best_sk.upper() == "PROD":
        decision_status = "mapping_conflict_exact_alternate_found"
        recommended = "update_target_stream_to_PROD"
        exclusion = "declared_OUT_EXHAUST/13;main_matches_PROD"
    if tid in ("P08_T001", "P08_T002"):
        decision_status = "exclude_until_fixed"
        recommended = "require_manual_review"
        include_main = False
        include_formula = False
        exclusion = "no_tolerance_alternate_in_scan"
    if tid == "P08_T003":
        decision_status = "main_all_nan_streams_available"
        recommended = "exclude_from_v4_macro_until_label_fixed"
        include_main = False
        include_formula = False
        include_streams_only = include_streams_only_in_formula_macro
        exclusion = "RESTEAM_H2O_Mole_all_nan_in_main"

    amount_source = "unresolved"
    if include_main:
        amount_source = "main_verified_stream_formula"
    elif include_streams_only:
        amount_source = "streams_only_formula"
    elif decision_status == "mapping_conflict_exact_alternate_found":
        amount_source = "stream_formula_unverified"
    elif decision_status == "exclude_until_fixed":
        amount_source = "excluded_mapping_conflict"

    return {
        "process_id": pid,
        "target_id": tid,
        "target_feature": target_feature,
        "target_species": str(formula_row["target_species"]),
        "declared_target_stream": declared_stream,
        "reference_answer_edge_id": str(ref_row.get("target_answer_edge_id", "")) if ref_row is not None else "",
        "reference_stream_key": ans_sk,
        "canonical_stream_key": ref_sk,
        "canonical_dst_node": str(ref_row.get("dst_node", "")) if ref_row is not None else "",
        "main_formula_status": main_formula_status,
        "validation_status": val_status,
        "matched_stream_key": str(val_row.get("matched_stream_key", "")) if val_row is not None else "",
        "best_candidate_stream_key": best_sk,
        "best_candidate_diff_abs_max": best_max,
        "best_candidate_mismatch_rate": best_mr,
        "best_candidate_r2_main_vs_calc": best_r2,
        "best_candidate_status": scan_status,
        "decision_status": decision_status,
        "include_in_main_verified_macro": include_main,
        "include_in_formula_macro": include_formula,
        "include_in_streams_only_macro": include_streams_only,
        "exclusion_reason": exclusion,
        "recommended_action": recommended,
        "target_provenance": decision_status,
        "formula_status": main_formula_status,
        "amount_source": amount_source,
        "used_stream_key": str(val_row.get("matched_stream_key", "")) if val_row is not None else best_sk,
        "used_formula": str(formula_row.get("formula", "")),
        "scale": float(formula_row.get("scale", 1.0)),
        "reference_edge_id": ref_edge,
    }


def _patch_row(dec: dict[str, Any], formula_row: pd.Series) -> dict[str, Any]:
    tid = dec["target_id"]
    action = "keep"
    risk = "low"
    notes = dec.get("exclusion_reason", "")
    prop_stream = dec["declared_target_stream"]
    prop_edge = dec["reference_edge_id"]
    prop_sk = dec["canonical_stream_key"]
    evidence_type = "validation_summary"

    if dec["decision_status"] == "mapping_conflict_exact_alternate_found":
        action = "update_stream_mapping"
        risk = "medium"
        prop_stream = f"OUT_{dec['best_candidate_stream_key']}" if dec["best_candidate_stream_key"] else prop_stream
        prop_sk = dec["best_candidate_stream_key"]
        evidence_type = "alternate_stream_scan"
        if tid == "P02_T002":
            action = "update_answer_edge"
            notes = "Main PROD_CO2_MoleFlow matches PROD stream; canonical P02_E016/13 conflicts"
            prop_edge = "P02_E014"
            prop_sk = "PROD"
    elif dec["decision_status"] == "streams_only_no_main":
        action = "mark_streams_only"
        risk = "medium"
        notes = "No Main column; stream 17 formula computable only"
    elif dec["decision_status"] == "main_all_nan_streams_available":
        action = "exclude_from_metric"
        risk = "high"
        notes = "Main target all NaN"
    elif dec["decision_status"] == "exclude_until_fixed":
        action = "require_manual_review"
        risk = "high"
    elif dec["recommended_action"] == "keep_current_mapping":
        action = "keep"

    return {
        "process_id": dec["process_id"],
        "target_id": tid,
        "target_feature": dec["target_feature"],
        "current_target_stream": dec["declared_target_stream"],
        "proposed_target_stream": prop_stream,
        "current_answer_edge_id": dec.get("reference_answer_edge_id", dec["reference_edge_id"]),
        "proposed_answer_edge_id": prop_edge,
        "current_stream_key": dec["canonical_stream_key"],
        "proposed_stream_key": prop_sk,
        "evidence_type": evidence_type,
        "evidence_diff_abs_max": dec.get("best_candidate_diff_abs_max", float("nan")),
        "evidence_mismatch_rate": dec.get("best_candidate_mismatch_rate", float("nan")),
        "evidence_r2": dec.get("best_candidate_r2_main_vs_calc", float("nan")),
        "action": action,
        "risk_level": risk,
        "notes": notes,
    }


def _write_decision_report(
    out_dir: Path,
    decisions: pd.DataFrame,
    patches: pd.DataFrame,
    prior: pd.DataFrame,
) -> None:
    n = len(decisions)
    n_main = int(decisions["include_in_main_verified_macro"].sum())
    n_streams = int(decisions["include_in_streams_only_macro"].sum())
    n_mis = int(decisions["decision_status"].astype(str).str.contains("conflict|exclude|unresolved").sum())
    n_excl = n - n_main - n_streams

    lines = [
        "# V4 Target Provenance Decision Report",
        "",
        "## Summary",
        "",
        f"- Total targets: **{n}**",
        f"- include_in_main_verified_macro: **{n_main}**",
        f"- include_in_streams_only_macro: **{n_streams}**",
        f"- Excluded / conflict / unresolved: **{n - n_main}** (streams-only counted separately)",
        "",
        "## Problem targets",
        "",
    ]
    for tid in sorted(MANDATORY_SCAN_TARGETS):
        sub = decisions[decisions["target_id"] == tid]
        if sub.empty:
            continue
        r = sub.iloc[0]
        lines.extend(
            [
                f"### {tid} ({r['target_feature']})",
                f"- decision_status: `{r['decision_status']}`",
                f"- recommended_action: `{r['recommended_action']}`",
                f"- best_candidate_stream: `{r['best_candidate_stream_key']}` "
                f"(diff_abs_max={r['best_candidate_diff_abs_max']}, status={r['best_candidate_status']})",
                f"- include main verified macro: **{r['include_in_main_verified_macro']}**",
                f"- exclusion: {r['exclusion_reason']}",
                "",
            ]
        )

    if not patches.empty:
        chg = patches[patches["action"] != "keep"]
        lines.append("## Reference patch proposals (do not auto-apply)")
        lines.append("")
        for _, p in chg.iterrows():
            lines.append(
                f"- **{p['target_id']}**: {p['action']} ({p['risk_level']}) — "
                f"stream `{p['current_stream_key']}` → `{p['proposed_stream_key']}`; {p['notes']}"
            )

    if not prior.empty:
        lines.extend(["", "## Prior validation CSV", "", "Compared to `data/reference/v4/target_formula_validation.csv`."])

    (out_dir / "V4_TARGET_PROVENANCE_DECISION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _write_policy_report(
    validation_dir: Path,
    decisions: pd.DataFrame,
    patches: pd.DataFrame,
) -> None:
    n = len(decisions)
    n_main = int(decisions["include_in_main_verified_macro"].sum())
    n_formula = int(decisions["include_in_formula_macro"].sum())
    n_streams = int(decisions["include_in_streams_only_macro"].sum())
    excluded_ids = decisions[~decisions["include_in_main_verified_macro"]]["target_id"].tolist()

    lines = [
        "# V4 Mapping and Metric Policy Report",
        "",
        "## 1. Provenance summary",
        "",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total targets | {n} |",
        f"| main_verified (macro eligible) | {n_main} |",
        f"| formula_macro eligible | {n_formula} |",
        f"| streams_only_macro eligible | {n_streams} |",
        f"| Excluded from main verified | {n - n_main} |",
        "",
        "Excluded target_ids from **target_v4_macro_r2_main_verified**:",
        "",
        ", ".join(f"`{x}`" for x in excluded_ids) if excluded_ids else "(none)",
        "",
        "## 2. Problem targets",
        "",
        "### P02_T002 (PROD_CO2_MoleFlow)",
        "",
        "- Declared: OUT_EXHAUST / stream 13 (P02_E016)",
        "- Main matches **PROD** stream (`Mole_Flow * Frac_CO2`) with tolerance",
        "- **recommended_action**: `update_target_stream_to_PROD`",
        "- **Not included** in main_verified macro until mapping is patched",
        "",
        "### P08_T001 / P08_T002",
        "",
        "- Main columns exist but no stream candidate reaches tolerance vs `Mole_Flow * Frac_*`",
        "- **exclude_until_fixed** / require_manual_review",
        "",
        "### P08_T003 (RESTEAM_H2O_Mole)",
        "",
        "- Main column all NaN",
        "- Excluded from main_verified macro",
        "",
        "### P06_T004 (Stream17)",
        "",
        "- No Main columns for Stream17 product",
        "- streams_only_no_main; excluded from main_verified macro",
        "- Optional inclusion in formula_macro via policy flag",
        "",
        "## 3. Metric policy",
        "",
        "| Metric | Definition |",
        "|--------|------------|",
        "| `target_v4_macro_r2_main_verified` | Mean per-target R² where `include_in_main_verified_macro=True` |",
        "| `target_v4_macro_r2_formula_available` | Mean where `include_in_formula_macro=True` |",
        "| `target_v4_macro_r2_all_reported` | Legacy alias (deprecated warning if provenance loaded) |",
        "| `legacy_answer_fraction_macro_r2` | Frac_* on answer edges only; not amount R² |",
        "",
        "**Primary ranking metric for ablations**: `target_v4_macro_r2_main_verified`.",
        "",
        "## 4. Optuna legacy interpretation",
        "",
        "- Historical Process7 `best_value ≈ 0.9076` used **legacy answer fraction R²** "
        "(`answer_targets_r2` / `metric_answer_all_targets_mean_r2`), weighted H2+CO2 Frac only.",
        "- This is **not** `target_v4_macro_r2` on amount targets (P07_T001–T003 mole×frac).",
        "- Existing `study.db` files are **not modified**.",
        "- New runs: `--optuna-objective legacy_answer_fraction_r2` or `target_v4_main_verified_r2`.",
        "",
        "## 5. Re-run commands",
        "",
        "```powershell",
        "python scripts/validate_all_v4_target_formulas.py --out outputs/v4_target_formula_validation_all",
        "python scripts/scan_alternate_target_streams.py --validation-dir outputs/v4_target_formula_validation_all",
        "python scripts/build_v4_target_provenance_decisions.py --validation-dir outputs/v4_target_formula_validation_all",
        "```",
        "",
        "## 6. Files created/updated (this pass)",
        "",
        "- `scripts/scan_alternate_target_streams.py`",
        "- `scripts/build_v4_target_provenance_decisions.py`",
        "- `scripts/v4_validation_common.py`",
        "- `src/process_graph/experiment/target_v4_provenance.py`",
        "- `outputs/.../provenance_decisions/v4_target_provenance_decisions.csv`",
        "- `outputs/.../provenance_decisions/v4_reference_patch_proposal.csv`",
        "",
        "## 7. Manual review still required",
        "",
        "- P08_T001, P08_T002, P08_T003: raw Main target definitions",
        "- Apply `v4_reference_patch_proposal.csv` to reference CSVs manually after review",
        "",
    ]
    if not patches.empty:
        lines.append("## Patch proposals")
        lines.append("")
        for _, p in patches[patches["action"] != "keep"].iterrows():
            lines.append(f"- {p['target_id']}: {p['action']} — {p['notes']}")

    (validation_dir / "V4_MAPPING_AND_METRIC_POLICY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build v4 target provenance decisions.")
    parser.add_argument("--validation-dir", default="outputs/v4_target_formula_validation_all")
    parser.add_argument("--target-ref", default="data/reference/v3/target_answer_edges.csv")
    parser.add_argument("--edge-ref", default="data/reference/v3/canonical_edges.csv")
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--include-streams-only-in-formula-macro",
        action="store_true",
        help="Set include_in_formula_macro for streams-only targets (e.g. P06_T004)",
    )
    args = parser.parse_args()

    validation_dir = resolve_path(args.validation_dir)
    out_dir = resolve_path(args.out or str(validation_dir / "provenance_decisions"))
    out_dir.mkdir(parents=True, exist_ok=True)

    formulas = load_target_formula_table()
    summary = _load_optional(validation_dir / "target_formula_summary.csv")
    ref_check = _load_optional(validation_dir / "reference_mapping_check.csv")
    scan_top = _load_optional(validation_dir / "alternate_stream_scan" / "alternate_stream_scan_top_candidates.csv")
    prior = _load_optional(PROJECT_ROOT / "data/reference/v4/target_formula_validation.csv")

    decisions_rows: list[dict[str, Any]] = []
    for _, frow in formulas.iterrows():
        tid = str(frow["target_id"])
        val_row = summary[summary["target_id"] == tid].iloc[0] if not summary.empty and tid in set(
            summary["target_id"].astype(str)
        ) else None
        ref_row = ref_check[ref_check["target_id"] == tid].iloc[0] if not ref_check.empty and tid in set(
            ref_check["target_id"].astype(str)
        ) else None
        scan_best = _best_scan_row(scan_top, tid)
        decisions_rows.append(
            _decide_row(
                frow,
                val_row,
                ref_row,
                scan_best,
                include_streams_only_in_formula_macro=args.include_streams_only_in_formula_macro,
            )
        )

    decisions = pd.DataFrame(decisions_rows)
    patches = pd.DataFrame([_patch_row(r, formulas[formulas["target_id"] == r["target_id"]].iloc[0]) for r in decisions_rows])

    decisions.to_csv(out_dir / "v4_target_provenance_decisions.csv", index=False)
    patches.to_csv(out_dir / "v4_reference_patch_proposal.csv", index=False)
    _write_decision_report(out_dir, decisions, patches, prior)
    _write_policy_report(validation_dir, decisions, patches)

    meta = {
        "source": str(out_dir / "v4_target_provenance_decisions.csv"),
        "n_main_verified": int(decisions["include_in_main_verified_macro"].sum()),
        "excluded_target_ids": decisions[~decisions["include_in_main_verified_macro"]]["target_id"].tolist(),
    }
    (out_dir / "provenance_policy_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(decisions[["target_id", "decision_status", "include_in_main_verified_macro"]].to_string(index=False))
    print(f"[out] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
