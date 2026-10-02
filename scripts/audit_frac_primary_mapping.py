#!/usr/bin/env python3
"""Audit edge_all Frac-primary evaluation: reference CSV vs run artifacts (no training)."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from process_graph.experiment.target_metric_names import (  # noqa: E402
    EVAL_PRIMARY_FRAC_R2_BY_PROCESS,
    EVAL_SECONDARY_AMOUNT_R2_BY_PROCESS,
    METRIC_KIND_AMOUNT,
    METRIC_KIND_FRAC,
    resolve_metric_key,
)
from process_graph.experiment.target_v4_metrics import (  # noqa: E402
    SPECIES_TO_FRAC_PROPERTY,
    frac_property_for_species,
    load_target_stream_targets_v4,
)
from process_graph.experiment.target_v4_provenance import load_provenance_policy, resolve_provenance_path

SPECIES_FRAC = {"H2": "Frac_H2", "CO2": "Frac_CO2", "H2O": "Frac_H2O"}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _edge_nodes(canonical_edges: pd.DataFrame, edge_id: str) -> tuple[str, str]:
    sub = canonical_edges[canonical_edges["canonical_edge_id"].astype(str) == edge_id]
    if sub.empty:
        return "", ""
    r = sub.iloc[0]
    return str(r.get("src_node", "")), str(r.get("dst_node", ""))


def audit_reference_rows(
    ref_root: Path,
    *,
    frac_df: pd.DataFrame,
    amt_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    pred_frac_df: pd.DataFrame,
) -> pd.DataFrame:
    targets_path = ref_root / "v4" / "target_stream_targets.csv"
    canon_path = ref_root / "canonical_edges.csv"
    specs = load_target_stream_targets_v4(targets_path)
    flat = [s for plist in specs.values() for s in plist]
    canon = _read_csv(canon_path)
    prov = load_provenance_policy(resolve_provenance_path())

    rows: list[dict[str, Any]] = []
    for spec in flat:
        exp_frac = SPECIES_FRAC.get(spec.target_species, "")
        pid = spec.process_id
        tid = spec.target_id
        is_p06_s17 = tid == "P06_T004" or (
            "Stream17" in spec.target_name and spec.required_stream_key == "17"
        )

        fr = frac_df[frac_df["target_id"].astype(str) == tid] if not frac_df.empty else pd.DataFrame()
        am = amt_df[amt_df["target_id"].astype(str) == tid] if not amt_df.empty else pd.DataFrame()
        mp = mapping_df[mapping_df["target_id"].astype(str) == tid] if not mapping_df.empty else pd.DataFrame()
        pf = (
            pred_frac_df[pred_frac_df["target_id"].astype(str) == tid]
            if not pred_frac_df.empty and "target_id" in pred_frac_df.columns
            else pd.DataFrame()
        )

        actual_frac = ""
        included = ""
        skip = ""
        status = "OK"
        issue = ""

        if fr.empty:
            status = "MISSING_TARGET_ROW"
            issue = "no row in target_metrics_v4_frac.csv"
        else:
            r0 = fr.iloc[0]
            actual_frac = str(r0.get("frac_property", ""))
            included = str(r0.get("included_in_primary", ""))
            skip = str(r0.get("skip_reason", "") or "")
            if actual_frac != exp_frac:
                status = "WRONG_FRAC_PROPERTY"
                issue = f"expected {exp_frac}, got {actual_frac}"
            elif str(r0.get("metric_kind", "")) not in (METRIC_KIND_FRAC, "frac"):
                status = "WRONG_FRAC_PROPERTY"
                issue = f"metric_kind={r0.get('metric_kind')}"

        matched_edge = spec.canonical_edge_id
        if not fr.empty:
            ue = str(fr.iloc[0].get("used_edge_id", fr.iloc[0].get("canonical_answer_edge_id", "")))
            if ue:
                matched_edge = ue
        if not pf.empty and "source_canonical_edge_id" in pf.columns:
            src_edges = set(pf["source_canonical_edge_id"].astype(str).unique())
            if len(src_edges) > 1:
                status = "WRONG_EDGE_SUBSTITUTION" if status == "OK" else status
                issue = (issue + ";").strip(";") + f"multiple source edges: {src_edges}"
            elif len(src_edges) == 1 and matched_edge and src_edges != {matched_edge}:
                only = next(iter(src_edges))
                if is_p06_s17 and only != matched_edge:
                    status = "STREAM17_WRONG_EDGE"
                    issue = f"expected {matched_edge}, predictions use {only}"
                elif not is_p06_s17:
                    status = "WRONG_EDGE_SUBSTITUTION"
                    issue = f"canonical {matched_edge} vs pred {only}"

        frac_uses_mole = False
        frac_uses_scale = False
        amount_uses_mole = False
        amount_uses_scale = False
        if not pf.empty:
            # Frac primary values must equal raw Frac columns on same edge row
            if "target_true_value" in pf.columns and "target_pred_value" in pf.columns:
                pass  # values are Frac-only by construction in compute_target_v4_metrics
        if not am.empty:
            sf = float(pd.to_numeric(am.iloc[0].get("scale_factor", 1), errors="coerce"))
            if sf not in (1.0, 0.8) and spec.process_num == 2 and spec.target_species == "H2":
                pass
        if not am.empty and str(am.iloc[0].get("metric_kind", "")) in (METRIC_KIND_AMOUNT, "amount"):
            sf = float(pd.to_numeric(am.iloc[0].get("scale_factor", 1), errors="coerce"))
            amount_uses_scale = sf != 1.0 or "scale" in str(am.iloc[0].get("used_formula", ""))
            amount_uses_mole = "Mole_Flow" in str(am.iloc[0].get("used_formula", ""))

        if is_p06_s17:
            if fr.empty and mp.empty:
                status = "STREAM17_EDGE_NOT_FOUND"
                issue = "no frac metric row"
            elif included.lower() == "false" and skip == "no_predicted_stream17_edge":
                status = "STREAM17_EDGE_NOT_FOUND"
            elif included.lower() == "false" and "excluded_from_main_verified" in skip:
                status = "EXCLUDED_BY_PROVENANCE"
                issue = "edge may exist but provenance excludes from primary"
            elif matched_edge != "P06_E013":
                status = "STREAM17_WRONG_EDGE"
                issue = f"expected P06_E013, got {matched_edge}"

        if not prov.empty and tid in prov["target_id"].astype(str).values:
            pr = prov[prov["target_id"].astype(str) == tid].iloc[0]
            inc_prov = str(pr.get("include_in_main_verified_macro", "")).lower() in ("true", "1")
            if not inc_prov and included.lower() == "false" and status == "OK":
                status = "EXCLUDED_BY_PROVENANCE"
                issue = str(pr.get("exclusion_reason", "provenance"))

        if not fr.empty:
            r2 = float(pd.to_numeric(fr.iloc[0]["r2"], errors="coerce"))
            if not math.isfinite(r2) and "low_variance" in str(fr.iloc[0].get("status", "")):
                if status == "OK":
                    status = "R2_NAN_TRUE_VARIANCE_ZERO"

        src_n, dst_n = _edge_nodes(canon, matched_edge)
        rows.append(
            {
                "process_id": pid,
                "target_id": tid,
                "target_stream": spec.target_stream_node,
                "target_species": spec.target_species,
                "target_feature": spec.target_name,
                "expected_frac_property": exp_frac,
                "actual_frac_property": actual_frac,
                "expected_amount_formula": spec.target_formula,
                "matched_answer_edge_id": matched_edge,
                "matched_canonical_edge_id": spec.canonical_edge_id,
                "edge_src_node": src_n,
                "edge_dst_node": dst_n,
                "required_stream_key": spec.required_stream_key,
                "scale_factor_ref": spec.scale_factor,
                "used_for_frac_primary": not fr.empty,
                "used_for_amount_secondary": not am.empty,
                "included_in_primary": included,
                "skip_reason": skip,
                "frac_uses_mole_flow": frac_uses_mole,
                "frac_uses_scale_factor": frac_uses_scale,
                "amount_uses_mole_flow": amount_uses_mole,
                "amount_uses_scale_factor": amount_uses_scale,
                "status": status,
                "issue": issue,
            }
        )
    return pd.DataFrame(rows)


def check_run_artifacts(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    out: dict[str, Any] = {"run_dir": str(run_dir)}
    for name in (
        "target_metrics_v4.csv",
        "target_metrics_v4_frac.csv",
        "target_metrics_v4_frac_summary.csv",
        "skipped_targets_v4_frac.csv",
        "metrics_per_epoch.csv",
        "metrics.json",
        "edge_predictions.csv",
    ):
        for rel in (f"test/{name}", name):
            p = run_dir / rel
            out[f"has_{name}"] = p.is_file()
            if p.is_file() and name.endswith(".csv"):
                try:
                    df = pd.read_csv(p, nrows=2)
                    out[f"cols_{name}"] = list(df.columns)[:8]
                except Exception:
                    pass
    ep = run_dir / "metrics_per_epoch.csv"
    if ep.is_file():
        df = pd.read_csv(ep)
        prim_cols = [
            c
            for c in df.columns
            if resolve_metric_key(c.replace("val_", "")) == EVAL_PRIMARY_FRAC_R2_BY_PROCESS
            or "eval_primary_frac_r2_by_process" in c
            or "process_balanced_main_target_frac_r2_main_verified" in c
        ]
        amt_cols = [
            c
            for c in df.columns
            if "eval_secondary_amount" in c or "process_balanced_main_target_r2_main_verified" in c
        ]
        leg_cols = [c for c in df.columns if "legacy" in c and "fraction" in c]
        out["metrics_per_epoch_primary_cols"] = prim_cols
        out["metrics_per_epoch_amount_cols"] = amt_cols
        out["metrics_per_epoch_legacy_cols"] = leg_cols[:3]
    mj = run_dir / "metrics.json"
    if not mj.is_file():
        mj = run_dir / "test" / "metrics.json"
    if mj.is_file():
        try:
            m = json.loads(mj.read_text(encoding="utf-8"))
            out["metrics_json_primary_key"] = [
                k for k in m if EVAL_PRIMARY_FRAC_R2_BY_PROCESS in k or "frac_r2" in k
            ][:5]
        except Exception:
            pass
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Frac-primary target-row evaluation.")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--reference-root", type=str, default="data/reference")
    parser.add_argument("--output-csv", type=str, default="outputs/frac_primary_mapping_audit.csv")
    args = parser.parse_args()

    ref_root = (PROJECT_ROOT / args.reference_root).resolve()
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None

    frac_df = amt_df = mapping_df = pred_frac_df = pd.DataFrame()
    if run_dir:
        for base, var in (
            ("target_metrics_v4_frac.csv", "frac"),
            ("target_metrics_v4.csv", "amt"),
            ("target_mapping_v4.csv", "map"),
            ("target_predictions_v4.csv", "pred"),
        ):
            for rel in (f"test/{base}", base):
                p = run_dir / rel
                if p.is_file():
                    df = _read_csv(p)
                    if var == "frac":
                        frac_df = df
                    elif var == "amt":
                        amt_df = df
                    elif var == "map":
                        mapping_df = df
                    else:
                        pred_frac_df = df
                    break

    audit_df = audit_reference_rows(
        ref_root,
        frac_df=frac_df,
        amt_df=amt_df,
        mapping_df=mapping_df,
        pred_frac_df=pred_frac_df,
    )

    out_path = (PROJECT_ROOT / args.output_csv).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(out_path, index=False)

    n_ok = int((audit_df["status"] == "OK").sum())
    n_total = len(audit_df)
    print(f"Wrote {out_path} ({n_total} rows, {n_ok} OK)")
    if run_dir:
        art = check_run_artifacts(run_dir)
        print("Run artifacts:", json.dumps(art, indent=2))
    print("\nStatus counts:")
    print(audit_df["status"].value_counts().to_string())
    p06 = audit_df[audit_df["target_id"] == "P06_T004"]
    if not p06.empty:
        print("\nP06_T004:")
        print(p06.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
