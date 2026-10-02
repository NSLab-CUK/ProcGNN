#!/usr/bin/env python3
"""Audit H2O v4 target routing, loss weighting, and Main vs Streams formula alignment."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from process_graph.constants import STREAM_EDGE_FEATURE_SLOTS  # noqa: E402
from process_graph.experiment.target_v4_metrics import (  # noqa: E402
    SPECIES_TO_FRAC_PROPERTY,
    load_target_stream_targets_v4,
    normalize_stream_key,
)
from process_graph.experiment.v4_target_edge_weighting import (  # noqa: E402
    build_v4_target_loss_weight_tensor,
    load_v4_target_weight_rows,
)
from process_graph.experiment.train_utils import resolve_answer_edge_species_weights  # noqa: E402
import v4_validation_common as vc  # noqa: E402

V4_TARGETS = PROJECT_ROOT / "data/reference/v4/target_stream_targets.csv"
PROVENANCE = (
    PROJECT_ROOT
    / "outputs/v4_target_formula_validation_all/provenance_decisions/v4_target_provenance_decisions.csv"
)


def _r2(y: pd.Series, x: pd.Series) -> float:
    yv = pd.to_numeric(y, errors="coerce")
    xv = pd.to_numeric(x, errors="coerce")
    m = yv.notna() & xv.notna()
    if int(m.sum()) < 2:
        return float("nan")
    yt = yv[m].astype(float)
    xt = xv[m].astype(float)
    ss_res = float(((yt - xt) ** 2).sum())
    mean_y = float(yt.mean())
    ss_tot = float(((yt - mean_y) ** 2).sum())
    if ss_tot <= 1e-30:
        return float("nan")
    return max(-1.0, min(1.0, 1.0 - ss_res / ss_tot))


def _compare_main_calc(main: pd.DataFrame, streams: pd.DataFrame, main_col: str, stream_key: str) -> dict[str, Any]:
    sid = vc.sample_col(main)
    if sid is None:
        raise ValueError("Main CSV missing sample ID column")
    main = main.copy()
    main[sid] = main[sid].astype(str)
    calc_df = vc.extract_stream_calc(streams, stream_key, "Frac_H2O", 1.0)
    if calc_df.empty or main_col not in main.columns:
        return {"status": "unresolved_stream_column", "n_samples": 0}
    calc_df = calc_df.rename(columns={"sample_id": sid}).copy()
    calc_df[sid] = calc_df[sid].astype(str)
    joined = main[[sid, main_col]].merge(calc_df, on=sid, how="inner")
    main_v = pd.to_numeric(joined[main_col], errors="coerce")
    calc_v = pd.to_numeric(joined["calc_value"], errors="coerce")
    valid = main_v.notna() & calc_v.notna()
    n = int(valid.sum())
    if n == 0:
        return {"status": "no_valid_samples", "n_samples": 0}
    diff = (main_v[valid] - calc_v[valid]).abs()
    rel = diff / (main_v[valid].abs() + 1e-12)
    tol = vc.ABS_TOL + vc.REL_TOL * main_v[valid].abs()
    mismatch = float((diff > tol).mean())
    status = "ok" if mismatch < 0.01 else "mismatch"
    return {
        "status": status,
        "n_samples": n,
        "main_mean": float(main_v[valid].mean()),
        "calc_mean": float(calc_v[valid].mean()),
        "main_std": float(main_v[valid].std(ddof=0)),
        "calc_std": float(calc_v[valid].std(ddof=0)),
        "diff_abs_mean": float(diff.mean()),
        "diff_abs_max": float(diff.max()),
        "rel_diff_mean": float(rel.mean()),
        "rel_diff_max": float(rel.max()),
        "corr": float(main_v[valid].corr(calc_v[valid])),
        "r2_main_vs_calc": _r2(main_v[valid], calc_v[valid]),
        "mismatch_rate": mismatch,
    }


def _edge_row_from_canonical(edges: pd.DataFrame, edge_id: str) -> dict[str, Any]:
    sub = edges[edges["canonical_edge_id"].astype(str) == edge_id]
    if sub.empty:
        return {}
    r = sub.iloc[0]
    return {
        "edge_id": edge_id,
        "src_node": str(r.get("src_node", "")),
        "dst_node": str(r.get("dst_node", "")),
        "main_data_stream_key": normalize_stream_key(r.get("main_data_stream_key")),
        "stream_role": str(r.get("stream_role", "")),
        "dst_is_v_output": str(r.get("dst_node", "")) == "V_OUTPUT",
        "needs_review": str(r.get("needs_review", r.get("mapping_note", ""))),
    }


def write_loss_routing_trace(out_dir: Path) -> None:
    rows = [
        {
            "file": "configs/train/process_train.yaml",
            "function_or_class": "TrainConfig",
            "variable_name": "answer_edge_weight_h2o",
            "config_key": "train.answer_edge_weight_h2o",
            "is_read": "yes",
            "is_passed": "yes",
            "is_used_in_loss": "conditional",
            "applied_edge_source": "v4 canonical_edge_id + stream when use_v4_target_edge_weighting=true",
            "applied_feature_slot": "Frac_H2O (target_rows mode)",
            "applies_to_v4_target_edge": "yes",
            "notes": "Dead if all species weights <= 1.0",
        },
        {
            "file": "src/process_graph/experiment/v4_target_edge_weighting.py",
            "function_or_class": "build_v4_target_loss_weight_tensor",
            "variable_name": "edge_stream_loss_weight",
            "config_key": "train.use_v4_target_edge_weighting",
            "is_read": "yes",
            "is_passed": "yes",
            "is_used_in_loss": "yes",
            "applied_edge_source": "target_stream_targets canonical_answer_edge_id",
            "applied_feature_slot": "species Frac_* only in target_rows mode",
            "applies_to_v4_target_edge": "yes",
            "notes": "P07: P07_E017/RE/Frac_H2O not P07_E016",
        },
        {
            "file": "scripts/train_process_surrogate.py",
            "function_or_class": "_apply_v4_target_loss_weights",
            "variable_name": "target_masks[edge_stream_loss_weight]",
            "config_key": "train.*",
            "is_read": "yes",
            "is_passed": "yes",
            "is_used_in_loss": "yes",
            "applied_edge_source": "batch.edge_export_meta",
            "applied_feature_slot": "via v4_target_edge_weighting",
            "applies_to_v4_target_edge": "yes",
            "notes": "Called each training/val batch edge_all",
        },
        {
            "file": "src/process_graph/experiment/train_utils.py",
            "function_or_class": "compute_edge_all_training_loss",
            "variable_name": "element_weight",
            "config_key": "edge_stream_loss_weight tensor",
            "is_read": "yes",
            "is_passed": "yes",
            "is_used_in_loss": "yes",
            "applied_edge_source": "masked_edge_regression_loss",
            "applied_feature_slot": "per-column",
            "applies_to_v4_target_edge": "yes",
            "notes": "Base edge_all loss on all edges; weights multiply elements",
        },
        {
            "file": "src/process_graph/experiment/train_utils.py",
            "function_or_class": "evaluate_edge_all_epoch",
            "variable_name": "metric_target_r2 / extras Frac_H2O",
            "config_key": "answer_edge_pos",
            "is_read": "yes",
            "is_passed": "n/a",
            "is_used_in_loss": "no",
            "applied_edge_source": "target_answer_edges PROD/EX answer slots",
            "applied_feature_slot": "Frac_H2 legacy + extras all Frac_*",
            "applies_to_v4_target_edge": "no",
            "notes": "Legacy Optuna objective; NOT P07_T003 RE edge",
        },
        {
            "file": "scripts/tune_optuna.py",
            "function_or_class": "_objective_series_and_name",
            "variable_name": "answer_edge_weight_h2o",
            "config_key": "optuna_objective",
            "is_read": "yes",
            "is_passed": "stored only",
            "is_used_in_loss": "no",
            "applied_edge_source": "n/a for legacy objective",
            "applied_feature_slot": "n/a",
            "applies_to_v4_target_edge": "no",
            "notes": "legacy_answer_fraction_r2 uses H2+CO2 only; h2o in best_params not in objective",
        },
        {
            "file": "src/process_graph/experiment/target_v4_metrics.py",
            "function_or_class": "compute_target_v4_metrics_from_original_scale",
            "variable_name": "canonical_edge_id filter",
            "config_key": "v4 target table",
            "is_read": "yes",
            "is_passed": "n/a",
            "is_used_in_loss": "no",
            "applied_edge_source": "P07_E017 for P07_T003",
            "applied_feature_slot": "Mole_Flow * Frac_H2O amount",
            "applies_to_v4_target_edge": "yes",
            "notes": "Metrics not loss",
        },
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "h2o_loss_routing_trace.csv", index=False)
    md = [
        "# H2O loss routing audit",
        "",
        "## Summary",
        "",
        "- `answer_edge_weight_h2o` is **not a dead parameter** when weights > 1 and `use_v4_target_edge_weighting=true`.",
        "- It applies in **training loss** via `edge_stream_loss_weight` on the v4 target canonical edge **Frac_H2O** slot.",
        "- It does **not** enter legacy Optuna `answer_targets_r2` (H2+CO2 weighted Frac on answer slots).",
        "- Legacy `answer_target_Frac_H2O_r2` on PROD answer edge is auxiliary, not P07_T003.",
        "",
        "See `h2o_loss_routing_trace.csv` for file-level trace.",
        "",
    ]
    (out_dir / "H2O_LOSS_ROUTING_AUDIT.md").write_text("\n".join(md), encoding="utf-8")


def audit_process(
    *,
    process_id: int,
    target_ref: Path,
    edge_ref: Path,
    main_path: Path,
    streams_path: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    v4_specs = load_target_stream_targets_v4(V4_TARGETS)
    pid_label = f"Process{process_id}"
    h2o_specs = [s for s in v4_specs.get(pid_label, []) if s.target_species == "H2O"]
    if not h2o_specs:
        raise ValueError(f"No H2O v4 target for {pid_label}")
    h2o = h2o_specs[0]

    answers = pd.read_csv(target_ref, dtype={"process_id": int})
    edges = pd.read_csv(edge_ref, dtype={"process_id": int})
    edges_p = edges[edges["process_id"] == process_id]

    routing_rows = []
    for spec in v4_specs.get(pid_label, []):
        er = _edge_row_from_canonical(edges_p, spec.canonical_edge_id)
        routing_rows.append(
            {
                "target_id": spec.target_id,
                "target_species": spec.target_species,
                "target_feature": spec.target_name,
                "v4_canonical_edge_id": spec.canonical_edge_id,
                "v4_stream_key": spec.required_stream_key,
                "v4_formula": spec.target_formula,
                "edge_src_node": er.get("src_node", ""),
                "edge_dst_node": er.get("dst_node", ""),
                "edge_stream_key": er.get("main_data_stream_key", ""),
                "edge_dst_is_v_output": er.get("dst_is_v_output", ""),
            }
        )
    pd.DataFrame(routing_rows).to_csv(out_dir / f"process{process_id}_h2o_target_routing.csv", index=False)

    main = pd.read_csv(main_path)
    streams = pd.read_csv(streams_path)
    comp_rows = []
    for label, edge_id, stream_key, main_col in (
        ("v4_h2o_target", h2o.canonical_edge_id, h2o.required_stream_key, h2o.target_name),
        ("prod_answer_edge", "P07_E016" if process_id == 7 else "", "PROD", h2o.target_name),
    ):
        if not edge_id:
            continue
        stats = _compare_main_calc(main, streams, main_col, stream_key)
        er = _edge_row_from_canonical(edges_p, edge_id)
        comp_rows.append(
            {
                "comparison_label": label,
                "edge_id": edge_id,
                "stream_key": stream_key,
                "main_column": main_col,
                "formula": "Mole_Flow * Frac_H2O",
                **er,
                **stats,
            }
        )
    pd.DataFrame(comp_rows).to_csv(out_dir / f"process{process_id}_h2o_formula_comparison.csv", index=False)

    from types import SimpleNamespace

    class Export:
        pass

    exp = Export()
    exp.process_id = ["Process7", "Process7"]
    exp.main_data_stream_key = ["PROD", "RE"]
    exp.canonical_edge_id = ["P07_E016", "P07_E017"]

    cfg = SimpleNamespace(
        answer_edge_weight=5.0,
        answer_edge_weight_h2=5.0,
        answer_edge_weight_co2=5.0,
        answer_edge_weight_h2o=5.0,
        use_v4_target_edge_weighting=True,
        v4_target_weighting_mode="target_rows",
    )
    w, warns = build_v4_target_loss_weight_tensor(
        export_meta=exp,
        edge_target_columns=STREAM_EDGE_FEATURE_SLOTS,
        train_cfg=cfg,
        v4_targets_path=V4_TARGETS,
    )
    cols = list(STREAM_EDGE_FEATURE_SLOTS)
    h2o_i = cols.index("Frac_H2O")
    loss_check = {
        "P07_E016_Frac_H2O_weight": float(w[0, h2o_i]) if w is not None else 1.0,
        "P07_E017_Frac_H2O_weight": float(w[1, h2o_i]) if w is not None else 1.0,
        "weighting_warnings": "; ".join(warns),
    }

    report = [
        f"# Process {process_id} H2O routing report",
        "",
        f"## V4 H2O target (`{h2o.target_id}`)",
        "",
        f"- target_feature: `{h2o.target_name}`",
        f"- canonical edge: `{h2o.canonical_edge_id}`",
        f"- stream: `{h2o.required_stream_key}`",
        f"- formula: `{h2o.target_formula}`",
        "",
        "## Main vs Streams formula",
        "",
        "See `process7_h2o_formula_comparison.csv` — RE edge (P07_E017) should match RE_H2O_Mole;",
        "PROD edge Frac_H2O on P07_E016 should not.",
        "",
        "## Loss weight spot-check (mock Process7 two-edge batch)",
        "",
        f"- P07_E016 Frac_H2O weight: {loss_check['P07_E016_Frac_H2O_weight']} (expect 1.0)",
        f"- P07_E017 Frac_H2O weight: {loss_check['P07_E017_Frac_H2O_weight']} (expect >= 5.0)",
        "",
    ]
    if comp_rows:
        best = min(comp_rows, key=lambda r: r.get("diff_abs_mean", float("inf")))
        report.append(f"Best RE_H2O alignment: `{best['comparison_label']}` edge `{best['edge_id']}` "
                      f"status={best.get('status')} mismatch_rate={best.get('mismatch_rate')}")
    (out_dir / f"PROCESS{process_id}_H2O_ROUTING_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def write_final_report(out_dir: Path) -> None:
    text = [
        "# H2O target routing and fix report",
        "",
        "## Findings",
        "",
        "1. **Edge feature vector**: all edges include `Frac_H2O` in `STREAM_EDGE_FEATURE_SLOTS`.",
        "2. **Legacy answer slots**: only `target_h2` / `tailgas_co2`; no H2O answer slot.",
        "3. **Legacy Optuna objective** (`legacy_answer_fraction_r2`): weighted H2+CO2 Frac on answer edges only; **excludes H2O**.",
        "4. **`answer_edge_weight_h2o`**: applied to **training/eval loss** on v4 target-row canonical edge `Frac_H2O` when `use_v4_target_edge_weighting=true` (default).",
        "5. **Process7 P07_T003**: metrics use `P07_E017` / `RE` / `Mole_Flow * Frac_H2O`; loss weights `Frac_H2O` on `P07_E017` only.",
        "6. **P07_E016 PROD `Frac_H2O`**: not the v4 H2O target; legacy extras may still log `answer_*_Frac_H2O_r2` on answer edge.",
        "",
        "## Code changes",
        "",
        "- `src/process_graph/experiment/v4_target_edge_weighting.py` (new)",
        "- `scripts/train_process_surrogate.py`, `edge_all_reporting.py`, `target_v4_metrics.py`",
        "- `schema.py`, `process_train.yaml`, `process_surrogate_edge_all_v3.yaml`",
        "- `scripts/audit_h2o_target_routing.py`, `tests/test_v4_h2o_target_routing.py`",
        "",
        "## Verify",
        "",
        "```bash",
        "python scripts/audit_h2o_target_routing.py --process-id 7 --out outputs/h2o_target_routing_audit",
        "pytest tests/test_v4_h2o_target_routing.py -q",
        "```",
        "",
        "## Optuna note",
        "",
        "Historical `best_value≈0.9076` is legacy H2+CO2 answer-edge fraction R², not v4 macro with H2O.",
        "Use `--optuna-objective target_v4_main_verified_r2` for trials that include P07_T003.",
        "",
    ]
    (out_dir / "H2O_TARGET_ROUTING_AND_FIX_REPORT.md").write_text("\n".join(text), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit H2O v4 target routing and loss weights.")
    ap.add_argument("--process-id", type=int, default=7)
    ap.add_argument("--target-ref", type=Path, default=PROJECT_ROOT / "data/reference/v3/target_answer_edges.csv")
    ap.add_argument("--edge-ref", type=Path, default=PROJECT_ROOT / "data/reference/v3/canonical_edges.csv")
    ap.add_argument("--main", type=Path, default=None)
    ap.add_argument("--streams", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/h2o_target_routing_audit")
    args = ap.parse_args()
    pid = int(args.process_id)
    main_p = args.main or (PROJECT_ROOT / f"data/main_data/{pid}.Process_Main.csv")
    streams_p = args.streams or (PROJECT_ROOT / f"data/main_data_Streams/{pid}.Process_Streams.csv")
    out = vc.resolve_path(args.out)
    write_loss_routing_trace(out)
    audit_process(
        process_id=pid,
        target_ref=vc.resolve_path(args.target_ref),
        edge_ref=vc.resolve_path(args.edge_ref),
        main_path=vc.resolve_path(main_p),
        streams_path=vc.resolve_path(streams_p),
        out_dir=out,
    )
    write_final_report(out)
    print(f"Wrote audit under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
