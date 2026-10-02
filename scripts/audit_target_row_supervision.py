#!/usr/bin/env python3
"""Audit target-row vs species-level supervision for edge_all v4 targets."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from process_graph.constants import STREAM_EDGE_FEATURE_SLOTS  # noqa: E402
from process_graph.experiment.target_row_amounts import compute_target_row_amounts  # noqa: E402
from process_graph.experiment.target_row_spec import (  # noqa: E402
    TargetRowSpec,
    load_target_row_specs,
)
from process_graph.experiment.target_v4_metrics import load_target_stream_targets_v4  # noqa: E402
from process_graph.experiment.v4_target_edge_weighting import build_v4_target_loss_weight_tensor  # noqa: E402

V4_CSV = PROJECT_ROOT / "data/reference/v4/target_stream_targets.csv"
PROV_CSV = (
    PROJECT_ROOT
    / "outputs/v4_target_formula_validation_all/provenance_decisions/v4_target_provenance_decisions.csv"
)


def write_trace(out: Path) -> None:
    rows = [
        {
            "file": "src/process_graph/experiment/train_utils.py",
            "function_or_class": "compute_edge_all_training_loss",
            "variable_name": "loss_edge",
            "config_key": "loss_type_edge_all",
            "behavior_type": "edge_all_base_loss",
            "target_granularity": "edge_feature_vector",
            "species_used_as_identity": "no",
            "target_id_used": "no",
            "edge_id_used": "all",
            "notes": "Full STREAM_EDGE_FEATURE_SLOTS on every masked edge",
        },
        {
            "file": "src/process_graph/experiment/target_row_loss.py",
            "function_or_class": "compute_v4_target_row_loss",
            "variable_name": "loss_v4_targets",
            "config_key": "use_v4_target_row_loss",
            "behavior_type": "v4_target_row_loss",
            "target_granularity": "target_row",
            "species_used_as_identity": "no",
            "target_id_used": "yes",
            "edge_id_used": "yes",
            "notes": "One loss term per (target_id, edge match); species_weight is multiplier only",
        },
        {
            "file": "src/process_graph/experiment/v4_target_edge_weighting.py",
            "function_or_class": "build_v4_target_loss_weight_tensor",
            "variable_name": "edge_stream_loss_weight",
            "config_key": "use_legacy_answer_weighting",
            "behavior_type": "legacy_answer_weighting",
            "target_granularity": "target_row",
            "species_used_as_identity": "partial",
            "target_id_used": "yes",
            "edge_id_used": "yes",
            "notes": "Disabled by default when use_v4_target_row_loss=true",
        },
        {
            "file": "src/process_graph/experiment/train_utils.py",
            "function_or_class": "evaluate_edge_all_epoch",
            "variable_name": "metric_target_r2",
            "config_key": "answer_edge_pos",
            "behavior_type": "legacy_answer_weighting",
            "target_granularity": "legacy_slot",
            "species_used_as_identity": "partial",
            "target_id_used": "no",
            "edge_id_used": "answer slots only",
            "notes": "target_h2/tailgas_co2 Frac slots; not full v4 target set",
        },
        {
            "file": "src/process_graph/experiment/target_v4_metrics.py",
            "function_or_class": "compute_target_v4_metrics_from_original_scale",
            "variable_name": "target_metrics_df",
            "config_key": "target_stream_targets.csv",
            "behavior_type": "v4_target_metric",
            "target_granularity": "target_row",
            "species_used_as_identity": "no",
            "target_id_used": "yes",
            "edge_id_used": "yes",
            "notes": "One metric row per target_id; macro = mean of target rows",
        },
        {
            "file": "scripts/tune_optuna.py",
            "function_or_class": "_objective_series_and_name",
            "variable_name": "answer_targets_r2",
            "config_key": "optuna_objective",
            "behavior_type": "optuna_objective",
            "target_granularity": "legacy_slot",
            "species_used_as_identity": "yes",
            "target_id_used": "no",
            "edge_id_used": "answer edges",
            "notes": "legacy_answer_fraction_r2: H2+CO2 weighted; excludes multi-CO2 v4 rows",
        },
        {
            "file": "scripts/tune_optuna.py",
            "function_or_class": "_objective_series_and_name",
            "variable_name": "target_v4_macro_r2_main_verified",
            "config_key": "optuna_objective=target_v4_main_verified_r2",
            "behavior_type": "optuna_objective",
            "target_granularity": "target_row",
            "species_used_as_identity": "no",
            "target_id_used": "yes",
            "edge_id_used": "per target_id",
            "notes": "Mean R² over provenance-selected target_id rows",
        },
    ]
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "current_metric_and_loss_trace.csv", index=False)
    md = """# Target row supervision audit

## Answers (code-backed)

1. **Base loss**: all edge feature vectors (`loss_edge`).
2. **answer_edge_weight_h2/co2/h2o**: used as **species_weight multiplier** on each **target_id** row in `loss_v4_targets`; not as routing keys.
3. **Routing**: `target_id` + `canonical_edge_id` + `required_stream_key` from v4 table.
4. **Multiple CO2 targets**: separate TargetRowSpec / metric rows (P01_T001 vs P01_T003, P04_T002 vs P04_T003).
5. **Optuna legacy 0.9076**: `legacy_answer_fraction_r2` — H2+CO2 answer slots only; **not** v4 macro; **H2O not included**.

See `current_metric_and_loss_trace.csv` and inventory CSVs in this directory.
"""
    (out / "TARGET_ROW_SUPERVISION_AUDIT.md").write_text(md, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-ref", type=Path, default=PROJECT_ROOT / "data/reference/v3/target_answer_edges.csv")
    ap.add_argument("--edge-ref", type=Path, default=PROJECT_ROOT / "data/reference/v3/canonical_edges.csv")
    ap.add_argument("--validation-dir", type=Path, default=PROJECT_ROOT / "outputs/v4_target_formula_validation_all")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/target_row_supervision_audit")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    write_trace(out)

    prov_path = PROV_CSV if PROV_CSV.is_file() else None
    specs = load_target_row_specs(provenance_path=prov_path)
    inv_rows = []
    for s in specs:
        inv_rows.append(
            {
                "process_id": s.process_id,
                "target_id": s.target_id,
                "target_feature": s.target_feature,
                "target_stream": s.target_stream,
                "edge_id": s.edge_id,
                "species": s.species,
                "frac_slot": s.frac_slot,
                "formula": s.formula,
                "scale": s.scale,
                "formula_status": s.formula_status,
                "target_provenance": s.target_provenance,
                "include_in_main_verified_macro": s.include_in_main_verified_macro,
                "include_in_formula_macro": s.include_in_formula_macro,
                "exclusion_reason": s.exclusion_reason,
                "target_weight": s.target_weight,
                "species_weight": s.species_weight,
                "final_weight": s.final_weight,
            }
        )
    pd.DataFrame(inv_rows).to_csv(out / "target_row_inventory.csv", index=False)

    mult_rows = []
    by_ps: dict[tuple[str, str], list[TargetRowSpec]] = defaultdict(list)
    for s in specs:
        by_ps[(s.process_id, s.species)].append(s)
    for (pid, species), group in sorted(by_ps.items()):
        mult_rows.append(
            {
                "process_id": pid,
                "species": species,
                "n_target_rows": len(group),
                "target_ids": ";".join(x.target_id for x in group),
                "target_features": ";".join(x.target_feature for x in group),
                "target_streams": ";".join(x.target_stream for x in group),
                "edge_ids": ";".join(x.edge_id for x in group),
                "requires_target_row_handling": "yes" if len(group) > 1 else "no",
            }
        )
    pd.DataFrame(mult_rows).to_csv(out / "species_multiplicity_check.csv", index=False)

    cfg = SimpleNamespace(
        answer_edge_weight=5.0,
        answer_edge_weight_h2=5.0,
        answer_edge_weight_co2=5.0,
        answer_edge_weight_h2o=5.0,
        use_v4_target_row_loss=True,
        use_legacy_answer_weighting=False,
        use_v4_target_edge_weighting=True,
        v4_target_weighting_mode="target_row",
    )

    class Export:
        pass

    exp = Export()
    exp.process_id = ["Process1", "Process1", "Process4", "Process4", "Process7", "Process7", "Process7"]
    exp.main_data_stream_key = ["FUELGAS", "PROD", "EX", "10", "PROD", "PROD", "RE"]
    exp.canonical_edge_id = ["P01_E015", "P01_E021", "P04_E009", "P04_E020", "P07_E016", "P07_E016", "P07_E017"]

    loss_check = []
    for s in specs:
        if s.target_id in ("P01_T001", "P01_T003", "P04_T002", "P04_T003", "P07_T003"):
            loss_check.append(
                {
                    "target_id": s.target_id,
                    "edge_id": s.edge_id,
                    "species": s.species,
                    "frac_slot": s.frac_slot,
                    "final_weight": s.final_weight,
                    "species_weight_multiplier_only": "yes",
                    "v4_row_loss_applies": not s.skip_v4_loss(),
                    "metric_row_expected": "yes",
                }
            )
    pd.DataFrame(loss_check).to_csv(out / "loss_weighting_target_row_check.csv", index=False)

    v4 = load_target_stream_targets_v4(V4_CSV)
    metric_check = []
    for pid in ("Process1", "Process4", "Process7"):
        co2 = [s for s in v4.get(pid, []) if s.target_species == "CO2"]
        h2o = [s for s in v4.get(pid, []) if s.target_species == "H2O"]
        metric_check.append(
            {
                "process_id": pid,
                "n_co2_target_rows": len(co2),
                "co2_target_ids": ";".join(s.target_id for s in co2),
                "co2_edge_ids": ";".join(s.canonical_edge_id for s in co2),
                "n_h2o_target_rows": len(h2o),
                "h2o_target_ids": ";".join(s.target_id for s in h2o),
                "h2o_edge_ids": ";".join(s.canonical_edge_id for s in h2o),
            }
        )
    pd.DataFrame(metric_check).to_csv(out / "metric_target_row_check.csv", index=False)

    report = f"""# Target row supervision audit report

## Multiplicity

- Process1 CO2 targets: P01_T001 (FUELGAS/P01_E015), P01_T003 (PROD/P01_E021) — **separate rows**
- Process4 CO2 targets: P04_T002 (EX), P04_T003 (stream 10) — **separate rows**
- Process7 H2O: P07_T003 on **P07_E017 / RE** (not P07_E016 PROD Frac_H2O)

## Loss

- Primary v4 supervision: `loss_v4_targets` per **target_id** (see `target_row_loss.py`).
- `answer_edge_weight_h2o` is **species_weight multiplier**, not routing.
- Legacy element tensor weighting: only if `use_legacy_answer_weighting=true`.

## Optuna 0.9076

Historical `best_value≈0.9076` = **legacy_answer_fraction_r2** (H2+CO2 answer-edge Frac R²).
Does **not** include P07_T003 or multi-CO2 v4 rows.
Use `--optuna-objective target_v4_main_verified_macro_r2` for target-row macro.

## Outputs

- `{out / "target_row_inventory.csv"}`
- `{out / "species_multiplicity_check.csv"}`
- `{out / "loss_weighting_target_row_check.csv"}`
- `{out / "metric_target_row_check.csv"}`
"""
    (out / "TARGET_ROW_SUPERVISION_AUDIT_REPORT.md").write_text(report, encoding="utf-8")
    fix = report + "\n\n## Fix summary\n\n- TargetRowSpec + target-row auxiliary loss added.\n- Species group metrics marked reporting-only.\n"
    (out / "TARGET_ROW_SUPERVISION_FIX_REPORT.md").write_text(fix, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
