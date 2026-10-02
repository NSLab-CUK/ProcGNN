#!/usr/bin/env python3
"""Smoke validation for v4 target metrics (no training)."""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from process_graph.constants import STREAM_EDGE_FEATURE_SLOTS
from process_graph.experiment.target_v4_metrics import (
    METRIC_VERSION,
    compute_target_v4_metrics_from_original_scale,
    load_target_stream_targets_v4,
    normalize_stream_key,
    write_target_v4_split_artifacts,
)

V4_PATH = PROJECT_ROOT / "data/reference/v4/target_stream_targets.csv"
V3_ANSWERS = PROJECT_ROOT / "data/reference/v3/target_answer_edges.csv"


def _synthetic_edge_predictions(process_id: str, stream_key: str, edge_id: str, n: int = 5) -> pd.DataFrame:
    rows: list[dict] = []
    for i in range(n):
        base = {
            "split": "test",
            "process_id": process_id,
            "sample_id": str(i + 1),
            "canonical_edge_id": edge_id,
            "main_data_stream_key": stream_key,
            "y_edge_mask": 1.0,
            "is_answer_edge": 0,
        }
        vals = {
            "Temp": 300.0,
            "Pres": 1.0,
            "Vol_Flow": 1e5,
            "Mole_Flow": 100.0 + i,
            "Mass_Flow": 1e4,
            "Frac_H2O": 0.05,
            "Frac_H2": 0.7,
            "Frac_CH4": 0.05,
            "Frac_CO2": 0.1,
            "Frac_CO": 0.0,
            "Frac_O2": 0.0,
            "Frac_N2": 0.0,
        }
        for prop in STREAM_EDGE_FEATURE_SLOTS:
            rows.append(
                {
                    **base,
                    "property_name": prop,
                    "y_true_orig": vals[prop],
                    "y_pred_orig": vals[prop] * 0.95,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    errors: list[str] = []
    print(f"[validate] metric_version={METRIC_VERSION}")

    if not V4_PATH.is_file():
        errors.append(f"missing {V4_PATH}")
        print("[FAIL]", *errors, sep="\n  - ")
        return 1

    by_proc = load_target_stream_targets_v4(V4_PATH)
    if by_proc:
        print("[validate] process target counts:")
        for pid in sorted(by_proc.keys(), key=lambda x: int(x.replace("Process", ""))):
            specs = by_proc[pid]
            species = sorted({s.target_species for s in specs})
            print(f"  {pid}: n_targets={len(specs)} species={species}")
            for s in specs:
                if not s.required_stream_key:
                    errors.append(f"{pid} {s.target_id}: empty required_stream_key")
                if not math.isfinite(s.scale_factor):
                    errors.append(f"{pid} {s.target_id}: bad scale_factor")

        # P2 scale check
        p2 = {s.target_id: s.scale_factor for s in by_proc["Process2"]}
        if p2.get("P02_T001") != 0.8:
            errors.append(f"P02_T001 scale_factor expected 0.8 got {p2.get('P02_T001')}")
        if p2.get("P02_T002") != 1.0:
            errors.append(f"P02_T002 scale_factor expected 1.0 got {p2.get('P02_T002')}")

        # P9 no H2O
        if any(s.target_species == "H2O" for s in by_proc.get("Process9", [])):
            errors.append("Process9 should not have H2O target in v4 specs")

    # Per-process: build edge rows for every v4 target stream on that process.
    with tempfile.TemporaryDirectory() as td:
        for pid in sorted(by_proc.keys(), key=lambda x: int(x.replace("Process", ""))):
            specs = by_proc[pid]
            rows: list[dict] = []
            for spec in specs:
                rows.extend(
                    _synthetic_edge_predictions(
                        pid,
                        spec.required_stream_key,
                        spec.canonical_edge_id or f"E_{spec.target_id}",
                    ).to_dict(orient="records")
                )
            ep = pd.DataFrame(rows)
            out = Path(td) / pid
            result = compute_target_v4_metrics_from_original_scale(
                edge_predictions=ep,
                target_specs=specs,
                split_name="test",
            )
            write_target_v4_split_artifacts(out, result)
            payload = result.payload
            n_eval = len(result.target_metrics_df)
            n_skip = len(result.skipped_df)
            if n_eval != len(specs):
                errors.append(f"{pid}: evaluated {n_eval} targets, expected {len(specs)} (skipped={n_skip})")
            summ_path = out / "target_metrics_v4_summary.csv"
            met_path = out / "target_metrics_v4.csv"
            if not summ_path.is_file() or not met_path.is_file():
                errors.append(f"{pid}: missing v4 output files")
                continue
            summ = pd.read_csv(summ_path)
            macro = summ[summ["target_id"].astype(str) == "__ALL_macro_split__"]
            if macro.empty:
                errors.append(f"{pid}: no __ALL_macro_split__ row")
            else:
                mr2 = float(macro.iloc[0]["r2"])
                if not math.isfinite(mr2):
                    errors.append(f"{pid}: macro_r2 not finite")
            if "test/target_v4_macro_r2" not in payload and "target_v4_macro_r2" not in payload:
                errors.append(f"{pid}: macro_r2 missing from payload keys")
            print(f"  {pid}: evaluated={n_eval} skipped={n_skip} macro_r2={macro.iloc[0]['r2']:.4f}")

    # P4: two CO2 targets on different streams
    with tempfile.TemporaryDirectory() as td:
        p4_specs = by_proc["Process4"]
        co2_specs = [s for s in p4_specs if s.target_species == "CO2"]
        if len(co2_specs) != 2:
            errors.append(f"Process4 expected 2 CO2 targets, got {len(co2_specs)}")
        rows = []
        for spec in p4_specs:
            rows.extend(
                _synthetic_edge_predictions(
                    "Process4",
                    spec.required_stream_key,
                    spec.canonical_edge_id,
                ).to_dict(orient="records")
            )
        result = compute_target_v4_metrics_from_original_scale(
            edge_predictions=pd.DataFrame(rows),
            target_specs=p4_specs,
            split_name="test",
        )
        co2_ids = set(result.target_metrics_df[result.target_metrics_df["target_species"] == "CO2"]["target_id"])
        if co2_ids != {s.target_id for s in co2_specs}:
            errors.append(f"Process4 CO2 target_ids mismatch: {co2_ids}")

    # Legacy path: compute_target_metrics_v4 wrapper
    with tempfile.TemporaryDirectory() as td:
        ep = _synthetic_edge_predictions("Process2", "PROD", "P02_E014")
        from process_graph.experiment.edge_all_reporting import compute_target_metrics_v4

        out = Path(td) / "wrapper"
        payload = compute_target_metrics_v4(
            out_dir=out,
            edge_predictions=ep,
            target_stream_targets_path=V4_PATH,
        )
        if "target_v4_macro_r2" not in payload and not any(
            k.endswith("target_v4_macro_r2") for k in payload
        ):
            errors.append("compute_target_metrics_v4 wrapper: no macro_r2 in payload")

    # Legacy v3 answer edges still readable
    if V3_ANSWERS.is_file():
        ans = pd.read_csv(V3_ANSWERS)
        if len(ans) != 20:
            errors.append(f"expected 20 v3 answer rows, got {len(ans)}")
    else:
        errors.append(f"missing {V3_ANSWERS}")

    print(f"[validate] stream key norm: {normalize_stream_key('12.0')!r} -> {normalize_stream_key('12')!r}")

    if errors:
        print("[FAIL]")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("[PASS] v4 metrics validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
