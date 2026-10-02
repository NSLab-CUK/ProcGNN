#!/usr/bin/env python3
"""Document what historical Optuna best_value metrics represent (no study.db modification)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Optuna objective compatibility report.")
    parser.add_argument("--optuna-root", default="outputs/optuna")
    parser.add_argument("--out", default="outputs/v4_target_formula_validation_all/optuna_compatibility")
    args = parser.parse_args()

    root = Path(args.optuna_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for best_path in sorted(root.glob("**/best_hyperparameters.json")):
        try:
            data = json.loads(best_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        metric = data.get("metric", {}) or {}
        obj_name = metric.get("name", data.get("objective", {}).get("name", "unknown"))
        best_val = data.get("best_value", float("nan"))
        rows.append(f"| `{best_path.relative_to(PROJECT_ROOT)}` | {obj_name} | {best_val} |")

    lines = [
        "# Optuna Objective Compatibility Report",
        "",
        "Existing `study.db` files are **not modified**. This report interprets saved `best_hyperparameters.json` only.",
        "",
        "## Historical default (edge_all)",
        "",
        "- **Stored name**: often `optuna_objective` or `answer_targets_r2`",
        "- **Definition**: max validation **legacy answer fraction R²** — weighted mean of "
        "`metric_target_r2` (Frac_H2 on PROD answer edge) and `metric_tailgas_r2` (Frac_CO2), "
        "typically **H2 and CO2 only** (H2O omitted from Optuna weights).",
        "- **NOT** `target_v4_macro_r2` on amount targets (`Mole_Flow * Frac_*`).",
        "",
        "### Process7 example (~0.9076)",
        "",
        "- That value is **legacy fraction R²** on answer edges (P07_T001 H2 / P07_T002 CO2 Frac), "
        "not per-target v4 amount macro over P07_T001–P07_T003.",
        "- After provenance policy, compare new runs with:",
        "  - `--optuna-objective legacy_answer_fraction_r2` (reproduces old behavior)",
        "  - `--optuna-objective target_v4_main_verified_r2` (Main-verified amount targets only)",
        "",
        "## New objective names (forward)",
        "",
        "| CLI value | Epoch series |",
        "|-----------|----------------|",
        "| `legacy_answer_fraction_r2` | `val_answer_targets_r2_weighted` |",
        "| `target_v4_main_verified_r2` | `val_target_v4_macro_r2_main_verified` |",
        "| `target_v4_formula_available_r2` | `val_target_v4_macro_r2_formula_available` |",
        "",
        "## Discovered best_hyperparameters.json",
        "",
        "| Path | metric.name | best_value |",
        "|------|-------------|------------|",
        *rows,
        "",
    ]
    (out / "OPTUNA_OBJECTIVE_COMPATIBILITY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[out] {out / 'OPTUNA_OBJECTIVE_COMPATIBILITY.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
