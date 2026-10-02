"""Human-readable metric / eval key names for edge_all target-row evaluation.

Avoid version suffixes (v1, v2, v4). Legacy keys remain in DEPRECATED_* maps for reading old runs.
"""

from __future__ import annotations

# Schema tag written to metrics.json / summary CSV (not a version number).
METRIC_SCHEMA = "edge_all_target_row_metrics"

# --- Primary: Frac_H2 / Frac_CO2 / Frac_H2O on canonical answer edge (no Mole_Flow, no scale) ---
EVAL_PRIMARY_FRAC_R2_BY_PROCESS = "eval_primary_frac_r2_by_process"
EVAL_PRIMARY_FRAC_R2_BY_TARGET = "eval_primary_frac_r2_by_target"
EVAL_PRIMARY_FRAC_R2_FORMULA_BY_PROCESS = "eval_primary_frac_r2_formula_ok_by_process"
EVAL_PRIMARY_FRAC_R2_FORMULA_BY_TARGET = "eval_primary_frac_r2_formula_ok_by_target"
FRAC_R2_SHORT_ALIAS = "frac_r2_main_target_rows"

# --- Secondary: amount = scale * Mole_Flow * Frac (diagnostic, not early-stop) ---
EVAL_SECONDARY_AMOUNT_R2_BY_PROCESS = "eval_secondary_amount_r2_by_process"
EVAL_SECONDARY_AMOUNT_R2_BY_TARGET = "eval_secondary_amount_r2_by_target"
EVAL_SECONDARY_AMOUNT_R2_FORMULA_BY_PROCESS = "eval_secondary_amount_r2_formula_ok_by_process"
EVAL_SECONDARY_AMOUNT_R2_FORMULA_BY_TARGET = "eval_secondary_amount_r2_formula_ok_by_target"
AMOUNT_R2_SHORT_ALIAS = "amount_r2_main_target_rows"

# --- Legacy Optuna-comparable (answer-edge slot Frac H2/CO2 weighted, not per target_id) ---
EVAL_LEGACY_ANSWER_FRAC_R2 = "eval_legacy_answer_edge_frac_r2"
EVAL_LEGACY_ANSWER_FRAC_R2_MACRO = "eval_legacy_answer_edge_frac_r2_macro"

# Per-row CSV metric_kind column values
METRIC_KIND_FRAC = "target_row_frac_species"
METRIC_KIND_AMOUNT = "target_row_amount_mole_x_frac"

# Written into target_metrics JSON metadata
METRIC_VERSION_AMOUNT_ROWS = "edge_all_target_row_amount"
METRIC_VERSION_FRAC_ROWS = "edge_all_target_row_frac_species"

# summary_kind values in *summary.csv
SUMMARY_FRAC_R2_BY_PROCESS = "frac_r2_mean_by_process"
SUMMARY_FRAC_R2_BY_TARGET = "frac_r2_mean_by_target_row"
SUMMARY_FRAC_R2_FORMULA_BY_PROCESS = "frac_r2_formula_ok_mean_by_process"
SUMMARY_FRAC_R2_FORMULA_BY_TARGET = "frac_r2_formula_ok_mean_by_target_row"
SUMMARY_AMOUNT_R2_BY_PROCESS = "amount_r2_mean_by_process"
SUMMARY_AMOUNT_R2_BY_TARGET = "amount_r2_mean_by_target_row"

SUMMARY_ROW_ID_FRAC_PROCESS = "__ALL_frac_r2_by_process__"
SUMMARY_ROW_ID_FRAC_TARGET = "__ALL_frac_r2_by_target_row__"

# Species breakdown (epoch / payload)
EVAL_FRAC_R2_MEAN_H2 = "eval_frac_r2_mean_h2_targets"
EVAL_FRAC_R2_MEAN_CO2 = "eval_frac_r2_mean_co2_targets"
EVAL_FRAC_R2_MEAN_H2O = "eval_frac_r2_mean_h2o_targets"

EVAL_TARGET_ROWS_EVALUATED = "eval_target_rows_count"
EVAL_TARGET_ROWS_SKIPPED = "eval_target_rows_skipped_count"
EVAL_TARGET_ROWS_TOTAL = "eval_target_rows_total"

# --- Deprecated keys (old runs, Optuna history) -> canonical ---
DEPRECATED_TO_CANONICAL: dict[str, str] = {
    "target_row_primary_v2": METRIC_SCHEMA,
    "target_row_primary_v1": METRIC_SCHEMA,
    "process_balanced_main_target_frac_r2_main_verified": EVAL_PRIMARY_FRAC_R2_BY_PROCESS,
    "target_balanced_main_target_frac_r2_main_verified": EVAL_PRIMARY_FRAC_R2_BY_TARGET,
    "process_balanced_main_target_frac_r2_formula_available": EVAL_PRIMARY_FRAC_R2_FORMULA_BY_PROCESS,
    "target_balanced_main_target_frac_r2_formula_available": EVAL_PRIMARY_FRAC_R2_FORMULA_BY_TARGET,
    "target_v4_frac_main_verified_r2": FRAC_R2_SHORT_ALIAS,
    "process_balanced_main_target_r2_main_verified": EVAL_SECONDARY_AMOUNT_R2_BY_PROCESS,
    "target_balanced_main_target_r2_main_verified": EVAL_SECONDARY_AMOUNT_R2_BY_TARGET,
    "target_v4_main_verified_r2": AMOUNT_R2_SHORT_ALIAS,
    "target_v4_macro_r2_main_verified": AMOUNT_R2_SHORT_ALIAS,
    "legacy_answer_fraction_r2": EVAL_LEGACY_ANSWER_FRAC_R2,
    "legacy_answer_fraction_macro_r2": EVAL_LEGACY_ANSWER_FRAC_R2_MACRO,
    "process_balanced_macro_main_verified": SUMMARY_FRAC_R2_BY_PROCESS,
    "target_balanced_macro_main_verified": SUMMARY_FRAC_R2_BY_TARGET,
    "process_balanced_macro_formula_available": SUMMARY_FRAC_R2_FORMULA_BY_PROCESS,
    "target_balanced_macro_formula_available": SUMMARY_FRAC_R2_FORMULA_BY_TARGET,
    "__ALL_process_balanced_main_verified__": SUMMARY_ROW_ID_FRAC_PROCESS,
    "__ALL_target_balanced_main_verified__": SUMMARY_ROW_ID_FRAC_TARGET,
    "target_v4_all_targets_v1": METRIC_VERSION_AMOUNT_ROWS,
    "target_v4_frac_v1": METRIC_VERSION_FRAC_ROWS,
    "frac": METRIC_KIND_FRAC,
    "amount": METRIC_KIND_AMOUNT,
}

CANONICAL_TO_DEPRECATED: dict[str, list[str]] = {}
for old, new in DEPRECATED_TO_CANONICAL.items():
    CANONICAL_TO_DEPRECATED.setdefault(new, []).append(old)


def resolve_metric_key(key: str) -> str:
    """Map a deprecated metric key to the canonical name if known."""
    k = str(key or "").strip()
    return DEPRECATED_TO_CANONICAL.get(k, k)


def pick_from_mapping(mapping: dict, *keys: str, default=float("nan")):
    """Return first finite float found under canonical or deprecated keys."""
    import math

    for k in keys:
        for candidate in (k, resolve_metric_key(k)):
            if candidate not in mapping:
                continue
            try:
                v = float(mapping[candidate])
                if math.isfinite(v):
                    return v
            except (TypeError, ValueError):
                continue
    return default
