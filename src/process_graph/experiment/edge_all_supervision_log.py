"""Terminal/log formatting for edge_all target-row supervision (ablation-friendly)."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .target_metric_names import (
    AMOUNT_R2_SHORT_ALIAS,
    EVAL_LEGACY_ANSWER_FRAC_R2,
    EVAL_PRIMARY_FRAC_R2_BY_PROCESS,
    EVAL_PRIMARY_FRAC_R2_BY_TARGET,
    EVAL_PRIMARY_FRAC_R2_FORMULA_BY_PROCESS,
    EVAL_SECONDARY_AMOUNT_R2_BY_PROCESS,
    EVAL_SECONDARY_AMOUNT_R2_BY_TARGET,
    EVAL_TARGET_ROWS_EVALUATED,
    FRAC_R2_SHORT_ALIAS,
    pick_from_mapping,
)
from .target_row_spec import load_target_row_specs

PRIMARY_METRIC_NAME = EVAL_PRIMARY_FRAC_R2_BY_PROCESS
AMOUNT_SECONDARY_METRIC_NAME = EVAL_SECONDARY_AMOUNT_R2_BY_PROCESS

TARGET_PROPERTY_R2_FIELDS: tuple[tuple[str, str], ...] = (
    ("Temp", "temp"),
    ("Pres", "pres"),
    ("Frac_H2O", "frac_h2o"),
    ("Frac_H2", "frac_h2"),
    ("Frac_CH4", "frac_ch4"),
    ("Frac_CO2", "frac_co2"),
    ("Frac_CO", "frac_co"),
    ("Frac_O2", "frac_o2"),
    ("Frac_N2", "frac_n2"),
    ("Mass_Flow", "mass_flow"),
)

__all__ = [
    "edge_all_supervision_banner",
    "format_epoch_detail_edge_all_line",
    "format_epoch_summary_edge_all_parts",
    "format_val_metric_edge_all_line",
]


def edge_all_supervision_banner(
    train_cfg: Any,
    *,
    provenance_loaded: bool | None = None,
    verbosity: str = "verbose",
) -> str:
    """Multi-line banner printed once at train/ablation startup."""
    from .metric_policy import build_loss_policy, build_metric_policy

    n_rows = len(load_target_row_specs(train_cfg=train_cfg))
    use_legacy_el = bool(getattr(train_cfg, "use_legacy_answer_weighting", False))
    mp = build_metric_policy(train_cfg)
    lp = build_loss_policy(train_cfg)
    if str(verbosity or "verbose").strip().lower() != "verbose":
        prov = "loaded" if provenance_loaded else "missing"
        return (
            "[edge_all][supervision] "
            f"target_rows={n_rows} official_metric={mp['official_metric']} "
            f"early_stop={mp['early_stop_resolved']} "
            f"target_weighting={lp.get('use_target_stream_loss_weighting')} "
            f"r2_weight={lp['all_edge_r2_loss_weight']} "
            f"provenance={prov}"
        )
    lines = [
        "[edge_all][supervision] target identity = process_id + canonical target stream edge",
        f"[edge_all][supervision] target rows loaded: {n_rows} (e.g. Process1 CO2 x2 = P01_T001 + P01_T003)",
        (
            "[edge_all][supervision] loss: weighted_edge_all(target-stream edge weights) + "
            f"all_edge_r2_loss_weight({lp['all_edge_r2_loss_weight']})*loss_all_edge_r2 "
            f"(r2_loss_scope={lp['r2_loss_scope']}, r2_config_source={lp['all_edge_r2_config_source']}, "
            f"amount_auxiliary={lp['amount_auxiliary']})"
        ),
        (
            "[edge_all][supervision] target_stream_weighting: "
            f"enabled={lp.get('use_target_stream_loss_weighting')} "
            f"default_weight={lp.get('target_stream_loss_weight')} "
            f"mode={lp.get('target_stream_weighting_mode')} "
            f"conflict_policy={lp.get('target_stream_weight_conflict_policy')} "
            f"allow_yaml_only={lp.get('allow_yaml_only_target_streams')}"
        ),
        (
            "[edge_all][supervision] element edge_stream_loss_weight: "
            + ("ON (legacy)" if use_legacy_el else "OFF (target-row loss default)")
        ),
        "[edge_all][supervision] loss_primary_frac contributes to total: false",
        "[edge_all][supervision] amount metrics are primary: false",
        "[edge_all][supervision] target stream feature metrics json: enabled",
        (
            f"[edge_all][supervision] official metric: {mp['official_metric']} "
            "(mean of 10 pooled target-property R2 values)"
        ),
        f"[edge_all][supervision] early stop: {mp['early_stop_resolved']}",
        f"[edge_all][supervision] secondary: {AMOUNT_SECONDARY_METRIC_NAME} (amount diagnostic)",
        "[edge_all][supervision] terminal logs prioritize pooled target-property R2; strict target-row Frac/amount are diagnostics",
    ]
    if provenance_loaded is False:
        lines.append(
            "[edge_all][supervision][warn] provenance policy CSV missing; macro may include all evaluated rows"
        )
    elif provenance_loaded is True:
        lines.append(
            "[edge_all][supervision] provenance policy loaded; excluded target rows omitted from primary macro"
        )
    return "\n".join(lines)


def _fin(epoch_log: Mapping[str, float], *keys: str) -> float | None:
    v = pick_from_mapping(epoch_log, *keys, default=float("nan"))
    return v if math.isfinite(v) else None


def format_epoch_summary_edge_all_parts(epoch_log: Mapping[str, float]) -> list[str]:
    """Extra tokens for one-line epoch summary (edge_all only)."""
    parts: list[str] = []
    le = _fin(epoch_log, "train/loss_edge_all", "train_loss_edge_all", "train/loss_edge", "train_loss_edge")
    lf = _fin(
        epoch_log,
        "train/loss_primary_frac",
        "train_loss_primary_frac",
        "train/loss_target_frac_feature",
        "train_loss_target_frac_feature",
        "train/loss_target_row_frac",
        "train_loss_target_row_frac",
    )
    lr2 = _fin(
        epoch_log,
        "train/loss_all_edge_r2",
        "train_loss_all_edge_r2",
        "train/loss_primary_frac_r2",
        "train_loss_primary_frac_r2",
    )
    pcnt = _fin(epoch_log, "train/primary_frac_count", "train_primary_frac_count", "primary_frac_count")
    ratio = _fin(epoch_log, "train/primary_frac_std_ratio_norm", "train_primary_frac_std_ratio_norm")
    la = _fin(epoch_log, "train/loss_target_row_amount", "train_loss_target_row_amount", "train/loss_v4_targets")
    if le is not None:
        parts.append(f"edge_loss={le:.6f}")
    if lf is not None:
        parts.append(f"target_incident_loss={lf:.6f}")
    if lr2 is not None:
        parts.append(f"all_edge_r2_loss={lr2:.6f}")
    tsn = _fin(epoch_log, "train/num_target_stream_edges_weighted", "train_num_target_stream_edges_weighted")
    tsy = _fin(
        epoch_log,
        "train/num_target_stream_edges_with_yaml_override",
        "train_num_target_stream_edges_with_yaml_override",
    )
    if tsn is not None:
        parts.append(f"target_stream_edges={int(tsn)}")
    if tsy is not None:
        parts.append(f"yaml_weight_overrides={int(tsy)}")
    step_target_edges = _fin(epoch_log, "train/edge_step_target_edge_count", "train_edge_step_target_edge_count")
    target_edge_w = _fin(epoch_log, "train/target_edge_weight_mean", "train_target_edge_weight_mean")
    default_edge_w = _fin(epoch_log, "train/default_edge_weight_mean", "train_default_edge_weight_mean")
    if step_target_edges is not None:
        parts.append(f"num_target_edges={int(step_target_edges)}")
    if target_edge_w is not None:
        parts.append(f"target_edge_weight_mean={target_edge_w:.3f}")
    if default_edge_w is not None:
        parts.append(f"default_edge_weight_mean={default_edge_w:.3f}")
    if pcnt is not None:
        parts.append(f"target_incident_edges={int(pcnt)}")
    cov = _fin(epoch_log, "train/target_incident_coverage_edge_frac", "train_target_incident_coverage_edge_frac")
    eff_w = _fin(epoch_log, "train/target_incident_effective_weight_mean", "train_target_incident_effective_weight_mean")
    lratio = _fin(epoch_log, "train/target_incident_loss_ratio_total", "train_target_incident_loss_ratio_total")
    if cov is not None:
        parts.append(f"incident_coverage={cov:.3f}")
    if eff_w is not None:
        parts.append(f"incident_eff_w={eff_w:.3f}")
    if lratio is not None:
        parts.append(f"incident_loss_ratio={lratio:.3f}")
    if ratio is not None:
        parts.append(f"primary_std_ratio_norm={ratio:.3f}")
    recovered = _fin(epoch_log, "train/nonfinite_recovered", "train_nonfinite_recovered")
    if recovered is not None and recovered > 0:
        reason = str(epoch_log.get("train/nonfinite_recovery_reason", "nonfinite"))
        lr_new = _fin(epoch_log, "train/nonfinite_recovery_lr_new")
        suffix = f",lr={lr_new:.2e}" if lr_new is not None else ""
        parts.append(f"nonfinite_recovered={reason}{suffix}")

    te10_edges = _fin(
        epoch_log,
        "val/target_edge_10d_num_edges",
        "val_target_edge_10d_num_edges",
        "target_edge_10d_num_edges",
    )
    if te10_edges is not None and te10_edges > 0:
        target_property_mean_r2 = _fin(
            epoch_log,
            "val/target_edge_property_mean_r2",
            "val_target_edge_property_mean_r2",
            "target_edge_property_mean_r2",
        )
        te10_rmse = _fin(
            epoch_log,
            "val/target_edge_10d_rmse_property_macro",
            "val_target_edge_10d_rmse_property_macro",
            "target_edge_10d_rmse_property_macro",
        )
        parts.append(f"target_edge_10d_edges={int(te10_edges)}")
        if target_property_mean_r2 is not None:
            parts.append(f"target_edge_property_mean_r2={target_property_mean_r2:.6f}")
        if te10_rmse is not None:
            parts.append(f"target_edge_10d_rmse={te10_rmse:.6f}")
        return parts

    main_v = _fin(
        epoch_log,
        f"val/{EVAL_PRIMARY_FRAC_R2_BY_PROCESS}",
        f"val_{EVAL_PRIMARY_FRAC_R2_BY_PROCESS}",
        f"val/{FRAC_R2_SHORT_ALIAS}",
        f"val_{FRAC_R2_SHORT_ALIAS}",
    )
    if main_v is not None:
        parts.append(f"primary_frac_r2_by_process={main_v:.6f}")
    tgt_v = _fin(
        epoch_log,
        f"val/{EVAL_PRIMARY_FRAC_R2_BY_TARGET}",
        f"val_{EVAL_PRIMARY_FRAC_R2_BY_TARGET}",
    )
    if tgt_v is not None:
        parts.append(f"primary_frac_r2_by_target={tgt_v:.6f}")
    tb_v = _fin(
        epoch_log,
        f"val/{EVAL_SECONDARY_AMOUNT_R2_BY_TARGET}",
        f"val_{EVAL_SECONDARY_AMOUNT_R2_BY_TARGET}",
    )
    if tb_v is not None:
        parts.append(f"diag_amount_r2_by_target={tb_v:.6f}")
    form_v = _fin(
        epoch_log,
        f"val/{EVAL_PRIMARY_FRAC_R2_FORMULA_BY_PROCESS}",
        f"val_{EVAL_PRIMARY_FRAC_R2_FORMULA_BY_PROCESS}",
    )
    if form_v is not None:
        parts.append(f"primary_frac_r2_formula_ok={form_v:.6f}")
    n_inc = epoch_log.get("val_n_included_targets", epoch_log.get("val_n_main_verified_targets"))
    if n_inc is not None:
        try:
            parts.append(f"eval_target_rows_included={int(n_inc)}")
        except (TypeError, ValueError):
            pass

    legacy_v = _fin(
        epoch_log,
        EVAL_LEGACY_ANSWER_FRAC_R2,
        f"val_{EVAL_LEGACY_ANSWER_FRAC_R2}",
        "val_legacy_answer_fraction_r2",
        "legacy_answer_fraction_r2",
    )
    if legacy_v is not None:
        parts.append(f"diag_legacy_slot_frac_r2={legacy_v:.6f}")

    h2 = _fin(epoch_log, "metric_target_r2", "val/metric_target_r2")
    c2 = _fin(epoch_log, "metric_tailgas_r2", "val/metric_tailgas_r2")
    if h2 is not None:
        parts.append(f"diag_slot_h2_frac_r2={h2:.6f}")
    if c2 is not None:
        parts.append(f"diag_slot_co2_frac_r2={c2:.6f}")
    return parts


def _fmt(value: float | None, *, digits: int = 4, default: str = "nan") -> str:
    if value is None:
        return default
    return f"{value:.{digits}f}"


def _target_property_r2_text(metrics: Mapping[str, Any], *, epoch_log: bool) -> str:
    parts: list[str] = []
    for display_name, slug in TARGET_PROPERTY_R2_FIELDS:
        if epoch_log:
            value = _fin(
                metrics,
                f"val/target_edge_r2_{slug}",
                f"val_target_edge_r2_{slug}",
                f"target_edge_r2_{slug}",
            )
        else:
            raw = _vm(
                metrics,
                f"val_target_edge_r2_{slug}",
                f"target_edge_r2_{slug}",
            )
            value = raw if math.isfinite(raw) else None
        parts.append(f"{display_name}={_fmt(value, digits=5)}")
    return ", ".join(parts)


def format_epoch_detail_edge_all_line(epoch_log: Mapping[str, float]) -> str:
    """Readable per-epoch dashboard focused on pooled target-property R2."""
    te10_edges = _fin(
        epoch_log,
        "val/target_edge_10d_num_edges",
        "val_target_edge_10d_num_edges",
        "target_edge_10d_num_edges",
    )
    if te10_edges is not None and te10_edges > 0:
        train_edge = _fin(
            epoch_log,
            "train/edge_step_pi_loss_mean",
            "train_edge_step_pi_loss_mean",
            "train/loss_edge_all",
            "train_loss_edge_all",
            "train/loss_edge",
            "train_loss_edge",
        )
        val_edge = _fin(
            epoch_log,
            "val/loss_edge_all",
            "val_loss_edge_all",
            "val/loss_edge",
            "val_loss_edge",
            "val/loss",
            "val_loss",
        )
        te10_props = _fin(
            epoch_log,
            "val/target_edge_10d_num_properties",
            "val_target_edge_10d_num_properties",
            "target_edge_10d_num_properties",
        )
        te10_values = _fin(
            epoch_log,
            "val/target_edge_10d_num_valid_values",
            "val_target_edge_10d_num_valid_values",
            "target_edge_10d_num_valid_values",
        )
        te10_unstable = _fin(
            epoch_log,
            "val/target_edge_10d_num_r2_unstable",
            "val_target_edge_10d_num_r2_unstable",
            "target_edge_10d_num_r2_unstable",
        )
        te10_mae = _fin(
            epoch_log,
            "val/target_edge_10d_mae_property_macro",
            "val_target_edge_10d_mae_property_macro",
            "target_edge_10d_mae_property_macro",
        )
        te10_rmse = _fin(
            epoch_log,
            "val/target_edge_10d_rmse_property_macro",
            "val_target_edge_10d_rmse_property_macro",
            "target_edge_10d_rmse_property_macro",
        )
        target_property_mean_r2 = _fin(
            epoch_log,
            "val/target_edge_property_mean_r2",
            "val_target_edge_property_mean_r2",
            "target_edge_property_mean_r2",
        )
        return (
            "   [epoch-target-property-r2] "
            f"target_edge_property_mean_r2={_fmt(target_property_mean_r2, digits=5)} | "
            f"{_target_property_r2_text(epoch_log, epoch_log=True)} | "
            f"MAE={_fmt(te10_mae)} RMSE={_fmt(te10_rmse)} | "
            f"target_edges={int(te10_edges)} properties={int(te10_props or 0)} "
            f"valid_values={int(te10_values or 0)} unstable={int(te10_unstable or 0)} | "
            f"train_loss(edge_step_pi={_fmt(train_edge)}) val_loss(edge_step_pi={_fmt(val_edge)}) | "
            "legacy_primary_frac=diagnostic_only"
        )

    train_edge = _fin(epoch_log, "train/loss_edge_all", "train_loss_edge_all", "train/loss_edge", "train_loss_edge")
    train_frac = _fin(
        epoch_log,
        "train/loss_primary_frac",
        "train_loss_primary_frac",
        "train/loss_target_frac_feature",
        "train_loss_target_frac_feature",
    )
    train_r2_loss = _fin(epoch_log, "train/loss_primary_frac_r2", "train_loss_primary_frac_r2")
    val_edge = _fin(epoch_log, "val/loss_edge_all", "val_loss_edge_all", "val/loss_edge", "val_loss_edge")
    val_frac = _fin(
        epoch_log,
        "val/loss_primary_frac",
        "val_loss_primary_frac",
        "val/loss_target_frac_feature",
        "val_loss_target_frac_feature",
        "val/loss_target_row_frac",
        "val_loss_target_row_frac",
    )
    val_r2_loss = _fin(epoch_log, "val/loss_primary_frac_r2", "val_loss_primary_frac_r2")

    primary_proc = _fin(
        epoch_log,
        f"val/{EVAL_PRIMARY_FRAC_R2_BY_PROCESS}",
        f"val_{EVAL_PRIMARY_FRAC_R2_BY_PROCESS}",
        f"val/{FRAC_R2_SHORT_ALIAS}",
        f"val_{FRAC_R2_SHORT_ALIAS}",
    )
    primary_tgt = _fin(
        epoch_log,
        f"val/{EVAL_PRIMARY_FRAC_R2_BY_TARGET}",
        f"val_{EVAL_PRIMARY_FRAC_R2_BY_TARGET}",
    )
    formula_proc = _fin(
        epoch_log,
        f"val/{EVAL_PRIMARY_FRAC_R2_FORMULA_BY_PROCESS}",
        f"val_{EVAL_PRIMARY_FRAC_R2_FORMULA_BY_PROCESS}",
    )
    amount_tgt = _fin(
        epoch_log,
        f"val/{EVAL_SECONDARY_AMOUNT_R2_BY_TARGET}",
        f"val_{EVAL_SECONDARY_AMOUNT_R2_BY_TARGET}",
    )
    legacy = _fin(
        epoch_log,
        f"val/{EVAL_LEGACY_ANSWER_FRAC_R2}",
        f"val_{EVAL_LEGACY_ANSWER_FRAC_R2}",
        "val/legacy_answer_fraction_macro_r2",
        "val_legacy_answer_fraction_macro_r2",
    )
    edge_mae = _fin(epoch_log, "val/edge_all_mae", "val_edge_all_mae")
    train_std = _fin(epoch_log, "train/primary_frac_std_ratio_norm", "train_primary_frac_std_ratio_norm")
    val_std = _fin(epoch_log, "val/primary_frac_std_ratio_norm", "val_primary_frac_std_ratio_norm")
    val_std_orig = _fin(epoch_log, "val/primary_frac_std_ratio_orig", "val_primary_frac_std_ratio_orig")
    train_count = _fin(epoch_log, "train/primary_frac_count", "train_primary_frac_count")
    val_count = _fin(epoch_log, "val/primary_frac_count", "val_primary_frac_count")
    valid_features = _fin(
        epoch_log,
        "val/primary_frac_r2_valid_feature_count",
        "val_primary_frac_r2_valid_feature_count",
    )
    n_inc = epoch_log.get("val_n_included_targets", epoch_log.get("val_n_main_verified_targets"))
    try:
        rows = int(n_inc) if n_inc is not None else 0
    except (TypeError, ValueError):
        rows = 0

    return (
        "   [epoch-primary] "
        f"FracR2(proc={_fmt(primary_proc, digits=4)}, target={_fmt(primary_tgt, digits=4)}, "
        f"formula={_fmt(formula_proc, digits=4)}) rows={rows} | "
        f"train_loss(edge={_fmt(train_edge)}, frac={_fmt(train_frac)}, r2={_fmt(train_r2_loss)}) "
        f"val_loss(edge={_fmt(val_edge)}, frac={_fmt(val_frac)}, r2={_fmt(val_r2_loss)}) | "
        f"primary_count(train={int(train_count or 0)}, val={int(val_count or 0)}, "
        f"r2_features={int(valid_features or 0)}) "
        f"std_ratio_norm(train={_fmt(train_std, digits=3)}, val={_fmt(val_std, digits=3)}, "
        f"val_orig={_fmt(val_std_orig, digits=3)}) | "
        f"diag(edge_mae_norm={_fmt(edge_mae)}, amountR2_target={_fmt(amount_tgt, digits=4)}, "
        f"legacy_slotR2={_fmt(legacy, digits=4)})"
    )


def _vm(val_metrics: Mapping[str, Any], *keys: str, default: float = float("nan")) -> float:
    v = pick_from_mapping(val_metrics, *keys, default=default)
    return float(v) if math.isfinite(v) else default


def format_val_metric_edge_all_line(val_metrics: Mapping[str, Any]) -> str:
    """One-line validation metric summary for edge_all (per-epoch print)."""
    lt = _vm(val_metrics, "loss_total")
    le = _vm(val_metrics, "loss_edge_all", "loss_edge")
    te10_edges = _vm(val_metrics, "target_edge_10d_num_edges", "val_target_edge_10d_num_edges", default=0.0)
    if te10_edges > 0:
        te10_props = _vm(val_metrics, "target_edge_10d_num_properties", "val_target_edge_10d_num_properties", default=0.0)
        te10_values = _vm(val_metrics, "target_edge_10d_num_valid_values", "val_target_edge_10d_num_valid_values", default=0.0)
        te10_unstable = _vm(
            val_metrics,
            "target_edge_10d_num_r2_unstable",
            "val_target_edge_10d_num_r2_unstable",
            default=0.0,
        )
        te10_mae = _vm(val_metrics, "target_edge_10d_mae_property_macro", "val_target_edge_10d_mae_property_macro")
        te10_rmse = _vm(val_metrics, "target_edge_10d_rmse_property_macro", "val_target_edge_10d_rmse_property_macro")
        target_property_mean_r2 = _vm(
            val_metrics,
            "val_target_edge_property_mean_r2",
            "target_edge_property_mean_r2",
        )
        return (
            "   [val-target-property-r2][edge_all] "
            f"target_edge_property_mean_r2={target_property_mean_r2:.6f} | "
            f"{_target_property_r2_text(val_metrics, epoch_log=False)} | "
            f"mae_property_macro={te10_mae:.6f} "
            f"rmse_property_macro={te10_rmse:.6f} | "
            f"target_edges={int(te10_edges)} properties={int(te10_props)} "
            f"valid_values={int(te10_values)} unstable={int(te10_unstable)} | "
            f"loss_total={lt:.6f} edge={le:.6f} legacy_primary_frac=diagnostic_only"
        )

    lf = _vm(val_metrics, "loss_primary_frac", "loss_target_frac_feature", "loss_target_row_frac", "loss_v4_targets_frac")
    lr2 = _vm(val_metrics, "loss_all_edge_r2", "r2_loss_all", "loss_primary_frac_r2", default=0.0)
    la = _vm(val_metrics, "loss_target_row_amount", "loss_v4_targets_amount", "loss_v4_targets")
    pcnt = _vm(val_metrics, "primary_frac_count", default=0.0)
    vfeat = _vm(val_metrics, "primary_frac_r2_valid_feature_count", default=0.0)
    ratio = _vm(val_metrics, "primary_frac_std_ratio_norm", default=float("nan"))
    ratio_orig = _vm(val_metrics, "primary_frac_std_ratio_orig", default=float("nan"))
    edge_mae = _vm(val_metrics, "edge_all_mae", default=float("nan"))
    primary = _vm(
        val_metrics,
        EVAL_PRIMARY_FRAC_R2_BY_PROCESS,
        f"val_{EVAL_PRIMARY_FRAC_R2_BY_PROCESS}",
        FRAC_R2_SHORT_ALIAS,
        "process_balanced_main_target_frac_r2_main_verified",
    )
    primary_target = _vm(
        val_metrics,
        EVAL_PRIMARY_FRAC_R2_BY_TARGET,
        f"val_{EVAL_PRIMARY_FRAC_R2_BY_TARGET}",
        "target_balanced_main_target_frac_r2_main_verified",
    )
    secondary = _vm(
        val_metrics,
        EVAL_SECONDARY_AMOUNT_R2_BY_TARGET,
        f"val_{EVAL_SECONDARY_AMOUNT_R2_BY_TARGET}",
        AMOUNT_SECONDARY_METRIC_NAME,
        AMOUNT_R2_SHORT_ALIAS,
    )
    formula = _vm(
        val_metrics,
        EVAL_PRIMARY_FRAC_R2_FORMULA_BY_PROCESS,
        f"val_{EVAL_PRIMARY_FRAC_R2_FORMULA_BY_PROCESS}",
    )
    n_tgt = _vm(
        val_metrics,
        "eval_target_rows_count",
        "eval_target_rows_evaluated",
        f"val_{EVAL_TARGET_ROWS_EVALUATED}",
        f"val/{EVAL_TARGET_ROWS_EVALUATED}",
        "target_v4_num_targets_evaluated",
        default=0.0,
    )
    h2 = _vm(val_metrics, "metric_target_r2")
    c2 = _vm(val_metrics, "metric_tailgas_r2")
    return (
        f"   [val-primary][edge_all] "
        f"primary_frac_r2_by_process={primary:.6f} "
        f"primary_frac_r2_by_target={primary_target:.6f} "
        f"primary_frac_r2_formula_ok={formula:.6f} "
        f"target_rows={int(n_tgt)} | "
        f"loss_total={lt:.6f} edge={le:.6f} primary_frac={lf:.6f} all_edge_r2_loss={lr2:.6f} "
        f"primary_count={int(pcnt)} r2_valid_features={int(vfeat)} "
        f"std_ratio_norm={ratio:.3f} std_ratio_orig={ratio_orig:.3f} edge_mae_norm={edge_mae:.6f} | "
        f"diag_amount_r2_by_target={secondary:.6f} diag_amount_loss={la:.6f} "
        f"diag_slot_h2={h2:.6f} diag_slot_co2={c2:.6f}"
    )
