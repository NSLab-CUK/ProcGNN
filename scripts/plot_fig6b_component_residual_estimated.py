"""Create ONLY Fig. 6b: partially reconstructed component-balance ECDF.

No individual component residuals or paired no-physics predictions are saved
in this repository. This script fits positive distribution families to existing
threshold and fold-summary statistics only. It does not load a checkpoint,
invoke a model, or define a new conservation equation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import gamma, lognorm, weibull_min


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_WORKBOOK = (
    ROOT / "outputs" / "0819final" / "Paper_Tables_Final3_figures" / "final"
    / "experiment_results" / "experiment_results.xlsx"
)
ORIGINAL_CONSTRAINT_WORKBOOK = (
    ROOT / "outputs" / "0819final" / "Paper_Tables_Final3_figures" / "final"
    / "constraint_satisfaction_ablation_grouped_bars_typography_20260921_v4_top_left_legend"
    / "constraint_satisfaction_grouped_bar_data.xlsx"
)
FULL_PHYSICS_FOLD_SUMMARIES = ROOT / "outputs" / "0819final" / "three_server_aggregate" / "physics_conservation_all_runs.csv"
BALANCE_IMPLEMENTATION = ROOT / "src" / "process_graph" / "experiment" / "node_balance_pi.py"
CONSERVATION_EVALUATOR = ROOT / "scripts" / "evaluate_physics_conservation.py"
OUTPUT_DIR = ROOT / "outputs" / "0819final" / "Paper_Tables_Final3_figures" / "final" / "Fig6b_component_residual_estimated"

SEED = 260923
THRESHOLDS = np.array([0.01, 0.05, 0.10], dtype=float)
METHODS = {
    "Without physics-informed regularization": "w/o Physics-informed loss",
    "Full physics-informed model": "Proposed (multi-process)",
}
COLORS = {
    "Without physics-informed regularization": "#C44E52",
    "Full physics-informed model": "#1F77B4",
}
CLASSIFICATION = "PARTIALLY RECONSTRUCTED"


@dataclass(frozen=True)
class Fit:
    method: str
    family: str
    parameters: np.ndarray
    cdf_at_thresholds: np.ndarray
    mean: float
    std: float
    normalized_sum_squared_error: float
    max_abs_standardized_error: float
    selected: bool = False
    plausible: bool = False


def _parse_threshold(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0].astype(float) / 100.0


def _read_threshold_observations(workbook: Path, sheet: str) -> pd.DataFrame:
    table = pd.read_excel(workbook, sheet_name=sheet)
    required = {"Model", "Conservation term", "Tolerance", "Mean satisfaction rate", "Standard deviation"}
    missing = required.difference(table.columns)
    if missing:
        raise RuntimeError(f"{workbook} / {sheet} lacks: {sorted(missing)}")
    rows: list[pd.DataFrame] = []
    for method, model_name in METHODS.items():
        part = table.loc[
            (table["Model"].astype(str) == model_name)
            & (table["Conservation term"].astype(str).str.lower() == "component"),
            ["Tolerance", "Mean satisfaction rate", "Standard deviation"],
        ].copy()
        part["threshold"] = _parse_threshold(part["Tolerance"])
        part = part.drop_duplicates(subset=["threshold", "Mean satisfaction rate", "Standard deviation"])
        part = part.sort_values("threshold")
        if not np.array_equal(part["threshold"].to_numpy(dtype=float), THRESHOLDS):
            raise RuntimeError(f"Unexpected component thresholds for {method}: {part['threshold'].tolist()}")
        rows.append(
            pd.DataFrame(
                {
                    "method": method,
                    "statistic": ["Satisfaction @ 1%", "Satisfaction @ 5%", "Satisfaction @ 10%"],
                    "threshold": part["threshold"].to_numpy(dtype=float),
                    "value": part["Mean satisfaction rate"].to_numpy(dtype=float),
                    "fold_standard_deviation": part["Standard deviation"].to_numpy(dtype=float),
                    "source_file": str(workbook),
                    "source_type": "OBSERVED",
                    "notes": "Manuscript-table mean across folds; evaluator pools valid component residual entries.",
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _verify_threshold_sources(observed: pd.DataFrame) -> None:
    original = _read_threshold_observations(ORIGINAL_CONSTRAINT_WORKBOOK, "Physics objective")
    columns = ["method", "threshold", "value", "fold_standard_deviation"]
    left = observed[columns].sort_values(["method", "threshold"]).reset_index(drop=True)
    right = original[columns].sort_values(["method", "threshold"]).reset_index(drop=True)
    if not left[["method", "threshold"]].equals(right[["method", "threshold"]]) or not np.allclose(
        left[["value", "fold_standard_deviation"]].to_numpy(),
        right[["value", "fold_standard_deviation"]].to_numpy(),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("The manuscript and source constraint workbooks disagree on component observations.")


def _full_physics_component_folds() -> pd.DataFrame:
    table = pd.read_csv(FULL_PHYSICS_FOLD_SUMMARIES)
    columns = {
        "phase", "term", "fold", "valid_residual_count", "normalized_abs_residual_mae",
        "normalized_residual_rmse", "normalized_abs_residual_max",
        "satisfaction_rate_abs_le_0.01", "satisfaction_rate_abs_le_0.05", "satisfaction_rate_abs_le_0.10",
    }
    missing = columns.difference(table.columns)
    if missing:
        raise RuntimeError(f"{FULL_PHYSICS_FOLD_SUMMARIES} lacks: {sorted(missing)}")
    folds = table.loc[(table["phase"] == "Phase 1") & (table["term"] == "component")].copy().sort_values("fold")
    if len(folds) != 5 or folds["fold"].nunique() != 5:
        raise RuntimeError("Expected five Full-physics multi-process component fold summaries.")
    folds["derived_abs_residual_sd"] = np.sqrt(
        np.maximum(0.0, folds["normalized_residual_rmse"] ** 2 - folds["normalized_abs_residual_mae"] ** 2)
    )
    return folds


def _verify_evaluator_definition() -> None:
    """Guard the pooled, non-reactive component definition without recalculation."""
    implementation = BALANCE_IMPLEMENTATION.read_text(encoding="utf-8")
    evaluator = CONSERVATION_EVALUATOR.read_text(encoding="utf-8")
    implementation_terms = (
        "PI_SPECIES_ORDER",
        "NONREACTIVE_NODE_TYPES",
        "nonreactive = internal & _unit_name_mask(node_units, NONREACTIVE_NODE_TYPES)",
        "component_valid",
        "component_residual",
    )
    if any(term not in implementation for term in implementation_terms):
        raise RuntimeError("The saved evaluator implementation no longer exposes the expected component pooling definition.")
    if "THRESHOLDS = ((\"0.01\", \"0p01\"), (\"0.05\", \"0p05\"), (\"0.10\", \"0p10\"))" not in evaluator:
        raise RuntimeError("The saved conservation evaluator no longer exposes the expected tolerance statistic.")


def _full_physics_moments(folds: pd.DataFrame) -> tuple[float, float, float, float, int]:
    count = folds["valid_residual_count"].to_numpy(dtype=float)
    mean = float(np.sum(count * folds["normalized_abs_residual_mae"].to_numpy(dtype=float)) / count.sum())
    mean_square = float(np.sum(count * folds["normalized_residual_rmse"].to_numpy(dtype=float) ** 2) / count.sum())
    abs_residual_sd = float(np.sqrt(max(0.0, mean_square - mean**2)))
    mean_fold_sd = float(folds["normalized_abs_residual_mae"].std(ddof=1))
    abs_residual_sd_fold_sd = float(folds["derived_abs_residual_sd"].std(ddof=1))
    return mean, abs_residual_sd, mean_fold_sd, abs_residual_sd_fold_sd, int(count.sum())


def _single_cdf(family: str, parameters: np.ndarray, x: np.ndarray) -> np.ndarray:
    shape, scale = np.exp(parameters)
    if family == "Log-normal":
        return lognorm.cdf(x, s=shape, scale=scale)
    if family == "Gamma":
        return gamma.cdf(x, a=shape, scale=scale)
    if family == "Weibull":
        return weibull_min.cdf(x, c=shape, scale=scale)
    raise ValueError(family)


def _single_moments(family: str, parameters: np.ndarray) -> tuple[float, float]:
    shape, scale = np.exp(parameters)
    if family == "Log-normal":
        return float(lognorm.mean(s=shape, scale=scale)), float(lognorm.std(s=shape, scale=scale))
    if family == "Gamma":
        return float(gamma.mean(a=shape, scale=scale)), float(gamma.std(a=shape, scale=scale))
    if family == "Weibull":
        return float(weibull_min.mean(c=shape, scale=scale)), float(weibull_min.std(c=shape, scale=scale))
    raise ValueError(family)


def _mixture_cdf(parameters: np.ndarray, x: np.ndarray) -> np.ndarray:
    ln_shape, ln_scale, gamma_shape, gamma_scale = np.exp(parameters[:4])
    weight = 1.0 / (1.0 + np.exp(-parameters[4]))
    return weight * lognorm.cdf(x, s=ln_shape, scale=ln_scale) + (1.0 - weight) * gamma.cdf(
        x, a=gamma_shape, scale=gamma_scale
    )


def _mixture_moments(parameters: np.ndarray) -> tuple[float, float]:
    ln_shape, ln_scale, gamma_shape, gamma_scale = np.exp(parameters[:4])
    weight = 1.0 / (1.0 + np.exp(-parameters[4]))
    ln_mean, ln_variance = lognorm.stats(s=ln_shape, scale=ln_scale, moments="mv")
    gamma_mean, gamma_variance = gamma.stats(a=gamma_shape, scale=gamma_scale, moments="mv")
    mean = weight * ln_mean + (1.0 - weight) * gamma_mean
    second_moment = weight * (ln_variance + ln_mean**2) + (1.0 - weight) * (
        gamma_variance + gamma_mean**2
    )
    return float(mean), float(np.sqrt(max(0.0, second_moment - mean**2)))


def _multi_start(
    residual: Callable[[np.ndarray], np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    n_starts: int,
    rng: np.random.Generator,
) -> np.ndarray:
    best: np.ndarray | None = None
    best_objective = np.inf
    for _ in range(n_starts):
        initial = rng.uniform(lower, upper)
        result = least_squares(residual, initial, bounds=(lower, upper), max_nfev=20_000)
        score = float(np.sum(residual(result.x) ** 2))
        if result.success and np.isfinite(score) and score < best_objective:
            best, best_objective = result.x, score
    if best is None:
        raise RuntimeError("All distribution initializations failed.")
    return best


def _fit_single(
    method: str,
    family: str,
    points: pd.DataFrame,
    moments: tuple[tuple[float, float], tuple[float, float]] | None,
    rng: np.random.Generator,
) -> Fit:
    x = points["threshold"].to_numpy(dtype=float)
    observed = points["value"].to_numpy(dtype=float)
    uncertainty = points["fold_standard_deviation"].to_numpy(dtype=float)

    def residual(parameters: np.ndarray) -> np.ndarray:
        parts = [(_single_cdf(family, parameters, x) - observed) / uncertainty]
        if moments is not None:
            (mean_value, mean_sd), (std_value, std_sd) = moments
            predicted_mean, predicted_std = _single_moments(family, parameters)
            parts.append(np.asarray([(predicted_mean - mean_value) / mean_sd]))
            parts.append(np.asarray([(predicted_std - std_value) / std_sd]))
        return np.concatenate(parts)

    parameters = _multi_start(residual, np.array([-4.0, -12.0]), np.array([1.5, 0.0]), 32, rng)
    standardized = residual(parameters)
    mean, std = _single_moments(family, parameters)
    return Fit(
        method, family, parameters, _single_cdf(family, parameters, x), mean, std,
        float(np.sum(standardized**2)), float(np.max(np.abs(standardized))),
    )


def _fit_mixture(
    method: str,
    points: pd.DataFrame,
    moments: tuple[tuple[float, float], tuple[float, float]],
    rng: np.random.Generator,
) -> Fit:
    x = points["threshold"].to_numpy(dtype=float)
    observed = points["value"].to_numpy(dtype=float)
    uncertainty = points["fold_standard_deviation"].to_numpy(dtype=float)
    (mean_value, mean_sd), (std_value, std_sd) = moments

    def residual(parameters: np.ndarray) -> np.ndarray:
        predicted_mean, predicted_std = _mixture_moments(parameters)
        return np.r_[
            (_mixture_cdf(parameters, x) - observed) / uncertainty,
            (predicted_mean - mean_value) / mean_sd,
            (predicted_std - std_value) / std_sd,
        ]

    parameters = _multi_start(
        residual,
        np.array([-4.0, -12.0, -4.0, -12.0, -10.0]),
        np.array([1.5, 0.0, 1.5, 0.0, 10.0]),
        72,
        rng,
    )
    standardized = residual(parameters)
    mean, std = _mixture_moments(parameters)
    return Fit(
        method, "Log-normal + Gamma mixture", parameters, _mixture_cdf(parameters, x), mean, std,
        float(np.sum(standardized**2)), float(np.max(np.abs(standardized))),
    )


def _fit_distributions(
    observed: pd.DataFrame,
    full_moments: tuple[float, float, float, float, int],
) -> tuple[dict[str, list[Fit]], dict[str, Fit]]:
    full_mean, full_sd, full_mean_fold_sd, full_sd_fold_sd, _ = full_moments
    rng = np.random.default_rng(SEED)
    candidates: dict[str, list[Fit]] = {}
    selected: dict[str, Fit] = {}
    for method in METHODS:
        points = observed.loc[observed["method"] == method].sort_values("threshold")
        moment_constraints = (
            ((full_mean, full_mean_fold_sd), (full_sd, full_sd_fold_sd))
            if method == "Full physics-informed model"
            else None
        )
        trial = [_fit_single(method, family, points, moment_constraints, rng) for family in ("Log-normal", "Gamma", "Weibull")]
        best_single = min(trial, key=lambda fit: fit.normalized_sum_squared_error)
        # A two-component positive mixture is evaluated only where no simple
        # family reproduces every available normalized constraint closely.
        if best_single.max_abs_standardized_error > 0.25:
            if moment_constraints is None:
                raise RuntimeError(f"Mixture required but no moments exist for {method}.")
            mixture = _fit_mixture(method, points, moment_constraints, rng)
            trial.append(mixture)
            chosen = mixture if mixture.normalized_sum_squared_error < best_single.normalized_sum_squared_error else best_single
        else:
            chosen = best_single
        trial = [
            replace(
                fit,
                selected=fit.family == chosen.family and np.allclose(fit.parameters, chosen.parameters),
                plausible=fit.max_abs_standardized_error <= 1.0,
            )
            for fit in trial
        ]
        candidates[method] = trial
        selected[method] = next(fit for fit in trial if fit.selected)
    return candidates, selected


def _fit_cdf(fit: Fit, x: np.ndarray) -> np.ndarray:
    if fit.family == "Log-normal + Gamma mixture":
        return _mixture_cdf(fit.parameters, x)
    return _single_cdf(fit.family, fit.parameters, x)


def _observed_summary(observed: pd.DataFrame, folds: pd.DataFrame, full_moments: tuple[float, float, float, float, int]) -> pd.DataFrame:
    full_mean, full_sd, full_mean_fold_sd, full_sd_fold_sd, full_count = full_moments
    rows = [
        {
            "method": "Figure", "statistic": "result classification", "value": np.nan,
            "fold_standard_deviation": np.nan, "source_file": "not_applicable", "source_type": "ESTIMATED",
            "notes": CLASSIFICATION,
        },
        *observed[["method", "statistic", "value", "fold_standard_deviation", "source_file", "source_type", "notes"]].to_dict("records"),
    ]
    derived = [
        ("pooled valid component residual count", full_count, np.nan, "Count-weighted sum across five folds."),
        ("mean normalized absolute component residual", full_mean, full_mean_fold_sd, "Count-weighted pooled mean; uncertainty is fold-to-fold SD."),
        ("SD of normalized absolute component residual", full_sd, full_sd_fold_sd, "Derived as sqrt(E[R^2] - E[|R|]^2); uncertainty is fold-to-fold SD."),
        ("normalized component residual RMSE", float(folds["normalized_residual_rmse"].mean()), float(folds["normalized_residual_rmse"].std(ddof=1)), "Arithmetic fold mean and sample SD."),
        ("maximum normalized absolute component residual", float(folds["normalized_abs_residual_max"].max()), np.nan, "Maximum across existing held-out fold summaries."),
    ]
    rows.extend(
        {
            "method": "Full physics-informed model", "statistic": statistic, "value": value,
            "fold_standard_deviation": error, "source_file": str(FULL_PHYSICS_FOLD_SUMMARIES),
            "source_type": "DERIVED", "notes": notes,
        }
        for statistic, value, error, notes in derived
    )
    return pd.DataFrame(rows)


def _write_csvs(
    observed: pd.DataFrame,
    folds: pd.DataFrame,
    full_moments: tuple[float, float, float, float, int],
    candidates: dict[str, list[Fit]],
    selected: dict[str, Fit],
) -> tuple[Path, Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "Fig6b_component_observed_summary.csv"
    curve_path = OUTPUT_DIR / "Fig6b_component_reconstructed_curve.csv"
    fit_path = OUTPUT_DIR / "Fig6b_component_distribution_fit.csv"
    validation_path = OUTPUT_DIR / "Fig6b_component_validation.csv"
    _observed_summary(observed, folds, full_moments).to_csv(summary_path, index=False)

    full_mean, full_sd, full_mean_fold_sd, full_sd_fold_sd, _ = full_moments
    grid = np.linspace(0.0, 0.28, 1_401)
    curve_frames: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for method, fits in candidates.items():
        plausible = [fit for fit in fits if fit.plausible]
        if not plausible:
            plausible = [selected[method]]
        cdfs = np.vstack([_fit_cdf(fit, grid) for fit in plausible])
        lower, upper = cdfs.min(axis=0), cdfs.max(axis=0)
        points = observed.loc[observed["method"] == method].sort_values("threshold")
        for fit in fits:
            curve_frames.append(
                pd.DataFrame(
                    {
                        "method": method,
                        "distribution_family": fit.family,
                        "selected_distribution": fit.selected,
                        "plausible_alternative": fit.plausible,
                        "normalized_component_balance_residual": grid,
                        "reconstructed_cumulative_fraction": _fit_cdf(fit, grid),
                        "plausible_envelope_lower": lower,
                        "plausible_envelope_upper": upper,
                        "result_classification": CLASSIFICATION,
                    }
                )
            )
            row: dict[str, object] = {
                "method": method,
                "distribution_family": fit.family,
                "theoretical_mean": fit.mean,
                "theoretical_std": fit.std,
                "normalized_sum_squared_error": fit.normalized_sum_squared_error,
                "max_abs_standardized_constraint_error": fit.max_abs_standardized_error,
                "selected_distribution": fit.selected,
                "plausible_alternative": fit.plausible,
                "random_seed": SEED,
                "full_physics_mean_constraint": full_mean if method == "Full physics-informed model" else np.nan,
                "full_physics_mean_fold_sd": full_mean_fold_sd if method == "Full physics-informed model" else np.nan,
                "full_physics_sd_constraint": full_sd if method == "Full physics-informed model" else np.nan,
                "full_physics_sd_fold_sd": full_sd_fold_sd if method == "Full physics-informed model" else np.nan,
                "result_classification": CLASSIFICATION,
            }
            if fit.family == "Log-normal + Gamma mixture":
                row.update(
                    {
                        "lognormal_shape": float(np.exp(fit.parameters[0])),
                        "lognormal_scale": float(np.exp(fit.parameters[1])),
                        "gamma_shape": float(np.exp(fit.parameters[2])),
                        "gamma_scale": float(np.exp(fit.parameters[3])),
                        "lognormal_mixture_weight": float(1.0 / (1.0 + np.exp(-fit.parameters[4]))),
                    }
                )
            else:
                row.update({"shape": float(np.exp(fit.parameters[0])), "scale": float(np.exp(fit.parameters[1]))})
            parameter_rows.append(row)
        fit = selected[method]
        for point, reconstructed in zip(points.itertuples(index=False), fit.cdf_at_thresholds, strict=True):
            validation_rows.append(
                {
                    "validation_type": "threshold_cdf_fit", "method": method,
                    "distribution_family": fit.family, "metric": point.statistic,
                    "observed": point.value, "reconstructed": reconstructed,
                    "absolute_difference": abs(reconstructed - point.value),
                    "observed_uncertainty": point.fold_standard_deviation,
                    "standardized_difference": (reconstructed - point.value) / point.fold_standard_deviation,
                    "source_file": point.source_file,
                }
            )
    selected_full = selected["Full physics-informed model"]
    for metric, value, uncertainty, reconstructed in (
        ("mean normalized absolute component residual", full_mean, full_mean_fold_sd, selected_full.mean),
        ("SD of normalized absolute component residual", full_sd, full_sd_fold_sd, selected_full.std),
    ):
        validation_rows.append(
            {
                "validation_type": "fold_summary_moment_fit", "method": "Full physics-informed model",
                "distribution_family": selected_full.family, "metric": metric,
                "observed": value, "reconstructed": reconstructed,
                "absolute_difference": abs(reconstructed - value), "observed_uncertainty": uncertainty,
                "standardized_difference": (reconstructed - value) / uncertainty,
                "source_file": str(FULL_PHYSICS_FOLD_SUMMARIES),
            }
        )
    column_by_threshold = {
        0.01: "satisfaction_rate_abs_le_0.01",
        0.05: "satisfaction_rate_abs_le_0.05",
        0.10: "satisfaction_rate_abs_le_0.10",
    }
    for point in observed.loc[observed["method"] == "Full physics-informed model"].sort_values("threshold").itertuples(index=False):
        fold_mean = float(folds[column_by_threshold[float(point.threshold)]].mean())
        validation_rows.append(
            {
                "validation_type": "cross_source_threshold_check", "method": "Full physics-informed model",
                "distribution_family": "not_applicable", "metric": point.statistic,
                "observed": point.value, "reconstructed": fold_mean,
                "absolute_difference": abs(fold_mean - point.value), "observed_uncertainty": point.fold_standard_deviation,
                "standardized_difference": (fold_mean - point.value) / point.fold_standard_deviation,
                "source_file": str(FULL_PHYSICS_FOLD_SUMMARIES),
            }
        )
    pd.concat(curve_frames, ignore_index=True).to_csv(curve_path, index=False)
    pd.DataFrame(parameter_rows).to_csv(fit_path, index=False)
    pd.DataFrame(validation_rows).to_csv(validation_path, index=False)
    return summary_path, curve_path, fit_path, validation_path


def _plot(observed: pd.DataFrame, candidates: dict[str, list[Fit]], selected: dict[str, Fit]) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(6.65, 4.30), constrained_layout=False)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    grid = np.linspace(0.0, 0.28, 1_401)
    for method, fits in candidates.items():
        color = COLORS[method]
        plausible = [fit for fit in fits if fit.plausible]
        if len(plausible) > 1:
            cdfs = np.vstack([_fit_cdf(fit, grid) for fit in plausible])
            axis.fill_between(grid, cdfs.min(axis=0), cdfs.max(axis=0), color=color, alpha=0.15, linewidth=0, zorder=1)
        fit = selected[method]
        axis.plot(grid, _fit_cdf(fit, grid), color=color, linewidth=2.15, label=method, zorder=3)
        points = observed.loc[observed["method"] == method].sort_values("threshold")
        axis.errorbar(
            points["threshold"], points["value"], yerr=points["fold_standard_deviation"],
            fmt="o", color=color, markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.75,
            markersize=6.1, capsize=2.6, elinewidth=0.9, zorder=5,
        )
    for threshold in THRESHOLDS:
        axis.axvline(threshold, color="#AEB6C2", linewidth=0.7, linestyle=(0, (2, 3)), zorder=0)
    axis.set_xlim(0.0, 0.28)
    axis.set_ylim(0.0, 1.02)
    axis.set_xticks(np.arange(0.0, 0.281, 0.05))
    axis.set_yticks(np.arange(0.0, 1.01, 0.2))
    axis.set_xlabel("Normalized component-balance residual", fontsize=10.0, labelpad=6)
    axis.set_ylabel("Cumulative fraction", fontsize=10.0, labelpad=6)
    axis.set_title("(b) Component-Balance Residual Distribution", fontsize=12.0, pad=13, weight="normal")
    axis.tick_params(axis="both", labelsize=9.0, length=3.0, width=0.8, colors="#30343B")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#59616D")
        axis.spines[spine].set_linewidth(0.8)
    axis.legend(
        loc="lower right", fontsize=8.2, frameon=True, framealpha=0.96, facecolor="white",
        edgecolor="#D5DAE1", borderpad=0.55, handlelength=2.2,
    )
    figure.text(
        0.5, 0.012,
        "Solid circles: observed threshold satisfaction means (±1 fold SD).  "
        "Continuous curves: distributions reconstructed from saved summary statistics.",
        ha="center", va="bottom", fontsize=7.6, color="#4B5563",
    )
    # Redraw the explanatory note with an ASCII +/- for consistent font rendering.
    figure.text(
        0.5, 0.012,
        "Solid circles: observed threshold satisfaction means (+/- 1 fold SD); shading: plausible-family envelope; "
        "curves: reconstructed from saved summaries.",
        ha="center", va="bottom", fontsize=7.6, color="#4B5563",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15},
    )
    figure.subplots_adjust(left=0.135, right=0.975, top=0.865, bottom=0.18)
    pdf_path = OUTPUT_DIR / "Fig6b_component_residual_estimated.pdf"
    png_path = OUTPUT_DIR / "Fig6b_component_residual_estimated_600dpi.png"
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    figure.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return pdf_path, png_path


def main() -> None:
    sources = (
        MANUSCRIPT_WORKBOOK,
        ORIGINAL_CONSTRAINT_WORKBOOK,
        FULL_PHYSICS_FOLD_SUMMARIES,
        BALANCE_IMPLEMENTATION,
        CONSERVATION_EVALUATOR,
    )
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    _verify_evaluator_definition()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observed = _read_threshold_observations(MANUSCRIPT_WORKBOOK, "Constraint_PINN")
    _verify_threshold_sources(observed)
    folds = _full_physics_component_folds()
    moments = _full_physics_moments(folds)
    candidates, selected = _fit_distributions(observed, moments)
    summary_path, curve_path, fit_path, validation_path = _write_csvs(observed, folds, moments, candidates, selected)
    pdf_path, png_path = _plot(observed, candidates, selected)

    print(f"RESULT CLASSIFICATION: {CLASSIFICATION}")
    print("SOURCE FILE PATHS USED:")
    for source in sources:
        print(f"- {source}")
    print("\nOBSERVED THRESHOLD STATISTICS:")
    print(observed[["method", "threshold", "value", "fold_standard_deviation", "source_type"]].to_string(index=False))
    print("\nFULL-PHYSICS OBSERVED/DERIVED MOMENTS:")
    print(f"valid residual count={moments[4]:,}; mean={moments[0]:.9f}; SD(|residual|)={moments[1]:.9f}")
    print(f"fold SD of mean={moments[2]:.9f}; fold SD of SD(|residual|)={moments[3]:.9f}")
    print("\nSELECTED DISTRIBUTIONS:")
    for method, fit in selected.items():
        print(f"{method}: {fit.family}; objective={fit.normalized_sum_squared_error:.8g}; max standardized error={fit.max_abs_standardized_error:.8g}")
    validation = pd.read_csv(validation_path)
    print("\nSELECTED-FIT VALIDATION:")
    print(validation[validation["validation_type"] != "cross_source_threshold_check"][["method", "metric", "observed", "reconstructed", "absolute_difference"]].to_string(index=False))
    print("\nFIGURE SIZE: 6.65 x 4.30 in")
    print("OUTPUT PATHS:")
    for path in (pdf_path, png_path, summary_path, curve_path, fit_path, validation_path):
        print(f"- {path}")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.titleweight": "normal", "pdf.fonttype": 42, "ps.fonttype": 42})
    main()
