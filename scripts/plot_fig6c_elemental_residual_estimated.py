"""Create ONLY Fig. 6c: pooled elemental-balance residual distribution.

No per-sample atom residuals, no no-physics prediction exports, and no
element-specific C/H/O/N conservation summaries are retained in the final
result sources.  This script therefore draws only the evaluator's pooled
atom-conservation result, with continuous distributions reconstructed from
the saved threshold and fold-summary statistics.  It never invokes a model
or defines a new residual equation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
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
FULL_PHYSICS_FOLD_SUMMARIES = (
    ROOT / "outputs" / "0819final" / "three_server_aggregate" / "physics_conservation_all_runs.csv"
)
BALANCE_IMPLEMENTATION = ROOT / "src" / "process_graph" / "experiment" / "node_balance_pi.py"
CONSERVATION_EVALUATOR = ROOT / "scripts" / "evaluate_physics_conservation.py"
OUTPUT_DIR = (
    ROOT / "outputs" / "0819final" / "Paper_Tables_Final3_figures" / "final"
    / "Fig6c_elemental_residual_estimated"
)

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
REACTIVE_UNIT_TYPES = ("smr_reactor", "wgs_reactor", "burner")
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
    frames: list[pd.DataFrame] = []
    for method, model_name in METHODS.items():
        part = table.loc[
            (table["Model"].astype(str) == model_name)
            & (table["Conservation term"].astype(str).str.lower() == "atom"),
            ["Tolerance", "Mean satisfaction rate", "Standard deviation"],
        ].copy()
        part["threshold"] = _parse_threshold(part["Tolerance"])
        part = part.drop_duplicates(subset=["threshold", "Mean satisfaction rate", "Standard deviation"])
        part = part.sort_values("threshold")
        if not np.array_equal(part["threshold"].to_numpy(dtype=float), THRESHOLDS):
            raise RuntimeError(f"Unexpected atom thresholds for {method}: {part['threshold'].tolist()}")
        frames.append(
            pd.DataFrame(
                {
                    "method": method,
                    "element_scope": "Pooled C/H/O/N",
                    "statistic": ["Satisfaction @ 1%", "Satisfaction @ 5%", "Satisfaction @ 10%"],
                    "threshold": part["threshold"].to_numpy(dtype=float),
                    "value": part["Mean satisfaction rate"].to_numpy(dtype=float),
                    "fold_standard_deviation": part["Standard deviation"].to_numpy(dtype=float),
                    "source_file": str(workbook),
                    "source_type": "OBSERVED",
                    "notes": "Existing evaluator pools valid C/H/O/N atom residual entries; no element-specific values are saved.",
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


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
        raise RuntimeError("The manuscript and original constraint workbooks disagree on atom observations.")


def _full_physics_atom_folds() -> pd.DataFrame:
    table = pd.read_csv(FULL_PHYSICS_FOLD_SUMMARIES)
    needed = {
        "phase", "term", "fold", "valid_residual_count", "normalized_abs_residual_mae",
        "normalized_residual_rmse", "normalized_abs_residual_max",
        "satisfaction_rate_abs_le_0.01", "satisfaction_rate_abs_le_0.05", "satisfaction_rate_abs_le_0.10",
    }
    missing = needed.difference(table.columns)
    if missing:
        raise RuntimeError(f"{FULL_PHYSICS_FOLD_SUMMARIES} lacks: {sorted(missing)}")
    folds = table.loc[(table["phase"] == "Phase 1") & (table["term"] == "atom")].copy().sort_values("fold")
    if len(folds) != 5 or folds["fold"].nunique() != 5:
        raise RuntimeError("Expected five full-physics multi-process atom fold summaries.")
    folds["derived_abs_residual_sd"] = np.sqrt(
        np.maximum(0.0, folds["normalized_residual_rmse"] ** 2 - folds["normalized_abs_residual_mae"] ** 2)
    )
    return folds


def _full_physics_moments(folds: pd.DataFrame) -> tuple[float, float, float, float, int]:
    count = folds["valid_residual_count"].to_numpy(dtype=float)
    mean = float(np.sum(count * folds["normalized_abs_residual_mae"].to_numpy(dtype=float)) / count.sum())
    mean_square = float(np.sum(count * folds["normalized_residual_rmse"].to_numpy(dtype=float) ** 2) / count.sum())
    std = float(np.sqrt(max(0.0, mean_square - mean**2)))
    return (
        mean,
        std,
        float(folds["normalized_abs_residual_mae"].std(ddof=1)),
        float(folds["derived_abs_residual_sd"].std(ddof=1)),
        int(count.sum()),
    )


def _verify_evaluator_definition() -> None:
    """Verify scope/normalization semantics without recomputing residuals."""
    implementation = BALANCE_IMPLEMENTATION.read_text(encoding="utf-8")
    evaluator = CONSERVATION_EVALUATOR.read_text(encoding="utf-8")
    required_implementation = (
        'PI_ATOM_ORDER: tuple[str, ...] = ("C", "H", "O", "N")',
        'REACTIVE_NODE_TYPES: tuple[str, ...] = ("smr_reactor", "wgs_reactor", "burner")',
        "atom_base = reactive",
        "atom_residual, atom_valid, _ = _reduce_balance(",
        "scale = true_in_abs + true_out_abs",
        "residual = residual / scale.detach()",
    )
    if any(term not in implementation for term in required_implementation):
        raise RuntimeError("The current elemental-conservation implementation differs from the inspected definition.")
    if "TERMS = (\"mass\", \"component\", \"atom\")" not in evaluator or "THRESHOLDS = ((\"0.01\", \"0p01\"), (\"0.05\", \"0p05\"), (\"0.10\", \"0p10\"))" not in evaluator:
        raise RuntimeError("The saved conservation evaluator no longer exposes the expected pooled atom statistics.")


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
    second_moment = weight * (ln_variance + ln_mean**2) + (1.0 - weight) * (gamma_variance + gamma_mean**2)
    return float(mean), float(np.sqrt(max(0.0, second_moment - mean**2)))


def _multi_start(
    residual: Callable[[np.ndarray], np.ndarray], lower: np.ndarray, upper: np.ndarray, n_starts: int, rng: np.random.Generator
) -> np.ndarray:
    best_parameters: np.ndarray | None = None
    best_score = np.inf
    for _ in range(n_starts):
        result = least_squares(residual, rng.uniform(lower, upper), bounds=(lower, upper), max_nfev=20_000)
        score = float(np.sum(residual(result.x) ** 2))
        if result.success and np.isfinite(score) and score < best_score:
            best_parameters, best_score = result.x, score
    if best_parameters is None:
        raise RuntimeError("All distribution initializations failed.")
    return best_parameters


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
            mean, std = _single_moments(family, parameters)
            parts.extend((np.asarray([(mean - mean_value) / mean_sd]), np.asarray([(std - std_value) / std_sd])))
        return np.concatenate(parts)

    parameters = _multi_start(residual, np.array([-4.0, -12.0]), np.array([1.5, 0.0]), 32, rng)
    standardized = residual(parameters)
    mean, std = _single_moments(family, parameters)
    return Fit(method, family, parameters, _single_cdf(family, parameters, x), mean, std, float(np.sum(standardized**2)), float(np.max(np.abs(standardized))))


def _fit_mixture(
    method: str, points: pd.DataFrame, moments: tuple[tuple[float, float], tuple[float, float]], rng: np.random.Generator
) -> Fit:
    x = points["threshold"].to_numpy(dtype=float)
    observed = points["value"].to_numpy(dtype=float)
    uncertainty = points["fold_standard_deviation"].to_numpy(dtype=float)
    (mean_value, mean_sd), (std_value, std_sd) = moments

    def residual(parameters: np.ndarray) -> np.ndarray:
        mean, std = _mixture_moments(parameters)
        return np.r_[
            (_mixture_cdf(parameters, x) - observed) / uncertainty,
            (mean - mean_value) / mean_sd,
            (std - std_value) / std_sd,
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
    return Fit(method, "Log-normal + Gamma mixture", parameters, _mixture_cdf(parameters, x), mean, std, float(np.sum(standardized**2)), float(np.max(np.abs(standardized))))


def _fit_distributions(
    observed: pd.DataFrame, full_moments: tuple[float, float, float, float, int]
) -> tuple[dict[str, list[Fit]], dict[str, Fit]]:
    full_mean, full_std, full_mean_fold_sd, full_std_fold_sd, _ = full_moments
    candidates: dict[str, list[Fit]] = {}
    selected: dict[str, Fit] = {}
    rng = np.random.default_rng(SEED)
    for method in METHODS:
        points = observed.loc[observed["method"] == method].sort_values("threshold")
        moments = ((full_mean, full_mean_fold_sd), (full_std, full_std_fold_sd)) if method == "Full physics-informed model" else None
        trials = [_fit_single(method, family, points, moments, rng) for family in ("Log-normal", "Gamma", "Weibull")]
        best_single = min(trials, key=lambda fit: fit.normalized_sum_squared_error)
        if best_single.max_abs_standardized_error > 0.25:
            if moments is None:
                raise RuntimeError(f"Mixture required for {method}, but no stored moment constraints exist.")
            mixture = _fit_mixture(method, points, moments, rng)
            trials.append(mixture)
            chosen = min((best_single, mixture), key=lambda fit: fit.normalized_sum_squared_error)
        else:
            chosen = best_single
        trials = [
            replace(
                fit,
                selected=fit.family == chosen.family and np.allclose(fit.parameters, chosen.parameters),
                plausible=fit.max_abs_standardized_error <= 1.0,
            )
            for fit in trials
        ]
        candidates[method] = trials
        selected[method] = next(fit for fit in trials if fit.selected)
    return candidates, selected


def _cdf(fit: Fit, x: np.ndarray) -> np.ndarray:
    return _mixture_cdf(fit.parameters, x) if fit.family == "Log-normal + Gamma mixture" else _single_cdf(fit.family, fit.parameters, x)


def _quantile(fit: Fit, probability: float) -> float:
    lower, upper = 0.0, 1.0
    while float(_cdf(fit, np.array([upper]))[0]) < probability and upper < 1.0e12:
        upper *= 2.0
    if upper >= 1.0e12 and float(_cdf(fit, np.array([upper]))[0]) < probability:
        return float("nan")
    return float(brentq(lambda value: float(_cdf(fit, np.array([value]))[0] - probability), lower, upper))


def _write_outputs(
    observed: pd.DataFrame,
    folds: pd.DataFrame,
    full_moments: tuple[float, float, float, float, int],
    candidates: dict[str, list[Fit]],
    selected: dict[str, Fit],
) -> tuple[Path, Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observed_path = OUTPUT_DIR / "Fig6c_elemental_observed_summary.csv"
    reconstructed_path = OUTPUT_DIR / "Fig6c_elemental_reconstructed_summary.csv"
    fit_path = OUTPUT_DIR / "Fig6c_elemental_fit_parameters.csv"
    validation_path = OUTPUT_DIR / "Fig6c_elemental_validation.csv"
    full_mean, full_std, full_mean_fold_sd, full_std_fold_sd, full_count = full_moments

    observed_rows: list[dict[str, object]] = [
        {
            "method": "Figure", "element_scope": "Pooled C/H/O/N", "statistic": "result classification",
            "threshold": np.nan, "value": np.nan, "fold_standard_deviation": np.nan,
            "contributing_entries": "not_applicable", "source_type": "ESTIMATED",
            "source_file": "not_applicable", "notes": CLASSIFICATION,
        },
        {
            "method": "Figure", "element_scope": "Pooled C/H/O/N", "statistic": "element-specific availability",
            "threshold": np.nan, "value": np.nan, "fold_standard_deviation": np.nan,
            "contributing_entries": "not_saved", "source_type": "NOT_AVAILABLE",
            "source_file": "not_applicable", "notes": "No C/H/O/N-specific residual values or summaries were found; separate element curves are not identifiable.",
        },
    ]
    for row in observed.itertuples(index=False):
        observed_rows.append(
            {
                "method": row.method, "element_scope": row.element_scope, "statistic": row.statistic,
                "threshold": row.threshold, "value": row.value, "fold_standard_deviation": row.fold_standard_deviation,
                "contributing_entries": "not_saved" if row.method.startswith("Without") else full_count,
                "source_type": row.source_type, "source_file": row.source_file, "notes": row.notes,
            }
        )
    derived = [
        ("valid pooled atom-residual entries", full_count, np.nan, "Count-weighted total across five held-out folds."),
        ("mean normalized absolute atom residual", full_mean, full_mean_fold_sd, "Count-weighted pooled mean; uncertainty is fold-to-fold SD."),
        ("SD of normalized absolute atom residual", full_std, full_std_fold_sd, "Derived as sqrt(E[R^2] - E[|R|]^2); uncertainty is fold-to-fold SD."),
        ("normalized atom residual RMSE", float(folds["normalized_residual_rmse"].mean()), float(folds["normalized_residual_rmse"].std(ddof=1)), "Arithmetic fold mean and sample SD."),
        ("maximum normalized absolute atom residual", float(folds["normalized_abs_residual_max"].max()), np.nan, "Maximum across saved held-out fold summaries."),
    ]
    for statistic, value, sd, note in derived:
        observed_rows.append(
            {
                "method": "Full physics-informed model", "element_scope": "Pooled C/H/O/N", "statistic": statistic,
                "threshold": np.nan, "value": value, "fold_standard_deviation": sd,
                "contributing_entries": full_count, "source_type": "DERIVED", "source_file": str(FULL_PHYSICS_FOLD_SUMMARIES), "notes": note,
            }
        )
    pd.DataFrame(observed_rows).to_csv(observed_path, index=False)

    reconstructed_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    for method, fits in candidates.items():
        chosen = selected[method]
        reconstructed_rows.append(
            {
                "method": method,
                "element_scope": "Pooled C/H/O/N",
                "selected_distribution": chosen.family,
                "reconstructed_q05": _quantile(chosen, 0.05),
                "reconstructed_q25": _quantile(chosen, 0.25),
                "reconstructed_median": _quantile(chosen, 0.50),
                "reconstructed_q75": _quantile(chosen, 0.75),
                "reconstructed_q95": _quantile(chosen, 0.95),
                "reconstructed_mean": chosen.mean,
                "reconstructed_std": chosen.std,
                "result_classification": CLASSIFICATION,
                "notes": "Distributional values are reconstructed from saved pooled summaries; not empirical element-level quantiles.",
            }
        )
        points = observed.loc[observed["method"] == method].sort_values("threshold")
        for fit in fits:
            row: dict[str, object] = {
                "method": method, "element_scope": "Pooled C/H/O/N", "distribution_family": fit.family,
                "theoretical_mean": fit.mean, "theoretical_std": fit.std,
                "normalized_sum_squared_error": fit.normalized_sum_squared_error,
                "max_abs_standardized_constraint_error": fit.max_abs_standardized_error,
                "selected_distribution": fit.selected, "plausible_alternative": fit.plausible,
                "random_seed": SEED, "result_classification": CLASSIFICATION,
                "full_physics_mean_constraint": full_mean if method == "Full physics-informed model" else np.nan,
                "full_physics_mean_fold_sd": full_mean_fold_sd if method == "Full physics-informed model" else np.nan,
                "full_physics_std_constraint": full_std if method == "Full physics-informed model" else np.nan,
                "full_physics_std_fold_sd": full_std_fold_sd if method == "Full physics-informed model" else np.nan,
            }
            if fit.family == "Log-normal + Gamma mixture":
                row.update(
                    {
                        "lognormal_shape": float(np.exp(fit.parameters[0])), "lognormal_scale": float(np.exp(fit.parameters[1])),
                        "gamma_shape": float(np.exp(fit.parameters[2])), "gamma_scale": float(np.exp(fit.parameters[3])),
                        "lognormal_mixture_weight": float(1.0 / (1.0 + np.exp(-fit.parameters[4]))),
                    }
                )
            else:
                row.update({"shape": float(np.exp(fit.parameters[0])), "scale": float(np.exp(fit.parameters[1]))})
            fit_rows.append(row)
        for point, reconstructed in zip(points.itertuples(index=False), chosen.cdf_at_thresholds, strict=True):
            validation_rows.append(
                {
                    "validation_type": "threshold_cdf_fit", "method": method, "element_scope": "Pooled C/H/O/N",
                    "distribution_family": chosen.family, "metric": point.statistic,
                    "observed": point.value, "reconstructed": reconstructed,
                    "absolute_difference": abs(reconstructed - point.value), "observed_uncertainty": point.fold_standard_deviation,
                    "standardized_difference": (reconstructed - point.value) / point.fold_standard_deviation,
                    "source_file": point.source_file,
                }
            )
    selected_full = selected["Full physics-informed model"]
    for metric, observed_value, uncertainty, reconstructed in (
        ("mean normalized absolute atom residual", full_mean, full_mean_fold_sd, selected_full.mean),
        ("SD of normalized absolute atom residual", full_std, full_std_fold_sd, selected_full.std),
    ):
        validation_rows.append(
            {
                "validation_type": "fold_summary_moment_fit", "method": "Full physics-informed model", "element_scope": "Pooled C/H/O/N",
                "distribution_family": selected_full.family, "metric": metric, "observed": observed_value,
                "reconstructed": reconstructed, "absolute_difference": abs(reconstructed - observed_value),
                "observed_uncertainty": uncertainty, "standardized_difference": (reconstructed - observed_value) / uncertainty,
                "source_file": str(FULL_PHYSICS_FOLD_SUMMARIES),
            }
        )
    columns = {0.01: "satisfaction_rate_abs_le_0.01", 0.05: "satisfaction_rate_abs_le_0.05", 0.10: "satisfaction_rate_abs_le_0.10"}
    for point in observed.loc[observed["method"] == "Full physics-informed model"].sort_values("threshold").itertuples(index=False):
        fold_mean = float(folds[columns[float(point.threshold)]].mean())
        validation_rows.append(
            {
                "validation_type": "cross_source_threshold_check", "method": "Full physics-informed model", "element_scope": "Pooled C/H/O/N",
                "distribution_family": "not_applicable", "metric": point.statistic, "observed": point.value,
                "reconstructed": fold_mean, "absolute_difference": abs(fold_mean - point.value),
                "observed_uncertainty": point.fold_standard_deviation,
                "standardized_difference": (fold_mean - point.value) / point.fold_standard_deviation,
                "source_file": str(FULL_PHYSICS_FOLD_SUMMARIES),
            }
        )
    pd.DataFrame(reconstructed_rows).to_csv(reconstructed_path, index=False)
    pd.DataFrame(fit_rows).to_csv(fit_path, index=False)
    pd.DataFrame(validation_rows).to_csv(validation_path, index=False)
    return observed_path, reconstructed_path, fit_path, validation_path


def _plot(observed: pd.DataFrame, candidates: dict[str, list[Fit]], selected: dict[str, Fit]) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(6.65, 4.30), constrained_layout=False)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    grid = np.linspace(0.0, 0.28, 1_401)
    for method, fits in candidates.items():
        color = COLORS[method]
        plausible = [fit for fit in fits if fit.plausible]
        if len(plausible) > 1:
            curves = np.vstack([_cdf(fit, grid) for fit in plausible])
            axis.fill_between(grid, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.15, linewidth=0, zorder=1)
        fit = selected[method]
        axis.plot(grid, _cdf(fit, grid), color=color, linewidth=2.15, label=method, zorder=3)
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
    axis.set_xlabel("Normalized elemental-balance residual", fontsize=10.0, labelpad=6)
    axis.set_ylabel("Cumulative fraction", fontsize=10.0, labelpad=6)
    axis.set_title("Elemental-Balance Residual", fontsize=12.0, pad=13, weight="normal")
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
        "C, H, O, and N are pooled by the saved evaluator; element-specific residuals were not saved.\n"
        "Solid circles: observed threshold satisfaction means (+/- 1 fold SD); shading: plausible-family envelope; curves: reconstructed.",
        ha="center", va="bottom", fontsize=7.0, color="#4B5563",
    )
    figure.subplots_adjust(left=0.135, right=0.975, top=0.865, bottom=0.225)
    pdf_path = OUTPUT_DIR / "Fig6c_elemental_residual_estimated.pdf"
    png_path = OUTPUT_DIR / "Fig6c_elemental_residual_estimated_600dpi.png"
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    figure.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return pdf_path, png_path


def main() -> None:
    sources = (MANUSCRIPT_WORKBOOK, ORIGINAL_CONSTRAINT_WORKBOOK, FULL_PHYSICS_FOLD_SUMMARIES, BALANCE_IMPLEMENTATION, CONSERVATION_EVALUATOR)
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    _verify_evaluator_definition()
    observed = _read_threshold_observations(MANUSCRIPT_WORKBOOK, "Constraint_PINN")
    _verify_threshold_sources(observed)
    folds = _full_physics_atom_folds()
    moments = _full_physics_moments(folds)
    candidates, selected = _fit_distributions(observed, moments)
    observed_path, reconstructed_path, fit_path, validation_path = _write_outputs(observed, folds, moments, candidates, selected)
    pdf_path, png_path = _plot(observed, candidates, selected)

    print(f"RESULT CLASSIFICATION: {CLASSIFICATION}")
    print("SOURCE FILE PATHS USED:")
    for source in sources:
        print(f"- {source}")
    print("\nREACTIVE UNIT TYPES USED:")
    print(", ".join(REACTIVE_UNIT_TYPES))
    print("Element scope: pooled valid C/H/O/N atom residual entries; element-specific summaries are not saved.")
    print("\nCONTRIBUTING ENTRIES:")
    print(f"Full physics-informed model: {moments[4]:,} valid pooled atom residual entries across five held-out folds")
    print("Without physics-informed regularization: valid-entry count not saved in the inspected source results")
    print("\nOBSERVED POOLED ATOM SATISFACTION STATISTICS:")
    print(observed[["method", "threshold", "value", "fold_standard_deviation", "source_type"]].to_string(index=False))
    print("\nFULL-PHYSICS OBSERVED/DERIVED MOMENTS:")
    print(f"mean={moments[0]:.9f}; SD(|residual|)={moments[1]:.9f}; fold SD of mean={moments[2]:.9f}; fold SD of SD={moments[3]:.9f}")
    print("\nSELECTED DISTRIBUTIONS:")
    for method, fit in selected.items():
        print(f"{method}: {fit.family}; objective={fit.normalized_sum_squared_error:.8g}; max standardized error={fit.max_abs_standardized_error:.8g}")
    validation = pd.read_csv(validation_path)
    print("\nSELECTED-FIT VALIDATION:")
    print(validation.loc[validation["validation_type"] != "cross_source_threshold_check", ["method", "metric", "observed", "reconstructed", "absolute_difference"]].to_string(index=False))
    print("\nFIGURE SIZE: 6.65 x 4.30 in")
    print("OUTPUT PATHS:")
    for path in (pdf_path, png_path, observed_path, reconstructed_path, fit_path, validation_path):
        print(f"- {path}")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.titleweight": "normal", "pdf.fonttype": 42, "ps.fonttype": 42})
    main()
