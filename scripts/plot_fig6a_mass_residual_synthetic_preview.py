"""Create ONLY a clearly labelled synthetic preview for exploratory Fig. 6a.

The repository contains mass-conservation summary statistics but no individual
paired residual samples for the two requested models.  This script fits
positive distributions to those *observed summaries*, then draws deterministic
pseudo-samples for visual planning.  It must never be used as empirical data.
No model, checkpoint, or conservation evaluator is run here.
"""

from __future__ import annotations

from dataclasses import dataclass
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
OUTPUT_DIR = ROOT / "outputs" / "0819final" / "Paper_Tables_Final3_figures" / "final" / "Fig6a_mass_residual_synthetic_preview"

SEED = 260923
N_PSEUDO_SAMPLES = 50_000
N_VISIBLE_JITTER = 175
THRESHOLDS = np.array([0.01, 0.05, 0.10], dtype=float)
METHODS = {
    "No physics": "w/o Physics-informed loss",
    "Full physics": "Proposed (multi-process)",
}
COLORS = {"No physics": "#C44E52", "Full physics": "#1F77B4"}


@dataclass(frozen=True)
class CandidateFit:
    method: str
    family: str
    parameters: np.ndarray
    cdf_at_thresholds: np.ndarray
    theoretical_mean: float
    theoretical_std: float
    objective: float
    max_abs_standardized_error: float
    selected: bool = False
    attempted: bool = True


def _parse_threshold(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace("≤", "", regex=False).str.replace("��", "", regex=False)
    return text.str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0].astype(float) / 100.0


def _threshold_observations(workbook: Path, sheet: str) -> pd.DataFrame:
    frame = pd.read_excel(workbook, sheet_name=sheet)
    required = {"Model", "Conservation term", "Tolerance", "Mean satisfaction rate", "Standard deviation"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{workbook} / {sheet} is missing columns: {sorted(missing)}")
    extracted: list[pd.DataFrame] = []
    for method, model in METHODS.items():
        part = frame.loc[
            (frame["Model"].astype(str) == model)
            & (frame["Conservation term"].astype(str).str.lower() == "mass"),
            ["Tolerance", "Mean satisfaction rate", "Standard deviation"],
        ].copy()
        part["threshold"] = _parse_threshold(part["Tolerance"])
        part = part.drop_duplicates(subset=["threshold", "Mean satisfaction rate", "Standard deviation"])
        part = part.sort_values("threshold")
        if not np.array_equal(part["threshold"].to_numpy(dtype=float), THRESHOLDS):
            raise RuntimeError(f"Unexpected mass thresholds for {method}: {part['threshold'].tolist()}")
        extracted.append(
            pd.DataFrame(
                {
                    "method": method,
                    "metric": ["Satisfaction @ 1%", "Satisfaction @ 5%", "Satisfaction @ 10%"],
                    "threshold": part["threshold"].to_numpy(dtype=float),
                    "observed_value": part["Mean satisfaction rate"].to_numpy(dtype=float),
                    "observed_uncertainty": part["Standard deviation"].to_numpy(dtype=float),
                    "source_file": str(workbook),
                    "source_sheet": sheet,
                    "source_type": "OBSERVED",
                }
            )
        )
    return pd.concat(extracted, ignore_index=True)


def _verify_threshold_sources(observed: pd.DataFrame) -> None:
    original = _threshold_observations(ORIGINAL_CONSTRAINT_WORKBOOK, "Physics objective")
    cols = ["method", "threshold", "observed_value", "observed_uncertainty"]
    left = observed[cols].sort_values(["method", "threshold"]).reset_index(drop=True)
    right = original[cols].sort_values(["method", "threshold"]).reset_index(drop=True)
    if not left[["method", "threshold"]].equals(right[["method", "threshold"]]) or not np.allclose(
        left[["observed_value", "observed_uncertainty"]].to_numpy(),
        right[["observed_value", "observed_uncertainty"]].to_numpy(),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("The manuscript workbook and original constraint workbook disagree on mass statistics.")


def _full_physics_folds() -> pd.DataFrame:
    frame = pd.read_csv(FULL_PHYSICS_FOLD_SUMMARIES)
    required = {
        "phase", "term", "fold", "valid_residual_count", "normalized_abs_residual_mae",
        "normalized_residual_rmse", "normalized_abs_residual_max",
        "satisfaction_rate_abs_le_0.01", "satisfaction_rate_abs_le_0.05", "satisfaction_rate_abs_le_0.10",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{FULL_PHYSICS_FOLD_SUMMARIES} is missing columns: {sorted(missing)}")
    folds = frame.loc[(frame["phase"] == "Phase 1") & (frame["term"] == "mass")].copy().sort_values("fold")
    if len(folds) != 5 or folds["fold"].nunique() != 5:
        raise RuntimeError("Expected exactly five Full-physics multi-process mass fold summaries.")
    # Since E[R^2] = E[|R|^2], the individual-residual SD of |R| is exactly
    # derivable from the saved MAE and RMSE summaries; no samples are invented.
    folds["derived_abs_residual_sd"] = np.sqrt(
        np.maximum(0.0, folds["normalized_residual_rmse"] ** 2 - folds["normalized_abs_residual_mae"] ** 2)
    )
    return folds


def _weighted_full_physics_moments(folds: pd.DataFrame) -> tuple[float, float, float, float, int]:
    count = folds["valid_residual_count"].to_numpy(dtype=float)
    mean = float(np.sum(count * folds["normalized_abs_residual_mae"].to_numpy(dtype=float)) / np.sum(count))
    mean_square = float(np.sum(count * folds["normalized_residual_rmse"].to_numpy(dtype=float) ** 2) / np.sum(count))
    residual_sd = float(np.sqrt(max(0.0, mean_square - mean**2)))
    mean_fold_sd = float(folds["normalized_abs_residual_mae"].std(ddof=1))
    residual_sd_fold_sd = float(folds["derived_abs_residual_sd"].std(ddof=1))
    return mean, residual_sd, mean_fold_sd, residual_sd_fold_sd, int(np.sum(count))


def _single_moments(family: str, parameters: np.ndarray) -> tuple[float, float]:
    shape, scale = np.exp(parameters)
    if family == "Log-normal":
        return float(lognorm.mean(s=shape, scale=scale)), float(lognorm.std(s=shape, scale=scale))
    if family == "Gamma":
        return float(gamma.mean(a=shape, scale=scale)), float(gamma.std(a=shape, scale=scale))
    if family == "Weibull":
        return float(weibull_min.mean(c=shape, scale=scale)), float(weibull_min.std(c=shape, scale=scale))
    raise ValueError(family)


def _single_cdf(family: str, parameters: np.ndarray, x: np.ndarray) -> np.ndarray:
    shape, scale = np.exp(parameters)
    if family == "Log-normal":
        return lognorm.cdf(x, s=shape, scale=scale)
    if family == "Gamma":
        return gamma.cdf(x, a=shape, scale=scale)
    if family == "Weibull":
        return weibull_min.cdf(x, c=shape, scale=scale)
    raise ValueError(family)


def _mixture_cdf(parameters: np.ndarray, x: np.ndarray) -> np.ndarray:
    ln_shape, ln_scale, gamma_shape, gamma_scale = np.exp(parameters[:4])
    lognormal_weight = 1.0 / (1.0 + np.exp(-parameters[4]))
    return (
        lognormal_weight * lognorm.cdf(x, s=ln_shape, scale=ln_scale)
        + (1.0 - lognormal_weight) * gamma.cdf(x, a=gamma_shape, scale=gamma_scale)
    )


def _mixture_moments(parameters: np.ndarray) -> tuple[float, float]:
    ln_shape, ln_scale, gamma_shape, gamma_scale = np.exp(parameters[:4])
    lognormal_weight = 1.0 / (1.0 + np.exp(-parameters[4]))
    ln_mean, ln_variance = lognorm.stats(s=ln_shape, scale=ln_scale, moments="mv")
    gamma_mean, gamma_variance = gamma.stats(a=gamma_shape, scale=gamma_scale, moments="mv")
    mean = lognormal_weight * ln_mean + (1.0 - lognormal_weight) * gamma_mean
    second_moment = lognormal_weight * (ln_variance + ln_mean**2) + (1.0 - lognormal_weight) * (
        gamma_variance + gamma_mean**2
    )
    return float(mean), float(np.sqrt(max(0.0, second_moment - mean**2)))


def _multi_start_fit(
    residual: Callable[[np.ndarray], np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    n_starts: int,
    rng: np.random.Generator,
) -> np.ndarray:
    best_parameters: np.ndarray | None = None
    best_objective = np.inf
    for _ in range(n_starts):
        initial = rng.uniform(lower, upper)
        result = least_squares(residual, initial, bounds=(lower, upper), max_nfev=20_000)
        values = residual(result.x)
        objective = float(np.sum(values**2))
        if result.success and np.isfinite(objective) and objective < best_objective:
            best_parameters = result.x
            best_objective = objective
    if best_parameters is None:
        raise RuntimeError("All distribution-fit initializations failed.")
    return best_parameters


def _fit_single(
    method: str,
    family: str,
    points: pd.DataFrame,
    moment_constraints: tuple[tuple[float, float], tuple[float, float]] | None,
    rng: np.random.Generator,
) -> CandidateFit:
    threshold = points["threshold"].to_numpy(dtype=float)
    observed = points["observed_value"].to_numpy(dtype=float)
    uncertainty = points["observed_uncertainty"].to_numpy(dtype=float)

    def residual(parameters: np.ndarray) -> np.ndarray:
        values = [(_single_cdf(family, parameters, threshold) - observed) / uncertainty]
        if moment_constraints is not None:
            (mean_value, mean_sd), (std_value, std_sd) = moment_constraints
            predicted_mean, predicted_std = _single_moments(family, parameters)
            values.append(np.asarray([(predicted_mean - mean_value) / mean_sd]))
            values.append(np.asarray([(predicted_std - std_value) / std_sd]))
        return np.concatenate(values)

    parameters = _multi_start_fit(
        residual,
        lower=np.array([-4.0, -12.0]),
        upper=np.array([1.5, 0.0]),
        n_starts=32,
        rng=rng,
    )
    standardized = residual(parameters)
    return CandidateFit(
        method=method,
        family=family,
        parameters=parameters,
        cdf_at_thresholds=_single_cdf(family, parameters, threshold),
        theoretical_mean=_single_moments(family, parameters)[0],
        theoretical_std=_single_moments(family, parameters)[1],
        objective=float(np.sum(standardized**2)),
        max_abs_standardized_error=float(np.max(np.abs(standardized))),
    )


def _fit_lognormal_gamma_mixture(
    method: str,
    points: pd.DataFrame,
    moment_constraints: tuple[tuple[float, float], tuple[float, float]],
    rng: np.random.Generator,
) -> CandidateFit:
    threshold = points["threshold"].to_numpy(dtype=float)
    observed = points["observed_value"].to_numpy(dtype=float)
    uncertainty = points["observed_uncertainty"].to_numpy(dtype=float)
    (mean_value, mean_sd), (std_value, std_sd) = moment_constraints

    def residual(parameters: np.ndarray) -> np.ndarray:
        predicted_mean, predicted_std = _mixture_moments(parameters)
        return np.r_[
            (_mixture_cdf(parameters, threshold) - observed) / uncertainty,
            (predicted_mean - mean_value) / mean_sd,
            (predicted_std - std_value) / std_sd,
        ]

    parameters = _multi_start_fit(
        residual,
        lower=np.array([-4.0, -12.0, -4.0, -12.0, -10.0]),
        upper=np.array([1.5, 0.0, 1.5, 0.0, 10.0]),
        n_starts=96,
        rng=rng,
    )
    standardized = residual(parameters)
    mean, std = _mixture_moments(parameters)
    return CandidateFit(
        method=method,
        family="Log-normal + Gamma mixture",
        parameters=parameters,
        cdf_at_thresholds=_mixture_cdf(parameters, threshold),
        theoretical_mean=mean,
        theoretical_std=std,
        objective=float(np.sum(standardized**2)),
        max_abs_standardized_error=float(np.max(np.abs(standardized))),
    )


def _fit_candidates(
    observations: pd.DataFrame,
    moments: tuple[float, float, float, float],
) -> tuple[dict[str, list[CandidateFit]], dict[str, CandidateFit]]:
    rng = np.random.default_rng(SEED)
    full_mean, full_std, full_mean_fold_sd, full_std_fold_sd = moments
    candidates: dict[str, list[CandidateFit]] = {}
    selected: dict[str, CandidateFit] = {}
    for method in METHODS:
        points = observations.loc[observations["method"] == method].sort_values("threshold")
        constraints = (
            ((full_mean, full_mean_fold_sd), (full_std, full_std_fold_sd))
            if method == "Full physics"
            else None
        )
        fitted = [
            _fit_single(method, family, points, constraints, rng)
            for family in ("Log-normal", "Gamma", "Weibull")
        ]
        best_single = min(fitted, key=lambda fit: fit.objective)
        # Keep the simplest family only if it fits every standardized observed
        # constraint to within one quarter of a fold SD.  The Full-physics
        # summaries do not meet this condition with any two-parameter family,
        # so a numerically fitted two-component mixture is warranted.
        if best_single.max_abs_standardized_error > 0.25:
            if constraints is None:
                raise RuntimeError(f"No valid moment constraints for a needed mixture: {method}")
            mixture = _fit_lognormal_gamma_mixture(method, points, constraints, rng)
            fitted.append(mixture)
            best = mixture if mixture.objective < best_single.objective else best_single
        else:
            best = best_single
        candidates[method] = [
            CandidateFit(**{**fit.__dict__, "selected": fit.family == best.family and np.allclose(fit.parameters, best.parameters)})
            for fit in fitted
        ]
        selected[method] = next(fit for fit in candidates[method] if fit.selected)
    return candidates, selected


def _sample(candidate: CandidateFit, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    if candidate.family == "Log-normal":
        shape, scale = np.exp(candidate.parameters)
        return lognorm.rvs(s=shape, scale=scale, size=n_samples, random_state=rng)
    if candidate.family == "Gamma":
        shape, scale = np.exp(candidate.parameters)
        return gamma.rvs(a=shape, scale=scale, size=n_samples, random_state=rng)
    if candidate.family == "Weibull":
        shape, scale = np.exp(candidate.parameters)
        return weibull_min.rvs(c=shape, scale=scale, size=n_samples, random_state=rng)
    if candidate.family == "Log-normal + Gamma mixture":
        ln_shape, ln_scale, gamma_shape, gamma_scale = np.exp(candidate.parameters[:4])
        lognormal_weight = 1.0 / (1.0 + np.exp(-candidate.parameters[4]))
        select_lognormal = rng.random(n_samples) < lognormal_weight
        result = np.empty(n_samples, dtype=float)
        result[select_lognormal] = lognorm.rvs(
            s=ln_shape, scale=ln_scale, size=int(select_lognormal.sum()), random_state=rng
        )
        result[~select_lognormal] = gamma.rvs(
            a=gamma_shape, scale=gamma_scale, size=int((~select_lognormal).sum()), random_state=rng
        )
        return result
    raise ValueError(candidate.family)


def _write_fit_parameters(
    candidates: dict[str, list[CandidateFit]],
    observed_moments: tuple[float, float, float, float, int],
) -> Path:
    full_mean, full_std, full_mean_fold_sd, full_std_fold_sd, full_n = observed_moments
    rows: list[dict[str, object]] = []
    for method, method_candidates in candidates.items():
        for fit in method_candidates:
            row: dict[str, object] = {
                "method": method,
                "distribution_family": fit.family,
                "objective_normalized_sum_squared_error": fit.objective,
                "max_abs_standardized_constraint_error": fit.max_abs_standardized_error,
                "theoretical_mean": fit.theoretical_mean,
                "theoretical_std": fit.theoretical_std,
                "selected": fit.selected,
                "fit_initializations": 96 if fit.family.endswith("mixture") else 32,
                "random_seed": SEED,
                "full_physics_mean_constraint": full_mean if method == "Full physics" else np.nan,
                "full_physics_mean_fold_sd": full_mean_fold_sd if method == "Full physics" else np.nan,
                "full_physics_abs_residual_sd_constraint": full_std if method == "Full physics" else np.nan,
                "full_physics_abs_residual_sd_fold_sd": full_std_fold_sd if method == "Full physics" else np.nan,
                "full_physics_valid_residual_count": full_n if method == "Full physics" else np.nan,
                "source_file": str(FULL_PHYSICS_FOLD_SUMMARIES) if method == "Full physics" else str(MANUSCRIPT_WORKBOOK),
            }
            for index, value in enumerate(fit.parameters, start=1):
                row[f"optimization_parameter_{index}"] = float(value)
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
            rows.append(row)
    path = OUTPUT_DIR / "Fig6a_mass_residual_fit_parameters.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_samples(selected: dict[str, CandidateFit]) -> tuple[pd.DataFrame, Path]:
    rng = np.random.default_rng(SEED)
    samples: list[pd.DataFrame] = []
    for method in METHODS:
        values = _sample(selected[method], N_PSEUDO_SAMPLES, rng)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise RuntimeError(f"Synthetic sampler produced invalid residuals for {method}.")
        samples.append(
            pd.DataFrame(
                {
                    "method": method,
                    "pseudo_sample_id": np.arange(1, N_PSEUDO_SAMPLES + 1, dtype=int),
                    "residual": values,
                    "synthetic": True,
                }
            )
        )
    frame = pd.concat(samples, ignore_index=True)
    path = OUTPUT_DIR / "Fig6a_mass_residual_synthetic_samples.csv"
    frame.to_csv(path, index=False)
    return frame, path


def _validation_table(
    observations: pd.DataFrame,
    samples: pd.DataFrame,
    selected: dict[str, CandidateFit],
    observed_moments: tuple[float, float, float, float, int],
) -> tuple[pd.DataFrame, Path]:
    full_mean, full_std, full_mean_fold_sd, full_std_fold_sd, _ = observed_moments
    rows: list[dict[str, object]] = []
    for method in METHODS:
        values = samples.loc[samples["method"] == method, "residual"].to_numpy(dtype=float)
        points = observations.loc[observations["method"] == method].sort_values("threshold")
        for point, theoretical in zip(points.itertuples(index=False), selected[method].cdf_at_thresholds, strict=True):
            synthetic = float(np.mean(values <= point.threshold))
            tolerance = max(0.005, float(point.observed_uncertainty))
            rows.append(
                {
                    "method": method,
                    "metric": point.metric,
                    "source_type": "OBSERVED",
                    "observed": point.observed_value,
                    "synthetic": synthetic,
                    "fitted_distribution": theoretical,
                    "difference_synthetic_minus_observed": synthetic - point.observed_value,
                    "absolute_difference": abs(synthetic - point.observed_value),
                    "observed_uncertainty": point.observed_uncertainty,
                    "acceptance_tolerance": tolerance,
                    "passes_tolerance": abs(synthetic - point.observed_value) <= tolerance,
                    "source_file": point.source_file,
                }
            )
    values = samples.loc[samples["method"] == "Full physics", "residual"].to_numpy(dtype=float)
    for metric, observed, uncertainty, synthetic, fitted in (
        ("Mean normalized mass-balance residual", full_mean, full_mean_fold_sd, float(values.mean()), selected["Full physics"].theoretical_mean),
        ("SD of normalized absolute mass-balance residual", full_std, full_std_fold_sd, float(values.std(ddof=0)), selected["Full physics"].theoretical_std),
    ):
        rows.append(
            {
                "method": "Full physics",
                "metric": metric,
                "source_type": "DERIVED",
                "observed": observed,
                "synthetic": synthetic,
                "fitted_distribution": fitted,
                "difference_synthetic_minus_observed": synthetic - observed,
                "absolute_difference": abs(synthetic - observed),
                "observed_uncertainty": uncertainty,
                "acceptance_tolerance": max(0.001, uncertainty),
                "passes_tolerance": abs(synthetic - observed) <= max(0.001, uncertainty),
                "source_file": str(FULL_PHYSICS_FOLD_SUMMARIES),
            }
        )
    table = pd.DataFrame(rows)
    if not bool(table["passes_tolerance"].all()):
        failed = table.loc[~table["passes_tolerance"], ["method", "metric", "absolute_difference", "acceptance_tolerance"]]
        raise RuntimeError(f"Synthetic pseudo-samples did not meet observed-summary tolerance:\n{failed.to_string(index=False)}")
    path = OUTPUT_DIR / "Fig6a_mass_residual_synthetic_validation.csv"
    table.to_csv(path, index=False)
    return table, path


def _plot(samples: pd.DataFrame) -> tuple[Path, Path, float]:
    ordered_methods = list(METHODS)
    arrays = [samples.loc[samples["method"] == method, "residual"].to_numpy(dtype=float) for method in ordered_methods]
    # Use the larger method-wise 97.5th percentile only as a visual crop.  The
    # violin and box calculations use all pseudo-samples; no values are dropped
    # from the saved synthetic CSV.
    display_limit = float(np.ceil(max(np.quantile(values, 0.975) for values in arrays) / 0.05) * 0.05)
    # Fig. 6a--d use one fixed canvas so they can be assembled without
    # panel-specific rescaling.
    figure, axis = plt.subplots(figsize=(6.65, 4.30), constrained_layout=False)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    positions = np.arange(len(ordered_methods), dtype=float)
    violin = axis.violinplot(arrays, positions=positions, widths=0.74, showmeans=False, showmedians=False, showextrema=False)
    for body, method in zip(violin["bodies"], ordered_methods, strict=True):
        body.set_facecolor(COLORS[method])
        body.set_edgecolor(COLORS[method])
        body.set_alpha(0.30)
        body.set_linewidth(0.9)
    box = axis.boxplot(
        arrays, positions=positions, widths=0.27, patch_artist=True, showfliers=False,
        medianprops={"color": "#1F2937", "linewidth": 1.4},
        whiskerprops={"color": "#364152", "linewidth": 0.9},
        capprops={"color": "#364152", "linewidth": 0.9},
    )
    for patch, method in zip(box["boxes"], ordered_methods, strict=True):
        patch.set_facecolor("white")
        patch.set_edgecolor(COLORS[method])
        patch.set_linewidth(1.1)
    jitter_rng = np.random.default_rng(SEED + 1)
    for position, values, method in zip(positions, arrays, ordered_methods, strict=True):
        visible_index = jitter_rng.choice(len(values), size=N_VISIBLE_JITTER, replace=False)
        visible = values[visible_index]
        jitter = jitter_rng.uniform(-0.17, 0.17, size=N_VISIBLE_JITTER)
        axis.scatter(
            np.full(N_VISIBLE_JITTER, position) + jitter,
            visible,
            s=11.0,
            color=COLORS[method],
            alpha=0.38,
            linewidths=0.0,
            zorder=4,
        )
    for threshold, label in zip(THRESHOLDS, ("1%", "5%", "10%"), strict=True):
        axis.axhline(threshold, color="#AEB6C2", linewidth=0.75, linestyle=(0, (2, 3)), zorder=0)
        axis.text(1.44, threshold, label, va="bottom", ha="left", fontsize=7.4, color="#6B7280")
    axis.set_xlim(-0.55, 1.60)
    axis.set_ylim(0.0, display_limit)
    axis.set_xticks(positions, ordered_methods)
    axis.set_ylabel("Normalized mass-balance residual", fontsize=10.0, labelpad=6)
    axis.set_title("(a) Mass-Balance Residual Distribution", x=0.4383, fontsize=12.0, pad=13, weight="normal")
    axis.tick_params(axis="both", labelsize=9.0, length=3.0, width=0.8, colors="#30343B")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#59616D")
        axis.spines[spine].set_linewidth(0.8)
    figure.text(
        0.5,
        0.025,
        "Synthetic preview constrained by reported summary statistics; not empirical residual samples.\n"
        f"Display limited to {display_limit:.2f}; full pseudo-samples are retained in CSV.",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#4B5563",
    )
    figure.subplots_adjust(left=0.145, right=0.955, top=0.86, bottom=0.235)
    pdf_path = OUTPUT_DIR / "Fig6a_mass_residual_synthetic_preview.pdf"
    png_path = OUTPUT_DIR / "Fig6a_mass_residual_synthetic_preview_600dpi.png"
    figure.savefig(pdf_path, facecolor="white")
    figure.savefig(png_path, dpi=600, facecolor="white")
    plt.close(figure)
    return pdf_path, png_path, display_limit


def main() -> None:
    for source in (MANUSCRIPT_WORKBOOK, ORIGINAL_CONSTRAINT_WORKBOOK, FULL_PHYSICS_FOLD_SUMMARIES):
        if not source.is_file():
            raise FileNotFoundError(source)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observations = _threshold_observations(MANUSCRIPT_WORKBOOK, "Constraint_PINN")
    _verify_threshold_sources(observations)
    folds = _full_physics_folds()
    full_mean, full_std, full_mean_fold_sd, full_std_fold_sd, full_n = _weighted_full_physics_moments(folds)
    candidates, selected = _fit_candidates(
        observations,
        (full_mean, full_std, full_mean_fold_sd, full_std_fold_sd),
    )
    fit_path = _write_fit_parameters(
        candidates,
        (full_mean, full_std, full_mean_fold_sd, full_std_fold_sd, full_n),
    )
    samples, sample_path = _write_samples(selected)
    validation, validation_path = _validation_table(
        observations,
        samples,
        selected,
        (full_mean, full_std, full_mean_fold_sd, full_std_fold_sd, full_n),
    )
    pdf_path, png_path, display_limit = _plot(samples)

    print("SOURCE FILE PATHS USED:")
    for source in (MANUSCRIPT_WORKBOOK, ORIGINAL_CONSTRAINT_WORKBOOK, FULL_PHYSICS_FOLD_SUMMARIES):
        print(f"- {source}")
    print("\nOBSERVED MASS THRESHOLD STATISTICS:")
    print(observations[["method", "threshold", "observed_value", "observed_uncertainty", "source_type"]].to_string(index=False))
    print("\nFULL-PHYSICS OBSERVED/DERIVED MOMENTS FROM SAVED FOLD SUMMARIES:")
    print(f"valid residual count={full_n:,}; mean={full_mean:.9f}; SD(|residual|)={full_std:.9f}")
    print(f"fold SD of mean={full_mean_fold_sd:.9f}; fold SD of SD(|residual|)={full_std_fold_sd:.9f}")
    print("\nSELECTED DISTRIBUTIONS:")
    for method, fit in selected.items():
        print(f"{method}: {fit.family}; objective={fit.objective:.8g}; max standardized error={fit.max_abs_standardized_error:.8g}")
    print("\nSYNTHETIC-SAMPLE VALIDATION:")
    print(validation[["method", "metric", "observed", "synthetic", "difference_synthetic_minus_observed", "passes_tolerance"]].to_string(index=False))
    print(f"\nFIGURE SIZE: 6.65 x 4.30 in; display limit={display_limit:.2f}")
    print("OUTPUT PATHS:")
    for path in (pdf_path, png_path, sample_path, validation_path, fit_path):
        print(f"- {path}")
    print("\nTHIS FIGURE IS A SYNTHETIC PREVIEW AND MUST NOT BE PRESENTED AS EMPIRICAL RESIDUAL DATA.")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.titleweight": "normal", "pdf.fonttype": 42, "ps.fonttype": 42})
    main()
