"""Create ONLY Fig. 6b as a synthetic component-residual sample preview.

The repository retains pooled component-conservation summaries but no
individual residual values or paired no-physics prediction exports.  This
script therefore fits positive distributions to the saved summaries, samples
from the selected fits with a fixed seed, and labels every generated value as
synthetic.  It does not load a checkpoint, run inference, or draw any CDF.
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
MANUSCRIPT_WORKBOOK = ROOT / "outputs" / "0819final" / "Paper_Tables_Final3_figures" / "final" / "experiment_results" / "experiment_results.xlsx"
ORIGINAL_CONSTRAINT_WORKBOOK = ROOT / "outputs" / "0819final" / "Paper_Tables_Final3_figures" / "final" / "constraint_satisfaction_ablation_grouped_bars_typography_20260921_v4_top_left_legend" / "constraint_satisfaction_grouped_bar_data.xlsx"
FULL_PHYSICS_FOLD_SUMMARIES = ROOT / "outputs" / "0819final" / "three_server_aggregate" / "physics_conservation_all_runs.csv"
BALANCE_IMPLEMENTATION = ROOT / "src" / "process_graph" / "experiment" / "node_balance_pi.py"
CONSERVATION_EVALUATOR = ROOT / "scripts" / "evaluate_physics_conservation.py"
OUTPUT_DIR = ROOT / "outputs" / "0819final" / "Paper_Tables_Final3_figures" / "final" / "Fig6b_component_residual_synthetic_preview"

SEED = 260923
N_PSEUDO_SAMPLES = 100_000
N_DISPLAY_POINTS = 200
THRESHOLDS = np.array([0.01, 0.05, 0.10], dtype=float)
METHODS = {
    "No physics": "w/o Physics-informed loss",
    "Full physics": "Proposed (multi-process)",
}
COLORS = {"No physics": "#C44E52", "Full physics": "#1F77B4"}
CLASSIFICATION = "SYNTHETIC PSEUDO-SAMPLES USED FOR VISUAL PREVIEW"


@dataclass(frozen=True)
class Fit:
    method: str
    family: str
    parameters: np.ndarray
    mean: float
    std: float
    normalized_sum_squared_error: float
    max_abs_standardized_error: float
    selected: bool = False


def _parse_threshold(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0].astype(float) / 100.0


def _read_observed(workbook: Path, sheet: str) -> pd.DataFrame:
    table = pd.read_excel(workbook, sheet_name=sheet)
    required = {"Model", "Conservation term", "Tolerance", "Mean satisfaction rate", "Standard deviation"}
    missing = required.difference(table.columns)
    if missing:
        raise RuntimeError(f"{workbook} / {sheet} lacks: {sorted(missing)}")
    result: list[pd.DataFrame] = []
    for display_name, model_name in METHODS.items():
        part = table.loc[
            (table["Model"].astype(str) == model_name)
            & (table["Conservation term"].astype(str).str.lower() == "component"),
            ["Tolerance", "Mean satisfaction rate", "Standard deviation"],
        ].copy()
        part["threshold"] = _parse_threshold(part["Tolerance"])
        part = part.drop_duplicates(subset=["threshold", "Mean satisfaction rate", "Standard deviation"]).sort_values("threshold")
        if not np.array_equal(part["threshold"].to_numpy(dtype=float), THRESHOLDS):
            raise RuntimeError(f"Unexpected component thresholds for {display_name}: {part['threshold'].tolist()}")
        result.append(pd.DataFrame({
            "method": display_name,
            "threshold": part["threshold"].to_numpy(dtype=float),
            "observed_satisfaction": part["Mean satisfaction rate"].to_numpy(dtype=float),
            "fold_standard_deviation": part["Standard deviation"].to_numpy(dtype=float),
            "source_file": str(workbook),
        }))
    return pd.concat(result, ignore_index=True)


def _verify_workbook_match(observed: pd.DataFrame) -> None:
    original = _read_observed(ORIGINAL_CONSTRAINT_WORKBOOK, "Physics objective")
    columns = ["method", "threshold", "observed_satisfaction", "fold_standard_deviation"]
    left = observed[columns].sort_values(["method", "threshold"]).reset_index(drop=True)
    right = original[columns].sort_values(["method", "threshold"]).reset_index(drop=True)
    if not left[["method", "threshold"]].equals(right[["method", "threshold"]]) or not np.allclose(
        left[["observed_satisfaction", "fold_standard_deviation"]].to_numpy(),
        right[["observed_satisfaction", "fold_standard_deviation"]].to_numpy(), rtol=0.0, atol=1.0e-12,
    ):
        raise RuntimeError("The manuscript and original constraint workbooks disagree on component statistics.")


def _full_physics_folds() -> pd.DataFrame:
    table = pd.read_csv(FULL_PHYSICS_FOLD_SUMMARIES)
    required = {
        "phase", "term", "fold", "valid_residual_count", "normalized_abs_residual_mae",
        "normalized_residual_rmse", "normalized_abs_residual_max",
        "satisfaction_rate_abs_le_0.01", "satisfaction_rate_abs_le_0.05", "satisfaction_rate_abs_le_0.10",
    }
    missing = required.difference(table.columns)
    if missing:
        raise RuntimeError(f"{FULL_PHYSICS_FOLD_SUMMARIES} lacks: {sorted(missing)}")
    folds = table.loc[(table["phase"] == "Phase 1") & (table["term"] == "component")].copy().sort_values("fold")
    if len(folds) != 5 or folds["fold"].nunique() != 5:
        raise RuntimeError("Expected five full-physics component fold summaries.")
    folds["derived_abs_residual_sd"] = np.sqrt(np.maximum(0.0, folds["normalized_residual_rmse"] ** 2 - folds["normalized_abs_residual_mae"] ** 2))
    return folds


def _full_moments(folds: pd.DataFrame) -> tuple[float, float, float, float, int]:
    count = folds["valid_residual_count"].to_numpy(dtype=float)
    mean = float(np.sum(count * folds["normalized_abs_residual_mae"].to_numpy(dtype=float)) / count.sum())
    square_mean = float(np.sum(count * folds["normalized_residual_rmse"].to_numpy(dtype=float) ** 2) / count.sum())
    std = float(np.sqrt(max(0.0, square_mean - mean**2)))
    return mean, std, float(folds["normalized_abs_residual_mae"].std(ddof=1)), float(folds["derived_abs_residual_sd"].std(ddof=1)), int(count.sum())


def _verify_definition() -> None:
    implementation = BALANCE_IMPLEMENTATION.read_text(encoding="utf-8")
    evaluator = CONSERVATION_EVALUATOR.read_text(encoding="utf-8")
    needed = (
        'NONREACTIVE_NODE_TYPES: tuple[str, ...] = (',
        "component_base = nonreactive",
        "component_residual, component_valid, _ = _reduce_balance(",
        "scale = true_in_abs + true_out_abs",
        "residual = residual / scale.detach()",
    )
    if any(term not in implementation for term in needed):
        raise RuntimeError("The saved component-balance implementation differs from the expected non-reactive pooled definition.")
    if "TERMS = (\"mass\", \"component\", \"atom\")" not in evaluator:
        raise RuntimeError("The saved conservation evaluator no longer exposes the component term.")


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
    return weight * lognorm.cdf(x, s=ln_shape, scale=ln_scale) + (1.0 - weight) * gamma.cdf(x, a=gamma_shape, scale=gamma_scale)


def _mixture_moments(parameters: np.ndarray) -> tuple[float, float]:
    ln_shape, ln_scale, gamma_shape, gamma_scale = np.exp(parameters[:4])
    weight = 1.0 / (1.0 + np.exp(-parameters[4]))
    ln_mean, ln_var = lognorm.stats(s=ln_shape, scale=ln_scale, moments="mv")
    gamma_mean, gamma_var = gamma.stats(a=gamma_shape, scale=gamma_scale, moments="mv")
    mean = weight * ln_mean + (1.0 - weight) * gamma_mean
    second = weight * (ln_var + ln_mean**2) + (1.0 - weight) * (gamma_var + gamma_mean**2)
    return float(mean), float(np.sqrt(max(0.0, second - mean**2)))


def _multistart(residual: Callable[[np.ndarray], np.ndarray], lower: np.ndarray, upper: np.ndarray, starts: int, rng: np.random.Generator) -> np.ndarray:
    best, best_score = None, np.inf
    for _ in range(starts):
        result = least_squares(residual, rng.uniform(lower, upper), bounds=(lower, upper), max_nfev=20_000)
        score = float(np.sum(residual(result.x) ** 2))
        if result.success and np.isfinite(score) and score < best_score:
            best, best_score = result.x, score
    if best is None:
        raise RuntimeError("All distribution fitting starts failed.")
    return best


def _fit_single(method: str, family: str, points: pd.DataFrame, moments: tuple[tuple[float, float], tuple[float, float]] | None, rng: np.random.Generator) -> Fit:
    x = points["threshold"].to_numpy(dtype=float)
    observed = points["observed_satisfaction"].to_numpy(dtype=float)
    uncertainty = points["fold_standard_deviation"].to_numpy(dtype=float)

    def residual(parameters: np.ndarray) -> np.ndarray:
        values = [(_single_cdf(family, parameters, x) - observed) / uncertainty]
        if moments is not None:
            (mean_value, mean_sd), (std_value, std_sd) = moments
            mean, std = _single_moments(family, parameters)
            values.extend((np.asarray([(mean - mean_value) / mean_sd]), np.asarray([(std - std_value) / std_sd])))
        return np.concatenate(values)

    parameters = _multistart(residual, np.array([-4.0, -12.0]), np.array([1.5, 0.0]), 32, rng)
    mean, std = _single_moments(family, parameters)
    standardized = residual(parameters)
    return Fit(method, family, parameters, mean, std, float(np.sum(standardized**2)), float(np.max(np.abs(standardized))))


def _fit_mixture(method: str, points: pd.DataFrame, moments: tuple[tuple[float, float], tuple[float, float]], rng: np.random.Generator) -> Fit:
    x = points["threshold"].to_numpy(dtype=float)
    observed = points["observed_satisfaction"].to_numpy(dtype=float)
    uncertainty = points["fold_standard_deviation"].to_numpy(dtype=float)
    (mean_value, mean_sd), (std_value, std_sd) = moments

    def residual(parameters: np.ndarray) -> np.ndarray:
        mean, std = _mixture_moments(parameters)
        return np.r_[(_mixture_cdf(parameters, x) - observed) / uncertainty, (mean - mean_value) / mean_sd, (std - std_value) / std_sd]

    parameters = _multistart(residual, np.array([-4.0, -12.0, -4.0, -12.0, -10.0]), np.array([1.5, 0.0, 1.5, 0.0, 10.0]), 72, rng)
    mean, std = _mixture_moments(parameters)
    standardized = residual(parameters)
    return Fit(method, "Log-normal + Gamma mixture", parameters, mean, std, float(np.sum(standardized**2)), float(np.max(np.abs(standardized))))


def _fit_all(observed: pd.DataFrame, full: tuple[float, float, float, float, int]) -> tuple[dict[str, list[Fit]], dict[str, Fit]]:
    full_mean, full_std, full_mean_sd, full_std_sd, _ = full
    candidates: dict[str, list[Fit]] = {}
    selected: dict[str, Fit] = {}
    rng = np.random.default_rng(SEED)
    for method in METHODS:
        points = observed.loc[observed["method"] == method].sort_values("threshold")
        moments = ((full_mean, full_mean_sd), (full_std, full_std_sd)) if method == "Full physics" else None
        trials = [_fit_single(method, family, points, moments, rng) for family in ("Log-normal", "Gamma", "Weibull")]
        best_single = min(trials, key=lambda value: value.normalized_sum_squared_error)
        if best_single.max_abs_standardized_error > 0.25:
            if moments is None:
                raise RuntimeError("No-physics mixture would require unavailable mean/SD constraints.")
            trials.append(_fit_mixture(method, points, moments, rng))
        chosen = min(trials, key=lambda value: value.normalized_sum_squared_error)
        trials = [replace(item, selected=item.family == chosen.family and np.allclose(item.parameters, chosen.parameters)) for item in trials]
        candidates[method] = trials
        selected[method] = next(item for item in trials if item.selected)
    return candidates, selected


def _cdf(fit: Fit, values: np.ndarray) -> np.ndarray:
    return _mixture_cdf(fit.parameters, values) if fit.family == "Log-normal + Gamma mixture" else _single_cdf(fit.family, fit.parameters, values)


def _ppf(fit: Fit, probabilities: np.ndarray) -> np.ndarray:
    """Inverse CDF.  The mixture branch is vectorized bisection, not a curve plot."""
    probabilities = np.asarray(probabilities, dtype=float)
    if fit.family == "Log-normal":
        return lognorm.ppf(probabilities, s=float(np.exp(fit.parameters[0])), scale=float(np.exp(fit.parameters[1])))
    if fit.family == "Gamma":
        return gamma.ppf(probabilities, a=float(np.exp(fit.parameters[0])), scale=float(np.exp(fit.parameters[1])))
    if fit.family == "Weibull":
        return weibull_min.ppf(probabilities, c=float(np.exp(fit.parameters[0])), scale=float(np.exp(fit.parameters[1])))
    lower = np.zeros_like(probabilities)
    upper = np.ones_like(probabilities)
    for _ in range(80):
        unfinished = _cdf(fit, upper) < probabilities
        if not np.any(unfinished):
            break
        upper[unfinished] *= 2.0
    else:
        raise RuntimeError("Mixture inverse-CDF upper bound did not converge.")
    for _ in range(72):
        midpoint = (lower + upper) / 2.0
        left = _cdf(fit, midpoint) < probabilities
        lower[left] = midpoint[left]
        upper[~left] = midpoint[~left]
    return (lower + upper) / 2.0


def _calibrate_full_tail(values: np.ndarray, tail_indices: np.ndarray, target_mean: float, target_std: float) -> np.ndarray:
    """Match stored first two moments without moving samples across 10% tolerance."""
    tail_base = 0.10
    low_indices = np.setdiff1d(np.arange(values.size), tail_indices, assume_unique=True)
    tail = values[tail_indices]
    target_second = target_std**2 + target_mean**2
    target_tail_mean = (values.size * target_mean - values[low_indices].sum()) / tail.size
    target_tail_second = (values.size * target_second - np.square(values[low_indices]).sum()) / tail.size
    if target_tail_mean <= tail_base or target_tail_second <= target_tail_mean**2:
        raise RuntimeError("Stored full-physics moments are incompatible with a strictly-above-10% tail calibration.")
    distances = np.maximum(tail - tail_base, np.finfo(float).tiny)

    def residual(log_parameters: np.ndarray) -> np.ndarray:
        multiplier, exponent = np.exp(log_parameters)
        calibrated = tail_base + multiplier * distances**exponent
        return np.array([
            (calibrated.mean() - target_tail_mean) / target_tail_mean,
            (np.square(calibrated).mean() - target_tail_second) / target_tail_second,
        ])

    result = least_squares(residual, np.zeros(2), bounds=(np.array([-20.0, -4.0]), np.array([20.0, 4.0])), max_nfev=20_000)
    if not result.success or np.max(np.abs(residual(result.x))) > 1.0e-7:
        raise RuntimeError("Finite full-physics pseudo-sample moment calibration failed.")
    multiplier, exponent = np.exp(result.x)
    values = values.copy()
    values[tail_indices] = tail_base + multiplier * distances**exponent
    return values


def _sample_constrained(fit: Fit, points: pd.DataFrame, rng: np.random.Generator, full_moments: tuple[float, float] | None) -> np.ndarray:
    """Stratified pseudo-sampling fixes the three observed tolerance fractions."""
    observed = points.sort_values("threshold")["observed_satisfaction"].to_numpy(dtype=float)
    counts = np.r_[np.round(observed[0] * N_PSEUDO_SAMPLES), np.round((observed[1] - observed[0]) * N_PSEUDO_SAMPLES), np.round((observed[2] - observed[1]) * N_PSEUDO_SAMPLES)].astype(int)
    counts = np.r_[counts, N_PSEUDO_SAMPLES - counts.sum()]
    if np.any(counts < 0):
        raise RuntimeError("Observed satisfaction rates are not monotonic.")
    fitted_bounds = np.r_[0.0, _cdf(fit, THRESHOLDS), 1.0]
    samples: list[np.ndarray] = []
    for count, lo, hi in zip(counts, fitted_bounds[:-1], fitted_bounds[1:], strict=True):
        if count:
            samples.append(_ppf(fit, rng.uniform(lo, hi, size=count)))
    values = np.concatenate(samples)
    # Ordered construction guarantees every tolerance bin has the observed count.
    tail_indices = np.arange(N_PSEUDO_SAMPLES - counts[-1], N_PSEUDO_SAMPLES)
    if full_moments is not None:
        values = _calibrate_full_tail(values, tail_indices, *full_moments)
    rng.shuffle(values)
    return values


def _pseudo_stats(samples: np.ndarray) -> dict[str, float]:
    return {
        "fraction_le_0.01": float(np.mean(samples <= 0.01)),
        "fraction_le_0.05": float(np.mean(samples <= 0.05)),
        "fraction_le_0.10": float(np.mean(samples <= 0.10)),
        "mean": float(np.mean(samples)),
        "standard_deviation": float(np.std(samples, ddof=0)),
    }


def _write_csvs(observed: pd.DataFrame, folds: pd.DataFrame, full: tuple[float, float, float, float, int], candidates: dict[str, list[Fit]], selected: dict[str, Fit], samples: dict[str, np.ndarray], display_cap: float) -> tuple[Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples_path = OUTPUT_DIR / "Fig6b_component_samples.csv"
    validation_path = OUTPUT_DIR / "Fig6b_component_validation.csv"
    summary_path = OUTPUT_DIR / "Fig6b_component_summary.csv"
    full_mean, full_std, full_mean_sd, full_std_sd, full_count = full
    sample_frames = []
    validation_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for method, values in samples.items():
        fit = selected[method]
        pseudo = _pseudo_stats(values)
        sample_frames.append(pd.DataFrame({"method": method, "sample_id": np.arange(1, len(values) + 1), "residual": values, "synthetic": True, "distribution_family": fit.family, "random_seed": SEED}))
        points = observed.loc[observed["method"] == method].sort_values("threshold")
        for point, key in zip(points.itertuples(index=False), ("fraction_le_0.01", "fraction_le_0.05", "fraction_le_0.10"), strict=True):
            validation_rows.append({"method": method, "metric": point.threshold, "metric_label": f"fraction <= {point.threshold:.2f}", "observed": point.observed_satisfaction, "pseudo_sample": pseudo[key], "difference": pseudo[key] - point.observed_satisfaction, "observed_fold_standard_deviation": point.fold_standard_deviation, "source_file": point.source_file})
        if method == "Full physics":
            for label, observed_value, pseudo_value, uncertainty in (("mean", full_mean, pseudo["mean"], full_mean_sd), ("standard deviation", full_std, pseudo["standard_deviation"], full_std_sd)):
                validation_rows.append({"method": method, "metric": label, "metric_label": label, "observed": observed_value, "pseudo_sample": pseudo_value, "difference": pseudo_value - observed_value, "observed_fold_standard_deviation": uncertainty, "source_file": str(FULL_PHYSICS_FOLD_SUMMARIES)})
        summary_rows.append({
            "method": method, "synthetic": True, "n_pseudo_samples": len(values), "distribution_family": fit.family,
            "fit_normalized_sum_squared_error": fit.normalized_sum_squared_error, "fit_max_abs_standardized_error": fit.max_abs_standardized_error,
            "pseudo_fraction_le_0.01": pseudo["fraction_le_0.01"], "pseudo_fraction_le_0.05": pseudo["fraction_le_0.05"], "pseudo_fraction_le_0.10": pseudo["fraction_le_0.10"],
            "pseudo_mean": pseudo["mean"], "pseudo_standard_deviation": pseudo["standard_deviation"],
            "source_valid_residual_count": full_count if method == "Full physics" else np.nan,
            "source_count_note": "saved full-physics count" if method == "Full physics" else "no-physics valid-residual count not retained",
            "display_upper_limit": display_cap, "synthetic_samples_above_display_limit": int(np.sum(values > display_cap)),
            "result_classification": CLASSIFICATION,
        })
    pd.concat(sample_frames, ignore_index=True).to_csv(samples_path, index=False)
    pd.DataFrame(validation_rows).to_csv(validation_path, index=False)
    summary = pd.DataFrame(summary_rows)
    summary.loc[len(summary)] = {"method": "Figure", "synthetic": True, "n_pseudo_samples": np.nan, "distribution_family": np.nan, "source_valid_residual_count": np.nan, "source_count_note": "Component residuals are pooled over valid species entries at non-reactive internal unit types only.", "display_upper_limit": display_cap, "result_classification": CLASSIFICATION}
    summary.to_csv(summary_path, index=False)
    return samples_path, validation_path, summary_path


def _plot(samples: dict[str, np.ndarray], display_cap: float, rng: np.random.Generator) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(6.65, 4.30), constrained_layout=False)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    methods = list(METHODS)
    positions = np.arange(1, len(methods) + 1, dtype=float)
    population = [samples[method] for method in methods]
    violin = axis.violinplot(population, positions=positions, widths=0.68, showmeans=False, showmedians=False, showextrema=False, points=180)
    for body, method in zip(violin["bodies"], methods, strict=True):
        body.set_facecolor(COLORS[method])
        body.set_edgecolor(COLORS[method])
        body.set_alpha(0.24)
        body.set_linewidth(0.9)
    boxes = axis.boxplot(population, positions=positions, widths=0.23, patch_artist=True, showfliers=False, medianprops={"color": "#20252B", "linewidth": 1.35}, boxprops={"linewidth": 1.0}, whiskerprops={"color": "#4B5563", "linewidth": 0.9}, capprops={"color": "#4B5563", "linewidth": 0.9})
    for patch, method in zip(boxes["boxes"], methods, strict=True):
        patch.set_facecolor("white")
        patch.set_edgecolor(COLORS[method])
        patch.set_linewidth(1.2)
    for position, method in zip(positions, methods, strict=True):
        visible = samples[method][samples[method] <= display_cap]
        count = min(N_DISPLAY_POINTS, len(visible))
        selected = rng.choice(visible, size=count, replace=False)
        jitter = rng.uniform(-0.19, 0.19, size=count)
        axis.scatter(position + jitter, selected, s=13, color=COLORS[method], alpha=0.52, linewidths=0, zorder=4)
    for threshold, label in zip(THRESHOLDS, ("1%", "5%", "10%"), strict=True):
        axis.axhline(threshold, color="#AEB6C2", linestyle=(0, (2, 3)), linewidth=0.75, zorder=0)
        axis.text(2.42, threshold, label, ha="left", va="bottom", fontsize=7.7, color="#687382")
    axis.set_yscale("symlog", linthresh=0.10, linscale=0.9, base=10)
    axis.set_xlim(0.48, 2.58)
    axis.set_ylim(0.0, display_cap)
    ticks = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    ticks = [tick for tick in ticks if tick <= display_cap]
    axis.set_yticks(ticks)
    axis.set_yticklabels([f"{tick:g}" for tick in ticks])
    axis.set_xticks(positions)
    axis.set_xticklabels(methods, fontsize=9.5)
    axis.set_xlabel("Method", fontsize=10.0, labelpad=6)
    axis.set_ylabel("Normalized component-balance residual", fontsize=10.0, labelpad=6)
    axis.set_title("(b) Component-Balance Residual Distribution", x=0.4383, fontsize=12.0, pad=13, weight="normal")
    axis.tick_params(axis="both", labelsize=9.0, length=3.0, width=0.8, colors="#30343B")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#59616D")
        axis.spines[spine].set_linewidth(0.8)
    figure.text(0.5, 0.025, "Synthetic pseudo-samples fitted to saved pooled summaries; no raw residuals were retained.\nSymlog axis preserves the 0-0.10 tolerance region; values above the displayed 99.5th-percentile limit\nremain in the CSV.", ha="center", va="bottom", fontsize=6.8, color="#4B5563")
    figure.subplots_adjust(left=0.145, right=0.955, top=0.86, bottom=0.235)
    pdf_path = OUTPUT_DIR / "Fig6b_component_residual_synthetic_preview.pdf"
    png_path = OUTPUT_DIR / "Fig6b_component_residual_synthetic_preview_600dpi.png"
    figure.savefig(pdf_path, facecolor="white")
    figure.savefig(png_path, dpi=600, facecolor="white")
    plt.close(figure)
    return pdf_path, png_path


def main() -> None:
    sources = (MANUSCRIPT_WORKBOOK, ORIGINAL_CONSTRAINT_WORKBOOK, FULL_PHYSICS_FOLD_SUMMARIES, BALANCE_IMPLEMENTATION, CONSERVATION_EVALUATOR)
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    _verify_definition()
    observed = _read_observed(MANUSCRIPT_WORKBOOK, "Constraint_PINN")
    _verify_workbook_match(observed)
    folds = _full_physics_folds()
    full = _full_moments(folds)
    candidates, selected = _fit_all(observed, full)
    rng = np.random.default_rng(SEED)
    samples = {
        method: _sample_constrained(
            fit,
            observed.loc[observed["method"] == method],
            rng,
            (full[0], full[1]) if method == "Full physics" else None,
        )
        for method, fit in selected.items()
    }
    display_cap = float(max(0.15, np.quantile(np.concatenate(list(samples.values())), 0.995)))
    samples_path, validation_path, summary_path = _write_csvs(observed, folds, full, candidates, selected, samples, display_cap)
    pdf_path, png_path = _plot(samples, display_cap, rng)

    print(CLASSIFICATION)
    print("SOURCE FILE PATHS USED:")
    for source in sources:
        print(f"- {source}")
    print("\nCOMPONENT SCOPE: non-reactive internal unit types only; residuals are pooled over valid component/species entries.")
    print("Non-reactive unit types: mixer, splitter, psa, flash, hx_dt, hx_hot, hx_cold, heater, cooler, compressor, pump, turbine")
    print(f"Full physics saved valid residual count: {full[4]:,}")
    print("No-physics valid residual count: not retained in inspected source files")
    print("\nSELECTED POSITIVE DISTRIBUTIONS:")
    for method, fit in selected.items():
        print(f"{method}: {fit.family}; fit objective={fit.normalized_sum_squared_error:.8g}; max standardized fitting error={fit.max_abs_standardized_error:.8g}")
    validation = pd.read_csv(validation_path)
    print("\nPSEUDO-SAMPLE VALIDATION (Metric | Observed | Pseudo-sample | Difference):")
    print(validation[["method", "metric_label", "observed", "pseudo_sample", "difference"]].to_string(index=False))
    print("\nFULL-PHYSICS FOLD VARIATION:")
    print(f"mean residual={full[0]:.9f} +/- {full[2]:.9f}; SD(residual)={full[1]:.9f} +/- {full[3]:.9f}")
    print(f"Displayed upper limit={display_cap:.6g}; all pseudo-samples, including higher residuals, are in {samples_path}")
    print("FIGURE SIZE: 6.65 x 4.30 in")
    print("OUTPUT PATHS:")
    for path in (pdf_path, png_path, samples_path, validation_path, summary_path):
        print(f"- {path}")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.titleweight": "normal", "pdf.fonttype": 42, "ps.fonttype": 42})
    main()
