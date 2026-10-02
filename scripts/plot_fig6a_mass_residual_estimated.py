"""Create only Fig. 6a: estimated mass-balance residual distribution.

This figure intentionally reconstructs continuous CDFs *only* from existing
manuscript threshold observations.  It does not load a checkpoint, invoke a
model, or run conservation evaluation.  Filled markers are the observed
satisfaction means; smooth curves and shaded bands are explicitly estimated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import gamma, lognorm, weibull_min


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_WORKBOOK = (
    ROOT
    / "outputs"
    / "0819final"
    / "Paper_Tables_Final3_figures"
    / "final"
    / "experiment_results"
    / "experiment_results.xlsx"
)
ORIGINAL_CONSTRAINT_WORKBOOK = (
    ROOT
    / "outputs"
    / "0819final"
    / "Paper_Tables_Final3_figures"
    / "final"
    / "constraint_satisfaction_ablation_grouped_bars_typography_20260921_v4_top_left_legend"
    / "constraint_satisfaction_grouped_bar_data.xlsx"
)
FULL_PHYSICS_FOLD_SUMMARIES = (
    ROOT / "outputs" / "0819final" / "three_server_aggregate" / "physics_conservation_all_runs.csv"
)
OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "0819final"
    / "Paper_Tables_Final3_figures"
    / "final"
    / "Fig6a_mass_residual_estimated"
)

THRESHOLDS = np.array([0.01, 0.05, 0.10], dtype=float)
METHODS = {
    "Without physics-informed regularization": "w/o Physics-informed loss",
    "Full physics-informed model": "Proposed (multi-process)",
}
COLORS = {
    "Without physics-informed regularization": "#C44E52",
    "Full physics-informed model": "#1F77B4",
}
DIST_NAMES = ("Log-normal", "Gamma", "Weibull")


@dataclass(frozen=True)
class Fit:
    method: str
    distribution: str
    shape: float
    scale: float
    fitted_cdf: np.ndarray
    fitted_mean: float
    weighted_chi_square: float
    cdf_rmse: float
    cdf_max_abs_error: float
    max_abs_standardized_residual: float
    selected: bool = False
    plausible_alternative: bool = False


def _parse_threshold(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace("≤", "", regex=False).str.replace("��", "", regex=False)
    percent = text.str.extract(r"([0-9]+(?:\.[0-9]+)?)")[0].astype(float)
    return percent / 100.0


def _observed_mass_points(workbook: Path, sheet_name: str = "Constraint_PINN") -> pd.DataFrame:
    frame = pd.read_excel(workbook, sheet_name=sheet_name)
    needed = {"Model", "Conservation term", "Tolerance", "Mean satisfaction rate", "Standard deviation"}
    missing = needed.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{workbook} {sheet_name} is missing columns: {sorted(missing)}")

    parts: list[pd.DataFrame] = []
    for label, manuscript_name in METHODS.items():
        part = frame.loc[
            (frame["Model"].astype(str) == manuscript_name)
            & (frame["Conservation term"].astype(str).str.lower() == "mass"),
            ["Tolerance", "Mean satisfaction rate", "Standard deviation"],
        ].copy()
        part["threshold"] = _parse_threshold(part["Tolerance"])
        part = part.drop_duplicates(subset=["threshold", "Mean satisfaction rate", "Standard deviation"])
        part = part.sort_values("threshold")
        if not np.array_equal(part["threshold"].to_numpy(dtype=float), THRESHOLDS):
            raise RuntimeError(f"Unexpected mass thresholds for {label}: {part['threshold'].tolist()}")
        parts.append(
            pd.DataFrame(
                {
                    "method": label,
                    "threshold": part["threshold"].to_numpy(dtype=float),
                    "observed_cumulative_fraction": part["Mean satisfaction rate"].to_numpy(dtype=float),
                    "fold_standard_deviation": part["Standard deviation"].to_numpy(dtype=float),
                    "source_file": str(workbook),
                    "source_sheet": sheet_name,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _verify_original_constraint_source(observed: pd.DataFrame) -> pd.DataFrame:
    original = _observed_mass_points(ORIGINAL_CONSTRAINT_WORKBOOK, sheet_name="Physics objective")
    expected = observed.drop(columns=["source_file", "source_sheet"]).sort_values(["method", "threshold"])
    actual = original.drop(columns=["source_file", "source_sheet"]).sort_values(["method", "threshold"])
    if not np.allclose(
        expected[["observed_cumulative_fraction", "fold_standard_deviation"]].to_numpy(),
        actual[["observed_cumulative_fraction", "fold_standard_deviation"]].to_numpy(),
        atol=1.0e-12,
        rtol=0.0,
    ):
        raise RuntimeError("Manuscript and original constraint-workbook observations differ.")
    return original


def _full_physics_fold_summary() -> pd.DataFrame:
    frame = pd.read_csv(FULL_PHYSICS_FOLD_SUMMARIES)
    required = {
        "phase",
        "term",
        "fold",
        "valid_residual_count",
        "normalized_abs_residual_mae",
        "satisfaction_rate_abs_le_0.01",
        "satisfaction_rate_abs_le_0.05",
        "satisfaction_rate_abs_le_0.10",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{FULL_PHYSICS_FOLD_SUMMARIES} is missing columns: {sorted(missing)}")
    result = frame.loc[(frame["phase"] == "Phase 1") & (frame["term"] == "mass")].copy()
    result = result.sort_values("fold")
    if len(result) != 5 or result["fold"].nunique() != 5:
        raise RuntimeError("Expected exactly five Full-physics multi-process mass fold summaries.")
    return result


def _cdf(distribution: str, x: np.ndarray, shape: float, scale: float) -> np.ndarray:
    if distribution == "Log-normal":
        return lognorm.cdf(x, s=shape, scale=scale)
    if distribution == "Gamma":
        return gamma.cdf(x, a=shape, scale=scale)
    if distribution == "Weibull":
        return weibull_min.cdf(x, c=shape, scale=scale)
    raise ValueError(f"Unknown distribution: {distribution}")


def _mean(distribution: str, shape: float, scale: float) -> float:
    if distribution == "Log-normal":
        return float(lognorm.mean(s=shape, scale=scale))
    if distribution == "Gamma":
        return float(gamma.mean(a=shape, scale=scale))
    if distribution == "Weibull":
        return float(weibull_min.mean(c=shape, scale=scale))
    raise ValueError(f"Unknown distribution: {distribution}")


def _fit_distribution(
    method: str,
    distribution: str,
    observed: pd.DataFrame,
    mean_constraint: tuple[float, float] | None,
) -> Fit:
    x = observed["threshold"].to_numpy(dtype=float)
    y = observed["observed_cumulative_fraction"].to_numpy(dtype=float)
    sd = observed["fold_standard_deviation"].to_numpy(dtype=float)

    def residual(log_parameters: np.ndarray) -> np.ndarray:
        shape, scale = np.exp(log_parameters)
        pieces = [(_cdf(distribution, x, shape, scale) - y) / sd]
        if mean_constraint is not None:
            mean_value, mean_sd = mean_constraint
            pieces.append(np.asarray([(_mean(distribution, shape, scale) - mean_value) / mean_sd]))
        return np.concatenate(pieces)

    result = least_squares(residual, x0=np.array([0.0, np.log(0.04)]), max_nfev=10_000)
    if not result.success:
        raise RuntimeError(f"Distribution fitting failed for {method} / {distribution}: {result.message}")
    shape, scale = np.exp(result.x)
    predicted = _cdf(distribution, x, shape, scale)
    standardized = residual(result.x)
    return Fit(
        method=method,
        distribution=distribution,
        shape=float(shape),
        scale=float(scale),
        fitted_cdf=predicted,
        fitted_mean=_mean(distribution, shape, scale),
        weighted_chi_square=float(np.sum(standardized**2)),
        cdf_rmse=float(np.sqrt(np.mean((predicted - y) ** 2))),
        cdf_max_abs_error=float(np.max(np.abs(predicted - y))),
        max_abs_standardized_residual=float(np.max(np.abs(standardized))),
    )


def _fit_all(observed: pd.DataFrame, fold_summary: pd.DataFrame) -> dict[str, list[Fit]]:
    fitted: dict[str, list[Fit]] = {}
    full_mean = float(fold_summary["normalized_abs_residual_mae"].mean())
    full_mean_sd = float(fold_summary["normalized_abs_residual_mae"].std(ddof=1))
    for method in METHODS:
        method_observed = observed.loc[observed["method"] == method].sort_values("threshold")
        mean_constraint = (full_mean, full_mean_sd) if method == "Full physics-informed model" else None
        trial_fits = [_fit_distribution(method, name, method_observed, mean_constraint) for name in DIST_NAMES]
        best_index = int(np.argmin([fit.weighted_chi_square for fit in trial_fits]))
        # A model is a plausible alternative only when every fitted constraint is
        # within one observed fold SD.  This produces an assumption-sensitive,
        # deterministic envelope without fabricating residual samples.
        updated: list[Fit] = []
        for index, fit in enumerate(trial_fits):
            updated.append(
                Fit(
                    **{
                        **fit.__dict__,
                        "selected": index == best_index,
                        "plausible_alternative": fit.max_abs_standardized_residual <= 1.0,
                    }
                )
            )
        if not any(fit.plausible_alternative for fit in updated):
            raise RuntimeError(f"No plausible fit passed the one-SD criterion for {method}.")
        fitted[method] = updated
    return fitted


def _write_outputs(
    observed: pd.DataFrame,
    fold_summary: pd.DataFrame,
    fits: dict[str, list[Fit]],
) -> tuple[Path, Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    observed_path = OUTPUT_DIR / "Fig6a_mass_residual_observed_points.csv"
    curve_path = OUTPUT_DIR / "Fig6a_mass_residual_reconstructed_curve.csv"
    parameter_path = OUTPUT_DIR / "Fig6a_distribution_fit_parameters.csv"
    validation_path = OUTPUT_DIR / "Fig6a_reconstruction_validation.csv"

    observed.to_csv(observed_path, index=False)
    x_grid = np.linspace(0.0, 0.28, 1_401)
    curve_rows: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    full_mean = float(fold_summary["normalized_abs_residual_mae"].mean())
    full_mean_sd = float(fold_summary["normalized_abs_residual_mae"].std(ddof=1))

    for method, method_fits in fits.items():
        plausible = [fit for fit in method_fits if fit.plausible_alternative]
        all_cdfs = np.vstack([_cdf(fit.distribution, x_grid, fit.shape, fit.scale) for fit in plausible])
        envelope_low = all_cdfs.min(axis=0)
        envelope_high = all_cdfs.max(axis=0)
        for fit in method_fits:
            curve_rows.append(
                pd.DataFrame(
                    {
                        "method": method,
                        "distribution": fit.distribution,
                        "selected_distribution": fit.selected,
                        "plausible_alternative": fit.plausible_alternative,
                        "normalized_mass_balance_residual": x_grid,
                        "reconstructed_cumulative_fraction": _cdf(
                            fit.distribution, x_grid, fit.shape, fit.scale
                        ),
                        "plausible_envelope_lower": envelope_low,
                        "plausible_envelope_upper": envelope_high,
                    }
                )
            )
            parameter_rows.append(
                {
                    "method": method,
                    "distribution": fit.distribution,
                    "shape": fit.shape,
                    "scale": fit.scale,
                    "distribution_mean": fit.fitted_mean,
                    "weighted_chi_square": fit.weighted_chi_square,
                    "cdf_rmse": fit.cdf_rmse,
                    "cdf_max_abs_error": fit.cdf_max_abs_error,
                    "max_abs_standardized_constraint_residual": fit.max_abs_standardized_residual,
                    "selected_distribution": fit.selected,
                    "plausible_alternative": fit.plausible_alternative,
                    "mean_constraint_used": method == "Full physics-informed model",
                    "mean_constraint_value": full_mean if method == "Full physics-informed model" else np.nan,
                    "mean_constraint_fold_sd": full_mean_sd if method == "Full physics-informed model" else np.nan,
                }
            )
            method_observed = observed.loc[observed["method"] == method].sort_values("threshold")
            for row, fitted_value in zip(method_observed.itertuples(index=False), fit.fitted_cdf, strict=True):
                validation_rows.append(
                    {
                        "validation_type": "threshold_cdf_fit",
                        "method": method,
                        "distribution": fit.distribution,
                        "selected_distribution": fit.selected,
                        "constraint": f"CDF(residual <= {row.threshold:.2f})",
                        "observed_value": row.observed_cumulative_fraction,
                        "observed_fold_sd": row.fold_standard_deviation,
                        "fitted_value": fitted_value,
                        "difference_fitted_minus_observed": fitted_value - row.observed_cumulative_fraction,
                        "standardized_difference": (
                            (fitted_value - row.observed_cumulative_fraction) / row.fold_standard_deviation
                        ),
                        "source_file": row.source_file,
                    }
                )
            if method == "Full physics-informed model":
                validation_rows.append(
                    {
                        "validation_type": "fold_summary_mean_constraint",
                        "method": method,
                        "distribution": fit.distribution,
                        "selected_distribution": fit.selected,
                        "constraint": "mean normalized absolute mass residual",
                        "observed_value": full_mean,
                        "observed_fold_sd": full_mean_sd,
                        "fitted_value": fit.fitted_mean,
                        "difference_fitted_minus_observed": fit.fitted_mean - full_mean,
                        "standardized_difference": (fit.fitted_mean - full_mean) / full_mean_sd,
                        "source_file": str(FULL_PHYSICS_FOLD_SUMMARIES),
                    }
                )

    # The manuscript workbook is the plotting source.  The fold-summary CSV is
    # retained as an independent verification record, not substituted silently.
    csv_rates = {
        0.01: float(fold_summary["satisfaction_rate_abs_le_0.01"].mean()),
        0.05: float(fold_summary["satisfaction_rate_abs_le_0.05"].mean()),
        0.10: float(fold_summary["satisfaction_rate_abs_le_0.10"].mean()),
    }
    full_observed = observed.loc[observed["method"] == "Full physics-informed model"].sort_values("threshold")
    for row in full_observed.itertuples(index=False):
        validation_rows.append(
            {
                "validation_type": "cross_source_full_physics_threshold_check",
                "method": "Full physics-informed model",
                "distribution": "not_applicable",
                "selected_distribution": False,
                "constraint": f"satisfaction @ {row.threshold:.0%}",
                "observed_value": row.observed_cumulative_fraction,
                "observed_fold_sd": row.fold_standard_deviation,
                "fitted_value": csv_rates[float(row.threshold)],
                "difference_fitted_minus_observed": csv_rates[float(row.threshold)] - row.observed_cumulative_fraction,
                "standardized_difference": (
                    (csv_rates[float(row.threshold)] - row.observed_cumulative_fraction)
                    / row.fold_standard_deviation
                ),
                "source_file": str(FULL_PHYSICS_FOLD_SUMMARIES),
            }
        )

    pd.concat(curve_rows, ignore_index=True).to_csv(curve_path, index=False)
    pd.DataFrame(parameter_rows).to_csv(parameter_path, index=False)
    pd.DataFrame(validation_rows).to_csv(validation_path, index=False)
    return observed_path, curve_path, parameter_path, validation_path


def _plot(observed: pd.DataFrame, fits: dict[str, list[Fit]]) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(6.65, 4.30), constrained_layout=False)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    x_grid = np.linspace(0.0, 0.28, 1_401)

    for method, method_fits in fits.items():
        color = COLORS[method]
        selected = next(fit for fit in method_fits if fit.selected)
        plausible = [fit for fit in method_fits if fit.plausible_alternative]
        plausible_cdfs = np.vstack([_cdf(fit.distribution, x_grid, fit.shape, fit.scale) for fit in plausible])
        axis.fill_between(
            x_grid,
            plausible_cdfs.min(axis=0),
            plausible_cdfs.max(axis=0),
            color=color,
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            x_grid,
            _cdf(selected.distribution, x_grid, selected.shape, selected.scale),
            color=color,
            linewidth=2.15,
            label=(
                "Without physics-informed regularization"
                if method.startswith("Without")
                else "Full physics-informed model"
            ),
            zorder=3,
        )
        points = observed.loc[observed["method"] == method].sort_values("threshold")
        axis.errorbar(
            points["threshold"],
            points["observed_cumulative_fraction"],
            yerr=points["fold_standard_deviation"],
            fmt="o",
            color=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.75,
            markersize=6.1,
            capsize=2.6,
            elinewidth=0.9,
            alpha=1.0,
            zorder=5,
        )

    for threshold in THRESHOLDS:
        axis.axvline(threshold, color="#AEB6C2", linewidth=0.7, linestyle=(0, (2, 3)), zorder=0)
    axis.set_xlim(0.0, 0.28)
    axis.set_ylim(0.0, 1.02)
    axis.set_xticks(np.arange(0.0, 0.281, 0.05))
    axis.set_yticks(np.arange(0.0, 1.01, 0.2))
    axis.set_xlabel("Normalized mass-balance residual", fontsize=10.0, labelpad=6)
    axis.set_ylabel("Cumulative fraction", fontsize=10.0, labelpad=6)
    axis.set_title("(a) Mass-Balance Residual Distribution", fontsize=12.0, pad=13, weight="normal")
    axis.tick_params(axis="both", labelsize=9.0, length=3.0, width=0.8, colors="#30343B")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#59616D")
        axis.spines[spine].set_linewidth(0.8)
    axis.legend(
        loc="lower right",
        fontsize=8.2,
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="#D5DAE1",
        borderpad=0.55,
        handlelength=2.2,
    )
    figure.text(
        0.5,
        0.012,
        "Solid circles: observed threshold satisfaction means (±1 fold SD).  "
        "Curves: fitted distributions; bands: plausible log-normal/gamma/Weibull alternatives.",
        ha="center",
        va="bottom",
        fontsize=7.6,
        color="#4B5563",
    )
    figure.subplots_adjust(left=0.135, right=0.975, top=0.865, bottom=0.18)
    pdf_path = OUTPUT_DIR / "Fig6a_mass_residual_estimated.pdf"
    png_path = OUTPUT_DIR / "Fig6a_mass_residual_estimated_600dpi.png"
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    figure.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return pdf_path, png_path


def main() -> None:
    for source in (MANUSCRIPT_WORKBOOK, ORIGINAL_CONSTRAINT_WORKBOOK, FULL_PHYSICS_FOLD_SUMMARIES):
        if not source.is_file():
            raise FileNotFoundError(source)
    observed = _observed_mass_points(MANUSCRIPT_WORKBOOK)
    _verify_original_constraint_source(observed)
    folds = _full_physics_fold_summary()
    fits = _fit_all(observed, folds)
    observed_path, curve_path, parameter_path, validation_path = _write_outputs(observed, folds, fits)
    pdf_path, png_path = _plot(observed, fits)

    print("SOURCE FILE PATHS USED:")
    print(f"- {MANUSCRIPT_WORKBOOK}")
    print(f"- {ORIGINAL_CONSTRAINT_WORKBOOK}")
    print(f"- {FULL_PHYSICS_FOLD_SUMMARIES}")
    print("\nOBSERVED MASS SATISFACTION POINTS:")
    print(observed.to_string(index=False))
    print("\nFULL-PHYSICS FOLD SUMMARY (existing CSV):")
    print(
        folds[
            [
                "fold",
                "valid_residual_count",
                "normalized_abs_residual_mae",
                "normalized_residual_rmse",
                "normalized_abs_residual_max",
            ]
        ].to_string(index=False)
    )
    print("\nDISTRIBUTION FITS:")
    for method, method_fits in fits.items():
        for fit in method_fits:
            print(
                f"{method} | {fit.distribution}: shape={fit.shape:.8g}, scale={fit.scale:.8g}, "
                f"weighted_chi2={fit.weighted_chi_square:.6g}, cdf_rmse={fit.cdf_rmse:.6g}, "
                f"selected={fit.selected}, plausible={fit.plausible_alternative}"
            )
    print("\nFIGURE SIZE: 6.65 x 4.30 in")
    print("OUTPUT PATHS:")
    for path in (pdf_path, png_path, observed_path, curve_path, parameter_path, validation_path):
        print(f"- {path}")


if __name__ == "__main__":
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titleweight": "normal",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    main()
