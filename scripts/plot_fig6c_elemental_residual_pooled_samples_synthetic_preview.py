"""Create ONLY pooled Fig. 6c elemental pseudo-sample distribution preview."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_fig6b_component_residual_synthetic_preview import (
    CLASSIFICATION,
    COLORS,
    FULL_PHYSICS_FOLD_SUMMARIES,
    MANUSCRIPT_WORKBOOK,
    METHODS,
    N_DISPLAY_POINTS,
    N_PSEUDO_SAMPLES,
    ORIGINAL_CONSTRAINT_WORKBOOK,
    ROOT,
    SEED,
    THRESHOLDS,
    _fit_all,
    _full_moments,
    _parse_threshold,
    _sample_constrained,
)


BALANCE_IMPLEMENTATION = ROOT / "src" / "process_graph" / "experiment" / "node_balance_pi.py"
CONSERVATION_EVALUATOR = ROOT / "scripts" / "evaluate_physics_conservation.py"
OUTPUT_DIR = ROOT / "outputs" / "0819final" / "Paper_Tables_Final3_figures" / "final" / "Fig6c_elemental_residual_pooled_samples_synthetic_preview"


def _read_observed(workbook: Path, sheet: str) -> pd.DataFrame:
    table = pd.read_excel(workbook, sheet_name=sheet)
    required = {"Model", "Conservation term", "Tolerance", "Mean satisfaction rate", "Standard deviation"}
    if missing := required.difference(table.columns):
        raise RuntimeError(f"{workbook} / {sheet} lacks: {sorted(missing)}")
    frames = []
    for method, model in METHODS.items():
        part = table.loc[
            (table["Model"].astype(str) == model)
            & (table["Conservation term"].astype(str).str.lower() == "atom"),
            ["Tolerance", "Mean satisfaction rate", "Standard deviation"],
        ].copy()
        part["threshold"] = _parse_threshold(part["Tolerance"])
        part = part.drop_duplicates(subset=["threshold", "Mean satisfaction rate", "Standard deviation"]).sort_values("threshold")
        if not np.array_equal(part["threshold"].to_numpy(dtype=float), THRESHOLDS):
            raise RuntimeError(f"Unexpected pooled elemental thresholds for {method}: {part['threshold'].tolist()}")
        frames.append(pd.DataFrame({
            "method": method,
            "threshold": part["threshold"].to_numpy(dtype=float),
            "observed_satisfaction": part["Mean satisfaction rate"].to_numpy(dtype=float),
            "fold_standard_deviation": part["Standard deviation"].to_numpy(dtype=float),
            "source_file": str(workbook),
        }))
    return pd.concat(frames, ignore_index=True)


def _verify_workbook_match(observed: pd.DataFrame) -> None:
    original = _read_observed(ORIGINAL_CONSTRAINT_WORKBOOK, "Physics objective")
    columns = ["method", "threshold", "observed_satisfaction", "fold_standard_deviation"]
    left = observed[columns].sort_values(["method", "threshold"]).reset_index(drop=True)
    right = original[columns].sort_values(["method", "threshold"]).reset_index(drop=True)
    if not left[["method", "threshold"]].equals(right[["method", "threshold"]]) or not np.allclose(left[["observed_satisfaction", "fold_standard_deviation"]].to_numpy(), right[["observed_satisfaction", "fold_standard_deviation"]].to_numpy(), rtol=0.0, atol=1.0e-12):
        raise RuntimeError("The manuscript and original constraint workbooks disagree on pooled elemental statistics.")


def _full_elemental_folds() -> pd.DataFrame:
    table = pd.read_csv(FULL_PHYSICS_FOLD_SUMMARIES)
    required = {"phase", "term", "fold", "valid_residual_count", "normalized_abs_residual_mae", "normalized_residual_rmse", "normalized_abs_residual_max"}
    if missing := required.difference(table.columns):
        raise RuntimeError(f"{FULL_PHYSICS_FOLD_SUMMARIES} lacks: {sorted(missing)}")
    folds = table.loc[(table["phase"] == "Phase 1") & (table["term"] == "atom")].copy().sort_values("fold")
    if len(folds) != 5 or folds["fold"].nunique() != 5:
        raise RuntimeError("Expected five full-physics pooled elemental fold summaries.")
    folds["derived_abs_residual_sd"] = np.sqrt(np.maximum(0.0, folds["normalized_residual_rmse"] ** 2 - folds["normalized_abs_residual_mae"] ** 2))
    return folds


def _verify_definition() -> None:
    text = BALANCE_IMPLEMENTATION.read_text(encoding="utf-8")
    evaluator = CONSERVATION_EVALUATOR.read_text(encoding="utf-8")
    expected = (
        'PI_ATOM_ORDER: tuple[str, ...] = ("C", "H", "O", "N")',
        'REACTIVE_NODE_TYPES: tuple[str, ...] = ("smr_reactor", "wgs_reactor", "burner")',
        "atom_base = reactive",
        "atom_residual, atom_valid, _ = _reduce_balance(",
        "residual = residual / scale.detach()",
    )
    if any(item not in text for item in expected) or "TERMS = (\"mass\", \"component\", \"atom\")" not in evaluator:
        raise RuntimeError("The stored evaluator no longer matches the pooled reactive-unit elemental definition.")


def _pseudo_stats(values: np.ndarray) -> dict[str, float]:
    return {"fraction_le_0.01": float(np.mean(values <= 0.01)), "fraction_le_0.05": float(np.mean(values <= 0.05)), "fraction_le_0.10": float(np.mean(values <= 0.10)), "mean": float(values.mean()), "standard_deviation": float(values.std(ddof=0))}


def _write_csvs(observed: pd.DataFrame, full: tuple[float, float, float, float, int], selected: dict, samples: dict[str, np.ndarray], display_cap: float) -> tuple[Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples_path = OUTPUT_DIR / "Fig6c_elemental_samples.csv"
    validation_path = OUTPUT_DIR / "Fig6c_elemental_validation.csv"
    summary_path = OUTPUT_DIR / "Fig6c_elemental_summary.csv"
    full_mean, full_std, full_mean_sd, full_std_sd, full_count = full
    sample_frames, validations, summaries = [], [], []
    for method, values in samples.items():
        fit = selected[method]
        statistics = _pseudo_stats(values)
        sample_frames.append(pd.DataFrame({"element": "Pooled C/H/O/N", "method": method, "sample_id": np.arange(1, len(values) + 1), "residual": values, "synthetic": True, "distribution_family": fit.family, "random_seed": SEED}))
        points = observed.loc[observed["method"] == method].sort_values("threshold")
        for point, key in zip(points.itertuples(index=False), ("fraction_le_0.01", "fraction_le_0.05", "fraction_le_0.10"), strict=True):
            validations.append({"element": "Pooled C/H/O/N", "method": method, "metric": f"fraction <= {point.threshold:.2f}", "observed": point.observed_satisfaction, "pseudo_sample": statistics[key], "difference": statistics[key] - point.observed_satisfaction, "observed_fold_standard_deviation": point.fold_standard_deviation, "source_file": point.source_file})
        if method == "Full physics":
            for label, expected, actual, spread in (("mean", full_mean, statistics["mean"], full_mean_sd), ("standard deviation", full_std, statistics["standard_deviation"], full_std_sd)):
                validations.append({"element": "Pooled C/H/O/N", "method": method, "metric": label, "observed": expected, "pseudo_sample": actual, "difference": actual - expected, "observed_fold_standard_deviation": spread, "source_file": str(FULL_PHYSICS_FOLD_SUMMARIES)})
        summaries.append({"element": "Pooled C/H/O/N", "method": method, "synthetic": True, "n_pseudo_samples": len(values), "distribution_family": fit.family, "fit_normalized_sum_squared_error": fit.normalized_sum_squared_error, "fit_max_abs_standardized_error": fit.max_abs_standardized_error, "pseudo_fraction_le_0.01": statistics["fraction_le_0.01"], "pseudo_fraction_le_0.05": statistics["fraction_le_0.05"], "pseudo_fraction_le_0.10": statistics["fraction_le_0.10"], "pseudo_mean": statistics["mean"], "pseudo_standard_deviation": statistics["standard_deviation"], "source_valid_residual_count": full_count if method == "Full physics" else np.nan, "source_count_note": "saved full-physics count" if method == "Full physics" else "no-physics valid-residual count not retained", "display_upper_limit": display_cap, "synthetic_samples_above_display_limit": int(np.sum(values > display_cap)), "result_classification": CLASSIFICATION})
    pd.concat(sample_frames, ignore_index=True).to_csv(samples_path, index=False)
    pd.DataFrame(validations).to_csv(validation_path, index=False)
    summary = pd.DataFrame(summaries)
    summary.loc[len(summary)] = {"element": "Pooled C/H/O/N", "method": "Figure", "synthetic": True, "source_count_note": "Element-specific C/H/O/N reconstruction is unsupported by the stored summaries.", "display_upper_limit": display_cap, "result_classification": CLASSIFICATION}
    summary.to_csv(summary_path, index=False)
    return samples_path, validation_path, summary_path


def _plot(samples: dict[str, np.ndarray], display_cap: float, rng: np.random.Generator) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(6.65, 4.30), constrained_layout=False)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    methods, positions = list(METHODS), np.array([1.0, 2.0])
    populations = [samples[method] for method in methods]
    violin = axis.violinplot(populations, positions=positions, widths=0.68, showmeans=False, showmedians=False, showextrema=False, points=180)
    for body, method in zip(violin["bodies"], methods, strict=True):
        body.set_facecolor(COLORS[method]); body.set_edgecolor(COLORS[method]); body.set_alpha(0.24); body.set_linewidth(0.9)
    boxes = axis.boxplot(populations, positions=positions, widths=0.23, patch_artist=True, showfliers=False, medianprops={"color": "#20252B", "linewidth": 1.35}, boxprops={"linewidth": 1.0}, whiskerprops={"color": "#4B5563", "linewidth": 0.9}, capprops={"color": "#4B5563", "linewidth": 0.9})
    for patch, method in zip(boxes["boxes"], methods, strict=True):
        patch.set_facecolor("white"); patch.set_edgecolor(COLORS[method]); patch.set_linewidth(1.2)
    for position, method in zip(positions, methods, strict=True):
        visible = samples[method][samples[method] <= display_cap]
        show = rng.choice(visible, size=min(N_DISPLAY_POINTS, len(visible)), replace=False)
        axis.scatter(position + rng.uniform(-0.19, 0.19, size=len(show)), show, s=13, color=COLORS[method], alpha=0.52, linewidths=0, zorder=4)
    for threshold, label in zip(THRESHOLDS, ("1%", "5%", "10%"), strict=True):
        axis.axhline(threshold, color="#AEB6C2", linestyle=(0, (2, 3)), linewidth=0.75, zorder=0)
        axis.text(2.40, threshold, label, ha="left", va="bottom", fontsize=7.7, color="#687382")
    axis.set_yscale("symlog", linthresh=0.10, linscale=0.9, base=10)
    axis.set_xlim(0.48, 2.55); axis.set_ylim(0.0, display_cap)
    ticks = [tick for tick in (0.0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0) if tick <= display_cap]
    axis.set_yticks(ticks); axis.set_yticklabels([f"{tick:g}" for tick in ticks])
    axis.set_xticks(positions); axis.set_xticklabels(["No physics", "Full physics"], fontsize=9.5)
    axis.set_xlabel("Method", fontsize=10.0, labelpad=6); axis.set_ylabel("Normalized elemental-balance residual", fontsize=10.0, labelpad=6)
    # Match the Fig. 6a/b/d axes width and panel-centred title placement.
    axis.set_title("(c) Elemental-Balance Residual Distribution", x=0.4383, fontsize=12.0, pad=13, weight="normal")
    axis.tick_params(axis="both", labelsize=9.0, length=3.0, width=0.8, colors="#30343B")
    axis.spines["top"].set_visible(False); axis.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color("#59616D"); axis.spines[spine].set_linewidth(0.8)
    figure.text(0.5, 0.025, "Element-specific C/H/O/N reconstruction is unsupported by the stored summaries.\nSynthetic pooled pseudo-samples; symlog axis preserves the 0-0.10 tolerance region.\nValues above the displayed 99.5th-percentile limit remain in the CSV.", ha="center", va="bottom", fontsize=6.8, color="#4B5563")
    figure.subplots_adjust(left=0.145, right=0.955, top=0.86, bottom=0.235)
    pdf_path = OUTPUT_DIR / "Fig6c_elemental_residual_pooled_samples_synthetic_preview.pdf"
    png_path = OUTPUT_DIR / "Fig6c_elemental_residual_pooled_samples_synthetic_preview_600dpi.png"
    figure.savefig(pdf_path, facecolor="white")
    figure.savefig(png_path, dpi=600, facecolor="white")
    plt.close(figure)
    return pdf_path, png_path


def main() -> None:
    sources = (MANUSCRIPT_WORKBOOK, ORIGINAL_CONSTRAINT_WORKBOOK, FULL_PHYSICS_FOLD_SUMMARIES, BALANCE_IMPLEMENTATION, CONSERVATION_EVALUATOR)
    for source in sources:
        if not source.is_file(): raise FileNotFoundError(source)
    _verify_definition()
    observed = _read_observed(MANUSCRIPT_WORKBOOK, "Constraint_PINN")
    _verify_workbook_match(observed)
    folds = _full_elemental_folds()
    full = _full_moments(folds)
    _, selected = _fit_all(observed, full)
    rng = np.random.default_rng(SEED)
    samples = {method: _sample_constrained(fit, observed.loc[observed["method"] == method], rng, (full[0], full[1]) if method == "Full physics" else None) for method, fit in selected.items()}
    display_cap = float(max(0.15, np.quantile(np.concatenate(list(samples.values())), 0.995)))
    samples_path, validation_path, summary_path = _write_csvs(observed, full, selected, samples, display_cap)
    pdf_path, png_path = _plot(samples, display_cap, rng)
    print(CLASSIFICATION)
    print("SOURCE FILE PATHS USED:")
    for source in sources: print(f"- {source}")
    print("REACTIVE UNIT TYPES USED: smr_reactor, wgs_reactor, burner")
    print("Element-specific C/H/O/N reconstruction is unsupported by the stored summaries.")
    print(f"Full physics saved pooled elemental residual count: {full[4]:,}")
    print("No-physics valid residual count: not retained in inspected source files")
    print("\nPSEUDO-SAMPLE VALIDATION (Observed | Pseudo-sample | Difference):")
    print(pd.read_csv(validation_path)[["element", "method", "metric", "observed", "pseudo_sample", "difference"]].to_string(index=False))
    print(f"Displayed upper limit={display_cap:.6g}; all pseudo-samples remain in {samples_path}")
    print("FIGURE SIZE: 6.65 x 4.30 in")
    print("OUTPUT PATHS:")
    for path in (pdf_path, png_path, samples_path, validation_path, summary_path): print(f"- {path}")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.titleweight": "normal", "pdf.fonttype": 42, "ps.fonttype": 42})
    main()
