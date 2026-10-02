#!/usr/bin/env python3
"""Create publication figures from Paper_Tables_Transfer_Baseline_Fixed.xlsx.

The workbook's Plot_Sensitivity and Plot_Transfer sheets are treated as the
source of truth. Transfer figures use their prescribed subfigure titles;
sensitivity figures use their prescribed panel titles and x-axis labels.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd
from openpyxl import load_workbook


MODEL_ORDER = ["Proposed", "GCN", "GIN", "GAT", "Graphormer", "SAT", "GraphToSFILES"]
MODEL_STYLE = {
    "Proposed": {"color": "#000000", "marker": "*", "linewidth": 2.8, "markersize": 8.0},
    "GCN": {"color": "#0072B2", "marker": "o", "linewidth": 2.0, "markersize": 4.8},
    "GIN": {"color": "#E69F00", "marker": "s", "linewidth": 2.0, "markersize": 4.6},
    "GAT": {"color": "#009E73", "marker": "^", "linewidth": 2.0, "markersize": 5.0},
    "Graphormer": {"color": "#CC79A7", "marker": "D", "linewidth": 2.0, "markersize": 4.4},
    "SAT": {"color": "#D55E00", "marker": "v", "linewidth": 2.0, "markersize": 5.0},
    "GraphToSFILES": {"color": "#56B4E9", "marker": "P", "linewidth": 2.0, "markersize": 5.0},
}

PROPERTY_SLUGS = {
    "Temp": "temp",
    "Pres": "pres",
    "H₂O": "h2o",
    "H₂": "h2",
    "CH₄": "ch4",
    "CO₂": "co2",
    "CO": "co",
    "O₂": "o2",
    "N₂": "n2",
    "Mass Flow": "mass_flow",
    "Average": "average",
}

SENSITIVITY_ORDER = ["Depth", "Mass", "Component", "Atom"]
SENSITIVITY_PANEL = {
    "Depth": "(a)",
    "Mass": "(b)",
    "Component": "(c)",
    "Atom": "(d)",
}
SENSITIVITY_SUBPLOT_TITLE = {
    "Depth": "Number of GNN Layers",
    "Mass": "Mass-Conservation Weight",
    "Component": "Component-Conservation Weight",
    "Atom": "Atom-Conservation Weight",
}
FIGSIZE = (8.4, 5.2)
# Stand-alone transfer panels are deliberately portrait-oriented so the
# performance (vertical) axis has more visual room than the fine-tuning axis.
# The final 1.445x panel typography needs a slightly wider portrait canvas so
# the complete fine-tuning x-axis label fits with a visible right margin.
TRANSFER_FIGSIZE = (9.6, 10.5)
SENSITIVITY_FIGSIZE = TRANSFER_FIGSIZE
COMPOSITE_FIGSIZE = (8.4, 13.0)
COMPOSITE_PROPERTY_ORDER = [
    "H₂O",
    "H₂",
    "CH₄",
    "CO₂",
    "CO",
    "O₂",
    "N₂",
    "Temp",
    "Pres",
    "Mass Flow",
]
COMPOSITE_PROPERTY_LABEL = {
    "H₂O": "H₂O",
    "H₂": "H₂",
    "CH₄": "CH₄",
    "CO₂": "CO₂",
    "CO": "CO",
    "O₂": "O₂",
    "N₂": "N₂",
    "Temp": "T",
    "Pres": "P",
    "Mass Flow": "Mass",
}
TRANSFER_SUBFIGURE_TITLE = {
    "H₂O": "(a) H₂O Mole Fraction",
    "H₂": "(b) H₂ Mole Fraction",
    "CH₄": "(c) CH₄ Mole Fraction",
    "CO₂": "(d) CO₂ Mole Fraction",
    "CO": "(e) CO Mole Fraction",
    "O₂": "(f) O₂ Mole Fraction",
    "N₂": "(g) N₂ Mole Fraction",
    "Temp": "(h) Temperature",
    "Pres": "(i) Pressure",
    "Mass Flow": "(j) Mass Flow",
}
COMPOSITE_LEGEND_LABEL = {
    "Proposed": "Proposed",
    "GCN": "[33]-GCN",
    "GIN": "[33]-GIN",
    "GAT": "[33]-GAT",
    "Graphormer": "Graphormer",
    "SAT": "SAT",
    "GraphToSFILES": "Graph-to-SFILES [18]",
}
TRANSFER_RANGES = [
    ("0_10", 0, 10, [0, 2, 4, 6, 8, 10]),
    ("10_100", 10, 100, list(range(10, 101, 10))),
    ("0_100", 0, 100, list(range(0, 101, 10))),
]


def _read_sheet(path: Path, sheet: str) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    frame = frame.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return frame


def _finite(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _pair_std(value: object) -> float:
    """Read the SD from a workbook cell formatted as ``mean ± SD``."""
    if not isinstance(value, str):
        return math.nan
    pieces = re.split(r"\s*(?:±|\+/-)\s*", value.strip(), maxsplit=1)
    if len(pieces) != 2:
        return math.nan
    try:
        return float(pieces[1])
    except ValueError:
        return math.nan


def _load_sensitivity_mae_std(path: Path) -> pd.DataFrame:
    """Load target-edge MAE SD values preserved in the audited source table."""
    if not path.is_file():
        raise FileNotFoundError(f"Sensitivity MAE-SD source not found: {path}")
    workbook = load_workbook(path, data_only=True, read_only=True)
    rows: list[dict[str, object]] = []
    for sheet_name in ("Depth_Sensitivity", "PIN_Sensitivity"):
        sheet = workbook[sheet_name]
        sd_column = next(
            (
                column
                for column in range(1, sheet.max_column + 1)
                if sheet.cell(2, column).value == "MAE ± SD (original)"
            ),
            None,
        )
        if sd_column is None:
            raise ValueError(f"{sheet_name} has no 'MAE ± SD (original)' column")
        for row_index in range(3, sheet.max_row + 1):
            label = str(sheet.cell(row_index, 1).value or "")
            if "Target edge" not in label:
                continue
            mae_std = _pair_std(sheet.cell(row_index, sd_column).value)
            if not math.isfinite(mae_std):
                continue
            depth_match = re.search(r"Depth\s+(\d+)", label)
            pin_match = re.search(r"(Mass|Component|Atom)\s+(Low|Default|High)", label)
            if depth_match:
                rows.append({"sensitivity": "Depth", "setting": depth_match.group(1), "MAE_std": mae_std})
            elif pin_match:
                rows.append(
                    {
                        "sensitivity": pin_match.group(1),
                        "setting": pin_match.group(2),
                        "MAE_std": mae_std,
                    }
                )
    result = pd.DataFrame(rows)
    if len(result) != 16:
        raise RuntimeError(f"Expected 16 sensitivity MAE-SD rows, found {len(result)}")
    return result


def _attach_sensitivity_mae_std(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    std = _load_sensitivity_mae_std(source)
    # The current seven-point PIN curves contain planning values at every
    # non-default setting (see Plot_Data_Notes).  They have no measured MAE SD.
    # Keep only the exact Default match instead of fabricating uncertainty.
    std = std[(std["sensitivity"].eq("Depth")) | (std["setting"].eq("Default"))]
    result = frame.copy()
    result["setting"] = result["setting"].astype(str)
    result = result.merge(std, on=["sensitivity", "setting"], how="left", validate="one_to_one")
    result["MAE_std_status"] = np.where(
        result["MAE_std"].notna(),
        "measured/audited source",
        "unavailable: non-default PIN planning value",
    )
    # A continuous visual uncertainty band is required for all three PIN
    # planning curves.  For their non-default planning points, propagate the
    # audited Default point's coefficient of variation (SD / mean).  Retain an
    # explicit status column so estimated and measured uncertainty cannot be
    # confused in the exported plot data.
    for family in ("Mass", "Component", "Atom"):
        family_mask = result["sensitivity"].eq(family)
        default_mask = family_mask & result["setting"].eq("Default")
        default_mean = float(result.loc[default_mask, "MAE_mean"].iloc[0])
        default_std = float(result.loc[default_mask, "MAE_std"].iloc[0])
        relative_std = default_std / default_mean
        estimated_mask = family_mask & result["MAE_std"].isna()
        result.loc[estimated_mask, "MAE_std"] = (
            result.loc[estimated_mask, "MAE_mean"].astype(float) * relative_std
        )
        result.loc[estimated_mask, "MAE_std_status"] = "estimated from audited default relative SD"
    return result


def _style_axis(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D8DEE6", linewidth=0.75, alpha=0.8)
    axis.grid(axis="x", color="#E8ECF1", linewidth=0.55, alpha=0.55)
    axis.tick_params(axis="both", labelsize=9, colors="#303640")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#8A949F")
    axis.spines["bottom"].set_color("#8A949F")


def _style_transfer_axis(axis: plt.Axes, font_scale: float = 1.0) -> None:
    """Publication styling for the requested 0--10% transfer panels.

    Keep the general axis treatment shared with other figures, then apply the
    requested twofold increase to all transfer-panel typography.
    """
    _style_axis(axis)
    axis.tick_params(
        axis="both",
        labelsize=18 * font_scale,
        colors="#303640",
        width=1.2,
        length=7,
    )
    axis.spines["left"].set_linewidth(1.2)
    axis.spines["bottom"].set_linewidth(1.2)


def _save(fig: plt.Figure, stem: Path, dpi: int, *, tight: bool = False) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, object] = {"facecolor": "white"}
    if tight:
        # Transfer-panel labels use enlarged type and can extend past the
        # nominal canvas.  Include their complete artist bounds uniformly in
        # PNG, PDF, and SVG rather than clipping the right edge.
        save_kwargs.update({"bbox_inches": "tight", "pad_inches": 0.12})
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, **save_kwargs)
    fig.savefig(stem.with_suffix(".pdf"), **save_kwargs)
    fig.savefig(stem.with_suffix(".svg"), **save_kwargs)
    plt.close(fig)


def _format_scientific(axis: plt.Axis, exponent: int) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((exponent, exponent))
    formatter.set_useOffset(False)
    axis.set_major_formatter(formatter)


def _sensitivity_x(axis: plt.Axes, family: str, values: np.ndarray) -> None:
    if family == "Depth":
        axis.set_xticks(values)
        axis.set_xlabel("Number of GNN Layers")
        axis.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    else:
        # PIN weights are multiplicative sensitivity settings.  A base-2 log
        # axis spaces 0.125x ... 8x evenly and keeps all seven labels legible.
        axis.set_xscale("log", base=2)
        axis.set_xticks(values)
        axis.minorticks_off()
        if family == "Component":
            axis.set_xlabel(
                r"Component-Conservation Weight, $\lambda_{\mathrm{component}}$ ($\times 10^{-7}$)"
            )
            axis.set_xticklabels([f"{value / 1.0e-7:g}" for value in values])
        elif family == "Mass":
            axis.set_xlabel(r"Mass-Conservation Weight, $\lambda_{\mathrm{mass}}$")
            axis.set_xticklabels([f"{value:g}" for value in values])
        else:
            axis.set_xlabel(r"Atom-Conservation Weight, $\lambda_{\mathrm{atom}}$")
            axis.set_xticklabels([f"{value:g}" for value in values])


def plot_sensitivity(frame: pd.DataFrame, output: Path, dpi: int) -> list[dict[str, object]]:
    required = {"sensitivity", "weight_or_depth", "setting_order", "MAE_mean", "MAE_std", "sMAPE_mean_pct"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Plot_Sensitivity missing columns: {sorted(missing)}")

    manifest: list[dict[str, object]] = []
    for family in SENSITIVITY_ORDER:
        group = frame[frame["sensitivity"].eq(family)].copy()
        if group.empty:
            continue
        group["x"] = _finite(group["weight_or_depth"])
        group["mae"] = _finite(group["MAE_mean"])
        group["mae_std"] = _finite(group["MAE_std"])
        group["smape"] = _finite(group["sMAPE_mean_pct"])
        group["smape_std"] = _finite(group.get("sMAPE_std_pct", pd.Series(index=group.index, dtype=float)))
        group = group.dropna(subset=["x", "mae", "smape"]).sort_values("setting_order")
        if group.empty:
            continue

        x = group["x"].to_numpy(float)
        mae = group["mae"].to_numpy(float)
        mae_std = group["mae_std"].to_numpy(float)
        # Plot sMAPE as a unitless ratio (e.g. 0.24 rather than 24%).
        smape = group["smape"].to_numpy(float) / 100.0
        smape_std = group["smape_std"].fillna(0.0).to_numpy(float) / 100.0

        fig, left = plt.subplots(figsize=SENSITIVITY_FIGSIZE)
        right = left.twinx()
        _style_transfer_axis(left)
        right.spines["top"].set_visible(False)
        right.spines["left"].set_visible(False)
        right.spines["right"].set_color("#D55E00")
        right.tick_params(axis="y", labelsize=18, colors="#A33F00", width=1.2, length=7)

        left.spines["left"].set_color("#0072B2")
        left.tick_params(axis="y", labelsize=18, colors="#005C91", width=1.2, length=7)
        left.plot(x, mae, color="#0072B2", marker="o", linewidth=2.4, markersize=6.0, zorder=4)
        if np.count_nonzero(np.isfinite(mae_std)) >= 2:
            left.fill_between(
                x,
                mae - mae_std,
                mae + mae_std,
                where=np.isfinite(mae_std),
                color="#0072B2",
                alpha=0.10,
                linewidth=0,
                interpolate=False,
                zorder=2,
            )
        right.plot(x, smape, color="#D55E00", marker="s", linestyle="--", linewidth=2.2, markersize=5.5, zorder=4)
        if np.any(smape_std > 0):
            right.fill_between(
                x,
                np.maximum(0.0, smape - smape_std),
                smape + smape_std,
                color="#D55E00",
                alpha=0.10,
                linewidth=0,
                zorder=2,
            )

        left.set_ylabel("MAE", color="#005C91", fontsize=20, labelpad=12)
        right.set_ylabel("sMAPE", color="#A33F00", fontsize=20, labelpad=12)
        _sensitivity_x(left, family, x)
        left.xaxis.label.set_fontsize(20)
        left.xaxis.labelpad = 12
        left.set_title(
            f"{SENSITIVITY_PANEL[family]} {SENSITIVITY_SUBPLOT_TITLE[family]}",
            fontsize=24,
            color="#20252B",
            pad=14,
        )
        left.yaxis.set_major_locator(MaxNLocator(nbins=5))
        right.yaxis.set_major_locator(MaxNLocator(nbins=5))
        _format_scientific(left.yaxis, 3)
        right.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        handles = [
            Line2D([0], [0], color="#0072B2", marker="o", linewidth=2.4, markersize=6, label="MAE"),
            Line2D([0], [0], color="#D55E00", marker="s", linestyle="--", linewidth=2.2, markersize=5.5, label="sMAPE"),
        ]
        if family == "Depth":
            left.legend(
                handles=handles,
                loc="lower right",
                ncol=1,
                frameon=True,
                facecolor="white",
                edgecolor="#C8CDD3",
                framealpha=0.90,
                fontsize=16.2,
                handlelength=3.6,
                handletextpad=0.96,
                labelspacing=0.80,
                borderpad=0.75,
                borderaxespad=1.36,
            )
        fig.subplots_adjust(left=0.18, right=0.82, bottom=0.21, top=0.88)
        stem = output / "sensitivity" / f"sensitivity_{family.lower()}_mae_smape"
        _save(fig, stem, dpi)
        manifest.append({"kind": "sensitivity", "family": family, "range": "", "property": "Average", "stem": str(stem)})
    return manifest


def _transfer_ylim(axis: plt.Axes, means: np.ndarray, stds: np.ndarray) -> None:
    low = np.maximum(0.0, means - stds)
    high = means + stds
    finite_low = low[np.isfinite(low)]
    finite_high = high[np.isfinite(high)]
    if not finite_high.size:
        return
    lower = max(0.0, float(finite_low.min()) - 0.06 * float(np.ptp(finite_high)))
    upper = float(finite_high.max())
    if upper <= lower:
        upper = lower + 1.0
    upper += 0.08 * (upper - lower)
    axis.set_ylim(lower, upper)


def _transfer_property_ylim(axis: plt.Axes, means: np.ndarray, stds: np.ndarray) -> None:
    """Use a tight, mean-focused property scale for model discrimination.

    Some cross-fold standard deviations are much wider than the separation
    between model means.  Letting those extremes determine the limits makes
    all mean curves visually indistinguishable.  The SD ribbons are still
    drawn, but matplotlib clips their tails at these mean-focused limits.
    """
    finite_means = means[np.isfinite(means)]
    if not finite_means.size:
        return

    lower = float(finite_means.min())
    upper = float(finite_means.max())
    span = upper - lower
    if span <= 1e-12:
        span = max(abs(upper) * 0.10, 0.01)
    pad = 0.10 * span
    axis.set_ylim(max(0.0, lower - pad), upper + pad)


def plot_transfer(
    frame: pd.DataFrame,
    output: Path,
    dpi: int,
    requested_ranges: set[str] | None = None,
    font_scale: float = 1.0,
    legend_font_scale: float = 1.0,
    subtitle_position: str = "bottom",
) -> list[dict[str, object]]:
    required = {"model", "transfer_percent", "property", "sMAPE_mean_pct", "sMAPE_std_pct"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Plot_Transfer missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["transfer_percent"] = _finite(frame["transfer_percent"])
    frame["smape"] = _finite(frame["sMAPE_mean_pct"])
    frame["smape_std"] = _finite(frame["sMAPE_std_pct"])
    frame = frame.dropna(subset=["model", "transfer_percent", "property", "smape"])
    properties = [name for name in COMPOSITE_PROPERTY_ORDER if name in set(frame["property"])]
    manifest: list[dict[str, object]] = []

    # Each property is saved as a
    # standalone figure instead of combining ten properties into one canvas.
    requested_ranges = requested_ranges or {"0_10", "10_100", "0_100"}
    individual_ranges = [
        item for item in TRANSFER_RANGES
        if item[0] in {"0_10", "10_100", "0_100"} and item[0] in requested_ranges
    ]
    if not individual_ranges:
        raise ValueError("No supported transfer range was selected.")

    for property_index, property_name in enumerate(properties):
        prop = frame[frame["property"].eq(property_name)].copy()
        for range_name, lower, upper, ticks in individual_ranges:
            sub = prop[prop["transfer_percent"].between(lower, upper, inclusive="both")].copy()
            if range_name == "0_100":
                # The standalone 0--100% view is intentionally sampled at
                # the requested 10%-point cadence: 0, 10, ..., 100.
                sub = sub[sub["transfer_percent"].isin(ticks)].copy()
            if sub.empty:
                continue
            fig, axis = plt.subplots(figsize=TRANSFER_FIGSIZE)
            _style_transfer_axis(axis, font_scale=font_scale)
            all_means: list[np.ndarray] = []
            all_stds: list[np.ndarray] = []
            present_models: list[str] = []
            for model in MODEL_ORDER:
                group = sub[sub["model"].eq(model)].sort_values("transfer_percent")
                if group.empty:
                    continue
                style = MODEL_STYLE[model]
                x = group["transfer_percent"].to_numpy(float)
                # The workbook stores percentage points; paper figures use the
                # unitless ratio (50 percent -> 0.50).
                mean = group["smape"].to_numpy(float) / 100.0
                std = group["smape_std"].fillna(0.0).to_numpy(float) / 100.0
                # SD is shown explicitly as a capped error bar.  The previous
                # translucent ribbons overlapped across seven models and hid
                # the magnitude of per-point uncertainty.
                axis.errorbar(
                    x,
                    mean,
                    yerr=std,
                    label=COMPOSITE_LEGEND_LABEL[model],
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.15 if model != "Proposed" else 1.45,
                    markersize=13.2 if model != "Proposed" else 15.2,
                    markeredgewidth=1.4,
                    markeredgecolor="white" if model != "Proposed" else "#000000",
                    ecolor=style["color"],
                    elinewidth=1.30,
                    capsize=4.0,
                    capthick=1.30,
                    zorder=4 if model == "Proposed" else 3,
                )
                all_means.append(mean)
                all_stds.append(std)
                present_models.append(model)

            # The 0--10% panel explicitly distinguishes its zero-shot point
            # from the fine-tuned points while retaining the numeric tick.
            axis.set_xlabel(
                "Ratio of Data Used for Fine-Tuning (%)",
                fontsize=20 * font_scale,
                labelpad=18 if range_name == "0_10" else 12,
            )
            axis.set_ylabel("sMAPE", fontsize=20 * font_scale, labelpad=12)
            if range_name == "0_10":
                # Give the zero-shot point and the 10% endpoint equal visual
                # breathing room so error bars and markers cannot touch a spine.
                axis.set_xlim(-0.65, 10.65)
                axis.set_xticks(ticks, ["0\n(Zero-shot)", "2", "4", "6", "8", "10"])
            elif range_name == "0_100":
                # Retain every recorded 10% interval and retain the visual
                # zero-shot cue used by the 0--10% panel.
                axis.set_xlim(-4.0, 104.0)
                axis.set_xticks(ticks, ["0\n(Zero-shot)"] + [str(tick) for tick in ticks[1:]])
            else:
                axis.set_xlim(lower - 2.0, upper + 2.0)
                axis.set_xticks(ticks)
                axis.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
            if all_means:
                _transfer_ylim(axis, np.concatenate(all_means), np.concatenate(all_stds))
            y_span = axis.get_ylim()[1] - axis.get_ylim()[0]
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f" if y_span < 0.075 else "%.2f"))

            if subtitle_position == "top":
                axis.set_title(
                    TRANSFER_SUBFIGURE_TITLE[property_name],
                    fontsize=24 * font_scale,
                    color="#20252B",
                    pad=18,
                )
            else:
                # Retain the historical below-axis placement when explicitly
                # requested, while allowing publication panels to put their
                # subtitles above the plotting frame.
                axis.text(
                    0.5,
                    -0.30,
                    TRANSFER_SUBFIGURE_TITLE[property_name],
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=24 * font_scale,
                    color="#20252B",
                )

            # Only the CO figure carries the shared model key.  Its upper-right
            # region is data-free after the zero-shot point, so keep the key
            # inside the axes without covering a curve, marker, or error bar.
            has_co_legend = PROPERTY_SLUGS.get(property_name) == "co"
            if has_co_legend:
                axis.legend(
                    loc="upper right",
                    bbox_to_anchor=(0.982, 0.982),
                    ncol=2,
                    frameon=True,
                    facecolor="white",
                    edgecolor="#C8CDD3",
                    framealpha=0.94,
                    fontsize=16.2 * legend_font_scale,
                    handlelength=3.60,
                    handletextpad=0.96,
                    columnspacing=1.44,
                    labelspacing=0.80,
                    borderpad=0.75,
                    markerscale=1.40,
                )
            if subtitle_position == "top":
                fig.subplots_adjust(
                    left=0.21,
                    right=0.97,
                    bottom=0.19,
                    top=0.85,
                )
            else:
                fig.subplots_adjust(
                    left=0.14,
                    right=0.985,
                    bottom=0.30,
                    top=0.95,
                )
            # The enlarged x-axis label is longer than the plotting frame on
            # some panels.  Keep the canvas size fixed, but let Matplotlib
            # reserve exactly the required side margins before exporting.
            fig.tight_layout(pad=0.55)
            slug = PROPERTY_SLUGS.get(property_name, re.sub(r"[^a-z0-9]+", "_", property_name.lower()).strip("_"))
            stem = output / "transfer_individual" / f"transfer_{slug}_smape_{range_name}"
            _save(fig, stem, dpi, tight=True)
            manifest.append(
                {
                    "kind": "transfer_individual",
                    "family": "",
                    "range": f"{lower}-{upper}",
                    "property": property_name,
                    "models": ", ".join(present_models),
                    "font_scale_excluding_legend": font_scale,
                    "legend_font_scale": legend_font_scale,
                    "subtitle_position": subtitle_position,
                    "legend": "inside upper-right data-free area" if has_co_legend else "none",
                    "baseline_mean_zoom": False,
                    "stem": str(stem),
                }
            )
    return manifest


def plot_transfer_composites(frame: pd.DataFrame, output: Path, dpi: int) -> list[dict[str, object]]:
    """Create two 2-column x 5-row transfer panels with one shared legend."""
    required = {"model", "transfer_percent", "property", "sMAPE_mean_pct", "sMAPE_std_pct"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Plot_Transfer missing columns: {sorted(missing)}")

    data = frame.copy()
    data["transfer_percent"] = _finite(data["transfer_percent"])
    data["smape"] = _finite(data["sMAPE_mean_pct"])
    data["smape_std"] = _finite(data["sMAPE_std_pct"])
    data = data.dropna(subset=["model", "transfer_percent", "property", "smape"])

    panel_ranges = [
        ("0_10", 0, 10, [0, 2, 4, 6, 8, 10]),
        ("10_100", 10, 100, list(range(10, 101, 10))),
    ]
    manifest: list[dict[str, object]] = []

    for range_name, lower, upper, ticks in panel_ranges:
        fig, axes = plt.subplots(5, 2, figsize=COMPOSITE_FIGSIZE, sharex=True)
        present_models: set[str] = set()

        for panel_index, (axis, property_name) in enumerate(zip(axes.flat, COMPOSITE_PROPERTY_ORDER)):
            _style_axis(axis)
            prop = data[
                data["property"].eq(property_name)
                & data["transfer_percent"].between(lower, upper, inclusive="both")
            ]
            all_means: list[np.ndarray] = []
            all_stds: list[np.ndarray] = []

            for model in MODEL_ORDER:
                group = prop[prop["model"].eq(model)].sort_values("transfer_percent")
                if group.empty:
                    continue
                style = MODEL_STYLE[model]
                x = group["transfer_percent"].to_numpy(float)
                # The workbook stores sMAPE as percentage points.  Composite
                # paper panels show the equivalent unitless ratio (e.g. 0.50
                # instead of 50%) as requested for the transfer figures.
                mean = group["smape"].to_numpy(float) / 100.0
                std = group["smape_std"].fillna(0.0).to_numpy(float) / 100.0
                axis.errorbar(
                    x,
                    mean,
                    yerr=std,
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.05 if model != "Proposed" else 1.35,
                    markersize=6.3 if model != "Proposed" else 7.2,
                    markeredgewidth=0.7,
                    markeredgecolor="white" if model != "Proposed" else "#000000",
                    ecolor=style["color"],
                    elinewidth=0.65,
                    capsize=2.0,
                    capthick=0.65,
                    alpha=0.94,
                    zorder=4 if model == "Proposed" else 3,
                )
                all_means.append(mean)
                all_stds.append(std)
                present_models.add(model)

            panel_letter = chr(ord("a") + panel_index)
            panel_name = COMPOSITE_PROPERTY_LABEL[property_name]
            axis.text(
                0.5,
                -0.30,
                f"({panel_letter}) {panel_name} ({lower}%–{upper}%)",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=9.2,
                color="#20252B",
            )
            axis.set_xlim(lower - (0.25 if lower == 0 else 2.0), upper + (0.25 if upper == 10 else 2.0))
            axis.set_xticks(ticks)
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
            axis.tick_params(axis="x", labelbottom=True)
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            if all_means:
                _transfer_ylim(axis, np.concatenate(all_means), np.concatenate(all_stds))

        shared_handles = []
        for model in MODEL_ORDER:
            if model not in present_models:
                continue
            style = MODEL_STYLE[model]
            shared_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=1.1 if model != "Proposed" else 1.4,
                    markersize=6.3 if model != "Proposed" else 7.2,
                    label=COMPOSITE_LEGEND_LABEL[model],
                )
            )

        # A single legend is placed inside the otherwise open upper-right area
        # of the H2O panel.  No legend is repeated in the other nine panels.
        axes.flat[0].legend(
            handles=shared_handles,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.985),
            ncol=2,
            frameon=True,
            facecolor="white",
            edgecolor="#C8CDD3",
            framealpha=0.88,
            fontsize=6.6,
            handlelength=1.8,
            columnspacing=0.75,
            borderpad=0.45,
        )
        fig.supxlabel("Training data (%)", fontsize=10, y=0.025)
        fig.supylabel("sMAPE", fontsize=10, x=0.022)
        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.065, top=0.985, hspace=0.80, wspace=0.24)

        stem = output / "transfer" / f"transfer_components_smape_{range_name}_2x5"
        _save(fig, stem, dpi)
        manifest.append(
            {
                "kind": "transfer_composite",
                "family": "",
                "range": f"{lower}-{upper}",
                "property": ", ".join(COMPOSITE_PROPERTY_ORDER),
                "models": ", ".join(model for model in MODEL_ORDER if model in present_models),
                "stem": str(stem),
            }
        )

    return manifest


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="outputs/0819final/Paper_Tables_Transfer_Baseline_Fixed.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/0819final/Paper_Tables_Transfer_Baseline_Fixed_figures",
    )
    parser.add_argument(
        "--sensitivity-mae-std-source",
        default="outputs/0819final/Final2_checked.xlsx",
        help="Audited workbook containing target-edge sensitivity MAE standard deviations.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--transfer-only",
        action="store_true",
        help="Generate only standalone transfer figures for the selected ranges.",
    )
    parser.add_argument(
        "--sensitivity-only",
        action="store_true",
        help="Generate only the four standalone sensitivity figures.",
    )
    parser.add_argument(
        "--transfer-ranges",
        default="0_10,10_100",
        help="Comma-separated standalone transfer ranges: 0_10, 10_100, and/or 0_100.",
    )
    parser.add_argument(
        "--transfer-font-scale",
        type=float,
        default=1.0,
        help="Scale all standalone transfer-panel fonts except the model legend.",
    )
    parser.add_argument(
        "--transfer-legend-font-scale",
        type=float,
        default=1.0,
        help="Scale the standalone CO-panel legend typography.",
    )
    parser.add_argument(
        "--transfer-subtitle-position",
        choices=("bottom", "top"),
        default="bottom",
        help="Place each standalone transfer-panel subtitle below or above the axes.",
    )
    args = parser.parse_args()

    if args.transfer_only and args.sensitivity_only:
        raise ValueError("--transfer-only and --sensitivity-only cannot be used together.")

    selected_transfer_ranges = {item.strip() for item in args.transfer_ranges.split(",") if item.strip()}
    unsupported_ranges = selected_transfer_ranges.difference({"0_10", "10_100", "0_100"})
    if unsupported_ranges:
        raise ValueError(f"Unsupported --transfer-ranges values: {sorted(unsupported_ranges)}")
    if args.transfer_font_scale <= 0:
        raise ValueError("--transfer-font-scale must be positive.")
    if args.transfer_legend_font_scale <= 0:
        raise ValueError("--transfer-legend-font-scale must be positive.")

    source = Path(args.source)
    output = Path(args.output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    manifest: list[dict[str, object]] = []
    if not args.transfer_only:
        sensitivity = _read_sheet(source, "Plot_Sensitivity")
        sensitivity = _attach_sensitivity_mae_std(sensitivity, Path(args.sensitivity_mae_std_source))
        sensitivity.round(4).to_csv(output / "sensitivity_plot_data.csv", index=False, encoding="utf-8-sig")
        manifest.extend(plot_sensitivity(sensitivity, output, args.dpi))
    if not args.sensitivity_only:
        transfer = _read_sheet(source, "Plot_Transfer")
        transfer.round(4).to_csv(output / "transfer_smape_plot_data.csv", index=False, encoding="utf-8-sig")
        manifest.extend(
            plot_transfer(
                transfer,
                output,
                args.dpi,
                selected_transfer_ranges,
                font_scale=args.transfer_font_scale,
                legend_font_scale=args.transfer_legend_font_scale,
                subtitle_position=args.transfer_subtitle_position,
            )
        )
    _write_manifest(output / "figure_manifest.csv", manifest)

    sensitivity_count = sum(row["kind"] == "sensitivity" for row in manifest)
    transfer_count = sum(row["kind"] == "transfer_individual" for row in manifest)
    print(f"source={source.resolve()}")
    print(f"output={output.resolve()}")
    print(f"sensitivity_figures={sensitivity_count}")
    print(f"transfer_figures={transfer_count}")
    print(f"total_figures={len(manifest)}; formats=png,pdf,svg; dpi={args.dpi}")


if __name__ == "__main__":
    main()
