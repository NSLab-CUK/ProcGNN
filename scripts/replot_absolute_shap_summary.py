#!/usr/bin/env python3
"""Re-render the three absolute-SHAP summary panels from saved aggregate CSVs.

This is a presentation-only utility: it copies the precomputed P01--P10
means and process-level standard deviations without recomputing SHAP values.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


BLUE = "#2166AC"
GRID = "#DCE2E9"
EDGE = "#FFFFFF"
ERROR = "#344054"
FONT_SCALE = 1.5
FIGURE_WIDTH = 13.2

PANELS = (
    (
        "a",
        "absolute_shap_node_type_mean_all_processes.csv",
        "node_type_label",
        "mean_absolute_relative_shap_by_node_type_all_processes",
        "(a) Mean Absolute SHAP by Unit Type",
        False,
        True,
    ),
    (
        "b",
        "absolute_shap_operating_variable_mean_all_processes.csv",
        "operating_variable_label",
        "mean_absolute_relative_shap_by_operating_variable_all_processes",
        "(b) Mean Absolute SHAP by Operating Variable",
        False,
        False,
    ),
    (
        "c",
        "absolute_shap_feed_type_mean_all_processes.csv",
        "feed_type_label",
        "mean_absolute_relative_shap_by_feed_type_existing_processes",
        "(c) Mean Absolute SHAP by Feed Type",
        False,
        False,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/final/"
            "shap_absolute_node_feed_operating_variables_subtitles_acb_20260917"
        ),
        help="Existing absolute-SHAP summary folder containing aggregate CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/final/"
            "shap_absolute_abc_no_legend_20260921"
        ),
    )
    parser.add_argument("--dpi", type=int, default=900)
    parser.add_argument("--font-scale", type=float, default=FONT_SCALE)
    parser.add_argument(
        "--subtitle-position",
        choices=("bottom", "top"),
        default="bottom",
        help="Place each panel subtitle below or above the plotting axes.",
    )
    return parser.parse_args()


def _font(points: float, scale: float) -> float:
    return points * scale


def _format_labels(table: pd.DataFrame, label_column: str) -> pd.DataFrame:
    result = table.copy()
    result[label_column] = result[label_column].astype(str).str.replace(
        "Hot--cold", "Hot-cold", regex=False
    )
    return result


def _visible_error(values: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Clip only the lower SD arm at zero, the valid lower bound of |SHAP|."""
    return np.vstack([np.minimum(values, std), std])


def _style(axis: plt.Axes, *, font_scale: float) -> None:
    axis.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.9)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#8A949F")
    axis.spines["bottom"].set_color("#8A949F")
    axis.tick_params(axis="x", labelsize=_font(15, font_scale), width=1.1, length=6, colors="#303640")
    axis.tick_params(axis="y", labelsize=_font(15, font_scale), length=0, colors="#20252B")


def _draw_panel(
    table: pd.DataFrame,
    *,
    label_column: str,
    stem: Path,
    subtitle: str,
    show_legend: bool,
    use_symlog: bool,
    dpi: int,
    font_scale: float,
    subtitle_position: str,
) -> None:
    table = table.sort_values("mean_absolute_relative_shap").reset_index(drop=True)
    values = table["mean_absolute_relative_shap"].to_numpy(float)
    std = table["std_across_processes"].to_numpy(float)
    labels = table[label_column].astype(str).tolist()
    count = len(table)
    # Enlarged paper type requires a taller multi-row panel without truncation.
    short_panel = count <= 3
    # The three-row feed panel is intentionally compact: its plotting-axis
    # height is 2/5 of the preceding version while its bars retain the same
    # physical thickness as the multi-row panels.
    # Retain enough overall canvas for the large x label and below-axis
    # subtitle; the compactness is applied to the plotting-axis region below.
    height = 5.0 if short_panel else max(5.7, 0.52 * count + 2.0)
    # A three-row feed panel otherwise renders each bar roughly five times
    # thicker than the 16-row unit/operating panels.  Retain its data and
    # canvas but use the matching physical bar thickness.
    bar_height = 0.35 if short_panel else 0.63
    fig, axis = plt.subplots(figsize=(FIGURE_WIDTH, height), facecolor="white")
    y = np.arange(count)
    axis.barh(
        y,
        values,
        height=bar_height,
        color=BLUE,
        edgecolor=EDGE,
        linewidth=1.05,
        xerr=_visible_error(values, std),
        error_kw={
            "ecolor": ERROR,
            "elinewidth": 1.35,
            "capsize": 3.6,
            "capthick": 1.35,
            "zorder": 5,
        },
        zorder=3,
    )
    _style(axis, font_scale=font_scale)
    axis.set_yticks(y, labels=labels)
    axis.margins(y=0.055)

    maximum = max(float((values + std).max(initial=0.0)), 1.0e-6)
    if use_symlog:
        # Retains a zero baseline yet expands sub-0.02 SHAP differences.
        axis.set_xscale("symlog", linthresh=0.02, linscale=1.45, base=10)
        x_upper = maximum * 1.35
        axis.set_xlim(0.0, x_upper)
        # Keep only well-separated major labels at the enlarged type size;
        # the symlog grid still distinguishes the 0--0.005 interval.
        ticks = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
        tick_labels = ["0", "0.005", "0.01", "0.02", "0.05", "0.10", "0.20", "0.40"]
        visible = [(tick, label) for tick, label in zip(ticks, tick_labels) if tick <= x_upper * 1.001]
        axis.set_xticks([tick for tick, _ in visible])
        axis.set_xticklabels([label for _, label in visible])
        xlabel = "Mean |SHAP Value| (symlog scale)"
    else:
        axis.set_xlim(0.0, maximum * 1.12)
        xlabel = "Mean |SHAP Value|"
    axis.set_xlabel(xlabel, fontsize=_font(18, font_scale), labelpad=_font(12, font_scale))

    if show_legend:
        axis.legend(
            handles=[Patch(facecolor=BLUE, edgecolor=EDGE, label="Mean |SHAP| ± 1 SD")],
            loc="lower right",
            bbox_to_anchor=(0.985, 0.02),
            frameon=True,
            facecolor="white",
            edgecolor="#C8CDD3",
            framealpha=0.98,
            fontsize=_font(15, font_scale),
            handlelength=2.2,
            borderpad=0.5,
            borderaxespad=0.25,
        )
        legend = axis.get_legend()
        if legend is not None:
            legend.get_texts()[0].set_text("Mean |SHAP|")
            for text in legend.get_texts():
                text.set_fontsize(_font(11, font_scale))

    left_margin = 0.35 if count > 8 else 0.25
    if subtitle_position == "top":
        axis.set_title(
            subtitle,
            fontsize=_font(22, font_scale),
            fontweight="normal",
            color="#20252B",
            pad=_font(12, font_scale),
        )
        fig.subplots_adjust(
            left=left_margin,
            right=0.97,
            top=0.80 if short_panel else 0.88,
            bottom=0.25 if short_panel else 0.14,
        )
    else:
        bottom_margin = 0.55 if short_panel else 0.25
        fig.subplots_adjust(
            left=left_margin,
            right=0.97,
            top=0.92 if short_panel else (0.92 if show_legend else 0.965),
            bottom=bottom_margin,
        )
        fig.text(
            0.5,
            0.045,
            subtitle,
            ha="center",
            va="bottom",
            fontsize=_font(22, font_scale),
            fontweight="normal",
            color="#20252B",
        )
    for suffix, kwargs in ((".png", {"dpi": dpi}), (".pdf", {}), (".svg", {})):
        fig.savefig(stem.with_suffix(suffix), facecolor="white", bbox_inches="tight", **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.font_scale <= 0.0:
        raise ValueError("--font-scale must be positive")
    source = Path(args.source_dir)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)

    copied: list[pd.DataFrame] = []
    for panel, filename, label_column, stem_name, subtitle, show_legend, use_symlog in PANELS:
        table = pd.read_csv(source / filename)
        required = {label_column, "mean_absolute_relative_shap", "std_across_processes"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"{filename} lacks columns: {sorted(missing)}")
        table = _format_labels(table, label_column)
        # The prior final figure intentionally omitted only the two log-ratio features.
        if panel == "b":
            table = table.loc[
                ~table["feature_name"].isin(["log_air_ch4_ratio", "log_water_ch4_ratio"])
            ].copy()
        table.to_csv(output / filename, index=False)
        copied.append(table.assign(panel=panel))
        _draw_panel(
            table,
            label_column=label_column,
            stem=output / stem_name,
            subtitle=subtitle,
            show_legend=show_legend,
            use_symlog=use_symlog,
            dpi=args.dpi,
            font_scale=args.font_scale,
            subtitle_position=args.subtitle_position,
        )

    pd.concat(copied, ignore_index=True).to_csv(output / "absolute_shap_panels_display_data.csv", index=False)
    (output / "README.md").write_text(
        "# Absolute-SHAP summary replot\n\n"
        f"- Source aggregate tables: `{source.as_posix()}`\n"
        "- P01--P10 means and standard deviations are copied without numerical changes.\n"
        "- Horizontal bars show the mean absolute relative SHAP value; whiskers show ±1 standard deviation across processes, clipped only at the valid lower bound of zero.\n"
        "- No panel includes a legend; the common blue bar encoding is stated by the x-axis and caption.\n"
        "- The two log-ratio operating features remain excluded from the operating-variable figure, matching the source final figure.\n"
        f"- Panel subtitle position: {args.subtitle_position}.\n"
        f"- Typography multiplier: {args.font_scale:.1f}x.\n",
        encoding="utf-8",
    )
    print(f"source={source.resolve()}")
    print(f"output={output.resolve()}; panels=3; dpi={args.dpi}")


if __name__ == "__main__":
    main()
