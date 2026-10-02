#!/usr/bin/env python3
"""Regenerate the four individual seven-case sensitivity panels from a CSV.

The source data are retained exactly.  This utility is deliberately limited to
presentation changes: it plots the supplied mean and standard-deviation bands
and permits a uniform typography multiplier for paper-layout revisions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


# The taller panel gives the small sMAPE differences more vertical separation
# while retaining the established x and y display ranges.
BASE_FIGSIZE = (13.5, 12.4)
BASE_DPI = 600
FONT_SCALE = 1.7
BLUE = "#0072B2"
BLUE_DARK = "#005C91"
RED = "#C21833"
RED_DARK = "#A33F00"
DEFAULT = "#94A3B8"

FAMILY_SPECS = {
    "Depth": {
        "letter": "(a)",
        "subtitle": "Number of GNN Layers",
        "xlabel": "Number of GNN Layers",
        "slug": "depth",
    },
    "Mass": {
        "letter": "(b)",
        "subtitle": "Mass Conservation Weight",
        "xlabel": r"Mass Conservation Weight, $\lambda_{\mathrm{mass}}$",
        "slug": "mass",
    },
    "Component": {
        "letter": "(c)",
        "subtitle": "Component Conservation Weight",
        "xlabel": r"Component Conservation Weight, $\lambda_{\mathrm{component}}$ ($\times 10^{-8}$)",
        "slug": "component",
    },
    "Atom": {
        "letter": "(d)",
        "subtitle": "Atom Conservation Weight",
        "xlabel": r"Atom Conservation Weight, $\lambda_{\mathrm{atom}}$",
        "slug": "atom",
    },
}

# Keep the MAE and sMAPE curves in overlapping vertical regions while showing
# every mean plus/minus one-standard-deviation band in full.  Compared with
# the former separated display ranges, excess empty headroom is removed but
# all P01--P10 uncertainty envelopes remain within the plot frame.
DISPLAY_LIMITS = {
    "Depth": ((1100.0, 2700.0), (0.212, 0.302)),
    "Mass": ((1100.0, 1800.0), (0.212, 0.270)),
    "Component": ((1100.0, 1720.0), (0.212, 0.260)),
    "Atom": ((1100.0, 1680.0), (0.212, 0.265)),
}

TICK_LABELS = {
    "Depth": ["1", "2", "3", "4", "5", "6", "7"],
    "Mass": ["0.125", "0.25", "0.5", "1", "2", "4", "8"],
    "Component": ["1.875", "3.750", "7.500", "15.00", "30.00", "60.00", "120.0"],
    "Atom": ["0.025", "0.05", "0.1", "0.2", "0.4", "0.8", "1.6"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/final/"
            "sensitivity_7cases_center3_measured_outer4_illustrative_with_std_v2_20260916/"
            "sensitivity_7cases_supplied_table_data.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/final/"
            "sensitivity_7cases_subtitles_top_legend_b_20260921"
        ),
    )
    parser.add_argument("--font-scale", type=float, default=FONT_SCALE)
    parser.add_argument("--dpi", type=int, default=BASE_DPI)
    return parser.parse_args()


def _font(value: float, scale: float) -> float:
    return value * scale


def _style_axes(left: plt.Axes, right: plt.Axes, scale: float) -> None:
    left.grid(axis="y", color="#D8DEE6", linewidth=0.75, alpha=0.85)
    left.grid(axis="x", color="#E8ECF1", linewidth=0.55, alpha=0.60)
    left.set_axisbelow(True)
    left.spines["top"].set_visible(False)
    right.spines["top"].set_visible(False)
    left.spines["left"].set_color(BLUE)
    left.spines["bottom"].set_color("#303640")
    right.spines["right"].set_color(RED)
    for spine in (left.spines["left"], left.spines["bottom"], right.spines["right"]):
        spine.set_linewidth(1.2)
    left.tick_params(axis="x", labelsize=_font(16, scale), width=1.2, length=7, colors="#303640")
    left.tick_params(axis="y", labelsize=_font(18, scale), width=1.2, length=7, colors=BLUE_DARK)
    right.tick_params(axis="y", labelsize=_font(18, scale), width=1.2, length=7, colors=RED_DARK)


def _plot_family(
    frame: pd.DataFrame,
    family: str,
    output: Path,
    dpi: int,
    scale: float,
) -> None:
    spec = FAMILY_SPECS[family]
    data = frame.loc[frame["Sensitivity family"].eq(family)].sort_values("Case").copy()
    if len(data) != 7 or data["Case"].tolist() != list(range(1, 8)):
        raise ValueError(f"{family} must contain exactly Cases 1--7.")

    case = data["Case"]
    mae = data["MAE"]
    mae_sd = data["MAE std"]
    # The data file preserves its established percent column naming; Figure
    # display is the requested dimensionless 0.xx sMAPE convention.
    smape = data["sMAPE (%)"] / 100.0
    smape_sd = data["sMAPE std (%)"] / 100.0
    default_row = data.loc[data["Is default"].astype(bool)]
    if len(default_row) != 1:
        raise ValueError(f"{family} must have exactly one default case.")
    default_case = float(default_row["Case"].iloc[0])

    fig, left = plt.subplots(figsize=BASE_FIGSIZE, facecolor="white")
    right = left.twinx()
    _style_axes(left, right, scale)

    left.axvline(default_case, color=DEFAULT, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    left.fill_between(case, mae - mae_sd, mae + mae_sd, color=BLUE, alpha=0.13, linewidth=0, zorder=2)
    right.fill_between(case, smape - smape_sd, smape + smape_sd, color=RED, alpha=0.13, linewidth=0, zorder=2)
    left.plot(case, mae, color=BLUE, marker="o", markersize=10.5, linewidth=2.8, zorder=4)
    right.plot(case, smape, color=RED, marker="s", markersize=9.5, linewidth=2.8, zorder=4)
    left.plot(
        default_case, float(default_row["MAE"].iloc[0]), marker="o", markersize=13,
        markerfacecolor=BLUE, markeredgecolor="white", markeredgewidth=2.0,
        linestyle="none", zorder=5,
    )
    right.plot(
        default_case, float(smape.loc[default_row.index[0]]), marker="s", markersize=12,
        markerfacecolor=RED, markeredgecolor="white", markeredgewidth=2.0,
        linestyle="none", zorder=5,
    )

    left.set_xticks(case, TICK_LABELS[family])
    left.set_xlim(0.72, 7.28)
    left.set_xlabel(spec["xlabel"], fontsize=_font(20, scale), labelpad=_font(12, scale))
    left.set_ylabel("MAE", color=BLUE_DARK, fontsize=_font(20, scale), labelpad=_font(12, scale))
    right.set_ylabel("sMAPE", color=RED_DARK, fontsize=_font(20, scale), labelpad=_font(12, scale))
    left.set_ylim(*DISPLAY_LIMITS[family][0])
    right.set_ylim(*DISPLAY_LIMITS[family][1])
    # The panel subtitle is above the axes, leaving the x-axis region clean
    # and retaining the full plot height for the MAE/sMAPE traces.
    left.set_title(
        f"{spec['letter']} {spec['subtitle']}",
        fontsize=_font(24, scale), color="#20252B", pad=_font(18, scale),
    )

    handles = [
        Line2D([0], [0], color=BLUE, marker="o", linewidth=2.8, markersize=10, label="MAE"),
        Line2D([0], [0], color=RED, marker="s", linewidth=2.8, markersize=9, label="sMAPE"),
        Line2D([0], [0], color=DEFAULT, linestyle=(0, (4, 3)), linewidth=1.2, label="Default case"),
    ]
    if family == "Mass":
        # One shared legend is retained only in panel (b), which keeps the
        # four-panel manuscript layout uncluttered without losing its key.
        left.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3,
            fontsize=_font(15.0, scale), frameon=True, facecolor="white", edgecolor="#C8CDD3",
            framealpha=0.92, handlelength=1.65, columnspacing=0.44, borderpad=0.34,
        )
    # Expanded side margins prevent the enlarged axis labels from clipping;
    # moving the subtitle above the axes releases lower canvas space.
    fig.subplots_adjust(left=0.20, right=0.80, bottom=0.20, top=0.86)
    stem = output / "sensitivity" / f"sensitivity_{spec['slug']}_7cases_mae_smape"
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(stem.with_suffix(suffix), dpi=dpi, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.font_scale <= 0.0:
        raise ValueError("--font-scale must be positive.")
    source = Path(args.input_csv)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    frame = pd.read_csv(source)
    missing = {"Sensitivity family", "Case", "Setting", "MAE", "MAE std", "sMAPE (%)", "sMAPE std (%)", "Is default"}.difference(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
    if set(frame["Sensitivity family"].unique()) != set(FAMILY_SPECS):
        raise ValueError("Input CSV must contain exactly Depth, Mass, Component, and Atom.")
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "sensitivity_7cases_data.csv", index=False, encoding="utf-8-sig")
    for family in FAMILY_SPECS:
        _plot_family(frame, family, output, args.dpi, args.font_scale)
    (output / "README.md").write_text(
        "# Individual sensitivity panels: upper subtitles, 1.7x typography\n\n"
        f"- Source CSV: `{source.as_posix()}`\n"
        "- The means and standard-deviation bands are copied without numerical changes.\n"
        "- The sMAPE figure axis is rendered as a fraction (0.xx), while the source CSV retains percent columns.\n"
        "- The shared MAE/sMAPE/default-case legend appears only inside panel (b), Mass Conservation Weight.\n"
        "- The panel subtitles are above the plotting areas.\n"
        f"- Typography multiplier: {args.font_scale:.1f}x relative to the immediately preceding individual panels.\n",
        encoding="utf-8",
    )
    print(f"output={output.resolve()}")
    print(f"source={source.resolve()}")
    print(f"font_scale={args.font_scale:.2f}; panels=4; dpi={args.dpi}")


if __name__ == "__main__":
    main()
