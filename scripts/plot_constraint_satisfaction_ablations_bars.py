#!/usr/bin/env python3
"""Render six model-on-x-axis constraint-satisfaction bar charts.

Three conservation terms are drawn separately for module ablation and for
physics-informed objective ablation.  Values are imported verbatim from the
user-table transcription in ``plot_constraint_satisfaction_ablations``.
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

import plot_constraint_satisfaction_ablations as source


DPI = 600
# The existing artwork used 1.875x base text.  Apply the requested increases
# relative to that artwork: all general typography 1.3x and panel subtitles
# (the ``(a) ... Constraint`` titles) 1.5x.
CURRENT_FONT_SCALE = 1.875
FONT_SCALE = CURRENT_FONT_SCALE * 1.30
SUBTITLE_FONT_SCALE = CURRENT_FONT_SCALE * 1.50
# A distinct, publication-friendly pastel palette for the three tolerance
# levels.  Keeping the mapping fixed makes all six panels directly comparable.
COLORS = ("#C9B6E4", "#F1B6C5", "#A9D9D1")  # lavender, blush, aqua
TERMS = source.CONSERVATION_TERMS
TOLERANCES = source.TOLERANCES


CASE_LABELS = {
    "w/o Global Pooling": "w/o Global\nPooling",
    "w/o Flow Attention": "w/o Flow\nAttention",
    "w/o Differential Encoding": "w/o Differential\nEncoding",
    "Proposed (multi-process)": "Proposed\n(multi-process)",
    "Proposed (single-process)": "Proposed\n(single-process)",
    "w/o Physics-informed loss": "w/o Physics-\ninformed loss",
    "Mass + Component": "Mass +\nComponent",
    "Mass + Atom": "Mass +\nAtom",
    "Component + Atom": "Component +\nAtom",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/final/"
            "constraint_satisfaction_ablation_grouped_bars_typography_20260921_v4_top_left_legend"
        ),
    )
    parser.add_argument("--dpi", type=int, default=DPI)
    return parser.parse_args()


def _font(points: float) -> float:
    return points * FONT_SCALE


def _subtitle_font(points: float) -> float:
    return points * SUBTITLE_FONT_SCALE


def _case_label(model: str) -> str:
    return CASE_LABELS.get(model, model)


def _panel_subtitle(panel_letter: str, study_label: str, term: str) -> str:
    """Wrap the long physics-objective subtitle without shrinking its font."""
    if study_label == "Physics-informed Objective Ablation":
        return f"({panel_letter}) Physics-informed Objective\nAblation: {term} Constraint"
    return f"({panel_letter}) {study_label}: {term} Constraint"


def _plot_term(
    rows,
    *,
    term: str,
    study_label: str,
    slug: str,
    panel_letter: str,
    show_legend: bool,
    output: Path,
    dpi: int,
) -> pd.DataFrame:
    frame = source._to_frame(rows)
    models = [model for model, _ in rows]
    positions = np.arange(len(models), dtype=float)
    width = 0.235
    # Keep all six panels on one paper-ready canvas.  The module ablation
    # panels have one fewer model, so their extra horizontal space is retained
    # as whitespace instead of changing the visual aspect ratio.
    # Keep the visual font increase noticeable at a fixed manuscript width,
    # while giving wrapped case labels enough horizontal clearance.
    fig, axis = plt.subplots(figsize=(24.0, 13.5), facecolor="white")
    panel_subtitle = _panel_subtitle(panel_letter, study_label, term)
    for index, (tolerance, color) in enumerate(zip(TOLERANCES, COLORS)):
        level = (
            frame.loc[
                frame["Conservation term"].eq(term)
                & frame["Tolerance"].eq(tolerance)
            ]
            .set_index("Model")
            .loc[models]
        )
        offset = (index - 1) * width
        axis.bar(
            positions + offset,
            level["Mean satisfaction rate"],
            width=width,
            color=color,
            edgecolor="#172033",
            linewidth=0.8,
            yerr=level["Standard deviation"],
            error_kw={
                "ecolor": "#344054",
                "elinewidth": 1.3,
                "capsize": 3.8,
                "capthick": 1.3,
            },
            zorder=3,
        )
    # Reserve only a shallow data-free upper strip.  This retains a tall
    # plotting axis while providing room for the two panel-(c) legends.
    axis.set_ylim(0.0, 1.18)
    axis.set_yticks(np.arange(0.0, 1.01, 0.2))
    axis.grid(axis="y", color="#DCE2E9", linewidth=0.85, alpha=0.95)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#8A949F")
    axis.spines["bottom"].set_color("#8A949F")
    axis.tick_params(axis="y", labelsize=_font(13), width=1.1, length=6, colors="#303640")
    axis.set_xticks(
        positions,
        labels=[_case_label(model) for model in models],
        rotation=24,
        ha="right",
    )
    axis.tick_params(axis="x", labelsize=_font(11), pad=7, length=0, colors="#20252B")
    for tick in axis.get_xticklabels():
        if tick.get_text().startswith("Proposed"):
            tick.set_fontweight("bold")
    axis.set_ylabel("Constraint satisfaction rate", fontsize=_font(16), labelpad=10)
    axis.set_xlabel("Ablation Case", fontsize=_font(16), labelpad=12)
    legend_handles = [
        Patch(facecolor=color, edgecolor="#172033", linewidth=0.8, label=tolerance)
        for tolerance, color in zip(TOLERANCES, COLORS)
    ]
    if show_legend:
        # Exactly one legend per ablation family: panel (c), Atom.  It sits in
        # the unused upper-left corner, rather than spanning the centre or
        # consuming space outside the axes.  The three columns stay over the
        # low first ablation cases, clear of all bars/error whiskers.
        axis.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(0.015, 0.975),
            ncol=3,
            fontsize=_font(12),
            frameon=True,
            facecolor="white",
            edgecolor="#CBD5E1",
            framealpha=0.97,
            borderpad=0.42,
            handletextpad=0.45,
            columnspacing=1.05,
        )
    # A figure-level subtitle keeps the legend inside the axes.  The taller
    # axes position restores the plot height while leaving a clear gap between
    # the subtitle and the upper-left legend.
    fig.suptitle(
        panel_subtitle,
        fontsize=_subtitle_font(18),
        fontweight="normal",
        y=0.985,
        linespacing=1.05,
    )
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.35, top=0.82)
    stem = output / f"constraint_satisfaction_{slug}_{term.lower()}"
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(stem.with_suffix(suffix), dpi=dpi, facecolor="white")
    plt.close(fig)
    return frame


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    module_frames = [
        _plot_term(
            source.MODULE_ABLATION,
            term=term,
            study_label="Module Ablation",
            slug="module_ablation",
            panel_letter="abc"[index],
            show_legend=index == 2,
            output=output,
            dpi=args.dpi,
        )
        for index, term in enumerate(TERMS)
    ]
    objective_frames = [
        _plot_term(
            source.PHYSICS_OBJECTIVE_ABLATION,
            term=term,
            study_label="Physics-informed Objective Ablation",
            slug="physics_objective_ablation",
            panel_letter="abc"[index],
            show_legend=index == 2,
            output=output,
            dpi=args.dpi,
        )
        for index, term in enumerate(TERMS)
    ]
    with pd.ExcelWriter(output / "constraint_satisfaction_grouped_bar_data.xlsx", engine="openpyxl") as writer:
        pd.concat(module_frames, ignore_index=True).to_excel(writer, sheet_name="Module ablation", index=False)
        pd.concat(objective_frames, ignore_index=True).to_excel(writer, sheet_name="Physics objective", index=False)
        pd.DataFrame(
            {
                "Notes": [
                    "Values are transcribed from the two user-supplied tables.",
                    "Bars show reported means; vertical whiskers show plus/minus one reported standard deviation.",
                    "Six files are provided: Mass, Component, and Atom for each ablation family.",
                ]
            }
        ).to_excel(writer, sheet_name="Read me", index=False)
    (output / "README.md").write_text(
        "# Constraint-satisfaction grouped bar charts\n\n"
        "- Six individual plots: Mass, Component, and Atom for module and physics-objective ablations.\n"
        "- X-axis: ablation case. Y-axis: constraint satisfaction rate.\n"
        "- Bar colours represent the three tolerance thresholds; vertical whiskers are plus/minus one standard deviation.\n"
        "- Panel letters restart at (a) for each ablation family; legends appear only in the Atom panel (c), inside a reserved upper plot strip.\n"
        "- General typography is 1.3x and panel subtitles are 1.5x relative to the prior figure version.\n"
        "- Values are transcribed from the user-supplied tables and retained in the Excel workbook.\n",
        encoding="utf-8",
    )
    print(f"output={output.resolve()}")
    print("figures=6; data_rows=153")


if __name__ == "__main__":
    main()
