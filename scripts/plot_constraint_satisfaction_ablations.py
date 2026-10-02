#!/usr/bin/env python3
"""Render two physics-constraint satisfaction ablation figures.

The literal means and standard deviations below are transcribed from the two
user-supplied tables.  The figures use point-and-whisker displays to preserve
all three tolerance levels without obscuring their standard deviations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


TOLERANCES = ("≤ 1%", "≤ 5%", "≤ 10%")
CONSERVATION_TERMS = ("Mass", "Component", "Atom")
COLORS = ("#0072B2", "#E69F00", "#009E73")
MARKERS = ("o", "s", "^")
FONT_SCALE = 1.35
DPI = 900
FIGSIZE = (20.0, 10.4)


# Per model: ((Mass means, Mass SDs), (Component means, Component SDs),
#             (Atom means, Atom SDs)).
MODULE_ABLATION = (
    ("Undirected", (([0.1481, 0.6717, 0.8842], [0.0401, 0.0609, 0.0401]), ([0.4588, 0.7324, 0.8059], [0.0537, 0.0401, 0.0309]), ([0.2182, 0.6079, 0.7754], [0.0474, 0.0568, 0.0553]))),
    ("Backward-only", (([0.1267, 0.6291, 0.8509], [0.0437, 0.0662, 0.0459]), ([0.4183, 0.6849, 0.7691], [0.0583, 0.0438, 0.0382]), ([0.1859, 0.5562, 0.7279], [0.0499, 0.0621, 0.0609]))),
    ("Forward-only", (([0.1624, 0.7057, 0.9081], [0.0362, 0.0549, 0.0344]), ([0.4918, 0.7671, 0.8478], [0.0477, 0.0342, 0.0247]), ([0.2454, 0.6489, 0.8182], [0.0414, 0.0498, 0.0491]))),
    ("w/o Global Pooling", (([0.1778, 0.7301, 0.9217], [0.0388, 0.0521, 0.0288]), ([0.5274, 0.7907, 0.8833], [0.0481, 0.0319, 0.0451]), ([0.2697, 0.6781, 0.8448], [0.0419, 0.0471, 0.0418]))),
    ("w/o Flow Attention", (([0.1991, 0.7559, 0.9361], [0.0422, 0.0488, 0.0253]), ([0.5609, 0.8131, 0.9107], [0.0489, 0.0301, 0.0359]), ([0.2951, 0.7047, 0.8682], [0.0453, 0.0447, 0.0372]))),
    ("w/o Differential Encoding", (([0.2079, 0.7661, 0.9407], [0.0427, 0.0471, 0.0219]), ([0.5761, 0.8217, 0.9224], [0.0483, 0.0289, 0.0323]), ([0.3069, 0.7163, 0.8768], [0.0448, 0.0431, 0.0337]))),
    ("Proposed (multi-process)", (([0.2373, 0.7949, 0.9531], [0.0211, 0.0417, 0.0164]), ([0.6229, 0.8492, 0.9478], [0.0098, 0.0243, 0.0159]), ([0.3432, 0.7477, 0.9013], [0.0164, 0.0377, 0.0223]))),
    ("Proposed (single-process)", (([0.4417, 0.9492, 0.9917], [0.0817, 0.0341, 0.0089]), ([0.6953, 0.9458, 0.9892], [0.1064, 0.0359, 0.0133]), ([0.4379, 0.8981, 0.9797], [0.0857, 0.0701, 0.0279]))),
)

PHYSICS_OBJECTIVE_ABLATION = (
    ("w/o Physics-informed loss", (([0.0422, 0.3519, 0.6171], [0.0222, 0.1267, 0.1321]), ([0.1459, 0.3683, 0.4967], [0.0699, 0.1302, 0.1399]), ([0.0673, 0.2889, 0.4624], [0.0381, 0.1237, 0.1504]))),
    ("Mass only", (([0.1878, 0.7421, 0.9287], [0.0459, 0.0582, 0.0308]), ([0.1673, 0.4047, 0.5364], [0.0664, 0.1148, 0.1192]), ([0.0818, 0.3364, 0.5167], [0.0399, 0.1191, 0.1327]))),
    ("Component only", (([0.0663, 0.4667, 0.7224], [0.0264, 0.1048, 0.0852]), ([0.5329, 0.7951, 0.8879], [0.0597, 0.0394, 0.0527]), ([0.1223, 0.4407, 0.6251], [0.0442, 0.1009, 0.0962]))),
    ("Atom only", (([0.0587, 0.4383, 0.6977], [0.0247, 0.1094, 0.0929]), ([0.1791, 0.4257, 0.5574], [0.0671, 0.1108, 0.1121]), ([0.2638, 0.6712, 0.8377], [0.0518, 0.0591, 0.0537]))),
    ("Mass + Component", (([0.2171, 0.7747, 0.9451], [0.0483, 0.0489, 0.0221]), ([0.5717, 0.8201, 0.9189], [0.0527, 0.0322, 0.0357]), ([0.1121, 0.4179, 0.6034], [0.0441, 0.1068, 0.1054]))),
    ("Mass + Atom", (([0.2099, 0.7681, 0.9417], [0.0468, 0.0501, 0.0237]), ([0.2311, 0.5057, 0.6344], [0.0711, 0.0999, 0.0931]), ([0.2887, 0.6981, 0.8617], [0.0487, 0.0503, 0.0418]))),
    ("Component + Atom", (([0.0892, 0.5477, 0.7821], [0.0304, 0.0909, 0.0633]), ([0.5788, 0.8241, 0.9237], [0.0499, 0.0302, 0.0319]), ([0.3121, 0.7209, 0.8814], [0.0473, 0.0437, 0.0344]))),
    ("Proposed (multi-process)", (([0.2373, 0.7949, 0.9531], [0.0211, 0.0417, 0.0164]), ([0.6229, 0.8492, 0.9478], [0.0098, 0.0243, 0.0159]), ([0.3432, 0.7477, 0.9013], [0.0164, 0.0377, 0.0223]))),
    ("Proposed (single-process)", (([0.4417, 0.9492, 0.9917], [0.0817, 0.0341, 0.0089]), ([0.6953, 0.9458, 0.9892], [0.1064, 0.0359, 0.0133]), ([0.4379, 0.8981, 0.9797], [0.0857, 0.0701, 0.0279]))),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/final/"
            "constraint_satisfaction_ablation_pointwhisker_20260918"
        ),
    )
    parser.add_argument("--dpi", type=int, default=DPI)
    return parser.parse_args()


def _font(points: float) -> float:
    return points * FONT_SCALE


def _to_frame(rows) -> pd.DataFrame:
    result: list[dict[str, object]] = []
    for model, conservation_data in rows:
        for term, (means, stds) in zip(CONSERVATION_TERMS, conservation_data):
            for tolerance, mean, std in zip(TOLERANCES, means, stds):
                result.append(
                    {
                        "Model": model,
                        "Conservation term": term,
                        "Tolerance": tolerance,
                        "Mean satisfaction rate": mean,
                        "Standard deviation": std,
                    }
                )
    return pd.DataFrame(result)


def _style(axis: plt.Axes) -> None:
    axis.set_xlim(0.0, 1.06)
    axis.set_xticks(np.arange(0.0, 1.01, 0.2))
    axis.grid(axis="x", color="#DCE2E9", linewidth=0.85, alpha=0.95)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#8A949F")
    axis.spines["bottom"].set_color("#8A949F")
    axis.tick_params(axis="x", labelsize=_font(12), width=1.1, length=6, colors="#303640")
    axis.tick_params(axis="y", labelsize=_font(12), length=0, colors="#20252B")


def _plot(rows, output: Path, stem: str, subtitle: str, dpi: int) -> None:
    frame = _to_frame(rows)
    models = [model for model, _ in rows]
    y = np.arange(len(models))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True, facecolor="white")
    for axis, term in zip(axes, CONSERVATION_TERMS):
        subset = frame.loc[frame["Conservation term"].eq(term)]
        _style(axis)
        for index, (tolerance, color, marker) in enumerate(zip(TOLERANCES, COLORS, MARKERS)):
            level = subset.loc[subset["Tolerance"].eq(tolerance)].set_index("Model").loc[models]
            offset = (index - 1) * 0.22
            axis.errorbar(
                level["Mean satisfaction rate"],
                y + offset,
                xerr=level["Standard deviation"],
                fmt=marker,
                markersize=8.5,
                markerfacecolor=color,
                markeredgecolor="#172033",
                markeredgewidth=0.8,
                color=color,
                ecolor=color,
                elinewidth=1.65,
                capsize=4.0,
                capthick=1.65,
                linestyle="none",
                zorder=4,
            )
        axis.set_title(term, fontsize=_font(16), fontweight="normal", pad=10)
        axis.set_yticks(y, labels=models)
        for tick in axis.get_yticklabels():
            if tick.get_text().startswith("Proposed"):
                tick.set_fontweight("bold")
        axis.axhline(1.5, color="#E5E7EB", linewidth=0.9, zorder=0)
        axis.margins(y=0.09)

    for axis in axes[1:]:
        axis.tick_params(axis="y", labelleft=False)
    handles = [
        Line2D(
            [0], [0], marker=marker, color=color, markerfacecolor=color,
            markeredgecolor="#172033", markeredgewidth=0.8, linestyle="none",
            markersize=8.5, label=tolerance,
        )
        for tolerance, color, marker in zip(TOLERANCES, COLORS, MARKERS)
    ]
    # The legend occupies its own upper figure strip, inside the canvas and
    # outside the plotting areas, so no value/error bar is obscured.
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.60, 0.985),
        ncol=3,
        fontsize=_font(12),
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        framealpha=0.97,
        borderpad=0.42,
        handletextpad=0.42,
        columnspacing=1.05,
    )
    fig.text(0.60, 0.10, "Constraint satisfaction rate", ha="center", fontsize=_font(15))
    fig.text(0.60, 0.018, subtitle, ha="center", fontsize=_font(17), color="#20252B")
    fig.subplots_adjust(left=0.235, right=0.985, bottom=0.19, top=0.85, wspace=0.13)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig((output / stem).with_suffix(suffix), dpi=dpi, facecolor="white")
    plt.close(fig)
    return frame


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    module_frame = _plot(
        MODULE_ABLATION,
        output,
        "constraint_satisfaction_module_ablation",
        "(a) Module Ablation",
        args.dpi,
    )
    objective_frame = _plot(
        PHYSICS_OBJECTIVE_ABLATION,
        output,
        "constraint_satisfaction_physics_objective_ablation",
        "(b) Physics-informed Objective Ablation",
        args.dpi,
    )
    with pd.ExcelWriter(output / "constraint_satisfaction_ablation_plot_data.xlsx", engine="openpyxl") as writer:
        module_frame.to_excel(writer, sheet_name="Module ablation", index=False)
        objective_frame.to_excel(writer, sheet_name="Physics objective", index=False)
        pd.DataFrame(
            {
                "Notes": [
                    "Values were transcribed from the two user-supplied constraint-satisfaction tables.",
                    "Points show means; horizontal whiskers show plus/minus one reported standard deviation.",
                    "No values were recomputed, inferred, or rounded beyond the supplied precision.",
                ]
            }
        ).to_excel(writer, sheet_name="Read me", index=False)
    (output / "README.md").write_text(
        "# Constraint-satisfaction ablation plots\n\n"
        "- Two figures: module ablation and physics-informed objective ablation.\n"
        "- Each contains Mass, Component, and Atom conservation panels.\n"
        "- Markers show reported means and horizontal whiskers show plus/minus one reported standard deviation.\n"
        "- Plot values are transcribed from the user-supplied tables; see the accompanying Excel workbook.\n",
        encoding="utf-8",
    )
    print(f"output={output.resolve()}")
    print(f"module_rows={len(module_frame)}; objective_rows={len(objective_frame)}; dpi={args.dpi}")


if __name__ == "__main__":
    main()
