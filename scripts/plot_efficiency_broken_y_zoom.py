#!/usr/bin/env python3
"""Render Figure 3 panels with a transparent broken sMAPE y-axis.

The low segment preserves the Proposed point while the high segment expands
the baseline cluster.  All values, error bars, and baseline-only trend fits
are imported unchanged from ``plot_efficiency_four_panel_measured``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import plot_efficiency_four_panel_measured as base


# (a)/(b) share the training scale and (c)/(d) share the inference scale.
# The two pairs may use different ranges, which expands within-pair model
# differences while retaining full error-bar extents and their trend curves.
TRAIN_LOW_SMAPE_SEGMENT = (0.16, 0.30)
# Training panels only contain baseline values through roughly 0.80; trimming
# the unused headroom makes the baseline-to-baseline differences readable.
TRAIN_HIGH_SMAPE_SEGMENT = (0.59, 0.80)
INFERENCE_LOW_SMAPE_SEGMENT = (0.16, 0.30)
INFERENCE_HIGH_SMAPE_SEGMENT = (0.59, 1.12)


def _remove_subtitle(axis: plt.Axes, subtitle: str) -> None:
    """Remove the standard below-axis subtitle drawn by the shared helper."""
    for text in list(axis.texts):
        if text.get_text() == subtitle:
            text.remove()


def _draw_broken_panel(
    fig: plt.Figure,
    upper: plt.Axes,
    lower: plt.Axes,
    frame,
    *,
    x_column: str,
    x_sd_column: str,
    y_column: str,
    y_sd_column: str,
    xlabel: str,
    panel_key: str,
    subtitle: str,
    x_limits: tuple[float, float],
    low_y_segment: tuple[float, float],
    high_y_segment: tuple[float, float],
    add_aspen_reference: bool,
    x_errorbar_visual_scale: float,
    y_errorbar_visual_scale: float,
) -> None:
    common = dict(
        x_column=x_column,
        x_sd_column=x_sd_column,
        y_column=y_column,
        y_sd_column=y_sd_column,
        xlabel=xlabel,
        panel_key=panel_key,
        subtitle=subtitle,
        limits=x_limits,
        ylabel="sMAPE",
        x_errorbar_visual_scale=x_errorbar_visual_scale,
        y_errorbar_visual_scale=y_errorbar_visual_scale,
    )
    # The shared renderer writes identical points, error bars, and dashed
    # baseline fit to both pieces; Matplotlib clips each piece at its range.
    base._draw_panel(
        upper,
        frame,
        y_limits=high_y_segment,
        add_aspen_reference=add_aspen_reference,
        **common,
    )
    base._draw_panel(
        lower,
        frame,
        y_limits=low_y_segment,
        add_aspen_reference=False,
        **common,
    )
    if add_aspen_reference:
        lower.axvline(
            base.ASPEN_PLUS_INFERENCE_SECONDS,
            color="#C62828",
            linewidth=4.6,
            linestyle="-",
            zorder=1.8,
        )

    _remove_subtitle(upper, subtitle)
    _remove_subtitle(lower, subtitle)
    upper.set_xlabel("")
    upper.set_ylabel("")
    lower.set_ylabel("")
    upper.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    upper.spines["bottom"].set_visible(False)
    lower.spines["top"].set_visible(False)
    fig.text(
        0.075,
        0.61,
        "sMAPE",
        rotation=90,
        ha="center",
        va="center",
        fontsize=base.AXIS_LABEL_FONT_SIZE,
        color="#20252B",
    )
    fig.text(
        0.5,
        0.045,
        subtitle,
        ha="center",
        va="bottom",
        fontsize=base.SUBTITLE_FONT_SIZE,
        fontweight="normal",
        color="#1F2937",
        linespacing=1.05,
    )


def _model_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=base.MODEL_STYLE[model]["marker"],
            linestyle="none",
            markersize=13.5 if model != "Proposed" else 17.0,
            markerfacecolor=base.MODEL_STYLE[model]["color"],
            markeredgecolor="#111827",
            markeredgewidth=1.7,
            color=base.MODEL_STYLE[model]["color"],
            label=base.LEGEND_LABELS[model],
        )
        for model in base.MODEL_ORDER
    ]


def plot_individual_panels(frame, output: Path, dpi: int, only_panel: str | None) -> None:
    train_limits = base._shared_limits(
        frame["Training time (s)"].to_numpy(float),
        frame["All-stream training time (s)"].to_numpy(float),
    )
    inference_limits = base._shared_limits(
        frame["Target inference time (s/sample)"].to_numpy(float),
        frame["All-stream inference time (s/sample)"].to_numpy(float),
        references=(base.ASPEN_PLUS_INFERENCE_SECONDS,),
    )
    inference_display_limits = (inference_limits[0], inference_limits[1] * 1.55)
    specs = (
        ("a", "process_output_training", "Training time (s)", "Training time SD (s)", "sMAPE (%)", "Target sMAPE SD for display (%)", "Training Time (sec; log scale)", "target_train", "(a) Process Output Prediction\nvs. Training Time", train_limits, False),
        ("b", "streamwise_training", "All-stream training time (s)", "All-stream training time SD (s)", "All-stream performance: sMAPE (%)", "All-stream performance SD (%)", "Training Time (sec; log scale)", "all_train", "(b) Stream-wise Prediction\nvs. Training Time", train_limits, False),
        ("c", "process_output_inference", "Target inference time (s/sample)", "Target inference time SD (s/sample)", "sMAPE (%)", "Target sMAPE SD for display (%)", "Inference Time (sec/sample; log scale)", "target_infer", "(c) Process Output Prediction\nvs. Inference Time", inference_display_limits, True),
        ("d", "streamwise_inference", "All-stream inference time (s/sample)", "All-stream inference time SD (s/sample)", "All-stream performance: sMAPE (%)", "All-stream performance SD (%)", "Inference Time (sec/sample; log scale)", "all_infer", "(d) Stream-wise Prediction\nvs. Inference Time", inference_display_limits, True),
    )
    handles = _model_handles()
    for letter, slug, x_col, x_sd, y_col, y_sd, xlabel, panel_key, subtitle, x_limits, aspen in specs:
        if only_panel is not None and letter != only_panel:
            continue
        plot_frame, plot_y_col, plot_y_sd = base._fraction_smape_display(frame, y_col, y_sd)
        if letter in {"a", "b"}:
            low_y_segment = TRAIN_LOW_SMAPE_SEGMENT
            high_y_segment = TRAIN_HIGH_SMAPE_SEGMENT
        else:
            low_y_segment = INFERENCE_LOW_SMAPE_SEGMENT
            high_y_segment = INFERENCE_HIGH_SMAPE_SEGMENT
        fig = plt.figure(figsize=base.SINGLE_FIGSIZE, facecolor="white")
        grid = fig.add_gridspec(2, 1, height_ratios=(3.8, 1.0), hspace=0.08)
        upper = fig.add_subplot(grid[0])
        lower = fig.add_subplot(grid[1], sharex=upper)
        horizontal_scale, vertical_scale = base.PANEL_ERRORBAR_VISUAL_SCALES[panel_key]
        _draw_broken_panel(
            fig,
            upper,
            lower,
            plot_frame,
            x_column=x_col,
            x_sd_column=x_sd,
            y_column=plot_y_col,
            y_sd_column=plot_y_sd,
            xlabel=xlabel,
            panel_key=panel_key,
            subtitle=subtitle,
            x_limits=x_limits,
            low_y_segment=low_y_segment,
            high_y_segment=high_y_segment,
            add_aspen_reference=aspen,
            x_errorbar_visual_scale=horizontal_scale,
            y_errorbar_visual_scale=vertical_scale,
        )
        if letter == "b":
            legend = upper.legend(
                handles=handles,
                loc="upper left",
                ncol=1,
                fontsize=22.0,
                frameon=True,
                facecolor="white",
                edgecolor="#CBD5E1",
                framealpha=0.97,
                borderpad=0.58,
                labelspacing=0.34,
                handletextpad=0.50,
            )
            for text in legend.get_texts():
                if text.get_text() == "Proposed":
                    text.set_fontweight("bold")
        fig.subplots_adjust(
            left=0.23,
            right=0.985 if aspen else 0.975,
            bottom=0.33,
            top=0.965,
        )
        stem = output / f"efficiency_panel_{letter}_{slug}"
        for suffix in (".png", ".pdf", ".svg"):
            fig.savefig(stem.with_suffix(suffix), dpi=dpi, facecolor="white")
        plt.close(fig)


def main() -> None:
    args = base.parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    frame, paths = base.build_data(args)
    output.mkdir(parents=True, exist_ok=False)
    plot_individual_panels(frame, output, args.dpi, args.only_panel)
    base.write_outputs(frame, output, paths, only_panel=args.only_panel)
    readme = output / "README.md"
    with readme.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Broken sMAPE axis\n\n"
            "Panels (a)/(b) share low 0.16--0.30 and high 0.59--0.85 sMAPE "
            "segments; panels (c)/(d) share low 0.16--0.30 and high 0.59--1.12 "
            "segments. The unlabelled y-axis gap identifies the omitted interval; "
            "model values, SD error bars, and the continuous baseline-only trend "
            "are otherwise unchanged.\n"
        )
    print(f"models={len(frame)}")
    print(f"output={output.resolve()}")
    print(f"figures={1 if args.only_panel else 4} individual broken-axis panel(s); dpi={args.dpi}")


if __name__ == "__main__":
    main()
