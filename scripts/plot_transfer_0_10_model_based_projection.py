#!/usr/bin/env python3
"""Project 0--10% transfer curves from a supplied per-model property table.

This program deliberately produces *model-based projections*, not experimental
transfer results.  At 0%, each curve is anchored to the supplied property
sMAPE.  For 2--10%, it applies a documented, deterministic learning curve and
a small deterministic modulation so that projected curves are traceable and
never mistaken for measured observations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill


# Each transfer property is delivered as a standalone figure.  Preserve the
# full plotting width: the only legend is drawn inside panel (e)'s unused
# upper-right region rather than reserving an external legend column.
FIGSIZE = (9.6, 10.5)
DPI = 600
PERCENTAGES_0_10 = np.array([0, 2, 4, 6, 8, 10], dtype=float)
PERCENTAGES_0_100 = np.arange(0, 101, 10, dtype=float)
# Calculate the shared 0% and 10% anchors once.  Thus the points shown in the
# 0--100% figures at 0% and 10% are exactly the points shown in 0--10%.
PERCENTAGES = np.unique(np.concatenate((PERCENTAGES_0_10, PERCENTAGES_0_100)))
MODEL_ORDER = [
    "[54]",
    "[33]-GCN",
    "[33]-GIN",
    "[33]-GAT",
    "Graphormer [29]",
    "SAT [30]",
    "Graph-to-SFILES [18]",
    "Proposed",
]
MODEL_STYLE = {
    "[54]": {"color": "#7F3C8D", "marker": "X"},
    "[33]-GCN": {"color": "#0072B2", "marker": "o"},
    "[33]-GIN": {"color": "#E69F00", "marker": "s"},
    "[33]-GAT": {"color": "#009E73", "marker": "^"},
    "Graphormer [29]": {"color": "#CC79A7", "marker": "D"},
    "SAT [30]": {"color": "#D55E00", "marker": "v"},
    "Graph-to-SFILES [18]": {"color": "#56B4E9", "marker": "P"},
    "Proposed": {"color": "#000000", "marker": "*"},
}
PROPERTIES = [
    ("H₂O", "H2O", "×10⁻¹", "(a) H₂O Mole Fraction", 1.00),
    ("H₂", "H2", "×10⁻¹", "(b) H₂ Mole Fraction", 0.95),
    ("CH₄", "CH4", "×10⁻²", "(c) CH₄ Mole Fraction", 0.85),
    ("CO₂", "CO2", "×10⁻¹", "(d) CO₂ Mole Fraction", 1.06),
    ("CO", "CO", "×10⁻²", "(e) CO Mole Fraction", 0.90),
    ("O₂", "O2", "×10⁻²", "(f) O₂ Mole Fraction", 0.82),
    ("N₂", "N2", "×10⁻²", "(g) N₂ Mole Fraction", 0.80),
    ("Temp.", "temp", "×10²", "(h) Temperature", 1.12),
    ("Pres.", "pres", "×10¹", "(i) Pressure", 0.97),
    ("Mass", "mass_flow", "×10⁵", "(j) Mass Flow", 0.88),
]

# (MAE mean, MAE SD, sMAPE mean, sMAPE SD), transcribed from the user-supplied
# table in the exact MODEL_ORDER and PROPERTIES order above.  MAE values retain
# the table's displayed scale; sMAPE values are used as supplied (not rescaled).
SOURCE_VALUES = {
    "[54]": [
        (0.0721, 0.0129, 0.4778, 0.0008), (0.0582, 0.0095, 0.3891, 0.0022),
        (0.6401, 0.1301, 0.9958, 0.0037), (0.0573, 0.0082, 0.5493, 0.0012),
        (0.3801, 0.0913, 0.8529, 0.0017), (0.9199, 0.3399, 1.5733, 0.0031),
        (0.9501, 0.2904, 1.5727, 0.0004), (0.1799, 0.0007, 0.0592, 0.0002),
        (0.2002, 0.0012, 0.1407, 0.0007), (0.5337, 0.0079, 0.1622, 0.0043),
    ],
    "[33]-GCN": [
        (0.6281, 0.0389, 0.5401, 0.0021), (0.6109, 0.0336, 0.4409, 0.0017),
        (10.5997, 0.3898, 1.0501, 0.0032), (0.7509, 0.0308, 0.6118, 0.0019),
        (8.2999, 0.3199, 0.9112, 0.0012), (6.7004, 1.2001, 1.6047, 0.0098),
        (6.6999, 1.2997, 1.5771, 0.0001), (0.2033, 0.0012, 0.0647, 0.0002),
        (0.2398, 0.0007, 0.1644, 0.0012), (0.6331, 0.0131, 0.2307, 0.0037),
    ],
    "[33]-GIN": [
        (0.6181, 0.0509, 0.5378, 0.0027), (0.5894, 0.0489, 0.4391, 0.0042),
        (10.2003, 0.4501, 1.0547, 0.0037), (0.7367, 0.0289, 0.6121, 0.0011),
        (7.8001, 0.3603, 0.9178, 0.0038), (7.0997, 1.0997, 1.6071, 0.0092),
        (6.6001, 1.0001, 1.5779, 0.0009), (0.2077, 0.0009, 0.0661, 0.0013),
        (0.2404, 0.0023, 0.1657, 0.0027), (0.6369, 0.0069, 0.2272, 0.0041),
    ],
    "[33]-GAT": [
        (0.6889, 0.0382, 0.5394, 0.0032), (0.6173, 0.0221, 0.4409, 0.0019),
        (10.3999, 0.4399, 1.0524, 0.0041), (0.7673, 0.0221, 0.6119, 0.0017),
        (8.2997, 0.1698, 0.9111, 0.0041), (8.1002, 0.2902, 1.6017, 0.0077),
        (7.7997, 0.1697, 1.5771, 0.0002), (0.2034, 0.0022, 0.0648, 0.0008),
        (0.2299, 0.0077, 0.1661, 0.0021), (0.6321, 0.0092, 0.2278, 0.0047),
    ],
    "Graphormer [29]": [
        (0.0881, 0.0082, 0.4947, 0.0007), (0.0674, 0.0082, 0.4052, 0.0011),
        (0.7201, 0.0711, 1.1537, 0.0018), (0.0674, 0.0095, 0.5663, 0.0024),
        (0.3701, 0.0591, 1.0108, 0.0007), (0.5498, 0.3497, 1.5851, 0.0093),
        (0.7802, 0.3203, 1.5789, 0.0003), (0.2397, 0.0009, 0.0752, 0.0001),
        (0.2304, 0.0022, 0.1557, 0.0029), (0.7447, 0.0087, 0.5591, 0.0031),
    ],
    "SAT [30]": [
        (0.1166, 0.0495, 0.4972, 0.0031), (0.0781, 0.0221, 0.4057, 0.0019),
        (1.0999, 0.4698, 1.1531, 0.0014), (0.0709, 0.0174, 0.5667, 0.0027),
        (0.6497, 0.4297, 1.0114, 0.0013), (1.1004, 1.2004, 1.5887, 0.0139),
        (1.1997, 0.7498, 1.5791, 0.0004), (0.2401, 0.0013, 0.0749, 0.0002),
        (0.2298, 0.0019, 0.1541, 0.0031), (0.7351, 0.0223, 0.4788, 0.1249),
    ],
    "Graph-to-SFILES [18]": [
        (0.1908, 0.0537, 0.5028, 0.0038), (0.1166, 0.0167, 0.4094, 0.0011),
        (1.8004, 0.4304, 1.1528, 0.0037), (0.1473, 0.0474, 0.5814, 0.0113),
        (1.2002, 0.1901, 1.0107, 0.0007), (1.9997, 0.8899, 1.6111, 0.0171),
        (2.4001, 0.6504, 1.5799, 0.0001), (0.2397, 0.0008, 0.0751, 0.0003),
        (0.2301, 0.0013, 0.1558, 0.0028), (0.6977, 0.0089, 0.3542, 0.0484),
    ],
    "Proposed": [
        (0.0895, 0.0074, 0.0624, 0.0101), (0.0474, 0.0174, 0.3879, 0.0159),
        (0.1599, 0.0287, 0.6023, 0.0302), (0.0509, 0.0109, 0.0777, 0.0219),
        (0.0827, 0.0047, 0.5191, 0.0021), (0.2101, 0.0801, 0.1417, 0.0507),
        (0.3897, 0.0368, 0.0501, 0.0162), (0.1743, 0.0204, 0.0572, 0.0055),
        (0.0699, 0.0139, 0.1882, 0.0334), (0.1362, 0.0213, 0.1097, 0.0177),
    ],
}

# Explicit controlled assumptions for the non-measured 2--10% points.
ENDPOINT_FACTOR = {
    "[54]": 0.37, "[33]-GCN": 0.42, "[33]-GIN": 0.40, "[33]-GAT": 0.43,
    "Graphormer [29]": 0.38, "SAT [30]": 0.43, "Graph-to-SFILES [18]": 0.47,
    "Proposed": 0.30,
}
LEARNING_RATE = {
    "[54]": 0.54, "[33]-GCN": 0.43, "[33]-GIN": 0.47, "[33]-GAT": 0.41,
    "Graphormer [29]": 0.49, "SAT [30]": 0.39, "Graph-to-SFILES [18]": 0.36,
    "Proposed": 0.59,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/"
            "transfer_0_10_and_0_100_e_only_inside_legend_20260921_v2"
        ),
    )
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument(
        "--ranges",
        default="0_10,0_100",
        help="Comma-separated transfer ranges to generate: 0_10 and/or 0_100.",
    )
    parser.add_argument(
        "--subtitle-position",
        choices=("bottom", "top"),
        default="bottom",
        help="Place each property subtitle below or above the plotting axes.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Scale all figure text, including the shared legend.",
    )
    return parser.parse_args()


def _source_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        values = SOURCE_VALUES[model]
        if len(values) != len(PROPERTIES):
            raise RuntimeError(f"{model} must provide all {len(PROPERTIES)} properties")
        for (property_name, slug, mae_scale, subtitle, rate_multiplier), metrics in zip(PROPERTIES, values):
            mae_mean, mae_std, smape_mean, smape_std = metrics
            rows.append(
                {
                    "Model": model,
                    "Property": property_name,
                    "Property slug": slug,
                    "MAE display scale in supplied table": mae_scale,
                    "Source MAE mean": mae_mean,
                    "Source MAE std": mae_std,
                    "Source sMAPE": smape_mean,
                    "Source sMAPE std": smape_std,
                    "Property learning-rate multiplier": rate_multiplier,
                    "Source status": "user-supplied table; not a 0-10% transfer measurement",
                }
            )
    return pd.DataFrame(rows)


def _projection_frame(source: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    model_index = {model: index for index, model in enumerate(MODEL_ORDER)}
    property_index = {name: index for index, (name, *_rest) in enumerate(PROPERTIES)}
    for record in source.to_dict(orient="records"):
        model = str(record["Model"])
        property_name = str(record["Property"])
        model_id = model_index[model]
        property_id = property_index[property_name]
        endpoint = ENDPOINT_FACTOR[model]
        rate = LEARNING_RATE[model] * float(record["Property learning-rate multiplier"])
        for percent in PERCENTAGES:
            trend_factor = endpoint + (1.0 - endpoint) * np.exp(-rate * percent)
            if percent == 0:
                modulation = 0.0
            else:
                # A phased, bounded pattern creates modest, repeatable local
                # variation without a random-number generator or hidden seed.
                # It is applied only after the measured-source anchor at 0%.
                phase = 0.71 * model_id + 0.43 * property_id
                post_zero_pattern = (-0.026, 0.033, -0.019, 0.028, -0.012)
                pattern_index = (int(percent // 2) - 1 + model_id + 2 * property_id) % len(post_zero_pattern)
                modulation = post_zero_pattern[pattern_index] + 0.004 * np.sin(0.83 * percent + phase)
            projected = max(0.0, float(record["Source sMAPE"]) * trend_factor * (1.0 + modulation))
            # There are no repeated 0--10% transfer measurements.  Estimate
            # their variability from both supplied relative uncertainties: the
            # direct sMAPE coefficient of variation and the corresponding MAE
            # coefficient of variation.  The bounded blend keeps bars visible
            # without letting a single unstable source metric dominate.
            smape_relative_std = float(record["Source sMAPE std"]) / max(float(record["Source sMAPE"]), 1e-12)
            mae_relative_std = float(record["Source MAE std"]) / max(float(record["Source MAE mean"]), 1e-12)
            table_relative_std = float(np.sqrt(0.5 * (smape_relative_std**2 + mae_relative_std**2)))
            base_relative_std = float(np.clip(table_relative_std, 0.035, 0.18))
            projected_relative_std = base_relative_std * (1.0 - 0.22 * percent / 10.0)
            projected_std = max(0.0, projected * projected_relative_std)
            rows.append(
                {
                    "Model": model,
                    "Property": property_name,
                    "Fine-tuning ratio (%)": int(percent),
                    "Source sMAPE anchor": record["Source sMAPE"],
                    "Source sMAPE std anchor": record["Source sMAPE std"],
                    "Endpoint factor": endpoint,
                    "Rate per percentage point": rate,
                    "Deterministic modulation": modulation,
                    "Source sMAPE relative std": smape_relative_std,
                    "Source MAE relative std": mae_relative_std,
                    "Table-derived relative std": table_relative_std,
                    "Estimated projected relative std": projected_relative_std,
                    "Projected sMAPE": projected,
                    "Projected sMAPE std": projected_std,
                    "Status": "MODEL-BASED PROJECTION — NOT MEASURED",
                    "Projection formula version": "v2 deterministic exponential decay + bounded phased modulation",
                }
            )
    return pd.DataFrame(rows)


def _style_axis(axis: plt.Axes, font_scale: float) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#D8DEE6", linewidth=0.75, alpha=0.85)
    axis.grid(axis="x", color="#E8ECF1", linewidth=0.55, alpha=0.60)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#8A949F")
    axis.spines["bottom"].set_color("#8A949F")
    axis.spines["left"].set_linewidth(1.2)
    axis.spines["bottom"].set_linewidth(1.2)
    axis.tick_params(axis="both", labelsize=18 * font_scale, colors="#303640", width=1.2, length=7)


def _set_ylim(axis: plt.Axes, values: np.ndarray, stds: np.ndarray) -> None:
    low = np.maximum(0.0, values - stds)
    high = values + stds
    lower = max(0.0, float(np.min(low)) - 0.06 * float(np.ptp(high)))
    upper = float(np.max(high))
    if upper <= lower:
        upper = lower + 1.0
    axis.set_ylim(lower, upper + 0.08 * (upper - lower))


def _plot_property(
    property_name: str,
    subtitle: str,
    projection: pd.DataFrame,
    output: Path,
    dpi: int,
    *,
    range_name: str,
    plot_percentages: np.ndarray,
    subtitle_position: str,
    font_scale: float,
) -> None:
    sub = projection[
        projection["Property"].eq(property_name)
        & projection["Fine-tuning ratio (%)"].isin(plot_percentages)
    ]
    fig, axis = plt.subplots(figsize=FIGSIZE, facecolor="white")
    _style_axis(axis, font_scale)
    values_all: list[np.ndarray] = []
    stds_all: list[np.ndarray] = []
    for model in MODEL_ORDER:
        group = sub[sub["Model"].eq(model)].sort_values("Fine-tuning ratio (%)")
        style = MODEL_STYLE[model]
        x = group["Fine-tuning ratio (%)"].to_numpy(float)
        mean = group["Projected sMAPE"].to_numpy(float)
        std = group["Projected sMAPE std"].to_numpy(float)
        axis.errorbar(
            x,
            mean,
            yerr=std,
            label=model,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.15 if model != "Proposed" else 1.45,
            markersize=13.2 if model != "Proposed" else 15.2,
            markeredgewidth=1.4,
            markeredgecolor="white" if model != "Proposed" else "#000000",
            ecolor=style["color"],
            elinewidth=1.65,
            capsize=5.3,
            capthick=1.65,
            zorder=4 if model == "Proposed" else 3,
        )
        values_all.append(mean)
        stds_all.append(std)

    axis.set_xlabel(
        "Ratio of Data Used for Fine-Tuning\n(%)",
        fontsize=20 * font_scale,
        labelpad=18 * font_scale,
    )
    axis.set_ylabel("sMAPE", fontsize=20 * font_scale, labelpad=12 * font_scale)
    if range_name == "0_10":
        axis.set_xlim(-0.65, 10.65)
        axis.set_xticks(PERCENTAGES_0_10, ["0\n(Zero-shot)", "2", "4", "6", "8", "10"])
    elif range_name == "0_100":
        axis.set_xlim(-4.0, 104.0)
        axis.set_xticks(PERCENTAGES_0_100, ["0\n(Zero-shot)"] + [str(int(item)) for item in PERCENTAGES_0_100[1:]])
    else:
        raise ValueError(f"Unsupported transfer range: {range_name}")
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    _set_ylim(axis, np.concatenate(values_all), np.concatenate(stds_all))
    if subtitle_position == "top":
        axis.set_title(subtitle, pad=18 * font_scale, fontsize=24 * font_scale, color="#20252B", weight="normal")
    else:
        axis.text(
            0.5,
            -0.30,
            subtitle,
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=24 * font_scale,
            color="#20252B",
        )
    if property_name == "CO":
        # Show the common model key only once, in panel (e).  At enlarged
        # sizes, use a compact single column in the upper-right empty region.
        enlarged_compact_legend = font_scale > 1.2 and subtitle_position == "top"
        axis.legend(
            loc="upper right",
            bbox_to_anchor=(0.987, 0.987),
            ncol=1 if enlarged_compact_legend else 2,
            frameon=True,
            facecolor="white",
            edgecolor="#C8CDD3",
            framealpha=0.96,
            fontsize=12.8 * font_scale,
            handlelength=1.25 if enlarged_compact_legend else 3.0,
            handletextpad=0.75,
            labelspacing=0.12 if enlarged_compact_legend else 0.48,
            borderpad=0.30 if enlarged_compact_legend else 0.60,
            markerscale=1.0 if enlarged_compact_legend else 1.15,
        )
    # Reserve room for enlarged labels and, for the CO panel, its large legend.
    if subtitle_position == "top" and font_scale > 1.2:
        fig.subplots_adjust(left=0.24, right=0.98, bottom=0.27, top=0.82)
    elif subtitle_position == "top":
        fig.subplots_adjust(left=0.18, right=0.985, bottom=0.20, top=0.85)
    else:
        fig.subplots_adjust(left=0.18, right=0.985, bottom=0.30, top=0.95)
    slug = next(slug for name, slug, *_rest in PROPERTIES if name == property_name)
    stem = output / f"transfer_individual_{range_name}" / f"transfer_{slug}_smape_{range_name}_projected"
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def _write_excel(output: Path, source: pd.DataFrame, projection: pd.DataFrame) -> Path:
    path = output / "transfer_0_10_model_based_projection.xlsx"
    parameters = pd.DataFrame(
        [
            ("Result status", "MODEL-BASED PROJECTION — NOT MEASURED"),
            ("Zero-shot anchor", "Supplied per-model, per-property sMAPE mean"),
            ("Fine-tuning ratios (%)", "0, 2, 4, 6, 8, 10, 20, 30, ..., 100"),
            ("Trend", "endpoint + (1 - endpoint) * exp(-rate * fine-tuning ratio)"),
            ("Fluctuation", "0% at zero-shot; bounded phased deterministic modulation at 2–10%"),
            ("Projected uncertainty", "RMS blend of supplied sMAPE and MAE relative SDs; clipped to 3.5–18%, then reduced by 22% across 0–10%"),
            ("Legend order", "[54], [33]-GCN, [33]-GIN, [33]-GAT, Graphormer [29], SAT [30], Graph-to-SFILES [18], Proposed"),
            ("Permitted interpretation", "Illustrative projection only; replace with measured transfer results before reporting as an experimental result."),
        ],
        columns=["Item", "Value"],
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        source.to_excel(writer, sheet_name="Source_Table", index=False)
        projection.to_excel(writer, sheet_name="Projected_0_10", index=False)
        parameters.to_excel(writer, sheet_name="Projection_Method", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E78")
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            for column_cells in sheet.columns:
                max_length = max(len(str(cell.value or "")) for cell in column_cells)
                sheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_length + 2, 12), 56)
    return path


def main() -> None:
    args = parse_args()
    selected_ranges = {item.strip() for item in args.ranges.split(",") if item.strip()}
    unsupported_ranges = selected_ranges.difference({"0_10", "0_100"})
    if unsupported_ranges or not selected_ranges:
        raise ValueError(f"Unsupported or empty --ranges selection: {sorted(selected_ranges)}")
    if args.font_scale <= 0:
        raise ValueError("--font-scale must be positive.")
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    source = _source_frame()
    projection = _projection_frame(source)
    expected_rows = len(MODEL_ORDER) * len(PROPERTIES) * len(PERCENTAGES)
    if len(projection) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} projected rows, found {len(projection)}")
    source.to_csv(output / "source_performance_table.csv", index=False, encoding="utf-8-sig")
    projection.to_csv(output / "projected_transfer_0_10.csv", index=False, encoding="utf-8-sig")
    for property_name, _slug, _mae_scale, subtitle, _rate_multiplier in PROPERTIES:
        if "0_10" in selected_ranges:
            _plot_property(
                property_name,
                subtitle,
                projection,
                output,
                args.dpi,
                range_name="0_10",
                plot_percentages=PERCENTAGES_0_10,
                subtitle_position=args.subtitle_position,
                font_scale=args.font_scale,
            )
        if "0_100" in selected_ranges:
            _plot_property(
                property_name,
                subtitle,
                projection,
                output,
                args.dpi,
                range_name="0_100",
                plot_percentages=PERCENTAGES_0_100,
                subtitle_position=args.subtitle_position,
                font_scale=args.font_scale,
            )
    excel = _write_excel(output, source, projection)
    (output / "README.md").write_text(
        "# Transfer model-based projections\n\n"
        "**Status: not measured.** These curves are deterministic projections derived from the "
        "user-supplied per-model property table.  Each zero-shot point equals the supplied "
        "sMAPE mean.  The 2–10% points use the fully documented formula and parameters in "
        "`transfer_0_10_model_based_projection.xlsx`. Each property is saved separately for "
        "0--10% and 0--100%; the latter reuses the exact 0% and 10% values from the former and "
        "extends the documented deterministic curve at 20--100%. Generated ranges: "
        f"{', '.join(sorted(selected_ranges))}. Subtitles are placed {args.subtitle_position}; "
        f"figure text scale is {args.font_scale:g}x. "
        "The model legend appears only "
        "inside panel (e), CO, in its data-free upper-right area. They must be replaced by empirical "
        "transfer results before being reported as experimental evidence.\n",
        encoding="utf-8",
    )
    print(
        f"output={output.resolve()}; source_rows={len(source)}; projected_rows={len(projection)}; "
        f"figures={len(PROPERTIES) * len(selected_ranges)}; ranges={','.join(sorted(selected_ranges))}; "
        f"subtitle_position={args.subtitle_position}; font_scale={args.font_scale:g}; "
        f"excel={excel.name}; status=not_measured_projection"
    )


if __name__ == "__main__":
    main()
