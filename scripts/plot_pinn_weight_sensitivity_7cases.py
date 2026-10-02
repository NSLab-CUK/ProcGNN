#!/usr/bin/env python3
"""Render the seven-case, one-at-a-time PINN weight-sensitivity figures.

The values are transcribed verbatim from the table supplied for the paper
figure.  Each conservation term has its own seven-point sweep: three weights
below the default, the default, and three weights above it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


FIGSIZE = (10.5, 7.2)
DPI = 600

# Kept in increasing physical-weight order.  The middle entry in each sweep is
# the common default setting and is intentionally included only once.
SENSITIVITY = {
    "Mass": {
        "weights": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        "mae": [1538.4271, 1476.3184, 1402.7359, 1364.6846, 1352.9187, 1391.4632, 1460.8256],
        "smape": [0.253814, 0.245527, 0.235842, 0.230497, 0.229164, 0.233781, 0.242635],
        "panel": "(b)",
        "title": "Mass Conservation Weight",
        "xlabel": r"Mass Conservation Weight, $\lambda_{\mathrm{mass}}$",
        "labels": ["0.125", "0.25", "0.5", "1", "2", "4", "8"],
    },
    "Component": {
        "weights": [1.875e-8, 3.750e-8, 7.500e-8, 1.500e-7, 3.000e-7, 6.000e-7, 1.200e-6],
        "mae": [1456.3178, 1409.8245, 1357.9362, 1364.6846, 1378.4217, 1369.8534, 1432.7619],
        "smape": [0.243926, 0.237614, 0.229843, 0.230497, 0.232176, 0.231308, 0.239735],
        "panel": "(c)",
        "title": "Component Conservation Weight",
        "xlabel": r"Component Conservation Weight, $\lambda_{\mathrm{component}}$ ($\times10^{-8}$)",
        "labels": ["1.875", "3.750", "7.500", "15.00", "30.00", "60.00", "120.0"],
    },
    "Atom": {
        "weights": [0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6],
        "mae": [1416.5284, 1378.3461, 1360.2175, 1364.6846, 1354.7328, 1371.9463, 1411.2847],
        "smape": [0.237892, 0.232786, 0.229614, 0.230497, 0.228973, 0.231852, 0.237025],
        "panel": "(d)",
        "title": "Atom Conservation Weight",
        "xlabel": r"Atom Conservation Weight, $\lambda_{\mathrm{atom}}$",
        "labels": ["0.025", "0.05", "0.1", "0.2", "0.4", "0.8", "1.6"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/final/"
            "pinn_weight_sensitivity_7cases_20260916"
        ),
    )
    parser.add_argument("--dpi", type=int, default=DPI)
    return parser.parse_args()


def _style_axis(axis: plt.Axes, right: plt.Axes) -> None:
    axis.grid(axis="y", color="#D8DEE6", linewidth=0.75, alpha=0.85)
    axis.grid(axis="x", color="#E8ECF1", linewidth=0.55, alpha=0.60)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    right.spines["top"].set_visible(False)
    axis.spines["left"].set_color("#0072B2")
    axis.spines["bottom"].set_color("#303640")
    right.spines["right"].set_color("#C21833")
    axis.spines["left"].set_linewidth(1.2)
    axis.spines["bottom"].set_linewidth(1.2)
    right.spines["right"].set_linewidth(1.2)
    axis.tick_params(axis="x", labelsize=16, width=1.2, length=7, colors="#303640")
    axis.tick_params(axis="y", labelsize=18, width=1.2, length=7, colors="#005C91")
    right.tick_params(axis="y", labelsize=18, width=1.2, length=7, colors="#A33F00")


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for term, values in SENSITIVITY.items():
        for case, (weight, mae, smape, label) in enumerate(
            zip(values["weights"], values["mae"], values["smape"], values["labels"]), start=1
        ):
            rows.append({
                "Physics term": term,
                "Case": case,
                "Weight": weight,
                "Weight label": label,
                "MAE": mae,
                "sMAPE": smape,
                "Is default": case == 4,
            })
    return pd.DataFrame(rows)


def _separate_display_bands(
    left: plt.Axes, right: plt.Axes, mae: list[float], smape: list[float]
) -> None:
    """Keep the two raw-metric curves in separate visual bands.

    MAE and sMAPE have different units, so their independent y-axis limits can
    be set without transforming either metric.  The added headroom places MAE
    in the lower part of the panel and sMAPE in the upper part, avoiding the
    accidental apparent overlap of two similarly shaped curves.
    """
    mae_min, mae_max = min(mae), max(mae)
    smape_min, smape_max = min(smape), max(smape)
    mae_span = max(mae_max - mae_min, 1.0)
    smape_span = max(smape_max - smape_min, 1e-6)

    # Raw values are unmodified; only the two display ranges are padded.
    left.set_ylim(mae_min - 0.08 * mae_span, mae_max + 1.05 * mae_span)
    right.set_ylim(smape_min - 1.05 * smape_span, smape_max + 0.08 * smape_span)


def _plot_term(term: str, values: dict[str, object], output: Path, dpi: int) -> None:
    cases = list(range(1, 8))
    mae = values["mae"]
    smape = values["smape"]
    labels = values["labels"]
    fig, left = plt.subplots(figsize=FIGSIZE, facecolor="white")
    right = left.twinx()
    _style_axis(left, right)

    left.axvline(4, color="#94A3B8", linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    left.plot(cases, mae, color="#0072B2", marker="o", markersize=10.5, linewidth=2.8, zorder=4)
    right.plot(cases, smape, color="#C21833", marker="s", markersize=9.5, linewidth=2.8, zorder=4)
    # The default case is the common one across the three OAT sweeps.
    left.plot(4, mae[3], marker="o", markersize=13, markerfacecolor="#0072B2", markeredgecolor="white", markeredgewidth=2.0, linestyle="none", zorder=5)
    right.plot(4, smape[3], marker="s", markersize=12, markerfacecolor="#C21833", markeredgecolor="white", markeredgewidth=2.0, linestyle="none", zorder=5)

    left.set_xticks(cases, labels)
    left.set_xlim(0.72, 7.28)
    left.set_xlabel(values["xlabel"], fontsize=20, labelpad=12)
    left.set_ylabel("MAE", color="#005C91", fontsize=20, labelpad=12)
    right.set_ylabel("sMAPE", color="#A33F00", fontsize=20, labelpad=12)
    _separate_display_bands(left, right, mae, smape)
    left.text(
        0.5,
        -0.34,
        f"{values['panel']} {values['title']}",
        transform=left.transAxes,
        ha="center",
        va="top",
        fontsize=24,
        color="#20252B",
    )

    handles = [
        Line2D([0], [0], color="#0072B2", marker="o", linewidth=2.8, markersize=10, label="MAE"),
        Line2D([0], [0], color="#C21833", marker="s", linewidth=2.8, markersize=9, label="sMAPE"),
        Line2D([0], [0], color="#94A3B8", linestyle=(0, (4, 3)), linewidth=1.2, label="Default case"),
    ]
    left.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3,
        fontsize=15.0, frameon=True, facecolor="white", edgecolor="#C8CDD3",
        framealpha=0.92, handlelength=2.8, columnspacing=1.4, borderpad=0.55,
    )
    fig.subplots_adjust(left=0.17, right=0.83, bottom=0.32, top=0.93)
    stem = output / "sensitivity" / f"pinn_{term.lower()}_weight_7cases_mae_smape"
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    frame = _frame()
    if frame.groupby("Physics term").size().tolist() != [7, 7, 7]:
        raise RuntimeError("Each PINN conservation-term sweep must contain exactly seven cases.")
    frame.to_csv(output / "pinn_weight_sensitivity_7cases_data.csv", index=False, encoding="utf-8-sig")
    for term, values in SENSITIVITY.items():
        _plot_term(term, values, output, args.dpi)
    (output / "README.md").write_text(
        "# PINN conservation-weight sensitivity (seven cases per term)\n\n"
        "Each panel transcribes the supplied MAE and sMAPE table exactly.  "
        "The x-axis has the seven requested OAT weights: three below the common default, "
        "the default, and three above it.\n",
        encoding="utf-8",
    )
    print(f"output={output.resolve()}; terms=3; cases_per_term=7; formats=png,pdf,svg; dpi={args.dpi}")


if __name__ == "__main__":
    main()
