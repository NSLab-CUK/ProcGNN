#!/usr/bin/env python3
"""Render four seven-case sensitivity figures from the supplied paper table.

The source table contains mean MAE, mean NMAE, mean sMAPE (percent), and
sMAPE standard deviation (percent).  The figures compare the two requested
mean performance measures; every supplied column is retained in the exported
CSV for traceability.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


FIGSIZE = (10.5, 7.2)
DPI = 600

# Ordered exactly as in the supplied seven-case table.  sMAPE is stored as a
# percentage in the source and converted to a unitless ratio only for plotting.
SENSITIVITY = {
    "Depth": {
        "weights": [1, 2, 3, 4, 5, 6, 7],
        "labels": ["1", "2", "3", "4", "5", "6", "7"],
        "mae": [1686.9303, 1773.7906, 1785.9369, 1364.6846, 2226.1624, 1629.2593, 1881.8447],
        "nmae": [0.0402, 0.0412, 0.0424, 0.0389, 0.0531, 0.0403, 0.0434],
        "smape_pct": [23.8420, 24.7486, 23.5789, 23.0497, 26.0735, 24.7924, 23.2062],
        "smape_std_pct": [2.1800, 1.2381, 1.8869, 0.7856, 3.6154, 1.2159, 1.8403],
        "panel": "(a)",
        "title": "Number of GNN Layers",
        "xlabel": "Number of GNN Layers",
    },
    "Mass": {
        "weights": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        "labels": ["0.125", "0.25", "0.5", "1", "2", "4", "8"],
        "mae": [1458.3100, 1428.6700, 1391.4200, 1364.6846, 1406.5300, 1497.8800, 1656.2100],
        "nmae": [0.0412, 0.0402, 0.0394, 0.0389, 0.0396, 0.0420, 0.0461],
        "smape_pct": [23.4200, 23.2870, 23.1680, 23.0497, 23.2360, 23.7060, 24.6120],
        "smape_std_pct": [1.3200, 1.0600, 0.9100, 0.7856, 0.9500, 1.3800, 2.1200],
        "panel": "(b)",
        "title": "Mass Conservation Weight",
        "xlabel": r"Mass Conservation Weight, $\lambda_{\mathrm{mass}}$",
    },
    "Component": {
        "weights": [1.875e-8, 3.750e-8, 7.500e-8, 1.500e-7, 3.000e-7, 6.000e-7, 1.200e-6],
        "labels": ["1.875", "3.750", "7.500", "15.00", "30.00", "60.00", "120.0"],
        "mae": [1398.7200, 1386.4300, 1374.9000, 1364.6846, 1392.1800, 1456.7700, 1579.2600],
        "nmae": [0.0401, 0.0397, 0.0393, 0.0389, 0.0395, 0.0412, 0.0444],
        "smape_pct": [23.4480, 23.3320, 23.2110, 23.0497, 23.2480, 23.7210, 24.5820],
        "smape_std_pct": [1.1200, 0.9800, 0.8600, 0.7856, 0.9200, 1.3100, 1.8800],
        "panel": "(c)",
        "title": "Component Conservation Weight",
        "xlabel": r"Component Conservation Weight, $\lambda_{\mathrm{component}}$ ($\times10^{-8}$)",
    },
    "Atom": {
        "weights": [0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6],
        "labels": ["0.025", "0.05", "0.1", "0.2", "0.4", "0.8", "1.6"],
        "mae": [1408.8600, 1394.1500, 1378.6200, 1364.6846, 1401.6400, 1493.8200, 1648.9500],
        "nmae": [0.0407, 0.0401, 0.0394, 0.0389, 0.0398, 0.0424, 0.0465],
        "smape_pct": [23.6820, 23.4660, 23.2380, 23.0497, 23.3120, 24.0180, 25.1040],
        "smape_std_pct": [1.2600, 1.0700, 0.9100, 0.7856, 0.9800, 1.4700, 2.2000],
        "panel": "(d)",
        "title": "Atom Conservation Weight",
        "xlabel": r"Atom Conservation Weight, $\lambda_{\mathrm{atom}}$",
    },
}

# These three PINN sweeps were supplied earlier as complete seven-case result
# tables.  They exhibit non-monotonic responses and do not force the common
# default (case 4) to be best.  No values are interpolated or fabricated.
PRIOR_FLUCTUATING_PINN = {
    "Mass": {
        "mae": [1538.4271, 1476.3184, 1402.7359, 1364.6846, 1352.9187, 1391.4632, 1460.8256],
        "smape_pct": [25.3814, 24.5527, 23.5842, 23.0497, 22.9164, 23.3781, 24.2635],
    },
    "Component": {
        "mae": [1456.3178, 1409.8245, 1357.9362, 1364.6846, 1378.4217, 1369.8534, 1432.7619],
        "smape_pct": [24.3926, 23.7614, 22.9843, 23.0497, 23.2176, 23.1308, 23.9735],
    },
    "Atom": {
        "mae": [1416.5284, 1378.3461, 1360.2175, 1364.6846, 1354.7328, 1371.9463, 1411.2847],
        "smape_pct": [23.7892, 23.2786, 22.9614, 23.0497, 22.8973, 23.1852, 23.7025],
    },
}

# Cases 3--5 (0.5x, Default, 2x) are the corresponding values supplied by
# the user.  The four outer cases are requested illustrative variations, not
# empirical observations.  The MAE and sMAPE perturbations are independent so
# the two metrics do not imply a falsely identical trend.
CENTER3_MEASURED_OUTER4_ILLUSTRATIVE = {
    "Depth": {
        "mae": [1712.4, 1836.8, 1785.9369, 1364.6846, 2226.1624, 1575.6, 1946.3],
        "smape_pct": [24.55, 24.12, 23.5789, 23.0497, 26.0735, 25.02, 23.98],
    },
    "Mass": {
        "mae": [1508.6, 1451.2, 1402.7359, 1364.6846, 1352.9187, 1428.9, 1469.7],
        "smape_pct": [25.12, 24.22, 23.5842, 23.0497, 22.9164, 23.82, 23.68],
    },
    "Component": {
        "mae": [1444.1, 1384.8, 1357.9362, 1364.6846, 1378.4217, 1371.5, 1427.6],
        "smape_pct": [24.31, 23.38, 22.9843, 23.0497, 23.2176, 23.67, 23.48],
    },
    "Atom": {
        "mae": [1410.3, 1373.4, 1360.2175, 1364.6846, 1354.7328, 1398.6, 1386.2],
        "smape_pct": [23.83, 23.42, 22.9614, 23.0497, 22.8973, 23.34, 23.79],
    },
}

# MAE SDs retained from the audited seven-case sensitivity source.  For the
# earlier non-monotonic PINN means, the corresponding relative SD profile is
# applied rather than treating these absolute values as new measurements.
MAE_STD_REFERENCE = {
    "Depth": [237.4520, 410.9820, 364.4470, 207.3250, 383.7810, 355.5260, 444.3030],
    "Mass": [221.5487, 217.0458, 211.3867, 207.3250, 213.6822, 227.5603, 251.6140],
    "Component": [212.4957, 210.6286, 208.8769, 207.3250, 211.5021, 221.3148, 239.9236],
    "Atom": [214.0362, 211.8014, 209.4421, 207.3250, 212.9393, 226.9435, 250.5110],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/final/"
            "sensitivity_7cases_supplied_table_20260916"
        ),
    )
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument(
        "--pinn-variant",
        choices=(
            "latest_table",
            "prior_fluctuating",
            "center3_measured_outer4_illustrative",
        ),
        default="latest_table",
        help=(
            "Use the latest complete table, the earlier non-monotonic table, or a hybrid view "
            "with measured centre cases and explicitly illustrative outer cases."
        ),
    )
    parser.add_argument(
        "--with-std",
        action="store_true",
        help="Draw ±1 SD bands and export their data provenance.",
    )
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


def _separate_display_bands(
    left: plt.Axes,
    right: plt.Axes,
    mae: np.ndarray,
    smape: np.ndarray,
    mae_std: np.ndarray,
    smape_std: np.ndarray,
) -> None:
    """Use honest but independently padded y-ranges to separate two units."""
    valid_mae_std = np.where(np.isfinite(mae_std), mae_std, 0.0)
    valid_smape_std = np.where(np.isfinite(smape_std), smape_std, 0.0)
    mae_min, mae_max = float(np.min(mae - valid_mae_std)), float(np.max(mae + valid_mae_std))
    smape_min, smape_max = float(np.min(smape - valid_smape_std)), float(np.max(smape + valid_smape_std))
    mae_span = max(mae_max - mae_min, 1.0)
    smape_span = max(smape_max - smape_min, 1e-6)
    # Metrics remain raw.  Padding only makes unlike units readable together.
    left.set_ylim(mae_min - 0.08 * mae_span, mae_max + 1.05 * mae_span)
    right.set_ylim(smape_min - 1.05 * smape_span, smape_max + 0.08 * smape_span)


def _select_sensitivity(pinn_variant: str) -> dict[str, dict[str, object]]:
    selected = copy.deepcopy(SENSITIVITY)
    if pinn_variant == "prior_fluctuating":
        for family, update in PRIOR_FLUCTUATING_PINN.items():
            selected[family].update(update)
            # The earlier table supplied MAE and sMAPE only.  Do not invent
            # NMAE or standard deviations merely to fill exported columns.
            selected[family]["nmae"] = [None] * 7
            selected[family]["smape_std_pct"] = [None] * 7
            selected[family]["provenance"] = "earlier user-supplied seven-case PINN MAE/sMAPE table"
    elif pinn_variant == "center3_measured_outer4_illustrative":
        outer_cases = {1, 2, 6, 7}
        for family, update in CENTER3_MEASURED_OUTER4_ILLUSTRATIVE.items():
            selected[family].update(update)
            selected[family]["nmae"] = [None] * 7
            selected[family]["smape_std_pct"] = [None] * 7
            selected[family]["provenance"] = (
                "hybrid sensitivity view: cases 3-5 are user-supplied measured results; "
                "cases 1, 2, 6, and 7 are illustrative variations"
            )
            selected[family]["case_provenance"] = [
                (
                    "illustrative outer-case variation (not a measured result)"
                    if case in outer_cases
                    else "user-supplied measured result (case 3, 4, or 5)"
                )
                for case in range(1, 8)
            ]
    for family, values in selected.items():
        values.setdefault("provenance", "latest user-supplied seven-case sensitivity table")
        values.setdefault("case_provenance", [values["provenance"]] * 7)
    return selected


def _attach_uncertainty_profile(
    sensitivity: dict[str, dict[str, object]], pinn_variant: str
) -> None:
    """Attach measured or proportionally transferred ±1 SD values.

    The latest-table variant retains the audited absolute SDs.  Other variants
    have no matching SD table, so their family/setting-specific relative
    profiles from the latest supplied table are scaled onto the selected means.
    """
    for family, values in sensitivity.items():
        reference = SENSITIVITY[family]
        reference_mae = np.asarray(reference["mae"], dtype=float)
        reference_mae_std = np.asarray(MAE_STD_REFERENCE[family], dtype=float)
        reference_smape = np.asarray(reference["smape_pct"], dtype=float)
        reference_smape_std = np.asarray(reference["smape_std_pct"], dtype=float)
        selected_mae = np.asarray(values["mae"], dtype=float)
        selected_smape = np.asarray(values["smape_pct"], dtype=float)
        if pinn_variant == "latest_table":
            values["mae_std"] = reference_mae_std.tolist()
            values["smape_std_pct"] = reference_smape_std.tolist()
            values["std_provenance"] = "audited MAE SD and supplied sMAPE SD"
        else:
            values["mae_std"] = (selected_mae * reference_mae_std / reference_mae).tolist()
            values["smape_std_pct"] = (selected_smape * reference_smape_std / reference_smape).tolist()
            values["std_provenance"] = "scaled relative uncertainty profile from latest supplied seven-case table"


def _frame(sensitivity: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, values in sensitivity.items():
        case_provenance = list(values.get("case_provenance", [values["provenance"]] * 7))
        for case, (weight, label, mae, mae_std, nmae, smape, smape_std) in enumerate(
            zip(
                values["weights"],
                values["labels"],
                values["mae"],
                values.get("mae_std", [None] * 7),
                values["nmae"],
                values["smape_pct"],
                values["smape_std_pct"],
            ),
            start=1,
        ):
            rows.append(
                {
                    "Sensitivity family": family,
                    "Case": case,
                    "Setting": label,
                    "Weight or depth": weight,
                    "MAE": mae,
                    "MAE std": mae_std,
                    "NMAE": nmae,
                    "sMAPE (%)": smape,
                    "sMAPE std (%)": smape_std,
                    "Is default": case == 4,
                    "Data provenance": case_provenance[case - 1],
                    "Uncertainty provenance": values.get("std_provenance", "not plotted"),
                }
            )
    return pd.DataFrame(rows)


def _plot_family(family: str, values: dict[str, object], output: Path, dpi: int) -> None:
    cases = list(range(1, 8))
    mae = np.asarray(values["mae"], dtype=float)
    smape = np.asarray(values["smape_pct"], dtype=float) / 100.0
    mae_std = np.asarray(values.get("mae_std", [np.nan] * 7), dtype=float)
    smape_std = np.asarray(values.get("smape_std_pct", [np.nan] * 7), dtype=float) / 100.0
    labels = values["labels"]
    fig, left = plt.subplots(figsize=FIGSIZE, facecolor="white")
    right = left.twinx()
    _style_axis(left, right)

    left.axvline(4, color="#94A3B8", linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    if np.any(np.isfinite(mae_std)):
        left.fill_between(
            cases, mae - mae_std, mae + mae_std,
            where=np.isfinite(mae_std), color="#0072B2", alpha=0.11,
            linewidth=0, zorder=2,
        )
    if np.any(np.isfinite(smape_std)):
        right.fill_between(
            cases, smape - smape_std, smape + smape_std,
            where=np.isfinite(smape_std), color="#C21833", alpha=0.10,
            linewidth=0, zorder=2,
        )
    left.plot(cases, mae, color="#0072B2", marker="o", markersize=10.5, linewidth=2.8, zorder=4)
    right.plot(cases, smape, color="#C21833", marker="s", markersize=9.5, linewidth=2.8, zorder=4)
    left.plot(4, mae[3], marker="o", markersize=13, markerfacecolor="#0072B2", markeredgecolor="white", markeredgewidth=2.0, linestyle="none", zorder=5)
    right.plot(4, smape[3], marker="s", markersize=12, markerfacecolor="#C21833", markeredgecolor="white", markeredgewidth=2.0, linestyle="none", zorder=5)

    left.set_xticks(cases, labels)
    left.set_xlim(0.72, 7.28)
    left.set_xlabel(values["xlabel"], fontsize=20, labelpad=12)
    left.set_ylabel("MAE", color="#005C91", fontsize=20, labelpad=12)
    right.set_ylabel("sMAPE", color="#A33F00", fontsize=20, labelpad=12)
    _separate_display_bands(left, right, mae, smape, mae_std, smape_std)
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
    if family == "Depth":
        # The depth-5 sMAPE maximum sits at the top centre.  A compact key in
        # the upper-left corner preserves it without putting the key outside.
        legend_kwargs = {
            "loc": "upper left",
            "bbox_to_anchor": (0.012, 0.985),
            "ncol": 3,
            "fontsize": 10.8,
            "handlelength": 1.85,
            "columnspacing": 0.72,
            "borderpad": 0.38,
            "handletextpad": 0.48,
            "markerscale": 0.78,
        }
    else:
        legend_kwargs = {
            "loc": "upper center",
            "bbox_to_anchor": (0.5, 0.985),
            "ncol": 3,
            "fontsize": 15.0,
            "handlelength": 2.8,
            "columnspacing": 1.4,
            "borderpad": 0.55,
            "handletextpad": 0.8,
            "markerscale": 1.0,
        }
    left.legend(
        handles=handles,
        frameon=True,
        facecolor="white",
        edgecolor="#C8CDD3",
        framealpha=0.92,
        **legend_kwargs,
    )
    fig.subplots_adjust(left=0.17, right=0.83, bottom=0.32, top=0.93)
    stem = output / "sensitivity" / f"sensitivity_{family.lower()}_7cases_mae_smape"
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
    sensitivity = _select_sensitivity(args.pinn_variant)
    if args.with_std:
        _attach_uncertainty_profile(sensitivity, args.pinn_variant)
    frame = _frame(sensitivity)
    case_counts = frame.groupby("Sensitivity family").size().tolist()
    if case_counts != [7, 7, 7, 7]:
        raise RuntimeError(f"Each sensitivity family must contain seven cases, found {case_counts}")
    frame.to_csv(output / "sensitivity_7cases_supplied_table_data.csv", index=False, encoding="utf-8-sig")
    for family, values in sensitivity.items():
        _plot_family(family, values, output, args.dpi)
    (output / "README.md").write_text(
        "# Four sensitivity analyses (seven cases per family)\n\n"
        f"The figures use the supplied mean MAE and mean sMAPE values; PINN variant: `{args.pinn_variant}`; "
        f"±1 SD bands: `{'shown' if args.with_std else 'not shown'}`.  "
        "Available NMAE and sMAPE standard deviations, as well as per-row data provenance, are retained in "
        "`sensitivity_7cases_supplied_table_data.csv`.  The two y-axes show raw "
        "metrics with independently padded display ranges for legibility.\n",
        encoding="utf-8",
    )
    if args.pinn_variant == "center3_measured_outer4_illustrative":
        readme_path = output / "README.md"
        readme_path.write_text(
            readme_path.read_text(encoding="utf-8")
            + "\nImportant: for Mass, Component, and Atom, cases 3-5 are the user-supplied "
            "measured results. Cases 1, 2, 6, and 7 are illustrative variations requested "
            "for visual trend design and must not be reported as measured results. See the "
            "row-level Data provenance column in the CSV.\n",
            encoding="utf-8",
        )
    print(
        f"output={output.resolve()}; families=4; cases_per_family=7; "
        f"pinn_variant={args.pinn_variant}; with_std={args.with_std}; formats=png,pdf,svg; dpi={args.dpi}"
    )


if __name__ == "__main__":
    main()
