#!/usr/bin/env python3
"""Draw one publication-ready signed-SHAP flowsheet overlay per process.

The source SHAP files contain explanations for outputs with incompatible
physical units (temperature, pressure, fractions, and mass flow).  Therefore,
the node-level SHAP values are first converted to relative contributions within
each individual explanation before averaging across target edges, properties,
and samples.  Marker color encodes the mean signed relative contribution and
marker area encodes the mean absolute relative contribution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from PIL import Image


FIGSIZE = (8.4, 5.2)
PROCESS_IDS = tuple(range(1, 11))
GROUP_KEYS = ["process_id", "sample_index", "target_edge_id", "target_property"]
REQUIRED_SHAP_COLUMNS = {
    "process_id",
    "sample_index",
    "target_edge_id",
    "target_property",
    "node_name",
    "feature_name",
    "shap_value",
}
FEED_FEATURE_GROUP = {
    "feed_ch4_flow": "FEED_CH4",
    "feed_air_flow": "FEED_AIR",
    "log_air_ch4_ratio": "FEED_AIR",
    "feed_water_flow": "FEED_WATER",
    "log_water_ch4_ratio": "FEED_WATER",
}
DISPLAY_NODE_LABEL = {
    "FEED_CH4": r"CH$_4$",
    "FEED_AIR": "Air",
    "FEED_WATER": "Water",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot signed SHAP values on the ten process flowsheets."
    )
    parser.add_argument(
        "--shap-root",
        required=True,
        help="Directory containing P01 ... P10/sample_level_shap.csv.",
    )
    parser.add_argument(
        "--coordinate-file",
        default="data/process_overall_img/unit_coordinates.csv",
    )
    parser.add_argument(
        "--feed-coordinate-file",
        default="data/process_overall_img/feed_coordinates.csv",
    )
    parser.add_argument(
        "--split-v-input-feeds",
        action="store_true",
        help=(
            "Replace V_INPUT with separate CH4, AIR, and WATER markers. "
            "The AIR/CH4 ratio is grouped with AIR and WATER/CH4 with WATER."
        ),
    )
    parser.add_argument(
        "--target-output-coordinate-file",
        default="data/process_overall_img/target_output_coordinates.csv",
    )
    parser.add_argument(
        "--split-v-output-targets",
        action="store_true",
        help=(
            "Remove the zero-attribution V_OUTPUT marker and mark the physical "
            "outlet of the selected target edge instead. Requires target-edge mode."
        ),
    )
    parser.add_argument(
        "--image-dir",
        default="data/process_overall_img",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/0819final/Paper_Tables_Final3_figures/shap_signed_flowsheets",
    )
    parser.add_argument(
        "--aggregation-level",
        choices=("process", "target-edge"),
        default="process",
        help=(
            "Create one overview per process or one figure per target edge. "
            "Both modes average equally across the ten target properties."
        ),
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--label-all-nodes",
        action="store_true",
        help="Label every canonical node in addition to the labels already present in the PFD.",
    )
    parser.add_argument(
        "--linear-threshold",
        type=float,
        default=2.0e-3,
        help="Linear region around zero for the shared symmetric-log color scale.",
    )
    return parser.parse_args()


def _find_image(image_dir: Path, process_id: int) -> Path:
    candidates = [
        path
        for path in sorted(image_dir.glob("*.png"))
        if (match := re.match(r"^(\d+)", path.stem))
        and int(match.group(1)) == process_id
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one image beginning with {process_id!r} in "
            f"{image_dir}, found {[str(path) for path in candidates]}"
        )
    return candidates[0]


def _load_process_summary(
    shap_file: Path,
    process_id: int,
    aggregation_level: str,
    split_v_input_feeds: bool,
    split_v_output_targets: bool,
) -> pd.DataFrame:
    header = pd.read_csv(shap_file, nrows=0)
    missing = sorted(REQUIRED_SHAP_COLUMNS.difference(header.columns))
    if missing:
        raise ValueError(f"{shap_file} is missing columns: {missing}")

    frame = pd.read_csv(
        shap_file,
        usecols=sorted(REQUIRED_SHAP_COLUMNS),
        dtype={
            "process_id": "int16",
            "sample_index": "int32",
            "target_edge_id": "string",
            "target_property": "string",
            "node_name": "string",
            "feature_name": "string",
            "shap_value": "float64",
        },
    )
    found_processes = sorted(frame["process_id"].dropna().unique().tolist())
    if found_processes != [process_id]:
        raise ValueError(
            f"{shap_file} contains process IDs {found_processes}, expected [{process_id}]"
        )

    if split_v_input_feeds:
        is_v_input = frame["node_name"].eq("V_INPUT")
        mapped_feed = frame["feature_name"].map(FEED_FEATURE_GROUP)
        unmapped_v_input = is_v_input & mapped_feed.isna()
        unmapped_nonzero = frame.loc[unmapped_v_input, "shap_value"].abs().max()
        if pd.notna(unmapped_nonzero) and float(unmapped_nonzero) > 1.0e-12:
            names = sorted(frame.loc[unmapped_v_input, "feature_name"].astype(str).unique())
            raise ValueError(
                f"P{process_id:02d} has nonzero unmapped V_INPUT SHAP features: {names}"
            )
        frame.loc[is_v_input & mapped_feed.notna(), "node_name"] = mapped_feed.loc[
            is_v_input & mapped_feed.notna()
        ]
        frame = frame.loc[~unmapped_v_input].copy()

        # Some process topologies have no AIR feed.  Those rows are masked in
        # the model and have exactly zero SHAP; omit the nonexistent marker.
        feed_activity = (
            frame.loc[frame["node_name"].isin(FEED_FEATURE_GROUP.values())]
            .groupby("node_name", observed=True)["shap_value"]
            .apply(lambda values: float(values.abs().max()))
        )
        inactive_feeds = set(feed_activity.loc[feed_activity.le(1.0e-12)].index)
        if inactive_feeds:
            frame = frame.loc[~frame["node_name"].isin(inactive_feeds)].copy()

    if split_v_output_targets:
        is_v_output = frame["node_name"].eq("V_OUTPUT")
        maximum_output_shap = frame.loc[is_v_output, "shap_value"].abs().max()
        if pd.notna(maximum_output_shap) and float(maximum_output_shap) > 1.0e-12:
            raise ValueError(
                f"P{process_id:02d} has nonzero V_OUTPUT raw-feature SHAP and "
                "cannot be replaced by a structural outlet marker."
            )
        frame = frame.loc[~is_v_output].copy()

    # SHAP values are additive, so all raw features belonging to the same node
    # (including the direct-feed features of V_INPUT) are summed first.
    node_explanation = (
        frame.groupby(GROUP_KEYS + ["node_name"], observed=True, as_index=False)[
            "shap_value"
        ]
        .sum()
        .rename(columns={"shap_value": "node_shap"})
    )

    # Each output has a different physical unit.  Normalize within every
    # explanation so each sample/edge/property contributes equal total weight.
    denominator = node_explanation.groupby(GROUP_KEYS, observed=True)[
        "node_shap"
    ].transform(lambda values: values.abs().sum())
    valid = denominator > np.finfo(float).eps
    node_explanation["relative_signed_shap"] = np.where(
        valid, node_explanation["node_shap"] / denominator, 0.0
    )
    node_explanation["relative_absolute_shap"] = np.where(
        valid, node_explanation["node_shap"].abs() / denominator, 0.0
    )

    summary_keys = ["process_id", "node_name"]
    if aggregation_level == "target-edge":
        summary_keys.insert(1, "target_edge_id")
    summary = (
        node_explanation.groupby(summary_keys, observed=True, as_index=False)
        .agg(
            mean_signed_relative_shap=("relative_signed_shap", "mean"),
            mean_absolute_relative_shap=("relative_absolute_shap", "mean"),
            signed_relative_shap_std=("relative_signed_shap", "std"),
            explanation_count=("relative_signed_shap", "size"),
        )
        .sort_values("node_name")
        .reset_index(drop=True)
    )
    return summary


def _tick_label(value: float, _: int) -> str:
    if abs(value) < 5.0e-7:
        return "0"
    return f"{value:.3f}"


def _draw_process(
    *,
    process_id: int,
    image_path: Path,
    coordinates: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    color_norm: colors.Normalize,
    size_max: float,
    dpi: int,
    target_edge_id: str | None = None,
    target_output: pd.Series | None = None,
    label_all_nodes: bool = False,
) -> None:
    merged = coordinates.merge(summary, on=["process_id", "node_name"], how="left")
    metric_columns = [
        "mean_signed_relative_shap",
        "mean_absolute_relative_shap",
        "signed_relative_shap_std",
        "explanation_count",
    ]
    if merged[metric_columns].isna().any().any():
        missing = merged.loc[
            merged["mean_signed_relative_shap"].isna(), "node_name"
        ].tolist()
        raise ValueError(f"P{process_id:02d} has coordinates without SHAP values: {missing}")

    image = Image.open(image_path).convert("RGB")
    values = merged["mean_signed_relative_shap"].to_numpy(dtype=float)
    importance = merged["mean_absolute_relative_shap"].to_numpy(dtype=float)
    sizes = 20.0 + 205.0 * np.sqrt(np.clip(importance / size_max, 0.0, 1.0))

    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    # Reserve a narrow header for a compact color key and a footer for the
    # process/target-edge caption.  The flowsheet itself remains title-free.
    axis = fig.add_axes([0.035, 0.105, 0.93, 0.77])
    axis.imshow(image, interpolation="lanczos")
    axis.scatter(
        merged["x"],
        merged["y"],
        s=sizes + 18.0,
        facecolors="none",
        edgecolors="white",
        linewidths=1.15,
        alpha=0.92,
        zorder=2,
    )
    axis.scatter(
        merged["x"],
        merged["y"],
        c=values,
        s=sizes,
        cmap="RdBu_r",
        norm=color_norm,
        alpha=0.78,
        edgecolors="#303030",
        linewidths=0.36,
        zorder=3,
    )

    width, height = image.size
    x_offset = max(width * 0.004, 3.0)
    y_offset = max(height * 0.009, 3.0)
    for row in merged.itertuples(index=False):
        node_name = str(row.node_name)
        is_feed = node_name in DISPLAY_NODE_LABEL
        if not (is_feed or label_all_nodes):
            continue
        axis.text(
            float(row.x) + x_offset,
            float(row.y) - y_offset,
            DISPLAY_NODE_LABEL.get(node_name, node_name),
            fontsize=6.2 if is_feed else (5.0 if len(merged) <= 20 else 4.6),
            fontweight="semibold" if is_feed else "normal",
            color="#252525",
            ha="left",
            va="bottom",
            zorder=4,
            bbox={
                "facecolor": "white",
                "alpha": 0.86 if is_feed else 0.64,
                "edgecolor": "#D5D5D5" if is_feed else "none",
                "linewidth": 0.35,
                "boxstyle": "round,pad=0.16" if is_feed else "square,pad=0.08",
            },
        )

    if target_output is not None:
        target_x = float(target_output["x"])
        target_y = float(target_output["y"])
        output_label = str(target_output["output_label"])
        axis.scatter(
            [target_x],
            [target_y],
            s=122.0,
            marker="o",
            facecolors="none",
            edgecolors="#D39400",
            linewidths=1.55,
            zorder=6,
        )
        axis.scatter(
            [target_x],
            [target_y],
            s=31.0,
            marker="*",
            c="#D39400",
            edgecolors="white",
            linewidths=0.35,
            zorder=7,
        )

    axis.set_xlim(-0.012 * width, 1.012 * width)
    axis.set_ylim(1.018 * height, -0.018 * height)
    axis.axis("off")

    # Compact color key in the upper-left margin.  This stays clear of both
    # the flowsheet and the paper-style caption below the figure.
    color_axis = fig.add_axes([0.055, 0.925, 0.265, 0.011])
    scalar = ScalarMappable(norm=color_norm, cmap="RdBu_r")
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, cax=color_axis, orientation="horizontal")
    colorbar.set_label("Mean signed relative SHAP", fontsize=6.1, labelpad=1.1)
    colorbar.ax.tick_params(labelsize=5.3, length=1.8, pad=0.9)
    colorbar.ax.xaxis.set_major_formatter(FuncFormatter(_tick_label))
    colorbar.outline.set_linewidth(0.45)

    if target_edge_id is not None:
        caption = f"Process {process_id} \N{EN DASH} Target Edge {target_edge_id}"
        fig.text(
            0.5,
            0.035,
            caption,
            ha="center",
            va="center",
            fontsize=9.2,
            color="#20252B",
        )

    stem = (
        f"shap_signed_flowsheet_{target_edge_id}"
        if target_edge_id
        else f"shap_signed_flowsheet_P{process_id:02d}"
    )
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, facecolor="white")
    fig.savefig(output_dir / f"{stem}.pdf", facecolor="white")
    fig.savefig(output_dir / f"{stem}.svg", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    shap_root = Path(args.shap_root).expanduser().resolve()
    # Keep workspace-local paths relative on Windows.  Resolving a Korean UNC
    # working directory through a non-Unicode filesystem codec can corrupt the
    # path before matplotlib writes the figure.
    coordinate_file = Path(args.coordinate_file).expanduser()
    feed_coordinate_file = Path(args.feed_coordinate_file).expanduser()
    target_output_coordinate_file = (
        Path(args.target_output_coordinate_file).expanduser()
    )
    image_dir = Path(args.image_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.split_v_output_targets and args.aggregation_level != "target-edge":
        raise ValueError("--split-v-output-targets requires --aggregation-level target-edge")

    coordinates = pd.read_csv(coordinate_file)
    required_coordinates = {"process_id", "node_name", "x", "y"}
    missing_coordinates = sorted(required_coordinates.difference(coordinates.columns))
    if missing_coordinates:
        raise ValueError(f"Coordinate file is missing columns: {missing_coordinates}")
    coordinates = coordinates.loc[coordinates["process_id"].isin(PROCESS_IDS)].copy()
    coordinates["process_id"] = coordinates["process_id"].astype(int)
    coordinates["node_name"] = coordinates["node_name"].astype(str)

    if args.split_v_input_feeds:
        feed_coordinates = pd.read_csv(feed_coordinate_file)
        missing_feed_coordinates = sorted(
            required_coordinates.difference(feed_coordinates.columns)
        )
        if missing_feed_coordinates:
            raise ValueError(
                f"Feed coordinate file is missing columns: {missing_feed_coordinates}"
            )
        feed_coordinates = feed_coordinates.loc[
            feed_coordinates["process_id"].isin(PROCESS_IDS),
            coordinates.columns.intersection(feed_coordinates.columns),
        ].copy()
        feed_coordinates["process_id"] = feed_coordinates["process_id"].astype(int)
        feed_coordinates["node_name"] = feed_coordinates["node_name"].astype(str)
        coordinates = pd.concat(
            [coordinates.loc[~coordinates["node_name"].eq("V_INPUT")], feed_coordinates],
            ignore_index=True,
        )

    target_output_coordinates: pd.DataFrame | None = None
    if args.split_v_output_targets:
        target_output_coordinates = pd.read_csv(target_output_coordinate_file)
        required_target_output_columns = {
            "process_id",
            "target_edge_id",
            "output_label",
            "x",
            "y",
        }
        missing_target_output_columns = sorted(
            required_target_output_columns.difference(target_output_coordinates.columns)
        )
        if missing_target_output_columns:
            raise ValueError(
                "Target-output coordinate file is missing columns: "
                f"{missing_target_output_columns}"
            )
        target_output_coordinates["process_id"] = target_output_coordinates[
            "process_id"
        ].astype(int)
        target_output_coordinates["target_edge_id"] = target_output_coordinates[
            "target_edge_id"
        ].astype(str)
        if target_output_coordinates["target_edge_id"].duplicated().any():
            duplicated = target_output_coordinates.loc[
                target_output_coordinates["target_edge_id"].duplicated(False),
                "target_edge_id",
            ].tolist()
            raise ValueError(f"Duplicate target-output coordinates: {duplicated}")
        coordinates = coordinates.loc[~coordinates["node_name"].eq("V_OUTPUT")].copy()

    summaries: list[pd.DataFrame] = []
    source_files: dict[str, str] = {}
    for process_id in PROCESS_IDS:
        shap_file = shap_root / f"P{process_id:02d}" / "sample_level_shap.csv"
        if not shap_file.is_file():
            raise FileNotFoundError(f"Missing SHAP source: {shap_file}")
        summary = _load_process_summary(
            shap_file,
            process_id,
            args.aggregation_level,
            args.split_v_input_feeds,
            args.split_v_output_targets,
        )
        coordinate_nodes = set(
            coordinates.loc[coordinates["process_id"].eq(process_id), "node_name"]
        )
        shap_nodes = set(summary["node_name"].astype(str))
        if coordinate_nodes != shap_nodes:
            raise ValueError(
                f"P{process_id:02d} coordinate/SHAP mismatch: "
                f"missing coordinates={sorted(shap_nodes - coordinate_nodes)}, "
                f"coordinates without SHAP={sorted(coordinate_nodes - shap_nodes)}"
            )
        summaries.append(summary)
        source_files[f"P{process_id:02d}"] = str(shap_file)

    combined = pd.concat(summaries, ignore_index=True)
    combined = combined.merge(
        coordinates[["process_id", "node_name", "x", "y"]],
        on=["process_id", "node_name"],
        how="left",
        validate="one_to_one" if args.aggregation_level == "process" else "many_to_one",
    )
    summary_name = (
        "shap_node_summary.csv"
        if args.aggregation_level == "process"
        else "shap_target_edge_node_summary.csv"
    )
    combined.to_csv(output_dir / summary_name, index=False)

    maximum = float(combined["mean_signed_relative_shap"].abs().max())
    rounded_limit = max(np.ceil(maximum * 100.0) / 100.0, 0.01)
    linear_threshold = min(float(args.linear_threshold), rounded_limit / 2.0)
    color_norm = colors.SymLogNorm(
        linthresh=linear_threshold,
        linscale=1.0,
        vmin=-rounded_limit,
        vmax=rounded_limit,
        base=10,
    )
    size_max = max(float(combined["mean_absolute_relative_shap"].max()), 1.0e-12)

    rendered_outputs: list[str] = []
    for process_id in PROCESS_IDS:
        process_coordinates = coordinates.loc[
            coordinates["process_id"].eq(process_id)
        ].copy()
        process_rows = combined.loc[combined["process_id"].eq(process_id)].copy()
        target_edges: list[str | None]
        if args.aggregation_level == "target-edge":
            target_edges = sorted(process_rows["target_edge_id"].astype(str).unique())
        else:
            target_edges = [None]
        for target_edge_id in target_edges:
            selected = process_rows
            if target_edge_id is not None:
                selected = selected.loc[
                    selected["target_edge_id"].astype(str).eq(target_edge_id)
                ]
            process_summary = selected[
                [
                    "process_id",
                    "node_name",
                    "mean_signed_relative_shap",
                    "mean_absolute_relative_shap",
                    "signed_relative_shap_std",
                    "explanation_count",
                ]
            ].copy()
            target_output: pd.Series | None = None
            if target_output_coordinates is not None:
                matches = target_output_coordinates.loc[
                    target_output_coordinates["target_edge_id"].eq(str(target_edge_id))
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Expected one target-output coordinate for {target_edge_id}, "
                        f"found {len(matches)}"
                    )
                target_output = matches.iloc[0]
                if int(target_output["process_id"]) != process_id:
                    raise ValueError(
                        f"Target-output process mismatch for {target_edge_id}: "
                        f"{target_output['process_id']} != {process_id}"
                    )
            _draw_process(
                process_id=process_id,
                image_path=_find_image(image_dir, process_id),
                coordinates=process_coordinates,
                summary=process_summary,
                output_dir=output_dir,
                color_norm=color_norm,
                size_max=size_max,
                dpi=int(args.dpi),
                target_edge_id=target_edge_id,
                target_output=target_output,
                label_all_nodes=bool(args.label_all_nodes),
            )
            output_id = target_edge_id or f"P{process_id:02d}"
            rendered_outputs.append(output_id)
            print(f"[DONE] {output_id}", flush=True)

    manifest = {
        "source_shap_root": str(shap_root),
        "source_files": source_files,
        "coordinate_file": str(coordinate_file),
        "feed_coordinate_file": (
            str(feed_coordinate_file) if args.split_v_input_feeds else None
        ),
        "split_v_input_feeds": bool(args.split_v_input_feeds),
        "feed_feature_groups": FEED_FEATURE_GROUP if args.split_v_input_feeds else None,
        "target_output_coordinate_file": (
            str(target_output_coordinate_file) if args.split_v_output_targets else None
        ),
        "split_v_output_targets": bool(args.split_v_output_targets),
        "target_output_marker": (
            "gold diamond; structural locator only because V_OUTPUT raw-feature SHAP is zero"
            if args.split_v_output_targets
            else None
        ),
        "image_dir": str(image_dir),
        "process_ids": list(PROCESS_IDS),
        "aggregation_level": args.aggregation_level,
        "rendered_outputs": rendered_outputs,
        "figure_count": len(rendered_outputs),
        "figure_size_inches": list(FIGSIZE),
        "png_dpi": int(args.dpi),
        "label_all_nodes": bool(args.label_all_nodes),
        "color_map": "RdBu_r (blue=negative, white=zero, red=positive)",
        "shared_color_limits": [-rounded_limit, rounded_limit],
        "color_normalization": {
            "type": "SymLogNorm",
            "linear_threshold": linear_threshold,
            "base": 10,
        },
        "aggregation": {
            "step_1": "sum raw feature SHAP values within each node and explanation",
            "step_2": "divide node SHAP by the sum of absolute node SHAP values within that explanation",
            "step_3_color": (
                "mean signed relative SHAP across samples and target properties, separately for each target edge"
                if args.aggregation_level == "target-edge"
                else "mean signed relative SHAP across samples, target edges, and target properties"
            ),
            "step_3_size": (
                "mean absolute relative SHAP across samples and target properties, separately for each target edge"
                if args.aggregation_level == "target-edge"
                else "mean absolute relative SHAP across samples, target edges, and target properties"
            ),
            "reason": "target properties have incompatible physical units and scales",
        },
    }
    (output_dir / "shap_overlay_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[COMPLETE] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
