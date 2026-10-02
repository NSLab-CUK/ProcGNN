#!/usr/bin/env python3
"""Plot process-balanced mean signed SHAP values by canonical node type.

The input is the process-level SHAP node summary used for the ten signed-SHAP
flowsheets.  Its values are already relative signed contributions, normalized
within each explanation before averaging across samples, target edges, and
target properties.  This script first averages nodes of the same type within
each process, then averages those process means.  Thus a process with several
instances of a unit type does not receive extra weight.
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


REQUIRED_SHAP_COLUMNS = {
    "process_id",
    "node_name",
    "mean_signed_relative_shap",
}
REQUIRED_NODE_COLUMNS = {"process_id", "node_name", "unit_type"}
REQUIRED_FEED_COLUMNS = {
    "process_id",
    "target_edge_id",
    "node_name",
    "mean_signed_relative_shap",
}
NEGATIVE_BLUE = "#2166AC"
POSITIVE_RED = "#B2182B"
PAPER_FIGURE_ASPECT = 8.4 / 5.2
SHAP_FIGSIZE = (10.5, 10.5 / PAPER_FIGURE_ASPECT)
TICK_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 18
BAR_EDGE_COLOR = "#FFFFFF"
GRID_COLOR = "#DCE2E9"
ZERO_LINE_COLOR = "#4B5563"
FEED_NODE_ORDER = ("FEED_CH4", "FEED_AIR", "FEED_WATER")
DISPLAY_FEED_TYPE = {
    "FEED_CH4": r"CH$_4$ feed",
    "FEED_AIR": "Air feed",
    "FEED_WATER": "Water feed",
}
DISPLAY_NODE_TYPE = {
    "virtual_io": "Virtual input/output",
    "HX_cold": "Heat exchanger (cold)",
    "HX_hot": "Heat exchanger (hot)",
    "HX_dt": "Heat exchanger (ΔT)",
    "SMR reactor": "SMR reactor",
    "WGS reactor": "WGS reactor",
    "burner": "Burner",
    "compressor": "Compressor",
    "cooler": "Cooler",
    "flash": "Flash drum",
    "heater": "Heater",
    "mixer": "Mixer",
    "psa": "PSA",
    "pump": "Pump",
    "splitter": "Splitter",
    "turbine": "Turbine",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one signed-SHAP node-type mean across P01-P10."
    )
    parser.add_argument(
        "--shap-summary",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/"
            "shap_signed_flowsheets/shap_node_summary.csv"
        ),
        help="Process-level SHAP node summary for P01-P10.",
    )
    parser.add_argument(
        "--canonical-nodes",
        default="data/reference/v3/canonical_nodes.csv",
        help="Canonical node table providing the unit_type for every node.",
    )
    parser.add_argument(
        "--feed-summary",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/"
            "shap_signed_target_edges_feed_split/shap_target_edge_node_summary.csv"
        ),
        help="Target-edge SHAP summary with V_INPUT split into CH4, air, and water feeds.",
    )
    parser.add_argument(
        "--node-process-means",
        default=None,
        help="Optional cached process-level node-type means CSV for plot-only regeneration.",
    )
    parser.add_argument(
        "--feed-process-means",
        default=None,
        help="Optional cached process-level feed-type means CSV for plot-only regeneration.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/"
            "shap_node_type_mean_all_processes"
        ),
    )
    parser.add_argument(
        "--feed-only",
        action="store_true",
        help="Write only the feed-type figure and CSV files; preserve existing node-type outputs.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _display_type(unit_type: str) -> str:
    return DISPLAY_NODE_TYPE.get(unit_type, unit_type.replace("_", " "))


def build_aggregate(shap_summary: pd.DataFrame, canonical_nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return process-level type means and their process-balanced aggregate."""
    _require_columns(shap_summary, REQUIRED_SHAP_COLUMNS, "SHAP summary")
    _require_columns(canonical_nodes, REQUIRED_NODE_COLUMNS, "Canonical node table")

    shap = shap_summary[list(REQUIRED_SHAP_COLUMNS)].copy()
    nodes = canonical_nodes[list(REQUIRED_NODE_COLUMNS)].copy()
    shap["process_id"] = pd.to_numeric(shap["process_id"], errors="raise").astype(int)
    nodes["process_id"] = pd.to_numeric(nodes["process_id"], errors="raise").astype(int)
    shap["node_name"] = shap["node_name"].astype(str)
    nodes["node_name"] = nodes["node_name"].astype(str)
    shap["mean_signed_relative_shap"] = pd.to_numeric(
        shap["mean_signed_relative_shap"], errors="raise"
    )

    if shap.duplicated(["process_id", "node_name"]).any():
        raise ValueError("SHAP summary must contain exactly one row per process/node.")
    if nodes.duplicated(["process_id", "node_name"]).any():
        raise ValueError("Canonical node table has duplicate process/node rows.")

    merged = shap.merge(
        nodes,
        on=["process_id", "node_name"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    unmatched = merged.loc[merged["_merge"].ne("both"), ["process_id", "node_name", "_merge"]]
    if not unmatched.empty:
        raise ValueError(
            "SHAP and canonical nodes do not match exactly: "
            f"{unmatched.to_dict(orient='records')}"
        )
    if merged["unit_type"].isna().any() or merged["unit_type"].astype(str).str.strip().eq("").any():
        raise ValueError("Every SHAP node must have a non-empty unit_type.")

    process_means = (
        merged.groupby(["unit_type", "process_id"], as_index=False, observed=True)
        .agg(
            mean_signed_relative_shap=("mean_signed_relative_shap", "mean"),
            node_instance_count=("node_name", "size"),
        )
        .sort_values(["unit_type", "process_id"])
        .reset_index(drop=True)
    )
    aggregate = (
        process_means.groupby("unit_type", as_index=False, observed=True)
        .agg(
            mean_signed_relative_shap=("mean_signed_relative_shap", "mean"),
            signed_relative_shap_std_across_processes=("mean_signed_relative_shap", "std"),
            process_count=("process_id", "nunique"),
            node_instance_count=("node_instance_count", "sum"),
        )
        .fillna({"signed_relative_shap_std_across_processes": 0.0})
    )
    aggregate["node_type_label"] = aggregate["unit_type"].map(_display_type)
    aggregate = aggregate.sort_values("mean_signed_relative_shap", kind="stable").reset_index(drop=True)
    return process_means, aggregate


def build_feed_aggregate(feed_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return feed means per process and one P01-P10-balanced aggregate.

    A structurally absent feed has exactly zero SHAP and is omitted by the
    feed-split overlay source.  It remains excluded from that feed's mean;
    each feed is averaged only over processes where it physically exists.
    """
    _require_columns(feed_summary, REQUIRED_FEED_COLUMNS, "Feed SHAP summary")
    feed = feed_summary[list(REQUIRED_FEED_COLUMNS)].copy()
    feed["process_id"] = pd.to_numeric(feed["process_id"], errors="raise").astype(int)
    feed["node_name"] = feed["node_name"].astype(str)
    feed["mean_signed_relative_shap"] = pd.to_numeric(
        feed["mean_signed_relative_shap"], errors="raise"
    )
    feed = feed.loc[feed["node_name"].isin(FEED_NODE_ORDER)].copy()
    if feed.empty:
        raise ValueError("Feed SHAP summary contains none of the expected feed nodes.")

    process_ids = tuple(sorted(feed["process_id"].unique().tolist()))
    expected_process_ids = tuple(range(1, 11))
    if process_ids != expected_process_ids:
        raise ValueError(
            f"Feed SHAP summary must contain P01-P10; found process IDs {process_ids}."
        )
    observed = (
        feed.groupby(["process_id", "node_name"], as_index=False, observed=True)
        .agg(
            mean_signed_relative_shap=("mean_signed_relative_shap", "mean"),
            target_edge_count=("target_edge_id", "nunique"),
        )
        .rename(columns={"node_name": "feed_node"})
    )
    complete_index = pd.MultiIndex.from_product(
        [expected_process_ids, FEED_NODE_ORDER], names=["process_id", "feed_node"]
    )
    process_means = (
        observed.set_index(["process_id", "feed_node"])
        .reindex(complete_index)
        .reset_index()
    )
    process_means["structurally_absent"] = process_means["mean_signed_relative_shap"].isna()
    process_means["mean_signed_relative_shap"] = process_means[
        "mean_signed_relative_shap"
    ].fillna(0.0)
    process_means["target_edge_count"] = process_means["target_edge_count"].fillna(0).astype(int)
    process_means["feed_type_label"] = process_means["feed_node"].map(DISPLAY_FEED_TYPE)

    active_process_means = process_means.loc[
        ~process_means["structurally_absent"]
    ].copy()
    aggregate = (
        active_process_means.groupby("feed_node", as_index=False, observed=True)
        .agg(
            mean_signed_relative_shap=("mean_signed_relative_shap", "mean"),
            signed_relative_shap_std_across_processes=("mean_signed_relative_shap", "std"),
            process_count=("process_id", "nunique"),
            target_edge_count_total=("target_edge_count", "sum"),
        )
        .sort_values("feed_node", key=lambda values: values.map({name: i for i, name in enumerate(FEED_NODE_ORDER)}))
        .reset_index(drop=True)
    )
    absence_counts = (
        process_means.groupby("feed_node", as_index=False, observed=True)["structurally_absent"]
        .sum()
        .rename(columns={"structurally_absent": "structurally_absent_process_count"})
    )
    aggregate = aggregate.merge(absence_counts, on="feed_node", how="left", validate="one_to_one")
    aggregate["feed_type_label"] = aggregate["feed_node"].map(DISPLAY_FEED_TYPE)
    return process_means, aggregate


def build_node_aggregate_from_process_means(process_means: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the node-type aggregate from an already saved process-means CSV."""
    required = {"unit_type", "process_id", "mean_signed_relative_shap", "node_instance_count"}
    _require_columns(process_means, required, "Cached node process means")
    frame = process_means[list(required)].copy()
    frame["process_id"] = pd.to_numeric(frame["process_id"], errors="raise").astype(int)
    frame["mean_signed_relative_shap"] = pd.to_numeric(
        frame["mean_signed_relative_shap"], errors="raise"
    )
    frame["node_instance_count"] = pd.to_numeric(frame["node_instance_count"], errors="raise").astype(int)
    if frame.duplicated(["unit_type", "process_id"]).any():
        raise ValueError("Cached node process means has duplicate unit_type/process rows.")
    aggregate = (
        frame.groupby("unit_type", as_index=False, observed=True)
        .agg(
            mean_signed_relative_shap=("mean_signed_relative_shap", "mean"),
            signed_relative_shap_std_across_processes=("mean_signed_relative_shap", "std"),
            process_count=("process_id", "nunique"),
            node_instance_count=("node_instance_count", "sum"),
        )
        .fillna({"signed_relative_shap_std_across_processes": 0.0})
    )
    aggregate["node_type_label"] = aggregate["unit_type"].map(_display_type)
    aggregate = aggregate.sort_values("mean_signed_relative_shap", kind="stable").reset_index(drop=True)
    return frame, aggregate


def build_feed_aggregate_from_process_means(process_means: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the feed-type aggregate from an already saved process-means CSV."""
    required = {
        "process_id",
        "feed_node",
        "mean_signed_relative_shap",
        "target_edge_count",
        "structurally_absent",
    }
    _require_columns(process_means, required, "Cached feed process means")
    frame = process_means.copy()
    frame["process_id"] = pd.to_numeric(frame["process_id"], errors="raise").astype(int)
    frame["feed_node"] = frame["feed_node"].astype(str)
    frame["mean_signed_relative_shap"] = pd.to_numeric(
        frame["mean_signed_relative_shap"], errors="raise"
    )
    frame["target_edge_count"] = pd.to_numeric(frame["target_edge_count"], errors="raise").astype(int)
    frame["structurally_absent"] = (
        frame["structurally_absent"].astype(str).str.strip().str.lower().eq("true")
    )
    expected = pd.MultiIndex.from_product(
        [range(1, 11), FEED_NODE_ORDER], names=["process_id", "feed_node"]
    )
    observed = pd.MultiIndex.from_frame(frame[["process_id", "feed_node"]])
    if set(observed) != set(expected) or frame.duplicated(["process_id", "feed_node"]).any():
        raise ValueError("Cached feed process means must contain exactly P01-P10 x three feed types.")
    active = frame.loc[~frame["structurally_absent"]].copy()
    aggregate = (
        active.groupby("feed_node", as_index=False, observed=True)
        .agg(
            mean_signed_relative_shap=("mean_signed_relative_shap", "mean"),
            signed_relative_shap_std_across_processes=("mean_signed_relative_shap", "std"),
            process_count=("process_id", "nunique"),
            target_edge_count_total=("target_edge_count", "sum"),
        )
        .sort_values("feed_node", key=lambda values: values.map({name: i for i, name in enumerate(FEED_NODE_ORDER)}))
        .reset_index(drop=True)
    )
    absence_counts = (
        frame.groupby("feed_node", as_index=False, observed=True)["structurally_absent"]
        .sum()
        .rename(columns={"structurally_absent": "structurally_absent_process_count"})
    )
    aggregate = aggregate.merge(absence_counts, on="feed_node", how="left", validate="one_to_one")
    aggregate["feed_type_label"] = aggregate["feed_node"].map(DISPLAY_FEED_TYPE)
    return frame, aggregate


def plot_aggregate(aggregate: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    values = aggregate["mean_signed_relative_shap"].to_numpy(float)
    labels = aggregate["node_type_label"].astype(str).tolist()
    colors = np.where(values < 0.0, NEGATIVE_BLUE, POSITIVE_RED)
    # Both companion panels intentionally share a physical canvas and font
    # scale, so they remain visually matched when placed side by side.
    fig, axis = plt.subplots(figsize=SHAP_FIGSIZE, facecolor="white")
    y = np.arange(len(aggregate))
    axis.barh(y, values, height=0.64, color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=1.0, zorder=3)
    axis.axvline(0.0, color=ZERO_LINE_COLOR, linewidth=1.35, zorder=1)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.82)
    axis.set_axisbelow(True)
    axis.set_yticks(y, labels=labels, fontsize=TICK_FONT_SIZE)
    axis.tick_params(axis="x", labelsize=TICK_FONT_SIZE, width=1.1, length=6, colors="#303640")
    axis.tick_params(axis="y", length=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#8A949F")
    axis.spines["bottom"].set_color("#8A949F")

    limit = max(float(np.max(np.abs(values))) * 1.16, 0.001)
    limit = float(np.ceil(limit * 1000.0) / 1000.0)
    axis.set_xlim(-limit, limit)
    axis.set_xlabel("Mean SHAP Value", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=12)
    axis.margins(y=0.045)
    fig.subplots_adjust(left=0.35, right=0.97, top=0.95, bottom=0.18)
    stem = output_dir / "mean_signed_relative_shap_by_node_type_all_processes"
    for suffix, kwargs in ((".png", {"dpi": dpi}), (".pdf", {}), (".svg", {})):
        fig.savefig(stem.with_suffix(suffix), facecolor="white", bbox_inches="tight", **kwargs)
    plt.close(fig)
    return stem


def plot_feed_aggregate(aggregate: pd.DataFrame, output_dir: Path, dpi: int) -> Path:
    values = aggregate["mean_signed_relative_shap"].to_numpy(float)
    labels = aggregate["feed_type_label"].astype(str).tolist()
    colors = np.where(values < 0.0, NEGATIVE_BLUE, POSITIVE_RED)
    fig, axis = plt.subplots(figsize=SHAP_FIGSIZE, facecolor="white")
    y = np.arange(len(aggregate))
    axis.barh(y, values, height=0.58, color=colors, edgecolor=BAR_EDGE_COLOR, linewidth=1.0, zorder=3)
    axis.axvline(0.0, color=ZERO_LINE_COLOR, linewidth=1.35, zorder=1)
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.82)
    axis.set_axisbelow(True)
    axis.set_yticks(y, labels=labels, fontsize=TICK_FONT_SIZE)
    axis.tick_params(axis="x", labelsize=TICK_FONT_SIZE, width=1.1, length=6, colors="#303640")
    axis.tick_params(axis="y", length=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#8A949F")
    axis.spines["bottom"].set_color("#8A949F")

    limit = max(float(np.max(np.abs(values))) * 1.25, 0.001)
    limit = float(np.ceil(limit * 1000.0) / 1000.0)
    axis.set_xlim(-limit, limit)
    axis.set_xticks(np.linspace(-limit, limit, 5))
    axis.set_xlabel("Mean SHAP Value", fontsize=AXIS_LABEL_FONT_SIZE, labelpad=12)
    axis.legend(
        handles=[
            Patch(facecolor=NEGATIVE_BLUE, label="Negative mean SHAP"),
            Patch(facecolor=POSITIVE_RED, label="Positive mean SHAP"),
        ],
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#C8CDD3",
        fontsize=14,
        handlelength=2.4,
        handleheight=0.95,
        handletextpad=0.65,
        labelspacing=0.55,
        borderpad=0.55,
        borderaxespad=0.70,
        framealpha=0.96,
    )
    axis.margins(y=0.12)
    fig.subplots_adjust(left=0.25, right=0.97, top=0.95, bottom=0.18)
    stem = output_dir / "mean_signed_relative_shap_by_feed_type_existing_processes"
    for suffix, kwargs in ((".png", {"dpi": dpi}), (".pdf", {}), (".svg", {})):
        fig.savefig(stem.with_suffix(suffix), facecolor="white", bbox_inches="tight", **kwargs)
    plt.close(fig)
    return stem


def main() -> None:
    args = parse_args()
    shap_path = Path(args.shap_summary)
    nodes_path = Path(args.canonical_nodes)
    feed_path = Path(args.feed_summary)
    node_process_means_path = Path(args.node_process_means) if args.node_process_means else None
    feed_process_means_path = Path(args.feed_process_means) if args.feed_process_means else None
    output_dir = Path(args.output_dir)
    if not args.feed_only:
        if node_process_means_path is None:
            if not shap_path.is_file():
                raise FileNotFoundError(shap_path)
            if not nodes_path.is_file():
                raise FileNotFoundError(nodes_path)
        elif not node_process_means_path.is_file():
            raise FileNotFoundError(node_process_means_path)
    if feed_process_means_path is None and not feed_path.is_file():
        raise FileNotFoundError(feed_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.feed_only:
        if node_process_means_path is not None:
            process_means, aggregate = build_node_aggregate_from_process_means(
                pd.read_csv(node_process_means_path)
            )
            node_source = node_process_means_path
        else:
            process_means, aggregate = build_aggregate(
                pd.read_csv(shap_path), pd.read_csv(nodes_path)
            )
            node_source = shap_path
        process_means.to_csv(output_dir / "shap_node_type_process_means.csv", index=False)
        aggregate.to_csv(output_dir / "shap_node_type_mean_all_processes.csv", index=False)
        node_stem = plot_aggregate(aggregate, output_dir, args.dpi)
        print(f"source={node_source}")
        print(f"node_types={len(aggregate)}; processes=P01-P10")
        print(f"wrote={node_stem}")
    if feed_process_means_path is not None:
        feed_process_means, feed_aggregate = build_feed_aggregate_from_process_means(
            pd.read_csv(feed_process_means_path)
        )
        feed_source = feed_process_means_path
    else:
        feed_process_means, feed_aggregate = build_feed_aggregate(pd.read_csv(feed_path))
        feed_source = feed_path
    feed_process_means.to_csv(output_dir / "shap_feed_type_process_means.csv", index=False)
    feed_aggregate.to_csv(output_dir / "shap_feed_type_mean_all_processes.csv", index=False)
    feed_stem = plot_feed_aggregate(feed_aggregate, output_dir, args.dpi)
    print(f"feed_source={feed_source}")
    print(f"feed_types={len(feed_aggregate)}; wrote={feed_stem}")


if __name__ == "__main__":
    main()
