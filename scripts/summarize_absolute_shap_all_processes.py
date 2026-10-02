#!/usr/bin/env python3
"""Create process-balanced absolute-SHAP summaries for P01--P10.

Every explanation is normalized before it is pooled because the ten predicted
properties use incompatible units.  For node and feed figures, raw SHAP values
are summed within a node first, then ``abs`` is taken and divided by the total
node attribution of that explanation.  For operating variables, ``abs`` is
taken at the raw feature level first (as requested), then values for the same
feature across nodes are summed and normalized.  Consequently all reported
values are mean absolute *relative* SHAP contributions, not signed values.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PROCESS_IDS = tuple(range(1, 11))
GROUP_KEYS = ["process_id", "sample_index", "target_edge_id", "target_property"]
REQUIRED_RAW_COLUMNS = set(GROUP_KEYS) | {"node_name", "feature_name", "shap_value"}
REQUIRED_NODE_COLUMNS = {"process_id", "node_name", "unit_type"}
FEED_FEATURE_GROUP = {
    "feed_ch4_flow": "FEED_CH4",
    "feed_air_flow": "FEED_AIR",
    "log_air_ch4_ratio": "FEED_AIR",
    "feed_water_flow": "FEED_WATER",
    "log_water_ch4_ratio": "FEED_WATER",
}
FEED_ORDER = ("FEED_CH4", "FEED_AIR", "FEED_WATER")
FEED_LABELS = {
    "FEED_CH4": r"CH$_4$ feed",
    "FEED_AIR": "Air feed",
    "FEED_WATER": "Water feed",
}
NODE_LABELS = {
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
FEATURE_LABELS = {
    "ch4": r"CH$_4$ operating flow",
    "air": "Air operating flow",
    "h2o": r"H$_2$O operating flow",
    "h2": r"H$_2$ operating flow",
    "co2": r"CO$_2$ operating flow",
    "split_ratio": "Split ratio 1",
    "split_ratio2": "Split ratio 2",
    "split_flow": "Split flow",
    "press": "Pressure",
    "temp": "Temperature",
    "cold_temp": "Cold-side temperature",
    "cold_hot_dt": "Hot--cold ΔT",
    "hot_temp": "Hot-side temperature",
    "feed_ch4_flow": r"CH$_4$ feed flow",
    "feed_air_flow": "Air feed flow",
    "feed_water_flow": r"H$_2$O feed flow",
    "log_air_ch4_ratio": r"log(Air/CH$_4$)",
    "log_water_ch4_ratio": r"log(H$_2$O/CH$_4$)",
}

BLUE = "#2166AC"
GRID = "#DCE2E9"
EDGE = "#FFFFFF"
PAPER_FIGURE_ASPECT = 8.4 / 5.2
FIGSIZE = (10.5, 10.5 / PAPER_FIGURE_ASPECT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shap-root", required=True, help="Root containing P01 ... P10 raw SHAP CSVs.")
    parser.add_argument("--canonical-nodes", default="data/reference/v3/canonical_nodes.csv")
    parser.add_argument("--checkpoint", default=None, help="Optional exact checkpoint path for provenance.")
    parser.add_argument("--config", default=None, help="Optional exact model configuration path for provenance.")
    parser.add_argument(
        "--exclude-operating-features",
        nargs="*",
        default=(),
        metavar="FEATURE",
        help=(
            "Operating-feature names to omit only from the operating-variable figure. "
            "The CSV and Excel tables retain the complete feature set."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dpi", type=int, default=900)
    return parser.parse_args()


def _read_raw(shap_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for process_id in PROCESS_IDS:
        path = shap_root / f"P{process_id:02d}" / "sample_level_shap.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing required raw SHAP file: {path}")
        header = pd.read_csv(path, nrows=0)
        missing = sorted(REQUIRED_RAW_COLUMNS.difference(header.columns))
        if missing:
            raise ValueError(f"{path} lacks columns: {missing}")
        frame = pd.read_csv(path, usecols=sorted(REQUIRED_RAW_COLUMNS))
        found = sorted(pd.to_numeric(frame["process_id"], errors="raise").unique().tolist())
        if found != [process_id]:
            raise ValueError(f"{path} contains process IDs {found}, expected [{process_id}]")
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    raw["process_id"] = pd.to_numeric(raw["process_id"], errors="raise").astype(int)
    raw["sample_index"] = pd.to_numeric(raw["sample_index"], errors="raise").astype(int)
    raw["shap_value"] = pd.to_numeric(raw["shap_value"], errors="raise").astype(float)
    raw["node_name"] = raw["node_name"].astype(str)
    raw["feature_name"] = raw["feature_name"].astype(str)
    return raw


def _relative_node_values(raw: pd.DataFrame, *, split_feeds: bool) -> pd.DataFrame:
    frame = raw.copy()
    if split_feeds:
        v_input = frame["node_name"].eq("V_INPUT")
        mapped = frame["feature_name"].map(FEED_FEATURE_GROUP)
        unmapped = v_input & mapped.isna()
        nonzero = frame.loc[unmapped, "shap_value"].abs().max()
        if pd.notna(nonzero) and float(nonzero) > 1.0e-10:
            names = sorted(frame.loc[unmapped, "feature_name"].unique().tolist())
            raise ValueError(f"V_INPUT has non-feed SHAP attribution: {names}")
        frame.loc[v_input & mapped.notna(), "node_name"] = mapped.loc[v_input & mapped.notna()]
        frame = frame.loc[~unmapped].copy()

    node = (
        frame.groupby(GROUP_KEYS + ["node_name"], as_index=False, observed=True)["shap_value"]
        .sum()
        .rename(columns={"shap_value": "node_shap"})
    )
    denominator = node.groupby(GROUP_KEYS, observed=True)["node_shap"].transform(
        lambda values: values.abs().sum()
    )
    node["relative_absolute_shap"] = np.where(
        denominator.gt(np.finfo(float).eps), node["node_shap"].abs() / denominator, 0.0
    )
    return node


def _node_type_tables(raw: pd.DataFrame, canonical_nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    node = _relative_node_values(raw, split_feeds=False)
    per_node = (
        node.groupby(["process_id", "node_name"], as_index=False, observed=True)
        .agg(mean_absolute_relative_shap=("relative_absolute_shap", "mean"))
    )
    nodes = canonical_nodes[["process_id", "node_name", "unit_type"]].copy()
    nodes["process_id"] = pd.to_numeric(nodes["process_id"], errors="raise").astype(int)
    nodes["node_name"] = nodes["node_name"].astype(str)
    if nodes.duplicated(["process_id", "node_name"]).any():
        raise ValueError("Canonical node table has duplicate process/node rows.")
    merged = per_node.merge(nodes, on=["process_id", "node_name"], how="outer", indicator=True)
    unmatched = merged.loc[merged["_merge"].ne("both"), ["process_id", "node_name", "_merge"]]
    if not unmatched.empty:
        raise ValueError(f"Raw SHAP and canonical nodes do not match: {unmatched.to_dict(orient='records')}")
    process = (
        merged.groupby(["process_id", "unit_type"], as_index=False, observed=True)
        .agg(
            mean_absolute_relative_shap=("mean_absolute_relative_shap", "mean"),
            node_instance_count=("node_name", "size"),
        )
        .sort_values(["unit_type", "process_id"])
    )
    aggregate = (
        process.groupby("unit_type", as_index=False, observed=True)
        .agg(
            mean_absolute_relative_shap=("mean_absolute_relative_shap", "mean"),
            std_across_processes=("mean_absolute_relative_shap", "std"),
            process_count=("process_id", "nunique"),
            node_instance_count=("node_instance_count", "sum"),
        )
        .fillna({"std_across_processes": 0.0})
    )
    aggregate["node_type_label"] = aggregate["unit_type"].map(NODE_LABELS).fillna(aggregate["unit_type"])
    return process.reset_index(drop=True), aggregate.sort_values("mean_absolute_relative_shap").reset_index(drop=True)


def _feed_tables(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    node = _relative_node_values(raw, split_feeds=True)
    observed = node.loc[node["node_name"].isin(FEED_ORDER)].groupby(
        ["process_id", "node_name"], as_index=False, observed=True
    ).agg(
        mean_absolute_relative_shap=("relative_absolute_shap", "mean"),
        explanation_count=("relative_absolute_shap", "size"),
        max_absolute_relative_shap=("relative_absolute_shap", "max"),
    ).rename(columns={"node_name": "feed_node"})
    complete = pd.MultiIndex.from_product([PROCESS_IDS, FEED_ORDER], names=["process_id", "feed_node"])
    process = observed.set_index(["process_id", "feed_node"]).reindex(complete).reset_index()
    # Missing or exactly-zero feed attribution denotes a structurally unavailable feed;
    # it is shown in the process table but excluded from that feed's cross-process mean.
    process["structurally_absent"] = process["mean_absolute_relative_shap"].isna() | process["max_absolute_relative_shap"].fillna(0.0).le(1.0e-12)
    process["mean_absolute_relative_shap"] = process["mean_absolute_relative_shap"].fillna(0.0)
    process["explanation_count"] = process["explanation_count"].fillna(0).astype(int)
    process["feed_type_label"] = process["feed_node"].map(FEED_LABELS)
    active = process.loc[~process["structurally_absent"]].copy()
    aggregate = active.groupby("feed_node", as_index=False, observed=True).agg(
        mean_absolute_relative_shap=("mean_absolute_relative_shap", "mean"),
        std_across_processes=("mean_absolute_relative_shap", "std"),
        process_count=("process_id", "nunique"),
        explanation_count_total=("explanation_count", "sum"),
    ).fillna({"std_across_processes": 0.0})
    missing_feed_rows = pd.DataFrame({"feed_node": FEED_ORDER}).merge(aggregate, on="feed_node", how="left")
    if missing_feed_rows["mean_absolute_relative_shap"].isna().any():
        absent = missing_feed_rows.loc[missing_feed_rows["mean_absolute_relative_shap"].isna(), "feed_node"].tolist()
        raise ValueError(f"No active process has these feed types: {absent}")
    absence = process.groupby("feed_node", as_index=False, observed=True)["structurally_absent"].sum().rename(
        columns={"structurally_absent": "structurally_absent_process_count"}
    )
    aggregate = missing_feed_rows.merge(absence, on="feed_node", how="left")
    aggregate["feed_type_label"] = aggregate["feed_node"].map(FEED_LABELS)
    aggregate["_order"] = aggregate["feed_node"].map({name: i for i, name in enumerate(FEED_ORDER)})
    return process, aggregate.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _operating_variable_tables(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Absolute value is deliberately applied *before* averaging and before
    # grouping identical variables from separate unit nodes.
    feature = raw.assign(absolute_shap=raw["shap_value"].abs()).groupby(
        GROUP_KEYS + ["feature_name"], as_index=False, observed=True
    ).agg(absolute_shap=("absolute_shap", "sum"))
    denominator = feature.groupby(GROUP_KEYS, observed=True)["absolute_shap"].transform("sum")
    feature["relative_absolute_shap"] = np.where(
        denominator.gt(np.finfo(float).eps), feature["absolute_shap"] / denominator, 0.0
    )
    process = feature.groupby(["process_id", "feature_name"], as_index=False, observed=True).agg(
        mean_absolute_relative_shap=("relative_absolute_shap", "mean"),
        explanation_count=("relative_absolute_shap", "size"),
    )
    aggregate = process.groupby("feature_name", as_index=False, observed=True).agg(
        mean_absolute_relative_shap=("mean_absolute_relative_shap", "mean"),
        std_across_processes=("mean_absolute_relative_shap", "std"),
        process_count=("process_id", "nunique"),
        explanation_count_total=("explanation_count", "sum"),
    ).fillna({"std_across_processes": 0.0})
    aggregate["operating_variable_label"] = aggregate["feature_name"].map(FEATURE_LABELS).fillna(aggregate["feature_name"])
    process["operating_variable_label"] = process["feature_name"].map(FEATURE_LABELS).fillna(process["feature_name"])
    return process.sort_values(["process_id", "feature_name"]).reset_index(drop=True), aggregate.sort_values(
        "mean_absolute_relative_shap"
    ).reset_index(drop=True)


def _plot_bars(
    table: pd.DataFrame,
    *,
    label_column: str,
    stem: Path,
    dpi: int,
    subtitle: str,
) -> None:
    values = table["mean_absolute_relative_shap"].to_numpy(float)
    labels = table[label_column].astype(str).tolist()
    count = len(table)
    height = max(FIGSIZE[1], 0.38 * count + 1.2)
    fig, axis = plt.subplots(figsize=(FIGSIZE[0], height), facecolor="white")
    y = np.arange(count)
    axis.barh(y, values, height=0.63, color=BLUE, edgecolor=EDGE, linewidth=1.05, zorder=3)
    axis.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.9)
    axis.set_axisbelow(True)
    axis.set_yticks(y, labels=labels, fontsize=15)
    axis.tick_params(axis="x", labelsize=15, width=1.1, length=6, colors="#303640")
    axis.tick_params(axis="y", length=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#8A949F")
    axis.spines["bottom"].set_color("#8A949F")
    maximum = max(float(values.max(initial=0.0)), 1.0e-6)
    # Reserve a true blank 42% of the *interior* x-range for the legend.
    # The largest bar ends at <= 58.2%, so it cannot overlap the legend.
    axis.set_xlim(0.0, maximum * 1.72)
    axis.set_xlabel("Mean |SHAP Value|", fontsize=18, labelpad=12)
    axis.legend(
        handles=[Patch(facecolor=BLUE, edgecolor=EDGE, label="Mean |SHAP|")],
        loc="upper left", bbox_to_anchor=(0.625, 0.98), frameon=True,
        facecolor="white", edgecolor="#C8CDD3", framealpha=0.98,
        fontsize=14, handlelength=2.2, borderpad=0.5, borderaxespad=0.25,
    )
    axis.margins(y=0.055)
    fig.subplots_adjust(left=0.36 if count > 8 else 0.25, right=0.97, top=0.965, bottom=0.25)
    fig.text(
        0.5, 0.045, subtitle, ha="center", va="bottom",
        fontsize=19, fontweight="bold", color="#1F2937",
    )
    for suffix, kwargs in ((".png", {"dpi": dpi}), (".pdf", {}), (".svg", {})):
        fig.savefig(stem.with_suffix(suffix), facecolor="white", bbox_inches="tight", **kwargs)
    plt.close(fig)


def _write_readme(
    output_dir: Path,
    *,
    shap_root: Path,
    raw: pd.DataFrame,
    checkpoint: str | None,
    config: str | None,
    excluded_operating_features: tuple[str, ...],
) -> None:
    counts = raw.groupby("process_id", observed=True).size().to_dict()
    text = f"""# Absolute-SHAP summary (P01--P10)

All reported values are **mean absolute relative SHAP** contributions.

- Source: `{shap_root}` (`sample_level_shap.csv` for P01--P10).
- Checkpoint: `{checkpoint or "not supplied"}`.
- Model configuration: `{config or "not supplied"}`.
- Node type: raw feature SHAP is summed within each node; the node sum is made absolute and normalized by the sum of absolute node sums within the same sample/target-edge/property explanation. Node instances are averaged within each process/type, then process means are averaged equally across P01--P10.
- Feed type: `V_INPUT` contributions are split into CH4, air, and water groups before the same node-wise absolute normalization. A structurally unavailable feed is excluded from that feed's cross-process mean; the process table records these exclusions.
- Operating variable: raw SHAP is made absolute before aggregation, matching the requested `mean(abs(SHAP))` convention. Values for the same feature across nodes (and the direct feed head where present) are summed, normalized within each explanation, averaged within process, then averaged equally across P01--P10.
- Operating-variable figure exclusions: `{", ".join(excluded_operating_features) if excluded_operating_features else "none"}`. These exclusions apply **only** to the figure; the CSV and Excel tables retain all operating features.
- The figures reserve right-side in-axis space for the legend; bars end before the legend begins.
- Below-axis panel subtitles: (a) unit type, (c) feed type, and (b) operating variable.

Raw-row count by process: `{json.dumps(counts, sort_keys=True)}`.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def _excel(output: Path, sheets: Iterable[tuple[str, pd.DataFrame]], metadata: pd.DataFrame) -> None:
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, frame in sheets:
            frame.to_excel(writer, sheet_name=name[:31], index=False)
        metadata.to_excel(writer, sheet_name="method_and_provenance", index=False)
        for worksheet in writer.sheets.values():
            worksheet.freeze_panes = "A2"
            for column in worksheet.columns:
                letter = column[0].column_letter
                worksheet.column_dimensions[letter].width = min(
                    max(len(str(cell.value or "")) for cell in column) + 2, 42
                )


def main() -> None:
    args = parse_args()
    shap_root = Path(args.shap_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    nodes_path = Path(args.canonical_nodes).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = _read_raw(shap_root)
    canonical_nodes = pd.read_csv(nodes_path)
    missing = sorted(REQUIRED_NODE_COLUMNS.difference(canonical_nodes.columns))
    if missing:
        raise ValueError(f"{nodes_path} lacks columns: {missing}")
    node_process, node_aggregate = _node_type_tables(raw, canonical_nodes)
    feed_process, feed_aggregate = _feed_tables(raw)
    operating_process, operating_aggregate = _operating_variable_tables(raw)
    excluded_operating_features = tuple(dict.fromkeys(args.exclude_operating_features))
    unknown_exclusions = sorted(set(excluded_operating_features).difference(operating_aggregate["feature_name"]))
    if unknown_exclusions:
        raise ValueError(
            "Unknown --exclude-operating-features value(s): "
            f"{unknown_exclusions}. Available: {operating_aggregate['feature_name'].tolist()}"
        )
    operating_plot = operating_aggregate.loc[
        ~operating_aggregate["feature_name"].isin(excluded_operating_features)
    ].copy()
    if operating_plot.empty:
        raise ValueError("All operating features were excluded; no operating-variable figure can be created.")

    tables = {
        "absolute_shap_node_type_process_means.csv": node_process,
        "absolute_shap_node_type_mean_all_processes.csv": node_aggregate,
        "absolute_shap_feed_type_process_means.csv": feed_process,
        "absolute_shap_feed_type_mean_all_processes.csv": feed_aggregate,
        "absolute_shap_operating_variable_process_means.csv": operating_process,
        "absolute_shap_operating_variable_mean_all_processes.csv": operating_aggregate,
    }
    for name, frame in tables.items():
        frame.to_csv(output_dir / name, index=False)
    _plot_bars(
        node_aggregate,
        label_column="node_type_label",
        stem=output_dir / "mean_absolute_relative_shap_by_node_type_all_processes",
        dpi=args.dpi,
        subtitle="(a) Mean Absolute SHAP by Unit Type",
    )
    _plot_bars(
        feed_aggregate,
        label_column="feed_type_label",
        stem=output_dir / "mean_absolute_relative_shap_by_feed_type_existing_processes",
        dpi=args.dpi,
        subtitle="(c) Mean Absolute SHAP by Feed Type",
    )
    _plot_bars(
        operating_plot,
        label_column="operating_variable_label",
        stem=output_dir / "mean_absolute_relative_shap_by_operating_variable_all_processes",
        dpi=args.dpi,
        subtitle="(b) Mean Absolute SHAP by Operating Variable",
    )
    metadata = pd.DataFrame([
        {"item": "metric", "value": "Mean absolute relative SHAP value"},
        {"item": "absolute_value_convention", "value": "Node/feed: abs(sum feature SHAP within node); operating variables: sum(abs(feature SHAP))"},
        {"item": "normalization", "value": "Within each process/sample/target-edge/target-property explanation"},
        {"item": "cross_process_aggregation", "value": "Equal mean of process means across P01-P10"},
        {"item": "feed_absence_policy", "value": "Structurally absent (zero-attribution) feed processes excluded from that feed mean"},
        {"item": "operating_figure_exclusions", "value": ", ".join(excluded_operating_features) or "none (complete feature set plotted)"},
        {"item": "raw_shap_root", "value": str(shap_root)},
        {"item": "canonical_nodes", "value": str(nodes_path)},
        {"item": "checkpoint", "value": args.checkpoint or "not supplied"},
        {"item": "model_config", "value": args.config or "not supplied"},
    ])
    _excel(output_dir / "absolute_shap_summary_tables.xlsx", [
        ("node_type_all_processes", node_aggregate),
        ("node_type_by_process", node_process),
        ("feed_type_all_processes", feed_aggregate),
        ("feed_type_by_process", feed_process),
        ("operating_variable_all", operating_aggregate),
        ("operating_variable_by_process", operating_process),
    ], metadata)
    _write_readme(
        output_dir,
        shap_root=shap_root,
        raw=raw,
        checkpoint=args.checkpoint,
        config=args.config,
        excluded_operating_features=excluded_operating_features,
    )
    print(f"raw_rows={len(raw):,}; processes=P01-P10")
    print(f"wrote={output_dir}")


if __name__ == "__main__":
    main()
