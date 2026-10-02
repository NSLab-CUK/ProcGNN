#!/usr/bin/env python3
"""Build audited SHAP summaries for the 27 process/edge/material targets.

The raw SHAP runner explains one scalar ``target_edge x target_property`` at a
time.  This script keeps that property identity instead of averaging the ten
model outputs together.  Raw feature SHAP values are summed within each node,
normalised within each sample explanation, and then averaged across samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SPECIES_COUNTS = {"H2": 10, "CO2": 10, "H2O": 7}
TARGET_COLUMNS = {
    "process_id",
    "target_edge_id",
    "species",
    "target_property",
    "output_label",
}
RAW_COLUMNS = {
    "process_id",
    "sample_index",
    "target_edge_id",
    "target_property",
    "node_name",
    "feature_name",
    "shap_value",
}
GROUP_KEYS = [
    "process_id",
    "sample_index",
    "target_edge_id",
    "species",
    "target_property",
]
FEED_FEATURE_GROUP = {
    "feed_ch4_flow": "FEED_CH4",
    "feed_air_flow": "FEED_AIR",
    "log_air_ch4_ratio": "FEED_AIR",
    "feed_water_flow": "FEED_WATER",
    "log_water_ch4_ratio": "FEED_WATER",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shap-root",
        action="append",
        required=True,
        help=(
            "Root containing P01...P10/sample_level_shap.csv. Repeat the option "
            "to add supplemental roots; the first matching source wins."
        ),
    )
    parser.add_argument(
        "--target-map",
        default="data/reference/v3/shap_target_material_edges.csv",
    )
    parser.add_argument(
        "--unit-coordinates",
        default="data/process_overall_img/unit_coordinates.csv",
    )
    parser.add_argument(
        "--feed-coordinates",
        default="data/process_overall_img/feed_coordinates.csv",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _read_target_map(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    _require_columns(frame, TARGET_COLUMNS, "target-material map")
    frame["process_id"] = pd.to_numeric(frame["process_id"], errors="raise").astype(int)
    for column in ("target_edge_id", "species", "target_property", "output_label"):
        frame[column] = frame[column].astype(str)

    counts = frame.groupby("species", observed=True).size().to_dict()
    if counts != EXPECTED_SPECIES_COUNTS:
        raise ValueError(
            f"Expected target counts {EXPECTED_SPECIES_COUNTS}, found {counts}."
        )
    if len(frame) != sum(EXPECTED_SPECIES_COUNTS.values()):
        raise ValueError(f"Expected 27 target-material rows, found {len(frame)}.")
    if frame.duplicated(["process_id", "species"]).any():
        duplicate = frame.loc[
            frame.duplicated(["process_id", "species"], keep=False),
            ["process_id", "species", "target_edge_id"],
        ]
        raise ValueError(
            "Each process may have at most one representative target per species: "
            f"{duplicate.to_dict(orient='records')}"
        )
    return frame


def _read_raw(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    _require_columns(header, RAW_COLUMNS, str(path))
    frame = pd.read_csv(
        path,
        usecols=sorted(RAW_COLUMNS),
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
    return frame


def _select_raw_targets(
    target_map: pd.DataFrame,
    shap_roots: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []
    cache: dict[Path, pd.DataFrame] = {}

    for target in target_map.itertuples(index=False):
        chosen: pd.DataFrame | None = None
        chosen_path: Path | None = None
        for root in shap_roots:
            path = root / f"P{int(target.process_id):02d}" / "sample_level_shap.csv"
            if not path.is_file():
                continue
            if path not in cache:
                cache[path] = _read_raw(path)
            frame = cache[path]
            match = frame.loc[
                frame["process_id"].eq(int(target.process_id))
                & frame["target_edge_id"].eq(str(target.target_edge_id))
                & frame["target_property"].eq(str(target.target_property))
            ].copy()
            if not match.empty:
                chosen = match
                chosen_path = path
                break

        if chosen is None or chosen_path is None:
            raise FileNotFoundError(
                "No raw SHAP rows for "
                f"P{int(target.process_id):02d}/{target.target_edge_id}/"
                f"{target.target_property} under {[str(root) for root in shap_roots]}"
            )

        chosen["species"] = str(target.species)
        chosen["output_label"] = str(target.output_label)
        selected_frames.append(chosen)
        coverage_rows.append(
            {
                "process_id": int(target.process_id),
                "target_edge_id": str(target.target_edge_id),
                "species": str(target.species),
                "target_property": str(target.target_property),
                "sample_count": int(chosen["sample_index"].nunique()),
                "raw_row_count": int(len(chosen)),
                "source_file": str(chosen_path),
            }
        )

    selected = pd.concat(selected_frames, ignore_index=True)
    coverage = pd.DataFrame(coverage_rows)
    return selected, coverage


def _split_feeds_and_drop_virtual_outputs(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    is_v_input = frame["node_name"].eq("V_INPUT")
    mapped_feed = frame["feature_name"].map(FEED_FEATURE_GROUP)
    unmapped = is_v_input & mapped_feed.isna()
    maximum_unmapped = frame.loc[unmapped, "shap_value"].abs().max()
    if pd.notna(maximum_unmapped) and float(maximum_unmapped) > 1.0e-12:
        names = sorted(frame.loc[unmapped, "feature_name"].astype(str).unique())
        raise ValueError(f"V_INPUT contains nonzero unmapped SHAP features: {names}")
    frame.loc[is_v_input & mapped_feed.notna(), "node_name"] = mapped_feed.loc[
        is_v_input & mapped_feed.notna()
    ]
    frame = frame.loc[~unmapped].copy()

    is_v_output = frame["node_name"].eq("V_OUTPUT")
    maximum_output = frame.loc[is_v_output, "shap_value"].abs().max()
    if pd.notna(maximum_output) and float(maximum_output) > 1.0e-12:
        raise ValueError(
            "V_OUTPUT has nonzero raw-feature SHAP and cannot be replaced by a "
            "structural target marker."
        )
    frame = frame.loc[~is_v_output].copy()

    feed_rows = frame["node_name"].isin(FEED_FEATURE_GROUP.values())
    feed_activity = (
        frame.loc[feed_rows]
        .groupby(
            ["process_id", "target_edge_id", "species", "target_property", "node_name"],
            observed=True,
        )["shap_value"]
        .transform(lambda values: float(values.abs().max()))
    )
    inactive_index = frame.loc[feed_rows].index[feed_activity.le(1.0e-12)]
    if len(inactive_index):
        frame = frame.drop(index=inactive_index)
    return frame


def _summarise(frame: pd.DataFrame) -> pd.DataFrame:
    node_explanation = (
        frame.groupby(GROUP_KEYS + ["node_name"], observed=True, as_index=False)[
            "shap_value"
        ]
        .sum()
        .rename(columns={"shap_value": "node_shap"})
    )
    denominator = node_explanation.groupby(GROUP_KEYS, observed=True)[
        "node_shap"
    ].transform(lambda values: values.abs().sum())
    valid = denominator.gt(np.finfo(float).eps)
    node_explanation["relative_signed_shap"] = np.where(
        valid, node_explanation["node_shap"] / denominator, 0.0
    )
    node_explanation["relative_absolute_shap"] = np.where(
        valid, node_explanation["node_shap"].abs() / denominator, 0.0
    )

    summary_keys = [
        "process_id",
        "target_edge_id",
        "species",
        "target_property",
        "node_name",
    ]
    summary = (
        node_explanation.groupby(summary_keys, observed=True, as_index=False)
        .agg(
            mean_signed_relative_shap=("relative_signed_shap", "mean"),
            mean_absolute_relative_shap=("relative_absolute_shap", "mean"),
            signed_relative_shap_std=("relative_signed_shap", "std"),
            explanation_count=("relative_signed_shap", "size"),
        )
        .sort_values(summary_keys)
        .reset_index(drop=True)
    )
    summary["signed_relative_shap_std"] = summary[
        "signed_relative_shap_std"
    ].fillna(0.0)
    return summary


def _attach_coordinates(
    summary: pd.DataFrame,
    unit_path: Path,
    feed_path: Path,
) -> pd.DataFrame:
    unit = pd.read_csv(unit_path)
    feed = pd.read_csv(feed_path)
    coordinate_columns = ["process_id", "node_name", "x", "y"]
    coordinates = pd.concat(
        [unit[coordinate_columns], feed[coordinate_columns]], ignore_index=True
    )
    coordinates["process_id"] = pd.to_numeric(
        coordinates["process_id"], errors="raise"
    ).astype(int)
    if coordinates.duplicated(["process_id", "node_name"]).any():
        raise ValueError("Coordinate tables contain duplicate process/node rows.")
    merged = summary.merge(
        coordinates,
        on=["process_id", "node_name"],
        how="left",
        validate="many_to_one",
    )
    missing = merged.loc[merged[["x", "y"]].isna().any(axis=1), ["process_id", "node_name"]]
    if not missing.empty:
        raise ValueError(
            f"SHAP nodes lack coordinates: {missing.drop_duplicates().to_dict(orient='records')}"
        )
    return merged


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_map_path = Path(args.target_map)
    shap_roots = [Path(root) for root in args.shap_root]
    target_map = _read_target_map(target_map_path)
    selected, coverage = _select_raw_targets(target_map, shap_roots)
    selected = _split_feeds_and_drop_virtual_outputs(selected)
    summary = _summarise(selected)
    summary = _attach_coordinates(
        summary,
        Path(args.unit_coordinates),
        Path(args.feed_coordinates),
    )

    combination_columns = ["process_id", "target_edge_id", "species", "target_property"]
    combinations = summary[combination_columns].drop_duplicates()
    if len(combinations) != 27:
        raise RuntimeError(f"Expected 27 summarised target combinations, found {len(combinations)}.")

    selected.to_csv(output_dir / "selected_sample_level_shap.csv", index=False)
    summary.to_csv(output_dir / "shap_target_material_node_summary.csv", index=False)
    coverage.to_csv(output_dir / "target_material_shap_coverage.csv", index=False)
    target_map.to_csv(output_dir / "target_material_map.csv", index=False)

    manifest = {
        "target_map": str(target_map_path),
        "shap_roots_in_priority_order": [str(root) for root in shap_roots],
        "target_combination_count": int(len(combinations)),
        "species_counts": {
            key: int(value)
            for key, value in combinations.groupby("species", observed=True).size().to_dict().items()
        },
        "aggregation": {
            "step_1": "select exactly one target property for each process/edge/material combination",
            "step_2": "sum raw feature SHAP values within each node and sample explanation",
            "step_3": "divide by the sum of absolute node SHAP values within that explanation",
            "step_4": "average signed and absolute relative SHAP across samples only",
            "cross_property_averaging": False,
        },
        "outputs": {
            "summary": "shap_target_material_node_summary.csv",
            "coverage": "target_material_shap_coverage.csv",
            "selected_raw": "selected_sample_level_shap.csv",
        },
    }
    (output_dir / "target_material_shap_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "target_combinations=27; species_counts="
        f"{manifest['species_counts']}; output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
