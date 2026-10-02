#!/usr/bin/env python3
"""Render readable, value-labelled SHAP overlays for all target edges.

This is a plot-only renderer.  It consumes the audited target-edge node SHAP
summary already produced for the publication flowsheets, so no model inference
or SHAP recomputation is performed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, patheffects
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from PIL import Image


# A fixed publication canvas is used for every target edge.  ``imshow`` keeps
# each native PFD aspect ratio, adding white padding instead of stretching a
# flowsheet when its source dimensions differ from the others.  The wider
# layout gives the process drawings more horizontal space.
FIGSIZE = (16.0, 8.0)
PANEL_LETTERS = tuple("abcdefghij")
SPECIES_ORDER = ("H2", "CO2", "H2O", "CH4", "CO", "O2", "N2")
SPECIES_MATHTEXT = {
    "H2": r"$\mathrm{H_2}$",
    "CO2": r"$\mathrm{CO_2}$",
    "H2O": r"$\mathrm{H_2O}$",
    "CH4": r"$\mathrm{CH_4}$",
    "CO": r"$\mathrm{CO}$",
    "O2": r"$\mathrm{O_2}$",
    "N2": r"$\mathrm{N_2}$",
}
REQUIRED_SUMMARY_COLUMNS = {
    "process_id",
    "target_edge_id",
    "node_name",
    "mean_signed_relative_shap",
    "mean_absolute_relative_shap",
    "signed_relative_shap_std",
    "explanation_count",
    "x",
    "y",
}
REQUIRED_TARGET_COLUMNS = {"process_id", "target_edge_id", "x", "y"}
REQUIRED_MATERIAL_COLUMNS = {
    "process_id",
    "target_edge_id",
    "species",
    "target_property",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/"
            "shap_signed_target_edges_publication_compact_header_no_outlet_label/"
            "shap_target_edge_node_summary.csv"
        ),
        help="Target-edge node SHAP summary from the established publication rendering.",
    )
    parser.add_argument(
        "--target-output-coordinates",
        default="data/process_overall_img/target_output_coordinates.csv",
    )
    parser.add_argument("--image-dir", default="data/process_overall_img")
    parser.add_argument(
        "--output-dir",
        default=(
            "outputs/0819final/Paper_Tables_Final3_figures/"
            "shap_signed_target_edges_enhanced_values"
        ),
    )
    parser.add_argument(
        "--target-edge-ids",
        default="",
        help="Optional comma-separated target-edge IDs, for example P01_E015,P01_E021.",
    )
    parser.add_argument(
        "--split-prod-targets",
        action="store_true",
        help=(
            "Write target edges whose verified output label is PROD to prod_target/ "
            "and all other target edges to other_target/, with panel letters restarted "
            "in each directory."
        ),
    )
    parser.add_argument(
        "--split-target-species",
        action="store_true",
        help=(
            "Group panels by target molecule, restart panel letters in each "
            "molecule directory, and use P0x - chemical-formula subtitles."
        ),
    )
    parser.add_argument(
        "--target-answer-edges",
        default="data/reference/v3/target_answer_edges.csv",
        help="Verified target-species mapping used with --split-target-species.",
    )
    parser.add_argument(
        "--target-material-map",
        default="",
        help=(
            "Optional explicit process/edge/species/property map. When supplied with "
            "--split-target-species, this preserves separate material explanations "
            "for a shared edge (for example P07 H2 and CO2)."
        ),
    )
    parser.add_argument(
        "--color-limit",
        type=float,
        default=None,
        help=(
            "Symmetric colour limit for signed SHAP values.  Smaller values increase "
            "colour contrast; larger values are clipped to the end colours."
        ),
    )
    parser.add_argument(
        "--subtitle-size",
        type=float,
        default=20.0,
        help="Top subtitle size in points.",
    )
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")


def _find_image(image_dir: Path, process_id: int) -> Path:
    matches = [
        path
        for path in image_dir.glob("*.png")
        if (match := re.match(r"^(\d+)", path.stem)) and int(match.group(1)) == process_id
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one flowsheet image for P{process_id:02d}; found {[str(item) for item in matches]}"
        )
    return matches[0]


def _short_signed(value: float) -> str:
    """Three-significant-figure signed text suitable for a node marker."""
    if not np.isfinite(value) or abs(value) < 5.0e-12:
        return "0"
    rendered = f"{abs(value):.3g}"
    if rendered.startswith("0."):
        rendered = rendered[1:]
    return ("+" if value > 0.0 else "−") + rendered


def _text_color(value: float, color_map: ScalarMappable) -> tuple[str, str]:
    red, green, blue, _ = color_map.to_rgba(value)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return ("#151515", "white") if luminance >= 0.58 else ("white", "#151515")


def _panel_letter(index: int) -> str:
    if index >= len(PANEL_LETTERS):
        raise ValueError(f"Only {len(PANEL_LETTERS)} panel letters are supported.")
    return PANEL_LETTERS[index]


def _species_from_target_edge(
    target_edge_id: str,
    output_label: str,
    target_answer_edges: pd.DataFrame,
) -> str:
    """Return the verified molecule associated with a displayed target edge."""
    # PROD can have several stored target quantities (notably P07).  These
    # panels intentionally explain the hydrogen product, per the figure spec.
    if str(output_label).strip().upper() == "PROD":
        return "H2"
    rows = target_answer_edges.loc[
        target_answer_edges["canonical_answer_edge_id"].astype(str).eq(target_edge_id)
    ]
    if rows.empty:
        raise ValueError(f"No target-species mapping found for {target_edge_id}.")
    species: set[str] = set()
    # ``CO`` and ``O2`` are substrings of ``CO2``.  Parse underscore-delimited
    # property tokens instead of doing a substring search, so a CO2 target is
    # never incorrectly classified as three separate species.
    molecule_pattern = re.compile(
        r"(?:^|_)(H2O|CO2|CH4|H2|CO|O2|N2)(?:_|$)", re.IGNORECASE
    )
    for target_column in rows["target_column"].astype(str):
        match = molecule_pattern.search(target_column)
        if match:
            species.add(match.group(1).upper())
    if len(species) != 1:
        raise ValueError(
            f"Expected one target molecule for non-PROD {target_edge_id}; found {sorted(species)}."
        )
    return next(iter(species))


def _species_subtitle(panel_letter: str, process_id: int, species: str) -> str:
    if species not in SPECIES_MATHTEXT:
        raise ValueError(f"No mathematical formula configured for {species!r}.")
    return f"({panel_letter}) P{process_id:02d} - {SPECIES_MATHTEXT[species]}"


def _draw_target_edge(
    *,
    process_id: int,
    target_edge_id: str,
    summary: pd.DataFrame,
    target_output: pd.Series,
    image_path: Path,
    output_dir: Path,
    norm: colors.Normalize,
    size_max: float,
    panel_letter: str,
    panel_subtitle: str | None,
    subtitle_size: float,
    dpi: int,
) -> None:
    image = Image.open(image_path).convert("RGB")
    values = summary["mean_signed_relative_shap"].to_numpy(float)
    importance = summary["mean_absolute_relative_shap"].to_numpy(float)
    # Semi-transparent markers retain the signed numerical label while allowing
    # the underlying PFD unit symbol to remain visible.
    sizes = 420.0 + 900.0 * np.sqrt(np.clip(importance / size_max, 0.0, 1.0))
    color_map = ScalarMappable(norm=norm, cmap="RdBu_r")
    color_map.set_array([])

    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    # The content axis is deliberately identical for all panels.  The source
    # image remains aspect-preserving and is centred in this image region.
    # Reserve a generous header band when publication-sized subtitles are used.
    axis = fig.add_axes([0.025, 0.035, 0.95, 0.82])
    axis.imshow(image, interpolation="lanczos", aspect="equal")
    axis.set_anchor("C")

    # A restrained, translucent halo separates nodes without masking PFD units.
    axis.scatter(
        summary["x"],
        summary["y"],
        s=sizes + 100.0,
        facecolors="white",
        edgecolors="white",
        linewidths=1.55,
        alpha=0.42,
        zorder=3,
    )
    axis.scatter(
        summary["x"],
        summary["y"],
        c=values,
        s=sizes,
        cmap="RdBu_r",
        norm=norm,
        alpha=0.58,
        edgecolors="#20242A",
        linewidths=1.15,
        zorder=4,
    )
    for row in summary.itertuples(index=False):
        value = float(row.mean_signed_relative_shap)
        text_color, stroke_color = _text_color(value, color_map)
        axis.text(
            float(row.x),
            float(row.y),
            _short_signed(value),
            ha="center",
            va="center",
            fontsize=7.9,
            fontweight="bold",
            color=text_color,
            zorder=5,
            path_effects=[patheffects.withStroke(linewidth=1.15, foreground=stroke_color)],
        )

    # A small gold star and thin ring identify the target outlet without
    # competing with the SHAP-node values or the underlying PFD artwork.
    target_x = float(target_output["x"])
    target_y = float(target_output["y"])
    axis.scatter(
        [target_x],
        [target_y],
        s=760.0,
        marker="o",
        facecolors="none",
        edgecolors="#A56800",
        linewidths=1.5,
        alpha=0.88,
        zorder=7,
    )
    axis.scatter(
        [target_x],
        [target_y],
        s=310.0,
        marker="*",
        c="#FFD34E",
        edgecolors="#3B2B00",
        linewidths=0.95,
        alpha=0.92,
        zorder=8,
    )

    width, height = image.size
    axis.set_xlim(-0.012 * width, 1.012 * width)
    axis.set_ylim(1.018 * height, -0.018 * height)
    axis.axis("off")

    panel_subtitle = panel_subtitle or f"({panel_letter}) P{process_id:02d} – Target Edge {target_edge_id}"
    fig.text(
        0.5,
        0.92,
        panel_subtitle,
        ha="center",
        va="center",
        fontsize=subtitle_size,
        fontweight="normal",
        color="#20252B",
    )

    stem = output_dir / f"shap_signed_flowsheet_{panel_letter}_{target_edge_id}"
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    target_path = Path(args.target_output_coordinates)
    target_answer_path = Path(args.target_answer_edges)
    target_material_path = Path(args.target_material_map) if args.target_material_map else None
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not target_path.is_file():
        raise FileNotFoundError(target_path)
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)
    if args.split_prod_targets and args.split_target_species:
        raise ValueError("Use only one of --split-prod-targets or --split-target-species.")
    if (
        args.split_target_species
        and target_material_path is None
        and not target_answer_path.is_file()
    ):
        raise FileNotFoundError(target_answer_path)
    if target_material_path is not None and not target_material_path.is_file():
        raise FileNotFoundError(target_material_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_path)
    target_coordinates = pd.read_csv(target_path)
    _require_columns(summary, REQUIRED_SUMMARY_COLUMNS, "SHAP summary")
    _require_columns(target_coordinates, REQUIRED_TARGET_COLUMNS, "Target-output coordinates")
    summary["process_id"] = pd.to_numeric(summary["process_id"], errors="raise").astype(int)
    summary["target_edge_id"] = summary["target_edge_id"].astype(str)
    target_coordinates["process_id"] = pd.to_numeric(
        target_coordinates["process_id"], errors="raise"
    ).astype(int)
    target_coordinates["target_edge_id"] = target_coordinates["target_edge_id"].astype(str)
    summary_identity = ["process_id", "target_edge_id"]
    material_summary = {"species", "target_property"}.issubset(summary.columns)
    if material_summary:
        summary_identity.extend(["species", "target_property"])
    if summary.duplicated(summary_identity + ["node_name"]).any():
        raise ValueError(
            "SHAP summary has duplicate rows for identity "
            f"{summary_identity + ['node_name']}."
        )
    if target_coordinates.duplicated("target_edge_id").any():
        raise ValueError("Target-output coordinates has duplicate target_edge_id rows.")
    target_answer_edges = pd.DataFrame()
    target_material_map = pd.DataFrame()
    if target_material_path is not None:
        target_material_map = pd.read_csv(target_material_path)
        _require_columns(
            target_material_map,
            REQUIRED_MATERIAL_COLUMNS,
            "Target-material map",
        )
        target_material_map["process_id"] = pd.to_numeric(
            target_material_map["process_id"], errors="raise"
        ).astype(int)
        for column in ("target_edge_id", "species", "target_property"):
            target_material_map[column] = target_material_map[column].astype(str)
        if target_material_map.duplicated(
            ["process_id", "target_edge_id", "species", "target_property"]
        ).any():
            raise ValueError("Target-material map contains duplicate target combinations.")
    elif args.split_target_species:
        target_answer_edges = pd.read_csv(target_answer_path)
        _require_columns(
            target_answer_edges,
            {"canonical_answer_edge_id", "target_column"},
            "Target-answer edge mapping",
        )

    requested = {item.strip() for item in args.target_edge_ids.split(",") if item.strip()}
    available = set(summary["target_edge_id"])
    unknown = requested.difference(available)
    if unknown:
        raise ValueError(f"Unknown --target-edge-ids: {sorted(unknown)}")
    if requested:
        summary = summary.loc[summary["target_edge_id"].isin(requested)].copy()
    rendered_ids = sorted(summary["target_edge_id"].unique())
    missing_targets = sorted(set(rendered_ids).difference(set(target_coordinates["target_edge_id"])))
    if missing_targets:
        raise ValueError(f"Missing target-output coordinates for: {missing_targets}")

    maximum = float(summary["mean_signed_relative_shap"].abs().max())
    if args.color_limit is not None:
        if args.color_limit <= 0.0:
            raise ValueError("--color-limit must be positive.")
        limit = float(args.color_limit)
    else:
        limit = max(np.ceil(maximum * 100.0) / 100.0, 0.01)
    norm = colors.SymLogNorm(linthresh=min(0.002, limit / 2.0), vmin=-limit, vmax=limit, base=10)
    size_max = max(float(summary["mean_absolute_relative_shap"].max()), 1.0e-12)

    label_by_target = target_coordinates.set_index("target_edge_id")["output_label"].astype(str)
    if target_material_path is not None:
        if not material_summary:
            raise ValueError(
                "An explicit --target-material-map requires species and target_property "
                "columns in the SHAP summary."
            )
        target_order = target_material_map[
            ["process_id", "target_edge_id", "species", "target_property"]
        ].copy()
        summary_combinations = summary[
            ["process_id", "target_edge_id", "species", "target_property"]
        ].drop_duplicates()
        comparison = target_order.merge(
            summary_combinations,
            on=["process_id", "target_edge_id", "species", "target_property"],
            how="outer",
            indicator=True,
        )
        mismatch = comparison.loc[comparison["_merge"].ne("both")]
        if not mismatch.empty:
            raise ValueError(
                "Target-material map and SHAP summary combinations differ: "
                f"{mismatch.to_dict(orient='records')}"
            )
        target_order = target_order.sort_values(
            ["process_id", "target_edge_id", "species", "target_property"]
        )
    else:
        target_order = (
            summary[["process_id", "target_edge_id"]]
            .drop_duplicates()
            .sort_values(["process_id", "target_edge_id"])
        )
    if args.split_target_species:
        if target_material_path is None:
            target_order = target_order.copy()
            target_order["species"] = target_order["target_edge_id"].map(
                lambda edge_id: _species_from_target_edge(
                    str(edge_id), label_by_target.loc[str(edge_id)], target_answer_edges
                )
            )
        grouped_orders = [
            (
                f"{species}_target_edges",
                target_order.loc[target_order["species"].eq(species)]
                .sort_values(["process_id", "target_edge_id"])
                .copy(),
            )
            for species in SPECIES_ORDER
            if target_order["species"].eq(species).any()
        ]
    elif args.split_prod_targets:
        prod_targets = set(
            label_by_target.loc[label_by_target.str.upper().eq("PROD")].index
        )
        grouped_orders = [
            ("prod_target", target_order.loc[target_order["target_edge_id"].isin(prod_targets)]),
            ("other_target", target_order.loc[~target_order["target_edge_id"].isin(prod_targets)]),
        ]
    else:
        grouped_orders = [("", target_order)]

    rendered_by_group: dict[str, list[str]] = {}
    for group_name, group_order in grouped_orders:
        group_output_dir = output_dir / group_name if group_name else output_dir
        group_output_dir.mkdir(parents=True, exist_ok=True)
        rendered: list[str] = []
        group_frames: list[pd.DataFrame] = []
        for panel_index, row in enumerate(group_order.itertuples(index=False)):
            process_id = int(row.process_id)
            target_edge_id = str(row.target_edge_id)
            selected = summary.loc[
                summary["target_edge_id"].eq(target_edge_id)
            ].copy()
            if material_summary:
                selected = selected.loc[
                    selected["species"].eq(str(row.species))
                    & selected["target_property"].eq(str(row.target_property))
                ].copy()
            if selected.empty:
                raise ValueError(
                    f"No SHAP rows for {target_edge_id}/"
                    f"{getattr(row, 'species', '')}/{getattr(row, 'target_property', '')}."
                )
            target_output = target_coordinates.loc[
                target_coordinates["target_edge_id"].eq(target_edge_id)
            ]
            if len(target_output) != 1 or int(target_output.iloc[0]["process_id"]) != process_id:
                raise ValueError(f"Target-output coordinate mismatch for {target_edge_id}.")
            species = str(row.species) if args.split_target_species else ""
            _draw_target_edge(
                process_id=process_id,
                target_edge_id=target_edge_id,
                summary=selected,
                target_output=target_output.iloc[0],
                image_path=_find_image(image_dir, process_id),
                output_dir=group_output_dir,
                norm=norm,
                size_max=size_max,
                panel_letter=_panel_letter(panel_index),
                panel_subtitle=(
                    _species_subtitle(_panel_letter(panel_index), process_id, species)
                    if args.split_target_species
                    else None
                ),
                subtitle_size=float(args.subtitle_size),
                dpi=int(args.dpi),
            )
            rendered.append(target_edge_id)
            group_frames.append(selected)
            print(f"[DONE] {group_name or 'all'} {target_edge_id}", flush=True)
        group_summary = pd.concat(group_frames, ignore_index=True)
        group_summary.to_csv(group_output_dir / "shap_target_edge_node_summary.csv", index=False)
        rendered_by_group[group_name or "all"] = rendered

    summary.to_csv(output_dir / "shap_target_edge_node_summary.csv", index=False)
    if args.split_target_species:
        target_order.to_csv(output_dir / "target_edge_species_map.csv", index=False)
    manifest = {
        "source_summary": str(summary_path),
        "target_output_coordinates": str(target_path),
        "target_answer_edges": str(target_answer_path) if args.split_target_species else "",
        "target_material_map": str(target_material_path) if target_material_path else "",
        "image_dir": str(image_dir),
        "figure_count": sum(len(ids) for ids in rendered_by_group.values()),
        "rendered_outputs": rendered_by_group,
        "figure_size_inches": list(FIGSIZE),
        "png_dpi": int(args.dpi),
        "colour_limit": limit,
        "split_prod_targets": bool(args.split_prod_targets),
        "split_target_species": bool(args.split_target_species),
        "marker_style": {
            "fill_alpha": 0.58,
            "minimum_area_points_squared": 420.0,
            "maximum_area_points_squared": 1320.0,
            "outline_linewidth": 1.15,
            "numeric_label": "signed relative SHAP, up to three significant figures",
        },
        "target_edge_style": "small gold star with a thin gold ring",
        "subtitle": (
            "top-centred panel letter, process ID, and target-molecule formula"
            if args.split_target_species
            else "top-centred panel letter, process ID, and target-edge ID"
        ),
    }
    (output_dir / "shap_overlay_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    readme_lines = [
        "# Target-edge SHAP flowsheets",
        "",
        "All panels share a fixed 16 x 8 in canvas; native PFD geometry is preserved.",
        f"The signed-SHAP colour scale is symmetric at +/-{limit:.3g}.",
        "",
    ]
    for group_name, target_ids in rendered_by_group.items():
        heading = "PROD target edges" if group_name == "prod_target" else "Other target edges"
        if args.split_target_species and group_name.endswith("_target_edges"):
            species = group_name.removesuffix("_target_edges")
            heading = f"{SPECIES_MATHTEXT[species]} target edges"
        if group_name == "all":
            heading = "Target edges"
        readme_lines.extend([f"## {heading}", ""])
        for panel_index, target_edge_id in enumerate(target_ids):
            group_order_row = grouped_orders[
                list(rendered_by_group).index(group_name)
            ][1].iloc[panel_index]
            property_suffix = (
                f" (`{group_order_row['target_property']}`)"
                if "target_property" in group_order_row.index
                else ""
            )
            readme_lines.append(
                f"- ({_panel_letter(panel_index)}) `{target_edge_id}`{property_suffix}"
            )
        readme_lines.append("")
    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(
        f"figure_count={sum(len(ids) for ids in rendered_by_group.values())}; "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
