"""Create only Fig. 2c: process-unit presence across the ten SMR flowsheets.

This script extracts physical-unit metadata from the canonical P01--P10
topology table. It does not load models, predictions, or experimental metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
NODE_METADATA = ROOT / "data" / "reference" / "v3" / "canonical_nodes.csv"
CANONICAL_SPEC = ROOT / "data" / "reference" / "process_graph_canonical_spec.csv"
UNIT_CONSTANTS = ROOT / "src" / "process_graph" / "constants.py"
HX_ALIAS_REFERENCE = ROOT / "src" / "process_graph" / "experiment" / "node_balance_pi.py"
OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "0819final"
    / "Paper_Tables_Final3_figures"
    / "final"
    / "Fig2c_unit_type_presence"
)

PROCESS_IDS = list(range(1, 11))

# This order follows process interpretation: reaction -> heat transfer ->
# pressure change -> mixing/splitting -> separation.  Every raw metadata
# name is explicitly validated against canonical_nodes.csv before plotting.
CATEGORY_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("SMR reactor", ("SMR reactor",)),
    ("WGS reactor", ("WGS reactor",)),
    ("Burner", ("burner",)),
    ("Heat exchanger", ("HX_dt", "HX_hot", "HX_cold")),
    ("Heater", ("heater",)),
    ("Cooler", ("cooler",)),
    ("Compressor", ("compressor",)),
    ("Pump", ("pump",)),
    ("Turbine", ("turbine",)),
    ("Mixer", ("mixer",)),
    ("Splitter", ("splitter",)),
    ("PSA", ("psa",)),
    ("Flash", ("flash",)),
)


def _relative_path(path: Path) -> str:
    """Print an exact repository-relative path without console encoding ambiguity."""
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _require_sources() -> None:
    for source in (NODE_METADATA, CANONICAL_SPEC, UNIT_CONSTANTS, HX_ALIAS_REFERENCE):
        if not source.is_file():
            raise FileNotFoundError(f"Required repository source is missing: {source}")

    spec_text = CANONICAL_SPEC.read_text(encoding="utf-8-sig")
    if "V_INPUT/V_OUTPUT" not in spec_text or "물리 unit" not in spec_text:
        raise RuntimeError("Canonical graph specification no longer documents virtual/unit node separation.")

    constants_text = UNIT_CONSTANTS.read_text(encoding="utf-8")
    for raw_name, canonical_name in (("HX_dt", "hx_dt"), ("HX_hot", "hx_hot"), ("HX_cold", "hx_cold")):
        if f'"{raw_name}": "{canonical_name}"' not in constants_text:
            raise RuntimeError(f"Project canonical unit mapping for {raw_name} was not found.")

    alias_text = HX_ALIAS_REFERENCE.read_text(encoding="utf-8")
    expected_alias = '"heat_exchanger": {"hx_dt", "hx_hot", "hx_cold"}'
    if expected_alias not in alias_text:
        raise RuntimeError("Project heat_exchanger alias no longer groups hx_dt/hx_hot/hx_cold.")


def _physical_units() -> pd.DataFrame:
    nodes = pd.read_csv(NODE_METADATA)
    required = {"process_id", "node_type", "unit_type", "is_virtual"}
    missing = required.difference(nodes.columns)
    if missing:
        raise ValueError(f"Canonical node metadata is missing required columns: {sorted(missing)}")

    nodes["process_id"] = pd.to_numeric(nodes["process_id"], errors="raise").astype(int)
    virtual = nodes["is_virtual"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    physical = nodes.loc[nodes["node_type"].eq("unit") & ~virtual].copy()
    found_processes = sorted(physical["process_id"].unique().tolist())
    if found_processes != PROCESS_IDS:
        raise ValueError(
            "Expected physical metadata for P01--P10 exactly; "
            f"found process IDs {found_processes}."
        )
    if physical.empty:
        raise ValueError("No physical unit rows remain after virtual-node exclusion.")
    return physical


def _build_tables(physical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    expected_raw = {raw for _, raw_types in CATEGORY_SPECS for raw in raw_types}
    observed_raw = set(physical["unit_type"].astype(str).unique())
    unknown = sorted(observed_raw.difference(expected_raw))
    missing = sorted(expected_raw.difference(observed_raw))
    if unknown or missing:
        raise ValueError(
            "Unit-category mapping must cover exactly the observed physical unit metadata; "
            f"unknown={unknown}, expected-but-absent={missing}."
        )

    presence_rows: list[dict[str, int | str]] = []
    count_rows: list[dict[str, int | str]] = []
    units_by_process: dict[str, list[str]] = {}
    for process_id in PROCESS_IDS:
        subset = physical.loc[physical["process_id"].eq(process_id)]
        process_label = f"P{process_id:02d}"
        presence_row: dict[str, int | str] = {"Flowsheet": process_label}
        present_labels: list[str] = []
        for display_name, raw_types in CATEGORY_SPECS:
            raw_counts = (
                subset.loc[subset["unit_type"].isin(raw_types), "unit_type"]
                .value_counts()
                .reindex(raw_types, fill_value=0)
            )
            unit_count = int(raw_counts.sum())
            presence = int(unit_count > 0)
            presence_row[display_name] = presence
            if presence:
                present_labels.append(display_name)
            count_rows.append(
                {
                    "Flowsheet": process_label,
                    "Process_ID": process_id,
                    "Unit_type": display_name,
                    "Unit_count": unit_count,
                    "Presence": presence,
                    "Source_metadata_unit_types": "; ".join(raw_types),
                    "Source_metadata_counts": "; ".join(
                        f"{raw_type}={int(raw_counts.loc[raw_type])}" for raw_type in raw_types
                    ),
                }
            )
        presence_rows.append(presence_row)
        units_by_process[process_label] = present_labels

    presence = pd.DataFrame(presence_rows).set_index("Flowsheet")
    counts = pd.DataFrame(count_rows)
    return presence, counts, units_by_process


def _plot(presence: pd.DataFrame, pdf_path: Path, png_path: Path) -> tuple[float, float]:
    figure_size = (9.2, 5.5)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11.5,
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    ):
        fig, ax = plt.subplots(figsize=figure_size, layout="constrained")
        matrix = presence.to_numpy(dtype=int)
        cmap = ListedColormap(["#F3F6F8", "#1F5A8A"])
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, interpolation="nearest", aspect="auto")

        ax.set_xticks(np.arange(presence.shape[1]))
        ax.set_xticklabels(presence.columns, rotation=38, ha="right", rotation_mode="anchor")
        ax.set_yticks(np.arange(presence.shape[0]))
        ax.set_yticklabels(presence.index)
        ax.set_xlabel("Process-unit type", labelpad=10)
        ax.set_ylabel("Flowsheet", labelpad=10)
        ax.set_title(
            "(c) Process-Unit Types across the Ten SMR Flowsheets",
            pad=24,
            weight="normal",
            color="#20252B",
        )

        ax.set_xticks(np.arange(-0.5, presence.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, presence.shape[0], 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="both", which="major", length=0, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # The matrix has no data-free internal area. The compact legend is placed
        # immediately outside the axes to avoid covering any cell or label.
        legend = ax.legend(
            handles=[
                Patch(facecolor="#1F5A8A", edgecolor="none", label="Present"),
                Patch(facecolor="#F3F6F8", edgecolor="#AAB7C4", label="Absent"),
            ],
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=True,
            framealpha=1.0,
            facecolor="white",
            edgecolor="#C7D1DB",
            fontsize=11,
            handlelength=1.1,
            labelspacing=0.5,
        )
        legend.get_frame().set_linewidth(0.7)

        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=0.06)
        fig.savefig(png_path, format="png", dpi=600, bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
    return figure_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the four known Fig. 2c outputs in the dedicated output directory.",
    )
    args = parser.parse_args()
    _require_sources()
    print("Source files used:")
    for source in (NODE_METADATA, CANONICAL_SPEC, UNIT_CONSTANTS, HX_ALIAS_REFERENCE):
        print(f"  - {_relative_path(source)}")

    physical = _physical_units()
    presence, counts, units_by_process = _build_tables(physical)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / "Fig2c_unit_type_presence.pdf"
    png_path = OUTPUT_DIR / "Fig2c_unit_type_presence_600dpi.png"
    presence_path = OUTPUT_DIR / "Fig2c_unit_type_presence.csv"
    counts_path = OUTPUT_DIR / "Fig2c_unit_type_counts.csv"
    outputs = (pdf_path, png_path, presence_path, counts_path)
    existing = [output for output in outputs if output.exists()]
    if existing and not args.overwrite:
        existing_text = ", ".join(str(output) for output in existing)
        raise FileExistsError(
            "Refusing to overwrite existing Fig. 2c outputs. "
            f"Pass --overwrite to replace only these known files: {existing_text}"
        )
    if args.overwrite:
        for output in existing:
            output.unlink()

    presence.to_csv(presence_path, index=True, index_label="Flowsheet")
    counts.to_csv(counts_path, index=False)
    figure_size = _plot(presence, pdf_path, png_path)

    category_counts = presence.sum(axis=0)
    unique = category_counts[category_counts.eq(1)].index.tolist()
    shared_all = category_counts[category_counts.eq(len(PROCESS_IDS))].index.tolist()
    raw_type_count = physical["unit_type"].nunique()

    print("\nUnit types present in each flowsheet:")
    for flowsheet, labels in units_by_process.items():
        print(f"  {flowsheet}: {', '.join(labels)}")
    print(f"\nDistinct physical unit categories plotted: {presence.shape[1]}")
    print(
        "Raw metadata unit-type labels: "
        f"{raw_type_count} (HX_dt/HX_hot/HX_cold are merged into Heat exchanger)."
    )
    print(f"Unit types unique to one flowsheet: {', '.join(unique) if unique else 'None'}")
    print(f"Unit types shared by all ten flowsheets: {', '.join(shared_all) if shared_all else 'None'}")
    print(f"\nFigure size (inches): {figure_size[0]:.1f} x {figure_size[1]:.1f}")
    print("Output paths:")
    for output in outputs:
        print(f"  - {_relative_path(output)}")


if __name__ == "__main__":
    main()
