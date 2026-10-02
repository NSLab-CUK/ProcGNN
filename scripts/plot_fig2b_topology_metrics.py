"""Generate Fig. 2b only: topology metrics across the ten SMR flowsheets.

The figure is calculated exclusively from the canonical P01--P10 topology
metadata.  It neither trains nor evaluates a model.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = (
    ROOT
    / "outputs"
    / "0819final"
    / "Paper_Tables_Final3_figures"
    / "final"
    / "Fig2b_topology_metrics"
)

NODES_CSV = ROOT / "data" / "reference" / "v3" / "canonical_nodes.csv"
EDGES_CSV = ROOT / "data" / "reference" / "v3" / "canonical_edges.csv"
SPEC_CSV = ROOT / "data" / "reference" / "process_graph_canonical_spec.csv"
CONSTANTS_PY = ROOT / "src" / "process_graph" / "constants.py"
NODE_BALANCE_PY = ROOT / "src" / "process_graph" / "experiment" / "node_balance_pi.py"

SOURCE_FILES = (NODES_CSV, EDGES_CSV, SPEC_CSV, CONSTANTS_PY, NODE_BALANCE_PY)
PROCESS_IDS = tuple(range(1, 11))

BRANCH_DEFINITION = (
    "Count of physical unit nodes with more than one outgoing canonical "
    "material-stream edge; virtual boundary nodes are excluded."
)
RECYCLE_DEFINITION = (
    "Unavailable: canonical edge metadata has no explicit recycle marker. "
    "Directed cycles are not counted because heat-exchanger unit paths can "
    "form graph cycles that are not process recycle loops."
)
DIAMETER_DEFINITION = (
    "Maximum finite NetworkX diameter across connected components of the "
    "undirected (weak) projection of the canonical material-stream graph, "
    "including V_INPUT and V_OUTPUT boundary nodes; isolated nodes have diameter 0."
)


def relative_path(path: Path) -> str:
    """Return an ASCII-safe, repository-relative source/output path."""
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def literal_assignment(path: Path, name: str) -> Any:
    """Read a literal constant from project source without importing training code."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in statement.targets):
                return ast.literal_eval(statement.value)
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            if statement.target.id == name:
                return ast.literal_eval(statement.value)
    raise KeyError(f"Could not locate literal assignment {name!r} in {path}")


def physical_node_mask(nodes: pd.DataFrame) -> pd.Series:
    virtual = (
        nodes["is_virtual"].astype(str).str.strip().str.casefold().isin({"true", "1", "yes"})
    )
    return nodes["node_type"].eq("unit") & ~virtual


def validate_canonical_spec() -> None:
    """Verify the documented canonical-boundary convention used for the counts."""
    spec_text = SPEC_CSV.read_text(encoding="utf-8-sig")
    required_terms = ("V_INPUT", "V_OUTPUT", "canonical")
    missing_terms = [term for term in required_terms if term not in spec_text]
    if missing_terms:
        raise ValueError(
            "Canonical graph specification does not document the expected boundary/unit convention: "
            f"missing {missing_terms}"
        )


def validate_inputs(nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    expected_processes = set(PROCESS_IDS)
    actual_nodes = set(nodes["process_id"].unique())
    actual_edges = set(edges["process_id"].unique())
    if actual_nodes != expected_processes or actual_edges != expected_processes:
        raise ValueError(
            "Canonical topology files must contain exactly P01--P10; "
            f"node IDs={sorted(actual_nodes)}, edge IDs={sorted(actual_edges)}"
        )
    if nodes.duplicated(["process_id", "node_name"]).any():
        raise ValueError("Duplicate canonical node names found within a process.")
    if edges.duplicated(["process_id", "canonical_edge_id"]).any():
        raise ValueError("Duplicate canonical edge IDs found within a process.")
    allowed_roles = {"input", "internal", "output"}
    observed_roles = set(edges["stream_role"].dropna().unique())
    if not observed_roles <= allowed_roles:
        raise ValueError(f"Unexpected canonical stream roles: {sorted(observed_roles - allowed_roles)}")

    for process_id in PROCESS_IDS:
        node_names = set(nodes.loc[nodes["process_id"].eq(process_id), "node_name"])
        process_edges = edges.loc[edges["process_id"].eq(process_id)]
        referenced = set(process_edges["src_node"]) | set(process_edges["dst_node"])
        missing = referenced - node_names
        if missing:
            raise ValueError(f"P{process_id:02d} edges refer to unknown nodes: {sorted(missing)}")


def calculate_metrics(
    nodes: pd.DataFrame, edges: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Calculate only the requested topology metrics from canonical metadata."""
    unit_type_map: dict[str, str] = literal_assignment(CONSTANTS_PY, "UNIT_TYPE_TO_CANONICAL")
    reactive_types = set(literal_assignment(NODE_BALANCE_PY, "REACTIVE_NODE_TYPES"))
    hx_types = set(literal_assignment(NODE_BALANCE_PY, "HX_NODE_TYPES"))

    rows: list[dict[str, object]] = []
    connectivity_notes: list[str] = []
    for process_id in PROCESS_IDS:
        process_nodes = nodes.loc[nodes["process_id"].eq(process_id)].copy()
        process_edges = edges.loc[edges["process_id"].eq(process_id)].copy()
        physical_nodes = process_nodes.loc[physical_node_mask(process_nodes)].copy()
        physical_nodes["canonical_unit_type"] = physical_nodes["unit_type"].map(unit_type_map)
        if physical_nodes["canonical_unit_type"].isna().any():
            missing = sorted(physical_nodes.loc[physical_nodes["canonical_unit_type"].isna(), "unit_type"].unique())
            raise ValueError(f"P{process_id:02d} unit types missing from project alias map: {missing}")

        outgoing_counts = process_edges.groupby("src_node", sort=False).size()
        branches = sum(
            int(outgoing_counts.get(node_name, 0)) > 1
            for node_name in physical_nodes["node_name"]
        )

        material_graph = nx.MultiDiGraph()
        material_graph.add_nodes_from(process_nodes["node_name"])
        material_graph.add_edges_from(process_edges[["src_node", "dst_node"]].itertuples(index=False, name=None))
        undirected_graph = nx.Graph(material_graph)
        components = list(nx.connected_components(undirected_graph))
        diameter = max(nx.diameter(undirected_graph.subgraph(component)) for component in components)
        if len(components) > 1:
            isolated = sorted(
                node for component in components if len(component) == 1 for node in component
            )
            connectivity_notes.append(
                f"P{process_id:02d}: {len(components)} weak components; "
                f"isolated canonical nodes={isolated or 'none'}; "
                "reported diameter is the maximum finite component diameter."
            )

        rows.append(
            {
                "Flowsheet": f"P{process_id:02d}",
                "Units": int(len(physical_nodes)),
                "Streams": int(len(process_edges)),
                "Branches": int(branches),
                "Recycle loops": pd.NA,
                "Graph diameter": int(diameter),
                "Heat exchangers": int(physical_nodes["canonical_unit_type"].isin(hx_types).sum()),
                "Reactive units": int(physical_nodes["canonical_unit_type"].isin(reactive_types).sum()),
                "Branch definition": BRANCH_DEFINITION,
                "Recycle loops definition": RECYCLE_DEFINITION,
                "Graph diameter definition": DIAMETER_DEFINITION,
            }
        )

    raw = pd.DataFrame(rows)
    if raw["Flowsheet"].duplicated().any() or len(raw) != len(PROCESS_IDS):
        raise ValueError("Expected one raw-statistics row for each P01--P10 flowsheet.")
    return (
        raw,
        ["Units", "Streams", "Branches", "Graph diameter", "Heat exchangers", "Reactive units"],
        connectivity_notes,
    )


def minmax_normalize(raw: pd.DataFrame, metrics: list[str]) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    normalized = pd.DataFrame({"Flowsheet": raw["Flowsheet"]})
    minmax: dict[str, tuple[float, float]] = {}
    for metric in metrics:
        values = pd.to_numeric(raw[metric], errors="raise").astype(float)
        minimum, maximum = float(values.min()), float(values.max())
        minmax[metric] = (minimum, maximum)
        if np.isclose(maximum, minimum):
            normalized[metric] = 0.5
            print(f"NOTE: {metric} has zero variance; normalized display values set to 0.5.")
        else:
            normalized[metric] = (values - minimum) / (maximum - minimum)
    return normalized, minmax


def render_heatmap(raw: pd.DataFrame, normalized: pd.DataFrame, metrics: list[str], pdf_path: Path, png_path: Path) -> tuple[float, float]:
    mpl.rcParams.update(
        {
            # Matches the existing manuscript subfigure-title typography.
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure_size = (8.1, 5.35)
    fig, ax = plt.subplots(figsize=figure_size, layout="constrained", facecolor="white")
    values = normalized.loc[:, metrics].to_numpy(dtype=float)
    image = ax.imshow(values, cmap=mpl.colormaps["Blues"], vmin=0.0, vmax=1.0, aspect="auto")

    # Raw values, rather than the normalized color values, are the annotations.
    for row_index in range(values.shape[0]):
        for column_index, metric in enumerate(metrics):
            raw_value = int(raw.iloc[row_index][metric])
            text_color = "white" if values[row_index, column_index] >= 0.58 else "#17233a"
            ax.text(
                column_index,
                row_index,
                str(raw_value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="semibold",
            )

    labels = ["Units", "Streams", "Branches", "Graph\ndiameter", "Heat\nexchangers", "Reactive\nunits"]
    ax.set_xticks(np.arange(len(metrics)), labels=labels)
    ax.set_yticks(np.arange(len(raw)), labels=raw["Flowsheet"].tolist())
    ax.tick_params(axis="x", length=0, pad=7)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_title(
        "(b) Topology Metrics across the Ten SMR Flowsheets",
        pad=19,
        fontweight="normal",
        color="#20252B",
    )

    ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(raw), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(image, ax=ax, pad=0.025, shrink=0.96)
    colorbar.set_label("Column-wise normalized value", labelpad=10)
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.outline.set_linewidth(0.5)

    fig.savefig(pdf_path, format="pdf", facecolor="white", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, format="png", dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return figure_size


def prepare_outputs(overwrite: bool) -> dict[str, Path]:
    outputs = {
        "pdf": OUT_DIR / "Fig2b_topology_metrics.pdf",
        "png": OUT_DIR / "Fig2b_topology_metrics_600dpi.png",
        "raw": OUT_DIR / "Fig2b_topology_metrics_raw.csv",
        "normalized": OUT_DIR / "Fig2b_topology_metrics_normalized.csv",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing Fig. 2b outputs: {names}. Use --overwrite.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()
    return outputs


def print_validation(
    raw: pd.DataFrame,
    metrics: list[str],
    minmax: dict[str, tuple[float, float]],
    connectivity_notes: list[str],
) -> None:
    display_columns = [
        "Flowsheet",
        "Units",
        "Streams",
        "Branches",
        "Recycle loops",
        "Graph diameter",
        "Heat exchangers",
        "Reactive units",
    ]
    print("\nRaw topology statistics (P01-P10):")
    print(raw.loc[:, display_columns].fillna("unavailable").to_string(index=False))
    print("\nDefinitions:")
    print(f"- Branches: {BRANCH_DEFINITION}")
    print(f"- Recycle loops: {RECYCLE_DEFINITION}")
    print(f"- Graph diameter: {DIAMETER_DEFINITION}")
    print("WARNING: Recycle loops are unavailable and omitted from the heatmap; no value was inferred.")
    for note in connectivity_notes:
        print(f"NOTE: {note}")

    print("\nMinimum and maximum by plotted metric:")
    variation_rows: list[dict[str, object]] = []
    for metric in metrics:
        minimum, maximum = minmax[metric]
        numeric = pd.to_numeric(raw[metric], errors="raise")
        min_flowsheets = ", ".join(raw.loc[numeric.eq(minimum), "Flowsheet"])
        max_flowsheets = ", ".join(raw.loc[numeric.eq(maximum), "Flowsheet"])
        raw_range = maximum - minimum
        variation_rows.append({"Metric": metric, "Raw range": raw_range})
        print(
            f"- {metric}: min {minimum:g} ({min_flowsheets}); "
            f"max {maximum:g} ({max_flowsheets}); raw range {raw_range:g}"
        )
    ranked = pd.DataFrame(variation_rows).sort_values("Raw range", ascending=False, kind="stable")
    print("\nMetrics varying most by raw range (units are not commensurate):")
    print(ranked.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="replace only the four Fig. 2b output files")
    args = parser.parse_args()

    missing_sources = [path for path in SOURCE_FILES if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"Required source files are missing: {missing_sources}")
    print("Source files used:")
    for source_file in SOURCE_FILES:
        print(f"- {relative_path(source_file)}")

    nodes = pd.read_csv(NODES_CSV)
    edges = pd.read_csv(EDGES_CSV)
    validate_canonical_spec()
    validate_inputs(nodes, edges)
    raw, plot_metrics, connectivity_notes = calculate_metrics(nodes, edges)
    normalized, minmax = minmax_normalize(raw, plot_metrics)
    outputs = prepare_outputs(args.overwrite)

    raw.to_csv(outputs["raw"], index=False, na_rep="")
    normalized.to_csv(outputs["normalized"], index=False, float_format="%.6f")
    figure_size = render_heatmap(raw, normalized, plot_metrics, outputs["pdf"], outputs["png"])
    print_validation(raw, plot_metrics, minmax, connectivity_notes)

    print(f"\nFigure size: {figure_size[0]:.2f} x {figure_size[1]:.2f} inches")
    print("Output paths:")
    for path in outputs.values():
        print(f"- {relative_path(path)}")


if __name__ == "__main__":
    main()
