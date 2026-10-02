#!/usr/bin/env python3
"""Visualize v3 canonical process graphs for edge_all experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


ROLE_COLORS = {
    "input": "#2b8cbe",
    "internal": "#5aae61",
    "product": "#d95f02",
    "exhaust": "#d95f02",
    "output": "#d95f02",
    "zero_or_side_output": "#d95f02",
    "unknown": "#888888",
}

UNIT_COLORS = {
    "virtual": "#f3f3f3",
    "reactor": "#f4a261",
    "compressor": "#8ecae6",
    "HX_dt": "#b7e4c7",
    "HX_area": "#b7e4c7",
    "mixer": "#cdb4db",
    "separator": "#ffd166",
    "splitter": "#ffafcc",
    "burner": "#e76f51",
    "stream": "#dddddd",
}


def _truthy(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _edge_role(row: pd.Series) -> str:
    role = str(row.get("stream_role", "") or "").strip()
    if role:
        return role
    if _truthy(row.get("is_input_edge", False)):
        return "input"
    if _truthy(row.get("is_output_edge", False)):
        return "output"
    if _truthy(row.get("is_internal_edge", False)):
        return "internal"
    return "unknown"


def _node_label(row: pd.Series) -> str:
    name = str(row.get("node_name", ""))
    unit = str(row.get("unit_type", "") or "")
    if unit and unit != "nan" and name not in {"V_INPUT", "V_OUTPUT"}:
        return f"{name}\n{unit}"
    return name


def _edge_label(row: pd.Series) -> str:
    edge_id = str(row.get("canonical_edge_id", ""))
    key = str(row.get("main_data_stream_key", "") or "").strip()
    if key.lower() == "nan":
        key = ""
    visual = str(row.get("visual_stream_id", "") or "").strip()
    if visual.lower() == "nan":
        visual = ""
    stream = key or visual
    return f"{edge_id}\n{stream}" if stream else edge_id


def _topological_layout(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    try:
        generations = list(nx.topological_generations(graph))
    except nx.NetworkXUnfeasible:
        return nx.spring_layout(graph, seed=42, k=1.4)
    pos: dict[str, tuple[float, float]] = {}
    for x, generation in enumerate(generations):
        nodes = sorted(generation)
        n = max(len(nodes), 1)
        for i, node in enumerate(nodes):
            y = (n - 1) / 2.0 - i
            pos[str(node)] = (float(x), float(y))
    return pos


def draw_process_graph(
    *,
    pid: int,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    answers_df: pd.DataFrame,
    out_dir: Path,
    formats: tuple[str, ...],
) -> None:
    ndf = nodes_df[nodes_df["process_id"].astype(int) == int(pid)].copy()
    edf = edges_df[edges_df["process_id"].astype(int) == int(pid)].copy()
    adf = answers_df[answers_df["process_id"].astype(int) == int(pid)].copy()
    if ndf.empty or edf.empty:
        raise ValueError(f"No v3 graph rows found for process {pid}.")

    graph = nx.DiGraph()
    for _, row in ndf.iterrows():
        graph.add_node(
            str(row["node_name"]),
            label=_node_label(row),
            unit_type=str(row.get("unit_type", "virtual") or "virtual"),
            is_virtual=_truthy(row.get("is_virtual", False)),
        )
    for _, row in edf.iterrows():
        graph.add_edge(
            str(row["src_node"]),
            str(row["dst_node"]),
            edge_id=str(row["canonical_edge_id"]),
            label=_edge_label(row),
            role=_edge_role(row),
            stream_key=str(row.get("main_data_stream_key", "") or ""),
        )

    answer_edges = {str(x) for x in adf["canonical_answer_edge_id"].astype(str).tolist()}
    answer_label_by_edge = {
        str(row["canonical_answer_edge_id"]): str(row["task_name"]) for _, row in adf.iterrows()
    }
    pos = _topological_layout(graph)

    width = max(12.0, min(24.0, 1.3 * max(4, len(set(x for x, _ in pos.values())))))
    height = max(7.0, min(16.0, 0.6 * max(8, len(graph.nodes))))
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_title(
        f"Process {pid:02d} canonical stream-edge graph\n"
        "edge colors: input blue, internal green, output orange; answer edges highlighted",
        fontsize=13,
    )

    node_colors = []
    for node in graph.nodes:
        data = graph.nodes[node]
        if node == "V_INPUT":
            node_colors.append("#e0f3db")
        elif node == "V_OUTPUT":
            node_colors.append("#fee0d2")
        else:
            node_colors.append(UNIT_COLORS.get(str(data.get("unit_type")), "#dddddd"))

    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=2300,
        edgecolors="#333333",
        linewidths=1.0,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={node: str(data.get("label", node)) for node, data in graph.nodes(data=True)},
        ax=ax,
        font_size=8,
    )

    normal_edges = []
    answer_draw_edges = []
    for u, v, data in graph.edges(data=True):
        item = (u, v)
        if str(data.get("edge_id")) in answer_edges:
            answer_draw_edges.append(item)
        else:
            normal_edges.append(item)

    edge_colors = [
        ROLE_COLORS.get(str(graph.edges[e].get("role", "unknown")), ROLE_COLORS["unknown"])
        for e in normal_edges
    ]
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=normal_edges,
        edge_color=edge_colors,
        width=1.6,
        arrows=True,
        arrowsize=18,
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=answer_draw_edges,
        edge_color="#c1121f",
        width=3.8,
        arrows=True,
        arrowsize=24,
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )

    edge_labels = {}
    for u, v, data in graph.edges(data=True):
        label = str(data.get("label", ""))
        edge_id = str(data.get("edge_id", ""))
        if edge_id in answer_edges:
            label = f"{label}\n[{answer_label_by_edge.get(edge_id, 'answer')}]"
        edge_labels[(u, v)] = label
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=6,
        label_pos=0.5,
        rotate=False,
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75},
        ax=ax,
    )

    legend_items = [
        plt.Line2D([0], [0], color="#2b8cbe", lw=2, label="input edge"),
        plt.Line2D([0], [0], color="#5aae61", lw=2, label="internal edge"),
        plt.Line2D([0], [0], color="#d95f02", lw=2, label="output edge"),
        plt.Line2D([0], [0], color="#c1121f", lw=4, label="answer edge"),
    ]
    ax.legend(handles=legend_items, loc="lower center", ncol=4, frameon=False)
    ax.axis("off")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"process_{pid:02d}_canonical_graph"
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=220 if fmt == "png" else None)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize all v3 canonical process graphs.")
    parser.add_argument("--reference-dir", default="data/reference/v3")
    parser.add_argument("--output-dir", default="outputs/process_graph_visualizations")
    parser.add_argument("--process-ids", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--formats", nargs="+", default=["png", "svg"], choices=["png", "svg", "pdf"])
    args = parser.parse_args()

    ref_dir = (PROJECT_ROOT / args.reference_dir).resolve()
    out_dir = (PROJECT_ROOT / args.output_dir).resolve()
    nodes_df = pd.read_csv(ref_dir / "canonical_nodes.csv")
    edges_df = pd.read_csv(ref_dir / "canonical_edges.csv")
    answers_df = pd.read_csv(ref_dir / "target_answer_edges.csv")

    for pid in args.process_ids:
        draw_process_graph(
            pid=int(pid),
            nodes_df=nodes_df,
            edges_df=edges_df,
            answers_df=answers_df,
            out_dir=out_dir,
            formats=tuple(str(x).lower() for x in args.formats),
        )
        print(f"[graph] wrote Process {int(pid):02d} visualization under {out_dir}")


if __name__ == "__main__":
    main()
