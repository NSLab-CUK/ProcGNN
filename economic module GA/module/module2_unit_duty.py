"""Generic topology-based unit-duty helpers for the economic module."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

import properties as props


def build_edge_properties(edge_stream_data: Mapping[tuple[str, str], Mapping[str, object]]) -> dict[tuple[str, str], dict[str, float]]:
    """Reconstruct enthalpy and enthalpy flow for explicit edge streams."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    for edge, stream in edge_stream_data.items():
        try:
            temp = float(stream["temp_c"])
            pres = float(stream["pres_bar"])
            frac = np.asarray(stream["mole_fractions"], dtype=float)
            mass = float(stream["mass_flow_kgh"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid stream data for edge {edge!r}.") from exc
        if frac.shape != (len(props.SPECIES),):
            raise ValueError(f"Edge {edge!r}: expected {len(props.SPECIES)} mole fractions, got {frac.shape}.")
        if (not np.isfinite([temp, pres, mass]).all() or not np.isfinite(frac).all()
                or mass < 0.0 or pres <= 0.0 or np.any(frac < 0.0)
                or not np.isclose(frac.sum(), 1.0, atol=5e-2)):
            raise ValueError(f"Edge {edge!r}: invalid pressure, flow, or mole fractions.")
        reconstructed = props.reconstruct_stream_properties(
            np.asarray([temp]), np.asarray([pres]), frac.reshape(1, -1), np.asarray([mass])
        )
        h = float(reconstructed["enthalpy_kj_kg"][0])
        out[tuple(edge)] = {
            "enthalpy_kj_kg": h,
            "enthalpy_flow_kw": float(mass / 3600.0 * h),
            "density_kg_m3": float(reconstructed["density_kg_m3"][0]),
        }
    return out


def node_duty_kw(node: str, topology: Mapping[str, Mapping[str, list[str]]],
                 edge_properties: Mapping[tuple[str, str], Mapping[str, float]]) -> float:
    """Return ``sum(H_out) - sum(H_in)`` for one node in kW."""
    if node not in topology:
        raise ValueError(f"Node {node!r} is not present in the topology.")
    incoming = list(topology[node].get("in", []))
    outgoing = list(topology[node].get("out", []))
    if not incoming or not outgoing:
        raise ValueError(f"Node {node!r} needs at least one inlet and one outlet.")

    def flow(edge: tuple[str, str]) -> float:
        payload = edge_properties.get(edge)
        if payload is None or "enthalpy_flow_kw" not in payload:
            raise ValueError(f"Missing enthalpy properties for edge {edge!r}.")
        value = float(payload["enthalpy_flow_kw"])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite enthalpy flow for edge {edge!r}.")
        return value

    return float(sum(flow((source, node)) for source in incoming)
                 * -1.0 + sum(flow((node, destination)) for destination in outgoing))


def _workbook(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Topology workbook not found: {resolved}")
    return resolved


def _adjacency_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_excel(_workbook(path), sheet_name=0, header=None)
    labels = frame.iloc[0, 1:].dropna().astype(str).str.strip().tolist()
    if not labels:
        raise ValueError("Topology sheet has no node labels.")
    matrix = frame.iloc[1:1 + len(labels), 1:1 + len(labels)].copy()
    matrix.index = frame.iloc[1:1 + len(labels), 0].astype(str).str.strip().tolist()
    matrix.columns = labels
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def load_topology(path: str | Path) -> tuple[dict[str, dict[str, list[str]]], list[tuple[str, str]]]:
    """Load the first workbook sheet into inlet/outlet lists and directed edges."""
    matrix = _adjacency_frame(path)
    topology = {node: {"in": [], "out": []} for node in matrix.index}
    edges: list[tuple[str, str]] = []
    for source in matrix.index:
        for destination in matrix.columns:
            if float(matrix.loc[source, destination]) == 0.0:
                continue
            topology.setdefault(destination, {"in": [], "out": []})
            topology[source]["out"].append(destination)
            topology[destination]["in"].append(source)
            edges.append((source, destination))
    return topology, edges


def load_node_types(path: str | Path) -> dict[str, str]:
    """Load node types from the second workbook sheet."""
    frame = pd.read_excel(_workbook(path), sheet_name=1, header=None)
    result: dict[str, str] = {}
    for _, row in frame.iloc[2:].iterrows():
        node, node_type = str(row.iloc[0]).strip(), str(row.iloc[1]).strip()
        if node.lower() != "nan" and node_type.lower() != "nan":
            result[node] = node_type
    return result
