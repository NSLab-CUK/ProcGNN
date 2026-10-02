from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd


PROCESS_IDS = list(range(1, 11))
INPUT_NODE_PATTERNS = ("IN_", "INPUT_")
OUTPUT_NODE_PATTERNS = ("OUT_",)

TARGET_RULES: Dict[int, Dict[str, Tuple[str, str]]] = {
    1: {"target_h2": ("OUT_prod", "PROD_H2_Mole"), "tailgas_co2": ("OUT_exhaust", "FUELGAS_CO2_Mole")},
    2: {"target_h2": ("OUT_PROD", "0.8*PROD_H2_MoleFlow"), "tailgas_co2": ("OUT_EXHAUST", "PROD_CO2_Mole")},
    3: {"target_h2": ("OUT_PROD", "PROD_H2_Mole"), "tailgas_co2": ("OUT_CO2", "OUT_CO2_Mole")},
    4: {"target_h2": ("OUT_PROD", "PROD_H2_Mole"), "tailgas_co2": ("OUT_EXHAUST", "EX_CO2_Mole")},
    5: {"target_h2": ("OUT_PROD", "PROD_H2_Mole"), "tailgas_co2": ("OUT_CO2", "Stream12_CO2_Mole")},
    6: {"target_h2": ("OUT_PROD", "PROD_H2_Mole"), "tailgas_co2": ("OUT_CO2", "Stream12_CO2_Mole")},
    7: {"target_h2": ("OUT_PROD", "PROD_H2_Mole"), "tailgas_co2": ("OUT_PROD", "PROD_CO2_Mole")},
    8: {"target_h2": ("OUT_PROD", "PROD_H2_Mole"), "tailgas_co2": ("OUT_CO2", "12_CO2_Mole")},
    9: {"target_h2": ("OUT_PROD", "PROD_H2_Mole"), "tailgas_co2": ("OUT_CO2", "EXHAUS_CO2_Mole")},
    10: {"target_h2": ("OUT_PROD", "PROD_H2_Mole"), "tailgas_co2": ("OUT_CO2", "OUT_CO2_Mole")},
}

NO_MAPPING_NOTE = "no explicit mapping found in process_1_10_stream_edge_mapping_final.csv"
EXPLICIT_NOTE = "matched from process_1_10_stream_edge_mapping_final.csv"

# Manual fixes where canonical (src,dst) matches multiple mapping rows or adjacency omits streams present in inventory.
REFERENCE_EDGE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "P04_E003": {
        "main_data_stream_key": "FUEL",
        "stream_name_norm": "FUEL",
        "visual_stream_id": "",
        "join_on_stream_name": "FUEL",
        "mapping_confidence": "explicit_mapping",
        "mapping_note": "manual: IN_FUEL vs IN_AIR disambiguation per PFD (FUEL to BURNER); mapping CSV rows share (V_INPUT,BURNER).",
    },
    "P04_E004": {
        "main_data_stream_key": "AIR",
        "stream_name_norm": "AIR",
        "visual_stream_id": "",
        "join_on_stream_name": "AIR",
        "mapping_confidence": "explicit_mapping",
        "mapping_note": "manual: IN_FUEL vs IN_AIR disambiguation per PFD (AIR to BURNER); mapping CSV rows share (V_INPUT,BURNER).",
    },
    "P04_E020": {
        "main_data_stream_key": "10",
        "stream_name_norm": "10",
        "visual_stream_id": "10",
        "join_on_stream_name": "10",
        "mapping_confidence": "explicit_mapping",
        "mapping_note": "manual: SEP1 second outlet to OUT_stream mapped to stream 10 per PFD (vs PROD on OUT_PROD edge).",
    },
    "P07_E017": {
        "main_data_stream_key": "RE",
        "stream_name_norm": "RE",
        "visual_stream_id": "",
        "join_on_stream_name": "RE",
        "mapping_confidence": "explicit_mapping",
        "mapping_note": "manual: F1->OUT_H2O disambiguated to RE per PFD (vs PROD on OUT_PROD edge).",
    },
}

# Stable canonical edge IDs are externally persisted in checkpoints, metrics,
# and diagnostic exports.  When a bad source edge is removed, keep its ID as a
# tombstone instead of renumbering every later physical edge.
CANONICAL_EDGE_ID_TOMBSTONES_BEFORE: Dict[
    Tuple[int, str, str], Tuple[str, ...]
] = {
    (1, "SP1", "P1"): ("P01_E032",),
}

# Side-outlet / vent streams present in stream_key_inventory but absent from adjacency matrix edges.
# Each tuple: (process_id, src_node, dst_node, src_node_raw, dst_node_raw) — dst_node_raw must start with OUT_ so it collapses to V_OUTPUT.
# PFD side-outlet / vent stream names (Process_Streams Stream_Name) that must appear on an edge or be ignored.
EXPECTED_SIDE_STREAM_KEYS_BY_PID: Dict[int, Tuple[str, ...]] = {
    1: ("Z1", "Z2"),
    2: ("VENT",),
    3: ("Z1",),
    4: ("Z1", "Z2"),
    5: ("Z1",),
    6: ("Z1",),
    7: ("Z1", "Z2"),
    8: ("Z1", "Z2"),
    9: ("Z2",),
    10: ("Z1",),
}

SYNTHETIC_SIDE_STREAM_EDGES: List[Tuple[int, str, str, str, str]] = [
    (2, "R2", "V_OUTPUT", "R2", "OUT_VENT"),
    (3, "R2", "V_OUTPUT", "R2", "OUT_Z1"),
    (4, "R2", "V_OUTPUT", "R2", "OUT_Z1"),
    (5, "R2", "V_OUTPUT", "R2", "OUT_Z1"),
    (6, "R2", "V_OUTPUT", "R2", "OUT_Z1"),
    (7, "R2", "V_OUTPUT", "R2", "OUT_Z1"),
    (8, "R2", "V_OUTPUT", "R2", "OUT_Z1"),
    (9, "R2", "V_OUTPUT", "R2", "OUT_Z2"),
    (10, "R2", "V_OUTPUT", "R2", "OUT_Z1"),
]

# Visual labels on PFD that differ from adjacency raw unit names (does not change topology).
EXTRA_VISUAL_NODE_ALIASES: List[Dict[str, Any]] = [
    {
        "process_id": 1,
        "visual_node_name": "B2",
        "adjacency_node_name": "B2",
        "canonical_node_name": "MIX3",
        "alias_type": "visual_label",
        "confidence": "high",
        "note": "PFD label B2 corresponds to canonical unit MIX3 on Process1 diagram.",
    },
    {
        "process_id": 6,
        "visual_node_name": "R3",
        "adjacency_node_name": "R3",
        "canonical_node_name": "R3_BURNER",
        "alias_type": "visual_label",
        "confidence": "high",
        "note": "PFD reactor label R3 is the combined R3_BURNER unit in adjacency for Process6.",
    },
    {
        "process_id": 9,
        "visual_node_name": "B2",
        "adjacency_node_name": "B2",
        "canonical_node_name": "MIX1",
        "alias_type": "visual_label",
        "confidence": "high",
        "note": "PFD label B2 corresponds to canonical unit MIX1 on Process9 diagram.",
    },
    {
        "process_id": 10,
        "visual_node_name": "BN",
        "adjacency_node_name": "BN",
        "canonical_node_name": "BURNER",
        "alias_type": "visual_label",
        "confidence": "high",
        "note": "PFD label BN corresponds to canonical BURNER on Process10 diagram.",
    },
]

# Non-answer edges allowed to have empty main_data_stream_key (explicitly documented).
ALLOWED_EMPTY_STREAM_KEY_NON_ANSWER_EDGE_IDS: frozenset[str] = frozenset()


def _norm_name(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper().strip())


def _norm_stream_key_for_inventory(key: str) -> str:
    """Display norm for inventory; does not alter stored stream keys on edges."""
    return str(key).strip().upper()


def _is_input_node(name: str) -> bool:
    upper = str(name).upper()
    return any(upper.startswith(p) for p in INPUT_NODE_PATTERNS)


def _is_output_node(name: str) -> bool:
    upper = str(name).upper()
    return any(upper.startswith(p) for p in OUTPUT_NODE_PATTERNS)


def _to_canonical_node(raw_name: str) -> str:
    if _is_input_node(raw_name):
        return "V_INPUT"
    if _is_output_node(raw_name):
        return "V_OUTPUT"
    return str(raw_name).strip()


def _out_tail(dst_raw: str) -> str:
    """Strip leading OUT_ from raw sink name for disambiguation (case-insensitive)."""
    s = str(dst_raw).strip()
    if s.upper().startswith("OUT_"):
        return s[4:].strip()
    return s


def _disambiguate_mapping_rows(
    rows: Sequence[Mapping[str, Any]],
    dst_node_raw: str,
) -> List[Mapping[str, Any]]:
    """When multiple mapping rows share the same (process, src, dst), narrow using dst_node_raw and CSV columns only."""
    if len(rows) <= 1:
        return list(rows)

    tail = _out_tail(dst_node_raw).upper()
    dl = str(dst_node_raw).lower()

    def _one(cand: List[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
        return cand if len(cand) == 1 else list(rows)

    exact = [r for r in rows if str(r["stream_name"]).strip().upper() == tail]
    out = _one(exact)
    if len(out) == 1:
        return out

    exact_norm = [r for r in rows if str(r["stream_name_norm"]).strip().upper() == tail]
    out = _one(exact_norm)
    if len(out) == 1:
        return out

    if tail:
        sub = [r for r in rows if tail in str(r["stream_name"]).strip().upper()]
        out = _one(sub)
        if len(out) == 1:
            return out

    if "exhaust" in dl or "exhaus" in dl:
        r = [x for x in rows if str(x.get("stream_role", "")).strip().lower() == "exhaust"]
        out = _one(r)
        if len(out) == 1:
            return out

    if "prod" in dl and "co2" not in dl:
        r = [x for x in rows if str(x.get("stream_role", "")).strip().lower() == "product"]
        out = _one(r)
        if len(out) == 1:
            return out

    if "co2" in dl:
        r = [
            x
            for x in rows
            if "CO2" in str(x["stream_name"]).upper() or "CO2" in str(x.get("stream_name_norm", "")).upper()
        ]
        out = _one(r)
        if len(out) == 1:
            return out
        r_out = [x for x in rows if str(x.get("stream_role", "")).strip().lower() == "output"]
        out = _one(r_out)
        if len(out) == 1:
            return out

    if "resteam" in dl:
        r = [x for x in rows if "RESTEAM" in str(x["stream_name"]).upper()]
        out = _one(r)
        if len(out) == 1:
            return out

    if tail.upper() == "RE" or re.search(r"\bre\b", dl):
        r = [x for x in rows if str(x["stream_name"]).strip().upper() == "RE"]
        out = _one(r)
        if len(out) == 1:
            return out

    if "h2o" in dl or "water" in dl:
        r = [x for x in rows if "H2O" in str(x["stream_name"]).upper() or "WATER" in str(x["stream_name"]).upper()]
        out = _one(r)
        if len(out) == 1:
            return out

    return list(rows)


def _visual_stream_id_from_row(row: Mapping[str, Any]) -> str:
    if "visual_stream_id" in row and pd.notna(row.get("visual_stream_id")) and str(row["visual_stream_id"]).strip():
        return str(row["visual_stream_id"]).strip()
    sn = str(row["stream_name"]).strip()
    if sn and re.fullmatch(r"\d+", sn):
        return sn
    return ""


def load_stream_mapping(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            "process_id": int,
            "stream_name": str,
            "stream_name_norm": str,
            "src_node": str,
            "dst_node": str,
            "stream_role": str,
            "note": str,
        },
    )
    for col in ("src_node", "dst_node", "stream_name", "stream_name_norm"):
        df[col] = df[col].astype(str).str.strip()
    return df


def _mapping_row_dict(r: pd.Series) -> Dict[str, Any]:
    return {k: r[k] for k in r.index}


def collect_mapping_candidates(
    mapping_df: pd.DataFrame,
    process_id: int,
    src_node_raw: str,
    dst_node_raw: str,
    src_node: str,
    dst_node: str,
) -> List[Dict[str, Any]]:
    """Apply join priority: raw pair, then canonical pair, then normalized canonical pair."""
    pid = int(process_id)
    raw_s = str(src_node_raw).strip()
    raw_d = str(dst_node_raw).strip()
    c_s = str(src_node).strip()
    c_d = str(dst_node).strip()
    n_s, n_d = _norm_name(c_s), _norm_name(c_d)

    sub = mapping_df[mapping_df["process_id"] == pid]

    def _rows(mask: pd.Series) -> List[Dict[str, Any]]:
        return [_mapping_row_dict(r) for _, r in sub.loc[mask].iterrows()]

    m1 = (sub["src_node"] == raw_s) & (sub["dst_node"] == raw_d)
    r1 = _rows(m1)
    if len(r1) > 1:
        r1 = _disambiguate_mapping_rows(r1, dst_node_raw)
    if len(r1) == 1:
        return r1

    m2 = (sub["src_node"] == c_s) & (sub["dst_node"] == c_d)
    r2 = _rows(m2)
    if len(r2) > 1:
        r2 = _disambiguate_mapping_rows(r2, dst_node_raw)
    if len(r2) == 1:
        return r2

    sub_norm = sub.assign(
        _ns=sub["src_node"].map(_norm_name),
        _nd=sub["dst_node"].map(_norm_name),
    )
    m3 = (sub_norm["_ns"] == n_s) & (sub_norm["_nd"] == n_d)
    r3 = [_mapping_row_dict(r) for _, r in sub_norm.loc[m3].iterrows()]
    if len(r3) > 1:
        r3 = _disambiguate_mapping_rows(r3, dst_node_raw)
    if len(r3) == 1:
        return r3

    return []


def apply_explicit_stream_mapping(edges: List[dict], mapping_df: pd.DataFrame) -> None:
    """Mutates edges in place: fills stream fields or ambiguous / needs_review."""
    for e in edges:
        cands = collect_mapping_candidates(
            mapping_df,
            int(e["process_id"]),
            str(e["src_node_raw"]),
            str(e["dst_node_raw"]),
            str(e["src_node"]),
            str(e["dst_node"]),
        )
        if len(cands) == 1:
            row = cands[0]
            key = str(row["stream_name"])
            e["main_data_stream_key"] = key
            e["stream_name_norm"] = str(row.get("stream_name_norm", "") or _norm_stream_key_for_inventory(key))
            e["visual_stream_id"] = _visual_stream_id_from_row(row)
            e["join_on_stream_name"] = key
            e["mapping_confidence"] = "explicit_mapping"
            e["mapping_note"] = EXPLICIT_NOTE
        elif len(cands) > 1:
            keys = sorted({str(x["stream_name"]) for x in cands})
            e["main_data_stream_key"] = ""
            e["stream_name_norm"] = ""
            e["visual_stream_id"] = ""
            e["join_on_stream_name"] = "Stream_Name"
            e["mapping_confidence"] = "ambiguous"
            e["mapping_note"] = f"multiple explicit mappings found: {keys}"
        else:
            e["main_data_stream_key"] = ""
            e["stream_name_norm"] = ""
            e["visual_stream_id"] = ""
            e["join_on_stream_name"] = "Stream_Name"
            e["mapping_confidence"] = "needs_review"
            e["mapping_note"] = NO_MAPPING_NOTE


def build_stream_key_inventory(project_root: Path, streams_dir: Path) -> List[dict]:
    rows: List[dict] = []
    root_res = project_root.resolve()
    for pid in PROCESS_IDS:
        p = (streams_dir / f"{pid}.Process_Streams.csv").resolve()
        try:
            rel = p.relative_to(root_res).as_posix()
        except ValueError:
            rel = p.as_posix()
        df = pd.read_csv(p, usecols=["Stream_Name"], dtype={"Stream_Name": str})
        keys = df["Stream_Name"].astype(str).str.strip().unique().tolist()
        for key in sorted(keys, key=lambda x: (len(x), x)):
            rows.append(
                {
                    "process_id": pid,
                    "source_stream_file": rel.replace("\\", "/"),
                    "stream_key": key,
                    "stream_key_norm": _norm_stream_key_for_inventory(key),
                    "detected_format": "long",
                    "note": "Stream_Name column from Process_Streams.csv (wide mole-fraction columns per row).",
                }
            )
    return rows


def _inventory_key_set(inventory_rows: List[dict], pid: int) -> set[str]:
    return {str(r["stream_key"]) for r in inventory_rows if int(r["process_id"]) == pid}


def _max_edge_suffix(edges: Sequence[Mapping[str, Any]], process_id: int) -> int:
    pfx = f"P{int(process_id):02d}_E"
    mx = 0
    for e in edges:
        if int(e["process_id"]) != int(process_id):
            continue
        ce = str(e["canonical_edge_id"])
        if not ce.startswith(pfx):
            continue
        try:
            n = int(ce.split("_E", 1)[1])
        except (IndexError, ValueError):
            continue
        mx = max(mx, n)
    return mx


def _synthetic_side_stream_edge_dict(
    process_id: int,
    canonical_edge_id: str,
    src_node: str,
    dst_node: str,
    src_node_raw: str,
    dst_node_raw: str,
) -> dict:
    is_input_edge = src_node == "V_INPUT" and dst_node not in {"V_INPUT", "V_OUTPUT"}
    is_output_edge = dst_node == "V_OUTPUT" and src_node not in {"V_INPUT", "V_OUTPUT"}
    is_internal_edge = (not is_input_edge) and (not is_output_edge)
    return {
        "process_id": process_id,
        "canonical_edge_id": canonical_edge_id,
        "src_node": src_node,
        "dst_node": dst_node,
        "src_node_raw": src_node_raw,
        "dst_node_raw": dst_node_raw,
        "visual_stream_id": "",
        "main_data_stream_key": "",
        "stream_name_norm": "",
        "stream_role": "input" if is_input_edge else ("output" if is_output_edge else "internal"),
        "is_input_edge": bool(is_input_edge),
        "is_output_edge": bool(is_output_edge),
        "is_internal_edge": bool(is_internal_edge),
        "join_on_process_id": str(process_id),
        "join_on_id": "ID",
        "join_on_stream_name": "Stream_Name",
        "requires_ID": True,
        "mapping_confidence": "needs_review",
        "mapping_note": NO_MAPPING_NOTE,
    }


def append_synthetic_side_stream_edges(edges: List[dict]) -> None:
    """Add PFD side-outlet / vent edges not present as non-zero cells in the adjacency matrix."""
    for pid, src, dst, sraw, draw in SYNTHETIC_SIDE_STREAM_EDGES:
        nxt = _max_edge_suffix(edges, pid) + 1
        cid = f"P{int(pid):02d}_E{nxt:03d}"
        edges.append(_synthetic_side_stream_edge_dict(pid, cid, src, dst, sraw, draw))


def apply_reference_edge_overrides(edges: List[dict]) -> None:
    by_id = {str(e["canonical_edge_id"]): e for e in edges}
    for eid, patch in REFERENCE_EDGE_OVERRIDES.items():
        tgt = by_id.get(eid)
        if tgt is None:
            raise KeyError(f"REFERENCE_EDGE_OVERRIDES references unknown canonical_edge_id={eid}")
        for k, v in patch.items():
            tgt[k] = v


def merge_extra_visual_aliases(aliases: List[dict]) -> None:
    seen = {(int(a["process_id"]), str(a["visual_node_name"]), str(a["canonical_node_name"])) for a in aliases}
    for row in EXTRA_VISUAL_NODE_ALIASES:
        key = (int(row["process_id"]), str(row["visual_node_name"]), str(row["canonical_node_name"]))
        if key in seen:
            continue
        aliases.append(dict(row))
        seen.add(key)


def load_ignored_stream_keys_csv(path: Path) -> Dict[int, set[str]]:
    if not path.is_file():
        return {}
    df = pd.read_csv(path, dtype={"process_id": int, "stream_key": str, "reason": str})
    out: Dict[int, set[str]] = defaultdict(set)
    for _, r in df.iterrows():
        if pd.isna(r.get("stream_key")) or str(r["stream_key"]).strip() == "":
            continue
        out[int(r["process_id"])].add(str(r["stream_key"]).strip())
    return dict(out)


def append_stream_inventory_coverage_checks(
    validations: List[dict],
    edges_df: pd.DataFrame,
    inventory_rows: List[dict],
    ignored_by_pid: Dict[int, set[str]],
) -> None:
    inv_by_pid: Dict[int, set[str]] = defaultdict(set)
    for r in inventory_rows:
        inv_by_pid[int(r["process_id"])].add(str(r["stream_key"]))

    all_missing: List[str] = []
    for pid in PROCESS_IDS:
        inv = inv_by_pid[pid]
        sub_e = edges_df[edges_df["process_id"] == pid]
        keys_on_edges: set[str] = set()
        for _, r in sub_e.iterrows():
            k = str(r["main_data_stream_key"]).strip()
            if k:
                keys_on_edges.add(k)
        ign = ignored_by_pid.get(pid, set())
        missing = sorted(inv - keys_on_edges - ign)
        for k in missing:
            all_missing.append(f"{pid}:{k}")

        inv_count = len(inv)
        canon_key_count = len(keys_on_edges)
        ign_count = len(ign)
        st_cov = "PASS" if not missing else "ERROR"
        detail_cov = (
            f"process_id={pid} stream_key_inventory_count={inv_count} "
            f"canonical_edges_nonempty_stream_key_count={canon_key_count} "
            f"ignored_stream_keys_count={ign_count} "
            f"inventory_keys_missing_from_canonical_edges={missing}"
        )
        _append_validation(validations, pid, "stream_inventory_key_coverage", st_cov, detail_cov)
        _append_validation(
            validations,
            pid,
            "inventory_keys_missing_from_canonical_edges",
            st_cov,
            f"missing={missing}" if missing else "missing=[]",
        )
        _append_validation(
            validations,
            pid,
            "ignored_stream_keys_count",
            "INFO",
            f"count={ign_count}",
        )
        exp_side = EXPECTED_SIDE_STREAM_KEYS_BY_PID.get(pid, ())
        missing_side = sorted(set(exp_side) - keys_on_edges - ign)
        st_side = "PASS" if not missing_side else "ERROR"
        side_detail = (
            f"expected_side_keys={list(exp_side)} missing_not_on_edge_nor_ignored={missing_side}"
            if missing_side
            else f"expected_side_keys={list(exp_side)} ok (each on an edge as main_data_stream_key or listed ignored)"
        )
        _append_validation(validations, pid, "side_outlet_streams_mapped_or_ignored", st_side, side_detail)

    st_all = "PASS" if not all_missing else "ERROR"
    _append_validation(
        validations,
        "ALL",
        "stream_inventory_key_coverage",
        st_all,
        f"missing={all_missing}" if all_missing else "missing=[]",
    )
    total_ignored = sum(len(v) for v in ignored_by_pid.values())
    _append_validation(
        validations,
        "ALL",
        "ignored_stream_keys_count",
        "INFO",
        f"count={total_ignored}",
    )
    _append_validation(
        validations,
        "ALL",
        "inventory_keys_missing_from_canonical_edges",
        st_all,
        f"missing={all_missing}" if all_missing else "missing=[]",
    )
    all_missing_side: List[str] = []
    for pid in PROCESS_IDS:
        sub_e = edges_df[edges_df["process_id"] == pid]
        keys_on_edges = {str(x).strip() for x in sub_e["main_data_stream_key"] if str(x).strip()}
        ign = ignored_by_pid.get(pid, set())
        exp_side = EXPECTED_SIDE_STREAM_KEYS_BY_PID.get(pid, ())
        for k in sorted(set(exp_side) - keys_on_edges - ign):
            all_missing_side.append(f"{pid}:{k}")
    st_side_all = "PASS" if not all_missing_side else "ERROR"
    _append_validation(
        validations,
        "ALL",
        "side_outlet_streams_mapped_or_ignored",
        st_side_all,
        f"missing={all_missing_side}" if all_missing_side else "missing=[]",
    )


@dataclass
class BuildArtifacts:
    nodes: List[dict]
    edges: List[dict]
    aliases: List[dict]
    answers: List[dict]


def build_process(process_id: int, adjacency_path: Path) -> BuildArtifacts:
    ptag = f"P{process_id:02d}"
    node_sheet = pd.read_excel(adjacency_path, sheet_name=1, header=None)
    node_sheet = node_sheet.rename(columns={0: "raw_node_name", 1: "raw_node_type"})
    node_rows = node_sheet[["raw_node_name", "raw_node_type"]].dropna(subset=["raw_node_name"])
    node_rows["raw_node_name"] = node_rows["raw_node_name"].astype(str).str.strip()
    node_rows["raw_node_type"] = node_rows["raw_node_type"].astype(str).str.strip()
    node_rows = node_rows[node_rows["raw_node_name"] != ""]
    node_rows = node_rows.drop_duplicates(subset=["raw_node_name"], keep="first")

    adjacency_df = pd.read_excel(adjacency_path, sheet_name=0, index_col=0)
    adjacency_df.index = adjacency_df.index.map(lambda x: str(x).strip())
    adjacency_df.columns = [str(x).strip() for x in adjacency_df.columns]

    nodes: List[dict] = []
    aliases: List[dict] = []

    canonical_names: Dict[str, str] = {}
    for raw_name in node_rows["raw_node_name"].tolist():
        canonical_name = _to_canonical_node(raw_name)
        canonical_names[raw_name] = canonical_name
        if canonical_name in {"V_INPUT", "V_OUTPUT"}:
            continue
        ntype_raw = node_rows.loc[node_rows["raw_node_name"] == raw_name, "raw_node_type"].iloc[0]
        nodes.append(
            {
                "process_id": process_id,
                "node_id": f"{ptag}_N_{_norm_name(canonical_name)}",
                "node_name": canonical_name,
                "node_type": "unit",
                "unit_type": ntype_raw,
                "raw_node_name": raw_name,
                "visual_node_name": raw_name,
                "is_virtual": False,
                "source_file": str(adjacency_path).replace("\\", "/"),
                "mapping_confidence": "high",
                "mapping_note": "unit node copied from adjacency node spec",
            }
        )
        aliases.append(
            {
                "process_id": process_id,
                "visual_node_name": raw_name,
                "adjacency_node_name": raw_name,
                "canonical_node_name": canonical_name,
                "alias_type": "identity",
                "confidence": "high",
                "note": "unit node identity mapping",
            }
        )

    nodes.extend(
        [
            {
                "process_id": process_id,
                "node_id": f"{ptag}_N_V_INPUT",
                "node_name": "V_INPUT",
                "node_type": "virtual",
                "unit_type": "virtual_io",
                "raw_node_name": "INPUT_*",
                "visual_node_name": "INPUT_*",
                "is_virtual": True,
                "source_file": str(adjacency_path).replace("\\", "/"),
                "mapping_confidence": "high",
                "mapping_note": "collapsed all external feed nodes",
            },
            {
                "process_id": process_id,
                "node_id": f"{ptag}_N_V_OUTPUT",
                "node_name": "V_OUTPUT",
                "node_type": "virtual",
                "unit_type": "virtual_io",
                "raw_node_name": "OUT_*",
                "visual_node_name": "OUT_*",
                "is_virtual": True,
                "source_file": str(adjacency_path).replace("\\", "/"),
                "mapping_confidence": "high",
                "mapping_note": "collapsed all external output nodes",
            },
        ]
    )

    input_nodes = [r for r in canonical_names if _is_input_node(r)]
    output_nodes = [r for r in canonical_names if _is_output_node(r)]
    for raw_name in input_nodes:
        aliases.append(
            {
                "process_id": process_id,
                "visual_node_name": raw_name,
                "adjacency_node_name": raw_name,
                "canonical_node_name": "V_INPUT",
                "alias_type": "collapse_feed",
                "confidence": "high",
                "note": "external feed collapsed to V_INPUT",
            }
        )
    for raw_name in output_nodes:
        aliases.append(
            {
                "process_id": process_id,
                "visual_node_name": raw_name,
                "adjacency_node_name": raw_name,
                "canonical_node_name": "V_OUTPUT",
                "alias_type": "collapse_output",
                "confidence": "high",
                "note": "external output collapsed to V_OUTPUT",
            }
        )

    edges: List[dict] = []
    edge_idx = 1
    for src in adjacency_df.index:
        for dst in adjacency_df.columns:
            val = adjacency_df.loc[src, dst]
            if pd.isna(val) or float(val) == 0.0:
                continue
            for tombstone_id in CANONICAL_EDGE_ID_TOMBSTONES_BEFORE.get(
                (process_id, str(src), str(dst)), ()
            ):
                expected_id = f"{ptag}_E{edge_idx:03d}"
                if tombstone_id != expected_id:
                    raise RuntimeError(
                        "Canonical edge ID tombstone is out of sequence: "
                        f"expected={expected_id} configured={tombstone_id} "
                        f"before=({src},{dst})."
                    )
                edge_idx += 1
            src_can = _to_canonical_node(src)
            dst_can = _to_canonical_node(dst)
            is_input_edge = src_can == "V_INPUT" and dst_can not in {"V_INPUT", "V_OUTPUT"}
            is_output_edge = dst_can == "V_OUTPUT" and src_can not in {"V_INPUT", "V_OUTPUT"}
            is_internal_edge = src_can not in {"V_INPUT", "V_OUTPUT"} and dst_can not in {"V_INPUT", "V_OUTPUT"}
            edges.append(
                {
                    "process_id": process_id,
                    "canonical_edge_id": f"{ptag}_E{edge_idx:03d}",
                    "src_node": src_can,
                    "dst_node": dst_can,
                    "src_node_raw": src,
                    "dst_node_raw": dst,
                    "visual_stream_id": "",
                    "main_data_stream_key": "",
                    "stream_name_norm": "",
                    "stream_role": "input" if is_input_edge else ("output" if is_output_edge else "internal"),
                    "is_input_edge": bool(is_input_edge),
                    "is_output_edge": bool(is_output_edge),
                    "is_internal_edge": bool(is_internal_edge),
                    "join_on_process_id": str(process_id),
                    "join_on_id": "ID",
                    "join_on_stream_name": "Stream_Name",
                    "requires_ID": True,
                    "mapping_confidence": "needs_review",
                    "mapping_note": NO_MAPPING_NOTE,
                }
            )
            edge_idx += 1

    answer_rows: List[dict] = []
    process_rules = TARGET_RULES[process_id]
    for task_name, (raw_out_node, target_col) in process_rules.items():
        candidates = [
            e for e in edges if e["dst_node_raw"] == raw_out_node and e["dst_node"] == "V_OUTPUT" and e["is_output_edge"]
        ]
        if len(candidates) == 1:
            selected = candidates[0]
            conf = "high"
            note = "matched by fixed output node rule"
        elif len(candidates) == 0:
            selected = None
            conf = "error"
            note = f"no output edge found for raw output node {raw_out_node}"
        else:
            selected = candidates[0]
            conf = "needs_review"
            note = f"multiple candidate edges for {raw_out_node}; first selected (adjacency-only); verify with mapping CSV"

        answer_rows.append(
            {
                "process_id": process_id,
                "task_name": task_name,
                "canonical_answer_edge_id": selected["canonical_edge_id"] if selected else "",
                "target_column": target_col,
                "answer_src_node": selected["src_node"] if selected else "",
                "answer_dst_node": "V_OUTPUT",
                "must_satisfy_dst_node": "V_OUTPUT",
                "main_data_stream_key": "",
                "mapping_confidence": conf,
                "mapping_note": note,
            }
        )

    return BuildArtifacts(nodes=nodes, edges=edges, aliases=aliases, answers=answer_rows)


def _append_validation(rows: List[dict], process_id: Any, check: str, status: str, detail: str) -> None:
    rows.append({"process_id": process_id, "check_name": check, "status": status, "detail": detail})


def sync_answer_stream_keys(answers: List[dict], edges: List[dict]) -> None:
    edge_map = {(int(e["process_id"]), e["canonical_edge_id"]): e for e in edges}
    for a in answers:
        e = edge_map.get((int(a["process_id"]), str(a["canonical_answer_edge_id"])))
        if e is None:
            continue
        key = str(e.get("main_data_stream_key", "") or "").strip()
        a["main_data_stream_key"] = key
        if a.get("mapping_confidence") == "error":
            continue
        econf = str(e.get("mapping_confidence", "") or "")
        if key and econf == "explicit_mapping":
            a["mapping_confidence"] = "explicit_mapping"
            a["mapping_note"] = f"{a.get('mapping_note', '')}; stream key from canonical_edges".strip("; ")
        elif not key:
            a["mapping_confidence"] = "needs_review"
            a["mapping_note"] = (
                f"{a.get('mapping_note', '')}; answer edge has empty main_data_stream_key".strip("; ")
            )
        elif econf == "ambiguous":
            a["mapping_confidence"] = "needs_review"
            a["mapping_note"] = f"{a.get('mapping_note', '')}; canonical answer edge has ambiguous stream mapping".strip(
                "; "
            )


def build_edges_needs_review_extract(edges_df: pd.DataFrame, answers_df: pd.DataFrame) -> pd.DataFrame:
    """Edges with mapping_confidence needs_review or ambiguous, plus is_answer_edge flag."""
    ans_keys = answers_df[["process_id", "canonical_answer_edge_id"]].drop_duplicates()
    ans_keys = ans_keys.rename(columns={"canonical_answer_edge_id": "canonical_edge_id"})
    ans_keys["is_answer_edge"] = True
    sub = edges_df[edges_df["mapping_confidence"].isin(["needs_review", "ambiguous"])].copy()
    sub = sub.merge(ans_keys, on=["process_id", "canonical_edge_id"], how="left")
    sub["is_answer_edge"] = sub["is_answer_edge"].eq(True)
    cols = [
        "process_id",
        "canonical_edge_id",
        "src_node",
        "dst_node",
        "src_node_raw",
        "dst_node_raw",
        "stream_role",
        "is_input_edge",
        "is_output_edge",
        "is_internal_edge",
        "main_data_stream_key",
        "mapping_confidence",
        "mapping_note",
        "is_answer_edge",
    ]
    out = sub[cols].copy()
    out["main_data_stream_key"] = out["main_data_stream_key"].fillna("").astype(str)
    return out.sort_values(["process_id", "canonical_edge_id"])


def append_training_readiness_validations(
    validations: List[dict],
    edges_df: pd.DataFrame,
    answers_df: pd.DataFrame,
) -> None:
    """Reference-consumer / future dataloader contract checks (process_id ALL)."""
    e = edges_df.copy()
    e["_key_nonempty"] = e["main_data_stream_key"].fillna("").astype(str).str.strip().ne("")
    ans_keys = answers_df[["process_id", "canonical_answer_edge_id"]].drop_duplicates()
    ans_keys = ans_keys.rename(columns={"canonical_answer_edge_id": "canonical_edge_id"})
    ans_keys["_is_answer"] = True
    e = e.merge(ans_keys, on=["process_id", "canonical_edge_id"], how="left")
    e["_is_answer"] = e["_is_answer"].eq(True)

    nr_ans = e[e["_is_answer"] & e["mapping_confidence"].isin(["needs_review", "ambiguous"])]
    n_nr_ans = len(nr_ans)
    _append_validation(
        validations,
        "ALL",
        "needs_review_answer_edge_count",
        "PASS" if n_nr_ans == 0 else "ERROR",
        f"count={n_nr_ans} expected=0",
    )

    n_ans_key = int(answers_df["main_data_stream_key"].fillna("").astype(str).str.strip().ne("").sum())
    _append_validation(
        validations,
        "ALL",
        "answer_edges_with_stream_key",
        "PASS" if n_ans_key == 20 else "ERROR",
        f"count={n_ans_key} expected=20",
    )

    allowed_ids = ALLOWED_EMPTY_STREAM_KEY_NON_ANSWER_EDGE_IDS
    allowed_mask = e["canonical_edge_id"].astype(str).isin(allowed_ids)
    bad_unmapped = (~e["_key_nonempty"]) & (~e["_is_answer"]) & (~allowed_mask)
    n_bad_unmapped = int(bad_unmapped.sum())
    n_allowed_empty = int(((~e["_key_nonempty"]) & (~e["_is_answer"]) & allowed_mask).sum())
    _append_validation(
        validations,
        "ALL",
        "non_answer_edges_without_stream_key",
        "PASS" if n_bad_unmapped == 0 else "ERROR",
        f"unexpected_empty_non_answer={n_bad_unmapped} allowed_documented_empty={n_allowed_empty} "
        f"allowed_ids={sorted(allowed_ids)}",
    )

    bad_empty_on_answer = (~e["_key_nonempty"]) & e["_is_answer"]
    n_bad = int(bad_empty_on_answer.sum())
    _append_validation(
        validations,
        "ALL",
        "missing_stream_key_allowed_only_when_not_answer_edge",
        "PASS" if n_bad == 0 else "ERROR",
        f"answer_edges_with_empty_main_data_stream_key={n_bad} expected=0",
    )


def run_validations(
    validations: List[dict],
    nodes: List[dict],
    edges: List[dict],
    answers: List[dict],
    inventory_rows: List[dict],
    *,
    out_dir: Optional[Path] = None,
) -> None:
    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges)
    answers_df = pd.DataFrame(answers)

    inv_by_pid: Dict[int, set[str]] = defaultdict(set)
    for r in inventory_rows:
        inv_by_pid[int(r["process_id"])].add(str(r["stream_key"]))

    for pid in PROCESS_IDS:
        sub_e = edges_df[edges_df["process_id"] == pid]
        explicit = int((sub_e["mapping_confidence"] == "explicit_mapping").sum())
        amb = int((sub_e["mapping_confidence"] == "ambiguous").sum())
        nrev = int((sub_e["mapping_confidence"] == "needs_review").sum())
        unmapped = int(sub_e["main_data_stream_key"].astype(str).str.strip().eq("").sum())
        mapped = int(sub_e["main_data_stream_key"].astype(str).str.strip().ne("").sum())
        inv = inv_by_pid[pid]
        bad_inv = 0
        for _, r in sub_e.iterrows():
            k = str(r["main_data_stream_key"]).strip()
            if not k:
                continue
            if k not in inv:
                bad_inv += 1

        _append_validation(validations, pid, "mapped_edge_count", "INFO", f"count={mapped}")
        _append_validation(validations, pid, "unmapped_edge_count", "INFO", f"count={unmapped}")
        _append_validation(validations, pid, "ambiguous_edge_count", "INFO", f"count={amb}")
        _append_validation(
            validations,
            pid,
            "main_data_stream_key_inventory_missing_count",
            "ERROR" if bad_inv else "PASS",
            f"missing_count={bad_inv}",
        )
        _append_validation(validations, pid, "needs_review_count", "INFO", f"count={nrev}")
        _append_validation(validations, pid, "explicit_mapping_count", "INFO", f"count={explicit}")

        sub_a = answers_df[answers_df["process_id"] == pid]
        ans_with = int(sub_a["main_data_stream_key"].astype(str).str.strip().ne("").sum())
        _append_validation(
            validations,
            pid,
            "answer_edge_has_main_data_stream_key",
            "WARN" if ans_with < len(sub_a) else "PASS",
            f"with_key={ans_with} total={len(sub_a)}",
        )

        edge_ids = set(sub_e["canonical_edge_id"].astype(str))
        missing_ans = [t for t, ce in zip(sub_a["task_name"], sub_a["canonical_answer_edge_id"]) if ce not in edge_ids]
        _append_validation(
            validations,
            pid,
            "target_answer_edge_id_in_canonical_edges",
            "ERROR" if missing_ans else "PASS",
            "ok" if not missing_ans else f"missing={missing_ans}",
        )
        bad_dst = sub_a[sub_a["answer_dst_node"] != "V_OUTPUT"]
        _append_validation(
            validations,
            pid,
            "target_answer_edge_dst_v_output",
            "ERROR" if len(bad_dst) else "PASS",
            "ok" if not len(bad_dst) else f"rows={bad_dst.index.tolist()}",
        )

        nonempty_keys = [str(x).strip() for x in sub_e["main_data_stream_key"] if str(x).strip()]
        missing_keys = sorted({k for k in nonempty_keys if k not in inv})
        _append_validation(
            validations,
            pid,
            "nonempty_main_data_stream_key_in_inventory",
            "ERROR" if missing_keys else "PASS",
            "ok" if not missing_keys else f"missing_keys={missing_keys}",
        )

    # graph structure (global checks per process already above); add globals
    dup_edge = edges_df.duplicated(subset=["process_id", "canonical_edge_id"]).any()
    _append_validation(validations, "ALL", "canonical_edge_id_unique", "ERROR" if dup_edge else "PASS", "checked")

    dup_node = nodes_df.duplicated(subset=["process_id", "node_id"]).any()
    _append_validation(validations, "ALL", "canonical_node_id_unique", "ERROR" if dup_node else "PASS", "checked")

    node_names = nodes_df.groupby("process_id")["node_name"].apply(set).to_dict()
    for pid in PROCESS_IDS:
        names = node_names.get(pid, set())
        sub_e = edges_df[edges_df["process_id"] == pid]
        bad = sub_e[(~sub_e["src_node"].isin(names)) | (~sub_e["dst_node"].isin(names))]
        _append_validation(
            validations,
            pid,
            "edge_endpoints_in_nodes",
            "ERROR" if len(bad) else "PASS",
            f"bad_count={len(bad)}",
        )

    if len(answers_df) != 20:
        _append_validation(validations, "ALL", "target_answer_row_count", "ERROR", f"count={len(answers_df)} expected=20")
    else:
        _append_validation(validations, "ALL", "target_answer_row_count", "PASS", "count=20")

    p7 = answers_df[answers_df["process_id"] == 7]
    if len(p7) == 2 and p7["canonical_answer_edge_id"].nunique() == 1:
        _append_validation(validations, "ALL", "process7_shared_answer_edge", "PASS", "shared")
    else:
        _append_validation(validations, "ALL", "process7_shared_answer_edge", "ERROR", "Process7 must share one edge id")

    bad_ans_dst = answers_df[answers_df["answer_dst_node"].astype(str) != "V_OUTPUT"]
    _append_validation(
        validations,
        "ALL",
        "answer_edges_dst_node_is_v_output",
        "ERROR" if len(bad_ans_dst) else "PASS",
        "ok" if not len(bad_ans_dst) else f"bad_rows={bad_ans_dst[['process_id','task_name','answer_dst_node']].to_dict('records')}",
    )

    ignored_path = (out_dir if out_dir is not None else Path()) / "ignored_stream_keys.csv"
    ignored_by_pid = load_ignored_stream_keys_csv(ignored_path)
    append_stream_inventory_coverage_checks(validations, edges_df, inventory_rows, ignored_by_pid)

    append_training_readiness_validations(validations, edges_df, answers_df)


def build_readme_rules() -> pd.DataFrame:
    rules = [
        "Graph construction: feed stream nodes (INPUT_*/IN_*) collapse to V_INPUT; OUT_* collapse to V_OUTPUT.",
        "Canonical graph nodes are only V_INPUT, V_OUTPUT, and physical unit nodes.",
        "canonical_edge_id (e.g. P01_E001) is independent from stream numbering; never infer stream id from edge id order.",
        "canonical_edges.main_data_stream_key comes from process_1_10_stream_edge_mapping_final.csv join; join Process_Streams on process_id + ID + Stream_Name == main_data_stream_key.",
        "Do not drop canonical_edges rows when main_data_stream_key is empty: keep them as topology-only edges in edge_index.",
        "Do not join Process_Streams stream features for edges with empty main_data_stream_key.",
        "When building edge_attr, treat empty main_data_stream_key as edge feature missing (no stream row join).",
        "Dataloader must build edge_feature_mask: 1 if main_data_stream_key non-empty and stream feature join succeeds, else 0.",
        "For edge_feature_mask==0: keep edge in edge_index; use zero edge_attr (or learned missing embedding later). Preferred v1: concatenate [edge_attr | edge_feature_mask] with zero vector for missing.",
        "target_answer_edges.canonical_answer_edge_id must exist in canonical_edges.canonical_edge_id (same process_id).",
        "Never select target_h2 / tailgas_co2 readout edges by heuristic; always use target_answer_edges.csv.",
        "Current spec: needs_review/ambiguous edges listed in edges_needs_review_extract.csv are not answer edges (no direct target/tailgas readout impact) and must not receive Process_Streams joins until mapped.",
        "Ambiguous mapping_confidence on an edge means multiple mapping rows matched; do not pick arbitrarily—fix mapping source or disambiguation first.",
        "validation_checks.csv includes training-readiness checks: needs_review_answer_edge_count==0, answer_edges_with_stream_key==20, non_answer_edges_without_stream_key, and no empty stream key on answer edges.",
        "Reverse inventory check: every Stream_Name in stream_key_inventory for a process must appear as main_data_stream_key on at least one canonical_edges row for that process, unless listed in ignored_stream_keys.csv (process_id, stream_key, reason).",
        "stream_inventory_key_coverage and inventory_keys_missing_from_canonical_edges (per process and ALL) require every stream_key_inventory key to appear as main_data_stream_key on some canonical_edges row for that process, unless ignored_stream_keys.csv lists it.",
        "side_outlet_streams_mapped_or_ignored checks Z1/Z2/VENT (per process) the same way. ignored_stream_keys_count reports how many keys are intentionally excluded per process (ALL = sum).",
        "Side-outlet streams (Z1, Z2, VENT, etc.) not present in the adjacency matrix are appended as synthetic canonical_edges after adjacency build; they are then mapped via process_1_10_stream_edge_mapping_final.csv where possible.",
    ]
    return pd.DataFrame(
        [{"rule_id": f"R{i:02d}", "rule_text": txt, "source": "reference_v3_consumer_spec"} for i, txt in enumerate(rules, start=1)]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build canonical graph reference spec v3.")
    parser.add_argument("--adjacency-dir", default="data/adjacency_matrix")
    parser.add_argument("--streams-dir", default="data/main_data_Streams")
    parser.add_argument(
        "--stream-edge-mapping",
        default="data/main_data_Streams/process_1_10_stream_edge_mapping_final.csv",
    )
    parser.add_argument("--out-dir", default="data/reference/v3")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    adjacency_dir = (root / args.adjacency_dir).resolve()
    streams_dir = (root / args.streams_dir).resolve()
    mapping_path = (root / args.stream_edge_mapping).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping_df = load_stream_mapping(mapping_path)
    print("=== process_1_10_stream_edge_mapping_final.csv ===")
    print("columns:", list(mapping_df.columns))
    print(mapping_df.head(10).to_string(index=False))
    print()

    all_nodes: List[dict] = []
    all_edges: List[dict] = []
    all_aliases: List[dict] = []
    all_answers: List[dict] = []

    for pid in PROCESS_IDS:
        adj = adjacency_dir / f"Process{pid}_Adjacency_Matrix.xlsx"
        if not adj.exists():
            raise FileNotFoundError(f"Missing adjacency file: {adj}")
        built = build_process(pid, adj)
        all_nodes.extend(built.nodes)
        all_edges.extend(built.edges)
        all_aliases.extend(built.aliases)
        all_answers.extend(built.answers)

    append_synthetic_side_stream_edges(all_edges)
    apply_explicit_stream_mapping(all_edges, mapping_df)
    apply_reference_edge_overrides(all_edges)
    sync_answer_stream_keys(all_answers, all_edges)
    merge_extra_visual_aliases(all_aliases)

    inventory_rows = build_stream_key_inventory(root, streams_dir)
    validations: List[dict] = []

    ignored_csv = out_dir / "ignored_stream_keys.csv"
    if not ignored_csv.exists():
        ignored_csv.write_text("process_id,stream_key,reason\n", encoding="utf-8")

    run_validations(validations, all_nodes, all_edges, all_answers, inventory_rows, out_dir=out_dir)

    nodes_df = pd.DataFrame(all_nodes).sort_values(["process_id", "node_id"])
    edges_df = pd.DataFrame(all_edges).sort_values(["process_id", "canonical_edge_id"])
    answers_df = pd.DataFrame(all_answers).sort_values(["process_id", "task_name"])
    aliases_df = pd.DataFrame(all_aliases).sort_values(["process_id", "alias_type", "adjacency_node_name"])
    inventory_df = pd.DataFrame(inventory_rows).sort_values(["process_id", "stream_key"])
    validation_df = pd.DataFrame(validations)
    needs_review_extract_df = build_edges_needs_review_extract(edges_df, answers_df)

    nodes_df.to_csv(out_dir / "canonical_nodes.csv", index=False, encoding="utf-8")
    edges_df.to_csv(out_dir / "canonical_edges.csv", index=False, encoding="utf-8")
    answers_df.to_csv(out_dir / "target_answer_edges.csv", index=False, encoding="utf-8")
    aliases_df.to_csv(out_dir / "node_aliases.csv", index=False, encoding="utf-8")
    inventory_df.to_csv(out_dir / "stream_key_inventory.csv", index=False, encoding="utf-8")
    validation_df.to_csv(out_dir / "validation_checks.csv", index=False, encoding="utf-8")
    build_readme_rules().to_csv(out_dir / "README_rules.csv", index=False, encoding="utf-8")
    needs_review_extract_df.to_csv(out_dir / "edges_needs_review_extract.csv", index=False, encoding="utf-8")

    val_df = validation_df
    err_n = int((val_df["status"] == "ERROR").sum())
    warn_n = int((val_df["status"] == "WARN").sum())
    total_explicit = int((edges_df["mapping_confidence"] == "explicit_mapping").sum())
    total_needs = int((edges_df["mapping_confidence"] == "needs_review").sum())
    total_amb = int((edges_df["mapping_confidence"] == "ambiguous").sum())

    for pid in PROCESS_IDS:
        sub = edges_df[edges_df["process_id"] == pid]
        n_e = len(sub)
        n_exp = int((sub["mapping_confidence"] == "explicit_mapping").sum())
        n_nr = int((sub["mapping_confidence"] == "needs_review").sum())
        n_am = int((sub["mapping_confidence"] == "ambiguous").sum())
        inv_miss_row = val_df[
            (val_df["process_id"] == pid) & (val_df["check_name"] == "main_data_stream_key_inventory_missing_count")
        ]
        inv_miss = 0
        if len(inv_miss_row):
            m = re.search(r"missing_count=(\d+)", str(inv_miss_row.iloc[0]["detail"]))
            inv_miss = int(m.group(1)) if m else 0
        sub_a = answers_df[answers_df["process_id"] == pid]
        n_ans = len(sub_a)
        n_ans_key = int(sub_a["main_data_stream_key"].astype(str).str.strip().ne("").sum())
        print(f"Process{pid}:")
        print(f"  canonical_edges = {n_e}")
        print(f"  explicit_mapping = {n_exp}")
        print(f"  needs_review = {n_nr}")
        print(f"  ambiguous = {n_am}")
        print(f"  stream_inventory_missing = {inv_miss}")
        print(f"  answer_edges = {n_ans}")
        print(f"  answer_edges_with_stream_key = {n_ans_key}")
        print()

    print("TOTAL:")
    print(f"  validation_errors = {err_n}")
    print(f"  validation_warnings = {warn_n}")
    print(f"  explicit_mapping = {total_explicit}")
    print(f"  needs_review = {total_needs}")
    print(f"  ambiguous = {total_amb}")


if __name__ == "__main__":
    main()
