#!/usr/bin/env python3
"""Validate v4 target formulas: Process_Main targets vs Process_Streams-derived amounts."""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ABS_TOL = 1e-6
REL_TOL = 1e-5

SPECIES_FRAC = {"H2": "Frac_H2", "CO2": "Frac_CO2", "H2O": "Frac_H2O"}

# Explicit v4 target formula table (processes 1–10).
TARGET_FORMULAS: tuple[dict[str, Any], ...] = (
    # Process 1
    {
        "process_id": 1,
        "target_id": "P01_T001",
        "target_stream": "OUT_exhaust",
        "target_species": "CO2",
        "target_feature": "FUELGAS_CO2_Mole",
        "formula": "Mole_Flow * Frac_CO2",
        "scale": 1.0,
        "frac_column": "Frac_CO2",
        "special_case": False,
        "reference_edge_id": "P01_E015",
        "required_stream_key": "FUELGAS",
    },
    {
        "process_id": 1,
        "target_id": "P01_T002",
        "target_stream": "OUT_prod",
        "target_species": "H2",
        "target_feature": "PROD_H2_Mole",
        "formula": "Mole_Flow * Frac_H2",
        "scale": 1.0,
        "frac_column": "Frac_H2",
        "special_case": False,
        "reference_edge_id": "P01_E021",
        "required_stream_key": "PROD",
    },
    {
        "process_id": 1,
        "target_id": "P01_T003",
        "target_stream": "OUT_prod",
        "target_species": "CO2",
        "target_feature": "PROD_CO2_Mole",
        "formula": "Mole_Flow * Frac_CO2",
        "scale": 1.0,
        "frac_column": "Frac_CO2",
        "special_case": False,
        "reference_edge_id": "P01_E021",
        "required_stream_key": "PROD",
    },
    {
        "process_id": 1,
        "target_id": "P01_T004",
        "target_stream": "OUT_RESTEAM",
        "target_species": "H2O",
        "target_feature": "RESTEAM_H2O_Mole",
        "formula": "Mole_Flow * Frac_H2O",
        "scale": 1.0,
        "frac_column": "Frac_H2O",
        "special_case": False,
        "reference_edge_id": "P01_E031",
        "required_stream_key": "RESTEAM",
    },
    # Process 2
    {
        "process_id": 2,
        "target_id": "P02_T001",
        "target_stream": "OUT_PROD",
        "target_species": "H2",
        "target_feature": "0.8 * PROD_H2_MoleFlow",
        "formula": "0.8 * Mole_Flow * Frac_H2",
        "scale": 0.8,
        "frac_column": "Frac_H2",
        "special_case": False,
        "reference_edge_id": "P02_E014",
        "main_column": "PROD_H2_MoleFlow",
        "required_stream_key": "PROD",
    },
    {
        "process_id": 2,
        "target_id": "P02_T002",
        "target_stream": "OUT_EXHAUST",
        "target_species": "CO2",
        "target_feature": "PROD_CO2_MoleFlow",
        "formula": "Mole_Flow * Frac_CO2",
        "scale": 1.0,
        "frac_column": "Frac_CO2",
        "special_case": False,
        "reference_edge_id": "P02_E016",
        "required_stream_key": "13",
        "stream_keys": ["13", "OUT_EXHAUST"],
        "alternate_stream_keys": ["PROD"],
    },
    # Process 3
    {
        "process_id": 3,
        "target_id": "P03_T001",
        "target_stream": "OUT_PROD",
        "target_species": "H2",
        "target_feature": "PROD_H2_Mole",
        "formula": "Mole_Flow * Frac_H2",
        "scale": 1.0,
        "frac_column": "Frac_H2",
        "special_case": False,
        "reference_edge_id": "P03_E025",
    },
    {
        "process_id": 3,
        "target_id": "P03_T002",
        "target_stream": "OUT_CO2",
        "target_species": "CO2",
        "target_feature": "OUT_CO2_Mole",
        "formula": "Mole_Flow * Frac_CO2",
        "scale": 1.0,
        "frac_column": "Frac_CO2",
        "special_case": False,
        "reference_edge_id": "P03_E014",
    },
    {
        "process_id": 3,
        "target_id": "P03_T003",
        "target_stream": "OUT_H2O",
        "target_species": "H2O",
        "target_feature": "RE_H2O_Mole",
        "formula": "Mole_Flow * Frac_H2O",
        "scale": 1.0,
        "frac_column": "Frac_H2O",
        "special_case": False,
        "reference_edge_id": "P03_E023",
    },
    # Process 4
    {
        "process_id": 4,
        "target_id": "P04_T001",
        "target_stream": "OUT_PROD",
        "target_species": "H2",
        "target_feature": "PROD_H2_Mole",
        "formula": "Mole_Flow * Frac_H2",
        "scale": 1.0,
        "frac_column": "Frac_H2",
        "special_case": False,
        "reference_edge_id": "P04_E019",
    },
    {
        "process_id": 4,
        "target_id": "P04_T002",
        "target_stream": "OUT_EXHAUST",
        "target_species": "CO2",
        "target_feature": "EX_CO2_Mole",
        "formula": "Mole_Flow * Frac_CO2",
        "scale": 1.0,
        "frac_column": "Frac_CO2",
        "special_case": False,
        "reference_edge_id": "P04_E009",
    },
    {
        "process_id": 4,
        "target_id": "P04_T003",
        "target_stream": "OUT_stream",
        "target_species": "CO2",
        "target_feature": "Stream10_CO2_Mole",
        "formula": "Mole_Flow * Frac_CO2",
        "scale": 1.0,
        "frac_column": "Frac_CO2",
        "special_case": False,
        "reference_edge_id": "P04_E020",
    },
    # Process 5
    {
        "process_id": 5,
        "target_id": "P05_T001",
        "target_stream": "OUT_PROD",
        "target_species": "H2",
        "target_feature": "PROD_H2_Mole",
        "formula": "Mole_Flow * Frac_H2",
        "scale": 1.0,
        "frac_column": "Frac_H2",
        "special_case": False,
        "reference_edge_id": "P05_E016",
    },
    {
        "process_id": 5,
        "target_id": "P05_T002",
        "target_stream": "OUT_H2O",
        "target_species": "H2O",
        "target_feature": "RE_H2O_Mole",
        "formula": "Mole_Flow * Frac_H2O",
        "scale": 1.0,
        "frac_column": "Frac_H2O",
        "special_case": False,
        "reference_edge_id": "P05_E015",
    },
    {
        "process_id": 5,
        "target_id": "P05_T003",
        "target_stream": "OUT_CO2",
        "target_species": "CO2",
        "target_feature": "Stream12_CO2_Mole",
        "formula": "Mole_Flow * Frac_CO2",
        "scale": 1.0,
        "frac_column": "Frac_CO2",
        "special_case": False,
        "reference_edge_id": "P05_E014",
    },
    # Process 6
    {
        "process_id": 6,
        "target_id": "P06_T001",
        "target_stream": "OUT_PROD",
        "target_species": "H2",
        "target_feature": "PROD_H2_Mole",
        "formula": "Mole_Flow * Frac_H2",
        "scale": 1.0,
        "frac_column": "Frac_H2",
        "special_case": False,
        "reference_edge_id": "P06_E022",
    },
    {
        "process_id": 6,
        "target_id": "P06_T002",
        "target_stream": "OUT_H2O",
        "target_species": "H2O",
        "target_feature": "RE_H2O_Mole",
        "formula": "Mole_Flow * Frac_H2O",
        "scale": 1.0,
        "frac_column": "Frac_H2O",
        "special_case": False,
        "reference_edge_id": "P06_E021",
    },
    {
        "process_id": 6,
        "target_id": "P06_T003",
        "target_stream": "OUT_CO2",
        "target_species": "CO2",
        "target_feature": "Stream12_CO2_Mole",
        "formula": "Mole_Flow * Frac_CO2",
        "scale": 1.0,
        "frac_column": "Frac_CO2",
        "special_case": False,
        "reference_edge_id": "P06_E023",
    },
    {
        "process_id": 6,
        "target_id": "P06_T004",
        "target_stream": "OUT_EXHAUST",
        "target_species": "CO2",
        "target_feature": "Stream17_Mole_Flow * Stream17_Frac_CO2",
        "formula": "Stream17_Mole_Flow * Stream17_Frac_CO2",
        "scale": 1.0,
        "frac_column": "Frac_CO2",
        "special_case": True,
        "reference_edge_id": "P06_E013",
        "stream_keys": ["17", "Stream17"],
    },
    # Process 7
    {
        "process_id": 7,
        "target_id": "P07_T001",
        "target_stream": "OUT_PROD",
        "target_species": "H2",
        "target_feature": "PROD_H2_Mole",
        "formula": "Mole_Flow * Frac_H2",
        "scale": 1.0,
        "frac_column": "Frac_H2",
        "special_case": False,
        "reference_edge_id": "P07_E016",
    },
    {
        "process_id": 7,
        "target_id": "P07_T002",
        "target_stream": "OUT_PROD",
        "target_species": "CO2",
        "target_feature": "PROD_CO2_Mole",
        "formula": "Mole_Flow * Frac_CO2",
        "scale": 1.0,
        "frac_column": "Frac_CO2",
        "special_case": False,
        "reference_edge_id": "P07_E016",
    },
    {
        "process_id": 7,
        "target_id": "P07_T003",
        "target_stream": "OUT_H2O",
        "target_species": "H2O",
        "target_feature": "RE_H2O_Mole",
        "formula": "Mole_Flow * Frac_H2O",
        "scale": 1.0,
        "frac_column": "Frac_H2O",
        "special_case": False,
        "reference_edge_id": "P07_E017",
    },
    # Process 8
    {
        "process_id": 8,
        "target_id": "P08_T001",
        "target_stream": "OUT_PROD",
        "target_species": "H2",
        "target_feature": "PROD_H2_Mole",
        "formula": "Mole_Flow * Frac_H2",
        "scale": 1.0,
        "frac_column": "Frac_H2",
        "special_case": False,
        "reference_edge_id": "P08_E015",
    },
    {
        "process_id": 8,
        "target_id": "P08_T002",
        "target_stream": "OUT_CO2",
        "target_species": "CO2",
        "target_feature": "12_CO2_Mole",
        "formula": "Mole_Flow * Frac_CO2",
        "scale": 1.0,
        "frac_column": "Frac_CO2",
        "special_case": False,
        "reference_edge_id": "P08_E016",
    },
    {
        "process_id": 8,
        "target_id": "P08_T003",
        "target_stream": "OUT_H2O",
        "target_species": "H2O",
        "target_feature": "RESTEAM_H2O_Mole",
        "formula": "Mole_Flow * Frac_H2O",
        "scale": 1.0,
        "frac_column": "Frac_H2O",
        "special_case": False,
        "reference_edge_id": "P08_E014",
    },
    # Process 9
    {
        "process_id": 9,
        "target_id": "P09_T001",
        "target_stream": "OUT_PROD",
        "target_species": "H2",
        "target_feature": "PROD_H2_Mole",
        "formula": "Mole_Flow * Frac_H2",
        "scale": 1.0,
        "frac_column": "Frac_H2",
        "special_case": False,
        "reference_edge_id": "P09_E020",
    },
    {
        "process_id": 9,
        "target_id": "P09_T002",
        "target_stream": "OUT_CO2",
        "target_species": "CO2",
        "target_feature": "EXHAUS_CO2_Mole",
        "formula": "Mole_Flow * Frac_CO2",
        "scale": 1.0,
        "frac_column": "Frac_CO2",
        "special_case": False,
        "reference_edge_id": "P09_E022",
    },
    # Process 10
    {
        "process_id": 10,
        "target_id": "P10_T001",
        "target_stream": "OUT_PROD",
        "target_species": "H2",
        "target_feature": "PROD_H2_Mole",
        "formula": "Mole_Flow * Frac_H2",
        "scale": 1.0,
        "frac_column": "Frac_H2",
        "special_case": False,
        "reference_edge_id": "P10_E029",
    },
    {
        "process_id": 10,
        "target_id": "P10_T002",
        "target_stream": "OUT_CO2",
        "target_species": "CO2",
        "target_feature": "OUT_CO2_Mole",
        "formula": "Mole_Flow * Frac_CO2",
        "scale": 1.0,
        "frac_column": "Frac_CO2",
        "special_case": False,
        "reference_edge_id": "P10_E015",
    },
    {
        "process_id": 10,
        "target_id": "P10_T003",
        "target_stream": "OUT_H2O",
        "target_species": "H2O",
        "target_feature": "RE_H2O_Mole",
        "formula": "Mole_Flow * Frac_H2O",
        "scale": 1.0,
        "frac_column": "Frac_H2O",
        "special_case": False,
        "reference_edge_id": "P10_E027",
    },
)

# Fallback alias groups (used only after required/canonical keys fail).
STREAM_ALIAS_GROUPS: dict[str, list[str]] = {
    "OUT_PROD": ["OUT_PROD", "OUT_prod", "PROD"],
    "OUT_prod": ["OUT_PROD", "OUT_prod", "PROD"],
    "OUT_EXHAUST": ["OUT_EXHAUST", "OUT_exhaust", "EXHAUST", "FUELGAS", "EX"],
    "OUT_exhaust": ["OUT_EXHAUST", "OUT_exhaust", "EXHAUST", "FUELGAS"],
    "OUT_H2O": ["OUT_H2O", "RE", "RESTEAM"],
    "OUT_CO2": ["OUT_CO2", "OUT"],
    "OUT_stream": ["OUT_stream", "10"],
    "OUT_RESTEAM": ["OUT_RESTEAM", "RESTEAM"],
}


@dataclass
class TargetRow:
    process_id: int
    target_id: str
    target_stream: str
    target_species: str
    target_feature: str
    formula: str
    scale: float
    frac_column: str
    special_case: bool
    reference_edge_id: str
    main_column: str | None = None
    required_stream_key: str = ""
    stream_keys: list[str] = field(default_factory=list)
    alternate_stream_keys: list[str] = field(default_factory=list)


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _norm(value: Any) -> str:
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return re.sub(r"[^A-Z0-9]+", "", s.upper())


def _norm_col(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _sample_col(df: pd.DataFrame) -> str | None:
    for c in ("ID", "sample_id", "Sample_ID", "id", "index"):
        if c in df.columns:
            return c
    return None


def _priority_stream_keys(spec: TargetRow, canonical_sk: str = "") -> list[str]:
    """Ordered stream-key candidates (specific keys before broad alias groups)."""
    keys: list[str] = []
    for group in (spec.stream_keys, [spec.required_stream_key], [canonical_sk], [spec.target_stream]):
        for k in group:
            if k is None:
                continue
            s = str(k).strip()
            if s:
                keys.append(s)
    for m in re.findall(r"Stream?(\d+)", spec.target_feature, flags=re.I):
        keys.append(m)
    m_lead = re.match(r"^(\d+)_", spec.target_feature.strip())
    if m_lead:
        keys.append(m_lead.group(1))
    ts = _norm(spec.target_stream)
    for k, group in STREAM_ALIAS_GROUPS.items():
        if _norm(k) == ts:
            keys.extend(group)
    out: list[str] = []
    seen: set[str] = set()
    for k in keys:
        nk = _norm(k)
        if nk and nk not in seen:
            seen.add(nk)
            out.append(str(k).strip())
    return out


def _present_stream_keys(streams: pd.DataFrame) -> dict[str, list[str]]:
    """Map normalized stream key -> raw names appearing in Streams CSV."""
    if _is_long_format(streams):
        stream_col = next(
            (c for c in ("Stream_Name", "stream_key", "stream_name", "main_data_stream_key") if c in streams.columns),
            None,
        )
        if stream_col is None:
            return {}
        raw = streams[stream_col].astype(str)
        out: dict[str, list[str]] = {}
        for name in raw.unique():
            out.setdefault(_norm(name), []).append(name)
        return out
    # wide format: infer from column prefixes
    out: dict[str, list[str]] = {}
    for c in streams.columns:
        for suffix in ("_Mole_Flow", "_Frac_H2", "_Frac_CO2", "_Frac_H2O"):
            if c.endswith(suffix):
                prefix = c[: -len(suffix)]
                out.setdefault(_norm(prefix), []).append(prefix)
    return out


def _resolve_stream_key(
    streams: pd.DataFrame,
    spec: TargetRow,
    canonical_sk: str,
) -> tuple[str, list[str], str]:
    """Pick a single stream key; return (chosen_raw_key, all_hits_at_ambiguous_step, note)."""
    present = _present_stream_keys(streams)
    if not present:
        return "", [], "no_streams_in_file"

    priority = _priority_stream_keys(spec, canonical_sk)
    for key in priority:
        nk = _norm(key)
        if nk in present:
            return present[nk][0], [key], f"priority:{key}"

    return "", [], "unresolved_stream_key"


def _build_col_map(columns: list[str]) -> dict[str, str]:
    return {_norm_col(c): c for c in columns}


def _find_column(columns: list[str], candidates: list[str]) -> tuple[str | None, str]:
    cmap = _build_col_map(columns)
    tried: list[str] = []
    for cand in candidates:
        key = _norm_col(cand)
        tried.append(cand)
        if key in cmap:
            return cmap[key], f"exact:{cand}"
    # fuzzy: contains all tokens
    for cand in candidates:
        tokens = [t for t in re.split(r"[^a-z0-9]+", cand.lower()) if t]
        if not tokens:
            continue
        for ncol, orig in cmap.items():
            if all(t in ncol for t in tokens):
                return orig, f"fuzzy:{cand}"
    return None, f"not_found:{','.join(tried[:5])}"


def _main_target_series(main: pd.DataFrame, spec: TargetRow) -> tuple[pd.Series, str, str]:
    """Return main_value series, resolution note, status fragment."""
    if spec.special_case and "Stream17" in spec.target_feature:
        c1, n1 = _find_column(
            list(main.columns),
            ["Stream17_Mole_Flow", "Stream17_MoleFlow", "17_Mole_Flow"],
        )
        c2, n2 = _find_column(
            list(main.columns),
            ["Stream17_Frac_CO2", "Stream17_FracCO2", "17_Frac_CO2"],
        )
        if c1 and c2:
            v1 = pd.to_numeric(main[c1], errors="coerce")
            v2 = pd.to_numeric(main[c2], errors="coerce")
            return v1 * v2, f"main_product:{n1}*{n2}", ""
        return pd.Series([float("nan")] * len(main)), "no_stream17_main_columns", "unresolved_main_column"

    if spec.main_column:
        col, note = _find_column(list(main.columns), [spec.main_column, spec.target_feature])
        if col:
            scale = spec.scale
            return pd.to_numeric(main[col], errors="coerce") * scale if scale != 1.0 else pd.to_numeric(
                main[col], errors="coerce"
            ), f"scaled_column:{note}", ""

    # Direct column
    col, note = _find_column(list(main.columns), [spec.target_feature])
    if col:
        return pd.to_numeric(main[col], errors="coerce"), note, ""

    # 0.8 * PROD_H2_MoleFlow style
    m = re.match(r"^([\d.]+)\s*\*\s*(.+)$", spec.target_feature.replace(" ", ""))
    if m:
        scale = float(m.group(1))
        inner = m.group(2)
        col, note = _find_column(list(main.columns), [inner, inner.replace("_", "")])
        if col:
            return pd.to_numeric(main[col], errors="coerce") * scale, f"scaled_expr:{note}", ""

    return pd.Series([float("nan")] * len(main)), f"missing:{spec.target_feature}", "unresolved_main_column"


def _is_long_format(streams: pd.DataFrame) -> bool:
    return any(
        c in streams.columns
        for c in ("Stream_Name", "stream_key", "stream_name", "main_data_stream_key")
    )


def _streams_format(streams: pd.DataFrame) -> str:
    return "long" if _is_long_format(streams) else "wide"


def _extract_long_stream(
    streams: pd.DataFrame,
    sample_col: str,
    matched_key: str,
    frac_column: str,
) -> pd.DataFrame:
    stream_col = next(
        (c for c in ("Stream_Name", "stream_key", "stream_name", "main_data_stream_key") if c in streams.columns),
        None,
    )
    sub = streams[streams[stream_col].astype(str).map(_norm) == _norm(matched_key)].copy()
    if sub.empty:
        return pd.DataFrame(columns=[sample_col, "stream_key", "Mole_Flow", frac_column])
    if sub.duplicated([sample_col]).any():
        sub = sub.sort_values([sample_col, stream_col]).drop_duplicates([sample_col], keep="first")
    out = sub[[sample_col, stream_col, "Mole_Flow", frac_column]].copy()
    out = out.rename(columns={sample_col: "sample_id", stream_col: "matched_stream_key"})
    return out


def _extract_wide_stream(
    streams: pd.DataFrame,
    sample_col: str,
    matched_key: str,
    frac_column: str,
) -> pd.DataFrame:
    aliases = [matched_key, _norm(matched_key)]
    mole_col, _ = _find_column(
        list(streams.columns),
        [f"{a}_Mole_Flow" for a in aliases] + [f"{a}.Mole_Flow" for a in aliases],
    )
    frac_col, _ = _find_column(
        list(streams.columns),
        [f"{a}_{frac_column}" for a in aliases] + [f"{a}.{frac_column}" for a in aliases],
    )
    if mole_col is None or frac_col is None:
        return pd.DataFrame(columns=["sample_id", "matched_stream_key", "Mole_Flow", frac_column])
    out = streams[[sample_col, mole_col, frac_col]].copy()
    out["matched_stream_key"] = matched_key
    return out.rename(columns={sample_col: "sample_id", mole_col: "Mole_Flow", frac_col: frac_column})


def _calc_from_streams(
    streams: pd.DataFrame,
    sample_col: str,
    spec: TargetRow,
    canonical_sk: str,
) -> tuple[pd.DataFrame, str, str, str]:
    """Return per-sample stream calc frame, matched_stream_key, status, resolution note."""
    if spec.special_case and "Stream17" in spec.target_feature:
        key, _, note = _resolve_stream_key(streams, spec, canonical_sk or "17")
        if not key:
            m1, n1 = _find_column(list(streams.columns), ["Stream17_Mole_Flow", "17_Mole_Flow"])
            f1, n2 = _find_column(list(streams.columns), ["Stream17_Frac_CO2", "17_Frac_CO2"])
            if m1 and f1:
                out = streams[[sample_col, m1, f1]].copy()
                out["calc_value"] = pd.to_numeric(out[m1], errors="coerce") * pd.to_numeric(out[f1], errors="coerce")
                out = out.rename(columns={sample_col: "sample_id"})
                out["matched_stream_key"] = "Stream17_wide"
                out["mole_flow"] = out[m1]
                out["frac_value"] = out[f1]
                out["frac_column"] = "Frac_CO2"
                return (
                    out[["sample_id", "matched_stream_key", "mole_flow", "frac_column", "frac_value", "calc_value"]],
                    "Stream17_wide",
                    "",
                    f"wide:{n1}*{n2}",
                )
            return pd.DataFrame(), "", "unresolved_stream_key", note
        if _is_long_format(streams):
            s = _extract_long_stream(streams, sample_col, key, spec.frac_column)
        else:
            s = _extract_wide_stream(streams, sample_col, key, spec.frac_column)
        if s.empty:
            return pd.DataFrame(), key, "unresolved_stream_column", note
        s["calc_value"] = pd.to_numeric(s["Mole_Flow"], errors="coerce") * pd.to_numeric(s[spec.frac_column], errors="coerce")
        s["frac_value"] = pd.to_numeric(s[spec.frac_column], errors="coerce")
        s["mole_flow"] = pd.to_numeric(s["Mole_Flow"], errors="coerce")
        s["frac_column"] = spec.frac_column
        return s[["sample_id", "matched_stream_key", "mole_flow", "frac_column", "frac_value", "calc_value"]], key, "", note

    key, _, note = _resolve_stream_key(streams, spec, canonical_sk)
    if not key:
        return pd.DataFrame(), "", "unresolved_stream_key", note
    if _is_long_format(streams):
        s = _extract_long_stream(streams, sample_col, key, spec.frac_column)
    else:
        s = _extract_wide_stream(streams, sample_col, key, spec.frac_column)
    if s.empty or spec.frac_column not in s.columns:
        return pd.DataFrame(), key, "unresolved_stream_column", note
    mole = pd.to_numeric(s["Mole_Flow"], errors="coerce")
    frac = pd.to_numeric(s[spec.frac_column], errors="coerce")
    s["calc_value"] = spec.scale * mole * frac
    s["frac_value"] = frac
    s["mole_flow"] = mole
    s["frac_column"] = spec.frac_column
    return s[["sample_id", "matched_stream_key", "mole_flow", "frac_column", "frac_value", "calc_value"]], key, "", note


def _tolerance_match(abs_diff: float, rel_diff: float) -> bool:
    if math.isnan(abs_diff):
        return False
    if abs_diff <= ABS_TOL:
        return True
    if not math.isnan(rel_diff) and rel_diff <= REL_TOL:
        return True
    return False


def _r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    yt = pd.to_numeric(y_true, errors="coerce").astype(float)
    yp = pd.to_numeric(y_pred, errors="coerce").astype(float)
    valid = yt.notna() & yp.notna()
    yt, yp = yt[valid], yp[valid]
    if len(yt) < 2:
        return float("nan")
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    if ss_tot <= 0:
        return 1.0 if ss_res <= 0 else float("nan")
    return 1.0 - ss_res / ss_tot


def _target_row(d: dict[str, Any]) -> TargetRow:
    return TargetRow(
        process_id=int(d["process_id"]),
        target_id=str(d["target_id"]),
        target_stream=str(d["target_stream"]),
        target_species=str(d["target_species"]),
        target_feature=str(d["target_feature"]),
        formula=str(d["formula"]),
        scale=float(d.get("scale", 1.0)),
        frac_column=str(d.get("frac_column", SPECIES_FRAC.get(str(d["target_species"]), "Frac_CO2"))),
        special_case=bool(d.get("special_case", False)),
        reference_edge_id=str(d.get("reference_edge_id", "")),
        main_column=d.get("main_column"),
        required_stream_key=str(d.get("required_stream_key", "")),
        stream_keys=list(d.get("stream_keys", [])),
        alternate_stream_keys=list(d.get("alternate_stream_keys", [])),
    )


def _enrich_from_v4_targets(formulas: list[TargetRow]) -> None:
    path = PROJECT_ROOT / "data/reference/v4/target_stream_targets.csv"
    if not path.is_file():
        return
    tdf = pd.read_csv(path)
    by_id = {
        str(r["target_id"]): str(r.get("required_stream_key", "") or r.get("main_data_stream_key", ""))
        for _, r in tdf.iterrows()
    }
    for spec in formulas:
        if not spec.required_stream_key and spec.target_id in by_id:
            spec.required_stream_key = by_id[spec.target_id]


def _quick_stream_match(
    main: pd.DataFrame,
    streams: pd.DataFrame,
    spec: TargetRow,
    stream_key: str,
) -> tuple[float, float]:
    """Return (max_abs_diff, mismatch_rate) for a candidate stream key vs main."""
    sid = _sample_col(streams) or "ID"
    main_sid = _sample_col(main) or "sample_id"
    main_vals, _, st_main = _main_target_series(main, spec)
    if st_main:
        return float("nan"), float("nan")
    main_work = main.copy()
    main_work["sample_id"] = main_work[main_sid].astype(str) if main_sid in main.columns else main_work.index.astype(str)
    main_work["main_value"] = main_vals
    if main_work["sample_id"].duplicated().any():
        main_work = main_work.drop_duplicates("sample_id", keep="last")
    tmp = TargetRow(
        process_id=spec.process_id,
        target_id=spec.target_id,
        target_stream=spec.target_stream,
        target_species=spec.target_species,
        target_feature=spec.target_feature,
        formula=spec.formula,
        scale=spec.scale,
        frac_column=spec.frac_column,
        special_case=False,
        reference_edge_id=spec.reference_edge_id,
        required_stream_key=stream_key,
        stream_keys=[stream_key],
    )
    calc, key, st, _ = _calc_from_streams(streams, sid, tmp, stream_key)
    if st or calc.empty:
        return float("nan"), float("nan")
    calc["sample_id"] = calc["sample_id"].astype(str)
    joined = main_work[["sample_id", "main_value"]].merge(calc, on="sample_id", how="inner")
    valid = joined["main_value"].notna() & joined["calc_value"].notna()
    if not valid.any():
        return float("nan"), float("nan")
    diff = (joined.loc[valid, "main_value"] - joined.loc[valid, "calc_value"]).abs()
    tol = diff.apply(lambda x: x <= ABS_TOL)
    rel = diff / joined.loc[valid, "main_value"].abs().where(joined.loc[valid, "main_value"].abs() > 0, 1.0)
    tol = tol | rel.le(REL_TOL)
    return float(diff.max()), float((~tol).sum() / len(diff))


def _reference_mapping_checks(
    formulas: list[TargetRow],
    target_ref: pd.DataFrame,
    canonical: pd.DataFrame,
    needs_review: pd.DataFrame,
) -> pd.DataFrame:
    nr_edges = set(needs_review["canonical_edge_id"].astype(str)) if not needs_review.empty else set()
    amb_edges = set()
    if "mapping_confidence" in canonical.columns:
        amb_edges = set(
            canonical[canonical["mapping_confidence"].astype(str).str.lower().isin(["ambiguous", "needs_review"])][
                "canonical_edge_id"
            ].astype(str)
        )

    rows: list[dict[str, Any]] = []
    for spec in formulas:
        ref_edge = spec.reference_edge_id
        ce = canonical[canonical["canonical_edge_id"].astype(str) == ref_edge]
        ans = target_ref[
            (pd.to_numeric(target_ref["process_id"], errors="coerce") == spec.process_id)
            & (target_ref["target_column"].astype(str).map(_norm_col) == _norm_col(spec.target_feature))
        ]
        # fallback: match by edge id in answer table
        if ans.empty:
            ans = target_ref[target_ref["canonical_answer_edge_id"].astype(str) == ref_edge]
        ce_sk = str(ce.iloc[0]["main_data_stream_key"]) if not ce.empty else ""
        ce_dst = str(ce.iloc[0]["dst_node"]) if not ce.empty else ""
        ce_role = str(ce.iloc[0].get("dst_node_raw", ce.get("visual_stream_id", ""))) if not ce.empty else ""
        ans_sk = str(ans.iloc[0]["main_data_stream_key"]) if not ans.empty else ""
        status_parts: list[str] = []
        if ce.empty:
            status_parts.append("missing_canonical_edge")
        if ans.empty:
            status_parts.append("no_target_answer_row")
        elif _norm_col(str(ans.iloc[0]["target_column"])) != _norm_col(spec.target_feature):
            status_parts.append("target_column_mismatch")
        if ce_dst != "V_OUTPUT" and ce_dst:
            status_parts.append("dst_not_v_output")
        if ref_edge in nr_edges or ref_edge in amb_edges:
            status_parts.append("needs_review_edge_used")
        # stream key alignment
        stream_ok = _norm(ce_sk) in {_norm(a) for a in _priority_stream_keys(spec, ce_sk)} or _norm(ans_sk) in {
            _norm(a) for a in _priority_stream_keys(spec, ce_sk)
        }
        if not stream_ok and ce_sk:
            status_parts.append("stream_key_mismatch")
        mapping_status = "ok" if not status_parts else ";".join(status_parts)
        rows.append(
            {
                "target_id": spec.target_id,
                "process_id": spec.process_id,
                "target_feature": spec.target_feature,
                "target_stream": spec.target_stream,
                "target_species": spec.target_species,
                "reference_edge_id": ref_edge,
                "canonical_edge_id": ref_edge,
                "target_answer_edge_id": str(ans.iloc[0]["canonical_answer_edge_id"]) if not ans.empty else "",
                "canonical_main_data_stream_key": ce_sk,
                "canonical_dst_node_raw": ce_role,
                "target_answer_main_data_stream_key": ans_sk,
                "dst_node": ce_dst,
                "is_v_output": ce_dst == "V_OUTPUT",
                "needs_review": ref_edge in nr_edges or ref_edge in amb_edges,
                "mapping_status": mapping_status,
            }
        )
    return pd.DataFrame(rows)


def _validate_target(
    main: pd.DataFrame,
    streams: pd.DataFrame,
    spec: TargetRow,
    canonical_sk: str,
    *,
    row_order_join: bool,
    join_warning: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    main_sid = _sample_col(main)
    stream_sid = _sample_col(streams)
    row_order = row_order_join or main_sid is None or stream_sid is None

    main_vals, main_note, main_status = _main_target_series(main, spec)
    main_work = main.copy()
    if main_sid:
        main_work["sample_id"] = main_work[main_sid].astype(str)
    else:
        main_work["sample_id"] = main_work.index.astype(str)
    main_work["main_value"] = main_vals
    main_dup_note = ""
    if main_work["sample_id"].duplicated().any():
        n_dup = int(main_work["sample_id"].duplicated().sum())
        main_dup_note = f"main_duplicate_sample_ids={n_dup};kept=last"
        main_work = main_work.drop_duplicates("sample_id", keep="last")

    stream_calc, matched_key, stream_status, stream_note = _calc_from_streams(
        streams, stream_sid or "sample_id", spec, canonical_sk
    )
    alt_note = ""
    if spec.alternate_stream_keys and not main_vals.isna().all():
        alt_stats: list[str] = []
        for alt in spec.alternate_stream_keys:
            mx, mr = _quick_stream_match(main, streams, spec, alt)
            if not math.isnan(mx):
                alt_stats.append(f"{alt}:max_abs={mx:.6g},mismatch_rate={mr:.4g}")
        if alt_stats:
            alt_note = "alternate_stream_checks=" + ";".join(alt_stats)

    status = stream_status or main_status
    if join_warning or main_dup_note:
        status = status or "sample_join_problem"

    blocking = ("ambiguous_stream_alias", "unresolved_stream_key", "unresolved_stream_column")
    if status in blocking or (status == "unresolved_main_column" and not spec.special_case):
        empty_samples = pd.DataFrame()
        summary = {
            "process_id": spec.process_id,
            "target_id": spec.target_id,
            "target_stream": spec.target_stream,
            "matched_stream_key": matched_key,
            "target_species": spec.target_species,
            "target_feature": spec.target_feature,
            "formula": spec.formula,
            "scale": spec.scale,
            "frac_column": spec.frac_column,
            "main_resolution": main_note,
            "stream_resolution": f"{stream_note}|{alt_note}".strip("|"),
            "row_order_join_used": row_order,
            "join_warning": ";".join(x for x in [join_warning, main_dup_note] if x),
            "n_main": len(main),
            "n_stream": len(streams) if _is_long_format(streams) else len(streams),
            "n_joined": 0,
            "n_valid": 0,
            "status": status,
        }
        return empty_samples, summary

    if stream_calc.empty:
        status = status or "unresolved_stream_column"
        summary = {
            "process_id": spec.process_id,
            "target_id": spec.target_id,
            "target_stream": spec.target_stream,
            "matched_stream_key": "",
            "target_species": spec.target_species,
            "target_feature": spec.target_feature,
            "formula": spec.formula,
            "scale": spec.scale,
            "frac_column": spec.frac_column,
            "main_resolution": main_note,
            "stream_resolution": "",
            "row_order_join_used": row_order,
            "join_warning": join_warning,
            "n_main": len(main),
            "n_stream": len(streams),
            "n_joined": 0,
            "n_valid": 0,
            "status": status,
        }
        return pd.DataFrame(), summary

    if row_order:
        n = min(len(main_work), len(stream_calc))
        main_work = main_work.iloc[:n].copy()
        stream_calc = stream_calc.iloc[:n].copy()
        main_work["sample_id"] = main_work["sample_id"].astype(str).values
        stream_calc["sample_id"] = stream_calc["sample_id"].astype(str).values
        joined = pd.concat(
            [
                main_work[["sample_id", "main_value"]].reset_index(drop=True),
                stream_calc.reset_index(drop=True),
            ],
            axis=1,
        )
    else:
        stream_calc = stream_calc.copy()
        stream_calc["sample_id"] = stream_calc["sample_id"].astype(str)
        main_work["sample_id"] = main_work["sample_id"].astype(str)
        joined = main_work[["sample_id", "main_value"]].merge(stream_calc, on="sample_id", how="inner")

    joined["abs_diff"] = (joined["main_value"] - joined["calc_value"]).abs()
    denom = joined["main_value"].abs().where(joined["main_value"].abs() > 0, 1.0)
    joined["rel_diff"] = joined["abs_diff"] / denom
    joined["is_exact_match"] = joined["abs_diff"] <= ABS_TOL
    joined["is_tolerance_match"] = joined.apply(
        lambda r: _tolerance_match(float(r["abs_diff"]), float(r["rel_diff"])), axis=1
    )

    valid = joined["main_value"].notna() & joined["calc_value"].notna()
    n_valid = int(valid.sum())

    # Process6 Stream17: no Main target column — report streams-only status
    if spec.special_case and "Stream17" in spec.target_feature and main_vals.isna().all():
        if n_valid == 0 and not stream_calc.empty:
            # streams computed but no main to join
            n_valid = int(stream_calc["calc_value"].notna().sum())
            status = "ok_streams_only_no_main"
        elif n_valid == 0:
            status = "special_case_unresolved"
        else:
            status = "ok_streams_only_no_main"
        mismatch_count = 0
        exact = tol = 0
    elif n_valid == 0:
        if main_vals.isna().all() and not stream_calc.empty:
            status = "ok_streams_only_no_main" if spec.special_case else "unresolved_main_column"
        else:
            status = status or "unresolved_main_column"
        mismatch_count = 0
        exact = tol = 0
    else:
        g = joined[valid]
        mismatch_count = int((~g["is_tolerance_match"]).sum())
        exact = int(g["is_exact_match"].sum())
        tol = int(g["is_tolerance_match"].sum())
        if mismatch_count == 0:
            status = "ok" if exact == n_valid else "ok_with_tolerance"
        else:
            status = "mismatch"

    joined["process_id"] = spec.process_id
    joined["target_id"] = spec.target_id
    joined["target_stream"] = spec.target_stream
    joined["target_species"] = spec.target_species
    joined["target_feature"] = spec.target_feature
    joined["formula"] = spec.formula
    joined["scale"] = spec.scale
    joined["status"] = status

    summary = {
        "process_id": spec.process_id,
        "target_id": spec.target_id,
        "target_stream": spec.target_stream,
        "matched_stream_key": matched_key,
        "target_species": spec.target_species,
        "target_feature": spec.target_feature,
        "formula": spec.formula,
        "scale": spec.scale,
        "frac_column": spec.frac_column,
        "main_resolution": main_note,
        "stream_resolution": "|".join(x for x in [stream_note, alt_note, f"matched={matched_key}"] if x),
        "row_order_join_used": row_order,
        "join_warning": ";".join(x for x in [join_warning, main_dup_note] if x),
        "n_main": len(main),
        "n_stream": len(streams),
        "n_joined": len(joined),
        "n_valid": n_valid,
        "main_mean": float(joined.loc[valid, "main_value"].mean()) if n_valid else float("nan"),
        "calc_mean": float(joined.loc[valid, "calc_value"].mean()) if n_valid else float("nan"),
        "main_std": float(joined.loc[valid, "main_value"].std()) if n_valid else float("nan"),
        "calc_std": float(joined.loc[valid, "calc_value"].std()) if n_valid else float("nan"),
        "diff_mean": float((joined.loc[valid, "calc_value"] - joined.loc[valid, "main_value"]).mean())
        if n_valid
        else float("nan"),
        "diff_abs_mean": float(joined.loc[valid, "abs_diff"].mean()) if n_valid else float("nan"),
        "diff_abs_median": float(joined.loc[valid, "abs_diff"].median()) if n_valid else float("nan"),
        "diff_abs_max": float(joined.loc[valid, "abs_diff"].max()) if n_valid else float("nan"),
        "rel_diff_mean": float(joined.loc[valid, "rel_diff"].mean()) if n_valid else float("nan"),
        "rel_diff_median": float(joined.loc[valid, "rel_diff"].median()) if n_valid else float("nan"),
        "rel_diff_max": float(joined.loc[valid, "rel_diff"].max()) if n_valid else float("nan"),
        "corr": float(joined.loc[valid, "main_value"].corr(joined.loc[valid, "calc_value"])) if n_valid > 1 else float("nan"),
        "r2_main_vs_calc": _r2(joined.loc[valid, "main_value"], joined.loc[valid, "calc_value"]) if n_valid else float("nan"),
        "exact_match_count": exact,
        "tolerance_match_count": tol,
        "mismatch_count": mismatch_count,
        "mismatch_rate": float(mismatch_count / n_valid) if n_valid else float("nan"),
        "status": status,
    }
    return joined, summary


def _compare_prior_validation(summary: pd.DataFrame, prior_path: Path) -> str:
    if not prior_path.is_file():
        return "Prior validation file not found; no comparison performed."
    prior = pd.read_csv(prior_path)
    lines = ["## Comparison with prior v4 validation (`target_formula_validation.csv`)", ""]
    key_cols = ["process_id", "target_id"]
    merged = summary.merge(
        prior,
        on=key_cols,
        how="outer",
        suffixes=("_new", "_prior"),
    )
    for tid in ["P02_T002", "P05_T003", "P06_T004"]:
        sub = merged[merged["target_id"] == tid]
        if sub.empty:
            lines.append(f"- **{tid}**: not found in one of the files.")
            continue
        row = sub.iloc[0]
        new_status = row.get("status", "")
        prior_status = row.get("formula_match_status", row.get("formula_match_status_prior", ""))
        new_max = row.get("diff_abs_max", float("nan"))
        prior_max = row.get("max_abs_error", float("nan"))
        lines.append(
            f"- **{tid}**: prior=`{prior_status}` max_abs_error={prior_max}; "
            f"new=`{new_status}` diff_abs_max={new_max}"
        )
        if str(prior_status) != str(new_status):
            lines.append(f"  - Status changed: investigate stream alias / answer edge mapping.")
    return "\n".join(lines)


def _write_plots(out_dir: Path, summary: pd.DataFrame, samples: pd.DataFrame) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        sub = summary.copy()
        sub["label"] = sub["target_id"].astype(str)
        ax.bar(sub["label"], sub["mismatch_rate"].fillna(0))
        ax.set_title("Mismatch rate by target")
        ax.set_ylabel("mismatch_rate")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        fig.savefig(plot_dir / "mismatch_rate_by_target.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(sub["label"], sub["r2_main_vs_calc"].fillna(0))
        ax.set_title("R² main vs calc by target")
        ax.set_ylabel("r2_main_vs_calc")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        fig.savefig(plot_dir / "r2_main_vs_calc_by_target.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(sub["label"], sub["diff_abs_max"].fillna(0))
        ax.set_title("Max absolute diff by target")
        ax.set_ylabel("diff_abs_max")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        fig.savefig(plot_dir / "diff_abs_max_by_target.png", dpi=120)
        plt.close(fig)

        proc = summary.groupby("process_id", as_index=False)["mismatch_rate"].mean()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(proc["process_id"].astype(str), proc["mismatch_rate"].fillna(0))
        ax.set_title("Mean mismatch rate by process")
        ax.set_xlabel("process_id")
        fig.tight_layout()
        fig.savefig(plot_dir / "mismatch_rate_by_process.png", dpi=120)
        plt.close(fig)

    if not samples.empty:
        valid = samples[samples["main_value"].notna() & samples["calc_value"].notna()]
        if not valid.empty:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(valid["main_value"], valid["calc_value"], s=4, alpha=0.3)
            ax.set_xlabel("main_value")
            ax.set_ylabel("calc_value")
            ax.set_title("All targets: main vs calc")
            fig.tight_layout()
            fig.savefig(plot_dir / "scatter_main_vs_calc_all.png", dpi=120)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 4))
            rel = valid["rel_diff"].replace([float("inf"), -float("inf")], float("nan")).dropna()
            ax.hist(rel.clip(upper=rel.quantile(0.99) if len(rel) else 1), bins=50)
            ax.set_title("Relative diff distribution")
            ax.set_xlabel("rel_diff")
            fig.tight_layout()
            fig.savefig(plot_dir / "rel_diff_histogram.png", dpi=120)
            plt.close(fig)

        for tid, g in valid.groupby("target_id"):
            if len(g) < 2:
                continue
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(g["main_value"], g["calc_value"], s=6, alpha=0.4)
            ax.set_title(f"{tid}: main vs calc")
            ax.set_xlabel("main_value")
            ax.set_ylabel("calc_value")
            fig.tight_layout()
            fig.savefig(plot_dir / f"scatter_{tid}.png", dpi=120)
            plt.close(fig)


def _write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    mismatches: pd.DataFrame,
    ref_check: pd.DataFrame,
    prior_note: str,
    join_meta: dict[str, Any],
) -> None:
    def md_table(df: pd.DataFrame, max_rows: int = 50) -> str:
        if df.empty:
            return "(empty)"
        view = df.head(max_rows)
        cols = list(view.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in view.iterrows():
            lines.append("| " + " | ".join(str(row[c]).replace("\n", " ") for c in cols) + " |")
        return "\n".join(lines)

    ok_statuses = {"ok", "ok_with_tolerance", "ok_streams_only_no_main"}
    n_total = len(summary)
    n_ok = int(summary["status"].isin(ok_statuses).sum())
    n_mismatch = int((summary["status"] == "mismatch").sum())
    n_unresolved = int(summary["status"].str.startswith("unresolved").sum())
    n_special = int(summary["special_case"].sum()) if "special_case" in summary.columns else int(
        summary["target_id"].eq("P06_T004").sum()
    )

    lines: list[str] = [
        "# Process Target Formula Validation Report (v4, processes 1–10)",
        "",
        "## 1. Overall conclusion",
        "",
        f"- Total targets validated: **{n_total}**",
        f"- OK targets (exact or tolerance match): **{n_ok}**",
        f"- Mismatch targets: **{n_mismatch}**",
        f"- Unresolved targets: **{n_unresolved}**",
        f"- Special-case targets (Process6 Stream17): **{n_special}**",
        "",
        f"- Row-order join used for any process: **{join_meta.get('row_order_join_used', False)}**",
        f"- Sample count warnings: {join_meta.get('warnings', [])}",
        "",
        "## 2. Per-process results",
        "",
        md_table(
            summary[
                [
                    "process_id",
                    "target_id",
                    "target_feature",
                    "target_stream",
                    "matched_stream_key",
                    "formula",
                    "n_valid",
                    "mismatch_rate",
                    "r2_main_vs_calc",
                    "diff_abs_max",
                    "status",
                ]
            ]
        ),
        "",
        "## 3. Mismatch details (up to 20 samples per target)",
        "",
    ]

    if mismatches.empty:
        lines.append("No mismatches detected.")
    else:
        for tid, g in mismatches.groupby("target_id"):
            lines.append(f"### {tid}")
            lines.append("")
            lines.append(
                md_table(
                    g[
                        ["sample_id", "main_value", "calc_value", "abs_diff", "rel_diff", "matched_stream_key"]
                    ].head(20)
                )
            )
            lines.append("")

    lines.extend(
        [
            "## 4. Root-cause classification",
            "",
        ]
    )
    causes: list[str] = []
    if int((summary["status"] == "ambiguous_stream_alias").sum()):
        causes.append("- **Stream key alias ambiguity**")
    if int((summary["status"] == "sample_join_problem").sum()) or join_meta.get("warnings"):
        causes.append("- **Sample join problem** (ID mismatch or row-order join)")
    if int((summary["status"] == "unresolved_main_column").sum()):
        causes.append("- **Main target column alias / missing column**")
    if int((summary["status"] == "unresolved_stream_key").sum()) or int(
        (summary["status"] == "unresolved_stream_column").sum()
    ):
        causes.append("- **Streams stream key or property column unresolved**")
    if int((summary["status"] == "mismatch").sum()):
        causes.append("- **Main target vs Streams formula numeric mismatch**")
    if int((summary["target_id"] == "P06_T004").sum()):
        causes.append("- **Process6 Stream17 special case** (no Main direct column)")
    if not ref_check.empty and (ref_check["mapping_status"] != "ok").any():
        causes.append("- **Reference mapping issues** (target_answer_edges / canonical_edges)")
    lines.extend(causes or ["- No issues classified."])
    lines.extend(["", "## 5. v4 metric readiness", ""])
    if n_mismatch == 0 and n_unresolved == 0:
        lines.append(
            "All targets with Main columns match Streams-derived formulas. "
            "**Safe to proceed** with `target_metrics_v4` interpretation."
        )
    else:
        bad = summary[~summary["status"].isin(ok_statuses)][["process_id", "target_id", "status"]]
        lines.append("**Hold v4 R² interpretation** for:")
        lines.append("")
        lines.append(md_table(bad))
        unres = summary[summary["status"].str.startswith("unresolved")]
        if not unres.empty:
            lines.append("")
            lines.append("Unresolved columns/keys:")
            for _, r in unres.iterrows():
                lines.append(
                    f"- Process {r['process_id']} {r['target_id']}: "
                    f"main=`{r.get('main_resolution','')}` stream=`{r.get('stream_resolution','')}`"
                )

    lines.extend(["", prior_note, "", "## 6. Reference mapping check", "", md_table(ref_check), ""])
    (out_dir / "PROCESS_TARGET_FORMULA_VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate v4 target formulas for processes 1–10.")
    parser.add_argument("--main-dir", default="data/main_data")
    parser.add_argument("--streams-dir", default="data/main_data_Streams")
    parser.add_argument("--target-ref", default="data/reference/v3/target_answer_edges.csv")
    parser.add_argument("--edge-ref", default="data/reference/v3/canonical_edges.csv")
    parser.add_argument("--out", default="outputs/v4_target_formula_validation_all")
    parser.add_argument(
        "--prior-validation",
        default="data/reference/v4/target_formula_validation.csv",
        help="Optional prior validation CSV for comparison",
    )
    args = parser.parse_args()

    main_dir = _resolve_path(args.main_dir)
    streams_dir = _resolve_path(args.streams_dir)
    out_dir = _resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_ref = _read_csv(_resolve_path(args.target_ref))
    canonical = _read_csv(_resolve_path(args.edge_ref))
    needs_review_path = _resolve_path("data/reference/v3/edges_needs_review_extract.csv")
    needs_review = _read_csv(needs_review_path) if needs_review_path.is_file() else pd.DataFrame()

    formulas = [_target_row(d) for d in TARGET_FORMULAS]
    _enrich_from_v4_targets(formulas)
    ref_check = _reference_mapping_checks(formulas, target_ref, canonical, needs_review)
    ref_check.to_csv(out_dir / "reference_mapping_check.csv", index=False)

    all_samples: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    join_meta: dict[str, Any] = {"row_order_join_used": False, "warnings": []}

    for spec in formulas:
        pid = spec.process_id
        main_path = main_dir / f"{pid}.Process_Main.csv"
        streams_path = streams_dir / f"{pid}.Process_Streams.csv"
        if not main_path.is_file() or not streams_path.is_file():
            summaries.append(
                {
                    "process_id": pid,
                    "target_id": spec.target_id,
                    "status": "unresolved_main_column",
                    "join_warning": f"missing file: main={main_path.exists()} streams={streams_path.exists()}",
                }
            )
            continue

        main = _read_csv(main_path)
        streams = _read_csv(streams_path)
        ce = canonical[canonical["canonical_edge_id"].astype(str) == spec.reference_edge_id]
        canonical_sk = str(ce.iloc[0]["main_data_stream_key"]) if not ce.empty else ""

        main_sid = _sample_col(main)
        stream_sid = _sample_col(streams)
        join_warning = ""
        row_order = main_sid is None or stream_sid is None
        if main_sid and stream_sid:
            main_ids = set(main[main_sid].astype(str))
            stream_ids = set(streams[stream_sid].astype(str))
            if len(main_ids) != len(stream_ids) or main_ids != stream_ids:
                only_main = len(main_ids - stream_ids)
                only_stream = len(stream_ids - main_ids)
                join_warning = f"sample_id mismatch: only_main={only_main} only_stream={only_stream}"
                join_meta["warnings"].append(f"P{pid}: {join_warning}")
        else:
            row_order = True
            join_meta["warnings"].append(f"P{pid}: using row-order join (missing sample id column)")
        if row_order:
            join_meta["row_order_join_used"] = True

        samples, summary = _validate_target(
            main,
            streams,
            spec,
            canonical_sk,
            row_order_join=row_order,
            join_warning=join_warning,
        )
        summary["special_case"] = spec.special_case
        summaries.append(summary)
        if not samples.empty:
            all_samples.append(samples)

    summary_df = pd.DataFrame(summaries)
    samples_df = pd.concat(all_samples, ignore_index=True) if all_samples else pd.DataFrame()
    if not samples_df.empty:
        bad = ~samples_df["is_tolerance_match"].fillna(False)
        bad &= samples_df["main_value"].notna() & samples_df["calc_value"].notna()
        mismatches_df = samples_df[bad].copy()
    else:
        mismatches_df = pd.DataFrame()

    out_cols = [
        "process_id",
        "target_id",
        "sample_id",
        "target_stream",
        "matched_stream_key",
        "target_species",
        "target_feature",
        "formula",
        "scale",
        "mole_flow",
        "frac_column",
        "frac_value",
        "main_value",
        "calc_value",
        "abs_diff",
        "rel_diff",
        "is_exact_match",
        "is_tolerance_match",
        "status",
    ]
    summary_df.to_csv(out_dir / "target_formula_summary.csv", index=False)
    if not samples_df.empty:
        samples_df[out_cols].to_csv(out_dir / "target_formula_samples.csv", index=False)
        mismatches_df[out_cols].to_csv(out_dir / "target_formula_mismatches.csv", index=False)
    else:
        pd.DataFrame(columns=out_cols).to_csv(out_dir / "target_formula_samples.csv", index=False)
        pd.DataFrame(columns=out_cols).to_csv(out_dir / "target_formula_mismatches.csv", index=False)

    prior_note = _compare_prior_validation(summary_df, _resolve_path(args.prior_validation))
    _write_plots(out_dir, summary_df, samples_df)
    _write_report(out_dir, summary_df, mismatches_df, ref_check, prior_note, join_meta)

    print(summary_df[["target_id", "status", "mismatch_rate", "r2_main_vs_calc", "diff_abs_max"]].to_string(index=False))
    print(f"[out] {out_dir}")

    bad = summary_df[~summary_df["status"].isin({"ok", "ok_with_tolerance", "ok_streams_only_no_main"})]
    return 0 if bad.empty else 2


if __name__ == "__main__":
    raise SystemExit(main())
