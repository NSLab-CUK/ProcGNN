"""Shared helpers for v4 target formula validation and alternate stream scans."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ABS_TOL = 1e-6
REL_TOL = 1e-5

SPECIES_FRAC = {"H2": "Frac_H2", "CO2": "Frac_CO2", "H2O": "Frac_H2O"}

SCAN_STATUSES = {
    "mismatch",
    "unresolved_main_column",
    "unresolved_stream_key",
    "unresolved_stream_column",
    "ambiguous_stream_alias",
    "ok_streams_only_no_main",
    "special_case_unresolved",
    "sample_join_problem",
}

MANDATORY_SCAN_TARGETS = frozenset(
    {"P02_T002", "P08_T001", "P08_T002", "P08_T003", "P06_T004"}
)


def resolve_path(raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def norm_key(value: Any) -> str:
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return re.sub(r"[^A-Z0-9]+", "", s.upper())


def norm_col(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def sample_col(df: pd.DataFrame) -> str | None:
    for c in ("ID", "sample_id", "Sample_ID", "id", "index"):
        if c in df.columns:
            return c
    return None


def is_long_format(streams: pd.DataFrame) -> bool:
    return any(
        c in streams.columns
        for c in ("Stream_Name", "stream_key", "stream_name", "main_data_stream_key")
    )


def find_column(columns: list[str], candidates: list[str]) -> tuple[str | None, str]:
    cmap = {norm_col(c): c for c in columns}
    tried: list[str] = []
    for cand in candidates:
        key = norm_col(cand)
        tried.append(cand)
        if key in cmap:
            return cmap[key], f"exact:{cand}"
    for cand in candidates:
        tokens = [t for t in re.split(r"[^a-z0-9]+", cand.lower()) if t]
        if not tokens:
            continue
        for ncol, orig in cmap.items():
            if all(t in ncol for t in tokens):
                return orig, f"fuzzy:{cand}"
    return None, f"not_found:{','.join(tried[:5])}"


def parse_scale_from_feature(target_feature: str, default: float = 1.0) -> tuple[float, str | None]:
    m = re.match(r"^([\d.]+)\s*\*\s*(.+)$", target_feature.replace(" ", ""))
    if m:
        return float(m.group(1)), m.group(2)
    return default, None


def main_target_series(
    main: pd.DataFrame,
    *,
    target_feature: str,
    target_species: str,
    scale: float = 1.0,
    main_column: str | None = None,
    special_stream17: bool = False,
) -> tuple[pd.Series, str, bool]:
    """Return (values, resolution_note, main_all_nan)."""
    if special_stream17 and "Stream17" in target_feature:
        c1, n1 = find_column(
            list(main.columns),
            ["Stream17_Mole_Flow", "Stream17_MoleFlow", "17_Mole_Flow"],
        )
        c2, n2 = find_column(
            list(main.columns),
            ["Stream17_Frac_CO2", "Stream17_FracCO2", "17_Frac_CO2"],
        )
        if c1 and c2:
            v = pd.to_numeric(main[c1], errors="coerce") * pd.to_numeric(main[c2], errors="coerce")
            return v, f"main_product:{n1}*{n2}", bool(v.isna().all())
        return pd.Series([float("nan")] * len(main)), "no_stream17_main_columns", True

    if main_column:
        col, note = find_column(list(main.columns), [main_column, target_feature])
        if col:
            v = pd.to_numeric(main[col], errors="coerce")
            if scale != 1.0:
                v = v * scale
            return v, note, bool(v.isna().all())

    col, note = find_column(list(main.columns), [target_feature])
    if col:
        v = pd.to_numeric(main[col], errors="coerce")
        return v, note, bool(v.isna().all())

    sc, inner = parse_scale_from_feature(target_feature)
    if inner:
        col, note = find_column(list(main.columns), [inner, inner.replace("_", "")])
        if col:
            v = pd.to_numeric(main[col], errors="coerce") * sc
            return v, f"scaled_expr:{note}", bool(v.isna().all())

    return pd.Series([float("nan")] * len(main)), f"missing:{target_feature}", True


def list_stream_candidates(streams: pd.DataFrame) -> list[str]:
    if is_long_format(streams):
        stream_col = next(
            (
                c
                for c in ("Stream_Name", "stream_key", "stream_name", "main_data_stream_key")
                if c in streams.columns
            ),
            None,
        )
        if stream_col is None:
            return []
        return sorted(streams[stream_col].astype(str).unique().tolist(), key=lambda x: (norm_key(x), x))

    prefixes: set[str] = set()
    for c in streams.columns:
        for suffix in ("_Mole_Flow", "_Frac_H2", "_Frac_CO2", "_Frac_H2O"):
            if c.endswith(suffix):
                prefixes.add(c[: -len(suffix)])
    return sorted(prefixes, key=lambda x: (norm_key(x), x))


def extract_stream_calc(
    streams: pd.DataFrame,
    stream_key: str,
    frac_column: str,
    scale: float,
) -> pd.DataFrame:
    sid = sample_col(streams) or "ID"
    if is_long_format(streams):
        stream_col = next(
            (
                c
                for c in ("Stream_Name", "stream_key", "stream_name", "main_data_stream_key")
                if c in streams.columns
            ),
            None,
        )
        sub = streams[streams[stream_col].astype(str).map(norm_key) == norm_key(stream_key)].copy()
        if sub.empty:
            return pd.DataFrame()
        if sub.duplicated([sid]).any():
            sub = sub.sort_values([sid, stream_col]).drop_duplicates([sid], keep="first")
        out = sub[[sid, "Mole_Flow", frac_column]].copy()
        out = out.rename(columns={sid: "sample_id"})
        out["matched_stream_key"] = stream_key
    else:
        mole_col, _ = find_column(
            list(streams.columns),
            [f"{stream_key}_Mole_Flow", f"{stream_key}.Mole_Flow"],
        )
        frac_col, _ = find_column(
            list(streams.columns),
            [f"{stream_key}_{frac_column}", f"{stream_key}.{frac_column}"],
        )
        if mole_col is None or frac_col is None:
            return pd.DataFrame()
        out = streams[[sid, mole_col, frac_col]].copy()
        out = out.rename(
            columns={sid: "sample_id", mole_col: "Mole_Flow", frac_col: frac_column}
        )
        out["matched_stream_key"] = stream_key

    mole = pd.to_numeric(out["Mole_Flow"], errors="coerce")
    frac = pd.to_numeric(out[frac_column], errors="coerce")
    out["calc_value"] = scale * mole * frac
    out["mole_flow"] = mole
    out["frac_value"] = frac
    out["frac_column"] = frac_column
    return out


def tolerance_match(abs_diff: float, rel_diff: float) -> bool:
    if math.isnan(abs_diff):
        return False
    if abs_diff <= ABS_TOL:
        return True
    return not math.isnan(rel_diff) and rel_diff <= REL_TOL


def r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
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


def compare_main_calc(
    main: pd.DataFrame,
    streams: pd.DataFrame,
    *,
    target_feature: str,
    target_species: str,
    scale: float = 1.0,
    main_column: str | None = None,
    special_stream17: bool = False,
    stream_key: str,
) -> dict[str, Any]:
    frac_column = SPECIES_FRAC.get(target_species, "Frac_CO2")
    main_vals, main_note, main_all_nan = main_target_series(
        main,
        target_feature=target_feature,
        target_species=target_species,
        scale=scale,
        main_column=main_column,
        special_stream17=special_stream17,
    )
    sid_m = sample_col(main)
    main_work = main.copy()
    main_work["sample_id"] = (
        main_work[sid_m].astype(str) if sid_m else main_work.index.astype(str)
    )
    if main_work["sample_id"].duplicated().any():
        main_work = main_work.drop_duplicates("sample_id", keep="last")
    main_work["main_value"] = main_vals

    if main_all_nan:
        calc = extract_stream_calc(streams, stream_key, frac_column, scale)
        n_calc = int(calc["calc_value"].notna().sum()) if not calc.empty else 0
        return {
            "n_main": len(main),
            "n_candidate": n_calc,
            "n_joined": 0,
            "n_valid": 0,
            "main_all_nan": True,
            "main_resolution": main_note,
            "candidate_status": "main_all_nan",
            "r2_main_vs_calc": float("nan"),
            "mismatch_rate": float("nan"),
            "diff_abs_max": float("nan"),
            "diff_abs_mean": float("nan"),
            "tolerance_match_count": 0,
            "mismatch_count": 0,
        }

    calc = extract_stream_calc(streams, stream_key, frac_column, scale)
    if calc.empty:
        return {
            "n_main": len(main),
            "n_candidate": 0,
            "n_joined": 0,
            "n_valid": 0,
            "main_all_nan": False,
            "main_resolution": main_note,
            "candidate_status": "unresolved_candidate_columns",
            "r2_main_vs_calc": float("nan"),
            "mismatch_rate": float("nan"),
            "diff_abs_max": float("nan"),
            "diff_abs_mean": float("nan"),
            "tolerance_match_count": 0,
            "mismatch_count": 0,
        }

    calc["sample_id"] = calc["sample_id"].astype(str)
    joined = main_work[["sample_id", "main_value"]].merge(calc, on="sample_id", how="inner")
    valid = joined["main_value"].notna() & joined["calc_value"].notna()
    n_valid = int(valid.sum())
    if n_valid == 0:
        st = "unresolved_main" if main_vals.isna().all() else "sample_join_problem"
        return {
            "n_main": len(main),
            "n_candidate": len(calc),
            "n_joined": len(joined),
            "n_valid": 0,
            "main_all_nan": main_all_nan,
            "main_resolution": main_note,
            "candidate_status": st,
            "r2_main_vs_calc": float("nan"),
            "mismatch_rate": float("nan"),
            "diff_abs_max": float("nan"),
            "diff_abs_mean": float("nan"),
            "tolerance_match_count": 0,
            "mismatch_count": 0,
        }

    g = joined[valid]
    abs_diff = (g["main_value"] - g["calc_value"]).abs()
    denom = g["main_value"].abs().where(g["main_value"].abs() > 0, 1.0)
    rel_diff = abs_diff / denom
    tol = abs_diff.apply(lambda x: x <= ABS_TOL) | rel_diff.le(REL_TOL)
    n_tol = int(tol.sum())
    n_mis = int((~tol).sum())
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    r2 = r2_score(g["main_value"], g["calc_value"])

    if n_mis == 0 and n_tol == n_valid:
        if int((abs_diff <= ABS_TOL).sum()) == n_valid:
            cst = "exact_match"
        else:
            cst = "tolerance_match"
    elif math.isfinite(r2) and r2 > 0.99 and mean_abs < 1e-3:
        cst = "close_but_not_exact"
    else:
        cst = "mismatch"

    return {
        "n_main": len(main),
        "n_candidate": len(calc),
        "n_joined": len(joined),
        "n_valid": n_valid,
        "main_all_nan": main_all_nan,
        "main_resolution": main_note,
        "main_mean": float(g["main_value"].mean()),
        "calc_mean": float(g["calc_value"].mean()),
        "main_std": float(g["main_value"].std()),
        "calc_std": float(g["calc_value"].std()),
        "diff_mean": float((g["calc_value"] - g["main_value"]).mean()),
        "diff_abs_mean": mean_abs,
        "diff_abs_median": float(abs_diff.median()),
        "diff_abs_max": max_abs,
        "rel_diff_mean": float(rel_diff.mean()),
        "rel_diff_median": float(rel_diff.median()),
        "rel_diff_max": float(rel_diff.max()),
        "corr": float(g["main_value"].corr(g["calc_value"])),
        "r2_main_vs_calc": r2,
        "tolerance_match_count": n_tol,
        "mismatch_count": n_mis,
        "mismatch_rate": float(n_mis / n_valid) if n_valid else float("nan"),
        "candidate_status": cst,
    }


def load_target_formula_table() -> pd.DataFrame:
    from validate_all_v4_target_formulas import TARGET_FORMULAS, _enrich_from_v4_targets, _target_row

    rows = [_target_row(d) for d in TARGET_FORMULAS]
    _enrich_from_v4_targets(rows)
    return pd.DataFrame(
        [
            {
                "process_id": r.process_id,
                "target_id": r.target_id,
                "target_stream": r.target_stream,
                "target_species": r.target_species,
                "target_feature": r.target_feature,
                "formula": r.formula,
                "scale": r.scale,
                "frac_column": r.frac_column,
                "special_case": r.special_case,
                "reference_edge_id": r.reference_edge_id,
                "required_stream_key": r.required_stream_key,
                "main_column": r.main_column,
            }
            for r in rows
        ]
    )


def edge_lookup(canonical: pd.DataFrame) -> dict[tuple[str, str], dict[str, str]]:
    """(normalized_stream_key, process_id) -> edge info; also by edge id."""
    by_edge: dict[str, dict[str, str]] = {}
    by_stream: dict[tuple[int, str], dict[str, str]] = {}
    for _, row in canonical.iterrows():
        eid = str(row.get("canonical_edge_id", ""))
        pid = int(row.get("process_id", 0))
        sk = norm_key(row.get("main_data_stream_key", ""))
        info = {
            "canonical_edge_id": eid,
            "canonical_main_data_stream_key": str(row.get("main_data_stream_key", "")),
            "dst_node": str(row.get("dst_node", "")),
            "is_v_output": str(row.get("dst_node", "")) == "V_OUTPUT",
            "stream_name_norm": str(row.get("stream_name_norm", "")),
        }
        by_edge[eid] = info
        if sk:
            by_stream[(pid, sk)] = info
    return {"by_edge": by_edge, "by_stream": by_stream}
