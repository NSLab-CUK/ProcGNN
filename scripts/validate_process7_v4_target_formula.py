#!/usr/bin/env python3
"""Validate Process7 v4 target formulas against Main and Streams CSVs."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL_EDGES = PROJECT_ROOT / "data/reference/v3/canonical_edges.csv"
DEFAULT_TARGET_ANSWERS = PROJECT_ROOT / "data/reference/v3/target_answer_edges.csv"

ABS_TOL = 1e-6
REL_TOL = 1e-5


@dataclass(frozen=True)
class TargetSpec:
    target_id: str
    target_column: str
    target_species: str
    stream_key: str
    answer_edge: str
    frac_column: str
    formula: str


TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec(
        target_id="P07_T001",
        target_column="PROD_H2_Mole",
        target_species="H2",
        stream_key="PROD",
        answer_edge="P07_E016",
        frac_column="Frac_H2",
        formula="Mole_Flow * Frac_H2",
    ),
    TargetSpec(
        target_id="P07_T002",
        target_column="PROD_CO2_Mole",
        target_species="CO2",
        stream_key="PROD",
        answer_edge="P07_E016",
        frac_column="Frac_CO2",
        formula="Mole_Flow * Frac_CO2",
    ),
    TargetSpec(
        target_id="P07_T003",
        target_column="RE_H2O_Mole",
        target_species="H2O",
        stream_key="RE",
        answer_edge="P07_E017",
        frac_column="Frac_H2O",
        formula="Mole_Flow * Frac_H2O",
    ),
)


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _norm(value: Any) -> str:
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s.upper()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _sample_col(df: pd.DataFrame) -> str:
    for c in ("ID", "sample_id", "Sample_ID", "id"):
        if c in df.columns:
            return c
    raise KeyError(f"sample ID column not found. columns={list(df.columns)[:20]}")


def _load_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    canonical = _read_csv(DEFAULT_CANONICAL_EDGES)
    answers = _read_csv(DEFAULT_TARGET_ANSWERS)
    return canonical, answers


def _edge_reference_rows(canonical: pd.DataFrame, answers: pd.DataFrame) -> pd.DataFrame:
    ce = canonical[
        (pd.to_numeric(canonical["process_id"], errors="coerce") == 7)
        & (canonical["canonical_edge_id"].astype(str).isin(["P07_E016", "P07_E017"]))
    ].copy()
    ans = answers[pd.to_numeric(answers["process_id"], errors="coerce") == 7].copy()
    rows: list[dict[str, Any]] = []
    for spec in TARGETS:
        ce_match = ce[ce["canonical_edge_id"].astype(str) == spec.answer_edge]
        ans_match = ans[ans["canonical_answer_edge_id"].astype(str) == spec.answer_edge]
        if spec.target_column in set(ans_match.get("target_column", pd.Series(dtype=str)).astype(str)):
            ans_match = ans_match[ans_match["target_column"].astype(str) == spec.target_column]
        rows.append(
            {
                "target_id": spec.target_id,
                "target_column": spec.target_column,
                "answer_edge": spec.answer_edge,
                "requested_stream_key": spec.stream_key,
                "canonical_main_data_stream_key": (
                    str(ce_match.iloc[0].get("main_data_stream_key", "")) if not ce_match.empty else ""
                ),
                "canonical_stream_name_norm": (
                    str(ce_match.iloc[0].get("stream_name_norm", "")) if not ce_match.empty else ""
                ),
                "canonical_src_node": str(ce_match.iloc[0].get("src_node", "")) if not ce_match.empty else "",
                "canonical_dst_node": str(ce_match.iloc[0].get("dst_node", "")) if not ce_match.empty else "",
                "canonical_dst_node_raw": (
                    str(ce_match.iloc[0].get("dst_node_raw", "")) if not ce_match.empty else ""
                ),
                "target_answer_edges_rows": int(len(ans_match)),
                "target_answer_edges_stream_key": (
                    str(ans_match.iloc[0].get("main_data_stream_key", "")) if not ans_match.empty else ""
                ),
                "target_answer_edges_task": (
                    str(ans_match.iloc[0].get("task_name", "")) if not ans_match.empty else ""
                ),
                "reference_status": "ok" if not ce_match.empty else "missing_canonical_edge",
            }
        )
    return pd.DataFrame(rows)


def _stream_aliases(spec: TargetSpec, edge_ref: pd.DataFrame) -> list[str]:
    aliases = [
        spec.stream_key,
        f"OUT_{spec.stream_key}",
    ]
    if spec.stream_key == "RE":
        aliases.extend(["OUT_H2O", "OUT_RE", "RESTEAM"])
    row = edge_ref[edge_ref["target_id"].astype(str) == spec.target_id]
    if not row.empty:
        for c in ("canonical_main_data_stream_key", "canonical_stream_name_norm", "canonical_dst_node_raw"):
            v = str(row.iloc[0].get(c, "")).strip()
            if v:
                aliases.append(v)
    out: list[str] = []
    seen: set[str] = set()
    for a in aliases:
        na = _norm(a)
        if na and na not in seen:
            seen.add(na)
            out.append(a)
    return out


def _select_stream_long(streams: pd.DataFrame, sample_col: str, spec: TargetSpec, edge_ref: pd.DataFrame) -> pd.DataFrame:
    stream_col = "Stream_Name"
    if stream_col not in streams.columns:
        for c in ("stream_key", "stream_name", "main_data_stream_key"):
            if c in streams.columns:
                stream_col = c
                break
    if stream_col not in streams.columns:
        raise KeyError("long streams format requires Stream_Name/stream_key column")
    aliases = {_norm(a) for a in _stream_aliases(spec, edge_ref)}
    sub = streams[streams[stream_col].map(_norm).isin(aliases)].copy()
    if sub.empty:
        raise ValueError(f"no Streams rows for {spec.target_id} aliases={sorted(aliases)}")
    needed = [sample_col, stream_col, "Mole_Flow", spec.frac_column]
    missing = [c for c in needed if c not in sub.columns]
    if missing:
        raise KeyError(f"missing stream columns for {spec.target_id}: {missing}")
    if sub.duplicated([sample_col, stream_col]).any():
        # Keep first after sorting; duplicate count is reported separately.
        sub = sub.sort_values([sample_col, stream_col]).drop_duplicates([sample_col], keep="first")
    else:
        sub = sub.sort_values([sample_col, stream_col]).drop_duplicates([sample_col], keep="first")
    return sub[[sample_col, stream_col, "Mole_Flow", spec.frac_column]].rename(
        columns={sample_col: "sample_id", stream_col: "stream_key"}
    )


def _find_wide_column(streams: pd.DataFrame, aliases: list[str], suffixes: list[str]) -> str | None:
    norm_cols = {_norm(c).replace(" ", "").replace("*", ""): c for c in streams.columns}
    for alias in aliases:
        a = _norm(alias).replace(" ", "")
        for suffix in suffixes:
            candidates = [
                f"{a}_{suffix}".upper(),
                f"{a}.{suffix}".upper(),
                f"{a}{suffix}".upper(),
            ]
            for cand in candidates:
                key = cand.replace(" ", "").replace("*", "")
                if key in norm_cols:
                    return norm_cols[key]
    return None


def _select_stream_wide(streams: pd.DataFrame, sample_col: str, spec: TargetSpec, edge_ref: pd.DataFrame) -> pd.DataFrame:
    aliases = _stream_aliases(spec, edge_ref)
    mole_col = _find_wide_column(streams, aliases, ["Mole_Flow", "MOLE_FLOW"])
    frac_col = _find_wide_column(streams, aliases, [spec.frac_column])
    if mole_col is None or frac_col is None:
        raise KeyError(f"wide columns not found for {spec.target_id}: mole={mole_col} frac={frac_col}")
    out = streams[[sample_col, mole_col, frac_col]].copy()
    out["stream_key"] = spec.stream_key
    return out.rename(columns={sample_col: "sample_id", mole_col: "Mole_Flow", frac_col: spec.frac_column})


def _select_stream(streams: pd.DataFrame, sample_col: str, spec: TargetSpec, edge_ref: pd.DataFrame) -> pd.DataFrame:
    if any(c in streams.columns for c in ("Stream_Name", "stream_key", "stream_name", "main_data_stream_key")):
        return _select_stream_long(streams, sample_col, spec, edge_ref)
    return _select_stream_wide(streams, sample_col, spec, edge_ref)


def _r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    yt = pd.to_numeric(y_true, errors="coerce").astype(float)
    yp = pd.to_numeric(y_pred, errors="coerce").astype(float)
    valid = yt.notna() & yp.notna()
    yt = yt[valid]
    yp = yp[valid]
    if len(yt) < 2:
        return float("nan")
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    if ss_tot <= 0:
        return 1.0 if ss_res <= 0 else float("nan")
    return 1.0 - ss_res / ss_tot


def _corr(y_true: pd.Series, y_pred: pd.Series) -> float:
    yt = pd.to_numeric(y_true, errors="coerce").astype(float)
    yp = pd.to_numeric(y_pred, errors="coerce").astype(float)
    valid = yt.notna() & yp.notna()
    if int(valid.sum()) < 2:
        return float("nan")
    return float(yt[valid].corr(yp[valid]))


def _build_samples(main: pd.DataFrame, streams: pd.DataFrame, edge_ref: pd.DataFrame) -> pd.DataFrame:
    main_id = _sample_col(main)
    stream_id = _sample_col(streams)
    rows: list[pd.DataFrame] = []
    for spec in TARGETS:
        if spec.target_column not in main.columns:
            raise KeyError(f"missing Main target column: {spec.target_column}")
        s = _select_stream(streams, stream_id, spec, edge_ref)
        m = main[[main_id, spec.target_column]].rename(
            columns={main_id: "sample_id", spec.target_column: "main_value"}
        )
        joined = m.merge(s, on="sample_id", how="inner", validate="one_to_one")
        joined["target_id"] = spec.target_id
        joined["target_column"] = spec.target_column
        joined["target_species"] = spec.target_species
        joined["answer_edge"] = spec.answer_edge
        joined["formula"] = spec.formula
        joined["frac_column"] = spec.frac_column
        joined["frac_value"] = pd.to_numeric(joined[spec.frac_column], errors="coerce")
        joined["mole_flow"] = pd.to_numeric(joined["Mole_Flow"], errors="coerce")
        joined["main_value"] = pd.to_numeric(joined["main_value"], errors="coerce")
        joined["calc_value"] = joined["mole_flow"] * joined["frac_value"]
        joined["abs_diff"] = (joined["main_value"] - joined["calc_value"]).abs()
        denom = joined["main_value"].abs().where(joined["main_value"].abs() > 0, 1.0)
        joined["rel_diff"] = joined["abs_diff"] / denom
        joined["is_match"] = (joined["abs_diff"] <= ABS_TOL) | (joined["rel_diff"] <= REL_TOL)
        rows.append(
            joined[
                [
                    "sample_id",
                    "target_id",
                    "target_column",
                    "target_species",
                    "answer_edge",
                    "formula",
                    "stream_key",
                    "mole_flow",
                    "frac_column",
                    "frac_value",
                    "main_value",
                    "calc_value",
                    "abs_diff",
                    "rel_diff",
                    "is_match",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def _summary(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (tid, target_col), g in samples.groupby(["target_id", "target_column"], sort=False):
        rows.append(
            {
                "target_id": tid,
                "target_column": target_col,
                "target_species": str(g["target_species"].iloc[0]),
                "answer_edge": str(g["answer_edge"].iloc[0]),
                "stream_key": str(g["stream_key"].iloc[0]),
                "formula": str(g["formula"].iloc[0]),
                "n_samples": int(len(g)),
                "main_mean": float(g["main_value"].mean()),
                "calc_mean": float(g["calc_value"].mean()),
                "diff_mean": float((g["calc_value"] - g["main_value"]).mean()),
                "diff_abs_mean": float(g["abs_diff"].mean()),
                "diff_abs_max": float(g["abs_diff"].max()),
                "relative_error_mean": float(g["rel_diff"].mean()),
                "relative_error_max": float(g["rel_diff"].max()),
                "corr": _corr(g["main_value"], g["calc_value"]),
                "r2_between_main_and_calc": _r2(g["main_value"], g["calc_value"]),
                "exact_match_count": int(g["is_match"].sum()),
                "mismatch_count": int((~g["is_match"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def _diagnose(summary: pd.DataFrame, main: pd.DataFrame, streams: pd.DataFrame, samples: pd.DataFrame) -> str:
    if summary.empty:
        return "validation_failed_no_summary"
    if int(summary["mismatch_count"].sum()) == 0:
        return "ok_all_targets_match"
    if len(set(main[_sample_col(main)].astype(str)) & set(streams[_sample_col(streams)].astype(str))) < len(main):
        return "sample_join_problem"
    if (summary["corr"].fillna(0) > 0.999).all() and (summary["relative_error_mean"] > 1e-4).any():
        return "possible_scale_problem"
    if samples["stream_key"].isna().any():
        return "stream_key_problem"
    return "value_mismatch_review_stream_key_or_column_alias"


def _write_report(
    out_dir: Path,
    *,
    main_path: Path,
    streams_path: Path,
    edge_ref: pd.DataFrame,
    summary: pd.DataFrame,
    samples: pd.DataFrame,
    mismatches: pd.DataFrame,
    diagnosis: str,
) -> None:
    def _md_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "(empty)"
        cols = [str(c) for c in df.columns]
        lines = [
            "| " + " | ".join(cols) + " |",
            "| " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for _, row in df.iterrows():
            vals = [str(row.get(c, "")).replace("\n", " ") for c in df.columns]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines: list[str] = []
    lines.append("# Process7 Mapping Validation Report")
    lines.append("")
    lines.append(f"- Main file: `{main_path.as_posix()}`")
    lines.append(f"- Streams file: `{streams_path.as_posix()}`")
    lines.append(f"- Absolute tolerance: `{ABS_TOL}`")
    lines.append(f"- Relative tolerance: `{REL_TOL}`")
    lines.append("")
    lines.append("## Canonical Edge Mapping")
    lines.append("")
    lines.append(_md_table(edge_ref))
    lines.append("")
    lines.append("## Formula Validation Summary")
    lines.append("")
    lines.append(_md_table(summary))
    lines.append("")
    all_ok = int(summary["mismatch_count"].sum()) == 0 if not summary.empty else False
    lines.append(f"## Result: {'PASS' if all_ok else 'FAIL'}")
    lines.append("")
    lines.append(f"- Diagnosis: `{diagnosis}`")
    lines.append(
        "- Main target values and Streams-derived formula values "
        + ("match for all three Process7 targets." if all_ok else "do not fully match.")
    )
    lines.append(
        "- `target_metrics_v4.csv` may use this mapping directly."
        if all_ok
        else "- Do not interpret `target_metrics_v4.csv` for Process7 until mismatches are resolved."
    )
    lines.append("")
    lines.append("## Mismatch Examples")
    lines.append("")
    if mismatches.empty:
        lines.append("No mismatches.")
    else:
        cols = [
            "sample_id",
            "target_id",
            "target_column",
            "stream_key",
            "main_value",
            "calc_value",
            "abs_diff",
            "rel_diff",
        ]
        lines.append(_md_table(mismatches[cols].head(20)))
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `process7_target_formula_summary.csv`")
    lines.append("- `process7_target_formula_samples.csv`")
    lines.append("- `process7_target_formula_mismatches.csv`")
    (out_dir / "PROCESS7_MAPPING_VALIDATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Process7 v4 target formula mapping.")
    parser.add_argument("--main", required=True)
    parser.add_argument("--streams", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    main_path = _resolve_path(args.main)
    streams_path = _resolve_path(args.streams)
    out_dir = _resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_df = _read_csv(main_path)
    streams_df = _read_csv(streams_path)
    canonical, answers = _load_reference()
    edge_ref = _edge_reference_rows(canonical, answers)

    samples = _build_samples(main_df, streams_df, edge_ref)
    summary = _summary(samples)
    mismatches = samples[~samples["is_match"]].copy()
    diagnosis = _diagnose(summary, main_df, streams_df, samples)

    summary.to_csv(out_dir / "process7_target_formula_summary.csv", index=False)
    samples.to_csv(out_dir / "process7_target_formula_samples.csv", index=False)
    mismatches.to_csv(out_dir / "process7_target_formula_mismatches.csv", index=False)
    edge_ref.to_csv(out_dir / "process7_canonical_edge_reference.csv", index=False)
    _write_report(
        out_dir,
        main_path=main_path,
        streams_path=streams_path,
        edge_ref=edge_ref,
        summary=summary,
        samples=samples,
        mismatches=mismatches,
        diagnosis=diagnosis,
    )

    print(summary.to_string(index=False))
    print(f"[diagnosis] {diagnosis}")
    print(f"[out] {out_dir}")
    return 0 if mismatches.empty else 2


if __name__ == "__main__":
    raise SystemExit(main())
