#!/usr/bin/env python3
"""Scan all stream candidates vs Main target values for problematic v4 targets."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from v4_validation_common import (  # noqa: E402
    MANDATORY_SCAN_TARGETS,
    PROJECT_ROOT,
    SCAN_STATUSES,
    compare_main_calc,
    edge_lookup,
    list_stream_candidates,
    load_target_formula_table,
    norm_key,
    resolve_path,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _targets_to_scan(summary_path: Path, formulas: pd.DataFrame) -> pd.DataFrame:
    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        bad = summary[summary["status"].astype(str).isin(SCAN_STATUSES)]
        ids = set(bad["target_id"].astype(str)) | set(MANDATORY_SCAN_TARGETS)
        return formulas[formulas["target_id"].astype(str).isin(ids)].copy()
    return formulas[formulas["target_id"].astype(str).isin(MANDATORY_SCAN_TARGETS)].copy()


def _scan_target(
    row: pd.Series,
    main: pd.DataFrame,
    streams: pd.DataFrame,
    canonical: pd.DataFrame,
) -> list[dict]:
    pid = int(row["process_id"])
    tid = str(row["target_id"])
    target_feature = str(row["target_feature"])
    target_species = str(row["target_species"])
    declared_stream = str(row["target_stream"])
    scale = float(row.get("scale", 1.0))
    main_column = row.get("main_column")
    main_column = None if pd.isna(main_column) else str(main_column)
    special = bool(row.get("special_case", False))

    elu = edge_lookup(canonical)
    candidates = list_stream_candidates(streams)
    if special and "Stream17" in target_feature:
        for pref in ("17", "Stream17"):
            if pref not in candidates:
                candidates.insert(0, pref)

    rows: list[dict] = []
    for sk in candidates:
        stats = compare_main_calc(
            main,
            streams,
            target_feature=target_feature,
            target_species=target_species,
            scale=scale,
            main_column=main_column,
            special_stream17=special,
            stream_key=sk,
        )
        info = elu["by_stream"].get((pid, norm_key(sk)), {})
        eid = str(info.get("canonical_edge_id", ""))
        rows.append(
            {
                "process_id": pid,
                "target_id": tid,
                "target_feature": target_feature,
                "target_species": target_species,
                "declared_target_stream": declared_stream,
                "candidate_stream_key": sk,
                "candidate_edge_id": eid,
                "candidate_dst_node": info.get("dst_node", ""),
                "is_v_output": info.get("is_v_output", False),
                "reference_edge_id": str(row.get("reference_edge_id", "")),
                "required_stream_key": str(row.get("required_stream_key", "")),
                **stats,
            }
        )
    return rows


def _rank_scan(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for tid, g in out.groupby("target_id"):
        idx = g.index
        out.loc[idx, "rank_by_diff_abs_mean"] = (
            g["diff_abs_mean"].astype(float).rank(method="dense", ascending=True, na_option="bottom")
        )
        out.loc[idx, "rank_by_diff_abs_max"] = (
            g["diff_abs_max"].astype(float).rank(method="dense", ascending=True, na_option="bottom")
        )
        out.loc[idx, "rank_by_r2"] = (
            g["r2_main_vs_calc"].astype(float).rank(method="dense", ascending=False, na_option="bottom")
        )
    return out


def _top_candidates(scan: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for tid, g in scan.groupby("target_id"):
        g = g[g["candidate_status"].astype(str) != "unresolved_candidate_columns"]
        if g.empty:
            continue
        top_mean = g.nsmallest(10, "rank_by_diff_abs_mean")
        top_mean["top_kind"] = "rank_by_diff_abs_mean"
        top_r2 = g.nsmallest(10, "rank_by_r2")
        top_r2["top_kind"] = "rank_by_r2"
        exact = g[g["candidate_status"].isin(["exact_match", "tolerance_match"])]
        if not exact.empty:
            exact = exact.copy()
            exact["top_kind"] = "exact_or_tolerance_match"
            parts.append(exact)
        parts.extend([top_mean, top_r2])
    if not parts:
        return pd.DataFrame()
    top = pd.concat(parts, ignore_index=True).drop_duplicates(
        subset=["target_id", "candidate_stream_key", "top_kind"]
    )
    return top.sort_values(["target_id", "top_kind", "rank_by_diff_abs_mean"])


def _write_plots(out_dir: Path, scan: pd.DataFrame, top: pd.DataFrame) -> None:
    if not HAS_MPL or scan.empty:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for tid, g in scan.groupby("target_id"):
        sub = g[g["diff_abs_mean"].notna()].nsmallest(15, "rank_by_diff_abs_mean")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(sub["candidate_stream_key"].astype(str), sub["diff_abs_mean"].astype(float))
        ax.set_title(f"{tid}: diff_abs_mean by stream candidate")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(plot_dir / f"scan_{tid}_diff_abs_mean.png", dpi=120)
        plt.close(fig)

    if not top.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        best = top.groupby("target_id").first().reset_index()
        ok = top[top["candidate_status"].isin(["exact_match", "tolerance_match"])]
        if not ok.empty:
            for _, r in ok.iterrows():
                ax.scatter(
                    str(r["target_id"]),
                    float(r["diff_abs_max"]),
                    c="green",
                    s=80,
                    marker="*",
                    zorder=3,
                )
        ax.bar(best["target_id"].astype(str), best["diff_abs_max"].astype(float))
        ax.set_ylabel("diff_abs_max (best per target in top table)")
        ax.set_title("Alternate stream scan: best candidates")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(plot_dir / "scan_best_candidates.png", dpi=120)
        plt.close(fig)


def _write_report(out_dir: Path, scan: pd.DataFrame, top: pd.DataFrame, targets: pd.DataFrame) -> None:
    lines = [
        "# Alternate Stream Scan Report",
        "",
        f"- Targets scanned: **{targets['target_id'].nunique()}**",
        f"- Total candidate rows: **{len(scan)}**",
        "",
        "## Mandatory targets",
        "",
    ]
    for tid in sorted(MANDATORY_SCAN_TARGETS):
        sub = scan[scan["target_id"].astype(str) == tid]
        if sub.empty:
            lines.append(f"- **{tid}**: no scan rows")
            continue
        rank_col = pd.to_numeric(sub["rank_by_diff_abs_mean"], errors="coerce")
        best = sub.loc[rank_col.idxmin()] if rank_col.notna().any() else sub.iloc[0]
        exact = sub[sub["candidate_status"].isin(["exact_match", "tolerance_match"])]
        lines.append(
            f"- **{tid}**: best stream `{best['candidate_stream_key']}` "
            f"(status={best['candidate_status']}, diff_abs_max={best['diff_abs_max']}, "
            f"r2={best['r2_main_vs_calc']}); "
            f"exact/tolerance matches: {', '.join(exact['candidate_stream_key'].astype(str).tolist()) or 'none'}"
        )
    lines.extend(["", "## Top candidates file", "", "See `alternate_stream_scan_top_candidates.csv`.", ""])
    (out_dir / "alternate_stream_scan_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan alternate stream keys vs Main targets.")
    parser.add_argument("--main-dir", default="data/main_data")
    parser.add_argument("--streams-dir", default="data/main_data_Streams")
    parser.add_argument("--target-ref", default="data/reference/v3/target_answer_edges.csv")
    parser.add_argument("--edge-ref", default="data/reference/v3/canonical_edges.csv")
    parser.add_argument("--validation-dir", default="outputs/v4_target_formula_validation_all")
    parser.add_argument("--out", default="outputs/v4_target_formula_validation_all/alternate_stream_scan")
    args = parser.parse_args()

    main_dir = resolve_path(args.main_dir)
    streams_dir = resolve_path(args.streams_dir)
    out_dir = resolve_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    formulas = load_target_formula_table()
    summary_path = resolve_path(args.validation_dir) / "target_formula_summary.csv"
    targets = _targets_to_scan(summary_path, formulas)
    canonical = pd.read_csv(resolve_path(args.edge_ref))

    all_rows: list[dict] = []
    for _, row in targets.iterrows():
        pid = int(row["process_id"])
        main_path = main_dir / f"{pid}.Process_Main.csv"
        streams_path = streams_dir / f"{pid}.Process_Streams.csv"
        if not main_path.is_file() or not streams_path.is_file():
            continue
        main = pd.read_csv(main_path)
        streams = pd.read_csv(streams_path)
        all_rows.extend(_scan_target(row, main, streams, canonical))

    scan = pd.DataFrame(all_rows)
    if scan.empty:
        print("[scan] no rows produced")
        return 1
    scan = _rank_scan(scan)
    top = _top_candidates(scan)

    scan.to_csv(out_dir / "alternate_stream_scan.csv", index=False)
    top.to_csv(out_dir / "alternate_stream_scan_top_candidates.csv", index=False)
    _write_plots(out_dir, scan, top)
    _write_report(out_dir, scan, top, targets)

    print(f"[scan] targets={targets['target_id'].nunique()} rows={len(scan)}")
    print(f"[out] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
