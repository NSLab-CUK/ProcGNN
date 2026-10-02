"""Load v4 target provenance policy and apply macro inclusion flags."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PROVENANCE_SEARCH: tuple[Path, ...] = (
    PROJECT_ROOT / "outputs/v4_target_formula_validation_all/provenance_decisions/v4_target_provenance_decisions.csv",
    PROJECT_ROOT / "data/reference/v4/v4_target_provenance_decisions.csv",
)

PROVENANCE_METRIC_COLUMNS: tuple[str, ...] = (
    "formula_status",
    "target_provenance",
    "decision_status",
    "include_in_main_verified_macro",
    "include_in_formula_macro",
    "include_in_streams_only_macro",
    "exclusion_reason",
    "reference_answer_edge_id",
    "used_stream_key",
    "used_formula",
    "scale",
    "amount_source",
)


def resolve_provenance_path(explicit: Path | str | None = None) -> Path | None:
    if explicit is not None:
        p = Path(explicit)
        return p if p.is_file() else None
    for cand in DEFAULT_PROVENANCE_SEARCH:
        if cand.is_file():
            return cand
    return None


def load_provenance_policy(path: Path | str | None = None) -> pd.DataFrame:
    resolved = resolve_provenance_path(path)
    if resolved is None:
        return pd.DataFrame()
    df = pd.read_csv(resolved)
    for col in ("include_in_main_verified_macro", "include_in_formula_macro", "include_in_streams_only_macro"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(("true", "1", "yes"))
    return df


def merge_provenance_into_metrics(
    metric_df: pd.DataFrame,
    provenance: pd.DataFrame,
) -> pd.DataFrame:
    if metric_df.empty or provenance.empty:
        return metric_df
    cols = [c for c in PROVENANCE_METRIC_COLUMNS if c in provenance.columns]
    sub = provenance[["target_id"] + cols].drop_duplicates("target_id")
    out = metric_df.merge(sub, on="target_id", how="left")
    if "include_in_main_verified_macro" in out.columns:
        out["include_in_main_verified_macro"] = out["include_in_main_verified_macro"].fillna(True)
    if "include_in_formula_macro" in out.columns:
        out["include_in_formula_macro"] = out["include_in_formula_macro"].fillna(
            out.get("include_in_main_verified_macro", True)
        )
    if "include_in_streams_only_macro" in out.columns:
        out["include_in_streams_only_macro"] = out["include_in_streams_only_macro"].fillna(False)
    return out


def _macro_mean_r2(frame: pd.DataFrame, flag_col: str) -> float:
    if frame.empty or flag_col not in frame.columns:
        return float("nan")
    sub = frame[frame[flag_col].astype(bool)]
    if sub.empty:
        return float("nan")
    r2s = pd.to_numeric(sub["r2"], errors="coerce").dropna()
    return float(r2s.mean()) if not r2s.empty else float("nan")


def build_policy_macro_summary_rows(
    metric_df: pd.DataFrame,
    *,
    split: str,
    provenance_loaded: bool,
) -> list[dict[str, Any]]:
    """Append summary rows for main_verified / formula_available / all_reported macros."""
    if metric_df.empty:
        return []

    usable = metric_df[pd.to_numeric(metric_df["r2"], errors="coerce").notna()].copy()
    rows: list[dict[str, Any]] = []

    def _append(kind: str, key: str, sub: pd.DataFrame) -> None:
        if sub.empty:
            return
        r2s = pd.to_numeric(sub["r2"], errors="coerce").dropna()
        macro = float(r2s.mean()) if not r2s.empty else float("nan")
        scores = (
            pd.to_numeric(sub["relative_accuracy_score"], errors="coerce").dropna()
            if "relative_accuracy_score" in sub.columns
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "split": split,
                "process_id": "__ALL__",
                "target_id": key,
                "mae": float(pd.to_numeric(sub["mae"], errors="coerce").mean()),
                "rmse": float(pd.to_numeric(sub["rmse"], errors="coerce").mean()),
                "r2": macro,
                "relative_accuracy_score": float(scores.mean()) if not scores.empty else float("nan"),
                "n_samples": int(pd.to_numeric(sub["n_samples"], errors="coerce").sum()),
                "n_targets_present": int(sub["target_id"].nunique()),
                "summary_kind": kind,
                "provenance_policy_loaded": provenance_loaded,
            }
        )

    if provenance_loaded and "include_in_main_verified_macro" in usable.columns:
        _append("split_macro_main_verified", "__ALL_macro_main_verified__", usable[usable["include_in_main_verified_macro"]])
        _append(
            "split_macro_formula_available",
            "__ALL_macro_formula_available__",
            usable[usable["include_in_formula_macro"]],
        )
        streams = usable[usable["include_in_streams_only_macro"]]
        if not streams.empty:
            _append("split_macro_streams_only", "__ALL_macro_streams_only__", streams)
        excluded = usable[~usable["include_in_main_verified_macro"] & ~usable["include_in_formula_macro"]]
        if not excluded.empty:
            logger.warning(
                "v4 provenance: %d targets excluded from main/formula macro: %s",
                excluded["target_id"].nunique(),
                sorted(excluded["target_id"].astype(str).unique().tolist()),
            )
    else:
        _append("split_macro_all_reported", "__ALL_macro_split__", usable)

    return rows


def payload_from_policy_macros(summary_df: pd.DataFrame, *, split_name: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if summary_df.empty:
        return out

    mapping = {
        "split_macro_main_verified": "target_v4_macro_r2_main_verified",
        "split_macro_formula_available": "target_v4_macro_r2_formula_available",
        "split_macro_streams_only": "target_v4_macro_r2_streams_only",
        "split_macro_all_reported": "target_v4_macro_r2_all_reported",
        "split_macro": "target_v4_macro_r2",
    }

    sp = split_name
    for kind, key in mapping.items():
        sub = summary_df[summary_df.get("summary_kind", "") == kind]
        if sp and "split" in sub.columns:
            sub = sub[sub["split"].astype(str) == sp]
        if sub.empty:
            continue
        val = float(pd.to_numeric(sub.iloc[-1]["r2"], errors="coerce"))
        if not math.isfinite(val):
            continue
        out[f"{sp}/{key}" if sp else key] = val

    # Primary + deprecated alias
    if sp:
        if f"{sp}/target_v4_macro_r2_main_verified" in out:
            out["target_v4_macro_r2_main_verified"] = out[f"{sp}/target_v4_macro_r2_main_verified"]
            out[f"{sp}/target_v4_macro_r2"] = out[f"{sp}/target_v4_macro_r2_main_verified"]
            out["target_v4_macro_r2"] = out["target_v4_macro_r2_main_verified"]
            out[f"test/target_v4_macro_r2"] = out["target_v4_macro_r2_main_verified"]
        elif f"{sp}/target_v4_macro_r2_all_reported" in out:
            out["target_v4_macro_r2"] = out[f"{sp}/target_v4_macro_r2_all_reported"]
            out[f"test/target_v4_macro_r2"] = out[f"{sp}/target_v4_macro_r2_all_reported"]
            logger.warning(
                "v4 provenance policy not loaded; target_v4_macro_r2 uses all_reported (deprecated)."
            )
    return out
