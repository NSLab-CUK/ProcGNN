"""Shared helpers for capacity/method/k-fold ablation target-row metrics."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_RUN_DIR_RE = re.compile(r"^process_kfold_P(\d+)_F(\d+)(?:-\d{8}-\d{6})?$")

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from process_graph.experiment.target_row_primary_metrics import (  # noqa: E402
    LEGACY_ANSWER_FRACTION_METRIC,
    PRIMARY_METRIC_NAME,
    TARGET_BALANCED_METRIC_NAME,
    TARGETROW_METRIC_SCHEMA_VERSION,
)

# Re-export for scripts that import from ablation_targetrow_common only.

PRIMARY_VAL_COL = f"val_{PRIMARY_METRIC_NAME}"
PRIMARY_TEST_COL = f"test_{PRIMARY_METRIC_NAME}"
TARGET_BALANCED_VAL_COL = f"val_{TARGET_BALANCED_METRIC_NAME}"
TARGET_BALANCED_TEST_COL = f"test_{TARGET_BALANCED_METRIC_NAME}"
LEGACY_VAL_COL = f"val_{LEGACY_ANSWER_FRACTION_METRIC}"
LEGACY_TEST_COL = f"test_{LEGACY_ANSWER_FRACTION_METRIC}"

PROCESS_BALANCED_SUMMARY_TARGET_ID = "__ALL_process_balanced_main_verified__"
TARGET_BALANCED_SUMMARY_TARGET_ID = "__ALL_target_balanced_main_verified__"


def safe_read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def parse_run_dir(path: Path) -> tuple[int, int] | None:
    m = _RUN_DIR_RE.match(path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def discover_kfold_runs(root: Path) -> list[dict]:
    rows: list[dict] = []
    if not root.is_dir():
        return rows
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == "summary":
            continue
        exp_name = exp_dir.name
        for proc_dir in sorted(exp_dir.glob("Process*")):
            if not proc_dir.is_dir():
                continue
            process_id = proc_dir.name
            for fold_dir in sorted(proc_dir.glob("fold_*")):
                if not fold_dir.is_dir():
                    continue
                fold_m = re.match(r"fold_(\d+)", fold_dir.name)
                fold_id = int(fold_m.group(1)) if fold_m else 0
                for child in sorted(fold_dir.iterdir()):
                    if not child.is_dir():
                        continue
                    parsed = parse_run_dir(child)
                    if not parsed:
                        continue
                    _, fold_from_name = parsed
                    rows.append(
                        {
                            "exp_name": exp_name,
                            "experiment_name": exp_name,
                            "process_id": process_id,
                            "fold": fold_from_name,
                            "fold_id": fold_from_name,
                            "fold_dir": fold_dir,
                            "run_dir": child,
                        }
                    )
    return rows


def read_target_v4_table(run_dir: Path) -> pd.DataFrame | None:
    """Alias used by audit script."""
    for rel in ("test/target_metrics_v4.csv", "target_metrics_v4.csv"):
        p = run_dir / rel
        if p.is_file():
            try:
                return pd.read_csv(p)
            except (OSError, pd.errors.EmptyDataError):
                return None
    return None


def _read_summary_bundle_row(run_dir: Path) -> dict[str, Any]:
    summ = read_target_v4_summary(run_dir)
    if summ is None or summ.empty:
        return {}
    if "summary_kind" in summ.columns:
        bundle = summ[summ["summary_kind"].astype(str) == "metric_bundle_metadata"]
        if not bundle.empty:
            return bundle.iloc[0].to_dict()
    bundle = summ[summ["target_id"].astype(str) == "__METRIC_BUNDLE__"]
    if not bundle.empty:
        return bundle.iloc[0].to_dict()
    return {}


def read_target_v4_summary(run_dir: Path) -> pd.DataFrame | None:
    for rel in ("test/target_metrics_v4_summary.csv", "target_metrics_v4_summary.csv"):
        p = run_dir / rel
        if p.is_file():
            try:
                return pd.read_csv(p)
            except (OSError, pd.errors.EmptyDataError):
                return None
    return None


def _pick_summary_r2(summ: pd.DataFrame, target_id: str, summary_kind: str = "") -> float:
    if summ is None or summ.empty:
        return float("nan")
    macro = summ[summ["target_id"].astype(str) == target_id] if "target_id" in summ.columns else summ.iloc[0:0]
    if macro.empty and summary_kind and "summary_kind" in summ.columns:
        macro = summ[summ["summary_kind"].astype(str) == summary_kind]
    if macro.empty or "r2" not in macro.columns:
        return float("nan")
    v = pd.to_numeric(macro.iloc[0]["r2"], errors="coerce")
    return float(v) if pd.notna(v) else float("nan")


def pick_test_metrics(run_dir: Path) -> dict[str, Any]:
    tj = safe_read_json(run_dir / "test" / "metrics.json")
    if isinstance(tj, dict) and tj:
        return tj
    mj = safe_read_json(run_dir / "metrics.json") or {}
    return {k[5:]: v for k, v in mj.items() if str(k).startswith("test/")}


def read_run_schema_version(run_dir: Path) -> str:
    for path in (run_dir / "metrics.json", run_dir / "test" / "metrics.json"):
        mj = safe_read_json(path)
        if not isinstance(mj, dict):
            continue
        for key in (
            "targetrow_metric_schema_version",
            "metric_schema_version",
            "test/targetrow_metric_schema_version",
            "val/targetrow_metric_schema_version",
        ):
            if key in mj and mj[key]:
                return str(mj[key]).strip()
    ep = run_dir / "metrics_per_epoch.csv"
    if ep.is_file():
        try:
            df = pd.read_csv(ep, nrows=1)
            if "val_targetrow_metric_schema_version" in df.columns:
                v = str(df["val_targetrow_metric_schema_version"].iloc[0]).strip()
                if v and v != "nan":
                    return v
        except (OSError, pd.errors.EmptyDataError):
            pass
    return ""


def read_primary_metric_name(run_dir: Path) -> str:
    for path in (run_dir / "metrics.json", run_dir / "test" / "metrics.json"):
        mj = safe_read_json(path)
        if isinstance(mj, dict) and mj.get("primary_metric_name"):
            return str(mj["primary_metric_name"]).strip()
    return ""


def validate_run_targetrow_complete(run_dir: Path) -> dict[str, Any]:
    """Return whether run_dir has a complete target-row metric bundle (for skip-existing)."""
    reasons: list[str] = []
    v4_csv = run_dir / "test" / "target_metrics_v4.csv"
    if not v4_csv.is_file():
        v4_csv = run_dir / "target_metrics_v4.csv"
    v4_sum = run_dir / "test" / "target_metrics_v4_summary.csv"
    if not v4_sum.is_file():
        v4_sum = run_dir / "target_metrics_v4_summary.csv"
    frac_csv = run_dir / "test" / "target_metrics_v4_frac.csv"
    if not frac_csv.is_file():
        frac_csv = run_dir / "target_metrics_v4_frac.csv"
    frac_sum = run_dir / "test" / "target_metrics_v4_frac_summary.csv"
    if not frac_sum.is_file():
        frac_sum = run_dir / "target_metrics_v4_frac_summary.csv"

    has_v4 = v4_csv.is_file()
    has_sum = v4_sum.is_file()
    has_frac = frac_csv.is_file()
    has_frac_sum = frac_sum.is_file()
    if not has_v4:
        reasons.append("missing_target_metrics_v4_csv")
    if not has_sum:
        reasons.append("missing_target_metrics_v4_summary_csv")
    if not has_frac:
        reasons.append("missing_target_metrics_v4_frac_csv")
    if not has_frac_sum:
        reasons.append("missing_target_metrics_v4_frac_summary_csv")

    schema = read_run_schema_version(run_dir)
    from process_graph.experiment.target_metric_names import METRIC_SCHEMA, resolve_metric_key

    schema_ok = schema in (
        TARGETROW_METRIC_SCHEMA_VERSION,
        METRIC_SCHEMA,
        resolve_metric_key(schema or ""),
    )
    if not schema_ok:
        reasons.append(f"schema_mismatch:{schema or 'empty'}")

    pname = read_primary_metric_name(run_dir)
    if pname and pname != PRIMARY_METRIC_NAME:
        reasons.append(f"primary_metric_name_mismatch:{pname}")
    elif not pname:
        reasons.append("missing_primary_metric_name")

    meta = extract_primary_metrics(run_dir)
    test_primary = meta.get(PRIMARY_TEST_COL, float("nan"))
    if not math.isfinite(float(test_primary)):
        reasons.append("missing_finite_test_primary_metric")

    inc = str(meta.get("included_target_ids", ""))
    exc = str(meta.get("excluded_target_ids", ""))
    if "included_target_ids" not in meta:
        reasons.append("missing_included_target_ids_key")
    if "excluded_target_ids" not in meta:
        reasons.append("missing_excluded_target_ids_key")

    if has_v4 and not inc and meta.get("n_targets_main_verified", 0) == 0 and meta.get("n_targets_total", 0) > 0:
        reasons.append("empty_included_target_ids")

    ok = len(reasons) == 0
    return {
        "ok": ok,
        "reasons": ";".join(reasons),
        "targetrow_metric_schema_version": schema,
        "primary_metric_name": pname or PRIMARY_METRIC_NAME,
        "has_target_metrics_v4": has_v4,
        "has_target_metrics_v4_summary": has_sum,
        "has_target_metrics_v4_frac": has_frac,
        "has_target_metrics_v4_frac_summary": has_frac_sum,
        "included_target_ids": inc,
        "excluded_target_ids": exc,
        PRIMARY_TEST_COL: test_primary,
    }


def preflight_recompute(run_dir: Path, base_cfg: Path | None = None) -> dict[str, Any]:
    """Check files required for eval-only metric recompute."""
    missing: list[str] = []
    ckpt = None
    for rel in ("checkpoints/best.pt", "checkpoints/best_model.pt", "best.pt"):
        p = run_dir / rel
        if p.is_file():
            ckpt = p
            break
    if ckpt is None:
        ckpt_dir = run_dir / "checkpoints"
        if ckpt_dir.is_dir():
            pts = sorted(ckpt_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
            ckpt = pts[0] if pts else None
    if ckpt is None:
        missing.append("checkpoints/best.pt")

    overrides = run_dir.parent / "runtime_overrides.json"
    if not overrides.is_file():
        missing.append("runtime_overrides.json")

    cfg = base_cfg
    if cfg is None or not cfg.is_file():
        missing.append("experiment_config_yaml")

    if overrides.is_file():
        try:
            data = json.loads(overrides.read_text(encoding="utf-8"))
            data_block = data.get("data", {})
            for key in (
                "train_split_manifest_path",
                "val_split_manifest_path",
                "test_split_manifest_path",
                "train_data_path",
            ):
                if key not in data_block and key not in data:
                    missing.append(f"runtime_overrides_missing_{key}")
                else:
                    rel = data_block.get(key, data.get(key))
                    if rel:
                        p = PROJECT_ROOT / str(rel) if not Path(str(rel)).is_absolute() else Path(str(rel))
                        if key.endswith("_manifest_path") and not p.is_file():
                            missing.append(f"split_manifest_missing:{rel}")
        except (json.JSONDecodeError, OSError):
            missing.append("runtime_overrides_unreadable")

    if not (run_dir / "metrics.json").is_file() and cfg is None:
        missing.append("metrics.json_or_config")

    ok = len(missing) == 0
    return {
        "ok": ok,
        "failed_recompute_reason": ";".join(missing) if missing else "",
        "checkpoint": str(ckpt) if ckpt else "",
        "runtime_overrides": str(overrides) if overrides.is_file() else "",
        "config": str(cfg) if cfg and cfg.is_file() else "",
    }


def extract_primary_metrics(run_dir: Path) -> dict[str, Any]:
    """Read primary / auxiliary / legacy metrics from artifacts."""
    tm = pick_test_metrics(run_dir)
    mj = safe_read_json(run_dir / "metrics.json") or {}
    summ = read_target_v4_summary(run_dir)
    prov = provenance_metadata(run_dir)

    def _get(*keys: str) -> float:
        for k in keys:
            if k in tm:
                try:
                    v = float(tm[k])
                    if math.isfinite(v):
                        return v
                except (TypeError, ValueError):
                    pass
            if k in mj:
                try:
                    v = float(mj[f"test/{k}"] if f"test/{k}" in mj else mj[k])
                    if math.isfinite(v):
                        return v
                except (TypeError, ValueError):
                    pass
        return float("nan")

    val_primary = _get(PRIMARY_METRIC_NAME, f"val/{PRIMARY_METRIC_NAME}")
    test_primary = _get(PRIMARY_METRIC_NAME, f"test/{PRIMARY_METRIC_NAME}")
    val_tb = _get(TARGET_BALANCED_METRIC_NAME, f"val/{TARGET_BALANCED_METRIC_NAME}")
    test_tb = _get(TARGET_BALANCED_METRIC_NAME, f"test/{TARGET_BALANCED_METRIC_NAME}")
    val_legacy = _get(
        LEGACY_ANSWER_FRACTION_METRIC,
        "legacy_answer_fraction_macro_r2",
        "metric_answer_all_targets_mean_r2",
    )
    test_legacy = val_legacy if math.isfinite(val_legacy) else _get(
        LEGACY_ANSWER_FRACTION_METRIC,
        "legacy_answer_fraction_macro_r2",
    )

    if summ is not None and not summ.empty:
        if math.isnan(test_primary):
            test_primary = _pick_summary_r2(
                summ, PROCESS_BALANCED_SUMMARY_TARGET_ID, "process_balanced_macro_main_verified"
            )
        if math.isnan(test_tb):
            test_tb = _pick_summary_r2(
                summ, TARGET_BALANCED_SUMMARY_TARGET_ID, "target_balanced_macro_main_verified"
            )

    if math.isnan(val_primary) or math.isnan(test_primary):
        sys_path = str(PROJECT_ROOT / "src")
        if sys_path not in __import__("sys").path:
            __import__("sys").path.insert(0, sys_path)
        from process_graph.experiment.target_row_primary_metrics import pick_primary_r2_from_summary_csv

        if summ is not None and not summ.empty:
            if math.isnan(test_primary):
                test_primary = pick_primary_r2_from_summary_csv(summ, balanced="process")
            if math.isnan(test_tb):
                test_tb = pick_primary_r2_from_summary_csv(summ, balanced="target")

    out: dict[str, Any] = {
        "official_metric_name": PRIMARY_METRIC_NAME,
        "official_metric_value": test_primary,
        PRIMARY_METRIC_NAME: test_primary,
        "eval_primary_frac_r2_by_target": test_tb,
        "eval_secondary_amount_r2_by_process": _get(
            "eval_secondary_amount_r2_by_process", f"test/eval_secondary_amount_r2_by_process"
        ),
        "eval_legacy_answer_edge_frac_r2": test_legacy,
        "primary_metric_name": PRIMARY_METRIC_NAME,
        "targetrow_metric_schema_version": TARGETROW_METRIC_SCHEMA_VERSION,
        "metric_schema_version": TARGETROW_METRIC_SCHEMA_VERSION,
        PRIMARY_VAL_COL: val_primary,
        PRIMARY_TEST_COL: test_primary,
        TARGET_BALANCED_VAL_COL: val_tb,
        TARGET_BALANCED_TEST_COL: test_tb,
        LEGACY_VAL_COL: val_legacy,
        LEGACY_TEST_COL: test_legacy,
        **prov,
    }
    bundle = _read_summary_bundle_row(run_dir)
    if bundle:
        for k, v in bundle.items():
            if k in out and isinstance(out.get(k), float) and math.isfinite(float(out[k])):
                continue
            if v is not None and str(v) != "nan":
                out[k] = v
    return out


def provenance_metadata(run_dir: Path) -> dict[str, Any]:
    df = read_target_v4_table(run_dir)
    if df is None or df.empty or "target_id" not in df.columns:
        return {
            "n_targets_total": 0,
            "n_targets_main_verified": 0,
            "n_targets_formula_available": 0,
            "n_targets_excluded": 0,
            "included_target_ids": "",
            "excluded_target_ids": "",
            "excluded_target_reasons": "",
            "process_target_counts": "",
            "target_count_by_species": "",
        }
    sub = df[~df["target_id"].astype(str).str.startswith("__ALL")]
    out: dict[str, Any] = {"n_targets_total": int(sub["target_id"].nunique())}
    if "include_in_main_verified_macro" in sub.columns:
        ok = sub[sub["include_in_main_verified_macro"].astype(str).str.lower().isin(("true", "1", "yes"))]
        excl = sub[~sub["target_id"].isin(ok["target_id"])]
        out["n_targets_main_verified"] = int(ok["target_id"].nunique())
        out["n_targets_excluded"] = int(excl["target_id"].nunique())
        out["included_target_ids"] = ";".join(sorted(ok["target_id"].astype(str).unique()))
        out["excluded_target_ids"] = ";".join(sorted(excl["target_id"].astype(str).unique()))
        if "exclusion_reason" in excl.columns:
            out["excluded_target_reasons"] = ";".join(
                f"{r['target_id']}:{r['exclusion_reason']}"
                for _, r in excl.iterrows()
                if pd.notna(r.get("exclusion_reason"))
            )
    if "include_in_formula_macro" in sub.columns:
        form = sub[sub["include_in_formula_macro"].astype(str).str.lower().isin(("true", "1", "yes"))]
        out["n_targets_formula_available"] = int(form["target_id"].nunique())
    if "process_id" in sub.columns:
        counts = sub.groupby("process_id")["target_id"].nunique().to_dict()
        out["process_target_counts"] = ";".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda x: str(x[0])))
    if "target_species" in sub.columns:
        sc = sub.groupby("target_species")["target_id"].nunique().to_dict()
        out["target_count_by_species"] = ";".join(f"{k}:{v}" for k, v in sorted(sc.items(), key=lambda x: str(x[0])))
    return out


def best_epoch_by_primary(run_dir: Path) -> tuple[float, float]:
    csv_path = run_dir / "metrics_per_epoch.csv"
    if not csv_path.is_file():
        return float("nan"), float("nan")
    try:
        df = pd.read_csv(csv_path)
    except (OSError, pd.errors.EmptyDataError):
        return float("nan"), float("nan")
    col = None
    for c in (
        PRIMARY_VAL_COL,
        f"val_{PRIMARY_METRIC_NAME}",
        "val_process_balanced_main_target_r2_main_verified",
        "val_target_v4_macro_r2_main_verified",
        LEGACY_VAL_COL,
        "val_legacy_answer_fraction_macro_r2",
        "val_answer_targets_r2_weighted",
    ):
        if c in df.columns:
            col = c
            break
    if not col:
        return float("nan"), float("nan")
    vals = pd.to_numeric(df[col], errors="coerce")
    if not vals.notna().any():
        return float("nan"), float("nan")
    idx = int(vals.idxmax())
    return float(df.loc[idx, "epoch"]), float(vals.loc[idx])
