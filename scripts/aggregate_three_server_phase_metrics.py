from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROPERTIES = [
    "Temp", "Pres", "Frac_H2O", "Frac_H2", "Frac_CH4",
    "Frac_CO2", "Frac_CO", "Frac_O2", "Frac_N2", "Mass_Flow",
]


@dataclass(frozen=True)
class Server:
    name: str
    root: Path

    @property
    def output(self) -> Path:
        return self.root / "outputs" / "0819final"


def _num(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV with short retries for transient SFTP/RaiDrive handles."""
    last_error: OSError | None = None
    for attempt in range(4):
        try:
            return pd.read_csv(path)
        except OSError as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(0.25 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _latest_by_key(files: list[Path], key_fn) -> list[tuple[object, Path]]:
    chosen: dict[object, Path] = {}
    for path in files:
        key = key_fn(path)
        if key is None:
            continue
        old = chosen.get(key)
        if old is None or path.stat().st_mtime > old.stat().st_mtime:
            chosen[key] = path
    return sorted(chosen.items(), key=lambda item: str(item[0]))


def _combine_target_rows(frame: pd.DataFrame, proposed_policy: bool) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c) for c in frame.columns]
    prop_col = "property_name"
    n_col = "n_samples" if "n_samples" in frame else ("R2_n_valid" if "R2_n_valid" in frame else "n")
    mae_col = "MAE" if "MAE" in frame else "mae"
    sst_col = "SST" if "SST" in frame else ("R2_SST" if "R2_SST" in frame else "sst")
    sse_col = "SSE" if "SSE" in frame else ("R2_SSE" if "R2_SSE" in frame else "sse")
    mean_col = "true_mean" if "true_mean" in frame else "orig_true_mean"
    std_col = "true_std" if "true_std" in frame else "orig_true_std"
    rows: list[dict[str, object]] = []
    for prop in PROPERTIES:
        g = frame[frame[prop_col].astype(str) == prop].copy()
        if g.empty:
            continue
        n = pd.to_numeric(g[n_col], errors="coerce").fillna(0.0).to_numpy(float)
        mae = pd.to_numeric(g[mae_col], errors="coerce").to_numpy(float)
        means = pd.to_numeric(g[mean_col], errors="coerce").to_numpy(float)
        stds = pd.to_numeric(g[std_col], errors="coerce").to_numpy(float)
        sst = pd.to_numeric(g[sst_col], errors="coerce").to_numpy(float)
        sse = pd.to_numeric(g[sse_col], errors="coerce").to_numpy(float)
        valid_n = np.where(np.isfinite(n) & (n > 0), n, 0.0)
        total_n = float(valid_n.sum())
        pooled_mean = float(np.nansum(valid_n * means) / total_n) if total_n else math.nan
        # Prefer reported SST/SSE. Add between-edge variation because each row is an edge-local bundle.
        within_sst = np.where(np.isfinite(sst), sst, valid_n * np.square(stds))
        total_sst = float(np.nansum(within_sst + valid_n * np.square(means - pooled_mean))) if total_n else math.nan
        total_sse = float(np.nansum(sse)) if np.isfinite(sse).any() else math.nan
        pooled_std = math.sqrt(max(total_sst / total_n, 0.0)) if total_n and math.isfinite(total_sst) else math.nan
        pooled_mae = float(np.nansum(valid_n * mae) / total_n) if total_n else math.nan
        if total_n < 2 or not math.isfinite(total_sst):
            r2 = math.nan
            r2_status = "insufficient"
        elif total_sst <= 1e-12:
            r2 = 0.999 if proposed_policy else 0.0
            r2_status = "constant_target_proposed_0.999" if proposed_policy else "constant_target_baseline_0.0"
        else:
            r2 = 1.0 - total_sse / total_sst
            r2_status = "ok"
        if math.isfinite(pooled_std) and pooled_std > 1e-12:
            smae = pooled_mae / pooled_std
            smae_status = "ok"
        elif math.isfinite(pooled_mae) and abs(pooled_mae) <= 1e-12:
            smae = 0.0
            smae_status = "constant_exact"
        else:
            smae = math.nan
            smae_status = "constant_nonzero_error"
        rows.append({
            "property": prop, "r2": r2, "mae": pooled_mae, "smae": smae,
            "n": total_n, "true_mean": pooled_mean, "true_std": pooled_std,
            "sst": total_sst, "sse": total_sse,
            "r2_status": r2_status, "smae_status": smae_status,
        })
    return pd.DataFrame(rows)


def _proposed_file_metrics(path: Path) -> pd.DataFrame:
    frame = _read_csv(path)
    if "split" in frame:
        split = frame["split"].astype(str).str.lower()
        if (split == "test").any():
            frame = frame[split == "test"]
        elif (split == "val").any():
            frame = frame[split == "val"]
    return _combine_target_rows(frame, proposed_policy=True)


def _baseline_files_metrics(paths: list[Path]) -> pd.DataFrame:
    frames = [_read_csv(path) for path in paths]
    return _combine_target_rows(pd.concat(frames, ignore_index=True), proposed_policy=False)


def _add_run(
    out: list[pd.DataFrame], metrics: pd.DataFrame, *, phase: str, family: str,
    model: str, condition: str, process: str, fold: int, server: str,
    source: str, evaluation_split: str,
) -> None:
    if metrics.empty:
        return
    metrics = metrics.copy()
    metrics.insert(0, "phase", phase)
    metrics.insert(1, "family", family)
    metrics.insert(2, "model", model)
    metrics.insert(3, "condition", condition)
    metrics.insert(4, "process", process)
    metrics.insert(5, "fold", fold)
    metrics.insert(6, "server", server)
    metrics.insert(7, "evaluation_split", evaluation_split)
    metrics.insert(8, "source_file", source)
    metrics.insert(9, "reused_result", False)
    out.append(metrics)


def _collect_single_baselines(servers: list[Server], out: list[pd.DataFrame]) -> None:
    candidates: list[tuple[Server, Path]] = []
    for server in servers:
        # Baselines were written to three result roots over the course of the
        # study.  The later transformer/topology roots contain the additional
        # Graphormer, SAT, and GraphToSFILES runs (and replacement G1--G3).
        # Deduplication below keeps the newest artifact for the same logical
        # model/process/fold run.
        for root_name in ("baselines", "baselines_transformer_0908", "baselines_topology_sfiles_0908"):
            candidates += [
                (server, p)
                for p in (server.output / root_name / "single").glob(
                    "*/Process*/fold_*/test_target_edge_property_metrics.csv"
                )
            ]
    def key(item):
        server, path = item
        m = re.search(r"single[\\/]([^\\/]+)[\\/]Process(\d+)[\\/]fold_(\d+)", str(path))
        return (m.group(1), int(m.group(2)), int(m.group(3))) if m else None
    chosen: dict[object, tuple[Server, Path]] = {}
    for item in candidates:
        k = key(item)
        if k is not None and (k not in chosen or item[1].stat().st_mtime > chosen[k][1].stat().st_mtime):
            chosen[k] = item
    for (model, process, fold), (server, path) in sorted(chosen.items()):
        _add_run(out, _baseline_files_metrics([path]), phase="Phase 1", family="single_process_baseline",
                 model=model, condition="default", process=f"P{process:02d}", fold=fold,
                 server=server.name, source=str(path), evaluation_split="test")


def _collect_multi_baselines(servers: list[Server], out: list[pd.DataFrame]) -> None:
    grouped: dict[tuple[str, int], list[tuple[Server, Path]]] = {}
    for server in servers:
        for root_name in ("baselines", "baselines_transformer_0908", "baselines_topology_sfiles_0908"):
            for path in (server.output / root_name / "multi").glob("*/fold_*/Process*/test_target_edge_property_metrics.csv"):
                m = re.search(r"multi[\\/]([^\\/]+)[\\/]fold_(\d+)[\\/]Process(\d+)", str(path))
                if m:
                    grouped.setdefault((m.group(1), int(m.group(2))), []).append((server, path))
    for (model, fold), items in sorted(grouped.items()):
        # One file per process; if duplicated across servers keep the newest.
        by_process: dict[int, tuple[Server, Path]] = {}
        for server, path in items:
            p = int(re.search(r"Process(\d+)", str(path)).group(1))
            if p not in by_process or path.stat().st_mtime > by_process[p][1].stat().st_mtime:
                by_process[p] = (server, path)
        paths = [v[1] for _, v in sorted(by_process.items())]
        names = "+".join(sorted({v[0].name for v in by_process.values()}))
        _add_run(out, _baseline_files_metrics(paths), phase="Phase 1", family="multi_process_comparison",
                 model=model, condition="default", process="All", fold=fold, server=names,
                 source=";".join(map(str, paths)), evaluation_split="test")


def _collect_transfer_baselines(servers: list[Server], out: list[pd.DataFrame]) -> None:
    """Collect source-pretrained, target-ratio-finetuned graph baselines.

    These artifacts are deliberately a separate collector: their directory
    structure is heldout-process/fold/ratio rather than the Phase-1 joint
    baseline layout, and only the target test set is a reportable endpoint.
    """
    candidates: list[tuple[Server, Path]] = []
    for server in servers:
        root = server.output / "baselines_transfer_0909"
        if root.exists():
            candidates += [(server, path) for path in root.glob(
                "*/heldout_P*/fold_*/ratio_*/test_target_edge_property_metrics.csv"
            )]
    chosen: dict[tuple[str, int, int, int], tuple[Server, Path]] = {}
    for server, path in candidates:
        match = re.search(
            r"baselines_transfer_0909[\\/]([^\\/]+)[\\/]heldout_P(\d+)[\\/]fold_(\d+)[\\/]ratio_(\d+)",
            str(path),
        )
        if not match:
            continue
        key = (match.group(1), int(match.group(2)), int(match.group(3)), int(match.group(4)))
        if key not in chosen or path.stat().st_mtime > chosen[key][1].stat().st_mtime:
            chosen[key] = (server, path)
    for (model, process, fold, ratio), (server, path) in sorted(chosen.items()):
        _add_run(
            out, _baseline_files_metrics([path]), phase="Phase 3", family="baseline_transfer_data_efficiency",
            model=model, condition=f"ratio_{ratio:03d}", process=f"P{process:02d}", fold=fold,
            server=server.name, source=str(path), evaluation_split="test",
        )


def _collect_zero_shot_baselines(servers: list[Server], out: list[pd.DataFrame]) -> None:
    """Collect frozen source-only baseline evaluations on held-out processes."""
    candidates: list[tuple[Server, Path]] = []
    for server in servers:
        root = server.output / "baselines_zero_shot_0910"
        if root.exists():
            candidates += [(server, path) for path in root.glob(
                "*/heldout_P*/fold_*/test_target_edge_property_metrics.csv"
            )]
    chosen: dict[tuple[str, int, int], tuple[Server, Path]] = {}
    for server, path in candidates:
        match = re.search(
            r"baselines_zero_shot_0910[\\/]([^\\/]+)[\\/]heldout_P(\d+)[\\/]fold_(\d+)",
            str(path),
        )
        if not match:
            continue
        key = (match.group(1), int(match.group(2)), int(match.group(3)))
        if key not in chosen or path.stat().st_mtime > chosen[key][1].stat().st_mtime:
            chosen[key] = (server, path)
    for (model, process, fold), (server, path) in sorted(chosen.items()):
        _add_run(
            out, _baseline_files_metrics([path]), phase="Phase 1", family="baseline_zero_shot",
            model=model, condition="zero_shot", process=f"P{process:02d}", fold=fold,
            server=server.name, source=str(path), evaluation_split="test",
        )


def _collect_proposed(servers: list[Server], out: list[pd.DataFrame]) -> None:
    specs = [
        ("Phase 1-S", "proposed_single_process", "Proposed", "proposed_single_process", r"Process(\d+)[\\/]fold_(\d+)", "process"),
        ("Phase 1", "multi_process_comparison", "Proposed", "proposed_joint_10d_clean", r"All[\\/]fold_(\d+)", "all"),
        ("Phase 1", "proposed_zero_shot", "Proposed", "proposed_unseen", r"heldout_P(\d+)[\\/]fold_(\d+)[\\/]zero_shot", "process"),
        ("Phase 3", "data_efficiency", "Proposed-transfer", "data_efficiency", r"heldout_P(\d+)[\\/]fold_(\d+)[\\/]transfer[\\/]ratio_(\d+)", "ratio"),
    ]
    for phase, family, model, dirname, pattern, kind in specs:
        candidates: list[tuple[Server, Path, re.Match[str]]] = []
        for server in servers:
            for path in (server.output / dirname).rglob("target_edge_10d_metrics.csv") if (server.output / dirname).exists() else []:
                m = re.search(pattern, str(path))
                if m:
                    candidates.append((server, path, m))
        chosen: dict[tuple, tuple[Server, Path, re.Match[str]]] = {}
        for item in candidates:
            server, path, m = item
            if kind == "all":
                key = (int(m.group(1)),)
            elif kind == "process":
                key = (int(m.group(1)), int(m.group(2)))
            else:
                key = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if key not in chosen or path.stat().st_mtime > chosen[key][1].stat().st_mtime:
                chosen[key] = item
        for key, (server, path, m) in sorted(chosen.items()):
            if kind == "all":
                process, fold, condition = "All", key[0], "default"
            elif kind == "process":
                process, fold, condition = f"P{key[0]:02d}", key[1], "default"
            else:
                process, fold, condition = f"P{key[0]:02d}", key[1], f"ratio_{key[2]:03d}"
            _add_run(out, _proposed_file_metrics(path), phase=phase, family=family, model=model,
                     condition=condition, process=process, fold=fold, server=server.name,
                     source=str(path), evaluation_split="final_checkpoint_eval_reported_as_val")


def _collect_sensitivity(servers: list[Server], out: list[pd.DataFrame]) -> None:
    candidates: list[tuple[Server, Path, str, str, int]] = []
    for server in servers:
        root = server.output / "sensitivity_10d_clean"
        if not root.exists():
            continue
        for path in root.rglob("target_edge_10d_metrics.csv"):
            m = re.search(r"sensitivity_10d_clean[\\/](depth|pin)[\\/]([^\\/]+)[\\/]All[\\/]fold_(\d+)", str(path))
            if m:
                candidates.append((server, path, m.group(1), m.group(2), int(m.group(3))))
    chosen: dict[tuple[str, str, int], tuple[Server, Path]] = {}
    for server, path, group, condition, fold in candidates:
        key = (group, condition, fold)
        if key not in chosen or path.stat().st_mtime > chosen[key][1].stat().st_mtime:
            chosen[key] = (server, path)
    for (group, condition, fold), (server, path) in sorted(chosen.items()):
        _add_run(out, _proposed_file_metrics(path), phase="Phase 2",
                 family=f"sensitivity_{group}", model="Proposed", condition=condition,
                 process="All", fold=fold, server=server.name, source=str(path),
                 evaluation_split="final_checkpoint_eval_reported_as_val")


def _collect_physics_conservation(servers: list[Server]) -> pd.DataFrame:
    """Collect already-evaluated conservation artifacts without re-running models.

    The physics evaluator writes one CSV per logical run with three rows
    (mass, component, atom).  Only the canonical Proposed single-process and
    joint checkpoints have this evaluator output; absent rows are intentionally
    left absent so the paper table can display ``NA (not evaluated)`` rather
    than borrowing a value from another model.
    """
    candidates: list[tuple[Server, Path, str, int, int]] = []
    for server in servers:
        root = server.output / "physics_conservation"
        if not root.exists():
            continue
        for path in root.rglob("physics_conservation_metrics.csv"):
            text = str(path)
            single = re.search(r"proposed_single[\\/]Process(\d+)[\\/]fold_(\d+)", text)
            joint = re.search(r"proposed_joint[\\/]All[\\/]fold_(\d+)", text)
            if single:
                candidates.append((server, path, "proposed_single_process", int(single.group(1)), int(single.group(2))))
            elif joint:
                candidates.append((server, path, "multi_process_comparison", 0, int(joint.group(1))))

    chosen: dict[tuple[str, int, int], tuple[Server, Path]] = {}
    for server, path, family, process, fold in candidates:
        key = (family, process, fold)
        if key not in chosen or path.stat().st_mtime > chosen[key][1].stat().st_mtime:
            chosen[key] = (server, path)

    pieces: list[pd.DataFrame] = []
    for (family, process, fold), (server, path) in sorted(chosen.items()):
        frame = _read_csv(path)
        required = {"term", "satisfaction_rate_abs_le_0.05"}
        if not required.issubset(frame.columns):
            continue
        frame = frame[frame["term"].astype(str).isin(["mass", "component", "atom"])].copy()
        if frame.empty:
            continue
        frame.insert(0, "phase", "Phase 1-S" if family == "proposed_single_process" else "Phase 1")
        frame.insert(1, "family", family)
        # Evaluator CSVs already carry a human-readable model column; replace
        # it with the canonical comparison label rather than inserting a
        # duplicate column.
        frame["model"] = "Proposed"
        frame.insert(3, "condition", "default")
        frame.insert(4, "process", f"P{process:02d}" if process else "All")
        frame.insert(5, "fold", fold)
        frame.insert(6, "server", server.name)
        frame.insert(7, "source_file", str(path))
        pieces.append(frame)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["phase", "family", "process", "fold", "term"]
    )


def _summarize_physics_conservation(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame(columns=[
            "phase", "family", "model", "condition", "term", "logical_runs",
            "satisfaction_rate_abs_le_0.05_mean", "satisfaction_rate_abs_le_0.05_std",
        ])
    keys = ["phase", "family", "model", "condition", "term"]
    return detail.groupby(keys, dropna=False, observed=True).agg(**{
        "logical_runs": ("fold", "size"),
        "satisfaction_rate_abs_le_0.05_mean": ("satisfaction_rate_abs_le_0.05", "mean"),
        "satisfaction_rate_abs_le_0.05_std": ("satisfaction_rate_abs_le_0.05", "std"),
    }).reset_index()


def _macro_rows(detail: pd.DataFrame) -> pd.DataFrame:
    keys = ["phase", "family", "model", "condition", "process", "fold", "server", "evaluation_split", "source_file", "reused_result"]
    rows = []
    for values, g in detail.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, values))
        row.update({
            "property_count": int(g["property"].nunique()),
            "r2_property_macro": float(pd.to_numeric(g["r2"], errors="coerce").mean()),
            "mae_property_macro_raw_units": float(pd.to_numeric(g["mae"], errors="coerce").mean()),
            "smae_property_macro": float(pd.to_numeric(g["smae"], errors="coerce").mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _summaries(run: pd.DataFrame, detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_keys = ["phase", "family", "model", "condition"]
    summary = run.groupby(group_keys, dropna=False).agg(
        logical_runs=("fold", "size"),
        processes=("process", "nunique"),
        folds=("fold", "nunique"),
        r2_mean=("r2_property_macro", "mean"),
        r2_median=("r2_property_macro", "median"),
        r2_std=("r2_property_macro", "std"),
        mae_mean_raw_units=("mae_property_macro_raw_units", "mean"),
        mae_median_raw_units=("mae_property_macro_raw_units", "median"),
        mae_std_raw_units=("mae_property_macro_raw_units", "std"),
        smae_mean=("smae_property_macro", "mean"),
        smae_median=("smae_property_macro", "median"),
        smae_std=("smae_property_macro", "std"),
    ).reset_index()
    prop = detail.groupby(group_keys + ["property"], dropna=False, observed=True).agg(
        logical_runs=("fold", "size"), r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        smae_mean=("smae", "mean"), smae_std=("smae", "std"),
    ).reset_index()
    return summary, prop


def _expected_status(run: pd.DataFrame) -> pd.DataFrame:
    expected = {
        "single_process_baseline": 650,
        "multi_process_comparison": 25,  # four baselines plus Proposed, five folds each
        "proposed_zero_shot": 50,
        "baseline_zero_shot": 300,  # 6 graph baselines x 10 held-outs x 5 folds
        "proposed_single_process": 50,
        "data_efficiency": 650,
        "baseline_transfer_data_efficiency": 3900,  # 6 graph baselines x 10 held-outs x 5 folds x 13 ratios
        "sensitivity_depth": 30,  # depth 5 reuses five joint runs
        "sensitivity_pin": 30,    # three default settings reuse five joint runs each
    }
    rows = []
    for family, count in expected.items():
        actual = int(((run["family"] == family) & ~run["reused_result"].astype(bool)).sum())
        rows.append({"family": family, "expected_new_or_direct_runs": count, "found": actual,
                     "missing": max(0, count - actual), "note": "reuse excluded from expected" if family.startswith("sensitivity") else ""})
    # Explainability has no predictive R2/MAE/sMAE.
    shap = 0
    for server_path in [Path(r"T:\home\nsuser\화학공정0610loss아카이브"), Path(r"Z:\home\jw\화학공정final"), Path(r"Y:\home\oj\화학공정final")]:
        root = server_path / "outputs" / "0819final" / "explainability_shap"
        shap += sum(1 for p in root.glob("P??") if (p / "sample_level_shap.csv").is_file()) if root.exists() else 0
    rows.append({"family": "explainability_shap", "expected_new_or_direct_runs": 10, "found": shap,
                 "missing": max(0, 10 - shap), "note": "no predictive R2/MAE/sMAE; explainability artifact only"})
    return pd.DataFrame(rows)


def _add_sensitivity_reuse(detail: pd.DataFrame) -> pd.DataFrame:
    base = detail[
        (detail["phase"] == "Phase 1")
        & (detail["family"] == "multi_process_comparison")
        & (detail["model"] == "Proposed")
    ].copy()
    if base.empty:
        return detail
    copies = []
    for family, condition in [
        ("sensitivity_depth", "depth_5"),
        ("sensitivity_pin", "node_mass_default"),
        ("sensitivity_pin", "node_component_default"),
        ("sensitivity_pin", "node_atom_default"),
    ]:
        c = base.copy()
        c["phase"] = "Phase 2"
        c["family"] = family
        c["condition"] = condition
        c["reused_result"] = True
        copies.append(c)
    return pd.concat([detail, *copies], ignore_index=True)


def _collect_all_edge_from_target(detail: pd.DataFrame) -> pd.DataFrame:
    """Load the all-supervised-edge bundle paired with each target run."""
    keys = [
        "phase", "family", "model", "condition", "process", "fold", "server",
        "evaluation_split", "source_file", "reused_result",
    ]
    pieces: list[pd.DataFrame] = []
    for row in detail.loc[:, keys].drop_duplicates().itertuples(index=False):
        # Baseline transfer outputs intentionally contain only the target-edge
        # evaluation bundle.  Avoid one remote all-edge probe per transfer run.
        if row.family == "baseline_transfer_data_efficiency":
            continue
        source = str(row.source_file)
        if ";" in source:
            continue
        all_path = Path(source).with_name("pi_all_edge_property_r2.csv")
        if not all_path.is_file():
            continue
        metrics = _proposed_file_metrics(all_path)
        if metrics.empty:
            continue
        metrics.insert(0, "phase", row.phase)
        metrics.insert(1, "family", row.family)
        metrics.insert(2, "model", row.model)
        metrics.insert(3, "condition", row.condition)
        metrics.insert(4, "process", row.process)
        metrics.insert(5, "fold", row.fold)
        metrics.insert(6, "server", row.server)
        metrics.insert(7, "evaluation_split", row.evaluation_split)
        metrics.insert(8, "source_file", str(all_path))
        metrics.insert(9, "reused_result", row.reused_result)
        pieces.append(metrics)
    if not pieces:
        return pd.DataFrame(columns=detail.columns)
    out = pd.concat(pieces, ignore_index=True)
    out["property"] = pd.Categorical(out["property"], categories=PROPERTIES, ordered=True)
    return out.sort_values([
        "phase", "family", "model", "condition", "process", "fold", "property",
    ])


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame.loc[:, columns].copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6g}")
    header = "| " + " | ".join(view.columns) + " |"
    rule = "|" + "|".join(["---"] * len(view.columns)) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *body])


def _write_report(output: Path, summary: pd.DataFrame, status: pd.DataFrame, run: pd.DataFrame) -> None:
    show = summary.copy()
    show = show.sort_values(["phase", "family", "model", "condition"])
    report = f"""# Three-server final experiment metric summary

## Metric policy

- R2: property-wise pooled target-edge R2; the macro value is the mean of the 10 properties.
- MAE: property-wise physical-unit MAE. Its macro mixes heterogeneous units and is dominated by Mass_Flow.
- sMAE: MAE divided by pooled true population standard deviation. Lower is better.
- Baseline constant-target R2 follows the stored baseline policy (0.0).
- Proposed constant-target R2 follows the stored Proposed policy (0.999).
- Baselines use test artifacts. Proposed artifacts label the final-checkpoint metric bundle as val, so the CSV preserves that provenance explicitly.

## Completion audit

{_markdown_table(status, ['family', 'expected_new_or_direct_runs', 'found', 'missing', 'note'])}

## Phase/family macro performance

{_markdown_table(show, ['phase', 'family', 'model', 'condition', 'logical_runs', 'r2_mean', 'r2_median', 'r2_std', 'mae_mean_raw_units', 'smae_mean', 'smae_median', 'smae_std'])}

## Files

- `property_metrics_all_runs.csv`: every available logical run x 10 properties.
- `run_metrics_macro.csv`: 10-property macro metrics for every logical run.
- `phase_family_summary.csv`: phase/family/model/condition summary.
- `phase_family_property_summary.csv`: full 10-property summary.
- `completion_audit.csv`: expected versus currently found artifacts.
- `aggregation_metadata.json`: definitions and server roots.

## Current Phase 2 caveat

Phase 2 is incomplete if `completion_audit.csv` reports missing direct sensitivity runs. Depth 5 and the three default PIN settings reuse the five Proposed joint folds and are marked `reused_result=true`; these reused rows are not counted as newly completed sensitivity runs.
"""
    (output / "README_THREE_SERVER_METRICS.md").write_text(report, encoding="utf-8")


def _mean_std_text(mean: object, std: object) -> str:
    mean_value = _num(mean)
    std_value = _num(std)
    if not math.isfinite(mean_value):
        return ""
    std_text = f"{std_value:.6g}" if math.isfinite(std_value) else "NA"
    return f"{mean_value:.6g} ± {std_text}"


PHYSICS_005_COLUMNS = [
    "Mass pass rate (abs r <= 0.05)",
    "Component pass rate (abs r <= 0.05)",
    "Atom pass rate (abs r <= 0.05)",
]


def _physics_005_text(mean: object, std: object) -> str:
    mean_value = _num(mean)
    std_value = _num(std)
    if not math.isfinite(mean_value):
        return "NA (not evaluated)"
    std_text = f"{100.0 * std_value:.2f}%" if math.isfinite(std_value) else "NA"
    return f"{100.0 * mean_value:.2f}% ± {std_text}"


def _physics_lookup(
    physics: pd.DataFrame | None,
    frame: pd.DataFrame,
    label_col: str,
) -> dict[str, dict[str, str]] | None:
    """Return paper-display physics cells keyed by the displayed model name."""
    if physics is None:
        return None
    lookup: dict[str, dict[str, str]] = {}
    identities = frame.loc[:, ["family", "model", "condition", label_col]].drop_duplicates()
    term_to_column = {
        "mass": PHYSICS_005_COLUMNS[0],
        "component": PHYSICS_005_COLUMNS[1],
        "atom": PHYSICS_005_COLUMNS[2],
    }
    for identity in identities.itertuples(index=False):
        family, model, condition, label = identity
        cells = {column: "NA (not evaluated)" for column in PHYSICS_005_COLUMNS}
        matched = physics[
            (physics["family"] == family)
            & (physics["model"] == model)
            & (physics["condition"] == condition)
        ]
        for _, row in matched.iterrows():
            column = term_to_column.get(str(row["term"]))
            if column is not None:
                cells[column] = _physics_005_text(
                    row["satisfaction_rate_abs_le_0.05_mean"],
                    row["satisfaction_rate_abs_le_0.05_std"],
                )
        lookup[str(label)] = cells
    return lookup


# Keep the generated Markdown UTF-8-stable even when the editing shell uses a
# legacy code page.  These definitions intentionally supersede the legacy
# display helpers above, whose literal plus/minus character predates this
# cross-server aggregator.
def _mean_std_text(mean: object, std: object) -> str:
    mean_value = _num(mean)
    std_value = _num(std)
    if not math.isfinite(mean_value):
        return ""
    std_text = f"{std_value:.6g}" if math.isfinite(std_value) else "NA"
    return f"{mean_value:.6g} \u00b1 {std_text}"


def _physics_005_text(mean: object, std: object) -> str:
    mean_value = _num(mean)
    std_value = _num(std)
    if not math.isfinite(mean_value):
        return "NA (not evaluated)"
    std_text = f"{100.0 * std_value:.2f}%" if math.isfinite(std_value) else "NA"
    return f"{100.0 * mean_value:.2f}% \u00b1 {std_text}"


def _with_macro_stats(prop: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if prop.empty or summary.empty:
        return prop.copy()
    keys = ["phase", "family", "model", "condition"]
    macro = summary.loc[:, keys + [
        "r2_mean", "r2_std", "mae_mean_raw_units", "mae_std_raw_units",
        "smae_mean", "smae_std",
    ]].rename(columns={
        "r2_mean": "macro_r2_mean",
        "r2_std": "macro_r2_std",
        "mae_mean_raw_units": "macro_mae_mean",
        "mae_std_raw_units": "macro_mae_std",
        "smae_mean": "macro_smae_mean",
        "smae_std": "macro_smae_std",
    })
    return prop.merge(macro, on=keys, how="left", validate="many_to_one")


def _metric_matrix(
    frame: pd.DataFrame,
    label_col: str,
    *,
    formatted_mean_std: bool = False,
    physics_by_label: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, group in frame.groupby(label_col, sort=False, dropna=False, observed=True):
        indexed = group.set_index("property")
        for metric_name, value_col, std_col, macro_mean_col, macro_std_col in (
            ("sMAE", "smae_mean", "smae_std", "macro_smae_mean", "macro_smae_std"),
            ("MAE", "mae_mean", "mae_std", "macro_mae_mean", "macro_mae_std"),
        ):
            run_counts = pd.to_numeric(group["logical_runs"], errors="coerce").dropna()
            row: dict[str, object] = {
                "Model/Condition": label,
                "Metric": metric_name,
                "Logical runs": int(run_counts.max()) if not run_counts.empty else 0,
            }
            vals: list[float] = []
            for prop in PROPERTIES:
                value = _num(indexed.at[prop, value_col]) if prop in indexed.index else math.nan
                std = _num(indexed.at[prop, std_col]) if prop in indexed.index else math.nan
                row[prop] = _mean_std_text(value, std) if formatted_mean_std else value
                if math.isfinite(value):
                    vals.append(value)
            mean_value = float(np.mean(vals)) if vals else math.nan
            if formatted_mean_std:
                macro_mean = _num(group[macro_mean_col].iloc[0]) if macro_mean_col in group else mean_value
                macro_std = _num(group[macro_std_col].iloc[0]) if macro_std_col in group else math.nan
                row["Mean"] = _mean_std_text(macro_mean, macro_std)
            else:
                row["Mean"] = mean_value
            if physics_by_label is not None:
                cells = physics_by_label.get(str(label), {})
                for column in PHYSICS_005_COLUMNS:
                    # Conservation is model-level, not an output-head MAE.  It
                    # is shown once on the sMAE row to avoid a false MAE meaning.
                    row[column] = cells.get(column, "NA (not evaluated)") if metric_name == "sMAE" else "-"
            rows.append(row)
    columns = ["Model/Condition", "Metric", "Logical runs", *PROPERTIES, "Mean"]
    if physics_by_label is not None:
        columns += PHYSICS_005_COLUMNS
    return pd.DataFrame(rows, columns=columns)


def _build_model_metric_tables(
    prop: pd.DataFrame,
    *,
    formatted_mean_std: bool = False,
    physics: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}

    single = prop[prop["family"].isin(["single_process_baseline", "proposed_single_process"])].copy()
    single["display"] = single["model"].where(single["family"] == "single_process_baseline", "Proposed-single")
    order_single = [
        "M1", "M2", "M3", "B1", "B2", "B3", "B4", "B5", "B6", "B7",
        "G1", "G2", "G3", "Graphormer", "SAT", "GraphToSFILES", "Proposed-single",
    ]
    single["display"] = pd.Categorical(single["display"], order_single, ordered=True)
    tables["single_models"] = _metric_matrix(
        single.sort_values(["display", "property"]),
        "display",
        formatted_mean_std=formatted_mean_std,
        physics_by_label=_physics_lookup(physics, single, "display"),
    )

    multi = prop[prop["family"] == "multi_process_comparison"].copy()
    order_multi = ["B6", "GCN", "GIN", "GAT", "Graphormer", "SAT", "GraphToSFILES", "Proposed"]
    multi["display"] = pd.Categorical(multi["model"], order_multi, ordered=True)
    tables["multi_models"] = _metric_matrix(
        multi.sort_values(["display", "property"]),
        "display",
        formatted_mean_std=formatted_mean_std,
        physics_by_label=_physics_lookup(physics, multi, "display"),
    )

    zero = prop[prop["family"].isin(["proposed_zero_shot", "baseline_zero_shot"])].copy()
    zero["display"] = np.where(
        zero["family"].eq("proposed_zero_shot"), "Proposed-zero-shot",
        zero["model"].astype(str) + "-zero-shot",
    )
    order_zero = [
        "GCN-zero-shot", "GIN-zero-shot", "GAT-zero-shot", "Graphormer-zero-shot",
        "SAT-zero-shot", "GraphToSFILES-zero-shot", "Proposed-zero-shot",
    ]
    zero["display"] = pd.Categorical(zero["display"], order_zero, ordered=True)
    tables["zero_shot"] = _metric_matrix(
        zero,
        "display",
        formatted_mean_std=formatted_mean_std,
    )

    for family, name, prefix in [
        ("data_efficiency", "data_efficiency", ""),
        ("sensitivity_depth", "depth_sensitivity", "Proposed-"),
        ("sensitivity_pin", "pin_sensitivity", "Proposed-"),
    ]:
        sub = prop[prop["family"].isin([family, "baseline_transfer_data_efficiency"])].copy() if family == "data_efficiency" else prop[prop["family"] == family].copy()
        if family == "data_efficiency":
            sub["display"] = sub["model"].astype(str) + "-" + sub["condition"].astype(str)
        else:
            sub["display"] = prefix + sub["condition"].astype(str)
        tables[name] = _metric_matrix(
            sub.sort_values(["condition", "property"]),
            "display",
            formatted_mean_std=formatted_mean_std,
        )

    return tables


def _write_model_metric_tables(
    output: Path,
    target_prop: pd.DataFrame,
    all_prop: pd.DataFrame,
    target_summary: pd.DataFrame,
    all_summary: pd.DataFrame,
    physics_summary: pd.DataFrame,
    document_path: Path,
) -> None:
    scoped_tables: dict[str, pd.DataFrame] = {}
    for scope, prop in (("target_edge", target_prop), ("all_edge", all_prop)):
        if prop.empty:
            continue
        physics = physics_summary if scope == "target_edge" else None
        for name, table in _build_model_metric_tables(prop, physics=physics).items():
            if not table.empty:
                scoped_tables[f"{scope}_{name}"] = table

    display_tables: dict[str, pd.DataFrame] = {}
    for scope, prop, summary in (
        ("target_edge", target_prop, target_summary),
        ("all_edge", all_prop, all_summary),
    ):
        if prop.empty:
            continue
        display_prop = _with_macro_stats(prop, summary)
        physics = physics_summary if scope == "target_edge" else None
        for name, table in _build_model_metric_tables(
            display_prop,
            formatted_mean_std=True,
            physics=physics,
        ).items():
            if not table.empty:
                display_tables[f"{scope}_{name}"] = table

    for name, table in scoped_tables.items():
        table.to_csv(output / f"model_property_r2_mae_{name}.csv", index=False)
        table.to_csv(output / f"model_property_smae_mae_{name}.csv", index=False)
    # Keep the earlier unscoped CSV names synchronized with the target-edge
    # tables so downstream notebooks cannot silently read the stale snapshot.
    for name, table in _build_model_metric_tables(target_prop, physics=physics_summary).items():
        if not table.empty:
            table.to_csv(output / f"model_property_r2_mae_{name}.csv", index=False)
            table.to_csv(output / f"model_property_smae_mae_{name}.csv", index=False)
    try:
        with pd.ExcelWriter(output / "model_property_r2_mae_tables.xlsx", engine="openpyxl") as writer:
            for name, table in scoped_tables.items():
                table.to_excel(writer, sheet_name=name[:31], index=False)
                sheet = writer.sheets[name[:31]]
                sheet.freeze_panes = "D2"
                sheet.auto_filter.ref = sheet.dimensions
                sheet.column_dimensions["A"].width = 30
                sheet.column_dimensions["B"].width = 10
                sheet.column_dimensions["C"].width = 13
                for col in range(4, 15):
                    sheet.column_dimensions[chr(64 + col)].width = 16
    except ImportError:
        pass

    section_titles = [
        ("single_models", "Phase 1 / Phase 1-S: Single-process models"),
        ("multi_models", "Phase 1: Multi-process comparison"),
        ("zero_shot", "Phase 1: Zero-shot comparison"),
        ("depth_sensitivity", "Phase 2: Depth sensitivity"),
        ("pin_sensitivity", "Phase 2: PIN sensitivity"),
        ("data_efficiency", "Phase 3: Transfer data efficiency"),
    ]
    lines = [
        "# 전체 실험 성분별 R2 / MAE 결과",
        "",
        "> 세 서버의 현재 산출물을 logical run 단위로 통합한 결과이다.",
        "",
        "## 집계 기준",
        "",
        "- 행은 모델 또는 실험 조건마다 R2와 MAE 두 행으로 구성한다.",
        "- 각 성분은 `logical-run 평균 ± logical-run 표준편차`로 표시한다.",
        "- 열은 10개 예측 성분이며, `Mean`은 각 run의 10성분 평균을 다시 run 간 평균 ± 표준편차로 집계한 값이다.",
        "- `Logical runs`는 평균과 표준편차에 포함된 process/fold/ratio 조합 수이다. run이 하나면 표준편차는 `NA`로 표시한다.",
        "- Target edge는 공식 target-edge property 지표이고, All edge는 `pi_all_edge_property_r2.csv`의 supervised physical edges 지표이다.",
        "- 기존 baseline 패키지는 All-edge 성분 지표를 저장하지 않았으므로 Target-edge 표에만 포함한다.",
        "- MAE는 성분마다 물리 단위가 다르다. 마지막 MAE Mean은 요청에 따라 제공하지만 모델 비교에는 성분별 MAE를 우선 사용한다.",
        "- Phase 2의 default Depth 5 및 default PIN 조건은 Proposed joint 5-fold 결과를 재사용한다.",
        "",
    ]
    lines += [
        "## Reported predictive metrics",
        "",
        "- This version reports property-wise sMAE and physical-unit MAE only; R2 is intentionally excluded.",
        "- sMAE is MAE divided by the pooled true population standard deviation, so lower is better.",
        "",
        "## Physics conservation columns",
        "",
        "- The three rightmost columns in the Target edge tables are the stored physics-conservation satisfaction rates at `|normalized residual| <= 0.05`.",
        "- Each cell is the logical-run mean +/- sample standard deviation, displayed as a percentage. Higher is better.",
        "- Conservation is a model-level evaluator output, so it is printed on the sMAE row only; `-` on the paired MAE row means not applicable.",
        "- `NA (not evaluated)` means that no matched `physics_conservation_metrics.csv` exists for that exact model/condition. No cross-model value is substituted.",
        "",
    ]
    for key, title in section_titles:
        target = display_tables.get(f"target_edge_{key}")
        all_edge = display_tables.get(f"all_edge_{key}")
        if target is None and all_edge is None:
            continue
        lines += [f"## {title}", ""]
        if target is not None:
            lines += ["### Target edge", "", _markdown_table(target, list(target.columns)), ""]
        if all_edge is not None:
            lines += ["### All edge", "", _markdown_table(all_edge, list(all_edge.columns)), ""]
    lines += [
        "## 산출물 파일", "",
        "- `model_property_smae_mae_*.csv`: sMAE/MAE tables (the legacy `r2_mae` filename is kept as a compatibility alias)",
        "- `property_metrics_all_runs.csv`: Target-edge logical run별 원자료",
        "- `property_metrics_all_edges_all_runs.csv`: All-edge logical run별 원자료",
        "",
    ]
    rendered = "\n".join(lines)
    (output / "ALL_EXPERIMENT_RESULTS_R2_MAE.md").write_text(rendered, encoding="utf-8")
    document_path.parent.mkdir(parents=True, exist_ok=True)
    document_path.write_text(rendered, encoding="utf-8")


def _write_proposed_only_report(
    output: Path,
    target_prop: pd.DataFrame,
    all_prop: pd.DataFrame,
    target_summary: pd.DataFrame,
    all_summary: pd.DataFrame,
    physics_summary: pd.DataFrame,
    document_path: Path,
) -> None:
    experiment_names = {
        "multi_process_comparison": "Joint multi-process",
        "proposed_single_process": "Single-process",
        "proposed_zero_shot": "Zero-shot",
        "sensitivity_depth": "Depth sensitivity",
        "sensitivity_pin": "PIN sensitivity",
        "data_efficiency": "Transfer data efficiency",
    }
    family_order = {name: index for index, name in enumerate(experiment_names)}

    def build(
        prop: pd.DataFrame,
        summary: pd.DataFrame,
        physics: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        proposed = prop[prop["model"].astype(str).str.startswith("Proposed")].copy()
        proposed = proposed[proposed["family"].isin(experiment_names)].copy()
        if proposed.empty:
            return pd.DataFrame()
        proposed = _with_macro_stats(proposed, summary)
        proposed["family_order"] = proposed["family"].map(family_order)
        proposed = proposed.sort_values([
            "phase", "family_order", "condition", "property",
        ])
        pieces: list[pd.DataFrame] = []
        keys = ["phase", "family", "condition"]
        for (phase, family, condition), group in proposed.groupby(
            keys,
            sort=False,
            observed=True,
        ):
            group = group.copy()
            group["display"] = str(condition)
            matrix = _metric_matrix(
                group,
                "display",
                formatted_mean_std=True,
                physics_by_label=_physics_lookup(physics, group, "display"),
            ).rename(columns={"Model/Condition": "Condition"})
            matrix.insert(0, "Experiment", experiment_names[str(family)])
            matrix.insert(0, "Phase", str(phase))
            pieces.append(matrix)
        return pd.concat(pieces, ignore_index=True)

    target = build(target_prop, target_summary, physics_summary)
    all_edge = build(all_prop, all_summary)
    lines = [
        "# Proposed 모델 전체 실험 성능: R2 / MAE",
        "",
        "> Baseline을 제외하고 Proposed 계열 모델의 모든 현재 실험 결과만 정리하였다.",
        "",
        "## 집계 기준",
        "",
        "- 각 성분은 `logical-run 평균 ± logical-run 표준편차`이다.",
        "- `Mean`은 각 run의 10개 성분 평균을 계산한 뒤, 이를 run 간 평균 ± 표준편차로 집계한 값이다.",
        "- `Logical runs`는 집계에 포함된 process/fold/ratio 조합 수이다.",
        "- MAE는 성분마다 물리 단위가 다르므로 `MAE Mean`보다 성분별 MAE를 우선 해석한다.",
        "- Depth 5와 세 PIN default 조건은 Proposed joint 5-fold 결과를 재사용한다.",
        "- 아직 일부 fold가 끝나지 않은 조건은 현재 확보된 logical run만 집계하며 그 수를 명시한다.",
        "",
    ]
    lines += [
        "## Reported predictive metrics",
        "",
        "- This version reports property-wise sMAE and physical-unit MAE only; R2 is intentionally excluded.",
        "- sMAE is MAE divided by the pooled true population standard deviation; lower is better.",
        "",
    ]
    if not target.empty:
        lines += ["## Target edge", "", _markdown_table(target, list(target.columns)), ""]
    if not all_edge.empty:
        lines += ["## All edge", "", _markdown_table(all_edge, list(all_edge.columns)), ""]
    rendered = "\n".join(lines)
    (output / "PROPOSED_MODEL_RESULTS_R2_MAE.md").write_text(rendered, encoding="utf-8")
    document_path.parent.mkdir(parents=True, exist_ok=True)
    document_path.write_text(rendered, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate phase metrics across the three experiment servers")
    parser.add_argument("--server1", default=r"T:\home\nsuser\화학공정0610loss아카이브")
    parser.add_argument("--server2", default=r"Z:\home\jw\화학공정final")
    parser.add_argument("--server3", default=r"Y:\home\oj\화학공정final")
    parser.add_argument("--output-root", default="outputs/0819final/three_server_aggregate")
    parser.add_argument("--document-path", default="docs/ALL_EXPERIMENT_RESULTS_R2_MAE_0904.md")
    parser.add_argument("--proposed-document-path", default="docs/PROPOSED_MODEL_RESULTS_R2_MAE_0904.md")
    args = parser.parse_args()
    servers = [Server("server1", Path(args.server1)), Server("server2", Path(args.server2)), Server("server3", Path(args.server3))]
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pieces: list[pd.DataFrame] = []
    _collect_single_baselines(servers, pieces)
    _collect_multi_baselines(servers, pieces)
    _collect_transfer_baselines(servers, pieces)
    _collect_zero_shot_baselines(servers, pieces)
    _collect_proposed(servers, pieces)
    _collect_sensitivity(servers, pieces)
    physics_detail = _collect_physics_conservation(servers)
    physics_summary = _summarize_physics_conservation(physics_detail)
    if not pieces:
        raise RuntimeError("no metric artifacts found")
    detail = pd.concat(pieces, ignore_index=True)
    detail = _add_sensitivity_reuse(detail)
    detail["property"] = pd.Categorical(detail["property"], categories=PROPERTIES, ordered=True)
    detail = detail.sort_values(["phase", "family", "model", "condition", "process", "fold", "property"])
    run = _macro_rows(detail)
    summary, prop = _summaries(run, detail)
    all_detail = _collect_all_edge_from_target(detail)
    if all_detail.empty:
        all_run = pd.DataFrame()
        all_summary = pd.DataFrame()
        all_prop = pd.DataFrame()
    else:
        all_run = _macro_rows(all_detail)
        all_summary, all_prop = _summaries(all_run, all_detail)
    status = _expected_status(run)
    detail.to_csv(output / "property_metrics_all_runs.csv", index=False)
    run.to_csv(output / "run_metrics_macro.csv", index=False)
    summary.to_csv(output / "phase_family_summary.csv", index=False)
    prop.to_csv(output / "phase_family_property_summary.csv", index=False)
    all_detail.to_csv(output / "property_metrics_all_edges_all_runs.csv", index=False)
    all_run.to_csv(output / "run_metrics_macro_all_edges.csv", index=False)
    all_summary.to_csv(output / "phase_family_summary_all_edges.csv", index=False)
    all_prop.to_csv(output / "phase_family_property_summary_all_edges.csv", index=False)
    status.to_csv(output / "completion_audit.csv", index=False)
    physics_detail.to_csv(output / "physics_conservation_all_runs.csv", index=False)
    physics_summary.to_csv(output / "physics_conservation_summary.csv", index=False)
    metadata = {
        "servers": {s.name: str(s.root) for s in servers},
        "r2_definition": "pooled target-edge samples by property; macro is mean of 10 properties",
        "mae_definition": "pooled physical-unit MAE by property",
        "smae_definition": "MAE divided by pooled true population standard deviation; constant exact targets assigned 0",
        "baseline_constant_r2": 0.0,
        "proposed_constant_r2": 0.999,
        "warning": "Raw-unit MAE macro mixes heterogeneous units and is dominated by Mass_Flow; prefer R2 and sMAE for comparison.",
        "logical_run_count": int(len(run)),
        "property_row_count": int(len(detail)),
    }
    (output / "aggregation_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_report(output, summary, status, run)
    _write_model_metric_tables(
        output,
        prop,
        all_prop,
        summary,
        all_summary,
        physics_summary,
        Path(args.document_path).resolve(),
    )
    _write_proposed_only_report(
        output,
        prop,
        all_prop,
        summary,
        all_summary,
        physics_summary,
        Path(args.proposed_document_path).resolve(),
    )
    print(status.to_string(index=False))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
