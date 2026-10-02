"""Create full-data group-aware outer 5-fold manifests for all-process training."""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _id_key(value: Any) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "<na>"}:
        return ""
    try:
        fval = float(text)
        if math.isfinite(fval) and fval.is_integer():
            return str(int(fval))
    except (TypeError, ValueError):
        pass
    return text


def _process_num(value: Any) -> int:
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        raise ValueError(f"cannot parse process id from {value!r}")
    return int(digits)


def _as_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def _write_manifest(path: Path, df: pd.DataFrame, indices: list[int]) -> None:
    rows = df.loc[indices, ["process_id", "ID"]].copy()
    rows.insert(1, "sample_id", rows["ID"].map(_id_key))
    rows["merged_row_index"] = [int(i) for i in indices]
    rows["source_merged_row_index"] = rows["merged_row_index"]
    rows["group_key"] = rows["process_id"].astype(str) + "::" + rows["sample_id"].astype(str)
    rows = rows[["process_id", "sample_id", "merged_row_index", "source_merged_row_index", "group_key"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)


def _process_counts(df: pd.DataFrame, indices: list[int]) -> dict[str, int]:
    return {
        str(k): int(v)
        for k, v in df.loc[indices, "process_id"].value_counts(dropna=False).sort_index().items()
    }


def _target_stream_map(project_root: Path) -> dict[int, set[str]]:
    target_path = project_root / "data/reference/v4/target_stream_targets.csv"
    edge_path = project_root / "data/reference/v3/canonical_edges.csv"
    if not target_path.is_file() or not edge_path.is_file():
        return {}
    targets = pd.read_csv(target_path, usecols=["process_id", "canonical_answer_edge_id"])
    edges = pd.read_csv(edge_path, usecols=["process_id", "canonical_edge_id", "main_data_stream_key"])
    target_ids = {
        (int(row.process_id), str(row.canonical_answer_edge_id).strip())
        for row in targets.itertuples(index=False)
    }
    result: dict[int, set[str]] = {}
    for row in edges.itertuples(index=False):
        key = (int(row.process_id), str(row.canonical_edge_id).strip())
        if key in target_ids:
            result.setdefault(int(row.process_id), set()).add(_id_key(row.main_data_stream_key))
    return result


def _load_stream_cache(project_root: Path) -> dict[int, pd.DataFrame]:
    stream_dir = project_root / "data/main_data_Streams"
    cache: dict[int, pd.DataFrame] = {}
    columns = [
        "ID",
        "Stream_Name",
        "Mass_Flow",
        "Frac_H2O",
        "Frac_H2",
        "Frac_CH4",
        "Frac_CO2",
        "Frac_CO",
        "Frac_O2",
        "Frac_N2",
    ]
    for path in sorted(stream_dir.glob("*.Process_Streams.csv")):
        process_num = int(path.name.split(".", 1)[0])
        frame = pd.read_csv(path, usecols=lambda col: col in set(columns))
        frame["_sample_id"] = frame["ID"].map(_id_key)
        frame["_stream_key"] = frame["Stream_Name"].map(_id_key)
        cache[process_num] = frame
    return cache


def _target_distribution_summary(
    *,
    df: pd.DataFrame,
    indices: list[int],
    target_streams: dict[int, set[str]],
    stream_cache: dict[int, pd.DataFrame],
) -> dict[str, Any]:
    frac_cols = ["Frac_H2O", "Frac_H2", "Frac_CH4", "Frac_CO2", "Frac_CO", "Frac_O2", "Frac_N2"]
    pieces: list[pd.DataFrame] = []
    for process_num, sub in df.loc[indices].groupby("_process_num", sort=True):
        process_num = int(process_num)
        streams = target_streams.get(process_num, set())
        stream = stream_cache.get(process_num)
        if stream is None or not streams:
            continue
        wanted = set(sub["_sample_id"].map(_id_key))
        part = stream[stream["_sample_id"].isin(wanted) & stream["_stream_key"].isin(streams)]
        if not part.empty:
            pieces.append(part)
    if not pieces:
        return {"target_edge_row_count": 0}
    merged = pd.concat(pieces, ignore_index=True)
    frac_summary = {}
    for col in frac_cols:
        if col not in merged.columns:
            continue
        vals = pd.to_numeric(merged[col], errors="coerce").dropna()
        if vals.empty:
            continue
        frac_summary[col] = {
            "count": int(vals.shape[0]),
            "zero_ratio_le_1e-8": float((vals <= 1.0e-8).mean()),
            "positive_ratio_gt_1e-8": float((vals > 1.0e-8).mean()),
        }
    mass_quantiles = {}
    if "Mass_Flow" in merged.columns:
        mass = pd.to_numeric(merged["Mass_Flow"], errors="coerce").dropna()
        if not mass.empty:
            for q in [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]:
                mass_quantiles[f"q{q:.2f}"] = float(mass.quantile(q))
    return {
        "target_edge_row_count": int(merged.shape[0]),
        "target_fraction_zero_positive_ratio": frac_summary,
        "target_mass_flow_quantiles": mass_quantiles,
    }


def _split_process_groups(
    groups: list[str],
    *,
    n_splits: int,
    seed: int,
) -> list[list[str]]:
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    folds = [[] for _ in range(n_splits)]
    for idx, group in enumerate(shuffled):
        folds[idx % n_splits].append(group)
    return folds


def _check_no_overlap(*, train: set[str], val: set[str], test: set[str]) -> dict[str, Any]:
    tv = train & val
    tt = train & test
    vt = val & test
    ok = not tv and not tt and not vt
    return {
        "ok": bool(ok),
        "train_val_overlap": len(tv),
        "train_test_overlap": len(tt),
        "val_test_overlap": len(vt),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default="data/datasets_v3/process_main_merged.csv")
    parser.add_argument("--output-root", default="data/splits/all_processes_full100k_outer5_grouped_60_20_20")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=260716)
    parser.add_argument("--validation-fraction-of-outer-train", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_csv = _resolve(args.input_csv)
    output_root = _resolve(args.output_root)
    if not input_csv.is_file():
        raise FileNotFoundError(input_csv)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_root} already exists; pass --overwrite to replace manifests.")

    df = pd.read_csv(input_csv)
    required = {"process_id", "ID"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"input CSV missing required columns: {sorted(missing)}")
    df = df.reset_index(drop=True)
    df["_sample_id"] = df["ID"].map(_id_key)
    if bool(df["_sample_id"].eq("").any()):
        raise ValueError("empty ID values found; cannot build group key.")
    df["_group_key"] = df["process_id"].astype(str) + "::" + df["_sample_id"].astype(str)
    df["_process_num"] = df["process_id"].map(_process_num)
    target_streams = _target_stream_map(PROJECT_ROOT)
    stream_cache = _load_stream_cache(PROJECT_ROOT)

    process_group_folds: dict[int, list[list[str]]] = {}
    for process_num, proc_df in df.groupby("_process_num", sort=True):
        groups = sorted(proc_df["_group_key"].unique())
        process_group_folds[int(process_num)] = _split_process_groups(
            groups,
            n_splits=int(args.folds),
            seed=int(args.seed) + int(process_num) * 1009,
        )

    group_to_indices = {
        str(group): [int(i) for i in idxs]
        for group, idxs in df.groupby("_group_key", sort=False).groups.items()
    }
    all_test_counts: Counter[str] = Counter()
    metadata_rows: list[dict[str, Any]] = []
    for fold_idx in range(int(args.folds)):
        test_groups: set[str] = set()
        for process_num, folds in process_group_folds.items():
            test_groups.update(folds[fold_idx])
        outer_train_groups = sorted(set(group_to_indices) - test_groups)
        val_groups: set[str] = set()
        for process_num in sorted(process_group_folds):
            proc_train_groups = [
                group
                for group in outer_train_groups
                if _process_num(group.split("::", 1)[0]) == int(process_num)
            ]
            rng = random.Random(int(args.seed) + (fold_idx + 1) * 100003 + int(process_num) * 9176)
            rng.shuffle(proc_train_groups)
            val_n = max(1, int(round(len(proc_train_groups) * float(args.validation_fraction_of_outer_train))))
            val_groups.update(proc_train_groups[:val_n])
        train_groups = sorted(set(outer_train_groups) - val_groups)

        train_idx = sorted(i for group in train_groups for i in group_to_indices[group])
        val_idx = sorted(i for group in val_groups for i in group_to_indices[group])
        test_idx = sorted(i for group in test_groups for i in group_to_indices[group])
        train_set = set(train_groups)
        val_set = set(val_groups)
        test_set = set(test_groups)
        overlap = _check_no_overlap(train=train_set, val=val_set, test=test_set)
        if not overlap["ok"]:
            raise RuntimeError(f"fold {fold_idx + 1} group overlap detected: {overlap}")
        raw_overlap = {
            "train_val": len(set(train_idx) & set(val_idx)),
            "train_test": len(set(train_idx) & set(test_idx)),
            "val_test": len(set(val_idx) & set(test_idx)),
        }
        if any(raw_overlap.values()):
            raise RuntimeError(f"fold {fold_idx + 1} raw row overlap detected: {raw_overlap}")
        for group in test_groups:
            all_test_counts[group] += 1

        fold_dir = output_root / f"fold_{fold_idx + 1:02d}"
        _write_manifest(fold_dir / "train.csv", df, train_idx)
        _write_manifest(fold_dir / "val.csv", df, val_idx)
        _write_manifest(fold_dir / "test.csv", df, test_idx)
        meta = {
            "seed": int(args.seed),
            "fold": int(fold_idx + 1),
            "group_key_definition": "process_id + ID",
            "input_csv": _as_rel(input_csv),
            "git_commit": _git_commit(),
            "total_sample_count": int(len(df)),
            "total_group_count": int(len(group_to_indices)),
            "train_sample_count": int(len(train_idx)),
            "val_sample_count": int(len(val_idx)),
            "test_sample_count": int(len(test_idx)),
            "train_group_count": int(len(train_groups)),
            "val_group_count": int(len(val_groups)),
            "test_group_count": int(len(test_groups)),
            "target_ratios": {
                "train": (1.0 - 1.0 / max(1, int(args.folds)))
                * (1.0 - float(args.validation_fraction_of_outer_train)),
                "val": (1.0 - 1.0 / max(1, int(args.folds)))
                * float(args.validation_fraction_of_outer_train),
                "test": 1.0 / max(1, int(args.folds)),
            },
            "actual_ratios": {
                "train": len(train_idx) / max(1, len(df)),
                "val": len(val_idx) / max(1, len(df)),
                "test": len(test_idx) / max(1, len(df)),
            },
            "process_sample_counts": {
                "train": _process_counts(df, train_idx),
                "val": _process_counts(df, val_idx),
                "test": _process_counts(df, test_idx),
            },
            "target_distribution_summary": {
                "train": _target_distribution_summary(
                    df=df,
                    indices=train_idx,
                    target_streams=target_streams,
                    stream_cache=stream_cache,
                ),
                "val": _target_distribution_summary(
                    df=df,
                    indices=val_idx,
                    target_streams=target_streams,
                    stream_cache=stream_cache,
                ),
                "test": _target_distribution_summary(
                    df=df,
                    indices=test_idx,
                    target_streams=target_streams,
                    stream_cache=stream_cache,
                ),
            },
            "overlap_checks": {**overlap, "raw_row_overlap": raw_overlap},
        }
        (fold_dir / "split_metadata.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        metadata_rows.append(meta)
        print(
            f"[outer-folds] fold_{fold_idx + 1:02d} "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
            f"groups train={len(train_groups)} val={len(val_groups)} test={len(test_groups)}",
            flush=True,
        )

    bad_test = {group: count for group, count in all_test_counts.items() if int(count) != 1}
    if bad_test:
        raise RuntimeError(f"some groups are not test exactly once across folds: {list(bad_test.items())[:5]}")
    summary = {
        "seed": int(args.seed),
        "folds": int(args.folds),
        "input_csv": _as_rel(input_csv),
        "output_root": _as_rel(output_root),
        "group_key_definition": "process_id + ID",
        "total_sample_count": int(len(df)),
        "total_group_count": int(len(group_to_indices)),
        "test_once_check": {"ok": True, "group_count": int(len(all_test_counts))},
        "folds_metadata": metadata_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[outer-folds] wrote {output_root}", flush=True)


if __name__ == "__main__":
    main()
