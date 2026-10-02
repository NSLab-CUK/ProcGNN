#!/usr/bin/env python3
"""Create 5-fold unseen-process zero-shot manifests from per-process K-fold splits."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = ("process_id", "sample_id", "merged_row_index")


def _process_label(pid: int) -> str:
    return f"Process{int(pid)}"


def _fold_label(fold: int) -> str:
    return f"fold_{int(fold):02d}"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _read_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"manifest is empty: {path}")
    return df


def _write_manifest(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _row_set(df: pd.DataFrame) -> set[int]:
    return {int(x) for x in df["merged_row_index"].tolist()}


def _validate_processes(df: pd.DataFrame, *, allowed: set[str], label: str) -> None:
    found = set(df["process_id"].astype(str).unique().tolist())
    bad = sorted(found - allowed)
    if bad:
        raise RuntimeError(f"{label}: unexpected process_id values: {bad}; allowed={sorted(allowed)}")


def _validate_disjoint(name: str, *frames: pd.DataFrame) -> None:
    sets = [_row_set(df) for df in frames]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise RuntimeError(f"{name}: row overlap detected ({len(overlap)}), first={sorted(overlap)[:5]}")


def _validate_indices(df: pd.DataFrame, *, max_index: int | None, label: str) -> None:
    vals = pd.to_numeric(df["merged_row_index"], errors="raise").astype(int)
    if (vals < 0).any():
        raise RuntimeError(f"{label}: negative merged_row_index found")
    if max_index is not None and (vals >= int(max_index)).any():
        raise RuntimeError(f"{label}: merged_row_index exceeds merged CSV row count {max_index}")


def _concat_manifests(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [_read_manifest(path) for path in paths]
    cols = list(frames[0].columns)
    out = pd.concat(frames, ignore_index=True)
    return out.loc[:, cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create unseen-process zero-shot fold manifests.")
    parser.add_argument("--heldout-process", type=int, required=True)
    parser.add_argument("--source-process-ids", type=int, nargs="+", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--input-split-root", type=str, default="data/splits/process_kfold")
    parser.add_argument("--merged-csv", type=str, default="data/datasets_v3/process_main_merged_all_processes_10pct.csv")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    heldout_label = _process_label(args.heldout_process)
    source_labels = [_process_label(pid) for pid in args.source_process_ids]
    if heldout_label in set(source_labels):
        raise ValueError("heldout process cannot also be a source process")

    input_root = _resolve(args.input_split_root)
    output_root = _resolve(args.output_root)
    merged_path = _resolve(args.merged_csv)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output root already exists and is not empty: {output_root} (use --overwrite)")
    output_root.mkdir(parents=True, exist_ok=True)

    max_index: int | None = None
    index_validation_policy = "nonnegative_numeric_only"

    fold_rows: list[dict[str, object]] = []
    all_test_sets: dict[int, set[int]] = {}
    for fold in [int(f) for f in args.folds]:
        flabel = _fold_label(fold)
        source_train_paths = [input_root / proc / flabel / "train.csv" for proc in source_labels]
        source_val_paths = [input_root / proc / flabel / "val.csv" for proc in source_labels]
        heldout_test_path = input_root / heldout_label / flabel / "test.csv"

        source_train = _concat_manifests(source_train_paths)
        source_val = _concat_manifests(source_val_paths)
        heldout_test = _read_manifest(heldout_test_path)

        _validate_processes(source_train, allowed=set(source_labels), label=f"{flabel}/source_train")
        _validate_processes(source_val, allowed=set(source_labels), label=f"{flabel}/source_val")
        _validate_processes(heldout_test, allowed={heldout_label}, label=f"{flabel}/heldout_test")
        for label, frame in (
            (f"{flabel}/source_train", source_train),
            (f"{flabel}/source_val", source_val),
            (f"{flabel}/heldout_test", heldout_test),
        ):
            _validate_indices(frame, max_index=max_index, label=label)
        _validate_disjoint(flabel, source_train, source_val, heldout_test)

        found_source_train = set(source_train["process_id"].astype(str).unique().tolist())
        found_source_val = set(source_val["process_id"].astype(str).unique().tolist())
        if found_source_train != set(source_labels):
            raise RuntimeError(f"{flabel}/source_train missing source processes: {sorted(set(source_labels) - found_source_train)}")
        if found_source_val != set(source_labels):
            raise RuntimeError(f"{flabel}/source_val missing source processes: {sorted(set(source_labels) - found_source_val)}")

        fold_dir = output_root / flabel
        _write_manifest(source_train, fold_dir / "source_train.csv")
        _write_manifest(source_val, fold_dir / "source_val.csv")
        _write_manifest(heldout_test, fold_dir / "heldout_test.csv")

        test_set = _row_set(heldout_test)
        all_test_sets[fold] = test_set
        metadata = {
            "fold": fold,
            "heldout_process": int(args.heldout_process),
            "source_process_ids": [int(p) for p in args.source_process_ids],
            "source_train_count": int(len(source_train)),
            "source_val_count": int(len(source_val)),
            "heldout_test_count": int(len(heldout_test)),
            "source_train_processes": sorted(found_source_train),
            "source_val_processes": sorted(found_source_val),
            "heldout_test_processes": sorted(set(heldout_test["process_id"].astype(str).unique().tolist())),
            "input_split_root": input_root.relative_to(PROJECT_ROOT).as_posix() if input_root.is_relative_to(PROJECT_ROOT) else str(input_root),
            "source_train_paths": [str(p.relative_to(PROJECT_ROOT).as_posix()) if p.is_relative_to(PROJECT_ROOT) else str(p) for p in source_train_paths],
            "source_val_paths": [str(p.relative_to(PROJECT_ROOT).as_posix()) if p.is_relative_to(PROJECT_ROOT) else str(p) for p in source_val_paths],
            "heldout_test_path": str(heldout_test_path.relative_to(PROJECT_ROOT).as_posix()) if heldout_test_path.is_relative_to(PROJECT_ROOT) else str(heldout_test_path),
            "leakage_check": "pass",
        }
        (fold_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        fold_rows.append(metadata)

    overlap_rows: list[dict[str, object]] = []
    folds = sorted(all_test_sets)
    for i, fi in enumerate(folds):
        for fj in folds[i + 1 :]:
            overlap = all_test_sets[fi] & all_test_sets[fj]
            overlap_rows.append({"fold_i": fi, "fold_j": fj, "overlap_count": len(overlap)})
    test_overlap_policy = "disjoint" if all(int(r["overlap_count"]) == 0 for r in overlap_rows) else "overlap_present"

    root_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "heldout_process": int(args.heldout_process),
        "source_process_ids": [int(p) for p in args.source_process_ids],
        "folds": [int(f) for f in args.folds],
        "input_split_root": input_root.relative_to(PROJECT_ROOT).as_posix() if input_root.is_relative_to(PROJECT_ROOT) else str(input_root),
        "output_root": output_root.relative_to(PROJECT_ROOT).as_posix() if output_root.is_relative_to(PROJECT_ROOT) else str(output_root),
        "merged_csv": str(Path(args.merged_csv).as_posix()),
        "merged_row_index_validation_policy": index_validation_policy,
        "test_overlap_policy": test_overlap_policy,
        "test_overlap_checks": overlap_rows,
        "fold_summaries": [
            {
                "fold": int(row["fold"]),
                "source_train_count": int(row["source_train_count"]),
                "source_val_count": int(row["source_val_count"]),
                "heldout_test_count": int(row["heldout_test_count"]),
                "leakage_check": row["leakage_check"],
            }
            for row in fold_rows
        ],
    }
    (output_root / "metadata.json").write_text(json.dumps(root_meta, indent=2), encoding="utf-8")
    pd.DataFrame(root_meta["fold_summaries"]).to_csv(output_root / "fold_summary.csv", index=False)
    pd.DataFrame(overlap_rows).to_csv(output_root / "heldout_test_overlap_checks.csv", index=False)
    print(f"[unseen-splits] wrote {output_root}")
    for row in root_meta["fold_summaries"]:
        print(
            f"[unseen-splits][fold {int(row['fold']):02d}] "
            f"source_train={row['source_train_count']} source_val={row['source_val_count']} "
            f"heldout_test={row['heldout_test_count']} leakage={row['leakage_check']}"
        )
    print(f"[unseen-splits] heldout test overlap policy={test_overlap_policy}")


if __name__ == "__main__":
    main()
