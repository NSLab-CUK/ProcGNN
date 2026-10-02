#!/usr/bin/env python3
"""Create single-heldout full-transfer unseen manifests.

Layout per held-out process:

heldout_P01/
  source_train.csv        # source 9 processes, one fixed source fold train
  source_val.csv          # source 9 processes, one fixed source fold val
  fold_01/
    target_train.csv      # held-out process fold train, about 6000 rows
    target_val.csv        # held-out process fold validation, about 2000 rows
    target_test.csv       # held-out process fold test, about 2000 rows

The source pretrain manifests are intentionally written once per held-out
process so the pretrain checkpoint can be reused across five target folds.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = ("process_id", "sample_id", "merged_row_index")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _as_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _process_label(pid: int) -> str:
    return f"Process{int(pid)}"


def _heldout_label(pid: int) -> str:
    return f"heldout_P{int(pid):02d}"


def _fold_label(fold: int) -> str:
    return f"fold_{int(fold):02d}"


def _read_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"manifest is empty: {path}")
    return df


def _concat(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [_read_manifest(path) for path in paths]
    cols = list(frames[0].columns)
    return pd.concat(frames, ignore_index=True).loc[:, cols]


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _row_set(df: pd.DataFrame) -> set[int]:
    return {int(x) for x in pd.to_numeric(df["merged_row_index"], errors="raise").astype(int).tolist()}


def _validate_processes(df: pd.DataFrame, *, expected: set[str], label: str) -> None:
    found = set(df["process_id"].astype(str).unique().tolist())
    bad = sorted(found - expected)
    if bad:
        raise RuntimeError(f"{label}: unexpected process_id values {bad}; expected={sorted(expected)}")


def _validate_disjoint(label: str, *frames: pd.DataFrame) -> None:
    sets = [_row_set(frame) for frame in frames]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise RuntimeError(f"{label}: row overlap detected ({len(overlap)}), first={sorted(overlap)[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create single-process full unseen pretrain/transfer manifests.")
    parser.add_argument("--input-split-root", default="data/splits/process_kfold")
    parser.add_argument("--output-root", default="data/splits/single_process_full_unseen_60_20_20")
    parser.add_argument("--heldout-processes", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--pretrain-source-fold", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_root = _resolve(args.input_split_root)
    output_root = _resolve(args.output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output root exists and is not empty: {output_root} (use --overwrite)")
    output_root.mkdir(parents=True, exist_ok=True)

    all_processes = set(range(1, 11))
    global_rows: list[dict[str, object]] = []
    for heldout in [int(p) for p in args.heldout_processes]:
        if heldout not in all_processes:
            raise ValueError(f"heldout process must be in 1..10, got {heldout}")
        source_ids = sorted(all_processes - {heldout})
        source_labels = {_process_label(pid) for pid in source_ids}
        heldout_label = _heldout_label(heldout)
        heldout_dir = output_root / heldout_label
        source_fold = _fold_label(int(args.pretrain_source_fold))

        source_train = _concat(
            input_root / _process_label(pid) / source_fold / "train.csv" for pid in source_ids
        )
        source_val = _concat(
            input_root / _process_label(pid) / source_fold / "val.csv" for pid in source_ids
        )
        _validate_processes(source_train, expected=source_labels, label=f"{heldout_label}/source_train")
        _validate_processes(source_val, expected=source_labels, label=f"{heldout_label}/source_val")
        _validate_disjoint(f"{heldout_label}/source", source_train, source_val)
        _write(source_train, heldout_dir / "source_train.csv")
        _write(source_val, heldout_dir / "source_val.csv")

        fold_rows: list[dict[str, object]] = []
        for fold in [int(f) for f in args.folds]:
            flabel = _fold_label(fold)
            target_train = _read_manifest(input_root / _process_label(heldout) / flabel / "train.csv")
            target_val = _read_manifest(input_root / _process_label(heldout) / flabel / "val.csv")
            target_test = _read_manifest(input_root / _process_label(heldout) / flabel / "test.csv")
            target_expected = {_process_label(heldout)}
            _validate_processes(target_train, expected=target_expected, label=f"{heldout_label}/{flabel}/target_train")
            _validate_processes(target_val, expected=target_expected, label=f"{heldout_label}/{flabel}/target_val")
            _validate_processes(target_test, expected=target_expected, label=f"{heldout_label}/{flabel}/target_test")
            _validate_disjoint(
                f"{heldout_label}/{flabel}",
                source_train,
                source_val,
                target_train,
                target_val,
                target_test,
            )
            fold_dir = heldout_dir / flabel
            _write(target_train, fold_dir / "target_train.csv")
            _write(target_val, fold_dir / "target_val.csv")
            _write(target_test, fold_dir / "target_test.csv")
            row = {
                "heldout_process": heldout,
                "fold": fold,
                "source_train_rows": int(len(source_train)),
                "source_val_rows": int(len(source_val)),
                "target_train_rows": int(len(target_train)),
                "target_val_rows": int(len(target_val)),
                "target_test_rows": int(len(target_test)),
                "source_process_ids": ",".join(str(pid) for pid in source_ids),
            }
            fold_rows.append(row)
            global_rows.append(row)
            print(
                f"[single-full-splits] {heldout_label}/{flabel} "
                f"source_train={len(source_train)} source_val={len(source_val)} "
                f"target_train={len(target_train)} target_val={len(target_val)} "
                f"target_test={len(target_test)}",
                flush=True,
            )

        metadata = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "heldout_process": heldout,
            "source_process_ids": source_ids,
            "pretrain_source_fold": int(args.pretrain_source_fold),
            "folds": [int(f) for f in args.folds],
            "source_train_rows": int(len(source_train)),
            "source_val_rows": int(len(source_val)),
            "transfer_policy": (
                "independent target fold train/validation/test; data ratios apply only to target train"
            ),
            "folds_metadata": fold_rows,
        }
        (heldout_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        pd.DataFrame(fold_rows).to_csv(heldout_dir / "fold_summary.csv", index=False)

    root_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_split_root": _as_rel(input_root),
        "output_root": _as_rel(output_root),
        "heldout_processes": [int(p) for p in args.heldout_processes],
        "folds": [int(f) for f in args.folds],
        "pretrain_source_fold": int(args.pretrain_source_fold),
    }
    (output_root / "metadata.json").write_text(json.dumps(root_meta, indent=2), encoding="utf-8")
    pd.DataFrame(global_rows).to_csv(output_root / "global_split_summary.csv", index=False)
    print(f"[single-full-splits] wrote {output_root}", flush=True)


if __name__ == "__main__":
    main()
