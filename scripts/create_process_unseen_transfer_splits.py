#!/usr/bin/env python3
"""Create manifests for 5-fold unseen-process zero-shot/transfer experiments.

Default fold definition is process-level, not sample-level:

    fold_01 held-out: P01, P02
    fold_02 held-out: P03, P04
    fold_03 held-out: P05, P06
    fold_04 held-out: P07, P08
    fold_05 held-out: P09, P10

For each fold, the source set is the other eight processes. The two held-out
processes are treated as one target process set for zero-shot, transfer, and
scratch evaluation.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = ("process_id", "sample_id", "merged_row_index")
PROCESS_FOLDS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]


def _process_label(pid: int) -> str:
    return f"Process{int(pid)}"


def _heldout_group_label(pids: Iterable[int]) -> str:
    return "heldout_" + "_".join(f"P{int(pid):02d}" for pid in sorted(int(p) for p in pids))


def _fold_label(fold: int) -> str:
    return f"fold_{int(fold):02d}"


def _ratio_label(ratio: float) -> str:
    return f"ratio_{int(round(float(ratio) * 100.0)):02d}"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _as_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


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


def _sample_key(process_id: object, sample_id: object) -> str:
    pid_text = str(process_id).strip()
    if pid_text and not pid_text.startswith("Process"):
        try:
            pid_text = f"Process{int(float(pid_text))}"
        except (TypeError, ValueError):
            pass
    sid_text = str(sample_id).strip()
    try:
        sid_num = float(sid_text)
        if sid_num.is_integer():
            sid_text = str(int(sid_num))
    except (TypeError, ValueError):
        pass
    return f"{pid_text}::{sid_text}"


def _load_merged_row_index_map(merged_csv: Path) -> tuple[dict[str, int] | None, pd.DataFrame]:
    if not merged_csv.is_file():
        raise FileNotFoundError(f"merged CSV not found: {merged_csv}")
    frame = pd.read_csv(merged_csv, usecols=lambda c: c in {"process_id", "ID"})
    missing = [c for c in ("process_id", "ID") if c not in frame.columns]
    if missing:
        raise ValueError(f"{merged_csv} missing columns required for row remap: {missing}")
    mapping: dict[str, int] = {}
    duplicates: list[str] = []
    for idx, row in frame.iterrows():
        key = _sample_key(row["process_id"], row["ID"])
        if key in mapping:
            duplicates.append(key)
            continue
        mapping[key] = int(idx)
    if duplicates:
        print(
            "[unseen-transfer-splits][row-index] duplicate process_id/sample_id keys in merged CSV; "
            "validating existing manifest merged_row_index instead of remapping. "
            f"first_duplicates={duplicates[:5]}",
            flush=True,
        )
        return None, frame.reset_index(drop=True)
    return mapping, frame.reset_index(drop=True)


def _remap_merged_row_indices(
    df: pd.DataFrame,
    *,
    merged_index: dict[str, int] | None,
    merged_frame: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    out = df.copy()
    original = pd.to_numeric(out["merged_row_index"], errors="raise").astype(int)
    if merged_index is None:
        bad: list[tuple[int, object, object, object, object]] = []
        if int(original.min()) < 0 or int(original.max()) >= len(merged_frame):
            raise RuntimeError(f"{label}: manifest merged_row_index out of range for merged CSV.")
        for row in out.itertuples(index=False):
            idx = int(getattr(row, "merged_row_index"))
            merged_row = merged_frame.iloc[idx]
            expected = _sample_key(getattr(row, "process_id"), getattr(row, "sample_id"))
            found = _sample_key(merged_row["process_id"], merged_row["ID"])
            if expected != found:
                bad.append((idx, getattr(row, "process_id"), getattr(row, "sample_id"), merged_row["process_id"], merged_row["ID"]))
                if len(bad) >= 5:
                    break
        if bad:
            raise RuntimeError(
                f"{label}: manifest merged_row_index does not match merged CSV process_id/sample_id. "
                f"examples={bad}"
            )
        return out
    mapped: list[int] = []
    missing: list[str] = []
    for row in out.itertuples(index=False):
        key = _sample_key(getattr(row, "process_id"), getattr(row, "sample_id"))
        if key not in merged_index:
            missing.append(key)
            continue
        mapped.append(int(merged_index[key]))
    if missing:
        raise RuntimeError(f"{label}: sample keys not found in merged CSV: {missing[:10]}")
    out["source_manifest_row_index"] = original
    out["merged_row_index"] = mapped
    return out.loc[:, [c for c in out.columns if c != "source_manifest_row_index"] + ["source_manifest_row_index"]]


def _concat(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [_read_manifest(path) for path in paths]
    cols = list(frames[0].columns)
    return pd.concat(frames, ignore_index=True).loc[:, cols]


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _rows(df: pd.DataFrame) -> set[int]:
    return {int(x) for x in pd.to_numeric(df["merged_row_index"], errors="raise").astype(int).tolist()}


def _sample_keys(df: pd.DataFrame) -> list[str]:
    cols = df.loc[:, ["process_id", "sample_id"]].drop_duplicates()
    return [_sample_key(row.process_id, row.sample_id) for row in cols.itertuples(index=False)]


def _subset_by_keys(df: pd.DataFrame, keys: set[str]) -> pd.DataFrame:
    tmp = df.copy()
    key = tmp["process_id"].astype(str) + "::" + tmp["sample_id"].astype(str)
    return tmp[key.isin(keys)].copy()


def _validate_process_set(df: pd.DataFrame, *, allowed: set[str], label: str) -> None:
    found = set(df["process_id"].astype(str).unique().tolist())
    bad = sorted(found - allowed)
    if bad:
        raise RuntimeError(f"{label}: unexpected process_id values {bad}; allowed={sorted(allowed)}")


def _validate_disjoint(label: str, *frames: pd.DataFrame) -> None:
    sets = [_rows(df) for df in frames]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise RuntimeError(f"{label}: row overlap detected ({len(overlap)}), first={sorted(overlap)[:5]}")


def _split_adaptation_keys(
    keys: list[str],
    *,
    val_ratio: float,
    rng: np.random.RandomState,
) -> tuple[list[str], list[str]]:
    if not keys:
        return [], []
    perm = list(rng.permutation(sorted(keys)))
    n_val = max(1, int(round(len(perm) * float(val_ratio)))) if len(perm) >= 3 and val_ratio > 0 else 0
    n_val = min(n_val, max(len(perm) - 1, 0))
    return sorted(perm[n_val:]), sorted(perm[:n_val])


def _validate_ratio_nested(rows: list[dict[str, object]]) -> None:
    previous_train: set[str] = set()
    previous_val: set[str] = set()
    for row in rows:
        train_keys = set(row["train_keys"])
        val_keys = set(row["val_keys"])
        ratio = row["ratio"]
        if not previous_train.issubset(train_keys):
            raise RuntimeError(f"ratio train nested invariant failed at ratio={ratio}")
        if not previous_val.issubset(val_keys):
            raise RuntimeError(f"ratio val nested invariant failed at ratio={ratio}")
        previous_train = train_keys
        previous_val = val_keys


def _fold_specs(
    folds: list[int],
    *,
    heldout_processes: list[int],
    heldout_mode: str,
) -> list[tuple[int, list[int]]]:
    specs: list[tuple[int, list[int]]] = []
    if heldout_mode == "process_pair":
        for fold in folds:
            if fold < 1 or fold > len(PROCESS_FOLDS):
                raise ValueError("--folds must be in 1..5 for process-pair unseen experiments.")
            specs.append((fold, PROCESS_FOLDS[fold - 1]))
        return specs
    if heldout_mode == "single_process":
        for pid in heldout_processes:
            if int(pid) < 1 or int(pid) > 10:
                raise ValueError("--heldout-processes must be in 1..10 for single-process unseen experiments.")
            for fold in folds:
                if fold < 1:
                    raise ValueError("--folds must be positive sample-fold numbers.")
                specs.append((fold, [int(pid)]))
        return specs
    raise ValueError(f"unknown heldout mode: {heldout_mode}")
    return specs


def _process_ratio_keys(
    *,
    target_pool: pd.DataFrame,
    heldout_ids: list[int],
    fold: int,
    ratio: float,
    seed: int,
    adaptation_val_ratio: float,
) -> tuple[set[str], set[str], dict[str, dict[str, int]]]:
    train_keys: set[str] = set()
    val_keys: set[str] = set()
    per_process_counts: dict[str, dict[str, int]] = {}
    for pid in heldout_ids:
        proc_pool = target_pool[target_pool["process_id"].astype(str) == _process_label(pid)]
        pool_keys = _sample_keys(proc_pool)
        train_pool_keys, val_pool_keys = _split_adaptation_keys(
            pool_keys,
            val_ratio=adaptation_val_ratio,
            rng=np.random.RandomState(int(seed) + pid * 1000 + fold),
        )
        train_perm = list(np.random.RandomState(int(seed) + pid * 1000 + fold * 17 + 1).permutation(train_pool_keys))
        val_perm = list(np.random.RandomState(int(seed) + pid * 1000 + fold * 17 + 2).permutation(val_pool_keys))
        n_train = max(1, int(round(len(train_perm) * float(ratio))))
        n_val = max(1, int(round(len(val_perm) * float(ratio)))) if val_perm else 0
        proc_train_keys = {str(k) for k in train_perm[:n_train]}
        proc_val_keys = {str(k) for k in val_perm[:n_val]}
        train_keys.update(proc_train_keys)
        val_keys.update(proc_val_keys)
        per_process_counts[_process_label(pid)] = {
            "train_sample_count": int(len(proc_train_keys)),
            "val_sample_count": int(len(proc_val_keys)),
        }
    return train_keys, val_keys, per_process_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Create unseen zero-shot/transfer split manifests.")
    parser.add_argument("--heldout-processes", type=int, nargs="+", default=list(range(1, 11)), help="Held-out process ids for --heldout-mode single_process; kept for compatibility in process_pair mode.")
    parser.add_argument(
        "--heldout-mode",
        choices=("process_pair", "single_process"),
        default="process_pair",
        help="process_pair uses fixed held-out pairs [[1,2],...]; single_process creates one held-out process per requested process and sample fold.",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20, 0.50])
    parser.add_argument("--input-split-root", default="data/splits/all_processes_10pct")
    parser.add_argument(
        "--sample-fold",
        type=int,
        default=1,
        help=(
            "Per-process sample split fold to read from input-split-root. "
            "In process_pair mode this selects the per-process sample split fold. "
            "In single_process mode, each --folds value is used as the sample split fold."
        ),
    )
    parser.add_argument("--merged-csv", default="data/datasets_v3/process_main_merged_all_processes_10pct.csv")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--adaptation-val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    input_root = _resolve(args.input_split_root)
    merged_csv = _resolve(args.merged_csv)
    output_root = _resolve(args.output_root)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite and not args.validate_only:
        raise FileExistsError(f"output root exists and is not empty: {output_root} (use --overwrite)")
    if not args.validate_only:
        output_root.mkdir(parents=True, exist_ok=True)

    ratios = sorted(float(r) for r in args.ratios)
    all_processes = set(range(1, 11))
    global_rows: list[dict[str, object]] = []
    merged_index, merged_frame = _load_merged_row_index_map(merged_csv)

    for fold, heldout_ids in _fold_specs(
        [int(f) for f in args.folds],
        heldout_processes=[int(p) for p in args.heldout_processes],
        heldout_mode=str(args.heldout_mode),
    ):
        heldout_set = set(heldout_ids)
        source_ids = sorted(all_processes - heldout_set)
        source_labels = {_process_label(pid) for pid in source_ids}
        heldout_labels = {_process_label(pid) for pid in heldout_ids}
        group_label = _heldout_group_label(heldout_ids)
        flabel = _fold_label(fold)
        input_flabel = _fold_label(fold if str(args.heldout_mode) == "single_process" else int(args.sample_fold))
        fold_dir = output_root / group_label / flabel

        source_train = _concat(input_root / _process_label(pid) / input_flabel / "train.csv" for pid in source_ids)
        source_val = _concat(input_root / _process_label(pid) / input_flabel / "val.csv" for pid in source_ids)
        target_train = _concat(input_root / _process_label(pid) / input_flabel / "train.csv" for pid in heldout_ids)
        target_val = _concat(input_root / _process_label(pid) / input_flabel / "val.csv" for pid in heldout_ids)
        target_test = _concat(input_root / _process_label(pid) / input_flabel / "test.csv" for pid in heldout_ids)
        source_train = _remap_merged_row_indices(source_train, merged_index=merged_index, merged_frame=merged_frame, label=f"{group_label}/{flabel}/source_train")
        source_val = _remap_merged_row_indices(source_val, merged_index=merged_index, merged_frame=merged_frame, label=f"{group_label}/{flabel}/source_val")
        target_train = _remap_merged_row_indices(target_train, merged_index=merged_index, merged_frame=merged_frame, label=f"{group_label}/{flabel}/target_train")
        target_val = _remap_merged_row_indices(target_val, merged_index=merged_index, merged_frame=merged_frame, label=f"{group_label}/{flabel}/target_val")
        target_test = _remap_merged_row_indices(target_test, merged_index=merged_index, merged_frame=merged_frame, label=f"{group_label}/{flabel}/target_test")
        target_pool = pd.concat((target_train, target_val), ignore_index=True).loc[:, target_train.columns]

        _validate_process_set(source_train, allowed=source_labels, label=f"{group_label}/{flabel}/source_train")
        _validate_process_set(source_val, allowed=source_labels, label=f"{group_label}/{flabel}/source_val")
        _validate_process_set(target_pool, allowed=heldout_labels, label=f"{group_label}/{flabel}/target_pool")
        _validate_process_set(target_test, allowed=heldout_labels, label=f"{group_label}/{flabel}/target_test")
        _validate_disjoint(f"{group_label}/{flabel}/base", source_train, source_val, target_pool, target_test)
        if set(source_train["process_id"].astype(str).unique()) != source_labels:
            raise RuntimeError(f"{group_label}/{flabel}: source_train missing source process")
        if set(source_val["process_id"].astype(str).unique()) != source_labels:
            raise RuntimeError(f"{group_label}/{flabel}: source_val missing source process")

        ratio_meta: list[dict[str, object]] = []
        for ratio in ratios:
            train_keys, val_keys, per_process_counts = _process_ratio_keys(
                target_pool=target_pool,
                heldout_ids=heldout_ids,
                fold=fold,
                ratio=ratio,
                seed=int(args.seed),
                adaptation_val_ratio=float(args.adaptation_val_ratio),
            )
            ratio_train = _subset_by_keys(target_pool, train_keys)
            ratio_val = _subset_by_keys(target_pool, val_keys)
            if ratio_train.empty or ratio_val.empty:
                raise RuntimeError(f"{group_label}/{flabel}/{_ratio_label(ratio)} produced empty train/val")
            _validate_disjoint(f"{group_label}/{flabel}/{_ratio_label(ratio)}", ratio_train, ratio_val, target_test)
            if not args.validate_only:
                rdir = fold_dir / _ratio_label(ratio)
                _write(ratio_train, rdir / "train.csv")
                _write(ratio_val, rdir / "val.csv")
            ratio_payload = {
                "ratio": ratio,
                "ratio_label": _ratio_label(ratio),
                "train_rows": int(len(ratio_train)),
                "val_rows": int(len(ratio_val)),
                "train_sample_count": int(len(train_keys)),
                "val_sample_count": int(len(val_keys)),
                "per_heldout_process_counts": per_process_counts,
                "train_keys": sorted(train_keys),
                "val_keys": sorted(val_keys),
            }
            if not args.validate_only:
                (fold_dir / _ratio_label(ratio) / "metadata.json").write_text(
                    json.dumps({k: v for k, v in ratio_payload.items() if k not in {"train_keys", "val_keys"}}, indent=2),
                    encoding="utf-8",
                )
            ratio_meta.append(ratio_payload)
        _validate_ratio_nested(ratio_meta)

        if not args.validate_only:
            _write(source_train, fold_dir / "source_train.csv")
            _write(source_val, fold_dir / "source_val.csv")
            _write(target_pool, fold_dir / "target_adaptation_pool.csv")
            _write(target_test, fold_dir / "target_test.csv")
        fold_meta = {
            "heldout_processes": heldout_ids,
            "source_process_ids": source_ids,
            "fold": fold,
            "sample_split_fold": int(args.sample_fold),
            "heldout_mode": str(args.heldout_mode),
            "process_fold_definition": PROCESS_FOLDS,
            "adaptation_policy": "each_heldout_process_train_plus_val_then_same_ratio_per_process",
            "adaptation_val_ratio": float(args.adaptation_val_ratio),
            "source_train_rows": int(len(source_train)),
            "source_val_rows": int(len(source_val)),
            "target_adaptation_pool_rows": int(len(target_pool)),
            "target_test_rows": int(len(target_test)),
            "ratio_summaries": [{k: v for k, v in row.items() if k not in {"train_keys", "val_keys"}} for row in ratio_meta],
            "leakage_check": "pass",
        }
        if not args.validate_only:
            (fold_dir / "metadata.json").write_text(json.dumps(fold_meta, indent=2), encoding="utf-8")
            heldout_dir = output_root / group_label
            heldout_dir.mkdir(parents=True, exist_ok=True)
            (heldout_dir / "metadata.json").write_text(json.dumps(fold_meta, indent=2), encoding="utf-8")
            pd.DataFrame([fold_meta]).drop(columns=["ratio_summaries"], errors="ignore").to_csv(heldout_dir / "fold_summary.csv", index=False)
        global_rows.append(
            {
                "fold": fold,
                "heldout_processes": ",".join(str(p) for p in heldout_ids),
                "source_process_ids": ",".join(str(p) for p in source_ids),
                "source_train_rows": int(len(source_train)),
                "source_val_rows": int(len(source_val)),
                "target_adaptation_pool_rows": int(len(target_pool)),
                "target_test_rows": int(len(target_test)),
                "leakage_check": "pass",
            }
        )
        print(
            f"[unseen-transfer-splits] {group_label} {flabel} "
            f"source_train={len(source_train)} source_val={len(source_val)} "
            f"target_pool={len(target_pool)} target_test={len(target_test)}"
        )

    root_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "process_fold_definition": PROCESS_FOLDS,
        "heldout_mode": str(args.heldout_mode),
        "folds": [int(f) for f in args.folds],
        "sample_split_fold": int(args.sample_fold),
        "ratios": ratios,
        "input_split_root": _as_rel(input_root),
        "merged_csv": _as_rel(merged_csv),
        "output_root": _as_rel(output_root),
        "adaptation_val_ratio": float(args.adaptation_val_ratio),
        "seed": int(args.seed),
        "validate_only": bool(args.validate_only),
    }
    if not args.validate_only:
        (output_root / "metadata.json").write_text(json.dumps(root_meta, indent=2), encoding="utf-8")
        pd.DataFrame(global_rows).to_csv(output_root / "global_split_summary.csv", index=False)
    print(f"[unseen-transfer-splits] completed heldout_mode={args.heldout_mode} folds={args.folds}")


if __name__ == "__main__":
    main()
