#!/usr/bin/env python3
"""Create manifests for one held-out-process transfer-learning experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _process_label(pid: int) -> str:
    return f"Process{int(pid)}"


def _ratio_label(ratio: float) -> str:
    pct = int(round(float(ratio) * 100.0))
    return f"ratio_{pct:02d}"


def _id_key(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        fv = float(text)
        if fv.is_integer():
            return str(int(fv))
    except (TypeError, ValueError):
        pass
    return text


def _manifest(df: pd.DataFrame, indices: Iterable[int], *, process_id_col: str, sample_id_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx in sorted(int(i) for i in indices):
        row = df.loc[idx]
        rows.append(
            {
                "process_id": str(row[process_id_col]),
                "sample_id": row[sample_id_col],
                "merged_row_index": int(idx),
            }
        )
    return pd.DataFrame(rows)


def _group_indices_by_sample(
    df: pd.DataFrame,
    *,
    sample_id_col: str,
    process_id_col: str | None = None,
) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    group_cols = [sample_id_col] if process_id_col is None else [process_id_col, sample_id_col]
    for group_key, grp in df.groupby(group_cols, sort=False):
        if process_id_col is None:
            sid = group_key
            key = _id_key(sid)
        else:
            pid, sid = group_key
            pid_key = str(pid).strip()
            sample_key = _id_key(sid)
            key = f"{pid_key}::{sample_key}" if pid_key and sample_key else ""
        if not key:
            continue
        out[key] = [int(i) for i in grp.index.tolist()]
    return out


def _key_process_label(key: str) -> str:
    text = str(key)
    return text.split("::", 1)[0] if "::" in text else ""


def _expand(groups: dict[str, list[int]], keys: Iterable[str]) -> list[int]:
    rows: list[int] = []
    for key in keys:
        rows.extend(groups[str(key)])
    return sorted(rows)


def _split_keys(keys: list[str], *, val_ratio: float, test_ratio: float, rng: np.random.RandomState) -> tuple[list[str], list[str], list[str]]:
    if not keys:
        return [], [], []
    perm = list(rng.permutation(keys))
    n = len(perm)
    n_test = max(1, int(round(n * float(test_ratio)))) if test_ratio > 0.0 and n >= 3 else 0
    n_test = min(n_test, max(n - 2, 0))
    remaining = perm[: n - n_test]
    test = perm[n - n_test :] if n_test else []
    n_val = max(1, int(round(len(remaining) * float(val_ratio)))) if val_ratio > 0.0 and len(remaining) >= 3 else 0
    n_val = min(n_val, max(len(remaining) - 1, 0))
    val = remaining[:n_val]
    train = remaining[n_val:]
    return sorted(train), sorted(val), sorted(test)


def _split_train_val(keys: list[str], *, val_ratio: float, rng: np.random.RandomState) -> tuple[list[str], list[str]]:
    if not keys:
        return [], []
    perm = list(rng.permutation(keys))
    n_val = max(1, int(round(len(perm) * float(val_ratio)))) if val_ratio > 0.0 and len(perm) >= 3 else 0
    n_val = min(n_val, max(len(perm) - 1, 0))
    return sorted(perm[n_val:]), sorted(perm[:n_val])


def _validate_disjoint(name: str, *sets: set[int]) -> None:
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            if overlap:
                raise RuntimeError(f"{name}: split overlap detected ({len(overlap)} rows), first={sorted(overlap)[:5]}")


def _make_key_folds(keys: list[str], *, k_folds: int, rng: np.random.RandomState) -> list[list[str]]:
    if k_folds <= 1:
        return [sorted(keys)]
    if len(keys) < k_folds:
        raise ValueError(f"k_folds={k_folds} requires at least {k_folds} sample IDs, got {len(keys)}")
    perm = list(rng.permutation(sorted(keys)))
    return [sorted(str(x) for x in fold.tolist()) for fold in np.array_split(np.asarray(perm, dtype=object), k_folds)]


def _make_process_stratified_key_folds(
    keys: list[str],
    *,
    k_folds: int,
    rng: np.random.RandomState,
) -> list[list[str]]:
    if k_folds <= 1:
        return [sorted(keys)]
    by_process: dict[str, list[str]] = {}
    for key in sorted(keys):
        proc = _key_process_label(key)
        if not proc:
            raise ValueError(
                "process-stratified folds require process-qualified sample keys; "
                f"got {key!r}"
            )
        by_process.setdefault(proc, []).append(key)
    folds: list[list[str]] = [[] for _ in range(k_folds)]
    for proc, proc_keys in sorted(by_process.items()):
        if len(proc_keys) < k_folds:
            raise ValueError(
                f"k_folds={k_folds} requires at least {k_folds} sample IDs for {proc}, "
                f"got {len(proc_keys)}"
            )
        perm = list(rng.permutation(sorted(proc_keys)))
        for fold_idx, part in enumerate(np.array_split(np.asarray(perm, dtype=object), k_folds)):
            folds[fold_idx].extend(str(x) for x in part.tolist())
    return [sorted(fold) for fold in folds]


def _process_sample_counts(keys: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in keys:
        proc = _key_process_label(str(key))
        if not proc:
            continue
        counts[proc] = counts.get(proc, 0) + 1
    return dict(sorted(counts.items()))


def _write_split_dir(
    *,
    out_root: Path,
    df: pd.DataFrame,
    process_id_col: str,
    sample_id_col: str,
    source_groups: dict[str, list[int]],
    target_groups: dict[str, list[int]],
    source_train_keys: list[str],
    source_val_keys: list[str],
    adaptation_pool_keys: list[str],
    target_test_keys: list[str],
    ratios: list[float],
    target_val_ratio: float,
    rng: np.random.RandomState,
    metadata_base: dict[str, object],
    fold_index: int | None = None,
    fold_count: int | None = None,
) -> dict[str, object]:
    out_root.mkdir(parents=True, exist_ok=True)

    source_train_rows = _expand(source_groups, source_train_keys)
    source_val_rows = _expand(source_groups, source_val_keys)
    target_pool_rows = _expand(target_groups, adaptation_pool_keys)
    target_test_rows = _expand(target_groups, target_test_keys)
    _validate_disjoint("source", set(source_train_rows), set(source_val_rows))
    _validate_disjoint("target_pool_test", set(target_pool_rows), set(target_test_rows))

    _manifest(df, source_train_rows, process_id_col=process_id_col, sample_id_col=sample_id_col).to_csv(
        out_root / "source_train.csv", index=False
    )
    _manifest(df, source_val_rows, process_id_col=process_id_col, sample_id_col=sample_id_col).to_csv(
        out_root / "source_val.csv", index=False
    )
    _manifest(df, target_pool_rows, process_id_col=process_id_col, sample_id_col=sample_id_col).to_csv(
        out_root / "target_adaptation_pool.csv", index=False
    )
    _manifest(df, target_test_rows, process_id_col=process_id_col, sample_id_col=sample_id_col).to_csv(
        out_root / "target_test.csv", index=False
    )

    previous_subset: set[str] = set()
    ratio_rows: list[dict[str, object]] = []
    sorted_pool = list(rng.permutation(sorted(adaptation_pool_keys)))
    for ratio in sorted(float(r) for r in ratios):
        if ratio <= 0.0 or ratio > 1.0:
            raise ValueError(f"ratios must be in (0, 1], got {ratio}")
        n_subset = max(1, int(round(len(sorted_pool) * ratio)))
        subset = set(sorted_pool[:n_subset])
        if not previous_subset.issubset(subset):
            raise RuntimeError(f"nested subset invariant failed for ratio={ratio}")
        previous_subset = set(subset)
        subset_train_keys, subset_val_keys = _split_train_val(sorted(subset), val_ratio=float(target_val_ratio), rng=rng)
        train_rows = _expand(target_groups, subset_train_keys)
        val_rows = _expand(target_groups, subset_val_keys)
        _validate_disjoint(_ratio_label(ratio), set(train_rows), set(val_rows), set(target_test_rows))
        rdir = out_root / _ratio_label(ratio)
        rdir.mkdir(parents=True, exist_ok=True)
        _manifest(df, train_rows, process_id_col=process_id_col, sample_id_col=sample_id_col).to_csv(rdir / "train.csv", index=False)
        _manifest(df, val_rows, process_id_col=process_id_col, sample_id_col=sample_id_col).to_csv(rdir / "val.csv", index=False)
        ratio_rows.append(
            {
                "fold": fold_index,
                "ratio": ratio,
                "ratio_label": _ratio_label(ratio),
                "target_subset_sample_count": len(subset),
                "train_sample_count": len(subset_train_keys),
                "val_sample_count": len(subset_val_keys),
                "train_row_count": len(train_rows),
                "val_row_count": len(val_rows),
            }
        )

    metadata = {
        **metadata_base,
        "fold": fold_index,
        "k_folds": fold_count or 1,
        "source_train_rows": len(source_train_rows),
        "source_val_rows": len(source_val_rows),
        "target_adaptation_pool_rows": len(target_pool_rows),
        "target_test_rows": len(target_test_rows),
        "source_train_sample_count": len(source_train_keys),
        "source_val_sample_count": len(source_val_keys),
        "target_adaptation_pool_sample_count": len(adaptation_pool_keys),
        "target_test_sample_count": len(target_test_keys),
        "source_train_sample_count_by_process": _process_sample_counts(source_train_keys),
        "source_val_sample_count_by_process": _process_sample_counts(source_val_keys),
        "target_adaptation_pool_sample_count_by_process": _process_sample_counts(adaptation_pool_keys),
        "target_test_sample_count_by_process": _process_sample_counts(target_test_keys),
        "ratio_summaries": ratio_rows,
    }
    (out_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(ratio_rows).to_csv(out_root / "ratio_summary.csv", index=False)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Create held-out process transfer split manifests.")
    parser.add_argument("--merged-csv", type=str, default="data/datasets_v3/process_main_merged_all_processes_10pct.csv")
    parser.add_argument("--heldout-process", type=int, required=True)
    parser.add_argument("--source-process-ids", type=int, nargs="+", default=None)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20, 0.50])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-test-ratio", type=float, default=0.20)
    parser.add_argument("--target-val-ratio", type=float, default=0.20)
    parser.add_argument("--k-folds", type=int, default=1, help="Number of source/held-out transfer folds to create. 1 keeps the legacy layout.")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    merged_path = (PROJECT_ROOT / args.merged_csv).resolve()
    if not merged_path.is_file():
        raise FileNotFoundError(f"merged csv not found: {merged_path}")
    out_root = (PROJECT_ROOT / args.output_root).resolve()
    if out_root.exists() and any(out_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output root already exists and is not empty: {out_root} (use --overwrite)")
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(merged_path)
    process_id_col = "process_id"
    sample_id_col = "ID"
    for col in (process_id_col, sample_id_col):
        if col not in df.columns:
            raise ValueError(f"{merged_path} must contain {col!r}.")

    heldout_label = _process_label(args.heldout_process)
    source_ids = list(args.source_process_ids or [p for p in range(1, 11) if p != int(args.heldout_process)])
    source_labels = {_process_label(pid) for pid in source_ids}
    if heldout_label in source_labels:
        raise ValueError("heldout process cannot also be a source process")

    rng = np.random.RandomState(int(args.seed))
    proc_text = df[process_id_col].astype(str)
    target_df = df[proc_text == heldout_label]
    source_df = df[proc_text.isin(source_labels)]
    if target_df.empty:
        raise RuntimeError(f"no rows found for heldout process {heldout_label}")
    if source_df.empty:
        raise RuntimeError(f"no rows found for source processes {sorted(source_labels)}")

    target_groups = _group_indices_by_sample(
        target_df,
        sample_id_col=sample_id_col,
        process_id_col=process_id_col,
    )
    source_groups = _group_indices_by_sample(
        source_df,
        sample_id_col=sample_id_col,
        process_id_col=process_id_col,
    )
    ratios = [float(r) for r in sorted(float(r) for r in args.ratios)]
    metadata_base = {
        "merged_csv": str(Path(args.merged_csv).as_posix()),
        "heldout_process": int(args.heldout_process),
        "source_process_ids": [int(x) for x in source_ids],
        "ratios": ratios,
        "seed": int(args.seed),
        "target_test_ratio": float(args.target_test_ratio),
        "target_val_ratio": float(args.target_val_ratio),
    }

    if int(args.k_folds) > 1:
        k_folds = int(args.k_folds)
        source_folds = _make_process_stratified_key_folds(
            sorted(source_groups),
            k_folds=k_folds,
            rng=np.random.RandomState(int(args.seed) + 101),
        )
        target_folds = _make_key_folds(sorted(target_groups), k_folds=k_folds, rng=np.random.RandomState(int(args.seed) + 202))
        fold_summaries: list[dict[str, object]] = []
        for fold_idx in range(k_folds):
            fold_number = fold_idx + 1
            source_val_keys = source_folds[fold_idx]
            source_train_keys = sorted(set(source_groups) - set(source_val_keys))
            target_test_keys = target_folds[fold_idx]
            adaptation_pool_keys = sorted(set(target_groups) - set(target_test_keys))
            fold_dir = out_root / f"fold_{fold_number:02d}"
            metadata = _write_split_dir(
                out_root=fold_dir,
                df=df,
                process_id_col=process_id_col,
                sample_id_col=sample_id_col,
                source_groups=source_groups,
                target_groups=target_groups,
                source_train_keys=source_train_keys,
                source_val_keys=source_val_keys,
                adaptation_pool_keys=adaptation_pool_keys,
                target_test_keys=target_test_keys,
                ratios=ratios,
                target_val_ratio=float(args.target_val_ratio),
                rng=np.random.RandomState(int(args.seed) + 1009 * fold_number),
                metadata_base=metadata_base,
                fold_index=fold_number,
                fold_count=k_folds,
            )
            fold_summaries.append(
                {
                    "fold": fold_number,
                    "split_dir": fold_dir.relative_to(out_root).as_posix(),
                    "source_train_rows": metadata["source_train_rows"],
                    "source_val_rows": metadata["source_val_rows"],
                    "target_adaptation_pool_rows": metadata["target_adaptation_pool_rows"],
                    "target_test_rows": metadata["target_test_rows"],
                    "source_val_sample_count_by_process": metadata["source_val_sample_count_by_process"],
                }
            )
            print(
                f"[transfer-splits][fold {fold_number:02d}] "
                f"source train/val rows={metadata['source_train_rows']}/{metadata['source_val_rows']} "
                f"target pool/test rows={metadata['target_adaptation_pool_rows']}/{metadata['target_test_rows']}"
            )
            print(
                f"[transfer-splits][fold {fold_number:02d}] "
                "source val samples by process="
                + json.dumps(metadata["source_val_sample_count_by_process"], sort_keys=True)
            )

        root_metadata = {
            **metadata_base,
            "k_folds": k_folds,
            "layout": "kfold",
            "fold_summaries": fold_summaries,
        }
        (out_root / "metadata.json").write_text(json.dumps(root_metadata, indent=2), encoding="utf-8")
        pd.DataFrame(fold_summaries).to_csv(out_root / "split_summary.csv", index=False)
        print(f"[transfer-splits] wrote {out_root}")
        print(f"[transfer-splits] k_folds={k_folds}")
        return

    source_train_keys, source_val_keys, _ = _split_keys(
        sorted(source_groups), val_ratio=float(args.target_val_ratio), test_ratio=0.0, rng=rng
    )
    target_pool_keys, target_val0_keys, target_test_keys = _split_keys(
        sorted(target_groups),
        val_ratio=float(args.target_val_ratio),
        test_ratio=float(args.target_test_ratio),
        rng=rng,
    )
    # Put the target validation reserve back into the adaptation pool; each ratio then gets its own train/val split.
    adaptation_pool_keys = sorted(set(target_pool_keys) | set(target_val0_keys))

    metadata = _write_split_dir(
        out_root=out_root,
        df=df,
        process_id_col=process_id_col,
        sample_id_col=sample_id_col,
        source_groups=source_groups,
        target_groups=target_groups,
        source_train_keys=source_train_keys,
        source_val_keys=source_val_keys,
        adaptation_pool_keys=adaptation_pool_keys,
        target_test_keys=target_test_keys,
        ratios=ratios,
        target_val_ratio=float(args.target_val_ratio),
        rng=rng,
        metadata_base={**metadata_base, "layout": "single"},
    )
    print(f"[transfer-splits] wrote {out_root}")
    print(
        f"[transfer-splits] source train/val rows={metadata['source_train_rows']}/{metadata['source_val_rows']} "
        f"target pool/test rows={metadata['target_adaptation_pool_rows']}/{metadata['target_test_rows']}"
    )


if __name__ == "__main__":
    main()
