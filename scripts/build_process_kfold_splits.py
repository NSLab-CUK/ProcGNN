#!/usr/bin/env python3
"""Build per-process K-fold row manifests (train/val/test) from merged main CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _process_label(pid: int) -> str:
    return f"Process{int(pid)}"


def _build_manifest_rows(
    df_proc: pd.DataFrame,
    *,
    process_id_col: str,
    sample_id_col: str,
    row_indices: list[int],
) -> pd.DataFrame:
    rows_out: list[dict] = []
    for ri in row_indices:
        row = df_proc.loc[ri]
        pid = row[process_id_col]
        sid = row[sample_id_col]
        rows_out.append(
            {
                "process_id": str(pid),
                "sample_id": sid,
                "merged_row_index": int(ri),
            }
        )
    return pd.DataFrame(rows_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-process K-fold split manifests.")
    parser.add_argument(
        "--merged-csv",
        type=str,
        default="data/datasets_v3/process_main_merged.csv",
        help="Merged main table (must include process_id, ID, original row index = default RangeIndex).",
    )
    parser.add_argument(
        "--process-ids",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="Process ids 1..10 (default: all).",
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/splits/process_kfold",
        help="Root directory for Process*/fold_*/train|val|test.csv",
    )
    args = parser.parse_args()

    merged_path = (PROJECT_ROOT / args.merged_csv).resolve()
    if not merged_path.is_file():
        raise FileNotFoundError(f"merged csv not found: {merged_path}")

    df = pd.read_csv(merged_path)
    process_id_col = "process_id"
    sample_id_col = "ID"
    if process_id_col not in df.columns or sample_id_col not in df.columns:
        raise ValueError(f"{merged_path} must contain {process_id_col!r} and {sample_id_col!r}.")

    k_folds = max(2, int(args.k_folds))
    rng = np.random.RandomState(int(args.seed))
    out_root = (PROJECT_ROOT / args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    for pid in args.process_ids:
        plabel = _process_label(pid)
        df_proc = df[df[process_id_col].astype(str) == plabel].copy()
        if df_proc.empty:
            raise RuntimeError(f"No rows for {plabel} in {merged_path}")

        id_to_rows: dict[int, list[int]] = {}
        for sid, grp in df_proc.groupby(sample_id_col, sort=False):
            id_to_rows[int(sid)] = [int(i) for i in grp.index.tolist()]

        stable_ids = sorted(id_to_rows.keys())
        n_groups = len(stable_ids)
        eff_k = min(k_folds, n_groups)
        if eff_k < k_folds:
            print(f"[warn] {plabel}: only {n_groups} unique sample ids; using k_folds={eff_k} (requested {k_folds}).")

        perm = rng.permutation(n_groups)
        parts = np.array_split(perm, eff_k)

        for fold_1b in range(1, eff_k + 1):
            k = fold_1b - 1
            test_part = parts[k]
            val_part = parts[(k + 1) % eff_k]
            train_parts = [parts[j] for j in range(eff_k) if j not in (k, (k + 1) % eff_k)]
            train_flat = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)

            # parts[*] contain indices j into stable_ids (from permuted K-fold partition).
            def expand_perm_indices(pos_arr: np.ndarray) -> list[int]:
                if pos_arr.size == 0:
                    return []
                gids = [stable_ids[int(j)] for j in pos_arr.astype(int).tolist()]
                out: list[int] = []
                for g in gids:
                    out.extend(id_to_rows[g])
                return sorted(out)

            train_rows = expand_perm_indices(train_flat)
            val_rows = expand_perm_indices(val_part)
            test_rows = expand_perm_indices(test_part)

            st = set(train_rows)
            sv = set(val_rows)
            se = set(test_rows)
            train_val_ov = len(st & sv)
            train_test_ov = len(st & se)
            val_test_ov = len(sv & se)
            ok = train_val_ov == 0 and train_test_ov == 0 and val_test_ov == 0
            status = "ok" if ok else "overlap_error"

            fold_dir = out_root / plabel / f"fold_{fold_1b:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            _build_manifest_rows(df_proc, process_id_col=process_id_col, sample_id_col=sample_id_col, row_indices=train_rows).to_csv(
                fold_dir / "train.csv", index=False
            )
            _build_manifest_rows(df_proc, process_id_col=process_id_col, sample_id_col=sample_id_col, row_indices=val_rows).to_csv(
                fold_dir / "val.csv", index=False
            )
            _build_manifest_rows(df_proc, process_id_col=process_id_col, sample_id_col=sample_id_col, row_indices=test_rows).to_csv(
                fold_dir / "test.csv", index=False
            )

            summary_rows.append(
                {
                    "process_id": plabel,
                    "fold": fold_1b,
                    "n_train": len(train_rows),
                    "n_val": len(val_rows),
                    "n_test": len(test_rows),
                    "train_val_overlap": train_val_ov,
                    "train_test_overlap": train_test_ov,
                    "val_test_overlap": val_test_ov,
                    "status": status,
                }
            )

    summary_path = out_root / "split_summary.csv"
    pd.DataFrame(summary_rows).sort_values(["process_id", "fold"]).to_csv(summary_path, index=False)
    bad = [r for r in summary_rows if r["status"] != "ok"]
    print(f"[done] wrote manifests under {out_root}")
    print(f"[done] summary: {summary_path} ({len(summary_rows)} rows)")
    if bad:
        raise SystemExit(f"[fail] non-ok splits: {bad[:5]}{'...' if len(bad) > 5 else ''}")


if __name__ == "__main__":
    main()
