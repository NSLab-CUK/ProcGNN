#!/usr/bin/env python3
"""Materialize a balanced fraction of every process from existing split manifests."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_NAMES = ("train", "val", "test")


def _id_key(value: object) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample the same fraction from each process and preserve train/val/test membership."
    )
    parser.add_argument("--merged-csv", default="data/datasets_v3/process_main_merged.csv")
    parser.add_argument("--source-splits-dir", default="data/splits/process_kfold")
    parser.add_argument("--source-fold", type=int, default=1)
    parser.add_argument("--process-ids", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-csv",
        default="data/datasets_v3/process_main_merged_all_processes_10pct.csv",
    )
    parser.add_argument(
        "--output-splits-dir",
        default="data/splits/all_processes_10pct",
    )
    args = parser.parse_args()

    if not 0.0 < float(args.fraction) <= 1.0:
        raise ValueError("--fraction must be in (0, 1].")

    merged_path = (PROJECT_ROOT / args.merged_csv).resolve()
    source_root = (PROJECT_ROOT / args.source_splits_dir).resolve()
    output_csv = (PROJECT_ROOT / args.output_csv).resolve()
    output_root = (PROJECT_ROOT / args.output_splits_dir).resolve()
    merged = pd.read_csv(merged_path)
    required = {"process_id", "ID"}
    if not required.issubset(merged.columns):
        raise ValueError(f"{merged_path} must contain {sorted(required)}")

    selected: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    seen_source_rows: set[int] = set()
    fold_name = f"fold_{int(args.source_fold):02d}"

    for process_id in args.process_ids:
        process_label = f"Process{int(process_id)}"
        for split_index, split_name in enumerate(SPLIT_NAMES):
            manifest_path = source_root / process_label / fold_name / f"{split_name}.csv"
            manifest = pd.read_csv(manifest_path)
            required_manifest = {"process_id", "sample_id", "merged_row_index"}
            if not required_manifest.issubset(manifest.columns):
                raise ValueError(f"{manifest_path} must contain {sorted(required_manifest)}")
            sample_count = max(1, int(round(len(manifest) * float(args.fraction))))
            sampled = manifest.sample(
                n=sample_count,
                random_state=int(args.seed) + int(process_id) * 100 + split_index,
                replace=False,
            ).sort_values("merged_row_index")
            for row in sampled.itertuples(index=False):
                source_index = int(row.merged_row_index)
                if source_index in seen_source_rows:
                    raise RuntimeError(f"source row {source_index} appears in more than one selected split")
                source_row = merged.iloc[source_index]
                if str(source_row["process_id"]) != process_label:
                    raise RuntimeError(
                        f"process mismatch at row {source_index}: {source_row['process_id']} != {process_label}"
                    )
                if _id_key(source_row["ID"]) != _id_key(row.sample_id):
                    raise RuntimeError(
                        f"sample ID mismatch at row {source_index}: {source_row['ID']} != {row.sample_id}"
                    )
                seen_source_rows.add(source_index)
                selected.append(
                    {
                        "process_id": process_label,
                        "sample_id": row.sample_id,
                        "source_merged_row_index": source_index,
                        "split": split_name,
                    }
                )
            summary.append(
                {
                    "process_id": process_label,
                    "split": split_name,
                    "source_count": len(manifest),
                    "selected_count": sample_count,
                    "fraction": sample_count / max(len(manifest), 1),
                }
            )

    selection = pd.DataFrame(selected)
    materialized = merged.iloc[selection["source_merged_row_index"].astype(int).tolist()].copy()
    materialized.reset_index(drop=True, inplace=True)
    materialized["split"] = selection["split"].tolist()
    selection["merged_row_index"] = materialized.index.astype(int)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    materialized.to_csv(output_csv, index=False)

    manifest_columns = [
        "process_id",
        "sample_id",
        "merged_row_index",
        "source_merged_row_index",
    ]
    all_fold = output_root / "All" / "fold_01"
    all_fold.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        split_all = selection[selection["split"] == split_name][manifest_columns].copy()
        split_all.to_csv(all_fold / f"{split_name}.csv", index=False)
        for process_id in args.process_ids:
            process_label = f"Process{int(process_id)}"
            process_fold = output_root / process_label / "fold_01"
            process_fold.mkdir(parents=True, exist_ok=True)
            split_all[split_all["process_id"] == process_label].to_csv(
                process_fold / f"{split_name}.csv", index=False
            )

    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(output_root / "sampling_summary.csv", index=False)
    print(f"[done] dataset: {output_csv} ({len(materialized)} rows)")
    print(f"[done] manifests: {output_root}")
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
