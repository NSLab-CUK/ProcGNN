#!/usr/bin/env python3
"""Merge independently measured computational-efficiency fold shards."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_computational_efficiency_benchmark import _aggregate, _write_csv


def _read_csv(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing shard result: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-roots", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--expected-models", nargs="+", default=["gcn", "gin", "gat", "proposed"]
    )
    parser.add_argument("--expected-folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    hardware: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for raw_root in args.input_roots:
        root = Path(raw_root).resolve()
        shard_rows = _read_csv(root / "computational_efficiency_raw.csv")
        metadata_path = root / "hardware_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        hardware.append({"root": str(root), **metadata})
        for row in shard_rows:
            key = (str(row["model"]), int(row["fold"]))
            if key in seen:
                raise ValueError(f"duplicate model/fold result: {key}")
            seen.add(key)
            rows.append(row)

    expected = {
        (model, fold) for model in args.expected_models for fold in args.expected_folds
    }
    missing = sorted(expected - seen)
    unexpected = sorted(seen - expected)
    if missing or unexpected:
        raise RuntimeError(f"incomplete shard set: missing={missing} unexpected={unexpected}")

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (str(row["model"]), int(row["fold"])))
    _write_csv(output_root / "computational_efficiency_raw.csv", rows)
    _aggregate(rows, {"shards": hardware}, output_root)
    print(f"[efficiency-shards] merged rows={len(rows)} output={output_root}")


if __name__ == "__main__":
    main()
