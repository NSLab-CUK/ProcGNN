from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from process_graph.data.stream_keys import (  # noqa: E402
    STREAM_KEY_CANONICALIZATION_VERSION,
    canonicalize_process_id,
    canonicalize_stream_key,
)


PROPERTY_COLUMNS = [
    "Temp",
    "Pres",
    "Vol_Flow",
    "Mole_Flow",
    "Mass_Flow",
    "Frac_H2O",
    "Frac_H2",
    "Frac_CH4",
    "Frac_CO2",
    "Frac_CO",
    "Frac_O2",
    "Frac_N2",
    "Enthalpy",
    "Density",
]
P03_RECOVERY_EDGES = {
    "P03_E004",
    "P03_E007",
    "P03_E008",
    "P03_E009",
    "P03_E012",
    "P03_E016",
    "P03_E018",
    "P03_E019",
    "P03_E020",
}


def _empty_frame(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=columns)


def _process_number(value: Any) -> int:
    normalized = canonicalize_process_id(value)
    if not normalized:
        raise ValueError(f"Missing process ID: {value!r}")
    return int(normalized[1:])


def _main_sample_ids(main_frame: pd.DataFrame, process_id: int) -> set[str]:
    process_keys = main_frame["process_id"].map(canonicalize_process_id)
    selected = main_frame.loc[
        process_keys.eq(canonicalize_process_id(process_id)),
        "ID",
    ]
    return {
        canonicalize_stream_key(value, process_id=process_id, source="main_csv:ID")
        for value in selected
        if canonicalize_stream_key(value, process_id=process_id, source="main_csv:ID")
    }


def audit_stream_key_joins(
    *,
    reference_dir: Path,
    stream_dir: Path,
    main_csv: Path,
    output_dir: Path,
    process_ids: list[int],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    edges = pd.read_csv(
        reference_dir / "canonical_edges.csv",
        dtype={"main_data_stream_key": str, "stream_name_norm": str},
    )
    answers = pd.read_csv(reference_dir / "target_answer_edges.csv")
    v4_path = PROJECT_ROOT / "data/reference/v4/target_stream_targets.csv"
    v4_targets = pd.read_csv(v4_path) if v4_path.is_file() else pd.DataFrame()
    main_frame = pd.read_csv(main_csv, usecols=["process_id", "ID"])

    process_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    collision_rows: list[dict[str, Any]] = []
    sample_failure_rows: list[dict[str, Any]] = []
    supervision_rows: list[dict[str, Any]] = []
    p03_property_coverage: dict[str, dict[str, Any]] = {}

    for process_id in process_ids:
        process_key = canonicalize_process_id(process_id)
        process_edges = edges[
            pd.to_numeric(edges["process_id"], errors="coerce").eq(process_id)
        ].copy()
        if process_edges.empty:
            raise ValueError(f"No canonical edges found for {process_key}.")
        process_edges["_RAW_KEY"] = process_edges["main_data_stream_key"].fillna("")
        process_edges["_CANONICAL_KEY"] = process_edges["_RAW_KEY"].map(
            lambda value: canonicalize_stream_key(
                value,
                process_id=process_id,
                source=reference_dir / "canonical_edges.csv",
            )
        )
        required_edges = process_edges[
            process_edges["_CANONICAL_KEY"].astype(bool)
        ].copy()
        optional_edges = process_edges[
            ~process_edges["_CANONICAL_KEY"].astype(bool)
        ].copy()

        if not v4_targets.empty and "canonical_edge_id" in v4_targets.columns:
            target_ids = set(
                v4_targets.loc[
                    pd.to_numeric(v4_targets["process_id"], errors="coerce").eq(
                        process_id
                    ),
                    "canonical_edge_id",
                ]
                .dropna()
                .astype(str)
            )
        else:
            target_ids = set(
                answers.loc[
                    pd.to_numeric(answers["process_id"], errors="coerce").eq(
                        process_id
                    ),
                    "canonical_answer_edge_id",
                ]
                .dropna()
                .astype(str)
            )

        stream_path = stream_dir / f"{process_id}.Process_Streams.csv"
        usecols = ["ID", "Stream_Name", *PROPERTY_COLUMNS]
        stream_frame = pd.read_csv(
            stream_path,
            usecols=lambda column: column in usecols,
            dtype={"Stream_Name": str},
        )
        stream_frame["_ID_CANONICAL"] = stream_frame["ID"].map(
            lambda value: canonicalize_stream_key(
                value,
                process_id=process_id,
                source=f"{stream_path}:ID",
            )
        )
        stream_frame["_STREAM_KEY_CANONICAL"] = stream_frame["Stream_Name"].map(
            lambda value: canonicalize_stream_key(
                value,
                process_id=process_id,
                source=stream_path,
            )
        )
        expected_ids = _main_sample_ids(main_frame, process_id)
        csv_ids = set(
            stream_frame.loc[
                stream_frame["_ID_CANONICAL"].astype(bool),
                "_ID_CANONICAL",
            ].astype(str)
        )

        graph_key_counts = (
            process_edges.groupby(["_RAW_KEY", "_CANONICAL_KEY"], dropna=False)
            .size()
            .reset_index(name="row_count")
        )
        for row in graph_key_counts.to_dict(orient="records"):
            mapping_rows.append(
                {
                    "process_id": process_key,
                    "source": "canonical_graph",
                    "raw_key": row["_RAW_KEY"],
                    "canonical_key": row["_CANONICAL_KEY"],
                    "row_count": int(row["row_count"]),
                }
            )
        csv_key_counts = (
            stream_frame.groupby(
                ["Stream_Name", "_STREAM_KEY_CANONICAL"],
                dropna=False,
            )
            .size()
            .reset_index(name="row_count")
        )
        for row in csv_key_counts.to_dict(orient="records"):
            mapping_rows.append(
                {
                    "process_id": process_key,
                    "source": "process_stream_csv",
                    "raw_key": row["Stream_Name"],
                    "canonical_key": row["_STREAM_KEY_CANONICAL"],
                    "row_count": int(row["row_count"]),
                }
            )

        keyed_stream = stream_frame[
            stream_frame["_ID_CANONICAL"].astype(bool)
            & stream_frame["_STREAM_KEY_CANONICAL"].astype(bool)
        ]
        collision_groups = keyed_stream.groupby(
            ["_ID_CANONICAL", "_STREAM_KEY_CANONICAL"],
            sort=False,
        )
        for (sample_id, canonical_key), group in collision_groups:
            if len(group) <= 1:
                continue
            collision_rows.append(
                {
                    "process_id": process_key,
                    "sample_id": sample_id,
                    "canonical_key": canonical_key,
                    "raw_keys": "|".join(
                        sorted(set(group["Stream_Name"].astype(str)))
                    ),
                    "row_count": int(len(group)),
                    "csv_path": str(stream_path.resolve()),
                }
            )

        ids_by_raw: dict[str, set[str]] = defaultdict(set)
        ids_by_canonical: dict[str, set[str]] = defaultdict(set)
        available_by_sample: dict[str, set[str]] = defaultdict(set)
        for row in keyed_stream[
            ["_ID_CANONICAL", "Stream_Name", "_STREAM_KEY_CANONICAL"]
        ].to_dict(orient="records"):
            sample_id = str(row["_ID_CANONICAL"])
            ids_by_raw[str(row["Stream_Name"]).strip()].add(sample_id)
            ids_by_canonical[str(row["_STREAM_KEY_CANONICAL"])].add(sample_id)
            available_by_sample[sample_id].add(
                str(row["_STREAM_KEY_CANONICAL"])
            )

        before_count = 0
        after_count = 0
        target_expected = 0
        target_after = 0
        p03_recovery_expected = 0
        p03_recovery_after = 0
        p03_property_expected = 0
        p03_property_complete = 0
        missing_by_sample: dict[str, list[str]] = defaultdict(list)
        for edge in required_edges.to_dict(orient="records"):
            raw_key = str(edge["_RAW_KEY"]).strip()
            canonical_key = str(edge["_CANONICAL_KEY"])
            edge_id = str(edge["canonical_edge_id"])
            exact_ids = ids_by_raw.get(raw_key, set()) & expected_ids
            joined_ids = ids_by_canonical.get(canonical_key, set()) & expected_ids
            before_count += len(exact_ids)
            after_count += len(joined_ids)
            if edge_id in target_ids:
                target_expected += len(expected_ids)
                target_after += len(joined_ids)
            if process_id == 3 and edge_id in P03_RECOVERY_EDGES:
                p03_recovery_expected += len(expected_ids)
                p03_recovery_after += len(joined_ids)
                edge_rows = keyed_stream[
                    keyed_stream["_STREAM_KEY_CANONICAL"].astype(str).eq(
                        canonical_key
                    )
                    & keyed_stream["_ID_CANONICAL"].astype(str).isin(expected_ids)
                ]
                for property_name in PROPERTY_COLUMNS:
                    if property_name not in edge_rows.columns:
                        continue
                    complete = int(edge_rows[property_name].notna().sum())
                    expected = len(expected_ids)
                    stats = p03_property_coverage.setdefault(
                        property_name,
                        {"expected": 0, "complete": 0},
                    )
                    stats["expected"] += expected
                    stats["complete"] += complete
                    p03_property_expected += expected
                    p03_property_complete += complete
            for sample_id in sorted(expected_ids - joined_ids):
                missing_by_sample[sample_id].append(edge_id)
                missing_rows.append(
                    {
                        "process_id": process_key,
                        "sample_id": sample_id,
                        "canonical_edge_id": edge_id,
                        "raw_graph_key": raw_key,
                        "canonical_graph_key": canonical_key,
                        "available_csv_keys": "|".join(
                            sorted(available_by_sample.get(sample_id, set()))
                        ),
                        "csv_path": str(stream_path.resolve()),
                    }
                )

        for sample_id in sorted(expected_ids - csv_ids):
            missing_by_sample[sample_id].append("<all_stream_rows>")
        for sample_id, failed_edges in sorted(missing_by_sample.items()):
            sample_failure_rows.append(
                {
                    "process_id": process_key,
                    "sample_id": sample_id,
                    "failure_count": len(set(failed_edges)),
                    "failed_edges": "|".join(sorted(set(failed_edges))),
                    "available_csv_key_count": len(
                        available_by_sample.get(sample_id, set())
                    ),
                    "csv_path": str(stream_path.resolve()),
                }
            )

        expected_required = len(expected_ids) * len(required_edges)
        supervision_rows.append(
            {
                "process_id": process_key,
                "sample_count": len(expected_ids),
                "required_edge_count_per_sample": len(required_edges),
                "before_exact_join_supervision_count": before_count,
                "after_canonical_join_supervision_count": after_count,
                "supervision_count_delta": after_count - before_count,
                "expected_required_supervision_count": expected_required,
            }
        )
        process_rows.append(
            {
                "process_id": process_key,
                "sample_count": len(expected_ids),
                "stream_csv_sample_count": len(csv_ids),
                "canonical_edge_count": len(process_edges),
                "required_edge_count": len(required_edges),
                "optional_edge_count": len(optional_edges),
                "required_join_expected": expected_required,
                "required_join_before_exact": before_count,
                "required_join_after_canonical": after_count,
                "required_join_coverage_before": (
                    before_count / expected_required if expected_required else 1.0
                ),
                "required_join_coverage_after": (
                    after_count / expected_required if expected_required else 1.0
                ),
                "missing_required_join_count": expected_required - after_count,
                "target_join_expected": target_expected,
                "target_join_after": target_after,
                "target_join_coverage_after": (
                    target_after / target_expected if target_expected else 1.0
                ),
                "canonical_collision_count": sum(
                    1
                    for row in collision_rows
                    if row["process_id"] == process_key
                ),
                "p03_recovery_expected": p03_recovery_expected,
                "p03_recovery_after": p03_recovery_after,
                "p03_recovery_coverage": (
                    p03_recovery_after / p03_recovery_expected
                    if p03_recovery_expected
                    else None
                ),
                "p03_property_values_expected": p03_property_expected,
                "p03_property_values_complete": p03_property_complete,
                "p03_property_value_coverage": (
                    p03_property_complete / p03_property_expected
                    if p03_property_expected
                    else None
                ),
            }
        )

    process_frame = _empty_frame(
        process_rows,
        [
            "process_id",
            "sample_count",
            "required_join_expected",
            "required_join_after_canonical",
            "required_join_coverage_after",
        ],
    )
    missing_frame = _empty_frame(
        missing_rows,
        [
            "process_id",
            "sample_id",
            "canonical_edge_id",
            "raw_graph_key",
            "canonical_graph_key",
            "available_csv_keys",
            "csv_path",
        ],
    )
    mapping_frame = _empty_frame(
        mapping_rows,
        ["process_id", "source", "raw_key", "canonical_key", "row_count"],
    )
    collision_frame = _empty_frame(
        collision_rows,
        [
            "process_id",
            "sample_id",
            "canonical_key",
            "raw_keys",
            "row_count",
            "csv_path",
        ],
    )
    sample_failure_frame = _empty_frame(
        sample_failure_rows,
        [
            "process_id",
            "sample_id",
            "failure_count",
            "failed_edges",
            "available_csv_key_count",
            "csv_path",
        ],
    )
    supervision_frame = _empty_frame(
        supervision_rows,
        [
            "process_id",
            "before_exact_join_supervision_count",
            "after_canonical_join_supervision_count",
            "supervision_count_delta",
        ],
    )

    process_frame.to_csv(output_dir / "process_join_summary.csv", index=False)
    missing_frame.to_csv(output_dir / "missing_required_edges.csv", index=False)
    mapping_frame.to_csv(
        output_dir / "stream_key_raw_to_canonical.csv",
        index=False,
    )
    collision_frame.to_csv(
        output_dir / "canonical_key_collisions.csv",
        index=False,
    )
    sample_failure_frame.to_csv(
        output_dir / "sample_join_failures.csv",
        index=False,
    )
    supervision_frame.to_csv(
        output_dir / "supervision_count_before_after.csv",
        index=False,
    )

    expected_total = int(process_frame["required_join_expected"].sum())
    joined_total = int(process_frame["required_join_after_canonical"].sum())
    target_expected_total = int(process_frame["target_join_expected"].sum())
    target_joined_total = int(process_frame["target_join_after"].sum())
    summary = {
        "stream_key_canonicalization_version": STREAM_KEY_CANONICALIZATION_VERSION,
        "process_ids": [canonicalize_process_id(pid) for pid in process_ids],
        "required_join_expected": expected_total,
        "required_join_after_canonical": joined_total,
        "required_join_coverage_after": (
            joined_total / expected_total if expected_total else 1.0
        ),
        "target_join_expected": target_expected_total,
        "target_join_after": target_joined_total,
        "target_join_coverage_after": (
            target_joined_total / target_expected_total
            if target_expected_total
            else 1.0
        ),
        "missing_required_edge_count": int(len(missing_frame)),
        "sample_join_failure_count": int(len(sample_failure_frame)),
        "canonical_collision_count": int(len(collision_frame)),
        "p03_required_recovery_edges": sorted(P03_RECOVERY_EDGES),
        "p03_recovery_complete": bool(
            process_frame.loc[
                process_frame["process_id"].eq("P03"),
                "p03_recovery_coverage",
            ].eq(1.0).all()
        ),
        "p03_recovery_property_coverage": {
            property_name: {
                **counts,
                "coverage": (
                    counts["complete"] / counts["expected"]
                    if counts["expected"]
                    else 1.0
                ),
            }
            for property_name, counts in sorted(p03_property_coverage.items())
        },
        "audit_passed": bool(
            joined_total == expected_total
            and target_joined_total == target_expected_total
            and collision_frame.empty
        ),
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit canonical graph to Process_Streams joins for all processes."
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("data/reference/v3"),
    )
    parser.add_argument(
        "--stream-dir",
        type=Path,
        default=Path("data/main_data_Streams"),
    )
    parser.add_argument(
        "--main-csv",
        type=Path,
        default=Path("data/datasets_v3/process_main_merged.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stream_key_join_audit"),
    )
    parser.add_argument(
        "--process-ids",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when required coverage is not 100% or a collision exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = audit_stream_key_joins(
        reference_dir=(PROJECT_ROOT / args.reference_dir).resolve()
        if not args.reference_dir.is_absolute()
        else args.reference_dir,
        stream_dir=(PROJECT_ROOT / args.stream_dir).resolve()
        if not args.stream_dir.is_absolute()
        else args.stream_dir,
        main_csv=(PROJECT_ROOT / args.main_csv).resolve()
        if not args.main_csv.is_absolute()
        else args.main_csv,
        output_dir=(PROJECT_ROOT / args.output_dir).resolve()
        if not args.output_dir.is_absolute()
        else args.output_dir,
        process_ids=[int(value) for value in args.process_ids],
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    if args.strict and not summary["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
