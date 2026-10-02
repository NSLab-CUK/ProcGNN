#!/usr/bin/env python3
"""Standalone-ish zero-shot evaluation wrapper for edge-all PI runs.

This intentionally delegates model/data/metric construction to
``train_process_surrogate.py`` with ``--max-epochs 0`` and a model-only
checkpoint load. The wrapper writes evaluation metadata and never performs a
training epoch or checkpoint update.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _process_label(pid: int) -> str:
    return f"Process{int(pid)}"


def _sample_key(process_id: object, sample_id: object) -> str:
    pid_text = str(process_id).strip()
    if pid_text and not pid_text.startswith("Process"):
        try:
            pid_text = _process_label(int(float(pid_text)))
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


def _load_merged_index_frame(merged_csv: Path) -> pd.DataFrame:
    if not merged_csv.is_file():
        raise FileNotFoundError(f"merged CSV not found: {merged_csv}")
    frame = pd.read_csv(merged_csv, usecols=lambda c: c in {"process_id", "ID"})
    missing = [c for c in ("process_id", "ID") if c not in frame.columns]
    if missing:
        raise ValueError(f"{merged_csv} missing columns required for manifest validation: {missing}")
    return frame.reset_index(drop=True)


def _validate_manifest_against_merged_csv(
    manifest: Path,
    *,
    merged_frame: pd.DataFrame,
    expected_process_ids: list[int],
    label: str,
) -> int:
    df = pd.read_csv(manifest)
    if df.empty:
        raise RuntimeError(f"{label}: manifest is empty: {manifest}")
    required = {"merged_row_index", "process_id", "sample_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{label}: manifest missing columns {missing}: {manifest}")
    expected = {_process_label(int(pid)) for pid in expected_process_ids}
    idx = pd.to_numeric(df["merged_row_index"], errors="raise").astype(int)
    if bool(((idx < 0) | (idx >= len(merged_frame))).any()):
        bad = df[(idx < 0) | (idx >= len(merged_frame))].head(5).to_dict("records")
        raise RuntimeError(f"{label}: merged_row_index out of bounds; examples={bad}")
    merged_rows = merged_frame.iloc[idx.to_numpy()].reset_index(drop=True)
    manifest_keys = [
        _sample_key(pid, sid)
        for pid, sid in zip(df["process_id"].tolist(), df["sample_id"].tolist())
    ]
    merged_keys = [
        _sample_key(pid, sid)
        for pid, sid in zip(merged_rows["process_id"].tolist(), merged_rows["ID"].tolist())
    ]
    mismatch = [
        (i, manifest_keys[i], merged_keys[i], int(idx.iloc[i]))
        for i in range(len(manifest_keys))
        if manifest_keys[i] != merged_keys[i]
    ]
    if mismatch:
        raise RuntimeError(
            f"{label}: manifest merged_row_index does not match process_id/sample_id in merged CSV. "
            f"examples={mismatch[:5]}"
        )
    found = {str(x) for x in merged_rows["process_id"].astype(str).unique().tolist()}
    bad_processes = sorted(found - expected)
    if bad_processes:
        raise RuntimeError(f"{label}: unexpected process ids {bad_processes}; expected={sorted(expected)}")
    return int(len(df))


def _unique_process_ids(manifest: Path) -> list[str]:
    df = pd.read_csv(manifest, usecols=["process_id"])
    return sorted(df["process_id"].astype(str).unique().tolist())


def _latest_run_dir(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    candidates = [p for p in output_dir.iterdir() if p.is_dir() and (p / "metrics.json").is_file()]
    if not candidates:
        candidates = [p for p in output_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a process surrogate checkpoint on a held-out manifest.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--source-train-manifest", required=True)
    parser.add_argument("--source-val-manifest", required=True)
    parser.add_argument("--heldout-process", type=int, default=None, help="Legacy single held-out process id.")
    parser.add_argument("--heldout-process-ids", type=int, nargs="+", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--merged-csv", default="data/datasets_v3/process_main_merged_all_processes_10pct.csv")
    parser.add_argument("--source-process-ids", type=int, nargs="+", default=[])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force-reevaluate", action="store_true")
    parser.add_argument("--skip-startup-debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    heldout_process_ids = [int(p) for p in (args.heldout_process_ids or ([] if args.heldout_process is None else [args.heldout_process]))]
    if not heldout_process_ids:
        raise ValueError("provide --heldout-process-ids or legacy --heldout-process")

    base_config = _require_file(_resolve(args.base_config), "base config")
    checkpoint = _require_file(_resolve(args.checkpoint), "checkpoint")
    merged_csv = _resolve(args.merged_csv)
    test_manifest = _require_file(_resolve(args.test_manifest), "test manifest")
    source_train_manifest = _require_file(_resolve(args.source_train_manifest), "source train manifest")
    source_val_manifest = _require_file(_resolve(args.source_val_manifest), "source val manifest")
    output_dir = _resolve(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force_reevaluate and not args.dry_run:
        raise FileExistsError(f"output dir exists and is not empty: {output_dir} (use --force-reevaluate)")
    output_dir.mkdir(parents=True, exist_ok=True)

    heldout_labels = [f"Process{int(pid)}" for pid in sorted(heldout_process_ids)]
    unique_test_processes = _unique_process_ids(test_manifest)
    if unique_test_processes != heldout_labels:
        raise RuntimeError(
            f"held-out evaluation manifest must contain only {heldout_labels}; got {unique_test_processes}"
        )

    source_process_ids = sorted(int(p) for p in args.source_process_ids)
    if not source_process_ids:
        source_process_ids = [p for p in range(1, 11) if p not in set(heldout_process_ids)]
    process_ids = sorted(set(source_process_ids) | set(heldout_process_ids))
    merged_frame = _load_merged_index_frame(merged_csv)
    source_train_rows = _validate_manifest_against_merged_csv(
        source_train_manifest,
        merged_frame=merged_frame,
        expected_process_ids=source_process_ids,
        label="zero_shot/source_train",
    )
    source_val_rows = _validate_manifest_against_merged_csv(
        source_val_manifest,
        merged_frame=merged_frame,
        expected_process_ids=source_process_ids,
        label="zero_shot/source_val",
    )
    test_rows = _validate_manifest_against_merged_csv(
        test_manifest,
        merged_frame=merged_frame,
        expected_process_ids=heldout_process_ids,
        label="zero_shot/test",
    )
    print(
        f"[zero-shot-eval][manifest-ok] source_train={source_train_rows} "
        f"source_val={source_val_rows} test={test_rows}",
        flush=True,
    )
    base_payload = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
    model_payload = (base_payload.get("overrides") or {}).get("model") or {}
    edge_head_type = str(model_payload.get("edge_head_type", "")).strip().lower()
    export_property_metrics = edge_head_type != "hierarchical_reduced_pi"
    train_payload: dict[str, object] = {
        "epochs": 0,
        "final_target_edge_metric_splits": ["test"],
        # The reduced 11D head has no rho_pred/h_pred outputs. Its target and
        # all-edge component metrics are exported by the main-stream metric
        # path, so the legacy Density/Enthalpy-only export must stay disabled.
        "export_final_target_edge_property_metrics": export_property_metrics,
    }
    if args.batch_size is not None:
        train_payload["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        train_payload["num_workers"] = int(args.num_workers)
    runtime = {
        "seed": 42,
        "device": str(args.device),
        "experiment_name": "unseen_zero_shot_" + "_".join(f"P{int(pid):02d}" for pid in heldout_process_ids) + f"_F{int(args.fold):02d}",
        "output_dir": _rel(output_dir),
        "save_dir": f"{_rel(output_dir)}/checkpoints",
        "train": train_payload,
        "data": {
            "train_data_path": str(Path(args.merged_csv).as_posix()),
            "val_data_path": str(Path(args.merged_csv).as_posix()),
            "test_data_path": str(Path(args.merged_csv).as_posix()),
            "edge_all_processes": process_ids,
            "train_split_manifest_path": _rel(source_train_manifest),
            "val_split_manifest_path": _rel(source_val_manifest),
            "test_split_manifest_path": _rel(test_manifest),
        },
    }
    runtime_path = output_dir / "runtime_overrides.json"
    runtime_path.write_text(json.dumps(runtime, indent=2), encoding="utf-8")

    metadata = {
        "fold": int(args.fold),
        "heldout_processes": heldout_process_ids,
        "unique_process_ids": unique_test_processes,
        "test_sample_count": int(len(pd.read_csv(test_manifest))),
        "checkpoint_path": _rel(checkpoint),
        "source_train_manifest": _rel(source_train_manifest),
        "source_val_manifest": _rel(source_val_manifest),
        "heldout_test_manifest": _rel(test_manifest),
        "normalization_source": "source_train_manifest_recomputed_or_cached_by_train_process_surrogate",
        "heldout_test_excluded_from_normalization": True,
        "evaluation_backend": "train_process_surrogate_max_epochs_0",
        "edge_head_type": edge_head_type,
        "export_final_target_edge_property_metrics": export_property_metrics,
    }
    (output_dir / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str((PROJECT_ROOT / "scripts" / "train_process_surrogate.py").resolve()),
        "--config",
        _rel(base_config),
        "--runtime-overrides-file",
        _rel(runtime_path),
        "--max-epochs",
        "0",
        "--pretrained-checkpoint",
        _rel(checkpoint),
        "--pretrained-load-mode",
        "model_only",
        "--finetune-mode",
        "full",
        "--no-save-model-weights",
    ]
    if args.skip_startup_debug:
        cmd.append("--skip-startup-debug")
    if args.dry_run:
        print(" ".join(cmd))
        print("[zero-shot-eval][dry-run]")
        return
    print("[zero-shot-eval][run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    run_dir = _latest_run_dir(output_dir)
    if run_dir is not None:
        metadata["artifact_run_dir"] = _rel(run_dir)
        metrics_path = run_dir / "metrics.json"
        if metrics_path.is_file():
            metadata["metrics_json"] = _rel(metrics_path)
    (output_dir / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
