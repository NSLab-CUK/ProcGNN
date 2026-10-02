#!/usr/bin/env python3
"""Unified runner for unseen-process zero-shot and transfer experiments."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESS_FOLDS = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]


def _heldout_label(pids: Iterable[int] | int) -> str:
    if isinstance(pids, int):
        pids = [pids]
    return "heldout_" + "_".join(f"P{int(pid):02d}" for pid in sorted(int(p) for p in pids))


def _fold_label(fold: int) -> str:
    return f"fold_{int(fold):02d}"


def _ratio_label(ratio: float) -> str:
    return f"ratio_{int(round(float(ratio) * 100.0)):02d}"


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _source_ids(heldout_ids: Iterable[int]) -> list[int]:
    heldout_set = {int(p) for p in heldout_ids}
    return [p for p in range(1, 11) if p not in heldout_set]


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
    expected_process_ids: Iterable[int],
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
    bad_bounds = df[(idx < 0) | (idx >= len(merged_frame))]
    if not bad_bounds.empty:
        raise RuntimeError(
            f"{label}: merged_row_index out of bounds for merged CSV; "
            f"examples={bad_bounds.head(5).to_dict('records')}"
        )
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
            f"Use data/splits/all_processes_10pct with --sample-fold 1 for the 10pct merged CSV. "
            f"examples={mismatch[:5]}"
        )
    found = {str(x) for x in merged_rows["process_id"].astype(str).unique().tolist()}
    bad_processes = sorted(found - expected)
    if bad_processes:
        raise RuntimeError(
            f"{label}: unexpected merged CSV process ids {bad_processes}; expected={sorted(expected)}"
        )
    return int(len(df))


def _fold_specs(
    folds: Iterable[int],
    *,
    heldout_processes: Iterable[int],
    heldout_mode: str,
) -> list[tuple[int, list[int]]]:
    specs: list[tuple[int, list[int]]] = []
    if heldout_mode == "process_pair":
        for fold in folds:
            fold_i = int(fold)
            if fold_i < 1 or fold_i > len(PROCESS_FOLDS):
                raise ValueError("--folds must be in 1..5 for process-pair unseen experiments.")
            specs.append((fold_i, PROCESS_FOLDS[fold_i - 1]))
        return specs
    if heldout_mode == "single_process":
        for pid in heldout_processes:
            pid_i = int(pid)
            if pid_i < 1 or pid_i > 10:
                raise ValueError("--heldout-processes must be in 1..10 for single-process unseen experiments.")
            for fold in folds:
                fold_i = int(fold)
                if fold_i < 1:
                    raise ValueError("--folds must be positive sample-fold numbers.")
                specs.append((fold_i, [pid_i]))
        return specs
    raise ValueError(f"unknown heldout mode: {heldout_mode}")


def _latest_checkpoint(stage_dir: Path) -> Path | None:
    if not stage_dir.is_dir():
        return None
    candidates = sorted(stage_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _latest_file(stage_dir: Path, name: str) -> Path | None:
    if not stage_dir.is_dir():
        return None
    candidates = sorted(stage_dir.rglob(name), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _latest_metrics_json(stage_dir: Path) -> Path | None:
    if not stage_dir.is_dir():
        return None
    candidates = sorted(stage_dir.rglob("metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _train_artifact_problem(stage_dir: Path) -> str | None:
    if _latest_checkpoint(stage_dir) is None:
        return "missing best.pt"
    metrics_path = _latest_file(stage_dir, "metrics_per_epoch.csv")
    if metrics_path is None:
        return "missing metrics_per_epoch.csv"
    try:
        metrics = pd.read_csv(metrics_path)
    except Exception as exc:
        return f"cannot read metrics_per_epoch.csv: {exc}"
    if metrics.empty:
        return "metrics_per_epoch.csv is empty"
    if "had_validation" in metrics.columns:
        had_val = pd.to_numeric(metrics["had_validation"], errors="coerce").fillna(0)
        if not bool((had_val > 0).any()):
            return "no validation epoch recorded"
    if _latest_file(stage_dir, "target_edge_10d_metrics.csv") is None:
        return "missing target_edge_10d_metrics.csv"
    if _latest_file(stage_dir, "pi_all_edge_property_r2.csv") is None:
        return "missing pi_all_edge_property_r2.csv"
    return None


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_runtime_overrides(
    path: Path,
    *,
    seed: int,
    device: str,
    experiment_name: str,
    output_dir: Path,
    merged_csv: str,
    process_ids: Iterable[int],
    train_manifest: Path,
    val_manifest: Path,
    test_manifest: Path,
    max_epochs: int,
    monitor_metric: str,
    monitor_mode: str,
    learning_rate: float | None,
    batch_size: int | None,
) -> None:
    train: dict[str, object] = {
        "epochs": int(max_epochs),
        "monitor_metric": str(monitor_metric),
        "monitor_mode": str(monitor_mode),
    }
    if learning_rate is not None:
        train["learning_rate"] = float(learning_rate)
    if batch_size is not None:
        train["batch_size"] = int(batch_size)
    payload = {
        "seed": int(seed),
        "device": str(device),
        "experiment_name": str(experiment_name),
        "output_dir": _rel(output_dir),
        "save_dir": f"{_rel(output_dir)}/checkpoints",
        "train": train,
        "data": {
            "train_data_path": str(Path(merged_csv).as_posix()),
            "val_data_path": str(Path(merged_csv).as_posix()),
            "test_data_path": str(Path(merged_csv).as_posix()),
            "edge_all_processes": [int(p) for p in process_ids],
            "train_split_manifest_path": _rel(train_manifest),
            "val_split_manifest_path": _rel(val_manifest),
            "test_split_manifest_path": _rel(test_manifest),
        },
    }
    _write_json(path, payload)


def _status_path(run_dir: Path) -> Path:
    return run_dir / "status.json"


def _is_completed(run_dir: Path, *, kind: str) -> bool:
    status_path = _status_path(run_dir)
    if not status_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if status.get("status") != "completed":
        return False
    if kind == "train":
        return _train_artifact_problem(run_dir) is None
    if kind == "eval":
        return (run_dir / "evaluation_metadata.json").is_file() and _latest_metrics_json(run_dir) is not None
    return True


@dataclass
class Task:
    name: str
    kind: str
    run_dir: Path
    cmd: list[str]
    metadata: dict[str, object]
    gpu_id: str | None = None


def _prepare_task_files(task: Task) -> None:
    task.run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(task.run_dir / "run_metadata.json", task.metadata)
    (task.run_dir / "command.txt").write_text(" ".join(task.cmd), encoding="utf-8")
    _write_json(_status_path(task.run_dir), {"status": "pending", "task": task.name, "kind": task.kind})


def _run_tasks(
    tasks: list[Task],
    *,
    gpu_ids: list[str],
    max_parallel: int,
    resume_existing: bool,
    force_reevaluate: bool,
    dry_run: bool,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    if dry_run:
        for task in tasks:
            print(" ".join(task.cmd))
        print(f"[unseen-transfer][dry-run] tasks={len(tasks)}")
        return [{"task": t.name, "status": "dry_run"} for t in tasks]

    queue = list(tasks)
    running: list[tuple[Task, subprocess.Popen]] = []
    max_parallel = max(1, int(max_parallel))
    gpu_ids = gpu_ids or [""]

    while queue or running:
        while queue and len(running) < max_parallel:
            task = queue.pop(0)
            if resume_existing and not force_reevaluate and _is_completed(task.run_dir, kind=task.kind):
                _write_json(_status_path(task.run_dir), {"status": "skipped", "reason": "completed", "task": task.name})
                results.append({"task": task.name, "status": "skipped"})
                continue
            _prepare_task_files(task)
            gpu = gpu_ids[len(running) % len(gpu_ids)]
            env = os.environ.copy()
            if gpu != "":
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            _write_json(
                _status_path(task.run_dir),
                {"status": "running", "task": task.name, "kind": task.kind, "gpu": gpu, "started_at": time.time()},
            )
            print(f"[unseen-transfer][start] {task.name} run_dir={_rel(task.run_dir)}", flush=True)
            proc = subprocess.Popen(task.cmd, cwd=str(PROJECT_ROOT), env=env)
            running.append((task, proc))
        time.sleep(1.0)
        next_running: list[tuple[Task, subprocess.Popen]] = []
        for task, proc in running:
            rc = proc.poll()
            if rc is None:
                next_running.append((task, proc))
                continue
            artifact_problem = None
            if rc == 0 and task.kind == "train":
                artifact_problem = _train_artifact_problem(task.run_dir)
            status = "completed" if rc == 0 and artifact_problem is None else "failed"
            status_payload = {
                "status": status,
                "task": task.name,
                "kind": task.kind,
                "returncode": int(rc),
                "finished_at": time.time(),
            }
            if artifact_problem is not None:
                status_payload["artifact_problem"] = artifact_problem
            _write_json(
                _status_path(task.run_dir),
                status_payload,
            )
            print(
                f"[unseen-transfer][{status}] {task.name} returncode={int(rc)} run_dir={_rel(task.run_dir)}",
                flush=True,
            )
            if artifact_problem is not None:
                print(
                    f"[unseen-transfer][artifact-problem] {task.name}: {artifact_problem}",
                    flush=True,
                )
            results.append(
                {
                    "task": task.name,
                    "status": status,
                    "returncode": int(rc),
                    "run_dir": _rel(task.run_dir),
                    "artifact_problem": artifact_problem or "",
                }
            )
        running = next_running
    failed = [row for row in results if str(row.get("status")) == "failed"]
    if failed:
        previews: list[str] = []
        for row in failed[:5]:
            run_dir = _resolve(str(row.get("run_dir", ""))) if row.get("run_dir") else None
            stderr_path = run_dir / "stderr.log" if run_dir is not None else None
            tail = ""
            if stderr_path is not None and stderr_path.is_file():
                lines = stderr_path.read_text(encoding="utf-8", errors="replace").splitlines()
                tail = "\n".join(lines[-20:])
            problem = str(row.get("artifact_problem") or "").strip()
            previews.append(
                f"task={row.get('task')} returncode={row.get('returncode')} "
                f"run_dir={row.get('run_dir')} artifact_problem={problem}\n{tail}"
            )
        raise RuntimeError(
            "unseen-transfer task failed before aggregation:\n" + "\n\n".join(previews)
        )
    return results


def _train_cmd(base_config: Path, runtime_path: Path, max_epochs: int, *, pretrained: Path | None, finetune_mode: str, skip_debug: bool) -> list[str]:
    cmd = [
        sys.executable,
        str((PROJECT_ROOT / "scripts" / "train_process_surrogate.py").resolve()),
        "--config",
        _rel(base_config),
        "--runtime-overrides-file",
        _rel(runtime_path),
        "--max-epochs",
        str(int(max_epochs)),
        "--finetune-mode",
        str(finetune_mode),
    ]
    if pretrained is not None:
        cmd.extend(["--pretrained-checkpoint", _rel(pretrained), "--pretrained-load-mode", "model_only"])
    if skip_debug:
        cmd.append("--skip-startup-debug")
    return cmd


def _eval_cmd(
    *,
    base_config: Path,
    checkpoint: Path,
    source_train: Path,
    source_val: Path,
    target_test: Path,
    heldout_ids: list[int],
    fold: int,
    output_dir: Path,
    merged_csv: str,
    source_ids: list[int],
    device: str,
    skip_debug: bool,
    force_reevaluate: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str((PROJECT_ROOT / "scripts" / "eval_process_surrogate_edge_all.py").resolve()),
        "--base-config",
        _rel(base_config),
        "--checkpoint",
        _rel(checkpoint),
        "--test-manifest",
        _rel(target_test),
        "--source-train-manifest",
        _rel(source_train),
        "--source-val-manifest",
        _rel(source_val),
        "--heldout-process-ids",
        *[str(int(p)) for p in heldout_ids],
        "--fold",
        str(int(fold)),
        "--output-dir",
        _rel(output_dir),
        "--merged-csv",
        str(Path(merged_csv).as_posix()),
        "--source-process-ids",
        *[str(int(p)) for p in source_ids],
        "--device",
        str(device),
    ]
    if force_reevaluate:
        cmd.append("--force-reevaluate")
    if skip_debug:
        cmd.append("--skip-startup-debug")
    return cmd


def _build_split_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str((PROJECT_ROOT / "scripts" / "create_process_unseen_transfer_splits.py").resolve()),
        "--heldout-processes",
        *[str(int(p)) for p in args.heldout_processes],
        "--heldout-mode",
        str(args.heldout_mode),
        "--folds",
        *[str(int(f)) for f in args.folds],
        "--ratios",
        *[str(float(r)) for r in args.ratios],
        "--input-split-root",
        str(args.input_split_root),
        "--sample-fold",
        str(int(args.sample_fold)),
        "--merged-csv",
        str(args.merged_csv),
        "--output-root",
        str(args.split_root),
        "--adaptation-val-ratio",
        str(float(args.adaptation_val_ratio)),
        "--seed",
        str(int(args.seed)),
        *(["--overwrite"] if args.force_reevaluate else []),
    ]


def _rerun_self_with_mode(args: argparse.Namespace, mode: str) -> None:
    cmd = [sys.executable, str((PROJECT_ROOT / "scripts" / "run_unseen_process_transfer_experiments.py").resolve())]
    cmd.extend(["--base-config", str(args.base_config)])
    cmd.extend(["--heldout-processes", *[str(int(p)) for p in args.heldout_processes]])
    cmd.extend(["--heldout-mode", str(args.heldout_mode)])
    cmd.extend(["--folds", *[str(int(f)) for f in args.folds]])
    cmd.extend(["--ratios", *[str(float(r)) for r in args.ratios]])
    cmd.extend(["--methods", *[str(m) for m in args.methods]])
    cmd.extend(["--input-split-root", str(args.input_split_root)])
    cmd.extend(["--sample-fold", str(int(args.sample_fold))])
    cmd.extend(["--split-root", str(args.split_root)])
    cmd.extend(["--merged-csv", str(args.merged_csv)])
    cmd.extend(["--adaptation-val-ratio", str(float(args.adaptation_val_ratio))])
    cmd.extend(["--max-epochs-pretrain", str(int(args.max_epochs_pretrain))])
    cmd.extend(["--max-epochs-transfer", str(int(args.max_epochs_transfer))])
    if args.pretrain_learning_rate is not None:
        cmd.extend(["--pretrain-learning-rate", str(float(args.pretrain_learning_rate))])
    if args.finetune_learning_rate is not None:
        cmd.extend(["--finetune-learning-rate", str(float(args.finetune_learning_rate))])
    if args.scratch_learning_rate is not None:
        cmd.extend(["--scratch-learning-rate", str(float(args.scratch_learning_rate))])
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(int(args.batch_size))])
    cmd.extend(["--monitor-metric", str(args.monitor_metric)])
    cmd.extend(["--monitor-mode", str(args.monitor_mode)])
    cmd.extend(["--seed", str(int(args.seed))])
    cmd.extend(["--fold-seed-mode", str(args.fold_seed_mode)])
    cmd.extend(["--normalization-policy", str(args.normalization_policy)])
    cmd.extend(["--finetune-mode", str(args.finetune_mode)])
    cmd.extend(["--output-root", str(args.output_root)])
    cmd.extend(["--mode", mode])
    if str(args.checkpoint).strip():
        cmd.extend(["--checkpoint", str(args.checkpoint)])
    cmd.extend(["--gpu-ids", *[str(g) for g in args.gpu_ids]])
    cmd.extend(["--max-parallel", str(int(args.max_parallel))])
    if args.resume_existing:
        cmd.append("--resume-existing")
    if args.force_reevaluate and mode != "aggregate":
        cmd.append("--force-reevaluate")
    if args.allow_partial_aggregation:
        cmd.append("--allow-partial-aggregation")
    if args.skip_startup_debug:
        cmd.append("--skip-startup-debug")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _metrics_from_dir(stage_dir: Path) -> dict[str, float]:
    path = _latest_metrics_json(stage_dir)
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, float] = {}
    key_map = {
        "target_mean_r2": ["target_mean_r2", "val_target_mean_r2", "test/target_mean_r2"],
        "flatten_r2": ["target_edge_10d_r2_flatten", "val_target_edge_10d_r2_flatten", "test/target_edge_10d_r2_flatten"],
        "edge_macro_r2": ["target_edge_10d_r2_edge_macro", "val_target_edge_10d_r2_edge_macro", "test/target_edge_10d_r2_edge_macro"],
        "all_edge_mean_r2": ["pi_all_edge_mean_r2", "all_edge_property_mean_r2", "test/pi_all_edge_mean_r2"],
    }
    for out_key, keys in key_map.items():
        for key in keys:
            if key in payload:
                try:
                    out[out_key] = float(payload[key])
                    break
                except Exception:
                    pass
    return out


def aggregate(output_root: Path, *, allow_partial: bool) -> None:
    registry_rows: list[dict[str, object]] = []
    for status_path in output_root.rglob("status.json"):
        run_dir = status_path.parent
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            status = {}
        meta_path = run_dir / "run_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        row = {**meta, **{f"status_{k}": v for k, v in status.items()}, "run_dir": _rel(run_dir)}
        row.update(_metrics_from_dir(run_dir))
        registry_rows.append(row)
    agg_dir = output_root / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    reg = pd.DataFrame(registry_rows)
    reg.to_csv(agg_dir / "run_registry.csv", index=False)
    failed = reg[reg.get("status_status", pd.Series(dtype=str)).astype(str).isin(["failed", "running", "pending"])] if not reg.empty else pd.DataFrame()
    failed.to_csv(agg_dir / "failed_runs.csv", index=False)
    if not failed.empty and not allow_partial:
        raise RuntimeError(f"cannot aggregate with failed/incomplete runs without --allow-partial-aggregation: {len(failed)} rows")

    for stage_name, out_name in (
        ("zero_shot", "zero_shot_fold_metrics.csv"),
        ("finetune", "transfer_fold_metrics.csv"),
        ("scratch", "scratch_fold_metrics.csv"),
    ):
        sub = reg[reg.get("stage", pd.Series(dtype=str)).astype(str) == stage_name] if not reg.empty and "stage" in reg else pd.DataFrame()
        sub.to_csv(agg_dir / out_name, index=False)
    summary_rows: list[dict[str, object]] = []
    metric_cols = [c for c in ("target_mean_r2", "flatten_r2", "edge_macro_r2", "all_edge_mean_r2") if c in reg.columns]
    group_cols = [c for c in ("stage", "method", "ratio") if c in reg.columns]
    if metric_cols and group_cols:
        for keys, grp in reg.groupby(group_cols, dropna=False):
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            base = dict(zip(group_cols, key_tuple))
            for metric in metric_cols:
                vals = pd.to_numeric(grp[metric], errors="coerce").dropna()
                summary_rows.append(
                    {
                        **base,
                        "metric": metric,
                        "mean": float(vals.mean()) if not vals.empty else float("nan"),
                        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                        "min": float(vals.min()) if not vals.empty else float("nan"),
                        "max": float(vals.max()) if not vals.empty else float("nan"),
                        "valid_runs": int(len(vals)),
                    }
                )
    pd.DataFrame(summary_rows).to_csv(agg_dir / "global_summary.csv", index=False)
    _write_json(
        agg_dir / "aggregate_metadata.json",
        {"registry_rows": len(registry_rows), "failed_rows": int(len(failed)), "allow_partial": bool(allow_partial)},
    )
    print(f"[unseen-transfer][aggregate] wrote {agg_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full unseen-process zero-shot and transfer experiments.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--heldout-processes", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument(
        "--heldout-mode",
        choices=("process_pair", "single_process"),
        default="process_pair",
        help="process_pair uses fixed held-out pairs [[1,2],...]; single_process runs each requested process across the requested sample folds.",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20, 0.50])
    parser.add_argument("--methods", choices=("finetune", "scratch"), nargs="+", default=["finetune", "scratch"])
    parser.add_argument("--input-split-root", default="data/splits/all_processes_10pct")
    parser.add_argument("--sample-fold", type=int, default=1)
    parser.add_argument("--split-root", default="data/splits/unseen_transfer")
    parser.add_argument("--merged-csv", default="data/datasets_v3/process_main_merged_all_processes_10pct.csv")
    parser.add_argument("--adaptation-val-ratio", type=float, default=0.20)
    parser.add_argument("--max-epochs-pretrain", type=int, default=5)
    parser.add_argument("--max-epochs-transfer", type=int, default=5)
    parser.add_argument("--pretrain-learning-rate", type=float, default=None)
    parser.add_argument("--finetune-learning-rate", type=float, default=None)
    parser.add_argument("--scratch-learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--monitor-metric", default="val_target_mean_r2")
    parser.add_argument("--monitor-mode", default="max")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-seed-mode", choices=("fixed", "offset"), default="fixed")
    parser.add_argument("--normalization-policy", choices=("source_fixed", "target_adaptation"), default="source_fixed")
    parser.add_argument("--finetune-mode", choices=("full", "head_only"), default="full")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--mode", choices=("splits", "pretrain", "zero_shot", "finetune", "scratch", "aggregate", "all"), default="all")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--gpu-ids", nargs="+", default=["0"])
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--force-reevaluate", action="store_true")
    parser.add_argument("--allow-partial-aggregation", action="store_true")
    parser.add_argument("--skip-startup-debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_config = _require_file(_resolve(args.base_config), "base config")
    split_root = _resolve(args.split_root)
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    merged_frame = _load_merged_index_frame(_resolve(args.merged_csv))
    _write_json(
        output_root / "global_metadata.json",
        {
            "base_config": _rel(base_config),
            "heldout_processes": [int(p) for p in args.heldout_processes],
            "heldout_mode": str(args.heldout_mode),
            "folds": [int(f) for f in args.folds],
            "ratios": [float(r) for r in args.ratios],
            "methods": list(args.methods),
            "normalization_policy": str(args.normalization_policy),
            "note": "source_fixed records policy metadata; current backend recomputes scalers from the runtime train manifest.",
        },
    )

    if args.mode in {"splits", "all"}:
        split_cmd = _build_split_command(args)
        if args.dry_run:
            print(" ".join(split_cmd))
        else:
            subprocess.run(split_cmd, cwd=str(PROJECT_ROOT), check=True)
        if args.mode == "splits":
            return

    if args.mode == "aggregate":
        aggregate(output_root, allow_partial=bool(args.allow_partial_aggregation))
        return

    pretrain_tasks: list[Task] = []
    zero_tasks: list[Task] = []
    transfer_tasks: list[Task] = []
    scratch_tasks: list[Task] = []

    for fold, heldout_ids in _fold_specs(
        [int(f) for f in args.folds],
        heldout_processes=[int(p) for p in args.heldout_processes],
        heldout_mode=str(args.heldout_mode),
    ):
        source_ids = _source_ids(heldout_ids)
        group_label = _heldout_label(heldout_ids)
        group_name = group_label.replace("heldout_", "")
        fold_seed = int(args.seed) if args.fold_seed_mode == "fixed" else int(args.seed) + fold
        fold_split = split_root / group_label / _fold_label(fold)
        source_train = _require_file(fold_split / "source_train.csv", f"{group_label} fold {fold} source_train")
        source_val = _require_file(fold_split / "source_val.csv", f"{group_label} fold {fold} source_val")
        target_test = _require_file(fold_split / "target_test.csv", f"{group_label} fold {fold} target_test")
        source_train_rows = _validate_manifest_against_merged_csv(
            source_train,
            merged_frame=merged_frame,
            expected_process_ids=source_ids,
            label=f"{group_label}/{_fold_label(fold)}/source_train",
        )
        source_val_rows = _validate_manifest_against_merged_csv(
            source_val,
            merged_frame=merged_frame,
            expected_process_ids=source_ids,
            label=f"{group_label}/{_fold_label(fold)}/source_val",
        )
        target_test_rows = _validate_manifest_against_merged_csv(
            target_test,
            merged_frame=merged_frame,
            expected_process_ids=heldout_ids,
            label=f"{group_label}/{_fold_label(fold)}/target_test",
        )
        print(
            f"[unseen-transfer][manifest-ok] {group_label}/{_fold_label(fold)} "
            f"source_train={source_train_rows} source_val={source_val_rows} "
            f"target_test={target_test_rows}",
            flush=True,
        )
        fold_out = output_root / group_label / _fold_label(fold)
        pretrain_dir = fold_out / "pretrain"
        runtime = pretrain_dir / "runtime_overrides.json"
        _write_runtime_overrides(
            runtime,
            seed=fold_seed,
            device="cuda",
            experiment_name=f"unseen_pretrain_{group_name}_F{fold:02d}",
            output_dir=pretrain_dir,
            merged_csv=args.merged_csv,
            process_ids=source_ids,
            train_manifest=source_train,
            val_manifest=source_val,
            test_manifest=source_val,
            max_epochs=int(args.max_epochs_pretrain),
            monitor_metric=str(args.monitor_metric),
            monitor_mode=str(args.monitor_mode),
            learning_rate=args.pretrain_learning_rate,
            batch_size=args.batch_size,
        )
        pretrain_tasks.append(
            Task(
                name=f"{group_name}_F{fold:02d}_pretrain",
                kind="train",
                run_dir=pretrain_dir,
                cmd=_train_cmd(base_config, runtime, int(args.max_epochs_pretrain), pretrained=None, finetune_mode="full", skip_debug=bool(args.skip_startup_debug)),
                metadata={"stage": "pretrain", "heldout_processes": heldout_ids, "fold": fold, "source_process_ids": source_ids},
            )
        )
        pre_ckpt = _resolve(args.checkpoint) if str(args.checkpoint).strip() else _latest_checkpoint(pretrain_dir)
        if pre_ckpt is None and args.dry_run:
            pre_ckpt = pretrain_dir / "checkpoints" / "<RUN>" / "best.pt"
        if pre_ckpt is not None:
            zero_dir = fold_out / "zero_shot"
            zero_tasks.append(
                Task(
                    name=f"{group_name}_F{fold:02d}_zero_shot",
                    kind="eval",
                    run_dir=zero_dir,
                    cmd=_eval_cmd(
                        base_config=base_config,
                        checkpoint=pre_ckpt,
                        source_train=source_train,
                        source_val=source_val,
                        target_test=target_test,
                        heldout_ids=heldout_ids,
                        fold=fold,
                        output_dir=zero_dir,
                        merged_csv=args.merged_csv,
                        source_ids=source_ids,
                        device="cuda",
                        skip_debug=bool(args.skip_startup_debug),
                        force_reevaluate=bool(args.force_reevaluate),
                    ),
                    metadata={"stage": "zero_shot", "heldout_processes": heldout_ids, "fold": fold, "checkpoint": _rel(pre_ckpt)},
                )
            )
        for ratio in [float(r) for r in args.ratios]:
            rlabel = _ratio_label(ratio)
            ratio_dir = fold_split / rlabel
            rtrain = _require_file(ratio_dir / "train.csv", f"{group_label} {rlabel} train")
            rval = _require_file(ratio_dir / "val.csv", f"{group_label} {rlabel} val")
            rtrain_rows = _validate_manifest_against_merged_csv(
                rtrain,
                merged_frame=merged_frame,
                expected_process_ids=heldout_ids,
                label=f"{group_label}/{_fold_label(fold)}/{rlabel}/train",
            )
            rval_rows = _validate_manifest_against_merged_csv(
                rval,
                merged_frame=merged_frame,
                expected_process_ids=heldout_ids,
                label=f"{group_label}/{_fold_label(fold)}/{rlabel}/val",
            )
            print(
                f"[unseen-transfer][manifest-ok] {group_label}/{_fold_label(fold)}/{rlabel} "
                f"train={rtrain_rows} val={rval_rows}",
                flush=True,
            )
            for method in args.methods:
                if method == "finetune":
                    if pre_ckpt is None:
                        continue
                    stage_dir = fold_out / rlabel / f"finetune_{args.finetune_mode}"
                    lr = args.finetune_learning_rate
                    pretrained = pre_ckpt
                else:
                    stage_dir = fold_out / rlabel / "scratch"
                    lr = args.scratch_learning_rate
                    pretrained = None
                runtime_path = stage_dir / "runtime_overrides.json"
                # Transfer and scratch intentionally share the same ratio manifests.
                _write_runtime_overrides(
                    runtime_path,
                    seed=fold_seed,
                    device="cuda",
                    experiment_name=f"unseen_{method}_{group_name}_F{fold:02d}_{rlabel}",
                    output_dir=stage_dir,
                    merged_csv=args.merged_csv,
                    process_ids=heldout_ids,
                    train_manifest=rtrain,
                    val_manifest=rval,
                    test_manifest=target_test,
                    max_epochs=int(args.max_epochs_transfer),
                    monitor_metric=str(args.monitor_metric),
                    monitor_mode=str(args.monitor_mode),
                    learning_rate=lr,
                    batch_size=args.batch_size,
                )
                task = Task(
                    name=f"{group_name}_F{fold:02d}_{rlabel}_{method}",
                    kind="train",
                    run_dir=stage_dir,
                    cmd=_train_cmd(
                        base_config,
                        runtime_path,
                        int(args.max_epochs_transfer),
                        pretrained=pretrained,
                        finetune_mode=str(args.finetune_mode if method == "finetune" else "full"),
                        skip_debug=bool(args.skip_startup_debug),
                    ),
                    metadata={
                        "stage": method,
                        "method": method,
                        "heldout_processes": heldout_ids,
                        "fold": fold,
                        "ratio": ratio,
                        "ratio_label": rlabel,
                        "normalization_policy_requested": str(args.normalization_policy),
                        "normalization_policy_backend": "runtime_train_manifest",
                        "source_process_ids": source_ids,
                    },
                )
                (transfer_tasks if method == "finetune" else scratch_tasks).append(task)

    if args.mode in {"pretrain", "all"}:
        _run_tasks(pretrain_tasks, gpu_ids=[str(g) for g in args.gpu_ids], max_parallel=int(args.max_parallel), resume_existing=bool(args.resume_existing), force_reevaluate=bool(args.force_reevaluate), dry_run=bool(args.dry_run))
        if args.mode == "all" and not args.dry_run:
            _rerun_self_with_mode(args, "zero_shot")
            if "finetune" in set(args.methods):
                _rerun_self_with_mode(args, "finetune")
            if "scratch" in set(args.methods):
                _rerun_self_with_mode(args, "scratch")
            _rerun_self_with_mode(args, "aggregate")
            return
    if args.mode in {"zero_shot", "all"}:
        _run_tasks(zero_tasks, gpu_ids=[str(g) for g in args.gpu_ids], max_parallel=int(args.max_parallel), resume_existing=bool(args.resume_existing), force_reevaluate=bool(args.force_reevaluate), dry_run=bool(args.dry_run))
    if args.mode in {"finetune", "all"}:
        _run_tasks(transfer_tasks, gpu_ids=[str(g) for g in args.gpu_ids], max_parallel=int(args.max_parallel), resume_existing=bool(args.resume_existing), force_reevaluate=bool(args.force_reevaluate), dry_run=bool(args.dry_run))
    if args.mode in {"scratch", "all"}:
        _run_tasks(scratch_tasks, gpu_ids=[str(g) for g in args.gpu_ids], max_parallel=int(args.max_parallel), resume_existing=bool(args.resume_existing), force_reevaluate=bool(args.force_reevaluate), dry_run=bool(args.dry_run))
    if args.mode == "all" and not args.dry_run:
        aggregate(output_root, allow_partial=bool(args.allow_partial_aggregation))


if __name__ == "__main__":
    main()
