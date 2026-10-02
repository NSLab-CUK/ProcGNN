#!/usr/bin/env python3
"""Run single-heldout full-data unseen experiments.

This runner implements:
- one pretrain per held-out process on the other nine processes,
- five zero-shot evaluations per held-out process,
- five transfer runs per held-out process using target train+val as adaptation
  train and target test as validation/test evaluation.
"""
from __future__ import annotations

import argparse
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


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _rel(path: Path) -> str:
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


def _source_ids(heldout: int) -> list[int]:
    return [pid for pid in range(1, 11) if int(pid) != int(heldout)]


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _latest_checkpoint(stage_dir: Path) -> Path | None:
    if not stage_dir.is_dir():
        return None
    candidates = sorted(stage_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _latest_completed_checkpoint(stage_dir: Path) -> Path | None:
    """Prefer checkpoints whose matching run directory finished final export."""
    if not stage_dir.is_dir():
        return None
    candidates = sorted(stage_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    for checkpoint in candidates:
        run_dir = stage_dir / checkpoint.parent.name
        if (run_dir / "metrics.json").is_file():
            return checkpoint
    return candidates[0] if candidates else None


def _latest_file(stage_dir: Path, name: str) -> Path | None:
    if not stage_dir.is_dir():
        return None
    candidates = sorted(stage_dir.rglob(name), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _train_artifact_problem(stage_dir: Path) -> str | None:
    if _latest_checkpoint(stage_dir) is None:
        return "missing best.pt"
    if _latest_file(stage_dir, "metrics_per_epoch.csv") is None:
        return "missing metrics_per_epoch.csv"
    return None


def _completed(run_dir: Path, *, kind: str) -> bool:
    status_path = run_dir / "status.json"
    if not status_path.is_file():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    status_name = str(status.get("status") or "")
    completed_or_resume_skip = status_name == "completed" or (
        status_name == "skipped" and status.get("reason") == "completed"
    )
    if not completed_or_resume_skip:
        return False
    if kind == "train":
        return _train_artifact_problem(run_dir) is None
    if kind == "eval":
        return (run_dir / "evaluation_metadata.json").is_file()
    return True


def _write_runtime_overrides(
    path: Path,
    *,
    seed: int,
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
    epoch_sampler: Mapping[str, object] | None = None,
    early_stopping_patience: int | None = None,
    final_target_edge_metric_splits: Iterable[str] | None = None,
    train_overrides: Mapping[str, object] | None = None,
) -> None:
    train_payload: dict[str, object] = {
        "epochs": int(max_epochs),
        "monitor_metric": str(monitor_metric),
        "monitor_mode": str(monitor_mode),
    }
    if learning_rate is not None:
        train_payload["learning_rate"] = float(learning_rate)
    if batch_size is not None:
        train_payload["batch_size"] = int(batch_size)
    if epoch_sampler is not None:
        train_payload["epoch_sampler"] = dict(epoch_sampler)
    if early_stopping_patience is not None:
        train_payload["early_stopping_patience"] = int(early_stopping_patience)
    if final_target_edge_metric_splits is not None:
        train_payload["final_target_edge_metric_splits"] = [
            str(value) for value in final_target_edge_metric_splits
        ]
    if train_overrides is not None:
        train_payload.update(dict(train_overrides))
    payload = {
        "seed": int(seed),
        "device": "cuda",
        "experiment_name": str(experiment_name),
        "output_dir": _rel(output_dir),
        "save_dir": f"{_rel(output_dir)}/checkpoints",
        "data": {
            "train_data_path": str(Path(merged_csv).as_posix()),
            "val_data_path": str(Path(merged_csv).as_posix()),
            "test_data_path": str(Path(merged_csv).as_posix()),
            "edge_all_processes": [int(p) for p in process_ids],
            "train_split_manifest_path": _rel(train_manifest),
            "val_split_manifest_path": _rel(val_manifest),
            "test_split_manifest_path": _rel(test_manifest),
        },
        "train": train_payload,
    }
    _write_json(path, payload)


def _train_cmd(
    *,
    base_config: Path,
    runtime_path: Path,
    max_epochs: int,
    pretrained: Path | None,
    finetune_mode: str,
    skip_startup_debug: bool,
    no_training_plots: bool = False,
) -> list[str]:
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
    if skip_startup_debug:
        cmd.append("--skip-startup-debug")
    if no_training_plots:
        cmd.append("--no-training-plots")
    return cmd


def _eval_cmd(
    *,
    base_config: Path,
    checkpoint: Path,
    source_train: Path,
    source_val: Path,
    target_test: Path,
    heldout: int,
    fold: int,
    output_dir: Path,
    merged_csv: str,
    source_ids: list[int],
    skip_startup_debug: bool,
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
        str(int(heldout)),
        "--fold",
        str(int(fold)),
        "--output-dir",
        _rel(output_dir),
        "--merged-csv",
        str(Path(merged_csv).as_posix()),
        "--source-process-ids",
        *[str(int(pid)) for pid in source_ids],
        "--device",
        "cuda",
    ]
    if force_reevaluate:
        cmd.append("--force-reevaluate")
    if skip_startup_debug:
        cmd.append("--skip-startup-debug")
    return cmd


@dataclass
class Task:
    name: str
    kind: str
    run_dir: Path
    cmd: list[str]
    metadata: dict[str, object]


def _prepare_task(task: Task) -> None:
    task.run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(task.run_dir / "run_metadata.json", task.metadata)
    (task.run_dir / "command.txt").write_text(" ".join(task.cmd), encoding="utf-8")
    _write_json(task.run_dir / "status.json", {"status": "pending", "task": task.name, "kind": task.kind})


def _run_tasks(
    tasks: list[Task],
    *,
    gpu_ids: list[str],
    max_parallel: int,
    resume_existing: bool,
    force_reevaluate: bool,
    dry_run: bool,
) -> None:
    if dry_run:
        for task in tasks:
            print(" ".join(task.cmd))
        print(f"[single-full-unseen][dry-run] tasks={len(tasks)}")
        return
    queue = list(tasks)
    running: list[tuple[Task, subprocess.Popen, str, float]] = []
    gpu_ids = gpu_ids or [""]
    max_parallel = min(max(1, int(max_parallel)), len(gpu_ids))
    while queue or running:
        while queue and len(running) < max_parallel:
            task = queue.pop(0)
            if resume_existing and not force_reevaluate and _completed(task.run_dir, kind=task.kind):
                # Keep the authoritative completed status intact. Overwriting it
                # with "skipped" made a later resume treat the same task as
                # incomplete and restart training from epoch 1.
                print(f"[single-full-unseen][skip] {task.name}", flush=True)
                continue
            _prepare_task(task)
            used_gpus = {item[2] for item in running}
            gpu = next((candidate for candidate in gpu_ids if candidate not in used_gpus), gpu_ids[0])
            env = os.environ.copy()
            if gpu != "":
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            started_at = time.time()
            _write_json(
                task.run_dir / "status.json",
                {"status": "running", "task": task.name, "kind": task.kind, "gpu": gpu, "started_at": started_at},
            )
            print(f"[single-full-unseen][start] {task.name} gpu={gpu} run_dir={_rel(task.run_dir)}", flush=True)
            running.append((task, subprocess.Popen(task.cmd, cwd=str(PROJECT_ROOT), env=env), gpu, started_at))
        time.sleep(1.0)
        still_running: list[tuple[Task, subprocess.Popen, str, float]] = []
        for task, proc, gpu, started_at in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((task, proc, gpu, started_at))
                continue
            problem = _train_artifact_problem(task.run_dir) if rc == 0 and task.kind == "train" else None
            status = "completed" if rc == 0 and problem is None else "failed"
            payload: dict[str, object] = {
                "status": status,
                "task": task.name,
                "kind": task.kind,
                "returncode": int(rc),
                "gpu": gpu,
                "started_at": started_at,
                "finished_at": time.time(),
            }
            payload["duration_sec"] = float(payload["finished_at"]) - float(started_at)
            if problem:
                payload["artifact_problem"] = problem
            _write_json(task.run_dir / "status.json", payload)
            print(f"[single-full-unseen][{status}] {task.name} rc={int(rc)}", flush=True)
            if status == "failed":
                raise RuntimeError(f"task failed: {task.name} run_dir={_rel(task.run_dir)} problem={problem or ''}")
        running = still_running


def _build_split_cmd(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str((PROJECT_ROOT / "scripts" / "create_single_process_full_unseen_splits.py").resolve()),
        "--input-split-root",
        str(args.input_split_root),
        "--output-root",
        str(args.split_root),
        "--heldout-processes",
        *[str(int(p)) for p in args.heldout_processes],
        "--folds",
        *[str(int(f)) for f in args.folds],
        "--pretrain-source-fold",
        str(int(args.pretrain_source_fold)),
        *(["--overwrite"] if args.force_reevaluate else []),
    ]


def aggregate(output_root: Path) -> None:
    rows: list[dict[str, object]] = []
    for status_path in output_root.rglob("status.json"):
        run_dir = status_path.parent
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            status = {}
        meta_path = run_dir / "run_metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        row = {**meta, **{f"status_{k}": v for k, v in status.items()}, "run_dir": _rel(run_dir)}
        metrics = _latest_file(run_dir, "metrics.json")
        if metrics is not None:
            try:
                payload = json.loads(metrics.read_text(encoding="utf-8"))
                for key in (
                    "target_edge_property_mean_r2",
                    "target_mean_r2",
                    "target_edge_10d_r2_flatten",
                    "pi_all_edge_property_mean_r2",
                ):
                    if key in payload:
                        row[key] = payload[key]
            except Exception:
                pass
        rows.append(row)
    agg_dir = output_root / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(agg_dir / "run_registry.csv", index=False)
    print(f"[single-full-unseen][aggregate] wrote {_rel(agg_dir)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single-process full unseen pretrain/zero-shot/transfer experiments.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--input-split-root", default="data/splits/process_kfold")
    parser.add_argument("--split-root", default="data/splits/single_process_full_unseen_60_20_20")
    parser.add_argument("--merged-csv", default="data/datasets_v3/process_main_merged.csv")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--heldout-processes", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--pretrain-source-fold", type=int, default=1)
    parser.add_argument("--mode", choices=("splits", "pretrain", "zero_shot", "transfer", "aggregate", "all"), default="all")
    parser.add_argument("--max-epochs-pretrain", type=int, default=30)
    parser.add_argument("--max-epochs-transfer", type=int, default=30)
    parser.add_argument("--pretrain-learning-rate", type=float, default=None)
    parser.add_argument("--transfer-learning-rate", type=float, default=None)
    parser.add_argument(
        "--transfer-epoch-sample-size",
        type=int,
        default=1000,
        help=(
            "Uniform samples drawn without replacement from target_train per transfer epoch "
            "(default: 1000). Use 0 to inherit the base config sampler."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--monitor-metric", default="val_target_edge_property_mean_r2")
    parser.add_argument("--monitor-mode", default="max")
    parser.add_argument("--seed", type=int, default=260716)
    parser.add_argument("--fold-seed-mode", choices=("fixed", "offset"), default="fixed")
    parser.add_argument("--finetune-mode", choices=("full", "head_only"), default="full")
    parser.add_argument("--gpu-ids", nargs="+", default=["0"])
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--force-reevaluate", action="store_true")
    parser.add_argument("--skip-startup-debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if int(args.transfer_epoch_sample_size) < 0:
        parser.error("--transfer-epoch-sample-size must be >= 0")

    base_config = _require_file(_resolve(args.base_config), "base config")
    split_root = _resolve(args.split_root)
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.mode in {"splits", "all"}:
        cmd = _build_split_cmd(args)
        if args.dry_run:
            print(" ".join(cmd))
        else:
            subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        if args.mode == "splits":
            return

    if args.mode == "all":
        for mode in ("pretrain", "zero_shot", "transfer", "aggregate"):
            cmd = [sys.executable, str((PROJECT_ROOT / "scripts" / "run_single_process_full_unseen_experiments.py").resolve())]
            for key in (
                "base_config", "input_split_root", "split_root", "merged_csv", "output_root",
                "max_epochs_pretrain", "max_epochs_transfer", "monitor_metric", "monitor_mode",
                "seed", "fold_seed_mode", "finetune_mode", "max_parallel",
                "transfer_epoch_sample_size",
            ):
                name = key.replace("_", "-")
                value = getattr(args, key)
                cmd.extend([f"--{name}", str(value)])
            cmd.extend(["--heldout-processes", *[str(int(p)) for p in args.heldout_processes]])
            cmd.extend(["--folds", *[str(int(f)) for f in args.folds]])
            cmd.extend(["--pretrain-source-fold", str(int(args.pretrain_source_fold))])
            cmd.extend(["--gpu-ids", *[str(g) for g in args.gpu_ids]])
            cmd.extend(["--mode", mode])
            if args.pretrain_learning_rate is not None:
                cmd.extend(["--pretrain-learning-rate", str(float(args.pretrain_learning_rate))])
            if args.transfer_learning_rate is not None:
                cmd.extend(["--transfer-learning-rate", str(float(args.transfer_learning_rate))])
            if args.batch_size is not None:
                cmd.extend(["--batch-size", str(int(args.batch_size))])
            if args.resume_existing:
                cmd.append("--resume-existing")
            if args.force_reevaluate:
                cmd.append("--force-reevaluate")
            if args.skip_startup_debug:
                cmd.append("--skip-startup-debug")
            if args.dry_run:
                cmd.append("--dry-run")
            subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        return

    if args.mode == "aggregate":
        aggregate(output_root)
        return

    tasks: list[Task] = []
    for heldout in [int(p) for p in args.heldout_processes]:
        hlabel = _heldout_label(heldout)
        source_ids = _source_ids(heldout)
        split_dir = split_root / hlabel
        source_train = _require_file(split_dir / "source_train.csv", f"{hlabel} source_train")
        source_val = _require_file(split_dir / "source_val.csv", f"{hlabel} source_val")
        pretrain_dir = output_root / hlabel / "pretrain"
        pretrain_runtime = pretrain_dir / "runtime_overrides.json"
        if args.mode == "pretrain":
            _write_runtime_overrides(
                pretrain_runtime,
                seed=int(args.seed),
                experiment_name=f"single_unseen_pretrain_P{heldout:02d}",
                output_dir=pretrain_dir,
                merged_csv=str(args.merged_csv),
                process_ids=source_ids,
                train_manifest=source_train,
                val_manifest=source_val,
                test_manifest=source_val,
                max_epochs=int(args.max_epochs_pretrain),
                monitor_metric=str(args.monitor_metric),
                monitor_mode=str(args.monitor_mode),
                learning_rate=args.pretrain_learning_rate,
                batch_size=args.batch_size,
                epoch_sampler=None,
            )
            tasks.append(
                Task(
                    name=f"P{heldout:02d}_pretrain",
                    kind="train",
                    run_dir=pretrain_dir,
                    cmd=_train_cmd(
                        base_config=base_config,
                        runtime_path=pretrain_runtime,
                        max_epochs=int(args.max_epochs_pretrain),
                        pretrained=None,
                        finetune_mode="full",
                        skip_startup_debug=bool(args.skip_startup_debug),
                    ),
                    metadata={"stage": "pretrain", "heldout_process": heldout, "source_process_ids": source_ids},
                )
            )
            continue

        checkpoint = _latest_completed_checkpoint(pretrain_dir)
        if checkpoint is None and args.dry_run:
            checkpoint = pretrain_dir / "checkpoints" / "<RUN>" / "best.pt"
        if checkpoint is None:
            raise FileNotFoundError(f"pretrain checkpoint not found for {hlabel}: {pretrain_dir}")

        for fold in [int(f) for f in args.folds]:
            flabel = _fold_label(fold)
            target_train = _require_file(split_dir / flabel / "target_train.csv", f"{hlabel}/{flabel} target_train")
            target_val = _require_file(split_dir / flabel / "target_val.csv", f"{hlabel}/{flabel} target_val")
            target_test = _require_file(split_dir / flabel / "target_test.csv", f"{hlabel}/{flabel} target_test")
            fold_seed = int(args.seed) if args.fold_seed_mode == "fixed" else int(args.seed) + int(fold)
            if args.mode == "zero_shot":
                zero_dir = output_root / hlabel / flabel / "zero_shot"
                tasks.append(
                    Task(
                        name=f"P{heldout:02d}_F{fold:02d}_zero_shot",
                        kind="eval",
                        run_dir=zero_dir,
                        cmd=_eval_cmd(
                            base_config=base_config,
                            checkpoint=checkpoint,
                            source_train=source_train,
                            source_val=source_val,
                            target_test=target_test,
                            heldout=heldout,
                            fold=fold,
                            output_dir=zero_dir,
                            merged_csv=str(args.merged_csv),
                            source_ids=source_ids,
                            skip_startup_debug=bool(args.skip_startup_debug),
                            # _run_tasks prepares status.json/command.txt in this
                            # directory before launching the evaluator. Allow the
                            # evaluator to use that runner-owned non-empty directory;
                            # completed runs are still skipped by resume_existing.
                            force_reevaluate=True,
                        ),
                        metadata={"stage": "zero_shot", "heldout_process": heldout, "fold": fold, "checkpoint": _rel(checkpoint)},
                    )
                )
            elif args.mode == "transfer":
                transfer_dir = output_root / hlabel / flabel / "transfer_full"
                runtime_path = transfer_dir / "runtime_overrides.json"
                _write_runtime_overrides(
                    runtime_path,
                    seed=fold_seed,
                    experiment_name=f"single_unseen_transfer_P{heldout:02d}_F{fold:02d}",
                    output_dir=transfer_dir,
                    merged_csv=str(args.merged_csv),
                    process_ids=[heldout],
                    train_manifest=target_train,
                    val_manifest=target_val,
                    test_manifest=target_test,
                    max_epochs=int(args.max_epochs_transfer),
                    monitor_metric=str(args.monitor_metric),
                    monitor_mode=str(args.monitor_mode),
                    learning_rate=args.transfer_learning_rate,
                    batch_size=args.batch_size,
                    epoch_sampler=(
                        {
                            "enabled": True,
                            "mode": "random",
                            "base_epoch_size": int(args.transfer_epoch_sample_size),
                            "unique_within_epoch": True,
                            "replacement_for_uniform": False,
                            "mass_flow_tail": {"enabled": False},
                            "hard_target_edges": {"enabled": False},
                        }
                        if int(args.transfer_epoch_sample_size) > 0
                        else None
                    ),
                )
                tasks.append(
                    Task(
                        name=f"P{heldout:02d}_F{fold:02d}_transfer",
                        kind="train",
                        run_dir=transfer_dir,
                        cmd=_train_cmd(
                            base_config=base_config,
                            runtime_path=runtime_path,
                            max_epochs=int(args.max_epochs_transfer),
                            pretrained=checkpoint,
                            finetune_mode=str(args.finetune_mode),
                            skip_startup_debug=bool(args.skip_startup_debug),
                        ),
                        metadata={
                            "stage": "transfer",
                            "heldout_process": heldout,
                            "fold": fold,
                            "checkpoint": _rel(checkpoint),
                            "epoch_sampler": (
                                {
                                    "mode": "random_without_replacement",
                                    "epoch_sample_size": int(args.transfer_epoch_sample_size),
                                }
                                if int(args.transfer_epoch_sample_size) > 0
                                else {"mode": "inherit_base_config"}
                            ),
                        },
                    )
                )

    _run_tasks(
        tasks,
        gpu_ids=[str(g) for g in args.gpu_ids],
        max_parallel=int(args.max_parallel),
        resume_existing=bool(args.resume_existing),
        force_reevaluate=bool(args.force_reevaluate),
        dry_run=bool(args.dry_run),
    )


if __name__ == "__main__":
    main()
