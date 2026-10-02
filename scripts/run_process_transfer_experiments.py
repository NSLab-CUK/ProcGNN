#!/usr/bin/env python3
"""Run one held-out-process transfer experiment stage."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _process_label(pid: int) -> str:
    return f"Process{int(pid)}"


def _ratio_label(ratio: float) -> str:
    return f"ratio_{int(round(float(ratio) * 100.0)):02d}"


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_runtime_overrides(
    path: Path,
    *,
    seed: int,
    device: str,
    experiment_name: str,
    output_dir: Path,
    merged_csv: str,
    process_ids: Sequence[int],
    train_manifest: Path,
    val_manifest: Path,
    test_manifest: Path,
    max_epochs: int,
    learning_rate: float | None,
    batch_size: int | None,
) -> None:
    train_payload: dict[str, object] = {}
    if max_epochs > 0:
        train_payload["epochs"] = int(max_epochs)
    if learning_rate is not None:
        train_payload["learning_rate"] = float(learning_rate)
    if batch_size is not None:
        train_payload["batch_size"] = int(batch_size)
    payload: dict[str, object] = {
        "seed": int(seed),
        "device": str(device),
        "experiment_name": experiment_name,
        "output_dir": _rel(output_dir),
        "save_dir": f"{_rel(output_dir)}/checkpoints",
        "log_dir": f"{_rel(output_dir)}/logs",
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
    if train_payload:
        payload["train"] = train_payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _latest_checkpoint(stage_dir: Path) -> Path | None:
    if not stage_dir.is_dir():
        return None
    candidates = sorted(
        stage_dir.rglob("checkpoints/*/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    candidates = sorted(stage_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run held-out process transfer-learning experiments.")
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument("--heldout-process", type=int, required=True)
    parser.add_argument("--source-process-ids", type=int, nargs="+", required=True)
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20, 0.50])
    parser.add_argument("--split-root", type=str, required=True)
    parser.add_argument("--merged-csv", type=str, default="data/datasets_v3/process_main_merged_all_processes_10pct.csv")
    parser.add_argument("--pretrained-checkpoint", type=str, default="")
    parser.add_argument("--mode", type=str, choices=("pretrain", "finetune", "scratch"), required=True)
    parser.add_argument("--finetune-mode", type=str, choices=("full", "head_only"), default="full")
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=str, default="outputs/transfer")
    parser.add_argument("--k-folds", type=int, default=1, help="Number of transfer folds in split-root. 1 keeps the legacy layout.")
    parser.add_argument("--only-folds", type=int, nargs="+", default=None, help="Optional 1-based fold IDs to run when --k-folds > 1.")
    parser.add_argument("--force-reevaluate", action="store_true")
    parser.add_argument("--skip-startup-debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_cfg = _require_file((PROJECT_ROOT / args.base_config).resolve(), "base config")
    split_root = (PROJECT_ROOT / args.split_root).resolve()
    if not split_root.is_dir():
        raise FileNotFoundError(f"split root not found: {split_root}")
    out_root = (PROJECT_ROOT / args.output_root).resolve()
    heldout = int(args.heldout_process)
    heldout_dir = out_root / f"heldout_P{heldout:02d}"
    source_tag = "source_" + "_".join(str(int(p)) for p in args.source_process_ids)

    commands: list[list[str]] = []
    planned: list[dict[str, object]] = []

    def add_train_command(
        *,
        stage: str,
        output_dir: Path,
        process_ids: Sequence[int],
        train_manifest: Path,
        val_manifest: Path,
        test_manifest: Path,
        pretrained_checkpoint: Path | None,
        finetune_mode: str,
        fold: int | None = None,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if output_dir.exists() and any(output_dir.iterdir()) and not args.force_reevaluate and not args.dry_run:
            print(f"[transfer][skip-existing-dir] {output_dir} (use --force-reevaluate to run anyway)", flush=True)
            return
        overrides = output_dir / "runtime_overrides.json"
        fold_tag = f"_F{int(fold):02d}" if fold is not None else ""
        experiment_name = f"process_transfer_{stage}_P{heldout:02d}{fold_tag}"
        _write_runtime_overrides(
            overrides,
            seed=int(args.seed),
            device=str(args.device),
            experiment_name=experiment_name,
            output_dir=output_dir,
            merged_csv=str(Path(args.merged_csv).as_posix()),
            process_ids=process_ids,
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            test_manifest=test_manifest,
            max_epochs=int(args.max_epochs),
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
        metadata = {
            "stage": stage,
            "fold": fold,
            "k_folds": int(args.k_folds),
            "heldout_process": heldout,
            "source_process_ids": [int(p) for p in args.source_process_ids],
            "process_ids": [int(p) for p in process_ids],
            "train_manifest": _rel(train_manifest),
            "val_manifest": _rel(val_manifest),
            "test_manifest": _rel(test_manifest),
            "pretrained_checkpoint": _rel(pretrained_checkpoint) if pretrained_checkpoint else "",
            "finetune_mode": finetune_mode,
            "normalization_policy": "target_adaptation" if stage != "pretrain" else "source_train",
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "max_epochs": int(args.max_epochs),
            "seed": int(args.seed),
        }
        (output_dir / "transfer_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        cmd = [
            sys.executable,
            str((PROJECT_ROOT / "scripts" / "train_process_surrogate.py").resolve()),
            "--config",
            _rel(base_cfg),
            "--runtime-overrides-file",
            _rel(overrides),
            "--max-epochs",
            str(int(args.max_epochs)),
            "--finetune-mode",
            finetune_mode,
        ]
        if pretrained_checkpoint is not None:
            cmd.extend(["--pretrained-checkpoint", _rel(pretrained_checkpoint), "--pretrained-load-mode", "model_only"])
        if args.skip_startup_debug:
            cmd.append("--skip-startup-debug")
        planned.append(metadata)
        commands.append(cmd)

    if int(args.k_folds) <= 1:
        fold_specs: list[tuple[int | None, Path, Path]] = [(None, split_root, heldout_dir)]
    else:
        fold_ids = [int(f) for f in (args.only_folds or list(range(1, int(args.k_folds) + 1)))]
        invalid = [f for f in fold_ids if f < 1 or f > int(args.k_folds)]
        if invalid:
            raise ValueError(f"--only-folds contains IDs outside 1..{args.k_folds}: {invalid}")
        fold_specs = [
            (fold, _require_file(split_root / f"fold_{fold:02d}" / "metadata.json", f"fold_{fold:02d} metadata").parent, heldout_dir / f"fold_{fold:02d}")
            for fold in fold_ids
        ]

    for fold, fold_split_root, fold_heldout_dir in fold_specs:
        if args.mode == "pretrain":
            add_train_command(
                stage="pretrain",
                output_dir=fold_heldout_dir / "pretrain" / source_tag,
                process_ids=[int(p) for p in args.source_process_ids],
                train_manifest=_require_file(fold_split_root / "source_train.csv", "source train manifest"),
                val_manifest=_require_file(fold_split_root / "source_val.csv", "source val manifest"),
                test_manifest=_require_file(fold_split_root / "source_val.csv", "source validation-as-test manifest"),
                pretrained_checkpoint=None,
                finetune_mode="full",
                fold=fold,
            )
        else:
            pretrained: Path | None = None
            if args.mode == "finetune":
                if str(args.pretrained_checkpoint).strip():
                    pretrained = _require_file((PROJECT_ROOT / args.pretrained_checkpoint).resolve(), "pretrained checkpoint")
                else:
                    pretrained = _latest_checkpoint(fold_heldout_dir / "pretrain" / source_tag)
                    if pretrained is None:
                        raise FileNotFoundError(
                            "No pretrained checkpoint supplied and no pretrain best.pt found under "
                            f"{fold_heldout_dir / 'pretrain' / source_tag}"
                        )
            for ratio in sorted(float(r) for r in args.ratios):
                rlabel = _ratio_label(ratio)
                rdir = fold_split_root / rlabel
                stage = f"{args.mode}_{args.finetune_mode}" if args.mode == "finetune" else "scratch"
                add_train_command(
                    stage=f"{rlabel}_{stage}",
                    output_dir=fold_heldout_dir / rlabel / stage,
                    process_ids=[heldout],
                    train_manifest=_require_file(rdir / "train.csv", f"{rlabel} train manifest"),
                    val_manifest=_require_file(rdir / "val.csv", f"{rlabel} val manifest"),
                    test_manifest=_require_file(fold_split_root / "target_test.csv", "target test manifest"),
                    pretrained_checkpoint=pretrained,
                    finetune_mode=str(args.finetune_mode if args.mode == "finetune" else "full"),
                    fold=fold,
                )

    if args.dry_run:
        for cmd in commands:
            print(" ".join(cmd))
        print(f"[transfer][dry-run] commands={len(commands)}")
        return

    print(f"[transfer] planned commands={len(commands)} mode={args.mode}")
    for cmd in commands:
        print("[transfer][run]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
