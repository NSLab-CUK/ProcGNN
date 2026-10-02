"""Run the target-edge sMAPE evaluator for every locally available Proposed run.

This is an inference-only launcher.  It uses each run's original
``runtime_overrides.json`` to recover the exact test manifest and process
scope, then evaluates the saved ``best.pt`` without changing training output.
Use ``--shard-index`` / ``--num-shards`` to distribute the independent jobs
across servers or GPUs.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Job:
    checkpoint: Path
    runtime_overrides: Path
    manifest: str
    output_dir: Path
    label: str
    config: Path | None = None


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _nearest_runtime_overrides(path: Path, stop: Path) -> Path | None:
    current = path.parent
    while current != current.parent:
        candidate = current / "runtime_overrides.json"
        if candidate.is_file():
            return candidate
        if current == stop:
            return None
        current = current.parent
    return None


def _runtime_manifest(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    data = payload.get("data", {})
    value = data.get("test_split_manifest_path") if isinstance(data, dict) else None
    return str(value) if value else None


def _latest(paths: list[Path]) -> Path:
    return max(paths, key=lambda path: path.stat().st_mtime_ns)


def _label(final_root: Path, runtime: Path) -> str:
    rel = runtime.relative_to(final_root).as_posix()
    if rel.startswith("proposed_single_process/"):
        family = "Single"
    elif rel.startswith("proposed_joint_10d_clean/"):
        family = "Joint"
    elif rel.startswith("proposed_unseen/"):
        family = "Zero-shot"
    elif rel.startswith("data_efficiency/"):
        family = "Transfer"
    elif rel.startswith("sensitivity_10d_clean/"):
        family = "Sensitivity"
    else:
        family = "Proposed"
    return f"{family}: {rel}"


def _config_for_runtime(final_root: Path, runtime: Path) -> Path | None:
    """Return the original resolved YAML for sensitivity checkpoints.

    Runtime overrides preserve data and training settings, but architectural
    changes (for example ``depth_1``) live only in the sensitivity sweep's
    resolved YAML.  Reconstructing such a checkpoint with the default model
    YAML makes strict state-dict loading fail.
    """
    sensitivity_root = final_root / "sensitivity_10d_clean"
    try:
        relative = runtime.parent.relative_to(sensitivity_root)
    except ValueError:
        return None
    parts = relative.parts
    if len(parts) < 2:
        return None
    # Depth runs are already named ``depth_1`` etc.; PIN runs are grouped as
    # ``pin/node_atom_low`` and therefore need the ``pin_`` prefix restored.
    setting_name = parts[1] if parts[0] == "depth" else f"{parts[0]}_{parts[1]}"
    candidate = sensitivity_root / "resolved_configs" / f"{setting_name}.yaml"
    if not candidate.is_file():
        raise FileNotFoundError(
            "Sensitivity checkpoint has no matching resolved config: "
            f"{candidate} (runtime: {runtime})"
        )
    return candidate


def _discover(
    final_root: Path,
    output_root: Path,
    *,
    sensitivity_only: bool = False,
) -> tuple[list[Job], list[dict[str, str]]]:
    candidate_roots = [
        final_root / "proposed_joint_10d_clean",
        final_root / "proposed_single_process",
        final_root / "data_efficiency",
        final_root / "sensitivity_10d_clean",
    ]
    if sensitivity_only:
        candidate_roots = [final_root / "sensitivity_10d_clean"]
    grouped: dict[Path, list[Path]] = {}
    ignored: list[dict[str, str]] = []
    for root in candidate_roots:
        if not root.is_dir():
            continue
        for checkpoint in root.rglob("best.pt"):
            runtime = _nearest_runtime_overrides(checkpoint, root)
            if runtime is None:
                ignored.append({"path": str(checkpoint), "reason": "runtime_overrides.json not found"})
                continue
            grouped.setdefault(runtime, []).append(checkpoint)

    jobs: list[Job] = []
    for runtime, checkpoints in grouped.items():
        manifest = _runtime_manifest(runtime)
        if not manifest:
            ignored.append({"path": str(runtime), "reason": "test_split_manifest_path not found"})
            continue
        checkpoint = _latest(checkpoints)
        rel = runtime.parent.relative_to(final_root)
        jobs.append(Job(
            checkpoint=checkpoint,
            runtime_overrides=runtime,
            manifest=manifest,
            output_dir=output_root / rel,
            label=_label(final_root, runtime.parent),
            config=_config_for_runtime(final_root, runtime),
        ))

    # A zero-shot model is the held-out process's pretraining checkpoint,
    # evaluated with each zero-shot runtime manifest.  It has no own best.pt.
    unseen = final_root / "proposed_unseen"
    if not sensitivity_only and unseen.is_dir():
        for runtime in unseen.rglob("zero_shot/runtime_overrides.json"):
            manifest = _runtime_manifest(runtime)
            pretrain_root = runtime.parents[2] / "pretrain" / "checkpoints"
            checkpoints = list(pretrain_root.rglob("best.pt")) if pretrain_root.is_dir() else []
            if not manifest or not checkpoints:
                ignored.append({"path": str(runtime), "reason": "zero-shot manifest or pretrain best.pt missing"})
                continue
            rel = runtime.parent.relative_to(final_root)
            jobs.append(Job(
                checkpoint=_latest(checkpoints),
                runtime_overrides=runtime,
                manifest=manifest,
                output_dir=output_root / rel,
                label=_label(final_root, runtime.parent),
            ))
    jobs.sort(key=lambda job: str(job.output_dir))
    return jobs, ignored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-hoc target-edge sMAPE on saved Proposed checkpoints.")
    parser.add_argument("--final-root", default="outputs/0819final")
    parser.add_argument("--output-root", default="outputs/0819final/posthoc_target_smape")
    parser.add_argument("--config", default="configs/experiment/pinn/model_260805_10d_frac1.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument(
        "--sensitivity-only",
        action="store_true",
        help="Evaluate only sensitivity_10d_clean checkpoints; avoids scanning unrelated completed runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= --shard-index < --num-shards.")
    final_root = _resolve(args.final_root)
    output_root = _resolve(args.output_root)
    jobs, ignored = _discover(final_root, output_root, sensitivity_only=args.sensitivity_only)
    output_root.mkdir(parents=True, exist_ok=True)
    plan_suffix = f"_shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    with (output_root / f"evaluation_plan{plan_suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["job_index", "shard", "label", "config", "checkpoint", "runtime_overrides", "manifest", "output_dir"])
        writer.writeheader()
        for index, job in enumerate(jobs):
            writer.writerow({"job_index": index, "shard": index % args.num_shards, "label": job.label, "config": job.config or args.config, "checkpoint": job.checkpoint, "runtime_overrides": job.runtime_overrides, "manifest": job.manifest, "output_dir": job.output_dir})
    with (output_root / f"ignored_checkpoints{plan_suffix}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "reason"])
        writer.writeheader()
        writer.writerows(ignored)

    selected = [(index, job) for index, job in enumerate(jobs) if index % args.num_shards == args.shard_index]
    print(f"[discover] total_jobs={len(jobs)} selected={len(selected)} shard={args.shard_index}/{args.num_shards}", flush=True)
    for position, (index, job) in enumerate(selected, start=1):
        command = [
            sys.executable, "scripts/evaluate_target_smape.py",
            "--config", str(job.config or args.config),
            "--checkpoint", str(job.checkpoint),
            "--manifest", job.manifest,
            "--runtime-overrides-file", str(job.runtime_overrides),
            "--output-dir", str(job.output_dir),
            "--model-label", job.label,
            "--device", args.device,
            "--batch-size", str(args.batch_size),
            "--resume-existing",
        ]
        print(f"[job {position}/{len(selected)} id={index}] {job.label}", flush=True)
        if args.dry_run:
            print(" ".join(command), flush=True)
            continue
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    print(f"[done] selected_jobs={len(selected)} output={output_root}", flush=True)


if __name__ == "__main__":
    main()
