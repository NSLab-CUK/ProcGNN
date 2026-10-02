#!/usr/bin/env python3
"""Sequential method-ablation (one-factor-at-a-time) K-fold launcher."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from method_ablation_registry import (  # noqa: E402
    CATALOG_BY_NAME,
    changed_config_keys,
    is_baseline_equivalent,
    resolve_experiment_names,
)

_BASE_V3 = PROJECT_ROOT / "configs/experiment/process_surrogate_edge_all_v3.yaml"
_KFOLD_SCRIPT = PROJECT_ROOT / "scripts/run_process_kfold_experiments.py"
_CONFIG_DIR = PROJECT_ROOT / "configs/experiment/method_ablation"


def _parse_process_ids(values: list[str]) -> list[int]:
    out: list[int] = []
    for raw in values:
        s = str(raw).strip()
        if re.fullmatch(r"\d+\s*-\s*\d+", s):
            a, b = re.split(r"\s*-\s*", s, maxsplit=1)
            lo, hi = int(a), int(b)
            if lo > hi:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
            continue
        out.append(int(s))
    seen: set[int] = set()
    ordered: list[int] = []
    for pid in out:
        if pid not in seen:
            seen.add(pid)
            ordered.append(pid)
    return ordered


def _validate_process_ids(pids: list[int]) -> None:
    bad = [p for p in pids if p < 1 or p > 10]
    if bad:
        raise ValueError(f"process-id must be 1..10, got: {bad}")


def _config_path(exp_name: str) -> Path:
    for rel in (f"{exp_name}.yaml", f"optional/{exp_name}.yaml"):
        p = (_CONFIG_DIR / rel).resolve()
        if p.is_file():
            return p
    raise FileNotFoundError(f"config not found for {exp_name} under {_CONFIG_DIR}")


def _process_label(pid: int) -> str:
    return f"Process{int(pid)}"


def _find_run_dir(exp_root: Path, pid: int, fold: int) -> Path | None:
    fold_dir = exp_root / _process_label(pid) / f"fold_{int(fold):02d}"
    if not fold_dir.is_dir():
        return None
    prefix = f"process_kfold_P{int(pid):02d}_F{int(fold):02d}"
    matches = sorted(
        (c for c in fold_dir.iterdir() if c.is_dir() and c.name.startswith(prefix)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _fold_complete(run_dir: Path) -> bool:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from ablation_targetrow_common import validate_run_targetrow_complete

    return bool(validate_run_targetrow_complete(run_dir).get("ok"))


def _process_skippable(exp_root: Path, pid: int, k_folds: int) -> bool:
    for fold in range(1, k_folds + 1):
        run_dir = _find_run_dir(exp_root, pid, fold)
        if run_dir is None or not _fold_complete(run_dir):
            return False
    return True


def _git_commit() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _load_resolved_config_snapshot(config_path: Path) -> dict | None:
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from process_graph.experiment.loaders import load_experiment_config

        exp = load_experiment_config(config_path)
        entry = CATALOG_BY_NAME.get(config_path.stem, {})
        return {
            "experiment_name": exp.experiment_name,
            "ablation_group": entry.get("ablation_group"),
            "model_config": str(exp.model_config),
            "data_task_mode": getattr(exp.data, "task_mode", None),
            "model": {
                k: getattr(exp.model, k)
                for k in (
                    "hidden_dim",
                    "num_layers",
                    "attn_hidden_dim",
                    "diff_mode",
                    "fusion_mode",
                    "global_pool",
                    "use_edge_features",
                    "initial_residual_mode",
                    "initial_residual_alpha",
                    "layer_residual_mode",
                    "layer_residual_alpha",
                    "layer_residual_norm",
                    "use_final_projection",
                )
                if hasattr(exp.model, k)
            },
        }
    except Exception as exc:
        return {"load_error": str(exc)}


def _build_kfold_command(
    *,
    config_path: Path,
    process_id: int,
    k_folds: int,
    max_epochs: int,
    seed: int,
    device: str,
    output_root: Path,
    skip_existing: bool,
    only_folds: list[int] | None = None,
    skip_startup_debug: bool = True,
    force_reevaluate: bool = False,
    force_recompute_metrics: bool = False,
) -> list[str]:
    exp_name = config_path.stem
    exp_out = output_root / exp_name
    cmd = [
        sys.executable,
        str(_KFOLD_SCRIPT),
        "--base-config",
        str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "--process-ids",
        str(int(process_id)),
        "--k-folds",
        str(int(k_folds)),
        "--max-epochs",
        str(int(max_epochs)),
        "--output-root",
        str(exp_out.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "--seed",
        str(int(seed)),
        "--device",
        str(device),
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    if force_reevaluate:
        cmd.append("--force-reevaluate")
    if force_recompute_metrics:
        cmd.append("--force-recompute-metrics")
    if only_folds:
        cmd.extend(["--only-folds", *[str(int(f)) for f in only_folds]])
    if skip_startup_debug:
        cmd.append("--skip-startup-debug")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run method-ablation K-fold experiments sequentially.")
    parser.add_argument("--process-ids", nargs="+", default=["1-10"])
    parser.add_argument("--experiments", nargs="*", default=None)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-root", type=str, default="outputs/process_kfold_method_ablation")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force-reevaluate", action="store_true")
    parser.add_argument("--force-recompute-metrics", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--only-folds", type=int, nargs="*", default=None)
    parser.add_argument("--no-skip-startup-debug", action="store_true")
    args = parser.parse_args()

    if not _BASE_V3.is_file():
        print(f"[error] missing baseline reference: {_BASE_V3}", file=sys.stderr)
        return 1
    if not _KFOLD_SCRIPT.is_file():
        print(f"[error] missing kfold script: {_KFOLD_SCRIPT}", file=sys.stderr)
        return 1

    try:
        process_ids = _parse_process_ids(args.process_ids)
        _validate_process_ids(process_ids)
        experiments = resolve_experiment_names(args.experiments, include_optional=args.include_optional)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if not experiments:
        print("[error] no experiments to run after baseline-equivalent filtering", file=sys.stderr)
        return 1

    output_root = (PROJECT_ROOT / args.output_root).resolve()
    cuda_visible = __import__("os").environ.get("CUDA_VISIBLE_DEVICES")

    jobs: list[tuple[str, int, list[str], Path]] = []
    for exp_name in experiments:
        try:
            cfg = _config_path(exp_name)
        except FileNotFoundError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1
        entry = CATALOG_BY_NAME[exp_name]
        if args.dry_run:
            try:
                sys.path.insert(0, str(PROJECT_ROOT / "src"))
                from process_graph.experiment.loaders import load_experiment_config

                load_experiment_config(cfg)
            except Exception as exc:
                print(f"[error] config load failed for {cfg}: {exc}", file=sys.stderr)
                return 1
        for pid in process_ids:
            exp_out = output_root / exp_name
            if args.skip_existing and _process_skippable(exp_out, pid, args.k_folds):
                proc_dir = exp_out / _process_label(pid)
                proc_dir.mkdir(parents=True, exist_ok=True)
                meta = {
                    "experiment_name": exp_name,
                    "process_id": int(pid),
                    "status": "skipped",
                    "reason": "all_folds_complete",
                    "baseline_equivalent": is_baseline_equivalent(entry),
                    "changed_config_keys": changed_config_keys(entry),
                }
                (proc_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                if args.dry_run:
                    print(f"# skip-existing: {exp_name} Process{pid}")
                continue
            cmd = _build_kfold_command(
                config_path=cfg,
                process_id=pid,
                k_folds=args.k_folds,
                max_epochs=args.max_epochs,
                seed=args.seed,
                device=args.device,
                output_root=output_root,
                skip_existing=args.skip_existing,
                only_folds=args.only_folds,
                skip_startup_debug=not args.no_skip_startup_debug,
                force_reevaluate=args.force_reevaluate,
                force_recompute_metrics=args.force_recompute_metrics,
            )
            jobs.append((exp_name, pid, cmd, cfg))

    if args.dry_run:
        for exp_name, pid, cmd, _ in jobs:
            print(" ".join(cmd))
        print(
            f"[dry-run] total jobs: {len(jobs)} "
            f"(experiments={len(experiments)}, processes={len(process_ids)})",
            file=sys.stderr,
        )
        skipped_equiv = [
            e["name"]
            for e in CATALOG_BY_NAME.values()
            if e.get("baseline_equivalent_label") and e["name"] not in experiments
        ]
        if skipped_equiv:
            print(f"[dry-run] baseline-equivalent (not scheduled): {', '.join(sorted(set(skipped_equiv)))}", file=sys.stderr)
        return 0

    print(
        "[method-ablation] primary metric: process_balanced_main_target_r2_main_verified. "
        "See train banner / fold-summary in run.log.",
        flush=True,
    )

    failures = 0
    for exp_name, pid, cmd, cfg in jobs:
        entry = CATALOG_BY_NAME[exp_name]
        proc_dir = output_root / exp_name / _process_label(pid)
        proc_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = proc_dir / "run.log"
        stderr_path = proc_dir / "run.stderr.log"
        start = datetime.now(timezone.utc)
        meta: dict = {
            "experiment_name": exp_name,
            "ablation_group": entry.get("ablation_group"),
            "process_id": int(pid),
            "command": cmd,
            "base_config": str(cfg.relative_to(PROJECT_ROOT)),
            "output_root": str((output_root / exp_name).relative_to(PROJECT_ROOT)),
            "start_time": start.isoformat(),
            "end_time": None,
            "elapsed_seconds": None,
            "return_code": None,
            "status": "running",
            "git_commit": _git_commit(),
            "seed": int(args.seed),
            "max_epochs": int(args.max_epochs),
            "k_folds": int(args.k_folds),
            "device": str(args.device),
            "cuda_visible_devices": cuda_visible,
            "changed_config_keys": changed_config_keys(entry),
            "baseline_equivalent": is_baseline_equivalent(entry),
            "resolved_config": _load_resolved_config_snapshot(cfg),
        }
        (proc_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[run] {exp_name} Process{pid}", flush=True)
        with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
            proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), stdout=out_f, stderr=err_f, check=False)
        end = datetime.now(timezone.utc)
        meta["end_time"] = end.isoformat()
        meta["elapsed_seconds"] = float((end - start).total_seconds())
        meta["return_code"] = int(proc.returncode)
        meta["status"] = "success" if proc.returncode == 0 else "failed"
        (proc_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if proc.returncode != 0:
            failures += 1
            print(f"[fail] {exp_name} Process{pid} rc={proc.returncode}", file=sys.stderr)
            if not args.continue_on_error:
                return int(proc.returncode)
    if failures:
        print(f"[done] completed with {failures} failure(s)", file=sys.stderr)
        return 1
    print("[done] all jobs completed successfully", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
