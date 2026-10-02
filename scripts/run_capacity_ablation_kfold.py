#!/usr/bin/env python3
"""Sequential capacity-ablation K-fold launcher (per experiment × process)."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_EXPERIMENTS: list[str] = [
    "exp_L3",
    "exp_L4",
    "exp_L5",
    "exp_L6",
    "exp_H384",
    "exp_H448",
    "exp_H512",
    "exp_A256",
    "exp_A384",
    "exp_E_small",
    "exp_E_base",
    "exp_E_large",
    "exp_M_base",
    "exp_M_shallow",
    "exp_M_update_deep",
    "exp_M_input_deep",
]

# exp_A512 is identical to exp_H512; map alias and dedupe when both requested.
DEDUP_ALIAS: dict[str, str] = {"exp_A512": "exp_H512"}
DEDUP_EQUIVALENT: frozenset[str] = frozenset({"exp_H512", "exp_A512"})

_BASE_V3 = PROJECT_ROOT / "configs/experiment/process_surrogate_edge_all_v3.yaml"
_KFOLD_SCRIPT = PROJECT_ROOT / "scripts/run_process_kfold_experiments.py"
_CONFIG_DIR = PROJECT_ROOT / "configs/experiment/capacity_ablation"
_RUN_DIR_RE = re.compile(r"^process_kfold_P(\d+)_F(\d+)(?:-\d{8}-\d{6})?$")


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


def _resolve_experiments(names: list[str] | None) -> list[str]:
    raw = list(DEFAULT_EXPERIMENTS if names is None else names)
    resolved: list[str] = []
    seen_canonical: set[str] = set()
    for name in raw:
        canonical = DEDUP_ALIAS.get(name, name)
        if canonical in DEDUP_EQUIVALENT:
            if canonical in seen_canonical:
                print(f"[dedupe] skip duplicate {name} (same as exp_H512)", file=sys.stderr)
                continue
            seen_canonical.add(canonical)
        elif canonical in seen_canonical:
            print(f"[dedupe] skip duplicate {name}", file=sys.stderr)
            continue
        else:
            seen_canonical.add(canonical)
        resolved.append(canonical)
    return resolved


def _config_path(exp_name: str) -> Path:
    return (_CONFIG_DIR / f"{exp_name}.yaml").resolve()


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
        return {
            "experiment_name": exp.experiment_name,
            "model_config": str(exp.model_config),
            "train_config": str(exp.train_config),
            "data_config": str(exp.data_config),
            "model": {
                k: getattr(exp.model, k)
                for k in (
                    "hidden_dim",
                    "num_layers",
                    "attn_hidden_dim",
                    "role_emb_dim",
                    "unit_emb_dim",
                    "use_hx_role_embedding",
                    "input_mlp_layers",
                    "diff_mlp_layers",
                    "update_mlp_layers",
                    "final_mlp_layers",
                    "use_edge_decoder",
                    "use_edge_features",
                )
                if hasattr(exp.model, k)
            },
            "data_task_mode": getattr(exp.data, "task_mode", None),
        }
    except Exception as exc:
        return {"load_error": str(exc)}


def _run_subprocess_logged(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    tee: bool,
) -> int:
    """Run subprocess; optionally mirror stdout+stderr to terminal while writing log files."""
    if not tee:
        with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
            proc = subprocess.run(cmd, cwd=str(cwd), stdout=out_f, stderr=err_f, check=False)
            return int(proc.returncode)

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as out_f, stderr_path.open("w", encoding="utf-8") as err_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            out_f.write(line)
            err_f.write(line)
        return int(proc.wait())


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
    parser = argparse.ArgumentParser(description="Run capacity-ablation K-fold experiments sequentially.")
    parser.add_argument(
        "--process-ids",
        nargs="+",
        default=["1-10"],
        help="Process IDs (e.g. 1 5 9 or 1-10).",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help=f"Experiment names (default: all {len(DEFAULT_EXPERIMENTS)} configs).",
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/process_kfold_capacity_ablation",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--only-folds",
        type=int,
        nargs="*",
        default=None,
        help="1-based fold numbers passed to run_process_kfold_experiments (default: all).",
    )
    parser.add_argument(
        "--no-skip-startup-debug",
        action="store_true",
        help="Run global x_oper/target/tabular audits before each k-fold job (slow).",
    )
    parser.add_argument(
        "--force-reevaluate",
        action="store_true",
        help="Pass --force-reevaluate to k-fold runner (retrain all folds).",
    )
    parser.add_argument(
        "--force-recompute-metrics",
        action="store_true",
        help="Pass --force-recompute-metrics; stale folds get eval-only metric refresh.",
    )
    parser.add_argument(
        "--tee-logs",
        action="store_true",
        help=(
            "Mirror each job's stdout/stderr to the terminal while still writing "
            "Process*/run.log and run.stderr.log (loss / val metrics from train script)."
        ),
    )
    args = parser.parse_args()

    if not _BASE_V3.is_file():
        print(f"[error] missing baseline config: {_BASE_V3}", file=sys.stderr)
        return 1
    if not _KFOLD_SCRIPT.is_file():
        print(f"[error] missing kfold script: {_KFOLD_SCRIPT}", file=sys.stderr)
        return 1

    try:
        process_ids = _parse_process_ids(args.process_ids)
        _validate_process_ids(process_ids)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    experiments = _resolve_experiments(args.experiments)
    output_root = (PROJECT_ROOT / args.output_root).resolve()
    cuda_visible = __import__("os").environ.get("CUDA_VISIBLE_DEVICES")

    jobs: list[tuple[str, int, list[str], Path]] = []
    for exp_name in experiments:
        cfg = _config_path(exp_name)
        if not cfg.is_file():
            print(f"[error] missing experiment config: {cfg}", file=sys.stderr)
            return 1
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
                    "reason": "all_folds_have_primary_targetrow_metric",
                    "primary_metric_name": "process_balanced_main_target_r2_main_verified",
                    "output_root": str(exp_out),
                }
                (proc_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                if args.dry_run:
                    print(f"# skip-existing: {exp_name} Process{pid} (all folds complete)")
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
        return 0

    print(
        "[capacity-ablation] primary metric: process_balanced_main_target_r2_main_verified "
        "(per-process target-row mean). Auxiliary: target_balanced + legacy_answer_fraction_r2. "
        "Stale --skip-existing runs auto-recompute metrics when checkpoint exists.",
        flush=True,
    )

    failures = 0
    for exp_name, pid, cmd, cfg in jobs:
        proc_dir = output_root / exp_name / _process_label(pid)
        proc_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = proc_dir / "run.log"
        stderr_path = proc_dir / "run.stderr.log"
        start = datetime.now(timezone.utc)
        resolved = _load_resolved_config_snapshot(cfg)
        meta: dict = {
            "experiment_name": exp_name,
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
            "resolved_config": resolved,
        }
        (proc_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[run] {exp_name} Process{pid}", flush=True)
        print("  " + " ".join(cmd), flush=True)
        proc_rc = _run_subprocess_logged(
            cmd,
            cwd=PROJECT_ROOT,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            tee=bool(args.tee_logs),
        )
        end = datetime.now(timezone.utc)
        elapsed = (end - start).total_seconds()
        meta["end_time"] = end.isoformat()
        meta["elapsed_seconds"] = float(elapsed)
        meta["return_code"] = int(proc_rc)
        meta["status"] = "success" if proc_rc == 0 else "failed"
        meta["tee_logs"] = bool(args.tee_logs)
        (proc_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if proc_rc != 0:
            failures += 1
            print(f"[fail] {exp_name} Process{pid} rc={proc_rc}", file=sys.stderr)
            if not args.continue_on_error:
                return int(proc_rc)
    if failures:
        print(f"[done] completed with {failures} failure(s)", file=sys.stderr)
        return 1
    print("[done] all jobs completed successfully", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
