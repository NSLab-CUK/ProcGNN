#!/usr/bin/env python3
"""Re-build ``optuna_trial_metrics.json`` and study summaries using weighted ``answer_targets_r2``.

Use after switching ``tune_optuna.py`` back to the legacy H2/CO2 weighted objective.
Does not re-train; reads each trial's ``train_val_history.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tune_optuna import (  # noqa: E402
    _flatten_trial_for_table,
    _save_optuna_study_artifacts,
    build_optuna_trial_report,
)


def _trial_weights(trial: optuna.trial.FrozenTrial, *, fallback_h2: float, fallback_co2: float) -> tuple[float, float]:
    params = dict(trial.params)
    w_h2 = float(params.get("answer_edge_weight_h2", trial.user_attrs.get("answer_edge_weight_h2", fallback_h2)))
    w_co2 = float(params.get("answer_edge_weight_co2", trial.user_attrs.get("answer_edge_weight_co2", fallback_co2)))
    return w_h2, w_co2


def _trial_number_from_run_name(run_name: str) -> int | None:
    parts = run_name.split("_")
    if len(parts) >= 2 and parts[0] == "trial" and parts[1].isdigit():
        return int(parts[1])
    return None


def _write_summary_from_disk(proc_dir: Path, study: optuna.Study | None) -> None:
    """Build ``optuna_trials_summary.csv`` from on-disk ``optuna_trial_metrics.json``."""
    trial_by_number: dict[int, optuna.trial.FrozenTrial] = {}
    if study is not None:
        for t in study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            trial_by_number[int(t.number)] = t

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(proc_dir.glob("trial_*/")):
        mj = run_dir / "optuna_trial_metrics.json"
        if not mj.is_file():
            continue
        try:
            report = json.loads(mj.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        tnum = _trial_number_from_run_name(run_dir.name)
        t = trial_by_number.get(int(tnum)) if tnum is not None else None
        if t is None:
            class _Stub:
                number = int(tnum) if tnum is not None else -1
                state = optuna.trial.TrialState.COMPLETE
                params: dict[str, Any] = {}
                user_attrs = {"run_dir": str(run_dir)}

            t = _Stub()  # type: ignore[assignment]
        rows.append(_flatten_trial_for_table(t, report))  # type: ignore[arg-type]

    if not rows:
        print(f"[warn] no trial metrics under {proc_dir}")
        return
    import pandas as pd

    df = pd.DataFrame(rows)
    csv_path = proc_dir / "optuna_trials_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    sort_col = "objective_answer_targets_r2"
    top_n = min(15, len(df))
    if sort_col in df.columns:
        top = df.sort_values(sort_col, ascending=False).head(top_n)
    else:
        top = df.head(top_n)
    try:
        table = top.to_markdown(index=False)
    except Exception:
        table = top.to_string(index=False)
    (proc_dir / "optuna_trials_top.md").write_text(
        f"# Optuna trials (top {top_n} by {sort_col})\n\n{table}\n",
        encoding="utf-8",
    )
    print(f"[optuna] wrote {csv_path}")


def _regenerate_from_study(
    proc_dir: Path,
    *,
    preset: str,
    fallback_h2: float,
    fallback_co2: float,
) -> int:
    db = proc_dir / "study.db"
    study: optuna.Study | None = None
    if db.is_file():
        storage = f"sqlite:///{db.resolve().as_posix()}"
        summaries = list(optuna.study.get_all_study_summaries(storage=storage))
        if summaries:
            study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)
    n_ok = 0
    if study is not None:
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            run_dir_s = trial.user_attrs.get("run_dir")
            if not run_dir_s:
                continue
            run_dir = Path(str(run_dir_s))
            if not (run_dir / "train_val_history.json").is_file():
                continue
            w_h2, w_co2 = _trial_weights(trial, fallback_h2=fallback_h2, fallback_co2=fallback_co2)
            try:
                report = build_optuna_trial_report(run_dir, preset=preset, w_h2=w_h2, w_co2=w_co2)
            except Exception as exc:
                print(f"[warn] trial {trial.number} {run_dir.name}: {exc}")
                continue
            (run_dir / "optuna_trial_metrics.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            n_ok += 1
    else:
        n_ok = _regenerate_from_dirs(
            proc_dir, preset=preset, fallback_h2=fallback_h2, fallback_co2=fallback_co2
        )
    _write_summary_from_disk(proc_dir, study)
    if study is not None:
        try:
            _save_optuna_study_artifacts(study, proc_dir, no_plots=True)
        except OSError as exc:
            print(f"[warn] optional study artifacts skipped for {proc_dir.name}: {exc}")
    return n_ok


def _regenerate_from_dirs(
    proc_dir: Path,
    *,
    preset: str,
    fallback_h2: float,
    fallback_co2: float,
) -> int:
    n_ok = 0
    for run_dir in sorted(proc_dir.glob("trial_*/")):
        hist = run_dir / "train_val_history.json"
        if not hist.is_file():
            continue
        w_h2, w_co2 = fallback_h2, fallback_co2
        failed = run_dir / "optuna_trial_failed.json"
        if failed.is_file():
            try:
                payload = json.loads(failed.read_text(encoding="utf-8"))
                flat = payload.get("suggested_params_flat") or {}
                w_h2 = float(flat.get("answer_edge_weight_h2", w_h2))
                w_co2 = float(flat.get("answer_edge_weight_co2", w_co2))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
        try:
            report = build_optuna_trial_report(run_dir, preset=preset, w_h2=w_h2, w_co2=w_co2)
        except Exception as exc:
            print(f"[warn] {run_dir.name}: {exc}")
            continue
        (run_dir / "optuna_trial_metrics.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        n_ok += 1
    return n_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "optuna_0512",
        help="optuna output root (contains ProcessN/)",
    )
    parser.add_argument("--preset", default="edge_all", choices=["edge_all", "multitask"])
    parser.add_argument("--metric-w-h2", type=float, default=5.0)
    parser.add_argument("--metric-w-co2", type=float, default=3.0)
    args = parser.parse_args()
    root = args.root.resolve()
    total = 0
    for proc_dir in sorted(root.glob("Process*")):
        if not proc_dir.is_dir():
            continue
        n = _regenerate_from_study(
            proc_dir,
            preset=str(args.preset),
            fallback_h2=float(args.metric_w_h2),
            fallback_co2=float(args.metric_w_co2),
        )
        print(f"[ok] {proc_dir.name}: regenerated {n} trial reports")
        total += n
    print(f"[done] total trials regenerated: {total}")


if __name__ == "__main__":
    main()
