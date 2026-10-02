#!/usr/bin/env python3
"""Plot train/val loss curves from Optuna per-process tuning runs.

Reads ``outputs/optuna/Process{N}/trial_*/train_val_history.json`` (same layout as
``train_process_surrogate``). Optional ``study.db`` maps trials to objective values
and best trial.

Example::

    python scripts/plot_optuna_loss.py --output-root outputs/optuna
    python scripts/plot_optuna_loss.py --process 1 2 3 --plot-objective
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_TRIAL_DIR_RE = re.compile(r"^trial_(\d+)_")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _trial_sort_key(name: str) -> tuple[int, str]:
    m = _TRIAL_DIR_RE.match(name)
    if m:
        return (int(m.group(1)), name)
    return (10**9, name)


def _discover_trial_dirs(process_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(process_dir.iterdir(), key=lambda x: _trial_sort_key(x.name)):
        if not p.is_dir() or not p.name.startswith("trial_"):
            continue
        if (p / "train_val_history.json").is_file():
            out.append(p)
    return out


def _load_study_if_available(process_dir: Path) -> Any | None:
    db = process_dir / "study.db"
    if not db.is_file():
        return None
    try:
        import optuna
    except ImportError:
        print("[warn] optuna not installed; skipping objective / best-from-study plots", file=sys.stderr)
        return None

    storage = f"sqlite:///{db.as_posix()}"
    best_path = process_dir / "best_hyperparameters.json"
    study_name = ""
    if best_path.is_file():
        study_name = str(_load_json(best_path).get("study_name") or "").strip()
    if not study_name:
        summaries = optuna.study.get_all_study_summaries(storage)
        if len(summaries) == 1:
            study_name = summaries[0].study_name
        elif summaries:
            study_name = summaries[0].study_name
            print(
                f"[warn] multiple studies in {db}; using first: {study_name!r}",
                file=sys.stderr,
            )
    if not study_name:
        return None
    return optuna.load_study(study_name=study_name, storage=storage)


def _best_trial_number_from_files(process_dir: Path, study: Any | None) -> int | None:
    best_path = process_dir / "best_hyperparameters.json"
    if best_path.is_file():
        n = _load_json(best_path).get("best_trial_number")
        if n is not None:
            return int(n)
    if study is not None and study.best_trial is not None:
        return int(study.best_trial.number)
    return None


def _fallback_best_by_val_loss(series: dict[int, dict[str, Any]]) -> int | None:
    best_t: int | None = None
    best_v = float("inf")
    for tnum, s in series.items():
        vl = s.get("val_loss") or []
        if not vl:
            continue
        m = min(float(x) for x in vl)
        if m < best_v:
            best_v = m
            best_t = tnum
    return best_t


def _hist_val_loss_total(hist: dict[str, Any]) -> tuple[list[float], list[float]]:
    val = hist.get("val") or {}
    ep = [float(x) for x in (val.get("epoch") or [])]
    lo = [float(x) for x in (val.get("loss_total") or [])]
    n = min(len(ep), len(lo))
    return ep[:n], lo[:n]


def _hist_train_loss_total(hist: dict[str, Any]) -> tuple[list[float], list[float]]:
    train = hist.get("train") or {}
    ep = [float(x) for x in (train.get("epoch") or [])]
    lo = [float(x) for x in (train.get("loss_total") or [])]
    n = min(len(ep), len(lo))
    return ep[:n], lo[:n]


def _plot_process(
    process_dir: Path,
    *,
    plot_objective: bool,
    all_trials_alpha: float,
    dpi: int,
    out_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pf_match = re.search(r"Process(\d+)$", process_dir.name)
    pf = int(pf_match.group(1)) if pf_match else -1

    trial_dirs = _discover_trial_dirs(process_dir)
    if not trial_dirs:
        print(f"[skip] no trial_*/train_val_history.json under {process_dir}")
        return

    study = _load_study_if_available(process_dir)
    best_num = _best_trial_number_from_files(process_dir, study)

    # trial_number -> (val_ep, val_loss), (train_ep, train_loss)
    series: dict[int, dict[str, Any]] = {}
    for td in trial_dirs:
        m = _TRIAL_DIR_RE.match(td.name)
        if not m:
            continue
        tnum = int(m.group(1))
        hist = _load_json(td / "train_val_history.json")
        ve, vl = _hist_val_loss_total(hist)
        te, tl = _hist_train_loss_total(hist)
        series[tnum] = {
            "name": td.name,
            "val_ep": ve,
            "val_loss": vl,
            "train_ep": te,
            "train_loss": tl,
        }

    if best_num is None or best_num not in series:
        fb = _fallback_best_by_val_loss(series)
        if fb is not None:
            best_num = fb

    proc_out = out_dir / process_dir.name
    proc_out.mkdir(parents=True, exist_ok=True)

    # --- All trials: val loss_total ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for tnum in sorted(series.keys()):
        s = series[tnum]
        if not s["val_ep"]:
            continue
        is_best = best_num is not None and tnum == best_num
        ax.plot(
            s["val_ep"],
            s["val_loss"],
            alpha=1.0 if is_best else all_trials_alpha,
            linewidth=2.2 if is_best else 1.0,
            color="C3" if is_best else "C0",
            label=f"trial {tnum}" + (" (best)" if is_best else ""),
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("val loss_total")
    ax.set_title(f"Process {pf}: validation total loss (Optuna trials)")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) <= 20:
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(proc_out / "val_loss_total_all_trials.png", dpi=dpi)
    plt.close(fig)

    # --- Best trial: train vs val ---
    if best_num is not None and best_num in series:
        s = series[best_num]
        fig, ax = plt.subplots(figsize=(9, 5))
        if s["train_ep"] and s["train_loss"]:
            ax.plot(s["train_ep"], s["train_loss"], label="train (total)", color="C0", linewidth=1.5)
        if s["val_ep"] and s["val_loss"]:
            ax.plot(
                s["val_ep"],
                s["val_loss"],
                "o-",
                label="val (total)",
                color="C1",
                linewidth=1.5,
                markersize=4,
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(f"Process {pf}: best trial {best_num} (train vs val loss_total)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(proc_out / "loss_total_best_trial.png", dpi=dpi)
        plt.close(fig)
    else:
        print(
            f"[warn] {process_dir.name}: could not resolve best trial; skipped loss_total_best_trial.png",
            file=sys.stderr,
        )

    if plot_objective and study is not None:
        trials = [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]
        trials.sort(key=lambda t: t.number)
        if trials:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            xs = [t.number for t in trials]
            ys = [float(t.value) for t in trials]
            ax.scatter(xs, ys, s=36, alpha=0.85, c="C0", label="trial objective")
            best_y = min(ys)
            ax.axhline(best_y, color="C3", linestyle="--", linewidth=1.2, label=f"best = {best_y:.6g}")
            ax.set_xlabel("trial number")
            ax.set_ylabel("objective (min weighted val MAE)")
            ax.set_title(f"Process {pf}: Optuna objective (see tune_optuna.py metric)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(proc_out / "optuna_objective_per_trial.png", dpi=dpi)
            plt.close(fig)

    print(f"[plots] wrote under {proc_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot loss curves from Optuna tuning outputs.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/optuna",
        help="Root containing Process{{N}} directories (default: outputs/optuna)",
    )
    parser.add_argument(
        "--process",
        type=int,
        nargs="*",
        default=None,
        help="Process IDs to plot (default: all Process* under output-root)",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="",
        help="Where to save figures (default: <output-root>/loss_plots)",
    )
    parser.add_argument(
        "--plot-objective",
        action="store_true",
        help="Also save optuna_objective_per_trial.png when study.db exists",
    )
    parser.add_argument(
        "--all-trials-alpha",
        type=float,
        default=0.35,
        help="Line alpha for non-best trials in overlay plot",
    )
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    root = (PROJECT_ROOT / args.output_root).resolve()
    if not root.is_dir():
        raise SystemExit(f"output root not found: {root}")

    if args.plots_dir.strip():
        out_dir = (PROJECT_ROOT / args.plots_dir).resolve()
    else:
        out_dir = root / "loss_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.process:
        proc_dirs = [root / f"Process{p}" for p in args.process]
    else:
        proc_dirs = sorted(
            [p for p in root.iterdir() if p.is_dir() and re.match(r"^Process\d+$", p.name)],
            key=lambda p: int(re.search(r"(\d+)$", p.name).group(1)),
        )

    for pd in proc_dirs:
        if not pd.is_dir():
            print(f"[skip] missing directory: {pd}", file=sys.stderr)
            continue
        _plot_process(
            pd,
            plot_objective=args.plot_objective,
            all_trials_alpha=float(args.all_trials_alpha),
            dpi=int(args.dpi),
            out_dir=out_dir,
        )


if __name__ == "__main__":
    main()
