#!/usr/bin/env python3
"""Run 5-fold unseen-process zero-shot experiments."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _fold_label(fold: int) -> str:
    return f"fold_{int(fold):02d}"


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


def _write_runtime_overrides(
    path: Path,
    *,
    seed: int,
    device: str,
    experiment_name: str,
    output_dir: Path,
    merged_csv: str,
    source_process_ids: Iterable[int],
    train_manifest: Path,
    val_manifest: Path,
    max_epochs: int,
    monitor_metric: str,
    monitor_mode: str,
) -> None:
    payload = {
        "seed": int(seed),
        "device": str(device),
        "experiment_name": experiment_name,
        "output_dir": _rel(output_dir),
        "save_dir": f"{_rel(output_dir)}/checkpoints",
        "train": {
            "epochs": int(max_epochs),
            "monitor_metric": str(monitor_metric),
            "monitor_mode": str(monitor_mode),
        },
        "data": {
            "train_data_path": str(Path(merged_csv).as_posix()),
            "val_data_path": str(Path(merged_csv).as_posix()),
            "test_data_path": str(Path(merged_csv).as_posix()),
            "edge_all_processes": [int(p) for p in source_process_ids],
            "train_split_manifest_path": _rel(train_manifest),
            "val_split_manifest_path": _rel(val_manifest),
            "test_split_manifest_path": _rel(val_manifest),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _latest_checkpoint(stage_dir: Path) -> Path | None:
    if not stage_dir.is_dir():
        return None
    candidates = sorted(stage_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _latest_metrics_json(eval_dir: Path) -> Path | None:
    if not eval_dir.is_dir():
        return None
    candidates = sorted(eval_dir.rglob("metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _run(cmd: list[str], *, dry_run: bool) -> None:
    if dry_run:
        print(" ".join(cmd))
        return
    print("[unseen][run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _extract_summary_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    candidates = {
        "Target mean R2": ["target_mean_r2", "val_target_mean_r2", "test/target_mean_r2"],
        "Flatten R2": [
            "target_edge_10d_r2_flatten",
            "val_target_edge_10d_r2_flatten",
            "test/target_edge_10d_r2_flatten",
        ],
        "Edge macro R2": [
            "target_edge_10d_r2_edge_macro",
            "val_target_edge_10d_r2_edge_macro",
            "test/target_edge_10d_r2_edge_macro",
        ],
        "All-edge mean R2": [
            "pi_all_edge_mean_r2",
            "all_edge_property_mean_r2",
            "test/pi_all_edge_mean_r2",
            "test/all_edge_property_mean_r2",
        ],
    }
    out: dict[str, float] = {}
    for label, keys in candidates.items():
        val = float("nan")
        for key in keys:
            if key in metrics:
                try:
                    val = float(metrics[key])
                    break
                except (TypeError, ValueError):
                    pass
        out[label] = val
    return out


def _read_property_csvs(eval_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_frames: list[pd.DataFrame] = []
    all_edge_frames: list[pd.DataFrame] = []
    for path in eval_dir.rglob("target_edge_property_metrics.csv"):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        target_frames.append(df)
    for pattern in ("pi_all_edge_property_r2.csv", "pi_all_edge_property_metrics.csv"):
        for path in eval_dir.rglob(pattern):
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            all_edge_frames.append(df)
    return (
        pd.concat(target_frames, ignore_index=True) if target_frames else pd.DataFrame(),
        pd.concat(all_edge_frames, ignore_index=True) if all_edge_frames else pd.DataFrame(),
    )


def _property_rows(df: pd.DataFrame, *, scope: str, fold: int) -> list[dict[str, object]]:
    if df.empty:
        return []
    prop_col = next((c for c in ("property", "property_name", "target_property") if c in df.columns), None)
    r2_col = next((c for c in ("R2", "r2", "property_r2", "property_r2_orig") if c in df.columns), None)
    if prop_col is None or r2_col is None:
        return []
    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        split = str(row.get("split", "test"))
        if split and split.lower() not in {"test", "nan"}:
            continue
        try:
            r2 = float(row[r2_col])
        except (TypeError, ValueError):
            r2 = float("nan")
        rows.append({"fold": int(fold), "scope": scope, "property": str(row[prop_col]), "r2": r2})
    return rows


def _summary(values: pd.Series) -> dict[str, float | int]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "valid_folds": 0}
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "valid_folds": int(len(vals)),
    }


def aggregate_results(
    *,
    output_root: Path,
    folds: list[int],
    allow_partial: bool,
) -> None:
    fold_rows: list[dict[str, object]] = []
    prop_rows: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for fold in folds:
        eval_dir = output_root / _fold_label(fold) / "zero_shot"
        metrics_path = _latest_metrics_json(eval_dir)
        if metrics_path is None:
            missing.append({"fold": fold, "reason": "missing metrics.json", "path": _rel(eval_dir)})
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row: dict[str, object] = {"fold": fold, "metrics_json": _rel(metrics_path)}
        row.update(_extract_summary_metrics(metrics))
        fold_rows.append(row)
        target_df, all_edge_df = _read_property_csvs(eval_dir)
        prop_rows.extend(_property_rows(target_df, scope="target", fold=fold))
        prop_rows.extend(_property_rows(all_edge_df, scope="all_edge", fold=fold))

    if missing and not allow_partial:
        raise RuntimeError(f"missing fold artifacts and --allow-partial-aggregation is false: {missing}")

    agg_dir = output_root / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(agg_dir / "fold_metrics.csv", index=False)
    metric_rows: list[dict[str, object]] = []
    for col in ["Target mean R2", "Flatten R2", "Edge macro R2", "All-edge mean R2"]:
        if col in fold_df.columns:
            metric_rows.append({"metric": col, **_summary(fold_df[col])})
    pd.DataFrame(metric_rows).to_csv(agg_dir / "metric_summary.csv", index=False)

    prop_df = pd.DataFrame(prop_rows)
    if not prop_df.empty:
        for scope, out_name in (("target", "target_property_summary.csv"), ("all_edge", "all_edge_property_summary.csv")):
            sub = prop_df[prop_df["scope"] == scope]
            rows = []
            for prop, grp in sub.groupby("property", sort=True):
                rows.append({"property": prop, **{f"{k}_r2" if k != "valid_folds" else k: v for k, v in _summary(grp["r2"]).items()}})
            pd.DataFrame(rows).to_csv(agg_dir / out_name, index=False)
        prop_df.to_csv(agg_dir / "fold_property_metrics.csv", index=False)
    else:
        pd.DataFrame().to_csv(agg_dir / "target_property_summary.csv", index=False)
        pd.DataFrame().to_csv(agg_dir / "all_edge_property_summary.csv", index=False)

    metadata = {
        "folds_requested": folds,
        "successful_folds": [int(r["fold"]) for r in fold_rows],
        "failed_folds": missing,
        "allow_partial_aggregation": bool(allow_partial),
        "valid_folds": len(fold_rows),
    }
    (agg_dir / "aggregate_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[unseen][aggregate] wrote {agg_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 5-fold unseen-process zero-shot experiments.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--heldout-process", type=int, required=True)
    parser.add_argument("--source-process-ids", type=int, nargs="+", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--split-root", required=True)
    parser.add_argument("--merged-csv", default="data/datasets_v3/process_main_merged_all_processes_10pct.csv")
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--monitor-metric", default="val_target_mean_r2")
    parser.add_argument("--monitor-mode", default="max")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold-seed-mode", choices=("fixed", "offset"), default="fixed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--pretrain-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--allow-partial-aggregation", action="store_true")
    parser.add_argument("--force-reevaluate", action="store_true")
    parser.add_argument("--skip-startup-debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_config = _require_file(_resolve(args.base_config), "base config")
    split_root = _resolve(args.split_root)
    output_root = _resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    folds = [int(f) for f in args.folds]

    if args.aggregate_only:
        aggregate_results(output_root=output_root, folds=folds, allow_partial=bool(args.allow_partial_aggregation))
        return

    exp_meta = {
        "heldout_process": int(args.heldout_process),
        "source_process_ids": [int(p) for p in args.source_process_ids],
        "folds": folds,
        "split_root": _rel(split_root),
        "base_config": _rel(base_config),
        "max_epochs": int(args.max_epochs),
        "monitor_metric": str(args.monitor_metric),
        "monitor_mode": str(args.monitor_mode),
        "seed": int(args.seed),
        "fold_seed_mode": str(args.fold_seed_mode),
    }
    (output_root / "experiment_metadata.json").write_text(json.dumps(exp_meta, indent=2), encoding="utf-8")

    for fold in folds:
        flabel = _fold_label(fold)
        fold_split = split_root / flabel
        source_train = _require_file(fold_split / "source_train.csv", f"{flabel} source_train")
        source_val = _require_file(fold_split / "source_val.csv", f"{flabel} source_val")
        heldout_test = _require_file(fold_split / "heldout_test.csv", f"{flabel} heldout_test")
        fold_out = output_root / flabel
        pretrain_dir = fold_out / "pretrain"
        zero_shot_dir = fold_out / "zero_shot"
        fold_seed = int(args.seed) if args.fold_seed_mode == "fixed" else int(args.seed) + int(fold)

        if not args.eval_only:
            runtime_path = pretrain_dir / "runtime_overrides.json"
            _write_runtime_overrides(
                runtime_path,
                seed=fold_seed,
                device=str(args.device),
                experiment_name=f"unseen_pretrain_P{int(args.heldout_process):02d}_F{fold:02d}",
                output_dir=pretrain_dir,
                merged_csv=str(Path(args.merged_csv).as_posix()),
                source_process_ids=args.source_process_ids,
                train_manifest=source_train,
                val_manifest=source_val,
                max_epochs=int(args.max_epochs),
                monitor_metric=str(args.monitor_metric),
                monitor_mode=str(args.monitor_mode),
            )
            cmd = [
                sys.executable,
                str((PROJECT_ROOT / "scripts" / "train_process_surrogate.py").resolve()),
                "--config",
                _rel(base_config),
                "--runtime-overrides-file",
                _rel(runtime_path),
                "--max-epochs",
                str(int(args.max_epochs)),
            ]
            if args.skip_startup_debug:
                cmd.append("--skip-startup-debug")
            _run(cmd, dry_run=bool(args.dry_run))

        if args.pretrain_only:
            continue

        checkpoint: Path | None
        if str(args.checkpoint).strip():
            checkpoint = _require_file(_resolve(args.checkpoint), "checkpoint")
        else:
            checkpoint = _latest_checkpoint(pretrain_dir)
            if checkpoint is None:
                if args.dry_run:
                    checkpoint = pretrain_dir / "checkpoints" / "<RUN>" / "best.pt"
                else:
                    raise FileNotFoundError(f"no best.pt found under {pretrain_dir}; run pretrain first or pass --checkpoint")
        eval_cmd = [
            sys.executable,
            str((PROJECT_ROOT / "scripts" / "eval_process_surrogate_edge_all.py").resolve()),
            "--base-config",
            _rel(base_config),
            "--checkpoint",
            _rel(checkpoint),
            "--test-manifest",
            _rel(heldout_test),
            "--source-train-manifest",
            _rel(source_train),
            "--source-val-manifest",
            _rel(source_val),
            "--heldout-process",
            str(int(args.heldout_process)),
            "--fold",
            str(int(fold)),
            "--output-dir",
            _rel(zero_shot_dir),
            "--merged-csv",
            str(Path(args.merged_csv).as_posix()),
            "--source-process-ids",
            *[str(int(p)) for p in args.source_process_ids],
            "--device",
            str(args.device),
        ]
        if args.force_reevaluate:
            eval_cmd.append("--force-reevaluate")
        if args.skip_startup_debug:
            eval_cmd.append("--skip-startup-debug")
        _run(eval_cmd, dry_run=bool(args.dry_run))

    if not args.pretrain_only and not args.eval_only and not args.dry_run:
        aggregate_results(output_root=output_root, folds=folds, allow_partial=bool(args.allow_partial_aggregation))


if __name__ == "__main__":
    main()
