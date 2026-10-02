#!/usr/bin/env python3
"""Launch per-process K-fold training runs via train_process_surrogate.py + runtime overrides."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _id_key(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        fv = float(text)
        if fv.is_integer():
            return str(int(fv))
    except (TypeError, ValueError):
        pass
    return text


def _process_label(pid: int) -> str:
    return f"Process{int(pid)}"


def _run_name(pid: int, fold: int) -> str:
    return f"process_kfold_P{int(pid):02d}_F{int(fold):02d}"


def _joint_run_name(fold: int) -> str:
    return f"process_kfold_All_F{int(fold):02d}"


def _fold_dir_name(fold: int) -> str:
    return f"fold_{int(fold):02d}"


def _materialize_process_views_from_joint_manifests(
    *,
    joint_root: Path,
    destination: Path,
    process_ids: Sequence[int],
    folds: Sequence[int],
) -> Path:
    """Create per-process views without changing the canonical joint split rows."""
    view_root = destination / "process_split_views"
    for fold in folds:
        fold_name = _fold_dir_name(int(fold))
        for split_name in ("train", "val", "test"):
            source = joint_root / fold_name / f"{split_name}.csv"
            if not source.is_file():
                raise FileNotFoundError(f"Missing joint split manifest: {source}")
            frame = pd.read_csv(source)
            if "process_id" not in frame.columns:
                raise KeyError(f"missing process_id column: {source}")
            process_numbers = (
                frame["process_id"].astype(str).str.extract(r"(\d+)", expand=False).astype(int)
            )
            for pid in process_ids:
                subset = frame.loc[process_numbers.eq(int(pid))].copy()
                if subset.empty:
                    raise ValueError(f"no Process{int(pid)} rows in {source}")
                target = view_root / _process_label(int(pid)) / fold_name / f"{split_name}.csv"
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_file():
                    try:
                        existing = pd.read_csv(target)
                        if existing.equals(subset.reset_index(drop=True)):
                            continue
                    except Exception:
                        pass
                subset.to_csv(target, index=False)
    print(
        f"[process-manifest] materialized canonical per-process views: {view_root}",
        flush=True,
    )
    return view_root


def _write_joint_manifest(
    *,
    out_path: Path,
    splits_root: Path,
    stream_data_dir: Path,
    process_ids: Sequence[int],
    fold: int,
    split_name: str,
) -> None:
    frames: list[pd.DataFrame] = []
    dropped_total = 0
    for pid in process_ids:
        path = splits_root / _process_label(int(pid)) / _fold_dir_name(fold) / f"{split_name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing split manifest: {path}")
        frame = pd.read_csv(path)
        if "sample_id" in frame.columns:
            stream_path = stream_data_dir / f"{int(pid)}.Process_Streams.csv"
            if not stream_path.is_file():
                raise FileNotFoundError(f"Process stream CSV not found: {stream_path}")
            stream_ids = set(pd.read_csv(stream_path, usecols=["ID"])["ID"].map(_id_key).tolist())
            stream_ids.discard("")
            keep = frame["sample_id"].map(_id_key).isin(stream_ids)
            dropped = int((~keep).sum())
            if dropped:
                dropped_total += dropped
                frame = frame[keep].copy()
        frames.append(frame)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    if dropped_total:
        print(
            f"[joint-manifest][warn] {out_path}: dropped {dropped_total} rows without matching stream ID",
            flush=True,
        )


def _write_runtime_overrides(
    path: Path,
    *,
    seed: int,
    device: str,
    experiment_name: str,
    output_dir: str,
    save_dir: str,
    log_dir: str,
    merged_rel: str,
    process_ids: Sequence[int],
    train_manifest: str,
    val_manifest: str,
    test_manifest: str,
    train_epochs: int | None,
    train_overrides: dict | None = None,
) -> None:
    data: dict = {
        "train_data_path": merged_rel,
        "val_data_path": merged_rel,
        "test_data_path": merged_rel,
        "edge_all_processes": [int(p) for p in process_ids],
        "train_split_manifest_path": train_manifest,
        "val_split_manifest_path": val_manifest,
        "test_split_manifest_path": test_manifest,
    }
    payload: dict = {
        "seed": int(seed),
        "device": str(device),
        "experiment_name": str(experiment_name),
        "output_dir": str(output_dir),
        "save_dir": str(save_dir),
        "log_dir": str(log_dir),
        "data": data,
    }
    train_payload: dict = dict(train_overrides or {})
    if train_epochs is not None:
        train_payload["epochs"] = int(train_epochs)
    if train_payload:
        payload["train"] = train_payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _process_grouped_head_config(base_cfg: Path, process_id: int) -> Path | None:
    if base_cfg.name != "process_surrogate_edge_all_v3_grouped_head.yaml":
        return None
    candidate = (
        PROJECT_ROOT
        / "configs"
        / "experiment"
        / "process_grouped_head"
        / f"process_surrogate_edge_all_v3_grouped_head_p{int(process_id):02d}.yaml"
    )
    return candidate if candidate.is_file() else None


def _load_process_target_stream_train_overrides(base_cfg: Path, process_id: int) -> dict:
    process_cfg = _process_grouped_head_config(base_cfg, process_id)
    if process_cfg is None:
        return {}
    raw = yaml.safe_load(process_cfg.read_text(encoding="utf-8")) or {}
    train = ((raw.get("overrides") or {}).get("train") or {})
    keys = (
        "use_target_stream_loss_weighting",
        "target_stream_loss_weight",
        "target_stream_weighting_mode",
        "target_stream_weight_conflict_policy",
        "allow_yaml_only_target_streams",
        "target_stream_loss_weights",
    )
    return {key: train[key] for key in keys if key in train}


def _load_base_data_overrides(base_cfg: Path) -> dict:
    raw = yaml.safe_load(base_cfg.read_text(encoding="utf-8")) or {}
    return dict(((raw.get("overrides") or {}).get("data") or {}))


def _infer_splits_dir_from_manifest(manifest_path: str, *, default: str) -> str:
    path = Path(str(manifest_path or ""))
    fold_dir = path.parent
    if fold_dir.name.lower().startswith("fold_"):
        split_root = fold_dir.parent
        if split_root.name.lower() == "all" or re.fullmatch(
            r"process\d+",
            split_root.name,
            flags=re.IGNORECASE,
        ):
            split_root = split_root.parent
        return split_root.as_posix()
    return default


def _base_joint_manifests_for_fold(base_data: dict, fold: int) -> tuple[str, str, str] | None:
    paths: list[str] = []
    for key in ("train_split_manifest_path", "val_split_manifest_path", "test_split_manifest_path"):
        raw = str(base_data.get(key) or "").strip()
        if not raw:
            return None
        path = Path(raw)
        parts = list(path.parts)
        if any(re.fullmatch(r"process\d+", str(part), flags=re.IGNORECASE) for part in parts):
            return None
        if not any(str(part).lower().startswith("fold_") for part in parts):
            return None
        parts = [_fold_dir_name(fold) if str(part).lower().startswith("fold_") else part for part in parts]
        fold_path = Path(*parts)
        if not (PROJECT_ROOT / fold_path).is_file():
            return None
        paths.append(fold_path.as_posix())
    return paths[0], paths[1], paths[2]


def _find_run_dir_in_fold(fold_dir: Path, pid: int, fold: int) -> Path | None:
    """Resolve actual run directory (train appends -YYYYMMDD-HHMMSS to experiment_name)."""
    if not fold_dir.is_dir():
        return None
    prefix = _run_name(pid, fold)
    matches = sorted(
        (c for c in fold_dir.iterdir() if c.is_dir() and c.name.startswith(prefix)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _run_dir(output_root: Path, pid: int, fold: int) -> Path:
    fold_dir = (output_root / _process_label(pid) / _fold_dir_name(fold)).resolve()
    found = _find_run_dir_in_fold(fold_dir, pid, fold)
    if found is not None:
        return found
    return (fold_dir / _run_name(pid, fold)).resolve()


def _find_joint_run_dir_in_fold(fold_dir: Path, fold: int) -> Path | None:
    if not fold_dir.is_dir():
        return None
    prefix = _joint_run_name(fold)
    matches = sorted(
        (c for c in fold_dir.iterdir() if c.is_dir() and c.name.startswith(prefix)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _joint_run_dir(output_root: Path, fold: int) -> Path:
    fold_dir = (output_root / "All" / _fold_dir_name(fold)).resolve()
    found = _find_joint_run_dir_in_fold(fold_dir, fold)
    if found is not None:
        return found
    return (fold_dir / _joint_run_name(fold)).resolve()


def _run_targetrow_validation(run_dir: Path) -> dict:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from ablation_targetrow_common import validate_run_targetrow_complete

    return validate_run_targetrow_complete(run_dir)


def _find_checkpoint(run_dir: Path) -> Path | None:
    for rel in ("checkpoints/best.pt", "checkpoints/best_model.pt", "best.pt"):
        p = run_dir / rel
        if p.is_file():
            return p
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        pts = sorted(ckpt_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)
        if pts:
            return pts[0]
    return None


def _should_skip(run_dir: Path, *, force_reevaluate: bool = False) -> bool:
    if force_reevaluate:
        return False
    return bool(_run_targetrow_validation(run_dir).get("ok"))


def _recompute_metrics_only(run_dir: Path, base_cfg: Path) -> int:
    """Re-evaluate checkpoint to refresh target-row metrics without retraining."""
    ckpt = _find_checkpoint(run_dir)
    if ckpt is None:
        return 1
    overrides = run_dir.parent / "runtime_overrides.json"
    cmd = [
        sys.executable,
        str((PROJECT_ROOT / "scripts/eval_process_surrogate_edge_all.py").resolve()),
        "--config",
        str(base_cfg.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "--checkpoint",
        str(ckpt.resolve()),
        "--split",
        "all",
        "--output-dir",
        str(run_dir.resolve()),
    ]
    if overrides.is_file():
        cmd.extend(["--runtime-overrides-file", str(overrides.resolve())])
    print("[recompute-metrics]", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    return int(proc.returncode)


def _metric_aliases(metric: str) -> tuple[str, ...]:
    key = str(metric or "").strip()
    aliases = {
        "val_target_edge_property_mean_r2": (
            "val_target_edge_property_mean_r2",
            "target_edge_property_mean_r2",
            "val/target_edge_property_mean_r2",
        ),
        "target_edge_property_mean_r2": (
            "val_target_edge_property_mean_r2",
            "target_edge_property_mean_r2",
            "val/target_edge_property_mean_r2",
        ),
        "val_target_mean_r2": ("val_target_mean_r2", "target_mean_r2", "val/target_mean_r2"),
        "target_mean_r2": ("val_target_mean_r2", "target_mean_r2", "val/target_mean_r2"),
        "val_target_r2": ("val_target_r2", "target_r2", "val/target_r2"),
        "target_r2": ("val_target_r2", "target_r2", "val/target_r2"),
        "val_target_edge_10d_r2_flatten": (
            "val_target_edge_10d_r2_flatten",
            "target_edge_10d_r2_flatten",
            "val/target_edge_10d_r2_flatten",
        ),
        "target_edge_10d_r2_flatten": (
            "val_target_edge_10d_r2_flatten",
            "target_edge_10d_r2_flatten",
            "val/target_edge_10d_r2_flatten",
        ),
        "flatten_r2": (
            "val_target_edge_10d_r2_flatten",
            "target_edge_10d_r2_flatten",
            "val/target_edge_10d_r2_flatten",
        ),
    }
    return aliases.get(key, (key,) if key else ("val_target_edge_property_mean_r2",))


def _best_monitor_metric(run_dir: Path, monitor_metric: str) -> tuple[int, float, str]:
    metrics_path = run_dir / "metrics_per_epoch.csv"
    if not metrics_path.is_file():
        return 0, float("nan"), ""
    frame = pd.read_csv(metrics_path)
    if frame.empty:
        return 0, float("nan"), ""
    metric_col = next((col for col in _metric_aliases(monitor_metric) if col in frame.columns), "")
    if not metric_col:
        return 0, float("nan"), ""
    values = pd.to_numeric(frame[metric_col], errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return 0, float("nan"), metric_col
    row_index = valid.idxmax()
    epoch = int(frame.loc[row_index, "epoch"]) if "epoch" in frame.columns else int(row_index) + 1
    return epoch, float(valid.loc[row_index]), metric_col


def _best_target_mean_r2(run_dir: Path) -> tuple[int, float]:
    epoch, value, _ = _best_monitor_metric(run_dir, "val_target_mean_r2")
    return epoch, value


def _rel_or_abs(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run per-process K-fold experiments.")
    parser.add_argument("--base-config", type=str, required=True)
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        help="Split manifest root. In --joint-all-processes mode, defaults to the base config manifest root when present.",
    )
    parser.add_argument("--output-root", type=str, default="outputs/process_kfold")
    parser.add_argument(
        "--merged-csv",
        type=str,
        default=None,
        help="Merged Main CSV. In --joint-all-processes mode, defaults to the base config train_data_path.",
    )
    parser.add_argument("--stream-data-dir", type=str, default="data/main_data_Streams")
    parser.add_argument("--process-ids", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--joint-all-processes",
        action="store_true",
        help=(
            "Run one model per fold over all --process-ids together. "
            "Per-process fold manifests are concatenated into All/fold_XX manifests."
        ),
    )
    parser.add_argument("--max-epochs", type=int, default=0, help="If >0, passed as --max-epochs to train script.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--only-folds",
        type=int,
        nargs="*",
        default=None,
        help="1-based fold numbers to run (default: all 1..k-folds).",
    )
    parser.add_argument(
        "--skip-startup-debug",
        action="store_true",
        help="Pass --skip-startup-debug to train_process_surrogate (skip slow global audits).",
    )
    parser.add_argument(
        "--no-save-model-weights",
        action="store_true",
        help="Pass --no-save-model-weights to train_process_surrogate (skip best.pt/last.pt checkpoint writes).",
    )
    parser.add_argument(
        "--monitor-metric",
        type=str,
        default="",
        help="Optional train.monitor_metric override, e.g. val_target_edge_property_mean_r2.",
    )
    parser.add_argument(
        "--monitor-mode",
        type=str,
        choices=("min", "max"),
        default="",
        help="Optional train.monitor_mode override for fold runs, e.g. max for R2.",
    )
    parser.add_argument(
        "--force-reevaluate",
        action="store_true",
        help="Never skip folds (full retrain even if outputs exist).",
    )
    parser.add_argument(
        "--force-recompute-metrics",
        action="store_true",
        help="For skipped/stale folds with checkpoint, run eval-only metric refresh.",
    )
    parser.add_argument("--efficiency-benchmark", action="store_true")
    parser.add_argument("--inference-warmup-runs", type=int, default=5)
    parser.add_argument("--inference-measured-runs", type=int, default=20)
    parser.add_argument("--inference-max-batches", type=int, default=0)
    args = parser.parse_args()

    base_cfg = (PROJECT_ROOT / args.base_config).resolve()
    if not base_cfg.is_file():
        raise FileNotFoundError(f"base config not found: {base_cfg}")
    base_data = _load_base_data_overrides(base_cfg)

    default_merged_csv = "data/datasets_v3/process_main_merged.csv"
    default_splits_dir = "data/splits/process_kfold"
    if args.joint_all_processes:
        default_merged_csv = str(base_data.get("train_data_path") or default_merged_csv)
        default_splits_dir = _infer_splits_dir_from_manifest(
            str(base_data.get("train_split_manifest_path") or ""),
            default=default_splits_dir,
        )

    merged_csv = str(args.merged_csv or default_merged_csv)
    splits_dir = str(args.splits_dir or default_splits_dir)
    splits_root = (PROJECT_ROOT / splits_dir).resolve()
    out_root = (PROJECT_ROOT / args.output_root).resolve()
    stream_data_dir = (PROJECT_ROOT / args.stream_data_dir).resolve()
    merged_rel = str(Path(merged_csv).as_posix())
    k_folds = int(args.k_folds)
    only = set(args.only_folds) if args.only_folds is not None else None

    commands: list[list[str]] = []
    process_ids = [int(p) for p in args.process_ids]
    epochs_override = int(args.max_epochs) if args.max_epochs and int(args.max_epochs) > 0 else None
    selected_folds = [
        fold for fold in range(1, k_folds + 1)
        if only is None or fold in only
    ]
    if not args.joint_all_processes:
        has_process_manifests = all(
            (
                splits_root / _process_label(pid) / _fold_dir_name(fold) / f"{split_name}.csv"
            ).is_file()
            for pid in process_ids
            for fold in selected_folds
            for split_name in ("train", "val", "test")
        )
        has_joint_manifests = all(
            (splits_root / _fold_dir_name(fold) / f"{split_name}.csv").is_file()
            for fold in selected_folds
            for split_name in ("train", "val", "test")
        )
        if not has_process_manifests and has_joint_manifests:
            splits_root = _materialize_process_views_from_joint_manifests(
                joint_root=splits_root,
                destination=out_root,
                process_ids=process_ids,
                folds=selected_folds,
            )
    if args.joint_all_processes:
        for fold in range(1, k_folds + 1):
            if only is not None and fold not in only:
                continue
            fold_out = out_root / "All" / _fold_dir_name(fold)
            base_joint_manifests = _base_joint_manifests_for_fold(base_data, fold) if args.splits_dir is None else None
            if base_joint_manifests is not None:
                rel_tr, rel_va, rel_te = base_joint_manifests
            else:
                prebuilt = {
                    split_name: splits_root / _fold_dir_name(fold) / f"{split_name}.csv"
                    for split_name in ("train", "val", "test")
                }
                if all(path.is_file() for path in prebuilt.values()):
                    rel_tr = _rel_or_abs(prebuilt["train"], PROJECT_ROOT).replace("\\", "/")
                    rel_va = _rel_or_abs(prebuilt["val"], PROJECT_ROOT).replace("\\", "/")
                    rel_te = _rel_or_abs(prebuilt["test"], PROJECT_ROOT).replace("\\", "/")
                    print(
                        f"[joint-manifest] reusing prebuilt fold manifests: {prebuilt['train'].parent}",
                        flush=True,
                    )
                else:
                    manifest_dir = fold_out / "manifests"
                    tr_m = manifest_dir / "train.csv"
                    va_m = manifest_dir / "val.csv"
                    te_m = manifest_dir / "test.csv"
                    _write_joint_manifest(
                        out_path=tr_m,
                        splits_root=splits_root,
                        stream_data_dir=stream_data_dir,
                        process_ids=process_ids,
                        fold=fold,
                        split_name="train",
                    )
                    _write_joint_manifest(
                        out_path=va_m,
                        splits_root=splits_root,
                        stream_data_dir=stream_data_dir,
                        process_ids=process_ids,
                        fold=fold,
                        split_name="val",
                    )
                    _write_joint_manifest(
                        out_path=te_m,
                        splits_root=splits_root,
                        stream_data_dir=stream_data_dir,
                        process_ids=process_ids,
                        fold=fold,
                        split_name="test",
                    )
                    rel_tr = str(tr_m.relative_to(PROJECT_ROOT).as_posix())
                    rel_va = str(va_m.relative_to(PROJECT_ROOT).as_posix())
                    rel_te = str(te_m.relative_to(PROJECT_ROOT).as_posix())

            name = _joint_run_name(fold)
            overrides_path = fold_out / "runtime_overrides.json"
            try:
                out_rel = str(fold_out.relative_to(PROJECT_ROOT).as_posix())
            except ValueError:
                out_rel = str(fold_out.resolve())

            train_overrides: dict = {}
            if str(args.monitor_metric).strip():
                train_overrides["monitor_metric"] = str(args.monitor_metric).strip()
            if str(args.monitor_mode).strip():
                train_overrides["monitor_mode"] = str(args.monitor_mode).strip()
            _write_runtime_overrides(
                overrides_path,
                seed=args.seed,
                device=args.device,
                experiment_name=name,
                output_dir=out_rel,
                save_dir=f"{out_rel}/checkpoints",
                log_dir=f"{out_rel}/logs",
                merged_rel=merged_rel,
                process_ids=process_ids,
                train_manifest=rel_tr,
                val_manifest=rel_va,
                test_manifest=rel_te,
                train_epochs=epochs_override,
                train_overrides=train_overrides,
            )

            cmd = [
                sys.executable,
                str((PROJECT_ROOT / "scripts" / "train_process_surrogate.py").resolve()),
                "--config",
                _rel_or_abs(base_cfg, PROJECT_ROOT),
                "--runtime-overrides-file",
                _rel_or_abs(overrides_path, PROJECT_ROOT),
            ]
            if args.max_epochs and int(args.max_epochs) > 0:
                cmd.extend(["--max-epochs", str(int(args.max_epochs))])
            if args.skip_startup_debug:
                cmd.append("--skip-startup-debug")
            if args.no_save_model_weights:
                cmd.append("--no-save-model-weights")
            if args.efficiency_benchmark:
                cmd.extend(
                    [
                        "--efficiency-benchmark",
                        "--inference-warmup-runs",
                        str(int(args.inference_warmup_runs)),
                        "--inference-measured-runs",
                        str(int(args.inference_measured_runs)),
                        "--inference-max-batches",
                        str(int(args.inference_max_batches)),
                    ]
                )

            run_dir = _joint_run_dir(out_root, fold)
            if args.skip_existing and _should_skip(run_dir, force_reevaluate=args.force_reevaluate):
                print(
                    f"[skip-existing] All fold_{fold:02d} run_dir={run_dir.name} "
                    f"status=skipped_has_primary_metric",
                    flush=True,
                )
                continue
            commands.append(cmd)

    for pid in ([] if args.joint_all_processes else process_ids):
        plabel = _process_label(pid)
        for fold in range(1, k_folds + 1):
            if only is not None and fold not in only:
                continue
            split_fold = splits_root / plabel / _fold_dir_name(fold)
            tr_m = split_fold / "train.csv"
            va_m = split_fold / "val.csv"
            te_m = split_fold / "test.csv"
            if not tr_m.is_file() or not va_m.is_file() or not te_m.is_file():
                raise FileNotFoundError(f"Missing split manifests under {split_fold}")

            rel_tr = str(tr_m.relative_to(PROJECT_ROOT).as_posix())
            rel_va = str(va_m.relative_to(PROJECT_ROOT).as_posix())
            rel_te = str(te_m.relative_to(PROJECT_ROOT).as_posix())

            fold_out = out_root / plabel / _fold_dir_name(fold)
            name = _run_name(pid, fold)

            overrides_path = fold_out / "runtime_overrides.json"
            try:
                out_rel = str(fold_out.relative_to(PROJECT_ROOT).as_posix())
            except ValueError:
                out_rel = str(fold_out.resolve())

            train_overrides = _load_process_target_stream_train_overrides(base_cfg, pid)
            if str(args.monitor_metric).strip():
                train_overrides["monitor_metric"] = str(args.monitor_metric).strip()
            if str(args.monitor_mode).strip():
                train_overrides["monitor_mode"] = str(args.monitor_mode).strip()
            _write_runtime_overrides(
                overrides_path,
                seed=args.seed,
                device=args.device,
                experiment_name=name,
                output_dir=out_rel,
                save_dir=f"{out_rel}/checkpoints",
                log_dir=f"{out_rel}/logs",
                merged_rel=merged_rel,
                process_ids=[pid],
                train_manifest=rel_tr,
                val_manifest=rel_va,
                test_manifest=rel_te,
                train_epochs=epochs_override,
                train_overrides=train_overrides,
            )

            cmd = [
                sys.executable,
                str((PROJECT_ROOT / "scripts" / "train_process_surrogate.py").resolve()),
                "--config",
                _rel_or_abs(base_cfg, PROJECT_ROOT),
                "--runtime-overrides-file",
                _rel_or_abs(overrides_path, PROJECT_ROOT),
                "--process-filter",
                str(int(pid)),
            ]
            if args.max_epochs and int(args.max_epochs) > 0:
                cmd.extend(["--max-epochs", str(int(args.max_epochs))])
            if args.skip_startup_debug:
                cmd.append("--skip-startup-debug")
            if args.no_save_model_weights:
                cmd.append("--no-save-model-weights")
            if args.efficiency_benchmark:
                cmd.extend(
                    [
                        "--efficiency-benchmark",
                        "--inference-warmup-runs",
                        str(int(args.inference_warmup_runs)),
                        "--inference-measured-runs",
                        str(int(args.inference_measured_runs)),
                        "--inference-max-batches",
                        str(int(args.inference_max_batches)),
                    ]
                )

            run_dir = _run_dir(out_root, pid, fold)
            if args.skip_existing and _should_skip(run_dir, force_reevaluate=args.force_reevaluate):
                print(
                    f"[skip-existing] {plabel} fold_{fold:02d} run_dir={run_dir.name} "
                    f"status=skipped_has_primary_metric",
                    flush=True,
                )
                continue
            val_res = _run_targetrow_validation(run_dir)
            if (
                args.skip_existing
                and not args.force_reevaluate
                and _find_checkpoint(run_dir) is not None
                and not val_res.get("ok")
            ):
                print(
                    f"[stale-metrics] {plabel} fold_{fold:02d} — {val_res.get('reasons', 'incomplete')}; "
                    "eval-only metric recompute.",
                    flush=True,
                )
                rc = _recompute_metrics_only(run_dir, base_cfg)
                if rc == 0 and _run_targetrow_validation(run_dir).get("ok"):
                    continue
            commands.append(cmd)

    if args.dry_run:
        for c in commands:
            print(" ".join(c))
        print(f"[dry-run] total commands: {len(commands)}", file=sys.stderr)
        return

    monitor_msg = (
        f" monitor={args.monitor_metric}/{args.monitor_mode}"
        if str(args.monitor_metric).strip() or str(args.monitor_mode).strip()
        else ""
    )
    print(
        "[kfold] edge_all: val_target_edge_property_mean_r2 is the default optimization metric;"
        + monitor_msg,
        flush=True,
    )
    for cmd in commands:
        print("[run]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

    if args.joint_all_processes:
        for fold in range(1, k_folds + 1):
            if only is not None and fold not in only:
                continue
            run_dir = _joint_run_dir(out_root, fold)
            best_epoch, best_value, metric_col = _best_monitor_metric(
                run_dir, str(args.monitor_metric or "val_target_edge_property_mean_r2")
            )
            print(
                f"[joint-metric-summary] fold={fold} run_dir={run_dir} "
                f"best_epoch_by_{metric_col or 'monitor_metric'}={best_epoch} "
                f"best_{metric_col or 'monitor_metric'}={best_value}",
                flush=True,
            )
        print(
            f"[joint-summary] wrote joint fold runs under {(out_root / 'All').resolve()}",
            flush=True,
        )
        return

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from ablation_targetrow_common import (
        LEGACY_ANSWER_FRACTION_METRIC,
        PRIMARY_METRIC_NAME,
        discover_kfold_runs,
        extract_primary_metrics,
    )

    for rec in discover_kfold_runs(out_root):
        run_dir = rec["run_dir"]
        m = extract_primary_metrics(run_dir)
        best_ep, best_monitor_value, metric_col = _best_monitor_metric(
            run_dir, str(args.monitor_metric or "val_target_edge_property_mean_r2")
        )
        print(
            f"[fold-summary] exp={rec['exp_name']} process={rec['process_id']} fold={rec['fold']} "
            f"run_dir={run_dir} "
            f"primary_metric_name={m.get('primary_metric_name', PRIMARY_METRIC_NAME)} "
            f"best_epoch_by_{metric_col or 'monitor_metric'}={best_ep} "
            f"best_{metric_col or 'monitor_metric'}={best_monitor_value} "
            f"included_target_ids={m.get('included_target_ids', '')} "
            f"excluded_target_ids={m.get('excluded_target_ids', '')} "
            f"val_{PRIMARY_METRIC_NAME}={m.get(f'val_{PRIMARY_METRIC_NAME}', float('nan'))} "
            f"test_{PRIMARY_METRIC_NAME}={m.get(f'test_{PRIMARY_METRIC_NAME}', float('nan'))} "
            f"val_{LEGACY_ANSWER_FRACTION_METRIC}={m.get(f'val_{LEGACY_ANSWER_FRACTION_METRIC}', float('nan'))} "
            f"test_{LEGACY_ANSWER_FRACTION_METRIC}={m.get(f'test_{LEGACY_ANSWER_FRACTION_METRIC}', float('nan'))}",
            flush=True,
        )


if __name__ == "__main__":
    main()
