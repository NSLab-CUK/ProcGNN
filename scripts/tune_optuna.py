#!/usr/bin/env python3
"""Optuna-based hyperparameter tuning for the process-surrogate `edge_all` task.

W&B 없이 동작합니다. 각 trial은 ``train_process_surrogate.py`` 를 서브프로세스로
실행하고, 종료 후 ``train_val_history.json`` 의 val 섹션에서
가중 ``answer_targets_r2`` (H2·CO2 R², ``answer_edge_weight_h2/co2``) 시계열의 최대값을 objective 로 사용합니다.

기본 출력 구조::

    outputs/optuna/Process{N}/  (또는 All/)
        study.db                       # SQLite Optuna 스토리지
        best_hyperparameters.json      # 최종 best params + value
        optuna_trials_summary.csv      # 완료 trial 요약 (휴리스틱 트레이드오프용)
        optuna_trials_top.md           # objective 상위 trial 표
        optuna_numeric_correlations.csv # param vs objective 피어슨 상관
        optuna_param_importances.csv   # Fanova importance (실패 시 *_skipped.txt)
        warmstart_source.json          # 이전 스터디에서 불러온 warmstart 경로
        warmstart_runtime_overrides.json  # train/model 중첩 JSON (학습 재현용)
        best_progress.jsonl            # 스터디 best objective가 갱신될 때만 append
        optuna_report/                 # scatter / 히스토그램 등
        trial_0001_HHMMSS/
            train_val_history.json
            metrics_per_epoch.csv
            optuna_trial_metrics.json  # 해당 trial MAE/RMSE/SMAPE/R2 정리
            metrics_main_targets.json  # val-history fallback (export off)
            ...

Optuna trial 에서는 용량이 큰 아래 3종은 저장하지 않습니다 (trial 종료 후 자동 삭제):
    1) edge_all_all_edges_all_cell_metrics.csv (legacy; edge_predictions.csv no longer written)
    2) checkpoints/*.pt        (모델 가중치)
    3) plots/                  (학습/평가 그림)
또한 train/val/test split export 디렉터리도 함께 제거합니다.
``--keep-trial-artifacts`` 로 비활성화 가능.

빠른 스모크 (1 trial, 짧은 epoch, ``scripts/smoke_optuna.py``)::

    python scripts/smoke_optuna.py

이전 스터디 ``best_hyperparameters.json`` 을 초기값으로 쓰려면 (공정 N일 때 ``outputs/optuna/Process{N}/`` 등 자동 탐색; 기본 ON)::

    python scripts/tune_optuna.py --config ... --process-filter 1
    python scripts/tune_optuna.py --config ... --process-filter 1 --auto-warmstart-best
    python scripts/tune_optuna.py --config ... --warmstart-best-json outputs/optuna/Process1/best_hyperparameters.json
    python scripts/tune_optuna.py ... --no-warmstart-auto-discover   # 자동 탐색 끔
    python scripts/tune_optuna.py ... --no-warmstart-auto-discover --auto-warmstart-best  # 자동 탐색 다시 켬

병렬 실행 (GPU 0~3):

    CUDA_VISIBLE_DEVICES=0 python scripts/tune_optuna.py --process-filter 1 ... &
    CUDA_VISIBLE_DEVICES=1 python scripts/tune_optuna.py --process-filter 2 ... &
    ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd

from process_graph.experiment.optuna_warmstart import (  # noqa: E402
    discover_best_hyperparameters_json,
    ensure_categorical_value,
    flat_optuna_params_to_runtime_overrides,
    load_best_params_from_json,
)

try:
    import optuna
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "optuna 가 설치되어 있지 않습니다. `pip install optuna` 또는 "
        "`pip install -r requirements.txt` 로 설치하세요."
    ) from exc


TRAIN_SCRIPT = (PROJECT_ROOT / "scripts" / "train_process_surrogate.py").resolve()

# Optuna trial 에서 보관하지 않을 대용량 산출물 (3종 + split export tree).
OPTUNA_SKIP_EXPORT_TRAIN_KEYS: dict[str, Any] = {
    "edge_all_export_predictions": False,
    "edge_all_print_first_batch": False,
    "answer_edge_weight": 5.0,
    "answer_edge_weight_h2": 5.0,
    "answer_edge_weight_co2": 5.0,
    "answer_edge_weight_h2o": 5.0,
}
OPTUNA_PRUNE_SPLIT_EXPORT_DIRS: tuple[str, ...] = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------
def _round_to_multiple(value: float, multiple: int = 8, *, min_value: int = 8) -> int:
    return max(int(min_value), int(round(float(value) / float(multiple)) * int(multiple)))


def _scaled_model_dims(hidden_dim: int) -> dict[str, int]:
    """Tie secondary model widths to hidden_dim to keep Optuna's space compact."""
    h = int(hidden_dim)
    return {
        "attn_hidden_dim": _round_to_multiple(h / 4, min_value=64),
        "unit_emb_dim": _round_to_multiple(h / 8, min_value=32),
    }


def _scaled_edge_all_model_dims(hidden_dim: int) -> dict[str, int]:
    h = int(hidden_dim)
    out = _scaled_model_dims(h)
    out.update(
        {
            "edge_stream_role_emb_dim": _round_to_multiple(h / 16, min_value=16),
            "edge_stream_id_emb_dim": _round_to_multiple(h / 8, min_value=24),
            "edge_decoder_hidden_dim": h,
        }
    )
    return out


def _suggest_edge_all(
    trial: optuna.Trial,
    baseline_flat: dict[str, Any] | None = None,
    *,
    all_processes: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """edge_all 전용 탐색 공간 (sweep_config.build_edge_all_sweep_config 와 동치)."""
    b = baseline_flat or {}
    # Multi-process joint training needs a bit more width than single-process Optuna runs.
    hidden_choices = [320, 384, 448, 512, 576, 640] if all_processes else [256, 320, 384, 448, 512]

    train: Dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 3e-5, 1e-3, log=True),
    }

    model: Dict[str, Any] = {
        "hidden_dim": trial.suggest_categorical(
            "hidden_dim",
            ensure_categorical_value(hidden_choices, b.get("hidden_dim")),
        ),
        "num_layers": trial.suggest_categorical(
            "num_layers", ensure_categorical_value([3, 4, 5], b.get("num_layers"))
        ),
    }
    model.update(_scaled_edge_all_model_dims(int(model["hidden_dim"])))
    return {"train": train, "model": model}


def _suggest_multitask(
    trial: optuna.Trial,
    baseline_flat: dict[str, Any] | None = None,
    *,
    all_processes: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """multitask (target+tailgas) 탐색 공간."""
    b = baseline_flat or {}
    hidden_choices = [320, 384, 448, 512, 576, 640] if all_processes else [256, 320, 384, 448, 512]
    train: Dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True),
    }
    model: Dict[str, Any] = {
        "hidden_dim": trial.suggest_categorical(
            "hidden_dim",
            ensure_categorical_value(hidden_choices, b.get("hidden_dim")),
        ),
        "num_layers": trial.suggest_categorical(
            "num_layers", ensure_categorical_value([3, 4, 5], b.get("num_layers"))
        ),
    }
    model.update(_scaled_model_dims(int(model["hidden_dim"])))
    return {"train": train, "model": model}


def _make_suggester(preset: str, baseline_flat: dict[str, Any] | None, *, all_processes: bool):
    def _fn(trial: optuna.Trial) -> Dict[str, Dict[str, Any]]:
        if preset == "edge_all":
            return _suggest_edge_all(trial, baseline_flat, all_processes=all_processes)
        return _suggest_multitask(trial, baseline_flat, all_processes=all_processes)

    return _fn


def _has_existing_trial_with_params(study: optuna.Study, params: dict[str, Any]) -> bool:
    if not params:
        return False
    for t in study.trials:
        if dict(t.params) == params:
            return True
    return False


# ---------------------------------------------------------------------------
# Trial subprocess runner
# ---------------------------------------------------------------------------
def _run_train_subprocess(
    *,
    config: str,
    overrides_path: Path,
    run_name: str,
    max_epochs: int,
    process_filter: int,
    log_path: Path | None = None,
    skip_startup_debug: bool = False,
) -> int:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config", config,
        "--runtime-overrides-file", str(overrides_path),
        "--max-epochs", str(int(max_epochs)),
        "--process-filter", str(int(process_filter)),
        "--no-training-plots",
    ]
    if skip_startup_debug:
        cmd.append("--skip-startup-debug")
    env = os.environ.copy()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as logf:
            return subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT)
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT), env=env)


def _optuna_trial_train_overrides(
    suggested_train: Mapping[str, Any] | None = None,
    *,
    optuna_objective: str = "",
) -> dict[str, Any]:
    """Merge suggested train params with Optuna storage policy (no heavy export)."""
    out = dict(OPTUNA_SKIP_EXPORT_TRAIN_KEYS)
    if suggested_train:
        out.update(dict(suggested_train))
    out["edge_all_export_predictions"] = False
    # Main-target loss weights fixed at 5.0 during Optuna (do not tune per-species weights).
    out["answer_edge_weight"] = 5.0
    out["answer_edge_weight_h2"] = 5.0
    out["answer_edge_weight_co2"] = 5.0
    out["answer_edge_weight_h2o"] = 5.0
    mode = str(optuna_objective or "target_edge_property_mean_r2")
    if mode == "target_edge_property_mean_r2":
        out["monitor_metric"] = "val_target_edge_property_mean_r2"
        out["monitor_mode"] = "max"
    if mode in {"target_mean_r2", "target_edge_feature_mean_r2"}:
        out["monitor_metric"] = "val_target_mean_r2"
        out["monitor_mode"] = "max"
    if mode in {
        "target_edge_10d_r2_flatten",
        "val_target_edge_10d_r2_flatten",
        "flatten_r2",
    }:
        out["monitor_metric"] = "val_target_edge_10d_r2_flatten"
        out["monitor_mode"] = "max"
    return out


def _prune_large_optuna_trial_artifacts(
    *,
    run_dir: Path,
    checkpoint_dir: Path | None,
) -> list[str]:
    """Remove large per-trial files; keep train_val_history + optuna metrics only."""
    removed: list[str] = []

    if run_dir.is_dir():
        for pattern in ("edge_predictions.csv", "edge_all_all_edges_all_cell_metrics.csv"):
            for path in sorted(run_dir.rglob(pattern)):
                try:
                    rel = path.relative_to(run_dir).as_posix()
                    path.unlink()
                    removed.append(rel)
                except OSError:
                    pass
        plots_dir = run_dir / "plots"
        if plots_dir.is_dir():
            try:
                shutil.rmtree(plots_dir)
                removed.append("plots/")
            except OSError:
                pass
        for split_name in OPTUNA_PRUNE_SPLIT_EXPORT_DIRS:
            split_dir = run_dir / split_name
            if split_dir.is_dir():
                try:
                    shutil.rmtree(split_dir)
                    removed.append(f"{split_name}/")
                except OSError:
                    pass

    if checkpoint_dir is not None and checkpoint_dir.is_dir():
        for ckpt in sorted(checkpoint_dir.glob("*.pt")):
            try:
                rel = ckpt.relative_to(checkpoint_dir.parent).as_posix()
                ckpt.unlink()
                removed.append(rel)
            except OSError:
                pass
        try:
            if checkpoint_dir.is_dir() and not any(checkpoint_dir.iterdir()):
                checkpoint_dir.rmdir()
                removed.append(f"{checkpoint_dir.name}/ (empty)")
        except OSError:
            pass

    return removed


def _float_series(val: Mapping[str, Any], *keys: str) -> list[float]:
    for k in keys:
        arr = val.get(k)
        if isinstance(arr, list) and arr:
            return [float(x) for x in arr]
        if isinstance(arr, tuple) and arr:
            return [float(x) for x in arr]
    return []


def _pick_val_at(val: Mapping[str, Any], idx: int, *keys: str) -> float:
    for k in keys:
        arr = val.get(k)
        if isinstance(arr, list) and len(arr) > idx:
            return float(arr[idx])
    return float("nan")


def _norm_weights(w_h2: float, w_co2: float) -> tuple[float, float, float]:
    wsum = float(w_h2) + float(w_co2)
    if wsum <= 0:
        return 1.0, 1.0, 2.0
    return float(w_h2), float(w_co2), wsum


def _weighted_linear(a: float, b: float, w_h2: float, w_co2: float) -> float:
    wh, wc, wsum = _norm_weights(w_h2, w_co2)
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan")
    return (wh * a + wc * b) / wsum


def _combined_weighted_r2(a: float, b: float, wh: float, wc: float) -> float:
    """한쪽 R²만 유효하면 그 값을 objective로 사용 (작은 val에서 SST≈0으로 NaN인 경우)."""
    fa = math.isfinite(a)
    fb = math.isfinite(b)
    if fa and fb:
        wsum = wh + wc
        if wsum <= 0:
            return 0.5 * (a + b)
        return (wh * a + wc * b) / wsum
    if fa:
        return float(a)
    if fb:
        return float(b)
    return float("nan")


def _weighted_rmse_from_rmse(r_h2: float, r_co2: float, w_h2: float, w_co2: float) -> float:
    """가중 MSE의 제곱근 (RMSE를 선형 결합한 것과는 다름; 휴리스틱 보조 지표)."""
    wh, wc, wsum = _norm_weights(w_h2, w_co2)
    if not (math.isfinite(r_h2) and math.isfinite(r_co2)):
        return float("nan")
    mse = (wh * (r_h2**2) + wc * (r_co2**2)) / wsum
    return float(math.sqrt(max(mse, 0.0)))


def _r2_series_for_preset(val: Mapping[str, Any], preset: str) -> tuple[list[float], list[float]]:
    if preset == "edge_all":
        h2 = _float_series(
            val,
            "metric_target_r2",
            "val_metric_target_r2",
            "target_h2_r2",
            "metric_target_h2_r2",
        )
        co2 = _float_series(
            val,
            "metric_tailgas_r2",
            "val_metric_tailgas_r2",
            "tailgas_co2_r2",
            "metric_tailgas_co2_r2",
        )
        return h2, co2
    h2 = _float_series(val, "metric_target_r2", "val_metric_target_r2", "val/metric_target_r2")
    co2 = _float_series(val, "metric_tailgas_r2", "val_metric_tailgas_r2", "val/metric_tailgas_r2")
    return h2, co2


def _weighted_answer_r2_series(
    val: Mapping[str, Any],
    preset: str,
    *,
    w_h2: float,
    w_co2: float,
) -> list[float]:
    """에폭별 Optuna 목적값: (w_h2·R²_H2 + w_co2·R²_CO2) / (w_h2+w_co2)."""
    h2_r2_s, co2_r2_s = _r2_series_for_preset(val, preset)
    if not h2_r2_s or not co2_r2_s:
        return []
    wh, wc, _ = _norm_weights(w_h2, w_co2)
    n = min(len(h2_r2_s), len(co2_r2_s))
    return [
        _combined_weighted_r2(float(h2_r2_s[i]), float(co2_r2_s[i]), wh, wc) for i in range(n)
    ]


def _v4_macro_series(val: Mapping[str, Any], *keys: str) -> list[float]:
    for key in keys:
        s = _float_series(val, key)
        if s:
            return s
    return []


def _objective_series_and_name(
    val: Mapping[str, Any],
    preset: str,
    *,
    optuna_objective: str,
    w_h2: float,
    w_co2: float,
) -> tuple[list[float], str]:
    mode = str(optuna_objective or "target_edge_property_mean_r2")
    if mode == "target_edge_property_mean_r2":
        series = _v4_macro_series(
            val,
            "target_edge_property_mean_r2",
            "val_target_edge_property_mean_r2",
            "val/target_edge_property_mean_r2",
        )
        return series, "optuna_objective_target_edge_property_mean_r2"
    if mode in ("target_edge_10d_r2_flatten", "val_target_edge_10d_r2_flatten", "flatten_r2"):
        series = _v4_macro_series(
            val,
            "target_edge_10d_r2_flatten",
            "val_target_edge_10d_r2_flatten",
            "val/target_edge_10d_r2_flatten",
        )
        return series, "optuna_objective_target_edge_10d_r2_flatten"
    if mode in ("target_mean_r2", "target_edge_feature_mean_r2"):
        series = _v4_macro_series(
            val,
            "target_mean_r2",
            "val_target_mean_r2",
            "val/target_mean_r2",
        )
        return series, "optuna_objective_target_mean_r2"
    if mode in ("target_v4_main_verified_r2", "amount_r2_main_target_rows"):
        series = _v4_macro_series(
            val,
            "eval_secondary_amount_r2_by_process",
            "val_eval_secondary_amount_r2_by_process",
            "amount_r2_main_target_rows",
            "val_amount_r2_main_target_rows",
            "process_balanced_main_target_r2_main_verified",
            "val_process_balanced_main_target_r2_main_verified",
            "target_v4_main_verified_r2",
            "val_target_v4_main_verified_r2",
        )
        return series, "optuna_objective_amount_r2_main_target_rows"
    if mode == "target_v4_formula_available_r2":
        series = _v4_macro_series(
            val,
            "target_v4_macro_r2_formula_available",
            "val_target_v4_macro_r2_formula_available",
            "val/target_v4_macro_r2_formula_available",
        )
        return series, "optuna_objective_target_v4_formula_available_r2"
    if mode in ("target_v4_frac_main_verified_r2", "frac_r2_main_target_rows"):
        series = _v4_macro_series(
            val,
            "eval_primary_frac_r2_by_process",
            "val_eval_primary_frac_r2_by_process",
            "val/eval_primary_frac_r2_by_process",
            "frac_r2_main_target_rows",
            "val_frac_r2_main_target_rows",
            "process_balanced_main_target_frac_r2_main_verified",
            "val_process_balanced_main_target_frac_r2_main_verified",
            "target_v4_frac_main_verified_r2",
            "val_target_v4_frac_main_verified_r2",
        )
        return series, "optuna_objective_frac_r2_main_target_rows"
    series = _weighted_answer_r2_series(val, preset, w_h2=w_h2, w_co2=w_co2)
    return series, "optuna_objective_legacy_answer_fraction_r2"


def _mean_finite_floats(values: Sequence[float]) -> float:
    xs = [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(float(x))]
    if not xs:
        return float("nan")
    return float(sum(xs) / len(xs))


def _answer_edge_row_at(val: Mapping[str, Any], idx: int, preset: str) -> dict[str, float]:
    if preset == "edge_all":
        h2_m = _pick_val_at(val, idx, "target_h2_mae", "metric_target_h2_mae")
        c2_m = _pick_val_at(val, idx, "tailgas_co2_mae", "metric_tailgas_co2_mae")
        h2_r = _pick_val_at(val, idx, "target_h2_rmse", "metric_target_h2_rmse")
        c2_r = _pick_val_at(val, idx, "tailgas_co2_rmse", "metric_tailgas_co2_rmse")
        h2_sp = _pick_val_at(val, idx, "target_h2_smape", "metric_target_h2_smape")
        c2_sp = _pick_val_at(val, idx, "tailgas_co2_smape", "metric_tailgas_co2_smape")
        h2_r2 = _pick_val_at(val, idx, "metric_target_r2", "target_h2_r2", "metric_target_h2_r2")
        c2_r2 = _pick_val_at(val, idx, "metric_tailgas_r2", "tailgas_co2_r2", "metric_tailgas_co2_r2")
        loss_t = _pick_val_at(val, idx, "loss_total")
        base: dict[str, float] = {
            "loss_total": loss_t,
            "target_h2_mae": h2_m,
            "tailgas_co2_mae": c2_m,
            "target_h2_rmse": h2_r,
            "tailgas_co2_rmse": c2_r,
            "target_h2_smape": h2_sp,
            "tailgas_co2_smape": c2_sp,
            "metric_target_r2": h2_r2,
            "metric_tailgas_r2": c2_r2,
            "metric_answer_all_targets_mean_r2": _pick_val_at(
                val, idx, "metric_answer_all_targets_mean_r2"
            ),
        }
        for k, arr in val.items():
            if k in base or k == "epoch":
                continue
            if not isinstance(arr, (list, tuple)) or idx >= len(arr):
                continue
            ks = str(k)
            if (ks.startswith("answer_target_") or ks.startswith("answer_tailgas_")) and (
                ks.endswith("_mae") or ks.endswith("_rmse") or ks.endswith("_r2") or ks.endswith("_smape")
            ):
                try:
                    base[ks] = float(arr[idx])
                except (TypeError, ValueError):
                    pass
        return base
    h2_m = _pick_val_at(val, idx, "metric_target_mae")
    c2_m = _pick_val_at(val, idx, "metric_tailgas_mae")
    h2_r = _pick_val_at(val, idx, "metric_target_rmse")
    c2_r = _pick_val_at(val, idx, "metric_tailgas_rmse")
    h2_r2 = _pick_val_at(val, idx, "metric_target_r2")
    c2_r2 = _pick_val_at(val, idx, "metric_tailgas_r2")
    loss_t = _pick_val_at(val, idx, "loss_total")
    return {
        "loss_total": loss_t,
        "metric_target_mae": h2_m,
        "metric_tailgas_mae": c2_m,
        "metric_target_rmse": h2_r,
        "metric_tailgas_rmse": c2_r,
        "metric_target_r2": h2_r2,
        "metric_tailgas_r2": c2_r2,
        "metric_target_smape": float("nan"),
        "metric_tailgas_smape": float("nan"),
    }


def build_optuna_trial_report(
    run_dir: Path,
    *,
    preset: str,
    w_h2: float,
    w_co2: float,
    optuna_objective: str = "target_edge_property_mean_r2",
) -> dict[str, Any]:
    hist_path = run_dir / "train_val_history.json"
    if not hist_path.is_file():
        raise RuntimeError(f"missing history file: {hist_path}")
    payload = json.loads(hist_path.read_text(encoding="utf-8"))
    val = payload.get("val", {}) or {}

    objective_series, obj_name = _objective_series_and_name(
        val, preset, optuna_objective=optuna_objective, w_h2=w_h2, w_co2=w_co2
    )
    if not objective_series:
        raise RuntimeError(
            f"history missing objective series for {obj_name!r} under {hist_path} "
            f"(preset={preset!r}, optuna_objective={optuna_objective!r})"
        )
    n = len(objective_series)
    wh, wc, wsum = _norm_weights(w_h2, w_co2)
    weight_note: dict[str, Any] = {"h2": float(wh), "co2": float(wc), "sum": float(wsum)}

    best_i: int | None = None
    best_v = float("-inf")
    for i, v in enumerate(objective_series):
        fv = float(v)
        if math.isfinite(fv) and fv > best_v:
            best_v = fv
            best_i = i
    if best_i is None:
        raise RuntimeError(f"all epochs invalid (NaN/Inf) for objective R² under {hist_path}")
    epochs = _float_series(val, "epoch")
    best_epoch = float(epochs[best_i]) if best_i < len(epochs) else float("nan")

    def _weighted_answer_metrics(idx: int) -> dict[str, float]:
        row = _answer_edge_row_at(val, idx, preset)
        wh, wc, _ = _norm_weights(w_h2, w_co2)
        if preset == "edge_all":
            mae_w = _weighted_linear(row["target_h2_mae"], row["tailgas_co2_mae"], wh, wc)
            rmse_w = _weighted_rmse_from_rmse(
                row["target_h2_rmse"], row["tailgas_co2_rmse"], wh, wc
            )
            smape_w = _weighted_linear(
                row.get("target_h2_smape", float("nan")),
                row.get("tailgas_co2_smape", float("nan")),
                wh,
                wc,
            )
        else:
            mae_w = _weighted_linear(row["metric_target_mae"], row["metric_tailgas_mae"], wh, wc)
            rmse_w = _weighted_rmse_from_rmse(
                row["metric_target_rmse"], row["metric_tailgas_rmse"], wh, wc
            )
            smape_w = float("nan")
        r2_w = _combined_weighted_r2(
            row["metric_target_r2"], row["metric_tailgas_r2"], float(wh), float(wc)
        )
        out = dict(row)
        out["answer_targets_mae"] = mae_w
        out["answer_targets_rmse_heuristic"] = rmse_w
        out["answer_targets_smape_heuristic"] = smape_w
        out["answer_targets_r2"] = r2_w
        if preset == "edge_all":
            mr = float(row.get("metric_answer_all_targets_mean_r2", float("nan")))
            if math.isfinite(mr):
                out["metric_answer_all_targets_mean_r2"] = mr
        return out

    at_best = _weighted_answer_metrics(best_i)

    weighted_mae_series: list[float] = []
    for i in range(n):
        weighted_mae_series.append(_weighted_answer_metrics(i)["answer_targets_mae"])
    min_mae_i: int | None = None
    min_mae_v = float("inf")
    for i, v in enumerate(weighted_mae_series):
        if math.isfinite(v) and v < min_mae_v:
            min_mae_v = v
            min_mae_i = i
    at_min_mae: dict[str, Any] = {}
    if min_mae_i is not None:
        ep = float(epochs[min_mae_i]) if min_mae_i < len(epochs) else float("nan")
        at_min_mae = {
            "val_row_index": int(min_mae_i),
            "epoch": ep,
            "answer_targets_mae": float(min_mae_v),
            "answer_targets_r2": float(objective_series[min_mae_i])
            if min_mae_i < len(objective_series)
            else float("nan"),
        }

    final_i = n - 1
    at_final = _weighted_answer_metrics(final_i)

    defs = {
        "answer_targets_mae": "(w_h2*mae_h2 + w_co2*mae_co2)/(w_h2+w_co2)",
        "answer_targets_rmse_heuristic": "sqrt((w_h2*rmse_h2^2 + w_co2*rmse_co2^2)/(w_h2+w_co2))",
        "answer_targets_smape_heuristic": (
            "(w_h2*smape_h2 + w_co2*smape_co2)/(w_h2+w_co2) when logged; else NaN"
        ),
        "answer_targets_r2": (
            "양쪽 R² 유효 시 가중평균; 한쪽만 NaN이면 유효한 쪽 R²를 사용 "
            "(작은 val에서 SST≈0으로 tail/target R²가 NaN일 때)"
        ),
        "smape": "Symmetric MAPE in percent (mean over answer-edge points; edge_all only in history)",
    }
    if preset == "edge_all":
        defs["metric_answer_all_targets_mean_r2"] = (
            "참고용: 검증 시점 answer R²(H2, CO2, answer_* Frac_*) 산술평균 — Optuna objective 아님"
        )

    mode = str(optuna_objective or "target_edge_property_mean_r2")
    metric_key = obj_name
    if mode in ("target_edge_10d_r2_flatten", "val_target_edge_10d_r2_flatten", "flatten_r2"):
        metric_key = "val_target_edge_10d_r2_flatten"
    elif mode == "target_edge_property_mean_r2":
        metric_key = "val_target_edge_property_mean_r2"
    elif mode in ("target_mean_r2", "target_edge_feature_mean_r2"):
        metric_key = "val_target_mean_r2"
    elif mode == "target_v4_frac_main_verified_r2":
        metric_key = "val_process_balanced_main_target_frac_r2_main_verified"
    elif mode == "target_v4_main_verified_r2":
        metric_key = "val_target_v4_macro_r2_main_verified"
    elif mode == "target_v4_formula_available_r2":
        metric_key = "val_target_v4_macro_r2_formula_available"
    else:
        metric_key = "answer_targets_r2"

    def _pick_at(idx: int, *keys: str) -> Any:
        for k in keys:
            arr = val.get(k)
            if isinstance(arr, (list, tuple)) and idx < len(arr):
                return arr[idx]
        return None

    excluded_ids = _pick_at(best_i, "target_v4_excluded_target_ids") or val.get("target_v4_excluded_target_ids")
    if isinstance(excluded_ids, list):
        excluded_list = [str(x) for x in excluded_ids]
    else:
        excluded_list = []

    includes_h2o = mode in (
        "target_edge_property_mean_r2",
        "target_edge_10d_r2_flatten",
        "val_target_edge_10d_r2_flatten",
        "flatten_r2",
        "target_mean_r2",
        "target_edge_feature_mean_r2",
        "target_v4_main_verified_r2",
        "target_v4_formula_available_r2",
    )
    includes_h2 = includes_co2 = True
    if mode in ("target_edge_10d_r2_flatten", "val_target_edge_10d_r2_flatten", "flatten_r2"):
        legacy_note = (
            "target_edge_10d_r2_flatten = flattened R2 over all direct target-edge "
            "10D PI main-feature values in the validation split."
        )
    elif mode == "target_edge_property_mean_r2":
        legacy_note = (
            "target_edge_property_mean_r2 = arithmetic mean of the 10 pooled "
            "Target Property R2 values."
        )
    elif mode in ("target_mean_r2", "target_edge_feature_mean_r2"):
        legacy_note = (
            "target_mean_r2 = split-accumulated direct target-edge PI main feature R2; "
            "uses Temp, Pres, Mass_Flow, and Frac_* direct predictions."
        )
    elif mode.startswith("target_v4"):
        legacy_note = (
            "v4 macro objective averages per-target_id amount R² (H2/CO2/H2O where included); "
            "legacy answer_targets_r2 excluded H2O."
        )
    else:
        legacy_note = (
            "legacy_answer_fraction_r2 = weighted Frac_H2/Frac_CO2 on target_answer_edges slots only; "
            "answer edge weights are fixed during Optuna."
        )

    return {
        "preset": preset,
        "weights": weight_note,
        "objective": {
            "name": obj_name,
            "value": float(best_v),
            "best_val_row_index": int(best_i),
            "best_epoch": best_epoch,
            "objective_name": mode,
            "objective_metric_key": metric_key,
            "objective_granularity": "target_property_pooled"
            if mode == "target_edge_property_mean_r2"
            else "target_edge_10d_flatten"
            if mode in ("target_edge_10d_r2_flatten", "val_target_edge_10d_r2_flatten", "flatten_r2")
            else "target_edge_feature"
            if mode in ("target_mean_r2", "target_edge_feature_mean_r2")
            else ("target_row" if mode.startswith("target_v4") else "legacy_slot"),
            "includes_h2": includes_h2,
            "includes_co2": includes_co2,
            "includes_h2o": includes_h2o,
            "included_target_ids": _pick_at(best_i, "target_v4_included_target_ids"),
            "excluded_target_ids": excluded_list,
            "metric_definition": defs.get(metric_key, obj_name),
            "legacy_compatibility_note": legacy_note,
        },
        "at_best_objective": at_best,
        "at_min_weighted_mae": at_min_mae,
        "at_final_val_row": at_final,
        "definitions": defs,
    }


def _flatten_trial_for_table(trial: Any, report: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "trial_number": int(trial.number),
        "trial_state": str(trial.state),
        "objective_value": report["objective"]["value"],
        "objective_answer_targets_r2": report["objective"]["value"],
        "best_r2_val_row_index": report["objective"]["best_val_row_index"],
        "best_r2_epoch": report["objective"]["best_epoch"],
        "run_dir": trial.user_attrs.get("run_dir", ""),
    }
    for k, v in dict(trial.params).items():
        row[f"param_{k}"] = v
    ab = report.get("at_best_objective") or {}
    for k, v in ab.items():
        row[f"at_best_{k}"] = v
    mm = report.get("at_min_weighted_mae") or {}
    for k, v in mm.items():
        row[f"at_min_mae_{k}"] = v
    af = report.get("at_final_val_row") or {}
    for k, v in af.items():
        row[f"at_final_{k}"] = v
    return row


def _save_optuna_study_artifacts(
    study: optuna.Study,
    out_root: Path,
    *,
    no_plots: bool,
) -> None:
    rows: list[dict[str, Any]] = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        rep = t.user_attrs.get("optuna_metrics")
        if not isinstance(rep, dict):
            continue
        rows.append(_flatten_trial_for_table(t, rep))
    if not rows:
        print("[optuna] no completed trials with optuna_metrics; skipping summary CSV/plots")
        return
    df = pd.DataFrame(rows)
    csv_path = out_root / "optuna_trials_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[optuna] wrote {csv_path}")

    param_cols = [c for c in df.columns if str(c).startswith("param_")]
    extra_num = [c for c in ("objective_value",) if c in df.columns]
    numeric_cols = [c for c in param_cols + extra_num if c in df.columns]
    if len(numeric_cols) >= 2:
        corr_path = out_root / "optuna_numeric_correlations.csv"
        df[numeric_cols].corr(numeric_only=True).to_csv(corr_path, encoding="utf-8")
        print(f"[optuna] wrote {corr_path}")

    try:
        imp = optuna.importance.get_param_importances(study)
        imp_df = pd.DataFrame([{"param": k, "importance": float(v)} for k, v in imp.items()])
        imp_df = imp_df.sort_values("importance", ascending=False)
        imp_path = out_root / "optuna_param_importances.csv"
        imp_df.to_csv(imp_path, index=False, encoding="utf-8")
        print(f"[optuna] wrote {imp_path}")
    except Exception as exc:  # pragma: no cover
        (out_root / "optuna_param_importances_skipped.txt").write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )

    top_n = min(15, len(df))
    sort_col = "objective_value"
    if sort_col in df.columns:
        top = df.sort_values(sort_col, ascending=False).head(top_n)
    else:
        top = df.head(top_n)
    try:
        table = top.to_markdown(index=False)
    except Exception:
        table = top.to_string(index=False)
    md_lines = [
        f"# Optuna trials (top {top_n} by {sort_col})",
        "",
        table,
        "",
    ]
    (out_root / "optuna_trials_top.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[optuna] wrote {out_root / 'optuna_trials_top.md'}")

    if no_plots:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[optuna] matplotlib not installed; skipping plots")
        return

    rep_dir = out_root / "optuna_report"
    rep_dir.mkdir(parents=True, exist_ok=True)
    x_mae = df.get("at_best_answer_targets_mae")
    y_r2 = df.get("objective_value")
    if x_mae is not None and y_r2 is not None:
        mask = x_mae.notna() & y_r2.notna()
        if mask.any():
            fig, ax = plt.subplots(figsize=(6.5, 5.0))
            ax.scatter(x_mae[mask], y_r2[mask], alpha=0.75, edgecolors="k", linewidths=0.3)
            ax.set_xlabel("answer_targets_mae (at best-R2 val step)")
            ax.set_ylabel("Optuna objective (max over val)")
            ax.set_title("R² vs weighted MAE (completed trials)")
            fig.tight_layout()
            fig.savefig(rep_dir / "scatter_objective_vs_weighted_mae.png", dpi=140)
            plt.close(fig)
    x_sp = df.get("at_best_answer_targets_smape_heuristic")
    if x_sp is not None and y_r2 is not None:
        mask = x_sp.notna() & y_r2.notna()
        if mask.any():
            fig, ax = plt.subplots(figsize=(6.5, 5.0))
            ax.scatter(x_sp[mask], y_r2[mask], alpha=0.75, edgecolors="k", linewidths=0.3)
            ax.set_xlabel("answer_targets_smape heuristic (at best-R2 val step)")
            ax.set_ylabel("objective answer_targets_r2")
            ax.set_title("R² vs weighted SMAPE")
            fig.tight_layout()
            fig.savefig(rep_dir / "scatter_objective_vs_weighted_smape.png", dpi=140)
            plt.close(fig)
    if y_r2 is not None and y_r2.notna().any():
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        dropped = y_r2.dropna().astype(float)
        nb = max(1, min(20, max(5, int(dropped.shape[0]))))
        ax.hist(dropped, bins=nb)
        ax.set_xlabel("Optuna objective")
        ax.set_title("Objective distribution")
        fig.tight_layout()
        fig.savefig(rep_dir / "hist_objective_r2.png", dpi=140)
        plt.close(fig)
    mae_col, r2_col = "at_min_mae_answer_targets_mae", "at_min_mae_answer_targets_r2"
    if mae_col in df.columns and r2_col in df.columns:
        mask = df[mae_col].notna() & df[r2_col].notna()
        if mask.any():
            fig, ax = plt.subplots(figsize=(6.5, 5.0))
            ax.scatter(df.loc[mask, mae_col], df.loc[mask, r2_col], alpha=0.75, edgecolors="k", linewidths=0.3)
            ax.set_xlabel("min weighted MAE over val (any epoch)")
            ax.set_ylabel("R² at that epoch")
            ax.set_title("Min-MAE point: MAE vs R² tradeoff")
            fig.tight_layout()
            fig.savefig(rep_dir / "scatter_min_mae_vs_r2_at_that_step.png", dpi=140)
            plt.close(fig)
    print(f"[optuna] wrote plots under {rep_dir}")


def _build_best_hyperparameters_payload(
    study: optuna.Study,
    *,
    study_name: str,
    storage: str,
    preset: str,
    process_filter: int,
    process_scope: str,
    config: str,
    max_epochs: int,
    metric_w_h2: float,
    metric_w_co2: float,
    out_root: Path,
    saved_after_completed_trial_number: int | None = None,
) -> dict[str, Any] | None:
    """스터디에 완료된 trial이 없거나 best를 정할 수 없으면 None."""
    try:
        best_trial = study.best_trial
    except ValueError:
        return None
    payload: dict[str, Any] = {
        "study_name": study_name,
        "storage": storage,
        "preset": preset,
        "process_filter": process_filter,
        "process_scope": process_scope,
        "best_value": float(study.best_value),
        "best_trial_number": int(best_trial.number),
        "best_params": dict(study.best_params),
        "best_user_attrs": dict(best_trial.user_attrs),
        "n_trials": int(len(study.trials)),
        "metric": {
            "name": str(study.best_trial.user_attrs.get("optuna_objective_mode", "target_edge_property_mean_r2")),
            "definition": (
                "max over val steps of the selected optuna_objective series "
                "(legacy_answer_fraction_r2 = weighted Frac R²; "
                "target_edge_10d_r2_flatten = val_target_edge_10d_r2_flatten; "
                "target_v4_main_verified_r2 = val_target_v4_macro_r2_main_verified)"
            ),
            "weights_default": {"h2": float(metric_w_h2), "co2": float(metric_w_co2)},
        },
        "optuna_objective_mode": str(study.best_trial.user_attrs.get("optuna_objective_mode", "")),
        "objective_name": str(
            study.best_trial.user_attrs.get("optuna_objective_mode", "target_edge_property_mean_r2")
        ),
        "objective_metric_key": str(
            (study.best_trial.user_attrs.get("optuna_metrics") or {})
            .get("objective", {})
            .get("objective_metric_key", "val_target_edge_property_mean_r2")
        ),
        "includes_h2": True,
        "includes_co2": True,
        "includes_h2o": str(study.best_trial.user_attrs.get("optuna_objective_mode", ""))
        in {
            "target_edge_property_mean_r2",
            "target_edge_10d_r2_flatten",
            "val_target_edge_10d_r2_flatten",
            "flatten_r2",
            "target_mean_r2",
            "target_edge_feature_mean_r2",
            "target_v4_main_verified_r2",
            "target_v4_formula_available_r2",
        },
        "legacy_compatibility_note": (
            "Historical studies with legacy_answer_fraction_r2 exclude H2O from objective; "
            "answer edge weights are fixed during current Optuna runs."
        ),
        "config": str(config),
        "max_epochs": int(max_epochs),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "artifacts": {
            "trials_summary_csv": str((out_root / "optuna_trials_summary.csv").resolve()),
            "trials_top_md": str((out_root / "optuna_trials_top.md").resolve()),
            "report_plots_dir": str((out_root / "optuna_report").resolve()),
        },
    }
    if saved_after_completed_trial_number is not None:
        payload["saved_after_completed_trial_number"] = int(saved_after_completed_trial_number)
    return payload


def _read_disk_best_value(best_path: Path) -> float | None:
    """``best_hyperparameters.json`` 의 ``best_value`` 가 있으면 반환, 없거나 파싱 실패 시 None."""
    if not best_path.is_file():
        return None
    try:
        data = json.loads(best_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    v = data.get("best_value")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    fv = float(v)
    if not math.isfinite(fv):
        return None
    return fv


def _save_best_checkpoint(
    study: optuna.Study,
    out_root: Path,
    *,
    study_name: str,
    storage: str,
    preset: str,
    process_filter: int,
    process_scope: str,
    config: str,
    max_epochs: int,
    metric_w_h2: float,
    metric_w_co2: float,
    completed_trial_number: int,
    completed_trial_state: optuna.trial.TrialState,
) -> bool:
    """스터디 best가 디스크에 저장된 best보다 좋아질 때만 ``best_hyperparameters.json`` 을 덮어쓰고
    ``best_progress.jsonl`` 에 한 줄 append 한다. (maximize)

    Returns:
        파일을 갱신했으면 True.
    """
    payload = _build_best_hyperparameters_payload(
        study,
        study_name=study_name,
        storage=storage,
        preset=preset,
        process_filter=process_filter,
        process_scope=process_scope,
        config=config,
        max_epochs=max_epochs,
        metric_w_h2=metric_w_h2,
        metric_w_co2=metric_w_co2,
        out_root=out_root,
        saved_after_completed_trial_number=completed_trial_number,
    )
    if payload is None:
        return False
    best_path = out_root / "best_hyperparameters.json"
    new_val = float(payload["best_value"])
    old_val = _read_disk_best_value(best_path)
    if old_val is not None and not (new_val > old_val):
        return False
    best_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    progress_path = out_root / "best_progress.jsonl"
    line = {
        "saved_after_completed_trial_number": int(completed_trial_number),
        "completed_trial_state": str(completed_trial_state),
        "best_trial_number": int(payload["best_trial_number"]),
        "best_value": float(payload["best_value"]),
        "best_params": dict(payload["best_params"]),
        "saved_at": payload["saved_at"],
    }
    with progress_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(
        f"[optuna] checkpoint: wrote {best_path.name} "
        f"(after trial {completed_trial_number}, best trial {payload['best_trial_number']}, "
        f"best_value={payload['best_value']:.6g})"
    )
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for process-surrogate (no W&B).",
    )
    parser.add_argument("--config", required=True, help="experiment YAML 경로")
    parser.add_argument(
        "--process-filter",
        type=int,
        default=0,
        help="공정 ID (1..10). 0 또는 생략 시 전체 공정 설정(edge_all_processes) 사용.",
    )
    parser.add_argument(
        "--preset",
        choices=("edge_all", "multitask"),
        default="edge_all",
        help="탐색 프리셋 (default: edge_all)",
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/optuna",
        help="trial 출력 루트 디렉토리 (Process{N} 또는 All 하위에 trial 폴더 생성)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="",
        help="Optuna study 이름. 비우면 <preset>_r2_PNN 형식의 새 기본 이름 사용.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="",
        help="Optuna RDB storage URL. 비우면 outputs/optuna/Process{N} 또는 All/study.db 사용.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--metric-w-h2", type=float, default=5.0,
        help="answer_targets_r2 계산 시 H2 가중치 fallback (trial에 없을 때, 기본 5.0)",
    )
    parser.add_argument(
        "--metric-w-co2", type=float, default=5.0,
        help="answer_targets_r2 계산 시 CO2 가중치 fallback (trial에 없을 때, 기본 5.0)",
    )
    parser.add_argument(
        "--optuna-objective",
        choices=(
            "target_edge_property_mean_r2",
            "target_edge_10d_r2_flatten",
            "val_target_edge_10d_r2_flatten",
            "flatten_r2",
            "target_mean_r2",
            "target_edge_feature_mean_r2",
            "legacy_answer_fraction_r2",
            "frac_r2_main_target_rows",
            "amount_r2_main_target_rows",
            "target_v4_frac_main_verified_r2",
            "target_v4_main_verified_r2",
            "target_v4_formula_available_r2",
        ),
        default="target_edge_property_mean_r2",
        help=(
            "Optuna maximize: target_edge_property_mean_r2 "
            "(default; arithmetic mean of 10 pooled Target Property R2 values), "
            "target_mean_r2 (strict target-row macro), "
            "target_edge_10d_r2_flatten, "
            "eval_legacy_answer_edge_frac_r2, "
            "eval_primary_frac_r2_by_process (Frac per target_id), "
            "eval_secondary_amount_r2_by_process, or formula-ok macro. "
            "Names with target_v4_* are deprecated aliases."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optuna study.optimize timeout (초). 비우면 무제한.",
    )
    parser.add_argument(
        "--save-trial-stdout",
        action="store_true",
        help="각 trial의 stdout/stderr 를 trial_*.log 로 저장",
    )
    parser.add_argument(
        "--keep-trial-artifacts",
        action="store_true",
        help=(
            "trial 종료 후 대용량 산출물(edge_predictions, checkpoints, plots, train/val/test export) "
            "자동 삭제를 끔 (기본: 삭제함)"
        ),
    )
    parser.add_argument(
        "--no-enqueue-best",
        dest="enqueue_best",
        action="store_false",
        help=(
            "이전 best_hyperparameters.json 의 best_params 를 다음 trial에 넣지 않음 "
            "(기본은 enqueue; 중복 params면 스킵). Python 3.8 호환"
        ),
    )
    parser.add_argument(
        "--no-analysis-plots",
        action="store_true",
        help="matplotlib 기반 optuna_report 플롯 생략 (CSV/MD/JSON 만 저장)",
    )
    parser.add_argument(
        "--skip-startup-debug",
        action="store_true",
        help="각 trial 학습에 --skip-startup-debug 전달 (스모크/빠른 튜닝용)",
    )
    parser.add_argument(
        "--warmstart-best-json",
        default="",
        help=(
            "이전 ``best_hyperparameters.json`` 경로. 비우면 "
            "``--auto-warmstart-best``(기본 ON) 또는 ``--no-warmstart-auto-discover`` 미사용 시 "
            "``outputs/optuna/Process{N}/best_hyperparameters.json`` 등 자동 탐색."
        ),
    )
    parser.add_argument(
        "--auto-warmstart-best",
        action="store_true",
        help=(
            "이전 스터디 ``best_hyperparameters.json`` 자동 탐색을 명시적으로 켭니다(기본도 ON). "
            "``--no-warmstart-auto-discover`` 와 함께 주면 본 옵션이 우선합니다."
        ),
    )
    parser.add_argument(
        "--no-warmstart-auto-discover",
        action="store_true",
        help="warmstart JSON 경로가 비어 있을 때 기존 스터디 결과 파일 자동 탐색 끔 (--auto-warmstart-best 가 있으면 무시)",
    )
    parser.set_defaults(enqueue_best=True)
    args = parser.parse_args()

    pf = int(args.process_filter)
    all_processes_mode = pf <= 0

    scope_tag = "All" if all_processes_mode else f"Process{pf}"
    out_root = (PROJECT_ROOT / args.output_root / scope_tag).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    warm_path: Path | None = None
    warm_flat: dict[str, Any] = {}
    ws_arg = (args.warmstart_best_json or "").strip()
    if ws_arg:
        wp = Path(ws_arg)
        warm_path = wp if wp.is_absolute() else (PROJECT_ROOT / wp).resolve()
        warm_flat = load_best_params_from_json(warm_path)
    elif (bool(getattr(args, "auto_warmstart_best", False)) or not bool(args.no_warmstart_auto_discover)) and not ws_arg:
        wp = discover_best_hyperparameters_json(project_root=PROJECT_ROOT, process_filter=pf)
        if wp is not None:
            warm_path = wp
            warm_flat = load_best_params_from_json(warm_path)
    if warm_flat:
        print(f"[optuna] warmstart: {len(warm_flat)} params from {warm_path}")
        (out_root / "warmstart_source.json").write_text(
            json.dumps({"path": str(warm_path), "n_params": len(warm_flat)}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_root / "warmstart_runtime_overrides.json").write_text(
            json.dumps(
                flat_optuna_params_to_runtime_overrides(warm_flat, preset=str(args.preset)),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    suggester = _make_suggester(args.preset, warm_flat or None, all_processes=all_processes_mode)

    study_name = args.study_name.strip() or (
        f"{args.preset}_r2_all" if all_processes_mode else f"{args.preset}_r2_P{pf:02d}"
    )
    if args.storage.strip():
        storage = args.storage.strip()
    else:
        storage = f"sqlite:///{(out_root / 'study.db').as_posix()}"

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=True,
    )
    print(
        f"[optuna] study={study_name!r} storage={storage} "
        f"process_scope={scope_tag} n_trials={args.n_trials} preset={args.preset}"
    )
    if warm_flat:
        if _has_existing_trial_with_params(study, warm_flat):
            print("[optuna] enqueue warmstart skipped: same params already in study trials")
        else:
            study.enqueue_trial(warm_flat)
            print(
                f"[optuna] enqueue warmstart: queued {len(warm_flat)} params "
                f"(source={warm_path})"
            )
    if args.enqueue_best:
        best_params_path = out_root / "best_hyperparameters.json"
        best_params = load_best_params_from_json(best_params_path)
        if best_params:
            if _has_existing_trial_with_params(study, best_params):
                print("[optuna] enqueue-best skipped: same params already in study trials")
            else:
                study.enqueue_trial(best_params)
                print(
                    f"[optuna] enqueue-best: queued params from {best_params_path.name} "
                    f"(count={len(best_params)})"
                )
        else:
            print("[optuna] enqueue-best skipped: no valid best_params found")

    out_root_rel = out_root.relative_to(PROJECT_ROOT).as_posix()

    def objective(trial: optuna.Trial) -> float:
        suggested = suggester(trial)
        run_name = f"trial_{trial.number:04d}_{datetime.now():%H%M%S}"
        run_dir = out_root / run_name

        def _maybe_prune_trial_artifacts() -> None:
            if args.keep_trial_artifacts:
                return
            ckpt_dir = (PROJECT_ROOT / out_root_rel / "checkpoints" / run_name).resolve()
            pruned = _prune_large_optuna_trial_artifacts(run_dir=run_dir, checkpoint_dir=ckpt_dir)
            if pruned:
                print(
                    f"[optuna] trial {trial.number} pruned large artifacts: "
                    + ", ".join(pruned[:8])
                    + (" ..." if len(pruned) > 8 else "")
                )

        try:
            overrides_payload: Dict[str, Any] = dict(suggested)
            overrides_payload["train"] = _optuna_trial_train_overrides(
                suggested.get("train"),
                optuna_objective=str(args.optuna_objective),
            )
            overrides_payload["output_dir"] = out_root_rel
            overrides_payload["save_dir"] = f"{out_root_rel}/checkpoints"
            overrides_payload["log_dir"] = f"{out_root_rel}/logs"
            overrides_payload["experiment_name"] = run_name

            with tempfile.NamedTemporaryFile(
                "w",
                suffix=".json",
                prefix=f"optuna_{run_name}_",
                delete=False,
                encoding="utf-8",
            ) as f:
                json.dump(overrides_payload, f)
                ovr_path = Path(f.name)

            log_path = (out_root / f"{run_name}.log") if args.save_trial_stdout else None
            try:
                rc = _run_train_subprocess(
                    config=args.config,
                    overrides_path=ovr_path,
                    run_name=run_name,
                    max_epochs=args.max_epochs,
                    process_filter=pf,
                    log_path=log_path,
                    skip_startup_debug=bool(getattr(args, "skip_startup_debug", False)),
                )
            finally:
                try:
                    ovr_path.unlink()
                except OSError:
                    pass

            if rc != 0:
                fail_payload = {
                    "train_subprocess_rc": int(rc),
                    "run_name": run_name,
                    "suggested_params_flat": {**suggested.get("train", {}), **suggested.get("model", {})},
                }
                try:
                    (run_dir / "optuna_trial_failed.json").write_text(
                        json.dumps(fail_payload, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except OSError:
                    pass
                trial.set_user_attr("train_subprocess_rc", int(rc))
                trial.set_user_attr("run_dir", str(run_dir))
                trial.set_user_attr("run_name", run_name)
                print(
                    f"[optuna] trial {trial.number} train failed rc={rc} "
                    f"(logged {run_dir / 'optuna_trial_failed.json'}); returning worst objective."
                )
                _maybe_prune_trial_artifacts()
                return float("-1.0e300")

            w_h2 = float(suggested.get("train", {}).get("answer_edge_weight_h2", args.metric_w_h2))
            w_co2 = float(suggested.get("train", {}).get("answer_edge_weight_co2", args.metric_w_co2))
            w_h2o = float(suggested.get("train", {}).get("answer_edge_weight_h2o", 5.0))
            report = build_optuna_trial_report(
                run_dir,
                preset=args.preset,
                w_h2=w_h2,
                w_co2=w_co2,
                optuna_objective=str(args.optuna_objective),
            )
            trial.set_user_attr("optuna_objective_mode", str(args.optuna_objective))
            value = float(report["objective"]["value"])
            (run_dir / "optuna_trial_metrics.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            _maybe_prune_trial_artifacts()

            trial.set_user_attr("run_dir", str(run_dir))
            trial.set_user_attr("run_name", run_name)
            trial.set_user_attr("answer_edge_weight_h2", float(w_h2))
            trial.set_user_attr("answer_edge_weight_co2", float(w_co2))
            trial.set_user_attr("answer_edge_weight_h2o", float(w_h2o))
            trial.set_user_attr("optuna_metrics", report)
            return value
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            err_payload = {
                "trial_number": int(trial.number),
                "run_name": run_name,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "suggested_params_flat": {**suggested.get("train", {}), **suggested.get("model", {})},
            }
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "optuna_trial_failed.json").write_text(
                    json.dumps(err_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError:
                pass
            trial.set_user_attr("objective_exception", f"{type(exc).__name__}: {exc}")
            trial.set_user_attr("run_dir", str(run_dir))
            trial.set_user_attr("run_name", run_name)
            print(
                f"[optuna] trial {trial.number} objective error: {type(exc).__name__}: {exc} "
                f"(see {run_dir / 'optuna_trial_failed.json'})"
            )
            _maybe_prune_trial_artifacts()
            return float("-1.0e300")

    def _after_each_trial(study_: optuna.Study, trial_: optuna.trial.FrozenTrial) -> None:
        _save_best_checkpoint(
            study_,
            out_root,
            study_name=study_name,
            storage=storage,
            preset=str(args.preset),
            process_filter=pf,
            process_scope=scope_tag,
            config=str(args.config),
            max_epochs=int(args.max_epochs),
            metric_w_h2=float(args.metric_w_h2),
            metric_w_co2=float(args.metric_w_co2),
            completed_trial_number=int(trial_.number),
            completed_trial_state=trial_.state,
        )

    try:
        study.optimize(
            objective,
            n_trials=int(args.n_trials),
            timeout=args.timeout,
            gc_after_trial=True,
            callbacks=[_after_each_trial],
        )
    except ValueError as exc:
        msg = str(exc)
        if "CategoricalDistribution does not support dynamic value space" in msg:
            recommended_study_name = (
                f"{args.preset}_r2_all" if all_processes_mode else f"{args.preset}_r2_P{pf:02d}"
            )
            raise RuntimeError(
                "Optuna study search space mismatch detected. "
                "This usually means the study already contains trials created with older "
                "categorical choices. Use a new study name or a new output root.\n"
                f"Recommended: --study-name {recommended_study_name} "
                "--output-root outputs/optuna_r2"
            ) from exc
        raise

    _save_optuna_study_artifacts(study, out_root, no_plots=bool(args.no_analysis_plots))

    if not study.best_trial:
        print("[optuna] no completed trial; nothing to save")
        return

    last_t: optuna.trial.FrozenTrial | None = None
    for t in reversed(study.trials):
        if t.state == optuna.trial.TrialState.COMPLETE:
            last_t = t
            break

    best_path = out_root / "best_hyperparameters.json"
    if last_t is None:
        print("[optuna] no completed trial; skipping best_hyperparameters.json")
        return

    wrote = _save_best_checkpoint(
        study,
        out_root,
        study_name=study_name,
        storage=storage,
        preset=str(args.preset),
        process_filter=pf,
        process_scope=scope_tag,
        config=str(args.config),
        max_epochs=int(args.max_epochs),
        metric_w_h2=float(args.metric_w_h2),
        metric_w_co2=float(args.metric_w_co2),
        completed_trial_number=int(last_t.number),
        completed_trial_state=last_t.state,
    )

    payload = _build_best_hyperparameters_payload(
        study,
        study_name=study_name,
        storage=storage,
        preset=str(args.preset),
        process_filter=pf,
        process_scope=scope_tag,
        config=str(args.config),
        max_epochs=int(args.max_epochs),
        metric_w_h2=float(args.metric_w_h2),
        metric_w_co2=float(args.metric_w_co2),
        out_root=out_root,
        saved_after_completed_trial_number=int(last_t.number),
    )
    if payload is None:
        return
    if wrote:
        print(f"[optuna] wrote best params to {best_path}")
    else:
        print(
            f"[optuna] on-disk {best_path.name} left unchanged "
            f"(study best_value={float(payload['best_value']):.6g})"
        )
    _payload_txt = json.dumps(payload, indent=2, ensure_ascii=False)
    try:
        print(_payload_txt)
    except UnicodeEncodeError:
        print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
