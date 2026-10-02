"""Load prior Optuna ``best_hyperparameters.json`` and merge into training / search space."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

# Flat Optuna param names -> nested runtime overrides (train / model).
_TRAIN_KEYS_EDGE_ALL: frozenset[str] = frozenset(
    {
        "learning_rate",
    }
)
_MODEL_KEYS: frozenset[str] = frozenset(
    {
        "hidden_dim",
        "num_layers",
    }
)
_TRAIN_KEYS_MULTITASK: frozenset[str] = frozenset(
    {
        "learning_rate",
    }
)


def load_best_params_from_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    bp = payload.get("best_params")
    return dict(bp) if isinstance(bp, dict) else {}


def discover_best_hyperparameters_json(
    *,
    project_root: Path,
    process_filter: int,
    extra_roots: tuple[str, ...] = ("outputs/optuna", "outputs/optuna_r2", "outputs/optuna_smoke_tune_verify"),
) -> Path | None:
    """Pick the first existing ``best_hyperparameters.json`` for this process scope."""
    if process_filter and int(process_filter) > 0:
        scope_dirs = [f"Process{int(process_filter)}"]
    else:
        scope_dirs = ["All", "all"]
    for rel in extra_roots:
        base = (project_root / rel).resolve()
        for scope in scope_dirs:
            p = base / scope / "best_hyperparameters.json"
            if p.is_file():
                return p
    return None


def ensure_categorical_value(choices: list[Any], value: Any) -> list[Any]:
    """Include ``value`` in ``choices`` so a prior best is always sampleable."""
    out: list[Any] = list(choices)
    if value is None:
        return out
    for c in out:
        if c == value or (isinstance(c, (int, float)) and isinstance(value, (int, float)) and float(c) == float(value)):
            return out
    out.append(value)
    if all(isinstance(x, (int, float)) for x in out):
        return sorted(out, key=float)
    return sorted(out, key=str)


def flat_optuna_params_to_runtime_overrides(flat: Mapping[str, Any], *, preset: str) -> dict[str, Any]:
    """Map flat trial params to ``train_process_surrogate`` runtime shape (train/model keys)."""
    train_k = _TRAIN_KEYS_EDGE_ALL if preset == "edge_all" else _TRAIN_KEYS_MULTITASK
    train: dict[str, Any] = {}
    model: dict[str, Any] = {}
    for k, v in flat.items():
        if k in train_k:
            train[k] = v
        elif k in _MODEL_KEYS:
            model[k] = v
    out: dict[str, Any] = {}
    if train:
        out["train"] = train
    if model:
        out["model"] = model
    return out
