from __future__ import annotations

from typing import Any, Dict


def build_default_sweep_config(
    *,
    metric_name: str = "val/loss",
    metric_goal: str = "minimize",
    method: str = "bayes",
) -> Dict[str, Any]:
    """Search space for multitask / target+tailgas training (legacy default)."""
    if method not in {"bayes", "random"}:
        raise ValueError("sweep method must be one of: bayes, random")
    if metric_goal not in {"minimize", "maximize"}:
        raise ValueError("metric_goal must be one of: minimize, maximize")

    return {
        "method": method,
        "metric": {"name": metric_name, "goal": metric_goal},
        "parameters": {
            "train.learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 5e-3},
            "train.weight_decay": {"distribution": "log_uniform_values", "min": 1e-7, "max": 1e-2},
            "train.batch_size": {"values": [16, 32, 64]},
            "train.optimizer": {"values": ["adam", "adamw"]},
            "train.scheduler": {"values": ["none", "cosine", "plateau"]},
            "train.scheduler_t_max": {"values": [100, 200, 300]},
            "train.scheduler_step_size": {"values": [25, 50, 75]},
            "train.scheduler_gamma": {"values": [0.5, 0.7, 0.9]},
            "train.gradient_clip_norm": {"values": [0.0, 1.0, 5.0]},
            "train.task_loss_weights.target": {"values": [0.2, 0.3, 0.5, 1.0]},
            "train.task_loss_weights.tailgas": {"values": [0.8, 1.0, 1.2, 1.5]},
            "model.hidden_dim": {"values": [256, 320, 384, 448, 512]},
            "model.num_layers": {"values": [3, 4, 5]},
            "model.dropout": {"values": [0.0, 0.05, 0.1, 0.2, 0.3]},
            "model.attn_hidden_dim": {"values": [64, 96, 128, 192, 256]},
            "model.role_emb_dim": {"values": [32, 48, 64, 96, 128]},
            "model.unit_emb_dim": {"values": [32, 48, 64, 96, 128]},
            "model.hx_role_emb_dim": {"values": [16, 24, 32, 48, 64]},
        },
    }


def build_edge_all_sweep_config(
    *,
    metric_name: str = "val/answer_targets_mae",
    metric_goal: str = "minimize",
    method: str = "bayes",
) -> Dict[str, Any]:
    """Search space for task_mode=edge_all (all-edge stream regression + EdgeDecoder)."""
    if method not in {"bayes", "random"}:
        raise ValueError("sweep method must be one of: bayes, random")
    if metric_goal not in {"minimize", "maximize"}:
        raise ValueError("metric_goal must be one of: minimize, maximize")

    return {
        "method": method,
        "metric": {"name": metric_name, "goal": metric_goal},
        "parameters": {
            "train.learning_rate": {"distribution": "log_uniform_values", "min": 3e-5, "max": 1e-3},
            "train.weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 5e-3},
            "train.batch_size": {"values": [8, 16, 32]},
            "train.optimizer": {"values": ["adam", "adamw"]},
            "train.scheduler": {"values": ["none", "cosine", "plateau"]},
            "train.scheduler_t_max": {"values": [100, 200]},
            "train.gradient_clip_norm": {"values": [0.5, 1.0, 5.0]},
            "train.loss_type_edge_all": {"values": ["smooth_l1", "l1", "mse"]},
            "train.mixed_precision": {"values": [False, True]},
            "model.hidden_dim": {"values": [256, 320, 384, 448, 512]},
            "model.num_layers": {"values": [3, 4, 5]},
            "model.dropout": {"values": [0.0, 0.1, 0.2]},
            "model.attn_hidden_dim": {"values": [64, 96, 128, 192, 256]},
            "model.role_emb_dim": {"values": [32, 48, 64, 96, 128]},
            "model.unit_emb_dim": {"values": [32, 48, 64, 96, 128]},
            "model.hx_role_emb_dim": {"values": [16, 24, 32, 48, 64]},
            "model.edge_stream_role_emb_dim": {"values": [16, 24, 32, 48]},
            "model.edge_stream_id_emb_dim": {"values": [24, 32, 48, 64]},
            "model.edge_decoder_hidden_dim": {"values": [256, 320, 384, 512]},
            "model.edge_decoder_dropout": {"values": [0.0, 0.1, 0.2]},
        },
    }


def build_sweep_config(
    preset: str,
    *,
    metric_name: str = "val/loss",
    metric_goal: str = "minimize",
    method: str = "bayes",
) -> Dict[str, Any]:
    """Dispatch W&B sweep definition by training preset name."""
    key = preset.strip().lower()
    if key in {"default", "multitask", "base"}:
        return build_default_sweep_config(
            metric_name=metric_name, metric_goal=metric_goal, method=method
        )
    if key in {"edge_all", "edge-all", "edgeall"}:
        return build_edge_all_sweep_config(
            metric_name=metric_name, metric_goal=metric_goal, method=method
        )
    raise ValueError(
        f"Unknown sweep preset {preset!r}. Use one of: multitask, edge_all."
    )
