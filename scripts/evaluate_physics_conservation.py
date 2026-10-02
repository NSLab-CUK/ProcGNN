"""Evaluate held-out Mass/Component/Atom conservation for a trained PI model.

This is deliberately a post-hoc evaluator: it never performs an optimizer
step and reports how well a saved checkpoint obeys the three node-level
conservation equations on a supplied test manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _entry in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from process_graph.data.tabular_dataset import ProcessGraphTabularDataset  # noqa: E402
from process_graph.constants import STREAM_EDGE_FEATURE_SLOTS  # noqa: E402
from process_graph.experiment.config_builders import build_task_specs, model_yaml_to_encoder_config  # noqa: E402
from process_graph.experiment.edge_step_pi_training import (  # noqa: E402
    _ensure_pi_output_keys,
    compute_pi_node_balance_losses,
)
from process_graph.experiment.edge_step_training import build_target_edge_boolean_mask  # noqa: E402
from process_graph.experiment.loaders import load_experiment_config  # noqa: E402
from process_graph.models import ProcessSurrogateModel  # noqa: E402
from scripts.train_process_surrogate import _collate_for_experiment  # noqa: E402


TERMS = ("mass", "component", "atom")
THRESHOLDS = (("0.01", "0p01"), ("0.05", "0p05"), ("0.10", "0p10"))


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _checkpoint_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("model_state_dict") or payload.get("model")
    if not isinstance(state, Mapping):
        raise RuntimeError("Checkpoint has no model_state_dict/model mapping.")
    return state


def _resolve_y_edge_scaler(
    checkpoint_path: Path,
    explicit_path: str | None,
) -> tuple[Path, Mapping[str, Any]]:
    """Load the train-split stream-target scaler paired with ``best.pt``."""
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(_resolve(explicit_path))
    # .../<fold>/checkpoints/<run>/best.pt -> .../<fold>/<run>/y_edge_scaler.pt
    candidates.append(
        checkpoint_path.parent.parent.parent / checkpoint_path.parent.name / "y_edge_scaler.pt"
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise TypeError(f"y_edge_scaler payload must be a mapping: {candidate}")
        if any(payload.get(key) is None for key in ("mean", "std", "columns")):
            raise RuntimeError(f"y_edge_scaler is missing mean/std/columns: {candidate}")
        return candidate, payload
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Cannot find y_edge_scaler.pt paired with checkpoint. "
        f"Checked: {checked}. Pass --y-edge-scaler explicitly if the run was moved."
    )


def _as_number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate held-out Mass/Component/Atom conservation of a saved PI checkpoint."
    )
    parser.add_argument("--config", default="configs/experiment/pinn/model_260805_10d_frac1.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--y-edge-scaler",
        default=None,
        help="Optional path to y_edge_scaler.pt; inferred from --checkpoint by default.",
    )
    parser.add_argument("--manifest", required=True, help="Held-out split manifest, normally fold_NN/test.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-label", default="Proposed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    checkpoint_path = _resolve(args.checkpoint)
    manifest_path = _resolve(args.manifest)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Held-out manifest not found: {manifest_path}")

    device = torch.device(args.device)
    experiment = load_experiment_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint payload must be a mapping: {checkpoint_path}")
    oper_normalizer = checkpoint.get("oper_normalizer")
    if not isinstance(oper_normalizer, Mapping):
        raise RuntimeError("Checkpoint lacks the required train-split operating normalizer.")
    oper_mean = torch.as_tensor(oper_normalizer["mean"], dtype=torch.float32)
    oper_std = torch.as_tensor(oper_normalizer["std"], dtype=torch.float32)
    oper_std = torch.where(oper_std == 0, torch.ones_like(oper_std), oper_std)
    y_edge_scaler_path, pi_normalizer = _resolve_y_edge_scaler(
        checkpoint_path, args.y_edge_scaler
    )

    dataset = ProcessGraphTabularDataset(
        experiment.data.test_data_path,
        experiment.data,
        experiment.project_root,
        split_filter=None,
        split_manifest=manifest_path,
        oper_mean=oper_mean,
        oper_std=oper_std,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=_collate_for_experiment(experiment.data),
    )
    model = ProcessSurrogateModel(
        encoder_config=model_yaml_to_encoder_config(experiment.model, experiment.data),
        task_specs=build_task_specs(experiment.model, experiment.data),
    ).to(device)
    model.load_state_dict(_checkpoint_state(checkpoint), strict=True)
    model.eval()

    totals: dict[str, dict[str, float]] = {
        term: {
            "valid_count": 0.0,
            "abs_sum": 0.0,
            "sq_sum": 0.0,
            "abs_max": 0.0,
            "huber_sum": 0.0,
            **{f"le_{label}": 0.0 for _, label in THRESHOLDS},
        }
        for term in TERMS
    }
    graph_count = 0
    with torch.no_grad():
        for batch in loader:
            batch_data = {key: value.to(device) for key, value in batch.model_kwargs.items()}
            task_inputs = {
                name: {key: value.to(device) for key, value in payload.items()}
                for name, payload in batch.task_inputs.items()
            }
            n_edges = int(batch.targets["edge_stream"].shape[0])
            edge_columns = list(batch.edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)
            target_edge_mask, _ = build_target_edge_boolean_mask(
                train_cfg=experiment.train,
                edge_export_meta=batch.edge_export_meta,
                edge_target_columns=edge_columns,
                n_edges=n_edges,
                device=device,
            )
            model_batch = dict(batch_data)
            model_batch["target_edge_mask"] = target_edge_mask
            outputs = model(model_batch, task_inputs=task_inputs)
            outputs = _ensure_pi_output_keys(
                outputs,
                train_cfg=experiment.train,
                data_cfg=experiment.data,
                edge_target_columns=edge_columns,
                normalizer=pi_normalizer,
            )
            result = compute_pi_node_balance_losses(
                outputs=outputs,
                batch_data=batch_data,
                train_cfg=experiment.train,
                data_cfg=experiment.data,
                normalizer=pi_normalizer,
                edge_target_columns=edge_columns,
            )
            diagnostics = result["diagnostics"]
            graph_count += int(torch.unique(batch_data["batch"]).numel())
            for term in TERMS:
                count = _as_number(diagnostics.get(f"node_{term}_valid_count", 0.0))
                totals[term]["valid_count"] += count
                totals[term]["abs_sum"] += _as_number(diagnostics.get(f"node_{term}_residual_abs_sum", 0.0))
                totals[term]["sq_sum"] += _as_number(diagnostics.get(f"node_{term}_residual_sq_sum", 0.0))
                totals[term]["abs_max"] = max(
                    totals[term]["abs_max"], _as_number(diagnostics.get(f"node_{term}_residual_max", 0.0))
                )
                totals[term]["huber_sum"] += _as_number(diagnostics.get(f"loss_node_{term}", 0.0)) * count
                for _, label in THRESHOLDS:
                    totals[term][f"le_{label}"] += _as_number(
                        diagnostics.get(f"node_{term}_residual_abs_le_{label}_count", 0.0)
                    )

    rows: list[dict[str, Any]] = []
    for term in TERMS:
        total = totals[term]
        count = total["valid_count"]
        row: dict[str, Any] = {
            "model": args.model_label,
            "term": term,
            "scope": "held_out_manifest",
            "graphs": graph_count,
            "valid_residual_count": int(count),
            "normalized_abs_residual_mae": total["abs_sum"] / count if count else math.nan,
            "normalized_residual_rmse": math.sqrt(total["sq_sum"] / count) if count else math.nan,
            "normalized_abs_residual_max": total["abs_max"] if count else math.nan,
            "raw_huber_loss": total["huber_sum"] / count if count else math.nan,
        }
        for display, label in THRESHOLDS:
            row[f"satisfaction_rate_abs_le_{display}"] = total[f"le_{label}"] / count if count else math.nan
        rows.append(row)

    config = getattr(experiment.train, "node_balance_pi", None)
    payload = {
        "checkpoint": str(checkpoint_path),
        "y_edge_scaler": str(y_edge_scaler_path),
        "config": str(config_path),
        "manifest": str(manifest_path),
        "scope": "held_out_manifest",
        "definition": {
            "residual": "absolute normalized relative node-balance residual used by the configured Mass/Component/Atom equations",
            "mae": "mean absolute normalized residual over all valid residual entries",
            "rmse": "root mean squared normalized residual over all valid residual entries",
            "satisfaction_rates": "fraction of valid residual entries within the named absolute normalized-residual threshold",
        },
        "node_balance_config": {
            "mass": getattr(config, "mass", None),
            "component": getattr(config, "component", None),
            "atom": getattr(config, "atom", None),
        },
        "results": rows,
    }
    (output_dir / "physics_conservation_metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    with (output_dir / "physics_conservation_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["term"])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            "[physics-conservation] "
            f"{row['term']} mae={row['normalized_abs_residual_mae']:.6e} "
            f"rmse={row['normalized_residual_rmse']:.6e} "
            f"max={row['normalized_abs_residual_max']:.6e} "
            f"within_5pct={row['satisfaction_rate_abs_le_0.05']:.2%} "
            f"n={row['valid_residual_count']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
