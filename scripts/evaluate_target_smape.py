"""Post-hoc target-edge sMAPE evaluation for a saved proposed-model checkpoint.

The evaluator deliberately performs forward inference only.  It reconstructs
the split-specific normalizers stored alongside ``best.pt``, applies the exact
physical output decoding used by PI validation, and intersects label masks with
the paper's predictable target-edge mask.  No optimizer, scheduler, or model
state is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for entry in (PROJECT_ROOT / "src", PROJECT_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from process_graph.data.tabular_dataset import ProcessGraphTabularDataset  # noqa: E402
from process_graph.experiment.config_builders import build_task_specs, model_yaml_to_encoder_config  # noqa: E402
from process_graph.experiment.edge_step_pi_training import (  # noqa: E402
    collect_pi_all_edge_property_metrics_from_loader,
)
from process_graph.experiment.loaders import load_experiment_config  # noqa: E402
from process_graph.models import ProcessSurrogateModel  # noqa: E402
from scripts.train_process_surrogate import _apply_runtime_overrides, _collate_for_experiment  # noqa: E402


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
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(_resolve(explicit_path))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate target-edge, physical-scale sMAPE (%) without retraining."
    )
    parser.add_argument("--config", default="configs/experiment/pinn/model_260805_10d_frac1.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--runtime-overrides-file",
        default=None,
        help="Original run's runtime_overrides.json; restores its data/model overrides before evaluation.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-label", default="Proposed")
    parser.add_argument("--y-edge-scaler", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument("--resume-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve(args.config)
    checkpoint_path = _resolve(args.checkpoint)
    manifest_path = _resolve(args.manifest)
    output_dir = _resolve(args.output_dir)
    metrics_path = output_dir / "target_property_smape.csv"
    if args.resume_existing and metrics_path.is_file() and metrics_path.stat().st_size > 0:
        print(f"[SKIP] existing {metrics_path}", flush=True)
        return
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment = load_experiment_config(config_path)
    if args.runtime_overrides_file:
        overrides_path = _resolve(args.runtime_overrides_file)
        if not overrides_path.is_file():
            raise FileNotFoundError(f"Runtime-overrides file not found: {overrides_path}")
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
        if not isinstance(overrides, Mapping):
            raise TypeError(f"Runtime overrides must be a mapping: {overrides_path}")
        _apply_runtime_overrides(experiment, overrides)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint payload must be a mapping: {checkpoint_path}")
    oper_normalizer = checkpoint.get("oper_normalizer")
    if not isinstance(oper_normalizer, Mapping):
        raise RuntimeError("Checkpoint lacks the required train-split operating normalizer.")
    oper_mean = torch.as_tensor(oper_normalizer["mean"], dtype=torch.float32)
    oper_std = torch.as_tensor(oper_normalizer["std"], dtype=torch.float32)
    oper_std = torch.where(oper_std == 0, torch.ones_like(oper_std), oper_std)
    scaler_path, normalizer = _resolve_y_edge_scaler(checkpoint_path, args.y_edge_scaler)

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
    device = torch.device(args.device)
    model = ProcessSurrogateModel(
        encoder_config=model_yaml_to_encoder_config(experiment.model, experiment.data),
        task_specs=build_task_specs(experiment.model, experiment.data),
    ).to(device)
    model.load_state_dict(_checkpoint_state(checkpoint), strict=True)

    acc = collect_pi_all_edge_property_metrics_from_loader(
        model=model,
        loader=loader,
        device=device,
        train_cfg=experiment.train,
        data_cfg=experiment.data,
        use_amp=False,
        split_name="test",
        normalizer=normalizer,
        store_scatter_samples=False,
        progress_label=str(args.model_label),
        progress_interval=int(args.progress_interval),
        target_edge_only=True,
    )
    rows = acc.rows()
    for row in rows:
        row.update(
            {
                "model": str(args.model_label),
                "split": "test",
                "metric_scope": "target_predictable_edges_by_property",
                "smape_definition": "100 * mean(2*abs(y_pred-y_true)/(abs(y_pred)+abs(y_true)+1e-8))",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(metrics_path, index=False)
    frame.to_csv(output_dir / "target_property_metrics.csv", index=False)
    finite = frame.loc[pd.to_numeric(frame["sMAPE_pct"], errors="coerce").map(math.isfinite)]
    summary = {
        "model": str(args.model_label),
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "manifest": str(manifest_path),
        "y_edge_scaler": str(scaler_path),
        "metric_scope": "target_predictable_edges_by_property",
        "smape_unit": "percent",
        "smape_formula": "100 * mean(2*abs(y_pred-y_true)/(abs(y_pred)+abs(y_true)+1e-8))",
        "properties": int(len(frame)),
        "macro_mean_smape_pct": float(finite["sMAPE_pct"].mean()) if not finite.empty else math.nan,
    }
    (output_dir / "target_property_smape.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )
    display = " ".join(
        f"{row.property_name}={float(row.sMAPE_pct):.4f}%"
        for row in frame.itertuples(index=False)
        if math.isfinite(float(row.sMAPE_pct))
    )
    print(f"[sMAPE][target/by-property] {display}", flush=True)
    print(f"[sMAPE][target/macro] {summary['macro_mean_smape_pct']:.4f}%", flush=True)


if __name__ == "__main__":
    main()
