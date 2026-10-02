from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for entry in (SRC_ROOT, PROJECT_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from process_graph.data.tabular_dataset import ProcessGraphTabularDataset  # noqa: E402
from process_graph.experiment.config_builders import (  # noqa: E402
    build_task_specs,
    model_yaml_to_encoder_config,
)
from process_graph.experiment.edge_step_training import (  # noqa: E402
    build_target_edge_boolean_mask,
)
from process_graph.experiment.loaders import load_experiment_config  # noqa: E402
from process_graph.known_feed import operating_feature_names  # noqa: E402
from process_graph.models import ProcessSurrogateModel  # noqa: E402
from scripts.train_process_surrogate import _collate_for_experiment  # noqa: E402


DEFAULT_PROPERTIES = (
    "Temp", "Pres", "Frac_H2O", "Frac_H2", "Frac_CH4",
    "Frac_CO2", "Frac_CO", "Frac_O2", "Frac_N2", "Mass_Flow",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _process_number(value: Any) -> int:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse process number from {value!r}")
    return int(digits)


def _checkpoint_model_state(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state = payload.get("model_state_dict") or payload.get("model")
    if not isinstance(state, Mapping):
        raise RuntimeError("Checkpoint has no model_state_dict/model mapping.")
    return state


def _raw_vector(
    batch_data: Mapping[str, torch.Tensor],
    *,
    oper_mean: torch.Tensor,
    oper_std: torch.Tensor,
) -> torch.Tensor:
    x_normalized = batch_data["x_oper"].detach().cpu()
    raw_x = x_normalized * oper_std.reshape(1, -1) + oper_mean.reshape(1, -1)
    if "graph_feed_values" in batch_data:
        feed_dim = int(batch_data["graph_feed_values"].shape[-1])
        feed_normalized = batch_data["graph_feed_values"].detach().cpu().reshape(-1)
        start = 13
        raw_feed = (
            feed_normalized * oper_std[start : start + feed_dim]
            + oper_mean[start : start + feed_dim]
        )
    else:
        raw_feed = raw_x.new_empty((0,))
    return torch.cat([raw_x.reshape(-1), raw_feed], dim=0)


class RawTargetPrediction(nn.Module):
    """Differentiable raw-feature wrapper for one graph topology and one target."""

    def __init__(
        self,
        *,
        model: nn.Module,
        static_batch: Mapping[str, torch.Tensor],
        task_inputs: Mapping[str, Mapping[str, torch.Tensor]],
        node_count: int,
        oper_dim: int,
        target_edge_index: int,
        property_index: int,
        oper_mean: torch.Tensor,
        oper_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.model = model
        self.static_batch = dict(static_batch)
        self.task_inputs = {name: dict(payload) for name, payload in task_inputs.items()}
        self.node_count = int(node_count)
        self.oper_dim = int(oper_dim)
        self.target_edge_index = int(target_edge_index)
        self.property_index = int(property_index)
        self.register_buffer("oper_mean", oper_mean.reshape(-1))
        self.register_buffer("oper_std", oper_std.reshape(-1))

    def forward(self, raw_vector: torch.Tensor) -> torch.Tensor:
        if raw_vector.ndim == 1:
            raw_vector = raw_vector.unsqueeze(0)
        outputs: list[torch.Tensor] = []
        x_width = self.node_count * self.oper_dim
        for row in raw_vector:
            raw_x = row[:x_width].reshape(self.node_count, self.oper_dim)
            x_oper = (raw_x - self.oper_mean) / self.oper_std
            x_mask = self.static_batch["x_oper_mask"]
            if self.oper_dim > 13:
                x_oper = torch.cat(
                    [
                        x_oper[:, :13],
                        torch.where(
                            x_mask[:, 13:] > 0,
                            x_oper[:, 13:],
                            torch.zeros_like(x_oper[:, 13:]),
                        ),
                    ],
                    dim=-1,
                )
            model_batch = dict(self.static_batch)
            model_batch["x_oper"] = x_oper
            feed_width = int(row.numel() - x_width)
            if feed_width:
                raw_feed = row[x_width:]
                feed = (raw_feed - self.oper_mean[13 : 13 + feed_width]) / self.oper_std[
                    13 : 13 + feed_width
                ]
                feed_mask = self.static_batch["graph_feed_mask"].reshape(-1)
                feed = torch.where(feed_mask > 0, feed, torch.zeros_like(feed))
                model_batch["graph_feed_values"] = feed.reshape(1, -1)
                model_batch["graph_feed_input"] = torch.cat(
                    [feed.reshape(1, -1), self.static_batch["graph_feed_mask"]], dim=-1
                )
            prediction = self.model(model_batch, task_inputs=self.task_inputs)["main_stream_pred"]
            outputs.append(prediction[self.target_edge_index, self.property_index])
        return torch.stack(outputs).reshape(-1, 1)


def _as_shap_vector(values: Any) -> np.ndarray:
    array = np.asarray(values)
    while array.ndim > 2 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise RuntimeError(f"Unexpected SHAP value shape: {array.shape}")
    return array


def _gradient_shap_values(
    explainer: Any,
    explained: torch.Tensor,
    *,
    gradient_samples: int | None,
) -> Any:
    """Run SHAP gradients without cuDNN's eval-mode RNN backward restriction.

    The Proposed encoder contains the Set2Set LSTM.  SHAP needs input
    gradients while the model remains in evaluation mode so dropout stays
    deterministic.  cuDNN does not support that RNN backward combination,
    whereas PyTorch's native CUDA RNN implementation does.
    """
    if explained.device.type == "cuda" and torch.backends.cudnn.is_available():
        with torch.backends.cudnn.flags(enabled=False):
            return explainer.shap_values(
                explained,
                **({} if gradient_samples is None else {"nsamples": gradient_samples}),
            )
    return explainer.shap_values(
        explained,
        **({} if gradient_samples is None else {"nsamples": gradient_samples}),
    )


def _load_coordinates(path: Path, process_id: int, node_names: Sequence[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Explicit flowsheet coordinates are required; missing {path}. "
            "Automatic OCR/coordinate guessing is intentionally disabled."
        )
    frame = pd.read_csv(path)
    required = {"process_id", "node_name", "x", "y"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Coordinate file is missing columns: {missing}")
    selected = frame.loc[frame["process_id"].map(_process_number).eq(process_id)].copy()
    missing_nodes = sorted(set(node_names).difference(selected["node_name"].astype(str)))
    if missing_nodes:
        raise ValueError(f"P{process_id:02d} coordinates missing nodes: {missing_nodes}")
    return selected


def _plot_flowsheet(
    *,
    image_path: Path,
    coordinates: pd.DataFrame,
    node_importance: pd.DataFrame,
    output_path: Path,
) -> None:
    image = Image.open(image_path)
    merged = coordinates.merge(node_importance, on="node_name", how="left").fillna({"importance": 0.0})
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    values = merged["importance"].to_numpy(dtype=float)
    size = 80.0 + 500.0 * values / max(float(values.max()), 1.0e-12)
    scatter = ax.scatter(
        merged["x"], merged["y"], c=values, s=size, cmap="magma", alpha=0.8,
        edgecolors="white", linewidths=0.7,
    )
    for row in merged.itertuples(index=False):
        ax.text(
            float(row.x) + 5.0,
            float(row.y) - 5.0,
            str(row.node_name),
            fontsize=7,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1},
        )
    fig.colorbar(scatter, ax=ax, label="mean |SHAP|")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final Proposed target-only SHAP analysis")
    parser.add_argument("--config", default="configs/experiment/pinn/model_260805_10d_frac1.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--process-id", type=int, required=True)
    parser.add_argument("--output-root", default="outputs/final_paper/explainability_shap")
    parser.add_argument("--target-edge-ids", nargs="*", default=[])
    parser.add_argument("--target-properties", nargs="+", default=list(DEFAULT_PROPERTIES))
    parser.add_argument("--background-size", type=int, default=32)
    parser.add_argument("--explain-samples", type=int, default=16)
    parser.add_argument(
        "--gradient-samples",
        type=int,
        default=None,
        help=(
            "Number of Monte-Carlo samples used by SHAP GradientExplainer. "
            "Omit to retain SHAP's default; set explicitly for reproducible runtime."
        ),
    )
    parser.add_argument("--seed", type=int, default=260716)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--coordinate-file", default="data/process_overall_img/unit_coordinates.csv"
    )
    parser.add_argument("--require-flowsheet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gradient_samples is not None and args.gradient_samples <= 0:
        raise ValueError("--gradient-samples must be a positive integer when supplied.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    output_root = _resolve(args.output_root) / f"P{args.process_id:02d}"
    output_root.mkdir(parents=True, exist_ok=True)

    experiment = load_experiment_config(_resolve(args.config))
    checkpoint = torch.load(_resolve(args.checkpoint), map_location="cpu", weights_only=False)
    normalizer = checkpoint.get("oper_normalizer")
    if not isinstance(normalizer, Mapping):
        raise RuntimeError("Checkpoint lacks the train-only operating normalizer.")
    oper_mean = torch.as_tensor(normalizer["mean"], dtype=torch.float32)
    oper_std = torch.as_tensor(normalizer["std"], dtype=torch.float32)
    oper_std = torch.where(oper_std == 0, torch.ones_like(oper_std), oper_std)

    manifest = pd.read_csv(_resolve(args.manifest))
    process_rows = manifest.loc[manifest["process_id"].map(_process_number).eq(args.process_id)].copy()
    if process_rows.empty:
        raise RuntimeError(f"Manifest contains no Process{args.process_id} samples.")
    process_manifest = output_root / "evaluation_manifest.csv"
    process_rows.to_csv(process_manifest, index=False)
    dataset = ProcessGraphTabularDataset(
        experiment.data.test_data_path,
        experiment.data,
        experiment.project_root,
        split_filter=None,
        split_manifest=process_manifest,
        oper_mean=oper_mean,
        oper_std=oper_std,
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=_collate_for_experiment(experiment.data),
    )
    batches = []
    vectors = []
    node_names: list[str] | None = None
    edge_ids: list[str] | None = None
    for batch_index, batch in enumerate(loader):
        batch_data = {key: value.to(device) for key, value in batch.model_kwargs.items()}
        if node_names is None:
            record = dataset[batch_index].graph
            node_names = list(record.node_names)
            edge_ids = list(record.canonical_edge_ids)
        vector = _raw_vector(batch_data, oper_mean=oper_mean, oper_std=oper_std)
        vectors.append(vector)
        batches.append(batch)
        if len(batches) >= max(args.background_size, args.explain_samples):
            break
    if not batches or node_names is None or edge_ids is None:
        raise RuntimeError("No explanation samples were loaded.")
    if any(vector.shape != vectors[0].shape for vector in vectors):
        raise RuntimeError("SHAP samples do not share one fixed Process topology.")

    target_edge_ids = args.target_edge_ids or [
        edge_id for edge_id, flag in zip(edge_ids, batches[0].model_kwargs["edge_is_target"].tolist())
        if float(flag) > 0.5
    ]
    unknown_edges = sorted(set(target_edge_ids).difference(edge_ids))
    unknown_properties = sorted(set(args.target_properties).difference(DEFAULT_PROPERTIES))
    if unknown_edges or unknown_properties:
        raise ValueError(f"Unknown target edges={unknown_edges} properties={unknown_properties}")
    plan = {
        "method": "shap.GradientExplainer",
        "reason": "PyTorch model is differentiable; raw-feature wrapper preserves the trained normalization path.",
        "process_id": args.process_id,
        "background_size": min(args.background_size, len(vectors)),
        "explain_samples": min(args.explain_samples, len(vectors)),
        "gradient_samples": args.gradient_samples if args.gradient_samples is not None else "shap_default",
        "target_edge_ids": target_edge_ids,
        "target_properties": args.target_properties,
        "stream_attribution": "not_available_no_sample_varying_raw_stream_input",
        "coordinate_file": str(_resolve(args.coordinate_file)),
    }
    (output_root / "shap_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    model = ProcessSurrogateModel(
        encoder_config=model_yaml_to_encoder_config(experiment.model, experiment.data),
        task_specs=build_task_specs(experiment.model, experiment.data),
    ).to(device)
    model.load_state_dict(_checkpoint_model_state(checkpoint), strict=True)
    model.eval()
    background = torch.stack(vectors[: args.background_size]).to(device)
    explained = torch.stack(vectors[: args.explain_samples]).to(device)
    feature_names = list(operating_feature_names(experiment.data.known_feed_condition))
    rows: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    first_batch = batches[0]
    static_batch = {key: value.to(device) for key, value in first_batch.model_kwargs.items()}
    task_inputs = {
        head: {key: value.to(device) for key, value in payload.items()}
        for head, payload in first_batch.task_inputs.items()
    }
    target_mask, _ = build_target_edge_boolean_mask(
        train_cfg=experiment.train,
        edge_export_meta=first_batch.edge_export_meta,
        edge_target_columns=list(first_batch.edge_target_columns),
        n_edges=len(edge_ids),
        device=device,
    )
    static_batch["target_edge_mask"] = target_mask
    x_width = len(node_names) * len(feature_names)

    for edge_id in target_edge_ids:
        edge_index = edge_ids.index(edge_id)
        for property_name in args.target_properties:
            property_index = DEFAULT_PROPERTIES.index(property_name)
            wrapper = RawTargetPrediction(
                model=model,
                static_batch=static_batch,
                task_inputs=task_inputs,
                node_count=len(node_names),
                oper_dim=len(feature_names),
                target_edge_index=edge_index,
                property_index=property_index,
                oper_mean=oper_mean.to(device),
                oper_std=oper_std.to(device),
            ).to(device)
            explainer = shap.GradientExplainer(wrapper, background)
            shap_values = _as_shap_vector(
                _gradient_shap_values(
                    explainer,
                    explained,
                    gradient_samples=args.gradient_samples,
                )
            )
            node_values = np.abs(shap_values[:, :x_width]).reshape(
                shap_values.shape[0], len(node_names), len(feature_names)
            )
            for sample_index in range(shap_values.shape[0]):
                for node_index, node_name in enumerate(node_names):
                    for feature_index, feature_name in enumerate(feature_names):
                        rows.append({
                            "process_id": args.process_id,
                            "sample_index": sample_index,
                            "target_edge_id": edge_id,
                            "target_property": property_name,
                            "source": "node_operating",
                            "node_name": node_name,
                            "feature_name": feature_name,
                            "raw_value": float(explained[sample_index, node_index * len(feature_names) + feature_index]),
                            "shap_value": float(shap_values[sample_index, node_index * len(feature_names) + feature_index]),
                        })
                for feed_index, feed_name in enumerate(("feed_ch4_flow", "feed_air_flow", "feed_water_flow")):
                    if x_width + feed_index < shap_values.shape[1]:
                        rows.append({
                            "process_id": args.process_id,
                            "sample_index": sample_index,
                            "target_edge_id": edge_id,
                            "target_property": property_name,
                            "source": "direct_feed_head",
                            "node_name": "V_INPUT",
                            "feature_name": feed_name,
                            "raw_value": float(explained[sample_index, x_width + feed_index]),
                            "shap_value": float(shap_values[sample_index, x_width + feed_index]),
                        })
            importance = node_values.mean(axis=(0, 2))
            for node_name, value in zip(node_names, importance):
                node_rows.append({
                    "process_id": args.process_id,
                    "target_edge_id": edge_id,
                    "target_property": property_name,
                    "node_name": node_name,
                    "importance": float(value),
                })
            top = sorted(zip(node_names, importance), key=lambda item: item[1], reverse=True)[:10]
            for rank, (node_name, value) in enumerate(top, start=1):
                summary_rows.append({
                    "process_id": args.process_id,
                    "target_edge_id": edge_id,
                    "target_property": property_name,
                    "rank": rank,
                    "node_name": node_name,
                    "importance": float(value),
                })

    sample_frame = pd.DataFrame(rows)
    node_frame = pd.DataFrame(node_rows)
    summary_frame = pd.DataFrame(summary_rows)
    sample_frame.to_csv(output_root / "sample_level_shap.csv", index=False)
    node_frame.to_csv(output_root / "node_importance.csv", index=False)
    summary_frame.to_csv(output_root / "topk_nodes.csv", index=False)
    (
        sample_frame.assign(abs_shap=lambda frame: frame["shap_value"].abs())
        .groupby(["target_edge_id", "target_property", "source", "node_name", "feature_name"], as_index=False)
        ["abs_shap"].mean()
        .sort_values("abs_shap", ascending=False)
        .to_csv(output_root / "global_feature_importance.csv", index=False)
    )
    pd.DataFrame(
        [{"status": "not_available", "reason": "no sample-varying raw stream input is attributable"}]
    ).to_csv(output_root / "stream_importance_status.csv", index=False)

    coordinate_path = _resolve(args.coordinate_file)
    try:
        coordinates = _load_coordinates(coordinate_path, args.process_id, node_names)
        image_path = _resolve(f"data/process_overall_img/{args.process_id}번.png")
        for (edge_id, property_name), frame in node_frame.groupby(["target_edge_id", "target_property"]):
            _plot_flowsheet(
                image_path=image_path,
                coordinates=coordinates,
                node_importance=frame,
                output_path=output_root / f"flowsheet_{edge_id}_{property_name}.png",
            )
        flowsheet_status = {"status": "completed", "coordinate_file": str(coordinate_path)}
    except (FileNotFoundError, ValueError) as exc:
        flowsheet_status = {"status": "blocked", "reason": str(exc)}
        if args.require_flowsheet:
            raise
    (output_root / "flowsheet_status.json").write_text(
        json.dumps(flowsheet_status, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({**plan, "flowsheet": flowsheet_status}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
