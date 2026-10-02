#!/usr/bin/env python3
"""Smoke test for the single vs grouped-property edge_all decoder heads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from process_graph.experiment.config_builders import model_yaml_to_encoder_config  # noqa: E402
from process_graph.experiment.loaders import load_experiment_config  # noqa: E402
from process_graph.models import ProcessSurrogateModel  # noqa: E402


def _grad_norm(module: torch.nn.Module | None) -> float:
    if module is None:
        return 0.0
    total = 0.0
    for param in module.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().pow(2).sum().cpu())
    return total ** 0.5


def _dummy_batch(num_nodes: int = 6, *, edge_oper_dim: int = 5, edge_struct_dim: int = 5) -> dict[str, torch.Tensor]:
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 0, 3],
            [1, 2, 5, 4, 5, 2, 5],
        ],
        dtype=torch.long,
    )
    num_edges = int(edge_index.size(1))
    return {
        "edge_index": edge_index,
        "x_role": torch.tensor([0, 1, 1, 0, 1, 2], dtype=torch.long),
        "x_unit": torch.tensor([0, 2, 2, 0, 2, 0], dtype=torch.long),
        "x_hx_role": torch.zeros(num_nodes, dtype=torch.long),
        "x_oper": torch.randn(num_nodes, 13),
        "x_oper_mask": torch.ones(num_nodes, 13),
        "edge_stream_role": torch.zeros(num_edges, dtype=torch.long),
        "edge_stream_id": torch.arange(num_edges, dtype=torch.long),
        "edge_oper": torch.randn(num_edges, edge_oper_dim),
        "edge_oper_mask": torch.ones(num_edges, edge_oper_dim),
        "edge_struct_attr": torch.randn(num_edges, edge_struct_dim),
        "batch": torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long),
    }


def _build_model(config_path: Path, *, edge_head_type: str) -> ProcessSurrogateModel:
    experiment = load_experiment_config(config_path)
    experiment.model.edge_head_type = edge_head_type
    experiment.model.use_edge_decoder = True
    experiment.model.use_edge_stream_head = False
    experiment.model.stream_target_dim = 12
    cfg = model_yaml_to_encoder_config(experiment.model)
    return ProcessSurrogateModel(encoder_config=cfg, task_specs=[])


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test grouped-property edge head.")
    parser.add_argument(
        "--config",
        default="configs/experiment/process_surrogate_edge_all_v3_grouped_head.yaml",
        help="Experiment config to load before forcing single/grouped head variants.",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    config_path = (PROJECT_ROOT / args.config).resolve()
    experiment = load_experiment_config(config_path)
    batch = _dummy_batch(
        edge_oper_dim=int(getattr(experiment.model, "edge_oper_dim", 5)),
        edge_struct_dim=int(getattr(experiment.model, "edge_struct_dim", 5)),
    )

    single = _build_model(config_path, edge_head_type="single")
    grouped = _build_model(config_path, edge_head_type="grouped_property")
    single.train()
    grouped.train()

    single_out = single(batch)["y_edge_pred"]
    grouped_out = grouped(batch)["y_edge_pred"]
    if single_out.shape != grouped_out.shape:
        raise AssertionError(f"single/grouped output shapes differ: {single_out.shape} vs {grouped_out.shape}")
    if grouped_out.shape[-1] != 12:
        raise AssertionError(f"grouped output dim must be 12, got {grouped_out.shape[-1]}")

    target = torch.randn_like(grouped_out)
    loss = torch.nn.functional.mse_loss(grouped_out, target)
    loss.backward()

    decoder = grouped.edge_decoder
    grad_norms = {
        "shared_proj": _grad_norm(getattr(decoder, "shared_proj", None)),
        "frac_head": _grad_norm(getattr(decoder, "frac_head", None)),
        "flow_head": _grad_norm(getattr(decoder, "flow_head", None)),
        "cond_head": _grad_norm(getattr(decoder, "cond_head", None)),
    }
    missing = [name for name, value in grad_norms.items() if value <= 0.0]
    if missing:
        raise AssertionError(f"Grouped head branches missing gradients: {missing}")

    print(f"single_output_shape={tuple(single_out.shape)}")
    print(f"grouped_output_shape={tuple(grouped_out.shape)}")
    print(f"dummy_mse_loss={float(loss.detach().cpu()):.6f}")
    for name, value in grad_norms.items():
        print(f"grad_norm_{name}={value:.6f}")
    if decoder is not None and hasattr(decoder, "describe_head"):
        print(f"head_summary={decoder.describe_head()}")


if __name__ == "__main__":
    main()
