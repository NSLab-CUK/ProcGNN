from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from process_graph.data.tabular_dataset import (
    ProcessGraphTabularDataset,
    collate_graph_batch,
    compute_y_edge_scaler,
)
from process_graph.experiment.config_builders import build_task_specs, model_yaml_to_encoder_config
from process_graph.experiment.loaders import load_experiment_config
from process_graph.experiment.train_utils import compute_training_loss
from process_graph.models.process_surrogate import ProcessSurrogateModel


def _train_split_filter(train_csv: Path, split_column: str) -> str | None:
    import pandas as pd

    if not train_csv.is_file():
        return None
    peek = pd.read_csv(train_csv, nrows=1)
    if split_column in peek.columns:
        return "train"
    return None


def test_v3_collate_shapes_and_encoder_smoke() -> None:
    exp_path = PROJECT_ROOT / "configs" / "experiment" / "process_surrogate_base.yaml"
    experiment = load_experiment_config(exp_path)
    assert getattr(experiment.data, "use_canonical_graph_spec_v3", False)
    assert experiment.data.topology_mode == "stream_edge"

    train_csv = (experiment.project_root / experiment.data.train_data_path).resolve()
    split_filter = _train_split_filter(train_csv, experiment.data.split_column)
    stats_cfg = replace(experiment.data, normalize_x_oper=False, normalize_targets=False)
    ds = ProcessGraphTabularDataset(
        experiment.data.train_data_path,
        stats_cfg,
        experiment.project_root,
        split_filter=split_filter,
    )
    assert len(ds) > 0

    indices = list(range(min(3, len(ds))))
    records = [ds[i] for i in indices]
    batch = collate_graph_batch(records)

    mk = batch.model_kwargs
    num_edges = mk["edge_index"].size(1)

    assert mk["edge_oper"].shape == mk["edge_struct_attr"].shape
    assert mk["edge_oper"].shape == (num_edges, 5)
    assert mk["y_edge_true"].shape == (num_edges, 12)
    assert mk["y_edge_mask"].shape == (num_edges,)
    assert mk["edge_feature_mask"].shape == (num_edges,)
    assert mk["edge_batch"].shape == (num_edges,)
    assert float(mk["y_edge_mask"].sum().item()) <= float(num_edges)

    assert "edge_stream" in batch.targets
    assert batch.targets["edge_stream"].shape == mk["y_edge_true"].shape
    assert "edge_stream" in batch.task_inputs
    assert batch.task_inputs["edge_stream"]["edge_batch"].shape == mk["edge_batch"].shape
    assert batch.edge_target_columns == list(records[0].graph.edge_target_columns)

    model_cfg = replace(experiment.model, use_edge_stream_head=True)
    train_cfg = replace(experiment.train, edge_stream_loss_weight=0.5)
    encoder_cfg = model_yaml_to_encoder_config(model_cfg)
    task_specs = build_task_specs(model_cfg, experiment.data)
    model = ProcessSurrogateModel(encoder_config=encoder_cfg, task_specs=task_specs).train()
    batch_data = {k: v for k, v in mk.items()}
    task_inputs = {k: dict(v) for k, v in batch.task_inputs.items()}
    preds = model(batch_data, task_inputs=task_inputs)
    assert "edge_stream" in preds
    assert preds["edge_stream"].shape == batch.targets["edge_stream"].shape
    loss_items = compute_training_loss(
        preds,
        batch.targets,
        batch.target_masks,
        train_cfg,
        experiment.data,
    )
    assert "loss_edge_stream" in loss_items
    assert not torch.isnan(loss_items["loss_total"])
    loss_items["loss_total"].backward()
    with torch.no_grad():
        enc = model.encode(batch_data)
    assert enc.edge_embeddings is not None
    assert enc.edge_embeddings.shape[0] == num_edges
    assert enc.node_embeddings.shape[0] == mk["x_oper"].shape[0]


def test_answer_edge_pos_batched_offsets_match_collate() -> None:
    exp_path = PROJECT_ROOT / "configs" / "experiment" / "process_surrogate_base.yaml"
    experiment = load_experiment_config(exp_path)
    train_csv = (experiment.project_root / experiment.data.train_data_path).resolve()
    split_filter = _train_split_filter(train_csv, experiment.data.split_column)
    stats_cfg = replace(experiment.data, normalize_x_oper=False, normalize_targets=False)
    ds = ProcessGraphTabularDataset(
        experiment.data.train_data_path,
        stats_cfg,
        experiment.project_root,
        split_filter=split_filter,
    )
    records = [ds[i] for i in range(min(2, len(ds)))]
    batch = collate_graph_batch(records)
    assert batch.answer_edge_pos is not None
    edge_off = 0
    for i, rec in enumerate(records):
        ne = len(rec.graph.edge_index[0])
        slot_map = rec.slot_edge_index or {}
        for slot, local_i in slot_map.items():
            assert batch.answer_edge_pos[i][str(slot)] == edge_off + int(local_i)
        edge_off += ne


def test_edge_all_forward_loss_smoke() -> None:
    exp_path = PROJECT_ROOT / "configs" / "experiment" / "process_surrogate_edge_all_v3.yaml"
    experiment = load_experiment_config(exp_path)
    train_csv = (experiment.project_root / experiment.data.train_data_path).resolve()
    split_filter = _train_split_filter(train_csv, experiment.data.split_column)
    stats_cfg = replace(experiment.data, normalize_x_oper=False, normalize_targets=False)
    base_ds = ProcessGraphTabularDataset(
        experiment.data.train_data_path,
        stats_cfg,
        experiment.project_root,
        split_filter=split_filter,
    )
    stats_records = [base_ds[i] for i in range(min(8, len(base_ds)))]
    encoder_cfg = model_yaml_to_encoder_config(experiment.model)
    y_fit = compute_y_edge_scaler(stats_records, stream_target_dim=int(encoder_cfg.stream_target_dim))
    y_m, y_s = y_fit.mean, y_fit.std

    ds = ProcessGraphTabularDataset(
        experiment.data.train_data_path,
        replace(experiment.data, normalize_x_oper=False, normalize_targets=False),
        experiment.project_root,
        split_filter=split_filter,
    )
    batch = collate_graph_batch([ds[0]])
    device = torch.device("cpu")
    model = ProcessSurrogateModel(
        encoder_config=encoder_cfg, task_specs=build_task_specs(experiment.model, experiment.data)
    ).to(device)
    batch_data = {k: v.to(device) for k, v in batch.model_kwargs.items()}
    targets = {k: v.to(device) for k, v in batch.targets.items()}
    target_masks = {k: v.to(device) for k, v in batch.target_masks.items()}
    task_inputs = {h: {k: v.to(device) for k, v in p.items()} for h, p in batch.task_inputs.items()}
    y_raw = targets["edge_stream"]
    y_m_d = y_m.to(device).view(1, -1)
    y_s_d = y_s.to(device).view(1, -1)
    y_n = (y_raw - y_m_d) / y_s_d
    preds = model(batch_data, task_inputs=task_inputs)
    assert "y_edge_pred" in preds
    loss_items = compute_training_loss(
        preds,
        {**targets, "edge_stream": y_n},
        target_masks,
        experiment.train,
        experiment.data,
    )
    assert "loss_edge" in loss_items
    loss_items["loss_total"].backward()


def test_edge_stream_mask_broadcast_matches_train_eval() -> None:
    """Regression: [E] mask with [E, D] preds must use expand_as before boolean indexing (val loop)."""
    p_es = torch.randn(20, 12)
    t_es = torch.randn(20, 12)
    em = torch.cat([torch.ones(17), torch.zeros(3)])
    m = em.to(dtype=p_es.dtype)
    while m.ndim < p_es.ndim:
        m = m.unsqueeze(-1)
    m = m.expand_as(p_es)
    valid = m > 0
    diff = (p_es - t_es)[valid]
    assert diff.numel() == 17 * 12
