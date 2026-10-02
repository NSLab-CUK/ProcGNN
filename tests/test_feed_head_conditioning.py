from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
from pathlib import Path

import torch
import pytest
import yaml

from process_graph.constants import OPER_FEATURE_SLOTS
from process_graph.data.tabular_dataset import (
    GraphSampleRecord,
    ProcessGraphTabularDataset,
    collate_graph_batch,
)
from process_graph.experiment.config_builders import model_yaml_to_encoder_config
from process_graph.experiment.loaders import load_experiment_config
from process_graph.experiment.schema import DataConfig
from process_graph.feed_head import parse_feed_head_conditioning
from process_graph.known_feed import parse_known_feed_condition
from process_graph.models.process_surrogate import ProcessSurrogateModel
from process_graph.schema import GraphSample


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs/experiment/pinn/co_ratio_regime_260803"
R0 = CFG / "c1_f1_ratio.yaml"
FEED = CFG / "c1_f1_ratio_r0_feed_head.yaml"
FEED32 = CFG / "c1_f1_ratio_r0_feed_head_32d.yaml"
HX = CFG / "c1_f1_ratio_r0_hx_pair.yaml"
COMBINED = CFG / "c1_f1_ratio_r0_hx_pair_feed_head.yaml"
COMBINED32 = CFG / "c1_f1_ratio_r0_hx_pair_feed_head_32d.yaml"


def _model(path: Path) -> ProcessSurrogateModel:
    exp = load_experiment_config(path)
    return ProcessSurrogateModel(model_yaml_to_encoder_config(exp.model, exp.data), [])


def _sample(values: list[float], masks: list[int]) -> GraphSample:
    return GraphSample(
        process_id="P01",
        node_names=["A", "B"],
        edge_index=[[0], [1]],
        x_role=[0, 0],
        x_unit=[0, 0],
        x_hx_role=[0, 0],
        x_oper=[[0.0] * 18, [0.0] * 18],
        x_oper_mask=[[0] * 18, [0] * 18],
        targets={"target": {}},
        graph_feed_values=values,
        graph_feed_mask=masks,
    )


def _record(sample: GraphSample) -> GraphSampleRecord:
    return GraphSampleRecord(
        graph=sample,
        slot_targets={"target": 0.0},
        slot_masks={"target": 1.0},
        slot_node_index={"target": 1},
        category_node_indices={},
        category_values={},
        sample_meta={"sample_id": "x"},
    )


def test_r0_is_unchanged_and_feed_config_is_opt_in() -> None:
    r0 = _model(R0)
    feed = _model(FEED)
    feed32 = _model(FEED32)
    assert r0.edge_decoder is not None and feed.edge_decoder is not None
    assert r0.edge_decoder.feed_encoder is None
    assert not any(k.startswith("edge_decoder.feed_encoder.") for k in r0.state_dict())
    assert r0.edge_decoder.describe_head()["hierarchical_descriptor_dim"] == 1536
    assert feed.edge_decoder.describe_head()["hierarchical_descriptor_dim"] == 1600
    assert feed.edge_decoder.describe_head()["feed_encoder_parameters"] == 2528
    assert feed32.edge_decoder.describe_head()["hierarchical_descriptor_dim"] == 1568
    assert feed32.edge_decoder.describe_head()["feed_encoder_parameters"] == 752
    assert sum(p.numel() for p in feed.parameters()) - sum(p.numel() for p in r0.parameters()) == 51680
    assert (
        sum(p.numel() for p in feed32.parameters())
        - sum(p.numel() for p in r0.parameters())
        == 25328
    )
    clone = _model(R0)
    clone.load_state_dict(r0.state_dict(), strict=True)


def test_feed32_changes_only_feed_width_descriptor_contract_and_paths() -> None:
    raw64 = yaml.safe_load(FEED.read_text(encoding="utf-8"))
    raw32 = yaml.safe_load(FEED32.read_text(encoding="utf-8"))
    for raw in (raw64, raw32):
        for key in ("experiment_name", "output_dir", "save_dir", "log_dir"):
            raw.pop(key)
    feed64 = raw64["overrides"]["model"]["feed_head_conditioning"]["encoder"]
    feed32 = raw32["overrides"]["model"]["feed_head_conditioning"]["encoder"]
    assert feed64 == {"hidden_dim": 32, "output_dim": 64, "dropout": 0.0}
    assert feed32 == {"hidden_dim": 16, "output_dim": 32, "dropout": 0.0}
    feed32.update(feed64)
    raw32["overrides"]["model"]["dimension_design"]["decoder"].pop("input_dim")
    assert raw32 == raw64


def test_feed_scaling_mask_and_batch_collation() -> None:
    ds = object.__new__(ProcessGraphTabularDataset)
    ds.data_cfg = replace(DataConfig(), normalize_x_oper=True, normalize_targets=False)
    ds._known_feed_cfg = parse_known_feed_condition(
        {"enabled": True, "feed_names": ["CH4", "AIR", "WATER"], "include_mask": True}
    )
    ds._feed_head_cfg = parse_feed_head_conditioning({"enabled": True})
    ds.oper_mean = torch.zeros(18)
    ds.oper_std = torch.ones(18)
    start = len(OPER_FEATURE_SLOTS)
    ds.oper_mean[start:start + 3] = torch.tensor([10.0, 20.0, 30.0])
    ds.oper_std[start:start + 3] = torch.tensor([2.0, 4.0, 5.0])

    a = ds._apply_normalization(_sample([12.0, 28.0, 0.0], [1, 1, 0]))
    b = ds._apply_normalization(_sample([0.0, 20.0, 30.0], [1, 1, 1]))
    assert torch.allclose(torch.tensor(a.graph_feed_values), torch.tensor([1.0, 2.0, 0.0]))
    assert a.graph_feed_mask == [1, 1, 0]
    assert b.graph_feed_mask[0] == 1
    assert b.graph_feed_values[0] == -5.0

    batch = collate_graph_batch([_record(a), _record(b)], fixed_slots=("target",))
    assert batch.model_kwargs["graph_feed_values"].shape == (2, 3)
    assert batch.model_kwargs["graph_feed_mask"].shape == (2, 3)
    assert batch.model_kwargs["graph_feed_input"].shape == (2, 6)
    assert torch.equal(batch.model_kwargs["graph_feed_input"][:, 3:], torch.tensor([[1., 1., 0.], [1., 1., 1.]]))


@pytest.mark.parametrize(
    ("config_path", "feed_hidden_dim", "feed_output_dim", "descriptor_dim"),
    (
        (FEED, 32, 64, 1600),
        (FEED32, 16, 32, 1568),
    ),
)
def test_feed_broadcast_sensitivity_gradients_and_checkpoint_roundtrip(
    tmp_path: Path,
    config_path: Path,
    feed_hidden_dim: int,
    feed_output_dim: int,
    descriptor_dim: int,
) -> None:
    torch.manual_seed(7)
    model = _model(config_path)
    model.train()
    oper_dim = model.encoder.config.oper_dim
    data = {
        "edge_index": torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        "x_role": torch.zeros(4, dtype=torch.long),
        "x_unit": torch.zeros(4, dtype=torch.long),
        "x_hx_role": torch.zeros(4, dtype=torch.long),
        "x_oper": torch.randn(4, oper_dim),
        "x_oper_mask": torch.ones(4, oper_dim),
        "batch": torch.tensor([0, 0, 1, 1]),
        "edge_batch": torch.tensor([0, 0, 1, 1]),
        "edge_stream_role": torch.zeros(4, dtype=torch.long),
        "edge_property_stream_role": torch.zeros(4, dtype=torch.long),
        "edge_stream_id": torch.arange(4),
        "edge_oper": torch.randn(4, 3),
        "edge_struct_attr": torch.randn(4, 3),
        "edge_oper_mask": torch.ones(4, 3),
        "edge_is_predictable": torch.tensor([1., 0., 1., 1.]),
        "graph_feed_input": torch.tensor([[0., 1., 2., 1., 1., 1.], [3., 4., 5., 1., 1., 1.]]),
    }
    captured: list[torch.Tensor] = []
    assert model.edge_decoder is not None and model.edge_decoder.hierarchical_pi_head is not None
    hook = model.edge_decoder.hierarchical_pi_head.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach())
    )
    out = model(data)
    hook.remove()
    pred = out["main_stream_pred"]
    assert pred.shape == (4, 11)
    descriptor = captured[0]
    assert descriptor.shape == (4, descriptor_dim)
    first_linear = model.edge_decoder.feed_encoder[0]
    second_linear = model.edge_decoder.feed_encoder[3]
    assert tuple(first_linear.weight.shape) == (feed_hidden_dim, 6)
    assert tuple(second_linear.weight.shape) == (
        feed_output_dim,
        feed_hidden_dim,
    )
    with torch.no_grad():
        hidden = model.edge_decoder.feed_encoder(data["graph_feed_input"])
    assert hidden.shape == (2, feed_output_dim)
    assert torch.allclose(descriptor[0, -feed_output_dim:], hidden[0])
    assert torch.equal(
        descriptor[1, -feed_output_dim:], torch.zeros(feed_output_dim)
    )
    assert torch.allclose(
        descriptor[2:, -feed_output_dim:], hidden[1].expand(2, -1)
    )

    changed = {k: v.clone() for k, v in data.items()}
    changed["graph_feed_input"][0, 1] += 5.0
    changed_out = model(changed)["main_stream_pred"]
    assert not torch.allclose(pred[:2], changed_out[:2])

    with torch.no_grad():
        real_zero = model.edge_decoder.feed_encoder(
            torch.tensor([[0., 0., 0., 1., 1., 1.]])
        )
        missing_zero = model.edge_decoder.feed_encoder(
            torch.tensor([[0., 0., 0., 0., 1., 1.]])
        )
    assert not torch.allclose(real_zero, missing_zero)

    loss = pred.square().mean()
    loss.backward()
    required = {
        "feed": model.edge_decoder.feed_encoder,
        "shared": model.edge_decoder.hierarchical_pi_head.shared_input,
        "condition": model.edge_decoder.hierarchical_pi_head.condition_head,
        "fraction": model.edge_decoder.hierarchical_pi_head.fraction_head,
        "mass": model.edge_decoder.hierarchical_pi_head.mass_head,
        "node": model.encoder.input_encoder,
        "edge": model.encoder.edge_input_encoder,
    }
    for name, module in required.items():
        assert module is not None
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), name
        assert any(bool((g != 0).any()) for g in grads), name

    ckpt = tmp_path / "feed.pt"
    torch.save(model.state_dict(), ckpt)
    clone = _model(config_path)
    clone.load_state_dict(torch.load(ckpt, weights_only=True), strict=True)


def test_hx_pair_feed_head_combination_shape_isolation_and_gradients() -> None:
    raw_r0 = yaml.safe_load(R0.read_text(encoding="utf-8"))
    raw_hx = yaml.safe_load(HX.read_text(encoding="utf-8"))
    raw_feed = yaml.safe_load(FEED.read_text(encoding="utf-8"))
    raw_combined = yaml.safe_load(COMBINED.read_text(encoding="utf-8"))
    assert (
        raw_r0["overrides"]["train"]
        == raw_hx["overrides"]["train"]
        == raw_feed["overrides"]["train"]
        == raw_combined["overrides"]["train"]
    )
    assert (
        raw_r0["overrides"]["data"]
        == raw_hx["overrides"]["data"]
        == raw_feed["overrides"]["data"]
        == raw_combined["overrides"]["data"]
    )
    combined_model_config = deepcopy(raw_combined["overrides"]["model"])
    assert combined_model_config.pop("hx_pair_relation")["enabled"] is True
    assert combined_model_config.pop("feed_head_conditioning")["enabled"] is True
    base_model_config = deepcopy(raw_r0["overrides"]["model"])
    assert combined_model_config == base_model_config

    torch.manual_seed(13)
    model = _model(COMBINED).train()
    assert model.edge_decoder is not None
    assert model.edge_decoder.hx_pair_relation is not None
    assert model.edge_decoder.feed_encoder is not None
    assert model.edge_decoder.describe_head()["hierarchical_descriptor_dim"] == 1600
    oper_dim = model.encoder.config.oper_dim
    batch = {
        "edge_index": torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        "x_role": torch.zeros(4, dtype=torch.long),
        "x_unit": torch.zeros(4, dtype=torch.long),
        "x_hx_role": torch.zeros(4, dtype=torch.long),
        "x_oper": torch.randn(4, oper_dim),
        "x_oper_mask": torch.ones(4, oper_dim),
        "batch": torch.zeros(4, dtype=torch.long),
        "edge_batch": torch.zeros(4, dtype=torch.long),
        "edge_stream_role": torch.zeros(4, dtype=torch.long),
        "edge_property_stream_role": torch.zeros(4, dtype=torch.long),
        "edge_stream_id": torch.arange(4),
        "edge_oper": torch.randn(4, 3),
        "edge_struct_attr": torch.randn(4, 3),
        "edge_oper_mask": torch.ones(4, 3),
        "edge_is_predictable": torch.ones(4),
        "graph_feed_input": torch.tensor([[0., 1., 2., 1., 1., 1.]]),
        "hx_pair_current_edge_index": torch.tensor([0, 1]),
        "hx_paired_edge_index": torch.tensor([1, 0]),
        "hx_pair_mask": torch.ones(2),
        "hx_pair_side": torch.tensor([1, 2]),
    }
    captured: list[torch.Tensor] = []
    hook = model.edge_decoder.hierarchical_pi_head.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach())
    )
    prediction = model(batch)["main_stream_pred"]
    hook.remove()
    assert prediction.shape == (4, 11)
    assert captured[0].shape == (4, 1600)
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    for name, module in {
        "hx": model.edge_decoder.hx_pair_relation,
        "feed": model.edge_decoder.feed_encoder,
        "shared": model.edge_decoder.hierarchical_pi_head.shared_input,
        "condition": model.edge_decoder.hierarchical_pi_head.condition_head,
        "fraction": model.edge_decoder.hierarchical_pi_head.fraction_head,
        "mass": model.edge_decoder.hierarchical_pi_head.mass_head,
        "node": model.encoder.input_encoder,
        "edge": model.encoder.edge_input_encoder,
    }.items():
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads), name
        assert any(bool((g != 0).any()) for g in grads), name


def test_hx_pair_feed_head32_shapes_broadcast_isolation_and_gradients() -> None:
    raw64 = yaml.safe_load(COMBINED.read_text(encoding="utf-8"))
    raw32 = yaml.safe_load(COMBINED32.read_text(encoding="utf-8"))
    for raw in (raw64, raw32):
        for key in ("experiment_name", "output_dir", "save_dir", "log_dir"):
            raw.pop(key)
    encoder64 = raw64["overrides"]["model"]["feed_head_conditioning"]["encoder"]
    encoder32 = raw32["overrides"]["model"]["feed_head_conditioning"]["encoder"]
    assert encoder32 == {"hidden_dim": 16, "output_dim": 32, "dropout": 0.0}
    encoder32.update(encoder64)
    raw32["overrides"]["model"]["dimension_design"]["decoder"].pop("input_dim")
    assert raw32 == raw64

    torch.manual_seed(17)
    model = _model(COMBINED32).train()
    decoder = model.edge_decoder
    assert decoder is not None
    assert decoder.hx_pair_relation is not None
    assert decoder.feed_encoder is not None
    assert decoder.describe_head()["hierarchical_descriptor_dim"] == 1568
    oper_dim = model.encoder.config.oper_dim
    batch = {
        "edge_index": torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
        "x_role": torch.zeros(4, dtype=torch.long),
        "x_unit": torch.zeros(4, dtype=torch.long),
        "x_hx_role": torch.zeros(4, dtype=torch.long),
        "x_oper": torch.randn(4, oper_dim),
        "x_oper_mask": torch.ones(4, oper_dim),
        "batch": torch.tensor([0, 0, 1, 1]),
        "edge_batch": torch.tensor([0, 0, 1, 1]),
        "edge_stream_role": torch.zeros(4, dtype=torch.long),
        "edge_property_stream_role": torch.zeros(4, dtype=torch.long),
        "edge_stream_id": torch.arange(4),
        "edge_oper": torch.randn(4, 3),
        "edge_struct_attr": torch.randn(4, 3),
        "edge_oper_mask": torch.ones(4, 3),
        "edge_is_predictable": torch.ones(4),
        "graph_feed_input": torch.tensor(
            [[0., 1., 2., 1., 1., 1.], [3., 4., 5., 1., 1., 1.]]
        ),
        "hx_pair_current_edge_index": torch.tensor([0, 1, 2, 3]),
        "hx_paired_edge_index": torch.tensor([1, 0, 3, 2]),
        "hx_pair_mask": torch.ones(4),
        "hx_pair_side": torch.tensor([1, 2, 1, 2]),
    }
    captured: list[torch.Tensor] = []
    hook = decoder.hierarchical_pi_head.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach())
    )
    prediction = model(batch)["main_stream_pred"]
    hook.remove()
    descriptor = captured[0]
    assert prediction.shape == (4, 11)
    assert descriptor.shape == (4, 1568)
    with torch.no_grad():
        feed_hidden = decoder.feed_encoder(batch["graph_feed_input"])
    assert feed_hidden.shape == (2, 32)
    assert torch.allclose(descriptor[:2, -32:], feed_hidden[0].expand(2, -1))
    assert torch.allclose(descriptor[2:, -32:], feed_hidden[1].expand(2, -1))

    prediction.square().mean().backward()
    for name, module in {
        "hx": decoder.hx_pair_relation,
        "feed": decoder.feed_encoder,
        "shared": decoder.hierarchical_pi_head.shared_input,
        "condition": decoder.hierarchical_pi_head.condition_head,
        "fraction": decoder.hierarchical_pi_head.fraction_head,
        "mass": decoder.hierarchical_pi_head.mass_head,
        "node": model.encoder.input_encoder,
        "edge": model.encoder.edge_input_encoder,
    }.items():
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert grads and all(
            g is not None and torch.isfinite(g).all() for g in grads
        ), name
        assert any(bool((g != 0).any()) for g in grads), name


@pytest.mark.parametrize(
    ("source", "feed_output_dim", "wrong_descriptor_dim"),
    (
        (FEED32, 32, 1600),
        (FEED, 64, 1568),
    ),
)
def test_feed_descriptor_dimension_validation(
    tmp_path: Path,
    source: Path,
    feed_output_dim: int,
    wrong_descriptor_dim: int,
) -> None:
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["overrides"]["model"]["dimension_design"]["decoder"][
        "input_dim"
    ] = wrong_descriptor_dim
    invalid = tmp_path / f"invalid_feed_{feed_output_dim}.yaml"
    invalid.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    experiment = load_experiment_config(invalid)
    with pytest.raises(
        ValueError,
        match=(
            rf"configured={wrong_descriptor_dim}, computed="
            rf"{1536 + feed_output_dim}, feed_output_dim={feed_output_dim}"
        ),
    ):
        model_yaml_to_encoder_config(experiment.model, experiment.data)


def test_feed32_and_feed64_checkpoints_are_intentionally_incompatible() -> None:
    feed64 = _model(FEED)
    feed32 = _model(FEED32)
    with pytest.raises(RuntimeError, match="size mismatch"):
        feed32.load_state_dict(feed64.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="size mismatch"):
        feed64.load_state_dict(feed32.state_dict(), strict=True)
