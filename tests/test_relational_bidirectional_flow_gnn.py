from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from process_graph.experiment.config_builders import model_yaml_to_encoder_config
from process_graph.experiment.loaders import load_experiment_config
from process_graph.experiment.node_balance_pi import compute_node_balance_pinn_losses
from process_graph.models.edge_decoder import (
    BidirectionalEdgeFusionMLP,
    BidirectionalRelationalEdgeReadout,
)
from process_graph.models.process_encoder import (
    ProcessEncoderConfig,
    ProcessGraphEncoder,
    ProcessGraphEncoderLayer,
    RelationalBidirectionalFlowGNNLayer,
    RelationalDifferentialUpdateBlock,
)
from process_graph.models.process_surrogate import ProcessSurrogateModel


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs/experiment/pinn/co_ratio_regime_260803"
SPECIES = ("H2O", "H2", "CH4", "CO2", "CO", "O2", "N2")


def _small_config(*, architecture: str, readout: str) -> ProcessEncoderConfig:
    return ProcessEncoderConfig(
        hidden_dim=8,
        num_layers=1,
        role_emb_dim=4,
        unit_emb_dim=4,
        oper_dim=13,
        input_mlp_layers=2,
        diff_mlp_layers=2,
        update_mlp_layers=2,
        final_mlp_layers=2,
        attn_hidden_dim=6,
        diff_mode="concat",
        fusion_mode="concat",
        global_pool="concat_set2set",
        set2set_processing_steps=2,
        dropout=0.0,
        activation="relu",
        use_role_embedding=True,
        use_unit_embedding=True,
        oper_mask_mode="concat",
        initial_residual_mode="per_layer",
        initial_residual_alpha=0.05,
        layer_residual_mode="interp",
        layer_residual_alpha=0.5,
        use_final_projection=True,
        use_edge_features=True,
        edge_stream_role_emb_dim=4,
        edge_stream_id_emb_dim=4,
        edge_stream_vocab_size=32,
        edge_hidden_dim=8,
        edge_mlp_layers=2,
        edge_oper_dim=3,
        use_edge_decoder=True,
        edge_decoder_hidden_dim=8,
        edge_decoder_dropout=0.0,
        stream_target_dim=11,
        edge_struct_dim=3,
        edge_head_type="hierarchical_reduced_pi",
        edge_head_dropout=0.0,
        cond_head_output_dim=2,
        frac_head_output_dim=7,
        mass_head_output_dim=1,
        property_head_hidden_dim=6,
        fraction_temperature=0.5,
        species_order=SPECIES,
        hierarchical_global_projection=True,
        hierarchical_global_dim=8,
        hierarchical_decoder_intermediate_dim=12,
        hierarchical_condition_hidden_dim=6,
        hierarchical_fraction_hidden_dim=7,
        hierarchical_mass_hidden_dim=6,
        hierarchical_shared_residual=False,
        hierarchical_level2_hidden_dim=4,
        flow_gnn_architecture=architecture,  # type: ignore[arg-type]
        relational_message_hidden_dim=10,
        relational_attention_hidden_dim=6,
        relational_update_hidden_dim=10,
        relational_fusion_hidden_dim=10,
        relational_dropout=0.0,
        edge_readout_mode=readout,  # type: ignore[arg-type]
        edge_readout_hidden_dim=10,
        edge_readout_dropout=0.0,
        property_prediction_edges_only=(architecture == "relational_bidirectional"),
    )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "edge_index": torch.tensor(
            [[0, 1, 0, 1], [2, 2, 3, 3]], dtype=torch.long
        ),
        "x_role": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "x_unit": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "x_oper": torch.randn(4, 13),
        "x_oper_mask": torch.ones(4, 13),
        "edge_stream_role": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "edge_stream_id": torch.tensor([2, 3, 4, 5], dtype=torch.long),
        "edge_oper": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        ),
        "edge_oper_mask": torch.ones(4, 3),
        "edge_struct_attr": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        ),
        "batch": torch.zeros(4, dtype=torch.long),
        "target_edge_mask": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "edge_is_context": torch.tensor([0.0, 0.0, 0.0, 1.0]),
        "edge_is_predictable": torch.tensor([1.0, 1.0, 1.0, 0.0]),
        "edge_is_supervised": torch.tensor([1.0, 1.0, 1.0, 0.0]),
        "edge_pinn_mask": torch.tensor([1.0, 1.0, 1.0, 0.0]),
        "y_edge_true": torch.randn(4, 11),
        "y_edge_mask": torch.tensor([1.0, 1.0, 1.0, 0.0]),
    }


def _load_config(name: str) -> tuple[ProcessEncoderConfig, ProcessSurrogateModel]:
    experiment = load_experiment_config(CONFIG_DIR / name)
    config = model_yaml_to_encoder_config(experiment.model, experiment.data)
    return config, ProcessSurrogateModel(config)


def _assert_nonzero_finite_gradient(module: torch.nn.Module) -> None:
    gradients = [parameter.grad for parameter in module.parameters()]
    assert gradients and any(
        gradient is not None
        and torch.isfinite(gradient).all()
        and bool((gradient.abs() > 0).any())
        for gradient in gradients
    )


def test_r0_r1_r2_config_contract_and_preserved_nonarchitecture_settings() -> None:
    r0_path = CONFIG_DIR / "c1_f1_ratio.yaml"
    r1_path = CONFIG_DIR / "c1_f1_ratio_r1_relational_gnn.yaml"
    r2_path = CONFIG_DIR / "c1_f1_ratio_r2_relational_edge_readout.yaml"
    raw = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in (r0_path, r1_path, r2_path)]
    r0, r1, r2 = raw

    assert r0["overrides"]["train"] == r1["overrides"]["train"] == r2["overrides"]["train"]
    assert r0["overrides"]["data"] == r1["overrides"]["data"] == r2["overrides"]["data"]
    model0 = deepcopy(r0["overrides"]["model"])
    model1 = deepcopy(r1["overrides"]["model"])
    model2 = deepcopy(r2["overrides"]["model"])
    model1.pop("flow_gnn")
    model1.pop("edge_readout")
    model2.pop("flow_gnn")
    model2.pop("edge_readout")
    model2["dimension_design"]["decoder"]["intermediate_dim"] = 768
    assert model0["known_feed_condition"]["ratio_features"]["enabled"] is False
    assert model1["known_feed_condition"]["ratio_features"]["enabled"] is True
    assert model2["known_feed_condition"]["ratio_features"]["enabled"] is True
    # Normalize the intentional ratio ablation before comparing all other settings.
    model1["known_feed_condition"]["ratio_features"]["enabled"] = False
    model2["known_feed_condition"]["ratio_features"]["enabled"] = False
    assert model0 == model1 == model2

    cfg0, model0 = _load_config(r0_path.name)
    cfg1, model1 = _load_config(r1_path.name)
    cfg2, model2 = _load_config(r2_path.name)
    assert cfg0.flow_gnn_architecture == "legacy"
    assert cfg0.edge_readout_mode == "legacy"
    assert all(isinstance(layer, ProcessGraphEncoderLayer) for layer in model0.encoder.layers)
    assert not any("relational" in key for key in model0.state_dict())
    assert model0.edge_decoder.describe_head()["hierarchical_descriptor_dim"] == 1536
    assert model0.edge_decoder.describe_head()["hierarchical_decoder_intermediate_dim"] == 768

    assert cfg1.flow_gnn_architecture == "relational_bidirectional"
    assert cfg1.edge_readout_mode == "legacy"
    assert cfg1.property_prediction_edges_only is True
    assert all(isinstance(layer, RelationalBidirectionalFlowGNNLayer) for layer in model1.encoder.layers)
    assert model1.edge_decoder.describe_head()["hierarchical_descriptor_dim"] == 1536
    assert model1.edge_decoder.describe_head()["hierarchical_decoder_intermediate_dim"] == 768

    assert cfg2.flow_gnn_architecture == "relational_bidirectional"
    assert cfg2.edge_readout_mode == "bidirectional_separate"
    assert cfg2.property_prediction_edges_only is True
    assert model2.edge_decoder.describe_head()["hierarchical_descriptor_dim"] == 1152
    assert model2.edge_decoder.describe_head()["hierarchical_decoder_intermediate_dim"] == 896


def test_legacy_default_and_explicit_selection_have_identical_state_and_output() -> None:
    default_config = _small_config(architecture="legacy", readout="legacy")
    explicit_config = replace(default_config, flow_gnn_architecture="legacy")
    torch.manual_seed(19)
    default_model = ProcessGraphEncoder(default_config).eval()
    torch.manual_seed(19)
    explicit_model = ProcessGraphEncoder(explicit_config).eval()
    assert list(default_model.state_dict()) == list(explicit_model.state_dict())
    explicit_model.load_state_dict(default_model.state_dict(), strict=True)
    for key in default_model.state_dict():
        torch.testing.assert_close(
            default_model.state_dict()[key], explicit_model.state_dict()[key]
        )
    batch = _batch()
    output_default = default_model(data=batch)
    output_explicit = explicit_model(data=batch)
    torch.testing.assert_close(
        output_default.local_node_embeddings,
        output_explicit.local_node_embeddings,
    )
    torch.testing.assert_close(output_default.global_embedding, output_explicit.global_embedding)


def test_relational_attention_shapes_normalization_degree_zero_and_sensitivity() -> None:
    config = _small_config(
        architecture="relational_bidirectional", readout="legacy"
    )
    layer = RelationalBidirectionalFlowGNNLayer(config).eval()
    h = torch.randn(5, 8)
    edge_index = torch.tensor([[0, 1, 0, 1], [2, 2, 3, 3]], dtype=torch.long)
    edge_embeddings = torch.randn(4, 8)
    output, diagnostic = layer.forward_with_diagnostics(h, edge_index, edge_embeddings)

    assert output.shape == (5, 8)
    for name in ("forward_messages", "backward_messages"):
        assert diagnostic[name].shape == (4, 8)
    for name in (
        "forward_attention_logits",
        "backward_attention_logits",
        "forward_attention",
        "backward_attention",
    ):
        assert diagnostic[name].shape == (4,)
    for name in (
        "forward_aggregate",
        "backward_aggregate",
        "forward_updated_nodes",
        "backward_updated_nodes",
    ):
        assert diagnostic[name].shape == (5, 8)
    # Forward messages still travel src -> dst, but Flow Attention is
    # normalized across each sender's outgoing edges.
    # Backward messages travel dst -> src. Standard receiver-wise attention
    # therefore also groups by the original src index (the reverse receiver).
    for node in (0, 1):
        mask = edge_index[0] == node
        torch.testing.assert_close(diagnostic["forward_attention"][mask].sum(), torch.tensor(1.0))
    for node in (0, 1):
        mask = edge_index[0] == node
        torch.testing.assert_close(diagnostic["backward_attention"][mask].sum(), torch.tensor(1.0))
    torch.testing.assert_close(diagnostic["forward_aggregate"][[0, 1, 4]], torch.zeros(3, 8))
    torch.testing.assert_close(diagnostic["backward_aggregate"][[2, 3, 4]], torch.zeros(3, 8))
    assert torch.isfinite(output).all()

    changed_edges = edge_embeddings.clone()
    changed_edges[0] += 2.0
    changed_output, changed_diagnostic = layer.forward_with_diagnostics(
        h, edge_index, changed_edges
    )
    assert not torch.allclose(output, changed_output)
    for name in (
        "forward_messages",
        "backward_messages",
        "forward_attention_logits",
        "backward_attention_logits",
    ):
        assert not torch.allclose(diagnostic[name], changed_diagnostic[name])

    reversed_output = layer(h, edge_index.flip(0), edge_embeddings)
    assert not torch.allclose(output, reversed_output)
    assert layer.forward_message is not layer.backward_message
    assert layer.forward_attention is not layer.backward_attention
    assert isinstance(layer.forward_update, RelationalDifferentialUpdateBlock)
    assert isinstance(layer.backward_update, RelationalDifferentialUpdateBlock)
    assert layer.forward_update is not layer.backward_update
    for name in (
        "forward_differential",
        "backward_differential",
        "forward_encoded_differential",
        "backward_encoded_differential",
    ):
        assert diagnostic[name].shape == (5, 8)


def test_r2_parallel_edges_have_distinct_readouts_and_predictions() -> None:
    config = _small_config(
        architecture="relational_bidirectional",
        readout="bidirectional_separate",
    )
    model = ProcessSurrogateModel(config).eval()
    batch = _batch()
    batch["edge_index"][:, 1] = batch["edge_index"][:, 0]
    result = model(batch, return_encoder_output=True)
    outputs = result["predictions"]
    encoder = result["encoder_output"]
    readout = model.edge_decoder.bidirectional_edge_readout
    assert isinstance(readout, BidirectionalRelationalEdgeReadout)
    src, dst = encoder.edge_index
    z_forward, z_backward = readout(
        encoder.local_node_embeddings[src],
        encoder.local_node_embeddings[dst],
        encoder.edge_embeddings,
    )
    assert z_forward.shape == z_backward.shape == (4, 8)
    assert not torch.allclose(z_forward[0], z_forward[1])
    assert not torch.allclose(z_backward[0], z_backward[1])
    assert outputs["shared_edge_latent"].shape == (4, 8)
    assert outputs["main_stream_pred"].shape == (4, 11)
    assert not torch.allclose(outputs["main_stream_pred"][0], outputs["main_stream_pred"][1])
    torch.testing.assert_close(outputs["main_stream_pred"][-1], torch.zeros(11))


def test_r2_property_backward_reaches_every_new_and_preserved_module() -> None:
    config = _small_config(
        architecture="relational_bidirectional",
        readout="bidirectional_separate",
    )
    model = ProcessSurrogateModel(config).train()
    outputs = model(_batch())
    loss = outputs["main_stream_pred"].square().mean()
    loss.backward()

    layer = model.encoder.layers[0]
    assert isinstance(layer, RelationalBidirectionalFlowGNNLayer)
    modules = [
        model.encoder.input_encoder,
        model.encoder.edge_input_encoder,
        layer.forward_message,
        layer.backward_message,
        layer.forward_attention,
        layer.backward_attention,
        layer.forward_update,
        layer.backward_update,
        layer.fusion,
        model.encoder.set2set_pool,
        model.edge_decoder.hierarchical_global_projection,
        model.edge_decoder.bidirectional_edge_readout.forward_readout,
        model.edge_decoder.bidirectional_edge_readout.backward_readout,
        model.edge_decoder.hierarchical_pi_head.shared_input,
        model.edge_decoder.hierarchical_pi_head.condition_head,
        model.edge_decoder.hierarchical_pi_head.fraction_head,
        model.edge_decoder.hierarchical_pi_head.mass_head,
    ]
    for module in modules:
        assert module is not None
        _assert_nonzero_finite_gradient(module)
    _assert_nonzero_finite_gradient(layer.forward_update.diff_encoder)
    _assert_nonzero_finite_gradient(layer.backward_update.diff_encoder)


def _node_pinn_config() -> SimpleNamespace:
    return SimpleNamespace(
        use_node_mass_balance_loss=True,
        lambda_node_mass=1.0,
        node_mass_balance_relative=True,
        use_node_component_balance_loss=False,
        lambda_node_component=0.0,
        node_component_balance_relative=True,
        use_node_atom_balance_loss=False,
        lambda_node_atom=0.0,
        node_atom_balance_relative=True,
        use_node_energy_balance_loss=False,
        lambda_node_energy=0.0,
        node_energy_balance_relative=True,
        node_energy_valid_unit_types=[],
        node_energy_exclude_unit_types=[],
        node_energy_use_q=False,
        node_balance_pi={"enabled": True},
    )


def test_context_edges_affect_encoder_but_masks_exclude_labels_metrics_and_pinn() -> None:
    config = _small_config(
        architecture="relational_bidirectional",
        readout="bidirectional_separate",
    )
    model = ProcessSurrogateModel(config).eval()
    batch = _batch()
    changed_context = {key: value.clone() for key, value in batch.items()}
    changed_context["edge_stream_id"][-1] += 7
    encoded_a = model(batch, return_encoder_output=True)["encoder_output"]
    encoded_b = model(changed_context, return_encoder_output=True)["encoder_output"]
    assert not torch.allclose(encoded_a.local_node_embeddings, encoded_b.local_node_embeddings)
    assert not torch.allclose(encoded_a.global_embedding, encoded_b.global_embedding)

    # Labels and sampler metadata never enter model.forward.
    changed_labels = {key: value.clone() for key, value in batch.items()}
    changed_labels["y_edge_true"] += 1.0e6
    changed_labels["sampler_selection_flag"] = torch.ones(4)
    prediction_a = model(batch)["main_stream_pred"]
    prediction_b = model(changed_labels)["main_stream_pred"]
    torch.testing.assert_close(prediction_a, prediction_b)

    mask = batch["y_edge_mask"].unsqueeze(-1)
    target_a = torch.zeros_like(prediction_a)
    target_b = target_a.clone()
    target_b[-1] = 1.0e9
    loss_a = ((prediction_a - target_a).square() * mask).sum() / mask.sum()
    loss_b = ((prediction_a - target_b).square() * mask).sum() / mask.sum()
    torch.testing.assert_close(loss_a, loss_b)

    physical_a = {
        "mass_flow": torch.ones(4, 1),
        "frac": torch.full((4, 7), 1.0 / 7.0),
    }
    physical_b = {key: value.clone() for key, value in physical_a.items()}
    physical_b["mass_flow"][-1] = 1.0e9
    common = dict(
        edge_index=batch["edge_index"],
        edge_batch_or_graph_id=torch.zeros(4, dtype=torch.long),
        node_batch_or_graph_id=torch.zeros(4, dtype=torch.long),
        node_unit_type=batch["x_unit"],
        species_order=SPECIES,
        train_cfg=_node_pinn_config(),
        edge_pinn_mask=batch["edge_pinn_mask"],
    )
    pinn_a = compute_node_balance_pinn_losses(physical_outputs=physical_a, **common)
    pinn_b = compute_node_balance_pinn_losses(physical_outputs=physical_b, **common)
    torch.testing.assert_close(pinn_a["weighted_node_total"], pinn_b["weighted_node_total"])


def test_r0_r1_r2_hx_pair_collated_train_validation_test_smoke() -> None:
    variants = (
        ("legacy", "legacy", False),
        ("relational_bidirectional", "legacy", False),
        ("relational_bidirectional", "bidirectional_separate", False),
        ("legacy", "legacy", True),
    )
    for architecture, readout, hx_pair_enabled in variants:
        config = _small_config(architecture=architecture, readout=readout)
        if hx_pair_enabled:
            config = replace(
                config,
                hx_pair_relation_enabled=True,
                hx_pair_side_embedding_dim=4,
                hx_pair_relation_hidden_dim=6,
                hx_pair_relation_output_dim=8,
                hx_pair_relation_dropout=0.0,
                hx_pair_gate_init=0.05,
            )
        model = ProcessSurrogateModel(
            config
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
        metric_squared_error = torch.zeros(())
        for split in ("train", "validation", "test"):
            batch = _batch()
            if hx_pair_enabled:
                batch.update(
                    {
                        "hx_pair_current_edge_index": torch.tensor([0, 1]),
                        "hx_paired_edge_index": torch.tensor([1, 0]),
                        "hx_pair_mask": torch.ones(2),
                        "hx_pair_side": torch.tensor([1, 2]),
                    }
                )
            if split == "train":
                model.train()
                optimizer.zero_grad(set_to_none=True)
                outputs = model(batch)
            else:
                model.eval()
                with torch.no_grad():
                    outputs = model(batch)
            prediction = outputs["main_stream_pred"]
            mask = batch["y_edge_mask"].unsqueeze(-1)
            loss = ((prediction - batch["y_edge_true"]).square() * mask).sum() / mask.sum()
            assert torch.isfinite(loss)
            if split == "train":
                loss.backward()
                optimizer.step()
            metric_squared_error += loss.detach()

            node_pi = compute_node_balance_pinn_losses(
                physical_outputs={
                    "mass_flow": prediction[:, 9:10],
                    "frac": prediction[:, 2:9],
                },
                edge_index=batch["edge_index"],
                edge_batch_or_graph_id=torch.zeros(4, dtype=torch.long),
                node_batch_or_graph_id=torch.zeros(4, dtype=torch.long),
                node_unit_type=batch["x_unit"],
                species_order=SPECIES,
                train_cfg=_node_pinn_config(),
                edge_pinn_mask=batch["edge_pinn_mask"],
            )
            assert torch.isfinite(node_pi["weighted_node_total"])
        assert torch.isfinite(metric_squared_error)


def test_all_ten_r_experiment_configs_have_expected_shapes() -> None:
    expected = {
        "c1_f1_ratio.yaml": ("legacy", 1536, 768, False, False),
        "c1_f1_ratio_r0_hx_pair.yaml": ("legacy", 1536, 768, True, False),
        "c1_f1_ratio_r0_feed_head.yaml": ("legacy", 1600, 768, False, True),
        "c1_f1_ratio_r0_feed_head_32d.yaml": (
            "legacy", 1568, 768, False, True
        ),
        "c1_f1_ratio_r0_hx_pair_feed_head.yaml": (
            "legacy", 1600, 768, True, True
        ),
        "c1_f1_ratio_r0_hx_pair_feed_head_32d.yaml": (
            "legacy", 1568, 768, True, True
        ),
        "c1_f1_ratio_r1_relational_gnn.yaml": (
            "legacy", 1536, 768, False, False
        ),
        "c1_f1_ratio_r2a_relational_edge_readout_768.yaml": (
            "bidirectional_separate", 1152, 768, False, False
        ),
        "c1_f1_ratio_r2_relational_edge_readout.yaml": (
            "bidirectional_separate", 1152, 896, False, False
        ),
        "c1_f1_ratio_r3_relational_fused_edge_readout.yaml": (
            "bidirectional_fused", 768, 768, False, False
        ),
    }
    for filename, wanted in expected.items():
        config, model = _load_config(filename)
        summary = model.edge_decoder.describe_head()
        actual = (
            config.edge_readout_mode,
            summary["hierarchical_descriptor_dim"],
            summary["hierarchical_decoder_intermediate_dim"],
            summary["hx_pair_relation"],
            summary["feed_head_conditioning"],
        )
        assert actual == wanted, filename

    _, r2a = _load_config("c1_f1_ratio_r2a_relational_edge_readout_768.yaml")
    _, r2b = _load_config("c1_f1_ratio_r2_relational_edge_readout.yaml")
    linear_a = r2a.edge_decoder.hierarchical_pi_head.shared_input[0]
    linear_b = r2b.edge_decoder.hierarchical_pi_head.shared_input[0]
    assert tuple(linear_a.weight.shape) == (768, 1152)
    assert tuple(linear_b.weight.shape) == (896, 1152)

    raw_a = yaml.safe_load(
        (CONFIG_DIR / "c1_f1_ratio_r2a_relational_edge_readout_768.yaml").read_text(
            encoding="utf-8"
        )
    )
    raw_b = yaml.safe_load(
        (CONFIG_DIR / "c1_f1_ratio_r2_relational_edge_readout.yaml").read_text(
            encoding="utf-8"
        )
    )
    for raw in (raw_a, raw_b):
        for key in ("experiment_name", "output_dir", "save_dir", "log_dir"):
            raw.pop(key)
        raw["overrides"]["model"]["dimension_design"]["decoder"][
            "intermediate_dim"
        ] = 768
    assert raw_a == raw_b


def test_r3_fusion_shapes_sensitivity_parallel_edges_and_gradients() -> None:
    config = replace(
        _small_config(
            architecture="relational_bidirectional",
            readout="bidirectional_fused",
        ),
        edge_readout_fusion_hidden_dim=10,
        edge_readout_fusion_output_dim=8,
        edge_readout_fusion_dropout=0.0,
    )
    model = ProcessSurrogateModel(config).train()
    batch = _batch()
    batch["edge_index"][:, 1] = batch["edge_index"][:, 0]
    captured: list[torch.Tensor] = []
    hook = model.edge_decoder.hierarchical_pi_head.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach())
    )
    result = model(batch, return_encoder_output=True)
    hook.remove()
    prediction = result["predictions"]["main_stream_pred"]
    encoder = result["encoder_output"]
    readout = model.edge_decoder.bidirectional_edge_readout
    fusion = model.edge_decoder.bidirectional_edge_fusion
    assert isinstance(readout, BidirectionalRelationalEdgeReadout)
    assert isinstance(fusion, BidirectionalEdgeFusionMLP)
    src, dst = encoder.edge_index
    z_forward, z_backward = readout(
        encoder.local_node_embeddings[src],
        encoder.local_node_embeddings[dst],
        encoder.edge_embeddings,
    )
    z_bidirectional = fusion(z_forward, z_backward)
    assert z_forward.shape == z_backward.shape == z_bidirectional.shape == (4, 8)
    assert captured[0].shape == (3, 16)
    assert prediction.shape == (4, 11)
    assert not torch.allclose(z_bidirectional[0], z_bidirectional[1])
    assert not torch.allclose(
        z_bidirectional,
        fusion(z_forward + 1.0, z_backward),
    )
    assert not torch.allclose(
        z_bidirectional,
        fusion(z_forward, z_backward + 1.0),
    )

    prediction.square().mean().backward()
    layer = model.encoder.layers[0]
    modules = (
        model.encoder.input_encoder,
        model.encoder.edge_input_encoder,
        layer.forward_message,
        layer.backward_message,
        readout.forward_readout,
        readout.backward_readout,
        fusion,
        model.edge_decoder.hierarchical_pi_head.shared_input,
        model.edge_decoder.hierarchical_pi_head.condition_head,
        model.edge_decoder.hierarchical_pi_head.fraction_head,
        model.edge_decoder.hierarchical_pi_head.mass_head,
    )
    for module in modules:
        _assert_nonzero_finite_gradient(module)
