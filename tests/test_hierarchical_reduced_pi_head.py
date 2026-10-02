from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd
import torch

from process_graph.experiment.config_builders import (
    build_task_specs,
    model_yaml_to_encoder_config,
)
from process_graph.experiment.edge_step_pi_training import (
    _backward_clip_optimizer_step,
    get_pi_targets,
)
from process_graph.experiment.loaders import load_experiment_config
from process_graph.experiment.target_edge_10d_metrics import (
    build_target_edge_internal_metrics_by_property,
)
from process_graph.experiment.node_balance_pi import compute_node_balance_pinn_losses
from process_graph.models.edge_decoder import (
    EdgeDecoder,
    HierarchicalReducedPIHead,
)
from process_graph.models.process_encoder import ProcessEncoderConfig
from process_graph.models.process_surrogate import ProcessSurrogateModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPECIES = ("H2O", "H2", "CH4", "CO2", "CO", "O2", "N2")
SOURCE_COLUMNS = (
    "Temp",
    "Pres",
    "Vol_Flow",
    "Mole_Flow",
    "Mass_Flow",
    "Frac_H2O",
    "Frac_H2",
    "Frac_CH4",
    "Frac_CO2",
    "Frac_CO",
    "Frac_O2",
    "Frac_N2",
    "Density",
    "Enthalpy",
)
REDUCED_OUTPUT_COLUMNS = (
    "Temp",
    "Pres",
    "Frac_H2O",
    "Frac_H2",
    "Frac_CH4",
    "Frac_CO2",
    "Frac_CO",
    "Frac_O2",
    "Frac_N2",
    "Mass_Flow",
    "Vol_Flow",
)


def _small_encoder_config() -> ProcessEncoderConfig:
    return ProcessEncoderConfig(
        hidden_dim=16,
        num_layers=2,
        role_emb_dim=4,
        unit_emb_dim=4,
        oper_dim=13,
        input_mlp_layers=2,
        diff_mlp_layers=2,
        update_mlp_layers=2,
        final_mlp_layers=2,
        attn_hidden_dim=8,
        diff_mode="concat",
        fusion_mode="concat",
        global_pool="concat_set2set",
        set2set_processing_steps=2,
        dropout=0.0,
        activation="relu",
        use_role_embedding=True,
        use_unit_embedding=True,
        oper_mask_mode="concat",
        input_residual=True,
        initial_residual_mode="per_layer",
        layer_residual_mode="interp",
        use_final_projection=True,
        use_edge_features=True,
        edge_stream_role_emb_dim=4,
        edge_stream_id_emb_dim=8,
        edge_stream_vocab_size=64,
        edge_mlp_layers=2,
        edge_oper_dim=3,
        use_edge_decoder=True,
        edge_decoder_hidden_dim=12,
        stream_target_dim=11,
        edge_struct_dim=3,
        edge_head_type="hierarchical_reduced_pi",
        edge_head_dropout=0.0,
        cond_head_output_dim=2,
        frac_head_output_dim=7,
        mass_head_output_dim=1,
        property_head_hidden_dim=8,
        fraction_activation="softmax",
        fraction_temperature=0.5,
        species_order=SPECIES,
        property_stream_role_enabled=False,
        hierarchical_global_projection=True,
        hierarchical_global_dim=16,
        hierarchical_level2_hidden_dim=6,
        hierarchical_detach_level1_predictions=True,
    )


def _small_batch() -> dict[str, torch.Tensor]:
    return {
        "edge_index": torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.long),
        "x_role": torch.tensor([0, 1, 2], dtype=torch.long),
        "x_unit": torch.tensor([0, 1, 0], dtype=torch.long),
        "x_oper": torch.randn(3, 13),
        "x_oper_mask": torch.ones(3, 13),
        "edge_stream_role": torch.tensor([0, 0, 1], dtype=torch.long),
        "edge_stream_id": torch.tensor([3, 7, 9], dtype=torch.long),
        "edge_oper": torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        "edge_oper_mask": torch.ones(3, 3),
        "edge_struct_attr": torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        "batch": torch.zeros(3, dtype=torch.long),
    }


def _node_pi_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        use_node_mass_balance_loss=True,
        lambda_node_mass=0.75,
        node_mass_balance_relative=True,
        use_node_component_balance_loss=True,
        lambda_node_component=1.0e-7,
        node_component_balance_relative=True,
        use_node_atom_balance_loss=True,
        lambda_node_atom=0.2,
        node_atom_balance_relative=True,
        use_node_energy_balance_loss=False,
        lambda_node_energy=0.0,
        node_energy_balance_relative=True,
        node_energy_valid_unit_types=[],
        node_energy_exclude_unit_types=[],
        node_energy_use_q=False,
        node_balance_pi={
            "enabled": True,
            "mass": {"enabled": True, "weight": 0.75},
            "component": {"enabled": True, "weight": 1.0e-7},
            "atom": {"enabled": True, "weight": 0.2},
            "energy": {"enabled": False, "weight": 0.0},
        },
    )


def test_integrated_forward_shape_fraction_closure_and_parallel_edges() -> None:
    model = ProcessSurrogateModel(_small_encoder_config(), task_specs=[]).eval()
    batch = _small_batch()
    result = model(batch, return_encoder_output=True)
    outputs = result["predictions"]
    encoder = result["encoder_output"]

    assert outputs["main_stream_pred"].shape == (3, 11)
    assert outputs["level1_pred"].shape == (3, 10)
    assert outputs["volume_flow_pred"].shape == (3, 1)
    assert outputs["shared_edge_latent"].shape == (3, 12)
    assert torch.all(outputs["frac_pred"] >= 0)
    assert torch.allclose(
        outputs["frac_pred"].sum(dim=-1),
        torch.ones(3),
        atol=1.0e-6,
    )
    assert encoder.edge_embeddings is not None
    assert encoder.edge_embeddings.shape == (3, 16)
    assert not torch.allclose(
        encoder.edge_embeddings[0],
        encoder.edge_embeddings[1],
    )
    assert not torch.allclose(
        outputs["shared_edge_latent"][0],
        outputs["shared_edge_latent"][1],
    )


def test_level2_detach_blocks_level1_and_descriptor_gradients() -> None:
    head = HierarchicalReducedPIHead(
        input_dim=20,
        shared_dim=12,
        branch_dim=8,
        level2_hidden_dim=6,
        num_species=7,
        fraction_temperature=0.5,
        dropout=0.0,
        detach_level1_predictions=True,
        level2_unfreeze_epoch=None,
    )
    descriptor = torch.randn(4, 20, requires_grad=True)
    outputs = head(descriptor)
    outputs["volume_flow_pred"].sum().backward()

    assert descriptor.grad is None
    assert all(param.grad is None for param in head.shared_input.parameters())
    assert all(param.grad is None for param in head.condition_head.parameters())
    assert all(param.grad is None for param in head.fraction_head.parameters())
    assert all(param.grad is None for param in head.mass_head.parameters())
    assert any(param.grad is not None for param in head.volume_head.parameters())

    head.zero_grad(set_to_none=True)
    descriptor.grad = None
    head(descriptor)["level1_pred"].sum().backward()
    assert descriptor.grad is not None
    assert torch.isfinite(descriptor.grad).all()
    assert any(param.grad is not None for param in head.shared_input.parameters())
    assert all(param.grad is None for param in head.volume_head.parameters())


def test_optional_level2_unfreeze_epoch_restores_joint_gradient() -> None:
    head = HierarchicalReducedPIHead(
        input_dim=20,
        shared_dim=12,
        branch_dim=8,
        level2_hidden_dim=6,
        num_species=7,
        fraction_temperature=0.5,
        dropout=0.0,
        detach_level1_predictions=True,
        level2_unfreeze_epoch=3,
    )
    descriptor = torch.randn(4, 20, requires_grad=True)
    head.set_training_epoch(2)
    head(descriptor)["volume_flow_pred"].sum().backward()
    assert descriptor.grad is None

    head.zero_grad(set_to_none=True)
    descriptor.grad = None
    head.set_training_epoch(3)
    head(descriptor)["volume_flow_pred"].sum().backward()
    assert descriptor.grad is not None


def test_volume_flow_prediction_can_be_removed_without_changing_level1() -> None:
    head = HierarchicalReducedPIHead(
        input_dim=20,
        shared_dim=12,
        branch_dim=8,
        level2_hidden_dim=6,
        num_species=7,
        fraction_temperature=0.5,
        dropout=0.0,
        detach_level1_predictions=True,
        level2_unfreeze_epoch=None,
        predict_volume_flow=False,
    )
    outputs = head(torch.randn(4, 20))

    assert outputs["main_stream_pred"].shape == (4, 10)
    assert outputs["level1_pred"].shape == (4, 10)
    assert "volume_flow_pred" not in outputs
    assert head.volume_head is None
    assert not any(name.startswith("volume_head.") for name, _ in head.named_parameters())


def test_model_260811_novol_fast_builds_exact_10d_contract() -> None:
    experiment = load_experiment_config(
        PROJECT_ROOT / "configs" / "experiment" / "pinn" / "model_260811_novol_fast.yaml"
    )
    encoder_config = model_yaml_to_encoder_config(experiment.model, experiment.data)
    model = ProcessSurrogateModel(encoder_config, task_specs=[])
    description = model.edge_decoder.describe_head()

    assert encoder_config.stream_target_dim == 10
    assert encoder_config.hierarchical_predict_volume_flow is False
    assert description["stream_target_dim"] == 10
    assert description["flow_head_output_dim"] == 1
    assert description["hierarchical_predict_volume_flow"] is False
    assert description["stream_feature_order"] == [
        "Temp",
        "Pres",
        "Frac_H2O",
        "Frac_H2",
        "Frac_CH4",
        "Frac_CO2",
        "Frac_CO",
        "Frac_O2",
        "Frac_N2",
        "Mass_Flow",
    ]


def test_reduced_output_target_mapping_preserves_column_identity() -> None:
    y = torch.arange(14, dtype=torch.float32).view(1, 14)
    outputs = {
        "main_stream_pred": torch.zeros(1, 11),
        "T_pred": torch.zeros(1, 1),
        "P_pred": torch.zeros(1, 1),
        "frac_pred": torch.full((1, 7), 1.0 / 7.0),
        "mass_flow_pred": torch.zeros(1, 1),
        "volume_flow_pred": torch.zeros(1, 1),
    }
    mapped = get_pi_targets(
        targets={"edge_stream": y},
        target_masks={"edge_stream": torch.ones(1)},
        outputs=outputs,
        train_cfg=SimpleNamespace(
            use_rho_loss=False,
            use_h_loss=False,
            use_volume_loss=False,
            use_enthalpy_flow_loss=False,
            use_mole_flow_diagnostic=False,
        ),
        data_cfg=SimpleNamespace(species_order=SPECIES),
        edge_target_columns=SOURCE_COLUMNS,
    )
    expected = torch.tensor(
        [[float(SOURCE_COLUMNS.index(name)) for name in REDUCED_OUTPUT_COLUMNS]]
    )
    assert torch.equal(mapped["main_target"], expected)


def test_node_mass_component_atom_work_without_enthalpy() -> None:
    mass = torch.tensor([[2.0], [2.0]], requires_grad=True)
    frac = torch.zeros(2, 7)
    frac[:, 2] = 1.0
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    result = compute_node_balance_pinn_losses(
        physical_outputs={
            "mass_flow": mass,
            "frac": frac,
            "enthalpy_flow": None,
        },
        edge_index=edge_index,
        edge_batch_or_graph_id=torch.zeros(2, dtype=torch.long),
        node_batch_or_graph_id=torch.zeros(3, dtype=torch.long),
        node_unit_type=torch.tensor([17, 0, 18], dtype=torch.long),
        species_order=SPECIES,
        train_cfg=_node_pi_cfg(),
    )
    assert result["loss_node_energy"].item() == 0.0
    assert result["diagnostics"]["node_energy_contributes_to_total"] is False
    assert torch.isfinite(result["weighted_node_total"])
    result["weighted_node_total"].backward()
    assert mass.grad is not None


def test_role16_is_rejected_and_old_checkpoint_is_not_silently_loaded() -> None:
    with pytest.raises(ValueError, match="forbids property stream role"):
        EdgeDecoder(
            node_hidden_dim=16,
            global_dim=32,
            edge_struct_dim=3,
            out_dim=11,
            hidden_dim=12,
            activation="relu",
            dropout=0.0,
            edge_head_type="hierarchical_reduced_pi",
            frac_head_output_dim=7,
            mass_head_output_dim=1,
            property_head_hidden_dim=8,
            species_order=SPECIES,
            property_stream_role_enabled=True,
        )

    old_model = ProcessSurrogateModel(
        ProcessEncoderConfig(
            **{
                **_small_encoder_config().__dict__,
                "edge_head_type": "pi_grouped_property",
                "stream_target_dim": 14,
                "mass_head_output_dim": 3,
            }
        ),
        task_specs=[],
    )
    new_model = ProcessSurrogateModel(_small_encoder_config(), task_specs=[])
    with pytest.raises(RuntimeError):
        new_model.load_state_dict(old_model.state_dict(), strict=True)


def test_hierarchical_module_gradient_diagnostics_capture_pre_and_post_clip() -> None:
    model = ProcessSurrogateModel(_small_encoder_config(), task_specs=[]).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0e-3)
    diagnostics: dict[str, float] = {}
    loss = model(_small_batch())["main_stream_pred"].square().sum()
    _, applied = _backward_clip_optimizer_step(
        loss=loss,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        use_amp=False,
        grad_clip=1.0e-8,
        train_cfg=SimpleNamespace(scheduler_step_unit="epoch"),
        gradient_diagnostics=diagnostics,
        gradient_update_role="target",
    )
    assert applied
    assert diagnostics["grad_target_update_count"] == 1.0
    assert diagnostics["grad_target_clip_count"] == 1.0
    for group in (
        "edge_encoder",
        "gnn",
        "shared_edge_decoder",
        "condition_head",
        "fraction_head",
        "mass_flow_head",
        "volume_flow_head",
    ):
        pre = diagnostics.get(f"grad_target_pre_clip_{group}_sum", 0.0)
        post = diagnostics.get(f"grad_target_post_clip_{group}_sum", 0.0)
        assert pre >= 0.0
        assert post <= pre


def test_target_internal_metrics_keep_pooled_and_equal_edge_r2_separate() -> None:
    rows = pd.DataFrame(
        [
            {
                "split": "val",
                "property_name": "Frac_CO",
                "n": 100,
                "true_mean": 1.0,
                "sst": 100.0,
                "sse": 10.0,
                "mae": 0.1,
                "r2": 0.9,
            },
            {
                "split": "val",
                "property_name": "Frac_CO",
                "n": 10,
                "true_mean": 2.0,
                "sst": 10.0,
                "sse": 8.0,
                "mae": 0.4,
                "r2": 0.2,
            },
        ]
    )
    result = build_target_edge_internal_metrics_by_property(rows).iloc[0]
    assert result["edge_count"] == 2
    assert result["edge_macro_R2"] == pytest.approx(0.55)
    assert result["pooled_R2"] != pytest.approx(result["edge_macro_R2"])


@pytest.mark.parametrize(
    ("name", "descriptor", "shared", "branch"),
    (
        ("d0_global1024_shared384_branch128.yaml", 2560, 384, 128),
        ("d1_global512_shared384_branch128.yaml", 2048, 384, 128),
        ("d2_global512_shared256_branch128.yaml", 2048, 256, 128),
        ("d3_global512_shared256_branch96.yaml", 2048, 256, 96),
    ),
)
def test_ablation_configs_parse_and_build(
    name: str,
    descriptor: int,
    shared: int,
    branch: int,
) -> None:
    experiment = load_experiment_config(
        PROJECT_ROOT / "configs" / "experiment" / "pinn" / "head_260729" / name
    )
    encoder_config = model_yaml_to_encoder_config(experiment.model)
    model = ProcessSurrogateModel(
        encoder_config,
        build_task_specs(experiment.model, experiment.data),
    )
    description = model.edge_decoder.describe_head()
    assert description["hierarchical_descriptor_dim"] == descriptor
    assert description["hierarchical_shared_dim"] == shared
    assert description["hierarchical_branch_dim"] == branch
    assert description["stream_feature_order"] == list(REDUCED_OUTPUT_COLUMNS)
    assert encoder_config.property_stream_role_enabled is False
    assert experiment.train.use_node_energy_balance_loss is False
    assert experiment.train.lambda_node_energy == 0.0


@pytest.mark.parametrize(
    (
        "name",
        "hidden",
        "edge_hidden",
        "global_dim",
        "descriptor",
        "intermediate",
        "shared",
        "condition",
        "fraction",
        "mass",
        "level2",
        "uses_stream_id",
    ),
    (
        ("e0_current512.yaml", 512, 512, 1024, 2560, 384, 384, 128, 128, 128, 64, True),
        ("e1_embed512.yaml", 512, 512, 512, 2048, 384, 384, 128, 128, 128, 64, True),
        ("e2_balanced384.yaml", 384, 384, 384, 1536, 768, 384, 96, 128, 96, 32, True),
        ("e3_balanced384_sid16.yaml", 384, 384, 384, 1536, 768, 384, 96, 128, 96, 32, True),
        ("e4_balanced384_no_sid.yaml", 384, 384, 384, 1536, 768, 384, 96, 128, 96, 32, False),
        ("e5_compact256.yaml", 256, 256, 256, 1024, 512, 256, 64, 96, 64, 32, True),
        ("e6_large512.yaml", 512, 512, 512, 2048, 1024, 512, 128, 192, 128, 64, True),
        ("e7_balanced384_narrow.yaml", 384, 384, 384, 1536, 512, 256, 64, 96, 64, 32, True),
        ("e8_compact256_midhead_focus.yaml", 256, 256, 256, 1024, 768, 384, 96, 128, 96, 32, True),
    ),
)
def test_dimension_ablation_configs_build_exact_modules(
    name: str,
    hidden: int,
    edge_hidden: int,
    global_dim: int,
    descriptor: int,
    intermediate: int,
    shared: int,
    condition: int,
    fraction: int,
    mass: int,
    level2: int,
    uses_stream_id: bool,
) -> None:
    experiment = load_experiment_config(
        PROJECT_ROOT
        / "configs"
        / "experiment"
        / "pinn"
        / "head_dimension_260729"
        / name
    )
    encoder_config = model_yaml_to_encoder_config(
        experiment.model, experiment.data
    )
    model = ProcessSurrogateModel(encoder_config, task_specs=[])
    description = model.edge_decoder.describe_head()
    assert encoder_config.hidden_dim == hidden
    assert (
        encoder_config.edge_hidden_dim or encoder_config.hidden_dim
    ) == edge_hidden
    assert description["hierarchical_global_dim"] == global_dim
    assert description["hierarchical_descriptor_dim"] == descriptor
    assert description["hierarchical_decoder_intermediate_dim"] == intermediate
    assert description["hierarchical_shared_dim"] == shared
    assert description["hierarchical_condition_hidden_dim"] == condition
    assert description["hierarchical_fraction_hidden_dim"] == fraction
    assert description["hierarchical_mass_hidden_dim"] == mass
    assert description["hierarchical_level2_hidden_dim"] == level2
    assert (
        model.encoder.edge_input_encoder.stream_embedding is not None
    ) is uses_stream_id
    assert encoder_config.property_stream_role_enabled is False
    assert experiment.train.edge_weight_target_edge == pytest.approx(5.0)
    assert experiment.train.use_node_energy_balance_loss is False


def test_dimension_design_supports_edge_hidden_half_width() -> None:
    cfg = ProcessEncoderConfig(
        **{
            **_small_encoder_config().__dict__,
            "edge_hidden_dim": 8,
            "edge_structural_hidden_dim": 4,
            "operating_hidden_dim": 6,
        }
    )
    model = ProcessSurrogateModel(cfg, task_specs=[]).eval()
    result = model(_small_batch(), return_encoder_output=True)
    encoder = result["encoder_output"]
    outputs = result["predictions"]
    assert encoder.edge_embeddings.shape == (3, 8)
    assert outputs["main_stream_pred"].shape == (3, 11)
    assert model.edge_decoder.describe_head()[
        "hierarchical_descriptor_dim"
    ] == 56


def test_stream_id_unknown_diagnostic_keeps_checkpoint_shape() -> None:
    base_cfg = _small_encoder_config()
    diagnostic_cfg = ProcessEncoderConfig(
        **{
            **base_cfg.__dict__,
            "force_unknown_edge_stream_id": True,
        }
    )
    base = ProcessSurrogateModel(base_cfg, task_specs=[]).eval()
    diagnostic = ProcessSurrogateModel(diagnostic_cfg, task_specs=[]).eval()
    diagnostic.load_state_dict(base.state_dict(), strict=True)
    batch = _small_batch()
    with torch.no_grad():
        base_pred = base(batch)["main_stream_pred"]
        unknown_pred = diagnostic(batch)["main_stream_pred"]
    assert base_pred.shape == unknown_pred.shape
    assert not torch.allclose(base_pred, unknown_pred)
