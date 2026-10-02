from __future__ import annotations

from typing import List, Literal

from ..models.process_encoder import ProcessEncoderConfig
from ..known_feed import operating_feature_dim, parse_known_feed_condition
from ..hx_pair import parse_hx_pair_relation
from ..feed_head import parse_feed_head_conditioning
from ..models.process_readout import TaskSpec
from .schema import DECODER_CATEGORIES, DataConfig, ModelYamlConfig

FeatureSource = Literal["hbar", "local", "global"]


def map_readout_feature_source(value: str) -> FeatureSource:
    if value not in {"hbar", "local", "global"}:
        raise ValueError(
            f"Unsupported readout_feature_source '{value}'. "
            "Expected one of: hbar, local, global."
        )
    return value  # type: ignore[return-value]


def model_yaml_to_encoder_config(
    model: ModelYamlConfig,
    data: DataConfig | None = None,
) -> ProcessEncoderConfig:
    known_feed_cfg = parse_known_feed_condition(
        getattr(model, "known_feed_condition", None)
    )
    oper_mask_mode = model.oper_mask_mode
    if oper_mask_mode not in {"ignore", "concat", "gated"}:
        raise ValueError(
            f"Unsupported oper_mask_mode '{oper_mask_mode}' from model yaml. "
            "Expected one of: ignore, concat, gated."
        )
    oper_mask_mode_typed: Literal["ignore", "concat", "gated"] = oper_mask_mode

    initial_residual_mode = model.initial_residual_mode
    if initial_residual_mode not in {"none", "final", "per_layer", "both"}:
        raise ValueError(
            f"Unsupported initial_residual_mode '{initial_residual_mode}' from model yaml. "
            "Expected one of: none, final, per_layer, both."
        )
    initial_residual_mode_typed: Literal[
        "none", "final", "per_layer", "both"
    ] = initial_residual_mode

    layer_residual_mode = getattr(model, "layer_residual_mode", "none")
    if layer_residual_mode not in {"none", "interp"}:
        raise ValueError(
            f"Unsupported layer_residual_mode '{layer_residual_mode}' from model yaml. "
            "Expected one of: none, interp."
        )
    layer_residual_mode_typed: Literal["none", "interp"] = layer_residual_mode

    activation = model.activation
    if activation not in {"relu", "gelu", "silu", "tanh", "leaky_relu"}:
        raise ValueError(
            f"Unsupported activation '{activation}' from model yaml. "
            "Expected one of: relu, gelu, silu, tanh, leaky_relu."
        )

    edge_head_type = str(getattr(model, "edge_head_type", "single") or "single")
    if edge_head_type not in {
        "single",
        "grouped_property",
        "pi_grouped_property",
        "hierarchical_reduced_pi",
    }:
        raise ValueError(
            f"Unsupported edge_head_type '{edge_head_type}' from model yaml. "
            "Expected one of: single, grouped_property, pi_grouped_property, "
            "hierarchical_reduced_pi."
        )
    model_species_order = [
        str(s) for s in (getattr(model, "species_order", None) or [])
    ]
    data_species_order = (
        [str(s) for s in (getattr(data, "species_order", None) or [])]
        if data is not None
        else []
    )
    species_order = model_species_order or data_species_order
    frac_head_output_dim = int(getattr(model, "frac_head_output_dim", 7))
    if species_order and len(species_order) != frac_head_output_dim:
        raise ValueError(
            "species_order length must match frac_head_output_dim when both are configured: "
            f"len(species_order)={len(species_order)} "
            f"frac_head_output_dim={frac_head_output_dim}."
        )

    adapter_raw = getattr(model, "target_branch_hidden_adapter", {}) or {}
    if not isinstance(adapter_raw, dict):
        raise TypeError("model.target_branch_hidden_adapter must be a mapping.")
    adapter_enabled = bool(adapter_raw.get("enabled", False))
    adapter_bottleneck_dim = int(adapter_raw.get("bottleneck_dim", 16))
    adapter_scale = float(adapter_raw.get("scale", 0.1))
    adapter_detach_input = bool(adapter_raw.get("detach_input", True))
    adapter_zero_init_output = bool(adapter_raw.get("zero_init_output", True))
    adapter_apply_to = tuple(
        str(v).strip().lower() for v in (adapter_raw.get("apply_to") or ())
    )
    if adapter_enabled:
        if adapter_bottleneck_dim <= 0:
            raise ValueError(
                "model.target_branch_hidden_adapter.bottleneck_dim must be positive."
            )
        if adapter_scale < 0.0:
            raise ValueError(
                "model.target_branch_hidden_adapter.scale must be non-negative."
            )
        if not adapter_apply_to:
            adapter_apply_to = ("condition", "fraction", "flow")
    allowed_adapter_branches = {"condition", "fraction", "flow"}
    unknown_adapter_branches = sorted(
        set(adapter_apply_to) - allowed_adapter_branches
    )
    if unknown_adapter_branches:
        raise ValueError(
            "model.target_branch_hidden_adapter.apply_to supports only "
            f"{sorted(allowed_adapter_branches)}, got {unknown_adapter_branches}."
        )

    property_role_raw = getattr(model, "property_stream_role", {}) or {}
    if not isinstance(property_role_raw, dict):
        raise TypeError("model.property_stream_role must be a mapping.")
    property_role_enabled = bool(property_role_raw.get("enabled", False))
    property_role_embedding_dim = int(
        property_role_raw.get("embedding_dim", 16)
    )
    property_role_unknown_id = int(property_role_raw.get("unknown_role_id", 0))
    if property_role_embedding_dim <= 0:
        raise ValueError("model.property_stream_role.embedding_dim must be positive.")
    if property_role_unknown_id < 0:
        raise ValueError(
            "model.property_stream_role.unknown_role_id must be non-negative."
        )

    flow_gnn_raw = getattr(model, "flow_gnn", {}) or {}
    if not isinstance(flow_gnn_raw, dict):
        raise TypeError("model.flow_gnn must be a mapping.")
    flow_gnn_architecture = str(
        flow_gnn_raw.get("architecture", "legacy") or "legacy"
    ).strip().lower()
    if flow_gnn_architecture not in {"legacy", "relational_bidirectional"}:
        raise ValueError(
            "model.flow_gnn.architecture must be 'legacy' or "
            f"'relational_bidirectional', got {flow_gnn_architecture!r}."
        )
    relational_message_hidden_dim = int(
        flow_gnn_raw.get("message_hidden_dim", 512)
    )
    relational_attention_hidden_dim = int(
        flow_gnn_raw.get("attention_hidden_dim", 256)
    )
    relational_update_hidden_dim = int(
        flow_gnn_raw.get("update_hidden_dim", 512)
    )
    relational_fusion_hidden_dim = int(
        flow_gnn_raw.get("fusion_hidden_dim", 512)
    )
    relational_dropout = float(flow_gnn_raw.get("dropout", 0.1))
    if any(
        value <= 0
        for value in (
            relational_message_hidden_dim,
            relational_attention_hidden_dim,
            relational_update_hidden_dim,
            relational_fusion_hidden_dim,
        )
    ):
        raise ValueError("model.flow_gnn hidden dimensions must be positive.")
    if not 0.0 <= relational_dropout < 1.0:
        raise ValueError("model.flow_gnn.dropout must be in [0, 1).")
    if str(flow_gnn_raw.get("forward_attention_group", "source")) != "source":
        raise ValueError(
            "relational FlowGNN forward attention must group by source."
        )
    if str(flow_gnn_raw.get("backward_attention_group", "source")) != "source":
        raise ValueError(
            "relational FlowGNN backward attention must group by source."
        )
    if bool(flow_gnn_raw.get("update_edge_embeddings", False)):
        raise ValueError(
            "relational FlowGNN keeps edge embeddings static; "
            "model.flow_gnn.update_edge_embeddings must be false."
        )

    edge_readout_raw = getattr(model, "edge_readout", {}) or {}
    if not isinstance(edge_readout_raw, dict):
        raise TypeError("model.edge_readout must be a mapping.")
    edge_readout_mode = str(
        edge_readout_raw.get("mode", "legacy") or "legacy"
    ).strip().lower()
    if edge_readout_mode not in {
        "legacy",
        "bidirectional_separate",
        "bidirectional_fused",
    }:
        raise ValueError(
            "model.edge_readout.mode must be 'legacy', "
            "'bidirectional_separate', or 'bidirectional_fused'; "
            f"got {edge_readout_mode!r}."
        )
    edge_readout_hidden_dim = int(edge_readout_raw.get("hidden_dim", 512))
    edge_readout_dropout = float(edge_readout_raw.get("dropout", 0.1))
    edge_fusion_raw = edge_readout_raw.get("fusion", {}) or {}
    if not isinstance(edge_fusion_raw, dict):
        raise TypeError("model.edge_readout.fusion must be a mapping.")
    edge_readout_fusion_hidden_dim = int(edge_fusion_raw.get("hidden_dim", 512))
    edge_readout_fusion_output_dim = int(
        edge_fusion_raw.get("output_dim", 0) or 0
    )
    edge_readout_fusion_dropout = float(
        edge_fusion_raw.get("dropout", edge_readout_dropout)
    )
    if edge_readout_hidden_dim <= 0:
        raise ValueError("model.edge_readout.hidden_dim must be positive.")
    if not 0.0 <= edge_readout_dropout < 1.0:
        raise ValueError("model.edge_readout.dropout must be in [0, 1).")
    if edge_readout_fusion_hidden_dim <= 0 or edge_readout_fusion_output_dim < 0:
        raise ValueError(
            "model.edge_readout.fusion hidden_dim must be positive and "
            "output_dim must be non-negative."
        )
    if not 0.0 <= edge_readout_fusion_dropout < 1.0:
        raise ValueError("model.edge_readout.fusion.dropout must be in [0, 1).")

    hx_pair_cfg = parse_hx_pair_relation(
        getattr(model, "hx_pair_relation", None)
    )
    feed_head_cfg = parse_feed_head_conditioning(
        getattr(model, "feed_head_conditioning", None)
    )

    hierarchical_raw = getattr(model, "hierarchical_pi_head", {}) or {}
    if not isinstance(hierarchical_raw, dict):
        raise TypeError("model.hierarchical_pi_head must be a mapping.")
    dimensions_raw = getattr(model, "dimension_design", {}) or {}
    if not isinstance(dimensions_raw, dict):
        raise TypeError("model.dimension_design must be a mapping.")
    node_dims = dimensions_raw.get("node", {}) or {}
    edge_dims = dimensions_raw.get("edge", {}) or {}
    gnn_dims = dimensions_raw.get("gnn", {}) or {}
    global_dims = dimensions_raw.get("global", {}) or {}
    decoder_dims = dimensions_raw.get("decoder", {}) or {}
    level1_dims = dimensions_raw.get("level1", {}) or {}
    level2_dims = dimensions_raw.get("level2", {}) or {}
    for name, block in (
        ("node", node_dims),
        ("edge", edge_dims),
        ("gnn", gnn_dims),
        ("global", global_dims),
        ("decoder", decoder_dims),
        ("level1", level1_dims),
        ("level2", level2_dims),
    ):
        if not isinstance(block, dict):
            raise TypeError(f"model.dimension_design.{name} must be a mapping.")

    operating_hidden_dim = int(node_dims.get("operating_hidden_dim", 0) or 0)
    role_emb_dim = int(
        node_dims.get("role_embedding_dim", model.role_emb_dim)
    )
    unit_emb_dim = int(
        node_dims.get("unit_embedding_dim", model.unit_emb_dim)
    )
    edge_role_emb_dim = int(
        edge_dims.get(
            "role_embedding_dim",
            model.edge_stream_role_emb_dim,
        )
    )
    edge_stream_id_emb_dim = int(
        edge_dims.get(
            "stream_id_embedding_dim",
            model.edge_stream_id_emb_dim,
        )
    )
    edge_structural_hidden_dim = int(
        edge_dims.get("structural_hidden_dim", 0) or 0
    )
    edge_hidden_dim = int(edge_dims.get("hidden_dim", 0) or 0)
    if edge_readout_fusion_output_dim == 0:
        edge_readout_fusion_output_dim = edge_hidden_dim
    use_edge_stream_id = bool(edge_dims.get("use_stream_id", True))
    force_unknown_edge_stream_id = bool(
        edge_dims.get("force_unknown_stream_id", False)
    )
    attention_num_heads = int(gnn_dims.get("attention_num_heads", 1) or 1)
    configured_node_hidden = int(node_dims.get("hidden_dim", model.hidden_dim))
    configured_gnn_hidden = int(gnn_dims.get("hidden_dim", model.hidden_dim))
    if configured_node_hidden != int(model.hidden_dim):
        raise ValueError(
            "model.dimension_design.node.hidden_dim must equal "
            "model.hidden_dim."
        )
    if configured_gnn_hidden != int(model.hidden_dim):
        raise ValueError(
            "model.dimension_design.gnn.hidden_dim must equal "
            "model.hidden_dim."
        )
    for name, value in (
        ("node.role_embedding_dim", role_emb_dim),
        ("node.unit_embedding_dim", unit_emb_dim),
        ("node.operating_hidden_dim", operating_hidden_dim),
        ("edge.role_embedding_dim", edge_role_emb_dim),
        ("edge.stream_id_embedding_dim", edge_stream_id_emb_dim),
        ("edge.structural_hidden_dim", edge_structural_hidden_dim),
        ("edge.hidden_dim", edge_hidden_dim),
    ):
        if value < 0:
            raise ValueError(
                f"model.dimension_design.{name} must be >= 0, got {value}."
            )
    if attention_num_heads != 1:
        raise ValueError(
            "The current directional FlowGNN uses one scalar attention head "
            "per direction; model.dimension_design.gnn.attention_num_heads "
            "must be 1."
        )
    if int(model.hidden_dim) % attention_num_heads != 0:
        raise ValueError(
            "model.hidden_dim must be divisible by attention_num_heads."
        )
    if model.use_role_embedding and role_emb_dim <= 0:
        raise ValueError(
            "node.role_embedding_dim must be positive when role embedding is enabled."
        )
    if model.use_unit_embedding and unit_emb_dim <= 0:
        raise ValueError(
            "node.unit_embedding_dim must be positive when unit embedding is enabled."
        )
    if edge_role_emb_dim <= 0:
        raise ValueError("edge.role_embedding_dim must be positive.")
    if use_edge_stream_id and edge_stream_id_emb_dim <= 0:
        raise ValueError(
            "edge.stream_id_embedding_dim must be positive when "
            "edge.use_stream_id=true."
        )

    hierarchical_global_projection = bool(
        global_dims.get(
            "project_output",
            hierarchical_raw.get("global_projection", True),
        )
    )
    hierarchical_global_dim = int(
        global_dims.get(
            "output_dim",
            hierarchical_raw.get("global_dim", 0),
        )
        or 0
    )
    hierarchical_decoder_intermediate_dim = int(
        decoder_dims.get("intermediate_dim", 0) or 0
    )
    configured_descriptor_input_dim = int(
        decoder_dims.get("input_dim", 0) or 0
    )
    hierarchical_shared_dim = int(
        decoder_dims.get("shared_hidden_dim", model.edge_decoder_hidden_dim)
    )
    hierarchical_condition_hidden_dim = int(
        level1_dims.get("condition_hidden_dim", 0) or 0
    )
    hierarchical_fraction_hidden_dim = int(
        level1_dims.get("fraction_hidden_dim", 0) or 0
    )
    hierarchical_mass_hidden_dim = int(
        level1_dims.get("mass_hidden_dim", 0) or 0
    )
    hierarchical_shared_residual = bool(
        decoder_dims.get("residual", True)
    )
    hierarchical_level2_hidden_dim = int(
        level2_dims.get(
            "hidden_dim",
            hierarchical_raw.get("level2_hidden_dim", 64),
        )
    )
    hierarchical_detach_level1 = bool(
        level2_dims.get(
            "detach_level1_predictions",
            hierarchical_raw.get("detach_level1_predictions", True),
        )
    )
    hierarchical_predict_volume_flow = bool(
        hierarchical_raw.get("predict_volume_flow", True)
    )
    raw_unfreeze_epoch = hierarchical_raw.get("level2_unfreeze_epoch")
    hierarchical_unfreeze_epoch = (
        None if raw_unfreeze_epoch is None else int(raw_unfreeze_epoch)
    )
    if hierarchical_global_dim < 0:
        raise ValueError(
            "model.hierarchical_pi_head.global_dim must be >= 0; "
            "use 0 to select model.hidden_dim."
        )
    for name, value in (
        ("decoder.shared_hidden_dim", hierarchical_shared_dim),
        ("level2.hidden_dim", hierarchical_level2_hidden_dim),
    ):
        if value <= 0:
            raise ValueError(
                f"model.dimension_design.{name} must be positive."
            )
    for name, value in (
        ("decoder.input_dim", configured_descriptor_input_dim),
        ("decoder.intermediate_dim", hierarchical_decoder_intermediate_dim),
        ("level1.condition_hidden_dim", hierarchical_condition_hidden_dim),
        ("level1.fraction_hidden_dim", hierarchical_fraction_hidden_dim),
        ("level1.mass_hidden_dim", hierarchical_mass_hidden_dim),
    ):
        if value < 0:
            raise ValueError(
                f"model.dimension_design.{name} must be >= 0."
            )
    resolved_intermediate_dim = (
        hierarchical_decoder_intermediate_dim or hierarchical_shared_dim
    )
    if hierarchical_shared_residual and (
        resolved_intermediate_dim != hierarchical_shared_dim
    ):
        raise ValueError(
            "model.dimension_design.decoder.residual=true requires "
            "intermediate_dim == shared_hidden_dim."
        )
    if hierarchical_unfreeze_epoch is not None and hierarchical_unfreeze_epoch < 1:
        raise ValueError(
            "model.hierarchical_pi_head.level2_unfreeze_epoch must be >= 1 or null."
        )
    if edge_head_type == "hierarchical_reduced_pi":
        if property_role_enabled:
            raise ValueError(
                "hierarchical_reduced_pi forbids property_stream_role direct concat."
            )
        if int(getattr(model, "mass_head_output_dim", 1)) != 1:
            raise ValueError(
                "hierarchical_reduced_pi requires mass_head_output_dim=1."
            )
        expected_stream_target_dim = 11 if hierarchical_predict_volume_flow else 10
        if int(getattr(model, "stream_target_dim", 11)) != expected_stream_target_dim:
            raise ValueError(
                "hierarchical_reduced_pi stream_target_dim does not match "
                "hierarchical_pi_head.predict_volume_flow: "
                f"expected={expected_stream_target_dim}."
            )
    if flow_gnn_architecture == "relational_bidirectional" and not bool(
        model.use_edge_features
    ):
        raise ValueError(
            "relational_bidirectional FlowGNN requires model.use_edge_features=true."
        )
    if edge_readout_mode in {"bidirectional_separate", "bidirectional_fused"}:
        if flow_gnn_architecture != "relational_bidirectional":
            raise ValueError(
                f"{edge_readout_mode} edge readout requires the relational "
                "bidirectional FlowGNN architecture."
            )
        if edge_head_type != "hierarchical_reduced_pi":
            raise ValueError(
                f"{edge_readout_mode} edge readout currently requires "
                "edge_head_type=hierarchical_reduced_pi."
            )
    if edge_readout_mode == "bidirectional_fused" and (
        edge_readout_fusion_output_dim != edge_hidden_dim
    ):
        raise ValueError(
            "bidirectional_fused output_dim must equal edge hidden dimension: "
            f"{edge_readout_fusion_output_dim} != {edge_hidden_dim}."
        )
    if hx_pair_cfg.enabled:
        if edge_head_type != "hierarchical_reduced_pi":
            raise ValueError(
                "HXPair requires edge_head_type=hierarchical_reduced_pi."
            )
        if int(hx_pair_cfg.output_dim) != int(edge_hidden_dim):
            raise ValueError(
                "hx_pair_relation output_dim must equal the static edge hidden "
                f"dimension: {hx_pair_cfg.output_dim} != {edge_hidden_dim}."
            )
    if feed_head_cfg.enabled:
        if not known_feed_cfg.enabled:
            raise ValueError(
                "FeedHead requires known_feed_condition.enabled=true."
            )
        if data is None or not bool(data.normalize_x_oper):
            raise ValueError(
                "FeedHead requires normalize_x_oper=true so the train-fitted operating scaler is reused."
            )
        if edge_head_type != "hierarchical_reduced_pi":
            raise ValueError(
                "FeedHead requires edge_head_type=hierarchical_reduced_pi."
            )

    if edge_head_type == "hierarchical_reduced_pi":
        encoder_global_dim = (
            2 * int(model.hidden_dim)
            if model.global_pool == "concat_set2set"
            else int(model.hidden_dim)
        )
        if hierarchical_global_projection:
            descriptor_global_dim = (
                int(hierarchical_global_dim) or int(model.hidden_dim)
            )
        else:
            descriptor_global_dim = encoder_global_dim
        resolved_edge_hidden_dim = int(edge_hidden_dim) or int(model.hidden_dim)
        if edge_readout_mode == "legacy":
            expected_descriptor_input_dim = (
                2 * int(model.hidden_dim)
                + descriptor_global_dim
                + resolved_edge_hidden_dim
                + (int(feed_head_cfg.output_dim) if feed_head_cfg.enabled else 0)
            )
        elif edge_readout_mode == "bidirectional_fused":
            expected_descriptor_input_dim = (
                int(edge_readout_fusion_output_dim)
                + descriptor_global_dim
                + (int(feed_head_cfg.output_dim) if feed_head_cfg.enabled else 0)
            )
        else:
            expected_descriptor_input_dim = (
                2 * int(model.hidden_dim)
                + descriptor_global_dim
                + (int(feed_head_cfg.output_dim) if feed_head_cfg.enabled else 0)
            )
        if (
            configured_descriptor_input_dim > 0
            and configured_descriptor_input_dim != expected_descriptor_input_dim
        ):
            raise ValueError(
                "model.dimension_design.decoder.input_dim does not match the "
                "computed edge descriptor width: "
                f"configured={configured_descriptor_input_dim}, "
                f"computed={expected_descriptor_input_dim}, "
                f"feed_output_dim={feed_head_cfg.output_dim if feed_head_cfg.enabled else 0}."
            )

    return ProcessEncoderConfig(
        hidden_dim=model.hidden_dim,
        num_layers=model.num_layers,
        role_emb_dim=role_emb_dim,
        unit_emb_dim=unit_emb_dim,
        hx_role_emb_dim=model.hx_role_emb_dim,
        oper_dim=operating_feature_dim(known_feed_cfg),
        input_mlp_layers=model.input_mlp_layers,
        diff_mlp_layers=model.diff_mlp_layers,
        update_mlp_layers=model.update_mlp_layers,
        final_mlp_layers=model.final_mlp_layers,
        attn_hidden_dim=model.attn_hidden_dim,
        dropout=model.dropout,
        activation=activation,
        diff_mode=model.diff_mode,
        fusion_mode=model.fusion_mode,
        global_pool=model.global_pool,
        set2set_processing_steps=int(
            getattr(model, "set2set_processing_steps", 3)
        ),
        use_role_embedding=model.use_role_embedding,
        use_unit_embedding=model.use_unit_embedding,
        use_hx_role_embedding=model.use_hx_role_embedding,
        oper_mask_mode=oper_mask_mode_typed,
        input_residual=model.input_residual,
        initial_residual_mode=initial_residual_mode_typed,
        initial_residual_alpha=model.initial_residual_alpha,
        layer_residual_mode=layer_residual_mode_typed,
        layer_residual_alpha=float(
            getattr(model, "layer_residual_alpha", 0.5)
        ),
        layer_residual_norm=bool(
            getattr(model, "layer_residual_norm", False)
        ),
        use_final_projection=model.use_final_projection,
        use_edge_features=model.use_edge_features,
        edge_stream_role_emb_dim=edge_role_emb_dim,
        edge_stream_id_emb_dim=edge_stream_id_emb_dim,
        edge_stream_vocab_size=model.edge_stream_vocab_size,
        use_edge_stream_id=use_edge_stream_id,
        force_unknown_edge_stream_id=force_unknown_edge_stream_id,
        operating_hidden_dim=operating_hidden_dim,
        edge_structural_hidden_dim=edge_structural_hidden_dim,
        edge_hidden_dim=edge_hidden_dim,
        attention_num_heads=attention_num_heads,
        edge_mlp_layers=model.edge_mlp_layers,
        edge_oper_dim=model.edge_oper_dim,
        use_edge_stream_head=model.use_edge_stream_head,
        edge_stream_out_dim=model.edge_stream_out_dim,
        edge_stream_mlp_hidden_dim=model.edge_stream_mlp_hidden_dim,
        edge_stream_mlp_layers=model.edge_stream_mlp_layers,
        use_edge_decoder=model.use_edge_decoder,
        edge_decoder_hidden_dim=hierarchical_shared_dim,
        edge_decoder_dropout=model.edge_decoder_dropout,
        stream_target_dim=model.stream_target_dim,
        edge_struct_dim=model.edge_struct_dim,
        edge_head_type=edge_head_type,  # type: ignore[arg-type]
        edge_head_dropout=float(getattr(model, "edge_head_dropout", 0.1)),
        edge_head_shared_dims=tuple(
            int(v)
            for v in getattr(model, "edge_head_shared_dims", (512, 512))
        ),
        edge_head_frac_dims=tuple(
            int(v)
            for v in getattr(model, "edge_head_frac_dims", (256, 128))
        ),
        edge_head_flow_dims=tuple(
            int(v)
            for v in getattr(model, "edge_head_flow_dims", (512, 256, 128))
        ),
        edge_head_cond_dims=tuple(
            int(v)
            for v in getattr(model, "edge_head_cond_dims", (256, 128))
        ),
        cond_head_output_dim=int(getattr(model, "cond_head_output_dim", 2)),
        frac_head_output_dim=frac_head_output_dim,
        mass_head_output_dim=int(getattr(model, "mass_head_output_dim", 1)),
        property_head_hidden_dim=int(
            getattr(model, "property_head_hidden_dim", 128)
        ),
        property_head_num_layers=int(
            getattr(model, "property_head_num_layers", 2)
        ),
        fraction_activation=str(
            getattr(model, "fraction_activation", "softmax")
        ),
        fraction_temperature=float(
            getattr(model, "fraction_temperature", 1.0)
        ),
        species_order=tuple(species_order),
        target_branch_hidden_adapter_enabled=adapter_enabled,
        target_branch_hidden_adapter_bottleneck_dim=adapter_bottleneck_dim,
        target_branch_hidden_adapter_scale=adapter_scale,
        target_branch_hidden_adapter_detach_input=adapter_detach_input,
        target_branch_hidden_adapter_zero_init_output=adapter_zero_init_output,
        target_branch_hidden_adapter_apply_to=adapter_apply_to,
        property_stream_role_enabled=property_role_enabled,
        property_stream_role_embedding_dim=property_role_embedding_dim,
        property_stream_role_unknown_role_id=property_role_unknown_id,
        hierarchical_global_projection=hierarchical_global_projection,
        hierarchical_global_dim=hierarchical_global_dim,
        hierarchical_level2_hidden_dim=hierarchical_level2_hidden_dim,
        hierarchical_decoder_intermediate_dim=(
            hierarchical_decoder_intermediate_dim
        ),
        hierarchical_condition_hidden_dim=(
            hierarchical_condition_hidden_dim
        ),
        hierarchical_fraction_hidden_dim=(
            hierarchical_fraction_hidden_dim
        ),
        hierarchical_mass_hidden_dim=hierarchical_mass_hidden_dim,
        hierarchical_shared_residual=hierarchical_shared_residual,
        hierarchical_detach_level1_predictions=hierarchical_detach_level1,
        hierarchical_level2_unfreeze_epoch=hierarchical_unfreeze_epoch,
        hierarchical_predict_volume_flow=hierarchical_predict_volume_flow,
        flow_gnn_architecture=flow_gnn_architecture,  # type: ignore[arg-type]
        relational_message_hidden_dim=relational_message_hidden_dim,
        relational_attention_hidden_dim=relational_attention_hidden_dim,
        relational_update_hidden_dim=relational_update_hidden_dim,
        relational_fusion_hidden_dim=relational_fusion_hidden_dim,
        relational_dropout=relational_dropout,
        edge_readout_mode=edge_readout_mode,  # type: ignore[arg-type]
        edge_readout_hidden_dim=edge_readout_hidden_dim,
        edge_readout_dropout=edge_readout_dropout,
        edge_readout_fusion_hidden_dim=edge_readout_fusion_hidden_dim,
        edge_readout_fusion_output_dim=edge_readout_fusion_output_dim,
        edge_readout_fusion_dropout=edge_readout_fusion_dropout,
        property_prediction_edges_only=(
            flow_gnn_architecture == "relational_bidirectional"
        ),
        hx_pair_relation_enabled=hx_pair_cfg.enabled,
        hx_pair_side_embedding_dim=hx_pair_cfg.side_embedding_dim,
        hx_pair_relation_hidden_dim=hx_pair_cfg.hidden_dim,
        hx_pair_relation_output_dim=hx_pair_cfg.output_dim,
        hx_pair_relation_dropout=hx_pair_cfg.dropout,
        hx_pair_gate_init=hx_pair_cfg.gate_init,
        feed_head_conditioning_enabled=feed_head_cfg.enabled,
        feed_head_input_dim=feed_head_cfg.input_dim,
        feed_head_hidden_dim=feed_head_cfg.hidden_dim,
        feed_head_output_dim=feed_head_cfg.output_dim,
        feed_head_dropout=feed_head_cfg.dropout,
    )


def build_task_specs(
    model: ModelYamlConfig,
    data: DataConfig,
) -> List[TaskSpec]:
    if getattr(data, "task_mode", "multitask") == "edge_all":
        return []
    feature_source = map_readout_feature_source(model.readout_feature_source)
    target_cfg = data.fixed_tasks["target"]
    aux_cfg = data.aux_task
    fixed_readout_type = (
        "edge"
        if getattr(data, "topology_mode", "excel_node") == "stream_edge"
        else target_cfg.readout_type
    )
    specs = [
        TaskSpec(
            name="target",
            readout_type=fixed_readout_type,
            out_dim=1,
            feature_source=feature_source,
            pooling="mean",
        )
    ]
    if getattr(data, "task_mode", "multitask") == "multitask":
        tailgas_cfg = data.fixed_tasks["tailgas"]
        specs.append(
            TaskSpec(
                name="tailgas",
                readout_type=(
                    "edge"
                    if getattr(data, "topology_mode", "excel_node")
                    == "stream_edge"
                    else tailgas_cfg.readout_type
                ),
                out_dim=1,
                feature_source=feature_source,
                pooling="mean",
            )
        )
    decoder_on = any(t.enabled for t in data.decoder_tasks.values())
    if decoder_on:
        for category in DECODER_CATEGORIES:
            cfg = data.decoder_tasks.get(category)
            if cfg is None or not cfg.enabled:
                continue
            specs.append(
                TaskSpec(
                    name=category,
                    readout_type=cfg.readout_type,
                    out_dim=1,
                    feature_source=feature_source,
                    pooling="mean",
                )
            )
        return specs
    if aux_cfg.enabled:
        specs.append(
            TaskSpec(
                name="aux",
                readout_type=aux_cfg.readout_type,
                out_dim=1,
                feature_source=feature_source,
                pooling="mean",
            )
        )
    return specs
