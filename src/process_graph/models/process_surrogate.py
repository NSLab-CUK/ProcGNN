from __future__ import annotations

from typing import Any, Mapping, Sequence

from torch import Tensor, nn

from .edge_decoder import EdgeDecoder
from .edge_stream_head import EdgeStreamPredictionHead
from .process_encoder import EncoderOutput, ProcessEncoderConfig, ProcessGraphEncoder, global_embedding_dim
from .process_readout import TaskReadoutHead, TaskSpec


class ProcessSurrogateModel(nn.Module):
    """Top-level model that combines a reusable encoder with flexible task heads."""

    def __init__(
        self,
        encoder_config: ProcessEncoderConfig,
        task_specs: Sequence[TaskSpec] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = ProcessGraphEncoder(encoder_config)
        self.task_specs = list(task_specs or [])
        self._validate_task_specs(self.task_specs)

        self.heads = nn.ModuleDict(
            {
                spec.name: TaskReadoutHead(
                    spec=spec,
                    hidden_dim=self.encoder.hidden_dim,
                    config=encoder_config,
                )
                for spec in self.task_specs
            }
        )
        self.edge_stream_head: EdgeStreamPredictionHead | None = None
        self.edge_decoder: EdgeDecoder | None = None
        if getattr(encoder_config, "use_edge_stream_head", False) and getattr(encoder_config, "use_edge_decoder", False):
            raise ValueError("use_edge_stream_head and use_edge_decoder cannot both be True.")
        if getattr(encoder_config, "use_edge_stream_head", False):
            if not encoder_config.use_edge_features:
                raise ValueError("encoder_config.use_edge_stream_head=True requires use_edge_features=True.")
            act = encoder_config.activation
            if act not in {"relu", "gelu", "silu", "tanh", "leaky_relu"}:
                raise ValueError(f"Unsupported activation for edge stream head: {act!r}.")
            self.edge_stream_head = EdgeStreamPredictionHead(
                hidden_dim=encoder_config.hidden_dim,
                out_dim=int(encoder_config.edge_stream_out_dim),
                mlp_hidden_dim=int(encoder_config.edge_stream_mlp_hidden_dim),
                mlp_layers=int(encoder_config.edge_stream_mlp_layers),
                activation=act,
                dropout=float(encoder_config.dropout),
            )
        if getattr(encoder_config, "use_edge_decoder", False):
            act = encoder_config.activation
            if act not in {"relu", "gelu", "silu", "tanh", "leaky_relu"}:
                raise ValueError(f"Unsupported activation for edge decoder: {act!r}.")
            # edge_all EdgeDecoder uses batch edge_struct_attr; attention edge_embeddings are optional.
            self.edge_decoder = EdgeDecoder(
                node_hidden_dim=int(encoder_config.hidden_dim),
                global_dim=global_embedding_dim(encoder_config),
                edge_struct_dim=int(encoder_config.edge_struct_dim),
                edge_embedding_dim=(
                    int(encoder_config.edge_hidden_dim)
                    if int(encoder_config.edge_hidden_dim) > 0
                    else int(encoder_config.hidden_dim)
                ),
                out_dim=int(encoder_config.stream_target_dim),
                hidden_dim=int(encoder_config.edge_decoder_hidden_dim),
                activation=act,
                dropout=float(encoder_config.dropout),
                mlp_dropout=float(getattr(encoder_config, "edge_decoder_dropout", 0.0)),
                edge_head_type=getattr(encoder_config, "edge_head_type", "single"),
                edge_head_dropout=float(getattr(encoder_config, "edge_head_dropout", 0.1)),
                edge_head_shared_dims=getattr(encoder_config, "edge_head_shared_dims", (512, 512)),
                edge_head_frac_dims=getattr(encoder_config, "edge_head_frac_dims", (256, 128)),
                edge_head_flow_dims=getattr(encoder_config, "edge_head_flow_dims", (512, 256, 128)),
                edge_head_cond_dims=getattr(encoder_config, "edge_head_cond_dims", (256, 128)),
                cond_head_output_dim=int(getattr(encoder_config, "cond_head_output_dim", 2)),
                frac_head_output_dim=int(getattr(encoder_config, "frac_head_output_dim", 7)),
                mass_head_output_dim=int(getattr(encoder_config, "mass_head_output_dim", 1)),
                property_head_hidden_dim=int(getattr(encoder_config, "property_head_hidden_dim", 128)),
                property_head_num_layers=int(getattr(encoder_config, "property_head_num_layers", 2)),
                fraction_activation=str(getattr(encoder_config, "fraction_activation", "softmax")),
                fraction_temperature=float(getattr(encoder_config, "fraction_temperature", 1.0)),
                species_order=getattr(encoder_config, "species_order", ()),
                target_branch_hidden_adapter_enabled=bool(
                    getattr(encoder_config, "target_branch_hidden_adapter_enabled", False)
                ),
                target_branch_hidden_adapter_bottleneck_dim=int(
                    getattr(encoder_config, "target_branch_hidden_adapter_bottleneck_dim", 16)
                ),
                target_branch_hidden_adapter_scale=float(
                    getattr(encoder_config, "target_branch_hidden_adapter_scale", 0.1)
                ),
                target_branch_hidden_adapter_detach_input=bool(
                    getattr(encoder_config, "target_branch_hidden_adapter_detach_input", True)
                ),
                target_branch_hidden_adapter_zero_init_output=bool(
                    getattr(encoder_config, "target_branch_hidden_adapter_zero_init_output", True)
                ),
                target_branch_hidden_adapter_apply_to=getattr(
                    encoder_config, "target_branch_hidden_adapter_apply_to", ()
                ),
                property_stream_role_enabled=bool(
                    getattr(encoder_config, "property_stream_role_enabled", False)
                ),
                property_stream_role_embedding_dim=int(
                    getattr(encoder_config, "property_stream_role_embedding_dim", 16)
                ),
                property_stream_role_unknown_role_id=int(
                    getattr(encoder_config, "property_stream_role_unknown_role_id", 0)
                ),
                hierarchical_global_projection=bool(
                    getattr(encoder_config, "hierarchical_global_projection", True)
                ),
                hierarchical_global_dim=int(
                    getattr(encoder_config, "hierarchical_global_dim", 0)
                ),
                hierarchical_level2_hidden_dim=int(
                    getattr(encoder_config, "hierarchical_level2_hidden_dim", 64)
                ),
                hierarchical_decoder_intermediate_dim=int(
                    getattr(
                        encoder_config,
                        "hierarchical_decoder_intermediate_dim",
                        0,
                    )
                ),
                hierarchical_condition_hidden_dim=int(
                    getattr(
                        encoder_config,
                        "hierarchical_condition_hidden_dim",
                        0,
                    )
                ),
                hierarchical_fraction_hidden_dim=int(
                    getattr(
                        encoder_config,
                        "hierarchical_fraction_hidden_dim",
                        0,
                    )
                ),
                hierarchical_mass_hidden_dim=int(
                    getattr(
                        encoder_config,
                        "hierarchical_mass_hidden_dim",
                        0,
                    )
                ),
                hierarchical_shared_residual=bool(
                    getattr(
                        encoder_config,
                        "hierarchical_shared_residual",
                        True,
                    )
                ),
                hierarchical_detach_level1_predictions=bool(
                    getattr(
                        encoder_config,
                        "hierarchical_detach_level1_predictions",
                        True,
                    )
                ),
                hierarchical_level2_unfreeze_epoch=getattr(
                    encoder_config,
                    "hierarchical_level2_unfreeze_epoch",
                    None,
                ),
                hierarchical_predict_volume_flow=bool(
                    getattr(encoder_config, "hierarchical_predict_volume_flow", True)
                ),
                edge_readout_mode=str(
                    getattr(encoder_config, "edge_readout_mode", "legacy")
                ),
                edge_readout_hidden_dim=int(
                    getattr(encoder_config, "edge_readout_hidden_dim", 512)
                ),
                edge_readout_dropout=float(
                    getattr(encoder_config, "edge_readout_dropout", 0.1)
                ),
                edge_readout_fusion_hidden_dim=int(
                    getattr(encoder_config, "edge_readout_fusion_hidden_dim", 512)
                ),
                edge_readout_fusion_output_dim=int(
                    getattr(
                        encoder_config,
                        "edge_readout_fusion_output_dim",
                        encoder_config.hidden_dim,
                    )
                ),
                edge_readout_fusion_dropout=float(
                    getattr(encoder_config, "edge_readout_fusion_dropout", 0.1)
                ),
                property_prediction_edges_only=bool(
                    getattr(
                        encoder_config,
                        "property_prediction_edges_only",
                        False,
                    )
                ),
                hx_pair_relation_enabled=bool(
                    getattr(encoder_config, "hx_pair_relation_enabled", False)
                ),
                hx_pair_side_embedding_dim=int(
                    getattr(encoder_config, "hx_pair_side_embedding_dim", 16)
                ),
                hx_pair_relation_hidden_dim=int(
                    getattr(encoder_config, "hx_pair_relation_hidden_dim", 256)
                ),
                hx_pair_relation_output_dim=int(
                    getattr(encoder_config, "hx_pair_relation_output_dim", 384)
                ),
                hx_pair_relation_dropout=float(
                    getattr(encoder_config, "hx_pair_relation_dropout", 0.1)
                ),
                hx_pair_gate_init=float(
                    getattr(encoder_config, "hx_pair_gate_init", 0.05)
                ),
                feed_head_conditioning_enabled=bool(
                    getattr(encoder_config, "feed_head_conditioning_enabled", False)
                ),
                feed_head_input_dim=int(
                    getattr(encoder_config, "feed_head_input_dim", 6)
                ),
                feed_head_hidden_dim=int(
                    getattr(encoder_config, "feed_head_hidden_dim", 32)
                ),
                feed_head_output_dim=int(
                    getattr(encoder_config, "feed_head_output_dim", 64)
                ),
                feed_head_dropout=float(
                    getattr(encoder_config, "feed_head_dropout", 0.0)
                ),
            )

    @staticmethod
    def _validate_task_specs(task_specs: Sequence[TaskSpec]) -> None:
        seen: set[str] = set()
        duplicates: list[str] = []

        for spec in task_specs:
            if spec.name in seen:
                duplicates.append(spec.name)
            seen.add(spec.name)

        if duplicates:
            dup_str = ", ".join(sorted(set(duplicates)))
            raise ValueError(f"Duplicate task head names are not allowed: {dup_str}.")

    def encode(self, data: Any = None, **kwargs: Any) -> EncoderOutput:
        """Return encoder features without applying any task decoder."""
        return self.encoder(data=data, **kwargs)

    def predict(
        self,
        encoder_output: EncoderOutput,
        task_inputs: Mapping[str, Mapping[str, Any]] | None = None,
        *,
        batch_dict: Mapping[str, Any] | None = None,
    ) -> dict[str, Tensor]:
        """Apply all configured task heads to a precomputed encoder output."""
        predictions: dict[str, Tensor] = {}
        task_inputs = task_inputs or {}

        for name, head in self.heads.items():
            predictions[name] = head(encoder_output, task_inputs.get(name))
        if self.edge_stream_head is not None:
            edge_task = (task_inputs or {}).get("edge_stream") or {}
            edge_batch = edge_task.get("edge_batch")
            if edge_batch is None:
                raise ValueError(
                    "edge_stream prediction head is enabled but task_inputs['edge_stream']['edge_batch'] is missing."
                )
            predictions["edge_stream"] = self.edge_stream_head(encoder_output, edge_batch)
        if self.edge_decoder is not None:
            src = batch_dict or {}
            edge_struct = src.get("edge_struct_attr")
            if edge_struct is None:
                edge_struct = src.get("edge_oper")
            if edge_struct is None:
                raise ValueError(
                    "edge_decoder is enabled but batch dict has no 'edge_struct_attr' or 'edge_oper' (structural edge features)."
                )
            edge_decoded = self.edge_decoder(
                encoder_output.local_node_embeddings,
                encoder_output.edge_index,
                edge_struct,
                encoder_output.global_embedding,
                encoder_output.batch,
                edge_embeddings=encoder_output.edge_embeddings,
                target_edge_mask=src.get("target_edge_mask"),
                property_stream_role=src.get("edge_property_stream_role"),
                # ``edge_prediction_mask`` is an opt-in inference override.
                # It permits target-edge-only decoding while preserving the
                # full flowsheet encoder pass required for graph context.
                prediction_edge_mask=src.get(
                    "edge_prediction_mask", src.get("edge_is_predictable")
                ),
                hx_pair_current_edge_index=src.get("hx_pair_current_edge_index"),
                hx_paired_edge_index=src.get("hx_paired_edge_index"),
                hx_pair_mask=src.get("hx_pair_mask"),
                hx_pair_side=src.get("hx_pair_side"),
                graph_feed_input=src.get("graph_feed_input"),
            )
            if isinstance(edge_decoded, Mapping):
                predictions.update(edge_decoded)
            else:
                predictions["y_edge_pred"] = edge_decoded
        return predictions

    def set_training_epoch(self, epoch: int) -> None:
        """Update epoch-aware decoder behavior without changing forward inputs."""
        if self.edge_decoder is not None:
            self.edge_decoder.set_training_epoch(int(epoch))

    def forward(
        self,
        data: Any = None,
        *,
        decode: bool = True,
        return_encoder_output: bool = False,
        task_inputs: Mapping[str, Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> EncoderOutput | dict[str, Tensor] | dict[str, EncoderOutput | dict[str, Tensor]]:
        """Run encoder only or encoder + task heads."""
        encoder_output = self.encode(data=data, **kwargs)

        if not decode:
            return encoder_output

        batch_dict = data if isinstance(data, dict) else None
        has_any_head = bool(self.heads) or self.edge_stream_head is not None or self.edge_decoder is not None
        predictions = (
            self.predict(encoder_output, task_inputs=task_inputs, batch_dict=batch_dict)
            if has_any_head
            else {}
        )

        if return_encoder_output:
            return {
                "encoder_output": encoder_output,
                "predictions": predictions,
            }

        if not predictions:
            return encoder_output

        return predictions
