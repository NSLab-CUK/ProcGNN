from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from process_graph import GraphSample
from process_graph.models import ProcessEncoderConfig, ProcessSurrogateModel, TaskSpec
from process_graph.models.process_encoder import ProcessGraphEncoderLayer


def build_single_graph_sample() -> GraphSample:
    return GraphSample(
        process_id="ProcessSingle",
        node_names=["feed", "reactor", "product"],
        edge_index=[[0, 1, 0], [1, 2, 2]],
        x_role=[0, 1, 2],
        x_unit=[0, 2, 0],
        x_hx_role=[0, 0, 0],
        x_oper=[
            [1.0] * 13,
            [0.5] * 13,
            [0.1] * 13,
        ],
        x_oper_mask=[
            [1] * 13,
            [1] * 13,
            [1] * 13,
        ],
        targets={},
    )


def build_batched_graph_tensors() -> dict[str, torch.Tensor]:
    edge_index = torch.tensor(
        [
            [0, 1, 0, 3, 4, 3],
            [1, 2, 2, 4, 5, 5],
        ],
        dtype=torch.long,
    )
    x_role = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    x_unit = torch.tensor([0, 2, 0, 0, 2, 0], dtype=torch.long)
    x_hx_role = torch.zeros(6, dtype=torch.long)
    x_oper = torch.tensor(
        [
            [1.0] * 13,
            [0.5] * 13,
            [0.1] * 13,
            [0.9] * 13,
            [0.4] * 13,
            [0.2] * 13,
        ],
        dtype=torch.float32,
    )
    x_oper_mask = torch.ones_like(x_oper)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    return {
        "edge_index": edge_index,
        "x_role": x_role,
        "x_unit": x_unit,
        "x_hx_role": x_hx_role,
        "x_oper": x_oper,
        "x_oper_mask": x_oper_mask,
        "batch": batch,
    }


def build_task_specs() -> list[TaskSpec]:
    return [
        TaskSpec(name="graph_target", readout_type="graph", out_dim=2, feature_source="global"),
        TaskSpec(name="node_target", readout_type="node", out_dim=1, feature_source="hbar"),
        TaskSpec(name="set_target", readout_type="node_set", out_dim=1, pooling="mean"),
    ]


def build_single_task_inputs() -> dict[str, dict[str, torch.Tensor]]:
    return {
        "node_target": {"node_index": torch.tensor([1])},
        "set_target": {
            "node_index": torch.tensor([0, 1]),
            "selection_batch": torch.tensor([0, 0]),
        },
    }


def build_batched_task_inputs() -> dict[str, dict[str, torch.Tensor]]:
    return {
        "node_target": {"node_index": torch.tensor([1, 4])},
        "set_target": {
            "node_index": torch.tensor([0, 1, 3, 4]),
            "selection_batch": torch.tensor([0, 0, 1, 1]),
        },
    }


def build_config(
    *,
    diff_mode: str,
    fusion_mode: str,
    global_pool: str,
) -> ProcessEncoderConfig:
    return ProcessEncoderConfig(
        hidden_dim=32,
        num_layers=2,
        role_emb_dim=8,
        unit_emb_dim=8,
        hx_role_emb_dim=4,
        input_mlp_layers=2,
        diff_mlp_layers=2,
        update_mlp_layers=2,
        final_mlp_layers=2,
        attn_hidden_dim=16,
        diff_mode=diff_mode,
        fusion_mode=fusion_mode,
        global_pool=global_pool,
        dropout=0.0,
        activation="relu",
        use_unit_embedding=True,
        use_hx_role_embedding=True,
        oper_mask_mode="ignore",
    )


def first_linear(module: nn.Module) -> nn.Linear:
    for child in module.modules():
        if isinstance(child, nn.Linear):
            return child
    raise AssertionError(f"No Linear layer found in module {module.__class__.__name__}.")


class ProcessSurrogateOptionCoverageTest(unittest.TestCase):
    REQUIRED_CONFIGS = {
        "A_add_sum_mean": {
            "diff_mode": "add",
            "fusion_mode": "sum",
            "global_pool": "mean",
        },
        "B_add_mean_sum": {
            "diff_mode": "add",
            "fusion_mode": "mean",
            "global_pool": "sum",
        },
        "C_concat_concat_max": {
            "diff_mode": "concat",
            "fusion_mode": "concat",
            "global_pool": "max",
        },
        "D_concat_weighted_set2set": {
            "diff_mode": "concat",
            "fusion_mode": "weighted",
            "global_pool": "concat_set2set",
        },
    }

    def assert_tensor_finite(self, tensor: torch.Tensor) -> None:
        self.assertTrue(torch.isfinite(tensor).all().item(), "Found NaN or Inf in tensor output.")

    def assert_encoder_output_shapes(
        self,
        output,
        *,
        num_nodes: int,
        num_graphs: int,
        hidden_dim: int,
        num_layers: int,
        global_dim: int | None = None,
    ) -> None:
        ge_dim = global_dim if global_dim is not None else hidden_dim
        self.assertEqual(output.local_node_embeddings.shape, (num_nodes, hidden_dim))
        self.assertEqual(output.global_embedding.shape, (num_graphs, ge_dim))
        self.assertEqual(output.node_embeddings.shape, (num_nodes, hidden_dim))
        self.assertEqual(len(output.layer_node_embeddings), num_layers)

        self.assert_tensor_finite(output.local_node_embeddings)
        self.assert_tensor_finite(output.global_embedding)
        self.assert_tensor_finite(output.node_embeddings)

        for layer_output in output.layer_node_embeddings:
            self.assertEqual(layer_output.shape, (num_nodes, hidden_dim))
            self.assert_tensor_finite(layer_output)

    def assert_prediction_shapes(
        self,
        predictions: dict[str, torch.Tensor],
        *,
        num_graphs: int,
        num_selected_nodes: int,
        num_selected_sets: int,
    ) -> None:
        self.assertEqual(set(predictions.keys()), {"graph_target", "node_target", "set_target"})
        self.assertEqual(predictions["graph_target"].shape, (num_graphs, 2))
        self.assertEqual(predictions["node_target"].shape, (num_selected_nodes, 1))
        self.assertEqual(predictions["set_target"].shape, (num_selected_sets, 1))

        for value in predictions.values():
            self.assert_tensor_finite(value)

    def assert_config_sensitive_internal_shapes(
        self,
        model: ProcessSurrogateModel,
        config: ProcessEncoderConfig,
    ) -> None:
        for layer in model.encoder.layers:
            forward_update_linear = first_linear(layer.forward_update.update)
            backward_update_linear = first_linear(layer.backward_update.update)
            expected_update_input = config.hidden_dim if config.diff_mode == "add" else config.hidden_dim * 2

            self.assertEqual(forward_update_linear.in_features, expected_update_input)
            self.assertEqual(backward_update_linear.in_features, expected_update_input)

            if config.fusion_mode == "concat":
                self.assertIsNotNone(layer.fusion.concat_projection)
                self.assertEqual(layer.fusion.concat_projection.in_features, config.hidden_dim * 2)
                self.assertEqual(layer.fusion.concat_projection.out_features, config.hidden_dim)
            elif config.fusion_mode == "weighted":
                self.assertIsNotNone(layer.fusion.weight_gate)
                self.assertEqual(layer.fusion.weight_gate.in_features, config.hidden_dim * 2)
                self.assertEqual(layer.fusion.weight_gate.out_features, 1)
            else:
                self.assertIsNone(layer.fusion.concat_projection)
                self.assertIsNone(layer.fusion.weight_gate)

        if config.global_pool == "attention":
            self.assertIsNotNone(model.encoder.attention_pool)
            self.assertIsNone(model.encoder.set2set_pool)
        elif config.global_pool == "concat_set2set":
            self.assertIsNone(model.encoder.attention_pool)
            self.assertIsNotNone(model.encoder.set2set_pool)
            if config.use_final_projection:
                proj_in = first_linear(model.encoder.final_projection).in_features
                self.assertEqual(proj_in, config.hidden_dim * 3)
        else:
            self.assertIsNone(model.encoder.attention_pool)
            self.assertIsNone(model.encoder.set2set_pool)

    def test_required_option_combinations_on_single_and_batched_graphs(self) -> None:
        single_graph = build_single_graph_sample()
        batched_graph = build_batched_graph_tensors()

        for case_name, options in self.REQUIRED_CONFIGS.items():
            config = build_config(**options)
            model = ProcessSurrogateModel(encoder_config=config, task_specs=build_task_specs())
            self.assert_config_sensitive_internal_shapes(model, config)

            ge_dim = config.hidden_dim * 2 if config.global_pool == "concat_set2set" else config.hidden_dim
            with self.subTest(case=case_name, graph_type="single"):
                encoder_output = model(single_graph, decode=False)
                self.assert_encoder_output_shapes(
                    encoder_output,
                    num_nodes=3,
                    num_graphs=1,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    global_dim=ge_dim,
                )

                predictions = model(single_graph, task_inputs=build_single_task_inputs())
                self.assert_prediction_shapes(
                    predictions,
                    num_graphs=1,
                    num_selected_nodes=1,
                    num_selected_sets=1,
                )

            with self.subTest(case=case_name, graph_type="batched"):
                encoder_output = model(batched_graph, decode=False)
                self.assert_encoder_output_shapes(
                    encoder_output,
                    num_nodes=6,
                    num_graphs=2,
                    hidden_dim=config.hidden_dim,
                    num_layers=config.num_layers,
                    global_dim=ge_dim,
                )

                predictions = model(batched_graph, task_inputs=build_batched_task_inputs())
                self.assert_prediction_shapes(
                    predictions,
                    num_graphs=2,
                    num_selected_nodes=2,
                    num_selected_sets=2,
                )

    def test_forward_and_backward_branches_use_distinct_normalization_paths(self) -> None:
        config = build_config(diff_mode="add", fusion_mode="sum", global_pool="mean")
        layer = ProcessGraphEncoderLayer(config)

        captured: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        def capture_forward(self, h, sender_index, receiver_index, normalize_index):
            captured["forward"] = (
                sender_index.detach().clone(),
                receiver_index.detach().clone(),
                normalize_index.detach().clone(),
            )
            return torch.zeros_like(h)

        def capture_backward(self, h, sender_index, receiver_index, normalize_index):
            captured["backward"] = (
                sender_index.detach().clone(),
                receiver_index.detach().clone(),
                normalize_index.detach().clone(),
            )
            return torch.zeros_like(h)

        layer.forward_attention.forward = types.MethodType(capture_forward, layer.forward_attention)
        layer.backward_attention.forward = types.MethodType(capture_backward, layer.backward_attention)

        hidden = torch.randn(4, config.hidden_dim)
        edge_index = torch.tensor(
            [
                [0, 0, 1, 2],
                [1, 2, 3, 3],
            ],
            dtype=torch.long,
        )
        src_index, dst_index = edge_index

        _ = layer(hidden, edge_index)

        forward_sender, forward_receiver, forward_norm = captured["forward"]
        backward_sender, backward_receiver, backward_norm = captured["backward"]

        self.assertTrue(torch.equal(forward_sender, src_index))
        self.assertTrue(torch.equal(forward_receiver, dst_index))
        self.assertTrue(torch.equal(forward_norm, src_index))

        self.assertTrue(torch.equal(backward_sender, dst_index))
        self.assertTrue(torch.equal(backward_receiver, src_index))
        self.assertTrue(torch.equal(backward_norm, src_index))

        self.assertFalse(torch.equal(forward_sender, backward_sender))
        self.assertFalse(torch.equal(forward_receiver, backward_receiver))

    def test_diff_mode_controls_update_mlp_input_dimension(self) -> None:
        hidden_dim = 32
        add_config = build_config(diff_mode="add", fusion_mode="sum", global_pool="mean")
        concat_config = build_config(diff_mode="concat", fusion_mode="concat", global_pool="max")

        add_model = ProcessSurrogateModel(encoder_config=add_config, task_specs=build_task_specs())
        concat_model = ProcessSurrogateModel(encoder_config=concat_config, task_specs=build_task_specs())

        for layer in add_model.encoder.layers:
            self.assertEqual(first_linear(layer.forward_update.update).in_features, hidden_dim)
            self.assertEqual(first_linear(layer.backward_update.update).in_features, hidden_dim)

        for layer in concat_model.encoder.layers:
            self.assertEqual(first_linear(layer.forward_update.update).in_features, hidden_dim * 2)
            self.assertEqual(first_linear(layer.backward_update.update).in_features, hidden_dim * 2)

    def test_layer_residual_interp_preserves_hidden_shape(self) -> None:
        batched_graph = build_batched_graph_tensors()

        config = build_config(diff_mode="add", fusion_mode="sum", global_pool="mean")
        config.layer_residual_mode = "interp"
        config.layer_residual_alpha = 0.5
        config.layer_residual_norm = False

        model = ProcessSurrogateModel(encoder_config=config, task_specs=build_task_specs())
        self.assertIsNone(model.encoder.layer_residual_norms)
        output = model(batched_graph, decode=False)
        self.assert_encoder_output_shapes(
            output,
            num_nodes=6,
            num_graphs=2,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )

        config_norm = build_config(diff_mode="add", fusion_mode="sum", global_pool="mean")
        config_norm.layer_residual_mode = "interp"
        config_norm.layer_residual_alpha = 0.3
        config_norm.layer_residual_norm = True
        model_norm = ProcessSurrogateModel(encoder_config=config_norm, task_specs=build_task_specs())
        self.assertIsNotNone(model_norm.encoder.layer_residual_norms)
        self.assertEqual(len(model_norm.encoder.layer_residual_norms), config_norm.num_layers)
        output_norm = model_norm(batched_graph, decode=False)
        self.assert_encoder_output_shapes(
            output_norm,
            num_nodes=6,
            num_graphs=2,
            hidden_dim=config_norm.hidden_dim,
            num_layers=config_norm.num_layers,
        )


if __name__ == "__main__":
    unittest.main()
