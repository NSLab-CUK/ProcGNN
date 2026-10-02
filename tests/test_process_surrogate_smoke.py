from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from process_graph import GraphSample
from process_graph.models import ProcessEncoderConfig, ProcessGraphEncoder, ProcessSurrogateModel, TaskSpec


def build_single_graph_sample() -> GraphSample:
    return GraphSample(
        process_id="Process1",
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


class ProcessSurrogateSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ProcessEncoderConfig(
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
            diff_mode="concat",
            fusion_mode="weighted",
            global_pool="attention",
            dropout=0.0,
            activation="relu",
            use_unit_embedding=True,
            use_hx_role_embedding=True,
            oper_mask_mode="ignore",
        )

    def test_encoder_accepts_graph_sample(self) -> None:
        encoder = ProcessGraphEncoder(self.config)
        graph_sample = build_single_graph_sample()

        output = encoder(graph_sample)

        self.assertEqual(output.local_node_embeddings.shape, (3, 32))
        self.assertEqual(output.global_embedding.shape, (1, 32))
        self.assertEqual(output.node_embeddings.shape, (3, 32))
        self.assertEqual(len(output.layer_node_embeddings), 2)

    def test_surrogate_returns_prediction_dict_for_batched_graphs(self) -> None:
        model = ProcessSurrogateModel(
            encoder_config=self.config,
            task_specs=[
                TaskSpec(name="graph_target", readout_type="graph", out_dim=2, feature_source="global"),
                TaskSpec(name="node_target", readout_type="node", out_dim=1, feature_source="hbar"),
                TaskSpec(name="set_target", readout_type="node_set", out_dim=1, pooling="mean"),
            ],
        )
        batch_data = build_batched_graph_tensors()

        predictions = model(
            batch_data,
            task_inputs={
                "node_target": {"node_index": torch.tensor([1, 4])},
                "set_target": {
                    "node_index": torch.tensor([0, 1, 3, 4]),
                    "selection_batch": torch.tensor([0, 0, 1, 1]),
                },
            },
        )

        self.assertEqual(predictions["graph_target"].shape, (2, 2))
        self.assertEqual(predictions["node_target"].shape, (2, 1))
        self.assertEqual(predictions["set_target"].shape, (2, 1))

    def test_surrogate_can_return_encoder_output_only(self) -> None:
        model = ProcessSurrogateModel(encoder_config=self.config)
        batch_data = build_batched_graph_tensors()

        encoder_output = model(batch_data, decode=False)

        self.assertEqual(encoder_output.global_embedding.shape, (2, 32))
        self.assertEqual(encoder_output.node_embeddings.shape, (6, 32))


if __name__ == "__main__":
    unittest.main()
