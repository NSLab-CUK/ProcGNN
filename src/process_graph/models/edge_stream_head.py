from __future__ import annotations

import torch
from torch import Tensor, nn

from .process_encoder import ActivationName, EncoderOutput, MLP


class EdgeStreamPredictionHead(nn.Module):
    """Predict per-edge stream feature vectors from node/edge/global embeddings."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        out_dim: int,
        mlp_hidden_dim: int,
        mlp_layers: int,
        activation: ActivationName,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        in_dim = 4 * hidden_dim
        self.mlp = MLP(
            input_dim=in_dim,
            hidden_dim=mlp_hidden_dim,
            output_dim=out_dim,
            num_layers=mlp_layers,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, encoder_output: EncoderOutput, edge_batch: Tensor) -> Tensor:
        if encoder_output.edge_embeddings is None:
            raise RuntimeError("Edge stream prediction requires encoder_output.edge_embeddings (use_edge_features=True).")
        src = encoder_output.edge_index[0]
        dst = encoder_output.edge_index[1]
        bank = encoder_output.node_embeddings
        h_src = bank.index_select(0, src)
        h_dst = bank.index_select(0, dst)
        h_e = encoder_output.edge_embeddings
        g = encoder_output.global_embedding.index_select(0, edge_batch.to(encoder_output.global_embedding.device))
        x = torch.cat([h_src, h_dst, h_e, g], dim=-1)
        return self.mlp(x)
