from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
from torch import Tensor, nn

from .process_encoder import EncoderOutput, MLP, ProcessEncoderConfig, global_embedding_dim

FeatureSource = Literal["hbar", "local", "global"]
ReadoutType = Literal["node", "node_set", "graph", "edge"]
SetPoolMode = Literal["mean", "sum", "max", "attention"]


@dataclass
class TaskSpec:
    """Declarative readout specification for one prediction head."""

    name: str
    readout_type: ReadoutType
    out_dim: int
    feature_source: FeatureSource = "hbar"
    pooling: SetPoolMode = "mean"
    head_hidden_dim: int | None = None
    head_layers: int = 2


def _scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    if src.numel() == 0:
        return out
    out.index_add_(0, index, src)
    return out


def _scatter_mean(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    if src.numel() == 0:
        return torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    summed = _scatter_sum(src, index, dim_size)
    counts = torch.bincount(index, minlength=dim_size).to(src.device, src.dtype).unsqueeze(-1)
    return summed / counts.clamp_min(1.0)


def _scatter_max(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    out = torch.full(
        (dim_size, src.size(-1)),
        fill_value=torch.finfo(src.dtype).min,
        device=src.device,
        dtype=src.dtype,
    )
    if src.numel() == 0:
        return out
    out.scatter_reduce_(0, index.unsqueeze(-1).expand_as(src), src, reduce="amax", include_self=True)
    out[out == torch.finfo(src.dtype).min] = 0.0
    return out


def _segment_softmax(logits: Tensor, index: Tensor, num_segments: int) -> Tensor:
    if logits.ndim != 1:
        raise ValueError(f"Segment softmax expects 1D logits, got shape {tuple(logits.shape)}.")

    if logits.numel() == 0:
        return logits

    max_values = torch.full(
        (num_segments,),
        fill_value=torch.finfo(logits.dtype).min,
        device=logits.device,
        dtype=logits.dtype,
    )
    max_values.scatter_reduce_(0, index, logits, reduce="amax", include_self=True)
    stabilized = logits - max_values.index_select(0, index)
    exp_logits = stabilized.exp()
    denom = torch.zeros(num_segments, device=logits.device, dtype=logits.dtype)
    denom.index_add_(0, index, exp_logits)
    return exp_logits / denom.index_select(0, index).clamp_min(1e-12)


def _to_long_tensor(value: Any, device: torch.device, name: str) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device=device, dtype=torch.long)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(tensor.shape)}.")
    return tensor


def _to_bool_tensor(value: Any, device: torch.device, name: str) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device=device, dtype=torch.bool)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {tuple(tensor.shape)}.")
    return tensor


def _make_stable_contiguous_group_index(index: Tensor) -> tuple[Tensor, int]:
    """
    Remap arbitrary group ids to [0, 1, ..., K-1] while preserving
    order of first appearance.

    Example:
        [7, 7, 3, 3] -> [0, 0, 1, 1]
        [3, 3, 7, 7] -> [0, 0, 1, 1]
    """
    if index.ndim != 1:
        raise ValueError(f"group index must be 1D, got shape {tuple(index.shape)}.")
    if index.numel() == 0:
        return index, 0

    unique_vals, inverse = torch.unique(index, sorted=True, return_inverse=True)
    num_groups = unique_vals.numel()

    positions = torch.arange(index.numel(), device=index.device, dtype=torch.long)
    first_pos = torch.full(
        (num_groups,),
        fill_value=index.numel(),
        device=index.device,
        dtype=torch.long,
    )
    first_pos.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)

    order = torch.argsort(first_pos)
    remap = torch.empty_like(order)
    remap[order] = torch.arange(num_groups, device=index.device, dtype=torch.long)

    stable_inverse = remap.index_select(0, inverse)
    return stable_inverse, int(num_groups)


class SetAttentionPool(nn.Module):
    """Attention pooling over arbitrary node subsets."""

    def __init__(self, hidden_dim: int, config: ProcessEncoderConfig) -> None:
        super().__init__()
        self.scorer = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=2,
            activation=config.activation,
            dropout=config.dropout,
        )

    def forward(self, x: Tensor, group_index: Tensor, num_groups: int) -> Tensor:
        scores = self.scorer(x).squeeze(-1)
        weights = _segment_softmax(scores, group_index, num_groups)
        return _scatter_sum(x * weights.unsqueeze(-1), group_index, num_groups)


class TaskReadoutHead(nn.Module):
    """Flexible task head supporting node, node-set, and graph readout."""

    def __init__(self, spec: TaskSpec, hidden_dim: int, config: ProcessEncoderConfig) -> None:
        super().__init__()
        self.spec = spec
        self.hidden_dim = hidden_dim

        self._validate_spec()

        self.set_attention_pool = (
            SetAttentionPool(hidden_dim, config) if spec.pooling == "attention" else None
        )

        decoder_hidden_dim = spec.head_hidden_dim or hidden_dim
        if decoder_hidden_dim <= 0:
            raise ValueError(
                f"Task '{self.spec.name}' has invalid head_hidden_dim={decoder_hidden_dim}."
            )

        decoder_input_dim = hidden_dim
        if spec.readout_type == "edge":
            decoder_input_dim = hidden_dim * (3 if config.use_edge_features else 2)
        elif spec.feature_source == "global":
            decoder_input_dim = global_embedding_dim(config)

        self.decoder = MLP(
            input_dim=decoder_input_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=spec.out_dim,
            num_layers=spec.head_layers,
            activation=config.activation,
            dropout=config.dropout,
        )

    def _validate_spec(self) -> None:
        if self.spec.head_layers < 1:
            raise ValueError(
                f"Task '{self.spec.name}' has invalid head_layers={self.spec.head_layers}. "
                "head_layers must be >= 1."
            )

        if self.spec.readout_type in {"node", "node_set"} and self.spec.feature_source == "global":
            raise ValueError(
                f"Task '{self.spec.name}' uses readout_type='{self.spec.readout_type}' with "
                "feature_source='global'. This combination is invalid."
            )
        if self.spec.readout_type == "edge" and self.spec.feature_source == "global":
            raise ValueError(
                f"Task '{self.spec.name}' uses readout_type='edge' with feature_source='global'."
            )

    def _select_feature_bank(self, encoder_output: EncoderOutput) -> Tensor:
        if self.spec.feature_source == "hbar":
            return encoder_output.node_embeddings
        if self.spec.feature_source == "local":
            return encoder_output.local_node_embeddings
        if self.spec.feature_source == "global":
            return encoder_output.global_embedding
        raise ValueError(f"Unsupported feature_source '{self.spec.feature_source}'.")

    def _pool(self, x: Tensor, group_index: Tensor, num_groups: int) -> Tensor:
        if self.spec.pooling == "mean":
            return _scatter_mean(x, group_index, num_groups)
        if self.spec.pooling == "sum":
            return _scatter_sum(x, group_index, num_groups)
        if self.spec.pooling == "max":
            return _scatter_max(x, group_index, num_groups)
        if self.spec.pooling == "attention":
            if self.set_attention_pool is None:
                raise RuntimeError("set_attention_pool was not initialized for attention pooling.")
            return self.set_attention_pool(x, group_index, num_groups)
        raise ValueError(f"Unsupported pooling mode '{self.spec.pooling}'.")

    def _node_readout(
        self,
        encoder_output: EncoderOutput,
        task_input: Mapping[str, Any] | None,
    ) -> Tensor:
        if not task_input or "node_index" not in task_input:
            raise ValueError(
                f"Task '{self.spec.name}' expects task_inputs['{self.spec.name}']['node_index'] "
                "for readout_type='node'."
            )

        node_index = _to_long_tensor(
            task_input["node_index"],
            device=encoder_output.batch.device,
            name=f"{self.spec.name}.node_index",
        )

        feature_bank = self._select_feature_bank(encoder_output)
        if node_index.numel() > 0:
            if node_index.min().item() < 0 or node_index.max().item() >= feature_bank.size(0):
                raise IndexError(
                    f"Task '{self.spec.name}' node_index contains out-of-range values for "
                    f"{feature_bank.size(0)} nodes."
                )
        return feature_bank.index_select(0, node_index)

    def _node_set_readout(
        self,
        encoder_output: EncoderOutput,
        task_input: Mapping[str, Any] | None,
    ) -> Tensor:
        if not task_input:
            raise ValueError(
                f"Task '{self.spec.name}' expects node selection data for readout_type='node_set'."
            )

        feature_bank = self._select_feature_bank(encoder_output)

        if "node_mask" in task_input:
            node_mask = _to_bool_tensor(
                task_input["node_mask"],
                device=encoder_output.batch.device,
                name=f"{self.spec.name}.node_mask",
            )
            if node_mask.size(0) != feature_bank.size(0):
                raise ValueError(
                    f"Task '{self.spec.name}' node_mask length mismatch: expected "
                    f"{feature_bank.size(0)}, got {node_mask.size(0)}."
                )
            node_index = node_mask.nonzero(as_tuple=False).view(-1)
        elif "node_index" in task_input:
            node_index = _to_long_tensor(
                task_input["node_index"],
                device=encoder_output.batch.device,
                name=f"{self.spec.name}.node_index",
            )
        else:
            raise ValueError(
                f"Task '{self.spec.name}' needs either 'node_mask' or 'node_index' for node_set readout."
            )

        if node_index.numel() > 0:
            if node_index.min().item() < 0 or node_index.max().item() >= feature_bank.size(0):
                raise IndexError(
                    f"Task '{self.spec.name}' node_index contains out-of-range values for "
                    f"{feature_bank.size(0)} nodes."
                )

        if "selection_batch" in task_input:
            selection_batch = _to_long_tensor(
                task_input["selection_batch"],
                device=encoder_output.batch.device,
                name=f"{self.spec.name}.selection_batch",
            )
            if selection_batch.size(0) != node_index.size(0):
                raise ValueError(
                    f"Task '{self.spec.name}' selection_batch length mismatch: expected "
                    f"{node_index.size(0)}, got {selection_batch.size(0)}."
                )
        else:
            selection_batch = encoder_output.batch.index_select(0, node_index)

        if selection_batch.numel() > 0 and selection_batch.min().item() < 0:
            raise ValueError(
                f"Task '{self.spec.name}' selection_batch must be non-negative."
            )

        selection_batch, num_groups = _make_stable_contiguous_group_index(selection_batch)
        selected_features = feature_bank.index_select(0, node_index)
        return self._pool(selected_features, selection_batch, num_groups)

    def _graph_readout(self, encoder_output: EncoderOutput) -> Tensor:
        if self.spec.feature_source == "global":
            return encoder_output.global_embedding

        feature_bank = self._select_feature_bank(encoder_output)
        batch_index, num_groups = _make_stable_contiguous_group_index(encoder_output.batch)
        return self._pool(feature_bank, batch_index, num_groups)

    def _edge_readout(
        self,
        encoder_output: EncoderOutput,
        task_input: Mapping[str, Any] | None,
    ) -> Tensor:
        if not task_input or "edge_readout_index" not in task_input:
            raise ValueError(
                f"Task '{self.spec.name}' expects task_inputs['{self.spec.name}']['edge_readout_index'] "
                "for readout_type='edge'."
            )
        edge_readout_index = _to_long_tensor(
            task_input["edge_readout_index"],
            device=encoder_output.batch.device,
            name=f"{self.spec.name}.edge_readout_index",
        )
        num_edges = encoder_output.edge_index.size(1)
        if edge_readout_index.numel() > 0:
            if edge_readout_index.min().item() < 0 or edge_readout_index.max().item() >= num_edges:
                raise IndexError(
                    f"Task '{self.spec.name}' edge_readout_index contains out-of-range values for "
                    f"{num_edges} edges."
                )
        src_index = encoder_output.edge_index[0].index_select(0, edge_readout_index)
        dst_index = encoder_output.edge_index[1].index_select(0, edge_readout_index)
        feature_bank = self._select_feature_bank(encoder_output)
        features = [
            feature_bank.index_select(0, src_index),
            feature_bank.index_select(0, dst_index),
        ]
        if encoder_output.edge_embeddings is not None:
            features.append(encoder_output.edge_embeddings.index_select(0, edge_readout_index))
        return torch.cat(features, dim=-1)

    def forward(
        self,
        encoder_output: EncoderOutput,
        task_input: Mapping[str, Any] | None = None,
    ) -> Tensor:
        if self.spec.readout_type == "node":
            features = self._node_readout(encoder_output, task_input)
        elif self.spec.readout_type == "node_set":
            features = self._node_set_readout(encoder_output, task_input)
        elif self.spec.readout_type == "graph":
            features = self._graph_readout(encoder_output)
        elif self.spec.readout_type == "edge":
            features = self._edge_readout(encoder_output, task_input)
        else:
            raise ValueError(f"Unsupported readout_type '{self.spec.readout_type}'.")
        return self.decoder(features)