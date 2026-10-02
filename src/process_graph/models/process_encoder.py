from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn

from ..constants import HX_ROLE_TO_IDX, OPER_FEATURE_SLOTS, ROLE_TO_IDX, STREAM_EDGE_FEATURE_SLOTS, STREAM_ROLE_TO_IDX, UNIT_TO_IDX
from ..schema import GraphSample

ActivationName = Literal["relu", "gelu", "silu", "tanh", "leaky_relu"]
EdgeHeadType = Literal[
    "single",
    "grouped_property",
    "pi_grouped_property",
    "hierarchical_reduced_pi",
]
DiffMode = Literal["add", "concat"]
FusionMode = Literal["concat", "sum", "mean", "weighted"]
GlobalPoolMode = Literal["mean", "sum", "max", "attention", "concat_set2set"]
OperMaskMode = Literal["ignore", "concat", "gated"]
InitialResidualMode = Literal["none", "final", "per_layer", "both"]
LayerResidualMode = Literal["none", "interp"]
FlowGNNArchitecture = Literal["legacy", "relational_bidirectional"]
EdgeReadoutMode = Literal[
    "legacy",
    "bidirectional_separate",
    "bidirectional_fused",
]


@dataclass
class ProcessEncoderConfig:
    """Configuration for the directed process graph encoder."""

    hidden_dim: int = 256
    num_layers: int = 3
    role_emb_dim: int = 16
    unit_emb_dim: int = 16
    hx_role_emb_dim: int = 8
    oper_dim: int = len(OPER_FEATURE_SLOTS)
    input_mlp_layers: int = 2
    diff_mlp_layers: int = 2
    update_mlp_layers: int = 2
    final_mlp_layers: int = 2
    attn_hidden_dim: int = 64
    diff_mode: DiffMode = "add"
    fusion_mode: FusionMode = "concat"
    global_pool: GlobalPoolMode = "mean"
    set2set_processing_steps: int = 3
    dropout: float = 0.0
    activation: ActivationName = "relu"
    use_role_embedding: bool = False
    use_unit_embedding: bool = True
    use_hx_role_embedding: bool = False
    oper_mask_mode: OperMaskMode = "ignore"
    input_residual: bool = True
    initial_residual_mode: InitialResidualMode = "final"
    initial_residual_alpha: float = 0.2
    layer_residual_mode: LayerResidualMode = "none"
    layer_residual_alpha: float = 0.5
    layer_residual_norm: bool = False
    use_final_projection: bool = True
    use_edge_features: bool = False
    edge_stream_role_emb_dim: int = 8
    edge_stream_id_emb_dim: int = 16
    edge_stream_vocab_size: int = 512
    use_edge_stream_id: bool = True
    force_unknown_edge_stream_id: bool = False
    operating_hidden_dim: int = 0
    edge_structural_hidden_dim: int = 0
    edge_hidden_dim: int = 0
    attention_num_heads: int = 1
    edge_mlp_layers: int = 2
    edge_oper_dim: int = 5
    use_edge_stream_head: bool = False
    edge_stream_out_dim: int = 12
    edge_stream_mlp_hidden_dim: int = 128
    edge_stream_mlp_layers: int = 2
    use_edge_decoder: bool = False
    edge_decoder_hidden_dim: int = 128
    edge_decoder_dropout: float = 0.0
    stream_target_dim: int = 12
    edge_struct_dim: int = 5
    edge_head_type: EdgeHeadType = "single"
    edge_head_dropout: float = 0.1
    edge_head_shared_dims: Sequence[int] = field(default_factory=lambda: (512, 512))
    edge_head_frac_dims: Sequence[int] = field(default_factory=lambda: (256, 128))
    edge_head_flow_dims: Sequence[int] = field(default_factory=lambda: (512, 256, 128))
    edge_head_cond_dims: Sequence[int] = field(default_factory=lambda: (256, 128))
    cond_head_output_dim: int = 2
    frac_head_output_dim: int = 7
    mass_head_output_dim: int = 3
    property_head_hidden_dim: int = 128
    property_head_num_layers: int = 2
    fraction_activation: str = "softmax"
    fraction_temperature: float = 1.0
    species_order: Sequence[str] = field(default_factory=tuple)
    target_branch_hidden_adapter_enabled: bool = False
    target_branch_hidden_adapter_bottleneck_dim: int = 16
    target_branch_hidden_adapter_scale: float = 0.1
    target_branch_hidden_adapter_detach_input: bool = True
    target_branch_hidden_adapter_zero_init_output: bool = True
    target_branch_hidden_adapter_apply_to: Sequence[str] = field(default_factory=tuple)
    property_stream_role_enabled: bool = False
    property_stream_role_embedding_dim: int = 16
    property_stream_role_unknown_role_id: int = 0
    hierarchical_global_projection: bool = True
    hierarchical_global_dim: int = 0
    hierarchical_level2_hidden_dim: int = 64
    hierarchical_decoder_intermediate_dim: int = 0
    hierarchical_condition_hidden_dim: int = 0
    hierarchical_fraction_hidden_dim: int = 0
    hierarchical_mass_hidden_dim: int = 0
    hierarchical_shared_residual: bool = True
    hierarchical_detach_level1_predictions: bool = True
    hierarchical_level2_unfreeze_epoch: int | None = None
    hierarchical_predict_volume_flow: bool = True
    flow_gnn_architecture: FlowGNNArchitecture = "legacy"
    relational_message_hidden_dim: int = 512
    relational_attention_hidden_dim: int = 256
    relational_update_hidden_dim: int = 512
    relational_fusion_hidden_dim: int = 512
    relational_dropout: float = 0.1
    edge_readout_mode: EdgeReadoutMode = "legacy"
    edge_readout_hidden_dim: int = 512
    edge_readout_dropout: float = 0.1
    edge_readout_fusion_hidden_dim: int = 512
    edge_readout_fusion_output_dim: int = 384
    edge_readout_fusion_dropout: float = 0.1
    property_prediction_edges_only: bool = False
    hx_pair_relation_enabled: bool = False
    hx_pair_side_embedding_dim: int = 16
    hx_pair_relation_hidden_dim: int = 256
    hx_pair_relation_output_dim: int = 384
    hx_pair_relation_dropout: float = 0.1
    hx_pair_gate_init: float = 0.05
    feed_head_conditioning_enabled: bool = False
    feed_head_input_dim: int = 6
    feed_head_hidden_dim: int = 32
    feed_head_output_dim: int = 64
    feed_head_dropout: float = 0.0


@dataclass
class EncoderOutput:
    """Container returned by `ProcessGraphEncoder`."""

    initial_node_embeddings: Tensor
    local_node_embeddings: Tensor
    global_embedding: Tensor
    node_embeddings: Tensor
    edge_embeddings: Tensor | None
    batch: Tensor
    edge_index: Tensor
    layer_node_embeddings: Sequence[Tensor]


def resolve_activation(name: ActivationName) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(
        f"Unsupported activation '{name}'. Expected one of: relu, gelu, silu, tanh, leaky_relu."
    )


class MLP(nn.Module):
    """Plain feed-forward stack used across encoder and decoder blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: ActivationName = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"MLP num_layers must be >= 1, got {num_layers}.")

        layers: list[nn.Module] = []
        in_dim = input_dim
        act = resolve_activation(activation)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act.__class__())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def _scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Sum values grouped by `index`."""
    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    if src.numel() == 0:
        return out
    out.index_add_(0, index, src)
    return out


def _scatter_mean(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Mean aggregation grouped by `index`."""
    if src.numel() == 0:
        return torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    summed = _scatter_sum(src, index, dim_size)
    counts = torch.bincount(index, minlength=dim_size).to(src.device, src.dtype).unsqueeze(-1)
    return summed / counts.clamp_min(1.0)


def _scatter_max(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Max aggregation grouped by `index`."""
    out = torch.full(
        (dim_size, src.size(-1)),
        fill_value=torch.finfo(src.dtype).min,
        device=src.device,
        dtype=src.dtype,
    )
    if src.numel() == 0:
        return out
    expanded_index = index.unsqueeze(-1).expand_as(src)
    out.scatter_reduce_(0, expanded_index, src, reduce="amax", include_self=True)
    out[out == torch.finfo(src.dtype).min] = 0.0
    return out


def _segment_softmax(logits: Tensor, index: Tensor, num_segments: int) -> Tensor:
    """Softmax over variable-sized groups defined by `index`."""
    if logits.ndim != 1:
        raise ValueError(
            f"Segment softmax expects 1D logits, got shape {tuple(logits.shape)}."
        )
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
    # AMP/autocast can produce float32 intermediates while logits/denom stay float16.
    # index_add_ requires exact dtype match between destination and source.
    if exp_logits.dtype != logits.dtype:
        exp_logits = exp_logits.to(dtype=logits.dtype)
    denom = torch.zeros(num_segments, device=logits.device, dtype=logits.dtype)
    denom.index_add_(0, index, exp_logits)
    return exp_logits / denom.index_select(0, index).clamp_min(1e-12)


def _as_long_tensor(value: Any, device: torch.device, name: str) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device=device, dtype=torch.long)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a 1D tensor, got shape {tuple(tensor.shape)}.")
    return tensor


def _as_float_tensor(value: Any, device: torch.device, name: str) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device=device, dtype=torch.float32)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2D tensor, got shape {tuple(tensor.shape)}.")
    return tensor


def _as_edge_index(value: Any, device: torch.device) -> Tensor:
    tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device=device, dtype=torch.long)
    if tensor.ndim != 2 or tensor.size(0) != 2:
        raise ValueError(
            f"edge_index must have shape [2, num_edges], got {tuple(tensor.shape)}."
        )
    return tensor


def _infer_num_graphs(batch: Tensor) -> int:
    if batch.numel() == 0:
        return 0
    return int(batch.max().item()) + 1


def _make_stable_contiguous_group_index(index: Tensor) -> tuple[Tensor, int]:
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


def _make_stable_contiguous_index(index: Tensor) -> Tensor:
    return _make_stable_contiguous_group_index(index)[0]


def _get_field(data: Any, field: str) -> Any:
    if isinstance(data, Mapping):
        return data.get(field)
    return getattr(data, field, None)


class ProcessInputEncoder(nn.Module):
    """Initial node encoder."""

    def __init__(self, config: ProcessEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.role_embedding = (
            nn.Embedding(len(ROLE_TO_IDX), config.role_emb_dim)
            if config.use_role_embedding
            else None
        )
        self.unit_embedding = (
            nn.Embedding(len(UNIT_TO_IDX), config.unit_emb_dim)
            if config.use_unit_embedding
            else None
        )
        self.hx_role_embedding = (
            nn.Embedding(len(HX_ROLE_TO_IDX), config.hx_role_emb_dim)
            if config.use_hx_role_embedding
            else None
        )

        oper_input_dim = config.oper_dim
        if config.oper_mask_mode == "concat":
            oper_input_dim += config.oper_dim
        self.operating_encoder: nn.Module | None = None
        if int(config.operating_hidden_dim) > 0:
            self.operating_encoder = nn.Sequential(
                nn.Linear(oper_input_dim, int(config.operating_hidden_dim)),
                resolve_activation(config.activation),
                nn.LayerNorm(int(config.operating_hidden_dim)),
            )
            oper_feature_dim = int(config.operating_hidden_dim)
        else:
            oper_feature_dim = oper_input_dim

        input_dim = oper_feature_dim
        if config.use_role_embedding:
            input_dim += config.role_emb_dim
        if config.use_unit_embedding:
            input_dim += config.unit_emb_dim
        if config.use_hx_role_embedding:
            input_dim += config.hx_role_emb_dim

        self.input_mlp = MLP(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_layers=config.input_mlp_layers,
            activation=config.activation,
            dropout=config.dropout,
        )

    def forward(
        self,
        x_role: Tensor | None,
        x_oper: Tensor,
        x_unit: Tensor | None = None,
        x_hx_role: Tensor | None = None,
        x_oper_mask: Tensor | None = None,
    ) -> Tensor:
        if x_oper.size(-1) != self.config.oper_dim:
            raise ValueError(
                f"x_oper feature dimension mismatch: expected {self.config.oper_dim}, "
                f"got {x_oper.size(-1)}."
            )

        features: list[Tensor] = []

        if self.config.use_role_embedding:
            if x_role is None:
                raise ValueError(
                    "x_role is required when use_role_embedding=True, but it was not provided."
                )
            if self.role_embedding is None:
                raise RuntimeError("role_embedding was not created although use_role_embedding=True.")
            features.append(self.role_embedding(x_role))

        if self.config.use_unit_embedding:
            if x_unit is None:
                raise ValueError(
                    "x_unit is required when use_unit_embedding=True, but it was not provided."
                )
            if self.unit_embedding is None:
                raise RuntimeError("unit_embedding was not created although use_unit_embedding=True.")
            features.append(self.unit_embedding(x_unit))

        if self.config.use_hx_role_embedding:
            if x_hx_role is None:
                raise ValueError(
                    "x_hx_role is required when use_hx_role_embedding=True, but it was not provided."
                )
            if self.hx_role_embedding is None:
                raise RuntimeError(
                    "hx_role_embedding was not created although use_hx_role_embedding=True."
                )
            features.append(self.hx_role_embedding(x_hx_role))

        if self.config.oper_mask_mode == "ignore":
            oper_features = x_oper
        elif self.config.oper_mask_mode == "concat":
            if x_oper_mask is None:
                raise ValueError(
                    "x_oper_mask is required when oper_mask_mode='concat', but it was not provided."
                )
            oper_features = torch.cat([x_oper, x_oper_mask.to(x_oper.dtype)], dim=-1)
        elif self.config.oper_mask_mode == "gated":
            if x_oper_mask is None:
                raise ValueError(
                    "x_oper_mask is required when oper_mask_mode='gated', but it was not provided."
                )
            oper_features = x_oper * x_oper_mask.to(x_oper.dtype)
        else:
            raise ValueError(f"Unsupported oper_mask_mode '{self.config.oper_mask_mode}'.")

        if self.operating_encoder is not None:
            oper_features = self.operating_encoder(oper_features)
        features.append(oper_features)
        return self.input_mlp(torch.cat(features, dim=-1))


class ProcessEdgeInputEncoder(nn.Module):
    """Initial stream-edge encoder for edge-aware process graphs."""

    def __init__(self, config: ProcessEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.role_embedding = nn.Embedding(len(STREAM_ROLE_TO_IDX), config.edge_stream_role_emb_dim)
        self.stream_embedding = (
            nn.Embedding(config.edge_stream_vocab_size, config.edge_stream_id_emb_dim)
            if config.use_edge_stream_id
            else None
        )
        self.structural_encoder: nn.Module | None = None
        if int(config.edge_structural_hidden_dim) > 0:
            self.structural_encoder = nn.Sequential(
                nn.Linear(config.edge_oper_dim, int(config.edge_structural_hidden_dim)),
                resolve_activation(config.activation),
                nn.LayerNorm(int(config.edge_structural_hidden_dim)),
            )
            structural_dim = int(config.edge_structural_hidden_dim)
        else:
            structural_dim = int(config.edge_oper_dim)
        stream_dim = int(config.edge_stream_id_emb_dim) if config.use_edge_stream_id else 0
        input_dim = int(config.edge_stream_role_emb_dim) + stream_dim + structural_dim
        output_dim = (
            int(config.edge_hidden_dim)
            if int(config.edge_hidden_dim) > 0
            else int(config.hidden_dim)
        )
        self.input_mlp = MLP(
            input_dim=input_dim,
            hidden_dim=output_dim,
            output_dim=output_dim,
            num_layers=config.edge_mlp_layers,
            activation=config.activation,
            dropout=config.dropout,
        )

    def forward(
        self,
        edge_stream_role: Tensor,
        edge_stream_id: Tensor,
        edge_oper: Tensor,
        edge_oper_mask: Tensor | None = None,
    ) -> Tensor:
        if edge_oper.size(-1) != self.config.edge_oper_dim:
            raise ValueError(
                f"edge_oper feature dimension mismatch: expected {self.config.edge_oper_dim}, "
                f"got {edge_oper.size(-1)}."
            )
        if edge_oper_mask is not None:
            edge_oper = edge_oper * edge_oper_mask.to(edge_oper.dtype)
        if self.structural_encoder is not None:
            edge_oper = self.structural_encoder(edge_oper)
        features = [self.role_embedding(edge_stream_role)]
        if self.stream_embedding is not None:
            if self.config.force_unknown_edge_stream_id:
                edge_stream_id = torch.zeros_like(edge_stream_id)
            edge_stream_id = torch.remainder(
                edge_stream_id, self.config.edge_stream_vocab_size
            )
            features.append(self.stream_embedding(edge_stream_id))
        features.append(edge_oper)
        return self.input_mlp(torch.cat(features, dim=-1))


class DirectedAttentionAggregation(nn.Module):
    """Single-branch directed attention aggregator."""

    def __init__(
        self,
        hidden_dim: int,
        attn_hidden_dim: int,
        dropout: float,
        activation: ActivationName,
        edge_hidden_dim: int | None = None,
    ):
        super().__init__()
        edge_hidden_dim = int(edge_hidden_dim or hidden_dim)
        self.src_proj = nn.Linear(hidden_dim, attn_hidden_dim)
        self.dst_proj = nn.Linear(hidden_dim, attn_hidden_dim)
        self.score_proj = nn.Linear(attn_hidden_dim, 1, bias=False)
        self.edge_proj = nn.Linear(edge_hidden_dim, attn_hidden_dim)
        self.edge_message_proj: nn.Module = (
            nn.Identity()
            if edge_hidden_dim == int(hidden_dim)
            else nn.Linear(edge_hidden_dim, hidden_dim, bias=False)
        )
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.activation = resolve_activation(activation)
        self.attn_dropout_p = dropout

    def _apply_attention_dropout(
        self,
        alpha: Tensor,
        normalize_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """
        Drop attention weights while preserving segment-wise normalization.
        If an entire segment is dropped, fall back to the original alpha.
        """
        if not self.training or self.attn_dropout_p <= 0.0 or alpha.numel() == 0:
            return alpha

        keep_mask = (torch.rand_like(alpha) >= self.attn_dropout_p).to(alpha.dtype)
        dropped_alpha = alpha * keep_mask

        denom = torch.zeros(num_nodes, device=alpha.device, dtype=alpha.dtype)
        denom.index_add_(0, normalize_index, dropped_alpha)
        denom_per_edge = denom.index_select(0, normalize_index)

        has_survivor = denom_per_edge > 0
        renorm_alpha = torch.where(
            has_survivor,
            dropped_alpha / denom_per_edge.clamp_min(1e-12),
            alpha,
        )
        return renorm_alpha

    def forward(
        self,
        h: Tensor,
        sender_index: Tensor,
        receiver_index: Tensor,
        normalize_index: Tensor,
        edge_embeddings: Tensor | None = None,
    ) -> Tensor:
        num_nodes = h.size(0)
        if sender_index.numel() == 0:
            return torch.zeros_like(h)

        h_sender = h.index_select(0, sender_index)
        h_receiver = h.index_select(0, receiver_index)
        score_features = self.src_proj(h_sender) + self.dst_proj(h_receiver)
        if edge_embeddings is not None:
            score_features = score_features + self.edge_proj(edge_embeddings)
        logits = self.score_proj(self.activation(score_features)).squeeze(-1)

        alpha = _segment_softmax(logits, normalize_index, num_segments=num_nodes)
        alpha = self._apply_attention_dropout(alpha, normalize_index, num_nodes)
        message_source = (
            h_sender
            if edge_embeddings is None
            else h_sender + self.edge_message_proj(edge_embeddings)
        )
        messages = self.msg_proj(message_source) * alpha.unsqueeze(-1)
        return _scatter_sum(messages, receiver_index, dim_size=num_nodes)


class DifferentialUpdateBlock(nn.Module):
    """Differential encoding plus plain update MLP."""

    def __init__(self, config: ProcessEncoderConfig) -> None:
        super().__init__()
        self.diff_mode = config.diff_mode
        self.diff_encoder = MLP(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_layers=config.diff_mlp_layers,
            activation=config.activation,
            dropout=config.dropout,
        )
        update_input_dim = config.hidden_dim if config.diff_mode == "add" else config.hidden_dim * 2
        self.update = MLP(
            input_dim=update_input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            num_layers=config.update_mlp_layers,
            activation=config.activation,
            dropout=config.dropout,
        )

    def forward(self, agg: Tensor, h: Tensor) -> Tensor:
        delta = agg - h
        encoded_delta = self.diff_encoder(delta)

        if self.diff_mode == "add":
            z = agg + encoded_delta
        elif self.diff_mode == "concat":
            z = torch.cat([agg, encoded_delta], dim=-1)
        else:
            raise ValueError(f"Unsupported diff_mode '{self.diff_mode}'.")

        return self.update(z)


class LocalFusion(nn.Module):
    """Merge forward and backward node states into a local node representation."""

    def __init__(self, hidden_dim: int, mode: FusionMode) -> None:
        super().__init__()
        self.mode = mode
        self.concat_projection = nn.Linear(hidden_dim * 2, hidden_dim) if mode == "concat" else None
        self.weight_gate = nn.Linear(hidden_dim * 2, 1) if mode == "weighted" else None

    def forward(self, h_forward: Tensor, h_backward: Tensor) -> Tensor:
        if self.mode == "concat":
            if self.concat_projection is None:
                raise RuntimeError("concat_projection was not initialized for concat fusion.")
            return self.concat_projection(torch.cat([h_forward, h_backward], dim=-1))
        if self.mode == "sum":
            return h_forward + h_backward
        if self.mode == "mean":
            return 0.5 * (h_forward + h_backward)
        if self.mode == "weighted":
            if self.weight_gate is None:
                raise RuntimeError("weight_gate was not initialized for weighted fusion.")
            gate = torch.sigmoid(self.weight_gate(torch.cat([h_forward, h_backward], dim=-1)))
            return gate * h_forward + (1.0 - gate) * h_backward
        raise ValueError(f"Unsupported fusion_mode '{self.mode}'.")


def global_embedding_dim(config: ProcessEncoderConfig) -> int:
    """Graph-level readout width (Set2Set concat readout is 2× hidden_dim)."""
    if config.global_pool == "concat_set2set":
        return int(config.hidden_dim) * 2
    return int(config.hidden_dim)


class AttentionGlobalPool(nn.Module):
    """Attention pooling over the last local node embeddings only."""

    def __init__(self, hidden_dim: int, activation: ActivationName, dropout: float) -> None:
        super().__init__()
        self.scorer = MLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=2,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, x: Tensor, batch: Tensor, num_graphs: int) -> Tensor:
        scores = self.scorer(x).squeeze(-1)
        weights = _segment_softmax(scores, batch, num_segments=num_graphs)
        return _scatter_sum(x * weights.unsqueeze(-1), batch, dim_size=num_graphs)


class Set2SetGlobalPool(nn.Module):
    """
    Set2Set readout (Vinyals et al.): iterative LSTM query + attention over nodes.
    Output per graph is concat(query, readout) with dim = 2 * hidden_dim.
    """

    def __init__(self, hidden_dim: int, processing_steps: int = 3) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(hidden_dim) * 2
        self.processing_steps = max(1, int(processing_steps))
        self.lstm = nn.LSTM(self.output_dim, self.hidden_dim)

    def _set2set_single(self, x_g: Tensor) -> Tensor:
        device = x_g.device
        hx = (
            torch.zeros(1, 1, self.hidden_dim, device=device),
            torch.zeros(1, 1, self.hidden_dim, device=device),
        )
        q_star = torch.zeros(1, 1, self.output_dim, device=device)
        for _ in range(self.processing_steps):
            _, (h_n, c_n) = self.lstm(q_star, hx)
            hx = (h_n, c_n)
            q = h_n.squeeze(0).squeeze(0)
            e = (x_g * q.unsqueeze(0)).sum(dim=-1)
            a = torch.softmax(e, dim=0)
            r = torch.sum(a.unsqueeze(-1) * x_g, dim=0)
            q_star = torch.cat([q, r], dim=-1).view(1, 1, -1)
        return q_star.view(-1)

    def forward(self, x: Tensor, batch: Tensor, num_graphs: int) -> Tensor:
        batch = batch.long()
        rows: list[Tensor] = []
        for g in range(int(num_graphs)):
            mask = batch == g
            if not bool(mask.any()):
                rows.append(x.new_zeros(self.output_dim))
                continue
            rows.append(self._set2set_single(x[mask]))
        return torch.stack(rows, dim=0)


class ProcessGraphEncoderLayer(nn.Module):
    """One directed encoder layer with forward/backward branches."""

    def __init__(self, config: ProcessEncoderConfig) -> None:
        super().__init__()
        self.forward_attention = DirectedAttentionAggregation(
            hidden_dim=config.hidden_dim,
            attn_hidden_dim=config.attn_hidden_dim,
            dropout=config.dropout,
            activation=config.activation,
            edge_hidden_dim=(
                config.edge_hidden_dim
                if config.edge_hidden_dim > 0
                else config.hidden_dim
            ),
        )
        self.backward_attention = DirectedAttentionAggregation(
            hidden_dim=config.hidden_dim,
            attn_hidden_dim=config.attn_hidden_dim,
            dropout=config.dropout,
            activation=config.activation,
            edge_hidden_dim=(
                config.edge_hidden_dim
                if config.edge_hidden_dim > 0
                else config.hidden_dim
            ),
        )
        self.forward_update = DifferentialUpdateBlock(config)
        self.backward_update = DifferentialUpdateBlock(config)
        self.fusion = LocalFusion(hidden_dim=config.hidden_dim, mode=config.fusion_mode)

    def forward(self, h: Tensor, edge_index: Tensor, edge_embeddings: Tensor | None = None) -> Tensor:
        src_index, dst_index = edge_index

        forward_kwargs = {
            "h": h,
            "sender_index": src_index,
            "receiver_index": dst_index,
            "normalize_index": src_index,
        }
        if edge_embeddings is not None:
            forward_kwargs["edge_embeddings"] = edge_embeddings
        agg_forward = self.forward_attention(**forward_kwargs)

        backward_kwargs = {
            "h": h,
            "sender_index": dst_index,
            "receiver_index": src_index,
            "normalize_index": src_index,
        }
        if edge_embeddings is not None:
            backward_kwargs["edge_embeddings"] = edge_embeddings
        agg_backward = self.backward_attention(**backward_kwargs)

        h_forward = self.forward_update(agg_forward, h)
        h_backward = self.backward_update(agg_backward, h)
        return self.fusion(h_forward, h_backward)


class RelationalDifferentialUpdateBlock(nn.Module):
    """Update a relational node state using the aggregate-state difference."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        update_hidden_dim: int,
        dropout: float,
        diff_mode: DiffMode,
    ) -> None:
        super().__init__()
        self.diff_mode = diff_mode
        self.diff_encoder = nn.Sequential(
            nn.Linear(hidden_dim, update_hidden_dim),
            nn.LayerNorm(update_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(update_hidden_dim, hidden_dim),
        )
        update_input_dim = hidden_dim if diff_mode == "add" else 2 * hidden_dim
        self.update = nn.Sequential(
            nn.Linear(update_input_dim, update_hidden_dim),
            nn.LayerNorm(update_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(update_hidden_dim, hidden_dim),
        )

    def forward(
        self,
        aggregate: Tensor,
        state: Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        differential = aggregate - state
        encoded_differential = self.diff_encoder(differential)
        if self.diff_mode == "add":
            update_input = aggregate + encoded_differential
        elif self.diff_mode == "concat":
            update_input = torch.cat([aggregate, encoded_differential], dim=-1)
        else:
            raise ValueError(f"Unsupported diff_mode '{self.diff_mode}'.")
        output = self.update(update_input)
        if return_diagnostics:
            return output, differential, encoded_differential
        return output


class RelationalBidirectionalFlowGNNLayer(nn.Module):
    """Edge-conditioned bidirectional layer with physical-direction softmax groups."""

    def __init__(self, config: ProcessEncoderConfig) -> None:
        super().__init__()
        hidden_dim = int(config.hidden_dim)
        edge_dim = int(config.edge_hidden_dim or config.hidden_dim)
        message_hidden = int(config.relational_message_hidden_dim)
        attention_hidden = int(config.relational_attention_hidden_dim)
        update_hidden = int(config.relational_update_hidden_dim)
        fusion_hidden = int(config.relational_fusion_hidden_dim)
        dropout = float(config.relational_dropout)

        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.forward_message = self._message_mlp(
            hidden_dim + edge_dim, message_hidden, hidden_dim, dropout
        )
        self.backward_message = self._message_mlp(
            hidden_dim + edge_dim, message_hidden, hidden_dim, dropout
        )
        self.forward_attention = self._attention_mlp(
            (2 * hidden_dim) + edge_dim, attention_hidden
        )
        self.backward_attention = self._attention_mlp(
            (2 * hidden_dim) + edge_dim, attention_hidden
        )
        self.forward_update = RelationalDifferentialUpdateBlock(
            hidden_dim=hidden_dim,
            update_hidden_dim=update_hidden,
            dropout=dropout,
            diff_mode=config.diff_mode,
        )
        self.backward_update = RelationalDifferentialUpdateBlock(
            hidden_dim=hidden_dim,
            update_hidden_dim=update_hidden,
            dropout=dropout,
            diff_mode=config.diff_mode,
        )
        self.fusion = self._update_mlp(
            2 * hidden_dim, fusion_hidden, hidden_dim, dropout
        )

    @staticmethod
    def _message_mlp(
        input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    @staticmethod
    def _attention_mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _update_mlp(
        input_dim: int, hidden_dim: int, output_dim: int, dropout: float
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def _validate_inputs(
        self, h: Tensor, edge_index: Tensor, edge_embeddings: Tensor | None
    ) -> Tensor:
        if edge_embeddings is None:
            raise ValueError(
                "relational_bidirectional FlowGNN requires static edge embeddings."
            )
        num_edges = int(edge_index.shape[1])
        expected = (num_edges, self.edge_dim)
        if edge_embeddings.ndim != 2 or tuple(edge_embeddings.shape) != expected:
            raise ValueError(
                "relational edge embeddings must have shape "
                f"{expected}, got {tuple(edge_embeddings.shape)}."
            )
        return edge_embeddings.to(device=h.device, dtype=h.dtype)

    def _forward_impl(
        self, h: Tensor, edge_index: Tensor, edge_embeddings: Tensor | None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        edge_embeddings = self._validate_inputs(h, edge_index, edge_embeddings)
        src_index, dst_index = edge_index
        num_nodes = int(h.shape[0])
        h_src = h.index_select(0, src_index)
        h_dst = h.index_select(0, dst_index)

        forward_messages = self.forward_message(
            torch.cat([h_src, edge_embeddings], dim=-1)
        )
        forward_logits = self.forward_attention(
            torch.cat([h_src, h_dst, edge_embeddings], dim=-1)
        ).squeeze(-1)
        # Physical-flow edges compete at their sender/source node.  Messages
        # still travel src -> dst; only the softmax normalization group is
        # changed from receiver-wise (standard GAT) to sender-wise (FlowGNN).
        forward_alpha = _segment_softmax(
            forward_logits, src_index, num_segments=num_nodes
        )
        forward_agg = _scatter_sum(
            forward_messages * forward_alpha.unsqueeze(-1),
            dst_index,
            dim_size=num_nodes,
        )

        backward_messages = self.backward_message(
            torch.cat([h_dst, edge_embeddings], dim=-1)
        )
        backward_logits = self.backward_attention(
            torch.cat([h_dst, h_src, edge_embeddings], dim=-1)
        ).squeeze(-1)
        # Reverse messages travel physical dst -> src and use standard
        # receiver-wise normalization.  That reverse receiver is src_index in
        # the original edge coordinates, so this is intentionally src_index.
        backward_alpha = _segment_softmax(
            backward_logits, src_index, num_segments=num_nodes
        )
        backward_agg = _scatter_sum(
            backward_messages * backward_alpha.unsqueeze(-1),
            src_index,
            dim_size=num_nodes,
        )

        h_forward, forward_delta, forward_encoded_delta = self.forward_update(
            forward_agg, h, return_diagnostics=True
        )
        h_backward, backward_delta, backward_encoded_delta = self.backward_update(
            backward_agg, h, return_diagnostics=True
        )
        output = self.fusion(torch.cat([h_forward, h_backward], dim=-1))
        diagnostics = {
            "forward_messages": forward_messages,
            "backward_messages": backward_messages,
            "forward_attention_logits": forward_logits,
            "backward_attention_logits": backward_logits,
            "forward_attention": forward_alpha,
            "backward_attention": backward_alpha,
            "forward_aggregate": forward_agg,
            "backward_aggregate": backward_agg,
            "forward_differential": forward_delta,
            "backward_differential": backward_delta,
            "forward_encoded_differential": forward_encoded_delta,
            "backward_encoded_differential": backward_encoded_delta,
            "forward_updated_nodes": h_forward,
            "backward_updated_nodes": h_backward,
        }
        return output, diagnostics

    def forward(
        self, h: Tensor, edge_index: Tensor, edge_embeddings: Tensor | None = None
    ) -> Tensor:
        return self._forward_impl(h, edge_index, edge_embeddings)[0]

    def forward_with_diagnostics(
        self, h: Tensor, edge_index: Tensor, edge_embeddings: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Return attention/aggregate tensors for focused tests and diagnostics."""
        return self._forward_impl(h, edge_index, edge_embeddings)


class ProcessGraphEncoder(nn.Module):
    """Reusable encoder for directed process graphs."""

    def __init__(self, config: ProcessEncoderConfig) -> None:
        super().__init__()
        self.config = config
        if int(self.config.attention_num_heads) != 1:
            raise ValueError(
                "FlowGNN currently implements one scalar attention head per "
                "direction; attention_num_heads must be 1."
            )
        if int(self.config.edge_hidden_dim) < 0:
            raise ValueError("edge_hidden_dim must be >= 0.")
        if int(self.config.operating_hidden_dim) < 0:
            raise ValueError("operating_hidden_dim must be >= 0.")
        if int(self.config.edge_structural_hidden_dim) < 0:
            raise ValueError("edge_structural_hidden_dim must be >= 0.")
        if not (0.0 <= self.config.initial_residual_alpha <= 1.0):
            raise ValueError(
                "initial_residual_alpha must be in [0, 1], "
                f"got {self.config.initial_residual_alpha}."
            )
        if not (0.0 <= self.config.layer_residual_alpha <= 1.0):
            raise ValueError(
                "layer_residual_alpha must be in [0, 1], "
                f"got {self.config.layer_residual_alpha}."
            )
        if self.config.flow_gnn_architecture not in {
            "legacy",
            "relational_bidirectional",
        }:
            raise ValueError(
                "flow_gnn_architecture must be 'legacy' or "
                f"'relational_bidirectional', got {self.config.flow_gnn_architecture!r}."
            )
        self.input_encoder = ProcessInputEncoder(config)
        self.edge_input_encoder = ProcessEdgeInputEncoder(config) if config.use_edge_features else None
        layer_cls: type[nn.Module] = (
            ProcessGraphEncoderLayer
            if self.config.flow_gnn_architecture == "legacy"
            else RelationalBidirectionalFlowGNNLayer
        )
        self.layers = nn.ModuleList(layer_cls(config) for _ in range(config.num_layers))
        self.layer_residual_norms = (
            nn.ModuleList(nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers))
            if config.layer_residual_norm
            else None
        )
        self.attention_pool = (
            AttentionGlobalPool(config.hidden_dim, config.activation, config.dropout)
            if config.global_pool == "attention"
            else None
        )
        self.set2set_pool = (
            Set2SetGlobalPool(config.hidden_dim, processing_steps=config.set2set_processing_steps)
            if config.global_pool == "concat_set2set"
            else None
        )
        _ge_dim = global_embedding_dim(config)
        self.final_projection = (
            MLP(
                input_dim=config.hidden_dim + _ge_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.hidden_dim,
                num_layers=config.final_mlp_layers,
                activation=config.activation,
                dropout=config.dropout,
            )
            if config.use_final_projection
            else None
        )

    @property
    def hidden_dim(self) -> int:
        return self.config.hidden_dim

    @property
    def _use_per_layer_initial_residual(self) -> bool:
        return self.config.initial_residual_mode in {"per_layer", "both"}

    @property
    def _use_final_initial_residual(self) -> bool:
        return self.config.initial_residual_mode in {"final", "both"}

    def _coerce_inputs(
        self,
        data: GraphSample | Mapping[str, Any] | Any | None,
        edge_index: Any,
        x_role: Any,
        x_unit: Any,
        x_hx_role: Any,
        x_oper: Any,
        x_oper_mask: Any,
        edge_stream_role: Any,
        edge_stream_id: Any,
        edge_oper: Any,
        edge_oper_mask: Any,
        batch: Any,
    ) -> tuple[
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor | None,
        Tensor,
    ]:
        device = next(self.parameters()).device

        source = data if data is not None else {}
        edge_index = edge_index if edge_index is not None else _get_field(source, "edge_index")
        x_role = x_role if x_role is not None else _get_field(source, "x_role")
        x_unit = x_unit if x_unit is not None else _get_field(source, "x_unit")
        x_hx_role = x_hx_role if x_hx_role is not None else _get_field(source, "x_hx_role")
        x_oper = x_oper if x_oper is not None else _get_field(source, "x_oper")
        x_oper_mask = x_oper_mask if x_oper_mask is not None else _get_field(source, "x_oper_mask")
        edge_stream_role = (
            edge_stream_role if edge_stream_role is not None else _get_field(source, "edge_stream_role")
        )
        edge_stream_id = edge_stream_id if edge_stream_id is not None else _get_field(source, "edge_stream_id")
        edge_oper = edge_oper if edge_oper is not None else _get_field(source, "edge_oper")
        edge_oper_mask = edge_oper_mask if edge_oper_mask is not None else _get_field(source, "edge_oper_mask")
        if edge_stream_role == []:
            edge_stream_role = None
        if edge_stream_id == []:
            edge_stream_id = None
        if edge_oper == []:
            edge_oper = None
        if edge_oper_mask == []:
            edge_oper_mask = None
        batch = batch if batch is not None else _get_field(source, "batch")

        if edge_index is None or x_oper is None:
            raise ValueError(
                "ProcessGraphEncoder requires edge_index and x_oper. "
                "These can be passed directly or supplied through a GraphSample/dict/data object."
            )

        edge_index_tensor = _as_edge_index(edge_index, device)
        x_role_tensor = None if x_role is None else _as_long_tensor(x_role, device, "x_role")
        x_oper_tensor = _as_float_tensor(x_oper, device, "x_oper")
        x_unit_tensor = None if x_unit is None else _as_long_tensor(x_unit, device, "x_unit")
        x_hx_role_tensor = None if x_hx_role is None else _as_long_tensor(x_hx_role, device, "x_hx_role")
        x_oper_mask_tensor = (
            None
            if x_oper_mask is None
            else _as_float_tensor(x_oper_mask, device, "x_oper_mask")
        )
        edge_stream_role_tensor = (
            None if edge_stream_role is None else _as_long_tensor(edge_stream_role, device, "edge_stream_role")
        )
        edge_stream_id_tensor = (
            None if edge_stream_id is None else _as_long_tensor(edge_stream_id, device, "edge_stream_id")
        )
        edge_oper_tensor = None if edge_oper is None else _as_float_tensor(edge_oper, device, "edge_oper")
        edge_oper_mask_tensor = (
            None
            if edge_oper_mask is None
            else _as_float_tensor(edge_oper_mask, device, "edge_oper_mask")
        )

        num_nodes = x_oper_tensor.size(0)
        if x_role_tensor is not None and x_role_tensor.size(0) != num_nodes:
            raise ValueError(
                f"x_role node count mismatch: expected {num_nodes}, got {x_role_tensor.size(0)}."
            )
        if x_unit_tensor is not None and x_unit_tensor.size(0) != num_nodes:
            raise ValueError(
                f"x_unit node count mismatch: expected {num_nodes}, got {x_unit_tensor.size(0)}."
            )
        if x_hx_role_tensor is not None and x_hx_role_tensor.size(0) != num_nodes:
            raise ValueError(
                f"x_hx_role node count mismatch: expected {num_nodes}, got {x_hx_role_tensor.size(0)}."
            )
        if x_oper_mask_tensor is not None and x_oper_mask_tensor.shape != x_oper_tensor.shape:
            raise ValueError(
                "x_oper_mask shape mismatch: expected the same shape as x_oper, "
                f"got x_oper={tuple(x_oper_tensor.shape)} and x_oper_mask={tuple(x_oper_mask_tensor.shape)}."
            )

        if edge_index_tensor.numel() > 0:
            if edge_index_tensor.min().item() < 0 or edge_index_tensor.max().item() >= num_nodes:
                raise IndexError(
                    f"edge_index contains out-of-range node ids for {num_nodes} nodes."
                )
        num_edges = edge_index_tensor.size(1)
        if self.config.use_edge_features:
            if (
                edge_stream_role_tensor is None
                or edge_oper_tensor is None
                or (
                    self.config.use_edge_stream_id
                    and edge_stream_id_tensor is None
                )
            ):
                raise ValueError(
                    "use_edge_features=True requires edge_stream_role and edge_oper; "
                    "edge_stream_id is additionally required when "
                    "use_edge_stream_id=True."
                )
            indexed_tensors = [("edge_stream_role", edge_stream_role_tensor)]
            if edge_stream_id_tensor is not None:
                indexed_tensors.append(("edge_stream_id", edge_stream_id_tensor))
            for name, tensor in indexed_tensors:
                if tensor.size(0) != num_edges:
                    raise ValueError(f"{name} length mismatch: expected {num_edges}, got {tensor.size(0)}.")
            if edge_oper_tensor.size(0) != num_edges:
                raise ValueError(
                    f"edge_oper edge count mismatch: expected {num_edges}, got {edge_oper_tensor.size(0)}."
                )
            if edge_oper_mask_tensor is not None and edge_oper_mask_tensor.shape != edge_oper_tensor.shape:
                raise ValueError(
                    "edge_oper_mask shape mismatch: expected the same shape as edge_oper, "
                    f"got edge_oper={tuple(edge_oper_tensor.shape)} and "
                    f"edge_oper_mask={tuple(edge_oper_mask_tensor.shape)}."
                )

        if batch is None:
            batch_tensor = torch.zeros(num_nodes, device=device, dtype=torch.long)
        else:
            batch_tensor = _as_long_tensor(batch, device, "batch")
            if batch_tensor.size(0) != num_nodes:
                raise ValueError(
                    f"batch length mismatch: expected {num_nodes}, got {batch_tensor.size(0)}."
                )
            if batch_tensor.numel() > 0 and batch_tensor.min().item() < 0:
                raise ValueError("batch must be non-negative.")
            batch_tensor = _make_stable_contiguous_index(batch_tensor)

        return (
            edge_index_tensor,
            x_role_tensor,
            x_unit_tensor,
            x_hx_role_tensor,
            x_oper_tensor,
            x_oper_mask_tensor,
            edge_stream_role_tensor,
            edge_stream_id_tensor,
            edge_oper_tensor,
            edge_oper_mask_tensor,
            batch_tensor,
        )

    def _global_pool(self, x: Tensor, batch: Tensor) -> Tensor:
        num_graphs = _infer_num_graphs(batch)
        if self.config.global_pool == "sum":
            return _scatter_sum(x, batch, dim_size=num_graphs)
        if self.config.global_pool == "mean":
            return _scatter_mean(x, batch, dim_size=num_graphs)
        if self.config.global_pool == "max":
            return _scatter_max(x, batch, dim_size=num_graphs)
        if self.config.global_pool == "attention":
            if self.attention_pool is None:
                raise RuntimeError("attention_pool was not initialized for attention global pooling.")
            return self.attention_pool(x, batch, num_graphs=num_graphs)
        if self.config.global_pool == "concat_set2set":
            if self.set2set_pool is None:
                raise RuntimeError("set2set_pool was not initialized for concat_set2set global pooling.")
            return self.set2set_pool(x, batch, num_graphs=num_graphs)
        raise ValueError(f"Unsupported global_pool '{self.config.global_pool}'.")

    def forward(
        self,
        data: GraphSample | Mapping[str, Any] | Any | None = None,
        *,
        edge_index: Any = None,
        x_role: Any = None,
        x_unit: Any = None,
        x_hx_role: Any = None,
        x_oper: Any = None,
        x_oper_mask: Any = None,
        edge_stream_role: Any = None,
        edge_stream_id: Any = None,
        edge_oper: Any = None,
        edge_oper_mask: Any = None,
        batch: Any = None,
    ) -> EncoderOutput:
        (
            edge_index_tensor,
            x_role_tensor,
            x_unit_tensor,
            x_hx_role_tensor,
            x_oper_tensor,
            x_oper_mask_tensor,
            edge_stream_role_tensor,
            edge_stream_id_tensor,
            edge_oper_tensor,
            edge_oper_mask_tensor,
            batch_tensor,
        ) = self._coerce_inputs(
            data=data,
            edge_index=edge_index,
            x_role=x_role,
            x_unit=x_unit,
            x_hx_role=x_hx_role,
            x_oper=x_oper,
            x_oper_mask=x_oper_mask,
            edge_stream_role=edge_stream_role,
            edge_stream_id=edge_stream_id,
            edge_oper=edge_oper,
            edge_oper_mask=edge_oper_mask,
            batch=batch,
        )

        h0 = self.input_encoder(
            x_role=x_role_tensor,
            x_unit=x_unit_tensor,
            x_hx_role=x_hx_role_tensor,
            x_oper=x_oper_tensor,
            x_oper_mask=x_oper_mask_tensor,
        )
        h = h0
        edge_embeddings = None
        if self.edge_input_encoder is not None:
            if edge_stream_role_tensor is None or edge_oper_tensor is None:
                raise RuntimeError("edge input tensors were not prepared for edge_input_encoder.")
            if edge_stream_id_tensor is None:
                edge_stream_id_tensor = torch.zeros(
                    edge_stream_role_tensor.shape[0],
                    device=edge_stream_role_tensor.device,
                    dtype=torch.long,
                )
            edge_embeddings = self.edge_input_encoder(
                edge_stream_role=edge_stream_role_tensor,
                edge_stream_id=edge_stream_id_tensor,
                edge_oper=edge_oper_tensor,
                edge_oper_mask=edge_oper_mask_tensor,
            )

        layer_outputs: list[Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            h_prev = h
            h_new = layer(h_prev, edge_index_tensor, edge_embeddings=edge_embeddings)
            if self.config.layer_residual_mode == "none":
                h = h_new
            elif self.config.layer_residual_mode == "interp":
                alpha = self.config.layer_residual_alpha
                h = h_prev + alpha * (h_new - h_prev)
            else:
                raise ValueError(
                    f"Unsupported layer_residual_mode '{self.config.layer_residual_mode}'."
                )
            if self._use_per_layer_initial_residual:
                h = h + (self.config.initial_residual_alpha * h0)
            if self.config.layer_residual_norm:
                if self.layer_residual_norms is None:
                    raise RuntimeError("layer_residual_norms is None while layer_residual_norm=True.")
                h = self.layer_residual_norms[layer_idx](h)
            layer_outputs.append(h)

        # Preserve node-local operational signal to avoid collapse on fixed-topology graphs.
        local_node_embeddings = h + h0 if self._use_final_initial_residual else h
        global_embedding = self._global_pool(local_node_embeddings, batch_tensor)
        global_context = global_embedding.index_select(0, batch_tensor)

        if self.final_projection is None:
            node_embeddings = local_node_embeddings
        else:
            node_embeddings = self.final_projection(
                torch.cat([local_node_embeddings, global_context], dim=-1)
            )
            if self._use_final_initial_residual:
                node_embeddings = node_embeddings + h0

        return EncoderOutput(
            initial_node_embeddings=h0,
            local_node_embeddings=local_node_embeddings,
            global_embedding=global_embedding,
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            batch=batch_tensor,
            edge_index=edge_index_tensor,
            layer_node_embeddings=tuple(layer_outputs),
        )
