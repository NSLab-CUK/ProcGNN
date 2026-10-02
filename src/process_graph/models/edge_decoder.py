from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..constants import PROPERTY_STREAM_ROLE_VOCAB, STREAM_EDGE_FEATURE_SLOTS
from .process_encoder import ActivationName, EdgeHeadType, resolve_activation


def _positive_dims(values, name: str) -> tuple[int, ...]:
    dims = tuple(int(v) for v in values)
    if not dims:
        raise ValueError(f"{name} must contain at least one hidden dimension.")
    if any(v <= 0 for v in dims):
        raise ValueError(f"{name} must contain positive dimensions, got {dims}.")
    return dims


def _norm_activation_dropout_block(in_dim: int, out_dim: int, dropout: float) -> list[nn.Module]:
    return [
        nn.Linear(int(in_dim), int(out_dim)),
        nn.LayerNorm(int(out_dim)),
        nn.GELU(),
        nn.Dropout(float(dropout)),
    ]


def _hidden_stack(input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.extend(_norm_activation_dropout_block(in_dim, int(hidden_dim), dropout))
        in_dim = int(hidden_dim)
    return nn.Sequential(*layers)


def _branch_head(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.extend(_norm_activation_dropout_block(in_dim, int(hidden_dim), dropout))
        in_dim = int(hidden_dim)
    layers.append(nn.Linear(in_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _count_parameters(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def _l1_normalize_nonnegative(values: Tensor) -> Tensor:
    total = values.sum(dim=-1, keepdim=True)
    normalized = values / total.clamp_min(torch.finfo(values.dtype).tiny)
    uniform = torch.full_like(values, 1.0 / values.shape[-1])
    return torch.where(total > 0.0, normalized, uniform)


class TargetBranchHiddenAdapter(nn.Module):
    """Small target-edge residual adapter for a PI branch hidden state."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        bottleneck_dim: int = 16,
        scale: float = 0.1,
        detach_input: bool = True,
        zero_init_output: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        bottleneck_dim = int(bottleneck_dim)
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if bottleneck_dim <= 0:
            raise ValueError(f"bottleneck_dim must be positive, got {bottleneck_dim}.")
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.scale = float(scale)
        self.detach_input = bool(detach_input)
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim)
        if bool(zero_init_output):
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, hidden: Tensor, target_edge_mask: Tensor | None) -> Tensor:
        if target_edge_mask is None:
            return hidden
        if target_edge_mask.ndim != 1:
            raise ValueError(
                "target_edge_mask must be a 1D boolean tensor aligned with edge rows; "
                f"got shape={tuple(target_edge_mask.shape)}."
            )
        if int(target_edge_mask.shape[0]) != int(hidden.shape[0]):
            raise ValueError(
                "target_edge_mask length must match branch hidden edge rows: "
                f"mask={int(target_edge_mask.shape[0])} hidden={int(hidden.shape[0])}."
            )
        mask = target_edge_mask.to(device=hidden.device, dtype=hidden.dtype).reshape(-1, 1)
        adapter_input = hidden.detach() if self.detach_input else hidden
        delta_hidden = self.up(self.activation(self.down(self.norm(adapter_input))))
        return hidden + (mask * self.scale * delta_hidden)


GROUPED_FRAC_ORDER: tuple[str, ...] = (
    "Frac_CH4",
    "Frac_CO",
    "Frac_CO2",
    "Frac_H2",
    "Frac_H2O",
    "Frac_N2",
    "Frac_O2",
)
GROUPED_FLOW_ORDER: tuple[str, ...] = ("Mass_Flow", "Mole_Flow", "Vol_Flow")
GROUPED_COND_ORDER: tuple[str, ...] = ("Pres", "Temp")


def _stream_feature_indices(names: tuple[str, ...]) -> tuple[int, ...]:
    slots = tuple(str(s) for s in STREAM_EDGE_FEATURE_SLOTS)
    missing = [name for name in names if name not in slots]
    if missing:
        raise ValueError(
            f"grouped_property edge head requires stream feature slots {missing}, "
            f"but STREAM_EDGE_FEATURE_SLOTS={list(slots)}."
        )
    return tuple(slots.index(name) for name in names)


class PIGroupedPropertyHead(nn.Module):
    """Physics-informed stream head with separate intensive property outputs."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_species: int,
        property_head_hidden_dim: int = 128,
        property_head_num_layers: int = 2,
        cond_head_output_dim: int = 2,
        mass_head_output_dim: int = 3,
        fraction_activation: str = "softmax",
        fraction_temperature: float = 1.0,
        dropout: float = 0.0,
        target_branch_hidden_adapter_enabled: bool = False,
        target_branch_hidden_adapter_bottleneck_dim: int = 16,
        target_branch_hidden_adapter_scale: float = 0.1,
        target_branch_hidden_adapter_detach_input: bool = True,
        target_branch_hidden_adapter_zero_init_output: bool = True,
        target_branch_hidden_adapter_apply_to=(),
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_species = int(num_species)
        self.cond_head_output_dim = int(cond_head_output_dim)
        self.mass_head_output_dim = int(mass_head_output_dim)
        self.main_stream_output_dim = self.cond_head_output_dim + self.num_species + self.mass_head_output_dim
        self.fraction_activation = str(fraction_activation or "softmax").strip().lower()
        self.fraction_temperature = float(fraction_temperature)
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}.")
        if self.num_species <= 0:
            raise ValueError(f"num_species must be positive, got {self.num_species}.")
        if self.cond_head_output_dim != 2:
            raise ValueError(f"PIGroupedPropertyHead requires cond_head_output_dim=2, got {self.cond_head_output_dim}.")
        if self.mass_head_output_dim not in {1, 3}:
            raise ValueError(
                "PIGroupedPropertyHead requires mass_head_output_dim=1 or 3 "
                f"([Mass_Flow] or [Mass_Flow, Mole_Flow, Vol_Flow]), got {self.mass_head_output_dim}."
            )
        supported_fraction_activations = (
            "softmax",
            "sigmoid",
            "relu",
            "relu_l1",
            "softplus_l1",
            "identity",
        )
        if self.fraction_activation not in supported_fraction_activations:
            raise ValueError(
                "PIGroupedPropertyHead supports fraction_activation in "
                f"{supported_fraction_activations}, got {fraction_activation!r}."
            )
        if self.fraction_temperature <= 0.0:
            raise ValueError(f"fraction_temperature must be positive, got {self.fraction_temperature}.")
        hidden_dim = int(property_head_hidden_dim)
        num_layers = int(property_head_num_layers)
        if hidden_dim <= 0:
            raise ValueError(f"property_head_hidden_dim must be positive, got {hidden_dim}.")
        if num_layers <= 0:
            raise ValueError(f"property_head_num_layers must be >= 1, got {num_layers}.")
        self.property_head_hidden_dim = hidden_dim
        self.property_head_num_layers = num_layers
        hidden_dims = tuple(hidden_dim for _ in range(max(0, num_layers - 1)))
        self.condition_head = _branch_head(self.input_dim, hidden_dims, 2, dropout)
        self.fraction_head = _branch_head(self.input_dim, hidden_dims, self.num_species, dropout)
        self.mass_flow_head = _branch_head(self.input_dim, hidden_dims, self.mass_head_output_dim, dropout)
        thermo_input_dim = 2 + self.num_species
        self.rho_head = _branch_head(thermo_input_dim, hidden_dims, 1, dropout)
        self.h_head = _branch_head(thermo_input_dim, hidden_dims, 1, dropout)
        self.target_branch_hidden_adapter_enabled = bool(target_branch_hidden_adapter_enabled)
        apply_to = tuple(str(v).strip().lower() for v in (target_branch_hidden_adapter_apply_to or ()))
        allowed_apply_to = {"condition", "fraction", "flow"}
        unknown_apply_to = sorted(set(apply_to) - allowed_apply_to)
        if unknown_apply_to:
            raise ValueError(
                "target_branch_hidden_adapter_apply_to supports only "
                f"{sorted(allowed_apply_to)}, got {unknown_apply_to}."
            )
        self.target_branch_hidden_adapter_apply_to = apply_to
        self.target_condition_adapter: TargetBranchHiddenAdapter | None = None
        self.target_fraction_adapter: TargetBranchHiddenAdapter | None = None
        self.target_flow_adapter: TargetBranchHiddenAdapter | None = None
        if self.target_branch_hidden_adapter_enabled:
            if not hidden_dims:
                raise ValueError(
                    "target_branch_hidden_adapter requires property_head_num_layers >= 2 "
                    "so each PI branch has a 128D hidden state before its output layer."
                )
            adapter_kwargs = dict(
                hidden_dim=hidden_dim,
                bottleneck_dim=int(target_branch_hidden_adapter_bottleneck_dim),
                scale=float(target_branch_hidden_adapter_scale),
                detach_input=bool(target_branch_hidden_adapter_detach_input),
                zero_init_output=bool(target_branch_hidden_adapter_zero_init_output),
            )
            if "condition" in self.target_branch_hidden_adapter_apply_to:
                self.target_condition_adapter = TargetBranchHiddenAdapter(**adapter_kwargs)
            if "fraction" in self.target_branch_hidden_adapter_apply_to:
                self.target_fraction_adapter = TargetBranchHiddenAdapter(**adapter_kwargs)
            if "flow" in self.target_branch_hidden_adapter_apply_to:
                self.target_flow_adapter = TargetBranchHiddenAdapter(**adapter_kwargs)

    @staticmethod
    def _forward_branch(
        branch: nn.Sequential,
        z_e: Tensor,
        *,
        adapter: TargetBranchHiddenAdapter | None = None,
        target_edge_mask: Tensor | None = None,
    ) -> Tensor:
        if adapter is None:
            return branch(z_e)
        if len(branch) < 2:
            raise RuntimeError("TargetBranchHiddenAdapter requires a branch with hidden layers and an output layer.")
        h = z_e
        for idx in range(len(branch) - 1):
            h = branch[idx](h)
        h = adapter(h, target_edge_mask=target_edge_mask)
        return branch[-1](h)

    def forward(self, z_e: Tensor, *, target_edge_mask: Tensor | None = None) -> dict[str, Tensor]:
        condition_pred = self._forward_branch(
            self.condition_head,
            z_e,
            adapter=self.target_condition_adapter,
            target_edge_mask=target_edge_mask,
        )
        T_pred = condition_pred[:, 0:1]
        P_pred = condition_pred[:, 1:2]
        frac_logits = self._forward_branch(
            self.fraction_head,
            z_e,
            adapter=self.target_fraction_adapter,
            target_edge_mask=target_edge_mask,
        )
        relu_l1_denominator = None
        relu_l1_all_zero_mask = None
        if self.fraction_activation == "softmax":
            frac_pred = torch.softmax(frac_logits / self.fraction_temperature, dim=-1)
        elif self.fraction_activation == "sigmoid":
            frac_pred = torch.sigmoid(frac_logits)
        elif self.fraction_activation == "relu":
            # A7 ablation: intentionally non-negative but not simplex-normalized.
            frac_pred = F.relu(frac_logits)
        elif self.fraction_activation == "relu_l1":
            frac_raw = F.relu(frac_logits)
            relu_l1_denominator = frac_raw.sum(dim=-1, keepdim=True)
            relu_l1_all_zero_mask = relu_l1_denominator <= 0.0
            frac_pred = _l1_normalize_nonnegative(frac_raw)
        elif self.fraction_activation == "softplus_l1":
            frac_pred = _l1_normalize_nonnegative(F.softplus(frac_logits))
        else:
            frac_pred = frac_logits
        flow_pred = self._forward_branch(
            self.mass_flow_head,
            z_e,
            adapter=self.target_flow_adapter,
            target_edge_mask=target_edge_mask,
        )
        mass_flow_pred = flow_pred[:, 0:1]
        thermo_input = torch.cat([T_pred, P_pred, frac_pred], dim=-1)
        rho_pred = self.rho_head(thermo_input)
        h_pred = self.h_head(thermo_input)
        main_stream_pred = torch.cat([T_pred, P_pred, frac_pred, flow_pred], dim=-1)
        result = {
            "main_stream_pred": main_stream_pred,
            "condition_pred": condition_pred,
            "T_pred": T_pred,
            "P_pred": P_pred,
            "frac_pred": frac_pred,
            "mass_flow_pred": mass_flow_pred,
            "flow_pred": flow_pred,
            "rho_pred": rho_pred,
            "h_pred": h_pred,
        }
        if self.mass_head_output_dim == 3:
            result["mole_flow_pred"] = flow_pred[:, 1:2]
            result["volume_flow_pred"] = flow_pred[:, 2:3]
        if relu_l1_denominator is not None and relu_l1_all_zero_mask is not None:
            result["relu_l1_denominator"] = relu_l1_denominator
            result["relu_l1_all_zero_mask"] = relu_l1_all_zero_mask
        return result


class HierarchicalReducedPIHead(nn.Module):
    """Shared-latent PI head with an optional detached Vol_Flow branch."""

    def __init__(
        self,
        *,
        input_dim: int,
        shared_dim: int,
        branch_dim: int,
        level2_hidden_dim: int,
        num_species: int,
        fraction_temperature: float,
        dropout: float,
        detach_level1_predictions: bool,
        level2_unfreeze_epoch: int | None,
        decoder_intermediate_dim: int = 0,
        condition_hidden_dim: int = 0,
        fraction_hidden_dim: int = 0,
        mass_hidden_dim: int = 0,
        shared_residual: bool = True,
        predict_volume_flow: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.shared_dim = int(shared_dim)
        self.branch_dim = int(branch_dim)
        self.decoder_intermediate_dim = int(
            decoder_intermediate_dim or shared_dim
        )
        self.condition_hidden_dim = int(condition_hidden_dim or branch_dim)
        self.fraction_hidden_dim = int(fraction_hidden_dim or branch_dim)
        self.mass_hidden_dim = int(mass_hidden_dim or branch_dim)
        self.shared_residual = bool(shared_residual)
        self.predict_volume_flow = bool(predict_volume_flow)
        self.level2_hidden_dim = int(level2_hidden_dim)
        self.num_species = int(num_species)
        self.fraction_temperature = float(fraction_temperature)
        self.detach_level1_predictions = bool(detach_level1_predictions)
        self.level2_unfreeze_epoch = (
            None
            if level2_unfreeze_epoch is None
            else int(level2_unfreeze_epoch)
        )
        self.current_epoch = 1
        for name, value in (
            ("input_dim", self.input_dim),
            ("shared_dim", self.shared_dim),
            ("branch_dim", self.branch_dim),
            ("decoder_intermediate_dim", self.decoder_intermediate_dim),
            ("condition_hidden_dim", self.condition_hidden_dim),
            ("fraction_hidden_dim", self.fraction_hidden_dim),
            ("mass_hidden_dim", self.mass_hidden_dim),
            ("level2_hidden_dim", self.level2_hidden_dim),
            ("num_species", self.num_species),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.fraction_temperature <= 0.0:
            raise ValueError("fraction_temperature must be positive.")
        if self.level2_unfreeze_epoch is not None and self.level2_unfreeze_epoch < 1:
            raise ValueError("level2_unfreeze_epoch must be >= 1 or None.")

        self.shared_input = nn.Sequential(
            nn.Linear(self.input_dim, self.decoder_intermediate_dim),
            nn.LayerNorm(self.decoder_intermediate_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.shared_update = nn.Sequential(
            nn.Linear(self.decoder_intermediate_dim, self.shared_dim),
            nn.LayerNorm(self.shared_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.shared_output_norm = nn.LayerNorm(self.shared_dim)
        self.condition_head = _branch_head(
            self.shared_dim, (self.condition_hidden_dim,), 2, dropout
        )
        self.fraction_head = _branch_head(
            self.shared_dim,
            (self.fraction_hidden_dim,),
            self.num_species,
            dropout,
        )
        self.mass_head = _branch_head(
            self.shared_dim, (self.mass_hidden_dim,), 1, dropout
        )
        level1_dim = 2 + self.num_species + 1
        self.volume_head = (
            _branch_head(level1_dim, (self.level2_hidden_dim,), 1, dropout)
            if self.predict_volume_flow
            else None
        )

    def set_training_epoch(self, epoch: int) -> None:
        self.current_epoch = max(1, int(epoch))

    @property
    def level1_detached(self) -> bool:
        if not self.detach_level1_predictions:
            return False
        if self.level2_unfreeze_epoch is None:
            return True
        return self.current_epoch < self.level2_unfreeze_epoch

    def forward(self, descriptor: Tensor) -> dict[str, Tensor]:
        shared_input = self.shared_input(descriptor)
        shared_update = self.shared_update(shared_input)
        if self.shared_residual:
            if shared_input.shape[-1] != shared_update.shape[-1]:
                raise RuntimeError(
                    "shared_residual requires decoder_intermediate_dim == "
                    "shared_dim."
                )
            shared_update = shared_input + shared_update
        shared_latent = self.shared_output_norm(shared_update)
        condition_pred = self.condition_head(shared_latent)
        T_pred = condition_pred[:, 0:1]
        P_pred = condition_pred[:, 1:2]
        frac_logits = self.fraction_head(shared_latent)
        frac_pred = torch.softmax(
            frac_logits / self.fraction_temperature,
            dim=-1,
        )
        mass_flow_pred = self.mass_head(shared_latent)
        level1_pred = torch.cat(
            [T_pred, P_pred, frac_pred, mass_flow_pred],
            dim=-1,
        )
        result = {
            "main_stream_pred": level1_pred,
            "level1_pred": level1_pred,
            "condition_pred": condition_pred,
            "T_pred": T_pred,
            "P_pred": P_pred,
            "frac_pred": frac_pred,
            "mass_flow_pred": mass_flow_pred,
            "shared_edge_latent": shared_latent,
        }
        if self.volume_head is not None:
            level2_input = level1_pred.detach() if self.level1_detached else level1_pred
            volume_flow_pred = self.volume_head(level2_input)
            result["volume_flow_pred"] = volume_flow_pred
            result["main_stream_pred"] = torch.cat([level1_pred, volume_flow_pred], dim=-1)
        return result


class BidirectionalRelationalEdgeReadout(nn.Module):
    """Separate physical forward/backward edge readouts from static edge context."""

    def __init__(self, *, node_dim: int, edge_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        input_dim = int(node_dim) + int(edge_dim)
        self.forward_readout = self._make_branch(input_dim, hidden_dim, output_dim, dropout)
        self.backward_readout = self._make_branch(input_dim, hidden_dim, output_dim, dropout)

    @staticmethod
    def _make_branch(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)), nn.LayerNorm(int(hidden_dim)),
            nn.GELU(), nn.Dropout(float(dropout)), nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(self, h_src: Tensor, h_dst: Tensor, edge_embeddings: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self.forward_readout(torch.cat([h_src, edge_embeddings], dim=-1)),
            self.backward_readout(torch.cat([h_dst, edge_embeddings], dim=-1)),
        )


class BidirectionalEdgeFusionMLP(nn.Module):
    """Fuse the two physical directions without adding a direction gate."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(output_dim)),
            nn.LayerNorm(int(output_dim)),
        )

    def forward(self, z_forward: Tensor, z_backward: Tensor) -> Tensor:
        if z_forward.shape != z_backward.shape:
            raise ValueError("Forward and backward edge readouts must have equal shape.")
        return self.network(torch.cat([z_forward, z_backward], dim=-1))


class HXPairRelationAdapter(nn.Module):
    """Inject the statically paired HX path into final edge embeddings only."""

    def __init__(self, *, edge_dim: int, side_embedding_dim: int = 16,
                 hidden_dim: int = 256, output_dim: int = 384,
                 dropout: float = 0.1, gate_init: float = 0.05) -> None:
        super().__init__()
        edge_dim = int(edge_dim)
        output_dim = int(output_dim)
        if edge_dim != output_dim:
            raise ValueError("HX pair residual requires output_dim == edge_dim.")
        if not 0.0 < float(gate_init) < 1.0:
            raise ValueError("HX pair gate_init must be strictly between 0 and 1.")
        self.edge_dim = edge_dim
        self.side_embedding = nn.Embedding(3, int(side_embedding_dim))
        self.relation = nn.Sequential(
            nn.Linear(2 * edge_dim + int(side_embedding_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)), nn.GELU(), nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), output_dim),
        )
        self.output_norm = nn.LayerNorm(output_dim)
        gate_logit = math.log(float(gate_init) / (1.0 - float(gate_init)))
        self.gate_logit = nn.Parameter(torch.tensor(gate_logit, dtype=torch.float32))

    @property
    def gate(self) -> Tensor:
        return torch.sigmoid(self.gate_logit)

    def forward(self, edge_embeddings: Tensor, current_edge_index: Tensor,
                paired_edge_index: Tensor, pair_mask: Tensor, pair_side: Tensor,
                *, edge_graph_ids: Tensor | None = None) -> Tensor:
        num_edges = int(edge_embeddings.shape[0])
        if edge_embeddings.ndim != 2 or edge_embeddings.shape[1] != self.edge_dim:
            raise ValueError("edge_embeddings must have shape [num_edges, edge_dim].")
        num_relations = int(current_edge_index.numel())
        for name, value in (("current_edge_index", current_edge_index),
                            ("paired_edge_index", paired_edge_index),
                            ("pair_mask", pair_mask), ("pair_side", pair_side)):
            if value.ndim != 1 or value.numel() != num_relations:
                raise ValueError(f"{name} must have shape [{num_relations}].")
        current_edge_index = current_edge_index.to(edge_embeddings.device, torch.long)
        paired_edge_index = paired_edge_index.to(edge_embeddings.device, torch.long)
        pair_mask = pair_mask.to(edge_embeddings.device).reshape(-1) > 0.5
        pair_side = pair_side.to(edge_embeddings.device, torch.long)
        valid_relation = torch.nonzero(pair_mask, as_tuple=False).reshape(-1)
        if valid_relation.numel() == 0:
            return edge_embeddings
        current_index = current_edge_index.index_select(0, valid_relation)
        partner = paired_edge_index.index_select(0, valid_relation)
        if bool((((current_index < 0) | (current_index >= num_edges)).any()) or
                (((partner < 0) | (partner >= num_edges)).any())):
            raise ValueError("HX paired edge index is outside the current batch.")
        if bool((partner == current_index).any()):
            raise ValueError("HX edge cannot be paired with itself.")
        relations = set(zip(current_index.tolist(), partner.tolist()))
        if any((paired, current) not in relations for current, paired in relations):
            raise ValueError("HX paired-edge mapping is not symmetric.")
        valid_side = pair_side.index_select(0, valid_relation)
        if bool(((valid_side < 1) | (valid_side > 2)).any()):
            raise ValueError("HX pair side must be 1=inlet or 2=outlet.")
        relation_sides = set(zip(current_index.tolist(), partner.tolist(), valid_side.tolist()))
        if any((paired, current, 3 - side) not in relation_sides
               for current, paired, side in relation_sides):
            raise ValueError("HX paired-edge sides are not symmetric opposites.")
        if edge_graph_ids is not None:
            graph_ids = edge_graph_ids.to(edge_embeddings.device, torch.long)
            if graph_ids.ndim != 1 or graph_ids.numel() != num_edges:
                raise ValueError("edge_graph_ids must align with edge rows.")
            if bool((graph_ids.index_select(0, current_index) !=
                     graph_ids.index_select(0, partner)).any()):
                raise ValueError("HX paired edges cannot cross graph boundaries.")
        current = edge_embeddings.index_select(0, current_index)
        paired = edge_embeddings.index_select(0, partner)
        side = self.side_embedding(valid_side).to(dtype=edge_embeddings.dtype)
        delta = self.relation(torch.cat([current, paired, side], dim=-1))
        relation_update = self.output_norm(current + self.gate.to(current.dtype) * delta)
        unique_current, inverse = torch.unique(current_index, sorted=True, return_inverse=True)
        update_sum = relation_update.new_zeros((unique_current.numel(), relation_update.shape[-1]))
        update_sum.index_add_(0, inverse, relation_update)
        update_count = torch.bincount(inverse, minlength=unique_current.numel()).to(
            dtype=relation_update.dtype, device=relation_update.device
        )
        updated = update_sum / update_count.unsqueeze(-1)
        result = edge_embeddings.clone()
        result.index_copy_(0, unique_current, updated.to(dtype=result.dtype))
        return result


class EdgeDecoder(nn.Module):
    """Predict per-canonical-edge stream feature vector from node embeddings + structural edge attr."""

    def __init__(
        self,
        *,
        node_hidden_dim: int,
        global_dim: int,
        edge_struct_dim: int,
        out_dim: int,
        hidden_dim: int,
        activation: ActivationName,
        dropout: float,
        edge_embedding_dim: int | None = None,
        mlp_dropout: float | None = None,
        edge_head_type: EdgeHeadType = "single",
        edge_head_dropout: float = 0.1,
        edge_head_shared_dims=(512, 512),
        edge_head_frac_dims=(256, 128),
        edge_head_flow_dims=(512, 256, 128),
        edge_head_cond_dims=(256, 128),
        cond_head_output_dim: int = 2,
        frac_head_output_dim: int = 7,
        mass_head_output_dim: int = 3,
        property_head_hidden_dim: int = 128,
        property_head_num_layers: int = 2,
        fraction_activation: str = "softmax",
        fraction_temperature: float = 1.0,
        species_order=(),
        target_branch_hidden_adapter_enabled: bool = False,
        target_branch_hidden_adapter_bottleneck_dim: int = 16,
        target_branch_hidden_adapter_scale: float = 0.1,
        target_branch_hidden_adapter_detach_input: bool = True,
        target_branch_hidden_adapter_zero_init_output: bool = True,
        target_branch_hidden_adapter_apply_to=(),
        property_stream_role_enabled: bool = False,
        property_stream_role_embedding_dim: int = 16,
        property_stream_role_unknown_role_id: int = 0,
        hierarchical_global_projection: bool = True,
        hierarchical_global_dim: int = 0,
        hierarchical_level2_hidden_dim: int = 64,
        hierarchical_decoder_intermediate_dim: int = 0,
        hierarchical_condition_hidden_dim: int = 0,
        hierarchical_fraction_hidden_dim: int = 0,
        hierarchical_mass_hidden_dim: int = 0,
        hierarchical_shared_residual: bool = True,
        hierarchical_detach_level1_predictions: bool = True,
        hierarchical_level2_unfreeze_epoch: int | None = None,
        hierarchical_predict_volume_flow: bool = True,
        edge_readout_mode: str = "legacy",
        edge_readout_hidden_dim: int = 512,
        edge_readout_dropout: float = 0.1,
        edge_readout_fusion_hidden_dim: int = 512,
        edge_readout_fusion_output_dim: int = 384,
        edge_readout_fusion_dropout: float = 0.1,
        property_prediction_edges_only: bool = False,
        hx_pair_relation_enabled: bool = False,
        hx_pair_side_embedding_dim: int = 16,
        hx_pair_relation_hidden_dim: int = 256,
        hx_pair_relation_output_dim: int = 384,
        hx_pair_relation_dropout: float = 0.1,
        hx_pair_gate_init: float = 0.05,
        feed_head_conditioning_enabled: bool = False,
        feed_head_input_dim: int = 6,
        feed_head_hidden_dim: int = 32,
        feed_head_output_dim: int = 64,
        feed_head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        in_dim = 2 * int(node_hidden_dim) + int(global_dim) + int(edge_struct_dim)
        self.dropout = float(dropout)
        self.in_dim = int(in_dim)
        self.edge_embedding_dim = int(edge_embedding_dim or node_hidden_dim)
        self.property_stream_role_enabled = bool(property_stream_role_enabled)
        self.property_stream_role_embedding_dim = int(
            property_stream_role_embedding_dim
        )
        self.property_stream_role_unknown_role_id = int(
            property_stream_role_unknown_role_id
        )
        if self.property_stream_role_embedding_dim <= 0:
            raise ValueError("property_stream_role_embedding_dim must be positive.")
        if not 0 <= self.property_stream_role_unknown_role_id < len(
            PROPERTY_STREAM_ROLE_VOCAB
        ):
            raise ValueError(
                "property_stream_role_unknown_role_id is outside the property "
                f"role vocabulary: {self.property_stream_role_unknown_role_id}."
            )
        self.property_stream_role_embedding: nn.Embedding | None = None
        if self.property_stream_role_enabled:
            self.property_stream_role_embedding = nn.Embedding(
                len(PROPERTY_STREAM_ROLE_VOCAB),
                self.property_stream_role_embedding_dim,
                padding_idx=self.property_stream_role_unknown_role_id,
            )
        self.property_head_input_dim = self.in_dim + (
            self.property_stream_role_embedding_dim
            if self.property_stream_role_enabled
            else 0
        )
        self.out_dim = int(out_dim)
        self.edge_head_type = str(edge_head_type or "single")
        if self.edge_head_type not in {
            "single",
            "grouped_property",
            "pi_grouped_property",
            "hierarchical_reduced_pi",
        }:
            raise ValueError(
                f"Unsupported edge_head_type={self.edge_head_type!r}. "
                "Expected 'single', 'grouped_property', 'pi_grouped_property', "
                "or 'hierarchical_reduced_pi'."
            )
        self.net: nn.Sequential | None = None
        self.grouped_input_proj: nn.Sequential | None = None
        self.shared_proj: nn.Sequential | None = None
        self.frac_head: nn.Sequential | None = None
        self.flow_head: nn.Sequential | None = None
        self.cond_head: nn.Sequential | None = None
        self.pi_head: PIGroupedPropertyHead | None = None
        self.hierarchical_pi_head: HierarchicalReducedPIHead | None = None
        self.edge_readout_mode = str(edge_readout_mode or "legacy").strip().lower()
        if self.edge_readout_mode not in {
            "legacy",
            "bidirectional_separate",
            "bidirectional_fused",
        }:
            raise ValueError(
                "edge_readout_mode must be 'legacy', 'bidirectional_separate', "
                f"or 'bidirectional_fused', got {self.edge_readout_mode!r}."
            )
        self.bidirectional_edge_readout: BidirectionalRelationalEdgeReadout | None = None
        self.bidirectional_edge_fusion: BidirectionalEdgeFusionMLP | None = None
        self.edge_readout_fusion_output_dim = int(edge_readout_fusion_output_dim)
        self.property_prediction_edges_only = bool(property_prediction_edges_only)
        self.hx_pair_relation_enabled = bool(hx_pair_relation_enabled)
        self.hx_pair_relation: HXPairRelationAdapter | None = None
        if self.hx_pair_relation_enabled:
            self.hx_pair_relation = HXPairRelationAdapter(
                edge_dim=self.edge_embedding_dim,
                side_embedding_dim=int(hx_pair_side_embedding_dim),
                hidden_dim=int(hx_pair_relation_hidden_dim),
                output_dim=int(hx_pair_relation_output_dim),
                dropout=float(hx_pair_relation_dropout),
                gate_init=float(hx_pair_gate_init),
            )
        self.feed_head_conditioning_enabled = bool(feed_head_conditioning_enabled)
        self.feed_head_input_dim = int(feed_head_input_dim)
        self.feed_head_output_dim = int(feed_head_output_dim)
        self.feed_encoder: nn.Module | None = None
        if self.feed_head_conditioning_enabled:
            if self.edge_head_type != "hierarchical_reduced_pi":
                raise ValueError("Feed-head conditioning requires hierarchical_reduced_pi.")
            if float(feed_head_dropout) != 0.0:
                raise ValueError("Feed-head encoder dropout must be 0.0.")
            self.feed_encoder = nn.Sequential(
                nn.Linear(self.feed_head_input_dim, int(feed_head_hidden_dim)),
                nn.LayerNorm(int(feed_head_hidden_dim)), nn.GELU(),
                nn.Linear(int(feed_head_hidden_dim), self.feed_head_output_dim),
                nn.LayerNorm(self.feed_head_output_dim), nn.GELU(),
            )
        self.hierarchical_global_projection_enabled = bool(
            hierarchical_global_projection
        )
        requested_global_dim = int(hierarchical_global_dim)
        self.hierarchical_global_dim = (
            int(node_hidden_dim)
            if requested_global_dim <= 0
            else requested_global_dim
        )
        self.hierarchical_global_projection: nn.Linear | None = None
        self.pi_species_order = tuple(str(s) for s in (species_order or ()))
        self.pi_num_species = len(self.pi_species_order) if self.pi_species_order else int(frac_head_output_dim)
        if self.pi_species_order and len(self.pi_species_order) != int(frac_head_output_dim):
            raise ValueError(
                "species_order length must match frac_head_output_dim: "
                f"len(species_order)={len(self.pi_species_order)} frac_head_output_dim={int(frac_head_output_dim)}."
            )
        self.grouped_frac_indices = _stream_feature_indices(GROUPED_FRAC_ORDER)
        self.grouped_flow_indices = _stream_feature_indices(GROUPED_FLOW_ORDER)
        self.grouped_cond_indices = _stream_feature_indices(GROUPED_COND_ORDER)
        grouped_all = self.grouped_frac_indices + self.grouped_flow_indices + self.grouped_cond_indices
        if sorted(grouped_all) != list(range(12)):
            raise ValueError(
                "grouped_property edge head mapping must cover each of the 12 stream target slots exactly once; "
                f"got indices={list(grouped_all)} for STREAM_EDGE_FEATURE_SLOTS={list(STREAM_EDGE_FEATURE_SLOTS)}."
            )

        if self.edge_head_type == "grouped_property" and self.out_dim != 12:
            raise ValueError(
                "edge_head_type='grouped_property' requires stream_target_dim/out_dim == 12 "
                f"for [Frac(7), Mass/Mole/Vol flow(3), Pres/Temp(2)] reconstruction, got {self.out_dim}."
            )
        mlp_do = float(mlp_dropout) if mlp_dropout is not None else float(dropout)
        if self.edge_head_type == "single":
            self.net = nn.Sequential(
                nn.Linear(in_dim, int(hidden_dim)),
                resolve_activation(activation),
                nn.Dropout(mlp_do),
                nn.Linear(int(hidden_dim), int(out_dim)),
            )
        elif self.edge_head_type == "grouped_property":
            head_do = float(edge_head_dropout)
            shared_dims = _positive_dims(edge_head_shared_dims, "edge_head_shared_dims")
            frac_dims = _positive_dims(edge_head_frac_dims, "edge_head_frac_dims")
            flow_dims = _positive_dims(edge_head_flow_dims, "edge_head_flow_dims")
            cond_dims = _positive_dims(edge_head_cond_dims, "edge_head_cond_dims")
            self.grouped_input_proj = _hidden_stack(in_dim, (int(hidden_dim),), head_do)
            self.shared_proj = _hidden_stack(int(hidden_dim), shared_dims, head_do)
            shared_out_dim = int(shared_dims[-1])
            self.frac_head = _branch_head(shared_out_dim, frac_dims, 7, head_do)
            self.flow_head = _branch_head(shared_out_dim, flow_dims, 3, head_do)
            self.cond_head = _branch_head(shared_out_dim, cond_dims, 2, head_do)
        elif self.edge_head_type == "pi_grouped_property":
            self.pi_head = PIGroupedPropertyHead(
                input_dim=self.property_head_input_dim,
                num_species=self.pi_num_species,
                property_head_hidden_dim=int(property_head_hidden_dim),
                property_head_num_layers=int(property_head_num_layers),
                cond_head_output_dim=int(cond_head_output_dim),
                mass_head_output_dim=int(mass_head_output_dim),
                fraction_activation=str(fraction_activation),
                fraction_temperature=float(fraction_temperature),
                dropout=float(edge_head_dropout),
                target_branch_hidden_adapter_enabled=bool(target_branch_hidden_adapter_enabled),
                target_branch_hidden_adapter_bottleneck_dim=int(target_branch_hidden_adapter_bottleneck_dim),
                target_branch_hidden_adapter_scale=float(target_branch_hidden_adapter_scale),
                target_branch_hidden_adapter_detach_input=bool(target_branch_hidden_adapter_detach_input),
                target_branch_hidden_adapter_zero_init_output=bool(target_branch_hidden_adapter_zero_init_output),
                target_branch_hidden_adapter_apply_to=target_branch_hidden_adapter_apply_to,
            )
        else:
            if self.property_stream_role_enabled:
                raise ValueError(
                    "hierarchical_reduced_pi forbids property stream role direct concat."
                )
            expected_hierarchical_out_dim = (
                11 if bool(hierarchical_predict_volume_flow) else 10
            )
            if int(out_dim) != expected_hierarchical_out_dim:
                raise ValueError(
                    "hierarchical_reduced_pi output width does not match "
                    "hierarchical_pi_head.predict_volume_flow: "
                    f"expected={expected_hierarchical_out_dim}, got={int(out_dim)}."
                )
            if int(mass_head_output_dim) != 1:
                raise ValueError(
                    "hierarchical_reduced_pi requires mass_head_output_dim=1."
                )
            if self.hierarchical_global_projection_enabled:
                self.hierarchical_global_projection = nn.Linear(
                    int(global_dim),
                    self.hierarchical_global_dim,
                )
                descriptor_global_dim = self.hierarchical_global_dim
            else:
                if requested_global_dim not in {0, int(global_dim)}:
                    raise ValueError(
                        "hierarchical_global_dim cannot change global width when "
                        "hierarchical_global_projection=false."
                    )
                descriptor_global_dim = int(global_dim)
                self.hierarchical_global_dim = int(global_dim)
            if self.edge_readout_mode == "legacy":
                descriptor_dim = 2 * int(node_hidden_dim) + descriptor_global_dim + self.edge_embedding_dim
                if self.feed_head_conditioning_enabled:
                    descriptor_dim += self.feed_head_output_dim
            else:
                self.bidirectional_edge_readout = BidirectionalRelationalEdgeReadout(
                    node_dim=int(node_hidden_dim), edge_dim=self.edge_embedding_dim,
                    hidden_dim=int(edge_readout_hidden_dim), output_dim=int(node_hidden_dim),
                    dropout=float(edge_readout_dropout),
                )
                if self.edge_readout_mode == "bidirectional_fused":
                    if self.edge_readout_fusion_output_dim != int(node_hidden_dim):
                        raise ValueError(
                            "bidirectional_fused output width must equal node hidden width."
                        )
                    self.bidirectional_edge_fusion = BidirectionalEdgeFusionMLP(
                        input_dim=int(node_hidden_dim),
                        hidden_dim=int(edge_readout_fusion_hidden_dim),
                        output_dim=self.edge_readout_fusion_output_dim,
                        dropout=float(edge_readout_fusion_dropout),
                    )
                    descriptor_dim = (
                        self.edge_readout_fusion_output_dim + descriptor_global_dim
                    )
                else:
                    descriptor_dim = 2 * int(node_hidden_dim) + descriptor_global_dim
                if self.feed_head_conditioning_enabled:
                    descriptor_dim += self.feed_head_output_dim
            self.hierarchical_pi_head = HierarchicalReducedPIHead(
                input_dim=descriptor_dim,
                shared_dim=int(hidden_dim),
                branch_dim=int(property_head_hidden_dim),
                decoder_intermediate_dim=int(
                    hierarchical_decoder_intermediate_dim
                ),
                condition_hidden_dim=int(hierarchical_condition_hidden_dim),
                fraction_hidden_dim=int(hierarchical_fraction_hidden_dim),
                mass_hidden_dim=int(hierarchical_mass_hidden_dim),
                shared_residual=bool(hierarchical_shared_residual),
                level2_hidden_dim=int(hierarchical_level2_hidden_dim),
                num_species=self.pi_num_species,
                fraction_temperature=float(fraction_temperature),
                dropout=float(edge_head_dropout),
                detach_level1_predictions=bool(
                    hierarchical_detach_level1_predictions
                ),
                level2_unfreeze_epoch=hierarchical_level2_unfreeze_epoch,
                predict_volume_flow=bool(hierarchical_predict_volume_flow),
            )

    @property
    def grouped_property_enabled(self) -> bool:
        return self.edge_head_type == "grouped_property"

    @property
    def pi_grouped_property_enabled(self) -> bool:
        return self.edge_head_type == "pi_grouped_property"

    @property
    def hierarchical_reduced_pi_enabled(self) -> bool:
        return self.edge_head_type == "hierarchical_reduced_pi"

    def set_training_epoch(self, epoch: int) -> None:
        if self.hierarchical_pi_head is not None:
            self.hierarchical_pi_head.set_training_epoch(int(epoch))

    def describe_head(self) -> dict[str, int | str | bool]:
        hierarchical_output_order = (
            ["Temp", "Pres"]
            + [f"Frac_{name}" for name in self.pi_species_order]
            + ["Mass_Flow"]
            + (
                ["Vol_Flow"]
                if self.hierarchical_pi_head is not None
                and self.hierarchical_pi_head.predict_volume_flow
                else []
            )
        )
        return {
            "edge_head_type": self.edge_head_type,
            "stream_target_dim": self.out_dim,
            "grouped_property": self.grouped_property_enabled,
            "pi_grouped_property": self.pi_grouped_property_enabled,
            "hierarchical_reduced_pi": self.hierarchical_reduced_pi_enabled,
            "frac_head_output_dim": (
                7
                if self.grouped_property_enabled
                else self.pi_num_species
                if (self.pi_grouped_property_enabled or self.hierarchical_reduced_pi_enabled)
                else 0
            ),
            "flow_head_output_dim": (
                3
                if self.grouped_property_enabled
                else self.pi_head.mass_head_output_dim
                if self.pi_head is not None
                else 1 + int(self.hierarchical_pi_head.predict_volume_flow)
                if self.hierarchical_pi_head is not None
                else 0
            ),
            "mass_head_output_dim": (
                self.pi_head.mass_head_output_dim
                if self.pi_head is not None
                else 1
                if self.hierarchical_pi_head is not None
                else 0
            ),
            "cond_head_output_dim": 2
            if (
                self.grouped_property_enabled
                or self.pi_grouped_property_enabled
                or self.hierarchical_reduced_pi_enabled
            )
            else 0,
            "pi_num_species": self.pi_num_species
            if (self.pi_grouped_property_enabled or self.hierarchical_reduced_pi_enabled)
            else 0,
            "pi_main_stream_output_dim": (
                self.pi_head.main_stream_output_dim
                if self.pi_head is not None
                else self.out_dim
                if self.hierarchical_pi_head is not None
                else 0
            ),
            "pi_fraction_activation": (
                self.pi_head.fraction_activation
                if self.pi_head is not None
                else "softmax"
                if self.hierarchical_pi_head is not None
                else ""
            ),
            "pi_fraction_temperature": (
                self.pi_head.fraction_temperature
                if self.pi_head is not None
                else self.hierarchical_pi_head.fraction_temperature
                if self.hierarchical_pi_head is not None
                else ""
            ),
            "target_branch_hidden_adapter": self.pi_head.target_branch_hidden_adapter_enabled if self.pi_head is not None else False,
            "target_branch_hidden_adapter_apply_to": list(self.pi_head.target_branch_hidden_adapter_apply_to)
            if self.pi_head is not None
            else [],
            "target_branch_hidden_adapter_parameters": (
                _count_parameters(self.pi_head.target_condition_adapter)
                + _count_parameters(self.pi_head.target_fraction_adapter)
                + _count_parameters(self.pi_head.target_flow_adapter)
                if self.pi_head is not None
                else 0
            ),
            "property_stream_role": self.property_stream_role_enabled,
            "property_stream_role_embedding_dim": (
                self.property_stream_role_embedding_dim
                if self.property_stream_role_enabled
                else 0
            ),
            "property_head_input_dim": (
                self.hierarchical_pi_head.input_dim
                if self.hierarchical_pi_head is not None
                else self.property_head_input_dim
            ),
            "pi_rho_h_in_main_stream_pred": (
                False
                if (self.pi_grouped_property_enabled or self.hierarchical_reduced_pi_enabled)
                else ""
            ),
            "final_output_dim": (
                self.pi_head.main_stream_output_dim
                if self.pi_head is not None
                else self.out_dim
            ),
            "hierarchical_descriptor_dim": (
                self.hierarchical_pi_head.input_dim
                if self.hierarchical_pi_head is not None
                else 0
            ),
            "edge_readout_mode": self.edge_readout_mode,
            "bidirectional_edge_readout_parameters": _count_parameters(self.bidirectional_edge_readout),
            "bidirectional_edge_fusion_parameters": _count_parameters(self.bidirectional_edge_fusion),
            "property_prediction_edges_only": self.property_prediction_edges_only,
            "hx_pair_relation": self.hx_pair_relation_enabled,
            "hx_pair_relation_parameters": _count_parameters(self.hx_pair_relation),
            "feed_head_conditioning": self.feed_head_conditioning_enabled,
            "feed_encoder_parameters": _count_parameters(self.feed_encoder),
            "feed_head_input_dim": self.feed_head_input_dim
            if self.feed_head_conditioning_enabled
            else 0,
            "feed_head_output_dim": self.feed_head_output_dim
            if self.feed_head_conditioning_enabled
            else 0,
            "hierarchical_global_projection": self.hierarchical_global_projection_enabled
            if self.hierarchical_pi_head is not None
            else False,
            "hierarchical_global_dim": self.hierarchical_global_dim
            if self.hierarchical_pi_head is not None
            else 0,
            "hierarchical_shared_dim": self.hierarchical_pi_head.shared_dim
            if self.hierarchical_pi_head is not None
            else 0,
            "hierarchical_branch_dim": self.hierarchical_pi_head.branch_dim
            if self.hierarchical_pi_head is not None
            else 0,
            "hierarchical_decoder_intermediate_dim": (
                self.hierarchical_pi_head.decoder_intermediate_dim
                if self.hierarchical_pi_head is not None
                else 0
            ),
            "hierarchical_condition_hidden_dim": (
                self.hierarchical_pi_head.condition_hidden_dim
                if self.hierarchical_pi_head is not None
                else 0
            ),
            "hierarchical_fraction_hidden_dim": (
                self.hierarchical_pi_head.fraction_hidden_dim
                if self.hierarchical_pi_head is not None
                else 0
            ),
            "hierarchical_mass_hidden_dim": (
                self.hierarchical_pi_head.mass_hidden_dim
                if self.hierarchical_pi_head is not None
                else 0
            ),
            "hierarchical_shared_residual": (
                self.hierarchical_pi_head.shared_residual
                if self.hierarchical_pi_head is not None
                else False
            ),
            "hierarchical_edge_embedding_dim": (
                self.edge_embedding_dim
                if self.hierarchical_pi_head is not None
                else 0
            ),
            "hierarchical_level2_hidden_dim": self.hierarchical_pi_head.level2_hidden_dim
            if self.hierarchical_pi_head is not None
            and self.hierarchical_pi_head.predict_volume_flow
            else 0,
            "hierarchical_predict_volume_flow": (
                self.hierarchical_pi_head.predict_volume_flow
                if self.hierarchical_pi_head is not None
                else False
            ),
            "hierarchical_level1_detached": self.hierarchical_pi_head.level1_detached
            if self.hierarchical_pi_head is not None
            else False,
            "stream_feature_order": (
                hierarchical_output_order
                if self.hierarchical_pi_head is not None
                else list(STREAM_EDGE_FEATURE_SLOTS)
            ),
            "grouped_frac_order": (
                [f"Frac_{name}" for name in self.pi_species_order]
                if self.hierarchical_pi_head is not None
                else list(GROUPED_FRAC_ORDER)
            ),
            "grouped_frac_indices": (
                list(range(2, 2 + self.pi_num_species))
                if self.hierarchical_pi_head is not None
                else list(self.grouped_frac_indices)
            ),
            "grouped_flow_order": (
                ["Mass_Flow"]
                + (["Vol_Flow"] if self.hierarchical_pi_head.predict_volume_flow else [])
                if self.hierarchical_pi_head is not None
                else list(GROUPED_FLOW_ORDER)
            ),
            "grouped_flow_indices": (
                [2 + self.pi_num_species]
                + ([3 + self.pi_num_species] if self.hierarchical_pi_head.predict_volume_flow else [])
                if self.hierarchical_pi_head is not None
                else list(self.grouped_flow_indices)
            ),
            "grouped_cond_order": (
                ["Temp", "Pres"]
                if self.hierarchical_pi_head is not None
                else list(GROUPED_COND_ORDER)
            ),
            "grouped_cond_indices": (
                [0, 1]
                if self.hierarchical_pi_head is not None
                else list(self.grouped_cond_indices)
            ),
            "single_head_parameters": _count_parameters(self.net),
            "grouped_input_proj_parameters": _count_parameters(self.grouped_input_proj),
            "shared_proj_parameters": _count_parameters(self.shared_proj),
            "frac_head_parameters": _count_parameters(self.frac_head),
            "flow_head_parameters": _count_parameters(self.flow_head),
            "cond_head_parameters": _count_parameters(self.cond_head),
            "pi_head_parameters": _count_parameters(self.pi_head),
            "hierarchical_pi_head_parameters": _count_parameters(
                self.hierarchical_pi_head
            ),
            "hierarchical_global_projection_parameters": _count_parameters(
                self.hierarchical_global_projection
            ),
            "total_edge_head_parameters": _count_parameters(self),
        }

    def _forward_grouped_property(self, z: Tensor) -> Tensor:
        if (
            self.grouped_input_proj is None
            or self.shared_proj is None
            or self.frac_head is None
            or self.flow_head is None
            or self.cond_head is None
        ):
            raise RuntimeError("Grouped property edge head modules were not initialized.")
        h = self.grouped_input_proj(z)
        h = self.shared_proj(h)
        frac_out = self.frac_head(h)
        flow_out = self.flow_head(h)
        cond_out = self.cond_head(h)

        if frac_out.shape[-1] != 7:
            raise AssertionError(f"frac_head output dim must be 7, got {frac_out.shape[-1]}.")
        if flow_out.shape[-1] != 3:
            raise AssertionError(f"flow_head output dim must be 3, got {flow_out.shape[-1]}.")
        if cond_out.shape[-1] != 2:
            raise AssertionError(f"cond_head output dim must be 2, got {cond_out.shape[-1]}.")

        out = frac_out.new_empty(*frac_out.shape[:-1], 12)
        for src_i, dst_i in enumerate(self.grouped_frac_indices):
            out[..., dst_i] = frac_out[..., src_i]
        for src_i, dst_i in enumerate(self.grouped_flow_indices):
            out[..., dst_i] = flow_out[..., src_i]
        for src_i, dst_i in enumerate(self.grouped_cond_indices):
            out[..., dst_i] = cond_out[..., src_i]
        if out.shape[-1] != 12:
            raise AssertionError(f"grouped_property final output dim must be 12, got {out.shape[-1]}.")
        return out

    def forward(
        self,
        node_emb_local: Tensor,
        edge_index: Tensor,
        edge_struct_attr: Tensor,
        global_emb: Tensor,
        batch: Tensor,
        edge_embeddings: Tensor | None = None,
        target_edge_mask: Tensor | None = None,
        property_stream_role: Tensor | None = None,
        prediction_edge_mask: Tensor | None = None,
        hx_pair_current_edge_index: Tensor | None = None,
        hx_paired_edge_index: Tensor | None = None,
        hx_pair_mask: Tensor | None = None,
        hx_pair_side: Tensor | None = None,
        graph_feed_input: Tensor | None = None,
    ) -> Tensor | dict[str, Tensor]:
        row, col = edge_index[0], edge_index[1]
        hi = node_emb_local[row]
        hj = node_emb_local[col]
        graph_ids = batch[row].long()
        if graph_ids.device != global_emb.device:
            graph_ids = graph_ids.to(global_emb.device)
        g_e = global_emb[graph_ids]
        z = torch.cat([hi, hj, g_e, edge_struct_attr], dim=-1)
        if self.edge_head_type == "single":
            if self.net is None:
                raise RuntimeError("Single edge decoder head was not initialized.")
            out = self.net(z)
        elif self.edge_head_type == "grouped_property":
            out = self._forward_grouped_property(z)
        elif self.edge_head_type == "pi_grouped_property":
            if self.pi_head is None:
                raise RuntimeError("PI grouped property edge head was not initialized.")
            if self.property_stream_role_enabled:
                if property_stream_role is None:
                    raise ValueError(
                        "edge_property_stream_role is required when "
                        "model.property_stream_role.enabled=true."
                    )
                if property_stream_role.ndim != 1 or property_stream_role.size(0) != z.size(0):
                    raise ValueError(
                        "edge_property_stream_role must have shape [num_edges], "
                        f"got {tuple(property_stream_role.shape)} for {z.size(0)} edges."
                    )
                property_stream_role = property_stream_role.to(
                    device=z.device, dtype=torch.long
                )
                invalid = (property_stream_role < 0) | (
                    property_stream_role >= len(PROPERTY_STREAM_ROLE_VOCAB)
                )
                if bool(invalid.any()):
                    invalid_values = torch.unique(property_stream_role[invalid]).tolist()
                    raise ValueError(
                        "edge_property_stream_role contains IDs outside the "
                        f"vocabulary: {invalid_values}."
                    )
                if self.property_stream_role_embedding is None:
                    raise RuntimeError(
                        "property_stream_role_embedding was not initialized."
                    )
                role_embedding = self.property_stream_role_embedding(
                    property_stream_role
                ).to(dtype=z.dtype)
                z = torch.cat([z, role_embedding], dim=-1)
            return self.pi_head(z, target_edge_mask=target_edge_mask)
        else:
            if self.hierarchical_pi_head is None:
                raise RuntimeError(
                    "Hierarchical reduced PI head was not initialized."
                )
            if edge_embeddings is None:
                raise RuntimeError(
                    "hierarchical_reduced_pi requires the encoder's edge_embeddings."
                )
            if edge_embeddings.ndim != 2 or edge_embeddings.shape != (
                z.shape[0],
                self.edge_embedding_dim,
            ):
                raise ValueError(
                    "edge_embeddings must align with edge rows and node hidden width: "
                    f"got {tuple(edge_embeddings.shape)}, expected "
                    f"({int(z.shape[0])}, {self.edge_embedding_dim})."
                )
            global_context_source = global_emb
            if self.hierarchical_global_projection is not None:
                # Project once per graph, then gather to edges. This is
                # equivalent to per-edge projection but avoids repeated work.
                global_context_source = self.hierarchical_global_projection(
                    global_context_source
                )
            edge_embeddings = edge_embeddings.to(device=hi.device, dtype=hi.dtype)
            if self.hx_pair_relation_enabled:
                if self.hx_pair_relation is None:
                    raise RuntimeError("HX pair relation module was not initialized.")
                if (hx_pair_current_edge_index is None or hx_paired_edge_index is None
                        or hx_pair_mask is None or hx_pair_side is None):
                    raise ValueError(
                        "HX pair relation requires current/paired indices, mask, and side."
                    )
                edge_embeddings = self.hx_pair_relation(
                    edge_embeddings, hx_pair_current_edge_index,
                    hx_paired_edge_index, hx_pair_mask, hx_pair_side,
                    edge_graph_ids=graph_ids,
                )
            if self.property_prediction_edges_only:
                if prediction_edge_mask is None:
                    raise ValueError("relational property prediction requires edge_is_predictable.")
                if prediction_edge_mask.ndim != 1 or prediction_edge_mask.numel() != hi.shape[0]:
                    raise ValueError("edge_is_predictable must align with edge rows.")
                prediction_index = torch.nonzero(
                    prediction_edge_mask.to(device=hi.device).reshape(-1) > 0.5,
                    as_tuple=False,
                ).reshape(-1)
            else:
                prediction_index = torch.arange(hi.shape[0], device=hi.device, dtype=torch.long)
            pred_hi = hi.index_select(0, prediction_index)
            pred_hj = hj.index_select(0, prediction_index)
            pred_graph_ids = graph_ids.index_select(0, prediction_index)
            pred_edge_embeddings = edge_embeddings.index_select(0, prediction_index)
            global_context = global_context_source[pred_graph_ids]
            if self.edge_readout_mode == "legacy":
                descriptor = torch.cat([pred_hi, pred_hj, global_context, pred_edge_embeddings], dim=-1)
            else:
                if self.bidirectional_edge_readout is None:
                    raise RuntimeError("bidirectional relational edge readout was not initialized.")
                z_forward, z_backward = self.bidirectional_edge_readout(
                    pred_hi, pred_hj, pred_edge_embeddings
                )
                if self.edge_readout_mode == "bidirectional_fused":
                    if self.bidirectional_edge_fusion is None:
                        raise RuntimeError(
                            "bidirectional fused edge readout was not initialized."
                        )
                    z_bidirectional = self.bidirectional_edge_fusion(
                        z_forward, z_backward
                    )
                    descriptor = torch.cat(
                        [z_bidirectional, global_context], dim=-1
                    )
                else:
                    descriptor = torch.cat(
                        [z_forward, z_backward, global_context], dim=-1
                    )
            if self.feed_head_conditioning_enabled:
                if self.feed_encoder is None:
                    raise RuntimeError("Feed encoder was not initialized.")
                if graph_feed_input is None:
                    raise ValueError(
                        "Feed-head conditioning requires graph_feed_input [B, 6]."
                    )
                if graph_feed_input.ndim != 2 or graph_feed_input.shape != (
                    global_emb.shape[0], self.feed_head_input_dim
                ):
                    raise ValueError(
                        "graph_feed_input must have shape "
                        f"[{int(global_emb.shape[0])}, {self.feed_head_input_dim}], "
                        f"got {tuple(graph_feed_input.shape)}."
                    )
                graph_feed_input = graph_feed_input.to(
                    device=descriptor.device, dtype=descriptor.dtype
                )
                feed_hidden = self.feed_encoder(graph_feed_input)
                edge_feed_hidden = feed_hidden.index_select(0, pred_graph_ids)
                if prediction_edge_mask is not None:
                    if (
                        prediction_edge_mask.ndim != 1
                        or prediction_edge_mask.numel() != hi.shape[0]
                    ):
                        raise ValueError("edge_is_predictable must align with edge rows.")
                    pred_feed_mask = prediction_edge_mask.to(
                        device=descriptor.device, dtype=descriptor.dtype
                    ).index_select(0, prediction_index).unsqueeze(-1)
                    edge_feed_hidden = edge_feed_hidden * pred_feed_mask
                descriptor = torch.cat([descriptor, edge_feed_hidden], dim=-1)
            if descriptor.shape[-1] != self.hierarchical_pi_head.input_dim:
                raise AssertionError(
                    "hierarchical descriptor width mismatch: "
                    f"expected {self.hierarchical_pi_head.input_dim}, "
                    f"got {int(descriptor.shape[-1])}."
                )
            predicted = self.hierarchical_pi_head(descriptor)
            if not self.property_prediction_edges_only:
                return predicted
            scattered: dict[str, Tensor] = {}
            for name, value in predicted.items():
                full_value = value.new_zeros((hi.shape[0], *value.shape[1:]))
                full_value.index_copy_(0, prediction_index, value)
                scattered[name] = full_value
            return scattered
        if out.shape[-1] != self.out_dim:
            raise AssertionError(
                f"edge decoder output dim mismatch: expected {self.out_dim}, got {out.shape[-1]}."
            )
        return out
