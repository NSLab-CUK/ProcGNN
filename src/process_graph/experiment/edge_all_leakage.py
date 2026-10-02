"""Hard assertions for task_mode=edge_all: no stream targets in structural edge inputs."""

from __future__ import annotations

from typing import Sequence

import torch

from ..constants import STREAM_EDGE_FEATURE_SLOTS
from ..data.tabular_dataset import GraphBatch, V3_EDGE_STRUCT_FEATURE_COLUMNS, V3_EDGE_STRUCT_ROLE_COLUMNS


def assert_no_edge_all_target_leakage(
    batch: GraphBatch,
    *,
    edge_struct_dim: int,
    edge_target_columns: Sequence[str] | None,
) -> None:
    """Fail fast if structural edge inputs look like stream targets or duplicate y_edge."""
    allowed_widths = {len(V3_EDGE_STRUCT_ROLE_COLUMNS), len(V3_EDGE_STRUCT_FEATURE_COLUMNS)}
    if edge_struct_dim not in allowed_widths:
        raise RuntimeError(
            f"edge_struct_dim={edge_struct_dim} must match a supported v3 structural width: "
            f"{len(V3_EDGE_STRUCT_ROLE_COLUMNS)} role-only columns {V3_EDGE_STRUCT_ROLE_COLUMNS} or "
            f"{len(V3_EDGE_STRUCT_FEATURE_COLUMNS)} legacy columns {V3_EDGE_STRUCT_FEATURE_COLUMNS}."
        )
    struct = batch.model_kwargs.get("edge_struct_attr")
    if struct is None:
        struct = batch.model_kwargs.get("edge_oper")
    if struct is None:
        raise RuntimeError("edge_all leakage check: missing edge_struct_attr / edge_oper.")
    if struct.ndim != 2 or struct.shape[-1] != edge_struct_dim:
        raise RuntimeError(
            f"edge_struct_attr expected (*, {edge_struct_dim}), got {tuple(struct.shape)}."
        )
    lo = float(struct.min().detach().cpu())
    hi = float(struct.max().detach().cpu())
    if lo < -1e-5 or hi > 1.0 + 1e-5:
        raise RuntimeError(
            "edge_struct_attr values outside [0, 1]: v3 canonical structural features are binary / "
            f"join flags only; got min={lo}, max={hi}. Stream properties (T, P, flow, fractions) must not "
            "be concatenated into edge_struct_attr."
        )

    if "edge_attr" in batch.model_kwargs:
        ea = batch.model_kwargs["edge_attr"]
        if isinstance(ea, torch.Tensor) and ea.numel() > 0:
            raise RuntimeError(
                "edge_all leakage check: edge_attr must not be used to carry stream targets; "
                "use edge_struct_attr for structural features only."
            )

    yt = batch.targets.get("edge_stream")
    ym = batch.target_masks.get("edge_stream")
    if yt is None or ym is None:
        return
    if struct.data_ptr() == yt.data_ptr() and struct.shape == yt.shape:
        raise RuntimeError("edge_all leakage check: edge_struct_attr and y_edge_true share the same storage.")

    cols = [str(c) for c in (edge_target_columns or STREAM_EDGE_FEATURE_SLOTS)]
    if len(cols) != yt.shape[-1]:
        raise RuntimeError(
            f"edge_target_columns length {len(cols)} != y_edge_true width {yt.shape[-1]}."
        )

    d_st = struct.shape[-1]
    d_y = int(yt.shape[-1])
    if d_y >= d_st:
        m = ym.to(dtype=yt.dtype, device=yt.device)
        while m.ndim < yt.ndim:
            m = m.unsqueeze(-1)
        m = m.expand_as(yt)
        diff = (yt[:, :d_st] - struct).abs()
        max_sup = float((m[:, :d_st] * diff).max().detach().cpu())
        if max_sup < 1e-10 and float(m.sum().detach().cpu()) > 0.0:
            raise RuntimeError(
                "edge_all leakage check: y_edge_true[:, :edge_struct_dim] matches edge_struct_attr on "
                "supervised edges — targets may have been copied into inputs."
            )
