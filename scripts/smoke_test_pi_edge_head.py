from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from process_graph.models.edge_decoder import PIGroupedPropertyHead  # noqa: E402


def main() -> None:
    torch.manual_seed(42)
    num_edges = 16
    hidden_dim = 512
    num_species = 7

    z = torch.randn(num_edges, hidden_dim)
    head = PIGroupedPropertyHead(
        input_dim=hidden_dim,
        num_species=num_species,
        property_head_hidden_dim=128,
        property_head_num_layers=2,
    )
    outputs = head(z)

    assert outputs["main_stream_pred"].shape == (num_edges, 2 + num_species + 3)
    assert outputs["condition_pred"].shape == (num_edges, 2)
    assert outputs["T_pred"].shape == (num_edges, 1)
    assert outputs["P_pred"].shape == (num_edges, 1)
    assert outputs["frac_pred"].shape == (num_edges, num_species)
    assert outputs["mass_flow_pred"].shape == (num_edges, 1)
    assert outputs["mole_flow_pred"].shape == (num_edges, 1)
    assert outputs["volume_flow_pred"].shape == (num_edges, 1)
    assert outputs["flow_pred"].shape == (num_edges, 3)
    assert outputs["rho_pred"].shape == (num_edges, 1)
    assert outputs["h_pred"].shape == (num_edges, 1)

    frac_sum = outputs["frac_pred"].sum(dim=-1)
    assert torch.allclose(frac_sum, torch.ones_like(frac_sum), atol=1.0e-5)
    for key, value in outputs.items():
        assert torch.isfinite(value).all(), f"{key} contains NaN or Inf"

    print(f"main_stream_pred shape: {tuple(outputs['main_stream_pred'].shape)}")
    print(f"rho_pred shape: {tuple(outputs['rho_pred'].shape)}")
    print(f"h_pred shape: {tuple(outputs['h_pred'].shape)}")
    print(f"frac_sum min/max: {float(frac_sum.min()):.8f} / {float(frac_sum.max()):.8f}")
    print("PI edge head smoke test passed")


if __name__ == "__main__":
    main()
