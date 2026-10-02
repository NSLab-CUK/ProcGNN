from __future__ import annotations

import pytest
import torch

from process_graph.models.edge_decoder import PIGroupedPropertyHead


@pytest.mark.parametrize("temperature", [0.5, 0.3, 0.2])
def test_softmax_fraction_temperature_is_applied_to_logits(temperature):
    head = PIGroupedPropertyHead(
        input_dim=4,
        num_species=3,
        property_head_hidden_dim=4,
        property_head_num_layers=1,
        fraction_activation="softmax",
        fraction_temperature=temperature,
    )
    final_linear = head.fraction_head[-1]
    with torch.no_grad():
        final_linear.weight.zero_()
        final_linear.bias.copy_(torch.tensor([0.0, 1.0, 2.0]))

    actual = head(torch.zeros(2, 4))["frac_pred"]
    expected = torch.softmax(
        torch.tensor([0.0, 1.0, 2.0]) / temperature,
        dim=-1,
    ).expand_as(actual)

    assert torch.allclose(actual, expected)
    assert torch.all(actual > 0.0)
    assert torch.allclose(actual.sum(dim=-1), torch.ones(2))


def test_relu_l1_fraction_reports_all_zero_fallback_diagnostics():
    head = PIGroupedPropertyHead(
        input_dim=4,
        num_species=3,
        property_head_hidden_dim=4,
        property_head_num_layers=1,
        fraction_activation="relu_l1",
    )
    final_linear = head.fraction_head[-1]
    with torch.no_grad():
        final_linear.weight.zero_()
        final_linear.bias.fill_(-1.0)

    outputs = head(torch.zeros(2, 4))

    assert torch.allclose(
        outputs["frac_pred"],
        torch.full((2, 3), 1.0 / 3.0),
    )
    assert torch.equal(
        outputs["relu_l1_denominator"],
        torch.zeros(2, 1),
    )
    assert torch.equal(
        outputs["relu_l1_all_zero_mask"],
        torch.ones(2, 1, dtype=torch.bool),
    )
