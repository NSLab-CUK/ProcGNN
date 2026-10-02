"""End-to-end smoke test: all-process GNN inference followed by economics."""

from __future__ import annotations

import math

from gnn_economic import GNNEconomicAdapter


def main() -> None:
    with GNNEconomicAdapter(device="cpu", input_policy="masked_proxy") as adapter:
        for process_id in range(1, 11):
            result = adapter.evaluate(process_id)
            objectives = result["objectives"]
            assert result["stream_count"] > 0
            assert math.isfinite(float(objectives["lcoh_usd_per_kg"]))
            assert math.isfinite(float(objectives["specific_co2_kg_per_kg_h2"]))
            audit = result["input_audit"]
            assert audit["proxy_masked"] is True
            print(
                f"[PASS] P{process_id:02d} streams={result['stream_count']} "
                f"LCOH={objectives['lcoh_usd_per_kg']:.4f} "
                f"specific_CO2={objectives['specific_co2_kg_per_kg_h2']:.4f}"
            )


if __name__ == "__main__":
    main()
