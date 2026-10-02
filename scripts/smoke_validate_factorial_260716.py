#!/usr/bin/env python3
"""Validate factorial_260716 configs without training or writing checkpoints."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from process_graph.experiment.config_builders import model_yaml_to_encoder_config  # noqa: E402
from process_graph.experiment.loaders import load_experiment_config  # noqa: E402
from process_graph.models import ProcessSurrogateModel  # noqa: E402
from process_graph.models.process_encoder import global_embedding_dim  # noqa: E402


def _schedule_scale(train_cfg, user_epoch: int) -> float:
    schedule = getattr(train_cfg, "pinn_weight_schedule", None) or {}
    if not isinstance(schedule, dict) or not bool(schedule.get("enabled", False)):
        return 1.0
    stages = schedule.get("stages") or []
    for stage in stages:
        start = int(stage.get("start_epoch", 1) or 1)
        end_raw = stage.get("end_epoch")
        end = None if end_raw is None else int(end_raw)
        if user_epoch >= start and (end is None or user_epoch <= end):
            return float(stage.get("multiplier", stage.get("scale", 1.0)))
    return 1.0


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="configs/experiment/factorial_260716/manifest.csv",
        help="Factorial manifest CSV.",
    )
    args = parser.parse_args()

    manifest_path = (PROJECT_ROOT / args.manifest).resolve()
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    seen = set()
    combos = set()
    failures: list[str] = []

    for row in rows:
        eid = row["experiment_id"]
        cfg_path = (PROJECT_ROOT / row["config_path"]).resolve()
        exp = load_experiment_config(cfg_path)
        enc_cfg = model_yaml_to_encoder_config(exp.model, exp.data)
        model = ProcessSurrogateModel(encoder_config=enc_cfg, task_specs=[])
        decoder = model.edge_decoder
        if decoder is None or decoder.pi_head is None:
            failures.append(f"{eid}: missing PI edge decoder")
            continue

        adapter_expected = _as_bool(row["adapter_enabled"])
        sched_expected = _as_bool(row["pinn_schedule_enabled"])
        tw_expected = float(row["target_edge_weight"])
        hidden_expected = int(row["encoder_hidden_dim"])
        input_expected = int(row["expected_pi_head_input_dim"])
        combo = (adapter_expected, sched_expected, tw_expected, hidden_expected)
        if combo in combos:
            failures.append(f"{eid}: duplicate combo {combo}")
        combos.add(combo)
        seen.add(eid)

        actual = {
            "adapter": bool(decoder.pi_head.target_branch_hidden_adapter_enabled),
            "sched": bool((getattr(exp.train, "pinn_weight_schedule", {}) or {}).get("enabled", False)),
            "tw": float(getattr(exp.train, "edge_weight_target_edge")),
            "hidden": int(enc_cfg.hidden_dim),
            "global": int(global_embedding_dim(enc_cfg)),
            "decoder_in": int(decoder.in_dim),
            "pi_in": int(decoder.pi_head.input_dim),
            "head_hidden": int(decoder.pi_head.property_head_hidden_dim),
        }
        expected_global = 2 * hidden_expected
        expected = {
            "adapter": adapter_expected,
            "sched": sched_expected,
            "tw": tw_expected,
            "hidden": hidden_expected,
            "global": expected_global,
            "decoder_in": input_expected,
            "pi_in": input_expected,
            "head_hidden": 128,
        }
        for key, exp_value in expected.items():
            got = actual[key]
            if got != exp_value:
                failures.append(f"{eid}: {key} expected {exp_value!r}, got {got!r}")

        z = torch.randn(5, int(decoder.pi_head.input_dim))
        target_mask = torch.tensor([1, 0, 1, 0, 0], dtype=torch.bool)
        out = decoder.pi_head(z, target_edge_mask=target_mask)
        if tuple(out["main_stream_pred"].shape) != (5, 12):
            failures.append(f"{eid}: main_stream_pred shape {tuple(out['main_stream_pred'].shape)}")
        frac_sum = out["frac_pred"].sum(dim=-1)
        if not torch.allclose(frac_sum, torch.ones_like(frac_sum), atol=1.0e-5):
            failures.append(f"{eid}: fraction softmax sum drift")
        if sched_expected:
            schedule_values = [_schedule_scale(exp.train, e) for e in (1, 5, 6, 7, 8, 30)]
            if schedule_values != [0.0, 0.0, 0.5, 0.5, 1.0, 1.0]:
                failures.append(f"{eid}: schedule values {schedule_values}")
        else:
            schedule_values = [_schedule_scale(exp.train, e) for e in (1, 5, 6, 7, 8, 30)]
            if any(not math.isclose(v, 1.0) for v in schedule_values):
                failures.append(f"{eid}: fixed schedule values {schedule_values}")

        print(
            f"{eid} ok adapter={actual['adapter']} sched={actual['sched']} "
            f"tw={actual['tw']:.0f} hidden={actual['hidden']} "
            f"global={actual['global']} pi_in={actual['pi_in']}"
        )

    expected_ids = {f"E{i:02d}" for i in range(1, 17)}
    if seen != expected_ids:
        failures.append(f"manifest ids mismatch missing={sorted(expected_ids - seen)} extra={sorted(seen - expected_ids)}")
    if len(combos) != 16:
        failures.append(f"expected 16 unique combos, got {len(combos)}")

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nPASS factorial_260716 config/model smoke validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
