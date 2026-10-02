from __future__ import annotations

from pathlib import Path

from process_graph.experiment.config_builders import model_yaml_to_encoder_config
from process_graph.experiment.loaders import load_experiment_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = (
    PROJECT_ROOT / "configs" / "experiment" / "pinn" / "ablation_260728"
)

EXPECTED = {
    "a0_r0_j0_m0": (False, False, False, False),
    "a1_r1_j0_m0": (True, False, False, False),
    "a2_r0_j1_m0": (False, True, False, False),
    "a3_r0_j0_m1": (False, False, True, True),
}


def test_three_method_ablation_matrix_resolves_exactly() -> None:
    actual_names = {path.stem for path in CONFIG_DIR.glob("*.yaml")}
    assert actual_names == set(EXPECTED)

    for name, expected in EXPECTED.items():
        experiment = load_experiment_config(CONFIG_DIR / f"{name}.yaml")
        encoder = model_yaml_to_encoder_config(experiment.model)
        node_cfg = experiment.train.node_pinn_optimization
        mass_cfg = experiment.train.mass_flow_physical_auxiliary
        tail_cfg = experiment.train.epoch_sampler.mass_flow_tail

        actual = (
            encoder.property_stream_role_enabled,
            node_cfg.update_mode == "joint_with_supervised_anchor",
            mass_cfg.enabled,
            tail_cfg.enabled,
        )
        assert actual == expected
        assert experiment.train.epochs == 30
        assert experiment.train.val_random_sample_size == 1000
        assert experiment.train.epoch_sampler.epoch_fraction == 0.02
        assert experiment.train.epoch_sampler.hard_fill_total_size == 2000
        assert experiment.train.early_stopping
        assert experiment.train.early_stopping_patience == 5

        if expected[1]:
            assert experiment.train.lambda_h == 0.02
            assert node_cfg.node_outer_weight == 0.05
        else:
            assert experiment.train.lambda_h == 1.0e-11
            assert node_cfg.node_outer_weight == 1.0

        if expected[2]:
            assert mass_cfg.weight == 0.10
            assert mass_cfg.scale_method == "std"
        if expected[3]:
            assert tail_cfg.target_items_per_epoch == 350
            assert sum(tail_cfg.bin_quotas.values()) == 350
