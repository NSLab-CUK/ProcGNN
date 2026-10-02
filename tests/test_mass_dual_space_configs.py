from pathlib import Path

import pytest

from process_graph.experiment.loaders import load_experiment_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiment" / "pinn" / "mass_dual_space_260730"


@pytest.mark.parametrize(
    ("name", "enabled", "log_weight", "physical_weight"),
    [
        ("d0_mass_dual", False, 1.0, 0.0),
        ("d1_mass_dual", True, 1.0, 0.05),
        ("d2_mass_dual", True, 1.0, 0.10),
        ("d3_mass_dual", True, 1.0, 0.20),
        ("d4_mass_dual", True, 1.0, 0.40),
        ("d5_mass_dual", False, 1.5, 0.0),
        ("d6_mass_dual", False, 2.0, 0.0),
        ("d7_mass_dual", False, 3.0, 0.0),
        ("d8_mass_dual", True, 1.5, 0.10),
        ("d9_mass_dual", True, 2.0, 0.10),
        ("d10_mass_dual", True, 2.0, 0.20),
    ],
)
def test_mass_dual_space_configs_change_only_the_intended_weight(
    name: str,
    enabled: bool,
    log_weight: float,
    physical_weight: float,
) -> None:
    experiment = load_experiment_config(CONFIG_DIR / f"{name}.yaml")
    cfg = experiment.train.mass_flow_physical_auxiliary

    assert cfg.enabled is enabled
    assert cfg.log_weight == pytest.approx(log_weight)
    assert cfg.weight == pytest.approx(physical_weight)
    assert cfg.scale_method == "train_std"
    assert cfg.schedule_type == "linear_warmup"
    assert cfg.schedule_start_epoch == 3
    assert cfg.schedule_end_epoch == 7
    assert cfg.schedule_start_weight == 0.0
    assert cfg.schedule_end_weight == pytest.approx(physical_weight)
    assert cfg.gradient_diagnostics is enabled
    assert experiment.train.epochs == 30
    assert experiment.train.epoch_sampler.hard_fill_total_size == 2000
    assert experiment.train.edge_weight_target_edge == 5.0
