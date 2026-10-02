from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from process_graph.experiment.loaders import load_experiment_config  # noqa: E402
from process_graph.experiment.yaml_utils import deep_merge, read_yaml_mapping  # noqa: E402


class YamlOverrideTests(unittest.TestCase):
    def test_deep_merge_nested_dicts(self) -> None:
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 4}}
        merged = deep_merge(base, override)
        self.assertEqual(merged["a"]["x"], 1)
        self.assertEqual(merged["a"]["y"], 99)
        self.assertEqual(merged["a"]["z"], 4)
        self.assertEqual(merged["b"], 3)

    def test_read_yaml_mapping_tolerates_non_utf8_comment_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_comment.yaml"
            path.write_bytes(b"# invalid comment byte: \x99\nlearning_rate: 0.001\n")
            payload = read_yaml_mapping(path)
        self.assertEqual(payload["learning_rate"], 0.001)

    def test_experiment_overrides_apply_to_sub_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs" / "model").mkdir(parents=True)
            (root / "configs" / "train").mkdir(parents=True)
            (root / "configs" / "data").mkdir(parents=True)
            (root / "configs" / "experiment").mkdir(parents=True)

            (root / "configs" / "model" / "m.yaml").write_text(
                textwrap.dedent(
                    """
                    hidden_dim: 128
                    num_layers: 4
                    role_emb_dim: 16
                    unit_emb_dim: 16
                    hx_role_emb_dim: 8
                    input_mlp_layers: 2
                    diff_mlp_layers: 2
                    update_mlp_layers: 2
                    final_mlp_layers: 2
                    attn_hidden_dim: 64
                    dropout: 0.0
                    activation: relu
                    diff_mode: concat
                    fusion_mode: weighted
                    global_pool: attention
                    use_role_embedding: false
                    use_unit_embedding: false
                    use_hx_role_embedding: false
                    use_oper_mask: false
                    final_projection: true
                    readout_feature_source: hbar
                    """
                ).strip(),
                encoding="utf-8",
            )
            (root / "configs" / "train" / "t.yaml").write_text(
                textwrap.dedent(
                    """
                    epochs: 200
                    batch_size: 16
                    learning_rate: 0.001
                    weight_decay: 0.0001
                    optimizer: adam
                    scheduler: cosine
                    scheduler_t_max: 200
                    scheduler_eta_min: 1.0e-6
                    scheduler_step_size: 50
                    scheduler_gamma: 0.1
                    scheduler_patience: 10
                    scheduler_factor: 0.5
                    scheduler_min_lr: 1.0e-6
                    gradient_clip_norm: 5.0
                    loss_type_target: mse
                    loss_type_tailgas: mse
                    loss_type_aux: mse
                    task_loss_weights:
                      target: 1.0
                      tailgas: 1.0
                      aux: 0.3
                    early_stopping: false
                    early_stopping_patience: 20
                    save_best_only: true
                    monitor_metric: val_loss
                    monitor_mode: min
                    num_workers: 0
                    pin_memory: false
                    mixed_precision: false
                    log_interval: 10
                    val_interval: 5
                    """
                ).strip(),
                encoding="utf-8",
            )
            (root / "configs" / "data" / "d.yaml").write_text(
                textwrap.dedent(
                    """
                    process_spec_dir: data/process_specs/raw
                    process_spec_pattern: "Process*_Adjacency_Matrix.xlsx"
                    train_data_path: data/datasets/train.csv
                    val_data_path: data/datasets/val.csv
                    test_data_path: data/datasets/test.csv
                    passthrough_policy: zero
                    normalize_x_oper: true
                    normalize_targets: true
                    fixed_tasks:
                      target:
                        enabled: true
                        target_name: target
                        target_variable: H2
                        readout_type: node
                        loss_type: mse
                        masked: false
                      tailgas:
                        enabled: true
                        target_name: tailgas
                        target_variable: CO2
                        readout_type: node
                        loss_type: mse
                        masked: false
                    aux_task:
                      enabled: true
                      task_name: heat_duty
                      target_name: aux
                      readout_type: node
                      loss_type: mse
                      masked: true
                    missing_value_strategy: zero
                    use_oper_mask: true
                    graph_label_mode: task_dict
                    cache_process_specs: true
                    process_id_column: process_id
                    split_column: split
                    """
                ).strip(),
                encoding="utf-8",
            )
            (root / "configs" / "experiment" / "e.yaml").write_text(
                textwrap.dedent(
                    """
                    experiment_name: tmp_exp
                    seed: 1
                    device: cpu
                    eval_only: false
                    model_config: configs/model/m.yaml
                    train_config: configs/train/t.yaml
                    data_config: configs/data/d.yaml
                    output_dir: outputs/tmp
                    save_dir: outputs/tmp/ckpt
                    log_dir: outputs/tmp/logs
                    resume_path: ""
                    overrides:
                      model:
                        hidden_dim: 256
                      train:
                        learning_rate: 0.0005
                        batch_size: 8
                      data:
                        normalize_x_oper: false
                        aux_task:
                          task_name: power
                    """
                ).strip(),
                encoding="utf-8",
            )

            # Point loader at our temp repo root by monkeypatching project_root_from_here
            import process_graph.experiment.loaders as loaders

            previous = loaders.project_root_from_here

            def _tmp_root() -> Path:
                return root

            loaders.project_root_from_here = _tmp_root
            try:
                experiment = load_experiment_config(root / "configs" / "experiment" / "e.yaml")
            finally:
                loaders.project_root_from_here = previous

            self.assertEqual(experiment.model.hidden_dim, 256)
            self.assertEqual(experiment.train.learning_rate, 0.0005)
            self.assertEqual(experiment.train.batch_size, 8)
            self.assertFalse(experiment.data.normalize_x_oper)
            self.assertEqual(experiment.data.aux_task.task_name, "power")


if __name__ == "__main__":
    unittest.main()
