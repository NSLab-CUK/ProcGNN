from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional

OptimizerName = Literal["adam", "adamw"]
SchedulerName = Literal["none", "cosine", "step", "plateau"]
LossName = Literal["mse", "l1", "smooth_l1", "percentage"]
TargetTransformName = Literal["none", "log1p"]
PIMassFlowOutputSpace = Literal["raw_z", "log1p"]
MassFlowTransform = Literal["log1p", "tempered_log", "scaled_log"]
DebugTaskMode = Literal["all", "target_only", "tailgas_only"]
PassthroughPolicy = Literal["zero", "inherit_upstream"]
MissingValueStrategy = Literal["zero", "nan"]
GraphLabelMode = Literal["task_dict"]
TaskModeName = Literal["multitask", "target_only", "edge_all"]
MonitorMetric = Literal[
    "val_loss",
    "val_mae",
    "val_rmse",
    "val_r2",
    "val_target_r2",
    "val_target_mean_r2",
    "val_target_edge_property_mean_r2",
    "target_r2",
    "target_mean_r2",
    "target_edge_property_mean_r2",
    "val_target_edge_10d_r2_flatten",
    "target_edge_10d_r2_flatten",
    "flatten_r2",
    "val_eval_primary_frac_r2_by_process",
]
MonitorMode = Literal["min", "max"]
ReadoutTypeName = Literal["node", "node_set", "graph", "edge"]
TopologyModeName = Literal["excel_node", "stream_edge"]
AuxTaskName = Literal["heat_duty", "power", "flow", "hx_area"]

DECODER_CATEGORIES: tuple[str, ...] = ("power", "heat_duty", "flow", "hx_area")


@dataclass
class ModelYamlConfig:
    """YAML mirror of high-level model settings (mapped to `ProcessEncoderConfig`)."""

    hidden_dim: int = 256
    num_layers: int = 4
    role_emb_dim: int = 16
    unit_emb_dim: int = 16
    hx_role_emb_dim: int = 8
    input_mlp_layers: int = 2
    diff_mlp_layers: int = 2
    update_mlp_layers: int = 2
    final_mlp_layers: int = 2
    attn_hidden_dim: int = 64
    dropout: float = 0.0
    activation: str = "relu"
    diff_mode: str = "concat"
    fusion_mode: str = "weighted"
    global_pool: str = "concat_set2set"
    set2set_processing_steps: int = 3
    use_role_embedding: bool = False
    use_unit_embedding: bool = True
    use_hx_role_embedding: bool = False
    oper_mask_mode: str = "ignore"
    input_residual: bool = True
    initial_residual_mode: str = "final"
    initial_residual_alpha: float = 0.2
    layer_residual_mode: str = "none"
    layer_residual_alpha: float = 0.5
    layer_residual_norm: bool = False
    use_final_projection: bool = True
    readout_feature_source: str = "hbar"
    use_edge_features: bool = False
    edge_stream_role_emb_dim: int = 8
    edge_stream_id_emb_dim: int = 16
    edge_stream_vocab_size: int = 512
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
    edge_head_type: str = "single"
    edge_head_dropout: float = 0.1
    edge_head_shared_dims: list[int] = field(default_factory=lambda: [512, 512])
    edge_head_frac_dims: list[int] = field(default_factory=lambda: [256, 128])
    edge_head_flow_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    edge_head_cond_dims: list[int] = field(default_factory=lambda: [256, 128])
    cond_head_output_dim: int = 2
    frac_head_output_dim: int = 7
    mass_head_output_dim: int = 3
    property_head_hidden_dim: int = 128
    property_head_num_layers: int = 2
    fraction_activation: str = "softmax"
    fraction_temperature: float = 1.0
    species_order: list[str] = field(default_factory=list)
    target_branch_hidden_adapter: dict[str, Any] = field(default_factory=dict)
    property_stream_role: dict[str, Any] = field(default_factory=dict)
    hierarchical_pi_head: dict[str, Any] = field(default_factory=dict)
    dimension_design: dict[str, Any] = field(default_factory=dict)
    flow_gnn: dict[str, Any] = field(default_factory=dict)
    edge_readout: dict[str, Any] = field(default_factory=dict)
    hx_pair_relation: dict[str, Any] = field(default_factory=dict)
    known_feed_condition: dict[str, Any] = field(default_factory=dict)
    feed_head_conditioning: dict[str, Any] = field(default_factory=dict)


@dataclass
class RarePositiveSamplerConfig:
    enabled: bool = False
    components: list[str] = field(
        default_factory=lambda: ["Frac_CH4", "Frac_CO", "Frac_CO2"]
    )
    positive_threshold: float = 1.0e-4
    max_weight: float = 10.0
    mode: str = "any_target_edge"
    replacement: bool = True


@dataclass
class RareTargetEdgeAugmentationConfig:
    enabled: bool = False
    components: list[str] = field(
        default_factory=lambda: ["Frac_CH4", "Frac_CO2", "Frac_CO"]
    )
    positive_threshold: float = 1.0e-4
    high_positive_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"Frac_CO2": 0.05}
    )
    default_factor: int = 2
    high_positive_factor: int = 3
    max_augmented_ratio: float = 0.5
    target_edges_only: bool = True
    seed: int = 42


@dataclass
class TargetEdgeAugmentationConfig:
    enabled: bool = False
    max_augmented_items: int = 0
    max_augmented_ratio: Optional[float] = None
    factor: int = 3
    seed: int = 42
    target_properties: list[str] = field(
        default_factory=lambda: ["Frac_CO2", "Frac_CH4", "Frac_CO"]
    )
    selection_mode: str = "balanced_property_edge"
    property_quota: Dict[str, int] = field(default_factory=dict)
    max_per_target_edge: int = 150
    min_positive_threshold: Dict[str, float] = field(
        default_factory=lambda: {
            "Frac_CO2": 0.05,
            "Frac_CH4": 0.02,
            "Frac_CO": 0.01,
        }
    )
    adaptive_bins: list[dict[str, Any]] = field(default_factory=list)
    repeat_cap_per_original: int = 20
    mass_flow_priority_edges: Dict[str, float] = field(default_factory=dict)
    keep_original_items: bool = True
    apply_to: list[str] = field(default_factory=lambda: ["train"])


@dataclass
class SampleHybridTargetEdgeStepPIConfig:
    enabled: bool = False
    non_target_reduction: str = "edge_group_macro_mean"
    target_update_order: str = "canonical_edge_id"
    use_existing_target_edge_weight: bool = False
    require_batch_size_one: bool = True
    allow_duplicate_canonical_edge_rows: bool = True
    log_sample_update_details: bool = False
    collect_update_diagnostics: bool = False
    profile_timing: bool = False
    log_module_gradient_norms: bool = False


@dataclass
class NodePinnOptimizationConfig:
    update_mode: str = "separate"
    supervised_anchor_weight: float = 1.0
    node_outer_weight: float = 1.0
    anchor_apply_target_weight: bool = False
    log_gradient_diagnostics: bool = False


@dataclass
class MassFlowPhysicalAuxiliaryConfig:
    enabled: bool = False
    log_weight: float = 1.0
    weight: float = 0.10
    loss: str = "huber"
    delta: float = 1.0
    residual_clip: float | None = 10.0
    scale_method: str = "train_std"
    fixed_scale: float | None = None
    resolved_scale: float | None = None
    minimum_scale: float = 1.0e-8
    epsilon: float = 1.0e-8
    schedule_type: str = "linear_warmup"
    schedule_start_epoch: int = 3
    schedule_end_epoch: int = 7
    schedule_start_weight: float = 0.0
    schedule_end_weight: float | None = None
    current_epoch: int = 1
    gradient_diagnostics: bool = False


@dataclass
class EpochSamplerMixtureConfig:
    uniform: float = 0.40
    sparse_positive: float = 0.30
    quantile_balance: float = 0.20
    mass_flow_tail: float = 0.10
    hard_target_edges: float = 0.0


@dataclass
class EpochSamplerSparsePositiveConfig:
    properties: list[str] = field(
        default_factory=lambda: ["Frac_CH4", "Frac_CO", "Frac_CO2", "Frac_H2"]
    )
    zero_threshold: float = 1.0e-8
    positive_quantiles: list[float] = field(default_factory=lambda: [0.0, 0.5, 0.9, 1.0])


@dataclass
class EpochSamplerQuantileBalanceConfig:
    properties: list[str] = field(
        default_factory=lambda: [
            "Temp",
            "Pres",
            "Mass_Flow",
            "Frac_H2O",
            "Frac_H2",
            "Frac_CH4",
            "Frac_CO2",
            "Frac_CO",
            "Frac_O2",
            "Frac_N2",
        ]
    )
    quantiles: list[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


@dataclass
class EpochSamplerMassFlowTailConfig:
    enabled: bool = False
    target_items_per_epoch: int = 0
    property: str = "Mass_Flow"
    quantiles: list[float] = field(default_factory=lambda: [0.90, 0.95, 0.99, 1.00])
    bin_quotas: dict[str, int] = field(
        default_factory=lambda: {
            "q0.90_0.95": 100,
            "q0.95_0.99": 150,
            "q0.99_1.00": 100,
        }
    )
    quantile_scope: str = "process"
    process_balanced: bool = True
    canonical_edge_balanced: bool = True
    allow_duplicate_samples: bool = False
    canonical_edge_min_count: int = 50


@dataclass
class EpochSamplerHardTargetEdgesConfig:
    enabled: bool = False
    property_quotas: dict[str, int] = field(default_factory=dict)
    rule_quotas: dict[str, int] = field(default_factory=dict)
    rules: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EpochBalancedSamplerConfig:
    enabled: bool = False
    epoch_fraction: float = 0.01
    base_epoch_size: int = 0
    hard_fill_total_size: int = 0
    hard_fill_ratio: float = 0.40
    hard_fill_base_exclude_hard_target_edges: bool = True
    balance_processes: bool = True
    mode: str = "mixture"
    warmup_end_epoch: int = 5
    transition_end_epoch: int = 7
    transition_hard_ratio: float = 0.20
    final_hard_ratio: float = 0.30
    unique_within_epoch: bool = True
    replacement_for_rare_buckets: bool = True
    replacement_for_uniform: bool = False
    max_repeat_per_sample_per_epoch: int = 1
    sampler_seed: int = 260716
    log_sample_selection_counts: bool = True
    mixture: EpochSamplerMixtureConfig = field(default_factory=EpochSamplerMixtureConfig)
    sparse_positive: EpochSamplerSparsePositiveConfig = field(
        default_factory=EpochSamplerSparsePositiveConfig
    )
    quantile_balance: EpochSamplerQuantileBalanceConfig = field(
        default_factory=EpochSamplerQuantileBalanceConfig
    )
    mass_flow_tail: EpochSamplerMassFlowTailConfig = field(
        default_factory=EpochSamplerMassFlowTailConfig
    )
    hard_target_edges: EpochSamplerHardTargetEdgesConfig = field(
        default_factory=EpochSamplerHardTargetEdgesConfig
    )


@dataclass
class TrainConfig:
    epochs: int = 200
    # When positive, training stops after exactly this many applied optimizer
    # updates. This is distinct from batches/epochs because edge-step training
    # can perform several optimizer updates for one graph sample.
    max_optimizer_steps: int = 0
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: OptimizerName = "adam"
    scheduler: SchedulerName = "cosine"
    scheduler_t_max: int = 200
    scheduler_eta_min: float = 1e-6
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.1
    scheduler_patience: int = 10
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    gradient_clip_norm: float = 5.0
    loss_type_target: LossName = "smooth_l1"
    loss_type_tailgas: LossName = "smooth_l1"
    loss_type_decoder: LossName = "smooth_l1"
    loss_type_aux: LossName = "smooth_l1"
    loss_type_edge_stream: LossName = "smooth_l1"
    edge_stream_loss_weight: float = 0.0
    loss_type_edge_all: LossName = "smooth_l1"
    answer_edge_weight: float = 5.0
    answer_edge_weight_h2: float = 5.0
    answer_edge_weight_co2: float = 5.0
    answer_edge_weight_h2o: float = 5.0
    use_v4_target_edge_weighting: bool = True
    v4_target_weighting_mode: str = "target_row"
    use_edge_all_loss: bool = True
    use_target_feature_loss: bool = True
    use_v4_target_row_loss: bool = False
    v4_target_row_loss_weight: float = 0.1
    use_target_row_frac_loss: bool = True
    use_target_row_amount_loss: bool = False
    lambda_target_feature: float = 1.0
    lambda_target_row_frac: float = 1.0
    lambda_target_row_amount: float = 0.0
    primary_frac_loss_weight: float = 5.0
    all_edge_r2_loss_weight: Optional[float] = None
    all_edge_r2_min_count: Optional[int] = None
    all_edge_r2_sst_threshold: Optional[float] = None
    all_edge_r2_loss_cap: Optional[float] = None
    primary_frac_r2_loss_weight: float = 0.1
    primary_frac_r2_min_count: int = 8
    primary_frac_r2_sst_threshold: float = 1.0e-4
    primary_frac_r2_loss_cap: float = 0.0
    target_feature_loss_scope: str = "target_incident_edges"
    use_target_stream_loss_weighting: bool = False
    target_stream_loss_weight: float = 5.0
    target_stream_weighting_mode: str = "unique_target_stream_edge_by_process"
    target_stream_weight_conflict_policy: str = "max"
    allow_yaml_only_target_streams: bool = False
    target_stream_loss_weights: list[dict[str, Any]] | Dict[str, Any] | None = None
    target_incident_exclude_virtual_nodes: bool = True
    target_incident_virtual_node_names: list[str] = field(default_factory=lambda: ["V_INPUT", "V_OUTPUT"])
    target_incident_include_target_edge: bool = True
    target_incident_use_answer_edge_weight: bool = False
    target_feature_loss_space: str = "normalized"
    target_feature_loss_balancing: str = "process_target_balanced"
    use_legacy_answer_weighting: bool = False
    edge_all_use_y_edge_mask: bool = True
    edge_all_print_first_batch: bool = False
    edge_all_assert_no_leakage: bool = False
    edge_all_export_target_feature_mapping_audit: bool = False
    edge_all_export_predictions: bool = True
    task_loss_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "target": 1.0,
            "tailgas": 1.0,
            "edge_stream": 1.0,
            "power": 0.3,
            "heat_duty": 0.3,
            "flow": 0.3,
            "hx_area": 0.3,
        }
    )
    early_stopping: bool = False
    early_stopping_patience: int = 20
    save_last_checkpoint: bool = True
    checkpoint_include_optimizer_state: bool = True
    save_best_only: bool = True
    monitor_metric: MonitorMetric = "val_loss"
    monitor_mode: MonitorMode = "min"
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = True
    prefetch_factor: int = 2
    rare_positive_sampler: RarePositiveSamplerConfig = field(
        default_factory=RarePositiveSamplerConfig
    )
    rare_target_edge_augmentation: RareTargetEdgeAugmentationConfig = field(
        default_factory=RareTargetEdgeAugmentationConfig
    )
    target_edge_augmentation: TargetEdgeAugmentationConfig = field(
        default_factory=TargetEdgeAugmentationConfig
    )
    mixed_precision: bool = False
    eval_batch_size: int = 0
    log_interval: int = 10
    terminal_log_verbosity: str = "compact"
    val_interval: int = 5
    val_random_sample_size: int = 0
    val_random_sample_seed: int = 42
    val_random_sample_each_epoch: bool = True
    compute_validation_pinn_loss: bool = True
    debug_loss_threshold: float = 100.0
    debug_top_batches_to_report: int = 5
    debug_mode: bool = False
    debug_epochs: int = 1
    debug_vector_batches: int = 3
    debug_task_mode: DebugTaskMode = "all"
    tiny_overfit_samples: int = 0
    max_nonfinite_train_batches_per_epoch: int = 0
    recover_nonfinite_edge_all: bool = True
    backward_mode: str = "batch_total"
    shuffle_edges_each_batch: bool = True
    scheduler_step_unit: str = "epoch"
    gradient_accumulation_steps: int = 1
    sample_hybrid_target_edge_step_pi: SampleHybridTargetEdgeStepPIConfig = field(
        default_factory=SampleHybridTargetEdgeStepPIConfig
    )
    node_pinn_optimization: NodePinnOptimizationConfig = field(default_factory=NodePinnOptimizationConfig)
    mass_flow_physical_auxiliary: MassFlowPhysicalAuxiliaryConfig = field(
        default_factory=MassFlowPhysicalAuxiliaryConfig
    )
    pinn_weight_schedule: Dict[str, Any] = field(default_factory=dict)
    epoch_sampler: EpochBalancedSamplerConfig = field(default_factory=EpochBalancedSamplerConfig)
    debug_edge_step: bool = False
    debug_edge_step_pi: bool = False
    max_train_batches: int = 0
    edge_weight_default: float = 1.0
    edge_weight_target_edge: float = 5.0
    use_all_edge_r2_loss_in_edge_step: bool = False
    use_total_loss_in_edge_step_pi: bool = False
    use_all_edge_r2_loss_in_edge_step_pi: bool = False
    use_primary_frac_loss_in_edge_step_pi: bool = False
    use_pinn_loss: bool = True
    pi_main_loss_type: str = ""
    pi_fraction_loss_type: str = "legacy"
    pi_normalize_fraction_loss: bool = False
    pi_fraction_loss_std_min: float = 1.0e-3
    pi_runtime_physical_checks: bool = True
    pi_fraction_clr_eps: float = 1.0e-6
    pi_fraction_clr_loss_weight: float = 1.0
    pi_fraction_log_eps: float = 1.0e-6
    pi_fraction_log_loss_weight: float = 1.0
    pi_fraction_closure_loss_weight: float = 0.0
    pi_fraction_component_penalty: Dict[str, Any] = field(default_factory=dict)
    use_zero_flow_fraction_mask: bool = False
    zero_flow_fraction_mask_eps: float = 1.0e-8
    use_log1p_mass_flow_loss: bool = False
    log1p_mass_flow_loss_std_min: float = 1.0e-6
    pi_mass_flow_output_space: PIMassFlowOutputSpace = "raw_z"
    mass_flow_transform: MassFlowTransform = "log1p"
    mass_flow_log_tau: float = 1.0
    mass_flow_log_scale: float = 1.0
    mass_flow_log_eps: float = 1.0e-8
    mass_flow_log_loss_weight: float = 1.0
    compute_target_edge_property_metrics: bool = False
    export_final_target_edge_property_metrics: bool = False
    save_target_edge_actual_vs_pred_plots: bool = False
    save_best_actual_vs_pred_plots: bool = True
    save_pi_all_edge_actual_vs_pred_plots: bool = False
    actual_vs_pred_max_points_per_property: int = 50000
    final_target_edge_metric_splits: list[str] = field(default_factory=lambda: ["val"])
    metric_relevance_enabled: bool = False
    metric_relevance_fraction_threshold: float = 0.6
    metric_relevance_flow_threshold: float = 0.6
    save_oracle_diagnostic_metrics: bool = False
    oracle_threshold_sweep_enabled: bool = True
    oracle_calibration_enabled: bool = True
    oracle_fraction_postprocess_enabled: bool = True
    oracle_threshold_candidates: list[float] = field(
        default_factory=lambda: [0.0, 1.0e-6, 1.0e-4, 1.0e-3, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6]
    )
    oracle_quantile_candidates: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
    oracle_min_samples: int = 100
    save_diagnostic_display_metrics: bool = True
    diagnostic_corr2_enabled: bool = True
    diagnostic_calibrated_r2_enabled: bool = True
    diagnostic_transformed_r2_enabled: bool = True
    diagnostic_fraction_postprocess_enabled: bool = True
    diagnostic_min_samples: int = 100
    lambda_main: float = 1.0
    lambda_rho: float = 0.01
    lambda_h: float = 0.01
    lambda_volume: float = 0.0
    rho_residual_abs_clip: Optional[float] = None
    lambda_enthalpy_flow: float = 0.0
    lambda_atom: float = 0.0
    lambda_energy: float = 0.0
    compute_node_balance_diagnostics: bool = True
    compute_collapse_diagnostics: bool = True
    use_node_mass_balance_loss: bool = False
    lambda_node_mass: float = 0.0
    node_mass_balance_relative: bool = True
    use_node_component_balance_loss: bool = False
    lambda_node_component: float = 0.0
    node_component_balance_relative: bool = True
    use_node_atom_balance_loss: Optional[bool] = None
    lambda_node_atom: Optional[float] = None
    node_atom_balance_relative: bool = True
    use_node_energy_balance_loss: Optional[bool] = None
    lambda_node_energy: Optional[float] = None
    node_energy_balance_relative: bool = True
    node_energy_valid_unit_types: list[str] = field(default_factory=list)
    node_energy_exclude_unit_types: list[str] = field(
        default_factory=lambda: ["heater", "cooler", "reactor"]
    )
    node_energy_use_q: bool = False
    use_rho_loss: bool = True
    use_h_loss: bool = True
    use_volume_loss: bool = False
    use_enthalpy_flow_loss: bool = False
    use_atom_balance_loss: bool = False
    use_energy_balance_loss: bool = False
    pinn_loss_reduction: str = "mean"
    target_metric_min_count: int = 1
    target_metric_sst_threshold: float = 1.0e-6
    target_metric_r2_floor: float = -0.07
    target_mean_r2_excluded_properties: list[str] = field(default_factory=list)
    eps: float = 1.0e-8
    h_basis: str = "mass_specific"
    molecular_weight_unit: str = "g_per_mol"
    mw_unit_scale: Optional[float] = None
    volume_unit_scale: float = 1.0
    h_loss_normalized: bool = True
    volume_loss_normalized: bool = True
    h_loss_scale_floor: Optional[float] = None
    h_normalized_loss_type: str = "mse"
    h_normalized_huber_delta: float = 1.0
    h_normalized_residual_abs_clip: Optional[float] = None
    volume_loss_scale_floor: Optional[float] = None
    volume_normalized_loss_type: str = "mse"
    volume_normalized_huber_delta: float = 1.0
    volume_normalized_residual_abs_clip: Optional[float] = None
    nonfinite_recovery_lr_factor: float = 0.5
    nonfinite_recovery_min_lr: float = 1.0e-6
    probe_final_grad_norms: bool = False
    target_weight_by_id: Dict[str, float] | None = None
    node_balance_pi: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FixedTaskConfig:
    enabled: bool = True
    target_name: str = "target"
    target_variable: str = ""
    readout_type: ReadoutTypeName = "node"
    loss_type: LossName = "smooth_l1"
    masked: bool = False
    transform: TargetTransformName = "none"
    clip_max_percentile: Optional[float] = None


@dataclass
class AuxTaskConfig:
    enabled: bool = False
    task_name: AuxTaskName = "heat_duty"
    target_name: str = "aux"
    readout_type: ReadoutTypeName = "node"
    loss_type: LossName = "smooth_l1"
    masked: bool = True


@dataclass
class DecoderTaskConfig:
    enabled: bool = False
    loss_type: LossName = "smooth_l1"
    readout_type: ReadoutTypeName = "node"


@dataclass
class DataConfig:
    process_spec_dir: str = "data/process_specs/raw"
    process_spec_pattern: str = "Process*_Adjacency_Matrix.xlsx"
    topology_mode: TopologyModeName = "excel_node"
    use_canonical_graph_spec_v3: bool = False
    canonical_graph_spec_v3_dir: str = "data/reference/v3"
    stream_edge_mapping_path: str = "data/main_data_Streams/process_1_10_stream_edge_mapping_final.csv"
    stream_data_dir: str = "data/main_data_Streams"
    main_data_dir: str = "data/main_data"
    train_data_path: str = "data/datasets/train.csv"
    val_data_path: str = "data/datasets/val.csv"
    test_data_path: str = "data/datasets/test.csv"
    passthrough_policy: PassthroughPolicy = "zero"
    use_oper_mask: bool = True
    missing_value_strategy: MissingValueStrategy = "zero"
    normalize_x_oper: bool = True
    normalize_targets: bool = True
    task_mode: TaskModeName = "multitask"
    fixed_tasks: Dict[str, FixedTaskConfig] = field(
        default_factory=lambda: {
            "target": FixedTaskConfig(target_name="target"),
            "tailgas": FixedTaskConfig(target_name="tailgas"),
        }
    )
    decoder_tasks: Dict[str, DecoderTaskConfig] = field(
        default_factory=lambda: {name: DecoderTaskConfig() for name in DECODER_CATEGORIES}
    )
    aux_task: AuxTaskConfig = field(default_factory=AuxTaskConfig)
    graph_label_mode: GraphLabelMode = "task_dict"
    cache_process_specs: bool = True
    process_id_column: str = "process_id"
    split_column: str = "split"
    normalize_y_edge: bool = False
    edge_all_reference_dir: str = ""
    edge_all_stream_dir: str = ""
    edge_all_processes: Optional[list] = None
    edge_all_split_mode: str = "seen_process"
    edge_struct_include_join_flags: bool = True
    edge_all_train_processes: Optional[list] = None
    edge_all_val_processes: Optional[list] = None
    edge_all_test_processes: Optional[list] = None
    edge_all_holdout_test_process: Optional[int] = None
    train_split_manifest_path: str = ""
    val_split_manifest_path: str = ""
    test_split_manifest_path: str = ""
    edge_all_cap_train_samples: Optional[int] = None
    edge_all_cap_val_samples: Optional[int] = None
    edge_all_cap_test_samples: Optional[int] = None
    use_rho_target: bool = True
    use_h_target: bool = True
    use_volume_flow_target: bool = False
    use_enthalpy_flow_target: bool = False
    molecular_weight: Dict[str, float] = field(default_factory=dict)
    log_v3_graph_stats_once: bool = False
    edge_target_columns_mode: str = "auto"
    edge_target_columns: Optional[list[str]] = None
    stream_row_cache_size: int = 128
    downcast_main_frame_numeric: bool = True
    species_order: list[str] = field(default_factory=list)
    node_energy_consistency_cache_path: str = ""
    excluded_graph_samples_path: str = ""
    known_feed_condition: dict[str, Any] = field(default_factory=dict)
    hx_pair_relation: dict[str, Any] = field(default_factory=dict)
    feed_head_conditioning: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    experiment_name: str = "process_surrogate"
    seed: int = 42
    device: str = "cpu"
    eval_only: bool = False
    model_config: str = "configs/model/process_encoder.yaml"
    train_config: str = "configs/train/process_train.yaml"
    data_config: str = "configs/data/process_data.yaml"
    output_dir: str = "outputs/process_surrogate"
    save_dir: str = "outputs/process_surrogate/checkpoints"
    log_dir: str = "outputs/process_surrogate/logs"
    resume_path: str = ""
    project_root: Path = field(default_factory=lambda: Path("."))
    model: ModelYamlConfig = field(default_factory=ModelYamlConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)




