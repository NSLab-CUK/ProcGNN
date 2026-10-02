# Method ablation (one-factor-at-a-time)

## Purpose

After fixing encoder **capacity/depth** (see `configs/experiment/capacity_ablation/`), verify which **methodology** choices help on the edge_all surrogate. This is **not** a full factorial study: each run changes **one** model setting relative to the shared baseline.

- Primary metric: **`target_v4_macro_r2`** (macro mean of per-target test R²).
- Secondary: species mean R² (H2 / CO2 / H2O), legacy `target_h2_r2` / `tailgas_co2_r2`, `edge_all_r2`.
- Training loss, scaler, splits, and target mapping match `process_surrogate_edge_all_v3.yaml`.

## Baseline resolved config

Loaded from `exp_B0_baseline.yaml` + `configs/model/process_encoder.yaml` overrides.

Snapshot: [`baseline_resolved.json`](baseline_resolved.json)

| Field | Value |
|-------|-------|
| hidden_dim / num_layers / attn_hidden_dim | 512 / 5 / 512 |
| role_emb_dim / unit_emb_dim | 96 / 64 |
| use_hx_role_embedding | false |
| input/diff/update/final_mlp_layers | 2 / 2 / 2 / 2 |
| diff_mode / fusion_mode | concat / concat |
| global_pool | concat_set2set |
| use_edge_features | true |
| initial_residual_mode / alpha | per_layer / 0.05 |
| layer_residual_mode / alpha / norm | interp / 0.5 / false |
| use_final_projection | true |

Capacity ablation winner: when `outputs/process_kfold_capacity_ablation/summary/experiment_summary.csv` exists, pick the best `mean_target_v4_macro_r2` experiment and update `exp_B0_baseline.yaml` overrides if it differs from the table above.

## Experiment catalog

### Default runs (scheduled)

| Experiment | Group | Change vs baseline |
|------------|-------|-------------------|
| exp_B0_baseline | baseline | (none) |
| exp_R1_layer_residual_none | layer_residual | `layer_residual_mode: none` |
| exp_I0_initial_none | initial_residual | `initial_residual_mode: none` |
| exp_I2_initial_final | initial_residual | `initial_residual_mode: final` |
| exp_I3_initial_both | initial_residual | `initial_residual_mode: both` |
| exp_D1_diff_add | differential | `diff_mode: add` |
| exp_EF1_edge_features_off | edge_features | `use_edge_features: false` **(see blocker below)** |
| exp_F1_fusion_sum | fusion | `fusion_mode: sum` |
| exp_F2_fusion_mean | fusion | `fusion_mode: mean` |
| exp_F3_fusion_weighted | fusion | `fusion_mode: weighted` |
| exp_G1_global_mean | global_pool | `global_pool: mean` |
| exp_G2_global_sum | global_pool | `global_pool: sum` |
| exp_G3_global_max | global_pool | `global_pool: max` |
| exp_G4_global_attention | global_pool | `global_pool: attention` |

### Baseline-equivalent (YAML exists, **not** scheduled)

| Experiment | Reason |
|------------|--------|
| exp_R2_layer_residual_interp | Same as baseline (`interp`, α=0.5) |
| exp_I1_initial_per_layer | Same as baseline (`per_layer`, α=0.05) |
| exp_D0_diff_concat | Same as baseline (`concat`) |
| exp_F0_fusion_concat | Same as baseline |
| exp_G0_global_concat_set2set | Same as baseline |
| exp_EF0_edge_features_on | Same as baseline |

### Optional (`--include-optional`)

| Experiment | Group | Change |
|------------|-------|--------|
| optional/exp_R3_layer_residual_interp_norm | layer_residual | `layer_residual_norm: true` |
| optional/exp_P1_final_projection_off | final_projection | `use_final_projection: false` |

## Code / design notes

### edge_all readout path

- **EdgeDecoder** uses `local_node_embeddings`, `edge_struct_attr`, and **`global_embedding`** (`process_surrogate.py`).
- **initial_residual**: `per_layer` injects α·h0 each layer; `final`/`both` add h0 to local embeddings before pooling — affects edge decoder inputs.
- **final_projection**: applied to `node_embeddings` (with global context), not directly to `local_node_embeddings`; impact on edge_all may be **small** but global path still changes.

### differential (`diff_mode`)

This compares **how delta features are fused** (`concat` vs `add`). It is **not** “no differential encoding” (that would need a code change).

### edge features (`use_edge_features`)

- When `true`, edge stream embeddings feed **attention** (`edge_proj` in `DirectedAttentionAggregation`).
- **EdgeDecoder** still uses `edge_struct_attr` from the batch.
- When `false`, `edge_embeddings=None` in attention; **current code raises** if `use_edge_decoder=true` and `use_edge_features=false` (`ProcessSurrogateModel`). **exp_EF1 will fail at model init** until that guard is relaxed or edge_decoder is disabled.

### fusion (`fusion_mode`)

- Options: `concat`, `sum`, `mean`, `weighted` (learnable gate — extra parameters).
- **Forward-only / backward-only** branches are **not** configurable via yaml.

### global_pool

- `concat_set2set` readout dim = **2×hidden_dim**; others use `hidden_dim` (`global_embedding_dim()`).
- `final_projection` input dim follows `global_embedding_dim` — should remain consistent when pooling changes.

### Not in scope (config-only impossible today)

- True **no-differential** encoding
- **Forward-only / backward-only** message passing
- **Attention → mean aggregation** swap (separate from `global_pool: mean`)
- **hidden_dim / num_layers / attn_hidden_dim** (capacity ablation only)

## Output layout

```
outputs/process_kfold_method_ablation/
  <EXP_NAME>/
    Process<id>/
      run.log, run.stderr.log, run_meta.json
      fold_01/
        process_kfold_P<id>_F01-<timestamp>/
        plots/                    # aggregator convergence plots
      plots/                      # 5-fold mean convergence
  summary/
    experiment_summary.csv
    experiment_process_summary.csv
    plots/
```

## Commands

### Regenerate configs (after registry edit)

```bash
python scripts/generate_method_ablation_configs.py
```

### 1) Dry-run

```bash
python scripts/run_method_ablation_kfold.py \
  --process-ids 1-10 \
  --k-folds 5 --max-epochs 100 --seed 42 --device cuda \
  --output-root outputs/process_kfold_method_ablation \
  --dry-run
```

### 2) Pilot (Process 1, 5, 9)

```bash
python scripts/run_method_ablation_kfold.py \
  --process-ids 1 5 9 \
  --k-folds 5 --max-epochs 100 --seed 42 --device cuda \
  --output-root outputs/process_kfold_method_ablation \
  --skip-existing --continue-on-error
```

### 3) Aggregate + plots

```bash
python scripts/aggregate_method_ablation_results.py \
  --root outputs/process_kfold_method_ablation --make-plots
```

### 4) Full 10 processes

```bash
python scripts/run_method_ablation_kfold.py \
  --process-ids 1-10 \
  --k-folds 5 --max-epochs 100 --seed 42 --device cuda \
  --output-root outputs/process_kfold_method_ablation \
  --skip-existing --continue-on-error
```

### Optional experiments

```bash
python scripts/run_method_ablation_kfold.py \
  --process-ids 1 5 9 --include-optional \
  --k-folds 5 --max-epochs 100 --seed 42 --device cuda \
  --output-root outputs/process_kfold_method_ablation \
  --skip-existing --continue-on-error
```

## Convergence plots

| Level | Path | Files |
|-------|------|-------|
| Training (existing) | `<run_dir>/plots/` | `loss_total_train_val.png`, `val_r2_*.png`, … |
| Fold (aggregator) | `fold_<K>/plots/` | `convergence_loss.png`, `convergence_r2.png`, `convergence_mae_rmse.png` |
| Process (aggregator) | `Process<P>/plots/` | `convergence_*_mean_std.png` |
| Summary | `summary/plots/` | `method_ablation_rank_*.png`, `ablation_delta_from_baseline.png`, heatmap, group summary, species R², convergence diagnostics |

Diagnostics: `summary/convergence_diagnostics.csv` (reference thresholds: `SLOPE_EPS=1e-4`, `OVERFIT_GAP_THRESHOLD=0.1`, `MIN_EPOCHS=20`).

- Many **`still_improving`** → consider `max_epochs` > 100.
- Many **`overfitting`** → review dropout, weight decay, early stopping, residuals.

## Convergence status rules (reference only)

| Status | Meaning |
|--------|---------|
| converged | Last 10 epochs: flat val R² and stable val loss |
| still_improving | Val macro R² still rising |
| overfitting | Train improves, val worsens |
| unstable | High val R² variance |
| insufficient_data | Too few val metrics |

Do not use these alone as early-stop policy.
