# Capacity / depth ablation (encoder size)

## Purpose

Before methodology ablations (differential, edge features, fusion, global pooling), fix a reasonable **encoder capacity** (hidden size, depth, attention MLP, embeddings, block MLP depth). Each process (Process 1–10) is trained with **5-fold** splits; outputs are isolated per experiment name.

Baseline matches `process_surrogate_edge_all_v3.yaml` + `configs/model/process_encoder.yaml` defaults:

| Key | Value |
|-----|-------|
| hidden_dim | 512 |
| num_layers | 5 |
| attn_hidden_dim | 512 |
| role_emb_dim / unit_emb_dim | 96 / 64 |
| use_hx_role_embedding | false |
| input/diff/update/final_mlp_layers | 2 / 2 / 2 / 2 |
| diff_mode / fusion_mode | concat / concat |
| global_pool | concat_set2set |
| use_edge_features | true |
| initial_residual_mode / alpha | per_layer / 0.05 |
| layer_residual_mode / alpha | interp / 0.5 |

Training: `max_epochs=100`, `seed=42`, `k_folds=5`, existing `data/splits/process_kfold/`.

## Experiment list

| ID | Group | Overrides (`model`) |
|----|-------|---------------------|
| exp_L3 | layers | num_layers: 3 |
| exp_L4 | layers | num_layers: 4 |
| exp_L5 | layers | num_layers: 5 (baseline depth) |
| exp_L6 | layers | num_layers: 6 |
| exp_H384 | hidden | hidden_dim, attn_hidden_dim: 384 |
| exp_H448 | hidden | 448 |
| exp_H512 | hidden | 512 (baseline width) |
| exp_A256 | attention MLP | hidden_dim: 512, attn_hidden_dim: 256 |
| exp_A384 | attention MLP | attn_hidden_dim: 384 |
| exp_E_small | embeddings | role_emb_dim: 64, unit_emb_dim: 48 |
| exp_E_base | embeddings | 96 / 64 (baseline) |
| exp_E_large | embeddings | 128 / 96 |
| exp_M_base | MLP depth | all block MLPs: 2 |
| exp_M_shallow | MLP depth | all: 1 |
| exp_M_update_deep | MLP depth | update_mlp_layers: 3 |
| exp_M_input_deep | MLP depth | input_mlp_layers: 3 |

**Note:** `exp_H512` and a hypothetical `exp_A512` are identical (`hidden_dim=512`, `attn_hidden_dim=512`). Only `exp_H512` is defined; the runner dedupes if both names are passed.

Layer sweeps fix other dims at baseline; hidden/attention sweeps use `num_layers=5` via base model yaml unless overridden in that experiment file.

## Output layout

```
outputs/process_kfold_capacity_ablation/
  <experiment_name>/
    Process<id>/
      run.log
      run.stderr.log
      run_meta.json
      fold_01/
        process_kfold_P<id>_F01/
          metrics.json, train_val_history.json, metrics_per_epoch.csv
          plots/   # from train_process_surrogate.py
        plots/     # from aggregate_capacity_ablation_results.py
      plots/       # 5-fold mean convergence
  summary/
    experiment_process_summary.csv
    experiment_summary.csv
    convergence_diagnostics.csv
    plots/
```

## Primary metric

- **target_v4_macro_r2** — mean of per-target test R² (`__ALL_macro_split__` in `test/target_metrics_v4_summary.csv`).
- **main_r2 = target_v4_macro_r2**.
- **legacy_answer_fraction_r2** is an auxiliary diagnostic computed on answer-edge component fractions
  (`Frac_H2`, `Frac_CO2`, etc.). It is intentionally separate from actual target amount performance
  (`Mole_Flow * Frac_*`) and should not be read as the main capacity-ablation R².
- **target_v4_relative_accuracy_score** is an optional auxiliary score
  `clip(1 - RMSE / mean(abs(true)), 0, 1)`. It is less sensitive to target variance than R², but it
  is not R² and should be reported separately.
- Legacy: **target_h2_r2**, **tailgas_co2_r2**, **edge_all_r2** (still logged).

Early stopping in training remains default **`val_r2`** (unchanged).

## Existing training plots (per run)

`train_process_surrogate.py` already writes under `<run_dir>/plots/`:

| File | Content |
|------|---------|
| loss_total_train_val.png | train vs val total loss |
| loss_train_components.png | per-task train losses |
| loss_val_components.png | per-task val losses |
| val_r2_target_h2_tailgas_co2.png | legacy H2 / CO2 / mean answer R² |
| val_mae_answer_edges.png | target / tailgas MAE |
| val_edge_all_mae_mse_norm.png | edge_all MAE/MSE (normalized) |

Also: `train_val_history.json`, `metrics_per_epoch.csv`, `metrics_main_targets.json`.

## Aggregator plots (`--make-plots`, default)

| Level | Path | Files |
|-------|------|-------|
| Fold | `.../fold_<K>/plots/` | convergence_loss, convergence_r2, convergence_mae_rmse, learning_diagnostics (if lr/grad columns exist) |
| Process | `.../Process<P>/plots/` | convergence_loss_mean_std, convergence_target_v4_r2_mean_std, convergence_h2_co2_h2o_r2_mean |
| Summary | `summary/plots/` | experiment_rank_*, heatmap, layer/hidden/attention sweeps, embedding/MLP comparisons |

**learning_diagnostics:** only if `train_lr` or grad norm is in `metrics_per_epoch.csv`; otherwise documented as *not available*.

## Convergence diagnosis (reference only)

`summary/convergence_diagnostics.csv` uses:

| Constant | Value |
|----------|-------|
| SLOPE_EPS | 1e-4 |
| OVERFIT_GAP_THRESHOLD | 0.1 |
| MIN_EPOCHS_FOR_DIAGNOSIS | 20 |

Statuses: `converged`, `still_improving`, `overfitting`, `unstable`, `insufficient_data`. Do **not** use these alone to stop training.

- Many **`still_improving`** → consider `max_epochs` > 100.
- Many **`overfitting`** → check dropout, weight decay, early stopping, residual settings.

## Recommended run order

### 1) Dry-run (all commands)

```bash
python scripts/run_capacity_ablation_kfold.py \
  --process-ids 1-10 \
  --k-folds 5 \
  --max-epochs 100 \
  --seed 42 \
  --device cuda \
  --output-root outputs/process_kfold_capacity_ablation \
  --dry-run
```

### 2) Pilot — Process 1, 5, 9

```bash
python scripts/run_capacity_ablation_kfold.py \
  --process-ids 1 5 9 \
  --k-folds 5 \
  --max-epochs 100 \
  --seed 42 \
  --device cuda \
  --output-root outputs/process_kfold_capacity_ablation \
  --skip-existing \
  --continue-on-error \
  --tee-logs
```

`--tee-logs`: 학습 중 **loss / val R²·MAE** 출력을 터미널에도 보이게 하면서 `Process*/run.log`에도 저장합니다.  
(기본값은 파일만 저장 → 터미널에는 `[run] exp_…` 라인만 보임.)

이미 돌아가는 job 로그만 보려면:

```bash
tail -f outputs/process_kfold_capacity_ablation/exp_L3/Process5/run.log
```

### 3) Aggregate + plots

```bash
python scripts/aggregate_capacity_ablation_results.py \
  --root outputs/process_kfold_capacity_ablation \
  --make-plots
```

### 4) Full 10 processes

```bash
python scripts/run_capacity_ablation_kfold.py \
  --process-ids 1-10 \
  --k-folds 5 \
  --max-epochs 100 \
  --seed 42 \
  --device cuda \
  --output-root outputs/process_kfold_capacity_ablation \
  --skip-existing \
  --continue-on-error
```

### Partial experiments example

```bash
python scripts/run_capacity_ablation_kfold.py \
  --process-ids 1 5 9 \
  --experiments exp_L4 exp_L5 exp_L6 exp_H384 exp_H512 exp_A256 exp_A384 \
  --k-folds 5 --max-epochs 100 --seed 42 --device cuda \
  --output-root outputs/process_kfold_capacity_ablation
```

## Direct k-fold command (single experiment)

```bash
python scripts/run_process_kfold_experiments.py \
  --base-config configs/experiment/capacity_ablation/exp_L5.yaml \
  --process-ids 8 \
  --k-folds 5 \
  --max-epochs 100 \
  --output-root outputs/process_kfold_capacity_ablation/exp_L5 \
  --seed 42 \
  --device cuda
```

## Notes

- Do **not** edit `process_surrogate_edge_all_v3.yaml`; only files under `capacity_ablation/`.
- Loss, optimizer, scaler, and target mapping stay aligned with the v3 edge_all baseline.
- This sweep does **not** change diff_mode, fusion_mode, global_pool, or use_edge_features.
