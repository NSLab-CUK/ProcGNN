# Mass Flow Dual-Space Loss Experiment (260730)

## 1. Experiment Objective

This experiment changes only the supervised Mass Flow loss. The E2 model
architecture, 2,000-sample epoch sampler, target weight, PINN terms and
schedule, optimizer, learning rate, split, and checkpoint monitor remain
unchanged.

The four Fold-1 runs test whether a small physical-space correction improves
high-flow underprediction without sacrificing the existing transformed-space
fit.

## 2. Canonical Mass Transform

The active E2 configuration predicts Mass Flow in scaled-log space:

```text
z = 2 * log(max(m, 0) + 1e-8)
m_hat = max(exp(z_hat / 2) - 1e-8, 0)
```

The same canonical inverse function is used by:

- the physical Mass Flow auxiliary loss;
- node Mass/Component/Atom PINN calculations;
- validation and test metrics;
- prediction export.

There is no detached path between the physical loss and the Mass head. The
physical loss therefore sends gradients through the inverse transform, Mass
head, shared decoder, FlowGNN, and input encoders.

The exponential input is limited only by the floating-point overflow boundary.
It is not clipped to P90, P95, P99, or the observed training maximum.

## 3. Exact Loss

For one valid Mass Flow target:

```text
L_log = SmoothL1(z_hat, z_true)

r_phys = (m_hat - m_true) / s_mass

L_phys = Huber(r_phys, 0; delta=1)

L_mass(epoch) = 1.0 * L_log + lambda_phys(epoch) * L_phys
```

Both terms use the same valid Mass feature mask, the same edge grouping, and
the existing target/non-target edge weighting. They enter the existing
feature-mean supervised objective at the original Mass feature position.

The physical residual has a safety clip of `[-10, 10]`. This limits only the
normalized residual used by Huber; it does not clamp the prediction or target.

## 4. Physical Scale

Default:

```text
s_mass = standard deviation of valid physical Mass_Flow values
         from the training split only
```

Supported modes:

| Mode | Definition |
| --- | --- |
| `train_std` | Training-split physical standard deviation |
| `train_iqr` | Training-split physical P75 minus P25 |
| `fixed` | Positive finite value supplied by `fixed_scale` |

Validation and test values never contribute to this scale. A non-finite or
near-zero scale raises an error instead of being silently replaced.

The resolved value and provenance are written to:

- `mass_dual_space_loss_metadata.json`;
- `y_edge_scaler.pt`;
- `best.pt` and `last.pt`;
- the resolved config snapshot inside the checkpoint.

Checkpoint evaluation restores the persisted value.

## 5. Physical Weight Schedule

Epoch numbers are 1-based:

```text
epoch < 3:       lambda_phys = 0
epoch = 3:       lambda_phys = 0
3 < epoch < 7:   linear interpolation
epoch >= 7:      lambda_phys = configured maximum
```

For D2 (`max_weight=0.10`):

| Epoch | Weight |
| ---: | ---: |
| 1, 2 | 0.000 |
| 3 | 0.000 |
| 4 | 0.025 |
| 5 | 0.050 |
| 6 | 0.075 |
| 7+ | 0.100 |

This schedule is independent of the existing PINN schedule.

## 6. Controlled Configurations

All configs are under
`configs/experiment/pinn/mass_dual_space_260730/`.

| Run | Log weight | Physical max weight | Physical term |
| --- | ---: | ---: | --- |
| D0 | 1.0 | 0.00 | Disabled baseline |
| D1 | 1.0 | 0.05 | Epoch 3-7 warmup |
| D2 | 1.0 | 0.10 | Epoch 3-7 warmup |
| D3 | 1.0 | 0.20 | Epoch 3-7 warmup |
| D4 | 1.0 | 0.40 | Higher physical-space pressure |
| D5 | 1.5 | 0.00 | Log-space-only weight search |
| D6 | 2.0 | 0.00 | Log-space-only weight search |
| D7 | 3.0 | 0.00 | High log-space-only stress test |
| D8 | 1.5 | 0.10 | Moderate combined weighting |
| D9 | 2.0 | 0.10 | Higher log plus moderate physical |
| D10 | 2.0 | 0.20 | High combined weighting |

D0-D4 isolate the physical-space weight while keeping the transformed/log
weight fixed. D0/D5/D6/D7 isolate the transformed/log weight with the
physical term disabled. D8-D10 test whether the two useful axes remain
compatible when applied together.

Shared sampler:

| Component | Samples per epoch |
| --- | ---: |
| Base sampler | 1,000 |
| Target Frac_CO hard | 400 |
| Target Mass_Flow hard | 400 |
| Target Frac_CH4 hard | 200 |
| Total | 2,000 |

No new P90/P95/P99 sampler, edge oversampling, label-density weight,
asymmetric penalty, or alternate head was added.

## 7. Diagnostics

The compact training line now reports:

```text
mass=log...+w...*phys...=total...
```

Per-epoch metrics also contain:

- `mass_log_loss_mean`;
- `mass_physical_loss_mean`;
- `mass_physical_weight_mean`;
- `mass_total_loss_mean`;
- mean predicted and true physical Mass Flow;
- separate log and physical gradient norms;
- log/physical gradient cosine;
- Mass head, shared decoder, FlowGNN, node encoder, and edge encoder norms.

The separate gradient diagnostics use `autograd.grad`, retain the live graph,
do not write `.grad`, and do not perform an optimizer step. They run once per
epoch and the value is carried into the epoch aggregate.

Existing evaluation already writes physical Mass metrics by property and
`mass_flow_tail_metrics.csv` for train-threshold P90/P95/P99 subsets. These
tail thresholds remain evaluation diagnostics only.

## 8. Fold-1 Commands

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 python scripts/train_process_surrogate.py --config configs/experiment/pinn/mass_dual_space_260730/d0_mass_dual.yaml --skip-startup-debug
```

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 python scripts/train_process_surrogate.py --config configs/experiment/pinn/mass_dual_space_260730/d1_mass_dual.yaml --skip-startup-debug
```

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 python scripts/train_process_surrogate.py --config configs/experiment/pinn/mass_dual_space_260730/d2_mass_dual.yaml --skip-startup-debug
```

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 python scripts/train_process_surrogate.py --config configs/experiment/pinn/mass_dual_space_260730/d3_mass_dual.yaml --skip-startup-debug
```

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 python scripts/train_process_surrogate.py --config configs/experiment/pinn/mass_dual_space_260730/d4_mass_dual.yaml --skip-startup-debug
```

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 python scripts/train_process_surrogate.py --config configs/experiment/pinn/mass_dual_space_260730/d5_mass_dual.yaml --skip-startup-debug
```

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 python scripts/train_process_surrogate.py --config configs/experiment/pinn/mass_dual_space_260730/d6_mass_dual.yaml --skip-startup-debug
```

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 python scripts/train_process_surrogate.py --config configs/experiment/pinn/mass_dual_space_260730/d7_mass_dual.yaml --skip-startup-debug
```

```bash
CUDA_VISIBLE_DEVICES=8 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 python scripts/train_process_surrogate.py --config configs/experiment/pinn/mass_dual_space_260730/d8_mass_dual.yaml --skip-startup-debug
```

```bash
CUDA_VISIBLE_DEVICES=9 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 python scripts/train_process_surrogate.py --config configs/experiment/pinn/mass_dual_space_260730/d9_mass_dual.yaml --skip-startup-debug
```

```bash
CUDA_VISIBLE_DEVICES=10 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 python scripts/train_process_surrogate.py --config configs/experiment/pinn/mass_dual_space_260730/d10_mass_dual.yaml --skip-startup-debug
```

## 9. Verification

Passing checks:

- canonical transform/inverse round trip;
- physical loss gradient through the canonical inverse;
- 1-based schedule boundary and interpolation;
- `train_std`, `train_iqr`, and `fixed` scale resolution;
- disabled physical term compatibility;
- mask and train-scale behavior;
- diagnostic `autograd.grad` does not write parameter gradients;
- D0-D10 config loading and intended weight matrix;
- existing epoch sampler regression.

The broad edge-step test file has 42 passing tests and 7 unrelated failures.
Those failures reference historical YAML files that are no longer present in
the repository; they are not numerical or training-path failures.

## 10. Selection Rule

Do not select a winner using only pooled Mass Flow R2. Compare:

- Target and all-edge physical R2, MAE, and RMSE;
- P90/P95/P99 R2 and signed bias;
- edge-internal macro and pooled R2;
- prediction standard deviation on `P04_E009`, `P06_E013`, `P01_E015`,
  and `P09_E022`;
- log/physical gradient cosine and clipping behavior;
- all other target and all-edge properties for regressions.

After Fold 1, extend only the two stable top configurations to five folds.
