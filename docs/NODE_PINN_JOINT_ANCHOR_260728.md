# Node PINN Joint Supervised Anchor (2026-07-28)

> Default model update: E0/E1/E2 now all use the 16D property stream role
> embedding (`property_stream_role.enabled=true`). Their PI branch input is
> 2067D rather than the previous 2051D Role16-disabled input.

## 1. 문제 확인

기존 `sample_hybrid_target_edge_step_pi`의 sample당 update 순서는 다음과 같았다.

1. non-target canonical edge group 전체의 macro mean supervised update 1회
2. target canonical edge group별 supervised update
3. Node PINN loss만 사용하는 별도 update 1회

세 번째 update에는 edge supervision이 없었다. 따라서 Node PINN schedule이 0에서 양수로
전환될 때, Mass Flow와 Enthalpy 예측을 supervised optimum에 유지하는 직접적인 항이 없었다.

`outputs/0727_allproc5fold_m260716_tw40_pinnsafe`의 실제 validation 결과:

| Fold | Epoch 5 Mass Flow R2 | Epoch 6 Mass Flow R2 | Epoch 5 Enthalpy R2 | Epoch 6 Enthalpy R2 |
|---|---:|---:|---:|---:|
| 1 | 0.4851 | 0.3885 | -0.1272 | -0.5297 |
| 2 | 0.5342 | 0.3933 | -0.1070 | -0.5409 |
| 3 | 0.5071 | 0.3467 | -0.1128 | -0.4764 |
| 4 | 0.5018 | 0.3354 | -0.1140 | -0.6510 |
| 5 | 0.5412 | 0.1935 | -0.0953 | -0.6103 |

PINN multiplier가 처음 0.5가 되는 epoch 6에 5개 fold 모두 같은 붕괴가 발생했다.

## 2. 구현

새 설정:

```yaml
node_pinn_optimization:
  update_mode: joint_with_supervised_anchor
  supervised_anchor_weight: 1.0
  node_outer_weight: 0.05
  anchor_apply_target_weight: false
  log_gradient_diagnostics: false
```

기본값은 `update_mode: separate`이므로 기존 YAML과 checkpoint 동작은 유지된다.

Joint mode의 Node 단계:

```text
fresh forward 1회
  -> 모든 valid canonical edge group의 기존 edge loss 계산
  -> group별 loss의 macro mean = L_anchor
  -> 기존 Node PINN 계산 = L_node
  -> L_joint = eta_anchor * L_anchor
             + gamma_epoch * eta_node * L_node
  -> backward 1회
  -> optimizer.step 1회
```

실제 코드에서는 기존 schedule이 Node PINN 내부 가중치에 이미 반영된
`scheduled_node_loss = gamma_epoch * L_node`를 만든다. 따라서 joint 조합은 다음처럼
계산해 schedule 중복 적용을 방지한다.

```text
L_joint = eta_anchor * L_anchor + eta_node * scheduled_node_loss
```

### Anchor 정의

- 기존 `compute_single_edge_pinn_loss()`를 그대로 재사용한다.
- canonical edge group별 loss를 먼저 계산한다.
- 모든 valid group을 동일 비중으로 macro mean한다.
- 기본적으로 target weight 40을 적용하지 않고 neutral weight 1을 사용한다.
- `anchor_apply_target_weight: true`일 때만 기존 edge weight를 적용한다.

### Skip 규칙

- schedule multiplier가 0이면 Node/joint 추가 update를 수행하지 않는다.
- valid supervised anchor가 없으면 joint 전체를 skip한다.
- valid Node PINN 항이 없거나 non-finite이면 joint 전체를 skip한다.
- anchor-only 또는 PINN-only fallback update는 수행하지 않는다.

## 3. 실험 설정

| 실험 | Config | Node update | `lambda_h` | Node outer weight |
|---|---|---|---:|---:|
| E0 | `model_260716_pi_tw40_legacy_fracfocus_f01.yaml` | separate PINN-only | 1e-11 | 1.0 |
| E1 | `model_260716_pi_tw40_joint_anchor_e1.yaml` | joint anchor + PINN | 1e-11 | 0.05 |
| E2 | `model_260716_pi_tw40_joint_anchor_e2_h002.yaml` | joint anchor + PINN | 0.02 | 0.05 |

나머지 모델 구조, sampler, target weight, Node PINN residual 수식, energy consistency
filter, optimizer, scheduler, validation sampling은 E0와 동일하다.

## 4. 추가 로그

다음 값이 batch/epoch metrics에 기록된다.

- `node_pinn_supervised_anchor_raw`
- `node_pinn_supervised_anchor_weighted`
- `node_pinn_raw`
- `node_pinn_outer_weighted`
- `node_pinn_joint_loss`
- `node_pinn_schedule_multiplier`
- `node_pinn_outer_weight`
- `joint_node_optimizer_step_count`
- `joint_node_skip_count`
- `collapse_mass_flow_pred_true_ratio`
- `collapse_enthalpy_pred_true_ratio`

Validation에는 전체 valid edge를 모아 다음 median-scale diagnostic을 기록한다.

```text
median(abs(prediction)) / median(abs(ground truth))
```

- `collapse_mass_flow_pred_true_ratio`
- `collapse_enthalpy_pred_true_ratio`

1에서 멀어지며 0에 가까워지면 예측 scale collapse로 해석한다.

`log_gradient_diagnostics: true`일 때만 anchor, Node, joint 각각의 total/shared
encoder/Mass Flow head/Enthalpy head gradient norm을 추가로 계산한다. 일반 학습에서는
추가 autograd 비용을 피하기 위해 false를 사용한다.

## 5. 검증 결과

- Python compile: 통과
- joint objective 및 schedule 단일 적용: 통과
- supervised + Node gradient 합성: 통과
- 기존 config의 기본 `separate` 모드: 통과
- 신규 E1/E2 config load: 통과
- 신규 unit test: 3 passed
- 기존 Node PINN 핵심 계산 test: 21 passed

전체 `test_node_balance_pi.py` 중 16개는 이 checkout에 존재하지 않는 오래된 A6 YAML을
참조하여 실패했다. 새 joint 구현에서 발생한 계산 실패는 아니다.

## 6. Fold 1 실행 명령

### E0: 기존 PINN-only baseline

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_process_kfold_experiments.py \
  --base-config configs/experiment/pinn/model_260716_pi_tw40_legacy_fracfocus_f01.yaml \
  --merged-csv data/datasets_v3/process_main_merged.csv \
  --process-ids 1 2 3 4 5 6 7 8 9 10 \
  --joint-all-processes --only-folds 1 --max-epochs 30 \
  --monitor-metric val_target_mean_r2 --monitor-mode max \
  --output-root outputs/0728_nodejoint_e0_f01 \
  --force-reevaluate --skip-startup-debug
```

### E1: joint anchor, legacy Enthalpy weight

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_process_kfold_experiments.py \
  --base-config configs/experiment/pinn/model_260716_pi_tw40_joint_anchor_e1.yaml \
  --merged-csv data/datasets_v3/process_main_merged.csv \
  --process-ids 1 2 3 4 5 6 7 8 9 10 \
  --joint-all-processes --only-folds 1 --max-epochs 30 \
  --monitor-metric val_target_mean_r2 --monitor-mode max \
  --output-root outputs/0728_nodejoint_e1_f01 \
  --force-reevaluate --skip-startup-debug
```

### E2: joint anchor, Enthalpy weight 0.02

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_process_kfold_experiments.py \
  --base-config configs/experiment/pinn/model_260716_pi_tw40_joint_anchor_e2_h002.yaml \
  --merged-csv data/datasets_v3/process_main_merged.csv \
  --process-ids 1 2 3 4 5 6 7 8 9 10 \
  --joint-all-processes --only-folds 1 --max-epochs 30 \
  --monitor-metric val_target_mean_r2 --monitor-mode max \
  --output-root outputs/0728_nodejoint_e2_f01 \
  --force-reevaluate --skip-startup-debug
```

우선 판정 구간은 epoch 5-8이다. E1이 epoch 6에서 collapse를 막는지 먼저 확인한 뒤,
E2의 Enthalpy supervision 강화가 다른 성분을 훼손하지 않는지 비교한다.
