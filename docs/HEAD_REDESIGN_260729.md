# Hierarchical Reduced PI Head 설계 및 구현 보고서 (2026-07-29)

## 1. 결론

기존 A1의 `PIGroupedPropertyHead`를 직접 확장하지 않고 새
`hierarchical_reduced_pi` head를 추가했다. GNN backbone은 H=512, 5-layer
FlowGNN 그대로 유지한다. 새 head는 GNN이 이미 만든 node/global/edge
representation을 재사용하고, 직접 예측 항목을 11개로 줄인다.

권장 시작점은 **D2 (global 512, shared 256, branch 128)** 이다. D1은 용량
상한 비교군, D3는 branch 축소 비교군, D0는 global projection의 효과를
분리하는 비교군이다. D0-D3 사이에는 head 차원 외 학습 설정 차이가 없다.

## 2. 구현 전 A1 구조와 문제

### 2.1 실제 입력과 차원

- Node role: 5개 category를 96D embedding으로 표현
- Unit type: 19개 category를 64D embedding으로 표현
- Edge stream role: 10개 category를 64D embedding으로 표현
- Edge stream ID: stream key를 안정 hash한 뒤 vocab 512에 넣는 96D embedding
- Node hidden: 512D
- Global readout: concat Set2Set이므로 1024D
- A1 property descriptor:
  `src 512 + dst 512 + global 1024 + edge_struct 3 + Role16 = 2067`
- A1 `PIGroupedPropertyHead`: 2067D를 condition/fraction/flow branch가 각각
  직접 받는다.

Role16 자체의 파라미터 수는 작지만, head가 2067D descriptor를 branch별로
반복 소비한다. 또한 Role16은 작은 학습 집합에서는 shortcut이 될 수 있고,
평행 edge 구분에는 이미 512D edge embedding이 존재하므로 정보가 중복된다.

### 2.2 A1 산출물에서 확인한 증상

| 범위 | 성분 | R2 |
|---|---:|---:|
| Target | Frac_CH4 | 0.4859 |
| Target | Frac_CO | 0.3187 |
| Target | Mass_Flow | 0.6542 |
| Target | Vol_Flow | 0.5262 |
| All-edge | Frac_CO2 | -2.2004 |
| All-edge | Mass_Flow | 0.5385 |
| All-edge | Vol_Flow | 0.5334 |
| All-edge | Frac_O2 | 0.6211 |

이는 단순히 head가 작아서 생긴 문제로 보기 어렵다. 조건/조성/유량이 같은
큰 descriptor를 독립적으로 다시 해석하고, global 및 identity-like 정보가
강해 shared representation의 역할이 약한 구조가 더 직접적인 문제다.

## 3. 유지한 GNN backbone

### 3.1 Input encoding

Node input은 role embedding, unit embedding, 13D operating feature와 13D mask를
concat한 뒤 2-layer MLP로 512D가 된다. Edge input은 stream-role embedding
64D, hashed stream-ID embedding 96D, structural/operating feature 3D를 concat한
뒤 2-layer MLP로 512D가 된다.

### 3.2 Direction-aware FlowGNN

각 layer는 정방향과 역방향 attention을 별도로 계산한다.

```text
forward:  src -> dst 방향으로 attention aggregation
backward: dst -> src 방향으로 attention aggregation
```

Attention score에는 sender node, receiver node, edge embedding이 모두
사용된다. message에는 `sender hidden + edge embedding`이 들어간다. 두 방향
결과는 concat projection으로 다시 512D로 합쳐진다.

### 3.3 Residual과 초기 상태 재주입

각 layer의 새 출력 `h_new`에 대해:

```text
h_layer = 0.5 * h_new + 0.5 * h_previous
h_layer = h_layer + 0.05 * h_initial
```

이를 5개 layer에서 반복한다. `initial_residual_mode=per_layer`이므로 초기
node representation이 매 layer 다시 들어간다. `layer_residual_norm=false`라
이 보간 뒤 별도 LayerNorm은 적용하지 않는다.

## 4. 새 hierarchical head

### 4.1 Descriptor

```text
d_e = concat(h_src, h_dst, g_graph, e_gnn)
```

- `h_src`, `h_dst`: 각각 512D
- `e_gnn`: GNN attention에도 사용된 512D edge embedding
- D0의 `g_graph`: 1024D
- D1-D3의 `g_graph`: graph별로 1024 -> 512 projection한 뒤 edge에 gather
- Role16을 사용하지 않는다.
- raw `edge_struct_attr`를 다시 concat하지 않는다. 그 정보는 `e_gnn` 안에
  이미 인코딩되어 있다.

따라서 D0 descriptor는 2560D, D1-D3는 2048D다.

### 4.2 Shared residual decoder

```text
u_e = Dropout(GELU(LayerNorm(W_in d_e + b_in)))
v_e = Dropout(GELU(LayerNorm(W_res u_e + b_res)))
z_e = LayerNorm(u_e + v_e)
```

`W_in`의 출력 폭이 실제 `edge_decoder_hidden_dim`이다. 기존 all-process
runtime이 작은 decoder를 강제로 384로 바꾸던 legacy 보정은 이 새 head에는
적용하지 않는다. 따라서 D2/D3의 shared=256이 실제 runtime에서도 유지된다.

### 4.3 Level 1

동일한 shared latent `z_e`에서 세 branch가 나온다.

```text
condition: z_e -> branch -> [Temp, Pres]                  (2)
fraction:  z_e -> branch -> logits -> softmax(T=0.5)     (7)
mass:      z_e -> branch -> Mass_Flow                    (1)
```

Level-1 출력 순서:

```text
[Temp, Pres,
 Frac_H2O, Frac_H2, Frac_CH4, Frac_CO2, Frac_CO, Frac_O2, Frac_N2,
 Mass_Flow]                                               (10)
```

Fraction은 항상 0 이상이고 합이 1이다.

### 4.4 Level 2

```text
y1_stop = stop_gradient(y1)
Vol_Flow = MLP_10_to_64_to_1(y1_stop)
```

기본값은 전체 학습 동안 detach다. 따라서 Volume loss는 volume branch만
학습하고 condition/fraction/mass/shared/GNN으로 역전파되지 않는다. 선택적인
`level2_unfreeze_epoch`를 지정하면 해당 epoch부터 joint gradient를 허용할 수
있지만 D0-D3에서는 사용하지 않는다.

최종 직접 출력은 다음 11개뿐이다.

```text
Temp, Pres, 7 fractions, Mass_Flow, Vol_Flow
```

Density, Enthalpy, Mole_Flow는 direct output, supervised loss, 정식 target/all-edge
metric에서 제거했다. 원본 14D CSV는 scaler와 name-based target mapping을 위해
그대로 읽되, 학습 대상은 이름으로 선택한 11D다.

## 5. 유지한 node PINN

Energy PINN은 제거했다. 남은 loss는 Mass, Component, Atom 세 개다.

### 5.1 Mass balance

내부 node `v`에서:

```text
r_mass(v) = sum_in(m_e) - sum_out(m_e)
```

true incident flow scale로 정규화하고 Huber loss를 적용한다.

### 5.2 Component molar balance

Fraction은 mole fraction으로 해석한다.

```text
MW_mix(e) = sum_i x_ei * MW_i
n_e       = m_e / max(MW_mix(e), eps)
n_ei      = n_e * x_ei
r_comp(v,i)= sum_in(n_ei) - sum_out(n_ei)
```

non-reactive internal node에 적용한다.

### 5.3 Atom balance

species-atom matrix `A[i,a]`를 이용한다.

```text
n_atom(e,a) = sum_i n_ei * A[i,a]
r_atom(v,a) = sum_in(n_atom(e,a)) - sum_out(n_atom(e,a))
```

reactive internal node에 적용한다.

### 5.4 Weight와 안전장치

```text
L_node = 0.75 * L_mass + 1.5e-7 * L_component + 0.2 * L_atom
```

각 residual은 relative scale, `eps=1e-6`, `scale_floor=1`, Huber delta 0.5,
residual clip 10을 사용한다. Node PINN schedule은 유지한다.

```text
epoch 1-5: multiplier 0.0
epoch 6-7: multiplier 0.5
epoch 8+:  multiplier 1.0
```

## 6. 실제 데이터 물리 검증

10개 `Process_Streams.csv`의 2,499,969 row를 직접 검사했다.

- finite row: 2,499,967
- fraction sum 오차 <= 1e-6: 2,360,403 row (94.42%)
- 나머지 139,564 row는 정확히 zero Mass/Mole flow이며 fraction sum도 0
- active-flow row의 fraction 합은 사실상 1
- 저장 Mole_Flow와 `Mass_Flow / MW_mix`의 median relative error:
  `3.18e-6`
- p99 relative error: `1.87e-5`

따라서 7개 fraction은 실제로 mole-fraction closure를 만족하며 Component/Atom
수식과 맞는다. Zero-flow row는 flow threshold와 true incident scale mask로
보호된다.

## 7. 학습 update와 metric

A1의 sample-hybrid 전략을 유지한다.

```text
non-target canonical groups mean -> 1 update
각 target canonical group        -> 각 1 update
node PINN                         -> 1 separate update
```

Target weight는 40, batch size는 1이다. 실제 21-edge sample에서 non-target 1,
target 3, node 1로 optimizer step 5회와 fresh forward 5회가 일치했다.

정식 metric은 11개 direct output만 대상으로 한다.

- target property pooled R2
- target property equal-edge macro R2
- target property MAE/RMSE
- all-edge property R2/MAE/RMSE
- 기존 checkpoint monitor `val_target_mean_r2`

새 artifact `target_edge_internal_metrics_by_property.csv`에는 pooled R2와
edge-macro R2를 동시에 기록한다.

Gradient 진단은 non-target/target/node update별로 다음 모듈의 pre-clip,
post-clip norm과 clip ratio를 기록한다.

```text
Edge Encoder, GNN, Shared Edge Decoder,
Condition Head, Fraction Head, Mass Head, Volume Head
```

## 8. Ablation과 파라미터

| 설정 | Global | Descriptor | Shared | Branch | L2 | 전체 params | Edge head params |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 | 1024 | 2067 | branch direct | 128 | 없음 | 32,018,942 | 800,030 |
| D0 | 1024 | 2560 | 384 | 128 | 64 | 32,503,275 | 1,284,363 |
| D1 | 512 | 2048 | 384 | 128 | 64 | 32,831,467 | 1,612,555 |
| D2 | 512 | 2048 | 256 | 128 | 64 | 32,437,227 | 1,218,315 |
| D3 | 512 | 2048 | 256 | 96 | 64 | 32,412,043 | 1,193,131 |

D1은 1024->512 global projection 자체가 524,800 params라 D0보다 전체 head
파라미터가 많다. projection의 목적은 단순 파라미터 절약이 아니라 global
shortcut 압축이다.

FP32 parameter memory는 A1 122.14 MiB, D0 123.99 MiB, D1 125.24 MiB,
D2 123.74 MiB, D3 123.64 MiB다. parameter+gradient+Adam 상태를 단순 16
byte/parameter로 잡으면 각각 488.57, 495.96, 500.97, 494.95, 494.57 MiB다.

최대 40-edge graph에서 head 내부 activation의 해석적 하한은 약
D0 1.18 MiB, D1 1.07 MiB, D2 0.91 MiB, D3 0.84 MiB다. CUDA allocator,
kernel workspace와 전체 GNN activation은 로컬 GPU가 없어 실측하지 않았다.
서버 비교에서는 `torch.cuda.max_memory_allocated()`를 함께 기록해야 한다.

## 9. 호환성과 테스트

- 기존 A1 checkpoint는 descriptor/output schema가 달라 strict load에서 명확히
  실패한다. silent partial load는 허용하지 않는다.
- D0-D3 config parse/build test 통과
- 최종 output shape `(N_edge, 11)` test 통과
- fraction closure test 통과
- Level-2 detach 및 선택적 unfreeze gradient test 통과
- parallel edge가 서로 다른 edge embedding/latent를 갖는 test 통과
- Energy 없이 Mass/Component/Atom node PINN backward test 통과
- module gradient pre/post clipping test 통과
- target pooled R2와 edge-macro R2 분리 test 통과
- 새 전용 test: 12 passed
- 실제 D2 CLI `edge-all-forward-debug`: exit code 0, `(24,11)` 확인
- 실제 D2 full sample-hybrid one-batch: optimizer 5, forward 5, Energy 0

광범위 회귀 suite에서는 관련 assertion 45개가 통과했다. 16개는 코드 실패가
아니라 저장소에서 이미 삭제된 legacy A6 YAML fixture를 test가 참조하여
`FileNotFoundError`로 실행되지 않았다.

## 10. 실행 명령

### D0: global 1024, shared 384, branch 128

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 python scripts/train_process_surrogate.py --config configs/experiment/pinn/head_260729/d0_global1024_shared384_branch128.yaml --skip-startup-debug
```

### D1: global 512 projection, shared 384, branch 128

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 python scripts/train_process_surrogate.py --config configs/experiment/pinn/head_260729/d1_global512_shared384_branch128.yaml --skip-startup-debug
```

### D2: global 512 projection, shared 256, branch 128 (권장)

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 python scripts/train_process_surrogate.py --config configs/experiment/pinn/head_260729/d2_global512_shared256_branch128.yaml --skip-startup-debug
```

### D3: global 512 projection, shared 256, branch 96

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 python scripts/train_process_surrogate.py --config configs/experiment/pinn/head_260729/d3_global512_shared256_branch96.yaml --skip-startup-debug
```

모든 config는 30 epoch, early stopping patience 5, fold 1의 동일 60/20/20
manifest, epoch당 2% base + hard fill 총 2000개, validation 1000개, target
weight 40을 사용한다.
