# F2 Independent Feed Nodes 구현 보고서

작성일: 2026-07-30

## 1. 결론

F2를 다음 구조로 구현했다.

```text
CH4_Flow   -> V_FEED_CH4   -> 실제 CH4 유입 장치
AIR_Flow   -> V_FEED_AIR   -> 실제 AIR 유입 장치
WATER_Flow -> V_FEED_WATER -> 실제 WATER 유입 장치
```

연결 목적지는 node 이름 추측이나 P03 전용 규칙으로 결정하지 않는다.
기존 canonical input edge의 `src_node_raw`, `src_node`, `dst_node` metadata를
사용하여 각 feed의 실제 목적지를 찾는다.

F2의 새 node와 edge는 다음 의미만 갖는다.

```text
FlowGNN message passing: 포함
Set2Set global pooling: 포함
supervised edge loss: 제외
target edge loss: 제외
evaluation metric: 제외
prediction export: 제외
hard sampler edge 후보: 제외
node PINN incidence/residual: 제외
```

F1 `V_INPUT node operating feature` 경로는 바꾸지 않았다.

## 2. 실제 feed 데이터

### 2.1 값의 의미와 단위

세 feed 값은 공정 실행 전에 주어지는 simulation operating input이다.
Process specification에서 source node의 operating reference로 읽으며,
downstream 결과나 target edge Mass Flow에서 역산하지 않는다.

원본 CSV에는 단위 문자열이 별도로 기록되어 있지 않다. 다만 270,003개
raw feed 값과 source stream의 `Mole_Flow`를 대조했을 때 269,997개
(99.9978%)가 `rtol=1e-6, atol=1e-6`에서 일치했다. 따라서 F2 값의 단위는
각 `Process_Streams.csv`의 `Mole_Flow`와 동일하다. 파일 근거 없이 별도
단위명을 하드코딩하지 않았다.

일치하지 않은 6개 값은 다음 두 sample에만 집중되어 있다.

| Process | ID | 불일치 feed |
|---|---:|---|
| P01 | 9325 | CH4, AIR, WATER |
| P10 | 4233 | CH4, AIR, WATER |

F2는 알려진 operating input이라는 실험 목적에 따라 raw main-column 값을
사용한다. 위 두 sample은 source stream 결과표와 raw input 사이의 기존
데이터 불일치이므로 성능 분석 시 별도로 추적해야 한다.

P03의 `P03_E014`는 downstream `OUT` stream이다. Feed 값과 이 edge의
`Mass_Flow`가 정확히 같은 sample은 10,000개 중 0개다. 즉 target 값을
복사한 입력 누출은 아니다.

### 2.2 공정별 column과 분포

표의 값은 `min / mean / std / max` 순서다.

| Process | CH4 | AIR | WATER |
|---|---|---|---|
| P01 | `CH4_Flow`: 5.53 / 363.84 / 234.39 / 999.60 | `AIR_Flow`: 533.17 / 13320.82 / 5129.88 / 21904.33 | `WATER_Flow`: 94.77 / 3351.73 / 1446.57 / 7755.74 |
| P02 | `CH4_Total`: 5.09 / 503.61 / 285.84 / 999.99 | `AIR_Flow`: 19.21 / 1980.75 / 1139.26 / 4342.39 | `WATER_Flow`: 10.02 / 1194.15 / 697.85 / 3004.65 |
| P03 | `CH4_Flow`: 5.02 / 504.25 / 287.27 / 999.96 | `AIR_Flow`: 10.87 / 1799.47 / 1136.37 / 4859.02 | `WATER_Flow`: 13.87 / 1497.80 / 868.94 / 3435.94 |
| P04 | `CH4_Flow`: 5.20 / 504.26 / 286.63 / 999.96 | `AIR_Flow`: 269.69 / 55522.21 / 31186.36 / 109523.26 | `WATER_Flow`: 17.01 / 1912.39 / 1131.32 / 4970.71 |
| P05 | `CH4_Flow`: 5.22 / 508.35 / 284.44 / 999.99 | 없음 | `WATER_Flow`: 16.79 / 1920.45 / 1117.63 / 4989.33 |
| P06 | `CH4_Flow`: 5.02 / 503.46 / 286.34 / 999.69 | `AIR_Flow`: 35.65 / 28759.95 / 23831.09 / 107984.82 | `WATER_Flow`: 16.41 / 2007.77 / 1188.11 / 4947.67 |
| P07 | `CH4_Flow`: 5.17 / 501.30 / 287.08 / 999.96 | 없음 | `WATER_Flow`: 17.07 / 2001.92 / 1195.13 / 4975.32 |
| P08 | `CH4_Flow`: 5.08 / 505.60 / 289.10 / 999.90 | 없음 | `WATER_Flow`: 15.88 / 2026.95 / 1208.34 / 4996.87 |
| P09 | `CH4_Flow`: 5.04 / 499.09 / 286.57 / 999.90 | `AIR_Flow`: 282.35 / 12542.75 / 6320.30 / 24997.22 | `WATER_Flow`: 18.15 / 1989.36 / 1189.24 / 4941.40 |
| P10 | `CH4_Flow`: 5.04 / 502.43 / 287.95 / 999.99 | `AIR_Flow`: 17.97 / 2120.27 / 1250.30 / 4906.18 | `WATER_Flow`: 16.31 / 1738.65 / 1019.59 / 4625.53 |

존재하는 feed column에는 NaN과 0이 없었다. P05, P07, P08의 AIR는
단순 missing value가 아니라 해당 process specification에 AIR source가
없으므로 `V_FEED_AIR` node와 edge를 생성하지 않는다.

## 3. 공정별 실제 연결과 graph 크기

괄호 안 ID는 F2가 목적지를 찾는 데 사용한 기존 canonical input edge다.

| Process | 실제 연결 | Node F0 -> F2 | Edge F0 -> F2 |
|---|---|---:|---:|
| P01 | CH4->C1 (`P01_E001`), AIR->C3 (`P01_E004`), WATER->SP1 (`P01_E002`) | 27 -> 30 | 39 -> 42 |
| P02 | CH4->C1 (`P02_E001`), AIR->C2 (`P02_E002`), WATER->P1 (`P02_E003`) | 19 -> 22 | 25 -> 28 |
| P03 | CH4->C2 (`P03_E001`), AIR->MIX2 (`P03_E003`), WATER->P1 (`P03_E002`) | 17 -> 20 | 26 -> 29 |
| P04 | CH4->C1 (`P04_E001`), AIR->BURNER (`P04_E004`), WATER->P1 (`P04_E002`) | 15 -> 18 | 21 -> 24 |
| P05 | CH4->C1 (`P05_E001`), WATER->P1 (`P05_E002`) | 13 -> 15 | 18 -> 20 |
| P06 | CH4->C1 (`P06_E001`), AIR->C3 (`P06_E004`), WATER->P1 (`P06_E002`) | 17 -> 20 | 24 -> 27 |
| P07 | CH4->C1 (`P07_E001`), WATER->P1 (`P07_E002`) | 14 -> 16 | 18 -> 20 |
| P08 | CH4->MIX1 (`P08_E001`), WATER->HEAT1 (`P08_E002`) | 14 -> 16 | 17 -> 19 |
| P09 | CH4->C1 (`P09_E001`), AIR->C2 (`P09_E004`), WATER->P1 (`P09_E002`) | 18 -> 21 | 28 -> 31 |
| P10 | CH4->C1 (`P10_E001`), AIR->C2 (`P10_E003`), WATER->P1 (`P10_E002`) | 20 -> 23 | 30 -> 33 |

기존 node와 edge는 삭제하거나 재번호화하지 않았다. F2 node와 edge를
기존 목록 뒤에 append하므로 `P03_E014`의 semantic ID, source, destination,
target flag, target value와 기존 local position이 모두 유지된다.

새 semantic ID는 sample마다 바뀌지 않는다.

```text
P03_F2_FEED_CH4
P03_F2_FEED_AIR
P03_F2_FEED_WATER
```

## 4. Node 및 edge 표현

### 4.1 Feed node

Feed identity는 별도 process ID나 feed-type embedding 없이 operating slot으로
표현한다.

```text
V_FEED_CH4:   [CH4=value/mask1, AIR=0/mask0, WATER=0/mask0]
V_FEED_AIR:   [CH4=0/mask0, AIR=value/mask1, WATER=0/mask0]
V_FEED_WATER: [CH4=0/mask0, AIR=0/mask0, WATER=value/mask1]
```

Node role은 기존 `source`, unit type은 기존 `stream`을 재사용한다. 일반
node와 `V_INPUT`의 세 feed slot은 모두 value 0, mask 0이다.

### 4.2 Operating shape

```text
기존 operating values        13
feed values                   3
= values                     16

기존 operating masks         13
feed masks                    3
= masks                      16

[value, mask] encoder input  32
Operating encoder output     64
Node role embedding          32
Unit type embedding          24
Node concat                 120
Node initial hidden         384
```

Scaler는 Fold train manifest만 사용하며 feature별로 fit한다. Feed가 없는
node의 0은 통계에 포함되지 않는다. 각 feed slot은 실제로 활성화된 해당
feed node 값만으로 mean/std를 계산한다.

### 4.3 Feed context edge

Feed edge는 기존 generic `feed` stream role과 기존 structural input-edge
schema를 사용한다. 별도 reverse edge는 만들지 않는다. Bidirectional
FlowGNN layer가 동일 edge index에서 forward와 backward attention을 모두
계산하기 때문이다.

Feed edge는 안정적인 CRC 기반 stream identity와 F2 semantic edge ID를
갖지만 property target은 갖지 않는다.

## 5. Context-only mask

| 대상 | context | predictable | supervised | target | PINN |
|---|---:|---:|---:|---:|---:|
| 기존 edge | 0 | 1 | 기존 mask | 기존 mask | 1 |
| F2 feed edge | 1 | 0 | 0 | 0 | 0 |
| 기존 node | 0 | - | - | - | 기존 규칙 |
| F2 feed node | 1 | - | - | - | 0 |

학습 edge grouping은 `edge_is_predictable=0`인 F2 edge를 건너뛴다.
Detailed evaluation/export는 `edge_is_context=1`인 row를 생성하지 않는다.
따라서 feed edge의 임의 decoder 출력은 loss, R2, CSV에 섞이지 않는다.

P03 한 sample의 실제 count diff는 다음과 같다.

| Count | F0 | F2 |
|---|---:|---:|
| Nodes | 17 | 20 |
| Edges | 26 | 29 |
| Supervised edges | 17 | 17 |
| Target edges | 2 | 2 |
| PINN edges | 26 | 26 |
| Context nodes | 0 | 3 |
| Context edges | 0 | 3 |

## 6. PINN 불변성

F2 feed edge는 incidence를 만들기 전에 `edge_pinn_mask=0`으로 제거한다.
Feed node는 Q/W valid mask를 0으로 두고 mass/component/atom/energy
exclusion mask를 모두 1로 둔다.

P03 F0와 F2에 동일한 기존-edge prediction을 넣고, F2 context edge에
`1e12` Mass Flow를 넣는 강한 검사를 수행했다. 다음이 정확히 같았다.

```text
node mass loss
node component loss
node atom loss
weighted node total
mass/component/atom/energy valid residual count
```

즉 F2가 기존 PINN objective를 암묵적으로 바꾸지 않는다.

## 7. P03_E014까지의 전달 경로

`P03_E014` source는 `HX3`다.

| Feed node | Directed distance to HX3 | Undirected distance | 5-layer bidirectional 도달 |
|---|---:|---:|---|
| V_FEED_CH4 | 3 | 3 | 가능 |
| V_FEED_AIR | 2 | 2 | 가능 |
| V_FEED_WATER | 7 | 4 | 가능 |

WATER는 forward-only라면 7 layer가 필요하지만 현재 FlowGNN의 backward
attention까지 포함하면 4 layer에 도달한다. 따라서 임의 shortcut edge를
추가하지 않았다.

각 feed 값을 독립적으로 20% perturb한 검사에서 다음 값이 모두 변했다.

```text
feed node h0
첫 연결 목적지의 layer-1 hidden
P03_E014 source hidden
P03_E014 destination hidden
Set2Set global embedding
P03_E014 CO prediction
```

또한 `P03_E014 CO / CH4`, `P03_E014 CO / AIR`,
`P03_E014 CO / WATER` gradient가 모두 finite 및 non-zero였다.

## 8. F0와 F2의 clean ablation

자동 config diff로 `known_feed_condition`과 output identity를 제외한 다음
설정이 동일함을 검사했다.

```text
model body와 head
5-layer FlowGNN
sampler와 epoch sample 수
loss와 PINN weight/schedule
target weight
optimizer와 learning rate
seed
Fold 1 split
max epoch 30
early stopping patience 5
validation 1000 sampling
```

F0 operating encoder는 `26 -> 64`, F2는 `32 -> 64`이므로 checkpoint shape가
다르다. F2 clean ablation은 기존 checkpoint partial load 없이 처음부터
학습한다.

## 9. 변경 파일

| 파일 | 주요 변경 |
|---|---|
| `src/process_graph/known_feed.py` | F2 config parsing, feed/source binding, metadata 기반 목적지와 semantic ID |
| `src/process_graph/schema.py` | context, predictability, supervision, target, PINN mask |
| `src/process_graph/data/tabular_dataset.py` | F2 node/edge 생성, scaler input, batching, startup debug metadata |
| `src/process_graph/experiment/edge_step_pi_training.py` | context edge optimizer grouping 제외, PINN edge mask 전달 |
| `src/process_graph/experiment/edge_step_training.py` | 일반 edge-step 경로에서도 context edge 제외 |
| `src/process_graph/experiment/node_balance_pi.py` | PINN incidence 계산 전 context edge 제거 |
| `src/process_graph/experiment/edge_all_reporting.py` | context edge metric/export 제외 |
| `src/process_graph/experiment/loaders.py` | F1/F2 canonical v3 topology 검증 |
| `tests/test_known_feed_independent_nodes.py` | F2 topology, scaler, batching, PINN, perturbation, gradient, forward smoke tests |

F2 config:

```text
configs/experiment/pinn/known_feed_260730/f2_e2_independent_feed_nodes.yaml
```

F0 comparison config:

```text
configs/experiment/pinn/known_feed_260730/f0_e2_baseline.yaml
```

## 10. 검증 결과

```text
F2 tests:                 9 passed
기존 F1 tests:            7 passed
현재 Node PINN core:     18 passed
Python py_compile:        passed
```

`test_node_balance_pi.py`의 별도 legacy config 테스트 16개는 저장소에 없는
옛 `a6_pinn_*.yaml`을 참조하여 실패한다. F2 코드 실패가 아니며 core
계산 테스트 18개는 통과했다.

Windows Python 3.12에서 pytest 종료 뒤 `pyarrow` access-violation 문구가
출력되지만, 각 성공 run의 pytest exit code는 0이다. 테스트 본체와
분리된 현재 로컬 interpreter 종료 문제다.

## 11. Fold 1 실행 명령

### F0 baseline

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 python scripts/train_process_surrogate.py \
  --config configs/experiment/pinn/known_feed_260730/f0_e2_baseline.yaml \
  --max-epochs 30 \
  --skip-startup-debug
```

### F2 independent feed nodes

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 python scripts/train_process_surrogate.py \
  --config configs/experiment/pinn/known_feed_260730/f2_e2_independent_feed_nodes.yaml \
  --max-epochs 30 \
  --skip-startup-debug
```

두 config 자체에 같은 Fold 1의 60/20/20 manifest가 지정되어 있다.

```text
data/splits/all_processes_full100k_outer5_grouped_60_20_20/fold_01/
```

## 12. 이번 단계에서 실행하지 않은 항목

요구사항대로 장시간 학습과 full validation은 실행하지 않았다. 따라서
F2가 P03_E014 CO 성능을 실제로 개선하는지는 위 두 Fold 1 run의 best
checkpoint를 동일 evaluation set에서 비교한 뒤 판정해야 한다.
