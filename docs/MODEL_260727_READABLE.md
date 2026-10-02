# 화학공정 그래프 Surrogate 모델 설명서

기준일: 2026-07-27

기준 대표 설정:
`model_260716_pi_tw40_legacy_fracfocus_f01.yaml`

이 문서는 현재 사용 중인 모델을 코드 지식 없이도 이해할 수 있도록 정리한
설명서다. 모델이 어떤 데이터를 받고, 어떻게 예측하며, 어떤 loss로 학습되고,
어떤 기준으로 좋은 모델을 선택하는지 순서대로 설명한다.

---

## 1. 모델의 목적

이 모델의 목적은 화학공정 하나를 그래프로 표현하고, 공정 안의 모든 stream
상태를 동시에 예측하는 것이다.

```text
공정 운전조건 + 공정 연결구조
        ↓
양방향 FlowGNN
        ↓
각 stream의 온도, 압력, 유량, 조성, 밀도, 엔탈피 예측
```

예측 대상은 특정 제품 stream 하나에 한정되지 않는다. 공정 그래프에 존재하는
모든 유효 edge를 예측하면서, 그중 중요한 target edge에는 더 강한 학습 신호를
준다.

---

## 2. 공정을 그래프로 표현하는 방법

### 2.1 Node

Node는 공정 장치를 나타낸다.

예:

- Mixer
- Splitter
- Reactor
- Heat exchanger
- Heater
- Cooler
- Compressor
- Pump
- PSA
- Flash
- 공정 입출력 경계

각 node에는 다음 정보가 들어간다.

- 장치의 역할
- 장치 종류
- 장치 운전조건
- 각 운전조건 값의 유효 여부

### 2.2 Edge

Edge는 장치 사이를 흐르는 stream을 나타낸다.

각 edge에는 다음 정보가 들어간다.

- 출발 node
- 도착 node
- stream 역할
- stream 식별 정보
- 구조적 edge 특성 3개

모델은 edge마다 stream 물성을 예측한다.

### 2.3 Graph sample

데이터의 한 행은 특정 공정 조건에서의 graph sample 하나가 된다.

```text
한 graph sample
    = 고정된 공정 topology
    + 해당 행의 운전조건
    + 모든 유효 stream의 정답
```

현재 학습 batch에는 graph sample 하나만 들어간다.

---

## 3. 사용하는 데이터

현재 Process 1부터 Process 10까지 총 10개 공정을 사용한다.

전체 데이터는 약 100,000개 graph sample로 구성된다.

### 전체공정 5-fold

각 fold는 대략 다음 비율을 사용한다.

| 구분 | 비율 | 약 샘플 수 |
|---|---:|---:|
| Train pool | 60% | 60,000 |
| Validation | 20% | 20,000 |
| Test | 20% | 20,000 |

5개 fold 모두에서 train, validation, test 역할을 바꾸어 평가한다.

### Unseen 공정 실험

특정 공정 하나를 unseen target으로 남기고 나머지 9개 공정으로 pretrain한다.

| 단계 | 사용하는 데이터 |
|---|---|
| Pretrain | 나머지 9개 공정 |
| Zero-shot | 학습에 쓰지 않은 target 공정 |
| Transfer | target 공정의 약 8,000개 adaptation sample |
| Transfer 평가 | target 공정의 약 2,000개 독립 sample |

target 공정의 약 10,000개 데이터는 5개 fold로 나누며, 각 fold에서 약 2,000개를
평가에 사용한다.

---

## 4. 모델이 예측하는 값

각 edge에서 다음 14개 값을 다룬다.

### 운전 상태

1. Temperature
2. Pressure

### 유량

3. Volume Flow
4. Mole Flow
5. Mass Flow

### 몰분율

6. H2O
7. H2
8. CH4
9. CO2
10. CO
11. O2
12. N2

### 열역학 물성

13. Enthalpy
14. Density

앞의 12개 stream property와 Density, Enthalpy는 구조적으로 구분되어 있다.
Density와 Enthalpy는 온도, 압력, 조성 예측을 추가로 이용한다.

---

## 5. 전체 모델 흐름

```text
1. Node와 edge 입력 구성
          ↓
2. Node 초기 embedding 생성
          ↓
3. 5개 양방향 FlowGNN layer에서 순방향/역방향 stream 정보 교환
          ↓
4. 공정 전체를 나타내는 global embedding 생성
          ↓
5. 각 edge의 source, destination, global, 구조정보 결합
          ↓
6. Condition, Fraction, Flow branch에서 stream property 예측
          ↓
7. 예측된 온도, 압력, 조성으로 Density와 Enthalpy 예측
          ↓
8. 정답 supervision과 PINN 조건으로 학습
```

---

## 6. Node encoder

각 node의 범주형 정보와 운전조건을 결합해 512차원 초기 표현
\(h_i^{(0)}\)를 만든다.

```text
장치 역할 embedding 96D
+ 장치 종류 embedding 64D
+ 정규화된 운전조건 x_oper
+ 운전조건 valid mask x_oper_mask
        ↓
Linear(input_dim, 512)
→ ReLU
→ Dropout(0.1)
→ Linear(512, 512)
        ↓
초기 node embedding 512D
```

운전조건이 없는 항목은 0으로 채우되, valid mask를 함께 제공하므로 모델은
실제 0과 결측으로 채운 0을 구분할 수 있다.

HX role embedding 24D를 위한 모듈은 정의되어 있지만 현재 설정에서는
비활성화되어 node encoder 입력에 들어가지 않는다.

설정에 남아 있는 `input_residual: true`는 과거 설정 호환용 이름이다. 현재
동작은 명시된 `initial_residual_mode: per_layer`가 우선하며, 입력 MLP 자체
안에 별도의 skip connection이 생기는 것은 아니다.

---

## 7. FlowGNN encoder

### 7.1 기본 구조

현재 GNN은 512차원 hidden state를 사용하는 5-layer 양방향 FlowGNN 구조다.

각 layer는 정방향과 역방향 branch를 따로 계산한 뒤 결합한다.

```text
현재 node state h
    ├─ 정방향 directed attention → differential update → h_forward
    └─ 역방향 directed attention → differential update → h_backward
                                      ↓
                         concat + Linear projection
                                      ↓
                              raw layer output
```

edge attention용 embedding은 다음 입력으로 만든다.

```text
stream role embedding 64D
+ stream ID embedding 96D
+ masked edge operating features 3D
        ↓
2-layer MLP
        ↓
edge embedding 512D
```

stream ID는 vocabulary 크기 512 안에서 modulo 처리된다. edge operating
feature는 valid mask와 곱한 뒤 encoder에 들어간다.

### 7.2 FlowGNN으로서의 방향 처리

코드에 `FlowGNN`이라는 이름의 단일 클래스가 따로 있는 것은 아니지만, 현재
encoder layer는 stream flow 방향을 명시적으로 사용하는 양방향 FlowGNN
동작을 한다.

원래 공정 stream edge가 \(u\rightarrow v\)라면 두 branch는 다음처럼 다르게
메시지를 보낸다.

| Branch | sender | receiver | 메시지 의미 |
|---|---|---|---|
| Forward | 원래 source \(u\) | 원래 destination \(v\) | 실제 유체 흐름을 따라 upstream 정보를 downstream으로 전달 |
| Backward | 원래 destination \(v\) | 원래 source \(u\) | downstream 상태를 실제 흐름의 반대 방향으로 upstream에 전달 |

```text
원래 stream:             u ─────────→ v

Forward branch:          u ──message→ v
Backward branch:         u ←message── v
```

두 branch는 attention projection과 differential-update MLP를 공유하지 않고
각자 별도 parameter를 가진다. 따라서 backward branch는 forward 결과를 단순히
뒤집는 것이 아니라, 반대 방향 문맥을 독립적으로 학습한다.

이 구조의 역할은 다음과 같다.

- Forward: feed 조건과 upstream 장치 상태가 downstream stream과 장치에 미치는
  영향을 전달한다.
- Backward: 제품측 상태, 후단 장치, recycle 방향의 문맥을 upstream node가
  참고할 수 있게 한다.
- Fusion: 두 방향 표현을 합쳐 한 node가 upstream과 downstream 문맥을 모두
  갖게 한다.

#### 분기 node 예시

원래 topology가 \(A\rightarrow B\), \(A\rightarrow C\)라면:

```text
Forward:
    A ──α_AB──→ B
    A ──α_AC──→ C

Backward:
    A ←─β_AB── B
    A ←─β_AC── C
```

Forward branch는 A의 정보를 B와 C에 나누어 전달한다. Backward branch는 B와
C의 정보를 다시 A로 모아 후단 두 경로의 상태를 A에 전달한다.

#### 현재 attention 정규화의 중요한 세부사항

현재 두 branch 모두 softmax grouping key로 **원래 source index**를 사용한다.

| Branch | message 방향 | softmax 기준 |
|---|---|---|
| Forward | `src → dst` | 원래 `src` |
| Backward | `dst → src` | 원래 `src` |

따라서 위 분기 예시에서는:

\[
\alpha_{AB}+\alpha_{AC}=1
\]

\[
\beta_{AB}+\beta_{AC}=1
\]

이 된다. Forward에서는 원래 source A가 여러 outgoing stream에 보낼 attention
비중을 정규화한다. Backward에서는 B와 C가 A로 되돌려 보내는 메시지를 동일한
원래 source A 그룹 안에서 정규화한다.

반대로 여러 source가 하나의 destination으로 합류하는 경우, forward
attention은 destination 기준으로 다시 정규화하지 않는다. 각 source 그룹에서
계산된 메시지를 destination에서 scatter-sum한다. 그러므로 현재 구현은 일반
GAT의 "destination별 incoming-edge softmax"와 다르다.

### 7.3 Directed attention의 실제 계산

edge \(e=(s\rightarrow r)\)에 대해 sender, receiver, edge embedding을
각각 \(h_s,h_r,e_e\)라 두면 attention score는 다음과 같다.

\[
u_e =
W_{\mathrm{src}}h_s+
W_{\mathrm{dst}}h_r+
W_{\mathrm{edge}}e_e
\]

\[
\ell_e =
w_{\mathrm{score}}^\top\operatorname{ReLU}(u_e)
\]

segment softmax로 \(\alpha_e\)를 만든 뒤 메시지를 계산한다.

\[
m_e =
\alpha_e\,W_{\mathrm{msg}}(h_s+e_e)
\]

\[
a_r = \sum_{e:\operatorname{receiver}(e)=r}m_e
\]

위 식의 sender와 receiver는 branch에 따라 바뀐다.

\[
\begin{aligned}
\text{Forward: }&
(s,r,\operatorname{normalize})=(\mathrm{src},\mathrm{dst},\mathrm{src})\\
\text{Backward: }&
(s,r,\operatorname{normalize})=(\mathrm{dst},\mathrm{src},\mathrm{src})
\end{aligned}
\]

attention weight에는 dropout 0.1이 적용된다. 살아남은 weight는 segment
안에서 다시 합이 1이 되도록 정규화한다. 한 segment의 weight가 모두
dropout되면 원래 attention weight로 되돌아간다.

### 7.4 Differential update

attention 집계 결과를 \(a\), 현재 node state를 \(h\)라 두면 단순히
\([h,a]\)를 연결하지 않는다. 먼저 차이를 계산한다.

\[
\Delta = a-h
\]

\[
d = \operatorname{MLP}_{\mathrm{diff}}(\Delta)
\]

현재 `diff_mode=concat`이므로 branch 출력은 다음과 같다.

\[
h_{\mathrm{dir}} =
\operatorname{MLP}_{\mathrm{update}}([a\,\|\,d])
\]

두 MLP 모두 512차원, 2-layer 구조이며 ReLU와 dropout 0.1을 사용한다.
정방향과 역방향 결과는 concat한 뒤 1024→512 Linear projection으로
결합한다.

\[
\tilde h^{(l)} =
W_{\mathrm{fuse}}
[h_{\mathrm{forward}}^{(l)}\|
  h_{\mathrm{backward}}^{(l)}]
\]

fusion 직후에는 별도 activation이나 normalization이 없다.

### 7.5 Layer residual connection

GNN layer의 raw 출력 \(\tilde h^{(l)}\)을 그대로 다음 layer에 넘기지 않고,
이전 state와 interpolation한다. 현재 \(\alpha_{\mathrm{layer}}=0.5\)다.

\[
\bar h^{(l)}
= h^{(l-1)}
+0.5\left(\tilde h^{(l)}-h^{(l-1)}\right)
\]

즉:

\[
\bar h^{(l)}
=0.5h^{(l-1)}+0.5\tilde h^{(l)}
\]

이 항이 이전 layer 정보를 보존하는 실제 layer residual connection이다.

### 7.6 초기 정보 반복 주입

각 layer의 interpolation residual 뒤에 최초 node embedding의 5%를 더한다.

\[
h^{(l)}=\bar h^{(l)}+0.05h^{(0)}
\qquad l=1,\ldots,5
\]

따라서 한 layer의 전체 갱신은 다음 한 식으로 쓸 수 있다.

\[
h^{(l)}
=0.5h^{(l-1)}
+0.5\tilde h^{(l)}
+0.05h^{(0)}
\]

이 식은 \(h^{(0)}\)을 추가로 더하므로 단순한 50:50 convex average는 아니다.
초기 장치 종류와 운전조건 정보가 5개 layer 모두에서 반복 주입된다.

현재 `layer_residual_norm=false`이므로 이 덧셈 뒤에 LayerNorm을 적용하지
않는다. 또한 `initial_residual_mode=per_layer`이므로 5번째 layer가 끝난 뒤
\(h^{(0)}\)을 한 번 더 더하는 final initial residual은 없다.

### 7.7 최종 node 표현

5번째 layer 출력이 edge prediction에 쓰이는 local node embedding이다.

```text
local_node_embeddings = h^(5)   # 512D
```

별도로 `use_final_projection=true`이므로
`[local node, global context]`를 2-layer MLP에 넣은 512D
`node_embeddings`도 계산한다. 하지만 현재 all-edge PI decoder는 이 projected
`node_embeddings`가 아니라 `local_node_embeddings`를 사용한다.

즉 현재 edge 예측 경로에서 실제로 중요한 residual은 다음 두 개다.

```text
layer interpolation residual 0.5
+ per-layer initial embedding injection 0.05
```

---

## 8. 공정 전체 정보

GNN을 통과한 모든 node embedding을 Set2Set 방식으로 모아 공정 전체를
나타내는 global embedding을 만든다.

```text
모든 node의 512D 표현
        ↓
3-step Set2Set pooling
        ↓
공정 전체 1024D 표현
```

Set2Set은 512D LSTM query \(q_t\)와 node state의 내적으로 attention을
계산한다.

\[
a_{i,t}=\operatorname{softmax}_i(h_i^\top q_t)
\]

\[
r_t=\sum_i a_{i,t}h_i,\qquad
q_t^*=[q_t\|r_t]
\]

이 과정을 3번 반복하고 마지막 \([q_t\|r_t]\)를 1024D global embedding으로
사용한다. batch에 graph가 여러 개면 graph별로 독립 수행하지만, 현재 학습은
batch size 1이다.

---

## 9. Edge별 예측 입력

stream 하나를 예측할 때 다음 네 가지 정보를 합친다.

| 정보 | 차원 |
|---|---:|
| 출발 node 표현 | 512 |
| 도착 node 표현 | 512 |
| 공정 전체 표현 | 1024 |
| edge 구조 특성 | 3 |
| 합계 | 2051 |

따라서 edge별 예측 head의 실제 입력은 2051차원이다.

```text
edge feature
    = [출발 장치 정보,
       도착 장치 정보,
       전체 공정 정보,
       stream 구조 정보]
```

이 구조를 통해 동일한 종류의 stream이라도 앞뒤 장치와 전체 공정 상태에 따라
다른 값을 예측할 수 있다.

현재 PI grouped-property 경로에서는 이 2051D concat tensor가 곧 각 property
branch의 입력이다. 설정의 `edge_decoder_hidden_dim: 384`는 다른 decoder
경로를 위한 값이며 현재 PI head 앞에 별도의 384D shared edge decoder를
만들지 않는다.

---

## 10. Property head

2051차원 edge feature는 세 branch로 나뉜다.

세 branch 사이에 shared hidden MLP는 없다. 각 branch가 동일한 2051D 입력을
직접 받아 독립적인 128D hidden state를 만든다. 현재 branch 내부에도 별도
residual connection은 없다.

### 10.1 Condition branch

```text
2051D → 128D → Temperature, Pressure
```

중간 128차원 층에는:

- Linear
- Layer normalization
- GELU
- Dropout 0.1

이 적용된다.

정확한 순서는:

```text
Linear(2051, 128)
→ LayerNorm(128)
→ GELU
→ Dropout(0.1)
→ Linear(128, output_dim)
```

condition output의 첫 열은 Temperature, 둘째 열은 Pressure다.

### 10.2 Fraction branch

```text
2051D → 128D → 7개 fraction logit → softmax
```

softmax temperature는 0.5다.

```text
x_k = exp(logit_k / 0.5) / Σ exp(logit_j / 0.5)
```

따라서 모든 조성 예측은:

```text
0 ≤ x_k ≤ 1
Σx_k = 1
```

을 만족한다.

### 10.3 Flow branch

```text
2051D → 128D → Mass Flow, Mole Flow, Volume Flow
```

Mass Flow는 값의 범위가 매우 크기 때문에 그대로 회귀하지 않고 scaled-log
공간에서 예측한다.

```text
u_mass = 2 × log(Mass Flow + 1e-8)
```

평가와 물리 loss에서는 다시 physical Mass Flow로 복원한다.

### 10.4 Density와 Enthalpy

예측한 Temperature, Pressure, 7개 fraction을 결합한다.

```text
thermo input = [Temperature, Pressure, 7 fractions]
```

이 9차원 입력을 두 개의 별도 head에 넣는다.

```text
9D → 128D → Density
9D → 128D → Enthalpy
```

따라서 Density와 Enthalpy는 edge feature만 보는 것이 아니라, 모델이 예측한
열역학 상태와 조성에 직접 의존한다.

여기서 thermo head에 들어가는 Temperature와 Pressure는 inverse scaling 전의
network output이다. Density와 Enthalpy도 network/scaler 공간에서 출력된 뒤,
edge 보조 loss와 metric 계산 시 physical 값으로 역변환된다.

현재 `target_branch_hidden_adapter.enabled=false`이므로 target edge hidden
state에만 적용되는 추가 residual adapter도 동작하지 않는다. 따라서 target과
non-target edge는 같은 branch parameter를 공유하고, 차이는 sampling,
target weight 40, target별 optimizer step에서 생긴다.

---

## 11. 정규화

입력 운전조건과 edge target scaler는 train 데이터로만 계산한다.

### Network/scaler 좌표를 사용하는 출력

- Temperature
- Pressure
- Mole Flow
- Volume Flow
- Density
- Enthalpy

### physical 공간을 유지하는 값

- 7개 mole fraction

### 별도 transformed 공간을 사용하는 값

- Mass Flow

loss 공간은 property마다 다르다.

- Temperature, Pressure, Mole Flow, Volume Flow의 main supervision은 scaler
  공간에서 계산한다.
- Density와 Enthalpy의 edge auxiliary loss는 inverse scaling 후 physical
  공간에서 계산한다.
- Volume Flow는 main scaler-space loss와 physical normalized auxiliary
  loss를 둘 다 갖는다.
- Node PINN과 최종 metric은 필요한 모든 값을 physical 단위로 되돌려 계산한다.

따라서 모델의 raw 출력 숫자, main loss에 들어가는 숫자, 최종 physical
prediction 숫자는 서로 같지 않을 수 있다.

---

## 12. Target edge

모든 edge를 동일하게 취급하지 않는다.

공정별 mapping에 정의된 주요 제품 또는 주요 평가 stream을 target edge로
표시한다.

```text
일반 edge weight = 1
target edge weight = 40
```

target edge가 한 graph에 여러 개 있으면 모두 target으로 처리한다.

현재 Target Branch Hidden Adapter는 사용하지 않는다. 즉 target edge를 위해
별도의 출력 head를 추가하는 대신:

1. target edge를 더 자주 학습 데이터에서 선택하고
2. target edge loss에 더 큰 weight를 주고
3. target edge마다 별도 optimizer update를 수행한다.

---

## 13. Main supervision

### 13.1 공통 Smooth L1

현재 main criterion은 PyTorch 기본 `SmoothL1Loss(beta=1)`이다. 오차를
\(r=\hat y-y\)라 하면:

\[
\mathcal S(r)=
\begin{cases}
\frac12r^2, & |r|<1\\
|r|-\frac12, & |r|\ge 1
\end{cases}
\]

큰 이상치에는 MSE보다 완만하고 작은 오차에서는 제곱 오차처럼 동작한다.

### 13.2 Main property loss

PI head 내부의 12개 출력 순서는 다음과 같다.

```text
Temp, Pres,
Frac_H2O, Frac_H2, Frac_CH4, Frac_CO2, Frac_CO, Frac_O2, Frac_N2,
Mass_Flow, Mole_Flow, Vol_Flow
```

fraction 7개는 아래의 별도 log-fraction loss로 보내므로 main mask에서는
제거된다. 남은 유효 feature 집합을 \(\mathcal D_i\)라 하면 edge row \(i\)의
main loss는:

\[
L_{\mathrm{main},i}
=
\frac{
\sum_{d\in\mathcal D_i}
w_{i,d}\,
\mathcal S(\hat y_{i,d}-y_{i,d})
}{
|\mathcal D_i|
}
\]

여기서:

- Temp, Pres, Mole Flow, Volume Flow는 train scaler 공간에서 비교한다.
- fraction은 이 항에서 제외한다.
- Mass Flow는 아래의 scaled-log 공간에서 비교한다.
- mask가 0인 feature는 분자와 유효 feature 수에서 모두 제외한다.
- 현재 Mass Flow의 feature weight \(w_{i,d}\)는 1이다.

### 13.3 Fraction log loss

희박 성분의 작은 값도 학습할 수 있도록 fraction은 물리 mole-fraction
공간에서 log 변환한 뒤 비교한다. 구현은 `log(x + eps)`가 아니라
`log(clamp_min(x, eps))`다.

\[
\phi(x)=\log(\max(x,10^{-6}))
\]

유효 species mask를 \(M_{i,k}\)라 하면:

\[
L_{\mathrm{frac},i}
=
\frac{
\sum_{k=1}^{7}M_{i,k}
\mathcal S\left(\phi(\hat x_{i,k})-\phi(x_{i,k})\right)
}{
\sum_{k=1}^{7}M_{i,k}
}
\]

이 항의 weight는 0.75다. fraction은 이 log loss로만 supervision되며
일반 main loss에 중복해서 포함되지 않는다. CLR loss와 별도 closure loss는
현재 weight 0이다. fraction 합 1은 softmax head가 구조적으로 보장한다.

### 13.4 Fraction row 유효 조건

다음 조건을 모두 만족하는 row만 fraction log loss에 사용한다.

\[
\mathrm{MassFlow}_{i,\mathrm{true}}>10^{-8}
\]

\[
\sum_k M_{i,k}>0,\qquad
\sum_k x_{i,k,\mathrm{true}}>10^{-8}
\]

따라서 zero-flow stream 또는 유효 fraction 정답이 없는 row는 fraction
supervision에서 제외한다.

### 13.5 Mass Flow 변환

Mass Flow는 physical 값을 먼저 복원한 다음 다음 scaled-log 변환을 사용한다.

\[
u_m(m)=2\log(\max(m,0)+10^{-8})
\]

\[
L_{\mathrm{mass},i}
=\mathcal S\left(\hat u_{m,i}-u_m(m_{i,\mathrm{true}})\right)
\]

이 항은 \(L_{\mathrm{main}}\) 안의 Mass Flow feature로 들어간다.
설정 키 `pi_mass_flow_output_space: log1p`의 이름만 보고 단순
\(\log(1+m)\)로 해석하면 안 된다. 실제 변환 종류는
`mass_flow_transform: scaled_log`가 정하며, 현재 식은 위의
\(2\log(m+10^{-8})\)다.

---

## 14. Edge-level loss의 엄밀한 구조

이 절의 Density, Enthalpy, Volume 항은 이름에 PINN이 붙어 출력되기도 하지만,
보존식 residual은 아니다. 세 항 모두 physical-scale 정답과 직접 비교하는
**edge-level auxiliary supervision**이다. 실제 보존법칙 PINN은 15절의
node balance 항들이다.

### 14.1 Density auxiliary loss

physical Density 오차를 먼저 자른 뒤 main criterion과 같은 Smooth L1을
적용한다.

\[
r_{\rho,i}
=
\operatorname{clip}
(\hat\rho_i-\rho_i,-1000,1000)
\]

\[
L_{\rho,i}=\mathcal S(r_{\rho,i})
\]

유효 Density target이 여러 row에 있으면 유효 row 평균을 사용한다. Density와
다른 property 사이의 EOS 식을 강제하는 항은 현재 없다.

### 14.2 Enthalpy와 Volume의 대칭 정규화

Enthalpy와 Volume은 정답 크기만으로 나누지 않는다. prediction과 target의
절댓값 합을 쓰는 대칭 scale을 사용한다.

\[
s(\hat y,y)
=
\max(|\hat y|+|y|,1)+10^{-8}
\]

\[
r(\hat y,y)
=
\operatorname{clip}
\left(
\frac{\hat y-y}{s(\hat y,y)},
-10,10
\right)
\]

이 residual에 delta 0.5인 Huber loss를 적용한다.

\[
\mathcal H_{0.5}(r)=
\begin{cases}
\frac12r^2, & |r|\le0.5\\
0.5(|r|-0.25), & |r|>0.5
\end{cases}
\]

\[
L_{h,i}=\mathcal H_{0.5}(r(\hat h_i,h_i))
\]

\[
L_{V,i}=\mathcal H_{0.5}(r(\hat V_i,V_i))
\]

Volume Flow는 이미 \(L_{\mathrm{main}}\)에서도 scaler-space Smooth L1로
학습된다. \(L_V\)는 같은 target을 physical 대칭 정규화 공간에서 한 번 더
보는 보조항이다. 현재 Volume을 \(m/\rho\)로 유도해 맞추는 물리식은 아니다.
마찬가지로 Enthalpy와 Density도 별도 head의 direct target supervision이다.

### 14.3 Edge row의 전체 loss

edge row \(i\)의 target weight 적용 전 loss는:

\[
\begin{aligned}
L_{\mathrm{edge},i}^{\mathrm{base}}
=\;&1.0L_{\mathrm{main},i}
+0.75L_{\mathrm{frac},i}\\
&+10^{-4}L_{\rho,i}
+10^{-11}L_{h,i}
+0.1L_{V,i}
\end{aligned}
\]

현재 target edge weight는 main loss에만 곱해지는 것이 아니라 위의 전체
묶음에 곱해진다.

\[
w_i=
\begin{cases}
40, & i\text{가 target edge}\\
1, & i\text{가 non-target edge}
\end{cases}
\]

\[
L_{\mathrm{edge},i}
=w_iL_{\mathrm{edge},i}^{\mathrm{base}}
\]

같은 canonical edge group에 여러 row가 있으면 유효 row의
\(L_{\mathrm{edge},i}\)를 평균하여 그 group loss를 만든다.

### 14.4 현재 0으로 꺼진 edge 항

- CLR fraction loss
- 별도 fraction closure loss
- fraction component penalty
- edge enthalpy-flow target loss
- per-edge atom balance
- per-edge energy balance

Mole Flow와 Volume Flow는 head가 직접 예측한다. fallback으로
`Mass/MW` 또는 `Mass/Density`에서 유도하는 경로가 존재하지만 현재 PI head
출력에는 direct prediction이 있으므로 그 fallback은 사용되지 않는다.

---

## 15. Node-level PINN

Node PINN은 stream별 정답 오차가 아니라 장치 주위의 유입량과 유출량이
보존식을 만족하는지 계산한다.

### 15.1 공통 node와 residual 정의

edge \(e\)의 source와 destination을 각각 \(s(e),d(e)\)라 하고, edge quantity
\(q_e\)의 node balance를 다음처럼 정의한다.

\[
B_n(\hat q)
=
\sum_{e:d(e)=n}\hat q_e
-
\sum_{e:s(e)=n}\hat q_e
\]

정규화 scale은 prediction이 아니라 detached true quantity의 incident
절댓값 합으로 만든다.

\[
s_n(q)
=
\max\left(
\sum_{e:d(e)=n}|q_e^{\mathrm{true}}|
+
\sum_{e:s(e)=n}|q_e^{\mathrm{true}}|,
1
\right)
+10^{-6}
\]

\[
r_n(q)=\frac{B_n(\hat q)}{s_n(q)}
\]

유효 residual은 \([-10,10]\)으로 자른 뒤 14.2절의
\(\mathcal H_{0.5}\)를 적용한다. loss는 유효한 node×quantity 항 전체의
평균이다. incident prediction 또는 true 값에 NaN/Inf가 하나라도 있으면
해당 node는 그 balance 항에서 제외한다.

공통 node 집합은 다음처럼 나뉜다.

- **internal**: incoming과 outgoing edge가 모두 있고 virtual 입출력 경계가
  아닌 node
- **non-reactive**: Mixer, Splitter, PSA, Flash, HX, Heater, Cooler,
  Compressor, Pump, Turbine인 internal node
- **reactive**: SMR reactor, WGS reactor, Burner인 internal node

### 15.2 Mass balance

physical Mass Flow \(\dot m_e\)를 사용한다.

\[
r_n^{\mathrm{mass}}
=
\frac{
\sum_{\mathrm{in}}\hat{\dot m}_e
-
\sum_{\mathrm{out}}\hat{\dot m}_e
}{
\max(
\sum_{\mathrm{in}}|\dot m_e^{\mathrm{true}}|
+
\sum_{\mathrm{out}}|\dot m_e^{\mathrm{true}}|,
1)+10^{-6}
}
\]

\[
L_{\mathrm{mass}}
=
\operatorname{mean}_{n\in\mathrm{internal}}
\mathcal H_{0.5}
\left(
\operatorname{clip}(r_n^{\mathrm{mass}},-10,10)
\right)
\]

virtual boundary를 제외한 모든 internal node에 적용하며 weight는 0.75다.

### 15.3 Component molar balance

species 순서는 H2O, H2, CH4, CO2, CO, O2, N2이고 molecular weight 숫자는
다음과 같다.

```text
18.01528, 2.01588, 16.04246, 44.0095,
28.0101, 31.9988, 28.0134
```

\[
\overline{MW}_e=\sum_{k=1}^{7}\hat x_{e,k}MW_k
\]

\[
\hat{\dot n}_e
=
\frac{\hat{\dot m}_e}
{\max(\overline{MW}_e,10^{-6})}
\]

\[
\hat{\dot n}_{e,k}=\hat{\dot n}_e\hat x_{e,k}
\]

각 species \(k\)에 대해 15.1절의 balance와 true incident scale을 적용한다.

\[
L_{\mathrm{component}}
=
\operatorname{mean}_{n\in\mathrm{nonreactive}},k
\mathcal H_{0.5}
\left(
\operatorname{clip}(r_{n,k}^{\mathrm{component}},-10,10)
\right)
\]

reaction이 없는 장치에만 적용하며 weight는 \(1.5\times10^{-7}\)이다.
구현은 위 molecular weight 숫자와 Mass Flow의 현재 단위 convention을 그대로
사용하므로 데이터의 Mass Flow/MW 단위가 이 convention과 일치해야 한다.

`component_flow_threshold=1e-6`도 설정되어 있지만 scale을 먼저 최소 1로
올리므로, 현재 조합에서는 이 threshold가 유효 node를 추가로 제거하지는
않는다.

### 15.4 Atom balance

반응 장치에서는 species가 바뀔 수 있으므로 component balance 대신 C, H, O,
N 원자 흐름을 사용한다. species별 원자수 행렬 \(A\)는:

\[
\begin{array}{c|rrrr}
 & C&H&O&N\\\hline
\mathrm{H_2O}&0&2&1&0\\
\mathrm{H_2}&0&2&0&0\\
\mathrm{CH_4}&1&4&0&0\\
\mathrm{CO_2}&1&0&2&0\\
\mathrm{CO}&1&0&1&0\\
\mathrm{O_2}&0&0&2&0\\
\mathrm{N_2}&0&0&0&2
\end{array}
\]

\[
\hat{\mathbf a}_e
=
\hat{\boldsymbol{\dot n}}_e A
\]

\[
L_{\mathrm{atom}}
=
\operatorname{mean}_{n\in\mathrm{reactive}},a
\mathcal H_{0.5}
\left(
\operatorname{clip}(r_{n,a}^{\mathrm{atom}},-10,10)
\right)
\]

SMR reactor, WGS reactor, Burner에 적용하며 weight는 0.2다.

### 15.5 Energy balance

현재 `h_basis=mass_specific`이므로 edge enthalpy flow는:

\[
\hat{\dot H}_e=\hat{\dot m}_e\hat h_e
\]

\[
\dot H_e^{\mathrm{true}}
=\dot m_e^{\mathrm{true}}h_e^{\mathrm{true}}
\]

일반 energy residual과 true-data consistency residual은 각각:

\[
r_n^{E,\mathrm{pred}}
=
\frac{
\sum_{\mathrm{in}}\hat{\dot H}_e
+Q_n+W_n
-\sum_{\mathrm{out}}\hat{\dot H}_e
}{
s_n^E
}
\]

\[
r_n^{E,\mathrm{true}}
=
\frac{
\sum_{\mathrm{in}}\dot H_e^{\mathrm{true}}
+Q_n+W_n
-\sum_{\mathrm{out}}\dot H_e^{\mathrm{true}}
}{
s_n^E
}
\]

\[
s_n^E=
\max\left(
\sum_{\mathrm{in}}|\dot H_e^{\mathrm{true}}|
+
\sum_{\mathrm{out}}|\dot H_e^{\mathrm{true}}|
+|Q_n|+|W_n|,
1
\right)+10^{-6}
\]

현재 `use_process_main_qw=false`이므로 \(Q_n=W_n=0\)이다. 따라서 지금
학습되는 식은 실질적으로 inlet/outlet enthalpy-flow balance다.

Energy 적용 후보는 internal node에서 다음을 제외한 집합이다.

- Heater
- Cooler
- SMR reactor
- WGS reactor

설정의 `reactor` alias는 SMR/WGS reactor로 확장된다. **Burner는 이 제외
목록에 들어가지 않으므로 현재 Energy PINN 적용 후보에 남는다.**

또한 true 데이터 자체가 단순 energy balance와 맞는 node만 사용한다.

\[
|r_n^{E,\mathrm{true}}|\le0.05
\]

최종 energy loss는:

\[
L_{\mathrm{energy}}
=
\operatorname{mean}_{n\in\mathcal V_E,\,
|r_n^{E,\mathrm{true}}|\le0.05}
\mathcal H_{0.5}
\left(
\operatorname{clip}(r_n^{E,\mathrm{pred}},-10,10)
\right)
\]

weight는 0.01이다. 현재 prediction residual 크기를 이유로 node를 제외하는
추가 filter는 설정되어 있지 않다.

### 15.6 Node PINN 합

\[
L_{\mathrm{node}}
=
0.75L_{\mathrm{mass}}
+1.5\times10^{-7}L_{\mathrm{component}}
+0.2L_{\mathrm{atom}}
+0.01L_{\mathrm{energy}}
\]

어떤 항도 유효 node가 없으면 그 항은 autograd 연결을 유지한 0이 된다.
이 합은 edge loss에 숫자로 더해지는 것이 아니라 별도 node optimizer
update의 목적함수다.

### 15.7 Loss 역할 요약

| 항 | 성격 | 계산 공간 | 기본 weight | update |
|---|---|---|---:|---|
| Main property | 정답 supervision | scaler/변환 공간 | 1.0 | edge |
| Fraction log | 정답 supervision | physical fraction의 log | 0.75 | edge |
| Density | 정답 보조 supervision | physical | `1e-4` | edge |
| Enthalpy | 정답 보조 supervision | physical 대칭 정규화 | `1e-11` | edge |
| Volume | 정답 보조 supervision | physical 대칭 정규화 | 0.1 | edge |
| Node Mass | conservation PINN | physical relative residual | 0.75 | node |
| Node Component | conservation PINN | molar-flow relative residual | `1.5e-7` | node |
| Node Atom | conservation PINN | atom-flow relative residual | 0.2 | node |
| Node Energy | conservation PINN | enthalpy-flow relative residual | 0.01 | node |

target weight 40은 앞의 다섯 edge 항을 합친 edge group loss에 적용한다.
PINN schedule은 뒤의 네 node 항에만 적용한다.

---

## 16. PINN schedule

학습 초기에 node PINN이 아직 부정확한 예측을 강하게 제한하지 않도록 단계적으로
활성화한다.

| Epoch | Node PINN 세기 |
|---:|---:|
| 1-5 | 0% |
| 6-7 | 50% |
| 8 이후 | 100% |

epoch \(e\)의 multiplier를 \(\gamma_e\)라 하면:

\[
\gamma_e=
\begin{cases}
0, & 1\le e\le5\\
0.5, & 6\le e\le7\\
1, & e\ge8
\end{cases}
\]

\[
L_{\mathrm{node}}^{(e)}
=\gamma_e
\left(
0.75L_{\mathrm{mass}}
+1.5\times10^{-7}L_{\mathrm{component}}
+0.2L_{\mathrm{atom}}
+0.01L_{\mathrm{energy}}
\right)
\]

이 schedule은 node PINN 네 항의 weight에 같은 multiplier를 곱한다. epoch
1-5에는 multiplier가 0이므로 node optimizer step도 수행하지 않는다.

Density, Enthalpy, Volume의 edge-level auxiliary supervision에는 이 schedule이
적용되지 않으며 epoch 1부터 활성화된다. 따라서 초기 epoch 로그에도 edge
physics 진단 숫자가 나타나는 것은 정상이다.

---

## 17. 한 sample의 학습 순서

현재 핵심 학습 방식은 target edge와 일반 edge를 서로 다른 optimizer update로
분리하는 것이다.

### 17.1 Non-target update

```text
fresh forward 1회
        ↓
모든 non-target canonical edge group loss 계산
        ↓
group별 loss를 동일 비중으로 평균
        ↓
backward
        ↓
optimizer step 1회
```

non-target canonical group 수를 \(G_{\mathrm{NT}}\)라 하면:

\[
L_{\mathrm{NT}}
=
\frac{1}{G_{\mathrm{NT}}}
\sum_{g=1}^{G_{\mathrm{NT}}}L_g
\]

이것은 canonical group macro mean이다. group에 속한 edge-row 수가 많다고
더 큰 group weight를 받지 않는다. non-target edge가 여러 개여도 이
목적함수로 optimizer step은 한 번만 수행한다.

### 17.2 Target update

target edge는 각각 독립적으로 학습한다.

```text
target edge 1
    fresh forward → loss → backward → optimizer step

target edge 2
    fresh forward → loss → backward → optimizer step

target edge 3
    fresh forward → loss → backward → optimizer step
```

각 target edge는 다른 edge의 평균 loss에 묻히지 않는 독립적인 gradient 신호를
얻는다. 현재 `use_existing_target_edge_weight=true`이므로 각 target update의
loss에는 14.3절의 target weight 40도 실제로 적용된다.

### 17.3 Node PINN update

Node PINN이 활성화된 epoch에서는 다시 fresh forward를 수행한다.

```text
fresh forward
    ↓
Mass + Component + Atom + Energy PINN
    ↓
backward
    ↓
optimizer step 1회
```

### 17.4 update 수 예시

target edge 3개, non-target 존재, node PINN 활성인 sample:

```text
non-target update 1회
target update 3회
node PINN update 1회

총 forward 5회
총 optimizer step 5회
```

이 여러 번의 forward는 중복 오류가 아니라 의도된 학습 방식이다.

중요하게도 이 학습은 한 sample에 대해 다음 하나의 scalar를 만든 뒤 한 번
backward하는 방식이 아니다.

```text
L_non-target + ΣL_target + L_node
```

대신 위 목적함수들을 순서대로 각각 backward/step한다. optimizer parameter가
각 step 사이에 바뀌므로 뒤 update는 앞 update가 반영된 새 model로 fresh
forward한다. 따라서 `retain_graph=True`로 한 forward graph를 재사용하지
않는다. 터미널의 sample `loss`는 성공한 update loss들의 요약값이지, 실제로
한 번에 backward된 단일 통합 objective가 아니다.

---

## 18. Epoch sampling

전체 train pool을 매 epoch 전부 사용하지 않는다.

현재 sampling은 두 단계다.

### 18.1 Base sampling

train pool의 2%를 먼저 선택한다.

전체공정 train pool이 약 60,000개이면:

```text
약 1,200개
```

base sample 내부 구성:

| 구성 | 비율 |
|---|---:|
| 일반 random | 25% |
| 희소 양수 성분 | 37.5% |
| 물성 quantile 균형 | 25% |
| Mass Flow tail | 12.5% |

이 단계에서는 저조 target edge를 별도로 과대표집하지 않는다.

### 18.2 Hard-sample 추가

base sampling 뒤 성능이 낮았던 target 성분 sample을 추가한다.

집중 성분:

- Target Frac_CH4
- Target Frac_CO
- Target Frac_CO2

최종 epoch 크기:

```text
2,000개
```

목표 hard-sample 비율:

```text
40% = 약 800개
```

hard sample이 부족하면 남은 자리는 일반 random sample로 채운다.

### 18.3 중복

한 epoch 안에서는 같은 원본 sample을 중복 선택하지 않는다.

```text
epoch당 최대 1회
```

epoch가 바뀌면 seed가 달라지므로 다른 sample 조합이 선택된다.

---

## 19. Validation

매 epoch validation을 수행한다.

전체 validation pool을 매번 모두 평가하지 않고, 매 epoch 무작위 1,000개를
선택한다.

```text
Validation pool 약 20,000
        ↓
Random 1,000
        ↓
성분별 target/all-edge metric 계산
```

validation에서는 학습용 PINN loss 전체를 다시 계산하지 않는다. 예측 metric
계산에 집중해 validation 시간을 줄인다.

---

## 20. 모델 선택과 Early stopping

학습 loss가 가장 낮은 모델이 아니라 target edge validation R2가 가장 좋은
모델을 선택한다.

```text
monitor = validation target mean R2
mode = maximize
patience = 5
max epoch = 30
```

5번의 validation 동안 최고 target mean R2가 개선되지 않으면 학습을 멈춘다.

중요:

- train loss는 backprop에 사용된다.
- validation target R2는 checkpoint 선택과 early stopping에 사용된다.
- 두 값의 역할은 다르다.

---

## 21. 성능을 확인하는 방법

평균 R2 하나만 보면 특정 성분의 실패가 가려질 수 있다.

따라서 다음 두 범주를 성분별로 확인한다.

### Target-edge 성분별 성능

- Temperature
- Pressure
- Volume Flow
- Mole Flow
- Mass Flow
- Frac_H2O
- Frac_H2
- Frac_CH4
- Frac_CO2
- Frac_CO
- Frac_O2
- Frac_N2

### All-edge 성분별 성능

- 위 12개 property
- Density
- Enthalpy

특히 현재 hard sampling의 직접 목표는 다음 세 target 성분이다.

```text
Frac_CH4
Frac_CO
Frac_CO2
```

따라서 sampler 변경 효과는 이 세 성분의 target R2와 다른 성분의 손실 여부를
함께 비교해야 한다.

---

## 22. Optimizer와 학습 설정

| 설정 | 값 |
|---|---:|
| Optimizer | AdamW |
| Learning rate | `1e-4` |
| Weight decay | `1e-5` |
| Max epochs | 30 |
| Batch size | 1 |
| Gradient accumulation | 1 |
| Gradient clip norm | 0.5 |
| Mixed precision | 사용하지 않음 |
| Scheduler | Cosine |
| Scheduler minimum LR | `1e-6` |
| Early-stopping patience | 5 |

scheduler는 sample별 optimizer step이 아니라 epoch마다 한 번 진행한다.

---

## 23. Loss 폭발 방지

현재 모델에는 여러 단계의 안정화가 들어가 있다.

### 물리 residual의 수치 안정화

- node balance denominator에 epsilon `1e-6` 추가
- Enthalpy/Volume 대칭 denominator에 epsilon `1e-8` 추가
- relative loss denominator의 scale floor를 1로 설정
- node residual 최대 절댓값 10
- Enthalpy/Volume residual 최대 절댓값 10
- Density residual 최대 절댓값 1000
- node와 Enthalpy/Volume residual에 delta 0.5 Huber 적용

여기의 residual은 loss 계산용 물리 residual이다. 7절의 GNN residual
connection과는 이름만 같고 서로 다른 기능이다.

### Gradient 안정화

- 전체 gradient norm을 0.5로 clipping
- loss가 NaN/Inf이면 해당 update 건너뜀
- gradient norm이 NaN/Inf이면 optimizer step을 수행하지 않음

### Epoch 복구

epoch 종료 후 model parameter에 NaN/Inf가 발견되면:

1. epoch 시작 상태로 model과 optimizer를 복원한다.
2. learning rate를 절반으로 낮춘다.
3. 최소 learning rate `1e-6`은 유지한다.

---

## 24. 로그 읽는 방법

예:

```text
[train e=3/30 step=80/2000 ...]
loss=...
edge=t.../nt...
pinn=(...)
updates=opt5/t3/n1/node1
groups=t3/nt18
fwd=5
skip=0
gnorm=...
```

의미:

| 항목 | 의미 |
|---|---|
| `e=3/30` | 30 epoch 중 3번째 |
| `step=80/2000` | 이번 epoch의 2,000 sample 중 80번째 |
| `edge=t/nt` | target/non-target edge loss |
| `pinn=(...)` | PINN component 진단값 |
| `opt5` | optimizer step 총 5회 |
| `t3` | target update 3회 |
| `n1` | non-target 평균 update 1회 |
| `node1` | node PINN update 1회 |
| `groups=t3/nt18` | target 3 group, non-target 18 group |
| `fwd=5` | forward 5회 |
| `skip=0` | 건너뛴 update 없음 |
| `gnorm` | clipping 전 계산된 gradient norm |

PINN raw diagnostic 숫자가 커 보여도 실제 gradient에는 각 weight와 residual
normalization/clipping이 적용될 수 있다. 그러나 모든 로그 숫자가 자동으로
안전하다는 뜻은 아니다. `loss`, clipping 전 `gnorm`, `skip`이 반복적으로
커지면 실제 update 불안정을 의심해야 한다.

---

## 25. 전체공정, Pretrain, Zero-shot, Transfer

### 전체공정 학습

```text
Process 1-10
    ↓
60/20/20 outer 5-fold
    ↓
fold별 train
    ↓
fold별 independent test
```

### Pretrain

```text
target 공정 1개 제외
    ↓
나머지 9개 공정으로 학습
    ↓
source validation으로 best checkpoint 선택
```

### Zero-shot

```text
pretrain checkpoint
    ↓
추가 학습 없음
    ↓
unseen target 공정의 5개 test fold 평가
```

### Transfer

```text
pretrain checkpoint
    ↓
target 공정 약 8,000개로 전체 parameter fine-tuning
    ↓
남은 약 2,000개로 평가
```

각 target 공정마다 pretrain은 한 번만 수행하고, 그 checkpoint를 zero-shot과
5개 transfer fold가 공유한다.

---

## 26. 현재 사용하지 않는 기능

현재 모델 설명에서 제외해야 하는 기능:

- Target Branch Hidden Adapter
- CLR fraction loss
- 별도 fraction closure loss
- 기존 target augmentation
- 기존 rare-positive sampler
- 별도 target/tailgas node output head
- mixed precision
- validation PINN loss

이 기능들은 일부 설정 정의나 구현이 남아 있을 수 있지만 현재 대표 학습
경로에는 적용되지 않는다.

---

## 27. 현재 모델의 핵심 의도

현재 설계는 다음 네 가지 목표를 동시에 만족하려고 한다.

### 1. 전체 공정 학습

모든 edge를 예측해 공정 전체 상태를 이해한다.

### 2. 중요한 target 강화

target edge를 40배 가중하고 각각 독립적으로 update해 평균 loss에 묻히지 않게
한다.

### 3. 희소 target 성분 개선

Frac_CH4, Frac_CO, Frac_CO2가 존재하는 어려운 target sample을 epoch sample의
약 40%까지 추가한다.

### 4. 물리적 일관성

Mass, Component, Atom, Energy의 node conservation residual을 실제 PINN으로
사용한다. Density, Enthalpy, Volume은 보존식이 아니라 physical-space
auxiliary target supervision이다. Energy는 true-data consistency filter를
통과한 node에만 적용하고, residual과 gradient를 제한해 학습 폭발을 막는다.

---

## 28. 방법론별 설계 이유와 검증 논리

이 절은 현재 활성화된 방법론을 단순히 나열하지 않고, 각각 어떤 문제를
해결하려고 도입했는지 설명한다.

중요하게도 아래의 "기대 효과"는 구조로부터 도출한 **설계 가설**이다.
실제 성능 향상 여부는 동일 split과 seed를 사용한 ablation 및 target/all-edge
성분별 지표로 확인해야 한다.

### 28.1 공정을 node-edge graph로 표현

**문제**

평탄한 tabular 모델은 한 stream의 값이 어떤 장치에서 왔고 어디로 가는지,
분기·합류·recycle이 어떻게 연결되는지를 직접 표현하기 어렵다.

**선택한 방법**

- 장치를 node로 표현한다.
- 물질 stream을 방향이 있는 edge로 표현한다.
- 운전조건은 node/edge feature로 넣는다.
- 예측도 공정 그래프의 모든 유효 edge에서 수행한다.

**이유**

화학공정의 상호작용은 topology에 강하게 의존한다. 동일한 온도나 압력이라도
앞뒤 장치와 연결 방향이 다르면 의미가 달라진다. graph 표현은 연결관계를
모델 입력에 직접 보존한다.

**주의점과 검증**

고정 topology가 강한 process identity 신호가 될 수 있다. Seen-process
성능만으로 일반화를 판단하지 않고, 공정을 완전히 제외한 unseen zero-shot과
transfer 결과를 별도로 확인해야 한다.

### 28.2 양방향 FlowGNN

**문제**

실제 유체 흐름만 따라 메시지를 보내면 downstream은 upstream 조건을 받을 수
있지만, upstream node는 제품측 상태나 후단 장치 문맥을 직접 받기 어렵다.

**선택한 방법**

- Forward branch: 실제 흐름 `src → dst`
- Backward branch: 반대 방향 `dst → src`
- 방향별 attention과 update parameter를 독립적으로 학습
- 두 결과를 concat하고 512D로 projection

**이유**

Forward branch는 feed와 전단 운전조건의 영향을 후단으로 전달한다. Backward
branch는 제품측, recycle측, 후단 장치 문맥을 전단으로 돌려보낸다. 두 방향을
결합하면 각 node가 upstream과 downstream 조건을 함께 표현할 수 있다.

현재 source-group softmax는 분기 node의 여러 outgoing path 사이 중요도를
학습하는 데 자연스럽다. Backward에서도 같은 원래 source 그룹을 사용하므로,
분기 후단의 여러 경로가 upstream node에 되돌려 주는 정보의 비중을 학습한다.

**대가와 검증**

- 단방향 GNN보다 attention/update 계산량이 약 2배다.
- 합류 node에서는 incoming edge를 destination 기준으로 softmax하지 않고
  source별 메시지를 합산한다.
- `forward only`, `backward only`, `bidirectional fusion` ablation으로 실제
  이득을 확인해야 한다.
- recycle 또는 분기·합류가 많은 공정에서 개선이 더 큰지 공정별로 확인한다.

### 28.3 Edge-conditioned attention

**문제**

같은 source/destination node embedding이라도 stream 역할과 edge 운전정보가
다르면 전달해야 하는 정보가 다르다.

**선택한 방법**

attention score와 message에 stream role, stream ID, edge operating feature를
인코딩한 512D edge embedding을 함께 사용한다.

**이유**

node topology만으로는 서로 다른 stream의 기능을 충분히 구분하기 어렵다.
edge embedding을 score와 message 양쪽에 넣으면 "어느 장치가 연결됐는가"뿐
아니라 "어떤 stream으로 연결됐는가"도 message passing에 반영된다.

**주의점과 검증**

stream ID가 process-specific identity를 과도하게 학습하면 unseen 공정에
불리할 수 있다. seen 5-fold와 unseen zero-shot 간 격차를 확인하고, 필요하면
stream ID embedding 제거 ablation을 수행해야 한다.

### 28.4 Differential update

**문제**

집계 message \(a\)만 사용하면 현재 node state \(h\)와 이웃 정보가 얼마나
다른지 명시적으로 표현되지 않는다.

**선택한 방법**

\[
\Delta=a-h,\qquad
d=\operatorname{MLP}_{\mathrm{diff}}(\Delta)
\]

\[
h_{\mathrm{dir}}
=\operatorname{MLP}_{\mathrm{update}}([a\|d])
\]

**이유**

\(\Delta\)는 현재 node와 이웃에서 들어온 정보 사이의 변화량이다. 절대적인
집계값 \(a\)와 변화량 표현 \(d\)를 함께 사용하면, 장치 상태 자체와 주변
stream 때문에 필요한 수정량을 구분해서 학습할 수 있다.

**주의점과 검증**

일반 message-passing update보다 parameter와 연산량이 늘어난다.
`agg only`, `agg + raw delta`, `agg + encoded delta` 비교로 이 구조가 실제로
필요한지 확인할 수 있다.

### 28.5 Layer residual과 초기 embedding 반복 주입

**문제**

5개 message-passing layer를 지나면서 초기 장치 종류와 운전조건이 희석되거나,
인접 node 표현이 지나치게 비슷해지는 over-smoothing이 발생할 수 있다.

**선택한 방법**

\[
h^{(l)}
=0.5h^{(l-1)}
+0.5\tilde h^{(l)}
+0.05h^{(0)}
\]

**이유**

- \(0.5h^{(l-1)}\): 이전 layer 표현과 gradient 경로를 보존한다.
- \(0.5\tilde h^{(l)}\): 새 topology/message 정보를 반영한다.
- \(0.05h^{(0)}\): 장치 identity와 원래 운전조건을 매 layer 다시 제공한다.

일반 residual과 초기값 재주입은 목적이 다르다. 전자는 깊은 network의 안정화,
후자는 초기 local signal 보존을 담당한다.

**주의점과 검증**

초기값을 반복해서 더하므로 50:50 convex interpolation만으로 해석할 수 없다.
초기 feature 의존성이 너무 강해질 가능성도 있다. layer별 embedding 분산,
node 간 cosine similarity, `initial_residual_alpha` ablation으로 확인한다.

### 28.6 Set2Set global context

**문제**

edge 예측은 양 끝 node뿐 아니라 전체 feed 조건, 공정 규모, 멀리 떨어진 장치
상태에도 영향을 받을 수 있다.

**선택한 방법**

모든 local node embedding을 3-step Set2Set으로 모아 1024D global embedding을
만들고 모든 edge prediction에 제공한다.

**이유**

단순 평균은 모든 node를 고정된 비중으로 취급한다. Set2Set은 반복 attention으로
현재 graph 상태에서 중요한 node에 동적으로 집중할 수 있다.

**대가와 검증**

graph별 LSTM/attention 계산이 추가된다. global context가 없을 때, mean pooling,
Set2Set을 비교해 성분별 이득과 실행시간을 함께 평가해야 한다.

### 28.7 Grouped property head와 fraction softmax

**문제**

온도·압력, 조성, 유량은 통계적 성격과 물리적 제약이 다르다. 하나의 단일
출력층에서 모두 예측하면 서로 다른 scale과 gradient가 간섭할 수 있다.

**선택한 방법**

- Condition branch: Temperature, Pressure
- Fraction branch: 7개 mole fraction
- Flow branch: Mass, Mole, Volume Flow
- Thermo heads: 예측된 Temperature, Pressure, fraction으로 Density와
  Enthalpy 예측
- fraction logits에는 temperature 0.5 softmax 적용

**이유**

branch를 분리하면 property group별 hidden representation을 학습할 수 있다.
fraction softmax는 음수 조성과 합이 1이 아닌 예측을 architecture 단계에서
차단한다. Density와 Enthalpy가 예측된 상태·조성에 의존하게 하여 출력 사이
연결도 일부 보존한다.

**주의점과 검증**

- softmax는 모든 fraction을 경쟁시키므로 한 성분 증가가 다른 성분 감소로
  이어진다.
- temperature 0.5는 분포를 더 날카롭게 하므로 희박성분 gradient에 영향을
  줄 수 있다.
- Density/Enthalpy head의 thermo input은 예측값이므로 condition/fraction
  오차가 전파된다.
- fraction 합과 음수 여부뿐 아니라 희박성분별 R2와 MAE를 확인해야 한다.

### 28.8 Log-fraction과 scaled-log Mass Flow supervision

**문제**

fraction과 Mass Flow는 여러 자릿수 범위를 갖는다. physical 값에서 동일한
절대오차를 쓰면 큰 값이 loss를 지배하고 희박성분이나 저유량 영역이 무시될
수 있다.

**선택한 방법**

\[
\phi(x)=\log(\max(x,10^{-6}))
\]

\[
u_m(m)=2\log(\max(m,0)+10^{-8})
\]

**이유**

log 공간은 multiplicative/relative 차이를 더 잘 드러낸다. 예를 들어
\(10^{-4}\rightarrow10^{-3}\)의 차이가 큰 bulk fraction에 묻히는 것을
줄인다. scaled-log Mass Flow도 큰 dynamic range를 압축한다.

**주의점과 검증**

- epsilon 근처에서는 작은 physical 차이가 큰 log 차이가 될 수 있다.
- zero-flow row는 fraction 의미가 불명확하므로 mask한다.
- log-space 성능만 보지 않고 physical MAE/RMSE/R2도 함께 봐야 한다.

### 28.9 Target-aware sequential optimization

**문제**

한 graph의 많은 non-target edge를 모두 평균하면 소수의 중요한 target edge
gradient가 묻힐 수 있다.

**선택한 방법**

1. non-target canonical group들을 macro mean하여 한 번 update한다.
2. target canonical group은 각각 fresh forward와 독립 update를 수행한다.
3. target edge 전체 loss bundle에 weight 40을 적용한다.
4. node PINN은 다시 별도 fresh forward와 update를 수행한다.

**이유**

- non-target는 한 번에 학습해 계산량을 제한한다.
- target는 edge별 update를 보장해 다른 edge 평균에 희석되지 않게 한다.
- fresh forward는 앞 optimizer step 이후의 새 parameter로 loss를 다시
  계산하므로 stale activation 문제를 피한다.
- canonical group macro mean은 edge row 수가 많은 group의 독점을 막는다.

**중요한 해석**

target 강화는 단순한 "40배 loss"보다 강하다. target은 weight 40에 더해
각 edge별 optimizer step까지 받는다. 따라서 실질적인 영향은 target 수,
gradient 방향, AdamW state에 따라 달라지며 정확히 40배라고 해석할 수 없다.

**대가와 검증**

- sample당 forward/optimizer step 수가 target edge 수에 따라 증가한다.
- 너무 강하면 all-edge 성능이나 다른 target property가 떨어질 수 있다.
- target/all-edge 성분별 R2, gradient norm, target/non-target loss를 함께
  비교해야 한다.

### 28.10 Base 2%와 hard-target fill의 2단계 sampling

**문제**

전체 train pool을 매 epoch 모두 사용하면 시간이 길다. 반면 단순 2% random
sampling만 사용하면 희소 양수, 극단 유량, 성능이 낮은 target edge를 충분히
보기 어렵다.

**선택한 방법**

첫 단계는 train pool의 2%를 다음 mixture로 선택한다.

| Base bucket | 비율 | 목적 |
|---|---:|---|
| Uniform | 25% | 원래 데이터 분포 보존 |
| Sparse positive | 37.5% | 희소 fraction 양수 사례 확보 |
| Quantile balance | 25% | property 범위 전체 확보 |
| Mass Flow tail | 12.5% | 고유량 극단영역 확보 |

두 번째 단계는 최종 2,000개 중 약 800개, 즉 40%를 성능이 낮았던 target
Frac_CH4, Frac_CO, Frac_CO2 edge의 유효 양수 sample로 채운다. base 단계에서는
hard target edge를 제외해 두 단계의 역할이 겹치지 않게 한다. 한 epoch 안의
원본 sample 중복은 허용하지 않고 process 균형도 적용한다.

**이유**

base 2%는 전체 분포와 다양한 물성 범위를 유지한다. hard fill은 기존 산점도와
성분별 지표에서 반복적으로 낮았던 target edge에 추가 학습 기회를 준다.
단순히 base sampler의 비율을 왜곡하지 않고 별도 fill로 추가하므로, 일반
coverage와 문제영역 집중을 분리할 수 있다.

**대가와 검증**

- hard sample은 원래 데이터 분포보다 과대표집된다.
- 특정 edge에 과적합하거나 다른 공정 성능이 떨어질 수 있다.
- hard edge의 target CH4/CO/CO2뿐 아니라 다른 target 성분과 all-edge 성분을
  반드시 함께 비교한다.
- hard sample이 부족할 때 random fallback 수가 증가하므로 sampler 로그의
  `hard`, `hard_fallback_to_random`, `unique`를 확인한다.

### 28.11 장치 의미에 따른 node conservation PINN

**문제**

정답 supervision만으로는 개별 stream 예측이 좋아도 장치 주위 유입·유출이
서로 모순될 수 있다.

**선택한 방법**

- 모든 internal node: Mass balance
- 비반응 장치: Component molar balance
- 반응 장치: Atom balance
- 열·일 정보 없이 적용 가능한 후보 node: Energy balance

**이유**

모든 장치에 동일한 conservation 식을 강제하면 반응기에서 species balance가
깨지는 등 잘못된 제약이 된다. 장치 의미에 맞게 component와 atom balance를
분리하면 반응 전후 species 변화는 허용하면서 원자 보존은 요구할 수 있다.

**주의점과 검증**

- molecular weight와 Mass Flow 단위 convention이 맞아야 한다.
- \(Q/W\)가 비활성이라 Heater, Cooler, SMR/WGS reactor는 Energy에서 제외한다.
- Burner는 현재 Energy 후보에 남으므로 true-data consistency 통과율을 별도로
  확인해야 한다.
- PINN residual 감소와 예측 R2 향상은 같은 의미가 아니므로 둘 다 기록한다.

### 28.12 Energy true-data consistency mask

**문제**

정답 데이터 자체가 단순 inlet/outlet energy balance를 만족하지 않는 node에
Energy PINN을 적용하면 모델에게 데이터와 모순되는 제약을 준다.

**선택한 방법**

\[
|r_n^{E,\mathrm{true}}|\le0.05
\]

인 node만 Energy PINN에 사용한다.

**이유**

이 mask는 prediction이 나쁜 sample을 제거하는 것이 아니다. 모델 prediction과
무관하게 true stream 값으로 먼저 물리가정의 적용 가능성을 판정한다. 따라서
모델이 큰 residual을 냈다는 이유로 어려운 sample을 회피하는 누출을 막는다.

**주의점과 검증**

threshold 0.05는 물리 상수라기보다 데이터 품질 gate다. 공정·장치별 accept
ratio와 제외 node를 확인하고 threshold sensitivity를 검증해야 한다.

### 28.13 Node-only PINN schedule

**문제**

초기 random prediction으로 계산한 conservation residual은 매우 불안정할 수
있다. 처음부터 강한 PINN update를 적용하면 supervised signal을 배우기 전에
잘못된 방향으로 parameter가 크게 이동할 수 있다.

**선택한 방법**

```text
Epoch 1-5: node PINN 0%
Epoch 6-7: node PINN 50%
Epoch 8+:  node PINN 100%
```

**이유**

먼저 edge supervision으로 기본적인 stream prediction을 학습하고, 이후
conservation constraint를 점진적으로 추가하는 curriculum이다.

Density, Enthalpy, Volume edge auxiliary supervision은 정답을 직접 사용하고
node conservation보다 안정적이므로 이 schedule 대상이 아니다.

**주의점과 검증**

schedule 전환 epoch에서 `loss`, `gnorm`, `skip`이 갑자기 증가하는지 확인한다.
`no PINN`, `full from epoch 1`, `scheduled PINN`을 동일 seed로 비교해야 한다.

### 28.14 Residual normalization, clipping, Huber, gradient clipping

**문제**

유량과 enthalpy-flow는 공정·sample에 따라 scale 차이가 매우 크며, 작은
denominator와 이상치가 gradient 폭발을 일으킬 수 있다.

**선택한 방법**

- true incident-flow scale로 node residual 정규화
- denominator floor 1과 epsilon 사용
- node 및 Enthalpy/Volume residual을 \([-10,10]\)으로 제한
- Density residual을 \([-1000,1000]\)으로 제한
- Huber loss 사용
- 전체 gradient norm을 0.5로 clipping
- non-finite update skip 및 epoch 복구

**이유**

정규화는 공정 규모 차이를 줄이고, residual clipping과 Huber는 단일 이상치의
영향을 제한한다. gradient clipping은 여러 loss가 합쳐진 뒤 parameter update
자체가 지나치게 커지는 것을 막는다.

**한계**

안정화는 잘못된 수식이나 단위 문제를 해결하지 않는다. raw residual이
지속적으로 clip 경계에 몰리거나 clipping 전 gradient norm이 계속 크면,
weight보다 먼저 단위, mask, denominator를 다시 점검해야 한다.

### 28.15 매 epoch random validation 1,000개

**문제**

validation pool 전체를 매 epoch 평가하면 학습보다 validation 시간이 더 길어질
수 있다.

**선택한 방법**

매 epoch validation pool에서 random 1,000개를 선택하고 target 성분별 및
all-edge 성분별 metric을 계산한다.

**이유**

validation 비용을 제한하면서 학습 추세와 checkpoint 선택 신호를 자주 얻기
위한 절충이다.

**대가와 검증**

매 epoch sample이 바뀌므로 monitor metric에 sampling noise가 생긴다.
early stopping으로 선택된 best checkpoint는 마지막에 전체 validation/test로
재평가해야 한다. 서로 다른 모델을 비교할 때는 동일한 evaluation sample 또는
전체 독립 test를 사용해야 한다.

### 28.16 Target metric 기반 checkpoint 선택

**문제**

all-edge 평균이 좋아도 중요한 target stream 성능이 낮을 수 있다.

**선택한 방법**

`val_target_mean_r2`를 최대화하는 checkpoint를 선택하고 patience 5로 early
stopping한다.

**이유**

학습의 최종 목적이 주요 target stream 예측 개선이므로 checkpoint 선택 기준도
그 목적과 맞춘다.

**주의점과 검증**

평균 하나는 특정 성분 실패를 숨길 수 있다. checkpoint 선택 후 반드시 target
12개 property와 all-edge 14개 property를 각각 확인한다. target metric의
minimum count, SST 조건, R2 floor가 집계값에 미치는 영향도 raw 성분별 R2와
구분해야 한다.

### 28.17 Seen 5-fold와 unseen pretrain/zero-shot/transfer 분리

**문제**

sample 단위 random split 성능만으로는 새로운 공정 topology에 일반화되는지
판단할 수 없다.

**선택한 방법**

- 전체공정 5-fold: Process 1-10을 포함한 60/20/20 평가
- Pretrain: target 공정 하나를 완전히 제외하고 나머지 9개 공정으로 학습
- Zero-shot: target 공정에서 추가 학습 없이 평가
- Transfer: pretrain checkpoint를 target 공정 adaptation 데이터로 fine-tune

**이유**

세 단계는 서로 다른 질문에 답한다.

| 실험 | 확인하는 능력 |
|---|---|
| 전체공정 5-fold | 이미 본 공정 family 안의 보간 성능 |
| Zero-shot | 새 공정에 대한 직접 일반화 |
| Transfer | 적은 target 데이터로 빠르게 적응하는 능력 |

**주의점과 검증**

pretrain, adaptation, zero-shot/test 사이 sample 누출이 없어야 한다.
Zero-shot과 Transfer는 같은 pretrain checkpoint를 사용해야 공정한 비교가
된다.

### 28.18 방법론의 우선순위

현재 설계를 목적별로 묶으면 다음과 같다.

| 목적 | 핵심 방법론 |
|---|---|
| topology와 방향성 학습 | graph 표현, 양방향 FlowGNN, edge-conditioned attention |
| 깊은 GNN 안정화 | layer residual, 초기 embedding 반복 주입 |
| 공정 전체 문맥 | Set2Set global embedding |
| property 구조 반영 | grouped head, fraction softmax, thermo heads |
| 희박·광범위 target 학습 | log-fraction, scaled-log Mass Flow |
| target edge 집중 | weight 40, target별 독립 update, hard-target fill |
| 물리적 일관성 | 장치별 Mass/Component/Atom/Energy PINN |
| PINN 안정화 | true-data mask, schedule, normalization, clipping, Huber |
| 계산시간 관리 | epoch 2%+fill, validation 1,000개 |
| 일반화 검증 | seen 5-fold, unseen zero-shot, transfer |

가장 중요한 해석은 다음과 같다.

```text
모델 구조는 topology와 방향성을 학습한다.
sampling과 optimizer 구조는 중요한 target을 더 자주, 더 독립적으로 학습한다.
PINN은 예측을 대신하지 않고 물리적으로 모순된 해를 줄이는 보조 제약이다.
validation/test는 이 설계 가설이 실제 성분별 성능으로 이어졌는지 판정한다.
```

---

## 29. 최종 요약

현재 모델은 장치와 stream으로 구성된 화학공정을 5-layer 양방향 FlowGNN으로
처리한다. 실제 유체 방향의 forward message와 반대 방향의 backward message를
별도 attention으로 계산해 결합한다. 각 node는 512차원으로 표현되고, 공정
전체는 1024차원으로 요약된다.
각 stream은 출발 node, 도착 node, 전체 공정, edge 구조 정보를 합친 2051차원
정보를 이용해 예측한다.

Property head는 Condition, Fraction, Flow branch로 구성되며, fraction은
softmax를 통해 합이 1이 되도록 보장한다. 예측된 온도, 압력, 조성은 Density와
Enthalpy 예측에도 사용된다.

학습에서는 non-target edge 전체를 한 번에 update하고, target edge는 각각
독립적으로 update한다. Node PINN도 별도 update를 사용한다. 기본 train pool의
2%를 선택한 뒤 성능이 낮았던 target CH4, CO, CO2 sample을 추가해 epoch당
2,000개를 학습한다.

Validation은 매 epoch random 1,000개로 수행하며 target mean R2를 기준으로
best checkpoint와 early stopping을 결정한다. 최종 성능은 평균 하나가 아니라
target-edge와 all-edge의 성분별 R2를 기준으로 판단해야 한다.
