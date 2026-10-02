# MODEL 260805 10D FRAC1: 현재 최종 모델

## 1. 기준 정의

현재 canonical 설정은 `configs/experiment/pinn/model_260805_10d_frac1.yaml`이다.

- 출력: 10D (`Vol_Flow`, `Mole_Flow` 제외)
- fraction log loss 계수: 1.0
- GNN: 5-layer Relational Bidirectional FlowGNN
- differential encoding: 활성, `concat`
- HX inlet/outlet pair correction: 활성
- FeedHead: 64D
- target weight: 5
- epoch sampling: base 1,000 + hard 1,000 = 2,000
- validation: 매 epoch 무작위 1,000
- early stopping patience: 5

### 1.1 모델 목적과 범위

본 모델은 서로 다른 화학공정 flowsheet를 하나의 공유 파라미터 집합으로 처리하는 directed
graph surrogate이다. 장치(unit operation)를 node, 물질 stream을 directed edge로 표현하고,
각 예측 가능 edge마다 10개 물성의 벡터를 출력한다. Process ID에 따라 prediction head를
선택하지 않으며, 모든 공정과 모든 edge가 동일한 encoder와 동일한 property head를 공유한다.

### 1.2 기호와 차원

공정 graph를 $\mathcal{G}=(\mathcal{V},\mathcal{E})$라 둔다. Directed edge
$e=(i,j)\in\mathcal{E}$는 source node $i$에서 destination node $j$로 흐르는 stream이다.

| 기호 | 의미 | 현재 차원 |
|---|---|---:|
| $N=|\mathcal V|$ | graph 내 node 수 | graph별 가변 |
| $E=|\mathcal E|$ | graph 내 edge 수 | graph별 가변 |
| $L$ | message-passing layer 수 | 5 |
| $d$ | node/edge hidden dimension | 384 |
| $\mathbf h_i^{(\ell)}$ | layer $\ell$의 node state | $\mathbb R^{384}$ |
| $\mathbf e_{ij}$ | 모든 layer에서 공유하는 static edge state | $\mathbb R^{384}$ |
| $\mathbf g$ | projected graph representation | $\mathbb R^{384}$ |
| $\mathbf d_{ij}$ | property head에 들어가는 edge descriptor | $\mathbb R^{1600}$ |
| $\mathbf z_{ij}$ | shared edge latent | $\mathbb R^{384}$ |
| $\hat{\mathbf y}_{ij}$ | 최종 edge property prediction | $\mathbb R^{10}$ |

Batch는 여러 graph의 node와 edge를 연결하지 않은 disjoint union으로 구성한다. 모든 scatter,
attention normalization, Set2Set pooling 및 HX pairing은 graph 경계를 넘지 않는다.

## 2. 전체 차원 흐름

```text
[Node input]
Node semantic role embedding 32
Unit type embedding          24
Operating values             18
Operating mask               18
Operating encoder            36 -> 64
Concat                       32 + 24 + 64 = 120
Node encoder                 120 -> 384 -> 384
= initial node state h0: 384D
        |
        v
[Static edge input]
Stream semantic role embedding 24
Stream ID embedding            32
Structural attribute encoder    3 -> 16
Concat                          72
Edge encoder                    72 -> 384 -> 384
= static edge state e: 384D
        |
        v
[5-layer Relational Bidirectional FlowGNN]
Forward edge-conditioned message/attention
Backward edge-conditioned message/attention
Directional differential encoding
Forward/backward fusion -> 384D
Layer interpolation residual 0.5
Per-layer h0 reinjection 0.05
        |
        +------------------------------+
        |                              |
        v                              v
Local node state 384             Set2Set 384 -> 768
                                 Global projection -> 384

[HX edge correction]
Current edge 384 + paired edge 384 + side 16 = 784
784 -> 256 -> 384
Gated residual, initial gate 0.05

[FeedHead]
Scaled CH4/AIR/WATER values + masks = 6
6 -> 32 -> 64

[Edge descriptor]
Source node 384
Destination node 384
Global graph 384
HX-corrected edge 384
Direct feed 64
= 1600D

[Shared decoder]
1600 -> 768 -> 384

[Hierarchical property head]
Condition: 384 -> 96 -> Temp, Pres
Fraction:  384 -> 128 -> 7 fractions -> softmax(T=0.5)
Mass:      384 -> 96 -> Mass_Flow

[Final output]
Temp, Pres,
Frac_H2O, Frac_H2, Frac_CH4, Frac_CO2, Frac_CO, Frac_O2, Frac_N2,
Mass_Flow
= 10D
```

## 3. 입력 표현

### 3.1 Node

Node $i$의 입력은 공정 간 공유 semantic role $r_i$, unit type $u_i$, operating value
$\mathbf o_i\in\mathbb R^{18}$와 관측 mask $\mathbf b_i\in\{0,1\}^{18}$로 구성한다. 값이
0인 경우와 관측되지 않은 경우를 구분하기 위해 값과 mask를 별도 채널로 유지한다.

$$
\mathbf c_i^{\mathrm{op}}
=\phi_{\mathrm{op}}([\mathbf o_i\Vert\mathbf b_i])\in\mathbb R^{64},
$$

$$
\mathbf x_i
= [\operatorname{Emb}_{r}(r_i)\Vert
   \operatorname{Emb}_{u}(u_i)\Vert
   \mathbf c_i^{\mathrm{op}}]
\in\mathbb R^{120},
\qquad
\mathbf h_i^{(0)}=\phi_{\mathrm{node}}(\mathbf x_i)\in\mathbb R^{384}.
$$

$\phi_{\mathrm{node}}$는 `120 -> 384 -> 384`이며, role과 unit embedding은 각각 32D와
24D이다. Feed가 알려진 경우 CH4/AIR/WATER 공급량과 다음 두 비율을 operating feature에
포함한다.

$$
q_{A/C}=\log\frac{F_{\mathrm{AIR}}+\varepsilon_f}
                         {F_{\mathrm{CH4}}+\varepsilon_f},
\qquad
q_{W/C}=\log\frac{F_{\mathrm{WATER}}+\varepsilon_f}
                         {F_{\mathrm{CH4}}+\varepsilon_f},
\qquad \varepsilon_f=10^{-6}.
$$

CH4 공급량이 없거나 0에 가까운 경우에는 유한한 대체값만으로 의미를 만들지 않고 관측 mask를
함께 사용한다. 이 정보는 V_INPUT과 BURNER node에 주입되며, 이후 message passing과 global
pooling을 통해 국소 반응 조건과 graph 전체 운전 조건에 모두 반영된다.

### 3.2 Edge

Edge $(i,j)$의 입력은 stream semantic role $r_{ij}^{e}$, graph 안에서 stream을 구분하는 ID
$s_{ij}$, 3D structural attribute $\mathbf a_{ij}$로 구성한다.

$$
\mathbf c_{ij}^{\mathrm{str}}=\phi_{\mathrm{str}}(\mathbf a_{ij})\in\mathbb R^{16},
$$

$$
\mathbf x_{ij}^{e}
= [\operatorname{Emb}_{e}(r_{ij}^{e})\Vert
   \operatorname{Emb}_{s}(s_{ij})\Vert
   \mathbf c_{ij}^{\mathrm{str}}]
\in\mathbb R^{72},
\qquad
\mathbf e_{ij}=\phi_{\mathrm{edge}}(\mathbf x_{ij}^{e})\in\mathbb R^{384}.
$$

각 항의 차원은 `24 + 32 + 16 = 72`이고 edge encoder는 `72 -> 384 -> 384`이다.
$\mathbf e_{ij}$는 layer마다 갱신하지 않는 static state이며, 모든 message와 attention logit에
반복 주입한다. 이는 깊은 message passing에서 서로 평행하거나 같은 node pair를 연결하는 stream의
정체성이 소실되는 것을 줄이기 위한 선택이다. Stream ID는 graph 내부의 edge 구별자일 뿐,
Process ID에 따른 head routing이나 process-specific output head로 사용되지 않는다.

## 4. Relational Bidirectional FlowGNN

각 layer는 물리적 stream 방향을 따르는 forward branch와 그 역방향을 따르는 backward
branch를 독립적으로 계산한다. 두 branch는 파라미터를 공유하지 않는다. Attention head는 방향별
하나의 scalar head이며 message와 attention logit 모두 static edge state를 조건으로 사용한다.

### 4.1 Forward message와 sender-wise Flow Attention

Node $j$로 들어오는 물리적 upstream 이웃 집합을
$\mathcal N_{\mathrm{in}}(j)=\{i:(i,j)\in\mathcal E\}$, node $i$에서 나가는 downstream
이웃 집합을 $\mathcal N_{\mathrm{out}}(i)=\{j:(i,j)\in\mathcal E\}$라 둔다. Layer $\ell$에서:

$$
\mathbf m_{ij}^{(\ell,\rightarrow)}
=\phi_m^{(\ell,\rightarrow)}
\left([\mathbf h_i^{(\ell-1)}\Vert\mathbf e_{ij}]\right),
$$

$$
s_{ij}^{(\ell,\rightarrow)}
=\phi_a^{(\ell,\rightarrow)}
\left([\mathbf h_i^{(\ell-1)}\Vert
       \mathbf h_j^{(\ell-1)}\Vert\mathbf e_{ij}]\right),
$$

$$
\beta_{ij}^{(\ell,\rightarrow)}
=\frac{\exp(s_{ij}^{(\ell,\rightarrow)})}
       {\sum_{k\in\mathcal N_{\mathrm{out}}(i)}
        \exp(s_{ik}^{(\ell,\rightarrow)})},
\qquad
\mathbf a_j^{(\ell,\rightarrow)}
=\sum_{i\in\mathcal N_{\mathrm{in}}(j)}
\beta_{ij}^{(\ell,\rightarrow)}\mathbf m_{ij}^{(\ell,\rightarrow)}.
$$

따라서 각 sender $i$에 대해
$\sum_{j\in\mathcal N_{\mathrm{out}}(i)}\beta_{ij}^{(\ell,\rightarrow)}=1$이다.
Message는 여전히 물리적 방향 $i\rightarrow j$로 전달되며, 뒤집히는 것은 message passing 방향이
아니라 softmax normalization 기준뿐이다.

구현상 $\phi_m$은 `768 -> 512 -> 384`, $\phi_a$는 `1152 -> 256 -> 1`이다.
Message MLP는 `Linear-LayerNorm-GELU-Dropout(0.1)-Linear`, attention MLP는
`Linear-GELU-Linear`이다. Softmax는 source index별 segment softmax다.

### 4.2 Backward message와 source-wise attention

Backward branch는 같은 physical edge $(i,j)$를 $j\rightarrow i$ 방향으로 읽는다. Source $i$에서
물리적으로 나가는 이웃 집합을 $\mathcal N_{\mathrm{out}}(i)=\{j:(i,j)\in\mathcal E\}$라 두면:

$$
\mathbf m_{ji}^{(\ell,\leftarrow)}
=\phi_m^{(\ell,\leftarrow)}
\left([\mathbf h_j^{(\ell-1)}\Vert\mathbf e_{ij}]\right),
$$

$$
s_{ji}^{(\ell,\leftarrow)}
=\phi_a^{(\ell,\leftarrow)}
\left([\mathbf h_j^{(\ell-1)}\Vert
       \mathbf h_i^{(\ell-1)}\Vert\mathbf e_{ij}]\right),
$$

$$
\alpha_{ji}^{(\ell,\leftarrow)}
=\frac{\exp(s_{ji}^{(\ell,\leftarrow)})}
       {\sum_{k\in\mathcal N_{\mathrm{out}}(i)}
        \exp(s_{ki}^{(\ell,\leftarrow)})},
\qquad
\mathbf a_i^{(\ell,\leftarrow)}
=\sum_{j\in\mathcal N_{\mathrm{out}}(i)}
\alpha_{ji}^{(\ell,\leftarrow)}\mathbf m_{ji}^{(\ell,\leftarrow)}.
$$

따라서 forward attention은 같은 physical source에서 나가는 stream 사이에 flow/message를
분배한다. Backward attention은 역방향 message가 도착하는 같은 physical source를 기준으로
정규화된다. Forward는 feed 및 upstream
운전조건을 downstream으로 전달하고, backward는 downstream constraint와 recycle 영향을
upstream state에 반영한다.

### 4.3 Differential encoding과 방향별 update

단순 aggregate가 아니라 현재 state 대비 aggregate의 차이를 학습한다. 방향
$q\in\{\rightarrow,\leftarrow\}$에 대해:

$$
\boldsymbol\delta_i^{(\ell,q)}
=\mathbf a_i^{(\ell,q)}-\mathbf h_i^{(\ell-1)},
\qquad
\tilde{\boldsymbol\delta}_i^{(\ell,q)}
=\phi_\Delta^{(\ell,q)}(\boldsymbol\delta_i^{(\ell,q)}),
$$

$$
\mathbf u_i^{(\ell,q)}
=\phi_u^{(\ell,q)}
\left([\mathbf a_i^{(\ell,q)}\Vert
\tilde{\boldsymbol\delta}_i^{(\ell,q)}]\right).
$$

$\phi_\Delta$는 `384 -> 512 -> 384`, $\phi_u$는 `768 -> 512 -> 384`이며 둘 다
`Linear-LayerNorm-GELU-Dropout(0.1)-Linear`이다. 현재 `diff_mode=concat`이므로 위 concat
식을 사용한다. $\boldsymbol\delta$는 이웃이 전달한 상태가 현재 node state를 어느 방향과 크기로
수정해야 하는지를 명시한다.

양방향 update는 다음 fusion MLP로 합친다.

$$
\tilde{\mathbf h}_i^{(\ell)}
=\phi_f^{(\ell)}
\left([\mathbf u_i^{(\ell,\rightarrow)}\Vert
       \mathbf u_i^{(\ell,\leftarrow)}]\right),
\qquad \phi_f:768\rightarrow512\rightarrow384.
$$

방향별 DiffEncoder와 update MLP는 서로 독립적이다. Static edge state $\mathbf e_{ij}$는 layer별로
갱신하지 않고 5개 layer에 반복 주입한다.

학습 로그에서 다음 문자열로 활성 상태를 확인할 수 있다.

```text
gnn=relational_bidirectional differential=concat
```

Differential encoder parameter는 3,951,360개다. 실제 생성 모델 기준 전체 parameter는
26,704,101개, GNN layer 부분은 21,699,210개다.

### 4.4 Layer residual과 initial-state reinjection

Fusion candidate 뒤에 interpolation residual을 적용하고, 이어서 최초 node encoding을 직접
재주입한다. 실제 연산 순서는 다음과 같다.

$$
\bar{\mathbf h}_i^{(\ell)}
=\mathbf h_i^{(\ell-1)}
+\beta\left(\tilde{\mathbf h}_i^{(\ell)}-\mathbf h_i^{(\ell-1)}\right),
\qquad \beta=0.5,
$$

$$
\mathbf h_i^{(\ell)}
=\bar{\mathbf h}_i^{(\ell)}+\gamma\mathbf h_i^{(0)},
\qquad \gamma=0.05,
\qquad \ell=1,\ldots,5.
$$

주의할 점은 현재 구현의 reinjection이 별도 `InitialResidualMLP`를 거치지 않고
$\gamma\mathbf h_i^{(0)}$를 직접 더한다는 것이다. `initial_residual_mode=per_layer`이므로 마지막
layer 뒤에 $\mathbf h^{(0)}$를 한 번 더 더하지 않는다. `layer_residual_norm=false`이므로 이 두
연산 직후 별도 LayerNorm도 없다.

Residual은 직전 layer의 학습 상태와 gradient path를 보존하고, reinjection은 role, unit type,
operating condition이 깊은 message passing에서 희석되는 것을 방지한다.

### 4.5 Set2Set global representation

마지막 local state 집합 $\{\mathbf h_i^{(L)}\}$에서 graph별로 3-step Set2Set을 수행한다.
$\mathbf q_t^*$를 LSTM에 다시 넣는 표준 Set2Set recurrence는

$$
\mathbf q_t=\operatorname{LSTM}(\mathbf q_{t-1}^{*}),
\qquad
\alpha_{i,t}=\frac{\exp(\mathbf h_i^{(L)\top}\mathbf q_t)}
{\sum_{k\in\mathcal V_g}\exp(\mathbf h_k^{(L)\top}\mathbf q_t)},
$$

$$
\mathbf r_t=\sum_{i\in\mathcal V_g}\alpha_{i,t}\mathbf h_i^{(L)},
\qquad
\mathbf q_t^*=[\mathbf q_t\Vert\mathbf r_t]\in\mathbb R^{768}.
$$

최종 $\mathbf q_3^*$를 projection하여 $\mathbf g\in\mathbb R^{384}$를 만든다. 단순 평균과
달리 학습된 query가 현재 예측에 중요한 unit state를 선택할 수 있으며, pooling은 batch 전체가
아니라 각 graph의 node 집합 $\mathcal V_g$ 안에서만 수행된다.

## 5. HX pair와 FeedHead

### 5.1 HX pair relation

열교환기에서는 hot/cold 유로의 inlet과 outlet이 같은 HX node에 연결되므로 일반 incidence만으로
어떤 outlet이 어떤 inlet과 대응하는지 모호할 수 있다. Metadata가 제공하는 대칭적인 one-to-one
pair mapping을 $p(e)$라 하고, hot/cold side embedding을 $\mathbf s_e\in\mathbb R^{16}$라 둔다.

$$
\Delta\mathbf e_e
=\phi_{\mathrm{HX}}
\left([\mathbf e_e\Vert\mathbf e_{p(e)}\Vert\mathbf s_e]\right),
\qquad
\phi_{\mathrm{HX}}:784\rightarrow256\rightarrow384,
$$

$$
g_{\mathrm{HX}}=\sigma(a_{\mathrm{HX}}),
\qquad
\bar{\mathbf e}_e
=\operatorname{LayerNorm}
\left(\mathbf e_e+g_{\mathrm{HX}}\Delta\mathbf e_e\right).
$$

모든 paired HX edge가 공유하는 scalar gate $g_{\mathrm{HX}}$는 초기값이 0.05가 되도록 logit을
초기화한다. 따라서 학습 시작 시 기존 edge representation을 거의
유지하면서, 데이터가 지지하는 만큼만 paired-stream correction을 키운다. Pairing은 같은 graph
안에서만 적용되고 symmetric/one-to-one 조건을 검사한다. 이 correction은 FlowGNN 내부가 아니라
최종 edge descriptor를 만들기 직전에 적용된다.

### 5.2 Direct FeedHead

CH4/AIR/WATER의 train-only scaler 적용값 $\tilde{\mathbf f}\in\mathbb R^3$와 관측 mask
$\mathbf b_f\in\{0,1\}^3$를 직접 feed encoder에 넣는다.

$$
\mathbf f_g
=\phi_f([\tilde{\mathbf f}\Vert\mathbf b_f])
\in\mathbb R^{64},
\qquad
\phi_f:6\rightarrow32\rightarrow64.
$$

이 64D 표현은 graph의 모든 prediction edge에 broadcast된다. Log feed ratio는 node operating
path에 들어가지만, direct FeedHead에는 의도적으로 raw scaled feed와 mask만 넣는다. 즉 같은
정보를 그대로 두 번 복제하지 않고, GNN을 거친 국소/전역 경로와 head 직전의 직접 조건 경로를
서로 보완적으로 사용한다.

### 5.3 최종 edge descriptor

Edge $e=(i,j)$의 head 입력은

$$
\mathbf d_e=
[\mathbf h_i^{(L)}\Vert
 \mathbf h_j^{(L)}\Vert
 \mathbf g\Vert
 \bar{\mathbf e}_e\Vert
 \mathbf f_g]
\in\mathbb R^{1600},
$$

$$
1600=384_{\rm src}+384_{\rm dst}+384_{\rm global}
     +384_{\rm edge}+64_{\rm feed}.
$$

Batch에서 edge 순서는 원래 prediction edge 순서를 유지하며, descriptor를 만들기 위한 별도의
process-specific head routing은 없다.

## 6. Prediction head

### 6.1 Shared decoder

모든 공정과 모든 edge가 같은 decoder를 공유한다.

$$
\mathbf u_e
=\operatorname{Drop}_{0.1}
\left(\operatorname{GELU}
\left(\operatorname{LN}(W_1\mathbf d_e+\mathbf b_1)\right)\right)
\in\mathbb R^{768},
$$

$$
\mathbf z_e
=\operatorname{LN}
\left(
\operatorname{Drop}_{0.1}
\left(\operatorname{GELU}
\left(\operatorname{LN}(W_2\mathbf u_e+\mathbf b_2)\right)\right)
\right)
\in\mathbb R^{384}.
$$

실제 shared path는 `1600 -> 768 -> 384`이고 decoder residual은 비활성이다. 여기서
$\mathbf z_e$가 condition, fraction, mass branch가 함께 사용하는 최종 shared edge latent다.

### 6.2 Property branches

Condition과 Mass Flow는 독립 branch를 사용한다. 각 branch의 hidden block은
`Linear -> LayerNorm -> GELU -> Dropout(0.1)`이고 마지막 output projection은 Linear다.

$$
[\hat T_e,\hat P_e]
=\phi_c(\mathbf z_e),
\qquad \phi_c:384\rightarrow96\rightarrow2,
$$

$$
\hat z_e^m=\phi_m(\mathbf z_e),
\qquad \phi_m:384\rightarrow96\rightarrow1.
$$

Fraction logits $\boldsymbol\ell_e\in\mathbb R^7$에는 temperature softmax를 적용한다.

$$
\boldsymbol\ell_e=\phi_x(\mathbf z_e),
\qquad \phi_x:384\rightarrow128\rightarrow7,
$$

$$
\hat x_{e,k}
=\frac{\exp(\ell_{e,k}/\tau_x)}
       {\sum_{r=1}^{7}\exp(\ell_{e,r}/\tau_x)},
\qquad \tau_x=0.5.
$$

따라서 수치 오차 범위에서 $\hat x_{e,k}\ge0$이고 $\sum_k\hat x_{e,k}=1$이다. Species 순서는
`H2O, H2, CH4, CO2, CO, O2, N2`다.

Mass branch의 직접 출력은 물리 유량이 아니라 scaled-log 좌표다. 현재 변환과 역변환은

$$
\mathcal T_m(m)=2\log(\max(m,0)+10^{-8}),
$$

$$
\hat m_e
=\max\left\{\exp\left(\frac{\hat z_e^m}{2}\right)-10^{-8},0\right\}.
$$

최종 출력 순서는

$$
\hat{\mathbf y}_e=
[\hat T_e,\hat P_e,
 \hat x_{e,\mathrm{H2O}},\hat x_{e,\mathrm{H2}},
 \hat x_{e,\mathrm{CH4}},\hat x_{e,\mathrm{CO2}},
 \hat x_{e,\mathrm{CO}},\hat x_{e,\mathrm{O2}},
 \hat x_{e,\mathrm{N2}},\hat m_e]\in\mathbb R^{10}.
$$

`Vol_Flow`와 `Mole_Flow`는 현재 output, supervised loss, monitor metric에서 모두 제외된다.
Density와 Enthalpy도 이 최종 10D head에서는 예측하지 않는다.

## 7. Supervised loss

### 7.1 기본 robust loss

PyTorch 기본 $\beta=1$인 SmoothL1을 사용한다.

$$
\rho_{\mathrm{SL1}}(r)=
\begin{cases}
\frac12r^2,& |r|<1,\\
|r|-\frac12,& |r|\ge1.
\end{cases}
$$

Temp와 Pres는 configured training coordinate, Mass Flow는 위의 $\mathcal T_m$ 좌표에서 비교한다.
유효 target mask를 $M_{e,p}$라 하고 $w_T=w_P=1$, $w_m=2$라 두면 main loss는

$$
\mathcal L_{e}^{\mathrm{main}}
=\frac{\sum_{p\in\{T,P,m\}}
M_{e,p}w_p\,
\rho_{\mathrm{SL1}}(\hat z_{e,p}-z_{e,p})}
{\max\left(1,\sum_{p\in\{T,P,m\}}M_{e,p}\right)}.
$$

여기서 Mass Flow의 계수 2는 분자 loss contribution에만 곱해지고 유효 property 수를 세는
분모에는 들어가지 않는다.

### 7.2 Fraction log loss

Fraction은 physical-scale main loss에서 **제외**하고 별도의 log-space loss만 사용한다. 즉
physical fraction loss와 log fraction loss를 중복 합산하지 않는다. True Mass Flow가
$10^{-8}$보다 큰 row와 유효 성분에 대해

$$
\mathcal I_e=
\left\{k:M_{e,k}=1,\ m_e^{\rm true}>10^{-8}\right\},
$$

$$
\mathcal L_e^{\mathrm{frac}}
=\frac{1}{|\mathcal I_e|}
\sum_{k\in\mathcal I_e}
\rho_{\mathrm{SL1}}
\left(
\log(\hat x_{e,k}+10^{-6})-
\log(x_{e,k}+10^{-6})
\right).
$$

$|\mathcal I_e|=0$이면 해당 row의 fraction term은 0으로 두고 집계에서 제외한다. 최종 edge
supervision은

$$
\boxed{
\mathcal L_e^{\mathrm{sup}}
=\mathcal L_e^{\mathrm{main}}
+1.0\,\mathcal L_e^{\mathrm{frac}}
}
$$

이다. 과거의 fraction 계수 0.75는 사용하지 않으며 현재 계수는 1.0이다. CLR loss, fraction
closure penalty, physical Mass Flow auxiliary loss도 모두 비활성이다.

### 7.3 Target-edge weighting

Edge weight는

$$
w_e^{\mathrm{edge}}=
\begin{cases}
5,&e\in\mathcal E_{\mathrm{target}},\\
1,&e\notin\mathcal E_{\mathrm{target}}
\end{cases}
$$

이다. 다만 현재 `sample_hybrid_target_edge_step_pi`는 모든 edge를 한 번에 평균한 단일 loss가
아니라 non-target macro step과 target-edge별 step을 분리한다. 따라서 이 값은 실제 해당
supervised step의 scaling으로 이해해야 하며, 단순히 dataset 전체에서 target row를 5배 복제한
것과 같지 않다.

## 8. Node PINN

### 8.1 적용 node 집합

입력과 출력 edge가 모두 존재하고 boundary unit이 아닌 node만 internal node로 정의한다.

$$
\mathcal V_{\mathrm{int}}
=\{v:\deg^-(v)>0,\ \deg^+(v)>0,\ v\notin\mathcal V_{\mathrm{boundary}}\}.
$$

Mass balance는 internal node 전체, component balance는 non-reactive internal node, atom balance는
reactive internal node에 적용한다. Metadata의 개별 제외 mask와 finite-value 검사를 추가로
적용하며, graph batch 경계를 넘어 유량을 합산하지 않는다.

### 8.2 일반 balance residual과 정규화

Edge quantity $\mathbf q_e$에 대해 node $v$의 predicted balance와 true-flow scale을

$$
\mathbf B_v(\hat{\mathbf q})
=\sum_{e\in\delta^-(v)}\hat{\mathbf q}_e
-\sum_{e\in\delta^+(v)}\hat{\mathbf q}_e,
$$

$$
\mathbf S_v(\mathbf q)
=\sum_{e\in\delta^-(v)}|\mathbf q_e|
+\sum_{e\in\delta^+(v)}|\mathbf q_e|
$$

로 정의한다. 현재 relative residual은

$$
\mathbf r_v(\mathbf q)
=\frac{\mathbf B_v(\hat{\mathbf q})}
{\operatorname{stopgrad}(\max(\mathbf S_v(\mathbf q),1)+10^{-6})}.
$$

따라서 정규화 분모는 정답으로부터 계산되어 gradient가 흐르지 않는다. 유효 residual은 먼저
$[-10,10]$으로 clipping한 뒤 $\delta=0.5$인 Huber loss를 적용한다.

$$
\rho_{\delta}(r)=
\begin{cases}
\frac12r^2,&|r|\le\delta,\\
\delta(|r|-\frac12\delta),&|r|>\delta,
\end{cases}
\qquad \delta=0.5.
$$

각 PINN term은 유효 node-component residual에 대한 평균이다. Component/atom의 configured
valid-scale threshold는 $10^{-6}$이지만 현재 scale floor가 1이므로 실제로는 이 threshold보다
floor가 우선한다. NaN/Inf 값이 연결된 node와 metadata mask에서 제외된 node는 유효 집합에서
제외한다.

### 8.3 Mass balance

물리 단위로 inverse-transform한 Mass Flow를 사용한다.

$$
r_v^{m}
=\frac{
\sum_{e\in\delta^-(v)}\hat m_e-
\sum_{e\in\delta^+(v)}\hat m_e}
{\operatorname{stopgrad}
\left(\max\left(
\sum_{e\in\delta^-(v)}|m_e|+
\sum_{e\in\delta^+(v)}|m_e|,1\right)+10^{-6}\right)},
$$

$$
\mathcal L_{\mathrm{mass}}
=\operatorname{mean}_{v\in\mathcal V_{\mathrm{int}}}
\rho_{0.5}(\operatorname{clip}(r_v^m,-10,10)).
$$

### 8.4 Component balance

현재 구현은 예측 mole fraction과 mixture molecular weight로 component molar flow를 만든다.
Species molecular weight를 $M_k$라 하면

$$
\hat M_e^{\mathrm{mix}}=\sum_{k=1}^{7}\hat x_{e,k}M_k,
\qquad
\hat n_e^{\mathrm{tot}}
=\frac{\hat m_e}{\max(\hat M_e^{\mathrm{mix}},10^{-6})},
$$

$$
\hat n_{e,k}=\hat n_e^{\mathrm{tot}}\hat x_{e,k}.
$$

각 non-reactive internal node와 species에 대해

$$
B_{v,k}^{\mathrm{comp}}
=\sum_{e\in\delta^-(v)}\hat n_{e,k}
-\sum_{e\in\delta^+(v)}\hat n_{e,k}
$$

를 만들고 8.2절과 같은 true-flow scale, clipping, Huber reduction을 적용하여
$\mathcal L_{\mathrm{component}}$를 얻는다.

### 8.5 Atom balance

Species-atom incidence matrix를 $A\in\mathbb R^{7\times n_a}$라 하면 edge별 atom molar flow는

$$
\hat{\mathbf a}_e=\hat{\mathbf n}_eA.
$$

Reactive internal node에서는 반응 전후에 component 자체는 변할 수 있으므로 component balance
대신 원소 보존을 적용한다.

$$
\mathbf B_v^{\mathrm{atom}}
=\sum_{e\in\delta^-(v)}\hat{\mathbf a}_e
-\sum_{e\in\delta^+(v)}\hat{\mathbf a}_e.
$$

이 역시 같은 정규화와 robust reduction을 거쳐 $\mathcal L_{\mathrm{atom}}$이 된다.

### 8.6 PINN 합성과 schedule

현재 활성 PINN은 정확히 다음 세 항이다.

$$
\boxed{
\mathcal L_{\mathrm{PINN}}
=1.0\mathcal L_{\mathrm{mass}}
+1.5\times10^{-7}\mathcal L_{\mathrm{component}}
+0.2\mathcal L_{\mathrm{atom}}
}
$$

Energy, Density, Enthalpy, Volume 관련 loss는 모두 weight 0 또는 disabled다. Epoch $t$의 node-only
schedule multiplier는

$$
s(t)=
\begin{cases}
0,&1\le t\le5,\\
0.5,&6\le t\le7,\\
1,&t\ge8.
\end{cases}
$$

Node optimizer step은 supervised anchor와 결합된다.

$$
\boxed{
\mathcal L_{\mathrm{node-step}}
=1.0\mathcal L_{\mathrm{anchor}}
+0.05\,s(t)\mathcal L_{\mathrm{PINN}}
}
$$

따라서 epoch 1부터 PINN raw diagnostic 값이 계산되어 로그에 보일 수 있어도, epoch 1--5에는
$s(t)=0$이므로 backprop objective에는 기여하지 않는다. 이 warm-up은 초기의 부정확한 물리 예측이
큰 residual gradient로 supervision 학습을 방해하는 것을 줄이기 위한 것이다.

## 9. Optimizer update 구조

현재 backward mode는 `sample_hybrid_target_edge_step_pi`이고 batch size는 graph 1개다. 한 graph
sample에서 가능한 update는 다음 순서로 분리된다.

1. Non-target edge group의 macro-mean supervised update
2. Canonical edge ID 순서의 target-edge별 supervised update
3. 같은 graph의 supervised anchor와 PINN을 결합한 node update

이 구조는 target edge 수가 많은 graph가 단순 row 평균만으로 과도하게 지배하는 것을 막고, 각
target edge에 직접 gradient 기회를 부여한다. 동시에 non-target edge 표현과 node-level 물리
일관성도 별도의 update로 유지한다. 로그의 `updates`, `t`, `n`, `node`, `fwd`, `skip`은 각각
실제로 수행된 optimizer update와 target/non-target/node step 및 forward/skip 수를 뜻한다.

Optimizer는 AdamW, learning rate $10^{-4}$, weight decay $10^{-5}$와 cosine learning-rate
schedule을 사용한다. Global gradient norm은 0.5로 clipping한다. 현재 mixed precision은
비활성화되어 학습 tensor는 FP32를 사용한다.

## 10. Sampling과 validation

### 10.1 Epoch training sampler

매 epoch training graph는 총 2,000개이며 pool 전체를 줄인 것이 아니라 train pool에서 epoch별로
비복원 추출하는 sample budget이다.

| 구성 | 개수 |
|---|---:|
| 기본 분포 sampling | 1,000 |
| Target CO hard | 400 |
| Target Mass_Flow hard | 400 |
| Target CH4 hard | 200 |
| 합계 | **2,000** |

Mass hard 400개는 일반 고유량 240개와 극고유량 160개로 나눈다. 기본 1,000개는 hard target
값만 좇지 않고 공정 간 노출과 원래 분포의 다양성을 보존하는 역할을 한다. Hard pool은 반복적으로
저조했던 CO, Mass Flow, CH4 영역의 gradient 빈도를 보완한다. 후보가 부족하면 남은 budget을
random sample로 채우고, 한 epoch 안에서는 가능한 한 중복을 제거한다. 따라서 hard sampling은
loss weight를 바꾸는 것이 아니라 해당 graph를 optimizer에 노출하는 빈도를 바꾼다.

### 10.2 Validation과 early stopping

Validation은 고정된 validation pool 중 매 epoch seed 기반 random 1,000개를 사용한다. Training
hard sampler와 validation sampler는 독립이므로 validation 분포를 hard case 중심으로 왜곡하지
않는다. Early stopping 설정은 다음과 같다.

| 항목 | 값 |
|---|---|
| monitor | `val_target_edge_property_mean_r2` |
| mode | `max` |
| patience | 5 validation epochs |
| validation interval | 매 epoch |

Best checkpoint는 monitor가 가장 높은 epoch, last checkpoint는 실제 마지막 epoch를 나타낸다.

## 11. 저장 지표와 산출물

### 11.1 R2 정의

Property $p$의 raw coefficient of determination은

$$
R_p^2=1-
\frac{\sum_{n\in\mathcal D_p}(y_{n,p}-\hat y_{n,p})^2}
     {\sum_{n\in\mathcal D_p}(y_{n,p}-\bar y_p)^2},
\qquad
\bar y_p=\frac1{|\mathcal D_p|}\sum_{n\in\mathcal D_p}y_{n,p}.
$$

R2 분모에는 epsilon을 더하지 않는다. SST가 정확히 0이면 raw R2는 수학적으로 정의되지 않는다.
Target monitor 집계에는 최소 count 1, SST threshold $10^{-6}$, R2 floor 0.1 정책이 별도로
적용된다. Checkpoint 선택은 10개 pooled Target Property R2의 산술평균인
`target_edge_property_mean_r2`를 사용한다. Strict target-row macro인 `target_mean_r2`는
별도 진단 지표이므로 두 값을 같은 숫자로 해석하지 않는다. 이 10개는 `Temp`, `Pres`,
`Mass_Flow`, `Frac_H2O`, `Frac_H2`, `Frac_CH4`, `Frac_CO2`, `Frac_CO`, `Frac_O2`,
`Frac_N2`이다. `Mole_Flow`, `Vol_Flow`, `Density`, `Enthalpy`는 예측, loss, metric,
산출물에서 모두 제외한다.

Fraction metric은 true `Mass_Flow > 1e-8`인 edge만 사용한다. 이는 유량이 0인 stream에서
조성이 물리적으로 정의되지 않거나 임의값으로 기록된 사례가 All-edge R2를 왜곡하는 것을 막는다.

### 11.2 지표 namespace와 주요 파일

- Target-edge property별 R2와 target aggregate
- All-edge property별 R2
- zero-flow fraction mask의 포함/제외 count
- epoch별 supervised loss, PINN raw/weighted term, learning rate
- `metrics_per_epoch.csv`
- 평가 runner가 요청한 property/edge metric CSV
- `best.pt`, `last.pt`
- best actual-vs-prediction plot과 실행/설정 metadata

Train loss는 backprop objective, validation R2는 checkpoint monitor, final test R2는 선택된 checkpoint의
일반화 성능이므로 서로 다른 의미를 가진다. 논문 표에는 test metric을 사용하고 validation metric을
test 결과처럼 보고하지 않는다.

## 12. Checkpoint 호환성

현재 모델에는 방향별 differential encoder parameter가 실제 `state_dict`에 포함된다. 이 encoder가
없던 checkpoint 또는 11D/Vol-Flow head checkpoint는 현재 10D 구조와 key/shape가 다르므로
`strict=True` resume에 사용할 수 없다. 임의로 `strict=False`를 사용하면 일부 parameter가 random
initialization인 채 평가될 수 있으므로 논문용 결과에는 허용하지 않는다.

동일한 10D 구조, differential mode, FeedHead 64D, HX pair 설정으로 저장한 checkpoint끼리는
pretrain, zero-shot, full fine-tuning transfer에 공유할 수 있다. Zero-shot/transfer 전에는 checkpoint
metadata의 output schema와 mass transform도 함께 일치하는지 확인해야 한다.

## 13. 설계 선택 요약

| 방법 | 적용 이유 | 기대 효과 |
|---|---|
| Bidirectional edge-conditioned attention | 공정의 정방향 전달과 recycle/하류 제약을 함께 표현 | 방향별 유량 의존성 학습 |
| Differential encoding | aggregate 자체뿐 아니라 현재 state 대비 변화량을 입력 | 작은 운전조건 변화와 update 방향 구분 |
| Layer interpolation residual | 5-layer GNN의 급격한 state 변형 억제 | gradient 전달 및 안정성 개선 |
| Initial-state reinjection | role/unit/operating 정보의 깊이별 희석 방지 | over-smoothing 완화 |
| Static edge reuse | 모든 layer에 stream identity 재제공 | 평행 edge와 stream 역할 구별 |
| Set2Set global context | 단순 평균 대신 중요한 unit을 학습적으로 선택 | graph 운전상태 요약 |
| HX pair gated correction | incidence만으로 모호한 inlet/outlet 유로 관계 제공 | HX stream pairing 구별 |
| Direct FeedHead | GNN을 거치며 희석될 수 있는 feed를 head에 직접 제공 | 연소/혼합 regime 식별 |
| Fraction softmax | mole fraction의 비음수성과 closure를 구조적으로 보장 | 불가능한 조성 출력 방지 |
| Fraction log loss | 희소한 작은 fraction의 상대적 차이를 확대 | CH4/CO 등 trace 성분 학습 강화 |
| Scaled-log Mass Flow | 넓은 동적 범위와 고유량 outlier 영향 완화 | 저유량과 고유량 동시 학습 |
| Robust normalized PINN | 물리 residual의 단위와 극단값 영향 제한 | supervision과 물리 제약의 안정적 결합 |
| Base + hard sampling | 자연분포를 유지하면서 저성능 영역 노출 보완 | CO/Mass/CH4 학습 기회 증가 |

## 14. 현재 최종 설정 요약

| 범주 | 최종 값 |
|---|---|
| Output | 10D, Vol/Mole Flow 제외 |
| GNN | 5-layer relational bidirectional, hidden 384 |
| Differential encoding | `concat`, 활성 |
| Global pooling | Set2Set 3 steps, 768 -> 384 |
| Edge descriptor | 1600D |
| Shared latent | 384D |
| Target edge weight | 5 |
| Fraction loss | log-SmoothL1, weight 1.0 |
| Mass loss | scaled-log SmoothL1, feature weight 2.0 |
| Active PINN | mass, component, atom |
| Node outer weight | 0.05 |
| Epoch train samples | base 1000 + hard 1000 = 2000 |
| Epoch validation samples | 1000 |
| Early stopping | patience 5, `val_target_edge_property_mean_r2` max |
| Total parameters | 26,704,101 |
