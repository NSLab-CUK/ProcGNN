# MODEL 260805 10D FRAC1 — Equation & Notation Reviewed

> 기준 config: `configs/experiment/pinn/model_260805_10d_frac1.yaml`  
> 구현 기준 최종 점검: 2026-08-24

## 0. 전체 흐름

본 모델은 directed process graph
$$
\mathcal G=(\mathcal V,\mathcal E)
$$
를 입력으로 받아 `edge_is_predictable=true`인 물리적 material stream edge
$e=(u,v)\in\mathcal E_{\mathrm{pred}}\subseteq\mathcal E$의 10개 물성을 예측한다.
Virtual/context-only edge는 message passing에는 참여할 수 있지만 property supervision과
공식 metric에서는 제외된다.

전체 계산 흐름은 다음과 같이 요약할 수 있다.

$$
\boxed{
\text{Node/Edge Input}
\rightarrow
\text{Bidirectional FlowGNN}
\rightarrow
\text{Graph Readout}
\rightarrow
\text{Edge Conditioning}
\rightarrow
\text{Shared Decoder}
\rightarrow
\text{Property Heads}
}
$$

학습 시에는 edge-level supervised loss와 node-level physical consistency loss를 사용한다.
다만 두 항을 하나의 global scalar로 합쳐 한 번만 backward하는 구조는 아니다.

$$
\boxed{
\text{Prediction}
\rightarrow
\begin{cases}
\text{Non-target supervised update},\\
\text{Target-edge supervised updates},\\
\text{Conditional node anchor+PINN update}
\end{cases}
}
$$

현재 최종 출력은

$$
\hat{\mathbf y}_e
=
[
\hat T_e,\hat P_e,
\hat x_{e,1},\ldots,\hat x_{e,7},
\hat{\dot m}_e
]
\in\mathbb R^{10}.
$$

즉 Temperature, Pressure, 7개 mole fraction, Mass Flow를 예측하며
Volumetric Flow, Molar Flow, Density, Enthalpy는 현재 output head에서 제외한다.
현재 configuration의 전체 학습 가능 parameter 수는 **26,704,101개**이고,
이 중 relational differential encoder가 추가하는 parameter는 **3,951,360개**다.

---

## 1. Notation

공정 graph에서 node는 unit operation, directed edge는 material stream을 나타낸다.

$$
\mathcal G=(\mathcal V,\mathcal E),
\qquad
e=(u,v)\in\mathcal E.
$$

여기서 $u$는 source node, $v$는 destination node이다.

| 기호 | 의미 |
|---|---|
| $\mathcal G,\mathcal V,\mathcal E$ | graph, node set, edge set |
| $u,v$ | node index |
| $e=(u,v)$ | directed stream edge |
| $\ell$ | GNN layer index |
| $L$ | 전체 GNN layer 수, 현재 $L=5$ |
| $k$ | chemical species index |
| $\mathbf x$ | 입력 feature |
| $\mathbf h$ | hidden representation |
| $\mathbf m$ | GNN message |
| $\eta$ | attention logit |
| $\alpha$ | normalized attention coefficient |
| $\mathbf z$ | decoder latent |
| $\hat{\mathbf y}$ | model prediction |
| $\dot m$ | mass flow rate |
| $\dot n$ | molar flow rate |
| $\varepsilon$ | normalized physical residual |

주요 hidden dimension은

$$
d_h=384.
$$

Node state와 static edge state는 384D다. Set2Set raw graph state는 768D이며,
edge descriptor에 넣기 직전에 384D로 projection한다.

---

# Part I. Input Encoding

## 2. Node encoding

Node $v$는 semantic role $r_v$, unit type $t_v$, operating value
$\mathbf o_v\in\mathbb R^{18}$, observation mask
$\mathbf b_v^{\mathrm{op}}\in\{0,1\}^{18}$를 가진다.

먼저 operating condition을 encoding한다.

$$
\mathbf h_v^{\mathrm{op}}
=
\operatorname{Enc}_{\mathrm{op}}
\left(
[\mathbf o_v\Vert\mathbf b_v^{\mathrm{op}}]
\right)
\in\mathbb R^{64}.
$$

**설명.**
Operating value와 observation mask를 함께 넣어 실제 값이 0인 경우와 missing value를 구분한다.

Node input은

$$
\mathbf x_v
=
[
\operatorname{Emb}_{\mathrm{role}}(r_v)
\Vert
\operatorname{Emb}_{\mathrm{type}}(t_v)
\Vert
\mathbf h_v^{\mathrm{op}}
]
\in\mathbb R^{120}
$$

로 구성하고,

$$
\boxed{
\mathbf h_v^{(0)}
=
\operatorname{Enc}_{V}(\mathbf x_v)
\in\mathbb R^{384}
}
$$

로 초기 node representation을 만든다.

**설명.**
이 $\mathbf h_v^{(0)}$가 이후 모든 GNN layer의 출발점이며,
later-layer reinjection에도 다시 사용된다.

### 2.1 Feed-ratio feature

CH4, AIR, WATER feed가 존재하는 경우 다음 log-ratio를 operating feature로 사용한다.

$$
\kappa_{A/C}
=
\log
\frac{
F_{\mathrm{AIR}}+\epsilon_f
}{
F_{\mathrm{CH4}}+\epsilon_f
},
$$

$$
\kappa_{W/C}
=
\log
\frac{
F_{\mathrm{WATER}}+\epsilon_f
}{
F_{\mathrm{CH4}}+\epsilon_f
},
\qquad
\epsilon_f=10^{-6}.
$$

**설명.**
절대 feed 양뿐 아니라 AIR-to-CH4, WATER-to-CH4의 상대적인 operating regime를 표현한다.
Scaled CH4/AIR/WATER 값과 유효 mask는 `V_INPUT` node에 주입되고,
두 log-ratio는 `V_INPUT`과 `BURNER` node에 제공된다. Raw process-ID embedding은 사용하지 않는다.

---

## 3. Edge encoding

Stream edge $e=(u,v)$는 stream role $r_e$, stream ID $s_e$,
structural attribute $\boldsymbol\xi_e\in\mathbb R^3$를 가진다.

Structural feature를 먼저 encoding한다.

$$
\mathbf h_e^{\mathrm{str}}
=
\operatorname{Enc}_{\mathrm{str}}
(\boldsymbol\xi_e)
\in\mathbb R^{16}.
$$

Edge input은

$$
\mathbf x_e
=
[
\operatorname{Emb}_{\mathrm{role}}^{E}(r_e)
\Vert
\operatorname{Emb}_{\mathrm{id}}(s_e)
\Vert
\mathbf h_e^{\mathrm{str}}
]
\in\mathbb R^{72},
$$

$$
\boxed{
\mathbf h_e
=
\operatorname{Enc}_{E}(\mathbf x_e)
\in\mathbb R^{384}.
}
$$

**설명.**
$\mathbf h_e$는 layer마다 갱신하지 않는 static edge representation이다.
각 GNN layer의 message와 attention 계산에 반복적으로 주입되어 stream identity가 깊은 layer에서
희석되는 것을 줄인다.

---

# Part II. Relational Bidirectional FlowGNN

## 4. Forward propagation

Forward branch는 실제 물질 흐름 방향 $u\rightarrow v$를 따른다.

Node $v$의 incoming neighbor set을

$$
\mathcal N_{\mathrm{in}}(v)
=
\{u:(u,v)\in\mathcal E\}
$$

로 둔다.

Forward attention 정규화를 위해 physical source $u$의 outgoing neighbor set을

$$
\mathcal N_{\mathrm{out}}(u)
=
\{v:(u,v)\in\mathcal E\}
$$

로 둔다.

### 4.1 Forward message

$$
\boxed{
\mathbf m_{u\to v}^{(\ell,\rightarrow)}
=
\operatorname{MLP}_{\mathrm{msg}}^{(\ell,\rightarrow)}
\left(
[
\mathbf h_u^{(\ell-1)}
\Vert
\mathbf h_e
]
\right)
}
$$

**설명.**
Upstream node state와 해당 stream representation을 결합해 downstream으로 전달할 message를 만든다.
현재 message MLP는 $768\rightarrow512\rightarrow384$다.

### 4.2 Forward attention

Attention logit은 source, destination, edge 정보를 모두 사용한다.

$$
\eta_{u\to v}^{(\ell,\rightarrow)}
=
\operatorname{MLP}_{\mathrm{att}}^{(\ell,\rightarrow)}
\left(
[
\mathbf h_u^{(\ell-1)}
\Vert
\mathbf h_v^{(\ell-1)}
\Vert
\mathbf h_e
]
\right).
$$

같은 sender $u$에서 나가는 edge들 사이에서 softmax를 적용한다.

$$
\boxed{
\beta_{u\to v}^{(\ell,\rightarrow)}
=
\frac{
\exp
\left(
\eta_{u\to v}^{(\ell,\rightarrow)}
\right)
}{
\sum_{w\in\mathcal N_{\mathrm{out}}(u)}
\exp
\left(
\eta_{u\to w}^{(\ell,\rightarrow)}
\right)
}
}
$$

Forward aggregate는

$$
\boxed{
\bar{\mathbf m}_v^{(\ell,\rightarrow)}
=
\sum_{u\in\mathcal N_{\mathrm{in}}(v)}
\beta_{u\to v}^{(\ell,\rightarrow)}
\mathbf m_{u\to v}^{(\ell,\rightarrow)}.
}
$$

**설명.**
각 sender가 자신의 message/flow를 여러 destination에 어떻게 분배할지 정한다. 따라서
$\sum_{v\in\mathcal N_{\mathrm{out}}(u)}\beta_{u\to v}^{(\ell,\rightarrow)}=1$이다.
Message는 계속 $u\rightarrow v$로 전달되며, 바뀌는 것은 message passing 방향이 아니라
softmax normalization 방향뿐이다.
Attention MLP는 concat된 1152D 입력에 대해 $1152\rightarrow256\rightarrow1$을 사용한다.

---

## 5. Backward propagation

Backward branch는 같은 physical edge $e=(u,v)$를 반대 방향 $v\rightarrow u$으로 읽는다.

Physical source $u$의 outgoing neighbor set은

$$
\mathcal N_{\mathrm{out}}(u)
=
\{v:(u,v)\in\mathcal E\}.
$$

### 5.1 Backward message

$$
\boxed{
\mathbf m_{v\to u}^{(\ell,\leftarrow)}
=
\operatorname{MLP}_{\mathrm{msg}}^{(\ell,\leftarrow)}
\left(
[
\mathbf h_v^{(\ell-1)}
\Vert
\mathbf h_e
]
\right)
}
$$

### 5.2 Backward attention

$$
\eta_{v\to u}^{(\ell,\leftarrow)}
=
\operatorname{MLP}_{\mathrm{att}}^{(\ell,\leftarrow)}
\left(
[
\mathbf h_v^{(\ell-1)}
\Vert
\mathbf h_u^{(\ell-1)}
\Vert
\mathbf h_e
]
\right),
$$

$$
\boxed{
\alpha_{v\to u}^{(\ell,\leftarrow)}
=
\frac{
\exp
\left(
\eta_{v\to u}^{(\ell,\leftarrow)}
\right)
}{
\sum_{w\in\mathcal N_{\mathrm{out}}(u)}
\exp
\left(
\eta_{w\to u}^{(\ell,\leftarrow)}
\right)
}
}
$$

$$
\boxed{
\bar{\mathbf m}_u^{(\ell,\leftarrow)}
=
\sum_{v\in\mathcal N_{\mathrm{out}}(u)}
\alpha_{v\to u}^{(\ell,\leftarrow)}
\mathbf m_{v\to u}^{(\ell,\leftarrow)}.
}
$$

**설명.**
Forward branch가 upstream condition을 downstream으로 전달한다면,
backward branch는 downstream constraint와 recycle-related information을 upstream state에 전달한다.
Backward softmax는 reverse graph에서 receiver인 $u$로 들어오는 message들 사이에서 정규화되므로
$\sum_{v\in\mathcal N_{\mathrm{out}}(u)}\alpha_{v\to u}^{(\ell,\leftarrow)}=1$이다.
즉 backward branch는 standard receiver-wise graph attention의 의미를 갖는다. 구현상 두 branch 모두
original source index로 grouping하지만, forward에서는 physical sender-wise 분배이고 backward에서는
reverse-message receiver-wise 선택이라는 점이 다르다.

---

## 6. Differential encoding

방향을 $q\in\{\rightarrow,\leftarrow\}$라 하자.

먼저 directional aggregate와 현재 node state의 차이를 계산한다.

$$
\boxed{
\Delta\mathbf h_v^{(\ell,q)}
=
\bar{\mathbf m}_v^{(\ell,q)}
-
\mathbf h_v^{(\ell-1)}
}
$$

이를 별도의 differential encoder로 변환한다.

$$
\widetilde{\Delta\mathbf h}_v^{(\ell,q)}
=
\operatorname{MLP}_{\mathrm{diff}}^{(\ell,q)}
\left(
\Delta\mathbf h_v^{(\ell,q)}
\right).
$$

Directional update는

$$
\boxed{
\mathbf h_v^{(\ell,q)}
=
\operatorname{MLP}_{\mathrm{upd}}^{(\ell,q)}
\left(
[
\bar{\mathbf m}_v^{(\ell,q)}
\Vert
\widetilde{\Delta\mathbf h}_v^{(\ell,q)}
]
\right).
}
$$

**설명.**
단순히 "이웃으로부터 어떤 정보가 왔는가"뿐 아니라
"그 정보가 현재 node state를 얼마나, 어느 방향으로 바꾸는가"를 함께 학습한다.
현재 각 방향의 differential encoder는 $384\rightarrow512\rightarrow384$이고,
update block은 concat된 768D 입력에 대해 $768\rightarrow512\rightarrow384$를 사용한다.
Forward와 backward는 이 parameter를 공유하지 않는다.

---

## 7. Bidirectional fusion

Forward와 backward representation을 concat한 뒤 하나의 node representation으로 합친다.

$$
\boxed{
\tilde{\mathbf h}_v^{(\ell)}
=
\operatorname{MLP}_{\mathrm{fuse}}^{(\ell)}
\left(
[
\mathbf h_v^{(\ell,\rightarrow)}
\Vert
\mathbf h_v^{(\ell,\leftarrow)}
]
\right).
}
$$

**설명.**
한 layer에서 얻은 upstream evidence와 downstream constraint를 하나의 latent state로 통합한다.
Fusion MLP는 $768\rightarrow512\rightarrow384$다.

---

## 8. Layer residual & initial-state reinjection

먼저 이전 layer state와 새 candidate 사이를 interpolation한다.

$$
\boxed{
\bar{\mathbf h}_v^{(\ell)}
=
(1-\beta)\mathbf h_v^{(\ell-1)}
+
\beta\tilde{\mathbf h}_v^{(\ell)},
\qquad
\beta=0.5.
}
$$

이는 기존 구현의

$$
\mathbf h_v^{(\ell-1)}
+
\beta
\left(
\tilde{\mathbf h}_v^{(\ell)}
-
\mathbf h_v^{(\ell-1)}
\right)
$$

와 동일하다.

그 다음 최초 node representation을 다시 주입한다.

$$
\boxed{
\mathbf h_v^{(\ell)}
=
\bar{\mathbf h}_v^{(\ell)}
+
\gamma\mathbf h_v^{(0)},
\qquad
\gamma=0.05.
}
$$

**설명.**
Residual은 layer-to-layer optimization을 안정화하고,
initial-state reinjection은 role, unit type, operating condition 같은 원래 정보가
깊은 message passing에서 희석되는 것을 줄인다.

이 과정을

$$
\ell=1,\ldots,L,\qquad L=5
$$

까지 반복한다.

---

# Part III. Graph-level Context

## 9. Set2Set graph readout

마지막 node representation 집합

$$
\{\mathbf h_v^{(L)}\}_{v\in\mathcal V_g}
$$

으로부터 graph representation을 만든다.

Set2Set recurrent step을 $s=1,\ldots,S$라 하고 현재 $S=3$이다.

$$
\mathbf q_s
=
\operatorname{LSTM}
\left(
\mathbf q_{s-1}^{*}
\right).
$$

각 node의 readout weight는

$$
\pi_{v,s}
=
\frac{
\exp
\left(
\mathbf h_v^{(L)\top}\mathbf q_s
\right)
}{
\sum_{u\in\mathcal V_g}
\exp
\left(
\mathbf h_u^{(L)\top}\mathbf q_s
\right)
}.
$$

Readout context는

$$
\mathbf c_s
=
\sum_{v\in\mathcal V_g}
\pi_{v,s}
\mathbf h_v^{(L)}.
$$

Query와 context를 결합한다.

$$
\boxed{
\mathbf q_s^{*}
=
[
\mathbf q_s
\Vert
\mathbf c_s
]
\in\mathbb R^{768}.
}
$$

마지막 recurrent state는 graph encoder에서 raw graph representation으로 유지된다.

$$
\boxed{
\mathbf g_{\mathcal G}^{\mathrm{raw}}
=
\mathbf q_S^{*}
\in\mathbb R^{768}.
}
$$

Edge decoder의 hierarchical global projection이 descriptor를 만들기 직전에 이를 384D로 변환한다.

$$
\boxed{
\mathbf h_{\mathcal G}
=
\operatorname{Proj}_{\mathcal G}
(\mathbf g_{\mathcal G}^{\mathrm{raw}})
\in\mathbb R^{384}.
}
$$

**설명.**
단순 mean pooling과 달리 learned query가 graph 안에서 현재 global state를 구성하는 데 중요한 node를
가중 선택한다.

---

# Part IV. Edge-specific Conditioning

## 10. Heat-exchanger pair correction

Heat exchanger에서 paired stream relation을 $p(e)$라 하고,
hot/cold side embedding을

$$
\mathbf c_e^{\mathrm{HX}}
\in\mathbb R^{16}
$$

로 둔다.

Paired edge 정보를 이용한 correction은

$$
\Delta\mathbf h_e^{\mathrm{HX}}
=
\operatorname{MLP}_{\mathrm{HX}}
\left(
[
\mathbf h_e
\Vert
\mathbf h_{p(e)}
\Vert
\mathbf c_e^{\mathrm{HX}}
]
\right).
$$

Scalar gate는

$$
g_{\mathrm{HX}}
=
\sigma(\omega_{\mathrm{HX}}),
\qquad
g_{\mathrm{HX}}^{(0)}=0.05.
$$

최종 corrected edge state는

$$
\boxed{
\bar{\mathbf h}_e
=
\operatorname{LN}
\left(
\mathbf h_e
+
g_{\mathrm{HX}}
\Delta\mathbf h_e^{\mathrm{HX}}
\right).
}
$$

**설명.**
일반 graph incidence만으로는 구분하기 어려운 HX inlet/outlet pairing 정보를
edge representation에 직접 주입한다. Gate는 학습 초기에 correction이 과도하게 작동하는 것을 막는다.
검증된 paired HX edge에만 이 correction을 적용하며, HX pair가 없는 edge는 원래 edge state를 그대로 사용한다.

---

## 11. Direct feed context

Scaled CH4/AIR/WATER feed vector를

$$
\tilde{\mathbf f}\in\mathbb R^3
$$

라 하고 observation mask를

$$
\mathbf b_f\in\{0,1\}^3
$$

라 한다.

$$
\boxed{
\mathbf h_{\mathrm{feed}}
=
\operatorname{MLP}_{\mathrm{feed}}
\left(
[
\tilde{\mathbf f}
\Vert
\mathbf b_f
]
\right)
\in\mathbb R^{64}.
}
$$

**설명.**
Feed information이 여러 GNN layer를 지나며 희석되는 것을 방지하기 위해
prediction head 직전에도 직접 condition으로 제공한다.

---

## 12. Edge descriptor

Prediction edge $e=(u,v)$마다 다음 정보를 결합한다.

$$
\boxed{
\mathbf d_e
=
[
\mathbf h_u^{(L)}
\Vert
\mathbf h_v^{(L)}
\Vert
\mathbf h_{\mathcal G}
\Vert
\bar{\mathbf h}_e
\Vert
\mathbf h_{\mathrm{feed}}
]
\in\mathbb R^{1600}.
}
$$

차원은

$$
1600
=
384_{\mathrm{src}}
+
384_{\mathrm{dst}}
+
384_{\mathrm{global}}
+
384_{\mathrm{edge}}
+
64_{\mathrm{feed}}.
$$

**설명.**
하나의 edge를 예측할 때 source/destination의 local state,
graph 전체 context, stream identity, direct operating condition을 동시에 사용한다.

---

# Part V. Prediction Head

## 13. Shared decoder

모든 process와 모든 edge가 동일한 decoder를 공유한다.

$$
\mathbf h_e^{\mathrm{dec}}
=
\operatorname{Drop}_{0.1}
\left[
\operatorname{GELU}
\left(
\operatorname{LN}
(W_1\mathbf d_e+\mathbf b_1)
\right)
\right]
\in\mathbb R^{768},
$$

$$
\boxed{
\mathbf z_e
=
\operatorname{LN}
\left[
\operatorname{Drop}_{0.1}
\left(
\operatorname{GELU}
\left(
\operatorname{LN}
(W_2\mathbf h_e^{\mathrm{dec}}+\mathbf b_2)
\right)
\right)
\right]
\in\mathbb R^{384}.
}
$$

**설명.**
$\mathbf z_e$는 모든 property branch가 공유하는 edge-level latent representation이다.

---

## 14. Property-specific heads

### 14.1 Temperature and Pressure

$$
\boxed{
[\hat T_e,\hat P_e]
=
\operatorname{MLP}_{\mathrm{cond}}
(\mathbf z_e).
}
$$

현재 구조는

$$
384\rightarrow96\rightarrow2.
$$

---

### 14.2 Mole fractions

Fraction logit을

$$
\boldsymbol\ell_e
=
\operatorname{MLP}_{\mathrm{frac}}
(\mathbf z_e)
\in\mathbb R^7
$$

라 한다.

Temperature-softmax는

$$
\boxed{
\hat x_{e,k}
=
\frac{
\exp(\ell_{e,k}/\tau_x)
}{
\sum_{j=1}^{7}
\exp(\ell_{e,j}/\tau_x)
},
\qquad
\tau_x=0.5.
}
$$

따라서

$$
\hat x_{e,k}\ge0,
\qquad
\sum_{k=1}^{7}\hat x_{e,k}=1.
$$

**설명.**
Mole fraction의 non-negativity와 closure constraint를 architecture 자체로 보장한다.

---

### 14.3 Mass Flow

Mass Flow는 직접 physical scale을 regression하지 않고 scaled-log coordinate에서 예측한다.

Transformation은

$$
\boxed{
\tilde m_e
=
\mathcal T_m(\dot m_e)
=
2\log
\left(
\max(\dot m_e,0)+10^{-8}
\right).
}
$$

Mass head는

$$
\boxed{
\hat{\tilde m}_e
=
\operatorname{MLP}_{\mathrm{mass}}
(\mathbf z_e).
}
$$

Inverse transform은

$$
\boxed{
\hat{\dot m}_e
=
\max
\left\{
\exp
\left(
\frac{\hat{\tilde m}_e}{2}
\right)
-
10^{-8},
0
\right\}.
}
$$

**설명.**
Mass Flow의 dynamic range가 크기 때문에 log coordinate에서 학습하여
저유량과 고유량을 동시에 다루기 쉽게 한다.

---

## 15. Final prediction

최종 edge prediction은

$$
\boxed{
\hat{\mathbf y}_e
=
[
\hat T_e,
\hat P_e,
\hat x_{e,\mathrm{H_2O}},
\hat x_{e,\mathrm{H_2}},
\hat x_{e,\mathrm{CH_4}},
\hat x_{e,\mathrm{CO_2}},
\hat x_{e,\mathrm{CO}},
\hat x_{e,\mathrm{O_2}},
\hat x_{e,\mathrm{N_2}},
\hat{\dot m}_e
]
\in\mathbb R^{10}.
}
$$

Property head와 loss/metric은 $e\in\mathcal E_{\mathrm{pred}}$에만 적용한다.
Batch tensor의 edge 축을 맞추기 위해 context-only edge 위치에는 0을 scatter하지만,
그 0은 물성 예측값이나 평가 표본으로 사용하지 않는다.

---

# Part VI. Supervised Learning Objective

## 16. SmoothL1

기본 robust regression loss는

$$
\rho_{\mathrm{SL1}}(r)
=
\begin{cases}
\frac12r^2,
& |r|<1,\\[3pt]
|r|-\frac12,
& |r|\ge1.
\end{cases}
$$

**설명.**
작은 error에는 quadratic penalty를,
큰 error에는 linear penalty를 적용해 outlier의 영향을 제한한다.

---

## 17. Condition + Mass main loss

Training coordinate의 target과 prediction을 각각

$$
y_{e,p}^{\mathrm{tr}},
\qquad
\hat y_{e,p}^{\mathrm{tr}}
$$

라 하고 valid mask를 $M_{e,p}$라 한다.

현재 property weight는

$$
w_T=w_P=1,
\qquad
w_m=2.
$$

Main loss는

$$
\boxed{
\mathcal L_e^{\mathrm{main}}
=
\frac{
\sum_{p\in\{T,P,m\}}
M_{e,p}w_p\,
\rho_{\mathrm{SL1}}
\left(
\hat y_{e,p}^{\mathrm{tr}}
-
y_{e,p}^{\mathrm{tr}}
\right)
}{
\max
\left(
1,
\sum_{p\in\{T,P,m\}}
M_{e,p}
\right)
}.
}
$$

**설명.**
Temperature, Pressure, transformed Mass Flow를 함께 학습하며,
Mass Flow는 상대적으로 더 큰 weight $2$를 사용한다.

---

## 18. Fraction log loss

유효 fraction species set을

$$
\mathcal I_e
=
\left\{
k:
M_{e,k}=1,\;
\dot m_e^{\mathrm{true}}>10^{-8},\;
\sum_{r=1}^{7}x_{e,r}>10^{-6}
\right\}
$$

로 정의한다.

Fraction loss는

$$
\boxed{
\mathcal L_e^{\mathrm{frac}}
=
\frac{1}{|\mathcal I_e|}
\sum_{k\in\mathcal I_e}
\rho_{\mathrm{SL1}}
\left[
\log\!\left(\max(\hat x_{e,k},10^{-6})\right)
-
\log\!\left(\max(x_{e,k},10^{-6})\right)
\right].
}
$$

**설명.**
작은 fraction의 절대 오차보다 상대적인 차이를 더 민감하게 학습하기 위해 log space에서 비교한다.
여기서 $10^{-6}$은 값에 더하는 smoothing constant가 아니라 log 입력의 lower clamp다.
True Mass Flow가 거의 0이거나 true fraction 합이 유효하지 않은 edge는 fraction supervision에서 제외한다.

---

## 19. Edge supervised loss

$$
\boxed{
\mathcal L_e^{\mathrm{sup}}
=
\mathcal L_e^{\mathrm{main}}
+
\lambda_{\mathrm{frac}}
\mathcal L_e^{\mathrm{frac}},
\qquad
\lambda_{\mathrm{frac}}=1.
}
$$

Target edge scaling은

$$
w_e^{\mathrm{edge}}
=
\begin{cases}
5,
& e\in\mathcal E_{\mathrm{target}},\\
1,
& e\notin\mathcal E_{\mathrm{target}}.
\end{cases}
$$

각 edge step에서 실제 scaling된 supervised objective는

$$
\widetilde{\mathcal L}_e^{\mathrm{sup}}
=
w_e^{\mathrm{edge}}\mathcal L_e^{\mathrm{sup}}
$$

이다. Node step의 supervised anchor에는 이 target weight를 다시 적용하지 않는다.

**설명.**
Target edge에 더 강한 optimization signal을 주지만,
현재 구현은 단순 weighted global mean이 아니라 target/non-target optimizer step을 분리한다.

---

# Part VII. Physics-Informed Objective

## 20. Internal nodes

Physics loss를 적용할 internal node set은

$$
\boxed{
\mathcal V_{\mathrm{int}}
=
\left\{
v:
\deg^{-}(v)>0,\;
\deg^{+}(v)>0,\;
v\notin\mathcal V_{\mathrm{boundary}}
\right\}.
}
$$

**설명.**
Feed/source나 final product/sink 같은 boundary node가 아니라
실제로 incoming/outgoing stream이 모두 존재하는 unit에 conservation constraint를 적용한다.
또한 `edge_pinn_mask`로 virtual/context-only edge를 balance incidence에서 제거하고,
공정 metadata에 지정된 node/edge exclusion mask도 함께 적용한다.

---

## 21. Generic normalized balance

Edge quantity $\mathbf q_e$에 대해 predicted balance를

$$
\mathbf B_v(\hat{\mathbf q})
=
\sum_{e\in\delta^{-}(v)}
\hat{\mathbf q}_e
-
\sum_{e\in\delta^{+}(v)}
\hat{\mathbf q}_e
$$

로 둔다.

True-flow magnitude scale은

$$
\mathbf S_v(\mathbf q)
=
\sum_{e\in\delta^{-}(v)}
|\mathbf q_e|
+
\sum_{e\in\delta^{+}(v)}
|\mathbf q_e|.
$$

Normalized residual은

$$
\boxed{
\boldsymbol\varepsilon_v(\mathbf q)
=
\frac{
\mathbf B_v(\hat{\mathbf q})
}{
\operatorname{stopgrad}
\left[
\max(\mathbf S_v(\mathbf q),1)
+
10^{-6}
\right]
}.
}
$$

**설명.**
절대 residual 대신 true flow magnitude로 정규화해
작은 공정과 큰 공정의 balance error scale을 맞춘다.
벡터 quantity에서는 $\max(\cdot,1)$을 component-wise로 적용하며,
분모에는 gradient가 흐르지 않는다.

Residual은

$$
[-10,10]
$$

으로 clipping한 뒤 Huber loss를 적용한다.

$$
\rho_{\delta}(r)
=
\begin{cases}
\frac12r^2,
& |r|\le\delta,\\[3pt]
\delta
\left(
|r|-\frac12\delta
\right),
& |r|>\delta,
\end{cases}
\qquad
\delta=0.5.
$$

---

## 22. Mass balance

Physical mass-flow prediction $\hat{\dot m}_e$를 이용한다.

$$
\boxed{
\varepsilon_v^{\mathrm{mass}}
=
\frac{
\sum_{e\in\delta^{-}(v)}
\hat{\dot m}_e
-
\sum_{e\in\delta^{+}(v)}
\hat{\dot m}_e
}{
\operatorname{stopgrad}
\left[
\max
\left(
\sum_{e\in\delta^{-}(v)}
|\dot m_e|
+
\sum_{e\in\delta^{+}(v)}
|\dot m_e|,
1
\right)
+
10^{-6}
\right]
}.
}
$$

Mass conservation loss는

$$
\boxed{
\mathcal L_{\mathrm{mass}}
=
\operatorname{mean}_{v\in\mathcal V_{\mathrm{int}}}
\rho_{0.5}
\left(
\operatorname{clip}
(
\varepsilon_v^{\mathrm{mass}},
-10,
10
)
\right).
}
$$

**설명.**
각 internal unit에서 total incoming mass flow와 outgoing mass flow가 일치하도록 유도한다.

---

## 23. Component balance

Species $k$의 molecular weight를 $\mu_k$라 한다.

Predicted mixture molecular weight는

$$
\hat \mu_e^{\mathrm{mix}}
=
\sum_{k=1}^{7}
\hat x_{e,k}\mu_k.
$$

Total molar flow는

$$
\boxed{
\hat{\dot n}_e^{\mathrm{tot}}
=
\frac{
\hat{\dot m}_e
}{
\max
(
\hat \mu_e^{\mathrm{mix}},
10^{-6}
)
}.
}
$$

Species $k$의 component molar flow는

$$
\boxed{
\hat{\dot n}_{e,k}
=
\hat{\dot n}_e^{\mathrm{tot}}
\hat x_{e,k}.
}
$$

Non-reactive node에서 component balance는

$$
\boxed{
B_{v,k}^{\mathrm{comp}}
=
\sum_{e\in\delta^{-}(v)}
\hat{\dot n}_{e,k}
-
\sum_{e\in\delta^{+}(v)}
\hat{\dot n}_{e,k}.
}
$$

이 residual에 Section 21과 동일한 normalization, clipping, Huber reduction을 적용하여

$$
\mathcal L_{\mathrm{component}}
$$

를 얻는다.

각 component의 true-flow scale이 $10^{-6}$ 이하인 항은 수치적으로 유효한 balance 표본으로
보지 않고 reduction에서 제외한다.

**설명.**
비반응 장치에서는 각 species 자체의 molar flow가 보존되어야 한다는 제약이다.

---

## 24. Atom balance

Species-atom incidence matrix를

$$
A\in\mathbb R^{7\times n_a}
$$

라 한다.

Edge별 atom molar flow는

$$
\boxed{
\hat{\dot{\mathbf a}}_e
=
\hat{\dot{\mathbf n}}_eA.
}
$$

Reactive node에서 atom balance는

$$
\boxed{
\mathbf B_v^{\mathrm{atom}}
=
\sum_{e\in\delta^{-}(v)}
\hat{\dot{\mathbf a}}_e
-
\sum_{e\in\delta^{+}(v)}
\hat{\dot{\mathbf a}}_e.
}
$$

동일한 normalized robust reduction을 적용하여

$$
\mathcal L_{\mathrm{atom}}
$$

을 얻는다.

Atom 항도 true atom-flow scale이 $10^{-6}$보다 큰 항만 reduction에 포함한다.

**설명.**
반응 장치에서는 chemical species는 바뀔 수 있지만 원소 자체는 보존되어야 하므로
component balance 대신 atom balance를 적용한다.

---

## 25. Total PINN loss

현재 active physics term은 mass, component, atom balance이다.

$$
\boxed{
\mathcal L_{\mathrm{PINN}}
=
\lambda_{\mathrm{mass}}
\mathcal L_{\mathrm{mass}}
+
\lambda_{\mathrm{comp}}
\mathcal L_{\mathrm{component}}
+
\lambda_{\mathrm{atom}}
\mathcal L_{\mathrm{atom}}
}
$$

with

$$
\lambda_{\mathrm{mass}}=1,
\qquad
\lambda_{\mathrm{comp}}=1.5\times10^{-7},
\qquad
\lambda_{\mathrm{atom}}=0.2.
$$

**설명.**
서로 단위와 numerical scale이 다른 physics residual을 별도 coefficient로 조절한다.

---

## 26. PINN warm-up schedule

Epoch $t$에서 physics multiplier는

$$
\boxed{
\lambda_{\mathrm{phy}}(t)
=
\begin{cases}
0,
&1\le t\le5,\\
0.5,
&6\le t\le7,\\
1,
&t\ge8.
\end{cases}
}
$$

Node objective의 값은

$$
\boxed{
\mathcal L_{\mathrm{node}}
=
\mathcal L_{\mathrm{anchor}}
+
\lambda_{\mathrm{node}}
\lambda_{\mathrm{phy}}(t)
\mathcal L_{\mathrm{PINN}},
\qquad
\lambda_{\mathrm{node}}=0.05.
}
$$

**설명.**
이 식은 node update가 실제로 실행될 때의 objective다. 현재 구현은
$\lambda_{\mathrm{phy}}(t)=0$이거나 유효한 PINN 항이 없으면 node optimizer step 자체를 건너뛴다.
따라서 epoch 1--5에는 anchor-only node step도 실행되지 않고 edge supervised update만 수행된다.
Epoch 6--7에는 half-strength physics를, epoch 8 이후에는 full-strength physics를 사용한다.
유효한 supervised anchor와 남은 optimizer-step budget도 node update 실행 조건에 포함된다.

---

# Part VIII. Optimization Flow

## 27. Optimizer-step structure

현재 batch size는 graph 1개다. 그러나 graph sample 하나가 optimizer step 하나와 같은 것은 아니며,
optimization은 단일 global loss를 한 번 backward하는 구조가 아니라 다음 update들로 분리된다.

$$
\boxed{
\text{Non-target supervised step}
\rightarrow
\text{Target-edge supervised steps}
\rightarrow
\text{Conditional node supervised+PINN step}
}
$$

구체적으로는

1. 유효한 non-target edge group이 있으면 macro-mean supervised update,
2. 각 유효 target edge마다 별도의 supervised update,
3. warm-up multiplier가 양수이고 유효 PINN/anchor 및 budget이 있을 때만 node update

순으로 수행된다. 따라서 처음 5 epoch에는 세 번째 update가 없으며, graph마다 실제 optimizer-step
개수는 target edge 수와 유효 mask에 따라 달라질 수 있다.

**설명.**
Target edge의 direct learning signal을 보장하면서
non-target representation과 node-level physics consistency도 별도로 유지한다.

Optimizer는 AdamW를 사용하며 learning rate와 weight decay는 각각

$$
\mathrm{lr}=10^{-4},
\qquad
\mathrm{wd}=10^{-5}
$$

이다. Global gradient norm은

$$
\|\nabla_\theta\mathcal L\|_2\le0.5
$$

가 되도록 clipping한다.

---

# Part IX. Evaluation

## 28. Coefficient of determination

공식 target-property metric은 단일 edge별 $R^2$를 먼저 계산해 평균하지 않는다.
Property $p$에 대해 모든 official target edge의 유효 표본을 먼저 합친 pooled set을

$$
\mathcal D_p^{\mathrm{target}}
=
\biguplus_{e\in\mathcal E_{\mathrm{target}}}
\mathcal D_{e,p}
$$

로 정의한다. 이 pooled set에서 raw $R^2$는

$$
\boxed{
R_{p,\mathrm{raw}}^2
=
1-
\frac{
\sum_{i\in\mathcal D_p^{\mathrm{target}}}
(y_{i,p}-\hat y_{i,p})^2
}{
\sum_{i\in\mathcal D_p^{\mathrm{target}}}
(y_{i,p}-\bar y_p)^2
},
}
$$

$$
\bar y_p
=
\frac{1}{|\mathcal D_p^{\mathrm{target}}|}
\sum_{i\in\mathcal D_p^{\mathrm{target}}}
y_{i,p}.
$$

현재 Proposed metric의 constant/insufficient-sample 처리는 다음과 같다.

$$
R_p^2
=
\begin{cases}
\mathrm{NaN},
& |\mathcal D_p^{\mathrm{target}}|<2,\\[3pt]
0.999,
& |\mathcal D_p^{\mathrm{target}}|\ge2
\quad\text{and}\quad
\sum_i(y_{i,p}-\bar y_p)^2\le10^{-12},\\[3pt]
R_{p,\mathrm{raw}}^2,
& \text{otherwise}.
\end{cases}
$$

공식 10개 property 순서는

$$
\mathcal P_{10}
=
\{
T,P,
x_{\mathrm{H_2O}},x_{\mathrm{H_2}},x_{\mathrm{CH_4}},
x_{\mathrm{CO_2}},x_{\mathrm{CO}},x_{\mathrm{O_2}},x_{\mathrm{N_2}},
\dot m
\}
$$

이며, finite property만 평균한다.

$$
\boxed{
\texttt{target\_edge\_property\_mean\_r2}
=
\frac{1}{|\mathcal P_{\mathrm{finite}}|}
\sum_{p\in\mathcal P_{\mathrm{finite}}}R_p^2.
}
$$

Validation checkpoint와 early stopping은 정확히
`val_target_edge_property_mean_r2`를 maximize한다. Patience는 5이고 max epoch는 30이다.

Fraction metric은

$$
\dot m_e^{\mathrm{true}}>10^{-8}
$$

인 edge만 사용한다.

**해석 및 namespace 주의.**
$R^2=1$이면 완전 예측, $R^2=0$이면 target mean을 항상 예측하는 것과 같은 수준이며
일반 raw $R^2$는 음수도 가능하다. 표본이 2개 미만인 property의 NaN은 평균에서 제외된다.
`target_edge_property_mean_r2`는 위의 property-wise pooled 공식 지표다.
반면 `target_mean_r2`는 target-row/species resolution을 따르는 별도 strict metric family이므로
두 이름을 같은 값으로 해석하면 안 된다.

---

# Part X. Current Training Protocol

## 29. Epoch sampling, validation, and checkpointing

기본 all-process/Proposed 학습의 현재 runtime 기준은 다음과 같다.

| 항목 | 현재 값 |
|---|---:|
| Maximum epochs | 30 |
| Graph batch size | 1 |
| Mixed precision | disabled |
| Scheduler | cosine, minimum LR $10^{-6}$ |
| Base samples per epoch | 1,000 |
| Hard-fill samples per epoch | 1,000 |
| Total graph samples per epoch | 2,000 |
| Validation samples per epoch | 1,000 |
| Early-stopping patience | 5 |
| Monitor | `val_target_edge_property_mean_r2` (max) |

Hard-fill 1,000개는 다음 quota를 사용한다.

| Target bucket | Samples/epoch |
|---|---:|
| Target Frac_CO | 400 |
| Target Mass_Flow | 400 |
| └ high-flow | 240 |
| └ extreme-flow | 160 |
| Target Frac_CH4 | 200 |

Base sampler와 hard-fill은 가능한 범위에서 epoch 내부 중복을 피하고 process balance를 유지한다.
Validation은 매 epoch 기존 validation pool에서 seed 기반 1,000개를 sampling하며,
validation PINN loss는 계산하지 않는다. `best.pt` 선택과 early stopping은 위 공식 monitor를 사용하고,
`last.pt`는 마지막 실행 상태를 보존한다.

Transfer data-efficiency처럼 runner가 계산량 통제를 위해 early stopping을 끄고 정확한 optimizer-step
cap을 주는 실험은 이 기본 protocol의 명시적 runtime override다. 이 경우에도 validation 기반
best-checkpoint 선택 자체는 유지된다.

---

# 30. End-to-end equation flow

전체 모델을 가장 압축해서 쓰면 다음과 같다.

### Step 1. Encode nodes and edges

$$
\mathbf h_v^{(0)}
=
\operatorname{Enc}_{V}(\mathbf x_v),
\qquad
\mathbf h_e
=
\operatorname{Enc}_{E}(\mathbf x_e).
$$

### Step 2. Bidirectional message passing

$$
(\mathbf h^{(\ell-1)},\mathbf h_e)
\rightarrow
\bar{\mathbf m}^{(\ell,\rightarrow)},
\bar{\mathbf m}^{(\ell,\leftarrow)}
\rightarrow
\Delta\mathbf h^{(\ell,\rightarrow)},
\Delta\mathbf h^{(\ell,\leftarrow)}
$$

$$
\rightarrow
\tilde{\mathbf h}^{(\ell)}
\rightarrow
\mathbf h^{(\ell)}.
$$

이를 $L=5$번 반복한다.

### Step 3. Obtain graph context

$$
\{\mathbf h_v^{(L)}\}_{v\in\mathcal V}
\rightarrow
\operatorname{Set2Set}
\rightarrow
\mathbf g_{\mathcal G}^{\mathrm{raw}}\in\mathbb R^{768}
\rightarrow
\operatorname{Proj}_{\mathcal G}
\rightarrow
\mathbf h_{\mathcal G}\in\mathbb R^{384}.
$$

### Step 4. Build edge-specific descriptor

$$
\mathbf d_e
=
[
\mathbf h_u^{(L)}
\Vert
\mathbf h_v^{(L)}
\Vert
\mathbf h_{\mathcal G}
\Vert
\bar{\mathbf h}_e
\Vert
\mathbf h_{\mathrm{feed}}
].
$$

### Step 5. Decode shared edge latent

$$
\mathbf d_e
\rightarrow
\mathbf z_e.
$$

### Step 6. Predict physical properties

$$
\mathbf z_e
\rightarrow
\{
\hat T_e,
\hat P_e,
\hat{\mathbf x}_e,
\hat{\dot m}_e
\}.
$$

### Step 7. Train with data supervision

$$
\hat{\mathbf y}_e
\rightarrow
\begin{cases}
\text{non-target macro update},\\
\text{per-target-edge updates}.
\end{cases}
$$

### Step 8. Enforce process physics

$$
(\hat{\dot m}_e,\hat{\mathbf x}_e)
\rightarrow
\{
\mathcal L_{\mathrm{mass}},
\mathcal L_{\mathrm{component}},
\mathcal L_{\mathrm{atom}}
\}
\rightarrow
\mathcal L_{\mathrm{PINN}}.
$$

이 physics objective는 warm-up이 끝난 뒤 유효 PINN 항과 anchor가 있을 때만 별도 node update로
실행된다. 따라서 Step 7과 Step 8은 하나의 global loss backward를 뜻하지 않는다.

### Step 9. Validate and select checkpoint

$$
\{R_p^2:p\in\mathcal P_{10}\}
\rightarrow
\texttt{val\_target\_edge\_property\_mean\_r2}
\rightarrow
\texttt{best.pt}.
$$

따라서 모델의 핵심 구조는

$$
\boxed{
\text{Local graph propagation}
+
\text{Global graph context}
+
\text{Edge-specific conditioning}
+
\text{Physical consistency}
}
$$

로 정리할 수 있다.

---

# 31. 최종 notation 검토 결과

이번 정리에서는 수식 간 의미 충돌을 줄이기 위해 다음 원칙으로 통일하였다.

- Node index는 $u,v$로 고정한다.
- Edge는 $e=(u,v)$로 고정한다.
- Node/edge hidden representation은 모두 $\mathbf h$ 계열을 사용한다.
- GNN message는 $\mathbf m$으로만 사용한다.
- Physical mass flow는 $\dot m$, molar flow는 $\dot n$으로 분리한다.
- Attention logit은 $\eta$, normalized attention은 $\alpha$로 구분한다.
- Differential quantity는 $\Delta\mathbf h$로 표시한다.
- Physics normalized residual은 $\varepsilon$로 표시하여 node/edge role $r_v,r_e$와 충돌하지 않게 한다.
- Set2Set recurrent index는 $s$, training epoch는 $t$로 분리한다.
- Shared decoder latent는 $\mathbf z_e$로만 사용한다.
- Loss coefficient는 $\lambda_{\cdot}$ 계열로 통일한다.
- Species molecular weight는 $\mu_k$로 두어 valid-mask $M_{e,p}$와 구분한다.
- Feed log-ratio는 $\kappa$로 두어 robust loss 함수 $\rho(\cdot)$와 구분한다.
- Raw Set2Set output은 $\mathbf g_{\mathcal G}^{\mathrm{raw}}\in\mathbb R^{768}$,
  projected graph context는 $\mathbf h_{\mathcal G}\in\mathbb R^{384}$로 구분한다.
- 공식 target-property aggregate는 `target_edge_property_mean_r2`로 고정하고,
  별도 strict `target_mean_r2` namespace와 혼용하지 않는다.

이 notation을 기준으로 Method section과 Appendix의 수식을 동일하게 유지하는 것이 가장 안전하다.
