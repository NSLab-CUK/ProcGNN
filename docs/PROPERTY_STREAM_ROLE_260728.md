# Property Stream Role (Role16) Default (260728)

## Current default status

As of 2026-07-28, Role16 is no longer an ablation-only option. It is enabled
by default in `configs/model/process_encoder.yaml`, and is also explicit in the
current E0/E1/E2 training YAML files.

```yaml
property_stream_role:
  enabled: true
  embedding_dim: 16
  unknown_role_id: 0
```

All `model_260716*.yaml` configs that inherit `process_encoder.yaml` now resolve
to this setting. The old `*_streamrole16.yaml` file is retained only as a
historical ablation record.

The decoder input is therefore 2067D by default:

```text
base edge feature 2051D + property stream role embedding 16D = 2067D
```

This changes the first linear-layer shape of the condition, fraction, and flow
branches. Old Role16-disabled checkpoints cannot be loaded strictly into the
new default architecture without an explicit migration or retraining.

## A. 조사 결과

### 현재 edge prediction 경로

1. `ProcessEdgeInputEncoder`는 기존 generic `edge_stream_role` 64D,
   hashed stream ID 96D, 구조 edge feature 3D를 합쳐 512D edge embedding을
   만든다. 이 값은 directed attention/message passing에만 사용된다.
2. 5개 GNN layer 뒤 `local_node_embeddings`는 512D다.
3. Set2Set은 graph마다 1024D global embedding을 만든다.
4. `ProcessSurrogateModel.predict()`는 local/global embedding과
   `edge_struct_attr`만 `EdgeDecoder`로 전달하던 구조였다.
5. `EdgeDecoder.forward()`의 기존 입력은
   `[h_src(512), h_dst(512), g(1024), edge_struct(3)] = 2051D`다.
6. 이 2051D tensor 하나가 PI head의 Condition, Fraction, Flow branch에
   공통 입력된다.
7. Density와 Enthalpy는 edge feature를 직접 받지 않고
   `[Temp, Pres, fraction(7)] = 9D` thermo input만 받는다.

주요 위치:

- GNN edge embedding: `src/process_graph/models/process_encoder.py:394`
- local/Set2Set output: `src/process_graph/models/process_encoder.py:999`
- decoder 호출: `src/process_graph/models/process_surrogate.py:150`
- 2051D concat: `src/process_graph/models/edge_decoder.py:577`
- PI branch/thermo heads: `src/process_graph/models/edge_decoder.py:257`

### 기존 role의 한계

기존 `STREAM_ROLE_VOCAB`은 10개이며 GNN용 embedding module은
`ProcessEdgeInputEncoder.role_embedding`이다. 하지만 P04/P06/P07/P08의
평행 출력 edge는 모두 generic role `output`, ID 6이다. 따라서 이 ID를
그대로 Property Head에 추가해도 평행 edge를 구분할 수 없다.

| Process | Edge A | Edge B | Same src/dst | Generic IDs | Property IDs |
| --- | --- | --- | --- | --- | --- |
| P04 | P04_E019 | P04_E020 | yes | 6 / 6 | 10 / 16 |
| P06 | P06_E022 | P06_E023 | yes | 6 / 6 | 10 / 12 |
| P07 | P07_E016 | P07_E017 | yes | 6 / 6 | 10 / 11 |
| P08 | P08_E015 | P08_E016 | yes | 6 / 6 | 10 / 12 |

별도 property vocabulary는 공정 ID나 canonical edge ID가 아니라
flowsheet에 이미 존재하는 output port metadata만 사용한다:

- `OUT_PROD -> output_product`
- `OUT_H2O -> output_h2o`
- `OUT_CO2 -> output_co2`
- `OUT_exhaust -> output_exhaust`
- `OUT_RESTEAM -> output_resteam`
- `OUT_VENT -> output_vent`
- `OUT_Z1/OUT_Z2 -> zero_or_side_output`
- `OUT_stream -> output_other`

정답 fraction, flow, density, enthalpy 값은 role 생성에 사용하지 않는다.

### 정확한 원인

평행 쌍은 동일 src/dst node, 동일 graph global context, 동일 구조 flag를
가져 기존 2051D 입력이 정확히 같았다. Shared deterministic PI head는 같은
입력에 같은 출력을 내므로, 이전 체크포인트의 평행 edge 예측이 모든 성분에서
완전히 같았던 것은 학습 실패가 아니라 표현상 식별 불가능성이다.

## B. 수정 파일

| 파일 | 수정 내용 |
| --- | --- |
| `src/process_graph/constants.py` | property semantic role vocabulary와 metadata mapping |
| `src/process_graph/schema.py` | graph sample의 property role field |
| `src/process_graph/data/tabular_dataset.py` | v3/legacy 생성, copy, collate 경로 전달 |
| `src/process_graph/experiment/schema.py` | nested config key |
| `src/process_graph/experiment/config_builders.py` | config 검증 및 encoder config 전달 |
| `src/process_graph/models/process_encoder.py` | disabled-by-default runtime config |
| `src/process_graph/models/process_surrogate.py` | batch role tensor를 decoder로 전달 |
| `src/process_graph/models/edge_decoder.py` | 별도 embedding과 PI head 직전 concat |
| `tests/test_property_stream_role.py` | pair/shape/gradient/shuffle/compatibility tests |

## C. 최종 구조

```text
GNN local node embeddings                       512D
Set2Set global embedding                       1024D

base z_e = [src 512, dst 512, global 1024, struct 3] = 2051D
role r_e = PropertyStreamRoleEmbedding(role_id)       =   16D
z'_e = [z_e, r_e]                                    = 2067D

Condition: 2067 -> 128 -> 2
Fraction:  2067 -> 128 -> 7 -> softmax(T=0.5)
Flow:      2067 -> 128 -> 3

thermo = [Temp, Pres, fraction(7)] = 9D
Density:   9 -> 128 -> 1
Enthalpy:  9 -> 128 -> 1
```

Target Branch Hidden Adapter는 계속 비활성화다. Role은 target 여부와
무관하게 모든 edge에 적용되며 target weight, optimizer update, sampling,
PINN schedule/lambda, GNN, Set2Set, branch hidden size는 변경하지 않았다.

## D. 검증 결과

실제 Fold 1 P07 sample ID 1689:

```text
same src/dst: true / true
generic role IDs: 6 / 6
property role IDs: 10 / 11
base feature: [18, 2051]
base max abs diff: 0.0
role embedding max abs diff: 3.1478
augmented feature: [18, 2067]
augmented max abs diff: 3.1478
initial prediction pair max abs diff: 0.1767
```

네 평행 쌍 모두 실제 encoder 기반 base difference가 0.0이고 property role
및 augmented feature difference는 0보다 컸다.

테스트:

- property role 전용 및 기존 decoder/model 회귀: `20 passed`
- role embedding 및 Condition/Fraction/Flow 첫 Linear gradient: finite, norm > 0
- role shuffle: prediction 변화 확인
- disabled: embedding parameter 없음, input 2051D, role tensor와 무관한 exact output
- enabled: input 2067D, main/rho/h output shape 유지
- 별도 v3 smoke 2개는 저장소에 이미 없는
  `configs/experiment/process_surrogate_base.yaml`을 참조해 collection 후 실패했으며,
  이번 변경과 관련된 assertion 실패는 아니다.

## E. Config와 Fold 1 실행

Role16 config:

`configs/experiment/pinn/model_260716_pi_tw40_legacy_fracfocus_f01_streamrole16.yaml`

이 config는 baseline과 다음 차이만 갖는다.

```yaml
property_stream_role:
  enabled: true
  embedding_dim: 16
  unknown_role_id: 0
```

Baseline E0:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_process_kfold_experiments.py \
  --base-config configs/experiment/pinn/model_260716_pi_tw40_legacy_fracfocus_f01.yaml \
  --merged-csv data/datasets_v3/process_main_merged.csv \
  --process-ids 1 2 3 4 5 6 7 8 9 10 \
  --joint-all-processes --only-folds 1 --max-epochs 30 \
  --monitor-metric val_target_mean_r2 --monitor-mode max \
  --output-root outputs/0728_parallelrole_e0_base_f01 \
  --force-reevaluate --skip-startup-debug
```

Role16 E1:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python scripts/run_process_kfold_experiments.py \
  --base-config configs/experiment/pinn/model_260716_pi_tw40_legacy_fracfocus_f01_streamrole16.yaml \
  --merged-csv data/datasets_v3/process_main_merged.csv \
  --process-ids 1 2 3 4 5 6 7 8 9 10 \
  --joint-all-processes --only-folds 1 --max-epochs 30 \
  --monitor-metric val_target_mean_r2 --monitor-mode max \
  --output-root outputs/0728_parallelrole_e1_role16_f01 \
  --force-reevaluate --skip-startup-debug
```

## F. 남은 위험

1. `OUT_stream`은 의미가 포괄적이어서 `output_other`로만 구분된다.
2. unseen process도 output port metadata를 제공해야 한다. 모르는 port는
   generic role 또는 UNKNOWN으로 돌아가며 평행 edge 분리력이 약해질 수 있다.
3. Role-enabled head의 첫 Linear 입력 shape가 달라 기존 checkpoint는 strict
   load할 수 없다. Fold 1/5-fold ablation은 scratch training으로 실행한다.
4. 구조적 식별 가능성은 확보했지만 CH4/CO/CO2 R2 개선 여부는 학습 결과로
   판단해야 한다. 비평행/all-edge 성능과 함께 비교해야 한다.
