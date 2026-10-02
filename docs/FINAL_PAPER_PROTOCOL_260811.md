# 최종 논문 실험 프로토콜 260811

## 1. 최종 기준

- Proposed: `model_260805` 구조를 계승한 `model_260811_novol_fast.yaml`
- 최종 target 10D: Temp, Pres, H2O, H2, CH4, CO2, CO, O2, N2, Mass_Flow
- Vol_Flow: prediction, loss, monitor, final metric에서 완전 제외
- split: 60/20/20, five-fold
- monitor: `val_target_edge_property_mean_r2`, mode=max, patience=5
- test: checkpoint 선택에 사용하지 않고 최종 평가에만 사용
- Ablation과 GraphSAGE: final registry에서 제외

## 2. Repository Audit

| Experiment / Component | Required Final Design | Current Implementation | Status | Required Modification | Relevant Files |
|---|---|---|---|---|---|
| SVR/RF/XGB | 공정별 D_process, 10D target | 기존 baseline dataset/preprocessor 재사용 | READY | final runner 연결 | `run_final_baseline_experiments.py` |
| B1-B7 | 문헌 구조 보존, 현재 I/O만 적용 | 기존 registry/config 재사용 | READY | 13-model registry 고정 | Baselines `configs/baseline/b*.yaml` |
| GCN/GIN/GAT | 동일 64D/3-layer/target-edge head | connectivity와 target src/dst readout, GAT 4 heads x 16D | READY | 모든 공정/target edge가 하나의 10D head 공유 | Baselines `simple_gnn.py` |
| GraphSAGE | final 제외 | legacy 구현 존재 | OUTDATED | final runner/summary에서 제외 완료 | legacy config only |
| Multi MLP | process adapter -> shared MLP -> process head | joint model 신규 연결 | READY | 없음 | Baselines `joint.py` |
| Multi GCN/GIN/GAT | shared graph encoder와 shared target-edge head | process head 제거, 10 process batch scheduler 유지 | READY | 없음 | Baselines `joint.py` |
| Proposed joint | 10-process one model, 5 folds | no-Vol 10D Proposed | READY | final split 경로 명시 | `model_260811_novol_fast.yaml` |
| Zero-shot | Proposed only, 9 -> 1, 10x5 | runner 재사용 가능 | READY | 11D 과거 결과는 legacy, 10D 재학습 | `run_single_process_full_unseen_experiments.py` |
| Transfer | Proposed adaptation | 기존 일반 transfer 존재 | PARTIAL | final main table에서는 data efficiency만 사용 | same runner |
| Data efficiency | 2/4/6/8% 및 10-90%, nested, fixed S | ratio/subset/fixed-step 통합 | READY | 2/4/6/8% 추가 실행 | `run_transfer_data_efficiency_experiments.py` |
| Time/memory | joint 5 models, 5 folds | wall time/peak allocated 기록 및 집계 | READY | 전용 GPU에서 production 실행 | `aggregate_final_efficiency.py` |
| Big-O | 5 joint models | 공통 symbol 표현 저장 | READY | 없음 | same aggregator |
| SHAP | Proposed target prediction, raw inputs | GradientExplainer, node aggregation, 좌표 사전검증 구현 | READY | Proposed fold-1 checkpoint 필요 | `run_final_shap_*.py` |
| Stream attribution | 실제 attributable raw stream만 | sample-varying raw stream input 없음 | READY | unavailable status 저장, 임의 생성 금지 | SHAP runner |
| Flowsheet overlay | 실제 이미지와 명시 좌표 | 10개 이미지, canonical 174-node 좌표와 preview 검증 완료 | READY | production SHAP 실행 | `unit_coordinates.csv` |
| Depth sensitivity | full block depth 1-7 | `model.num_layers`만 변경 | READY | depth 5 main run 재사용 | `run_final_sensitivity_experiments.py` |
| PIN sensitivity | active term OAT 0.5/1/2x | mass/component/atom만 등록 | READY | 공통 default 재사용 | same runner |
| Split/scaler | train-only fit, val/test 독립 | 기존 loader 재사용, subset scaler assertion | READY | 없음 | data-efficiency runner |
| Registry/status/aggregation | dry/resume/status/8 summaries | 신규 통합 | READY | 없음 | `final_*`, `aggregate_final_*` |

Legacy로 유지하지만 final aggregate에서 제외하는 것은 GraphSAGE, 11D Vol_Flow
결과, 5/10/25/50/100 data-efficiency, independent-process 결과를 multi-process로
합산한 과거 efficiency 결과, ablation 결과다.

## 3. Baseline 설계

Single-process는 M1-M3, B1-B7, G1-G3의 13개이며 10 process x 5 folds,
총 650 runs다. 각 process는 자기 D_process만 가지며 train split에서 feature
선택, imputation, X/Y scaler를 fit한다. Output은 target edge x 10 properties다.

G1/G2/G3은 GCN/GIN/GAT이다. 세 모델은 64D hidden, 3 layers, sum pooling과
동일한 target-edge readout을 쓴다. GAT은 4 heads x 16D라서 최종 node
representation은 다른 모델과 같은 64D다. 각 target edge는
`[h_src(64); h_dst(64); h_global(64)] = 192D` instance가 되고, 모든 target
edge에 같은 `192 -> 64 -> 10` MLP head를 적용한다. 따라서 한 graph의 target
edge가 3개면 10D prediction 3개가 나오며, graph-level 10D 값을 복제하지 않는다.

Multi-process는 MLP/GCN/GIN/GAT/Proposed 5개 x 5 folds다. MLP는 공정별
input adapter와 output head 사이의 shared MLP를 쓴다. Generic GNN은 basic
node encoder, message passing, sum pooling, target-edge head를 모두 공유한다.
Generic GNN forward에는 Process ID가 들어가지 않으며 process-specific adapter/head도 없다.
Process metadata는 가변 target-edge 개수에 맞춰 평가 배열을 다시 펼칠 때만 사용한다. Generic GNN에는 HX pair,
FeedHead, semantic role, Set2Set, edge-conditioned FlowGNN, PIN을 넣지 않는다.

## 4. Proposed와 PIN

Proposed의 depth 하나는 forward/reverse relational message, 양방향 fusion,
residual update를 모두 포함한 full message-passing block 1회다. 기본 depth는 5다.

현재 active PIN term과 기본 weight:

| Term | Weight | 적용 | Schedule |
|---|---:|---|---|
| node mass | 1.0 | node balance step | epoch 1-5=0, 6-7=0.5x, 8+=1x |
| node component | 1.5e-7 | component mass balance | 동일 |
| node atom | 0.2 | atom balance | 동일 |

Energy, density, enthalpy, volume PIN은 최종 config에서 0이다. Sensitivity는 위
세 active term만 한 번에 하나씩 0.5x/default/2x로 바꾼다. 0은 ablation이므로 쓰지 않는다.

## 5. Data Efficiency

- 10 held-out processes x 5 folds x 13 ratios = 650 runs
- ratio: 10,20,30,40,50,60,70,80,90%
- fold별 deterministic permutation prefix로 nested subset 생성
- subset count/hash/indices 저장
- train subset만 scaler fit, validation/test는 모든 ratio에서 동일
- early stopping으로 종료하지 않음
- 모든 ratio는 최대 20 epoch까지 학습
- 80,000 applied optimizer steps는 정확한 목표가 아니라 안전 상한
- validation은 20 epoch 안에서 best checkpoint를 선택하는 데만 사용
- 작은 subset은 subset 내부 sample만 여러 epoch 반복

종료 시점은 `20 epoch`와 `80,000 applied optimizer steps` 중 먼저 도달한
시점이다. 작은 ratio를 step 수에 맞추기 위해 한 epoch 안에서 과도하게 반복하지
않는다. 새 runner는 실제 applied step 수, step 상한, 상한 도달 여부를 저장하며,
상한을 초과한 경우에만 aggregation을 거부한다. 따라서 ratio별 실제 step 수는
다를 수 있고, 비교의 공통 학습 horizon은 20 epoch이다.

## 6. Efficiency

같은 joint 10-process 5-fold production run에서 Target Mean R2와 함께 training
시작부터 종료까지 wall-clock 합계, `torch.cuda.max_memory_allocated()` peak,
parameter count를 기록한다. 논문용 시간은 동일 GPU 사양의 독점 실행만 인정한다.

| Model | 핵심 Big-O |
|---|---|
| MLP | O(B d_in d + B L d^2) |
| GCN | O(L (|E|d + |V|d^2)) |
| GIN | O(L (|E|d + |V|d^2)) |
| GAT | O(L H (|E|d_h + |V|d_h^2)), H d_h=d |
| Proposed | O(L (|E|d^2 + |V|d^2) + Set2Set + edge readout) |

PIN은 inference Big-O에 포함하지 않고 training-only extra forward/backward로 표시한다.

## 7. Explainability

`shap.GradientExplainer`를 선택했다. Proposed가 PyTorch differentiable model이고
KernelExplainer보다 계산량이 작으면서 실제 target scalar까지 gradient 경로가 보존되기
때문이다. Background 32, explanation sample 16을 기본으로 한다.

입력은 raw node operating variables와 raw direct feed 값이다. Wrapper 내부에서 해당
checkpoint의 train-only mean/std로 정규화하여 기존 forward를 그대로 호출한다.
target edge/property마다 sample-level SHAP, global feature importance, node별 mean
absolute SHAP, Top-K를 저장한다. 실제 sample-varying raw stream feature가 없으므로
stream importance는 `not_available`로 저장한다.

공정도 overlay는 `unit_coordinates.csv`의 이미지 대조 좌표만 허용한다. 물리 장치는
원본 이미지의 장치 중심, `V_INPUT/V_OUTPUT`은 feed/product 경계 anchor에 둔다.
canonical 174개 node의 누락·중복·이미지 경계 검사는
`validate_shap_flowsheet_coordinates.py`가 SHAP 실행 전에 수행하며, 검증 preview 10개도
저장한다. production runner는 `--require-flowsheet`를 사용하므로 overlay가 없으면 해당
SHAP run을 성공으로 처리하지 않는다.

## 8. 실행 규모

| Family | Logical runs | 새 학습/평가 |
|---|---:|---:|
| Single baselines | 650 | 650 |
| Multi comparison | 25 | 25 |
| Zero-shot | 50 | pretrain 10 + eval 50 |
| Data efficiency | 650 | 650 |
| Efficiency | 25 | multi run 재사용 |
| SHAP | 10 process | fold-1 Proposed checkpoint 재사용 |
| Depth | 35 | depth-5 재사용, 새 30 |
| PIN | 45 | default 재사용, 새 30 |

대규모 새 training은 1,195 runs다. 구성은 single 650 + multi 25 +
zero-shot용 10D pretrain 10 + data-efficiency 650 + depth 30 + PIN 30이다.
Zero-shot evaluation 50과 SHAP Process 분석 10은 training과 별도다.

## 9. 실행 명령

전체 dry plan:

```bash
PYTHONPATH=src python3 scripts/run_final_experiments.py --families all
```

전체 순차 실행:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python3 scripts/run_final_experiments.py --families all --execute
```

GPU 병렬화는 runner 하나를 중복 실행하지 않고 process/fold/model 인자를 서로 겹치지
않게 나눈다. 예를 들어 single baseline은 GPU별 Process를, joint/sensitivity는 fold를,
zero-shot/data-efficiency는 held-out Process를 분할한다. 각 shard는 동일 output root와
`--resume-existing`을 사용한다.

상태와 최종 집계:

```bash
PYTHONPATH=src python3 scripts/final_experiment_status.py --output-root outputs/final_paper
PYTHONPATH=src python3 scripts/aggregate_final_efficiency.py
PYTHONPATH=src python3 scripts/aggregate_final_experiments.py --root outputs/final_paper
```

생성 summary:

```text
single_process_baseline_summary.csv
multi_process_summary.csv
zero_shot_summary.csv
data_efficiency_summary.csv
computational_efficiency_summary.csv
explainability_summary.csv
depth_sensitivity_summary.csv
pin_sensitivity_summary.csv
```

## 10. 검증 결과

- Python compile: final runners/config helpers 모두 통과
- Baseline actual smoke: single GCN/GIN/GAT 및 joint GCN/GIN/GAT 완료
- joint generic GNN은 P1 3 edges/P2 2 edges를 같은 `192 -> 64 -> 10` head로 처리하며
  `process_specific_output_heads=false`, `process_id_usage=metadata_only`를 확인
- Data subset dry-run: P1/F1 pool=6,001, 10%=600, nested/disjoint/subset-only=true
- Fixed-step actual smoke: budget=3, 실제 누적 optimizer steps=3.0, best checkpoint/eval 완료
- SHAP actual smoke: P03_E014 Frac_CO, background=2, sample=1 완료
- Protocol/transfer/target-metric regression pytest: 9 passed
- Windows pytest 종료 뒤 pyarrow access-violation 문구는 발생했으나 pytest exit result는 성공

## 11. 수정 및 추가 파일

Current repository:

- `configs/final_experiments/final_protocol_260811.yaml`
- `configs/experiment/pinn/model_260811_novol_fast.yaml`
- `scripts/run_final_baseline_experiments.py`
- `scripts/run_final_experiments.py`
- `scripts/final_experiment_registry.py`
- `scripts/final_experiment_status.py`
- `scripts/run_transfer_data_efficiency_experiments.py`
- `scripts/run_final_sensitivity_experiments.py`
- `scripts/derive_final_fixed_step_budget.py`
- `scripts/aggregate_final_efficiency.py`
- `scripts/aggregate_final_experiments.py`
- `scripts/run_final_shap_explainability.py`
- `scripts/run_final_shap_experiments.py`
- `scripts/train_process_surrogate.py`
- `src/process_graph/experiment/schema.py`
- `src/process_graph/experiment/metric_policy.py`
- `src/process_graph/experiment/edge_step_pi_training.py`
- `tests/test_final_paper_protocol_260811.py`

Sibling Baselines repository:

- `src/process_graph/baselines/models/simple_gnn.py`
- `src/process_graph/baselines/models/joint.py`
- `src/process_graph/baselines/registry.py`
- `configs/baseline/g2_flowsheet_gin_final.yaml`
- `configs/baseline/g3_flowsheet_gat_final.yaml`

## 12. 최종 판정

학습 runner, fixed-step, registry, status, aggregation, SHAP 계산 및 sensitivity는
구현 완료다. 전체 대규모 학습은 지침대로 실행하지 않았다. 논문용 flowsheet overlay는
10개 공정 이미지와 canonical 174-node 좌표 검증을 완료했으며, production SHAP에서
target edge/property별 node importance를 실제 공정도 위의 색과 marker 크기로 표시한다.

## 13. 3-server phase 실행과 GPU 분배

`run_final_protocol_3servers.sh`가 서버별 물리 GPU를 직접 배정하므로 명령 앞에
`CUDA_VISIBLE_DEVICES=0`을 붙이지 않는다.

```bash
# server 1: GPU 0,1,2,3 -> Process 1,2,3,4
bash scripts/run_final_protocol_3servers.sh 1 1

# server 2: GPU 0,1,2,3 -> Process 5,6,7,8
bash scripts/run_final_protocol_3servers.sh 2 1

# server 3: GPU 0,1 -> Process 9,10; GPU 2 -> joint five-fold
bash scripts/run_final_protocol_3servers.sh 3 1
```

Phase 2와 3은 두 번째 인자를 각각 `2`, `3`으로 바꾼다. launcher는 상속된 단일-GPU
mask를 해제하고, `nvidia-smi` 기준으로 서버 1/2는 최소 4개, 서버 3은 최소 3개 GPU가
보이는지 확인한다. 시작 로그의 `[gpu-plan]` 줄과 `nvidia-smi`에서 worker가 각 GPU에
분산되었는지 확인한다.
