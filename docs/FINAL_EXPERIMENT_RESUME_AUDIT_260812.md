# 최종 실험 재개 Audit (260812)

## 재개 시점 상태

| 분류 | 항목 | 재개 시점 판단 |
|---|---|---|
| DONE | 최종 10D target schema | Vol_Flow를 제외한 10개 물성으로 registry와 Proposed config가 연결되어 있었다. |
| DONE | 최종 실험 runner 골격 | single baseline, multi-process, zero-shot, data efficiency, sensitivity, SHAP, aggregation 명령 생성이 구현되어 있었다. |
| DONE | Single-process 목록 | M1-M3, B1-B7, G1-G3의 13개 코드와 G1=GCN, G2=GIN, G3=GAT 매핑이 final YAML에 있었다. |
| DONE | GAT operator | 4 heads x 16D, 총 hidden 64D 구현과 GAT config가 있었다. |
| PARTIAL | Generic GNN 입력 | graph sample은 만들었지만 baseline 정리 단계에서 adjacency까지 삭제했다. |
| PARTIAL | Generic GNN target-edge readout | shared 10D head 표기는 있었지만 실제 target edge별 representation이 없었다. |
| PARTIAL | Single GNN validation | runner가 `val_data`를 전달했지만 model이 이를 무시해 patience와 best-state 복원이 작동하지 않았다. |
| PARTIAL | 최종 문서와 smoke | smoke는 실행됐지만 잘못된 process-specific head를 정상 설계로 기록했다. |
| DONE | SHAP flowsheet overlay | 10개 공정 이미지에 canonical 174-node 좌표를 대조하고 preview 및 필수검증 경로를 추가했다. |
| NOT STARTED | 전체 production 학습 | 요청대로 대규모 5-fold 학습은 아직 실행하지 않았다. |
| WRONG IMPLEMENTATION | Single-process GNN | graph-level 10D 예측 하나를 모든 target edge에 반복했다. 서로 다른 target edge를 구분할 수 없었다. |
| WRONG IMPLEMENTATION | Multi-process GNN | process마다 서로 다른 output head를 선택했다. Process ID가 head routing에 사용됐다. |
| WRONG IMPLEMENTATION | GNN graph construction | `edge_index`를 비우고 operating value 목록을 가짜 node 축으로 사용해 실제 message passing이 사라졌다. |
| WRONG IMPLEMENTATION | Baselines model registry | final GIN/GAT과 별도로 GraphSAGE 및 과거 GIN alias가 활성 registry에 남아 있었다. |

이전 작업은 runner와 실험 protocol을 만든 뒤, single GNN의 반복 예측과 joint GNN의
process-specific head를 smoke에서 허용한 지점에서 멈춰 있었다.

## 이어서 구현한 내용

### Generic GNN 공통 구조

GCN, GIN, GAT 모두 다음 구조를 쓴다.

```text
node operating input
-> shared node encoder
-> shared GCN / GIN / GAT message passing
-> global sum pooling + projection
-> each target edge: concat(h_src, h_dst, h_global)
-> SAME shared MLP: 192 -> 64 -> 10
-> one 10D property vector per target edge
```

- target edge는 canonical edge ID embedding 없이 실제 graph의 src/dst node 위치로만 조회한다.
- directed adjacency는 보존하지만 semantic stream role, Stream ID, HX pair, edge attribute,
  target flag, PIN loss는 generic baseline에 제공하지 않는다.
- Multi-process GNN forward에는 Process ID가 들어가지 않는다.
- process metadata는 process별 target edge 개수에 맞게 `[sample, edge, 10]`을 평가용
  flat array로 복원할 때만 사용한다.
- Multi-process MLP의 process별 input adapter/output head는 GNN correction 대상이 아니므로 유지했다.
- Single GNN은 validation MSE 기준 patience=3 early stopping과 best-state 복원을 사용한다.

### Registry와 final 구성

Single-process final baseline은 정확히 다음 13개다.

```text
SVR
Random Forest
XGBoost
Kriging-KR31
Cubic-RBF
GTL-ANN
Cumene-Efficiency-ANN
Cumene-Destruction-ANN
Reusable-Distillation-ANN
Distillation-Boundary-GP
GCN
GIN
GAT
```

GraphSAGE는 active model registry와 final runner에서 제거했다. 과거 YAML 파일은 legacy
참조용으로만 남고 final registry, command plan, aggregation에는 들어가지 않는다.

Multi-process 비교는 MLP, GCN, GIN, GAT, Proposed의 5개를 유지한다.

## 검증 결과

- Baselines regression tests: 29 passed
- Final protocol regression tests: 5 passed
- Single runner smoke: GCN, GIN, GAT 모두 Process1/Fold1에서 completed
- Multi runner smoke: GCN, GIN, GAT 모두 Process1+Process2/Fold1에서 completed
- Joint metadata: `process_specific_output_heads=false`
- Joint metadata: `shared_target_edge_head=true`
- P1 target edge count 3, P2 target edge count 2를 동일한 `192 -> 64 -> 10` head로 처리
- 실제 P1 smoke의 같은 sample/Temp 예측도 세 target edge에서 서로 달라 graph-level 반복 예측이 제거됨
- Legacy efficiency diagnostic GAT smoke도 completed

Windows pytest 종료 뒤 알려진 pyarrow access-violation 문구가 출력되지만 pytest 결과 자체는
성공이다.

## 이번에 수정한 파일

현재 repository:

- `configs/final_experiments/final_protocol_260811.yaml`
- `scripts/final_experiment_registry.py`
- `scripts/run_final_baseline_experiments.py`
- `scripts/run_computational_efficiency_benchmark.py`
- `scripts/aggregate_computational_efficiency_shards.py`
- `tests/test_final_paper_protocol_260811.py`
- `docs/FINAL_PAPER_PROTOCOL_260811.md`
- `docs/COMPUTATIONAL_EFFICIENCY_260811.md`
- `docs/FINAL_EXPERIMENT_RESUME_AUDIT_260812.md`

Sibling Baselines repository:

- `src/process_graph/baselines/dataset.py`
- `src/process_graph/baselines/models/simple_gnn.py`
- `src/process_graph/baselines/models/joint.py`
- `src/process_graph/baselines/registry.py`
- `configs/baseline/g1_flowsheet_gcn.yaml`
- `configs/baseline/g2_flowsheet_gin_final.yaml`
- `configs/baseline/g3_flowsheet_gat_final.yaml`
- `tests/baselines/test_model_shapes.py`
- `tests/baselines/test_process_specific_inputs.py`

## 남은 작업

1. 대규모 5-fold production 학습 실행
2. SHAP flowsheet overlay용 수동 검증 unit 좌표 파일 작성

둘 다 이번 요청의 smoke-test 범위 밖이므로 실행하거나 임의 생성하지 않았다.
