# Model Handoff: Edge-All Process Surrogate

이 문서는 다른 생성형 AI 또는 다음 작업자가 현재 모델 구조와 loss 동작을 빠르게 이어받을 수 있도록 정리한 handoff 문서입니다.

대상 코드는 `task_mode: edge_all` 기반 chemical process graph surrogate입니다. 현재 핵심 변경점은 target row의 특정 Frac feature만 강조하던 보조 loss를, target edge 양끝 노드에 연결된 incident edge 전체 feature를 강조하는 방식으로 바꾼 것입니다.

## 1. 핵심 요약

현재 모델은 공정 그래프의 모든 canonical stream edge에 대해 12개 stream property를 예측합니다.

```text
y_edge_pred: [num_edges_in_batch, 12]
y_edge_true: [num_edges_in_batch, 12]
y_edge_mask: [num_edges_in_batch, 12] 또는 broadcast 가능한 mask
```

최종 training loss는 현재 다음 세 항으로 구성됩니다.

```text
loss_total_final
= loss_edge_all
+ primary_frac_loss_weight * loss_target_incident_edges
+ all_edge_r2_loss_weight * loss_all_edge_r2
```

코드 호환성 때문에 로그와 dict key에는 아직 `loss_primary_frac`, `loss_target_row_frac`, `loss_v4_targets_frac` 이름이 남아 있습니다. 하지만 `target_feature_loss_scope: target_incident_edges` 설정에서는 이 값들이 실제로는 `loss_target_incident_edges`를 의미합니다.

## 2. 주요 파일

| 역할 | 파일 |
|---|---|
| 학습 entrypoint | `scripts/train_process_surrogate.py` |
| 모델 wrapper | `src/process_graph/models/process_surrogate.py` |
| graph encoder | `src/process_graph/models/process_encoder.py` |
| edge decoder | `src/process_graph/models/edge_decoder.py` |
| edge stream prediction head | `src/process_graph/models/edge_stream_head.py` |
| edge_all loss assembly | `src/process_graph/experiment/train_utils.py` |
| target row / incident loss | `src/process_graph/experiment/target_row_loss.py` |
| v4 target row spec | `src/process_graph/experiment/target_row_spec.py` |
| target v4 metrics | `src/process_graph/experiment/target_v4_metrics.py` |
| loss/metric policy logging | `src/process_graph/experiment/metric_policy.py` |
| terminal supervision summary | `src/process_graph/experiment/edge_all_supervision_log.py` |

주요 config:

| 역할 | 파일 |
|---|---|
| 기본 train config | `configs/train/process_train.yaml` |
| edge_all baseline | `configs/experiment/process_surrogate_edge_all_v3.yaml` |
| grouped head common | `configs/experiment/process_surrogate_edge_all_v3_grouped_head.yaml` |
| grouped head per-process | `configs/experiment/process_grouped_head/process_surrogate_edge_all_v3_grouped_head_p01.yaml` - `p10.yaml` |

## 3. 모델 동작 흐름

`edge_all` 모드에서는 fixed target head가 아니라 edge decoder가 핵심입니다.

```text
GraphBatch
  model_kwargs:
    edge_index
    node features
    edge_struct_attr 또는 edge_oper
    batch ids
  targets:
    edge_stream = y_edge_true
  target_masks:
    edge_stream = y_edge_mask
  edge_export_meta:
    process_id
    canonical_edge_id
    main_data_stream_key
    ...

ProcessSurrogateModel.forward()
  -> ProcessGraphEncoder
  -> EdgeDecoder
  -> preds["y_edge_pred"]
```

`ProcessSurrogateModel.predict()`에서 `edge_decoder`가 켜져 있으면 다음 출력을 만듭니다.

```text
predictions["y_edge_pred"] = edge_decoder(
    encoder_output.local_node_embeddings,
    encoder_output.edge_index,
    edge_struct_attr,
    encoder_output.global_embedding,
    encoder_output.batch,
)
```

`edge_index`는 나중에 incident edge loss에서도 사용됩니다. `scripts/train_process_surrogate.py`의 `_edge_all_loss_kwargs()`가 `batch.model_kwargs["edge_index"]`를 loss 함수로 넘깁니다.

## 4. Edge Feature Target

모델은 각 edge마다 `STREAM_EDGE_FEATURE_SLOTS` 순서의 stream property를 예측합니다.

```text
0  Temp
1  Pres
2  Vol_Flow
3  Mole_Flow
4  Mass_Flow
5  Frac_H2O
6  Frac_H2
7  Frac_CH4
8  Frac_CO2
9  Frac_CO
10 Frac_O2
11 Frac_N2
```

학습 loss는 기본적으로 normalized `y_edge` 공간에서 계산됩니다.

```text
y_edge_norm = (y_edge_raw - y_edge_mean) / y_edge_std
```

metric/export 단계에서는 original scale로 inverse transform한 뒤 평가합니다.

## 5. 최종 Loss Assembly

중앙 assembly point는 `src/process_graph/experiment/train_utils.py`의 `compute_edge_all_training_loss()`입니다.

현재 흐름:

```text
1. loss_edge_all 계산
2. loss_all_edge_r2 계산
3. target_feature_loss_scope 확인
4. target_incident_edges이면 compute_target_incident_edge_loss() 호출
5. amount auxiliary 계산
6. 최종 loss_total_final 조립
```

실제 합산:

```python
weighted_edge_all = loss_edge_all
weighted_primary_frac = lam_primary * loss_primary_frac
weighted_all_edge_r2 = lam_primary_r2 * loss_all_edge_r2
weighted_amount_auxiliary_disabled = lam_amount * loss_amount

loss_total_final = (
    weighted_edge_all
    + weighted_primary_frac
    + weighted_all_edge_r2
)
```

주의: `weighted_amount_auxiliary_disabled`는 계산만 하고 `loss_total_final`에는 포함하지 않습니다.

## 6. Base Loss: loss_edge_all

`loss_edge_all`은 전체 supervised edge-stream cell에 대한 masked regression loss입니다.

```text
loss_edge_all
= mean over (edge e, feature f) where mask[e, f] > 0:
    SmoothL1(y_pred[e, f], y_true[e, f])
```

정확히는 `loss_type_edge_all` 설정을 따릅니다.

```yaml
loss_type_edge_all: smooth_l1
```

`masked_edge_regression_loss()`가 사용되며, mask가 positive인 cell만 평균에 들어갑니다.

legacy element weighting이 켜져 있으면 `edge_stream_loss_weight`가 element-wise로 곱해질 수 있습니다. 하지만 현재 주요 grouped config에서는 다음처럼 꺼져 있습니다.

```yaml
use_legacy_answer_weighting: false
```

## 7. New Auxiliary Loss: loss_target_incident_edges

이번 변경의 핵심입니다.

### 7.1 이전 방식

이전 보조 loss는 target row가 지정한 target edge의 특정 Frac feature만 강조했습니다.

예:

```text
H2 target row  -> target edge의 Frac_H2만 강조
CO2 target row -> target edge의 Frac_CO2만 강조
H2O target row -> target edge의 Frac_H2O만 강조
```

### 7.2 현재 방식

현재 `target_feature_loss_scope: target_incident_edges`이면 target edge의 양끝 노드를 기준으로 incident edge를 찾고, 그 edge들의 전체 supervised feature를 강조합니다. 단, 기본 설정에서는 `V_INPUT`, `V_OUTPUT` 같은 virtual node는 expansion 기준에서 제외하고 target edge 자기 자신은 항상 다시 포함합니다.

target row spec 하나가 가리키는 target edge를 다음처럼 두면:

```text
target edge e_t = (src_t, dst_t)
```

incident edge set은 다음입니다.

```text
incident_edges_raw(e_t)
= { e | e.src == src_t
       or e.dst == src_t
       or e.src == dst_t
       or e.dst == dst_t }

active_endpoint_nodes
= {src_t if src_node_name not in virtual_names}
  union {dst_t if dst_node_name not in virtual_names}

incident_edges_final
= edges touching active_endpoint_nodes
  union {e_t if target_incident_include_target_edge}
```

그 target row 하나의 loss는:

```text
L_incident(target_id)
= final_weight(target_id)
  * mean over e in incident_edges, f where mask[e, f] > 0:
      SmoothL1(y_pred[e, f], y_true[e, f])
```

여기서 `final_weight(target_id)`는 target/species weight입니다.

```text
internal_weight = target_weight_by_id[target_id] if exists else target_weight
if target_incident_use_answer_edge_weight:
    internal_weight *= answer_edge_weight_<species>
```

species weight fallback:

```yaml
answer_edge_weight: 1.0
answer_edge_weight_h2: 1.0
answer_edge_weight_co2: 1.0
answer_edge_weight_h2o: 1.0
```

현재 기본 설정에서는 `target_incident_use_answer_edge_weight: false`이므로 incident loss 내부에서 `answer_edge_weight_h2/co2/h2o`를 다시 곱하지 않습니다. 이 설정은 `answer_edge_weight_*`와 `primary_frac_loss_weight`가 함께 곱해져 effective weight가 25배처럼 커지는 것을 막기 위한 것입니다.

중요한 해석:

```text
loss_target_incident_edges 내부:
  target_weight_by_id 또는 target_weight만 반영
  기본값에서는 answer_edge_weight_* 미반영

loss_total_final 조립:
  primary_frac_loss_weight 추가 반영
```

만약 과거처럼 incident loss 내부에도 species별 answer weight를 적용하고 싶으면 다음을 명시적으로 켭니다.

```yaml
target_incident_use_answer_edge_weight: true
```

### 7.3 평균 방식

`target_feature_loss_balancing: process_target_balanced`일 때:

```text
1. 같은 (process_id, target_id)의 term 평균
2. process 안의 target 평균
3. batch 안의 process 평균
```

이 구조는 특정 process 또는 특정 target row가 batch에서 많이 등장해 loss를 지배하지 않게 하려는 목적입니다.

### 7.4 코드 위치

`compute_target_incident_edge_loss()`:

```text
src/process_graph/experiment/target_row_loss.py
```

핵심 로직:

```python
src = edge_index[0, target_edge]
dst = edge_index[1, target_edge]

incident = (
    (edge_index[0] == src)
    | (edge_index[1] == src)
    | (edge_index[0] == dst)
    | (edge_index[1] == dst)
)

edge_rows = torch.nonzero(incident).flatten()
li = masked_edge_regression_loss(
    y_pred[edge_rows],
    y_true[edge_rows],
    y_mask[edge_rows],
)
```

## 8. R2 Auxiliary: loss_all_edge_r2

`loss_all_edge_r2`는 전체 supervised edge cell을 대상으로 feature-wise SSE/SST loss를 계산합니다. 현재는 새 config key인 `all_edge_r2_*`를 우선 사용하고, 기존 `primary_frac_r2_*`는 backward compatibility용 fallback입니다.

feature `f`마다:

```text
SSE_f = sum((y_pred[:, f] - y_true[:, f])^2)
SST_f = sum((y_true[:, f] - mean(y_true[:, f]))^2)

R2Loss_f = SSE_f / max(SST_f, threshold)
```

최종:

```text
loss_all_edge_r2 = mean_f(R2Loss_f)
```

단, 다음 조건을 만족하지 않는 feature는 제외됩니다.

```text
num_supervised_cells_for_feature < all_edge_r2_min_count
SST_f <= all_edge_r2_sst_threshold
```

config 예:

```yaml
all_edge_r2_loss_weight: 0.05
all_edge_r2_min_count: 8
all_edge_r2_sst_threshold: 1.0e-3
all_edge_r2_loss_cap: 5.0

target_incident_exclude_virtual_nodes: true
target_incident_virtual_node_names: [V_INPUT, V_OUTPUT]
target_incident_include_target_edge: true
target_incident_use_answer_edge_weight: false
```

`all_edge_r2_loss_cap`이 0보다 크면 feature별 R2 loss term을 cap합니다.

Fallback 우선순위:

```text
1. all_edge_r2_* key가 있으면 사용
2. 없으면 primary_frac_r2_* key 사용
3. 둘 다 없으면 코드 default 사용
```

로그에는 `all_edge_r2_config_source`가 `new_all_edge_r2_keys`, `legacy_primary_frac_r2_keys`, `default` 중 하나로 남습니다.

## 9. Amount Auxiliary

`compute_v4_target_row_amount_loss()`도 여전히 호출됩니다.

하지만 현재 최종 loss에는 포함하지 않습니다.

```text
loss_amount_auxiliary = 계산됨
weighted_amount_auxiliary_disabled = lam_amount * loss_amount_auxiliary
loss_total_final에는 더하지 않음
```

따라서 amount term은 현재 diagnostic/logging 용도입니다.

## 10. Config Snapshot

`configs/experiment/process_grouped_head/process_surrogate_edge_all_v3_grouped_head_p01.yaml` 기준:

```yaml
use_target_row_amount_loss: false
lambda_target_feature: 1.0
lambda_target_row_frac: 1.0
lambda_target_row_amount: 0.0

primary_frac_loss_weight: 5.0
all_edge_r2_loss_weight: 0.05
all_edge_r2_min_count: 8
all_edge_r2_sst_threshold: 1.0e-3
all_edge_r2_loss_cap: 5.0

# legacy fallback keys; 삭제하지 않음
primary_frac_r2_loss_weight: 0.05
primary_frac_r2_min_count: 8
primary_frac_r2_sst_threshold: 1.0e-3
primary_frac_r2_loss_cap: 5.0

target_feature_loss_scope: target_incident_edges
target_feature_loss_space: normalized
target_feature_loss_balancing: process_target_balanced

answer_edge_weight: 1.0
answer_edge_weight_h2: 1.0
answer_edge_weight_co2: 1.0
answer_edge_weight_h2o: 1.0
target_incident_use_answer_edge_weight: false
use_v4_target_edge_weighting: true
v4_target_weighting_mode: target_row
use_v4_target_row_loss: false
use_legacy_answer_weighting: false
```

위 설정에서 실질적인 loss는:

```text
loss_total_final
= loss_edge_all
+ 5.0 * loss_target_incident_edges
+ 0.05 * loss_all_edge_r2
```

현재 grouped head 기본값에서는 incident 내부 `internal_weight ~= 1.0`, 최종 `effective_weight ~= 5.0`입니다. `answer_edge_weight_*`는 legacy/element weighting 호환을 위해 남아 있지만, `target_incident_use_answer_edge_weight: false`이면 incident auxiliary 내부에는 곱하지 않습니다.

## 11. Logging Key 해석 주의

현재 key 이름과 실제 의미:

| key | 현재 의미 |
|---|---|
| `loss_edge_all` | 전체 supervised edge feature base loss |
| `loss_primary_frac` | scope가 `target_incident_edges`이면 incident edge auxiliary loss |
| `loss_target_row_frac` | `loss_primary_frac` alias |
| `loss_v4_targets_frac` | `loss_primary_frac` alias |
| `loss_all_edge_r2` | all-edge feature-wise SSE/SST auxiliary |
| `loss_amount_auxiliary` | amount diagnostic, 최종 loss에는 미포함 |
| `weighted_edge_all` | `loss_edge_all` |
| `weighted_primary_frac` | `primary_frac_loss_weight * loss_primary_frac` |
| `weighted_all_edge_r2` | `all_edge_r2_loss_weight * loss_all_edge_r2` |
| `loss_total_final` | 실제 backprop loss |
| `target_incident_coverage_edge_frac` | batch edge 중 incident auxiliary가 덮은 edge 비율 |
| `target_incident_effective_weight_mean` | incident 내부 weight 평균에 `primary_frac_loss_weight`를 곱한 평균 effective weight |
| `target_incident_loss_ratio_total` | `weighted_target_incident_edges / loss_total_final` |
| `target_incident_loss_ratio_edge` | `weighted_target_incident_edges / weighted_edge_all` |
| `all_edge_r2_config_source` | 새 all-edge R2 key를 썼는지 legacy fallback인지 |
| `amount_auxiliary_enabled` | amount auxiliary 값이 계산되었는지 |
| `amount_auxiliary_contributes_to_total` | 기본값은 false. final loss에는 더하지 않음 |
| `weighted_amount_auxiliary_disabled` | 계산만 된 amount weighted diagnostic |

다른 AI가 작업할 때는 `loss_primary_frac` 이름만 보고 “Frac 한 칸 loss”라고 오해하면 안 됩니다.

## 12. Target Row Matching

Target row spec은 v4 reference에서 옵니다.

```text
data/reference/v4/target_stream_targets.csv
```

target row matching에는 보통 다음이 쓰입니다.

```text
process_id
required_stream_key 또는 main_data_stream_key
canonical_answer_edge_id
target_species
```

incident loss에서도 먼저 target row가 batch의 어떤 canonical edge와 매칭되는지 찾고, 그 edge의 `edge_index` src/dst를 사용합니다.

## 13. Evaluation Metric 관점

학습 loss는 normalized space에서 계산합니다. 반면 주요 평가 metric은 original scale로 복원한 뒤 target row 기준으로 계산됩니다.

주요 primary metric:

```text
eval_primary_frac_r2_by_process
eval_primary_frac_r2_by_target
```

monitor metric 예:

```yaml
monitor_metric: val_eval_primary_frac_r2_by_process
monitor_mode: max
```

amount metric은 현재 secondary/diagnostic 성격입니다.

## 14. 작업자가 특히 확인해야 할 지점

1. Loss 변경 작업 시 `compute_edge_all_training_loss()`를 먼저 볼 것.
2. `target_feature_loss_scope` 분기를 확인할 것.
3. `loss_primary_frac`라는 이름이 실제 의미와 다를 수 있음을 기억할 것.
4. `edge_index`가 loss kwargs로 전달되지 않으면 incident loss가 0이 될 수 있음.
5. amount auxiliary는 계산되지만 final loss에는 포함되지 않음.
6. `use_legacy_answer_weighting=true`를 켜면 base edge loss에 element-level weight가 추가될 수 있음.
7. all-edge R2는 feature별 분산이 거의 없거나 sample 수가 부족하면 skip됨.
8. run directory의 `incident_loss_diagnostics.csv`에서 incident coverage, effective weight, loss ratio를 확인할 것.

## 15. incident_loss_diagnostics.csv 해석

학습 중 run directory에 `incident_loss_diagnostics.csv`가 저장됩니다. 이 파일은 epoch summary가 아니라 target row 단위 diagnostic입니다.

핵심 확인 컬럼:

| column | 확인 내용 |
|---|---|
| `target_edge_canonical_id` | 실제 매칭된 target edge |
| `src_node`, `dst_node`, `src_is_virtual`, `dst_is_virtual` | V_INPUT/V_OUTPUT 판별이 문자열 node name 기준으로 됐는지 |
| `num_incident_edges_before_virtual_filter` | virtual filtering 전 incident 범위 |
| `num_incident_edges_after_virtual_filter` | virtual filtering 후 incident 범위 |
| `num_edges_removed_by_virtual_filter` | virtual node fanout 제거 개수 |
| `included_edge_canonical_ids` | 최종 auxiliary 대상 edge 목록 |
| `internal_weight` | incident loss 내부 target/spec weight |
| `effective_weight` | `internal_weight * primary_frac_loss_weight` |
| `ratio_weighted_target_incident_to_edge_all` | incident 보조항이 base edge loss 대비 얼마나 큰지 |
| `all_edge_r2_config_source` | `new_all_edge_r2_keys`이면 새 key 사용 중 |
| `amount_auxiliary_contributes_to_total` | 기본적으로 false여야 함 |

정상적인 grouped-head 기본 기대:

```text
matched_target_edges > 0
loss_target_incident_edges > 0
src_node 또는 dst_node가 V_OUTPUT이면 *_is_virtual=true
num_edges_removed_by_virtual_filter > 0인 row 존재
matched row에서는 target_edge_canonical_id가 included_edge_canonical_ids 안에 존재
internal_weight ~= 1.0
effective_weight ~= 5.0
effective_weight ~= 25.0인 row 없음
all_edge_r2_config_source = new_all_edge_r2_keys
```

## 16. 1 Epoch Smoke Train 확인

요청 command의 `--device`, `--wandb-mode` 옵션은 현재 스크립트 CLI에는 없을 수 있습니다. 이 경우 다음처럼 실행해도 diagnostics 생성 확인에는 충분합니다.

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
$env:PYTHONPATH='src'
python scripts/train_process_surrogate.py `
  --config configs/experiment/process_grouped_head/process_surrogate_edge_all_v3_grouped_head_p01.yaml `
  --process-filter 1 `
  --max-epochs 1 `
  --skip-startup-debug `
  --no-training-plots `
  --no-save-model-weights
```

2026-05-28 smoke 결과에서는 `incident_loss_diagnostics.csv`가 생성되었고, Process1 matched rows에서 V_OUTPUT filtering, target edge 유지, `effective_weight=5.0`, `all_edge_r2_config_source=new_all_edge_r2_keys`가 확인되었습니다.

## 17. 현재 변경에 대한 최소 검증

이번 incident-edge loss 변경 후 확인한 테스트:

```powershell
$env:PYTHONPATH='src'
pytest -q `
  tests/test_target_row_frac_loss.py `
  tests/test_target_row_supervision.py::test_target_incident_edge_loss_weights_all_features_on_edges_touching_target_nodes `
  tests/test_primary_frac_loss.py::test_primary_frac_losses_are_finite_and_backward_flows `
  tests/test_v4_h2o_target_routing.py `
  tests/test_target_incident_diagnostics.py
```

기대 결과:

```text
22 passed
```

테스트 의미:

- target edge 양끝 노드에 incident한 edge에만 gradient가 흐르는지 확인
- 기존 primary frac loss 경로가 여전히 finite/backward 가능한지 확인
- v4 H2O target routing/legacy weighting 테스트 유지

## 18. 한 문장 요약

현재 edge_all 모델은 모든 canonical stream edge의 12개 property를 예측하고, 기본 전체 edge loss에 더해 target edge 주변 incident edge 전체 feature를 강하게 맞추며, 전체 supervised edge cell에 대한 R2 보조항을 추가하는 구조입니다.
