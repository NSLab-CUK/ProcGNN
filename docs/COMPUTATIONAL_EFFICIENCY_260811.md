# Computational Efficiency Benchmark (260811)

> 이 문서는 process별 진단 runner의 기록이다. 최종 논문의 multi-process 계산효율 표는
> `run_final_baseline_experiments.py --scope multi`의 MLP/GCN/GIN/GAT 결과와 Proposed
> joint 결과를 `aggregate_final_efficiency.py`로 집계한다.

## 목적

제안 모델의 성능을 계산 비용과 함께 비교한다. 메인 비교 대상은 `GCN`, `GIN`, `GAT`, `Proposed model_260811_novol_fast` 네 가지다.

## 공정한 비교 조건

- 동일한 5 outer folds와 train/validation/test sample membership
- 동일 physical GPU, float32, training batch size 1
- max 30 epochs, early-stopping patience 3
- 동일 `val_target_mean_r2` membership 정책과 Vol_Flow monitor 제외 정책
- batch-1 inference, warm-up 10회, 측정 50회
- GCN/GIN/GAT은 process별 target-edge instance를 동일한 shared 10D head로 처리하며, process별 10개 모델의 순차 학습 비용과 parameter 수를 합산
- Proposed는 10개 process를 함께 학습하는 joint model 1개의 비용 사용

두 배포 형태의 차이는 결과 metadata에 명시한다. Timing 비교 중에는 한 GPU에 다른 작업을 같이 실행하지 않는다.

## 측정값

```text
optimization_time_sec
  optimizer loop의 CUDA-synchronized 누적 시간

end_to_end_training_time_sec
  첫 epoch 시작부터 validation, early stopping, checkpoint 저장/선택,
  best checkpoint reload까지 포함한 wall-clock 시간

end_to_end_time_until_best_sec
  첫 epoch 시작부터 선택된 best checkpoint 시점까지의 시간
```

Startup import, 최초 CSV materialization, scaler fit은 training timer에서 제외한다. Inference는 mean/std/median latency, throughput, allocated/reserved peak GPU memory를 저장하며 각 측정 전후 CUDA synchronize를 수행한다.

## 실행

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python3 scripts/run_computational_efficiency_benchmark.py \
  --base-config configs/experiment/pinn/model_260805.yaml \
  --split-root data/splits/all_processes_full100k_outer5_grouped_60_20_20 \
  --merged-csv data/datasets_v3/process_main_merged.csv \
  --output-root outputs/computational_efficiency_260811 \
  --models gcn gin gat proposed --folds 1 2 3 4 5 \
  --max-epochs 30 --early-stopping-patience 3 --training-batch-size 1 \
  --warmup-runs 10 --inference-runs 50 --inference-batch-size 1 \
  --gpu-id 0 --resume-existing
```

## 산출물

- `computational_efficiency_raw.csv`: fold/model별 원자료
- `computational_efficiency_summary.csv`: 모델별 mean/std 집계
- `performance_cost_tradeoff.csv`: R2 대 시간, latency, memory, parameter plotting용
- Proposed run의 `computational_efficiency_training.json`
- 각 모델의 best epoch, stopped epoch, patience, target mean R2

Smoke cap이나 공유 GPU에서 얻은 timing은 실행 경로 확인용일 뿐 논문 표에 사용하지 않는다.
