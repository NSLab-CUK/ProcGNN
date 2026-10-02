# Transfer & Data Efficiency (260811)

## 목적

새 공정 데이터가 얼마나 있어야 충분히 적응하는지, 같은 데이터 양에서 9개 공정 사전학습이 scratch 학습보다 얼마나 유리한지 측정한다.

## 데이터 분리

각 held-out process와 fold에서 다음 세 집합을 엄격히 분리한다.

```text
target_train.csv -> ratio subset 구성, 학습, scaler fit 전용
target_val.csv   -> early stopping과 best checkpoint 선택 전용
target_test.csv  -> 선택된 best checkpoint의 최종 평가 전용
```

세 manifest는 `merged_row_index` 기준으로 서로 겹치지 않아야 하며, 겹치면 runner가 학습 전에 실패한다. 5/10/25/50/100% 비율은 `target_train.csv`에만 적용한다. 제외 샘플 반영 후 train pool은 50개 process-fold에서 5,999~6,001개이고 일반적으로 6,000개다. 따라서 대표 sample 수는 300, 600, 1,500, 3,000, 6,000개다.

## Subset과 전처리

한 번 생성한 deterministic permutation의 prefix를 사용하므로 다음 포함 관계가 유지된다.

```text
5% subset < 10% subset < 25% subset < 50% subset < 100% subset
```

같은 process/fold/ratio의 Transfer와 Scratch는 동일 subset manifest와 동일 index SHA-256을 공유한다. 입력 normalizer와 target scaler는 해당 ratio의 train subset만으로 새로 fit한다. validation/test는 scaler 통계에 포함하지 않는다. 각 run의 `scaler_fit_provenance.json`에 fit manifest, row 수, ordered index hash를 저장하며 aggregation 단계가 subset hash와 일치하는지 검증한다.

## 비교 조건

- Transfer: 기존 9-process pretrained `best.pt`에서 전체 parameter를 fine-tuning한다.
- Scratch: 동일 subset에서 random initialization으로 학습한다.
- 두 조건은 initialization을 제외한 optimizer, scheduler, batch size, max epoch, patience, loss, seed, validation/test가 같다.
- 기존 pretrain checkpoint와 zero-shot test 결과는 재사용한다.
- 예전 transfer 결과는 매 epoch 전체 pool에서 2,000개를 다시 뽑은 실험이므로 현재의 고정 unique-data ratio point로 재사용하지 않는다.
- 새 실험은 각 고정 subset 안에서 매 epoch `min(subset size, 1,000)`개를 비복원 추출한다. subset 밖 sample에는 접근하지 않는다.

## 학습 종료 정책

- 최대 학습 epoch는 모든 ratio에서 `20`이다.
- early stopping은 사용하지 않는다. validation은 20 epoch 안에서 best checkpoint를 고르는 데만 사용한다.
- `80,000 optimizer steps`는 정확히 채워야 하는 목표가 아니라 안전 상한이다.
- 따라서 종료 시점은 `20 epoch`와 `80,000 optimizer steps` 중 먼저 도달한 시점이다.
- 작은 ratio를 80,000 step에 맞추려고 같은 sample을 한 epoch 안에서 과도하게 반복하지 않는다.
- 실제 수행한 optimizer step 수와 상한 도달 여부는 결과 CSV에 별도로 저장한다.

## 실행

```bash
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
python3 scripts/run_transfer_data_efficiency_experiments.py \
  --base-config configs/experiment/pinn/model_260811_novol_fast.yaml \
  --split-root data/splits/single_process_full_unseen_60_20_20 \
  --merged-csv data/datasets_v3/process_main_merged.csv \
  --existing-unseen-root outputs/260805_singleproc_unseen \
  --output-root outputs/transfer_data_efficiency_patience3_pilot \
  --max-epochs 20 --total-optimizer-steps 80000 --early-stopping-patience 3 \
  --finetune-mode full --monitor-metric val_target_edge_property_mean_r2 --monitor-mode max \
  --gpu-ids 0 1 2 --max-parallel 3 --resume-existing --skip-startup-debug
```

본 실험은 `--heldout-processes 1 2 3 4 5 6 7 8 9 10 --folds 1 2 3 4 5 --data-ratios 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 --modes transfer`로 실행한다.

## 산출물

- Run metadata: process, fold, mode, ratio, sample 수, pool 수, seed, checkpoint, 시간
- Subset CSV와 SHA-256
- Validation 및 test의 target/property/edge별 metric
- `aggregate/run_results.csv`
- `aggregate/property_results.csv`
- `aggregate/learning_curve_summary.csv`
- `aggregate/transfer_gain_summary.csv`
- `aggregate/adaptation_gain_summary.csv`
- `aggregate/patience_pilot_epoch_history.csv`
- `aggregate/patience_pilot_summary.csv`
