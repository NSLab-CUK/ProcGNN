#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <all|heldout-process-id> <gpu-id>" >&2
  exit 2
fi

TARGET="$1"
GPU_ID="$2"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  if [[ ! -x "$PYTHON_BIN" ]] && ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "PYTHON_BIN is not executable or on PATH: $PYTHON_BIN" >&2
    exit 2
  fi
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Python was not found. Activate the training environment first." >&2
  exit 127
fi

CONFIG="${MODEL_260805_CONFIG:-configs/experiment/pinn/model_260805.yaml}"
MERGED_CSV="${MODEL_260805_MERGED_CSV:-data/datasets_v3/process_main_merged.csv}"
SPLIT_ROOT="${MODEL_260805_SPLIT_ROOT:-data/splits/single_process_full_unseen_60_20_20}"
ALL_OUTPUT_ROOT="${MODEL_260805_ALL_OUTPUT_ROOT:-outputs/260805_allproc5fold}"
UNSEEN_OUTPUT_ROOT="${MODEL_260805_UNSEEN_OUTPUT_ROOT:-outputs/260805_singleproc_unseen}"
MAX_EPOCHS="${MODEL_260805_MAX_EPOCHS:-30}"
TRANSFER_EPOCH_SIZE="${MODEL_260805_TRANSFER_EPOCH_SIZE:-1000}"
FOLDS_TEXT="${MODEL_260805_FOLDS:-1 2 3 4 5}"
read -r -a FOLDS <<< "$FOLDS_TEXT"

for required in "$CONFIG" "$MERGED_CSV"; do
  if [[ ! -f "$required" ]]; then
    echo "Required file is missing: $required" >&2
    exit 2
  fi
done

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONIOENCODING="utf-8"
export TQDM_DISABLE="1"

echo "[model-260805] python=$PYTHON_BIN target=$TARGET gpu=$GPU_ID folds=${FOLDS[*]}"

if [[ "$TARGET" == "all" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" scripts/run_process_kfold_experiments.py \
    --base-config "$CONFIG" \
    --output-root "$ALL_OUTPUT_ROOT" \
    --process-ids 1 2 3 4 5 6 7 8 9 10 \
    --joint-all-processes \
    --k-folds 5 \
    --only-folds "${FOLDS[@]}" \
    --max-epochs "$MAX_EPOCHS" \
    --device cuda \
    --monitor-metric val_target_edge_property_mean_r2 \
    --monitor-mode max \
    --skip-existing \
    --skip-startup-debug
  exit 0
fi

if ! [[ "$TARGET" =~ ^([1-9]|10)$ ]]; then
  echo "Held-out process ID must be in 1..10, got: $TARGET" >&2
  exit 2
fi

HELDOUT_LABEL="$(printf 'heldout_P%02d' "$TARGET")"
if [[ ! -f "$SPLIT_ROOT/$HELDOUT_LABEL/source_train.csv" ]] || \
   [[ ! -f "$SPLIT_ROOT/$HELDOUT_LABEL/source_val.csv" ]]; then
  echo "Prebuilt split is missing under: $SPLIT_ROOT/$HELDOUT_LABEL" >&2
  exit 2
fi
for fold in "${FOLDS[@]}"; do
  fold_label="$(printf 'fold_%02d' "$fold")"
  for name in target_train.csv target_val.csv target_test.csv; do
    if [[ ! -f "$SPLIT_ROOT/$HELDOUT_LABEL/$fold_label/$name" ]]; then
      echo "Prebuilt split is missing: $SPLIT_ROOT/$HELDOUT_LABEL/$fold_label/$name" >&2
      exit 2
    fi
  done
done

for mode in pretrain zero_shot transfer; do
  echo "[model-260805] heldout=P$(printf '%02d' "$TARGET") stage=$mode gpu=$GPU_ID"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" \
    scripts/run_single_process_full_unseen_experiments.py \
    --base-config "$CONFIG" \
    --split-root "$SPLIT_ROOT" \
    --merged-csv "$MERGED_CSV" \
    --output-root "$UNSEEN_OUTPUT_ROOT" \
    --heldout-processes "$TARGET" \
    --folds "${FOLDS[@]}" \
    --pretrain-source-fold 1 \
    --mode "$mode" \
    --max-epochs-pretrain "$MAX_EPOCHS" \
    --max-epochs-transfer "$MAX_EPOCHS" \
    --transfer-epoch-sample-size "$TRANSFER_EPOCH_SIZE" \
    --finetune-mode full \
    --monitor-metric val_target_edge_property_mean_r2 \
    --monitor-mode max \
    --gpu-ids "$GPU_ID" \
    --max-parallel 1 \
    --resume-existing \
    --skip-startup-debug
done
