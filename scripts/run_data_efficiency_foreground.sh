#!/usr/bin/env bash
set -Eeuo pipefail

GPU_ID="${1:?usage: bash scripts/run_data_efficiency_foreground.sh GPU_ID PROCESS_ID FOLD...}"
PROCESS_ID="${2:?missing PROCESS_ID}"
shift 2
if (( $# == 0 )); then
  echo "at least one fold is required" >&2
  exit 2
fi
FOLDS=("$@")

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/foreground_gpu_guard.sh
foreground_gpu_guard "$GPU_ID"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

CONFIG="${CONFIG:-configs/experiment/pinn/model_260805_10d_frac1.yaml}"
SPLIT_ROOT="${SPLIT_ROOT:-data/splits/single_process_full_unseen_60_20_20}"
MERGED_CSV="${MERGED_CSV:-data/datasets_v3/process_main_merged.csv}"
FINAL_ROOT="${FINAL_ROOT:-outputs/0819final}"
EXISTING_UNSEEN_ROOT="${EXISTING_UNSEEN_ROOT:-$FINAL_ROOT/proposed_unseen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$FINAL_ROOT/data_efficiency}"

export PYTHONPATH=src
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

echo "[data-efficiency][foreground] process=P$(printf '%02d' "$PROCESS_ID") folds=${FOLDS[*]} physical_gpu=$GPU_ID"
echo "[data-efficiency][foreground] max_epochs=30 optimizer_step_cap=80000 output_root=$OUTPUT_ROOT"

exec python3 -u scripts/run_transfer_data_efficiency_experiments.py \
  --base-config "$CONFIG" \
  --split-root "$SPLIT_ROOT" \
  --merged-csv "$MERGED_CSV" \
  --existing-unseen-root "$EXISTING_UNSEEN_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --heldout-processes "$PROCESS_ID" \
  --folds "${FOLDS[@]}" \
  --data-ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --modes transfer \
  --finetune-mode full \
  --total-optimizer-steps 80000 \
  --max-epochs 30 \
  --early-stopping-patience 5 \
  --monitor-metric val_target_edge_property_mean_r2 \
  --monitor-mode max \
  --gpu-ids "$GPU_ID" \
  --max-parallel 1 \
  --skip-final-aggregation \
  --resume-existing \
  --skip-startup-debug
