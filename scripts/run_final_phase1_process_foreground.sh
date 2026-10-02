#!/usr/bin/env bash
set -Eeuo pipefail

GPU_ID="${1:?usage: bash scripts/run_final_phase1_process_foreground.sh GPU_ID PROCESS_ID}"
PROCESS_ID="${2:?missing PROCESS_ID}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/foreground_gpu_guard.sh
foreground_gpu_guard "$GPU_ID"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

CONFIG=configs/experiment/pinn/model_260805_10d_frac1.yaml
SPLIT_ROOT=data/splits/single_process_full_unseen_60_20_20
MERGED_CSV=data/datasets_v3/process_main_merged.csv
FINAL_ROOT="${FINAL_ROOT:-outputs/0819final}"
UNSEEN_ROOT="$FINAL_ROOT/proposed_unseen"
BASELINE_ROOT="${BASELINE_OUTPUT_ROOT:-$FINAL_ROOT/baselines}"
BASELINE_REPO="${BASELINE_REPO:-../화학공정Baselines}"
MODELS=(M1 M2 M3 B1 B2 B3 B4 B5 B6 B7 G1 G2 G3)
SHIFT_BY=$(( (PROCESS_ID - 1) % ${#MODELS[@]} ))
MODEL_ORDER=("${MODELS[@]:SHIFT_BY}" "${MODELS[@]:0:SHIFT_BY}")

echo "[phase1][P$(printf '%02d' "$PROCESS_ID")][1/3] 9-process pretrain on physical GPU $GPU_ID"
python3 -u scripts/run_single_process_full_unseen_experiments.py \
  --base-config "$CONFIG" --split-root "$SPLIT_ROOT" --merged-csv "$MERGED_CSV" \
  --output-root "$UNSEEN_ROOT" --heldout-processes "$PROCESS_ID" --folds 1 2 3 4 5 \
  --pretrain-source-fold 1 --mode pretrain --max-epochs-pretrain 30 \
  --monitor-metric val_target_edge_property_mean_r2 --monitor-mode max --gpu-ids "$GPU_ID" \
  --max-parallel 1 --resume-existing --skip-startup-debug

echo "[phase1][P$(printf '%02d' "$PROCESS_ID")][2/3] zero-shot fold 1-5"
python3 -u scripts/run_single_process_full_unseen_experiments.py \
  --base-config "$CONFIG" --split-root "$SPLIT_ROOT" --merged-csv "$MERGED_CSV" \
  --output-root "$UNSEEN_ROOT" --heldout-processes "$PROCESS_ID" --folds 1 2 3 4 5 \
  --pretrain-source-fold 1 --mode zero_shot --gpu-ids "$GPU_ID" --max-parallel 1 \
  --resume-existing --skip-startup-debug

echo "[phase1][P$(printf '%02d' "$PROCESS_ID")][3/3] 13 single-process baselines"
CUDA_VISIBLE_DEVICES="$GPU_ID" exec python3 -u scripts/run_final_baseline_experiments.py \
  --scope single --models "${MODEL_ORDER[@]}" --process-ids "$PROCESS_ID" \
  --folds 1 2 3 4 5 --baseline-root "$BASELINE_REPO" \
  --output-root "$BASELINE_ROOT" --max-epochs 30 --patience 5 --resume-existing
