#!/usr/bin/env bash
set -Eeuo pipefail

GPU_ID="${1:?usage: bash scripts/run_final_phase1_proposed_single_foreground.sh GPU_ID PROCESS_ID [FOLD ...]}"
PROCESS_ID="${2:?missing PROCESS_ID}"
shift 2
if (( $# > 0 )); then
  FOLDS=("$@")
else
  FOLDS=(1 2 3 4 5)
fi

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
JOINT_SPLIT=data/splits/all_processes_full100k_outer5_grouped_60_20_20
MERGED_CSV=data/datasets_v3/process_main_merged.csv
FINAL_ROOT="${FINAL_ROOT:-outputs/0819final}"
OUTPUT_ROOT="$FINAL_ROOT/proposed_single_process"

echo "[phase1-proposed-single] process=P$(printf '%02d' "$PROCESS_ID") folds=${FOLDS[*]} physical_gpu=$GPU_ID"
python3 -u scripts/run_process_kfold_experiments.py \
  --base-config "$CONFIG" --splits-dir "$JOINT_SPLIT" --merged-csv "$MERGED_CSV" \
  --process-ids "$PROCESS_ID" --only-folds "${FOLDS[@]}" --max-epochs 30 \
  --monitor-metric val_target_edge_property_mean_r2 --monitor-mode max \
  --output-root "$OUTPUT_ROOT" --skip-existing --skip-startup-debug
