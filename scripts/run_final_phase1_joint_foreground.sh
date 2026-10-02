#!/usr/bin/env bash
set -Eeuo pipefail

GPU_ID="${1:?usage: bash scripts/run_final_phase1_joint_foreground.sh GPU_ID}"
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
PROPOSED_ROOT="$FINAL_ROOT/proposed_joint_10d_clean"
BASELINE_ROOT="${BASELINE_OUTPUT_ROOT:-$FINAL_ROOT/baselines}"
BASELINE_REPO="${BASELINE_REPO:-../화학공정Baselines}"

echo "[phase1][joint][1/2] Proposed all-process five-fold on physical GPU $GPU_ID"
python3 -u scripts/run_process_kfold_experiments.py \
  --base-config "$CONFIG" --splits-dir "$JOINT_SPLIT" --merged-csv "$MERGED_CSV" \
  --process-ids 1 2 3 4 5 6 7 8 9 10 --joint-all-processes \
  --only-folds 1 2 3 4 5 --max-epochs 30 \
  --monitor-metric val_target_edge_property_mean_r2 --monitor-mode max \
  --output-root "$PROPOSED_ROOT" --skip-existing --skip-startup-debug

echo "[phase1][joint][2/2] B6/GCN/GIN/GAT multi-process baselines"
exec python3 -u scripts/run_final_baseline_experiments.py \
  --scope multi --models B6 GCN GIN GAT --process-ids 1 2 3 4 5 6 7 8 9 10 \
  --folds 1 2 3 4 5 --baseline-root "$BASELINE_REPO" \
  --output-root "$BASELINE_ROOT" --max-epochs 30 --patience 5 --resume-existing
