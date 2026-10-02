#!/usr/bin/env bash
set -Eeuo pipefail

GPU_ID="${1:?usage: bash scripts/run_final_phase2_shap_foreground.sh GPU_ID PROCESS...}"
shift
if (( $# == 0 )); then
  echo "at least one process is required" >&2
  exit 2
fi
PROCESSES=("$@")

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/foreground_gpu_guard.sh
foreground_gpu_guard "$GPU_ID"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

echo "[phase2][SHAP] processes=${PROCESSES[*]} physical_gpu=$GPU_ID"
FINAL_ROOT="${FINAL_ROOT:-outputs/0819final}"
exec python3 -u scripts/run_final_shap_experiments.py \
  --proposed-root "$FINAL_ROOT/proposed_joint_10d_clean" \
  --split-root data/splits/all_processes_full100k_outer5_grouped_60_20_20 \
  --output-root "$FINAL_ROOT/explainability_shap" \
  --process-ids "${PROCESSES[@]}" --fold 1 \
  --background-size 32 --explain-samples 16 --device cuda \
  --require-flowsheet --resume-existing
