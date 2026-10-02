#!/usr/bin/env bash
set -Eeuo pipefail

GPU_ID="${1:?usage: bash scripts/run_final_phase2_sensitivity_foreground.sh GPU_ID FOLD...}"
shift
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
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

echo "[phase2][sensitivity] folds=${FOLDS[*]} physical_gpu=$GPU_ID"
FINAL_ROOT="${FINAL_ROOT:-outputs/0819final}"
exec python3 -u scripts/run_final_sensitivity_experiments.py \
  --mode all --folds "${FOLDS[@]}" \
  --base-config configs/experiment/pinn/model_260805_10d_frac1.yaml \
  --default-result-root "$FINAL_ROOT/proposed_joint_10d_clean" \
  --output-root "$FINAL_ROOT/sensitivity_10d_clean" \
  --max-epochs 30 --resume-existing
