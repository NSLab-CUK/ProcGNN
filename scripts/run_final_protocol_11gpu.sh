#!/usr/bin/env bash
set -Eeuo pipefail

# Final 10D paper protocol for one shared host exposing GPU IDs 0..10.
# Run inside tmux/screen. Every stage is resumable and writes a separate log.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH=src
export PYTHONIOENCODING=utf-8
export TQDM_DISABLE=1

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[gpu-plan] ignoring inherited CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  unset CUDA_VISIBLE_DEVICES
fi

CONFIG=configs/experiment/pinn/model_260805_10d_frac1.yaml
JOINT_SPLIT=data/splits/all_processes_full100k_outer5_grouped_60_20_20
UNSEEN_SPLIT=data/splits/single_process_full_unseen_60_20_20
MERGED=data/datasets_v3/process_main_merged.csv
FINAL="${FINAL_ROOT:-outputs/0819final}"
BASELINES="${BASELINE_OUTPUT_ROOT:-$FINAL/baselines}"
BASELINE_REPO="${BASELINE_REPO:-../화학공정Baselines}"
PROPOSED="$FINAL/proposed_joint_10d_clean"
UNSEEN="$FINAL/proposed_unseen"
SENS="$FINAL/sensitivity_10d_clean"
DATA_EFF="$FINAL/data_efficiency"
LOG_ROOT="$FINAL/logs_11gpu"
mkdir -p "$LOG_ROOT"

PIDS=()
wait_all() {
  local failed=0 pid
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  PIDS=()
  if (( failed != 0 )); then
    echo "[final-11gpu] at least one worker failed; inspect $LOG_ROOT" >&2
    exit 1
  fi
}

single_process_worker() {
  local gpu="$1" process="$2" shift_by order
  local models=(M1 M2 M3 B1 B2 B3 B4 B5 B6 B7 G1 G2 G3)
  shift_by=$(( (process - 1) % ${#models[@]} ))
  order=("${models[@]:shift_by}" "${models[@]:0:shift_by}")
  # Dependency-local sequence: pretrain once, then five zero-shot evaluations.
  python3 scripts/run_single_process_full_unseen_experiments.py \
    --base-config "$CONFIG" --split-root "$UNSEEN_SPLIT" --merged-csv "$MERGED" \
    --output-root "$UNSEEN" --heldout-processes "$process" --folds 1 2 3 4 5 \
    --pretrain-source-fold 1 --mode pretrain --max-epochs-pretrain 30 \
    --monitor-metric val_target_edge_property_mean_r2 --monitor-mode max --gpu-ids "$gpu" \
    --max-parallel 1 --resume-existing --skip-startup-debug
  python3 scripts/run_single_process_full_unseen_experiments.py \
    --base-config "$CONFIG" --split-root "$UNSEEN_SPLIT" --merged-csv "$MERGED" \
    --output-root "$UNSEEN" --heldout-processes "$process" --folds 1 2 3 4 5 \
    --pretrain-source-fold 1 --mode zero_shot --gpu-ids "$gpu" --max-parallel 1 \
    --resume-existing --skip-startup-debug

  # Rotate model order by process so CPU-heavy and GPU-heavy baselines are staggered.
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/run_final_baseline_experiments.py \
    --scope single --models "${order[@]}" --process-ids "$process" \
    --folds 1 2 3 4 5 --baseline-root "$BASELINE_REPO" \
    --output-root "$BASELINES" --max-epochs 30 --patience 5 --resume-existing
}

joint_worker() {
  export CUDA_VISIBLE_DEVICES=10
  python3 scripts/run_process_kfold_experiments.py \
    --base-config "$CONFIG" --splits-dir "$JOINT_SPLIT" --merged-csv "$MERGED" \
    --process-ids 1 2 3 4 5 6 7 8 9 10 --joint-all-processes \
    --only-folds 1 2 3 4 5 --max-epochs 30 \
    --monitor-metric val_target_edge_property_mean_r2 --monitor-mode max \
    --output-root "$PROPOSED" --skip-existing --skip-startup-debug
  python3 scripts/run_final_baseline_experiments.py \
    --scope multi --models B6 GCN GIN GAT --process-ids 1 2 3 4 5 6 7 8 9 10 \
    --folds 1 2 3 4 5 --baseline-root "$BASELINE_REPO" \
    --output-root "$BASELINES" --max-epochs 30 --patience 5 --resume-existing
}

echo "[final-11gpu] preflight: materialize shared per-process split views"
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_final_baseline_experiments.py \
  --scope multi --models B6 --process-ids 1 2 3 4 5 6 7 8 9 10 \
  --folds 1 2 3 4 5 --baseline-root "$BASELINE_REPO" \
  --output-root "$BASELINES" --max-epochs 30 --patience 5 \
  --resume-existing --dry-run >"$LOG_ROOT/preflight_baseline_views.log" 2>&1

echo "[final-11gpu] phase 1: baselines, Proposed, pretrain, zero-shot"
for process in $(seq 1 10); do
  gpu=$((process - 1))
  single_process_worker "$gpu" "$process" >"$LOG_ROOT/phase1_gpu${gpu}_P$(printf '%02d' "$process").log" 2>&1 &
  PIDS+=("$!")
done
joint_worker >"$LOG_ROOT/phase1_gpu10_joint.log" 2>&1 &
PIDS+=("$!")
wait_all

# Rebuild the shared unseen registry after concurrent process workers finish.
python3 scripts/run_single_process_full_unseen_experiments.py \
  --base-config "$CONFIG" --split-root "$UNSEEN_SPLIT" --merged-csv "$MERGED" \
  --output-root "$UNSEEN" --heldout-processes 1 2 3 4 5 6 7 8 9 10 \
  --folds 1 2 3 4 5 --pretrain-source-fold 1 --mode aggregate \
  --gpu-ids 0 --max-parallel 1 --resume-existing --skip-startup-debug

sensitivity_fold_worker() {
  local gpu="$1" fold="$2"
  export CUDA_VISIBLE_DEVICES="$gpu"
  python3 scripts/run_final_sensitivity_experiments.py \
    --mode all --folds "$fold" --base-config "$CONFIG" \
    --default-result-root "$PROPOSED" --output-root "$SENS" \
    --max-epochs 30 --resume-existing
}

shap_worker() {
  local gpu="$1"; shift
  export CUDA_VISIBLE_DEVICES="$gpu"
  python3 scripts/run_final_shap_experiments.py \
    --proposed-root "$PROPOSED" --split-root "$JOINT_SPLIT" \
    --output-root "$FINAL/explainability_shap" --process-ids "$@" --fold 1 \
    --background-size 32 --explain-samples 16 --device cuda \
    --require-flowsheet --resume-existing
}

echo "[final-11gpu] phase 2: sensitivity and SHAP"
# Generate shared sensitivity configs once before fold workers read them.
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_final_sensitivity_experiments.py \
  --mode all --folds 1 2 3 4 5 --base-config "$CONFIG" \
  --default-result-root "$PROPOSED" --output-root "$SENS" \
  --max-epochs 30 --resume-existing --dry-run >"$LOG_ROOT/phase2_preflight.log" 2>&1
for fold in $(seq 1 5); do
  gpu=$((fold - 1))
  sensitivity_fold_worker "$gpu" "$fold" >"$LOG_ROOT/phase2_gpu${gpu}_sensitivity_f${fold}.log" 2>&1 &
  PIDS+=("$!")
done
shap_worker 5 1 2 3 4 5 >"$LOG_ROOT/phase2_gpu5_shap_P01_P05.log" 2>&1 & PIDS+=("$!")
shap_worker 6 6 7 8 9 10 >"$LOG_ROOT/phase2_gpu6_shap_P06_P10.log" 2>&1 & PIDS+=("$!")
wait_all

# Restore complete plan metadata after fold/process-partitioned workers.
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_final_sensitivity_experiments.py \
  --mode all --folds 1 2 3 4 5 --base-config "$CONFIG" \
  --default-result-root "$PROPOSED" --output-root "$SENS" \
  --max-epochs 30 --resume-existing --dry-run >"$LOG_ROOT/phase2_sensitivity_plan.log" 2>&1
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_final_shap_experiments.py \
  --proposed-root "$PROPOSED" --split-root "$JOINT_SPLIT" \
  --output-root "$FINAL/explainability_shap" --process-ids 1 2 3 4 5 6 7 8 9 10 \
  --fold 1 --background-size 32 --explain-samples 16 --device cuda \
  --resume-existing --dry-run >"$LOG_ROOT/phase2_shap_plan.log" 2>&1

data_efficiency_run() {
  local gpu="$1" process="$2"; shift 2
  python3 scripts/run_transfer_data_efficiency_experiments.py \
    --base-config "$CONFIG" --split-root "$UNSEEN_SPLIT" --merged-csv "$MERGED" \
    --existing-unseen-root "$UNSEEN" --output-root "$DATA_EFF" \
    --heldout-processes "$process" --folds "$@" \
    --data-ratios 0.02 0.04 0.06 0.08 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --modes transfer --finetune-mode full --total-optimizer-steps 80000 \
    --max-epochs 30 --early-stopping-patience 5 \
    --monitor-metric val_target_edge_property_mean_r2 --monitor-mode max --gpu-ids "$gpu" \
    --max-parallel 1 --skip-final-aggregation --resume-existing --skip-startup-debug
}

echo "[final-11gpu] phase 3: prepare all fixed subsets"
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_transfer_data_efficiency_experiments.py \
  --base-config "$CONFIG" --split-root "$UNSEEN_SPLIT" --merged-csv "$MERGED" \
  --existing-unseen-root "$UNSEEN" --output-root "$DATA_EFF" \
  --heldout-processes 1 2 3 4 5 6 7 8 9 10 --folds 1 2 3 4 5 \
    --data-ratios 0.02 0.04 0.06 0.08 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --modes transfer --finetune-mode full --total-optimizer-steps 80000 \
  --max-epochs 30 --early-stopping-patience 5 --gpu-ids 0 \
  --max-parallel 1 --resume-existing --skip-startup-debug --prepare-only \
  >"$LOG_ROOT/phase3_prepare.log" 2>&1

echo "[final-11gpu] phase 3: 650 transfer runs, balanced as 65/52 per GPU"
for process in $(seq 1 6); do
  gpu=$((process - 1))
  data_efficiency_run "$gpu" "$process" 1 2 3 4 5 >"$LOG_ROOT/phase3_gpu${gpu}.log" 2>&1 &
  PIDS+=("$!")
done
( data_efficiency_run 6 7 1 2 3 4 ) >"$LOG_ROOT/phase3_gpu6.log" 2>&1 & PIDS+=("$!")
( data_efficiency_run 7 7 5; data_efficiency_run 7 8 1 2 3 ) >"$LOG_ROOT/phase3_gpu7.log" 2>&1 & PIDS+=("$!")
( data_efficiency_run 8 8 4 5; data_efficiency_run 8 9 1 2 ) >"$LOG_ROOT/phase3_gpu8.log" 2>&1 & PIDS+=("$!")
( data_efficiency_run 9 9 3 4 5; data_efficiency_run 9 10 1 ) >"$LOG_ROOT/phase3_gpu9.log" 2>&1 & PIDS+=("$!")
( data_efficiency_run 10 10 2 3 4 5 ) >"$LOG_ROOT/phase3_gpu10.log" 2>&1 & PIDS+=("$!")
wait_all

echo "[final-11gpu] final aggregation"
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_transfer_data_efficiency_experiments.py \
  --base-config "$CONFIG" --split-root "$UNSEEN_SPLIT" --merged-csv "$MERGED" \
  --existing-unseen-root "$UNSEEN" --output-root "$DATA_EFF" \
  --heldout-processes 1 2 3 4 5 6 7 8 9 10 --folds 1 2 3 4 5 \
    --data-ratios 0.02 0.04 0.06 0.08 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
  --modes transfer --finetune-mode full --total-optimizer-steps 80000 \
  --max-epochs 30 --early-stopping-patience 5 --gpu-ids 0 \
  --max-parallel 1 --resume-existing --skip-startup-debug --aggregate-only
python3 scripts/aggregate_final_efficiency.py \
  --baseline-root "$BASELINES" --proposed-root "$PROPOSED" \
  --output-root "$FINAL/computational_efficiency"
python3 scripts/aggregate_final_experiments.py \
  --root "$FINAL" --baseline-root "$BASELINES"
python3 scripts/final_experiment_status.py \
  --output-root "$FINAL" --baseline-root "$BASELINES"
echo "[final-11gpu] complete"
