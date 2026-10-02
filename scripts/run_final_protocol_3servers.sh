#!/usr/bin/env bash
set -Eeuo pipefail

SERVER_ID="${1:?usage: bash scripts/run_final_protocol_3servers.sh SERVER_ID PHASE}"
PHASE="${2:?phase must be 1, 2, or 3}"
if [[ ! "$SERVER_ID" =~ ^[123]$ ]] || [[ ! "$PHASE" =~ ^[123]$ ]]; then
  echo "SERVER_ID and PHASE must be one of 1, 2, 3" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src PYTHONIOENCODING=utf-8 TQDM_DISABLE=1

# This launcher assigns physical GPU IDs itself. An inherited single-GPU mask
# (for example CUDA_VISIBLE_DEVICES=0) would collapse every worker onto GPU 0.
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
LOG_ROOT="$FINAL/logs_server${SERVER_ID}"
mkdir -p "$LOG_ROOT"

case "$SERVER_ID" in
  1) LOCAL_PROCESSES=(1 2 3 4); GPU_COUNT=4 ;;
  2) LOCAL_PROCESSES=(5 6 7 8); GPU_COUNT=4 ;;
  3) LOCAL_PROCESSES=(9 10); GPU_COUNT=3 ;;
esac

if command -v nvidia-smi >/dev/null 2>&1; then
  DETECTED_GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  if (( DETECTED_GPU_COUNT < GPU_COUNT )); then
    echo "server $SERVER_ID requires $GPU_COUNT GPUs (IDs 0..$((GPU_COUNT - 1))), detected $DETECTED_GPU_COUNT" >&2
    exit 2
  fi
fi
echo "[gpu-plan] server=$SERVER_ID phase=$PHASE workers=${#LOCAL_PROCESSES[@]} gpu_ids=0..$((GPU_COUNT - 1))"

PIDS=()
wait_all() {
  local failed=0 pid
  for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  PIDS=()
  (( failed == 0 )) || { echo "worker failure: inspect $LOG_ROOT" >&2; exit 1; }
}

process_worker() {
  local gpu="$1" process="$2" shift_by order
  local models=(M1 M2 M3 B1 B2 B3 B4 B5 B6 B7 G1 G2 G3)
  shift_by=$(( (process - 1) % ${#models[@]} ))
  order=("${models[@]:shift_by}" "${models[@]:0:shift_by}")
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
  CUDA_VISIBLE_DEVICES="$gpu" python3 scripts/run_final_baseline_experiments.py \
    --scope single --models "${order[@]}" --process-ids "$process" \
    --folds 1 2 3 4 5 --baseline-root "$BASELINE_REPO" \
    --output-root "$BASELINES" --max-epochs 30 --patience 5 --resume-existing
}

joint_worker() {
  export CUDA_VISIBLE_DEVICES=2
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

run_phase1() {
  local preflight_processes=("${LOCAL_PROCESSES[@]}")
  [[ "$SERVER_ID" == 3 ]] && preflight_processes=(1 2 3 4 5 6 7 8 9 10)
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_final_baseline_experiments.py \
    --scope single --models M1 --process-ids "${preflight_processes[@]}" \
    --folds 1 2 3 4 5 --baseline-root "$BASELINE_REPO" \
    --output-root "$BASELINES" --max-epochs 30 --patience 5 --resume-existing --dry-run \
    >"$LOG_ROOT/phase1_preflight.log" 2>&1
  local index=0 process gpu
  for process in "${LOCAL_PROCESSES[@]}"; do
    gpu="$index"; index=$((index + 1))
    process_worker "$gpu" "$process" >"$LOG_ROOT/phase1_gpu${gpu}_P$(printf '%02d' "$process").log" 2>&1 &
    PIDS+=("$!")
  done
  if [[ "$SERVER_ID" == 3 ]]; then
    joint_worker >"$LOG_ROOT/phase1_gpu2_joint.log" 2>&1 & PIDS+=("$!")
  fi
  wait_all
  python3 scripts/run_single_process_full_unseen_experiments.py \
    --base-config "$CONFIG" --split-root "$UNSEEN_SPLIT" --merged-csv "$MERGED" \
    --output-root "$UNSEEN" --heldout-processes "${LOCAL_PROCESSES[@]}" \
    --folds 1 2 3 4 5 --pretrain-source-fold 1 --mode aggregate \
    --gpu-ids 0 --max-parallel 1 --resume-existing --skip-startup-debug
}

sensitivity_worker() {
  local gpu="$1"; shift
  export CUDA_VISIBLE_DEVICES="$gpu"
  python3 scripts/run_final_sensitivity_experiments.py \
    --mode all --folds "$@" --base-config "$CONFIG" \
    --default-result-root "$PROPOSED" --output-root "$SENS" \
    --max-epochs 30 --resume-existing
}

run_phase2() {
  if [[ "$SERVER_ID" != 3 ]]; then
    echo "phase 2 is intentionally assigned only to server 3"
    return
  fi
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_final_sensitivity_experiments.py \
    --mode all --folds 1 2 3 4 5 --base-config "$CONFIG" \
    --default-result-root "$PROPOSED" --output-root "$SENS" \
    --max-epochs 30 --resume-existing --dry-run >"$LOG_ROOT/phase2_preflight.log" 2>&1
  sensitivity_worker 0 1 2 >"$LOG_ROOT/phase2_gpu0_sensitivity_f01_f02.log" 2>&1 & PIDS+=("$!")
  sensitivity_worker 1 3 4 >"$LOG_ROOT/phase2_gpu1_sensitivity_f03_f04.log" 2>&1 & PIDS+=("$!")
  (
    sensitivity_worker 2 5
    CUDA_VISIBLE_DEVICES=2 python3 scripts/run_final_shap_experiments.py \
      --proposed-root "$PROPOSED" --split-root "$JOINT_SPLIT" \
      --output-root "$FINAL/explainability_shap" \
      --process-ids 1 2 3 4 5 6 7 8 9 10 --fold 1 \
      --background-size 32 --explain-samples 16 --device cuda \
      --require-flowsheet --resume-existing
  ) >"$LOG_ROOT/phase2_gpu2_sensitivity_f05_shap.log" 2>&1 & PIDS+=("$!")
  wait_all
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_final_sensitivity_experiments.py \
    --mode all --folds 1 2 3 4 5 --base-config "$CONFIG" \
    --default-result-root "$PROPOSED" --output-root "$SENS" \
    --max-epochs 30 --resume-existing --dry-run >"$LOG_ROOT/phase2_final_plan.log" 2>&1
  python3 scripts/aggregate_final_efficiency.py \
    --baseline-root "$BASELINES" --proposed-root "$PROPOSED" \
    --output-root "$FINAL/computational_efficiency"
}

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

run_phase3() {
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_transfer_data_efficiency_experiments.py \
    --base-config "$CONFIG" --split-root "$UNSEEN_SPLIT" --merged-csv "$MERGED" \
    --existing-unseen-root "$UNSEEN" --output-root "$DATA_EFF" \
    --heldout-processes "${LOCAL_PROCESSES[@]}" --folds 1 2 3 4 5 \
    --data-ratios 0.02 0.04 0.06 0.08 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --modes transfer --finetune-mode full --total-optimizer-steps 80000 \
    --max-epochs 30 --early-stopping-patience 5 --gpu-ids 0 \
    --max-parallel 1 --resume-existing --skip-startup-debug --prepare-only \
    >"$LOG_ROOT/phase3_preflight.log" 2>&1
  if [[ "$SERVER_ID" == 1 || "$SERVER_ID" == 2 ]]; then
    local index=0 process
    for process in "${LOCAL_PROCESSES[@]}"; do
      data_efficiency_run "$index" "$process" 1 2 3 4 5 >"$LOG_ROOT/phase3_gpu${index}.log" 2>&1 &
      PIDS+=("$!"); index=$((index + 1))
    done
  else
    data_efficiency_run 0 9 1 2 3 >"$LOG_ROOT/phase3_gpu0.log" 2>&1 & PIDS+=("$!")
    ( data_efficiency_run 1 9 4 5; data_efficiency_run 1 10 1 ) >"$LOG_ROOT/phase3_gpu1.log" 2>&1 & PIDS+=("$!")
    data_efficiency_run 2 10 2 3 4 5 >"$LOG_ROOT/phase3_gpu2.log" 2>&1 & PIDS+=("$!")
  fi
  wait_all
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_transfer_data_efficiency_experiments.py \
    --base-config "$CONFIG" --split-root "$UNSEEN_SPLIT" --merged-csv "$MERGED" \
    --existing-unseen-root "$UNSEEN" --output-root "$DATA_EFF" \
    --heldout-processes "${LOCAL_PROCESSES[@]}" --folds 1 2 3 4 5 \
    --data-ratios 0.02 0.04 0.06 0.08 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --modes transfer --finetune-mode full --total-optimizer-steps 80000 \
    --max-epochs 30 --early-stopping-patience 5 --gpu-ids 0 \
    --max-parallel 1 --resume-existing --skip-startup-debug --aggregate-only
}

case "$PHASE" in
  1) run_phase1 ;;
  2) run_phase2 ;;
  3) run_phase3 ;;
esac
echo "[server $SERVER_ID] phase $PHASE complete"
