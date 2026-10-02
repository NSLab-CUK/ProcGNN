#!/usr/bin/env bash
set -Eeuo pipefail

SERVER_ID="${1:?usage: bash scripts/run_baseline_rerun_0908.sh SERVER_ID}"
if [[ ! "$SERVER_ID" =~ ^[123]$ ]]; then
  echo "SERVER_ID must be 1, 2, or 3" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

OUTPUT_ROOT="${BASELINE_OUTPUT_ROOT:-outputs/0819final/baselines}"
LOG_ROOT="outputs/0819final/logs_baseline_rerun_0908_server${SERVER_ID}"
MODELS=(M1 M2 M3 B1 B2 B3 B4 B5 B6 B7 G1 G2 G3)
FOLDS=(1 2 3 4 5)

case "$SERVER_ID" in
  1) PROCESSES=(1 2 3 4); GPUS=(0 1 2 3); DEFAULT_BASELINE_REPO=/home/nsuser/화학공정Baselines ;;
  2) PROCESSES=(5 6 7 8); GPUS=(0 1 2 3); DEFAULT_BASELINE_REPO=/home/jw/화학공정Baselines ;;
  3) PROCESSES=(9 10); GPUS=(0 1); DEFAULT_BASELINE_REPO=/home/oj/화학공정Baselines ;;
esac
BASELINE_REPO="${BASELINE_REPO:-$DEFAULT_BASELINE_REPO}"
if [[ ! -f "$BASELINE_REPO/src/process_graph/baselines/dataset.py" ]]; then
  echo "baseline repository not found or incomplete: $BASELINE_REPO" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT"

preflight_single() {
  CUDA_VISIBLE_DEVICES=0 python3 -u scripts/run_final_baseline_experiments.py \
    --scope single --models "${MODELS[@]}" --process-ids "${PROCESSES[@]}" \
    --folds "${FOLDS[@]}" --baseline-root "$BASELINE_REPO" \
    --output-root "$OUTPUT_ROOT" --input-policy core_controls \
    --max-epochs 30 --patience 5 --dry-run
}

preflight_multi() {
  CUDA_VISIBLE_DEVICES=2 python3 -u scripts/run_final_baseline_experiments.py \
    --scope multi --models B6 GCN GIN GAT \
    --process-ids 1 2 3 4 5 6 7 8 9 10 --folds "${FOLDS[@]}" \
    --baseline-root "$BASELINE_REPO" --output-root "$OUTPUT_ROOT" \
    --input-policy core_controls --max-epochs 30 --patience 5 --batch-size 32 \
    --dry-run
}

run_single_process() {
  local gpu="$1" process="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python3 -u scripts/run_final_baseline_experiments.py \
    --scope single --models "${MODELS[@]}" --process-ids "$process" \
    --folds "${FOLDS[@]}" --baseline-root "$BASELINE_REPO" \
    --output-root "$OUTPUT_ROOT" --input-policy core_controls \
    --max-epochs 30 --patience 5 --resume-existing
}

run_multi() {
  CUDA_VISIBLE_DEVICES=2 python3 -u scripts/run_final_baseline_experiments.py \
    --scope multi --models B6 GCN GIN GAT \
    --process-ids 1 2 3 4 5 6 7 8 9 10 --folds "${FOLDS[@]}" \
    --baseline-root "$BASELINE_REPO" --output-root "$OUTPUT_ROOT" \
    --input-policy core_controls --max-epochs 30 --patience 5 --batch-size 32 \
    --resume-existing
}

preflight_single >"$LOG_ROOT/preflight_single.log" 2>&1
if [[ "$SERVER_ID" == 3 ]]; then
  preflight_multi >"$LOG_ROOT/preflight_multi.log" 2>&1
fi

pids=()
for index in "${!PROCESSES[@]}"; do
  process="${PROCESSES[$index]}"
  gpu="${GPUS[$index]}"
  run_single_process "$gpu" "$process" \
    >"$LOG_ROOT/gpu${gpu}_P$(printf '%02d' "$process").log" 2>&1 &
  pids+=("$!")
  echo "[started] server=$SERVER_ID gpu=$gpu process=P$(printf '%02d' "$process") pid=$!"
done

if [[ "$SERVER_ID" == 3 ]]; then
  run_multi >"$LOG_ROOT/gpu2_multi.log" 2>&1 &
  pids+=("$!")
  echo "[started] server=3 gpu=2 multi-baselines pid=$!"
fi

failed=0
for worker_pid in "${pids[@]}"; do
  if ! wait "$worker_pid"; then
    failed=1
  fi
done

if (( failed != 0 )); then
  echo "One or more baseline workers failed. Inspect $LOG_ROOT and rerun this command." >&2
  exit 1
fi

# Restore a complete server-local plan after per-process workers have finished.
preflight_single >"$LOG_ROOT/final_single_plan.log" 2>&1
if [[ "$SERVER_ID" == 3 ]]; then
  preflight_multi >"$LOG_ROOT/final_multi_plan.log" 2>&1
fi

echo "[completed] server=$SERVER_ID baseline rerun; output=$OUTPUT_ROOT"
