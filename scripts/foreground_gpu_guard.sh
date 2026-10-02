#!/usr/bin/env bash

foreground_gpu_guard() {
  local physical_gpu="${1:?physical GPU ID required}"
  if [[ ! "$physical_gpu" =~ ^[0-9]+$ ]]; then
    echo "[gpu-guard][error] GPU ID must be a non-negative integer: $physical_gpu" >&2
    return 2
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_line
    gpu_line="$(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader | awk -F', *' -v id="$physical_gpu" '$1 == id {print; exit}')"
    if [[ -z "$gpu_line" ]]; then
      echo "[gpu-guard][error] physical GPU $physical_gpu is not visible" >&2
      nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader >&2
      return 2
    fi
    echo "[gpu-guard] selected physical GPU: $gpu_line"
  fi
  echo "[gpu-guard] child CUDA_VISIBLE_DEVICES will be $physical_gpu; PyTorch will display it as logical cuda:0"
}
