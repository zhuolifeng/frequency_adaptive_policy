#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Arguments are passed directly to train.py.
# GPU_ID is a comma-separated list of local CUDA device IDs.
# The default uses the first available device, including on a multi-GPU allocation.
visible_devices="${CUDA_VISIBLE_DEVICES-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-${visible_devices%%,*}}"
num_processes=1
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    IFS=',' read -ra gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
    num_processes=${#gpu_ids[@]}
fi
num_processes="${NUM_PROCESSES:-$num_processes}"
args=(--num_processes "$num_processes" --main_process_port "${MAIN_PROCESS_PORT:-29500}")
if (( num_processes > 1 )); then
    args+=(--multi_gpu)
fi

exec accelerate launch "${args[@]}" train.py "$@"
