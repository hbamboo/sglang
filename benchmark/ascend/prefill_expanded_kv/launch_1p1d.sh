#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?baseline or candidate}"
shift
[[ "$MODE" == baseline || "$MODE" == candidate ]]
BASELINE_ROOT="${BASELINE_ROOT:-/sgl-workspace/sglang-p-kv-baseline}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-/sgl-workspace/sglang-p-kv-opt}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/sgl-workspace/p-kv-opt-artifacts}"
MODEL="${MODEL:-/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8}"
LAUNCHER="${LAUNCHER:-/mnt/sfs_turbo/hxd/script/chat/start_single_node_pd_perf.sh}"
PREFILL_DEVICE="${PREFILL_DEVICE:-6}"
DECODE_DEVICE="${DECODE_DEVICE:-7}"
CASE_SUFFIX="${CASE_SUFFIX:-$(date +%Y%m%d-%H%M%S-%N)}"

export PYTHONPATH="${PYTHONPATH:-}"
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /mnt/sfs_turbo/hxd/pkgs/flash_mla_delivery/env.sh
export PYTHONPATH="${BASELINE_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_NPU_USE_MLAPO=0
export SGLANG_NPU_PREFILL_EXPANDED_KV=0
PREFILL_ROOT="$BASELINE_ROOT"
EXPANDED=0
if [[ "$MODE" == candidate ]]; then
    PREFILL_ROOT="$CANDIDATE_ROOT"
    EXPANDED=1
fi

exec bash "$LAUNCHER" \
    --case-name "p-kv-opt-${MODE}-${CASE_SUFFIX}" \
    --runs-root "${ARTIFACT_ROOT}/stacks" \
    --model-path "$MODEL" --served-model-name chat --quantization modelslim \
    --prefill-groups "$PREFILL_DEVICE" --decode-groups "$DECODE_DEVICE" \
    --prefill-tp-size 1 --decode-tp-size 1 \
    --max-model-len 33024 \
    --prefill-mem-fraction-static 0.9 --decode-mem-fraction-static 0.9 \
    --prefill-max-running-requests 64 --decode-max-running-requests 64 \
    --cuda-graph-max-bs-decode 64 \
    --prefill-chunked-prefill-size 32768 --prefill-max-prefill-tokens 32768 \
    --prefill-schedule-policy fcfs --decode-schedule-policy fcfs \
    --disable-overlap-schedule \
    --prefill-port-base 23000 --prefill-bootstrap-port-base 23001 \
    --decode-port-base 24000 --router-port 19082 \
    --prefill-distributed-port-base 29660 --decode-distributed-port-base 29670 \
    --store-port 29960 --prometheus-port 29060 \
    --router-policy round_robin --router-request-timeout-secs 300 \
    --env ENABLE_ASCEND_FLASH_MLA=1 --env ASCEND_USE_FIA=1 \
    --env SGLANG_NPU_USE_MLAPO=0 --env SGLANG_NPU_FUSED_ADD_RMSNORM_QUANT=1 \
    --prefill-env "PYTHONPATH=${PREFILL_ROOT}/python:${PYTHONPATH}" \
    --prefill-env "SGLANG_NPU_PREFILL_EXPANDED_KV=${EXPANDED}" \
    --prefill-env "SGLANG_TORCH_PROFILER_DIR=${ARTIFACT_ROOT}/profiles/${MODE}" \
    "$@"
