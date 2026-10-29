#!/usr/bin/env bash
# Run inside hxd-glm after port 19255 and physical NPU 5 have been released.
set -euo pipefail
variant=${1:?Usage: reproduce_server.sh baseline|optimized perf|quality}
purpose=${2:-perf}
case "$variant" in
  baseline) export SGLANG_NPU_SHARED_EXPERT_OVERLAP=0 ;;
  optimized) export SGLANG_NPU_SHARED_EXPERT_OVERLAP=1 ;;
  *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac
export ASCEND_RT_VISIBLE_DEVICES=5
export ASCEND_USE_FIA=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_NPU_SHARED_EXPERT_OVERLAP_MIN_TOKENS=8192
export SGLANG_NPU_SHARED_EXPERT_OVERLAP_PREQUANT=0
export SGLANG_NPU_SHARED_EXPERT_OVERLAP_CUBE_CORES=-1
export SGLANG_NPU_SHARED_EXPERT_OVERLAP_VECTOR_CORES=-1
case "$purpose" in
  perf) graph=(--disable-cuda-graph) ;;
  quality) graph=(--cuda-graph-backend-prefill disabled
                  --cuda-graph-backend-decode full
                  --cuda-graph-max-bs-decode 32 --cuda-graph-bs-decode 32) ;;
  *) echo "Unknown purpose: $purpose" >&2; exit 2 ;;
esac
cd /sgl-workspace/sglang
exec python3 -m sglang.launch_server \
  --model-path /mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8 \
  --host 127.0.0.1 --port 19255 --device npu --tp 1 \
  --mem-fraction-static 0.80 --context-length 40960 \
  --max-prefill-tokens 32768 --chunked-prefill-size 32768 \
  --max-running-requests 32 --disable-radix-cache \
  --disable-shared-experts-fusion --attention-backend ascend \
  --random-seed 42 "${graph[@]}"
