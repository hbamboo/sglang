#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PREFILL_DEVICE="${PREFILL_DEVICE:-5;6}"
export DECODE_DEVICE="${DECODE_DEVICE:-7}"

# P keeps paged expanded K/V + shared overlap. The independent layout switch
# retains the same latent transport ABI while both new attention operators
# stay off on P. D uses V3's step scratch, then FlashMLA reads the unified pool.
exec bash "${SCRIPT_DIR}/launch_reserved_1p1d.sh" \
  --decode-distributed-port-base 29700 \
  --prefill-env ENABLE_ASCEND_FLASH_MLA=0 \
  --prefill-env SGLANG_NPU_MLA_UNIFIED_KV=1 \
  --prefill-env SGLANG_NPU_USE_MLAPO=0 \
  --prefill-env SGLANG_NPU_USE_MLAPO_V3=0 \
  --prefill-env SGLANG_USE_FIA_NZ=0 \
  --decode-env ENABLE_ASCEND_FLASH_MLA=1 \
  --decode-env SGLANG_NPU_USE_MLAPO=0 \
  --decode-env SGLANG_NPU_USE_MLAPO_V3=1 \
  --decode-env SGLANG_USE_FIA_NZ=0 \
  "$@"
