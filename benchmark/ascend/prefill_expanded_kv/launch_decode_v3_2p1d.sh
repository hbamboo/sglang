#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PREFILL_DEVICE="${PREFILL_DEVICE:-5;6}"
export DECODE_DEVICE="${DECODE_DEVICE:-7}"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
# The packed Query API and its compiled capability must come from one package.
DECODE_KERNEL_PYTHONPATH="${DECODE_KERNEL_PYTHONPATH:-/mnt/paas/hxdnpu/sgl-kernel-npu/python/sgl_kernel_npu}"


# P keeps paged expanded K/V + shared overlap. The independent layout switch
# retains the same latent transport ABI while both new attention operators
# stay off on P. D writes a packed Query that FlashMLA consumes without cat.
# The existing DIRECT_CACHE switch independently controls latent KV landing.
exec bash "${SCRIPT_DIR}/launch_reserved_1p1d.sh" \
  --decode-distributed-port-base 29700 \
  --prefill-env ENABLE_ASCEND_FLASH_MLA=0 \
  --prefill-env SGLANG_NPU_MLA_UNIFIED_KV=1 \
  --prefill-env SGLANG_NPU_DISABLE_YARN_KV_SINK=0 \
  --prefill-env SGLANG_NPU_USE_MLAPO=0 \
  --prefill-env SGLANG_NPU_USE_MLAPO_V3=0 \
  --prefill-env SGLANG_USE_FIA_NZ=0 \
  --decode-env "PYTHONPATH=${DECODE_KERNEL_PYTHONPATH}:${CANDIDATE_ROOT:-$REPO_ROOT}/python:/opt/flash_mla_pylibs:${PYTHONPATH:-}" \
  --decode-env SGLANG_NPU_MLAPO_V3_PACKED_QUERY=1 \
  --decode-env MLA_PROLOG_V3_DECODE_TUNE=1 \
  --decode-env ENABLE_ASCEND_FLASH_MLA=1 \
  --decode-env SGLANG_NPU_USE_MLAPO=0 \
  --decode-env SGLANG_NPU_USE_MLAPO_V3=1 \
  --decode-env SGLANG_USE_FIA_NZ=0 \
  "$@"
