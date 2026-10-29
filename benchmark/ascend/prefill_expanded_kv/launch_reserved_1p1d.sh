#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
export BASELINE_ROOT="${BASELINE_ROOT:-$REPO_ROOT}"
export CANDIDATE_ROOT="${CANDIDATE_ROOT:-$REPO_ROOT}"
exec bash "${SCRIPT_DIR}/launch_1p1d.sh" candidate \
  --env PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  --env SGLANG_NPU_SHARED_EXPERT_OVERLAP_PREQUANT=0 \
  --env SGLANG_NPU_SHARED_EXPERT_OVERLAP_CUBE_CORES=-1 \
  --env SGLANG_NPU_SHARED_EXPERT_OVERLAP_VECTOR_CORES=-1 \
  --prefill-env SGLANG_NPU_SHARED_EXPERT_OVERLAP=1 \
  --prefill-env SGLANG_NPU_SHARED_EXPERT_OVERLAP_MIN_TOKENS=8192 \
  --prefill-env SGLANG_NPU_SHARED_EXPERT_OVERLAP_ANNOTATE=0 \
  --prefill-env "SGLANG_NPU_PREFILL_TRANSIENT_RESERVE_MB=${RESERVE_MB:-5120}" \
  --decode-env SGLANG_NPU_SHARED_EXPERT_OVERLAP=0 \
  "$@"
