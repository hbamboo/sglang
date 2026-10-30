# opt/p_kv_shared_tune: P/D environment

Pushed revision: b8323f7e8651266058bdb6eb19715cda1f66bc33.

Load the existing FlashMLA delivery environment first and use the target checkout:

```bash
source /mnt/sfs_turbo/hxd/pkgs/flash_mla_delivery/env.sh
export PYTHONPATH="/sgl-workspace/sglang-p-kv-opt/python:${PYTHONPATH:-}"
```

Common settings on both roles:

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_USE_FIA=1
export SGLANG_NPU_FUSED_ADD_RMSNORM_QUANT=1
export SGLANG_USE_FIA_NZ=0
export SGLANG_NPU_DISABLE_YARN_KV_SINK=0
export SGLANG_NPU_SHARED_EXPERT_OVERLAP_ANNOTATE=0
export SGLANG_NPU_SHARED_EXPERT_OVERLAP_PREQUANT=0
export SGLANG_NPU_SHARED_EXPERT_OVERLAP_CUBE_CORES=-1
export SGLANG_NPU_SHARED_EXPERT_OVERLAP_VECTOR_CORES=-1
export SGLANG_NPU_MLAPO_V3_DEBUG=0
```

Each P instance:

```bash
export ENABLE_ASCEND_FLASH_MLA=0
export SGLANG_NPU_MLA_UNIFIED_KV=1
export SGLANG_NPU_USE_MLAPO=0
export SGLANG_NPU_USE_MLAPO_V3=0
export SGLANG_NPU_PREFILL_EXPANDED_KV=1
export SGLANG_NPU_SHARED_EXPERT_OVERLAP=1
export SGLANG_NPU_SHARED_EXPERT_OVERLAP_MIN_TOKENS=8192
export SGLANG_NPU_PREFILL_TRANSIENT_RESERVE_MB=5120
```

D instance:

```bash
export ENABLE_ASCEND_FLASH_MLA=1
export SGLANG_NPU_MLA_UNIFIED_KV=1
export SGLANG_NPU_USE_MLAPO=0
export SGLANG_NPU_USE_MLAPO_V3=1
export SGLANG_NPU_PREFILL_EXPANDED_KV=0
export SGLANG_NPU_SHARED_EXPERT_OVERLAP=0
export SGLANG_NPU_PREFILL_TRANSIENT_RESERVE_MB=0
```

UNIFIED_KV selects the transfer layout, not an attention operator. V3 implies the
prolog route without enabling legacy MLAPO. Keep DISABLE_YARN_KV_SINK=0 to retain
the tested P fused sink. The 5120 MiB P reserve is the measured setting for the
current model and workload, not a universal value.

Use ASCEND_RT_VISIBLE_DEVICES=5 and =6 for the two separate P processes, and =7
for D. Both roles keep --disable-shared-experts-fusion; tested TP size is 1.

The pushed launcher configures these role switches and the reserved P setup:

```bash
bash benchmark/ascend/prefill_expanded_kv/launch_decode_v3_2p1d.sh
```

Retained verification before the connection outage: 36 hardware tests passed;
final 2P1D main comparison completed successfully; P first-token IDs and logprobs
matched the baseline in all 10 quality cases, including 32K context; profiles
show no MLA operators on either P, with 1645 V3 and 1645 FlashMla calls on D.
Rate10 TTFT baseline: 283.937/360.984 ms, final: 281.681/359.244 ms.

All four accepted variants completed: 5200 measured requests succeeded, with no
request-time KV transfer errors. At rate10, final TTFT was320.463ms, baseline
322.460ms (repeat325.910ms), and no-expanded control566.626ms. Both P workers
retained the expanded-cache optimization. Resource-release audit confirmed no
remaining task processes or ports and no running processes on devices5,6,7.
