# Ascend HiCache on the qualified r97 release

This work starts at SGLang commit
`bba2139690fc499ce7b39e18bfd4222f2a8051bd`, branch
`release/codex-1031-best-r97-20261102`. Development uses the separate worktree
`/sgl-workspace/sglang-hicache`, branch `opt/codex-1031-hicache`.
The original worktree, release refs, packaged wheels and published image remain
unchanged. No new operator binary is required.

## What changed

Ascend already had HiCache configuration resolution and an all-layer native
transfer operation. The release's host-pool paths no longer matched every
physical device layout.

For MLA, backup entered CUDA pointer-array code before reaching the Ascend
branch. Unified FlashMLA stores latent and RoPE in one contiguous tensor, while
its 512/64-wide K/V views are non-contiguous. The native copy operation computes
pitches from shape products, so copying those views as dense tensors is wrong.
The old host layout also omitted indexer memory from its budget and lacked a
complete storage-page interface.

The Ascend MLA host pool copies a unified buffer as one component, or copies
legacy latent/RoPE components individually. Optional indexer data is another
component. Each host page is one contiguous payload holding all its components,
including every local layer. Its flat page and zero-copy metadata describe the
same bytes. Empty V arguments are five-dimensional because the existing native
operator reads the fifth dimension even when V has zero elements.

For MHA, FIA exposes per-layer list views to graph execution. The transfer path
recovers the original five-dimensional allocation without copying, after
checking shared storage, order and strides. Both equal and unequal K/V widths
use this path. Host page-first layer references now use the correct axis.

Existing transfer streams, start fences and layer completion events are reused.
Ascend's operation enqueues all local layers before recording layer-0 completion;
this preserves correctness while allowing the normal consumer-stream waits.

Service validation also exposed an eager FlashMLA short-context bug: metadata
rounded a 129-token context up to512 while its block table held only256 tokens.
The eager upper bound is now capped by the table capacity, and the same bound
is used for metadata and execution. Graph mode keeps its dynamic `-1` behavior.
This fix changes no attention arithmetic and introduces no tensor-value read.

## Enabling it

Use this worktree's Python source, rather than the published image's source:

```bash
export PYTHONPATH=/sgl-workspace/sglang-hicache/python${PYTHONPATH:+:$PYTHONPATH}
export SGLANG_NPU_PREFILL_EXPANDED_KV=0
```

Add these options to the existing Ascend launch command:

```bash
--enable-hierarchical-cache \
--hicache-ratio 2 \
--hicache-write-policy write_through
```

Ascend resolves `hicache_io_backend=kernel_ascend`; MLA uses
`page_first_kv_split`, MHA uses `page_first_direct`. Unified MLA remains controlled
by the existing `ENABLE_ASCEND_FLASH_MLA` / `SGLANG_NPU_MLA_UNIFIED_KV` switches.
Choose host capacity for all colocated ranks. A fixed `--hicache-size` remains
supported and includes the MLA indexer allocation.

For a server that decodes with unified MLA KV, enable
`ENABLE_ASCEND_FLASH_MLA=1` and retain the qualified FlashMLA OPP/library paths.
Enabling only `SGLANG_NPU_MLA_UNIFIED_KV=1` while decoding through FIA is not a
supported combination: FIA rejects the unified buffer's strided RoPE view.
The layout-only switch remains useful for prefill workers whose decode peer
uses FlashMLA.

The old best-r97 P-worker script explicitly enables expanded prefill KV.
Change that worker environment setting to `0` when enabling HiCache; an outer
export is insufficient if the launch script overrides it. Expanded prefill KV
is a separate derived cache and is not restored by this implementation.

For persistent MLA caching, the verified backend is `file`:

```bash
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/path/to/hicache
# Also add: --hicache-storage-backend file
```

A versioned format fingerprint is automatically added to the storage namespace.
It distinguishes unified/split ordering, dtype, page size, layer range and each
component's dimensions. Equal byte length does not imply equal serialization.
Old cache pages therefore miss safely, while matching formats can reuse pages
across different pool capacities. Configuration objects are copied, and runtime
storage attachment uses the same checks as startup.

Other MLA L3 backends fail before loading their modules. DCP, layer-sharded
pools, packed MTP and HiSparse are outside this implementation's supported
combinations. The existing CUDA/HIP host-pool classes remain selected for their
devices. NPU PD direct-to-host offload is a different transfer ABI and is not
qualified by these tests.

## Validation

Run CPU regressions without automatic accelerator backend loading:

```bash
cd /sgl-workspace/sglang-hicache
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=python python3 test/manual/ascend/test_hicache_mha_cpu.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=python python3 test/registered/unit/mem_cache/test_npu_hicache_mla.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=python python3 test/registered/unit/mem_cache/test_npu_hicache_storage.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=python python3 test/manual/ascend/test_flash_mla_short_table_cpu.py
```

MHA has eight CPU regressions. MLA has nine, including twelve dtype/layout/index
round-trip combinations and exact native-pitch address simulation. File storage
has seven, including real file IO, complete data round trips and format
isolation. Existing DCP host-pool and staged write-back tests contribute 25
passes and one existing skip; this does not imply Ascend DCP support.
Six additional FlashMLA regressions cover short tables, bucket boundaries,
graph behavior and padded output. The original method fails the four short-table
cases; the fixed method passes all six. The CPU total is55 passes and one skip.

Run actual DMA and stream/event tests on one free NPU:

```bash
ASCEND_RT_VISIBLE_DEVICES=7 PYTHONPATH=python python3 test/registered/npu/basic_function/HiCache/test_npu_hicache_transfer.py
ASCEND_RT_VISIBLE_DEVICES=7 PYTHONPATH=python python3 test/registered/npu/basic_function/HiCache/test_npu_hicache_mha_transfer.py
```

Both actual-NPU matrices passed on Ascend950DT: eight MLA combinations and eight
MHA combinations. They cover BF16/FP16, three layers, 128-token pages, permuted
non-adjacent pages, restoration into different slots, untouched-page sentinels
and consumer access through per-layer events. These are correctness tests,
not kernel performance or serving-QPS measurements.

Detailed logs, service-smoke harness/results and ownership/cleanup evidence are
stored outside the repository at
`/mnt/sfs_turbo/hxd/p_kv_shared_tune_test/codex-1031/hicache-20261102`.

The service tests ran in eager mode with graph execution and schedule overlap
disabled; they do not qualify graph/overlap execution or a multi-worker PD
topology. The service smoke test deliberately fills the device cache with other prefixes,
then requests the original synthetic 384-token prompt again. A pass requires
256 cached host tokens, zero cached device tokens, matching cold/load-back
output token IDs, successful requests and complete owned-process cleanup.
The final synthetic model uses the target GLM attention geometry: 20 heads,
latent width512, QK width192+64 and V width256. It has only two random-weight
layers, hidden size320 and vocabulary1024; this is not a production-model
accuracy or throughput evaluation. Logprob differences are recorded separately
from output-token equality.

The split service passed seven requests with host256/device0 on reload and
matching output token985; the cold/reload logprob difference was
`7.62939453125e-6`. The final unified+FlashMLA service, after the short-table fix,
passed complete warmup and seven two-token requests. Cold and host-reloaded
outputs were both `[985, 985]`; logprob differences were
`[7.62939453125e-6, 1.7642974853515625e-5]`. It covers actual decode continuation
after host reload. All owned processes exited and card7 was verified idle.
Raw DMA checks are byte-exact; these service logprobs are not claimed bit-exact.

Final directories under the evidence root are
`service-host-evict-split-20261102-182747` and
`service-host-evict-unified-flashmla-out2-20261102-184143`.

An earlier DeepSeek-style small model with QK192/V128 failed in the inherited
ATB RingMLA prefix-attention path. A control test with HiCache disabled and an
ordinary device-prefix hit failed at the same operation. That attention-kernel
limitation was not changed here. It must not be represented as a successful
service test or confused with the passing raw-DMA matrices.

## Published image provenance

`sglang:codex-1031-best-r97-20261102`, image ID
`sha256:0e87d41c14db6b258376dafc160ec71345bb636abe5ffeb017f9d3e8f4ddb8f1`,
was built before the release branch was created. Its labels record framework
baseline `73a2cbb65555d303c1836411edfe7dd9400eaa82` and operator source
`79ab3b050e7a9109eea593ce4d4bf491f1e66df6`, plus the packaged working-tree
changes subsequently committed separately. The release freeze verified the
same 4679 tracked framework runtime paths and one generated version file.
Thus the image's runtime contents match the release freeze, but it was not
built by checking out that release branch. It contains none of these new
HiCache commits. Its internal scripts still use the qualified client port9020;
the separately delivered `r97_script` uses client port9001.

The image's earlier 10.7342 QPS/card and mean E2E1445.54ms are not HiCache
performance results. No HiCache throughput improvement is claimed here.
