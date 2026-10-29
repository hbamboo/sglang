# Dense MLA prefill expanded KV

`SGLANG_NPU_PREFILL_EXPANDED_KV=1` enables an expanded K/V cache on a dense
Ascend MLA **prefill** worker. The decode worker keeps the existing latent KV.
The flag defaults to off and currently requires ordinary PD prefill, BF16/FP16
KV, equal K/V head dimensions, CP=1, no speculative decoding and no HiCache.
For this base revision, set `ENABLE_ASCEND_FLASH_MLA=1` on both workers to
select the matching unified latent transfer layout. P still computes MHA/FIA;
D uses FlashMLA. The provided launcher sets this consistently.

The prefill pool owns three buffers per layer:

- The original unified 576-wide latent buffer is the only buffer registered
  for PD transfer. Decode receives exactly the existing cache layout.
- Separate contiguous paged K/V buffers hold the already expanded projection
  outputs. New tokens are scattered once. Radix-cache entries reuse the same
  allocator slots in all buffers.

On a prefix hit, paged FIA reads the expanded buffers using the existing block
table. The backend skips historical `kv_b_proj`, latent-page gathers, prefix
flattening metadata and prefix/chunk concatenation. With no prefix, it still
populates the expanded cache for future hits, and computes attention from the
current contiguous K/V outputs.

Pool sizing accounts for both latent and expanded KV, including the dummy
physical page. For the tested 20-head GLM geometry, this is 10,816 BF16 elements
per token/layer versus 576 originally: **18.78 times the cache bytes per token**.
At the same memory fraction, token capacity decreases accordingly. Expanded KV
is deliberately absent from `get_contiguous_buf_infos()`.

There is no expanded-KV-to-latent conversion at the PD boundary. The existing
MHA preparation first computes the normalized latent and RoPE key, writes that
latent cache, and then applies the current-token `kv_b_proj` to produce K/V.
The new path additionally saves those expanded outputs with two scatter writes
per layer. PD transfers the already available latent cache; its per-token payload
and decode layout are unchanged. Latent generation and its cache write already
exist in the baseline. The added expanded-cache writes are part of measured
prefill latency, and client E2E includes both PD transfer and subsequent decode.

The speedup applies to the measured multi-request workload with prefix reuse.
A request with no cached prefix that fits in one prefill chunk has no historical
projection/gather/cat to remove, while still paying the added K/V cache writes.
The concurrency-8 average does not establish an isolated concurrency-1 speedup.

## Source and startup prerequisite

The requested base is `fc79e7caa00cad115dcc4f78f2277b6c17ee1c15`.
It references `envs` in `AscendAttnBackend` without importing it, preventing
both workers from starting. The common prerequisite commit
`bc85c8ef1499140d497d84c8c0cd46920b36d94d` adds that one import. Both measured
variants contain the identical prerequisite; baseline contains no cache change.

## Numerical tests

```bash
ASCEND_RT_VISIBLE_DEVICES=6 P_KV_TEST_DEVICE=npu:0 PYTHONPATH=$PWD/python \
  python3 -m pytest -q -s test/registered/ascend/test_prefill_expanded_kv.py
```

Tests cover physical page permutation, multi-request and shared prefixes,
non-page-aligned lengths, KV length above 2048, zero-query requests, padding,
expanded-cache writes, memory sizing and unchanged latent PD descriptors.
FP32 causal attention is the golden; numerical-error reductions use FP64.

## 1P1D dataset comparison

The final workload is **100 original requests at concurrency 8**, from
`/mnt/sfs_turbo/hxd/CHAT-mxfp8-customer-2000.jsonl`, seed 42. Requests are not
concatenated into one long prompt. Both prefill limits are 32768 **new tokens
per scheduled batch**, not a per-request input length. The client preserves
the dataset's individual input and output lengths.

`launch_1p1d.sh` starts P on physical 6 and D on physical 7 with separate ports.
Set `BASELINE_ROOT`, `CANDIDATE_ROOT`, `MODEL`, `ARTIFACT_ROOT`, `PREFILL_DEVICE`
or `DECODE_DEVICE` to override the example installation paths. D uses the same
baseline code for both variants. Stop the first stack before launching the
second; the launcher refuses occupied ports.

```bash
bash benchmark/ascend/prefill_expanded_kv/launch_1p1d.sh baseline
# Or: launch_1p1d.sh candidate

PYTHONPATH=/sgl-workspace/sglang-p-kv-baseline/python \
python3 benchmark/ascend/prefill_expanded_kv/bench_dataset.py \
  --variant baseline --server-run-dir /path/printed/by/launcher \
  --max-concurrency 8 --num-prompts 100

python3 benchmark/ascend/prefill_expanded_kv/analyze_dataset.py \
  --concurrency 8 --num-prompts 100
```

The wrapper uses the original `sglang.bench_serving` custom-dataset client,
including the same thinking template option as `bench_glm.sh`. It saves all
per-request TTFTs and output lengths. The client performs one warmup request,
then flushes the server cache before the timed run. Batch statistics additionally
include that warmup; the TTFT summary includes only the requested 100 requests.
The wrapper validates successful completion, actual multi-request batches and
the observed 32768-token cap.

Profiling is collected separately from timed runs. `audit_profile.py` checks
the 47-layer current-token projection count per prefill step, historical
projections, latent page gathers and prefix/chunk K/V cats. It also checks that
no `aten::index`, `aten::index_select` or `aten::cat` occurs within the annotated
expanded-attention core. Small metadata indexing and current-token projection
work remain necessary.

`bench_ttft.py` and `analyze_ab.py` are optional single-request diagnostics for
the cache/attention path. Their 32K prompt is not the final dataset workload.

For the final steady comparison, run 100 requests as an untimed preparation
pass for each variant, then run a separate 100-request measurement. Every pass
still performs the client's own warmup and cache flush. Use distinct variant
labels (`baseline-steady`, `candidate-steady`) to preserve the initial results;
pass these names to `analyze_dataset.py`. This separates first-use latency
outliers from the steady measurement without selecting individual requests.

`check_logprobs.py` additionally checks 64 fixed inputs with a 640-token cached
prefix. It scores the last prompt tokens plus the dataset response using
teacher forcing. `compare_logprobs.py` compares matching token probabilities;
it does not require autoregressive generated text to be identical across
different dynamic batch schedules.
