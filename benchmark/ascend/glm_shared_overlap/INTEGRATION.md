# GLM-4.7-Flash shared-expert overlap: integration version

The first integration checkpoint was `c34ea0d0ca`. Validation is now complete:
the three-stage implementation, real NPU numerical checks, two complete
customer timing comparisons, five-capture profiling and paired full GSM8K
are available. Both GSM8K runs scored 1170/1319 (88.7036%); all 1319 complete
response texts match, with the same 15 truncated samples retained.

## Base and integration scope

Validated repository base: `0bcd822377da7b5718e674eaf9c870d349424dd1`.
The latest core implementation change is
`b8ab4bf4ae85259ad4ec85785d2b71874ce15b06`; later commits add validation and
documentation. All task commits are authored by `huangxudong`.

The code was tested in `hxd-glm` with the container's existing NPU runtime
adaptations. Seven pre-existing uncommitted runtime changes are deliberately
kept separate from this feature branch. Their original patch remains at
`/mnt/paas/hxdnpu/glm-shared-overlap-20261030/baseline-runtime.patch`, SHA256
`f803ea461f19aec048c1f682a91cb9499e86022ff10262e12a22b76ecd296062`.
It is byte-identical to the current runtime diff. Keep the corresponding NPU
adaptations in the integration target; this branch is not a replacement for
the container's environment and compatibility setup.

Core files:

- `python/sglang/srt/hardware_backend/npu/moe/shared_expert_overlap.py`
- `python/sglang/srt/models/glm4_moe_lite.py`
- `python/sglang/srt/layers/moe/moe_runner/ascend.py`
- `python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py`

## Enable and verify

Use separate shared experts (`--disable-shared-experts-fusion`) and:

```bash
export ASCEND_RT_VISIBLE_DEVICES=5
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_NPU_SHARED_EXPERT_OVERLAP=1
export SGLANG_NPU_SHARED_EXPERT_OVERLAP_MIN_TOKENS=8192
```

The three stages are shared gate/up with TopK/init routing, shared
SwiGLU+MX quant with routed GMM2, and shared down with finalize routing.
The final add waits for the shared result. Ordinary single-rank BF16 prefill
with native Ascend MXFP8 routed/shared experts is supported. Decode and
unsupported paths retain their original implementation. Actual prefills below
the selected threshold run serially; the threshold's code default is 0.
Leave PREQUANT off and cube/vector caps at -1.

In the validated container, run the real NPU regression:

```bash
ASCEND_RT_VISIBLE_DEVICES=5 python3 test/manual/ascend/test_glm_shared_expert_overlap.py
```

It covers GLM-sized MXFP8 projections, 1/128/1024/32768 tokens, repeated event
reuse, a non-default parent stream, allocator churn, exception cleanup and
decode fallback. All checked shared outputs were bitwise equal (max abs 0).

For a complete service invocation, use
`bash benchmark/ascend/glm_shared_overlap/reproduce_server.sh optimized perf`
after the task's service has released NPU 5 and port 19255. Use `baseline perf`
for the matched control; retain the same allocator in both variants.

## Evidence available now

The customer workload is 2000 original requests, 248 multi-request bursts,
at most 32768 new tokens per burst, and exactly one output token per request.
The two opposite-order comparisons reduced mean TTFT by 4.122% and 2.003%,
and P95 by 3.227% and 2.278%. All 2000 first output tokens matched in each
comparison. These are controlled burst results, not an arrival-rate or maximum
QPS claim. The separate full GSM8K result above supplies the quality evidence.

Five captures per variant show all three requested overlaps in the 30861-token
prefill (138/138 selected layer instances after discarding the first two
captures). The 1606-token scheduler tail benefits from serial fallback.
The full profiling package is available locally and in the remote result
directory as `final-customer-profiling.zip`, SHA256
`661e438908186667f10049cc24cb57d3ed2186e0d3cf8453d514b9582ae38ec8`.

See `PERFORMANCE_STATUS.md` for complete measurements and limitations.
Final GSM8K used 1319 questions per variant, fixed four-shot prompts, greedy
sampling and the same 8192-token budget. All samples including truncations
were retained, and all 2638 responses were independently rescored against the
original references. Detailed responses stay remote; the downloadable audit
contains numbers, equality flags and digests without model response text.
