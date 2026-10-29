# GLM-4.7-Flash MXFP8 staged shared-expert overlap

Target: `hxd-glm`, model `/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8`, physical NPU 5.
Runtime base: SGLang `0bcd822377da7b5718e674eaf9c870d349424dd1` plus the container's pre-existing local changes, saved separately as `baseline-runtime.patch` in the result directory. Those changes are not part of the overlap commit.

Enable with `SGLANG_NPU_SHARED_EXPERT_OVERLAP=1`; the default is off. The path requires ordinary prefill, BF16 hidden states, single-rank TP, separate shared experts, native Ascend MXFP8 routed experts and MXFP8 shared projections. Unsupported paths and decode retain the original implementation. Use `--disable-shared-experts-fusion` to keep shared experts separate.

The optional `SGLANG_NPU_SHARED_EXPERT_OVERLAP_ANNOTATE=1` emits CPU profiler ranges. It is not enabled in timed comparisons.

## Device schedule

| Main stream | Shared stream |
| --- | --- |
| Router GEMM, then top-k and init routing | Input-ready event, gate/up projection |
| Routed GMM1 | Gate/up completion |
| Record GMM1 event; routed GMM2 | Wait GMM1 event; SwiGLU and MXFP8 quantization |
| Record GMM2 event; finalize routing | Wait GMM2 event; shared down projection |
| Routed scale; wait shared-done event; add | Record shared-done event |

The same per-device auxiliary stream is reused across layers. Events are reused per layer. No device-wide or CPU synchronization is added to the steady-state path. Cross-stream input and output allocations use `record_stream`; the scoped context also joins work on exceptions. The Ascend runner consumes only the context for the identical expert-layer object.

`NPUMXFP8LinearMethod.apply_quantized` reuses the existing matmul/scales/bias/rounding implementation while allowing activation quantization to be issued before the down projection's scheduling boundary.

## Verification

```bash
ASCEND_RT_VISIBLE_DEVICES=5 python3 test/manual/ascend/test_glm_shared_expert_overlap.py
```

This uses real NPU MXFP8 projections at GLM's shared-expert dimensions (2048 -> 3072 -> 1536 -> 2048), tests 1/128/1024/32768 token inputs, repeated reuse, a non-default parent stream, allocator churn, exceptional cleanup and decode fallback. All tested shared outputs must be bitwise equal to the original serial implementation.

`benchmark.py` sends exactly 32768 token IDs and requests one token with greedy sampling and `ignore_eos=True`. It measures the first SSE packet containing a generated token, checks zero cached tokens, performs three full-size warmups and records 15 timed repetitions. Profiler runs are separate from timed runs.

`gsm8k_eval.py` evaluates all 1319 GSM8K test questions, with four fixed examples from the train split, greedy sampling, up to 8192 generated tokens, 16 concurrent requests and the installed EvalScope answer parser/scorer. Every response, prompt hash, finish reason and score is retained. Baseline and optimized runs must use identical prompts and settings; compare per-question answers as well as aggregate accuracy and truncation.

`run_validation.py` orchestrates serial runs against localhost port 19255 on physical NPU 5. It only terminates server process groups that it created and verifies their command line before termination. Progress is recorded atomically in `validation-state.json`; inspect the live process before restarting anything.

Results and full profiling are written to `/mnt/paas/hxdnpu/glm-shared-overlap-20261030`. Performance and quality conclusions require the completed result files, not only a passing unit test or an enabled environment flag.
