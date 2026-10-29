# GLM-4.7-Flash MXFP8 staged shared-expert overlap

Target: `hxd-glm`, model `/mnt/sfs_turbo/models/chatdpo-glm-flash-0727-mxfp8`, physical NPU 5.
Runtime base: SGLang `0bcd822377da7b5718e674eaf9c870d349424dd1` plus the container's pre-existing local changes, saved separately as `baseline-runtime.patch` in the result directory. Those changes are not part of the overlap commit.

Enable with `SGLANG_NPU_SHARED_EXPERT_OVERLAP=1`; the default is off. The path requires ordinary prefill, BF16 hidden states, single-rank TP, separate shared experts, native Ascend MXFP8 routed experts and MXFP8 shared projections. Unsupported paths and decode retain the original implementation. Use `--disable-shared-experts-fusion` to keep shared experts separate.

The optional `SGLANG_NPU_SHARED_EXPERT_OVERLAP_ANNOTATE=1` emits CPU profiler ranges. It is not enabled in timed comparisons.

The selected customer-workload configuration is:

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_NPU_SHARED_EXPERT_OVERLAP=1
export SGLANG_NPU_SHARED_EXPERT_OVERLAP_MIN_TOKENS=8192
```

Use the same allocator setting in the baseline, with the overlap flag set to 0.
The token threshold applies to each actual prefill invocation, including scheduler
tails. Below it, the original serial MLP runs. The threshold defaults to 0 for
compatibility; decode and unsupported paths always use the original path.
Leave `SGLANG_NPU_SHARED_EXPERT_OVERLAP_PREQUANT=0` and stream cube/vector limits
at -1. The measured alternatives did not consistently beat the original three
stages. `TUNING_ROUND2.md` contains all candidate results and their limits.

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

The primary workload is the user's `/mnt/sfs_turbo/hxd/CHAT-mxfp8-customer-2000.jsonl`: 2000 requests, 7566887 input tokens. `customer_benchmark.py` tokenizes `conversations[0].content` without changing the prompt, preserves file order, and forms 248 multi-request bursts with at most 32768 new tokens and 32 requests each. Observed pack sizes are 21333..32743 tokens, with up to 27 requests. Every request generates exactly one token. Prefix caching is disabled for the controlled comparison. This is a controlled burst benchmark, not an arrival-rate or queueing benchmark.

`benchmark.py` is a separate single-request microbenchmark: exactly 32768 input token IDs and one output token. Its results must not be substituted for the multi-request customer-workload result. Both clients measure the first SSE packet containing a generated token and check zero cached tokens. Profiler runs are separate from timed runs.

`gsm8k_eval.py` evaluates all 1319 GSM8K test questions, with four fixed examples from the train split, greedy sampling, up to 8192 generated tokens and the installed EvalScope answer parser/scorer. Every response, prompt hash, finish reason and score is retained. The original partial journals used concurrency 16 without decode graphs. The final paired evaluation uses `--parallel 32` with identical full decode graphs in both services; prefill graphs remain disabled so the changed prefill path executes. Final journals are separate from the old partial runs. Baseline and optimized runs must use identical prompts and settings; compare per-question answers as well as aggregate accuracy and truncation. Quality-run latency is not a customer TTFT measurement.

`run_customer_validation.py` orchestrates the primary customer-workload performance and profiling comparisons, then the full GSM8K baseline and optimized runs. `run_validation.py` contains server/job helpers and the original microbenchmark workflow. They use localhost port 19255 on physical NPU 5 and only terminate server process groups they created after verifying the command line. Progress is recorded atomically in `validation-state.json`; inspect the live process before restarting anything. GSM8K `--resume` preserves completed records in the same variant's JSONL journal.

Final quality services use `--cuda-graph-backend-prefill disabled
--cuda-graph-backend-decode full --cuda-graph-max-bs-decode 32
--cuda-graph-bs-decode 32` with padding enabled. Capturing six batch-size graphs
exhausted SQ/CQ resources on first replay in the baseline; the single graph
configuration successfully replays. Keep identical graph settings in both arms.
Run `compare_gsm8k.py --results-dir <results> --rescore` after both full runs:
it checks all 1319 questions, references, prompt hashes, completion reasons,
truncation and summary counts, then recomputes 2638 EvalScope scores.
`record_provenance.py` records package/data/config hashes and verifies that the
pre-existing runtime patch remains byte-identical. `build_delivery.py` bundles
text-free numeric audits, per-case equality flags, sources and the Git patch;
the large profiling ZIP stays separate and is referenced by checksum. Detailed
responses, logs and launch configurations stay in the remote result directory.
`export_audit_summary.py` creates the summary using explicit field whitelists.

After the task releases its service, `bash reproduce_server.sh baseline perf`
or `bash reproduce_server.sh optimized perf` starts the exact controlled
performance configuration. Run `customer_benchmark.py --output <new-result.json>`
against each service in turn. For GSM8K, use the `quality` purpose and
`gsm8k_eval.py --parallel 32 --output <new-result.json>`. Use separate output
directories and preserve both full journals. These commands only use NPU 5
and do not stop other processes.

Results and full profiling are written to `/mnt/paas/hxdnpu/glm-shared-overlap-20261030`. Performance and quality conclusions require the completed result files, not only a passing unit test or an enabled environment flag.

## Final measured result

Two opposite-order, fresh-service customer comparisons reduced mean TTFT by
4.122% and 2.003%, and P95 by 3.227% and 2.278%. No outliers were excluded.
Both full GSM8K runs scored 1170/1319 (88.7036%), with the same 15 truncated
samples. All 1319 prompt hashes, predicted answers and full response texts
matched. Independent reference-based rescoring checked all 2638 responses.
See `PERFORMANCE_STATUS.md` for the full scope, raw-data locations and limits.
