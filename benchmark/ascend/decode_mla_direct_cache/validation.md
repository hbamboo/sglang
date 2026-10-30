# Decode MLA direct-cache validation — 2026-10-30

Only the KV Cat and standalone ScatterNdUpdate are removed. Query Cat, Query math, FlashMLA, final 576-wide latent-cache allocation and PD transport remain unchanged.

The native V3 change adds physical cache row/page stride handling. Passing the final 512/64 views to the previous operator was tested and produced 8,000 differing cache elements; that interface cannot safely replace staging without the native fix.

## Correctness and trace

- 34 hardware regressions passed, including page crossings, untouched slots/layers, transfer waits, unchanged PD descriptors and graph replay with changed token slots.
- Six fixed greedy requests match exactly: output text and all 192 output-token logprobs.
- All 2,400 timed requests succeeded; paired input/output lengths match across variants.
- Each independent decode profile covers 12 steps × 47 layers; CSV, op/API statistics and full trace JSON were validated.

| Kernel | Feature off | Feature on |
|---|---:|---:|
| V3 prolog | 564 | 564 |
| KV Cat | 564 | 0 |
| ScatterNdUpdate | 564 | 0 |
| Query Cat | 564 | 564 |
| FlashMLA | 564 | 564 |

## Service timing

Physical 6=P / 7=D, 1P1D TP1, original CHAT-mxfp8-customer-2000.jsonl, client seed 42, thinking enabled, cache flushed. P max/chunk prefill is 32,768. Both D variants use the same private native build, changing only the direct-cache flag in computation settings. Automatic server seeds and runtime timestamps differ. Timings exclude preparation and profiling.

| Load, two rounds each | Flag | Mean TTFT ms | Mean TPOT ms | Mean E2E ms | QPS |
|---|---:|---:|---:|---:|---:|
| Concurrency 8, 100 requests/round | 0 | 675.66 | 6.103 | 846.55 | 9.1961 |
| Concurrency 8, 100 requests/round | 1 | 665.79 | 5.997 | 833.71 | 9.3590 |
| Rate 10, 500 requests/round | 0 | 1110.19 | 6.897 | 1303.22 | 9.8657 |
| Rate 10, 500 requests/round | 1 | 1123.50 | 6.201 | 1297.06 | 9.8690 |

Rate-10 mean TPOT decreases about 10.1%; mean E2E decreases about 0.5%. Both satisfy mean E2E ≤ 1.5 s. Fixed-rate QPS is demand-limited; these runs do not establish a maximum-QPS capacity gain. Concurrency-8 E2E changes about 1.5%, with visible variation between rounds. There are only two rounds per load.

## Isolated prepare-stage timing

Captured graph: V3 + cache landing + unchanged Query Cat. Each replay contains 32 prepare stages, with 10 alternating groups, 3 warmup replays and 20 timed replays/group. NPU-event median divided by the number of stages; outputs match exactly. The initial single-stage graph was host-submission-limited and is excluded.

| Decode batch | Off µs/layer | On µs/layer | Reduction |
|---|---:|---:|---:|
| 1 | 25.547 | 20.132 | 21.2% |
| 8 | 26.891 | 20.578 | 23.5% |
| 32 | 31.424 | 24.570 | 21.8% |
| 64 | 41.156 | 34.163 | 17.0% |

This stage timing excludes FlashMLA, the rest of the layer, P queuing and PD transfer. It must not be interpreted as a whole-model speedup.

## Enable and reproduce

On D, select the private updated native package through PYTHONPATH and set `ENABLE_ASCEND_FLASH_MLA=1`, `SGLANG_NPU_USE_MLAPO_V3=1`, `SGLANG_NPU_MLAPO_V3_DIRECT_CACHE=1`. The new flag defaults off. The native direct module must advertise `mla_prolog_v3_direct_cache_version >= 1`. See README.md for the supported BF16 split-N scope and regression command.

Artifacts: `/sgl-workspace/p-kv-opt-artifacts/decode-direct-cache`. Raw customer records and profiler traces remain on the server. The CANN SDK, globally installed package and original working branches were not modified by this task. Only this test’s service process groups were stopped.

Dataset SHA256: `bb722901ab4a6857571837bbef58b30a4ca00ce08c1aaf170ada3c94694d0f0a`.
Native library SHA256: `bfde3380f09b4c182a0e0ffb11b6907024f557f3db8a9747b7531e99e65a86ea`.
