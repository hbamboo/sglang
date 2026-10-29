# NPU prefill transient workspace budget

`SGLANG_NPU_PREFILL_TRANSIENT_RESERVE_MB` keeps a measured number of MiB out of
the profiled KV allocation budget. It defaults to 0 and is valid only on an NPU
PD prefill worker. The paged KV layout, dtype and latent PD transfer ABI do not
change. Page alignment and the expanded pool's dummy page are still charged.
Negative values, non-prefill/non-NPU use and a budget with no complete KV page
are rejected. Explicit token caps remain absolute counts, not another byte
budget to subtract twice.

The knob reserves *capacity*, not a dummy tensor. It leaves physical memory
available for activations, asynchronous lifetimes and operator workspace.
The amount is workload/device dependent; it is not a universal 5 GiB rule.

## Why this is needed for expanded KV plus shared overlap

On the tested 96 GiB Ascend950DT, real 24721-token MoE windows were faster with
the staged implementation: 5.132 -> 4.653 ms. Gate/up and down physically overlap
their main-stream work. Routed GMM2 and finalize do slow down (about 10% and
19%), but their added device time does not explain the serving regression.

The unprofiled full-model diagnostic at nominal 9.7 req/s, 500 original customer
requests, showed the system-level memory cliff:

| Configuration | Actual QPS | Mean E2E ms | Forward allocation retries |
| --- | ---: | ---: | ---: |
| Expanded KV, shared overlap off, P memory fraction .90 | 9.572 | 1328.5 | 0 |
| Expanded KV + staged overlap, P memory fraction .90 | 8.474 | 5423.9 | 41 |
| Same staged overlap, P memory fraction .85 | 9.575 | 1162.9 | 0 |

All retries in the second row occurred inside forward during the measured
requests, not weight loading or warmup. Large-prefill host forward time rose
from about 94 to 1078 ms. Reducing only the KV static budget removed the retries
and restored throughput while retaining the original auxiliary stream.

The new schedule's transient working set was not covered by the previous KV
capacity plan. Operator overlap alone was therefore an insufficient acceptance
criterion. The diagnostic probes and rejected gate-only schedule are preserved
in Git history and experiment artifacts; they are absent from the final path.

## Reproduce the selected budget

```bash
# 6 P / 7 D, both features on, 5 GiB kept out of P's KV budget.
# The normal launcher still uses mem_fraction_static=.90 on both workers.
bash benchmark/ascend/prefill_expanded_kv/launch_reserved_1p1d.sh
```

Set `RESERVE_MB` to change the measured budget, and use the ordinary launcher's
`--prefill-env SGLANG_NPU_PREFILL_TRANSIENT_RESERVE_MB=0` for the old capacity
calculation. Do not set this variable globally on D.

The selected GLM shared-overlap threshold is 8192 actual prefill tokens; smaller
prefills retain the original serial path. The provided wrapper keeps allocator,
FIA, latent transport and D behavior consistent with the preceding comparison.
The final serving report records exact capacity, revisions, per-run metrics and
whether actual QPS >= 9.5 and mean E2E <= 1500 ms were verified.

The wrapper copies the site launcher and explicitly adds
`--disable-shared-experts-fusion` on both PD roles. The site original is untouched.
The generated command script was byte-identical to the launcher used for the
reported timed acceptance. Final timed application revision: `7027a44515`.
