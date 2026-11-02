"""NPU timing + kernel-profile harness for SGLANG_NPU_SPLIT_KV_B_PROJ.

Formal latency measures the complete per-token projection assembly:

  baseline : one matmul -> view [T,H,Dk+Dv] -> V.contiguous() -> final K assembly
  candidate: two matmuls (prepared Wk/Wv) -> final K assembly

Both include allocations, output creation and the remaining K destination /
RoPE writes.  Warmup, alternating order, NPU event timing, median/mean/p95.
Static weight preparation is measured/reported separately.

Profiling additionally records every NPU kernel and checks that the baseline
V-source and K_nope-source slices disappear without an equivalent hidden pack.
"""

import json
import os
import statistics
import sys

FRAMEWORK = "/mnt/sfs_turbo/hxd/split-kv-matmul-20261102/framework"
sys.path.insert(0, f"{FRAMEWORK}/python")
sys.path.insert(0, f"{FRAMEWORK}/test/manual/ascend")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "5")
os.environ["SGLANG_NPU_SPLIT_KV_B_PROJ"] = "1"

import torch
import torch_npu

torch.npu.set_device(0)

from sglang.srt.hardware_backend.npu.modules import split_kv_b_proj as skv

import split_kv_test_utils as tu
import test_split_kv_b_proj_npu as T

DEV = "npu:0"
HEAD = 20
DK, DV, DR = 192, 256, 64
TOTAL = DK + DV
TOKENS = [0, 1, 31, 127, 128, 129, 3787, 15148]
DTYPES = [torch.bfloat16, torch.float16]
TOL = {torch.bfloat16: (0.5, 2e-2), torch.float16: (0.05, 1e-2)}


def setup_runtime():
    T.attn_mod.get_token_to_kv_pool = lambda: T.DummyPool()
    T.attn_mod.can_fuse_expanded_kv = lambda *args, **kwargs: False
    import sglang.srt.model_executor.forward_context as fctx

    fctx.get_token_to_kv_pool = lambda: T.DummyPool()


def make_projection_fns(m, tokens, dtype):
    x = T.npu_random((tokens, 512), dtype, 1000 + tokens)
    kv_a = T.actual_kv_a(m, x)
    k_pe = T.npu_random((tokens, 1, DR), dtype, 2000 + tokens)

    def baseline():
        kv = m.kv_b_proj(kv_a)[0]
        kv = kv.view(-1, HEAD, TOTAL)
        k_nope = kv[..., :DK]
        v = kv[..., DK:].contiguous()
        k = m._concat_and_cast_mha_k(k_nope, k_pe, None)
        return k, v

    def candidate():
        split = skv.maybe_forward_split_kv_b_proj(m, m.kv_b_proj, kv_a, None)
        assert split is not None, "split route unexpectedly inactive"
        k_nope, v = split
        k = m._concat_and_cast_mha_k(k_nope, k_pe, None)
        return k, v

    return baseline, candidate, kv_a, k_pe


def timed_alternating(base_fn, cand_fn, warmup, iters):
    for _ in range(warmup):
        base_fn()
        cand_fn()
    torch.npu.synchronize()
    events = {"baseline": [], "candidate": []}
    hold = {}
    for i in range(iters):
        order = (
            (("baseline", base_fn), ("candidate", cand_fn))
            if i % 2 == 0
            else (("candidate", cand_fn), ("baseline", base_fn))
        )
        for name, fn in order:
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            events[name].append((start, end))
            hold[name] = out
    torch.npu.synchronize()

    def summarize(evs):
        ms = [s.elapsed_time(e) for s, e in evs]

        def all_finite(out):
            if torch.is_tensor(out):
                return bool(torch.isfinite(out).all().item())
            if isinstance(out, (tuple, list)):
                return all(all_finite(x) for x in out)
            return True

        finite = all_finite(hold["baseline"]) and all_finite(hold["candidate"])
        s = sorted(ms)
        return {
            "n": len(ms),
            "median_ms": statistics.median(ms),
            "mean_ms": statistics.fmean(ms),
            "p95_ms": s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))],
            "min_ms": s[0],
            "max_ms": s[-1],
            "stdev_ms": statistics.pstdev(ms) if len(ms) > 1 else 0.0,
            "outputs_finite": bool(finite),
            "sample_ms": ms[:10],
        }

    return summarize(events["baseline"]), summarize(events["candidate"])


def bench_matrix(metrics):
    rows = []
    for dtype in DTYPES:
        for tokens in TOKENS:
            m, x, pos, fb = T.build_case(dtype, False, tokens)
            state, _ = skv.prepare_split_kv_b_proj(m, m.kv_b_proj)
            assert state is not None
            base_fn, cand_fn, _, _ = make_projection_fns(m, tokens, dtype)
            warmup = 20 if tokens < 1000 else 10
            iters = 200 if tokens < 1000 else (100 if tokens < 10000 else 60)
            b, c = timed_alternating(base_fn, cand_fn, warmup, iters)
            delta = c["median_ms"] - b["median_ms"]
            ratio = c["median_ms"] / b["median_ms"] if b["median_ms"] else float("nan")
            row = {
                "dtype": T.DTYPE_NAMES[dtype],
                "T": tokens,
                "baseline": b,
                "candidate": c,
                "delta_ms": delta,
                "ratio": ratio,
                "iters": iters,
                "warmup": warmup,
            }
            rows.append(row)
            print(
                "LAT",
                json.dumps(
                    {
                        "dtype": row["dtype"],
                        "T": tokens,
                        "iters": iters,
                        "base_med": round(b["median_ms"], 4),
                        "cand_med": round(c["median_ms"], 4),
                        "base_mean": round(b["mean_ms"], 4),
                        "cand_mean": round(c["mean_ms"], 4),
                        "base_p95": round(b["p95_ms"], 4),
                        "cand_p95": round(c["p95_ms"], 4),
                        "ratio": round(ratio, 4),
                        "stdev_b": round(b["stdev_ms"], 4),
                        "stdev_c": round(c["stdev_ms"], 4),
                        "finite": b["outputs_finite"] and c["outputs_finite"],
                    }
                ),
            )
    metrics["latency"] = rows


def full_prepare_timing(metrics):
    rows = []
    for dtype in (torch.bfloat16,):
        for tokens in (1, 128, 3787):
            m_old, x, pos, fb = T.build_case(dtype, False, tokens, seed=777)
            m_new, x2, pos2, fb2 = T.build_case(dtype, False, tokens, seed=777)
            assert torch.equal(m_old.kv_b_proj.weight, m_new.kv_b_proj.weight)
            state, _ = skv.prepare_split_kv_b_proj(m_new, m_new.kv_b_proj)
            assert state is not None
            base_fn = lambda: T.run_prepare(m_old, x, pos, fb)
            cand_fn = lambda: T.run_prepare(m_new, x, pos, fb)
            b, c = timed_alternating(base_fn, cand_fn, 10, 100)
            rows.append(
                {
                    "dtype": T.DTYPE_NAMES[dtype],
                    "T": tokens,
                    "baseline": b,
                    "candidate": c,
                    "delta_ms": c["median_ms"] - b["median_ms"],
                }
            )
            print(
                "FULLPREP",
                json.dumps(
                    {
                        "T": tokens,
                        "base_med": round(b["median_ms"], 4),
                        "cand_med": round(c["median_ms"], 4),
                        "ratio": round(c["median_ms"] / b["median_ms"], 4),
                    }
                ),
            )
    metrics["full_prepare"] = rows


def static_preparation_cost(metrics):
    import time

    m, x, pos, fb = T.build_case(torch.bfloat16, False, 128)
    module = m.kv_b_proj
    per_run = []
    deltas = []
    for _ in range(10):
        skv.invalidate_split_kv_b_proj(module)
        torch.npu.synchronize()
        before = torch.npu.memory_allocated()
        t0 = time.perf_counter()
        state, reason = skv.prepare_split_kv_b_proj(m, module, force=True)
        torch.npu.synchronize()
        t1 = time.perf_counter()
        after = torch.npu.memory_allocated()
        assert state is not None
        per_run.append((t1 - t0) * 1e3)
        deltas.append(after - before)
    bytes_expected = (
        module.weight.numel() * module.weight.element_size()
        + (module.bias.numel() * module.bias.element_size() if module.bias is not None else 0)
    )
    rec = {
        "per_layer_median_ms": statistics.median(per_run),
        "per_layer_mean_ms": statistics.fmean(per_run),
        "per_layer_min_ms": min(per_run),
        "per_layer_max_ms": max(per_run),
        "runs": per_run,
        "added_bytes_expected": bytes_expected,
        "added_bytes_alloc_delta": deltas,
        "projected_47_layers_mb": bytes_expected * 47 / (1024 * 1024),
        "projected_47_layers_seconds": statistics.median(per_run) * 47 / 1000,
        "weight_dtype": str(module.weight.dtype),
        "note": "one-time post-load; 2x local kv_b_proj bytes retained for fallback",
    }
    metrics["static_preparation"] = rec
    print("PREP", json.dumps({k: v for k, v in rec.items() if k != "runs"}))


def _prof_summary(events, tokens):
    dev = {}
    cpu = {}
    slice_inputs = []
    copy_inputs = []
    for e in events:
        if e.get("ph") != "X":
            continue
        name = e.get("name", "")
        cat = e.get("cat", "")
        args = e.get("args", {})
        if cat == "cpu_op":
            cpu[name] = cpu.get(name, 0) + 1
            dims = args.get("Input Dims", "")
            if name == "aten::slice":
                slice_inputs.append(dims)
            if name == "aten::copy_":
                copy_inputs.append(dims)
        elif name.startswith("aclnn") or args.get("Task Type") in (
            "KERNEL_AICORE",
            "KERNEL_AIVECTOR",
            "KERNEL_AIVEC",
        ):
            dev[name] = dev.get(name, 0) + 1
    return {
        "tokens": tokens,
        "device_kernels": dev,
        "cpu_ops": {k: v for k, v in sorted(cpu.items(), key=lambda kv: -kv[1]) if v},
        "slice_inputs": slice_inputs,
        "copy_inputs": copy_inputs,
        "matmul_kernels": sum(v for k, v in dev.items() if "MatMul" in k or "matmul" in k),
        "slice_kernels": sum(v for k, v in dev.items() if "Slice" in k),
        "viewcopy_kernels": sum(v for k, v in dev.items() if "ViewCopy" in k),
        "broadcast_kernels": sum(v for k, v in dev.items() if "Broadcast" in k),
        "contiguous_ops": cpu.get("aten::contiguous", 0),
        "packed_source_slices": sum(
            1 for dims in slice_inputs if dims.startswith(f"{tokens},20,448;")
        ),
    }


def profile_ops(metrics):
    import pathlib

    pathlib.Path(tu.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    results = {}
    for tokens in (129, 3787):
        m, x, pos, fb = T.build_case(torch.bfloat16, False, tokens)
        state, _ = skv.prepare_split_kv_b_proj(m, m.kv_b_proj)
        assert state is not None
        base_fn, cand_fn, _, _ = make_projection_fns(m, tokens, torch.bfloat16)
        for side, fn in (("baseline", base_fn), ("candidate", cand_fn)):
            for _ in range(5):
                fn()
            torch.npu.synchronize()
            trace = f"{tu.RESULTS_DIR}/profile_{side}_T{tokens}.json"
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                record_shapes=True,
            ) as prof:
                for _ in range(5):
                    fn()
            prof.export_chrome_trace(trace)
            events = json.load(open(trace))
            if not isinstance(events, list):
                events = events.get("traceEvents", [])
            summary = _prof_summary(events, tokens)
            summary["trace"] = trace
            results[f"{side}_T{tokens}"] = summary
            print(
                "PROFILE",
                json.dumps(
                    {
                        "side": side,
                        "T": tokens,
                        "matmul": summary["matmul_kernels"],
                        "slice": summary["slice_kernels"],
                        "viewcopy": summary["viewcopy_kernels"],
                        "broadcast": summary["broadcast_kernels"],
                        "contiguous_ops": summary["contiguous_ops"],
                        "packed_source_slices": summary["packed_source_slices"],
                        "device_kernels": summary["device_kernels"],
                    }
                ),
            )
            if side == "baseline":
                assert summary["slice_kernels"] >= 5, summary
                assert summary["matmul_kernels"] == 5, summary
                assert summary["contiguous_ops"] >= 5, summary
                assert summary["packed_source_slices"] >= 10, summary
            else:
                assert summary["slice_kernels"] == 0, summary
                assert summary["matmul_kernels"] == 10, summary
                assert summary["contiguous_ops"] == 0, summary
                assert summary["packed_source_slices"] == 0, summary
                assert summary["viewcopy_kernels"] == 10, summary
                assert summary["broadcast_kernels"] == 5, summary
                # Only the remaining destination K_nope and RoPE writes.
                assert all(
                    dims.startswith(f"{tokens},20,192;") or dims.startswith(f"{tokens},20,64;")
                    for dims in summary["copy_inputs"]
                ), summary["copy_inputs"]
    metrics["profile"] = results


def main():
    setup_runtime()
    def run(metrics):
        bench_matrix(metrics)
        full_prepare_timing(metrics)
        static_preparation_cost(metrics)
        profile_ops(metrics)

    tu.run_entrypoint("profile_split_kv_b_proj_npu", run)


if __name__ == "__main__":
    main()
