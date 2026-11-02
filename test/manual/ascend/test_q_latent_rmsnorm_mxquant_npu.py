"""Granted-NPU correctness / profiling / timing harness for the Q-latent
RMSNorm + DynamicMxQuant fusion.

This file is intentionally inert without an explicit NPU grant:

    QNORM_MXQUANT_NPU_GRANTED=1 PYTHONPATH=<worktree>/python \
        python test/manual/ascend/test_q_latent_rmsnorm_mxquant_npu.py

It never runs as part of the CPU contract suite.  Results (JSON + raw logs)
are written under <task-root>/results/npu/.
"""

import argparse
import contextlib
import json
import os
import statistics
import sys
import traceback

if os.environ.get("QNORM_MXQUANT_NPU_GRANTED") != "1":
    print(
        "REFUSED: QNORM_MXQUANT_NPU_GRANTED=1 is required. "
        "No NPU is touched without the architect's explicit grant."
    )
    sys.exit(0)

import torch  # noqa: E402

import sglang.srt.layers.quantization  # noqa: F401,E402
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (  # noqa: E402
    NPUMXFP8LinearMethod,
)
from sglang.srt.hardware_backend.npu.modules import (  # noqa: E402
    deepseek_v2_attention_mla_npu as mha,
)
from sglang.srt.hardware_backend.npu.modules import (  # noqa: E402
    q_latent_rmsnorm_mxquant as qlm,
)

import torch_npu  # noqa: E402

BLOCK = 32
EMAX_E4M3 = 8
Q_LORA = 768
QKV_WIDTH = 1344  # q_lora 768 + kv_lora 512 + qk_rope 64 (baseline qkv-a output)
DEFAULT_OUT = 2048
TOKENS = [0, 1, 31, 127, 128, 129, 3787, 15148]


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------
def stats(values):
    t = torch.as_tensor(values, dtype=torch.float64)
    if t.numel() == 0:
        return {"count": 0}
    return {
        "count": int(t.numel()),
        "finite": int(torch.isfinite(t).sum().item()),
        "bad": int((~torch.isfinite(t)).sum().item()),
        "max": float(t.max().item()),
        "mean": float(t.mean().item()),
        "min": float(t.min().item()),
    }


def err_stats(a, b):
    d = (a.float() - b.float()).abs().flatten()
    return {
        "count": int(d.numel()),
        "finite": int(torch.isfinite(d).sum().item()),
        "bad": int((~torch.isfinite(d)).sum().item()),
        "max_abs": float(d.max().item()) if d.numel() else 0.0,
        "mean_abs": float(d.mean().item()) if d.numel() else 0.0,
    }


def rel_stats(ref, got):
    r = ref.float().flatten()
    g = got.float().flatten()
    denom = r.abs().clamp_min(1e-6)
    d = (g - r).abs() / denom
    return {
        "max_rel": float(d.max().item()) if d.numel() else 0.0,
        "mean_rel": float(d.mean().item()) if d.numel() else 0.0,
    }


def byte_mismatch(a, b):
    ab = a.contiguous().cpu().view(torch.uint8).flatten()
    bb = b.contiguous().cpu().view(torch.uint8).flatten()
    return {
        "count": int(ab.numel()),
        "mismatch": int((ab != bb).sum().item()),
    }


def e8m0_to_float(scale_u8):
    # Installed layout: uint8 E8M0 with implicit bias 127 (value = 2**(b-127)).
    return torch.pow(2.0, scale_u8.float() - 127.0)


def float_scale_to_e8m0_bytes(scale_float):
    # Oracle scales are exact powers of two; encode with bias 127.
    return (torch.log2(scale_float).round().long() + 127).clamp(0, 255).to(torch.uint8)


def dequant_mx(q_fp8, scale_u8):
    """Dequantise [T, K] FP8 with [T, K/64, 2] E8M0 block scales."""
    rows, width = q_fp8.shape
    blocks = width // BLOCK
    scale = e8m0_to_float(scale_u8.reshape(rows, blocks))
    return q_fp8.float() * scale.repeat_interleave(BLOCK, dim=-1)


def cpu_rms_oracle(x_bf16, gamma_bf16, eps):
    xf = x_bf16.float()
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return xf * inv * gamma_bf16.float()


def cpu_mx_quant_oracle(norm_fp32):
    """OCP/scale_alg=0 MXFP8 E4M3 block-32 quantisation, independent of NPU."""
    rows, width = norm_fp32.shape
    blocks = norm_fp32.reshape(rows, width // BLOCK, BLOCK)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    _, exp = torch.frexp(amax)  # amax = m * 2**e, 0.5 <= m < 1
    shared_exp = torch.where(amax > 0, (exp - 1) - EMAX_E4M3, torch.zeros_like(exp))
    scale = torch.pow(2.0, shared_exp.float())
    q = (blocks / scale).to(torch.float8_e4m3fn).float() * scale
    return q.reshape(rows, width), scale.squeeze(-1)


def rows_for_oracle(rows):
    # Norm and block quantisation are per-row; row sampling is exact for the
    # sampled rows.  Large cases use deterministic spread rows to bound CPU time.
    if rows <= 2048:
        return torch.arange(rows)
    idx = torch.linspace(0, rows - 1, 256).long().unique()
    return idx


def make_input(rows, seed, extreme=False, pad=64):
    g = torch.Generator().manual_seed(seed)
    if extreme:
        base = torch.zeros((rows, QKV_WIDTH + pad), dtype=torch.bfloat16)
        vals = torch.tensor(
            [448.0, 224.0, 1.0, 0.5, 2.0**-9, 2.0**-20, 0.0, -448.0, -1.0, -2.0**-9],
            dtype=torch.bfloat16,
        )
        region = base[:, pad : pad + QKV_WIDTH]
        region.copy_(
            vals[
                torch.randint(0, vals.numel(), (rows, QKV_WIDTH), generator=g)
            ]
        )
    else:
        base = torch.randn((rows, QKV_WIDTH + pad), generator=g).to(torch.bfloat16)
    qkv = base[:, pad : pad + QKV_WIDTH]
    return base, qkv


def make_gamma(seed, eps):
    g = torch.Generator().manual_seed(seed + 991)
    gamma = (1.0 + 0.1 * torch.randn((Q_LORA,), generator=g)).to(torch.bfloat16)
    return gamma, eps


def make_layer(rows, out_features, seed, device):
    """Build the real NPUMXFP8LinearMethod consumer with processed weights."""
    g = torch.Generator().manual_seed(seed + 17)
    layer = {
        "weight": torch.randn((out_features, Q_LORA), generator=g)
        .to(torch.bfloat16)
        .to(device),
        "bias": None,
    }

    class L:
        pass

    obj = L()
    obj.weight = torch.nn.Parameter(layer["weight"], requires_grad=False)
    obj.bias = None
    kernel = NPUMXFP8LinearMethod()
    kernel.process_weights_after_loading(obj)
    assert getattr(obj, "_sgl_npu_mx_quant_fusable", False)
    obj._mx_kernel = kernel
    obj.quant_method = kernel
    # ColumnParallelLinear-compatible surface used by the route helper.
    obj.tp_size = 1
    obj.gather_output = False
    obj.skip_bias_add = False

    def call(x):
        return kernel.apply(obj, x, None), None

    obj.__call__ = call
    return obj


@contextlib.contextmanager
def count_op(namespace, name):
    original = getattr(namespace, name)
    counter = {"n": 0}

    def wrapped(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    setattr(namespace, name, wrapped)
    try:
        yield counter
    finally:
        setattr(namespace, name, original)


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------
def run_case(rows, seed, eps, device, out_features, extreme=False, pad=64):
    torch.npu.synchronize()
    base, qkv = make_input(rows, seed, extreme=extreme, pad=pad)
    qkv = qkv.to(device)
    gamma_host, eps = make_gamma(seed, eps)
    gamma = gamma_host.to(device)
    layer = make_layer(rows, out_features, seed, device)

    q = qkv.split([Q_LORA, QKV_WIDTH - Q_LORA], dim=-1)[0]  # stride 1344 view
    meta = {
        "rows": rows,
        "seed": seed,
        "eps": eps,
        "extreme": extreme,
        "qkv_stride": list(q.stride()),
        "qkv_storage_offset": int(q.storage_offset()),
        "q_contiguous": bool(q.is_contiguous()),
        "beta": None,
    }

    if rows == 0:
        meta["route"] = "fallback_empty"
        meta["fused_gate"] = qlm.fusion_blocker(
            _norm_stub(gamma, eps), layer, q, is_compiling=lambda: False,
            is_capturing=lambda: False,
        )
        return meta

    # ---- old path: Slice -> RmsNorm -> DynamicMxQuant -> q_b_proj ----
    q_norm = torch_npu.npu_rms_norm(q, gamma, eps)[0]
    q_old, scale_old = torch.ops.npu.npu_dynamic_mx_quant(
        q_norm, dst_type=torch.float8_e4m3fn
    )
    out_old = layer._mx_kernel.apply(layer, q_norm, None)
    out_old_tuple = layer._mx_kernel.apply(layer, (q_old, scale_old), None)
    torch.npu.synchronize()

    # ---- new path: Slice -> fused RmsNorm+DynamicMxQuant -> same q_b_proj ----
    with count_op(torch.ops.npu, "npu_dynamic_mx_quant") as dyn_count:
        out_new = qlm.forward_q_latent_rmsnorm_mxquant(
            _norm_stub(gamma, eps),
            layer,
            q,
            is_compiling=lambda: False,
            is_capturing=lambda: False,
        )
        q_new, scale_new = _LAST_FUSED_OUTPUT
        torch.npu.synchronize()
    assert out_new is not None

    meta.update(
        {
            "route": "fused",
            "extra_dynamic_mx_quant_after_fused": dyn_count["n"],
            "old_tuple_vs_old_tensor_max_abs": err_stats(out_old, out_old_tuple)[
                "max_abs"
            ],
            "fp8_bytes": byte_mismatch(q_old, q_new),
            "scale_bytes": byte_mismatch(scale_old, scale_new),
            "dequant_old_vs_new": err_stats(
                dequant_mx(q_old, scale_old), dequant_mx(q_new, scale_new)
            ),
            "output_old_vs_new": err_stats(out_old, out_new),
            "output_rel_old_vs_new": rel_stats(out_old, out_new),
            "output_new_vs_old_tuple": err_stats(out_old_tuple, out_new),
            "out_old_dtype": str(out_old.dtype),
            "out_new_dtype": str(out_new.dtype),
            "q_new_dtype": str(q_new.dtype),
            "scale_new_shape": list(scale_new.shape),
            "scale_new_dtype": str(scale_new.dtype),
        }
    )

    # ---- independent CPU FP32 RMS + OCP MX oracle ----
    idx = rows_for_oracle(rows)
    idx_dev = idx.to(device)
    x_cpu = q[idx_dev].float().cpu()
    g_cpu = gamma.float().cpu()
    norm_oracle = cpu_rms_oracle(x_cpu, g_cpu, eps)
    q_oracle, scale_oracle = cpu_mx_quant_oracle(norm_oracle)
    q_old_cpu = q_old[idx_dev].cpu()
    scale_old_cpu = scale_old[idx_dev].cpu()
    q_new_cpu = q_new[idx_dev].cpu()
    scale_new_cpu = scale_new[idx_dev].cpu()
    meta["oracle_rows"] = int(idx.numel())
    meta["oracle_note"] = (
        "full" if idx.numel() == rows else "deterministic 256-row sample; "
        "RMS and block-32 quant are per-row so sampled rows are exact"
    )
    meta["old_dequant_vs_oracle"] = err_stats(
        dequant_mx(q_old_cpu, scale_old_cpu), q_oracle
    )
    meta["new_dequant_vs_oracle"] = err_stats(
        dequant_mx(q_new_cpu, scale_new_cpu), q_oracle
    )
    oracle_scale_bytes = float_scale_to_e8m0_bytes(scale_oracle)
    meta["old_scale_vs_oracle"] = byte_mismatch(
        scale_old_cpu.reshape(idx.numel(), -1), oracle_scale_bytes
    )
    meta["new_scale_vs_oracle"] = byte_mismatch(
        scale_new_cpu.reshape(idx.numel(), -1), oracle_scale_bytes
    )
    return meta


_LAST_FUSED_OUTPUT = (None, None)
_NORM_STUBS = {}


class NormStub:
    def __init__(self, gamma, eps):
        self.has_weight = True
        self.weight = torch.nn.Parameter(gamma, requires_grad=False)
        self.variance_epsilon = eps
        self.variance_size_override = None
        self.cast_x_before_out_mul = False
        self._forward_method = None

    def __call__(self, x):
        return torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)[0]


def _norm_stub(gamma, eps):
    key = (int(gamma.data_ptr()), float(eps))
    stub = _NORM_STUBS.get(key)
    if stub is None:
        stub = NormStub(gamma, eps)
        _NORM_STUBS[key] = stub
    return stub


def install_fused_recorder():
    global _LAST_FUSED_OUTPUT
    real = torch_npu.npu_rms_norm_dynamic_mx_quant

    def recorder(x, gamma, **kwargs):
        out = real(x, gamma, **kwargs)
        globals()["_LAST_FUSED_OUTPUT"] = (out[0], out[1])
        return out

    qlm._fused_rmsnorm_mxquant_op = lambda: recorder


# ---------------------------------------------------------------------------
# real prepare route (real kernels, mocked non-q-latent branches)
# ---------------------------------------------------------------------------
def run_prepare_route(rows, seed, eps, device, out_features):
    torch.npu.synchronize()
    _, qkv = make_input(rows, seed, pad=64)
    qkv = qkv.to(device)
    gamma, _ = make_gamma(seed, eps)
    gamma = gamma.to(device)
    layer = make_layer(rows, out_features, seed, device)
    norm = _norm_stub(gamma, eps)

    pool_state = {"sets": 0}

    class Pool:
        def set_kv_buffer(self, *args):
            pool_state["sets"] += 1

        def get_kv_buffer(self, layer_id):
            return (
                torch.empty((rows, 1, 512), dtype=torch.bfloat16, device=device),
                torch.empty((rows, 1, 64), dtype=torch.bfloat16, device=device),
            )

    class FakeKvB:
        def __call__(self, x):
            return (
                torch.zeros((x.shape[0], 10), dtype=torch.bfloat16, device=device),
                None,
            )

    m = type("M", (), {})()
    m.q_lora_rank = Q_LORA
    m.use_dsa = False
    m.kv_lora_rank = 512
    m.qk_rope_head_dim = 64
    m.num_local_heads = 1
    m.qk_head_dim = 68
    m.qk_nope_head_dim = 4
    m.q_a_layernorm = norm
    m.q_b_proj = layer
    m.kv_a_layernorm = _norm_stub(
        torch.ones(512, dtype=torch.bfloat16, device=device), eps
    )
    m.kv_b_proj = FakeKvB()
    m.use_deepseek_yarn_rope = False
    m.rotary_emb = None
    m.layer_id = 0
    m._concat_and_cast_mha_k = lambda k_nope, k_pe, batch: k_nope

    old_globals = {
        "get_attn_tp_context": mha.get_attn_tp_context,
        "get_token_to_kv_pool": mha.get_token_to_kv_pool,
        "can_fuse_expanded_kv": mha.can_fuse_expanded_kv,
        "scattered_to_tp_attn_full": mha.scattered_to_tp_attn_full,
        "_use_ag_after_qlora": mha._use_ag_after_qlora,
    }
    mha.get_attn_tp_context = lambda: type(
        "Ctx", (), {"fetch_qkv_latent": lambda self: qkv}
    )()
    mha.get_token_to_kv_pool = lambda: Pool()
    mha.can_fuse_expanded_kv = lambda *a: False
    mha.scattered_to_tp_attn_full = lambda t, fb: t
    mha._use_ag_after_qlora = False

    fused_calls = {"n": 0}
    real_fused = qlm._fused_rmsnorm_mxquant_op

    def counting_op():
        op = real_fused()

        def wrapped(*args, **kwargs):
            fused_calls["n"] += 1
            return op(*args, **kwargs)

        return wrapped

    qlm._fused_rmsnorm_mxquant_op = counting_op
    try:
        with count_op(torch.ops.npu, "npu_dynamic_mx_quant") as dyn_count:
            q, k, v, fb = mha.forward_mha_prepare_npu(
                m,
                torch.arange(rows, device=device),
                torch.zeros((rows, 8), dtype=torch.bfloat16, device=device),
                type("B", (), {"out_cache_loc": torch.arange(rows, device=device)})(),
                None,
                type("S", (), {"layer_input_mode": None, "attn_mode": None})(),
            )
            torch.npu.synchronize()
    finally:
        qlm._fused_rmsnorm_mxquant_op = real_fused
        for key, value in old_globals.items():
            setattr(mha, key, value)

    return {
        "rows": rows,
        "fused_op_calls": fused_calls["n"],
        "extra_dynamic_mx_quant_after_fused": dyn_count["n"],
        "q_dtype": str(q.dtype),
        "q_shape": list(q.shape),
        "kv_cache_sets": pool_state["sets"],
        "ok": fused_calls["n"] == 1 and dyn_count["n"] == 0
        and q.dtype == torch.bfloat16,
    }


# ---------------------------------------------------------------------------
# profile + timing
# ---------------------------------------------------------------------------
def chain_old(qkv, gamma, eps, layer):
    q = qkv.split([Q_LORA, QKV_WIDTH - Q_LORA], dim=-1)[0]
    q = torch_npu.npu_rms_norm(q, gamma, eps)[0]
    out = layer._mx_kernel.apply(layer, q, None)
    return out.view(-1, out.shape[-1])


def chain_new(qkv, norm, layer):
    q = qkv.split([Q_LORA, QKV_WIDTH - Q_LORA], dim=-1)[0]
    out = qlm.forward_q_latent_rmsnorm_mxquant(
        norm, layer, q, is_compiling=lambda: False, is_capturing=lambda: False
    )
    return out.view(-1, out.shape[-1])


def time_alternating(rows, seed, eps, device, out_features, warmup, iters):
    _, qkv = make_input(rows, seed)
    qkv = qkv.to(device)
    gamma, _ = make_gamma(seed, eps)
    gamma = gamma.to(device)
    norm = _norm_stub(gamma, eps)
    layer = make_layer(rows, out_features, seed, device)

    def run(fn):
        fn()
        torch.npu.synchronize()

    for _ in range(warmup):
        run(lambda: chain_old(qkv, gamma, eps, layer))
        run(lambda: chain_new(qkv, norm, layer))

    old_ms, new_ms = [], []
    keep = []
    for _ in range(iters):
        for fn, bucket in (
            (lambda: chain_old(qkv, gamma, eps, layer), old_ms),
            (lambda: chain_new(qkv, norm, layer), new_ms),
        ):
            start = torch.npu.Event(enable_timing=True)
            end = torch.npu.Event(enable_timing=True)
            start.record()
            keep.append(fn())
            end.record()
            torch.npu.synchronize()
            bucket.append(start.elapsed_time(end))
    return {
        "rows": rows,
        "warmup": warmup,
        "iters": iters,
        "old_ms": {
            "median": statistics.median(old_ms),
            "mean": statistics.mean(old_ms),
            "p95": sorted(old_ms)[max(0, int(0.95 * len(old_ms)) - 1)],
        },
        "new_ms": {
            "median": statistics.median(new_ms),
            "mean": statistics.mean(new_ms),
            "p95": sorted(new_ms)[max(0, int(0.95 * len(new_ms)) - 1)],
        },
        "net_gain_ms_median": statistics.median(old_ms) - statistics.median(new_ms),
        "keep_outputs": len(keep),
    }


def summarize_trace(path):
    try:
        with open(path) as f:
            data = json.load(f)
        events = data.get("traceEvents", [])
        kernels = [
            e
            for e in events
            if e.get("ph") == "X"
            and (
                "kernel" in str(e.get("cat", "")).lower()
                or "npu" in str(e.get("cat", "")).lower()
            )
        ]
        by_name = {}
        for e in kernels:
            item = by_name.setdefault(e.get("name", "?"), {"count": 0, "dur_us": 0.0})
            item["count"] += 1
            item["dur_us"] += float(e.get("dur", 0.0))
        return {
            "path": path,
            "kernel_event_count": len(kernels),
            "kernels": dict(
                sorted(by_name.items(), key=lambda kv: kv[1]["dur_us"], reverse=True)
            ),
        }
    except Exception as exc:  # pragma: no cover - device-only
        return {"path": path, "error": repr(exc)}


def profile_chains(rows, seed, eps, device, out_features, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    _, qkv = make_input(rows, seed)
    qkv = qkv.to(device)
    gamma, _ = make_gamma(seed, eps)
    gamma = gamma.to(device)
    norm = _norm_stub(gamma, eps)
    layer = make_layer(rows, out_features, seed, device)
    results = {}
    for label, fn in (
        ("old_slice_rmsnorm_dynquant_qb", lambda: chain_old(qkv, gamma, eps, layer)),
        ("new_slice_fused_qb", lambda: chain_new(qkv, norm, layer)),
    ):
        fn()
        torch.npu.synchronize()
        trace_path = os.path.join(out_dir, f"trace_{label}.json")
        try:
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                record_shapes=False,
            ) as prof:
                fn()
                torch.npu.synchronize()
            try:
                prof.export_chrome_trace(trace_path)
            except Exception:
                trace_path = None
            table = ""
            try:
                table = prof.key_averages().table(
                    sort_by="self_npu_time_total", row_limit=50
                )
            except Exception as exc:
                table = f"key_averages unavailable: {exc!r}"
            results[label] = {
                "trace": summarize_trace(trace_path) if trace_path else None,
                "table": table,
            }
        except Exception as exc:  # pragma: no cover - device-only
            results[label] = {"error": repr(exc), "traceback": traceback.format_exc()}
    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="/mnt/sfs_turbo/hxd/rmsnorm-mxquant-fusion-20261102/results/npu",
    )
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if not torch.npu.is_available():
        print("REFUSED: torch.npu.is_available() is False")
        return 1
    device = "npu:0"
    os.environ.setdefault("SGLANG_NPU_Q_RMSNORM_MXQUANT", "1")
    qlm._FUSION_ENABLED = True
    install_fused_recorder()

    result = {
        "grant_env": os.environ.get("QNORM_MXQUANT_NPU_GRANTED"),
        "device": torch.npu.get_device_name(0),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "flag": qlm.fusion_enabled(),
        "note": (
            "Historical ~0.70ms Slice + ~0.99ms Norm + ~0.67ms Quant "
            "(~2.37ms/47 layers) is the pre-existing cost, not a measured gain "
            "of this harness; only the tables below are measured here."
        ),
        "correctness": [],
        "prepare_route": [],
        "timing": [],
    }

    try:
        for rows in TOKENS:
            seeds = (1234,) if rows > 2048 else (1234, 5678, 91011)
            for seed in seeds:
                try:
                    result["correctness"].append(
                        run_case(rows, seed, 1e-6, device, DEFAULT_OUT)
                    )
                except Exception as exc:
                    result["correctness"].append(
                        {
                            "rows": rows,
                            "seed": seed,
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
        # Extreme values / boundary case and an epsilon variation.
        for extra in (
            dict(rows=128, seed=4242, extreme=True),
            dict(rows=128, seed=4243, eps=1e-5),
        ):
            try:
                result["correctness"].append(
                    run_case(
                        extra["rows"],
                        extra["seed"],
                        extra.get("eps", 1e-6),
                        device,
                        DEFAULT_OUT,
                        extreme=extra.get("extreme", False),
                    )
                )
            except Exception as exc:
                result["correctness"].append(
                    {"rows": extra["rows"], "seed": extra["seed"], "error": repr(exc)}
                )

        for rows in (128, 3787):
            try:
                result["prepare_route"].append(
                    run_prepare_route(rows, 1234, 1e-6, device, DEFAULT_OUT)
                )
            except Exception as exc:
                result["prepare_route"].append(
                    {
                        "rows": rows,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

        for rows in (1, 128, 3787, 15148):
            warmup, iters = (20, 60) if rows <= 3787 else (10, 30)
            try:
                result["timing"].append(
                    time_alternating(
                        rows, 1234, 1e-6, device, DEFAULT_OUT, warmup, iters
                    )
                )
            except Exception as exc:
                result["timing"].append(
                    {"rows": rows, "error": repr(exc), "traceback": traceback.format_exc()}
                )

        result["profiling"] = profile_chains(
            3787, 1234, 1e-6, device, DEFAULT_OUT, os.path.join(args.out_dir, "profiling")
        )
    finally:
        path = os.path.join(args.out_dir, "qnorm_mxquant_npu_results.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=1)
        print(json.dumps(result, indent=1))
        print(f"\nRESULTS: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
