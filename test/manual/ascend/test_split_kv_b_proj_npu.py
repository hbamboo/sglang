"""NPU correctness + actual prepare-route tests for SGLANG_NPU_SPLIT_KV_B_PROJ.

Run via the authorized harness (physical NPU5 only):

    ASCEND_RT_VISIBLE_DEVICES=5 SGLANG_NPU_SPLIT_KV_B_PROJ=1 python3 \
        test/manual/ascend/test_split_kv_b_proj_npu.py

Compares, for every case:
  * old ``forward_mha_prepare_npu`` (one matmul + strided V copy) vs new
    (two matmuls + contiguous K_nope/V), full outputs;
  * both against an independent CPU FP32 per-head einsum reference built from
    the recorded post-norm kv_a and the original packed weight.
Also exercises flag/default-off, bias, TP local heads, strided kv_a input,
identifiable weights, in-place reload/dtype invalidation and unsupported
routes.
"""

import os
import sys

FRAMEWORK = "/mnt/sfs_turbo/hxd/split-kv-matmul-20261102/framework"
sys.path.insert(0, f"{FRAMEWORK}/python")
sys.path.insert(0, f"{FRAMEWORK}/test/manual/ascend")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "5")

import math

import torch
import torch_npu

import split_kv_test_utils as tu

tu.assert_npu5_idle()
torch.npu.set_device(0)

from sglang.srt.hardware_backend.npu.modules import (
    deepseek_v2_attention_mla_npu as attn_mod,
)
from sglang.srt.hardware_backend.npu.modules import split_kv_b_proj as skv
from sglang.srt.layers.linear import ColumnParallelLinear

DTYPE_NAMES = {torch.bfloat16: "bf16", torch.float16: "fp16"}
TOL = {torch.bfloat16: (0.5, 2e-2), torch.float16: (0.05, 1e-2)}
DEV = "npu:0"


class DummyPool:
    prefill_expanded_geometry = None

    def set_kv_buffer(self, *args, **kwargs):
        return None

    def get_kv_buffer(self, *args, **kwargs):
        raise AssertionError("unexpected get_kv_buffer in MHA prepare")


class FakeForwardMode:
    def __init__(self, extend=False):
        self._extend = extend

    def is_extend(self):
        return self._extend


class FakeForwardBatch:
    def __init__(self, tokens, extend=False):
        self.forward_mode = FakeForwardMode(extend)
        self.out_cache_loc = torch.zeros(tokens, dtype=torch.int64, device=DEV)
        self.extend_prefix_lens_cpu = [1] if extend else []


class RecordingRMSNorm(torch.nn.Module):
    def __init__(self, dim, dtype, strided=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=dtype, device=DEV))
        self.variance_epsilon = 1e-6
        self.strided = strided
        self.last_output = None

    def forward(self, x):
        xf = x.float()
        y = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
        y = y.to(x.dtype) * self.weight
        if self.strided:
            buf = y.new_empty((y.shape[0], y.shape[1] * 2))
            buf[:, : y.shape[1]].copy_(y)
            y = buf[:, : y.shape[1]]
        self.last_output = y.detach()
        return y


class FakeRotary(torch.nn.Module):
    """Deterministic stand-in applying the same transform to both routes."""

    def __init__(self, dim, dtype):
        super().__init__()
        self.register_buffer("q_scale", torch.linspace(0.5, 0.9, dim, dtype=dtype, device=DEV))
        self.register_buffer("k_scale", torch.linspace(0.2, 0.6, dim, dtype=dtype, device=DEV))

    def forward(self, positions, q_pe, k_pe):
        return q_pe * self.q_scale, k_pe * self.k_scale


class FakeMLA(torch.nn.Module):
    def __init__(self, dtype, bias, local_heads, global_heads, tp_rank, tp_size,
                 dk, dv, dr, k_dim, hidden, strided):
        super().__init__()
        total = dk + dv
        self.q_lora_rank = None
        self.use_dsa = False
        self.use_deepseek_yarn_rope = False
        self.layer_id = 0
        self.num_local_heads = local_heads
        self.num_heads = global_heads
        self.qk_nope_head_dim = dk
        self.qk_rope_head_dim = dr
        self.qk_head_dim = dk + dr
        self.v_head_dim = dv
        self.kv_lora_rank = k_dim
        self.q_proj = ColumnParallelLinear(
            hidden, local_heads * (dk + dr), bias=False, tp_rank=0, tp_size=1,
            params_dtype=dtype,
        )
        self.kv_a_proj_with_mqa = ColumnParallelLinear(
            hidden, k_dim + dr, bias=False, tp_rank=0, tp_size=1, params_dtype=dtype
        )
        self.kv_b_proj = ColumnParallelLinear(
            k_dim, global_heads * total, bias=bias, tp_rank=tp_rank,
            tp_size=tp_size, params_dtype=dtype,
        )
        self.kv_a_layernorm = RecordingRMSNorm(k_dim, dtype, strided=strided)
        self.rotary_emb = FakeRotary(dr, dtype)
        self.last_k_nope = None
        self.last_k_pe = None

    def _concat_and_cast_mha_k(self, k_nope, k_pe, forward_batch):
        self.last_k_nope = k_nope
        self.last_k_pe = k_pe
        k = k_nope.new_empty(
            (k_nope.shape[0], self.num_local_heads, self.qk_head_dim)
        )
        k[..., : self.qk_nope_head_dim] = k_nope
        k[..., self.qk_nope_head_dim :] = k_pe
        return k


def npu_random(shape, dtype, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g, dtype=torch.float32) * scale).to(dtype).to(DEV)


def copy_weights(module, seed, scale=1.0):
    with torch.no_grad():
        module.weight.copy_(npu_random(tuple(module.weight.shape), module.weight.dtype, seed, scale))
        if module.bias is not None:
            module.bias.copy_(npu_random(tuple(module.bias.shape), module.bias.dtype, seed + 1, scale))


def build_case(
    dtype,
    bias,
    tokens,
    *,
    local_heads=20,
    global_heads=20,
    tp_rank=0,
    tp_size=1,
    strided=False,
    identifiable=False,
    seed=1234,
):
    if identifiable:
        dk, dv, dr, k_dim, hidden = 8, 12, 4, 32, 32
        global_heads = local_heads
        total = dk + dv
        g = torch.Generator().manual_seed(seed)
        weight = torch.randn(local_heads * total, k_dim, generator=g, dtype=torch.float32)
        m = FakeMLA(dtype, bias, local_heads, global_heads, tp_rank, tp_size,
                    dk, dv, dr, k_dim, hidden, strided)
        m = m.to(DEV)
        copy_weights(m.q_proj, seed + 11, scale=0.05)
        with torch.no_grad():
            m.kv_b_proj.weight.copy_(weight.to(dtype).to(DEV))
            if bias:
                b = torch.randn(local_heads * total, generator=g, dtype=torch.float32)
                m.kv_b_proj.bias.copy_(b.to(dtype).to(DEV))
            m.kv_a_proj_with_mqa.weight.zero_()
            # latent_cache[:, :k_dim] == hidden_states; rope part stays zero.
            m.kv_a_proj_with_mqa.weight[:k_dim, :k_dim].copy_(
                torch.eye(k_dim, dtype=dtype, device=DEV)
            )
        x = (torch.eye(k_dim, dtype=torch.float32) * math.sqrt(k_dim)).to(dtype).to(DEV)
        tokens = k_dim
    else:
        dk, dv, dr, k_dim, hidden = 192, 256, 64, 512, 512
        m = FakeMLA(dtype, bias, local_heads, global_heads, tp_rank, tp_size,
                    dk, dv, dr, k_dim, hidden, strided)
        m = m.to(DEV)
        copy_weights(m.q_proj, seed + 11, scale=0.05)
        copy_weights(m.kv_a_proj_with_mqa, seed + 22, scale=0.1)
        copy_weights(m.kv_b_proj, seed + 33)
        x = npu_random((tokens, hidden), dtype, seed + 44)
    positions = torch.arange(tokens, dtype=torch.float32, device=DEV)
    fb = FakeForwardBatch(tokens)
    return m, x, positions, fb


def run_prepare(m, x, positions, fb):
    return attn_mod.forward_mha_prepare_npu(m, positions, x, fb, None, None)


def actual_kv_a(m, x):
    latent = m.kv_a_proj_with_mqa(x)[0]
    return m.kv_a_layernorm(latent[..., : m.kv_lora_rank])


def sampled_rows(tokens):
    if tokens <= 129:
        return list(range(tokens))
    head = list(range(0, 32))
    tail = list(range(tokens - 32, tokens))
    step = max(1, tokens // 64)
    mid = list(range(32, tokens - 32, step))[:64]
    return sorted(set(head + tail + mid))


def cpu_reference(kv_a_cpu, weight_cpu, bias_cpu, dk, dv, heads, rows):
    if not rows:
        return None, None
    x = kv_a_cpu[rows].float()
    total = dk + dv
    w3 = weight_cpu.float().view(heads, total, -1)
    gk = torch.einsum("tk,hdk->thd", x, w3[:, :dk, :])
    gv = torch.einsum("tk,hdk->thd", x, w3[:, dk:, :])
    if bias_cpu is not None:
        b3 = bias_cpu.float().view(heads, total)
        gk = gk + b3[:, :dk].unsqueeze(0)
        gv = gv + b3[:, dk:].unsqueeze(0)
    return gk, gv


def case_stats(dtype, tokens, label, old, new, m, kv_a_cpu, rows, metrics, strides):
    atol, rtol = TOL[dtype]
    heads = m.num_local_heads
    dk, dv, dr = m.qk_nope_head_dim, m.v_head_dim, m.qk_rope_head_dim
    total = dk + dv
    q_old, k_old, v_old, k_nope_old, k_pe_old = old
    q_new, k_new, v_new, k_nope_new, k_pe_new = new
    kn_old_stride, kn_new_stride, v_new_stride = strides

    rec = {"dtype": DTYPE_NAMES[dtype], "T": tokens, "label": label}
    rec["old_vs_new_q"] = tu.close_stats(q_new, q_old, atol, rtol)
    rec["old_vs_new_k"] = tu.close_stats(k_new, k_old, atol, rtol)
    rec["old_vs_new_v"] = tu.close_stats(v_new, v_old, atol, rtol)
    rec["old_vs_new_k_nope"] = tu.close_stats(k_nope_new, k_nope_old, atol, rtol)
    rec["new_k_nope_contiguous"] = bool(k_nope_new.is_contiguous())
    rec["new_v_contiguous"] = bool(v_new.is_contiguous())
    rec["new_v_stride"] = list(v_new_stride)
    rec["new_k_nope_stride"] = list(kn_new_stride)
    rec["old_k_nope_stride"] = list(kn_old_stride)
    rec["old_k_nope_contiguous"] = bool(k_nope_old.is_contiguous())
    rec["new_state_present"] = skv.split_kv_b_proj_state(m.kv_b_proj) is not None
    rec["k_pe_old_vs_new"] = tu.close_stats(k_pe_new, k_pe_old, atol, rtol)

    for name, stats in rec.items():
        if isinstance(stats, dict):
            if stats["numel"] and (stats["bad"] or not stats["finite"]):
                raise AssertionError(f"{label} {name} mismatch {stats}")

    if tokens and heads > 1:
        # A per-head slice of the packed [T,H,Dk+Dv] output strides over the
        # full head width; is_contiguous() is vacuously true for T==1.
        assert kn_old_stride[-2] == dk + dv, (label, kn_old_stride)
        assert kn_new_stride[-2] == dk, (label, kn_new_stride)
        assert k_nope_new.is_contiguous(), f"{label}: new K_nope not contiguous"
    assert v_new.is_contiguous(), f"{label}: new V not contiguous"
    assert tuple(v_new_stride) == (heads * dv, dv, 1), (label, v_new_stride)

    w_cpu = m.kv_b_proj.weight.detach().float().cpu()
    b_cpu = m.kv_b_proj.bias.detach().float().cpu() if m.kv_b_proj.bias is not None else None
    gk, gv = cpu_reference(kv_a_cpu, w_cpu, b_cpu, dk, dv, heads, rows)
    if gk is None:
        rec["golden_new_k_nope"] = {"numel": 0, "bad": 0, "finite": True,
                                    "max_abs": 0.0, "mean_abs": 0.0, "max_rel": 0.0}
        rec["golden_new_v"] = dict(rec["golden_new_k_nope"])
        rec["golden_old_k_nope"] = dict(rec["golden_new_k_nope"])
        rec["golden_old_v"] = dict(rec["golden_new_k_nope"])
    else:
        kp_rows = k_pe_new.detach().float().cpu()[rows]
        gk_full = torch.cat([gk, kp_rows.expand(-1, heads, -1)], dim=-1)
        rec["golden_new_k_nope"] = tu.close_stats(k_nope_new.detach()[rows], gk, atol, rtol)
        rec["golden_new_v"] = tu.close_stats(v_new.detach()[rows], gv, atol, rtol)
        rec["golden_new_k"] = tu.close_stats(k_new.detach()[rows], gk_full, atol, rtol)
        rec["golden_old_k_nope"] = tu.close_stats(k_nope_old.detach()[rows], gk, atol, rtol)
        rec["golden_old_v"] = tu.close_stats(v_old.detach()[rows], gv, atol, rtol)
        for name in ("golden_new_k_nope", "golden_new_v", "golden_new_k",
                     "golden_old_k_nope", "golden_old_v"):
            st = rec[name]
            if st["numel"] and (st["bad"] or not st["finite"]):
                raise AssertionError(f"{label} {name} golden mismatch {st}")
    metrics[label] = rec
    print(
        f"PASS {label} dtype={DTYPE_NAMES[dtype]} T={tokens} heads={heads} "
        f"kmax={rec['old_vs_new_k']['max_abs']:.3g} vmax={rec['old_vs_new_v']['max_abs']:.3g} "
        f"gnew_k={rec['golden_new_k_nope']['max_abs']:.3g} "
        f"gnew_v={rec['golden_new_v']['max_abs']:.3g} "
        f"contig(K_new={rec['new_k_nope_contiguous']},V={rec['new_v_contiguous']},"
        f"K_old={rec['old_k_nope_contiguous']})"
    )


def run_case(dtype, bias, tokens, label, metrics, **kwargs):
    eager_env = os.environ.get("SGLANG_NPU_SPLIT_KV_B_PROJ")
    os.environ["SGLANG_NPU_SPLIT_KV_B_PROJ"] = "1"
    try:
        m, x, positions, fb = build_case(dtype, bias, tokens, **kwargs)
        w_before = m.kv_b_proj.weight.detach().clone()
        x_before = x.detach().clone()

        # Old path exactly: no prepared state -> one matmul + V contiguous.
        skv.invalidate_split_kv_b_proj(m.kv_b_proj)
        q0, k0, v0, _ = run_prepare(m, x, positions, fb)
        kv_a0 = m.kv_a_layernorm.last_output.detach().clone()
        kn0_stride = tuple(m.last_k_nope.stride())
        kn0 = m.last_k_nope.detach().clone()
        kp0 = m.last_k_pe.detach().clone()

        # New path through the actual prepare and env-gated prepared state.
        state, reason = skv.prepare_split_kv_b_proj(m, m.kv_b_proj)
        assert state is not None and reason in ("prepared", "already prepared"), reason
        q1, k1, v1, _ = run_prepare(m, x, positions, fb)
        kv_a1 = m.kv_a_layernorm.last_output.detach().clone()
        kn1_stride = tuple(m.last_k_nope.stride())
        kn1 = m.last_k_nope.detach().clone()
        kp1 = m.last_k_pe.detach().clone()
        v1_stride = tuple(v1.stride())

        assert torch.equal(w_before, m.kv_b_proj.weight.detach()), f"{label}: weight mutated"
        assert torch.equal(x_before, x), f"{label}: input mutated"
        assert tu.close_stats(kv_a1, kv_a0, 0.0, 0.0)["bad"] == 0, f"{label}: kv_a drift"

        rows = sampled_rows(tokens)
        kv_a_cpu = kv_a1.float().cpu()
        case_stats(
            dtype, tokens, label,
            (q0, k0, v0, kn0, kp0),
            (q1, k1, v1, kn1, kp1),
            m, kv_a_cpu, rows, metrics,
            (kn0_stride, kn1_stride, v1_stride),
        )
    finally:
        if eager_env is None:
            os.environ.pop("SGLANG_NPU_SPLIT_KV_B_PROJ", None)
        else:
            os.environ["SGLANG_NPU_SPLIT_KV_B_PROJ"] = eager_env


def route_checks(metrics):
    dtype = torch.bfloat16
    label = "routes"

    # Flag off: no buffers prepared, actual prepare equals old path.
    os.environ["SGLANG_NPU_SPLIT_KV_B_PROJ"] = "0"
    m, x, positions, fb = build_case(dtype, False, 31)
    state, reason = skv.prepare_split_kv_b_proj(m, m.kv_b_proj)
    assert state is None and reason == "env disabled", reason
    assert skv.split_kv_b_proj_state(m.kv_b_proj) is None
    assert not any(name.startswith("_split_kv_b_proj_w") for name in m.kv_b_proj._buffers)
    q, k, v, _ = run_prepare(m, x, positions, fb)
    assert not m.last_k_nope.is_contiguous()
    metrics["flag_off"] = {"reason": reason, "k_nope_contiguous": False}

    # Flag on, but helper route must bypass under capture/compile.
    os.environ["SGLANG_NPU_SPLIT_KV_B_PROJ"] = "1"
    state, reason = skv.prepare_split_kv_b_proj(m, m.kv_b_proj)
    assert state is not None
    kv_a = actual_kv_a(m, x)
    assert skv.maybe_forward_split_kv_b_proj(m, m.kv_b_proj, kv_a, fb) is not None
    orig_capture = skv._capture_or_compile_active
    skv._capture_or_compile_active = lambda: True
    try:
        assert skv.maybe_forward_split_kv_b_proj(m, m.kv_b_proj, kv_a, fb) is None
    finally:
        skv._capture_or_compile_active = orig_capture
    metrics["capture_fallback"] = {"route_none": True}

    # Expanded KV possible: pool geometry present + extend batch.
    import sglang.srt.model_executor.forward_context as fctx
    orig_pool = fctx.get_token_to_kv_pool
    pool = DummyPool()
    pool.prefill_expanded_geometry = object()
    fctx.get_token_to_kv_pool = lambda: pool
    try:
        fb_ext = FakeForwardBatch(31, extend=True)
        assert skv._expanded_kv_may_fuse(fb_ext) is True
        assert skv.maybe_forward_split_kv_b_proj(m, m.kv_b_proj, kv_a, fb_ext) is None
    finally:
        fctx.get_token_to_kv_pool = orig_pool
    metrics["expanded_fallback"] = {"route_none": True}

    # SP active with TP>1 must use ColumnParallelLinear's own forward.
    m_tp, x_tp, pos_tp, fb_tp = build_case(
        dtype, False, 31, local_heads=4, global_heads=16, tp_rank=2, tp_size=4
    )
    state, reason = skv.prepare_split_kv_b_proj(m_tp, m_tp.kv_b_proj)
    assert state is not None and state.tp_size == 4
    kv_a_tp = actual_kv_a(m_tp, x_tp)
    import sglang.srt.layers.communicator as comm
    orig_get_forward = comm.get_forward

    class Fwd:
        sp_active = True

    comm.get_forward = lambda: Fwd()
    try:
        assert skv.maybe_forward_split_kv_b_proj(m_tp, m_tp.kv_b_proj, kv_a_tp, fb_tp) is None
    finally:
        comm.get_forward = orig_get_forward
    metrics["sp_fallback"] = {"route_none": True}

    # Unsupported modules/graphs: quant method, DSA, gather/skip bias.
    class Fp8LinearMethod:
        pass

    m_q = m_tp.kv_b_proj
    orig_qm = m_q.quant_method
    m_q.quant_method = Fp8LinearMethod()
    try:
        state, reason = skv.prepare_split_kv_b_proj(m_tp, m_q)
        assert state is None and "quant_method" in reason, reason
        assert skv.maybe_forward_split_kv_b_proj(m_tp, m_q, kv_a_tp, fb_tp) is None
    finally:
        m_q.quant_method = orig_qm

    m_dsa, x_dsa, pos_dsa, fb_dsa = build_case(dtype, False, 31)
    m_dsa.use_dsa = True
    state, reason = skv.prepare_split_kv_b_proj(m_dsa, m_dsa.kv_b_proj)
    assert state is None and reason == "DSA", reason
    assert skv.maybe_forward_split_kv_b_proj(m_dsa, m_dsa.kv_b_proj, x_dsa, fb_dsa) is None

    m_gather, *_ = build_case(dtype, False, 31)
    m_gather.kv_b_proj.gather_output = True
    state, reason = skv.prepare_split_kv_b_proj(m_gather, m_gather.kv_b_proj)
    assert state is None and reason == "gather_output=True", reason

    metrics["unsupported"] = {
        "quant": "quant_method",
        "dsa": "DSA",
        "gather_output": True,
    }
    print("PASS route checks: flag off, capture, expanded KV, SP, unsupported")

    # In-place reload must invalidate immediately, then rebuild correctly.
    m_r, x_r, pos_r, fb_r = build_case(dtype, False, 31)
    state, _ = skv.prepare_split_kv_b_proj(m_r, m_r.kv_b_proj)
    assert state is not None
    q_r, k_r, v_r, _ = run_prepare(m_r, x_r, pos_r, fb_r)
    kv_a_r = m_r.kv_a_layernorm.last_output.detach().clone()
    assert skv.maybe_forward_split_kv_b_proj(m_r, m_r.kv_b_proj, kv_a_r, fb_r) is not None
    with torch.no_grad():
        m_r.kv_b_proj.weight.add_(0.25)
    assert skv.maybe_forward_split_kv_b_proj(m_r, m_r.kv_b_proj, kv_a_r, fb_r) is None
    old_out = m_r.kv_b_proj(kv_a_r)[0].view(-1, m_r.num_local_heads, 448)
    state, reason = skv.prepare_split_kv_b_proj(m_r, m_r.kv_b_proj)
    assert state is not None
    q_new, k_new, v_new, _ = run_prepare(m_r, x_r, pos_r, fb_r)
    reload_k_stats = tu.close_stats(m_r.last_k_nope.detach(), old_out[..., :192], 0.05, 2e-2)
    reload_v_stats = tu.close_stats(v_new, old_out[..., 192:], 0.05, 2e-2)
    print("RELOAD_STATS", reload_k_stats, reload_v_stats)
    assert reload_k_stats["bad"] == 0, reload_k_stats
    assert reload_v_stats["bad"] == 0, reload_v_stats
    metrics["reload"] = {"rebuilt": True}

    # Device/dtype migration: stale buffers are never used.
    m_d, x_d, pos_d, fb_d = build_case(dtype, False, 31)
    state, _ = skv.prepare_split_kv_b_proj(m_d, m_d.kv_b_proj)
    kv_a_d = actual_kv_a(m_d, x_d)
    assert skv.maybe_forward_split_kv_b_proj(m_d, m_d.kv_b_proj, kv_a_d, fb_d) is not None
    m_d.kv_b_proj.weight = torch.nn.Parameter(m_d.kv_b_proj.weight.detach().clone())
    assert skv.maybe_forward_split_kv_b_proj(m_d, m_d.kv_b_proj, kv_a_d, fb_d) is None
    metrics["retarget_fallback"] = {"route_none": True}
    print("PASS reload invalidation and storage-replacement fallback")


def loader_integration(metrics):
    """Exercise the real DeepseekV2WeightLoaderMixin.post_load_weights hook."""
    from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
        DeepseekV2WeightLoaderMixin,
    )

    class LoaderAttn(torch.nn.Module):
        def __init__(self, dtype, seed):
            super().__init__()
            self.num_local_heads = 4
            self.qk_nope_head_dim = 8
            self.v_head_dim = 12
            self.kv_lora_rank = 32
            self.use_dsa = False
            self.kv_b_proj = ColumnParallelLinear(
                32, 4 * 20, bias=False, tp_rank=0, tp_size=1, params_dtype=dtype
            )
            self.w_kc = None
            self.w_vc = None
            self.seed = seed

        def fill(self):
            copy_weights(self.kv_b_proj, self.seed, scale=0.2)

    class Layer(torch.nn.Module):
        def __init__(self, attn):
            super().__init__()
            self.self_attn = attn

    class Model(torch.nn.Module):
        def __init__(self, attns):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer(a) for a in attns])
            self.start_layer = 0
            self.end_layer = len(attns)

    class Config:
        num_hidden_layers = 3

    class Loader(DeepseekV2WeightLoaderMixin):
        pass

    attns = [LoaderAttn(torch.bfloat16, 5000 + i).to(DEV) for i in range(3)]
    for attn in attns:
        attn.fill()
    loader = Loader()
    loader.model = Model(attns)
    loader.config = Config()
    loader.quant_config = None

    os.environ["SGLANG_NPU_SPLIT_KV_B_PROJ"] = "1"
    loader.post_load_weights(weight_names=None)
    assert all(skv.split_kv_b_proj_state(a.kv_b_proj) is not None for a in attns)
    assert all(a.w_kc is not None and a.w_vc is not None for a in attns)

    # Partial reload naming only layer 1 keeps that layer and drops the rest.
    loader.post_load_weights(
        weight_names=["model.layers.1.self_attn.kv_b_proj.weight"]
    )
    assert skv.split_kv_b_proj_state(attns[0].kv_b_proj) is None
    assert skv.split_kv_b_proj_state(attns[1].kv_b_proj) is not None
    assert skv.split_kv_b_proj_state(attns[2].kv_b_proj) is None

    # In-place reload through the loader rebuilds from the new weight.
    with torch.no_grad():
        attns[1].kv_b_proj.weight.add_(0.5)
    stale = skv.split_kv_b_proj_state(attns[1].kv_b_proj)
    loader.post_load_weights(
        weight_names=["model.layers.1.self_attn.kv_b_proj.weight"]
    )
    fresh = skv.split_kv_b_proj_state(attns[1].kv_b_proj)
    assert fresh is not None and fresh.weight_version != stale.weight_version
    assert fresh is not stale

    # Unrelated partial reload invalidates every derived split.
    loader.post_load_weights(weight_names=["model.layers.0.mlp.weight"])
    assert all(skv.split_kv_b_proj_state(a.kv_b_proj) is None for a in attns)

    metrics["loader_integration"] = {
        "prepared_all": True,
        "partial_reload_kept_layer1": True,
        "inplace_reload_rebuilt": True,
        "unrelated_reload_invalidated": True,
    }
    print("PASS real post_load_weights integration and partial-reload invalidation")


def main():
    os.environ["SGLANG_NPU_SPLIT_KV_B_PROJ"] = "1"
    attn_mod.get_token_to_kv_pool = lambda: DummyPool()
    attn_mod.can_fuse_expanded_kv = lambda *args, **kwargs: False
    import sglang.srt.model_executor.forward_context as fctx

    fctx.get_token_to_kv_pool = lambda: DummyPool()

    metrics = {}

    def run(metrics):
        for dtype in (torch.bfloat16, torch.float16):
            for tokens in (0, 1, 31, 127, 128, 129, 3787, 15148):
                run_case(
                    dtype, False, tokens,
                    f"main_{DTYPE_NAMES[dtype]}_T{tokens}",
                    metrics,
                )
        run_case(torch.bfloat16, True, 129, "bias_bf16_T129", metrics)
        run_case(torch.bfloat16, True, 127, "bias_bf16_T127", metrics)
        run_case(torch.bfloat16, False, 127, "tp4_local4_bf16_T127", metrics,
                 local_heads=4, global_heads=16, tp_rank=2, tp_size=4)
        run_case(torch.float16, False, 31, "tp4_local4_fp16_T31", metrics,
                 local_heads=4, global_heads=16, tp_rank=1, tp_size=4)
        run_case(torch.bfloat16, False, 31, "one_local_head_bf16_T31", metrics,
                 local_heads=1, global_heads=1)
        run_case(torch.bfloat16, False, 129, "strided_kv_a_bf16_T129", metrics,
                 strided=True)
        run_case(torch.float16, False, 129, "strided_kv_a_fp16_T129", metrics,
                 strided=True)
        run_case(torch.bfloat16, True, 32, "identifiable_bias_bf16", metrics,
                 local_heads=4, identifiable=True)
        run_case(torch.float16, False, 32, "identifiable_fp16", metrics,
                 local_heads=4, identifiable=True)
        route_checks(metrics)
        loader_integration(metrics)

    tu.run_entrypoint("test_split_kv_b_proj_npu", run)


if __name__ == "__main__":
    main()
