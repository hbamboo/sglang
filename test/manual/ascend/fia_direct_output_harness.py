#!/usr/bin/env python3
"""Self-validation harness for SGLANG_NPU_FIA_DIRECT_OUTPUT.

Exercises the real AscendAttnBackend.forward_extend fresh/prefix FIA BSND
loops on one explicitly selected NPU.  Sections are independent:

  * correctness  - init/flag gating, old-vs-new full outputs (bitwise), CPU
                   FP32 causal goldens, zero-length/ragged/q_len1, prefix,
                   repeated calls, non-zero output storage offset sentinels.
  * timing       - alternating baseline/candidate NPU-event timings and wall
                   time, with every allocation and the full Python/native
                   call inside the measured interval.
  * profile      - separate baseline/candidate torch_npu profiles; parses all
                   device kernels and proves the candidate has no TensorMove
                   / InplaceCopy / ViewCopy / Concat style assembly.

Run on physical NPU 6 (logical 0), with no service on that device:

  ASCEND_RT_VISIBLE_DEVICES=6 python fia_direct_output_harness.py \
      --outdir /mnt/sfs_turbo/hxd/attention-output-copy-20261102/artifacts

The harness aborts (and exits non-zero) on any failed required case; it does
not catch exceptions and continue.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import statistics
import sys
import time
import types
from types import SimpleNamespace

os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "6")

WT = os.environ.get(
    "FIA_DIRECT_WORKTREE",
    "/mnt/sfs_turbo/hxd/attention-output-copy-20261102/framework",
)
sys.path.insert(0, WT + "/python")

import torch
import torch_npu
import numpy as np

import sglang.srt.hardware_backend.npu.attention.ascend_backend as ab

# ---- harness-only model/pool shims (no real model is loaded) ---------------
ab.get_flags = lambda: SimpleNamespace(
    capture=SimpleNamespace(enable_torch_compile=False)
)
ab.get_spec = lambda: SimpleNamespace(speculative_num_draft_tokens=None)
ab.get_parallel = lambda: SimpleNamespace(attn_tp_size=1)
ab.AscendTorchNativeAttnBackend = lambda: None
ab.is_mla_preprocess_for_batch = lambda _fb: False


class _DllmConfig:
    from_server_args = staticmethod(lambda _args: None)


ab.DllmConfig = _DllmConfig

H = HKV = 20
D = 256
QK_NOPE = 192
QK_ROPE = 64
assert QK_NOPE + QK_ROPE == D
SCALE = 1.0 / math.sqrt(D)


class FakePool:
    prefill_expanded_geometry = None
    req_to_token = object()

    def __init__(self):
        self.k = None
        self.v = None

    def get_key_buffer(self, _layer_id):
        return self.k

    def get_value_buffer(self, _layer_id):
        return self.v


class ModelConfig:
    attention_arch = ab.AttentionArch.MLA
    dtype = torch.bfloat16
    kv_lora_rank = 512
    qk_rope_head_dim = QK_ROPE
    qk_nope_head_dim = QK_NOPE
    context_len = 32768
    num_attention_heads = H
    hf_config = SimpleNamespace(architectures=["Glm4MoeForCausalLM"])


def make_backend(flag: str):
    os.environ["SGLANG_NPU_FIA_DIRECT_OUTPUT"] = flag
    pool = FakePool()
    mr = SimpleNamespace(
        device="npu:0",
        dtype=torch.bfloat16,
        page_size=128,
        model_config=ModelConfig(),
        req_to_token_pool=FakePool(),
        token_to_kv_pool=pool,
        is_hybrid_swa=False,
        is_draft_worker=False,
        server_args=None,
        ps=SimpleNamespace(attn_cp_size=1),
    )
    backend = ab.AscendAttnBackend(mr)
    backend._fake_pool = pool
    return backend


class FakeForwardMode:
    def is_target_verify(self): return False
    def is_draft_extend_v2(self): return False
    def is_context_parallel_extend(self): return False


def make_layer(kv_b_proj=None, h=H, hkv=HKV, d=D):
    return SimpleNamespace(
        qk_head_dim=d,
        v_head_dim=d,
        tp_q_head_num=h,
        tp_k_head_num=hkv,
        scaling=SCALE,
        layer_id=0,
        kv_b_proj=kv_b_proj,
    )


def make_fb(qlens, plens):
    return SimpleNamespace(
        extend_prefix_lens_cpu=list(plens),
        extend_seq_lens_cpu=list(qlens),
        forward_mode=FakeForwardMode(),
        num_token_non_padded_cpu=sum(qlens),
    )


class KVProj:
    """Deterministic kv_b_proj stand-in used by the prefix tests."""

    def __init__(self, out_dim, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.w = (
            torch.randn(1, out_dim, dtype=torch.bfloat16, generator=g) * 0.02
        ).to("npu")

    def __call__(self, x):
        return (x @ self.w,)


def set_prefix_state(backend, kbuf, vbuf, qlens, plens):
    backend.forward_metadata = SimpleNamespace(
        flatten_prefix_block_tables=torch.arange(
            kbuf.shape[0], dtype=torch.int64, device="npu"
        ),
        extend_seq_lens_cpu_int=list(qlens),
        prefix_lens=list(plens),
    )
    backend._fake_pool.k = kbuf
    backend._fake_pool.v = vbuf


# ---------------------------------------------------------------- correctness
def _stats(samples):
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "n": int(arr.size),
        "median_ms": float(np.median(arr)),
        "mean_ms": float(arr.mean()),
        "p95_ms": float(np.percentile(arr, 95)),
        "std_ms": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "cv_pct": float(100.0 * arr.std(ddof=1) / arr.mean()) if arr.size > 1 else 0.0,
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def _golden_fresh_rows(q, k, v, qlens, scale, global_rows):
    """FP32 causal CPU golden for selected global query rows (fresh)."""
    qf, kf, vf = q.float().cpu(), k.float().cpu(), v.float().cpu()
    outs = {}
    off = 0
    for req, length in enumerate(qlens):
        local = [r - off for r in global_rows if off <= r < off + length]
        if local:
            Q = qf[off : off + length][local]
            K = kf[off : off + length]
            V = vf[off : off + length]
            scores = torch.einsum("nhd,mhd->hnm", Q, K) * scale
            idx = torch.arange(length)
            causal = idx[None, :] > idx[local][:, None]
            scores = scores.masked_fill(causal[None], float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            outs.update(
                {
                    off + local[i]: torch.einsum("hm,mhd->hd", probs[:, i], V)
                    for i in range(len(local))
                }
            )
        off += length
    return outs


def _golden_fresh_all(q, k, v, qlens, scale):
    rows = list(range(sum(qlens)))
    got = _golden_fresh_rows(q, k, v, qlens, scale, rows)
    return torch.stack([got[r] for r in rows])


def _golden_prefix(q, k, v, qlens, plens, kbuf, vbuf, kvp, qk_nope, hkv, scale):
    ptotal = sum(plens)
    kv = (kbuf.float().cpu() @ kvp.w.float().cpu()).view(
        ptotal, hkv, qk_nope + D
    )
    k_nope, v_pre = kv.split([qk_nope, D], dim=-1)
    k_rope = vbuf.float().cpu()[:, 0, 0, :].unsqueeze(1).expand(
        ptotal, hkv, QK_ROPE
    )
    k_pre = torch.cat([k_nope, k_rope], dim=-1)
    qf, kf, vf = q.float().cpu(), k.float().cpu(), v.float().cpu()
    outs = []
    q_off = 0
    p_off = 0
    for q_len, p_len in zip(qlens, plens):
        K = torch.cat([k_pre[p_off : p_off + p_len], kf[q_off : q_off + q_len]], 0)
        V = torch.cat([v_pre[p_off : p_off + p_len], vf[q_off : q_off + q_len]], 0)
        Q = qf[q_off : q_off + q_len]
        scores = torch.einsum("nhd,mhd->hnm", Q, K) * scale
        idx = torch.arange(K.shape[0])
        key_pos = idx[None, :]
        allowed = key_pos <= (p_len + torch.arange(q_len)[:, None])
        scores = scores.masked_fill(~allowed[None], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hml,lhd->mhd", probs, V))
        q_off += q_len
        p_off += p_len
    return torch.cat(outs, 0)


def run_correctness(b0, b1, outdir):
    records = []
    checks = []

    def add(name, passed, **detail):
        checks.append(name)
        records.append({"name": name, "passed": bool(passed), **detail})
        assert passed, "%s failed: %s" % (name, detail)

    # init flag caching / gating
    add("init_flag_on", b1.fia_direct_output is True)
    add("init_flag_off", b0.fia_direct_output is False)
    os.environ["SGLANG_NPU_FIA_DIRECT_OUTPUT"] = "0"
    add("flag_cached_at_init", b1.fia_direct_output is True)
    os.environ["SGLANG_NPU_FIA_DIRECT_OUTPUT"] = "1"
    qt = torch.zeros(2, H, D, dtype=torch.bfloat16, device="npu")
    add("gate_eager_bf16", b1._can_use_fia_direct_output(qt) is True)
    b1.graph_mode = True
    add("gate_graph_falls_back", b1._can_use_fia_direct_output(qt) is False)
    b1.graph_mode = False
    b1.enable_torch_compile = True
    add("gate_compile_falls_back", b1._can_use_fia_direct_output(qt) is False)
    b1.enable_torch_compile = False
    add(
        "gate_fp32_falls_back",
        b1._can_use_fia_direct_output(
            torch.zeros(2, H, D, dtype=torch.float32, device="npu")
        )
        is False,
    )

    # fresh: minimum, ragged, zero-length, repeated
    torch.manual_seed(11)
    fresh_cases = [
        ("fresh_bs1_128", [128]),
        ("fresh_bs2_128", [128, 128]),
        ("fresh_ragged_zero_q1", [13, 1, 0, 64, 7, 33]),
    ]
    for name, qlens in fresh_cases:
        T = sum(qlens)
        q = torch.randn(T, H, D, dtype=torch.bfloat16, device="npu")
        k = torch.randn(T, HKV, D, dtype=torch.bfloat16, device="npu")
        v = torch.randn(T, HKV, D, dtype=torch.bfloat16, device="npu")
        fb = make_fb(qlens, [0] * len(qlens))
        o1 = b1.forward_extend(q, k, v, make_layer(), fb)
        o0 = b0.forward_extend(q, k, v, make_layer(), fb)
        torch.npu.synchronize()
        diff = float((o1.float() - o0.float()).abs().max())
        add(
            name,
            o1.shape == (T, H * D) and torch.equal(o1, o0),
            qlens=qlens,
            shape=list(o1.shape),
            bitwise_equal=bool(torch.equal(o1, o0)),
            max_abs_diff=diff,
        )
        if T <= 256:
            golden = _golden_fresh_all(q, k, v, qlens, SCALE)
            gdiff = float(
                (o1.float().cpu() - golden.reshape(-1, H * D)).abs().max()
            )
            add(
                name + "_fp32_golden",
                gdiff <= 0.05,
                max_abs_diff_vs_fp32_golden=gdiff,
                atol=0.05,
            )

    # repeated calls on the same backends
    for it in range(3):
        T = 256
        q = torch.randn(T, H, D, dtype=torch.bfloat16, device="npu")
        k = torch.randn(T, HKV, D, dtype=torch.bfloat16, device="npu")
        v = torch.randn(T, HKV, D, dtype=torch.bfloat16, device="npu")
        fb = make_fb([128, 128], [0, 0])
        o1 = b1.forward_extend(q, k, v, make_layer(), fb)
        o0 = b0.forward_extend(q, k, v, make_layer(), fb)
        torch.npu.synchronize()
        add("fresh_repeat_%d" % it, torch.equal(o1, o0))

    # fp16: only claim it if the same-kernel comparison is bitwise equal
    T = 24
    q16 = torch.randn(T, H, D, dtype=torch.float16, device="npu")
    k16 = torch.randn(T, HKV, D, dtype=torch.float16, device="npu")
    v16 = torch.randn(T, HKV, D, dtype=torch.float16, device="npu")
    fb16 = make_fb([16, 8], [0, 0])
    o1f = b1.forward_extend(q16, k16, v16, make_layer(), fb16)
    o0f = b0.forward_extend(q16, k16, v16, make_layer(), fb16)
    torch.npu.synchronize()
    add("fresh_fp16_bitwise", torch.equal(o1f, o0f))

    # prefix: lengths 0 / 128 / 640 with matched K/V, plus repeated calls
    qlens = [16, 32, 64]
    plens = [0, 128, 640]
    P = sum(plens)
    T = sum(qlens)
    torch.manual_seed(23)
    kbuf = torch.randn(P, 1, dtype=torch.bfloat16, device="npu")
    vbuf = torch.randn(P, 1, 1, QK_ROPE, dtype=torch.bfloat16, device="npu")
    kvp = KVProj(HKV * (QK_NOPE + D))
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device="npu")
    k = torch.randn(T, HKV, D, dtype=torch.bfloat16, device="npu")
    v = torch.randn(T, HKV, D, dtype=torch.bfloat16, device="npu")
    fb = make_fb(qlens, plens)
    for backend in (b0, b1):
        set_prefix_state(backend, kbuf, vbuf, qlens, plens)
    o1p = b1.forward_extend(q, k, v, make_layer(kvp), fb)
    o0p = b0.forward_extend(q, k, v, make_layer(kvp), fb)
    torch.npu.synchronize()
    add(
        "prefix_0_128_640",
        o1p.shape == (T, H * D) and torch.equal(o1p, o0p),
        qlens=qlens,
        plens=plens,
        shape=list(o1p.shape),
        bitwise_equal=bool(torch.equal(o1p, o0p)),
        max_abs_diff=float((o1p.float() - o0p.float()).abs().max()),
    )
    for it in range(2):
        q2 = torch.randn(T, H, D, dtype=torch.bfloat16, device="npu")
        k2 = torch.randn(T, HKV, D, dtype=torch.bfloat16, device="npu")
        v2 = torch.randn(T, HKV, D, dtype=torch.bfloat16, device="npu")
        a = b1.forward_extend(q2, k2, v2, make_layer(kvp), fb)
        b = b0.forward_extend(q2, k2, v2, make_layer(kvp), fb)
        torch.npu.synchronize()
        add("prefix_repeat_%d" % it, torch.equal(a, b))

    # prefix zero-length requests, including a request with prefix_len 0
    qlens_z, plens_z = [16, 0, 16, 32], [128, 0, 0, 640]
    Pz, Tz = sum(plens_z), sum(qlens_z)
    kbz = torch.randn(Pz, 1, dtype=torch.bfloat16, device="npu")
    vbz = torch.randn(Pz, 1, 1, QK_ROPE, dtype=torch.bfloat16, device="npu")
    qz = torch.randn(Tz, H, D, dtype=torch.bfloat16, device="npu")
    kz = torch.randn(Tz, HKV, D, dtype=torch.bfloat16, device="npu")
    vz = torch.randn(Tz, HKV, D, dtype=torch.bfloat16, device="npu")
    kvz = KVProj(HKV * (QK_NOPE + D), seed=1)
    fbz = make_fb(qlens_z, plens_z)
    for backend in (b0, b1):
        set_prefix_state(backend, kbz, vbz, qlens_z, plens_z)
    oz1 = b1.forward_extend(qz, kz, vz, make_layer(kvz), fbz)
    oz0 = b0.forward_extend(qz, kz, vz, make_layer(kvz), fbz)
    torch.npu.synchronize()
    add(
        "prefix_zero_length",
        torch.equal(oz1, oz0) and not bool(torch.isnan(oz1).any()),
        bitwise_equal=bool(torch.equal(oz1, oz0)),
    )

    # small prefix FP32 golden (mask/prefix-offset semantics)
    ql_g, pl_g = [4, 8, 4], [0, 8, 16]
    Pg, Tg = sum(pl_g), sum(ql_g)
    kbg = torch.randn(Pg, 1, dtype=torch.bfloat16, device="npu")
    vbg = torch.randn(Pg, 1, 1, QK_ROPE, dtype=torch.bfloat16, device="npu")
    qg = torch.randn(Tg, H, D, dtype=torch.bfloat16, device="npu")
    kg = torch.randn(Tg, HKV, D, dtype=torch.bfloat16, device="npu")
    vg = torch.randn(Tg, HKV, D, dtype=torch.bfloat16, device="npu")
    kvg = KVProj(HKV * (QK_NOPE + D), seed=2)
    fbg = make_fb(ql_g, pl_g)
    for backend in (b0, b1):
        set_prefix_state(backend, kbg, vbg, ql_g, pl_g)
    og = b1.forward_extend(qg, kg, vg, make_layer(kvg), fbg)
    torch.npu.synchronize()
    golden = _golden_prefix(
        qg, kg, vg, ql_g, pl_g, kbg, vbg, kvg, QK_NOPE, HKV, SCALE
    )
    gdiff = float(
        (og.float().cpu() - golden.reshape(-1, H * D)).abs().max()
    )
    add("prefix_fp32_golden", gdiff <= 0.05, max_abs_diff_vs_fp32_golden=gdiff)

    # large profile shapes: full old-vs-new outputs + sampled CPU golden rows
    large = [
        ("fresh_bs1_T3787", [3787]),
        ("fresh_bs4_T15148", [3787] * 4),
    ]
    large_summary = {}
    for name, qlens_l in large:
        T = sum(qlens_l)
        q = torch.randn(T, H, D, dtype=torch.bfloat16, device="npu")
        k = torch.randn(T, HKV, D, dtype=torch.bfloat16, device="npu")
        v = torch.randn(T, HKV, D, dtype=torch.bfloat16, device="npu")
        fbl = make_fb(qlens_l, [0] * len(qlens_l))
        o1 = b1.forward_extend(q, k, v, make_layer(), fbl)
        o0 = b0.forward_extend(q, k, v, make_layer(), fbl)
        torch.npu.synchronize()
        equal = bool(torch.equal(o1, o0))
        row_ids = [0, 1, 100, 1000, qlens_l[0] - 1, qlens_l[0], T - 1]
        golden_rows = _golden_fresh_rows(q, k, v, qlens_l, SCALE, row_ids)
        o1_cpu = o1.float().cpu()
        sampled = {
            int(r): float((o1_cpu[r] - golden_rows[r].reshape(-1)).abs().max())
            for r in golden_rows
        }
        gmax = max(sampled.values())
        add(
            name,
            equal and o1.shape == (T, H * D) and gmax <= 0.05,
            qlens=qlens_l,
            shape=list(o1.shape),
            bitwise_equal=equal,
            sampled_golden_max_abs_diff=gmax,
            sampled_rows={str(k): v for k, v in sampled.items()},
            atol=0.05,
        )
        large_summary[name] = {
            "T": T,
            "bitwise_equal": equal,
            "sampled_golden_max_abs_diff": gmax,
        }

    # non-zero output storage offset: real FIA writes the middle slice only
    os.environ["SGLANG_NPU_FIA_DIRECT_OUTPUT"] = "1"
    T, pre, post = 32, 7, 11
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device="npu")
    k = torch.randn(T, HKV, D, dtype=torch.bfloat16, device="npu")
    v = torch.randn(T, HKV, D, dtype=torch.bfloat16, device="npu")
    fb = make_fb([T], [0])
    ref = b0.forward_extend(q, k, v, make_layer(), fb)
    buf = torch.full((T + pre + post, H, D), 12345.0, dtype=torch.bfloat16, device="npu")
    out_view = buf[pre : pre + T].unsqueeze(0)
    b1._fia_bsnd_prefill_out(
        q[None],
        k[None],
        v[None].contiguous(),
        out_view,
        b1.fia_mask.unsqueeze(0),
        3,
        SCALE,
        H,
        HKV,
    )
    torch.npu.synchronize()
    sent_ok = bool((buf[:pre] == 12345).all()) and bool((buf[pre + T :] == 12345).all())
    add(
        "offset_sentinel_direct_out",
        out_view.storage_offset() == pre * H * D
        and out_view.is_contiguous()
        and sent_ok
        and torch.equal(buf[pre : pre + T].reshape(T, H * D), ref),
        storage_offset_elements=int(out_view.storage_offset()),
        sentinels_untouched=sent_ok,
        alias_data_ptr_match=out_view.data_ptr()
        == buf.data_ptr() + pre * H * D * buf.element_size(),
    )

    return records, large_summary


# --------------------------------------------------------------------- timing
def _case_inputs(qlens, plens=None, prefix=False, seed=100):
    torch.manual_seed(seed)
    T = sum(qlens)
    q = torch.randn(T, H, D, dtype=torch.bfloat16, device="npu")
    k = torch.randn(T, HKV, D, dtype=torch.bfloat16, device="npu")
    v = torch.randn(T, HKV, D, dtype=torch.bfloat16, device="npu")
    if not prefix:
        return dict(q=q, k=k, v=v, layer=make_layer(), fb=make_fb(qlens, [0] * len(qlens)))
    P = sum(plens)
    kbuf = torch.randn(P, 1, dtype=torch.bfloat16, device="npu")
    vbuf = torch.randn(P, 1, 1, QK_ROPE, dtype=torch.bfloat16, device="npu")
    kvp = KVProj(HKV * (QK_NOPE + D), seed=seed)
    return dict(
        q=q,
        k=k,
        v=v,
        layer=make_layer(kvp),
        fb=make_fb(qlens, plens),
        kbuf=kbuf,
        vbuf=vbuf,
        qlens=qlens,
        plens=plens,
    )


def _prepare(backend, case):
    if "kbuf" in case:
        set_prefix_state(
            backend, case["kbuf"], case["vbuf"], case["qlens"], case["plens"]
        )


def _call(backend, case):
    return backend.forward_extend(case["q"], case["k"], case["v"], case["layer"], case["fb"])


def measure_timing(b0, b1, outdir):
    cases = {
        "fresh_bs2_T256": _case_inputs([128, 128], seed=201),
        "fresh_bs1_T3787": _case_inputs([3787], seed=202),
        "fresh_bs4_T15148": _case_inputs([3787] * 4, seed=203),
        "prefix_128_640": _case_inputs([64, 64], [128, 640], prefix=True, seed=204),
    }
    rounds = {
        "fresh_bs2_T256": 30,
        "fresh_bs1_T3787": 16,
        "fresh_bs4_T15148": 8,
        "prefix_128_640": 20,
    }
    results = {}
    for name, case in cases.items():
        n = rounds[name]
        _prepare(b0, case)
        _prepare(b1, case)
        for _ in range(5):
            _call(b0, case)
            _call(b1, case)
        torch.npu.synchronize()
        samples = {"old": [], "new": []}
        retained = {"old": [], "new": []}
        for r in range(n):
            order = [("old", b0), ("new", b1)] if r % 2 == 0 else [("new", b1), ("old", b0)]
            for label, backend in order:
                start = torch.npu.Event(enable_timing=True)
                end = torch.npu.Event(enable_timing=True)
                start.record()
                out = _call(backend, case)
                end.record()
                end.synchronize()
                samples[label].append(float(start.elapsed_time(end)))
                retained[label].append(out)
        torch.npu.synchronize()
        stats = {k: _stats(v) for k, v in samples.items()}
        paired = [a - b for a, b in zip(samples["old"], samples["new"])]
        net = float(np.median(paired))
        # wall time: batch enqueue (CPU overhead) and per-call sync wall
        wall = {}
        for label, backend in (("old", b0), ("new", b1)):
            for _ in range(3):
                _call(backend, case)
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                _call(backend, case)
            torch.npu.synchronize()
            enqueue = (time.perf_counter() - t0) * 1e3 / 5
            per_call = []
            for _ in range(5):
                torch.npu.synchronize()
                t = time.perf_counter()
                _call(backend, case)
                torch.npu.synchronize()
                per_call.append((time.perf_counter() - t) * 1e3)
            wall[label] = {
                "enqueue_ms_per_call": enqueue,
                "sync_wall_ms_per_call": float(np.median(per_call)),
            }
        results[name] = {
            "qlens": case["fb"].extend_seq_lens_cpu,
            "prefix_lens": case["fb"].extend_prefix_lens_cpu,
            "rounds": n,
            "old": stats["old"],
            "new": stats["new"],
            "paired_net_saved_ms": {
                "median": net,
                "mean": float(np.mean(paired)),
                "p95": float(np.percentile(paired, 95)),
            },
            "net_saved_ms_per_layer": net,
            "net_saved_ms_47_layers_equivalent": net * 47,
            "speedup_pct": 100.0 * net / stats["old"]["median_ms"],
            "wall": wall,
        }
        del retained
    return results


# -------------------------------------------------------------------- profile
_COPY_KEYS = ("tensormove", "inplacecopy", "viewcopy")
_ASSEMBLY_KEYS = ("concat", "cat_", "gather", "scatter")


def _find_new_profile_dir(since):
    bases = [
        os.path.join(os.getcwd(), "export_only_prof_dir"),
        "/workspace/export_only_prof_dir",
    ]
    dirs = [
        d
        for base in bases
        for d in glob.glob(base + "/*")
        if os.path.isdir(d) and os.path.getmtime(d) >= since - 2
    ]
    if not dirs:
        raise RuntimeError("no parsed profiler output directory found")
    return max(dirs, key=os.path.getmtime)


def _parse_task_time(path):
    rows = []
    with open(path, newline="", errors="replace") as fh:
        for row in csv.DictReader(fh):
            row = {str(k).strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            try:
                rows.append(
                    {
                        "kernel": row.get("kernel_name", ""),
                        "type": row.get("kernel_type", ""),
                        "stream": row.get("stream_id", ""),
                        "task": row.get("task_id", ""),
                        "start_us": float(row.get("task_start(us)", "nan")),
                        "stop_us": float(row.get("task_stop(us)", "nan")),
                        "dur_us": float(row.get("task_time(us)", "nan")),
                    }
                )
            except (TypeError, ValueError):
                continue
    rows.sort(key=lambda r: r["start_us"])
    return rows


def _profile_arm(backend, case, iters, label, outdir):
    exp = torch_npu.profiler._ExperimentalConfig(
        l2_cache=False, op_attr=False, data_simplification=False
    )
    since = time.time()
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        experimental_config=exp,
    ) as prof:
        for _ in range(iters):
            _call(backend, case)
        torch.npu.synchronize()
    chrome = os.path.join(outdir, "profile_%s_trace.json" % label)
    prof.export_chrome_trace(chrome)
    pdir = _find_new_profile_dir(since)
    csvs = glob.glob(pdir + "/**/task_time_*.csv", recursive=True)
    if not csvs:
        raise RuntimeError("task_time csv missing under " + pdir)
    csvs.sort(key=os.path.getmtime)
    copy_path = os.path.join(outdir, "profile_%s_task_time.csv" % label)
    with open(csvs[-1], "r", errors="replace") as src, open(copy_path, "w") as dst:
        dst.write(src.read())
    rows = _parse_task_time(copy_path)
    counts = {}
    for r in rows:
        counts[r["kernel"]] = counts.get(r["kernel"], 0) + 1
    lower = {k.lower(): v for k, v in counts.items()}
    copies = {k: v for k, v in lower.items() if any(x in k for x in _COPY_KEYS)}
    assembly = {k: v for k, v in lower.items() if any(x in k for x in _ASSEMBLY_KEYS)}
    fia = {k: v for k, v in lower.items() if "fusedinferattention" in k or "fused_infer" in k}
    # Linear check: every FIA kernel is followed by a copy on the same stream.
    pairs = 0
    for i, r in enumerate(rows):
        if not (("fusedinferattention" in r["kernel"].lower()) or ("fused_infer" in r["kernel"].lower())):
            continue
        for nxt in rows[i + 1 :]:
            if nxt["stream"] == r["stream"] and nxt["start_us"] >= r["stop_us"]:
                if any(x in nxt["kernel"].lower() for x in _COPY_KEYS):
                    pairs += 1
                break
    return {
        "label": label,
        "iters": iters,
        "parsed_dir": pdir,
        "task_time_csv": copy_path,
        "device_kernel_count": len(rows),
        "kernel_counts": counts,
        "fia_kernel_counts": fia,
        "copy_kernel_counts": copies,
        "assembly_kernel_counts": assembly,
        "fia_then_copy_pairs": pairs,
        "kernels": rows,
    }


def run_profile(b0, b1, outdir):
    cases = {
        "fresh_bs1_T3787": (_case_inputs([3787], seed=301), 2),
        "fresh_bs4_T15148": (_case_inputs([3787] * 4, seed=302), 1),
    }
    extract = {}
    for name, (case, iters) in cases.items():
        _prepare(b0, case)
        _prepare(b1, case)
        base = _profile_arm(b0, case, iters, "baseline_" + name, outdir)
        cand = _profile_arm(b1, case, iters, "candidate_" + name, outdir)
        copies_base = sum(base["copy_kernel_counts"].values())
        copies_cand = sum(cand["copy_kernel_counts"].values())
        assembly_cand = sum(cand["assembly_kernel_counts"].values())
        fia_cand = sum(cand["fia_kernel_counts"].values())
        expected_fia = iters * len(case["fb"].extend_seq_lens_cpu)
        assert copies_base > 0, "baseline profile has no copy kernels"
        assert copies_cand == 0, "candidate still has copy kernels: %s" % cand["copy_kernel_counts"]
        assert assembly_cand == 0, "candidate has assembly kernels: %s" % cand["assembly_kernel_counts"]
        assert fia_cand == expected_fia, (fia_cand, expected_fia)
        extract[name] = {
            "iters": iters,
            "baseline": {k: v for k, v in base.items() if k not in ("kernels", "parsed_dir")},
            "candidate": {k: v for k, v in cand.items() if k not in ("kernels", "parsed_dir")},
            "baseline_all_kernels": base["kernels"],
            "candidate_all_kernels": cand["kernels"],
            "copy_kernels_removed": copies_base,
            "candidate_fia_count_matches_expected": True,
        }
    return extract


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir",
        default="/mnt/sfs_turbo/hxd/attention-output-copy-20261102/artifacts",
    )
    ap.add_argument("--skip-profile", action="store_true")
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    torch.npu.set_device(0)
    free, total = torch.npu.mem_get_info()
    if free < 60 * 1024**3:
        raise RuntimeError(
            "NPU 6 does not look idle: only %.1f GiB free of %.1f GiB"
            % (free / 1024**3, total / 1024**3)
        )
    meta = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "visible_devices_env": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "logical_device": torch.npu.current_device(),
        "device_name": torch_npu.npu.get_device_name(torch.npu.current_device()),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "free_gib": free / 1024**3,
        "total_gib": total / 1024**3,
        "worktree": WT,
        "flag_env": os.environ.get("SGLANG_NPU_FIA_DIRECT_OUTPUT"),
    }

    b1 = make_backend("1")
    b0 = make_backend("0")
    correctness, large_summary = run_correctness(b0, b1, args.outdir)
    result = {
        "meta": meta,
        "correctness": correctness,
        "large_summary": large_summary,
    }
    if not args.skip_timing:
        result["timing"] = measure_timing(b0, b1, args.outdir)
    if not args.skip_profile:
        result["profile"] = run_profile(b0, b1, args.outdir)
    result["all_pass"] = all(r["passed"] for r in correctness)

    out = os.path.join(args.outdir, "fia_direct_output_results.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=2)
    prof_out = os.path.join(args.outdir, "fia_direct_output_profile_extract.json")
    with open(prof_out, "w") as fh:
        json.dump(result.get("profile", {}), fh, indent=2)
    print(json.dumps({"all_pass": result["all_pass"], "results": out, "profile": prof_out}, indent=2))
    assert result["all_pass"]
    print("ALL SELECTED CHECKS PASSED")


if __name__ == "__main__":
    main()
