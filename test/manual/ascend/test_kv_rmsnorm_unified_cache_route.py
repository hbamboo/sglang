"""Minimal routing tests for the unified-576 latent-cache framework path.

Physical NPU 7 only. No model service is started. The tests:

  * select the new route through the real ``forward_mha_prepare_npu`` hunk with
    stub model/pool objects and the real unified layer view, verify the new op
    is invoked once, outputs are returned and the cache is updated;
  * prove the fallback calls the original operator for: env off, PA_NZ,
    unsupported geometry/dtype, non-unified pool, torch.compile active;
  * compare the helper's cache/output result bitwise with the installed old
    operator on a real 512/64 split of the same unified pool.
"""

import os
import sys
import types

import pytest
import torch
import torch_npu

ROOT = os.environ.get("KVRMSNORM_REPO_ROOT", "/mnt/sfs_turbo/hxd/kv-rmsnorm-rope-migration-20261102")
LIB = os.environ.get(
    "KV_RMSNORM_UNIFIED_LIB",
    ROOT + "/kernel/python/sgl_kernel_npu/sgl_kernel_npu/lib/libsgl_kernel_npu_kv_rmsnorm_unified.so",
)
torch.ops.load_library(LIB)
torch.npu.set_device(0)

from sglang.srt.hardware_backend.npu.modules import deepseek_v2_attention_mla_npu as site  # noqa: E402
from sglang.srt.hardware_backend.npu.modules import kv_rmsnorm_unified_cache_npu as helper  # noqa: E402


class StubPool:
    def __init__(self, cache, unified=True, start_layer=0):
        self.kv_buffer = [cache]
        self.unified_kv = unified
        self.start_layer = start_layer

    def get_kv_buffer(self, layer_id):
        layer = self.kv_buffer[layer_id - self.start_layer]
        return layer[..., :512], layer[..., 512:]


class StubNorm:
    def __init__(self, weight, eps=1e-5):
        self.weight = weight
        self.variance_epsilon = eps


def make_state(T=17, pages=2, dtype=torch.bfloat16, strided=True, seed=0):
    g = torch.Generator().manual_seed(seed + 5)
    if strided:
        qkv = torch.randn(T, 1344, generator=g).to(dtype).to("npu")
        latent2d = qkv[:, 768:]
    else:
        latent2d = torch.randn(T, 576, generator=g).to(dtype).to("npu")
    latent = latent2d.unsqueeze(1)  # [T,1,576] as in forward_mha_prepare_npu
    gamma = (torch.randn(512, generator=g) * 0.5 + 1.0).to(dtype).to("npu")
    cos = torch.randn(T, 1, 1, 64, generator=g).to(dtype).to("npu")
    sin = torch.randn(T, 1, 1, 64, generator=g).to(dtype).to("npu")
    idx = torch.randperm(pages * 128, generator=g)[:T].to(torch.int64).to("npu")
    if T >= 4:
        idx[1] = -1
    cache = torch.full((pages, 128, 1, 576), -7.0, dtype=dtype, device="npu")
    return latent, gamma, cos, sin, idx, cache


def old_op(latent, gamma, cos, sin, idx, cache):
    T = latent.shape[0]
    r = torch_npu.npu_kv_rmsnorm_rope_cache(
        latent.view(T, 1, 1, 576), gamma, cos, sin, idx,
        cache[..., 512:], cache[..., :512],
        k_rope_scale=None, c_kv_scale=None, k_rope_offset=None, c_kv_offset=None,
        epsilon=1e-5, cache_mode="PA_BNSD", is_output_kv=True,
    )
    return r[2], r[3]


def make_model(dtype=torch.bfloat16, T=17, kv_lora=512, rope=64):
    H, nope, vdim = 2, 128, 128
    weight = (torch.randn(kv_lora, device="npu") * 0.5 + 1.0).to(dtype)
    m = types.SimpleNamespace(
        layer_id=0,
        q_lora_rank=None,
        use_deepseek_yarn_rope=True,
        use_dsa=False,
        kv_lora_rank=kv_lora,
        qk_rope_head_dim=rope,
        qk_nope_head_dim=nope,
        qk_head_dim=nope + rope,
        v_head_dim=vdim,
        num_local_heads=H,
        kv_a_layernorm=StubNorm(weight),
        q_proj=lambda h: (torch.randn(T, H, nope + rope, dtype=dtype, device="npu"), None),
        kv_a_proj_with_mqa=lambda h: (torch.randn(T, kv_lora + rope, dtype=dtype, device="npu"), None),
        kv_b_proj=lambda x: (torch.randn(T, H, nope + vdim, dtype=dtype, device="npu"), None),
        _concat_and_cast_mha_k=lambda kn, kp, fb: torch.empty(T, H, nope + rope, dtype=dtype, device="npu"),
    )
    return m


def run_site(monkeypatch, env_on, fia_nz=False, model=None, pool=None, dtype=torch.bfloat16):
    T = 17
    latent, gamma, cos, sin, idx, cache = make_state(T=T, dtype=dtype)
    model = model or make_model(dtype=dtype, T=T)
    model.rotary_emb = types.SimpleNamespace(
        get_cos_sin_cache=lambda pos, dt, offsets=None: (cos, sin)
    )
    pool = pool or StubPool(cache)
    calls = {"new": 0, "old": 0}
    real_invoke = helper._invoke_new_op

    def invoke(*a, **kw):
        calls["new"] += 1
        return real_invoke(*a, **kw)

    def old_stub(*a, **kw):
        calls["old"] += 1
        tt = latent.shape[0]
        return (None, None,
                torch.zeros(tt, 1, 1, 64, dtype=dtype, device="npu"),
                torch.zeros(tt, 1, 1, 512, dtype=dtype, device="npu"))

    monkeypatch.setenv("SGLANG_NPU_KV_RMSNORM_UNIFIED_CACHE", "1" if env_on else "0")
    helper.clear_gate_cache()
    monkeypatch.setattr(helper, "_invoke_new_op", invoke)
    monkeypatch.setattr(site, "get_token_to_kv_pool", lambda: pool)
    monkeypatch.setattr(site, "is_fia_nz", lambda: fia_nz)
    monkeypatch.setattr(site, "can_fuse_expanded_kv", lambda *a, **kw: False)
    monkeypatch.setattr(torch_npu, "npu_interleave_rope", lambda x, c, s: x)
    monkeypatch.setattr(torch_npu, "npu_kv_rmsnorm_rope_cache", old_stub)
    fb = types.SimpleNamespace(out_cache_loc=idx)
    hidden = torch.randn(T, 64, dtype=dtype, device="npu")
    pos = torch.arange(T, dtype=torch.int64, device="npu")
    q, k, v, out_fb = site.forward_mha_prepare_npu(model, pos, hidden, fb, None, None)
    torch.npu.synchronize()
    return dict(calls=calls, q=q, k=k, v=v, fb=out_fb, cache=cache, latent=latent, idx=idx)


def test_real_forward_site_selects_new_route(monkeypatch):
    r = run_site(monkeypatch, env_on=True)
    assert r["calls"] == {"new": 1, "old": 0}
    assert r["k"].shape == (17, 2, 192)
    assert r["v"].shape == (17, 2, 128)
    flat = r["cache"].view(-1, 576)
    touched = set(int(x) for x in r["idx"].cpu().tolist() if int(x) >= 0)
    for slot in touched:
        assert not torch.equal(flat[slot], torch.full_like(flat[slot], -7.0))
    print("[route] new op selected once, cache slots updated")


@pytest.mark.parametrize(
    "case",
    ["env_off", "pa_nz", "non_unified_pool", "wrong_geometry", "fp32_dtype", "compiling"],
)
def test_real_forward_site_falls_back(monkeypatch, case):
    kwargs = dict(env_on=True)
    if case == "env_off":
        kwargs["env_on"] = False
    elif case == "pa_nz":
        kwargs["fia_nz"] = True
    elif case == "non_unified_pool":
        T = 17
        dummy = torch.empty(2, 128, 1, 576, dtype=torch.bfloat16, device="npu")
        kwargs["pool"] = StubPool(dummy, unified=False)
    elif case == "wrong_geometry":
        kwargs["model"] = make_model(T=17, kv_lora=256, rope=32)
    elif case == "fp32_dtype":
        T = 17
        kwargs["dtype"] = torch.float32
        m = make_model(dtype=torch.float32, T=T)
        kwargs["model"] = m
    elif case == "compiling":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    r = run_site(monkeypatch, **kwargs)
    assert r["calls"] == {"new": 0, "old": 1}, (case, r["calls"])
    print(f"[fallback] {case}: original operator used, new route not invoked")


def test_helper_matches_old_op_bitwise(monkeypatch):
    T, pages = 129, 9
    latent, gamma, cos, sin, idx, cache = make_state(T=T, pages=pages, strided=True, seed=3)
    ref = cache.clone()
    calls = {"new": 0}
    real = helper._invoke_new_op

    def invoke(*a, **kw):
        calls["new"] += 1
        return real(*a, **kw)

    monkeypatch.setenv("SGLANG_NPU_KV_RMSNORM_UNIFIED_CACHE", "1")
    helper.clear_gate_cache()
    monkeypatch.setattr(helper, "_invoke_new_op", invoke)
    m = make_model(T=T)
    m.kv_lora_rank, m.qk_rope_head_dim = 512, 64
    m.kv_a_layernorm.weight = gamma
    pool = StubPool(cache)
    out = helper.maybe_unified_kv_rmsnorm_rope_cache(m, latent, cos, sin, idx, pool, False)
    assert out is not None and calls["new"] == 1
    ko, vo = out
    ko_old, vo_old = old_op(latent, gamma, cos, sin, idx, ref)
    torch.npu.synchronize()
    assert torch.equal(ko.view(torch.uint8), ko_old.view(torch.uint8))
    assert torch.equal(vo.view(torch.uint8), vo_old.view(torch.uint8))
    assert torch.equal(cache.view(torch.uint8), ref.view(torch.uint8))
    print("[helper] strided 1344 latent view, outputs and unified cache bitwise vs old op")


def test_helper_rejects_non_packed_layer_view(monkeypatch):
    T = 9
    latent, gamma, cos, sin, idx, _ = make_state(T=T, pages=2)
    big = torch.empty(8, 128, 1, 576, dtype=torch.bfloat16, device="npu")
    gap = big.as_strided((2, 128, 1, 576), (128 * 576 * 2, 576, 576, 1), 0)
    monkeypatch.setenv("SGLANG_NPU_KV_RMSNORM_UNIFIED_CACHE", "1")
    helper.clear_gate_cache()
    m = make_model(T=T)
    assert helper.unified_layer_cache_view(StubPool(gap), 0) is None
    assert helper.maybe_unified_kv_rmsnorm_rope_cache(m, latent, cos, sin, idx, StubPool(gap), False) is None
    print("[fallback] page-gap layer view rejected")


def test_helper_rejects_npu_capture(monkeypatch):
    T = 9
    latent, gamma, cos, sin, idx, cache = make_state(T=T, pages=2)
    monkeypatch.setenv("SGLANG_NPU_KV_RMSNORM_UNIFIED_CACHE", "1")
    helper.clear_gate_cache()
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: True)
    m = make_model(T=T)
    assert helper.maybe_unified_kv_rmsnorm_rope_cache(m, latent, cos, sin, idx, StubPool(cache), False) is None
    print("[fallback] NPU graph capture excluded")
