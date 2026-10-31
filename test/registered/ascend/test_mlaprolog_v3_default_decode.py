"""Default V3 decode uses final KV and a shared Query, with explicit rollback."""
from types import SimpleNamespace
import types

import pytest
import torch
import torch_npu

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention import mla_preprocess as prolog
from sglang.srt.hardware_backend.npu.attention import flash_mla_adapter as adapter
from sglang.srt.hardware_backend.npu import memory_pool_npu as pools


@pytest.mark.parametrize("setting,expected", [(None, True), ("0", False), ("1", True), ("false", False), ("true", True)])
def test_direct_cache_default_and_override(monkeypatch, setting, expected):
    if setting is None:
        monkeypatch.delenv("SGLANG_NPU_MLAPO_V3_DIRECT_CACHE", raising=False)
    else:
        monkeypatch.setenv("SGLANG_NPU_MLAPO_V3_DIRECT_CACHE", setting)
    prolog.is_mlaprolog_v3_direct_cache_enabled.cache_clear()
    try:
        assert envs.SGLANG_NPU_MLAPO_V3_DIRECT_CACHE.get() is expected
        assert prolog.is_mlaprolog_v3_direct_cache_enabled() is expected
    finally:
        prolog.is_mlaprolog_v3_direct_cache_enabled.cache_clear()


@pytest.mark.parametrize("unified", [False, True])
def test_default_keeps_non_flash_backend_compatible(monkeypatch, unified):
    monkeypatch.delenv("SGLANG_NPU_MLAPO_V3_DIRECT_CACHE", raising=False)
    prolog.is_mlaprolog_v3_direct_cache_enabled.cache_clear()
    tokens, page = 2, 128
    k = torch.zeros(1, page, 1, 512, dtype=torch.bfloat16)
    r = torch.zeros(1, page, 1, 64, dtype=torch.bfloat16)
    slots = torch.tensor([4, 19], dtype=torch.int64)
    landings = []
    pool = SimpleNamespace(flash_mla=unified, store_dtype=k.dtype, dtype=k.dtype,
        page_size=page, k_buffer=[k], v_buffer=[r], start_layer=0,
        set_unified_mla_kv=lambda *args: landings.append(args))
    obj = SimpleNamespace(has_preprocess_weights=True, layer_id=0, kv_lora_rank=512,
        qk_rope_head_dim=64, weight_dq=torch.empty(1), weight_uq_qr=torch.empty(1),
        weight_uk=torch.empty(1), weight_dkv_kr=torch.empty(1),
        q_a_layernorm=SimpleNamespace(weight=torch.ones(768), variance_epsilon=1e-5),
        kv_a_layernorm=SimpleNamespace(weight=torch.ones(512), variance_epsilon=1e-5),
        get_sin_cos=lambda *args: (torch.zeros(tokens, 64), torch.zeros(tokens, 64)),
        get_kv_cache_and_cache_idx=lambda _: (k, r, slots))
    obj._v3_scratch_cache = types.MethodType(prolog.NPUFusedMLAPreprocess._v3_scratch_cache, obj)
    monkeypatch.setattr(prolog, "get_token_to_kv_pool", lambda: pool)
    monkeypatch.setattr(prolog, "get_attn_backend", lambda: SimpleNamespace(use_flash_mla=False, page_size=page))
    called = []
    def legacy_kernel(**kwargs):
        assert kwargs["kv_cache"].is_contiguous() and kwargs["kr_cache"].is_contiguous()
        called.append(True)
        return (torch.zeros(tokens, 20, 512), torch.zeros(tokens, 20, 64),
                torch.empty(0), torch.zeros(tokens, 768), torch.empty(0))
    monkeypatch.setattr(torch.ops.npu, "mla_prolog_v3", legacy_kernel)
    try:
        result = prolog.NPUFusedMLAPreprocess.forward_mlaprolog_v3(obj,
            torch.arange(tokens), torch.empty(tokens, 2048), SimpleNamespace(out_cache_loc=slots))
        assert called and result[2].shape == (tokens, 20, 512)
        assert bool(landings) is unified
    finally:
        prolog.is_mlaprolog_v3_direct_cache_enabled.cache_clear()


@pytest.mark.parametrize("tokens", [1, 7, 9, 10, 11, 64])
def test_default_decode_skips_concat_scratch_and_landing(monkeypatch, tokens):
    torch.manual_seed(42)
    def rand(*shape):
        return (torch.randn(*shape) * .02).to(dtype=torch.bfloat16, device="npu:0")
    def nz(*shape):
        return torch_npu.npu_format_cast(rand(*shape), 29)
    monkeypatch.setattr(pools, "use_unified_mla_kv_layout", lambda: True)
    pool = pools.NPUMLATokenToKVPool(size=2048, page_size=128, dtype=torch.bfloat16,
        kv_lora_rank=512, qk_rope_head_dim=64, layer_num=1,
        device="npu:0", enable_memory_saver=False)
    pool.kv_buffer.fill_(123)
    slots = torch.arange(tokens, device="npu:0", dtype=torch.int64) * 17 + 127
    x, sin, cos = rand(tokens, 2048), rand(tokens, 64), rand(tokens, 64)
    obj = SimpleNamespace(has_preprocess_weights=True, layer_id=0, kv_lora_rank=512,
        qk_rope_head_dim=64, weight_dq=nz(2048, 768), weight_uq_qr=nz(768, 5120),
        weight_uk=rand(20, 192, 512), weight_dkv_kr=nz(2048, 576),
        q_a_layernorm=SimpleNamespace(weight=rand(768) + 1, variance_epsilon=1e-5),
        kv_a_layernorm=SimpleNamespace(weight=rand(512) + 1, variance_epsilon=1e-5),
        get_sin_cos=lambda *args: (cos, sin))
    obj._v3_scratch_cache = types.MethodType(prolog.NPUFusedMLAPreprocess._v3_scratch_cache, obj)
    monkeypatch.setattr(prolog, "get_token_to_kv_pool", lambda: pool)
    monkeypatch.setattr(prolog, "get_attn_backend", lambda: SimpleNamespace(use_flash_mla=True))
    batch, positions = SimpleNamespace(out_cache_loc=slots), torch.arange(tokens, device="npu:0")
    run = lambda: prolog.NPUFusedMLAPreprocess.forward_mlaprolog_v3(obj, positions, x, batch)
    monkeypatch.setenv("SGLANG_NPU_MLAPO_V3_DIRECT_CACHE", "0")
    monkeypatch.setenv("SGLANG_NPU_MLAPO_V3_PACKED_QUERY", "0")
    prolog.is_mlaprolog_v3_direct_cache_enabled.cache_clear()
    expected = run()
    expected_cache = pool.kv_buffer.clone()
    expected_query = torch.cat([expected[2], expected[0]], -1)
    descriptor = pool.get_contiguous_buf_infos()
    pool.kv_buffer.fill_(123)
    monkeypatch.delenv("SGLANG_NPU_MLAPO_V3_DIRECT_CACHE")
    monkeypatch.delenv("SGLANG_NPU_MLAPO_V3_PACKED_QUERY")
    prolog.is_mlaprolog_v3_direct_cache_enabled.cache_clear()
    def forbidden(*args, **kwargs):
        raise AssertionError("default decode must skip concat, scratch and landing")
    try:
        with monkeypatch.context() as hot_path:
            hot_path.setattr(obj, "_v3_scratch_cache", forbidden)
            hot_path.setattr(pool, "set_unified_mla_kv", forbidden)
            hot_path.setattr(torch, "cat", forbidden)
            actual = run()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                captured = run()
            pool.kv_buffer.fill_(123)
            graph.replay()
            torch.npu.synchronize()
            full = adapter.combine_query(captured[2], captured[0])
            assert full.is_contiguous() and full.data_ptr() == captured[2].data_ptr()
            assert torch.equal(full, expected_query)
            assert torch.equal(pool.kv_buffer, expected_cache)
            assert all(torch.equal(actual[i], expected[i]) for i in [0, 2, 4])
            assert all(torch.equal(captured[i], expected[i]) for i in [0, 2, 4])
        assert pool.get_contiguous_buf_infos() == descriptor
    finally:
        prolog.is_mlaprolog_v3_direct_cache_enabled.cache_clear()
