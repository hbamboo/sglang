"""Regression contracts for SGLANG_NPU_FIA_DIRECT_OUTPUT (CPU only).

These tests exercise the real AscendAttnBackend.forward_extend routing with
CPU tensors and fake FIA operators.  They pin:
  * the init-cached flag / eager-only gate,
  * the ``.out`` helper call shape, offset preservation and kwargs,
  * fresh + prefix loop routing (default op vs direct out),
  * zero-length request skipping and q_len==1 sparse_mode selection.

Run from the repository root with the worktree's python/ on PYTHONPATH.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.hardware_backend.npu.attention import ascend_backend as ab


def _gate(**kw):
    vals = dict(fia_direct_output=True, graph_mode=False, enable_torch_compile=False)
    vals.update(kw)
    return SimpleNamespace(**vals)


def _tensor(dtype=torch.bfloat16, device_type="npu", shape=(2, 2, 8)):
    return SimpleNamespace(dtype=dtype, device=SimpleNamespace(type=device_type), shape=shape)


def test_gate_flag_cached_and_eager_only():
    gate = ab.AscendAttnBackend._can_use_fia_direct_output
    q = _tensor()
    assert gate(_gate(), q)
    assert not gate(_gate(fia_direct_output=False), q)
    assert not gate(_gate(graph_mode=True), q)
    assert not gate(_gate(enable_torch_compile=True), q)
    assert not gate(_gate(), _tensor(device_type="cpu"))
    assert not gate(_gate(), _tensor(dtype=torch.float32))
    assert gate(_gate(), _tensor(dtype=torch.float16))


def test_gate_runtime_compile_and_capture_fall_back(monkeypatch):
    gate = ab.AscendAttnBackend._can_use_fia_direct_output
    q = _tensor()
    monkeypatch.setattr(ab.torch.compiler, "is_compiling", lambda: True)
    assert not gate(_gate(), q)
    monkeypatch.undo()

    monkeypatch.setattr(
        ab.torch.npu, "is_current_stream_capturing", lambda: True
    )
    assert not gate(_gate(), q)


def test_helper_calls_out_with_offset_and_expected_kwargs(monkeypatch):
    captured = {}

    def fake_out(*args, **kwargs):
        out = kwargs["out"][0]
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["out_ptr"] = out.data_ptr()
        out.fill_(3.0)
        return out, kwargs["out"][1]

    monkeypatch.setattr(
        ab.torch_npu,
        "npu_fused_infer_attention_score",
        SimpleNamespace(out=fake_out),
        raising=False,
    )

    H, D, qlen = 2, 8, 4
    query = torch.zeros(1, qlen, H, D, dtype=torch.bfloat16)
    key = torch.zeros(1, qlen, 1, D, dtype=torch.bfloat16)
    value = torch.zeros(1, qlen, 1, D, dtype=torch.bfloat16)
    buf = torch.zeros(7, H, D, dtype=torch.bfloat16)
    out_view = buf[2 : 2 + qlen].unsqueeze(0)
    assert out_view.storage_offset() == 2 * H * D
    assert out_view.is_contiguous()
    mask = torch.zeros(qlen, qlen, dtype=torch.bool)

    ab.AscendAttnBackend._fia_bsnd_prefill_out(
        query, key, value, out_view, mask, 3, 0.5, H, 1
    )

    kw = captured["kwargs"]
    assert kw["num_heads"] == H and kw["num_key_value_heads"] == 1
    assert kw["input_layout"] == "BSND"
    assert kw["sparse_mode"] == 3
    assert kw["scale"] == 0.5
    assert kw["next_tokens"] == 0
    assert "workspace" not in kw
    assert kw["out"][0] is out_view
    assert captured["out_ptr"] == out_view.data_ptr()
    lse = kw["out"][1]
    assert lse.shape == (1,) and lse.dtype == torch.bfloat16
    assert torch.equal(buf[2 : 2 + qlen], torch.full((qlen, H, D), 3.0))


class _FM:
    def is_target_verify(self): return False
    def is_draft_extend_v2(self): return False
    def is_context_parallel_extend(self): return False


class _Pool:
    prefill_expanded_geometry = None
    req_to_token = object()

    def __init__(self):
        self.k = None
        self.v = None

    def get_key_buffer(self, _layer_id):
        return self.k

    def get_value_buffer(self, _layer_id):
        return self.v


def _backend(direct):
    b = SimpleNamespace(
        fia_direct_output=direct,
        graph_mode=False,
        enable_torch_compile=False,
        use_mla=True,
        is_dllm_model=False,
        token_to_kv_pool=_Pool(),
        qk_nope_head_dim=6,
        fia_mask=torch.triu(torch.ones(64, 64, dtype=torch.bool), diagonal=1),
    )
    b._can_use_fia_direct_output = lambda q: direct
    b._fia_bsnd_prefill_out = ab.AscendAttnBackend._fia_bsnd_prefill_out
    return b


def _layer(h=2, hkv=1, d=8, kv_b_proj=None):
    return SimpleNamespace(
        qk_head_dim=d,
        v_head_dim=d,
        tp_q_head_num=h,
        tp_k_head_num=hkv,
        scaling=0.5,
        layer_id=0,
        kv_b_proj=kv_b_proj,
    )


def _fb(qlens, plens):
    return SimpleNamespace(
        extend_prefix_lens_cpu=list(plens),
        extend_seq_lens_cpu=list(qlens),
        forward_mode=_FM(),
        num_token_non_padded_cpu=sum(qlens),
    )


@pytest.fixture
def fake_ops(monkeypatch):
    calls = {"default": 0, "out": 0, "sparse": []}

    def default_op(*args, **kwargs):
        calls["default"] += 1
        calls["sparse"].append(kwargs.get("sparse_mode"))
        return args[0] * 2, torch.empty(0)

    def out_op(*args, **kwargs):
        calls["out"] += 1
        calls["sparse"].append(kwargs.get("sparse_mode"))
        kwargs["out"][0].fill_(7.0)
        return kwargs["out"][0], kwargs["out"][1]

    monkeypatch.setattr(
        ab, "is_mla_preprocess_for_batch", lambda _fb: False, raising=False
    )
    monkeypatch.setattr(
        ab.torch, "ops", SimpleNamespace(npu=SimpleNamespace(
            npu_fused_infer_attention_score=default_op)), raising=False
    )
    monkeypatch.setattr(
        ab.torch_npu,
        "npu_fused_infer_attention_score",
        SimpleNamespace(out=out_op),
        raising=False,
    )
    return calls


def test_fresh_loop_direct_and_fallback(fake_ops):
    torch.manual_seed(0)
    T, H, D = 8, 2, 8
    q = torch.randn(T, H, D, dtype=torch.bfloat16)
    k = torch.randn(T, 1, D, dtype=torch.bfloat16)
    v = torch.randn(T, 1, D, dtype=torch.bfloat16)
    qlens = [4, 0, 4]

    direct = ab.AscendAttnBackend.forward_extend(
        _backend(True), q, k, v, _layer(), _fb(qlens, [0, 0, 0])
    )
    assert fake_ops["out"] == 2 and fake_ops["default"] == 0
    assert direct.shape == (T, H * D)
    assert torch.equal(direct, torch.full((T, H * D), 7.0))
    assert fake_ops["sparse"] == [3, 3]

    fake_ops["sparse"].clear()
    fallback = ab.AscendAttnBackend.forward_extend(
        _backend(False), q, k, v, _layer(), _fb(qlens, [0, 0, 0])
    )
    assert fake_ops["default"] == 2 and fake_ops["out"] == 2
    expected = torch.cat([q[:4] * 2, q[4:8] * 2], dim=0).reshape(T, H * D)
    assert torch.equal(fallback, expected)

    # q_len == 1 uses sparse_mode 0 on the original route and the same on .out.
    q1 = torch.randn(1, H, D, dtype=torch.bfloat16)
    k1 = torch.randn(1, 1, D, dtype=torch.bfloat16)
    v1 = torch.randn(1, 1, D, dtype=torch.bfloat16)
    fake_ops["sparse"].clear()
    ab.AscendAttnBackend.forward_extend(
        _backend(True), q1, k1, v1, _layer(), _fb([1], [0])
    )
    assert fake_ops["sparse"] == [0]


class _KVProj:
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def __call__(self, x):
        p = x.shape[0]
        return (torch.ones(p, self.out_dim, dtype=torch.bfloat16),)


def _prefix_state(backend, qlens, plens, hkv=1):
    p = sum(plens)
    backend.forward_metadata = SimpleNamespace(
        flatten_prefix_block_tables=torch.arange(p),
        extend_seq_lens_cpu_int=list(qlens),
        prefix_lens=list(plens),
    )
    backend.token_to_kv_pool.k = torch.zeros(p, 1, dtype=torch.bfloat16)
    backend.token_to_kv_pool.v = torch.zeros(p, 1, 1, 2, dtype=torch.bfloat16)


def test_prefix_loop_direct_and_fallback(fake_ops):
    torch.manual_seed(0)
    H, HKV, D = 2, 1, 8
    qlens, plens = [1, 2], [2, 2]
    T = sum(qlens)
    q = torch.randn(T, H, D, dtype=torch.bfloat16)
    k = torch.randn(T, HKV, D, dtype=torch.bfloat16)
    v = torch.randn(T, HKV, D, dtype=torch.bfloat16)
    kvp = _KVProj(HKV * (6 + D))

    b1 = _backend(True)
    _prefix_state(b1, qlens, plens)
    direct = ab.AscendAttnBackend.forward_extend(
        b1, q, k, v, _layer(kv_b_proj=kvp), _fb(qlens, plens)
    )
    assert fake_ops["out"] == 2 and fake_ops["default"] == 0
    assert direct.shape == (T, H * D)
    assert torch.equal(direct, torch.full((T, H * D), 7.0))

    b0 = _backend(False)
    _prefix_state(b0, qlens, plens)
    fallback = ab.AscendAttnBackend.forward_extend(
        b0, q, k, v, _layer(kv_b_proj=kvp), _fb(qlens, plens)
    )
    assert fake_ops["default"] == 2
    assert torch.equal(fallback, torch.cat([q[:1] * 2, q[1:3] * 2], 0).reshape(T, H * D))

    # Zero-length request in a prefix batch: skipped, cursors still advance.
    qlens_z, plens_z = [1, 0, 2], [2, 0, 2]
    Tz = sum(qlens_z)
    qz = torch.randn(Tz, H, D, dtype=torch.bfloat16)
    kz = torch.randn(Tz, HKV, D, dtype=torch.bfloat16)
    vz = torch.randn(Tz, HKV, D, dtype=torch.bfloat16)
    bz = _backend(True)
    _prefix_state(bz, qlens_z, plens_z)
    fake_ops["out"] = 0
    ab.AscendAttnBackend.forward_extend(
        bz, qz, kz, vz, _layer(kv_b_proj=kvp), _fb(qlens_z, plens_z)
    )
    assert fake_ops["out"] == 2
