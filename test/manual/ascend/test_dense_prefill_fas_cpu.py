"""CPU-only contract test for the opt-in dense (non-expanded-KV) prefill FAS.

The real module and the real integration functions are exercised with a fake
public ``npu_fusion_attention``; no torch-NPU import, model or device setup.

Covered:
  * cold multi-query ragged batch: one whole-batch public FAS call, exact input
    tensor identity/data_ptr, cumulative q/kv lengths, scale/mask/sparse-mode
    kwargs, output shape and a zero-copy view, per-step plan reuse;
  * merged prefix/current context: ragged prefixes, q_len=1 and non-128-multiple
    queries, one call with cumulative prefix+new kv lengths;
  * fallbacks: flag off, zero query, padded query, non-contiguous/dtype/
    geometry/spec/capture/expanded-KV/PA_NZ/scatter/ordinary-prefix miss,
    inconsistent merged offsets;
  * the real core dispatcher still falls back to RadixAttention and applies
    gate + o_proj; the real merged consumer still runs its per-request FIA loop
    when dense FAS declines, and returns the FAS view when it accepts.
"""

import ast
import sys
import textwrap
from typing import Optional
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python"))

import torch  # noqa: E402

from sglang.srt.hardware_backend.npu.modules import (  # noqa: E402
    dense_prefill_fas_npu as fas,
)
from sglang.srt.hardware_backend.npu.modules.prefix_kv_b_proj_npu import (  # noqa: E402
    iter_fia_request_slices,
)


def make_layer(**overrides):
    layer = SimpleNamespace(
        tp_q_head_num=20,
        tp_k_head_num=20,
        tp_v_head_num=20,
        qk_head_dim=256,
        v_head_dim=256,
        is_cross_attention=False,
        sliding_window_size=-1,
        scaling=0.0625,
        layer_id=3,
    )
    for key, value in overrides.items():
        setattr(layer, key, value)
    return layer


class Pool:
    unified_kv = True
    use_dsa = False
    prefill_expanded_geometry = None


def make_backend(graph_mode=False):
    return SimpleNamespace(
        token_to_kv_pool=Pool(),
        fia_mask=torch.zeros((2048, 2048), dtype=torch.int8),
        forward_metadata=SimpleNamespace(),
        graph_mode=graph_mode,
        attn_cp_size=1,
    )


def make_batch(prefix, seq, mode=None, spec=None, cp=None):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return SimpleNamespace(
        forward_mode=mode or ForwardMode.EXTEND,
        extend_prefix_lens_cpu=list(prefix),
        extend_seq_lens_cpu=list(seq),
        spec_info=spec,
        attn_cp_metadata=cp,
        num_token_non_padded_cpu=sum(seq),
    )


class Patches:
    def __init__(self, **flags):
        self.flags = {
            "enabled": True,
            "graph": False,
            "backend": True,
            "nz": False,
            "scatter": False,
            "parallel": True,
            "npu": True,
        }
        self.flags.update(flags)
        self.saved = []

    def __enter__(self):
        f = self.flags
        for name, value in (
            ("_fas_enabled", lambda: f["enabled"]),
            ("_graph_or_compile_active", lambda backend: f["graph"]),
            ("_backend_is_supported", lambda backend: f["backend"]),
            ("_pa_nz_active", lambda: f["nz"]),
            ("_scatter_active", lambda: f["scatter"]),
            ("_parallel_tp1", lambda: f["parallel"]),
            ("_npu_tensor_ok", lambda tensor: f["npu"]),
        ):
            self.saved.append((name, getattr(fas, name)))
            setattr(fas, name, value)
        return self

    def __exit__(self, *exc):
        for name, value in reversed(self.saved):
            setattr(fas, name, value)
        return False


def make_fake_fas(contiguous=True):
    calls = []

    def fake(q, k, v, **kwargs):
        rows = kwargs["actual_seq_qlen"][-1]
        out = torch.zeros((rows, 20, 256), dtype=q.dtype)
        calls.append(
            SimpleNamespace(q=q, k=k, v=v, kwargs=kwargs, out=out)
        )
        return (out,) + tuple(torch.empty(0) for _ in range(6))

    return fake, calls


def run_cold(prefix, seq, q, k, v, layer, backend, batch, flags=None):
    patches = Patches(**(flags or {}))
    with patches:
        fake, calls = make_fake_fas()
        saved = fas._invoke_public_fas
        fas._invoke_public_fas = fake
        try:
            out = fas.maybe_forward_dense_prefill_fas(
                backend, q, k, v, layer, batch
            )
        finally:
            fas._invoke_public_fas = saved
    return out, calls


def test_cold_multiquery_one_call_and_identity():
    prefix, seq = [0, 0, 0], [3, 5, 2]
    q = torch.zeros((10, 20, 256), dtype=torch.bfloat16)
    k = torch.zeros((10, 20, 256), dtype=torch.bfloat16)
    v = torch.zeros((10, 20, 256), dtype=torch.bfloat16)
    backend, layer = make_backend(), make_layer()
    batch = make_batch(prefix, seq)
    out, calls = run_cold(prefix, seq, q, k, v, layer, backend, batch)
    assert out is not None
    assert len(calls) == 1
    call = calls[0]
    assert call.q is q and call.k is k and call.v is v
    assert call.q.data_ptr() == q.data_ptr()
    assert call.k.data_ptr() == k.data_ptr()
    assert call.v.data_ptr() == v.data_ptr()
    kw = call.kwargs
    assert kw["head_num"] == 20 and kw["input_layout"] == "TND"
    assert kw["atten_mask"] is backend.fia_mask
    assert kw["scale"] == 0.0625 and kw["keep_prob"] == 1.0
    assert kw["actual_seq_qlen"] == (3, 8, 10)
    assert kw["actual_seq_kvlen"] == (3, 8, 10)
    assert kw["sparse_mode"] == 3
    assert kw["pre_tockens"] == 2147483647 and kw["next_tockens"] == 0
    assert tuple(out.shape) == (10, 20 * 256)
    assert out.data_ptr() == call.out.data_ptr(), "must be a view, not a copy"
    assert fas.dense_fas_call_count() == 1
    plan = backend.forward_metadata.dense_fas_plan
    assert plan is not None and plan.total_q == 10
    # Second layer of the same step reuses the cached host plan, still one call
    # per layer (the native op is per layer).
    out2, calls2 = run_cold(prefix, seq, q, k, v, layer, backend, batch)
    assert out2 is not None and len(calls2) == 1
    assert backend.forward_metadata.dense_fas_plan is plan
    print("PASS cold: one whole-batch call, identities, lengths, view, plan reuse")


def test_merged_ragged_prefix_and_cumulative_kv():
    prefix, seq = (64, 0, 128), (1, 4, 7)
    total = sum(prefix) + sum(seq)
    q = torch.zeros((12, 20, 256), dtype=torch.bfloat16)
    k = torch.zeros((total, 20, 256), dtype=torch.bfloat16)
    v = torch.zeros((total, 20, 256), dtype=torch.bfloat16)
    kv_offsets, q_offsets, kv, qq = [], [], 0, 0
    for p, length in zip(prefix, seq):
        kv_offsets.append(kv)
        q_offsets.append(qq)
        kv += p + length
        qq += length
    context = SimpleNamespace(
        prefix_lens=prefix,
        seq_lens=seq,
        q_offsets=tuple(q_offsets),
        kv_offsets=tuple(kv_offsets),
        total_rows=total,
        k_full=k,
        v_full=v,
        num_heads=20,
        qk_head_dim=256,
        v_head_dim=256,
        layer_id=3,
    )
    backend, layer = make_backend(), make_layer()
    batch = make_batch(prefix, seq)
    patches = Patches()
    with patches:
        fake, calls = make_fake_fas()
        saved = fas._invoke_public_fas
        fas._invoke_public_fas = fake
        try:
            out = fas.maybe_forward_merged_dense_prefill_fas(
                backend, q, context, layer, batch
            )
        finally:
            fas._invoke_public_fas = saved
    assert out is not None and len(calls) == 1
    call = calls[0]
    assert call.q is q and call.k is k and call.v is v
    assert call.kwargs["actual_seq_qlen"] == (1, 5, 12)
    assert call.kwargs["actual_seq_kvlen"] == (65, 69, 204)
    assert call.kwargs["sparse_mode"] == 3
    assert tuple(out.shape) == (12, 20 * 256)
    assert out.data_ptr() == call.out.data_ptr()
    print("PASS merged: ragged prefix/q_len1/non-128 + one call, exact cum lengths")


def test_merged_early_return_and_fia_fallback_in_real_consumer():
    src = (REPO / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py").read_text()
    tree = ast.parse(src)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "forward_merged_prefix_kv"
    )
    body = textwrap.dedent(ast.get_source_segment(src, method))

    prefix, seq = (2, 1), (1, 3)
    total = 7
    context = SimpleNamespace(
        prefix_lens=prefix,
        seq_lens=seq,
        q_offsets=(0, 1),
        kv_offsets=(0, 3),
        total_rows=total,
        k_full=torch.zeros((total, 20, 256), dtype=torch.bfloat16),
        v_full=torch.zeros((total, 20, 256), dtype=torch.bfloat16),
        num_heads=20,
        qk_head_dim=256,
        v_head_dim=256,
        layer_id=3,
    )
    layer = make_layer()
    q = torch.zeros((4, 20, 256), dtype=torch.bfloat16)
    fia_calls = []

    class Self:
        fia_mask = torch.zeros((2048, 2048), dtype=torch.int8)

        @staticmethod
        def _can_use_fia_direct_output(query):
            return True

        @staticmethod
        def _fia_bsnd_prefill_out(query, key, value, out, *args):
            fia_calls.append((query.shape, key.shape, value.shape, out.shape))

    def build(fas_result):
        ns = {
            "torch": torch,
            "iter_fia_request_slices": iter_fia_request_slices,
            "ForwardBatch": object,
            "MergedPrefixKVProjection": object,
            "maybe_forward_merged_dense_prefill_fas": lambda *a, **k: fas_result,
        }
        exec(compile(body, "<merged-consumer>", "exec"), ns)
        return ns["forward_merged_prefix_kv"]

    sentinel = torch.full((4, 20 * 256), 7.0, dtype=torch.bfloat16)
    out = build(sentinel)(Self(), q, context, layer, object())
    assert out is sentinel and fia_calls == [], "FAS result must return directly"

    out2 = build(None)(Self(), q, context, layer, object())
    assert tuple(out2.shape) == (4, 20 * 256)
    assert len(fia_calls) == 2, "old FIA loop must still run per request"
    print("PASS merged consumer: FAS early return, old FIA fallback intact")


def test_plan_build_count_and_step_freshness():
    prefix, seq = (64, 0, 128), (1, 4, 7)
    total = sum(prefix) + sum(seq)
    q = torch.zeros((12, 20, 256), dtype=torch.bfloat16)
    k = torch.zeros((total, 20, 256), dtype=torch.bfloat16)
    v = torch.zeros((total, 20, 256), dtype=torch.bfloat16)
    kv_offsets, q_offsets, kv, qq = [], [], 0, 0
    for p, length in zip(prefix, seq):
        kv_offsets.append(kv)
        q_offsets.append(qq)
        kv += p + length
        qq += length
    context = SimpleNamespace(
        prefix_lens=prefix, seq_lens=seq,
        q_offsets=tuple(q_offsets), kv_offsets=tuple(kv_offsets),
        total_rows=total, k_full=k, v_full=v,
        num_heads=20, qk_head_dim=256, v_head_dim=256, layer_id=3,
    )
    backend, layer = make_backend(), make_layer()
    batch = make_batch(prefix, seq)

    builds = []
    original_build = fas.build_dense_fas_plan

    def counting_build(*args, **kwargs):
        plan = original_build(*args, **kwargs)
        builds.append(plan)
        return plan

    fake, calls = make_fake_fas()
    saved_fas = fas._invoke_public_fas
    fas.build_dense_fas_plan = counting_build
    fas._invoke_public_fas = fake
    try:
        with Patches():
            out1 = fas.maybe_forward_merged_dense_prefill_fas(
                backend, q, context, layer, batch
            )
            plan1 = backend.forward_metadata.dense_fas_plan
            out2 = fas.maybe_forward_merged_dense_prefill_fas(
                backend, q, context, layer, batch
            )
        assert out1 is not None and out2 is not None
        assert len(builds) == 1, "one host plan per step, not per layer"
        assert backend.forward_metadata.dense_fas_plan is plan1
        assert len(calls) == 2, "native FAS still runs once per layer"

        # Same host key, new scheduler step (fresh ForwardMetadata): rebuild.
        backend.forward_metadata = SimpleNamespace()
        with Patches():
            out3 = fas.maybe_forward_merged_dense_prefill_fas(
                backend, q, context, layer, batch
            )
        assert out3 is not None and len(builds) == 2
        assert backend.forward_metadata.dense_fas_plan is not plan1
        assert len(calls) == 3

        # FAS disabled: no host planning and no native call.
        backend.forward_metadata = SimpleNamespace()
        with Patches(enabled=False):
            out4 = fas.maybe_forward_merged_dense_prefill_fas(
                backend, q, context, layer, batch
            )
        assert out4 is None and len(builds) == 2 and len(calls) == 3
    finally:
        fas.build_dense_fas_plan = original_build
        fas._invoke_public_fas = saved_fas
    print("PASS plan count: once/step, fresh per new metadata, zero when disabled")


def test_core_dispatch_keeps_gate_and_oproj():
    src = (
        REPO
        / "python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py"
    ).read_text()
    tree = ast.parse(src)
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "forward_mha_core_npu"
    )
    body = textwrap.dedent(ast.get_source_segment(src, func))

    class Marker:
        pass

    def build(fas_result):
        ns = {
            "torch": torch,
            "PackedExpandedKV": Marker,
            "MergedPrefixKVProjection": Marker,
            "forward_expanded_prefill": lambda *a, **k: None,
            "get_attn_backend": lambda: object(),
            "maybe_forward_dense_prefill_fas": lambda *a, **k: fas_result,
            "Optional": Optional,
        }
        exec(compile(body, "<core>", "exec"), ns)
        return ns["forward_mha_core_npu"]

    class M:
        num_local_heads = 20
        v_head_dim = 256

        def __init__(self):
            self.radix_calls = 0
            self.gated = 0
            self.o_calls = 0

        def attn_mha(self, q, k, v, fb, save_kv_cache=True):
            self.radix_calls += 1
            assert save_kv_cache is False
            return torch.zeros((q.shape[0], 20, 256), dtype=q.dtype)

        def _apply_gated(self, x, gate):
            self.gated += 1
            return x

        def o_proj(self, x):
            self.o_calls += 1
            return x, None

    q = torch.zeros((4, 20, 256), dtype=torch.bfloat16)
    gate = object()

    m1 = M()
    sentinel = torch.full((4, 20 * 256), 3.0, dtype=torch.bfloat16)
    out = build(sentinel)(m1, q, q, q, object(), gate)
    assert m1.radix_calls == 0 and m1.gated == 1 and m1.o_calls == 1
    assert out.data_ptr() == sentinel.data_ptr()
    assert tuple(out.shape) == (4, 20 * 256)

    m2 = M()
    out2 = build(None)(m2, q, q, q, object(), gate)
    assert m2.radix_calls == 1 and m2.gated == 1 and m2.o_calls == 1
    assert tuple(out2.shape) == (4, 20 * 256)
    print("PASS core dispatch: FAS short-circuit, FIA fallback, gate+o_proj unchanged")


def test_fallbacks_preserve_old_path():
    prefix, seq = [0, 0, 0], [3, 5, 2]
    q = torch.zeros((10, 20, 256), dtype=torch.bfloat16)
    k = torch.zeros((10, 20, 256), dtype=torch.bfloat16)
    v = torch.zeros((10, 20, 256), dtype=torch.bfloat16)
    layer, batch = make_layer(), make_batch(prefix, seq)

    def attempt(**flags):
        backend = make_backend()
        out, calls = run_cold(prefix, seq, q, k, v, layer, backend, batch, flags)
        return out, calls

    assert attempt(enabled=False) == (None, [])
    assert attempt(graph=True) == (None, [])
    assert attempt(backend=False) == (None, [])
    assert attempt(nz=True) == (None, [])
    assert attempt(scatter=True) == (None, [])
    assert attempt(parallel=False) == (None, [])
    assert attempt(npu=False) == (None, [])

    # zero query in the batch
    b0 = make_batch(prefix, [3, 0, 2])
    out, calls = run_cold(prefix, [3, 0, 2], torch.zeros((5, 20, 256), dtype=torch.bfloat16),
                          torch.zeros((5, 20, 256), dtype=torch.bfloat16),
                          torch.zeros((5, 20, 256), dtype=torch.bfloat16),
                          layer, make_backend(), b0)
    assert out is None and calls == []

    # padded query
    bp = make_batch(prefix, seq)
    bp.num_token_non_padded_cpu = 9
    out, calls = run_cold(prefix, seq, q, k, v, layer, make_backend(), bp)
    assert out is None and calls == []

    # non-contiguous query: refused without materialising a copy
    q_nc = torch.zeros((10, 256, 20), dtype=torch.bfloat16).transpose(1, 2)
    assert not q_nc.is_contiguous()
    ptr = q_nc.data_ptr()
    out, calls = run_cold(prefix, seq, q_nc, k, v, layer, make_backend(), batch)
    assert out is None and calls == []
    assert q_nc.data_ptr() == ptr and not q_nc.is_contiguous()

    # dtype / geometry / spec / expanded KV
    for dtype in (torch.float16, torch.float32):
        out, calls = run_cold(prefix, seq, q.to(dtype), k.to(dtype), v.to(dtype),
                              layer, make_backend(), batch)
        assert out is None and calls == []
    for overrides in (
        {"tp_q_head_num": 16},
        {"qk_head_dim": 192},
        {"is_cross_attention": True},
        {"sliding_window_size": 1024},
    ):
        out, calls = run_cold(prefix, seq, q, k, v, make_layer(**overrides),
                              make_backend(), batch)
        assert out is None and calls == []
    out, calls = run_cold(prefix, seq, q, k, v, layer, make_backend(),
                          make_batch(prefix, seq, spec=object()))
    assert out is None and calls == []
    expanded = make_backend()
    expanded.token_to_kv_pool.prefill_expanded_geometry = object()
    out, calls = run_cold(prefix, seq, q, k, v, layer, expanded, batch)
    assert out is None and calls == []

    # ordinary prefix>0 is not routed through current-only k/v
    out, calls = run_cold([5, 0, 0], seq, q, k, v, layer, make_backend(),
                          make_batch([5, 0, 0], seq))
    assert out is None and calls == []

    # inconsistent merged offsets
    ctx = SimpleNamespace(
        prefix_lens=(2,), seq_lens=(3,), q_offsets=(1,), kv_offsets=(0,),
        total_rows=5,
        k_full=torch.zeros((5, 20, 256), dtype=torch.bfloat16),
        v_full=torch.zeros((5, 20, 256), dtype=torch.bfloat16),
    )
    with Patches():
        fake, calls = make_fake_fas()
        saved = fas._invoke_public_fas
        fas._invoke_public_fas = fake
        try:
            out = fas.maybe_forward_merged_dense_prefill_fas(
                make_backend(), torch.zeros((3, 20, 256), dtype=torch.bfloat16),
                ctx, layer, make_batch((2,), (3,)),
            )
        finally:
            fas._invoke_public_fas = saved
    assert out is None and calls == []
    print("PASS fallbacks: flag0/zero/padded/noncontig/dtype/geometry/spec/capture/pool/prefix")


def test_prepare_cache_writer_still_present():
    src = (
        REPO
        / "python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py"
    ).read_text()
    prepare = src.split("def forward_mha_prepare_npu", 1)[1].split(
        "def forward_mha_core_npu", 1
    )[0]
    assert "maybe_unified_kv_rmsnorm_rope_cache(" in prepare
    assert "maybe_forward_split_kv_b_proj(" in prepare
    assert "save_kv_cache=False" in src.split("def forward_mha_core_npu", 1)[1]
    print("PASS handshake: original latent cache writer/split/save_kv_cache retained")


def main():
    test_cold_multiquery_one_call_and_identity()
    test_merged_ragged_prefix_and_cumulative_kv()
    test_merged_early_return_and_fia_fallback_in_real_consumer()
    test_core_dispatch_keeps_gate_and_oproj()
    test_plan_build_count_and_step_freshness()
    test_fallbacks_preserve_old_path()
    test_prepare_cache_writer_still_present()
    print("ALL DENSE PREFILL FAS CPU TESTS PASSED")


if __name__ == "__main__":
    main()
