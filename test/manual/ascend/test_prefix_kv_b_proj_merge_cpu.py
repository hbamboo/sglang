"""CPU-only tests for the merged prefix/current KV-B projection route.

No NPU device is opened.  The gather helper and the backend-type predicate are
replaced with CPU fakes, and the real split-KV weight layout / GEMM helper is
exercised against an independent per-request golden.

Covered:

  * layout derivation for one/two/four ragged requests, mixed zero/nonzero
    prefixes, a zero-query request among active requests, shared/duplicate/
    unsorted pages, page-padding trim and host page-count validation;
  * pure packing with distinct sentinels: per-request ``[prefix, current]``
    order, sources unchanged, bitwise output checks;
  * FIA interval slices for K/V/q against an independent per-request golden;
  * the merge glue on CPU: exactly two KV-B GEMMs at M=sum(prefix+current),
    no current-only/prefix-only projection, the old separate split + original
    packed projection compared against the merged result for FP32/BF16 with
    real per-head split weights and bias, plus the documented fallbacks;
  * the non-GEMM readiness split does not weaken the stale-weight guard.

Run:  python3 test/manual/ascend/test_prefix_kv_b_proj_merge_cpu.py
"""

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = os.environ.get(
    "PREFIX_KVB_MERGE_REPO", str(Path(__file__).resolve().parents[3])
)
_PYTHON = str(Path(REPO) / "python")
if _PYTHON not in sys.path:
    sys.path.insert(0, _PYTHON)

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.npu.modules import (
    prefix_kv_b_proj_npu as pk,
)
from sglang.srt.hardware_backend.npu.modules import (
    prefix_gather_unified_cache_npu as pg,
)
from sglang.srt.hardware_backend.npu.modules import split_kv_b_proj as skv

H = 20
DK = 192
DV = 256
K = 512
ROPE = 64
PAGE = 128


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class UnquantizedLinearMethod:
    pass


class ColumnParallelLinear(torch.nn.Module):
    pass


def make_module(heads=H, dk=DK, dv=DV, k=K, *, dtype=torch.bfloat16, bias=False):
    module = ColumnParallelLinear()
    out = heads * (dk + dv)
    module.weight = torch.nn.Parameter(torch.zeros(out, k, dtype=dtype))
    module.quant_method = UnquantizedLinearMethod()
    module.bias = (
        torch.nn.Parameter(torch.zeros(out, dtype=dtype)) if bias else None
    )
    module.gather_output = False
    module.skip_bias_add = False
    module.output_size_per_partition = out
    module.input_size = k
    module.tp_size = 1
    return module


def randomize_module(module, seed=0):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        module.weight.copy_(
            torch.randn(module.weight.shape, generator=gen, dtype=torch.float32)
            .to(module.weight.dtype)
            .mul_(0.05)
        )
        if module.bias is not None:
            module.bias.copy_(
                torch.randn(module.bias.shape, generator=gen, dtype=torch.float32)
                .to(module.bias.dtype)
                .mul_(0.01)
            )


def prepare_cpu_state(attn, module):
    """Prepare the real split state on CPU by skipping the NPU device gate."""
    original = skv.unsupported_reason
    skv.unsupported_reason = lambda attn_, module_, **kwargs: None
    try:
        state, reason = skv.prepare_split_kv_b_proj(attn, module, force=True)
    finally:
        skv.unsupported_reason = original
    assert state is not None, reason
    return state


class FakeLayer:
    def __init__(self, module, heads=H, layer_id=3, qk=DK + ROPE, v=DV, **kw):
        self.kv_b_proj = module
        self.qk_head_dim = qk
        self.v_head_dim = v
        self.tp_q_head_num = heads
        self.tp_k_head_num = heads
        self.tp_v_head_num = heads
        self.is_cross_attention = False
        self.sliding_window_size = -1
        self.layer_id = layer_id
        for key, value in kw.items():
            setattr(self, key, value)


class FakeAttn:
    def __init__(self, module, heads=H, layer_id=3, **layer_kw):
        self.num_local_heads = heads
        self.qk_nope_head_dim = DK
        self.qk_rope_head_dim = ROPE
        self.v_head_dim = DV
        self.kv_lora_rank = K
        self.use_dsa = False
        self.use_deepseek_yarn_rope = True
        self.layer_id = layer_id
        self.attn_mha = FakeLayer(module, heads=heads, layer_id=layer_id, **layer_kw)

    def _concat_and_cast_mha_k(self, k_nope, k_pe, forward_batch):
        k = k_nope.new_empty(
            (k_nope.shape[0], self.num_local_heads, DK + ROPE)
        )
        k[..., :DK] = k_nope
        k[..., DK:] = k_pe
        return k


class FakePool:
    def __init__(self, dtype=torch.bfloat16, page_size=PAGE):
        self.unified_kv = True
        self.use_dsa = False
        self.kv_lora_rank = K
        self.qk_rope_head_dim = ROPE
        self.page_size = page_size
        self.prefill_expanded_geometry = None
        self.dtype = dtype
        self.store_dtype = dtype
        self.start_layer = 0


def make_backend(pool, prefix_lens, seq_lens, num_pages):
    return SimpleNamespace(
        use_mla=True,
        token_to_kv_pool=pool,
        attn_cp_size=1,
        graph_mode=False,
        enable_torch_compile=False,
        is_dllm_model=False,
        forward_metadata=SimpleNamespace(
            prefix_lens=torch.tensor(list(prefix_lens), dtype=torch.int64),
            extend_seq_lens_cpu_int=torch.tensor(list(seq_lens), dtype=torch.int32),
            flatten_prefix_block_tables=torch.arange(
                num_pages, dtype=torch.int32
            ),
        ),
    )


def make_forward_batch(prefix_lens, seq_lens):
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        extend_prefix_lens_cpu=list(prefix_lens),
        extend_seq_lens_cpu=list(seq_lens),
        attn_cp_metadata=None,
        encoder_lens=None,
    )


class Patches:
    """Context manager for the helper's external predicates."""

    def __init__(self, *, enabled=True, graph=False, gather=None, backend_ok=True):
        self.enabled = enabled
        self.graph = graph
        self.gather = gather
        self.backend_ok = backend_ok
        self._saved = []

    def _set(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def __enter__(self):
        self._set(pk, "merge_prefix_kv_b_proj_enabled", lambda: self.enabled)
        self._set(pk, "_graph_or_compile_active", lambda: self.graph)
        self._set(pk, "_backend_is_supported", lambda backend: self.backend_ok)
        # No NPU forward context exists on CPU; the real production guard sees
        # the token pool and expanded-KV geometry is None in these fixtures.
        self._set(skv, "_expanded_kv_may_fuse", lambda forward_batch: False)
        if self.gather is not None:
            self._set(pg, "maybe_gather_unified_prefix_kv", self.gather)
        return self

    def __exit__(self, *exc):
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)
        return False


def history_tensors(num_pages, dtype=torch.float32, seed=1, tag=0.0, scale=1000.0):
    """Synthetic gathered pages with row-unique sentinels."""
    rows = num_pages * PAGE
    base = torch.arange(rows, dtype=torch.float32).view(rows, 1)
    latent = (base * scale + tag).repeat(1, K).to(dtype)
    rope = (base + tag).repeat(1, ROPE).to(dtype)
    return latent, rope.view(rows, 1, ROPE)


def current_tensors(sum_q, dtype=torch.float32, seed=2, tag=0.0, scale=1000.0, offset=500.0):
    base = torch.arange(sum_q, dtype=torch.float32).view(sum_q, 1)
    latent = (base * scale + offset + tag).repeat(1, K).to(dtype)
    rope = (base + offset + tag).repeat(1, ROPE).to(dtype)
    return latent, rope


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------
def test_layout_ragged_and_edges():
    one = pk.build_merge_layout([7], [3], 1)
    assert one is not None
    assert one.pages_per_req == (1,)
    assert one.kv_offsets == (0,) and one.q_offsets == (0,)
    assert one.total_rows == 10 and one.sum_prefix == 7 and one.sum_seq == 3

    two = pk.build_merge_layout([0, 300], [5, 0], 3)
    assert two is not None
    assert two.pages_per_req == (0, 3)
    assert two.kv_offsets == (0, 5)
    assert two.q_offsets == (0, 5)
    assert two.total_rows == 305 and two.total_pages == 3

    four = pk.build_merge_layout([1, 128, 129, 0], [1, 2, 0, 3], 4)
    assert four is not None
    assert four.pages_per_req == (1, 1, 2, 0)
    assert four.page_offsets == (0, 1, 2, 4)
    assert four.kv_offsets == (0, 2, 132, 261)
    assert four.q_offsets == (0, 1, 3, 3)
    assert four.total_rows == 264 and four.total_pages == 4

    # Page alignment/padding: p=100 needs one 128-row page; 129 needs two.
    padded = pk.build_merge_layout([100, 129], [4, 4], 3)
    assert padded is not None and padded.pages_per_req == (1, 2)

    assert pk.build_merge_layout([100], [4], 0) is None  # missing page
    assert pk.build_merge_layout([128], [4], 2) is None  # extra page
    assert pk.build_merge_layout([0, 0], [1, 1], 0) is None  # no prefix pages
    assert pk.build_merge_layout([10], [0], 1) is None  # no new queries
    assert pk.build_merge_layout([10, 5], [1], 2) is None  # length mismatch
    assert pk.build_merge_layout([-1], [1], 1) is None
    assert pk.build_merge_layout([1.5], [1], 1) is None
    assert pk.build_merge_layout([True], [1], 1) is None
    assert pk.build_merge_layout(None, [1], 1) is None
    assert pk.build_merge_layout([1], [1], -1) is None
    print("PASS layout: ragged/zero/shared-page/padding and rejections")


# ---------------------------------------------------------------------------
# packing
# ---------------------------------------------------------------------------
def test_pack_order_and_immutability():
    prefix = [3, 0, 130, 1]
    seq = [2, 4, 0, 1]
    pages = 1 + 0 + 2 + 1
    layout = pk.build_merge_layout(prefix, seq, pages)
    assert layout is not None

    hist_lat, hist_rope = history_tensors(pages)
    cur_lat, cur_rope = current_tensors(sum(seq))
    src = [t.clone() for t in (hist_lat, hist_rope, cur_lat, cur_rope)]

    out = pk.pack_merged_rows(hist_lat, hist_rope, cur_lat, cur_rope, layout)
    assert out is not None
    merged_lat, merged_rope = out
    assert merged_lat.shape == (layout.total_rows, K)
    assert merged_rope.shape == (layout.total_rows, 1, ROPE)
    assert merged_lat.is_contiguous() and merged_rope.is_contiguous()

    for j, (p, q) in enumerate(zip(prefix, seq)):
        kv_row = layout.kv_offsets[j]
        hist_row = layout.page_offsets[j] * PAGE
        for t in range(p):
            # History rows come from the request's page block, trimmed to p.
            assert merged_lat[kv_row + t, 0].item() == float((hist_row + t) * 1000)
            assert merged_rope[kv_row + t, 0, 0].item() == float(hist_row + t)
        for t in range(q):
            assert merged_lat[kv_row + p + t, 0].item() == float(
                (layout.q_offsets[j] + t) * 1000 + 500
            )
            assert merged_rope[kv_row + p + t, 0, 0].item() == float(
                layout.q_offsets[j] + t + 500
            )
        # Full rows, not only column 0.
        assert torch.all(merged_lat[kv_row + p : kv_row + p + q] == merged_lat[
            kv_row + p : kv_row + p + q, :1
        ])

    for before, after in zip(src, (hist_lat, hist_rope, cur_lat, cur_rope)):
        assert torch.equal(before, after)
    print("PASS pack: [p,c] per-request order, padding trimmed, sources unchanged")


def test_pack_bitwise_bf16():
    prefix = [5, 2]
    seq = [3, 4]
    layout = pk.build_merge_layout(prefix, seq, 2)
    hist_lat, hist_rope = history_tensors(2, dtype=torch.bfloat16)
    cur_lat, cur_rope = current_tensors(sum(seq), dtype=torch.bfloat16)
    merged_lat, merged_rope = pk.pack_merged_rows(
        hist_lat, hist_rope, cur_lat, cur_rope, layout
    )
    assert merged_lat.dtype == torch.bfloat16 and merged_rope.dtype == torch.bfloat16
    # Recompute the expected layout independently and compare bitwise.
    expected = torch.empty_like(merged_lat)
    expected_rope = torch.empty_like(merged_rope)
    for j, (p, q) in enumerate(zip(prefix, seq)):
        kv_row = layout.kv_offsets[j]
        hist_row = layout.page_offsets[j] * PAGE
        expected[kv_row : kv_row + p] = hist_lat.view(-1, K)[hist_row : hist_row + p]
        expected[kv_row + p : kv_row + p + q] = cur_lat[
            layout.q_offsets[j] : layout.q_offsets[j] + q
        ]
        expected_rope[kv_row : kv_row + p] = hist_rope.view(-1, 1, ROPE)[
            hist_row : hist_row + p
        ]
        expected_rope[kv_row + p : kv_row + p + q] = cur_rope[
            layout.q_offsets[j] : layout.q_offsets[j] + q
        ].view(-1, 1, ROPE)
    assert torch.equal(merged_lat, expected)
    assert torch.equal(merged_rope, expected_rope)
    print("PASS pack: BF16 bitwise vs independent layout")


# ---------------------------------------------------------------------------
# FIA interval slices
# ---------------------------------------------------------------------------
def make_context(prefix, seq):
    total = sum(prefix) + sum(seq)
    kv_offsets = []
    q_offsets = []
    kv = 0
    qq = 0
    for p, q in zip(prefix, seq):
        kv_offsets.append(kv)
        q_offsets.append(qq)
        kv += p + q
        qq += q
    heads = 2
    width = 4
    k_full = torch.arange(total * heads * width, dtype=torch.float32).view(
        total, heads, width
    )
    v_full = -k_full - 1
    return pk.MergedPrefixKVProjection(
        k_full=k_full,
        v_full=v_full,
        prefix_lens=tuple(prefix),
        seq_lens=tuple(seq),
        kv_offsets=tuple(kv_offsets),
        q_offsets=tuple(q_offsets),
        total_rows=total,
        num_heads=heads,
        qk_head_dim=width,
        v_head_dim=width,
        layer_id=0,
    ), torch.arange(sum(seq) * heads * width, dtype=torch.float32).view(
        sum(seq), heads, width
    ) + 100000


def test_fia_interval_slices():
    prefix = [3, 0, 4, 1]
    seq = [2, 0, 1, 3]
    context, q = make_context(prefix, seq)
    seen = []
    for req, q_start, kv_start, kv_end, q_len in pk.iter_fia_request_slices(context):
        seen.append((req, q_start, kv_start, kv_end, q_len))
        p = prefix[req]
        assert kv_end - kv_start == p + q_len
        # Independent golden: the request's block is [history p, current q].
        expected_k = context.k_full[kv_start:kv_end]
        expected_v = context.v_full[kv_start:kv_end]
        assert torch.equal(expected_k, context.k_full[kv_start : kv_start + p + q_len])
        assert torch.equal(expected_v, context.v_full[kv_start : kv_start + p + q_len])
        if q_len:
            expected_q = q[q_start : q_start + q_len]
            assert expected_q.shape[0] == q_len
            assert expected_q[0, 0, 0].item() == 100000 + q_start * context.num_heads * 4
    assert seen == [(0, 0, 0, 5, 2), (1, 2, 5, 5, 0), (2, 2, 5, 10, 1), (3, 3, 10, 14, 3)]
    # q=0 keeps both cursors correct: request 2's query starts at global row 2.
    print("PASS FIA intervals: per-request K/V/q golden incl. p=0 and q=0")


# ---------------------------------------------------------------------------
# glue: exactly two GEMMs over all rows
# ---------------------------------------------------------------------------
def _run_merge_glue(
    *,
    prefix,
    seq,
    pages,
    dtype=torch.bfloat16,
    bias=False,
    state_tp=None,
    layer_kw=None,
    extra_rows=0,
):
    module = make_module(dtype=dtype, bias=bias)
    randomize_module(module)
    if state_tp is not None:
        module.tp_size = state_tp
    attn = FakeAttn(module, **(layer_kw or {}))
    prepare_cpu_state(attn, module)
    pool = FakePool(dtype=dtype)
    backend = make_backend(pool, prefix, seq, pages)
    fb = make_forward_batch(prefix, seq)

    total_cur = sum(seq) + extra_rows
    # FP16 has a ~65504 range; keep the synthetic row sentinels in range.  The
    # GEMM comparison uses the same packed tensor, so uniqueness is irrelevant
    # for the structural checks in this harness.
    scale = 10.0 if dtype == torch.float16 else 1000.0
    offset = 5.0 if dtype == torch.float16 else 500.0
    hist_lat, hist_rope = history_tensors(pages, dtype=dtype, scale=scale)
    cur_lat, cur_rope = current_tensors(
        total_cur, dtype=dtype, scale=scale, offset=offset
    )
    kv_a = cur_lat.view(total_cur, 1, 1, K)
    k_pe = cur_rope.view(total_cur, 1, ROPE)
    q = torch.zeros(total_cur, H, DK + ROPE, dtype=dtype)

    gather_calls = []

    def fake_gather(pool_, layer_id, page_indices):
        gather_calls.append((layer_id, int(page_indices.numel())))
        return hist_lat.view(pages, PAGE, 1, K), hist_rope.view(
            pages * PAGE, 1, ROPE
        )

    original_linear = torch.nn.functional.linear
    linear_calls = []

    def counting_linear(inp, weight, bias_=None):
        linear_calls.append((tuple(inp.shape), tuple(weight.shape), bias_ is not None))
        return original_linear(inp, weight, bias_)

    with Patches(gather=fake_gather):
        torch.nn.functional.linear = counting_linear
        try:
            context = pk.maybe_merge_prefix_kv_b_proj(
                attn,
                module,
                kv_a,
                k_pe,
                q,
                fb,
                unified_cache_ready=True,
                backend=backend,
                pool=pool,
            )
        finally:
            torch.nn.functional.linear = original_linear

    return (
        context,
        linear_calls,
        gather_calls,
        module,
        pool,
        backend,
        fb,
        hist_lat,
        hist_rope,
        cur_lat,
        cur_rope,
        q,
        prefix,
        seq,
        pages,
    )


def test_glue_two_gemms_and_golden():
    prefix = [3, 0, 130, 1]
    seq = [2, 4, 0, 1]
    pages = 1 + 0 + 2 + 1
    (
        context,
        linear_calls,
        gather_calls,
        module,
        pool,
        backend,
        fb,
        hist_lat,
        hist_rope,
        cur_lat,
        cur_rope,
        q,
        prefix,
        seq,
        pages,
    ) = _run_merge_glue(prefix=prefix, seq=seq, pages=pages)

    total = sum(prefix) + sum(seq)
    assert context is not None
    assert gather_calls == [(3, pages)]
    assert len(linear_calls) == 2, linear_calls
    widths = sorted(call[1][0] for call in linear_calls)
    assert widths == [H * DK, H * DV], widths
    assert all(call[0] == (total, K) for call in linear_calls)
    assert all(call[2] is False for call in linear_calls)
    assert all(call[1][1] == K for call in linear_calls)

    layout = pk.build_merge_layout(prefix, seq, pages)
    merged_lat, merged_rope = pk.pack_merged_rows(
        hist_lat, hist_rope, cur_lat, cur_rope, layout
    )
    state = skv.split_kv_b_proj_state(module)
    exp_k = F.linear(merged_lat, state.w_k, state.b_k).view(total, H, DK)
    exp_v = F.linear(merged_lat, state.w_v, state.b_v).view(total, H, DV)
    assert torch.equal(context.k_full[..., :DK], exp_k)
    assert torch.equal(context.v_full, exp_v)
    assert torch.equal(context.k_full[..., DK:], merged_rope.expand(-1, H, -1))
    assert context.k_full.shape == (total, H, DK + ROPE)
    assert context.v_full.shape == (total, H, DV)
    assert context.k_full.is_contiguous() and context.v_full.is_contiguous()
    assert context.kv_offsets == layout.kv_offsets
    assert context.q_offsets == layout.q_offsets
    assert context.total_rows == total

    # Only the first sum(seq) latent rows are current tokens.
    assert len(linear_calls[0][0]) == 2
    print(
        f"PASS glue: 2 GEMMs M={total} -> {H * DK}/{H * DV}; "
        f"gather once at layer 3"
    )


def test_real_glm_geometry_256():
    """Independent real-model geometry: H20, latent512, rope64, Dk192, Dv256.

    Q/K head width must be 192+64=256 (the FIA width), never the packed
    K_nope+V width 448.  With all other gates satisfied the real helper must
    return K/V ``[M, 20, 256]`` and exactly the two split GEMMs with the
    measured 3840 / 5120 output widths.
    """
    prefix = [37, 128]
    seq = [11, 5]
    pages = 1 + 1
    # Literal expected dimensions, not imported from the implementation.
    expected_heads = 20
    expected_latent = 512
    expected_rope = 64
    expected_qk = 192 + 64
    expected_k_rows = 20 * 192
    expected_v_rows = 20 * 256

    (
        context,
        linear_calls,
        gather_calls,
        module,
        pool,
        backend,
        fb,
        hist_lat,
        hist_rope,
        cur_lat,
        cur_rope,
        q,
        prefix,
        seq,
        pages,
    ) = _run_merge_glue(prefix=prefix, seq=seq, pages=pages)

    total = sum(prefix) + sum(seq)
    assert total == 181
    assert q.shape == (sum(seq), expected_heads, expected_qk), q.shape
    assert context is not None, "real 192+64=256 geometry must be eligible"
    assert context.qk_head_dim == expected_qk
    assert context.v_head_dim == 256
    assert context.num_heads == expected_heads
    assert context.k_full.shape == (total, expected_heads, expected_qk)
    assert context.v_full.shape == (total, expected_heads, 256)
    assert context.k_full.is_contiguous() and context.v_full.is_contiguous()
    assert len(linear_calls) == 2, linear_calls
    assert sorted(call[1][0] for call in linear_calls) == [
        expected_k_rows,
        expected_v_rows,
    ]
    assert all(call[0] == (total, expected_latent) for call in linear_calls)
    assert gather_calls == [(3, pages)]

    # The packed 448 per-head width (K_nope+V) must be rejected, both on the
    # layer Q/K width and on the query tensor itself.
    bad_layer = _run_merge_glue(
        prefix=prefix, seq=seq, pages=pages, layer_kw={"qk": 448}
    )
    assert bad_layer[0] is None
    assert bad_layer[1] == [] and bad_layer[2] == []

    # An actual q with last dim 448 is likewise not this geometry.
    module2 = make_module()
    randomize_module(module2)
    attn2 = FakeAttn(module2)
    prepare_cpu_state(attn2, module2)
    pool2 = FakePool()
    backend2 = make_backend(pool2, prefix, seq, pages)
    fb2 = make_forward_batch(prefix, seq)
    h_lat, h_rope = history_tensors(pages, dtype=torch.bfloat16)
    c_lat, c_rope = current_tensors(sum(seq), dtype=torch.bfloat16)
    gather_calls2 = []

    def gather2(pool_, layer_id, page_indices):
        gather_calls2.append(layer_id)
        return h_lat.view(pages, PAGE, 1, K), h_rope.view(pages * PAGE, 1, ROPE)

    original_linear = torch.nn.functional.linear
    calls2 = []

    def counting2(inp, weight, bias_=None):
        calls2.append(tuple(inp.shape))
        return original_linear(inp, weight, bias_)

    with Patches(gather=gather2):
        torch.nn.functional.linear = counting2
        try:
            out = pk.maybe_merge_prefix_kv_b_proj(
                attn2,
                module2,
                c_lat.view(sum(seq), 1, 1, K),
                c_rope.view(sum(seq), 1, ROPE),
                torch.zeros(sum(seq), H, 448, dtype=torch.bfloat16),
                fb2,
                unified_cache_ready=True,
                backend=backend2,
                pool=pool2,
            )
        finally:
            torch.nn.functional.linear = original_linear
    assert out is None and calls2 == [] and gather_calls2 == []
    print(
        "PASS real geometry: 192+64=256 accepted (K/V [181,20,256], "
        "GEMM rows 3840/5120); 448 rejected"
    )


def test_narrow_contract_fallbacks():
    prefix, seq, pages = [5, 3], [4, 2], 2
    # Mismatched V head count must fall back before gather/GEMM.
    ctx, calls, gathers, *_ = _run_merge_glue(
        prefix=prefix,
        seq=seq,
        pages=pages,
        layer_kw={"tp_v_head_num": H - 1},
    )
    assert ctx is None and calls == [] and gathers == []
    # Padded current rows (q.shape[0] > sum(seq)) must fall back before the
    # dedicated consumer could expose uninitialised output rows.
    ctx, calls, gathers, *_ = _run_merge_glue(
        prefix=prefix, seq=seq, pages=pages, extra_rows=2
    )
    assert ctx is None and calls == [] and gathers == []
    print("PASS narrow contract: tp_v mismatch and padded query reject early")


def test_glue_bias_and_bf16_fp16():
    for dtype in (torch.bfloat16, torch.float16):
        prefix = [7, 2]
        seq = [3, 5]
        pages = 1 + 1
        (
            context,
            linear_calls,
            gather_calls,
            module,
            pool,
            backend,
            fb,
            hist_lat,
            hist_rope,
            cur_lat,
            cur_rope,
            q,
            prefix,
            seq,
            pages,
        ) = _run_merge_glue(prefix=prefix, seq=seq, pages=pages, dtype=dtype, bias=True)
        assert context is not None and len(linear_calls) == 2, linear_calls
        assert all(call[2] is True for call in linear_calls)
        layout = pk.build_merge_layout(prefix, seq, pages)
        merged_lat, _ = pk.pack_merged_rows(
            hist_lat, hist_rope, cur_lat, cur_rope, layout
        )
        state = skv.split_kv_b_proj_state(module)
        exp_k = F.linear(merged_lat, state.w_k, state.b_k).view(
            layout.total_rows, H, DK
        )
        assert torch.equal(context.k_full[..., :DK], exp_k)
        assert gather_calls and len(gather_calls) == 1
    print("PASS glue: bias + BF16/FP16 split weights")


def test_glue_fallbacks_do_not_project():
    prefix, seq, pages = [5, 3], [4, 2], 2
    module = make_module()
    randomize_module(module)
    attn = FakeAttn(module)
    state = prepare_cpu_state(attn, module)
    pool = FakePool()
    backend = make_backend(pool, prefix, seq, pages)
    fb = make_forward_batch(prefix, seq)
    hist_lat, hist_rope = history_tensors(pages, dtype=torch.bfloat16)
    cur_lat, cur_rope = current_tensors(sum(seq), dtype=torch.bfloat16)
    kv_a = cur_lat.view(sum(seq), 1, 1, K)
    k_pe = cur_rope.view(sum(seq), 1, ROPE)
    q = torch.zeros(sum(seq), H, DK + ROPE, dtype=torch.bfloat16)

    gather_calls = []

    def fake_gather(pool_, layer_id, page_indices):
        gather_calls.append(layer_id)
        return hist_lat.view(pages, PAGE, 1, K), hist_rope.view(
            pages * PAGE, 1, ROPE
        )

    original_linear = torch.nn.functional.linear
    linear_calls = []

    def counting_linear(inp, weight, bias_=None):
        linear_calls.append(tuple(inp.shape))
        return original_linear(inp, weight, bias_)

    def run(*, unified_cache_ready=True, **patch_kwargs):
        gather_calls.clear()
        linear_calls.clear()
        with Patches(gather=fake_gather, **patch_kwargs):
            torch.nn.functional.linear = counting_linear
            try:
                return pk.maybe_merge_prefix_kv_b_proj(
                    attn,
                    module,
                    kv_a,
                    k_pe,
                    q,
                    fb,
                    unified_cache_ready=unified_cache_ready,
                    backend=backend,
                    pool=pool,
                )
            finally:
                torch.nn.functional.linear = original_linear

    assert run() is not None
    assert len(linear_calls) == 2 and len(gather_calls) == 1

    assert run(enabled=False) is None
    assert linear_calls == [] and gather_calls == []
    assert run(graph=True) is None
    assert linear_calls == [] and gather_calls == []

    # unified writer did not produce the current rows
    with Patches(gather=fake_gather):
        torch.nn.functional.linear = counting_linear
        try:
            out = pk.maybe_merge_prefix_kv_b_proj(
                attn, module, kv_a, k_pe, q, fb,
                unified_cache_ready=False, backend=backend, pool=pool,
            )
        finally:
            torch.nn.functional.linear = original_linear
    assert out is None and linear_calls == [] and gather_calls == []

    # stale split weights: reject before gather or projection
    with torch.no_grad():
        module.weight.add_(0.5)
    assert run() is None
    assert linear_calls == [] and gather_calls == []
    prepare_cpu_state(attn, module)

    # gather helper unsupported -> old path, no projection
    def no_gather(pool_, layer_id, page_indices):
        gather_calls.append(layer_id)
        return None

    with Patches(gather=no_gather):
        torch.nn.functional.linear = counting_linear
        try:
            out = pk.maybe_merge_prefix_kv_b_proj(
                attn, module, kv_a, k_pe, q, fb,
                unified_cache_ready=True, backend=backend, pool=pool,
            )
        finally:
            torch.nn.functional.linear = original_linear
    assert out is None and linear_calls == [] and len(gather_calls) == 1

    # page metadata mismatch rejects before gather
    bad_backend = make_backend(pool, prefix, seq, pages + 1)
    gather_calls.clear()
    with Patches(gather=fake_gather):
        torch.nn.functional.linear = counting_linear
        try:
            out = pk.maybe_merge_prefix_kv_b_proj(
                attn, module, kv_a, k_pe, q, fb,
                unified_cache_ready=True, backend=bad_backend, pool=pool,
            )
        finally:
            torch.nn.functional.linear = original_linear
    assert out is None and linear_calls == [] and gather_calls == []

    # TP>1 split state is not eligible (the glue rejects before gather/GEMM)
    module_tp = make_module()
    randomize_module(module_tp)
    module_tp.tp_size = 2
    attn_tp = FakeAttn(module_tp)
    state_tp = prepare_cpu_state(attn_tp, module_tp)
    assert state_tp.tp_size == 2
    gather_calls.clear()
    with Patches(gather=fake_gather):
        torch.nn.functional.linear = counting_linear
        try:
            out = pk.maybe_merge_prefix_kv_b_proj(
                attn_tp, module_tp, kv_a, k_pe, q, fb,
                unified_cache_ready=True, backend=backend, pool=pool,
            )
        finally:
            torch.nn.functional.linear = original_linear
    assert out is None and linear_calls == [] and gather_calls == []
    print("PASS fallbacks: gate/graph/unified/stale/gather/pages/TP reject early")


# ---------------------------------------------------------------------------
# old separate projections vs merged two GEMMs
# ---------------------------------------------------------------------------
def test_numeric_old_vs_merged_fp32():
    import random

    random.seed(0)
    heads = 4
    prefix = [3, 0, 130, 1]
    seq = [2, 4, 0, 1]
    pages = 1 + 0 + 2 + 1
    layout = pk.build_merge_layout(prefix, seq, pages)
    total = layout.total_rows

    W = torch.randn(heads * (DK + DV), K, dtype=torch.float32)
    bias = torch.randn(heads * (DK + DV), dtype=torch.float32)
    wk, wv, bk, bv = skv.split_kv_b_proj_weight(W, bias, heads, DK, DV)

    hist_lat, hist_rope = history_tensors(pages, dtype=torch.float32)
    cur_lat, cur_rope = current_tensors(sum(seq), dtype=torch.float32)
    merged_lat, _ = pk.pack_merged_rows(
        hist_lat, hist_rope, cur_lat, cur_rope, layout
    )

    # New route: two split GEMMs over ALL merged rows.
    new_k = F.linear(merged_lat, wk, bk).view(total, heads, DK)
    new_v = F.linear(merged_lat, wv, bv).view(total, heads, DV)

    # Old current route: split GEMMs over the current rows only.
    old_cur_k = F.linear(cur_lat, wk, bk).view(sum(seq), heads, DK)
    old_cur_v = F.linear(cur_lat, wv, bv).view(sum(seq), heads, DV)

    # Old historical route: the ORIGINAL packed kv_b_proj over every gathered
    # page row (the backend projects padding too), then per-request slices.
    hist_rows = hist_lat.view(pages * PAGE, K)
    old_pre = F.linear(hist_rows, W, bias).view(pages * PAGE, heads, DK + DV)
    old_pre_k, old_pre_v = old_pre[..., :DK], old_pre[..., DK:]

    max_k = max_v = 0.0
    for j, (p, q) in enumerate(zip(prefix, seq)):
        kv_row = layout.kv_offsets[j]
        hist_row = layout.page_offsets[j] * PAGE
        if p:
            k_err = (new_k[kv_row : kv_row + p] - old_pre_k[hist_row : hist_row + p]).abs()
            v_err = (new_v[kv_row : kv_row + p] - old_pre_v[hist_row : hist_row + p]).abs()
            max_k = max(max_k, k_err.max().item())
            max_v = max(max_v, v_err.max().item())
        if q:
            q_row = layout.q_offsets[j]
            k_err = (
                new_k[kv_row + p : kv_row + p + q]
                - old_cur_k[q_row : q_row + q]
            ).abs()
            v_err = (
                new_v[kv_row + p : kv_row + p + q]
                - old_cur_v[q_row : q_row + q]
            ).abs()
            max_k = max(max_k, k_err.max().item())
            max_v = max(max_v, v_err.max().item())

    # Empty-prefix request has no historical contribution; zero-query request
    # keeps its prefix rows in place.
    assert max_k <= 1e-4, max_k
    assert max_v <= 1e-4, max_v
    print(
        f"PASS numeric FP32: old split+packed vs merged 2 GEMMs "
        f"max_abs K={max_k:.3e} V={max_v:.3e}"
    )


def test_numeric_old_vs_merged_bf16():
    heads = 4
    prefix = [5, 0, 130]
    seq = [3, 2, 0]
    pages = 1 + 0 + 2
    layout = pk.build_merge_layout(prefix, seq, pages)
    total = layout.total_rows

    Wf = torch.randn(heads * (DK + DV), K, dtype=torch.float32).mul_(0.05)
    biasf = torch.randn(heads * (DK + DV), dtype=torch.float32).mul_(0.01)
    W = Wf.bfloat16()
    bias = biasf.bfloat16()
    wk, wv, bk, bv = skv.split_kv_b_proj_weight(W, bias, heads, DK, DV)

    hist_lat, hist_rope = history_tensors(pages, dtype=torch.bfloat16)
    cur_lat, cur_rope = current_tensors(sum(seq), dtype=torch.bfloat16)
    merged_lat, _ = pk.pack_merged_rows(
        hist_lat, hist_rope, cur_lat, cur_rope, layout
    )

    new_k = F.linear(merged_lat, wk, bk).view(total, heads, DK)
    new_v = F.linear(merged_lat, wv, bv).view(total, heads, DV)
    old_cur_k = F.linear(cur_lat, wk, bk).view(sum(seq), heads, DK)
    old_cur_v = F.linear(cur_lat, wv, bv).view(sum(seq), heads, DV)
    old_pre = F.linear(
        hist_lat.view(pages * PAGE, K), W, bias
    ).view(pages * PAGE, heads, DK + DV)
    old_pre_k, old_pre_v = old_pre[..., :DK], old_pre[..., DK:]

    max_k = max_v = 0.0
    mean_k = mean_v = 0.0
    count = 0
    for j, (p, q) in enumerate(zip(prefix, seq)):
        kv_row = layout.kv_offsets[j]
        hist_row = layout.page_offsets[j] * PAGE
        if p:
            dk_ = (new_k[kv_row : kv_row + p].float() - old_pre_k[hist_row : hist_row + p].float()).abs()
            dv_ = (new_v[kv_row : kv_row + p].float() - old_pre_v[hist_row : hist_row + p].float()).abs()
            max_k = max(max_k, dk_.max().item()); mean_k += dk_.mean().item() * p
            max_v = max(max_v, dv_.max().item()); mean_v += dv_.mean().item() * p
            count += p
        if q:
            q_row = layout.q_offsets[j]
            dk_ = (new_k[kv_row + p : kv_row + p + q].float() - old_cur_k[q_row : q_row + q].float()).abs()
            dv_ = (new_v[kv_row + p : kv_row + p + q].float() - old_cur_v[q_row : q_row + q].float()).abs()
            max_k = max(max_k, dk_.max().item()); mean_k += dk_.mean().item() * q
            max_v = max(max_v, dv_.max().item()); mean_v += dv_.mean().item() * q
            count += q
    mean_k /= max(count, 1)
    mean_v /= max(count, 1)
    # BF16 output quantization is ~2^-8 relative; values here are O(1).  A
    # per-row-identical CPU GEMM should stay exact; keep a bound documented by
    # the measurement rather than silently loosening it.
    assert max_k <= 2e-2, (max_k, mean_k)
    assert max_v <= 2e-2, (max_v, mean_v)
    print(
        f"PASS numeric BF16: old split+packed vs merged 2 GEMMs "
        f"max_abs K={max_k:.3e} V={max_v:.3e} mean_abs K={mean_k:.3e} V={mean_v:.3e}"
    )


# ---------------------------------------------------------------------------
# readiness refactor / no host sync in the new module
# ---------------------------------------------------------------------------
def test_split_readiness_refactor():
    module = make_module()
    randomize_module(module)
    attn = FakeAttn(module)
    state = prepare_cpu_state(attn, module)
    x = torch.randn(9, K, dtype=torch.bfloat16)
    assert skv.split_kv_b_proj_state_for_forward(attn, module, None) is state
    assert skv.split_kv_b_proj_input_supported(x, state)
    assert skv.maybe_forward_split_kv_b_proj(attn, module, x, None) is not None
    with torch.no_grad():
        module.weight.add_(1.0)
    assert skv.split_kv_b_proj_state_for_forward(attn, module, None) is None
    assert skv.maybe_forward_split_kv_b_proj(attn, module, x, None) is None
    # A strided input is still rejected by the input half of the readiness
    # split; the state check alone must not accept it.
    strided = x.transpose(0, 1).contiguous().T
    assert not strided.is_contiguous()
    state_again = prepare_cpu_state(attn, module)
    assert skv.split_kv_b_proj_input_supported(strided, state_again) is False
    print("PASS readiness: split helper reuses the untouched stale-weight guard")


def test_no_device_scalar_reads_in_new_module():
    path = Path(pk.__file__)
    tree = ast.parse(path.read_text())
    forbidden = {"item", "tolist", "cpu"}
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in forbidden:
                hits.append((node.func.attr, node.lineno))
    assert hits == [], hits
    print("PASS host-sync audit: no .item/.tolist/.cpu in the new module")


def main():
    test_layout_ragged_and_edges()
    test_pack_order_and_immutability()
    test_pack_bitwise_bf16()
    test_fia_interval_slices()
    test_glue_two_gemms_and_golden()
    test_real_glm_geometry_256()
    test_narrow_contract_fallbacks()
    test_glue_bias_and_bf16_fp16()
    test_glue_fallbacks_do_not_project()
    test_numeric_old_vs_merged_fp32()
    test_numeric_old_vs_merged_bf16()
    test_split_readiness_refactor()
    test_no_device_scalar_reads_in_new_module()
    print("ALL MERGED PREFIX KV-B CPU TESTS PASSED")


if __name__ == "__main__":
    main()
