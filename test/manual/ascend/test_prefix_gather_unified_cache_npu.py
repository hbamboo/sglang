"""Physical-NPU tests for the page-first unified prefix gather.

Runs on the authorized physical NPU only (``ASCEND_RT_VISIBLE_DEVICES=5``,
visible index 0).  No model service is started.  The suite:

  * uses the REAL ``NPUMLATokenToKVPool`` (unified 576 layout) getters and
    compares the new gather bitwise with the old two-field ``index_select``
    route for BF16 and FP16, unsorted/repeated/first/last/dummy/empty pages;
  * validates nonzero ``start_layer`` / local layer indexing, nonzero storage
    offset, rank-3 and rank-4 packed views, an independent per-page golden and
    a source-cache immutability check;
  * allocates one REAL full 7133-page, 47-layer shared backing (~49 GiB) and
    validates selected 20/30-page gathers for the first/middle/last layer, so
    the mechanism is proven on the actual model geometry, not a tiny fixture.

Full geometry is opt-in via ``PREFIX_GATHER_FULL_GEOMETRY=1`` because of the
~49 GiB allocation; the small value tests always run.
"""

import os

import pytest
import torch
import torch_npu

torch.npu.set_device(0)

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool  # noqa: E402
from sglang.srt.hardware_backend.npu.modules import (  # noqa: E402
    prefix_gather_unified_cache_npu as helper,
)

_KV_LATENT = 512
_KV_ROPE = 64
_KV_WIDTH = 576
_PAGE = 128
_DEV = "npu"


def _enable(monkeypatch):
    monkeypatch.setenv("SGLANG_NPU_UNIFIED_PREFIX_GATHER", "1")
    monkeypatch.delenv("SGLANG_USE_FIA_NZ", raising=False)
    monkeypatch.delenv("ENABLE_QKV_NORM_ROPE_OPT", raising=False)
    helper.clear_gate_cache()


def _make_pool(blocks=8, layer_num=3, *, dtype=torch.bfloat16, start_layer=None):
    return NPUMLATokenToKVPool(
        size=(blocks - 1) * _PAGE,
        page_size=_PAGE,
        dtype=dtype,
        kv_lora_rank=_KV_LATENT,
        qk_rope_head_dim=_KV_ROPE,
        layer_num=layer_num,
        device=_DEV,
        enable_memory_saver=False,
        start_layer=start_layer,
    )


def _fill_layer(pool, local_layer, dtype):
    """Deterministic values that differ per (page, row, column)."""
    blocks = pool.kv_buffer.shape[1]
    pg = torch.arange(blocks, dtype=torch.float32, device=_DEV).view(blocks, 1, 1, 1)
    rw = torch.arange(_PAGE, dtype=torch.float32, device=_DEV).view(1, _PAGE, 1, 1)
    cl = torch.arange(_KV_WIDTH, dtype=torch.float32, device=_DEV).view(
        1, 1, 1, _KV_WIDTH
    )
    vals = ((pg * 131 + rw * 3 + cl) % 251).to(dtype)
    pool.kv_buffer[local_layer].copy_(vals)


def _old_two_field(pool, layer_id, page_idx):
    k_buffer = pool.get_key_buffer(layer_id)
    v_buffer = pool.get_value_buffer(layer_id)
    kv_cached = torch.index_select(k_buffer, 0, page_idx)
    k_rope_cached = torch.index_select(v_buffer, 0, page_idx).flatten(0, 1)
    return kv_cached, k_rope_cached


def _independent_golden(pool, local_layer, pages, dtype):
    """Per-page read-back, independent of index_select on the field views."""
    if not pages:
        return (
            torch.empty(0, _PAGE, 1, _KV_LATENT, dtype=dtype),
            torch.empty(0, 1, _KV_ROPE, dtype=dtype),
        )
    per_page = [pool.kv_buffer[local_layer, p].cpu() for p in pages]
    packed = torch.stack(per_page, 0)
    return packed[..., :_KV_LATENT].clone(), packed[..., _KV_LATENT:].reshape(
        len(pages) * _PAGE, 1, _KV_ROPE
    )


PAGE_SETS = [
    ([7, 0, 3, 7, 1], "mixed-unsorted-repeat-first-last-dummy"),
    ([0], "dummy-page-0"),
    ([7], "last-page"),
    ([], "empty"),
]


@pytest.mark.parametrize("pages,label", PAGE_SETS, ids=[p[1] for p in PAGE_SETS])
@pytest.mark.parametrize("idx_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_small_bitwise_old_vs_new(monkeypatch, dtype, idx_dtype, pages, label):
    _enable(monkeypatch)
    pool = _make_pool(blocks=8, layer_num=3, dtype=dtype)
    for local in range(3):
        _fill_layer(pool, local, dtype)
    torch.npu.synchronize()

    for layer_id in (0, 1, 2):  # first / middle / last
        page_idx = torch.tensor(pages, dtype=idx_dtype, device=_DEV)
        before = pool.kv_buffer[layer_id].clone()
        new = helper.maybe_gather_unified_prefix_kv(pool, layer_id, page_idx)
        old = _old_two_field(pool, layer_id, page_idx)
        assert new is not None, f"helper fell back for {label}"
        assert torch.equal(new[0], old[0])
        assert torch.equal(new[1], old[1])
        assert tuple(new[0].shape) == (len(pages), _PAGE, 1, _KV_LATENT)
        assert tuple(new[1].shape) == (len(pages) * _PAGE, 1, _KV_ROPE)
        assert new[0].dtype == dtype and new[1].dtype == dtype
        exp_kv, exp_rope = _independent_golden(pool, layer_id, pages, dtype)
        assert torch.equal(new[0].cpu(), exp_kv)
        assert torch.equal(new[1].cpu(), exp_rope)
        assert torch.equal(pool.kv_buffer[layer_id], before), "source mutated"


def test_nonzero_start_layer_uses_local_index(monkeypatch):
    _enable(monkeypatch)
    pool = _make_pool(blocks=6, layer_num=2, start_layer=5)
    _fill_layer(pool, 0, torch.bfloat16)
    _fill_layer(pool, 1, torch.bfloat16)
    # The two layers must differ so a wrong local index is visible.
    pool.kv_buffer[1].add_(1.0)
    torch.npu.synchronize()

    pages = [4, 1, 4, 0]
    page_idx = torch.tensor(pages, dtype=torch.int64, device=_DEV)
    for global_layer, local in ((5, 0), (6, 1)):
        new = helper.maybe_gather_unified_prefix_kv(pool, global_layer, page_idx)
        old = _old_two_field(pool, global_layer, page_idx)
        assert new is not None
        assert torch.equal(new[0], old[0])
        assert torch.equal(new[1], old[1])
        exp_kv, _ = _independent_golden(pool, local, pages, torch.bfloat16)
        assert torch.equal(new[0].cpu(), exp_kv)
    # Cross-layer bleed check: layer 5 and 6 outputs differ.
    a = helper.maybe_gather_unified_prefix_kv(pool, 5, page_idx)
    b = helper.maybe_gather_unified_prefix_kv(pool, 6, page_idx)
    assert not torch.equal(a[0], b[0])


class _Rank3Pool:
    """Pool exposing the packed ``[rows, 1, 576]`` layer form."""

    def __init__(self, view, start_layer=0):
        self.kv_buffer = [view]
        self.unified_kv = True
        self.use_dsa = False
        self.kv_lora_rank = _KV_LATENT
        self.qk_rope_head_dim = _KV_ROPE
        self.page_size = _PAGE
        self.prefill_expanded_geometry = None
        self.dtype = view.dtype
        self.store_dtype = view.dtype
        self.start_layer = start_layer

    def get_kv_buffer(self, layer_id):
        view = self.kv_buffer[layer_id - self.start_layer]
        return view[..., :_KV_LATENT], view[..., _KV_LATENT:]


def test_rank3_packed_view_matches_rank4(monkeypatch):
    _enable(monkeypatch)
    pages, page_size, width = 8, _PAGE, _KV_WIDTH
    pg = torch.arange(pages, dtype=torch.float32, device=_DEV).view(pages, 1, 1, 1)
    rw = torch.arange(page_size, dtype=torch.float32, device=_DEV).view(
        1, page_size, 1, 1
    )
    cl = torch.arange(width, dtype=torch.float32, device=_DEV).view(1, 1, 1, width)
    packed = ((pg * 131 + rw * 3 + cl) % 251).to(torch.bfloat16)
    rank4 = packed.clone()
    rank3 = packed.reshape(pages * page_size, 1, width)
    torch.npu.synchronize()

    idx = torch.tensor([6, 2, 2, 0], dtype=torch.int64, device=_DEV)
    out3 = helper.maybe_gather_unified_prefix_kv(_Rank3Pool(rank3), 0, idx)
    out4 = helper.maybe_gather_unified_prefix_kv(_Rank3Pool(rank4), 0, idx)
    assert out3 is not None and out4 is not None
    assert torch.equal(out3[0], out4[0])
    assert torch.equal(out3[1], out4[1])
    assert torch.equal(out3[0].cpu(), rank4[torch.tensor([6, 2, 2, 0])][..., :_KV_LATENT].cpu())


@pytest.mark.skipif(
    os.environ.get("PREFIX_GATHER_FULL_GEOMETRY") != "1",
    reason="set PREFIX_GATHER_FULL_GEOMETRY=1 to allocate the ~49 GiB 47-layer pool",
)
def test_full_geometry_7133_pages_47_layers(monkeypatch):
    _enable(monkeypatch)
    blocks = 7133
    layer_num = 47
    pool = _make_pool(blocks=blocks, layer_num=layer_num, dtype=torch.bfloat16)
    assert pool.kv_buffer.shape[1] == blocks
    dtype = torch.bfloat16

    tested_layers = (0, layer_num // 2, layer_num - 1)
    for local in tested_layers:
        _fill_layer(pool, local, dtype)
    torch.npu.synchronize()

    selections = {
        "20": [7000, 0, 3566, 42, 7132, 1, 5000, 2500, 1234, 6000,
               17, 6999, 300, 4000, 2, 7131, 88, 512, 1024, 2048],
    }
    # 30 pages: the 20 above plus 10 more, with a duplicate and last page.
    selections["30"] = selections["20"] + [
        3, 7130, 4444, 5555, 6666, 111, 222, 333, 7000, 42,
    ]
    for name, pages in selections.items():
        idx = torch.tensor(pages, dtype=torch.int32, device=_DEV)
        for local in tested_layers:
            before = pool.kv_buffer[local][torch.tensor(pages, device=_DEV)].clone()
            new = helper.maybe_gather_unified_prefix_kv(pool, local, idx)
            old = _old_two_field(pool, local, idx)
            assert new is not None, f"helper fell back at {name} pages layer {local}"
            assert torch.equal(new[0], old[0]), f"kv mismatch {name}/{local}"
            assert torch.equal(new[1], old[1]), f"rope mismatch {name}/{local}"
            exp_kv, exp_rope = _independent_golden(pool, local, pages, dtype)
            assert torch.equal(new[0].cpu(), exp_kv)
            assert torch.equal(new[1].cpu(), exp_rope)
            after = pool.kv_buffer[local][torch.tensor(pages, device=_DEV)]
            assert torch.equal(after, before), "source selected pages mutated"
            print(
                f"[full-geom] layer={local} pages={name} "
                f"kv={tuple(new[0].shape)} rope={tuple(new[1].shape)}"
            )


def test_helper_source_has_no_host_sync_calls():
    import inspect

    src = inspect.getsource(helper.maybe_gather_unified_prefix_kv)
    for forbidden in (".item(", ".tolist(", ".numpy(", ".nonzero(", ".cpu("):
        assert forbidden not in src, f"host sync/cpu call in hot path: {forbidden}"
