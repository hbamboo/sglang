"""CPU-only gate/ordering/value tests for the page-first prefix gather helper.

No NPU device is opened.  The packed layer view is a CPU tensor and the strict
NPU view validator is monkeypatched, so this file exercises:

  * every narrow gate rejects before the public getter is called (no side
    effect on the fallback path), including flag off, graph/compile, PA_NZ,
    non-unified pool, DSA, wrong latent/rope/page/store dtype, expanded KV,
    bad page-index metadata and a missing getter;
  * the public ``pool.get_kv_buffer(layer_id)`` (hence the layer-transfer
    wait) runs BEFORE the raw gather and receives the global layer id while
    the wait receives ``layer_id - start_layer``;
  * rank-4 and rank-3 packed views produce the exact original caller shapes,
    with unsorted/repeated/first/last/dummy/empty page selections, values
    matching an independent per-page CPU golden, and no source mutation.
"""

import importlib.util
import os
from pathlib import Path
import types

import pytest
import torch

_KV_LATENT = 512
_KV_ROPE = 64
_KV_WIDTH = 576
_PAGE = 128


def _helper_path() -> Path:
    override = os.environ.get("PREFIX_GATHER_HELPER_PATH")
    if override:
        return Path(override)
    # test/manual/ascend/<this file> -> framework/
    return (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/hardware_backend/npu/modules/prefix_gather_unified_cache_npu.py"
    )


def _load_helper():
    # The module imports the shared unified-view validator from the package, so
    # the framework's python/ directory must be on sys.path (the test runner
    # sets PREFIX_GATHER_REPO/python, same as the other NPU manual tests).
    repo = os.environ.get("PREFIX_GATHER_REPO")
    if repo:
        import sys

        python_dir = str(Path(repo) / "python")
        if python_dir not in sys.path:
            sys.path.insert(0, python_dir)
    path = _helper_path()
    assert path.is_file(), f"helper not found: {path}"
    spec = importlib.util.spec_from_file_location(
        "prefix_gather_unified_cache_npu_cpu_ut", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Counter:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def wait_until(self, layer):
        self.calls.append(layer)
        self.events.append(("wait", layer))


class SpyPool:
    """Mimics NPUMLATokenToKVPool's unified_kv getter/sync contract."""

    def __init__(
        self,
        view,
        *,
        start_layer=2,
        unified=True,
        dtype=torch.bfloat16,
        store_dtype=None,
        kv_lora_rank=_KV_LATENT,
        qk_rope_head_dim=_KV_ROPE,
        page_size=_PAGE,
        use_dsa=False,
        expanded=False,
        with_getter=True,
    ):
        self.kv_buffer = [view]
        self.unified_kv = unified
        self.start_layer = start_layer
        self.dtype = dtype
        self.store_dtype = dtype if store_dtype is None else store_dtype
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.page_size = page_size
        self.use_dsa = use_dsa
        self.prefill_expanded_geometry = object() if expanded else None
        self.layer_transfer_counter = None
        self.events = []
        self.get_calls = []
        if with_getter:
            self.get_kv_buffer = self._get_kv_buffer

    def _get_kv_buffer(self, layer_id):
        self.get_calls.append(layer_id)
        self.events.append(("getter", layer_id))
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        local = layer_id - self.start_layer
        view = self.kv_buffer[local if local < len(self.kv_buffer) else 0]
        return view[..., :_KV_LATENT], view[..., _KV_LATENT:]


def _view(pages=8, dtype=torch.bfloat16, rank=4, seed=0):
    n = pages * _PAGE * _KV_WIDTH
    base = torch.arange(n, dtype=torch.float32)
    base = (base * (seed + 1)) % 512
    if rank == 4:
        return base.to(dtype).reshape(pages, _PAGE, 1, _KV_WIDTH)
    return base.to(dtype).reshape(pages * _PAGE, 1, _KV_WIDTH)


def _activate(monkeypatch, helper, view, seen=None):
    monkeypatch.setattr(helper, "_gate_enabled", lambda: True)
    monkeypatch.setattr(helper, "_graph_or_compile_active", lambda: False)
    monkeypatch.setattr(helper, "_pa_nz_active", lambda: False)

    def fake_view(pool, layer_id):
        if seen is not None:
            seen["view_arg"] = layer_id
            seen["view_calls"] += 1
        return view

    monkeypatch.setattr(helper, "unified_layer_cache_view", fake_view)


def _idx(values, dtype=torch.int64):
    return torch.tensor(values, dtype=dtype)


# ---------------------------------------------------------------------------
# gates: None and no getter side effect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        lambda kw: kw.update(unified=False),
        lambda kw: kw.update(use_dsa=True),
        lambda kw: kw.update(kv_lora_rank=511),
        lambda kw: kw.update(qk_rope_head_dim=63),
        lambda kw: kw.update(page_size=64),
        lambda kw: kw.update(expanded=True),
        lambda kw: kw.update(dtype=torch.bfloat16, store_dtype=torch.uint8),
        lambda kw: kw.update(dtype=torch.float32, store_dtype=torch.float32),
        lambda kw: kw.update(with_getter=False),
    ],
    ids=[
        "not-unified",
        "dsa",
        "latent-not-512",
        "rope-not-64",
        "page-not-128",
        "expanded-kv",
        "quant-store-dtype",
        "unsupported-dtype",
        "missing-getter",
    ],
)
def test_unsupported_pool_falls_back_before_getter(monkeypatch, mutate):
    helper = _load_helper()
    view = _view()
    _activate(monkeypatch, helper, view)
    kwargs = dict()
    mutate(kwargs)
    pool = SpyPool(view, **kwargs)
    assert (
        helper.maybe_gather_unified_prefix_kv(pool, 5, _idx([1, 2])) is None
    )
    assert pool.get_calls == []
    assert pool.events == []


@pytest.mark.parametrize(
    "bad_indices",
    [
        None,
        [[0, 1]],
        _idx([0.0, 1.0], dtype=torch.float32),
        _idx([0, 1], dtype=torch.int16),
    ],
    ids=["not-a-tensor", "rank2", "float32", "int16"],
)
def test_bad_page_index_metadata_falls_back_before_getter(monkeypatch, bad_indices):
    helper = _load_helper()
    view = _view()
    _activate(monkeypatch, helper, view)
    pool = SpyPool(view)
    assert helper.maybe_gather_unified_prefix_kv(pool, 5, bad_indices) is None
    assert pool.get_calls == []


def test_flag_off_falls_back(monkeypatch):
    helper = _load_helper()
    view = _view()
    _activate(monkeypatch, helper, view)
    monkeypatch.setattr(helper, "_gate_enabled", lambda: False)
    pool = SpyPool(view)
    assert helper.maybe_gather_unified_prefix_kv(pool, 5, _idx([1])) is None
    assert pool.get_calls == []


def test_graph_capture_falls_back(monkeypatch):
    helper = _load_helper()
    view = _view()
    _activate(monkeypatch, helper, view)
    monkeypatch.setattr(helper, "_graph_or_compile_active", lambda: True)
    pool = SpyPool(view)
    assert helper.maybe_gather_unified_prefix_kv(pool, 5, _idx([1])) is None
    assert pool.get_calls == []


def test_pa_nz_falls_back(monkeypatch):
    helper = _load_helper()
    view = _view()
    _activate(monkeypatch, helper, view)
    monkeypatch.setattr(helper, "_pa_nz_active", lambda: True)
    pool = SpyPool(view)
    assert helper.maybe_gather_unified_prefix_kv(pool, 5, _idx([1])) is None
    assert pool.get_calls == []


def test_unvalidated_view_falls_back(monkeypatch):
    helper = _load_helper()
    pool = SpyPool(_view())
    _activate(monkeypatch, helper, None)  # validator rejects the layout
    assert helper.maybe_gather_unified_prefix_kv(pool, 5, _idx([1])) is None
    assert pool.get_calls == []


def test_view_dtype_or_device_mismatch_falls_back(monkeypatch):
    helper = _load_helper()
    view = _view()
    _activate(monkeypatch, helper, view)
    pool = SpyPool(view, dtype=torch.float16, store_dtype=torch.float16)
    assert helper.maybe_gather_unified_prefix_kv(pool, 5, _idx([1])) is None
    assert pool.get_calls == []


# ---------------------------------------------------------------------------
# ordering: wait before raw gather
# ---------------------------------------------------------------------------


def test_getter_wait_before_gather_nonzero_start_layer(monkeypatch):
    helper = _load_helper()
    view = _view()
    seen = {"view_calls": 0}
    _activate(monkeypatch, helper, view, seen)
    pool = SpyPool(view, start_layer=2)
    pool.layer_transfer_counter = Counter(pool.events)

    real_gather = helper._invoke_gather

    def spy_gather(layer_pages, page_indices):
        pool.events.append(("gather", None))
        return real_gather(layer_pages, page_indices)

    monkeypatch.setattr(helper, "_invoke_gather", spy_gather)

    out = helper.maybe_gather_unified_prefix_kv(pool, 5, _idx([1, 4]))
    assert out is not None
    # global id to the getter/view, local id (5-2=3) to the transfer wait
    assert seen["view_arg"] == 5 and seen["view_calls"] == 1
    assert pool.get_calls == [5]
    assert pool.layer_transfer_counter.calls == [3]
    assert pool.events == [("getter", 5), ("wait", 3), ("gather", None)]


# ---------------------------------------------------------------------------
# values and shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rank", [4, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "pages",
    [
        [3, 0, 7, 3, 1],  # unsorted + duplicate + first/last
        [0],              # dummy page 0
        [7],              # last page
        [],               # empty selection
    ],
    ids=["mixed", "page0", "last", "empty"],
)
@pytest.mark.parametrize("idx_dtype", [torch.int64, torch.int32])
def test_selected_values_and_shapes(monkeypatch, rank, dtype, pages, idx_dtype):
    helper = _load_helper()
    view = _view(pages=8, dtype=dtype, rank=rank)
    _activate(monkeypatch, helper, view)
    pool = SpyPool(view, dtype=dtype, store_dtype=dtype)
    before = view.clone()

    page_idx = _idx(pages, dtype=idx_dtype)
    out = helper.maybe_gather_unified_prefix_kv(pool, 5, page_idx)
    assert out is not None
    kv_cached, k_rope_cached = out
    s = len(pages)

    assert tuple(kv_cached.shape) == (s, _PAGE, 1, _KV_LATENT)
    assert tuple(k_rope_cached.shape) == (s * _PAGE, 1, _KV_ROPE)
    assert kv_cached.dtype == dtype and k_rope_cached.dtype == dtype
    assert kv_cached.device.type == "cpu" and k_rope_cached.device.type == "cpu"

    # Independent per-page golden: index the packed source page by page.
    pages4 = view.reshape(8, _PAGE, 1, _KV_WIDTH)
    if pages:
        exp_kv = torch.stack([pages4[p] for p in pages], 0)[..., :_KV_LATENT]
        exp_rope = (
            torch.stack([pages4[p] for p in pages], 0)[..., _KV_LATENT:]
            .reshape(s * _PAGE, 1, _KV_ROPE)
        )
    else:
        exp_kv = torch.empty(0, _PAGE, 1, _KV_LATENT, dtype=dtype)
        exp_rope = torch.empty(0, 1, _KV_ROPE, dtype=dtype)
    assert torch.equal(kv_cached, exp_kv)
    assert torch.equal(k_rope_cached, exp_rope)
    assert torch.equal(view, before), "source packed cache must not be mutated"


def test_rank3_and_rank4_agree(monkeypatch):
    helper = _load_helper()
    v4 = _view(pages=8, rank=4)
    v3 = v4.reshape(8 * _PAGE, 1, _KV_WIDTH)
    _activate(monkeypatch, helper, v4)
    out4 = helper.maybe_gather_unified_prefix_kv(
        SpyPool(v4), 5, _idx([6, 2, 2, 0])
    )
    _activate(monkeypatch, helper, v3)
    out3 = helper.maybe_gather_unified_prefix_kv(
        SpyPool(v3), 5, _idx([6, 2, 2, 0])
    )
    assert out3 is not None and out4 is not None
    assert torch.equal(out3[0], out4[0])
    assert torch.equal(out3[1], out4[1])


def test_layer_ids_are_passed_to_local_view(monkeypatch):
    helper = _load_helper()
    views = [_view(pages=4, seed=s) for s in range(3)]
    seen = {"view_calls": 0}

    def fake_view(pool, layer_id):
        seen["view_arg"] = layer_id
        seen["view_calls"] += 1
        return views[layer_id - pool.start_layer]

    monkeypatch.setattr(helper, "_gate_enabled", lambda: True)
    monkeypatch.setattr(helper, "_graph_or_compile_active", lambda: False)
    monkeypatch.setattr(helper, "_pa_nz_active", lambda: False)
    monkeypatch.setattr(helper, "unified_layer_cache_view", fake_view)

    pool = SpyPool(views[0], start_layer=10)
    pool.kv_buffer = views
    for global_layer in (10, 11, 12):
        out = helper.maybe_gather_unified_prefix_kv(
            pool, global_layer, _idx([1, 3])
        )
        assert out is not None
        local = global_layer - 10
        exp = torch.stack([views[local][p] for p in (1, 3)], 0)[..., :_KV_LATENT]
        assert torch.equal(out[0], exp)
        assert seen["view_arg"] == global_layer


def test_both_prefix_callsites_use_helper_and_keep_fallback():
    """Static check: the two old index_select blocks survive as the fallback."""
    repo = Path(os.environ.get("PREFIX_GATHER_REPO", "")).resolve()
    src_path = (
        repo
        / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
    )
    assert src_path.is_file(), src_path
    src = src_path.read_text()
    assert src.count("maybe_gather_unified_prefix_kv(") == 2
    assert src.count("if _gathered is None:") == 2
    assert src.count("else:\n                    kv_cached, k_rope_cached = _gathered") == 2
    # The original two-field route must still be present for every unsupported
    # mode; never delete it while adding the opt-in path.
    assert src.count("k_buffer = self.token_to_kv_pool.get_key_buffer(layer.layer_id)") == 2
    assert src.count("v_buffer = self.token_to_kv_pool.get_value_buffer(layer.layer_id)") == 2
