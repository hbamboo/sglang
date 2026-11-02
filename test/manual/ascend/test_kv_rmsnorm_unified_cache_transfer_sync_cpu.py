"""CPU-only regression for the helper's layer-transfer ordering.

No NPU device is opened and no NPU tensor is allocated.  The helper module is
loaded directly with importlib and the gates/dispatch are monkeypatched so CPU
tensors only exercise the control flow.

Contract under test (restored from the real pool getters):
  * after all narrow gates pass and before the new op is dispatched, the helper
    calls the public ``pool.get_kv_buffer(layer_id)``;
  * that getter performs ``layer_transfer_counter.wait_until(
    layer_id - start_layer)`` exactly as in ``memory_pool_npu.py``;
  * events are ordered ``getter(layer) -> wait(local) -> invoke``;
  * unsupported/fallback paths never call the getter (the original route owns
    its own wait), and a pool without a counter stays wait-free;
  * a pool without a public getter falls back instead of bypassing the contract.
"""

import importlib.util
import os
from pathlib import Path
import types

import pytest
import torch


def _helper_path() -> Path:
    override = os.environ.get("KVRMSNORM_HELPER_PATH")
    if override:
        return Path(override)
    # test/manual/ascend/<this file> -> framework/
    return (
        Path(__file__).resolve().parents[3]
        / "python/sglang/srt/hardware_backend/npu/modules/kv_rmsnorm_unified_cache_npu.py"
    )


def _load_helper():
    path = _helper_path()
    assert path.is_file(), f"helper not found: {path}"
    spec = importlib.util.spec_from_file_location("kv_rmsnorm_unified_cache_npu_cpu_ut", path)
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
    """Mimics NPUMLATokenToKVPool.get_kv_buffer's public sync contract."""

    def __init__(self, view, start_layer=2, counter=None, unified=True):
        self.kv_buffer = [view]
        self.unified_kv = unified
        self.start_layer = start_layer
        self.layer_transfer_counter = counter
        self.events = []
        self.get_calls = []
        self.k_view = view[..., :512]
        self.v_view = view[..., 512:]

    def get_kv_buffer(self, layer_id):
        self.get_calls.append(layer_id)
        self.events.append(("getter", layer_id))
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self.k_view, self.v_view


def _tensors(T=4, dtype=torch.bfloat16):
    latent = torch.zeros(T, 1, 576, dtype=dtype)
    cos = torch.zeros(T, 1, 1, 64, dtype=dtype)
    sin = torch.zeros(T, 1, 1, 64, dtype=dtype)
    index = torch.zeros(T, dtype=torch.int64)
    weight = torch.ones(512, dtype=dtype)
    view = torch.zeros(2, 128, 1, 576, dtype=dtype)
    return latent, cos, sin, index, weight, view


def _model(weight, layer_id=5, kv_lora=512, rope=64, use_dsa=False):
    return types.SimpleNamespace(
        layer_id=layer_id,
        kv_lora_rank=kv_lora,
        qk_rope_head_dim=rope,
        use_dsa=use_dsa,
        kv_a_layernorm=types.SimpleNamespace(weight=weight, variance_epsilon=1e-5),
    )


def _install(monkeypatch, helper, view, seen):
    monkeypatch.setattr(helper, "_gate_enabled", lambda: True)
    monkeypatch.setattr(helper, "_op_loaded", lambda: True)
    monkeypatch.setattr(helper, "_graph_or_compile_active", lambda: False)

    def fake_view(pool, layer_id):
        seen["view_arg"] = layer_id
        seen["view_calls"] += 1
        return view

    monkeypatch.setattr(helper, "unified_layer_cache_view", fake_view)


def _events(pool, monkeypatch, helper):
    def fake_invoke(*args, **kwargs):
        pool.events.append(("invoke", None))

    monkeypatch.setattr(helper, "_invoke_new_op", fake_invoke)
    return pool.events


def test_getter_wait_before_dispatch_with_nonzero_start_layer(monkeypatch):
    helper = _load_helper()
    latent, cos, sin, index, weight, view = _tensors()
    seen = {"view_calls": 0}
    _install(monkeypatch, helper, view, seen)
    pool = SpyPool(view, start_layer=2)
    pool.layer_transfer_counter = Counter(pool.events)
    events = _events(pool, monkeypatch, helper)

    out = helper.maybe_unified_kv_rmsnorm_rope_cache(
        _model(weight, layer_id=5), latent, cos, sin, index, pool, False
    )

    assert out is not None
    assert tuple(out[0].shape) == (4, 1, 1, 64) and out[0].device.type == "cpu"
    assert tuple(out[1].shape) == (4, 1, 1, 512) and out[1].device.type == "cpu"
    # getter receives the global layer id; its internal wait receives 5 - 2 = 3.
    assert seen["view_arg"] == 5 and seen["view_calls"] == 1
    assert pool.get_calls == [5]
    assert pool.layer_transfer_counter.calls == [3]
    assert events == [("getter", 5), ("wait", 3), ("invoke", None)]
    print(f"[cpu-sync] events={events} (getter(global) -> wait(local) -> invoke)")


def test_no_counter_keeps_normal_path_wait_free(monkeypatch):
    helper = _load_helper()
    latent, cos, sin, index, weight, view = _tensors()
    seen = {"view_calls": 0}
    _install(monkeypatch, helper, view, seen)
    pool = SpyPool(view, start_layer=0, counter=None)
    events = _events(pool, monkeypatch, helper)

    out = helper.maybe_unified_kv_rmsnorm_rope_cache(
        _model(weight, layer_id=3), latent, cos, sin, index, pool, False
    )
    assert out is not None
    assert events == [("getter", 3), ("invoke", None)]
    print(f"[cpu-sync] no counter: events={events}")


def test_no_public_getter_falls_back(monkeypatch):
    helper = _load_helper()
    latent, cos, sin, index, weight, view = _tensors()
    seen = {"view_calls": 0}
    _install(monkeypatch, helper, view, seen)
    pool = types.SimpleNamespace(
        kv_buffer=[view],
        unified_kv=True,
        start_layer=0,
        layer_transfer_counter=Counter([]),
        events=[],
        get_calls=[],
    )
    events = _events(pool, monkeypatch, helper)
    out = helper.maybe_unified_kv_rmsnorm_rope_cache(
        _model(weight, layer_id=1), latent, cos, sin, index, pool, False
    )
    assert out is None
    assert events == []
    assert pool.layer_transfer_counter.calls == []
    print("[cpu-sync] no public getter: fallback, no wait")


@pytest.mark.parametrize(
    "case",
    ["gate_off", "graph", "pa_nz", "dsa", "not_unified", "wrong_geometry", "bad_dtype", "bad_view"],
)
def test_unsupported_paths_do_not_call_getter(monkeypatch, case):
    helper = _load_helper()
    dtype = torch.float32 if case == "bad_dtype" else torch.bfloat16
    latent, cos, sin, index, weight, view = _tensors(dtype=dtype)
    seen = {"view_calls": 0}
    _install(monkeypatch, helper, view, seen)
    counter = Counter([])
    pool = SpyPool(view, start_layer=2, counter=counter)
    events = _events(pool, monkeypatch, helper)

    model = _model(weight, layer_id=7)
    fia_nz = False
    if case == "gate_off":
        monkeypatch.setattr(helper, "_gate_enabled", lambda: False)
    elif case == "graph":
        monkeypatch.setattr(helper, "_graph_or_compile_active", lambda: True)
    elif case == "pa_nz":
        fia_nz = True
    elif case == "dsa":
        model.use_dsa = True
    elif case == "not_unified":
        pool.unified_kv = False
    elif case == "wrong_geometry":
        model.kv_lora_rank, model.qk_rope_head_dim = 256, 32
    elif case == "bad_dtype":
        pass  # fp32 latent is rejected before the view/getter/dispatch
    elif case == "bad_view":
        monkeypatch.setattr(helper, "unified_layer_cache_view", lambda p, l: None)

    out = helper.maybe_unified_kv_rmsnorm_rope_cache(model, latent, cos, sin, index, pool, fia_nz)
    assert out is None
    assert pool.get_calls == []
    assert counter.calls == []
    assert not any(e[0] in ("getter", "wait", "invoke") for e in events)
    print(f"[cpu-sync] {case}: no getter call, no wait, no dispatch; events={events}")
