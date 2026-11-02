"""CPU native-pitch tests for latent + expanded Ascend HiCache host mirrors."""

import contextlib
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache import l2_transfer
from sglang.srt.mem_cache.pool_host import npu as npu_host
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang.test.ci.ci_register import register_cpu_ci
from test_npu_hicache_mla import (
    _NativeCopySimulator,
    _components,
    _device_pool,
    _fill_components,
    _make_host,
    _pages,
    _tensor_bytes,
)
from test_npu_hicache_storage import _config

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _dual_pool(*, unified=True, index=True, dtype=torch.float32, geometry=(2, 3, 5)):
    pool = _device_pool(unified=unified, index=index, dtype=dtype)
    heads, key_dim, value_dim = geometry
    pool.prefill_expanded_geometry = SimpleNamespace(
        heads=heads, key_dim=key_dim, value_dim=value_dim
    )
    # Each layer owns its actual allocation, exactly as the NPU device pool does.
    shape = (pool.k_buffer.shape[1], pool.page_size, heads)
    pool.prefill_k_buffer = [
        torch.empty((*shape, key_dim), dtype=dtype) for _ in range(pool.layer_num)
    ]
    pool.prefill_v_buffer = [
        torch.empty((*shape, value_dim), dtype=dtype) for _ in range(pool.layer_num)
    ]
    return pool


def _all_buffers(pool):
    return _components(pool) + pool.prefill_k_buffer + pool.prefill_v_buffer


def _fill(pool):
    _fill_components(pool)
    for number, buffer in enumerate(pool.prefill_k_buffer + pool.prefill_v_buffer):
        values = torch.arange(buffer.numel(), dtype=torch.int64).reshape(buffer.shape)
        buffer.copy_(((values * 19 + number * 41) % 251).to(buffer.dtype))


class _MirrorCopySimulator(_NativeCopySimulator):
    """Reuse the raw C++ pitch simulator for both operands of a K/V call."""

    def __init__(self, testcase, events=None):
        super().__init__(testcase)
        self.wrapper_calls = []
        self.events = [] if events is None else events

    def transfer(self, **kwargs):
        self.testcase.assertNotIn("device_index_k", kwargs)
        if kwargs["device_v"].numel() == 0:
            super().transfer(**kwargs)
            kind = "latent"
        else:
            self.testcase.assertEqual(kwargs["device_k"].shape[0], 1)
            self.testcase.assertEqual(kwargs["device_v"].shape[0], 1)
            self.testcase.assertEqual(
                kwargs["device_k"].shape[3], kwargs["device_v"].shape[3]
            )
            empty = torch.empty((0, 0, 0, 0, 0), dtype=kwargs["host_k"].dtype)
            for component in ("k", "v"):
                primary = dict(kwargs)
                primary["device_k"] = kwargs[f"device_{component}"]
                primary["host_k"] = kwargs[f"host_{component}"]
                primary["device_v"] = primary["host_v"] = empty
                super().transfer(**primary)
            kind = "expanded"
        self.wrapper_calls.append(kwargs)
        self.events.append(("copy", kind, kwargs["direction"]))


class TestNPUHiCacheDualMirror(unittest.TestCase):
    def _host(self, pool, **kwargs):
        host = _make_host(pool, **kwargs)
        self.addCleanup(host.destroy)
        return host

    def test_native_pitches_copy_independent_expanded_layers_bidirectionally(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for unified in (False, True):
                for index in (False, True):
                    with self.subTest(dtype=dtype, unified=unified, index=index):
                        pool = _dual_pool(unified=unified, index=index, dtype=dtype)
                        host = self._host(pool)
                        self.assertTrue(pool._hicache_expanded_host_mirror)
                        _fill(pool)
                        snapshots = [buffer.clone() for buffer in _all_buffers(pool)]
                        host.kv_buffer.fill_(-1)
                        simulator = _MirrorCopySimulator(self)
                        source_pages = (4, 1, 3)
                        host_pages = (2, 6, 1)
                        target_pages = (0, 2, 4)
                        with simulator.patch():
                            host.backup_from_device_all_layer(
                                pool,
                                _pages(*host_pages),
                                _pages(*source_pages),
                                "kernel_ascend",
                            )
                            native_calls = len(_components(pool)) + pool.layer_num
                            self.assertEqual(len(simulator.wrapper_calls), native_calls)
                            for local_layer, call in enumerate(
                                simulator.wrapper_calls[len(_components(pool)):]
                            ):
                                # A stack would allocate another tensor and fail
                                # this pointer-identity check before being copied.
                                self.assertEqual(
                                    call["device_k"].data_ptr(),
                                    pool.prefill_k_buffer[local_layer].data_ptr(),
                                )
                                self.assertEqual(
                                    call["device_v"].data_ptr(),
                                    pool.prefill_v_buffer[local_layer].data_ptr(),
                                )
                            for hp, dp in zip(host_pages, source_pages):
                                for layer in range(pool.layer_num):
                                    self.assertTrue(
                                        torch.equal(
                                            host._expanded_host_k[hp, layer],
                                            pool.prefill_k_buffer[layer][dp],
                                        )
                                    )
                                    self.assertTrue(
                                        torch.equal(
                                            host._expanded_host_v[hp, layer],
                                            pool.prefill_v_buffer[layer][dp],
                                        )
                                    )
                            for page in set(range(host.page_num)) - set(host_pages):
                                self.assertTrue(torch.all(host.kv_buffer[page] == -1))
                            for buffer in _all_buffers(pool):
                                buffer.fill_(-3)
                            for layer in range(pool.layer_num):
                                host.load_to_device_per_layer(
                                    pool,
                                    _pages(*host_pages),
                                    _pages(*target_pages),
                                    layer,
                                    "kernel_ascend",
                                )
                            self.assertEqual(
                                len(simulator.wrapper_calls), 2 * native_calls
                            )
                        latent_count = len(_components(pool))
                        for number, (buffer, source) in enumerate(
                            zip(_all_buffers(pool), snapshots)
                        ):
                            for target, original in zip(target_pages, source_pages):
                                actual = (
                                    buffer[:, target]
                                    if number < latent_count
                                    else buffer[target]
                                )
                                expected = (
                                    source[:, original]
                                    if number < latent_count
                                    else source[original]
                                )
                                self.assertTrue(torch.equal(actual, expected))
                            page_count = (
                                buffer.shape[1]
                                if number < latent_count
                                else buffer.shape[0]
                            )
                            for page in set(range(page_count)) - set(target_pages):
                                actual = (
                                    buffer[:, page]
                                    if number < latent_count
                                    else buffer[page]
                                )
                                self.assertTrue(torch.all(actual == -3))

    def test_complete_dual_payload_survives_real_file_storage_and_native_reload(self):
        pool = _dual_pool(unified=False, index=True, dtype=torch.bfloat16)
        host = self._host(pool)
        _fill(pool)
        snapshots = [buffer.clone() for buffer in _all_buffers(pool)]
        simulator = _MirrorCopySimulator(self)
        with simulator.patch():
            host.backup_from_device_all_layer(
                pool, _pages(2), _pages(3), "kernel_ascend"
            )
        expected_parts = [buffer[:, 3].reshape(-1) for buffer in _components(pool)]
        expected_parts += [buffer[3].reshape(-1) for buffer in pool.prefill_k_buffer]
        expected_parts += [buffer[3].reshape(-1) for buffer in pool.prefill_v_buffer]
        payload = host.get_data_page(8).clone()
        self.assertTrue(torch.equal(payload, torch.cat(expected_parts)))
        geometry = pool.prefill_expanded_geometry
        expected_size = pool.layer_num * pool.store_dtype.itemsize * (
            pool.kv_lora_rank + pool.qk_rope_head_dim + pool.index_head_dim
            + geometry.heads * (geometry.key_dim + geometry.value_dim)
        )
        self.assertEqual(host.size_per_token, expected_size)
        self.assertEqual(host.kv_buffer.nbytes, host.size * expected_size)
        ptrs, sizes = host.get_page_buffer_meta(_pages(2))
        self.assertEqual(ptrs, [host.get_data_page(8).data_ptr()])
        self.assertEqual(sizes, [payload.nbytes])
        with tempfile.TemporaryDirectory(prefix="sglang-hicache-dual-") as directory:
            with mock.patch.dict(
                os.environ, {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": directory}
            ):
                backend = StorageBackendFactory.create_backend("file", _config(), host)
                self.assertTrue(backend.set("dual-page", payload))
                path = os.path.join(directory, f"dual-page{backend.config_suffix}.bin")
                with open(path, "rb") as stored:
                    self.assertEqual(stored.read(), _tensor_bytes(payload))
                host.kv_buffer.zero_()
                target = host.get_dummy_flat_data_page()
                self.assertIs(backend.get("dual-page", target), target)
                host.set_from_flat_data_page(4, target)
        for buffer in _all_buffers(pool):
            buffer.fill_(-7)
        with simulator.patch():
            host.load_to_device_per_layer(
                pool, _pages(1), _pages(4), 0, "kernel_ascend"
            )
        latent_count = len(_components(pool))
        for number, (buffer, source) in enumerate(zip(_all_buffers(pool), snapshots)):
            actual = buffer[:, 4] if number < latent_count else buffer[4]
            expected = source[:, 3] if number < latent_count else source[3]
            self.assertTrue(torch.equal(actual, expected))

    def test_layer_completion_and_first_enqueue_marker_follow_all_native_copies(self):
        pool = _dual_pool()
        host = self._host(pool)
        _fill(pool)
        events = []
        simulator = _MirrorCopySimulator(self, events)

        class Event:
            def __init__(self, **_):
                pass

            def record(self):
                events.append(("record_event",))

            def wait(self, stream):
                events.append(("wait_event",))

        device_module = SimpleNamespace(
            Stream=object, Event=Event, stream=lambda _: contextlib.nullcontext()
        )
        transfer = l2_transfer.L2Transfer(host, pool, _pages(1), _pages(3))

        def logged(template, direction, *args):
            self.assertIn("dual KV %s first enqueue", template)
            events.append(("mirror_marker", direction))

        with (
            simulator.patch(),
            mock.patch.object(l2_transfer, "device_module", device_module),
            mock.patch.object(npu_host.logger, "info", side_effect=logged),
        ):
            engine = l2_transfer.L2TransferEngine("kernel_ascend")
            engine.submit_device_to_host([transfer])
            copy_positions = [i for i, event in enumerate(events) if event[0] == "copy"]
            marker = events.index(("mirror_marker", "D2H"))
            self.assertLess(max(copy_positions), marker)
            self.assertEqual(events[-1], ("record_event",))
            events.clear()
            engine.submit_host_to_device(
                [transfer],
                layer_num=pool.layer_num,
                on_layer_done=lambda layer: events.append(("layer_done", layer)),
            )
            copy_positions = [i for i, event in enumerate(events) if event[0] == "copy"]
            marker = events.index(("mirror_marker", "H2D"))
            first_done = events.index(("layer_done", 0))
            self.assertLess(max(copy_positions), marker)
            self.assertLess(marker, first_done)
            self.assertEqual(
                [event[1] for event in events if event[0] == "layer_done"],
                list(range(pool.layer_num)),
            )

    def test_partial_native_failure_does_not_report_a_completed_dual_enqueue(self):
        pool = _dual_pool()
        host = self._host(pool)
        simulator = _MirrorCopySimulator(self)
        with (
            simulator.patch(),
            mock.patch.object(npu_host.logger, "info") as logged,
            mock.patch.object(
                npu_host,
                "_transfer_expanded_kv_pages",
                side_effect=RuntimeError("copy failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                host.load_to_device_per_layer(
                    pool, _pages(1), _pages(3), 0, "kernel_ascend"
                )
            logged.assert_not_called()
        self.assertNotIn("H2D", host._expanded_enqueue_logged)

    def test_expanded_geometry_and_presence_isolate_file_namespaces(self):
        latent = self._host(_device_pool())
        dual = self._host(_dual_pool())
        same = self._host(_dual_pool())
        changed = self._host(_dual_pool(geometry=(2, 4, 4)))
        # Equal element counts with different K/V splits must remain distinct.
        self.assertEqual(dual.size_per_token, changed.size_per_token)
        self.assertEqual(dual.storage_format, same.storage_format)
        self.assertNotEqual(dual.storage_format, changed.storage_format)
        self.assertNotEqual(dual.storage_format, latent.storage_format)

    def test_invalid_expanded_layers_fail_before_host_memory_allocation(self):
        def replace(pool, name, value):
            setattr(pool, name, value)

        mutations = [
            lambda p: replace(p, "prefill_k_buffer", p.prefill_k_buffer[:-1]),
            lambda p: replace(p, "prefill_v_buffer", None),
            lambda p: p.prefill_k_buffer.__setitem__(1, p.prefill_k_buffer[1][..., ::2]),
            lambda p: p.prefill_v_buffer.__setitem__(
                2, p.prefill_v_buffer[2].to(torch.float16)
            ),
            lambda p: replace(
                p,
                "prefill_expanded_geometry",
                SimpleNamespace(heads=0, key_dim=3, value_dim=5),
            ),
        ]
        for number, mutate in enumerate(mutations):
            pool = _dual_pool()
            mutate(pool)
            with self.subTest(mutation=number):
                with mock.patch.object(npu_host, "ALLOC_MEMORY_FUNCS") as allocate:
                    with self.assertRaises(ValueError):
                        _make_host(pool)
                    allocate.__getitem__.assert_not_called()

    def test_empty_dual_transfers_do_not_copy_or_emit_enqueue_markers(self):
        pool = _dual_pool()
        host = self._host(pool)
        empty = torch.empty(0, dtype=torch.int64)
        with (
            mock.patch.object(npu_host, "_transfer_kv_pages") as latent,
            mock.patch.object(npu_host, "_transfer_expanded_kv_pages") as expanded,
            mock.patch.object(npu_host.logger, "info") as logged,
        ):
            host.backup_from_device_all_layer(pool, empty, empty, "kernel_ascend")
            host.load_to_device_per_layer(pool, empty, empty, 0, "kernel_ascend")
            latent.assert_not_called()
            expanded.assert_not_called()
            logged.assert_not_called()


if __name__ == "__main__":
    unittest.main()
