"""CPU tests for Ascend MLA HiCache's native copy/layout contract.

No NPU runtime is loaded: the wrapper is replaced by a CPU implementation of
the native aclrtMemcpy2dAsync address formula, using real tensor storage. This
deliberately does not use tensor assignment, which would hide stride errors in
the split views of a unified MLA device cache.
"""

import ctypes
import enum
import sys
import types
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.pool_host import npu as npu_host
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.npu import NPUMLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Direction(enum.Enum):
    H2D = 1
    D2H = 2


def _pages(*page_ids, page_size=4):
    return torch.tensor(
        [page * page_size + token for page in page_ids for token in range(page_size)],
        dtype=torch.int64,
    )


def _device_pool(unified=True, index=True, dtype=torch.float32, layers=3, pages=6):
    page_size, latent, rope, index_width = 4, 8, 4, 3
    shape = (layers, pages, page_size, 1)
    pool = types.SimpleNamespace(
        device="cpu",
        dtype=dtype,
        store_dtype=dtype,
        size=(pages - 1) * page_size,
        page_size=page_size,
        layer_num=layers,
        start_layer=7,
        end_layer=7 + layers - 1,
        kv_lora_rank=latent,
        qk_rope_head_dim=rope,
        index_head_dim=index_width if index else None,
        layer_shard_enabled=False,
        prefill_expanded_geometry=None,
        unified_kv=unified,
        flash_mla=unified,
        kv_buffer=None,
        index_k_buffer=None,
    )
    if unified:
        pool.kv_buffer = torch.empty((*shape, latent + rope), dtype=dtype)
        pool.k_buffer = pool.kv_buffer[..., :latent]
        pool.v_buffer = pool.kv_buffer[..., latent:]
    else:
        pool.k_buffer = torch.empty((*shape, latent), dtype=dtype)
        pool.v_buffer = torch.empty((*shape, rope), dtype=dtype)
    if index:
        pool.index_k_buffer = torch.empty((*shape, index_width), dtype=dtype)
    return pool


def _components(pool):
    buffers = [pool.kv_buffer] if pool.unified_kv else [pool.k_buffer, pool.v_buffer]
    if pool.index_k_buffer is not None:
        buffers.append(pool.index_k_buffer)
    return buffers


def _make_host(pool, **kwargs):
    options = dict(
        host_to_device_ratio=1.4,
        host_size=0,
        page_size=pool.page_size,
        layout="page_first_kv_split",
        pin_memory=False,
        device="cpu",
    )
    options.update(kwargs)
    return NPUMLATokenToKVPoolHost(device_pool=pool, **options)


def _fill_components(pool):
    for component, buffer in enumerate(_components(pool)):
        # Small exactly representable values, varying across every axis, also
        # work in BF16. A uniform tensor would conceal the original stride bug.
        values = torch.arange(buffer.numel(), dtype=torch.int64).reshape(buffer.shape)
        buffer.copy_(((values * 13 + component * 37) % 251).to(buffer.dtype))


def _tensor_bytes(tensor):
    return bytes(tensor.contiguous().view(torch.uint8).reshape(-1).tolist())


class _NativeCopySimulator:
    """Mirror native size-derived pitches, but copy only CPU-owned storage."""

    def __init__(self, testcase):
        self.testcase = testcase
        self.calls = []
        self.module = types.ModuleType("sgl_kernel_npu.kvcacheio")
        self.module.TransferDirection = _Direction
        self.module.transfer_kv_dim_exchange = self.transfer

    @staticmethod
    def _check_range(tensor, pointer, size):
        storage = tensor.untyped_storage()
        if not storage.data_ptr() <= pointer <= pointer + size <= (
            storage.data_ptr() + storage.nbytes()
        ):
            raise AssertionError("native copy would access outside tensor storage")

    def transfer(self, **kwargs):
        self.testcase.assertEqual(
            set(kwargs),
            {
                "device_k",
                "host_k",
                "device_v",
                "host_v",
                "device_indices",
                "host_indices",
                "page_size",
                "direction",
            },
        )
        for key in ("device_v", "host_v"):
            # The current native kernel reads sizes()[4] even if V is empty.
            self.testcase.assertEqual(kwargs[key].ndim, 5)
            self.testcase.assertEqual(kwargs[key].numel(), 0)
        device, host = kwargs["device_k"], kwargs["host_k"]
        page_size = kwargs["page_size"]
        device_indices, host_indices = kwargs["device_indices"], kwargs["host_indices"]
        self.testcase.assertEqual(device.device.type, "cpu")
        self.testcase.assertEqual(host.device.type, "cpu")
        self.testcase.assertEqual(device.ndim, 5)
        self.testcase.assertEqual(host.ndim, 5)
        self.testcase.assertEqual(device.dtype, host.dtype)
        self.testcase.assertEqual(device.shape[0], host.shape[1])
        self.testcase.assertEqual(device.shape[2:], host.shape[2:])
        self.testcase.assertTrue(device.is_contiguous())
        self.testcase.assertEqual(device_indices.dtype, torch.int64)
        self.testcase.assertEqual(host_indices.dtype, torch.int64)
        self.testcase.assertEqual(device_indices.numel(), host_indices.numel())
        self.testcase.assertEqual(device_indices.numel() % page_size, 0)
        width = page_size * device.shape[3] * device.shape[4] * device.element_size()
        device_pitch = device.shape[1] * width
        host_pitch = width
        for token in range(0, device_indices.numel(), page_size):
            device_page = int(device_indices[token]) // page_size
            host_page = int(host_indices[token]) // page_size
            # Selecting each page uses the actual page stride, exactly as the
            # C++ kernel does. Its layer pitches then come from sizes(), not
            # strides(); host component sections may have a wider page stride.
            device_base = device[0, device_page].data_ptr()
            host_base = host[host_page, 0].data_ptr()
            for layer in range(device.shape[0]):
                device_ptr = device_base + layer * device_pitch
                host_ptr = host_base + layer * host_pitch
                self._check_range(device, device_ptr, width)
                self._check_range(host, host_ptr, width)
                if kwargs["direction"] == _Direction.D2H:
                    ctypes.memmove(host_ptr, device_ptr, width)
                else:
                    self.testcase.assertEqual(kwargs["direction"], _Direction.H2D)
                    ctypes.memmove(device_ptr, host_ptr, width)
        self.calls.append(kwargs)

    def patch(self):
        return mock.patch.dict(sys.modules, {self.module.__name__: self.module})


class TestNPUMLAHiCache(unittest.TestCase):
    def test_native_pitched_copy_roundtrip_all_layouts_and_components(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for unified in (True, False):
                for index in (False, True):
                    with self.subTest(dtype=dtype, unified=unified, index=index):
                        pool = _device_pool(unified=unified, index=index, dtype=dtype)
                        host = _make_host(pool)
                        _fill_components(pool)
                        expected = [buffer.clone() for buffer in _components(pool)]
                        host.kv_buffer.fill_(-1)
                        source_pages = (4, 1, 3)
                        host_pages = (2, 6, 1)
                        target_pages = (0, 2, 4)
                        di = _pages(*source_pages)
                        hi = _pages(*host_pages)
                        ti = _pages(*target_pages)
                        simulator = _NativeCopySimulator(self)
                        with simulator.patch():
                            host.backup_from_device_all_layer(
                                pool, hi, di, "kernel_ascend"
                            )
                            component_count = len(_components(pool))
                            self.assertEqual(len(simulator.calls), component_count)
                            self.assertTrue(
                                all(
                                    c["direction"] == _Direction.D2H
                                    for c in simulator.calls
                                )
                            )
                            for host_component, source in zip(
                                host._host_components, expected
                            ):
                                for hp, dp in zip(host_pages, source_pages):
                                    self.assertTrue(
                                        torch.equal(host_component[hp], source[:, dp])
                                    )
                            for page in set(range(host.page_num)) - set(host_pages):
                                self.assertTrue(torch.all(host.kv_buffer[page] == -1))
                            for buffer in _components(pool):
                                buffer.fill_(-3)
                            for layer in range(pool.layer_num):
                                host.load_to_device_per_layer(
                                    pool, hi, ti, layer, "kernel_ascend"
                                )
                            # One all-layer transfer, rather than once per layer.
                            self.assertEqual(len(simulator.calls), 2 * component_count)
                            self.assertTrue(
                                all(
                                    c["direction"] == _Direction.H2D
                                    for c in simulator.calls[component_count:]
                                )
                            )
                        for buffer, source in zip(_components(pool), expected):
                            for dp, sp in zip(target_pages, source_pages):
                                self.assertTrue(
                                    torch.equal(buffer[:, dp], source[:, sp])
                                )
                            for page in set(range(buffer.shape[1])) - set(target_pages):
                                self.assertTrue(torch.all(buffer[:, page] == -3))
                        host.destroy()

    def test_storage_page_roundtrip_and_metadata_cover_every_component(self):
        for unified in (False, True):
            with self.subTest(unified=unified):
                pool = _device_pool(unified=unified, index=True)
                host = _make_host(pool)
                _fill_components(pool)
                simulator = _NativeCopySimulator(self)
                with simulator.patch():
                    host.backup_from_device_all_layer(
                        pool, _pages(5), _pages(3), "kernel_ascend"
                    )
                payload = host.get_data_page(5 * pool.page_size).clone()
                self.assertTrue(payload.is_contiguous())
                self.assertEqual(
                    payload.numel() * payload.element_size(),
                    host.size_per_token * pool.page_size,
                )
                expected_parts = [
                    buffer[:, 3].reshape(-1) for buffer in _components(pool)
                ]
                self.assertTrue(torch.equal(payload, torch.cat(expected_parts)))
                self.assertEqual(
                    host.get_data_page(20, flat=False).shape, (1, payload.numel())
                )
                host.kv_buffer.zero_()
                host.set_from_flat_data_page(8, payload)
                self.assertTrue(torch.equal(host.get_data_page(8), payload))
                ptrs, sizes = host.get_page_buffer_meta(_pages(2, 5))
                self.assertEqual(len(ptrs), 2)
                self.assertEqual(sizes, [payload.nbytes] * 2)
                self.assertEqual(
                    ptrs,
                    [
                        host.get_data_page(8).data_ptr(),
                        host.get_data_page(20).data_ptr(),
                    ],
                )
                self.assertEqual(
                    ctypes.string_at(ptrs[0], sizes[0]), _tensor_bytes(payload)
                )
                registration, lengths, item_lengths = host.get_contiguous_buf_infos()
                self.assertEqual(registration, [host.kv_buffer.data_ptr()])
                self.assertEqual(lengths, [host.kv_buffer.nbytes])
                self.assertEqual(item_lengths, [payload.nbytes])
                self.assertEqual(host.get_hybrid_pool_buffer(), [host.kv_buffer])
                dummy = host.get_dummy_flat_data_page()
                self.assertEqual(dummy.dtype, host.dtype)
                self.assertEqual(dummy.shape, payload.shape)
                self.assertEqual(dummy.count_nonzero().item(), 0)
                host.destroy()

    def test_index_cache_is_included_in_host_budget(self):
        for index in (False, True):
            with self.subTest(index=index):
                pool = _device_pool(index=index)
                host = _make_host(pool, host_size=0.0001)
                expected = (
                    pool.layer_num
                    * (
                        pool.kv_lora_rank
                        + pool.qk_rope_head_dim
                        + (pool.index_head_dim or 0)
                    )
                    * pool.store_dtype.itemsize
                )
                self.assertEqual(host.size_per_token, expected)
                self.assertEqual(host.get_ksize_per_token(), expected)
                self.assertEqual(host.kv_buffer.nbytes, host.size * expected)
                self.assertLessEqual(
                    host.kv_buffer.nbytes, 100000 + pool.page_size * expected
                )
                self.assertGreater(
                    host.kv_buffer.nbytes, 100000 - pool.page_size * expected
                )
                host.destroy()

    def test_empty_transfer_does_not_call_native_wrapper(self):
        pool = _device_pool()
        host = _make_host(pool)
        simulator = _NativeCopySimulator(self)
        empty = torch.empty(0, dtype=torch.int64)
        with simulator.patch():
            host.backup_from_device_all_layer(pool, empty, empty, "kernel_ascend")
            for layer in range(pool.layer_num):
                host.load_to_device_per_layer(
                    pool, empty, empty, layer, "kernel_ascend"
                )
        self.assertEqual(simulator.calls, [])
        self.assertEqual(host.get_page_buffer_meta(empty), ([], []))
        host.destroy()

    def test_invalid_transfer_indices_fail_before_native_copy(self):
        pool = _device_pool()
        host = _make_host(pool)
        good = _pages(1)
        invalid = [
            ("partial_page", torch.tensor([4, 5], dtype=torch.int64)),
            ("unaligned", torch.tensor([5, 6, 7, 8], dtype=torch.int64)),
            ("nonconsecutive", torch.tensor([4, 5, 7, 6], dtype=torch.int64)),
            ("negative", _pages(-1)),
            ("out_of_capacity", _pages(host.page_num + 2)),
            ("wrong_dtype", good.to(torch.int32)),
            ("wrong_rank", good.view(1, -1)),
        ]
        with mock.patch.object(npu_host, "_transfer_kv_pages") as transfer:
            for side in ("host", "device"):
                for label, bad in invalid:
                    with self.subTest(side=side, invalid=label):
                        hi, di = (bad, good) if side == "host" else (good, bad)
                        with self.assertRaises(ValueError):
                            host.backup_from_device_all_layer(
                                pool, hi, di, "kernel_ascend"
                            )
            transfer.assert_not_called()
        host.destroy()

    def test_unsupported_constructor_combinations_fail_explicitly(self):
        invalid = [
            {"layout": "page_first"},
            {"page_size": 8},
            {"dcp_size": 2},
            {"dcp_rank": 1},
            {"mtp_draft_device_pools": (object(),)},
            {"override_kv_cache_dim": 8},
        ]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                _make_host(_device_pool(), **options)
        for name, value in (
            ("layer_shard_enabled", True),
            ("prefill_expanded_geometry", object()),
        ):
            pool = _device_pool()
            setattr(pool, name, value)
            with self.subTest(field=name), self.assertRaises(ValueError):
                _make_host(pool)

    def test_noncontiguous_or_mismatched_device_components_are_rejected(self):
        for component in ("kv_buffer", "index_k_buffer"):
            for mutation in ("strided", "dtype", "shape"):
                pool = _device_pool()
                buffer = getattr(pool, component)
                bad = {
                    "strided": lambda: buffer[..., ::2],
                    "dtype": lambda: buffer.to(torch.float16),
                    "shape": lambda: buffer[0],
                }
                setattr(pool, component, bad[mutation]())
                with (
                    self.subTest(component=component, mutation=mutation),
                    self.assertRaises(ValueError),
                ):
                    _make_host(pool)
        # The precise pre-fix mistake: label the interleaved K/R views as split.
        pool = _device_pool()
        pool.unified_kv = False
        with self.assertRaises(ValueError):
            _make_host(pool)

    def test_invalid_load_and_storage_operations_fail(self):
        pool = _device_pool()
        host = _make_host(pool)
        good = _pages(1)
        for layer in (-1, pool.layer_num):
            with self.subTest(layer=layer), self.assertRaises(ValueError):
                host.load_to_device_per_layer(pool, good, good, layer, "kernel_ascend")
        with self.assertRaises(ValueError):
            host.load_to_device_per_layer(
                pool, good, good, 0, "kernel_ascend", is_draft=True
            )
        with self.assertRaises(ValueError):
            host.backup_from_device_all_layer(pool, good, good, "kernel")
        with self.assertRaises(ValueError):
            host.backup_from_device_all_layer(
                _device_pool(), good, good, "kernel_ascend"
            )
        for index in (-4, 1, host.size):
            with self.subTest(index=index), self.assertRaises(ValueError):
                host.get_data_page(index)
        with self.assertRaises(ValueError):
            host.set_from_flat_data_page(0, torch.zeros(1, dtype=host.dtype))
        with self.assertRaises(ValueError):
            host.set_from_flat_data_page(0, host.get_data_page(0).to(torch.float16))
        host.destroy()

    def test_mla_type_and_small_page_count_do_not_use_cuda_pointer_arrays(self):
        pool = _device_pool(layers=5, pages=2, index=False)
        host = _make_host(pool, host_to_device_ratio=1)
        self.assertLess(host.page_num, host.layer_num)
        self.assertIsInstance(host, MLATokenToKVPoolHost)
        self.assertFalse(host.can_use_jit)
        self.assertFalse(host.can_use_write_back_jit)
        host.destroy()
        host.destroy()
        self.assertIsNone(host.kv_buffer)
        self.assertIsNone(host.k_buffer)
        self.assertIsNone(host.v_buffer)
        self.assertIsNone(host.index_k_buffer)
        self.assertEqual(host._host_components, [])


if __name__ == "__main__":
    unittest.main()
