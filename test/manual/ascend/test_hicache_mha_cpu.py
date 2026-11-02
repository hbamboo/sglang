"""CPU regression coverage for Ascend MHA HiCache transfer dispatch.

Run with TORCH_DEVICE_BACKEND_AUTOLOAD=0. Real torch tensors and the real host
pool classes are used; imports for optional accelerator packages are isolated.
The native copy is modeled on CPU, so this is not hardware DMA qualification.
"""

import abc
import ast
import logging
import threading
import unittest
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Sequence

import torch

ROOT = Path(__file__).resolve().parents[3]


def _load_pool_classes():
    def alloc(dims, *, dtype, device, **kwargs):
        return torch.empty(dims, dtype=dtype, device=device)

    namespace = dict(
        abc=abc,
        logging=logging,
        threading=threading,
        wraps=wraps,
        Optional=Optional,
        Sequence=Sequence,
        torch=torch,
        is_cuda=lambda: False,
        is_hip=lambda: False,
        is_npu=lambda: False,
        is_xpu=lambda: False,
        is_mps=lambda: False,
        psutil=SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(available=1 << 40)
        ),
        get_allocator_from_storage=lambda _: None,
        ALLOC_MEMORY_FUNCS={"cpu": alloc},
        can_use_hicache_jit_kernel=lambda **_: False,
        can_use_write_back_jit_kernel=lambda **_: False,
    )
    for name in ("base", "mha"):
        path = ROOT / f"python/sglang/srt/mem_cache/pool_host/{name}.py"
        tree = ast.parse(path.read_text(), filename=str(path))
        # Keep every production class/function unchanged. Only external imports
        # are supplied above, avoiding optional sgl_kernel/CUDA/NPU imports.
        tree.body = [
            node
            for node in tree.body
            if not isinstance(node, (ast.Import, ast.ImportFrom))
            or (isinstance(node, ast.ImportFrom) and node.module == "__future__")
        ]
        exec(compile(tree, str(path), "exec"), namespace)
    namespace["TransferDirection"] = SimpleNamespace(H2D="H2D", D2H="D2H")
    return namespace


POOLS = _load_pool_classes()


def device_pool(*, layer_num=3, v_dim=4, fia=False, device_pages=4):
    shape = (layer_num, device_pages, 2, 2)
    k = torch.arange(layer_num * device_pages * 2 * 2 * 4).reshape(*shape, 4)
    v = 1000 + torch.arange(layer_num * device_pages * 2 * 2 * v_dim).reshape(
        *shape, v_dim
    )
    k, v = k.to(torch.bfloat16), v.to(torch.bfloat16)
    fields = dict(
        layer_num=layer_num,
        size=(device_pages - 1) * 2,
        page_size=2,
        store_dtype=k.dtype,
        head_num=2,
        head_dim=4,
        v_head_dim=v_dim,
        start_layer=0,
        end_layer=layer_num,
        device="cpu",
        layer_shard_enabled=False,
        k_buffer=[k[i].view(-1, 1, 2, 4) for i in range(layer_num)] if fia else k,
        v_buffer=[v[i].view(-1, 1, 2, v_dim) for i in range(layer_num)] if fia else v,
    )
    return SimpleNamespace(**fields), k, v


def host_pool(pool, *, layout="page_first_direct", ratio=2.0):
    cls = POOLS["get_mha_host_pool_cls"](pool)
    return cls(pool, ratio, 0, pool.page_size, layout, pin_memory=False)


class TestAscendMhaHiCacheCPU(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def native_copy(**kwargs):
            self.calls.append(kwargs)
            ps = kwargs["page_size"]
            hi, di = kwargs["host_indices"], kwargs["device_indices"]
            for device_key, host_key in (
                ("device_k", "host_k"), ("device_v", "host_v")
            ):
                device, host = kwargs[device_key], kwargs[host_key]
                self.assertIsInstance(device, torch.Tensor)
                self.assertEqual(device.ndim, 5)
                self.assertTrue(device.is_contiguous())
                for offset in range(0, len(di), ps):
                    dp, hp = int(di[offset]) // ps, int(hi[offset]) // ps
                    if kwargs["direction"] == "D2H":
                        host[hp].copy_(device[:, dp])
                    else:
                        device[:, dp].copy_(host[hp])

        POOLS["transfer_kv_dim_exchange"] = native_copy

    def test_fia_list_is_zero_copy_and_preserves_every_layer(self):
        pool, k, _ = device_pool(fia=True)
        actual = POOLS["_ascend_paged_buffer"](pool.k_buffer, 2, 3)
        self.assertEqual(actual.data_ptr(), k.data_ptr())
        torch.testing.assert_close(actual, k, rtol=0, atol=0)
        actual[2, 3, 1, 0, 2] = -19
        self.assertEqual(pool.k_buffer[2][-1, 0, 0, 2], -19)
        self.assertEqual(k[2, 3, 1, 0, 2], -19)

    def test_fia_tuple_and_dense_tensor_are_accepted(self):
        pool, k, _ = device_pool(fia=True)
        actual = POOLS["_ascend_paged_buffer"](tuple(pool.k_buffer), 2, 3)
        self.assertEqual(actual.data_ptr(), k.data_ptr())
        self.assertIs(POOLS["_ascend_paged_buffer"](k, 2, 3), k)

    def test_separate_or_reordered_fia_layers_are_rejected(self):
        pool, _, _ = device_pool(fia=True)
        for layers in (
            [layer.clone() for layer in pool.k_buffer],
            list(reversed(pool.k_buffer)),
            pool.k_buffer[:2],
            [pool.k_buffer[0]] * 3,
        ):
            with self.subTest(layers=len(layers)), self.assertRaises(ValueError):
                POOLS["_ascend_paged_buffer"](layers, 2, 3)

    def test_noncontiguous_and_wrong_page_geometry_are_rejected(self):
        pool, k, _ = device_pool(fia=True)
        for buffer, ps, layers in (
            (k[..., ::2], 2, 3),
            (k, 3, 3),
            ([layer.transpose(2, 3) for layer in pool.k_buffer], 2, 3),
            ([torch.empty(7, 1, 2, 4)] * 3, 2, 3),
        ):
            with self.subTest(page_size=ps), self.assertRaises(ValueError):
                POOLS["_ascend_paged_buffer"](buffer, ps, layers)

    def test_host_layer_views_work_when_fewer_pages_than_layers(self):
        pool, _, _ = device_pool(layer_num=4)
        host = host_pool(pool, ratio=0.0)
        self.assertEqual(host.page_num, 1)
        for layer, view in enumerate(host.k_data_refs):
            torch.testing.assert_close(view, host.k_buffer[:, layer])
            self.assertEqual(view.data_ptr(), host.k_buffer[0, layer].data_ptr())

    def test_roundtrip_dense_fia_and_asymmetric_preserves_unselected_pages(self):
        for fia in (False, True):
            for v_dim in (4, 6):
                with self.subTest(fia=fia, v_dim=v_dim):
                    self.calls.clear()
                    pool, k, v = device_pool(fia=fia, v_dim=v_dim)
                    original_k, original_v = k.clone(), v.clone()
                    host = host_pool(pool)
                    host.k_buffer.fill_(-3)
                    host.v_buffer.fill_(-5)
                    hi = torch.tensor([4, 5, 0, 1])
                    di = torch.tensor([2, 3, 6, 7])
                    host.backup_from_device_all_layer(pool, hi, di, "kernel_ascend")
                    self.assertEqual(len(self.calls), 1)
                    self.assertEqual(self.calls[0]["device_k"].data_ptr(), k.data_ptr())
                    for hp, dp in ((2, 1), (0, 3)):
                        torch.testing.assert_close(host.k_buffer[hp], original_k[:, dp])
                        torch.testing.assert_close(host.v_buffer[hp], original_v[:, dp])
                    self.assertTrue(torch.all(host.k_buffer[1] == -3))
                    self.assertTrue(torch.all(host.v_buffer[1] == -5))
                    k.fill_(-7)
                    v.fill_(-9)
                    for layer in range(pool.layer_num):
                        host.load_to_device_per_layer(
                            pool, hi, di, layer, "kernel_ascend"
                        )
                    self.assertEqual(len(self.calls), 2)
                    for dp in (1, 3):
                        torch.testing.assert_close(k[:, dp], original_k[:, dp])
                        torch.testing.assert_close(v[:, dp], original_v[:, dp])
                    self.assertTrue(torch.all(k[:, 0] == -7))
                    self.assertTrue(torch.all(v[:, 2] == -9))

    def test_unsupported_layout_and_sidecar_fail_before_native_submission(self):
        pool, _, _ = device_pool()
        host = host_pool(pool, layout="page_first")
        indices = torch.arange(2)
        with self.assertRaisesRegex(ValueError, "page_first_direct"):
            host.backup_from_device_all_layer(pool, indices, indices, "kernel_ascend")
        host = host_pool(pool)
        with self.assertRaisesRegex(NotImplementedError, "sidecar"):
            host.load_to_device_per_layer(
                pool, indices, indices, 0, "kernel_ascend", is_draft=True
            )
        host.mtp_draft_device_pools = (pool,)
        with self.assertRaisesRegex(NotImplementedError, "MTP"):
            host.load_to_device_per_layer(pool, indices, indices, 0, "kernel_ascend")
        host.mtp_draft_device_pools = ()
        pool.layer_shard_enabled = True
        with self.assertRaisesRegex(NotImplementedError, "layer-sharded"):
            host.load_to_device_per_layer(pool, indices, indices, 0, "kernel_ascend")
        self.assertFalse(self.calls)

    def test_gpu_direct_dispatch_still_uses_independent_asymmetric_buffers(self):
        pool, _, _ = device_pool(v_dim=6)
        host = host_pool(pool)
        calls = []
        POOLS["transfer_kv_per_layer_direct_pf_lf"] = lambda **kw: calls.append(kw)
        host.load_to_device_per_layer(
            pool, torch.arange(2), torch.arange(2), 1, "direct"
        )
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0]["src_ptrs"][0], host.k_buffer)
        self.assertIs(calls[1]["src_ptrs"][0], host.v_buffer)
        self.assertEqual(calls[0]["dst_ptrs"][0].shape[-1], 4)
        self.assertEqual(calls[1]["dst_ptrs"][0].shape[-1], 6)
        self.assertFalse(self.calls)


if __name__ == "__main__":
    unittest.main()
