"""Exact-copy HiCache round trips for compact and expanded NPU MLA KV.

Run on one free NPU, e.g. ASCEND_RT_VISIBLE_DEVICES=7. No model/projection
is used: expanded rows are copied byte-for-byte, including each independently
allocated layer. This checks real DMA and completion events, not throughput.
"""

import gc
import json
import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=30, suite="nightly-1-npu-a3", nightly=True)


class TestNPUHiCacheExpandedTransfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise unittest.SkipTest("Ascend NPU required")
        torch.npu.set_device(0)

    def assertSameBits(self, actual, expected, label):
        self.assertEqual(actual.dtype, expected.dtype, label)
        self.assertEqual(actual.shape, expected.shape, label)
        self.assertTrue(
            torch.equal(
                actual.contiguous().view(torch.int16),
                expected.contiguous().view(torch.int16),
            ),
            label,
        )

    def _run_round_trip(self, dtype, unified):
        from sglang.srt.hardware_backend.npu.attention.prefill_expanded_kv import (
            ExpandedKVGeometry,
        )
        from sglang.srt.hardware_backend.npu.memory_pool_npu import (
            NPUMLATokenToKVPool,
        )
        from sglang.srt.managers.cache_controller import LayerDoneCounter
        from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
        from sglang.srt.mem_cache.pool_host.mla import get_mla_host_pool_cls

        page_size, layers, heads, dim = 128, 3, 20, 256
        source_pages = [4, 1, 3]
        host_pages = [2, 6, 1]
        target_pages = [0, 2, 4]

        def slots(pages):
            return torch.cat(
                [torch.arange(p * page_size, (p + 1) * page_size) for p in pages]
            )

        with mock.patch(
            "sglang.srt.hardware_backend.npu.memory_pool_npu.use_unified_mla_kv_layout",
            return_value=unified,
        ):
            device = NPUMLATokenToKVPool(
                size=4 * page_size,
                page_size=page_size,
                dtype=dtype,
                kv_lora_rank=512,
                qk_rope_head_dim=64,
                index_head_dim=128,
                layer_num=layers,
                device="npu",
                enable_memory_saver=False,
                prefill_expanded_geometry=ExpandedKVGeometry(heads, dim, dim),
                start_layer=5,
                end_layer=7,
            )
        # The production expanded pools are independent layer allocations.
        for name in ("prefill_k_buffer", "prefill_v_buffer"):
            buffers = getattr(device, name)
            self.assertIsInstance(buffers, list)
            self.assertEqual(len(buffers), layers)
            self.assertEqual(len({x.untyped_storage().data_ptr() for x in buffers}), layers)
        host = get_mla_host_pool_cls(device)(
            device,
            host_to_device_ratio=2,
            host_size=0,
            page_size=page_size,
            layout="page_first_kv_split",
        )
        try:
            self.assertTrue(device._hicache_expanded_host_mirror)
            names = ["kv_buffer"] if unified else ["k_buffer", "v_buffer"]
            names += ["index_k_buffer", "prefill_k_buffer", "prefill_v_buffer"]
            originals = {}
            for component, name in enumerate(names):
                originals[name] = []
                for local, buffer in enumerate(getattr(device, name)):
                    values = (
                        torch.arange(buffer.numel(), dtype=torch.int64)
                        .remainder(509)
                        .sub(254)
                        .div(64)
                        .add(component * 8 + local)
                        .to(dtype)
                        .reshape(buffer.shape)
                    )
                    buffer.copy_(values)
                    originals[name].append(values)
            host.kv_buffer.fill_(-37)
            engine = L2TransferEngine("kernel_ascend")
            complete = engine.submit_device_to_host(
                [L2Transfer(host, device, slots(host_pages), slots(source_pages))]
            )
            complete.finish_event.synchronize()
            self.assertTrue(complete.finish_event.query())
            for local in range(layers):
                latent = originals["kv_buffer"][local] if unified else None
                original_key = latent[..., :512] if unified else originals["k_buffer"][local]
                original_rope = latent[..., 512:] if unified else originals["v_buffer"][local]
                for source, target in zip(source_pages, host_pages):
                    for label, host_view, original in (
                        ("latent", host.k_buffer, original_key),
                        ("rope", host.v_buffer, original_rope),
                        ("index", host.index_k_buffer, originals["index_k_buffer"][local]),
                        ("expanded-key", host._expanded_host_k, originals["prefill_k_buffer"][local]),
                        ("expanded-value", host._expanded_host_v, originals["prefill_v_buffer"][local]),
                    ):
                        self.assertSameBits(
                            host_view[target, local], original[source],
                            f"D2H {label} layer={local} source={source}",
                        )
            for page in set(range(host.page_num)) - set(host_pages):
                self.assertTrue((host.get_data_page(page * page_size) == -37).all().item())

            # A serialized page must include compact and expanded sections.
            host_before = host.kv_buffer.clone()
            page_start = host_pages[0] * page_size
            serialized = host.get_data_page(page_start).clone()
            host.get_data_page(page_start).fill_(-29)
            host.set_from_flat_data_page(page_start, serialized)
            self.assertSameBits(host.kv_buffer, host_before, "flat storage page restore")

            for name in names:
                for buffer in getattr(device, name):
                    buffer.fill_(-91)
            counter = LayerDoneCounter(layers)
            producer = counter.update_producer()
            device.register_layer_transfer_counter(counter)
            complete = engine.submit_host_to_device(
                [L2Transfer(host, device, slots(host_pages), slots(target_pages))],
                layer_num=layers,
                on_layer_done=counter.events[producer].complete,
            )
            counter.set_consumer(producer)

            def expected_pages(original):
                expected = torch.full_like(original, -91)
                for source, target in zip(source_pages, target_pages):
                    expected[target] = original[source]
                return expected

            # Read expanded buffers first, before any compact getter or the
            # overall completion wait can mask a missing expanded-layer fence.
            for local in range(layers):
                global_layer = device.start_layer + local
                expanded_key, expanded_value = device.get_prefill_expanded_kv(global_layer)
                expanded_key, expanded_value = expanded_key.cpu(), expanded_value.cpu()
                key = device.get_key_buffer(global_layer).cpu()
                rope = device.get_value_buffer(global_layer).cpu()
                index = device.get_index_k_buffer(global_layer).cpu()
                latent = originals["kv_buffer"][local] if unified else None
                original_key = latent[..., :512] if unified else originals["k_buffer"][local]
                original_rope = latent[..., 512:] if unified else originals["v_buffer"][local]
                for label, actual, original in (
                    ("latent", key, original_key),
                    ("rope", rope, original_rope),
                    ("index", index, originals["index_k_buffer"][local]),
                    ("expanded-key", expanded_key, originals["prefill_k_buffer"][local]),
                    ("expanded-value", expanded_value, originals["prefill_v_buffer"][local]),
                ):
                    self.assertSameBits(
                        actual, expected_pages(original), f"H2D {label} layer={local}"
                    )
            complete.finish_event.synchronize()
            for event in counter.events[producer].load_events:
                self.assertTrue(event.query())
        finally:
            torch.npu.synchronize()
            host.destroy()

    def test_compact_and_expanded_exact_round_trips(self):
        for dtype in (torch.bfloat16, torch.float16):
            for unified in (False, True):
                with self.subTest(dtype=dtype, unified=unified):
                    try:
                        self._run_round_trip(dtype, unified)
                        print(
                            "CASE_PASS "
                            + json.dumps({"dtype": str(dtype), "unified": unified}),
                            flush=True,
                        )
                    finally:
                        torch.npu.synchronize()
                        gc.collect()
                        torch.npu.empty_cache()


if __name__ == "__main__":
    unittest.main(verbosity=2)
