"""Ascend MHA HiCache byte-exact DMA round trips and layer completion events.

Run on one free NPU, e.g. ASCEND_RT_VISIBLE_DEVICES=7. Exercises real NPU
pools and L2TransferEngine; no model or customer prompts are required.
"""

import gc
import json
import os
import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=30, suite="nightly-1-npu-a3", nightly=True)


class TestNPUHiCacheMHATransfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise unittest.SkipTest("Ascend NPU required")
        torch.npu.set_device(0)

    def _run_round_trip(self, dtype, fia, v_dim):
        from sglang.srt.hardware_backend.npu.memory_pool_npu import (
            NPUMHATokenToKVPool,
        )
        from sglang.srt.managers.cache_controller import LayerDoneCounter
        from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
        from sglang.srt.mem_cache.pool_host.mha import (
            AsymmetricMHATokenToKVPoolHost,
            MHATokenToKVPoolHost,
            get_mha_host_pool_cls,
        )

        page_size, layers, heads, k_dim = 128, 3, 2, 128
        source_pages, host_pages, target_pages = [4, 1, 3], [2, 6, 1], [0, 2, 4]

        def slots(pages):
            return torch.cat(
                [torch.arange(p * page_size, (p + 1) * page_size) for p in pages]
            )

        with mock.patch.dict(os.environ, {"ASCEND_USE_FIA": str(fia).lower()}):
            device = NPUMHATokenToKVPool(
                size=4 * page_size,
                page_size=page_size,
                dtype=dtype,
                head_num=heads,
                head_dim=k_dim,
                v_head_dim=v_dim,
                layer_num=layers,
                device="npu",
                enable_memory_saver=False,
                start_layer=5,
                end_layer=7,
            )
        self.assertEqual(device.use_fia, fia)
        self.assertEqual(isinstance(device.k_buffer, list), fia)
        expected_cls = (
            AsymmetricMHATokenToKVPoolHost
            if k_dim != v_dim
            else MHATokenToKVPoolHost
        )
        self.assertIs(get_mha_host_pool_cls(device), expected_cls)
        host = get_mha_host_pool_cls(device)(
            device,
            host_to_device_ratio=2,
            host_size=0,
            page_size=page_size,
            layout="page_first_direct",
        )
        try:
            originals = []
            for component, (name, dim) in enumerate(
                (("k_buffer", k_dim), ("v_buffer", v_dim))
            ):
                shape = (layers, 5, page_size, heads, dim)
                values = (
                    torch.arange(layers * 5 * page_size * heads * dim)
                    .remainder(509)
                    .sub(254)
                    .div(64)
                    .add(component * 8)
                    .to(dtype)
                    .reshape(shape)
                )
                originals.append(values)
                buffers = getattr(device, name)
                for local in range(layers):
                    buffers[local].copy_(values[local].reshape(buffers[local].shape))
            host.k_buffer.fill_(-37)
            host.v_buffer.fill_(-41)
            engine = L2TransferEngine("kernel_ascend")
            completion = engine.submit_device_to_host(
                [L2Transfer(host, device, slots(host_pages), slots(source_pages))]
            )
            completion.finish_event.synchronize()
            self.assertTrue(completion.finish_event.query())
            for original, view, sentinel in zip(
                originals, (host.k_buffer, host.v_buffer), (-37, -41)
            ):
                for dp, hp in zip(source_pages, host_pages):
                    self.assertTrue(torch.equal(original[:, dp], view[hp]))
                for hp in set(range(host.page_num)) - set(host_pages):
                    self.assertTrue((view[hp] == sentinel).all().item())

            # Loading into a different page map catches accidental identity copies.
            for name in ("k_buffer", "v_buffer"):
                for local in range(layers):
                    getattr(device, name)[local].fill_(-91)
            counter = LayerDoneCounter(layers)
            producer = counter.update_producer()
            device.register_layer_transfer_counter(counter)
            completion = engine.submit_host_to_device(
                [L2Transfer(host, device, slots(host_pages), slots(target_pages))],
                layer_num=layers,
                on_layer_done=counter.events[producer].complete,
            )
            counter.set_consumer(producer)
            # Pool getters must fence each consumer layer before the final
            # completion synchronization, including FIA's flattened views.
            observed = []
            for local in range(layers):
                key = device.get_key_buffer(device.start_layer + local).cpu()
                value = device.get_value_buffer(device.start_layer + local).cpu()
                observed.append(
                    (
                        key.reshape(5, page_size, heads, k_dim),
                        value.reshape(5, page_size, heads, v_dim),
                    )
                )
            completion.finish_event.synchronize()
            for component, original in enumerate(originals):
                expected = torch.full_like(original, -91)
                for source, target in zip(source_pages, target_pages):
                    expected[:, target] = original[:, source]
                for local in range(layers):
                    self.assertTrue(
                        torch.equal(observed[local][component], expected[local]),
                        f"component={component} layer={local}",
                    )
            for event in counter.events[producer].load_events:
                self.assertTrue(event.query())
        finally:
            torch.npu.synchronize()
            host.destroy()

    def test_mha_round_trips_and_layer_events(self):
        for dtype in (torch.bfloat16, torch.float16):
            for fia in (False, True):
                for v_dim in (128, 64):
                    with self.subTest(dtype=dtype, fia=fia, v_dim=v_dim):
                        try:
                            self._run_round_trip(dtype, fia, v_dim)
                            print(
                                "CASE_PASS "
                                + json.dumps(
                                    {"dtype": str(dtype), "fia": fia, "v_dim": v_dim}
                                ),
                                flush=True,
                            )
                        finally:
                            torch.npu.synchronize()
                            gc.collect()
                            torch.npu.empty_cache()


if __name__ == "__main__":
    unittest.main(verbosity=2)
