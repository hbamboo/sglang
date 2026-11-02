"""Ascend HiCache byte-exact page round trips, including transfer-stream events.

Run with one free NPU, for example ASCEND_RT_VISIBLE_DEVICES=7. No model or
customer prompts are needed. This checks correctness, not serving throughput.
"""

import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=30, suite="nightly-1-npu-a3", nightly=True)


class TestNPUHiCacheTransfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not hasattr(torch, "npu") or not torch.npu.is_available():
            raise unittest.SkipTest("Ascend NPU required")
        torch.npu.set_device(0)

    def test_mla_round_trip_and_layer_events(self):
        from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
        from sglang.srt.managers.cache_controller import LayerDoneCounter
        from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
        from sglang.srt.mem_cache.pool_host.mla import get_mla_host_pool_cls
        from sglang.srt.mem_cache.pool_host.npu import NPUMLATokenToKVPoolHost

        page_size = 128
        source_pages = [4, 1, 3]
        target_pages = [0, 2, 4]
        host_pages = [2, 6, 1]

        def slots(pages):
            return torch.cat(
                [torch.arange(p * page_size, (p + 1) * page_size) for p in pages]
            )

        for dtype in (torch.bfloat16, torch.float16):
            for unified in (False, True):
                for index_dim in (None, 128):
                    with self.subTest(dtype=dtype, unified=unified, index_dim=index_dim):
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
                                layer_num=3,
                                device="npu",
                                enable_memory_saver=False,
                                index_head_dim=index_dim,
                                start_layer=5,
                                end_layer=7,
                            )
                        self.assertIs(get_mla_host_pool_cls(device), NPUMLATokenToKVPoolHost)
                        host = get_mla_host_pool_cls(device)(
                            device,
                            host_to_device_ratio=2,
                            host_size=0,
                            page_size=page_size,
                            layout="page_first_kv_split",
                        )
                        names = ["kv_buffer"] if unified else ["k_buffer", "v_buffer"]
                        if index_dim is not None:
                            names.append("index_k_buffer")
                        originals = []
                        for component, name in enumerate(names):
                            buf = getattr(device, name)
                            values = (
                                torch.arange(buf.numel(), dtype=torch.int64).remainder(251)
                                + component * 7
                            ).to(dtype).reshape(buf.shape)
                            buf.copy_(values)
                            originals.append(values)
                        host.kv_buffer.fill_(-37)
                        engine = L2TransferEngine("kernel_ascend")
                        completion = engine.submit_device_to_host(
                            [L2Transfer(host, device, slots(host_pages), slots(source_pages))]
                        )
                        completion.finish_event.synchronize()
                        host_before = host.kv_buffer.clone()
                        for original, view in zip(originals, host._host_components):
                            for dp, hp in zip(source_pages, host_pages):
                                self.assertTrue(torch.equal(original[:, dp], view[hp]))
                        for hp in set(range(host.page_num)) - set(host_pages):
                            self.assertTrue((host.kv_buffer[hp] == -37).all().item())

                        # Serialize a complete page, overwrite it, and restore
                        # through the storage interface before loading it back.
                        page_start = host_pages[0] * page_size
                        serialized = host.get_data_page(page_start).clone()
                        host.get_data_page(page_start).zero_()
                        host.set_from_flat_data_page(page_start, serialized)
                        self.assertTrue(torch.equal(host.kv_buffer, host_before))

                        for name in names:
                            getattr(device, name).fill_(-91)
                        counter = LayerDoneCounter(device.layer_num)
                        producer = counter.update_producer()
                        device.register_layer_transfer_counter(counter)
                        completion = engine.submit_host_to_device(
                            [L2Transfer(host, device, slots(host_pages), slots(target_pages))],
                            layer_num=device.layer_num,
                            on_layer_done=counter.events[producer].complete,
                        )
                        counter.set_consumer(producer)
                        # Read on the consumer stream before explicitly waiting
                        # for the overall completion event. Pool access must wait
                        # on the corresponding per-layer transfer event.
                        observed = []
                        for local in range(device.layer_num):
                            observed.append(
                                (
                                    device.get_key_buffer(device.start_layer + local).cpu(),
                                    device.get_value_buffer(device.start_layer + local).cpu(),
                                )
                            )
                        completion.finish_event.synchronize()
                        for original, name in zip(originals, names):
                            actual = getattr(device, name).cpu()
                            expected = torch.full_like(original, -91)
                            for sp, tp in zip(source_pages, target_pages):
                                expected[:, tp] = original[:, sp]
                            self.assertTrue(torch.equal(actual, expected), name)
                        for local, (key, rope) in enumerate(observed):
                            expected_key = torch.full_like(key, -91)
                            expected_rope = torch.full_like(rope, -91)
                            for sp, tp in zip(source_pages, target_pages):
                                if unified:
                                    expected_key[tp] = originals[0][local, sp, ..., :512]
                                    expected_rope[tp] = originals[0][local, sp, ..., 512:]
                                else:
                                    expected_key[tp] = originals[0][local, sp]
                                    expected_rope[tp] = originals[1][local, sp]
                            self.assertTrue(torch.equal(key, expected_key))
                            self.assertTrue(torch.equal(rope, expected_rope))
                        self.assertTrue(counter.events[producer].finish_event.query())
                        host.destroy()
                        del host, device, engine
        torch.npu.synchronize()


if __name__ == "__main__":
    unittest.main(verbosity=2)
