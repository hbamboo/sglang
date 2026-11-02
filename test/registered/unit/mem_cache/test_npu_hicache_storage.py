"""CPU integration tests for Ascend MLA's file HiCache page format."""

import copy
import os
import tempfile
import unittest
from dataclasses import asdict
from itertools import product
from pathlib import Path
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
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

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config(model_name="test/mla", extra_config=None):
    return HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=0,
        attn_cp_size=1,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=True,
        model_name=model_name,
        extra_config=extra_config
        or {"enable_metadata_cache": False, "max_size": 0, "min_free_space": 0},
    )


class TestNPUHiCacheFileStorage(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="sglang-npu-hicache-test-")
        self.addCleanup(directory.cleanup)
        self.storage_dir = Path(directory.name)
        environment = mock.patch.dict(
            os.environ,
            {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": directory.name},
        )
        environment.start()
        self.addCleanup(environment.stop)

    def _host(
        self,
        *,
        unified=True,
        dtype=torch.float16,
        page_size=4,
        layers=3,
        latent=8,
        rope=4,
        index_width=3,
        start_layer=7,
        pages=6,
    ):
        pool = _device_pool(
            unified=unified,
            index=bool(index_width),
            dtype=dtype,
            layers=layers,
            pages=pages,
        )
        pool.page_size = page_size
        pool.size = (pages - 1) * page_size
        pool.start_layer, pool.end_layer = start_layer, start_layer + layers - 1
        pool.kv_lora_rank, pool.qk_rope_head_dim = latent, rope
        pool.index_head_dim = index_width or None
        shape = (layers, pages, page_size, 1)
        if unified:
            pool.kv_buffer = torch.empty((*shape, latent + rope), dtype=dtype)
            pool.k_buffer = pool.kv_buffer[..., :latent]
            pool.v_buffer = pool.kv_buffer[..., latent:]
        else:
            pool.k_buffer = torch.empty((*shape, latent), dtype=dtype)
            pool.v_buffer = torch.empty((*shape, rope), dtype=dtype)
        pool.index_k_buffer = (
            torch.empty((*shape, index_width), dtype=dtype) if index_width else None
        )
        host = _make_host(pool)
        self.addCleanup(host.destroy)
        return pool, host

    @staticmethod
    def _backend(host, config=None):
        backend = StorageBackendFactory.create_backend(
            "file", config or _config(), host
        )
        backend.register_mem_pool_host(host)
        backend.register_mem_host_pool_v2(host, PoolName.KV)
        return backend

    def test_real_file_roundtrip_preserves_latent_rope_and_index(self):
        for unified in (False, True):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                with self.subTest(unified=unified, dtype=dtype):
                    pool, host = self._host(unified=unified, dtype=dtype)
                    backend = self._backend(host)
                    _fill_components(pool)
                    expected = [buffer.clone() for buffer in _components(pool)]
                    host_indices = _pages(1, 6)
                    simulator = _NativeCopySimulator(self)
                    with simulator.patch():
                        host.backup_from_device_all_layer(
                            pool, host_indices, _pages(3, 1), "kernel_ascend"
                        )
                    keys = ["prefix-a", "prefix-b"]
                    transfer = PoolTransfer(
                        PoolName.KV, host_indices=host_indices, keys=keys
                    )
                    self.assertEqual(
                        backend.batch_set_v2([transfer]), {PoolName.KV: [True, True]}
                    )
                    self.assertEqual(backend.batch_exists(keys), 2)
                    self.assertEqual(
                        backend.batch_exists_v2(keys).kv_hit_pages, 2
                    )
                    payloads = [host.get_data_page(i).clone() for i in (4, 24)]
                    for key, payload in zip(keys, payloads):
                        path = self.storage_dir / f"{key}{backend.config_suffix}.bin"
                        self.assertEqual(path.stat().st_size, payload.nbytes)
                        self.assertEqual(path.read_bytes(), _tensor_bytes(payload))
                    host.kv_buffer.zero_()
                    self.assertEqual(
                        backend.batch_get_v2([transfer]), {PoolName.KV: [True, True]}
                    )
                    for page, expected_payload in zip((4, 24), payloads):
                        self.assertTrue(
                            torch.equal(host.get_data_page(page), expected_payload)
                        )
                    for buffer in _components(pool):
                        buffer.fill_(-5)
                    with simulator.patch():
                        for layer in range(pool.layer_num):
                            host.load_to_device_per_layer(
                                pool,
                                host_indices,
                                _pages(4, 2),
                                layer,
                                "kernel_ascend",
                            )
                    for buffer, source in zip(_components(pool), expected):
                        self.assertTrue(torch.equal(buffer[:, 4], source[:, 3]))
                        self.assertTrue(torch.equal(buffer[:, 2], source[:, 1]))

    def test_equal_size_unified_split_and_legacy_pages_are_isolated(self):
        _, unified = self._host(unified=True)
        _, split = self._host(unified=False)
        self.assertEqual(unified.get_data_page(0).nbytes, split.get_data_page(0).nbytes)
        unified_backend = self._backend(unified)
        split_backend = self._backend(split)
        legacy_backend = HiCacheFile(_config())
        self.assertNotEqual(unified_backend.config_suffix, split_backend.config_suffix)
        self.assertNotEqual(unified_backend.config_suffix, legacy_backend.config_suffix)
        key = "same-prompt-hash"
        original = unified.get_data_page(0)
        original.copy_(torch.arange(original.numel(), dtype=original.dtype))
        self.assertTrue(unified_backend.set(key, original))
        for backend in (split_backend, legacy_backend):
            self.assertFalse(backend.exists(key))
            target = torch.full_like(original, -3)
            self.assertIsNone(backend.get(key, target))
            self.assertTrue(torch.all(target == -3))
        self.assertTrue(split_backend.set(key, torch.full_like(original, -7)))
        self.assertTrue(legacy_backend.set(key, torch.full_like(original, -11)))
        target = torch.empty_like(original)
        self.assertIs(unified_backend.get(key, target), target)
        self.assertTrue(torch.equal(target, original))
        self.assertEqual(len(list(self.storage_dir.glob("*.bin"))), 3)

    def test_physical_format_variants_cannot_consume_each_others_pages(self):
        _, source = self._host()
        writer = self._backend(source)
        key = "unchanged-token-hash"
        self.assertTrue(writer.set(key, source.get_dummy_flat_data_page()))
        variations = [
            {"dtype": torch.bfloat16},  # Same byte width as FP16.
            {"page_size": 2, "layers": 6},  # Same total page byte count.
            {"page_size": 8},
            {"layers": 4},
            {"index_width": 4},
            {"index_width": 0},
            {"start_layer": 8},
            {"latent": 6, "rope": 6},  # Same unified total width.
        ]
        for geometry in variations:
            with self.subTest(geometry=geometry):
                _, host = self._host(**geometry)
                backend = self._backend(host)
                self.assertNotEqual(host.storage_format, source.storage_format)
                self.assertNotEqual(backend.config_suffix, writer.config_suffix)
                self.assertFalse(backend.exists(key))
                self.assertEqual(backend.batch_exists([key]), 0)
                self.assertEqual(backend.batch_exists_v2([key]).kv_hit_pages, 0)

    def test_same_format_reuses_pages_across_different_pool_capacities(self):
        _, first_host = self._host(pages=6)
        _, larger_host = self._host(pages=12)
        self.assertNotEqual(first_host.size, larger_host.size)
        self.assertEqual(first_host.storage_format, larger_host.storage_format)
        first_backend = self._backend(first_host)
        larger_backend = self._backend(larger_host)
        self.assertEqual(first_backend.config_suffix, larger_backend.config_suffix)
        page = first_host.get_data_page(0)
        page.copy_(torch.arange(page.numel(), dtype=page.dtype))
        self.assertTrue(first_backend.set("shared-page", page))
        self.assertTrue(larger_backend.exists("shared-page"))
        larger_page = larger_host.get_data_page(0)
        self.assertIs(larger_backend.get("shared-page", larger_page), larger_page)
        self.assertTrue(torch.equal(larger_page, page))

    def test_cache_layout_environment_is_namespaced_and_frozen_at_creation(self):
        flag_names = (
            "SGLANG_USE_FIA_NZ",
            "SGLANG_NPU_DISABLE_YARN_KV_SINK",
            "ENABLE_QKV_NORM_ROPE_OPT",
        )
        key = "same-tokens-under-different-cache-layouts"
        records = []
        for number, flags in enumerate(product((False, True), repeat=3), start=1):
            with self.subTest(flags=flags):
                environment = dict(
                    zip(flag_names, ("1" if flag else "0" for flag in flags))
                )
                with mock.patch.dict(os.environ, environment):
                    _, host = self._host(unified=False)
                    _, same_format = self._host(unified=False, pages=12)
                    signature = host.storage_format
                # Factory construction happens after leaving the initialization
                # environment: no flag may be reread to change the namespace.
                backend = self._backend(host)
                reader = self._backend(same_format)
                self.assertEqual(host.storage_format, signature)
                self.assertEqual(same_format.storage_format, signature)
                self.assertEqual(backend.config_suffix, reader.config_suffix)
                self.assertFalse(backend.exists(key))
                payload = host.get_data_page(0)
                payload.fill_(number)
                self.assertTrue(backend.set(key, payload))
                self.assertTrue(reader.exists(key))
                target = same_format.get_dummy_flat_data_page()
                self.assertIs(reader.get(key, target), target)
                self.assertTrue(torch.equal(target, payload))
                records.append((flags, host, signature, backend.config_suffix, number))
        self.assertEqual(len({record[2] for record in records}), 8)
        self.assertEqual(len({record[3] for record in records}), 8)
        self.assertEqual(len(list(self.storage_dir.glob("*.bin"))), 8)
        for flags, host, signature, suffix, number in records:
            with self.subTest(frozen_flags=flags):
                opposite_environment = dict(
                    zip(flag_names, ("0" if flag else "1" for flag in flags))
                )
                with mock.patch.dict(os.environ, opposite_environment):
                    self.assertEqual(host.storage_format, signature)
                    backend = self._backend(host)
                    self.assertEqual(backend.config_suffix, suffix)
                    target = host.get_dummy_flat_data_page()
                    self.assertIs(backend.get(key, target), target)
                    # Identical token keys in every other layout namespace
                    # must neither replace nor leak into this host's payload.
                    self.assertTrue(torch.all(target == number))

    def test_factory_keeps_input_config_unchanged_across_repeated_creation(self):
        _, host = self._host()
        for model_name in (None, "test/mla"):
            with self.subTest(model_name=model_name):
                config = _config(model_name=model_name)
                before = copy.deepcopy(asdict(config))
                prepared = []
                prepare = host.prepare_storage_config

                def capture(backend_name, original):
                    self.assertIs(original, config)
                    result = prepare(backend_name, original)
                    self.assertIsNot(result, original)
                    prepared.append(result)
                    return result

                with mock.patch.object(host, "prepare_storage_config", capture):
                    backends = [self._backend(host, config) for _ in range(3)]
                self.assertEqual(asdict(config), before)
                self.assertEqual(len({b.config_suffix for b in backends}), 1)
                for result in prepared:
                    self.assertEqual(
                        result.model_name,
                        f"{model_name or ''}--{host.storage_format}",
                    )
                    self.assertEqual(result.model_name.count(host.storage_format), 1)
                    self.assertEqual(result.extra_config, before["extra_config"])

    def test_unsupported_backends_fail_before_loading_their_modules(self):
        _, host = self._host()
        config = _config(
            extra_config={
                "backend_name": "external",
                "module_path": "must_not_be_imported",
                "class_name": "ExternalBackend",
            }
        )
        backends = list(StorageBackendFactory._registry)
        backends.remove("file")
        backends.extend(["dynamic", "unknown-backend"])
        with (
            mock.patch.object(StorageBackendFactory, "_load_backend_class") as load,
            mock.patch.object(
                StorageBackendFactory, "_create_dynamic_backend"
            ) as dynamic,
        ):
            for backend_name in backends:
                with self.subTest(backend=backend_name):
                    with self.assertRaisesRegex(
                        NotImplementedError, "only the 'file' L3 backend"
                    ):
                        StorageBackendFactory.create_backend(
                            backend_name, config, host
                        )
            load.assert_not_called()
            dynamic.assert_not_called()

    def test_factory_preserves_existing_host_behavior_without_hook(self):
        config = _config()
        before = copy.deepcopy(asdict(config))
        direct = HiCacheFile(config)
        ordinary = StorageBackendFactory.create_backend("file", config, object())
        self.assertEqual(ordinary.config_suffix, direct.config_suffix)
        self.assertEqual(asdict(config), before)


if __name__ == "__main__":
    unittest.main()
