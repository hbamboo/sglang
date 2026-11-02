"""CPU-only unit tests for the opt-in MHA RoPE cross-layer cache.

These tests never import ``sglang`` or ``torch_npu`` and never launch an NPU
kernel.  The helper module is loaded directly from its source file and its
pure cache logic is exercised with a mock rotary that reproduces the existing
``DeepseekScalingRotaryEmbedding.get_cos_sin_cache`` implementation.

The last tests are source-level contracts: they assert the real prepare call
site was rewired and that the real model forward still resets the per-forward
dictionary.  They are wiring checks, not NPU integration results.

Run with::

    python3 -m pytest test/registered/unit/npu/attention/test_mha_rope_cache_cpu.py -v
    # or: python3 test/registered/unit/npu/attention/test_mha_rope_cache_cpu.py -v
"""

import importlib.util
import pathlib
import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[5]
HELPER_PATH = (
    REPO_ROOT
    / "python/sglang/srt/hardware_backend/npu/attention/mha_rope_cache.py"
)
PREPARE_PATH = (
    REPO_ROOT
    / "python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py"
)
MODEL_PATH = REPO_ROOT / "python/sglang/srt/models/glm4_moe_lite.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location(
        "mha_rope_cache_under_test", HELPER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cache_mod = _load_helper()

TABLE_ROWS = 20000
ROPE_DIM = 64


def make_tables(rows=TABLE_ROWS):
    """Simple, independent FP32 source tables (no randomness)."""
    row = torch.arange(rows, dtype=torch.float32).unsqueeze(1) * 0.5
    col = torch.arange(ROPE_DIM, dtype=torch.float32).unsqueeze(0) * 0.01
    cos_table = row + col + 1.0
    sin_table = row * 0.25 - col + 1000.0
    return cos_table, sin_table


class MockRotary:
    """Mirrors rope_variant.py::get_cos_sin_cache for counting and comparison."""

    def __init__(self, cos_table, sin_table):
        self.cos_cached_total = cos_table
        self.sin_cached_total = sin_table
        self.get_calls = 0
        self.gathers = 0

    def get_cos_sin_cache(self, positions, dtype, offsets=None):
        self.get_calls += 1
        index = positions if offsets is None else torch.add(positions, offsets)
        # Two table Index ops in the original implementation.
        cos = self.cos_cached_total[index]
        sin = self.sin_cached_total[index]
        self.gathers += 2
        cos = cos.unsqueeze(-2).unsqueeze(-2).to(dtype)
        sin = sin.unsqueeze(-2).unsqueeze(-2).to(dtype)
        return cos.to(positions.device), sin.to(positions.device)


def expected_cos_sin(rotary, positions, dtype, offsets=None):
    """Independent expected values: same public semantics, no helper call."""
    index = positions if offsets is None else torch.add(positions, offsets)
    cos = (
        rotary.cos_cached_total[index]
        .unsqueeze(-2)
        .unsqueeze(-2)
        .to(dtype)
        .to(positions.device)
    )
    sin = (
        rotary.sin_cached_total[index]
        .unsqueeze(-2)
        .unsqueeze(-2)
        .to(dtype)
        .to(positions.device)
    )
    return cos, sin


@contextmanager
def forced_eligible_guards(capturing=False, compiling=False):
    """Patch runtime guards so the production wrapper can run on CPU tensors."""
    with patch.object(cache_mod, "_ENABLED", True), patch.object(
        cache_mod, "_is_compiling", return_value=compiling
    ), patch.object(
        cache_mod, "_positions_device_supported", return_value=True
    ), patch.object(
        cache_mod, "_is_stream_capturing", return_value=capturing
    ), patch.object(
        cache_mod, "_current_stream", return_value=None
    ):
        yield


def make_positions(size, dtype=torch.long, seed=1234):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, TABLE_ROWS, (size,), generator=generator, dtype=torch.long)


class TestValueEquivalence(unittest.TestCase):
    def test_hit_matches_original_for_supported_lengths(self):
        cos_table, sin_table = make_tables()
        for length in (0, 1, 31, 128, 3787, 15148):
            with self.subTest(length=length):
                rotary = MockRotary(cos_table, sin_table)
                positions = make_positions(length)
                cache = {}
                cos, sin = cache_mod.get_or_compute_cached(
                    rotary, positions, torch.bfloat16, cache
                )
                want_cos, want_sin = expected_cos_sin(
                    rotary, positions, torch.bfloat16
                )
                self.assertEqual(tuple(cos.shape), (length, 1, 1, ROPE_DIM))
                self.assertEqual(cos.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(cos, want_cos))
                self.assertTrue(torch.equal(sin, want_sin))
                self.assertEqual(tuple(cos.stride()), tuple(want_cos.stride()))
                self.assertEqual(tuple(sin.stride()), tuple(want_sin.stride()))

                calls = rotary.get_calls
                gathers = rotary.gathers
                cos2, sin2 = cache_mod.get_or_compute_cached(
                    rotary, positions, torch.bfloat16, cache
                )
                self.assertIs(cos2, cos)
                self.assertIs(sin2, sin)
                self.assertEqual(rotary.get_calls, calls)
                self.assertEqual(rotary.gathers, gathers)

    def test_float16_and_float32_dtypes_are_preserved(self):
        cos_table, sin_table = make_tables()
        positions = torch.tensor([3, 0, 3, 7], dtype=torch.long)
        for dtype in (torch.float16, torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                rotary = MockRotary(cos_table, sin_table)
                cos, sin = cache_mod.get_or_compute_cached(
                    rotary, positions, dtype, {}
                )
                want_cos, want_sin = expected_cos_sin(rotary, positions, dtype)
                self.assertEqual(cos.dtype, dtype)
                self.assertTrue(torch.equal(cos, want_cos))
                self.assertTrue(torch.equal(sin, want_sin))

    def test_non_monotonic_and_duplicate_positions(self):
        cos_table, sin_table = make_tables()
        rotary = MockRotary(cos_table, sin_table)
        positions = torch.tensor([9, 1, 9, 0, 5, 5, 2], dtype=torch.long)
        cos, sin = cache_mod.get_or_compute_cached(
            rotary, positions, torch.bfloat16, {}
        )
        want_cos, want_sin = expected_cos_sin(rotary, positions, torch.bfloat16)
        self.assertTrue(torch.equal(cos, want_cos))
        self.assertTrue(torch.equal(sin, want_sin))


class TestMissCost(unittest.TestCase):
    def test_first_call_does_exactly_two_gathers_then_zero(self):
        rotary = MockRotary(*make_tables())
        positions = torch.tensor([1, 2, 3], dtype=torch.long)
        cache = {}
        cache_mod.get_or_compute_cached(rotary, positions, torch.bfloat16, cache)
        self.assertEqual(rotary.get_calls, 1)
        self.assertEqual(rotary.gathers, 2)
        for _ in range(5):
            cache_mod.get_or_compute_cached(
                rotary, positions, torch.bfloat16, cache
            )
        self.assertEqual(rotary.get_calls, 1)
        self.assertEqual(rotary.gathers, 2)
        self.assertEqual(len(cache), 1)  # one private namespace entry

    def test_47_layers_share_one_lookup(self):
        rotary = MockRotary(*make_tables())
        positions = make_positions(2048)
        cache = {}
        results = [
            cache_mod.get_or_compute_cached(
                rotary, positions, torch.bfloat16, cache
            )
            for _ in range(47)
        ]
        self.assertEqual(rotary.get_calls, 1)
        self.assertEqual(rotary.gathers, 2)
        for cos, sin in results[1:]:
            self.assertIs(cos, results[0][0])
            self.assertIs(sin, results[0][1])

    def test_table_boundary_rows(self):
        rotary = MockRotary(*make_tables())
        positions = torch.tensor([0, TABLE_ROWS - 1], dtype=torch.long)
        cos, sin = cache_mod.get_or_compute_cached(
            rotary, positions, torch.bfloat16, {}
        )
        want_cos, want_sin = expected_cos_sin(rotary, positions, torch.bfloat16)
        self.assertTrue(torch.equal(cos, want_cos))
        self.assertTrue(torch.equal(sin, want_sin))


class TestInvalidation(unittest.TestCase):
    def test_per_forward_reset_same_shape_and_storage(self):
        rotary = MockRotary(*make_tables())
        positions = torch.tensor([1, 2, 3], dtype=torch.long)
        cache = {}
        first = cache_mod.get_or_compute_cached(
            rotary, positions, torch.bfloat16, cache
        )
        # Framework resets the whole dictionary at every Python forward.
        cache = {}
        second = cache_mod.get_or_compute_cached(
            rotary, positions, torch.bfloat16, cache
        )
        self.assertEqual(rotary.get_calls, 2)
        self.assertIsNot(first[0], second[0])
        self.assertTrue(torch.equal(first[0], second[0]))

    def test_per_forward_reset_new_positions_same_storage(self):
        rotary = MockRotary(*make_tables())
        positions = torch.tensor([1, 2, 3], dtype=torch.long)
        cache_mod.get_or_compute_cached(
            rotary, positions, torch.bfloat16, {}
        )
        positions.copy_(torch.tensor([9, 8, 7]))
        cache = {}
        cos, sin = cache_mod.get_or_compute_cached(
            rotary, positions, torch.bfloat16, cache
        )
        want_cos, want_sin = expected_cos_sin(rotary, positions, torch.bfloat16)
        self.assertTrue(torch.equal(cos, want_cos))
        self.assertTrue(torch.equal(sin, want_sin))

    def test_inplace_positions_mutation_misses(self):
        rotary = MockRotary(*make_tables())
        positions = torch.tensor([4, 5, 6], dtype=torch.long)
        cache = {}
        cache_mod.get_or_compute_cached(rotary, positions, torch.bfloat16, cache)
        positions.add_(3)
        cos, sin = cache_mod.get_or_compute_cached(
            rotary, positions, torch.bfloat16, cache
        )
        self.assertEqual(rotary.get_calls, 2)
        want_cos, want_sin = expected_cos_sin(rotary, positions, torch.bfloat16)
        self.assertTrue(torch.equal(cos, want_cos))
        self.assertTrue(torch.equal(sin, want_sin))

    def test_positions_copy_style_inplace_mutation_misses(self):
        rotary = MockRotary(*make_tables())
        positions = torch.tensor([4, 5, 6], dtype=torch.long)
        cache = {}
        cache_mod.get_or_compute_cached(rotary, positions, torch.bfloat16, cache)
        positions.copy_(torch.tensor([12, 13, 14]))
        cos, _ = cache_mod.get_or_compute_cached(
            rotary, positions, torch.bfloat16, cache
        )
        self.assertEqual(rotary.get_calls, 2)
        want_cos, _ = expected_cos_sin(rotary, positions, torch.bfloat16)
        self.assertTrue(torch.equal(cos, want_cos))

    def test_inplace_source_table_mutation_misses(self):
        rotary = MockRotary(*make_tables())
        positions = torch.tensor([2, 4], dtype=torch.long)
        cache = {}
        cache_mod.get_or_compute_cached(rotary, positions, torch.bfloat16, cache)
        rotary.cos_cached_total.add_(1.0)
        rotary.sin_cached_total.add_(2.0)
        cos, sin = cache_mod.get_or_compute_cached(
            rotary, positions, torch.bfloat16, cache
        )
        self.assertEqual(rotary.get_calls, 2)
        want_cos, want_sin = expected_cos_sin(rotary, positions, torch.bfloat16)
        self.assertTrue(torch.equal(cos, want_cos))
        self.assertTrue(torch.equal(sin, want_sin))

    def test_source_table_replacement_misses(self):
        cos_table, sin_table = make_tables()
        rotary = MockRotary(cos_table, sin_table)
        positions = torch.tensor([2, 4], dtype=torch.long)
        cache = {}
        cache_mod.get_or_compute_cached(rotary, positions, torch.bfloat16, cache)
        rotary.cos_cached_total = cos_table + 3.0
        cos, _ = cache_mod.get_or_compute_cached(
            rotary, positions, torch.bfloat16, cache
        )
        self.assertEqual(rotary.get_calls, 2)
        want_cos, _ = expected_cos_sin(rotary, positions, torch.bfloat16)
        self.assertTrue(torch.equal(cos, want_cos))


class TestIdentityAndMetadata(unittest.TestCase):
    def test_positions_views_are_not_shared(self):
        rotary = MockRotary(*make_tables())
        buffer = torch.arange(32, dtype=torch.long)
        first_view = buffer[:8]
        second_view = buffer[:8]
        self.assertIsNot(first_view, second_view)
        cache = {}
        cache_mod.get_or_compute_cached(rotary, first_view, torch.bfloat16, cache)
        cos, _ = cache_mod.get_or_compute_cached(
            rotary, second_view, torch.bfloat16, cache
        )
        self.assertEqual(rotary.get_calls, 2)
        want_cos, _ = expected_cos_sin(rotary, second_view, torch.bfloat16)
        self.assertTrue(torch.equal(cos, want_cos))

    def test_non_contiguous_positions_view(self):
        rotary = MockRotary(*make_tables())
        positions = torch.arange(0, 40, dtype=torch.long)[::3]
        self.assertFalse(positions.is_contiguous())
        cos, sin = cache_mod.get_or_compute_cached(
            rotary, positions, torch.bfloat16, {}
        )
        want_cos, want_sin = expected_cos_sin(rotary, positions, torch.bfloat16)
        self.assertTrue(torch.equal(cos, want_cos))
        self.assertTrue(torch.equal(sin, want_sin))

    def test_distinct_rotary_instances_do_not_share_entries(self):
        cos_table, sin_table = make_tables()
        rotary_a = MockRotary(cos_table, sin_table)
        rotary_b = MockRotary(cos_table, sin_table)
        positions = torch.tensor([1, 2, 3], dtype=torch.long)
        cache = {}
        cache_mod.get_or_compute_cached(
            rotary_a, positions, torch.bfloat16, cache
        )
        cache_mod.get_or_compute_cached(
            rotary_b, positions, torch.bfloat16, cache
        )
        self.assertEqual(rotary_a.get_calls, 1)
        self.assertEqual(rotary_b.get_calls, 1)
        cache_mod.get_or_compute_cached(
            rotary_a, positions, torch.bfloat16, cache
        )
        self.assertEqual(rotary_a.get_calls, 1)

    def test_dtype_is_part_of_the_key(self):
        rotary = MockRotary(*make_tables())
        positions = torch.tensor([1, 2, 3], dtype=torch.long)
        cache = {}
        cache_mod.get_or_compute_cached(
            rotary, positions, torch.bfloat16, cache
        )
        cos, _ = cache_mod.get_or_compute_cached(
            rotary, positions, torch.float32, cache
        )
        self.assertEqual(rotary.get_calls, 2)
        self.assertEqual(cos.dtype, torch.float32)

    def test_stream_identity_is_part_of_the_key(self):
        rotary = MockRotary(*make_tables())
        positions = torch.tensor([1, 2, 3], dtype=torch.long)
        stream_a = SimpleNamespace(npu_stream=11)
        same_native_stream = SimpleNamespace(npu_stream=11)
        other_stream = SimpleNamespace(npu_stream=12)
        cache = {}
        cache_mod.get_or_compute_cached(
            rotary, positions, torch.bfloat16, cache, stream=stream_a
        )
        cache_mod.get_or_compute_cached(
            rotary, positions, torch.bfloat16, cache, stream=same_native_stream
        )
        self.assertEqual(rotary.get_calls, 1)
        cache_mod.get_or_compute_cached(
            rotary, positions, torch.bfloat16, cache, stream=other_stream
        )
        self.assertEqual(rotary.get_calls, 2)

    def test_offsets_hit_and_invalidation(self):
        rotary = MockRotary(*make_tables())
        positions = torch.tensor([1, 2, 3], dtype=torch.long)
        offsets = torch.tensor([10, 20, 30], dtype=torch.long)
        cache = {}
        cos, sin = cache_mod.get_or_compute_cached(
            rotary, positions, torch.bfloat16, cache, offsets=offsets
        )
        want_cos, want_sin = expected_cos_sin(
            rotary, positions, torch.bfloat16, offsets=offsets
        )
        self.assertTrue(torch.equal(cos, want_cos))
        self.assertTrue(torch.equal(sin, want_sin))
        self.assertEqual(rotary.get_calls, 1)
        cache_mod.get_or_compute_cached(
            rotary, positions, torch.bfloat16, cache, offsets=offsets
        )
        self.assertEqual(rotary.get_calls, 1)

        # Same values but a different offsets object is a distinct key.
        other_offsets = offsets.clone()
        cache_mod.get_or_compute_cached(
            rotary, positions, torch.bfloat16, cache, offsets=other_offsets
        )
        self.assertEqual(rotary.get_calls, 2)

        # Normal in-place mutation of the same offsets object invalidates.
        offsets.add_(1)
        cos, _ = cache_mod.get_or_compute_cached(
            rotary, positions, torch.bfloat16, cache, offsets=offsets
        )
        self.assertEqual(rotary.get_calls, 3)
        want_cos, _ = expected_cos_sin(
            rotary, positions, torch.bfloat16, offsets=offsets
        )
        self.assertTrue(torch.equal(cos, want_cos))


class TestFallbacks(unittest.TestCase):
    def test_missing_or_none_cache_calls_original(self):
        rotary = MockRotary(*make_tables())
        positions = torch.tensor([1, 2], dtype=torch.long)
        for cache in (None, [], 0, "not-a-dict"):
            with self.subTest(cache=cache):
                cos, sin = cache_mod.get_or_compute_cached(
                    rotary, positions, torch.bfloat16, cache
                )
                want_cos, want_sin = expected_cos_sin(
                    rotary, positions, torch.bfloat16
                )
                self.assertTrue(torch.equal(cos, want_cos))
                self.assertTrue(torch.equal(sin, want_sin))
        self.assertEqual(rotary.get_calls, 4)

    def test_missing_rotary_tables_fall_back(self):
        rotary = MockRotary(*make_tables())
        del rotary.cos_cached_total
        positions = torch.tensor([1, 2], dtype=torch.long)
        cache = {}
        # The original method would itself fail; the helper must not pretend
        # it can cache and hide that failure.
        with self.assertRaises(AttributeError):
            cache_mod.get_or_compute_cached(
                rotary, positions, torch.bfloat16, cache
            )
        self.assertEqual(cache, {})

    def test_non_tensor_positions_are_not_cached(self):
        rotary = MockRotary(*make_tables())
        self.assertFalse(
            cache_mod._cache_inputs_supported(rotary, [1, 2, 3], None)
        )
        cache = {}
        with self.assertRaises(AttributeError):
            cache_mod.get_or_compute_cached(
                rotary, [1, 2, 3], torch.bfloat16, cache
            )
        self.assertEqual(cache, {})

    def test_inference_positions_fall_back_without_crashing(self):
        rotary = MockRotary(*make_tables())
        cache = {}
        with torch.inference_mode():
            positions = torch.tensor([1, 2, 3], dtype=torch.long)
            self.assertTrue(cache_mod._is_inference_tensor(positions))
            self.assertIsNone(cache_mod._version_or_none(positions))
            for expected_calls in (1, 2):
                cos, _ = cache_mod.get_or_compute_cached(
                    rotary, positions, torch.float32, cache
                )
                self.assertEqual(rotary.get_calls, expected_calls)
                self.assertEqual(cache, {})
                want_cos, _ = expected_cos_sin(
                    rotary, positions, torch.float32
                )
                self.assertTrue(torch.equal(cos, want_cos))

    def test_inference_source_tables_fall_back_without_crashing(self):
        with torch.inference_mode():
            cos_table = torch.randn(64, 64)
            sin_table = torch.randn(64, 64)
        rotary = MockRotary(cos_table, sin_table)
        positions = torch.tensor([1, 2, 3], dtype=torch.long)
        cache = {}
        with torch.inference_mode():
            for expected_calls in (1, 2):
                cache_mod.get_or_compute_cached(
                    rotary, positions, torch.float32, cache
                )
                self.assertEqual(rotary.get_calls, expected_calls)
                self.assertEqual(cache, {})


class TestProductionGuards(unittest.TestCase):
    def _rotary_and_positions(self):
        return (
            MockRotary(*make_tables()),
            torch.tensor([1, 2, 3], dtype=torch.long),
        )

    def test_disabled_flag_calls_original_every_time(self):
        rotary, positions = self._rotary_and_positions()
        batch = SimpleNamespace(**{cache_mod.CACHE_ATTR: {}})
        with patch.object(cache_mod, "_ENABLED", False):
            for expected_calls in (1, 2):
                cache_mod.get_mha_rope_cos_sin(
                    rotary, positions, torch.bfloat16, batch
                )
                self.assertEqual(rotary.get_calls, expected_calls)
        self.assertEqual(batch.npu_mlaprolog_rope_cache, {})

    def test_compile_falls_back_before_cache_and_stream_queries(self):
        rotary, positions = self._rotary_and_positions()
        batch = SimpleNamespace(**{cache_mod.CACHE_ATTR: {}})
        with patch.object(cache_mod, "_ENABLED", True), patch.object(
            cache_mod, "_is_compiling", return_value=True
        ), patch.object(
            cache_mod, "_is_stream_capturing", side_effect=AssertionError
        ), patch.object(
            cache_mod, "_current_stream", side_effect=AssertionError
        ):
            cos, _ = cache_mod.get_mha_rope_cos_sin(
                rotary, positions, torch.bfloat16, batch
            )
        self.assertEqual(rotary.get_calls, 1)
        self.assertEqual(batch.npu_mlaprolog_rope_cache, {})
        self.assertEqual(cos.dtype, torch.bfloat16)

    def test_capture_falls_back_before_stream_query(self):
        rotary, positions = self._rotary_and_positions()
        batch = SimpleNamespace(**{cache_mod.CACHE_ATTR: {}})
        with patch.object(cache_mod, "_ENABLED", True), patch.object(
            cache_mod, "_is_compiling", return_value=False
        ), patch.object(
            cache_mod, "_positions_device_supported", return_value=True
        ), patch.object(
            cache_mod, "_is_stream_capturing", return_value=True
        ), patch.object(
            cache_mod, "_current_stream", side_effect=AssertionError
        ):
            cache_mod.get_mha_rope_cos_sin(
                rotary, positions, torch.bfloat16, batch
            )
        self.assertEqual(rotary.get_calls, 1)
        self.assertEqual(batch.npu_mlaprolog_rope_cache, {})

    def test_unsupported_device_falls_back(self):
        rotary, positions = self._rotary_and_positions()
        batch = SimpleNamespace(**{cache_mod.CACHE_ATTR: {}})
        with patch.object(cache_mod, "_ENABLED", True), patch.object(
            cache_mod, "_is_compiling", return_value=False
        ), patch.object(
            cache_mod, "_is_stream_capturing", side_effect=AssertionError
        ):
            cache_mod.get_mha_rope_cos_sin(
                rotary, positions, torch.bfloat16, batch
            )
        self.assertEqual(rotary.get_calls, 1)
        self.assertEqual(batch.npu_mlaprolog_rope_cache, {})

    def test_missing_dict_or_none_falls_back(self):
        rotary, positions = self._rotary_and_positions()
        batches = (
            None,
            SimpleNamespace(),
            SimpleNamespace(**{cache_mod.CACHE_ATTR: None}),
            SimpleNamespace(**{cache_mod.CACHE_ATTR: []}),
        )
        with forced_eligible_guards():
            for batch in batches:
                with self.subTest(batch=batch):
                    rotary.get_calls = 0
                    cache_mod.get_mha_rope_cos_sin(
                        rotary, positions, torch.bfloat16, batch
                    )
                    self.assertEqual(rotary.get_calls, 1)

    def test_unknown_stream_falls_back(self):
        rotary, positions = self._rotary_and_positions()
        batch = SimpleNamespace(**{cache_mod.CACHE_ATTR: {}})
        with patch.object(cache_mod, "_ENABLED", True), patch.object(
            cache_mod, "_is_compiling", return_value=False
        ), patch.object(
            cache_mod, "_positions_device_supported", return_value=True
        ), patch.object(
            cache_mod, "_is_stream_capturing", return_value=False
        ), patch.object(
            cache_mod, "_current_stream", return_value=cache_mod._UNKNOWN_STREAM
        ):
            cache_mod.get_mha_rope_cos_sin(
                rotary, positions, torch.bfloat16, batch
            )
        self.assertEqual(rotary.get_calls, 1)
        self.assertEqual(batch.npu_mlaprolog_rope_cache, {})

    def test_enabled_guarded_path_reuses(self):
        rotary, positions = self._rotary_and_positions()
        batch = SimpleNamespace(**{cache_mod.CACHE_ATTR: {}})
        with forced_eligible_guards():
            cos, sin = cache_mod.get_mha_rope_cos_sin(
                rotary, positions, torch.bfloat16, batch
            )
            cos2, sin2 = cache_mod.get_mha_rope_cos_sin(
                rotary, positions, torch.bfloat16, batch
            )
        self.assertEqual(rotary.get_calls, 1)
        self.assertIs(cos, cos2)
        self.assertIs(sin, sin2)
        want_cos, want_sin = expected_cos_sin(
            rotary, positions, torch.bfloat16
        )
        self.assertTrue(torch.equal(cos, want_cos))
        self.assertTrue(torch.equal(sin, want_sin))


class TestTboChildIndependence(unittest.TestCase):
    def test_each_child_dict_is_independent(self):
        rotary = MockRotary(*make_tables())
        child_a = SimpleNamespace(**{cache_mod.CACHE_ATTR: {}})
        child_b = SimpleNamespace(**{cache_mod.CACHE_ATTR: {}})
        pos_a = torch.tensor([1, 2, 3], dtype=torch.long)
        pos_b = torch.tensor([1, 2, 3], dtype=torch.long)
        with forced_eligible_guards():
            cos_a, _ = cache_mod.get_mha_rope_cos_sin(
                rotary, pos_a, torch.bfloat16, child_a
            )
            cos_b, _ = cache_mod.get_mha_rope_cos_sin(
                rotary, pos_b, torch.bfloat16, child_b
            )
            # Both children paid their own first miss.
            self.assertEqual(rotary.get_calls, 2)
            self.assertIsNot(cos_a, cos_b)
            cos_a2, _ = cache_mod.get_mha_rope_cos_sin(
                rotary, pos_a, torch.bfloat16, child_a
            )
            self.assertEqual(rotary.get_calls, 2)
            self.assertIs(cos_a, cos_a2)
        self.assertEqual(len(child_a.npu_mlaprolog_rope_cache), 1)
        self.assertEqual(len(child_b.npu_mlaprolog_rope_cache), 1)


class TestSourceContracts(unittest.TestCase):
    def test_prepare_call_site_uses_helper_with_forward_batch(self):
        text = PREPARE_PATH.read_text()
        self.assertIn(
            "from sglang.srt.hardware_backend.npu.attention.mha_rope_cache import",
            text,
        )
        self.assertIn("get_mha_rope_cos_sin", text)
        body = _function_body(text, "forward_mha_prepare_npu")
        self.assertIn("get_mha_rope_cos_sin(", body)
        self.assertIn("m.rotary_emb,", body)
        self.assertIn("forward_batch,", body)
        self.assertNotIn("m.rotary_emb.get_cos_sin_cache(", body)
        self.assertEqual(body.count("get_cos_sin_cache"), 0)
        # Adjacent owned hunks must remain untouched.
        self.assertIn("npu_interleave_rope", body)
        self.assertIn("npu_kv_rmsnorm_rope_cache", body)
        self.assertIn("m.kv_b_proj(kv_a)", body)

    def test_model_forward_resets_the_shared_dictionary(self):
        text = MODEL_PATH.read_text()
        marker = "forward_batch.npu_mlaprolog_rope_cache = {}"
        self.assertIn(marker, text)
        window_start = max(0, text.index(marker) - 300)
        window = text[window_start : text.index(marker) + 400]
        self.assertIn('getattr(forward_batch, "tbo_children", None)', window)
        self.assertIn("child.npu_mlaprolog_rope_cache = {}", window)
        self.assertIn("_is_npu", window)

    def test_helper_uses_the_reset_attribute_and_no_positions_to_cpu(self):
        text = HELPER_PATH.read_text()
        self.assertIn('CACHE_ATTR = "npu_mlaprolog_rope_cache"', text)
        self.assertNotIn("data_ptr", text)
        self.assertNotIn(".cpu(", text)
        self.assertNotIn(".tolist(", text)
        # Reuse must never fabricate a process-wide cache.
        self.assertNotIn("_GLOBAL", text)


def _function_body(text, name):
    start = text.index("def " + name + "(")
    end = text.find("\ndef ", start + 1)
    if end == -1:
        end = len(text)
    return text[start:end]


if __name__ == "__main__":
    unittest.main(verbosity=2)
