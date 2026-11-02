"""CPU dispatch/data-flow tests for FAS consuming restored expanded HiCache KV.

Run the production FAS dispatcher and slot builder against a small fake NPU
facade. Each logical [20, 256] cache row carries one CPU integer marker, so the
real eligible shapes can be checked without allocating large device tensors.
This verifies dispatch and payload selection, not NPU operator precision.
"""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/hardware_backend/npu/attention/prefill_fas.py"
SPEC = importlib.util.spec_from_file_location("hicache_fas_cpu_source", SOURCE)
fas = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fas
SPEC.loader.exec_module(fas)


class _Indices:
    def __init__(self, values, events, shape=None, device=None):
        self.values = list(values)
        self.events = events
        self.shape = shape or (len(self.values),)
        self.ndim = len(self.shape)
        self.device = device

    def to(self, _):
        return self

    def reshape(self, *shape):
        if shape != (-1,):
            raise AssertionError(f"Unexpected index reshape {shape}")
        return _Indices(self.values, self.events, device=self.device)

    def index_select(self, dim, indices):
        if dim != 0:
            raise AssertionError("Only index_select(0) is expected")
        return _Indices(
            [self.values[i] for i in indices.values],
            self.events,
            device=self.device,
        )

    def __mul__(self, number):
        return _Indices([v * number for v in self.values], self.events)

    def __add__(self, other):
        values = (
            other.values
            if isinstance(other, _Indices)
            else [other] * len(self.values)
        )
        return _Indices([a + b for a, b in zip(self.values, values)], self.events)

    def __floordiv__(self, number):
        return _Indices([v // number for v in self.values], self.events)

    def __mod__(self, number):
        return _Indices([v % number for v in self.values], self.events)

    def record_stream(self, stream):
        self.events.append(("record_slots", stream))


class _Rows:
    def __init__(self, markers, name, events):
        self.markers = list(markers)
        self.name = name
        self.events = events

    def index_select(self, dim, slots):
        if dim != 0:
            raise AssertionError("Only the token axis may be gathered")
        self.events.append(("gather", self.name))
        return _Rows([self.markers[i] for i in slots.values], self.name, self.events)


class _Cache:
    def __init__(self, markers, name, events, device, dtype):
        self.markers = list(markers)
        self.name = name
        self.events = events
        self.device = device
        self.dtype = dtype
        self.shape = (len(self.markers) // 128, 128, 20, 256)
        self.ndim = 4
        self.contiguous = True

    def is_contiguous(self):
        return self.contiguous

    def reshape(self, *shape):
        if shape != (-1, 20, 256):
            raise AssertionError(f"Unexpected cache reshape {shape}")
        return _Rows(self.markers, self.name, self.events)


class _Stream:
    def __init__(self, name, events):
        self.name, self.events = name, events

    def wait_event(self, event):
        self.events.append(("wait_slots", self, event))


class _Event:
    def __init__(self, events):
        self.events = events

    def record(self, stream):
        self.events.append(("slots_ready", self, stream))


class _Fixture:
    def __init__(self, q_lengths=(4096, 4096), prefixes=(640, 640)):
        self.events = []
        self.device = SimpleNamespace(type="npu")
        self.bfloat16, self.float16 = object(), object()
        self.stream = _Stream("producer", self.events)
        self.kv_lengths = tuple(q + p for q, p in zip(q_lengths, prefixes))
        # Disjoint, interleaved physical blocks exercise the block-table path;
        # the second sequence must not continue reading the first one's rows.
        columns = (max(self.kv_lengths) + 127) // 128
        self.tables = [
            [1 + batch + 2 * column for column in range(columns)]
            for batch in range(len(q_lengths))
        ]
        physical_pages = max(max(row) for row in self.tables) + 1
        self.caches = {}
        for layer in (7, 8):
            key = [-1000] * (physical_pages * 128)
            value = [-2000] * (physical_pages * 128)
            for batch, length in enumerate(self.kv_lengths):
                for token in range(length):
                    slot = self.tables[batch][token // 128] * 128 + token % 128
                    # Restored prefix and newly scattered rows have different
                    # provenance markers; every token and layer is distinct.
                    provenance = 100000 if token < prefixes[batch] else 200000
                    key[slot] = layer * 1000000 + provenance + slot
                    value[slot] = -key[slot]
            self.caches[layer] = (
                _Cache(key, f"K{layer}", self.events, self.device, self.bfloat16),
                _Cache(value, f"V{layer}", self.events, self.device, self.bfloat16),
            )
        table = _Indices(
            [block for row in self.tables for block in row],
            self.events,
            shape=(len(q_lengths), columns),
            device=self.device,
        )
        self.metadata = SimpleNamespace(
            prefill_fas_decision=None,
            prefill_fas_staging=None,
            seq_lens_list_cumsum=fas._cumulative(q_lengths),
            block_tables=table,
        )

        def get_cache(layer):
            self.events.append(("get_cache", layer))
            return self.caches[layer]

        pool = SimpleNamespace(
            page_size=128,
            prefill_expanded_geometry=object(),
            get_prefill_expanded_kv=get_cache,
        )
        self.backend = SimpleNamespace(
            graph_mode=False,
            forward_metadata=self.metadata,
            token_to_kv_pool=pool,
            _get_seq_lens_cpu_int_list=lambda: self.kv_lengths,
            fia_mask=object(),
        )
        self.query = SimpleNamespace(
            shape=(sum(q_lengths), 20, 256),
            dtype=self.bfloat16,
            device=self.device,
            is_contiguous=lambda: True,
        )
        self.layer = SimpleNamespace(
            layer_id=7,
            tp_q_head_num=20,
            tp_k_head_num=20,
            qk_head_dim=256,
            v_head_dim=256,
            scaling=0.125,
        )
        self.batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_extend=lambda: True,
                is_target_verify=lambda: False,
                is_draft_extend_v2=lambda: False,
            ),
            extend_seq_lens_cpu=q_lengths,
            extend_prefix_lens_cpu=prefixes,
            num_token_non_padded_cpu=sum(q_lengths),
        )
        modules = {}
        for name in (
            "sglang",
            "sglang.srt",
            "sglang.srt.model_executor",
            "sglang.srt.model_executor.runner_backend_utils",
        ):
            module = ModuleType(name)
            module.__path__ = []
            modules[name] = module
        runtime = ModuleType("sglang.srt.runtime_context")
        self.disagg = SimpleNamespace(disaggregation_mode="prefill")
        runtime.get_disagg = lambda: self.disagg
        modules[runtime.__name__] = runtime
        environ = ModuleType("sglang.srt.environ")
        environ.envs = SimpleNamespace(
            SGLANG_NPU_PREFILL_FAS_EXTENDED_SHAPES=SimpleNamespace(get=lambda: False)
        )
        modules[environ.__name__] = environ
        context = ModuleType(
            "sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph"
        )
        context.get_tc_piecewise_forward_context = lambda: None
        modules[context.__name__] = context
        torch = ModuleType("torch")
        torch.bfloat16, torch.int64 = self.bfloat16, object()
        torch.compiler = SimpleNamespace(is_compiling=lambda: False)
        torch.npu = SimpleNamespace(
            current_stream=lambda: self.stream,
            is_current_stream_capturing=lambda: False,
            Event=lambda: _Event(self.events),
        )
        torch.arange = lambda length, **_: _Indices(range(length), self.events)
        torch.cat = lambda arrays: _Indices(
            [v for array in arrays for v in array.values], self.events
        )
        modules["torch"] = torch
        torch_npu = ModuleType("torch_npu")
        self.output = object()
        self.fusion = Mock(
            side_effect=lambda *args, **kwargs: (self.output,) + (None,) * 6
        )
        torch_npu.npu_fusion_attention = self.fusion
        modules["torch_npu"] = torch_npu
        self.modules = modules

    def run(self):
        with patch.dict(sys.modules, self.modules):
            return fas.maybe_forward_expanded_prefill_fas(
                self.backend, self.query, self.layer, self.batch
            )

    def expected_rows(self, layer, value=False):
        cache = self.caches[layer][int(value)]
        return [
            cache.markers[self.tables[batch][token // 128] * 128 + token % 128]
            for batch, length in enumerate(self.kv_lengths)
            for token in range(length)
        ]


class TestHiCacheFASCPU(unittest.TestCase):
    def test_eligible_restored_prefix_and_new_rows_reach_actual_fas_call(self):
        fixture = _Fixture()
        self.assertIs(fixture.run(), fixture.output)
        fixture.fusion.assert_called_once()
        query, dense_key, dense_value = fixture.fusion.call_args.args
        self.assertIs(query, fixture.query)
        self.assertEqual(dense_key.markers, fixture.expected_rows(7))
        self.assertEqual(dense_value.markers, fixture.expected_rows(7, value=True))
        self.assertEqual(len(dense_key.markers), 9472)
        options = fixture.fusion.call_args.kwargs
        self.assertEqual(options["actual_seq_qlen"], (4096, 8192))
        self.assertEqual(options["actual_seq_kvlen"], (4736, 9472))
        self.assertEqual(options["input_layout"], "TND")
        self.assertEqual(options["head_num"], 20)
        self.assertEqual(options["keep_prob"], 1.0)
        self.assertEqual(options["sparse_mode"], 3)
        self.assertIs(options["atten_mask"], fixture.backend.fia_mask)
        self.assertEqual(options["scale"], fixture.layer.scaling)
        self.assertEqual(fas.prefill_fas_step_counts(fixture.backend), {"eligible": 1})
        self.assertEqual(fixture.backend.prefill_fas_execution_count, 1)

    def test_slots_reuse_does_not_reuse_previous_layers_kv_payload(self):
        fixture = _Fixture()
        fixture.run()
        first_staging = fixture.metadata.prefill_fas_staging
        fixture.layer.layer_id = 8
        fixture.run()
        self.assertEqual(fixture.fusion.call_count, 2)
        self.assertIs(fixture.metadata.prefill_fas_staging, first_staging)
        first_key = fixture.fusion.call_args_list[0].args[1]
        second_key = fixture.fusion.call_args_list[1].args[1]
        self.assertNotEqual(first_key.markers, second_key.markers)
        self.assertEqual(second_key.markers, fixture.expected_rows(8))
        self.assertEqual(
            fixture.fusion.call_args_list[1].args[2].markers,
            fixture.expected_rows(8, value=True),
        )
        self.assertEqual(fas.prefill_fas_step_counts(fixture.backend), {"eligible": 1})
        self.assertEqual(fixture.backend.prefill_fas_execution_count, 2)
        self.assertEqual(
            len([event for event in fixture.events if event[0] == "slots_ready"]),
            1,
        )

    def test_cross_stream_wait_precedes_both_kv_gathers(self):
        fixture = _Fixture()
        fixture.run()
        staging = fixture.metadata.prefill_fas_staging
        fixture.events.clear()
        fixture.stream = _Stream("consumer", fixture.events)
        fixture.run()
        wait = next(
            i for i, event in enumerate(fixture.events) if event[0] == "wait_slots"
        )
        record = next(
            i for i, event in enumerate(fixture.events) if event[0] == "record_slots"
        )
        gathers = [i for i, event in enumerate(fixture.events) if event[0] == "gather"]
        self.assertEqual(len(gathers), 2)
        self.assertLess(wait, record)
        self.assertLess(record, min(gathers))
        self.assertIs(fixture.events[wait][2], staging.ready_event)
        self.assertEqual(fixture.fusion.call_count, 2)
        self.assertEqual(fixture.backend.prefill_fas_execution_count, 2)

    def test_fallbacks_never_call_fusion(self):
        cases = [
            ("fresh_unchanged", lambda: _Fixture(prefixes=(0, 0)), None),
            ("batch_size", lambda: _Fixture((4096,), (640,)), None),
            (
                "query_dtype_layout",
                _Fixture,
                lambda f: setattr(f.query, "dtype", f.float16),
            ),
            (
                "cache_dtype_layout",
                _Fixture,
                lambda f: setattr(f.caches[7][0], "contiguous", False),
            ),
            (
                "not_pd_prefill",
                _Fixture,
                lambda f: setattr(f.disagg, "disaggregation_mode", "decode"),
            ),
            (
                "geometry",
                _Fixture,
                lambda f: setattr(f.backend.token_to_kv_pool, "page_size", 64),
            ),
            (
                "block_table",
                _Fixture,
                lambda f: setattr(f.metadata.block_tables, "shape", (2, 36)),
            ),
        ]
        for expected_reason, build, mutate in cases:
            with self.subTest(reason=expected_reason):
                fixture = build()
                if mutate is not None:
                    mutate(fixture)
                self.assertIsNone(fixture.run())
                fixture.fusion.assert_not_called()
                self.assertEqual(
                    fas.prefill_fas_step_counts(fixture.backend), {expected_reason: 1}
                )
                self.assertEqual(
                    getattr(fixture.backend, "prefill_fas_execution_count", 0), 0
                )

    def test_eligible_step_counter_alone_does_not_prove_later_layer_used_fas(self):
        fixture = _Fixture()
        fixture.run()
        fixture.layer.layer_id = 8
        fixture.caches[8][1].contiguous = False
        self.assertIsNone(fixture.run())
        self.assertEqual(fixture.fusion.call_count, 1)
        self.assertEqual(fas.prefill_fas_step_counts(fixture.backend), {"eligible": 1})
        self.assertEqual(fixture.backend.prefill_fas_execution_count, 1)

    def test_execution_count_and_first_execution_log_follow_operator_success(self):
        fixture = _Fixture()
        with patch.object(fas.logger, "info") as info:
            fixture.run()
            fixture.layer.layer_id = 8
            fixture.run()
        self.assertEqual(fixture.backend.prefill_fas_execution_count, 2)
        execution_logs = [
            call for call in info.call_args_list if "first execution:" in call.args[0]
        ]
        self.assertEqual(len(execution_logs), 1)
        self.assertEqual(execution_logs[0].args[1:], (7, 2, 8192, (640, 640)))
        fixture.fusion.side_effect = RuntimeError("operator submission failed")
        with self.assertRaisesRegex(RuntimeError, "operator submission failed"):
            fixture.run()
        # An eligible step whose operator raised must not increase the count.
        self.assertEqual(fixture.backend.prefill_fas_execution_count, 2)
        self.assertEqual(fas.prefill_fas_step_counts(fixture.backend), {"eligible": 1})

    def test_captured_graph_is_rejected_without_building_a_decision(self):
        fixture = _Fixture()
        fixture.backend.graph_mode = True
        self.assertIsNone(fixture.run())
        fixture.fusion.assert_not_called()
        self.assertEqual(fas.prefill_fas_step_counts(fixture.backend), {})
        self.assertIsNone(fixture.metadata.prefill_fas_staging)
        self.assertEqual(
            getattr(fixture.backend, "prefill_fas_execution_count", 0), 0
        )


if __name__ == "__main__":
    unittest.main()
