"""CPU-only contracts for the opt-in eager FAS candidate; no torch import."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "python/sglang/srt/hardware_backend/npu/attention/prefill_fas.py"
spec = importlib.util.spec_from_file_location("prefill_fas_contract_source", SOURCE)
fas = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fas
spec.loader.exec_module(fas)


class PrefillFASContractTests(unittest.TestCase):
    def test_measured_large_prefix_shapes_are_eligible(self):
        for batch in (2, 4, 8):
            self.assertEqual(
                fas._shape_reason((4096,) * batch, (640,) * batch, 4096 * batch),
                "eligible",
            )

    def test_new_measured_mixed_positive_shapes(self):
        for q, prefix in [
            ((512, 3584, 4096), (0, 640, 640)),
            ((512, 3584, 4096), (640, 640, 640)),
            ((2048,) * 4, (640,) * 4),
            ((6528, 2048), (640, 512)),
            ((2048,) * 8, (640,) * 8),
        ]:
            with self.subTest(q=q, prefix=prefix):
                self.assertEqual(fas._shape_reason(q, prefix, sum(q)), "eligible")

    def test_negative_shapes_and_caps_keep_fia(self):
        examples = [
            ((1, 17, 129), (2856, 2840, 1024), 147, "total_tokens"),
            ((127, 128, 129), (128, 257, 511), 384, "total_tokens"),
            ((4096, 4096), (0, 0), 8192, "fresh_unchanged"),
            ((4096,), (640,), 4096, "batch_size"),
            ((1024,) * 8, (640,) * 8, 8192, "mean_query_length"),
            ((3744, 635, 635, 635, 635, 636, 636, 636), (640,) * 8, 8192, "mean_query_length"),
            ((511, 3585, 4096), (0, 640, 640), 8192, "extend_length"),
            ((6529, 2048), (640, 512), 8577, "extend_length"),
            ((2048, 2048, 2048, 2048, 2047), (640,) * 5, 10239, "mean_query_length"),
            ((4096, 4096), (641, 640), 8192, "prefix_length"),
            ((8192, 8192), (640, 640), 16384, "extend_length"),
            ((4096,) * 9, (640,) * 9, 36864, "batch_size"),
            ((4415,) * 8, (640,) * 8, 35320, "total_tokens"),
            ((4096, 4096), (640, 640), 8191, "length_metadata"),
            ((4096, 4096), (640,), 8192, "length_metadata"),
        ]
        for q, prefix, tokens, reason in examples:
            with self.subTest(q=q, prefix=prefix):
                self.assertEqual(fas._shape_reason(q, prefix, tokens), reason)

    def test_extended_shapes_measured_points_need_opt_in(self):
        for q, prefix in [
            ((2048, 4415), (256, 128)),
            ((2048, 4415), (640, 640)),
            ((2048,) * 3, (640,) * 3),
            ((512, 1536, 4096), (640,) * 3),
            ((2048,) * 12, (640,) * 12),
            ((2048,) * 16, (640,) * 16),
        ]:
            with self.subTest(q=q):
                self.assertNotEqual(fas._shape_reason(q, prefix, sum(q)), "eligible")
                self.assertEqual(fas._shape_reason(q, prefix, sum(q), extended_shapes=True), "eligible")

    def test_extended_shapes_keep_other_guard_boundaries(self):
        cases = [
            ((2048, 4095), (640,) * 2, "total_tokens"),
            ((1536,) * 4, (640,) * 4, "mean_query_length"),
            ((2048,) * 17, (640,) * 17, "batch_size"),
            ((2048,) * 15 + (2049,), (640,) * 16, "total_tokens"),
            ((511, 1537, 4096), (640,) * 3, "extend_length"),
            ((6529, 2048), (640, 640), "extend_length"),
            ((2048,) * 3, (641, 640, 640), "prefix_length"),
            ((2048,) * 3, (0,) * 3, "fresh_unchanged"),
            ((6144,), (640,), "batch_size"),
        ]
        for q, prefix, reason in cases:
            with self.subTest(q=q, reason=reason):
                self.assertEqual(fas._shape_reason(q, prefix, sum(q), extended_shapes=True), reason)

    @staticmethod
    def metadata():
        return SimpleNamespace(prefill_fas_decision=None, prefill_fas_staging=None)

    @staticmethod
    def decision(owner, reason="eligible"):
        return fas.PrefillFASDecision(
            owner=owner, reason=reason, q_lengths=(4096, 4096),
            kv_lengths=(4736, 4736), q_cumulative=(4096, 8192),
            kv_cumulative=(4736, 9472),
        )

    def test_decisions_and_reason_counts_are_once_per_step(self):
        backend, owner = SimpleNamespace(), object()
        metadata = self.metadata()
        build = Mock(return_value=self.decision(owner))
        first = fas._get_step_decision(backend, metadata, owner, build)
        for _ in range(47):
            self.assertIs(fas._get_step_decision(backend, metadata, owner, build), first)
        self.assertEqual(build.call_count, 1)
        self.assertEqual(fas.prefill_fas_step_counts(backend), {"eligible": 1})
        # The same ForwardBatch identity and shapes next step still use a new
        # ForwardMetadata. No shape-only/backend cache may suppress this count.
        next_metadata = self.metadata()
        next_build = Mock(return_value=self.decision(owner, "mean_query_length"))
        fas._get_step_decision(backend, next_metadata, owner, next_build)
        self.assertEqual(next_build.call_count, 1)
        self.assertEqual(fas.prefill_fas_step_counts(backend),
                         {"eligible": 1, "mean_query_length": 1})
        snapshot = fas.prefill_fas_step_counts(backend)
        snapshot.clear()
        self.assertEqual(backend.prefill_fas_step_reason_counts["eligible"], 1)

    def test_slots_reuse_within_step_and_rebuild_next_step(self):
        owner, block_tables = object(), object()
        decision = self.decision(owner)
        metadata = self.metadata()

        def build():
            return fas.PrefillFASStaging(owner, block_tables, decision.q_lengths,
                                        decision.kv_lengths, object(), object())

        builder = Mock(side_effect=build)
        first = fas._get_step_staging(metadata, decision, block_tables, builder)
        for _ in range(47):
            self.assertIs(fas._get_step_staging(metadata, decision, block_tables, builder), first)
        self.assertEqual(builder.call_count, 1)
        second = fas._get_step_staging(self.metadata(), decision, block_tables, builder)
        self.assertIsNot(second, first)
        self.assertEqual(builder.call_count, 2)
        changed_table = object()
        third = fas._get_step_staging(metadata, decision, changed_table, builder)
        self.assertIsNot(third, first)
        self.assertEqual(builder.call_count, 3)

    def test_same_stream_has_no_wait_or_record_stream(self):
        producer = Mock()
        slots = Mock()
        state = fas.PrefillFASStaging(object(), object(), (), (), slots, producer)
        event = Mock()
        state.ready_event = event
        fas._slots_for_stream(state, producer)
        event.record.assert_not_called()
        producer.wait_event.assert_not_called()
        slots.record_stream.assert_not_called()
        consumer = Mock()
        fas._slots_for_stream(state, consumer)
        event.record.assert_not_called()
        consumer.wait_event.assert_called_once_with(event)
        slots.record_stream.assert_called_once_with(consumer)
        second_consumer = Mock()
        fas._slots_for_stream(state, second_consumer)
        event.record.assert_not_called()
        second_consumer.wait_event.assert_called_once_with(event)

    def test_compile_piecewise_graph_and_capture_exit_before_step_state(self):
        package_names = [
            "sglang", "sglang.srt", "sglang.srt.model_executor",
            "sglang.srt.model_executor.runner_backend_utils",
        ]
        modules = {}
        for name in package_names:
            module = ModuleType(name)
            module.__path__ = []
            modules[name] = module
        runtime = ModuleType("sglang.srt.runtime_context")
        runtime.get_disagg = Mock()
        modules[runtime.__name__] = runtime
        context = ModuleType("sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph")
        modules[context.__name__] = context
        modules["torch_npu"] = ModuleType("torch_npu")
        for compiling, piecewise, graph_mode, capturing in [
            (True, None, False, False),
            (False, object(), False, False),
            (False, None, True, False),
            (False, None, False, True),
        ]:
            fake_torch = ModuleType("torch")
            fake_torch.compiler = SimpleNamespace(is_compiling=lambda: compiling)
            fake_torch.npu = SimpleNamespace(is_current_stream_capturing=lambda: capturing)
            modules["torch"] = fake_torch
            context.get_tc_piecewise_forward_context = lambda: piecewise
            backend = SimpleNamespace(graph_mode=graph_mode)
            with patch.dict(sys.modules, modules):
                # Deliberately no metadata, tensor, or device state: a graph
                # path must return before touching any of them.
                self.assertIsNone(fas.maybe_forward_expanded_prefill_fas(backend, None, None, None))
            self.assertFalse(hasattr(backend, "prefill_fas_step_reason_counts"))
        runtime.get_disagg.assert_not_called()

    def test_flag_is_off_metadata_is_new_and_no_errors_swallowed(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        self.assertFalse(any(isinstance(n, ast.ExceptHandler) for n in ast.walk(tree)))
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and n.func.attr == "synchronize" for n in ast.walk(tree)))
        environ = ast.parse((ROOT / "python/sglang/srt/environ.py").read_text(encoding="utf-8"))
        flags = [n for n in ast.walk(environ) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "SGLANG_NPU_PREFILL_FAS" for t in n.targets)]
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0].value.func.id, "EnvBool")
        self.assertIs(flags[0].value.args[0].value, False)
        extended = [n for n in ast.walk(environ) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "SGLANG_NPU_PREFILL_FAS_EXTENDED_SHAPES" for t in n.targets)]
        self.assertEqual(len(extended), 1)
        self.assertEqual(extended[0].value.func.id, "EnvBool")
        self.assertIs(extended[0].value.args[0].value, False)
        backend = ast.parse((ROOT / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py").read_text(encoding="utf-8"))
        metadata = next(n for n in backend.body if isinstance(n, ast.ClassDef) and n.name == "ForwardMetadata")
        fields = {n.target.id: n.value for n in metadata.body if isinstance(n, ast.AnnAssign)}
        self.assertIs(fields["prefill_fas_decision"].value, None)
        self.assertIs(fields["prefill_fas_staging"].value, None)
        cls = next(n for n in backend.body if isinstance(n, ast.ClassDef) and n.name == "AscendAttnBackend")
        initialize = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "init_forward_metadata")
        self.assertTrue(any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                            and isinstance(n.value.func, ast.Name) and n.value.func.id == "ForwardMetadata"
                            for n in ast.walk(initialize)))


if __name__ == "__main__":
    unittest.main()
