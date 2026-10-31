"""CPU-only producer/consumer and stream-order regression for GLM SwiGLU.

No SGLang or torch import is needed: execute the real leaf methods with test
doubles so this can run while all NPUs are serving a benchmark.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from contextlib import nullcontext


SOURCE_ROOT = Path(os.getenv("SGLANG_TEST_SOURCE_ROOT", Path(__file__).resolve().parents[3]))


def source_path(relative):
    path = SOURCE_ROOT / relative
    return path if path.exists() else SOURCE_ROOT / Path(relative).name


def load_method(relative, class_name, method, namespace):
    tree = ast.parse(source_path(relative).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative, "exec"), namespace)
    return namespace[method]


class FakeTensor:
    def __init__(self, rows=8, dtype="bf16"):
        self.shape = (rows, 64)
        self.dtype = dtype

    def dim(self):
        return 2

    def numel(self):
        return self.shape[0] * self.shape[1]


class TestGLMSwigluContract(unittest.TestCase):
    def test_dense_mlp_passes_downstream_linear_and_tuple_without_requant(self):
        values, scales, output = object(), object(), object()
        down_input = []
        down = lambda x: (down_input.append(x) or output, None)
        activation_calls = []

        def activation(x, quant_linear=None):
            activation_calls.append((x, quant_linear))
            return values, scales

        block = SimpleNamespace(
            tp_size=1, gate_up_proj=lambda x: ("gate_up", None),
            act_fn=activation, down_proj=down,
        )
        forward = load_method("python/sglang/srt/models/glm4_moe_lite.py", "Glm4MoeLiteMLP", "forward", {})
        self.assertIs(forward(block, FakeTensor()), output)
        self.assertEqual(activation_calls, [("gate_up", down)])
        self.assertEqual(down_input, [(values, scales)])

    def test_empty_tp1_retains_early_exit(self):
        block = SimpleNamespace(tp_size=1)
        forward = load_method("python/sglang/srt/models/glm4_moe_lite.py", "Glm4MoeLiteMLP", "forward", {})
        empty = FakeTensor(rows=0)
        self.assertIs(forward(block, empty), empty)

    def test_existing_gate_and_fallback(self):
        relative = "python/sglang/srt/layers/activation.py"
        tree = ast.parse(source_path(relative).read_text())
        predicate = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_npu_fuse_swiglu_mx_quant")
        # Annotations are not part of the runtime behavior under test.
        predicate.returns = None
        for arg in predicate.args.args:
            arg.annotation = None
        ns = {"torch": SimpleNamespace(bfloat16="bf16"), "_SGLANG_NPU_FUSED_SWIGLU_QUANT": False}
        exec(compile(ast.Module(body=[predicate], type_ignores=[]), relative, "exec"), ns)
        gate = ns["_npu_fuse_swiglu_mx_quant"]
        linear = SimpleNamespace(_sgl_npu_mx_quant_fusable=True, tp_size=1)
        self.assertFalse(gate(FakeTensor(), linear))
        ns["_SGLANG_NPU_FUSED_SWIGLU_QUANT"] = True
        self.assertTrue(gate(FakeTensor(), linear))
        self.assertFalse(gate(FakeTensor(), None))
        self.assertFalse(gate(FakeTensor(rows=0), linear))
        self.assertFalse(gate(FakeTensor(dtype="fp16"), linear))
        self.assertFalse(gate(FakeTensor(), SimpleNamespace(tp_size=1)))
        self.assertFalse(gate(FakeTensor(), SimpleNamespace(_sgl_npu_mx_quant_fusable=True, tp_size=2)))

    def test_overlap_fused_and_serial_quantize_exactly_once_on_side_stream(self):
        relative = "python/sglang/srt/hardware_backend/npu/moe/shared_expert_overlap.py"
        for fused in (False, True):
            with self.subTest(fused=fused):
                calls = []
                values, scales = object(), object()
                down = object()

                class Event:
                    def record(self, stream):
                        calls.append(("record", stream))

                class Stream:
                    def wait_event(self, event):
                        calls.append(("wait", event))

                def activation(x, quant_linear=None):
                    calls.append(("activation", x, quant_linear))
                    return (values, scales) if fused else "activated"

                def quantize(x, dst_type):
                    calls.append(("quantize", x, dst_type))
                    return values, scales

                def stream_context(stream):
                    calls.append(("stream", stream))
                    return nullcontext()

                event, stream, main = Event(), Stream(), object()
                block = SimpleNamespace(
                    stage=1, main=main, stream=stream, gmm1_done=event,
                    _label=lambda label: nullcontext(), gate_up="gate_up",
                    shared=SimpleNamespace(act_fn=activation, down_proj=down),
                )
                ns = {"torch": SimpleNamespace(npu=SimpleNamespace(stream=stream_context), ops=SimpleNamespace(npu=SimpleNamespace(npu_dynamic_mx_quant=quantize)), float8_e4m3fn="fp8")}
                before_gmm2 = load_method(relative, "SharedExpertOverlap", "before_gmm2", ns)
                before_gmm2(block)
                self.assertEqual(calls[:3], [("record", main), ("wait", event), ("stream", stream)])
                self.assertIn(("activation", "gate_up", down), calls)
                self.assertEqual(sum(call[0] == "quantize" for call in calls), 0 if fused else 1)
                self.assertIs(block.quantized, values)
                self.assertIs(block.scales, scales)
                self.assertIsNone(block.gate_up)
                self.assertEqual(block.stage, 2)
                with self.assertRaises(RuntimeError):
                    before_gmm2(block)


if __name__ == "__main__":
    unittest.main(verbosity=2)
