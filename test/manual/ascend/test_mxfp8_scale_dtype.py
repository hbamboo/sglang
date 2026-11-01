"""MXFP8 scale dtype regression for eager, Meta, and compiled NPU matmuls.

Run on Ascend A5: python test/manual/ascend/test_mxfp8_scale_dtype.py
"""

import unittest
from types import SimpleNamespace

import torch
import torch_npu

# Initialize quantization registrations before importing their NPU implementation.
import sglang.srt.layers.quantization  # noqa: F401
from torch._subclasses.fake_tensor import FakeTensorMode

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUMXFP8LinearMethod,
    _get_float8_e8m0fnu_dtype,
)


class TestMXFP8ScaleDtype(unittest.TestCase):
    def test_npu_scalar_type(self):
        self.assertEqual(_get_float8_e8m0fnu_dtype(), torch_npu.float8_e8m0fnu)

    def test_fake_tensor_buckets(self):
        method = NPUMXFP8LinearMethod()
        with FakeTensorMode():
            layer = SimpleNamespace(
                weight=torch.empty(
                    (1344, 2048), device="npu", dtype=torch.float8_e4m3fn
                ).T,
                weight_scale_inv=torch.empty(
                    (1344, 32, 2), device="npu", dtype=torch.uint8
                ).transpose(0, 1),
            )
            for tokens in (1024, 2048, 4096, 5120):
                with self.subTest(tokens=tokens):
                    qx = torch.empty(
                        (tokens, 2048), device="npu", dtype=torch.float8_e4m3fn
                    )
                    scale = torch.empty(
                        (tokens, 32, 2), device="npu", dtype=torch.uint8
                    )
                    y = method.apply_quantized(layer, qx, scale, torch.bfloat16)
                    self.assertEqual(y.shape, (tokens, 1344))
                    self.assertEqual(y.dtype, torch.bfloat16)

    @torch.inference_mode()
    def test_eager_compile_and_replay(self):
        torch.manual_seed(123)
        method = NPUMXFP8LinearMethod()
        w = torch.randn((1344, 2048), device="npu", dtype=torch.bfloat16)
        qw, ws = torch.ops.npu.npu_dynamic_mx_quant(w, dst_type=torch.float8_e4m3fn)
        layer = SimpleNamespace(weight=qw.T, weight_scale_inv=ws.transpose(0, 1))

        def forward(qx, scale):
            return method.apply_quantized(layer, qx, scale, torch.bfloat16)

        compiled = torch.compile(
            forward, backend="aot_eager", fullgraph=True, dynamic=True
        )
        for tokens in (1024, 2048, 4096, 5120):
            with self.subTest(tokens=tokens):
                x = torch.randn((tokens, 2048), device="npu", dtype=torch.bfloat16)
                qx, xs = torch.ops.npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
                reference = torch.ops.npu.npu_quant_matmul(
                    qx,
                    layer.weight,
                    layer.weight_scale_inv,
                    scale_dtype=torch.float8_e8m0fnu,
                    pertoken_scale=xs,
                    pertoken_scale_dtype=torch.float8_e8m0fnu,
                    output_dtype=torch.bfloat16,
                    group_sizes=[1, 1, 32],
                )
                torch.testing.assert_close(forward(qx, xs), reference, rtol=0, atol=0)
                torch.testing.assert_close(compiled(qx, xs), reference, rtol=0, atol=0)

        # Capture the production helper path, then change the graph input before
        # replay so the check cannot pass on a stale capture output.
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                forward(qx, xs)
        torch.npu.current_stream().wait_stream(stream)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            captured = forward(qx, xs)
        qx2, xs2 = torch.ops.npu.npu_dynamic_mx_quant(
            x * 0.5, dst_type=torch.float8_e4m3fn
        )
        qx.copy_(qx2)
        xs.copy_(xs2)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(captured, forward(qx, xs), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
