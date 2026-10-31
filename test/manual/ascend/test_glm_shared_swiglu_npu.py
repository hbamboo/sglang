"""Opt-in real NPU numerical, stream and graph checks for GLM SwiGLU fusion.

Run only on a free NPU, for example:
  ASCEND_RT_VISIBLE_DEVICES=7 python test/manual/ascend/test_glm_shared_swiglu_npu.py
The default token counts cover tiny decode, ordinary prefill and overlap-sized
prefill. Change SGLANG_SWIGLU_TEST_TOKENS for additional large batch coverage.
"""

import ast
from contextlib import contextmanager
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

import torch
import torch_npu

import sglang.srt.layers.quantization  # noqa: F401
import sglang.srt.layers.activation as activation
from sglang.srt.hardware_backend.npu.moe.shared_expert_overlap import (
    SharedExpertOverlap,
    shared_expert_overlap_context,
)
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import NPUMXFP8LinearMethod


@contextmanager
def fusion(enabled):
    saved = activation._SGLANG_NPU_FUSED_SWIGLU_QUANT
    activation._SGLANG_NPU_FUSED_SWIGLU_QUANT = enabled
    try:
        yield
    finally:
        activation._SGLANG_NPU_FUSED_SWIGLU_QUANT = saved


class MXLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        weight = torch.randn(out_features, in_features, device="npu", dtype=torch.bfloat16) * 0.02
        qw, scales = torch.ops.npu.npu_dynamic_mx_quant(weight, dst_type=torch.float8_e4m3fn)
        self.weight = qw.T
        self.weight_scale_inv = scales.transpose(0, 1)
        self.quant_method = NPUMXFP8LinearMethod()
        self._sgl_npu_mx_quant_fusable = True
        self.tp_size = 1
        self.bias = None

    def forward(self, x):
        return self.quant_method.apply(self, x), None


def make_shared():
    # GLM checkpoint hidden width=2048, shared intermediate width=1536.
    block = SimpleNamespace(
        tp_size=1, gate_up_proj=MXLinear(2048, 3072),
        act_fn=activation.SiluAndMul(), down_proj=MXLinear(1536, 2048),
    )
    model_path = Path(__file__).resolve().parents[3] / "python/sglang/srt/models/glm4_moe_lite.py"
    tree = ast.parse(model_path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Glm4MoeLiteMLP")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    ns = {}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(model_path), "exec"), ns)
    block.forward = MethodType(ns["forward"], block)
    return block


def run_overlap(pipeline, hidden):
    with shared_expert_overlap_context(pipeline):
        pipeline.start(hidden)
        pipeline.before_gmm2()
        pipeline.after_gmm2()
        return pipeline.finish()


class TestGLMSharedSwigluNPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1031)
        cls.shared = make_shared()
        cls.tokens = [int(v) for v in os.getenv("SGLANG_SWIGLU_TEST_TOKENS", "1,8,32,1024,8192").split(",")]

    @torch.inference_mode()
    def test_quant_bytes_scales_and_dense_mlp(self):
        for tokens in self.tokens:
            with self.subTest(tokens=tokens):
                x = torch.randn(tokens, 2048, device="npu", dtype=torch.bfloat16)
                gate_up = self.shared.gate_up_proj(x)[0]
                reference_q, reference_s = torch.ops.npu.npu_dynamic_mx_quant(
                    torch_npu.npu_swiglu(gate_up), dst_type=torch.float8_e4m3fn
                )
                with fusion(True):
                    q, scale = self.shared.act_fn(gate_up, quant_linear=self.shared.down_proj)
                    actual = self.shared.forward(x)
                with fusion(False):
                    reference = self.shared.forward(x)
                torch.npu.synchronize()
                self.assertTrue(torch.equal(q.view(torch.uint8), reference_q.view(torch.uint8)))
                self.assertTrue(torch.equal(scale.view(torch.uint8), reference_s.view(torch.uint8)))
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    @torch.inference_mode()
    def test_overlap_nondefault_caller_and_allocator_reuse(self):
        caller = torch.npu.Stream()
        caller.wait_stream(torch.npu.current_stream())
        pipeline = SharedExpertOverlap(object(), self.shared)
        for tokens in (32, 8192):
            with self.subTest(tokens=tokens), torch.npu.stream(caller):
                with fusion(False):
                    x = torch.randn(tokens, 2048, device="npu", dtype=torch.bfloat16)
                    expected = self.shared.forward(x)
                with fusion(True):
                    actual = run_overlap(pipeline, x)
                    # Release the input and pressure the allocator after close()
                    # has joined the side stream to this non-default caller.
                    del x
                    scratch = torch.empty(tokens, 4096, device="npu", dtype=torch.bfloat16)
                    scratch.fill_(17)
                caller.synchronize()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_overlap_graph_replay_with_new_values(self):
        pipeline = SharedExpertOverlap(object(), self.shared)
        x = torch.randn(32, 2048, device="npu", dtype=torch.bfloat16)
        with fusion(True):
            warmup = torch.npu.Stream()
            warmup.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(warmup):
                for _ in range(3):
                    run_overlap(pipeline, x)
            torch.npu.current_stream().wait_stream(warmup)
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                actual = run_overlap(pipeline, x)
            for factor in (0.25, -0.75):
                x.copy_(torch.randn_like(x) * factor)
                with fusion(False):
                    reference = self.shared.forward(x)
                graph.replay()
                torch.npu.synchronize()
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
