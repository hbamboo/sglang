"""Pending real-NPU checks for the GLM input-norm/QKV fusion candidate.

This file is intentionally not executed while service benchmarks occupy NPUs.
After applying the candidate, run on a free, explicitly selected device:
  ASCEND_RT_VISIBLE_DEVICES=0 python test/manual/ascend/test_glm_prefill_qkv_norm_npu.py
"""

from contextlib import contextmanager
import os
from types import SimpleNamespace
import unittest

import torch
import torch_npu

import sglang.srt.layers.quantization  # noqa: F401
import sglang.srt.layers.layernorm as layernorm
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import NPUMXFP8LinearMethod


@contextmanager
def norm_fusion(enabled):
    saved = layernorm._SGLANG_NPU_FUSED_ADD_RMSNORM_QUANT
    layernorm._SGLANG_NPU_FUSED_ADD_RMSNORM_QUANT = enabled
    try:
        yield
    finally:
        layernorm._SGLANG_NPU_FUSED_ADD_RMSNORM_QUANT = saved


class TestGLMPrefillQKVNormNPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1031)
        cls.method = NPUMXFP8LinearMethod()
        weight = torch.randn(1344, 2048, device="npu", dtype=torch.bfloat16) * 0.02
        qw, scales = torch.ops.npu.npu_dynamic_mx_quant(weight, dst_type=torch.float8_e4m3fn)
        cls.consumer = SimpleNamespace(
            weight=qw.T, weight_scale_inv=scales.transpose(0, 1),
            _sgl_npu_mx_quant_fusable=True, tp_size=1,
        )
        cls.norm = layernorm.RMSNorm(2048, eps=1e-5).to(device="npu", dtype=torch.bfloat16)
        # Nonuniform gamma checks multiplication and BF16 rounding, not only rms.
        cls.norm.weight.data.copy_(torch.randn(2048, device="npu", dtype=torch.bfloat16) * 0.1 + 1)
        cls.tokens = [int(v) for v in os.getenv("SGLANG_NORM_TEST_TOKENS", "1,32,1024,8192").split(",")]

    def compare(self, x, residual):
        reference_norm, _, reference_residual = torch_npu.npu_add_rms_norm(
            residual, x, self.norm.weight.data, self.norm.variance_epsilon
        )
        reference_q, reference_scale = torch.ops.npu.npu_dynamic_mx_quant(
            reference_norm, dst_type=torch.float8_e4m3fn
        )
        reference_projection = self.method.apply_quantized(
            self.consumer, reference_q, reference_scale, torch.bfloat16
        )
        with norm_fusion(True):
            quantized, actual_residual = self.norm(
                x, residual, quant_linear=self.consumer
            )
            self.assertIsInstance(quantized, tuple)
            q, scale = quantized
            actual_projection = self.method.apply(self.consumer, quantized)
        torch.npu.synchronize()
        self.assertTrue(torch.equal(q.view(torch.uint8), reference_q.view(torch.uint8)))
        self.assertTrue(torch.equal(scale.view(torch.uint8), reference_scale.view(torch.uint8)))
        torch.testing.assert_close(actual_residual, reference_residual, rtol=0, atol=0)
        torch.testing.assert_close(actual_projection, reference_projection, rtol=0, atol=0)

    @torch.inference_mode()
    def test_residual_quant_bytes_scales_and_qkv_projection(self):
        for tokens in self.tokens:
            for magnitude in (0.01, 1.0, 100.0):
                with self.subTest(tokens=tokens, magnitude=magnitude):
                    x = torch.randn(tokens, 2048, device="npu", dtype=torch.bfloat16) * magnitude
                    residual = torch.randn_like(x) * magnitude
                    self.compare(x, residual)

    @torch.inference_mode()
    def test_existing_gate_and_no_residual_fallbacks(self):
        x = torch.randn(32, 2048, device="npu", dtype=torch.bfloat16)
        residual = torch.randn_like(x)
        with norm_fusion(False):
            out, _ = self.norm(x, residual, quant_linear=self.consumer)
            self.assertIsInstance(out, torch.Tensor)
            self.assertEqual(out.dtype, torch.bfloat16)
        with norm_fusion(True):
            out = self.norm(x, quant_linear=self.consumer)
            self.assertIsInstance(out, torch.Tensor)
            self.assertEqual(out.dtype, torch.bfloat16)
            out, _ = self.norm(x, residual, quant_linear=None)
            self.assertIsInstance(out, torch.Tensor)
            self.assertEqual(out.dtype, torch.bfloat16)

    @torch.inference_mode()
    def test_nondefault_caller_stream(self):
        caller = torch.npu.Stream()
        caller.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(caller):
            x = torch.randn(1024, 2048, device="npu", dtype=torch.bfloat16)
            residual = torch.randn_like(x)
            self.compare(x, residual)
        caller.synchronize()


if __name__ == "__main__":
    unittest.main(verbosity=2)
