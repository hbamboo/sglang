"""CPU-only qualification and scale-view contracts for opt-in routing."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.moe import init_routing as routing


class TensorSpec:
    def __init__(self, shape, dtype, device, contiguous=True):
        self.shape, self.dtype, self.device = shape, dtype, device
        self.ndim = len(shape)
        self.contiguous = contiguous

    def is_contiguous(self):
        return self.contiguous


class TestPrequantRoutingContract(unittest.TestCase):
    def specs(self, tokens=8192, hidden=2048, dtype=torch.bfloat16, idtype=torch.int32, contiguous=True):
        device = SimpleNamespace(type='npu')
        return (TensorSpec((tokens, hidden), dtype, device, contiguous),
                TensorSpec((tokens, 4), idtype, device))

    def test_default_off_short_circuits_runtime_and_tensor_queries(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('SGLANG_NPU_MOE_PREQUANT_ROUTING', None)
            with patch.object(routing, 'get_disagg', side_effect=AssertionError('runtime queried')):
                self.assertFalse(routing._can_prequantize_mxfp8(object(), object(), 64, 4, 3))

    def test_positive_boundaries_and_exclusions(self):
        with patch.dict(os.environ, {'SGLANG_NPU_MOE_PREQUANT_ROUTING': '1'}):
            with patch.object(routing, 'get_disagg', return_value=SimpleNamespace(disaggregation_mode='prefill')):
                for tokens in (8192, 17226, 32768, 36000):
                    self.assertTrue(routing._can_prequantize_mxfp8(*self.specs(tokens), 64, 4, 3))
                for tokens in (1, 128, 2048, 8191, 36001):
                    self.assertFalse(routing._can_prequantize_mxfp8(*self.specs(tokens), 64, 4, 3))
                for kwargs in ({'hidden': 4096}, {'dtype': torch.float16},
                               {'idtype': torch.int64}, {'contiguous': False}):
                    self.assertFalse(routing._can_prequantize_mxfp8(*self.specs(**kwargs), 64, 4, 3))
                for experts, topk, mode in ((32, 4, 3), (64, 8, 3), (64, 4, -1), (64, 4, 1)):
                    self.assertFalse(routing._can_prequantize_mxfp8(*self.specs(), experts, topk, mode))
            for role in ('decode', 'null'):
                with patch.object(routing, 'get_disagg', return_value=SimpleNamespace(disaggregation_mode=role)):
                    self.assertFalse(routing._can_prequantize_mxfp8(*self.specs(), 64, 4, 3))

    def test_scale_normalization_preserves_storage_and_bytes(self):
        flat = torch.arange(256, dtype=torch.uint8).reshape(4, 64)
        result = routing._normalize_mxfp_scale(flat)
        self.assertEqual(result.shape, (4, 32, 2))
        self.assertEqual(result.data_ptr(), flat.data_ptr())
        self.assertTrue(torch.equal(result.reshape(4, 64), flat))
        self.assertIs(routing._normalize_mxfp_scale(result), result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
