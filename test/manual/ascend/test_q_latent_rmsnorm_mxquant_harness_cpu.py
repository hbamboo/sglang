"""CPU-only checks for the granted-NPU harness's pure acceptance/oracle logic.

No NPU, no torch_npu import, no sglang runtime: the harness module is imported
directly and only its pure helpers are exercised.

    PYTHONPATH=<worktree>/python python test/manual/ascend/
        test_q_latent_rmsnorm_mxquant_harness_cpu.py -v
"""

import importlib.util
import os
import sys
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS_PATH = os.path.join(HERE, "test_q_latent_rmsnorm_mxquant_npu.py")


def load_harness():
    spec = importlib.util.spec_from_file_location("qnorm_npu_harness", HARNESS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = load_harness()


class HarnessImportTest(unittest.TestCase):
    def test_import_does_not_load_runtime(self):
        # The harness must be importable for CPU self-tests without touching an
        # NPU runtime; _load_runtime is deliberately lazy.
        self.assertFalse(hasattr(H, "torch_npu"))
        self.assertFalse(hasattr(H, "qlm"))


class SaturatingCastTest(unittest.TestCase):
    def test_cpu_cast_is_not_saturating_but_oracle_is(self):
        vals = torch.tensor(
            [0.0, 448.0, 460.0, 512.0, 1000.0, -1000.0, float("nan")],
            dtype=torch.float32,
        )
        # Demonstrates why the oracle cannot use a bare CPU .to(): 512 -> NaN.
        bare = vals.to(torch.float8_e4m3fn).float()
        self.assertTrue(torch.isnan(bare[3]))
        self.assertTrue(torch.isnan(bare[4]))
        oracle = H.cast_e4m3_saturating(vals).float()
        self.assertEqual(float(oracle[0]), 0.0)
        self.assertEqual(float(oracle[1]), 448.0)
        self.assertEqual(float(oracle[2]), 448.0)
        self.assertEqual(float(oracle[3]), 448.0)
        self.assertEqual(float(oracle[4]), 448.0)
        self.assertEqual(float(oracle[5]), -448.0)
        self.assertTrue(torch.isnan(oracle[6]))


class OracleTest(unittest.TestCase):
    def test_e4m3_scale_alg0_rounding_and_scale_byte(self):
        # One block whose max is exactly 32 -> shared_exp = 5 - 8 = -3.
        norm = torch.arange(1, 33, dtype=torch.float32).reshape(1, 32)
        out = H.cpu_mx_quant_oracle(norm)
        self.assertEqual(int(out["scale_bytes"][0, 0]), 127 - 3)
        deq = out["dequant"][0]
        self.assertTrue(torch.isfinite(deq).all())
        self.assertEqual(out["saturation_count"], 0)
        # Dequant error is bounded by the E4M3 conversion step.
        self.assertLessEqual(float((deq - norm[0]).abs().max()), 2.0)

    def test_zero_block_uses_scale_byte_zero(self):
        out = H.cpu_mx_quant_oracle(torch.zeros((2, 64)))
        self.assertTrue(torch.equal(out["scale_bytes"], torch.zeros((2, 2), dtype=torch.uint8)))
        self.assertTrue(torch.equal(out["dequant"], torch.zeros((2, 64))))
        self.assertEqual(out["saturation_count"], 0)

    def test_saturation_count_and_clamped_dequant(self):
        norm = torch.zeros((1, 32), dtype=torch.float32)
        norm[0, 0] = 500.0  # amax 500 -> scale 1.0 -> 500 exceeds E4M3 max 448
        out = H.cpu_mx_quant_oracle(norm)
        self.assertEqual(int(out["scale_bytes"][0, 0]), 127)  # 2**0
        self.assertGreaterEqual(out["saturation_count"], 1)
        self.assertEqual(float(out["dequant"][0, 0]), 448.0)
        self.assertTrue(torch.isfinite(out["dequant"]).all())

    def test_bf16_rounded_oracle_differs_from_fp32_at_bf16_boundary(self):
        x = torch.tensor([[1.0] + [0.0] * 31], dtype=torch.bfloat16)
        gamma = torch.ones(32, dtype=torch.bfloat16)
        fp32 = H.cpu_rms_oracle(x, gamma, 1e-6)
        bf16 = H.round_to_bf16_fp32(fp32)
        self.assertTrue(torch.equal(bf16, fp32.to(torch.bfloat16).float()))
        self.assertFalse(torch.equal(bf16, fp32))


class AcceptanceStatsTest(unittest.TestCase):
    def test_match_stats_required_standard(self):
        ref = torch.randn((8, 16), dtype=torch.bfloat16)
        same = H.match_stats(ref, ref, 2.0**-6, 2.0**-6)
        self.assertTrue(H.bf16_standard_pass(same))
        self.assertEqual(same["matched_ratio"], 1.0)

        noisy = ref.float() * (1.0 + 2.0**-9)
        near = H.match_stats(ref, noisy, 2.0**-8, 2.0**-8)
        self.assertTrue(H.near_exact_pass(near))

        bad = ref.float() + 2.0
        stats = H.match_stats(ref, bad, 2.0**-6, 2.0**-6)
        self.assertFalse(H.bf16_standard_pass(stats))

    def test_match_stats_rejects_nonfinite(self):
        ref = torch.ones((2, 4))
        got = torch.ones((2, 4))
        got[0, 0] = float("nan")
        stats = H.match_stats(ref, got, 1e-6, 1e-6)
        self.assertEqual(stats["bad"], 1)
        self.assertFalse(H.bf16_standard_pass(stats))

    def test_quant_oracle_tolerance_accepts_one_ulp_and_rejects_gross_scale(self):
        oracle = torch.full((1, 64), 256.0)
        scale_bytes = torch.full((1, 2), 127, dtype=torch.uint8)  # scale 1
        # 256 -> 288 is one E4M3 ULP at this exponent (2**5 step is 32).
        one_ulp = torch.full((1, 64), 288.0)
        stats = H.quant_oracle_stats(oracle, one_ulp, scale_bytes)
        self.assertTrue(H.quant_oracle_pass(stats))
        gross = torch.full((1, 64), 1024.0)  # 4x: wrong scale/order
        stats = H.quant_oracle_stats(oracle, gross, scale_bytes)
        self.assertFalse(H.quant_oracle_pass(stats))

    def test_byte_match_counts_and_scales_bytes(self):
        a = torch.tensor([0, 1, 2, 3], dtype=torch.uint8)
        b = torch.tensor([0, 1, 9, 3], dtype=torch.uint8)
        m = H.byte_match(a, b)
        self.assertEqual(m["mismatch"], 1)
        self.assertAlmostEqual(m["matched_ratio"], 0.75)
        self.assertTrue(H.scale_old_new_pass(H.byte_match(a, a)))

    def test_criteria_are_the_declared_values(self):
        self.assertEqual(H.CRITERIA["bf16_downstream"]["atol"], 2.0**-6)
        self.assertEqual(H.CRITERIA["bf16_downstream"]["rtol"], 2.0**-6)
        self.assertGreaterEqual(
            H.CRITERIA["bf16_downstream"]["min_matched_ratio"], 0.99
        )
        self.assertEqual(H.CRITERIA["bf16_downstream"]["max_abs"], 1.0)
        self.assertEqual(H.CRITERIA["near_exact_diagnostic"]["atol"], 2.0**-8)


class StrideOffsetBuilderTest(unittest.TestCase):
    def test_stride1344_nonzero_offset(self):
        res = H.make_qkv(
            4, 1234, "cpu", layout="stride1344_offset", offset=64
        )
        self.assertTrue(res["layout_ok"])
        self.assertEqual(res["observed_stride"], [1344, 1])
        self.assertEqual(res["observed_offset"], 64)
        self.assertNotEqual(res["observed_offset"], 0)
        self.assertEqual(tuple(res["qkv"].shape), (4, 1344))

    def test_padded_rowstride_nonzero_offset(self):
        res = H.make_qkv(
            3, 7, "cpu", layout="padded_stride_offset", offset=32, padded=64
        )
        self.assertTrue(res["layout_ok"])
        self.assertEqual(res["observed_stride"], [1408, 1])
        self.assertEqual(res["observed_offset"], 32)
        self.assertEqual(tuple(res["qkv"].shape), (3, 1344))

    def test_zero_rows_keeps_layout_metadata(self):
        res = H.make_qkv(0, 7, "cpu", layout="stride1344_offset", offset=64)
        self.assertTrue(res["layout_ok"])
        self.assertEqual(res["observed_stride"], [1344, 1])
        self.assertEqual(res["observed_offset"], 64)
        self.assertEqual(tuple(res["qkv"].shape), (0, 1344))


if __name__ == "__main__":
    unittest.main(verbosity=2)
