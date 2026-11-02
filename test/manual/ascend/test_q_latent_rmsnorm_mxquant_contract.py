"""CPU-only behavioral contract tests for the Q-latent RMSNorm+MXFP8 fusion.

Runs in the isolated worktree with CPU torch and no NPU.  The real framework
``forward_mha_prepare_npu`` function and the real ``NPUMXFP8LinearMethod``
consumer are exercised; only the Ascend kernel entry points are substituted.

    PYTHONPATH=<worktree>/python python test/manual/ascend/
        test_q_latent_rmsnorm_mxquant_contract.py -v
"""

import os
import sys
import unittest
from types import SimpleNamespace

import torch

# Import order matters for the worktree's modelslim quantization package: the
# package __init__ must run before the hardware kernel module to avoid its
# pre-existing circular import.
import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    NPUMXFP8LinearMethod,
)
from sglang.srt.hardware_backend.npu.modules import q_latent_rmsnorm_mxquant as qlm
from sglang.srt.hardware_backend.npu.modules import deepseek_v2_attention_mla_npu as mha

TREE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if TREE not in sys.path:
    sys.path.insert(0, TREE)


class FakeFusedOp:
    """Stand-in for torch_npu.npu_rms_norm_dynamic_mx_quant."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, gamma, **kwargs):
        self.calls.append({"x": x, "gamma": gamma, "kwargs": kwargs})
        rows, width = x.shape
        quant = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.empty((rows, (width + 63) // 64, 2), dtype=torch.uint8)
        rstd = torch.empty((rows, 1), dtype=torch.float32)
        return quant, scale, rstd


class FakeQuantMethod:
    """Consumer contract: only the pre-quantised (values, scale) tuple."""

    def __init__(self, out_features, out_dtype=torch.bfloat16):
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.calls = []
        self.output = None

    def apply(self, layer, x, bias=None):
        if not isinstance(x, tuple):
            raise AssertionError(
                "MXFP8 consumer received a tensor: would call DynamicMxQuant"
            )
        if len(x) != 2:
            raise AssertionError(f"expected (values, scale), got {len(x)} tensors")
        self.calls.append({"layer": layer, "x": x, "bias": bias})
        rows = x[0].shape[0]
        self.output = torch.zeros((rows, self.out_features), dtype=self.out_dtype)
        return self.output


class FakeNorm:
    def __init__(self, width=768, eps=1e-6):
        self.has_weight = True
        self.weight = torch.ones(width, dtype=torch.bfloat16)
        self.variance_epsilon = eps
        self.variance_size_override = None
        self.cast_x_before_out_mul = False
        self.call_count = 0

    def __call__(self, x):
        self.call_count += 1
        return torch.zeros_like(x)


class FakeProj:
    def __init__(self, out_features, in_features=768):
        self.out_features = out_features
        self.tp_size = 1
        self.gather_output = False
        self.skip_bias_add = False
        self.bias = None
        self._sgl_npu_mx_quant_fusable = True
        self.weight = torch.empty(
            (in_features, out_features), dtype=torch.float8_e4m3fn
        )
        self.weight_scale_inv = torch.empty((1, 1, 2), dtype=torch.uint8)
        self.quant_method = FakeQuantMethod(out_features)
        self.call_count = 0

    def __call__(self, x):
        self.call_count += 1
        return torch.zeros((x.shape[0], self.out_features), dtype=torch.bfloat16), None


def make_fakes(out_features=68, width=768):
    return FakeNorm(width), FakeProj(out_features)


class HelperContractTest(unittest.TestCase):
    def setUp(self):
        self._old_enabled = qlm._FUSION_ENABLED
        self._old_op = qlm._fused_rmsnorm_mxquant_op
        self._old_compile = qlm._compiler_is_compiling
        self._old_capture = qlm._device_is_capturing
        qlm._FUSION_ENABLED = True
        qlm._compiler_is_compiling = lambda: False
        qlm._device_is_capturing = lambda: False
        self.op = FakeFusedOp()
        qlm._fused_rmsnorm_mxquant_op = lambda: self.op

    def tearDown(self):
        qlm._FUSION_ENABLED = self._old_enabled
        qlm._fused_rmsnorm_mxquant_op = self._old_op
        qlm._compiler_is_compiling = self._old_compile
        qlm._device_is_capturing = self._old_capture

    def call(self, norm=None, proj=None, x=None, **kwargs):
        norm = norm if norm is not None else FakeNorm()
        proj = proj if proj is not None else FakeProj(68)
        if x is None:
            x = torch.zeros((4, 768), dtype=torch.bfloat16)
        return qlm.forward_q_latent_rmsnorm_mxquant(norm, proj, x, **kwargs)

    def test_flag_disabled_short_circuits_before_any_probe(self):
        qlm._FUSION_ENABLED = False

        def boom():
            raise AssertionError("probe must not run when the flag is off")

        norm, proj = make_fakes()
        out = self.call(
            norm=norm,
            proj=proj,
            x=torch.zeros((4, 768), dtype=torch.bfloat16),
            is_compiling=boom,
            is_capturing=boom,
        )
        self.assertIsNone(out)
        self.assertEqual(self.op.calls, [])
        self.assertEqual(proj.quant_method.calls, [])

    def test_compile_and_capture_gate_before_consumer_metadata(self):
        class ExplodingProj:
            def __getattr__(self, name):
                raise AssertionError(f"consumer metadata {name!r} queried too early")

        x = torch.zeros((4, 768), dtype=torch.bfloat16)
        capture_calls = []
        out = self.call(
            proj=ExplodingProj(),
            x=x,
            is_compiling=lambda: True,
            is_capturing=lambda: capture_calls.append(1) or True,
        )
        self.assertIsNone(out)
        # Compile is checked first, so capture is never consulted.
        self.assertEqual(capture_calls, [])
        out = self.call(
            proj=ExplodingProj(),
            x=x,
            is_compiling=lambda: False,
            is_capturing=lambda: True,
        )
        self.assertIsNone(out)
        self.assertEqual(self.op.calls, [])

    def test_unsupported_routes_fall_back(self):
        x2d = torch.zeros((4, 768), dtype=torch.bfloat16)
        cases = {}

        def case(name, mutate):
            norm, proj = make_fakes()
            x = x2d
            x = mutate(norm, proj, x)
            cases[name] = (norm, proj, x)

        case("fp16_input", lambda n, p, x: x.to(torch.float16))
        case("fp32_input", lambda n, p, x: x.to(torch.float32))
        case("3d_input", lambda n, p, x: x.unsqueeze(0))
        case("width_1024", lambda n, p, x: torch.zeros((4, 1024), dtype=torch.bfloat16))
        case("empty_input", lambda n, p, x: torch.zeros((0, 768), dtype=torch.bfloat16))
        case("norm_no_weight", lambda n, p, x: setattr(n, "has_weight", False) or x)
        case(
            "norm_weight_fp16",
            lambda n, p, x: setattr(
                n, "weight", torch.ones(768, dtype=torch.float16)
            )
            or x,
        )
        case(
            "variance_override",
            lambda n, p, x: setattr(n, "variance_size_override", 512) or x,
        )
        case(
            "hf_cast_semantics",
            lambda n, p, x: setattr(n, "cast_x_before_out_mul", True) or x,
        )
        case(
            "forced_native_forward",
            lambda n, p, x: setattr(n, "_forward_method", lambda *a, **k: None) or x,
        )
        case(
            "compile_mode",
            lambda n, p, x: setattr(n, "is_torch_compile", True) or x,
        )
        case("norm_forward_hook", lambda n, p, x: setattr(n, "_forward_hooks", {1: 0}) or x)
        case("proj_tp", lambda n, p, x: setattr(p, "tp_size", 2) or x)
        case(
            "proj_gather_output",
            lambda n, p, x: setattr(p, "gather_output", True) or x,
        )
        case(
            "proj_forward_hook",
            lambda n, p, x: setattr(p, "_forward_hooks", {1: 0}) or x,
        )
        case(
            "proj_compiled",
            lambda n, p, x: setattr(p, "_compiled_call_impl", object()) or x,
        )
        case(
            "not_fusable",
            lambda n, p, x: setattr(p, "_sgl_npu_mx_quant_fusable", False) or x,
        )
        case(
            "no_quant_apply",
            lambda n, p, x: setattr(p, "quant_method", SimpleNamespace()) or x,
        )
        case(
            "weight_not_fp8",
            lambda n, p, x: setattr(
                p, "weight", torch.empty((1, 1), dtype=torch.bfloat16)
            )
            or x,
        )
        case(
            "proj_width_mismatch",
            lambda n, p, x: setattr(
                p, "weight", torch.empty((512, 68), dtype=torch.float8_e4m3fn)
            )
            or x,
        )
        case(
            "missing_weight_scale_inv",
            lambda n, p, x: setattr(p, "weight_scale_inv", None) or x,
        )

        for name, (norm, proj, x) in cases.items():
            with self.subTest(name=name):
                self.assertIsNone(self.call(norm=norm, proj=proj, x=x))
                self.assertEqual(self.op.calls, [])
                self.assertEqual(getattr(proj.quant_method, "calls", []), [])

    def test_supported_route_passes_original_gamma_epsilon_and_tuple(self):
        norm = FakeNorm(eps=1.25e-5)
        proj = FakeProj(68)
        # Original qkv stride 1344, nonzero storage offset, q-latent slice.
        base = torch.randn((5, 1400), dtype=torch.bfloat16)
        qkv = base[:, 17 : 17 + 1344]
        x = qkv[:, :768]
        self.assertFalse(x.is_contiguous())
        self.assertEqual(x.stride(0), 1400)
        self.assertNotEqual(x.storage_offset(), 0)

        out = self.call(norm=norm, proj=proj, x=x)

        self.assertEqual(len(self.op.calls), 1)
        call = self.op.calls[0]
        self.assertEqual(call["gamma"].data_ptr(), norm.weight.data_ptr())
        self.assertTrue(torch.equal(call["gamma"], norm.weight.data))
        self.assertEqual(call["kwargs"]["epsilon"], 1.25e-5)
        self.assertIsNone(call["kwargs"]["beta"])
        self.assertEqual(call["kwargs"]["scale_alg"], 0)
        self.assertEqual(call["kwargs"]["round_mode"], "rint")
        self.assertEqual(call["kwargs"]["dst_type"], torch.float8_e4m3fn)
        self.assertNotEqual(call["kwargs"]["dst_type"], 296)  # 296 is FP4 here
        # Exactly one materialisation of the strided slice.
        self.assertTrue(call["x"].is_contiguous())
        self.assertTrue(torch.equal(call["x"], x))

        self.assertEqual(len(proj.quant_method.calls), 1)
        (quant, scale) = proj.quant_method.calls[0]["x"]
        self.assertEqual(quant.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.dtype, torch.uint8)
        self.assertEqual(tuple(scale.shape), (5, 12, 2))
        self.assertIs(proj.quant_method.calls[0]["bias"], None)
        self.assertIs(out, proj.quant_method.output)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(tuple(out.shape), (5, 68))

    def test_resolved_plain_npu_forward_is_still_eligible(self):
        # BaseFusedOp caches _forward_method after first use; the plain NPU
        # forward is the same residual=None path the fused kernel replaces.
        def forward_npu(*args, **kwargs):
            raise AssertionError("original forward must not be called")

        norm, proj = make_fakes()
        norm._forward_method = forward_npu
        x = torch.zeros((4, 768), dtype=torch.bfloat16)
        self.assertIsNotNone(self.call(norm=norm, proj=proj, x=x))

    def test_token_counts_route_and_empty_falls_back(self):
        for rows in (1, 31, 127, 128, 129):
            with self.subTest(rows=rows):
                norm, proj = make_fakes()
                x = torch.zeros((rows, 768), dtype=torch.bfloat16)
                out = self.call(norm=norm, proj=proj, x=x)
                self.assertIsNotNone(out)
                self.assertEqual(tuple(out.shape), (rows, 68))
                self.assertEqual(tuple(self.op.calls[-1]["x"].shape), (rows, 768))
        norm, proj = make_fakes()
        x = torch.zeros((0, 768), dtype=torch.bfloat16)
        self.assertIsNone(self.call(norm=norm, proj=proj, x=x))
        self.assertEqual(
            qlm.fusion_blocker(
                norm, proj, x, is_compiling=lambda: False, is_capturing=lambda: False
            ),
            "empty_input",
        )

    def test_contiguous_input_is_not_copied_again(self):
        norm, proj = make_fakes()
        x = torch.zeros((4, 768), dtype=torch.bfloat16)
        out = self.call(norm=norm, proj=proj, x=x)
        self.assertIsNotNone(out)
        self.assertIs(self.op.calls[0]["x"], x)


class RealConsumerContractTest(unittest.TestCase):
    """The real NPUMXFP8LinearMethod tuple path must not quantise twice."""

    def setUp(self):
        self._orig_matmul = torch.ops.npu.npu_quant_matmul
        self._orig_dyn = torch.ops.npu.npu_dynamic_mx_quant

    def tearDown(self):
        torch.ops.npu.npu_quant_matmul = self._orig_matmul
        torch.ops.npu.npu_dynamic_mx_quant = self._orig_dyn

    def layer(self, rows, out_features=68):
        return SimpleNamespace(
            weight=torch.empty((768, out_features), dtype=torch.float8_e4m3fn),
            weight_scale_inv=torch.empty((12, out_features, 2), dtype=torch.uint8),
            bias=None,
            bias_fp32=None,
        )

    def test_tuple_input_skips_dynamic_mx_quant_and_returns_bf16(self):
        rows = 5
        quant = torch.empty((rows, 768), dtype=torch.float8_e4m3fn)
        scale = torch.empty((rows, 12, 2), dtype=torch.uint8)
        recorded = {}

        def fake_matmul(x1, x2, scale_arg, **kwargs):
            recorded["x1"] = x1
            recorded["x2"] = x2
            recorded["scale"] = scale_arg
            recorded["kwargs"] = kwargs
            return torch.zeros((rows, x2.shape[1]), dtype=torch.bfloat16)

        def forbid_dynamic(*args, **kwargs):
            raise AssertionError("tuple path called npu_dynamic_mx_quant again")

        torch.ops.npu.npu_quant_matmul = fake_matmul
        torch.ops.npu.npu_dynamic_mx_quant = forbid_dynamic
        layer = self.layer(rows)
        out = NPUMXFP8LinearMethod().apply(layer, (quant, scale), None)
        self.assertEqual(tuple(out.shape), (rows, 68))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertIs(recorded["x1"], quant)
        self.assertIs(recorded["scale"], layer.weight_scale_inv)
        self.assertIs(recorded["kwargs"]["pertoken_scale"], scale)
        self.assertEqual(recorded["kwargs"]["output_dtype"], torch.bfloat16)
        self.assertEqual(recorded["kwargs"]["group_sizes"], [1, 1, 32])

    def test_unquantised_tensor_still_uses_dynamic_mx_quant(self):
        rows = 5
        quant = torch.empty((rows, 768), dtype=torch.float8_e4m3fn)
        scale = torch.empty((rows, 12, 2), dtype=torch.uint8)
        calls = []

        def fake_dynamic(x, **kwargs):
            calls.append((x, kwargs))
            return quant, scale

        torch.ops.npu.npu_dynamic_mx_quant = fake_dynamic
        torch.ops.npu.npu_quant_matmul = lambda x1, x2, s, **kw: torch.zeros(
            (rows, x2.shape[1]), dtype=torch.bfloat16
        )
        out = NPUMXFP8LinearMethod().apply(
            self.layer(rows), torch.zeros((rows, 768), dtype=torch.bfloat16), None
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["dst_type"], torch.float8_e4m3fn)
        self.assertEqual(out.dtype, torch.bfloat16)


class PrepareRouteContractTest(unittest.TestCase):
    """Exercise the real forward_mha_prepare_npu body with CPU tensors."""

    def setUp(self):
        self._old = {
            "get_attn_tp_context": mha.get_attn_tp_context,
            "get_token_to_kv_pool": mha.get_token_to_kv_pool,
            "can_fuse_expanded_kv": mha.can_fuse_expanded_kv,
            "scattered_to_tp_attn_full": mha.scattered_to_tp_attn_full,
            "is_fia_nz": mha.is_fia_nz,
            "torch_npu": mha.torch_npu,
            "use_ag": mha._use_ag_after_qlora,
        }
        self._old_qlm = {
            "enabled": qlm._FUSION_ENABLED,
            "op": qlm._fused_rmsnorm_mxquant_op,
            "compile": qlm._compiler_is_compiling,
            "capture": qlm._device_is_capturing,
        }
        qlm._FUSION_ENABLED = True
        qlm._compiler_is_compiling = lambda: False
        qlm._device_is_capturing = lambda: False
        self.op = FakeFusedOp()
        qlm._fused_rmsnorm_mxquant_op = lambda: self.op
        mha._use_ag_after_qlora = False

    def tearDown(self):
        mha.get_attn_tp_context = self._old["get_attn_tp_context"]
        mha.get_token_to_kv_pool = self._old["get_token_to_kv_pool"]
        mha.can_fuse_expanded_kv = self._old["can_fuse_expanded_kv"]
        mha.scattered_to_tp_attn_full = self._old["scattered_to_tp_attn_full"]
        mha.is_fia_nz = self._old["is_fia_nz"]
        mha.torch_npu = self._old["torch_npu"]
        mha._use_ag_after_qlora = self._old["use_ag"]
        qlm._FUSION_ENABLED = self._old_qlm["enabled"]
        qlm._fused_rmsnorm_mxquant_op = self._old_qlm["op"]
        qlm._compiler_is_compiling = self._old_qlm["compile"]
        qlm._device_is_capturing = self._old_qlm["capture"]

    def fixture(
        self,
        rows=3,
        use_dsa=False,
        use_yarn=False,
        q_lora_rank=768,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=4,
        v_head_dim=6,
    ):
        self.rows = rows
        self.quant_method = FakeQuantMethod(qk_nope_head_dim + qk_rope_head_dim)
        self.proj = FakeProj(qk_nope_head_dim + qk_rope_head_dim)
        self.proj.quant_method = self.quant_method
        self.norm = FakeNorm(q_lora_rank)
        self.qkv = torch.randn(
            (rows, q_lora_rank + kv_lora_rank + qk_rope_head_dim),
            dtype=torch.bfloat16,
        )
        self.kv_a_layernorm = FakeNorm(kv_lora_rank)
        self.kv_b_proj_calls = []

        def kv_b_proj(x):
            self.kv_b_proj_calls.append(x)
            return torch.zeros(
                (x.shape[0], qk_nope_head_dim + v_head_dim), dtype=torch.bfloat16
            ), None

        class Pool:
            def __init__(self):
                self.set_calls = []
                self.get_calls = []

            def set_kv_buffer(self, *args):
                self.set_calls.append(args)

            def get_kv_buffer(self, layer_id):
                self.get_calls.append(layer_id)
                return (
                    torch.empty((rows, 1, kv_lora_rank), dtype=torch.bfloat16),
                    torch.empty((rows, 1, qk_rope_head_dim), dtype=torch.bfloat16),
                )

        self.pool = Pool()
        mha.get_attn_tp_context = lambda: SimpleNamespace(
            fetch_qkv_latent=lambda: self.qkv
        )
        mha.get_token_to_kv_pool = lambda: self.pool
        mha.can_fuse_expanded_kv = lambda *args: False
        mha.scattered_to_tp_attn_full = lambda t, batch: t
        mha.is_fia_nz = lambda: False

        cos_sin_dtypes = []
        self.cos_sin_dtypes = cos_sin_dtypes

        class Rotary:
            def get_cos_sin_cache(self, positions, dtype, offsets=None):
                cos_sin_dtypes.append(dtype)
                return (
                    torch.empty((positions.numel(), 8)),
                    torch.empty((positions.numel(), 8)),
                )

        self.m = SimpleNamespace(
            q_lora_rank=q_lora_rank,
            use_dsa=use_dsa,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            num_local_heads=1,
            qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
            q_a_layernorm=self.norm,
            q_b_proj=self.proj,
            kv_a_layernorm=self.kv_a_layernorm,
            kv_b_proj=kv_b_proj,
            use_deepseek_yarn_rope=use_yarn,
            rotary_emb=Rotary() if use_yarn else None,
            layer_id=0,
            _concat_and_cast_mha_k=lambda k_nope, k_pe, batch: k_nope,
        )
        if use_dsa:
            self.indexer_calls = []

            def indexer(**kwargs):
                self.indexer_calls.append(kwargs)

            self.m.indexer = indexer
        self.batch = SimpleNamespace(out_cache_loc=torch.arange(rows))

        if use_yarn:
            rope_calls = []
            self.rope_calls = rope_calls

            def fused_rope(x, gamma, cos, sin, *args, **kwargs):
                rope_calls.append({"x": x, "gamma": gamma, "kwargs": kwargs})
                k_pe = torch.zeros((rows, qk_rope_head_dim), dtype=torch.bfloat16)
                kv_a = torch.zeros((rows, 1, kv_lora_rank), dtype=torch.bfloat16)
                return None, None, k_pe, kv_a

            mha.torch_npu = SimpleNamespace(
                npu_interleave_rope=lambda x, cos, sin: x,
                npu_kv_rmsnorm_rope_cache=fused_rope,
            )
        self.scatter_modes = SimpleNamespace(layer_input_mode=None, attn_mode=None)

    def run_prepare(self, **kwargs):
        self.fixture(**kwargs)
        result = mha.forward_mha_prepare_npu(
            self.m,
            torch.arange(self.rows),
            torch.zeros((self.rows, 8), dtype=torch.bfloat16),
            self.batch,
            None,
            self.scatter_modes,
        )
        return result

    def test_opt_in_route_hits_fused_kernel_and_preserves_slice(self):
        q, _, _, _ = self.run_prepare()

        self.assertEqual(len(self.op.calls), 1)
        call = self.op.calls[0]
        # The preceding split is preserved: same values as qkv[..., :768].
        self.assertTrue(call["x"].is_contiguous())
        self.assertTrue(torch.equal(call["x"], self.qkv[:, :768]))
        self.assertEqual(call["gamma"].data_ptr(), self.norm.weight.data_ptr())
        self.assertEqual(call["kwargs"]["epsilon"], self.norm.variance_epsilon)

        # Exactly one consumer invocation, on the pre-quantised tuple.
        self.assertEqual(len(self.quant_method.calls), 1)
        quant, scale = self.quant_method.calls[0]["x"]
        self.assertEqual(quant.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(scale.shape), (self.rows, 12, 2))
        self.assertEqual(self.proj.call_count, 0)  # no second quant path
        self.assertEqual(self.norm.call_count, 0)  # norm fused inside

        # BF16 projection output reaches the rest of the prepare function.
        self.assertEqual(q.dtype, torch.bfloat16)
        self.assertEqual(tuple(q.shape), (self.rows, 1, 68))
        self.assertEqual(len(self.pool.set_calls), 1)

    def test_rope_and_cache_receive_bf16_not_fp8(self):
        self.run_prepare(use_yarn=True)
        self.assertEqual(self.cos_sin_dtypes, [torch.bfloat16])
        self.assertEqual(len(self.rope_calls), 1)
        self.assertEqual(self.rope_calls[0]["gamma"].dtype, torch.bfloat16)

    def test_ag_after_qlora_falls_back_to_original_bf16_norm(self):
        mha._use_ag_after_qlora = True
        self.fixture()
        self.scatter_modes = SimpleNamespace(
            layer_input_mode=mha.ScatterMode.SCATTERED,
            attn_mode=mha.ScatterMode.TP_ATTN_FULL,
        )
        mha.forward_mha_prepare_npu(
            self.m,
            torch.arange(self.rows),
            torch.zeros((self.rows, 8), dtype=torch.bfloat16),
            self.batch,
            None,
            self.scatter_modes,
        )
        self.assertEqual(self.op.calls, [])
        self.assertEqual(self.quant_method.calls, [])
        self.assertEqual(self.norm.call_count, 1)
        self.assertEqual(self.proj.call_count, 1)

    def test_dsa_keeps_bf16_norm_output_for_indexer(self):
        self.run_prepare(use_dsa=True)
        self.assertEqual(self.op.calls, [])
        self.assertEqual(self.quant_method.calls, [])
        self.assertEqual(self.norm.call_count, 1)
        self.assertEqual(len(self.indexer_calls), 1)
        q_lora = self.indexer_calls[0]["q_lora"]
        self.assertEqual(q_lora.dtype, torch.bfloat16)

    def test_disabled_flag_falls_back_in_real_route(self):
        qlm._FUSION_ENABLED = False
        self.run_prepare()
        self.assertEqual(self.op.calls, [])
        self.assertEqual(self.norm.call_count, 1)
        self.assertEqual(self.proj.call_count, 1)

    def test_compile_and_capture_fall_back_in_real_route(self):
        qlm._compiler_is_compiling = lambda: True
        self.run_prepare()
        self.assertEqual(self.op.calls, [])
        self.assertEqual(self.norm.call_count, 1)
        qlm._compiler_is_compiling = lambda: False
        qlm._device_is_capturing = lambda: True
        self.run_prepare()
        self.assertEqual(self.op.calls, [])
        self.assertEqual(self.norm.call_count, 1)

    def test_unsupported_width_falls_back_in_real_route(self):
        self.run_prepare(q_lora_rank=512)
        self.assertEqual(self.op.calls, [])
        self.assertEqual(self.norm.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
