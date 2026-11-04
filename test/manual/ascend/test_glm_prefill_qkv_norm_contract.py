"""CPU-only checks for the local GLM prefill QKV norm-fusion candidate.

Set SGLANG_NORM_CANDIDATE_SOURCE to a source tree or the local after/ folder.
No torch, NPU, distributed process group, or model weights are imported.
"""

import ast
from contextlib import nullcontext
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(os.getenv("SGLANG_NORM_CANDIDATE_SOURCE", Path(__file__).resolve().parents[3]))
MODEL = "python/sglang/srt/models/glm4_moe_lite.py"
MHA = "python/sglang/srt/hardware_backend/npu/modules/deepseek_v2_attention_mla_npu.py"


def load(relative, function, namespace, class_name=None):
    path = ROOT / relative
    if not path.exists():
        path = ROOT / Path(relative).name
    nodes = ast.parse(path.read_text()).body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = deepcopy(next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == function))
    node.decorator_list = []
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function]


class Tensor:
    def __init__(self, rows=8, dtype="bf16", dims=2):
        self.shape = (rows, 2048)
        self.dtype, self.dims = dtype, dims

    def dim(self):
        return self.dims

    def split(self, *args, **kwargs):
        return Tensor(), Tensor()

    def view(self, *args):
        return self

    def reshape(self, *args):
        return self

    def unsqueeze(self, *args):
        return self

    def contiguous(self):
        return self

    def to(self, *args):
        return self

    def __getitem__(self, key):
        return self

    def __setitem__(self, key, value):
        pass


class TestPrefillNormContract(unittest.TestCase):
    def fixture(self):
        self.mode = SimpleNamespace(EXTEND="extend")
        self.batch = SimpleNamespace(forward_mode="extend", spec_info=None, can_run_tbo=False)
        self.parallel = SimpleNamespace(**dict.fromkeys(("world_size", "tp_size", "attn_tp_size", "attn_dp_size", "attn_cp_size", "attn_dcp_size"), 1))
        self.disagg = SimpleNamespace(disaggregation_mode="prefill")
        self.spec = SimpleNamespace(speculative_algorithm=None)
        self.attn_context = SimpleNamespace(input_scattered=False, clear_attn_inputs=lambda: None)
        self.graph = SimpleNamespace(cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(backend="disabled")))
        self.dispatch = "mha_npu"
        self.attn = SimpleNamespace(use_dsa=False, q_lora_rank=768, fused_qkv_a_proj_with_mqa=SimpleNamespace(_sgl_npu_mx_quant_fusable=True), dispatch_attn_forward_method=lambda batch: self.dispatch)
        self.layer = SimpleNamespace(self_attn=self.attn)
        self.compiling, self.capture, self.dp_enabled = False, False, False
        self.ns = {
            "_NPU_GLM_PREFILL_QKV_NORM_FUSION": True,
            "torch": SimpleNamespace(Tensor=Tensor, bfloat16="bf16", compiler=SimpleNamespace(is_compiling=lambda: self.compiling)),
            "ForwardMode": self.mode,
            "Backend": SimpleNamespace(DISABLED="disabled"),
            "AttnForwardMethod": SimpleNamespace(MHA_NPU="mha_npu"),
            "get_disagg": lambda: self.disagg,
            "get_spec": lambda: self.spec,
            "get_parallel": lambda: self.parallel,
            "get_exec": lambda: SimpleNamespace(graph=self.graph),
            "get_attn_tp_context": lambda: self.attn_context,
            "get_is_capture_mode": lambda: self.capture,
            "is_dp_attention_enabled": lambda: self.dp_enabled,
        }
        self.consumer = load(MODEL, "_glm_prefill_qkv_norm_consumer", self.ns)

    def setUp(self):
        self.fixture()

    def candidate(self, hidden=None, residual=None):
        return self.consumer(self.layer, hidden if hidden is not None else Tensor(), residual if residual is not None else Tensor(), self.batch)

    def test_gate_default_and_eligible_consumer(self):
        self.ns["_NPU_GLM_PREFILL_QKV_NORM_FUSION"] = False
        # Gate-off must not consult initialized distributed/NPU runtime state.
        self.assertIsNone(self.consumer(None, None, None, None))
        self.ns["_NPU_GLM_PREFILL_QKV_NORM_FUSION"] = True
        self.assertIs(self.candidate(), self.attn.fused_qkv_a_proj_with_mqa)

    def test_decode_and_every_nonordinary_extend_mode_fall_back(self):
        for mode in ("decode", "idle", "target_verify", "draft_extend_v2", "mixed", "split_prefill", "dllm_extend", "prebuilt"):
            with self.subTest(mode=mode):
                self.batch.forward_mode = mode
                self.assertIsNone(self.candidate())

    def test_input_shape_dtype_empty_and_first_layer_fall_back(self):
        self.assertIsNone(self.consumer(self.layer, Tensor(), None, self.batch))
        for hidden in (Tensor(rows=0), Tensor(dtype="fp16"), Tensor(dims=3), (Tensor(dtype="fp8"), Tensor(dtype="u8"))):
            with self.subTest(hidden=hidden):
                self.assertIsNone(self.consumer(self.layer, hidden, Tensor(), self.batch))

    def test_each_parallel_width_and_input_scatter_fall_back(self):
        for width in vars(self.parallel):
            with self.subTest(width=width):
                setattr(self.parallel, width, 2)
                self.assertIsNone(self.candidate())
                setattr(self.parallel, width, 1)
        self.dp_enabled = True
        self.assertIsNone(self.candidate())
        self.dp_enabled = False
        self.attn_context.input_scattered = True
        self.assertIsNone(self.candidate())

    def test_graph_compile_tbo_and_speculation_fall_back(self):
        for flag in ("compiling", "capture"):
            setattr(self, flag, True)
            self.assertIsNone(self.candidate())
            setattr(self, flag, False)
        self.batch.can_run_tbo = True
        self.assertIsNone(self.candidate())
        self.batch.can_run_tbo = False
        self.batch.spec_info = object()
        self.assertIsNone(self.candidate())
        self.batch.spec_info = None
        self.spec.speculative_algorithm = "EAGLE"
        self.assertIsNone(self.candidate())
        self.spec.speculative_algorithm = None
        for backend in ("tc_piecewise", "full", "breakable"):
            self.graph.cuda_graph_config.prefill.backend = backend
            self.assertIsNone(self.candidate())

    def test_attention_and_role_fallbacks(self):
        self.disagg.disaggregation_mode = "decode"
        self.assertIsNone(self.candidate())
        self.disagg.disaggregation_mode = "prefill"
        for field, value in (("use_dsa", True), ("q_lora_rank", None), ("q_a_proj", object())):
            with self.subTest(field=field):
                old = getattr(self.attn, field, None)
                setattr(self.attn, field, value)
                self.assertIsNone(self.candidate())
                if field == "q_a_proj":
                    delattr(self.attn, field)
                else:
                    setattr(self.attn, field, old)
        self.attn.fused_qkv_a_proj_with_mqa._sgl_npu_mx_quant_fusable = False
        self.assertIsNone(self.candidate())
        self.attn.fused_qkv_a_proj_with_mqa._sgl_npu_mx_quant_fusable = True
        self.dispatch = "mla_npu"
        self.assertIsNone(self.candidate())

    def test_qkv_tuple_ends_at_attention_and_moe_keeps_bf16(self):
        events = []

        def pre_attn(hidden, residual, batch, quant_format, quant_linear=None):
            self.assertIs(quant_linear, self.attn.fused_qkv_a_proj_with_mqa)
            events.append("input_norm_tuple")
            return (Tensor(dtype="fp8"), Tensor(dtype="u8")), residual

        class Attention:
            use_dsa = False
            q_lora_rank = 768
            fused_qkv_a_proj_with_mqa = self.attn.fused_qkv_a_proj_with_mqa
            dispatch_attn_forward_method = lambda _, batch: "mha_npu"

            def __call__(_, **kwargs):
                self.assertIsInstance(kwargs["hidden_states"], tuple)
                events.append("attention_consumed_tuple")
                return Tensor()

        def post_attn(hidden, residual, batch):
            self.assertIsInstance(hidden, Tensor)
            self.assertEqual(hidden.dtype, "bf16")
            events.append("post_attention_norm_bf16")
            return hidden, residual

        def mlp(hidden, batch):
            self.assertIsInstance(hidden, Tensor)
            self.assertEqual(hidden.dtype, "bf16")
            events.append("router_and_shared_overlap_bf16")
            return hidden

        self.layer.self_attn = Attention()
        self.layer.layer_scatter_modes = None
        self.layer.layer_communicator = SimpleNamespace(
            prepare_attn=pre_attn, prepare_mlp=post_attn,
            should_fuse_mlp_allreduce_with_next_layer=lambda batch: False,
            should_use_reduce_scatter=lambda batch: False,
            postprocess_layer=lambda hidden, residual, batch: (hidden, residual),
        )
        self.layer.mlp = mlp
        self.ns["_glm_prefill_qkv_norm_consumer"] = self.consumer
        self.ns["get_forward"] = lambda: SimpleNamespace(scoped=lambda **kwargs: nullcontext())
        forward = load(MODEL, "forward", self.ns, "Glm4MoeLiteDecoderLayer")
        output, _ = forward(self.layer, Tensor(), Tensor(), self.batch, Tensor(), None)
        self.assertEqual(output.dtype, "bf16")
        self.assertEqual(events, ["input_norm_tuple", "attention_consumed_tuple", "post_attention_norm_bf16", "router_and_shared_overlap_bf16"])

    def test_mha_rope_uses_projected_bf16_dtype_not_fp8_tuple(self):
        requested = []
        m = SimpleNamespace(
            q_lora_rank=768, kv_lora_rank=512, qk_rope_head_dim=64,
            qk_nope_head_dim=192, qk_head_dim=256, v_head_dim=256,
            num_local_heads=20, layer_id=0, use_dsa=False, use_deepseek_yarn_rope=True,
            q_a_layernorm=lambda x: x, q_b_proj=lambda x: (Tensor(), None),
            kv_b_proj=lambda x: (Tensor(), None),
            kv_a_layernorm=SimpleNamespace(weight=Tensor(), variance_epsilon=1e-5),
            rotary_emb=SimpleNamespace(get_cos_sin_cache=lambda positions, dtype, offsets: (requested.append(dtype) or Tensor(), Tensor())),
            _concat_and_cast_mha_k=lambda *args: Tensor(),
        )
        ns = {
            "_use_ag_after_qlora": False,
            "torch": SimpleNamespace(Tensor=Tensor, int64="i64"),
            "get_attn_tp_context": lambda: SimpleNamespace(fetch_qkv_latent=lambda: Tensor()),
            "get_token_to_kv_pool": lambda: SimpleNamespace(get_kv_buffer=lambda layer: (Tensor(), Tensor())),
            "torch_npu": SimpleNamespace(npu_interleave_rope=lambda x, *args: x, npu_kv_rmsnorm_rope_cache=lambda *args, **kwargs: (None, None, Tensor(), Tensor())),
            "is_fia_nz": lambda: False,
        }
        prepare = load(MHA, "forward_mha_prepare_npu", ns)
        batch = SimpleNamespace(out_cache_loc=Tensor())
        q, _, _, _ = prepare(m, Tensor(), (Tensor(dtype="fp8"), Tensor(dtype="u8")), batch, None, None)
        self.assertEqual(requested, ["bf16"])
        self.assertEqual(q.dtype, "bf16")


if __name__ == "__main__":
    unittest.main(verbosity=2)
