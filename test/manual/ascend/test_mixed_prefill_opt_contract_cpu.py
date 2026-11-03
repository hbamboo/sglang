"""CPU-only contract tests for SGLANG_NPU_MIXED_PREFILL_OPT role guards.

No model/NPU import: the two real functions are extracted from the pinned
source with ``ast`` and executed against minimal mocks.

Covered:
  * QKV norm consumer: original PD prefill accepted with the flag OFF; mixed
    ``null`` rejected OFF / accepted ON; PD decode always rejected; capture,
    speculative and non-EXTEND batches still rejected with the flag ON.
  * MoE prequant routing: original PD prefill accepted; mixed null OFF rejected /
    ON accepted; decode always rejected; <=512 rows rejected even with the flag;
    8192..36000 row window, dtype, width, dtype/device/contiguity, top-k,
    expert count and quant-mode guards preserved.
  * each source reads the opt-in exactly once at module level.

Run: python3 test/manual/ascend/test_mixed_prefill_opt_contract_cpu.py
"""

import ast
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]


def load_functions(relpath, names):
    path = REPO / relpath
    src = path.read_text()
    tree = ast.parse(src)
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = ast.get_source_segment(src, node)
    missing = set(names) - set(found)
    assert not missing, (relpath, missing)
    return found, src


def run_defs(defs, ns):
    for name, text in defs.items():
        exec(compile(text, f"<extracted:{name}>", "exec"), ns)
    return ns


class Ctx:
    def __init__(self):
        self.mode = "prefill"
        self.extend = True
        self.opt = False
        self.capture = False
        self.compiling = False
        self.spec = None
        self.dp = False
        self.scattered = False


class FakeTensor:
    pass


class FakeHidden(FakeTensor):
    def __init__(self, rows=8192, width=2048, dtype="bf16", device="npu"):
        self.shape = (rows, width)
        self.ndim = 2
        self.dtype = dtype
        self.device = SimpleNamespace(type=device)

    def is_contiguous(self):
        return True


class FakeTopk:
    def __init__(self, rows, top_k=4, dtype="int32", device="npu", cont=True):
        self.shape = (rows, top_k)
        self.dtype = dtype
        self.device = SimpleNamespace(type=device)
        self._cont = cont

    def is_contiguous(self):
        return self._cont


def qkv_namespace(ctx):
    consumer = SimpleNamespace(_sgl_npu_mx_quant_fusable=True)
    attn = SimpleNamespace(
        fused_qkv_a_proj_with_mqa=consumer,
        use_dsa=False,
        q_lora_rank=8,
        dispatch_attn_forward_method=lambda fb: "MHA_NPU",
    )
    layer = SimpleNamespace(self_attn=attn)
    torch_stub = SimpleNamespace(
        Tensor=FakeTensor,
        bfloat16="bf16",
        compiler=SimpleNamespace(is_compiling=lambda: ctx.compiling),
    )
    forward_mode = "EXTEND" if ctx.extend else "DECODE"
    batch = SimpleNamespace(
        forward_mode=forward_mode, spec_info=ctx.spec, can_run_tbo=False
    )
    hidden = FakeHidden(rows=17)
    hidden.dim = lambda: 2
    ns = {
        "torch": torch_stub,
        "ForwardMode": SimpleNamespace(EXTEND="EXTEND"),
        "get_bool_env_var": lambda *a, **k: "0",
        "get_disagg": lambda: SimpleNamespace(disaggregation_mode=ctx.mode),
        "get_spec": lambda: SimpleNamespace(speculative_algorithm=ctx.spec),
        "get_is_capture_mode": lambda: ctx.capture,
        "is_dp_attention_enabled": lambda: ctx.dp,
        "get_attn_tp_context": lambda: SimpleNamespace(
            input_scattered=ctx.scattered
        ),
        "get_parallel": lambda: SimpleNamespace(
            world_size=1,
            tp_size=1,
            attn_tp_size=1,
            attn_dp_size=1,
            attn_cp_size=1,
            attn_dcp_size=1,
        ),
        "get_exec": lambda: SimpleNamespace(
            graph=SimpleNamespace(cuda_graph_config=None)
        ),
        "Backend": SimpleNamespace(DISABLED="disabled"),
        "AttnForwardMethod": SimpleNamespace(MHA_NPU="MHA_NPU"),
        "_NPU_GLM_PREFILL_QKV_NORM_FUSION": True,
        "_NPU_MIXED_PREFILL_OPT": ctx.opt,
    }
    return ns, layer, hidden, batch, consumer


def routing_namespace(ctx):
    torch_stub = SimpleNamespace(
        bfloat16="bf16",
        int32="int32",
        float8_e8m0fnu="e8m0",
        Tensor=FakeTensor,
    )
    ns = {
        "torch": torch_stub,
        "envs": SimpleNamespace(
            SGLANG_NPU_MOE_PREQUANT_ROUTING=SimpleNamespace(get=lambda: True)
        ),
        "get_disagg": lambda: SimpleNamespace(disaggregation_mode=ctx.mode),
        "get_bool_env_var": lambda *a, **k: "1" if ctx.opt else "0",
        "_NPU_MIXED_PREFILL_OPT": ctx.opt,
        "MXFP8_QUANT_MODE": 3,
        "Optional": object,
        "Tuple": object,
    }
    return ns


def qkv_accepts(ctx):
    defs, _ = load_functions(
        "python/sglang/srt/models/glm4_moe_lite.py",
        {"_glm_prefill_qkv_norm_consumer", "_prefill_or_mixed_opt_role"},
    )
    ns, layer, hidden, batch, consumer = qkv_namespace(ctx)
    run_defs(defs, ns)
    return ns["_glm_prefill_qkv_norm_consumer"](layer, hidden, object(), batch)


def routing_accepts(ctx, rows=8192, **overrides):
    defs, _ = load_functions(
        "python/sglang/srt/hardware_backend/npu/moe/init_routing.py",
        {"_can_prequantize_mxfp8", "_prefill_or_mixed_opt_role"},
    )
    ns = routing_namespace(ctx)
    run_defs(defs, ns)
    hidden = FakeHidden(rows=rows, **{
        k: v for k, v in overrides.items() if k in ("width", "dtype", "device")
    })
    topk = FakeTopk(
        rows,
        top_k=overrides.get("top_k_meta", 4),
        dtype=overrides.get("topk_dtype", "int32"),
        device=overrides.get("device", "npu"),
        cont=overrides.get("topk_cont", True),
    )
    return ns["_can_prequantize_mxfp8"](
        hidden,
        topk,
        overrides.get("num_experts", 64),
        overrides.get("top_k", 4),
        overrides.get("quant_mode", 3),
    )


def test_qkv_role_contract():
    c = Ctx(); c.mode = "prefill"; c.opt = False; c.extend = True
    assert qkv_accepts(c) is not None, "original PD prefill must stay enabled"

    c = Ctx(); c.mode = "null"; c.opt = False; c.extend = True
    assert qkv_accepts(c) is None, "mixed null must be rejected with flag OFF"

    c = Ctx(); c.mode = "null"; c.opt = True; c.extend = True
    assert qkv_accepts(c) is not None, "mixed null must be accepted with flag ON"

    c = Ctx(); c.mode = "decode"; c.opt = True; c.extend = True
    assert qkv_accepts(c) is None, "PD decode must stay rejected even with flag ON"

    c = Ctx(); c.mode = "null"; c.opt = True; c.extend = False
    assert qkv_accepts(c) is None, "non-EXTEND batch must stay rejected"

    c = Ctx(); c.mode = "null"; c.opt = True; c.extend = True; c.capture = True
    assert qkv_accepts(c) is None, "capture must stay rejected"

    c = Ctx(); c.mode = "null"; c.opt = True; c.extend = True; c.compiling = True
    assert qkv_accepts(c) is None, "torch.compile must stay rejected"

    c = Ctx(); c.mode = "null"; c.opt = True; c.extend = True
    c.spec = object()
    assert qkv_accepts(c) is None, "speculative batch must stay rejected"
    print("PASS QKV role/capture/spec/mode contract")


def test_routing_role_and_row_contract():
    c = Ctx(); c.mode = "prefill"; c.opt = False
    assert routing_accepts(c, rows=8192) is True, "original PD prefill stays enabled"

    c = Ctx(); c.mode = "null"; c.opt = False
    assert routing_accepts(c, rows=8192) is False, "mixed null rejected with flag OFF"

    c = Ctx(); c.mode = "null"; c.opt = True
    assert routing_accepts(c, rows=8192) is True, "mixed null accepted with flag ON"
    assert routing_accepts(c, rows=36000) is True

    c = Ctx(); c.mode = "decode"; c.opt = True
    assert routing_accepts(c, rows=8192) is False, "PD decode rejected even with flag"
    assert routing_accepts(c, rows=512) is False, "decode-sized <=512 rows rejected"
    assert routing_accepts(c, rows=8191) is False
    assert routing_accepts(c, rows=36001) is False

    c = Ctx(); c.mode = "null"; c.opt = True
    assert routing_accepts(c, dtype="fp32") is False
    assert routing_accepts(c, width=2047) is False
    assert routing_accepts(c, top_k=2) is False
    assert routing_accepts(c, num_experts=32) is False
    assert routing_accepts(c, quant_mode=2) is False
    assert routing_accepts(c, device="cpu") is False
    assert routing_accepts(c, topk_dtype="int64") is False
    assert routing_accepts(c, top_k_meta=2) is False
    assert routing_accepts(c, topk_cont=False) is False
    print("PASS routing role/window/dtype/geometry contract")


def test_optin_read_once_at_module_level():
    for relpath in (
        "python/sglang/srt/models/glm4_moe_lite.py",
        "python/sglang/srt/hardware_backend/npu/moe/init_routing.py",
    ):
        src = (REPO / relpath).read_text()
        assert src.count('"SGLANG_NPU_MIXED_PREFILL_OPT"') == 1, relpath
        tree = ast.parse(src)
        asserts = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id == "_NPU_MIXED_PREFILL_OPT"
                for t in node.targets
            )
        ]
        assert len(asserts) == 1, (relpath, len(asserts))
        assert asserts[0].col_offset == 0, "module-level read expected"
    print("PASS opt-in read exactly once at module level")


def main():
    test_qkv_role_contract()
    test_routing_role_and_row_contract()
    test_optin_read_once_at_module_level()
    print("ALL MIXED PREFILL OPT CONTRACT TESTS PASSED")


if __name__ == "__main__":
    main()
