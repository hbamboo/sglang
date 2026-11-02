"""CPU contract tests for SGLANG_NPU_SPLIT_KV_B_PROJ.

Run in the test container (no NPU required):

    python3 test/manual/ascend/test_split_kv_b_proj_contract.py

Covers the per-head weight/bias mapping, the independent CPU FP32 reference
against the original one-matmul projection, support/fallback predicates and
stale-state invalidation.  NPU routing and timing live in the sibling scripts.
"""

import os
import sys

FRAMEWORK_ROOT = os.environ.get(
    "SPLIT_KV_FRAMEWORK_ROOT",
    "/mnt/sfs_turbo/hxd/split-kv-matmul-20261102/framework/python",
)
if FRAMEWORK_ROOT not in sys.path:
    sys.path.insert(0, FRAMEWORK_ROOT)

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.npu.modules import split_kv_b_proj as skv


class UnquantizedLinearMethod:  # class name is part of the contract
    pass


class Fp8LinearMethod:
    pass


class ColumnParallelLinear(torch.nn.Module):  # class name is part of the contract
    pass


class OtherLinear:
    pass


class FakeAttn:
    def __init__(self, heads=4, dk=192, dv=256, k=512, use_dsa=False):
        self.num_local_heads = heads
        self.qk_nope_head_dim = dk
        self.v_head_dim = dv
        self.kv_lora_rank = k
        self.use_dsa = use_dsa


def make_module(
    heads=4,
    dk=192,
    dv=256,
    k=512,
    *,
    dtype=torch.bfloat16,
    bias=False,
    module_cls=ColumnParallelLinear,
    quant_cls=UnquantizedLinearMethod,
):
    module = module_cls()
    out = heads * (dk + dv)
    module.weight = torch.randn(out, k, dtype=dtype)
    module.quant_method = quant_cls()
    module.bias = torch.randn(out, dtype=dtype) if bias else None
    module.gather_output = False
    module.skip_bias_add = False
    module.output_size_per_partition = out
    module.input_size = k
    module.tp_size = 1
    return module


def test_identifiable_mapping():
    heads, dk, dv, k = 20, 192, 256, 64
    total = dk + dv
    h = torch.arange(heads).view(heads, 1, 1) * 100_000
    d = torch.arange(total).view(1, total, 1) * 100
    kk = torch.arange(k).view(1, 1, k)
    weight = (h + d + kk).to(torch.float64).reshape(heads * total, k)
    b = (torch.arange(heads).view(heads, 1) * 10_000 + torch.arange(total).view(1, total))
    bias = b.to(torch.float64).reshape(-1)

    w_k, w_v, b_k, b_v = skv.split_kv_b_proj_weight(weight, bias, heads, dk, dv)
    assert w_k.shape == (heads * dk, k) and w_v.shape == (heads * dv, k)
    assert b_k.shape == (heads * dk,) and b_v.shape == (heads * dv,)
    assert w_k.data_ptr() != weight.data_ptr() and w_v.data_ptr() != weight.data_ptr()
    assert b_k.data_ptr() != bias.data_ptr() and b_v.data_ptr() != bias.data_ptr()
    for hh in range(heads):
        assert torch.equal(w_k[hh * dk : (hh + 1) * dk], weight[hh * total : hh * total + dk])
        assert torch.equal(
            w_v[hh * dv : (hh + 1) * dv],
            weight[hh * total + dk : (hh + 1) * total],
        )
        assert torch.equal(b_k[hh * dk : (hh + 1) * dk], bias[hh * total : hh * total + dk])
        assert torch.equal(
            b_v[hh * dv : (hh + 1) * dv],
            bias[hh * total + dk : (hh + 1) * total],
        )
    print("PASS identifiable per-head weight/bias mapping (H=20, Dk=192, Dv=256)")


def test_cpu_reference():
    torch.manual_seed(0)
    heads, dk, dv, k = 20, 192, 256, 512
    for t in (0, 1, 31, 129):
        weight = torch.randn(heads * (dk + dv), k, dtype=torch.float32)
        bias = torch.randn(heads * (dk + dv), dtype=torch.float32)
        x = torch.randn(t, k, dtype=torch.float32)
        old = (x @ weight.t() + bias).view(t, heads, dk + dv)
        old_k, old_v = old[..., :dk], old[..., dk:]
        w_k, w_v, b_k, b_v = skv.split_kv_b_proj_weight(weight, bias, heads, dk, dv)
        new_k = F.linear(x, w_k, b_k).view(t, heads, dk)
        new_v = F.linear(x, w_v, b_v).view(t, heads, dv)
        k_err = (old_k - new_k).abs()
        v_err = (old_v - new_v).abs()
        k_max = k_err.max().item() if k_err.numel() else 0.0
        v_max = v_err.max().item() if v_err.numel() else 0.0
        # Different output widths may pick different BLAS blocking, so the FP32
        # sums need not be bit-identical even though the math is the same.
        assert k_max <= 1e-4, (t, "K", k_max)
        assert v_max <= 1e-4, (t, "V", v_max)
        assert new_k.is_contiguous() and new_v.is_contiguous()
        print(
            f"PASS CPU FP32 split vs original T={t} "
            f"K max_abs={k_max:.3e} V max_abs={v_max:.3e}"
        )
    print("PASS CPU reference (FP32 tolerance atol=1e-4)")


def test_support_predicates():
    attn = FakeAttn()
    mod = make_module()
    assert skv.unsupported_reason(attn, mod, check_device=False) is None

    assert "module type" in skv.unsupported_reason(
        attn, make_module(module_cls=OtherLinear), check_device=False
    )
    assert "quant_method" in skv.unsupported_reason(
        attn, make_module(quant_cls=Fp8LinearMethod), check_device=False
    )
    mod_f32 = make_module(dtype=torch.float32)
    assert skv.unsupported_reason(attn, mod_f32, check_device=False).startswith("dtype")
    mod_bf16 = make_module(dtype=torch.bfloat16)
    assert "device" in skv.unsupported_reason(attn, mod_bf16, check_device=True)
    assert skv.unsupported_reason(attn, mod_bf16, check_device=False) is None
    mod_gather = make_module(dtype=torch.bfloat16)
    mod_gather.gather_output = True
    assert skv.unsupported_reason(attn, mod_gather, check_device=False) == "gather_output=True"
    mod_skip = make_module(dtype=torch.bfloat16)
    mod_skip.skip_bias_add = True
    assert skv.unsupported_reason(attn, mod_skip, check_device=False) == "skip_bias_add=True"
    assert skv.unsupported_reason(FakeAttn(use_dsa=True), make_module(dtype=torch.bfloat16), check_device=False) == "DSA"
    mod_shape = make_module(heads=4, dtype=torch.bfloat16)
    attn5 = FakeAttn(heads=5)
    assert "local H*(Dk+Dv)" in skv.unsupported_reason(attn5, mod_shape, check_device=False)
    mod_bias = make_module(dtype=torch.bfloat16, bias=True)
    mod_bias.bias = mod_bias.bias[:-1]
    assert "bias numel" in skv.unsupported_reason(attn, mod_bias, check_device=False)
    mod_hook = make_module(dtype=torch.bfloat16)
    mod_hook.register_forward_hook(lambda *args: None)
    assert skv.unsupported_reason(attn, mod_hook, check_device=False) == "forward hooks"
    mod_lora = make_module(dtype=torch.bfloat16)
    mod_lora.set_lora = True
    assert "LoRA" in skv.unsupported_reason(attn, mod_lora, check_device=False)
    print("PASS support/fallback predicates")


def test_state_lifecycle_cpu():
    # Force the state path on CPU; production never reaches it because the
    # device check rejects non-NPU weights.
    orig_check = skv.unsupported_reason
    skv.unsupported_reason = lambda attn, module, **kw: None
    try:
        attn = FakeAttn(heads=4, dk=8, dv=12, k=16)
        mod = make_module(heads=4, dk=8, dv=12, k=16, dtype=torch.float32)
        state, reason = skv.prepare_split_kv_b_proj(attn, mod, force=True)
        assert reason == "prepared" and state is not None
        state2, reason2 = skv.prepare_split_kv_b_proj(attn, mod, force=True)
        # force always rebuilds: the real loader cannot rely on _version.
        assert reason2 == "prepared" and state2 is not state
        orig_enabled = skv.is_split_kv_b_proj_enabled
        skv.is_split_kv_b_proj_enabled = lambda: True
        try:
            state_reuse, reason_reuse = skv.prepare_split_kv_b_proj(attn, mod)
            assert reason_reuse == "already prepared" and state_reuse is state2
        finally:
            skv.is_split_kv_b_proj_enabled = orig_enabled

        x = torch.randn(7, 16)
        out = skv.maybe_forward_split_kv_b_proj(attn, mod, x, None)
        assert out is not None
        k_nope, v = out
        assert k_nope.shape == (7, 4, 8) and v.shape == (7, 4, 12)
        assert k_nope.is_contiguous() and v.is_contiguous()

        # Actual ColumnParallelLinear.weight_loader writes through param.data;
        # this does not bump Parameter._version, so only the loader's forced
        # invalidation/rebuild can prevent stale split buffers.
        version_before = mod.weight._version
        new_weight = torch.randn_like(mod.weight)
        with torch.no_grad():
            mod.weight.data.copy_(new_weight)
        assert mod.weight._version == version_before
        exp_k, exp_v, _, _ = skv.split_kv_b_proj_weight(
            new_weight, None, 4, 8, 12
        )
        state4, reason4 = skv.prepare_split_kv_b_proj(attn, mod, force=True)
        assert reason4 == "prepared"
        assert torch.equal(state4.w_k, exp_k) and torch.equal(state4.w_v, exp_v)
        assert state4 is not state2
        fresh = skv.maybe_forward_split_kv_b_proj(attn, mod, x, None)
        assert fresh is not None
        assert torch.allclose(
            fresh[0].reshape(-1, 32),
            torch.nn.functional.linear(x, exp_k),
            rtol=0,
            atol=1e-4,
        )
        assert torch.allclose(
            fresh[1].reshape(-1, 48),
            torch.nn.functional.linear(x, exp_v),
            rtol=0,
            atol=1e-4,
        )

        # In-place modification that bumps _version -> immediate fallback.
        with torch.no_grad():
            mod.weight.add_(1.0)
        assert skv.maybe_forward_split_kv_b_proj(attn, mod, x, None) is None
        state3, reason3 = skv.prepare_split_kv_b_proj(attn, mod, force=True)
        assert reason3 == "prepared"
        assert skv.maybe_forward_split_kv_b_proj(attn, mod, x, None) is not None

        # replacing the weight storage -> fall back, never stale.
        mod.weight = mod.weight.double()
        assert skv.maybe_forward_split_kv_b_proj(attn, mod, x.double(), None) is None

        # Final after-load hook: refresh only stale derived state.
        state_f, reason_f = skv.prepare_split_kv_b_proj(attn, mod, force=True)
        assert state_f is not None

        class Holder(torch.nn.Module):
            def __init__(self, child):
                super().__init__()
                self.kv = child

        holder = Holder(mod)
        assert skv.finalize_split_kv_b_proj(holder) == 1
        final_weight = torch.randn_like(mod.weight)
        mod.weight = torch.nn.Parameter(final_weight)
        assert skv.maybe_forward_split_kv_b_proj(
            attn, mod, x.double(), None
        ) is None
        assert skv.finalize_split_kv_b_proj(holder) == 1
        state_g = skv.split_kv_b_proj_state(mod)
        assert state_g is not None and skv._state_is_current(state_g, mod)
        exp_k, exp_v, _, _ = skv.split_kv_b_proj_weight(
            final_weight, None, 4, 8, 12
        )
        assert torch.equal(state_g.w_k, exp_k) and torch.equal(state_g.w_v, exp_v)
        assert skv.maybe_forward_split_kv_b_proj(
            attn, mod, x.double(), None
        ) is not None

        # Late param.data.copy_ does not bump _version; the finalizer still
        # rebuilds from the copied final weight.
        late = torch.randn_like(mod.weight)
        late_version = mod.weight._version
        with torch.no_grad():
            mod.weight.data.copy_(late)
        assert mod.weight._version == late_version
        assert skv.finalize_split_kv_b_proj(holder) == 1
        state_h = skv.split_kv_b_proj_state(mod)
        exp_k, exp_v, _, _ = skv.split_kv_b_proj_weight(late, None, 4, 8, 12)
        assert torch.equal(state_h.w_k, exp_k) and torch.equal(state_h.w_v, exp_v)
        assert skv.maybe_forward_split_kv_b_proj(
            attn, mod, x.double(), None
        ) is not None

        skv.invalidate_split_kv_b_proj(mod)
        assert skv.split_kv_b_proj_state(mod) is None
        assert mod._buffers["_split_kv_b_proj_w_k"] is None
    finally:
        skv.unsupported_reason = orig_check
    print("PASS state lifecycle: reload/dtype invalidation and invalidation API")


def main():
    test_identifiable_mapping()
    test_cpu_reference()
    test_support_predicates()
    test_state_lifecycle_cpu()
    print("ALL CPU CONTRACT TESTS PASSED")


if __name__ == "__main__":
    main()
