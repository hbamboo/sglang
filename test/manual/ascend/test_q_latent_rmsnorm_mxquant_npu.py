"""Granted-NPU correctness / profiling / timing harness for the Q-latent
RMSNorm + DynamicMxQuant fusion.

Inert without an explicit grant:

    QNORM_MXQUANT_NPU_GRANTED=1 PYTHONPATH=<worktree>/python \
        python test/manual/ascend/test_q_latent_rmsnorm_mxquant_npu.py

Importing this module only imports ``torch``; sglang/torch_npu are loaded
lazily inside ``main`` after the grant check, so the CPU oracle self-test can
import and exercise the pure acceptance helpers without touching an NPU.

Acceptance thresholds are declared in ``CRITERIA`` before any device
observation and must not be loosened after a failure.
"""

import argparse
import contextlib
import json
import math
import os
import statistics
import sys
import traceback

import torch

# ---------------------------------------------------------------------------
# geometry: representative DeepSeek q_b_proj slice (20 heads x 256)
# ---------------------------------------------------------------------------
Q_LORA = 768
KV_LORA = 512
QK_ROPE = 64
QK_NOPE = 192
QK_HEAD = QK_NOPE + QK_ROPE  # 256
NUM_HEADS = 20
V_HEAD = 64
OUT_FEATURES = NUM_HEADS * QK_HEAD  # 5120
KV_B_OUT = NUM_HEADS * (QK_NOPE + V_HEAD)  # 5120
QKV_WIDTH = Q_LORA + KV_LORA + QK_ROPE  # 1344
BLOCK = 32
EMAX_E4M3 = 8
FP8_MAX = 448.0
E8M0_BIAS = 127
TOKENS = [0, 1, 31, 127, 128, 129, 3787, 15148]
DEFAULT_OUT_DIR = "/mnt/sfs_turbo/hxd/rmsnorm-mxquant-fusion-20261102/results/npu"

# Predeclared acceptance criteria (do not loosen after observing failures).
CRITERIA = {
    "bf16_downstream": {
        "atol": 2.0**-6,
        "rtol": 2.0**-6,
        "min_matched_ratio": 0.99,
        "max_abs": 1.0,
    },
    "near_exact_diagnostic": {
        "atol": 2.0**-8,
        "rtol": 2.0**-8,
        "min_matched_ratio": 0.99,
        "max_abs": 2.0**-4,
    },
    "quant_vs_oracle": {
        # One E4M3 ULP is 2**-3 relative for a normal value; +2**-9*block_scale
        # covers subnormals.  One percent slack covers block-scale boundary
        # flips caused by BF16-vs-FP32 norm rounding.
        "rel": 2.0**-3,
        "scale_abs": 2.0**-9,
        "min_matched_ratio": 0.99,
    },
    "scale_vs_oracle": {"min_matched_ratio": 0.99},
    "fp8_old_vs_new": {
        # A BF16-vs-FP32 norm perturbation (~2**-9 rel) crosses an FP8
        # rounding boundary with ~3% probability; 5% is the declared bound.
        "min_matched_ratio": 0.95,
    },
    "scale_old_vs_new": {"min_matched_ratio": 0.99},
    "old_tensor_vs_old_tuple": {"mismatch": 0},
}

_RUNTIME = {}


def _load_runtime():
    """Import torch_npu/sglang only after an explicit NPU grant."""
    global torch_npu, qlm, mha, NPUMXFP8LinearMethod, RMSNorm, _RUNTIME
    import torch_npu as _torch_npu

    import sglang.srt.layers.quantization  # noqa: F401
    from sglang.srt.hardware_backend.npu.modules import (
        deepseek_v2_attention_mla_npu as _mha,
    )
    from sglang.srt.hardware_backend.npu.modules import (
        q_latent_rmsnorm_mxquant as _qlm,
    )
    from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
        NPUMXFP8LinearMethod as _kernel,
    )
    from sglang.srt.layers.layernorm import RMSNorm as _rmsnorm

    torch_npu = _torch_npu
    qlm = _qlm
    mha = _mha
    NPUMXFP8LinearMethod = _kernel
    RMSNorm = _rmsnorm
    _RUNTIME.update(
        torch_npu=_torch_npu, qlm=_qlm, mha=_mha, kernel=_kernel, rmsnorm=_rmsnorm
    )


def _sync():
    torch.npu.synchronize()


# ---------------------------------------------------------------------------
# pure acceptance / statistics helpers (CPU-testable)
# ---------------------------------------------------------------------------
def match_stats(ref, got, atol, rtol, chunk_rows=1024):
    """BF16-style per-element match statistics; memory-bounded by row chunks."""
    if tuple(ref.shape) != tuple(got.shape):
        raise ValueError(f"shape mismatch {tuple(ref.shape)} vs {tuple(got.shape)}")
    if ref.numel() == 0:
        return {
            "count": 0,
            "matched": 0,
            "matched_ratio": 1.0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "finite": 0,
            "bad": 0,
        }
    rows = ref.shape[0]
    rf = ref.reshape(rows, -1)
    gf = got.reshape(rows, -1)
    total = matched = 0
    sum_abs = 0.0
    max_abs = 0.0
    finite = 0
    bad = 0
    for start in range(0, rows, chunk_rows):
        r = rf[start : start + chunk_rows].float()
        g = gf[start : start + chunk_rows].float()
        d = (g - r).abs()
        tol = atol + rtol * r.abs()
        m = d <= tol
        fin = torch.isfinite(d)
        total += int(d.numel())
        matched += int(m.sum().item())
        finite += int(fin.sum().item())
        bad += int((~fin).sum().item())
        sum_abs += float(d.sum().item())
        max_abs = max(max_abs, float(d.max().item()))
    return {
        "count": total,
        "matched": matched,
        "matched_ratio": matched / total if total else 1.0,
        "max_abs": max_abs,
        "mean_abs": sum_abs / total if total else 0.0,
        "finite": finite,
        "bad": bad,
    }


def tolerance_stats(diff, tol):
    diff = diff.reshape(-1)
    finite = torch.isfinite(diff)
    matched = int(((diff <= tol) & finite).sum().item())
    total = int(diff.numel())
    return {
        "count": total,
        "matched": matched,
        "matched_ratio": matched / total if total else 1.0,
        "max_abs": float(diff.max().item()) if total else 0.0,
        "mean_abs": float(diff.mean().item()) if total else 0.0,
        "finite": int(finite.sum().item()),
        "bad": total - int(finite.sum().item()),
    }


def bf16_standard_pass(stats):
    c = CRITERIA["bf16_downstream"]
    return (
        stats["bad"] == 0
        and stats["matched_ratio"] >= c["min_matched_ratio"]
        and stats["max_abs"] <= c["max_abs"]
    )


def near_exact_pass(stats):
    c = CRITERIA["near_exact_diagnostic"]
    return (
        stats["bad"] == 0
        and stats["matched_ratio"] >= c["min_matched_ratio"]
        and stats["max_abs"] <= c["max_abs"]
    )


def quant_oracle_pass(stats):
    return (
        stats["bad"] == 0
        and stats["matched_ratio"] >= CRITERIA["quant_vs_oracle"]["min_matched_ratio"]
    )


def scale_oracle_pass(stats):
    return (
        stats["matched_ratio"] >= CRITERIA["scale_vs_oracle"]["min_matched_ratio"]
    )


def fp8_old_new_pass(stats):
    return (
        stats["matched_ratio"] >= CRITERIA["fp8_old_vs_new"]["min_matched_ratio"]
    )


def scale_old_new_pass(stats):
    return (
        stats["matched_ratio"] >= CRITERIA["scale_old_vs_new"]["min_matched_ratio"]
    )


def byte_match(a, b, chunk_rows=2048):
    """Byte-level equality count for same-dtype tensors (FP8/uint8/BF16).

    Row-chunked so large BF16 outputs do not require a full CPU copy.
    """
    if tuple(a.shape) != tuple(b.shape):
        raise ValueError(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.dtype != b.dtype:
        raise ValueError(f"dtype mismatch {a.dtype} vs {b.dtype}")
    if a.dim() == 0:
        a = a.reshape(1)
        b = b.reshape(1)
    if a.dim() == 1:
        ab = a.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
        bb = b.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
        total = int(ab.numel())
        matched = int((ab == bb).sum().item())
    else:
        total = matched = 0
        rows = a.shape[0]
        for start in range(0, rows, chunk_rows):
            ab = (
                a[start : start + chunk_rows]
                .detach()
                .contiguous()
                .cpu()
                .reshape(-1)
                .view(torch.uint8)
            )
            bb = (
                b[start : start + chunk_rows]
                .detach()
                .contiguous()
                .cpu()
                .reshape(-1)
                .view(torch.uint8)
            )
            total += int(ab.numel())
            matched += int((ab == bb).sum().item())
    return {
        "count": total,
        "mismatch": total - matched,
        "matched": matched,
        "matched_ratio": matched / total if total else 1.0,
    }


def e8m0_scale_float(scale_u8):
    return torch.pow(2.0, scale_u8.float() - float(E8M0_BIAS))


def dequant_mx(q_fp8, scale_u8):
    rows, width = q_fp8.shape
    blocks = width // BLOCK
    scale = e8m0_scale_float(scale_u8.reshape(rows, blocks))
    return q_fp8.float() * scale.repeat_interleave(BLOCK, dim=-1)


def gather_fp8_rows_to_cpu(t, idx):
    """Gather rows of an FP8 tensor for the CPU oracle.

    Installed aclnnIndex does not implement DT_FLOAT8_E4M3FN, so cast to a
    supported float type BEFORE the NPU gather.  Byte-level old/new comparisons
    happen on the raw FP8 tensors before this conversion.
    """
    return t.float()[idx].cpu()


def cast_e4m3_saturating(x):
    """Saturating E4M3FN conversion used by the declared CPU oracle.

    torch CPU ``.to(float8_e4m3fn)`` is not saturating above the rounding
    threshold (e.g. 512 -> NaN; 460 -> 448).  Native MX kernels convert with a
    saturating finite E4M3FN cast, so the oracle clamps to +/-448 first.
    """
    x = torch.nan_to_num(x, nan=float("nan"), posinf=FP8_MAX, neginf=-FP8_MAX)
    return torch.clamp(x, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def cpu_rms_oracle(x_bf16, gamma_bf16, eps):
    """Independent CPU FP32 RMSNorm semantics."""
    xf = x_bf16.float()
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return xf * inv * gamma_bf16.float()


def round_to_bf16_fp32(x):
    """The baseline npu_rms_norm emits a BF16 intermediate; match that."""
    return x.to(torch.bfloat16).float()


def cpu_mx_quant_oracle(norm_fp32):
    """OCP scale_alg=0 E4M3FN block-32 quantisation, independent of NPU.

    Returns dequantised values, scale bytes in E8M0 bias-127 form, and the
    count of values that exercise the saturating cast.  Zero blocks use scale
    byte 0, matching the installed scale kernel's zero-mask branch.
    """
    rows, width = norm_fp32.shape
    nblocks = width // BLOCK
    blocks = norm_fp32.reshape(rows, nblocks, BLOCK)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    _, exp = torch.frexp(amax)  # amax = m * 2**e, 0.5 <= m < 1
    shared_exp = torch.where(
        amax > 0, (exp - 1) - EMAX_E4M3, torch.zeros_like(exp)
    )
    safe_scale = torch.where(
        amax > 0,
        torch.pow(2.0, shared_exp.float()),
        torch.ones_like(amax),
    )
    scaled = blocks / safe_scale
    saturation_count = int((scaled.abs() > FP8_MAX).sum().item())
    q = cast_e4m3_saturating(scaled)
    dequant = q.float() * safe_scale
    scale_bytes = torch.where(
        amax > 0,
        shared_exp + E8M0_BIAS,
        torch.zeros_like(shared_exp),
    ).clamp(0, 255).to(torch.uint8)
    return {
        "dequant": dequant.reshape(rows, width),
        "q": q.reshape(rows, width),
        "scale_bytes": scale_bytes.squeeze(-1),
        "saturation_count": saturation_count,
    }


def quant_oracle_stats(oracle_dequant, npu_dequant, scale_bytes):
    c = CRITERIA["quant_vs_oracle"]
    scale = e8m0_scale_float(scale_bytes).repeat_interleave(BLOCK, dim=-1)
    diff = (npu_dequant.float() - oracle_dequant.float()).abs()
    tol = c["rel"] * oracle_dequant.float().abs() + c["scale_abs"] * scale
    return tolerance_stats(diff, tol)


# ---------------------------------------------------------------------------
# input / module builders
# ---------------------------------------------------------------------------
def make_qkv(rows, seed, device, layout="stride1344_offset", offset=64, padded=64,
             extreme=False):
    """Build the backing tensor on CPU, move it, then form the strided view.

    Forming the view after ``.to(device)`` preserves the non-zero storage
    offset and row stride; converting a CPU strided view to NPU would compact
    it.
    """
    g = torch.Generator().manual_seed(seed)
    if layout == "stride1344_offset":
        backing = (
            torch.randn(rows * QKV_WIDTH + offset, generator=g)
            .to(torch.bfloat16)
            .to(device)
        )
        qkv = backing[offset:].view(rows, QKV_WIDTH)
        expected_stride = (QKV_WIDTH, 1)
        expected_offset = offset
        backing_shape = [rows * QKV_WIDTH + offset]
    elif layout == "padded_stride_offset":
        backing = (
            torch.randn((rows, QKV_WIDTH + padded), generator=g)
            .to(torch.bfloat16)
            .to(device)
        )
        qkv = backing[:, offset : offset + QKV_WIDTH]
        expected_stride = (QKV_WIDTH + padded, 1)
        expected_offset = offset
        backing_shape = [rows, QKV_WIDTH + padded]
    else:
        raise ValueError(f"unknown layout {layout!r}")

    observed_stride = tuple(qkv.stride())
    observed_offset = int(qkv.storage_offset())
    layout_ok = (
        observed_stride == expected_stride and observed_offset == expected_offset
    )
    if extreme and rows > 0:
        vals = torch.tensor(
            [448.0, 224.0, 1.0, 0.5, 2.0**-9, 2.0**-20, 0.0, -448.0, -1.0, -2.0**-9],
            dtype=torch.bfloat16,
        )
        idx = torch.randint(0, vals.numel(), (rows, QKV_WIDTH), generator=g)
        qkv.copy_(vals[idx].to(device))
    return {
        "backing": backing,
        "qkv": qkv,
        "layout": layout,
        "backing_shape": backing_shape,
        "expected_stride": list(expected_stride),
        "expected_offset": expected_offset,
        "observed_stride": list(observed_stride),
        "observed_offset": observed_offset,
        "layout_ok": layout_ok,
    }


def make_gamma(seed, eps, device):
    g = torch.Generator().manual_seed(seed + 991)
    gamma = (1.0 + 0.1 * torch.randn((Q_LORA,), generator=g)).to(torch.bfloat16)
    return gamma.to(device)


def make_norm(width, gamma, eps, device):
    norm = RMSNorm(width, eps=eps, weight_dtype=torch.bfloat16)
    norm.to(device)
    with torch.no_grad():
        norm.weight.data.copy_(gamma)
    return norm


class _LinearAdapter:
    """ColumnParallelLinear-compatible surface for the real MXFP8 kernel."""

    def __init__(self, weight, kernel):
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.bias = None
        self.quant_method = kernel
        self._mx_kernel = kernel
        self.tp_size = 1
        self.gather_output = False
        self.skip_bias_add = False
        self._sgl_npu_mx_quant_fusable = True

    def __call__(self, x):
        return self.quant_method.apply(self, x, None), None


def make_layer(out_features, seed, device):
    g = torch.Generator().manual_seed(seed + 17)
    weight = (
        torch.randn((out_features, Q_LORA), generator=g)
        .to(torch.bfloat16)
        .to(device)
    )
    layer = _LinearAdapter(weight, None)
    kernel = NPUMXFP8LinearMethod()
    kernel.process_weights_after_loading(layer)
    if not getattr(layer, "_sgl_npu_mx_quant_fusable", False):
        raise AssertionError("NPUMXFP8LinearMethod did not set the fusable contract")
    layer.quant_method = kernel
    layer._mx_kernel = kernel
    return layer


def rows_for_oracle(rows):
    if rows <= 2048:
        return torch.arange(rows)
    return torch.linspace(0, rows - 1, 256).long().unique()


@contextlib.contextmanager
def count_op(namespace, name):
    original = getattr(namespace, name)
    counter = {"n": 0}

    def wrapped(*args, **kwargs):
        counter["n"] += 1
        return original(*args, **kwargs)

    setattr(namespace, name, wrapped)
    try:
        yield counter
    finally:
        setattr(namespace, name, original)


@contextlib.contextmanager
def record_fused_outputs():
    """Correctness-only recorder.  Never installed around timed/profile routes."""
    real = qlm._fused_rmsnorm_mxquant_op
    state = {}

    def factory():
        op = real()

        def wrapped(*args, **kwargs):
            out = op(*args, **kwargs)
            state["q"], state["scale"], state["rstd"] = out[0], out[1], out[2]
            return out

        return wrapped

    qlm._fused_rmsnorm_mxquant_op = factory
    try:
        yield state
    finally:
        qlm._fused_rmsnorm_mxquant_op = real


def _finite_stats(t, chunk_rows=2048):
    total = int(t.numel())
    finite = 0
    if t.dim() >= 2:
        rows = t.shape[0]
        for start in range(0, rows, chunk_rows):
            chunk = t[start : start + chunk_rows].float()
            finite += int(torch.isfinite(chunk).sum().item())
    else:
        finite = int(torch.isfinite(t.float()).sum().item())
    return {"count": total, "finite": finite, "bad": total - finite}


# ---------------------------------------------------------------------------
# correctness
# ---------------------------------------------------------------------------
def run_case(rows, seed, eps, device, layout="stride1344_offset", offset=64,
             extreme=False):
    meta = {
        "rows": rows,
        "seed": seed,
        "eps": eps,
        "layout": layout,
        "offset": offset,
        "extreme": extreme,
        "beta": None,
    }
    failures = []
    try:
        inp = make_qkv(
            rows, seed, device, layout=layout, offset=offset, extreme=extreme
        )
        meta.update(
            {
                "backing_shape": inp["backing_shape"],
                "expected_stride": inp["expected_stride"],
                "expected_offset": inp["expected_offset"],
                "observed_stride": inp["observed_stride"],
                "observed_offset": inp["observed_offset"],
                "layout_ok": inp["layout_ok"],
            }
        )
        if not inp["layout_ok"]:
            failures.append("qkv_layout_mismatch")

        qkv = inp["qkv"]
        q = qkv.split([Q_LORA, QKV_WIDTH - Q_LORA], dim=-1)[0]
        meta["q_stride"] = list(q.stride())
        meta["q_storage_offset"] = int(q.storage_offset())
        if (
            q.stride(0) != inp["expected_stride"][0]
            or q.storage_offset() != inp["expected_offset"]
        ):
            failures.append("q_slice_layout_mismatch")

        gamma = make_gamma(seed, eps, device)
        layer = make_layer(OUT_FEATURES, seed, device)
        norm = make_norm(Q_LORA, gamma, eps, device)

        if rows == 0:
            meta["route"] = "fallback_empty"
            meta["gate_reason"] = qlm.fusion_blocker(norm, layer, q)
            if meta["gate_reason"] != "empty_input":
                failures.append(f"empty_gate={meta['gate_reason']}")
            meta["failures"] = failures
            meta["ok"] = not failures
            return meta

        # Old path: real norm module -> explicit DynamicMxQuant -> same kernel.
        q_norm = norm(q)
        q_old, scale_old = torch.ops.npu.npu_dynamic_mx_quant(
            q_norm, dst_type=torch.float8_e4m3fn
        )
        out_old = layer._mx_kernel.apply(layer, q_norm, None)
        out_old_tuple = layer._mx_kernel.apply(layer, (q_old, scale_old), None)
        _sync()

        # New path: helper without injected gate overrides (real capture query).
        with record_fused_outputs() as rec:
            with count_op(torch.ops.npu, "npu_dynamic_mx_quant") as dyn_count:
                out_new = qlm.forward_q_latent_rmsnorm_mxquant(norm, layer, q)
                _sync()
        q_new = rec.get("q")
        scale_new = rec.get("scale")
        meta["extra_dynamic_mx_quant_after_fused"] = dyn_count["n"]
        if dyn_count["n"] != 0:
            failures.append("double_quant_after_fused")
        if out_new is None:
            failures.append("fused_route_returned_none")

        # Shape / dtype / layout / finiteness.
        def check(cond, msg):
            if not cond:
                failures.append(msg)

        check(isinstance(q_new, torch.Tensor), "q_new_missing")
        check(isinstance(scale_new, torch.Tensor), "scale_new_missing")
        if isinstance(q_new, torch.Tensor):
            meta["q_new_dtype"] = str(q_new.dtype)
            meta["q_new_shape"] = list(q_new.shape)
            check(q_new.dtype == torch.float8_e4m3fn, "q_new_dtype")
            check(tuple(q_new.shape) == (rows, Q_LORA), "q_new_shape")
        if isinstance(scale_new, torch.Tensor):
            meta["scale_new_dtype"] = str(scale_new.dtype)
            meta["scale_new_shape"] = list(scale_new.shape)
            check(scale_new.dtype == torch.uint8, "scale_new_dtype")
            check(
                tuple(scale_new.shape) == (rows, Q_LORA // 64, 2),
                "scale_new_shape",
            )
        meta["q_old_dtype"] = str(q_old.dtype)
        meta["q_old_shape"] = list(q_old.shape)
        check(q_old.dtype == torch.float8_e4m3fn, "q_old_dtype")
        check(tuple(q_old.shape) == (rows, Q_LORA), "q_old_shape")
        check(scale_old.dtype == torch.uint8, "scale_old_dtype")
        check(tuple(scale_old.shape) == (rows, Q_LORA // 64, 2), "scale_old_shape")
        if not (
            isinstance(q_new, torch.Tensor)
            and isinstance(scale_new, torch.Tensor)
            and isinstance(out_new, torch.Tensor)
        ):
            meta["failures"] = failures
            meta["ok"] = False
            return meta
        if isinstance(out_new, torch.Tensor):
            meta["out_new_dtype"] = str(out_new.dtype)
            meta["out_new_shape"] = list(out_new.shape)
            check(out_new.dtype == torch.bfloat16, "out_new_dtype")
            check(tuple(out_new.shape) == (rows, OUT_FEATURES), "out_new_shape")
            meta["out_new_finite"] = _finite_stats(out_new)
            check(meta["out_new_finite"]["bad"] == 0, "out_new_nonfinite")
        meta["out_old_finite"] = _finite_stats(out_old)
        check(meta["out_old_finite"]["bad"] == 0, "out_old_nonfinite")
        if isinstance(q_new, torch.Tensor):
            meta["q_new_finite"] = _finite_stats(q_new)
            check(meta["q_new_finite"]["bad"] == 0, "q_new_nonfinite")
        meta["q_old_finite"] = _finite_stats(q_old)
        check(meta["q_old_finite"]["bad"] == 0, "q_old_nonfinite")

        # Old tensor path vs old pre-quantised tuple path: bitwise equal.
        old_tuple = byte_match(out_old, out_old_tuple)
        meta["old_tensor_vs_old_tuple"] = old_tuple
        if old_tuple["mismatch"] != CRITERIA["old_tensor_vs_old_tuple"]["mismatch"]:
            failures.append("old_tensor_vs_old_tuple_not_bitwise")

        # FP8 / scale differences between old and new activation quantisation.
        fp8_diff = byte_match(q_old, q_new)
        scale_diff = byte_match(scale_old, scale_new)
        meta["fp8_old_vs_new"] = fp8_diff
        meta["scale_old_vs_new"] = scale_diff
        if not fp8_old_new_pass(fp8_diff):
            failures.append("fp8_old_vs_new_exceeds_bound")
        if not scale_old_new_pass(scale_diff):
            failures.append("scale_old_vs_new_exceeds_bound")
        meta["dequant_old_vs_new"] = match_stats(
            dequant_mx(q_old, scale_old),
            dequant_mx(q_new, scale_new),
            atol=0.0,
            rtol=0.0,
        )

        # Downstream BF16 standard (required) and near-exact diagnostic.
        bf16_stats = match_stats(
            out_old,
            out_new,
            CRITERIA["bf16_downstream"]["atol"],
            CRITERIA["bf16_downstream"]["rtol"],
        )
        near_stats = match_stats(
            out_old,
            out_new,
            CRITERIA["near_exact_diagnostic"]["atol"],
            CRITERIA["near_exact_diagnostic"]["rtol"],
        )
        meta["output_old_vs_new_bf16"] = bf16_stats
        meta["output_old_vs_new_near_exact"] = near_stats
        meta["output_old_vs_new_near_exact_pass"] = near_exact_pass(near_stats)
        if not bf16_standard_pass(bf16_stats):
            failures.append("downstream_bf16_standard_failed")

        # Independent CPU oracle.  Old chain rounds the norm result to BF16;
        # the fused kernel's internal norm precision is not assumed, so the
        # candidate passes if it matches either declared norm semantics.
        idx = rows_for_oracle(rows)
        idx_dev = idx.to(device)
        x_cpu = q[idx_dev].float().cpu()
        gamma_cpu = gamma.float().cpu()
        norm_fp32 = cpu_rms_oracle(x_cpu, gamma_cpu, eps)
        norm_bf16 = round_to_bf16_fp32(norm_fp32)
        oracle_fp32 = cpu_mx_quant_oracle(norm_fp32)
        oracle_bf16 = cpu_mx_quant_oracle(norm_bf16)

        q_old_cpu = gather_fp8_rows_to_cpu(q_old, idx_dev)
        scale_old_cpu = scale_old[idx_dev].cpu()
        q_new_cpu = gather_fp8_rows_to_cpu(q_new, idx_dev)
        scale_new_cpu = scale_new[idx_dev].cpu()
        old_deq = dequant_mx(q_old_cpu, scale_old_cpu)
        new_deq = dequant_mx(q_new_cpu, scale_new_cpu)

        meta["oracle_rows"] = int(idx.numel())
        meta["oracle_note"] = (
            "full"
            if idx.numel() == rows
            else "deterministic 256-row sample; RMS and block-32 quant are "
            "per-row, so sampled-row comparison is exact"
        )
        meta["oracle_saturation_count_fp32"] = oracle_fp32["saturation_count"]
        meta["oracle_saturation_count_bf16"] = oracle_bf16["saturation_count"]

        meta["old_dequant_vs_oracle_bf16"] = quant_oracle_stats(
            oracle_bf16["dequant"], old_deq, oracle_bf16["scale_bytes"]
        )
        meta["new_dequant_vs_oracle_fp32"] = quant_oracle_stats(
            oracle_fp32["dequant"], new_deq, oracle_fp32["scale_bytes"]
        )
        meta["new_dequant_vs_oracle_bf16"] = quant_oracle_stats(
            oracle_bf16["dequant"], new_deq, oracle_bf16["scale_bytes"]
        )
        meta["old_dequant_vs_oracle_fp32"] = quant_oracle_stats(
            oracle_fp32["dequant"], old_deq, oracle_fp32["scale_bytes"]
        )
        meta["old_scale_vs_oracle_bf16"] = byte_match(
            scale_old_cpu.reshape(idx.numel(), -1), oracle_bf16["scale_bytes"]
        )
        meta["new_scale_vs_oracle_fp32"] = byte_match(
            scale_new_cpu.reshape(idx.numel(), -1), oracle_fp32["scale_bytes"]
        )
        meta["new_scale_vs_oracle_bf16"] = byte_match(
            scale_new_cpu.reshape(idx.numel(), -1), oracle_bf16["scale_bytes"]
        )

        old_oracle_ok = quant_oracle_pass(
            meta["old_dequant_vs_oracle_bf16"]
        ) and scale_oracle_pass(meta["old_scale_vs_oracle_bf16"])
        new_fp32_ok = quant_oracle_pass(
            meta["new_dequant_vs_oracle_fp32"]
        ) and scale_oracle_pass(meta["new_scale_vs_oracle_fp32"])
        new_bf16_ok = quant_oracle_pass(
            meta["new_dequant_vs_oracle_bf16"]
        ) and scale_oracle_pass(meta["new_scale_vs_oracle_bf16"])
        meta["oracle_pass"] = {
            "old_bf16_semantics": old_oracle_ok,
            "new_fp32_semantics": new_fp32_ok,
            "new_bf16_semantics": new_bf16_ok,
            "new_matches_one_declared_semantics": new_fp32_ok or new_bf16_ok,
        }
        if not old_oracle_ok:
            failures.append("old_vs_bf16_oracle_failed")
        if not (new_fp32_ok or new_bf16_ok):
            failures.append("new_vs_declared_oracle_failed")

    except Exception as exc:  # keep raw diagnostics for every failure
        failures.append(f"exception={exc!r}")
        meta["traceback"] = traceback.format_exc()
    meta["failures"] = failures
    meta["ok"] = not failures
    return meta


# ---------------------------------------------------------------------------
# real prepare route
# ---------------------------------------------------------------------------
def build_prepare_fixture(rows, seed, eps, device, layout="stride1344_offset",
                          offset=64):
    inp = make_qkv(rows, seed, device, layout=layout, offset=offset)
    gamma = make_gamma(seed, eps, device)
    layer = make_layer(OUT_FEATURES, seed, device)
    norm = make_norm(Q_LORA, gamma, eps, device)
    kv_gamma = torch.ones(KV_LORA, dtype=torch.bfloat16, device=device)
    kv_norm = make_norm(KV_LORA, kv_gamma, eps, device)
    kv_buffer = torch.empty(
        (rows, KV_B_OUT), dtype=torch.bfloat16, device=device
    )

    class Pool:
        def __init__(self):
            self.sets = 0

        def set_kv_buffer(self, *args):
            self.sets += 1

        def get_kv_buffer(self, layer_id):
            return (
                torch.empty((rows, 1, KV_LORA), dtype=torch.bfloat16, device=device),
                torch.empty((rows, 1, QK_ROPE), dtype=torch.bfloat16, device=device),
            )

    class KvB:
        def __call__(self, x):
            return kv_buffer, None

    m = type("M", (), {})()
    m.q_lora_rank = Q_LORA
    m.use_dsa = False
    m.kv_lora_rank = KV_LORA
    m.qk_rope_head_dim = QK_ROPE
    m.num_local_heads = NUM_HEADS
    m.qk_head_dim = QK_HEAD
    m.qk_nope_head_dim = QK_NOPE
    m.v_head_dim = V_HEAD
    m.q_a_layernorm = norm
    m.q_b_proj = layer
    m.kv_a_layernorm = kv_norm
    m.kv_b_proj = KvB()
    m.use_deepseek_yarn_rope = False
    m.rotary_emb = None
    m.layer_id = 0
    m._concat_and_cast_mha_k = lambda k_nope, k_pe, batch: k_nope
    pool = Pool()
    return {
        "m": m,
        "inp": inp,
        "qkv": inp["qkv"],
        "layer": layer,
        "norm": norm,
        "pool": pool,
        "batch": type("B", (), {"out_cache_loc": torch.arange(rows, device=device)})(),
        "positions": torch.arange(rows, device=device),
        "hidden": torch.zeros((rows, 8), dtype=torch.bfloat16, device=device),
        "scatter_modes": type(
            "S", (), {"layer_input_mode": None, "attn_mode": None}
        )(),
        "rows": rows,
        "layout_ok": inp["layout_ok"],
        "expected_stride": inp["expected_stride"],
        "expected_offset": inp["expected_offset"],
        "observed_stride": inp["observed_stride"],
        "observed_offset": inp["observed_offset"],
    }


@contextlib.contextmanager
def prepare_environment(fx):
    old = {
        "get_attn_tp_context": mha.get_attn_tp_context,
        "get_token_to_kv_pool": mha.get_token_to_kv_pool,
        "can_fuse_expanded_kv": mha.can_fuse_expanded_kv,
        "scattered_to_tp_attn_full": mha.scattered_to_tp_attn_full,
        "_use_ag_after_qlora": mha._use_ag_after_qlora,
    }
    mha.get_attn_tp_context = lambda: type(
        "Ctx", (), {"fetch_qkv_latent": lambda self: fx["qkv"]}
    )()
    mha.get_token_to_kv_pool = lambda: fx["pool"]
    mha.can_fuse_expanded_kv = lambda *a: False
    mha.scattered_to_tp_attn_full = lambda t, fb: t
    mha._use_ag_after_qlora = False
    try:
        yield
    finally:
        for key, value in old.items():
            setattr(mha, key, value)


def _call_prepare(fx):
    return mha.forward_mha_prepare_npu(
        fx["m"],
        fx["positions"],
        fx["hidden"],
        fx["batch"],
        None,
        fx["scatter_modes"],
    )


def run_prepare_route(rows, seed, eps, device, layout="stride1344_offset",
                      offset=64):
    meta = {"rows": rows, "seed": seed, "eps": eps, "layout": layout}
    failures = []
    try:
        fx = build_prepare_fixture(rows, seed, eps, device, layout, offset)
        meta.update(
            {
                "layout_ok": fx["layout_ok"],
                "expected_stride": fx["expected_stride"],
                "expected_offset": fx["expected_offset"],
                "observed_stride": fx["observed_stride"],
                "observed_offset": fx["observed_offset"],
            }
        )
        if not fx["layout_ok"]:
            failures.append("qkv_layout_mismatch")

        fused_calls = {"n": 0}
        recorded = {}
        real_op = qlm._fused_rmsnorm_mxquant_op

        def factory():
            op = real_op()

            def wrapped(*args, **kwargs):
                fused_calls["n"] += 1
                recorded["x"] = args[0] if args else kwargs.get("x")
                return op(*args, **kwargs)

            return wrapped

        qlm._fused_rmsnorm_mxquant_op = factory
        try:
            with prepare_environment(fx):
                with count_op(torch.ops.npu, "npu_dynamic_mx_quant") as dyn:
                    q, k, v, fb = _call_prepare(fx)
                    _sync()
        finally:
            qlm._fused_rmsnorm_mxquant_op = real_op

        meta["fused_op_calls"] = fused_calls["n"]
        meta["extra_dynamic_mx_quant_after_fused"] = dyn["n"]
        if fused_calls["n"] != 1:
            failures.append("fused_route_not_hit")
        if dyn["n"] != 0:
            failures.append("double_quant_after_fused")
        meta["q_dtype"] = str(q.dtype)
        meta["q_shape"] = list(q.shape)
        if q.dtype != torch.bfloat16:
            failures.append("q_not_bf16")
        if tuple(q.shape) != (rows, NUM_HEADS, QK_HEAD):
            failures.append("q_geometry_mismatch")
        if fx["pool"].sets != 1:
            failures.append("kv_cache_write_count")
        q_finite = _finite_stats(q)
        meta["q_finite"] = q_finite
        if q_finite["bad"] != 0:
            failures.append("q_nonfinite")

        recorded_x = recorded.get("x")
        if recorded_x is None:
            failures.append("fused_input_not_recorded")
        else:
            meta["fused_input_contiguous"] = bool(recorded_x.is_contiguous())
            if not recorded_x.is_contiguous():
                failures.append("fused_input_not_contiguous")
            order = byte_match(recorded_x, fx["qkv"][:, :Q_LORA])
            meta["fused_input_vs_qkv_slice"] = order
            if order["mismatch"] != 0:
                failures.append("fused_input_token_order_or_values")
    except Exception as exc:
        failures.append(f"exception={exc!r}")
        meta["traceback"] = traceback.format_exc()
    meta["failures"] = failures
    meta["ok"] = not failures
    return meta


# ---------------------------------------------------------------------------
# timing (no injected gate overrides, no fused recorder)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def fusion_flag(enabled):
    # Measurement-only toggle of the already-cached flag.  Neither route gets
    # an injected compile/capture predicate.
    old = qlm._FUSION_ENABLED
    qlm._FUSION_ENABLED = enabled
    try:
        yield
    finally:
        qlm._FUSION_ENABLED = old


def _event_pair():
    return torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)


def _timing_stats(values):
    ordered = sorted(values)
    return {
        "count": len(values),
        "median_ms": statistics.median(values),
        "mean_ms": statistics.mean(values),
        "p95_ms": ordered[max(0, int(0.95 * len(ordered)) - 1)],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def time_prepare_alternating(rows, seed, eps, device, warmup, iters,
                             layout="stride1344_offset", offset=64):
    """Headline timing: full real forward_mha_prepare_npu route, both flags."""
    fx = build_prepare_fixture(rows, seed, eps, device, layout, offset)
    with prepare_environment(fx):
        for _ in range(warmup):
            with fusion_flag(False):
                _call_prepare(fx)
            with fusion_flag(True):
                _call_prepare(fx)
            _sync()
        old_ms, new_ms, keep = [], [], {"old": None, "new": None}
        for _ in range(iters):
            for name, enabled, bucket in (("old", False, old_ms), ("new", True, new_ms)):
                with fusion_flag(enabled):
                    start, end = _event_pair()
                    start.record()
                    keep[name] = _call_prepare(fx)
                    end.record()
                    _sync()
                bucket.append(start.elapsed_time(end))
    return {
        "kind": "prepare_path_complete_chain",
        "rows": rows,
        "layout": layout,
        "old": _timing_stats(old_ms),
        "new": _timing_stats(new_ms),
        "net_gain_ms_median": _timing_stats(old_ms)["median_ms"]
        - _timing_stats(new_ms)["median_ms"],
        "outputs_retained": sum(v is not None for v in keep.values()),
        "route_equivalence_note": (
            "both arms use the real gate, real RMSNorm module, real MXFP8 "
            "kernel and forward_mha_prepare_npu; KV/RoPE branches are identical "
            "stubs shared by both arms"
        ),
    }


def _chain_kernel_old(qkv, norm, layer):
    q = qkv.split([Q_LORA, QKV_WIDTH - Q_LORA], dim=-1)[0]
    q_norm = norm(q)
    out = layer._mx_kernel.apply(layer, q_norm, None)
    return out.view(-1, OUT_FEATURES)


def _chain_kernel_new(qkv, norm, layer):
    q = qkv.split([Q_LORA, QKV_WIDTH - Q_LORA], dim=-1)[0]
    out = qlm.forward_q_latent_rmsnorm_mxquant(norm, layer, q)
    return out.view(-1, OUT_FEATURES)


def time_kernel_alternating(rows, seed, eps, device, warmup, iters,
                            layout="stride1344_offset", offset=64):
    """Diagnostic: slice+norm/quant+q_b_proj kernel chain without module calls."""
    inp = make_qkv(rows, seed, device, layout=layout, offset=offset)
    qkv = inp["qkv"]
    gamma = make_gamma(seed, eps, device)
    norm = make_norm(Q_LORA, gamma, eps, device)
    layer = make_layer(OUT_FEATURES, seed, device)
    for _ in range(warmup):
        _chain_kernel_old(qkv, norm, layer)
        with fusion_flag(True):
            _chain_kernel_new(qkv, norm, layer)
        _sync()
    old_ms, new_ms, keep = [], [], {"old": None, "new": None}
    for _ in range(iters):
        for name, fn, bucket in (
            ("old", lambda: _chain_kernel_old(qkv, norm, layer), old_ms),
            ("new", lambda: _chain_kernel_new(qkv, norm, layer), new_ms),
        ):
            with fusion_flag(name == "new"):
                start, end = _event_pair()
                start.record()
                keep[name] = fn()
                end.record()
                _sync()
            bucket.append(start.elapsed_time(end))
    return {
        "kind": "kernel_chain_after_slice_diagnostic",
        "rows": rows,
        "layout": layout,
        "old": _timing_stats(old_ms),
        "new": _timing_stats(new_ms),
        "net_gain_ms_median": _timing_stats(old_ms)["median_ms"]
        - _timing_stats(new_ms)["median_ms"],
        "outputs_retained": sum(v is not None for v in keep.values()),
        "note": "diagnostic only; headline is prepare_path_complete_chain",
    }


# ---------------------------------------------------------------------------
# profiling
# ---------------------------------------------------------------------------
def summarize_trace(path):
    try:
        with open(path) as f:
            data = json.load(f)
        events = data.get("traceEvents", [])
        kernels = [
            e
            for e in events
            if e.get("ph") == "X"
            and (
                "kernel" in str(e.get("cat", "")).lower()
                or "npu" in str(e.get("cat", "")).lower()
                or "aic" in str(e.get("cat", "")).lower()
            )
        ]
        if not kernels:
            kernels = [
                e
                for e in events
                if e.get("ph") == "X"
                and e.get("cat")
                not in ("cpu_op", "user_annotation", "python_function")
            ]
        by_name = {}
        for e in kernels:
            item = by_name.setdefault(e.get("name", "?"), {"count": 0, "dur_us": 0.0})
            item["count"] += 1
            item["dur_us"] += float(e.get("dur", 0.0))
        return {
            "path": path,
            "kernel_event_count": len(kernels),
            "kernels": dict(
                sorted(by_name.items(), key=lambda kv: kv[1]["dur_us"], reverse=True)
            ),
        }
    except Exception as exc:
        return {"path": path, "error": repr(exc)}


def _profile(label, fn, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fn()
    _sync()
    trace_path = os.path.join(out_dir, f"trace_{label}.json")
    try:
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            record_shapes=False,
        ) as prof:
            fn()
            _sync()
        try:
            prof.export_chrome_trace(trace_path)
        except Exception:
            trace_path = None
        try:
            table = prof.key_averages().table(
                sort_by="self_npu_time_total", row_limit=80
            )
        except Exception as exc:
            table = f"key_averages unavailable: {exc!r}"
        return {
            "trace": summarize_trace(trace_path) if trace_path else None,
            "table": table,
        }
    except Exception as exc:
        return {"error": repr(exc), "traceback": traceback.format_exc()}


def profile_chains(rows, seed, eps, device, out_dir):
    out = {}
    # Kernel-level evidence: Slice remains; standalone RmsNorm + DynamicMxQuant
    # are replaced by the fused kernel.  No recorder, no injected gate.
    inp = make_qkv(rows, seed, device)
    qkv = inp["qkv"]
    gamma = make_gamma(seed, eps, device)
    norm = make_norm(Q_LORA, gamma, eps, device)
    layer = make_layer(OUT_FEATURES, seed, device)
    out["kernel_old"] = _profile(
        "kernel_old", lambda: _chain_kernel_old(qkv, norm, layer), out_dir
    )
    with fusion_flag(True):
        out["kernel_new"] = _profile(
            "kernel_new", lambda: _chain_kernel_new(qkv, norm, layer), out_dir
        )

    # Complete prepare route: real gate, real module/norm dispatch.
    fx = build_prepare_fixture(rows, seed, eps, device)
    with prepare_environment(fx):
        with fusion_flag(False):
            out["prepare_old"] = _profile(
                "prepare_old", lambda: _call_prepare(fx), out_dir
            )
        with fusion_flag(True):
            out["prepare_new"] = _profile(
                "prepare_new", lambda: _call_prepare(fx), out_dir
            )
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _summary_line(item):
    status = "PASS" if item.get("ok") else "FAIL"
    detail = item.get("failures", [])
    return (
        f"{status} rows={item.get('rows')} seed={item.get('seed')} "
        f"layout={item.get('layout')} {detail if detail else ''}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if os.environ.get("QNORM_MXQUANT_NPU_GRANTED") != "1":
        print(
            "REFUSED: QNORM_MXQUANT_NPU_GRANTED=1 is required. "
            "No NPU is touched without the architect's explicit grant."
        )
        return 0
    _load_runtime()
    if not torch.npu.is_available():
        print("REFUSED: torch.npu.is_available() is False")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    device = "npu:0"
    os.environ.setdefault("SGLANG_NPU_Q_RMSNORM_MXQUANT", "1")
    qlm._FUSION_ENABLED = True

    result = {
        "grant_env": os.environ.get("QNORM_MXQUANT_NPU_GRANTED"),
        "device": torch.npu.get_device_name(0),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "flag": qlm.fusion_enabled(),
        "geometry": {
            "q_lora": Q_LORA,
            "kv_lora": KV_LORA,
            "qk_nope": QK_NOPE,
            "qk_rope": QK_ROPE,
            "qk_head": QK_HEAD,
            "num_heads": NUM_HEADS,
            "q_b_proj_out": OUT_FEATURES,
            "qkv_width": QKV_WIDTH,
        },
        "criteria": CRITERIA,
        "byte_mismatch_explanation": (
            "Old-vs-new FP8/scale byte mismatches are bounded, not ignored: "
            "the bound allows BF16-vs-FP32 norm rounding to cross FP8/scale "
            "boundaries. They are recorded per case alongside dequant and "
            "scale oracle comparisons; systematic layout/semantics errors fail "
            "the oracle checks."
        ),
        "historical_cost_note": (
            "Historical ~0.70ms Slice + ~0.99ms Norm + ~0.67ms Quant "
            "(~2.37ms/47 layers) is the pre-existing cost, not a measured gain "
            "of this harness; only the measured tables are this task's result."
        ),
        "correctness": [],
        "prepare_route": [],
        "timing": [],
        "profiling": {},
        "summary": {},
    }
    exit_code = 1
    try:
        # Correctness, including the real prepare route.
        for rows in TOKENS:
            layouts = [("stride1344_offset", 64)]
            if rows in (129, 3787):
                layouts.append(("padded_stride_offset", 32))
            for layout, offset in layouts:
                seeds = (1234,) if rows > 2048 else (1234, 5678, 91011)
                for seed in seeds:
                    result["correctness"].append(
                        run_case(rows, seed, 1e-6, device, layout, offset)
                    )
        for extra in (
            dict(rows=128, seed=4242, extreme=True),
            dict(rows=128, seed=4243, eps=1e-5),
        ):
            result["correctness"].append(
                run_case(
                    extra["rows"],
                    extra["seed"],
                    extra.get("eps", 1e-6),
                    device,
                    "stride1344_offset",
                    64,
                    extra.get("extreme", False),
                )
            )
        for rows in (128, 3787):
            result["prepare_route"].append(
                run_prepare_route(rows, 1234, 1e-6, device)
            )

        correctness_ok = all(c["ok"] for c in result["correctness"]) and all(
            p["ok"] for p in result["prepare_route"]
        )
        result["summary"]["correctness"] = "PASS" if correctness_ok else "FAIL"
        result["summary"]["prepare_route"] = (
            "PASS" if all(p["ok"] for p in result["prepare_route"]) else "FAIL"
        )
        if not correctness_ok:
            result["summary"]["performance"] = "SKIPPED_CORRECTNESS_FAILED"
            result["summary"]["overall"] = "FAIL"
            print("CORRECTNESS FAILURES — performance not run")
            for item in result["correctness"] + result["prepare_route"]:
                print(_summary_line(item))
            return 1

        # Performance only after correctness passes.  No recorder, no injected
        # gate overrides in either arm.
        timing_ok = True
        for rows in (1, 128, 3787, 15148):
            warmup, iters = (20, 60) if rows <= 3787 else (10, 30)
            try:
                prep = time_prepare_alternating(
                    rows, 1234, 1e-6, device, warmup, iters
                )
                prep["ok"] = True
                result["timing"].append(prep)
                kernel = time_kernel_alternating(
                    rows, 1234, 1e-6, device, warmup, iters
                )
                kernel["ok"] = True
                result["timing"].append(kernel)
            except Exception as exc:
                timing_ok = False
                result["timing"].append(
                    {
                        "rows": rows,
                        "ok": False,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
        try:
            result["profiling"] = profile_chains(
                3787, 1234, 1e-6, device, os.path.join(args.out_dir, "profiling")
            )
            profiling_ok = all(
                "error" not in v for v in result["profiling"].values()
            )
        except Exception as exc:
            profiling_ok = False
            result["profiling"] = {
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        result["summary"]["performance"] = (
            "PASS" if timing_ok and profiling_ok else "FAIL"
        )
        result["summary"]["overall"] = (
            "PASS"
            if result["summary"]["correctness"] == "PASS"
            and result["summary"]["performance"] == "PASS"
            else "FAIL"
        )
        exit_code = 0 if result["summary"]["overall"] == "PASS" else 1
        if exit_code != 0:
            print("PERFORMANCE FAILURES")
            for item in result["timing"]:
                if not item.get("ok", True):
                    print(item)
    except Exception as exc:
        result["exception"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        result["summary"]["overall"] = "FAIL"
        exit_code = 1
    finally:
        path = os.path.join(args.out_dir, "qnorm_mxquant_npu_results.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=1)
        summary_path = os.path.join(args.out_dir, "npu_summary.txt")
        with open(summary_path, "w") as f:
            f.write(json.dumps(result.get("summary", {}), indent=1) + "\n")
            for item in result.get("correctness", []):
                f.write(_summary_line(item) + "\n")
            for item in result.get("prepare_route", []):
                f.write("prepare " + _summary_line(item) + "\n")
        print(json.dumps(result.get("summary", {}), indent=1))
        print(f"RESULTS: {path}")
        print(f"SUMMARY: {summary_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
