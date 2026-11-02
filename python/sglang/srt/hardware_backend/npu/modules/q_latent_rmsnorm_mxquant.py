"""Fused Q-latent RMSNorm + DynamicMxQuant for Ascend MHA prefill.

Baseline Q-latent path (``forward_mha_prepare_npu``)::

    q, latent_cache = qkv_latent.split(
        [q_lora_rank, kv_lora_rank + qk_rope_head_dim], dim=-1
    )
    q = q_a_layernorm(q)       # BF16 RMSNorm, original gamma/epsilon
    q = q_b_proj(q)[0]         # MXFP8 linear: activation DynamicMxQuant + GEMM

This helper keeps the preceding ``split`` untouched and replaces only the
RMSNorm + activation quantisation pair with the fused Ascend kernel
``npu_rms_norm_dynamic_mx_quant``.  The resulting ``(values, e8m0_scale)`` pair
is the pre-quantised activation protocol already accepted by
``NPUMXFP8LinearMethod.apply``, so the ``q_b_proj`` GEMM consumes it without
running ``npu_dynamic_mx_quant`` a second time.  ``q_b_proj`` still emits the
original BF16 attention dtype; FP8 never reaches the RoPE/cache code.

The route is deliberately narrow and default-off.  Every unsupported case
returns ``None`` so the caller executes the original BF16 path verbatim.  Gate
order is safety-critical: flag, fail-closed compile guard, input device check
(CPU/meta must fall back before any device probe), fail-closed stream-capture
guard, then the remaining shape/norm/consumer metadata.  A CPU/meta probe can
therefore never initialise an NPU, and a failed compile/capture probe falls
back instead of enabling an unverified route.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

FLAG_ENV = "SGLANG_NPU_Q_RMSNORM_MXQUANT"
SUPPORTED_Q_LORA_WIDTH = 768

# Read the environment once at import time.  The flag is fixed for the life of
# the process (set it before starting the server); the layer hot path must not
# re-read ``os.environ`` per token or per layer.
_FUSION_ENABLED = os.environ.get(FLAG_ENV, "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def fusion_enabled() -> bool:
    return _FUSION_ENABLED


def _compiler_is_compiling() -> bool:
    # Do not swallow failures here: the gate treats a failed compile probe as
    # unsupported and falls back (fail closed).
    return bool(torch.compiler.is_compiling())


def _device_is_capturing() -> bool:
    # torch_npu is imported lazily so this module (and its CPU contract tests)
    # stays importable without initialising an Ascend device.  A failed probe
    # must fall back; the caller handles the exception.
    import torch_npu  # noqa: F401

    return bool(torch.npu.is_current_stream_capturing())


def _input_is_npu(x: torch.Tensor) -> bool:
    """Isolated device predicate.

    CPU contract tests patch this predicate to simulate an NPU tensor; the real
    predicate is used to prove that CPU/meta inputs fall back before the
    stream-capture probe can initialise a device.
    """
    device = getattr(x, "device", None)
    return device is not None and device.type == "npu"


def _same_device(a, b) -> bool:
    """Compare torch.device (or device-like) values by type and index."""
    return (a.type, a.index) == (b.type, b.index)


def _consumer_forward_blocker(proj) -> Optional[str]:
    """Only the plain ColumnParallelLinear forward may be bypassed.

    Directly calling ``quant_method.apply`` is equivalent to
    ``ColumnParallelLinear.forward`` only when the layer is that exact class,
    has no per-instance/class forward override, is TP1, does not gather its
    output, and has no bias/skip-bias behavior.  Resolved lazily and only after
    the flag/compile/device/capture gates, so default-off startup pays nothing
    and import cycles are avoided.
    """
    try:
        from sglang.srt.layers.linear import ColumnParallelLinear
    except Exception:
        return "consumer_class_import_error"
    if type(proj) is not ColumnParallelLinear:
        return "proj_not_plain_column_parallel"
    if "forward" in getattr(proj, "__dict__", {}):
        return "proj_instance_forward_override"
    if getattr(type(proj), "forward", None) is not ColumnParallelLinear.forward:
        return "proj_forward_override"
    if getattr(proj, "skip_bias_add", False):
        return "proj_skip_bias_add"
    if getattr(proj, "bias", None) is not None:
        return "proj_bias"
    if getattr(proj, "parametrizations", None):
        return "proj_parametrized"
    return None


def _compiler_guard(is_compiling) -> Optional[str]:
    # Fail closed: an exception means we cannot prove the route is eager.
    try:
        return "compiling" if is_compiling() else None
    except Exception:
        return "compiler_guard_error"


def _capture_guard(is_capturing) -> Optional[str]:
    # Fail closed: an exception means we cannot prove capture is inactive.
    try:
        return "capturing" if is_capturing() else None
    except Exception:
        return "capture_guard_error"


def _fused_rmsnorm_mxquant_op():
    # Resolved lazily and through this indirection so CPU contract tests can
    # substitute a recorder without importing/initialising torch_npu.
    import torch_npu  # noqa: F401

    return torch_npu.npu_rms_norm_dynamic_mx_quant


def _has_forward_hooks(module: torch.nn.Module) -> bool:
    # The fused route calls the kernel and the quant method directly, so a
    # module-level forward hook would silently stop running.  Fall back instead.
    return bool(getattr(module, "_forward_hooks", None)) or bool(
        getattr(module, "_forward_pre_hooks", None)
    )


def _norm_weight_data(norm: torch.nn.Module) -> Optional[torch.Tensor]:
    weight = getattr(norm, "weight", None)
    if weight is None:
        return None
    return getattr(weight, "data", weight)


def fusion_blocker(
    norm: torch.nn.Module,
    proj: torch.nn.Module,
    x: torch.Tensor,
    *,
    is_compiling=None,
    is_capturing=None,
) -> Optional[str]:
    """Return a short reason when the fused route must not run, else ``None``.

    ``is_compiling``/``is_capturing`` are injectable only for CPU tests; in the
    real route they default to the framework/device queries.
    """
    if not _FUSION_ENABLED:
        return "flag_disabled"
    if is_compiling is None:
        is_compiling = _compiler_is_compiling
    if is_capturing is None:
        is_capturing = _device_is_capturing

    # 1) Compile guard.  Fail closed.
    blocker = _compiler_guard(is_compiling)
    if blocker is not None:
        return blocker

    # 2) Safe input device metadata.  A CPU/meta tensor must fall back here,
    # BEFORE the capture probe, so a CPU probe can never initialise an NPU.
    if not _input_is_npu(x):
        return "input_not_npu"

    # 3) Capture guard, only for real NPU input.  Fail closed.
    blocker = _capture_guard(is_capturing)
    if blocker is not None:
        return blocker

    # 4) Remaining input metadata.
    if x.dim() != 2:
        return "input_not_2d"
    if x.dtype != torch.bfloat16:
        return "input_not_bfloat16"
    if x.shape[-1] != SUPPORTED_Q_LORA_WIDTH:
        return "q_lora_width_not_768"
    if x.numel() == 0:
        return "empty_input"

    # 5) Norm metadata.
    if not getattr(norm, "has_weight", False):
        return "norm_without_weight"
    norm_weight = getattr(norm, "weight", None)
    if norm_weight is None or norm_weight.dtype != x.dtype:
        return "norm_weight_dtype"
    if not _same_device(norm_weight.device, x.device):
        return "norm_device_mismatch"
    if getattr(norm, "variance_size_override", None) is not None:
        return "norm_variance_override"
    if getattr(norm, "cast_x_before_out_mul", False):
        return "norm_hf_cast_semantics"
    # BaseFusedOp resolves ``_forward_method`` on first use; bypassing it is
    # only equivalent while it is still the plain NPU forward (residual=None
    # case).  A forced native / backend / compile-mode dispatch must run.
    forward_method = getattr(norm, "_forward_method", None)
    if forward_method is not None and getattr(
        forward_method, "__name__", None
    ) != "forward_npu":
        return "norm_custom_forward"
    if getattr(norm, "is_torch_compile", False):
        return "norm_compile_mode"
    if _has_forward_hooks(norm):
        return "norm_forward_hooks"

    # 6) Consumer metadata.
    if getattr(proj, "tp_size", 1) != 1:
        return "proj_tp"
    if getattr(proj, "gather_output", False):
        return "proj_gather_output"
    if _has_forward_hooks(proj):
        return "proj_forward_hooks"
    if getattr(proj, "_compiled_call_impl", None) is not None:
        return "proj_compiled"

    # Producer/consumer contract set by NPUMXFP8LinearMethod.  This is the
    # only consumer that can take the pre-quantised tuple without quantising
    # again; plain BF16/unquantised linears must stay on the original path.
    if not getattr(proj, "_sgl_npu_mx_quant_fusable", False):
        return "proj_not_mxfp8_fusable"
    blocker = _consumer_forward_blocker(proj)
    if blocker is not None:
        return blocker
    quant_method = getattr(proj, "quant_method", None)
    if quant_method is None or not callable(getattr(quant_method, "apply", None)):
        return "proj_no_quant_apply"
    proj_weight = getattr(proj, "weight", None)
    if proj_weight is None or proj_weight.dtype != torch.float8_e4m3fn:
        return "proj_weight_not_mxfp8"
    if not _same_device(proj_weight.device, x.device):
        return "proj_device_mismatch"
    if proj_weight.dim() != 2 or proj_weight.shape[0] != x.shape[-1]:
        return "proj_width_mismatch"
    if getattr(proj, "weight_scale_inv", None) is None:
        return "proj_missing_weight_scale_inv"
    return None


def forward_q_latent_rmsnorm_mxquant(
    norm: torch.nn.Module,
    proj: torch.nn.Module,
    x: torch.Tensor,
    *,
    is_compiling=None,
    is_capturing=None,
) -> Optional[torch.Tensor]:
    """Run fused norm+quant followed by ``proj``; return BF16 output or None.

    ``x`` is the q-latent view produced by the caller's ``split`` (typically
    stride ``kv_lora_rank + qk_rope_head_dim``).  ``None`` means "unsupported,
    run the original path"; the caller owns that fallback.
    """
    if (
        fusion_blocker(
            norm,
            proj,
            x,
            is_compiling=is_compiling,
            is_capturing=is_capturing,
        )
        is not None
    ):
        return None

    # Preserve the caller's split.  ``contiguous()`` is a no-op for an already
    # contiguous input and performs the single materialisation of the strided
    # q-latent view; the ACLNN entry would otherwise do the same copy itself.
    x = x.contiguous()

    fused = _fused_rmsnorm_mxquant_op()
    # Outputs are (values, mxscale, rstd).  scale_alg/round_mode match the
    # baseline npu_dynamic_mx_quant defaults used by NPUMXFP8LinearMethod.
    # dst_type must be the E4M3FN scalar type (296 is FP4 in this torch_npu).
    outputs = fused(
        x,
        _norm_weight_data(norm),
        beta=None,
        epsilon=norm.variance_epsilon,
        scale_alg=0,
        round_mode="rint",
        dst_type=torch.float8_e4m3fn,
    )
    quant, scale = outputs[0], outputs[1]

    bias = None if getattr(proj, "skip_bias_add", False) else getattr(
        proj, "bias", None
    )
    # Pre-quantised tuple: the consumer skips its own DynamicMxQuant.  The
    # verified tuple contract fixes the output dtype to BF16.
    return proj.quant_method.apply(proj, (quant, scale), bias)
