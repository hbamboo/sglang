"""Static per-head split of DeepSeek MLA ``kv_b_proj`` for NPU MHA prepare.

The unquantized ``kv_b_proj`` weight is ``[H * (Dk + Dv), K]`` and its output
rows are laid out per head as ``[K_nope(Dk), V(Dv)]``.  The generic MHA prepare
path projects once, views ``[T, H, Dk + Dv]`` and then needs
``V`` as a contiguous ``[T, H, Dv]`` tensor, which costs a strided copy, and a
``K_nope`` source slice for the final K assembly.

This module prepares (once, after weights are loaded) two dense matrices

    Wk = W.view(H, Dk + Dv, K)[:, :Dk, :].reshape(H * Dk, K)
    Wv = W.view(H, Dk + Dv, K)[:, Dk:, :].reshape(H * Dv, K)

(with the bias split the same way), so the current-token NPU MHA prepare can
produce physically contiguous ``[T, H, Dk]`` and ``[T, H, Dv]`` outputs with
two ordinary unquantized linears.  The original module weight is kept intact
for every other consumer (prefix/MHA one-shot, fallbacks, decode).

Only ``ColumnParallelLinear`` + ``UnquantizedLinearMethod`` BF16/FP16 modules
are supported.  Quantized modules, DSA, gather/skip-bias semantics, expanded
KV fusion, and compile/graph-capture executions fall back to the original
single projection.  The feature is off by default behind
``SGLANG_NPU_SPLIT_KV_B_PROJ`` and is prepared only when that gate is set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)

_STATE_ATTR = "_split_kv_b_proj_state"
_W_K_BUF = "_split_kv_b_proj_w_k"
_W_V_BUF = "_split_kv_b_proj_w_v"
_B_K_BUF = "_split_kv_b_proj_b_k"
_B_V_BUF = "_split_kv_b_proj_b_v"
_BUFFER_NAMES = (_W_K_BUF, _W_V_BUF, _B_K_BUF, _B_V_BUF)


@dataclass
class SplitKVBProjState:
    """Prepared weights plus the host metadata used to reject stale buffers."""

    num_heads: int
    k_head_dim: int
    v_head_dim: int
    input_size: int
    weight: torch.Tensor
    weight_data_ptr: int
    weight_version: int
    weight_dtype: torch.dtype
    weight_device: torch.device
    weight_shape: Tuple[int, ...]
    quant_method: object
    bias: Optional[torch.Tensor]
    bias_data_ptr: Optional[int]
    bias_version: Optional[int]
    tp_size: int
    w_k: torch.Tensor
    w_v: torch.Tensor
    b_k: Optional[torch.Tensor]
    b_v: Optional[torch.Tensor]


def is_split_kv_b_proj_enabled() -> bool:
    """Read the opt-in gate.  Called at post-load time, not per forward."""
    from sglang.srt.environ import envs

    return envs.SGLANG_NPU_SPLIT_KV_B_PROJ.get()


def split_kv_b_proj_weight(
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    num_heads: int,
    k_head_dim: int,
    v_head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Split a per-head packed projection without blind row slicing.

    The returned tensors own their storage (``copy_`` into fresh allocations)
    so they never alias the module weight, even when a slice happens to be
    contiguous (e.g. one local head).
    """
    num_heads = int(num_heads)
    k_head_dim = int(k_head_dim)
    v_head_dim = int(v_head_dim)
    head_width = k_head_dim + v_head_dim
    if num_heads <= 0 or k_head_dim <= 0 or v_head_dim <= 0:
        raise ValueError(
            f"invalid split geometry H={num_heads} Dk={k_head_dim} Dv={v_head_dim}"
        )
    if weight.dim() != 2:
        raise ValueError(f"kv_b_proj weight must be 2D, got {tuple(weight.shape)}")
    expected = num_heads * head_width
    if weight.shape[0] != expected:
        raise ValueError(
            f"kv_b_proj weight rows {weight.shape[0]} != H*(Dk+Dv) {expected}"
        )

    in_size = weight.shape[1]
    weight3 = weight.detach().reshape(num_heads, head_width, in_size)
    w_k = weight.new_empty((num_heads * k_head_dim, in_size))
    w_v = weight.new_empty((num_heads * v_head_dim, in_size))
    with torch.no_grad():
        # Copy directly from the strided head slice; no explicit reshape copy.
        w_k.view(num_heads, k_head_dim, in_size).copy_(weight3[:, :k_head_dim, :])
        w_v.view(num_heads, v_head_dim, in_size).copy_(weight3[:, k_head_dim:, :])

    b_k = None
    b_v = None
    if bias is not None:
        if bias.dim() != 1 or bias.numel() != expected:
            raise ValueError(
                f"kv_b_proj bias must be 1D of {expected}, got {tuple(bias.shape)}"
            )
        bias3 = bias.detach().reshape(num_heads, head_width)
        b_k = bias.new_empty((num_heads * k_head_dim,))
        b_v = bias.new_empty((num_heads * v_head_dim,))
        with torch.no_grad():
            b_k.view(num_heads, k_head_dim).copy_(bias3[:, :k_head_dim])
            b_v.view(num_heads, v_head_dim).copy_(bias3[:, k_head_dim:])
    return w_k, w_v, b_k, b_v


def _module_type_name(module: object) -> str:
    return type(module).__name__


def unsupported_reason(
    attn: object,
    module: object,
    *,
    check_device: bool = True,
) -> Optional[str]:
    """Return the first reason this module cannot use the split path."""
    if module is None:
        return "no kv_b_proj module"
    if _module_type_name(module) != "ColumnParallelLinear":
        return f"module type {_module_type_name(module)}"
    quant_method = getattr(module, "quant_method", None)
    if quant_method is None or _module_type_name(quant_method) != (
        "UnquantizedLinearMethod"
    ):
        return f"quant_method {_module_type_name(quant_method)}"
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        return "weight is not a 2D tensor"
    if weight.dtype not in _SUPPORTED_DTYPES:
        return f"dtype {weight.dtype}"
    if check_device and weight.device.type != "npu":
        return f"device {weight.device}"
    if getattr(module, "gather_output", False):
        return "gather_output=True"
    if getattr(module, "skip_bias_add", False):
        return "skip_bias_add=True"
    if getattr(attn, "use_dsa", False):
        return "DSA"

    num_heads = getattr(attn, "num_local_heads", None)
    k_head_dim = getattr(attn, "qk_nope_head_dim", None)
    v_head_dim = getattr(attn, "v_head_dim", None)
    if not isinstance(num_heads, int) or num_heads <= 0:
        return f"num_local_heads {num_heads}"
    if not isinstance(k_head_dim, int) or k_head_dim <= 0:
        return f"qk_nope_head_dim {k_head_dim}"
    if not isinstance(v_head_dim, int) or v_head_dim <= 0:
        return f"v_head_dim {v_head_dim}"
    expected = num_heads * (k_head_dim + v_head_dim)
    if weight.shape[0] != expected:
        return f"weight rows {weight.shape[0]} != local H*(Dk+Dv) {expected}"
    local_out = getattr(module, "output_size_per_partition", None)
    if local_out is not None and local_out != weight.shape[0]:
        return (
            f"output_size_per_partition {local_out} != weight rows "
            f"{weight.shape[0]}"
        )
    module_input = getattr(module, "input_size", weight.shape[1])
    if module_input != weight.shape[1]:
        return f"input_size {module_input} != weight cols {weight.shape[1]}"
    kv_lora_rank = getattr(attn, "kv_lora_rank", None)
    if kv_lora_rank is not None and weight.shape[1] != kv_lora_rank:
        return f"weight cols {weight.shape[1]} != kv_lora_rank {kv_lora_rank}"

    bias = getattr(module, "bias", None)
    if bias is not None:
        if not isinstance(bias, torch.Tensor) or bias.dim() != 1:
            return "bias is not a 1D tensor"
        if bias.numel() != weight.shape[0]:
            return f"bias numel {bias.numel()} != weight rows {weight.shape[0]}"
        if bias.dtype != weight.dtype:
            return f"bias dtype {bias.dtype} != weight dtype {weight.dtype}"
        if check_device and bias.device != weight.device:
            return f"bias device {bias.device} != weight device {weight.device}"
    return None


def _set_buffer(module: torch.nn.Module, name: str, value: Optional[torch.Tensor]):
    if name in getattr(module, "_buffers", {}):
        module._buffers[name] = value
    elif value is not None:
        module.register_buffer(name, value, persistent=False)


def invalidate_split_kv_b_proj(module: object) -> None:
    """Drop prepared buffers so a later forward cannot use stale weights."""
    if module is None or not isinstance(module, torch.nn.Module):
        return
    setattr(module, _STATE_ATTR, None)
    for name in _BUFFER_NAMES:
        _set_buffer(module, name, None)


def _state_matches_attn(state: SplitKVBProjState, attn: object) -> bool:
    return (
        state.num_heads == getattr(attn, "num_local_heads", None)
        and state.k_head_dim == getattr(attn, "qk_nope_head_dim", None)
        and state.v_head_dim == getattr(attn, "v_head_dim", None)
    )


def _state_is_current(state: SplitKVBProjState, module: object) -> bool:
    """Cheap host-metadata check; never reads device tensor values."""
    weight = getattr(module, "weight", None)
    if weight is not state.weight:
        return False
    if weight.data_ptr() != state.weight_data_ptr:
        return False
    if weight._version != state.weight_version:
        return False
    if weight.dtype != state.weight_dtype or weight.device != state.weight_device:
        return False
    if tuple(weight.shape) != state.weight_shape:
        return False
    if getattr(module, "quant_method", None) is not state.quant_method:
        return False
    if getattr(module, "gather_output", False) or getattr(
        module, "skip_bias_add", False
    ):
        return False
    bias = getattr(module, "bias", None)
    if bias is not state.bias:
        return False
    if bias is not None:
        if (
            bias.data_ptr() != state.bias_data_ptr
            or bias._version != state.bias_version
            or bias.dtype != state.weight_dtype
            or bias.device != state.weight_device
        ):
            return False
    for tensor in (state.w_k, state.w_v, state.b_k, state.b_v):
        if tensor is not None and (
            tensor.device != state.weight_device or tensor.dtype != state.weight_dtype
        ):
            return False
    return True


def prepare_split_kv_b_proj(
    attn: object,
    module: object,
    *,
    force: bool = False,
) -> Tuple[Optional[SplitKVBProjState], str]:
    """Prepare (or refresh) the split buffers for one attention module.

    ``force`` is for tests only.  Production callers use the env gate.
    """
    if not force and not is_split_kv_b_proj_enabled():
        return None, "env disabled"

    existing = getattr(module, _STATE_ATTR, None) if module is not None else None
    if (
        existing is not None
        and _state_is_current(existing, module)
        and _state_matches_attn(existing, attn)
    ):
        return existing, "already prepared"

    reason = unsupported_reason(attn, module)
    if reason is not None:
        invalidate_split_kv_b_proj(module)
        logger.debug("SGLANG_NPU_SPLIT_KV_B_PROJ unsupported: %s", reason)
        return None, reason

    weight = module.weight
    bias = getattr(module, "bias", None)
    w_k, w_v, b_k, b_v = split_kv_b_proj_weight(
        weight,
        bias,
        attn.num_local_heads,
        attn.qk_nope_head_dim,
        attn.v_head_dim,
    )
    state = SplitKVBProjState(
        num_heads=attn.num_local_heads,
        k_head_dim=attn.qk_nope_head_dim,
        v_head_dim=attn.v_head_dim,
        input_size=weight.shape[1],
        weight=weight,
        weight_data_ptr=weight.data_ptr(),
        weight_version=weight._version,
        weight_dtype=weight.dtype,
        weight_device=weight.device,
        weight_shape=tuple(weight.shape),
        quant_method=module.quant_method,
        bias=bias,
        bias_data_ptr=bias.data_ptr() if bias is not None else None,
        bias_version=bias._version if bias is not None else None,
        tp_size=int(getattr(module, "tp_size", 1)),
        w_k=w_k,
        w_v=w_v,
        b_k=b_k,
        b_v=b_v,
    )
    _set_buffer(module, _W_K_BUF, w_k)
    _set_buffer(module, _W_V_BUF, w_v)
    _set_buffer(module, _B_K_BUF, b_k)
    _set_buffer(module, _B_V_BUF, b_v)
    setattr(module, _STATE_ATTR, state)
    logger.debug(
        "SGLANG_NPU_SPLIT_KV_B_PROJ prepared H=%d Dk=%d Dv=%d K=%d dtype=%s",
        state.num_heads,
        state.k_head_dim,
        state.v_head_dim,
        state.input_size,
        weight.dtype,
    )
    return state, "prepared"


def _capture_or_compile_active() -> bool:
    try:
        if torch.compiler.is_compiling():
            return True
    except Exception:
        return True
    npu = getattr(torch, "npu", None)
    capture_fn = (
        getattr(npu, "is_current_stream_capturing", None) if npu is not None else None
    )
    if capture_fn is not None:
        try:
            if capture_fn():
                return True
        except Exception:
            return True
    return False


def _sp_active_needs_old_path(module: object, state: SplitKVBProjState) -> bool:
    if state.tp_size <= 1:
        return False
    try:
        from sglang.srt.layers.communicator import get_forward

        return bool(get_forward().sp_active)
    except Exception:
        # TP>1 without a verifiable SP context: do not bypass ColumnParallelLinear.
        return True


def _expanded_kv_may_fuse(forward_batch: object) -> bool:
    if forward_batch is None:
        return False
    try:
        from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

        pool = get_token_to_kv_pool()
    except Exception:
        return True
    if getattr(pool, "prefill_expanded_geometry", None) is None:
        return False
    try:
        return bool(forward_batch.forward_mode.is_extend())
    except Exception:
        return True


def maybe_forward_split_kv_b_proj(
    attn: object,
    module: object,
    kv_a: torch.Tensor,
    forward_batch: object = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return contiguous ``(k_nope[T,H,Dk], v[T,H,Dv])`` or ``None``.

    ``None`` means the caller must run the original one-matmul path.
    """
    state = getattr(module, _STATE_ATTR, None) if module is not None else None
    if state is None:
        return None
    if not _state_is_current(state, module):
        return None
    if not _state_matches_attn(state, attn):
        return None
    if getattr(attn, "use_dsa", False):
        return None
    if _capture_or_compile_active():
        return None
    if _sp_active_needs_old_path(module, state):
        return None
    if _expanded_kv_may_fuse(forward_batch):
        return None
    if not isinstance(kv_a, torch.Tensor):
        return None
    if kv_a.dim() < 1 or kv_a.shape[-1] != state.input_size:
        return None
    if kv_a.dtype != state.weight_dtype or kv_a.device != state.weight_device:
        return None

    k_nope = torch.nn.functional.linear(kv_a, state.w_k, state.b_k)
    v = torch.nn.functional.linear(kv_a, state.w_v, state.b_v)
    k_nope = k_nope.view(-1, state.num_heads, state.k_head_dim)
    v = v.view(-1, state.num_heads, state.v_head_dim)
    return k_nope, v


def split_kv_b_proj_state(module: object) -> Optional[SplitKVBProjState]:
    """Diagnostic accessor used by tests/reports."""
    return getattr(module, _STATE_ATTR, None)
