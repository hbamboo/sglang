"""Opt-in native write of the unified 576-wide latent cache.

The legacy route calls ``torch_npu.npu_kv_rmsnorm_rope_cache`` with the two
non-contiguous slices ``kv_buffer[layer][..., :512]`` and
``kv_buffer[layer][..., 512:]``.  On Ascend that op packs both slices into
contiguous temporaries, runs the fused RMSNorm+RoPE kernel, then copies the
results back (two whole-pool ``Slice`` + two whole-pool ``ViewCopy`` kernels),
which costs ~1.5 ms per layer for a 7194x128 pool regardless of token count.

This helper instead hands the real unified ``[pages, 128, 1, 576]`` (or packed
``[rows, 1, 576]``) layer view to ``torch.ops.npu.kv_rmsnorm_rope_cache``,
which writes latent at ``slot*576`` and rope at ``slot*576+512`` for the
touched slots only.  The view's ``data_ptr`` already includes the layer storage
offset, so no layer arithmetic against the raw pool is performed.

The route is deliberately narrow and defaults off:

  * ``SGLANG_NPU_KV_RMSNORM_UNIFIED_CACHE=1`` is required;
  * DeepSeek-yarn MHA (not DSA), PA_BNSD ND only (never PA_NZ);
  * latent512 / rope64, BF16 or FP16 payload, page size 128;
  * the pool must expose a packed 576-wide per-layer 576 view (``unified_kv``);
  * the isolated kernel library must already be loaded;
  * never taken under torch.compile/piecewise compilation or NPU graph capture
    (auto_functionalized/allocator behaviour there has not been profiled; the
    original route is kept so graph capture remains correct).

The legacy path reached the cache through ``pool.get_kv_buffer``, which first
performs ``layer_transfer_counter.wait_until(layer_id - start_layer)`` under
HiCache/async layer transfer.  This helper reads ``pool.kv_buffer`` directly, so
once every narrow gate has passed (and before dispatching the new op) it calls
the same public ``pool.get_kv_buffer(layer_id)`` and discards the two returned
split views.  That call preserves all of the getter's synchronization side
effects (and the getter computes the start_layer offset once); returning two
metadata views does not run any CANN contiguous copy.  The already-validated
packed 576-wide layer view is what gets passed to the kernel, and its
``data_ptr`` already contains the layer storage offset.  Unsupported paths
return before this getter call, so the original route owns its wait exactly
once.  If a pool has no public getter we fall back rather than bypass the
transfer contract.

Everything else returns ``None`` and the caller keeps the original route.
"""

import functools
import os
from typing import Optional, Tuple

import torch

_ENV = "SGLANG_NPU_KV_RMSNORM_UNIFIED_CACHE"
_OP_NAME = "kv_rmsnorm_rope_cache"
_KV_LATENT = 512
_KV_ROPE = 64
_KV_WIDTH = 576
_PAGE = 128


def _env_enabled() -> bool:
    try:
        from sglang.srt.utils import get_bool_env_var

        return get_bool_env_var(_ENV, "false")
    except Exception:
        return os.environ.get(_ENV, "0").strip().lower() in ("1", "true", "yes", "on")


@functools.lru_cache(maxsize=1)
def _gate_enabled() -> bool:
    return _env_enabled()


def clear_gate_cache() -> None:
    """Test hook: re-read the environment switch."""
    _gate_enabled.cache_clear()


def _graph_or_compile_active() -> bool:
    try:
        if torch.compiler.is_compiling():
            return True
    except Exception:
        pass
    try:
        capture = getattr(torch.npu, "is_current_stream_capturing", None)
        if capture is not None and capture():
            return True
    except Exception:
        pass
    return False


def _op_loaded() -> bool:
    try:
        return hasattr(torch.ops.npu, _OP_NAME)
    except Exception:
        return False


def unified_layer_cache_view(pool, layer_id: int):
    """Return the packed 576-wide per-layer view, or None if unsupported.

    Accepted shapes (strictly packed, no page gap):
      * [pages, 128, 1, 576] with strides (73728, 576, 576, 1)
      * [rows, 1, 576]      with strides (576, 576, 1)
    """
    kv_buffer = getattr(pool, "kv_buffer", None)
    if kv_buffer is None:
        return None
    local = int(layer_id - getattr(pool, "start_layer", 0))
    try:
        if isinstance(kv_buffer, (list, tuple)):
            if local < 0 or local >= len(kv_buffer):
                return None
            view = kv_buffer[local]
        else:
            if not hasattr(kv_buffer, "shape"):
                return None
            if local < 0 or local >= kv_buffer.shape[0]:
                return None
            view = kv_buffer[local]
    except Exception:
        return None
    if not isinstance(view, torch.Tensor) or view.device.type != "npu":
        return None
    if view.dtype not in (torch.bfloat16, torch.float16):
        return None
    if view.stride(-1) != 1:
        return None
    if view.dim() == 4:
        if tuple(view.shape[1:]) != (_PAGE, 1, _KV_WIDTH):
            return None
        if view.stride(1) != _KV_WIDTH or view.stride(2) != _KV_WIDTH:
            return None
        if view.stride(0) != _PAGE * _KV_WIDTH:
            return None
    elif view.dim() == 3:
        if tuple(view.shape[1:]) != (1, _KV_WIDTH):
            return None
        if view.stride(0) != _KV_WIDTH:
            return None
    else:
        return None
    return view


def _invoke_new_op(kv4, gamma, cos, sin, index, cache, k_out, v_out, epsilon):
    """Single dispatch point so tests can observe that the new route was chosen."""
    getattr(torch.ops.npu, _OP_NAME)(kv4, gamma, cos, sin, index, cache, k_out, v_out, epsilon)


def maybe_unified_kv_rmsnorm_rope_cache(
    m,
    latent_cache: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    out_cache_loc: torch.Tensor,
    pool,
    fia_nz: bool,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Try the unified-576 direct-cache route.

    Returns ``(k_rope, c_kv)`` shaped ``[T,1,1,64]`` / ``[T,1,1,512]``, or None
    when any precondition fails (caller must use the original operator).
    """
    if not _gate_enabled() or not _op_loaded() or _graph_or_compile_active():
        return None
    if fia_nz:
        return None
    if bool(getattr(m, "use_dsa", False)):
        return None
    if getattr(m, "kv_lora_rank", None) != _KV_LATENT or getattr(m, "qk_rope_head_dim", None) != _KV_ROPE:
        return None
    if not bool(getattr(pool, "unified_kv", False)):
        return None

    dtype = latent_cache.dtype
    if dtype not in (torch.bfloat16, torch.float16):
        return None
    if latent_cache.dim() != 3 or latent_cache.shape[1] != 1 or latent_cache.shape[2] != _KV_WIDTH:
        return None
    if latent_cache.stride(-1) != 1:
        return None
    tokens = latent_cache.shape[0]
    if cos.shape != (tokens, 1, 1, _KV_ROPE) or sin.shape != (tokens, 1, 1, _KV_ROPE):
        return None
    if cos.dtype != dtype or sin.dtype != dtype or cos.stride(-1) != 1 or sin.stride(-1) != 1:
        return None
    if out_cache_loc.dim() != 1 or out_cache_loc.numel() != tokens:
        return None
    if out_cache_loc.dtype not in (torch.int64, torch.int32):
        return None

    layer_view = unified_layer_cache_view(pool, m.layer_id)
    if layer_view is None or layer_view.dtype != dtype or layer_view.device != latent_cache.device:
        return None
    weight = getattr(m.kv_a_layernorm, "weight", None)
    if weight is None or weight.dtype != dtype or weight.numel() != _KV_LATENT:
        return None
    if weight.device != latent_cache.device:
        return None
    eps = getattr(m.kv_a_layernorm, "variance_epsilon", None)
    if eps is None:
        return None

    # Preserve the public getter's cross-layer transfer dependency (HiCache /
    # async layer transfer) immediately before the direct write.  The old MHA
    # path inherited this wait through get_kv_buffer; simply reading kv_buffer
    # would bypass it.  The returned 512/64 views are metadata only and are
    # intentionally discarded; no CANN contiguous/copy kernel is involved.
    # Unsupported paths returned above, so fallback never waits twice and a
    # pool without a transfer counter keeps its normal path.
    get_kv_buffer = getattr(pool, "get_kv_buffer", None)
    if get_kv_buffer is None:
        return None
    get_kv_buffer(m.layer_id)

    kv4 = latent_cache.unsqueeze(2)  # [T,1,1,576]; row stride is preserved
    index = out_cache_loc if out_cache_loc.dtype == torch.int64 else out_cache_loc.to(torch.int64)
    k_out = torch.empty(tokens, 1, 1, _KV_ROPE, dtype=dtype, device=latent_cache.device)
    v_out = torch.empty(tokens, 1, 1, _KV_LATENT, dtype=dtype, device=latent_cache.device)
    _invoke_new_op(kv4, weight, cos, sin, index, layer_view, k_out, v_out, float(eps))
    return k_out, v_out
