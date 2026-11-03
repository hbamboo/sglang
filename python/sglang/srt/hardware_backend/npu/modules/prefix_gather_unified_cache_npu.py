"""Opt-in page-first gather of the unified 576-wide MLA prefix cache.

The GLM-4.7-Flash prefill prefix branches in ``ascend_backend.py`` used to read
the historical KV through the two public field views::

    k_buffer = pool.get_key_buffer(layer_id)      # [pages, 128, 1, 512] view
    v_buffer = pool.get_value_buffer(layer_id)    # [pages, 128, 1,  64] view
    kv_cached = torch.index_select(k_buffer, 0, prefix_pages)
    k_rope    = torch.index_select(v_buffer, 0, prefix_pages).flatten(0, 1)

Those two views are non-contiguous 512/64 slices of the packed 576-wide
unified cache, so Ascend's ``index_select`` first materializes EACH WHOLE
per-layer field pool through a ``Slice`` kernel and only then gathers the
selected pages.  For a 7133-page pool that is 7133*128*512 and 7133*128*64
elements of dead traffic per layer pair even when only 20-30 pages are
selected.

This helper gathers the pages ONCE from the current layer's contiguous
``[pages, 128, 1, 576]`` view (or its packed ``[rows, 1, 576]`` form, reshaped
to pages) and only then materializes the selected 512/64 fields::

    selected = index_select(layer_view, 0, prefix_pages)   # [S, 128, 1, 576]
    kv_cached      = selected[..., :512].contiguous()      # [S, 128, 1, 512]
    k_rope_cached  = selected[..., 512:].contiguous().flatten(0, 1)

Page order, duplicates, empty selections, dtype/device, ``start_layer`` and
the view's storage offset are all preserved.  The caller's downstream
``kv_b_proj`` / FIA / ring-MLA code is untouched.

The route is deliberately narrow and defaults off:

  * ``SGLANG_NPU_UNIFIED_PREFIX_GATHER=1`` is required;
  * the pool must be the NPU unified MLA pool (``pool.unified_kv``) with
    latent512 / rope64, page size 128, and ``store_dtype == dtype``
    (BF16 or FP16, so no quantized byte store);
  * no DSA, no expanded prefill KV geometry, no PA_NZ layout;
  * never taken under torch.compile/piecewise compilation or NPU graph
    capture (the original route is kept so capture stays correct);
  * the pool must expose the strict packed 576-wide per-layer view validated by
    ``unified_layer_cache_view`` and a public ``get_kv_buffer`` getter.

The legacy path inherits HiCache / async-layer-transfer readiness from
``get_key_buffer``/``get_value_buffer``.  The packed view read directly from
``pool.kv_buffer`` does NOT wait, so after every gate has passed this helper
calls the public ``pool.get_kv_buffer(layer_id)`` once and discards the two
returned metadata views, exactly as the unified-cache writer module does.  That
preserves the transfer-counter wait before the raw gather.  ``get_kv_buffer``
on the NPU MLA pool waits once and slices only (no CANN copy).  Unsupported
paths return before that call, so the original route owns its own wait exactly
once and nothing waits twice.

Genuine runtime errors from the gather/slice are NOT caught: a device failure
must surface rather than silently fall back.
"""

import functools
import os
from typing import Optional, Tuple

import torch

from sglang.srt.hardware_backend.npu.modules.kv_rmsnorm_unified_cache_npu import (
    _graph_or_compile_active,
    unified_layer_cache_view,
)

_ENV = "SGLANG_NPU_UNIFIED_PREFIX_GATHER"
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


def _pa_nz_active() -> bool:
    """PA_NZ layouts and the QKV-Norm-RoPE PA_NZ writer are unsupported."""
    try:
        from sglang.srt.utils import get_bool_env_var

        return get_bool_env_var("SGLANG_USE_FIA_NZ") or get_bool_env_var(
            "ENABLE_QKV_NORM_ROPE_OPT"
        )
    except Exception:
        # Cannot establish the layout invariant -> stay on the original route.
        return True


def _pages_view(layer_view: torch.Tensor) -> Optional[torch.Tensor]:
    """Return a contiguous ``[pages, 128, 1, 576]`` view, or None.

    ``unified_layer_cache_view`` has already validated the exact stride
    patterns, so the rank-3 ``[rows, 1, 576]`` form reshapes without a copy.
    """
    if layer_view.dim() == 4:
        return layer_view
    rows = layer_view.shape[0]
    if rows % _PAGE != 0:
        return None
    return layer_view.reshape(rows // _PAGE, _PAGE, 1, _KV_WIDTH)


def _invoke_gather(layer_pages: torch.Tensor, page_indices: torch.Tensor):
    """Single dispatch point so tests can observe that the new route was chosen."""
    return torch.index_select(layer_pages, 0, page_indices)


def maybe_gather_unified_prefix_kv(
    pool, layer_id: int, page_indices: torch.Tensor
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Gather selected prefix pages from the packed 576-wide layer cache.

    Returns ``(kv_cached, k_rope_cached)`` in the original caller shapes
    ``[S, 128, 1, 512]`` and ``[S*128, 1, 64]``, or None when any precondition
    fails (the caller must use the original two-field ``index_select`` route).
    """
    # -- cheap, side-effect-free gates (reject before the transfer wait) --
    if not _gate_enabled():
        return None
    if _graph_or_compile_active():
        return None
    if _pa_nz_active():
        return None
    if not isinstance(page_indices, torch.Tensor):
        return None
    if page_indices.dim() != 1:
        return None
    if page_indices.dtype not in (torch.int32, torch.int64):
        return None
    if not bool(getattr(pool, "unified_kv", False)):
        return None
    if bool(getattr(pool, "use_dsa", False)):
        return None
    if getattr(pool, "kv_lora_rank", None) != _KV_LATENT:
        return None
    if getattr(pool, "qk_rope_head_dim", None) != _KV_ROPE:
        return None
    if getattr(pool, "page_size", None) != _PAGE:
        return None
    if getattr(pool, "prefill_expanded_geometry", None) is not None:
        return None
    dtype = getattr(pool, "dtype", None)
    if dtype is None or getattr(pool, "store_dtype", None) != dtype:
        return None
    if dtype not in (torch.bfloat16, torch.float16):
        return None

    layer_view = unified_layer_cache_view(pool, layer_id)
    if layer_view is None:
        return None
    if layer_view.dtype != dtype or layer_view.device != page_indices.device:
        return None
    layer_pages = _pages_view(layer_view)
    if layer_pages is None:
        return None

    get_kv_buffer = getattr(pool, "get_kv_buffer", None)
    if get_kv_buffer is None:
        return None

    # Preserve the public getter's cross-layer transfer dependency (HiCache /
    # async layer transfer) immediately before the raw gather.  The returned
    # 512/64 views are metadata only and are intentionally discarded; no CANN
    # contiguous/copy kernel is involved.  Every unsupported path returned
    # above, so the fallback never waits twice.
    get_kv_buffer(layer_id)

    selected = _invoke_gather(layer_pages, page_indices)
    kv_cached = selected[..., :_KV_LATENT].contiguous()
    k_rope_cached = selected[..., _KV_LATENT:].contiguous().flatten(0, 1)
    return kv_cached, k_rope_cached
