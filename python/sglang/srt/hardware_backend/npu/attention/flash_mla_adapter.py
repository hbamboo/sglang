"""FlashMLA adapter for the Ascend NPU attention backend.

``cann_ops_transformer.flash_mla`` (op name ``FlashMla``) is the Ascend
950PR / 950DT dense-MLA operator.  Both the query and the paged latent KV
carry the *absorbed* 512(nope)+64(rope) layout, so it replaces the
``FusedInferAttentionScore`` latent path only when all of the following hold:

1. **Absorbed MLA only.**  ``head_dim_qk == 576`` and ``head_dim_v == 512``
   (``kv_lora_rank + qk_rope_head_dim`` / ``kv_lora_rank``), ``num_kv_heads ==
   1``.  The unabsorbed prefill path (``qk_nope + qk_rope`` head dims) has no
   equivalent.
2. **Contiguous latent KV.**  ``k_cache`` must be PA_BBND
   ``(num_blocks, block_size, 1, 576)``.  sglang's NPU MLA pool normally keeps
   the two halves in two separate buffers, which cannot be fed to the op; with
   ``ENABLE_ASCEND_FLASH_MLA`` the pool allocates one 576-wide buffer instead
   and ``get_flash_mla_kv_buffer()`` returns the matching view.  Copying the
   two halves together per step would move the whole cache (hundreds of MB per
   layer), so the layout has to be decided at allocation time.
3. **Metadata is a shape-only AICPU op.**  ``flash_mla_metadata`` computes the
   AIC/AIV task split; its output depends on batch/heads/mask/layout/max
   seqlen, *not* on the ``cache_seqlens`` values (verified bit-identical when
   the values change, see the delivery's ``meta_reuse.py``).  One tensor per
   batch bucket therefore serves every layer and every step that hits the
   bucket.  Building it per layer costs ~35us of AI_CPU time per layer, i.e.
   ~1.3ms per step for a 40-layer model -- many times the op itself.  Inside a
   captured decode graph the *first* attention layer records the single
   metadata node, the remaining layers read its output, and each replay
   recomputes it from the in-place refreshed ``seqused_kv`` buffer.

Nothing here is sgl-kernel-npu: the op comes from the CANN custom operator
package (``aclnnFlashMla`` / ``aclnnFlashMlaMetadata``) plus the
``cann_ops_transformer`` python binding.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = [
    "is_importable",
    "unsupported_reason",
    "metadata",
    "decode",
    "extend",
    "warmup",
    "KV_LAYOUT_BBND",
    "HEAD_DIM_QK",
    "HEAD_DIM_V",
]

KV_LAYOUT_BBND = "PA_BBND"

HEAD_DIM_QK = 576
HEAD_DIM_V = 512

LAYOUT_DENSE = "BSND"
LAYOUT_RAGGED = "TND"

MASK_MODE_FULL = 0
MASK_MODE_CAUSAL = 3

# block_size (== sglang page_size) constraints from the operator docs.
MIN_BLOCK_SIZE = 16
MAX_BLOCK_SIZE = 1024

_MASK_SIZE = 2048
_CAUSAL_MASK_CACHE: dict = {}


def _as_int32(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None
    return t if t.dtype == torch.int32 else t.to(torch.int32)


def _ops():
    """Import ``cann_ops_transformer`` lazily; returns (flash_mla, metadata)."""
    import cann_ops_transformer  # noqa: PLC0415  (optional dependency)

    return (
        cann_ops_transformer.ops.flash_mla,
        cann_ops_transformer.ops.flash_mla_metadata,
    )


def is_importable() -> bool:
    try:
        _ops()
        return True
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return False


def unsupported_reason(
    device_name: Optional[str],
    is_mla: bool,
    page_size: Optional[int],
    head_dim_qk: Optional[int] = None,
    head_dim_v: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
) -> Optional[str]:
    """Return ``None`` when FlashMLA is usable, otherwise a log-ready reason."""
    if not is_mla:
        return "flash_mla is a dense-MLA operator (model attention is MHA)"
    if not is_importable():
        return (
            "cann_ops_transformer with flash_mla is not importable "
            "(source /mnt/sfs_turbo/hxd/pkgs/flash_mla_delivery/env.sh)"
        )
    name = (device_name or "").lower()
    if "950" not in name:
        return f"flash_mla supports Ascend 950PR/950DT only (device is {device_name!r})"
    if (
        not isinstance(page_size, int)
        or page_size < MIN_BLOCK_SIZE
        or page_size > MAX_BLOCK_SIZE
        or page_size % 16 != 0
    ):
        return (
            f"page_size must be in [{MIN_BLOCK_SIZE}, {MAX_BLOCK_SIZE}] and a "
            f"multiple of 16 (page_size is {page_size!r})"
        )
    if head_dim_qk is not None and head_dim_qk != HEAD_DIM_QK:
        return f"flash_mla needs head_dim_qk={HEAD_DIM_QK} (got {head_dim_qk})"
    if head_dim_v is not None and head_dim_v != HEAD_DIM_V:
        return f"flash_mla needs head_dim_v={HEAD_DIM_V} (got {head_dim_v})"
    if num_kv_heads is not None and num_kv_heads != 1:
        return f"flash_mla needs num_kv_heads=1 (got {num_kv_heads})"
    return None


def causal_mask(device=None) -> torch.Tensor:
    """``int8`` 2048x2048 ``triu(ones, 1)`` mask (1 = masked), cached per device."""
    if device is None:
        device = torch.npu.current_device()
    key = str(device)
    mask = _CAUSAL_MASK_CACHE.get(key)
    if mask is None:
        mask = torch.triu(
            torch.ones((_MASK_SIZE, _MASK_SIZE), dtype=torch.int8, device=device), 1
        )
        _CAUSAL_MASK_CACHE[key] = mask
    return mask


def metadata(
    num_q_heads: int,
    *,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    head_dim_qk: int = HEAD_DIM_QK,
    head_dim_v: int = HEAD_DIM_V,
    mask_mode: int = MASK_MODE_FULL,
    layout_q: str = LAYOUT_DENSE,
) -> torch.Tensor:
    """Generate the scheduling metadata for one (batch bucket, head config).

    The result is shape-only: callers should cache it per batch bucket rather
    than per layer (see the module docstring).
    """
    _, fmm = _ops()
    return fmm(
        cache_seqlens,
        num_q_heads,
        1,
        cu_seqlens_q=_as_int32(cu_seqlens_q),
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        mask_mode=mask_mode,
        win_left=-1,
        win_right=-1,
        layout_q=layout_q,
    )


def decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    num_q_heads: int,
    scale: float,
    max_seqlen_kv: int,
    head_dim_v: int = HEAD_DIM_V,
    head_dim_qk: int = HEAD_DIM_QK,
    md: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Paged latent-KV decode.

    q: ``(bs, 1, num_q_heads, 576)`` (BSND, absorbed nope + rope already
    concatenated); ``k_cache``: PA_BBND ``(num_blocks, block_size, 1, 576)``.
    Returns ``(bs, 1, num_q_heads, 512)``.
    """
    fm, _ = _ops()
    if q.shape[-2] != num_q_heads or q.shape[-1] != head_dim_qk:
        raise ValueError(
            f"q shape {tuple(q.shape)} disagrees with num_q_heads={num_q_heads}, "
            f"head_dim_qk={head_dim_qk}"
        )
    if k_cache.shape[-2] != 1 or k_cache.shape[-1] != head_dim_qk:
        raise ValueError(
            f"k_cache shape {tuple(k_cache.shape)} is not paged latent "
            f"PA_BBND (..., 1, {head_dim_qk})"
        )
    block_table = _as_int32(block_table).contiguous()
    cache_seqlens = _as_int32(cache_seqlens).contiguous()
    if md is None:
        md = metadata(
            num_q_heads,
            batch_size=q.shape[0],
            max_seqlen_q=1,
            max_seqlen_kv=max_seqlen_kv,
            cache_seqlens=cache_seqlens,
            head_dim_qk=head_dim_qk,
            head_dim_v=head_dim_v,
            mask_mode=MASK_MODE_FULL,
            layout_q=LAYOUT_DENSE,
        )
    out, _lse = fm(
        q,
        k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        metadata=md,
        head_dim_v=head_dim_v,
        softmax_scale=scale,
        mask_mode=MASK_MODE_FULL,
        max_seqlen_q=1,
        max_seqlen_kv=max_seqlen_kv,
        layout_q=LAYOUT_DENSE,
        layout_kv=KV_LAYOUT_BBND,
        layout_out=LAYOUT_DENSE,
        return_softmax_lse=False,
    )
    return out


def extend(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    num_q_heads: int,
    scale: float,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    head_dim_v: int = HEAD_DIM_V,
    head_dim_qk: int = HEAD_DIM_QK,
    md: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Paged latent-KV causal prefill / extend (TND queries, ragged).

    ``q``: ``(num_tokens, num_q_heads, 576)``; ``cu_seqlens_q`` is the int32
    ``(bs+1,)`` cumulative query length.  Returns ``(num_tokens, heads, 512)``.
    """
    fm, _ = _ops()
    block_table = _as_int32(block_table).contiguous()
    cache_seqlens = _as_int32(cache_seqlens).contiguous()
    cu_seqlens_q = _as_int32(cu_seqlens_q).contiguous()
    if md is None:
        md = metadata(
            num_q_heads,
            batch_size=cu_seqlens_q.numel() - 1,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens_q,
            head_dim_qk=head_dim_qk,
            head_dim_v=head_dim_v,
            mask_mode=MASK_MODE_CAUSAL,
            layout_q=LAYOUT_RAGGED,
        )
    out, _lse = fm(
        q,
        k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        metadata=md,
        attn_mask=causal_mask(q.device),
        head_dim_v=head_dim_v,
        softmax_scale=scale,
        mask_mode=MASK_MODE_CAUSAL,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        layout_q=LAYOUT_RAGGED,
        layout_kv=KV_LAYOUT_BBND,
        layout_out=LAYOUT_RAGGED,
        return_softmax_lse=False,
    )
    return out


def warmup(device=None, dtype: torch.dtype = torch.bfloat16) -> None:
    """Run the op once so the JIT-compiled extension is paid for at startup."""
    device = device if device is not None else "npu"
    bs, heads, page = 2, 2, 128
    q = torch.zeros((bs, 1, heads, HEAD_DIM_QK), dtype=dtype, device=device)
    kc = torch.zeros((1, page, 1, HEAD_DIM_QK), dtype=dtype, device=device)
    bt = torch.zeros((bs, 1), dtype=torch.int32, device=device)
    cs = torch.ones((bs,), dtype=torch.int32, device=device)
    decode(
        q,
        kc,
        bt,
        cs,
        num_q_heads=heads,
        scale=HEAD_DIM_QK**-0.5,
        max_seqlen_kv=page,
    )
    torch.npu.synchronize()
