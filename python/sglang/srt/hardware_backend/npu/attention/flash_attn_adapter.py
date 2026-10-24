"""FlashAttn adapter for the Ascend NPU attention backend.

``cann_ops_transformer.ops.flash_attn`` (op name ``FlashAttn``) is the Ascend
950PR / 950DT native replacement for ``FusedInferAttentionScore`` (FIA).  It is
*not* a drop-in swap: four properties of the operator fail with misleading
errors if you get them wrong, so every call goes through this module.

1. **Paged KV layout.**  k/v must be 4-D ``(num_blocks, block_size, KV_N, D)``
   with ``layout_kv="PA_BBND"``.  Handing it ``"BSND"`` for paged KV makes
   tiling fail and surfaces only as ``Check NnopbaseRunForWorkspace(*) failed``.
   When the KV pool is physically stored PA_NZ (``ENABLE_QKV_NORM_ROPE_OPT``,
   written by ``qkv_rms_norm_rope_cache``), the very same buffer must instead be
   viewed 5-D ``(num_blocks, KV_N, D//16, block_size, 16)`` and passed with
   ``layout_kv="PA_NZ"``; feeding a PA_NZ buffer with the 4-D layout instead
   reads a permuted element order (wrong results, no error).  See
   ``paged_kv_view``.
2. **Causal mask convention.**  ``mask_mode=3`` needs an ``int8``
   ``triu(ones(2048, 2048), 1)`` mask (1 = masked).  FIA's ``bool ~tril`` mask
   silently produces garbage (cos ~ 0.07).  The mask is a fixed template, so
   sequences longer than 2048 still work.
3. **Metadata ordering.**  ``flash_attn_metadata`` must be generated
   immediately before the op with *identical* arguments.  The kernel trusts the
   metadata, so a mismatch (e.g. stale metadata after updating ``seqused_kv``)
   crashes the device stream instead of raising a Python error.
4. **Metadata granularity is per step, not per layer.**  Metadata carries no
   layer-dependent state (bit-exact across distinct KV caches) and its shape
   only depends on batch size / head counts, so one tensor per step is shared
   by every layer.  Computing it per layer costs ~38us and makes a 36-layer
   decode roughly 3x slower.  In a captured decode graph that means one
   metadata *node*: the first attention layer records it, the remaining layers
   read its output, and each replay recomputes it from the refreshed
   ``seqused_kv`` (see ``AscendAttnBackend._fa_metadata_for``).

The operator supports Ascend 950PR / 950DT only.  It has no MLA (rope-split
q/k) and no quantized-KV variant, so those paths must stay on FIA.

Why the adapter also matters for graphs: FIA keeps the per-step sequence
lengths on the host (a Python list) and sglang patches the captured graph via
``NPUGraph.update(cpu_update_input={"actual_seq_lengths_kv": ...})``.  FlashAttn
takes them as *device* tensors (``seqused_kv`` plus ``metadata``), so no update
channel is needed -- but the two must always be refreshed together (point 3).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

__all__ = [
    "is_importable",
    "unsupported_reason",
    "causal_mask",
    "paged_kv_view",
    "metadata",
    "decode",
    "extend",
    "warmup",
    "KV_LAYOUT_BBND",
    "KV_LAYOUT_NZ",
]

# Paged KV physical layouts:
#   PA_BBND (default): (num_blocks, block_size, KV_N, D)
#   PA_NZ            : (num_blocks, KV_N, D//16, block_size, 16)
KV_LAYOUT_BBND = "PA_BBND"
KV_LAYOUT_NZ = "PA_NZ"
_KV_LAYOUT = KV_LAYOUT_BBND
_LAYOUT_DENSE = "BSND"
_LAYOUT_RAGGED = "TND"

_MASK_MODE_FULL = 0
_MASK_MODE_CAUSAL = 3

# The causal mask is a fixed 2048x2048 template; it does not cap seq length.
_MASK_SIZE = 2048

# ``block_size`` (== sglang page_size) constraints from the operator docs.
MIN_BLOCK_SIZE = 16
MAX_BLOCK_SIZE = 1024

_CAUSAL_MASK_CACHE: dict = {}


def _as_int32(t: torch.Tensor) -> torch.Tensor:
    """FlashAttn requires int32 block tables / sequence lengths."""
    return t if t.dtype == torch.int32 else t.to(torch.int32)


def _ops() -> Tuple[object, object]:
    """Import ``cann_ops_transformer`` lazily; returns (flash_attn, metadata)."""
    import cann_ops_transformer  # noqa: PLC0415  (optional dependency)

    return (
        cann_ops_transformer.ops.flash_attn,
        cann_ops_transformer.ops.flash_attn_metadata,
    )


def is_importable() -> bool:
    try:
        _ops()
        return True
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return False


def unsupported_reason(
    device_name: Optional[str], is_mla: bool, page_size: Optional[int]
) -> Optional[str]:
    """Return ``None`` when FlashAttn is usable, otherwise a log-ready reason."""
    if is_mla:
        return "MLA attention has no FlashAttn equivalent (no rope-split q/k input)"
    if not is_importable():
        return "cann_ops_transformer is not importable in this environment"
    name = (device_name or "").lower()
    if "950" not in name:
        return f"FlashAttn supports Ascend 950PR/950DT only (device is {device_name!r})"
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


def paged_kv_view(
    kv_cache: torch.Tensor,
    page_size: int,
    num_kv_heads: int,
    head_dim: int,
    layout_kv: str = KV_LAYOUT_BBND,
) -> torch.Tensor:
    """Normalise a paged KV pool buffer to the operator's layout.

    ``PA_BBND`` -> ``(blocks, page, KV_N, D)``;
    ``PA_NZ``   -> ``(blocks, KV_N, D // 16, page, 16)``.

    The operator rejects non-contiguous k/v, so a plain ``reshape`` is used and
    a layout mismatch raises instead of silently copying the whole cache.  Both
    layouts describe the same flat page buffer, so any full-page-aligned view of
    it (1-D flat, ``(blocks, page, KV_N * D)``, or an already-correct view) is
    accepted as input.
    """
    if not kv_cache.is_contiguous():
        raise ValueError(
            "FlashAttn requires contiguous k/v; got a non-contiguous tensor of "
            f"shape {tuple(kv_cache.shape)}"
        )
    page_elems = page_size * num_kv_heads * head_dim
    if kv_cache.numel() % page_elems:
        raise ValueError(
            "KV cache size is not a whole number of pages: "
            f"numel={kv_cache.numel()}, page_elems={page_elems} "
            f"(page_size={page_size}, kv_heads={num_kv_heads}, head_dim={head_dim})"
        )
    # The pool buffer is a linear page buffer; its tensor shape differs between
    # call sites (e.g. plain [tokens, H] in extend, [blocks, page, H] in the
    # graph path), so normalise by flattening to pages first -- the same thing
    # FIA's ``view(-1, page_size, kv_heads * head_dim)`` does.
    if layout_kv == KV_LAYOUT_NZ:
        return kv_cache.reshape(-1, num_kv_heads, head_dim // 16, page_size, 16)
    if layout_kv != KV_LAYOUT_BBND:
        raise NotImplementedError(f"FlashAttn adapter: layout_kv={layout_kv!r}")
    if (
        kv_cache.dim() == 4
        and kv_cache.shape[1] == page_size
        and kv_cache.shape[2] == num_kv_heads
        and kv_cache.shape[3] == head_dim
    ):
        return kv_cache
    return kv_cache.reshape(-1, page_size, num_kv_heads, head_dim)


def metadata(
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    *,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    seqused_kv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    mask_mode: int = _MASK_MODE_FULL,
    layout_q: str = _LAYOUT_DENSE,
    layout_kv: str = _KV_LAYOUT,
) -> torch.Tensor:
    """Generate the scheduling metadata for one step.

    Call this once per (step, batch-size bucket) and share the result across all
    layers; see point 4 in the module docstring.
    """
    _, fam = _ops()
    if seqused_kv is not None:
        seqused_kv = _as_int32(seqused_kv)
    if cu_seqlens_q is not None:
        cu_seqlens_q = _as_int32(cu_seqlens_q)
    return fam(
        num_q_heads,
        num_kv_heads,
        head_dim,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        mask_mode=mask_mode,
        win_left=-1,
        win_right=-1,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_q,
    )


def _run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    scale: float,
    mask_mode: int,
    seqused_kv: torch.Tensor,
    layout_q: str,
    block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    md: Optional[torch.Tensor] = None,
    layout_kv: str = _KV_LAYOUT,
) -> torch.Tensor:
    fa, _ = _ops()
    if mask_mode not in (_MASK_MODE_FULL, _MASK_MODE_CAUSAL):
        raise NotImplementedError(f"FlashAttn adapter: mask_mode={mask_mode}")
    if q.shape[-2] != num_q_heads or q.shape[-1] != head_dim:
        raise ValueError(
            f"q shape {tuple(q.shape)} disagrees with num_q_heads={num_q_heads}, "
            f"head_dim={head_dim}"
        )
    if layout_kv == KV_LAYOUT_NZ:
        # (blocks, KV_N, D // 16, page, 16)
        if (
            k.shape[1] != num_kv_heads
            or k.shape[2] != head_dim // 16
            or k.shape[4] != 16
        ):
            raise ValueError(
                f"k shape {tuple(k.shape)} disagrees with num_kv_heads={num_kv_heads}, "
                f"head_dim={head_dim} for layout_kv={layout_kv}"
            )
    elif k.shape[-2] != num_kv_heads or k.shape[-1] != head_dim:
        raise ValueError(
            f"k shape {tuple(k.shape)} disagrees with num_kv_heads={num_kv_heads}, "
            f"head_dim={head_dim}"
        )

    # The operator wants int32 metadata/sequence buffers.
    seqused_kv = _as_int32(seqused_kv)
    if block_table is not None:
        block_table = _as_int32(block_table)
    if cu_seqlens_q is not None:
        cu_seqlens_q = _as_int32(cu_seqlens_q)

    # Metadata must mirror the op arguments exactly (point 3).
    if md is None:
        md = metadata(
            num_q_heads,
            num_kv_heads,
            head_dim,
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            seqused_kv=seqused_kv,
            cu_seqlens_q=cu_seqlens_q,
            mask_mode=mask_mode,
            layout_q=layout_q,
            layout_kv=layout_kv,
        )

    out, _lse = fa(
        q,
        k,
        v,
        block_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_kv=seqused_kv,
        metadata=md,
        softmax_scale=scale,
        mask_mode=mask_mode,
        attn_mask=causal_mask(q.device) if mask_mode == _MASK_MODE_CAUSAL else None,
        win_left=-1,
        win_right=-1,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_q,
    )
    return out


def decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_kv: torch.Tensor,
    *,
    page_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    max_seqlen_kv: int,
    md: Optional[torch.Tensor] = None,
    layout_kv: str = _KV_LAYOUT,
) -> torch.Tensor:
    """Paged-KV decode.

    q: ``(bs, 1, num_q_heads, head_dim)`` (BSND); k/v pools are reshaped to
    ``layout_kv`` internally.  Returns ``(bs, 1, num_q_heads, head_dim)``.
    """
    k = paged_kv_view(k_cache, page_size, num_kv_heads, head_dim, layout_kv)
    v = paged_kv_view(v_cache, page_size, num_kv_heads, head_dim, layout_kv)
    return _run(
        q,
        k,
        v,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        batch_size=q.shape[0],
        max_seqlen_q=1,
        max_seqlen_kv=max_seqlen_kv,
        scale=scale,
        mask_mode=_MASK_MODE_FULL,
        seqused_kv=seqused_kv,
        layout_q=_LAYOUT_DENSE,
        block_table=block_table,
        md=md,
        layout_kv=layout_kv,
    )


def extend(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    *,
    page_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    scale: float,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    md: Optional[torch.Tensor] = None,
    layout_kv: str = _KV_LAYOUT,
) -> torch.Tensor:
    """Paged-KV causal prefill / extend.

    q: ``(num_tokens, num_q_heads, head_dim)`` (TND, ragged via ``cu_seqlens_q``).
    """
    k = paged_kv_view(k_cache, page_size, num_kv_heads, head_dim, layout_kv)
    v = paged_kv_view(v_cache, page_size, num_kv_heads, head_dim, layout_kv)
    return _run(
        q,
        k,
        v,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        batch_size=cu_seqlens_q.numel() - 1,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        scale=scale,
        mask_mode=_MASK_MODE_CAUSAL,
        seqused_kv=seqused_kv,
        layout_q=_LAYOUT_RAGGED,
        block_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        md=md,
        layout_kv=layout_kv,
    )


def warmup(device=None, dtype: torch.dtype = torch.bfloat16) -> None:
    """Run the op once so the JIT-compiled extension is paid for at startup."""
    device = device if device is not None else "npu"
    n, kv, d, blk = 4, 2, 128, 128
    q = torch.zeros((1, 1, n, d), dtype=dtype, device=device)
    kc = torch.zeros((1, blk, kv * d), dtype=dtype, device=device)
    vc = torch.zeros((1, blk, kv * d), dtype=dtype, device=device)
    bt = torch.zeros((1, 1), dtype=torch.int32, device=device)
    seqkv = torch.ones((1,), dtype=torch.int32, device=device)
    decode(
        q,
        kc,
        vc,
        bt,
        seqkv,
        page_size=blk,
        num_q_heads=n,
        num_kv_heads=kv,
        head_dim=d,
        scale=d**-0.5,
        max_seqlen_kv=blk,
    )
    torch.npu.synchronize()
